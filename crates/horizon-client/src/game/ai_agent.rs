//! The built-in AI agent: `/agent <instruction>` in chat hands the player
//! to an OpenAI-compatible chat model that acts through the very same
//! command machinery as the control socket (`control_command`), so the
//! tool surface is identical to `horizon-ctl`'s. A worker thread owns the
//! HTTP conversation; every tool call is executed on the game thread.
//!
//! Configuration lives in `horizon-agent.json` next to the control
//! endpoint file (the launcher's home on Android) and is re-read for
//! every job, so the launcher can rewrite it without restarting the game:
//! `{ "endpoint": "http://host:port/v1", "api_key": "...", "model": "...",
//!    "enabled": true }`. The API key is stored as plain text inside the
//! app's private data, like the game's other account files.

use super::Game;
use super::event_loop::EventLoopCtl;
use horizon_client::text::Text;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::Duration;

/// Tool-call rounds per instruction before the agent must wrap up.
const MAX_TOOL_ROUNDS: usize = 12;
/// How long the worker waits for one game command (long `goto`s included).
const EXEC_TIMEOUT: Duration = Duration::from_secs(180);
/// Per-request HTTP timeout.
const HTTP_TIMEOUT: Duration = Duration::from_secs(90);

const SYSTEM_PROMPT: &str = "\
You are an AI agent playing Minecraft (Java Edition rules) inside the \
Horizon game client, acting on behalf of the player. The player gives you \
instructions through /agent chat commands. You act through tools; each \
tool runs one game command and returns a JSON result. \
Rules: \
1. If you are unsure about the surroundings, call state, looking_at or \
entities first. \
2. After moving or acting, verify with state or looking_at when it \
matters. \
3. Coordinates are absolute Minecraft coordinates. The state snapshot \
below tells the player's position, health, game mode, yaw and pitch. \
4. When the task is done, stop calling tools and answer with a short \
summary of what you did, in the same language the player used, plain \
text, no markdown. \
5. If a tool returns an error, read it and adjust; never repeat the \
exact same failing call. \
6. You are the player's character: moving, breaking and placing blocks \
changes their world, so stay on the task they gave you. \
7. When an action succeeded, do not repeat it; verify with state or \
looking_at instead. \
8. Copy coordinates, names and ids exactly from tool results.";

pub(super) struct AiConfig {
    endpoint: String,
    api_key: String,
    model: String,
}

/// One message from the agent worker thread to the game thread.
pub(super) enum AiToGame {
    /// Run one control line; the game answers through `reply` (deferred
    /// commands are forwarded once they complete).
    Exec { line: String, reply: Sender<String> },
    /// The job finished with this final chat text.
    Finished(String),
    /// The job failed or was cancelled, with a player-facing reason.
    Failed(String),
}

/// One `/agent` job for the worker thread.
pub(super) struct AiJob {
    instruction: String,
    state_json: String,
    history: Vec<Value>,
    config: AiConfig,
    to_game: Sender<AiToGame>,
    cancel: Arc<AtomicBool>,
}

/// The argument of a `/agent` chat line, when the line is one: `/agent`
/// alone yields an empty argument.
pub(super) fn agent_command_arg(message: &str) -> Option<&str> {
    if message == "/agent" {
        return Some("");
    }
    message.strip_prefix("/agent ")
}

impl Game<'_> {
    /// The `/agent` chat command: validate and queue the instruction; the
    /// job itself starts in `poll_ai_agent`, which owns an event loop.
    pub(super) fn ai_agent_command(&mut self, arg: &str) {
        let arg = arg.trim();
        if arg == "stop" {
            if self.ai_busy {
                self.ai_cancel.store(true, Ordering::Relaxed);
                self.ai_say("[Agent] Menghentikan…");
            } else {
                self.ai_say("[Agent] Tidak ada perintah yang sedang berjalan.");
            }
            return;
        }
        if arg.is_empty() {
            self.ai_say(
                "[Agent] Pakai: /agent <perintah> — contoh: /agent ambil kayu dari pohon terdekat",
            );
            return;
        }
        if self.settings.extras.no_ai {
            self.ai_say(horizon_core::no_ai::REFUSAL);
            return;
        }
        if self.ai_busy {
            self.ai_say(
                "[Agent] Masih mengerjakan perintah sebelumnya. Tunggu, atau /agent stop untuk membatalkan.",
            );
            return;
        }
        self.ai_request = Some(arg.to_owned());
    }

    /// Advance the built-in agent: start queued jobs, serve the worker's
    /// tool calls on this thread, and surface its chat lines. Called every
    /// tick next to `poll_control`.
    pub(super) fn poll_ai_agent(&mut self, event_loop: &dyn EventLoopCtl) {
        if !self.ai_busy
            && let Some(instruction) = self.ai_request.take()
        {
            self.ai_start_job(instruction, event_loop);
        }
        // Deferred control replies that have since completed.
        let mut i = 0;
        while i < self.ai_pending_exec.len() {
            match self.ai_pending_exec[i].0.try_recv() {
                Ok(text) => {
                    let (_, reply) = self.ai_pending_exec.remove(i);
                    let _ = reply.send(text);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => i += 1,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let (_, reply) = self.ai_pending_exec.remove(i);
                    let _ = reply.send(json!({ "error": "command channel closed" }).to_string());
                }
            }
        }
        let Some(inbox) = self.ai_inbox.take() else {
            return;
        };
        let mut disconnected = false;
        loop {
            match inbox.try_recv() {
                Ok(AiToGame::Exec { line, reply }) => {
                    let (tx, rx) = channel::<String>();
                    match self.control_command(&line, &tx, event_loop) {
                        Ok(Some(value)) => {
                            let _ = reply.send(value.to_string());
                        }
                        Ok(None) => self.ai_pending_exec.push((rx, reply)),
                        Err(e) => {
                            let _ = reply.send(json!({ "error": e.to_string() }).to_string());
                        }
                    }
                }
                Ok(AiToGame::Finished(text)) => {
                    if let Some(instruction) = self.ai_current.take() {
                        self.ai_history
                            .push(json!({ "role": "user", "content": instruction }));
                    }
                    self.ai_history
                        .push(json!({ "role": "assistant", "content": text }));
                    // Keep the last five exchanges.
                    while self.ai_history.len() > 10 {
                        self.ai_history.remove(0);
                    }
                    self.ai_busy = false;
                    self.ai_say(&format!("[Agent] {text}"));
                }
                Ok(AiToGame::Failed(reason)) => {
                    self.ai_current = None;
                    self.ai_busy = false;
                    self.ai_say(&format!("[Agent] {reason}"));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        self.ai_inbox = Some(inbox);
        if disconnected && self.ai_busy {
            self.ai_busy = false;
            self.ai_current = None;
            self.ai_say("[Agent] Proses agent berhenti tak terduga.");
        }
    }

    fn ai_start_job(&mut self, instruction: String, event_loop: &dyn EventLoopCtl) {
        let config = match self.ai_load_config() {
            Ok(config) => config,
            Err(message) => {
                self.ai_say(&format!("[Agent] {message}"));
                return;
            }
        };
        // A fresh state snapshot rides along with the system prompt.
        let (tx, _rx) = channel::<String>();
        let state_json = match self.control_command("state", &tx, event_loop) {
            Ok(Some(value)) => value.to_string(),
            _ => "{}".to_owned(),
        };
        self.ai_ensure_worker();
        let (Some(jobs), Some(to_game)) = (self.ai_job_tx.as_ref(), self.ai_to_game_tx.as_ref())
        else {
            self.ai_say("[Agent] Worker tidak tersedia.");
            return;
        };
        self.ai_cancel.store(false, Ordering::Relaxed);
        let job = AiJob {
            instruction: instruction.clone(),
            state_json,
            history: self.ai_history.clone(),
            config,
            to_game: to_game.clone(),
            cancel: self.ai_cancel.clone(),
        };
        if jobs.send(job).is_ok() {
            self.ai_current = Some(instruction);
            self.ai_busy = true;
            self.ai_say("[Agent] Sending Request...");
        } else {
            self.ai_say("[Agent] Worker tidak tersedia.");
        }
    }

    fn ai_ensure_worker(&mut self) {
        if self.ai_job_tx.is_some() {
            return;
        }
        let (job_tx, job_rx) = channel::<AiJob>();
        let (to_game_tx, to_game_rx) = channel::<AiToGame>();
        std::thread::spawn(move || {
            while let Ok(job) = job_rx.recv() {
                run_ai_job(job);
            }
        });
        self.ai_job_tx = Some(job_tx);
        self.ai_to_game_tx = Some(to_game_tx);
        self.ai_inbox = Some(to_game_rx);
    }

    fn ai_say(&mut self, text: &str) {
        self.system_message(Text::literal(text));
    }

    /// `horizon-agent.json` sits next to the control endpoint file (the
    /// launcher's home); without a control path it is the game folder.
    fn ai_config_path(&self) -> PathBuf {
        if let Some(control) = &self.options.control
            && let Some(home) = control.parent()
        {
            return home.join("horizon-agent.json");
        }
        PathBuf::from("horizon-agent.json")
    }

    fn ai_load_config(&self) -> Result<AiConfig, String> {
        let path = self.ai_config_path();
        let text = std::fs::read_to_string(&path).map_err(|_| {
            format!(
                "Belum diatur. Buat file {} berisi endpoint, api_key, dan model, atau atur dari launcher.",
                path.display()
            )
        })?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| format!("File pengaturan agent rusak: {e}"))?;
        if value.get("enabled").and_then(Value::as_bool) == Some(false) {
            return Err("Agent sedang dimatikan di pengaturan.".to_owned());
        }
        let field = |name: &str| {
            value
                .get(name)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        let (Some(endpoint), Some(api_key), Some(model)) =
            (field("endpoint"), field("api_key"), field("model"))
        else {
            return Err(
                "Pengaturan agent belum lengkap: endpoint, api_key, dan model wajib diisi."
                    .to_owned(),
            );
        };
        Ok(AiConfig {
            endpoint,
            api_key,
            model,
        })
    }
}

/// The worker thread's job loop: talk to the model, translate its tool
/// calls into control lines, run them on the game thread, feed the
/// results back, and finish with the model's chat reply.
fn run_ai_job(job: AiJob) {
    let to_game = job.to_game.clone();
    let outcome = run_ai_conversation(&job);
    let message = match outcome {
        Ok(text) => AiToGame::Finished(text),
        Err(reason) => AiToGame::Failed(reason),
    };
    let _ = to_game.send(message);
}

fn run_ai_conversation(job: &AiJob) -> Result<String, String> {
    let url = format!(
        "{}/chat/completions",
        job.config.endpoint.trim_end_matches('/')
    );
    let http = ureq::AgentBuilder::new().timeout(HTTP_TIMEOUT).build();
    let mut messages: Vec<Value> = vec![json!({ "role": "system", "content": format!(
            "{SYSTEM_PROMPT}\n\nCurrent game state (JSON):\n{}", job.state_json
        ) })];
    messages.extend(job.history.iter().cloned());
    messages.push(json!({ "role": "user", "content": job.instruction }));
    let tools = ai_tools();

    for _round in 0..MAX_TOOL_ROUNDS {
        if job.cancel.load(Ordering::Relaxed) {
            return Err("Dihentikan oleh pemain.".to_owned());
        }
        let body = json!({
            "model": job.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": false,
        });
        let message = fetch_message(&http, &url, &job.config.api_key, &body)?;
        messages.push(message.clone());

        let calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("(selesai, tanpa keterangan)");
            return Ok(content.to_owned());
        }
        for call in &calls {
            if job.cancel.load(Ordering::Relaxed) {
                return Err("Dihentikan oleh pemain.".to_owned());
            }
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args_text = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let args: Value = serde_json::from_str(args_text).unwrap_or_else(|_| json!({}));
            let result = match tool_line(name, &args) {
                Ok(line) => {
                    let (tx, rx) = channel::<String>();
                    if job
                        .to_game
                        .send(AiToGame::Exec { line, reply: tx })
                        .is_err()
                    {
                        return Err("Game sudah tertutup.".to_owned());
                    }
                    rx.recv_timeout(EXEC_TIMEOUT)
                        .unwrap_or_else(|_| json!({ "error": "command timed out" }).to_string())
                }
                Err(e) => json!({ "error": e }).to_string(),
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": result,
            }));
        }
    }
    // Out of tool rounds: one last call without tools, so the player
    // gets a summary of what happened instead of a dead end.
    messages.push(json!({
        "role": "user",
        "content": "(Step limit reached. Reply with a brief summary of what you did and the result. Do not call any tools.)",
    }));
    let body = json!({
        "model": job.config.model,
        "messages": messages,
        "stream": false,
    });
    match fetch_message(&http, &url, &job.config.api_key, &body) {
        Ok(message) => {
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("(selesai, tanpa keterangan)");
            Ok(content.to_owned())
        }
        Err(_) => Ok(
            "Sudah mencapai batas langkah untuk perintah ini, jadi berhenti di sini.".to_owned(),
        ),
    }
}

/// One chat-completions request: POST the body and return the first
/// choice's message object.
fn fetch_message(
    http: &ureq::Agent,
    url: &str,
    api_key: &str,
    body: &Value,
) -> Result<Value, String> {
    let response = http
        .post(url)
        .set("Authorization", &format!("Bearer {api_key}"))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| match e {
            ureq::Error::Status(code, response) => {
                let detail = response.into_string().unwrap_or_default();
                format!("Endpoint menolak permintaan (HTTP {code}): {detail}")
            }
            ureq::Error::Transport(e) => format!("Tidak bisa menghubungi endpoint: {e}"),
        })?;
    let text = response
        .into_string()
        .map_err(|e| format!("Balasan endpoint tidak terbaca: {e}"))?;
    let value = parse_completion(&text)
        .ok_or_else(|| "Balasan endpoint bukan JSON yang valid.".to_owned())?;
    value
        .pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| "Endpoint tidak mengembalikan pesan.".to_owned())
}

/// Extract a chat completion from an endpoint's response body. Plain
/// JSON is the norm, but some proxies answer with a full JSON object
/// followed by stream framing (`data: [DONE]`), and some force real
/// server-sent events; accept all three.
fn parse_completion(text: &str) -> Option<Value> {
    // One JSON value, ignoring whatever trails it.
    if let Some(Ok(value)) = serde_json::Deserializer::from_str(text)
        .into_iter::<Value>()
        .next()
        && value.get("choices").is_some()
    {
        return Some(value);
    }
    // Real SSE: merge the delta chunks into one message.
    let mut content = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut saw_chunk = false;
    for line in text.lines() {
        let Some(payload) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        let Some(delta) = chunk.pointer("/choices/0/delta") else {
            continue;
        };
        saw_chunk = true;
        if let Some(part) = delta.get("content").and_then(Value::as_str) {
            content += part;
        }
        if let Some(list) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in list {
                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                while calls.len() <= index {
                    calls.push(json!({
                        "id": "",
                        "type": "function",
                        "function": { "name": "", "arguments": "" },
                    }));
                }
                let slot = &mut calls[index];
                if let Some(id) = call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    slot["id"] = json!(id);
                }
                if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                    let current = slot["function"]["name"].as_str().unwrap_or("").to_owned();
                    slot["function"]["name"] = json!(format!("{current}{name}"));
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str) {
                    let current = slot["function"]["arguments"]
                        .as_str()
                        .unwrap_or("")
                        .to_owned();
                    slot["function"]["arguments"] = json!(format!("{current}{args}"));
                }
            }
        }
    }
    if !saw_chunk {
        return None;
    }
    let mut message = json!({ "role": "assistant", "content": content });
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    Some(json!({ "choices": [{ "message": message }] }))
}

/// Translate one model tool call into a control command line.
fn tool_line(name: &str, args: &Value) -> Result<String, String> {
    let num = |key: &str| args.get(key).and_then(Value::as_f64);
    let req_num = |key: &str| num(key).ok_or_else(|| format!("missing number argument {key}"));
    let int = |key: &str| req_num(key).map(|v| v.round() as i64);
    let text = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_owned);
    let flag = |key: &str, suffix: &str| {
        if args.get(key).and_then(Value::as_bool) == Some(true) {
            format!(" {suffix}")
        } else {
            String::new()
        }
    };
    let line = match name {
        "state" => "state".to_owned(),
        "looking_at" => "looking-at".to_owned(),
        "inventory" => "inventory".to_owned(),
        "screenshot" => "screenshot agent-view.png".to_owned(),
        "entities" => {
            let mut line = "entities".to_owned();
            if let Some(radius) = num("radius") {
                line += &format!(" {radius}");
            }
            if let Some(kind) = text("type") {
                line += &format!(" {kind}");
            }
            line
        }
        "block" => format!("block {} {} {}", int("x")?, int("y")?, int("z")?),
        "find_block" => {
            let kind = text("kind").ok_or("missing string argument kind")?;
            let mut line = format!("find-block {kind}");
            if let Some(radius) = num("radius") {
                line += &format!(" {radius}");
            }
            line
        }
        "look" => format!("look {} {}", req_num("yaw")?, req_num("pitch")?),
        "turn" => {
            let mut line = format!("turn {} {}", req_num("dyaw")?, req_num("dpitch")?);
            if let Some(over) = num("over") {
                line += &format!(" --over {over}");
            }
            line
        }
        "move" => {
            let direction = text("direction").ok_or("missing string argument direction")?;
            let mut line = format!("move {} {}", direction, req_num("seconds")?);
            line += &flag("sprint", "--sprint");
            line += &flag("sneak", "--sneak");
            line += &flag("jump", "--jump");
            line
        }
        "goto" => {
            let mut line = format!("goto {} {} {}", req_num("x")?, req_num("y")?, req_num("z")?);
            line += &flag("sprint", "--sprint");
            if let Some(timeout) = num("timeout") {
                line += &format!(" --timeout {timeout}");
            }
            line
        }
        "jump" => "jump".to_owned(),
        "wait" => format!("wait {}", int("ticks")?),
        "break_block" => format!("break {} {} {}", int("x")?, int("y")?, int("z")?),
        "place" => {
            let mut line = format!("place {} {} {}", int("x")?, int("y")?, int("z")?);
            if let Some(item) = text("item") {
                line += &format!(" --item {item}");
            }
            line
        }
        "interact" => {
            let mut line = format!("interact {} {} {}", int("x")?, int("y")?, int("z")?);
            if let Some(face) = text("face") {
                line += &format!(" {face}");
            }
            line
        }
        "attack" => match num("id") {
            Some(id) => format!("attack {}", id as i64),
            None => "attack".to_owned(),
        },
        "use_item" => {
            let mut line = "use-item".to_owned();
            if let Some(hold) = num("hold") {
                line += &format!(" --hold {hold}");
            }
            line
        }
        "chat" => {
            let text = text("text").ok_or("missing string argument text")?;
            format!("chat {text}")
        }
        "command" => {
            let text = text("text").ok_or("missing string argument text")?;
            format!("command {text}")
        }
        other => return Err(format!("unknown tool {other}")),
    };
    Ok(line)
}

/// The tool schemas offered to the model. Descriptions double as its
/// usage instructions, alongside the system prompt.
fn ai_tools() -> Value {
    let function = |name: &str, description: &str, properties: Value, required: &[&str]| {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    };
    let number = json!({ "type": "number" });
    let integer = json!({ "type": "integer" });
    let string = json!({ "type": "string" });
    let boolean = json!({ "type": "boolean" });
    let xyz = |props: &mut serde_json::Map<String, Value>| {
        props.insert("x".to_owned(), integer.clone());
        props.insert("y".to_owned(), integer.clone());
        props.insert("z".to_owned(), integer.clone());
    };
    let mut block_props = serde_json::Map::new();
    xyz(&mut block_props);
    let mut goto_props = serde_json::Map::new();
    goto_props.insert("x".to_owned(), number.clone());
    goto_props.insert("y".to_owned(), number.clone());
    goto_props.insert("z".to_owned(), number.clone());
    goto_props.insert("sprint".to_owned(), boolean.clone());
    goto_props.insert("timeout".to_owned(), number.clone());
    json!([
        function(
            "state",
            "Full game state as JSON: player position, health, mode, dimension, chunks, FPS and more. Call when unsure.",
            json!({}),
            &[]
        ),
        function(
            "looking_at",
            "The block or entity under the crosshair.",
            json!({}),
            &[]
        ),
        function(
            "entities",
            "Entities near the player.",
            json!({ "radius": number, "type": string }),
            &[]
        ),
        function(
            "block",
            "The block at a position.",
            Value::Object(block_props.clone()),
            &["x", "y", "z"]
        ),
        function(
            "find_block",
            "Nearest loaded blocks of a kind, e.g. minecraft:oak_log.",
            json!({ "kind": string, "radius": number }),
            &["kind"]
        ),
        function(
            "inventory",
            "The player's inventory and held items.",
            json!({}),
            &[]
        ),
        function(
            "screenshot",
            "Save a screenshot of the current view (agent-view.png in the game folder).",
            json!({}),
            &[]
        ),
        function(
            "look",
            "Set the view direction in degrees (yaw, pitch).",
            json!({ "yaw": number, "pitch": number }),
            &["yaw", "pitch"]
        ),
        function(
            "turn",
            "Rotate the view by dyaw and dpitch degrees, optionally over N seconds.",
            json!({ "dyaw": number, "dpitch": number, "over": number }),
            &["dyaw", "dpitch"]
        ),
        function(
            "move",
            "Walk in a direction for N seconds.",
            json!({ "direction": { "type": "string", "enum": ["forward", "back", "left", "right"] }, "seconds": number, "sprint": boolean, "sneak": boolean, "jump": boolean }),
            &["direction", "seconds"]
        ),
        function(
            "goto",
            "Walk to coordinates, pathfinding around obstacles. Returns when arrived or timed out.",
            Value::Object(goto_props),
            &["x", "y", "z"]
        ),
        function("jump", "Jump once.", json!({}), &[]),
        function(
            "wait",
            "Wait N game ticks (20 ticks = 1 second).",
            json!({ "ticks": integer }),
            &["ticks"]
        ),
        function(
            "break_block",
            "Mine the block at a position until it breaks; returns what it dropped.",
            Value::Object(block_props.clone()),
            &["x", "y", "z"]
        ),
        function(
            "place",
            "Place the held (or named) block against the position.",
            json!({ "x": integer, "y": integer, "z": integer, "item": string }),
            &["x", "y", "z"]
        ),
        function(
            "interact",
            "Right-click the block or entity at a position, optionally on a face (up down north south west east).",
            json!({ "x": integer, "y": integer, "z": integer, "face": string }),
            &["x", "y", "z"]
        ),
        function(
            "attack",
            "Attack the entity under the crosshair, or the entity with the given id.",
            json!({ "id": integer }),
            &[]
        ),
        function(
            "use_item",
            "Use the held item (eat, drink, draw a bow), optionally holding for N seconds.",
            json!({ "hold": number }),
            &[]
        ),
        function(
            "chat",
            "Send a chat line as the player (a leading / makes it a command).",
            json!({ "text": string }),
            &["text"]
        ),
        function(
            "command",
            "Run a game command (the leading / is optional), e.g. time set day.",
            json!({ "text": string }),
            &["text"]
        ),
    ])
}
