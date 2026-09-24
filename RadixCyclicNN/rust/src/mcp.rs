//! The tools and the network itself over the Model Context Protocol
//! (`radixnet/mcp.py`, `go/radixnet/mcp.go`).
//!
//! Two things are offered to any MCP client (Claude Desktop, an editor,
//! another agent), and the second is the interesting one:
//!
//! * the **tools** of [`crate::tools`] - `web_search`, `web_fetch`,
//!   `web_links`, `calculator`, and `python` / `read_file` when they are
//!   switched on: the registry the network calls by writing text, offered to
//!   whoever else wants it;
//! * the **model** - `radixnet_predict`, `radixnet_generate`,
//!   `radixnet_score`, `radixnet_stats`, `radixnet_solve` (one task through
//!   the whole agent loop: criteria, tool calls, judging) and
//!   `radixnet_judge` (the negative network's verdict on a text).  A client
//!   can ask *this* network what it thinks, not just borrow its browser.
//!
//! # The stream is the contract
//!
//! MCP's stdio transport is JSON-RPC 2.0, one JSON object per line on stdin
//! and one per line on stdout.  A client does not know which implementation
//! it started, so every answer is Python's to the byte: the protocol
//! revision, the tool names and schemas, the texts the tools answer, and the
//! errors - down to `invalid JSON: Expecting ',' delimiter: line 1 column 9
//! (char 8)` for a line that is not JSON, which is why the line is read by
//! [`pyjson`], CPython's scanner retold, rather than by the model file's
//! reader.  The answers are written the way Python's `json.dumps(...,
//! ensure_ascii=False)` writes them, separators and all.
//!
//! Anything on stdout that is not a response corrupts the stream, so nothing
//! here prints: the notes a person reads go through the logger to stderr
//! (raised to `info` unless the log was configured, so they are seen).
//!
//! # A tool that calls tools
//!
//! `radixnet_solve` runs the agent loop, and the loop calls tools - the
//! toolbox the server offers, `radixnet_solve` included, as Python's does.
//! So the solve tool reaches its toolbox through a weak handle filled in once
//! the toolbox is complete ([`ToolboxRef`]), and the model sits behind a mutex
//! taken for one step at a time ([`Nets::Locked`]): a network that writes
//! `<tool>radixnet_predict ...</tool>` while it solves gets its answer instead
//! of waiting on itself.

pub mod pyjson;

use std::io::{BufRead, Write};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use crate::agent::{AgentConfig, AgentLlm, AgentTrainer, Task};
use crate::cli::Ctx;
use crate::codegen::{Hooks, LoopError, Nets};
use crate::json::Json;
use crate::llm::fields::truthy;
use crate::llm::{format_g, LlmError};
use crate::model::{GenerateOptions, Model, PredictOptions};
use crate::negative::{python_repr, JudgeOptions};
use crate::ollama::OllamaClient;
use crate::tools::{dumps, Param, Tool, ToolBox};

const LOG: &str = "mcp";

/// The MCP revision this server speaks (a client that asks for another has
/// its own echoed).
pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "radixnet";
/// What the network's own tools are called: `radixnet_predict`, ...
pub const PREFIX: &str = "radixnet_";

/// JSON-RPC's error codes (the spec's, and MCP's use of them).
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// What `initialize` tells a client this server is for.
const INSTRUCTIONS: &str = "The tools of a RadixCyclicNN instance: browsing and a calculator, plus the network's own \
                            predictions, generations, scores and judgements.";

/// One tool in MCP's shape: `inputSchema` rather than `function.parameters`.
fn schema(tool: &Tool) -> Json {
    Json::obj([
        ("name", Json::str(tool.name.clone())),
        ("description", Json::str(tool.description.clone())),
        ("inputSchema", tool.schema().at("function").at("parameters").clone()),
    ])
}

/// A JSON-RPC error response.
pub fn error(ident: Json, code: i64, message: &str) -> Json {
    Json::obj([
        ("jsonrpc", Json::str("2.0")),
        ("id", ident),
        (
            "error",
            Json::obj([("code", Json::Int(code)), ("message", Json::str(message))]),
        ),
    ])
}

/// What a request's handler refuses with: a bad argument (`INVALID_PARAMS`,
/// Python's `ToolError`).  Anything worse is a panic, which
/// [`McpServer::answer_line`] turns into `INTERNAL_ERROR`.
enum Refusal {
    Params(String),
}

// -- the model as tools -------------------------------------------------------------------------

/// Where `radixnet_solve` finds the toolbox it is registered in: set once the
/// toolbox is complete, and weak, so the toolbox does not keep itself alive.
#[derive(Clone, Default)]
pub struct ToolboxRef(Arc<OnceLock<Weak<ToolBox>>>);

impl ToolboxRef {
    /// Points the handle at the finished toolbox.
    pub fn set(&self, toolbox: &Arc<ToolBox>) {
        let _ = self.0.set(Arc::downgrade(toolbox));
    }

    fn get(&self) -> Option<Arc<ToolBox>> {
        self.0.get().and_then(Weak::upgrade)
    }
}

/// What `radixnet_solve` needs: the LLM that writes the criteria, mediates and
/// judges, and the tools the network calls.
pub struct Solver {
    pub client: Arc<dyn AgentLlm>,
    pub toolbox: ToolboxRef,
}

fn lock(model: &Mutex<Model>) -> MutexGuard<'_, Model> {
    model.lock().unwrap_or_else(|e| e.into_inner())
}

/// A model's refusal as Python's toolbox reports an exception it caught.
fn value_error(why: String) -> String {
    format!("ValueError: {why}")
}

fn llm_error(err: LlmError) -> String {
    format!("OllamaError: {}", err.message)
}

fn loop_error(err: LoopError) -> String {
    match err {
        LoopError::Llm(err) => llm_error(err),
        LoopError::Other(why) => value_error(why),
    }
}

/// `format(x, ".Nf")`, `nan` and `inf` spelled as Python spells them.
fn fixed(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{x:.digits$}")
    }
}

/// `json.dumps(value, indent=indent)`: one item per line, non-ASCII escaped.
fn dumps_indent(value: &Json, indent: usize, level: usize, out: &mut String) {
    let pad = |out: &mut String, level: usize| {
        out.push('\n');
        out.push_str(&" ".repeat(indent * level));
    };
    match value {
        Json::Arr(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                dumps_indent(item, indent, level + 1, out);
            }
            pad(out, level);
            out.push(']');
        }
        Json::Obj(pairs) if !pairs.is_empty() => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                out.push_str(&dumps(&Json::str(key.clone()), false, true));
                out.push_str(": ");
                dumps_indent(item, indent, level + 1, out);
            }
            pad(out, level);
            out.push('}');
        }
        other => out.push_str(&dumps(other, false, true)),
    }
}

/// A parameter with a fixed set of values.
fn one_of(param: Param, choices: &[&str]) -> Param {
    Param {
        choices: Some(choices.iter().map(|c| c.to_string()).collect()),
        ..param
    }
}

fn text_arg(args: &Json, name: &str) -> String {
    args.at(name).as_str().unwrap_or("").to_string()
}

fn int_arg(args: &Json, name: &str, fallback: i64) -> i64 {
    args.at(name).as_i64().unwrap_or(fallback)
}

fn number_arg(args: &Json, name: &str, fallback: f64) -> f64 {
    args.at(name).as_f64().unwrap_or(fallback)
}

/// The network's own operations as tools (Python's `model_tools`): predict,
/// generate, score and stats always; `judge` with a negative network; `solve`
/// with an LLM and the toolbox to solve with.  Each takes the model's lock for
/// as long as it needs the model and no longer.
pub fn model_tools(
    model: Arc<Mutex<Model>>,
    negative: Option<Arc<Mutex<Model>>>,
    solver: Option<Solver>,
    prefix: &str,
) -> Vec<Tool> {
    let mut tools = Vec::new();

    let m = model.clone();
    tools.push(Tool::new(
        &format!("{prefix}predict"),
        "Continue a prefix with the RadixCyclicNN network.",
        vec![
            Param::new("prefix_text", "string", "the text to continue"),
            Param::optional("length", "integer", "characters to add", Json::Int(40)),
            one_of(
                Param::optional(
                    "mode",
                    "string",
                    "dijkstra (cheapest path), beam or sample",
                    Json::str("dijkstra"),
                ),
                &["dijkstra", "beam", "sample"],
            ),
            Param::optional("temperature", "number", "sampling temperature", Json::Num(1.0)),
        ],
        move |args| {
            let length = int_arg(args, "length", 40);
            let options = PredictOptions {
                length: length.max(0) as usize,
                mode: text_arg(args, "mode"),
                temperature: number_arg(args, "temperature", 1.0),
                max_length: Some(length.max(1) as usize),
                ..Default::default()
            };
            let found = lock(&m)
                .predict(&text_arg(args, "prefix_text"), &options)
                .map_err(value_error)?;
            Ok((found.best.full_text, Json::Null))
        },
    ));

    let m = model.clone();
    tools.push(Tool::new(
        &format!("{prefix}generate"),
        "Generate whole texts from the network.",
        vec![
            Param::optional("count", "integer", "how many", Json::Int(3)),
            Param::optional("max_length", "integer", "characters per text", Json::Int(80)),
            Param::optional("temperature", "number", "sampling temperature", Json::Num(1.0)),
        ],
        move |args| {
            let options = GenerateOptions {
                max_length: int_arg(args, "max_length", 80).max(0) as usize,
                mode: "sample".to_string(),
                temperature: number_arg(args, "temperature", 1.0),
                count: int_arg(args, "count", 3).max(0) as usize,
                ..Default::default()
            };
            let results = lock(&m).generate(&options).map_err(value_error)?;
            let lines: Vec<String> = results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("{}. {}", i + 1, r.text))
                .collect();
            let text = if lines.is_empty() {
                "(the network generated nothing)".to_string()
            } else {
                lines.join("\n")
            };
            Ok((text, Json::Null))
        },
    ));

    let m = model.clone();
    tools.push(Tool::new(
        &format!("{prefix}score"),
        "How likely the network thinks a text is (log probability per character).",
        vec![Param::new("text", "string", "the text to score")],
        move |args| {
            let s = lock(&m).score(&text_arg(args, "text"));
            Ok((
                format!(
                    "log probability {}, {} per character over {} characters ({} unknown transitions)",
                    fixed(s.log_prob, 3),
                    fixed(s.per_char, 4),
                    s.chars,
                    s.unknown_transitions
                ),
                Json::Null,
            ))
        },
    ));

    let m = model.clone();
    tools.push(Tool::new(
        &format!("{prefix}stats"),
        "The network's size, compression, training history and backend.",
        Vec::new(),
        move |_| {
            let mut out = String::new();
            dumps_indent(&crate::kinds::stats(&lock(&m)), 2, 0, &mut out);
            Ok((out, Json::Null))
        },
    ));

    if let Some(negative) = negative.clone() {
        tools.push(Tool::new(
            &format!("{prefix}judge"),
            "Ask the negative network whether a text looks like something that has gone wrong before, and why.",
            vec![Param::new("text", "string", "the text to judge")],
            move |args| {
                let verdict = lock(&negative)
                    .judge(&text_arg(args, "text"), &JudgeOptions::default())
                    .ok_or_else(|| value_error("the negative network cannot judge".to_string()))?;
                let reasons: Vec<String> = verdict
                    .reasons
                    .iter()
                    .map(|r| format!("{} ({})", r.reason, fixed(r.blame, 1)))
                    .collect();
                let reasons = if reasons.is_empty() {
                    "none".to_string()
                } else {
                    reasons.join(", ")
                };
                let spans: Vec<String> = verdict.spans.iter().take(3).map(|s| s.fragment.clone()).collect();
                let spans = spans.join("; ");
                let mut text = format!(
                    "{}: {}\nrisk {}, coverage {}\nreasons: {reasons}",
                    verdict.verdict,
                    verdict.why,
                    fixed(verdict.risk, 2),
                    fixed(verdict.coverage, 2)
                );
                if !spans.is_empty() {
                    text.push_str(&format!("\nworst fragments: {spans}"));
                }
                Ok((text, Json::Null))
            },
        ));
    }

    if let Some(solver) = solver {
        let mut tool = Tool::new(
            &format!("{prefix}solve"),
            "Put one task through the whole agent loop: acceptance criteria, tool calls and a judged answer.",
            vec![
                Param::new("task", "string", "the question to settle"),
                Param::optional("max_steps", "integer", "tool calls allowed", Json::Int(4)),
                one_of(
                    Param::optional(
                        "source",
                        "string",
                        "model (the network attempts it) or teacher (the LLM demonstrates)",
                        Json::str("model"),
                    ),
                    &["model", "teacher"],
                ),
            ],
            move |args| solve(&model, negative.as_deref(), &solver, args),
        );
        tool.network = true;
        tools.push(tool);
    }
    tools
}

/// `radixnet_solve`: criteria, one attempt (the network's, or the teacher's
/// demonstration) and the judge's verdict, as text.
fn solve(
    model: &Mutex<Model>,
    negative: Option<&Mutex<Model>>,
    solver: &Solver,
    args: &Json,
) -> Result<(String, Json), String> {
    let toolbox = solver
        .toolbox
        .get()
        .ok_or_else(|| "RuntimeError: the toolbox is gone".to_string())?;
    let config = AgentConfig {
        max_steps: int_arg(args, "max_steps", 4).max(0) as usize,
        model_attempts: 1,
        teach_on_failure: false,
        ..Default::default()
    };
    let task = Task::new("mcp", &text_arg(args, "task"));
    let mut trainer = AgentTrainer::new(solver.client.as_ref(), &toolbox, config).map_err(value_error)?;
    let criteria = trainer.criteria_for(&task).map_err(llm_error)?;
    let mut attempt = if text_arg(args, "source") == "teacher" {
        trainer.solve_with_teacher(&task, &criteria, 0, None)
    } else {
        let mut nets = Nets::Locked { model, negative };
        let mut hooks = Hooks {
            progress: &mut |_| {},
            stop: &|| false,
            checkpoints: None,
        };
        trainer.solve_with_model(&mut nets, &task, 0, "model", &mut hooks)
    }
    .map_err(loop_error)?;
    attempt.verdict = trainer.judge(&task, &criteria, &attempt).map_err(llm_error)?;
    let marks: Vec<String> = criteria.iter().map(|c| format!("  - {c}")).collect();
    let answer = attempt.answer.clone().filter(|a| !a.is_empty());
    Ok((
        format!(
            "answer: {}\nverdict: {}{}\ncriteria:\n{}\n\ntranscript:\n{}",
            answer.as_deref().unwrap_or("(none)"),
            if attempt.verdict.correct { "correct" } else { "failed" },
            attempt
                .verdict
                .score
                .map_or(String::new(), |score| format!(" (score {})", format_g(score))),
            marks.join("\n"),
            attempt.text
        ),
        Json::Null,
    ))
}

// -- the server ---------------------------------------------------------------------------------

/// JSON-RPC 2.0 with one toolbox behind it: MCP's stdio transport.
///
/// [`McpServer::handle`] turns one request into one response (or none, for a
/// notification), so the protocol can be exercised without any stream at all.
pub struct McpServer {
    pub toolbox: Arc<ToolBox>,
    pub name: String,
    pub version: String,
    pub instructions: String,
    /// Whether the client said it is initialised.
    pub initialized: bool,
    /// Tool calls answered.
    pub calls: usize,
}

impl McpServer {
    /// A server over `toolbox`, which must offer something.
    pub fn new(toolbox: Arc<ToolBox>) -> Result<McpServer, String> {
        if toolbox.is_empty() {
            return Err("the toolbox is empty: an MCP server with no tools is of no use".to_string());
        }
        Ok(McpServer {
            toolbox,
            name: SERVER_NAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            instructions: INSTRUCTIONS.to_string(),
            initialized: false,
            calls: 0,
        })
    }

    /// One JSON-RPC request to one response; `None` for a notification, which
    /// is acted on and never answered.
    pub fn handle(&mut self, message: &Json) -> Option<Json> {
        let Json::Obj(_) = message else {
            return Some(error(Json::Null, INVALID_REQUEST, "not a JSON-RPC 2.0 message"));
        };
        if message.get("jsonrpc") != Some(&Json::str("2.0")) {
            return Some(error(Json::Null, INVALID_REQUEST, "not a JSON-RPC 2.0 message"));
        }
        let ident = message.at("id").clone();
        let Some(Json::Str(method)) = message.get("method") else {
            return Some(error(ident, INVALID_REQUEST, "no method"));
        };
        let params = match message.get("params") {
            // JSON-RPC allows positional params; MCP never uses them, and an
            // empty list means none
            None | Some(Json::Null) => Json::Obj(Vec::new()),
            Some(Json::Arr(items)) if items.is_empty() => Json::Obj(Vec::new()),
            Some(other) => other.clone(),
        };
        if !matches!(params, Json::Obj(_)) {
            return Some(error(ident, INVALID_PARAMS, "params must be an object"));
        }
        if ident.is_null() {
            if method == "notifications/initialized" {
                self.initialized = true;
            }
            return None;
        }
        let answer = match method.as_str() {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(Json::Obj(Vec::new())),
            "tools/list" => Ok(self.tools_list()),
            "tools/call" => self.tools_call(&params),
            other => {
                return Some(error(
                    ident,
                    METHOD_NOT_FOUND,
                    &format!("unknown method {}", python_repr(other)),
                ))
            }
        };
        Some(match answer {
            Ok(result) => Json::obj([("jsonrpc", Json::str("2.0")), ("id", ident), ("result", result)]),
            Err(Refusal::Params(why)) => error(ident, INVALID_PARAMS, &why),
        })
    }

    fn initialize(&mut self, params: &Json) -> Json {
        self.initialized = true;
        let asked = match params.get("protocolVersion") {
            Some(Json::Str(version)) if !version.is_empty() => version.clone(),
            _ => PROTOCOL_VERSION.to_string(),
        };
        Json::obj([
            ("protocolVersion", Json::str(asked)),
            (
                "capabilities",
                Json::obj([("tools", Json::obj([("listChanged", Json::Bool(false))]))]),
            ),
            (
                "serverInfo",
                Json::obj([
                    ("name", Json::str(self.name.clone())),
                    ("version", Json::str(self.version.clone())),
                ]),
            ),
            ("instructions", Json::str(self.instructions.clone())),
        ])
    }

    fn tools_list(&self) -> Json {
        Json::obj([("tools", Json::Arr(self.toolbox.tools().iter().map(schema).collect()))])
    }

    fn tools_call(&mut self, params: &Json) -> Result<Json, Refusal> {
        let name = match params.get("name") {
            Some(Json::Str(name)) if !name.is_empty() => name.clone(),
            _ => return Err(Refusal::Params("tools/call needs a tool 'name'".to_string())),
        };
        let arguments = match params.get("arguments") {
            Some(value) if truthy(value) => value.clone(),
            _ => Json::Obj(Vec::new()),
        };
        if !matches!(arguments, Json::Obj(_)) {
            return Err(Refusal::Params("'arguments' must be an object".to_string()));
        }
        if !self.toolbox.has(&name) {
            return Err(Refusal::Params(format!(
                "unknown tool {} (have: {})",
                python_repr(&name),
                self.toolbox.names().join(", ")
            )));
        }
        let result = self.toolbox.call(&name, &arguments);
        self.calls += 1;
        // a tool that failed is a result with isError, not a protocol error:
        // the client shows it to its model
        let text = if result.ok {
            result.output.clone()
        } else {
            format!("ERROR: {}", result.error.as_deref().unwrap_or(""))
        };
        Ok(Json::obj([
            (
                "content",
                Json::Arr(vec![Json::obj([
                    ("type", Json::str("text")),
                    ("text", Json::str(text)),
                ])]),
            ),
            ("isError", Json::Bool(!result.ok)),
        ]))
    }

    /// One line of the stream: its answer, if it gets one.  A line that is not
    /// JSON is answered with Python's parse error; a message the server
    /// failed on is answered with an internal error, and the session goes on.
    pub fn answer_line(&mut self, line: &str) -> Option<Json> {
        let message = match pyjson::loads(line) {
            Ok(message) => message,
            Err(why) => return Some(error(Json::Null, PARSE_ERROR, &format!("invalid JSON: {why}"))),
        };
        match std::panic::catch_unwind(AssertUnwindSafe(|| self.handle(&message))) {
            Ok(response) => response,
            Err(_) => {
                crate::log_error!(LOG, "the server failed to handle a message");
                let ident = match &message {
                    Json::Obj(_) => message.at("id").clone(),
                    _ => Json::Null,
                };
                Some(error(ident, INTERNAL_ERROR, "the server failed to handle the message"))
            }
        }
    }

    /// Reads newline-delimited requests and writes the responses until the
    /// input ends.  Lines end at `\n`, `\r\n` or `\r`, as Python's text stdin
    /// reads them; blank lines are skipped.
    pub fn run(&mut self, input: &mut dyn BufRead, output: &mut dyn Write) -> std::io::Result<()> {
        let mut raw = Vec::new();
        loop {
            raw.clear();
            if input.read_until(b'\n', &mut raw)? == 0 {
                return Ok(());
            }
            let text = String::from_utf8_lossy(&raw).replace("\r\n", "\n");
            for line in text.split(['\n', '\r']) {
                let line = line.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
                if line.is_empty() {
                    continue;
                }
                if let Some(response) = self.answer_line(line) {
                    output.write_all(dumps(&response, false, false).as_bytes())?;
                    output.write_all(b"\n")?;
                    output.flush()?;
                }
            }
        }
    }
}

// -- the command line ---------------------------------------------------------------------------

/// `radixnet mcp`: the tools (the tool flags pick them) and - unless
/// `--no-model` - the network at `--model`, its negative network (with
/// `--blame`, or when the file exists) and `radixnet_solve` (unless
/// `--no-solve`; `--url`, `--agent-model` and `--timeout` reach the LLM),
/// served on stdin / stdout until the client closes the stream.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    if std::env::var_os("RADIXNET_LOG").is_none()
        && args.get("log").is_none()
        && !args.on("verbose")
        && !args.on("quiet")
    {
        // the notes are the one thing a person reads, and stderr is theirs
        crate::log::set_level(crate::log::Level::Info);
    }
    let mut toolbox = crate::tools::build_toolbox(args)?;
    let slot = ToolboxRef::default();
    let mut model_note = None;
    if !args.on("no-model") {
        let model = ctx.open(false)?;
        let origin = if std::path::Path::new(&ctx.model_path).is_file() {
            format!(
                "model {} ({}, {} epochs trained)",
                ctx.model_path,
                model.kind(),
                model.meta.epochs_total.value
            )
        } else {
            let mut detail = format!("seed {}, kind {}", ctx.seed, model.kind());
            if !ctx.encoding.is_default() {
                detail.push_str(&format!(", {}", ctx.encoding.describe()));
            }
            format!("new model ({detail})")
        };
        let negative = if args.on("blame") || std::path::Path::new(&ctx.negative_path()).is_file() {
            Some(ctx.open_negative(false)?)
        } else {
            None
        };
        let solver = if args.on("no-solve") {
            None
        } else {
            // the requests name the agent's model, as Python's do; the
            // client's own is only its fallback
            let client = OllamaClient::new(
                &args.str("url", ""),
                &args.str("agent-model", ""),
                crate::llm::timeout_flag(ctx)?,
            )?;
            Some(Solver {
                client: Arc::new(client),
                toolbox: slot.clone(),
            })
        };
        model_note = Some(format!(
            "  model: {origin}{}",
            if negative.is_some() {
                ", negative network attached"
            } else {
                ""
            }
        ));
        let negative = negative.map(|n| Arc::new(Mutex::new(n)));
        for tool in model_tools(Arc::new(Mutex::new(model)), negative, solver, PREFIX) {
            toolbox.register(tool)?;
        }
    }
    let toolbox = Arc::new(toolbox);
    slot.set(&toolbox);
    // stdout carries the protocol, so everything a person reads goes to stderr
    crate::log_info!(LOG, "radixnet MCP server: {} tool(s) on stdin/stdout", toolbox.len());
    crate::log_info!(LOG, "  {}", toolbox.names().join(", "));
    if let Some(note) = model_note {
        crate::log_info!(LOG, "{note}");
    }
    let mut server = McpServer::new(toolbox)?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    server
        .run(&mut stdin.lock(), &mut stdout.lock())
        .map_err(|err| format!("the MCP stream failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::toolbox::{default_toolbox, ToolOptions};

    fn offline() -> Arc<ToolBox> {
        Arc::new(
            default_toolbox(ToolOptions {
                offline: true,
                ..Default::default()
            })
            .unwrap(),
        )
    }

    fn request(ident: i64, method: &str, params: &str) -> Json {
        parse(&format!(
            "{{\"jsonrpc\": \"2.0\", \"id\": {ident}, \"method\": \"{method}\", \"params\": {params}}}"
        ))
        .unwrap()
    }

    fn trained() -> Model {
        let mut model = Model::new(0, Default::default()).unwrap();
        let texts = vec![
            "the cat sat on the mat".to_string(),
            "the dog runs in the park".to_string(),
        ];
        model.train(&texts, &Default::default()).unwrap();
        model
    }

    #[test]
    fn the_protocol_answers_as_python_s_does() {
        let mut server = McpServer::new(offline()).unwrap();
        let answer = server.handle(&request(1, "initialize", "{}")).unwrap();
        assert_eq!(
            dumps(&answer, false, false),
            format!(
                "{{\"jsonrpc\": \"2.0\", \"id\": 1, \"result\": {{\"protocolVersion\": \"2024-11-05\", \"capabilities\": \
                 {{\"tools\": {{\"listChanged\": false}}}}, \"serverInfo\": {{\"name\": \"radixnet\", \"version\": \
                 \"{}\"}}, \"instructions\": \"{INSTRUCTIONS}\"}}}}",
                env!("CARGO_PKG_VERSION")
            )
        );
        assert!(server.initialized);
        let echoed = server
            .handle(&request(1, "initialize", "{\"protocolVersion\": \"2025-06-18\"}"))
            .unwrap();
        assert_eq!(echoed.at("result").at("protocolVersion").as_str(), Some("2025-06-18"));
        let tools = server.handle(&request(2, "tools/list", "{}")).unwrap();
        assert_eq!(
            dumps(tools.at("result"), false, false),
            "{\"tools\": [{\"name\": \"calculator\", \"description\": \"Work out an arithmetic expression, for \
             example 2 * (3 + 4) or sqrt(841).\", \"inputSchema\": {\"type\": \"object\", \"properties\": \
             {\"expression\": {\"type\": \"string\", \"description\": \"the expression to evaluate\"}}, \
             \"required\": [\"expression\"]}}]}"
        );
        let call = server
            .handle(&request(
                3,
                "tools/call",
                "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"6*7\"}}",
            ))
            .unwrap();
        assert_eq!(
            dumps(call.at("result"), false, false),
            "{\"content\": [{\"type\": \"text\", \"text\": \"42\"}], \"isError\": false}"
        );
        assert_eq!(server.calls, 1);
        let failed = server
            .handle(&request(
                4,
                "tools/call",
                "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1/0\"}}",
            ))
            .unwrap();
        assert_eq!(failed.at("result").at("isError"), &Json::Bool(true));
        assert!(failed.at("result").at("content").as_array()[0]
            .at("text")
            .as_str()
            .unwrap()
            .starts_with("ERROR: "));
        let missing = server
            .handle(&request(8, "tools/call", "{\"name\": \"calculator\"}"))
            .unwrap();
        assert!(missing.at("result").at("content").as_array()[0]
            .at("text")
            .as_str()
            .unwrap()
            .contains("missing argument"));
    }

    #[test]
    fn refusals_carry_python_s_codes_and_words() {
        let mut server = McpServer::new(offline()).unwrap();
        let refused = |server: &mut McpServer, message: &str| -> (i64, String) {
            let answer = server.handle(&parse(message).unwrap()).unwrap();
            let e = answer.at("error");
            (
                e.at("code").as_i64().unwrap(),
                e.at("message").as_str().unwrap().to_string(),
            )
        };
        assert_eq!(
            refused(
                &mut server,
                "{\"jsonrpc\": \"2.0\", \"id\": 5, \"method\": \"tools/call\", \"params\": {\"name\": \"nope\"}}"
            ),
            (INVALID_PARAMS, "unknown tool 'nope' (have: calculator)".to_string())
        );
        assert_eq!(
            refused(
                &mut server,
                "{\"jsonrpc\": \"2.0\", \"id\": 6, \"method\": \"tools/call\"}"
            ),
            (INVALID_PARAMS, "tools/call needs a tool 'name'".to_string())
        );
        assert_eq!(
            refused(
                &mut server,
                "{\"jsonrpc\": \"2.0\", \"id\": 7, \"method\": \"tools/call\", \"params\": {\"name\": \"calculator\", \
                 \"arguments\": \"text\"}}"
            ),
            (INVALID_PARAMS, "'arguments' must be an object".to_string())
        );
        assert_eq!(
            refused(
                &mut server,
                "{\"jsonrpc\": \"2.0\", \"id\": 10, \"method\": \"nosuch\"}"
            ),
            (METHOD_NOT_FOUND, "unknown method 'nosuch'".to_string())
        );
        assert_eq!(
            refused(&mut server, "\"not an object\""),
            (INVALID_REQUEST, "not a JSON-RPC 2.0 message".to_string())
        );
        assert_eq!(
            refused(&mut server, "{\"id\": 1, \"method\": \"ping\"}"),
            (INVALID_REQUEST, "not a JSON-RPC 2.0 message".to_string())
        );
        assert_eq!(
            refused(&mut server, "{\"jsonrpc\": \"2.0\", \"id\": 1}"),
            (INVALID_REQUEST, "no method".to_string())
        );
        assert_eq!(
            refused(
                &mut server,
                "{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\", \"params\": [1, 2]}"
            ),
            (INVALID_PARAMS, "params must be an object".to_string())
        );
        let ping = server
            .handle(&parse("{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\", \"params\": []}").unwrap())
            .unwrap();
        assert_eq!(ping.at("result"), &Json::Obj(Vec::new()));
        // notifications are acted on and never answered
        assert!(server
            .handle(&parse("{\"jsonrpc\": \"2.0\", \"method\": \"notifications/initialized\"}").unwrap())
            .is_none());
        assert!(server.initialized);
        assert!(server
            .handle(&parse("{\"jsonrpc\": \"2.0\", \"id\": null, \"method\": \"ping\"}").unwrap())
            .is_none());
        assert!(McpServer::new(Arc::new(ToolBox::new())).is_err());
    }

    #[test]
    fn the_stream_answers_line_by_line_and_carries_on() {
        let mut server = McpServer::new(offline()).unwrap();
        let input = "{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"initialize\"}\n\n  \r\n\
                     {\"jsonrpc\": \"2.0\", \"method\": \"notifications/initialized\"}\n\
                     {not json}\n\
                     {\"jsonrpc\": \"2.0\", \"id\": \"b\", \"method\": \"tools/call\", \"params\": {\"name\": \
                     \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}}\r\
                     {\"jsonrpc\": \"2.0\", \"id\": 3, \"method\": \"ping\"}";
        let mut out = Vec::new();
        server.run(&mut input.as_bytes(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert!(lines[0].starts_with("{\"jsonrpc\": \"2.0\", \"id\": 1, \"result\": {\"protocolVersion\""));
        assert_eq!(
            lines[1],
            "{\"jsonrpc\": \"2.0\", \"id\": null, \"error\": {\"code\": -32700, \"message\": \"invalid JSON: \
             Expecting property name enclosed in double quotes: line 1 column 2 (char 1)\"}}"
        );
        assert_eq!(
            lines[2],
            "{\"jsonrpc\": \"2.0\", \"id\": \"b\", \"result\": {\"content\": [{\"type\": \"text\", \"text\": \"4\"}], \
             \"isError\": false}}"
        );
        assert_eq!(lines[3], "{\"jsonrpc\": \"2.0\", \"id\": 3, \"result\": {}}");
    }

    #[test]
    fn the_network_is_offered_as_tools() {
        let model = Arc::new(Mutex::new(trained()));
        let names: Vec<String> = model_tools(model.clone(), None, None, PREFIX)
            .iter()
            .map(|t| t.name.clone())
            .collect();
        assert_eq!(
            names,
            [
                "radixnet_predict",
                "radixnet_generate",
                "radixnet_score",
                "radixnet_stats"
            ]
        );
        let mut box_ = ToolBox::new();
        for tool in model_tools(model.clone(), None, None, PREFIX) {
            box_.register(tool).unwrap();
        }
        let predicted = box_.call(
            "radixnet_predict",
            &parse("{\"prefix_text\": \"the cat\", \"length\": 10}").unwrap(),
        );
        assert!(predicted.output.starts_with("the cat"), "{predicted:?}");
        let generated = box_.call(
            "radixnet_generate",
            &parse("{\"count\": 2, \"max_length\": 20}").unwrap(),
        );
        assert!(generated.ok && generated.output.starts_with("1. "), "{generated:?}");
        let scored = box_.call(
            "radixnet_score",
            &parse("{\"text\": \"the cat sat on the mat\"}").unwrap(),
        );
        assert!(
            scored.output.contains(" per character over 22 characters ("),
            "{}",
            scored.output
        );
        let stats = box_.call("radixnet_stats", &Json::Obj(Vec::new()));
        assert!(
            stats.output.starts_with("{\n  \"kind\": \"count\",\n  \"nodes\": "),
            "{}",
            stats.output
        );
        let bad = box_.call(
            "radixnet_predict",
            &parse("{\"prefix_text\": \"x\", \"mode\": \"nope\"}").unwrap(),
        );
        assert!(!bad.ok);
        let prefixed = model_tools(model, None, None, "net.");
        assert!(prefixed.iter().all(|t| t.name.starts_with("net.")));
    }

    #[test]
    fn judge_needs_a_negative_network() {
        let mut negative = Model::new_negative(1, &Default::default()).unwrap();
        negative
            .blame_with(
                &["the cat sat on the sky".to_string()],
                &crate::negative::BlameOptions {
                    reason: "false".to_string(),
                    severity: 2.0,
                    ..Default::default()
                },
                &mut |_| true,
            )
            .unwrap();
        let tools = model_tools(
            Arc::new(Mutex::new(trained())),
            Some(Arc::new(Mutex::new(negative))),
            None,
            PREFIX,
        );
        let judge = tools.iter().find(|t| t.name == "radixnet_judge").unwrap();
        let (text, _) =
            (judge.handler.as_ref().unwrap())(&parse("{\"text\": \"the cat sat on the sky\"}").unwrap()).unwrap();
        assert!(text.starts_with("reject: "), "{text}");
        assert!(text.contains("\nrisk ") && text.contains("reasons: false ("), "{text}");
    }

    #[test]
    fn numbers_and_documents_are_written_as_python_writes_them() {
        assert_eq!(
            (fixed(f64::NAN, 2), fixed(f64::NEG_INFINITY, 3), fixed(0.125, 2)),
            ("nan".into(), "-inf".into(), "0.12".into())
        );
        let mut out = String::new();
        dumps_indent(
            &parse("{\"a\": [1, {}], \"b\": [], \"é\": \"ü\"}").unwrap(),
            2,
            0,
            &mut out,
        );
        assert_eq!(
            out,
            "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": [],\n  \"\\u00e9\": \"\\u00fc\"\n}"
        );
    }
}
