//! The model in today's format: messages in, an assistant message out -
//! thinking, text, tool calls, streamed (Python's `radixnet/assistant.py`).
//!
//! Every language model is talked to the same way now: a list of `{"role",
//! "content"}` messages goes in, an assistant message comes back, and while it
//! is being written the client sees it arrive - the model's *thinking* first,
//! then the answer, chunk by chunk, over server-sent events.  Two dialects
//! cover every client there is: OpenAI's Chat Completions
//! (`POST /v1/chat/completions`, `reasoning_content`, `data: [DONE]`) and
//! Anthropic's Messages (`POST /v1/messages`, typed content blocks, one event
//! per block).  This module gives the Rust port both.
//!
//! Nothing here is a prompt trick.  A reply is what [`crate::dialogue::reply`]
//! says next after the last line of the conversation, exactly as in the
//! Converse and Chat tabs, and the format is a rendering of what that search
//! already produces:
//!
//! * the **thinking** is the search's own trace, line by line as it happens
//!   (which tail of the line it looked for and whether the graph knew it, how
//!   many paths it weighed, what the negative network vetoed and why, where it
//!   caught itself repeating and how it backed out, what it finally said at what
//!   cost) - never prose the model did not produce;
//! * the **text** streams one chunk per node of the walk;
//! * a `<tool>name {...}</tool>` line the model writes comes back as a **tool
//!   call** when the request offered that tool, and a tool's answer goes back
//!   in as the `<result>` text the agent trains on;
//! * the **stop reason** says how the walk ended, and **usage** counts *units of
//!   the model's encoding* - a character of a character model, a word of a
//!   word model - the units `max_tokens` caps.
//!
//! The thinking's lines are written here character for character as Python
//! writes them, and `../tests/test_rust_parity_assistant.py` holds the two to
//! each other.

use std::cell::RefCell;
use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::cli::Ctx;
use crate::dialogue::{reply, Heard, ReplyOptions, Turn, EXPLORE};
use crate::duo::{guard_report, Filter, FilterConfig, FilterVerdict};
use crate::encoding::{Encoding, Unit};
use crate::graph::FIRST;
use crate::http::{Answer, ApiError, Request, Server, Streamed};
use crate::json::{parse, Json};
use crate::model::Model;
use crate::mt19937::Mt19937;
use crate::service::Service;
use crate::tools::{call_text, parse_call, result_text, Param, Tool, ToolBox, DEFAULT_OBSERVATION_CHARS, PARAM_TYPES};

pub const OPENAI: &str = "openai";
pub const ANTHROPIC: &str = "anthropic";
/// The two dialects.
pub const FORMATS: [&str; 2] = [OPENAI, ANTHROPIC];

pub const END_TURN: &str = "end_turn";
pub const MAX_TOKENS: &str = "max_tokens";
pub const STOP_SEQUENCE: &str = "stop_sequence";
pub const TOOL_USE: &str = "tool_use";
pub const REFUSAL: &str = "refusal";

/// A model's id is its kind behind this prefix: `radixnet-count`.
pub const MODEL_PREFIX: &str = "radixnet-";
/// Units a reply may add when the request sets no `max_tokens`.
pub const DEFAULT_MAX_TOKENS: usize = 60;
/// How many alternative replies (`n`) one request may ask for.
pub const MAX_CHOICES: usize = 8;

/// How a reply ended, in Chat Completions' words.
pub fn finish_reason(stop: &str) -> &'static str {
    match stop {
        MAX_TOKENS => "length",
        TOOL_USE => "tool_calls",
        REFUSAL => "content_filter",
        _ => "stop",
    }
}

/// A request that cannot be answered as it stands; `param` names the field
/// when one is to blame.
#[derive(Clone, Debug)]
pub struct AskError {
    pub message: String,
    pub param: Option<String>,
}

impl AskError {
    fn new(message: impl Into<String>, param: Option<&str>) -> AskError {
        AskError {
            message: message.into(),
            param: param.map(str::to_string),
        }
    }
}

/// One line of the conversation as the model reads it.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub text: String,
}

/// A tool the request offered: a written call to it comes back as a tool call.
#[derive(Clone, Debug)]
pub struct OfferedTool {
    pub name: String,
    pub description: String,
    /// The JSON schema of its arguments (an object).
    pub parameters: Json,
}

/// A request in either dialect, normalised.
#[derive(Clone, Debug)]
pub struct Ask {
    pub messages: Vec<Message>,
    pub system: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub stop: Vec<String>,
    pub n: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub tools: Vec<OfferedTool>,
    pub thinking: bool,
    pub mode: String,
    pub context: usize,
    pub k: usize,
    /// 0: the search's own default width.
    pub beam: usize,
    pub step_penalty: f64,
    pub explore: usize,
    pub avoid_repeats: bool,
    pub avoid_word_repeats: bool,
    pub learn: bool,
    pub guard: bool,
    pub seed: Option<i64>,
}

impl Default for Ask {
    fn default() -> Ask {
        Ask {
            messages: Vec::new(),
            system: String::new(),
            model: String::new(),
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: 1.0,
            stop: Vec::new(),
            n: 1,
            stream: false,
            include_usage: false,
            tools: Vec::new(),
            thinking: true,
            mode: "beam".to_string(),
            context: 12,
            k: 5,
            beam: 0,
            step_penalty: 0.0,
            explore: EXPLORE,
            avoid_repeats: true,
            avoid_word_repeats: true,
            learn: true,
            guard: true,
            seed: None,
        }
    }
}

fn is_tool_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.first().is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

impl Ask {
    /// Why the request cannot be answered, if it cannot.
    pub fn validate(&self) -> Result<(), AskError> {
        if self.messages.is_empty() {
            return Err(AskError::new(
                "messages must hold at least one user or assistant message",
                Some("messages"),
            ));
        }
        for (i, message) in self.messages.iter().enumerate() {
            if !matches!(message.role.as_str(), "user" | "assistant" | "tool") {
                return Err(AskError::new(
                    format!(
                        "messages[{i}].role must be user, assistant or tool (got {})",
                        py_quote(&message.role)
                    ),
                    Some("messages"),
                ));
            }
        }
        if self.max_tokens < 1 {
            return Err(AskError::new("max_tokens must be >= 1", Some("max_tokens")));
        }
        if self.temperature < 0.0 {
            return Err(AskError::new("temperature must be >= 0", Some("temperature")));
        }
        if self.n < 1 || self.n > MAX_CHOICES {
            return Err(AskError::new(
                format!("n must be between 1 and {MAX_CHOICES}"),
                Some("n"),
            ));
        }
        if self.mode != "beam" && self.mode != "sample" {
            return Err(AskError::new("mode must be beam or sample", Some("mode")));
        }
        if self.k < 1 {
            return Err(AskError::new("k must be >= 1", Some("k")));
        }
        if self.step_penalty < 0.0 {
            return Err(AskError::new("step_penalty must be >= 0", Some("step_penalty")));
        }
        for (i, seq) in self.stop.iter().enumerate() {
            if seq.is_empty() {
                return Err(AskError::new(
                    format!("stop[{i}] must be a non-empty string"),
                    Some("stop"),
                ));
            }
        }
        let mut names: Vec<&str> = Vec::new();
        for (i, tool) in self.tools.iter().enumerate() {
            if !is_tool_name(&tool.name) {
                return Err(AskError::new(
                    format!("tools[{i}].name must be an identifier (got {})", py_quote(&tool.name)),
                    Some("tools"),
                ));
            }
            if names.contains(&tool.name.as_str()) {
                return Err(AskError::new(
                    format!("tools[{i}].name {} is offered twice", py_quote(&tool.name)),
                    Some("tools"),
                ));
            }
            names.push(&tool.name);
        }
        Ok(())
    }

    /// The line the reply continues: the last message.
    pub fn previous(&self) -> &str {
        self.messages.last().map(|m| m.text.as_str()).unwrap_or("")
    }

    /// Whether the last message is the assistant's own: the reply then
    /// continues it and returns only what it adds.
    pub fn prefill(&self) -> bool {
        self.messages.last().is_some_and(|m| m.role == "assistant")
    }
}

/// Python's `repr` of a string, for the messages that quote a value.
fn py_quote(text: &str) -> String {
    crate::negative::python_repr(text)
}

fn json_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Int(_) | Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Arr(_) => "array",
        Json::Obj(_) => "object",
    }
}

fn field_int(body: &Json, name: &str, default: Option<i64>, minimum: Option<i64>) -> Result<Option<i64>, AskError> {
    let value = match body.get(name) {
        None | Some(Json::Null) => return Ok(default),
        Some(value) => value,
    };
    let number = match value {
        Json::Int(i) => *i,
        Json::Num(f) if f.fract() == 0.0 && f.is_finite() => *f as i64,
        other => {
            return Err(AskError::new(
                format!("{name} must be an integer (got {})", json_type(other)),
                Some(name),
            ))
        }
    };
    if let Some(minimum) = minimum {
        if number < minimum {
            return Err(AskError::new(
                format!("{name} must be >= {minimum} (got {number})"),
                Some(name),
            ));
        }
    }
    Ok(Some(number))
}

fn field_num(body: &Json, name: &str, default: f64, minimum: Option<f64>) -> Result<f64, AskError> {
    let value = match body.get(name) {
        None | Some(Json::Null) => return Ok(default),
        Some(value) => value,
    };
    let number = match value {
        Json::Int(i) => *i as f64,
        Json::Num(f) if !f.is_nan() => *f,
        other => {
            return Err(AskError::new(
                format!("{name} must be a number (got {})", json_type(other)),
                Some(name),
            ))
        }
    };
    if let Some(minimum) = minimum {
        if number < minimum {
            return Err(AskError::new(
                format!("{name} must be >= {} (got {})", py_num(minimum), py_num(number)),
                Some(name),
            ));
        }
    }
    Ok(number)
}

/// A number as Python prints it in a message: `-1`, `0.5`.
fn py_num(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        crate::json::py_repr(value)
    }
}

fn field_bool(body: &Json, name: &str, default: bool) -> Result<bool, AskError> {
    match body.get(name) {
        None | Some(Json::Null) => Ok(default),
        Some(Json::Bool(b)) => Ok(*b),
        Some(other) => Err(AskError::new(
            format!("{name} must be a boolean (got {})", json_type(other)),
            Some(name),
        )),
    }
}

fn field_str(body: &Json, name: &str, default: &str) -> Result<String, AskError> {
    match body.get(name) {
        None | Some(Json::Null) => Ok(default.to_string()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(other) => Err(AskError::new(
            format!("{name} must be a string (got {})", json_type(other)),
            Some(name),
        )),
    }
}

/// The dialogue's own dials, read by the names `/api/converse` uses.
fn dials(body: &Json, ask: &mut Ask) -> Result<(), AskError> {
    let mode = field_str(body, "mode", "beam")?.to_lowercase();
    ask.mode = if mode.is_empty() || mode == "dijkstra" {
        "beam".to_string()
    } else {
        mode
    };
    ask.context = field_int(body, "context", Some(12), Some(0))?.unwrap_or(12) as usize;
    ask.k = field_int(body, "k", Some(5), Some(1))?.unwrap_or(5) as usize;
    ask.beam = field_int(body, "beam", None, Some(1))?.unwrap_or(0) as usize;
    ask.step_penalty = field_num(body, "step_penalty", 0.0, Some(0.0))?;
    ask.explore = field_int(body, "explore", Some(EXPLORE as i64), Some(0))?.unwrap_or(EXPLORE as i64) as usize;
    ask.avoid_repeats = field_bool(body, "avoid_repeats", true)?;
    ask.avoid_word_repeats = field_bool(body, "avoid_word_repeats", true)?;
    ask.learn = field_bool(body, "learn", true)?;
    ask.guard = field_bool(body, "guard", true)?;
    ask.seed = field_int(body, "seed", None, None)?;
    Ok(())
}

/// The text of a message's `content`: a string, or a list of typed parts joined by newlines.
fn text_parts(content: Option<&Json>, where_: &str, dialect: &str) -> Result<String, AskError> {
    let content = match content {
        None | Some(Json::Null) => return Ok(String::new()),
        Some(Json::Str(s)) => return Ok(s.clone()),
        Some(Json::Arr(items)) => items,
        Some(_) => {
            return Err(AskError::new(
                format!("{where_}.content must be a string or a list of content parts"),
                Some("messages"),
            ))
        }
    };
    let mut parts: Vec<String> = Vec::new();
    for (i, part) in content.iter().enumerate() {
        if let Json::Str(s) = part {
            parts.push(s.clone());
            continue;
        }
        if !matches!(part, Json::Obj(_)) {
            return Err(AskError::new(
                format!("{where_}.content[{i}] must be an object with a type"),
                Some("messages"),
            ));
        }
        let kind = part.at("type").as_str().unwrap_or("");
        match kind {
            "text" | "refusal" => {
                let key = if kind == "text" { "text" } else { "refusal" };
                match part.at(key).as_str() {
                    Some(text) => parts.push(text.to_string()),
                    None => {
                        return Err(AskError::new(
                            format!("{where_}.content[{i}].text must be a string"),
                            Some("messages"),
                        ))
                    }
                }
            }
            "thinking" | "redacted_thinking" => continue, // what the model thought is not what it said
            "tool_use" if dialect == ANTHROPIC => {
                let name = match part.at("name").as_str() {
                    Some(name) if !name.is_empty() => name,
                    _ => {
                        return Err(AskError::new(
                            format!("{where_}.content[{i}].name must be the tool's name"),
                            Some("messages"),
                        ))
                    }
                };
                let input = match part.get("input") {
                    None | Some(Json::Null) => Json::Obj(Vec::new()),
                    Some(Json::Obj(pairs)) => Json::Obj(pairs.clone()),
                    Some(_) => {
                        return Err(AskError::new(
                            format!("{where_}.content[{i}].input must be an object"),
                            Some("messages"),
                        ))
                    }
                };
                parts.push(call_text(name, &input).trim_end_matches('\n').to_string());
            }
            "tool_result" if dialect == ANTHROPIC => {
                let inner = text_parts(part.get("content"), &format!("{where_}.content[{i}]"), dialect)?;
                parts.push(
                    result_text(&inner, Some(DEFAULT_OBSERVATION_CHARS))
                        .trim_end_matches('\n')
                        .to_string(),
                );
            }
            _ => {
                let what = match part.get("type") {
                    Some(Json::Str(s)) => py_quote(s),
                    Some(other) => json_type(other).to_string(),
                    None => "null".to_string(),
                };
                return Err(AskError::new(
                    format!("{where_}.content[{i}]: content of type {what} is not supported; this model reads text"),
                    Some("messages"),
                ));
            }
        }
    }
    Ok(parts.join("\n"))
}

/// Chat Completions: an assistant message's `tool_calls` as the `<tool>` lines the model writes.
fn tool_calls_text(calls: Option<&Json>, where_: &str) -> Result<String, AskError> {
    let calls = match calls {
        None | Some(Json::Null) => return Ok(String::new()),
        Some(Json::Arr(items)) => items,
        Some(_) => {
            return Err(AskError::new(
                format!("{where_}.tool_calls must be a list"),
                Some("messages"),
            ))
        }
    };
    let mut lines = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        let function = call.at("function");
        let name = match function.at("name").as_str() {
            Some(name) if matches!(function, Json::Obj(_)) => name,
            _ => {
                return Err(AskError::new(
                    format!("{where_}.tool_calls[{i}] must be a function call with a name"),
                    Some("messages"),
                ))
            }
        };
        let arguments = match function.get("arguments") {
            Some(Json::Str(raw)) if !raw.trim().is_empty() => match parse(raw) {
                Ok(Json::Obj(pairs)) => Json::Obj(pairs),
                Ok(_) => {
                    return Err(AskError::new(
                        format!("{where_}.tool_calls[{i}].function.arguments must encode an object"),
                        Some("messages"),
                    ))
                }
                Err(why) => {
                    return Err(AskError::new(
                        format!("{where_}.tool_calls[{i}].function.arguments is not JSON: {why}"),
                        Some("messages"),
                    ))
                }
            },
            Some(Json::Obj(pairs)) => Json::Obj(pairs.clone()),
            _ => Json::Obj(Vec::new()),
        };
        lines.push(call_text(name, &arguments).trim_end_matches('\n').to_string());
    }
    Ok(lines.join("\n"))
}

/// The tools offered, in one shape.
fn tools(items: Option<&Json>, dialect: &str) -> Result<Vec<OfferedTool>, AskError> {
    let items = match items {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(Json::Arr(items)) => items,
        Some(_) => return Err(AskError::new("tools must be a list", Some("tools"))),
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if !matches!(item, Json::Obj(_)) {
            return Err(AskError::new(format!("tools[{i}] must be an object"), Some("tools")));
        }
        let (spec, schema) = if dialect == OPENAI {
            if item.at("type").as_str().unwrap_or("function") != "function" {
                return Err(AskError::new(
                    format!("tools[{i}].type must be \"function\""),
                    Some("tools"),
                ));
            }
            let spec = item.get("function").unwrap_or(item);
            if !matches!(spec, Json::Obj(_)) {
                return Err(AskError::new(
                    format!("tools[{i}].function must be an object"),
                    Some("tools"),
                ));
            }
            (spec, spec.get("parameters"))
        } else {
            (item, item.get("input_schema"))
        };
        let name = match spec.at("name").as_str() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => return Err(AskError::new(format!("tools[{i}] has no name"), Some("tools"))),
        };
        let parameters = match schema {
            None | Some(Json::Null) => Json::Obj(Vec::new()),
            Some(Json::Obj(pairs)) => Json::Obj(pairs.clone()),
            Some(_) => {
                return Err(AskError::new(
                    format!("tools[{i}]: the parameters must be a JSON schema object"),
                    Some("tools"),
                ))
            }
        };
        let description = match spec.get("description") {
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Null) | None => String::new(),
            Some(other) => crate::tools::py_str(other),
        };
        out.push(OfferedTool {
            name,
            description,
            parameters,
        });
    }
    Ok(out)
}

/// The stop sequences: a list of non-empty strings (Chat Completions also takes one string).
fn stops(value: Option<&Json>, name: &str, one_string: bool) -> Result<Vec<String>, AskError> {
    let items: Vec<String> = match value {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(Json::Str(s)) if one_string => vec![s.clone()],
        Some(Json::Arr(items)) if items.iter().all(|s| matches!(s, Json::Str(_))) => {
            items.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()
        }
        Some(_) => {
            let what = if one_string {
                "a string or a list of strings"
            } else {
                "a list of strings"
            };
            return Err(AskError::new(format!("{name} must be {what}"), Some(name)));
        }
    };
    if items.iter().any(|s| s.is_empty()) {
        return Err(AskError::new(
            format!("{name} must not hold an empty sequence"),
            Some(name),
        ));
    }
    Ok(items)
}

/// The conversation and the system prompt out of a `messages` list.
fn messages(items: &Json, dialect: &str) -> Result<(Vec<Message>, String), AskError> {
    let items = match items {
        Json::Arr(items) => items,
        _ => {
            return Err(AskError::new(
                "messages must be a list of {role, content} objects",
                Some("messages"),
            ))
        }
    };
    let mut system: Vec<String> = Vec::new();
    let mut out: Vec<Message> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let where_ = format!("messages[{i}]");
        if !matches!(item, Json::Obj(_)) {
            return Err(AskError::new(
                format!("{where_} must be an object with a role and content"),
                Some("messages"),
            ));
        }
        let role = match item.at("role").as_str() {
            Some(role) => role.to_lowercase(),
            None => {
                return Err(AskError::new(
                    format!("{where_}.role must be a string"),
                    Some("messages"),
                ))
            }
        };
        let role = match role.as_str() {
            "developer" => "system".to_string(),
            "function" => "tool".to_string(),
            _ => role,
        };
        if dialect == ANTHROPIC && role != "user" && role != "assistant" {
            return Err(AskError::new(
                format!("{where_}.role must be user or assistant (got {})", py_quote(&role)),
                Some("messages"),
            ));
        }
        if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
            return Err(AskError::new(
                format!(
                    "{where_}.role must be one of system, user, assistant, tool (got {})",
                    py_quote(&role)
                ),
                Some("messages"),
            ));
        }
        let mut text = text_parts(item.get("content"), &where_, dialect)?;
        if role == "assistant" && dialect == OPENAI {
            let calls = tool_calls_text(item.get("tool_calls"), &where_)?;
            text = [text, calls]
                .into_iter()
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }
        if role == "tool" && dialect == OPENAI {
            text = result_text(&text, Some(DEFAULT_OBSERVATION_CHARS))
                .trim_end_matches('\n')
                .to_string();
        }
        if role == "system" {
            if !text.trim().is_empty() {
                system.push(text);
            }
            continue;
        }
        if text.trim().is_empty() {
            continue; // an empty line says nothing and is not heard
        }
        out.push(Message { role, text });
    }
    Ok((out, system.join("\n")))
}

/// A Chat Completions request as an [`Ask`] (Python's `parse_openai`).
pub fn parse_openai(body: &Json) -> Result<Ask, AskError> {
    if !matches!(body, Json::Obj(_)) {
        return Err(AskError::new("the request body must be a JSON object", None));
    }
    let mut ask = Ask {
        model: field_str(body, "model", "")?,
        ..Ask::default()
    };
    let Some(items) = body.get("messages") else {
        return Err(AskError::new("messages is required", Some("messages")));
    };
    let (found, system) = messages(items, OPENAI)?;
    ask.messages = found;
    ask.system = system;
    let mut limit = field_int(body, "max_completion_tokens", None, Some(1))?;
    if limit.is_none() {
        limit = field_int(body, "max_tokens", None, Some(1))?;
    }
    ask.max_tokens = limit.map(|n| n as usize).unwrap_or(DEFAULT_MAX_TOKENS);
    ask.temperature = field_num(body, "temperature", 1.0, Some(0.0))?;
    ask.stop = stops(body.get("stop"), "stop", true)?;
    ask.n = field_int(body, "n", Some(1), Some(1))?.unwrap_or(1) as usize;
    ask.stream = field_bool(body, "stream", false)?;
    match body.get("stream_options") {
        None | Some(Json::Null) => {}
        Some(options @ Json::Obj(_)) => ask.include_usage = field_bool(options, "include_usage", false)?,
        Some(_) => {
            return Err(AskError::new(
                "stream_options must be an object",
                Some("stream_options"),
            ))
        }
    }
    ask.tools = tools(body.get("tools"), OPENAI)?;
    if body.at("tool_choice").as_str() == Some("none") {
        ask.tools.clear();
    }
    ask.thinking = field_bool(body, "thinking", true)? && field_str(body, "reasoning_effort", "")? != "none";
    dials(body, &mut ask)?;
    ask.validate()?;
    Ok(ask)
}

/// A Messages request as an [`Ask`] (Python's `parse_anthropic`).
pub fn parse_anthropic(body: &Json) -> Result<Ask, AskError> {
    if !matches!(body, Json::Obj(_)) {
        return Err(AskError::new("the request body must be a JSON object", None));
    }
    let mut ask = Ask {
        model: field_str(body, "model", "")?,
        ..Ask::default()
    };
    let Some(items) = body.get("messages") else {
        return Err(AskError::new("messages is required", Some("messages")));
    };
    let (found, _) = messages(items, ANTHROPIC)?;
    ask.messages = found;
    ask.system = text_parts(body.get("system"), "system", ANTHROPIC)?;
    ask.max_tokens = field_int(body, "max_tokens", None, Some(1))?
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    ask.temperature = field_num(body, "temperature", 1.0, Some(0.0))?;
    ask.stop = stops(body.get("stop_sequences"), "stop_sequences", false)?;
    ask.stream = field_bool(body, "stream", false)?;
    ask.tools = tools(body.get("tools"), ANTHROPIC)?;
    if body.at("tool_choice").at("type").as_str() == Some("none") {
        ask.tools.clear();
    }
    match body.get("thinking") {
        None | Some(Json::Null) => {}
        Some(Json::Bool(b)) => ask.thinking = *b,
        Some(obj @ Json::Obj(_)) if matches!(obj.at("type").as_str(), Some("enabled" | "disabled" | "adaptive")) => {
            ask.thinking = obj.at("type").as_str() != Some("disabled");
        }
        Some(_) => {
            return Err(AskError::new(
                "thinking must be {\"type\": \"enabled\"} or {\"type\": \"disabled\"}",
                Some("thinking"),
            ))
        }
    }
    dials(body, &mut ask)?;
    ask.validate()?;
    Ok(ask)
}

/// [`parse_openai`] or [`parse_anthropic`], by dialect.
pub fn parse_request(body: &Json, dialect: &str) -> Result<Ask, AskError> {
    match dialect {
        OPENAI => parse_openai(body),
        ANTHROPIC => parse_anthropic(body),
        other => Err(AskError::new(
            format!("format must be one of openai, anthropic (got {})", py_quote(other)),
            Some("format"),
        )),
    }
}

// -- the model as a model id ----------------------------------------------------------------------

/// `radixnet-<kind>`: what a model is called in `/v1/models` and in every answer.
pub fn model_id(model: &Model) -> String {
    format!("{MODEL_PREFIX}{}", model.kind())
}

/// The kind a requested model id names (`""`: whichever model is active).
pub fn kind_of_id(name: &str) -> String {
    let text = name.trim().to_lowercase();
    let text = text.strip_prefix(MODEL_PREFIX).unwrap_or(&text).to_string();
    match text.as_str() {
        "" | "radixnet" | "default" | "active" => String::new(),
        _ => text,
    }
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 24 hex characters nothing else in this process gets: the time, and a count.
fn token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mixed = nanos ^ (count.wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ (std::process::id() as u64) << 32;
    format!("{mixed:016x}{:08x}", (count as u32).wrapping_mul(2_654_435_761))
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// -- the reply ------------------------------------------------------------------------------------

/// A call the model wrote to a tool the request offered.
#[derive(Clone, Debug)]
pub struct ToolCallOut {
    pub token: String,
    pub name: String,
    pub input: Json,
}

/// One reply: the thinking, the text, the calls, how it ended, and the turn behind it.
#[derive(Clone, Debug)]
pub struct Choice {
    pub index: usize,
    pub thinking: String,
    pub text: String,
    pub tool_calls: Vec<ToolCallOut>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub turn: Option<Turn>,
    /// The guard's report, `Json::Null` when nothing guarded.
    pub guard: Json,
    pub output_units: usize,
    pub thinking_units: usize,
}

/// What [`respond`] returns.
#[derive(Clone, Debug)]
pub struct Reply {
    pub token: String,
    pub model: String,
    pub created: i64,
    pub kind: String,
    pub units: String,
    pub input_units: usize,
    pub choices: Vec<Choice>,
    pub thinking: bool,
}

/// What the request holds, in the model's units (`count_tokens`).
pub fn input_units(encoding: &Encoding, ask: &Ask) -> usize {
    encoding.len(&ask.system) + ask.messages.iter().map(|m| encoding.len(&m.text)).sum::<usize>()
}

/// `{"input", "output", "thinking", "total"}` in the model's units.
pub fn usage_of(reply: &Reply) -> Json {
    let output: usize = reply.choices.iter().map(|c| c.output_units).sum();
    let thinking: usize = reply.choices.iter().map(|c| c.thinking_units).sum();
    Json::obj([
        ("input", Json::Int(reply.input_units as i64)),
        ("output", Json::Int(output as i64)),
        ("thinking", Json::Int(thinking as i64)),
        ("total", Json::Int((reply.input_units + output) as i64)),
    ])
}

/// Text in double quotes, JSON style - the quoting every port's thinking uses.
fn q(text: &str) -> String {
    Json::str(text).render(0)
}

fn count(n: usize, what: &str) -> String {
    format!("{n} {what}(s)")
}

/// Turns the search's trace into the lines of the thinking, one event at a time.
struct Narrator<'e> {
    index: usize,
    lines: Vec<String>,
    emit: &'e mut dyn FnMut(&Json),
}

impl<'e> Narrator<'e> {
    fn line(&mut self, text: String) {
        let delta = if self.lines.is_empty() {
            text.clone()
        } else {
            format!("\n{text}")
        };
        self.lines.push(text);
        (self.emit)(&Json::obj([
            ("type", Json::str("thinking")),
            ("index", Json::Int(self.index as i64)),
            ("text", Json::str(delta)),
        ]));
    }

    fn trace(&mut self, event: &Json) {
        let kind = event.at("kind").as_str().unwrap_or("");
        match kind {
            "context" => {
                if event.at("usable").as_bool() != Some(true) {
                    let context = event.at("context").as_str().unwrap_or("");
                    self.line(format!(
                        "looking for {} in the graph: not there whole; dropping a word",
                        q(context)
                    ));
                }
            }
            "candidates" => {
                let n = event.at("offered").as_i64().unwrap_or(0) as usize;
                let what = if event.at("mode").as_str() == Some("sample") {
                    format!("{} drawn", count(n, "walk"))
                } else {
                    format!("{} weighed", count(n, "path"))
                };
                let context = event.at("context").as_str().unwrap_or("");
                if context.is_empty() {
                    self.line(format!("starting a fresh text from the beginning: {what}"));
                } else {
                    self.line(format!("looking for {} in the graph: found, {what}", q(context)));
                }
            }
            "pick" => {
                let skipped = event.at("skipped").as_i64().unwrap_or(0) - event.at("vetoed").as_i64().unwrap_or(0);
                if event.at("repeat").as_bool() == Some(true) {
                    self.line(
                        "every path repeats something already said; the best of them is kept as a last resort"
                            .to_string(),
                    );
                } else if skipped > 0 {
                    self.line(format!("skipped {skipped} (empty, or already said)"));
                }
            }
            "rethink" => {
                let noticed = event.at("noticed").as_str().unwrap_or("");
                let caught = if event.at("kind_of").as_str() == Some("stutter") {
                    format!("caught itself saying {} twice", q(noticed))
                } else {
                    format!("caught itself repeating {}", q(noticed))
                };
                let explored = event.at("explored").as_i64().unwrap_or(0) as usize;
                let cut = event.at("cut").as_str().unwrap_or("");
                let mut text = if event.at("steps").as_i64().unwrap_or(0) == 0 {
                    format!("{caught}; the words it picked up, not its own")
                } else if event.at("found").as_bool() == Some(true) {
                    format!(
                        "{caught}; kept {} and found another way on in {}",
                        q(cut),
                        count(explored, "path")
                    )
                } else {
                    format!(
                        "{caught}; kept {}, weighed {}, found nothing new",
                        q(cut),
                        count(explored, "path")
                    )
                };
                if event.at("taught").as_i64().unwrap_or(-1) >= 0 {
                    text.push_str(" (and learned to hand over there)");
                }
                self.line(text);
            }
            "fresh" => self.line("nothing new follows the line; changing the subject with a fresh text".to_string()),
            _ => {}
        }
    }
}

/// `text` cut into what each node of the walk added, so the pieces join back
/// into `text` (Python's `deltas`).  A text the walk does not line up with (one
/// cut at the cap) comes back whole.
pub fn deltas(encoding: &Encoding, labels: &[String], node_ids: &[usize], text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let real: Vec<&String> = node_ids
        .iter()
        .zip(labels.iter())
        .filter(|(node, _)| **node >= FIRST)
        .map(|(_, label)| label)
        .collect();
    if real.len() < 2 {
        return vec![text.to_string()];
    }
    let overlap = encoding.overlap();
    let pieces: Vec<String> = real[1..]
        .iter()
        .map(|label| {
            let units = encoding.units(label);
            units.slice(overlap, units.len())
        })
        .collect();
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let tail = encoding.join(&refs);
    if tail.is_empty() {
        return vec![text.to_string()];
    }
    let words = encoding.unit == Unit::Words;
    let head = if words {
        if text == tail {
            String::new()
        } else if let Some(head) = text.strip_suffix(&format!(" {tail}")) {
            head.to_string()
        } else {
            return vec![text.to_string()];
        }
    } else if let Some(head) = text.strip_suffix(tail.as_str()) {
        head.to_string()
    } else {
        return vec![text.to_string()];
    };
    let mut out: Vec<String> = Vec::new();
    if !head.is_empty() {
        out.push(head);
    }
    for piece in pieces {
        if piece.is_empty() {
            continue;
        }
        if words && !out.is_empty() {
            out.push(format!(" {piece}"));
        } else {
            out.push(piece);
        }
    }
    out
}

/// The pieces cut to the bytes `[start, end)` of their concatenation.
fn windowed(pieces: &[String], start: usize, end: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    for piece in pieces {
        let lo = start.max(at);
        let hi = end.min(at + piece.len());
        if hi > lo {
            out.push(piece[lo - at..hi - at].to_string());
        }
        at += piece.len();
    }
    out
}

/// Where the earliest stop sequence begins (and which), or `(text.len(), None)`.
fn first_stop(text: &str, stops: &[String]) -> (usize, Option<String>) {
    let mut best = text.len();
    let mut which = None;
    for seq in stops {
        if let Some(at) = text.find(seq.as_str()) {
            if at < best {
                best = at;
                which = Some(seq.clone());
            }
        }
    }
    (best, which)
}

/// The tools a request offered, as a registry the call parser reads arguments against.
pub fn offered_toolbox(tools: &[OfferedTool]) -> ToolBox {
    let mut box_ = ToolBox::new();
    for tool in tools {
        let properties = match tool.parameters.get("properties") {
            Some(Json::Obj(pairs)) => pairs.clone(),
            _ => Vec::new(),
        };
        let required: Vec<String> = tool.parameters.at("required").to_strings();
        let params: Vec<Param> = properties
            .iter()
            .map(|(name, spec)| {
                let kind = match spec.at("type").as_str() {
                    Some(kind) if PARAM_TYPES.contains(&kind) => kind,
                    _ => "string",
                };
                let description = spec.at("description").as_str().unwrap_or("");
                let mut param = Param::new(name, kind, description);
                param.required = required.contains(name);
                param
            })
            .collect();
        let _ = box_.register(Tool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            params,
            handler: None,
            network: false,
        });
    }
    box_
}

/// Answer `ask` with `model`, streaming every event to `emit` as it happens,
/// and return the whole reply (Python's `respond`).
///
/// `negative` is the guard's other half: with one, the negative network
/// vetoes a candidate before it is spoken and every veto is a line of the
/// thinking with the reason.  The events, in order, per choice: `start`;
/// `thinking` deltas; `text` deltas (one per node of the walk); `tool_use`;
/// `done` with the stop reason and the turn record.  After the last choice one
/// `end` carries the usage of the whole reply.
pub fn respond(
    model: &mut Model,
    negative: Option<&mut Model>,
    config: &FilterConfig,
    ask: &Ask,
    name: Option<&str>,
    emit: &mut dyn FnMut(&Json),
) -> Result<Reply, String> {
    ask.validate().map_err(|e| e.message)?;
    let enc = model.encoding();
    let mut negative = negative;
    let mut reply = Reply {
        token: token(),
        model: name.map(str::to_string).unwrap_or_else(|| model_id(model)),
        created: now(),
        kind: model.kind().to_string(),
        units: enc.units_name().to_string(),
        input_units: input_units(&enc, ask),
        choices: Vec::new(),
        thinking: ask.thinking,
    };
    let said: Vec<String> = ask.messages.iter().map(|m| m.text.clone()).collect();
    let mut heard = Heard::new(&said);
    let previous = ask.previous().to_string();
    let mut rng = ask.seed.map(Mt19937::new);
    let offered: Vec<String> = ask.tools.iter().map(|t| t.name.clone()).collect();
    let toolbox = if ask.tools.is_empty() {
        None
    } else {
        Some(offered_toolbox(&ask.tools))
    };
    let earlier = ask.messages.len() - 1;
    for index in 0..ask.n {
        emit(&Json::obj([
            ("type", Json::str("start")),
            ("index", Json::Int(index as i64)),
            ("id", Json::str(reply.token.clone())),
            ("model", Json::str(reply.model.clone())),
            ("created", Json::Int(reply.created)),
            ("input_units", Json::Int(reply.input_units as i64)),
            ("thinking", Json::Bool(ask.thinking)),
        ]));
        let narrator = RefCell::new(Narrator {
            index,
            lines: Vec::new(),
            emit,
        });
        {
            let mut n = narrator.borrow_mut();
            let mut opening = if ask.prefill() {
                format!("continuing its own last line {}", q(&previous))
            } else {
                format!("answering {}", q(&previous))
            };
            if earlier > 0 {
                opening.push_str(&format!(" ({} heard)", count(earlier, "earlier line")));
            }
            n.line(opening);
            if !ask.system.trim().is_empty() {
                n.line(
                    "a system prompt was given; the network continues text and cannot follow instructions, so it is not read"
                        .to_string(),
                );
            }
            if !ask.tools.is_empty() {
                n.line(format!(
                    "{} offered ({}); a reply that writes one comes back as a tool call",
                    count(ask.tools.len(), "tool"),
                    offered.join(", ")
                ));
            }
        }
        let verdicts: RefCell<Vec<FilterVerdict>> = RefCell::new(Vec::new());
        let seen: RefCell<Vec<(String, bool)>> = RefCell::new(Vec::new());
        let turn = {
            let guarding = negative.is_some();
            let mut neg = negative.as_deref_mut();
            let mut veto = |positive: &mut Model, text: &str| -> bool {
                // a candidate offered again after a shorter context is judged once
                if let Some((_, refused)) = seen.borrow().iter().find(|(t, _)| t == text) {
                    return *refused;
                }
                let Some(negative) = neg.as_deref_mut() else {
                    return false;
                };
                let mut pair = Filter {
                    positive,
                    negative,
                    config: config.clone(),
                };
                let verdict = pair.judge(text);
                let refused = verdict.decision == "reject";
                seen.borrow_mut().push((text.to_string(), refused));
                if refused {
                    narrator
                        .borrow_mut()
                        .line(format!("the negative network vetoed {}: {}", q(text), verdict.why));
                }
                verdicts.borrow_mut().push(verdict);
                refused
            };
            let mut trace = |event: &Json| narrator.borrow_mut().trace(event);
            let options = ReplyOptions {
                heard: &heard,
                index: earlier + 1,
                speaker: "assistant",
                mode: &ask.mode,
                max_length: ask.max_tokens,
                context: ask.context,
                temperature: ask.temperature,
                k: ask.k,
                beam: ask.beam,
                step_penalty: ask.step_penalty,
                avoid_repeats: ask.avoid_repeats,
                avoid_word_repeats: ask.avoid_word_repeats,
                explore: ask.explore,
                learn: ask.learn,
                think: true,
                think_depth: crate::thinking::THINK_DEPTH,
                veto: if guarding {
                    Some(&mut veto as &mut crate::dialogue::Veto)
                } else {
                    None
                },
                trace: Some(&mut trace as &mut crate::dialogue::Trace),
            };
            reply_or_err(model, &previous, options, &mut rng)?
        };
        let verdicts = verdicts.into_inner();
        let vetoed_any = seen.borrow().iter().any(|(_, refused)| *refused);
        let guard = match negative.as_deref_mut() {
            Some(neg) => {
                let refused: Vec<String> = verdicts
                    .iter()
                    .filter(|v| v.decision == "reject")
                    .map(|v| v.text.clone())
                    .collect();
                let mut pair = Filter {
                    positive: model,
                    negative: neg,
                    config: config.clone(),
                };
                pair.learn(&refused, "vetoed in conversation")?;
                guard_report(&pair, &verdicts, Vec::new())
            }
            None => Json::Null,
        };
        let mut choice = Choice {
            index,
            thinking: String::new(),
            text: String::new(),
            tool_calls: Vec::new(),
            stop_reason: END_TURN.to_string(),
            stop_sequence: None,
            turn: None,
            guard,
            output_units: 0,
            thinking_units: 0,
        };
        let mut n = narrator.into_inner();
        match turn {
            None => {
                if vetoed_any {
                    n.line("nothing to say: the guard vetoed everything it could say".to_string());
                    choice.stop_reason = REFUSAL.to_string();
                } else {
                    n.line("nothing to say: the graph has no way on from here".to_string());
                }
            }
            Some(turn) => {
                let ending = if turn.reached_end {
                    "reached the end of a text".to_string()
                } else {
                    format!("cut at {} {}", ask.max_tokens, enc.units_name())
                };
                let mut said = format!(
                    "saying {}: cost {:.4}, probability {:.4}, {ending}",
                    q(&turn.text),
                    turn.cost,
                    turn.probability
                );
                if turn.repeat {
                    said.push_str("; every path repeated something, so this is a repeat");
                }
                n.line(said);
                let whole = turn.text.clone();
                let pieces = deltas(&enc, &turn.labels, &turn.node_ids, &whole);
                let start = if ask.prefill() && !turn.context.is_empty() {
                    turn.context.len()
                } else {
                    0
                };
                let mut end = whole.len();
                choice.stop_reason = if turn.reached_end { END_TURN } else { MAX_TOKENS }.to_string();
                let spoken = &whole[start..];
                let (cut, seq) = first_stop(spoken, &ask.stop);
                if let Some(seq) = seq {
                    end = start + cut;
                    choice.stop_reason = STOP_SEQUENCE.to_string();
                    n.line(format!("stopped at the stop sequence {}", q(&seq)));
                    choice.stop_sequence = Some(seq);
                }
                let mut call = parse_call(&whole[start..end], None);
                if let Some(found) = call.as_ref() {
                    if offered.contains(&found.name) {
                        call = parse_call(&whole[start..end], toolbox.as_ref());
                        // the offered tool's schema reads the arguments
                    }
                }
                if let Some(call) = call {
                    if call.name.is_empty() {
                        n.line(format!(
                            "wrote a tool call it could not finish ({}); it stays text",
                            call.error.clone().unwrap_or_default()
                        ));
                    } else if !offered.contains(&call.name) {
                        if offered.is_empty() {
                            n.line(format!(
                                "wrote a call to {}, but no tools were offered; it stays text",
                                call.name
                            ));
                        } else {
                            n.line(format!(
                                "wrote a call to {}, which was not offered; it stays text",
                                call.name
                            ));
                        }
                    } else if let Some(error) = call.error.as_ref() {
                        n.line(format!(
                            "wrote a call to {} it could not finish ({error}); it stays text",
                            call.name
                        ));
                    } else {
                        n.line(format!(
                            "wrote a tool call: {}",
                            call_text(&call.name, &call.arguments).trim()
                        ));
                        end = start + call.span.0;
                        choice.tool_calls.push(ToolCallOut {
                            token: token(),
                            name: call.name.clone(),
                            input: call.arguments.clone(),
                        });
                        choice.stop_reason = TOOL_USE.to_string();
                        choice.stop_sequence = None;
                    }
                }
                choice.text = whole[start..end].to_string();
                for piece in windowed(&pieces, start, end) {
                    (n.emit)(&Json::obj([
                        ("type", Json::str("text")),
                        ("index", Json::Int(index as i64)),
                        ("text", Json::str(piece)),
                    ]));
                }
                for call in &choice.tool_calls {
                    (n.emit)(&Json::obj([
                        ("type", Json::str("tool_use")),
                        ("index", Json::Int(index as i64)),
                        ("id", Json::str(call.token.clone())),
                        ("name", Json::str(call.name.clone())),
                        ("input", call.input.clone()),
                    ]));
                }
                heard.remember(&turn.text, if turn.context.is_empty() { "" } else { &turn.reply });
                choice.turn = Some(turn);
            }
        }
        choice.thinking = n.lines.join("\n");
        choice.thinking_units = enc.len(&choice.thinking);
        choice.output_units = enc.len(&choice.text) + choice.thinking_units;
        let done = Json::obj([
            ("type", Json::str("done")),
            ("index", Json::Int(index as i64)),
            ("stop_reason", Json::str(choice.stop_reason.clone())),
            (
                "stop_sequence",
                choice.stop_sequence.clone().map(Json::Str).unwrap_or(Json::Null),
            ),
            ("output_units", Json::Int(choice.output_units as i64)),
            ("thinking_units", Json::Int(choice.thinking_units as i64)),
            ("turn", choice.turn.as_ref().map(|t| t.to_json()).unwrap_or(Json::Null)),
            ("guard", choice.guard.clone()),
        ]);
        (n.emit)(&done);
        reply.choices.push(choice);
    }
    emit(&Json::obj([("type", Json::str("end")), ("usage", usage_of(&reply))]));
    Ok(reply)
}

fn reply_or_err(
    model: &mut Model,
    previous: &str,
    options: ReplyOptions,
    rng: &mut Option<Mt19937>,
) -> Result<Option<Turn>, String> {
    reply(model, None, previous, options, rng)
}

// -- the two dialects -----------------------------------------------------------------------------

fn openai_tool_call(call: &ToolCallOut) -> Json {
    Json::obj([
        ("id", Json::str(format!("call_{}", call.token))),
        ("type", Json::str("function")),
        (
            "function",
            Json::obj([
                ("name", Json::str(call.name.clone())),
                ("arguments", Json::str(crate::tools::dumps(&call.input, true, false))),
            ]),
        ),
    ])
}

/// The model's own account of the answer, beside the standard fields.
fn record(reply: &Reply) -> Json {
    Json::obj([
        ("kind", Json::str(reply.kind.clone())),
        ("units", Json::str(reply.units.clone())),
        (
            "choices",
            Json::Arr(
                reply
                    .choices
                    .iter()
                    .map(|c| {
                        Json::obj([
                            ("index", Json::Int(c.index as i64)),
                            ("stop_reason", Json::str(c.stop_reason.clone())),
                            ("turn", c.turn.as_ref().map(|t| t.to_json()).unwrap_or(Json::Null)),
                            ("guard", c.guard.clone()),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

fn openai_usage(usage: &Json) -> Json {
    Json::obj([
        ("prompt_tokens", usage.at("input").clone()),
        ("completion_tokens", usage.at("output").clone()),
        ("total_tokens", usage.at("total").clone()),
        (
            "completion_tokens_details",
            Json::obj([("reasoning_tokens", usage.at("thinking").clone())]),
        ),
    ])
}

/// The reply as a Chat Completions `chat.completion` object.
pub fn to_openai(reply: &Reply) -> Json {
    let choices: Vec<Json> = reply
        .choices
        .iter()
        .map(|c| {
            let mut message = vec![
                ("role".to_string(), Json::str("assistant")),
                (
                    "content".to_string(),
                    if c.text.is_empty() && !c.tool_calls.is_empty() {
                        Json::Null
                    } else {
                        Json::str(c.text.clone())
                    },
                ),
            ];
            if reply.thinking {
                message.push(("reasoning_content".to_string(), Json::str(c.thinking.clone())));
            }
            if !c.tool_calls.is_empty() {
                message.push((
                    "tool_calls".to_string(),
                    Json::Arr(c.tool_calls.iter().map(openai_tool_call).collect()),
                ));
            }
            Json::obj([
                ("index", Json::Int(c.index as i64)),
                ("message", Json::Obj(message)),
                ("logprobs", Json::Null),
                ("finish_reason", Json::str(finish_reason(&c.stop_reason))),
            ])
        })
        .collect();
    Json::obj([
        ("id", Json::str(format!("chatcmpl-{}", reply.token))),
        ("object", Json::str("chat.completion")),
        ("created", Json::Int(reply.created)),
        ("model", Json::str(reply.model.clone())),
        ("choices", Json::Arr(choices)),
        ("usage", openai_usage(&usage_of(reply))),
        ("radixnet", record(reply)),
    ])
}

fn anthropic_content(c: &Choice, thinking: bool) -> Json {
    let mut blocks = Vec::new();
    if thinking {
        blocks.push(Json::obj([
            ("type", Json::str("thinking")),
            ("thinking", Json::str(c.thinking.clone())),
            ("signature", Json::str("")),
        ]));
    }
    if !c.text.is_empty() || c.tool_calls.is_empty() {
        blocks.push(Json::obj([
            ("type", Json::str("text")),
            ("text", Json::str(c.text.clone())),
        ]));
    }
    for call in &c.tool_calls {
        blocks.push(Json::obj([
            ("type", Json::str("tool_use")),
            ("id", Json::str(format!("toolu_{}", call.token))),
            ("name", Json::str(call.name.clone())),
            ("input", call.input.clone()),
        ]));
    }
    Json::Arr(blocks)
}

/// The reply as a Messages `message` object (the first choice is the answer).
pub fn to_anthropic(reply: &Reply) -> Json {
    let empty = Choice {
        index: 0,
        thinking: String::new(),
        text: String::new(),
        tool_calls: Vec::new(),
        stop_reason: END_TURN.to_string(),
        stop_sequence: None,
        turn: None,
        guard: Json::Null,
        output_units: 0,
        thinking_units: 0,
    };
    let c = reply.choices.first().unwrap_or(&empty);
    Json::obj([
        ("id", Json::str(format!("msg_{}", reply.token))),
        ("type", Json::str("message")),
        ("role", Json::str("assistant")),
        ("model", Json::str(reply.model.clone())),
        ("content", anthropic_content(c, reply.thinking)),
        ("stop_reason", Json::str(c.stop_reason.clone())),
        (
            "stop_sequence",
            c.stop_sequence.clone().map(Json::Str).unwrap_or(Json::Null),
        ),
        (
            "usage",
            Json::obj([
                ("input_tokens", Json::Int(reply.input_units as i64)),
                ("output_tokens", Json::Int(c.output_units as i64)),
            ]),
        ),
        ("radixnet", record(reply)),
    ])
}

/// [`to_openai`] or [`to_anthropic`], by dialect.
pub fn to_format(reply: &Reply, dialect: &str) -> Json {
    if dialect == ANTHROPIC {
        to_anthropic(reply)
    } else {
        to_openai(reply)
    }
}

/// One server-sent event: `event: name` (when given), then `data: <json>` and a blank line.
pub fn sse(data: &Json, event: Option<&str>) -> String {
    sse_text(&data.render(0), event)
}

fn sse_text(data: &str, event: Option<&str>) -> String {
    match event {
        Some(name) => format!("event: {name}\ndata: {data}\n\n"),
        None => format!("data: {data}\n\n"),
    }
}

/// The frame a failure mid-stream is reported with, in Chat Completions' shape.
pub fn openai_error(message: &str) -> String {
    sse(
        &Json::obj([(
            "error",
            Json::obj([
                ("message", Json::str(message)),
                ("type", Json::str("server_error")),
                ("param", Json::Null),
                ("code", Json::Null),
            ]),
        )]),
        None,
    )
}

/// The frame a failure mid-stream is reported with, in Messages' shape.
pub fn anthropic_error(message: &str) -> String {
    sse(
        &Json::obj([
            ("type", Json::str("error")),
            (
                "error",
                Json::obj([("type", Json::str("api_error")), ("message", Json::str(message))]),
            ),
        ]),
        Some("error"),
    )
}

/// Renders [`respond`]'s events as Chat Completions chunks.
pub struct OpenAiStream {
    pub token: String,
    pub model: String,
    pub created: i64,
    pub include_usage: bool,
    pub thinking: bool,
}

impl OpenAiStream {
    fn chunk(&self, choices: Vec<Json>, extra: Vec<(&str, Json)>) -> String {
        let mut pairs = vec![
            ("id".to_string(), Json::str(format!("chatcmpl-{}", self.token))),
            ("object".to_string(), Json::str("chat.completion.chunk")),
            ("created".to_string(), Json::Int(self.created)),
            ("model".to_string(), Json::str(self.model.clone())),
            ("choices".to_string(), Json::Arr(choices)),
        ];
        for (key, value) in extra {
            pairs.push((key.to_string(), value));
        }
        sse(&Json::Obj(pairs), None)
    }

    fn choice(index: i64, delta: Json, finish: Json) -> Json {
        Json::obj([("index", Json::Int(index)), ("delta", delta), ("finish_reason", finish)])
    }

    /// The frames an event becomes.
    pub fn frames(&mut self, event: &Json) -> Vec<String> {
        let index = event.at("index").as_i64().unwrap_or(0);
        match event.at("type").as_str().unwrap_or("") {
            "start" => vec![self.chunk(
                vec![Self::choice(
                    index,
                    Json::obj([("role", Json::str("assistant")), ("content", Json::str(""))]),
                    Json::Null,
                )],
                Vec::new(),
            )],
            "thinking" if self.thinking => vec![self.chunk(
                vec![Self::choice(
                    index,
                    Json::obj([("reasoning_content", event.at("text").clone())]),
                    Json::Null,
                )],
                Vec::new(),
            )],
            "text" => vec![self.chunk(
                vec![Self::choice(
                    index,
                    Json::obj([("content", event.at("text").clone())]),
                    Json::Null,
                )],
                Vec::new(),
            )],
            "tool_use" => {
                let call = Json::obj([
                    ("index", Json::Int(0)),
                    (
                        "id",
                        Json::str(format!("call_{}", event.at("id").as_str().unwrap_or(""))),
                    ),
                    ("type", Json::str("function")),
                    (
                        "function",
                        Json::obj([
                            ("name", event.at("name").clone()),
                            (
                                "arguments",
                                Json::str(crate::tools::dumps(event.at("input"), true, false)),
                            ),
                        ]),
                    ),
                ]);
                vec![self.chunk(
                    vec![Self::choice(
                        index,
                        Json::obj([("tool_calls", Json::Arr(vec![call]))]),
                        Json::Null,
                    )],
                    Vec::new(),
                )]
            }
            "done" => {
                let stop = event.at("stop_reason").as_str().unwrap_or(END_TURN);
                vec![self.chunk(
                    vec![Self::choice(
                        index,
                        Json::Obj(Vec::new()),
                        Json::str(finish_reason(stop)),
                    )],
                    vec![(
                        "radixnet",
                        Json::obj([
                            ("stop_reason", Json::str(stop)),
                            ("turn", event.at("turn").clone()),
                            ("guard", event.at("guard").clone()),
                        ]),
                    )],
                )]
            }
            "end" => {
                let mut out = Vec::new();
                if self.include_usage {
                    out.push(self.chunk(Vec::new(), vec![("usage", openai_usage(event.at("usage")))]));
                }
                out.push(sse_text("[DONE]", None));
                out
            }
            _ => Vec::new(),
        }
    }
}

/// Renders [`respond`]'s events as Messages events, one content block at a time.
pub struct AnthropicStream {
    pub token: String,
    pub model: String,
    pub created: i64,
    pub input_units: usize,
    pub thinking: bool,
    block: i64,
    open: Option<&'static str>,
    answered: bool,
}

impl AnthropicStream {
    pub fn new(token: &str, model: &str, created: i64, input_units: usize, thinking: bool) -> AnthropicStream {
        AnthropicStream {
            token: token.to_string(),
            model: model.to_string(),
            created,
            input_units,
            thinking,
            block: -1,
            open: None,
            answered: false,
        }
    }

    fn delta(&self, delta: Json) -> String {
        sse(
            &Json::obj([
                ("type", Json::str("content_block_delta")),
                ("index", Json::Int(self.block)),
                ("delta", delta),
            ]),
            Some("content_block_delta"),
        )
    }

    fn open_block(&mut self, kind: &'static str, block: Json) -> Vec<String> {
        let mut out = self.close();
        self.block += 1;
        self.open = Some(kind);
        if kind != "thinking" {
            self.answered = true;
        }
        out.push(sse(
            &Json::obj([
                ("type", Json::str("content_block_start")),
                ("index", Json::Int(self.block)),
                ("content_block", block),
            ]),
            Some("content_block_start"),
        ));
        out
    }

    fn close(&mut self) -> Vec<String> {
        let Some(open) = self.open else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if open == "thinking" {
            out.push(self.delta(Json::obj([
                ("type", Json::str("signature_delta")),
                ("signature", Json::str("")),
            ])));
        }
        out.push(sse(
            &Json::obj([
                ("type", Json::str("content_block_stop")),
                ("index", Json::Int(self.block)),
            ]),
            Some("content_block_stop"),
        ));
        self.open = None;
        out
    }

    /// The frames an event becomes.
    pub fn frames(&mut self, event: &Json) -> Vec<String> {
        let kind = event.at("type").as_str().unwrap_or("");
        if kind == "end" {
            return Vec::new();
        }
        if event.at("index").as_i64().unwrap_or(0) != 0 {
            return Vec::new(); // a Messages reply is one message: the first choice
        }
        match kind {
            "start" => {
                let message = Json::obj([
                    ("id", Json::str(format!("msg_{}", self.token))),
                    ("type", Json::str("message")),
                    ("role", Json::str("assistant")),
                    ("model", Json::str(self.model.clone())),
                    ("content", Json::Arr(Vec::new())),
                    ("stop_reason", Json::Null),
                    ("stop_sequence", Json::Null),
                    (
                        "usage",
                        Json::obj([
                            ("input_tokens", Json::Int(self.input_units as i64)),
                            ("output_tokens", Json::Int(0)),
                        ]),
                    ),
                ]);
                vec![sse(
                    &Json::obj([("type", Json::str("message_start")), ("message", message)]),
                    Some("message_start"),
                )]
            }
            "thinking" => {
                if !self.thinking {
                    return Vec::new();
                }
                let mut out = if self.open != Some("thinking") {
                    self.open_block(
                        "thinking",
                        Json::obj([
                            ("type", Json::str("thinking")),
                            ("thinking", Json::str("")),
                            ("signature", Json::str("")),
                        ]),
                    )
                } else {
                    Vec::new()
                };
                out.push(self.delta(Json::obj([
                    ("type", Json::str("thinking_delta")),
                    ("thinking", event.at("text").clone()),
                ])));
                out
            }
            "text" => {
                let mut out = if self.open != Some("text") {
                    self.open_block(
                        "text",
                        Json::obj([("type", Json::str("text")), ("text", Json::str(""))]),
                    )
                } else {
                    Vec::new()
                };
                out.push(self.delta(Json::obj([
                    ("type", Json::str("text_delta")),
                    ("text", event.at("text").clone()),
                ])));
                out
            }
            "tool_use" => {
                let mut out = self.open_block(
                    "tool_use",
                    Json::obj([
                        ("type", Json::str("tool_use")),
                        (
                            "id",
                            Json::str(format!("toolu_{}", event.at("id").as_str().unwrap_or(""))),
                        ),
                        ("name", event.at("name").clone()),
                        ("input", Json::Obj(Vec::new())),
                    ]),
                );
                out.push(self.delta(Json::obj([
                    ("type", Json::str("input_json_delta")),
                    (
                        "partial_json",
                        Json::str(crate::tools::dumps(event.at("input"), true, false)),
                    ),
                ])));
                out.extend(self.close());
                out
            }
            "done" => {
                let mut out = Vec::new();
                if !self.answered {
                    // a reply with nothing to say is still one (empty) text block
                    out.extend(self.open_block(
                        "text",
                        Json::obj([("type", Json::str("text")), ("text", Json::str(""))]),
                    ));
                }
                out.extend(self.close());
                out.push(sse(
                    &Json::obj([
                        ("type", Json::str("message_delta")),
                        (
                            "delta",
                            Json::obj([
                                ("stop_reason", event.at("stop_reason").clone()),
                                ("stop_sequence", event.at("stop_sequence").clone()),
                            ]),
                        ),
                        (
                            "usage",
                            Json::obj([("output_tokens", event.at("output_units").clone())]),
                        ),
                        (
                            "radixnet",
                            Json::obj([("turn", event.at("turn").clone()), ("guard", event.at("guard").clone())]),
                        ),
                    ]),
                    Some("message_delta"),
                ));
                out.push(sse(
                    &Json::obj([("type", Json::str("message_stop"))]),
                    Some("message_stop"),
                ));
                out
            }
            _ => Vec::new(),
        }
    }
}

// -- the routes -----------------------------------------------------------------------------------

/// An error as a `/v1` client expects it: the dialect's own envelope (Python's `_shape_error`).
pub fn shape_error(dialect: &str, status: u16, message: &str, param: Option<&str>) -> Json {
    if dialect == OPENAI {
        let kind = if status >= 500 {
            "server_error"
        } else {
            "invalid_request_error"
        };
        let code = if status == 404 && param == Some("model") {
            Json::str("model_not_found")
        } else {
            Json::Null
        };
        return Json::obj([(
            "error",
            Json::obj([
                ("message", Json::str(message)),
                ("type", Json::str(kind)),
                ("param", param.map(Json::str).unwrap_or(Json::Null)),
                ("code", code),
            ]),
        )]);
    }
    let kind = if status >= 500 {
        "api_error"
    } else if status == 404 {
        "not_found_error"
    } else if status == 413 {
        "request_too_large"
    } else {
        "invalid_request_error"
    };
    Json::obj([
        ("type", Json::str("error")),
        (
            "error",
            Json::obj([("type", Json::str(kind)), ("message", Json::str(message))]),
        ),
    ])
}

/// Which dialect a `/v1` path speaks.
pub fn dialect_of(path: &str) -> &'static str {
    if path.starts_with("/v1/messages") {
        ANTHROPIC
    } else {
        OPENAI
    }
}

/// Runs `f` on the model a request names, with the negative network beside it
/// when the guard is on and has something to veto with.  The negative network
/// is loaded before the model is locked, never inside (the lock rule of
/// `duo.rs`); the guard stands down when the running model is the negative
/// network, which cannot filter itself.
fn with_nets<T>(
    svc: &Service,
    ask: &Ask,
    f: impl FnOnce(&mut Model, Option<&mut Model>, &FilterConfig) -> T,
) -> Result<T, ApiError> {
    let config = svc.guard_config();
    let guarding = ask.guard && svc.guard_ready();
    if guarding {
        svc.ensure_negative()?;
    }
    svc.with_voice(&ask.model, |model| {
        if guarding {
            svc.with_negative(|negative| f(model, Some(negative), &config))
        } else {
            Ok(f(model, None, &config))
        }
    })?
}

/// The whole reply at once.
fn answer(svc: &Service, ask: &Ask) -> Result<Reply, ApiError> {
    with_nets(svc, ask, |model, negative, config| {
        let name = model_id(model);
        let mut quiet = |_: &Json| {};
        respond(model, negative, config, ask, Some(&name), &mut quiet)
    })?
    .map_err(ApiError::bad_request)
}

/// The reply as frames, written from inside the search.
fn stream_reply(svc: &Service, ask: &Ask, dialect: &str, write: &mut crate::http::FrameSink) -> Result<(), ApiError> {
    let outcome = with_nets(svc, ask, |model, negative, config| {
        let name = model_id(model);
        let mut openai: Option<OpenAiStream> = None;
        let mut anthropic: Option<AnthropicStream> = None;
        let mut failed: Option<std::io::Error> = None;
        let include_usage = ask.include_usage;
        let thinking = ask.thinking;
        let mut emit = |event: &Json| {
            if failed.is_some() {
                return;
            }
            if event.at("type").as_str() == Some("start") && openai.is_none() && anthropic.is_none() {
                let token = event.at("id").as_str().unwrap_or("").to_string();
                let model = event.at("model").as_str().unwrap_or("").to_string();
                let created = event.at("created").as_i64().unwrap_or(0);
                if dialect == OPENAI {
                    openai = Some(OpenAiStream {
                        token,
                        model,
                        created,
                        include_usage,
                        thinking,
                    });
                } else {
                    let input_units = event.at("input_units").as_i64().unwrap_or(0) as usize;
                    anthropic = Some(AnthropicStream::new(&token, &model, created, input_units, thinking));
                }
            }
            let frames = match (openai.as_mut(), anthropic.as_mut()) {
                (Some(stream), _) => stream.frames(event),
                (_, Some(stream)) => stream.frames(event),
                _ => Vec::new(),
            };
            for frame in frames {
                if let Err(err) = write(&frame) {
                    failed = Some(err);
                    return;
                }
            }
        };
        let result = respond(model, negative, config, ask, Some(&name), &mut emit);
        (result, failed)
    })?;
    let (result, failed) = outcome;
    result.map_err(ApiError::bad_request)?;
    match failed {
        Some(err) => Err(ApiError::with_status(500, err.to_string())),
        None => Ok(()),
    }
}

fn ask_of(r: &Request, dialect: &str) -> Result<Ask, Json> {
    parse_request(&r.body, dialect).map_err(|e| shape_error(dialect, 400, &e.message, e.param.as_deref()))
}

/// A service error as the dialect's document: a model that is not here names the `model` field.
fn shaped(dialect: &str, err: &ApiError) -> Json {
    let param = if err.status == 404 { Some("model") } else { None };
    shape_error(dialect, err.status, &err.message, param)
}

fn talk_route(svc: &Arc<Service>, r: &Request, dialect: &'static str) -> Result<Streamed, ApiError> {
    let ask = match ask_of(r, dialect) {
        Ok(ask) => ask,
        Err(doc) => return Ok(Streamed::Document(400, doc)),
    };
    if !ask.stream {
        return Ok(match answer(svc, &ask) {
            Ok(reply) => Streamed::Document(200, to_format(&reply, dialect)),
            Err(err) => Streamed::Document(err.status, shaped(dialect, &err)),
        });
    }
    // a model that is not here is a 404 before any frame goes out
    if let Err(err) = svc.with_voice(&ask.model, |_| ()) {
        return Ok(Streamed::Document(err.status, shaped(dialect, &err)));
    }
    let svc = Arc::clone(svc);
    let error: fn(&str) -> String = if dialect == OPENAI {
        openai_error
    } else {
        anthropic_error
    };
    Ok(Streamed::Events {
        run: Box::new(move |write| stream_reply(&svc, &ask, dialect, write)),
        error,
    })
}

fn chat_completions(svc: &Arc<Service>, r: &Request) -> Result<Streamed, ApiError> {
    talk_route(svc, r, OPENAI)
}

fn messages_route(svc: &Arc<Service>, r: &Request) -> Result<Streamed, ApiError> {
    talk_route(svc, r, ANTHROPIC)
}

fn count_tokens(svc: &Arc<Service>, r: &Request) -> Answer {
    let ask = match ask_of(r, ANTHROPIC) {
        Ok(ask) => ask,
        Err(doc) => return Ok(crate::http::answer_with(400, doc)),
    };
    match svc.with_voice(&ask.model, |model| input_units(&model.encoding(), &ask)) {
        Ok(units) => Ok(Json::obj([("input_tokens", Json::Int(units as i64))])),
        Err(err) => Ok(crate::http::answer_with(err.status, shaped(ANTHROPIC, &err))),
    }
}

fn models(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(svc.models_json())
}

/// `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/messages/count_tokens`, `GET /v1/models`.
pub fn routes(server: &mut Server<Service>) {
    server.route_stream("POST", "/v1/chat/completions", chat_completions);
    server.route_stream("POST", "/v1/messages", messages_route);
    server.route("POST", "/v1/messages/count_tokens", count_tokens);
    server.route("GET", "/v1/models", models);
}

// -- the CLI --------------------------------------------------------------------------------------

/// A request in `dialect`'s shape from the `talk` flags (Python's `_talk_body`).
fn talk_body(ctx: &Ctx, dialect: &str, history: &[Json], said: &str) -> Result<Json, String> {
    let args = &ctx.args;
    let mut messages: Vec<Json> = history.to_vec();
    messages.push(Json::obj([("role", Json::str("user")), ("content", Json::str(said))]));
    let system = args.str("system", "");
    let mut body = vec![
        (
            "max_tokens".to_string(),
            Json::Int(args.usize("max-tokens", DEFAULT_MAX_TOKENS)? as i64),
        ),
        ("temperature".to_string(), Json::Num(args.float("temperature", 1.0)?)),
        ("mode".to_string(), Json::str(args.str("mode", "beam"))),
        ("context".to_string(), Json::Int(args.usize("context", 12)? as i64)),
        ("k".to_string(), Json::Int(args.usize("k", 5)? as i64)),
        ("step_penalty".to_string(), Json::Num(args.float("step-penalty", 0.0)?)),
        ("explore".to_string(), Json::Int(args.usize("explore", EXPLORE)? as i64)),
        ("avoid_repeats".to_string(), Json::Bool(!args.on("allow-repeats"))),
        (
            "avoid_word_repeats".to_string(),
            Json::Bool(!args.on("allow-word-repeats")),
        ),
        ("learn".to_string(), Json::Bool(!args.on("no-learn"))),
        ("guard".to_string(), Json::Bool(!args.on("no-guard"))),
    ];
    let beam = args.usize("beam", 0)?;
    if beam > 0 {
        body.push(("beam".to_string(), Json::Int(beam as i64)));
    }
    if args.get("seed").is_some() {
        body.push(("seed".to_string(), Json::Int(ctx.seed)));
    }
    let stops = args.all("stop");
    let tools = args.all("tool");
    let no_thinking = args.on("no-thinking");
    if dialect == ANTHROPIC {
        if !system.is_empty() {
            body.push(("system".to_string(), Json::str(system)));
        }
        if !stops.is_empty() {
            body.push(("stop_sequences".to_string(), Json::strs(stops)));
        }
        body.push((
            "thinking".to_string(),
            Json::obj([("type", Json::str(if no_thinking { "disabled" } else { "enabled" }))]),
        ));
        if !tools.is_empty() {
            body.push((
                "tools".to_string(),
                Json::Arr(
                    tools
                        .iter()
                        .map(|name| {
                            Json::obj([
                                ("name", Json::str(name.clone())),
                                ("input_schema", Json::obj([("type", Json::str("object"))])),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
    } else {
        if !system.is_empty() {
            messages.insert(
                0,
                Json::obj([("role", Json::str("system")), ("content", Json::str(system))]),
            );
        }
        if !stops.is_empty() {
            body.push(("stop".to_string(), Json::strs(stops)));
        }
        let n = args.usize("n", 1)?;
        if n > 1 {
            body.push(("n".to_string(), Json::Int(n as i64)));
        }
        body.push(("thinking".to_string(), Json::Bool(!no_thinking)));
        if !tools.is_empty() {
            body.push((
                "tools".to_string(),
                Json::Arr(
                    tools
                        .iter()
                        .map(|name| {
                            Json::obj([
                                ("type", Json::str("function")),
                                (
                                    "function",
                                    Json::obj([
                                        ("name", Json::str(name.clone())),
                                        ("parameters", Json::obj([("type", Json::str("object"))])),
                                    ]),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
    }
    body.insert(0, ("messages".to_string(), Json::Arr(messages)));
    Ok(Json::Obj(body))
}

/// The reply's text, to carry the conversation on with.
fn spoken_text(doc: &Json, dialect: &str) -> String {
    if dialect == ANTHROPIC {
        return doc
            .at("content")
            .as_array()
            .iter()
            .filter(|b| b.at("type").as_str() == Some("text"))
            .map(|b| b.at("text").as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("");
    }
    doc.at("choices")
        .as_array()
        .first()
        .and_then(|c| c.at("message").at("content").as_str())
        .unwrap_or("")
        .to_string()
}

/// `radixnet talk`: talk to the model in today's format - the thinking first,
/// then the text, streamed; `--json` prints the dialect's own document.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let dialect: &'static str = match args.str("format", OPENAI).to_lowercase().as_str() {
        "openai" => OPENAI,
        "anthropic" => ANTHROPIC,
        other => return Err(format!("--format must be openai or anthropic, got {other:?}")),
    };
    let mut model = ctx.open(true)?;
    let guard = ctx.open_guard()?;
    let (mut negative, config) = match guard {
        Some((negative, config)) => (Some(negative), config),
        None => (None, FilterConfig::default()),
    };
    let units = model.encoding().units_name();
    let json_mode = ctx.json;
    let mut taught: Vec<i64> = Vec::new();
    let mut exchanges: Vec<Json> = Vec::new();

    let mut one = |body: &Json, model: &mut Model, negative: Option<&mut Model>| -> Result<Json, String> {
        let ask = parse_request(body, dialect).map_err(|e| format!("bad request: {}", e.message))?;
        let mut writing = false;
        let mut out = std::io::stdout();
        let mut show = |event: &Json| {
            if json_mode {
                return;
            }
            match event.at("type").as_str().unwrap_or("") {
                "start" => {
                    if event.at("index").as_i64().unwrap_or(0) > 0 {
                        println!("--- choice {} ---", event.at("index").as_i64().unwrap_or(0) + 1);
                    }
                    writing = false;
                }
                "thinking" => {
                    println!(
                        "  · {}",
                        event.at("text").as_str().unwrap_or("").trim_start_matches('\n')
                    );
                }
                "text" => {
                    if !writing {
                        print!("model: ");
                        writing = true;
                    }
                    print!("{}", event.at("text").as_str().unwrap_or(""));
                    let _ = out.flush();
                }
                "tool_use" => {
                    if writing {
                        println!();
                        writing = false;
                    }
                    println!(
                        "tool call: {} {}",
                        event.at("name").as_str().unwrap_or(""),
                        crate::tools::dumps(event.at("input"), true, false)
                    );
                }
                "done" => {
                    if writing {
                        println!();
                    } else if event.at("turn").is_null() {
                        println!("model: (nothing to say)");
                    }
                    let said = event.at("output_units").as_i64().unwrap_or(0)
                        - event.at("thinking_units").as_i64().unwrap_or(0);
                    println!(
                        "    [{}; {said} {units} said, {} thought]",
                        event.at("stop_reason").as_str().unwrap_or(""),
                        event.at("thinking_units").as_i64().unwrap_or(0)
                    );
                }
                _ => {}
            }
        };
        let reply = respond(
            model,
            if ask.guard { negative } else { None },
            &config,
            &ask,
            None,
            &mut show,
        )?;
        for choice in &reply.choices {
            if let Some(turn) = &choice.turn {
                if let Some(rethink) = &turn.rethink {
                    if rethink.taught >= 0 && !taught.contains(&rethink.taught) {
                        taught.push(rethink.taught);
                    }
                }
            }
        }
        let doc = to_format(&reply, dialect);
        exchanges.push(doc.clone());
        Ok(doc)
    };

    let mut history: Vec<Json> = Vec::new();
    let mut last: Option<Json> = None;
    if let Some(path) = args.get("request") {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read the request {path}: {e}"))?;
        let request = parse(&text).map_err(|e| format!("cannot read the request {path}: {e}"))?;
        if !matches!(request, Json::Obj(_)) {
            return Err(format!(
                "{path} must hold a JSON object: a request in the {dialect} shape"
            ));
        }
        last = Some(one(&request, &mut model, negative.as_mut())?);
    } else if !args.all("message").is_empty() {
        for said in args.all("message") {
            if !json_mode {
                println!("you: {said}");
            }
            let body = talk_body(ctx, dialect, &history, &said)?;
            let doc = one(&body, &mut model, negative.as_mut())?;
            history.push(Json::obj([("role", Json::str("user")), ("content", Json::str(said))]));
            history.push(Json::obj([
                ("role", Json::str("assistant")),
                ("content", Json::str(spoken_text(&doc, dialect))),
            ]));
            last = Some(doc);
        }
    } else {
        let stdin = std::io::stdin();
        let interactive = stdin.is_terminal() && !json_mode;
        if interactive {
            println!(
                "talking to {} ({dialect} format); an empty line or Ctrl-D ends it",
                ctx.model_path
            );
        }
        loop {
            if interactive {
                print!("you> ");
                let _ = std::io::stdout().flush();
            }
            let mut line = String::new();
            let read = stdin.lock().read_line(&mut line).map_err(|e| e.to_string())?;
            if read == 0 || line.trim().is_empty() {
                break;
            }
            let said = line.trim_end_matches(['\n', '\r']).to_string();
            if !interactive && !json_mode {
                println!("you: {said}");
            }
            let body = talk_body(ctx, dialect, &history, &said)?;
            let doc = one(&body, &mut model, negative.as_mut())?;
            history.push(Json::obj([
                ("role", Json::str("user")),
                ("content", Json::str(said.clone())),
            ]));
            history.push(Json::obj([
                ("role", Json::str("assistant")),
                ("content", Json::str(spoken_text(&doc, dialect))),
            ]));
            last = Some(doc);
        }
        if last.is_none() {
            return Err("nothing was said: give --message TEXT, --request FILE, or type a line".to_string());
        }
    }
    let mut saved = Json::Null;
    if !taught.is_empty() && args.on("save") {
        let path = ctx.save(&mut model)?;
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        saved = Json::obj([("path", Json::str(path.clone())), ("bytes", Json::Int(bytes as i64))]);
        if !json_mode {
            println!("saved {path} (it learned to hand over at {} node(s))", taught.len());
        }
    } else if !taught.is_empty() && !json_mode {
        println!(
            "it learned to hand over at {} node(s); --save writes that into the model",
            taught.len()
        );
    }
    if !json_mode {
        return Ok(());
    }
    let mut doc = if exchanges.len() == 1 {
        match last {
            Some(Json::Obj(pairs)) => pairs,
            _ => Vec::new(),
        }
    } else {
        taught.sort_unstable();
        vec![
            ("format".to_string(), Json::str(dialect)),
            ("exchanges".to_string(), Json::Arr(exchanges)),
            ("taught".to_string(), Json::ints(taught.iter().copied())),
        ]
    };
    if !saved.is_null() {
        doc.push(("saved".to_string(), saved));
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TrainOptions;
    use crate::GraphOptions;

    fn trained() -> Model {
        let mut m = Model::new(1, GraphOptions::default()).unwrap();
        let texts: Vec<String> = [
            "the cat sat on the mat",
            "the mat was red and the cat was black",
            "a dog sat on the log",
            "the dog and the cat are friends",
            "the cat night",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        m.train(
            &texts,
            &TrainOptions {
                epochs: 2,
                ..Default::default()
            },
        )
        .unwrap();
        m
    }

    fn user(text: &str, extra: &str) -> Json {
        let body = format!(r#"{{"messages": [{{"role": "user", "content": "{text}"}}]{extra}}}"#);
        parse(&body).unwrap()
    }

    #[test]
    fn a_request_is_read_in_both_dialects() {
        let ask = parse_openai(&user(
            "hello",
            r#", "max_tokens": 30, "stop": "x", "n": 2, "mode": "sample", "seed": 3"#,
        ))
        .unwrap();
        assert_eq!(
            (ask.max_tokens, ask.n, ask.mode.as_str(), ask.seed),
            (30, 2, "sample", Some(3))
        );
        assert_eq!(ask.stop, vec!["x".to_string()]);
        let ask = parse_anthropic(&parse(
            r#"{"system": [{"type": "text", "text": "be brief"}], "messages": [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "yo"}]}],
                "thinking": {"type": "disabled"}, "stop_sequences": ["!"]}"#,
        )
        .unwrap())
        .unwrap();
        assert_eq!(ask.system, "be brief");
        assert!(ask.prefill());
        assert_eq!(ask.previous(), "yo");
        assert!(!ask.thinking);
        let bad = parse_openai(&user("x", r#", "max_tokens": 0"#)).unwrap_err();
        assert_eq!(
            (bad.message.as_str(), bad.param.as_deref()),
            ("max_tokens must be >= 1 (got 0)", Some("max_tokens"))
        );
        let bad =
            parse_anthropic(&parse(r#"{"messages": [{"role": "system", "content": "x"}]}"#).unwrap()).unwrap_err();
        assert_eq!(bad.message, "messages[0].role must be user or assistant (got 'system')");
    }

    #[test]
    fn the_reply_is_a_turn_and_the_thinking_its_trace() {
        let mut model = trained();
        let ask = parse_openai(&user("tell me about the cat", r#", "learn": false"#)).unwrap();
        let mut seen: Vec<Json> = Vec::new();
        let reply = respond(
            &mut model,
            None,
            &FilterConfig::default(),
            &ask,
            None,
            &mut |e: &Json| seen.push(e.clone()),
        )
        .unwrap();
        let choice = &reply.choices[0];
        let turn = choice.turn.as_ref().expect("a reply");
        assert_eq!(choice.text, turn.text);
        let lines: Vec<&str> = choice.thinking.split('\n').collect();
        assert_eq!(lines[0], "answering \"tell me about the cat\"");
        assert!(lines
            .last()
            .unwrap()
            .starts_with(&format!("saying {}: cost {:.4}", q(&turn.text), turn.cost)));
        let text: String = seen
            .iter()
            .filter(|e| e.at("type").as_str() == Some("text"))
            .map(|e| e.at("text").as_str().unwrap_or("").to_string())
            .collect();
        assert_eq!(text, choice.text);
        assert_eq!(seen.last().unwrap().at("type").as_str(), Some("end"));
        assert_eq!(reply.input_units, "tell me about the cat".len());
        let doc = to_openai(&reply);
        assert_eq!(doc.at("object").as_str(), Some("chat.completion"));
        assert_eq!(
            doc.at("choices").as_array()[0].at("message").at("content").as_str(),
            Some(choice.text.as_str())
        );
        let doc = to_anthropic(&reply);
        assert_eq!(doc.at("content").as_array()[0].at("type").as_str(), Some("thinking"));
        assert_eq!(
            doc.at("content").as_array()[1].at("text").as_str(),
            Some(choice.text.as_str())
        );
    }

    #[test]
    fn deltas_follow_the_walk() {
        let enc = Encoding::default();
        let labels: Vec<String> = ["<s>", "the", "he ", "e c", " ca", "cat", "</s>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // the ids below FIRST (START, END, BACK, THINK) are the sentinels; the first real node is FIRST
        assert_eq!(
            deltas(&enc, &labels, &[0, 4, 5, 6, 7, 8, 1], "the cat"),
            vec!["the", " ", "c", "a", "t"]
        );
        let labels: Vec<String> = ["the", "he ", "e cat"].iter().map(|s| s.to_string()).collect();
        assert_eq!(deltas(&enc, &labels, &[4, 5, 6], "xthe cat"), vec!["xthe", " ", "cat"]);
        assert_eq!(deltas(&enc, &labels, &[4, 5, 6], "the ca"), vec!["the ca"]);
        let words = Encoding::new(Unit::Words, 2, 1).unwrap();
        let labels: Vec<String> = ["<s>", "the cat", "cat sat", "sat on"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            deltas(&words, &labels, &[0, 4, 5, 6], "the cat sat on"),
            vec!["the cat", " sat", " on"]
        );
    }

    #[test]
    fn the_streams_open_and_close_their_blocks() {
        let mut model = trained();
        let ask = parse_anthropic(&user("the cat sat", r#", "learn": false"#)).unwrap();
        let mut seen: Vec<Json> = Vec::new();
        let reply = respond(
            &mut model,
            None,
            &FilterConfig::default(),
            &ask,
            None,
            &mut |e: &Json| seen.push(e.clone()),
        )
        .unwrap();
        let mut stream = AnthropicStream::new(&reply.token, &reply.model, reply.created, reply.input_units, true);
        let frames: Vec<String> = seen.iter().flat_map(|e| stream.frames(e)).collect();
        assert!(frames[0].starts_with("event: message_start\n"));
        assert!(frames[frames.len() - 1].starts_with("event: message_stop\n"));
        let starts = frames
            .iter()
            .filter(|f| f.starts_with("event: content_block_start"))
            .count();
        assert_eq!(starts, 2);
        let mut stream = OpenAiStream {
            token: reply.token.clone(),
            model: reply.model.clone(),
            created: reply.created,
            include_usage: true,
            thinking: true,
        };
        let frames: Vec<String> = seen.iter().flat_map(|e| stream.frames(e)).collect();
        assert_eq!(frames.last().unwrap(), "data: [DONE]\n\n");
        assert!(frames[frames.len() - 2].contains("\"usage\""));
    }

    #[test]
    fn errors_take_the_dialect_s_shape() {
        let doc = shape_error(OPENAI, 404, "no such model", Some("model"));
        assert_eq!(doc.at("error").at("code").as_str(), Some("model_not_found"));
        let doc = shape_error(ANTHROPIC, 400, "bad", None);
        assert_eq!(doc.at("error").at("type").as_str(), Some("invalid_request_error"));
        assert_eq!(kind_of_id("RadixNet-Count"), "count");
        assert_eq!(kind_of_id("radixnet"), "");
    }
}
