//! External tools the network can call, and the text format it learns them
//! in (`radixnet/tools.py`, `go/radixnet/tools.go`).
//!
//! The network is a character-level graph model: it cannot *decide* to call a
//! function, it can only emit characters.  So a tool call is a piece of text
//! like any other training data:
//!
//! ```text
//! <tool>web_fetch {"url": "https://example.com"}</tool>
//! ```
//!
//! and what the tool answers comes back as another piece of text:
//!
//! ```text
//! <result>Example Domain. This domain is for use in ...</result>
//! ```
//!
//! A whole attempt at a task is therefore one training text
//! ([`transcript_text`]) the network can be rewarded or punished for as a
//! unit.  Because the emission is only text it is routinely malformed, so
//! [`parse_call`] is deliberately lenient - JSON arguments (even cut off
//! mid-string), `key=value` pairs, or a bare value for a single-argument tool -
//! and whatever it still cannot read comes back with its `error` set rather
//! than dropped, for an LLM to repair.
//!
//! **The text format is the contract between the implementations**: a call is
//! written with its JSON arguments sorted and spaced exactly as Python's
//! `json.dumps(..., sort_keys=True)` writes them, the tools have Python's
//! names, parameters and descriptions ([`crate::toolbox`]), and the refusals
//! are Python's words - so a transcript written by one implementation is a
//! transcript the others read, and a model trained on one set of transcripts
//! carries over.
//!
//! [`ToolBox::call`] never fails outright: a failure is a [`ToolResult`] with
//! `ok` false and a message the network and the LLM can read, because a failed
//! call is training data too.
//!
//! Also here: `radixnet tools list | describe | call | browser`, and the
//! routes `GET /api/tools` and `POST /api/tools/call`, with [`State`] - the
//! server's tool defaults, read from the `serve` flags.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cli::{Args, Ctx};
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::{parse, py_repr, Json};
use crate::negative::python_repr;
use crate::service::Service;
use crate::toolbox::sandbox::{Sandbox, SandboxOptions};
use crate::toolbox::{default_toolbox, ToolOptions};
use crate::web::{collapse, WebOptions};

/// What this area's lines are filed under.
const LOG: &str = "tools";

/// The markers of the text format.
pub const CALL_OPEN: &str = "<tool>";
pub const CALL_CLOSE: &str = "</tool>";
pub const RESULT_OPEN: &str = "<result>";
pub const RESULT_CLOSE: &str = "</result>";
pub const ANSWER_OPEN: &str = "<answer>";
pub const ANSWER_CLOSE: &str = "</answer>";

/// How much of a tool's output goes back into the transcript by default.
pub const DEFAULT_OBSERVATION_CHARS: usize = 600;

/// The argument types a tool may declare.
pub const PARAM_TYPES: [&str; 4] = ["string", "number", "integer", "boolean"];

/// How a call is written, as `/api/tools` tells a client.
pub const CALL_FORMAT: &str = "<tool>name {\"argument\": \"value\"}</tool>";

// -- Python's view of a JSON value ------------------------------------------------------------------

/// A JSON value as Python's `json.dumps` writes it: `", "` and `": "` between
/// items, keys sorted when asked, and non-ASCII escaped when asked.
pub fn dumps(value: &Json, sort_keys: bool, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_dumps(value, sort_keys, ensure_ascii, &mut out);
    out
}

fn write_dumps(value: &Json, sort_keys: bool, ensure_ascii: bool, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(n) => out.push_str(&n.to_string()),
        Json::Num(x) => out.push_str(&py_repr(*x)),
        Json::Str(s) => dumps_string(s, ensure_ascii, out),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_dumps(item, sort_keys, ensure_ascii, out);
            }
            out.push(']');
        }
        Json::Obj(pairs) => {
            let mut pairs: Vec<&(String, Json)> = pairs.iter().collect();
            if sort_keys {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dumps_string(key, ensure_ascii, out);
                out.push_str(": ");
                write_dumps(item, sort_keys, ensure_ascii, out);
            }
            out.push('}');
        }
    }
}

fn dumps_string(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's name for the type of a decoded JSON value.
fn py_type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Num(_) => "float",
        Json::Str(_) => "str",
        Json::Arr(_) => "list",
        Json::Obj(_) => "dict",
    }
}

/// `str(value)` of a decoded JSON value.
pub fn py_str(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => py_repr_value(other),
    }
}

/// `repr(value)` of a decoded JSON value, as Python's messages quote it.
pub fn py_repr_value(value: &Json) -> String {
    match value {
        Json::Null => "None".to_string(),
        Json::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Json::Int(n) => n.to_string(),
        Json::Num(x) => crate::calc::float_repr(*x),
        Json::Str(s) => python_repr(s),
        Json::Arr(items) => format!("[{}]", items.iter().map(py_repr_value).collect::<Vec<_>>().join(", ")),
        Json::Obj(pairs) => format!(
            "{{{}}}",
            pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", python_repr(k), py_repr_value(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `float(value)` of a decoded JSON value: `None` where Python raises.
fn py_float(value: &Json) -> Option<f64> {
    match value {
        Json::Int(n) => Some(*n as f64),
        Json::Num(x) => Some(*x),
        Json::Bool(b) => Some(f64::from(u8::from(*b))),
        Json::Str(s) => parse_py_float(s),
        _ => None,
    }
}

/// `float(text)`: surrounding whitespace, underscores between digits, `inf`
/// and `nan` in any case.
fn parse_py_float(text: &str) -> Option<f64> {
    let text = text.trim();
    let bytes = text.as_bytes();
    if text.contains('_') {
        for (i, b) in bytes.iter().enumerate() {
            if *b == b'_'
                && !(i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))
            {
                return None;
            }
        }
    }
    text.replace('_', "").parse::<f64>().ok()
}

/// An object with `key` set: replaced where it already is, else added at the end.
fn set_key(pairs: &mut Vec<(String, Json)>, key: &str, value: Json) {
    match pairs.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => pairs.push((key.to_string(), value)),
    }
}

/// An object as a Python dict: a repeated key keeps its first place and its
/// last value.
fn dedupe(value: Json) -> Json {
    match value {
        Json::Obj(pairs) => {
            let mut out: Vec<(String, Json)> = Vec::with_capacity(pairs.len());
            for (key, item) in pairs {
                set_key(&mut out, &key, dedupe(item));
            }
            Json::Obj(out)
        }
        Json::Arr(items) => Json::Arr(items.into_iter().map(dedupe).collect()),
        other => other,
    }
}

fn empty() -> Json {
    Json::Obj(Vec::new())
}

// -- the text format --------------------------------------------------------------------------------

/// The first line of every transcript - what the network continues from.
pub fn task_header(prompt: &str) -> String {
    format!("TASK: {}\n", collapse(prompt))
}

/// `<tool>name {"arg": "value"}</tool>` - one line, JSON arguments with their
/// keys sorted, as Python writes them.
pub fn call_text(name: &str, arguments: &Json) -> String {
    let payload = match arguments {
        Json::Obj(pairs) if !pairs.is_empty() => dumps(arguments, true, false),
        _ => "{}".to_string(),
    };
    format!("{CALL_OPEN}{name} {payload}{CALL_CLOSE}\n")
}

/// `<result>...</result>` - the observation, whitespace collapsed and clipped
/// to `limit` characters (`None`: no clipping).
pub fn result_text(output: &str, limit: Option<usize>) -> String {
    let mut text = collapse(output);
    if let Some(limit) = limit {
        if text.chars().count() > limit {
            let clipped: String = text.chars().take(limit).collect();
            text = format!("{} ...", clipped.trim_end());
        }
    }
    format!("{RESULT_OPEN}{text}{RESULT_CLOSE}\n")
}

/// `<answer>...</answer>` - the last line of a finished transcript.
pub fn answer_text(answer: &str) -> String {
    format!("{ANSWER_OPEN}{}{ANSWER_CLOSE}\n", collapse(answer))
}

/// What a tool call adds to the transcript: the output, or the error
/// prefixed with `ERROR:`.
pub fn format_observation(result: &ToolResult, limit: Option<usize>) -> String {
    if result.ok {
        result_text(&result.output, limit)
    } else {
        result_text(&format!("ERROR: {}", result.error.as_deref().unwrap_or("")), limit)
    }
}

/// One call of a transcript, with what it answered.
#[derive(Clone, Debug)]
pub struct TranscriptStep {
    pub name: String,
    pub arguments: Json,
    pub result: ToolResult,
}

/// One attempt as a single training text: the task, every call with its
/// result, then the answer (`None`: the attempt never answered).
pub fn transcript_text(prompt: &str, steps: &[TranscriptStep], answer: Option<&str>, limit: Option<usize>) -> String {
    let mut out = task_header(prompt);
    for step in steps {
        out.push_str(&call_text(&step.name, &step.arguments));
        out.push_str(&format_observation(&step.result, limit));
    }
    if let Some(answer) = answer {
        out.push_str(&answer_text(answer));
    }
    out
}

/// JSON out of text a generator may have cut off mid-string or mid-object.
///
/// A call is one line by construction, so the line is tried first, then the
/// text up to its last `}`, then the same completed with the closing quote
/// and braces it is missing.
fn loads_truncated(text: &str) -> Option<Json> {
    let head = text.split('\n').next().unwrap_or("").trim();
    let bases: Vec<&str> = if head != text { vec![head, text] } else { vec![text] };
    let mut candidates: Vec<String> = Vec::new();
    for base in bases {
        candidates.push(base.to_string());
        if let Some(at) = base.rfind('}') {
            candidates.push(base[..=at].to_string());
        }
        for tail in ["\"}", "}", "\"}}", "}}", "\"]}"] {
            candidates.push(format!("{base}{tail}"));
        }
    }
    candidates
        .iter()
        .find_map(|candidate| parse(candidate).ok())
        .map(dedupe)
}

/// `[A-Za-z_][A-Za-z0-9_]*`, from `at`: where it ends.
fn identifier(bytes: &[u8], at: usize) -> Option<usize> {
    let first = *bytes.get(at)?;
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    let mut end = at + 1;
    while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
        end += 1;
    }
    Some(end)
}

/// Python's `_PAIR_RE.findall`: every `key = value` / `key: value` pair, the
/// value a quoted string or a run up to the next `,` or `;`.
fn find_pairs(text: &str) -> Vec<(String, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut at = 0;
    'scan: while at < bytes.len() {
        let Some(key_end) = identifier(bytes, at) else {
            at += text[at..].chars().next().map_or(1, char::len_utf8);
            continue;
        };
        let mut i = key_end;
        while let Some(c) = text[i..].chars().next().filter(|c| c.is_whitespace()) {
            i += c.len_utf8();
        }
        if !matches!(bytes.get(i), Some(b'=') | Some(b':')) {
            at += 1;
            continue;
        }
        let after_sep = i + 1;
        let mut ws_end = after_sep;
        while let Some(c) = text[ws_end..].chars().next().filter(|c| c.is_whitespace()) {
            ws_end += c.len_utf8();
        }
        // the three alternatives, in order, at the end of the whitespace
        for quote in [b'"', b'\''] {
            if bytes.get(ws_end) == Some(&quote) {
                if let Some(close) = bytes[ws_end + 1..].iter().position(|b| *b == quote) {
                    let end = ws_end + 1 + close + 1;
                    out.push((text[at..key_end].to_string(), text[ws_end..end].to_string()));
                    at = end;
                    continue 'scan;
                }
            }
        }
        let run_end = text[ws_end..].find([',', ';']).map_or(text.len(), |n| ws_end + n);
        if run_end > ws_end {
            out.push((text[at..key_end].to_string(), text[ws_end..run_end].to_string()));
            at = run_end;
            continue;
        }
        if ws_end > after_sep {
            // the whitespace gives one character back so the run can match it
            let last = text[after_sep..ws_end].chars().last().map_or(1, char::len_utf8);
            out.push((text[at..key_end].to_string(), text[ws_end - last..ws_end].to_string()));
            at = ws_end;
            continue;
        }
        at += 1;
    }
    out
}

/// Strips `chars` from both ends (Python's `str.strip(chars)`).
fn strip_chars<'a>(text: &'a str, chars: &[char]) -> &'a str {
    text.trim_matches(|c| chars.contains(&c))
}

/// The arguments of a call: JSON, `key=value` pairs, or a bare value for a
/// single-argument tool.
pub fn parse_arguments(raw: &str, tool: Option<&Tool>) -> Result<Json, String> {
    let text = strip_chars(raw.trim(), &[',', ';']);
    if text.is_empty() {
        return Ok(empty());
    }
    if text.starts_with('{') || text.starts_with('[') {
        match loads_truncated(text) {
            Some(object @ Json::Obj(_)) => return Ok(object),
            Some(Json::Arr(items)) => {
                if let Some(tool) = tool {
                    let pairs = tool
                        .params
                        .iter()
                        .zip(items)
                        .map(|(param, value)| (param.name.clone(), value))
                        .collect();
                    return Ok(Json::Obj(pairs));
                }
            }
            _ => {}
        }
    }
    let pairs = find_pairs(text);
    if !pairs.is_empty() {
        let mut out = Vec::new();
        for (key, value) in pairs {
            set_key(&mut out, &key, Json::str(strip_chars(value.trim(), &['"', '\''])));
        }
        return Ok(Json::Obj(out));
    }
    if let Some(tool) = tool {
        let required: Vec<&Param> = tool.params.iter().filter(|p| p.required).collect();
        let wanted: Vec<&Param> = if required.is_empty() {
            tool.params.iter().collect()
        } else {
            required
        };
        if wanted.len() == 1 {
            return Ok(Json::obj([(
                wanted[0].name.clone(),
                Json::str(strip_chars(text, &['"', '\''])),
            )]));
        }
    }
    let shown: String = raw.trim().chars().take(120).collect();
    Err(format!(
        "cannot read the arguments of the call: {}",
        python_repr(&shown)
    ))
}

/// The byte span of the first `<open>...</close>` (or `<open>...` to the end)
/// and the text inside it.
fn block<'a>(text: &'a str, open: &str, close: &str) -> Option<((usize, usize), &'a str)> {
    let start = text.find(open)?;
    let inner_start = start + open.len();
    match text[inner_start..].find(close) {
        Some(at) => Some((
            (start, inner_start + at + close.len()),
            &text[inner_start..inner_start + at],
        )),
        None => Some(((start, text.len()), &text[inner_start..])),
    }
}

/// The first `<tool>` call in `text` (`None` when there is none).
///
/// Lenient on purpose: an unterminated call is read to the end of the text,
/// the tool name is whatever leading identifier is there, and the arguments
/// go through [`parse_arguments`].  A call that cannot be understood comes
/// back with `error` set rather than dropped, so the caller can hand it to
/// the LLM mediator instead of guessing.
pub fn parse_call(text: &str, toolbox: Option<&ToolBox>) -> Option<ToolCall> {
    let (span, inner) = block(text, CALL_OPEN, CALL_CLOSE)?;
    let body = inner.trim();
    let bytes = body.as_bytes();
    let name_end = match bytes.first() {
        Some(b) if b.is_ascii_alphabetic() || *b == b'_' => bytes
            .iter()
            .position(|b| !(b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')))
            .unwrap_or(bytes.len()),
        _ => {
            let shown: String = body.chars().take(80).collect();
            return Some(ToolCall {
                name: String::new(),
                arguments: empty(),
                raw: body.to_string(),
                span,
                error: Some(format!("no tool name in {}", python_repr(&shown))),
            });
        }
    };
    let name = body[..name_end].trim_end_matches(['.', '-']).to_string();
    let rest = body[name_end..].trim().to_string();
    let tool = match toolbox {
        Some(bx) => match bx.get(&name) {
            Ok(tool) => Some(tool),
            Err(_) => {
                return Some(ToolCall {
                    error: Some(format!(
                        "unknown tool {} (have: {})",
                        python_repr(&name),
                        bx.names().join(", ")
                    )),
                    name,
                    arguments: empty(),
                    raw: rest,
                    span,
                })
            }
        },
        None => None,
    };
    match parse_arguments(&rest, tool) {
        Ok(arguments) => Some(ToolCall {
            name,
            arguments,
            raw: rest,
            span,
            error: None,
        }),
        Err(why) => Some(ToolCall {
            name,
            arguments: empty(),
            raw: rest,
            span,
            error: Some(why),
        }),
    }
}

/// The text of the first `<answer>` block (`None` when the attempt did not
/// answer, or answered nothing).
pub fn find_answer(text: &str) -> Option<String> {
    let (_, inner) = block(text, ANSWER_OPEN, ANSWER_CLOSE)?;
    Some(collapse(inner)).filter(|a| !a.is_empty())
}

// -- tools ------------------------------------------------------------------------------------------

/// One argument of a tool, with the little bit of schema an LLM needs to fill
/// it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    /// One of [`PARAM_TYPES`].
    pub kind: String,
    pub description: String,
    pub required: bool,
    /// `Json::Null` is no default.
    pub default: Json,
    pub choices: Option<Vec<String>>,
}

impl Param {
    /// A required argument.
    pub fn new(name: &str, kind: &str, description: &str) -> Param {
        Param {
            name: name.to_string(),
            kind: kind.to_string(),
            description: description.to_string(),
            required: true,
            default: Json::Null,
            choices: None,
        }
    }

    /// An optional argument with its default.
    pub fn optional(name: &str, kind: &str, description: &str, default: Json) -> Param {
        Param {
            required: false,
            default,
            ..Param::new(name, kind, description)
        }
    }

    /// The parameter as JSON schema.
    pub fn schema(&self) -> Json {
        let mut pairs = vec![
            ("type".to_string(), Json::str(self.kind.clone())),
            ("description".to_string(), Json::str(self.description.clone())),
        ];
        if let Some(choices) = self.choices.as_ref().filter(|c| !c.is_empty()) {
            pairs.push(("enum".to_string(), Json::strs(choices.clone())));
        }
        Json::Obj(pairs)
    }

    /// `value` as this parameter's type, or why it cannot be.
    pub fn coerce(&self, value: &Json) -> Result<Json, String> {
        let bad = || {
            format!(
                "argument {} must be a {} (got {})",
                python_repr(&self.name),
                self.kind,
                py_repr_value(value)
            )
        };
        let out = match self.kind.as_str() {
            "string" => match value {
                Json::Str(_) => value.clone(),
                Json::Obj(_) | Json::Arr(_) => Json::str(dumps(value, false, true)),
                other => Json::str(py_str(other)),
            },
            "boolean" => match value {
                Json::Bool(_) => value.clone(),
                other => Json::Bool(matches!(
                    py_str(other).trim().to_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )),
            },
            "integer" => match value {
                Json::Bool(b) => Json::Int(i64::from(*b)),
                other => {
                    let x = py_float(other).ok_or_else(bad)?;
                    if x.is_nan() {
                        return Err(bad());
                    }
                    if x.is_infinite() {
                        return Err("OverflowError: cannot convert float infinity to integer".to_string());
                    }
                    Json::Int(x.trunc() as i64)
                }
            },
            _ => Json::Num(py_float(value).ok_or_else(bad)?),
        };
        if let Some(choices) = self.choices.as_ref().filter(|c| !c.is_empty()) {
            if !choices.contains(&py_str(&out)) {
                return Err(format!(
                    "argument {} must be one of {} (got {})",
                    python_repr(&self.name),
                    choices.join(", "),
                    py_repr_value(&out)
                ));
            }
        }
        Ok(out)
    }

    /// The parameter as the API and the CLI report it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("name", Json::str(self.name.clone())),
            ("type", Json::str(self.kind.clone())),
            ("description", Json::str(self.description.clone())),
            ("required", Json::Bool(self.required)),
            ("default", self.default.clone()),
            (
                "enum",
                match self.choices.as_ref().filter(|c| !c.is_empty()) {
                    Some(choices) => Json::strs(choices.clone()),
                    None => Json::Null,
                },
            ),
        ])
    }
}

/// What a handler answers: the output, and whatever metadata the caller wants
/// (an object; empty for none).
pub type Output = (String, Json);

/// Runs a tool on its validated arguments (an object); `Err` is a clean
/// failure the transcript records.
pub type Handler = Arc<dyn Fn(&Json) -> Result<Output, String> + Send + Sync>;

/// A named function the network can call by emitting text.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub params: Vec<Param>,
    pub handler: Option<Handler>,
    /// Whether the tool reaches the network (what `--offline` removes).
    pub network: bool,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("params", &self.params)
            .field("network", &self.network)
            .finish()
    }
}

impl Tool {
    /// A tool; `handler` runs it.
    pub fn new(
        name: &str,
        description: &str,
        params: Vec<Param>,
        handler: impl Fn(&Json) -> Result<Output, String> + Send + Sync + 'static,
    ) -> Tool {
        Tool {
            name: name.to_string(),
            description: description.to_string(),
            params,
            handler: Some(Arc::new(handler)),
            network: false,
        }
    }

    /// `name(arg: type, [optional]: type)` - the one-liner the prompts and the
    /// CLI show.
    pub fn signature(&self) -> String {
        let parts: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                if p.required {
                    format!("{}: {}", p.name, p.kind)
                } else {
                    format!("[{}: {}]", p.name, p.kind)
                }
            })
            .collect();
        format!("{}({})", self.name, parts.join(", "))
    }

    /// The tool in Ollama's / OpenAI's `tools` format.
    pub fn schema(&self) -> Json {
        Json::obj([
            ("type", Json::str("function")),
            (
                "function",
                Json::obj([
                    ("name", Json::str(self.name.clone())),
                    ("description", Json::str(self.description.clone())),
                    (
                        "parameters",
                        Json::obj([
                            ("type", Json::str("object")),
                            (
                                "properties",
                                Json::Obj(self.params.iter().map(|p| (p.name.clone(), p.schema())).collect()),
                            ),
                            (
                                "required",
                                Json::strs(self.params.iter().filter(|p| p.required).map(|p| p.name.clone())),
                            ),
                        ]),
                    ),
                ]),
            ),
        ])
    }

    /// Validated arguments: types coerced, defaults filled in, unknown names
    /// dropped (a name is matched without regard to case), missing ones refused.
    pub fn coerce(&self, arguments: &Json) -> Result<Json, String> {
        let Json::Obj(pairs) = arguments else {
            return Err(format!(
                "{}: arguments must be an object (got {})",
                self.name,
                py_type_name(arguments)
            ));
        };
        let mut out: Vec<(String, Json)> = Vec::new();
        for (key, value) in pairs {
            let param = match self.params.iter().find(|p| p.name == *key) {
                Some(p) => Some(p),
                None => {
                    let folded = key.trim().to_lowercase();
                    self.params.iter().rev().find(|p| p.name.to_lowercase() == folded)
                }
            };
            let Some(param) = param else { continue };
            if value.is_null() {
                continue;
            }
            let coerced = param.coerce(value)?;
            set_key(&mut out, &param.name, coerced);
        }
        let missing: Vec<&str> = self
            .params
            .iter()
            .filter(|p| p.required && !out.iter().any(|(k, _)| *k == p.name))
            .map(|p| p.name.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{}: missing argument(s) {} — {}",
                self.name,
                missing.join(", "),
                self.signature()
            ));
        }
        for param in &self.params {
            if !out.iter().any(|(k, _)| *k == param.name) && !param.default.is_null() {
                let coerced = param.coerce(&param.default)?;
                out.push((param.name.clone(), coerced));
            }
        }
        Ok(Json::Obj(out))
    }

    /// The tool as the API and the CLI report it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("name", Json::str(self.name.clone())),
            ("description", Json::str(self.description.clone())),
            ("signature", Json::str(self.signature())),
            ("network", Json::Bool(self.network)),
            ("params", Json::Arr(self.params.iter().map(Param::to_json).collect())),
        ])
    }
}

/// A call read out of generated text: the name, the arguments and why it is
/// unusable (`error`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// An object.
    pub arguments: Json,
    /// The text after the name, as written.
    pub raw: String,
    /// The byte range of the whole `<tool>...</tool>` in the text it was read from.
    pub span: (usize, usize),
    pub error: Option<String>,
}

impl ToolCall {
    /// Whether the call can be run.
    pub fn ok(&self) -> bool {
        self.error.is_none() && !self.name.is_empty()
    }

    /// The call as the network would have written it.
    pub fn text(&self) -> String {
        call_text(&self.name, &self.arguments)
    }

    /// `{"tool", "arguments", "raw", "error"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("tool", Json::str(self.name.clone())),
            ("arguments", self.arguments.clone()),
            ("raw", Json::str(self.raw.clone())),
            ("error", self.error.clone().map_or(Json::Null, Json::Str)),
        ])
    }
}

/// What a tool answered: `output` when `ok`, otherwise `error` - both readable
/// by the network and the LLM.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub tool: String,
    pub arguments: Json,
    pub ok: bool,
    pub output: String,
    pub error: Option<String>,
    pub seconds: f64,
    pub meta: Json,
}

impl ToolResult {
    /// A failed call.
    pub fn failed(tool: &str, arguments: Json, error: String, seconds: f64) -> ToolResult {
        ToolResult {
            tool: tool.to_string(),
            arguments,
            ok: false,
            output: String::new(),
            error: Some(error),
            seconds,
            meta: empty(),
        }
    }

    /// `{"tool", "arguments", "ok", "output", "error", "seconds", "meta"}`,
    /// the seconds rounded to four places as Python rounds them.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("tool", Json::str(self.tool.clone())),
            ("arguments", self.arguments.clone()),
            ("ok", Json::Bool(self.ok)),
            ("output", Json::str(self.output.clone())),
            ("error", self.error.clone().map_or(Json::Null, Json::Str)),
            (
                "seconds",
                Json::Num(crate::calc::round_float(self.seconds, 4).unwrap_or(self.seconds)),
            ),
            ("meta", self.meta.clone()),
        ])
    }
}

/// The registry: what exists, what it looks like to an LLM, and how a call is
/// run.
#[derive(Debug, Default)]
pub struct ToolBox {
    tools: Vec<Tool>,
    calls: AtomicUsize,
}

impl ToolBox {
    pub fn new() -> ToolBox {
        ToolBox::default()
    }

    /// Adds a tool, replacing one of the same name where it stands.
    pub fn register(&mut self, tool: Tool) -> Result<(), String> {
        let bytes = tool.name.as_bytes();
        let valid = bytes.first().is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
            && bytes
                .iter()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
        if !valid {
            return Err(format!("invalid tool name {}", python_repr(&tool.name)));
        }
        if let Some(param) = tool.params.iter().find(|p| !PARAM_TYPES.contains(&p.kind.as_str())) {
            return Err(format!(
                "param {}: type must be one of {} (got {})",
                python_repr(&param.name),
                PARAM_TYPES.join(", "),
                python_repr(&param.kind)
            ));
        }
        match self.tools.iter_mut().find(|t| t.name == tool.name) {
            Some(slot) => *slot = tool,
            None => self.tools.push(tool),
        }
        Ok(())
    }

    /// Drops a tool; says whether there was one.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.tools.len();
        self.tools.retain(|t| t.name != name);
        self.tools.len() != before
    }

    pub fn has(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    /// The tool of that name, or the error the network reads.
    pub fn get(&self, name: &str) -> Result<&Tool, String> {
        self.tools.iter().find(|t| t.name == name).ok_or_else(|| {
            let names = self.names().join(", ");
            format!(
                "unknown tool {} (have: {})",
                python_repr(name),
                if names.is_empty() { "none".to_string() } else { names }
            )
        })
    }

    /// The names, in registration order.
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// How many calls this toolbox has run.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// Every tool as the API reports it.
    pub fn describe(&self) -> Json {
        Json::Arr(self.tools.iter().map(Tool::to_json).collect())
    }

    /// Every tool in Ollama's `tools` format.
    pub fn schemas(&self) -> Json {
        Json::Arr(self.tools.iter().map(Tool::schema).collect())
    }

    /// The tool list as prompt text: one `- name(args) — description` line each.
    pub fn catalogue(&self) -> String {
        self.tools
            .iter()
            .map(|t| format!("- {} — {}", t.signature(), t.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Runs one tool; a failure is reported, never raised.
    pub fn call(&self, name: &str, arguments: &Json) -> ToolResult {
        let started = Instant::now();
        self.calls.fetch_add(1, Ordering::Relaxed);
        let given = match arguments {
            Json::Obj(_) => arguments.clone(),
            _ => empty(),
        };
        let fail = |why: String| {
            crate::log_debug!(LOG, "{name} failed: {why}");
            ToolResult::failed(name, given.clone(), why, started.elapsed().as_secs_f64())
        };
        let tool = match self.get(name) {
            Ok(tool) => tool,
            Err(why) => return fail(why),
        };
        let values = match tool.coerce(if arguments.is_null() { &given } else { arguments }) {
            Ok(values) => values,
            Err(why) => return fail(why),
        };
        let Some(handler) = &tool.handler else {
            return fail(format!("{name}: no handler is installed"));
        };
        match handler(&values) {
            Ok((output, meta)) => {
                crate::log_debug!(LOG, "{name} answered {} char(s)", output.chars().count());
                ToolResult {
                    tool: name.to_string(),
                    arguments: values,
                    ok: true,
                    output,
                    error: None,
                    seconds: started.elapsed().as_secs_f64(),
                    meta: if meta.is_null() { empty() } else { meta },
                }
            }
            Err(why) => fail(why),
        }
    }

    /// [`ToolBox::call`] for a parsed call; a broken one becomes a failed result.
    pub fn run(&self, call: &ToolCall) -> ToolResult {
        if !call.ok() {
            let why = call.error.clone().unwrap_or_else(|| "unusable tool call".to_string());
            return ToolResult::failed(&call.name, call.arguments.clone(), why, 0.0);
        }
        self.call(&call.name, &call.arguments)
    }
}

// -- the tool settings: the CLI's flags and the server's defaults -----------------------------------

/// Which tools exist and how the browsing behaves: the `tools` flags on the
/// command line, and the server's defaults for `/api/tools` and the agent
/// routes (which a request may narrow).  Python's `tool_options`.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// No web tools at all.
    pub offline: bool,
    /// Let the web tools reach private addresses (a local test site).
    pub allow_private: bool,
    /// `None`: `$RADIXNET_SEARCH_URL`, else DuckDuckGo.
    pub search_url: Option<String>,
    /// Seconds per page.
    pub web_timeout: f64,
    /// The most bytes read from one page.
    pub max_bytes: usize,
    /// Also offer the sandboxed `python` tool.
    pub python_tool: bool,
    /// Seconds a sandboxed program may run.
    pub sandbox_timeout: f64,
    /// Run sandboxed programs in a network namespace of their own.
    pub network_isolation: bool,
    /// Draw pages in Chrome ([`crate::browser`]).
    pub browser: bool,
    /// Seconds a page may take to draw.
    pub page_timeout: f64,
    /// Chrome without a window (a command line's `--no-headless` turns it
    /// off; a server's browser is always headless, as Python's is).
    pub headless: bool,
    /// The one browser a server's requests share, started on first use.
    pub drawn_by: crate::browser::Shared,
}

impl Default for State {
    /// Python's defaults.
    fn default() -> State {
        State {
            offline: false,
            allow_private: false,
            search_url: None,
            web_timeout: 20.0,
            max_bytes: 2_000_000,
            python_tool: false,
            sandbox_timeout: 10.0,
            network_isolation: true,
            browser: false,
            page_timeout: crate::browser::DEFAULT_PAGE_TIMEOUT,
            headless: true,
            drawn_by: crate::browser::Shared::default(),
        }
    }
}

/// A number flag that must be finite and at least `minimum`.
fn float_flag(args: &Args, name: &str, fallback: f64, minimum: f64) -> Result<f64, String> {
    let Some(text) = args.get(name) else {
        return Ok(fallback);
    };
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("argument --{name}: expected a number, got {}", python_repr(text)))?;
    if !value.is_finite() || value < minimum {
        return Err(format!(
            "argument --{name}: must be a finite number >= {}, got {text}",
            crate::toolbox::sandbox::format_g(minimum)
        ));
    }
    Ok(value)
}

impl State {
    /// Reads the tool flags (`--offline`, `--allow-private`, `--search-url`,
    /// `--web-timeout`, `--max-bytes`, `--python-tool`, `--sandbox-timeout`,
    /// `--no-network-isolation`, `--browser`, `--page-timeout`) over these defaults.
    pub fn configure(&mut self, args: &Args) -> Result<(), String> {
        self.offline = args.on("offline");
        self.allow_private = args.on("allow-private");
        self.search_url = args
            .get("search-url")
            .map(str::to_string)
            .filter(|u| !u.trim().is_empty());
        self.web_timeout = float_flag(args, "web-timeout", self.web_timeout, 0.1)?;
        if let Some(text) = args.get("max-bytes") {
            let value: i64 = text
                .trim()
                .parse()
                .map_err(|_| format!("argument --max-bytes: expected an integer, got {}", python_repr(text)))?;
            if value < 1024 {
                return Err(format!("argument --max-bytes: must be >= 1024, got {value}"));
            }
            self.max_bytes = value as usize;
        }
        self.python_tool = args.on("python-tool");
        self.sandbox_timeout = float_flag(args, "sandbox-timeout", self.sandbox_timeout, 0.1)?;
        self.network_isolation = !args.on("no-network-isolation");
        self.browser = args.on("browser");
        self.page_timeout = float_flag(args, "page-timeout", self.page_timeout, 1.0)?;
        Ok(())
    }

    /// The settings as the API reports them, in Python's order.
    pub fn options_json(&self) -> Json {
        Json::obj([
            ("offline", Json::Bool(self.offline)),
            ("allow_private", Json::Bool(self.allow_private)),
            ("search_url", self.search_url.clone().map_or(Json::Null, Json::Str)),
            ("web_timeout", Json::Num(self.web_timeout)),
            ("max_bytes", Json::Int(self.max_bytes as i64)),
            ("python_tool", Json::Bool(self.python_tool)),
            ("sandbox_timeout", Json::Num(self.sandbox_timeout)),
            ("browser", Json::Bool(self.browser)),
            ("page_timeout", Json::Num(self.page_timeout)),
        ])
    }

    /// The sandbox the `python` tool runs in, when it is offered.
    pub fn sandbox(&self) -> Result<Option<Arc<Sandbox>>, String> {
        if !self.python_tool {
            return Ok(None);
        }
        let sandbox = Sandbox::new(SandboxOptions {
            timeout: Duration::from_secs_f64(self.sandbox_timeout),
            isolate_network: self.network_isolation,
            ..Default::default()
        })?;
        Ok(Some(Arc::new(sandbox)))
    }

    /// The toolbox these settings build, `read_file` over `upload_dir` when
    /// there is one, its pages drawn in the shared browser when `browser` is on.
    pub fn toolbox(&self, upload_dir: Option<&str>) -> Result<ToolBox, String> {
        let browser = if self.browser && !self.offline {
            Some(self.drawn_by.get(self.page_timeout, self.headless)?)
        } else {
            None
        };
        default_toolbox(ToolOptions {
            offline: self.offline,
            web: WebOptions {
                timeout: Duration::from_secs_f64(self.web_timeout),
                max_bytes: self.max_bytes,
                allow_private: self.allow_private,
                search_url: self.search_url.clone(),
                browser,
                ..Default::default()
            },
            sandbox: self.sandbox()?,
            upload_dir: upload_dir.filter(|d| !d.trim().is_empty()).map(str::to_string),
            extra: Vec::new(),
        })
    }
}

/// What `tools browser` and `/api/tools` say about the browser: the
/// chromedriver and Chrome found, their versions, and why they would not work.
fn browser_json() -> Json {
    crate::browser::describe().to_json()
}

/// The toolbox a command-line run gets from its flags: browsing (unless
/// `--offline`, drawn in Chrome with `--browser`), the calculator, the sandbox
/// (`--python-tool`) and the uploads (`--upload-dir`).  What `tools`, `agent`,
/// `explore` and `mcp` share.
pub fn build_toolbox(args: &Args) -> Result<ToolBox, String> {
    let mut state = State::default();
    state.configure(args)?;
    if state.browser && !state.offline {
        state.headless = !args.on("no-headless");
        let browser = crate::browser::from_flags(state.page_timeout, state.headless)?;
        state.drawn_by = crate::browser::Shared::with(browser);
    }
    state.toolbox(args.get("upload-dir"))
}

// -- the command line -------------------------------------------------------------------------------

/// `radixnet tools list | describe | call | browser`: the tools, one in
/// detail, one called directly (no model, no LLM), and the Chrome `--browser`
/// would drive.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    match args.action() {
        "" | "list" => {
            let toolbox = build_toolbox(args)?;
            ctx.emit(Json::obj([
                ("tools", toolbox.describe()),
                ("offline", Json::Bool(args.on("offline"))),
            ]));
            Ok(())
        }
        "describe" => {
            let name = args.str("tool", "");
            if name.trim().is_empty() {
                return Err("`tools describe` needs --tool NAME".to_string());
            }
            let toolbox = build_toolbox(args)?;
            let tool = toolbox.get(name.trim())?;
            ctx.emit(Json::obj([("tool", tool.to_json()), ("schema", tool.schema())]));
            Ok(())
        }
        "call" => {
            let toolbox = build_toolbox(args)?;
            let (name, arguments) = match args.get("call").filter(|c| !c.is_empty()) {
                Some(raw) => {
                    let text = if raw.starts_with(CALL_OPEN) {
                        raw.to_string()
                    } else {
                        format!("{CALL_OPEN}{raw}{CALL_CLOSE}")
                    };
                    match parse_call(&text, Some(&toolbox)) {
                        Some(call) if call.ok() => (call.name, call.arguments),
                        Some(call) => return Err(call.error.unwrap_or_default()),
                        None => return Err(format!("no tool call in {}", python_repr(raw))),
                    }
                }
                None => {
                    let name = args.str("tool", "");
                    if name.is_empty() {
                        return Err(
                            "`tools call` needs --tool NAME (with --arg) or --call '<tool>name {...}</tool>'"
                                .to_string(),
                        );
                    }
                    let mut pairs = Vec::new();
                    for pair in args.all("arg") {
                        let Some((key, value)) = pair.split_once('=') else {
                            return Err(format!("--arg expects name=value, got {}", python_repr(&pair)));
                        };
                        set_key(&mut pairs, key.trim(), Json::str(value));
                    }
                    (name, Json::Obj(pairs))
                }
            };
            let result = toolbox.call(&name, &arguments);
            if !result.ok {
                // a failed call is an observation to the loop, but an error to a shell
                return Err(format!("{}: {}", result.tool, result.error.unwrap_or_default()));
            }
            ctx.emit(result.to_json());
            Ok(())
        }
        "browser" => {
            // the flags are read (and refused) first, as for every action
            build_toolbox(args)?;
            ctx.emit(browser_json());
            Ok(())
        }
        other => Err(format!(
            "unknown tools action {} (have: list, describe, call, browser)",
            python_repr(other)
        )),
    }
}

// -- the routes -------------------------------------------------------------------------------------

/// Python's `_json_type`: how a request field's type is named in an error.
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

fn bad_field(name: &str, expected: &str, value: &Json) -> ApiError {
    ApiError::bad_request(format!(
        "'{name}' must be {expected} (got {} {})",
        json_type(value),
        py_repr_value(value)
    ))
}

/// A body field that is there and not `null`.
fn field<'a>(r: &'a Request, name: &str) -> Option<&'a Json> {
    r.body.get(name).filter(|v| !v.is_null())
}

fn text_field(r: &Request, name: &str) -> Result<Option<String>, ApiError> {
    match field(r, name) {
        None => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.clone())),
        Some(other) => Err(bad_field(name, "a string", other)),
    }
}

/// A request's own word on a switch: the body's (a boolean), else the query's.
fn flag_field(r: &Request, name: &str) -> Result<Option<bool>, ApiError> {
    match field(r, name) {
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(bad_field(name, "a boolean", other)),
        None => Ok(r
            .query(name)
            .map(|raw| matches!(raw.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))),
    }
}

fn number_field(r: &Request, name: &str, minimum: f64) -> Result<Option<f64>, ApiError> {
    let Some(value) = field(r, name) else { return Ok(None) };
    let x = match value {
        Json::Int(n) => *n as f64,
        Json::Num(x) if x.is_finite() => *x,
        other => return Err(bad_field(name, "a finite number", other)),
    };
    if x < minimum {
        return Err(ApiError::bad_request(format!(
            "'{name}' must be >= {} (got {})",
            crate::calc::float_repr(minimum),
            py_str(value)
        )));
    }
    Ok(Some(x))
}

fn integer_field(r: &Request, name: &str, minimum: i64) -> Result<Option<i64>, ApiError> {
    let Some(value) = field(r, name) else { return Ok(None) };
    let n = match value {
        Json::Int(n) => *n,
        Json::Num(x) if x.is_finite() && x.fract() == 0.0 => *x as i64,
        other => return Err(bad_field(name, "an integer", other)),
    };
    if n < minimum {
        return Err(ApiError::bad_request(format!(
            "'{name}' must be >= {minimum} (got {n})"
        )));
    }
    Ok(Some(n))
}

/// The tools of one request: the server's defaults with the request's own
/// `offline` / `allow_private` / `python_tool` / `search_url` /
/// `web_timeout` / `max_bytes` applied.  What the agent routes build on too.
pub(crate) fn toolbox_from(svc: &Service, r: &Request) -> Result<ToolBox, ApiError> {
    let mut state = svc.tools.clone();
    if let Some(offline) = flag_field(r, "offline")? {
        state.offline = offline;
    }
    if let Some(allow) = flag_field(r, "allow_private")? {
        state.allow_private = allow;
    }
    if let Some(python) = flag_field(r, "python_tool")? {
        state.python_tool = python;
    }
    if let Some(browser) = flag_field(r, "browser")? {
        state.browser = browser;
    }
    if let Some(url) = text_field(r, "search_url")? {
        state.search_url = Some(url).filter(|u| !u.trim().is_empty());
    }
    if let Some(timeout) = number_field(r, "web_timeout", 0.1)? {
        state.web_timeout = timeout;
    }
    if let Some(bytes) = integer_field(r, "max_bytes", 1024)? {
        state.max_bytes = bytes as usize;
    }
    state.toolbox(svc.upload_dir.as_deref()).map_err(ApiError::bad_request)
}

/// The `tools` block of Python's `/api/status`: the names on offer, sorted,
/// then the settings (`{"names": [...], "offline": ..., ...}`).
pub fn status_json(svc: &Service) -> Json {
    let mut names = svc
        .tools
        .toolbox(svc.upload_dir.as_deref())
        .map(|toolbox| toolbox.names())
        .unwrap_or_default();
    names.sort();
    let mut pairs = vec![("names".to_string(), Json::strs(names))];
    if let Json::Obj(options) = svc.tools.options_json() {
        pairs.extend(options);
    }
    Json::Obj(pairs)
}

/// `GET /api/tools`: what the network may call, and how.
fn tools_route(svc: &Arc<Service>, _r: &Request) -> Answer {
    let toolbox = svc
        .tools
        .toolbox(svc.upload_dir.as_deref())
        .map_err(ApiError::bad_request)?;
    Ok(Json::obj([
        ("tools", toolbox.describe()),
        ("names", Json::strs(toolbox.names())),
        ("count", Json::Int(toolbox.len() as i64)),
        ("options", svc.tools.options_json()),
        ("upload_dir", svc.upload_dir.clone().map_or(Json::Null, Json::Str)),
        ("call_format", Json::str(CALL_FORMAT)),
        ("browser", browser_json()),
    ]))
}

/// `POST /api/tools/call`: `{tool, arguments}` or `{call: 'name {...}'}`, plus
/// the tool overrides; answers the result - a tool that fails is a 200 with
/// `ok` false, because the failure is the answer, not an API error.
fn call_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let toolbox = toolbox_from(svc, r)?;
    let raw = text_field(r, "call")?.unwrap_or_default();
    let (name, arguments) = if !raw.trim().is_empty() {
        let text = if raw.trim_start().starts_with(CALL_OPEN) {
            raw.clone()
        } else {
            format!("{CALL_OPEN}{raw}{CALL_CLOSE}")
        };
        match parse_call(&text, Some(&toolbox)) {
            Some(call) if call.ok() => (call.name, call.arguments),
            Some(call) => return Err(ApiError::bad_request(call.error.unwrap_or_default())),
            None => {
                let shown: String = raw.chars().take(120).collect();
                return Err(ApiError::bad_request(format!(
                    "no tool call in {}",
                    python_repr(&shown)
                )));
            }
        }
    } else {
        let name = text_field(r, "tool")?
            .ok_or_else(|| ApiError::bad_request("missing field 'tool' (string)"))?
            .trim()
            .to_string();
        let arguments = match r.body.get("arguments") {
            None | Some(Json::Null) => empty(),
            Some(Json::Obj(pairs)) if pairs.is_empty() => empty(),
            Some(object @ Json::Obj(_)) => dedupe(object.clone()),
            // Python reads any falsy value as "no arguments"
            Some(Json::Bool(false)) | Some(Json::Int(0)) => empty(),
            Some(Json::Str(s)) if s.is_empty() => empty(),
            Some(Json::Arr(items)) if items.is_empty() => empty(),
            Some(_) => return Err(ApiError::bad_request("'arguments' must be an object")),
        };
        (name, arguments)
    };
    if !toolbox.has(&name) {
        return Err(ApiError::not_found(format!(
            "unknown tool {} (have: {})",
            python_repr(&name),
            toolbox.names().join(", ")
        )));
    }
    // no tool touches the model, so the model lock is never taken
    let result = toolbox.call(&name, &arguments);
    crate::log_info!(
        LOG,
        "{name}: {} in {:.3}s",
        if result.ok { "ok" } else { "failed" },
        result.seconds
    );
    Ok(result.to_json())
}

/// This area's routes:
/// `GET /api/tools`
/// `POST /api/tools/call`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/tools", tools_route);
    server.route("POST", "/api/tools/call", call_route);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parse_args;
    use crate::toolbox::ToolOptions;

    fn offline() -> ToolBox {
        default_toolbox(ToolOptions {
            offline: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn obj(text: &str) -> Json {
        parse(text).unwrap()
    }

    fn demo() -> Tool {
        let mut mode = Param::optional("mode", "string", "which mode", Json::Null);
        mode.choices = Some(vec!["a".into(), "b".into()]);
        Tool::new(
            "demo",
            "A demo.",
            vec![
                Param::new("text", "string", "some text"),
                Param::optional("count", "integer", "how many", Json::Int(2)),
                Param::optional("flag", "boolean", "on or off", Json::Null),
                mode,
            ],
            |args| Ok((args.at("text").as_str().unwrap_or("").repeat(2), Json::Null)),
        )
    }

    #[test]
    fn the_text_format_is_python_s() {
        assert_eq!(
            call_text("web_fetch", &obj("{\"url\": \"u\"}")),
            "<tool>web_fetch {\"url\": \"u\"}</tool>\n"
        );
        assert_eq!(call_text("now", &Json::Null), "<tool>now {}</tool>\n");
        assert_eq!(
            call_text("t", &obj("{\"b\": [1, 2.5, \"ü\"], \"a\": {\"z\": true, \"y\": null}}")),
            "<tool>t {\"a\": {\"y\": null, \"z\": true}, \"b\": [1, 2.5, \"ü\"]}</tool>\n",
            "sorted keys, Python's spacing, UTF-8 kept"
        );
        assert_eq!(result_text("a\n  b\tc", Some(600)), "<result>a b c</result>\n");
        assert_eq!(
            result_text(&"x".repeat(50), Some(10)),
            format!("<result>{} ...</result>\n", "x".repeat(10))
        );
        assert_eq!(
            result_text(&"x".repeat(50), None),
            format!("<result>{}</result>\n", "x".repeat(50))
        );
        assert_eq!(answer_text(" four\nlegs "), "<answer>four legs</answer>\n");
        assert_eq!(task_header("  how   many?\n"), "TASK: how many?\n");
        assert_eq!(find_answer("noise <answer> 42 </answer> more").as_deref(), Some("42"));
        assert_eq!(find_answer("<answer>truncated").as_deref(), Some("truncated"));
        assert_eq!(find_answer("nothing here"), None);
        assert_eq!(find_answer("<answer>   </answer>"), None);
        let ok = ToolResult {
            tool: "calculator".into(),
            arguments: empty(),
            ok: true,
            output: "2".into(),
            error: None,
            seconds: 0.0,
            meta: empty(),
        };
        let step = TranscriptStep {
            name: "calculator".into(),
            arguments: obj("{\"expression\": \"1+1\"}"),
            result: ok,
        };
        assert_eq!(
            transcript_text("What is 1+1?", &[step], Some("two"), Some(DEFAULT_OBSERVATION_CHARS)),
            "TASK: What is 1+1?\n<tool>calculator {\"expression\": \"1+1\"}</tool>\n<result>2</result>\n<answer>two</answer>\n"
        );
        assert_eq!(transcript_text("q", &[], None, None), "TASK: q\n");
        let bad = ToolResult::failed("t", empty(), "boom".into(), 0.0);
        assert_eq!(format_observation(&bad, Some(600)), "<result>ERROR: boom</result>\n");
    }

    #[test]
    fn the_reader_is_lenient() {
        let bx = offline();
        for (text, want) in [
            ("<tool>calculator {\"expression\": \"1+1\"}</tool>", "1+1"),
            ("<tool>calculator 5*5</tool>", "5*5"),
            ("<tool>calculator expression=2+2</tool>", "2+2"),
            ("<tool>calculator {\"expression\": \"9\"}", "9"),
            ("<tool>calculator {\"expression\": \"9\"", "9"),
            ("<tool>calculator expression: '3'</tool>", "3"),
        ] {
            let call = parse_call(text, Some(&bx)).unwrap();
            assert!(call.ok(), "{text}: {call:?}");
            assert_eq!(call.name, "calculator");
            assert_eq!(call.arguments.at("expression").as_str(), Some(want), "{text}");
        }
        assert_eq!(parse_call("just some text", Some(&bx)), None);
        let first = parse_call("<tool>calculator 1</tool><tool>calculator 2</tool>", Some(&bx)).unwrap();
        assert_eq!(first.arguments.at("expression").as_str(), Some("1"));
        assert_eq!(first.span, (0, 25));
        let unknown = parse_call("<tool>nosuchtool {}</tool>", Some(&bx)).unwrap();
        assert_eq!(
            unknown.error.as_deref(),
            Some("unknown tool 'nosuchtool' (have: calculator)")
        );
        assert_eq!(unknown.name, "nosuchtool");
        let nameless = parse_call("<tool>{}</tool>", Some(&bx)).unwrap();
        assert_eq!(nameless.error.as_deref(), Some("no tool name in '{}'"));
        assert_eq!(
            parse_arguments("{\"a\": \"b\"} trailing junk", None).unwrap(),
            obj("{\"a\": \"b\"}")
        );
        assert_eq!(
            parse_arguments("{\"url\": \"http://example.com", None).unwrap(),
            obj("{\"url\": \"http://example.com\"}")
        );
        assert_eq!(
            parse_arguments("a=1, b = \"x, y\"; c:3", None).unwrap(),
            obj("{\"a\": \"1\", \"b\": \"x, y\", \"c\": \"3\"}")
        );
        assert_eq!(
            parse_arguments("a= ,b=2", None).unwrap(),
            obj("{\"a\": \"\", \"b\": \"2\"}")
        );
        assert_eq!(
            parse_arguments("x y=1 z = \"q", None).unwrap(),
            obj("{\"y\": \"1 z = \\\"q\"}")
        );
        assert_eq!(
            parse_arguments("???", None).unwrap_err(),
            "cannot read the arguments of the call: '???'"
        );
        let pair = Tool::new(
            "pair",
            "Two.",
            vec![Param::new("a", "string", ""), Param::new("b", "string", "")],
            |_| Ok(("".into(), Json::Null)),
        );
        assert!(
            parse_arguments("???", Some(&pair)).is_err(),
            "a bare value needs a one-argument tool"
        );
        assert_eq!(
            parse_arguments("[1, 2, 3]", Some(&pair)).unwrap(),
            obj("{\"a\": 1, \"b\": 2}")
        );
    }

    #[test]
    fn a_tool_coerces_its_arguments() {
        let tool = demo();
        assert_eq!(
            tool.signature(),
            "demo(text: string, [count: integer], [flag: boolean], [mode: string])"
        );
        let schema = tool.schema();
        assert_eq!(
            schema.at("function").at("parameters").at("required"),
            &Json::strs(["text"])
        );
        assert_eq!(
            schema
                .at("function")
                .at("parameters")
                .at("properties")
                .at("mode")
                .at("enum"),
            &Json::strs(["a", "b"])
        );
        let values = tool
            .coerce(&obj(
                "{\"text\": 5, \"count\": \"3\", \"flag\": \"yes\", \"MODE\": \"b\"}",
            ))
            .unwrap();
        assert_eq!(
            values,
            obj("{\"text\": \"5\", \"count\": 3, \"flag\": true, \"mode\": \"b\"}")
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": \"x\"}")).unwrap(),
            obj("{\"text\": \"x\", \"count\": 2}")
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": {\"k\": \"é\"}}"))
                .unwrap()
                .at("text")
                .as_str(),
            Some("{\"k\": \"\\u00e9\"}"),
            "an object is json.dumps'd, ASCII only"
        );
        assert_eq!(
            tool.coerce(&empty()).unwrap_err(),
            "demo: missing argument(s) text — demo(text: string, [count: integer], [flag: boolean], [mode: string])"
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": \"x\", \"count\": \"many\"}")).unwrap_err(),
            "argument 'count' must be a integer (got 'many')"
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": \"x\", \"mode\": \"c\"}")).unwrap_err(),
            "argument 'mode' must be one of a, b (got 'c')"
        );
        assert_eq!(
            tool.coerce(&obj("[\"not\", \"a\", \"dict\"]")).unwrap_err(),
            "demo: arguments must be an object (got list)"
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": \"x\", \"nonsense\": 1}")).unwrap(),
            obj("{\"text\": \"x\", \"count\": 2}")
        );
        assert_eq!(
            tool.coerce(&obj("{\"text\": \"x\", \"count\": \"1_000\"}"))
                .unwrap()
                .at("count"),
            &Json::Int(1000)
        );
    }

    #[test]
    fn the_registry_never_raises() {
        let mut bx = ToolBox::new();
        bx.register(Tool::new(
            "echo",
            "Echo.",
            vec![Param::new("text", "string", "")],
            |args| Ok((args.at("text").as_str().unwrap_or("").to_uppercase(), Json::Null)),
        ))
        .unwrap();
        bx.register(Tool::new("boom", "Fails.", vec![], |_| Err("kaboom".to_string())))
            .unwrap();
        assert!(bx
            .register(Tool::new("bad name!", "x", vec![], |_| Ok(("".into(), Json::Null))))
            .is_err());
        let mut odd = Tool::new("odd", "x", vec![Param::new("x", "list", "")], |_| {
            Ok(("".into(), Json::Null))
        });
        assert!(bx.register(odd.clone()).is_err());
        odd.handler = None;
        odd.params.clear();
        bx.register(odd).unwrap();
        assert_eq!(bx.names(), ["echo", "boom", "odd"]);
        assert!(bx.catalogue().starts_with("- echo(text: string) — Echo."));
        let result = bx.call("echo", &obj("{\"text\": \"hi\"}"));
        assert!(result.ok && result.output == "HI");
        assert_eq!(result.to_json().at("error"), &Json::Null);
        assert_eq!(bx.call("boom", &empty()).error.as_deref(), Some("kaboom"));
        assert!(bx
            .call("missing", &Json::Null)
            .error
            .unwrap()
            .starts_with("unknown tool 'missing'"));
        assert!(bx.call("echo", &empty()).error.unwrap().contains("missing argument(s)"));
        assert_eq!(
            bx.call("odd", &empty()).error.as_deref(),
            Some("odd: no handler is installed")
        );
        assert_eq!(bx.calls(), 5);
        let broken = parse_call("<tool>unknown x</tool>", Some(&bx)).unwrap();
        assert!(bx.run(&broken).error.unwrap().contains("unknown tool"));
        assert!(bx.remove("odd"));
        assert!(!bx.remove("odd"));
    }

    #[test]
    fn the_flags_and_the_request_set_the_tools() {
        let (_, args) = parse_args(&[
            "tools".to_string(),
            "--offline".to_string(),
            "--web-timeout".to_string(),
            "5".to_string(),
        ])
        .unwrap();
        let mut state = State::default();
        state.configure(&args).unwrap();
        assert!(state.offline);
        assert_eq!(state.web_timeout, 5.0);
        assert_eq!(state.toolbox(None).unwrap().names(), ["calculator"]);
        let (_, bad) = parse_args(&["tools".to_string(), "--max-bytes".to_string(), "10".to_string()]).unwrap();
        assert_eq!(
            State::default().configure(&bad).unwrap_err(),
            "argument --max-bytes: must be >= 1024, got 10"
        );
        let (_, bad) = parse_args(&["tools".to_string(), "--web-timeout=0".to_string()]).unwrap();
        assert!(State::default().configure(&bad).unwrap_err().contains(">= 0.1"));
        let names = State::default().toolbox(Some("uploads")).unwrap().names();
        assert_eq!(
            names,
            ["web_search", "web_fetch", "web_links", "calculator", "read_file"]
        );
        assert_eq!(
            State::default().options_json().render(0),
            "{\"offline\":false,\"allow_private\":false,\"search_url\":null,\"web_timeout\":20.0,\
             \"max_bytes\":2000000,\"python_tool\":false,\"sandbox_timeout\":10.0,\"browser\":false,\
             \"page_timeout\":30.0}"
        );
    }

    #[test]
    fn the_routes_answer_the_frontend_s_json() {
        let model = crate::Model::new(0, Default::default()).unwrap();
        let mut service = Service::new(model, String::new(), 0, 1);
        service.tools.allow_private = true;
        let svc = Arc::new(service);
        let doc = tools_route(&svc, &Request::json("GET", "/api/tools", Json::Null)).unwrap();
        assert_eq!(
            doc.at("names"),
            &Json::strs(["web_search", "web_fetch", "web_links", "calculator"])
        );
        let status = status_json(&svc);
        assert_eq!(
            status.at("names"),
            &Json::strs(["calculator", "web_fetch", "web_links", "web_search"])
        );
        assert_eq!(status.at("allow_private"), &Json::Bool(true));
        assert_eq!(doc.at("tools").as_array()[0].at("name").as_str(), Some("web_search"));
        assert!(doc.at("call_format").as_str().unwrap().contains("<tool>"));
        assert_eq!(doc.at("options").at("allow_private"), &Json::Bool(true));
        let call = |body: &str| call_route(&svc, &Request::json("POST", "/api/tools/call", obj(body)));
        let result = call("{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"6*7\"}}").unwrap();
        assert_eq!(
            (result.at("ok"), result.at("output").as_str()),
            (&Json::Bool(true), Some("42"))
        );
        let result = call("{\"call\": \"calculator {\\\"expression\\\": \\\"2+3\\\"}\"}").unwrap();
        assert_eq!(result.at("output").as_str(), Some("5"));
        assert_eq!(call("{\"tool\": \"nope\"}").unwrap_err().status, 404);
        assert_eq!(call("{\"call\": \"not a call\"}").unwrap_err().status, 400);
        assert_eq!(
            call("{\"tool\": \"calculator\", \"arguments\": \"text\"}")
                .unwrap_err()
                .status,
            400
        );
        let failed = call("{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"1/0\"}}").unwrap();
        assert_eq!(failed.at("ok"), &Json::Bool(false));
        assert_eq!(failed.at("error").as_str(), Some("division by zero"));
        let off = call("{\"tool\": \"web_fetch\", \"arguments\": {\"url\": \"http://x/\"}, \"offline\": true}");
        assert_eq!(off.unwrap_err().status, 404, "browsing is off for this request");
        let err = call("{\"tool\": \"calculator\", \"max_bytes\": 10}").unwrap_err();
        assert_eq!(err.message, "'max_bytes' must be >= 1024 (got 10)");
        let err = call("{\"tool\": \"calculator\", \"offline\": \"yes\"}").unwrap_err();
        assert_eq!(err.message, "'offline' must be a boolean (got string 'yes')");
        assert_eq!(call("{}").unwrap_err().message, "missing field 'tool' (string)");
    }
}
