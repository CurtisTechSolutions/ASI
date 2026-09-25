//! Provider-independent plumbing for the large language models the teaching
//! loops talk to (`radixnet/llm.py`, `go/radixnet/llm.go`).
//!
//! Everything that asks an LLM a question - the adversarial review, the
//! automatic negative loop, and the tutor, chat, codegen and agent loops that
//! build on this module - needs the same three things of a provider: a list of
//! models, one completion and one chat turn.  Two providers offer them:
//!
//! * `ollama` - a model served locally by Ollama ([`crate::ollama`]): no key,
//!   no cost, nothing leaves the machine.
//! * `chatgpt` - OpenAI's hosted models, or any OpenAI-compatible server
//!   ([`crate::chatgpt`]): stronger, but it needs `OPENAI_API_KEY` and every
//!   prompt is sent to the provider.
//!
//! [`LlmClient`] is the shape both clients share, [`new_client`] builds one
//! from a provider name, and [`LlmError`] is the failure both return, so a
//! caller holding an `LlmError` knows "the LLM did not work" whichever
//! provider it was.  A *configuration* mistake - an unknown provider, a URL
//! that is not one - is a plain `String` error instead, because it is the
//! caller's to fix and no provider was ever asked.
//!
//! # Why the answers are parsed leniently
//!
//! A model asked for JSON writes JSON inside a code fence, or after a
//! sentence of preamble, or as a bare list when an object was asked for.
//! [`loads_lenient`] and [`parse_lines`] are the two readers every loop uses to
//! get data back out of an answer, exactly as the Python and Go ports read it,
//! so the three agree on what a reviewer said.
//!
//! # The server's defaults
//!
//! [`State`] is what `radixnet serve` keeps for this area: the Ollama and
//! ChatGPT endpoints and models a request falls back to when it names none,
//! read from `--ollama-url`, `--ollama-model`, `--chatgpt-url` and
//! `--chatgpt-model`, else from the environment the way the other two ports
//! read it (`OLLAMA_HOST`, `RADIXNET_OLLAMA_MODEL`, `OPENAI_BASE_URL` /
//! `OPENAI_API_BASE`, `RADIXNET_OPENAI_MODEL`; `RADIXNET_TUTOR_MODEL` for the
//! Ollama teacher).  The API key is never a flag, a field or a default: it is
//! read from `OPENAI_API_KEY` (or the file `OPENAI_API_KEY_FILE` names) at the
//! moment a request is sent.

pub mod fields;

use std::time::Duration;

use crate::chatgpt::ChatGptClient;
use crate::http::ApiError;
use crate::json::{parse, Json};
use crate::ollama::OllamaClient;

/// What this module's own lines are filed under.
const LOG: &str = "llm";

/// The local provider.
pub const OLLAMA: &str = "ollama";
/// OpenAI's hosted models (or any OpenAI-compatible server).
pub const CHATGPT: &str = "chatgpt";
/// Every provider that can answer as a reviewer, a tutor or a judge.
pub const PROVIDERS: &[&str] = &[OLLAMA, CHATGPT];
/// The provider chosen when nothing else is asked for: the one that needs no key.
pub const DEFAULT_PROVIDER: &str = OLLAMA;

/// `"OpenAI"` / `"gpt"` -> `"chatgpt"`, `""` -> the default provider; an
/// unknown name is an error that lists the ones there are.
pub fn normalise_provider(name: &str) -> Result<&'static str, String> {
    let text: String = name.trim().to_lowercase().replace(' ', "");
    match text.as_str() {
        "" => Ok(DEFAULT_PROVIDER),
        "ollama" | "local" => Ok(OLLAMA),
        "chatgpt" | "openai" | "gpt" | "chat-gpt" | "chat_gpt" => Ok(CHATGPT),
        _ => Err(format!(
            "provider must be one of {} (got {})",
            PROVIDERS.join(", "),
            crate::negative::python_repr(name)
        )),
    }
}

/// The model a provider answers with when none is named.
pub fn default_model(provider: &str) -> String {
    match normalise_provider(provider) {
        Ok(CHATGPT) => crate::chatgpt::default_model(),
        _ => crate::ollama::default_model(),
    }
}

/// The endpoint a provider is reached at when none is given.
pub fn default_url(provider: &str) -> String {
    match normalise_provider(provider) {
        Ok(CHATGPT) => crate::chatgpt::default_url(),
        _ => crate::ollama::default_url(),
    }
}

/// An LLM that is unreachable, refused the request, or answered something
/// unusable - `radixnet.llm.LLMError`, whichever provider raised it.
#[derive(Clone, Debug, PartialEq)]
pub struct LlmError {
    /// `"ollama"` or `"chatgpt"`.
    pub provider: &'static str,
    pub message: String,
    /// The HTTP status the provider answered with, when it answered at all.
    pub status: Option<u16>,
    /// OpenAI's `error.code` and `error.param`, which say which request field
    /// a model refused (the ChatGPT client drops it and asks again).
    pub code: Option<String>,
    pub param: Option<String>,
    /// The provider did not answer in time (rather than refusing or failing).
    pub timeout: bool,
}

impl LlmError {
    /// A failure with nothing but a message.
    pub fn new<S: Into<String>>(provider: &'static str, message: S) -> LlmError {
        LlmError {
            provider,
            message: message.into(),
            status: None,
            code: None,
            param: None,
            timeout: false,
        }
    }
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LlmError {}

impl From<LlmError> for String {
    fn from(err: LlmError) -> String {
        err.message
    }
}

impl From<LlmError> for ApiError {
    /// An upstream failure: the request was fine, the provider was not (502),
    /// which is how the Python and Go servers answer it.
    fn from(err: LlmError) -> ApiError {
        ApiError::with_status(502, err.message)
    }
}

/// The knobs of one completion or chat turn, whichever provider answers it.
///
/// ```
/// use radixnet::llm::LlmOptions;
/// let o = LlmOptions::default().system("be brief").temperature(0.2).json();
/// assert_eq!(o.system, "be brief");
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LlmOptions {
    /// The system instruction (`""` = none).
    pub system: String,
    /// The model to answer with (`""` = the client's own).
    pub model: String,
    /// Ask for a JSON answer: Ollama's `format: "json"`, OpenAI's
    /// `response_format: {"type": "json_object"}`.
    pub json: bool,
    /// Sampling knobs under Ollama's names (`temperature`, `num_predict`,
    /// `top_p`, `seed`, `stop`, ...), in the order given.  Ollama receives
    /// them as its `options`; the ChatGPT client translates the ones the
    /// chat-completions API has and drops the rest, as Python's does.
    pub options: Vec<(String, Json)>,
    /// How long this one answer may take (`None` = the client's own timeout).
    pub timeout: Option<Duration>,
    /// Ollama's own switch for a thinking model's reasoning: `true` / `false`,
    /// or a level (`"low"`, `"medium"`, `"high"`); `None` leaves the choice to
    /// the model.  The ChatGPT client ignores it.
    pub think: Option<Json>,
}

impl LlmOptions {
    /// The system instruction.
    pub fn system<S: Into<String>>(mut self, system: S) -> LlmOptions {
        self.system = system.into();
        self
    }
    /// A model other than the client's own.
    pub fn model<S: Into<String>>(mut self, model: S) -> LlmOptions {
        self.model = model.into();
        self
    }
    /// Ask for a JSON answer.
    pub fn json(mut self) -> LlmOptions {
        self.json = true;
        self
    }
    /// The sampling temperature (replacing one already set).
    pub fn temperature(self, temperature: f64) -> LlmOptions {
        self.option("temperature", Json::Num(temperature))
    }
    /// Ask a thinking model for its reasoning (Ollama's `think`).
    pub fn think(mut self, value: Json) -> LlmOptions {
        self.think = Some(value);
        self
    }
    /// One sampling knob under its Ollama name (replacing one already set).
    pub fn option(mut self, name: &str, value: Json) -> LlmOptions {
        match self.options.iter_mut().find(|(k, _)| k == name) {
            Some(slot) => slot.1 = value,
            None => self.options.push((name.to_string(), value)),
        }
        self
    }
    /// A timeout for this one answer.
    pub fn timeout(mut self, timeout: Duration) -> LlmOptions {
        self.timeout = Some(timeout);
        self
    }
    /// The model this call asks for: its own, else `fallback` (the client's).
    pub fn model_or<'a>(&'a self, fallback: &'a str) -> &'a str {
        let own = self.model.trim();
        if own.is_empty() {
            fallback
        } else {
            own
        }
    }
}

/// What the rest of the crate needs of a provider.
///
/// [`OllamaClient`] and [`ChatGptClient`] both implement it, which is why the
/// review, the critic and the tutors never look at which one they got.
/// Messages are `{"role", "content"}` objects ([`message`] builds one), the
/// shape both chat APIs take.
pub trait LlmClient: Send + Sync {
    /// `"ollama"` or `"chatgpt"`.
    fn provider(&self) -> &'static str;
    /// The endpoint it talks to.
    fn url(&self) -> &str;
    /// The model it answers with when a call names none.
    fn model(&self) -> &str;
    /// The models it can answer with, `{"name", ...}` each.
    fn models(&self) -> Result<Vec<Json>, LlmError>;
    /// Whether it answers right now; never fails.
    fn available(&self) -> bool {
        self.models().is_ok()
    }
    /// One completion of `prompt` (with `o.system` as the instruction).
    fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError>;
    /// One completion and the thinking behind it, `(answer, thinking)`: a
    /// client that can say what it thought (Ollama, with `o.think` as the
    /// level) gives both; every other answers as `generate` does and thought
    /// nothing anyone can read.
    fn complete_thinking(&self, prompt: &str, o: &LlmOptions) -> Result<(String, String), LlmError> {
        Ok((self.generate(prompt, o)?, String::new()))
    }
    /// One chat turn: the assistant's text.
    fn chat(&self, messages: &[Json], o: &LlmOptions) -> Result<String, LlmError>;
}

/// One chat message, `{"role": role, "content": content}`.
pub fn message(role: &str, content: &str) -> Json {
    Json::obj([("role", Json::str(role)), ("content", Json::str(content))])
}

/// A client for `provider`; an empty `url` or `model`, or no `timeout`, takes
/// that provider's default.  The ChatGPT key is read from the environment when
/// a request is sent, never here.
pub fn new_client(
    provider: &str,
    url: &str,
    model: &str,
    timeout: Option<Duration>,
) -> Result<Box<dyn LlmClient>, String> {
    let client: Box<dyn LlmClient> = match normalise_provider(provider)? {
        CHATGPT => Box::new(ChatGptClient::new(url, model, timeout)?),
        _ => Box::new(OllamaClient::new(url, model, timeout)?),
    };
    crate::log_debug!(
        LOG,
        "{} client: {} at {}",
        client.provider(),
        client.model(),
        client.url()
    );
    Ok(client)
}

/// A timeout in seconds as the CLI and the API give it (`0` = the client's own).
pub fn seconds(value: f64) -> Option<Duration> {
    if value.is_finite() && value > 0.0 {
        Some(Duration::from_secs_f64(value))
    } else {
        None
    }
}

// -- reading an answer --------------------------------------------------------------------------

/// JSON out of an LLM answer: tolerates a code fence and prose around the
/// object or list (`radixnet.llm.loads_lenient`); `None` when there is none.
pub fn loads_lenient(raw: &str) -> Option<Json> {
    let mut text = raw.trim();
    let unfenced;
    if text.starts_with("```") {
        let mut inner = text.trim_matches('`');
        if inner.get(..4).is_some_and(|head| head.eq_ignore_ascii_case("json")) {
            inner = &inner[4..];
        }
        unfenced = inner.trim().to_string();
        text = &unfenced;
    }
    if let Ok(value) = parse(text) {
        return Some(value);
    }
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(start), Some(end)) = (text.find(open), text.rfind(close)) {
            if start < end {
                if let Ok(value) = parse(&text[start..=end]) {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// Lines as Python's `str.splitlines()` finds them: every line break Python
/// knows (`\n`, `\r\n`, `\r`, and the Unicode separators), and no empty line
/// after a final break.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, ch)) = chars.next() {
        let breaks = matches!(
            ch,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        if !breaks {
            continue;
        }
        lines.push(&text[start..at]);
        let mut next = at + ch.len_utf8();
        if ch == '\r' {
            if let Some(&(_, '\n')) = chars.peek() {
                chars.next();
                next += 1;
            }
        }
        start = next;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// The numbering or bullet an LLM puts in front of a line: `- `, `* `, `• `,
/// `1. `, `2) `, `(3) `, `4: `, `5 - ` - the regex
/// `^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*` of the Python and Go ports.
fn line_prefix_len(line: &str) -> usize {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let end = |i: usize| chars.get(i).map(|(at, _)| *at).unwrap_or(line.len());
    let mut i = 0;
    while i < chars.len() && chars[i].1.is_whitespace() {
        i += 1;
    }
    let digits_from = |mut j: usize| {
        let from = j;
        while j < chars.len() && chars[j].1.is_ascii_digit() {
            j += 1;
        }
        (j > from).then_some(j)
    };
    // the three alternatives, in the order the regex tries them
    let matched = {
        let mut j = i;
        while j < chars.len() && matches!(chars[j].1, '-' | '*' | '•') {
            j += 1;
        }
        if j > i {
            Some(j)
        } else {
            // `\(?` matching nothing cannot help when there is a `(`: the digits
            // would have to start on it
            let open = usize::from(chars.get(i).map(|c| c.1) == Some('('));
            let numbered = digits_from(i + open)
                .filter(|&j| matches!(chars.get(j).map(|c| c.1), Some('.' | ')' | ':')))
                .map(|j| j + 1);
            numbered.or_else(|| {
                digits_from(i).and_then(|mut j| {
                    while j < chars.len() && chars[j].1.is_whitespace() {
                        j += 1;
                    }
                    (chars.get(j).map(|c| c.1) == Some('-')).then_some(j + 1)
                })
            })
        }
    };
    let Some(mut j) = matched else { return 0 };
    while j < chars.len() && chars[j].1.is_whitespace() {
        j += 1;
    }
    end(j)
}

/// An LLM answer as clean lines (`radixnet.ollama.parse_lines`): numbering,
/// bullets and quotes stripped; blank lines, code fences and duplicates (by
/// their lower case) dropped; at most `limit` of them when one is given.
pub fn parse_lines(text: &str, limit: Option<usize>) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for raw in splitlines(text) {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("```") {
            continue;
        }
        let line = line[line_prefix_len(line)..]
            .trim()
            .trim_matches(|c| matches!(c, '"' | '\'' | '`' | '“' | '”'))
            .trim();
        if line.is_empty() {
            continue;
        }
        let key = line.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        lines.push(line.to_string());
        if limit.is_some_and(|n| lines.len() >= n) {
            break;
        }
    }
    lines
}

/// A number as Python's `format(x, "g")` writes it - six significant digits,
/// no trailing zeros, an exponent outside `[1e-4, 1e6)` - which is how the
/// prompts print a pass mark (`6`, `6.5`), so they match the other two ports
/// byte for byte.
pub fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let strip = |text: &str| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            text.to_string()
        }
    };
    let scientific = format!("{x:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("{:e} writes an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} writes a decimal exponent");
    if (-4..6).contains(&exponent) {
        strip(&format!("{:.*}", (5 - exponent) as usize, x))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exponent.abs())
    }
}

/// An environment variable, trimmed; `""` when unset.
pub(crate) fn env(name: &str) -> String {
    std::env::var(name).map(|v| v.trim().to_string()).unwrap_or_default()
}

/// The reason phrase of an HTTP status, for an error answer that came without
/// a body to quote.
pub(crate) fn http_reason(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

/// The first `limit` characters of a text (Python's `text[:limit]`).
pub(crate) fn clip_chars(text: &str, limit: usize) -> &str {
    match text.char_indices().nth(limit) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

// -- the command line's LLM flags --------------------------------------------------------------

/// `--timeout` of an LLM command, which has to leave a model time to answer
/// at all (Python refuses less than a second); `None` when it is not given.
pub(crate) fn timeout_flag(ctx: &crate::cli::Ctx) -> Result<Option<Duration>, String> {
    match ctx.args.maybe_float("timeout")? {
        Some(t) if t.is_nan() || t < 1.0 => Err(format!("--timeout must be at least 1 second, got {t}")),
        Some(t) => Ok(seconds(t)),
        None => Ok(None),
    }
}

/// A count flag that must be at least `minimum` - `--count 0` is refused
/// rather than read as "until stopped" or "none".
pub(crate) fn count_flag(ctx: &crate::cli::Ctx, name: &str, default: i64, minimum: i64) -> Result<usize, String> {
    let value = ctx.args.int(name, default)?;
    if value < minimum {
        return Err(format!("--{name} must be >= {minimum}, got {value}"));
    }
    Ok(value as usize)
}

/// A number flag that must be finite and not negative.
pub(crate) fn nonneg_flag(ctx: &crate::cli::Ctx, name: &str, default: f64) -> Result<f64, String> {
    let value = ctx.args.float(name, default)?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("--{name} must be a number >= 0, got {value}"));
    }
    Ok(value)
}

/// `{"path", "bytes"}` of a file just saved, as the Python CLI reports one.
pub(crate) fn saved(path: &str) -> Json {
    let bytes = std::fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
    Json::obj([("path", Json::str(path)), ("bytes", Json::Int(bytes))])
}

// -- the server's defaults ----------------------------------------------------------------------

/// What the server keeps for this area: the teacher defaults a request falls
/// back to when it names no endpoint or model.
#[derive(Clone, Debug)]
pub struct State {
    /// `--ollama-url`, else `$OLLAMA_HOST`, else `http://127.0.0.1:11434`.
    pub ollama_url: String,
    /// `--ollama-model`, else `$RADIXNET_OLLAMA_MODEL`, else `llama3.2`.
    pub ollama_model: String,
    /// `--chatgpt-url`, else `$OPENAI_BASE_URL` (or `$OPENAI_API_BASE`), else
    /// `https://api.openai.com/v1`.
    pub chatgpt_url: String,
    /// `--chatgpt-model`, else `$RADIXNET_OPENAI_MODEL`, else `gpt-4o-mini`.
    pub chatgpt_model: String,
}

impl Default for State {
    /// The environment's defaults, before the flags are read.
    fn default() -> State {
        State {
            ollama_url: crate::ollama::default_url(),
            ollama_model: crate::ollama::default_model(),
            chatgpt_url: crate::chatgpt::default_url(),
            chatgpt_model: crate::chatgpt::default_model(),
        }
    }
}

impl State {
    /// Reads the `serve` command's flags for this area.  A URL that is not one
    /// stops the server from starting, as it does the Python one.
    pub fn configure(&mut self, args: &crate::cli::Args) -> Result<(), String> {
        let given = |name: &str| args.get(name).map(str::trim).filter(|v| !v.is_empty());
        if let Some(url) = given("ollama-url") {
            self.ollama_url = crate::ollama::normalise_url(url)?;
        }
        if let Some(model) = given("ollama-model") {
            self.ollama_model = model.to_string();
        }
        if let Some(url) = given("chatgpt-url") {
            self.chatgpt_url = crate::chatgpt::normalise_url(url)?;
        }
        if let Some(model) = given("chatgpt-model") {
            self.chatgpt_model = model.to_string();
        }
        crate::log_info!(
            LOG,
            "ollama: {} at {}; chatgpt: {} at {} ({})",
            self.ollama_model,
            self.ollama_url,
            self.chatgpt_model,
            self.chatgpt_url,
            if crate::chatgpt::key_configured() {
                "key set"
            } else {
                "no OPENAI_API_KEY: ChatGPT is off"
            }
        );
        Ok(())
    }

    /// An Ollama client for a request's overrides (`None` or `""` falls back
    /// to the server's own); a URL that is not one is a 400.
    pub fn ollama(
        &self,
        url: Option<&str>,
        model: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<OllamaClient, ApiError> {
        let url = url.filter(|u| !u.is_empty()).unwrap_or(&self.ollama_url);
        let model = model.filter(|m| !m.trim().is_empty()).unwrap_or(&self.ollama_model);
        OllamaClient::new(url, model, timeout).map_err(ApiError::bad_request)
    }

    /// A ChatGPT client for a request's overrides; the key is always the
    /// server's own, never a request field.
    pub fn chatgpt(
        &self,
        url: Option<&str>,
        model: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<ChatGptClient, ApiError> {
        let url = url.filter(|u| !u.is_empty()).unwrap_or(&self.chatgpt_url);
        let model = model.filter(|m| !m.trim().is_empty()).unwrap_or(&self.chatgpt_model);
        ChatGptClient::new(url, model, timeout).map_err(ApiError::bad_request)
    }

    /// A client for `provider` (`""` = the default); a ChatGPT one is refused
    /// with a 400 before anything is sent when this server has no key.
    pub fn client(
        &self,
        provider: &str,
        url: Option<&str>,
        model: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Box<dyn LlmClient>, ApiError> {
        match normalise_provider(provider).map_err(ApiError::bad_request)? {
            CHATGPT => {
                if !crate::chatgpt::key_configured() {
                    return Err(ApiError::bad_request(
                        "ChatGPT is not configured on this server: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) in \
                         its environment and restart it, or use the 'ollama' provider",
                    ));
                }
                Ok(Box::new(self.chatgpt(url, model, timeout)?))
            }
            _ => Ok(Box::new(self.ollama(url, model, timeout)?)),
        }
    }

    /// The model this server teaches with on a provider: the ChatGPT default,
    /// or `$RADIXNET_TUTOR_MODEL` and then the Ollama default.
    pub fn tutor_model(&self, provider: &str) -> String {
        if normalise_provider(provider) == Ok(CHATGPT) {
            return self.chatgpt_model.clone();
        }
        let tutor = env("RADIXNET_TUTOR_MODEL");
        if tutor.is_empty() {
            self.ollama_model.clone()
        } else {
            tutor
        }
    }

    /// What `/api/status` says about the providers: `ollama: {url, model}` and
    /// `chatgpt: {url, model, configured}` - which the Ollama tab reads its
    /// defaults from.
    pub fn status_fields(&self) -> Vec<(String, Json)> {
        vec![
            (
                "ollama".to_string(),
                Json::obj([
                    ("url", Json::str(self.ollama_url.clone())),
                    ("model", Json::str(self.ollama_model.clone())),
                ]),
            ),
            (
                "chatgpt".to_string(),
                Json::obj([
                    ("url", Json::str(self.chatgpt_url.clone())),
                    ("model", Json::str(self.chatgpt_model.clone())),
                    ("configured", Json::Bool(crate::chatgpt::key_configured())),
                ]),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_are_named_the_way_people_name_them() {
        assert_eq!(normalise_provider("OpenAI"), Ok(CHATGPT));
        assert_eq!(normalise_provider(" gpt "), Ok(CHATGPT));
        assert_eq!(normalise_provider("Chat GPT"), Ok(CHATGPT));
        assert_eq!(normalise_provider(""), Ok(OLLAMA));
        assert_eq!(normalise_provider("local"), Ok(OLLAMA));
        let err = normalise_provider("bard").unwrap_err();
        assert_eq!(err, "provider must be one of ollama, chatgpt (got 'bard')");
        assert!(new_client("bard", "", "", None).is_err());
        let client = new_client("chatgpt", "http://127.0.0.1:9", "m", None).unwrap();
        assert_eq!(
            (client.provider(), client.url(), client.model()),
            ("chatgpt", "http://127.0.0.1:9/v1", "m")
        );
        let client = new_client("", "127.0.0.1:9", "", None).unwrap();
        assert_eq!((client.provider(), client.url()), ("ollama", "http://127.0.0.1:9"));
        assert!(!client.model().is_empty());
    }

    #[test]
    fn options_build_up() {
        let o = LlmOptions::default()
            .temperature(0.9)
            .option("num_predict", Json::Int(8))
            .temperature(0.2)
            .model(" other ")
            .json();
        assert_eq!(
            o.options,
            vec![
                ("temperature".to_string(), Json::Num(0.2)),
                ("num_predict".to_string(), Json::Int(8))
            ]
        );
        assert_eq!(o.model_or("mine"), "other");
        assert_eq!(LlmOptions::default().model_or("mine"), "mine");
        assert!(o.json);
        assert_eq!(seconds(0.0), None);
        assert_eq!(seconds(2.5), Some(Duration::from_millis(2500)));
    }

    #[test]
    fn json_is_found_wherever_the_model_put_it() {
        let doc = |t: &str| loads_lenient(t).map(|v| v.render(0));
        assert_eq!(doc("{\"a\": 1}").as_deref(), Some("{\"a\":1}"));
        assert_eq!(doc("```json\n{\"a\": 1}\n```").as_deref(), Some("{\"a\":1}"));
        assert_eq!(
            doc("Sure! Here it is: {\"a\": [1, 2]} hope it helps").as_deref(),
            Some("{\"a\":[1,2]}")
        );
        assert_eq!(doc("[1, 2]").as_deref(), Some("[1,2]"));
        assert_eq!(doc("no json here"), None);
    }

    #[test]
    fn lines_come_back_clean() {
        let text =
            "1. The cat sat.\n\n- The dog ran.\n* \"Quoted line\"\n2) the cat sat.\n```\n(3) Third one\nThird one\n";
        assert_eq!(
            parse_lines(text, None),
            vec!["The cat sat.", "The dog ran.", "Quoted line", "Third one"]
        );
        assert_eq!(parse_lines(text, Some(2)), vec!["The cat sat.", "The dog ran."]);
        assert!(parse_lines("", None).is_empty());
        assert_eq!(
            parse_lines("12 - twelve\n3: three\n•• dot", None),
            vec!["twelve", "three", "dot"]
        );
        // a number that is not numbering stays
        assert_eq!(parse_lines("1984 was a year", None), vec!["1984 was a year"]);
        assert_eq!(parse_lines("(3 - x", None), vec!["(3 - x"]);
    }

    #[test]
    fn lines_split_where_python_splits_them() {
        assert_eq!(splitlines("a\r\nb\rc\nd\u{2028}e\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(splitlines("a\n\nb"), vec!["a", "", "b"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn numbers_print_as_python_prints_them() {
        for (x, want) in [
            (6.0, "6"),
            (6.5, "6.5"),
            (0.0, "0"),
            (10.0, "10"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1234567.0, "1.23457e+06"),
            (999999.5, "1e+06"),
            (123456.0, "123456"),
            (6.123456789, "6.12346"),
            (-2.25, "-2.25"),
        ] {
            assert_eq!(format_g(x), want, "{x}");
        }
    }

    #[test]
    fn the_status_names_both_providers() {
        let state = State {
            ollama_url: "http://o:1".to_string(),
            ollama_model: "m".to_string(),
            chatgpt_url: "https://c/v1".to_string(),
            chatgpt_model: "g".to_string(),
        };
        let fields = state.status_fields();
        assert_eq!(fields[0].0, "ollama");
        assert_eq!(fields[0].1.render(0), "{\"url\":\"http://o:1\",\"model\":\"m\"}");
        assert_eq!(fields[1].1.at("model").as_str(), Some("g"));
        assert!(state.client("gemini", None, None, None).is_err());
        let client = state.client("ollama", None, Some("other"), None).unwrap();
        assert_eq!((client.url(), client.model()), ("http://o:1", "other"));
        assert!(
            state.ollama(Some("   "), None, None).is_err(),
            "a blank URL is not the default"
        );
        assert_eq!(state.tutor_model("chatgpt"), "g");
    }
}
