//! The ChatGPT (OpenAI) client (`radixnet/chatgpt.py`,
//! `go/radixnet/chatgpt.go`), the `radixnet chatgpt` command and
//! `GET /api/chatgpt/models`.
//!
//! The same three calls as the Ollama client - the models a key may use
//! (`GET /models`), one completion and one chat turn (`POST
//! /chat/completions`) - so a [`ChatGptClient`] stands in for an
//! [`crate::ollama::OllamaClient`] wherever an [`LlmClient`] is taken.
//!
//! # HTTPS, and the key
//!
//! The hosted API is HTTPS, which the standard library cannot speak; the
//! request goes through [`crate::fetch`], which hands `https://` to the
//! system's `curl` (D-076).  The key travels in the `Authorization` header,
//! and `fetch` gives curl its headers in a `0600` file rather than on the
//! command line, where any user on the machine could read them.  A plain
//! `http://` endpoint - an OpenAI-compatible server on this machine, or a
//! test's fake - never leaves the process.
//!
//! Configuration comes from the environment, the way the OpenAI tools expect
//! it: `OPENAI_API_KEY` (or `OPENAI_API_KEY_FILE`, a file holding it, for
//! Docker secrets), `OPENAI_BASE_URL` (or `OPENAI_API_BASE`) for any
//! OpenAI-compatible server, `RADIXNET_OPENAI_MODEL` for the default model,
//! and `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` for the optional account
//! headers.  The key is read when a request is sent and never kept in a
//! config, a record, a log line or the client's `Debug` output; there is no
//! way to pass one through the HTTP API.  Plain `http://` to anything but a
//! local server is refused unless `RADIXNET_OPENAI_ALLOW_INSECURE` is set,
//! because the key would cross the network unencrypted.
//!
//! Ollama's option names are translated (`num_predict` ->
//! `max_completion_tokens`, ...), and a model that rejects an optional field -
//! the reasoning models refuse `temperature`, older ones `response_format` -
//! is asked again without it instead of failing.

use std::sync::Arc;
use std::time::Duration;

use crate::cli::Ctx;
use crate::fetch::Request as FetchRequest;
use crate::http::{Answer, Request, Server};
use crate::json::{parse, Json};
use crate::llm::fields::{py_str_of, truthy};
use crate::llm::{clip_chars, env, http_reason, message, LlmClient, LlmError, LlmOptions, CHATGPT};
use crate::negative::python_repr;
use crate::service::Service;

/// What this module's own lines are filed under.
const LOG: &str = "llm";

/// How long one answer may take.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// The model used when neither the caller nor the environment names one.
pub const DEFAULT_MODEL_NAME: &str = "gpt-4o-mini";

/// The hosted API; `$OPENAI_BASE_URL` overrides it.
pub const OPENAI_URL: &str = "https://api.openai.com/v1";

/// The most of an answer that is read (32 MiB).
const MAX_ANSWER: usize = 32 << 20;

/// Hosts that are this machine: an OpenAI-compatible server there (llama.cpp,
/// vLLM, LM Studio, a test double) may be spoken to over plain HTTP.
const LOCAL_HOSTS: &[&str] = &[
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
    "0.0.0.0",
    "host.docker.internal",
];

fn is_local(host: &str) -> bool {
    let host = host.to_lowercase();
    LOCAL_HOSTS.contains(&host.as_str()) || host.ends_with(".local")
}

fn env_flag(name: &str) -> bool {
    matches!(env(name).to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// The host of a URL's authority, as `urllib.parse.urlsplit(...).hostname`
/// reads it: no credentials, no port, no brackets, lower case.
fn hostname(netloc: &str) -> String {
    let authority = netloc.rsplit('@').next().unwrap_or(netloc);
    let host = if let Some(inner) = authority.strip_prefix('[') {
        inner.split(']').next().unwrap_or(inner)
    } else {
        authority.split(':').next().unwrap_or(authority)
    };
    host.to_lowercase()
}

/// `host` or a full URL -> a base URL without a trailing slash (`/v1` added
/// when there is no path).  A key must not travel unencrypted to a remote
/// host, so plain `http://` is refused there unless
/// `RADIXNET_OPENAI_ALLOW_INSECURE` is set.
pub fn normalise_url(url: &str) -> Result<String, String> {
    let text = url.trim();
    if text.is_empty() {
        return Err("the OpenAI base URL is empty".to_string());
    }
    let text = if text.contains("://") {
        text.to_string()
    } else {
        format!("https://{text}")
    };
    let (scheme, rest) = text.split_once("://").expect("a scheme was added");
    let scheme = scheme.to_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(format!(
            "the OpenAI base URL must be http:// or https:// (got {})",
            python_repr(url)
        ));
    }
    let rest = rest.split('#').next().unwrap_or("");
    let (netloc, path) = match rest.find(['/', '?']) {
        Some(at) => (&rest[..at], rest[at..].split('?').next().unwrap_or("")),
        None => (rest, ""),
    };
    if netloc.is_empty() {
        return Err(format!("the OpenAI base URL has no host (got {})", python_repr(url)));
    }
    let host = hostname(netloc);
    if scheme == "http" && !is_local(&host) && !env_flag("RADIXNET_OPENAI_ALLOW_INSECURE") {
        return Err(format!(
            "refusing to send the API key unencrypted to {}: use https://, a local server, or set \
             RADIXNET_OPENAI_ALLOW_INSECURE=1",
            python_repr(&host)
        ));
    }
    let path = path.trim_end_matches('/');
    Ok(format!(
        "{scheme}://{netloc}{}",
        if path.is_empty() { "/v1" } else { path }
    ))
}

/// `$OPENAI_BASE_URL` (or `$OPENAI_API_BASE`), else the hosted API.  A broken
/// value falls back rather than stopping anything that never talks to ChatGPT.
pub fn default_url() -> String {
    let mut configured = env("OPENAI_BASE_URL");
    if configured.is_empty() {
        configured = env("OPENAI_API_BASE");
    }
    if configured.is_empty() {
        return OPENAI_URL.to_string();
    }
    normalise_url(&configured).unwrap_or_else(|_| OPENAI_URL.to_string())
}

/// `$RADIXNET_OPENAI_MODEL`, else [`DEFAULT_MODEL_NAME`].
pub fn default_model() -> String {
    let model = env("RADIXNET_OPENAI_MODEL");
    if model.is_empty() {
        DEFAULT_MODEL_NAME.to_string()
    } else {
        model
    }
}

/// The key: `explicit`, else `$OPENAI_API_KEY`, else the content of the file
/// `$OPENAI_API_KEY_FILE` names.  Never logged, never put in a document.
pub fn api_key(explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit.map(str::trim).filter(|k| !k.is_empty()) {
        return Some(key.to_string());
    }
    let key = env("OPENAI_API_KEY");
    if !key.is_empty() {
        return Some(key);
    }
    let path = env("OPENAI_API_KEY_FILE");
    if path.is_empty() {
        return None;
    }
    std::fs::read_to_string(&path)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|key| !key.is_empty())
}

/// Whether a key is available (its value is never returned).
pub fn key_configured() -> bool {
    api_key(None).is_some()
}

/// Ollama's option names -> the chat-completions fields they become.
const OPTION_NAMES: &[(&str, &str)] = &[
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("seed", "seed"),
    ("stop", "stop"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"),
    // Ollama's name for the answer length
    ("num_predict", "max_completion_tokens"),
    ("max_tokens", "max_completion_tokens"),
    ("max_completion_tokens", "max_completion_tokens"),
];

/// Body fields worth dropping and retrying when a model rejects them.
const OPTIONAL_FIELDS: &[&str] = &[
    "temperature",
    "top_p",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "max_completion_tokens",
    "response_format",
];

/// Ollama-style options -> chat-completions fields; an unknown name, a `null`
/// and a length of 0 or less (Ollama's "no limit") are dropped.
pub fn options_body(options: &[(String, Json)]) -> Vec<(String, Json)> {
    let mut body: Vec<(String, Json)> = Vec::new();
    for (name, value) in options {
        let Some((_, field)) = OPTION_NAMES.iter().find(|(n, _)| n == name) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let value = if *field == "max_completion_tokens" {
            let tokens = match value {
                Json::Int(n) => Some(*n),
                Json::Num(x) if x.is_finite() => Some(x.trunc() as i64),
                Json::Str(s) => s.trim().parse().ok(),
                _ => None,
            };
            match tokens {
                Some(n) if n > 0 => Json::Int(n),
                _ => continue,
            }
        } else {
            value.clone()
        };
        match body.iter_mut().find(|(k, _)| k == field) {
            Some(slot) => slot.1 = value,
            None => body.push((field.to_string(), value)),
        }
    }
    body
}

/// Every `name` in `text` that follows `argument:` or `argument supplied:`
/// (any case), as `argument(?: supplied)?:\s*([A-Za-z_]\w*)` finds them.
fn argument_names(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut names = Vec::new();
    let mut from = 0;
    while let Some(at) = lower[from..].find("argument") {
        let mut i = from + at + "argument".len();
        if lower[i..].starts_with(" supplied") {
            i += " supplied".len();
        }
        from = from + at + 1;
        if !lower[i..].starts_with(':') {
            continue;
        }
        i += 1;
        let rest = &text[i..];
        let rest = rest.trim_start();
        if let Some(name) = identifier(rest) {
            names.push(name.to_string());
        }
    }
    names
}

/// The identifier at the start of `text`, if one starts there.
fn identifier(text: &str) -> Option<&str> {
    let mut end = 0;
    for (i, c) in text.char_indices() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || c == '_'
        };
        if !ok {
            break;
        }
        end = i + c.len_utf8();
    }
    (end > 0).then(|| &text[..end])
}

/// Every quoted identifier in `text` - `'name'`, `"name"` or `` `name` `` -
/// as the pattern ``['"`]([A-Za-z_]\w*)['"`]`` finds them, left to right.
fn quoted_names(text: &str) -> Vec<String> {
    let quote = |c: char| matches!(c, '\'' | '"' | '`');
    let mut names = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().expect("in bounds");
        if quote(c) {
            if let Some(name) = identifier(&text[i + 1..]) {
                let after = i + 1 + name.len();
                if text[after..].chars().next().is_some_and(quote) {
                    names.push(name.to_string());
                    i = after + 1;
                    continue;
                }
            }
        }
        i += c.len_utf8();
    }
    names
}

/// The optional body field an error blames, when dropping it is worth one
/// more try: a 400 that says a parameter is unsupported, naming the field in
/// `error.param`, after `argument:`, or in quotes.
fn rejected_field(err: &LlmError, body: &[(String, Json)]) -> Option<String> {
    let lower = err.message.to_lowercase();
    let rejected = [
        "unsupported",
        "unrecognized",
        "not supported",
        "does not support",
        "unknown parameter",
    ]
    .iter()
    .any(|p| lower.contains(p));
    if err.status != Some(400) || !rejected {
        return None;
    }
    let mut names: Vec<String> = err.param.iter().cloned().collect();
    names.extend(argument_names(&err.message));
    names.extend(quoted_names(&err.message));
    names
        .into_iter()
        .find(|name| OPTIONAL_FIELDS.contains(&name.as_str()) && body.iter().any(|(k, _)| k == name))
}

/// The assistant text of a chat-completions answer.
fn answer_text(data: &Json) -> Result<String, LlmError> {
    let error = |message: String| LlmError::new(CHATGPT, message);
    let choice = match data.get("choices") {
        Some(Json::Arr(choices)) if matches!(choices.first(), Some(Json::Obj(_))) => &choices[0],
        _ => return Err(error("unexpected chat completion (no 'choices')".to_string())),
    };
    let empty = Json::Obj(Vec::new());
    let message = match choice.get("message") {
        Some(m @ Json::Obj(_)) => m,
        _ => &empty,
    };
    let content = match message.get("content") {
        Some(Json::Str(text)) => Some(text.clone()),
        // content parts instead of one string
        Some(Json::Arr(parts)) => Some(
            parts
                .iter()
                .filter(|p| matches!(p, Json::Obj(_)))
                .map(|p| p.at("text").as_str().unwrap_or(""))
                .collect(),
        ),
        _ => None,
    };
    if let Some(text) = content.filter(|t| !t.trim().is_empty()) {
        return Ok(text);
    }
    if let Some(refusal) = message.at("refusal").as_str().filter(|r| !r.trim().is_empty()) {
        return Err(error(format!("the model refused to answer: {}", refusal.trim())));
    }
    let finish = match choice.get("finish_reason") {
        Some(Json::Str(reason)) => python_repr(reason),
        _ => "None".to_string(),
    };
    Err(error(format!(
        "the model returned an empty answer (finish_reason={finish})"
    )))
}

/// One OpenAI-compatible endpoint.
#[derive(Clone)]
pub struct ChatGptClient {
    /// The base URL, `https://api.openai.com/v1` or another server's.
    pub url: String,
    /// The model a call answers with when it names none.
    pub model: String,
    /// How long one answer may take when a call does not say.
    pub timeout: Duration,
    /// A key given in code; `None` reads the environment per request.
    api_key: Option<String>,
    /// A server on this machine: never reached through a proxy.
    local: bool,
}

impl std::fmt::Debug for ChatGptClient {
    /// Never shows the key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChatGptClient(url={:?}, model={:?})", self.url, self.model)
    }
}

impl ChatGptClient {
    /// A client; an empty `url` or `model`, or no `timeout`, takes the default.
    pub fn new(url: &str, model: &str, timeout: Option<Duration>) -> Result<ChatGptClient, String> {
        let url = if url.is_empty() {
            default_url()
        } else {
            normalise_url(url)?
        };
        let netloc = url.split_once("://").map(|(_, rest)| rest).unwrap_or("");
        let local = is_local(&hostname(netloc.split('/').next().unwrap_or("")));
        let model = match model.trim() {
            "" => default_model(),
            name => name.to_string(),
        };
        Ok(ChatGptClient {
            url,
            model,
            timeout: timeout.unwrap_or(DEFAULT_TIMEOUT),
            api_key: None,
            local,
        })
    }

    /// The same client with a key of its own rather than the environment's.
    pub fn with_key(mut self, key: &str) -> ChatGptClient {
        self.api_key = Some(key.to_string()).filter(|k| !k.trim().is_empty());
        self
    }

    /// Whether a key is available for this client.
    pub fn configured(&self) -> bool {
        api_key(self.api_key.as_deref()).is_some()
    }

    fn error<S: Into<String>>(message: S) -> LlmError {
        LlmError::new(CHATGPT, message)
    }

    /// One call: the answer's JSON, or why there is none.  Nothing is sent
    /// without a key.
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Json>,
        timeout: Option<Duration>,
    ) -> Result<Json, LlmError> {
        let key = api_key(self.api_key.as_deref()).ok_or_else(|| {
            Self::error(
                "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) in the environment of the process \
                 that talks to ChatGPT",
            )
        })?;
        let url = format!("{}{path}", self.url);
        let mut request = match body {
            Some(doc) => FetchRequest::post_json(&url, doc),
            None => FetchRequest::new(method, &url),
        };
        request = request
            .header("Authorization", &format!("Bearer {key}"))
            .header("Accept", "application/json")
            .timeout(timeout.unwrap_or(self.timeout))
            .limit(MAX_ANSWER);
        let organization = env("OPENAI_ORG_ID");
        if !organization.is_empty() {
            request = request.header("OpenAI-Organization", &organization);
        }
        let project = env("OPENAI_PROJECT_ID");
        if !project.is_empty() {
            request = request.header("OpenAI-Project", &project);
        }
        // the hosted API goes through the environment's proxy; a server on
        // this machine never does
        if self.local {
            request = request.direct();
        }
        let response = request.send().map_err(|err| LlmError {
            timeout: err.timeout,
            ..Self::error(format!("cannot reach the OpenAI API at {}: {}", self.url, err.message))
        })?;
        let text = response.text();
        if response.status >= 400 {
            return Err(self.http_error(&format!("{method} {path}"), response.status, &text));
        }
        parse(&text).map_err(|err| Self::error(format!("the OpenAI API returned invalid JSON for {path}: {err}")))
    }

    /// An error answer, with OpenAI's own message, code and param when it gave
    /// them, and a hint for the statuses that have an obvious cause.
    fn http_error(&self, what: &str, status: u16, raw: &str) -> LlmError {
        let raw = clip_chars(raw, 1000).trim().to_string();
        let mut detail = raw.clone();
        let mut code = None;
        let mut param = None;
        if let Ok(doc @ Json::Obj(_)) = parse(&raw) {
            match doc.get("error") {
                Some(error @ Json::Obj(_)) => {
                    detail = match error.get("message").filter(|m| truthy(m)) {
                        Some(message) => py_str_of(message),
                        None => raw.clone(),
                    };
                    code = error.at("code").as_str().map(str::to_string);
                    param = error.at("param").as_str().map(str::to_string);
                }
                Some(Json::Str(message)) => detail = message.clone(),
                _ => {}
            }
        }
        if detail.is_empty() {
            detail = http_reason(status).to_string();
        }
        let hint = match status {
            401 => " (check OPENAI_API_KEY)".to_string(),
            403 => " (the key may not be allowed to use this model)".to_string(),
            404 => format!(" (does the key have access to model {}?)", python_repr(&self.model)),
            429 => " (rate limit or exhausted quota)".to_string(),
            _ => String::new(),
        };
        LlmError {
            status: Some(status),
            code,
            param,
            ..Self::error(format!("OpenAI {what} failed with HTTP {status}: {detail}{hint}"))
        }
    }

    /// One non-streamed chat completion; a field the model rejects is dropped
    /// and the question asked again.
    fn complete(&self, messages: Vec<Json>, o: &LlmOptions) -> Result<String, LlmError> {
        let mut body = vec![
            ("model".to_string(), Json::str(o.model_or(&self.model))),
            ("messages".to_string(), Json::Arr(messages)),
        ];
        for (field, value) in options_body(&o.options) {
            body.push((field, value));
        }
        if o.json {
            body.push((
                "response_format".to_string(),
                Json::obj([("type", Json::str("json_object"))]),
            ));
        }
        loop {
            match self.request("POST", "/chat/completions", Some(&Json::Obj(body.clone())), o.timeout) {
                Ok(data) => return answer_text(&data),
                Err(err) => {
                    let Some(field) = rejected_field(&err, &body) else {
                        return Err(err);
                    };
                    crate::log_debug!(
                        LOG,
                        "{} refuses {field:?}: asking again without it",
                        o.model_or(&self.model)
                    );
                    body.retain(|(k, _)| *k != field);
                }
            }
        }
    }
}

impl LlmClient for ChatGptClient {
    fn provider(&self) -> &'static str {
        CHATGPT
    }

    fn url(&self) -> &str {
        &self.url
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// The models the key may use (`GET /models`), as `{"name", "id",
    /// "owned_by", "created"}`, sorted by name.
    fn models(&self) -> Result<Vec<Json>, LlmError> {
        let data = self.request("GET", "/models", None, None)?;
        let Some(Json::Arr(items)) = data.get("data") else {
            return Err(Self::error("unexpected /models response (no 'data' list)"));
        };
        let mut models: Vec<(String, Json)> = items
            .iter()
            .filter(|m| matches!(m, Json::Obj(_)) && truthy(m.at("id")))
            .map(|m| {
                let id = py_str_of(m.at("id"));
                let row = Json::obj([
                    ("name", Json::str(id.clone())),
                    ("id", Json::str(id.clone())),
                    ("owned_by", m.at("owned_by").clone()),
                    ("created", m.at("created").clone()),
                ]);
                (id, row)
            })
            .collect();
        models.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(models.into_iter().map(|(_, row)| row).collect())
    }

    /// One completion: a single user turn after the system instruction.
    fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
        let mut messages = Vec::new();
        if !o.system.is_empty() {
            messages.push(message("system", &o.system));
        }
        messages.push(message("user", prompt));
        self.complete(messages, o)
    }

    /// One chat turn.
    fn chat(&self, messages: &[Json], o: &LlmOptions) -> Result<String, LlmError> {
        self.complete(messages.to_vec(), o)
    }
}

// -- the command line ---------------------------------------------------------------------------

/// `radixnet chatgpt <models | ask>`: which models the key may use, and one
/// question.  `--url`, `--chatgpt-model` and `--timeout` override
/// `$OPENAI_BASE_URL`, `$RADIXNET_OPENAI_MODEL` and the 120 seconds; `ask`
/// takes `--prompt`, `--system`, `--temperature` (0.7) and `--json-answer`.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let action = ctx.args.action();
    if !matches!(action, "models" | "ask") {
        return Err(match action {
            "" => "chatgpt needs an action: models or ask".to_string(),
            other => format!("unknown chatgpt action {other:?}; expected models or ask"),
        });
    }
    if !key_configured() {
        return Err(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE, a file holding it) and try again"
                .to_string(),
        );
    }
    let client = ChatGptClient::new(
        &ctx.args.str("url", ""),
        &ctx.args.str("chatgpt-model", ""),
        crate::llm::timeout_flag(ctx)?,
    )?;
    if action == "models" {
        let models = client.models()?;
        ctx.emit(Json::obj([
            ("url", Json::str(client.url.clone())),
            ("model", Json::str(client.model.clone())),
            ("models", Json::Arr(models)),
        ]));
        return Ok(());
    }
    let prompt = ctx.args.str("prompt", "");
    if prompt.is_empty() {
        return Err("--prompt is required: say what to ask".to_string());
    }
    let mut o = LlmOptions::default()
        .system(ctx.args.str("system", ""))
        .temperature(crate::llm::nonneg_flag(ctx, "temperature", 0.7)?);
    if ctx.args.on("json-answer") {
        o = o.json();
    }
    let answer = client.generate(&prompt, &o)?;
    ctx.emit(Json::obj([
        ("url", Json::str(client.url.clone())),
        ("model", Json::str(client.model.clone())),
        ("prompt", Json::str(prompt)),
        ("answer", Json::str(answer)),
    ]));
    Ok(())
}

// -- the routes ---------------------------------------------------------------------------------

/// This area's routes:
/// `GET /api/chatgpt/models`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/chatgpt/models", models_route);
}

/// Is ChatGPT usable on this server, and which models does its key have
/// (`?url=` asks another endpoint)?  Never fails: it reports.
fn models_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let client = svc.llm.chatgpt(r.query("url"), None, None)?;
    let configured = key_configured();
    let mut error = if configured {
        Json::Null
    } else {
        Json::str("no API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) for the server process")
    };
    let mut models = Vec::new();
    if configured {
        match client.models() {
            Ok(found) => models = found,
            Err(err) => error = Json::str(err.message),
        }
    }
    let rows = models
        .iter()
        .map(|m| {
            Json::obj([
                ("name", m.at("name").clone()),
                ("owned_by", m.at("owned_by").clone()),
                ("created", m.at("created").clone()),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("available", Json::Bool(error.is_null())),
        ("configured", Json::Bool(configured)),
        ("url", Json::str(client.url.clone())),
        ("model", Json::str(client.model.clone())),
        ("models", Json::Arr(rows)),
        ("error", error),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::tests::fake;

    fn openai_answers(path: &str, body: &Json) -> (u16, String) {
        let content = |text: &str| {
            format!(
                "{{\"choices\": [{{\"index\": 0, \"message\": {{\"role\": \"assistant\", \"content\": {}}}, \
                 \"finish_reason\": \"stop\"}}]}}",
                Json::str(text).render(0)
            )
        };
        let refuse = |field: &str, param: bool| {
            let message = if param {
                format!("Unsupported parameter: '{field}' is not supported with this model.")
            } else {
                format!("Unrecognized request argument supplied: {field}")
            };
            let param = if param { Json::str(field) } else { Json::Null };
            (
                400,
                format!(
                    "{{\"error\": {{\"message\": {}, \"code\": null, \"param\": {}}}}}",
                    Json::str(message).render(0),
                    param.render(0)
                ),
            )
        };
        match path {
            "/v1/models" => (
                200,
                "{\"data\": [{\"id\": \"zeta\", \"owned_by\": \"openai\", \"created\": 1}, {\"id\": \"alpha\"}, \
                 {\"id\": \"\"}]}"
                    .to_string(),
            ),
            "/v1/chat/completions" => {
                let user = body
                    .at("messages")
                    .as_array()
                    .last()
                    .map(|m| m.at("content").as_str().unwrap_or(""));
                match user.unwrap_or("") {
                    "picky" if body.get("temperature").is_some() => refuse("temperature", true),
                    "picky" if body.get("response_format").is_some() => refuse("response_format", false),
                    "refuse" => (
                        200,
                        "{\"choices\": [{\"message\": {\"content\": \"\", \"refusal\": \"no\"}}]}".to_string(),
                    ),
                    "empty" => (200, content("")),
                    "parts" => (
                        200,
                        "{\"choices\": [{\"message\": {\"content\": [{\"text\": \"a\"}, {\"text\": \"b\"}]}}]}"
                            .to_string(),
                    ),
                    "limit" => (429, "{\"error\": {\"message\": \"slow down\"}}".to_string()),
                    other => (200, content(&format!("answer to: {other}"))),
                }
            }
            _ => (404, "{\"error\": \"unknown path\"}".to_string()),
        }
    }

    #[test]
    fn urls_are_checked_before_a_key_goes_anywhere() {
        assert_eq!(normalise_url("api.openai.com").unwrap(), "https://api.openai.com/v1");
        assert_eq!(
            normalise_url("https://api.openai.com/v1/").unwrap(),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalise_url("  http://localhost:1234/v1  ").unwrap(),
            "http://localhost:1234/v1"
        );
        assert_eq!(
            normalise_url("http://127.0.0.1:8080").unwrap(),
            "http://127.0.0.1:8080/v1"
        );
        assert_eq!(
            normalise_url("http://[::1]:8080/x/?q=1").unwrap(),
            "http://[::1]:8080/x"
        );
        for bad in ["", "   ", "ftp://host/v1", "https://"] {
            assert!(normalise_url(bad).is_err(), "{bad:?}");
        }
        let err = normalise_url("http://gateway.example.com/v1").unwrap_err();
        assert!(err.contains("unencrypted to 'gateway.example.com'"), "{err}");
        assert_eq!(hostname("user:pw@Box.Local:9"), "box.local");
        assert!(is_local("printer.local"));
    }

    #[test]
    fn options_are_translated() {
        let o = vec![
            ("temperature".to_string(), Json::Num(0.3)),
            ("num_predict".to_string(), Json::Int(64)),
            ("top_p".to_string(), Json::Num(0.9)),
            ("mirostat".to_string(), Json::Int(2)),
        ];
        assert_eq!(
            Json::Obj(options_body(&o)).render(0),
            "{\"temperature\":0.3,\"max_completion_tokens\":64,\"top_p\":0.9}"
        );
        let o = vec![
            ("num_predict".to_string(), Json::Int(-1)),
            ("max_tokens".to_string(), Json::str("nope")),
            ("seed".to_string(), Json::Int(7)),
        ];
        assert_eq!(Json::Obj(options_body(&o)).render(0), "{\"seed\":7}");
    }

    #[test]
    fn a_rejected_field_is_found_however_it_is_named() {
        let body = vec![
            ("temperature".to_string(), Json::Num(0.2)),
            ("response_format".to_string(), Json::Null),
        ];
        let err = |message: &str, param: Option<&str>| LlmError {
            status: Some(400),
            param: param.map(str::to_string),
            ..LlmError::new(CHATGPT, message)
        };
        assert_eq!(
            rejected_field(
                &err("Unsupported parameter: 'temperature' is not supported", None),
                &body
            )
            .as_deref(),
            Some("temperature")
        );
        assert_eq!(
            rejected_field(
                &err("Unrecognized request argument supplied: response_format", None),
                &body
            )
            .as_deref(),
            Some("response_format")
        );
        assert_eq!(
            rejected_field(&err("this model does not support it", Some("temperature")), &body).as_deref(),
            Some("temperature")
        );
        assert_eq!(
            rejected_field(&err("Unsupported: 'model'", None), &body),
            None,
            "not optional"
        );
        assert_eq!(
            rejected_field(&err("quota exceeded 'temperature'", None), &body),
            None,
            "not a refusal"
        );
        assert_eq!(quoted_names("a 'x' \"y_1\" `z` 'bad"), vec!["x", "y_1", "z"]);
    }

    #[test]
    fn it_speaks_the_chat_completions_api() {
        let server = fake(openai_answers);
        let client = ChatGptClient::new(&server.url, "fake-gpt", Some(Duration::from_secs(5)))
            .unwrap()
            .with_key("sk-test");
        assert!(format!("{client:?}").contains("fake-gpt") && !format!("{client:?}").contains("sk-test"));
        assert!(client.configured());
        let names: Vec<String> = client
            .models()
            .unwrap()
            .iter()
            .map(|m| m.at("name").as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
        let o = LlmOptions::default().system("be brief").temperature(0.3);
        assert_eq!(client.generate("hi there", &o).unwrap(), "answer to: hi there");
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(
            body.render(0),
            "{\"model\":\"fake-gpt\",\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},\
             {\"role\":\"user\",\"content\":\"hi there\"}],\"temperature\":0.3}"
        );
        let messages = [message("user", "and again")];
        let o = LlmOptions::default().model("another-gpt");
        assert_eq!(client.chat(&messages, &o).unwrap(), "answer to: and again");
        assert_eq!(client.generate("parts", &LlmOptions::default()).unwrap(), "ab");
    }

    #[test]
    fn a_picky_model_is_asked_again_and_failures_carry_the_reason() {
        let server = fake(openai_answers);
        let client = ChatGptClient::new(&server.url, "fake-gpt", Some(Duration::from_secs(5)))
            .unwrap()
            .with_key("sk-test");
        let before = server.seen.lock().unwrap().len();
        let o = LlmOptions::default().temperature(0.3).json();
        assert_eq!(client.generate("picky", &o).unwrap(), "answer to: picky");
        let seen = server.seen.lock().unwrap();
        assert_eq!(
            seen.len() - before,
            3,
            "temperature, then response_format, then an accepted body"
        );
        let last = &seen.last().unwrap().1;
        assert!(last.get("temperature").is_none() && last.get("response_format").is_none());
        drop(seen);
        let err = client.generate("limit", &LlmOptions::default()).unwrap_err();
        assert_eq!(
            err.message,
            "OpenAI POST /chat/completions failed with HTTP 429: slow down (rate limit or exhausted quota)"
        );
        assert_eq!(err.status, Some(429));
        let err = client.generate("refuse", &LlmOptions::default()).unwrap_err();
        assert_eq!(err.message, "the model refused to answer: no");
        let err = client.generate("empty", &LlmOptions::default()).unwrap_err();
        assert_eq!(err.message, "the model returned an empty answer (finish_reason='stop')");
    }
}
