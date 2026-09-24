//! The Ollama client (`radixnet/ollama.py`, `go/radixnet/ollama.go`), the
//! `radixnet ollama` command and the `/api/ollama/*` routes.
//!
//! Ollama serves a model on this machine (or the local network) over plain
//! HTTP, so this client never leaves the process: [`crate::fetch`] writes the
//! request out over a socket, and - as the Python client does - never through
//! a proxy, which could only fail to reach a server beside it.  Three calls:
//! the installed models (`GET /api/tags`), one completion (`POST
//! /api/generate`) and one chat turn (`POST /api/chat`, with tools for the
//! agent), none of them streamed.
//!
//! The endpoint follows Ollama's own convention - `$OLLAMA_HOST` (a bare
//! `host:port` is accepted), else `http://127.0.0.1:11434` - and the default
//! model is `$RADIXNET_OLLAMA_MODEL`, else `llama3.2`.
//!
//! What is done *with* the answers - a corpus written to order, the
//! adversarial review - is provider-independent and lives in
//! [`crate::review`]; this module is the transport and the two front doors
//! (the command line and the HTTP routes) onto it.  One thing is Ollama's
//! own: a *thinking* model's reasoning (`think: true`, returned as
//! `thinking`, or written inline between `<think>` tags by older models),
//! which [`thoughts_from_prompt`] collects and the network is taught as its
//! own thoughts ([`crate::thinking::think_on`]).

use std::sync::Arc;
use std::time::Duration;

use crate::blame::TeachOptions;
use crate::cli::{negative_stats, read_named, Ctx};
use crate::fetch::Request as FetchRequest;
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::{parse, Json};
use crate::kinds::TrainSettings;
use crate::llm::fields::{py_str_of, Fields};
use crate::llm::{
    clip_chars, count_flag, env, http_reason, nonneg_flag, saved, seconds, timeout_flag, LlmClient, LlmError,
    LlmOptions, OLLAMA,
};
use crate::radix::{Feedback, TrainConfig};
use crate::report::stats;
use crate::review::{self, ReviewOptions};
use crate::service::Service;

/// What this module's own lines are filed under.
const LOG: &str = "llm";

/// How long one answer may take: local models can be slow.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// The model used when neither the caller nor the environment names one.
pub const DEFAULT_MODEL_NAME: &str = "llama3.2";

/// Where Ollama listens when `$OLLAMA_HOST` does not say.
pub const DEFAULT_HOST: &str = "127.0.0.1:11434";

/// The most of an answer that is read (32 MiB).
const MAX_ANSWER: usize = 32 << 20;

/// `host:port` or a full URL -> `http://host:port`, without a trailing slash.
pub fn normalise_url(url: &str) -> Result<String, String> {
    let text = url.trim().trim_end_matches('/');
    if text.is_empty() {
        return Err("the Ollama URL is empty".to_string());
    }
    if text.contains("://") {
        Ok(text.to_string())
    } else {
        Ok(format!("http://{text}"))
    }
}

/// `$OLLAMA_HOST`, else the local default.
pub fn default_url() -> String {
    let host = env("OLLAMA_HOST");
    normalise_url(if host.is_empty() { DEFAULT_HOST } else { &host })
        .unwrap_or_else(|_| format!("http://{DEFAULT_HOST}"))
}

/// `$RADIXNET_OLLAMA_MODEL`, else [`DEFAULT_MODEL_NAME`].
pub fn default_model() -> String {
    let model = env("RADIXNET_OLLAMA_MODEL");
    if model.is_empty() {
        DEFAULT_MODEL_NAME.to_string()
    } else {
        model
    }
}

/// One Ollama server.  It implements [`LlmClient`], so the review and the
/// tutors cannot tell it from a ChatGPT client.
#[derive(Clone, Debug)]
pub struct OllamaClient {
    /// `http://host:port`, without a trailing slash.
    pub url: String,
    /// The model a call answers with when it names none.
    pub model: String,
    /// How long one answer may take when a call does not say.
    pub timeout: Duration,
}

impl OllamaClient {
    /// A client; an empty `url` or `model`, or no `timeout`, takes the
    /// default.  A URL of nothing but spaces is refused rather than defaulted:
    /// somebody typed it.
    pub fn new(url: &str, model: &str, timeout: Option<Duration>) -> Result<OllamaClient, String> {
        let url = if url.is_empty() {
            default_url()
        } else {
            normalise_url(url)?
        };
        let model = match model.trim() {
            "" => default_model(),
            name => name.to_string(),
        };
        Ok(OllamaClient {
            url,
            model,
            timeout: timeout.unwrap_or(DEFAULT_TIMEOUT),
        })
    }

    fn error<S: Into<String>>(message: S) -> LlmError {
        LlmError::new(OLLAMA, message)
    }

    /// One call: the answer's JSON, or why there is none.
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Json>,
        timeout: Option<Duration>,
    ) -> Result<Json, LlmError> {
        let url = format!("{}{path}", self.url);
        let request = match body {
            Some(doc) => FetchRequest::post_json(&url, doc),
            None => FetchRequest::new(method, &url).header("Content-Type", "application/json"),
        };
        let response = request
            .header("Accept", "application/json")
            .timeout(timeout.unwrap_or(self.timeout))
            .limit(MAX_ANSWER)
            .direct()
            .send()
            .map_err(|err| LlmError {
                timeout: err.timeout,
                ..Self::error(format!("cannot reach Ollama at {}: {}", self.url, err.message))
            })?;
        let text = response.text();
        if response.status >= 400 {
            let mut detail = clip_chars(&text, 500).trim().to_string();
            if let Ok(doc @ Json::Obj(_)) = parse(&detail) {
                if let Some(error) = doc.get("error") {
                    detail = py_str_of(error);
                }
            }
            if detail.is_empty() {
                detail = http_reason(response.status).to_string();
            }
            return Err(LlmError {
                status: Some(response.status),
                ..Self::error(format!(
                    "Ollama {method} {path} failed with HTTP {}: {detail}",
                    response.status
                ))
            });
        }
        parse(&text).map_err(|err| Self::error(format!("Ollama returned invalid JSON for {path}: {err}")))
    }

    /// The whole assistant message of one chat turn.
    ///
    /// `tools` are JSON-schema function definitions (Ollama's own `tools`
    /// format); a model that supports tool calling answers with `tool_calls`
    /// beside (or instead of) `content`.  The message comes back as it
    /// arrived, with `content` guaranteed to be a string.
    pub fn chat_message(&self, messages: &[Json], o: &LlmOptions, tools: &[Json]) -> Result<Json, LlmError> {
        let mut body = vec![
            ("model".to_string(), Json::str(o.model_or(&self.model))),
            ("messages".to_string(), Json::Arr(messages.to_vec())),
            ("stream".to_string(), Json::Bool(false)),
        ];
        if o.json {
            body.push(("format".to_string(), Json::str("json")));
        }
        if !o.options.is_empty() {
            body.push(("options".to_string(), Json::Obj(o.options.clone())));
        }
        if !tools.is_empty() {
            body.push(("tools".to_string(), Json::Arr(tools.to_vec())));
        }
        if let Some(think) = &o.think {
            body.push(("think".to_string(), think.clone()));
        }
        let data = self.request("POST", "/api/chat", Some(&Json::Obj(body)), o.timeout)?;
        let Some(Json::Obj(mut message)) = data.get("message").cloned() else {
            return Err(Self::error("unexpected /api/chat response (no message)"));
        };
        match message.iter_mut().find(|(k, _)| k == "content") {
            Some((_, Json::Str(_))) => {}
            Some((_, content)) => *content = Json::str(""),
            None => message.push(("content".to_string(), Json::str(""))),
        }
        Ok(Json::Obj(message))
    }
}

impl OllamaClient {
    /// One completion with the model's thinking beside its answer
    /// (`OllamaClient.complete`).  `o.think` asks a thinking model for its
    /// reasoning; it comes back as Ollama's `thinking` field when the server
    /// separates it, or is cut out of the answer when the model wrote it inline
    /// between `<think>` tags ([`split_thinking`]); it is `""` for a model that
    /// does not think.
    pub fn complete(&self, prompt: &str, o: &LlmOptions) -> Result<Completion, LlmError> {
        let mut body = vec![
            ("model".to_string(), Json::str(o.model_or(&self.model))),
            ("prompt".to_string(), Json::str(prompt)),
            ("stream".to_string(), Json::Bool(false)),
        ];
        if !o.system.is_empty() {
            body.push(("system".to_string(), Json::str(o.system.clone())));
        }
        if o.json {
            body.push(("format".to_string(), Json::str("json")));
        }
        if !o.options.is_empty() {
            body.push(("options".to_string(), Json::Obj(o.options.clone())));
        }
        if let Some(think) = &o.think {
            body.push(("think".to_string(), think.clone()));
        }
        let data = self.request("POST", "/api/generate", Some(&Json::Obj(body)), o.timeout)?;
        let text = match data.get("response") {
            Some(Json::Str(text)) => text.clone(),
            _ => return Err(Self::error("unexpected /api/generate response (no 'response' text)")),
        };
        let (thinking, response) = match data.get("thinking") {
            Some(Json::Str(thinking)) if !thinking.trim().is_empty() => (thinking.trim().to_string(), text),
            _ => split_thinking(&text),
        };
        Ok(Completion { response, thinking })
    }
}

/// One completion of a thinking model: what it answered, and how it got there.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Completion {
    pub response: String,
    pub thinking: String,
}

/// The reasoning levels a thinking model may be asked for, beside plain on / off.
pub const THINK_LEVELS: &[&str] = &["low", "medium", "high"];

/// Ollama's `think` field for a request, read leniently from a command line or
/// a JSON body (`ollama.think_value`): `None` (not sent) for `""` / `none` /
/// `default`, a bool for `true` / `on` / `yes` and `false` / `off` / `no`, a
/// level for one of [`THINK_LEVELS`].
pub fn think_value(text: &str) -> Result<Option<Json>, String> {
    let text = text.trim().to_ascii_lowercase();
    match text.as_str() {
        "" | "none" | "default" => Ok(None),
        "true" | "on" | "yes" | "1" => Ok(Some(Json::Bool(true))),
        "false" | "off" | "no" | "0" => Ok(Some(Json::Bool(false))),
        level if THINK_LEVELS.contains(&level) => Ok(Some(Json::str(level))),
        other => Err(format!(
            "think must be true, false or one of {} (got {})",
            THINK_LEVELS.join(", "),
            crate::negative::python_repr(other)
        )),
    }
}

/// `(thinking, answer)` of an answer that wrote its reasoning inline between
/// `<think>` (or `<thinking>` / `<reasoning>`) tags: the first tagged block is
/// the thinking and everything else the answer; a block left open is thinking
/// to the end; a text without tags is all answer.
pub fn split_thinking(text: &str) -> (String, String) {
    let lower = text.to_ascii_lowercase();
    let mut open: Option<(usize, &str)> = None;
    for tag in ["think", "thinking", "reasoning"] {
        if let Some(at) = lower.find(&format!("<{tag}>")) {
            if open.is_none_or(|(best, _)| at < best) {
                open = Some((at, tag));
            }
        }
    }
    let Some((start, tag)) = open else {
        return (String::new(), text.trim().to_string());
    };
    let open_end = start + tag.len() + 2;
    match lower[open_end..].find(&format!("</{tag}>")) {
        None => (text[open_end..].trim().to_string(), text[..start].trim().to_string()),
        Some(rel) => {
            let close = open_end + rel;
            let close_end = close + tag.len() + 3;
            let thinking = text[open_end..close].trim().to_string();
            let answer = format!("{}{}", &text[..start], &text[close_end..]).trim().to_string();
            (thinking, answer)
        }
    }
}

const QUESTIONS_SYSTEM: &str = "You write questions for a small language model to think about. Answer with exactly \
                                {n} lines and nothing else: one short, concrete question per line, plain text, no \
                                numbering, no bullets, no quotes, no blank lines, no headings and no commentary. \
                                Every question must be about the topic requested and answerable in a sentence or two.";
const THINK_SYSTEM: &str = "Think the question through before you answer, step by step, in short plain sentences - \
                            say what you know, what you are unsure of and ask yourself whether you are right - and \
                            then answer in one short sentence.";

/// Asks the LLM for `lines` short questions about `prompt` - what the network
/// will be taught to think about (`ollama.questions_from_prompt`).
pub fn questions_from_prompt(
    client: &OllamaClient,
    prompt: &str,
    lines: usize,
    model: &str,
) -> Result<Vec<String>, review::ReviewError> {
    if prompt.trim().is_empty() {
        return Err(review::ReviewError::Invalid(
            "prompt must be a non-empty string".to_string(),
        ));
    }
    if lines < 1 {
        return Err(review::ReviewError::Invalid("lines must be >= 1".to_string()));
    }
    let user = format!(
        "Topic / instructions: {}\n\nWrite the {lines} questions now.",
        prompt.trim()
    );
    let o = LlmOptions::default()
        .system(QUESTIONS_SYSTEM.replace("{n}", &lines.to_string()))
        .model(model)
        .temperature(0.9);
    let text = client.generate(&user, &o)?;
    Ok(crate::llm::parse_lines(&text, Some(lines)))
}

/// One question the LLM thought about: the question, its thinking and its
/// answer, each on one line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Thinking {
    pub question: String,
    pub thinking: String,
    pub answer: String,
}

impl Thinking {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("question", Json::str(self.question.clone())),
            ("thinking", Json::str(self.thinking.clone())),
            ("answer", Json::str(self.answer.clone())),
        ])
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The LLM's thinking about `prompt` (`ollama.thoughts_from_prompt`): `lines`
/// questions about it, and the thinking behind each answer.  `thinking` is
/// `""` for a model that does not think, which the caller reports rather than
/// trains on.  The thinking is what [`crate::thinking::think_on`] teaches the
/// network as its own thoughts.
pub fn thoughts_from_prompt(
    client: &OllamaClient,
    prompt: &str,
    lines: usize,
    model: &str,
    think: Option<Json>,
    temperature: f64,
) -> Result<Vec<Thinking>, review::ReviewError> {
    let questions = questions_from_prompt(client, prompt, lines, model)?;
    let mut out = Vec::with_capacity(questions.len());
    for question in questions {
        let mut o = LlmOptions::default()
            .system(THINK_SYSTEM)
            .model(model)
            .temperature(temperature);
        if let Some(value) = &think {
            o = o.think(value.clone());
        }
        let got = client.complete(&question, &o)?;
        out.push(Thinking {
            question,
            thinking: one_line(&got.thinking),
            answer: one_line(&got.response),
        });
    }
    Ok(out)
}

impl LlmClient for OllamaClient {
    fn provider(&self) -> &'static str {
        OLLAMA
    }

    fn url(&self) -> &str {
        &self.url
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// The installed models (`GET /api/tags`).
    fn models(&self) -> Result<Vec<Json>, LlmError> {
        let data = self.request("GET", "/api/tags", None, None)?;
        match data.get("models") {
            Some(Json::Arr(models)) => Ok(models.iter().filter(|m| matches!(m, Json::Obj(_))).cloned().collect()),
            _ => Err(Self::error("unexpected /api/tags response (no 'models' list)")),
        }
    }

    /// One completion (`POST /api/generate`, not streamed); the model's
    /// thinking, if any, is cut out of it ([`OllamaClient::complete`] keeps both).
    fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
        Ok(self.complete(prompt, o)?.response)
    }

    /// One chat turn (`POST /api/chat`): the assistant's text.
    fn chat(&self, messages: &[Json], o: &LlmOptions) -> Result<String, LlmError> {
        let message = self.chat_message(messages, o, &[])?;
        Ok(message.at("content").as_str().unwrap_or("").to_string())
    }
}

// -- the command line ---------------------------------------------------------------------------

/// `radixnet ollama <models | corpus | review>`: a corpus written to order,
/// and an adversarial review of what the network itself writes.  `--url`,
/// `--ollama-model` and `--timeout` override `$OLLAMA_HOST`,
/// `$RADIXNET_OLLAMA_MODEL` and the 120 seconds one answer may take.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    match ctx.args.action() {
        "models" => models_cli(ctx),
        "corpus" => corpus_cli(ctx),
        "review" => review_cli(ctx),
        "think" => think_cli(ctx),
        "" => Err("ollama needs an action: models, corpus, review or think".to_string()),
        other => Err(format!(
            "unknown ollama action {other:?}; expected models, corpus, review or think"
        )),
    }
}

/// `radixnet ollama think --prompt TOPIC [--lines 5] [--think LEVEL] [--out FILE]
/// [--train [--with-answers] [--no-questions] [--epochs] [--lr] [--batch-size]
/// [--model-out]]`: a thinking model thinks about a prompt, and the network is
/// taught its thinking as thoughts of its own (Python's `cmd_ollama_think`).
fn think_cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let prompt = args.str("prompt", "");
    if prompt.trim().is_empty() {
        return Err("--prompt is required: say what the questions should be about".to_string());
    }
    let lines = count_flag(ctx, "lines", 5, 1)?;
    let think = think_value(&args.str("think", "true"))?;
    let temperature = nonneg_flag(ctx, "temperature", 0.7)?;
    let client = client_of(ctx)?;
    let thoughts = thoughts_from_prompt(&client, &prompt, lines, &client.model, think.clone(), temperature)?;
    if thoughts.is_empty() {
        return Err(format!(
            "Ollama model {} wrote no questions to think about",
            crate::negative::python_repr(&client.model)
        ));
    }
    let thinking: Vec<String> = thoughts
        .iter()
        .filter(|t| !t.thinking.is_empty())
        .map(|t| t.thinking.clone())
        .collect();
    crate::log_info!(
        LOG,
        "ollama {} thought about {} question(s), {} with its thinking",
        client.model,
        thoughts.len(),
        thinking.len()
    );
    let mut doc = vec![
        ("url".to_string(), Json::str(client.url.clone())),
        ("model".to_string(), Json::str(client.model.clone())),
        ("prompt".to_string(), Json::str(prompt)),
        ("think".to_string(), think.clone().unwrap_or(Json::Null)),
        ("count".to_string(), Json::Int(thoughts.len() as i64)),
        ("thinking".to_string(), Json::Int(thinking.len() as i64)),
        (
            "thoughts".to_string(),
            Json::Arr(thoughts.iter().map(|t| t.to_json()).collect()),
        ),
        ("out".to_string(), Json::Null),
        ("trained".to_string(), Json::Null),
    ];
    if let Some(out) = args.get("out") {
        let content = if thinking.is_empty() {
            String::new()
        } else {
            thinking.join("\n") + "\n"
        };
        std::fs::write(out, content).map_err(|err| format!("cannot write {out}: {err}"))?;
        doc[7].1 = Json::str(out);
    }
    if args.on("train") {
        if thinking.is_empty() {
            return Err(format!(
                "Ollama model {} returned no thinking to train on: use a thinking model (qwen3, deepseek-r1, \
                 gpt-oss, ...) on an Ollama that separates it, or --think true",
                crate::negative::python_repr(&client.model)
            ));
        }
        let existed = std::path::Path::new(&ctx.model_path).exists();
        let mut model = ctx.open(false)?;
        if model.is_negative() {
            return Err(
                "the negative network judges; it does not think (--kind negative cannot be taught thoughts)"
                    .to_string(),
            );
        }
        let target = args.str("model-out", &ctx.model_path);
        let settings = TrainSettings {
            config: TrainConfig {
                epochs: count_flag(ctx, "epochs", 10, 0)?,
                lr: nonneg_flag(ctx, "lr", 0.5)?,
                batch_size: count_flag(ctx, "batch-size", 4, 1)?,
                ..Default::default()
            },
            ..Default::default()
        };
        let learned = crate::thinking::think_on(
            &mut model,
            &thinking,
            &settings,
            !args.on("no-questions"),
            1.0,
            &mut |_| true,
        )?;
        let mut answers_trained = 0usize;
        if args.on("with-answers") {
            let answers: Vec<String> = thoughts
                .iter()
                .filter(|t| !t.answer.is_empty())
                .map(|t| t.answer.clone())
                .collect();
            if !answers.is_empty() {
                answers_trained = crate::kinds::train(&mut model, &answers, &settings, &mut |_| true)?.len();
            }
        }
        model.save(&target)?;
        doc[8].1 = Json::obj([
            (
                "model",
                Json::obj([
                    ("kind", Json::str(if existed { "model" } else { "new" })),
                    (
                        "path",
                        if existed {
                            Json::str(ctx.model_path.clone())
                        } else {
                            Json::Null
                        },
                    ),
                ]),
            ),
            ("out", Json::str(target.clone())),
            ("thoughts", Json::Int(learned.thoughts as i64)),
            ("questions", Json::Int(learned.questions as i64)),
            ("taught", Json::Int(learned.taught as i64)),
            (
                "epochs",
                Json::Arr(learned.epochs.iter().map(|r| r.to_json()).collect()),
            ),
            ("answers", Json::Int(answers_trained as i64)),
            ("interrupted", Json::Bool(false)),
            ("saved", saved(&target)),
            ("stats", stats(&model)),
        ]);
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

fn client_of(ctx: &Ctx) -> Result<OllamaClient, String> {
    OllamaClient::new(
        &ctx.args.str("url", ""),
        &ctx.args.str("ollama-model", ""),
        timeout_flag(ctx)?,
    )
}

fn models_cli(ctx: &Ctx) -> Result<(), String> {
    let client = client_of(ctx)?;
    let models = client.models()?;
    crate::log_info!(LOG, "ollama {}: {} model(s)", client.url, models.len());
    ctx.emit(Json::obj([
        ("url", Json::str(client.url.clone())),
        ("model", Json::str(client.model.clone())),
        ("models", Json::Arr(models)),
    ]));
    Ok(())
}

fn corpus_cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let prompt = args.str("prompt", "");
    if prompt.is_empty() {
        return Err("--prompt is required: say what the lines should be about".to_string());
    }
    let lines = count_flag(ctx, "lines", 20, 1)?;
    let style = args.str("style", "good");
    if !review::STYLES.contains(&style.as_str()) {
        return Err(format!("--style must be good or garbage, got {style:?}"));
    }
    let client = client_of(ctx)?;
    let texts = review::corpus_from_prompt(&client, &prompt, lines, &style, &client.model)?;
    if texts.is_empty() {
        return Err(format!(
            "Ollama model {} returned no usable lines",
            crate::negative::python_repr(&client.model)
        ));
    }
    let mut doc = vec![
        ("url".to_string(), Json::str(client.url.clone())),
        ("model".to_string(), Json::str(client.model.clone())),
        ("prompt".to_string(), Json::str(prompt)),
        ("style".to_string(), Json::str(style)),
        ("lines".to_string(), Json::Int(texts.len() as i64)),
        ("texts".to_string(), Json::strs(texts.clone())),
        ("out".to_string(), Json::Null),
        ("trained".to_string(), Json::Null),
    ];
    if let Some(out) = args.get("out") {
        std::fs::write(out, texts.join("\n") + "\n").map_err(|err| format!("cannot write {out}: {err}"))?;
        doc[6].1 = Json::str(out);
    }
    if args.on("train") {
        let existed = std::path::Path::new(&ctx.model_path).exists();
        let mut model = ctx.open(false)?;
        let target = args.str("model-out", &ctx.model_path);
        // as its kind learns: a prompt corpus is small, so the sine model's
        // rate is high (Python's `--lr 0.5 --batch-size 4`)
        let records = crate::kinds::train_at(
            &mut model,
            &texts,
            count_flag(ctx, "epochs", 10, 0)?,
            nonneg_flag(ctx, "lr", 0.5)?,
            count_flag(ctx, "batch-size", 4, 1)?,
        )?;
        model.save(&target)?;
        doc[7].1 = Json::obj([
            (
                "model",
                Json::obj([
                    ("kind", Json::str(if existed { "model" } else { "new" })),
                    (
                        "path",
                        if existed {
                            Json::str(ctx.model_path.clone())
                        } else {
                            Json::Null
                        },
                    ),
                ]),
            ),
            ("out", Json::str(target.clone())),
            ("epochs", Json::Arr(records.iter().map(|r| r.to_json()).collect())),
            ("interrupted", Json::Bool(false)),
            ("saved", saved(&target)),
            ("stats", stats(&model)),
        ]);
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

fn review_cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let client = client_of(ctx)?;
    let given = args.all("text");
    let texts = if !given.is_empty() {
        Some(given)
    } else if args.get("data").is_some() {
        Some(read_named(args, "data")?)
    } else {
        None
    };
    let two_nrl = args.on("2nrl");
    let mut model = if texts.is_none() || two_nrl {
        Some(ctx.open(texts.is_none())?)
    } else {
        None
    };
    let threshold = nonneg_flag(ctx, "threshold", review::DEFAULT_THRESHOLD)?;
    let o = ReviewOptions {
        count: count_flag(ctx, "count", 8, 1)?,
        prefix: args.str("prefix", ""),
        max_length: count_flag(ctx, "max-length", 60, 0)?,
        temperature: nonneg_flag(ctx, "temperature", 1.0)?,
        texts,
        threshold,
        context: args.str("context", ""),
        model: client.model.clone(),
        // a run is reproducible when --seed says so, and otherwise walks on
        // from the generator state the model file carries, as Python's does
        seed: args.get("seed").map(|_| ctx.seed),
        batch: 0,
    };
    let result = review::adversarial_review(model.as_mut(), &client, &o)?;
    let mut doc = match result.to_json() {
        Json::Obj(pairs) => pairs,
        _ => Vec::new(),
    };
    doc.push(("two_nrl".to_string(), Json::Null));
    doc.push(("negative".to_string(), Json::Null));
    if args.on("blame") {
        let path = ctx.negative_path();
        let mut negative = ctx.open_negative(false)?;
        let report = review::teach_reviews(
            &mut negative,
            &result.reviews,
            result.threshold,
            true,
            "review",
            &TeachOptions::default(),
        )?;
        negative.save(&path)?;
        doc.last_mut().expect("the negative slot").1 = Json::obj([
            ("blamed", Json::Int(report.blamed as i64)),
            ("cleared", Json::Int(report.cleared as i64)),
            ("edges", Json::Int(report.edges as i64)),
            ("reasons", review::reason_rows(&negative)),
            (
                "lessons",
                Json::Arr(report.faults.iter().map(|f| f.to_json()).collect()),
            ),
            ("path", Json::str(path.clone())),
            ("saved", saved(&path)),
            ("stats", negative_stats(&mut negative)),
        ]);
    }
    if two_nrl {
        let bad = result.bad.clone();
        let mut good = result.good.clone();
        good.extend(read_named(args, "good")?);
        if bad.is_empty() {
            return Err("nothing failed the review, so there is no garbage for the negative phase".to_string());
        }
        if good.is_empty() {
            return Err("no text passed the review and no --good file was given for the positive phase".to_string());
        }
        let model = model.as_mut().expect("--2nrl opens the model");
        // 2NRL through the model's own kind: the sine model reads the rates
        // and the batch size (4: a review is a handful of texts), the count
        // and phase models the strength
        let o = Feedback {
            neg_epochs: count_flag(ctx, "neg-epochs", 3, 0)?,
            pos_epochs: count_flag(ctx, "pos-epochs", 3, 0)?,
            neg_lr: nonneg_flag(ctx, "neg-lr", 0.05)?,
            pos_lr: nonneg_flag(ctx, "pos-lr", 0.01)?,
            strength: args.float("strength", 1.0)?,
            batch_size: Some(count_flag(ctx, "batch-size", 4, 1)?),
            ..Feedback::two_nrl()
        };
        let (negative, positive) = crate::kinds::two_nrl(model, &bad, &good, &o, &mut |_| true)?;
        let out = args.str("out", &ctx.model_path);
        model.save(&out)?;
        let slot = doc.len() - 2;
        doc[slot].1 = Json::obj([
            ("bad_texts", Json::Int(bad.len() as i64)),
            ("good_texts", Json::Int(good.len() as i64)),
            ("negative", Json::Arr(negative.iter().map(|r| r.to_json()).collect())),
            ("positive", Json::Arr(positive.iter().map(|r| r.to_json()).collect())),
            ("inverted", Json::Bool(model.g.inverted)),
            ("interrupted", Json::Bool(false)),
            ("saved", saved(&out)),
            ("stats", stats(model)),
        ]);
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

// -- the routes ---------------------------------------------------------------------------------

/// This area's routes:
/// `GET /api/ollama/models`
/// `POST /api/ollama/corpus`
/// `POST /api/ollama/review`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/ollama/models", models_route);
    server.route("POST", "/api/ollama/corpus", corpus_route);
    server.route("POST", "/api/ollama/review", review_route);
    server.route("POST", "/api/ollama/think", think_route);
}

/// `POST /api/ollama/think {prompt, lines, think, temperature, model, url,
/// timeout, save_as, train, questions, with_answers, epochs, lr, batch_size}`:
/// a thinking model thinks about a prompt; its thinking is returned and, with
/// `train`, taught to the network as thoughts (202 with the job).
fn think_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let prompt = f.required_text("prompt")?;
    if prompt.trim().is_empty() {
        return Err(ApiError::bad_request("'prompt' must not be empty"));
    }
    let lines = f.count_or("lines", 5, 1)?;
    // missing - or null, which the API reads as missing - asks the model to think; "default" leaves it to the model
    let level = match f.get("think") {
        None | Some(Json::Null) => Some(Json::Bool(true)),
        Some(Json::Bool(b)) => Some(Json::Bool(*b)),
        Some(Json::Str(s)) => think_value(s).map_err(ApiError::bad_request)?,
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "'think' must be true, false or one of {} (got {})",
                THINK_LEVELS.join(", "),
                crate::llm::fields::py_repr_of(other)
            )))
        }
    };
    let temperature = f.number_or("temperature", 0.7, Some(0.0))?;
    let train = f.flag("train", false)?;
    let with_answers = f.flag("with_answers", false)?;
    let questions = f.flag("questions", true)?;
    let client = client_of_request(svc, &f)?;
    let config = crate::service::train_config(r)?;
    let save_as = f.text("save_as")?.filter(|name| !name.is_empty());
    if train {
        // an LLM call is not spent on a request that cannot start its job
        svc.ensure_idle()?;
        if svc.with_model(|m| m.is_negative()) {
            return Err(ApiError::bad_request("the negative network judges; it does not think"));
        }
    }
    let thoughts = thoughts_from_prompt(&client, &prompt, lines, &client.model, level.clone(), temperature)?;
    if thoughts.is_empty() {
        return Err(ApiError::with_status(
            502,
            format!(
                "Ollama model {} wrote no questions to think about",
                crate::negative::python_repr(&client.model)
            ),
        ));
    }
    let thinking: Vec<String> = thoughts
        .iter()
        .filter(|t| !t.thinking.is_empty())
        .map(|t| t.thinking.clone())
        .collect();
    let upload = match save_as {
        Some(name) => {
            let content = if thinking.is_empty() {
                String::new()
            } else {
                thinking.join("\n") + "\n"
            };
            store_upload(svc, &name, &content)?
        }
        None => Json::Null,
    };
    let mut doc = vec![
        ("prompt".to_string(), Json::str(prompt)),
        ("model".to_string(), Json::str(client.model.clone())),
        ("url".to_string(), Json::str(client.url.clone())),
        ("think".to_string(), level.clone().unwrap_or(Json::Null)),
        ("count".to_string(), Json::Int(thoughts.len() as i64)),
        ("thinking".to_string(), Json::Int(thinking.len() as i64)),
        (
            "thoughts".to_string(),
            Json::Arr(thoughts.iter().map(|t| t.to_json()).collect()),
        ),
        ("upload".to_string(), upload),
        ("job".to_string(), Json::Null),
    ];
    if !train {
        return Ok(Json::Obj(doc));
    }
    if thinking.is_empty() {
        return Err(ApiError::with_status(
            502,
            format!(
                "Ollama model {} returned no thinking to train on: use a thinking model (qwen3, deepseek-r1, \
                 gpt-oss, ...) on an Ollama that separates it",
                crate::negative::python_repr(&client.model)
            ),
        ));
    }
    svc.ensure_idle()?;
    svc.start_job("train");
    let settings = TrainSettings {
        config,
        ..Default::default()
    };
    let answers: Vec<String> = if with_answers {
        thoughts
            .iter()
            .filter(|t| !t.answer.is_empty())
            .map(|t| t.answer.clone())
            .collect()
    } else {
        Vec::new()
    };
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| -> Result<Vec<Json>, String> {
            let learned = crate::thinking::think_on(m, &thinking, &settings, questions, 1.0, &mut |record| {
                worker.job_progress(record.to_json());
                !worker.stopping()
            })?;
            let mut records: Vec<Json> = learned.epochs.iter().map(|r| r.to_json()).collect();
            if !answers.is_empty() && !worker.stopping() {
                let more = crate::kinds::train(m, &answers, &settings, &mut |record| {
                    worker.job_progress(record.to_json());
                    !worker.stopping()
                })?;
                records.extend(more.iter().map(|r| r.to_json()));
            }
            Ok(records)
        });
        if outcome.is_ok() {
            worker.autosave();
        }
        worker.finish_job(outcome);
    });
    doc[8].1 = svc.job_json();
    Ok(accepted(Json::Obj(doc)))
}

/// The request's Ollama overrides (`url`, `model`, `timeout`), falling back
/// to the server's own.
fn client_of_request(svc: &Service, f: &Fields) -> Result<OllamaClient, ApiError> {
    let url = f.text("url")?;
    let model = f.text("model")?;
    let timeout = f.number("timeout", Some(1.0))?.and_then(seconds);
    svc.llm.ollama(url.as_deref(), model.as_deref(), timeout)
}

/// Is Ollama reachable here, and which models has it (`?url=` asks another
/// one)?  Never fails: "it is not there" is the answer, not an error.
fn models_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let client = svc.llm.ollama(r.query("url"), None, None)?;
    let (models, error) = match client.models() {
        Ok(models) => (models, Json::Null),
        Err(err) => (Vec::new(), Json::str(err.message)),
    };
    let rows = models
        .iter()
        .map(|m| {
            Json::obj([
                ("name", m.at("name").clone()),
                ("size", m.at("size").clone()),
                ("modified_at", m.at("modified_at").clone()),
                ("details", m.at("details").clone()),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("available", Json::Bool(error.is_null())),
        ("url", Json::str(client.url.clone())),
        ("model", Json::str(client.model.clone())),
        ("models", Json::Arr(rows)),
        ("error", error),
    ]))
}

/// A training corpus written to order: `{prompt, lines, style: good|garbage,
/// url, model, timeout, save_as (an upload), train (a training job on the
/// lines), epochs}`.
fn corpus_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let prompt = f.required_text("prompt")?;
    let lines = f.count_or("lines", 20, 1)?;
    let style = f.text_or("style", "good")?.trim().to_lowercase();
    if !review::STYLES.contains(&style.as_str()) {
        return Err(ApiError::bad_request(format!(
            "'style' must be one of {} (got {})",
            review::STYLES.join(", "),
            crate::negative::python_repr(&style)
        )));
    }
    let train = f.flag("train", false)?;
    let client = client_of_request(svc, &f)?;
    // Python's `_train_config(f)`: every kind's settings, each kind reading
    // what applies to it
    let config = crate::service::train_config(r)?;
    let every = config.checkpoint_every;
    if train && every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    let save_as = f.text("save_as")?.filter(|name| !name.is_empty());
    if train {
        // an LLM call is not spent on a request that cannot start its job
        svc.ensure_idle()?;
    }
    let texts = review::corpus_from_prompt(&client, &prompt, lines, &style, &client.model)?;
    if texts.is_empty() {
        return Err(ApiError::with_status(
            502,
            format!(
                "Ollama model {} returned no usable lines",
                crate::negative::python_repr(&client.model)
            ),
        ));
    }
    let upload = match save_as {
        Some(name) => store_upload(svc, &name, &(texts.join("\n") + "\n"))?,
        None => Json::Null,
    };
    let mut doc = vec![
        ("prompt".to_string(), Json::str(prompt)),
        ("style".to_string(), Json::str(style)),
        ("model".to_string(), Json::str(client.model.clone())),
        ("url".to_string(), Json::str(client.url.clone())),
        ("lines".to_string(), Json::Int(texts.len() as i64)),
        ("texts".to_string(), Json::strs(texts.clone())),
        ("upload".to_string(), upload),
        ("job".to_string(), Json::Null),
    ];
    if !train {
        return Ok(Json::Obj(doc));
    }
    svc.ensure_idle()?;
    svc.start_job("train");
    let settings = crate::kinds::TrainSettings {
        config,
        ..Default::default()
    };
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = crate::checkpoint::train_job(&worker, &texts, &settings, every);
        if outcome.is_ok() {
            worker.autosave();
        }
        worker.finish_job(outcome.map(|records| records.iter().map(|r| r.to_json()).collect()));
    });
    doc[7].1 = svc.job_json();
    Ok(accepted(Json::Obj(doc)))
}

/// Stores a text as an upload, the way `POST /api/uploads` does, and says what
/// was stored: `{"name", "bytes", "chars", "lines", "modified", "replaced"}`.
fn store_upload(svc: &Service, name: &str, content: &str) -> Result<Json, ApiError> {
    let dir = svc
        .upload_dir
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("no upload directory is configured"))?;
    let path = crate::service::safe_upload(dir, name)?;
    std::fs::create_dir_all(dir).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let replaced = path.is_file();
    let part = path.with_extension(match path.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });
    std::fs::write(&part, content.as_bytes()).map_err(|err| ApiError::bad_request(err.to_string()))?;
    std::fs::rename(&part, &path).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let modified = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| {
            let stamp = crate::clock::iso8601(d.as_secs() as i64);
            let (head, zone) = stamp.split_at(stamp.len() - "+00:00".len());
            format!("{head}.{:03}{zone}", d.subsec_millis())
        })
        .unwrap_or_default();
    crate::log_info!(LOG, "upload {name} ({} bytes)", content.len());
    Ok(Json::obj([
        ("name", Json::str(name)),
        ("bytes", Json::Int(content.len() as i64)),
        ("chars", Json::Int(content.chars().count() as i64)),
        (
            "lines",
            Json::Int(
                crate::llm::splitlines(content)
                    .iter()
                    .filter(|l| !l.trim().is_empty())
                    .count() as i64,
            ),
        ),
        ("modified", Json::str(modified)),
        ("replaced", Json::Bool(replaced)),
    ]))
}

/// The adversarial review of the model's samples or of `{texts}`: `{count,
/// prefix, max_length, temperature, seed, threshold, context, url, model,
/// timeout, blame (teach the negative network what failed and why), apply:
/// none|2nrl (and good, good_text, good_files, neg_epochs, pos_epochs,
/// strength)}`.
fn review_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let given = f.texts_optional("texts", "text")?;
    let apply = f.text_or("apply", "none")?.trim().to_lowercase();
    if apply != "none" && apply != "2nrl" {
        return Err(ApiError::bad_request(format!(
            "'apply' must be 'none' or '2nrl' (got {})",
            crate::negative::python_repr(&apply)
        )));
    }
    let client = client_of_request(svc, &f)?;
    let threshold = f.number_or("threshold", review::DEFAULT_THRESHOLD, Some(0.0))?;
    if apply == "2nrl" {
        svc.ensure_idle()?;
    }
    let (samples, source) = if !given.is_empty() {
        (given, "given")
    } else {
        let count = f.count_or("count", 8, 1)?;
        let prefix = f.text_or("prefix", "")?;
        let max_length = f.count_or("max_length", 60, 0)?;
        let temperature = f.number_or("temperature", 1.0, Some(0.0))?;
        let seed = f.integer("seed", None)?;
        // the model writes under its lock; the reviewer thinks without it
        let drawn = svc.with_model(|m| review::sample_texts(m, count, &prefix, max_length, temperature, seed))?;
        (drawn, "model")
    };
    let context = f.text_or("context", "")?;
    let reviews = review::review_texts(
        &client,
        &samples,
        &context,
        &client.model,
        threshold,
        review::DEFAULT_BATCH,
    )?;
    let result = review::summarise_reviews(source, &client.model, threshold, samples, reviews);
    let mut doc = match result.to_json() {
        Json::Obj(pairs) => pairs,
        _ => Vec::new(),
    };
    doc.push(("url".to_string(), Json::str(client.url.clone())));
    doc.push(("job".to_string(), Json::Null));
    doc.push(("negative".to_string(), Json::Null));
    if f.flag("blame", false)? {
        let taught = crate::critic::with_negative(svc, |negative| -> Result<Json, String> {
            let report = review::teach_reviews(
                negative,
                &result.reviews,
                threshold,
                true,
                "review",
                &TeachOptions::default(),
            )?;
            Ok(Json::obj([
                ("blamed", Json::Int(report.blamed as i64)),
                ("cleared", Json::Int(report.cleared as i64)),
                ("unmatched", Json::Int(report.unmatched as i64)),
                ("edges", Json::Int(report.edges as i64)),
                ("reasons", report.to_json().at("reasons").clone()),
                (
                    "lessons",
                    Json::Arr(report.faults.iter().map(|f| f.to_json()).collect()),
                ),
                ("severity_mean", Json::Num(report.severity_mean)),
                ("stats", negative_stats(negative)),
                ("reason_table", review::reason_rows(negative)),
            ]))
        })??;
        doc.last_mut().expect("the negative slot").1 = taught;
    }
    if apply != "2nrl" {
        return Ok(Json::Obj(doc));
    }
    let bad = result.bad.clone();
    let mut good = result.good.clone();
    good.extend(f.texts_optional("good", "good_text")?);
    let whole = f.flag("whole_file", false)?;
    for name in f.names("good_files")? {
        good.extend(svc.upload_texts(&name, whole)?);
    }
    if bad.is_empty() {
        return Err(ApiError::bad_request(
            "nothing failed the review, so there is no garbage for the negative phase",
        ));
    }
    if good.is_empty() {
        return Err(ApiError::bad_request(
            "no text passed the review and no good texts were given for the positive phase",
        ));
    }
    // Python's `start_two_nrl(bad, good, neg_epochs=3, pos_epochs=3,
    // neg_lr=0.05, pos_lr=0.01, **_train_overrides(f))`, through the model's
    // own kind; `strength` is this port's knob for the count and phase models
    let present = |name: &str| f.present(name);
    let o = Feedback {
        neg_epochs: f.count_or("neg_epochs", 3, 0)?,
        pos_epochs: f.count_or("pos_epochs", 3, 0)?,
        neg_lr: f.number_or("neg_lr", 0.05, Some(0.0))?,
        pos_lr: f.number_or("pos_lr", 0.01, Some(0.0))?,
        strength: f.number_or("strength", 1.0, Some(0.0))?,
        batch_size: if present("batch_size") {
            Some(f.count_or("batch_size", 1, 1)?)
        } else {
            None
        },
        auto_compress: if present("auto_compress") {
            Some(f.flag("auto_compress", true)?)
        } else {
            None
        },
        clip: if present("clip") {
            Some(f.number_or("clip", 5.0, None)?)
        } else {
            None
        },
        shuffle: if present("shuffle") {
            Some(f.flag("shuffle", true)?)
        } else {
            None
        },
        ..Feedback::two_nrl()
    };
    svc.ensure_idle()?;
    svc.start_job("2nrl");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| {
            let mut watch = |record: &crate::model::EpochRecord| {
                worker.job_progress(record.to_json());
                !worker.stopping()
            };
            crate::kinds::two_nrl(m, &bad, &good, &o, &mut watch)
        });
        worker.finish_job(
            outcome.map(|(negative, positive)| negative.iter().chain(positive.iter()).map(|r| r.to_json()).collect()),
        );
    });
    let slot = doc.len() - 2;
    doc[slot].1 = svc.job_json();
    Ok(accepted(Json::Obj(doc)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;

    /// A fake Ollama on a local port: `/api/tags`, `/api/generate` and
    /// `/api/chat`, answering `answer(path, body)` and remembering every body.
    pub(crate) struct Fake {
        pub url: String,
        pub seen: Arc<Mutex<Vec<(String, Json)>>>,
    }

    pub(crate) fn fake(answer: fn(&str, &Json) -> (u16, String)) -> Fake {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let seen: Arc<Mutex<Vec<(String, Json)>>> = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                let (path, body) = loop {
                    let n = s.read(&mut buf).unwrap_or(0);
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).into_owned();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let head = &text[..end];
                        let length = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().to_string())
                            })
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        if raw.len() >= end + 4 + length || n == 0 {
                            let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                            let body = String::from_utf8_lossy(&raw[end + 4..]).into_owned();
                            break (path, parse(&body).unwrap_or(Json::Null));
                        }
                    }
                    if n == 0 {
                        break ("/".to_string(), Json::Null);
                    }
                };
                let (status, text) = answer(&path, &body);
                log.lock().unwrap().push((path, body));
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{text}",
                    text.len()
                );
                let _ = s.write_all(reply.as_bytes());
            }
        });
        Fake { url, seen }
    }

    fn ollama_answers(path: &str, body: &Json) -> (u16, String) {
        match path {
            "/api/tags" => (
                200,
                "{\"models\": [{\"name\": \"fake:latest\", \"size\": 123}, \"junk\", {\"name\": \"other:7b\"}]}"
                    .to_string(),
            ),
            "/api/generate"
                if body
                    .at("system")
                    .as_str()
                    .is_some_and(|s| s.contains("one short, concrete question per line")) =>
            {
                (
                    200,
                    "{\"response\": \"1. question 1 about the sea?\\n2. question 2 about the sea?\"}".to_string(),
                )
            }
            "/api/generate" if body.get("think").is_some() => (
                200,
                format!(
                    "{{\"response\": \"The answer to {p}\", \"thinking\": \"Let me think about {p} Is that right? Yes, it is.\"}}",
                    p = body.at("prompt").as_str().unwrap_or("")
                ),
            ),
            "/api/generate" if body.at("format").as_str() == Some("json") => (
                200,
                "{\"response\": \"{\\\"reviews\\\": [{\\\"index\\\": 0, \\\"rating\\\": 9}]}\"}".to_string(),
            ),
            "/api/generate" => (200, "{\"response\": \"1. one\\n2. two\"}".to_string()),
            "/api/chat" => (
                200,
                "{\"message\": {\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"x\": 1}]}}".to_string(),
            ),
            "/api/boom" => (500, "{\"error\": \"boom\"}".to_string()),
            _ => (200, "not json".to_string()),
        }
    }

    #[test]
    fn urls_are_normalised() {
        assert_eq!(normalise_url("localhost:11434").unwrap(), "http://localhost:11434");
        assert_eq!(normalise_url("https://box:1/").unwrap(), "https://box:1");
        assert_eq!(normalise_url("  http://a  ").unwrap(), "http://a");
        assert!(normalise_url("").is_err());
        assert!(default_url().starts_with("http"));
        assert!(!default_model().is_empty());
        let client = OllamaClient::new("", " m ", None).unwrap();
        assert_eq!((client.model.as_str(), client.timeout), ("m", DEFAULT_TIMEOUT));
        assert!(OllamaClient::new("  ", "m", None).is_err());
    }

    #[test]
    fn it_speaks_ollamas_api() {
        let server = fake(ollama_answers);
        let client = OllamaClient::new(&server.url, "fake:latest", Some(Duration::from_secs(5))).unwrap();
        let models = client.models().unwrap();
        assert_eq!(models.len(), 2, "what is not an object is dropped");
        assert!(client.available());
        let o = LlmOptions::default()
            .system("exactly 2 lines")
            .model("other:7b")
            .temperature(0.1);
        assert_eq!(client.generate("hello", &o).unwrap(), "1. one\n2. two");
        let (path, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(path, "/api/generate");
        assert_eq!(
            body.render(0),
            "{\"model\":\"other:7b\",\"prompt\":\"hello\",\"stream\":false,\"system\":\"exactly 2 lines\",\
             \"options\":{\"temperature\":0.1}}"
        );
        client.generate("x", &LlmOptions::default().json()).unwrap();
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(body.at("format").as_str(), Some("json"));
        assert_eq!(body.at("model").as_str(), Some("fake:latest"));
        assert!(body.get("options").is_none() && body.get("system").is_none());
        let messages = [crate::llm::message("user", "hi")];
        assert_eq!(client.chat(&messages, &LlmOptions::default()).unwrap(), "");
        let message = client
            .chat_message(
                &messages,
                &LlmOptions::default(),
                &[Json::obj([("type", Json::str("function"))])],
            )
            .unwrap();
        assert_eq!(message.at("tool_calls").as_array().len(), 1);
        let (path, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(path, "/api/chat");
        assert_eq!(body.at("tools").as_array().len(), 1);
        assert_eq!(body.at("messages").as_array()[0].at("content").as_str(), Some("hi"));
    }

    #[test]
    fn failures_say_what_failed() {
        let server = fake(ollama_answers);
        let client = OllamaClient::new(&server.url, "m", Some(Duration::from_secs(5))).unwrap();
        let err = client.request("GET", "/api/boom", None, None).unwrap_err();
        assert_eq!(err.message, "Ollama GET /api/boom failed with HTTP 500: boom");
        assert_eq!(err.status, Some(500));
        let err = client.request("GET", "/other", None, None).unwrap_err();
        assert!(
            err.message.starts_with("Ollama returned invalid JSON for /other"),
            "{err}"
        );
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = closed.local_addr().unwrap().port();
        drop(closed);
        let down = OllamaClient::new(&format!("127.0.0.1:{port}"), "m", Some(Duration::from_secs(2))).unwrap();
        assert!(!down.available());
        let err = down.models().unwrap_err();
        assert!(
            err.message.starts_with("cannot reach Ollama at http://127.0.0.1:"),
            "{err}"
        );
        let api: ApiError = err.into();
        assert_eq!(api.status, 502);
    }

    #[test]
    fn thinking_comes_back_beside_the_answer() {
        let server = fake(ollama_answers);
        let client = OllamaClient::new(&server.url, "fake:latest", Some(Duration::from_secs(5))).unwrap();
        let got = client
            .complete("why is the sky blue?", &LlmOptions::default().think(Json::Bool(true)))
            .unwrap();
        assert_eq!(got.response, "The answer to why is the sky blue?");
        assert_eq!(
            got.thinking,
            "Let me think about why is the sky blue? Is that right? Yes, it is."
        );
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(body.at("think").as_bool(), Some(true));
        client
            .complete("x", &LlmOptions::default().think(Json::str("high")))
            .unwrap();
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(body.at("think").as_str(), Some("high"));
        client.generate("x", &LlmOptions::default()).unwrap();
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert!(body.get("think").is_none());
        // a model that writes its thinking inline is read the same way, and one that does not think thinks nothing
        assert_eq!(
            split_thinking("<think>hmm</think> the answer"),
            ("hmm".to_string(), "the answer".to_string())
        );
        assert_eq!(
            split_thinking("before <THINKING>\nhmm\n</THINKING> after"),
            ("hmm".to_string(), "before  after".to_string())
        );
        assert_eq!(
            split_thinking("<think>open ended"),
            ("open ended".to_string(), String::new())
        );
        assert_eq!(split_thinking("no tags"), (String::new(), "no tags".to_string()));
        assert_eq!(think_value("default").unwrap(), None);
        assert_eq!(think_value("on").unwrap(), Some(Json::Bool(true)));
        assert_eq!(think_value("false").unwrap(), Some(Json::Bool(false)));
        assert_eq!(think_value(" High ").unwrap(), Some(Json::str("high")));
        assert!(think_value("loud").is_err());
    }

    #[test]
    fn thoughts_are_collected_question_by_question() {
        let server = fake(ollama_answers);
        let client = OllamaClient::new(&server.url, "fake:latest", Some(Duration::from_secs(5))).unwrap();
        let thoughts = thoughts_from_prompt(&client, "the sea", 2, "", Some(Json::Bool(true)), 0.7).unwrap();
        assert_eq!(
            thoughts.iter().map(|t| t.question.as_str()).collect::<Vec<_>>(),
            vec!["question 1 about the sea?", "question 2 about the sea?"]
        );
        assert_eq!(
            thoughts[0].thinking,
            "Let me think about question 1 about the sea? Is that right? Yes, it is."
        );
        assert_eq!(thoughts[0].answer, "The answer to question 1 about the sea?");
        let seen = server.seen.lock().unwrap();
        let asked: Vec<&(String, Json)> = seen.iter().filter(|(p, _)| p == "/api/generate").collect();
        assert_eq!(asked.len(), 3, "the questions, then one thought each");
        assert!(asked[0].1.get("think").is_none());
        assert!(asked[1..].iter().all(|(_, b)| b.at("think").as_bool() == Some(true)));
        drop(seen);
        assert!(thoughts_from_prompt(&client, "", 2, "", None, 0.7).is_err());
    }

    #[test]
    fn a_review_goes_out_over_http() {
        let server = fake(ollama_answers);
        let client = OllamaClient::new(&server.url, "fake:latest", Some(Duration::from_secs(5))).unwrap();
        let o = ReviewOptions {
            texts: Some(vec!["a text".to_string()]),
            ..Default::default()
        };
        let result = review::adversarial_review(None, &client, &o).unwrap();
        assert_eq!(result.good, vec!["a text".to_string()]);
        let (_, body) = server.seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(
            body.at("prompt").as_str(),
            Some("Review these 1 texts:\n[0] a text\n\nReturn the JSON now.")
        );
        assert_eq!(body.at("options").render(0), "{\"temperature\":0.2}");
    }
}
