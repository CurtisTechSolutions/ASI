//! Tool use: the network browses and solves on its own, an LLM sets the bar
//! and teaches (`radixnet/agent.py`, `go/radixnet/agent.go`,
//! `go/server/agent.go`).
//!
//! ```text
//! task -> acceptance criteria (LLM) -> the network works with tools -> judge (LLM)
//!      -> teach when it failed (LLM) -> 2NRL (punish the failures, reward the correct run)
//! ```
//!
//! The network only ever emits characters, so a tool call is text it writes
//! (`<tool>web_fetch {"url": "..."}</tool>`, [`crate::tools`]) and the answer
//! comes back as text it reads; a whole attempt is one training text (D-056).
//! Around that the LLM plays four roles, ordered so that it does as little of
//! the work as it can (D-057):
//!
//! * **criteria** - [`write_criteria`] asks, *before* anything is attempted,
//!   for the checkable statements a correct answer must satisfy, so the bar
//!   is not set after the fact;
//! * **mediator** - [`mediate_call`] turns an emission the network cannot
//!   form yet into one valid call against the real schemas (Ollama's own tool
//!   calling, then JSON, then a guess), so even an untrained network makes
//!   progress and learns well-formed calls; the share it wrote itself is the
//!   `autonomy` the records track;
//! * **judge** - [`judge_attempt`] marks the finished transcript against the
//!   criteria written up front;
//! * **teacher** - [`teach_task`] solves the task with the same real tools,
//!   and only when the network failed.
//!
//! Learning focuses on failure: every failed transcript is trained on the
//! harder the worse it was, then the model is inverted and fine-tuned on what
//! was right ([`AgentTrainer::learn`]).  With a negative network attached, each
//! failure is also blamed at the granularity it happened at (D-058,
//! [`faults_from_agent`]), and a candidate the negative network recognises as
//! a known failure is passed over for the next one.  The lesson goes through
//! the model's kind ([`crate::kinds`]), as Python hands one call to whichever
//! model is loaded: a sine model descends at the configured rates and flips
//! the nodes of a failed path, a count model takes reward off its edges, a
//! phase model rotates them out of phase.
//!
//! [`AgentTrainer::explore`] runs the same cycle over tasks the network
//! *chooses*: it continues `TASK:` into whatever it is reaching for, the LLM
//! turns that into one question browsing can settle, and the links found on
//! the way become the frontier.
//!
//! The prompts are Python's byte for byte, and so is the arithmetic of the
//! verdict, the gap and the learning - `tests/test_rust_parity_agent.py` runs
//! both implementations against one fake Ollama and one fake site and compares
//! what the fake was asked and what the model came out as.  Python's agent
//! speaks to Ollama only (its mediator and teacher need Ollama's tool calling),
//! and so does this one: [`AgentLlm`] is that one extra call.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::blame::{
    classify, severity_from_gap, severity_from_rating, severity_of, teach, Fault, TeachOptions, TeachReport,
    AGENT_SEVERITY,
};
use crate::checkpoint::Schedule;
use crate::cli::Ctx;
use crate::codegen::{
    as_dict, choice, clamp_score, extension, feedback, get_either, head, last_loss, number_at_least, origin_json,
    py_float, py_type_name, save_negative, set, set_solved, solutions_json, unique, Hooks, LoopError, Nets,
};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::kinds;
use crate::llm::fields::{py_str_of, truthy, Fields};
use crate::llm::{loads_lenient, message, parse_lines, LlmClient, LlmError, LlmOptions};
use crate::model::{EpochRecord, Model, PredictOptions, TrainOptions};
use crate::negative::{python_repr, JudgeOptions};
use crate::ollama::OllamaClient;
use crate::report::stats;
use crate::service::Service;
use crate::tools::{
    answer_text, call_text, find_answer, format_observation, parse_call, task_header, transcript_text, ToolBox,
    ToolCall, ToolResult, TranscriptStep, ANSWER_CLOSE, ANSWER_OPEN, CALL_CLOSE, CALL_OPEN,
};
use crate::web::collapse;

/// What this module's own lines are filed under.
const LOG: &str = "agent";

/// Who attempts a task: the LLM demonstrating, or the network.
pub const PHASES: &[&str] = &["teacher", "model"];

/// When the mediator steps in: only for an unusable emission, always, or never.
pub const MEDIATION: &[&str] = &["repair", "always", "never"];

/// The most acceptance criteria a task gets.
pub const MAX_CRITERIA: usize = 8;

/// How much of a transcript the judge is shown (its tail).
pub const MAX_TRANSCRIPT_CHARS: usize = 6000;

/// Characters the cheapest path is asked for: enough for a whole
/// `<tool>name {...}</tool>` line.
pub const MIN_EMISSION: usize = 48;

/// The model that writes the criteria, mediates, judges and teaches:
/// `$RADIXNET_AGENT_MODEL`, else Ollama's default.
pub fn default_agent_model() -> String {
    let named = crate::llm::env("RADIXNET_AGENT_MODEL");
    if named.is_empty() {
        crate::ollama::default_model()
    } else {
        named
    }
}

// -- tasks ---------------------------------------------------------------------------------------

/// Something to find out: a question, optionally with criteria, a known answer
/// and pages to start from already written down.
#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    pub criteria: Vec<String>,
    pub answer: Option<String>,
    pub seeds: Vec<String>,
}

impl Task {
    /// A task with nothing but a question.
    pub fn new(id: &str, prompt: &str) -> Task {
        Task {
            id: id.to_string(),
            prompt: prompt.to_string(),
            criteria: Vec::new(),
            answer: None,
            seeds: Vec::new(),
        }
    }

    /// `{"id", "prompt", "criteria", "answer", "seeds"}` - what [`parse_tasks`] reads back.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(self.id.clone())),
            ("prompt", Json::str(self.prompt.clone())),
            ("criteria", Json::strs(self.criteria.clone())),
            ("answer", self.answer.clone().map_or(Json::Null, Json::Str)),
            ("seeds", Json::strs(self.seeds.clone())),
        ])
    }

    /// One task out of a string or an object (`prompt` / `task` / `question` /
    /// `goal`, `criteria`, `answer` / `expected`, `seeds` / `urls`, `id`).
    pub fn from_json(item: &Json, index: usize) -> Result<Task, String> {
        match item {
            Json::Str(text) => {
                if text.trim().is_empty() {
                    return Err(format!("task {index}: empty prompt"));
                }
                Ok(Task::new(&format!("t{index}"), text.trim()))
            }
            Json::Obj(_) => {
                let prompt = ["prompt", "task", "question", "goal"]
                    .iter()
                    .find_map(|key| item.at(key).as_str().filter(|t| !t.trim().is_empty()))
                    .map(|t| t.trim().to_string())
                    .ok_or_else(|| format!("task {index}: missing 'prompt'"))?;
                let criteria: Vec<String> = match item.at("criteria") {
                    value if !truthy(value) => Vec::new(),
                    Json::Str(text) => crate::llm::splitlines(text)
                        .into_iter()
                        .filter(|l| !l.trim().is_empty())
                        .map(str::to_string)
                        .collect(),
                    Json::Arr(items) if items.iter().all(|c| matches!(c, Json::Str(_))) => {
                        items.iter().filter_map(|c| c.as_str().map(str::to_string)).collect()
                    }
                    _ => return Err(format!("task {index}: 'criteria' must be a list of strings")),
                };
                let seeds_value = if truthy(item.at("seeds")) {
                    item.at("seeds")
                } else {
                    item.at("urls")
                };
                let seeds: Vec<String> = match seeds_value {
                    value if !truthy(value) => Vec::new(),
                    Json::Str(one) => vec![one.clone()],
                    Json::Arr(items) if items.iter().all(|s| matches!(s, Json::Str(_))) => {
                        items.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()
                    }
                    _ => return Err(format!("task {index}: 'seeds' must be a list of URLs")),
                };
                let answer = match get_either(item, "answer", "expected") {
                    Json::Null => None,
                    Json::Str(text) => Some(text.clone()),
                    _ => return Err(format!("task {index}: 'answer' must be a string")),
                };
                let id = match item.at("id") {
                    Json::Null => format!("t{index}"),
                    Json::Str(s) if s.is_empty() => format!("t{index}"),
                    raw => py_str_of(raw).trim().to_string(),
                };
                Ok(Task {
                    id,
                    prompt,
                    criteria: criteria
                        .iter()
                        .map(|c| c.trim().to_string())
                        .filter(|c| !c.is_empty())
                        .collect(),
                    answer,
                    seeds: seeds
                        .iter()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                })
            }
            other => Err(format!(
                "task {index}: must be a string or an object, got {}",
                py_type_name(other)
            )),
        }
    }
}

/// Tasks from strings and objects (ids default to `t1`, `t2`, ...); a repeated
/// id is refused.
pub fn parse_tasks(items: &[Json]) -> Result<Vec<Task>, String> {
    let mut tasks: Vec<Task> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        tasks.push(Task::from_json(item, i + 1)?);
    }
    let mut seen: Vec<&str> = Vec::new();
    for task in &tasks {
        if seen.contains(&task.id.as_str()) {
            return Err(format!("duplicate task id {}", python_repr(&task.id)));
        }
        seen.push(&task.id);
    }
    Ok(tasks)
}

/// A task file's content: `.txt` one question per line (`#` comments),
/// `.jsonl` objects, `.json` a list or `{"tasks": [...]}`.
pub fn parse_task_file(text: &str, ext: &str) -> Result<Vec<Task>, String> {
    let items: Vec<Json> = match ext.to_lowercase().as_str() {
        ".jsonl" => crate::llm::splitlines(text)
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .map(crate::mcp::pyjson::loads)
            .collect::<Result<_, _>>()?,
        ".json" => {
            let refused = || "a JSON task file must hold a list or an object with a 'tasks' list".to_string();
            match crate::mcp::pyjson::loads(text)? {
                Json::Arr(items) => items,
                doc @ Json::Obj(_) => match get_either(&doc, "tasks", "problems") {
                    Json::Arr(items) => items.clone(),
                    _ => return Err(refused()),
                },
                _ => return Err(refused()),
            }
        }
        _ => crate::llm::splitlines(text)
            .into_iter()
            .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
            .map(|line| Json::str(line.trim()))
            .collect(),
    };
    if items.is_empty() {
        return Err("no tasks in the file".to_string());
    }
    parse_tasks(&items)
}

/// A task file from disk.
pub fn load_tasks(path: &str) -> Result<Vec<Task>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    parse_task_file(&text, &extension(path))
}

// -- one step, one attempt, one verdict -----------------------------------------------------------

/// One tool call inside an attempt, and where the call came from.
#[derive(Clone, Debug)]
pub struct Step {
    pub index: usize,
    pub call: ToolCall,
    pub result: ToolResult,
    /// `model`, `mediator` or `teacher`.
    pub source: String,
    /// What the network wrote, when the call was its own or repaired from it.
    pub emission: String,
}

impl Step {
    /// The call as it appears in the transcript.
    pub fn text(&self) -> String {
        call_text(&self.call.name, &self.call.arguments)
    }

    /// The step as the API and the CLI report it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("index", Json::Int(self.index as i64)),
            ("tool", Json::str(self.call.name.clone())),
            ("arguments", self.call.arguments.clone()),
            ("source", Json::str(self.source.clone())),
            ("ok", Json::Bool(self.result.ok)),
            ("output", Json::str(head(&self.result.output, 500))),
            ("error", self.result.error.clone().map_or(Json::Null, Json::Str)),
            ("seconds", Json::Num(round_to(self.result.seconds, 4))),
            ("emission", Json::str(head(&self.emission, 200))),
            ("text", Json::str(self.text())),
        ])
    }
}

/// Python's `round(x, digits)`.
fn round_to(x: f64, digits: i32) -> f64 {
    crate::calc::round_float(x, digits as i128).unwrap_or(x)
}

/// What the judge decided about a finished attempt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    pub correct: bool,
    pub score: Option<f64>,
    pub met: Vec<bool>,
    pub critique: String,
    pub issues: Vec<String>,
    /// `ollama`, `answer` or `none`.
    pub judged_by: String,
}

impl Verdict {
    /// Not judged yet: incorrect, by nobody.
    pub fn unjudged() -> Verdict {
        Verdict {
            judged_by: "none".to_string(),
            ..Default::default()
        }
    }

    /// `{"correct", "score", "met", "critique", "issues", "judged_by"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("correct", Json::Bool(self.correct)),
            ("score", self.score.map_or(Json::Null, Json::Num)),
            ("met", Json::Arr(self.met.iter().map(|m| Json::Bool(*m)).collect())),
            ("critique", Json::str(self.critique.clone())),
            ("issues", Json::strs(self.issues.clone())),
            ("judged_by", Json::str(self.judged_by.clone())),
        ])
    }
}

/// A failed transcript and how badly it failed (`gap` in `[0, 1]`, 1 = it got
/// nothing right).
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub text: String,
    pub gap: f64,
    pub task: String,
    pub source: String,
}

/// One run at a task: the steps taken, the answer given, the transcript
/// learned and the verdict.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub index: usize,
    /// `model` or `teacher`.
    pub source: String,
    pub steps: Vec<Step>,
    pub answer: Option<String>,
    pub text: String,
    pub verdict: Verdict,
    pub seconds: f64,
    /// How badly it failed, in `[0, 1]`; `None` until it is judged.
    pub gap: Option<f64>,
}

impl Attempt {
    pub fn calls(&self) -> usize {
        self.steps.len()
    }

    /// Calls the network wrote by itself (the mediator repaired the rest).
    pub fn own_calls(&self) -> usize {
        self.steps.iter().filter(|s| s.source == "model").count()
    }

    /// The share of the calls the network wrote itself; `None` with no calls.
    pub fn autonomy(&self) -> Option<f64> {
        (!self.steps.is_empty()).then(|| self.own_calls() as f64 / self.calls() as f64)
    }

    /// Why the attempt was rejected, in the words the teacher gets to see.
    pub fn feedback(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.answer.is_none() {
            parts.push("It never gave an answer.".to_string());
        }
        let failed: Vec<String> = self
            .steps
            .iter()
            .filter(|s| !s.result.ok)
            .take(4)
            .map(|s| format!("{}: {}", s.call.name, s.result.error.as_deref().unwrap_or("None")))
            .collect();
        if !failed.is_empty() {
            parts.push(format!("Failed tool calls: {}", failed.join("; ")));
        }
        if !self.verdict.issues.is_empty() {
            parts.push(format!(
                "Unmet criteria: {}",
                self.verdict
                    .issues
                    .iter()
                    .take(6)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        if !self.verdict.critique.is_empty() {
            parts.push(format!("Judge: {}", self.verdict.critique));
        }
        if parts.is_empty() {
            "It was judged incorrect.".to_string()
        } else {
            parts.join("\n")
        }
    }

    /// The attempt as the API and the CLI report it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("index", Json::Int(self.index as i64)),
            ("source", Json::str(self.source.clone())),
            ("answer", self.answer.clone().map_or(Json::Null, Json::Str)),
            ("calls", Json::Int(self.calls() as i64)),
            ("own_calls", Json::Int(self.own_calls() as i64)),
            ("autonomy", self.autonomy().map_or(Json::Null, Json::Num)),
            ("steps", Json::Arr(self.steps.iter().map(Step::to_json).collect())),
            ("verdict", self.verdict.to_json()),
            ("correct", Json::Bool(self.verdict.correct)),
            ("text_chars", Json::Int(self.text.chars().count() as i64)),
            ("text", Json::str(self.text.clone())),
            ("gap", self.gap.map_or(Json::Null, Json::Num)),
            ("seconds", Json::Num(self.seconds)),
        ])
    }
}

/// The tail of a long transcript - what matters for the next step.
pub fn clip(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    format!("... {}", crate::codegen::tail(text, limit))
}

// -- the four LLM roles --------------------------------------------------------------------------

/// The LLM the agent works with: the four roles through [`LlmClient`], and the
/// one call Python's agent makes beyond it - a chat turn with tool schemas
/// offered, for Ollama's native tool calling.
pub trait AgentLlm: Send + Sync {
    /// The client behind it, for the ordinary calls.
    fn llm(&self) -> &dyn LlmClient;
    /// One assistant message, `tools` offered (Ollama's `tool_calls` may come back).
    fn chat_message(&self, messages: &[Json], o: &LlmOptions, tools: &[Json]) -> Result<Json, LlmError>;
}

impl AgentLlm for OllamaClient {
    fn llm(&self) -> &dyn LlmClient {
        self
    }
    fn chat_message(&self, messages: &[Json], o: &LlmOptions, tools: &[Json]) -> Result<Json, LlmError> {
        OllamaClient::chat_message(self, messages, o, tools)
    }
}

/// What the criteria writer is told it is; `{count}` is how many it may write.
const CRITERIA_SYSTEM: &str = "You write acceptance criteria for a task that will be solved by browsing the web with \
tools. Write at most {count} short, checkable statements that any correct answer must satisfy: what the answer has to \
contain, how specific it has to be, and what would make it wrong. Judge only the answer, never the wording or the \
effort. Do not solve the task and do not state the answer. Reply with JSON only, of the form {\"criteria\": \
[\"<statement>\", ...]}.";

/// What the mediator is told it is.
pub const MEDIATOR_SYSTEM: &str = "You are the tool mediator for a very small character-level neural network that is \
learning to use tools. It writes its intentions as raw text, which is often garbled, truncated or empty. Your job is \
to turn what it wrote into exactly ONE valid tool call that makes progress on the task — never to solve the task \
yourself. Read its text charitably: if it reaches for a search, search for what it seems to want; if it names a URL, \
open that URL; if it is unreadable, choose the call that the task and the transcript so far make obvious. Never repeat \
a call that is already in the transcript. Reply with JSON only, of the form {\"tool\": \"<name>\", \"arguments\": \
{...}}.";

/// What the judge is told it is.
pub const JUDGE_SYSTEM: &str = "You are the judge of an attempt at a task by a very small neural network that browses \
with tools. You are given the acceptance criteria written before the attempt, the transcript of the tool calls and \
their results, and the answer. Mark each criterion strictly: met only if the answer really satisfies it — an answer \
that is empty, vague, off-topic, or contradicted by the transcript meets nothing. Never reward effort, only the \
answer. Reply with JSON only, of the form {\"met\": [true, false, ...] with one entry per criterion in order, \
\"score\": <0-10>, \"correct\": true or false, \"critique\": \"<one sentence naming the worst flaw, or 'none'>\"}.";

/// What the teacher is told it is.
pub const TEACHER_SYSTEM: &str = "You solve a task with the tools you are given, so that a very small neural network \
can learn from your example. Work the way it must: call one tool at a time, read the result, and call another if you \
need to. Use the tools for anything factual — never answer from memory alone. When you know the answer, stop calling \
tools and reply with the answer itself, one short sentence, nothing else.";

/// What the proposer (exploration's task setter) is told it is.
pub const PROPOSE_SYSTEM: &str = "A very small neural network that browses the web is choosing what to look into \
next, and it writes its intentions as raw, often garbled text. Turn what it wrote into ONE concrete question that can \
be settled by browsing and answered in a sentence. Follow whatever it seems to be reaching for; when it is unreadable, \
ask something that follows on from what it has already looked at, not a repeat of it. Prefer a question about one of \
the unvisited pages when they fit. Reply with JSON only, of the form {\"question\": \"<the question>\", \"seed\": \"<a \
URL to start from, or empty>\"}.";

fn tools_block(toolbox: &ToolBox) -> String {
    format!("The tools are:\n{}", toolbox.catalogue())
}

/// The statements a correct answer must satisfy, written before anything is
/// attempted.  A task that carries criteria keeps them; an answer with nothing
/// usable in it falls back to one criterion naming the task, so the judge
/// always has something objective to mark against.
pub fn write_criteria(client: &dyn LlmClient, task: &Task, model: &str, count: usize) -> Result<Vec<String>, LlmError> {
    if !task.criteria.is_empty() {
        return Ok(task.criteria.clone());
    }
    let count = count.clamp(1, MAX_CRITERIA);
    let o = LlmOptions::default()
        .system(CRITERIA_SYSTEM.replace("{count}", &count.to_string()))
        .model(model)
        .json()
        .temperature(0.2);
    let raw = client.generate(&format!("Task: {}\n\nWrite the criteria now.", task.prompt), &o)?;
    let data = loads_lenient(&raw).map(as_dict);
    let mut items: Json = match &data {
        Some(doc @ Json::Obj(_)) => doc.at("criteria").clone(),
        Some(other) => other.clone(),
        None => Json::Null,
    };
    if let Json::Obj(pairs) = &items {
        items = Json::Arr(pairs.iter().map(|(_, v)| v.clone()).collect());
    }
    let mut criteria: Vec<String> = Vec::new();
    if let Json::Arr(list) = &items {
        for item in list {
            let text = match item {
                Json::Str(s) => s.clone(),
                Json::Obj(_) => {
                    if truthy(item.at("criterion")) {
                        py_str_of(item.at("criterion"))
                    } else if truthy(item.at("text")) {
                        py_str_of(item.at("text"))
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            };
            let text = collapse(&text);
            if !text.is_empty() && !criteria.contains(&text) {
                criteria.push(text);
            }
        }
    }
    if criteria.is_empty() {
        criteria = parse_lines(&raw, Some(count));
    }
    if criteria.is_empty() {
        criteria = vec![format!(
            "The answer states, correctly and specifically, what the task asked for: {}",
            task.prompt
        )];
    }
    criteria.truncate(count);
    Ok(criteria)
}

/// A call with no text behind it (the LLM wrote it).
fn bare_call(name: &str, arguments: Json) -> ToolCall {
    ToolCall {
        name: name.to_string(),
        arguments,
        raw: String::new(),
        span: (0, 0),
        error: None,
    }
}

/// The arguments of a call as the LLM wrote them: an object, JSON text of one,
/// or nothing usable (`{}`).
fn arguments_of(value: &Json) -> Json {
    let value = match value {
        Json::Str(text) => loads_lenient(text).map(as_dict).unwrap_or(Json::Null),
        other => other.clone(),
    };
    match value {
        object @ Json::Obj(_) => object,
        _ => Json::Obj(Vec::new()),
    }
}

/// One call out of Ollama's native `tool_calls`, or out of a JSON answer in
/// the content; `None` when neither names a tool the box has.
pub fn call_from_message(message: &Json, toolbox: &ToolBox) -> Option<ToolCall> {
    if let Json::Arr(calls) = message.at("tool_calls") {
        for entry in calls {
            let function = entry.at("function");
            if !matches!(function, Json::Obj(_)) {
                continue;
            }
            let name = if truthy(function.at("name")) {
                py_str_of(function.at("name")).trim().to_string()
            } else {
                String::new()
            };
            if !name.is_empty() && toolbox.has(&name) {
                return Some(bare_call(&name, arguments_of(function.at("arguments"))));
            }
        }
    }
    let content = message.at("content").as_str().unwrap_or("");
    if let Some(data @ Json::Obj(_)) = loads_lenient(content).map(as_dict) {
        let named = ["tool", "name", "function"]
            .iter()
            .map(|k| data.at(k))
            .find(|v| truthy(v))
            .map(|v| py_str_of(v).trim().to_string())
            .unwrap_or_default();
        let arguments = match data.get("arguments") {
            Some(value) => value,
            None => get_either(&data, "args", "parameters"),
        };
        if !named.is_empty() && toolbox.has(&named) {
            return Some(bare_call(&named, arguments_of(arguments)));
        }
    }
    None
}

/// Turns the network's unusable emission into one valid call - Ollama's tool
/// calling, then JSON, then a guess.  Fails only when the LLM cannot be
/// reached; an answer that names no known tool falls back to the first network
/// tool with the task as its argument, so the loop always makes a move.
pub fn mediate_call(
    client: &dyn AgentLlm,
    emission: &str,
    toolbox: &ToolBox,
    task: &Task,
    transcript: &str,
    model: &str,
    hint: Option<&str>,
) -> Result<ToolCall, LlmError> {
    let seen = clip(transcript, 2000);
    let written = head(emission.trim(), 500);
    let mut user = format!(
        "Task: {}\n\n{}\n\nTranscript so far:\n{}\n\nWhat the network just wrote:\n{}\n\n",
        task.prompt,
        tools_block(toolbox),
        if seen.is_empty() { "(nothing yet)" } else { &seen },
        if written.is_empty() { "(nothing)" } else { written },
    );
    if let Some(hint) = hint.filter(|h| !h.is_empty()) {
        user.push_str(&format!("{hint}\n\n"));
    }
    user.push_str("Give the one tool call to make now.");
    let messages = vec![message("system", MEDIATOR_SYSTEM), message("user", &user)];
    let o = LlmOptions::default().model(model).temperature(0.2);
    let answered = client.chat_message(&messages, &o, toolbox.schemas().as_array()).ok();
    let mut call = answered.as_ref().and_then(|m| call_from_message(m, toolbox));
    if call.is_none() {
        // a model without tool calling: ask for the JSON directly
        let raw = client.llm().chat(&messages, &o.clone().json())?;
        call = call_from_message(&Json::obj([("content", Json::str(raw))]), toolbox);
    }
    let mut call = match call {
        Some(call) => call,
        None => {
            let fallback = toolbox
                .tools()
                .iter()
                .find(|t| t.network)
                .or_else(|| toolbox.tools().first());
            let Some(fallback) = fallback else {
                let mut none = bare_call("", Json::Obj(Vec::new()));
                none.error = Some("no tools are installed".to_string());
                return Ok(none);
            };
            let arguments = match fallback.params.iter().find(|p| p.required) {
                Some(param) => Json::obj([(param.name.clone(), Json::str(task.prompt.clone()))]),
                None => Json::Obj(Vec::new()),
            };
            bare_call(&fallback.name, arguments)
        }
    };
    match toolbox.get(&call.name).and_then(|tool| tool.coerce(&call.arguments)) {
        Ok(values) => call.arguments = values,
        // reported on the call, executed as a failure
        Err(why) => call.error = Some(why),
    }
    Ok(call)
}

/// One criterion's mark: a boolean, a number, or a word the LLM used for it
/// (`None` = unreadable).
fn mark(value: &Json) -> Option<bool> {
    match value {
        Json::Bool(b) => Some(*b),
        Json::Int(n) => Some(*n != 0),
        Json::Num(x) => Some(*x != 0.0),
        Json::Obj(pairs) => ["met", "ok", "pass", "passed", "result", "verdict"]
            .iter()
            .find_map(|key| pairs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v))
            .and_then(mark),
        Json::Str(text) => {
            let word = text.trim().trim_matches('.').to_lowercase();
            match word.as_str() {
                "true" | "yes" | "y" | "pass" | "passed" | "met" | "ok" | "correct" | "1" => Some(true),
                "false" | "no" | "n" | "fail" | "failed" | "unmet" | "not_met" | "missing" | "incorrect" | "0" => {
                    Some(false)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The per-criterion marks of a judge's answer, or none when any of them
/// cannot be read - all or nothing, because a partly readable list would
/// silently misalign with the criteria it marks.
fn marks(value: &Json) -> Vec<bool> {
    let Json::Arr(items) = value else { return Vec::new() };
    items
        .iter()
        .map(mark)
        .collect::<Option<Vec<bool>>>()
        .unwrap_or_default()
}

/// What the judge said about a finished attempt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Judged {
    pub met: Vec<bool>,
    pub score: Option<f64>,
    pub correct: Option<bool>,
    pub critique: String,
}

/// Marks a finished attempt against the criteria written before it started.
pub fn judge_attempt(
    client: &dyn LlmClient,
    task: &Task,
    criteria: &[String],
    transcript: &str,
    answer: Option<&str>,
    model: &str,
) -> Result<Judged, LlmError> {
    let numbered: Vec<String> = criteria
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {c}", i + 1))
        .collect();
    let numbered = if numbered.is_empty() {
        "1. The answer is correct.".to_string()
    } else {
        numbered.join("\n")
    };
    let mut user = format!(
        "Task: {}\n\nAcceptance criteria:\n{numbered}\n\nTranscript:\n{}\n\nAnswer: {}\n\n",
        task.prompt,
        clip(transcript, MAX_TRANSCRIPT_CHARS),
        answer.filter(|a| !a.is_empty()).unwrap_or("(no answer was given)")
    );
    if let Some(known) = task.answer.as_deref().filter(|a| !a.is_empty()) {
        user.push_str(&format!("The known correct answer is: {known}\n\n"));
    }
    user.push_str(&format!(
        "Return the JSON with exactly {} entries in \"met\".",
        criteria.len().max(1)
    ));
    let o = LlmOptions::default()
        .system(JUDGE_SYSTEM)
        .model(model)
        .json()
        .temperature(0.1);
    let raw = client.generate(&user, &o)?;
    let data = match loads_lenient(&raw).map(as_dict) {
        Some(data @ Json::Obj(_)) => data,
        _ => {
            return Ok(Judged {
                critique: "the judge returned nothing usable".to_string(),
                ..Default::default()
            })
        }
    };
    let met_value = match data.get("met") {
        Some(value) => value,
        None => get_either(&data, "criteria_met", "results"),
    };
    let correct = match get_either(&data, "correct", "verdict") {
        Json::Str(word) => Some(matches!(
            word.trim().to_lowercase().as_str(),
            "true" | "pass" | "yes" | "correct"
        )),
        Json::Bool(b) => Some(*b),
        _ => None,
    };
    let critique = if truthy(data.at("critique")) {
        py_str_of(data.at("critique"))
    } else if truthy(data.at("reason")) {
        py_str_of(data.at("reason"))
    } else {
        String::new()
    };
    Ok(Judged {
        met: marks(met_value),
        score: py_float(get_either(&data, "score", "rating")).map(clamp_score),
        correct,
        critique: collapse(&critique),
    })
}

/// The judge's answer as a verdict: no answer is never correct, and `strict`
/// needs every criterion met.
pub fn decide(judged: &Judged, criteria: &[String], answer: Option<&str>, strict: bool) -> Verdict {
    let met = judged.met.clone();
    let issues: Vec<String> = criteria
        .iter()
        .zip(&met)
        .filter(|(_, ok)| !**ok)
        .map(|(c, _)| c.clone())
        .collect();
    if answer.is_none_or(|a| a.is_empty()) {
        return Verdict {
            correct: false,
            score: judged.score,
            met,
            critique: if judged.critique.is_empty() {
                "no answer was given".to_string()
            } else {
                judged.critique.clone()
            },
            issues: if issues.is_empty() { criteria.to_vec() } else { issues },
            judged_by: "answer".to_string(),
        };
    }
    let all_met = !met.is_empty() && met.iter().all(|m| *m) && met.len() >= criteria.len();
    let mut correct = match judged.correct {
        Some(correct) => correct,
        None if !met.is_empty() => all_met,
        None => judged.score.is_some_and(|s| s >= 6.0),
    };
    if strict {
        correct = correct && (met.is_empty() || all_met);
    }
    Verdict {
        correct,
        score: judged.score,
        judged_by: if !judged.met.is_empty() || judged.score.is_some() {
            "ollama"
        } else {
            "none"
        }
        .to_string(),
        met,
        critique: judged.critique.clone(),
        issues,
    }
}

/// The LLM solves the task with the *real* tools: the steps it took and the
/// answer it reached.  Every call is executed, so the demonstration the network
/// learns from is a transcript of things that actually happened.
#[allow(clippy::too_many_arguments)]
pub fn teach_task(
    client: &dyn AgentLlm,
    task: &Task,
    toolbox: &ToolBox,
    model: &str,
    max_steps: usize,
    criteria: &[String],
    feedback: Option<&str>,
    observation_chars: usize,
) -> Result<(Vec<Step>, Option<String>), LlmError> {
    let numbered: Vec<String> = criteria.iter().map(|c| format!("- {c}")).collect();
    let mut user = format!("Task: {}\n\n{}\n", task.prompt, tools_block(toolbox));
    if !numbered.is_empty() {
        user.push_str(&format!("\nA correct answer must satisfy:\n{}\n", numbered.join("\n")));
    }
    if let Some(feedback) = feedback.filter(|f| !f.is_empty()) {
        user.push_str(&format!("\nAn earlier attempt failed:\n{feedback}\n"));
    }
    if !task.seeds.is_empty() {
        user.push_str(&format!("\nStart from: {}\n", task.seeds.join(", ")));
    }
    user.push_str("\nMake the first tool call now.");
    let mut messages = vec![message("system", TEACHER_SYSTEM), message("user", &user)];
    let schemas = toolbox.schemas().as_array().to_vec();
    let o = LlmOptions::default().model(model).temperature(0.3);
    let mut steps: Vec<Step> = Vec::new();
    let mut native = true;
    for index in 0..max_steps.max(1) {
        let mut answered = Json::Obj(Vec::new());
        if native {
            match client.chat_message(&messages, &o, &schemas) {
                Ok(m) => answered = m,
                Err(_) => native = false,
            }
        }
        if !native {
            answered = Json::obj([
                ("role", Json::str("assistant")),
                ("content", Json::str(client.llm().chat(&messages, &o)?)),
            ]);
        }
        let call = call_from_message(&answered, toolbox);
        let content = answered.at("content").as_str().unwrap_or("").trim().to_string();
        let Some(call) = call else {
            if !content.is_empty() {
                return Ok((steps, Some(collapse(&content))));
            }
            if !steps.is_empty() {
                break;
            }
            messages.push(message("user", "Make a tool call now, or give the answer."));
            continue;
        };
        let result = toolbox.run(&call);
        let observation = if result.ok {
            result.output.clone()
        } else {
            format!("ERROR: {}", result.error.as_deref().unwrap_or("None"))
        };
        steps.push(Step {
            index,
            call,
            result,
            source: "teacher".to_string(),
            emission: String::new(),
        });
        if truthy(answered.at("role")) {
            messages.push(answered);
        } else {
            messages.push(message("assistant", &content));
        }
        messages.push(message("tool", head(&observation, (observation_chars * 2).max(100))));
    }
    let mut last = messages.clone();
    last.push(message(
        "user",
        "Give the answer now, one short sentence, nothing else.",
    ));
    let answer = client
        .llm()
        .chat(&last, &LlmOptions::default().model(model).temperature(0.2))?;
    let answer = collapse(&answer);
    Ok((steps, (!answer.is_empty()).then_some(answer)))
}

/// Turns the network's own emission into one concrete, answerable question -
/// exploration picks its own tasks.
pub fn propose_task(
    client: &dyn LlmClient,
    emission: &str,
    model: &str,
    frontier: &[String],
    visited: &[String],
    index: usize,
) -> Result<Task, LlmError> {
    let unvisited: Vec<&String> = frontier.iter().take(12).collect();
    let seen: Vec<&String> = visited.iter().skip(visited.len().saturating_sub(8)).collect();
    let listed = |title: &str, urls: &[&String]| -> String {
        if urls.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = urls.iter().map(|u| format!("- {u}")).collect();
        format!("{title}:\n{}\n\n", lines.join("\n"))
    };
    let written = head(emission.trim(), 500);
    let user = format!(
        "What the network wrote:\n{}\n\n{}{}Give the question now.",
        if written.is_empty() { "(nothing)" } else { written },
        listed("Unvisited pages it has come across", &unvisited),
        listed("Pages it has already read", &seen)
    );
    let o = LlmOptions::default()
        .system(PROPOSE_SYSTEM)
        .model(model)
        .json()
        .temperature(0.8);
    let raw = client.generate(&user, &o)?;
    let (mut question, mut seed) = (String::new(), String::new());
    if let Some(data @ Json::Obj(_)) = loads_lenient(&raw).map(as_dict) {
        if let Some(value) = ["question", "task", "prompt"]
            .iter()
            .map(|k| data.at(k))
            .find(|v| truthy(v))
        {
            question = collapse(&py_str_of(value));
        }
        if let Some(value) = ["seed", "url"].iter().map(|k| data.at(k)).find(|v| truthy(v)) {
            seed = py_str_of(value).trim().to_string();
        }
    }
    if question.is_empty() {
        question = parse_lines(&raw, Some(1)).into_iter().next().unwrap_or_default();
    }
    if question.is_empty() {
        question = match unvisited.first() {
            Some(url) => format!("What can be found out about {url}?"),
            None => "What is on the front page of a well-known encyclopaedia?".to_string(),
        };
    }
    let mut task = Task::new(&format!("x{index}"), &question);
    if seed.starts_with("http") {
        task.seeds = vec![seed];
    }
    Ok(task)
}

// -- teaching the negative network ----------------------------------------------------------------

/// Why an attempt was rejected, read from how far through the task it got:
/// it never called anything, the mediator had to write its call, a tool
/// refused it, it never answered - and when it got through all of that, the
/// judge's own words (Python's `blame.agent_reason`).
pub fn agent_reason(attempt: &Attempt) -> String {
    if attempt.steps.is_empty() {
        return "no-call".to_string();
    }
    if attempt.steps.iter().any(|s| s.source == "mediator") {
        return "bad-call".to_string();
    }
    if attempt.steps.iter().any(|s| !s.result.ok) {
        return "tool-error".to_string();
    }
    if attempt.answer.as_deref().is_none_or(str::is_empty) {
        return "no-answer".to_string();
    }
    let v = &attempt.verdict;
    let words = if v.critique.is_empty() {
        v.issues.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
    } else {
        v.critique.clone()
    };
    classify(&words, "", v.score, "")
}

/// `(faults, correct_texts)` of one task's attempts, each failure blamed at
/// the granularity it happened at (D-058): the transcript for how badly it
/// failed (diffed against `correction`, a correct run of the same task, when
/// there is one); each emission the mediator had to repair, for itself
/// (`bad-call`); each call the network wrote that the tool refused
/// (`tool-error`).  `steps` false keeps only the attempt-level faults.
pub fn faults_from_agent(
    attempts: &[Attempt],
    correction: Option<&str>,
    threshold: f64,
    source: &str,
    steps: bool,
) -> (Vec<Fault>, Vec<String>) {
    let mut faults = Vec::new();
    let mut correct = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for attempt in attempts {
        let text = attempt.text.clone();
        let v = &attempt.verdict;
        if v.correct {
            if !text.is_empty() {
                correct.push(text);
            }
            continue;
        }
        let mut note = v.critique.clone();
        if !v.issues.is_empty() {
            note = format!(
                "{note} Unmet: {}",
                v.issues.iter().take(4).cloned().collect::<Vec<_>>().join("; ")
            )
            .trim()
            .to_string();
        }
        if !text.is_empty() {
            let severity = match attempt.gap {
                Some(gap) => severity_from_gap(Some(gap)),
                None => severity_from_rating(v.score, threshold),
            };
            let mut fault = Fault::new(
                &text,
                &agent_reason(attempt),
                severity,
                if note.is_empty() {
                    "the attempt was judged incorrect"
                } else {
                    &note
                },
                source,
            );
            if let Some(correction) = correction.filter(|c| !c.is_empty() && *c != text) {
                fault.correction = correction.to_string();
            }
            faults.push(fault);
        }
        if !steps {
            continue;
        }
        for step in &attempt.steps {
            if step.source == "mediator" {
                // what the network wrote is the failure, and it is not in the
                // transcript, which holds the repaired call instead
                let emission = collapse(head(&step.emission, 200));
                if emission.chars().count() >= 3 && !seen.contains(&emission) {
                    seen.push(emission.clone());
                    faults.push(Fault::new(
                        &emission,
                        "bad-call",
                        severity_of(AGENT_SEVERITY, "bad-call"),
                        &format!(
                            "could not be read as a call to {}, which the mediator had to write instead",
                            step.call.name
                        ),
                        source,
                    ));
                }
            } else if !step.result.ok && step.source == "model" {
                let call = step.text().trim().to_string();
                if !call.is_empty() && !seen.contains(&call) {
                    seen.push(call.clone());
                    faults.push(Fault::new(
                        &call,
                        "tool-error",
                        severity_of(AGENT_SEVERITY, "tool-error"),
                        step.result.error.as_deref().unwrap_or("the tool refused the call"),
                        source,
                    ));
                }
            }
        }
    }
    // the correction already clears what it shares with the failure structure;
    // clearing it again would take off the blame the diff just placed
    let correct = correct.into_iter().filter(|t| Some(t.as_str()) != correction).collect();
    (faults, correct)
}

/// Feeds one task's attempts into the negative network (Python's
/// `blame.teach_agent`).
pub fn teach_agent(
    negative: &mut Model,
    attempts: &[Attempt],
    correction: Option<&str>,
    threshold: f64,
    source: &str,
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    let (faults, correct) = faults_from_agent(attempts, correction, threshold, source, true);
    let mut report = teach(negative, &faults, &correct, o)?;
    report.source = source.to_string();
    report.threshold = Some(threshold);
    report.faults = faults;
    report.passed = correct.len();
    Ok(report)
}

// -- configuration --------------------------------------------------------------------------------

/// How the loop runs, Python's `AgentConfig` field for field: who attempts,
/// how far, how strictly it is judged, and how hard it is taught.  The
/// learning rates and the batch size are the sine model's; the count and
/// phase models accept and ignore them, as Python's do.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentConfig {
    pub agent_model: String,
    pub judge_model: Option<String>,
    pub phases: Vec<String>,
    pub rounds: usize,
    /// Tool calls per attempt.
    pub max_steps: usize,
    /// Attempts by the network per task.
    pub model_attempts: usize,
    pub teacher_attempts: usize,
    /// The first attempt is the beam search, not a sample.
    pub first_attempt_dijkstra: bool,
    /// Continuations the network offers per step before the mediator steps in.
    pub candidates: usize,
    pub temperature: f64,
    /// Characters the network writes per step.
    pub max_length: usize,
    pub mediation: String,
    pub criteria_count: usize,
    /// Every criterion must be met.
    pub strict: bool,
    pub use_judge: bool,
    pub teach_on_failure: bool,
    pub observation_chars: usize,
    /// `task` or `round`.
    pub twonrl_per: String,
    pub replay: bool,
    pub replay_limit: usize,
    /// Also fine-tune on the text of the pages that were read.
    pub read_reward: bool,
    /// Pass over a candidate the negative network has seen fail.
    pub avoid_blamed: bool,
    /// The risk at which it is passed over.
    pub avoid_threshold: f64,
    pub blatant_mode: String,
    pub blatant_margin: f64,
    pub blatant_boost: f64,
    /// The judge's 0-10 score a correct answer is expected to reach.
    pub pass_score: f64,
    pub neg_epochs: usize,
    pub pos_epochs: usize,
    pub neg_lr: f64,
    pub pos_lr: f64,
    pub batch_size: usize,
    /// Checkpoint every N tasks (0 = off).
    pub checkpoint_every: usize,
    pub seed: i64,
}

impl Default for AgentConfig {
    /// Python's defaults.
    fn default() -> AgentConfig {
        AgentConfig {
            agent_model: default_agent_model(),
            judge_model: None,
            phases: vec!["model".to_string()],
            rounds: 1,
            max_steps: 6,
            model_attempts: 2,
            teacher_attempts: 1,
            first_attempt_dijkstra: true,
            candidates: 5,
            temperature: 1.0,
            max_length: 200,
            mediation: "repair".to_string(),
            criteria_count: 4,
            strict: true,
            use_judge: true,
            teach_on_failure: true,
            observation_chars: 600,
            twonrl_per: "task".to_string(),
            replay: true,
            replay_limit: 64,
            read_reward: false,
            avoid_blamed: true,
            avoid_threshold: 1.0,
            blatant_mode: "fail_invert".to_string(),
            blatant_margin: 0.5,
            blatant_boost: 4.0,
            pass_score: 6.0,
            neg_epochs: 2,
            pos_epochs: 3,
            neg_lr: 0.5,
            pos_lr: 0.1,
            batch_size: 4,
            checkpoint_every: 0,
            seed: 0,
        }
    }
}

impl AgentConfig {
    /// Refuses settings the loop cannot run with, in Python's words.
    pub fn validate(&self) -> Result<(), String> {
        let modes = crate::gan::BLATANT_MODES;
        if self.phases.is_empty() || self.phases.iter().any(|p| !PHASES.contains(&p.as_str())) {
            return Err(format!("phases must be a non-empty subset of {}", PHASES.join(", ")));
        }
        if !MEDIATION.contains(&self.mediation.as_str()) {
            return Err(format!(
                "mediation must be one of {} (got {})",
                MEDIATION.join(", "),
                python_repr(&self.mediation)
            ));
        }
        if self.rounds < 1 {
            return Err("rounds must be >= 1".to_string());
        }
        if self.max_steps < 1 {
            return Err("max_steps must be >= 1".to_string());
        }
        if self.model_attempts < 1 || self.teacher_attempts < 1 {
            return Err("model_attempts and teacher_attempts must be >= 1".to_string());
        }
        if self.max_length < 1 {
            return Err("max_length must be >= 1".to_string());
        }
        if self.candidates < 1 {
            return Err("candidates must be >= 1".to_string());
        }
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        if !(1..=MAX_CRITERIA).contains(&self.criteria_count) {
            return Err(format!("criteria_count must be between 1 and {MAX_CRITERIA}"));
        }
        if self.twonrl_per != "task" && self.twonrl_per != "round" {
            return Err("twonrl_per must be 'task' or 'round'".to_string());
        }
        if !modes.contains(&self.blatant_mode.as_str()) {
            return Err(format!(
                "blatant_mode must be one of {} (got {})",
                modes.join(", "),
                python_repr(&self.blatant_mode)
            ));
        }
        if self.blatant_margin.is_nan() || self.blatant_margin <= 0.0 {
            return Err("blatant_margin must be > 0".to_string());
        }
        if self.blatant_boost.is_nan() || self.blatant_boost < 1.0 {
            return Err("blatant_boost must be >= 1".to_string());
        }
        if !(self.pass_score > 0.0 && self.pass_score <= 10.0) {
            return Err("pass_score must be in (0, 10]".to_string());
        }
        if self.avoid_threshold.is_nan() || self.avoid_threshold < 0.0 {
            return Err("avoid_threshold must be >= 0".to_string());
        }
        if self.neg_lr < 0.0 || self.pos_lr < 0.0 {
            return Err("epochs and learning rates must be >= 0".to_string());
        }
        if self.batch_size < 1 {
            return Err("batch_size must be >= 1".to_string());
        }
        Ok(())
    }

    /// The model the judge answers with.
    pub fn judge_model_name(&self) -> &str {
        match self.judge_model.as_deref() {
            Some(model) if !model.is_empty() => model,
            _ => &self.agent_model,
        }
    }

    /// The settings as a document (`dataclasses.asdict` of Python's config).
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("agent_model", Json::str(self.agent_model.clone())),
            ("judge_model", self.judge_model.clone().map_or(Json::Null, Json::Str)),
            ("phases", Json::strs(self.phases.clone())),
            ("rounds", Json::Int(self.rounds as i64)),
            ("max_steps", Json::Int(self.max_steps as i64)),
            ("model_attempts", Json::Int(self.model_attempts as i64)),
            ("teacher_attempts", Json::Int(self.teacher_attempts as i64)),
            ("first_attempt_dijkstra", Json::Bool(self.first_attempt_dijkstra)),
            ("candidates", Json::Int(self.candidates as i64)),
            ("temperature", Json::Num(self.temperature)),
            ("max_length", Json::Int(self.max_length as i64)),
            ("mediation", Json::str(self.mediation.clone())),
            ("criteria_count", Json::Int(self.criteria_count as i64)),
            ("strict", Json::Bool(self.strict)),
            ("use_judge", Json::Bool(self.use_judge)),
            ("teach_on_failure", Json::Bool(self.teach_on_failure)),
            ("observation_chars", Json::Int(self.observation_chars as i64)),
            ("twonrl_per", Json::str(self.twonrl_per.clone())),
            ("replay", Json::Bool(self.replay)),
            ("replay_limit", Json::Int(self.replay_limit as i64)),
            ("read_reward", Json::Bool(self.read_reward)),
            ("avoid_blamed", Json::Bool(self.avoid_blamed)),
            ("avoid_threshold", Json::Num(self.avoid_threshold)),
            ("blatant_mode", Json::str(self.blatant_mode.clone())),
            ("blatant_margin", Json::Num(self.blatant_margin)),
            ("blatant_boost", Json::Num(self.blatant_boost)),
            ("pass_score", Json::Num(self.pass_score)),
            ("neg_epochs", Json::Int(self.neg_epochs as i64)),
            ("pos_epochs", Json::Int(self.pos_epochs as i64)),
            ("neg_lr", Json::Num(self.neg_lr)),
            ("pos_lr", Json::Num(self.pos_lr)),
            ("batch_size", Json::Int(self.batch_size as i64)),
            ("checkpoint_every", Json::Int(self.checkpoint_every as i64)),
            ("seed", Json::Int(self.seed)),
        ])
    }
}

// -- the trainer ----------------------------------------------------------------------------------

/// One step's move: the call (or `None`), where it came from, what the network
/// wrote, and the answer it gave instead of a call.
type Move = (Option<ToolCall>, &'static str, String, Option<String>);

/// What one task comes to: its record, the failed transcripts (with how badly
/// each failed) and the correct ones.
pub type TaskOutcome = (Vec<(String, Json)>, Vec<Failure>, Vec<String>);

/// Runs tasks (or self-chosen exploration) through tools and teaches the
/// network with 2NRL.
pub struct AgentTrainer<'a> {
    pub client: &'a dyn AgentLlm,
    pub toolbox: &'a ToolBox,
    pub config: AgentConfig,
    /// Every record written, in order.
    pub history: Vec<Json>,
    pub replay_buffer: Vec<String>,
    /// The correct transcript of each task solved, by id.
    pub solved: Vec<(String, String)>,
    /// The criteria each task was given, by id.
    pub criteria: Vec<(String, Vec<String>)>,
    /// Pages the tools turned up and nobody has read yet.
    pub frontier: Vec<String>,
    pub visited: Vec<String>,
    /// Page texts read this task (`read_reward`).
    pub pages: Vec<String>,
    /// Candidates passed over because the negative network had seen them fail.
    pub avoided: usize,
    exploring: bool,
}

impl<'a> AgentTrainer<'a> {
    /// The loop, once its settings are sound and there is something to call.
    pub fn new(
        client: &'a dyn AgentLlm,
        toolbox: &'a ToolBox,
        config: AgentConfig,
    ) -> Result<AgentTrainer<'a>, String> {
        if toolbox.is_empty() {
            return Err("the toolbox is empty: there is nothing for the network to call".to_string());
        }
        config.validate()?;
        Ok(AgentTrainer {
            client,
            toolbox,
            config,
            history: Vec::new(),
            replay_buffer: Vec::new(),
            solved: Vec::new(),
            criteria: Vec::new(),
            frontier: Vec::new(),
            visited: Vec::new(),
            pages: Vec::new(),
            avoided: 0,
            exploring: false,
        })
    }

    fn emit(&mut self, hooks: &mut Hooks, record: Json) {
        (hooks.progress)(&record);
        self.history.push(record);
    }

    /// Keeps the pages a tool turned up as the frontier of the exploration.
    fn note_urls(&mut self, result: &ToolResult) {
        let mut found: Vec<String> = Vec::new();
        for key in ["results", "links"] {
            for item in result.meta.at(key).as_array() {
                if let Some(url) = item.at("url").as_str().filter(|u| u.starts_with("http")) {
                    found.push(url.to_string());
                }
            }
        }
        if let Some(url) = result.meta.at("url").as_str().filter(|u| u.starts_with("http")) {
            if !self.visited.iter().any(|v| v == url) {
                self.visited.push(url.to_string());
            }
        }
        for candidate in found {
            if !self.visited.contains(&candidate) && !self.frontier.contains(&candidate) {
                self.frontier.push(candidate);
            }
        }
        if self.frontier.len() > 200 {
            let excess = self.frontier.len() - 200;
            self.frontier.drain(..excess);
        }
        if self.config.read_reward && result.ok && truthy(result.meta.at("chars")) {
            let page = result.output.trim();
            if page.chars().count() >= 3 {
                self.pages.push(page.to_string());
                if self.pages.len() > 32 {
                    let excess = self.pages.len() - 32;
                    self.pages.drain(..excess);
                }
            }
        }
    }

    /// The acceptance criteria of a task, written once by the LLM and remembered.
    pub fn criteria_for(&mut self, task: &Task) -> Result<Vec<String>, LlmError> {
        if let Some((_, known)) = self.criteria.iter().find(|(id, _)| *id == task.id) {
            return Ok(known.clone());
        }
        let criteria = if !task.criteria.is_empty() {
            task.criteria.clone()
        } else if !self.config.use_judge {
            vec![format!("The answer states what the task asked for: {}", task.prompt)]
        } else {
            write_criteria(
                self.client.llm(),
                task,
                &self.config.agent_model,
                self.config.criteria_count,
            )?
        };
        self.criteria.push((task.id.clone(), criteria.clone()));
        Ok(criteria)
    }

    /// What the network offers to write next, most likely first: on the first
    /// attempt the beam search's candidates (a single cheapest path is no good
    /// here - Dijkstra accepts reaching END as a goal, so it answers with a
    /// couple of characters that can never be a whole call), later samples.
    /// A candidate that closed its tag comes before one that only started.
    fn emissions(&self, nets: &mut Nets, transcript: &str, attempt: usize) -> Result<Vec<String>, String> {
        let cfg = &self.config;
        let prefix = clip(transcript, 2000);
        let texts: Vec<String> = if attempt == 0 && cfg.first_attempt_dijkstra {
            let found = nets.model(|m| {
                m.predict(
                    &prefix,
                    &PredictOptions {
                        length: MIN_EMISSION.min(cfg.max_length),
                        mode: "beam".to_string(),
                        k: cfg.candidates,
                        max_length: Some(cfg.max_length),
                        ..Default::default()
                    },
                )
            })?;
            if found.top.is_empty() {
                vec![found.best.text]
            } else {
                found.top.into_iter().map(|r| r.text).collect()
            }
        } else {
            let mut texts = Vec::new();
            for _ in 0..cfg.candidates {
                let found = nets.model(|m| {
                    m.predict(
                        &prefix,
                        &PredictOptions {
                            length: 1,
                            mode: "sample".to_string(),
                            temperature: cfg.temperature,
                            max_length: Some(cfg.max_length),
                            ..Default::default()
                        },
                    )
                })?;
                texts.push(found.best.text);
            }
            texts
        };
        let offered: Vec<String> = unique(&texts).into_iter().filter(|t| !t.trim().is_empty()).collect();
        let (whole, started): (Vec<String>, Vec<String>) = offered
            .into_iter()
            .partition(|t| t.contains(CALL_CLOSE) || t.contains(ANSWER_CLOSE));
        Ok(whole.into_iter().chain(started).collect())
    }

    /// Whether the negative network recognises this as a way of going wrong it
    /// has seen before - what the failures taught it, coming back.
    fn blamed(&self, nets: &mut Nets, text: &str) -> bool {
        if !nets.has_negative() || !self.config.avoid_blamed || text.trim().chars().count() < 3 {
            return false;
        }
        let o = JudgeOptions {
            threshold: Some(self.config.avoid_threshold),
            ..Default::default()
        };
        // a network that cannot judge never vetoes
        matches!(nets.negative(|n| n.judge(text, &o)), Ok(Some(Some(v))) if v.verdict == "reject")
    }

    /// One step: what the network wrote, repaired if it has to be.  Every
    /// candidate is tried in turn; the first usable call (or an answer given
    /// instead) is the network's own move, and only when none can be used does
    /// the mediator step in, on the best one.
    fn next_call(&mut self, nets: &mut Nets, task: &Task, transcript: &str, attempt: usize) -> Result<Move, LoopError> {
        let candidates = if self.config.mediation == "always" {
            Vec::new()
        } else {
            self.emissions(nets, transcript, attempt)?
        };
        let mut broken: Option<ToolCall> = None;
        for emission in &candidates {
            if self.blamed(nets, emission) {
                // the negative network has seen this go wrong: try the next one
                self.avoided += 1;
                continue;
            }
            let answer_at = emission.find(ANSWER_OPEN);
            let call_at = emission.find(CALL_OPEN);
            if let Some(answer_at) = answer_at {
                if call_at.is_none_or(|c| answer_at < c) {
                    if let Some(answer) = find_answer(emission) {
                        return Ok((None, "model", emission.clone(), Some(answer)));
                    }
                }
            }
            let call = parse_call(emission, Some(self.toolbox));
            if let Some(call) = call {
                if call.ok() {
                    return Ok((Some(call), "model", emission.clone(), None));
                }
                if broken.is_none() {
                    broken = Some(call);
                }
            }
        }
        let emission = candidates.first().cloned().unwrap_or_default();
        if self.config.mediation == "never" {
            // a broken call is executed as a failure and learned from; nothing at all ends the attempt
            return Ok((broken, "model", emission, None));
        }
        let hint = broken
            .as_ref()
            .and_then(|b| b.error.as_ref())
            .map(|e| format!("Its call could not be read: {e}"));
        let mediated = mediate_call(
            self.client,
            &emission,
            self.toolbox,
            task,
            transcript,
            &self.config.agent_model,
            hint.as_deref(),
        )?;
        Ok((Some(mediated), "mediator", emission, None))
    }

    fn emit_step(&mut self, hooks: &mut Hooks, phase: &str, task: &Task, attempt: usize, step: &Step) {
        let record = Json::obj([
            ("kind", Json::str("step")),
            ("phase", Json::str(phase)),
            ("task", Json::str(task.id.clone())),
            ("attempt", Json::Int(attempt as i64 + 1)),
            ("step", Json::Int(step.index as i64 + 1)),
            ("tool", Json::str(step.call.name.clone())),
            ("arguments", step.call.arguments.clone()),
            ("source", Json::str(step.source.clone())),
            ("ok", Json::Bool(step.result.ok)),
            ("error", step.result.error.clone().map_or(Json::Null, Json::Str)),
            ("output", Json::str(head(&step.result.output, 300))),
            ("seconds", Json::Num(round_to(step.result.seconds, 3))),
        ]);
        self.emit(hooks, record);
    }

    fn emit_attempt(&mut self, hooks: &mut Hooks, phase: &str, round: usize, task: &Task, attempt: &Attempt) {
        let record = Json::obj([
            ("kind", Json::str("attempt")),
            ("phase", Json::str(phase)),
            ("round", Json::Int(round as i64)),
            ("task", Json::str(task.id.clone())),
            ("attempt", Json::Int(attempt.index as i64 + 1)),
            ("source", Json::str(attempt.source.clone())),
            ("calls", Json::Int(attempt.calls() as i64)),
            ("own_calls", Json::Int(attempt.own_calls() as i64)),
            ("autonomy", attempt.autonomy().map_or(Json::Null, Json::Num)),
            ("answer", attempt.answer.clone().map_or(Json::Null, Json::Str)),
            ("correct", Json::Bool(attempt.verdict.correct)),
            ("score", attempt.verdict.score.map_or(Json::Null, Json::Num)),
            (
                "met",
                Json::Arr(attempt.verdict.met.iter().map(|m| Json::Bool(*m)).collect()),
            ),
            ("critique", Json::str(attempt.verdict.critique.clone())),
            ("seconds", Json::Num(attempt.seconds)),
        ]);
        self.emit(hooks, record);
    }

    /// One attempt by the network: it writes calls, the toolbox runs them, it
    /// answers - and out of steps, whatever it writes next is its answer.
    pub fn solve_with_model(
        &mut self,
        nets: &mut Nets,
        task: &Task,
        index: usize,
        phase: &str,
        hooks: &mut Hooks,
    ) -> Result<Attempt, LoopError> {
        let started = Instant::now();
        let mut transcript = task_header(&task.prompt);
        if !task.seeds.is_empty() {
            transcript.push_str(&format!("START: {}\n", task.seeds.join(", ")));
        }
        let mut steps: Vec<Step> = Vec::new();
        let mut answer: Option<String> = None;
        for step_index in 0..self.config.max_steps {
            if (hooks.stop)() {
                break;
            }
            let (call, source, emission, said) = self.next_call(nets, task, &transcript, index)?;
            let Some(call) = call else {
                answer = said;
                break;
            };
            let result = self.toolbox.run(&call);
            self.note_urls(&result);
            transcript.push_str(&call_text(&call.name, &call.arguments));
            transcript.push_str(&format_observation(&result, Some(self.config.observation_chars)));
            let step = Step {
                index: step_index,
                call,
                result,
                source: source.to_string(),
                emission,
            };
            self.emit_step(hooks, phase, task, index, &step);
            steps.push(step);
        }
        if answer.is_none() && !(hooks.stop)() {
            // out of steps: whatever it writes now is its answer, tagged or not - an untrained
            // network answers with noise, and that noise is the failure the judge rejects
            let offered = self.emissions(nets, &transcript, index)?;
            answer = offered.iter().find_map(|e| find_answer(e));
            if answer.is_none() {
                answer = offered.first().map(|o| head(&collapse(o), 200).to_string());
            }
        }
        let mut text = transcript;
        if let Some(answer) = answer.as_deref().filter(|a| !a.is_empty()) {
            text.push_str(&answer_text(answer));
        }
        Ok(Attempt {
            index,
            source: "model".to_string(),
            steps,
            answer,
            text,
            verdict: Verdict::unjudged(),
            seconds: started.elapsed().as_secs_f64(),
            gap: None,
        })
    }

    /// The LLM demonstrates: the same tools, really called, and the answer it reaches.
    pub fn solve_with_teacher(
        &mut self,
        task: &Task,
        criteria: &[String],
        index: usize,
        feedback: Option<&str>,
    ) -> Result<Attempt, LoopError> {
        let started = Instant::now();
        let (steps, answer) = teach_task(
            self.client,
            task,
            self.toolbox,
            &self.config.agent_model,
            self.config.max_steps,
            criteria,
            feedback,
            self.config.observation_chars,
        )?;
        for step in &steps {
            self.note_urls(&step.result);
        }
        let transcript: Vec<TranscriptStep> = steps
            .iter()
            .map(|s| TranscriptStep {
                name: s.call.name.clone(),
                arguments: s.call.arguments.clone(),
                result: s.result.clone(),
            })
            .collect();
        let text = transcript_text(
            &task.prompt,
            &transcript,
            answer.as_deref(),
            Some(self.config.observation_chars),
        );
        Ok(Attempt {
            index,
            source: "teacher".to_string(),
            steps,
            answer,
            text,
            verdict: Verdict::unjudged(),
            seconds: started.elapsed().as_secs_f64(),
            gap: None,
        })
    }

    /// Marks an attempt against the criteria written before it started.
    pub fn judge(&self, task: &Task, criteria: &[String], attempt: &Attempt) -> Result<Verdict, LlmError> {
        let Some(answer) = attempt.answer.as_deref().filter(|a| !a.is_empty()) else {
            return Ok(Verdict {
                correct: false,
                critique: "no answer was given".to_string(),
                issues: criteria.to_vec(),
                judged_by: "answer".to_string(),
                ..Default::default()
            });
        };
        if !self.config.use_judge {
            let known = task.answer.as_deref().filter(|a| !a.is_empty());
            let correct = known.is_some_and(|k| answer.to_lowercase().contains(&k.trim().to_lowercase()));
            return Ok(Verdict {
                correct,
                score: Some(if correct { 10.0 } else { 0.0 }),
                met: Vec::new(),
                critique: if correct {
                    String::new()
                } else {
                    "the known answer is not in it".to_string()
                },
                issues: if correct { Vec::new() } else { criteria.to_vec() },
                judged_by: if known.is_some() { "answer" } else { "none" }.to_string(),
            });
        }
        let judged = judge_attempt(
            self.client.llm(),
            task,
            criteria,
            &attempt.text,
            Some(answer),
            self.config.judge_model_name(),
        )?;
        Ok(decide(&judged, criteria, Some(answer), self.config.strict))
    }

    /// How badly an attempt failed, in `[0, 1]`: the worse of how far below the
    /// pass score it scored and what share of the criteria it missed, with a
    /// floor of 0.1 - a failure that is only just a failure still trains.
    pub fn gap_of(&self, attempt: &Attempt, criteria: &[String]) -> f64 {
        if attempt.verdict.correct {
            return 0.0;
        }
        if attempt.answer.as_deref().is_none_or(str::is_empty) {
            return 1.0;
        }
        let mut gaps: Vec<f64> = Vec::new();
        if let Some(score) = attempt.verdict.score {
            let pass = self.config.pass_score;
            gaps.push(((pass - score) / pass).max(0.0));
        }
        let met = &attempt.verdict.met;
        if !met.is_empty() {
            gaps.push(met.iter().filter(|ok| !**ok).count() as f64 / met.len() as f64);
        } else if !criteria.is_empty() {
            gaps.push(1.0);
        }
        let worst = gaps.iter().copied().fold(None, |best: Option<f64>, g| {
            Some(match best {
                Some(b) if b >= g => b,
                _ => g,
            })
        });
        worst.unwrap_or(1.0).clamp(0.1, 1.0)
    }

    /// Nothing went right: the negative passes, heaviest failure first, then
    /// the inversion (Python's `_punish_weighted`), on whichever kind is
    /// loaded.
    fn punish_weighted(
        &self,
        nets: &mut Nets,
        bad: &[String],
        weights: &[f64],
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<EpochRecord>, String> {
        let cfg = &self.config;
        if weights.is_empty() {
            let o = feedback(cfg.neg_epochs, cfg.pos_epochs, cfg.neg_lr, cfg.pos_lr, cfg.batch_size);
            return nets.model(|m| kinds::punish(m, bad, &o, &mut |_| !stop()));
        }
        let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
        for (text, weight) in bad.iter().zip(weights) {
            let key = round_to(*weight, 3);
            match groups.iter_mut().find(|(w, _)| *w == key) {
                Some((_, group)) => group.push(text.clone()),
                None => groups.push((key, vec![text.clone()])),
            }
        }
        groups.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut records = Vec::new();
        for (weight, group) in groups {
            if stop() {
                break;
            }
            let written = nets.model(|m| -> Result<Vec<EpochRecord>, String> {
                let mut written = train_negative_phase(m, &group, cfg, weight, stop)?;
                // the weight goes into the history too, as Python tags the one record it keeps in both
                let first = m.history.len() - written.len().min(m.history.len());
                for (record, kept) in written.iter_mut().zip(&mut m.history[first..]) {
                    record.extra.push(("weight".to_string(), Json::Num(weight)));
                    kept.extra.push(("weight".to_string(), Json::Num(weight)));
                }
                Ok(written)
            })?;
            records.extend(written);
        }
        if !stop() {
            nets.model(|m| m.invert());
        }
        Ok(records)
    }

    /// Trains on the failures - the worse, the harder - then inverts, then
    /// fine-tunes on what was right (the GAN's blatant-failure handling applied
    /// to tool use).  The keys it adds to a record: `bad`, `good`, `action`,
    /// `neg_loss`, `pos_loss`, `failures`, `blatant`, `mode`, `flipped`,
    /// `boost_mean`, `boost_max`.
    pub fn learn(
        &mut self,
        nets: &mut Nets,
        failures: &[Failure],
        good: &[String],
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<(String, Json)>, String> {
        let cfg = self.config.clone();
        let mut good_all = unique(good);
        if cfg.read_reward && !self.pages.is_empty() {
            for page in &self.pages {
                if !good_all.contains(page) {
                    good_all.push(page.clone());
                }
            }
        }
        if cfg.replay {
            let extra: Vec<String> = self
                .replay_buffer
                .iter()
                .filter(|t| !good_all.contains(t))
                .cloned()
                .collect();
            if cfg.replay_limit > 0 {
                good_all.extend(extra[extra.len().saturating_sub(cfg.replay_limit)..].iter().cloned());
            }
        }
        // a transcript that failed twice keeps its worst gap
        let mut seen: Vec<(String, f64)> = Vec::new();
        for failure in failures {
            if failure.text.is_empty() || good_all.contains(&failure.text) {
                continue;
            }
            match seen.iter_mut().find(|(t, _)| *t == failure.text) {
                Some(slot) => slot.1 = slot.1.max(failure.gap),
                None => seen.push((failure.text.clone(), failure.gap.max(0.0))),
            }
        }
        let mut bad: Vec<String> = seen.iter().map(|(t, _)| t.clone()).collect();
        let mut gaps: Vec<f64> = seen.iter().map(|(_, g)| *g).collect();
        let blatant: Vec<String> = seen
            .iter()
            .filter(|(_, g)| *g > cfg.blatant_margin)
            .map(|(t, _)| t.clone())
            .collect();
        let mut result = vec![
            ("bad".to_string(), Json::Int(bad.len() as i64)),
            ("good".to_string(), Json::Int(good_all.len() as i64)),
            ("action".to_string(), Json::Null),
            ("neg_loss".to_string(), Json::Null),
            ("pos_loss".to_string(), Json::Null),
            ("failures".to_string(), Json::Int(bad.len() as i64)),
            ("blatant".to_string(), Json::Int(blatant.len() as i64)),
            ("mode".to_string(), Json::str(cfg.blatant_mode.clone())),
            ("flipped".to_string(), Json::Int(0)),
            ("boost_mean".to_string(), Json::Null),
            ("boost_max".to_string(), Json::Null),
        ];
        if bad.is_empty() && good_all.is_empty() {
            return Ok(result);
        }
        let mut weights: Vec<f64> = Vec::new();
        if !bad.is_empty() && cfg.blatant_mode == "fail_invert" {
            weights = gaps
                .iter()
                .map(|gap| cfg.blatant_boost.min(1.0 + gap / cfg.blatant_margin))
                .collect();
        } else if !bad.is_empty() && (cfg.blatant_mode == "activation" || cfg.blatant_mode == "state") {
            // the local variant: every edge of a failed path loses reward, the worse the more
            let amounts: Vec<f64> = gaps
                .iter()
                .map(|gap| (gap / (2.0 * cfg.blatant_margin)).min(1.0))
                .collect();
            let (flipped, amount_mean) = nets.model(|m| invert_paths(m, &bad, &cfg.blatant_mode, &amounts))?;
            set(&mut result, "flipped", Json::Int(flipped));
            set(&mut result, "boost_mean", Json::Num(amount_mean));
            set(&mut result, "boost_max", Json::Num(max_of(&amounts)));
            let keep: Vec<(String, f64)> = bad
                .iter()
                .cloned()
                .zip(gaps.iter().copied())
                .filter(|(t, _)| !blatant.contains(t))
                .collect();
            bad = keep.iter().map(|(t, _)| t.clone()).collect();
            gaps = keep.iter().map(|(_, g)| *g).collect();
        }
        let _ = gaps;
        if !weights.is_empty() {
            set(
                &mut result,
                "boost_mean",
                Json::Num(weights.iter().sum::<f64>() / weights.len() as f64),
            );
            set(&mut result, "boost_max", Json::Num(max_of(&weights)));
        }
        let mut o = feedback(cfg.neg_epochs, cfg.pos_epochs, cfg.neg_lr, cfg.pos_lr, cfg.batch_size);
        if !bad.is_empty() && !good_all.is_empty() {
            o.bad_weights = (!weights.is_empty()).then(|| weights.clone());
            let (negative, positive) = nets.model(|m| kinds::two_nrl(m, &bad, &good_all, &o, &mut |_| !stop()))?;
            set(&mut result, "action", Json::str("2nrl"));
            set(&mut result, "neg_loss", last_loss(&negative));
            set(&mut result, "pos_loss", last_loss(&positive));
        } else if !good_all.is_empty() {
            let records = nets.model(|m| kinds::reward(m, &good_all, &o, &mut |_| !stop()))?;
            set(&mut result, "action", Json::str("reward"));
            set(&mut result, "pos_loss", last_loss(&records));
        } else if !bad.is_empty() {
            let records = self.punish_weighted(nets, &bad, &weights, stop)?;
            set(&mut result, "action", Json::str("punish"));
            set(&mut result, "neg_loss", last_loss(&records));
        }
        for text in good {
            if !self.replay_buffer.contains(text) {
                self.replay_buffer.push(text.clone());
            }
        }
        if cfg.replay_limit > 0 && self.replay_buffer.len() > cfg.replay_limit {
            let excess = self.replay_buffer.len() - cfg.replay_limit;
            self.replay_buffer.drain(..excess);
        }
        self.pages.clear();
        Ok(result)
    }

    /// Hands a task's failures to the negative network - the judge is its tutor
    /// - with the correct run of the same task as the correction.
    fn teach_negative(
        &self,
        nets: &mut Nets,
        attempts: &[Attempt],
        correct: Option<&Attempt>,
    ) -> Result<Option<TeachReport>, String> {
        if !nets.has_negative() || attempts.is_empty() {
            return Ok(None);
        }
        let source = if self.exploring { "explore" } else { "agent" };
        let correction = correct.map(|a| a.text.as_str());
        let o = TeachOptions::default();
        nets.negative(|n| teach_agent(n, attempts, correction, self.config.pass_score, source, &o))?
            .transpose()
    }

    /// Criteria, attempts, judging and (on failure) the teacher's
    /// demonstration for one task: `(record, failures, correct texts)`.
    pub fn run_task(
        &mut self,
        nets: &mut Nets,
        task: &Task,
        phase: &str,
        round: usize,
        hooks: &mut Hooks,
    ) -> Result<TaskOutcome, LoopError> {
        let started = Instant::now();
        let cfg = self.config.clone();
        let criteria = self.criteria_for(task)?;
        let record = Json::obj([
            ("kind", Json::str("criteria")),
            ("phase", Json::str(phase)),
            ("round", Json::Int(round as i64)),
            ("task", Json::str(task.id.clone())),
            ("prompt", Json::str(head(&task.prompt, 300))),
            ("criteria", Json::strs(criteria.clone())),
        ]);
        self.emit(hooks, record);
        let mut attempts: Vec<Attempt> = Vec::new();
        if phase == "model" {
            for index in 0..cfg.model_attempts {
                if (hooks.stop)() {
                    break;
                }
                let mut attempt = self.solve_with_model(nets, task, index, phase, hooks)?;
                attempt.verdict = self.judge(task, &criteria, &attempt)?;
                attempt.gap = Some(self.gap_of(&attempt, &criteria));
                self.emit_attempt(hooks, phase, round, task, &attempt);
                let correct = attempt.verdict.correct;
                attempts.push(attempt);
                if correct {
                    break;
                }
            }
        }
        let mut taught = false;
        let any_correct = attempts.iter().any(|a| a.verdict.correct);
        if (phase == "teacher" || (cfg.teach_on_failure && !any_correct)) && !(hooks.stop)() {
            let mut feedback = attempts.last().map(Attempt::feedback);
            for _ in 0..cfg.teacher_attempts {
                let mut attempt = self.solve_with_teacher(task, &criteria, attempts.len(), feedback.as_deref())?;
                attempt.verdict = self.judge(task, &criteria, &attempt)?;
                attempt.gap = Some(self.gap_of(&attempt, &criteria));
                taught = true;
                self.emit_attempt(hooks, phase, round, task, &attempt);
                let done = attempt.verdict.correct || (hooks.stop)();
                feedback = Some(attempt.feedback());
                attempts.push(attempt);
                if done {
                    break;
                }
            }
        }
        let correct = attempts.iter().find(|a| a.verdict.correct);
        let good: Vec<String> = attempts
            .iter()
            .filter(|a| a.verdict.correct)
            .map(|a| a.text.clone())
            .collect();
        let failures: Vec<Failure> = attempts
            .iter()
            .filter(|a| !a.verdict.correct)
            .map(|a| Failure {
                text: a.text.clone(),
                gap: a.gap.unwrap_or_else(|| self.gap_of(a, &criteria)),
                task: task.id.clone(),
                source: a.source.clone(),
            })
            .collect();
        if let Some(correct) = correct {
            set_solved(&mut self.solved, &task.id, &correct.text);
        }
        let by_model: Vec<&Attempt> = attempts.iter().filter(|a| a.source == "model").collect();
        let own: usize = by_model.iter().map(|a| a.own_calls()).sum();
        let calls: usize = by_model.iter().map(|a| a.calls()).sum();
        let shown = correct.or(attempts.last());
        let gap_max = failures.iter().map(|f| f.gap).fold(None, |best: Option<f64>, g| {
            Some(match best {
                Some(b) if b >= g => b,
                _ => g,
            })
        });
        let mut record = vec![
            ("kind".to_string(), Json::str("task")),
            ("phase".to_string(), Json::str(phase)),
            ("round".to_string(), Json::Int(round as i64)),
            ("task".to_string(), Json::str(task.id.clone())),
            ("prompt".to_string(), Json::str(head(&task.prompt, 200))),
            ("criteria".to_string(), Json::Int(criteria.len() as i64)),
            ("attempts".to_string(), Json::Int(attempts.len() as i64)),
            ("correct".to_string(), Json::Bool(correct.is_some())),
            (
                "solved_by".to_string(),
                correct.map_or(Json::Null, |a| Json::str(a.source.clone())),
            ),
            (
                "model_solved".to_string(),
                Json::Bool(attempts.iter().any(|a| a.source == "model" && a.verdict.correct)),
            ),
            ("taught".to_string(), Json::Bool(taught)),
            ("calls".to_string(), Json::Int(calls as i64)),
            ("own_calls".to_string(), Json::Int(own as i64)),
            (
                "autonomy".to_string(),
                if calls > 0 {
                    Json::Num(own as f64 / calls as f64)
                } else {
                    Json::Null
                },
            ),
            (
                "answer".to_string(),
                shown.and_then(|a| a.answer.clone()).map_or(Json::Null, Json::Str),
            ),
            (
                "score".to_string(),
                shown.and_then(|a| a.verdict.score).map_or(Json::Null, Json::Num),
            ),
            (
                "issues".to_string(),
                Json::strs(
                    shown
                        .map(|a| a.verdict.issues.iter().take(4).cloned().collect::<Vec<_>>())
                        .unwrap_or_default(),
                ),
            ),
            ("bad".to_string(), Json::Int(failures.len() as i64)),
            ("good".to_string(), Json::Int(good.len() as i64)),
            ("gap_max".to_string(), gap_max.map_or(Json::Null, Json::Num)),
            ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
        ];
        if let Some(taught) = self.teach_negative(nets, &attempts, correct)? {
            record.push(("negative_blamed".to_string(), Json::Int(taught.blamed as i64)));
            record.push(("negative_edges".to_string(), Json::Int(taught.edges as i64)));
            record.push((
                "negative_reasons".to_string(),
                Json::Obj(
                    taught
                        .reasons
                        .iter()
                        .map(|(r, n)| (r.clone(), Json::Int(*n as i64)))
                        .collect(),
                ),
            ));
            record.push(("negative_cleared".to_string(), Json::Int(taught.cleared as i64)));
        }
        Ok((record, failures, good))
    }

    /// Every round and phase over `tasks`; the task and round records (the
    /// criteria, steps and attempts go to the progress hook too).
    pub fn run(&mut self, nets: &mut Nets, tasks: &[Task], hooks: &mut Hooks) -> Result<Vec<Json>, LoopError> {
        if tasks.is_empty() {
            return Err(LoopError::Other("no tasks to solve".to_string()));
        }
        let cfg = self.config.clone();
        let mut records = Vec::new();
        let mut step = 0usize;
        for round in 1..=cfg.rounds {
            for phase in &cfg.phases {
                let started = Instant::now();
                let (mut round_bad, mut round_good) = (Vec::new(), Vec::new());
                let (mut solved, mut model_solved, mut done) = (0, 0, 0);
                for task in tasks {
                    if (hooks.stop)() {
                        break;
                    }
                    let begun = Instant::now();
                    let (mut record, bad, good) = self.run_task(nets, task, phase, round, hooks)?;
                    if cfg.twonrl_per == "task" {
                        for (key, value) in self.learn(nets, &bad, &good, hooks.stop)? {
                            set(&mut record, &key, value);
                        }
                    } else {
                        round_bad.extend(bad);
                        round_good.extend(good);
                    }
                    done += 1;
                    set(&mut record, "seconds", Json::Num(begun.elapsed().as_secs_f64()));
                    let record = Json::Obj(record);
                    if record.at("correct") == &Json::Bool(true) {
                        solved += 1;
                    }
                    if record.at("model_solved") == &Json::Bool(true) {
                        model_solved += 1;
                    }
                    crate::log_info!(
                        LOG,
                        "{phase} {}: {} by {}",
                        task.id,
                        if record.at("correct") == &Json::Bool(true) {
                            "solved"
                        } else {
                            "unsolved"
                        },
                        record.at("solved_by").as_str().unwrap_or("-")
                    );
                    self.emit(hooks, record.clone());
                    records.push(record.clone());
                    step += 1;
                    if let Some(manager) = hooks.checkpoints.filter(|_| cfg.checkpoint_every > 0) {
                        if step % cfg.checkpoint_every == 0 {
                            let metrics =
                                Json::obj(["phase", "round", "task", "correct"].map(|k| (k, record.at(k).clone())));
                            nets.model(|m| manager.save(m, step as i64, "agent", Some(metrics)))?;
                        }
                    }
                }
                let learned = if cfg.twonrl_per == "round"
                    && (!round_bad.is_empty() || !round_good.is_empty())
                    && !(hooks.stop)()
                {
                    self.learn(nets, &round_bad, &round_good, hooks.stop)?
                } else {
                    Vec::new()
                };
                let mut summary = vec![
                    ("kind".to_string(), Json::str("round")),
                    ("phase".to_string(), Json::str(phase.clone())),
                    ("round".to_string(), Json::Int(round as i64)),
                    ("tasks".to_string(), Json::Int(done)),
                    ("solved".to_string(), Json::Int(solved)),
                    ("model_solved".to_string(), Json::Int(model_solved)),
                    ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
                ];
                summary.extend(learned);
                let summary = Json::Obj(summary);
                self.emit(hooks, summary.clone());
                records.push(summary);
                if (hooks.stop)() {
                    return Ok(records);
                }
            }
        }
        Ok(records)
    }

    /// The network chooses what to look into next - it continues `TASK:`, the
    /// way every transcript it has learned starts - and the LLM turns that into
    /// an answerable question.
    pub fn propose(&mut self, nets: &mut Nets, index: usize) -> Result<(Task, String), LoopError> {
        let seed = task_header("").trim().to_string();
        let cfg = &self.config;
        let emission = nets.model(|m| {
            m.predict(
                &seed,
                &PredictOptions {
                    length: 1,
                    mode: "sample".to_string(),
                    temperature: cfg.temperature.max(0.8),
                    max_length: Some(cfg.max_length),
                    ..Default::default()
                },
            )
        })?;
        let emission = emission.best.text;
        let task = propose_task(
            self.client.llm(),
            &emission,
            &self.config.agent_model,
            &self.frontier,
            &self.visited,
            index,
        )?;
        if !task.seeds.is_empty() {
            self.frontier.retain(|u| !task.seeds.contains(u));
        }
        Ok((task, emission))
    }

    /// Browses on the network's own initiative: it picks each task, the loop
    /// judges and teaches it.  `steps` 0 runs until the stop hook says so.
    pub fn explore(&mut self, nets: &mut Nets, steps: usize, hooks: &mut Hooks) -> Result<Vec<Json>, LoopError> {
        self.exploring = true;
        let cfg = self.config.clone();
        let mut records = Vec::new();
        let mut index = 0usize;
        while !(hooks.stop)() && (steps == 0 || index < steps) {
            index += 1;
            let started = Instant::now();
            let (task, emission) = self.propose(nets, index)?;
            let proposal = Json::obj([
                ("kind", Json::str("proposal")),
                ("step", Json::Int(index as i64)),
                ("task", Json::str(task.id.clone())),
                ("prompt", Json::str(task.prompt.clone())),
                ("emission", Json::str(head(&collapse(&emission), 200))),
                ("seeds", Json::strs(task.seeds.clone())),
                ("frontier", Json::Int(self.frontier.len() as i64)),
                ("visited", Json::Int(self.visited.len() as i64)),
            ]);
            self.emit(hooks, proposal);
            if (hooks.stop)() {
                break;
            }
            let (mut record, bad, good) = self.run_task(nets, &task, "model", index, hooks)?;
            for (key, value) in self.learn(nets, &bad, &good, hooks.stop)? {
                set(&mut record, &key, value);
            }
            set(&mut record, "kind", Json::str("explore"));
            set(&mut record, "step", Json::Int(index as i64));
            set(&mut record, "frontier", Json::Int(self.frontier.len() as i64));
            set(&mut record, "visited", Json::Int(self.visited.len() as i64));
            set(&mut record, "seconds", Json::Num(started.elapsed().as_secs_f64()));
            let record = Json::Obj(record);
            self.emit(hooks, record.clone());
            records.push(record.clone());
            if let Some(manager) = hooks.checkpoints.filter(|_| cfg.checkpoint_every > 0) {
                if index % cfg.checkpoint_every == 0 {
                    let metrics = Json::obj(["task", "correct", "autonomy"].map(|k| (k, record.at(k).clone())));
                    nets.model(|m| manager.save(m, index as i64, "explore", Some(metrics)))?;
                }
            }
        }
        Ok(records)
    }
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Python's `model.train(group, epochs=neg_epochs, lr=neg_lr * weight,
/// phase="negative", batch_size=...)` on whichever kind is loaded: the sine
/// model descends at the scaled rate, the count and phase models count the
/// failures under the negative phase (the inversion that follows turns them
/// into avoidance), and the negative network - for which training is blaming
/// - blames them.
fn train_negative_phase(
    model: &mut Model,
    texts: &[String],
    cfg: &AgentConfig,
    weight: f64,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Vec<EpochRecord>, String> {
    let config = crate::radix::TrainConfig {
        epochs: cfg.neg_epochs,
        lr: cfg.neg_lr * weight,
        batch_size: cfg.batch_size,
        ..Default::default()
    };
    config.validate()?;
    let on_epoch = &mut |_: &EpochRecord| !stop();
    match model.kind() {
        "radix" => model.radix_train(texts, &config, Some("negative"), on_epoch),
        "resonant" => model.resonant_train(
            texts,
            config.epochs,
            config.auto_compress,
            Some("negative"),
            &config.plan,
            on_epoch,
        ),
        "negative" => {
            let settings = kinds::TrainSettings {
                config,
                ..Default::default()
            };
            kinds::train(model, texts, &settings, on_epoch)
        }
        _ => {
            let options = TrainOptions {
                epochs: config.epochs,
                auto_compress: config.auto_compress,
                phase: Some("negative".to_string()),
                chunk_size: 0,
                plan: config.plan.clone(),
            };
            model.train_with(texts, &options, on_epoch)
        }
    }
}

/// Python's `model.invert_paths(bad, mode=blatant_mode, amounts=...)` on
/// whichever kind is loaded ([`kinds::invert_paths`]), as `(flipped,
/// amount_mean)`: the sine model moves every other node of a failed path
/// toward its negation, the phase model rotates or decoheres the path's edges,
/// the count model takes reward off them and the negative network blames them,
/// each at its own default strength, since Python's agent passes none (2 for
/// the count model, 1 for the negative network).
fn invert_paths(model: &mut Model, texts: &[String], mode: &str, amounts: &[f64]) -> Result<(i64, f64), String> {
    let strength = if model.kind() == "negative" { 1.0 } else { 2.0 };
    let outcome = kinds::invert_paths(model, texts, mode, Some(amounts), strength)?;
    Ok((
        outcome.at("flipped").as_i64().unwrap_or(0),
        outcome.at("amount_mean").as_f64().unwrap_or(0.0),
    ))
}

// -- the command line ------------------------------------------------------------------------------

/// The agent settings of `agent` and `explore`, from their flags.
fn config_from_args(ctx: &Ctx, phase: &str, every: usize) -> Result<AgentConfig, String> {
    let args = &ctx.args;
    let d = AgentConfig::default();
    let whole = |name: &str, default: usize, minimum: usize| -> Result<usize, String> {
        crate::llm::count_flag(ctx, name, default as i64, minimum as i64)
    };
    let config = AgentConfig {
        agent_model: args
            .get("agent-model")
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .unwrap_or(d.agent_model),
        judge_model: args.get("judge-model").map(str::to_string),
        phases: if phase == "both" {
            PHASES.iter().map(|p| p.to_string()).collect()
        } else {
            vec![phase.to_string()]
        },
        rounds: whole("rounds", d.rounds, 1)?,
        max_steps: whole("max-steps", d.max_steps, 1)?,
        model_attempts: whole("model-attempts", d.model_attempts, 1)?,
        teacher_attempts: whole("teacher-attempts", d.teacher_attempts, 1)?,
        first_attempt_dijkstra: !args.on("sample-first"),
        candidates: whole("candidates", d.candidates, 1)?,
        temperature: crate::llm::nonneg_flag(ctx, "temperature", d.temperature)?,
        max_length: whole("max-length", d.max_length, 1)?,
        mediation: choice(ctx, "mediation", &d.mediation, MEDIATION)?,
        criteria_count: whole("criteria", d.criteria_count, 1)?,
        strict: !args.on("lenient"),
        use_judge: !args.on("no-judge"),
        teach_on_failure: !args.on("no-teach"),
        observation_chars: whole("observation-chars", d.observation_chars, 0)?,
        twonrl_per: if ctx.command == "explore" {
            d.twonrl_per.clone()
        } else {
            choice(ctx, "twonrl-per", &d.twonrl_per, &["task", "round"])?
        },
        replay: !args.on("no-replay"),
        replay_limit: d.replay_limit,
        read_reward: args.on("read-reward"),
        avoid_blamed: !args.on("no-avoid"),
        avoid_threshold: d.avoid_threshold,
        blatant_mode: choice(ctx, "blatant-mode", &d.blatant_mode, crate::gan::BLATANT_MODES)?,
        blatant_margin: number_at_least(ctx, "blatant-margin", d.blatant_margin, 0.001)?,
        blatant_boost: number_at_least(ctx, "blatant-boost", d.blatant_boost, 1.0)?,
        pass_score: d.pass_score,
        neg_epochs: whole("neg-epochs", d.neg_epochs, 0)?,
        pos_epochs: whole("pos-epochs", d.pos_epochs, 0)?,
        neg_lr: crate::llm::nonneg_flag(ctx, "neg-lr", d.neg_lr)?,
        pos_lr: crate::llm::nonneg_flag(ctx, "pos-lr", d.pos_lr)?,
        batch_size: whole("batch-size", d.batch_size, 1)?,
        checkpoint_every: every,
        seed: ctx.seed,
    };
    config.validate()?;
    Ok(config)
}

/// What `agent` and `explore` share: the settings, the tools, the LLM, the
/// model and (with `--blame`) the negative network, and the report afterwards.
fn run_command(ctx: &Ctx, tasks: Option<(Vec<Task>, String)>) -> Result<(), String> {
    let args = &ctx.args;
    let exploring = tasks.is_none();
    let schedule = Schedule::from_args(args)?;
    let phase = if exploring {
        "model".to_string()
    } else {
        choice(ctx, "phase", "model", &["model", "teacher", "both"])?
    };
    let every = if schedule.manager().is_some() {
        schedule.every
    } else {
        0
    };
    let config = config_from_args(ctx, &phase, every)?;
    let toolbox = crate::tools::build_toolbox(args)?;
    let client = OllamaClient::new(
        &args.str("url", ""),
        &config.agent_model,
        crate::llm::timeout_flag(ctx)?,
    )?;
    let origin = origin_json(ctx);
    let mut model = ctx.open(false)?;
    let mut negative = if args.on("blame") {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let out = ctx.out.clone().unwrap_or_else(|| ctx.model_path.clone());
    let steps = if exploring {
        crate::llm::count_flag(ctx, "steps", 10, 0)?
    } else {
        0
    };
    crate::log_info!(
        LOG,
        "{}; tools {}: {}; ollama {} at {}; mediation {}",
        match &tasks {
            Some((tasks, path)) => format!(
                "{} task(s) from {path}, {} round(s) of {}",
                tasks.len(),
                config.rounds,
                config.phases.join(" -> ")
            ),
            None => format!(
                "chosen by the network: {} step(s)",
                if steps == 0 {
                    "until stopped".to_string()
                } else {
                    steps.to_string()
                }
            ),
        },
        toolbox.len(),
        toolbox.names().join(", "),
        config.agent_model,
        client.url,
        config.mediation
    );
    let mut trainer = AgentTrainer::new(&client, &toolbox, config.clone())?;
    if exploring {
        trainer.frontier.extend(args.all("seed-url"));
    }
    let records = {
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: negative.as_mut(),
        };
        let mut progress = |record: &Json| {
            let kind = record.at("kind").as_str().unwrap_or("");
            match kind {
                "proposal" => crate::log_info!(
                    LOG,
                    "  {} chose: {}",
                    record.at("task").as_str().unwrap_or(""),
                    record.at("prompt").as_str().unwrap_or("")
                ),
                "step" => crate::log_info!(
                    LOG,
                    "    {} call {} [{}] {} {}",
                    record.at("task").as_str().unwrap_or(""),
                    record.at("step").as_i64().unwrap_or(0),
                    record.at("source").as_str().unwrap_or(""),
                    record.at("tool").as_str().unwrap_or(""),
                    if record.at("ok") == &Json::Bool(true) {
                        "ok"
                    } else {
                        "FAILED"
                    }
                ),
                "attempt" => crate::log_info!(
                    LOG,
                    "    {} attempt {} [{}] {}",
                    record.at("task").as_str().unwrap_or(""),
                    record.at("attempt").as_i64().unwrap_or(0),
                    record.at("source").as_str().unwrap_or(""),
                    if record.at("correct") == &Json::Bool(true) {
                        "correct"
                    } else {
                        "failed"
                    }
                ),
                _ => {}
            }
        };
        let mut hooks = Hooks {
            progress: &mut progress,
            stop: &|| false,
            checkpoints: schedule.manager(),
        };
        match &tasks {
            Some((tasks, _)) => trainer.run(&mut nets, tasks, &mut hooks)?,
            None => trainer.explore(&mut nets, steps, &mut hooks)?,
        }
    };
    model.save(&out)?;
    let done: Vec<&Json> = records
        .iter()
        .filter(|r| matches!(r.at("kind").as_str(), Some("task") | Some("explore")))
        .collect();
    let count = |key: &str| done.iter().filter(|r| r.at(key) == &Json::Bool(true)).count();
    let total = |key: &str| done.iter().map(|r| r.at(key).as_i64().unwrap_or(0)).sum::<i64>();
    let (solved, by_model) = (count("correct"), count("model_solved"));
    let (calls, own, fails) = (total("calls"), total("own_calls"), total("failures"));
    crate::log_info!(
        LOG,
        "{solved}/{} task(s) solved ({by_model} by the network); {own}/{calls} tool call(s) written by the network \
         itself; {fails} failure(s) trained on",
        done.len()
    );
    let negative_doc = match negative.as_mut() {
        None => Json::Null,
        Some(negative) => {
            let mut pairs = vec![
                ("blamed".to_string(), Json::Int(total("negative_blamed"))),
                ("edges".to_string(), Json::Int(total("negative_edges"))),
            ];
            if let Json::Obj(saved) = save_negative(negative, &ctx.negative_path())? {
                pairs.extend(saved);
            }
            Json::Obj(pairs)
        }
    };
    let mut doc = vec![
        ("model".to_string(), origin),
        ("out".to_string(), Json::str(out.clone())),
        ("config".to_string(), config.to_json()),
        ("tools".to_string(), Json::strs(toolbox.names())),
        ("records".to_string(), Json::Arr(records.clone())),
        ("attempts".to_string(), Json::Arr(trainer.history.clone())),
        ("solved".to_string(), Json::Int(solved as i64)),
        ("model_solved".to_string(), Json::Int(by_model as i64)),
        ("calls".to_string(), Json::Int(calls)),
        ("own_calls".to_string(), Json::Int(own)),
        (
            "autonomy".to_string(),
            if calls > 0 {
                Json::Num(own as f64 / calls as f64)
            } else {
                Json::Null
            },
        ),
        ("failures".to_string(), Json::Int(fails)),
        (
            "criteria".to_string(),
            Json::Obj(
                trainer
                    .criteria
                    .iter()
                    .map(|(id, c)| (id.clone(), Json::strs(c.clone())))
                    .collect(),
            ),
        ),
        ("solutions".to_string(), solutions_json(&trainer.solved)),
        ("interrupted".to_string(), Json::Bool(false)),
        ("saved".to_string(), crate::llm::saved(&out)),
        ("stats".to_string(), stats(&model)),
        ("avoided".to_string(), Json::Int(trainer.avoided as i64)),
        ("negative".to_string(), negative_doc),
    ];
    if exploring {
        doc.push(("frontier".to_string(), Json::strs(trainer.frontier.clone())));
        doc.push(("visited".to_string(), Json::strs(trainer.visited.clone())));
    }
    let doc = Json::Obj(doc);
    if let Some(report) = args.get("report").filter(|r| !r.is_empty()) {
        let kept = Json::obj(
            [
                "config",
                "tools",
                "records",
                "criteria",
                "solutions",
                "solved",
                "autonomy",
            ]
            .map(|k| (k, doc.at(k).clone())),
        );
        std::fs::write(report, kept.render(2)).map_err(|err| format!("cannot write {report}: {err}"))?;
        crate::log_info!(LOG, "wrote report to {report}");
    }
    ctx.emit(doc);
    Ok(())
}

/// `radixnet agent --tasks FILE`: the network solves each task by writing tool
/// calls; the LLM (Ollama) writes the criteria, mediates, judges and teaches;
/// 2NRL learns - then the model is saved to `--out` (default `--model`), and
/// with `--blame` the negative network beside it.  The flags are Python's
/// (`--phase`, `--rounds`, `--twonrl-per`, `--agent-model`, `--judge-model`,
/// `--url`, `--timeout`, `--criteria`, `--lenient`, `--no-judge`,
/// `--mediation`, `--no-teach`, `--max-steps`, `--model-attempts`,
/// `--teacher-attempts`, `--sample-first`, `--temperature`, `--max-length`,
/// `--observation-chars`, `--read-reward`, `--no-replay`, `--blatant-mode`,
/// `--blatant-margin`, `--blatant-boost`, `--blame`, `--no-avoid`, the 2NRL
/// epochs, the tool and checkpoint options, `--report`).
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let path = ctx
        .args
        .get("tasks")
        .filter(|p| !p.is_empty())
        .ok_or("agent needs --tasks FILE (one question per line, or .json / .jsonl objects)")?
        .to_string();
    let tasks = load_tasks(&path).map_err(|err| format!("cannot load tasks from {path}: {err}"))?;
    run_command(ctx, Some((tasks, path)))
}

/// `radixnet explore`: the network picks every task itself (`--steps`, 0 =
/// until the process is stopped; `--seed-url` puts a page on the frontier to
/// start from), with the agent's flags otherwise.
pub fn explore_cli(ctx: &Ctx) -> Result<(), String> {
    run_command(ctx, None)
}

// -- the server ------------------------------------------------------------------------------------

/// What the server keeps for this area: the criteria, step, attempt, task and
/// round records of every agent and explore run, oldest first.
#[derive(Default)]
pub struct State {
    history: Mutex<Vec<Json>>,
}

impl State {
    /// Reads the `serve` command's flags for this area (the tool flags are
    /// [`crate::tools::State`]'s).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    fn push(&self, record: Json) {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).push(record);
    }

    /// Every record so far.
    pub fn history(&self) -> Vec<Json> {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// This area's routes:
/// `POST /api/agent/start`
/// `POST /api/agent/solve`
/// `POST /api/agent/explore`
/// `POST /api/agent/criteria`
/// `GET /api/agent/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/agent/start", start_route);
    server.route("POST", "/api/agent/explore", explore_route);
    server.route("GET", "/api/agent/history", history_route);
    server.route("POST", "/api/agent/criteria", criteria_route);
    server.route("POST", "/api/agent/solve", solve_route);
}

/// The agent settings of a request body; every one optional, every bad one a 400.
fn config_from_fields(f: &Fields) -> Result<AgentConfig, ApiError> {
    let d = AgentConfig::default();
    let phase = f.text_or("phase", "model")?.trim().to_lowercase();
    if !matches!(phase.as_str(), "model" | "teacher" | "both") {
        return Err(ApiError::bad_request(format!(
            "'phase' must be 'model', 'teacher' or 'both' (got {})",
            python_repr(&phase)
        )));
    }
    let mediation = f.text_or("mediation", &d.mediation)?.trim().to_lowercase();
    if !MEDIATION.contains(&mediation.as_str()) {
        return Err(ApiError::bad_request(format!(
            "'mediation' must be one of {} (got {})",
            MEDIATION.join(", "),
            python_repr(&mediation)
        )));
    }
    let blatant = f.text_or("blatant_mode", &d.blatant_mode)?.trim().to_lowercase();
    if !crate::gan::BLATANT_MODES.contains(&blatant.as_str()) {
        return Err(ApiError::bad_request(format!(
            "'blatant_mode' must be one of {} (got {})",
            crate::gan::BLATANT_MODES.join(", "),
            python_repr(&blatant)
        )));
    }
    let twonrl_per = f.text_or("twonrl_per", &d.twonrl_per)?.trim().to_lowercase();
    if twonrl_per != "task" && twonrl_per != "round" {
        return Err(ApiError::bad_request(format!(
            "'twonrl_per' must be 'task' or 'round' (got {})",
            python_repr(&twonrl_per)
        )));
    }
    let agent_model = match f.text("agent_model")?.filter(|m| !m.is_empty()) {
        Some(model) => model,
        None => f.text("model")?.filter(|m| !m.is_empty()).unwrap_or(d.agent_model),
    };
    let config = AgentConfig {
        agent_model,
        judge_model: f.text("judge_model")?,
        phases: if phase == "both" {
            PHASES.iter().map(|p| p.to_string()).collect()
        } else {
            vec![phase]
        },
        rounds: f.count_or("rounds", d.rounds, 1)?,
        max_steps: f.count_or("max_steps", d.max_steps, 1)?,
        model_attempts: f.count_or("model_attempts", d.model_attempts, 1)?,
        teacher_attempts: f.count_or("teacher_attempts", d.teacher_attempts, 1)?,
        first_attempt_dijkstra: f.flag("first_attempt_dijkstra", d.first_attempt_dijkstra)?,
        candidates: d.candidates,
        temperature: f.number_or("temperature", d.temperature, Some(0.0))?,
        max_length: f.count_or("max_length", d.max_length, 1)?,
        mediation,
        criteria_count: f.count_or("criteria", d.criteria_count, 1)?,
        strict: f.flag("strict", d.strict)?,
        use_judge: f.flag("judge", d.use_judge)?,
        teach_on_failure: f.flag("teach", d.teach_on_failure)?,
        observation_chars: f.count_or("observation_chars", d.observation_chars, 0)?,
        twonrl_per,
        replay: f.flag("replay", d.replay)?,
        replay_limit: d.replay_limit,
        read_reward: f.flag("read_reward", d.read_reward)?,
        avoid_blamed: f.flag("avoid_blamed", d.avoid_blamed)?,
        avoid_threshold: f.number_or("avoid_threshold", d.avoid_threshold, Some(0.0))?,
        blatant_mode: blatant,
        blatant_margin: f.number_or("blatant_margin", d.blatant_margin, Some(0.001))?,
        blatant_boost: f.number_or("blatant_boost", d.blatant_boost, Some(1.0))?,
        pass_score: d.pass_score,
        neg_epochs: f.count_or("neg_epochs", d.neg_epochs, 0)?,
        pos_epochs: f.count_or("pos_epochs", d.pos_epochs, 0)?,
        neg_lr: f.number_or("neg_lr", d.neg_lr, Some(0.0))?,
        pos_lr: f.number_or("pos_lr", d.pos_lr, Some(0.0))?,
        batch_size: f.count_or("batch_size", d.batch_size, 1)?,
        checkpoint_every: f.count_or("checkpoint_every", 0, 0)?,
        seed: f.integer_or("seed", 0, None)?,
    };
    config.validate().map_err(ApiError::bad_request)?;
    Ok(config)
}

/// The tasks of a request: `tasks` (strings / objects), `tasks_text` (one per
/// line) and `task_files` (uploads); ids are kept unique once combined.
fn tasks_from(svc: &Service, f: &Fields) -> Result<Vec<Task>, ApiError> {
    let mut items: Vec<Json> = Vec::new();
    match f.get("tasks") {
        None => {}
        Some(Json::Arr(list)) => items.extend(list.iter().cloned()),
        Some(_) => {
            return Err(ApiError::bad_request(
                "'tasks' must be a list of strings or {prompt, criteria, answer, seeds} objects",
            ))
        }
    }
    let text = f.text_or("tasks_text", "")?;
    if !text.trim().is_empty() {
        items.extend(
            crate::llm::splitlines(&text)
                .into_iter()
                .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
                .map(|line| Json::str(line.trim())),
        );
    }
    let mut tasks: Vec<Task> = if items.is_empty() {
        Vec::new()
    } else {
        parse_tasks(&items).map_err(ApiError::bad_request)?
    };
    for name in f.names("task_files")? {
        let content: Vec<String> = crate::codegen::upload_entries(svc, &name)?
            .into_iter()
            .map(|(_, text)| text)
            .collect();
        tasks.extend(parse_task_file(&content.join("\n\n"), &extension(&name)).map_err(ApiError::bad_request)?);
    }
    if tasks.is_empty() {
        return Err(ApiError::bad_request(
            "no tasks: give 'tasks', 'tasks_text' or 'task_files'",
        ));
    }
    for index in 0..tasks.len() {
        if tasks[..index].iter().any(|t| t.id == tasks[index].id) {
            tasks[index].id = format!("t{}", index + 1);
        }
    }
    Ok(tasks)
}

/// The Ollama client of an agent request (`url` / `timeout` override the
/// server's defaults).
fn client_from(svc: &Service, f: &Fields, config: &AgentConfig) -> Result<OllamaClient, ApiError> {
    let timeout = f.number("timeout", Some(1.0))?.and_then(crate::llm::seconds);
    let url = f.text("url")?.filter(|u| !u.is_empty());
    svc.llm.ollama(url.as_deref(), Some(&config.agent_model), timeout)
}

/// Starts an `agent` or `explore` job on a worker thread.
fn start_job(
    svc: &Arc<Service>,
    kind: &str,
    config: AgentConfig,
    client: OllamaClient,
    toolbox: ToolBox,
    blame: bool,
    work: Result<Vec<Task>, (usize, Vec<String>)>,
) -> Result<(), ApiError> {
    if config.checkpoint_every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    if blame {
        svc.ensure_negative()?;
    }
    svc.ensure_idle()?;
    svc.start_job(kind);
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let mut emitted: Vec<Json> = Vec::new();
        let outcome = (|| -> Result<(), String> {
            let mut trainer = AgentTrainer::new(&client, &toolbox, config)?;
            let mut nets = Nets::Server {
                svc: &worker,
                negative: blame,
            };
            let manager = crate::checkpoint::manager(&worker);
            let mut progress = |record: &Json| {
                worker.agent.push(record.clone());
                worker.job_progress(record.clone());
                emitted.push(record.clone());
            };
            let mut hooks = Hooks {
                progress: &mut progress,
                stop: &|| worker.stopping(),
                checkpoints: manager,
            };
            match work {
                Ok(tasks) => trainer.run(&mut nets, &tasks, &mut hooks)?,
                Err((steps, seeds)) => {
                    trainer.frontier.extend(seeds);
                    trainer.explore(&mut nets, steps, &mut hooks)?
                }
            };
            Ok(())
        })();
        worker.finish_job(outcome.map(|()| emitted));
    });
    Ok(())
}

/// Starts an `agent` job over the tasks: criteria, tool calls, judging,
/// teaching and 2NRL; with `blame` the negative network learns from every
/// failure.  Answers 202 with the job, the tasks, the tools and the settings.
fn start_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let tasks = tasks_from(svc, &f)?;
    let config = config_from_fields(&f)?;
    let toolbox = crate::tools::toolbox_from(svc, r)?;
    let client = client_from(svc, &f, &config)?;
    let blame = f.flag("blame", false)?;
    let names = toolbox.names();
    let (tasks_doc, config_doc) = (Json::Arr(tasks.iter().map(Task::to_json).collect()), config.to_json());
    start_job(svc, "agent", config, client, toolbox, blame, Ok(tasks))?;
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("tasks", tasks_doc),
        ("tools", Json::strs(names)),
        ("config", config_doc),
        ("blame", Json::Bool(blame)),
    ])))
}

/// Starts an `explore` job - the network chooses every task itself: `{steps
/// (0 = until stopped), seed_urls, ...the agent settings}`.
fn explore_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = config_from_fields(&f)?;
    let toolbox = crate::tools::toolbox_from(svc, r)?;
    let client = client_from(svc, &f, &config)?;
    let steps = f.count_or("steps", 10, 0)?;
    let seeds: Vec<String> = f
        .texts_optional("seed_urls", "seed_url")?
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect();
    let blame = f.flag("blame", false)?;
    let names = toolbox.names();
    let config_doc = config.to_json();
    start_job(
        svc,
        "explore",
        config,
        client,
        toolbox,
        blame,
        Err((steps, seeds.clone())),
    )?;
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("steps", if steps > 0 { Json::Int(steps as i64) } else { Json::Null }),
        ("seeds", Json::strs(seeds)),
        ("tools", Json::strs(names)),
        ("config", config_doc),
        ("blame", Json::Bool(blame)),
    ])))
}

/// The criteria, step, attempt, task and round records of every run.
fn history_route(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(Json::obj([("history", Json::Arr(svc.agent.history()))]))
}

/// The acceptance criteria the LLM writes for the tasks, without attempting anything.
fn criteria_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let tasks = tasks_from(svc, &f)?;
    let config = config_from_fields(&f)?;
    let client = client_from(svc, &f, &config)?;
    let mut out = Vec::new();
    for task in &tasks {
        let criteria = write_criteria(&client, task, &config.agent_model, config.criteria_count)?;
        out.push(Json::obj([
            ("task", Json::str(task.id.clone())),
            ("prompt", Json::str(task.prompt.clone())),
            ("criteria", Json::strs(criteria)),
        ]));
    }
    Ok(Json::obj([
        ("tasks", Json::Arr(out)),
        ("model", Json::str(config.agent_model.clone())),
        ("url", Json::str(client.url.clone())),
    ]))
}

/// One task through the loop without training - the network's attempt, or
/// the teacher's demonstration - and how badly it failed.  The model is taken
/// for each of its predictions; the tools and the LLM run without it.
fn solve_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let tasks = tasks_from(svc, &f)?;
    if tasks.len() != 1 {
        return Err(ApiError::bad_request(format!(
            "/api/agent/solve takes exactly one task (got {})",
            tasks.len()
        )));
    }
    let task = &tasks[0];
    let source = f.text_or("source", "model")?.trim().to_lowercase();
    if source != "model" && source != "teacher" {
        return Err(ApiError::bad_request(format!(
            "'source' must be 'model' or 'teacher' (got {})",
            python_repr(&source)
        )));
    }
    let mut config = config_from_fields(&f)?;
    config.teach_on_failure = false;
    let toolbox = crate::tools::toolbox_from(svc, r)?;
    let client = client_from(svc, &f, &config)?;
    let mut trainer = AgentTrainer::new(&client, &toolbox, config).map_err(ApiError::bad_request)?;
    let criteria = trainer.criteria_for(task)?;
    let mut nets = Nets::Server { svc, negative: false };
    let mut hooks = Hooks {
        progress: &mut |_| {},
        stop: &|| false,
        checkpoints: None,
    };
    let mut attempt = if source == "model" {
        trainer.solve_with_model(&mut nets, task, 0, "model", &mut hooks)?
    } else {
        trainer.solve_with_teacher(task, &criteria, 0, None)?
    };
    attempt.verdict = trainer.judge(task, &criteria, &attempt)?;
    let gap = trainer.gap_of(&attempt, &criteria);
    Ok(Json::obj([
        ("task", task.to_json()),
        ("source", Json::str(source)),
        ("criteria", Json::strs(criteria)),
        ("attempt", attempt.to_json()),
        ("transcript", Json::str(attempt.text.clone())),
        ("correct", Json::Bool(attempt.verdict.correct)),
        ("gap", Json::Num(gap)),
        ("frontier", Json::strs(trainer.frontier.iter().take(20).cloned())),
        ("tools", Json::strs(toolbox.names())),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::negative::NegativeOptions;
    use crate::tools::{Param, Tool};

    /// A scripted Ollama: the criteria writer, the mediator (native tool calls
    /// or JSON), the judge (an answer that says "four" is correct) and the
    /// teacher.  It keeps every request it was sent.
    struct Fake {
        native: bool,
        requests: Mutex<Vec<(String, Json)>>,
    }

    impl Fake {
        fn new(native: bool) -> Fake {
            Fake {
                native,
                requests: Mutex::new(Vec::new()),
            }
        }

        fn chat_reply(&self, messages: &[Json], tools: &[Json]) -> Json {
            let system = messages[0].at("content").as_str().unwrap_or("");
            let call = Json::obj([(
                "function",
                Json::obj([
                    ("name", Json::str("echo")),
                    ("arguments", Json::obj([("text", Json::str("cats"))])),
                ]),
            )]);
            if system.contains("tool mediator") {
                if self.native && !tools.is_empty() {
                    return Json::obj([
                        ("role", Json::str("assistant")),
                        ("content", Json::str("")),
                        ("tool_calls", Json::Arr(vec![call])),
                    ]);
                }
                return Json::obj([
                    ("role", Json::str("assistant")),
                    (
                        "content",
                        Json::str("{\"tool\": \"echo\", \"arguments\": {\"text\": \"cats\"}}"),
                    ),
                ]);
            }
            if system.contains("solve a task with the tools") {
                let used = messages.iter().any(|m| m.at("role").as_str() == Some("tool"));
                if !used {
                    return Json::obj([
                        ("role", Json::str("assistant")),
                        ("content", Json::str("")),
                        ("tool_calls", Json::Arr(vec![call])),
                    ]);
                }
            }
            Json::obj([
                ("role", Json::str("assistant")),
                ("content", Json::str("A cat has four legs.")),
            ])
        }
    }

    impl LlmClient for Fake {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://fake"
        }
        fn model(&self) -> &str {
            "fake"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.requests
                .lock()
                .unwrap()
                .push(("generate".to_string(), Json::str(prompt)));
            if o.system.contains("judge of an attempt") {
                let answer = prompt.rsplit("Answer:").next().unwrap_or("").to_lowercase();
                let good = answer.contains("four");
                return Ok(format!(
                    "{{\"met\": [{good}, {good}], \"score\": {}, \"correct\": {good}, \"critique\": \"{}\"}}",
                    if good { 9 } else { 1 },
                    if good { "fine" } else { "it never says how many" }
                ));
            }
            if o.system.contains("acceptance criteria") {
                return Ok(
                    "{\"criteria\": [\"The answer says how many legs a cat has\", \"The answer names four\"]}"
                        .to_string(),
                );
            }
            if o.system.contains("choosing what to look into next") {
                return Ok(
                    "{\"question\": \"How many legs do cats have?\", \"seed\": \"http://example.org/cats\"}"
                        .to_string(),
                );
            }
            Ok("unexpected".to_string())
        }
        fn chat(&self, messages: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(self
                .chat_reply(messages, &[])
                .at("content")
                .as_str()
                .unwrap_or("")
                .to_string())
        }
    }

    impl AgentLlm for Fake {
        fn llm(&self) -> &dyn LlmClient {
            self
        }
        fn chat_message(&self, messages: &[Json], _o: &LlmOptions, tools: &[Json]) -> Result<Json, LlmError> {
            self.requests
                .lock()
                .unwrap()
                .push(("chat".to_string(), Json::Arr(messages.to_vec())));
            if !self.native {
                return Err(LlmError::new("ollama", "no tool calling"));
            }
            Ok(self.chat_reply(messages, tools))
        }
    }

    fn toolbox() -> ToolBox {
        let mut bx = ToolBox::new();
        let mut echo = Tool::new(
            "echo",
            "Say the text back.",
            vec![Param::new("text", "string", "what to say")],
            |args| {
                Ok((
                    format!("cats have four legs ({})", args.at("text").as_str().unwrap_or("")),
                    Json::obj([(
                        "results",
                        Json::Arr(vec![Json::obj([("url", Json::str("http://example.org/dogs"))])]),
                    )]),
                ))
            },
        );
        echo.network = true;
        bx.register(echo).unwrap();
        bx.register(Tool::new("boom", "Fails.", vec![], |_| Err("kaboom".to_string())))
            .unwrap();
        bx
    }

    fn fast() -> AgentConfig {
        AgentConfig {
            max_steps: 2,
            model_attempts: 1,
            criteria_count: 2,
            neg_epochs: 1,
            pos_epochs: 1,
            ..Default::default()
        }
    }

    fn task() -> Task {
        Task::new("t1", "How many legs does a cat have?")
    }

    fn quiet<'h>(progress: &'h mut dyn FnMut(&Json)) -> Hooks<'h> {
        Hooks {
            progress,
            stop: &|| false,
            checkpoints: None,
        }
    }

    #[test]
    fn tasks_are_read_the_way_python_reads_them() {
        let items = parse(
            "[\"first?\", {\"question\": \"second?\", \"criteria\": \"a\\n\\n b \", \"expected\": \"x\", \"urls\": \
             \"u\", \"id\": 3}]",
        )
        .unwrap();
        let tasks = parse_tasks(items.as_array()).unwrap();
        assert_eq!((tasks[0].id.as_str(), tasks[1].id.as_str()), ("t1", "3"));
        assert_eq!(tasks[1].criteria, ["a", "b"]);
        assert_eq!(tasks[1].answer.as_deref(), Some("x"));
        assert_eq!(tasks[1].seeds, ["u"]);
        assert_eq!(
            tasks[0].to_json().render(0),
            "{\"id\":\"t1\",\"prompt\":\"first?\",\"criteria\":[],\"answer\":null,\"seeds\":[]}"
        );
        for (bad, why) in [
            ("[\"\"]", "task 1: empty prompt"),
            ("[{\"text\": \"x\"}]", "task 1: missing 'prompt'"),
            (
                "[{\"prompt\": \"x\", \"criteria\": [1]}]",
                "task 1: 'criteria' must be a list of strings",
            ),
            (
                "[{\"prompt\": \"x\", \"seeds\": 3}]",
                "task 1: 'seeds' must be a list of URLs",
            ),
            (
                "[{\"prompt\": \"x\", \"answer\": 3}]",
                "task 1: 'answer' must be a string",
            ),
            ("[4]", "task 1: must be a string or an object, got int"),
            (
                "[{\"prompt\": \"a\", \"id\": \"x\"}, {\"prompt\": \"b\", \"id\": \"x\"}]",
                "duplicate task id 'x'",
            ),
        ] {
            assert_eq!(parse_tasks(parse(bad).unwrap().as_array()).unwrap_err(), why, "{bad}");
        }
        assert_eq!(parse_task_file("# c\none?\n\ntwo?\n", ".txt").unwrap().len(), 2);
        assert_eq!(
            parse_task_file("{\"problems\": [\"a\"]}", ".json").unwrap()[0].prompt,
            "a"
        );
        assert_eq!(parse_task_file("", ".txt").unwrap_err(), "no tasks in the file");
        assert_eq!(
            parse_task_file("{\"x\": 1}", ".json").unwrap_err(),
            "a JSON task file must hold a list or an object with a 'tasks' list"
        );
    }

    #[test]
    fn the_criteria_are_read_leniently() {
        let fake = Fake::new(true);
        assert_eq!(
            write_criteria(&fake, &task(), "m", 2).unwrap(),
            ["The answer says how many legs a cat has", "The answer names four"]
        );
        let mut known = task();
        known.criteria = vec!["mine".to_string()];
        assert_eq!(write_criteria(&fake, &known, "m", 2).unwrap(), ["mine"]);
        struct Lines;
        impl LlmClient for Lines {
            fn provider(&self) -> &'static str {
                "ollama"
            }
            fn url(&self) -> &str {
                ""
            }
            fn model(&self) -> &str {
                ""
            }
            fn models(&self) -> Result<Vec<Json>, LlmError> {
                Ok(Vec::new())
            }
            fn generate(&self, _p: &str, _o: &LlmOptions) -> Result<String, LlmError> {
                Ok("1. first thing\n2. second thing\n3. third".to_string())
            }
            fn chat(&self, _m: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
                Ok(String::new())
            }
        }
        assert_eq!(
            write_criteria(&Lines, &task(), "m", 2).unwrap(),
            ["first thing", "second thing"]
        );
    }

    #[test]
    fn the_mediator_reads_native_calls_and_json() {
        let bx = toolbox();
        for native in [true, false] {
            let fake = Fake::new(native);
            let call = mediate_call(&fake, "grbl", &bx, &task(), "", "m", None).unwrap();
            assert_eq!(call.name, "echo", "native={native}");
            assert_eq!(call.arguments.render(0), "{\"text\":\"cats\"}");
            assert!(call.ok());
        }
        let message = parse("{\"content\": \"{\\\"name\\\": \\\"nope\\\"}\"}").unwrap();
        assert!(call_from_message(&message, &bx).is_none());
        let message =
            parse("{\"content\": \"{\\\"tool\\\": \\\"echo\\\", \\\"args\\\": \\\"{\\\\\\\"text\\\\\\\": 1}\\\"}\"}")
                .unwrap();
        assert_eq!(
            call_from_message(&message, &bx).unwrap().arguments.render(0),
            "{\"text\":1}"
        );
    }

    #[test]
    fn the_judge_and_the_verdict() {
        let criteria = vec!["a".to_string(), "b".to_string()];
        let judged = Judged {
            met: vec![true, false],
            score: Some(7.0),
            correct: Some(true),
            critique: String::new(),
        };
        let strict = decide(&judged, &criteria, Some("x"), true);
        assert!(!strict.correct, "strict needs every criterion");
        assert_eq!(strict.issues, ["b"]);
        assert_eq!(strict.judged_by, "ollama");
        assert!(decide(&judged, &criteria, Some("x"), false).correct);
        let unanswered = decide(&judged, &criteria, None, false);
        assert_eq!(
            (
                unanswered.correct,
                unanswered.critique.as_str(),
                unanswered.judged_by.as_str()
            ),
            (false, "no answer was given", "answer")
        );
        let only_score = Judged {
            score: Some(6.0),
            ..Default::default()
        };
        assert!(decide(&only_score, &criteria, Some("x"), true).correct);
        assert_eq!(
            marks(&parse("[\"Yes.\", 0, {\"met\": \"passed\"}]").unwrap()),
            [true, false, true]
        );
        assert!(marks(&parse("[true, \"maybe\"]").unwrap()).is_empty(), "all or nothing");
    }

    #[test]
    fn an_untrained_network_is_mediated_judged_and_taught() {
        let fake = Fake::new(true);
        let bx = toolbox();
        let mut model = Model::new(0, Default::default()).unwrap();
        let mut negative = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let mut trainer = AgentTrainer::new(&fake, &bx, fast()).unwrap();
        let mut kinds: Vec<String> = Vec::new();
        let records = {
            let mut nets = Nets::Owned {
                model: &mut model,
                negative: Some(&mut negative),
            };
            let mut progress = |r: &Json| kinds.push(r.at("kind").as_str().unwrap_or("").to_string());
            let mut hooks = quiet(&mut progress);
            trainer.run(&mut nets, &[task()], &mut hooks).unwrap()
        };
        assert_eq!(
            kinds,
            ["criteria", "step", "step", "attempt", "attempt", "task", "round"]
        );
        let record = &records[0];
        assert_eq!(record.at("correct"), &Json::Bool(true));
        assert_eq!(record.at("solved_by").as_str(), Some("teacher"));
        assert_eq!(record.at("taught"), &Json::Bool(true));
        assert_eq!(
            (record.at("calls").as_i64(), record.at("own_calls").as_i64()),
            (Some(2), Some(0))
        );
        assert_eq!(record.at("autonomy"), &Json::Num(0.0));
        assert_eq!(record.at("action").as_str(), Some("2nrl"));
        assert_eq!(record.at("failures").as_i64(), Some(1));
        assert_eq!(record.at("mode").as_str(), Some("fail_invert"));
        assert!(
            record.at("negative_blamed").as_i64().unwrap() >= 1,
            "{}",
            record.render(0)
        );
        let reasons = record.at("negative_reasons").render(0);
        assert!(reasons.contains("\"bad-call\""), "{reasons}");
        assert_eq!(trainer.visited, Vec::<String>::new());
        assert_eq!(trainer.frontier, ["http://example.org/dogs"]);
        let solution = &trainer.solved[0].1;
        assert!(solution.starts_with("TASK: How many legs does a cat have?\n<tool>echo {\"text\": \"cats\"}</tool>\n"));
        assert!(solution.ends_with("<answer>A cat has four legs.</answer>\n"));
        let keys: Vec<&str> = match record {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(&keys[..6], ["kind", "phase", "round", "task", "prompt", "criteria"]);
        assert!(keys.ends_with(&["flipped", "boost_mean", "boost_max"]));
    }

    #[test]
    fn mediation_never_executes_the_broken_call_and_always_skips_the_network() {
        let fake = Fake::new(true);
        let bx = toolbox();
        let mut model = Model::new(0, Default::default()).unwrap();
        let config = AgentConfig {
            mediation: "never".to_string(),
            teach_on_failure: false,
            ..fast()
        };
        let mut trainer = AgentTrainer::new(&fake, &bx, config).unwrap();
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: None,
        };
        let mut progress = |_: &Json| {};
        let mut hooks = quiet(&mut progress);
        let (record, failures, good) = trainer.run_task(&mut nets, &task(), "model", 1, &mut hooks).unwrap();
        let record = Json::Obj(record);
        assert_eq!(record.at("correct"), &Json::Bool(false));
        assert_eq!(
            record.at("calls").as_i64(),
            Some(0),
            "nothing written, nothing mediated"
        );
        assert_eq!((failures.len(), good.len()), (1, 0));
        assert_eq!(failures[0].gap, 1.0, "no answer is the worst failure");
        let asked = fake.requests.lock().unwrap();
        assert!(
            asked.iter().all(|(kind, _)| kind == "generate"),
            "the mediator was never asked"
        );
    }

    #[test]
    fn learning_weighs_failures_and_inverts_when_nothing_went_right() {
        let fake = Fake::new(true);
        let bx = toolbox();
        let mut model = Model::new(0, Default::default()).unwrap();
        let mut trainer = AgentTrainer::new(&fake, &bx, fast()).unwrap();
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: None,
        };
        let failures = vec![
            Failure {
                text: "TASK: q\n<answer>no</answer>\n".to_string(),
                gap: 0.4,
                task: "t1".to_string(),
                source: "model".to_string(),
            },
            Failure {
                text: "TASK: q\n<answer>no</answer>\n".to_string(),
                gap: 1.0,
                task: "t1".to_string(),
                source: "model".to_string(),
            },
        ];
        let learned = Json::Obj(trainer.learn(&mut nets, &failures, &[], &|| false).unwrap());
        assert_eq!(learned.at("action").as_str(), Some("punish"));
        assert_eq!(
            (learned.at("bad").as_i64(), learned.at("blatant").as_i64()),
            (Some(1), Some(1))
        );
        assert_eq!(learned.at("boost_max"), &Json::Num(3.0), "min(4, 1 + 1.0 / 0.5)");
        let inverted = nets.model(|m| m.g.inverted);
        assert!(inverted, "failures only: trained on, then inverted");
        let rewarded = Json::Obj(
            trainer
                .learn(
                    &mut nets,
                    &[],
                    &["TASK: q\n<answer>yes</answer>\n".to_string()],
                    &|| false,
                )
                .unwrap(),
        );
        assert_eq!(rewarded.at("action").as_str(), Some("reward"));
        let local = AgentConfig {
            blatant_mode: "activation".to_string(),
            ..fast()
        };
        let mut trainer = AgentTrainer::new(&fake, &bx, local).unwrap();
        let learned = Json::Obj(
            trainer
                .learn(&mut nets, &failures, &["good text here".to_string()], &|| false)
                .unwrap(),
        );
        assert!(learned.at("flipped").as_i64().unwrap() > 0);
        assert_eq!(
            learned.at("bad").as_i64(),
            Some(1),
            "counted before the blatant one left the garbage set"
        );
        assert_eq!(learned.at("action").as_str(), Some("reward"));
    }

    #[test]
    fn the_gap_follows_the_score_and_the_criteria() {
        let fake = Fake::new(true);
        let bx = toolbox();
        let trainer = AgentTrainer::new(&fake, &bx, fast()).unwrap();
        let criteria = vec!["a".to_string(), "b".to_string()];
        let mut attempt = Attempt {
            index: 0,
            source: "model".to_string(),
            steps: Vec::new(),
            answer: Some("x".to_string()),
            text: String::new(),
            verdict: Verdict {
                score: Some(4.5),
                met: vec![true, true],
                ..Verdict::unjudged()
            },
            seconds: 0.0,
            gap: None,
        };
        assert_eq!(trainer.gap_of(&attempt, &criteria), 0.25);
        attempt.verdict.score = Some(9.0);
        assert_eq!(trainer.gap_of(&attempt, &criteria), 0.1, "the floor");
        attempt.answer = None;
        assert_eq!(trainer.gap_of(&attempt, &criteria), 1.0);
        attempt.verdict.correct = true;
        assert_eq!(trainer.gap_of(&attempt, &criteria), 0.0);
    }

    #[test]
    fn failures_are_blamed_at_the_granularity_they_happened_at() {
        let tool_result = |ok: bool| ToolResult {
            tool: "echo".to_string(),
            arguments: Json::Obj(Vec::new()),
            ok,
            output: String::new(),
            error: (!ok).then(|| "refused".to_string()),
            seconds: 0.0,
            meta: Json::Obj(Vec::new()),
        };
        let step = |source: &str, ok: bool, emission: &str| Step {
            index: 0,
            call: bare_call("echo", Json::obj([("text", Json::str("x"))])),
            result: tool_result(ok),
            source: source.to_string(),
            emission: emission.to_string(),
        };
        let failed = Attempt {
            index: 0,
            source: "model".to_string(),
            steps: vec![step("mediator", true, "grbl  ntoo"), step("model", false, "")],
            answer: Some("purple".to_string()),
            text: "TASK: q\nwrong".to_string(),
            verdict: Verdict {
                critique: "it never says how many".to_string(),
                issues: vec!["names four".to_string()],
                score: Some(1.0),
                ..Verdict::unjudged()
            },
            seconds: 0.0,
            gap: Some(1.0),
        };
        assert_eq!(agent_reason(&failed), "bad-call");
        let correct = Attempt {
            verdict: Verdict {
                correct: true,
                ..Verdict::unjudged()
            },
            text: "TASK: q\nright".to_string(),
            steps: Vec::new(),
            ..failed.clone()
        };
        let (faults, cleared) = faults_from_agent(
            &[failed.clone(), correct.clone()],
            Some("TASK: q\nright"),
            6.0,
            "agent",
            true,
        );
        assert_eq!(faults.len(), 3);
        assert_eq!((faults[0].reason.as_str(), faults[0].severity), ("bad-call", 2.0));
        assert_eq!(faults[0].correction, "TASK: q\nright");
        assert_eq!(faults[0].note, "it never says how many Unmet: names four");
        assert_eq!(
            (faults[1].text.as_str(), faults[1].reason.as_str()),
            ("grbl ntoo", "bad-call")
        );
        assert_eq!(faults[2].text, "<tool>echo {\"text\": \"x\"}</tool>");
        assert_eq!(
            (faults[2].reason.as_str(), faults[2].note.as_str()),
            ("tool-error", "refused")
        );
        assert!(cleared.is_empty(), "the correction is not cleared twice");
        let (_, cleared) = faults_from_agent(&[correct], None, 6.0, "agent", false);
        assert_eq!(cleared, ["TASK: q\nright"]);
        let mut negative = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let report = teach_agent(&mut negative, &[failed], None, 6.0, "explore", &TeachOptions::default()).unwrap();
        assert_eq!(report.blamed, 3);
        assert!(negative.neg.as_ref().unwrap().log.iter().all(|e| e.source == "explore"));
    }

    #[test]
    fn exploring_chooses_its_own_tasks() {
        let fake = Fake::new(true);
        let bx = toolbox();
        let mut model = Model::new(0, Default::default()).unwrap();
        let mut trainer = AgentTrainer::new(&fake, &bx, fast()).unwrap();
        trainer.frontier.push("http://example.org/cats".to_string());
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: None,
        };
        let mut progress = |_: &Json| {};
        let mut hooks = quiet(&mut progress);
        let records = trainer.explore(&mut nets, 1, &mut hooks).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].at("kind").as_str(), Some("explore"));
        assert_eq!(records[0].at("step").as_i64(), Some(1));
        assert_eq!(records[0].at("task").as_str(), Some("x1"));
        assert_eq!(records[0].at("prompt").as_str(), Some("How many legs do cats have?"));
        assert!(
            !trainer.frontier.contains(&"http://example.org/cats".to_string()),
            "the seed left the frontier"
        );
        assert_eq!(trainer.history[0].at("kind").as_str(), Some("proposal"));
    }

    #[test]
    fn the_settings_are_checked_as_python_checks_them() {
        assert!(AgentConfig::default().validate().is_ok());
        for (config, why) in [
            (
                AgentConfig {
                    mediation: "sometimes".to_string(),
                    ..Default::default()
                },
                "mediation must be one of repair, always, never (got 'sometimes')",
            ),
            (
                AgentConfig {
                    criteria_count: 9,
                    ..Default::default()
                },
                "criteria_count must be between 1 and 8",
            ),
            (
                AgentConfig {
                    blatant_boost: 0.5,
                    ..Default::default()
                },
                "blatant_boost must be >= 1",
            ),
            (
                AgentConfig {
                    phases: vec!["both".to_string()],
                    ..Default::default()
                },
                "phases must be a non-empty subset of teacher, model",
            ),
        ] {
            assert_eq!(config.validate().unwrap_err(), why);
        }
        let keys: Vec<String> = match AgentConfig::default().to_json() {
            Json::Obj(pairs) => pairs.into_iter().map(|(k, _)| k).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys.len(), 34);
        assert!(AgentTrainer::new(&Fake::new(true), &ToolBox::new(), AgentConfig::default()).is_err());
    }

    #[test]
    fn the_routes_refuse_what_python_refuses() {
        let model = Model::new(0, Default::default()).unwrap();
        let svc = Arc::new(Service::new(model, String::new(), 0, 1));
        let call = |route: fn(&Arc<Service>, &Request) -> Answer, body: &str| {
            route(&svc, &Request::json("POST", "/api/agent", parse(body).unwrap()))
        };
        assert_eq!(
            call(start_route, "{}").unwrap_err().message,
            "no tasks: give 'tasks', 'tasks_text' or 'task_files'"
        );
        assert_eq!(
            call(start_route, "{\"tasks\": \"x\"}").unwrap_err().message,
            "'tasks' must be a list of strings or {prompt, criteria, answer, seeds} objects"
        );
        assert_eq!(
            call(start_route, "{\"tasks\": [\"x\"], \"phase\": \"all\"}")
                .unwrap_err()
                .message,
            "'phase' must be 'model', 'teacher' or 'both' (got 'all')"
        );
        assert_eq!(
            call(start_route, "{\"tasks\": [\"x\"], \"blatant_mode\": \"x\"}")
                .unwrap_err()
                .message,
            "'blatant_mode' must be one of none, fail_invert, activation, state (got 'x')"
        );
        assert_eq!(
            call(solve_route, "{\"tasks\": [\"a\", \"b\"]}").unwrap_err().message,
            "/api/agent/solve takes exactly one task (got 2)"
        );
        assert_eq!(
            call(solve_route, "{\"tasks\": [\"a\"], \"source\": \"x\"}")
                .unwrap_err()
                .message,
            "'source' must be 'model' or 'teacher' (got 'x')"
        );
        assert_eq!(
            call(explore_route, "{\"steps\": -1}").unwrap_err().message,
            "'steps' must be >= 0 (got -1)"
        );
        let tasks_of = |body: &str| tasks_from(&svc, &Fields::new(&parse(body).unwrap()));
        // the second task is numbered by its place, and that number is taken
        assert_eq!(
            tasks_of("{\"tasks\": [{\"prompt\": \"a\", \"id\": \"t2\"}, \"b\"]}")
                .unwrap_err()
                .message,
            "duplicate task id 't2'"
        );
        let tasks = tasks_of("{\"tasks\": [{\"prompt\": \"a\", \"id\": \"x\"}, \"b\"]}").unwrap();
        assert_eq!(tasks.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), ["x", "t2"]);
        let history = history_route(&svc, &Request::json("GET", "/api/agent/history", Json::Null)).unwrap();
        assert_eq!(history.render(0), "{\"history\":[]}");
    }
}
