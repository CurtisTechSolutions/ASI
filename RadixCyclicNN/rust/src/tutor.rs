//! Automated English lessons: the teacher writes the exercise, the network
//! completes it, the teacher marks it (`radixnet/tutor.py`,
//! `go/radixnet/tutor.go`).
//!
//! ```text
//! topic -> prefix (LLM) -> completion (the prediction search) -> grade (LLM) -> 2NRL
//! ```
//!
//! The Predict tab with nobody at the keyboard.  One *lesson* is the whole
//! prediction process run end to end without a human:
//!
//! 1. **the exercise** - the teacher writes sentence openings about a topic
//!    ([`write_exercises`]), each drilling one point of grammar and each with
//!    its own model answer, so a lesson can teach even when the network says
//!    nothing;
//! 2. **the completion** - the network continues the prefix with the ordinary
//!    prediction search, exactly what `predict` does for a typed prefix;
//! 3. **the grade** - the teacher marks the finished sentence
//!    ([`grade_completions`]): grammar, spelling and fluency out of 10, the
//!    worst mistake named from [`ERROR_TYPES`], one sentence of teaching and
//!    the *correction* - the same sentence written out in correct English;
//! 4. **the lesson learned** - a correction is taught as a correction: only
//!    what the teacher changed moves (`Model::correct`), and whatever has no
//!    correction to diff is 2NRL weighted by the mark (D-050) - a failure is
//!    pushed away by how bad it was and a pass kept by how good it was, so a
//!    rating decides how much the network learns rather than a single like;
//! 5. **why, and the same mistake again** - with a negative network attached,
//!    every failure is handed back to the teacher ([`explain_mistakes`]) for
//!    the rule it broke and more sentences that break it the same way, and all
//!    of them are blamed under the mistake the teacher named
//!    ([`teach_lessons`]);
//! 6. **the next lesson plan** - the report card goes back to the teacher,
//!    which plans the lessons that follow ([`crate::plan`]); the difficulty
//!    they step up by is read off the marks (D-051), and an auto run teaches
//!    batch after batch to the plan the last one led to (D-052).
//!
//! Grammar is what is being taught, so grammar is most of the mark:
//! `score = grammar_weight * grammar + (1 - grammar_weight) *
//! mean(spelling, fluency)` ([`overall_score`]).
//!
//! # Byte for byte
//!
//! Every prompt here is Python's, character for character, and every answer
//! is read with Python's tolerance of the shapes an LLM drifts into: the
//! parity suite points both ports at one fake teacher and compares what it was
//! asked.  So a mark is parsed the way `float()` parses it, a mistake is
//! mapped the way `_error_type` maps it, and a string is quoted into a prompt
//! the way `json.dumps(..., ensure_ascii=False)` quotes it.
//!
//! The loop itself ([`TutorTrainer`], [`TutorConfig`]) lives in
//! [`trainer`], the command and the routes in [`serve`].

pub mod serve;
pub mod trainer;

pub use serve::{cli, routes, State};
pub use trainer::{Networks, TutorConfig, TutorTrainer};

use crate::blame::{classify, severity_from_rating, teach, Fault, TeachOptions, TeachReport};
use crate::diff::Edit;
use crate::encoding::Encoding;
use crate::json::Json;
use crate::llm::fields::py_str_of;
use crate::llm::{loads_lenient, parse_lines, LlmClient, LlmError, LlmOptions, DEFAULT_PROVIDER};
use crate::model::Model;
use crate::negative::python_repr;
use crate::review::ReviewError;

/// What this module's own lines are filed under.
const LOG: &str = "tutor";

/// The mistakes a completion is marked with; `none` is a clean sentence.
pub const ERROR_TYPES: &[&str] = &[
    "none",
    "agreement",
    "tense",
    "article",
    "preposition",
    "plural",
    "pronoun",
    "word-order",
    "spelling",
    "punctuation",
    "vocabulary",
    "fragment",
    "nonsense",
];

/// How the network completes a prefix: the cheapest path, the beam search, or
/// a stochastic walk (the count model beams for `dijkstra` too).
pub const MODES: &[&str] = &["dijkstra", "beam", "sample"];

/// Apply 2NRL once per round over every graded sentence, or after each lesson.
pub const TWONRL_PER: &[&str] = &["round", "lesson"];

/// The positive-phase weight of text the teacher wrote (a correction, a model
/// answer, a drill): correct by construction, so always the full rate.
pub const TEACHER_WEIGHT: f64 = 1.0;

/// Sentences with the same mistake the teacher writes per failure, for the
/// negative network.
pub const DEFAULT_VARIANTS: usize = 3;

/// The most that can be asked for at once (a longer list starts repeating itself).
pub const MAX_VARIANTS: usize = 10;

/// The longest comment kept from a grade.
pub const MAX_COMMENT_CHARS: usize = 300;

/// The longest explanation kept from the teacher.
pub const MAX_WHY_CHARS: usize = 400;

/// The model that sets and marks the exercises when none is named:
/// `$RADIXNET_TUTOR_MODEL` for Ollama, else the provider's own default.
pub fn default_tutor_model(provider: &str) -> String {
    match crate::llm::normalise_provider(provider) {
        Ok(crate::llm::CHATGPT) => crate::chatgpt::default_model(),
        _ => {
            let tutor = crate::llm::env("RADIXNET_TUTOR_MODEL");
            if tutor.is_empty() {
                crate::ollama::default_model()
            } else {
                tutor
            }
        }
    }
}

// -- the small things every step shares ---------------------------------------------------------

/// What is actually fed to the search: the prefix with exactly one trailing
/// space, so a new word follows (`""` stays `""`).
pub fn cue(prefix: &str) -> String {
    let text = prefix.trim_end();
    if text.is_empty() {
        String::new()
    } else {
        format!("{text} ")
    }
}

/// A text on one line, at most `limit` characters, ending in `…` when it was cut.
pub fn clip(text: &str, limit: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= limit {
        return flat;
    }
    let head: String = flat.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", head.trim_end())
}

/// Words joined by single spaces (Python's `" ".join(text.split())`).
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python's `float()` of a string; `None` where it raises.
pub fn parse_float(text: &str) -> Option<f64> {
    text.trim().replace('_', "").parse().ok()
}

/// Python's `float(value)` of what an answer holds; `None` where it raises.
fn float_of(value: &Json) -> Option<f64> {
    match value {
        Json::Int(n) => Some(*n as f64),
        Json::Num(x) => Some(*x),
        Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Json::Str(s) => parse_float(s),
        _ => None,
    }
}

/// Python's `int(value)`; `None` where it raises.
fn int_of(value: &Json) -> Option<i64> {
    match value {
        Json::Int(n) => Some(*n),
        Json::Num(x) if x.is_finite() => Some(x.trunc() as i64),
        Json::Bool(b) => Some(*b as i64),
        Json::Str(s) => s.trim().replace('_', "").parse().ok(),
        _ => None,
    }
}

/// `item.get(a, item.get(b, ...))`: the value of the first key that is there
/// at all (even as `null`).
fn first_present<'a>(item: &'a Json, keys: &[&str]) -> Option<&'a Json> {
    keys.iter().find_map(|k| item.get(k))
}

/// The first of `keys` holding a non-blank string, on one line.
fn string_field(item: &Json, keys: &[&str]) -> String {
    keys.iter()
        .filter_map(|k| item.get(k).and_then(Json::as_str))
        .find(|v| !v.trim().is_empty())
        .map(squash)
        .unwrap_or_default()
}

/// `item.get(a) or item.get(b) or ...` as `str()`, on one line.
fn truthy_text(item: &Json, keys: &[&str]) -> String {
    keys.iter()
        .filter_map(|k| item.get(k))
        .find(|v| crate::llm::fields::truthy(v))
        .map(|v| squash(&py_str_of(v)))
        .unwrap_or_default()
}

/// A 0-10 mark from an answer (numbers, `"7/10"`, `"7"`); `None` when unusable.
fn score_of(value: Option<&Json>) -> Option<f64> {
    let number = match value? {
        Json::Str(s) => parse_float(s.split('/').next().unwrap_or("").trim())?,
        other => float_of(other)?,
    };
    if number.is_nan() {
        return None;
    }
    // Python's max(0.0, min(10.0, number))
    let low = if number < 10.0 { number } else { 10.0 };
    Some(if low > 0.0 { low } else { 0.0 })
}

/// The reported mistake mapped onto [`ERROR_TYPES`] (`other` for anything
/// unrecognised): `"subject-verb agreement"` is `agreement`, `"Verb Tense"`
/// is `tense`, `""` and `"no error"` are `none`.
pub fn error_type(value: &str) -> String {
    let text = value.trim().to_lowercase().replace(['_', ' '], "-");
    if matches!(text.as_str(), "" | "no-error" | "no-errors" | "correct" | "ok" | "n/a") {
        return "none".to_string();
    }
    if ERROR_TYPES.contains(&text.as_str()) {
        return text;
    }
    for known in &ERROR_TYPES[1..] {
        if text.contains(known) {
            return known.to_string();
        }
    }
    "other".to_string()
}

/// [`error_type`] of a JSON value: `str(value or "")`.
fn error_of(value: Option<&Json>) -> String {
    match value {
        Some(v) if crate::llm::fields::truthy(v) => error_type(&py_str_of(v)),
        _ => "none".to_string(),
    }
}

/// The mean of some marks, summed exactly (`statistics.fmean`).
pub(crate) fn fmean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| crate::fsum::fsum(values) / values.len() as f64)
}

/// One mark out of 10 from the three sub-marks, grammar carrying
/// `grammar_weight` of it.  Missing sub-marks drop out: with only grammar the
/// score *is* the grammar mark, with only spelling and fluency their mean.
pub fn overall_score(
    grammar: Option<f64>,
    spelling: Option<f64>,
    fluency: Option<f64>,
    grammar_weight: f64,
) -> Option<f64> {
    let rest: Vec<f64> = [spelling, fluency].into_iter().flatten().collect();
    let mean_rest = fmean(&rest);
    match (grammar, mean_rest) {
        (None, rest) => rest,
        (Some(g), None) => Some(g),
        (Some(g), Some(rest)) => Some(grammar_weight * g + (1.0 - grammar_weight) * rest),
    }
}

/// A string as `json.dumps(text, ensure_ascii=False)` writes it: quoted, on
/// one line, whatever it holds.
pub fn json_quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A failure the teacher did not cause: the request was fine, the answer was not.
fn llm_failure(client: &dyn LlmClient, message: String) -> ReviewError {
    ReviewError::Llm(LlmError::new(client.provider(), message))
}

// -- the exercise ---------------------------------------------------------------------------------

/// One sentence opening for the network to finish, with the point it drills
/// and a model answer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exercise {
    pub id: String,
    pub prefix: String,
    /// The point of grammar: "past tense", "plural nouns", ...
    pub focus: String,
    /// The teacher's own complete sentence, taught when the network fails.
    pub answer: String,
}

impl Exercise {
    /// [`cue`] of the prefix: what the prediction search is given.
    pub fn cue(&self) -> String {
        cue(&self.prefix)
    }

    /// `{"id", "prefix", "focus", "answer"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(self.id.clone())),
            ("prefix", Json::str(self.prefix.clone())),
            ("focus", Json::str(self.focus.clone())),
            ("answer", Json::str(self.answer.clone())),
        ])
    }
}

const EXERCISE_SYSTEM: &str = "You are an English teacher writing exercises for a beginner student who completes \
     sentences: you give the opening words, the student writes the rest. Every exercise is one unfinished sentence \
     in plain, simple, modern English. The prefix must be {words} words, must NOT end with punctuation and must be \
     genuinely unfinished, so that finishing it correctly needs the point of grammar you are drilling. Each exercise \
     also carries 'focus' (the point of grammar in two or three words, e.g. 'subject-verb agreement', 'past tense', \
     'plural nouns', 'articles', 'prepositions of place') and 'answer' (the whole sentence, prefix included, \
     finished correctly by you - short, natural and factual). Vary the focus and the vocabulary across the \
     exercises. Reply with JSON only, no prose, exactly of the form {\"exercises\": [{\"prefix\": \"...\", \"focus\": \
     \"...\", \"answer\": \"...\"}, ...]} with exactly {n} entries.";

/// What trails an opening that is not one: punctuation and dashes.
const OPENING_TAIL: &str = " .,:;!?-–—_";

/// An LLM answer as at most `count` exercises of round `round`; unusable and
/// duplicate entries are dropped.
///
/// Tolerates the shapes an LLM drifts into: `{"exercises": [...]}`, a bare
/// list, plain strings instead of objects, `opening` / `stem` / `sentence`
/// instead of `prefix`, and a model answer that omits the prefix.
pub fn parse_exercises(raw: &str, count: usize, round: usize) -> Vec<Exercise> {
    let round = round.max(1);
    let data = loads_lenient(raw);
    let mut items: Option<Vec<Json>> = None;
    match &data {
        Some(doc @ Json::Obj(_)) => {
            for key in ["exercises", "prefixes", "items", "results"] {
                if let Some(Json::Arr(list)) = doc.get(key) {
                    items = Some(list.clone());
                    break;
                }
            }
            if items.is_none() && ["prefix", "opening", "stem"].iter().any(|k| doc.get(k).is_some()) {
                items = Some(vec![doc.clone()]);
            }
        }
        Some(Json::Arr(list)) => items = Some(list.clone()),
        _ => {}
    }
    // plain lines: still usable
    let items = items.unwrap_or_else(|| {
        parse_lines(raw, Some(count))
            .into_iter()
            .map(|line| Json::obj([("prefix", Json::str(line))]))
            .collect()
    });
    let mut exercises: Vec<Exercise> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for item in items {
        let item = match item {
            Json::Str(text) => Json::obj([("prefix", Json::Str(text))]),
            obj @ Json::Obj(_) => obj,
            _ => continue,
        };
        // an opening, never a finished sentence
        let prefix = string_field(&item, &["prefix", "opening", "stem", "start", "sentence"])
            .trim_end_matches(|c| OPENING_TAIL.contains(c))
            .to_string();
        if prefix.is_empty() {
            continue;
        }
        let key = prefix.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let mut answer = string_field(&item, &["answer", "solution", "completion", "full_sentence", "example"]);
        if !answer.is_empty() && !answer.to_lowercase().starts_with(&prefix.to_lowercase()) {
            // the teacher answered with the continuation only
            answer = cue(&prefix) + answer.trim_start();
        }
        let focus = first_present(&item, &["focus", "point", "skill"])
            .and_then(Json::as_str)
            .map(|f| clip(f, 60))
            .unwrap_or_default();
        exercises.push(Exercise {
            id: format!("r{round}e{}", exercises.len() + 1),
            prefix,
            focus,
            answer,
        });
        if exercises.len() >= count {
            break;
        }
    }
    exercises
}

/// What the teacher is asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct ExerciseRequest {
    pub topic: String,
    pub count: usize,
    /// Pin every exercise to one point of grammar (`""` = none).
    pub focus: String,
    /// `beginner`, `intermediate`, ...
    pub level: String,
    /// The mistakes the student keeps making.
    pub weak: Vec<String>,
    /// How long a prefix is, e.g. `3 to 6`.
    pub words: String,
    /// The plan this batch of lessons is being taught to ([`crate::plan::LessonPlan::prompt`]).
    pub brief: String,
    /// The teacher's model (`""` = the client's own).
    pub model: String,
    /// 0 = the default 0.9.
    pub temperature: f64,
}

impl Default for ExerciseRequest {
    fn default() -> ExerciseRequest {
        ExerciseRequest {
            topic: String::new(),
            count: 5,
            focus: String::new(),
            level: "beginner".to_string(),
            weak: Vec::new(),
            words: "3 to 6".to_string(),
            brief: String::new(),
            model: String::new(),
            temperature: 0.0,
        }
    }
}

/// The weak points worth drilling: named, not `none`, at most five.
fn useful_weak(weak: &[String]) -> Vec<String> {
    weak.iter()
        .filter(|w| !w.is_empty() && w.as_str() != "none")
        .take(5)
        .cloned()
        .collect()
}

/// Asks the teacher for `count` sentence openings about the topic.
///
/// `focus` pins the point of grammar for every exercise, `weak` lists the
/// mistakes the student has been making (the previous round's report card),
/// which the teacher is asked to drill, and `brief` is the plan this batch of
/// lessons is being taught to.
pub fn write_exercises(client: &dyn LlmClient, req: &ExerciseRequest) -> Result<Vec<Exercise>, ReviewError> {
    if req.count < 1 {
        return Err(ReviewError::Invalid("count must be >= 1".to_string()));
    }
    let topic = req.topic.trim();
    if topic.is_empty() {
        return Err(ReviewError::Invalid("topic must be a non-empty string".to_string()));
    }
    let system = EXERCISE_SYSTEM
        // the count first: the words are the caller's, and are inserted as given
        .replace("{n}", &req.count.to_string())
        .replace("{words}", &req.words);
    let level = if req.level.is_empty() {
        "beginner"
    } else {
        req.level.as_str()
    };
    let mut lines = vec![format!("Topic: {topic}"), format!("Level: {}", level.trim())];
    if !req.brief.trim().is_empty() {
        lines.push(format!("The plan for this batch of lessons: {}", squash(&req.brief)));
    }
    if !req.focus.trim().is_empty() {
        lines.push(format!("Every exercise must drill: {}", req.focus.trim()));
    }
    let weak = useful_weak(&req.weak);
    if !weak.is_empty() {
        lines.push(format!(
            "The student keeps making these mistakes, so drill them: {}.",
            weak.join(", ")
        ));
    }
    lines.push(format!("Write the {} exercises now.", req.count));
    let temperature = if req.temperature > 0.0 { req.temperature } else { 0.9 };
    let o = LlmOptions::default()
        .system(system)
        .model(req.model.as_str())
        .json()
        .temperature(temperature);
    let raw = client.generate(&lines.join("\n"), &o)?;
    let exercises = parse_exercises(&raw, req.count, 1);
    if exercises.is_empty() {
        let name = if req.model.is_empty() {
            client.model()
        } else {
            req.model.as_str()
        };
        return Err(llm_failure(
            client,
            format!("the teacher model {} returned no usable exercises", python_repr(name)),
        ));
    }
    crate::log_debug!(LOG, "{} exercise(s) about {topic:?}", exercises.len());
    Ok(exercises)
}

const DRILL_SYSTEM: &str = "You are an English teacher writing model sentences for a beginner student to imitate. \
     Answer with exactly {n} lines and nothing else: one short, correct, natural sentence per line, plain text, no \
     numbering, no quotes, no commentary. Every sentence must be simple, factual and grammatically perfect, because \
     the student learns English by copying them.";

/// Correct example sentences about the topic for the fine-tune pass,
/// demonstrating the weak points (none when `count` is 0).
pub fn drill_sentences(
    client: &dyn LlmClient,
    topic: &str,
    count: usize,
    weak: &[String],
    model: &str,
) -> Result<Vec<String>, LlmError> {
    if count < 1 {
        return Ok(Vec::new());
    }
    let mut user = format!("Topic: {}\n", topic.trim());
    let weak = useful_weak(weak);
    if !weak.is_empty() {
        user.push_str(&format!(
            "Each sentence must clearly demonstrate the correct use of: {}.\n",
            weak.join(", ")
        ));
    }
    user.push_str(&format!("Write the {count} sentences now."));
    let o = LlmOptions::default()
        .system(DRILL_SYSTEM.replace("{n}", &count.to_string()))
        .model(model)
        .temperature(0.8);
    Ok(parse_lines(&client.generate(&user, &o)?, Some(count)))
}

// -- the grade ------------------------------------------------------------------------------------

/// The teacher's marking of one completed sentence.
#[derive(Clone, Debug, PartialEq)]
pub struct Grade {
    /// 0-10 overall (grammar-weighted); `None` when the answer could not be read.
    pub score: Option<f64>,
    pub grammar: Option<f64>,
    pub spelling: Option<f64>,
    pub fluency: Option<f64>,
    pub passed: bool,
    /// One of [`ERROR_TYPES`], or `other`.
    pub error: String,
    /// The whole sentence in correct English: what the network is taught.
    pub correction: String,
    /// One sentence of teaching.
    pub comment: String,
    /// The marking provider (`ollama` | `chatgpt`), `empty` (nothing to mark)
    /// or `unrated`.
    pub graded_by: String,
}

impl Default for Grade {
    fn default() -> Grade {
        Grade {
            score: None,
            grammar: None,
            spelling: None,
            fluency: None,
            passed: false,
            error: "none".to_string(),
            correction: String::new(),
            comment: String::new(),
            graded_by: DEFAULT_PROVIDER.to_string(),
        }
    }
}

fn mark(value: Option<f64>) -> Json {
    value.map(Json::Num).unwrap_or(Json::Null)
}

impl Grade {
    /// `{"score", "grammar", "spelling", "fluency", "passed", "error",
    /// "correction", "comment", "graded_by"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("score", mark(self.score)),
            ("grammar", mark(self.grammar)),
            ("spelling", mark(self.spelling)),
            ("fluency", mark(self.fluency)),
            ("passed", Json::Bool(self.passed)),
            ("error", Json::str(self.error.clone())),
            ("correction", Json::str(self.correction.clone())),
            ("comment", Json::str(self.comment.clone())),
            ("graded_by", Json::str(self.graded_by.clone())),
        ])
    }
}

/// A sentence the network wrote (or one wrong the same way), the sentence the
/// teacher wrote instead, and how much it weighs - Python's `Correction`.
#[derive(Clone, Debug, PartialEq)]
pub struct TutorCorrection {
    pub wrong: String,
    pub right: String,
    pub weight: f64,
}

impl TutorCorrection {
    /// `{"wrong", "right", "weight"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("wrong", Json::str(self.wrong.clone())),
            ("right", Json::str(self.right.clone())),
            ("weight", Json::Num(self.weight)),
        ])
    }
}

/// One exercise, one completion by the network, one grade.
#[derive(Clone, Debug, PartialEq)]
pub struct Lesson {
    pub exercise: Exercise,
    /// 0-based: attempt 0 uses the configured mode, later ones are samples.
    pub attempt: usize,
    pub mode: String,
    pub continuation: String,
    /// `cue(prefix) + continuation`: what is graded and, when it passes, learned.
    pub sentence: String,
    pub cost: f64,
    pub probability: f64,
    pub reached_end: bool,
    pub seconds: f64,
    pub grade: Grade,
    /// Why the sentence is wrong: the rule behind the mistake ([`explain_mistakes`]).
    pub why: String,
    /// More sentences that make the same mistake, each with its correct form -
    /// the negative network's lesson, and nothing else's.
    pub variants: Vec<TutorCorrection>,
}

impl Lesson {
    /// A lesson of what the network wrote, before it is marked.
    pub fn new(exercise: Exercise, attempt: usize, mode: &str, continuation: &str) -> Lesson {
        Lesson {
            sentence: exercise.cue() + continuation,
            exercise,
            attempt,
            mode: mode.to_string(),
            continuation: continuation.to_string(),
            cost: 0.0,
            probability: 1.0,
            reached_end: false,
            seconds: 0.0,
            grade: Grade::default(),
            why: String::new(),
            variants: Vec::new(),
        }
    }

    /// Whether the network wrote nothing at all.
    pub fn empty(&self) -> bool {
        self.continuation.trim().is_empty()
    }

    /// What the teacher changed, span by span (empty when the sentence passed
    /// or was left uncorrected).
    ///
    /// Aligned character by character whatever the model's encoding, as
    /// Python's `Lesson.changes` is: the record is for the person reading it,
    /// and `Model::correct` aligns in the model's own units when it teaches.
    pub fn changes(&self) -> Vec<Edit> {
        let grade = &self.grade;
        if grade.passed || grade.correction.trim().is_empty() || self.sentence.trim().is_empty() {
            return Vec::new();
        }
        crate::diff::summary(self.sentence.trim(), grade.correction.trim(), 8, &Encoding::default())
    }

    /// Python's `Lesson.to_dict`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("exercise", self.exercise.to_json()),
            ("attempt", Json::Int(self.attempt as i64)),
            ("mode", Json::str(self.mode.clone())),
            ("continuation", Json::str(self.continuation.clone())),
            ("sentence", Json::str(self.sentence.clone())),
            ("cost", Json::Num(self.cost)),
            ("probability", Json::Num(self.probability)),
            ("reached_end", Json::Bool(self.reached_end)),
            ("seconds", Json::Num(self.seconds)),
            ("grade", self.grade.to_json()),
            ("changes", Json::Arr(self.changes().iter().map(Edit::to_json).collect())),
            ("why", Json::str(self.why.clone())),
            (
                "variants",
                Json::Arr(self.variants.iter().map(TutorCorrection::to_json).collect()),
            ),
        ])
    }
}

const GRADE_SCHEMA: &str = "{\"grades\": [{\"index\": <int>, \"grammar\": <0-10>, \"spelling\": <0-10>, \"fluency\": \
     <0-10>, \"error\": \"<one of the error types>\", \"correction\": \"<the whole sentence in correct English>\", \
     \"comment\": \"<one sentence of teaching>\"}, ...]}";

const GRADE_SYSTEM: &str = "You are a strict but constructive English teacher marking sentence completions. The \
     student is a beginner learning English: it is given the opening words of a sentence (the prefix, shown in <<>>) \
     and writes the rest. Mark what the student wrote, judged as part of the whole sentence. For each completion \
     give three marks out of 10 - grammar (agreement, tense, articles, prepositions, word order, sentence structure), \
     spelling and fluency (does it read like natural English) - and name the single most important mistake as one \
     of: {types}. Use \"none\" only for a sentence a teacher would accept as it stands. A completion that is empty, \
     cut off, gibberish or not English scores 0 with the error \"nonsense\" or \"fragment\". 'correction' is the \
     whole sentence written out in correct English: keep the prefix word for word, change only what follows it, \
     stay as close to what the student wrote as the mistake allows, and finish the sentence properly - the student \
     learns English by being shown this sentence, so it must be correct and natural on its own. 'comment' is one \
     short sentence of teaching addressed to the student, naming the rule that was broken. Reply with JSON only, no \
     prose, exactly of the form {schema} with one entry per completion, in the given order and with the given \
     index.";

/// The marking's system prompt.
pub fn grade_system() -> String {
    let types: Vec<String> = ERROR_TYPES.iter().map(|t| format!("\"{t}\"")).collect();
    GRADE_SYSTEM
        .replace("{types}", &types.join(", "))
        .replace("{schema}", GRADE_SCHEMA)
}

/// The list an answer's entries are in: the first of `keys` holding a list,
/// else the object itself when it has one of `single`, else a bare list.
fn answer_items(raw: &str, keys: &[&str], single: &[&str]) -> Vec<Json> {
    match loads_lenient(raw) {
        Some(doc @ Json::Obj(_)) => {
            for key in keys {
                if let Some(Json::Arr(list)) = doc.get(key) {
                    return list.clone();
                }
            }
            if single.iter().any(|k| doc.get(k).is_some()) {
                return vec![doc];
            }
            Vec::new()
        }
        Some(Json::Arr(list)) => list,
        _ => Vec::new(),
    }
}

/// An entry's index: `index` when it reads as an integer, else its position.
fn index_of(item: &Json, position: usize) -> i64 {
    match item.get("index") {
        None => position as i64,
        Some(value) => int_of(value).unwrap_or(position as i64),
    }
}

/// `{index: Grade}` for the entries of an answer that could be understood;
/// `graded_by` is the provider that gave them.
pub fn parse_grades(
    raw: &str,
    count: usize,
    grammar_weight: f64,
    threshold: f64,
    graded_by: &str,
) -> std::collections::BTreeMap<usize, Grade> {
    let mut grades = std::collections::BTreeMap::new();
    let items = answer_items(
        raw,
        &["grades", "reviews", "results", "items", "marks"],
        &["grammar", "score", "correction"],
    );
    for (position, item) in items.iter().enumerate() {
        if !matches!(item, Json::Obj(_)) {
            continue;
        }
        let index = index_of(item, position);
        if index < 0 || index as usize >= count || grades.contains_key(&(index as usize)) {
            continue;
        }
        let grammar = score_of(first_present(item, &["grammar", "grammar_score"]));
        let spelling = score_of(first_present(item, &["spelling", "spelling_score"]));
        let fluency = score_of(first_present(item, &["fluency", "fluency_score", "naturalness"]));
        let Some(score) = overall_score(grammar, spelling, fluency, grammar_weight)
            .or_else(|| score_of(first_present(item, &["score", "rating"])))
        else {
            continue; // nothing to mark with: leave it unrated
        };
        let correction = match first_present(item, &["correction", "corrected", "fixed"]) {
            Some(v) if crate::llm::fields::truthy(v) => squash(&py_str_of(v)),
            _ => String::new(),
        };
        let comment = first_present(item, &["comment", "critique", "feedback", "reason"])
            .and_then(Json::as_str)
            .unwrap_or("");
        grades.insert(
            index as usize,
            Grade {
                score: Some(score),
                grammar,
                spelling,
                fluency,
                passed: score >= threshold,
                error: error_of(first_present(item, &["error", "error_type", "mistake"])),
                correction,
                comment: clip(comment, MAX_COMMENT_CHARS),
                graded_by: graded_by.to_string(),
            },
        );
    }
    grades
}

/// The knobs of one [`grade_completions`].
#[derive(Clone, Debug, PartialEq)]
pub struct GradeOptions {
    pub topic: String,
    pub threshold: f64,
    pub grammar_weight: f64,
    /// The marker's model (`""` = the client's own).
    pub model: String,
    pub batch: usize,
    /// 0 = the default 0.2.
    pub temperature: f64,
    /// The provider behind the marker (`""` = the client's own).
    pub graded_by: String,
    /// The thinking level asked of the marker (Ollama's `think`; `None` sends nothing).
    pub think: Option<Json>,
}

impl Default for GradeOptions {
    fn default() -> GradeOptions {
        GradeOptions {
            topic: String::new(),
            threshold: 6.0,
            grammar_weight: 0.6,
            model: String::new(),
            batch: 10,
            temperature: 0.0,
            graded_by: String::new(),
            think: None,
        }
    }
}

/// Marks every lesson in place, `batch` per call.
///
/// An empty completion is failed without asking (`graded_by` `empty`), with
/// the teacher's model answer as the correction; a lesson the answer said
/// nothing usable about keeps no score and `graded_by` `unrated`, and counts
/// as a failure.  Returns what the marker thought before each batch's marks -
/// one line per batch that thought.
pub fn grade_completions(
    client: &dyn LlmClient,
    lessons: &mut [Lesson],
    o: &GradeOptions,
) -> Result<Vec<String>, ReviewError> {
    if o.batch < 1 {
        return Err(ReviewError::Invalid("batch must be >= 1".to_string()));
    }
    let graded_by = if o.graded_by.is_empty() {
        client.provider()
    } else {
        o.graded_by.as_str()
    };
    let temperature = if o.temperature > 0.0 { o.temperature } else { 0.2 };
    let system = grade_system();
    let mut thinking: Vec<String> = Vec::new();
    for chunk in lessons.chunks_mut(o.batch) {
        let mut asked: Vec<usize> = Vec::new();
        let mut body: Vec<String> = Vec::new();
        for (i, lesson) in chunk.iter_mut().enumerate() {
            if lesson.empty() {
                lesson.grade = Grade {
                    score: Some(0.0),
                    grammar: Some(0.0),
                    spelling: Some(0.0),
                    fluency: Some(0.0),
                    passed: false,
                    error: "nonsense".to_string(),
                    correction: lesson.exercise.answer.clone(),
                    comment: "Nothing was written: the sentence has to be finished.".to_string(),
                    graded_by: "empty".to_string(),
                };
                continue;
            }
            asked.push(i);
            let mut line = format!("[{i}] <<{}>>{}", lesson.exercise.cue(), lesson.continuation);
            if !lesson.exercise.focus.is_empty() {
                line.push_str(&format!("   (drilling: {})", lesson.exercise.focus));
            }
            body.push(line);
        }
        if asked.is_empty() {
            continue;
        }
        let mut user = String::new();
        if !o.topic.trim().is_empty() {
            user.push_str(&format!("Topic of the lesson: {}\n\n", o.topic.trim()));
        }
        user.push_str(&format!(
            "Mark these {} completions:\n{}\n\nReturn the JSON now.",
            asked.len(),
            body.join("\n")
        ));
        let mut options = LlmOptions::default()
            .system(system.clone())
            .model(o.model.as_str())
            .json()
            .temperature(temperature);
        if let Some(level) = &o.think {
            options = options.think(level.clone());
        }
        let (raw, thought) = client.complete_thinking(&user, &options)?;
        let thought = thought.split_whitespace().collect::<Vec<_>>().join(" ");
        if !thought.is_empty() {
            thinking.push(thought);
        }
        let mut parsed = parse_grades(&raw, chunk.len(), o.grammar_weight, o.threshold, graded_by);
        for i in asked {
            let lesson = &mut chunk[i];
            let Some(mut grade) = parsed.remove(&i) else {
                lesson.grade = Grade {
                    error: "other".to_string(),
                    correction: lesson.exercise.answer.clone(),
                    comment: "no grade returned".to_string(),
                    graded_by: "unrated".to_string(),
                    ..Grade::default()
                };
                continue;
            };
            if grade.correction.is_empty() {
                grade.correction = if grade.passed {
                    lesson.sentence.clone()
                } else {
                    lesson.exercise.answer.clone()
                };
            }
            lesson.grade = grade;
        }
    }
    Ok(thinking)
}

// -- why it is wrong, and the same mistake again --------------------------------------------------

const WHY_SCHEMA: &str = "{\"mistakes\": [{\"index\": <int>, \"why\": \"<why the sentence is wrong>\", \"again\": \
     [{\"wrong\": \"<another sentence with the same mistake>\", \"right\": \"<that sentence in correct English>\"}, \
     ...]}, ...]}";

const WHY_SYSTEM: &str = "You are an English teacher explaining a beginner's mistake and then showing it again. For \
     every sentence you are given the student's wrong sentence, the mistake you named and the correct sentence. \
     Answer two things. 'why' explains in one or two sentences why the sentence is wrong: name the rule that was \
     broken and what the student is doing instead - the pattern, not just this one sentence. 'again' is {n} MORE \
     examples of the SAME mistake: each 'wrong' is a different short sentence that breaks that same rule in that \
     same way - a different subject, verb or noun, never a repeat of the student's sentence or of another example - \
     and its 'right' is that same sentence in correct English, changed only where the mistake is, so the two differ \
     in the mistake and nothing else. Keep every sentence short, simple and about everyday life. Reply with JSON \
     only, no prose, exactly of the form {schema} with one entry per sentence, in the given order and with the \
     given index.";

/// `[(wrong, right)]` of an `again` list - objects or bare strings, blanks and
/// repeats dropped, a correction that corrects nothing emptied.
fn pairs_of(value: Option<&Json>, limit: usize) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut skip: Vec<String> = Vec::new();
    let Some(Json::Arr(entries)) = value else {
        return out;
    };
    for entry in entries {
        let (wrong, right) = match entry {
            Json::Str(text) => (squash(text), String::new()),
            item @ Json::Obj(_) => (
                truthy_text(item, &["wrong", "sentence", "example"]),
                truthy_text(item, &["right", "correction", "correct"]),
            ),
            _ => continue,
        };
        let key = wrong.to_lowercase();
        if wrong.is_empty() || skip.contains(&key) {
            continue;
        }
        let right = if right.to_lowercase() == key {
            String::new()
        } else {
            right
        };
        skip.push(key);
        out.push((wrong, right));
        if out.len() >= limit {
            break;
        }
    }
    out
}

/// What the teacher said about one mistake: why it is wrong, and more
/// sentences that are wrong in the same way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Explanation {
    pub why: String,
    pub again: Vec<(String, String)>,
}

/// `{index: Explanation}` for the entries of an answer that could be understood.
pub fn parse_explanations(raw: &str, count: usize, limit: usize) -> std::collections::BTreeMap<usize, Explanation> {
    let mut out = std::collections::BTreeMap::new();
    let items = answer_items(
        raw,
        &["mistakes", "explanations", "results", "items", "grades"],
        &["why", "again", "reason"],
    );
    for (position, item) in items.iter().enumerate() {
        if !matches!(item, Json::Obj(_)) {
            continue;
        }
        let index = index_of(item, position);
        if index < 0 || index as usize >= count || out.contains_key(&(index as usize)) {
            continue;
        }
        let why = truthy_text(item, &["why", "reason", "explanation"]);
        let again = first_present(item, &["again", "examples", "variants", "sentences"]);
        out.insert(
            index as usize,
            Explanation {
                why: clip(&why, MAX_WHY_CHARS),
                again: pairs_of(again, limit),
            },
        );
    }
    out
}

/// The settings of one [`explain_mistakes`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExplainOptions {
    pub topic: String,
    /// Sentences with the same mistake per failure (clamped to 1..=[`MAX_VARIANTS`]).
    pub count: usize,
    /// Their share of the failure's severity.
    pub weight: f64,
    pub model: String,
    pub batch: usize,
    /// 0 = the default 0.9.
    pub temperature: f64,
}

impl Default for ExplainOptions {
    fn default() -> ExplainOptions {
        ExplainOptions {
            topic: String::new(),
            count: DEFAULT_VARIANTS,
            weight: 0.5,
            model: String::new(),
            batch: 10,
            temperature: 0.0,
        }
    }
}

/// Asks the teacher *why* each of these sentences is wrong, and for `count`
/// more with the same mistake; fills [`Lesson::why`] and [`Lesson::variants`]
/// in place and returns how many were asked about.
///
/// Blank completions are skipped (nothing was written to be wrong about), and
/// an entry the teacher said nothing usable about keeps its empty `why` and no
/// variants.
pub fn explain_mistakes(
    client: &dyn LlmClient,
    lessons: &mut [&mut Lesson],
    o: &ExplainOptions,
) -> Result<usize, ReviewError> {
    if o.batch < 1 {
        return Err(ReviewError::Invalid("batch must be >= 1".to_string()));
    }
    let count = o.count.clamp(1, MAX_VARIANTS);
    if o.weight < 0.0 {
        return Err(ReviewError::Invalid(format!(
            "weight must be >= 0, got {}",
            crate::json::py_repr(o.weight)
        )));
    }
    let temperature = if o.temperature > 0.0 { o.temperature } else { 0.9 };
    let system = WHY_SYSTEM
        .replace("{schema}", WHY_SCHEMA)
        .replace("{n}", &count.to_string());
    let mut asked: Vec<&mut Lesson> = lessons
        .iter_mut()
        .filter(|l| !l.sentence.trim().is_empty() && !l.empty())
        .map(|l| &mut **l)
        .collect();
    let total = asked.len();
    for chunk in asked.chunks_mut(o.batch) {
        let body: Vec<String> = chunk
            .iter()
            .enumerate()
            .map(|(i, lesson)| {
                let error = if lesson.grade.error.is_empty() {
                    "other"
                } else {
                    lesson.grade.error.as_str()
                };
                let right = if lesson.grade.correction.trim().is_empty() {
                    lesson.exercise.answer.trim()
                } else {
                    lesson.grade.correction.trim()
                };
                let mut line = format!(
                    "[{i}] mistake: {error}\n    the student wrote: {}\n    correct English:   {}",
                    json_quote(lesson.sentence.trim()),
                    json_quote(right)
                );
                if !lesson.grade.comment.trim().is_empty() {
                    line.push_str(&format!("\n    you told the student: {}", lesson.grade.comment.trim()));
                }
                line
            })
            .collect();
        let mut user = String::new();
        if !o.topic.trim().is_empty() {
            user.push_str(&format!("Topic of the lesson: {}\n\n", o.topic.trim()));
        }
        user.push_str(&format!(
            "Explain these {} mistakes and show each one again:\n{}\n\nReturn the JSON now.",
            chunk.len(),
            body.join("\n")
        ));
        let options = LlmOptions::default()
            .system(system.clone())
            .model(o.model.as_str())
            .json()
            .temperature(temperature);
        let raw = client.generate(&user, &options)?;
        let parsed = parse_explanations(&raw, chunk.len(), count);
        for (i, lesson) in chunk.iter_mut().enumerate() {
            let Some(entry) = parsed.get(&i) else { continue };
            let own = lesson.sentence.trim().to_lowercase();
            lesson.why = entry.why.clone();
            lesson.variants = entry
                .again
                .iter()
                .filter(|(wrong, _)| wrong.to_lowercase() != own)
                .map(|(wrong, right)| TutorCorrection {
                    wrong: wrong.clone(),
                    right: right.clone(),
                    weight: o.weight,
                })
                .collect();
        }
    }
    Ok(total)
}

// -- the report card ------------------------------------------------------------------------------

/// Marks and mistakes of a set of lessons: `{"lessons", "graded", "passed",
/// "failed", "pass_rate", "mean_score", "mean_grammar", "mean_spelling",
/// "mean_fluency", "errors", "weakest"}`.
///
/// `errors` is most frequent first, ties in the order the mistakes were first
/// made (Python's `Counter.most_common`), and `weakest` its first three.
pub fn report_card(lessons: &[Lesson]) -> Json {
    let graded: Vec<&Lesson> = lessons.iter().filter(|l| l.grade.score.is_some()).collect();
    let mut errors: Vec<(String, i64)> = Vec::new();
    for lesson in lessons {
        let error = &lesson.grade.error;
        if error.is_empty() || error == "none" {
            continue;
        }
        match errors.iter_mut().find(|(e, _)| e == error) {
            Some(slot) => slot.1 += 1,
            None => errors.push((error.clone(), 1)),
        }
    }
    // a stable sort: equal counts stay in the order they were first met
    errors.sort_by(|a, b| b.1.cmp(&a.1));
    let means = |pick: fn(&Grade) -> Option<f64>| -> Json {
        let values: Vec<f64> = graded.iter().filter_map(|l| pick(&l.grade)).collect();
        fmean(&values).map(Json::Num).unwrap_or(Json::Null)
    };
    let passed = lessons.iter().filter(|l| l.grade.passed).count();
    Json::obj([
        ("lessons", Json::Int(lessons.len() as i64)),
        ("graded", Json::Int(graded.len() as i64)),
        ("passed", Json::Int(passed as i64)),
        ("failed", Json::Int((lessons.len() - passed) as i64)),
        (
            "pass_rate",
            if lessons.is_empty() {
                Json::Null
            } else {
                Json::Num(passed as f64 / lessons.len() as f64)
            },
        ),
        ("mean_score", means(|g| g.score)),
        ("mean_grammar", means(|g| g.grammar)),
        ("mean_spelling", means(|g| g.spelling)),
        ("mean_fluency", means(|g| g.fluency)),
        (
            "errors",
            Json::Obj(errors.iter().map(|(e, n)| (e.clone(), Json::Int(*n))).collect()),
        ),
        ("weakest", Json::strs(errors.iter().take(3).map(|(e, _)| e.clone()))),
    ])
}

/// The pairs of a report card, for a record that carries them after its own.
pub(crate) fn card_pairs(card: Json) -> Vec<(String, Json)> {
    match card {
        Json::Obj(pairs) => pairs,
        _ => Vec::new(),
    }
}

// -- what the lessons teach the negative network ---------------------------------------------------

/// `(faults, passed texts)` from a round of English lessons
/// (`radixnet/blame.py`'s `faults_from_lessons`).
///
/// The English tutor is the richest source of negatives there is: the
/// mistake it named is the reason, its mark the severity, its sentence of
/// teaching the note, and its correction rides along in the fault so only the
/// characters the teacher changed are blamed.  A widened lesson's variants
/// are faults of their own under the same reason, at their own share of the
/// severity, sourced `<source>:similar` and noted with the explanation.  The
/// sentences that passed, the corrections and the model answers all come back
/// as the texts that clear blame.
pub fn faults_from_lessons(lessons: &[Lesson], threshold: f64, source: &str) -> (Vec<Fault>, Vec<String>) {
    let mut faults: Vec<Fault> = Vec::new();
    let mut passed: Vec<String> = Vec::new();
    for lesson in lessons {
        let sentence = squash(&lesson.sentence);
        let grade = &lesson.grade;
        let correction = squash(&grade.correction);
        let answer = squash(&lesson.exercise.answer);
        if !sentence.is_empty() && grade.passed {
            passed.push(sentence.clone());
        } else if !sentence.is_empty() {
            let error = grade.error.trim().to_lowercase();
            let reason = if !error.is_empty() && error != "none" {
                error
            } else {
                classify(&grade.comment, &grade.graded_by, None, "")
            };
            let mut fault = Fault::new(
                &sentence,
                &reason,
                severity_from_rating(grade.score, threshold),
                &grade.comment,
                source,
            );
            if !correction.is_empty() && correction != sentence {
                fault.correction = correction.clone();
            }
            let severity = fault.severity;
            faults.push(fault);
            let why = squash(&lesson.why);
            for variant in &lesson.variants {
                let wrong = squash(&variant.wrong);
                let right = squash(&variant.right);
                if wrong.is_empty() || wrong == sentence {
                    continue;
                }
                let weight = if variant.weight > 0.0 { variant.weight } else { 0.0 };
                let mut similar = Fault::new(
                    &wrong,
                    &reason,
                    severity * weight,
                    if why.is_empty() { &grade.comment } else { &why },
                    &format!("{source}:similar"),
                );
                if !right.is_empty() && right != wrong {
                    similar.correction = right.clone();
                    if !passed.contains(&right) {
                        passed.push(right);
                    }
                }
                faults.push(similar);
            }
        }
        for text in [correction, answer] {
            if !text.is_empty() && !passed.contains(&text) {
                passed.push(text);
            }
        }
    }
    (faults, passed)
}

/// Feeds a round of English lessons into the negative network: the failures
/// blame it, and - with `clear_passes` - the passed sentences, corrections and
/// model answers take blame off what they share with known failures.
pub fn teach_lessons(
    negative: &mut Model,
    lessons: &[Lesson],
    threshold: f64,
    clear_passes: bool,
    source: &str,
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    let source = if source.is_empty() { "tutor" } else { source };
    let (faults, passed) = faults_from_lessons(lessons, threshold, source);
    let clearing: &[String] = if clear_passes { &passed } else { &[] };
    let mut report = teach(negative, &faults, clearing, o)?;
    report.source = source.to_string();
    report.threshold = Some(threshold);
    report.faults = faults;
    report.passed = passed.len();
    crate::log_info!(
        LOG,
        "{source}: blamed {} failure(s) over {} edge(s), cleared {} of {} passed",
        report.blamed,
        report.edges,
        report.cleared,
        report.passed
    );
    Ok(report)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The corpus the fake teacher holds to be correct English.
    pub(crate) const CORPUS: [&str; 3] = [
        "the cat sat on the mat",
        "the dogs run in the park",
        "the cat likes the mat",
    ];

    /// A teacher that answers the way `tests/test_tutor.py`'s fake one does,
    /// and remembers what it was asked.
    pub(crate) struct Teacher {
        pub asked: Mutex<Vec<(String, LlmOptions)>>,
        /// Answers the exercise requests with this instead of the pool.
        pub exercises: Option<String>,
    }

    impl Teacher {
        pub(crate) fn new() -> Teacher {
            Teacher {
                asked: Mutex::new(Vec::new()),
                exercises: None,
            }
        }

        /// The prompts of one kind of call: `exercises`, `grades`, `why`,
        /// `drills` or `plan`.
        pub(crate) fn prompts(&self, kind: &str) -> Vec<String> {
            let marker = match kind {
                "exercises" => "writing exercises",
                "grades" => "marking sentence completions",
                "why" => "explaining a beginner's mistake",
                "drills" => "model sentences",
                _ => "planning the next lessons",
            };
            self.asked
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, o)| o.system.contains(marker))
                .map(|(p, _)| p.clone())
                .collect()
        }
    }

    /// The number written just before `after` (Python's `re.search(r"(\d+) after")`).
    fn number_before(text: &str, after: &str) -> usize {
        text.split(after)
            .next()
            .filter(|head| head.len() < text.len())
            .and_then(|head| head.split_whitespace().last())
            .and_then(|n| n.parse().ok())
            .unwrap_or(2)
    }

    /// The fake teacher's rule: only the corpus is correct English.
    fn marked(prefix: &str, continuation: &str) -> String {
        let sentence = squash(&format!("{prefix}{continuation}"));
        if CORPUS.contains(&sentence.as_str()) {
            return format!(
                "\"grammar\": 9, \"spelling\": 8, \"fluency\": 7, \"error\": \"none\", \"correction\": \
                 {}, \"comment\": \"Well written.\"",
                json_quote(&sentence)
            );
        }
        let correction = CORPUS
            .iter()
            .find(|t| t.starts_with(prefix.trim()))
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("{} the mat", prefix.trim()));
        format!(
            "\"grammar\": 2, \"spelling\": 3, \"fluency\": 1, \"error\": \"subject-verb agreement\", \"correction\": \
             {}, \"comment\": \"A plural subject takes a plural verb.\"",
            json_quote(&correction)
        )
    }

    impl LlmClient for Teacher {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://teacher"
        }
        fn model(&self) -> &str {
            "teacher"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.asked.lock().unwrap().push((prompt.to_string(), o.clone()));
            let system = o.system.as_str();
            if system.contains("writing exercises") {
                if let Some(answer) = &self.exercises {
                    return Ok(answer.clone());
                }
                let count = number_before(system, " entries");
                let pool = [
                    "{\"prefix\": \"the cat sat on\", \"focus\": \"prepositions of place\", \"answer\": \"the cat sat \
                     on the mat\"}",
                    "{\"prefix\": \"the dogs run\", \"focus\": \"subject-verb agreement\", \"answer\": \"the dogs \
                     run in the park\"}",
                    "{\"prefix\": \"the cat likes\", \"focus\": \"verb agreement\", \"answer\": \"the cat likes the \
                     mat\"}",
                ];
                let items: Vec<&str> = (0..count).map(|i| pool[i % pool.len()]).collect();
                return Ok(format!("{{\"exercises\": [{}]}}", items.join(", ")));
            }
            if system.contains("marking sentence completions") {
                let mut grades = Vec::new();
                for line in prompt.lines() {
                    let Some(rest) = line.strip_prefix('[') else { continue };
                    let Some((index, rest)) = rest.split_once("] <<") else {
                        continue;
                    };
                    let Some((prefix, rest)) = rest.split_once(">>") else {
                        continue;
                    };
                    let continuation = rest.split("   (drilling:").next().unwrap_or("");
                    grades.push(format!("{{\"index\": {index}, {}}}", marked(prefix, continuation)));
                }
                return Ok(format!("{{\"grades\": [{}]}}", grades.join(", ")));
            }
            if system.contains("explaining a beginner's mistake") {
                let count = number_before(system, " MORE examples");
                let mut mistakes = Vec::new();
                for line in prompt.lines() {
                    let Some(rest) = line.strip_prefix('[') else { continue };
                    let Some((index, error)) = rest.split_once("] mistake: ") else {
                        continue;
                    };
                    let again: Vec<String> = (0..count)
                        .map(|i| {
                            format!(
                                "{{\"wrong\": \"the dogs sits on the mat {i}\", \"right\": \"the dogs sit on the mat \
                                 {i}\"}}"
                            )
                        })
                        .collect();
                    mistakes.push(format!(
                        "{{\"index\": {index}, \"why\": \"A plural subject takes a plural verb; the student breaks \
                         that rule ({}).\", \"again\": [{}]}}",
                        error.trim(),
                        again.join(", ")
                    ));
                }
                return Ok(format!("{{\"mistakes\": [{}]}}", mistakes.join(", ")));
            }
            if system.contains("model sentences") {
                let count = number_before(system, " lines");
                let lines: Vec<String> = (0..count)
                    .map(|i| format!("{}. the cat sat on the mat number {i}", i + 1))
                    .collect();
                return Ok(lines.join("\n"));
            }
            if system.contains("planning the next lessons") {
                return Ok(
                    "{\"summary\": \"The student writes verbs badly.\", \"prompt\": \"Drill subject-verb \
                           agreement first, then plurals. Keep it at beginner.\", \"lessons\": [{\"focus\": \
                           \"subject-verb agreement\", \"targets\": \"agreement\", \"topic\": \"animals\", \"why\": \
                           \"Nearly every sentence lost marks here.\"}, {\"focus\": \"plural nouns\", \"targets\": \
                           \"plural\", \"topic\": \"the market\", \"why\": \"Plurals were shaky.\"}]}"
                        .to_string(),
                );
            }
            Ok("unexpected request".to_string())
        }
        fn chat(&self, _messages: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn lesson(prefix: &str, continuation: &str) -> Lesson {
        let exercise = Exercise {
            id: "e1".to_string(),
            prefix: prefix.to_string(),
            focus: "agreement".to_string(),
            answer: format!("{prefix} the mat"),
        };
        Lesson::new(exercise, 0, "beam", continuation)
    }

    #[test]
    fn the_small_things_are_python_s() {
        assert_eq!(cue("the cat  "), "the cat ");
        assert_eq!(cue(""), "");
        assert_eq!(clip("  a   b  ", 10), "a b");
        assert_eq!(clip("abcdef", 4), "abc…");
        assert_eq!(clip("ab cdef", 4), "ab…");
        assert_eq!(error_type("subject-verb agreement"), "agreement");
        assert_eq!(error_type("Verb Tense"), "tense");
        assert_eq!(error_type("word_order"), "word-order");
        assert_eq!(error_type("no error"), "none");
        assert_eq!(error_type("style"), "other");
        for name in ERROR_TYPES {
            assert_eq!(error_type(name), *name);
        }
        assert_eq!(overall_score(Some(10.0), Some(0.0), Some(0.0), 0.6), Some(6.0));
        assert_eq!(overall_score(Some(4.0), Some(8.0), Some(6.0), 0.5), Some(5.5));
        assert_eq!(overall_score(Some(7.0), None, None, 0.6), Some(7.0));
        assert_eq!(overall_score(None, Some(6.0), Some(8.0), 0.6), Some(7.0));
        assert_eq!(overall_score(None, None, None, 0.6), None);
        assert_eq!(score_of(Some(&Json::str("7/10"))), Some(7.0));
        assert_eq!(score_of(Some(&Json::Int(42))), Some(10.0));
        assert_eq!(score_of(Some(&Json::Num(-1.0))), Some(0.0));
        assert_eq!(score_of(Some(&Json::str("n/a"))), None);
        assert_eq!(json_quote("say \"hi\"\n\u{8}é"), "\"say \\\"hi\\\"\\n\\bé\"");
    }

    #[test]
    fn exercises_are_read_whatever_shape_they_came_in() {
        let raw = "{\"exercises\": [{\"prefix\": \"the cat sat on\", \"focus\": \"prepositions\", \"answer\": \"the \
                   cat sat on the mat\"}, {\"prefix\": \"the dogs run\", \"focus\": \"agreement\", \"answer\": \"in \
                   the park\"}]}";
        let got = parse_exercises(raw, 5, 2);
        assert_eq!((got[0].id.as_str(), got[0].prefix.as_str()), ("r2e1", "the cat sat on"));
        assert_eq!(got[0].cue(), "the cat sat on ");
        assert_eq!(got[1].answer, "the dogs run in the park");
        let drift = "[\"  the children were playing.  \", {\"opening\": \"The children were playing\"}, {\"stem\": \
                     \"every morning she\", \"point\": \"present simple\", \"solution\": \"every morning she \
                     walks\"}, {\"nothing\": \"usable\"}, 42]";
        let got = parse_exercises(drift, 5, 1);
        let prefixes: Vec<&str> = got.iter().map(|e| e.prefix.as_str()).collect();
        assert_eq!(prefixes, vec!["the children were playing", "every morning she"]);
        assert_eq!(got[1].focus, "present simple");
        let lines = parse_exercises("1. the cat sat on\n2. the dogs run\n", 5, 1);
        assert_eq!(lines.len(), 2);
        assert!(parse_exercises("", 3, 1).is_empty());
    }

    #[test]
    fn the_teacher_is_asked_for_exercises_the_python_way() {
        let teacher = Teacher::new();
        let req = ExerciseRequest {
            topic: " animals ".to_string(),
            count: 2,
            weak: vec!["none".to_string(), "agreement".to_string()],
            brief: "Drill  plurals.\nThen tense.".to_string(),
            focus: " past tense ".to_string(),
            model: "fake".to_string(),
            ..Default::default()
        };
        let got = write_exercises(&teacher, &req).unwrap();
        assert_eq!(got.len(), 2);
        let asked = teacher.asked.lock().unwrap();
        let (prompt, o) = &asked[0];
        assert_eq!(
            prompt,
            "Topic: animals\nLevel: beginner\nThe plan for this batch of lessons: Drill plurals. Then tense.\nEvery \
             exercise must drill: past tense\nThe student keeps making these mistakes, so drill them: \
             agreement.\nWrite the 2 exercises now."
        );
        assert!(o.system.contains("The prefix must be 3 to 6 words"));
        assert!(o.system.ends_with("with exactly 2 entries."));
        assert!(o.json && o.model == "fake");
        assert_eq!(o.options, vec![("temperature".to_string(), Json::Num(0.9))]);
        drop(asked);
        let mut none = Teacher::new();
        none.exercises = Some("```\n```".to_string());
        let err = write_exercises(&none, &req).unwrap_err();
        assert_eq!(err.to_string(), "the teacher model 'fake' returned no usable exercises");
        assert!(matches!(err, ReviewError::Llm(_)));
        assert!(write_exercises(
            &teacher,
            &ExerciseRequest {
                topic: " ".to_string(),
                ..Default::default()
            }
        )
        .is_err());
        let drills = drill_sentences(&teacher, "animals", 2, &["agreement".to_string()], "").unwrap();
        assert_eq!(
            drills,
            vec!["the cat sat on the mat number 0", "the cat sat on the mat number 1"]
        );
        assert_eq!(
            teacher.prompts("drills")[0],
            "Topic: animals\nEach sentence must clearly demonstrate the correct use of: agreement.\nWrite the 2 \
             sentences now."
        );
    }

    #[test]
    fn grades_are_marked_in_batches_and_empties_are_failed_unasked() {
        let teacher = Teacher::new();
        let mut lessons = vec![
            lesson("the cat sat on", "the mat"),
            lesson("the dogs run", "s fast"),
            lesson("the cat likes", "  "),
        ];
        let o = GradeOptions {
            topic: "animals".to_string(),
            threshold: 6.0,
            batch: 2,
            ..Default::default()
        };
        grade_completions(&teacher, &mut lessons, &o).unwrap();
        let first = &lessons[0].grade;
        assert!(first.passed);
        assert!((first.score.unwrap() - (0.6 * 9.0 + 0.4 * 7.5)).abs() < 1e-12);
        assert_eq!((first.error.as_str(), first.graded_by.as_str()), ("none", "ollama"));
        let second = &lessons[1].grade;
        assert!(!second.passed);
        assert_eq!(second.error, "agreement");
        assert_eq!(second.correction, "the dogs run in the park");
        let empty = &lessons[2].grade;
        assert_eq!((empty.graded_by.as_str(), empty.error.as_str()), ("empty", "nonsense"));
        assert_eq!(empty.correction, "the cat likes the mat");
        let prompts = teacher.prompts("grades");
        assert_eq!(prompts.len(), 1, "the empty completion is not asked about");
        assert_eq!(
            prompts[0],
            "Topic of the lesson: animals\n\nMark these 2 completions:\n[0] <<the cat sat on >>the mat   (drilling: \
             agreement)\n[1] <<the dogs run >>s fast   (drilling: agreement)\n\nReturn the JSON now."
        );
        let changes = lessons[1].changes();
        assert!(!changes.is_empty());
        assert!(lessons[0].changes().is_empty(), "a pass changes nothing");
        let card = report_card(&lessons);
        assert_eq!(card.at("lessons").as_i64(), Some(3));
        assert_eq!(card.at("passed").as_i64(), Some(1));
        assert_eq!(card.at("errors").render(0), "{\"agreement\":1,\"nonsense\":1}");
        assert_eq!(card.at("weakest").to_strings(), vec!["agreement", "nonsense"]);
    }

    #[test]
    fn an_unreadable_grade_is_unrated() {
        let parsed = parse_grades(
            "{\"grades\": [{\"index\": \"1\", \"score\": \"8/10\", \"error\": \"Verb Tense\", \"fixed\": \" a  b \"}, \
             {\"index\": 0, \"grammar\": null, \"grammar_score\": 5}, {\"index\": 9, \"score\": 1}]}",
            2,
            0.6,
            6.0,
            "chatgpt",
        );
        assert_eq!(parsed.len(), 1);
        let g = &parsed[&1];
        assert_eq!((g.score, g.passed, g.error.as_str()), (Some(8.0), true, "tense"));
        assert_eq!(g.correction, "a b");
        assert_eq!(g.graded_by, "chatgpt");
        assert!(parse_grades("[{\"rating\": 3}]", 1, 0.6, 6.0, "ollama")[&0].score == Some(3.0));
    }

    #[test]
    fn mistakes_are_explained_and_shown_again() {
        let teacher = Teacher::new();
        let mut lessons = [lesson("the dogs", "runs fast"), lesson("the cat", "")];
        lessons[0].grade = Grade {
            error: "agreement".to_string(),
            correction: "the dogs run \"fast\"".to_string(),
            comment: "Plural verbs.".to_string(),
            ..Grade::default()
        };
        let mut refs: Vec<&mut Lesson> = lessons.iter_mut().collect();
        let asked = explain_mistakes(
            &teacher,
            &mut refs,
            &ExplainOptions {
                count: 2,
                weight: 0.25,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(asked, 1, "a blank completion is not asked about");
        assert!(lessons[0].why.starts_with("A plural subject takes a plural verb"));
        assert_eq!(lessons[0].variants.len(), 2);
        assert_eq!(lessons[0].variants[1].right, "the dogs sit on the mat 1");
        assert_eq!(lessons[0].variants[0].weight, 0.25);
        assert_eq!(
            teacher.prompts("why")[0],
            "Explain these 1 mistakes and show each one again:\n[0] mistake: agreement\n    the student wrote: \"the \
             dogs runs fast\"\n    correct English:   \"the dogs run \\\"fast\\\"\"\n    you told the student: Plural \
             verbs.\n\nReturn the JSON now."
        );
        let parsed = parse_explanations(
            "{\"why\": \" it   is \", \"examples\": [\"x y\", \"X Y\", {\"sentence\": \"a\", \"correct\": \"A\"}, \
             {\"wrong\": \"\"}]}",
            1,
            5,
        );
        assert_eq!(parsed[&0].why, "it is");
        assert_eq!(
            parsed[&0].again,
            vec![("x y".to_string(), String::new()), ("a".to_string(), String::new())]
        );
    }

    #[test]
    fn a_round_of_lessons_becomes_faults_and_clearing_texts() {
        let mut failed = lesson("the dogs", "runs fast");
        failed.grade = Grade {
            score: Some(2.0),
            error: "agreement".to_string(),
            correction: "the dogs run fast".to_string(),
            comment: "Plural verbs.".to_string(),
            ..Grade::default()
        };
        failed.why = "The rule.".to_string();
        failed.variants = vec![
            TutorCorrection {
                wrong: "the cats sits".to_string(),
                right: "the cats sit".to_string(),
                weight: 0.5,
            },
            TutorCorrection {
                wrong: "the dogs runs fast".to_string(),
                right: String::new(),
                weight: 0.5,
            },
        ];
        let mut good = lesson("the cat sat on", "the mat");
        good.grade.passed = true;
        good.grade.correction = "the cat sat on the mat".to_string();
        let (faults, passed) = faults_from_lessons(&[failed.clone(), good], 6.0, "tutor");
        assert_eq!(
            faults.len(),
            2,
            "the variant that is the student's own sentence is dropped"
        );
        assert_eq!(faults[0].reason, "agreement");
        assert_eq!(faults[0].correction, "the dogs run fast");
        assert_eq!(faults[1].source, "tutor:similar");
        assert_eq!(faults[1].note, "The rule.");
        assert!((faults[1].severity - faults[0].severity * 0.5).abs() < 1e-12);
        assert_eq!(
            passed,
            vec![
                "the cats sit",
                "the dogs run fast",
                "the dogs the mat",
                "the cat sat on the mat"
            ]
        );
        let mut negative = Model::new_negative(0, &Default::default()).unwrap();
        let report = teach_lessons(&mut negative, &[failed], 6.0, true, "", &TeachOptions::default()).unwrap();
        assert_eq!((report.blamed, report.source.as_str()), (2, "tutor"));
        assert_eq!(report.reasons, vec![("agreement".to_string(), 2)]);
        assert_eq!(report.passed, 3);
    }
}
