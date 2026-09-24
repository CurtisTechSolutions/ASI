//! The adversarial LLM review and the prompt-driven corpus, over any LLM
//! client (`radixnet/ollama.py`, `go/radixnet/review.go`), and the bridge from
//! a review to the negative network (`radixnet/blame.py`'s
//! `faults_from_reviews` / `teach_reviews`).
//!
//! Two ways of hooking the network into an LLM, and they are the pair 2NRL
//! wants:
//!
//! * **A corpus from a prompt** - [`corpus_from_prompt`] asks for N lines about
//!   a topic, either correct (`good`) or deliberately wrong (`garbage`).
//! * **The adversarial review** - [`review_texts`] makes the LLM the harsh
//!   critic: every text gets a mark out of 10, a pass / fail verdict against
//!   the threshold and a one-sentence critique.  [`adversarial_review`] runs it
//!   over the network's own samples (or given texts) and splits them into what
//!   passed and what did not; [`teach_reviews`] hands the failures to the
//!   negative network, with the critique picking the reason and the mark
//!   setting the severity.
//!
//! Both only ever call [`LlmClient::generate`], so ChatGPT reviews as happily as
//! a local model does.  The prompts are the Python ones byte for byte - the
//! parity suite compares what a fake LLM receives from each port - which is
//! why a pass mark is printed with [`format_g`] and the texts are numbered by
//! their place in their batch, as Python numbers them.
//!
//! The conversation marking ([`chat_line`], [`review_conversation`]) lives here
//! too, beside the review it is a variant of, for the chat loop to build on.
//!
//! # Why the verdict is not the LLM's
//!
//! The reviewer is asked for a verdict and the answer is read for one, but the
//! verdict recorded is always the mark against the threshold: a model that
//! writes `"rating": 5, "verdict": "pass"` has contradicted itself, and the
//! number is the part of the answer the loop can act on consistently.

use std::collections::BTreeMap;

use crate::blame::{classify, severity_from_rating, teach, Fault, TeachOptions, TeachReport};
use crate::http::ApiError;
use crate::json::Json;
use crate::llm::fields::{py_str_of, truthy};
use crate::llm::{format_g, loads_lenient, parse_lines, LlmClient, LlmError, LlmOptions};
use crate::model::{GenerateOptions, Model, PredictOptions};

/// What this module's own lines are filed under.
const LOG: &str = "llm";

/// The two kinds of corpus an LLM is asked for.
pub const STYLES: &[&str] = &["good", "garbage"];

/// How many texts go into one review call.
pub const DEFAULT_BATCH: usize = 20;

/// The pass mark out of 10 when none is given.
pub const DEFAULT_THRESHOLD: f64 = 6.0;

const CORPUS_SYSTEM: &str = "You produce training data for a small character-level language model. Answer with \
     exactly {n} lines and nothing else: one short, self-contained sentence per line, plain text, no numbering, no \
     bullets, no quotes, no blank lines, no headings and no commentary. ";

const STYLE_GOOD: &str = "Every line must be correct, natural and factual, in the style and about the topic requested.";

const STYLE_GARBAGE: &str = "Every line must be deliberately WRONG in a way a careful reader would reject: false \
     facts, scrambled word order, broken grammar, nonsense words, contradictions. Keep the requested topic and \
     vocabulary so the errors are the only difference from good text.";

const REVIEW_SYSTEM: &str = "You are an adversarial reviewer of text produced by a small experimental language \
     model. Assume every text is flawed and hunt for the flaws: gibberish, broken grammar, wrong word order, \
     truncated or repeated fragments, contradictions, false statements, incoherence. Rate each text from 0 to 10: 10 \
     = indistinguishable from correct, fluent, factual human writing; 6 = acceptable with minor flaws; 0 = pure \
     gibberish. The verdict is \"pass\" for a rating of {threshold} or more, otherwise \"fail\". Reply with JSON \
     only, no prose, exactly of the form {\"reviews\": [{\"index\": <int>, \"rating\": <number>, \"verdict\": \
     \"pass\" or \"fail\", \"critique\": \"<one sentence naming the worst flaw, or 'no flaw found'>\"}, ...]} with \
     one entry per text, in the given order and with the given index.";

const CHAT_SYSTEM: &str = "You are having a short, ordinary conversation with a very small character-level neural \
     network that is learning to talk. It answers by continuing the last few words you wrote, so every line you \
     write must be short, plain and concrete, and must end on words that are easy to carry on from. Never mention \
     that it is a model, never explain yourself, never ask more than one thing at a time, and never write more \
     than one sentence. Reply with the next thing you would say and nothing else.";

const CONVERSATION_SYSTEM: &str = "You are marking a conversation between a person and a very small \
     character-level neural network that is learning to talk. Mark each of the network's lines out of 10 for one \
     thing only: is it a real reply to the line before it - does it follow on, is it about the same thing, is it a \
     sentence at all. Ignore style, length and ambition: a short plain line that follows on is a 10. A line that \
     merely repeats what was just said, that is gibberish, or that answers something nobody asked is 0 to 3. \
     {threshold} out of 10 is a pass. Reply with JSON only, of the form {\"reviews\": [{\"index\": <n>, \
     \"rating\": <0-10>, \"critique\": \"<one sentence>\"}, ...], \"overall\": {\"rating\": <0-10>, \"critique\": \
     \"<one sentence about the conversation as a whole>\"}}.";

/// Who is who in a transcript: the LLM partner, then the network.
pub const CHAT_SPEAKERS: [&str; 2] = ["Partner", "Model"];

/// Why a corpus or a review did not come back: the request was wrong (Python's
/// `ValueError`, a 400), or the LLM failed (a 502).
#[derive(Clone, Debug, PartialEq)]
pub enum ReviewError {
    Invalid(String),
    Llm(LlmError),
}

impl std::fmt::Display for ReviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReviewError::Invalid(message) => f.write_str(message),
            ReviewError::Llm(err) => f.write_str(&err.message),
        }
    }
}

impl From<LlmError> for ReviewError {
    fn from(err: LlmError) -> ReviewError {
        ReviewError::Llm(err)
    }
}

impl From<ReviewError> for String {
    fn from(err: ReviewError) -> String {
        err.to_string()
    }
}

impl From<ReviewError> for ApiError {
    fn from(err: ReviewError) -> ApiError {
        match err {
            ReviewError::Invalid(message) => ApiError::bad_request(message),
            ReviewError::Llm(err) => err.into(),
        }
    }
}

fn invalid<S: Into<String>>(message: S) -> ReviewError {
    ReviewError::Invalid(message.into())
}

// -- the corpus ---------------------------------------------------------------------------------

/// Asks the LLM for `lines` lines of text about `prompt`; `style` is `good`
/// (correct text) or `garbage` (deliberately wrong text) - exactly the pair of
/// inputs 2NRL wants.  `model` overrides the client's own (`""` = its own).
pub fn corpus_from_prompt(
    client: &dyn LlmClient,
    prompt: &str,
    lines: usize,
    style: &str,
    model: &str,
) -> Result<Vec<String>, ReviewError> {
    if prompt.trim().is_empty() {
        return Err(invalid("prompt must be a non-empty string"));
    }
    if lines < 1 {
        return Err(invalid("lines must be >= 1"));
    }
    let style = match style.trim().to_lowercase().as_str() {
        "" | "good" => "good",
        "garbage" => "garbage",
        _ => {
            return Err(invalid(format!(
                "style must be one of {} (got {})",
                STYLES.join(", "),
                crate::negative::python_repr(&style.trim().to_lowercase())
            )))
        }
    };
    let system =
        CORPUS_SYSTEM.replace("{n}", &lines.to_string()) + if style == "good" { STYLE_GOOD } else { STYLE_GARBAGE };
    let user = format!(
        "Topic / instructions: {}\n\nWrite the {lines} lines now.",
        prompt.trim()
    );
    let temperature = if style == "good" { 0.9 } else { 1.1 };
    let o = LlmOptions::default()
        .system(system)
        .model(model)
        .temperature(temperature);
    let raw = client.generate(&user, &o)?;
    let texts = parse_lines(&raw, Some(lines));
    crate::log_info!(LOG, "corpus: {} {style} line(s) about {:?}", texts.len(), prompt.trim());
    Ok(texts)
}

// -- the review ---------------------------------------------------------------------------------

/// One text as the reviewer marked it.
#[derive(Clone, Debug, PartialEq)]
pub struct Review {
    /// Its place in the reviewed set.
    pub index: usize,
    pub text: String,
    /// `None`, with verdict `"unrated"`, when the answer said nothing usable about it.
    pub rating: Option<f64>,
    /// `"pass"`, `"fail"` or `"unrated"`.
    pub verdict: String,
    pub critique: String,
    /// The line this text was a reply to, when it was one (a conversation).
    pub said: Option<String>,
}

impl Review {
    /// `{"index", "text", ["said",] "rating", "verdict", "critique"}`.
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("index".to_string(), Json::Int(self.index as i64)),
            ("text".to_string(), Json::str(self.text.clone())),
        ];
        if let Some(said) = &self.said {
            pairs.push(("said".to_string(), Json::str(said.clone())));
        }
        pairs.push(("rating".to_string(), self.rating.map(Json::Num).unwrap_or(Json::Null)));
        pairs.push(("verdict".to_string(), Json::str(self.verdict.clone())));
        pairs.push(("critique".to_string(), Json::str(self.critique.clone())));
        Json::Obj(pairs)
    }

    /// A review as it arrives in a request or a record (the inverse of [`Review::to_json`]).
    pub fn from_json(doc: &Json, position: usize) -> Review {
        Review {
            index: doc.at("index").as_i64().map(|i| i.max(0) as usize).unwrap_or(position),
            text: doc.at("text").as_str().unwrap_or("").to_string(),
            rating: doc.at("rating").as_f64(),
            verdict: doc.at("verdict").as_str().unwrap_or("").to_string(),
            critique: doc.at("critique").as_str().unwrap_or("").to_string(),
            said: doc.at("said").as_str().map(str::to_string),
        }
    }

    fn marked(index: usize, text: &str, rating: f64, threshold: f64, critique: &str) -> Review {
        Review {
            index,
            text: text.to_string(),
            rating: Some(rating),
            verdict: if rating >= threshold { "pass" } else { "fail" }.to_string(),
            critique: critique.to_string(),
            said: None,
        }
    }

    fn unmarked(index: usize, text: &str, rating: Option<f64>, verdict: &str, critique: &str) -> Review {
        Review {
            index,
            text: text.to_string(),
            rating,
            verdict: verdict.to_string(),
            critique: critique.to_string(),
            said: None,
        }
    }
}

/// A judge's verdict on a conversation as a whole.
#[derive(Clone, Debug, PartialEq)]
pub struct Overall {
    pub rating: Option<f64>,
    pub critique: String,
}

impl Overall {
    /// `{"rating", "critique"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("rating", self.rating.map(Json::Num).unwrap_or(Json::Null)),
            ("critique", Json::str(self.critique.clone())),
        ])
    }
}

/// A marked set, split into what passed and what did not.
#[derive(Clone, Debug, PartialEq)]
pub struct ReviewSummary {
    /// `"model"` (the network's own samples), `"given"`, or `"chat"`.
    pub source: String,
    /// The reviewer's model.
    pub model: String,
    pub threshold: f64,
    pub texts: Vec<String>,
    pub reviews: Vec<Review>,
    /// The mean over the rated texts; `None` when none was rated.
    pub mean_rating: Option<f64>,
    /// The share that passed; `None` when nothing was reviewed.
    pub pass_rate: Option<f64>,
    pub good: Vec<String>,
    /// Failed *and* unrated texts: nobody vouched for them.
    pub bad: Vec<String>,
    /// The judge's word on a whole conversation (`source == "chat"` only).
    pub overall: Option<Overall>,
}

impl ReviewSummary {
    /// `{"source", "model", "threshold", "texts", "reviews", "mean_rating",
    /// "pass_rate", "good", "bad"}`, and `overall` for a conversation.
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("source".to_string(), Json::str(self.source.clone())),
            ("model".to_string(), Json::str(self.model.clone())),
            ("threshold".to_string(), Json::Num(self.threshold)),
            ("texts".to_string(), Json::strs(self.texts.clone())),
            (
                "reviews".to_string(),
                Json::Arr(self.reviews.iter().map(Review::to_json).collect()),
            ),
            (
                "mean_rating".to_string(),
                self.mean_rating.map(Json::Num).unwrap_or(Json::Null),
            ),
            (
                "pass_rate".to_string(),
                self.pass_rate.map(Json::Num).unwrap_or(Json::Null),
            ),
            ("good".to_string(), Json::strs(self.good.clone())),
            ("bad".to_string(), Json::strs(self.bad.clone())),
        ];
        if self.source == "chat" {
            pairs.push((
                "overall".to_string(),
                self.overall.as_ref().map(Overall::to_json).unwrap_or(Json::Null),
            ));
        }
        Json::Obj(pairs)
    }
}

/// The system prompt of a review at `threshold`.
pub fn review_system(threshold: f64) -> String {
    REVIEW_SYSTEM.replace("{threshold}", &format_g(threshold))
}

/// Python's `min` / `max`, which keep the first argument unless the second is
/// strictly beyond it - so a NaN never wins.
fn clamp_mark(x: f64) -> f64 {
    let low = if x < 10.0 { x } else { 10.0 };
    if low > 0.0 {
        low
    } else {
        0.0
    }
}

/// Python's `float(value)` of what a JSON answer holds; `None` where it raises.
fn py_float(value: &Json) -> Option<f64> {
    match value {
        Json::Int(n) => Some(*n as f64),
        Json::Num(x) => Some(*x),
        Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Json::Str(s) => s.trim().replace('_', "").parse().ok(),
        _ => None,
    }
}

/// Python's `int(value)`; `None` where it raises.
fn py_int(value: &Json) -> Option<i64> {
    match value {
        Json::Int(n) => Some(*n),
        Json::Num(x) if x.is_finite() => Some(x.trunc() as i64),
        Json::Bool(b) => Some(*b as i64),
        Json::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `value.get(first) or value.get(second) or ...` as Python reads it: the
/// first truthy one, as `str()`, stripped; `""` when there is none.
fn first_text(item: &Json, keys: &[&str]) -> String {
    keys.iter()
        .filter_map(|k| item.get(k))
        .find(|v| truthy(v))
        .map(|v| py_str_of(v).trim().to_string())
        .unwrap_or_default()
}

/// `item.get("rating", item.get("score"))`: the rating when the key is there
/// at all (even as `null`), else the score.
fn rating_field(item: &Json) -> Option<&Json> {
    item.get("rating").or_else(|| item.get("score"))
}

/// `{index: (rating, critique)}` for the entries of a review answer that could
/// be understood, whatever shape the model drifted into: `{"reviews": [...]}`,
/// `{"results" | "items": [...]}`, a bare list, or one object when a single
/// text was asked about; `score` stands in for `rating`, and `reason` /
/// `comment` for `critique`.
pub fn parse_reviews(raw: &str, count: usize) -> BTreeMap<usize, (f64, String)> {
    let mut parsed = BTreeMap::new();
    let data = loads_lenient(raw);
    let single;
    let items: &[Json] = match &data {
        Some(doc @ Json::Obj(_)) => {
            let listed = doc
                .get("reviews")
                .filter(|v| !v.is_null())
                .or_else(|| doc.get("results").or_else(|| doc.get("items")).filter(|v| !v.is_null()));
            match listed {
                Some(Json::Arr(items)) => items,
                Some(_) => return parsed,
                None if doc.get("rating").is_some() || doc.get("score").is_some() => {
                    single = [doc.clone()];
                    &single
                }
                None => return parsed,
            }
        }
        Some(Json::Arr(items)) => items,
        _ => return parsed,
    };
    for (position, item) in items.iter().enumerate() {
        if !matches!(item, Json::Obj(_)) {
            continue;
        }
        let index = match item.get("index") {
            None => position as i64,
            Some(value) => py_int(value).unwrap_or(position as i64),
        };
        let Some(rating) = rating_field(item).and_then(py_float) else {
            continue;
        };
        if rating.is_nan() {
            continue;
        }
        let rating = clamp_mark(rating);
        let mut critique = first_text(item, &["critique", "reason", "comment"]);
        if critique.is_empty() {
            critique = "no critique given".to_string();
        }
        if index >= 0 && (index as usize) < count {
            parsed.entry(index as usize).or_insert((rating, critique));
        }
    }
    parsed
}

/// Asks the LLM to mark every text, in input order, `batch` texts per call.
///
/// A blank text is failed without asking (`"empty output"`); a text the answer
/// said nothing usable about comes back `"unrated"`, which still counts as a
/// failure but records that nobody said why.  `model` overrides the client's
/// own (`""` = its own); `context` tells the reviewer what the texts are meant
/// to be.
pub fn review_texts(
    client: &dyn LlmClient,
    texts: &[String],
    context: &str,
    model: &str,
    threshold: f64,
    batch: usize,
) -> Result<Vec<Review>, ReviewError> {
    if batch < 1 {
        return Err(invalid("batch must be >= 1"));
    }
    let system = review_system(threshold);
    let mut out = Vec::with_capacity(texts.len());
    for (chunk_no, chunk) in texts.chunks(batch).enumerate() {
        let start = chunk_no * batch;
        let asked: Vec<String> = chunk
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.trim().is_empty())
            .map(|(i, t)| format!("[{i}] {t}"))
            .collect();
        let mut parsed = BTreeMap::new();
        if !asked.is_empty() {
            let mut user = String::new();
            if !context.trim().is_empty() {
                user.push_str(&format!("Context: {}\n\n", context.trim()));
            }
            user.push_str(&format!(
                "Review these {} texts:\n{}\n\nReturn the JSON now.",
                asked.len(),
                asked.join("\n")
            ));
            let o = LlmOptions::default()
                .system(system.clone())
                .model(model)
                .json()
                .temperature(0.2);
            let raw = client.generate(&user, &o)?;
            parsed = parse_reviews(&raw, chunk.len());
        }
        for (i, text) in chunk.iter().enumerate() {
            let index = start + i;
            out.push(if text.trim().is_empty() {
                Review::unmarked(index, text, Some(0.0), "fail", "empty output")
            } else if let Some((rating, critique)) = parsed.get(&i) {
                Review::marked(index, text, *rating, threshold, critique)
            } else {
                Review::unmarked(index, text, None, "unrated", "no review returned")
            });
        }
    }
    crate::log_debug!(
        LOG,
        "reviewed {} text(s) at a pass mark of {}",
        texts.len(),
        format_g(threshold)
    );
    Ok(out)
}

/// `count` stochastic texts from the model: whole texts, or continuations of
/// `prefix` when one is given.  `seed` makes a run of whole texts
/// reproducible; without one the model's own generator is used (and moved on).
pub fn sample_texts(
    model: &mut Model,
    count: usize,
    prefix: &str,
    max_length: usize,
    temperature: f64,
    seed: Option<i64>,
) -> Result<Vec<String>, String> {
    if count < 1 {
        return Err("count must be >= 1".to_string());
    }
    if !prefix.is_empty() {
        let o = PredictOptions {
            length: max_length,
            mode: "sample".to_string(),
            temperature,
            max_length: Some(max_length),
            ..Default::default()
        };
        return (0..count)
            .map(|_| model.predict(prefix, &o).map(|found| found.best.full_text))
            .collect();
    }
    let o = GenerateOptions {
        max_length,
        mode: "sample".to_string(),
        temperature,
        count,
        seed,
        ..Default::default()
    };
    Ok(model.generate(&o)?.into_iter().map(|r| r.text).collect())
}

/// Splits a marked set into what passed and what did not (unrated texts count
/// as failures) and averages the marks.
///
/// It is separate from [`adversarial_review`] because a caller holding a model
/// lock has to sample and review in two steps: the sampling walks the graph
/// and must hold the lock, the reviewing is a network call and must not.
pub fn summarise_reviews(
    source: &str,
    model: &str,
    threshold: f64,
    texts: Vec<String>,
    reviews: Vec<Review>,
) -> ReviewSummary {
    let ratings: Vec<f64> = reviews.iter().filter_map(|r| r.rating).collect();
    let good: Vec<String> = reviews
        .iter()
        .filter(|r| r.verdict == "pass")
        .map(|r| r.text.clone())
        .collect();
    let bad: Vec<String> = reviews
        .iter()
        .filter(|r| r.verdict != "pass")
        .map(|r| r.text.clone())
        .collect();
    ReviewSummary {
        source: source.to_string(),
        model: model.to_string(),
        threshold,
        texts,
        mean_rating: (!ratings.is_empty()).then(|| crate::fsum::fsum(&ratings) / ratings.len() as f64),
        pass_rate: (!reviews.is_empty()).then(|| good.len() as f64 / reviews.len() as f64),
        reviews,
        good,
        bad,
        overall: None,
    }
}

/// The knobs of one [`adversarial_review`].
#[derive(Clone, Debug)]
pub struct ReviewOptions {
    /// Samples to draw from the model.
    pub count: usize,
    /// Continue this instead of generating whole texts.
    pub prefix: String,
    pub max_length: usize,
    pub temperature: f64,
    /// Review these instead of sampling from the model.
    pub texts: Option<Vec<String>>,
    pub threshold: f64,
    /// What the texts are meant to be: the reviewer's yardstick.
    pub context: String,
    /// The reviewer's model (`""` = the client's own).
    pub model: String,
    pub seed: Option<i64>,
    /// Texts per review call (0 = [`DEFAULT_BATCH`]).
    pub batch: usize,
}

impl Default for ReviewOptions {
    /// The Python defaults: eight samples of sixty characters, a pass at 6.
    fn default() -> ReviewOptions {
        ReviewOptions {
            count: 8,
            prefix: String::new(),
            max_length: 60,
            temperature: 1.0,
            texts: None,
            threshold: DEFAULT_THRESHOLD,
            context: String::new(),
            model: String::new(),
            seed: None,
            batch: 0,
        }
    }
}

/// Lets the LLM judge the network's own output (or `o.texts`) and splits it
/// into good and bad sets, ready for a 2NRL pass or for the negative network.
pub fn adversarial_review(
    model: Option<&mut Model>,
    client: &dyn LlmClient,
    o: &ReviewOptions,
) -> Result<ReviewSummary, ReviewError> {
    let (samples, source) = match (&o.texts, model) {
        (Some(texts), _) => (texts.clone(), "given"),
        (None, Some(model)) => (
            sample_texts(model, o.count, &o.prefix, o.max_length, o.temperature, o.seed).map_err(invalid)?,
            "model",
        ),
        (None, None) => return Err(invalid("either a model to sample from or texts to review is required")),
    };
    let batch = if o.batch == 0 { DEFAULT_BATCH } else { o.batch };
    let reviews = review_texts(client, &samples, &o.context, &o.model, o.threshold, batch)?;
    let reviewer = if o.model.is_empty() {
        client.model()
    } else {
        o.model.as_str()
    };
    Ok(summarise_reviews(source, reviewer, o.threshold, samples, reviews))
}

// -- conversing with the network, and marking the conversation ----------------------------------

/// The next line of the LLM's side of a conversation with the network;
/// `transcript` is what has been said so far as `(speaker, text)` pairs, and
/// an empty one opens the conversation.
///
/// The system prompt asks for short, plain lines ending on words that are easy
/// to carry on from, because that is what a character-level model can
/// actually reply to.  `temperature` is sent as given - 0 is greedy, as
/// Python's `chat_line` sends it (the default is the caller's: 0.8).
pub fn chat_line(
    client: &dyn LlmClient,
    transcript: &[(String, String)],
    topic: &str,
    persona: &str,
    model: &str,
    temperature: f64,
) -> Result<String, LlmError> {
    let mut system = CHAT_SYSTEM.to_string();
    if !topic.trim().is_empty() {
        system.push_str(&format!(" The conversation is about {}.", topic.trim()));
    }
    if !persona.trim().is_empty() {
        system.push_str(&format!(" You are {}.", persona.trim()));
    }
    let said: Vec<String> = transcript
        .iter()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(speaker, text)| format!("{speaker}: {text}"))
        .collect();
    let user = if !said.is_empty() {
        format!("The conversation so far:\n{}\n\nWrite your next line.", said.join("\n"))
    } else if !topic.trim().is_empty() {
        format!(
            "Open the conversation about {} with one short, plain line.",
            topic.trim()
        )
    } else {
        "Open the conversation with one short, plain line.".to_string()
    };
    let temperature = temperature.max(0.0);
    let o = LlmOptions::default()
        .system(system)
        .model(model)
        .temperature(temperature);
    Ok(first_line(&client.generate(&user, &o)?))
}

/// One line out of an LLM answer that may have written several, or quoted
/// itself with a speaker's name.
pub fn first_line(raw: &str) -> String {
    for line in crate::llm::splitlines(raw) {
        let mut text = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            continue;
        }
        for speaker in ["Partner:", "Model:", "You:", "Me:"] {
            if text
                .get(..speaker.len())
                .is_some_and(|head| head.eq_ignore_ascii_case(speaker))
            {
                text = text[speaker.len()..].trim().to_string();
            }
        }
        let text = text.trim_matches(|c| matches!(c, '"' | '\u{201c}' | '\u{201d}'));
        if !text.is_empty() {
            return text.to_string();
        }
    }
    String::new()
}

/// The judge's verdict on a conversation as a whole, when it gave one.
fn parse_overall(raw: &str) -> Option<Overall> {
    let data = loads_lenient(raw)?;
    if !matches!(data, Json::Obj(_)) {
        return None;
    }
    let item = ["overall", "conversation", "summary"]
        .iter()
        .filter_map(|k| data.get(k))
        .find(|v| truthy(v))?;
    if !matches!(item, Json::Obj(_)) {
        return None;
    }
    let rating = rating_field(item).and_then(py_float).map(clamp_mark);
    let critique = first_text(item, &["critique", "reason", "comment"]);
    if rating.is_none() && critique.is_empty() {
        return None;
    }
    Some(Overall {
        rating,
        critique: if critique.is_empty() {
            "no critique given".to_string()
        } else {
            critique
        },
    })
}

/// Marks every reply the network gave in a conversation - `(what was said to
/// it, what it replied)` pairs, in order - and the conversation as a whole.
///
/// The result is the [`summarise_reviews`] shape with `source` `"chat"`, so
/// [`teach_reviews`] takes it as it is, plus [`ReviewSummary::overall`].
pub fn review_conversation(
    client: &dyn LlmClient,
    exchanges: &[(String, String)],
    topic: &str,
    model: &str,
    threshold: f64,
) -> Result<ReviewSummary, LlmError> {
    let replies: Vec<String> = exchanges.iter().map(|(_, reply)| reply.clone()).collect();
    let blocks: Vec<String> = exchanges
        .iter()
        .enumerate()
        .filter(|(_, (_, reply))| !reply.trim().is_empty())
        .map(|(i, (said, reply))| format!("[{i}] {}: {said}\n    {}: {reply}", CHAT_SPEAKERS[0], CHAT_SPEAKERS[1]))
        .collect();
    let mut parsed = BTreeMap::new();
    let mut overall = None;
    if !blocks.is_empty() {
        let mut user = String::new();
        if !topic.trim().is_empty() {
            user.push_str(&format!("Topic: {}\n\n", topic.trim()));
        }
        user.push_str(&format!(
            "The conversation:\n{}\n\nReturn the JSON now.",
            blocks.join("\n\n")
        ));
        let o = LlmOptions::default()
            .system(CONVERSATION_SYSTEM.replace("{threshold}", &format_g(threshold)))
            .model(model)
            .json()
            .temperature(0.2);
        let raw = client.generate(&user, &o)?;
        parsed = parse_reviews(&raw, exchanges.len());
        overall = parse_overall(&raw);
    }
    let reviews: Vec<Review> = exchanges
        .iter()
        .enumerate()
        .map(|(i, (said, reply))| {
            let mut review = if reply.trim().is_empty() {
                Review::unmarked(i, reply, Some(0.0), "fail", "it said nothing")
            } else if let Some((rating, critique)) = parsed.get(&i) {
                Review::marked(i, reply, *rating, threshold, critique)
            } else {
                Review::unmarked(i, reply, None, "unrated", "no review returned")
            };
            review.said = Some(said.clone());
            review
        })
        .collect();
    let judge = if model.is_empty() { client.model() } else { model };
    let mut summary = summarise_reviews("chat", judge, threshold, replies, reviews);
    summary.overall = overall;
    Ok(summary)
}

// -- what a review teaches the negative network -------------------------------------------------

/// `(faults, passed texts)` from a review: everything the reviewer did not pass
/// becomes a fault whose reason comes from its critique and whose severity
/// comes from its mark; the texts it passed come back separately, to clear
/// blame.
pub fn faults_from_reviews(reviews: &[Review], threshold: f64, source: &str) -> (Vec<Fault>, Vec<String>) {
    let mut faults = Vec::new();
    let mut passed = Vec::new();
    for review in reviews {
        if review.text.is_empty() {
            continue;
        }
        let verdict = review.verdict.trim().to_lowercase();
        if verdict == "pass" {
            passed.push(review.text.clone());
            continue;
        }
        faults.push(Fault::new(
            &review.text,
            &classify(&review.critique, &verdict, review.rating, ""),
            severity_from_rating(review.rating, threshold),
            &review.critique,
            source,
        ));
    }
    (faults, passed)
}

/// Feeds a review straight into the negative network: the failures blame it,
/// and - with `clear_passes` - the passes take blame off what they share with
/// known failures.  The report says which source and threshold taught it,
/// which faults, and how many texts passed.
pub fn teach_reviews(
    negative: &mut Model,
    reviews: &[Review],
    threshold: f64,
    clear_passes: bool,
    source: &str,
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    let source = if source.is_empty() { "review" } else { source };
    let (faults, passed) = faults_from_reviews(reviews, threshold, source);
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

/// Everything a negative network has been blamed for, heaviest first -
/// `[{"reason", "blame", "fails", "fails_resets", "edges", "share"}]`, the
/// rows of Python's `NegativeNet.reasons()`, with how many live edges carry
/// each reason.
pub fn reason_rows(negative: &Model) -> Json {
    let Some(neg) = negative.g.neg.as_ref() else {
        return Json::Arr(Vec::new());
    };
    let mut edges = vec![0i64; neg.reason_names.len()];
    for (e, reasons) in neg.reasons.iter().enumerate() {
        if negative.g.is_edge_alive(e) {
            for r in reasons {
                if let Some(n) = edges.get_mut(r.id) {
                    *n += 1;
                }
            }
        }
    }
    let total: f64 = neg.reason_blame.iter().sum();
    let total = if total == 0.0 { 1.0 } else { total };
    let mut rows: Vec<(usize, &String)> = neg.reason_names.iter().enumerate().collect();
    rows.sort_by(|(a, name_a), (b, name_b)| {
        neg.reason_blame[*b]
            .partial_cmp(&neg.reason_blame[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(name_a.cmp(name_b))
    });
    Json::Arr(
        rows.into_iter()
            .map(|(id, name)| {
                Json::obj([
                    ("reason", Json::str(name.clone())),
                    ("blame", Json::Num(neg.reason_blame[id])),
                    ("fails", Json::Int(neg.reason_fails[id])),
                    (
                        "fails_resets",
                        Json::Int(neg.reason_fails_resets.get(&id).copied().unwrap_or(0)),
                    ),
                    ("edges", Json::Int(edges[id])),
                    ("share", Json::Num(neg.reason_blame[id] / total)),
                ])
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// How a scripted reviewer answers a prompt.
    type Script = Box<dyn Fn(&str, &LlmOptions) -> String + Send + Sync>;

    /// A reviewer that answers from a script and remembers what it was asked.
    pub(crate) struct Scripted {
        pub answer: Script,
        pub asked: Mutex<Vec<(String, LlmOptions)>>,
    }

    impl Scripted {
        pub(crate) fn new(answer: impl Fn(&str, &LlmOptions) -> String + Send + Sync + 'static) -> Scripted {
            Scripted {
                answer: Box::new(answer),
                asked: Mutex::new(Vec::new()),
            }
        }

        /// Marks every `[i] text` line: 9 when it says "good", else 2.
        pub(crate) fn reviewer() -> Scripted {
            Scripted::new(|prompt, _| {
                let mut reviews = Vec::new();
                for line in prompt.lines() {
                    if let Some(rest) = line.strip_prefix('[') {
                        if let Some((index, text)) = rest.split_once("] ") {
                            let rating = if text.to_lowercase().contains("good") { 9 } else { 2 };
                            reviews.push(format!(
                                "{{\"index\": {index}, \"rating\": {rating}, \"critique\": \"{}\"}}",
                                if rating >= 6 { "fine" } else { "nonsense" }
                            ));
                        }
                    }
                }
                format!("{{\"reviews\": [{}]}}", reviews.join(", "))
            })
        }
    }

    impl LlmClient for Scripted {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://scripted"
        }
        fn model(&self) -> &str {
            "scripted"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.asked.lock().unwrap().push((prompt.to_string(), o.clone()));
            Ok((self.answer)(prompt, o))
        }
        fn chat(&self, _messages: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_corpus_is_asked_for_as_python_asks_for_it() {
        let llm = Scripted::new(|_, _| "1. good line one\n\n2. good line two\n3. Good line one\n4. x".to_string());
        let lines = corpus_from_prompt(&llm, "  cats and dogs ", 3, "good", "").unwrap();
        assert_eq!(lines, vec!["good line one", "good line two", "x"]);
        let asked = llm.asked.lock().unwrap();
        let (prompt, o) = &asked[0];
        assert_eq!(prompt, "Topic / instructions: cats and dogs\n\nWrite the 3 lines now.");
        assert!(o.system.starts_with("You produce training data"));
        assert!(o.system.contains("exactly 3 lines"));
        assert!(o.system.ends_with("about the topic requested."));
        assert_eq!(o.options, vec![("temperature".to_string(), Json::Num(0.9))]);
        drop(asked);
        corpus_from_prompt(&llm, "x", 2, "GARBAGE", "other").unwrap();
        let asked = llm.asked.lock().unwrap();
        assert!(asked[1].1.system.contains("deliberately WRONG"));
        assert_eq!(asked[1].1.model, "other");
        assert_eq!(asked[1].1.options, vec![("temperature".to_string(), Json::Num(1.1))]);
        drop(asked);
        assert!(matches!(
            corpus_from_prompt(&llm, " ", 3, "good", ""),
            Err(ReviewError::Invalid(_))
        ));
        assert!(matches!(
            corpus_from_prompt(&llm, "x", 0, "good", ""),
            Err(ReviewError::Invalid(_))
        ));
        let err = corpus_from_prompt(&llm, "x", 1, "weird", "").unwrap_err();
        assert_eq!(err.to_string(), "style must be one of good, garbage (got 'weird')");
    }

    #[test]
    fn answers_are_read_whatever_shape_they_came_in() {
        let raw = "{\"reviews\": [{\"index\": 0, \"rating\": 8, \"verdict\": \"pass\", \"critique\": \"ok\"}, \
                   {\"index\": 1, \"rating\": \"3.5\", \"critique\": \"\"}, {\"index\": 2, \"rating\": 42}, \
                   {\"index\": 7, \"rating\": 5}, {\"index\": \"x\", \"score\": 4, \"reason\": \"meh\"}, \"junk\", \
                   {\"index\": 3, \"rating\": \"n/a\"}]}";
        let parsed = parse_reviews(raw, 5);
        assert_eq!(parsed[&0], (8.0, "ok".to_string()));
        assert_eq!(parsed[&1], (3.5, "no critique given".to_string()));
        assert_eq!(parsed[&2].0, 10.0, "clamped");
        assert!(!parsed.contains_key(&7), "out of range");
        assert_eq!(
            parsed[&4],
            (4.0, "meh".to_string()),
            "the position stands in for a bad index"
        );
        assert!(!parsed.contains_key(&3), "an unusable rating");
        assert!(parse_reviews("garbage", 3).is_empty());
        assert_eq!(
            parse_reviews("[{\"rating\": 7}]", 1)[&0],
            (7.0, "no critique given".to_string())
        );
        assert_eq!(
            parse_reviews("{\"rating\": 2, \"critique\": \"bad\"}", 1)[&0],
            (2.0, "bad".to_string())
        );
        // a rating that is there but null does not fall back to the score
        assert!(parse_reviews("[{\"rating\": null, \"score\": 5}]", 1).is_empty());
    }

    #[test]
    fn a_review_follows_the_mark_not_the_verdict() {
        let llm = Scripted::reviewer();
        let reviews = review_texts(
            &llm,
            &texts(&["a good sentence", "zzzz", "   ", "another good one"]),
            "",
            "",
            6.0,
            DEFAULT_BATCH,
        )
        .unwrap();
        let verdicts: Vec<&str> = reviews.iter().map(|r| r.verdict.as_str()).collect();
        assert_eq!(verdicts, vec!["pass", "fail", "fail", "pass"]);
        let ratings: Vec<Option<f64>> = reviews.iter().map(|r| r.rating).collect();
        assert_eq!(ratings, vec![Some(9.0), Some(2.0), Some(0.0), Some(9.0)]);
        assert_eq!(reviews[2].critique, "empty output");
        let asked = llm.asked.lock().unwrap();
        let (prompt, o) = &asked[0];
        assert_eq!(
            prompt,
            "Review these 3 texts:\n[0] a good sentence\n[1] zzzz\n[3] another good one\n\nReturn the JSON now."
        );
        assert!(o.json);
        assert!(o.system.contains("for a rating of 6 or more"));
        drop(asked);
        let reviews = review_texts(&llm, &texts(&["a good sentence"]), " nursery rhymes ", "", 9.5, 20).unwrap();
        assert_eq!(reviews[0].verdict, "fail", "9 < 9.5");
        let asked = llm.asked.lock().unwrap();
        assert!(asked[1]
            .0
            .starts_with("Context: nursery rhymes\n\nReview these 1 texts:"));
        assert!(asked[1].1.system.contains("for a rating of 9.5 or more"));
    }

    #[test]
    fn batches_number_from_zero_and_the_unreadable_is_unrated() {
        let llm = Scripted::reviewer();
        let many: Vec<String> = (0..45).map(|i| format!("good {i}")).collect();
        let reviews = review_texts(&llm, &many, "", "", 6.0, 20).unwrap();
        assert_eq!(reviews.len(), 45);
        assert_eq!(llm.asked.lock().unwrap().len(), 3);
        assert!(reviews.iter().all(|r| r.verdict == "pass"));
        assert_eq!(reviews[44].index, 44);
        assert!(llm.asked.lock().unwrap()[2].0.contains("[0] good 40"));
        let silent = Scripted::new(|_, _| "I cannot rate these.".to_string());
        let reviews = review_texts(&silent, &texts(&["good one", "bad one"]), "", "", 6.0, 20).unwrap();
        assert!(reviews.iter().all(|r| r.verdict == "unrated" && r.rating.is_none()));
        assert_eq!(reviews[0].critique, "no review returned");
        assert!(review_texts(&silent, &[], "", "", 6.0, 0).is_err());
    }

    #[test]
    fn a_summary_splits_and_averages() {
        let llm = Scripted::reviewer();
        let o = ReviewOptions {
            texts: Some(texts(&["good one", "nonsense zzz", "good two"])),
            ..Default::default()
        };
        let result = adversarial_review(None, &llm, &o).unwrap();
        assert_eq!((result.source.as_str(), result.model.as_str()), ("given", "scripted"));
        assert_eq!(result.good, texts(&["good one", "good two"]));
        assert_eq!(result.bad, texts(&["nonsense zzz"]));
        assert!((result.mean_rating.unwrap() - 20.0 / 3.0).abs() < 1e-12);
        assert!((result.pass_rate.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        let doc = result.to_json();
        assert_eq!(
            doc.at("reviews").as_array()[0].render(0),
            "{\"index\":0,\"text\":\"good one\",\"rating\":9.0,\"verdict\":\"pass\",\"critique\":\"fine\"}"
        );
        assert!(doc.get("overall").is_none());
        assert!(adversarial_review(None, &llm, &ReviewOptions::default()).is_err());
    }

    #[test]
    fn the_model_is_sampled_for_its_own_review() {
        let mut model = Model::new(1, crate::GraphOptions::default()).unwrap();
        let corpus = texts(&["a good sentence about cats", "a good sentence about dogs"]);
        model
            .train(
                &corpus,
                &crate::TrainOptions {
                    epochs: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let llm = Scripted::reviewer();
        let o = ReviewOptions {
            count: 3,
            max_length: 40,
            seed: Some(3),
            ..Default::default()
        };
        let result = adversarial_review(Some(&mut model), &llm, &o).unwrap();
        assert_eq!(result.source, "model");
        assert_eq!(result.texts.len(), 3);
        let again = adversarial_review(Some(&mut model), &llm, &o).unwrap();
        assert_eq!(result.texts, again.texts, "a seed makes the samples reproducible");
        let prefixed = sample_texts(&mut model, 2, "a good", 30, 1.0, None).unwrap();
        assert!(prefixed.iter().all(|t| t.starts_with("a good")));
        assert!(sample_texts(&mut model, 0, "", 30, 1.0, None).is_err());
    }

    #[test]
    fn a_conversation_is_marked_reply_by_reply() {
        let llm = Scripted::new(|_, _| {
            "{\"reviews\": [{\"index\": 0, \"rating\": 8, \"critique\": \"follows on\"}], \
             \"overall\": {\"score\": 12, \"reason\": \"short\"}}"
                .to_string()
        });
        let exchanges = vec![
            ("hello there".to_string(), "hello there friend".to_string()),
            ("how are you".to_string(), " ".to_string()),
            ("and you".to_string(), "zzz".to_string()),
        ];
        let result = review_conversation(&llm, &exchanges, "greetings", "", 6.0).unwrap();
        assert_eq!(result.source, "chat");
        let verdicts: Vec<&str> = result.reviews.iter().map(|r| r.verdict.as_str()).collect();
        assert_eq!(verdicts, vec!["pass", "fail", "unrated"]);
        assert_eq!(result.reviews[1].critique, "it said nothing");
        assert_eq!(result.reviews[0].said.as_deref(), Some("hello there"));
        assert_eq!(
            result.overall,
            Some(Overall {
                rating: Some(10.0),
                critique: "short".to_string()
            })
        );
        let asked = llm.asked.lock().unwrap();
        assert_eq!(
            asked[0].0,
            "Topic: greetings\n\nThe conversation:\n[0] Partner: hello there\n    Model: hello there friend\n\n\
             [2] Partner: and you\n    Model: zzz\n\nReturn the JSON now."
        );
        assert!(asked[0].1.system.contains(" 6 out of 10 is a pass."));
        let doc = result.to_json();
        assert_eq!(doc.at("overall").at("critique").as_str(), Some("short"));
        assert_eq!(doc.at("reviews").as_array()[0].at("said").as_str(), Some("hello there"));
    }

    #[test]
    fn a_partner_line_is_one_plain_line() {
        assert_eq!(first_line("\n  Partner:  \"Hi there!\"  \nsecond"), "Hi there!");
        assert_eq!(first_line("You: Me: nested"), "nested");
        assert_eq!(
            first_line("Me: You: nested"),
            "You: nested",
            "each name is tried once, in order"
        );
        assert_eq!(first_line("\n\n"), "");
        let llm = Scripted::new(|_, _| "Model: what did you eat today?".to_string());
        let transcript = vec![
            ("Partner".to_string(), "hi".to_string()),
            ("Model".to_string(), "".to_string()),
        ];
        let line = chat_line(&llm, &transcript, "food", "a cook", "", 0.8).unwrap();
        assert_eq!(line, "what did you eat today?");
        let asked = llm.asked.lock().unwrap();
        assert_eq!(
            asked[0].0,
            "The conversation so far:\nPartner: hi\n\nWrite your next line."
        );
        assert!(asked[0]
            .1
            .system
            .ends_with(" The conversation is about food. You are a cook."));
        assert_eq!(asked[0].1.options, vec![("temperature".to_string(), Json::Num(0.8))]);
        drop(asked);
        chat_line(&llm, &[], "food", "", "", 0.5).unwrap();
        assert_eq!(
            llm.asked.lock().unwrap()[1].0,
            "Open the conversation about food with one short, plain line."
        );
    }

    #[test]
    fn failures_blame_and_passes_clear() {
        let reviews = vec![
            Review::marked(0, "the cat sat on the mat", 9.0, 6.0, "fine"),
            Review::marked(1, "the cat sat on the sky", 2.0, 6.0, "a false statement"),
            Review::unmarked(2, "zzz zzz", None, "unrated", "no review returned"),
            Review::unmarked(3, "", Some(0.0), "fail", "empty output"),
        ];
        let (faults, passed) = faults_from_reviews(&reviews, 6.0, "review");
        assert_eq!(passed, texts(&["the cat sat on the mat"]));
        let reasons: Vec<&str> = faults.iter().map(|f| f.reason.as_str()).collect();
        assert_eq!(reasons, vec!["false", "unrated"]);
        assert!((faults[0].severity - (0.25 + 1.75 * 4.0 / 6.0)).abs() < 1e-12);
        assert_eq!(faults[1].severity, 1.0);
        let mut negative = Model::new_negative(0, &Default::default()).unwrap();
        let report = teach_reviews(&mut negative, &reviews, 6.0, true, "critic", &TeachOptions::default()).unwrap();
        assert_eq!((report.blamed, report.passed, report.cleared), (2, 1, 1));
        assert_eq!(report.source, "critic");
        let doc = report.to_json();
        assert_eq!(doc.at("threshold").as_f64(), Some(6.0));
        assert_eq!(doc.at("faults").as_array().len(), 2);
        let rows = reason_rows(&negative);
        let names: Vec<&str> = rows.as_array().iter().filter_map(|r| r.at("reason").as_str()).collect();
        assert_eq!(names.len(), 2);
        assert!(rows.as_array().iter().all(|r| r.at("edges").as_i64().unwrap() > 0));
        let share: f64 = rows.as_array().iter().filter_map(|r| r.at("share").as_f64()).sum();
        assert!((share - 1.0).abs() < 1e-12);
    }
}
