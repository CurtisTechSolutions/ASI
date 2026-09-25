//! Thinking (`radixnet/thinking.py`): the fourth sentinel at work - what
//! makes the model think, what it thinks, and what happens when it stops.
//!
//! The graph's `THINK` sentinel faces both ways.  Its **in-edges** are where
//! the model has learned to stop and think: `p -> THINK` is taught by
//! experience, the way `BACK`'s edges are, whenever an *event* at `p` called
//! for a thought.  Its **out-edges** are how thoughts begin: a thought is a
//! text whose walk starts at `THINK` instead of `START`, observed from an
//! LLM's thinking the way texts are observed from a corpus ([`think_on`], fed
//! by `radixnet ollama think`).
//!
//! [`think`] is one thought.  It is triggered by an event - a voice that
//! caught itself repeating ([`crate::dialogue::backtrack`]), a question asked
//! of the model (`radixnet think --about ...`), a thought questioning itself -
//! and it does four things, in order: it teaches *where* it had to think
//! (`observe_think`, unless the thought is a question the model asked
//! itself); it thinks - the prediction search run from `THINK` to the end of a
//! text; it may question itself - along its own path, at a node where the
//! model has learned to think, a nested thought that must say something the
//! chain above it has not; and when it stops it triggers the sentinel the
//! event calls for: a thought thinking its way out of a repeat hands over to
//! `BACK` (`observe_back`), a question returns to the thought that asked, a
//! thought merely asked for ends.

use std::sync::Arc;

use crate::cli::Ctx;
use crate::graph::{BACK, FIRST, THINK};
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::kinds::TrainSettings;
use crate::model::{EpochRecord, Model, SearchTuning};
use crate::mt19937::Mt19937;
use crate::penalty::DEFAULT_TRAVERSAL;
use crate::search::PathResult;
use crate::service::Service;

/// What this module's own lines are filed under.
const LOG: &str = "think";

/// The trigger of a thought somebody asked for.
pub const ASKED: &str = "asked";
/// The trigger of a thought a thought asked itself.
pub const QUESTIONED: &str = "questioned";
/// The triggers of a thought a voice had on catching itself repeating.
pub const STUTTER: &str = "stutter";
pub const REPEAT: &str = "repeat";

/// What a thought triggers when it stops: the `BACK` sentinel, the end of
/// it, or the thought it questioned.
pub const THEN_BACK: &str = "back";
pub const THEN_END: &str = "end";
pub const THEN_THINK: &str = "think";
/// How a thought stopped: it reached `END`, it ran out of length, or it had
/// nothing (new) to think.
pub const STOPPED_END: &str = "end";
pub const STOPPED_LENGTH: &str = "length";
pub const STOPPED_NOTHING: &str = "nothing";

/// Units a thought may run to by default.
pub const THINK_LENGTH: usize = 60;
/// How deep a thought may question itself by default.
pub const THINK_DEPTH: usize = 2;
/// Questions one thought may ask itself by default.
pub const THINK_QUESTIONS: usize = 1;

/// Whether a thought with this trigger hands over to `BACK` when it stops: it
/// was thinking its way out of a repeat.
pub fn backs_up(trigger: &str) -> bool {
    trigger == STUTTER || trigger == REPEAT
}

/// One thought: what triggered it, what it thought, what it questioned and
/// what it triggered when it stopped (`thinking.Thought`).
#[derive(Clone, Debug, PartialEq)]
pub struct Thought {
    pub trigger: String,
    /// The node it thought at (-1: nowhere in particular).
    pub at: i64,
    /// The text it was thinking about, when there was one.
    pub about: String,
    /// The thought itself (`""` when it had nothing to think with).
    pub text: String,
    /// 0 for a thought, 1 for a question it asked itself, 2 for one about that.
    pub depth: usize,
    pub stopped: String,
    /// The sentinel it triggered when it stopped.
    pub then: String,
    /// The node taught to think here (`p -> THINK`), or -1.
    pub taught: i64,
    /// The node taught to hand over when it stopped (`p -> BACK`), or -1.
    pub handed_over: i64,
    pub cost: f64,
    pub probability: f64,
    /// Search states the thought weighed.
    pub expanded: usize,
    pub questions: Vec<Thought>,
    pub labels: Vec<String>,
    pub node_ids: Vec<usize>,
    pub step_costs: Vec<f64>,
}

impl Default for Thought {
    fn default() -> Thought {
        Thought {
            trigger: ASKED.to_string(),
            at: -1,
            about: String::new(),
            text: String::new(),
            depth: 0,
            stopped: STOPPED_NOTHING.to_string(),
            then: THEN_END.to_string(),
            taught: -1,
            handed_over: -1,
            cost: 0.0,
            probability: 1.0,
            expanded: 0,
            questions: Vec::new(),
            labels: Vec::new(),
            node_ids: Vec::new(),
            step_costs: Vec::new(),
        }
    }
}

impl Thought {
    /// How many times the thought questioned itself.
    pub fn questioned(&self) -> usize {
        self.questions.len()
    }

    /// The record as Python's `Thought.to_dict()` writes it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("trigger", Json::str(self.trigger.clone())),
            ("at", Json::Int(self.at)),
            ("about", Json::str(self.about.clone())),
            ("text", Json::str(self.text.clone())),
            ("depth", Json::Int(self.depth as i64)),
            ("stopped", Json::str(self.stopped.clone())),
            ("then", Json::str(self.then.clone())),
            ("taught", Json::Int(self.taught)),
            ("handed_over", Json::Int(self.handed_over)),
            ("cost", Json::Num(self.cost)),
            ("probability", Json::Num(self.probability)),
            ("expanded", Json::Int(self.expanded as i64)),
            ("questioned", Json::Int(self.questioned() as i64)),
            (
                "questions",
                Json::Arr(self.questions.iter().map(|q| q.to_json()).collect()),
            ),
            ("labels", Json::strs(self.labels.clone())),
            ("node_ids", Json::ints(self.node_ids.iter().map(|&n| n as i64))),
            ("step_costs", Json::nums(self.step_costs.clone())),
        ])
    }

    /// A thought read back from [`Thought::to_json`] (what a streamed turn
    /// carries), questions and all; `None` for anything but an object.
    pub fn from_json(doc: &Json) -> Option<Thought> {
        let Json::Obj(_) = doc else { return None };
        let text = |key: &str, fallback: &str| doc.at(key).as_str().unwrap_or(fallback).to_string();
        let int = |key: &str, fallback: i64| doc.at(key).as_i64().unwrap_or(fallback);
        let base = Thought::default();
        Some(Thought {
            trigger: text("trigger", &base.trigger),
            at: int("at", -1),
            about: text("about", ""),
            text: text("text", ""),
            depth: int("depth", 0).max(0) as usize,
            stopped: text("stopped", &base.stopped),
            then: text("then", &base.then),
            taught: int("taught", -1),
            handed_over: int("handed_over", -1),
            cost: doc.at("cost").as_f64().unwrap_or(0.0),
            probability: doc.at("probability").as_f64().unwrap_or(1.0),
            expanded: int("expanded", 0).max(0) as usize,
            questions: doc
                .at("questions")
                .as_array()
                .iter()
                .filter_map(Thought::from_json)
                .collect(),
            labels: doc
                .at("labels")
                .as_array()
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            node_ids: doc
                .at("node_ids")
                .as_array()
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n.max(0) as usize))
                .collect(),
            step_costs: doc
                .at("step_costs")
                .as_array()
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
        })
    }
}

/// How one [`think`] runs (`thinking.think`'s keyword arguments).
#[derive(Clone, Debug)]
pub struct ThinkOptions {
    /// Think at the node where this text ends (when `at` says nothing).
    pub about: String,
    /// The node the event happened at.
    pub at: Option<usize>,
    pub trigger: String,
    /// The step the walk was about to loop through, and the one it took
    /// instead - what the hand-over teaches when the thought stops.
    pub went: Option<usize>,
    pub instead: Option<usize>,
    /// `"beam"` (the most likely thought) or `"sample"` (a drawn one).
    pub mode: String,
    pub k: usize,
    pub beam: usize,
    pub max_length: usize,
    pub step_penalty: f64,
    pub temperature: f64,
    pub seed: Option<i64>,
    pub max_depth: usize,
    pub max_questions: usize,
    /// Write what the thought learned into the model.
    pub learn: bool,
    pub amount: f64,
}

impl Default for ThinkOptions {
    fn default() -> ThinkOptions {
        ThinkOptions {
            about: String::new(),
            at: None,
            trigger: ASKED.to_string(),
            went: None,
            instead: None,
            mode: "beam".to_string(),
            k: 5,
            beam: 0,
            max_length: THINK_LENGTH,
            step_penalty: 0.0,
            temperature: 1.0,
            seed: None,
            max_depth: THINK_DEPTH,
            max_questions: THINK_QUESTIONS,
            learn: true,
            amount: 1.0,
        }
    }
}

const TERMINATORS: &[u8] = b".!?";

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// The sentences of `text` that end in a question mark, each with the byte
/// index it starts at (`thinking.questions_in`).  A sentence runs from the
/// first character after the previous sentence's terminators (`.`, `!`, `?`)
/// to the end of its own; a run of terminators containing `?` makes it a
/// question.  `"Hmm. Is that right? Yes."` holds one, at index 5.
pub fn questions_in(text: &str) -> Vec<(usize, String)> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        while i < n && is_space(bytes[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        while i < n && !TERMINATORS.contains(&bytes[i]) {
            i += 1;
        }
        let mut j = i;
        while j < n && TERMINATORS.contains(&bytes[j]) {
            j += 1;
        }
        if i > start && bytes[i..j].contains(&b'?') {
            out.push((start, text[start..j].trim().to_string()));
        }
        i = j;
    }
    out
}

/// The node the walk of `prefix` ends at, cut so that it ends there, or `-1`
/// when the graph cannot place it (`thinking.place`).  The last gram of the
/// prefix is located; when it sits inside a compressed node the node is split
/// after it, so that an edge taught from the node fires exactly where the
/// prefix ends.
pub fn place(model: &mut Model, prefix: &str) -> Result<i64, String> {
    let (node, offset, lead) = model.prefix_start(prefix);
    if node < FIRST || !lead.is_empty() || !model.g.is_alive(node) {
        return Ok(-1);
    }
    let enc = model.g.enc;
    if offset + enc.n < model.g.label_len(node) {
        model.g.split(node, offset + enc.stride)?; // the gram becomes the last of its node
    }
    Ok(node as i64)
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Up to `k` thoughts from `THINK`, most likely first (beam), or `k` walks
/// (sample), and the states weighed.
fn candidates(
    model: &mut Model,
    o: &ThinkOptions,
    rng: &mut Option<Mt19937>,
) -> Result<(Vec<PathResult>, usize), String> {
    let tuning = SearchTuning {
        origin: THINK,
        ..Default::default()
    };
    if o.mode == "sample" {
        let mut drawn = Vec::new();
        let mut expanded = 0;
        for _ in 0..o.k.max(1) {
            let walk = model.search(
                "",
                o.max_length,
                "sample",
                0,
                0,
                0.0,
                o.temperature,
                false,
                Some(o.max_length),
                rng.as_mut(),
                DEFAULT_TRAVERSAL,
                1.0,
                1.0,
                tuning,
            )?;
            expanded += walk.expanded;
            drawn.push(walk.best);
        }
        return Ok((drawn, expanded));
    }
    let found = model.search(
        "",
        0,
        "beam",
        o.k,
        o.beam,
        o.step_penalty,
        o.temperature,
        true,
        Some(o.max_length),
        None,
        DEFAULT_TRAVERSAL,
        1.0,
        1.0,
        tuning,
    )?;
    Ok((found.top, found.expanded))
}

/// One thought (`thinking.think`): triggered by `o.trigger` at `o.at`, or at
/// the end of `o.about`.  See the module documentation for the four steps.
/// `rng` is the private generator of a sampled thought (`o.seed` starts one
/// when there is none).
pub fn think(model: &mut Model, o: &ThinkOptions, rng: &mut Option<Mt19937>) -> Result<Thought, String> {
    let mut o = o.clone();
    o.mode = match o.mode.to_ascii_lowercase().as_str() {
        "" | "beam" | "dijkstra" => "beam".to_string(),
        "sample" => "sample".to_string(),
        other => return Err(format!("unknown mode {other:?}; expected 'beam' or 'sample'")),
    };
    if o.k < 1 {
        return Err(format!("k must be >= 1, got {}", o.k));
    }
    if o.temperature < 0.0 {
        return Err("temperature must be >= 0".to_string());
    }
    if o.amount < 0.0 {
        return Err(format!("amount must be >= 0, got {}", o.amount));
    }
    if rng.is_none() {
        if let Some(seed) = o.seed {
            *rng = Some(Mt19937::new(seed));
        }
    }
    let empty: Vec<String> = Vec::new();
    think_at_depth(model, &o, rng, 0, &empty)
}

fn think_at_depth(
    model: &mut Model,
    o: &ThinkOptions,
    rng: &mut Option<Mt19937>,
    depth: usize,
    thought_so_far: &[String],
) -> Result<Thought, String> {
    let mut at: i64 = match o.at {
        Some(node) => node as i64,
        None => -1,
    };
    if at < 0 && !o.about.trim().is_empty() {
        at = place(model, &o.about)?;
    }
    if at >= 0 && (at < FIRST as i64 || at as usize >= model.g.num_node_ids() || !model.g.is_alive(at as usize)) {
        return Err(format!("node {at} is not a real node to think at"));
    }
    let mut record = Thought {
        trigger: o.trigger.clone(),
        at,
        about: o.about.clone(),
        depth,
        ..Default::default()
    };

    // 1. the event teaches where to think - from experience, as BACK is taught
    if o.learn && at >= FIRST as i64 && o.trigger != QUESTIONED {
        model.g.observe_think(at as usize, o.amount)?;
        record.taught = at;
    }

    // 2. the thought: the search from THINK, in the language of the thoughts it was taught
    let (candidates, expanded) = candidates(model, o, rng)?;
    record.expanded = expanded;
    let chosen = candidates
        .into_iter()
        .find(|cand| !cand.text.trim().is_empty() && !thought_so_far.contains(&normalize(&cand.text)));
    if let Some(chosen) = chosen {
        record.text = chosen.text.clone();
        record.cost = chosen.cost;
        record.probability = chosen.probability();
        record.labels = chosen.labels.clone();
        record.node_ids = chosen.node_ids.clone();
        record.step_costs = chosen.step_costs.clone();
        record.stopped = if chosen.reached_end {
            STOPPED_END
        } else {
            STOPPED_LENGTH
        }
        .to_string();

        // 3. questioning itself: where, along its own path, the model has learned to think
        if depth < o.max_depth && o.max_questions > 0 {
            let mut so_far: Vec<String> = thought_so_far.to_vec();
            so_far.push(normalize(&record.text));
            let path = record.node_ids.clone();
            for node in path {
                if node < FIRST || !model.g.thinks_at(node) {
                    continue;
                }
                let asked = ThinkOptions {
                    about: String::new(),
                    at: Some(node),
                    trigger: QUESTIONED.to_string(),
                    went: None,
                    instead: None,
                    ..o.clone()
                };
                let question = think_at_depth(model, &asked, rng, depth + 1, &so_far)?;
                so_far.push(normalize(&question.text));
                record.questions.push(question);
                if record.questions.len() >= o.max_questions {
                    break;
                }
            }
        }
    }

    // 4. when thinking stops, it triggers the sentinel the event calls for
    if o.trigger == QUESTIONED {
        record.then = THEN_THINK.to_string();
    } else if backs_up(&o.trigger) && at >= FIRST as i64 {
        record.then = THEN_BACK.to_string();
        if o.learn {
            let p = at as usize;
            let went = o.went.filter(|&w| w != BACK && model.g.edge(p, w).is_some());
            let instead = o.instead.filter(|&i| model.g.edge(p, i).is_some());
            model.g.observe_back(p, went, instead, o.amount)?;
            record.handed_over = at;
        }
    } else {
        record.then = THEN_END.to_string();
    }
    Ok(record)
}

/// What [`think_on`] taught: the thoughts, the questions they asked
/// themselves, the nodes taught to stop and think, and the epoch records.
#[derive(Clone, Debug, Default)]
pub struct Learned {
    pub thoughts: usize,
    pub questions: usize,
    pub taught: usize,
    pub epochs: Vec<EpochRecord>,
}

impl Learned {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("thoughts", Json::Int(self.thoughts as i64)),
            ("questions", Json::Int(self.questions as i64)),
            ("taught", Json::Int(self.taught as i64)),
            ("epochs", Json::Arr(self.epochs.iter().map(|r| r.to_json()).collect())),
        ])
    }
}

/// Teaches the model thoughts (`thinking.think_on`): texts whose walk begins
/// at `THINK`, trained the way this model trains texts (`settings`, its origin
/// overridden).  With `questions`, every sentence ending in `?` marks the node
/// before it as a place the model stops to think (`observe_think` by
/// `amount`, on the node cut to end exactly there) and is trained as a thought
/// of its own.
pub fn think_on(
    model: &mut Model,
    thoughts: &[String],
    settings: &TrainSettings,
    questions: bool,
    amount: f64,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Learned, String> {
    if model.is_negative() {
        return Err("the negative network judges; it does not think".to_string());
    }
    let cleaned: Vec<String> = thoughts
        .iter()
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|t| !t.is_empty())
        .collect();
    let settings = TrainSettings {
        config: crate::radix::TrainConfig {
            origin: THINK,
            ..settings.config.clone()
        },
        ..settings.clone()
    };
    let mut learned = Learned {
        thoughts: cleaned.len(),
        ..Default::default()
    };
    learned.epochs = crate::kinds::train(model, &cleaned, &settings, on_epoch)?;
    if questions && !cleaned.is_empty() {
        let mut asked: Vec<String> = Vec::new();
        for t in &cleaned {
            for (start, question) in questions_in(t) {
                let node = place(model, &t[..start])?;
                if node >= FIRST as i64 {
                    model.g.observe_think(node as usize, amount)?;
                    learned.taught += 1;
                }
                asked.push(question);
            }
        }
        // a question is a thought of its own: the model may begin a thought with it
        let enc = model.g.enc;
        asked.retain(|q| enc.len(q) >= enc.n);
        learned.questions = asked.len();
        if !asked.is_empty() {
            learned
                .epochs
                .extend(crate::kinds::train(model, &asked, &settings, on_epoch)?);
        }
    }
    crate::log_info!(
        LOG,
        "taught {} thought(s), {} question(s); {} node(s) now stop to think",
        learned.thoughts,
        learned.questions,
        learned.taught
    );
    Ok(learned)
}

/// One line saying what a thought did, for transcripts (`thinking.summarize`).
pub fn summarize(thought: &Thought) -> String {
    let said = if !thought.text.is_empty() {
        format!("thought \"{}\"", thought.text.replace('"', "\\\""))
    } else if thought.stopped == STOPPED_NOTHING && thought.depth > 0 {
        "had nothing new to think".to_string()
    } else {
        "had nothing to think with yet".to_string()
    };
    let mut parts = vec![said];
    if !thought.questions.is_empty() {
        let times = if thought.questioned() == 1 {
            "once".to_string()
        } else {
            format!("{} times", thought.questioned())
        };
        parts.push(format!("questioned itself {times}"));
    }
    parts.push(match thought.then.as_str() {
        THEN_BACK => "then backed up".to_string(),
        THEN_THINK => "then went back to the thought".to_string(),
        THEN_END => "then went on".to_string(),
        other => format!("then {other}"),
    });
    parts.join("; ")
}

// -- the command line -----------------------------------------------------------------------------

/// `radixnet think`: one thought from the THINK sentinel, questioning itself
/// where the model learned to (Python's `cmd_think`).
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let mut model = ctx.open(true)?;
    let o = ThinkOptions {
        about: args.str("about", ""),
        mode: args.str("mode", "beam"),
        k: args.usize("k", 5)?,
        beam: args.usize("beam", 0)?,
        max_length: args.usize("max-length", THINK_LENGTH)?,
        step_penalty: args.float("step-penalty", 0.0)?,
        temperature: args.float("temperature", 1.0)?,
        seed: args.get("seed").map(|_| ctx.seed),
        max_depth: args.usize("depth", THINK_DEPTH)?,
        max_questions: args.usize("questions", THINK_QUESTIONS)?,
        learn: !args.on("no-learn"),
        ..Default::default()
    };
    let thought = think(&mut model, &o, &mut None)?;
    let learned = thought.taught >= 0 || thought.handed_over >= 0;
    let mut doc = vec![("kind".to_string(), Json::str(model.kind()))];
    if let Json::Obj(pairs) = thought.to_json() {
        doc.extend(pairs);
    }
    let saved = if learned && args.on("save") {
        let path = ctx.save(&mut model)?;
        crate::llm::saved(&path)
    } else {
        Json::Null
    };
    doc.push(("saved".to_string(), saved));
    doc.push(("summary".to_string(), Json::str(summarize(&thought))));
    ctx.emit(Json::Obj(doc));
    Ok(())
}

// -- the route ------------------------------------------------------------------------------------

/// `POST /api/think`: the model thinks (Python's `ModelService.think`).
fn think_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let o = ThinkOptions {
        about: r.text("about", ""),
        mode: r.text("mode", "beam"),
        k: r.usize("k", 5)?,
        beam: r.usize("beam", 0)?,
        max_length: r.usize("max_length", THINK_LENGTH)?,
        temperature: r.number("temperature", 1.0)?,
        step_penalty: r.number("step_penalty", 0.0)?,
        seed: r.body.get("seed").and_then(|v| v.as_i64()),
        max_depth: r.usize("depth", THINK_DEPTH)?,
        max_questions: r.usize("questions", THINK_QUESTIONS)?,
        learn: r.flag("learn", true),
        ..Default::default()
    };
    if o.k < 1 {
        return Err(ApiError::bad_request("'k' must be >= 1"));
    }
    let (kind, thought) = svc
        .with_model(|m| -> Result<(&'static str, Thought), String> {
            let thought = think(m, &o, &mut None)?;
            Ok((m.kind(), thought))
        })
        .map_err(ApiError::bad_request)?;
    let mut doc = vec![("kind".to_string(), Json::str(kind))];
    if let Json::Obj(pairs) = thought.to_json() {
        doc.extend(pairs);
    }
    Ok(Json::Obj(doc))
}

pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/think", think_route);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::START;
    use crate::model::TrainOptions;
    use crate::GraphOptions;

    fn texts(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|s| s.to_string()).collect()
    }

    fn trained(kind: &str) -> Model {
        let mut m = crate::kinds::new_model(kind, 3, crate::encoding::Encoding::default(), &[]).unwrap();
        m.workers = 1;
        m.g.workers = 1;
        let corpus = texts(&[
            "ha ha ha ha ha",
            "ha ha ho ho hum",
            "ha ha and then the cat sat",
            "the cat sat on the mat",
        ]);
        crate::kinds::train_at(&mut m, &corpus, 3, 1.0, 1).unwrap();
        m
    }

    fn settings() -> TrainSettings {
        TrainSettings {
            config: crate::radix::TrainConfig {
                epochs: 3,
                lr: 1.0,
                batch_size: 1,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn questions_are_the_sentences_ending_in_a_question_mark() {
        assert_eq!(
            questions_in("Hmm. Is that right? Yes. Why not?"),
            vec![(5, "Is that right?".to_string()), (25, "Why not?".to_string())]
        );
        assert_eq!(questions_in("no questions here."), vec![]);
        assert_eq!(questions_in("  really?!  "), vec![(2, "really?!".to_string())]);
        assert_eq!(
            questions_in("a? b. c?"),
            vec![(0, "a?".to_string()), (6, "c?".to_string())]
        );
        assert_eq!(questions_in(""), vec![]);
        assert_eq!(questions_in("???"), vec![]);
        assert_eq!(
            questions_in("hmm, what did the cat do? it sat."),
            vec![(0, "hmm, what did the cat do?".to_string())]
        );
    }

    #[test]
    fn a_model_taught_no_thoughts_has_nothing_to_think_with() {
        let mut m = trained("count");
        let thought = think(&mut m, &ThinkOptions::default(), &mut None).unwrap();
        assert_eq!(
            (thought.trigger.as_str(), thought.at, thought.text.as_str()),
            (ASKED, -1, "")
        );
        assert_eq!(
            (thought.stopped.as_str(), thought.then.as_str()),
            (STOPPED_NOTHING, THEN_END)
        );
        assert_eq!((thought.taught, thought.handed_over, thought.questioned()), (-1, -1, 0));
        assert!(summarize(&thought).contains("nothing to think with"));
    }

    #[test]
    fn asking_about_a_text_teaches_the_model_to_think_there() {
        for kind in ["count", "radix", "resonant"] {
            let mut m = trained(kind);
            let o = ThinkOptions {
                about: "the cat".to_string(),
                ..Default::default()
            };
            let thought = think(&mut m, &o, &mut None).unwrap();
            assert!(thought.at >= FIRST as i64, "{kind}");
            assert_eq!(thought.taught, thought.at);
            assert!(m.g.edge(thought.at as usize, THINK).is_some());
            assert_eq!(thought.then, THEN_END);
            assert_eq!(thought.handed_over, -1);
            assert!(m.g.think_cost(thought.at as usize).is_some());
            m.g.check_invariants(&[], false).unwrap();
            // read-only: nothing written
            let mut quiet = trained(kind);
            let o = ThinkOptions {
                about: "the cat".to_string(),
                learn: false,
                ..Default::default()
            };
            let thought = think(&mut quiet, &o, &mut None).unwrap();
            assert_eq!(thought.taught, -1);
            assert!(quiet.g.parents_of(THINK).is_empty());
        }
    }

    #[test]
    fn think_on_teaches_thoughts_and_where_they_question_themselves() {
        let thoughts = texts(&[
            "okay, the cat sat. is that right? yes it did.",
            "the sky is blue because of light",
            "hmm. what did the cat do? it sat on the mat.",
        ]);
        for kind in ["count", "radix", "resonant"] {
            let mut m = trained(kind);
            let learned = think_on(&mut m, &thoughts, &settings(), true, 1.0, &mut |_| true).unwrap();
            assert_eq!(
                (learned.thoughts, learned.questions, learned.taught),
                (3, 2, 2),
                "{kind}"
            );
            assert!(!learned.epochs.is_empty());
            assert!(m.g.degree(THINK) > 0);
            for t in &thoughts {
                let grams = m.g.enc.encode(t);
                assert!(m.g.node_path_from(THINK, &grams).is_some(), "{kind}: {t}");
                if !t.starts_with("the ") {
                    assert!(m.g.node_path(&grams).is_none(), "{kind}: {t} begins at START");
                }
            }
            m.g.check_invariants(&[], false).unwrap();
            let o = ThinkOptions {
                learn: false,
                ..Default::default()
            };
            let thought = think(&mut m, &o, &mut None).unwrap();
            assert!(!thought.text.is_empty(), "{kind}");
            assert_eq!(thought.node_ids[0], THINK);
            assert!(!thought.node_ids.contains(&START));
            let sampled = think(
                &mut m,
                &ThinkOptions {
                    mode: "sample".to_string(),
                    seed: Some(1),
                    learn: false,
                    ..Default::default()
                },
                &mut None,
            )
            .unwrap();
            assert!(!sampled.text.is_empty(), "{kind}");
        }
    }

    #[test]
    fn a_thought_questions_itself_where_the_model_learned_to_think() {
        for kind in ["count", "radix", "resonant"] {
            let mut m = trained(kind);
            let thoughts = texts(&["the sky is blue because of light", "why is that so?"]);
            think_on(&mut m, &thoughts, &settings(), false, 1.0, &mut |_| true).unwrap();
            let quiet = ThinkOptions {
                learn: false,
                ..Default::default()
            };
            let mut thought = think(&mut m, &quiet, &mut None).unwrap();
            let mut node: i64 = -1;
            for _round in 0..4 {
                let first = thought
                    .node_ids
                    .iter()
                    .copied()
                    .find(|&n| n >= FIRST)
                    .expect("a real node");
                while !m.g.thinks_at(first) {
                    m.g.observe_think(first, 1.0).unwrap();
                }
                thought = think(&mut m, &quiet, &mut None).unwrap();
                if let Some(&n) = thought.node_ids.iter().find(|&&n| n >= FIRST && m.g.thinks_at(n)) {
                    node = n as i64;
                    break;
                }
            }
            assert!(
                node >= FIRST as i64,
                "{kind}: no thought passed a node it learned to think at"
            );
            assert_eq!(thought.questioned(), 1, "{kind}");
            let question = &thought.questions[0];
            assert_eq!(
                (question.trigger.as_str(), question.at, question.depth),
                (QUESTIONED, node, 1)
            );
            assert_eq!((question.then.as_str(), question.taught), (THEN_THINK, -1));
            assert_ne!(normalize(&question.text), normalize(&thought.text));
            assert!(summarize(&thought).contains("questioned itself once"));
            let shallow = think(
                &mut m,
                &ThinkOptions {
                    learn: false,
                    max_depth: 0,
                    ..Default::default()
                },
                &mut None,
            )
            .unwrap();
            assert_eq!(shallow.questioned(), 0);
        }
    }

    #[test]
    fn a_thought_out_of_a_repeat_hands_over_to_back_when_it_stops() {
        let mut m = trained("count");
        let node = m.g.lookup("ha ").map(|l| l.node).unwrap();
        let went = m.g.children_pairs(node)[0].0;
        let o = ThinkOptions {
            at: Some(node),
            trigger: STUTTER.to_string(),
            went: Some(went),
            ..Default::default()
        };
        let thought = think(&mut m, &o, &mut None).unwrap();
        assert_eq!(thought.then, THEN_BACK);
        assert_eq!((thought.taught, thought.handed_over), (node as i64, node as i64));
        assert!(m.g.edge(node, THINK).is_some() && m.g.edge(node, BACK).is_some());
        assert!(m.g.back_cost(node).is_some());
        assert!(summarize(&thought).contains("then backed up"));
        let mut fresh = trained("count");
        let o = ThinkOptions {
            at: Some(node),
            trigger: STUTTER.to_string(),
            learn: false,
            ..Default::default()
        };
        let quiet = think(&mut fresh, &o, &mut None).unwrap();
        assert_eq!(
            (quiet.then.as_str(), quiet.taught, quiet.handed_over),
            (THEN_BACK, -1, -1)
        );
        assert!(fresh.g.parents_of(BACK).is_empty() && fresh.g.parents_of(THINK).is_empty());
    }

    #[test]
    fn validation_and_the_negative_network() {
        let mut m = trained("count");
        assert!(think(
            &mut m,
            &ThinkOptions {
                mode: "walk".to_string(),
                ..Default::default()
            },
            &mut None
        )
        .is_err());
        assert!(think(
            &mut m,
            &ThinkOptions {
                at: Some(START),
                ..Default::default()
            },
            &mut None
        )
        .is_err());
        assert!(think(
            &mut m,
            &ThinkOptions {
                k: 0,
                ..Default::default()
            },
            &mut None
        )
        .is_err());
        let mut negative = Model::new_negative(1, &crate::negative::NegativeOptions::default()).unwrap();
        assert!(think_on(
            &mut negative,
            &texts(&["a thought"]),
            &settings(),
            true,
            1.0,
            &mut |_| true
        )
        .is_err());
        let _ = TrainOptions::default();
        let _ = GraphOptions::default();
    }

    #[test]
    fn the_model_file_keeps_its_thoughts() {
        let mut m = trained("count");
        let thoughts = texts(&[
            "okay, the cat sat. is that right? yes it did.",
            "the sky is blue because of light",
        ]);
        think_on(&mut m, &thoughts, &settings(), true, 1.0, &mut |_| true).unwrap();
        let quiet = ThinkOptions {
            learn: false,
            ..Default::default()
        };
        let before = think(&mut m, &quiet, &mut None).unwrap();
        let doc = m.to_doc();
        let mut back = Model::from_doc(&doc).unwrap();
        back.g.check_invariants(&[], false).unwrap();
        let after = think(&mut back, &quiet, &mut None).unwrap();
        assert_eq!(after.text, before.text);
        assert_eq!(after.labels, before.labels);
        assert!((after.cost - before.cost).abs() < 1e-9);
    }
}
