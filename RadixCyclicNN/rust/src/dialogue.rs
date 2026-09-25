//! The model converses with itself (`radixnet/dialogue.py`,
//! `go/radixnet/dialogue.go`).
//!
//! Every reply is the prediction search picking up the tail of the previous
//! line: the tail is located in the graph and continued, the context loses a
//! word at a time while nothing follows it, and a voice with nothing left to
//! add changes the subject with a fresh text from START.  A conversation is a
//! loop over [`reply`], which is public because the other voice need not be
//! a model at all - the chat loop has an LLM speak every other line.
//!
//! Two things keep a conversation from going round in circles.  A candidate
//! that repeats what the conversation has heard (or its own words, twice in
//! a row: a *stutter*) is skipped, and the best of them is only spoken, and
//! flagged a repeat, when nothing else is left.  And a voice that catches
//! itself repeating *backs up* ([`backtrack`], D-062): it keeps what it said
//! up to the repetition and looks for another way on from there, and - with
//! `learn` on - teaches the graph where its walks go round
//! ([`teach_back`], `Graph::observe_back`), so the model itself learns it.
//!
//! The negative network guards every turn when the server or the CLI has one
//! ([`converse_guarded`]): a reply it vetoes is left unsaid, exactly like one
//! that had been said before, except that it may not even be the fallback.
//!
//! A conversation can be **streamed** as it happens (a [`Stream`]).  The
//! stream has two layers.  `turn` events are the response: a turn is spoken
//! once and never taken back, so they can be appended to a transcript as they
//! arrive.  Everything between two of them is the *window* - what the voice
//! is doing before it commits, and what a backtrack may still rewrite: `look`,
//! `draft`, `caught`, `backtrack`, `found` or `stuck`.  Streaming changes
//! nothing about what is said: the turns are the ones [`converse`] returns,
//! event for event, and the events are the ones Python and Go write.

use std::collections::HashSet;
use std::sync::Arc;

use crate::cli::Ctx;
use crate::duo::{guard_report, Filter, FilterVerdict};
use crate::graph::{FIRST, START};
use crate::http::{Answer, ApiError, Request, Server, Sink};
use crate::json::Json;
use crate::model::{Model, PredictOptions};
use crate::mt19937::Mt19937;
use crate::penalty::DEFAULT_TRAVERSAL;
use crate::search::PathResult;
use crate::service::Service;
use crate::thinking::{think, ThinkOptions, Thought, THINK_DEPTH, THINK_QUESTIONS};

/// The two voices of a conversation.
pub const DEFAULT_SPEAKERS: [&str; 2] = ["A", "B"];

/// How many times a voice may back up out of a repeat by default (0 turns
/// the exploring off).
pub const EXPLORE: usize = 3;

/// How long a run of words may be for its immediate repetition to count as
/// a stutter.
pub const LONGEST_STUTTER: usize = 4;

/// Where a conversation streams what it is doing, one event at a time, as it
/// happens (Python's `radixnet.dialogue.StreamFn`).
///
/// Every event carries `event` (its kind), `index` and `speaker` (whose turn
/// it is).  The committed layer is `turn` (`turn`: the [`Turn`] as
/// [`Turn::to_json`] writes it) - a turn is spoken once and never taken back,
/// so the turn events are the answer.  The window between two turns is what a
/// backtrack may still rewrite: `look` (`from`: the context it continues, `""`
/// for a fresh text from START), `draft` (`text`, `cost`: what it was about to
/// say before it caught itself), `caught` (`kind`, `noticed`, `cut`: what it
/// keeps, `""` when it cannot back up), `backtrack` (`step`, `cut`, `wider`),
/// `found` (`text`, `cost`, `explored`) or `stuck` (`explored`: nothing new
/// from any cut).
pub type Stream<'a> = dyn FnMut(Json) + 'a;

/// Every kind of event a streamed conversation emits, in the order a turn goes
/// through them.
pub const STREAM_EVENTS: [&str; 7] = ["look", "draft", "caught", "backtrack", "found", "stuck", "turn"];

/// An event with the turn's `index` and `speaker` written after its kind.
fn tagged(event: Json, index: usize, speaker: &str) -> Json {
    let Json::Obj(pairs) = event else { return event };
    let mut out: Vec<(String, Json)> = Vec::with_capacity(pairs.len() + 2);
    for (key, value) in pairs {
        out.push((key, value));
        if out.len() == 1 {
            out.push(("index".to_string(), Json::Int(index as i64)));
            out.push(("speaker".to_string(), Json::str(speaker)));
        }
    }
    Json::Obj(out)
}

/// The committed layer of the stream: a turn that has been spoken, and will
/// not be taken back.
fn spoken(stream: &mut Option<&mut Stream>, turn: &Turn) {
    if let Some(s) = stream.as_mut() {
        s(Json::obj([
            ("event", Json::str("turn")),
            ("index", Json::Int(turn.index as i64)),
            ("speaker", Json::str(turn.speaker.clone())),
            ("turn", turn.to_json()),
        ]));
    }
}

/// A voice catching itself repeating, and what it did about it.
///
/// `kind` is what it caught: `"stutter"` - its own words, twice in a row - or
/// `"repeat"`, something the conversation had already heard.  `noticed` is
/// the words themselves, `cut` what it kept of that attempt, `steps` how many
/// times it backed up, `explored` the paths it weighed from there and `found`
/// whether one of them said something new.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rethink {
    pub kind: String,
    pub noticed: String,
    pub cut: String,
    pub steps: usize,
    pub explored: usize,
    pub found: bool,
    /// The node it taught to hand over at, or -1 when it taught nothing.
    pub taught: i64,
    /// What it thought before backing up ([`crate::thinking::think`]).
    pub thought: Option<Thought>,
}

impl Rethink {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("kind", Json::str(self.kind.clone())),
            ("noticed", Json::str(self.noticed.clone())),
            ("cut", Json::str(self.cut.clone())),
            ("steps", Json::Int(self.steps as i64)),
            ("explored", Json::Int(self.explored as i64)),
            ("found", Json::Bool(self.found)),
            ("taught", Json::Int(self.taught)),
            (
                "thought",
                self.thought.as_ref().map(|t| t.to_json()).unwrap_or(Json::Null),
            ),
        ])
    }
}

/// One utterance of a conversation.
#[derive(Clone, Debug, Default)]
pub struct Turn {
    pub index: usize,
    pub speaker: String,
    pub text: String,
    /// The tail of the previous line this reply picked up (`""` for a fresh start).
    pub context: String,
    /// What the search added after the context.
    pub reply: String,
    pub cost: f64,
    pub probability: f64,
    pub reached_end: bool,
    /// Spoken from START: the opening, or nothing followed the previous line.
    pub fresh: bool,
    /// Supplied by the caller (the opening), not generated.
    pub given: bool,
    /// Every candidate repeated something; the best one was spoken anyway.
    pub repeat: bool,
    /// The utterance says the same run of words twice in a row.
    pub stutter: bool,
    pub rethink: Option<Rethink>,
    /// Continuations the search offered for this turn.
    pub candidates: usize,
    /// Candidates rejected (empty, or already said) before the spoken one.
    pub skipped: usize,
    /// Candidates the guard refused for this turn.
    pub vetoed: usize,
    pub labels: Vec<String>,
    pub node_ids: Vec<usize>,
    pub step_costs: Vec<f64>,
}

impl Turn {
    /// The turn as Python's `Turn.to_dict` writes it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("index", Json::Int(self.index as i64)),
            ("speaker", Json::str(self.speaker.clone())),
            ("text", Json::str(self.text.clone())),
            ("context", Json::str(self.context.clone())),
            ("reply", Json::str(self.reply.clone())),
            ("cost", Json::Num(self.cost)),
            ("probability", Json::Num(self.probability)),
            ("reached_end", Json::Bool(self.reached_end)),
            ("fresh", Json::Bool(self.fresh)),
            ("given", Json::Bool(self.given)),
            ("repeat", Json::Bool(self.repeat)),
            ("stutter", Json::Bool(self.stutter)),
            (
                "rethink",
                self.rethink.as_ref().map(|r| r.to_json()).unwrap_or(Json::Null),
            ),
            ("candidates", Json::Int(self.candidates as i64)),
            ("skipped", Json::Int(self.skipped as i64)),
            ("vetoed", Json::Int(self.vetoed as i64)),
            ("labels", Json::strs(self.labels.clone())),
            ("node_ids", Json::ints(self.node_ids.iter().map(|&n| n as i64))),
            ("step_costs", Json::nums(self.step_costs.clone())),
        ])
    }
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// The last `chars` characters of `text`, cut forward to a word boundary when
/// a word straddles the cut.
pub fn tail_context(text: &str, chars: usize) -> String {
    let text = text.trim_end();
    if chars == 0 || text.is_empty() {
        return String::new();
    }
    let count = text.chars().count();
    if count <= chars {
        return text.trim_start().to_string();
    }
    let start = text.char_indices().nth(count - chars).map(|(i, _)| i).unwrap_or(0);
    let mut tail = &text[start..];
    if let Some(cut) = tail.find(' ') {
        if !tail[cut + 1..].trim().is_empty() {
            tail = &tail[cut + 1..];
        }
    }
    tail.trim_start().to_string()
}

/// The key two utterances are compared by.
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// The words an utterance says twice in a row (a run of one to `longest`
/// words repeated immediately after itself - "the the west", "the cat the cat
/// sat"), or `""` when it says each thing once.
pub fn stutter(text: &str, longest: usize) -> String {
    caught(text, longest).0
}

/// Where a stutter starts saying itself again (a byte index into `text`):
/// where the walk went round, and so where a voice backing out keeps to.
pub fn stutter_at(text: &str, longest: usize) -> Option<usize> {
    caught(text, longest).1
}

/// The words of `text` with where each starts.
fn words_at(text: &str) -> Vec<(String, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if is_space(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !is_space(bytes[i]) {
            i += 1;
        }
        out.push((text[start..i].to_lowercase(), start));
    }
    out
}

/// The run said twice in a row, and where `text` starts saying it again.
fn caught(text: &str, longest: usize) -> (String, Option<usize>) {
    let words = words_at(text);
    for i in 0..words.len() {
        let limit = ((words.len() - i) / 2).min(longest);
        for run in 1..=limit {
            if (0..run).all(|j| words[i + j].0 == words[i + run + j].0) {
                let said: Vec<&str> = words[i..i + run].iter().map(|(w, _)| w.as_str()).collect();
                return (said.join(" "), Some(words[i + run].1));
            }
        }
    }
    (String::new(), None)
}

/// Where the last word of `text` starts, or `None` when it has fewer than
/// two: where a voice repeating a whole utterance has to differ.
fn last_word_at(text: &str) -> Option<usize> {
    let words = words_at(text);
    if words.len() < 2 {
        return None;
    }
    words.last().map(|(_, at)| *at)
}

/// `speaker: text` lines.
pub fn transcript(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| format!("{}: {}", t.speaker, t.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The duplicates a conversation could not avoid: the utterances of the turns
/// flagged repeat, each once - the texts to punish, so the model stops
/// offering them.
pub fn repeats(turns: &[Turn]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in turns {
        let key = normalize(&t.text);
        if t.repeat && !key.is_empty() && seen.insert(key) {
            out.push(t.text.clone());
        }
    }
    out
}

/// What a conversation has already heard, and so what counts as a duplicate:
/// said before, a reply adding what an earlier reply already added, or an
/// echo of a line already spoken (the whole utterance inside one of them).
#[derive(Clone, Debug, Default)]
pub struct Heard {
    keys: Vec<String>,
    said: HashSet<String>,
    added: HashSet<String>,
}

impl Heard {
    /// Remembers `texts` as the utterances a conversation opens with.
    pub fn new(texts: &[String]) -> Heard {
        let mut h = Heard::default();
        for t in texts {
            h.remember(t, "");
        }
        h
    }

    /// Takes an utterance (and the words its reply added) into the conversation.
    pub fn remember(&mut self, text: &str, reply: &str) {
        let key = normalize(text);
        if !key.is_empty() && self.said.insert(key.clone()) {
            self.keys.push(key);
        }
        let added = normalize(reply);
        if !added.is_empty() {
            self.added.insert(added);
        }
    }

    /// What speaking `text` (a reply adding `reply`) would repeat, or `""`.
    pub fn matches(&self, text: &str, reply: &str) -> String {
        let key = normalize(text);
        if key.is_empty() {
            return String::new();
        }
        if self.said.contains(&key) {
            return key;
        }
        let added = normalize(reply);
        if !added.is_empty() && self.added.contains(&added) {
            return added;
        }
        for heard in &self.keys {
            if heard.contains(&key) {
                return heard.clone();
            }
        }
        String::new()
    }

    /// Whether speaking `text` would repeat the conversation.
    pub fn duplicate(&self, text: &str, reply: &str) -> bool {
        !self.matches(text, reply).is_empty()
    }

    fn said_word_for_word(&self, text: &str) -> bool {
        self.said.contains(&normalize(text))
    }
}

/// What a voice may not say: `veto(model, utterance)` is true for a candidate
/// the speaker must not speak.  The model handed in is the one the veto
/// judges with - the conversation's own, whoever is speaking.
pub type Veto<'a> = dyn FnMut(&mut Model, &str) -> bool + 'a;

/// How one conversation runs.
#[derive(Clone, Debug)]
pub struct ConverseOptions {
    pub turns: usize,
    pub mode: String,
    pub max_length: usize,
    pub context: usize,
    pub temperature: f64,
    pub k: usize,
    pub beam: usize,
    pub step_penalty: f64,
    pub seed: Option<i64>,
    pub speakers: Vec<String>,
    /// Utterances so far, to continue.
    pub history: Vec<String>,
    pub avoid_repeats: bool,
    /// Keeps a reply from repeating its own words (a stutter).
    pub avoid_word_repeats: bool,
    /// Times a voice that caught itself repeating may back up.
    pub explore: usize,
    /// Teaches the model what each rethink found out; a conversation with
    /// this on changes the model.
    pub learn: bool,
    /// A voice that catches itself repeating thinks about it before it backs
    /// up ([`crate::thinking`]); the thought rides on the turn's rethink.
    pub think: bool,
    /// How deep such a thought may question itself.
    pub think_depth: usize,
}

impl Default for ConverseOptions {
    fn default() -> ConverseOptions {
        ConverseOptions {
            turns: 6,
            mode: "beam".to_string(),
            max_length: 60,
            context: 12,
            temperature: 1.0,
            k: 5,
            beam: 0,
            step_penalty: 0.0,
            seed: None,
            speakers: DEFAULT_SPEAKERS.iter().map(|s| s.to_string()).collect(),
            history: Vec::new(),
            avoid_repeats: true,
            avoid_word_repeats: true,
            explore: EXPLORE,
            learn: true,
            think: true,
            think_depth: THINK_DEPTH,
        }
    }
}

fn check_mode(mode: &str) -> Result<&'static str, String> {
    match mode {
        "" | "dijkstra" | "beam" => Ok("beam"),
        "sample" => Ok("sample"),
        other => Err(format!("unknown mode {other:?}; expected 'beam' or 'sample'")),
    }
}

/// The context without its last word.
fn shorter(context: &str) -> String {
    let context = context.trim_end_matches([' ', '\t']);
    match context.rfind(' ') {
        Some(i) => context[..i].trim_end_matches([' ', '\t']).to_string(),
        None => String::new(),
    }
}

/// Whether the graph knows the end of `context` whole: its last gram is a
/// node (no guessed partial match).
fn usable(voice: &Model, context: &str) -> bool {
    let (node, _, lead) = voice.prefix_start(context);
    node != START && lead.is_empty()
}

/// How one search for candidates runs.
#[derive(Clone, Copy)]
pub(crate) struct Look<'o> {
    mode: &'o str,
    beam: usize,
    max_length: usize,
    step_penalty: f64,
    temperature: f64,
}

/// Continuations of `context` (`""`: texts from START), most likely first.
fn candidates(
    voice: &mut Model,
    context: &str,
    look: Look,
    k: usize,
    rng: &mut Option<Mt19937>,
) -> Result<Vec<PathResult>, String> {
    if look.mode == "sample" {
        let walk = voice.search(
            context,
            0,
            "sample",
            0,
            0,
            look.step_penalty,
            look.temperature,
            false,
            Some(look.max_length),
            rng.as_mut(),
            DEFAULT_TRAVERSAL,
            1.0,
            1.0,
            crate::model::SearchTuning::default(),
        )?;
        return Ok(vec![walk.best]);
    }
    let found = voice.predict(
        context,
        &PredictOptions {
            length: 0,
            mode: "beam".to_string(),
            k,
            beam: look.beam,
            to_end: true,
            max_length: Some(look.max_length),
            step_penalty: look.step_penalty,
            ..Default::default()
        },
    )?;
    Ok(found.top)
}

/// Up to `count` continuations: the `count` most likely (beam), or that many
/// walks (sample).
fn offer(
    voice: &mut Model,
    context: &str,
    look: Look,
    count: usize,
    rng: &mut Option<Mt19937>,
) -> Result<Vec<PathResult>, String> {
    if look.mode != "sample" {
        return candidates(voice, context, look, count, rng);
    }
    let mut drawn = Vec::new();
    for _ in 0..count {
        drawn.extend(candidates(voice, context, look, 1, rng)?);
    }
    Ok(drawn)
}

/// Teaches the graph what a rethink just found out, and returns the node it
/// was taught at (-1: none).
///
/// The node it backed up to gets an edge into BACK, the step it was about to
/// loop through gets dearer, and the step it took instead (when it found one)
/// gets cheaper.  From then on the search itself hands over at that node,
/// wherever it is walking: the trait is the model's, not the conversation's.
pub fn teach_back(voice: &mut Model, text: &str, at: usize, found: Option<&PathResult>, amount: f64) -> i64 {
    let (node, went, instead) = backing(voice, text, at, found);
    if node < 0 {
        // nothing of its own to mark: the repeat started where the graph could not place it
        return -1;
    }
    match voice.g.observe_back(node as usize, went, instead, amount) {
        Ok(_) => node,
        Err(_) => -1,
    }
}

/// Where a rethink backs up to, and what it teaches: `(node, went, instead)`,
/// `node = -1` when the graph cannot place the repeat.
fn backing(voice: &Model, text: &str, at: usize, found: Option<&PathResult>) -> (i64, Option<usize>, Option<usize>) {
    let (node, _, lead) = voice.prefix_start(&text[..at]);
    if node < FIRST || !lead.is_empty() || !voice.g.is_alive(node) {
        return (-1, None, None);
    }
    let rest = &text[at..];
    let word = match rest.find(' ') {
        Some(i) => &rest[..i],
        None => rest,
    };
    let (went, _, went_lead) = voice.prefix_start(&text[..at + word.len()]);
    let went = if !went_lead.is_empty() || went == node || voice.g.edge(node, went).is_none() {
        None
    } else {
        Some(went)
    };
    let instead = found
        .and_then(|f| f.node_ids.get(1).copied())
        .filter(|&n| voice.g.edge(node, n).is_some());
    (node as i64, went, instead)
}

/// A voice that caught itself repeating **thinks** before it backs up
/// ([`crate::thinking::think`]): the event is the rethink itself (`kind`),
/// the node it thinks at is the one it backed up to, and when the thought
/// stops it hands over to `BACK` with the lesson [`teach_back`] used to write
/// directly.  With `learn` off the voice still thinks, and teaches nothing.
/// `None` when the graph cannot place the repeat: nothing of its own to think at.
#[allow(clippy::too_many_arguments)]
pub(crate) fn think_back(
    voice: &mut Model,
    text: &str,
    at: usize,
    found: Option<&PathResult>,
    kind: &str,
    learn: bool,
    think_depth: usize,
    look: Look,
    k: usize,
    rng: &mut Option<Mt19937>,
) -> Result<Option<Thought>, String> {
    let (node, went, instead) = backing(voice, text, at, found);
    if node < 0 {
        return Ok(None);
    }
    let o = ThinkOptions {
        about: text.to_string(),
        at: Some(node as usize),
        trigger: kind.to_string(),
        went,
        instead,
        mode: look.mode.to_string(),
        k,
        beam: look.beam,
        max_length: look.max_length,
        step_penalty: look.step_penalty,
        temperature: look.temperature,
        seed: None,
        max_depth: think_depth,
        max_questions: THINK_QUESTIONS,
        learn,
        amount: 1.0,
    };
    Ok(Some(think(voice, &o, rng)?))
}

/// What a rethink teaches: a thought that hands over to BACK when it stops
/// (`think`), else BACK directly (`learn`).  Returns the node taught to hand over.
#[allow(clippy::too_many_arguments)]
fn teach(
    voice: &mut Model,
    text: &str,
    at: usize,
    found: Option<&PathResult>,
    record: &mut Rethink,
    learn: bool,
    thinking: bool,
    think_depth: usize,
    look: Look,
    k: usize,
    rng: &mut Option<Mt19937>,
) -> Result<i64, String> {
    if thinking {
        let thought = think_back(voice, text, at, found, &record.kind, learn, think_depth, look, k, rng)?;
        let taught = thought.as_ref().map(|t| t.handed_over).unwrap_or(-1);
        record.thought = thought;
        return Ok(taught);
    }
    Ok(if learn {
        teach_back(voice, text, at, found, 1.0)
    } else {
        -1
    })
}

/// How one [`backtrack`] runs.
pub struct BacktrackOptions<'a, 'v, 's> {
    /// What the voice may not rewrite - the context it picked up.
    pub keep: &'a str,
    /// What the reply would add to the context.
    pub added: &'a str,
    pub heard: &'a Heard,
    pub explore: usize,
    pub mode: &'a str,
    pub k: usize,
    pub beam: usize,
    pub max_length: usize,
    pub step_penalty: f64,
    pub temperature: f64,
    pub avoid_repeats: bool,
    pub avoid_word_repeats: bool,
    pub learn: bool,
    pub veto: Option<&'a mut Veto<'v>>,
    /// Think about the repeat before backing up ([`think_back`]).
    pub think: bool,
    pub think_depth: usize,
    /// Watches the backing up happen ([`Stream`]): `caught` the moment it
    /// notices, `backtrack` for every step back, then `found` or `stuck`.  It
    /// changes nothing about what is found.
    pub stream: Option<&'a mut Stream<'s>>,
}

/// Has a voice that caught itself repeating go back to where it would have
/// started saying it again, and look for another way on.
///
/// A stutter is cut where the walk went round ([`stutter_at`]); a repeat of
/// what the conversation has heard is cut before its last word, since the
/// whole line is a retread.  The search runs again from the cut - a longer
/// prefix than the turn started with, which forces the walk to leave the line
/// at exactly that point - and, when nothing new turns up, backs up one word
/// further and looks wider, `explore` times over.  `keep` is never rewritten:
/// a voice rethinks what it said, never what it heard.
pub fn backtrack(
    voice: &mut Model,
    scorer: Option<&mut Model>,
    text: &str,
    o: BacktrackOptions,
    rng: &mut Option<Mt19937>,
) -> Result<(Option<PathResult>, Rethink), String> {
    let mut scorer = scorer;
    let mut veto = o.veto;
    let mut stream = o.stream;
    let mut record = Rethink {
        taught: -1,
        ..Default::default()
    };
    let mut at = None;
    if o.avoid_word_repeats {
        let (noticed, where_) = caught(text, LONGEST_STUTTER);
        if !noticed.is_empty() {
            record.kind = "stutter".to_string();
            record.noticed = noticed;
            at = where_;
        }
    }
    if record.noticed.is_empty() && o.avoid_repeats {
        let noticed = o.heard.matches(text, o.added);
        if !noticed.is_empty() {
            record.kind = "repeat".to_string();
            record.noticed = noticed;
            at = last_word_at(text);
        }
    }
    if !record.noticed.is_empty() {
        if let Some(s) = stream.as_mut() {
            // what it will keep - "" when there is no backing up from here: nowhere to cut, the exploring
            // off, the repeat inside the words it picked up, or nothing of its own before it
            let kept = match at {
                Some(at) if o.explore > 0 && at >= o.keep.len() && !text[..at].trim().is_empty() => &text[..at],
                _ => "",
            };
            s(Json::obj([
                ("event", Json::str("caught")),
                ("kind", Json::str(record.kind.clone())),
                ("noticed", Json::str(record.noticed.clone())),
                ("cut", Json::str(kept)),
            ]));
        }
    }
    let Some(at) = at else { return Ok((None, record)) };
    if record.noticed.is_empty() || o.explore == 0 {
        return Ok((None, record));
    }
    let mut cut = text[..at].to_string();
    if cut.len() < o.keep.len() {
        // the other voice's words: not this one's to rethink
        return Ok((None, record));
    }
    let look = Look {
        mode: o.mode,
        beam: o.beam,
        max_length: o.max_length,
        step_penalty: o.step_penalty,
        temperature: o.temperature,
    };
    for step in 0..o.explore {
        if cut.trim().is_empty() {
            break;
        }
        record.cut = cut.clone();
        record.steps = step + 1;
        // the further back it goes, the wider it looks
        let wider = o.k * (step + 2);
        if let Some(s) = stream.as_mut() {
            s(Json::obj([
                ("event", Json::str("backtrack")),
                ("step", Json::Int((step + 1) as i64)),
                ("cut", Json::str(cut.clone())),
                ("wider", Json::Int(wider as i64)),
            ]));
        }
        let offered = offer(voice, &cut, look, wider, rng)?;
        for cand in offered {
            record.explored += 1;
            if cand.text.trim().is_empty() {
                continue;
            }
            if let Some(v) = veto.as_mut() {
                let refused = match scorer.as_deref_mut() {
                    Some(s) => v(s, &cand.full_text),
                    None => v(voice, &cand.full_text),
                };
                if refused {
                    continue;
                }
            }
            if (o.avoid_word_repeats && !stutter(&cand.full_text, LONGEST_STUTTER).is_empty())
                || (o.avoid_repeats && o.heard.duplicate(&cand.full_text, &cand.text))
            {
                continue;
            }
            record.found = true;
            record.taught = teach(
                voice,
                text,
                at,
                Some(&cand),
                &mut record,
                o.learn,
                o.think,
                o.think_depth,
                look,
                o.k,
                rng,
            )?;
            if let Some(s) = stream.as_mut() {
                s(Json::obj([
                    ("event", Json::str("found")),
                    ("text", Json::str(cand.full_text.clone())),
                    ("cost", Json::Num(cand.cost)),
                    ("explored", Json::Int(record.explored as i64)),
                ]));
            }
            return Ok((Some(cand), record));
        }
        let mut less = shorter(&cut);
        if less.len() < o.keep.len() || less == cut {
            break;
        }
        if !less.is_empty() && !less.ends_with(' ') {
            less.push(' ');
        }
        cut = less;
    }
    // it goes round here even if it found no way out
    record.taught = teach(
        voice,
        text,
        at,
        None,
        &mut record,
        o.learn,
        o.think,
        o.think_depth,
        look,
        o.k,
        rng,
    )?;
    if record.steps > 0 {
        if let Some(s) = stream.as_mut() {
            s(Json::obj([
                ("event", Json::str("stuck")),
                ("explored", Json::Int(record.explored as i64)),
            ]));
        }
    }
    Ok((None, record))
}

/// What one look through the candidates turned up.
#[derive(Default)]
struct Picked {
    /// What to say, or the best repeat when everything repeated.
    spoken: Option<PathResult>,
    skipped: usize,
    /// `spoken` repeats something: nothing else was left.
    repeat: bool,
    vetoed: usize,
    /// The best candidate rejected for repeating: the one worth backing out of.
    caught: Option<PathResult>,
}

/// The first candidate that adds something, is not vetoed and repeats
/// nothing; when they all repeat, the best of them is the fallback (the
/// cheapest never said word for word, else the cheapest) and the turn is
/// flagged a repeat.  A vetoed candidate is never the fallback.
fn pick(
    cands: Vec<PathResult>,
    heard: &Heard,
    avoid_repeats: bool,
    avoid_word_repeats: bool,
    mut refuse: impl FnMut(&str) -> bool,
) -> Picked {
    let mut out = Picked::default();
    let mut fallback_word_for_word = true;
    for c in cands {
        if c.text.trim().is_empty() {
            out.skipped += 1;
            continue;
        }
        if refuse(&c.full_text) {
            out.vetoed += 1;
            out.skipped += 1;
            continue;
        }
        let heard_before = avoid_repeats && heard.duplicate(&c.full_text, &c.text);
        let went_round = avoid_word_repeats && !stutter(&c.full_text, LONGEST_STUTTER).is_empty();
        if heard_before || went_round {
            if out.caught.is_none() {
                out.caught = Some(c.clone());
            }
            let word_for_word = heard.said_word_for_word(&c.full_text);
            if out.spoken.is_none() || (fallback_word_for_word && !word_for_word) {
                out.spoken = Some(c);
                fallback_word_for_word = word_for_word;
            }
            out.skipped += 1;
            continue;
        }
        out.spoken = Some(c);
        return out;
    }
    out.repeat = out.spoken.is_some();
    out
}

/// How one voice finds what to say next.
pub struct ReplyOptions<'a, 'v, 's> {
    /// What the conversation has already heard; remember a turn in it before
    /// asking for the next one, or the same reply comes back.
    pub heard: &'a Heard,
    pub index: usize,
    pub speaker: &'a str,
    pub mode: &'a str,
    pub max_length: usize,
    pub context: usize,
    pub temperature: f64,
    pub k: usize,
    pub beam: usize,
    pub step_penalty: f64,
    pub avoid_repeats: bool,
    pub avoid_word_repeats: bool,
    pub explore: usize,
    pub learn: bool,
    pub veto: Option<&'a mut Veto<'v>>,
    /// Think about a repeat before backing up ([`think_back`]).
    pub think: bool,
    pub think_depth: usize,
    /// Watches the turn being found ([`Stream`]): `look` for every context it
    /// continues (and `""` for a fresh text), then - when a candidate is
    /// caught repeating - `draft` and what [`backtrack`] does about it.  The
    /// turn itself is not this function's event: it is the caller's to speak
    /// ([`converse`] streams it as `turn`), since a reply may still be refused
    /// for repeating a duplicate already repeated.
    pub stream: Option<&'a mut Stream<'s>>,
}

/// What `voice` says next after `previous` - one turn, or `None` when it has
/// nothing to say.  `scorer` is the model the veto judges with when it is not
/// the speaker (a partner speaking in a guarded conversation).
pub fn reply(
    voice: &mut Model,
    scorer: Option<&mut Model>,
    previous: &str,
    o: ReplyOptions,
    rng: &mut Option<Mt19937>,
) -> Result<Option<Turn>, String> {
    let mode = check_mode(o.mode)?;
    if o.k < 1 || o.temperature < 0.0 || o.step_penalty < 0.0 {
        return Err("invalid reply options".to_string());
    }
    let mut scorer = scorer;
    let mut veto = o.veto;
    let mut stream = o.stream;
    let look = Look {
        mode,
        beam: o.beam,
        max_length: o.max_length,
        step_penalty: o.step_penalty,
        temperature: o.temperature,
    };
    let mut ctx = tail_context(previous, o.context);
    let mut spoken: Option<PathResult> = None;
    let mut rethought: Option<Rethink> = None;
    let mut repeat = false;
    let draws = if mode == "sample" { o.k } else { 1 };
    let (index, speaker) = (o.index, o.speaker);
    // the window of the stream: what this turn does before it is spoken
    let looking = |stream: &mut Option<&mut Stream>, from: &str| {
        if let Some(s) = stream.as_mut() {
            s(tagged(
                Json::obj([("event", Json::str("look")), ("from", Json::str(from))]),
                index,
                speaker,
            ));
        }
    };
    // one look through what `from` offers: candidates, a pick, and - for a
    // candidate rejected for repeating - one rethink per turn
    let look_from = |voice: &mut Model,
                     scorer: &mut Option<&mut Model>,
                     veto: &mut Option<&mut Veto>,
                     stream: &mut Option<&mut Stream>,
                     from: &str,
                     keep: &str,
                     rethought: &mut Option<Rethink>,
                     counts: &mut (usize, usize, usize),
                     rng: &mut Option<Mt19937>|
     -> Result<(Option<PathResult>, bool), String> {
        let cands = candidates(voice, from, look, o.k, rng)?;
        counts.0 += cands.len();
        let p = {
            let mut refuse = |text: &str| match veto.as_mut() {
                Some(v) => match scorer.as_deref_mut() {
                    Some(s) => v(s, text),
                    None => v(voice, text),
                },
                None => false,
            };
            pick(cands, o.heard, o.avoid_repeats, o.avoid_word_repeats, &mut refuse)
        };
        counts.1 += p.skipped;
        counts.2 += p.vetoed;
        let Some(caught) = p.caught.as_ref() else {
            return Ok((p.spoken, p.repeat));
        };
        if o.explore == 0 || rethought.is_some() {
            return Ok((p.spoken, p.repeat));
        }
        if let Some(s) = stream.as_mut() {
            s(tagged(
                Json::obj([
                    ("event", Json::str("draft")),
                    ("text", Json::str(caught.full_text.clone())),
                    ("cost", Json::Num(caught.cost)),
                ]),
                index,
                speaker,
            ));
        }
        let mut watch = |event: Json| {
            if let Some(s) = stream.as_mut() {
                s(tagged(event, index, speaker));
            }
        };
        let (found, record) = backtrack(
            voice,
            scorer.as_deref_mut(),
            &caught.full_text,
            BacktrackOptions {
                keep,
                added: &caught.text,
                heard: o.heard,
                explore: o.explore,
                mode,
                k: o.k,
                beam: o.beam,
                max_length: o.max_length,
                step_penalty: o.step_penalty,
                temperature: o.temperature,
                avoid_repeats: o.avoid_repeats,
                avoid_word_repeats: o.avoid_word_repeats,
                learn: o.learn,
                veto: veto.as_deref_mut(),
                think: o.think,
                think_depth: o.think_depth,
                stream: Some(&mut watch as &mut Stream),
            },
            rng,
        )?;
        counts.0 += record.explored;
        *rethought = Some(record);
        match found {
            Some(found) => Ok((Some(found), false)),
            None => Ok((p.spoken, p.repeat)),
        }
    };
    let mut counts = (0usize, 0usize, 0usize);
    while !ctx.is_empty() {
        if usable(voice, &ctx) {
            looking(&mut stream, &ctx);
            for _ in 0..draws {
                let keep = ctx.clone();
                let (got, rep) = look_from(
                    voice,
                    &mut scorer,
                    &mut veto,
                    &mut stream,
                    &ctx,
                    &keep,
                    &mut rethought,
                    &mut counts,
                    rng,
                )?;
                spoken = got;
                repeat = rep;
                if spoken.is_some() && !repeat {
                    break;
                }
            }
            if spoken.is_some() && !repeat {
                break;
            }
        }
        ctx = shorter(&ctx);
    }
    if spoken.is_none() || repeat {
        // nothing (new) follows the previous line: change the subject with a fresh text
        let mut fresh: Option<PathResult> = None;
        let mut fresh_repeat = false;
        looking(&mut stream, "");
        for _ in 0..draws {
            let (got, rep) = look_from(
                voice,
                &mut scorer,
                &mut veto,
                &mut stream,
                "",
                "",
                &mut rethought,
                &mut counts,
                rng,
            )?;
            fresh = got;
            fresh_repeat = rep;
            if fresh.is_some() && !fresh_repeat {
                break;
            }
        }
        if fresh.is_some() && (spoken.is_none() || !fresh_repeat) {
            spoken = fresh;
            repeat = fresh_repeat;
            ctx = String::new();
        }
    }
    let (offered, skipped, vetoed) = counts;
    let Some(spoken) = spoken else { return Ok(None) };
    // the whole utterance whatever it was continued from - the context, or the
    // words a rethink kept
    let text = spoken.full_text.clone();
    let reply_text = text.get(ctx.len()..).unwrap_or("").to_string();
    Ok(Some(Turn {
        index: o.index,
        speaker: o.speaker.to_string(),
        stutter: !stutter(&text, LONGEST_STUTTER).is_empty(),
        text,
        fresh: ctx.is_empty(),
        context: ctx,
        reply: reply_text,
        cost: spoken.cost,
        probability: spoken.probability(),
        reached_end: spoken.reached_end,
        given: false,
        repeat,
        rethink: rethought,
        candidates: offered,
        skipped,
        vetoed,
        labels: spoken.labels.clone(),
        node_ids: spoken.node_ids.clone(),
        step_costs: spoken.step_costs.clone(),
    }))
}

/// The model talks to itself (or to `partner`) for `o.turns` new turns.
///
/// When every candidate duplicates the conversation the best one is spoken
/// and flagged a repeat; a duplicate already repeated ends the conversation
/// instead of going round in circles.
pub fn converse(
    model: &mut Model,
    partner: Option<&mut Model>,
    opening: &str,
    o: &ConverseOptions,
    veto: Option<&mut Veto>,
    stream: Option<&mut Stream>,
) -> Result<Vec<Turn>, String> {
    let mode = check_mode(&o.mode)?;
    if o.k < 1 || o.temperature < 0.0 || o.step_penalty < 0.0 {
        return Err("invalid converse options".to_string());
    }
    let speakers: Vec<String> = if o.speakers.is_empty() {
        DEFAULT_SPEAKERS.iter().map(|s| s.to_string()).collect()
    } else {
        o.speakers.clone()
    };
    if speakers.iter().any(|s| s.trim().is_empty()) {
        return Err("speakers must be non-empty names".to_string());
    }
    let mut rng = o.seed.map(Mt19937::new);
    let mut partner = partner;
    let mut veto = veto;
    let mut stream = stream;
    let mut said: Vec<String> = o.history.clone();
    let mut repeated: HashSet<String> = HashSet::new();
    let mut result: Vec<Turn> = Vec::new();
    let mut index = said.len();
    if !opening.trim().is_empty() {
        let cost = -model.score(opening).log_prob;
        let probability = PathResult {
            cost,
            ..Default::default()
        }
        .probability();
        result.push(Turn {
            index,
            speaker: speakers[index % speakers.len()].clone(),
            text: opening.to_string(),
            reply: opening.to_string(),
            cost,
            probability,
            reached_end: true,
            fresh: true,
            given: true,
            candidates: 1,
            ..Default::default()
        });
        spoken(&mut stream, result.last().expect("the opening was pushed"));
        said.push(opening.to_string());
        index += 1;
    }
    let mut heard = Heard::new(&said);
    for _ in 0..o.turns {
        let previous = said.last().cloned().unwrap_or_default();
        let options = ReplyOptions {
            heard: &heard,
            index,
            speaker: &speakers[index % speakers.len()],
            mode,
            max_length: o.max_length,
            context: o.context,
            temperature: o.temperature,
            k: o.k,
            beam: o.beam,
            step_penalty: o.step_penalty,
            avoid_repeats: o.avoid_repeats,
            avoid_word_repeats: o.avoid_word_repeats,
            explore: o.explore,
            learn: o.learn,
            veto: veto.as_deref_mut(),
            think: o.think,
            think_depth: o.think_depth,
            stream: stream.as_deref_mut(),
        };
        let said_next = match (index % 2, partner.as_deref_mut()) {
            (1, Some(other)) => reply(other, Some(&mut *model), &previous, options, &mut rng)?,
            _ => reply(model, None, &previous, options, &mut rng)?,
        };
        let Some(said_next) = said_next else { break };
        if said_next.repeat && repeated.contains(&normalize(&said_next.text)) {
            // the voice can only say a duplicate it has already repeated: the conversation is over
            break;
        }
        spoken(&mut stream, &said_next);
        said.push(said_next.text.clone());
        let added = if said_next.context.is_empty() {
            ""
        } else {
            said_next.reply.as_str()
        };
        heard.remember(&said_next.text, added);
        if said_next.repeat {
            repeated.insert(normalize(&said_next.text));
        }
        result.push(said_next);
        index += 1;
    }
    Ok(result)
}

/// A conversation held through the pair.
pub struct GuardedConversation {
    pub turns: Vec<Turn>,
    /// A verdict on every distinct candidate the voices considered.
    pub verdicts: Vec<FilterVerdict>,
    /// How often a candidate was refused (a turn may be offered the same one
    /// again after its context was shortened).
    pub vetoed: usize,
}

/// Converses through the pair: every candidate reply the negative network
/// refuses is left unsaid.  The conversation is the positive model's; the
/// filter only supplies the veto, so a turn whose every candidate is vetoed
/// falls back exactly as a dead end does.
pub fn converse_guarded(
    pair: &mut Filter,
    partner: Option<&mut Model>,
    opening: &str,
    o: &ConverseOptions,
    stream: Option<&mut Stream>,
) -> Result<GuardedConversation, String> {
    let config = pair.config.clone();
    let mut verdicts: Vec<FilterVerdict> = Vec::new();
    // the same candidate can be offered again after a shorter context
    let mut seen: Vec<(String, bool)> = Vec::new();
    let negative: &mut Model = pair.negative;
    let turns = {
        let mut veto = |positive: &mut Model, text: &str| -> bool {
            if let Some((_, refused)) = seen.iter().find(|(t, _)| t == text) {
                return *refused;
            }
            let mut judge = Filter {
                positive,
                negative: &mut *negative,
                config: config.clone(),
            };
            let verdict = judge.judge(text);
            let refused = verdict.decision == "reject";
            seen.push((text.to_string(), refused));
            verdicts.push(verdict);
            refused
        };
        converse(pair.positive, partner, opening, o, Some(&mut veto), stream)?
    };
    let refused: Vec<String> = verdicts
        .iter()
        .filter(|v| v.decision == "reject")
        .map(|v| v.text.clone())
        .collect();
    pair.learn(&refused, "vetoed in conversation")?;
    let vetoed = turns.iter().map(|t| t.vetoed).sum();
    Ok(GuardedConversation {
        turns,
        verdicts,
        vetoed,
    })
}

/// The document a conversation is answered with, shared by the CLI and the
/// route.
fn turns_json(turns: &[Turn]) -> Json {
    Json::Arr(turns.iter().map(|t| t.to_json()).collect())
}

// -- the CLI --------------------------------------------------------------------------------------

/// Text in double quotes with whitespace escaped (JSON string syntax), as the
/// Python and Go CLIs quote what a voice says.
fn quote(text: &str) -> String {
    Json::str(text).render(0)
}

/// One spoken turn of a conversation, as the transcript prints it: the line,
/// its numbers and flags, and what the voice noticed about a repeat of its own.
fn say_turn(turn: &Turn) {
    let mut flags: Vec<String> = Vec::new();
    if turn.given {
        flags.push("given".to_string());
    }
    if turn.fresh && !turn.given {
        flags.push("new topic".to_string());
    }
    if turn.repeat {
        flags.push("repeat".to_string());
    }
    if turn.stutter {
        flags.push("repeats itself".to_string());
    }
    if turn.vetoed > 0 {
        flags.push(format!("{} vetoed", turn.vetoed));
    }
    println!("{}: {}", turn.speaker, turn.text);
    let mut detail = format!("    cost {:.4}  p {:.4}", turn.cost, turn.probability);
    if !turn.context.is_empty() {
        detail.push_str(&format!("  picked up {}", quote(&turn.context)));
    }
    if !flags.is_empty() {
        detail.push_str(&format!("  [{}]", flags.join(", ")));
    }
    println!("{detail}");
    if let Some(r) = turn.rethink.as_ref() {
        let caught = if r.kind == "stutter" {
            format!("saying {} twice", quote(&r.noticed))
        } else {
            format!("repeating {}", quote(&r.noticed))
        };
        let mut thought = format!("    caught itself {caught}");
        if r.steps == 0 {
            thought.push_str("; the words it picked up, not its own");
        } else if r.found {
            thought.push_str(&format!(
                "; kept {} and found another way on in {} path(s)",
                quote(&r.cut),
                r.explored
            ));
        } else {
            let ending = if turn.repeat {
                "said it anyway"
            } else {
                "took a lesser answer"
            };
            thought.push_str(&format!(
                "; kept {}, weighed {} path(s), {ending}",
                quote(&r.cut),
                r.explored
            ));
        }
        println!("{thought}");
        if let Some(t) = r.thought.as_ref() {
            println!("    {}", crate::thinking::summarize(t));
        }
    }
}

/// The [`Stream`] of `radixnet converse --stream`: the conversation as it
/// happens.
///
/// A `turn` is printed the way the transcript prints it ([`say_turn`]) the
/// moment it is spoken.  The window between two turns - what the voice does
/// before it commits: the context it continues, the draft it caught itself
/// on, where it backed up to, what it found - is printed as it happens too,
/// indented and dimmed on a terminal, so the answer stands apart from the
/// thinking that may still be rewritten.  With `--json` every event is one
/// JSON line on stdout instead.
struct ConversePrinter {
    json: bool,
    dimmed: bool,
    /// The context the current turn last continued.
    looking: Option<String>,
}

impl ConversePrinter {
    fn new(json: bool) -> ConversePrinter {
        use std::io::IsTerminal;
        ConversePrinter {
            json,
            dimmed: !json && std::io::stdout().is_terminal(),
            looking: None,
        }
    }

    /// A line that belongs to the window rather than the answer.
    fn dim(&self, text: &str) {
        if self.dimmed {
            println!("\x1b[2m{text}\x1b[0m");
        } else {
            println!("{text}");
        }
    }

    fn event(&mut self, event: Json) {
        if self.json {
            println!("{}", event.render(0));
            return;
        }
        let text = |key: &str| event.at(key).as_str().unwrap_or("").to_string();
        let number = |key: &str| event.at(key).as_i64().unwrap_or(0);
        match event.at("event").as_str().unwrap_or("") {
            "turn" => {
                self.looking = None;
                if let Some(turn) = Turn::from_json(event.at("turn")) {
                    say_turn(&turn);
                }
            }
            "look" => {
                let from = text("from");
                if let Some(previous) = self.looking.as_ref() {
                    // the first look of a turn is the context the turn will say it picked up
                    let tried = if from.is_empty() {
                        "changes the subject".to_string()
                    } else {
                        format!("tries {}", quote(&from))
                    };
                    self.dim(&format!("    nothing new follows {}; {tried}", quote(previous)));
                }
                self.looking = Some(from);
            }
            "draft" => self.dim(&format!("    was about to say {}", quote(&text("text")))),
            "caught" => {
                let caught = if text("kind") == "stutter" {
                    format!("saying {} twice", quote(&text("noticed")))
                } else {
                    format!("repeating {}", quote(&text("noticed")))
                };
                let mut line = format!("    caught itself {caught}");
                if text("cut").is_empty() {
                    line.push_str("; the words it picked up, not its own");
                }
                self.dim(&line);
            }
            "backtrack" => self.dim(&format!(
                "    backs up to {} and weighs up to {} paths (step {})",
                quote(&text("cut")),
                number("wider"),
                number("step")
            )),
            "found" => self.dim(&format!(
                "    found another way on: {} ({} path(s) weighed)",
                quote(&text("text")),
                number("explored")
            )),
            "stuck" => self.dim(&format!("    nothing new in {} path(s)", number("explored"))),
            _ => {}
        }
    }
}

impl Turn {
    /// A turn read back from its JSON (what the stream carries); `None` for anything else.
    pub fn from_json(doc: &Json) -> Option<Turn> {
        let Json::Obj(_) = doc else { return None };
        let text = |key: &str| doc.at(key).as_str().unwrap_or("").to_string();
        let flag = |key: &str| doc.at(key).as_bool().unwrap_or(false);
        let count = |key: &str| doc.at(key).as_i64().unwrap_or(0).max(0) as usize;
        let rethink = match doc.at("rethink") {
            Json::Obj(_) => {
                let r = doc.at("rethink");
                Some(Rethink {
                    kind: r.at("kind").as_str().unwrap_or("").to_string(),
                    noticed: r.at("noticed").as_str().unwrap_or("").to_string(),
                    cut: r.at("cut").as_str().unwrap_or("").to_string(),
                    steps: r.at("steps").as_i64().unwrap_or(0).max(0) as usize,
                    explored: r.at("explored").as_i64().unwrap_or(0).max(0) as usize,
                    found: r.at("found").as_bool().unwrap_or(false),
                    taught: r.at("taught").as_i64().unwrap_or(-1),
                    thought: Thought::from_json(r.at("thought")),
                })
            }
            _ => None,
        };
        Some(Turn {
            index: count("index"),
            speaker: text("speaker"),
            text: text("text"),
            context: text("context"),
            reply: text("reply"),
            cost: doc.at("cost").as_f64().unwrap_or(0.0),
            probability: doc.at("probability").as_f64().unwrap_or(0.0),
            reached_end: flag("reached_end"),
            fresh: flag("fresh"),
            given: flag("given"),
            repeat: flag("repeat"),
            stutter: flag("stutter"),
            rethink,
            candidates: count("candidates"),
            skipped: count("skipped"),
            vetoed: count("vetoed"),
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

/// `radixnet converse`: the model converses with itself.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let mut model = ctx.open(true)?;
    let mut partner = match args.get("partner") {
        Some(path) => {
            if !std::path::Path::new(path).is_file() {
                return Err(format!("partner model file not found: {path}"));
            }
            Some(Model::load(path)?)
        }
        None => None,
    };
    let speakers: Vec<String> = args
        .str("speakers", "A,B")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let speakers = if speakers.is_empty() {
        DEFAULT_SPEAKERS.iter().map(|s| s.to_string()).collect()
    } else {
        speakers
    };
    let opening = args.str("opening", "");
    let o = ConverseOptions {
        turns: args.usize("turns", 6)?,
        mode: args.str("mode", "beam"),
        max_length: args.usize("max-length", 60)?,
        context: args.usize("context", 12)?,
        temperature: args.float("temperature", 1.0)?,
        k: args.usize("k", 5)?.max(1),
        beam: args.usize("beam", 0)?,
        step_penalty: args.float("step-penalty", 0.0)?,
        seed: args.get("seed").map(|_| ctx.seed),
        speakers: speakers.clone(),
        history: Vec::new(),
        avoid_repeats: !args.on("allow-repeats"),
        avoid_word_repeats: !args.on("allow-word-repeats"),
        explore: args.usize("explore", EXPLORE)?,
        learn: !args.on("no-learn"),
        think: !args.on("no-think"),
        think_depth: args.usize("think-depth", THINK_DEPTH)?,
    };
    // --stream: the conversation is printed as it happens - each turn the moment it is spoken, and
    // before it what the voice does: the context it continues, a draft it catches itself on, where it
    // backs up to (with --json: one JSON object per line, the usual document last)
    let streaming = args.on("stream");
    let mut printer = ConversePrinter::new(ctx.json);
    let mut watch = |event: Json| printer.event(event);
    let stream: Option<&mut Stream> = if streaming {
        Some(&mut watch as &mut Stream)
    } else {
        None
    };
    let (turns, guard) = match ctx.open_guard()? {
        Some((mut negative, config)) => {
            // a reply the negative network vetoes is left unsaid; the voice looks for another one
            let mut pair = Filter::new(&mut model, &mut negative, config)?;
            let outcome = converse_guarded(&mut pair, partner.as_mut(), &opening, &o, stream)?;
            let report = guard_report(
                &pair,
                &outcome.verdicts,
                vec![("refusals", Json::Int(outcome.vetoed as i64))],
            );
            (outcome.turns, report)
        }
        None => (
            converse(&mut model, partner.as_mut(), &opening, &o, None, stream)?,
            Json::Null,
        ),
    };
    let said_twice = repeats(&turns);
    let mut taught: Vec<i64> = turns
        .iter()
        .filter_map(|t| t.rethink.as_ref().map(|r| r.taught))
        .filter(|&n| n >= 0)
        .collect();
    taught.sort_unstable();
    taught.dedup();
    // the nodes the conversation's thoughts taught to stop and think at
    let mut thought_at: Vec<i64> = turns
        .iter()
        .filter_map(|t| t.rethink.as_ref().and_then(|r| r.thought.as_ref()).map(|th| th.taught))
        .filter(|&n| n >= 0)
        .collect();
    thought_at.sort_unstable();
    thought_at.dedup();
    let mut doc = vec![
        ("guard".to_string(), guard),
        ("turns".to_string(), turns_json(&turns)),
        ("count".to_string(), Json::Int(turns.len() as i64)),
        ("speakers".to_string(), Json::strs(speakers)),
        ("mode".to_string(), Json::str(o.mode.clone())),
        ("opening".to_string(), Json::str(opening)),
        ("kind".to_string(), Json::str(model.kind())),
        (
            "partner_kind".to_string(),
            partner.as_ref().map(|p| Json::str(p.kind())).unwrap_or(Json::Null),
        ),
        ("repeats".to_string(), Json::strs(said_twice)),
        ("taught".to_string(), Json::ints(taught.iter().copied())),
        ("thought_at".to_string(), Json::ints(thought_at.iter().copied())),
    ];
    if (!taught.is_empty() || !thought_at.is_empty()) && args.on("save") {
        let path = ctx.save(&mut model)?;
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        doc.push((
            "saved".to_string(),
            Json::obj([("path", Json::str(path)), ("bytes", Json::Int(bytes as i64))]),
        ));
    }
    doc.push(("transcript".to_string(), Json::str(transcript(&turns))));
    if streaming && ctx.json {
        // JSON Lines: the events went out as they happened, and the usual document is the last line
        doc.insert(0, ("event".to_string(), Json::str("done")));
        println!("{}", Json::Obj(doc).render(0));
        return Ok(());
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

// -- the route ------------------------------------------------------------------------------------

/// The one conversation `POST /api/converse` and `POST /api/converse/stream`
/// both read from a body: the opening, how it runs, the partner kind (another
/// kind kept in memory) and whether the guard is on.
struct ConverseRequest {
    opening: String,
    options: ConverseOptions,
    partner_kind: Option<&'static str>,
    guard_on: bool,
}

fn converse_request(svc: &Arc<Service>, r: &Request) -> Result<ConverseRequest, ApiError> {
    // a partner of another kind is the model of that kind kept in memory (Python's `converse`)
    let partner = r.text("partner", "");
    let active = svc.with_model(|m| m.kind());
    let partner_kind = if partner.trim().is_empty() || partner.trim().to_lowercase() == active {
        None
    } else {
        let kind = crate::kinds::parse_kind(&partner).map_err(ApiError::bad_request)?;
        if svc.with_parked_kind(kind, |_| ()).is_none() {
            return Err(ApiError::bad_request(format!(
                "no {kind} model in memory to converse with; select that kind once to load it"
            )));
        }
        Some(kind)
    };
    let speakers: Vec<String> = match r.body.get("speakers") {
        Some(Json::Arr(items)) => items
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .collect(),
        Some(Json::Str(s)) => s.split(',').map(|x| x.trim().to_string()).collect(),
        _ => Vec::new(),
    };
    let speakers: Vec<String> = speakers.into_iter().filter(|s| !s.is_empty()).collect();
    let speakers = if speakers.is_empty() {
        DEFAULT_SPEAKERS.iter().map(|s| s.to_string()).collect()
    } else {
        speakers
    };
    let k = r.usize("k", 5)?;
    if k < 1 {
        return Err(ApiError::bad_request("\"k\" must be >= 1"));
    }
    let mut history = r.texts("history");
    history.extend(r.texts("history_text"));
    let o = ConverseOptions {
        turns: r.usize("turns", 6)?,
        mode: r.text("mode", "beam"),
        max_length: r.usize("max_length", 60)?,
        context: r.usize("context", 12)?,
        temperature: r.number("temperature", 1.0)?,
        k,
        beam: r.usize("beam", 0)?,
        step_penalty: r.number("step_penalty", 0.0)?,
        seed: r.body.get("seed").and_then(|v| v.as_i64()),
        speakers,
        history,
        avoid_repeats: r.flag("avoid_repeats", true),
        avoid_word_repeats: r.flag("avoid_word_repeats", true),
        explore: r.usize("explore", EXPLORE)?,
        learn: r.flag("learn", true),
        think: r.flag("think", true),
        think_depth: r.usize("think_depth", THINK_DEPTH)?,
    };
    Ok(ConverseRequest {
        opening: r.text("opening", ""),
        options: o,
        partner_kind,
        guard_on: r.flag("guard", true),
    })
}

/// One conversation as a route holds it - with or without a partner: the
/// parked partner's lock is taken first, then the running model's, then the
/// guard's - and the document it is answered with.
fn hold(svc: &Arc<Service>, request: &ConverseRequest, stream: Option<&mut Stream>) -> Result<Json, ApiError> {
    let ConverseRequest {
        opening,
        options: o,
        partner_kind,
        guard_on,
    } = request;
    let talk = |partner: Option<&mut Model>, stream: Option<&mut Stream>| -> Result<(Vec<Turn>, Json), ApiError> {
        let mut partner = partner;
        let mut stream = stream;
        let guarded = if *guard_on {
            svc.guard(|pair| -> Result<(Vec<Turn>, Json), String> {
                let outcome = converse_guarded(pair, partner.as_deref_mut(), opening, o, stream.as_deref_mut())?;
                // `vetoed` counts the distinct texts refused, `refusals` how often one was
                let report = guard_report(
                    pair,
                    &outcome.verdicts,
                    vec![("refusals", Json::Int(outcome.vetoed as i64))],
                );
                Ok((outcome.turns, report))
            })?
        } else {
            None
        };
        Ok(match guarded {
            Some(outcome) => outcome?,
            None => (
                svc.with_model(|m| converse(m, partner, opening, o, None, stream))?,
                Json::Null,
            ),
        })
    };
    let (turns, guard) = match partner_kind {
        None => talk(None, stream)?,
        Some(kind) => svc
            .with_parked_kind(kind, |other| talk(Some(other), stream))
            .ok_or_else(|| ApiError::bad_request(format!("no {kind} model in memory to converse with")))??,
    };
    let kind = svc.with_model(|m| m.kind());
    Ok(Json::obj([
        ("kind", Json::str(kind)),
        ("partner", partner_kind.map(Json::str).unwrap_or(Json::Null)),
        ("speakers", Json::strs(o.speakers.clone())),
        ("turns", turns_json(&turns)),
        ("count", Json::Int(turns.len() as i64)),
        // the duplicates the search could not avoid, ready to be punished
        ("repeats", Json::strs(repeats(&turns))),
        ("guard", guard),
    ]))
}

fn converse_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let request = converse_request(svc, r)?;
    hold(svc, &request, None)
}

/// `POST /api/converse/stream`: the same conversation streamed as it happens
/// ([`Stream`]) - every event is one JSON line, `turn` events are the answer
/// and the rest is the window a backtrack may still rewrite, and the last
/// line is `{"event": "done", ...}` carrying the document `/api/converse`
/// answers with.  A request refused before anything was streamed is an
/// ordinary 400; the server turns a failure after that into the stream's last
/// event.
fn converse_stream_route(svc: &Arc<Service>, r: &Request, sink: &mut Sink) -> Result<(), ApiError> {
    let request = converse_request(svc, r)?;
    let mut watch = |event: Json| sink.send(&event);
    let document = hold(svc, &request, Some(&mut watch as &mut Stream))?;
    let Json::Obj(mut pairs) = document else { return Ok(()) };
    pairs.insert(0, ("event".to_string(), Json::str("done")));
    sink.send(&Json::Obj(pairs));
    Ok(())
}

/// `POST /api/converse` and `POST /api/converse/stream`.
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/converse", converse_route);
    server.event_route("POST", "/api/converse/stream", converse_stream_route);
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

    #[test]
    fn the_tail_is_cut_at_a_word() {
        assert_eq!(tail_context("the cat sat on the mat", 7), "mat");
        assert_eq!(tail_context("the cat sat on the mat", 8), "the mat");
        assert_eq!(tail_context("short", 12), "short");
        assert_eq!(tail_context("anything", 0), "");
    }

    #[test]
    fn a_stutter_is_a_run_said_twice_in_a_row() {
        assert_eq!(stutter("the the west", LONGEST_STUTTER), "the");
        assert_eq!(stutter("the cat the cat sat", LONGEST_STUTTER), "the cat");
        assert_eq!(stutter("where there is a will there is a way", LONGEST_STUTTER), "");
        assert_eq!(stutter_at("say morning morning", LONGEST_STUTTER), Some(12));
        assert_eq!(last_word_at("one two three"), Some(8));
        assert_eq!(last_word_at("one"), None);
    }

    #[test]
    fn what_was_heard_counts_as_a_duplicate() {
        let mut heard = Heard::new(&["The cat sat on the mat".to_string()]);
        assert!(heard.duplicate("the  cat sat on the MAT", ""));
        assert!(heard.duplicate("cat sat", ""), "an echo of a line already spoken");
        assert!(!heard.duplicate("the cat sat on the mat and slept", ""), "it says more");
        heard.remember("a dog barked", "barked");
        assert!(heard.duplicate("the fox barked", "barked"));
    }

    #[test]
    fn a_conversation_takes_turns_without_repeating_itself() {
        let mut m = trained();
        let o = ConverseOptions {
            turns: 4,
            ..Default::default()
        };
        let turns = converse(&mut m, None, "the cat", &o, None, None).unwrap();
        assert!(turns.len() >= 2, "{turns:?}");
        assert!(turns[0].given && turns[0].fresh);
        assert_eq!(turns[0].speaker, "A");
        assert_eq!(turns[1].speaker, "B");
        let spoken: Vec<String> = turns.iter().filter(|t| !t.repeat).map(|t| normalize(&t.text)).collect();
        let unique: HashSet<&String> = spoken.iter().collect();
        assert_eq!(
            unique.len(),
            spoken.len(),
            "a line was said twice without being flagged"
        );
        let doc = turns[1].to_json();
        assert!(doc.at("candidates").as_i64().unwrap() >= 1);
    }

    #[test]
    fn a_veto_is_never_spoken() {
        let mut m = trained();
        let o = ConverseOptions {
            turns: 3,
            ..Default::default()
        };
        let mut refused = 0;
        let mut veto = |_: &mut Model, text: &str| {
            let no = text.contains("log");
            if no {
                refused += 1;
            }
            no
        };
        let turns = converse(&mut m, None, "", &o, Some(&mut veto), None).unwrap();
        assert!(!turns.is_empty());
        assert!(turns.iter().all(|t| !t.text.contains("log")), "{turns:?}");
        assert!(refused > 0, "{turns:?}");
    }

    /// "ha ha ..." loops; the other two lines leave the loop after "ha ".
    fn ways() -> Model {
        let mut m = Model::new(3, GraphOptions::default()).unwrap();
        let texts: Vec<String> = ["ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        m.train(
            &texts,
            &TrainOptions {
                epochs: 3,
                ..Default::default()
            },
        )
        .unwrap();
        m
    }

    fn streamed(m: &mut Model, opening: &str, o: &ConverseOptions) -> (Vec<Turn>, Vec<Json>) {
        let mut events: Vec<Json> = Vec::new();
        let mut watch = |event: Json| events.push(event);
        let turns = converse(m, None, opening, o, None, Some(&mut watch as &mut Stream)).unwrap();
        (turns, events)
    }

    fn kind(event: &Json) -> &str {
        event.at("event").as_str().unwrap_or("")
    }

    #[test]
    fn the_turns_streamed_are_the_turns_returned() {
        let mut m = trained();
        let o = ConverseOptions {
            turns: 6,
            learn: false,
            ..Default::default()
        };
        let (turns, events) = streamed(&mut m, "the cat sat on the mat", &o);
        assert_eq!(turns.len(), 7);
        let spoken: Vec<Json> = events
            .iter()
            .filter(|e| kind(e) == "turn")
            .map(|e| e.at("turn").clone())
            .collect();
        let expected: Vec<Json> = turns.iter().map(|t| t.to_json()).collect();
        assert_eq!(spoken, expected);
        // every event is one of the kinds, and the window between two turns belongs to the turn that follows
        let mut owner: Option<i64> = None;
        for event in events.iter().rev() {
            assert!(STREAM_EVENTS.contains(&kind(event)), "{event:?}");
            assert!(event.at("speaker").as_str().is_some(), "{event:?}");
            if kind(event) == "turn" {
                owner = event.at("index").as_i64();
            } else {
                assert_eq!(event.at("index").as_i64(), owner, "{event:?}");
            }
        }
        // a turn's context is where its last look started from ("" when it changed the subject)
        let mut looks: Vec<(i64, String)> = Vec::new();
        for event in &events {
            match kind(event) {
                "look" => looks.push((
                    event.at("index").as_i64().unwrap(),
                    event.at("from").as_str().unwrap().to_string(),
                )),
                "turn" if !event.at("turn").at("given").as_bool().unwrap() => {
                    let index = event.at("index").as_i64().unwrap();
                    let last = looks.iter().rev().find(|(i, _)| *i == index).map(|(_, f)| f.clone());
                    assert_eq!(last.as_deref(), event.at("turn").at("context").as_str());
                }
                _ => {}
            }
        }
        // a turn read back from the stream is the turn
        let back = Turn::from_json(&spoken[1]).unwrap();
        assert_eq!(back.to_json(), spoken[1]);
    }

    #[test]
    fn the_window_shows_the_backing_up() {
        let mut m = ways();
        let o = ConverseOptions {
            turns: 6,
            learn: false,
            ..Default::default()
        };
        let (turns, events) = streamed(&mut m, "", &o);
        let rethought: Vec<&Turn> = turns.iter().filter(|t| t.rethink.is_some()).collect();
        assert!(!rethought.is_empty(), "{}", transcript(&turns));
        let mut found_one = false;
        for turn in rethought {
            let r = turn.rethink.as_ref().unwrap();
            let window: Vec<&Json> = events
                .iter()
                .filter(|e| e.at("index").as_i64() == Some(turn.index as i64) && kind(e) != "turn")
                .collect();
            let kinds: Vec<&str> = window.iter().map(|e| kind(e)).collect();
            let draft = kinds.iter().position(|k| *k == "draft").expect("a draft");
            let caught = kinds.iter().position(|k| *k == "caught").expect("what it caught");
            assert_eq!(draft + 1, caught, "{kinds:?}");
            assert_eq!(window[caught].at("kind").as_str(), Some(r.kind.as_str()));
            assert_eq!(window[caught].at("noticed").as_str(), Some(r.noticed.as_str()));
            let steps: Vec<&&Json> = window.iter().filter(|e| kind(e) == "backtrack").collect();
            assert_eq!(steps.len(), r.steps);
            if let Some(first) = steps.first() {
                assert_eq!(window[caught].at("cut").as_str(), first.at("cut").as_str());
                assert!(window[draft]
                    .at("text")
                    .as_str()
                    .unwrap()
                    .starts_with(first.at("cut").as_str().unwrap()));
                assert_eq!(steps.last().unwrap().at("cut").as_str(), Some(r.cut.as_str()));
                for (i, step) in steps.iter().enumerate() {
                    assert_eq!(step.at("step").as_i64(), Some(i as i64 + 1));
                    assert!(step.at("wider").as_i64().unwrap() >= 5);
                }
            } else {
                assert_eq!(window[caught].at("cut").as_str(), Some(""));
            }
            if r.found {
                found_one = true;
                let found = window.iter().find(|e| kind(e) == "found").expect("the way on");
                assert_eq!(found.at("text").as_str(), Some(turn.text.as_str()));
                assert_eq!(found.at("explored").as_i64(), Some(r.explored as i64));
                assert!(!kinds.contains(&"stuck"));
            } else if !steps.is_empty() {
                let stuck = window.iter().find(|e| kind(e) == "stuck").expect("stuck");
                assert_eq!(stuck.at("explored").as_i64(), Some(r.explored as i64));
                assert!(!kinds.contains(&"found"));
            }
        }
        assert!(found_one, "{}", transcript(&turns));
        // nothing to catch, nothing in the window but the looking
        let off = ConverseOptions {
            explore: 0,
            ..o.clone()
        };
        let (_, events) = streamed(&mut ways(), "", &off);
        assert!(events.iter().all(|e| matches!(kind(e), "look" | "turn")), "{events:?}");
    }

    #[test]
    fn streaming_changes_nothing() {
        let (mut silent, mut watched) = (trained(), trained());
        let o = ConverseOptions {
            turns: 10,
            ..Default::default()
        };
        let plain = converse(&mut silent, None, "the cat sat on the mat", &o, None, None).unwrap();
        let (turns, events) = streamed(&mut watched, "the cat sat on the mat", &o);
        let a: Vec<Json> = plain.iter().map(|t| t.to_json()).collect();
        let b: Vec<Json> = turns.iter().map(|t| t.to_json()).collect();
        assert_eq!(a, b);
        assert_eq!(silent.g.num_edges(), watched.g.num_edges(), "what was learned");
        assert_eq!(events.iter().filter(|e| kind(e) == "turn").count(), plain.len());
    }

    /// How the backtracks below run: `keep`, and a stream to watch them.
    fn watched<'a, 's>(
        keep: &'a str,
        heard: &'a Heard,
        watch: &'a mut Stream<'s>,
    ) -> BacktrackOptions<'a, 'static, 's> {
        BacktrackOptions {
            keep,
            added: "",
            heard,
            explore: EXPLORE,
            mode: "beam",
            k: 3,
            beam: 0,
            max_length: 60,
            step_penalty: 0.0,
            temperature: 1.0,
            avoid_repeats: true,
            avoid_word_repeats: true,
            learn: false,
            think: false,
            think_depth: THINK_DEPTH,
            veto: None,
            stream: Some(watch),
        }
    }

    #[test]
    fn backtrack_streams_on_its_own() {
        let mut m = ways();
        let heard = Heard::new(&[]);
        let mut events: Vec<Json> = Vec::new();
        let mut watch = |event: Json| events.push(event);
        let (found, record) = backtrack(&mut m, None, "ha ha ha", watched("", &heard, &mut watch), &mut None).unwrap();
        let found = found.expect("another way on");
        let kinds: Vec<&str> = events.iter().map(kind).collect();
        assert_eq!(kinds, ["caught", "backtrack", "found"]);
        assert_eq!(
            events[0],
            Json::obj([
                ("event", Json::str("caught")),
                ("kind", Json::str("stutter")),
                ("noticed", Json::str("ha")),
                ("cut", Json::str("ha ")),
            ])
        );
        assert_eq!(events[1].at("wider").as_i64(), Some(6));
        assert_eq!(events[2].at("text").as_str(), Some(found.full_text.as_str()));
        assert_eq!(events[2].at("explored").as_i64(), Some(record.explored as i64));
        events.clear();
        let mut watch = |event: Json| events.push(event);
        backtrack(
            &mut m,
            None,
            "ha ha ha",
            watched("ha ha ", &heard, &mut watch),
            &mut None,
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].at("cut").as_str(), Some(""));
        events.clear();
        let mut watch = |event: Json| events.push(event);
        backtrack(
            &mut m,
            None,
            "the cat sat on the mat",
            watched("", &heard, &mut watch),
            &mut None,
        )
        .unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn backing_up_teaches_the_graph_where_it_goes_round() {
        let mut m = trained();
        let before = m.g.num_edges();
        let heard = Heard::new(&[]);
        let (_, record) = backtrack(
            &mut m,
            None,
            "the cat the cat sat",
            BacktrackOptions {
                keep: "",
                added: "",
                heard: &heard,
                explore: 2,
                mode: "beam",
                k: 3,
                beam: 0,
                max_length: 30,
                step_penalty: 0.0,
                temperature: 1.0,
                avoid_repeats: true,
                avoid_word_repeats: true,
                learn: true,
                veto: None,
                think: true,
                think_depth: THINK_DEPTH,
                stream: None,
            },
            &mut None,
        )
        .unwrap();
        assert_eq!(record.kind, "stutter");
        assert_eq!(record.noticed, "the cat");
        assert!(record.steps >= 1);
        if record.taught >= 0 {
            assert!(m.g.num_edges() > before, "a BACK edge was taught");
        }
    }
}
