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

use std::collections::HashSet;
use std::sync::Arc;

use crate::cli::Ctx;
use crate::duo::{guard_report, Filter, FilterVerdict};
use crate::graph::{FIRST, START};
use crate::http::{Answer, ApiError, Request, Server};
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
pub struct BacktrackOptions<'a, 'v> {
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
pub struct ReplyOptions<'a, 'v> {
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
    // one look through what `from` offers: candidates, a pick, and - for a
    // candidate rejected for repeating - one rethink per turn
    let look_from = |voice: &mut Model,
                     scorer: &mut Option<&mut Model>,
                     veto: &mut Option<&mut Veto>,
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
            for _ in 0..draws {
                let keep = ctx.clone();
                let (got, rep) = look_from(
                    voice,
                    &mut scorer,
                    &mut veto,
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
        for _ in 0..draws {
            let (got, rep) = look_from(voice, &mut scorer, &mut veto, "", "", &mut rethought, &mut counts, rng)?;
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
        };
        let spoken = match (index % 2, partner.as_deref_mut()) {
            (1, Some(other)) => reply(other, Some(&mut *model), &previous, options, &mut rng)?,
            _ => reply(model, None, &previous, options, &mut rng)?,
        };
        let Some(spoken) = spoken else { break };
        if spoken.repeat && repeated.contains(&normalize(&spoken.text)) {
            // the voice can only say a duplicate it has already repeated: the conversation is over
            break;
        }
        said.push(spoken.text.clone());
        let added = if spoken.context.is_empty() {
            ""
        } else {
            spoken.reply.as_str()
        };
        heard.remember(&spoken.text, added);
        if spoken.repeat {
            repeated.insert(normalize(&spoken.text));
        }
        result.push(spoken);
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
        converse(pair.positive, partner, opening, o, Some(&mut veto))?
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
    let (turns, guard) = match ctx.open_guard()? {
        Some((mut negative, config)) => {
            // a reply the negative network vetoes is left unsaid; the voice looks for another one
            let mut pair = Filter::new(&mut model, &mut negative, config)?;
            let outcome = converse_guarded(&mut pair, partner.as_mut(), &opening, &o)?;
            let report = guard_report(
                &pair,
                &outcome.verdicts,
                vec![("refusals", Json::Int(outcome.vetoed as i64))],
            );
            (outcome.turns, report)
        }
        None => (converse(&mut model, partner.as_mut(), &opening, &o, None)?, Json::Null),
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
    ctx.emit(Json::Obj(doc));
    Ok(())
}

// -- the route ------------------------------------------------------------------------------------

fn converse_route(svc: &Arc<Service>, r: &Request) -> Answer {
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
        speakers: speakers.clone(),
        history,
        avoid_repeats: r.flag("avoid_repeats", true),
        avoid_word_repeats: r.flag("avoid_word_repeats", true),
        explore: r.usize("explore", EXPLORE)?,
        learn: r.flag("learn", true),
        think: r.flag("think", true),
        think_depth: r.usize("think_depth", THINK_DEPTH)?,
    };
    let opening = r.text("opening", "");
    let guard_on = r.flag("guard", true);
    let provenance = crate::duo::maybe_flag(r, "provenance")?;
    // one conversation, with or without a partner: the parked partner's lock is
    // taken first, then the running model's, then the guard's
    let talk = |partner: Option<&mut Model>| -> Result<(Vec<Turn>, Json), ApiError> {
        let mut partner = partner;
        let guarded = if guard_on {
            svc.guard(provenance, |pair| -> Result<(Vec<Turn>, Json), String> {
                let outcome = converse_guarded(pair, partner.as_deref_mut(), &opening, &o)?;
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
                svc.with_model(|m| converse(m, partner, &opening, &o, None))?,
                Json::Null,
            ),
        })
    };
    let (turns, guard) = match partner_kind {
        None => talk(None)?,
        Some(kind) => svc
            .with_parked_kind(kind, |other| talk(Some(other)))
            .ok_or_else(|| ApiError::bad_request(format!("no {kind} model in memory to converse with")))??,
    };
    let kind = svc.with_model(|m| m.kind());
    Ok(Json::obj([
        ("kind", Json::str(kind)),
        ("partner", partner_kind.map(Json::str).unwrap_or(Json::Null)),
        ("speakers", Json::strs(speakers)),
        ("turns", turns_json(&turns)),
        ("count", Json::Int(turns.len() as i64)),
        // the duplicates the search could not avoid, ready to be punished
        ("repeats", Json::strs(repeats(&turns))),
        ("guard", guard),
    ]))
}

/// `POST /api/converse`.
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/converse", converse_route);
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
        let turns = converse(&mut m, None, "the cat", &o, None).unwrap();
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
        let turns = converse(&mut m, None, "", &o, Some(&mut veto)).unwrap();
        assert!(!turns.is_empty());
        assert!(turns.iter().all(|t| !t.text.contains("log")), "{turns:?}");
        assert!(refused > 0, "{turns:?}");
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
