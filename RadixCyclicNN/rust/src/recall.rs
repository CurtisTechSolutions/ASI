//! What the network remembers of what it was shown: the recall tutor for
//! speech and images (`radixnet/recall.py`, `go/radixnet/recall.go`).
//!
//! The English tutor needs an LLM because nobody knows in advance what the
//! right sentence is.  Speech and images need none, because **the right answer
//! is on file**: an utterance or a picture was encoded into text
//! ([`crate::speech`], [`crate::vision`]) and trained on, so asking the network
//! to write that text again is an exercise whose correction already exists.
//!
//! One round is the English tutor's three steps:
//!
//! 1. **the exercise** - the opening of a text the network was taught: the
//!    utterance's own token and the waveform header, or an image header and a
//!    few characters of its payload (an image header is the same for every
//!    picture of that size, so without a lead nothing says which one is wanted);
//! 2. **the completion** - the network writes the rest;
//! 3. **the marking** - the completion is run back through the codec and
//!    compared with the original, marked out of 10, and when it fails the
//!    single worst thing wrong with it is named: `unreadable`, `truncated`,
//!    `overrun`, `garbled`, `silence` (`blank`), `clipping` (`noise`),
//!    `mishearing`, `distortion` (`drift`).
//!
//! The mark is the severity and the original text is the correction, which is
//! what the negative network wants: [`teach_recall`] blames only the
//! characters the network got wrong and clears the rest.  Nothing here needs a
//! decoder for pictures or a transcriber: the comparison is over the payload
//! bytes both codecs already produce.
//!
//! Positions are counted in characters (code points), as Python counts them,
//! so a token or a label outside ASCII cuts the cue where Python cuts it.

use std::sync::Arc;
use std::time::Instant;

use crate::blame::{self, Fault, TeachOptions, TeachReport, RECALL_AGREEMENT, RECALL_OVERRUN, RECALL_TRUNCATED};
use crate::cli::Ctx;
use crate::http::{Answer, ApiError};
use crate::json::Json;
use crate::model::{Model, PredictOptions};
use crate::multipart::base64::is_python_space;
use crate::multipart::Form;
use crate::negative::python_repr;
use crate::service::Service;
use crate::speech::{self, squash};

/// What this module's lines are filed under.
const LOG: &str = "media";

/// The two things a network is taught that it can be asked to remember exactly.
pub const MODALITIES: &[&str] = &["speech", "image"];

/// The payload characters the exercise gives away when no lead is asked for.
pub fn default_lead(modality: &str) -> i64 {
    if modality == "image" {
        16
    } else {
        0
    }
}

/// The byte distance at which two payloads stop agreeing at all (one eighth of
/// the range).  Agreement is per byte, `max(0, 1 - |written - true| /
/// TOLERANCE)`, averaged over the payload that was asked for - so a byte never
/// written counts as total disagreement: an exact recall scores 1, one off by
/// a quantisation step ~0.95, random bytes ~0.12.
pub const TOLERANCE: f64 = 32.0;

/// Below this peak amplitude a decoded waveform is silence.
const FLAT_PEAK: f64 = 0.02;
/// Below this byte standard deviation an image payload has no picture in it.
const FLAT_SPREAD: f64 = 2.0;
/// Above this share of railed bytes a completion is clipping / noise.
const RAILED: f64 = 0.5;

// -- the exercise -----------------------------------------------------------------------------

/// `"speech"` or `"image"`, read from the text's own header.
pub fn modality_of(text: &str) -> Result<&'static str, String> {
    if text.contains("aud:") {
        Ok("speech")
    } else if text.contains("img:") {
        Ok("image")
    } else {
        Err("not an encoded utterance or image: expected an 'aud:' or 'img:' header".to_string())
    }
}

/// The position of `needle` in `hay`, both as characters.
fn find_chars(hay: &[char], needle: &str) -> Option<usize> {
    let needle: Vec<char> = needle.chars().collect();
    (0..=hay.len().saturating_sub(needle.len())).find(|&i| hay[i..].starts_with(&needle))
}

/// The character index where the payload starts: just past the header's third colon.
fn payload_at(chars: &[char]) -> Result<usize, String> {
    for header in ["aud:", "img:"] {
        let Some(at) = find_chars(chars, header) else { continue };
        let mut colons = 0;
        for (i, &c) in chars.iter().enumerate().skip(at) {
            if c == ':' {
                colons += 1;
                if colons == 3 {
                    return Ok(i + 1);
                }
            }
        }
        return Ok(chars.len());
    }
    Err("not an encoded utterance or image: expected an 'aud:' or 'img:' header".to_string())
}

/// The opening the network is given: everything up to the payload, plus
/// `lead` characters of it (the modality's default when `None`).
pub fn cue_of(text: &str, lead: Option<i64>) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let start = payload_at(&chars)?;
    let lead = match lead {
        Some(lead) => lead,
        None => default_lead(modality_of(text)?),
    };
    let end = (start + lead.max(0) as usize).min(chars.len());
    Ok(chars[..end].iter().collect())
}

/// One text the network was taught, and the opening it is asked to continue.
#[derive(Clone, Debug, PartialEq)]
pub struct RecallExercise {
    pub id: String,
    pub modality: String,
    /// The whole true text.
    pub reference: String,
    pub cue: String,
    /// Where it came from: a file name, an utterance token.
    pub label: String,
    /// Which of the texts it came from (texts that are not encoded are
    /// skipped, so this is not its position among the exercises).
    pub index: usize,
}

impl RecallExercise {
    /// The continuation that would be perfect.
    pub fn answer(&self) -> String {
        self.reference.chars().skip(self.cue.chars().count()).collect()
    }

    /// `{"id", "modality", "label", "index", "cue", "cue_chars", "answer_chars"}`.
    pub fn to_json(&self) -> Json {
        let cue_chars = self.cue.chars().count();
        Json::obj([
            ("id", Json::str(self.id.clone())),
            ("modality", Json::str(self.modality.clone())),
            ("label", Json::str(self.label.clone())),
            ("index", Json::Int(self.index as i64)),
            ("cue", Json::str(self.cue.clone())),
            ("cue_chars", Json::Int(cue_chars as i64)),
            (
                "answer_chars",
                Json::Int((self.reference.chars().count() - cue_chars) as i64),
            ),
        ])
    }
}

/// Taught texts -> exercises; texts that are not encoded are skipped.
pub fn exercises(texts: &[String], lead: Option<i64>, labels: &[String]) -> Vec<RecallExercise> {
    let mut out = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let body = text.trim_matches(is_python_space);
        if body.is_empty() {
            continue;
        }
        let (Ok(modality), Ok(cue)) = (modality_of(body), cue_of(body, lead)) else {
            continue;
        };
        out.push(RecallExercise {
            id: format!("r{}", i + 1),
            modality: modality.to_string(),
            reference: body.to_string(),
            cue,
            label: labels.get(i).cloned().unwrap_or_default(),
            index: i,
        });
    }
    out
}

// -- the marking ------------------------------------------------------------------------------

/// `(payload, repaired, error)` of an encoded text; `error` is set when it
/// cannot be read at all.
fn bytes_of(text: &str, modality: &str) -> (Vec<u8>, bool, String) {
    let parsed = if modality == "speech" {
        speech::parse_text(text).map(|p| (p.payload, p.repaired))
    } else {
        crate::vision::parse_text(text).map(|p| (p.payload, p.repaired))
    };
    match parsed {
        Ok((payload, repaired)) => (payload, repaired, String::new()),
        Err(err) => (Vec::new(), false, err),
    }
}

/// The mean per-byte agreement over `truth`; bytes never written count as 0.
fn agreement(written: &[u8], truth: &[u8]) -> f64 {
    if truth.is_empty() {
        return if written.is_empty() { 1.0 } else { 0.0 };
    }
    let mut total = 0.0;
    for (i, &expected) in truth.iter().enumerate() {
        let Some(&got) = written.get(i) else { break };
        total += (1.0 - (got as f64 - expected as f64).abs() / TOLERANCE).max(0.0);
    }
    total / truth.len() as f64
}

/// The standard deviation of the payload bytes.
fn spread(payload: &[u8]) -> f64 {
    if payload.is_empty() {
        return 0.0;
    }
    let mean = payload.iter().map(|&b| b as u64).sum::<u64>() as f64 / payload.len() as f64;
    let squares = payload.iter().fold(0.0, |sum, &b| {
        let d = b as f64 - mean;
        sum + d * d
    });
    (squares / payload.len() as f64).sqrt()
}

/// The share of bytes sitting against either end of the byte range.
fn railed(payload: &[u8]) -> f64 {
    if payload.is_empty() {
        return 0.0;
    }
    payload.iter().filter(|&&b| b <= 2 || b >= 253).count() as f64 / payload.len() as f64
}

/// The codec a waveform text declares (`auto` when it declares none the reader understands).
fn codec_of(text: &str) -> String {
    speech::parse_text(text)
        .map(|p| p.codec)
        .unwrap_or_else(|_| "auto".to_string())
}

/// Does this waveform payload decode to (near) nothing?
fn silent(payload: &[u8], codec: &str) -> bool {
    if payload.is_empty() {
        return true; // no samples, no peak
    }
    match speech::get_codec(codec) {
        Ok(codec) => {
            let peak = codec
                .decode(payload)
                .iter()
                .fold(0.0f64, |peak, &v| peak.max((v as f64).abs()));
            peak < FLAT_PEAK
        }
        // a codec the reader does not know: fall back to the raw byte spread
        Err(_) => spread(payload) < FLAT_SPREAD,
    }
}

/// What a check reports: facts, not a verdict.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    pub modality: String,
    pub readable: bool,
    pub repaired: bool,
    pub error: String,
    pub written_bytes: usize,
    pub expected_bytes: usize,
    pub length_ratio: f64,
    pub agreement: f64,
    pub flat: bool,
    pub reference_flat: bool,
    pub extreme: bool,
    pub reference_extreme: bool,
    /// The words that were spoken, and the round-trip transcript.
    pub said: String,
    pub heard: String,
    /// Whether the two say the same thing; `None` unless both are known.
    pub matched: Option<bool>,
}

impl Facts {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("modality", Json::str(self.modality.clone())),
            ("readable", Json::Bool(self.readable)),
            ("repaired", Json::Bool(self.repaired)),
            ("error", Json::str(self.error.clone())),
            ("written_bytes", Json::Int(self.written_bytes as i64)),
            ("expected_bytes", Json::Int(self.expected_bytes as i64)),
            ("length_ratio", Json::Num(self.length_ratio)),
            ("agreement", Json::Num(self.agreement)),
            ("flat", Json::Bool(self.flat)),
            ("reference_flat", Json::Bool(self.reference_flat)),
            ("extreme", Json::Bool(self.extreme)),
            ("reference_extreme", Json::Bool(self.reference_extreme)),
            ("said", Json::str(self.said.clone())),
            ("heard", Json::str(self.heard.clone())),
            ("match", self.matched.map(Json::Bool).unwrap_or(Json::Null)),
        ])
    }
}

/// The facts both modalities share.
fn common(payload: &[u8], truth: &[u8], repaired: bool, error: String, modality: &str) -> Facts {
    let readable = error.is_empty();
    let length_ratio = if !truth.is_empty() {
        payload.len() as f64 / truth.len() as f64
    } else if payload.is_empty() {
        1.0
    } else {
        2.0
    };
    Facts {
        modality: modality.to_string(),
        readable,
        repaired,
        error,
        written_bytes: payload.len(),
        expected_bytes: truth.len(),
        length_ratio,
        agreement: if readable { agreement(payload, truth) } else { 0.0 },
        extreme: readable && railed(payload) > RAILED,
        reference_extreme: railed(truth) > RAILED,
        ..Default::default()
    }
}

/// The facts about a recalled waveform, against the waveform it should have
/// been; with both `said` (the words spoken) and `heard` (the round-trip
/// transcript) it is also marked on whether it still says the same thing.
pub fn check_speech(written: &str, reference: &str, heard: &str, said: &str) -> Facts {
    let (truth, _, _) = bytes_of(reference, "speech");
    let (payload, repaired, error) = bytes_of(written, "speech");
    let mut facts = common(&payload, &truth, repaired, error, "speech");
    if facts.readable {
        facts.flat = silent(&payload, &codec_of(written));
    }
    facts.reference_flat = !truth.is_empty() && silent(&truth, &codec_of(reference));
    facts.said = squash(said);
    facts.heard = squash(heard);
    if !facts.said.is_empty() && !facts.heard.is_empty() {
        facts.matched = Some(facts.heard.to_lowercase() == facts.said.to_lowercase());
    }
    facts
}

/// The facts about a recalled image text.
pub fn check_image(written: &str, reference: &str) -> Facts {
    let (truth, _, _) = bytes_of(reference, "image");
    let (payload, repaired, error) = bytes_of(written, "image");
    let mut facts = common(&payload, &truth, repaired, error, "image");
    if facts.readable {
        facts.flat = spread(&payload) < FLAT_SPREAD;
    }
    facts.reference_flat = !truth.is_empty() && spread(&truth) < FLAT_SPREAD;
    facts
}

/// [`check_speech`] or [`check_image`], decided by the reference's header.
pub fn check(written: &str, reference: &str, heard: &str, said: &str) -> Result<Facts, String> {
    Ok(if modality_of(reference)? == "speech" {
        check_speech(written, reference, heard, said)
    } else {
        check_image(written, reference)
    })
}

/// The mark out of 10 a set of facts earns.
///
/// The agreement over the payload is the mark, thinned by whatever was written
/// past the end (agreement alone would give a rambling perfect recall full
/// marks).  Repairing the base64 costs a point, and a completion that is
/// silent, railed or says the wrong words is capped however well its bytes
/// line up: those are failures of a different kind.
pub fn mark(facts: &Facts) -> f64 {
    if !facts.readable {
        return 0.0;
    }
    let mut score = 10.0 * facts.agreement;
    let (written, expected) = (facts.written_bytes, facts.expected_bytes);
    if written > expected && expected > 0 {
        score *= expected as f64 / written as f64; // the excess is waste: it was never asked for
    }
    if facts.repaired {
        score -= 1.0;
    }
    if facts.flat && !facts.reference_flat {
        score = score.min(2.0);
    }
    if facts.extreme && !facts.reference_extreme {
        score = score.min(3.0);
    }
    if facts.matched == Some(false) {
        score = score.min(4.0);
    }
    score.clamp(0.0, 10.0)
}

/// The single worst thing wrong with a recalled waveform or image
/// (`blame.recall_reason`), in the order the faults matter; `"none"` when
/// nothing is.
pub fn recall_reason(facts: &Facts) -> &'static str {
    let speech = facts.modality != "image";
    if !facts.readable {
        return "unreadable";
    }
    if facts.length_ratio < RECALL_TRUNCATED {
        return "truncated";
    }
    if facts.length_ratio > RECALL_OVERRUN {
        return "overrun";
    }
    if facts.repaired {
        return "garbled";
    }
    if facts.flat && !facts.reference_flat {
        return if speech { "silence" } else { "blank" };
    }
    if facts.extreme && !facts.reference_extreme {
        return if speech { "clipping" } else { "noise" };
    }
    if facts.matched == Some(false) {
        return "mishearing";
    }
    if facts.agreement < RECALL_AGREEMENT {
        return if speech { "distortion" } else { "drift" };
    }
    "none"
}

/// One sentence of teaching, in the plain words the journal keeps.
fn comment(facts: &Facts, reason: &str, score: f64) -> String {
    let thing = if facts.modality == "speech" {
        "waveform"
    } else {
        "image"
    };
    match reason {
        "unreadable" => format!(
            "the completion is not an encoded {thing}: {}",
            if facts.error.is_empty() {
                "the header is gone"
            } else {
                &facts.error
            }
        ),
        "truncated" => format!(
            "the {thing} stops after {} of {} bytes ({:.0}% of it)",
            facts.written_bytes,
            facts.expected_bytes,
            facts.length_ratio * 100.0
        ),
        "overrun" => format!(
            "the {thing} runs on to {} bytes where {} were wanted",
            facts.written_bytes, facts.expected_bytes
        ),
        "garbled" => format!("the base64 had to be repaired before the {thing} could be read"),
        "silence" => "it decodes to silence: nothing was said back".to_string(),
        "blank" => "it decodes to a flat image: nothing was drawn".to_string(),
        "clipping" => "the waveform is railed against the limits rather than shaped".to_string(),
        "noise" => "the image is all extremes: noise rather than a picture".to_string(),
        "mishearing" => format!(
            "it says {} where {} was said",
            python_repr(&facts.heard),
            python_repr(&facts.said)
        ),
        "distortion" | "drift" => format!(
            "the {thing} is the right shape but wrong: {:.0}% agreement with the original",
            facts.agreement * 100.0
        ),
        _ => format!("recalled at {score:.1}/10"),
    }
}

/// The marking of one recalled text: the shape the English tutor's grade has.
#[derive(Clone, Debug, PartialEq)]
pub struct RecallGrade {
    pub score: f64,
    pub passed: bool,
    /// The reason tag ([`recall_reason`]); `"none"` on a pass.
    pub error: String,
    /// The text it should have written; empty on a pass.
    pub correction: String,
    /// One sentence saying what went wrong; empty on a pass.
    pub comment: String,
    pub graded_by: String,
    pub facts: Facts,
}

impl RecallGrade {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("score", Json::Num(self.score)),
            ("passed", Json::Bool(self.passed)),
            ("error", Json::str(self.error.clone())),
            ("correction", Json::str(self.correction.clone())),
            ("comment", Json::str(self.comment.clone())),
            ("graded_by", Json::str(self.graded_by.clone())),
            ("facts", self.facts.to_json()),
        ])
    }
}

/// Facts -> a grade: the mark, the reason, the correction and a sentence of
/// teaching.  A pass carries no reason, correction or comment.
pub fn grade(facts: Facts, correction: &str, threshold: f64) -> RecallGrade {
    let score = mark(&facts);
    let passed = score >= threshold;
    let reason = if passed { "none" } else { recall_reason(&facts) };
    RecallGrade {
        score,
        passed,
        error: reason.to_string(),
        correction: if passed { String::new() } else { correction.to_string() },
        comment: if passed {
            String::new()
        } else {
            comment(&facts, reason, score)
        },
        graded_by: "recall".to_string(),
        facts,
    }
}

// -- the lesson -------------------------------------------------------------------------------

/// One exercise, one completion by the network, one grade.
#[derive(Clone, Debug, PartialEq)]
pub struct RecallLesson {
    pub exercise: RecallExercise,
    pub attempt: usize,
    pub mode: String,
    pub continuation: String,
    /// Cue + continuation: what was marked.
    pub text: String,
    pub cost: f64,
    pub reached_end: bool,
    pub seconds: f64,
    pub grade: RecallGrade,
}

impl RecallLesson {
    /// `{"exercise", "attempt", "mode", "sentence", "text", "continuation_chars",
    /// "cost", "reached_end", "seconds", "grade"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("exercise", self.exercise.to_json()),
            ("attempt", Json::Int(self.attempt as i64)),
            ("mode", Json::str(self.mode.clone())),
            ("sentence", Json::str(self.text.clone())),
            ("text", Json::str(self.text.clone())),
            (
                "continuation_chars",
                Json::Int(self.continuation.chars().count() as i64),
            ),
            ("cost", Json::Num(self.cost)),
            ("reached_end", Json::Bool(self.reached_end)),
            ("seconds", Json::Num(self.seconds)),
            ("grade", self.grade.to_json()),
        ])
    }
}

/// How one quiz runs (`recall.quiz`'s keywords).
pub struct QuizOptions<'a> {
    /// Payload characters given away; `None` is the modality's default.
    pub lead: Option<i64>,
    pub labels: Vec<String>,
    /// Tries per exercise (the first in `mode`, the rest sampled); it stops at
    /// the first pass.
    pub attempts: usize,
    pub mode: String,
    pub temperature: f64,
    /// Payload characters asked for (0: all of it); the reference is cut to
    /// match, so a capped quiz is a fair one.
    pub length: i64,
    pub threshold: f64,
    /// Transcribe what it said back and compare the words (needs a transcriber).
    pub listen_back: bool,
    /// The words each utterance says, by input position.
    pub said: Vec<String>,
    pub progress: Option<&'a dyn Fn(&RecallLesson)>,
    pub stop: Option<&'a dyn Fn() -> bool>,
}

impl Default for QuizOptions<'_> {
    /// The Python defaults.
    fn default() -> Self {
        QuizOptions {
            lead: None,
            labels: Vec::new(),
            attempts: 1,
            mode: "beam".to_string(),
            temperature: 1.0,
            length: 0,
            threshold: 6.0,
            listen_back: false,
            said: Vec::new(),
            progress: None,
            stop: None,
        }
    }
}

/// The words a decoded waveform says, when a transcriber is reachable; `""` otherwise.
fn transcript_of(text: &str) -> String {
    let Ok(decoded) = speech::decode_text(text, None) else {
        return String::new();
    };
    let o = speech::asr::AsrOptions {
        backend: "auto".to_string(),
        ..Default::default()
    };
    speech::asr::transcribe(&decoded.wav, &o)
        .map(|heard| squash(&heard.transcript))
        .unwrap_or_default()
}

/// The first `n` characters in Python's `s[:n]` sense (a negative `n` counts from the end).
fn python_prefix(text: &str, n: i64) -> String {
    let count = text.chars().count() as i64;
    let keep = if n >= 0 { n.min(count) } else { (count + n).max(0) };
    text.chars().take(keep as usize).collect()
}

/// Asks the network to write out texts it was taught, and marks what comes back.
pub fn quiz(model: &mut Model, texts: &[String], o: &QuizOptions) -> Result<Vec<RecallLesson>, String> {
    let items = exercises(texts, o.lead, &o.labels);
    let mut lessons = Vec::new();
    for item in items {
        let mut reference = item.reference.clone();
        let mut wanted = (reference.chars().count() - item.cue.chars().count()) as i64;
        if o.length != 0 && o.length < wanted {
            wanted = o.length;
            // a capped quiz is marked against what it asked for
            reference = format!("{}{}", item.cue, python_prefix(&item.answer(), wanted));
        }
        let budget = ((wanted as f64 * RECALL_OVERRUN) as i64 + 8).max(1) as usize;
        for attempt in 0..o.attempts.max(1) {
            if o.stop.is_some_and(|stop| stop()) {
                return Ok(lessons);
            }
            let mode = if attempt == 0 {
                o.mode.clone()
            } else {
                "sample".to_string()
            };
            let started = Instant::now();
            let found = model.predict(
                &item.cue,
                &PredictOptions {
                    length: budget,
                    mode: mode.clone(),
                    temperature: o.temperature,
                    max_length: Some(budget),
                    to_end: false,
                    ..Default::default()
                },
            )?;
            let written = model.encoding().join(&[&item.cue, &found.best.text]);
            let facts = if item.modality == "speech" {
                let said = o.said.get(item.index).map(String::as_str).unwrap_or("");
                let heard = if o.listen_back {
                    transcript_of(&written)
                } else {
                    String::new()
                };
                check_speech(&written, &reference, &heard, said)
            } else {
                check_image(&written, &reference)
            };
            let lesson = RecallLesson {
                exercise: item.clone(),
                attempt,
                mode,
                continuation: found.best.text.clone(),
                text: written,
                cost: found.best.cost,
                reached_end: found.best.reached_end,
                seconds: started.elapsed().as_secs_f64(),
                grade: grade(facts, &reference, o.threshold),
            };
            crate::log_debug!(
                LOG,
                "{} attempt {}: {:.1}/10 {}",
                item.id,
                attempt,
                lesson.grade.score,
                lesson.grade.error
            );
            let passed = lesson.grade.passed;
            if let Some(progress) = o.progress {
                progress(&lesson);
            }
            lessons.push(lesson);
            if passed {
                break; // it remembered: no need to ask again
            }
        }
    }
    Ok(lessons)
}

/// What a quiz came to: `{"lessons", "passed", "mean_score", "mean_agreement",
/// "reasons", "modality"}`, the reasons most frequent first.
pub fn report_card(lessons: &[RecallLesson]) -> Json {
    let mut reasons: Vec<(String, i64)> = Vec::new();
    for lesson in lessons.iter().filter(|l| !l.grade.passed && l.grade.error != "none") {
        match reasons.iter_mut().find(|(r, _)| *r == lesson.grade.error) {
            Some((_, n)) => *n += 1,
            None => reasons.push((lesson.grade.error.clone(), 1)),
        }
    }
    reasons.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let n = lessons.len() as f64;
    let scores = lessons.iter().fold(0.0, |sum, l| sum + l.grade.score);
    let agreements = lessons.iter().fold(0.0, |sum, l| sum + l.grade.facts.agreement);
    let mut kinds: Vec<&str> = lessons.iter().map(|l| l.exercise.modality.as_str()).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let modality = match kinds.len() {
        0 => "",
        1 => kinds[0],
        _ => "mixed",
    };
    Json::obj([
        ("lessons", Json::Int(lessons.len() as i64)),
        (
            "passed",
            Json::Int(lessons.iter().filter(|l| l.grade.passed).count() as i64),
        ),
        (
            "mean_score",
            if lessons.is_empty() {
                Json::Null
            } else {
                Json::Num(scores / n)
            },
        ),
        (
            "mean_agreement",
            Json::Num(if lessons.is_empty() { 0.0 } else { agreements / n }),
        ),
        (
            "reasons",
            Json::Obj(reasons.into_iter().map(|(r, c)| (r, Json::Int(c))).collect()),
        ),
        ("modality", Json::str(modality)),
    ])
}

// -- blaming what it forgot -------------------------------------------------------------------

/// A round of recall lessons as faults and the texts that clear blame
/// (`blame.faults_from_recall`).
///
/// The correct text is on file, so it rides along in each fault as the
/// correction: only the characters the network got wrong are blamed, and the
/// original - correct by construction - clears blame.
pub fn faults_from_recall(lessons: &[RecallLesson], threshold: f64, source: &str) -> (Vec<Fault>, Vec<String>) {
    let mut faults = Vec::new();
    let mut passed: Vec<String> = Vec::new();
    for lesson in lessons {
        let text = &lesson.text;
        if text.is_empty() {
            continue;
        }
        let g = &lesson.grade;
        if g.passed {
            passed.push(text.clone());
            continue;
        }
        let mut reason = g.error.trim().to_lowercase();
        if reason.is_empty() || reason == "none" {
            reason = recall_reason(&g.facts).to_string();
        }
        if reason == "none" {
            reason = blame::DEFAULT_REASON.to_string();
        }
        let mut fault = Fault::new(
            text,
            &reason,
            blame::severity_from_rating(Some(g.score), threshold),
            &g.comment,
            source,
        );
        if !g.correction.is_empty() && g.correction != *text {
            fault.correction = g.correction.clone();
        }
        faults.push(fault);
        if !g.correction.is_empty() && !passed.contains(&g.correction) {
            passed.push(g.correction.clone());
        }
    }
    (faults, passed)
}

/// Feeds a round of the recall tutor into the negative network
/// (`blame.teach_recall`); `clear_passes` off keeps what it remembered from
/// clearing anything.
pub fn teach_recall(
    negative: &mut Model,
    lessons: &[RecallLesson],
    threshold: f64,
    clear_passes: bool,
    source: &str,
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    let source = if source.is_empty() { "recall" } else { source };
    let (faults, passed) = faults_from_recall(lessons, threshold, source);
    let clearing: &[String] = if clear_passes { &passed } else { &[] };
    let mut report = blame::teach(negative, &faults, clearing, o)?;
    report.source = source.to_string();
    report.threshold = Some(threshold);
    report.faults = faults;
    report.passed = passed.len();
    Ok(report)
}

/// `{"blamed", "cleared", "edges", "reasons", "severity_mean"}`: what the
/// CLI and the API report of a round of teaching.
fn taught_json(report: &TeachReport) -> Json {
    let full = report.to_json();
    Json::obj(["blamed", "cleared", "edges", "reasons", "severity_mean"].map(|key| (key, full.at(key).clone())))
}

/// Everything the negative network has blamed, heaviest first (Python's
/// `reason_table`): `[{"reason", "blame", "fails", "fails_resets", "edges", "share"}]`.
pub fn reasons_json(negative: &Model) -> Json {
    let rows = negative.g.reason_table();
    let mut edges = vec![0i64; rows.iter().map(|r| r.id + 1).max().unwrap_or(0)];
    for e in 0..negative.g.num_edge_ids() {
        if negative.g.is_edge_alive(e) {
            for row in negative.g.edge_reasons(e) {
                if let Some(n) = edges.get_mut(row.id) {
                    *n += 1;
                }
            }
        }
    }
    let total: f64 = rows.iter().map(|r| r.blame).sum();
    let total = if total == 0.0 { 1.0 } else { total };
    let mut rows = rows;
    rows.sort_by(|a, b| {
        b.blame
            .partial_cmp(&a.blame)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.reason.cmp(&b.reason))
    });
    Json::Arr(
        rows.iter()
            .map(|r| {
                Json::obj([
                    ("reason", Json::str(r.reason.clone())),
                    ("blame", Json::Num(r.blame)),
                    ("fails", Json::Int(r.fails)),
                    ("fails_resets", Json::Int(r.fails_resets)),
                    ("edges", Json::Int(edges.get(r.id).copied().unwrap_or(0))),
                    ("share", Json::Num(r.blame / total)),
                ])
            })
            .collect(),
    )
}

// -- the command line and the routes ----------------------------------------------------------

/// The recall options of a command line.
fn quiz_options_cli(ctx: &Ctx) -> Result<QuizOptions<'static>, String> {
    let a = &ctx.args;
    let mode = a.str("mode", "beam");
    if !["beam", "dijkstra", "sample"].contains(&mode.as_str()) {
        return Err(format!("--mode must be one of beam, dijkstra, sample, got {mode:?}"));
    }
    let attempts = a.int("attempts", 1)?;
    if attempts < 1 {
        return Err(format!("--attempts must be at least 1, got {attempts}"));
    }
    let lead = match a.get("lead") {
        Some(_) => Some(a.int("lead", 0)?.max(0)),
        None => None,
    };
    Ok(QuizOptions {
        lead,
        attempts: attempts as usize,
        mode,
        temperature: a.float("temperature", 1.0)?,
        length: a.int("length", 0)?.max(0),
        threshold: a.float("threshold", 6.0)?,
        ..Default::default()
    })
}

/// `image tutor` and `speech tutor` (`cli._recall_tutor`): teach the texts
/// first with `--train`, quiz, mark, and with `--blame` teach the negative
/// network why each failure failed.
pub fn tutor_cli(
    ctx: &Ctx,
    texts: &[String],
    labels: &[String],
    modality: &str,
    said: &[String],
) -> Result<Json, String> {
    let a = &ctx.args;
    let mut options = quiz_options_cli(ctx)?;
    options.labels = labels.to_vec();
    options.said = said.to_vec();
    options.listen_back = modality == "speech" && a.on("listen-back");
    let train = a.on("train");
    let mut model = ctx.open(false)?;
    if train {
        let epochs = a.usize("epochs", 3)?;
        crate::log_info!(LOG, "teaching it first: epochs={epochs}");
        model.train(
            texts,
            &crate::model::TrainOptions {
                epochs,
                ..Default::default()
            },
        )?;
    }
    let mut negative = if a.on("blame") {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let lessons = quiz(&mut model, texts, &options)?;
    let card = report_card(&lessons);
    let mut doc = Json::obj([
        ("modality", Json::str(modality)),
        ("lessons", Json::Arr(lessons.iter().map(|l| l.to_json()).collect())),
        ("report", card),
        ("interrupted", Json::Bool(false)),
        ("trained", Json::Bool(train)),
        ("saved", Json::Null),
        ("negative", Json::Null),
    ]);
    if train {
        let out = a.str("model-out", &ctx.model_path);
        speech::extend(&mut doc, vec![("saved", speech::save_model(&mut model, &out)?)]);
    }
    if let Some(negative) = negative.as_mut() {
        let report = teach_recall(
            negative,
            &lessons,
            options.threshold,
            true,
            modality,
            &TeachOptions::default(),
        )?;
        crate::log_info!(
            LOG,
            "blamed {} failure(s) over {} edge(s); cleared {} fragment(s) from what it did remember",
            report.blamed,
            report.edges,
            report.cleared
        );
        let path = ctx.negative_path();
        let saved = speech::save_model(negative, &path)?;
        speech::extend(
            &mut doc,
            vec![(
                "negative",
                Json::obj([
                    ("path", Json::str(path)),
                    ("saved", saved),
                    ("reasons", reasons_json(negative)),
                    ("stats", crate::cli::negative_stats(negative)),
                    ("taught", taught_json(&report)),
                ]),
            )],
        );
    }
    Ok(doc)
}

/// The recall options of a request (`api._recall_options`).
fn quiz_options_api<'a>(form: &Form, modality: &str) -> Result<QuizOptions<'a>, ApiError> {
    let bad = |err: String| ApiError::bad_request(format!("bad recall option: {err}"));
    let lead = match form.option("lead") {
        None => None,
        Some(Json::Str(s)) if s.is_empty() => None,
        Some(_) => Some(form.int("lead", 0).map_err(bad)?),
    };
    Ok(QuizOptions {
        lead,
        length: form.int("length", 0).map_err(bad)?,
        attempts: form.int("attempts", 1).map_err(bad)?.max(1) as usize,
        mode: form.text("mode", "beam"),
        temperature: form.float("temperature", 1.0).map_err(bad)?,
        listen_back: modality == "speech" && form.flag("listen_back", false),
        ..Default::default()
    })
}

/// The service's negative network: the one in memory, else the file beside
/// the model, else a fresh one in the running model's encoding.
fn with_negative<T>(svc: &Service, f: impl FnOnce(&mut Model) -> Result<T, String>) -> Result<T, ApiError> {
    // the shared loader keeps the model and negative locks in one order
    Ok(svc.with_negative(f)??)
}

/// `POST /api/images/tutor` and `POST /api/speech/tutor` once the texts are in
/// hand (`ModelService.recall_quiz`): quiz the running model, mark, and with
/// `blame` teach the negative network.
pub fn quiz_route(
    svc: &Arc<Service>,
    form: &Form,
    texts: Vec<String>,
    labels: Vec<String>,
    said: Vec<String>,
    modality: &str,
) -> Answer {
    let blame_it = form.flag("blame", false);
    let threshold = form
        .float("threshold", 6.0)
        .map_err(|_| ApiError::bad_request("'threshold' must be a number out of 10"))?;
    let mut options = quiz_options_api(form, modality)?;
    if texts.is_empty() {
        return Err(ApiError::bad_request(format!(
            "nothing to ask about: send a {modality} or 'texts' (already-encoded)"
        )));
    }
    if blame_it {
        svc.ensure_idle()?;
    }
    options.labels = labels;
    options.said = said;
    options.threshold = threshold;
    let lessons = svc
        .with_model(|m| quiz(m, &texts, &options))
        .map_err(ApiError::bad_request)?;
    let mut doc = Json::obj([
        ("modality", Json::str(modality)),
        ("lessons", Json::Arr(lessons.iter().map(|l| l.to_json()).collect())),
        ("report", report_card(&lessons)),
        ("negative", Json::Null),
    ]);
    if blame_it {
        let negative = with_negative(svc, |negative| {
            let report = teach_recall(negative, &lessons, threshold, true, modality, &TeachOptions::default())?;
            Ok(Json::obj([
                ("taught", taught_json(&report)),
                ("reasons", reasons_json(negative)),
                ("stats", crate::cli::negative_stats(negative)),
            ]))
        })?;
        speech::extend(&mut doc, vec![("negative", negative)]);
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::{pack_text, Codec};
    use crate::vision::pack_text as pack_image;

    const RATE: i64 = 8000;

    /// One short recording as the waveform text it trains on, and its token.
    fn utterance(freq: f64) -> (String, String) {
        let samples: Vec<f32> = (0..(RATE as f64 * 0.05) as usize)
            .map(|i| (0.6 * (2.0 * std::f64::consts::PI * freq * i as f64 / RATE as f64).sin()) as f32)
            .collect();
        let wav = speech::wav_bytes(&samples, RATE, 1);
        let taught = speech::teach(
            &wav,
            &speech::TeachOptions {
                transcript: "hello there".into(),
                rate: RATE,
                ..Default::default()
            },
        )
        .unwrap();
        (taught.waveform().unwrap().clone(), taught.token)
    }

    /// An image text with a repeatable payload (no decoding needed).
    fn picture() -> (String, usize) {
        let n = 3 * 8 * 8;
        let payload: Vec<u8> = (0..n).map(|i| ((i * 7 + 11) % 256) as u8).collect();
        (pack_image("tiny", 64, 64, &payload), n)
    }

    #[test]
    fn the_exercise_is_the_opening_of_a_text() {
        let (text, token) = utterance(220.0);
        assert_eq!(modality_of(&text).unwrap(), "speech");
        assert_eq!(cue_of(&text, None).unwrap(), format!("{token} aud:mu:8000x1:"));
        let (image, _) = picture();
        assert_eq!(modality_of(&image).unwrap(), "image");
        assert_eq!(cue_of(&image, None).unwrap().len(), "img:tiny:64x64:".len() + 16);
        assert_eq!(cue_of(&image, Some(0)).unwrap(), "img:tiny:64x64:");
        let short = pack_image("tiny", 64, 64, &[1, 2]);
        assert_eq!(
            cue_of(&short, Some(999)).unwrap(),
            short,
            "the cue never runs past the text"
        );
        assert!(modality_of("the cat sat on the mat").is_err());
        // a token outside ASCII is counted in characters, as Python counts it
        assert_eq!(
            cue_of("<speech:é> aud:mu:8000x1:AAAA", Some(2)).unwrap(),
            "<speech:é> aud:mu:8000x1:AA"
        );
    }

    #[test]
    fn exercises_skip_what_is_not_encoded() {
        let (text, _) = utterance(220.0);
        let (image, _) = picture();
        let texts = vec![text.clone(), "the cat sat".into(), String::new(), image];
        let labels: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let items = exercises(&texts, None, &labels);
        assert_eq!(items.len(), 2);
        assert_eq!((items[0].index, items[1].index), (0, 3));
        assert_eq!((items[0].label.as_str(), items[1].label.as_str()), ("a", "d"));
        assert_eq!(format!("{}{}", items[0].cue, items[0].answer()), text);
        assert_eq!(items[1].id, "r4");
    }

    #[test]
    fn a_recalled_waveform_is_marked() {
        let (text, token) = utterance(220.0);
        let truth = speech::parse_text(&text).unwrap().payload;
        let perfect = check_speech(&text, &text, "", "");
        assert_eq!((mark(&perfect), recall_reason(&perfect)), (10.0, "none"));
        let prose = check_speech(&format!("{token} the cat sat"), &text, "", "");
        assert!(!prose.readable);
        assert_eq!((mark(&prose), recall_reason(&prose)), (0.0, "unreadable"));
        let half = check_speech(&text[..text.len() / 2], &text, "", "");
        assert_eq!(recall_reason(&half), "truncated");
        assert!(mark(&half) < 6.0);
        let doubled: Vec<u8> = truth.iter().chain(truth.iter()).copied().collect();
        let long = check_speech(
            &format!("{token} {}", pack_text("mu", RATE, 1, &doubled)),
            &text,
            "",
            "",
        );
        assert_eq!(recall_reason(&long), "overrun");
        assert!((mark(&long) - 5.0).abs() < 0.1, "perfect bytes, half of them waste");
        let quiet = Codec::Mu.encode(&vec![0.0; truth.len()]);
        let silence = check_speech(&format!("{token} {}", pack_text("mu", RATE, 1, &quiet)), &text, "", "");
        assert!(silence.flat && !silence.reference_flat);
        assert_eq!(recall_reason(&silence), "silence");
        assert!(mark(&silence) <= 2.0);
        let railed: Vec<u8> = (0..truth.len())
            .map(|i| if (i / 8) % 2 == 0 { 255 } else { 0 })
            .collect();
        let clipping = check_speech(&format!("{token} {}", pack_text("mu", RATE, 1, &railed)), &text, "", "");
        assert!(clipping.extreme);
        assert_eq!(recall_reason(&clipping), "clipping");
        let wrong: Vec<u8> = truth.iter().map(|&b| ((b as u32 + 90) % 256) as u8).collect();
        let distorted = check_speech(&format!("{token} {}", pack_text("mu", RATE, 1, &wrong)), &text, "", "");
        assert_eq!(recall_reason(&distorted), "distortion");
    }

    #[test]
    fn a_mishearing_is_capped_and_silence_answered_by_silence_is_right() {
        let (text, _) = utterance(220.0);
        let heard = check_speech(&text, &text, "yellow bear", "hello there");
        assert_eq!(heard.matched, Some(false));
        assert_eq!(recall_reason(&heard), "mishearing");
        assert!(mark(&heard) <= 4.0);
        assert_eq!(
            check_speech(&text, &text, "hello  there", "Hello there").matched,
            Some(true)
        );
        assert_eq!(check_speech(&text, &text, "", "").matched, None);
        let quiet = pack_text("mu", RATE, 1, &Codec::Mu.encode(&[0.0; 200]));
        let facts = check_speech(&quiet, &quiet, "", "");
        assert!(facts.flat && facts.reference_flat);
        assert_eq!(recall_reason(&facts), "none");
    }

    #[test]
    fn a_recalled_image_is_marked() {
        let (text, size) = picture();
        let truth = crate::vision::parse_text(&text).unwrap().payload;
        assert_eq!(mark(&check_image(&text, &text)), 10.0);
        assert_eq!(recall_reason(&check_image("the cat sat", &text)), "unreadable");
        let flat = pack_image("tiny", 64, 64, &vec![128; size]);
        assert_eq!(recall_reason(&check_image(&flat, &text)), "blank");
        let noisy: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect();
        assert_eq!(
            recall_reason(&check_image(&pack_image("tiny", 64, 64, &noisy), &text)),
            "noise"
        );
        let wrong: Vec<u8> = truth.iter().map(|&b| ((b as u32 + 90) % 256) as u8).collect();
        assert_eq!(
            recall_reason(&check_image(&pack_image("tiny", 64, 64, &wrong), &text)),
            "drift"
        );
        assert_eq!(check_image(&text, &text).matched, None);
    }

    #[test]
    fn the_agreement_curve() {
        let truth: Vec<u8> = (0..64).collect();
        assert_eq!(agreement(&truth, &truth), 1.0);
        assert!((agreement(&truth[..32], &truth) - 0.5).abs() < 0.01);
        let near: Vec<u8> = truth.iter().map(|b| b + 1).collect();
        assert!(agreement(&near, &truth) >= 0.95);
        assert_eq!(agreement(&[200], &[168]), 0.0);
        assert_eq!(agreement(&[], &[]), 1.0);
        assert_eq!(agreement(b"abc", &[]), 0.0);
    }

    #[test]
    fn a_grade_carries_the_correction_only_on_a_failure() {
        let (text, _) = picture();
        let pass = grade(check_image(&text, &text), &text, 6.0);
        assert!(pass.passed && pass.error == "none" && pass.correction.is_empty() && pass.comment.is_empty());
        let fail = grade(check_image("img:tiny:64x64:AAAA", &text), &text, 6.0);
        assert!(!fail.passed);
        assert_eq!(
            (fail.error.as_str(), fail.correction.as_str()),
            ("truncated", text.as_str())
        );
        assert_eq!(fail.comment, "the image stops after 3 of 192 bytes (2% of it)");
    }

    #[test]
    fn a_quiz_marks_what_the_network_writes_back() {
        let (text, _) = utterance(220.0);
        let mut model = Model::new(7, crate::GraphOptions::default()).unwrap();
        model
            .train(
                std::slice::from_ref(&text),
                &crate::model::TrainOptions {
                    epochs: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let o = QuizOptions {
            length: 120,
            ..Default::default()
        };
        let lessons = quiz(&mut model, std::slice::from_ref(&text), &o).unwrap();
        assert_eq!(lessons.len(), 1);
        assert!(lessons[0].text.starts_with(&lessons[0].exercise.cue));
        // 120 base64 characters were asked for: 90 bytes
        assert_eq!(lessons[0].grade.facts.expected_bytes, 90);
        let doc = lessons[0].to_json();
        assert_eq!(doc.at("exercise").at("modality").as_str(), Some("speech"));
        assert_eq!(doc.at("sentence"), doc.at("text"));
        // a stopped quiz asks nothing, and every lesson is reported as it is marked
        let stop = || true;
        let stopped = QuizOptions {
            stop: Some(&stop),
            ..Default::default()
        };
        assert!(quiz(&mut model, std::slice::from_ref(&text), &stopped)
            .unwrap()
            .is_empty());
        let seen = std::cell::Cell::new(0);
        let progress = |_: &RecallLesson| seen.set(seen.get() + 1);
        let watched = QuizOptions {
            length: 40,
            attempts: 3,
            progress: Some(&progress),
            ..Default::default()
        };
        let lessons = quiz(&mut model, std::slice::from_ref(&text), &watched).unwrap();
        assert_eq!(seen.get(), lessons.len());
        assert!(lessons.len() <= 3);
        for (i, lesson) in lessons.iter().enumerate() {
            assert!(
                !lesson.grade.passed || i == lessons.len() - 1,
                "a pass ends the attempts"
            );
        }
    }

    #[test]
    fn the_report_card() {
        let empty = report_card(&[]);
        assert_eq!(empty.at("lessons").as_i64(), Some(0));
        assert!(empty.at("mean_score").is_null());
        assert_eq!(empty.at("modality").as_str(), Some(""));
        let (text, _) = picture();
        let lesson = |written: &str| RecallLesson {
            exercise: RecallExercise {
                id: "r1".into(),
                modality: "image".into(),
                reference: text.clone(),
                cue: String::new(),
                label: String::new(),
                index: 0,
            },
            attempt: 0,
            mode: "beam".into(),
            continuation: written.into(),
            text: written.into(),
            cost: 0.0,
            reached_end: false,
            seconds: 0.0,
            grade: grade(check_image(written, &text), &text, 6.0),
        };
        let card = report_card(&[lesson(&text), lesson("img:tiny:64x64:AAAA")]);
        assert_eq!(
            (card.at("lessons").as_i64(), card.at("passed").as_i64()),
            (Some(2), Some(1))
        );
        assert_eq!(card.at("reasons").at("truncated").as_i64(), Some(1));
        assert_eq!(card.at("modality").as_str(), Some("image"));
    }

    #[test]
    fn what_it_forgot_is_blamed_and_what_it_remembered_clears() {
        let (text, _) = picture();
        let truth = crate::vision::parse_text(&text).unwrap().payload;
        let half = truth.len() / 2;
        let mut wrong = truth[..half].to_vec();
        wrong.extend(truth[half..].iter().map(|&b| ((b as u32 + 90) % 256) as u8));
        let drifted = pack_image("tiny", 64, 64, &wrong);
        let lesson = RecallLesson {
            exercise: RecallExercise {
                id: "r1".into(),
                modality: "image".into(),
                reference: text.clone(),
                cue: String::new(),
                label: "a.png".into(),
                index: 0,
            },
            attempt: 0,
            mode: "beam".into(),
            continuation: String::new(),
            text: drifted.clone(),
            cost: 0.0,
            reached_end: false,
            seconds: 0.0,
            grade: grade(check_image(&drifted, &text), &text, 6.0),
        };
        let (faults, passed) = faults_from_recall(std::slice::from_ref(&lesson), 6.0, "recall");
        assert_eq!(faults.len(), 1);
        assert_eq!(
            (faults[0].reason.as_str(), faults[0].correction.as_str()),
            ("drift", text.as_str())
        );
        assert_eq!(passed, vec![text.clone()]);
        let mut negative = Model::new_negative(7, &Default::default()).unwrap();
        let report = teach_recall(&mut negative, &[lesson], 6.0, true, "vision", &TeachOptions::default()).unwrap();
        assert_eq!((report.blamed, report.source.as_str()), (1, "vision"));
        assert!(report.edges > 0);
        let judge = crate::negative::JudgeOptions::default();
        assert!(negative.judge(&drifted, &judge).map(|v| v.blame).unwrap_or(0.0) > 0.0);
        let opening: String = text.chars().take("img:tiny:64x64:".len() + 40).collect();
        assert_eq!(negative.judge(&opening, &judge).map(|v| v.blame).unwrap_or(0.0), 0.0);
        let reasons = reasons_json(&negative);
        assert_eq!(reasons.as_array()[0].at("reason").as_str(), Some("drift"));
        assert!(reasons.as_array()[0].at("edges").as_i64().unwrap() > 0);
    }
}
