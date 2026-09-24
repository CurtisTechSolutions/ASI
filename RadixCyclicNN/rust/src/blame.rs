//! Where the negative network's data comes from: the verdicts of whatever
//! looked at an output and said it was wrong, turned into blame
//! (`radixnet/blame.py`, `go/radixnet/blame.go`).
//!
//! Nothing in the negative network is invented.  A [`Fault`] is a text, a
//! reason (the tutor's own error type, or one picked from its words by
//! [`classify`]), a severity (how badly it failed) and a note (the tutor's own
//! sentence, kept verbatim for the journal).  [`teach`] hands faults to the
//! negative network; the texts the tutor passed are cleared in the same call,
//! so a fragment that shows up in good and bad output alike stops carrying the
//! verdict on its own.
//!
//! This module is the core every tutor shares.  Each tutor turns its own
//! records into faults beside its own types: the review in `review.rs`, the
//! English lessons in `tutor.rs`, program attempts in `codegen.rs`, tool use
//! in `agent.rs`, recall in `recall.rs`.

use crate::json::Json;
use crate::model::{EpochRecord, Model};
use crate::negative::BlameOptions;

/// Recorded when nothing in the tutor's words is recognised.
pub const DEFAULT_REASON: &str = "other";

/// Reason tags for reviewed *text* (the order [`classify`] tries them in).
pub const REASONS: &[&str] = &[
    "empty",
    "gibberish",
    "repetition",
    "truncated",
    "grammar",
    "spelling",
    "contradiction",
    "false",
    "incoherent",
    "off-topic",
    DEFAULT_REASON,
];

/// Reason tags for reviewed *programs*.
pub const CODE_REASONS: &[&str] = &[
    "timeout",
    "crash",
    "wrong-output",
    "task-not-done",
    "style",
    "naming",
    DEFAULT_REASON,
];

/// How heavily each code failure is blamed (1 = one ordinary failure).
pub const CODE_SEVERITY: &[(&str, f64)] = &[
    ("timeout", 1.5),
    ("crash", 1.5),
    ("wrong-output", 1.25),
    ("task-not-done", 1.0),
    ("style", 0.5),
    ("naming", 0.5),
    (DEFAULT_REASON, 1.0),
];

/// Reason tags for an attempt at a task with *tools* (the order the agent's
/// reason is tried in).
pub const AGENT_REASONS: &[&str] = &["no-call", "bad-call", "tool-error", "no-answer", DEFAULT_REASON];

/// How heavily each tool-use failure is blamed.
pub const AGENT_SEVERITY: &[(&str, f64)] = &[
    ("no-call", 1.0),
    // one step of one attempt, not the whole attempt
    ("bad-call", 0.75),
    ("tool-error", 1.0),
    ("no-answer", 1.5),
    (DEFAULT_REASON, 1.0),
];

/// Reason tags for a waveform the network was asked to remember.
pub const SPEECH_REASONS: &[&str] = &[
    "unreadable",
    "truncated",
    "overrun",
    "garbled",
    "silence",
    "clipping",
    "mishearing",
    "distortion",
];

/// Reason tags for an image the network was asked to remember.
pub const IMAGE_REASONS: &[&str] = &[
    "unreadable",
    "truncated",
    "overrun",
    "garbled",
    "blank",
    "noise",
    "drift",
];

/// Payload length ratios outside which a recalled text is short or long
/// rather than merely wrong.
pub const RECALL_TRUNCATED: f64 = 0.9;
pub const RECALL_OVERRUN: f64 = 1.1;

/// Below this payload agreement a readable, right-length recall is still a
/// distortion (an image: a drift).
pub const RECALL_AGREEMENT: f64 = 0.9;

const PATTERNS: &[(&str, &[&str])] = &[
    ("empty", &["empty output", "no output", "produced nothing", "blank"]),
    (
        "gibberish",
        &[
            "gibberish",
            "nonsense word",
            "word salad",
            "garbled",
            "random character",
            "not words",
            "noise",
        ],
    ),
    (
        "repetition",
        &["repeat", "repetit", "duplicat", "over and over", "loops"],
    ),
    (
        "truncated",
        &[
            "truncat",
            "cut off",
            "cut short",
            "incomplete",
            "unfinished",
            "mid-sentence",
            "mid sentence",
        ],
    ),
    (
        "grammar",
        &[
            "grammar",
            "grammatic",
            "syntax",
            "word order",
            "punctuation",
            "agreement",
            "tense",
            "malformed sentence",
        ],
    ),
    ("spelling", &["spelling", "misspell", "typo"]),
    ("contradiction", &["contradict", "inconsisten", "conflicts with"]),
    (
        "false",
        &[
            "false",
            "factual",
            "inaccurate",
            "untrue",
            "wrong fact",
            "not true",
            "misleading",
        ],
    ),
    (
        "incoherent",
        &[
            "incoheren",
            "meaningless",
            "make no sense",
            "does not make sense",
            "doesn't make sense",
            "nonsensical",
            "confusing",
            "unintelligible",
        ],
    ),
    (
        "off-topic",
        &[
            "off-topic",
            "off topic",
            "irrelevant",
            "unrelated",
            "does not answer",
            "ignores the prompt",
        ],
    ),
];

/// The reason tag behind a tutor's critique: its own words decide, `default`
/// (else [`DEFAULT_REASON`]) covers what the vocabulary does not know.
///
/// `"unrated"` comes back when the tutor failed a text without a critique the
/// vocabulary recognises - it still failed, and the network records that
/// nobody said why.
pub fn classify(critique: &str, verdict: &str, rating: Option<f64>, default: &str) -> String {
    let text = critique.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    for (reason, needles) in PATTERNS {
        if needles.iter().any(|n| text.contains(n)) {
            return reason.to_string();
        }
    }
    if verdict.trim().to_lowercase() == "unrated" {
        return "unrated".to_string();
    }
    if rating.is_some_and(|r| r <= 0.0) && text.is_empty() {
        return "gibberish".to_string();
    }
    if default.is_empty() {
        DEFAULT_REASON.to_string()
    } else {
        default.to_string()
    }
}

/// How heavily a rated failure is blamed: 2.0 at rating 0, 0.25 at the pass
/// threshold; an unrated failure (`None`) is blamed like one ordinary failure.
pub fn severity_from_rating(rating: Option<f64>, threshold: f64) -> f64 {
    const FLOOR: f64 = 0.25;
    const CEILING: f64 = 2.0;
    let Some(rating) = rating else { return 1.0 };
    let limit = if threshold > 0.0 { threshold } else { 1.0 };
    let share = ((limit - rating) / limit).clamp(0.0, 1.0);
    FLOOR + (CEILING - FLOOR) * share
}

/// How heavily a failure is blamed from how badly it failed (`gap` in
/// `[0, 1]`), on the scale [`severity_from_rating`] maps ratings onto.
pub fn severity_from_gap(gap: Option<f64>) -> f64 {
    const FLOOR: f64 = 0.25;
    const CEILING: f64 = 2.0;
    match gap {
        None => 1.0,
        Some(g) => FLOOR + (CEILING - FLOOR) * g.clamp(0.0, 1.0),
    }
}

/// The severity a table like [`CODE_SEVERITY`] gives a reason (1 when absent).
pub fn severity_of(table: &[(&str, f64)], reason: &str) -> f64 {
    table.iter().find(|(r, _)| *r == reason).map(|(_, s)| *s).unwrap_or(1.0)
}

/// One lesson for the negative network.
#[derive(Clone, Debug, PartialEq)]
pub struct Fault {
    pub text: String,
    pub reason: String,
    pub severity: f64,
    pub note: String,
    pub source: String,
    /// The sentence written out correctly, when the tutor gave one: then only
    /// what it changed is blamed.
    pub correction: String,
}

impl Fault {
    pub fn new(text: &str, reason: &str, severity: f64, note: &str, source: &str) -> Fault {
        Fault {
            text: text.to_string(),
            reason: reason.to_string(),
            severity,
            note: note.to_string(),
            source: source.to_string(),
            correction: String::new(),
        }
    }

    /// `{"text", "reason", "severity", "note", "source"}` (+ `correction`).
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("text".to_string(), Json::str(self.text.clone())),
            ("reason".to_string(), Json::str(self.reason.clone())),
            ("severity".to_string(), Json::Num(self.severity)),
            ("note".to_string(), Json::str(self.note.clone())),
            ("source".to_string(), Json::str(self.source.clone())),
        ];
        if !self.correction.is_empty() {
            pairs.push(("correction".to_string(), Json::str(self.correction.clone())));
        }
        Json::Obj(pairs)
    }
}

/// What one round of blaming taught the negative network.
#[derive(Clone, Debug, Default)]
pub struct TeachReport {
    /// Faults blamed.
    pub blamed: usize,
    /// Passed texts that cleared something.
    pub cleared: usize,
    /// Passed texts the failure structure could not walk.
    pub unmatched: usize,
    /// Edges the blame touched.
    pub edges: usize,
    /// How many faults each reason had, in the order first met.
    pub reasons: Vec<(String, usize)>,
    pub severity_mean: f64,
    /// Every record: blame and clearing epochs, and correction outcomes.
    pub records: Vec<Json>,
    /// Filled in by the tutor that built the faults.
    pub source: String,
    pub threshold: Option<f64>,
    pub faults: Vec<Fault>,
    pub passed: usize,
}

impl TeachReport {
    /// `{"blamed", "cleared", "unmatched", "edges", "reasons",
    /// "severity_mean", "records"}` and, when a tutor filled them in,
    /// `source`, `threshold`, `faults` and `passed`.
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("blamed".to_string(), Json::Int(self.blamed as i64)),
            ("cleared".to_string(), Json::Int(self.cleared as i64)),
            ("unmatched".to_string(), Json::Int(self.unmatched as i64)),
            ("edges".to_string(), Json::Int(self.edges as i64)),
            (
                "reasons".to_string(),
                Json::Obj(
                    self.reasons
                        .iter()
                        .map(|(r, n)| (r.clone(), Json::Int(*n as i64)))
                        .collect(),
                ),
            ),
            ("severity_mean".to_string(), Json::Num(self.severity_mean)),
            ("records".to_string(), Json::Arr(self.records.clone())),
        ];
        if !self.source.is_empty() {
            pairs.push(("source".to_string(), Json::str(self.source.clone())));
            pairs.push((
                "threshold".to_string(),
                self.threshold.map(Json::Num).unwrap_or(Json::Null),
            ));
            pairs.push((
                "faults".to_string(),
                Json::Arr(self.faults.iter().map(|f| f.to_json()).collect()),
            ));
            pairs.push(("passed".to_string(), Json::Int(self.passed as i64)));
        }
        Json::Obj(pairs)
    }
}

/// How one [`teach`] call runs.
pub struct TeachOptions<'a> {
    /// Blame passes per fault (0 = one).
    pub epochs: usize,
    /// Clearing passes over the passed texts (0 = one).
    pub clear_epochs: usize,
    /// Asked between faults and after every epoch; `true` stops the round.
    pub stop: Option<&'a (dyn Fn() -> bool + Sync)>,
}

impl Default for TeachOptions<'_> {
    fn default() -> Self {
        TeachOptions {
            epochs: 1,
            clear_epochs: 1,
            stop: None,
        }
    }
}

impl TeachOptions<'_> {
    fn stopped(&self) -> bool {
        self.stop.is_some_and(|stop| stop())
    }
}

/// Hands the tutor's faults to the negative network; the `passed` texts clear
/// blame afterwards.  A fault that carries a correction is taught from its
/// diff, so only the units the teacher changed are blamed.
pub fn teach(
    negative: &mut Model,
    faults: &[Fault],
    passed: &[String],
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    if !negative.is_negative() {
        return Err(format!(
            "teaching needs the negative network (this is the {} model)",
            negative.kind()
        ));
    }
    let mut report = TeachReport::default();
    let mut severities: Vec<f64> = Vec::new();
    let count_reason =
        |reasons: &mut Vec<(String, usize)>, reason: &str| match reasons.iter_mut().find(|(r, _)| r == reason) {
            Some((_, n)) => *n += 1,
            None => reasons.push((reason.to_string(), 1)),
        };
    let record_json = |r: &EpochRecord| r.to_json();
    for fault in faults {
        if o.stopped() {
            break;
        }
        if fault.text.is_empty() {
            continue;
        }
        let reason = if fault.reason.is_empty() {
            DEFAULT_REASON.to_string()
        } else {
            fault.reason.clone()
        };
        let source = if fault.source.is_empty() {
            "tutor".to_string()
        } else {
            fault.source.clone()
        };
        let opts = BlameOptions {
            reason: reason.clone(),
            severity: fault.severity,
            source,
            note: fault.note.clone(),
            epochs: o.epochs,
            no_compress: false,
        };
        if !fault.correction.is_empty() {
            // the tutor wrote the sentence out correctly: blame only the units it changed
            let outcome = negative.blame_correction(&fault.text, &fault.correction, &opts, 1.0)?;
            report.edges += outcome.blamed;
            report.records.push(outcome.to_json());
        } else {
            let stop = o.stop;
            let records = negative.blame_with(std::slice::from_ref(&fault.text), &opts, &mut |_| {
                !stop.is_some_and(|s| s())
            })?;
            if let Some(last) = records.last() {
                report.edges += last.extra_value("edges_touched").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            }
            report.records.extend(records.iter().map(record_json));
        }
        count_reason(&mut report.reasons, &reason);
        severities.push(fault.severity);
    }
    let texts: Vec<String> = passed.iter().filter(|t| !t.is_empty()).cloned().collect();
    if !texts.is_empty() && !o.stopped() {
        let stop = o.stop;
        let records = negative.clear_with(&texts, 1.0, o.clear_epochs, &mut |_| !stop.is_some_and(|s| s()))?;
        if let Some(last) = records.last() {
            report.cleared = last.extra_value("matched").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            report.unmatched = last.extra_value("unmatched").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
        }
        report.records.extend(records.iter().map(record_json));
    }
    report.blamed = severities.len();
    report.severity_mean = if severities.is_empty() {
        0.0
    } else {
        severities.iter().sum::<f64>() / severities.len() as f64
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::negative::NegativeOptions;

    #[test]
    fn the_critique_names_the_reason() {
        assert_eq!(classify("This is word salad.", "", None, ""), "gibberish");
        assert_eq!(classify("wrong TENSE here", "", None, ""), "grammar");
        assert_eq!(classify("", "unrated", None, ""), "unrated");
        assert_eq!(classify("", "", Some(0.0), ""), "gibberish");
        assert_eq!(classify("meh", "", Some(3.0), "thumbs-down"), "thumbs-down");
        assert_eq!(classify("meh", "", None, ""), DEFAULT_REASON);
    }

    #[test]
    fn a_worse_mark_is_a_heavier_blame() {
        assert_eq!(severity_from_rating(None, 6.0), 1.0);
        assert_eq!(severity_from_rating(Some(0.0), 6.0), 2.0);
        assert_eq!(severity_from_rating(Some(6.0), 6.0), 0.25);
        assert!(severity_from_rating(Some(2.0), 6.0) > severity_from_rating(Some(4.0), 6.0));
        assert_eq!(severity_from_gap(Some(1.0)), 2.0);
        assert_eq!(severity_of(CODE_SEVERITY, "crash"), 1.5);
        assert_eq!(severity_of(CODE_SEVERITY, "nope"), 1.0);
    }

    #[test]
    fn faults_blame_and_passes_clear() {
        let mut neg = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let mut corrected = Fault::new("he go to school", "grammar", 1.5, "go -> goes", "tutor");
        corrected.correction = "he goes to school".to_string();
        let faults = vec![
            Fault::new("the cat sat on the sky", "false", 1.0, "", "review"),
            corrected,
            Fault::new("", "empty", 1.0, "", "review"),
        ];
        let passed = vec!["the cat sat on the mat".to_string()];
        let report = teach(&mut neg, &faults, &passed, &TeachOptions::default()).unwrap();
        assert_eq!(report.blamed, 2, "an empty text is not a fault");
        assert!(report.edges > 0);
        assert_eq!(
            report.reasons,
            vec![("false".to_string(), 1), ("grammar".to_string(), 1)]
        );
        assert_eq!(report.cleared + report.unmatched, 1);
        assert!(
            report.cleared == 1,
            "the pass shares 'the cat sat on the' with a failure"
        );
        assert!((report.severity_mean - 1.25).abs() < 1e-12);
        let doc = report.to_json();
        assert_eq!(doc.at("blamed").as_i64(), Some(2));
        // the correction blamed the changed word, not the whole sentence
        let correction = report
            .records
            .iter()
            .find(|r| r.at("phase").as_str() == Some("correction"))
            .unwrap();
        assert!(correction.at("blamed").as_i64().unwrap() > 0);
        assert_eq!(correction.at("edits").as_i64(), Some(1));
    }

    #[test]
    fn a_positive_model_cannot_be_taught_failures() {
        let mut m = Model::new(0, crate::GraphOptions::default()).unwrap();
        assert!(teach(&mut m, &[], &[], &TeachOptions::default()).is_err());
    }
}
