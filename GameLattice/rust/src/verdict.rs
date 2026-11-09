//! The one record for every source of credit in the organism (`DESIGN.md` §6.1, §7.4).
//!
//! A verdict is *sharp* when it knows which step it is about - the referee's refusal, the oracle's answer for a
//! prefix, the taste gate's line on one bite - and lands in full on that step's edge, on no other
//! (`Machine::credit_steps`).  It is *dull* when it is about a whole run - a teacher's mark, a benchmark hours
//! later - and lands discounted back along the run (`Machine::credit`), never across a sensing act's boundary.

use latticefsm::json::Json;

/// Who said so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The world's rules: a game's referee, the sandbox, a tool (free, instant, complete).
    Referee,
    /// An oracle that knows each step's right answer (the languages' membership function).
    Oracle,
    /// The taste gate: an innate line on a bite about to be consumed.
    Taste,
    /// A teacher's mark on a whole reply.
    Teacher,
    /// The body's own ledgers: wealth, the clearing price, decay.
    Intero,
    /// The forward model's disagreement with what the world returned (§8.2): a lesson, never credit.
    Forward,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Referee => "referee",
            Source::Oracle => "oracle",
            Source::Taste => "taste",
            Source::Teacher => "teacher",
            Source::Intero => "intero",
            Source::Forward => "forward",
        }
    }
}

/// One verdict.  `amount` is signed: a reward positive, a punishment negative, zero nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub amount: f64,
    /// `blame.py`'s taxonomy where it applies: `illegal`, `phi`, `syntax`, `timeout`, `unspecified`, ...
    pub reason: String,
    /// Per step (lands in full on `step`'s edge) or dull (discounted back along the run).
    pub sharp: bool,
    pub source: Source,
    /// What was consumed, for the identity-addressed path (§7.4): a smell signature, a path.  `None` mostly.
    pub origin: Option<String>,
    /// Which step of the sensing act a sharp verdict is about.
    pub step: Option<usize>,
}

impl Verdict {
    pub fn sharp(amount: f64, reason: &str, source: Source, step: usize) -> Verdict {
        Verdict {
            amount,
            reason: reason.to_string(),
            sharp: true,
            source,
            origin: None,
            step: Some(step),
        }
    }

    pub fn dull(amount: f64, reason: &str, source: Source) -> Verdict {
        Verdict {
            amount,
            reason: reason.to_string(),
            sharp: false,
            source,
            origin: None,
            step: None,
        }
    }

    /// A refusal with a reason: sharp, negative, from the referee.  `severity` is positive.
    pub fn refusal(reason: &str, severity: f64, step: usize) -> Verdict {
        Verdict::sharp(-severity.abs(), reason, Source::Referee, step)
    }

    /// Whether the reason is about the *belief* (what state the organism thought it was in) rather than the
    /// *spelling* of the act: only such a refusal reaches the encoder's game (§4.4).
    pub fn about_belief(&self) -> bool {
        matches!(self.reason.as_str(), "illegal" | "phi" | "wrong-output" | "wrong")
    }

    pub fn to_json(&self) -> Json {
        let mut j = Json::object();
        j.set("amount", Json::Number(self.amount));
        j.set("reason", Json::String(self.reason.clone()));
        j.set("sharp", Json::Bool(self.sharp));
        j.set("source", Json::String(self.source.name().to_string()));
        if let Some(o) = &self.origin {
            j.set("origin", Json::String(o.clone()));
        }
        if let Some(s) = self.step {
            j.set("step", Json::Number(s as f64));
        }
        j
    }
}
