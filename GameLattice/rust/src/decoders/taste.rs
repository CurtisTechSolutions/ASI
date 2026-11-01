//! The taste gate (`DESIGN.md` §6.3, §9.2): five innate lines with fixed signs, executed in a fixed order before
//! anything is learned, said or acted on; its thresholds are body state.  Not a stage game.

use super::{Decoder, Output};
use crate::lattice::GameLattice;
use crate::verdict::{Source, Verdict};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    Empty,
    Unreadable,
    Loop,
    KnownFailure,
    Clean,
}

impl Line {
    pub fn name(self) -> &'static str {
        match self {
            Line::Empty => "empty",
            Line::Unreadable => "unreadable",
            Line::Loop => "loop",
            Line::KnownFailure => "known-failure",
            Line::Clean => "clean",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swallow {
    Swallow,
    Caution,
    Spit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bite {
    pub line: Line,
    /// 1 to 3.
    pub intensity: u8,
    /// `+` clean, `0` caution, `-` reject, times the intensity.
    pub valence: f64,
    pub gate: Swallow,
    pub why: String,
}

pub struct Taste {
    pub sense: String,
    /// Substrings a bite is spat for: what the guard blamed.
    pub blamed: Vec<String>,
    /// The share of characters that must be printable.
    pub readable: f64,
    /// A stretch of at least this many characters coming back is a loop.
    pub loop_min: usize,
    pub last: Option<Bite>,
}

impl Taste {
    pub fn new(sense: &str) -> Taste {
        Taste {
            sense: sense.to_string(),
            blamed: Vec::new(),
            readable: 0.9,
            loop_min: 12,
            last: None,
        }
    }

    /// A stretch of at least `loop_min` characters that recurs, having covered at least half its period
    /// (`latticefsm::autotemp`'s rule).
    fn loops(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n < 2 * self.loop_min {
            return false;
        }
        for p in self.loop_min..=n / 2 {
            let mut r = 0;
            for i in p..n {
                if chars[i] == chars[i - p] {
                    r += 1;
                    if r >= self.loop_min.max(p / 2) {
                        return true;
                    }
                } else {
                    r = 0;
                }
            }
        }
        false
    }

    pub fn taste(&mut self, unit: &str) -> Bite {
        let trimmed = unit.trim();
        let bite = if trimmed.is_empty() {
            Bite {
                line: Line::Empty,
                intensity: 3,
                valence: -3.0,
                gate: Swallow::Spit,
                why: "nothing to swallow".to_string(),
            }
        } else {
            let total = trimmed.chars().count() as f64;
            let printable = trimmed.chars().filter(|c| !c.is_control() || c.is_whitespace()).count() as f64;
            if printable / total < self.readable {
                let intensity = if printable / total < 0.5 { 3 } else { 2 };
                Bite {
                    line: Line::Unreadable,
                    intensity,
                    valence: -(intensity as f64),
                    gate: Swallow::Spit,
                    why: format!("{:.0}% unreadable", 100.0 * (1.0 - printable / total)),
                }
            } else if let Some(b) = self.blamed.iter().find(|b| trimmed.contains(b.as_str())) {
                Bite {
                    line: Line::KnownFailure,
                    intensity: 2,
                    valence: -2.0,
                    gate: Swallow::Spit,
                    why: format!("contains the blamed {b:?}"),
                }
            } else if self.loops(trimmed) {
                Bite {
                    line: Line::Loop,
                    intensity: 1,
                    valence: 0.0,
                    gate: Swallow::Caution,
                    why: "goes round".to_string(),
                }
            } else {
                Bite {
                    line: Line::Clean,
                    intensity: 1,
                    valence: 0.25,
                    gate: Swallow::Swallow,
                    why: "clean".to_string(),
                }
            }
        };
        self.last = Some(bite.clone());
        bite
    }

    /// The bite as a verdict on the step that consumed it.
    pub fn verdict(bite: &Bite, step: usize) -> Verdict {
        Verdict::sharp(bite.valence, bite.line.name(), Source::Taste, step)
    }
}

impl Decoder for Taste {
    fn output_type(&self) -> &'static str {
        "verdict"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, _: &GameLattice) -> Output {
        match &self.last {
            Some(b) => Output::Verdict(b.gate == Swallow::Swallow),
            None => Output::Abstain("nothing tasted".to_string()),
        }
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}
