//! Referees (`DESIGN.md` §6.4): the world's rules, answering a proposed act with a verdict that has a reason,
//! never an exception.  Decoding an act is replay through one of these.
//!
//! Two referees here: the oracle of the four regular languages (`latticefsm::languages`), which judges a state
//! per prefix, and normal-play Nim on heaps of 3, 5 and 7 - `Experiments/UniversalGameEncoding`'s solved game,
//! ported so a move can be refused, replayed and graded without Python.

use latticefsm::languages::Language;
use latticefsm::rng::Rng;

use crate::verdict::{Source, Verdict};

/// The oracle for a language: is the state the organism named for this prefix right?  A state is right when
/// it accepts exactly when the prefix is in the language.
pub struct LanguageOracle {
    pub language: Language,
    pub reward: f64,
    pub punish: bool,
}

impl LanguageOracle {
    pub fn new(language: Language) -> LanguageOracle {
        LanguageOracle {
            language,
            reward: 1.0,
            punish: true,
        }
    }

    pub fn member(&self, prefix: &str) -> bool {
        (self.language.member)(prefix)
    }

    /// The verdict for step `i` (the prefix through the `i`-th character) landing in a state that `accepts`.
    pub fn judge(&self, prefix: &str, accepts: bool, step: usize) -> Verdict {
        if accepts == self.member(prefix) {
            Verdict::sharp(self.reward, "right", Source::Oracle, step)
        } else if self.punish {
            Verdict::sharp(-self.reward, "wrong", Source::Oracle, step)
        } else {
            Verdict::sharp(0.0, "wrong", Source::Oracle, step)
        }
    }
}

// ---- Nim ------------------------------------------------------------------------------------------------------

pub const NIM_HEAPS: [u8; 3] = [3, 5, 7];
pub const NIM_MAX: u8 = 7;

/// A Nim position: the heaps and who moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NimState {
    pub heaps: [u8; 3],
    pub mover: u8,
}

/// Take `n` from heap `h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NimMove {
    pub heap: u8,
    pub take: u8,
}

impl NimMove {
    /// `h * NIM_MAX + (n - 1)`: the action index of `uge.games.Nim`.
    pub fn index(self) -> usize {
        self.heap as usize * NIM_MAX as usize + (self.take as usize - 1)
    }

    pub fn from_index(i: usize) -> NimMove {
        NimMove {
            heap: (i / NIM_MAX as usize) as u8,
            take: (i % NIM_MAX as usize) as u8 + 1,
        }
    }

    /// The symbol a lattice reads for it: `h<heap>t<take>`.
    pub fn symbol(self) -> String {
        format!("h{}t{}", self.heap, self.take)
    }
}

pub struct Nim;

impl Nim {
    pub fn action_count() -> usize {
        NIM_HEAPS.len() * NIM_MAX as usize
    }

    /// Every move's symbol, in action-index order: the act lattice's alphabet.
    pub fn alphabet() -> Vec<String> {
        (0..Self::action_count())
            .map(|i| NimMove::from_index(i).symbol())
            .collect()
    }

    pub fn initial() -> NimState {
        NimState {
            heaps: NIM_HEAPS,
            mover: 0,
        }
    }

    pub fn legal(s: NimState) -> Vec<NimMove> {
        let mut out = Vec::new();
        for (h, &size) in s.heaps.iter().enumerate() {
            for n in 1..=size {
                out.push(NimMove { heap: h as u8, take: n });
            }
        }
        out
    }

    pub fn is_legal(s: NimState, m: NimMove) -> bool {
        (m.heap as usize) < s.heaps.len() && m.take >= 1 && m.take <= s.heaps[m.heap as usize]
    }

    pub fn apply(s: NimState, m: NimMove) -> NimState {
        let mut heaps = s.heaps;
        heaps[m.heap as usize] -= m.take;
        NimState {
            heaps,
            mover: 1 - s.mover,
        }
    }

    pub fn is_terminal(s: NimState) -> bool {
        s.heaps.iter().all(|&h| h == 0)
    }

    /// The player who took the last object has just moved.
    pub fn winner(s: NimState) -> Option<u8> {
        if Self::is_terminal(s) {
            Some(1 - s.mover)
        } else {
            None
        }
    }

    pub fn nim_sum(s: NimState) -> u8 {
        s.heaps.iter().fold(0, |x, &h| x ^ h)
    }

    /// Every move that leaves a xor of zero; all of them when the position is already lost.
    pub fn optimal(s: NimState) -> Vec<NimMove> {
        let legal = Self::legal(s);
        let best: Vec<NimMove> = legal
            .iter()
            .cloned()
            .filter(|&m| Self::nim_sum(Self::apply(s, m)) == 0)
            .collect();
        if best.is_empty() {
            legal
        } else {
            best
        }
    }

    /// `uge.games.Nim.state_code`: the heaps in mixed radix times the mover, 384 codes.
    pub fn state_code(s: NimState) -> usize {
        let mut code = 0usize;
        for (size, limit) in s.heaps.iter().rev().zip(NIM_HEAPS.iter().rev()) {
            code = code * (*limit as usize + 1) + *size as usize;
        }
        code * 2 + s.mover as usize
    }

    pub fn state_codes() -> usize {
        NIM_HEAPS.iter().map(|&l| l as usize + 1).product::<usize>() * 2
    }

    pub fn random_move(s: NimState, rng: &mut Rng) -> Option<NimMove> {
        let legal = Self::legal(s);
        if legal.is_empty() {
            None
        } else {
            Some(*rng.choice(&legal))
        }
    }

    /// The referee: a move's verdict at `step`, `None` when accepted.
    pub fn refuse(s: NimState, m: NimMove, step: usize) -> Option<Verdict> {
        if Self::is_terminal(s) || !Self::is_legal(s, m) {
            Some(Verdict::refusal("illegal", 1.0, step))
        } else {
            None
        }
    }
}
