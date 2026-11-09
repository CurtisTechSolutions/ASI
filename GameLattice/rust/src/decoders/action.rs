//! Propose-and-replay (`DESIGN.md` §6.3, §7.7): the repertoires propose, the selector releases one channel above
//! a reserve with a margin, the referee replays it; a refusal is credited at once and the ranking is walked.

use super::repertoire::SymbolDecoder;
use super::{Decoder, Output};
use crate::game::Payoff;
use crate::lattice::GameLattice;
use crate::select::{Release, Selection};
use crate::verdict::Verdict;

/// What a proposal came to.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    /// The accepted act's symbol id, or `None` (abstained, or every candidate refused).
    pub accepted: Option<usize>,
    pub refusals: Vec<Verdict>,
    /// Why nothing was released by the selector, when nothing was.
    pub abstained: Option<&'static str>,
    /// Whether the first candidate was the accepted one.
    pub first_try: bool,
    /// The proposer's distribution over the alphabet.
    pub ranking: Vec<f64>,
    /// The emission the proposal came from, so a grade can settle it later.
    pub emission: Option<super::repertoire::Emission>,
}

pub struct ActionDecoder {
    pub proposer: SymbolDecoder,
    pub release: Release,
    pub max_refusals: usize,
}

impl ActionDecoder {
    pub fn new(proposer: SymbolDecoder) -> ActionDecoder {
        ActionDecoder {
            proposer,
            release: Release::default(),
            max_refusals: 8,
        }
    }

    /// Propose on the lattice's current path and replay through `referee(symbol, step)`, which answers `None`
    /// when it accepts.  Every refusal lands on the proposer at once.
    pub fn propose(
        &mut self,
        lattice: &GameLattice,
        step: usize,
        mut referee: impl FnMut(usize, usize) -> Option<Verdict>,
    ) -> Proposal {
        let path = lattice.machine.path.clone();
        let e = self.proposer.emit(&path).clone();
        let ranking = e.aggregate.clone();
        let mut refusals = Vec::new();
        if e.symbol.is_none() {
            return Proposal {
                accepted: None,
                refusals,
                abstained: Some("no bidders"),
                first_try: false,
                ranking,
                emission: Some(e),
            };
        }
        let abstained = match self.release.select(&ranking) {
            Selection::Released(_) => None,
            Selection::PhotoFinish => Some("photo finish"),
            Selection::Abstain => Some("below the reserve"),
        };
        if abstained.is_some() {
            return Proposal {
                accepted: None,
                refusals,
                abstained,
                first_try: false,
                ranking,
                emission: Some(e),
            };
        }
        let mut order: Vec<usize> = (0..ranking.len()).collect();
        order.sort_by(|&a, &b| ranking[b].partial_cmp(&ranking[a]).unwrap().then(a.cmp(&b)));
        for (tries, &candidate) in order.iter().enumerate() {
            if tries > self.max_refusals {
                break;
            }
            match referee(candidate, step) {
                None => {
                    // the validity head: an accepted act is a right act, per step (DESIGN.md 7.4)
                    self.proposer.settle_emission(&e, Payoff::Correctness { y: candidate });
                    return Proposal {
                        accepted: Some(candidate),
                        refusals,
                        abstained: None,
                        first_try: tries == 0,
                        ranking,
                        emission: Some(e),
                    };
                }
                Some(v) => {
                    self.proposer.settle_emission(&e, Payoff::Refused { named: candidate });
                    refusals.push(v);
                }
            }
        }
        Proposal {
            accepted: None,
            refusals,
            abstained: Some("every candidate refused"),
            first_try: false,
            ranking,
            emission: Some(e),
        }
    }
}

impl Decoder for ActionDecoder {
    fn output_type(&self) -> &'static str {
        "action"
    }
    fn reads_back(&self) -> &str {
        self.proposer.reads_back()
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        match self.proposer.read(lattice) {
            Output::Symbols(s) => Output::Action(s.into_iter().next().unwrap_or_default()),
            other => other,
        }
    }
    fn credit(&mut self, verdict: &Verdict) -> usize {
        self.proposer.credit(verdict)
    }
}
