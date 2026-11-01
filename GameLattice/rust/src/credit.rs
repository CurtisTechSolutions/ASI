//! Where a verdict lands (`DESIGN.md` §7.4, §7.5): sharp per step on one edge, dull discounted within a sensing
//! act, never across one; reafference masked to exactly zero; a baseline option that subtracts an edge's own
//! expectation.

use latticefsm::machine::Machine;

use crate::efference::Return;
use crate::verdict::Verdict;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreditSettings {
    /// `k`: the share of an edge's own `net` subtracted from a verdict before it lands (0: as LatticeFSM).
    pub baseline: f64,
}

impl Default for CreditSettings {
    fn default() -> CreditSettings {
        CreditSettings { baseline: 0.0 }
    }
}

/// Per-step amounts for the run on `lattice`, with reafferent steps masked to zero and the baseline applied.
pub fn amounts(lattice: &Machine, verdicts: &[Option<Verdict>], returns: &[Return], k: f64) -> Vec<f64> {
    let path = &lattice.path;
    let mut out = vec![0.0; path.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let Some(v) = verdicts.get(i).and_then(|v| v.as_ref()) else {
            continue;
        };
        if returns.get(i).copied() == Some(Return::Reafferent) {
            continue;
        }
        let mut a = v.amount;
        if k != 0.0 {
            a -= k * lattice.lattice.edges[path[i]].net();
        }
        *slot = a;
    }
    out
}

/// Land a set of verdicts on the current run: the sharp ones in full on their steps, the dull ones
/// discounted back from the run's end.  Returns how many edges were credited.
pub fn land(lattice: &mut Machine, verdicts: &[Verdict], returns: &[Return], settings: CreditSettings) -> usize {
    let steps = lattice.path.len();
    let mut per_step: Vec<Option<Verdict>> = vec![None; steps];
    let mut dull = 0.0;
    for v in verdicts {
        match (v.sharp, v.step) {
            (true, Some(i)) if i < steps => {
                let slot = &mut per_step[i];
                match slot {
                    Some(existing) => existing.amount += v.amount,
                    None => *slot = Some(v.clone()),
                }
            }
            _ => dull += v.amount,
        }
    }
    let mut n = 0;
    if per_step.iter().any(|v| v.is_some()) {
        let a = amounts(lattice, &per_step, returns, settings.baseline);
        n += lattice.credit_steps(&a);
    }
    if dull != 0.0 {
        n += lattice.credit(dull);
    }
    n
}
