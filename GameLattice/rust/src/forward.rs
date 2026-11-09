//! The forward model (`DESIGN.md` §5.1, §8.2): an act lattice's row `(s, a) -> t` read quietly as a
//! prediction, and taught only by the periphery - a lesson on the row when what returned differs, never credit
//! for agreement.

use latticefsm::machine::Machine;

/// `(t, p)`: the row's greedy prediction and the probability it has.
pub fn predict(lattice: &Machine, s: usize, a: usize) -> (usize, f64) {
    let p = lattice.probabilities(s, a, None, Some(0.0));
    let t = latticefsm::machine::argmax(&p);
    (t, p[t])
}

/// The row's distribution at the machine's temperature.
pub fn distribution(lattice: &Machine, s: usize, a: usize) -> Vec<f64> {
    lattice.probabilities(s, a, None, None)
}

/// A lesson: the predicted edge is punished `amount` and the actual one taught `amount`.  Nothing when they
/// agree - a predicted sensation carries no credit (`DESIGN.md` §8.2).  Returns whether a lesson was given.
pub fn learn(
    lattice: &mut Machine,
    s: usize,
    a: usize,
    predicted: usize,
    actual: usize,
    amount: f64,
) -> Result<bool, String> {
    if predicted == actual {
        return Ok(false);
    }
    lattice.teach(s, a, predicted, -amount.abs())?;
    lattice.teach(s, a, actual, amount.abs())?;
    Ok(true)
}

/// The share of steps whose prediction agreed with what came: the collapse guard reads it beside
/// `GameGraph::diversity`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Agreement {
    pub agreed: u64,
    pub steps: u64,
}

impl Agreement {
    pub fn note(&mut self, agreed: bool) {
        self.steps += 1;
        if agreed {
            self.agreed += 1;
        }
    }

    pub fn rate(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.agreed as f64 / self.steps as f64
        }
    }
}
