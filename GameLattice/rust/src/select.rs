//! Action selection (`DESIGN.md` §6.3, §10.4): the default is no; one channel is released above a reserve with
//! a margin, else abstain.  A distribution is a belief; a release is a move.

use crate::game::margin;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Release {
    pub reserve: f64,
    pub margin_tau: f64,
}

impl Default for Release {
    fn default() -> Release {
        Release {
            reserve: 0.01,
            margin_tau: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Released(usize),
    /// A photo finish: the top two are too close.
    PhotoFinish,
    /// Nothing cleared the reserve.
    Abstain,
}

impl Release {
    /// The released channel of a distribution over candidates, or why none was.
    pub fn select(&self, p: &[f64]) -> Selection {
        if p.is_empty() {
            return Selection::Abstain;
        }
        let mut best = 0;
        for (i, x) in p.iter().enumerate() {
            if *x > p[best] {
                best = i;
            }
        }
        if p[best] < self.reserve {
            return Selection::Abstain;
        }
        if p.len() > 1 && margin(p) < self.margin_tau {
            return Selection::PhotoFinish;
        }
        Selection::Released(best)
    }
}
