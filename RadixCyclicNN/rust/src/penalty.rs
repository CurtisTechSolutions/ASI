//! The punishment traversal: a walk priced by what the network was punished
//! for, not by what it was rewarded for.
//!
//! Every search in this crate reads the graph through one funnel:
//! `[(child, edge, cost)]` for a node, with `cost = -log P(child | parent)`.
//! **Which cost function fills that list is the traversal**, and there are two
//! of them here:
//!
//! `reward` (the default)
//! : what the model believes.  The count / reward model's probability carries
//!   `exp(reward_scale * reward)`, so a path the tutor rewarded is cheap and
//!   the search *follows the rewards*.
//!
//! `punishment`
//! : what the model was punished for.  The rewards are taken out of the score
//!   altogether and only the **penalties** price the step, so the cheapest path
//!   is the one that accumulated the **least punishment**.
//!
//! The traversal rests on one split, [`Graph::child_evidence`], which takes an
//! edge's evidence apart:
//!
//! * `merit` - what speaks *for* the step with every reward taken out of it:
//!   frequency and structure, what the corpus did rather than what a judge said;
//! * `penalty >= 0` - what speaks *against* it, the punishment the edge carries.
//!
//! The step's score is `merit_scale * merit - penalty_scale * penalty` and the
//! cost the usual `-log softmax` over the parent's children, so costs stay
//! non-negative (the search keeps working), `exp(-cost)` is still a path's
//! probability, and the numbers stay comparable with the reward traversal's.
//! `merit_scale = 0` is the pure form: nothing but the punishment decides.
//!
//! This mirrors `radixnet/penalty.py` and `go/radixnet/penalty.go` exactly.
//!
//! The third name a search answers to, `least-punished`, is **not** here: it is
//! a *ranking* rather than a cost function, and it lives in
//! [`crate::search`] (`../../SPEC-LeastPunished.md`).  The two are different
//! currencies and are never composed; [`traversal_costs`] hands back `None` for
//! it, exactly as it does for `reward`.

use crate::fsum::fsum;
use crate::graph::{Graph, Transition};
use crate::hash::{map, Map};
use crate::paths::PathKey;
use crate::weights::ChildCost;

/// The traversals a search can run, in the order the CLI and the API offer them.
pub const TRAVERSALS: [&str; 3] = [REWARD, PUNISHMENT, LEAST_PUNISHED];

/// What the model believes: the cost function every release has had.
pub const REWARD: &str = "reward";
/// The same graph priced by the punishments alone.
pub const PUNISHMENT: &str = "punishment";
/// The ranking traversal of [`crate::search`], named here so one flag offers all three.
pub const LEAST_PUNISHED: &str = "least-punished";
/// What every search runs unless told otherwise.
pub const DEFAULT_TRAVERSAL: &str = REWARD;

/// Normalises a traversal name (`""` = the default) and rejects anything else.
pub fn resolve_traversal(name: &str) -> Result<&'static str, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | REWARD => Ok(REWARD),
        PUNISHMENT | "penalty" => Ok(PUNISHMENT),
        LEAST_PUNISHED | "least_punished" | "leastpunished" | "blame" => Ok(LEAST_PUNISHED),
        other => Err(format!(
            "unknown traversal {other:?}; expected 'reward', 'punishment' or 'least-punished'"
        )),
    }
}

/// One out-edge's evidence, split in two.
#[derive(Clone, Copy, Debug)]
pub struct EdgeEvidence {
    pub child: usize,
    pub edge: usize,
    /// What speaks for the step, with every reward taken out of it.
    pub merit: f64,
    /// What speaks against it; never negative.
    pub penalty: f64,
}

impl Graph {
    /// The merit / penalty split of `p`'s out-edges, as a walk that arrived
    /// from `prev` sees them.
    ///
    /// The weight of an edge here is `frequency terms + reward_scale * reward`,
    /// so the two halves come apart exactly: the merit is the weight with the
    /// whole reward subtracted back out, and the penalty is
    /// `reward_scale * max(0, -reward)`.  A judged context splits the same way -
    /// what says the step was right here is merit, what says it was wrong here
    /// is penalty - so the punishment traversal never sees a reward at all.
    pub fn child_evidence(&self, p: usize, prev: Option<usize>) -> Vec<EdgeEvidence> {
        let rs = self.reward_scale;
        let ps = self.path_scale;
        let context = prev.is_some() && ps != 0.0 && !self.paths_empty();
        let adj = self.children_of(p);
        let mut out = Vec::with_capacity(adj.len());
        for Transition { p: child, e: edge } in adj {
            let reward = self.edge_reward.get(edge).copied().unwrap_or(0.0);
            let mut merit = self.edge_w.get(edge).copied().unwrap_or(0.0) - rs * reward;
            let mut penalty = if reward < 0.0 { rs * -reward } else { 0.0 };
            if context {
                let term = ps * self.path_term(prev.expect("checked"), edge);
                if term > 0.0 {
                    merit += term;
                } else {
                    penalty += -term;
                }
            }
            out.push(EdgeEvidence {
                child,
                edge,
                merit,
                penalty,
            });
        }
        out
    }
}

/// `[(child, edge, score)] -> [(child, edge, -log softmax(score))]`, numerically stable.
fn softmax_costs(evidence: &[EdgeEvidence], scores: &[f64]) -> Vec<ChildCost> {
    if evidence.is_empty() {
        return Vec::new();
    }
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let terms: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let lse = top + fsum(&terms).ln();
    evidence
        .iter()
        .zip(scores)
        .map(|(ev, score)| ChildCost {
            child: ev.child,
            edge: ev.edge,
            cost: lse - score,
            // `punish` stays zero: it is the least-punished traversal's currency
            // (crate::search), and this traversal has priced the blame into `cost`
            punish: 0.0,
        })
        .collect()
}

/// The cost function of the punishment traversal, cached per `(parent, prev)`
/// until the graph moves under it - the way the graph caches its own.
pub struct PenaltyCosts {
    penalty_scale: f64,
    merit_scale: f64,
    cache: std::cell::RefCell<Map<PathKey, Vec<ChildCost>>>,
    version: std::cell::Cell<i64>,
}

impl PenaltyCosts {
    /// The two scales; both must be `>= 0`.
    pub fn new(penalty_scale: f64, merit_scale: f64) -> Result<PenaltyCosts, String> {
        if penalty_scale < 0.0 {
            return Err(format!("penalty_scale must be >= 0, got {penalty_scale}"));
        }
        if merit_scale < 0.0 {
            return Err(format!("merit_scale must be >= 0, got {merit_scale}"));
        }
        Ok(PenaltyCosts {
            penalty_scale,
            merit_scale,
            cache: std::cell::RefCell::new(map()),
            version: std::cell::Cell::new(-1),
        })
    }

    /// `p`'s out-edges as a walk that arrived from `prev` sees them.
    pub fn costs(&self, g: &Graph, p: usize, prev: Option<usize>) -> Vec<ChildCost> {
        let version = g.version().value;
        if self.version.get() != version {
            self.cache.borrow_mut().clear();
            self.version.set(version);
        }
        // the context makes no difference where nothing was judged, so one entry serves every caller
        let prev = match prev {
            Some(prev) if !g.paths_empty() => Some(prev),
            _ => None,
        };
        let key = PathKey {
            prev: prev.unwrap_or(usize::MAX),
            edge: p,
        };
        if let Some(found) = self.cache.borrow().get(&key) {
            return found.clone();
        }
        let evidence = g.child_evidence(p, prev);
        let scores: Vec<f64> = evidence
            .iter()
            .map(|ev| self.merit_scale * ev.merit - self.penalty_scale * ev.penalty)
            .collect();
        let costs = softmax_costs(&evidence, &scores);
        self.cache.borrow_mut().insert(key, costs.clone());
        costs
    }
}

/// The cost function a search should read the graph through, or `None` for the
/// model's own - which is what the reward traversal returns, and what the
/// least-punished ranking returns too, since it reads the blame itself.
pub fn traversal_costs(traversal: &str, penalty_scale: f64, merit_scale: f64) -> Result<Option<PenaltyCosts>, String> {
    match resolve_traversal(traversal)? {
        REWARD | LEAST_PUNISHED => Ok(None),
        _ => Ok(Some(PenaltyCosts::new(penalty_scale, merit_scale)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphOptions, Model, TrainOptions};

    fn trained() -> Model {
        let texts: Vec<String> = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "the dog sat on the mat",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut model = Model::new(0, GraphOptions::default()).unwrap();
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    #[test]
    fn the_names_it_answers_to() {
        for name in ["", "reward", "REWARD"] {
            assert_eq!(resolve_traversal(name).unwrap(), REWARD);
        }
        assert_eq!(resolve_traversal("Punishment").unwrap(), PUNISHMENT);
        assert_eq!(resolve_traversal("least-punished").unwrap(), LEAST_PUNISHED);
        assert!(resolve_traversal("sideways").is_err());
        assert!(traversal_costs(REWARD, 1.0, 1.0).unwrap().is_none());
        assert!(traversal_costs(LEAST_PUNISHED, 1.0, 1.0).unwrap().is_none());
        assert!(traversal_costs(PUNISHMENT, 1.0, 1.0).unwrap().is_some());
        assert!(traversal_costs(PUNISHMENT, -1.0, 1.0).is_err());
    }

    #[test]
    fn the_evidence_splits_the_reward_out() {
        let mut model = trained();
        model.punish(&["the dog sat on the mat".to_string()], 1, 3.0).unwrap();
        model.g.prepare();
        let mut punished = 0;
        for p in crate::FIRST..model.g.num_node_ids() {
            if !model.g.is_alive(p) {
                continue;
            }
            for ev in model.g.child_evidence(p, None) {
                assert!(ev.penalty >= 0.0, "a penalty is never negative");
                if ev.penalty > 0.0 {
                    punished += 1;
                }
                // the merit is the weight with the whole reward taken back out
                let reward = model.g.edge_reward[ev.edge];
                let expected = model.g.edge_w[ev.edge] - model.g.reward_scale * reward;
                assert!((ev.merit - expected).abs() < 1e-12);
            }
        }
        assert!(punished > 0, "the punished path should carry its penalties");
    }

    #[test]
    fn a_punished_step_costs_more_than_it_did() {
        let mut model = trained();
        model.punish(&["the dog sat on the mat".to_string()], 1, 3.0).unwrap();
        model.g.prepare();
        let costs = traversal_costs(PUNISHMENT, 1.0, 0.0).unwrap().unwrap();
        for p in crate::FIRST..model.g.num_node_ids() {
            if !model.g.is_alive(p) || model.g.children_of(p).len() < 2 {
                continue;
            }
            let priced = costs.costs(&model.g, p, None);
            let punished = priced.iter().find(|c| model.g.edge_punishment(c.edge) > 0.0);
            let clean = priced.iter().find(|c| model.g.edge_punishment(c.edge) == 0.0);
            if let (Some(punished), Some(clean)) = (punished, clean) {
                assert!(
                    punished.cost > clean.cost,
                    "with merit_scale 0 only the punishment decides"
                );
                return;
            }
        }
        panic!("no node offered both a punished and an unpunished step");
    }
}
