//! The metacognitive layer - the part of the brain that takes over when a walk
//! meets a cycle.
//!
//! `Research/CyclesAreAFeature.md`: *the brain is a directed cyclic graph,
//! where cycles are a feature, not a bug; when we encounter a cycle we use
//! metacognition or another part of the brain instead.*  In the phase model
//! ([`crate::resonance`]) that is taken literally: the ordinary edge costs
//! decide every step that goes somewhere new, and the moment a walk is about to
//! re-enter a state it has already been in **at the same phase** - a loop that
//! would repeat for ever - the search stops asking the graph and asks this
//! layer instead.
//!
//! A cycle is named by a *signature*: the first three characters of the node
//! being re-entered and how many steps ago it was last seen (`"lo :2"`).  Every
//! signature carries a score for each of three actions - `ride` (go round
//! again), `escape` (take a child that does not close the cycle) and `abort`
//! (walk to END) - learned by counting what the real corpus did.  At prediction
//! the layer adds `-log softmax(scores)` to the cost of the matching move, so a
//! cycle the corpus rides stays cheap to ride and one it never rides becomes
//! expensive; an unseen signature is judged by a prior over every cycle seen,
//! and an untrained layer is uniform, costing `log 3` whatever the walk does.
//!
//! A port of `radixnet/metacog.py`, sums and all: the layer's costs enter path
//! costs that are compared with Python's to the bit, so a row is summed in the
//! order Python sums it and the signatures keep their insertion order, as a
//! Python dict does.

use crate::fsum::fsum;
use crate::hash::{map, Map};
use crate::json::Json;

/// Go round again.
pub const RIDE: &str = "ride";
/// Take a child that does not close the cycle.
pub const ESCAPE: &str = "escape";
/// Stop here: walk to END.
pub const ABORT: &str = "abort";
/// The three actions, in the order every row keeps them.
pub const ACTIONS: [&str; 3] = [RIDE, ESCAPE, ABORT];

/// Loop lengths above this share one signature: past a point a loop is just
/// "a long one".
const MAX_LOOP: i64 = 8;

/// The index of an action in a row.
pub fn action_index(action: &str) -> Result<usize, String> {
    ACTIONS
        .iter()
        .position(|a| *a == action)
        .ok_or_else(|| format!("unknown action {action:?}; expected one of: ride, escape, abort"))
}

/// The name of a cycle: the re-entered node's first three characters and the
/// loop's length, clamped to `1..=8` (`label[:3]` - characters, as Python
/// slices a string, whatever the encoding).
pub fn cycle_signature(label: &str, loop_len: i64) -> String {
    let head: String = label.chars().take(3).collect();
    format!("{head}:{}", loop_len.clamp(1, MAX_LOOP))
}

/// Python's `max(range(3), key=row.__getitem__)`: the first largest.
fn argmax(row: &[f64; 3]) -> usize {
    let mut best = 0;
    for i in 1..3 {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

/// The learned `ride` / `escape` / `abort` policy per cycle signature.
///
/// Scores are plain floats rather than counts so feedback can push them
/// negative; `prior` is the same three summed over every signature.
#[derive(Clone, Debug)]
pub struct MetaLayer {
    pub smoothing: f64,
    pub back_scale: f64,
    pub prior_weight: f64,
    /// The signatures in the order they were first seen, and their rows.
    signatures: Vec<String>,
    rows: Vec<[f64; 3]>,
    index: Map<String, usize>,
    pub prior: [f64; 3],
    pub observed: i64,
}

impl Default for MetaLayer {
    fn default() -> MetaLayer {
        MetaLayer::new(1.0, 2.0, 2.0)
    }
}

impl MetaLayer {
    /// An empty layer with the given smoothing, `BACK` scale and prior weight.
    pub fn new(smoothing: f64, back_scale: f64, prior_weight: f64) -> MetaLayer {
        MetaLayer {
            smoothing,
            back_scale,
            prior_weight,
            signatures: Vec::new(),
            rows: Vec::new(),
            index: map(),
            prior: [0.0; 3],
            observed: 0,
        }
    }

    /// How many signatures the layer has a row for.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Whether the layer has seen no cycle at all.
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    fn row(&self, signature: &str) -> Option<&[f64; 3]> {
        self.index.get(signature).map(|&i| &self.rows[i])
    }

    // -- learning -------------------------------------------------------------

    /// Records that a real text resolved `signature` by doing `action`.
    pub fn observe(&mut self, signature: &str, action: &str, amount: f64) -> Result<(), String> {
        let idx = action_index(action)?;
        let at = match self.index.get(signature) {
            Some(&at) => at,
            None => {
                self.signatures.push(signature.to_string());
                self.rows.push([0.0; 3]);
                self.index.insert(signature.to_string(), self.rows.len() - 1);
                self.rows.len() - 1
            }
        };
        self.rows[at][idx] += amount;
        self.prior[idx] += amount;
        self.observed += 1;
        Ok(())
    }

    /// Thumbs up: makes `action` likelier for this cycle.
    pub fn reward(&mut self, signature: &str, action: &str, amount: f64) -> Result<(), String> {
        self.observe(signature, action, amount)
    }

    /// Thumbs down: makes `action` less likely for this cycle.
    pub fn punish(&mut self, signature: &str, action: &str, amount: f64) -> Result<(), String> {
        self.observe(signature, action, -amount)
    }

    /// 2NRL: every score negated, so what the layer preferred it now avoids.
    pub fn invert(&mut self) {
        for row in &mut self.rows {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        for v in self.prior.iter_mut() {
            *v = -*v;
        }
    }

    // -- the policy -----------------------------------------------------------

    /// The global prior rescaled to `prior_weight` observations: its
    /// direction, with the weight it has actually earned about one unseen
    /// cycle.
    fn prior_row(&self) -> [f64; 3] {
        // `sum(abs(v) for v in prior)`: left to right, from an integer 0
        let total = self.prior[0].abs() + self.prior[1].abs() + self.prior[2].abs();
        if total == 0.0 || self.prior_weight == 0.0 {
            return self.prior;
        }
        let f = self.prior_weight / total;
        [self.prior[0] * f, self.prior[1] * f, self.prior[2] * f]
    }

    /// What the layer knows about cycles at one node, summed over every loop
    /// length it has seen there.
    pub fn about(&self, label: &str) -> [f64; 3] {
        let key = format!("{}:", label.chars().take(3).collect::<String>());
        let mut row = [0.0; 3];
        for (signature, scores) in self.signatures.iter().zip(&self.rows) {
            if signature.starts_with(&key) {
                for k in 0..3 {
                    row[k] += scores[k];
                }
            }
        }
        row
    }

    /// Whether the layer has seen cycles at this node and, on balance, rode them.
    pub fn rides(&self, label: &str) -> bool {
        let row = self.about(label);
        if row.iter().all(|v| *v == 0.0) {
            return false;
        }
        argmax(&row) == 0
    }

    /// How much has been observed about this exact cycle.
    pub fn evidence(&self, signature: &str) -> f64 {
        self.row(signature)
            .map(|r| r[0].abs() + r[1].abs() + r[2].abs())
            .unwrap_or(0.0)
    }

    /// Whether this exact cycle has been observed (rather than judged by the prior).
    pub fn knows(&self, signature: &str) -> bool {
        self.evidence(signature) != 0.0
    }

    /// `[log P(ride), log P(escape), log P(abort)]` for a signature.
    ///
    /// A score is evidence, not a logit, so it goes through the sign-keeping
    /// `sign(s) * log(1 + |s| / smoothing)`: with the default smoothing of 1,
    /// `exp(logit) = 1 + score` and the softmax is add-one smoothing over the
    /// counts.  `back` - the graph's own probability of handing over at the
    /// node the cycle re-enters - counts `back_scale * back` observations
    /// *against* riding.
    pub fn log_policy(&self, signature: &str, back: f64) -> [f64; 3] {
        let mut row = match self.row(signature) {
            Some(r) => *r,
            None => self.prior_row(),
        };
        if back != 0.0 && self.back_scale != 0.0 {
            row[0] -= self.back_scale * back;
        }
        let s = if self.smoothing == 0.0 { 1.0 } else { self.smoothing };
        let logits: [f64; 3] = [
            (row[0].abs() / s).ln_1p().copysign(row[0]),
            (row[1].abs() / s).ln_1p().copysign(row[1]),
            (row[2].abs() / s).ln_1p().copysign(row[2]),
        ];
        // `max(logits)`: the first largest
        let mut m = logits[0];
        for &v in &logits[1..] {
            if v > m {
                m = v;
            }
        }
        let terms = [(logits[0] - m).exp(), (logits[1] - m).exp(), (logits[2] - m).exp()];
        let log_total = m + fsum(&terms).ln();
        [logits[0] - log_total, logits[1] - log_total, logits[2] - log_total]
    }

    /// `-log P(action | signature)`: what the layer adds to a move's cost.
    pub fn cost(&self, signature: &str, action: &str, back: f64) -> Result<f64, String> {
        Ok(-self.log_policy(signature, back)[action_index(action)?])
    }

    /// `-log P` of all three actions, in [`ACTIONS`] order.
    pub fn costs(&self, signature: &str, back: f64) -> [f64; 3] {
        let lp = self.log_policy(signature, back);
        [-lp[0], -lp[1], -lp[2]]
    }

    /// The action the layer would take on its own (the cheapest).
    pub fn decide(&self, signature: &str, back: f64) -> &'static str {
        ACTIONS[argmax(&self.log_policy(signature, back))]
    }

    // -- reporting ------------------------------------------------------------

    /// Summed score per action over every signature.
    pub fn totals(&self) -> Json {
        Json::obj([
            (RIDE, Json::Num(self.prior[0])),
            (ESCAPE, Json::Num(self.prior[1])),
            (ABORT, Json::Num(self.prior[2])),
        ])
    }

    /// The `n` most observed signatures with their policy.
    pub fn top(&self, n: usize) -> Json {
        let weight = |r: &[f64; 3]| r[0].abs() + r[1].abs() + r[2].abs();
        let mut order: Vec<usize> = (0..self.rows.len()).collect();
        // `sorted(..., key=-weight)`: stable, heaviest first
        order.sort_by(|&a, &b| {
            (-weight(&self.rows[a]))
                .partial_cmp(&-weight(&self.rows[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Json::Arr(
            order
                .into_iter()
                .take(n)
                .map(|i| {
                    let r = &self.rows[i];
                    Json::obj([
                        ("signature", Json::str(self.signatures[i].clone())),
                        (
                            "scores",
                            Json::obj([
                                (RIDE, Json::Num(r[0])),
                                (ESCAPE, Json::Num(r[1])),
                                (ABORT, Json::Num(r[2])),
                            ]),
                        ),
                        ("decision", Json::str(self.decide(&self.signatures[i], 0.0))),
                        ("weight", Json::Num(weight(r))),
                    ])
                })
                .collect(),
        )
    }

    /// `{"signatures", "observed", "totals", "decision"}` - the status bar's.
    pub fn stats(&self) -> Json {
        Json::obj([
            ("signatures", Json::Int(self.len() as i64)),
            ("observed", Json::Int(self.observed)),
            ("totals", self.totals()),
            (
                "decision",
                Json::str(if self.observed != 0 { self.decide("", 0.0) } else { RIDE }),
            ),
        ])
    }

    // -- the file -------------------------------------------------------------

    /// The layer as the model file keeps it; a signature whose row is all
    /// zeros is dropped.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("smoothing", Json::Num(self.smoothing)),
            ("back_scale", Json::Num(self.back_scale)),
            ("prior_weight", Json::Num(self.prior_weight)),
            ("observed", Json::Int(self.observed)),
            ("actions", Json::strs(ACTIONS)),
            ("prior", Json::nums(self.prior)),
            (
                "scores",
                Json::Obj(
                    self.signatures
                        .iter()
                        .zip(&self.rows)
                        .filter(|(_, r)| r.iter().any(|v| *v != 0.0))
                        .map(|(s, r)| (s.clone(), Json::nums(*r)))
                        .collect(),
                ),
            ),
        ])
    }

    /// Rebuilds a layer; a missing document gives a fresh one.
    pub fn from_json(doc: &Json) -> MetaLayer {
        let num = |key: &str, fallback: f64| doc.at(key).as_f64().unwrap_or(fallback);
        let mut layer = MetaLayer::new(num("smoothing", 1.0), num("back_scale", 2.0), num("prior_weight", 2.0));
        if doc.is_null() {
            return layer;
        }
        layer.observed = doc.at("observed").as_i64().unwrap_or(0);
        let prior = doc.at("prior").to_f64s();
        if prior.len() == 3 {
            layer.prior = [prior[0], prior[1], prior[2]];
        }
        if let Json::Obj(pairs) = doc.at("scores") {
            for (signature, row) in pairs {
                let values = row.to_f64s();
                if values.len() == 3 {
                    layer.signatures.push(signature.clone());
                    layer.rows.push([values[0], values[1], values[2]]);
                    layer.index.insert(signature.clone(), layer.rows.len() - 1);
                }
            }
        }
        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signature_is_the_head_and_the_loop() {
        assert_eq!(cycle_signature("lol lol", 2), "lol:2");
        assert_eq!(cycle_signature("ab", 0), "ab:1");
        assert_eq!(cycle_signature("abcdef", 40), "abc:8");
    }

    #[test]
    fn an_untrained_layer_is_uniform() {
        let layer = MetaLayer::default();
        let costs = layer.costs("abc:2", 0.0);
        for c in costs {
            assert!((c - 3f64.ln()).abs() < 1e-15);
        }
        assert_eq!(layer.stats().at("decision").as_str(), Some(RIDE));
    }

    #[test]
    fn what_the_corpus_rode_is_cheap_to_ride() {
        let mut layer = MetaLayer::default();
        for _ in 0..10 {
            layer.observe("aaa:1", RIDE, 1.0).unwrap();
        }
        layer.observe("bbb:2", ESCAPE, 1.0).unwrap();
        let known = layer.costs("aaa:1", 0.0);
        assert!(known[0] < known[1] && known[0] < known[2]);
        assert!(layer.rides("aaa"));
        assert!(!layer.rides("bbb"));
        assert_eq!(layer.decide("aaa:1", 0.0), RIDE);
        // the reflex counts against riding, but ten observations outweigh it
        assert_eq!(layer.decide("aaa:1", 0.9), RIDE);
        layer.invert();
        assert_eq!(layer.decide("aaa:1", 0.0), ESCAPE);
        assert!(layer.observe("x:1", "fly", 1.0).is_err());
    }

    #[test]
    fn the_file_keeps_the_order_and_drops_empty_rows() {
        let mut layer = MetaLayer::default();
        layer.observe("zzz:1", RIDE, 1.0).unwrap();
        layer.observe("aaa:1", ABORT, 2.0).unwrap();
        layer.observe("mmm:1", RIDE, 1.0).unwrap();
        layer.observe("mmm:1", RIDE, -1.0).unwrap();
        let doc = layer.to_json();
        let keys: Vec<&str> = match doc.at("scores") {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys, vec!["zzz:1", "aaa:1"]);
        let back = MetaLayer::from_json(&doc);
        assert_eq!(back.to_json().render(0), doc.render(0));
        assert_eq!(back.observed, 4);
    }
}
