//! The radix model - the sine-activation network the project started as.
//!
//! The same self-compressing cyclic graph as the count model, but its edges do
//! not *count*: every node owns a learnable sine activation
//! `f(x) = a * sin(b * (x - h)) + k` over a learnable state `z`
//! ([`crate::activation`]), every edge a learnable weight `w`, and the score of
//! a transition `p -> c` is `w * f(z_p) * f(z_c)`.  A node's children are drawn
//! by a softmax over those scores, and training is gradient descent on
//! `-log P(child | parent)` by the one-hop rule of [`crate::backend`], with a
//! learning rate - optionally a schedule of the epoch ([`crate::schedule`]).
//!
//! # A variant of the graph, not a second graph
//!
//! The negative network showed the way ([`crate::negative`]): the structure -
//! grams, splits, merges, the index, the counters - is the graph's, and only
//! what a kind keeps *beside* it differs.  Here that is [`RadixData`], the five
//! parameters of every node, hung on `Graph::radix`.  The graph calls four
//! small hooks where the sine model behaves differently from the count model:
//!
//! * a new gram node draws its state `z` from the model's RNG, and a new edge
//!   keeps the weight the count model draws and throws away;
//! * a split copies the node's parameters to the new half;
//! * a merge keeps the stronger activation and rescales the edges on the other
//!   side, so every score `w * f_p * f_c` - every probability - is unchanged;
//! * the costs are a softmax over the scores rather than over the weights.
//!
//! Everything else - the beam, the stochastic walk, scoring, the file's
//! structure - is the count model's code, reading those costs.
//!
//! # Numbers
//!
//! This is `radixnet/model.py`'s `RadixNet` and `radixnet/graph.py`'s sine
//! parts, in Python's operation order: the RNG draws happen in the same order
//! (a node's state, then an edge's weight), the transitions are shuffled with
//! Python's `random.shuffle` over the same generator, and the backend sums its
//! gradients as Python sums them - so a model trained here and one trained by
//! `python -m radixnet --kind radix` are the same model, to the bit.

use std::time::Instant;

use crate::activation::{DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K};
use crate::backend::{Csr, NodeParams, State};
use crate::counter::Counter;
use crate::dijkstra::dijkstra_predict;
use crate::encoding::Encoding;
use crate::graph::{Graph, GraphOptions, Transition, BACK, END, FIRST, START};
use crate::json::Json;
use crate::model::{EpochRecord, Model, PredictOptions};
use crate::mt19937::Mt19937;
use crate::penalty::EdgeEvidence;
use crate::search::PathResult;

/// The file format of the radix model - Python's `MODEL_FORMAT`, the
/// project's first and still its default kind.
pub const RADIX_FORMAT: &str = "radixnet";

/// A new node's state is drawn from `[-Z_RANGE, Z_RANGE]`.
pub const Z_RANGE: f64 = 4.5;

/// The `BACK` sentinel's state: fixed at the far edge of the range rather than
/// drawn, so adding the sentinel moved no random stream.
pub const BACK_Z: f64 = 4.5;

/// What the model's own lines are filed under.
const LOG: &str = "train";

/// The five learnable parameters of every node of a radix graph, parallel to
/// the node arrays.  `None` on every other kind of graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadixData {
    pub z: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub h: Vec<f64>,
    pub k: Vec<f64>,
}

impl RadixData {
    fn push(&mut self, z: f64, a: f64, b: f64, h: f64, k: f64) {
        self.z.push(z);
        self.a.push(a);
        self.b.push(b);
        self.h.push(h);
        self.k.push(k);
    }

    /// The activation of node `i`: `a * sin(b * (z - h)) + k`.
    #[inline]
    pub fn activation(&self, i: usize) -> f64 {
        crate::activation::sine(self.z[i], self.a[i], self.b[i], self.h[i], self.k[i])
    }
}

// -- the graph ------------------------------------------------------------------------------------

impl Graph {
    /// An empty radix graph: the three sentinels, START and END with random
    /// states (drawn in that order, as Python draws them) and BACK with its
    /// fixed one.
    pub fn new_radix(seed: i64, encoding: Encoding) -> Result<Graph, String> {
        let mut g = Graph::new(
            seed,
            GraphOptions {
                encoding,
                ..Default::default()
            },
        )?;
        let mut data = RadixData::default();
        for sentinel in [START, END, BACK] {
            let z = if sentinel == BACK {
                BACK_Z
            } else {
                g.rng.uniform(-Z_RANGE, Z_RANGE)
            };
            data.push(z, DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K);
        }
        g.radix = Some(Box::new(data));
        Ok(g)
    }

    /// Whether this graph learns its weights (the sine model).
    pub fn is_radix(&self) -> bool {
        self.radix.is_some()
    }

    /// Hook of `new_node`: a new node's parameters, at their defaults - the
    /// amplitude negated on an inverted network, as Python's `_new_node` does.
    /// The state is filled in by the caller: drawn for a new gram, copied on a
    /// split, read from the file on a load.
    pub(crate) fn radix_node_added(&mut self) {
        let inverted = self.inverted;
        if let Some(r) = self.radix.as_mut() {
            let a = if inverted { -DEFAULT_A } else { DEFAULT_A };
            r.push(0.0, a, DEFAULT_B, DEFAULT_H, DEFAULT_K);
        }
    }

    /// Hook of `create_gram_node`: a new gram's node draws its state.
    pub(crate) fn radix_gram_created(&mut self, node: usize) {
        if self.radix.is_none() {
            return;
        }
        let z = self.rng.uniform(-Z_RANGE, Z_RANGE);
        if let Some(r) = self.radix.as_mut() {
            r.z[node] = z;
        }
    }

    /// Hook of `split`: the new half takes the node's state and activation.
    pub(crate) fn radix_split(&mut self, a: usize, b: usize) {
        if let Some(r) = self.radix.as_mut() {
            r.z[b] = r.z[a];
            r.a[b] = r.a[a];
            r.b[b] = r.b[a];
            r.h[b] = r.h[a];
            r.k[b] = r.k[a];
        }
    }

    /// The weight a new edge starts with: the draw every graph makes (so the
    /// RNG stays in step with Python's), kept here and negated on an inverted
    /// network; the other kinds compute theirs.
    pub(crate) fn fresh_edge_weight(&self, drawn: f64) -> f64 {
        match (&self.radix, self.inverted) {
            (Some(_), false) => drawn,
            (Some(_), true) => -drawn,
            (None, _) => 0.0,
        }
    }

    /// Hook of `merge_child` before `c`'s out-edges move to `p`: keeps the
    /// activation of whichever node has the larger `|f|` and returns the ratio
    /// the moved out-edges are scaled by, so every edge score `w * f_p * f_c` -
    /// and so every probability and cost - is unchanged by the merge (the
    /// ratio is `<= 1` in magnitude, so weights never grow).  `1.0` on every
    /// other kind, whose activations are the constant 1.
    pub(crate) fn merge_rescale(&mut self, p: usize, c: usize) -> f64 {
        let Some(r) = self.radix.as_ref() else { return 1.0 };
        let fp = r.activation(p);
        let fc = r.activation(c);
        if fp.abs() >= fc.abs() {
            // keep p's activation; c's out-edges carried f_c, they now carry f_p
            return if fp != 0.0 { fc / fp } else { 1.0 };
        }
        // adopt c's activation; p's in-edges carried f_p, they now carry f_c
        let in_ratio = fp / fc;
        let incoming: Vec<usize> = self.parents[p].edges.clone();
        for e in incoming {
            self.edge_w[e] *= in_ratio;
        }
        if let Some(r) = self.radix.as_mut() {
            r.z[p] = r.z[c];
            r.a[p] = r.a[c];
            r.b[p] = r.b[c];
            r.h[p] = r.h[c];
            r.k[p] = r.k[c];
        }
        1.0
    }

    /// `f(z)` of one node: the sine activation, or the constant 1 of the
    /// kinds that do not learn one.
    pub fn activation_of(&self, node: usize) -> f64 {
        match &self.radix {
            Some(r) => r.activation(node),
            None => 1.0,
        }
    }

    /// Every node's activation, or `None` when they are all the constant 1.
    pub(crate) fn activations(&self) -> Option<Vec<f64>> {
        let r = self.radix.as_ref()?;
        Some((0..r.z.len()).map(|i| r.activation(i)).collect())
    }

    /// The node parameters as the graph view reports them: `(z, a, b, h, k)`.
    pub fn node_parameters(&self, node: usize) -> (f64, f64, f64, f64, f64) {
        match &self.radix {
            Some(r) => (r.z[node], r.a[node], r.b[node], r.h[node], r.k[node]),
            None => (if node == BACK { BACK_Z } else { 0.0 }, 0.0, DEFAULT_B, DEFAULT_H, 1.0),
        }
    }

    /// The merit / penalty split of a radix node's out-edges: the positive
    /// part of the score `w * f_p * f_c` and the negative part.
    ///
    /// The sine model keeps no ledger of its punishments - 2NRL trains a
    /// failure in and then inverts it - so what a punishment leaves behind *is*
    /// a negative score, and that is what the punishment traversal prices.
    pub(crate) fn radix_evidence(&self, p: usize) -> Vec<EdgeEvidence> {
        let Some(r) = self.radix.as_ref() else {
            return Vec::new();
        };
        let fp = r.activation(p);
        let adj = &self.children[p];
        adj.order
            .iter()
            .zip(adj.edges.iter())
            .map(|(&c, &e)| {
                let score = self.edge_w[e] * fp * r.activation(c);
                EdgeEvidence {
                    child: c,
                    edge: e,
                    merit: if score > 0.0 { score } else { 0.0 },
                    penalty: if -score > 0.0 { -score } else { 0.0 },
                }
            })
            .collect()
    }

    /// Negates every alive edge weight and every node's activation - both `a`
    /// and `k`, so `f` negates exactly whatever `k` has learned - and toggles
    /// `inverted`.  Every score changes sign, the ranking reverses exactly, and
    /// two inversions are the identity.
    pub(crate) fn radix_invert(&mut self) {
        for e in 0..self.edge_w.len() {
            if self.edge_alive[e] {
                self.edge_w[e] = -self.edge_w[e];
            }
        }
        if let Some(r) = self.radix.as_mut() {
            for v in r.a.iter_mut().chain(r.k.iter_mut()) {
                *v = -*v;
            }
        }
        self.inverted = !self.inverted;
        self.version.add(1);
    }

    /// Moves the activation (`mode = "activation"`: both `a` and `k`) or the
    /// state (`"state"`: `z`) of each listed node toward its negation,
    /// `value *= 1 - 2 * amount`: amount 1 is a sign flip, 0.5 zeroes the node
    /// (its transitions go neutral), a small amount only attenuates it.
    ///
    /// A node's activation enters the score of every edge into and out of it,
    /// so flipping every other node of a path makes that path as unlikely as
    /// it was likely - the local counterpart of inverting the whole network,
    /// and what the evolve loop's blatant mode does to a failure.  Sentinels,
    /// dead and unknown ids and zero amounts are ignored; returns how many
    /// nodes changed.
    pub fn flip_nodes(&mut self, nodes: &[(usize, f64)], mode: &str) -> Result<usize, String> {
        if mode != "activation" && mode != "state" {
            return Err(format!("mode must be 'activation' or 'state', got {mode:?}"));
        }
        let n = self.labels.len();
        let mut changed = 0;
        let alive = &self.alive;
        let Some(r) = self.radix.as_mut() else {
            return Err("flipping nodes needs the radix model".to_string());
        };
        for &(node, amount) in nodes {
            if node < FIRST || node >= n || node >= r.a.len() || !alive[node] || amount <= 0.0 || amount.is_nan() {
                continue;
            }
            let scale = 1.0 - 2.0 * amount;
            if mode == "activation" {
                r.a[node] *= scale;
                r.k[node] *= scale;
            } else {
                r.z[node] *= scale;
            }
            changed += 1;
        }
        if changed > 0 {
            self.version.add(1);
        }
        Ok(changed)
    }

    /// Moves the weight of `p -> c` so the transition gets `amount` likelier
    /// (negative: dearer) - which way depends on the sign of the two
    /// activations the weight sits between.  `false` when there is no such edge.
    pub fn nudge_edge(&mut self, p: usize, c: usize, amount: f64) -> bool {
        if amount == 0.0 || p < FIRST || p >= self.children.len() {
            return false;
        }
        let Some(e) = self.children[p].get(c) else { return false };
        let pull = self.activation_of(p) * self.activation_of(c);
        if pull > 0.0 {
            self.edge_w[e] += amount;
        } else if pull < 0.0 {
            self.edge_w[e] -= amount;
        } else {
            return false;
        }
        self.version.add(1);
        true
    }

    /// `observe_back` as the sine model learns it: the `BACK` edge is counted
    /// and nudged likelier, the step it looped through dearer and the step it
    /// took instead cheaper - weights moved directly, as everything here is.
    pub(crate) fn radix_observe_back(
        &mut self,
        p: usize,
        went: Option<usize>,
        instead: Option<usize>,
        amount: f64,
    ) -> Result<usize, String> {
        if p < FIRST || p >= self.labels.len() || !self.alive[p] {
            return Err(format!("node {p} is not a real node to go back from"));
        }
        if amount < 0.0 {
            return Err(format!("amount must be >= 0, got {amount}"));
        }
        let e = match self.edge(p, BACK) {
            Some(e) => e,
            None => self.new_edge(p, BACK, 0, 0),
        };
        self.bump_node(BACK);
        self.bump_edge(e);
        self.add_traversals(1);
        self.version.add(1);
        self.nudge_edge(p, BACK, amount);
        if let Some(went) = went.filter(|&w| w != BACK) {
            self.nudge_edge(p, went, -amount);
        }
        if let Some(instead) = instead.filter(|&i| i != BACK) {
            self.nudge_edge(p, instead, amount);
        }
        Ok(e)
    }

    // -- the backend's view -------------------------------------------------

    /// The edges as CSR over *all* node ids (a dead node has an empty row).
    pub fn to_csr(&self) -> Csr {
        let mut csr = Csr {
            indptr: vec![0],
            edge_pos: vec![usize::MAX; self.edge_w.len()],
            ..Default::default()
        };
        for adj in &self.children {
            for (&c, &e) in adj.order.iter().zip(adj.edges.iter()) {
                csr.edge_pos[e] = csr.indices.len();
                csr.indices.push(c);
                csr.edge_ids.push(e);
                csr.weights.push(self.edge_w[e]);
            }
            csr.indptr.push(csr.indices.len());
        }
        csr
    }

    /// Copies of `z, a, b, h, k` in node-id order.
    pub fn node_params(&self) -> NodeParams {
        match &self.radix {
            Some(r) => NodeParams {
                z: r.z.clone(),
                a: r.a.clone(),
                b: r.b.clone(),
                h: r.h.clone(),
                k: r.k.clone(),
            },
            None => NodeParams::default(),
        }
    }

    /// Writes trained weights (CSR order) back to the edges.
    pub fn apply_csr_weights(&mut self, csr: &Csr, weights: &[f64]) -> Result<(), String> {
        if weights.len() != csr.edge_ids.len() {
            return Err(format!(
                "expected {} weights, got {}",
                csr.edge_ids.len(),
                weights.len()
            ));
        }
        for (&e, &w) in csr.edge_ids.iter().zip(weights) {
            self.edge_w[e] = w;
        }
        self.version.add(1);
        Ok(())
    }

    /// Writes trained node parameters back.
    pub fn apply_node_params(&mut self, params: NodeParams) -> Result<(), String> {
        if params.z.len() != self.labels.len() {
            return Err(format!(
                "expected params for {} nodes, got {}",
                self.labels.len(),
                params.z.len()
            ));
        }
        if let Some(r) = self.radix.as_mut() {
            **r = RadixData {
                z: params.z,
                a: params.a,
                b: params.b,
                h: params.h,
                k: params.k,
            };
        }
        self.version.add(1);
        Ok(())
    }

    /// The parameters of the alive nodes in file order, or `None` when the
    /// graph learns none.
    pub(crate) fn radix_arrays(&self, order: &[usize]) -> Option<[Vec<f64>; 5]> {
        let r = self.radix.as_ref()?;
        let pick = |v: &Vec<f64>| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        Some([pick(&r.z), pick(&r.a), pick(&r.b), pick(&r.h), pick(&r.k)])
    }

    /// Reads a file's node parameters over the defaults `reset_nodes` left.
    pub(crate) fn radix_read_params(&mut self, nodes: &Json) -> Result<(), String> {
        let n = self.labels.len();
        let Some(r) = self.radix.as_mut() else { return Ok(()) };
        let read = |key: &str| nodes.at(key).to_f64s();
        let (z, a, b, h, k) = (read("z"), read("a"), read("b"), read("h"), read("k"));
        if [z.len(), a.len(), b.len(), h.len(), k.len()]
            .iter()
            .any(|&len| len != n)
        {
            return Err("node arrays have inconsistent lengths".to_string());
        }
        **r = RadixData { z, a, b, h, k };
        Ok(())
    }
}

// -- Python's random: shuffle ---------------------------------------------------------------------

/// `random.getrandbits(k)` for `k <= 64`: 32-bit words, least significant first.
fn getrandbits(rng: &mut Mt19937, k: u32) -> u64 {
    if k == 0 {
        return 0;
    }
    if k <= 32 {
        return (rng.next_u32() >> (32 - k)) as u64;
    }
    let low = rng.next_u32() as u64;
    let high = (rng.next_u32() >> (64 - k)) as u64;
    low | (high << 32)
}

/// `random._randbelow_with_getrandbits(n)`: rejection sampling on
/// `n.bit_length()` bits.
fn randbelow(rng: &mut Mt19937, n: usize) -> usize {
    let k = usize::BITS - n.leading_zeros();
    loop {
        let r = getrandbits(rng, k) as usize;
        if r < n {
            return r;
        }
    }
}

/// `random.shuffle`: Fisher-Yates from the end, as Python's.
pub fn shuffle<T>(rng: &mut Mt19937, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = randbelow(rng, i + 1);
        items.swap(i, j);
    }
}

// -- training -------------------------------------------------------------------------------------

/// The hyper-parameters of one training call - Python's `TrainConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub epochs: usize,
    /// The learning rate of the edge weights and node states.
    pub lr: f64,
    /// The learning rate of the activation parameters `a, b, h, k`.
    pub act_lr: f64,
    /// Transitions per backend step.
    pub batch_size: usize,
    pub clip: f64,
    /// Merge unary chains after every epoch (and once before the first).
    pub auto_compress: bool,
    /// Shuffle the transitions every epoch, with the model's seeded RNG.
    pub shuffle: bool,
    pub checkpoint_every: usize,
    pub verbose: bool,
    /// `lr` as a graph function of the epoch ([`crate::schedule`]).
    pub lr_schedule: Option<String>,
    /// `act_lr` as one; it may read `lr`, the epoch's learning rate.
    pub act_lr_schedule: Option<String>,
    /// Play the schedules backwards: a ramp up becomes a ramp down.
    pub reverse_schedule: bool,
    /// How every kind that learns by walking a list of texts walks it: the
    /// order, the curriculum, the rehearsal of the replay buffer and the
    /// early stop ([`crate::training`], `../../SPEC-SearchAndTraining.md`).
    pub plan: crate::training::Plan,
}

impl Default for TrainConfig {
    /// Python's defaults.
    fn default() -> TrainConfig {
        TrainConfig {
            epochs: 5,
            lr: 0.05,
            act_lr: 0.005,
            batch_size: 256,
            clip: 5.0,
            auto_compress: true,
            shuffle: true,
            checkpoint_every: 0,
            verbose: false,
            lr_schedule: None,
            act_lr_schedule: None,
            reverse_schedule: false,
            plan: crate::training::Plan::default(),
        }
    }
}

impl TrainConfig {
    /// `(lr, act_lr)` for every epoch, the schedules applied.
    pub fn rates(&self) -> Result<Vec<(f64, f64)>, String> {
        if self.lr_schedule.is_none() && self.act_lr_schedule.is_none() {
            return Ok(vec![(self.lr, self.act_lr); self.epochs]);
        }
        let points = crate::schedule::preview_points(
            self.lr_schedule.as_deref(),
            self.act_lr_schedule.as_deref(),
            self.epochs as i64,
            self.lr,
            self.act_lr,
            self.reverse_schedule,
        )?;
        Ok(points.into_iter().map(|p| (p.lr, p.act_lr)).collect())
    }

    /// The values that cannot be trained with, in Python's words.
    pub fn validate(&self) -> Result<(), String> {
        if self.lr_schedule.is_some() || self.act_lr_schedule.is_some() {
            self.rates()?;
        }
        if self.batch_size < 1 {
            return Err(format!("batch_size must be >= 1, got {}", self.batch_size));
        }
        for (name, value) in [("lr", self.lr), ("act_lr", self.act_lr)] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "{name} must be a finite non-negative number, got {}",
                    crate::json::py_repr(value)
                ));
            }
        }
        if self.clip.is_nan() || self.clip <= 0.0 {
            return Err(format!("clip must be > 0, got {}", crate::json::py_repr(self.clip)));
        }
        self.plan.check()
    }

    /// `dataclasses.asdict(config)`, in field order.
    pub fn to_json(&self) -> Json {
        let text = |s: &Option<String>| s.as_ref().map(|s| Json::str(s.clone())).unwrap_or(Json::Null);
        let mut doc = Json::obj([
            ("epochs", Json::Int(self.epochs as i64)),
            ("lr", Json::Num(self.lr)),
            ("act_lr", Json::Num(self.act_lr)),
            ("batch_size", Json::Int(self.batch_size as i64)),
            ("clip", Json::Num(self.clip)),
            ("auto_compress", Json::Bool(self.auto_compress)),
            ("shuffle", Json::Bool(self.shuffle)),
            ("checkpoint_every", Json::Int(self.checkpoint_every as i64)),
            ("verbose", Json::Bool(self.verbose)),
            ("lr_schedule", text(&self.lr_schedule)),
            ("act_lr_schedule", text(&self.act_lr_schedule)),
            ("reverse_schedule", Json::Bool(self.reverse_schedule)),
        ]);
        if let Json::Obj(pairs) = &mut doc {
            pairs.extend(self.plan.json_pairs());
        }
        doc
    }
}

/// The settings of one feedback call - 2NRL, a thumbs up or a thumbs down -
/// for every kind: each reads what applies to it (the radix model the
/// learning rates, the count and phase models the strength).
#[derive(Clone, Debug)]
pub struct Feedback {
    pub neg_epochs: usize,
    pub pos_epochs: usize,
    pub neg_lr: f64,
    pub pos_lr: f64,
    pub strength: f64,
    /// `batch_size`, `auto_compress`, `clip` and `shuffle`, when given.
    pub batch_size: Option<usize>,
    pub auto_compress: Option<bool>,
    pub clip: Option<f64>,
    pub shuffle: Option<bool>,
    /// One weight per text, `>= 0`: a rating rather than a thumb.
    pub good_weights: Option<Vec<f64>>,
    pub bad_weights: Option<Vec<f64>>,
}

impl Feedback {
    /// The `2nrl` command's and route's defaults.
    pub fn two_nrl() -> Feedback {
        Feedback {
            neg_epochs: 3,
            pos_epochs: 3,
            neg_lr: 0.05,
            pos_lr: 0.01,
            strength: 1.0,
            batch_size: None,
            auto_compress: None,
            clip: None,
            shuffle: None,
            good_weights: None,
            bad_weights: None,
        }
    }

    /// The `feedback` command's and route's defaults (rated sets are small).
    pub fn rated() -> Feedback {
        Feedback {
            neg_epochs: 2,
            pos_epochs: 3,
            neg_lr: 0.5,
            pos_lr: 0.1,
            batch_size: Some(4),
            ..Feedback::two_nrl()
        }
    }

    /// A training config with this feedback's overrides applied.
    fn config(&self, epochs: usize, lr: f64, act_lr: f64) -> TrainConfig {
        let base = TrainConfig::default();
        TrainConfig {
            epochs,
            lr,
            act_lr,
            batch_size: self.batch_size.unwrap_or(base.batch_size),
            auto_compress: self.auto_compress.unwrap_or(base.auto_compress),
            clip: self.clip.unwrap_or(base.clip),
            shuffle: self.shuffle.unwrap_or(base.shuffle),
            ..base
        }
    }
}

/// Python's `_weight_groups`: `[(weight, texts)]` grouping texts of equal
/// weight (to three decimals), heaviest first; a weight of 0 is dropped.
pub fn weight_groups(texts: &[String], weights: &[f64], name: &str) -> Result<Vec<(f64, Vec<String>)>, String> {
    if weights.len() != texts.len() {
        return Err(format!(
            "{name} has {} entries for {} texts",
            weights.len(),
            texts.len()
        ));
    }
    let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
    for (text, &weight) in texts.iter().zip(weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "{name} must be finite and >= 0, got {}",
                crate::json::py_repr(weight)
            ));
        }
        if weight > 0.0 {
            // `round(weight, 3)`: the decimal string correctly rounded, read back
            let key: f64 = format!("{weight:.3}").parse().unwrap_or(weight);
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, members)) => members.push(text.clone()),
                None => groups.push((key, vec![text.clone()])),
            }
        }
    }
    // `sorted(groups.items(), key=lambda item: -item[0])`: stable
    groups.sort_by(|a, b| (-a.0).partial_cmp(&-b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(groups)
}

/// `exp(min(loss, 700))` - `exp` overflows above ~709 - where a NaN loss
/// stays NaN, as Python's `min` keeps it.
pub(crate) fn perplexity(loss: f64) -> f64 {
    let capped = if 700.0 < loss { 700.0 } else { loss };
    capped.exp()
}

/// Python's `meta_add` for a lifetime counter the `Meta` struct has no field
/// for (`path_inversions`): the reading and its `_resets` live among the extra
/// keys, and a new counter is appended there, both keys, as Python appends it.
pub(crate) fn meta_add_extra(meta: &mut crate::model::Meta, key: &str, delta: i64) -> i64 {
    let resets_key = format!("{key}_resets");
    let read = |meta: &crate::model::Meta, k: &str| {
        meta.extra
            .iter()
            .find(|(name, _)| name == k)
            .and_then(|(_, v)| v.as_i64())
            .unwrap_or(0)
    };
    let counter = Counter::new(read(meta, key), read(meta, &resets_key)).bumped(delta);
    for (name, value) in [(key.to_string(), counter.value), (resets_key, counter.resets)] {
        match meta.extra.iter_mut().find(|(k, _)| *k == name) {
            Some((_, slot)) => *slot = Json::Int(value),
            None => meta.extra.push((name, Json::Int(value))),
        }
    }
    counter.value
}

impl Model {
    /// An untrained radix model.
    pub fn new_radix(seed: i64, encoding: Encoding) -> Result<Model, String> {
        Ok(Model::from_graph(Graph::new_radix(seed, encoding)?))
    }

    /// Whether this is the sine-activation model.
    pub fn is_radix(&self) -> bool {
        self.g.is_radix()
    }

    /// Registers encoded texts structurally and returns their transitions -
    /// Python's `_observe_grams`: when the first pass changed the structure,
    /// a second, uncounted pass re-derives every transition from the final
    /// one (a later text may have split a node an earlier one pointed at).
    fn observe_grams(&mut self, grams: &[Vec<String>], count: bool) -> Result<Vec<Transition>, String> {
        let before = self.g.structure_version;
        let mut transitions = Vec::new();
        for gram in grams {
            transitions.extend(self.g.observe(gram, count)?);
        }
        if self.g.structure_version != before {
            transitions.clear();
            for gram in grams {
                transitions.extend(self.g.observe(gram, false)?);
            }
        }
        Ok(transitions)
    }

    /// Trains on `texts`: every text registered structurally (and counted),
    /// then per epoch the graph exported as CSR, the transitions shuffled with
    /// the model's RNG and fed to the backend in batches, and the weights and
    /// node parameters written back.  `on_epoch` sees every record as it is
    /// made; returning `false` stops the run after that epoch.
    ///
    /// The texts are walked the way `cfg.plan` says - the order, the
    /// curriculum, the rehearsal of the replay buffer and the early stop of
    /// `../../SPEC-SearchAndTraining.md` - unless the call carries a `phase`:
    /// a pass stamped `"negative"` or `"positive"` is feedback, walks every
    /// text in corpus order and leaves the replay buffer alone.
    pub fn radix_train(
        &mut self,
        texts: &[String],
        cfg: &TrainConfig,
        phase: Option<&str>,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        cfg.validate()?;
        let enc = self.g.enc;
        let read = crate::training::read(&enc, texts, &cfg.plan);
        let texts: &[String] = &read;
        let mut skipped_short = 0usize;
        let kept: Vec<&String> = texts
            .iter()
            .filter(|t| {
                let ok = enc.len(t) >= enc.n;
                skipped_short += usize::from(!ok);
                ok
            })
            .collect();
        // a model that keeps a replay buffer offers every run's texts to it
        let mut plan = if phase.is_none() && (cfg.plan.active() || self.replay.is_some()) {
            Some(self.training_plan(kept.iter().map(|t| (*t).clone()).collect(), cfg.epochs, &cfg.plan)?)
        } else {
            None
        };
        let grams: Vec<Vec<String>> = kept.iter().map(|t| enc.encode(t)).collect();
        let mut transitions = self.observe_grams(&grams, true)?;
        let rehearsed = plan.as_ref().map(|p| p.replayed()).unwrap_or_default();
        if !rehearsed.is_empty() {
            // a rehearsed text is not new: nothing to count
            let again: Vec<Vec<String>> = rehearsed.iter().map(|t| enc.encode(t)).collect();
            self.observe_grams(&again, false)?;
            transitions = self.observe_grams(&grams, false)?;
        }
        let mut observed_version = self.g.structure_version;
        self.meta.trained_texts.add(kept.len() as i64);
        self.meta
            .trained_chars
            .add(kept.iter().map(|t| enc.len(t) as i64).sum());
        let mut pending_merges = if cfg.auto_compress { self.g.compress() } else { 0 };

        let rates = cfg.rates()?;
        let mut arrays_version: Option<Counter> = None;
        let (mut parents_all, mut positions_all): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut records = Vec::with_capacity(cfg.epochs);
        for (k, &(lr, act_lr)) in rates.iter().take(cfg.epochs).enumerate() {
            let started = Instant::now();
            if let Some(p) = plan.as_ref().filter(|p| !p.plain()) {
                // this epoch's texts: the order, the curriculum and the rehearsal
                let walked: Vec<Vec<String>> = p
                    .epoch_texts(k, self.meta.epochs_total.bumped(1).value)
                    .iter()
                    .map(|t| enc.encode(t))
                    .collect();
                transitions = self.observe_grams(&walked, false)?;
                observed_version = self.g.structure_version;
                arrays_version = None;
            } else if self.g.structure_version != observed_version {
                // a merge (or another structural change) moved edge ids
                transitions = self.observe_grams(&grams, false)?;
                observed_version = self.g.structure_version;
            }
            let csr = self.g.to_csr();
            let mut state = State::prepare(&csr, &self.g.node_params())?;
            if arrays_version != Some(observed_version) {
                parents_all = transitions.iter().map(|t| t.p).collect();
                positions_all = transitions.iter().map(|t| csr.edge_pos[t.e]).collect();
                arrays_version = Some(observed_version);
            }
            let n = parents_all.len();
            let (parents, positions) = if cfg.shuffle && n > 1 {
                let mut order: Vec<usize> = (0..n).collect();
                shuffle(&mut self.g.rng, &mut order);
                (
                    order.iter().map(|&i| parents_all[i]).collect::<Vec<_>>(),
                    order.iter().map(|&i| positions_all[i]).collect::<Vec<_>>(),
                )
            } else {
                (parents_all.clone(), positions_all.clone())
            };
            let mut loss_sum = 0.0;
            let mut start = 0;
            while start < n {
                let end = (start + cfg.batch_size).min(n);
                let step = state.step(&parents[start..end], &positions[start..end], lr, act_lr, cfg.clip)?;
                loss_sum += step * (end - start) as f64;
                start = end;
            }
            let (weights, params) = state.finalize();
            self.g.apply_csr_weights(&csr, &weights)?;
            self.g.apply_node_params(params)?;
            let merges = if cfg.auto_compress { self.g.compress() } else { 0 } + pending_merges;
            pending_merges = 0;
            let loss = if n > 0 { loss_sum / n as f64 } else { 0.0 };
            self.g.carry_counters(false); // the epoch is over: wrap whatever reached the limit
            self.meta.epochs_total.add(1);
            let stopping = plan.as_mut().is_some_and(|p| p.stop(k, loss));
            let mut extra = vec![
                ("lr".to_string(), Json::Num(lr)),
                ("act_lr".to_string(), Json::Num(act_lr)),
            ];
            extra.extend(crate::model::early_stop_extra(stopping));
            let record = EpochRecord {
                epoch: self.meta.epochs_total.value,
                loss,
                perplexity: perplexity(loss),
                nodes: self.g.num_nodes(),
                edges: self.g.num_edges(),
                trigrams: self.g.num_trigrams(),
                compression_ratio: self.g.compression_ratio(),
                merges,
                transitions: n as i64,
                seconds: started.elapsed().as_secs_f64(),
                skipped_short,
                traversed: true,
                reward: 0.0,
                phase: phase.map(str::to_string),
                extra,
            };
            crate::log_debug!(
                LOG,
                "epoch {} {}: loss {:.4}, lr {lr}, act_lr {act_lr}, {} nodes, {} edges, {} merge(s)",
                record.epoch,
                phase.unwrap_or("train"),
                record.loss,
                record.nodes,
                record.edges,
                record.merges
            );
            self.history.push(record.clone());
            self.epoch_done(&record)?;
            let go_on = on_epoch(&record);
            records.push(record);
            if stopping || !go_on {
                break;
            }
        }
        if let Some(p) = plan.as_mut() {
            self.replay = p.finish();
        }
        crate::log_info!(
            LOG,
            "{} epoch(s) of the sine model over {} text(s): loss {:.4} -> {:.4}, {} nodes",
            records.len(),
            kept.len(),
            records.first().map(|r| r.loss).unwrap_or(0.0),
            records.last().map(|r| r.loss).unwrap_or(0.0),
            self.g.num_nodes()
        );
        Ok(records)
    }

    /// One pass per group of equally weighted texts, the learning rates scaled
    /// by the weight; the records carry `"weight"`.
    fn radix_weighted(
        &mut self,
        texts: &[String],
        weights: &[f64],
        name: &str,
        cfg: &TrainConfig,
        phase: &str,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let mut records = Vec::new();
        let mut stopped = false;
        for (weight, group) in weight_groups(texts, weights, name)? {
            if stopped {
                break;
            }
            let scaled = TrainConfig {
                lr: cfg.lr * weight,
                act_lr: cfg.act_lr * weight,
                ..cfg.clone()
            };
            let group_records = self.radix_train(&group, &scaled, Some(phase), &mut |_| true)?;
            // Python stamps the weight on the very dicts the history holds
            let first = self.history.len() - group_records.len();
            for (i, mut record) in group_records.into_iter().enumerate() {
                record.extra.push(("weight".to_string(), Json::Num(weight)));
                self.history[first + i] = record.clone();
                if !on_epoch(&record) {
                    stopped = true;
                }
                records.push(record);
            }
        }
        Ok(records)
    }

    /// 2NRL: train on the bad texts, invert the network, fine-tune on the good
    /// ones with a smaller rate (`pos_lr`, and a tenth of it for the
    /// activations).  Returns `(negative, positive)`.
    pub fn radix_two_nrl(
        &mut self,
        bad: &[String],
        good: &[String],
        o: &Feedback,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<(Vec<EpochRecord>, Vec<EpochRecord>), String> {
        let stopped = std::cell::Cell::new(false);
        let mut watch = |r: &EpochRecord| {
            let go_on = on_epoch(r);
            stopped.set(stopped.get() || !go_on);
            go_on
        };
        let neg_cfg = o.config(o.neg_epochs, o.neg_lr, TrainConfig::default().act_lr);
        let negative = match &o.bad_weights {
            None => self.radix_train(bad, &neg_cfg, Some("negative"), &mut watch)?,
            Some(weights) => self.radix_weighted(bad, weights, "bad_weights", &neg_cfg, "negative", &mut watch)?,
        };
        // inverted even when stopped: never left in the garbage-favouring state
        self.g.radix_invert();
        let mut positive = Vec::new();
        if !stopped.get() {
            let pos_cfg = o.config(o.pos_epochs, o.pos_lr, o.pos_lr / 10.0);
            positive = match &o.good_weights {
                None => self.radix_train(good, &pos_cfg, Some("positive"), &mut watch)?,
                Some(weights) => {
                    self.radix_weighted(good, weights, "good_weights", &pos_cfg, "positive", &mut watch)?
                }
            };
        }
        self.meta.twonrl_runs.add(1);
        Ok((negative, positive))
    }

    /// Thumbs up: a positive-phase pass (`act_lr = lr / 10`); a rating scales
    /// each text's rates by its weight.
    pub fn radix_reward(
        &mut self,
        texts: &[String],
        o: &Feedback,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let cfg = o.config(o.pos_epochs, o.pos_lr, o.pos_lr / 10.0);
        match &o.good_weights {
            None => self.radix_train(texts, &cfg, Some("positive"), on_epoch),
            Some(weights) => self.radix_weighted(texts, weights, "weights", &cfg, "positive", on_epoch),
        }
    }

    /// Thumbs down: a negative-phase pass, then the network is inverted so
    /// those texts become unlikely (unless the run was stopped).
    pub fn radix_punish(
        &mut self,
        texts: &[String],
        o: &Feedback,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let stopped = std::cell::Cell::new(false);
        let mut watch = |r: &EpochRecord| {
            let go_on = on_epoch(r);
            stopped.set(stopped.get() || !go_on);
            go_on
        };
        let cfg = o.config(o.neg_epochs, o.neg_lr, TrainConfig::default().act_lr);
        let records = match &o.bad_weights {
            None => self.radix_train(texts, &cfg, Some("negative"), &mut watch)?,
            Some(weights) => self.radix_weighted(texts, weights, "weights", &cfg, "negative", &mut watch)?,
        };
        if !stopped.get() {
            self.g.radix_invert();
        }
        Ok(records)
    }

    /// Failures made unlikely locally: every other node on each text's path is
    /// moved toward its negation (`flip_nodes`), by the text's amount in
    /// `[0, 1]` - the parity that changes the sign of the most edges.  Returns
    /// `{"texts", "flipped", "unit": "nodes", "mode", "amount_mean"}`.
    pub fn radix_invert_paths(
        &mut self,
        texts: &[String],
        mode: &str,
        amounts: Option<&[f64]>,
    ) -> Result<Json, String> {
        let enc = self.g.enc;
        let kept: Vec<&String> = texts.iter().filter(|t| enc.len(t) >= enc.n).collect();
        let values: Vec<f64> = match amounts {
            None => vec![1.0; kept.len()],
            Some(values) => {
                if values.len() != kept.len() {
                    return Err(format!("amounts has {} entries for {} texts", values.len(), kept.len()));
                }
                values.to_vec()
            }
        };
        if let Some(bad) = values.iter().find(|v| !(0.0..=1.0).contains(*v)) {
            return Err(format!(
                "amounts must lie in [0, 1], got {}",
                crate::json::py_repr(*bad)
            ));
        }
        let mut paths: Vec<Vec<usize>> = Vec::new();
        for text in &kept {
            let grams = enc.encode(text);
            if grams.is_empty() {
                continue;
            }
            let path = match self.g.node_path(&grams) {
                Some(path) => Some(path),
                None => {
                    self.g.observe(&grams, false)?;
                    self.g.node_path(&grams)
                }
            };
            if let Some(path) = path {
                paths.push(path);
            }
        }
        let mut chosen: Vec<(usize, f64)> = Vec::new();
        for (path, &amount) in paths.iter().zip(&values) {
            let real: Vec<usize> = path.iter().copied().filter(|&n| n >= FIRST).collect();
            if real.is_empty() || amount <= 0.0 {
                continue;
            }
            let mut best: Option<(usize, Vec<usize>)> = None;
            for parity in 0..2 {
                let candidate: Vec<usize> = real
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == parity)
                    .map(|(_, &n)| n)
                    .collect();
                let in_state = |n: usize| candidate.contains(&n) || chosen.iter().any(|&(c, _)| c == n);
                let gain = path.windows(2).filter(|w| in_state(w[0]) != in_state(w[1])).count();
                if best.as_ref().is_none_or(|(g, _)| gain > *g) {
                    best = Some((gain, candidate));
                }
            }
            for n in best.map(|(_, c)| c).unwrap_or_default() {
                match chosen.iter_mut().find(|(c, _)| *c == n) {
                    Some((_, a)) => *a = a.max(amount),
                    None => chosen.push((n, amount)),
                }
            }
        }
        let flipped = self.g.flip_nodes(&chosen, mode)?;
        meta_add_extra(&mut self.meta, "path_inversions", flipped as i64);
        let applied: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();
        let mean = if applied.is_empty() {
            0.0
        } else {
            applied.iter().fold(0.0, |acc, v| acc + v) / applied.len() as f64
        };
        Ok(Json::obj([
            ("texts", Json::Int(kept.len() as i64)),
            ("flipped", Json::Int(flipped as i64)),
            ("unit", Json::str("nodes")),
            ("mode", Json::str(mode)),
            ("amount_mean", Json::Num(mean)),
        ]))
    }

    /// The radix model's prediction: `dijkstra` (the default) is the exact
    /// cheapest path, returned as a prediction whose `top` is empty - Python
    /// returns a plain path for it; `beam` and `sample` are the shared search.
    ///
    /// The least-punished traversal is refused: it ranks a walk by the blame
    /// on its worst step, which needs the judged paths this model does not keep.
    pub(crate) fn radix_predict(
        &mut self,
        prefix: &str,
        o: &PredictOptions,
    ) -> Result<crate::beam::Prediction, String> {
        if crate::search::parse_traversal(&o.traversal)? == crate::search::Traversal::LeastPunished {
            return Err(format!(
                "the {} traversal belongs to the count model; this model keeps no record of failure to rank a \
                 walk by (its punishments price a step instead: --traversal punishment)",
                crate::negative::python_repr(&o.traversal)
            ));
        }
        let mode = if o.mode.is_empty() {
            "dijkstra".to_string()
        } else {
            o.mode.to_ascii_lowercase()
        };
        if !matches!(mode.as_str(), "dijkstra" | "beam" | "sample") {
            return Err(format!(
                "unknown mode {}; expected 'dijkstra', 'beam' or 'sample'",
                crate::negative::python_repr(&mode)
            ));
        }
        let traversal = crate::penalty::resolve_traversal(&o.traversal)?;
        if mode != "dijkstra" {
            if o.step_penalty < 0.0 {
                return Err("step_penalty must be >= 0".to_string());
            }
            return self.search(
                prefix,
                o.length,
                &mode,
                o.k,
                o.beam,
                o.step_penalty,
                o.temperature,
                o.to_end,
                o.max_length,
                None,
                traversal,
                o.penalty_scale,
                o.merit_scale,
                o.tuning(),
            );
        }
        let best = self.radix_dijkstra(
            prefix,
            o.length,
            o.max_length,
            o.step_penalty,
            o.to_end,
            traversal,
            o.penalty_scale,
            o.merit_scale,
        )?;
        Ok(crate::beam::Prediction {
            best: best.clone(),
            top: Vec::new(),
            bottom: Vec::new(),
            k: o.k,
            beam: 0,
            mode: "dijkstra".to_string(),
            traversal: traversal.to_string(),
            expanded: best.expanded,
        })
    }

    /// The exact cheapest continuation of `prefix` (`RadixNet.predict` in
    /// `dijkstra` mode): no limit on the units emitted unless `max_length` is
    /// given, and the unmatched rest of a partly matched gram opens it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn radix_dijkstra(
        &mut self,
        prefix: &str,
        length: usize,
        max_length: Option<usize>,
        step_penalty: f64,
        to_end: bool,
        traversal: &str,
        penalty_scale: f64,
        merit_scale: f64,
    ) -> Result<PathResult, String> {
        let (node, offset, lead) = self.prefix_start(prefix);
        // `len(lead)`: Python counts the lead in characters
        let lead_len = lead.chars().count();
        let want = length.saturating_sub(lead_len);
        let (cap, max_chars) = match max_length {
            None if length == 0 => (Some(0), Some(0)), // "emit nothing" stays empty
            None => (None, None),                      // no limit: the whole cheapest path
            Some(max_length) => {
                let cap = length.max(max_length);
                (Some(cap), Some(want.max(cap.saturating_sub(lead_len))))
            }
        };
        let priced = crate::penalty::traversal_costs(traversal, penalty_scale, merit_scale)?;
        let mut result = dijkstra_predict(
            &mut self.g,
            node,
            offset,
            want,
            max_chars,
            step_penalty,
            to_end,
            None,
            priced.as_ref(),
        )?;
        let enc = self.g.enc;
        if !lead.is_empty() {
            let joined = enc.join(&[&lead, &result.text]);
            result.text = match cap {
                None => joined,
                Some(cap) => enc.truncate(&joined, cap),
            };
        }
        result.full_text = enc.join(&[prefix, &result.text]);
        Ok(result)
    }
}

// -- what it says about itself ----------------------------------------------------------------------

/// The radix model's statistics, with the keys Python's `RadixNet.stats` reports.
pub fn stats_json(model: &Model) -> Json {
    let g = &model.g;
    let enc = g.enc;
    let meta = &model.meta;
    let mut pairs: Vec<(String, Json)> = vec![
        ("kind".to_string(), Json::str("radix")),
        ("nodes".to_string(), Json::Int(g.num_nodes() as i64)),
        ("edges".to_string(), Json::Int(g.num_edges() as i64)),
        ("trigrams".to_string(), Json::Int(g.num_trigrams() as i64)),
        ("grams".to_string(), Json::Int(g.num_trigrams() as i64)),
        ("encoding".to_string(), Json::str(enc.to_string())),
        ("unit".to_string(), Json::str(enc.unit.name())),
        ("units".to_string(), Json::str(enc.units_name())),
        ("ngram".to_string(), Json::Int(enc.n as i64)),
        ("stride".to_string(), Json::Int(enc.stride as i64)),
        ("compression_ratio".to_string(), Json::Num(g.compression_ratio())),
        ("inverted".to_string(), Json::Bool(g.inverted)),
        ("backend".to_string(), Json::str(crate::backend::NAME)),
        ("device".to_string(), Json::str(crate::backend::DEVICE)),
    ];
    for (key, counter) in [
        ("epochs_total", meta.epochs_total),
        ("trained_chars", meta.trained_chars),
        ("trained_texts", meta.trained_texts),
        ("twonrl_runs", meta.twonrl_runs),
    ] {
        pairs.push((key.to_string(), Json::Int(counter.value)));
        pairs.push((format!("{key}_resets"), Json::Int(counter.resets)));
    }
    pairs.push(("history_len".to_string(), Json::Int(model.history.len() as i64)));
    pairs.push((
        "last_loss".to_string(),
        model.history.last().map(|r| Json::Num(r.loss)).unwrap_or(Json::Null),
    ));
    Json::Obj(pairs)
}

// -- the file -------------------------------------------------------------------------------------

/// The `meta` block of a radix file: the four lifetime counters, then whatever
/// else it carried (`path_inversions`, keys another implementation keeps).
pub(crate) fn meta_json(model: &Model) -> Json {
    let meta = &model.meta;
    let mut pairs: Vec<(String, Json)> = vec![
        ("created".to_string(), Json::str(meta.created.clone())),
        ("seed".to_string(), Json::Int(meta.seed)),
    ];
    for (key, c) in [
        ("epochs_total", meta.epochs_total),
        ("trained_chars", meta.trained_chars),
        ("trained_texts", meta.trained_texts),
        ("twonrl_runs", meta.twonrl_runs),
    ] {
        pairs.push((key.to_string(), Json::Int(c.value)));
        pairs.push((format!("{key}_resets"), Json::Int(c.resets)));
    }
    pairs.extend(meta.extra.iter().cloned());
    Json::Obj(pairs)
}

/// The model as the `radixnet` document Python writes: no `kind` (the format
/// is the kind), the backend that trained it, and the graph with every node's
/// parameters and no weight function (the weights are learned, not computed).
pub(crate) fn model_doc(model: &mut Model) -> Json {
    Json::obj([
        ("format", Json::str(RADIX_FORMAT)),
        ("version", Json::Int(crate::file::MODEL_FORMAT_VERSION)),
        ("saved_at", Json::str(crate::clock::utc_now())),
        ("meta", meta_json(model)),
        (
            "history",
            Json::Arr(model.history.iter().map(EpochRecord::to_json).collect()),
        ),
        ("backend", Json::str(crate::backend::NAME)),
        ("graph", model.g.to_doc()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts() -> Vec<String> {
        [
            "the cat sat on the mat",
            "the cat sat on the log",
            "a bird flew over the hill",
            "the dog sat on the mat",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn trained(seed: i64) -> Model {
        let mut m = Model::new_radix(seed, Encoding::default()).unwrap();
        let cfg = TrainConfig {
            epochs: 3,
            lr: 0.5,
            batch_size: 4,
            ..Default::default()
        };
        m.radix_train(&texts(), &cfg, None, &mut |_| true).unwrap();
        m
    }

    #[test]
    fn shuffle_is_pythons() {
        // r = random.Random(1); x = list(range(10)); r.shuffle(x)
        let mut rng = Mt19937::new(1);
        let mut x: Vec<usize> = (0..10).collect();
        shuffle(&mut rng, &mut x);
        assert_eq!(x, vec![6, 8, 9, 7, 5, 3, 0, 4, 1, 2]);
        // r = random.Random(7); [r.getrandbits(k) for k in (1, 5, 31, 32, 33, 64)]
        let mut rng = Mt19937::new(7);
        let bits: Vec<u64> = [1, 5, 31, 32, 33, 64]
            .iter()
            .map(|&k| getrandbits(&mut rng, k))
            .collect();
        assert_eq!(
            bits,
            vec![0, 30, 323946139, 1695753998, 2795742288, 15149836622520594227]
        );
    }

    #[test]
    fn training_lowers_the_loss_and_keeps_the_structure_sound() {
        let m = trained(3);
        assert_eq!(m.kind(), "radix");
        let first = m.history.first().unwrap().loss;
        let last = m.history.last().unwrap().loss;
        assert!(last < first, "{last} >= {first}");
        m.g.check_invariants(&texts(), false).unwrap();
        let r = m.g.radix.as_ref().unwrap();
        assert_eq!(r.z.len(), m.g.num_node_ids());
        assert!(r.b.iter().all(|&b| b >= crate::backend::MIN_B));
    }

    #[test]
    fn a_merge_changes_no_probability() {
        let mut g = Graph::new_radix(5, Encoding::default()).unwrap();
        let grams = g.enc.encode("hello world");
        g.observe(&grams, true).unwrap();
        // give the chain's nodes different activations, then merge it
        if let Some(r) = g.radix.as_mut() {
            for (i, z) in r.z.iter_mut().enumerate() {
                *z = (i as f64 * 0.37).sin() * 4.0;
            }
        }
        g.version.add(1);
        g.prepare();
        let before: Vec<f64> = g.child_costs(START).iter().map(|c| c.cost).collect();
        let start_child = g.children_of(START)[0].p;
        let path_cost = |g: &mut Graph| -> f64 {
            g.prepare();
            let (tr, _) = g.trace(&grams).unwrap();
            tr.iter().map(|t| g.edge_cost(t.e)).sum()
        };
        let cost_before = path_cost(&mut g);
        assert!(g.compress() > 0);
        let cost_after = path_cost(&mut g);
        assert!(
            (cost_before - cost_after).abs() < 1e-12,
            "{cost_before} vs {cost_after}"
        );
        let after: Vec<f64> = g.child_costs(START).iter().map(|c| c.cost).collect();
        assert_eq!(before.len(), after.len());
        assert!(g.is_alive(start_child));
    }

    #[test]
    fn inverting_twice_is_the_identity() {
        let mut m = trained(4);
        let before = m.g.node_params();
        let w = m.g.edge_w.clone();
        m.g.radix_invert();
        assert!(m.g.inverted);
        m.g.radix_invert();
        assert_eq!(m.g.node_params(), before);
        assert_eq!(m.g.edge_w, w);
    }

    #[test]
    fn it_predicts_by_the_cheapest_path() {
        let mut m = trained(6);
        let found = m
            .predict(
                "the cat",
                &PredictOptions {
                    mode: "dijkstra".to_string(),
                    length: 8,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(found.top.is_empty(), "a dijkstra answer is a single path");
        assert!(found.best.full_text.starts_with("the cat"));
        let refused = m.predict(
            "the cat",
            &PredictOptions {
                traversal: "least-punished".to_string(),
                ..Default::default()
            },
        );
        assert!(refused.unwrap_err().contains("belongs to the count model"));
    }

    #[test]
    fn flipping_a_path_makes_it_unlikely() {
        let mut m = trained(8);
        let text = "the cat sat on the mat".to_string();
        let before = m.score(&text).log_prob;
        let out = m
            .radix_invert_paths(std::slice::from_ref(&text), "activation", None)
            .unwrap();
        assert!(out.at("flipped").as_i64().unwrap() > 0);
        let after = m.score(&text).log_prob;
        assert!(after < before, "{after} >= {before}");
        assert!(m.g.flip_nodes(&[(3, 1.0)], "sideways").is_err());
    }

    #[test]
    fn a_rating_groups_texts_by_weight() {
        let t: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let groups = weight_groups(&t, &[0.5, 1.0, 0.0, 0.5004], "weights").unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], (1.0, vec!["b".to_string()]));
        assert_eq!(groups[1], (0.5, vec!["a".to_string(), "d".to_string()]));
        assert!(weight_groups(&t, &[1.0], "weights").is_err());
        assert!(weight_groups(&t, &[1.0, -1.0, 0.0, 0.0], "weights").is_err());
    }
}
