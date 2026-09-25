//! The resonant model - the phase model: an analog carrier on the same
//! self-compressing graph.
//!
//! `Research/SineWaveActivationFunction.md`: *a brain operates like an analog
//! computer, hence sine waves are the standard of information
//! encoding/decoding.*  The radix model takes that as a pointwise sine; this
//! one takes the other half of the idea: a sine wave has a **phase**, phases
//! **add** along a path, and two signals that meet in phase reinforce while two
//! that meet in antiphase cancel.
//!
//! A walk carries one more number than the node it stands on: the phase, one
//! of `buckets` positions on a ring.  Every gram advances it by a fixed
//! integer - a clock (`buckets / period` per gram) plus `kick_scale` times the
//! gram's own phase, a BLAKE2b hash of its text ([`crate::blake2b`]) - and a
//! node advances it by the sum over the grams its label covers, which is what
//! makes the phase exactly invariant under the radix split and merge.
//!
//! An edge learns the phases at which it was actually taken, as a circular
//! mean: each traversal at bucket `b` adds the unit vector of that bucket to
//! the edge's accumulator `(cx, cy)`, of total weight `cw`.  That gives the mean
//! phase `mu` at which the transition fires and its **coherence**
//! `r = |(cx, cy)| / (cw + concentration)`, and the score of an edge is
//!
//! ```text
//! score(p -> c | psi) = amplitude + resonance_scale * r * cos(psi - mu)
//! amplitude           = amp_scale * log(share of the parent) + reward_scale * reward
//! ```
//!
//! so an incoherent edge competes on its frequency alone and a coherent one is
//! cheap in phase and dear out of phase.  Learning is counting; `invert`
//! negates the accumulators, which rotates every `mu` by pi.
//!
//! A phase-locked cycle - re-entering a `(node, phase)` the walk has already
//! been in - hands the decision to the metacognitive layer
//! ([`crate::metacog`]); the searches over `(node, units, phase)` are in
//! [`crate::phasesearch`].
//!
//! # A variant of the graph
//!
//! As with the negative network, the structure is the graph's and what the
//! kind keeps beside it hangs on `Graph::res` ([`ResonantData`]); the model's
//! layer and its cycle settings hang on `Model::res` ([`ResonantModel`]).  The
//! rewards are the graph's own `edge_reward` and `reward_scale`, which mean the
//! same thing here as in the count model.
//!
//! This is `radixnet/resonance.py`, and it keeps that file's one subtle
//! property: the phase-aware costs are cached per `(node, bucket)` until the
//! graph's version changes, and a training walk counts traversals without
//! changing the version - so within a pass a walk reads the costs as they were
//! when the pass first asked, exactly as Python's does.

use std::f64::consts::PI;
use std::sync::Mutex;
use std::time::Instant;

use crate::counter::Counter;
use crate::encoding::Encoding;
use crate::fsum::fsum;
use crate::graph::{Graph, GraphOptions, BACK, END, FIRST, START, THINK};
use crate::hash::{map, Map};
use crate::json::Json;
use crate::metacog::{cycle_signature, MetaLayer, ABORT, ESCAPE, RIDE};
use crate::model::{EpochRecord, GenerateOptions, Model, PredictOptions, Score};
use crate::mt19937::Mt19937;
use crate::penalty::EdgeEvidence;
use crate::search::PathResult;
use crate::weights::ChildCost;

/// The file format of the phase model.
pub const RESONANT_FORMAT: &str = "radixnet-resonant";

/// A full turn.
pub const TAU: f64 = 2.0 * PI;

/// The add-`s` smoothing of an edge's share of its parent's traversals.
pub const SMOOTHING: f64 = 0.5;

/// `-log(1e-6)`: what an unknown step costs a walk.
fn unknown_cost() -> f64 {
    -(1e-6f64).ln()
}

/// What the model's own lines are filed under.
const LOG: &str = "train";

/// A gram's own phase in `[0, TAU)`: a stable hash, the same in every process
/// and every implementation - the first eight bytes of its BLAKE2b digest,
/// big-endian, as a fraction of 2^64.
pub fn trigram_phase(gram: &str) -> f64 {
    let digest = crate::blake2b::blake2b(gram.as_bytes(), 8);
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest);
    u64::from_be_bytes(bytes) as f64 / 18446744073709551616.0 * TAU
}

/// Python's `math.hypot(x, y)`, which is not the C library's: CPython 3.11
/// computes the norm with its own error-free scaled sum (`vector_norm` in
/// `Modules/mathmodule.c`), and the two disagree in the last bit about once
/// in a thousand pairs - enough to move a cost, so the port does the same
/// arithmetic, step for step.
pub(crate) fn py_hypot(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax.is_infinite() || ay.is_infinite() {
        return f64::INFINITY;
    }
    if ax.is_nan() || ay.is_nan() {
        return f64::NAN;
    }
    let max = if ax > ay { ax } else { ay };
    vector_norm(&[ax, ay], max)
}

/// CPython 3.11's `vector_norm` over finite, non-negative coordinates.
fn vector_norm(vec: &[f64], max: f64) -> f64 {
    const T27: f64 = 134_217_729.0; // 2**27 + 1: Dekker's split
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        // ldexp(1.0, -max_e) would overflow: bring the subnormals up first
        let scaled: Vec<f64> = vec.iter().map(|v| v / f64::MIN_POSITIVE).collect();
        return f64::MIN_POSITIVE * vector_norm(&scaled, max / f64::MIN_POSITIVE);
    }
    let scale = ldexp(1.0, -max_e);
    let (mut csum, mut frac1, mut frac2, mut frac3) = (1.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &v in vec {
        let x = v * scale;
        let t = x * T27;
        let hi = t - (t - x);
        let lo = x - hi;
        let sq = hi * hi;
        let old = csum;
        csum += sq;
        frac1 += (old - csum) + sq;
        let cross = 2.0 * hi * lo;
        let old = csum;
        csum += cross;
        frac2 += (old - csum) + cross;
        frac3 += lo * lo;
    }
    let h = (csum - 1.0 + (frac1 + frac2 + frac3)).sqrt();
    let t = h * T27;
    let hi = t - (t - h);
    let lo = h - hi;
    let x = -hi * hi;
    let old = csum;
    csum += x;
    frac1 += (old - csum) + x;
    let x = -2.0 * hi * lo;
    let old = csum;
    csum += x;
    frac2 += (old - csum) + x;
    let x = -lo * lo;
    let old = csum;
    csum += x;
    frac3 += (old - csum) + x;
    let x = csum - 1.0 + (frac1 + frac2 + frac3);
    (h + x / (2.0 * h)) / scale
}

/// The exponent `frexp` returns: `v = m * 2**e` with `0.5 <= m < 1`.
fn frexp_exponent(v: f64) -> i32 {
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // subnormal: normalise through a power of two
        return frexp_exponent(v * 2f64.powi(64)) - 64;
    }
    exp - 1022
}

/// `ldexp(x, e)` for the powers of two `vector_norm` scales by.
fn ldexp(x: f64, e: i32) -> f64 {
    if e > 1023 {
        x * 2f64.powi(1023) * 2f64.powi(e - 1023)
    } else if e < -1022 {
        x * 2f64.powi(-1022) * 2f64.powi(e + 1022)
    } else {
        x * f64::from_bits(((e + 1023) as u64) << 52)
    }
}

/// Python's float `%`: the result takes the sign of the divisor.
fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// The score function's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResonantOptions {
    pub buckets: usize,
    /// Units per turn of the clock; 0 means `buckets` (one bucket per unit).
    pub period: f64,
    pub kick_scale: f64,
    pub resonance_scale: f64,
    pub amp_scale: f64,
    pub reward_scale: f64,
    pub concentration: f64,
}

impl Default for ResonantOptions {
    /// Python's defaults: eight buckets, a pure clock.
    fn default() -> ResonantOptions {
        ResonantOptions {
            buckets: 8,
            period: 0.0,
            kick_scale: 0.0,
            resonance_scale: 1.0,
            amp_scale: 1.0,
            reward_scale: 1.0,
            concentration: 2.0,
        }
    }
}

/// The phase-aware costs, cached per `(node, bucket)` for one graph version.
#[derive(Default)]
struct PhaseCache {
    version: Option<Counter>,
    costs: Map<(usize, usize), Vec<ChildCost>>,
}

/// Everything a phase graph keeps that a count graph does not.
pub struct ResonantData {
    pub buckets: usize,
    pub period: f64,
    pub kick_scale: f64,
    pub resonance_scale: f64,
    pub amp_scale: f64,
    pub concentration: f64,
    /// Buckets traversing each node moves the phase (0 for a sentinel).
    pub advance: Vec<usize>,
    /// Each edge's circular accumulator: the summed unit vectors of the phases
    /// it fired at, and their total weight.
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    pub cw: Vec<f64>,
    advance_cache: Mutex<Map<String, usize>>,
    cache: Mutex<PhaseCache>,
}

impl ResonantData {
    fn new(o: &ResonantOptions) -> ResonantData {
        ResonantData {
            buckets: o.buckets,
            period: if o.period != 0.0 { o.period } else { o.buckets as f64 },
            kick_scale: o.kick_scale,
            resonance_scale: o.resonance_scale,
            amp_scale: o.amp_scale,
            concentration: o.concentration,
            advance: Vec::new(),
            cx: Vec::new(),
            cy: Vec::new(),
            cw: Vec::new(),
            advance_cache: Mutex::new(map()),
            cache: Mutex::new(PhaseCache::default()),
        }
    }

    /// Buckets one gram moves the phase: the clock plus `kick_scale` times its
    /// own phase, rounded half to even as Python's `round` does.
    pub fn gram_advance(&self, gram: &str) -> usize {
        let mut cache = self.advance_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&v) = cache.get(gram) {
            return v;
        }
        let clock = self.buckets as f64 / self.period;
        let kick = self.kick_scale * trigram_phase(gram) * self.buckets as f64 / TAU;
        let value = ((clock + kick).round_ties_even() as i64).rem_euclid(self.buckets as i64) as usize;
        cache.insert(gram.to_string(), value);
        value
    }

    /// The angle in `[0, TAU)` a bucket stands for.
    pub fn bucket_phase(&self, bucket: usize) -> f64 {
        TAU * (bucket % self.buckets) as f64 / self.buckets as f64
    }

    /// How consistently edge `e` fires at one phase, in `[0, 1]`.
    pub fn coherence(&self, e: usize) -> f64 {
        let w = self.cw[e];
        if w <= 0.0 {
            return 0.0;
        }
        py_hypot(self.cx[e], self.cy[e]) / (w + self.concentration)
    }

    /// The mean phase at which edge `e` fires (0 when it never has).
    pub fn mu(&self, e: usize) -> f64 {
        let (x, y) = (self.cx[e], self.cy[e]);
        if x != 0.0 || y != 0.0 {
            py_mod(y.atan2(x), TAU)
        } else {
            0.0
        }
    }

    fn append_edge(&mut self) {
        self.cx.push(0.0);
        self.cy.push(0.0);
        self.cw.push(0.0);
    }

    /// The score function's settings, as Python's `weight_config` lists them.
    pub fn weight_config(&self, reward_scale: f64) -> Json {
        Json::obj([
            ("buckets", Json::Int(self.buckets as i64)),
            ("period", Json::Num(self.period)),
            ("kick_scale", Json::Num(self.kick_scale)),
            ("resonance_scale", Json::Num(self.resonance_scale)),
            ("amp_scale", Json::Num(self.amp_scale)),
            ("reward_scale", Json::Num(reward_scale)),
            ("concentration", Json::Num(self.concentration)),
        ])
    }
}

// -- the graph ------------------------------------------------------------------------------------

impl Graph {
    /// An empty phase graph with the given score function.
    pub fn new_resonant(seed: i64, o: &ResonantOptions, encoding: Encoding) -> Result<Graph, String> {
        if o.buckets < 1 {
            return Err(format!("buckets must be >= 1, got {}", o.buckets));
        }
        if o.period < 0.0 || o.period.is_nan() {
            return Err(format!("period must be > 0, got {}", crate::json::py_repr(o.period)));
        }
        if o.concentration < 0.0 {
            return Err(format!(
                "concentration must be >= 0, got {}",
                crate::json::py_repr(o.concentration)
            ));
        }
        let mut g = Graph::new(
            seed,
            GraphOptions {
                encoding,
                ..Default::default()
            },
        )?;
        g.reward_scale = o.reward_scale;
        let mut data = ResonantData::new(o);
        // the sentinels emit nothing, so they move the phase by nothing
        data.advance = vec![0; g.num_node_ids()];
        data.cx = vec![0.0; g.num_edge_ids()];
        data.cy = vec![0.0; g.num_edge_ids()];
        data.cw = vec![0.0; g.num_edge_ids()];
        g.res = Some(Box::new(data));
        Ok(g)
    }

    /// Whether this graph carries a phase.
    pub fn is_resonant(&self) -> bool {
        self.res.is_some()
    }

    /// Buckets a label moves the phase: the sum over the grams it covers.
    pub fn label_advance(&self, label: &str) -> usize {
        let Some(r) = self.res.as_ref() else { return 0 };
        let total: usize = self.enc.encode(label).iter().map(|g| r.gram_advance(g)).sum();
        total % r.buckets
    }

    /// The phase of a piece of text - the bucket a walk that emitted it carries.
    pub fn text_bucket(&self, text: &str) -> usize {
        let Some(r) = self.res.as_ref() else { return 0 };
        let total: usize = self.enc.encode(text).iter().map(|g| r.gram_advance(g)).sum();
        total % r.buckets
    }

    fn node_advance(&self, node: usize) -> usize {
        if node < FIRST {
            0
        } else {
            self.label_advance(&self.labels[node])
        }
    }

    /// Hook of `new_node`, `split` and `merge_child`: a node's advance, from
    /// its label as it stands now.
    pub(crate) fn resonant_refresh(&mut self, node: usize) {
        if self.res.is_none() {
            return;
        }
        let value = self.node_advance(node);
        let r = self.res.as_mut().expect("checked");
        if node < r.advance.len() {
            r.advance[node] = value;
        } else {
            r.advance.resize(node, 0);
            r.advance.push(value);
        }
    }

    /// Hook of `new_edge`: an empty accumulator.
    pub(crate) fn resonant_edge_added(&mut self) {
        if let Some(r) = self.res.as_mut() {
            r.append_edge();
        }
    }

    /// Counts one traversal of `e` into its accumulator at `bucket` - or,
    /// with no bucket, without a phase: the weight grows but the vector does
    /// not, so the edge's coherence falls and it competes on its share alone.
    pub fn record_phase(&mut self, e: usize, bucket: Option<usize>, amount: f64) {
        let Some(r) = self.res.as_mut() else { return };
        if let Some(b) = bucket {
            let angle = r.bucket_phase(b);
            r.cx[e] += amount * angle.cos();
            r.cy[e] += amount * angle.sin();
        }
        r.cw[e] += amount.abs();
        self.bump_edge(e);
        self.total_traversals.add(1);
        self.add_traversals(1); // what bounds every counter
    }

    /// Moves the reward of every alive listed edge; recomputes the weights
    /// when anything moved.  Returns how many moved.
    pub fn resonant_add_reward(&mut self, edges: &[usize], amount: f64) -> usize {
        let mut moved = 0;
        for &e in edges {
            if e < self.edge_reward.len() && self.edge_alive[e] {
                self.edge_reward[e] += amount;
                moved += 1;
            }
        }
        if moved > 0 {
            self.resonant_recompute();
        }
        moved
    }

    /// Scales the accumulators: `> 1` locks an edge to its phase, `< 1`
    /// decoheres it; the resultant never exceeds the accumulated weight.
    pub fn sharpen(&mut self, edges: &[usize], factor: f64) -> Result<usize, String> {
        if factor < 0.0 {
            return Err(format!("factor must be >= 0, got {}", crate::json::py_repr(factor)));
        }
        let alive = &self.edge_alive;
        let Some(r) = self.res.as_mut() else { return Ok(0) };
        let mut changed = 0;
        for &e in edges {
            if e >= r.cx.len() || !alive[e] {
                continue;
            }
            let (mut x, mut y) = (r.cx[e] * factor, r.cy[e] * factor);
            let cap = r.cw[e];
            let length = py_hypot(x, y);
            if cap > 0.0 && length > cap {
                x = x * cap / length;
                y = y * cap / length;
            }
            r.cx[e] = x;
            r.cy[e] = y;
            changed += 1;
        }
        if changed > 0 {
            self.version.add(1);
        }
        Ok(changed)
    }

    /// Rotates the mean phase of edges by `turns` half-circles (1 = antiphase).
    pub fn rotate(&mut self, edges: &[usize], turns: f64) -> usize {
        let angle = PI * turns;
        let (ca, sa) = (angle.cos(), angle.sin());
        let alive = &self.edge_alive;
        let Some(r) = self.res.as_mut() else { return 0 };
        let mut changed = 0;
        for &e in edges {
            if e >= r.cx.len() || !alive[e] {
                continue;
            }
            let (x, y) = (r.cx[e], r.cy[e]);
            r.cx[e] = x * ca - y * sa;
            r.cy[e] = x * sa + y * ca;
            changed += 1;
        }
        if changed > 0 {
            self.version.add(1);
        }
        changed
    }

    /// Rewrites every edge weight from the counts and rewards: the phase-free
    /// part of the score, `amp_scale * log(share) + reward_scale * reward`.
    pub(crate) fn resonant_recompute(&mut self) {
        let Some(r) = self.res.as_ref() else { return };
        let (amp, rs) = (r.amp_scale, self.reward_scale);
        for p in 0..self.labels.len() {
            let adj = &self.children[p];
            if adj.size() == 0 || !self.alive[p] {
                continue;
            }
            let mut total = 0.0;
            for &e in &adj.edges {
                total += self.edge_traversals_f(e);
            }
            let degree = adj.size().max(1) as f64;
            for &e in &adj.edges {
                let share = (self.edge_traversals_f(e) + SMOOTHING) / (total + SMOOTHING * degree);
                self.edge_w[e] = amp * share.ln() + rs * self.edge_reward[e];
            }
        }
        self.dirty.clear();
        self.dirty_all = false;
        self.weights_structure = self.structure_version;
        self.version.add(1);
    }

    /// Rotates every edge's mean phase by pi and negates every reward: what
    /// arrived in phase now arrives in antiphase.
    pub(crate) fn resonant_invert(&mut self) {
        let alive = self.edge_alive.clone();
        if let Some(r) = self.res.as_mut() {
            for (e, ok) in alive.iter().enumerate() {
                if *ok {
                    r.cx[e] = -r.cx[e];
                    r.cy[e] = -r.cy[e];
                    self.edge_reward[e] = -self.edge_reward[e];
                }
            }
        }
        self.inverted = !self.inverted;
        self.resonant_recompute();
    }

    /// Changes the score function; changing `buckets`, `period` or
    /// `kick_scale` redefines the phase, so every advance is recomputed (what
    /// the edges learned is kept - their means refer to the new ring).
    pub fn configure_resonant(&mut self, options: &[(String, f64)]) -> Result<Json, String> {
        if self.res.is_none() {
            return Err("not a resonant graph".to_string());
        }
        let known = [
            "buckets",
            "period",
            "kick_scale",
            "resonance_scale",
            "amp_scale",
            "reward_scale",
            "concentration",
        ];
        let mut unknown: Vec<&str> = options
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !known.contains(k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(format!("unknown weight option(s): {}", unknown.join(", ")));
        }
        let mut rephase = false;
        for name in known {
            let Some(&(_, value)) = options.iter().rev().find(|(k, _)| k == name) else {
                continue;
            };
            let reward_scale = self.reward_scale;
            let r = self.res.as_mut().expect("checked");
            let changed = match name {
                "buckets" => {
                    let value = value.trunc();
                    if value < 1.0 {
                        return Err(format!("buckets must be >= 1, got {}", value as i64));
                    }
                    let old = r.buckets;
                    r.buckets = value as usize;
                    old != r.buckets
                }
                "period" => {
                    if value.is_nan() || value <= 0.0 {
                        return Err(format!("period must be > 0, got {}", crate::json::py_repr(value)));
                    }
                    let old = r.period;
                    r.period = value;
                    old != value
                }
                "kick_scale" => {
                    let old = r.kick_scale;
                    r.kick_scale = value;
                    old != value
                }
                "resonance_scale" => {
                    r.resonance_scale = value;
                    false
                }
                "amp_scale" => {
                    r.amp_scale = value;
                    false
                }
                "concentration" => {
                    if value < 0.0 {
                        return Err(format!(
                            "concentration must be >= 0, got {}",
                            crate::json::py_repr(value)
                        ));
                    }
                    r.concentration = value;
                    false
                }
                _ => {
                    let _ = reward_scale;
                    self.reward_scale = value;
                    false
                }
            };
            rephase |= changed && matches!(name, "buckets" | "period" | "kick_scale");
        }
        if rephase {
            let r = self.res.as_ref().expect("checked");
            r.advance_cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
            for n in 0..self.labels.len() {
                if self.alive[n] {
                    self.resonant_refresh(n);
                }
            }
        }
        self.resonant_recompute();
        Ok(self.res.as_ref().expect("checked").weight_config(self.reward_scale))
    }

    // -- phase-aware scores ---------------------------------------------------

    /// `[(child, edge, amplitude + resonance_scale * coherence * cos(phase - mu))]`.
    pub fn child_scores_at(&self, p: usize, bucket: usize) -> Vec<(usize, usize, f64)> {
        let Some(r) = self.res.as_ref() else { return Vec::new() };
        let angle = r.bucket_phase(bucket);
        let rs = r.resonance_scale;
        let adj = &self.children[p];
        adj.order
            .iter()
            .zip(adj.edges.iter())
            .map(|(&c, &e)| {
                let mut score = self.edge_w[e];
                if rs != 0.0 {
                    let w = r.cw[e];
                    if w > 0.0 {
                        let (x, y) = (r.cx[e], r.cy[e]);
                        let coherence = py_hypot(x, y) / (w + r.concentration);
                        if coherence > 0.0 {
                            score += rs * coherence * (angle - y.atan2(x)).cos();
                        }
                    }
                }
                (c, e, score)
            })
            .collect()
    }

    /// `[(child, edge, -log softmax)]` at one phase, cached until the version
    /// changes - and only then, as Python's cache is.
    pub fn child_costs_at(&self, p: usize, bucket: usize) -> Vec<ChildCost> {
        let Some(r) = self.res.as_ref() else { return Vec::new() };
        let key = (p, bucket % r.buckets);
        {
            let mut cache = r.cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache.version != Some(self.version) {
                cache.costs.clear();
                cache.version = Some(self.version);
            }
            if let Some(found) = cache.costs.get(&key) {
                return found.clone();
            }
        }
        let costs = softmax_costs(&self.child_scores_at(p, bucket));
        let mut cache = r.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.costs.insert(key, costs.clone());
        costs
    }

    /// The merit / penalty split at one phase: the resonance is merit (what the
    /// corpus did), the negative half of the reward the penalty.
    pub fn child_evidence_at(&self, p: usize, bucket: usize) -> Vec<EdgeEvidence> {
        let rs = self.reward_scale;
        self.child_scores_at(p, bucket)
            .into_iter()
            .map(|(c, e, score)| {
                let reward = self.edge_reward[e];
                EdgeEvidence {
                    child: c,
                    edge: e,
                    merit: score - rs * reward,
                    penalty: rs * if -reward > 0.0 { -reward } else { 0.0 },
                }
            })
            .collect()
    }

    /// `P(hand over | p)`: the share `p`'s `BACK` edge takes of its children
    /// under the phase-free weights, 0 when it has none.
    pub fn back_probability(&self, p: usize) -> f64 {
        let adj = &self.children[p];
        if adj.get(BACK).is_none() {
            return 0.0;
        }
        let scores: Vec<(usize, usize, f64)> = adj
            .order
            .iter()
            .zip(adj.edges.iter())
            .map(|(&c, &e)| (c, e, self.edge_w[e]))
            .collect();
        // the base class's `child_costs`: a plain left-to-right sum
        let mut m = f64::NEG_INFINITY;
        for &(_, _, s) in &scores {
            if s > m {
                m = s;
            }
        }
        let mut sum = 0.0;
        for &(_, _, s) in &scores {
            sum += (s - m).exp();
        }
        let lse = m + sum.ln();
        scores
            .iter()
            .find(|(c, _, _)| *c == BACK)
            .map(|&(_, _, s)| (-(lse - s)).exp())
            .unwrap_or(0.0)
    }

    /// `observe_back` as this model learns everything - by counting: the
    /// `BACK` edge is counted *without a phase* (the voice that backed out
    /// walked outside the search and cannot say which phase it was in), the
    /// step it looped through penalised and the one it took instead rewarded.
    pub(crate) fn resonant_observe_back(
        &mut self,
        p: usize,
        went: Option<usize>,
        instead: Option<usize>,
        amount: f64,
    ) -> Result<usize, String> {
        if p < FIRST || p >= self.labels.len() || !self.alive[p] {
            return Err(format!("node {p} is not a real node to go back from"));
        }
        let e = match self.edge(p, BACK) {
            Some(e) => e,
            None => self.new_edge(p, BACK, 0, 0),
        };
        self.bump_node(BACK);
        self.bump_edge(e);
        self.add_traversals(1);
        self.version.add(1);
        self.record_phase(e, None, if amount != 0.0 { amount } else { 1.0 });
        self.resonant_add_reward(&[e], amount);
        for (child, sign) in [(went, -1.0), (instead, 1.0)] {
            let edge = child.filter(|&c| c != BACK).and_then(|c| self.edge(p, c));
            if let Some(edge) = edge {
                self.resonant_add_reward(&[edge], sign * amount);
            }
        }
        self.resonant_recompute();
        Ok(e)
    }

    /// `observe_think` as this model learns everything - by counting: the
    /// `THINK` edge is counted *without a phase* (an event is not a walk of this
    /// model's search and cannot say which phase it was in) and rewarded.
    pub(crate) fn resonant_observe_think(&mut self, p: usize, amount: f64) -> Result<usize, String> {
        if p < FIRST || p >= self.labels.len() || !self.alive[p] {
            return Err(format!("node {p} is not a real node to think at"));
        }
        if amount < 0.0 {
            return Err(format!("amount must be >= 0, got {amount}"));
        }
        let e = match self.edge(p, THINK) {
            Some(e) => e,
            None => self.new_edge(p, THINK, 0, 0),
        };
        self.bump_node(THINK);
        self.bump_edge(e);
        self.add_traversals(1);
        self.version.add(1);
        self.record_phase(e, None, if amount != 0.0 { amount } else { 1.0 });
        self.resonant_add_reward(&[e], amount);
        self.resonant_recompute();
        Ok(e)
    }

    /// The opposite lesson: a walk at `p` faced a cycle and carried on, so its
    /// hand-over estimate is pushed back down.  `false` when `p` has no `BACK`
    /// edge to push on.
    pub fn observe_onward(&mut self, p: usize, amount: f64) -> bool {
        if !(FIRST..self.children.len()).contains(&p) || amount <= 0.0 {
            return false;
        }
        let Some(e) = self.edge(p, BACK) else { return false };
        self.resonant_add_reward(&[e], -amount);
        true
    }

    /// The per-edge arrays in file order, or `None` on another kind of graph.
    pub(crate) fn resonant_arrays(&self, edges: &[usize]) -> Option<[Vec<f64>; 4]> {
        let r = self.res.as_ref()?;
        let pick = |v: &Vec<f64>| edges.iter().map(|&e| v[e]).collect::<Vec<f64>>();
        Some([pick(&r.cx), pick(&r.cy), pick(&r.cw), pick(&self.edge_reward)])
    }

    /// The `weights` block of a phase graph's document.
    pub(crate) fn resonant_weights_doc(&self) -> Option<Json> {
        let r = self.res.as_ref()?;
        let Json::Obj(mut pairs) = r.weight_config(self.reward_scale) else {
            return None;
        };
        pairs.push(("kind".to_string(), Json::str("resonant")));
        pairs.push(("total_traversals".to_string(), Json::Int(self.total_traversals.value)));
        pairs.push((
            "total_traversals_resets".to_string(),
            Json::Int(self.total_traversals.resets),
        ));
        Some(Json::Obj(pairs))
    }

    /// Reads a file's accumulators (the rewards are read with every graph's)
    /// and recomputes what follows from the labels.
    pub(crate) fn resonant_read(&mut self, edges: &Json, n_edges: usize) -> Result<(), String> {
        if self.res.is_none() {
            return Ok(());
        }
        let mut arrays = Vec::new();
        for key in ["cx", "cy", "cw"] {
            let values = match edges.get(key) {
                None | Some(Json::Null) => vec![0.0; n_edges],
                Some(v) => {
                    let values = v.to_f64s();
                    if values.len() != n_edges {
                        return Err(format!("edge {key} array has an inconsistent length"));
                    }
                    values
                }
            };
            arrays.push(values);
        }
        let r = self.res.as_mut().expect("checked");
        r.cw = arrays.pop().expect("three arrays");
        r.cy = arrays.pop().expect("three arrays");
        r.cx = arrays.pop().expect("three arrays");
        r.advance = vec![0; self.labels.len()];
        for n in 0..self.labels.len() {
            if self.alive[n] {
                self.resonant_refresh(n);
            }
        }
        Ok(())
    }
}

/// `[(child, edge, score)] -> -log softmax`, with Python's exact sum.
fn softmax_costs(items: &[(usize, usize, f64)]) -> Vec<ChildCost> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut m = f64::NEG_INFINITY;
    for &(_, _, s) in items {
        if s > m {
            m = s;
        }
    }
    let terms: Vec<f64> = items.iter().map(|&(_, _, s)| (s - m).exp()).collect();
    let lse = m + fsum(&terms).ln();
    items
        .iter()
        .map(|&(c, e, s)| ChildCost {
            child: c,
            edge: e,
            cost: lse - s,
            punish: 0.0,
        })
        .collect()
}

/// The punishment traversal over the phase-unrolled graph: `child_costs_at`
/// priced by the punishments alone, cached per `(node, bucket)`.
pub struct PhasePenalty {
    penalty_scale: f64,
    merit_scale: f64,
    cache: std::cell::RefCell<Map<(usize, usize), Vec<ChildCost>>>,
}

impl PhasePenalty {
    /// The two scales; both must be `>= 0`.
    pub fn new(penalty_scale: f64, merit_scale: f64) -> Result<PhasePenalty, String> {
        if penalty_scale < 0.0 {
            return Err(format!("penalty_scale must be >= 0, got {penalty_scale}"));
        }
        if merit_scale < 0.0 {
            return Err(format!("merit_scale must be >= 0, got {merit_scale}"));
        }
        Ok(PhasePenalty {
            penalty_scale,
            merit_scale,
            cache: std::cell::RefCell::new(map()),
        })
    }

    /// `p`'s children at `bucket`, priced by merit and punishment.
    pub fn costs(&self, g: &Graph, p: usize, bucket: usize) -> Vec<ChildCost> {
        let buckets = g.res.as_ref().map(|r| r.buckets).unwrap_or(1);
        let key = (p, bucket % buckets);
        if let Some(found) = self.cache.borrow().get(&key) {
            return found.clone();
        }
        let scored: Vec<(usize, usize, f64)> = g
            .child_evidence_at(p, key.1)
            .into_iter()
            .map(|ev| {
                (
                    ev.child,
                    ev.edge,
                    self.merit_scale * ev.merit - self.penalty_scale * ev.penalty,
                )
            })
            .collect();
        let costs = softmax_costs(&scored);
        self.cache.borrow_mut().insert(key, costs.clone());
        costs
    }
}

/// The cost function a phase search reads the graph through, or `None` for
/// the graph's own (the reward traversal, and the least-punished one, which
/// the phase searches do not rank by).
pub fn phase_traversal_costs(
    traversal: &str,
    penalty_scale: f64,
    merit_scale: f64,
) -> Result<Option<PhasePenalty>, String> {
    match crate::penalty::resolve_traversal(traversal)? {
        crate::penalty::PUNISHMENT => Ok(Some(PhasePenalty::new(penalty_scale, merit_scale)?)),
        _ => Ok(None),
    }
}

// -- the model ------------------------------------------------------------------------------------

/// What a phase model keeps beside its graph: the metacognitive layer and how
/// a corpus declining a cycle teaches the `BACK` edge.
#[derive(Clone, Debug)]
pub struct ResonantModel {
    pub metacog: MetaLayer,
    /// Whether a corpus declining a cycle also teaches the node's `BACK` edge.
    pub teach_back: bool,
    /// How loudly it does - one whole traversal's worth at 1.0.
    pub back_strength: f64,
    /// The share observation may push `BACK` to; above it only experience speaks.
    pub back_ceiling: f64,
}

impl Default for ResonantModel {
    fn default() -> ResonantModel {
        ResonantModel {
            metacog: MetaLayer::default(),
            teach_back: false,
            back_strength: 0.25,
            back_ceiling: 0.10,
        }
    }
}

/// What one walk over a text did.
struct Walk {
    transitions: usize,
    cycles: usize,
    cost: f64,
    edges: Vec<usize>,
}

/// How one run of passes goes.
struct Pass<'a> {
    epochs: usize,
    auto_compress: bool,
    count: bool,
    reward: f64,
    strength: f64,
    phase: Option<&'a str>,
    learn: bool,
    sharpen: f64,
    /// How plain training walks the texts (`../../SPEC-SearchAndTraining.md`);
    /// `None` for the feedback passes, which walk every text in corpus order.
    plan: Option<&'a crate::training::Plan>,
    /// The sentinel every walk begins at: `START` for texts, `THINK` for thoughts.
    origin: usize,
}

impl Model {
    /// An untrained phase model.
    pub fn new_resonant(seed: i64, o: &ResonantOptions, encoding: Encoding) -> Result<Model, String> {
        let g = Graph::new_resonant(seed, o, encoding)?;
        let mut m = Model::from_graph(g);
        m.res = Some(Box::new(ResonantModel::default()));
        m.meta.extra.push(("cycles_seen".to_string(), Json::Int(0)));
        Ok(m)
    }

    /// Whether this is the phase model.
    pub fn is_resonant(&self) -> bool {
        self.res.is_some() && self.g.is_resonant()
    }

    fn resonant(&mut self) -> &mut ResonantModel {
        self.res.as_mut().expect("a resonant model")
    }

    /// The lifetime count of phase-locked cycles the training walks met.
    pub fn cycles_seen(&self) -> i64 {
        self.meta
            .extra
            .iter()
            .find(|(k, _)| k == "cycles_seen")
            .and_then(|(_, v)| v.as_i64())
            .unwrap_or(0)
    }

    fn add_cycles_seen(&mut self, n: i64) {
        let total = self.cycles_seen() + n;
        match self.meta.extra.iter_mut().find(|(k, _)| k == "cycles_seen") {
            Some((_, v)) => *v = Json::Int(total),
            None => self.meta.extra.insert(0, ("cycles_seen".to_string(), Json::Int(total))),
        }
    }

    /// Walks one text through the structure carrying its phase: counts each
    /// traversal at the phase the walk was in (`count`), moves the edges'
    /// reward (`reward`), and teaches the layer what the text did at every
    /// cycle it had the option to close (`learn`).
    fn phase_walk_text(
        &mut self,
        origin: usize,
        text: &str,
        learn: bool,
        count: bool,
        reward: f64,
        strength: f64,
    ) -> Result<Walk, String> {
        let grams = self.g.enc.encode(text);
        if grams.is_empty() {
            return Ok(Walk {
                transitions: 0,
                cycles: 0,
                cost: 0.0,
                edges: Vec::new(),
            });
        }
        let (transitions, path) = match self.g.trace_from(origin, &grams) {
            Some(traced) => traced,
            None => {
                self.g.observe_from(origin, &grams, false)?;
                self.g
                    .trace_from(origin, &grams)
                    .ok_or_else(|| "internal error: observed sequence is not walkable".to_string())?
            }
        };
        let buckets = self.g.res.as_ref().expect("a phase graph").buckets;
        let mut seen: Map<(usize, usize), usize> = map();
        let mut bucket = 0usize;
        let mut cycles = 0usize;
        let mut total_cost = 0.0;
        let mut edges = Vec::with_capacity(transitions.len());
        for (step, t) in transitions.iter().enumerate() {
            let child = path[step + 1];
            let cost = self
                .g
                .child_costs_at(t.p, bucket)
                .iter()
                .find(|c| c.child == child)
                .map(|c| c.cost)
                .unwrap_or_else(unknown_cost);
            total_cost += cost;
            if learn {
                cycles += self.teach_cycle(t.p, bucket, child, &seen, step, strength)?;
            }
            if count {
                self.g.record_phase(t.e, Some(bucket), 1.0);
            }
            edges.push(t.e);
            seen.insert((t.p, bucket), step);
            let advance = self.g.res.as_ref().expect("a phase graph").advance[child];
            bucket = (bucket + advance) % buckets;
        }
        if reward != 0.0 {
            self.g.resonant_add_reward(&edges, reward);
        }
        let n = transitions.len();
        Ok(Walk {
            transitions: n,
            cycles,
            cost: total_cost / n.max(1) as f64,
            edges,
        })
    }

    /// Teaches the layer what the text did where a phase-locked cycle was on
    /// offer: it rode the child that closes the loop, went to END, or took
    /// something else.  Returns 1 when there was a decision, else 0.
    fn teach_cycle(
        &mut self,
        p: usize,
        bucket: usize,
        taken: usize,
        seen: &Map<(usize, usize), usize>,
        step: usize,
        strength: f64,
    ) -> Result<usize, String> {
        let (buckets, advance) = {
            let r = self.g.res.as_ref().expect("a phase graph");
            (r.buckets, r.advance.clone())
        };
        let mut loops: Vec<(usize, i64)> = Vec::new();
        for &c in &self.g.children[p].order {
            if c < FIRST {
                continue; // a sentinel is not a node to loop through
            }
            if let Some(&first) = seen.get(&(c, (bucket + advance[c]) % buckets)) {
                loops.push((c, step as i64 + 1 - first as i64));
            }
        }
        if loops.is_empty() {
            return Ok(0);
        }
        let mut target = loops[0];
        for &l in &loops[1..] {
            if l.1 < target.1 {
                target = l;
            }
        }
        let signature = cycle_signature(self.g.label(target.0), target.1);
        let rides = loops.iter().any(|&(c, _)| c == taken);
        let action = if rides {
            RIDE
        } else if taken == END {
            ABORT
        } else {
            ESCAPE
        };
        self.resonant().metacog.observe(&signature, action, strength)?;
        let (teach_back, back_strength, back_ceiling) = {
            let r = self.resonant();
            (r.teach_back, r.back_strength, r.back_ceiling)
        };
        if teach_back && p >= FIRST && strength > 0.0 {
            let amount = strength * back_strength;
            if action == RIDE {
                self.g.observe_onward(p, amount);
            } else if self.g.back_probability(p) < back_ceiling {
                let instead = if taken == END { None } else { Some(taken) };
                self.g.resonant_observe_back(p, Some(target.0), instead, amount)?;
            }
        }
        Ok(1)
    }

    /// The routine behind train, reward and punish: `epochs` of walks over
    /// the texts, the structure settled (and compressed) first so every epoch
    /// walks the same transitions.  A pass with a plan (plain training) walks
    /// the texts the way it says - the order, the curriculum, the rehearsal of
    /// the replay buffer and the early stop.
    fn phase_passes(
        &mut self,
        texts: &[String],
        o: &Pass<'_>,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let enc = self.g.enc;
        let cleaned: Vec<&String> = texts.iter().filter(|t| enc.len(t) >= enc.n).collect();
        let skipped = texts.len() - cleaned.len();
        if let Some(plan) = o.plan {
            plan.check()?;
        }
        // a model that keeps a replay buffer offers every run's texts to it
        let mut plan = match o.plan {
            Some(p) if p.active() || self.replay.is_some() => {
                Some(self.training_plan(cleaned.iter().map(|t| (*t).clone()).collect(), o.epochs, p)?)
            }
            _ => None,
        };
        let rehearsed = plan.as_ref().map(|p| p.replayed()).unwrap_or_default();
        let base = self.meta.epochs_total.value; // `train` counts the epochs once the run is over
        if !cleaned.is_empty() {
            let grams: Vec<Vec<String>> = cleaned
                .iter()
                .map(|t| enc.encode(t))
                .chain(rehearsed.iter().map(|t| enc.encode(t)))
                .collect();
            self.observe_all(o.origin, &grams)?;
            if o.auto_compress {
                self.g.compress();
            }
            self.observe_all(o.origin, &grams)?;
        }
        let mut records = Vec::with_capacity(o.epochs);
        for epoch in 1..=o.epochs {
            let started = Instant::now();
            let (mut transitions, mut cycles) = (0usize, 0usize);
            let mut cost = 0.0;
            let mut touched: Vec<usize> = Vec::new();
            let mut in_touched: std::collections::HashSet<usize> = std::collections::HashSet::new();
            // this epoch's texts: the order, the curriculum and the rehearsal
            let planned_texts = match plan.as_ref() {
                Some(p) if !p.plain() => Some(p.epoch_texts(epoch - 1, base + epoch as i64)),
                _ => None,
            };
            let walked: Vec<&String> = match &planned_texts {
                Some(list) => list.iter().collect(),
                None => cleaned.clone(),
            };
            for text in &walked {
                let done = self.phase_walk_text(o.origin, text, o.learn, o.count, o.reward, o.strength)?;
                transitions += done.transitions;
                cycles += done.cycles;
                cost += done.cost * done.transitions as f64;
                for e in done.edges {
                    if in_touched.insert(e) {
                        touched.push(e);
                    }
                }
            }
            if o.sharpen != 1.0 && !touched.is_empty() {
                self.g.sharpen(&touched, o.sharpen)?;
            }
            if o.count || o.reward != 0.0 || o.sharpen != 1.0 {
                self.g.resonant_recompute();
            }
            // the dynamic window's step; this kind compresses once before its passes, so the step merges as well
            let stepped = self.window_epoch(o.auto_compress);
            self.g.carry_counters(false); // the epoch is over: wrap whatever reached the limit
            let loss = cost / transitions.max(1) as f64;
            let signatures = self.res.as_ref().map(|r| r.metacog.len()).unwrap_or(0);
            let stopping = plan.as_mut().is_some_and(|p| p.stop(epoch - 1, loss));
            let mut extra = vec![
                (
                    "traversed".to_string(),
                    Json::Int(if o.count { transitions as i64 } else { 0 }),
                ),
                ("cycles".to_string(), Json::Int(cycles as i64)),
                ("signatures".to_string(), Json::Int(signatures as i64)),
            ];
            extra.extend(crate::model::early_stop_extra(stopping));
            let record = EpochRecord {
                epoch: epoch as i64,
                loss,
                perplexity: crate::radix::perplexity(loss),
                nodes: self.g.num_nodes(),
                edges: self.g.num_edges(),
                trigrams: self.g.num_trigrams(),
                compression_ratio: self.g.compression_ratio(),
                merges: stepped.as_ref().map_or(0, |s| s.merges),
                splits: stepped.as_ref().map(|s| s.splits),
                window: stepped.as_ref().map(|s| s.from),
                transitions: transitions as i64,
                seconds: started.elapsed().as_secs_f64(),
                skipped_short: skipped,
                traversed: o.count,
                reward: o.reward * transitions as f64,
                phase: o.phase.map(str::to_string),
                extra,
            };
            crate::log_debug!(
                LOG,
                "epoch {epoch} {}: loss {:.4}, {cycles} cycle(s), {signatures} signature(s)",
                o.phase.unwrap_or("train"),
                loss
            );
            self.history.push(record.clone());
            self.epoch_done(&record)?;
            let go_on = on_epoch(&record);
            records.push(record);
            if stopping || !go_on {
                break;
            }
        }
        let seen: i64 = records
            .iter()
            .map(|r| r.extra_value("cycles").and_then(|v| v.as_i64()).unwrap_or(0))
            .sum();
        self.add_cycles_seen(seen);
        if let Some(p) = plan.as_mut() {
            self.replay = p.finish();
        }
        Ok(records)
    }

    /// Registers encoded texts structurally, uncounted (Python's `_observe`).
    fn observe_all(&mut self, origin: usize, grams: &[Vec<String>]) -> Result<(), String> {
        let before = self.g.structure_version;
        for gram in grams {
            self.g.observe_from(origin, gram, false)?;
        }
        if self.g.structure_version != before {
            for gram in grams {
                self.g.observe_from(origin, gram, false)?;
            }
        }
        Ok(())
    }

    /// Learns from texts: every traversal counted at its phase, and the cycle
    /// decisions beside it.  There is no learning rate here; `epochs`,
    /// `auto_compress` and the `plan` (the order, the curriculum, the
    /// rehearsal, the early stop) are what a training config means to this
    /// model.  A pass stamped with a `phase` is feedback: it walks every text
    /// in corpus order and leaves the replay buffer alone.
    pub fn resonant_train(
        &mut self,
        texts: &[String],
        epochs: usize,
        auto_compress: bool,
        phase: Option<&str>,
        plan: &crate::training::Plan,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        self.resonant_train_from(START, texts, epochs, auto_compress, phase, plan, on_epoch)
    }

    /// [`Model::resonant_train`] from either origin sentinel: `THINK` trains
    /// the texts as thoughts ([`crate::thinking`]).
    #[allow(clippy::too_many_arguments)]
    pub fn resonant_train_from(
        &mut self,
        origin: usize,
        texts: &[String],
        epochs: usize,
        auto_compress: bool,
        phase: Option<&str>,
        plan: &crate::training::Plan,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        if !crate::graph::is_origin(origin) {
            return Err(format!("a text begins at START or THINK, not at node {origin}"));
        }
        let enc = self.g.enc;
        let read = crate::training::read(&enc, texts, plan);
        let texts: &[String] = &read;
        let cleaned: Vec<&String> = texts.iter().filter(|t| enc.len(t) >= enc.n).collect();
        let pass = Pass {
            epochs,
            auto_compress,
            count: true,
            reward: 0.0,
            strength: 1.0,
            phase,
            learn: true,
            sharpen: 1.0,
            plan: phase.is_none().then_some(plan),
            origin,
        };
        let records = self.phase_passes(texts, &pass, on_epoch)?;
        self.meta.epochs_total.add(records.len() as i64);
        // the encoding's units, as every other kind counts them: words under a word encoding
        self.meta
            .trained_chars
            .add(cleaned.iter().map(|t| enc.len(t) as i64).sum());
        self.meta.trained_texts.add(cleaned.len() as i64);
        crate::log_info!(
            LOG,
            "{} epoch(s) of the phase model over {} text(s): loss {:.4} -> {:.4}",
            records.len(),
            cleaned.len(),
            records.first().map(|r| r.loss).unwrap_or(0.0),
            records.last().map(|r| r.loss).unwrap_or(0.0)
        );
        Ok(records)
    }

    /// `phase_passes` over all the texts at once, or once per group of equally
    /// weighted texts - the reward, the strength and the sharpening scaled by
    /// the weight.  `sharpen_rate` is how far one unit of strength moves the
    /// phase lock: `+0.5` tightens it, `-0.5` scrambles it.
    #[allow(clippy::too_many_arguments)]
    fn feedback_passes(
        &mut self,
        texts: &[String],
        epochs: usize,
        count: bool,
        reward: f64,
        strength: f64,
        phase: &str,
        sharpen_rate: f64,
        weights: Option<&[f64]>,
        name: &str,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let sharpen = |x: f64| if x > 0.0 { x } else { 0.0 };
        let Some(weights) = weights else {
            let pass = Pass {
                epochs,
                auto_compress: true,
                count,
                reward,
                strength,
                phase: Some(phase),
                learn: true,
                sharpen: sharpen(1.0 + sharpen_rate * strength),
                plan: None,
                origin: START,
            };
            return self.phase_passes(texts, &pass, on_epoch);
        };
        let mut records = Vec::new();
        let mut stopped = false;
        for (weight, group) in crate::radix::weight_groups(texts, weights, name)? {
            if stopped {
                break;
            }
            let pass = Pass {
                epochs,
                auto_compress: true,
                count,
                reward: reward * weight,
                strength: strength * weight,
                phase: Some(phase),
                learn: true,
                sharpen: sharpen(1.0 + sharpen_rate * strength * weight),
                plan: None,
                origin: START,
            };
            let group_records = self.phase_passes(&group, &pass, &mut |_| true)?;
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

    /// Thumbs up: counts the texts' paths, rewards their edges by `strength`
    /// and locks their phases tighter.
    pub fn resonant_reward(
        &mut self,
        texts: &[String],
        epochs: usize,
        strength: f64,
        weights: Option<&[f64]>,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let base = strength.abs();
        let records = self.feedback_passes(
            texts, epochs, true, base, base, "positive", 0.5, weights, "weights", on_epoch,
        )?;
        // `sum(r["reward"] for r in records)`, left to right
        self.meta.rewards_total += records.iter().fold(0.0, |acc, r| acc + r.reward);
        self.meta.feedback_passes.add(1);
        Ok(records)
    }

    /// Thumbs down: penalises the texts' edges and *decoheres* them - the
    /// phase lock that made them right is scrambled.  Nothing is counted.
    pub fn resonant_punish(
        &mut self,
        texts: &[String],
        epochs: usize,
        strength: f64,
        weights: Option<&[f64]>,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<Vec<EpochRecord>, String> {
        let base = strength.abs();
        let records = self.feedback_passes(
            texts, epochs, false, -base, base, "negative", -0.5, weights, "weights", on_epoch,
        )?;
        self.meta.penalties_total += records.iter().fold(0.0, |acc, r| acc + -r.reward);
        self.meta.feedback_passes.add(1);
        Ok(records)
    }

    /// 2NRL: learn the garbage, phases and all; invert (every mean phase
    /// rotated by pi, every reward negated, the layer flipped); then relock the
    /// phases on the good texts with a reward of `strength`.
    #[allow(clippy::too_many_arguments)]
    pub fn resonant_two_nrl(
        &mut self,
        bad: &[String],
        good: &[String],
        neg_epochs: usize,
        pos_epochs: usize,
        strength: f64,
        bad_weights: Option<&[f64]>,
        good_weights: Option<&[f64]>,
        on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
    ) -> Result<(Vec<EpochRecord>, Vec<EpochRecord>), String> {
        let base = strength.abs();
        let stopped = std::cell::Cell::new(false);
        let mut watch = |r: &EpochRecord| {
            let go_on = on_epoch(r);
            stopped.set(stopped.get() || !go_on);
            go_on
        };
        let negative = self.feedback_passes(
            bad,
            neg_epochs,
            true,
            0.0,
            base,
            "negative",
            0.0,
            bad_weights,
            "bad_weights",
            &mut watch,
        )?;
        self.resonant_invert();
        let mut positive = Vec::new();
        if !stopped.get() {
            positive = self.feedback_passes(
                good,
                pos_epochs,
                true,
                base,
                base,
                "positive",
                0.0,
                good_weights,
                "good_weights",
                &mut watch,
            )?;
        }
        self.meta.twonrl_runs.add(1);
        Ok((negative, positive))
    }

    /// Rotates every edge's phase lock into antiphase and flips the layer.
    pub fn resonant_invert(&mut self) {
        self.g.resonant_invert();
        if let Some(r) = self.res.as_mut() {
            r.metacog.invert();
        }
    }

    /// Failures made unlikely locally: `activation` rotates each text's edges
    /// by `amount` half-circles (1 puts them in antiphase), `state` decoheres
    /// them (1 erases the lock).  Returns `{"texts", "flipped", "unit":
    /// "edges", "mode", "amount_mean"}`.
    pub fn resonant_invert_paths(
        &mut self,
        texts: &[String],
        mode: &str,
        amounts: Option<&[f64]>,
    ) -> Result<Json, String> {
        let mode = mode.to_ascii_lowercase();
        if mode != "activation" && mode != "state" {
            return Err(format!(
                "unknown mode {}; expected 'activation' or 'state'",
                crate::negative::python_repr(&mode)
            ));
        }
        let enc = self.g.enc;
        let cleaned: Vec<String> = texts.iter().filter(|t| enc.len(t) >= enc.n).cloned().collect();
        let values: Vec<f64> = match amounts {
            None => vec![1.0; cleaned.len()],
            Some(values) if values.len() != cleaned.len() => {
                return Err(format!(
                    "amounts has {} entries for {} texts",
                    values.len(),
                    cleaned.len()
                ))
            }
            Some(values) => values.to_vec(),
        };
        if let Some(bad) = values.iter().find(|v| !(0.0..=1.0).contains(*v)) {
            return Err(format!(
                "amounts must lie in [0, 1], got {}",
                crate::json::py_repr(*bad)
            ));
        }
        let mut flipped = 0;
        let mut used: Vec<f64> = Vec::new();
        for (text, &amount) in cleaned.iter().zip(&values) {
            if amount <= 0.0 {
                continue;
            }
            let done = self.phase_walk_text(START, text, false, false, 0.0, 0.0)?;
            if done.edges.is_empty() {
                continue;
            }
            flipped += if mode == "activation" {
                self.g.rotate(&done.edges, amount)
            } else {
                self.g
                    .sharpen(&done.edges, if 1.0 - amount > 0.0 { 1.0 - amount } else { 0.0 })?
            };
            used.push(amount);
        }
        if flipped > 0 {
            self.g.resonant_recompute();
        }
        Ok(Json::obj([
            ("texts", Json::Int(cleaned.len() as i64)),
            ("flipped", Json::Int(flipped as i64)),
            ("unit", Json::str("edges")),
            ("mode", Json::str(mode)),
            (
                "amount_mean",
                Json::Num(if used.is_empty() {
                    0.0
                } else {
                    fsum(&used) / used.len() as f64
                }),
            ),
        ]))
    }

    /// Continues `prefix` over the phase-unrolled graph: `kbest` (the exact
    /// `k` cheapest walks, with the layer on the cycles they meet), `dijkstra`
    /// (the single cheapest, one label per state, no layer), `beam` (the only
    /// mode that also returns the `k` least likely) or `sample` (one walk,
    /// layer included; `rng` `None` uses the graph's own).
    pub fn resonant_predict(
        &mut self,
        prefix: &str,
        o: &PredictOptions,
        rng: Option<&mut Mt19937>,
    ) -> Result<crate::beam::Prediction, String> {
        use crate::phasesearch::{phase_beam, phase_dijkstra, phase_kbest, phase_walk};
        let mode = if o.mode.is_empty() {
            "kbest".to_string()
        } else {
            o.mode.to_ascii_lowercase()
        };
        if !matches!(mode.as_str(), "kbest" | "beam" | "dijkstra" | "sample") {
            return Err(format!(
                "unknown mode {}; expected 'kbest', 'beam', 'dijkstra' or 'sample'",
                crate::negative::python_repr(&mode)
            ));
        }
        let traversal = crate::penalty::resolve_traversal(&o.traversal)?;
        let costs = phase_traversal_costs(traversal, o.penalty_scale, o.merit_scale)?;
        let (node, offset, lead) = self.walk_start(prefix, o.origin)?;
        let bucket = if prefix.is_empty() {
            0
        } else {
            self.g.text_bucket(prefix)
        };
        let lead_len = self.g.enc.len(&lead); // in units, as `length` is
        let want = o.length.saturating_sub(lead_len);
        let (cap, max_chars) = if mode == "sample" {
            let cap = o.max_length.unwrap_or(o.length);
            (Some(cap), Some(cap.saturating_sub(lead_len)))
        } else {
            match o.max_length {
                None if o.length == 0 => (Some(0), Some(0)),
                None => (None, None),
                Some(max_length) => {
                    let cap = o.length.max(max_length);
                    (Some(cap), Some(want.max(cap.saturating_sub(lead_len))))
                }
            }
        };
        self.g.prepare();
        let meta = self.res.as_ref().map(|r| r.metacog.clone());
        let g: &Graph = &self.g;
        let mut width = 0;
        let (mut top, mut bottom, expanded, mut best): (Vec<PathResult>, Vec<PathResult>, usize, Option<PathResult>);
        match mode.as_str() {
            "kbest" => {
                let (found, n) = phase_kbest(
                    g,
                    meta.as_ref(),
                    node,
                    offset,
                    bucket,
                    want,
                    o.k,
                    max_chars,
                    o.step_penalty,
                    o.to_end,
                    costs.as_ref(),
                )?;
                best = found.first().cloned();
                top = found;
                bottom = Vec::new();
                expanded = n;
            }
            "dijkstra" => {
                let walk = phase_dijkstra(
                    g,
                    node,
                    offset,
                    bucket,
                    want,
                    max_chars,
                    o.step_penalty,
                    o.to_end,
                    costs.as_ref(),
                )?;
                expanded = walk.expanded;
                best = Some(walk.clone());
                top = vec![walk];
                bottom = Vec::new();
            }
            "beam" => {
                let (t, b, n) = phase_beam(
                    g,
                    meta.as_ref(),
                    node,
                    offset,
                    bucket,
                    want,
                    o.k,
                    o.beam,
                    max_chars,
                    o.step_penalty,
                    o.to_end,
                    costs.as_ref(),
                    o.diversity,
                )?;
                width = if o.beam == 0 {
                    crate::beam::default_beam(o.k)
                } else {
                    o.beam
                };
                best = t.first().cloned();
                top = t;
                bottom = b;
                expanded = n;
            }
            _ => {
                let walk = match rng {
                    Some(rng) => phase_walk(
                        g,
                        meta.as_ref(),
                        node,
                        offset,
                        bucket,
                        max_chars,
                        o.temperature,
                        rng,
                        costs.as_ref(),
                        o.tuning().filter,
                    )?,
                    None => {
                        // the graph's own generator, lent to the walk and put back
                        let mut own = std::mem::replace(&mut self.g.rng, Mt19937::new(0));
                        let walk = phase_walk(
                            &self.g,
                            meta.as_ref(),
                            node,
                            offset,
                            bucket,
                            max_chars,
                            o.temperature,
                            &mut own,
                            costs.as_ref(),
                            o.tuning().filter,
                        );
                        self.g.rng = own;
                        walk?
                    }
                };
                expanded = walk.expanded;
                best = Some(walk.clone());
                top = vec![walk];
                bottom = Vec::new();
            }
        }
        let enc = self.g.enc;
        let fix = |r: &mut PathResult| {
            if !lead.is_empty() {
                let joined = enc.join(&[&lead, &r.text]);
                r.text = match cap {
                    None => joined,
                    Some(cap) => enc.truncate(&joined, cap),
                };
            }
            r.full_text = enc.join(&[prefix, &r.text]);
        };
        top.iter_mut().for_each(fix);
        bottom.iter_mut().for_each(fix);
        // `best` is the first of `top` (or the walk itself), fixed with it
        best = match best {
            Some(_) if !top.is_empty() => Some(top[0].clone()),
            other => other,
        };
        let best = match best {
            Some(best) => best,
            None => {
                let text = match cap {
                    None => lead.clone(),
                    Some(cap) => enc.truncate(&lead, cap),
                };
                let full_text = enc.join(&[prefix, &text]);
                PathResult {
                    text,
                    labels: vec![self.g.label(node).to_string()],
                    node_ids: vec![node],
                    full_text,
                    ..Default::default()
                }
            }
        };
        Ok(crate::beam::Prediction {
            best,
            top,
            bottom,
            k: o.k,
            beam: width,
            mode,
            traversal: traversal.to_string(),
            expanded,
        })
    }

    /// Whole texts from the phase search; `kbest` returns the exact `count`
    /// cheapest, the other modes are the shared generation's over this
    /// model's own search.
    pub fn resonant_generate(&mut self, o: &GenerateOptions) -> Result<Vec<PathResult>, String> {
        let mode = if o.mode.is_empty() {
            "sample".to_string()
        } else {
            o.mode.to_ascii_lowercase()
        };
        let whole = |r: PathResult, enc: Encoding| -> PathResult {
            let full = enc.join(&[&o.prefix, &r.text]);
            PathResult {
                text: full.clone(),
                full_text: full,
                ..r
            }
        };
        if o.count == 0 && mode != "kbest" && !matches!(mode.as_str(), "beam" | "dijkstra" | "sample") {
            return Err(format!(
                "unknown mode {}; expected 'beam', 'dijkstra' or 'sample'",
                crate::negative::python_repr(&mode)
            ));
        }
        let enc = self.g.enc;
        let search = |length: usize, mode: &str, k: usize, to_end: bool| PredictOptions {
            length,
            mode: mode.to_string(),
            k,
            beam: o.beam,
            step_penalty: o.step_penalty,
            temperature: o.temperature,
            to_end,
            max_length: Some(o.max_length),
            traversal: o.traversal.clone(),
            penalty_scale: o.penalty_scale,
            merit_scale: o.merit_scale,
            top_k: o.top_k,
            top_p: o.top_p,
            min_p: o.min_p,
            diversity: o.diversity,
            origin: START,
        };
        match mode.as_str() {
            "kbest" => {
                if o.count == 0 {
                    return Ok(Vec::new());
                }
                let found = self.resonant_predict(&o.prefix, &search(0, "kbest", o.count, true), None)?;
                Ok(found.top.into_iter().map(|r| whole(r, enc)).collect())
            }
            "sample" => {
                if o.count == 0 {
                    return Ok(Vec::new());
                }
                let mut rng = o.seed.map(Mt19937::new);
                let mut out = Vec::with_capacity(o.count);
                for _ in 0..o.count {
                    let mut opts = search(o.max_length, "sample", 1, false);
                    opts.step_penalty = 0.0;
                    let found = self.resonant_predict(&o.prefix, &opts, rng.as_mut())?;
                    out.push(whole(found.best, enc));
                }
                Ok(out)
            }
            "dijkstra" => {
                if o.count == 0 {
                    return Ok(Vec::new());
                }
                let found = self.resonant_predict(&o.prefix, &search(0, "dijkstra", 5, true), None)?;
                Ok(vec![whole(found.best, enc)])
            }
            "beam" => {
                if o.count == 0 {
                    return Ok(Vec::new());
                }
                let found = self.resonant_predict(&o.prefix, &search(0, "beam", o.count, true), None)?;
                Ok(found.top.into_iter().map(|r| whole(r, enc)).collect())
            }
            other => Err(format!(
                "unknown mode {}; expected 'beam', 'dijkstra' or 'sample'",
                crate::negative::python_repr(other)
            )),
        }
    }

    /// The log-probability of a text under the model, phase included: every
    /// transition contributes `log P(child | parent, phase)`, and a step inside
    /// a compressed node moves the phase on without being a transition.
    /// `chars` is the text's length in the encoding's units, as every other
    /// kind counts it: a six-word text under a word encoding is 6, not 22.
    pub fn resonant_score(&mut self, text: &str) -> Score {
        let enc = self.g.enc;
        let grams = enc.encode(text);
        let chars = enc.len(text);
        if grams.is_empty() {
            return Score {
                chars,
                ..Default::default()
            };
        }
        let log_unknown = (1e-6f64).ln();
        let buckets = self.g.res.as_ref().map(|r| r.buckets).unwrap_or(1);
        let (mut log_prob, mut transitions, mut unknown) = (0.0, 0usize, 0usize);
        let (mut node, mut offset, mut bucket) = (START, 0usize, 0usize);
        let mut lost = false;
        let log_p = |g: &Graph, p: usize, offset: usize, c: usize, bucket: usize| -> Option<f64> {
            if p != START && offset + g.enc.n != g.label_len(p) {
                return None;
            }
            g.child_costs_at(p, bucket)
                .iter()
                .find(|cc| cc.child == c)
                .map(|cc| -cc.cost)
        };
        for t in &grams {
            let advance = self.g.res.as_ref().map(|r| r.gram_advance(t)).unwrap_or(0);
            match self.g.lookup(t) {
                None => {
                    transitions += 1;
                    unknown += 1;
                    log_prob += log_unknown;
                    lost = true;
                }
                Some(loc) => {
                    if !lost && loc.node == node && loc.off == offset + 1 {
                        // a deterministic step inside a node: no edge, no transition
                        offset = loc.off;
                        bucket = (bucket + advance) % buckets;
                        continue;
                    }
                    transitions += 1;
                    let lp = if lost || loc.off != 0 {
                        None
                    } else {
                        log_p(&self.g, node, offset, loc.node, bucket)
                    };
                    match lp {
                        Some(lp) => log_prob += lp,
                        None => {
                            unknown += 1;
                            log_prob += log_unknown;
                        }
                    }
                    node = loc.node;
                    offset = loc.off;
                    lost = false;
                }
            }
            bucket = (bucket + advance) % buckets;
        }
        transitions += 1;
        let lp = if lost {
            None
        } else {
            log_p(&self.g, node, offset, END, bucket)
        };
        match lp {
            Some(lp) => log_prob += lp,
            None => {
                unknown += 1;
                log_prob += log_unknown;
            }
        }
        Score {
            log_prob,
            per_char: log_prob / chars.max(1) as f64,
            chars,
            transitions,
            unknown_transitions: unknown,
        }
    }

    /// How phase-locked the model is overall: `(mean, max, edges)` of the
    /// alive edges' coherence.
    pub fn coherence(&self) -> (f64, f64, usize) {
        let Some(r) = self.g.res.as_ref() else {
            return (0.0, 0.0, 0);
        };
        let values: Vec<f64> = (0..r.cw.len())
            .filter(|&e| self.g.is_edge_alive(e))
            .map(|e| r.coherence(e))
            .collect();
        if values.is_empty() {
            return (0.0, 0.0, 0);
        }
        let mut max = values[0];
        for &v in &values[1..] {
            if v > max {
                max = v;
            }
        }
        (fsum(&values) / values.len() as f64, max, values.len())
    }
}

// -- what it says about itself ----------------------------------------------------------------------

/// The phase model's statistics, with the keys Python's `ResonantNet.stats` reports.
pub fn stats_json(model: &Model) -> Json {
    let g = &model.g;
    let enc = g.enc;
    let meta = &model.meta;
    let (mean, max, _) = model.coherence();
    let (mut rewards, mut penalties) = (0.0, 0.0);
    for e in 0..g.num_edge_ids() {
        if !g.is_edge_alive(e) {
            continue;
        }
        let value = g.edge_reward(e);
        if value > 0.0 {
            rewards += value;
        } else if value < 0.0 {
            penalties -= value;
        }
    }
    let r = model.res.as_ref();
    let weights = g
        .res
        .as_ref()
        .map(|d| d.weight_config(g.reward_scale))
        .unwrap_or(Json::Null);
    Json::Obj(vec![
        ("kind".to_string(), Json::str("resonant")),
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
        ("dynamic_window".to_string(), g.dynamic_window.size_json()),
        ("inverted".to_string(), Json::Bool(g.inverted)),
        ("backend".to_string(), Json::str(crate::backend::NAME)),
        ("device".to_string(), Json::str(crate::backend::DEVICE)),
        ("epochs_total".to_string(), Json::Int(meta.epochs_total.value)),
        ("trained_chars".to_string(), Json::Int(meta.trained_chars.value)),
        ("trained_texts".to_string(), Json::Int(meta.trained_texts.value)),
        ("twonrl_runs".to_string(), Json::Int(meta.twonrl_runs.value)),
        ("history_len".to_string(), Json::Int(model.history.len() as i64)),
        (
            "last_loss".to_string(),
            model.history.last().map(|r| Json::Num(r.loss)).unwrap_or(Json::Null),
        ),
        ("total_traversals".to_string(), Json::Int(g.total_traversals().value)),
        (
            "total_traversals_resets".to_string(),
            Json::Int(g.total_traversals().resets),
        ),
        ("rewards_total".to_string(), Json::Num(rewards)),
        ("penalties_total".to_string(), Json::Num(penalties)),
        ("feedback_passes".to_string(), Json::Int(meta.feedback_passes.value)),
        (
            "buckets".to_string(),
            Json::Int(g.res.as_ref().map(|d| d.buckets as i64).unwrap_or(0)),
        ),
        ("coherence_mean".to_string(), Json::Num(mean)),
        ("coherence_max".to_string(), Json::Num(max)),
        ("cycles_seen".to_string(), Json::Int(model.cycles_seen())),
        ("meta".to_string(), r.map(|r| r.metacog.stats()).unwrap_or(Json::Null)),
        ("teach_back".to_string(), Json::Bool(r.is_some_and(|r| r.teach_back))),
        (
            "back_strength".to_string(),
            Json::Num(r.map(|r| r.back_strength).unwrap_or(0.25)),
        ),
        (
            "back_ceiling".to_string(),
            Json::Num(r.map(|r| r.back_ceiling).unwrap_or(0.10)),
        ),
        ("weights".to_string(), weights),
    ])
}

// -- the file -------------------------------------------------------------------------------------

/// The `meta` block of a phase file: the lifetime counters, the feedback
/// totals, the cycles the walks met, then whatever else it carried.
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
    pairs.push(("rewards_total".to_string(), Json::Num(meta.rewards_total)));
    pairs.push(("penalties_total".to_string(), Json::Num(meta.penalties_total)));
    pairs.push(("feedback_passes".to_string(), Json::Int(meta.feedback_passes.value)));
    pairs.extend(meta.extra.iter().cloned());
    Json::Obj(pairs)
}

/// The model as the `radixnet-resonant` document Python writes.
pub(crate) fn model_doc(model: &mut Model) -> Json {
    let r = model.res.clone().unwrap_or_default();
    Json::obj([
        ("format", Json::str(RESONANT_FORMAT)),
        ("version", Json::Int(crate::file::MODEL_FORMAT_VERSION)),
        ("saved_at", Json::str(crate::clock::utc_now())),
        ("meta", meta_json(model)),
        (
            "history",
            Json::Arr(model.history.iter().map(EpochRecord::to_json).collect()),
        ),
        ("backend", Json::str(crate::backend::NAME)),
        ("metacog", r.metacog.to_json()),
        (
            "cycles",
            Json::obj([
                ("teach_back", Json::Bool(r.teach_back)),
                ("back_strength", Json::Num(r.back_strength)),
                ("back_ceiling", Json::Num(r.back_ceiling)),
            ]),
        ),
        ("graph", model.g.to_doc()),
    ])
}

/// The score function a graph document's `weights` block holds.
pub(crate) fn options_of(weights: &Json) -> ResonantOptions {
    let num = |key: &str, fallback: f64| weights.at(key).as_f64().unwrap_or(fallback);
    let buckets = weights.at("buckets").as_i64().unwrap_or(8).max(1) as usize;
    let period = num("period", 0.0);
    ResonantOptions {
        buckets,
        period: if period != 0.0 { period } else { buckets as f64 },
        kick_scale: num("kick_scale", 0.0),
        resonance_scale: num("resonance_scale", 1.0),
        amp_scale: num("amp_scale", 1.0),
        reward_scale: num("reward_scale", 1.0),
        concentration: num("concentration", 2.0),
    }
}

/// The layer, the cycle settings and the `cycles_seen` counter of a phase
/// document, onto a model whose graph has been read.
pub(crate) fn read_model(model: &mut Model, doc: &Json) {
    let cycles = doc.at("cycles");
    model.res = Some(Box::new(ResonantModel {
        metacog: MetaLayer::from_json(doc.at("metacog")),
        teach_back: cycles.at("teach_back").as_bool().unwrap_or(false),
        back_strength: cycles.at("back_strength").as_f64().unwrap_or(0.25),
        back_ceiling: cycles.at("back_ceiling").as_f64().unwrap_or(0.10),
    }));
    // `{**_new_meta(seed), **file_meta}`: cycles_seen sits where a new model has it
    if !model.meta.extra.iter().any(|(k, _)| k == "cycles_seen") {
        model.meta.extra.insert(0, ("cycles_seen".to_string(), Json::Int(0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts() -> Vec<String> {
        [
            "the cat sat on the mat",
            "the cat sat on the log",
            "lol lol lol lol",
            "a bird flew over the hill",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn trained() -> Model {
        let mut m = Model::new_resonant(1, &ResonantOptions::default(), Encoding::default()).unwrap();
        m.resonant_train(&texts(), 2, true, None, &crate::training::Plan::default(), &mut |_| {
            true
        })
        .unwrap();
        m
    }

    #[test]
    fn a_grams_phase_is_pythons() {
        // int.from_bytes(blake2b(b'the', digest_size=8).digest(), 'big') / 2.0**64 * TAU
        assert_eq!(trigram_phase("the"), 2.328071792738204);
        assert_eq!(py_mod(-1.0, TAU), TAU - 1.0);
        assert_eq!(py_mod(0.0, TAU), 0.0);
    }

    #[test]
    fn hypot_is_pythons_not_the_c_librarys() {
        // pairs where math.hypot and the C library's hypot disagree in the last bit
        let pairs = [
            (331.28837608635735, 332.62496671525935, 469.4585781645311),
            (-30.334959793691198, 268.96627659536705, 270.6715125964688),
            (119.84432945256415, 291.81923567881, 315.46969682379887),
            (-396.882204475601, 2009.1350949153239, 2047.9597930256884),
            (45.090152119839516, 39.4026857467171, 59.88066016874573),
            (-1187.045222022207, 731.4829212991157, 1394.3255083652593),
            (0.18116637863406668, -0.012799013636209665, 0.18161792724684941),
            (-3195.820583648111, 1197.867146312045, 3412.9393348084454),
            (-45.107096824469394, -728.2883473596612, 729.6838816116185),
            (267.18241526110944, -432.9869752559312, 508.7869532191645),
            (-0.7148846893299265, 3.984199131432003, 4.047826927864119),
            (-26.85589555934913, -3.262408467811369, 27.053325771622156),
        ];
        for (x, y, h) in pairs {
            assert_eq!(py_hypot(x, y), h, "hypot({x}, {y})");
        }
        assert_eq!(py_hypot(-3.0, 4.0), 5.0);
        assert_eq!(py_hypot(0.0, 0.0), 0.0);
        assert!(py_hypot(f64::NAN, 1.0).is_nan());
        assert_eq!(py_hypot(f64::NAN, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn training_locks_phases_and_meets_cycles() {
        let m = trained();
        assert_eq!(m.kind(), "resonant");
        let (mean, max, edges) = m.coherence();
        assert!(edges > 0 && mean > 0.0 && max <= 1.0);
        assert!(m.cycles_seen() > 0, "'lol lol lol' goes round");
        assert!(!m.res.as_ref().unwrap().metacog.is_empty());
        m.g.check_invariants(&texts(), false).unwrap();
    }

    #[test]
    fn the_phase_survives_compression() {
        let m = trained();
        let r = m.g.res.as_ref().unwrap();
        for n in FIRST..m.g.num_node_ids() {
            if m.g.is_alive(n) {
                assert_eq!(r.advance[n], m.g.label_advance(m.g.label(n)), "node {n}");
            }
        }
    }

    #[test]
    fn inverting_rotates_every_lock_by_pi() {
        let mut m = trained();
        let e = (0..m.g.num_edge_ids())
            .find(|&e| m.g.is_edge_alive(e) && m.g.res.as_ref().unwrap().coherence(e) > 0.0)
            .unwrap();
        let before = m.g.res.as_ref().unwrap().mu(e);
        m.resonant_invert();
        let after = m.g.res.as_ref().unwrap().mu(e);
        let turned = py_mod(after - before, TAU);
        assert!((turned - PI).abs() < 1e-9, "{turned}");
    }

    #[test]
    fn it_predicts_in_every_mode() {
        let mut m = trained();
        for mode in ["kbest", "dijkstra", "beam", "sample"] {
            let found = m
                .resonant_predict(
                    "the cat",
                    &PredictOptions {
                        mode: mode.to_string(),
                        length: 8,
                        k: 2,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            assert!(
                found.best.full_text.starts_with("the cat"),
                "{mode}: {}",
                found.best.full_text
            );
        }
        let s = m.resonant_score("the cat sat on the mat");
        assert_eq!(s.unknown_transitions, 0);
        assert!(s.log_prob < 0.0);
        assert!(m.resonant_score("zzqq never seen").unknown_transitions > 0);
    }

    #[test]
    fn a_reward_raises_the_score_and_a_penalty_lowers_it() {
        let mut m = trained();
        let text = "the cat sat on the log".to_string();
        let before = m.resonant_score(&text).log_prob;
        m.resonant_reward(std::slice::from_ref(&text), 1, 1.0, None, &mut |_| true)
            .unwrap();
        let rewarded = m.resonant_score(&text).log_prob;
        assert!(rewarded > before, "{rewarded} <= {before}");
        m.resonant_punish(std::slice::from_ref(&text), 1, 2.0, None, &mut |_| true)
            .unwrap();
        assert!(m.resonant_score(&text).log_prob < rewarded);
        assert_eq!(m.meta.feedback_passes.value, 2);
    }

    #[test]
    fn configure_rephases_and_refuses_nonsense() {
        let mut m = trained();
        let config = m.g.configure_resonant(&[("buckets".to_string(), 4.0)]).unwrap();
        assert_eq!(config.at("buckets").as_i64(), Some(4));
        assert!(m.g.configure_resonant(&[("period".to_string(), 0.0)]).is_err());
        assert!(m
            .g
            .configure_resonant(&[("sideways".to_string(), 1.0)])
            .unwrap_err()
            .contains("unknown weight option"));
    }
}
