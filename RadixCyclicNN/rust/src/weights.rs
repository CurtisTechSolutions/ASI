//! The dual frequency weight function over counts and rewards, the softmax
//! costs it turns into, and the punishment that the least-punished traversal
//! ranks by.

use std::sync::atomic::Ordering;

use crate::counter::{Counter, INVALID_STAMP};
use crate::fsum::fsum;
use crate::graph::Graph;
use crate::parallel::{parallel_ranges, Disjoint};
use crate::paths::PathKey;

/// The additive smoothing of both frequency ratios.
pub const SMOOTHING: f64 = 0.5;

/// One out-edge of a node with its `-log` softmax probability and the
/// punishment the model carries against that step.
#[derive(Clone, Copy, Debug)]
pub struct ChildCost {
    pub child: usize,
    pub edge: usize,
    pub cost: f64,
    pub punish: f64,
}

impl Graph {
    /// The dual frequency function for one edge:
    ///
    /// ```text
    /// R_all    = (count + s) / (parentTotal + s * degree)
    /// R_recent = (windowCount + s) / (windowTotal + s * degree)
    /// weight   = countScale * log(1 + count) + globalScale * log(R_all)
    ///          + windowScale * log(R_recent) + rewardScale * reward
    /// ```
    pub fn edge_weight(
        &self,
        count: f64,
        reward: f64,
        parent_total: f64,
        degree: usize,
        window_count: f64,
        window_total: f64,
    ) -> f64 {
        let s = SMOOTHING;
        let count = count.max(0.0);
        let window_count = window_count.max(0.0);
        let d = degree.max(1) as f64;
        let r_all = (count + s) / (parent_total.max(0.0) + s * d);
        let r_recent = (window_count + s) / (window_total.max(0.0) + s * d);
        self.count_scale * count.ln_1p()
            + self.global_scale * r_all.ln()
            + self.window_scale * r_recent.ln()
            + self.reward_scale * reward
    }

    /// The number of traversals currently inside the sliding window.
    pub fn window_traversals(&self) -> usize {
        self.window.len() - self.window_head
    }

    /// Counts traversals of edges in order: all time (already counted by the
    /// walk), in the sliding window and in the global total, and marks the
    /// parents whose rows changed.
    pub fn record_traversals(&mut self, edges: &[usize]) -> usize {
        let limit = self.window_size;
        for &e in edges {
            self.window.push(e);
            self.window_edge_count[e] += 1;
            self.dirty.insert(self.edge_parent[e]);
            if self.window.len() - self.window_head > limit {
                let old = self.window[self.window_head];
                self.window_head += 1;
                if old < self.window_edge_count.len() && self.window_edge_count[old] > 0 {
                    self.window_edge_count[old] -= 1;
                    self.dirty.insert(self.edge_parent[old]);
                }
            }
        }
        if self.window_head > 4096 && self.window_head > self.window.len() / 2 {
            // slide the live part down inside the same buffer: a counting pass
            // over a huge corpus compacts millions of times and must not allocate
            self.window.drain(..self.window_head);
            self.window_head = 0;
        }
        self.total_traversals.add(edges.len() as i64);
        edges.len()
    }

    /// Writes the weight function to every edge leaving `p`.
    ///
    /// # Safety
    /// `out` must be a view of `edge_w`, and no other thread may be writing the
    /// edges of `p` (see [`Disjoint`]).
    unsafe fn recompute_row(&self, p: usize, out: &Disjoint<'_, f64>) {
        let adj = &self.children[p];
        if adj.size() == 0 || !self.alive[p] {
            return;
        }
        let mut counts = Vec::with_capacity(adj.order.len());
        let mut total = 0.0;
        let mut recent = 0i64;
        for &e in &adj.edges {
            let c = self.edge_traversals_f(e);
            counts.push(c);
            total += c; // an explicit left-to-right sum, as in the Python implementation
            recent += self.window_edge_count[e];
        }
        let degree = adj.size();
        for (i, &e) in adj.edges.iter().enumerate() {
            let w = self.edge_weight(
                counts[i],
                self.edge_reward[e],
                total,
                degree,
                self.window_edge_count[e] as f64,
                recent as f64,
            );
            out.set(e, w);
        }
    }

    /// Writes the weight function to every alive edge, in parallel over the
    /// nodes for large graphs.
    pub fn recompute_weights(&mut self) {
        let n = self.labels.len();
        let mut edge_w = std::mem::take(&mut self.edge_w);
        {
            let out = Disjoint::new(&mut edge_w);
            if n < 4096 || crate::parallel::effective_workers(self.workers) == 1 {
                for p in 0..n {
                    // SAFETY: one thread, and every edge belongs to one parent.
                    unsafe { self.recompute_row(p, &out) };
                }
            } else {
                let this: &Graph = self;
                let out = &out;
                parallel_ranges(n, self.workers, move |lo, hi| {
                    for p in lo..hi {
                        // SAFETY: the ranges are disjoint and so are the parents'
                        // edges, so no two threads write the same index.
                        unsafe { this.recompute_row(p, out) };
                    }
                });
            }
        }
        self.edge_w = edge_w;
        self.dirty.clear();
        self.dirty_all = false;
        self.weights_structure = self.structure_version;
        self.version.add(1);
    }

    /// Brings the weights up to date: every row after a structural change or a
    /// global change, otherwise only the rows of the touched parents.
    pub fn flush_weights(&mut self) {
        if self.dirty_all || self.weights_structure != self.structure_version {
            self.recompute_weights();
            return;
        }
        if self.dirty.is_empty() {
            return;
        }
        if self.dirty.len() > self.labels.len() / 4 {
            self.recompute_weights();
            return;
        }
        let dirty: Vec<usize> = self.dirty.drain().collect();
        let mut edge_w = std::mem::take(&mut self.edge_w);
        {
            let out = Disjoint::new(&mut edge_w);
            for p in dirty {
                if p < self.labels.len() {
                    // SAFETY: one thread.
                    unsafe { self.recompute_row(p, &out) };
                }
            }
        }
        self.edge_w = edge_w;
        self.version.add(1);
    }

    /// Whether [`Graph::prepare`] would change anything.
    pub fn weights_stale(&self) -> bool {
        self.dirty_all || self.weights_structure != self.structure_version || !self.dirty.is_empty()
    }

    /// Adds `amount` (negative = penalty) to the reward of every listed alive
    /// edge; returns how many were touched.
    pub fn add_reward(&mut self, edges: &[usize], amount: f64) -> usize {
        let mut touched = 0;
        for &e in edges {
            if e < self.edge_reward.len() && self.edge_alive[e] {
                self.edge_reward[e] += amount;
                self.dirty.insert(self.edge_parent[e]);
                touched += 1;
            }
        }
        touched
    }

    /// `(sum of positive rewards, sum of negative rewards)` over alive edges.
    pub fn total_reward(&self) -> (f64, f64) {
        let (mut pos, mut neg) = (0.0, 0.0);
        for (e, &ok) in self.edge_alive.iter().enumerate() {
            if ok {
                let r = self.edge_reward[e];
                if r > 0.0 {
                    pos += r;
                } else {
                    neg += r;
                }
            }
        }
        (pos, neg)
    }

    /// Flips the sign of every reward.
    pub fn invert(&mut self) {
        for e in 0..self.edge_reward.len() {
            if self.edge_alive[e] {
                self.edge_reward[e] = -self.edge_reward[e];
            }
        }
        self.inverted = !self.inverted;
        self.dirty_all = true;
        self.flush_weights();
    }

    /// Changes the sliding window's size, dropping whatever no longer fits in
    /// it (as `Graph.configure` does in the other two implementations).
    pub fn set_window(&mut self, size: usize) {
        self.window_size = size.max(1);
        while self.window.len() - self.window_head > self.window_size {
            let old = self.window[self.window_head];
            self.window_head += 1;
            if old < self.window_edge_count.len() && self.window_edge_count[old] > 0 {
                self.window_edge_count[old] -= 1;
            }
        }
        self.dirty_all = true;
    }

    // -- punishment ---------------------------------------------------------

    /// What the model has been taught *against* one edge: the penalty side of
    /// its reward, on the same scale the reward function uses.
    ///
    /// It is deliberately one-sided.  A rewarded edge is not *less* punished
    /// than an edge nothing was ever said about - it is exactly as unpunished,
    /// which is what lets the least-punished traversal rank walks by what went
    /// wrong on them instead of by what went well.
    pub fn edge_punishment(&self, e: usize) -> f64 {
        match self.edge_reward.get(e) {
            // not `(-r).max(0.0)`: that hands back a negative zero for a
            // rewarded edge, and a punishment reads as a number one prints
            Some(&r) if r < 0.0 => self.reward_scale * -r,
            _ => 0.0,
        }
    }

    /// The punishment of one step taken from `prev`: the edge's own, plus
    /// `path_scale * log(1 + incorrect)` for the walks that came from `prev`
    /// and were judged wrong here.
    ///
    /// The second term counts the failures alone - not the failures against the
    /// successes, the way the cost function's path term weighs them.  That is
    /// the point: a step that was wrong here once is a step that was wrong
    /// here, and no amount of being right afterwards makes it a step nothing is
    /// held against.  Blame cannot be bought off, which is what the traversal
    /// is for.
    pub fn step_punishment(&self, prev: Option<usize>, e: usize) -> f64 {
        let mut punish = self.edge_punishment(e);
        if let Some(prev) = prev {
            if self.path_scale != 0.0 && !self.paths.is_empty() {
                punish += self.path_scale * (self.path_incorrect(prev, e) as f64).ln_1p();
            }
        }
        punish
    }

    // -- costs --------------------------------------------------------------

    /// Recomputes the per-edge costs when weights or structure changed
    /// (parallel over the nodes for large graphs).
    fn ensure_costs(&mut self) {
        if self.costs_version == self.version && self.edge_cost.len() == self.edge_w.len() {
            return;
        }
        if self.edge_cost.len() != self.edge_w.len() {
            self.edge_cost = vec![0.0; self.edge_w.len()];
            self.edge_punish = vec![0.0; self.edge_w.len()];
        }
        let n = self.labels.len();
        let mut edge_cost = std::mem::take(&mut self.edge_cost);
        let mut edge_punish = std::mem::take(&mut self.edge_punish);
        {
            let costs = Disjoint::new(&mut edge_cost);
            let punish = Disjoint::new(&mut edge_punish);
            let row = |this: &Graph, p: usize| {
                let adj = &this.children[p];
                if adj.size() == 0 || !this.alive[p] {
                    return;
                }
                let mut m = f64::NEG_INFINITY;
                for &e in &adj.edges {
                    if this.edge_w[e] > m {
                        m = this.edge_w[e];
                    }
                }
                let mut sum = 0.0;
                for &e in &adj.edges {
                    sum += (this.edge_w[e] - m).exp();
                }
                let lse = m + sum.ln();
                for &e in &adj.edges {
                    // SAFETY: every edge belongs to exactly one parent, so no
                    // other thread writes these indices.
                    unsafe {
                        costs.set(e, lse - this.edge_w[e]);
                        punish.set(e, this.edge_punishment(e));
                    }
                }
            };
            if n < 4096 || crate::parallel::effective_workers(self.workers) == 1 {
                for p in 0..n {
                    row(self, p);
                }
            } else {
                let this: &Graph = self;
                let row = &row;
                parallel_ranges(n, self.workers, move |lo, hi| {
                    for p in lo..hi {
                        row(this, p);
                    }
                });
            }
        }
        self.edge_cost = edge_cost;
        self.edge_punish = edge_punish;
        self.costs_version = self.version;
    }

    /// Brings weights, edge costs and the judged contexts up to date.  Call it
    /// before reading costs from several threads; afterwards the searches are
    /// safe to run concurrently as long as nothing mutates the graph.
    pub fn prepare(&mut self) {
        self.flush_weights();
        self.ensure_costs();
        self.ensure_context_costs();
    }

    /// Prices every judged context, once, while the graph is still exclusively
    /// borrowed: the searches then only read the table.
    fn ensure_context_costs(&mut self) {
        if self.ctx_version == self.version {
            return;
        }
        self.ctx_version = self.version;
        self.ctx_cache.clear();
        if self.paths.is_empty() || self.path_scale == 0.0 {
            return;
        }
        // (the node that called, the node whose children it prices)
        let mut pairs: Vec<(usize, usize)> = self
            .paths
            .keys()
            .filter(|key| key.edge < self.edge_parent.len())
            .map(|key| (key.prev, self.edge_parent[key.edge]))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let mut cache = crate::hash::map();
        for (prev, p) in pairs {
            if p >= self.children.len() || !self.alive[p] {
                continue;
            }
            let adj = &self.children[p];
            if adj.size() == 0 {
                continue;
            }
            let mut weights = Vec::with_capacity(adj.order.len());
            let mut punish = Vec::with_capacity(adj.order.len());
            let mut m = f64::NEG_INFINITY;
            for &e in &adj.edges {
                let w = self.edge_w[e] + self.path_scale * self.path_term(prev, e);
                punish.push(self.edge_punish[e] + self.path_scale * (self.path_incorrect(prev, e) as f64).ln_1p());
                weights.push(w);
                if w > m {
                    m = w;
                }
            }
            let terms: Vec<f64> = weights.iter().map(|w| (w - m).exp()).collect();
            let lse = m + fsum(&terms).ln();
            let costs: Vec<ChildCost> = adj
                .order
                .iter()
                .enumerate()
                .map(|(i, &c)| ChildCost {
                    child: c,
                    edge: adj.edges[i],
                    cost: lse - weights[i],
                    punish: punish[i],
                })
                .collect();
            cache.insert(PathKey { prev, edge: p }, costs);
        }
        self.ctx_cache = cache;
    }

    /// Lists `p`'s out-edges with their costs (`-log` softmax and nothing
    /// else).  The graph must be [`Graph::prepare`]d.
    pub fn child_costs(&self, p: usize) -> Vec<ChildCost> {
        let adj = &self.children[p];
        adj.order
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let e = adj.edges[i];
                ChildCost {
                    child: c,
                    edge: e,
                    cost: self.edge_cost[e],
                    punish: self.edge_punish[e],
                }
            })
            .collect()
    }

    /// Fills `out` with `p`'s out-edges as a walk that arrived from `prev` sees
    /// them.  Without a judged context the costs are the edge costs; where a
    /// path *has* been judged, its context adds
    /// `path_scale * log((correct + s) / (incorrect + s))` to that edge's
    /// weight before the softmax - so the same edge is cheap for the walk that
    /// was right here and dear for the one that was wrong.
    ///
    /// The buffer is the caller's, and a search reuses one: a beam expansion
    /// reads a node's children without allocating anything.
    pub fn child_costs_into(&self, p: usize, prev: Option<usize>, out: &mut Vec<ChildCost>) {
        out.clear();
        if let Some(prev) = prev {
            if self.path_scale != 0.0 && !self.paths.is_empty() {
                if let Some(costs) = self.ctx_cache.get(&PathKey { prev, edge: p }) {
                    out.extend_from_slice(costs);
                    return;
                }
            }
        }
        let adj = &self.children[p];
        out.reserve(adj.order.len());
        for (i, &c) in adj.order.iter().enumerate() {
            let e = adj.edges[i];
            out.push(ChildCost {
                child: c,
                edge: e,
                cost: self.edge_cost[e],
                punish: self.edge_punish[e],
            });
        }
    }

    /// [`Graph::child_costs_into`] into a fresh vector.
    pub fn child_costs_from(&self, p: usize, prev: Option<usize>) -> Vec<ChildCost> {
        let mut out = Vec::new();
        self.child_costs_into(p, prev, &mut out);
        out
    }

    /// The cost of one edge after [`Graph::prepare`].
    pub fn edge_cost(&self, e: usize) -> f64 {
        self.edge_cost[e]
    }

    /// The version stamp the caches are keyed on.
    pub fn version(&self) -> Counter {
        self.version
    }

    /// Marks every cache stale (the model file loader and the tests use it).
    pub fn invalidate(&mut self) {
        self.dirty_all = true;
        self.costs_version = INVALID_STAMP;
        self.ctx_version = INVALID_STAMP;
    }

    /// How often the visit counters were bumped in total.
    pub fn traversal_total(&self) -> Counter {
        self.traversals
    }

    /// The sliding window's total.
    pub fn total_traversals(&self) -> Counter {
        self.total_traversals
    }

    /// One node's out-edge counts, as the shares they make up of the node's
    /// traffic all time and inside the window.
    pub fn shares(&self, p: usize) -> Vec<(usize, usize, f64, f64)> {
        let adj = &self.children[p];
        let mut counts = Vec::with_capacity(adj.order.len());
        let mut total = 0.0;
        let mut recent = 0i64;
        for &e in &adj.edges {
            let c =
                crate::counter::counter_total(self.edge_count[e].load(Ordering::Relaxed), &self.edge_count_resets, e);
            counts.push(c);
            total += c;
            recent += self.window_edge_count[e];
        }
        adj.order
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let e = adj.edges[i];
                let all = if total > 0.0 { counts[i] / total } else { 0.0 };
                let rec = if recent > 0 {
                    self.window_edge_count[e] as f64 / recent as f64
                } else {
                    0.0
                };
                (c, e, all, rec)
            })
            .collect()
    }
}
