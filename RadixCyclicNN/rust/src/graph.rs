//! The self-compressing cyclic trigram graph: split, merge, and the invariants.
//!
//! Every node holds a label of three or more characters and therefore one or
//! more trigrams.  A unary chain - `p` has one child `c`, `c` has one parent
//! `p` - merges the way a radix tree merges a path (`"hel" + "ell" + "llo" ->
//! "hello"`), and a transition observed into or out of the *middle* of a merged
//! node splits it again.  Repeated trigrams create cycles; cycles are a feature.

use std::sync::atomic::{AtomicI64, Ordering};

use crate::counter::{carry, Counter, COUNTER_LIMIT, INVALID_STAMP};
use crate::encoding::{Encoding, Gram, BACK_LABEL, END_LABEL, START_LABEL};
use crate::hash::{map, Map, Set};
use crate::mt19937::Mt19937;
use crate::paths::{PathKey, PathRow};
use crate::weights::ChildCost;

/// Node ids of the sentinels.  `START` and `END` are where a text begins and
/// ends - observed in the corpus, like everything else.  `BACK` is where the
/// graph has *learned* that a walk goes round: nothing in a corpus says so, so
/// its edges are taught by the voices that caught themselves repeating.
/// `FIRST` is the first node id that is not a sentinel.
pub const START: usize = 0;
pub const END: usize = 1;
pub const BACK: usize = 2;
pub const FIRST: usize = 3;

/// New edge weights of the sine network are drawn from `[W_LOW, W_HIGH]`; the
/// count model discards the value but consumes the same random number, so the
/// RNG state stays in step with the Python implementation.
pub const W_LOW: f64 = 0.5;
pub const W_HIGH: f64 = 1.5;

/// Where a trigram lives: the node holding it and its offset in that label.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Loc {
    pub node: usize,
    pub off: usize,
}

/// One traversed edge: the parent it leaves and the edge id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Transition {
    pub p: usize,
    pub e: usize,
}

/// The degree above which an adjacency keeps a map index.
const ADJACENCY_INDEX_AT: usize = 16;

/// A node's edges keyed by the node at the other end - child -> edge id for the
/// out-edges, parent -> edge id for the in-edges - in insertion order, which
/// decides the summation order of the softmax and therefore keeps the numbers
/// identical to the other two implementations.  Two parallel vectors searched
/// linearly cover the small degrees of almost every node; a node past
/// [`ADJACENCY_INDEX_AT`] entries (the sentinels, hubs like `" th"`) gets a map
/// index too.
#[derive(Default, Clone)]
pub(crate) struct Adjacency {
    pub order: Vec<usize>,
    pub edges: Vec<usize>,
    idx: Option<Map<usize, usize>>,
}

impl Adjacency {
    #[inline]
    fn find(&self, c: usize) -> Option<usize> {
        match &self.idx {
            Some(idx) => idx.get(&c).copied(),
            None => self.order.iter().position(|&x| x == c),
        }
    }

    #[inline]
    pub fn get(&self, c: usize) -> Option<usize> {
        self.find(c).map(|i| self.edges[i])
    }

    /// Records edge `e` to / from node `c`, appending `c` when it is new.
    fn set(&mut self, c: usize, e: usize) {
        if let Some(i) = self.find(c) {
            self.edges[i] = e;
            return;
        }
        self.order.push(c);
        self.edges.push(e);
        match &mut self.idx {
            Some(idx) => {
                idx.insert(c, self.order.len() - 1);
            }
            None => {
                if self.order.len() > ADJACENCY_INDEX_AT {
                    let mut idx: Map<usize, usize> = map();
                    for (i, &x) in self.order.iter().enumerate() {
                        idx.insert(x, i);
                    }
                    self.idx = Some(idx);
                }
            }
        }
    }

    /// Drops node `c`; the last entry takes its slot, so the order of the rest
    /// changes - fine for the parents, which have no order to keep.
    fn unset(&mut self, c: usize) -> bool {
        let Some(i) = self.find(c) else { return false };
        let last = self.order.len() - 1;
        if i != last {
            self.order.swap(i, last);
            self.edges.swap(i, last);
            if let Some(idx) = &mut self.idx {
                idx.insert(self.order[i], i);
            }
        }
        self.order.pop();
        self.edges.pop();
        if let Some(idx) = &mut self.idx {
            idx.remove(&c);
        }
        true
    }

    fn clear(&mut self) {
        self.order.clear();
        self.edges.clear();
        self.idx = None;
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.order.len()
    }
}

/// The dual frequency function's scales and the window size.
#[derive(Clone, Copy, Debug)]
pub struct GraphOptions {
    pub count_scale: f64,
    pub reward_scale: f64,
    pub global_scale: f64,
    pub window_scale: f64,
    pub path_scale: f64,
    /// The size of the sliding *count* window, not the n of the n-gram - that
    /// is `encoding.n`.
    pub window: usize,
    /// How text becomes grams; the default is the character trigram.
    pub encoding: Encoding,
}

impl Default for GraphOptions {
    /// The Python defaults: the geometric mean of the two shares.
    fn default() -> GraphOptions {
        GraphOptions {
            count_scale: 0.0,
            reward_scale: 1.0,
            global_scale: 0.5,
            window_scale: 0.5,
            path_scale: 1.0,
            window: 10_000,
            encoding: Encoding::default(),
        }
    }
}

/// The self-compressing cyclic trigram graph of the count / reward model.
/// Storage is flat vectors indexed by node id / edge id; removed nodes and
/// edges are tombstoned and their ids are never reused.
pub struct Graph {
    pub seed: i64,
    pub(crate) rng: Mt19937,

    pub(crate) labels: Vec<String>,
    pub(crate) label_len: Vec<usize>,
    pub(crate) count: Vec<AtomicI64>,
    pub(crate) alive: Vec<bool>,
    pub(crate) children: Vec<Adjacency>,
    pub(crate) parents: Vec<Adjacency>,

    pub(crate) edge_w: Vec<f64>,
    pub(crate) edge_count: Vec<AtomicI64>,
    pub(crate) edge_alive: Vec<bool>,
    pub(crate) edge_parent: Vec<usize>,

    // Every visit count is a cyclic counter: the vectors above hold the
    // odometer readings, these maps the reset counts of the ids that ever
    // wrapped, and carry_counters does the wrapping - never the counting loops.
    pub(crate) count_resets: std::collections::HashMap<usize, i64>,
    pub(crate) edge_count_resets: std::collections::HashMap<usize, i64>,
    /// Counts every increment made to those counters and so bounds each of them.
    pub(crate) traversals: Counter,

    pub(crate) edge_reward: Vec<f64>,
    pub(crate) window_edge_count: Vec<i64>,
    pub(crate) window: Vec<usize>,
    pub(crate) window_head: usize,
    pub window_size: usize,
    pub(crate) total_traversals: Counter,
    pub count_scale: f64,
    pub reward_scale: f64,
    pub global_scale: f64,
    pub window_scale: f64,
    pub path_scale: f64,

    pub(crate) index: Map<Gram, Loc>,
    pub inverted: bool,

    /// How this graph turns text into grams and its labels back into text.
    /// Fixed when the graph is created - every label, every index key and
    /// every offset is measured in its units.
    pub enc: Encoding,

    pub(crate) version: Counter,
    pub(crate) structure_version: Counter,
    pub(crate) n_alive_nodes: usize,
    pub(crate) n_alive_edges: usize,

    // what the walks did: (the node before an edge's parent, the edge)
    pub(crate) paths: Map<PathKey, PathRow>,
    pub(crate) paths_by_edge: Map<usize, Set<usize>>,
    pub(crate) paths_by_prev: Map<usize, Set<usize>>,
    pub(crate) ctx_cache: Map<PathKey, Vec<ChildCost>>,
    pub(crate) ctx_version: Counter,

    // lazy weights and costs
    pub(crate) dirty: Set<usize>,
    pub(crate) dirty_all: bool,
    pub(crate) weights_structure: Counter,
    pub(crate) edge_cost: Vec<f64>,
    pub(crate) edge_punish: Vec<f64>,
    pub(crate) costs_version: Counter,

    /// The fan-out cap of the recomputes (0 = the machine).
    pub workers: usize,
}

impl Graph {
    /// An empty graph with the three sentinels.
    pub fn new(seed: i64, opts: GraphOptions) -> Result<Graph, String> {
        if opts.window < 1 {
            return Err(format!("window must be >= 1, got {}", opts.window));
        }
        opts.encoding.validate()?;
        let mut g = Graph {
            enc: opts.encoding,
            seed,
            rng: Mt19937::new(seed),
            labels: Vec::new(),
            label_len: Vec::new(),
            count: Vec::new(),
            alive: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            edge_w: Vec::new(),
            edge_count: Vec::new(),
            edge_alive: Vec::new(),
            edge_parent: Vec::new(),
            count_resets: std::collections::HashMap::new(),
            edge_count_resets: std::collections::HashMap::new(),
            traversals: Counter::default(),
            edge_reward: Vec::new(),
            window_edge_count: Vec::new(),
            window: Vec::new(),
            window_head: 0,
            window_size: opts.window,
            total_traversals: Counter::default(),
            count_scale: opts.count_scale,
            reward_scale: opts.reward_scale,
            global_scale: opts.global_scale,
            window_scale: opts.window_scale,
            path_scale: opts.path_scale,
            index: map(),
            inverted: false,
            version: Counter::default(),
            structure_version: Counter::default(),
            n_alive_nodes: 0,
            n_alive_edges: 0,
            paths: map(),
            paths_by_edge: map(),
            paths_by_prev: map(),
            ctx_cache: map(),
            ctx_version: INVALID_STAMP,
            dirty: crate::hash::set(),
            dirty_all: false,
            weights_structure: INVALID_STAMP,
            edge_cost: Vec::new(),
            edge_punish: Vec::new(),
            costs_version: INVALID_STAMP,
            workers: 0,
        };
        g.new_node(START_LABEL.to_string(), 0, 0);
        g.new_node(END_LABEL.to_string(), 0, 0);
        g.new_node(BACK_LABEL.to_string(), 0, 0);
        Ok(g)
    }

    // -- counters -----------------------------------------------------------

    /// How often node `i` was visited, as an odometer reading.
    pub fn node_count(&self, i: usize) -> Counter {
        Counter {
            value: self.count[i].load(Ordering::Relaxed),
            resets: self.count_resets.get(&i).copied().unwrap_or(0),
        }
    }

    /// How often edge `e` was traversed, as an odometer reading.
    pub fn edge_traversals(&self, e: usize) -> Counter {
        Counter {
            value: self.edge_count[e].load(Ordering::Relaxed),
            resets: self.edge_count_resets.get(&e).copied().unwrap_or(0),
        }
    }

    /// The exact traversal count of an edge, as the float the weight function sums.
    pub(crate) fn edge_traversals_f(&self, e: usize) -> f64 {
        crate::counter::counter_total(self.edge_count[e].load(Ordering::Relaxed), &self.edge_count_resets, e)
    }

    /// Sets every counter that reached the limit back to 0, counting the reset;
    /// returns how many wrapped.  This is the only place the visit counters
    /// wrap, so the counting loops stay plain increments.
    pub fn carry_counters(&mut self, force: bool) -> usize {
        if !force && self.traversals.resets == 0 {
            return 0;
        }
        fn sweep(values: &mut [AtomicI64], resets: &mut std::collections::HashMap<usize, i64>) -> usize {
            let mut wrapped = 0;
            for (i, slot) in values.iter_mut().enumerate() {
                let raw = *slot.get_mut();
                if !(0..COUNTER_LIMIT).contains(&raw) {
                    let (value, reset) = carry(raw, resets.get(&i).copied().unwrap_or(0));
                    *slot.get_mut() = value;
                    if reset != 0 {
                        resets.insert(i, reset);
                    } else {
                        resets.remove(&i);
                    }
                    wrapped += 1;
                }
            }
            wrapped
        }
        sweep(&mut self.count, &mut self.count_resets) + sweep(&mut self.edge_count, &mut self.edge_count_resets)
    }

    // -- construction -------------------------------------------------------

    pub(crate) fn new_node(&mut self, label: String, count: i64, count_resets: i64) -> usize {
        let nid = self.labels.len();
        self.label_len.push(self.enc.len(&label));
        self.labels.push(label);
        self.count.push(AtomicI64::new(count));
        if count_resets != 0 {
            self.count_resets.insert(nid, count_resets);
        }
        self.alive.push(true);
        self.children.push(Adjacency::default());
        self.parents.push(Adjacency::default());
        self.n_alive_nodes += 1;
        self.version.add(1);
        self.structure_version.add(1);
        nid
    }

    pub(crate) fn new_edge(&mut self, p: usize, c: usize, count: i64, count_resets: i64) -> usize {
        // the sine network draws a weight here; the count model recomputes the
        // weight but consumes the same random number
        self.rng.uniform(W_LOW, W_HIGH);
        let e = self.edge_w.len();
        self.edge_w.push(0.0);
        self.edge_count.push(AtomicI64::new(count));
        if count_resets != 0 {
            self.edge_count_resets.insert(e, count_resets);
        }
        self.edge_alive.push(true);
        self.edge_parent.push(p);
        self.edge_reward.push(0.0);
        self.window_edge_count.push(0);
        self.children[p].set(c, e);
        self.parents[c].set(p, e);
        self.n_alive_edges += 1;
        self.version.add(1);
        self.structure_version.add(1);
        self.dirty.insert(p);
        e
    }

    fn create_trigram_node(&mut self, trigram: Gram) -> usize {
        let nid = self.new_node(trigram.to_string(), 0, 0);
        self.index.insert(trigram, Loc { node: nid, off: 0 });
        nid
    }

    // -- sizes and lookup ---------------------------------------------------

    /// The number of alive nodes, sentinels included.
    pub fn num_nodes(&self) -> usize {
        self.n_alive_nodes
    }
    /// The number of alive edges.
    pub fn num_edges(&self) -> usize {
        self.n_alive_edges
    }
    /// The number of distinct trigrams stored.
    pub fn num_trigrams(&self) -> usize {
        self.index.len()
    }
    /// How many node ids exist (alive or tombstoned).
    pub fn num_node_ids(&self) -> usize {
        self.labels.len()
    }
    /// How many edge ids exist.
    pub fn num_edge_ids(&self) -> usize {
        self.edge_w.len()
    }
    /// Grams per real node.
    pub fn compression_ratio(&self) -> f64 {
        let real = (self.n_alive_nodes as i64 - FIRST as i64).max(1) as f64;
        self.index.len() as f64 / real
    }
    /// Whether an edge id is alive.
    pub fn is_edge_alive(&self, e: usize) -> bool {
        self.edge_alive.get(e).copied().unwrap_or(false)
    }
    /// One edge's reward (negative = a penalty).
    pub fn edge_reward(&self, e: usize) -> f64 {
        self.edge_reward.get(e).copied().unwrap_or(0.0)
    }
    /// Whether a node id is alive (a tombstoned or unknown id is not).
    pub fn is_alive(&self, node: usize) -> bool {
        self.alive.get(node).copied().unwrap_or(false)
    }
    /// A node's label (`""` when the id is out of range).
    pub fn label(&self, node: usize) -> &str {
        self.labels.get(node).map(|s| s.as_str()).unwrap_or("")
    }
    /// The character length of a node's label.
    pub fn label_len(&self, node: usize) -> usize {
        self.label_len[node]
    }
    /// Finds the node and offset holding a trigram.
    pub fn lookup(&self, trigram: &Gram) -> Option<Loc> {
        self.index.get(trigram).copied()
    }
    /// The id of edge `p -> c`.
    pub fn edge(&self, p: usize, c: usize) -> Option<usize> {
        self.children.get(p).and_then(|a| a.get(c))
    }
    /// The number of out-edges of `p`.
    pub fn degree(&self, p: usize) -> usize {
        self.children[p].size()
    }
    /// `(child, edge)` pairs of `p` in insertion order.
    pub fn children_of(&self, p: usize) -> Vec<Transition> {
        let adj = &self.children[p];
        adj.order
            .iter()
            .zip(adj.edges.iter())
            .map(|(&c, &e)| Transition { p: c, e })
            .collect()
    }
    /// The node an edge leaves, or `None`.
    pub fn parent_of_edge(&self, edge: usize) -> Option<usize> {
        self.edge_parent.get(edge).copied()
    }

    /// Every trigram with the node and offset holding it.
    pub fn index_entries(&self) -> impl Iterator<Item = (&Gram, Loc)> + '_ {
        self.index.iter().map(|(t, &l)| (t, l))
    }

    /// Counts one traversal of an edge.  The counting passes call it from
    /// several threads at once, which is what makes the counters atomic.
    #[inline]
    pub fn bump_edge(&self, e: usize) {
        self.edge_count[e].fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one visit of a node.
    #[inline]
    pub fn bump_node(&self, n: usize) {
        self.count[n].fetch_add(1, Ordering::Relaxed);
    }

    /// Records that `n` counter increments were made, which is what bounds
    /// every single counter and so decides when a carry sweep has work.
    pub fn add_traversals(&mut self, n: i64) {
        self.traversals.add(n);
    }

    // -- structural operations ----------------------------------------------

    /// Cuts a node between its trigrams `i - 1` and `i`: `a` keeps the id with
    /// `label[..i + 2]`, `b` is a new node with `label[i..]` that inherits
    /// `a`'s out-edges (edge ids kept) and its count; `a` gets the single new
    /// edge `a -> b`.
    pub fn split(&mut self, node: usize, i: usize) -> Result<(usize, usize), String> {
        if node < FIRST {
            return Err("cannot split a sentinel node".to_string());
        }
        if node >= self.labels.len() || !self.alive[node] {
            return Err(format!("node {node} is not alive"));
        }
        let enc = self.enc;
        let (stride, ov, n) = (enc.stride, enc.overlap(), enc.n);
        let owned = self.labels[node].clone();
        let label = enc.units(&owned);
        let length = label.len();
        if i < stride || i + n > length || i % stride != 0 {
            return Err(format!(
                "split index {i} out of range {stride}..{} (a multiple of the stride {stride}) for label {:?}",
                length.saturating_sub(n),
                self.labels[node]
            ));
        }
        let a = node;
        let a_resets = self.count_resets.get(&a).copied().unwrap_or(0);
        let a_count = self.count[a].load(Ordering::Relaxed);
        let b = self.new_node(label.slice(i, None).to_string(), a_count, a_resets);

        let moved: Vec<usize> = self.children[a].edges.clone();
        let pairs: Vec<(usize, usize)> = self.children[a]
            .order
            .iter()
            .copied()
            .zip(self.children[a].edges.iter().copied())
            .collect();
        for (c, e) in pairs {
            self.children[b].set(c, e);
            self.parents[c].unset(a);
            self.parents[c].set(b, e);
            self.edge_parent[e] = b;
        }
        self.children[a].clear();
        self.new_edge(a, b, a_count, a_resets);
        let mut j = i;
        while j + n <= length {
            self.index.insert(enc.gram(&label, j), Loc { node: b, off: j - i });
            j += stride;
        }
        self.labels[a] = label.slice(0, Some(i + ov)).to_string();
        self.label_len[a] = i + ov;
        self.dirty_all = true;
        let bridge = self.children[a].get(b);
        self.split_paths(a, b, &moved, bridge); // q -> P -> c is now q -> A -> B -> c
        Ok((a, b))
    }

    /// Merges `p`'s single child into it when the chain is unary (`p` has
    /// exactly one child, that child exactly one parent, no sentinels, and the
    /// two are not the same node).
    pub fn merge_child(&mut self, p: usize) -> bool {
        if p < FIRST || p >= self.labels.len() || !self.alive[p] {
            return false;
        }
        if self.children[p].size() != 1 {
            return false;
        }
        let c = self.children[p].order[0];
        if c == p || c < FIRST || self.parents[c].size() != 1 {
            return false;
        }
        let enc = self.enc;
        let (lp_owned, lc_owned) = (self.labels[p].clone(), self.labels[c].clone());
        let lp = enc.units(&lp_owned);
        let lc = enc.units(&lc_owned);
        let ov = enc.overlap();
        let shift = lp.len() - ov;
        let e = self.children[p].edges[0];
        let moved_out: Vec<usize> = self.children[c].edges.clone();
        self.children[p].clear();
        self.parents[c].clear();
        self.edge_alive[e] = false;
        self.n_alive_edges -= 1;
        // activations are the constant 1: no rescale of the moved edges is needed
        let pairs: Vec<(usize, usize)> = self.children[c]
            .order
            .iter()
            .copied()
            .zip(self.children[c].edges.iter().copied())
            .collect();
        for (target, e2) in pairs {
            let target = if target == c { p } else { target };
            self.parents[target].unset(c);
            self.parents[target].set(p, e2);
            self.children[p].set(target, e2);
            self.edge_parent[e2] = p;
        }
        self.children[c].clear();
        let mut j = 0;
        while j + enc.n <= lc.len() {
            self.index.insert(
                enc.gram(&lc, j),
                Loc {
                    node: p,
                    off: shift + j,
                },
            );
            j += enc.stride;
        }
        self.labels[p] = enc.join(&[lp.text(), lc.slice(ov, None)]);
        self.label_len[p] = lp.len() + lc.len() - ov;
        self.labels[c] = String::new();
        self.label_len[c] = 0;
        if self.node_count(p).less(self.node_count(c)) {
            let value = self.count[c].load(Ordering::Relaxed);
            self.count[p].store(value, Ordering::Relaxed);
            match self.count_resets.get(&c).copied().unwrap_or(0) {
                0 => {
                    self.count_resets.remove(&p);
                }
                r => {
                    self.count_resets.insert(p, r);
                }
            }
        }
        self.alive[c] = false;
        self.n_alive_nodes -= 1;
        self.version.add(1);
        self.structure_version.add(1);
        self.dirty_all = true;
        self.merge_paths(p, c, e, &moved_out); // the chain was unary: what its contexts knew was never a choice
        true
    }

    /// Merges every unary chain until none remains; returns the merge count.
    pub fn compress(&mut self) -> usize {
        let mut merges = 0;
        loop {
            let mut done = 0;
            for p in FIRST..self.labels.len() {
                if self.alive[p] {
                    while self.merge_child(p) {
                        done += 1;
                    }
                }
            }
            if done == 0 {
                return merges;
            }
            merges += done;
        }
    }

    /// Registers a training sequence `START -> t0 -> ... -> tn -> END`,
    /// splitting nodes so every transition runs from the last trigram of one
    /// node to the first of another over an edge created on demand.  `count`
    /// also bumps the visit counters.
    pub fn observe(&mut self, trigrams: &[Gram], count: bool) -> Result<Vec<Transition>, String> {
        if trigrams.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self.enc;
        let (stride, n) = (enc.stride, enc.n);
        let mut did_split = false;
        let mut transitions: Vec<Transition> = Vec::with_capacity(trigrams.len() + 1);

        // the grams stay borrowed: a pass walks millions of them, and a
        // packed one is only cheap to copy, not free
        let mut x = &trigrams[0];
        let (mut px, mut ox) = match self.index.get(x).copied() {
            Some(l) => (l.node, l.off),
            None => (self.create_trigram_node(x.clone()), 0),
        };
        if ox != 0 {
            let (_, b) = self.split(px, ox)?;
            px = b;
            ox = 0;
            did_split = true;
        }
        let e = match self.children[START].get(px) {
            Some(e) => e,
            None => self.new_edge(START, px, 0, 0),
        };
        transitions.push(Transition { p: START, e });
        if count {
            self.count[START].fetch_add(1, Ordering::Relaxed);
            self.count[px].fetch_add(1, Ordering::Relaxed);
            self.edge_count[e].fetch_add(1, Ordering::Relaxed);
        }
        for y in &trigrams[1..] {
            let (mut py, mut oy) = match self.index.get(y).copied() {
                Some(l) => (l.node, l.off),
                None => (self.create_trigram_node(y.clone()), 0),
            };
            if py == px && oy == ox + stride {
                ox = oy;
                x = y;
                continue;
            }
            if !enc.overlaps(x, y) {
                return Err(format!("grams {x} -> {y} do not overlap"));
            }
            if ox + n < self.label_len[px] {
                self.split(px, ox + stride)?;
                did_split = true;
                let ly = self.index[y];
                py = ly.node;
                oy = ly.off;
            }
            if oy != 0 {
                let (_, b) = self.split(py, oy)?;
                py = b;
                did_split = true;
                px = self.index[x].node; // the split moved x; only the node is read below
            }
            let e = match self.children[px].get(py) {
                Some(e) => e,
                None => self.new_edge(px, py, 0, 0),
            };
            transitions.push(Transition { p: px, e });
            if count {
                self.count[py].fetch_add(1, Ordering::Relaxed);
                self.edge_count[e].fetch_add(1, Ordering::Relaxed);
            }
            px = py;
            ox = 0;
            x = y;
        }
        if ox + n < self.label_len[px] {
            self.split(px, ox + stride)?;
            did_split = true;
        }
        let e = match self.children[px].get(END) {
            Some(e) => e,
            None => self.new_edge(px, END, 0, 0),
        };
        transitions.push(Transition { p: px, e });
        if count {
            self.count[END].fetch_add(1, Ordering::Relaxed);
            self.edge_count[e].fetch_add(1, Ordering::Relaxed);
            // every transition bumped one node counter and one edge counter, plus START's
            self.traversals.add(2 * transitions.len() as i64 + 1);
        }
        if did_split {
            let (traced, _) = self
                .trace(trigrams)
                .ok_or_else(|| "internal error: observed sequence is not walkable".to_string())?;
            transitions = traced;
        }
        Ok(transitions)
    }

    /// Walks a sequence through the structure without modifying it: the
    /// transitions and the node path `START ... END`, or `None` when a trigram
    /// is unknown, an edge is missing or a split would be needed.
    pub fn trace(&self, trigrams: &[Gram]) -> Option<(Vec<Transition>, Vec<usize>)> {
        if trigrams.is_empty() {
            return None;
        }
        let l = self.index.get(&trigrams[0])?;
        if l.off != 0 {
            return None;
        }
        let (mut px, mut ox) = (l.node, l.off);
        let e = self.children[START].get(px)?;
        let mut transitions = Vec::with_capacity(trigrams.len() + 1);
        transitions.push(Transition { p: START, e });
        let mut path = Vec::with_capacity(trigrams.len() + 2);
        path.push(START);
        path.push(px);
        for gram in &trigrams[1..] {
            let l = self.index.get(gram)?;
            let (py, oy) = (l.node, l.off);
            if py == px && oy == ox + self.enc.stride {
                ox = oy;
                continue;
            }
            if oy != 0 || ox + self.enc.n != self.label_len[px] {
                return None;
            }
            let e = self.children[px].get(py)?;
            transitions.push(Transition { p: px, e });
            path.push(py);
            px = py;
            ox = 0;
        }
        if ox + self.enc.n != self.label_len[px] {
            return None;
        }
        let e = self.children[px].get(END)?;
        transitions.push(Transition { p: px, e });
        path.push(END);
        Some((transitions, path))
    }

    /// The node path of a sequence (`START ... END`).
    pub fn node_path(&self, trigrams: &[Gram]) -> Option<Vec<usize>> {
        self.trace(trigrams).map(|(_, path)| path)
    }

    /// Verifies the structural invariants, and - with texts - that every text
    /// walks through the graph and decodes back to itself.  `compressed`
    /// additionally requires that no unary chain remains.
    pub fn check_invariants(&self, texts: &[String], compressed: bool) -> Result<(), String> {
        let enc = self.enc;
        let n = self.labels.len();
        if self.label_len.len() != n
            || self.count.len() != n
            || self.alive.len() != n
            || self.children.len() != n
            || self.parents.len() != n
        {
            return Err("node arrays have inconsistent lengths".into());
        }
        let m = self.edge_w.len();
        if self.edge_count.len() != m
            || self.edge_alive.len() != m
            || self.edge_reward.len() != m
            || self.window_edge_count.len() != m
            || self.edge_parent.len() != m
        {
            return Err("edge arrays have inconsistent lengths".into());
        }
        if n < FIRST || !self.alive[START] || !self.alive[END] || !self.alive[BACK] {
            return Err("sentinels missing or changed".into());
        }
        if self.parents[START].size() != 0 || self.children[END].size() != 0 {
            return Err("START has parents or END has children".into());
        }
        let mut alive_nodes = 0;
        let mut seen: Set<usize> = crate::hash::set();
        for p in 0..n {
            if !self.alive[p] {
                if self.children[p].size() != 0 || self.parents[p].size() != 0 {
                    return Err(format!("dead node {p} still has edges"));
                }
                continue;
            }
            alive_nodes += 1;
            if p >= FIRST && self.label_len[p] < enc.n {
                return Err(format!("node {p} label {:?} is shorter than {}", self.labels[p], enc.n));
            }
            if self.label_len[p] != enc.len(&self.labels[p]) {
                return Err(format!("node {p} has a stale label length"));
            }
            for (i, &c) in self.children[p].order.iter().enumerate() {
                let e = self.children[p].edges[i];
                if e >= m || !self.edge_alive[e] || !seen.insert(e) {
                    return Err(format!("edge {e} on {p}->{c} is invalid, dead or listed twice"));
                }
                if !self.alive[c] || c == START || p == END {
                    return Err(format!("edge {p}->{c} touches a dead node or a sentinel illegally"));
                }
                if self.parents[c].get(p) != Some(e) {
                    return Err(format!("edge {p}->{c} missing from parents"));
                }
                if self.edge_parent[e] != p {
                    return Err(format!(
                        "edge {e} records parent {}, listed under {p}",
                        self.edge_parent[e]
                    ));
                }
                if p >= FIRST && c >= FIRST && enc.overlap() > 0 {
                    let lp = &self.labels[p];
                    let tail = enc.slice(lp, enc.len(lp) - enc.overlap(), None);
                    if tail != enc.slice(&self.labels[c], 0, Some(enc.overlap())) {
                        return Err(format!("edge {p}->{c} violates the gram overlap"));
                    }
                }
            }
        }
        if alive_nodes != self.n_alive_nodes {
            return Err("alive node counter is stale".into());
        }
        let alive_edges = self.edge_alive.iter().filter(|&&ok| ok).count();
        if seen.len() != alive_edges || alive_edges != self.n_alive_edges {
            return Err("alive edge bookkeeping is stale".into());
        }
        let mut expected = 0;
        for p in FIRST..n {
            if !self.alive[p] {
                continue;
            }
            let owned = self.labels[p].clone();
            let label = enc.units(&owned);
            let mut o = 0;
            while o + enc.n <= label.len() {
                let t = enc.gram(&label, o);
                match self.index.get(&t) {
                    Some(l) if l.node == p && l.off == o => {}
                    other => return Err(format!("gram {t} of node {p}@{o} indexed as {other:?}")),
                }
                expected += 1;
                o += enc.stride;
            }
            let rest = (label.len() - enc.n) % enc.stride;
            if rest != 0 {
                return Err(format!(
                    "node {p} label {owned:?} holds {} units, {rest} past its last whole gram",
                    label.len()
                ));
            }
        }
        if self.index.len() != expected {
            return Err(format!(
                "gram index has {} entries, expected {expected}",
                self.index.len()
            ));
        }
        if compressed {
            for p in FIRST..n {
                if self.alive[p] && self.children[p].size() == 1 {
                    let c = self.children[p].order[0];
                    if c != p && c > END && self.parents[c].size() == 1 {
                        return Err(format!("unary chain {p}->{c} survived compress"));
                    }
                }
            }
        }
        for text in texts {
            let grams = enc.encode(text);
            if grams.is_empty() {
                continue;
            }
            let Some(path) = self.node_path(&grams) else {
                return Err(format!("text {text:?} is not walkable through the graph"));
            };
            let labels: Vec<&str> = path[1..path.len() - 1]
                .iter()
                .map(|&id| self.labels[id].as_str())
                .collect();
            // what comes back is what the encoding can represent: the text itself
            // under a sliding character window, its words under a word encoding,
            // everything but the ragged tail under a grouping one
            let decoded = enc.decode_path(&labels, 0, true);
            let want = enc.normalize(text);
            if decoded != want {
                return Err(format!("round trip of {want:?} gave {decoded:?}"));
            }
        }
        Ok(())
    }
}
