//! `Model` - the count / reward model: train, predict, generate, score,
//! feedback / 2NRL and the statistics.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

use crate::beam::{default_beam, BeamOptions, Prediction};
use crate::counter::Counter;
use crate::encoding::{Encoding, Gram};
use crate::graph::{Graph, GraphOptions, Loc, Transition, END, FIRST, START};
use crate::hash::Map;
use crate::mt19937::Mt19937;
use crate::parallel::{effective_workers, parallel_fill};
use crate::paths::PathOutcome;
use crate::search::{PathResult, Traversal};

/// The probability charged for a transition the structure does not know.
pub const UNKNOWN_PROB: f64 = 1e-6;
const MAX_LOG_PERPLEXITY: f64 = 700.0;

/// How many texts a pass takes at a time.
pub const DEFAULT_CHUNK_SIZE: usize = 8192;

/// The lifetime counters a model keeps beside its graph.
#[derive(Clone, Copy, Default, Debug)]
pub struct Meta {
    pub epochs_total: Counter,
    pub trained_chars: Counter,
    pub trained_texts: Counter,
    pub twonrl_runs: Counter,
    pub feedback_passes: Counter,
    pub rewards_total: f64,
    pub penalties_total: f64,
}

/// What one epoch did.
#[derive(Clone, Debug, Default)]
pub struct EpochRecord {
    pub epoch: i64,
    pub loss: f64,
    pub perplexity: f64,
    pub nodes: usize,
    pub edges: usize,
    pub trigrams: usize,
    pub compression_ratio: f64,
    pub merges: usize,
    pub transitions: i64,
    pub seconds: f64,
    pub skipped_short: usize,
    pub traversed: bool,
    pub reward: f64,
    pub phase: Option<String>,
}

/// The passes over the texts.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub auto_compress: bool,
    pub phase: Option<String>,
    /// The number of texts processed per chunk (0 = [`DEFAULT_CHUNK_SIZE`]).
    pub chunk_size: usize,
}

impl Default for TrainOptions {
    /// The Python defaults: 5 epochs, compression after every one.
    fn default() -> TrainOptions {
        TrainOptions {
            epochs: 5,
            auto_compress: true,
            phase: None,
            chunk_size: 0,
        }
    }
}

/// The count / reward model: a [`Graph`] plus its training history.
///
/// Training fans threads out over the texts of a chunk for encoding, tracing
/// and counting, with atomic increments - the Go port's `--exact` counting,
/// which is the mode the two are compared in, because a benchmark of a
/// deliberate data race measures the race.
pub struct Model {
    pub g: Graph,
    pub history: Vec<EpochRecord>,
    pub meta: Meta,
    /// The fan-out cap (0 = the machine).
    pub workers: usize,
}

impl Model {
    /// An untrained model.
    pub fn new(seed: i64, opts: GraphOptions) -> Result<Model, String> {
        let mut g = Graph::new(seed, opts)?;
        g.workers = 0;
        Ok(Model {
            g,
            history: Vec::new(),
            meta: Meta::default(),
            workers: 0,
        })
    }

    /// The model kind shared with the Python and Go implementations.
    pub fn kind(&self) -> &'static str {
        "count"
    }

    fn workers(&self) -> usize {
        self.workers
    }

    // -- training -----------------------------------------------------------

    /// Counts one traversal of every text's path per epoch.
    pub fn train(&mut self, texts: &[String], opts: &TrainOptions) -> Result<Vec<EpochRecord>, String> {
        self.passes(texts, opts, true, 0.0)
    }

    /// Thumbs up: `epochs` passes that traverse and reward (+`strength`) every path.
    pub fn reward(&mut self, texts: &[String], epochs: usize, strength: f64) -> Result<Vec<EpochRecord>, String> {
        let opts = TrainOptions {
            epochs,
            phase: Some("positive".into()),
            ..Default::default()
        };
        self.passes(texts, &opts, true, strength.abs())
    }

    /// Thumbs down: `epochs` passes that penalise (-`strength`) every path; no
    /// traversal is counted.
    pub fn punish(&mut self, texts: &[String], epochs: usize, strength: f64) -> Result<Vec<EpochRecord>, String> {
        let opts = TrainOptions {
            epochs,
            phase: Some("negative".into()),
            ..Default::default()
        };
        self.passes(texts, &opts, false, -strength.abs())
    }

    /// 2NRL: penalise the bad texts, then count and reward the good ones.
    pub fn two_nrl(
        &mut self,
        bad: &[String],
        good: &[String],
        neg_epochs: usize,
        pos_epochs: usize,
        strength: f64,
    ) -> Result<(Vec<EpochRecord>, Vec<EpochRecord>), String> {
        let negative = self.punish(bad, neg_epochs, strength)?;
        let positive = self.reward(good, pos_epochs, strength)?;
        self.meta.twonrl_runs.add(1);
        Ok((negative, positive))
    }

    /// Flips the sign of every reward.
    pub fn invert(&mut self) {
        self.g.invert();
    }

    fn passes(
        &mut self,
        texts: &[String],
        opts: &TrainOptions,
        count: bool,
        reward: f64,
    ) -> Result<Vec<EpochRecord>, String> {
        // a rewarded path was judged correct, a penalised one wrong, a plain
        // training pass neither
        let outcome = if reward > 0.0 {
            PathOutcome::Correct
        } else if reward < 0.0 {
            PathOutcome::Incorrect
        } else {
            PathOutcome::Unjudged
        };
        let chunk_size = if opts.chunk_size == 0 {
            DEFAULT_CHUNK_SIZE
        } else {
            opts.chunk_size
        };
        let workers = self.workers();
        let mut skipped_short = 0;
        let usable: Vec<&String> = texts
            .iter()
            .filter(|t| {
                let ok = self.g.enc.len(t) >= self.g.enc.n;
                if !ok {
                    skipped_short += 1;
                }
                ok
            })
            .collect();

        // build the structure first (no counting) and compress it, so every
        // pass - the first included - walks the same transitions
        let mut chars = 0i64;
        for chunk in usable.chunks(chunk_size) {
            let grams = self.encode_all(chunk);
            let mut novel = vec![false; chunk.len()];
            {
                let g = &self.g;
                parallel_fill(&mut novel, workers, |i, slot| *slot = g.trace(&grams[i]).is_none());
            }
            for (i, &is_novel) in novel.iter().enumerate() {
                chars += self.g.enc.len(chunk[i]) as i64;
                if is_novel {
                    self.g.observe(&grams[i], false)?;
                }
            }
        }
        if count {
            self.meta.trained_texts.add(usable.len() as i64);
            self.meta.trained_chars.add(chars);
        }
        let mut pending_merges = if opts.auto_compress { self.g.compress() } else { 0 };

        let mut records = Vec::with_capacity(opts.epochs);
        for _ in 0..opts.epochs {
            let started = Instant::now();
            let mut traversed: Vec<AtomicI64> = (0..self.g.num_edge_ids()).map(|_| AtomicI64::new(0)).collect();
            let mut extra: Map<usize, i64> = crate::hash::map();
            let mut total = 0i64;

            for chunk in usable.chunks(chunk_size) {
                let grams = self.encode_all(chunk);
                let mut traced: Vec<Option<(Vec<Transition>, Vec<usize>)>> = vec![None; chunk.len()];
                {
                    let g = &self.g;
                    let traversed = &traversed;
                    parallel_fill(&mut traced, workers, |i, slot| {
                        let Some((tr, path)) = g.trace(&grams[i]) else { return };
                        for t in &tr {
                            traversed[t.e].fetch_add(1, Ordering::Relaxed);
                        }
                        if count {
                            for t in &tr {
                                g.bump_edge(t.e);
                            }
                            for &n in &path {
                                g.bump_node(n);
                            }
                        }
                        *slot = Some((tr, path));
                    });
                }
                // a text that needs a split after all: observe it, trace and count it here
                for i in 0..chunk.len() {
                    if traced[i].is_some() {
                        continue;
                    }
                    self.g.observe(&grams[i], false)?;
                    let Some((tr, path)) = self.g.trace(&grams[i]) else {
                        continue;
                    };
                    for t in &tr {
                        if t.e < traversed.len() {
                            traversed[t.e].fetch_add(1, Ordering::Relaxed);
                        } else {
                            *extra.entry(t.e).or_insert(0) += 1;
                        }
                        if count {
                            self.g.bump_edge(t.e);
                        }
                    }
                    if count {
                        for &n in &path {
                            self.g.bump_node(n);
                        }
                    }
                    traced[i] = Some((tr, path));
                }

                let size: usize = traced.iter().flatten().map(|(tr, _)| tr.len()).sum();
                let mut edges = Vec::with_capacity(size);
                let mut node_bumps = 0usize;
                for (tr, path) in traced.iter().flatten() {
                    edges.extend(tr.iter().map(|t| t.e));
                    node_bumps += path.len();
                }
                total += edges.len() as i64;
                if count {
                    // every transition bumped one edge counter and every node of
                    // a path one node counter: the increments bound each counter
                    self.g.add_traversals((edges.len() + node_bumps) as i64);
                    self.g.record_traversals(&edges); // the sliding window follows the corpus order
                }
                for (tr, _) in traced.iter().flatten() {
                    // and what each text did, in its own context
                    let tr = tr.clone();
                    self.g.record_path(&tr, outcome, outcome != PathOutcome::Unjudged);
                }
                if reward != 0.0 {
                    self.g.add_reward(&edges, reward);
                }
            }

            if reward != 0.0 {
                self.meta.feedback_passes.add(1);
                if reward > 0.0 {
                    self.meta.rewards_total += reward * total as f64;
                } else {
                    self.meta.penalties_total += -reward * total as f64;
                }
            }
            self.g.prepare();
            let loss = self.weighted_cost(&mut traversed, &extra, total);
            let mut merges = pending_merges;
            pending_merges = 0;
            if opts.auto_compress {
                merges += self.g.compress();
            }
            self.g.carry_counters(false); // the epoch is over: wrap whatever reached the limit
            self.meta.epochs_total.add(1);
            let record = EpochRecord {
                epoch: self.meta.epochs_total.value,
                loss,
                perplexity: loss.min(MAX_LOG_PERPLEXITY).exp(),
                nodes: self.g.num_nodes(),
                edges: self.g.num_edges(),
                trigrams: self.g.num_trigrams(),
                compression_ratio: self.g.compression_ratio(),
                merges,
                transitions: total,
                seconds: started.elapsed().as_secs_f64(),
                skipped_short,
                traversed: count,
                reward,
                phase: opts.phase.clone(),
            };
            self.history.push(record.clone());
            records.push(record);
        }
        Ok(records)
    }

    /// Encodes a chunk of texts on the worker threads.
    fn encode_all(&self, texts: &[&String]) -> Vec<Vec<Gram>> {
        let enc = self.g.enc;
        let mut grams = vec![Vec::new(); texts.len()];
        parallel_fill(&mut grams, self.workers(), |i, slot| *slot = enc.encode(texts[i]));
        grams
    }

    /// How this model turns text into grams and back - its graph's encoding.
    pub fn encoding(&self) -> Encoding {
        self.g.enc
    }

    /// The mean cost of the pass's traversals: every edge's cost times how
    /// often the pass traversed it.
    fn weighted_cost(&self, traversed: &mut [AtomicI64], extra: &Map<usize, i64>, total: i64) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for (e, &c) in extra {
            sum += c as f64 * self.g.edge_cost(*e);
        }
        for (e, slot) in traversed.iter_mut().enumerate() {
            let c = *slot.get_mut();
            if c != 0 {
                sum += c as f64 * self.g.edge_cost(e);
            }
        }
        sum / total as f64
    }

    // -- prediction ---------------------------------------------------------

    /// Where `prefix` ends in the graph: `(node, offset, matched units of the
    /// located gram)`.
    fn locate(&self, prefix: &str) -> (usize, usize, usize) {
        let enc = self.g.enc;
        let n = enc.len(prefix);
        if n == 0 {
            return (START, 0, 0);
        }
        if n >= enc.n {
            // the gram the prefix ends on, then - when the stride skips it -
            // the last gram of the prefix's own grid
            let units = enc.units(prefix);
            let mut found = enc
                .gram_of(units.slice(n - enc.n, Some(n)))
                .and_then(|g| self.g.lookup(&g));
            if found.is_none() {
                let aligned = (n - enc.n) / enc.stride * enc.stride;
                if aligned != n - enc.n {
                    found = enc
                        .gram_of(units.slice(aligned, Some(aligned + enc.n)))
                        .and_then(|g| self.g.lookup(&g));
                }
            }
            if let Some(l) = found {
                return (l.node, l.off, enc.n);
            }
            for k in [enc.n.saturating_sub(1), 1] {
                if k < 1 || k >= enc.n {
                    continue;
                }
                if let Some(l) = self.best_trigram(units.slice(n - k, Some(n))) {
                    return (l.node, l.off, k);
                }
            }
            return (START, 0, 0);
        }
        if let Some(node) = self.best_node_with_prefix(prefix) {
            return (node, 0, n);
        }
        (START, 0, 0)
    }

    /// The most visited `(node, offset)` holding a gram that starts with `key`
    /// (whole units: `"the ca"` is not a prefix of `"the cat sat"` in words).
    fn best_trigram(&self, key: &str) -> Option<Loc> {
        let enc = self.g.enc;
        let mut best: Option<(i64, usize, usize, Loc)> = None;
        for (t, l) in self.g.index_entries() {
            if !enc.has_unit_prefix(&t.to_string(), key) {
                continue;
            }
            let rank = (-(self.g.node_count(l.node).float() as i64), l.node, l.off);
            let better = match best {
                None => true,
                Some((c, node, off, _)) => rank < (c, node, off),
            };
            if better {
                best = Some((rank.0, rank.1, rank.2, l));
            }
        }
        best.map(|(_, _, _, l)| l)
    }

    /// The most visited real node whose label starts with `prefix`.
    fn best_node_with_prefix(&self, prefix: &str) -> Option<usize> {
        let enc = self.g.enc;
        let mut best: Option<usize> = None;
        let mut best_count = Counter { value: -1, resets: 0 };
        for node in (START + 2)..self.g.num_node_ids() {
            if !self.g.is_alive(node) || !enc.has_unit_prefix(self.g.label(node), prefix) {
                continue;
            }
            let c = self.g.node_count(node);
            if best_count.less(c) {
                best = Some(node);
                best_count = c;
            }
        }
        best
    }

    /// `(node, offset, lead)`: where the prefix ends and the unmatched
    /// remainder of the located trigram, which every predicted path starts with.
    fn prefix_start(&self, prefix: &str) -> (usize, usize, String) {
        let (node, offset, matched) = self.locate(prefix);
        let enc = self.g.enc;
        let lead = if node != START && matched < enc.n {
            enc.slice(self.g.label(node), offset + matched, Some(offset + enc.n))
        } else {
            String::new()
        };
        (node, offset, lead)
    }

    /// Continues `prefix`: the K most likely and the K least likely
    /// continuations in one beam search, or one stochastic walk.
    pub fn predict(&mut self, prefix: &str, o: &PredictOptions) -> Result<Prediction, String> {
        let mode = match o.mode.as_str() {
            "" | "dijkstra" | "beam" => "beam",
            "sample" => "sample",
            other => {
                return Err(format!(
                    "unknown mode {other:?}; expected 'beam', 'dijkstra' or 'sample'"
                ))
            }
        };
        if o.step_penalty < 0.0 {
            return Err("step_penalty must be >= 0".to_string());
        }
        if o.temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        self.search(
            prefix,
            o.length,
            mode,
            o.k,
            o.beam,
            o.step_penalty,
            o.temperature,
            o.to_end,
            o.max_length,
            None,
            o.traversal,
        )
    }

    /// The prediction engine shared by [`Model::predict`] and [`Model::generate`].
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        prefix: &str,
        length: usize,
        mode: &str,
        k: usize,
        beam: usize,
        step_penalty: f64,
        temperature: f64,
        to_end: bool,
        max_length: Option<usize>,
        rng: Option<&mut Mt19937>,
        traversal: Traversal,
    ) -> Result<Prediction, String> {
        let enc = self.g.enc;
        let (node, offset, lead) = self.prefix_start(prefix);
        let lead_len = enc.len(&lead);
        let want = length.saturating_sub(lead_len);
        let mut cap: Option<usize> = None;
        let (top, bottom, expanded, width);
        if mode == "beam" {
            let mut max_chars = None;
            if max_length.is_none() && length == 0 {
                cap = Some(0);
                max_chars = Some(0);
            } else if let Some(max_length) = max_length {
                let c = length.max(max_length);
                cap = Some(c);
                max_chars = Some(c.saturating_sub(lead_len).max(want));
            }
            let opts = BeamOptions {
                k,
                beam,
                max_chars,
                step_penalty,
                to_end,
                traversal,
                ..Default::default()
            };
            let (t, b, e) = self.g.beam_predict(node, offset, want, opts)?;
            top = t;
            bottom = b;
            expanded = e;
            width = if beam == 0 { default_beam(k) } else { beam };
        } else {
            cap = max_length.or(Some(length));
            let max_chars = cap.map(|c| c.saturating_sub(lead_len));
            let walk = self
                .g
                .sample_walk(node, offset, max_chars, temperature, rng, None, traversal)?;
            expanded = walk.expanded;
            top = vec![walk];
            bottom = Vec::new();
            width = 0;
        }
        let fix = |r: &mut PathResult| {
            if !lead.is_empty() {
                r.text = enc.join(&[&lead, &r.text]);
                if let Some(cap) = cap {
                    r.text = enc.truncate(&r.text, cap);
                }
            }
            r.full_text = enc.join(&[prefix, &r.text]);
        };
        let mut top = top;
        let mut bottom = bottom;
        top.iter_mut().for_each(&fix);
        bottom.iter_mut().for_each(&fix);
        let best = match top.first() {
            Some(best) => best.clone(),
            None => {
                let text = match cap {
                    Some(cap) => enc.truncate(&lead, cap),
                    None => lead.clone(),
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
        Ok(Prediction {
            best,
            top,
            bottom,
            k,
            beam: width,
            mode: mode.to_string(),
            traversal,
            expanded,
        })
    }

    /// Generates whole texts with the prediction search, from `START` or
    /// continuing a prefix.
    pub fn generate(&mut self, o: &GenerateOptions) -> Result<Vec<PathResult>, String> {
        let mode = if o.mode.is_empty() { "sample" } else { o.mode.as_str() };
        if !matches!(mode, "beam" | "dijkstra" | "sample") {
            return Err(format!(
                "unknown mode {mode:?}; expected 'beam', 'dijkstra' or 'sample'"
            ));
        }
        if o.count == 0 {
            return Ok(Vec::new());
        }
        let whole = |r: &mut PathResult, prefix: &str| {
            r.full_text = format!("{prefix}{}", r.text);
            r.text = r.full_text.clone();
        };
        if mode == "sample" {
            let mut rng = o.seed.map(Mt19937::new);
            let mut results = Vec::with_capacity(o.count);
            for _ in 0..o.count {
                let mut found = self.search(
                    &o.prefix,
                    o.max_length,
                    "sample",
                    0,
                    0,
                    0.0,
                    o.temperature,
                    false,
                    Some(o.max_length),
                    rng.as_mut(),
                    o.traversal,
                )?;
                whole(&mut found.best, &o.prefix);
                results.push(found.best);
            }
            return Ok(results);
        }
        let k = if mode == "dijkstra" { 1 } else { o.count };
        let found = self.search(
            &o.prefix,
            0,
            "beam",
            k,
            o.beam,
            o.step_penalty,
            1.0,
            true,
            Some(o.max_length),
            None,
            o.traversal,
        )?;
        let mut results = found.top;
        results.iter_mut().for_each(|r| whole(r, &o.prefix));
        if mode == "dijkstra" {
            results.truncate(1);
        }
        Ok(results)
    }

    // -- scoring ------------------------------------------------------------

    /// `log P(c | p)` for the edge taken from `p`'s trigram at `offset`.
    fn edge_log_prob(&self, p: usize, offset: usize, c: usize) -> Option<f64> {
        if p != START && offset + self.g.enc.n != self.g.label_len(p) {
            return None;
        }
        self.g.edge(p, c).map(|e| -self.g.edge_cost(e))
    }

    /// The log-probability of a text: unknown trigrams, missing edges and
    /// transitions that would need a split cost `log(UNKNOWN_PROB)`.
    pub fn score(&mut self, text: &str) -> Score {
        self.g.prepare();
        let enc = self.g.enc;
        let grams = enc.encode(text);
        let chars = enc.len(text);
        if grams.is_empty() {
            return Score {
                chars,
                ..Default::default()
            };
        }
        let log_unknown = UNKNOWN_PROB.ln();
        let mut log_prob = 0.0;
        let (mut transitions, mut unknown) = (0usize, 0usize);
        let (mut node, mut offset) = (START, 0usize);
        let mut lost = false;
        for t in &grams {
            let Some(l) = self.g.lookup(t) else {
                transitions += 1;
                unknown += 1;
                log_prob += log_unknown;
                lost = true;
                continue;
            };
            if !lost && l.node == node && l.off == offset + enc.stride {
                offset = l.off;
                continue;
            }
            transitions += 1;
            let known = if !lost && l.off == 0 {
                self.edge_log_prob(node, offset, l.node)
            } else {
                None
            };
            match known {
                Some(lp) => log_prob += lp,
                None => {
                    unknown += 1;
                    log_prob += log_unknown;
                }
            }
            node = l.node;
            offset = l.off;
            lost = false;
        }
        transitions += 1;
        let known = if !lost {
            self.edge_log_prob(node, offset, END)
        } else {
            None
        };
        match known {
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

    /// How the counters are bumped; always exact here.
    pub fn counting(&self) -> &'static str {
        "exact"
    }

    /// The thread pool as the statistics report it.
    pub fn device_label(&self) -> String {
        format!("cpu x{}", effective_workers(self.workers()))
    }
}

/// The log-probability of a text under the model.
#[derive(Clone, Copy, Default, Debug)]
pub struct Score {
    pub log_prob: f64,
    pub per_char: f64,
    pub chars: usize,
    pub transitions: usize,
    pub unknown_transitions: usize,
}

/// The settings of one prediction.
#[derive(Clone, Debug)]
pub struct PredictOptions {
    pub length: usize,
    pub mode: String,
    pub k: usize,
    pub beam: usize,
    pub step_penalty: f64,
    pub temperature: f64,
    pub to_end: bool,
    /// `None` = no cap
    pub max_length: Option<usize>,
    pub traversal: Traversal,
}

impl Default for PredictOptions {
    /// The Python defaults.
    fn default() -> PredictOptions {
        PredictOptions {
            length: 20,
            mode: "beam".into(),
            k: 5,
            beam: 0,
            step_penalty: 0.0,
            temperature: 1.0,
            to_end: false,
            max_length: None,
            traversal: Traversal::Reward,
        }
    }
}

/// The settings of one generation.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_length: usize,
    pub mode: String,
    pub temperature: f64,
    pub count: usize,
    pub seed: Option<i64>,
    pub prefix: String,
    pub step_penalty: f64,
    pub beam: usize,
    pub traversal: Traversal,
}

impl Default for GenerateOptions {
    fn default() -> GenerateOptions {
        GenerateOptions {
            max_length: 60,
            mode: "sample".into(),
            temperature: 1.0,
            count: 1,
            seed: None,
            prefix: String::new(),
            step_penalty: 0.0,
            beam: 0,
            traversal: Traversal::Reward,
        }
    }
}

/// Everything the model can say about itself.
pub fn stats(model: &Model) -> Vec<(String, String)> {
    let g = &model.g;
    let (pos, neg) = g.total_reward();
    let totals = g.path_totals();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut put = |k: &str, v: String| out.push((k.to_string(), v));
    put("kind", model.kind().to_string());
    put("backend", "rust".to_string());
    put("device", model.device_label());
    put("counting", model.counting().to_string());
    put("nodes", g.num_nodes().to_string());
    put("edges", g.num_edges().to_string());
    put("trigrams", g.num_trigrams().to_string());
    put("compression_ratio", format!("{:.6}", g.compression_ratio()));
    put("inverted", g.inverted.to_string());
    put("history_len", model.history.len().to_string());
    put("rewards_total", format!("{}", model.meta.rewards_total));
    put("penalties_total", format!("{}", model.meta.penalties_total));
    put("path_contexts", totals.contexts.to_string());
    put("path_judged", totals.judged.to_string());
    put("edge_reward_positive", format!("{pos}"));
    put("edge_reward_negative", format!("{neg}"));
    put("epochs_total", model.meta.epochs_total.value.to_string());
    put("trained_chars", model.meta.trained_chars.value.to_string());
    put("trained_texts", model.meta.trained_texts.value.to_string());
    put("total_traversals", g.total_traversals().value.to_string());
    put("window_traversals", g.window_traversals().to_string());
    out
}

/// The node ids the model treats as real (not sentinels).
pub const FIRST_REAL_NODE: usize = FIRST;
