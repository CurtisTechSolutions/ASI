//! `Model` - the count / reward model: train, predict, generate, score,
//! feedback / 2NRL and the statistics.
//!
//! The other kinds hang what they keep beside the graph on it - the negative
//! network on `neg`, the phase model on `res`, the sine model on the graph
//! alone - and the entry points below hand a call to their own code
//! ([`crate::radix`], [`crate::resonance`]) when the kind has its own answer.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

use crate::beam::{default_beam, BeamOptions, Prediction};
use crate::counter::Counter;
use crate::encoding::Encoding;
use crate::graph::{Graph, GraphOptions, Loc, Transition, END, FIRST, START};
use crate::hash::Map;
use crate::json::Json;
use crate::mt19937::Mt19937;
use crate::parallel::{effective_workers, parallel_fill};
use crate::paths::PathOutcome;
use crate::search::{parse_traversal, PathResult};

/// The probability charged for a transition the structure does not know.
pub const UNKNOWN_PROB: f64 = 1e-6;
const MAX_LOG_PERPLEXITY: f64 = 700.0;

/// How many texts a pass takes at a time.
pub const DEFAULT_CHUNK_SIZE: usize = 8192;

/// The lifetime counters a model keeps beside its graph - the `meta` block of a
/// model file, in the order Python writes it.
///
/// `extra` is whatever else the file carried.  Another implementation may keep
/// a counter this one knows nothing about, and a model that loses it on a round
/// trip through here is not interchangeable with anything.
#[derive(Clone, Default, Debug)]
pub struct Meta {
    pub created: String,
    pub seed: i64,
    pub epochs_total: Counter,
    pub trained_chars: Counter,
    pub trained_texts: Counter,
    pub twonrl_runs: Counter,
    pub rewards_total: f64,
    pub penalties_total: f64,
    pub feedback_passes: Counter,
    pub extra: Vec<(String, Json)>,
}

/// The `meta` entries that are an odometer plus a reset count.
const META_COUNTERS: [&str; 5] = [
    "epochs_total",
    "trained_chars",
    "trained_texts",
    "twonrl_runs",
    "feedback_passes",
];

impl Meta {
    /// A new model's metadata.
    pub fn new(seed: i64) -> Meta {
        Meta {
            created: crate::clock::utc_now(),
            seed,
            ..Default::default()
        }
    }

    fn counter(&self, key: &str) -> Counter {
        match key {
            "epochs_total" => self.epochs_total,
            "trained_chars" => self.trained_chars,
            "trained_texts" => self.trained_texts,
            "twonrl_runs" => self.twonrl_runs,
            _ => self.feedback_passes,
        }
    }

    fn set_counter(&mut self, key: &str, value: Counter) {
        match key {
            "epochs_total" => self.epochs_total = value,
            "trained_chars" => self.trained_chars = value,
            "trained_texts" => self.trained_texts = value,
            "twonrl_runs" => self.twonrl_runs = value,
            "feedback_passes" => self.feedback_passes = value,
            _ => {}
        }
    }

    /// The `meta` block, in Python's key order.
    pub fn to_json(&self) -> Json {
        let mut pairs: Vec<(String, Json)> = vec![
            ("created".to_string(), Json::str(self.created.clone())),
            ("seed".to_string(), Json::Int(self.seed)),
        ];
        for key in ["epochs_total", "trained_chars", "trained_texts", "twonrl_runs"] {
            let c = self.counter(key);
            pairs.push((key.to_string(), Json::Int(c.value)));
            pairs.push((format!("{key}_resets"), Json::Int(c.resets)));
        }
        pairs.push(("rewards_total".to_string(), Json::Num(self.rewards_total)));
        pairs.push(("penalties_total".to_string(), Json::Num(self.penalties_total)));
        pairs.push(("feedback_passes".to_string(), Json::Int(self.feedback_passes.value)));
        pairs.push((
            "feedback_passes_resets".to_string(),
            Json::Int(self.feedback_passes.resets),
        ));
        pairs.extend(self.extra.iter().cloned());
        Json::Obj(pairs)
    }

    /// Reads a file's `meta` over this one, wrapping every counter it carried (a
    /// file written by hand can hold a reading past the limit) and keeping every
    /// key this implementation does not know about.
    pub fn merge_json(&mut self, doc: &Json) {
        let Json::Obj(pairs) = doc else { return };
        for (key, value) in pairs {
            match key.as_str() {
                "created" => self.created = value.as_str().unwrap_or(&self.created).to_string(),
                "seed" => self.seed = value.as_i64().unwrap_or(self.seed),
                "rewards_total" => self.rewards_total = value.as_f64().unwrap_or(0.0),
                "penalties_total" => self.penalties_total = value.as_f64().unwrap_or(0.0),
                _ => {
                    let base = key.strip_suffix("_resets").unwrap_or(key);
                    if META_COUNTERS.contains(&base) {
                        let value = doc.at(base).as_i64().unwrap_or(0);
                        let resets = doc.at(&format!("{base}_resets")).as_i64().unwrap_or(0);
                        self.set_counter(base, Counter::new(value, resets));
                    } else if !self.extra.iter().any(|(k, _)| k == key) {
                        self.extra.push((key.clone(), value.clone()));
                    }
                }
            }
        }
    }
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
    /// Whatever else the record carried when it was read from a file.
    pub extra: Vec<(String, Json)>,
}

impl EpochRecord {
    /// Whether this is a negative network's blame or clearing pass.
    pub fn is_negative_pass(&self) -> bool {
        matches!(self.phase.as_deref(), Some("negative") | Some("clear"))
            && !self.extra.is_empty()
            && self.extra.iter().any(|(k, _)| k == "edges_touched")
    }

    /// A field the record carries beyond the common ones.
    pub fn extra_value(&self, key: &str) -> Option<&Json> {
        self.extra.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The layout of this record's `history` entry: the phase model writes
    /// its walk counts between `transitions` and `seconds`, the sine model its
    /// learning rates after `skipped_short`, and every other kind the count
    /// model's `traversed` / `reward`.  Told apart by the keys only that kind
    /// writes.
    fn layout(&self) -> &'static str {
        if matches!(self.extra_value("traversed"), Some(Json::Int(_))) && self.extra_value("cycles").is_some() {
            "resonant"
        } else if self.extra_value("lr").is_some() && self.extra_value("act_lr").is_some() && !self.is_negative_pass() {
            "radix"
        } else {
            "count"
        }
    }

    /// The record as a `history` entry.
    pub fn to_json(&self) -> Json {
        match self.layout() {
            "resonant" => return self.keyed_json(&["traversed", "reward", "cycles", "signatures"], false),
            "radix" => return self.keyed_json(&["lr", "act_lr"], true),
            _ => {}
        }
        let mut pairs: Vec<(String, Json)> = vec![
            ("epoch".to_string(), Json::Int(self.epoch)),
            ("loss".to_string(), Json::Num(self.loss)),
            ("perplexity".to_string(), Json::Num(self.perplexity)),
            ("nodes".to_string(), Json::Int(self.nodes as i64)),
            ("edges".to_string(), Json::Int(self.edges as i64)),
            ("trigrams".to_string(), Json::Int(self.trigrams as i64)),
            ("compression_ratio".to_string(), Json::Num(self.compression_ratio)),
            ("merges".to_string(), Json::Int(self.merges as i64)),
            ("transitions".to_string(), Json::Int(self.transitions)),
            ("seconds".to_string(), Json::Num(self.seconds)),
            ("skipped_short".to_string(), Json::Int(self.skipped_short as i64)),
        ];
        // a negative network's passes neither traverse nor reward: their
        // records carry what was blamed or cleared instead (Python's `_pass`)
        if !self.is_negative_pass() {
            pairs.push(("traversed".to_string(), Json::Bool(self.traversed)));
            pairs.push(("reward".to_string(), Json::Num(self.reward)));
        }
        if let Some(phase) = &self.phase {
            pairs.push(("phase".to_string(), Json::str(phase.clone())));
        }
        pairs.extend(self.extra.iter().cloned());
        Json::Obj(pairs)
    }

    /// A radix or resonant entry, in Python's key order: the common fields to
    /// `transitions`, then `own` (read from the extras, `reward` from the
    /// field) either after `skipped_short` (`own_last`) or before `seconds`,
    /// then the phase and whatever else the record carries (`weight`).
    fn keyed_json(&self, own: &[&str], own_last: bool) -> Json {
        let mut pairs: Vec<(String, Json)> = vec![
            ("epoch".to_string(), Json::Int(self.epoch)),
            ("loss".to_string(), Json::Num(self.loss)),
            ("perplexity".to_string(), Json::Num(self.perplexity)),
            ("nodes".to_string(), Json::Int(self.nodes as i64)),
            ("edges".to_string(), Json::Int(self.edges as i64)),
            ("trigrams".to_string(), Json::Int(self.trigrams as i64)),
            ("compression_ratio".to_string(), Json::Num(self.compression_ratio)),
            ("merges".to_string(), Json::Int(self.merges as i64)),
            ("transitions".to_string(), Json::Int(self.transitions)),
        ];
        let own_pairs: Vec<(String, Json)> = own
            .iter()
            .map(|&key| {
                let value = if key == "reward" {
                    Json::Num(self.reward)
                } else {
                    self.extra_value(key).cloned().unwrap_or(Json::Null)
                };
                (key.to_string(), value)
            })
            .collect();
        if !own_last {
            pairs.extend(own_pairs.iter().cloned());
        }
        pairs.push(("seconds".to_string(), Json::Num(self.seconds)));
        pairs.push(("skipped_short".to_string(), Json::Int(self.skipped_short as i64)));
        if own_last {
            pairs.extend(own_pairs);
        }
        if let Some(phase) = &self.phase {
            pairs.push(("phase".to_string(), Json::str(phase.clone())));
        }
        pairs.extend(self.extra.iter().filter(|(k, _)| !own.contains(&k.as_str())).cloned());
        Json::Obj(pairs)
    }

    /// One `history` entry, keeping any field this implementation does not write.
    pub fn from_json(doc: &Json) -> EpochRecord {
        const KNOWN: [&str; 14] = [
            "epoch",
            "loss",
            "perplexity",
            "nodes",
            "edges",
            "trigrams",
            "compression_ratio",
            "merges",
            "transitions",
            "seconds",
            "skipped_short",
            "traversed",
            "reward",
            "phase",
        ];
        let extra = match doc {
            Json::Obj(pairs) => pairs
                .iter()
                // the phase model's `traversed` is a count, not the flag: kept as it is
                .filter(|(k, v)| !KNOWN.contains(&k.as_str()) || (k == "traversed" && !matches!(v, Json::Bool(_))))
                .cloned()
                .collect(),
            _ => Vec::new(),
        };
        EpochRecord {
            epoch: doc.at("epoch").as_i64().unwrap_or(0),
            loss: doc.at("loss").as_f64().unwrap_or(0.0),
            perplexity: doc.at("perplexity").as_f64().unwrap_or(0.0),
            nodes: doc.at("nodes").as_i64().unwrap_or(0) as usize,
            edges: doc.at("edges").as_i64().unwrap_or(0) as usize,
            trigrams: doc.at("trigrams").as_i64().unwrap_or(0) as usize,
            compression_ratio: doc.at("compression_ratio").as_f64().unwrap_or(0.0),
            merges: doc.at("merges").as_i64().unwrap_or(0) as usize,
            transitions: doc.at("transitions").as_i64().unwrap_or(0),
            seconds: doc.at("seconds").as_f64().unwrap_or(0.0),
            skipped_short: doc.at("skipped_short").as_i64().unwrap_or(0) as usize,
            traversed: doc.at("traversed").as_bool().unwrap_or(true),
            reward: doc.at("reward").as_f64().unwrap_or(0.0),
            phase: doc.at("phase").as_str().map(str::to_string),
            extra,
        }
    }
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

/// What the model's own lines are filed under.
const LOG: &str = "train";

/// The count / reward model: a [`Graph`] plus its training history.
///
/// Training fans threads out over the texts of a chunk for encoding, tracing
/// and counting, with atomic increments - the Go port's `--exact` counting,
/// which is the mode the two are compared in, because a benchmark of a
/// deliberate data race measures the race.
pub struct Model {
    /// The journal and judgement settings of the *negative* network
    /// ([`crate::negative`]); `None` on a count / reward model.
    pub neg: Option<Box<crate::negative::Negative>>,
    /// The metacognitive layer and cycle settings of the *phase* model
    /// ([`crate::resonance`]); `None` on every other kind.
    pub res: Option<Box<crate::resonance::ResonantModel>>,
    pub g: Graph,
    pub history: Vec<EpochRecord>,
    pub meta: Meta,
    /// The fan-out cap (0 = the machine).
    pub workers: usize,
}

impl Model {
    /// An untrained model.
    pub fn new(seed: i64, opts: GraphOptions) -> Result<Model, String> {
        Ok(Model::from_graph(Graph::new(seed, opts)?))
    }

    /// A model around a graph that already exists - what the file reader builds.
    pub fn from_graph(g: Graph) -> Model {
        let seed = g.seed;
        Model {
            g,
            history: Vec::new(),
            meta: Meta::new(seed),
            workers: 0,
            neg: None,
            res: None,
        }
    }

    /// The model kind shared with the Python and Go implementations.
    ///
    /// Words are not a kind: they are an encoding, so a word model is this
    /// model with `unit = word` and nothing else changed.  The negative network
    /// *is* a kind - it keeps blame where this keeps counts and rewards.
    pub fn kind(&self) -> &'static str {
        if self.is_negative() {
            "negative"
        } else if self.g.is_radix() {
            "radix"
        } else if self.g.is_resonant() {
            "resonant"
        } else {
            "count"
        }
    }

    /// The file format: the encoding never changes it, but the kind does.
    pub fn format(&self) -> &'static str {
        if self.is_negative() {
            crate::negative::NEGATIVE_FORMAT
        } else if self.g.is_radix() {
            crate::radix::RADIX_FORMAT
        } else if self.g.is_resonant() {
            crate::resonance::RESONANT_FORMAT
        } else {
            crate::file::MODEL_FORMAT
        }
    }

    /// How this model reads a text and writes one back.
    pub fn encoding(&self) -> Encoding {
        self.g.enc
    }

    /// What the model counts in: lengths, caps and per-unit scores are in
    /// these (`"words"` under a word encoding, else `"chars"`).
    pub fn units(&self) -> &'static str {
        self.g.units()
    }

    /// `text` as this model can represent it - what a round trip returns.
    ///
    /// The default encoding returns it unchanged; a word encoding writes single
    /// spaces, and a grouping one drops the tail that fills no group.
    pub fn normalise(&self, text: &str) -> String {
        self.g.enc.normalize(text)
    }

    /// The alphabet of a word encoding: every word the graph's grams are made
    /// of, most read first, ties alphabetically.  `limit` 0 is all of them.
    ///
    /// A model that does not count in words has none.  The order is the Python
    /// and Go one, word for word, so three implementations list one alphabet.
    pub fn top_words(&mut self, limit: usize) -> Vec<crate::words::WordRow> {
        self.g.prepare();
        if self.g.enc.unit != crate::encoding::Unit::Words {
            return Vec::new();
        }
        let grams = self.g.gram_index();
        let counted = crate::encoding::vocabulary(&self.g.enc, grams.iter().copied());
        let mut rows: Vec<crate::words::WordRow> = counted
            .into_iter()
            .enumerate()
            .map(|(id, (word, grams))| crate::words::WordRow { word, id, grams })
            .collect();
        if limit > 0 {
            rows.truncate(limit);
        }
        rows
    }

    fn workers(&self) -> usize {
        self.workers
    }

    // -- training -----------------------------------------------------------

    /// Counts one traversal of every text's path per epoch.
    pub fn train(&mut self, texts: &[String], opts: &TrainOptions) -> Result<Vec<EpochRecord>, String> {
        if self.g.is_radix() {
            let cfg = crate::radix::TrainConfig {
                epochs: opts.epochs,
                auto_compress: opts.auto_compress,
                ..Default::default()
            };
            return self.radix_train(texts, &cfg, opts.phase.as_deref(), &mut |_| true);
        }
        if self.is_resonant() {
            return self.resonant_train(
                texts,
                opts.epochs,
                opts.auto_compress,
                opts.phase.as_deref(),
                &mut |_| true,
            );
        }
        self.passes(texts, opts, true, 0.0)
    }

    /// Thumbs up: `epochs` passes that traverse and reward (+`strength`) every path.
    pub fn reward(&mut self, texts: &[String], epochs: usize, strength: f64) -> Result<Vec<EpochRecord>, String> {
        if self.g.is_radix() {
            let o = crate::radix::Feedback {
                pos_epochs: epochs,
                strength,
                ..crate::radix::Feedback::two_nrl()
            };
            return self.radix_reward(texts, &o, &mut |_| true);
        }
        if self.is_resonant() {
            return self.resonant_reward(texts, epochs, strength, None, &mut |_| true);
        }
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
        if self.g.is_radix() {
            let o = crate::radix::Feedback {
                neg_epochs: epochs,
                strength,
                ..crate::radix::Feedback::two_nrl()
            };
            return self.radix_punish(texts, &o, &mut |_| true);
        }
        if self.is_resonant() {
            return self.resonant_punish(texts, epochs, strength, None, &mut |_| true);
        }
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
        if self.g.is_radix() {
            let o = crate::radix::Feedback {
                neg_epochs,
                pos_epochs,
                strength,
                ..crate::radix::Feedback::two_nrl()
            };
            return self.radix_two_nrl(bad, good, &o, &mut |_| true);
        }
        if self.is_resonant() {
            return self.resonant_two_nrl(bad, good, neg_epochs, pos_epochs, strength, None, None, &mut |_| true);
        }
        let negative = self.punish(bad, neg_epochs, strength)?;
        let positive = self.reward(good, pos_epochs, strength)?;
        self.meta.twonrl_runs.add(1);
        Ok((negative, positive))
    }

    /// Flips the sign of every reward - or, on the sine model, of every
    /// weight and activation, and on the phase model rotates every lock by pi
    /// and inverts the layer.
    pub fn invert(&mut self) {
        if self.is_resonant() {
            self.resonant_invert();
            return;
        }
        self.g.invert();
    }

    /// The steps of a traced text that wrote a unit inside one of `spans`, as
    /// `(prev node, edge)`.
    ///
    /// Every step is charged with the units it adds to the text: the first
    /// with the whole of its node's label, a later one with everything past
    /// what it overlaps its parent by, and the step into END with the position
    /// just past the last unit - where a sentence that stopped too early went
    /// wrong.  Used to move only the nodes a correction's diff marks as
    /// changed (`radixnet/model.py` `_steps_over`).
    pub fn steps_over(&self, grams: &[String], length: usize, spans: &[crate::diff::Span]) -> Vec<(usize, usize)> {
        let g = &self.g;
        let Some(path) = g.node_path(grams) else {
            return Vec::new();
        };
        if path.len() < 2 {
            return Vec::new();
        }
        let overlap = g.enc.overlap();
        let mut out = Vec::new();
        // the gram index of the node being entered
        let mut position = 0usize;
        for index in 1..path.len() {
            let node = path[index];
            // who called the step: START begins every walk
            let prev = if index >= 2 { path[index - 2] } else { START };
            let edge = g.edge(path[index - 1], node);
            if node == END {
                if let Some(e) = edge {
                    if crate::diff::spans_touch(length, length + 1, spans) {
                        out.push((prev, e));
                    }
                }
                break;
            }
            let size = g.label_len(node);
            let lo = if index == 1 { 0 } else { position + overlap };
            if let Some(e) = edge {
                if crate::diff::spans_touch(lo, position + size, spans) {
                    out.push((prev, e));
                }
            }
            position = (position + size).saturating_sub(overlap);
        }
        out
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
        // a text too short to hold one gram of this encoding is skipped
        let enc = self.g.enc;
        let usable: Vec<&String> = texts
            .iter()
            .filter(|t| {
                let ok = enc.len(t) >= enc.n;
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
                chars += enc.len(chunk[i]) as i64;
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
                extra: Vec::new(),
            };
            crate::log_debug!(
                LOG,
                "epoch {} {}: loss {:.4}, {} nodes, {} edges, {} merge(s), {:.2}s",
                record.epoch,
                opts.phase.as_deref().unwrap_or("train"),
                record.loss,
                record.nodes,
                record.edges,
                record.merges,
                record.seconds
            );
            self.history.push(record.clone());
            records.push(record);
        }
        crate::log_info!(
            LOG,
            "{} over {} text(s): loss {:.4} -> {:.4}, {} nodes",
            match opts.phase.as_deref() {
                Some(phase) => format!("{} pass(es), {phase}", records.len()),
                None => format!("{} epoch(s)", records.len()),
            },
            usable.len(),
            records.first().map(|r| r.loss).unwrap_or(0.0),
            records.last().map(|r| r.loss).unwrap_or(0.0),
            self.g.num_nodes()
        );
        if skipped_short > 0 {
            crate::log_warn!(
                LOG,
                "{skipped_short} text(s) skipped: shorter than one gram of {}",
                self.g.enc
            );
        }
        Ok(records)
    }

    /// Encodes a chunk of texts on the worker threads.
    fn encode_all(&self, texts: &[&String]) -> Vec<Vec<String>> {
        let enc = self.g.enc;
        let mut grams = vec![Vec::new(); texts.len()];
        parallel_fill(&mut grams, self.workers(), |i, slot| *slot = enc.encode(texts[i]));
        grams
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

    /// Where `prefix` ends in the graph: `(node, offset, units of the located
    /// gram that the prefix matched)`.
    fn locate(&self, prefix: &str) -> (usize, usize, usize) {
        let enc = self.g.enc;
        let u = enc.units(prefix);
        let n = u.len();
        if n == 0 {
            return (START, 0, 0);
        }
        if n >= enc.n {
            // the gram the prefix ends on, then - when the stride skips it -
            // the last gram of the prefix's own grid
            if let Some(l) = self.g.lookup(&u.slice(n - enc.n, n)) {
                return (l.node, l.off, enc.n);
            }
            let aligned = (n - enc.n) / enc.stride * enc.stride;
            if aligned != n - enc.n {
                if let Some(l) = self.g.lookup(&u.slice(aligned, aligned + enc.n)) {
                    return (l.node, l.off, enc.n);
                }
            }
            for k in [enc.n - 1, 1] {
                if k < 1 || k >= enc.n {
                    continue;
                }
                if let Some(l) = self.best_gram(&u.slice(n - k, n)) {
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

    /// The most visited `(node, offset)` holding a gram that starts with `key`.
    fn best_gram(&self, key: &str) -> Option<Loc> {
        let enc = self.g.enc;
        let mut best: Option<(i64, usize, usize, Loc)> = None;
        for (t, l) in self.g.index_entries() {
            if !enc.has_unit_prefix(t, key) {
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
        let mut best: Option<usize> = None;
        let mut best_count = Counter { value: -1, resets: 0 };
        for node in (START + 2)..self.g.num_node_ids() {
            // on a unit boundary: "the" is a prefix of "the cat sat", not of
            // "there is a" - a word encoding matches whole words
            if !self.g.is_alive(node) || !self.g.enc.has_unit_prefix(self.g.label(node), prefix) {
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
    pub(crate) fn prefix_start(&self, prefix: &str) -> (usize, usize, String) {
        let (node, offset, matched) = self.locate(prefix);
        let enc = self.g.enc;
        let lead = if node != START && matched < enc.n {
            enc.slice(self.g.label(node), offset + matched, offset + enc.n)
        } else {
            String::new()
        };
        (node, offset, lead)
    }

    /// Continues `prefix`: the K most likely and the K least likely
    /// continuations in one beam search, or one stochastic walk.
    pub fn predict(&mut self, prefix: &str, o: &PredictOptions) -> Result<Prediction, String> {
        if self.g.is_radix() {
            return self.radix_predict(prefix, o);
        }
        if self.is_resonant() {
            return self.resonant_predict(prefix, o, None);
        }
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
            &o.traversal,
            o.penalty_scale,
            o.merit_scale,
        )
    }

    /// The prediction engine shared by [`Model::predict`] and [`Model::generate`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search(
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
        traversal: &str,
        penalty_scale: f64,
        merit_scale: f64,
    ) -> Result<Prediction, String> {
        if self.is_resonant() {
            // the phase model's search: the phase and the layer are part of every walk
            let o = PredictOptions {
                length,
                mode: mode.to_string(),
                k: k.max(1),
                beam,
                step_penalty,
                temperature,
                to_end,
                max_length,
                traversal: traversal.to_string(),
                penalty_scale,
                merit_scale,
            };
            return self.resonant_predict(prefix, &o, rng);
        }
        // the traversal is two independent things: a cost function (the punishment
        // one prices a step) and a ranking (the least-punished one orders the walks)
        let name = crate::penalty::resolve_traversal(traversal)?;
        let priced = crate::penalty::traversal_costs(name, penalty_scale, merit_scale)?;
        let costs = priced.as_ref();
        let traversal = parse_traversal(name)?;
        // lengths are counted in the encoding's units all the way through:
        // characters by default, words under a word encoding
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
            let (t, b, e) = self.g.beam_predict_by(node, offset, want, opts, costs)?;
            top = t;
            bottom = b;
            expanded = e;
            width = if beam == 0 { default_beam(k) } else { beam };
        } else {
            cap = max_length.or(Some(length));
            let max_chars = cap.map(|c| c.saturating_sub(lead_len));
            let walk = self
                .g
                .sample_walk(node, offset, max_chars, temperature, rng, None, costs, traversal)?;
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
        let found = Prediction {
            best,
            top,
            bottom,
            k,
            beam: width,
            mode: mode.to_string(),
            traversal: name.to_string(),
            expanded,
        };
        Ok(found)
    }

    /// Generates whole texts with the prediction search, from `START` or
    /// continuing a prefix.
    pub fn generate(&mut self, o: &GenerateOptions) -> Result<Vec<PathResult>, String> {
        if self.is_resonant() {
            return self.resonant_generate(o);
        }
        let mode = if o.mode.is_empty() { "sample" } else { o.mode.as_str() };
        if !matches!(mode, "beam" | "dijkstra" | "sample") {
            return Err(format!(
                "unknown mode {mode:?}; expected 'beam', 'dijkstra' or 'sample'"
            ));
        }
        if o.count == 0 {
            return Ok(Vec::new());
        }
        // the search is what joined the prefix to the continuation, in the model's own
        // symbols; a generated text never re-joins them here, because only the search
        // knows the alphabet (`../../SPEC-WordNGrams.md`)
        let whole = |r: &mut PathResult| r.text = r.full_text.clone();
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
                    &o.traversal,
                    o.penalty_scale,
                    o.merit_scale,
                )?;
                whole(&mut found.best);
                results.push(found.best);
            }
            return Ok(results);
        }
        if self.g.is_radix() {
            // the sine model's own predict: its dijkstra is the exact cheapest path
            let found = self.radix_predict(
                &o.prefix,
                &PredictOptions {
                    length: 0,
                    mode: mode.to_string(),
                    k: o.count,
                    beam: o.beam,
                    step_penalty: o.step_penalty,
                    to_end: true,
                    max_length: Some(o.max_length),
                    traversal: o.traversal.clone(),
                    penalty_scale: o.penalty_scale,
                    merit_scale: o.merit_scale,
                    ..Default::default()
                },
            )?;
            let mut results = if mode == "dijkstra" {
                vec![found.best]
            } else {
                found.top
            };
            results.iter_mut().for_each(whole);
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
            &o.traversal,
            o.penalty_scale,
            o.merit_scale,
        )?;
        let mut results = found.top;
        results.iter_mut().for_each(whole);
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
        if self.is_resonant() {
            return self.resonant_score(text);
        }
        self.g.prepare();
        // `chars` and `per_char` are in the encoding's units: characters by
        // default, words under a word encoding, where a word the model has never
        // read is an unknown transition charged what any unknown one is charged
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
            if !lost && l.node == node && l.off == offset + 1 {
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
    /// What the search looks for, as opposed to `mode`, which is how it looks:
    /// `"reward"` (the default), `"punishment"` (the rewards leave the score and
    /// the punishments price every step, [`crate::penalty`]) or
    /// `"least-punished"` (the prices are untouched and a walk is ranked by the
    /// blame on its worst step, `../../SPEC-LeastPunished.md`).
    pub traversal: String,
    /// The punishment traversal's scales; ignored by the other two.
    pub penalty_scale: f64,
    pub merit_scale: f64,
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
            traversal: crate::penalty::DEFAULT_TRAVERSAL.to_string(),
            penalty_scale: 1.0,
            merit_scale: 1.0,
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
    /// See [`PredictOptions::traversal`].
    pub traversal: String,
    pub penalty_scale: f64,
    pub merit_scale: f64,
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
            traversal: crate::penalty::DEFAULT_TRAVERSAL.to_string(),
            penalty_scale: 1.0,
            merit_scale: 1.0,
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
    // what every number above is counted in; a per-word number read as
    // per-character is read wrong
    put("units", model.units().to_string());
    put("encoding", g.enc.to_string());
    put("unit", g.enc.unit.name().to_string());
    put("ngram", g.enc.n.to_string());
    put("stride", g.enc.stride.to_string());
    if g.enc.unit == crate::encoding::Unit::Words {
        put(
            "vocabulary",
            crate::encoding::vocabulary(&g.enc, g.gram_index()).len().to_string(),
        );
    }
    out
}

/// The node ids the model treats as real (not sentinels).
pub const FIRST_REAL_NODE: usize = FIRST;
