//! A chain of nodes that predicts the next input, traversing only as deep as it needs to.
//!
//! The same model as `latticefsm/chain.py`, number for number.  One input to a node: reading `"abc"`, input node 1
//! holds `a`, node 2 `b`, node 3 `c`, and the output node after them predicts what comes next.  Every input node, in
//! its context (the last `depth` inputs), keeps what came next, its own guess (the most frequent), and how often that
//! guess was right when it was tested while training; its confidence is
//! `(right + rewarded) / (tested + rewarded + punished + 2)`.  A prediction walks from the output node back through
//! the input nodes, nearest first, gathering confidence `1 - Π (1 - c)`, and stops as soon as that reaches the
//! chain's `confidence` threshold: the more confident the near nodes, the shorter the walk.  It predicts the
//! Witten-Bell distribution of the node it stopped at, smoothed through every node nearer the output.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};

use crate::gzip;
use crate::json::{parse, Json};
use crate::ngram::{detokenize, goal_of, tokenize, train_corpus, Drill, Sequence, Teaching, BOS, EOS, UNITS, UNK};
use crate::rng::Rng;

const FORMAT: &str = "latticefsm-chain";
/// Imagined misses every node starts with: its confidence is 0 until it is right.
pub const PRIOR: f64 = 2.0;
pub const MAX_NODES: usize = 16;

/// An input node in one context: what came next, its guess, and its record.
#[derive(Clone, Debug, Default)]
pub struct Node {
    pub counts: HashMap<u32, u64>,
    pub total: u64,
    pub best: Option<u32>,
    pub tested: u64,
    pub right: u64,
    pub rewarded: f64,
    pub punished: f64,
}

impl Node {
    pub fn confidence(&self) -> f64 {
        (self.right as f64 + self.rewarded) / (self.tested as f64 + self.rewarded + self.punished + PRIOR)
    }

    /// `n` more `w` after this context; the guess stays the most frequent (ties: the lower id).
    fn add(&mut self, w: u32, n: u64) {
        let c = self.counts.entry(w).or_insert(0);
        *c += n;
        let c = *c;
        self.total += n;
        self.best = Some(match self.best {
            None => w,
            Some(b) => {
                let bc = self.counts[&b];
                if c > bc || (c == bc && w < b) {
                    w
                } else {
                    b
                }
            }
        });
    }
}

/// What `evaluate` measures: over every input and `</s>`, perplexity, bits, how often the first guess was right,
/// the input nodes traversed on average, and how many walks stopped at each depth.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainEvaluation {
    pub tokens: usize,
    pub perplexity: Option<f64>,
    pub bits: Option<f64>,
    pub accuracy: Option<f64>,
    pub depth: Option<f64>,
    pub depths: Vec<usize>,
}

/// What predicting while training measured: inputs predicted, first guesses right (a share), perplexity, bits, the
/// input nodes traversed on average.
#[derive(Clone, Debug, PartialEq)]
pub struct Online {
    pub predicted: u64,
    pub accuracy: Option<f64>,
    pub perplexity: Option<f64>,
    pub bits: Option<f64>,
    pub depth: Option<f64>,
}

pub struct Chain {
    pub nodes: usize,
    pub unit: String,
    pub lower: bool,
    pub confidence: f64,
    pub vocab: Vec<String>,
    index: HashMap<String, u32>,
    /// `tables[k]`: the input nodes at depth `k` by context (the last `k` inputs); `tables[0]` the output node alone.
    pub tables: Vec<HashMap<Vec<u32>, Node>>,
    /// The output node's counts as buckets (count -> ids), for the guess among inputs no deeper node has seen.
    ranks: BTreeMap<u64, BTreeSet<u32>>,
    pub clock: u64,
    pub sentences: u64,
    pub sources: Vec<(String, usize, usize)>,
    pub history: Vec<Json>,
    /// Predict every input before learning it (the default).
    pub predict: bool,
    /// While predicting, reward the nodes that guessed a right first guess, punish those that guessed a wrong one.
    pub self_credit: f64,
    /// With `self_credit`: reward right guesses too (the default), or only punish wrong ones.
    pub self_reward: bool,
    /// (right first guesses, inputs predicted, sum of ln P, nodes traversed) since the last `reset_online`.
    pub online: (u64, u64, f64, u64),
}

impl Chain {
    pub fn new(nodes: usize, unit: &str, lower: bool, confidence: f64) -> Result<Chain, String> {
        if !(2..=MAX_NODES).contains(&nodes) {
            return Err(format!("nodes is 2 to {MAX_NODES}, got {nodes}"));
        }
        if !UNITS.contains(&unit) {
            return Err(format!("unit must be one of {UNITS:?}, got {unit:?}"));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(format!("confidence is 0 to 1, got {confidence}"));
        }
        let vocab: Vec<String> = [BOS, EOS, UNK].iter().map(|s| s.to_string()).collect();
        let index = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        Ok(Chain {
            nodes,
            unit: unit.to_string(),
            lower,
            confidence,
            vocab,
            index,
            tables: vec![HashMap::new(); nodes],
            ranks: BTreeMap::new(),
            clock: 0,
            sentences: 0,
            sources: Vec::new(),
            history: Vec::new(),
            predict: true,
            self_credit: 0.0,
            self_reward: true,
            online: (0, 0, 0.0, 0),
        })
    }

    /// The most input nodes a walk can traverse.
    pub fn depth(&self) -> usize {
        self.nodes - 1
    }

    // ---- inputs --------------------------------------------------------------------------------------------------

    pub fn tokens(&self, text: &str) -> Sequence {
        tokenize(text, &self.unit, self.lower)
    }

    pub fn prompt_tokens(&self, text: &str) -> Sequence {
        if self.unit == "base64" {
            return crate::b64::encode_prefix(text.as_bytes())
                .chars()
                .map(String::from)
                .collect();
        }
        self.tokens(text)
    }

    fn id(&mut self, token: &str, grow: bool) -> u32 {
        if let Some(&i) = self.index.get(token) {
            return i;
        }
        if !grow {
            return 2;
        }
        let i = self.vocab.len() as u32;
        self.vocab.push(token.to_string());
        self.index.insert(token.to_string(), i);
        i
    }

    /// `<s>` and the inputs as ids (unknown ones as `<unk>`): what the input nodes are filled from.
    pub fn history_ids(&self, tokens: &[String]) -> Vec<u32> {
        let mut ids = vec![0];
        ids.extend(tokens.iter().map(|t| self.index.get(t.as_str()).copied().unwrap_or(2)));
        ids
    }

    // ---- the walk ------------------------------------------------------------------------------------------------

    fn node(&self, ids: &[u32], k: usize) -> Option<&Node> {
        self.tables[k].get(&ids[ids.len() - k..])
    }

    /// How deep a prediction after `ids` walks (`None` before anything is learned): it stops once the confidence
    /// gathered on the way (`1 - Π (1 - c)`) reaches the threshold, at a node never reached, or when the inputs or
    /// the input nodes run out.  The walk is the nodes at depths `0..=d` along `ids`.
    pub fn walk(&self, ids: &[u32], confidence: Option<f64>) -> Option<usize> {
        let tau = confidence.unwrap_or(self.confidence);
        let first = self.node(ids, 0)?;
        let mut doubt = 1.0 - first.confidence();
        let reach = self.depth().min(ids.len());
        let mut k = 0;
        while 1.0 - doubt < tau && k < reach {
            match self.node(ids, k + 1) {
                None => break,
                Some(node) => {
                    k += 1;
                    doubt *= 1.0 - node.confidence();
                }
            }
        }
        Some(k)
    }

    fn uniform(&self) -> f64 {
        1.0 / (self.vocab.len() - 1) as f64
    }

    fn path<'a>(&'a self, ids: &[u32], d: Option<usize>) -> Vec<&'a Node> {
        match d {
            None => Vec::new(),
            Some(d) => (0..=d).map(|k| self.node(ids, k).expect("walked")).collect(),
        }
    }

    fn p(&self, path: &[&Node], w: u32) -> f64 {
        let mut p = if w != 0 { self.uniform() } else { 0.0 };
        for node in path {
            let t = node.counts.len() as u64;
            let c = node.counts.get(&w).copied().unwrap_or(0);
            p = (c as f64 + t as f64 * p) / (node.total + t) as f64;
        }
        p
    }

    fn dist(&self, path: &[&Node]) -> Vec<f64> {
        let u = self.uniform();
        let mut dist = vec![u; self.vocab.len()];
        dist[0] = 0.0;
        for node in path {
            let t = node.counts.len() as u64;
            let (tf, n) = (t as f64, (node.total + t) as f64);
            for (w, p) in dist.iter_mut().enumerate() {
                let c = node.counts.get(&(w as u32)).copied().unwrap_or(0);
                *p = (c as f64 + tf * *p) / n;
            }
        }
        dist[0] = 0.0;
        dist
    }

    /// The likeliest input after `path` (ties: the lower id; never `<s>`): an input no node past the output node has
    /// seen is ranked by the output node's counts alone, so only the seen ones and the best of the rest are scored.
    fn guess(&self, path: &[&Node]) -> u32 {
        if path.is_empty() {
            return 1;
        }
        let mut cand: BTreeSet<u32> = BTreeSet::new();
        for node in &path[1..] {
            cand.extend(node.counts.keys());
        }
        let mut outsider = None;
        for ids in self.ranks.values().rev() {
            if let Some(&w) = ids.iter().find(|w| !cand.contains(w)) {
                outsider = Some(w);
                break;
            }
        }
        if outsider.is_none() {
            outsider = (1..self.vocab.len() as u32).find(|w| !cand.contains(w) && !path[0].counts.contains_key(w));
        }
        cand.extend(outsider);
        let (mut best, mut bp) = (1, -1.0);
        for &w in &cand {
            let p = self.p(path, w);
            if p > bp {
                best = w;
                bp = p;
            }
        }
        best
    }

    /// Every node on the walk `0..=d` along `ids` whose own guess was `w` rewarded (`amount > 0`) or punished.
    fn credit_walk(&mut self, ids: &[u32], d: Option<usize>, w: u32, amount: f64) -> usize {
        let Some(d) = d else { return 0 };
        let mut n = 0;
        for k in 0..=d {
            let node = self.tables[k].get_mut(&ids[ids.len() - k..]).expect("walked");
            if node.best == Some(w) {
                if amount > 0.0 {
                    node.rewarded += amount;
                } else if amount < 0.0 {
                    node.punished -= amount;
                }
                n += 1;
            }
        }
        n
    }

    // ---- training ------------------------------------------------------------------------------------------------

    /// One sequence, `</s>` last: every input node already reached in its context is tested (its guess right or
    /// not) and then learns the input that came; with `predict` the whole chain first predicts it.  Returns how many
    /// inputs were learned.
    pub fn train_tokens(&mut self, tokens: &[String]) -> usize {
        let mut ids = vec![0u32];
        for t in tokens {
            let i = self.id(t, true);
            ids.push(i);
        }
        ids.push(1);
        for i in 1..ids.len() {
            let w = ids[i];
            let hist = &ids[..i];
            if self.predict {
                let d = self.walk(hist, None);
                let (guess, p) = {
                    let path = self.path(hist, d);
                    (self.guess(&path), self.p(&path, w))
                };
                self.online.0 += (guess == w) as u64;
                self.online.1 += 1;
                self.online.2 += p.max(1e-300).ln();
                self.online.3 += d.unwrap_or(0) as u64;
                let c = self.self_credit;
                if c != 0.0 && d.is_some() && (self.self_reward || guess != w) {
                    self.credit_walk(hist, d, guess, if guess == w { c } else { -c });
                }
            }
            for k in 0..=self.depth().min(i) {
                let ctx = &ids[i - k..i];
                if let Some(node) = self.tables[k].get_mut(ctx) {
                    node.tested += 1;
                    node.right += (node.best == Some(w)) as u64;
                } else {
                    self.tables[k].insert(ctx.to_vec(), Node::default());
                }
                let node = self.tables[k].get_mut(ctx).expect("there");
                if k == 0 {
                    let old = node.counts.get(&w).copied().unwrap_or(0);
                    if old > 0 {
                        let bucket = self.ranks.get_mut(&old).expect("ranked");
                        bucket.remove(&w);
                        if bucket.is_empty() {
                            self.ranks.remove(&old);
                        }
                    }
                    self.ranks.entry(old + 1).or_default().insert(w);
                }
                node.add(w, 1);
            }
            self.clock += 1;
        }
        self.sentences += 1;
        ids.len() - 1
    }

    /// What predicting while training measured since the last reset, and a fresh start.
    pub fn reset_online(&mut self) -> Online {
        let (right, n, logp, depth) = self.online;
        self.online = (0, 0, 0.0, 0);
        if n == 0 {
            return Online {
                predicted: 0,
                accuracy: None,
                perplexity: None,
                bits: None,
                depth: None,
            };
        }
        let nf = n as f64;
        Online {
            predicted: n,
            accuracy: Some(right as f64 / nf),
            perplexity: Some((-logp / nf).exp()),
            bits: Some(-logp / nf / 2f64.ln()),
            depth: Some(depth as f64 / nf),
        }
    }

    /// Every line one sequence; returns the inputs learned.
    pub fn train_lines(&mut self, lines: &[String], source: Option<&str>) -> usize {
        let seqs: Vec<Sequence> = lines.iter().map(|l| self.tokens(l)).collect();
        self.train_sequences(&seqs, source)
    }

    pub fn train_sequences(&mut self, seqs: &[Sequence], source: Option<&str>) -> usize {
        let (mut n, mut count) = (0, 0);
        for toks in seqs {
            if !toks.is_empty() {
                n += self.train_tokens(toks);
                count += 1;
            }
        }
        if let Some(s) = source {
            self.sources.push((s.to_string(), count, n));
        }
        n
    }

    /// Every full context of `seqs` - all the inputs a walk can reach - with what follows it and how often, and how
    /// many inputs there are.
    fn targets(&self, seqs: &[Sequence]) -> (HashMap<Vec<u32>, HashMap<u32, usize>>, usize) {
        let mut counts: HashMap<Vec<u32>, HashMap<u32, usize>> = HashMap::new();
        let mut n = 0;
        for seq in seqs {
            let mut ids = self.history_ids(seq);
            ids.push(1);
            for i in 1..ids.len() {
                let ctx = ids[i - self.depth().min(i)..i].to_vec();
                *counts.entry(ctx).or_default().entry(ids[i]).or_insert(0) += 1;
                n += 1;
            }
        }
        (counts, n)
    }

    /// How many inputs of `seqs` (each sequence and its `</s>`) any first guess could get right, of how many: in every
    /// full context only the input that follows it most often.
    pub fn ceiling(&self, seqs: &[Sequence]) -> (usize, usize) {
        let (counts, n) = self.targets(seqs);
        (counts.values().map(|r| r.values().copied().max().unwrap_or(0)).sum(), n)
    }

    /// Correct the chain on `seqs` (already learned) round after round until its first guess after every full context
    /// is the input that follows it most often, or `max_rounds`, or `patience` rounds without a gain: after every
    /// context guessed wrong the nodes on the walk that do not guess its input are punished by `2 ^ k`, and the deepest
    /// node it reaches, when it does not know, is given enough of the input to - as `Chain.drill`.
    pub fn drill(&mut self, seqs: &[Sequence], max_rounds: usize, patience: usize, threshold: f64) -> Drill {
        let (counts, n) = self.targets(seqs);
        let mut targets: Vec<(Vec<u32>, u32)> = counts
            .iter()
            .map(|(c, r)| {
                let t = r
                    .iter()
                    .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                    .map(|(&w, _)| w)
                    .unwrap_or(1);
                (c.clone(), t)
            })
            .collect();
        targets.sort();
        let best_possible: usize = counts.values().map(|r| r.values().copied().max().unwrap_or(0)).sum();
        let need = goal_of(threshold, best_possible);
        let mut streak: HashMap<Vec<u32>, i32> = HashMap::new();
        let mut rounds = Vec::new();
        let (mut best, mut stale) = (None, 0);
        loop {
            let mut right = 0;
            for seq in seqs {
                let mut ids = self.history_ids(seq);
                ids.push(1);
                for i in 1..ids.len() {
                    let path = self.path(&ids[..i], self.walk(&ids[..i], None));
                    right += (self.guess(&path) == ids[i]) as usize;
                }
            }
            rounds.push(right);
            if best.is_none_or(|b| right > b) {
                best = Some(right);
                stale = 0;
            } else {
                stale += 1;
            }
            if right >= need || rounds.len() >= max_rounds || stale >= patience {
                return Drill {
                    right,
                    of: n,
                    best: best_possible,
                    need,
                    reached: right >= need,
                    rounds,
                };
            }
            for (c, t) in &targets {
                let d = self.walk(c, None);
                let g = {
                    let path = self.path(c, d);
                    self.guess(&path)
                };
                if g != *t {
                    let k = streak.entry(c.clone()).or_insert(0);
                    let a = 2f64.powi(*k);
                    let more = (1u64 << (*k).min(62)) - 1;
                    *k += 1;
                    if let Some(d) = d {
                        for depth in 0..=d {
                            let node = self.tables[depth].get_mut(&c[c.len() - depth..]).expect("walked");
                            if node.best != Some(*t) {
                                node.punished += a;
                            }
                        }
                    }
                    let full = d == Some(c.len());
                    if let Some(deep) = self.tables[c.len()].get_mut(c.as_slice()) {
                        if deep.best != Some(*t) || full {
                            let extra = match deep.best.filter(|b| b != t) {
                                Some(b) => {
                                    deep.counts[&b] - deep.counts.get(t).copied().unwrap_or(0) + u64::from(*t > b)
                                }
                                None => 0,
                            };
                            if extra + more > 0 {
                                deep.add(*t, extra + more);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Train on every sequence under `paths`, streamed (`ngram::train_corpus`), every `held_out`-th sequence of each
    /// document kept back and returned.
    pub fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String> {
        let (unit, lower) = (self.unit.clone(), self.lower);
        let mut sources = std::mem::take(&mut self.sources);
        let r = train_corpus(
            paths,
            &unit,
            lower,
            held_out,
            &mut |t| self.train_tokens(t),
            &mut sources,
        );
        self.sources = sources;
        r
    }

    // ---- predicting ----------------------------------------------------------------------------------------------

    /// What the chain does after `toks`: the inputs in the input nodes (`null` where there is none yet), the walk -
    /// for every node traversed, the output node first, its depth, node number, input, confidence, the confidence
    /// gathered up to it, its guess and record - how deep it went, and the `k` likeliest next inputs.
    pub fn explain(&self, toks: &[String], k: usize, confidence: Option<f64>) -> Json {
        let ids = self.history_ids(toks);
        let shown = &ids[ids.len().saturating_sub(self.depth())..];
        let mut inputs: Vec<Json> = vec![Json::Null; self.depth() - shown.len()];
        inputs.extend(shown.iter().map(|&i| Json::from(self.vocab[i as usize].as_str())));
        let d = self.walk(&ids, confidence);
        let path = self.path(&ids, d);
        let mut steps = Vec::new();
        let mut doubt = 1.0;
        for (depth, node) in path.iter().enumerate() {
            doubt *= 1.0 - node.confidence();
            let input = if depth > 0 {
                Json::from(self.vocab[ids[ids.len() - depth] as usize].as_str())
            } else {
                Json::Null
            };
            steps.push(
                Json::object()
                    .with("depth", depth.into())
                    .with("node", (self.nodes - depth).into())
                    .with("input", input)
                    .with("confidence", node.confidence().into())
                    .with("gathered", (1.0 - doubt).into())
                    .with(
                        "guess",
                        node.best
                            .map(|b| Json::from(self.vocab[b as usize].as_str()))
                            .unwrap_or(Json::Null),
                    )
                    .with("seen", (node.total as f64).into())
                    .with("tested", (node.tested as f64).into())
                    .with("right", (node.right as f64).into())
                    .with("rewarded", node.rewarded.into())
                    .with("punished", node.punished.into()),
            );
        }
        let dist = self.dist(&path);
        let mut order: Vec<usize> = (1..dist.len()).collect();
        order.sort_by(|&a, &b| {
            dist[b]
                .partial_cmp(&dist[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order.truncate(k);
        let next = order
            .iter()
            .map(|&w| {
                Json::object()
                    .with("token", self.vocab[w].as_str().into())
                    .with("p", dist[w].into())
            })
            .collect();
        Json::object()
            .with("nodes", self.nodes.into())
            .with("unit", self.unit.as_str().into())
            .with("inputs", Json::Array(inputs))
            .with("walk", Json::Array(steps))
            .with("depth", d.unwrap_or(0).into())
            .with("confidence", confidence.unwrap_or(self.confidence).into())
            .with("next", Json::Array(next))
    }

    /// The `k` likeliest next inputs after `toks`, likeliest first (ties: the input seen first).
    pub fn predict_tokens(&self, toks: &[String], k: usize) -> Vec<(String, f64)> {
        let ids = self.history_ids(toks);
        let path = self.path(&ids, self.walk(&ids, None));
        let dist = self.dist(&path);
        let mut order: Vec<usize> = (1..dist.len()).collect();
        order.sort_by(|&a, &b| {
            dist[b]
                .partial_cmp(&dist[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order
            .into_iter()
            .take(k)
            .map(|w| (self.vocab[w].clone(), dist[w]))
            .collect()
    }

    /// Inputs continuing `toks` until `</s>` or `max_tokens` (never `<s>` or `<unk>`), and how deep each walk went.
    pub fn continue_tokens(
        &self,
        toks: &[String],
        max_tokens: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> (Vec<String>, Vec<usize>) {
        let mut ids = self.history_ids(toks);
        let (mut out, mut depths) = (Vec::new(), Vec::new());
        for _ in 0..max_tokens {
            let d = self.walk(&ids, None);
            let mut dist = self.dist(&self.path(&ids, d));
            dist[0] = 0.0;
            dist[2] = 0.0;
            let w = if temperature <= 0.0 {
                let mut best = 0;
                for i in 1..dist.len() {
                    if dist[i] > dist[best] {
                        best = i;
                    }
                }
                best
            } else {
                let weights: Vec<f64> = dist
                    .iter()
                    .map(|&p| if p > 0.0 { p.powf(1.0 / temperature) } else { 0.0 })
                    .collect();
                let r = rng.random() * weights.iter().sum::<f64>();
                let (mut acc, mut pick) = (0.0, 1);
                for (i, q) in weights.iter().enumerate() {
                    acc += q;
                    if r < acc {
                        pick = i;
                        break;
                    }
                }
                pick
            };
            depths.push(d.unwrap_or(0));
            if w == 1 {
                break;
            }
            out.push(self.vocab[w].clone());
            ids.push(w as u32);
        }
        (out, depths)
    }

    pub fn complete(&self, prompt: &str, max_tokens: usize, temperature: f64, rng: &mut Rng) -> String {
        let start = self.prompt_tokens(prompt);
        let (toks, _) = self.continue_tokens(&start, max_tokens, temperature, rng);
        if self.unit == "base64" {
            let mut all = start;
            all.extend(toks);
            return detokenize(&all, &self.unit);
        }
        detokenize(&toks, &self.unit)
    }

    // ---- measuring -----------------------------------------------------------------------------------------------

    /// On held-out sequences, at the chain's threshold or another: nothing moves.
    pub fn evaluate(&self, seqs: &[Sequence], confidence: Option<f64>) -> ChainEvaluation {
        let (mut logp, mut n, mut right, mut depth) = (0.0, 0usize, 0usize, 0usize);
        let mut depths = vec![0usize; self.nodes];
        for toks in seqs {
            if toks.is_empty() {
                continue;
            }
            let mut ids = self.history_ids(toks);
            ids.push(1);
            for i in 1..ids.len() {
                let w = ids[i];
                let d = self.walk(&ids[..i], confidence);
                let path = self.path(&ids[..i], d);
                let p = if path.is_empty() {
                    self.uniform()
                } else {
                    self.p(&path, w)
                };
                logp += p.max(1e-300).ln();
                right += (self.guess(&path) == w) as usize;
                let d = d.unwrap_or(0);
                depth += d;
                depths[d] += 1;
                n += 1;
            }
        }
        if n == 0 {
            return ChainEvaluation {
                tokens: 0,
                perplexity: None,
                bits: None,
                accuracy: None,
                depth: None,
                depths,
            };
        }
        let nf = n as f64;
        ChainEvaluation {
            tokens: n,
            perplexity: Some((-logp / nf).exp()),
            bits: Some(-logp / nf / 2f64.ln()),
            accuracy: Some(right as f64 / nf),
            depth: Some(depth as f64 / nf),
            depths,
        }
    }

    /// `evaluate` at every threshold: how deep the walks go against how well they predict.
    pub fn sweep(&self, seqs: &[Sequence], thresholds: &[f64]) -> Vec<(f64, ChainEvaluation)> {
        thresholds.iter().map(|&t| (t, self.evaluate(seqs, Some(t)))).collect()
    }

    // ---- credit --------------------------------------------------------------------------------------------------

    /// Every input of a text, `</s>` included, as a prediction: the nodes on the walk to it that guessed it rewarded
    /// (`amount > 0`) or punished.  Returns how many nodes were credited.
    pub fn credit_text(&mut self, text: &str, amount: f64) -> usize {
        let toks = self.tokens(text);
        self.credit_tokens(&toks, amount)
    }

    pub fn credit_tokens(&mut self, toks: &[String], amount: f64) -> usize {
        let mut ids = self.history_ids(toks);
        ids.push(1);
        let mut n = 0;
        for i in 1..ids.len() {
            let d = self.walk(&ids[..i], None);
            n += self.credit_walk(&ids[..i], d, ids[i], amount);
        }
        n
    }

    /// Every input of `toks` and the `</s>` after it, with the probability the chain gives it as it reads them and how
    /// deep the walk to it went.
    pub fn score_tokens(&self, toks: &[String]) -> Vec<(String, f64, usize)> {
        let mut ids = self.history_ids(toks);
        ids.push(1);
        (1..ids.len())
            .map(|i| {
                let d = self.walk(&ids[..i], None);
                let path = self.path(&ids[..i], d);
                let p = if path.is_empty() {
                    self.uniform()
                } else {
                    self.p(&path, ids[i])
                };
                let t = if i == ids.len() - 1 {
                    EOS.to_string()
                } else {
                    toks[i - 1].clone()
                };
                (t, p, d.unwrap_or(0))
            })
            .collect()
    }

    /// Teach the chain that `reply` follows `text` (inputs), round after round, until its first guess is right on every
    /// input of the reply and on the `</s>` after it, or `max_rounds` rounds have passed: a round punishes, at every
    /// wrong guess, the nodes on the walk that guessed it (by 1 the first round a position is wrong, doubling every
    /// round it stays wrong) and then learns `text + reply` as one sequence, as `Chain.teach` does.
    pub fn teach(&mut self, text: &[String], reply: &[String], max_rounds: usize, threshold: f64) -> Teaching {
        let of = reply.len() + 1;
        let need = goal_of(threshold, of);
        let mut history = Vec::new();
        let mut streak = vec![0i32; of];
        let mut rounds = 0;
        let mut all: Vec<String> = text.to_vec();
        all.extend_from_slice(reply);
        loop {
            let mut right = 0;
            let mut wrong: Vec<(Vec<u32>, u32, f64)> = Vec::new();
            let mut ids = self.history_ids(text);
            for i in 0..of {
                let w = if i == reply.len() {
                    1
                } else {
                    self.index.get(reply[i].as_str()).copied().unwrap_or(2)
                };
                let g = {
                    let path = self.path(&ids, self.walk(&ids, None));
                    self.guess(&path)
                };
                if g == w {
                    right += 1;
                    streak[i] = 0;
                } else {
                    wrong.push((ids.clone(), g, 2f64.powi(streak[i])));
                    streak[i] += 1;
                }
                ids.push(w);
            }
            history.push(right);
            if right >= need || rounds >= max_rounds {
                return Teaching {
                    right,
                    of,
                    need,
                    rounds,
                    reached: right >= need,
                    history,
                };
            }
            for (hist, g, amount) in wrong {
                let d = self.walk(&hist, None);
                self.credit_walk(&hist, d, g, -amount);
            }
            let predict = std::mem::replace(&mut self.predict, false);
            self.train_tokens(&all);
            self.predict = predict;
            rounds += 1;
        }
    }

    // ---- describing ----------------------------------------------------------------------------------------------

    pub fn stats(&self) -> Json {
        let per: Vec<Json> = self
            .tables
            .iter()
            .enumerate()
            .map(|(k, t)| {
                let tested: u64 = t.values().map(|n| n.tested).sum();
                let right: u64 = t.values().map(|n| n.right).sum();
                Json::object()
                    .with("depth", k.into())
                    .with("contexts", t.len().into())
                    .with("tested", (tested as f64).into())
                    .with(
                        "accuracy",
                        if tested > 0 {
                            Json::from(right as f64 / tested as f64)
                        } else {
                            Json::Null
                        },
                    )
            })
            .collect();
        Json::object()
            .with("nodes", self.nodes.into())
            .with("unit", self.unit.as_str().into())
            .with("lower", self.lower.into())
            .with("confidence", self.confidence.into())
            .with("vocab", (self.vocab.len() - 3).into())
            .with("contexts", self.tables.iter().map(HashMap::len).sum::<usize>().into())
            .with("tokens", (self.clock as f64).into())
            .with("sentences", (self.sentences as f64).into())
            .with("sources", self.sources.len().into())
            .with(
                "credited",
                self.tables
                    .iter()
                    .flat_map(|t| t.values())
                    .filter(|n| n.rewarded != 0.0 || n.punished != 0.0)
                    .count()
                    .into(),
            )
            .with("depths", Json::Array(per))
    }

    // ---- the file ------------------------------------------------------------------------------------------------

    pub fn to_json(&self) -> Json {
        let mut rows: Vec<(Vec<u32>, Json)> = Vec::new();
        for (k, table) in self.tables.iter().enumerate() {
            for (ctx, nd) in table {
                let mut key = vec![k as u32];
                key.extend(ctx);
                let mut row: Vec<Json> = key.iter().map(|&x| Json::from(x as f64)).collect();
                row.push((nd.tested as f64).into());
                row.push((nd.right as f64).into());
                row.push(nd.rewarded.into());
                row.push(nd.punished.into());
                let mut ws: Vec<&u32> = nd.counts.keys().collect();
                ws.sort();
                for w in ws {
                    row.push((*w as f64).into());
                    row.push((nd.counts[w] as f64).into());
                }
                rows.push((key, Json::Array(row)));
            }
        }
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Json::object()
            .with("format", FORMAT.into())
            .with("version", 1.0.into())
            .with("nodes", self.nodes.into())
            .with("unit", self.unit.as_str().into())
            .with("lower", self.lower.into())
            .with("confidence", self.confidence.into())
            .with("clock", (self.clock as f64).into())
            .with("sentences", (self.sentences as f64).into())
            .with("vocab", Json::strings(&self.vocab))
            .with("table", Json::Array(rows.into_iter().map(|(_, r)| r).collect()))
            .with(
                "sources",
                Json::Array(
                    self.sources
                        .iter()
                        .map(|(s, c, n)| Json::Array(vec![s.as_str().into(), (*c).into(), (*n).into()]))
                        .collect(),
                ),
            )
            .with("history", Json::Array(self.history.clone()))
    }

    pub fn from_json(v: &Json) -> Result<Chain, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let mut m = Chain::new(
            v.num("nodes", 8.0) as usize,
            v.str_or("unit", "char"),
            v.bool_or("lower", true),
            v.num("confidence", 0.9),
        )?;
        m.vocab = v
            .get("vocab")
            .and_then(Json::as_array)
            .ok_or("no vocab")?
            .iter()
            .map(|s| s.as_str().map(String::from).ok_or("the vocab is strings"))
            .collect::<Result<_, _>>()?;
        if m.vocab.len() < 3 || m.vocab[0] != BOS || m.vocab[1] != EOS || m.vocab[2] != UNK {
            return Err("the vocabulary does not start <s> </s> <unk>".to_string());
        }
        m.index = m.vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        for row in v
            .get("table")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let r: Vec<f64> = row
                .as_array()
                .map(|a| a.iter().filter_map(Json::as_f64).collect())
                .unwrap_or_default();
            let k = r.first().copied().unwrap_or(-1.0);
            if k < 0.0 || k as usize >= m.nodes || r.len() < k as usize + 5 || (r.len() - k as usize - 5) % 2 != 0 {
                return Err("a node row of the wrong shape".to_string());
            }
            let k = k as usize;
            let ctx: Vec<u32> = r[1..k + 1].iter().map(|&x| x as u32).collect();
            let mut nd = Node {
                tested: r[k + 1] as u64,
                right: r[k + 2] as u64,
                rewarded: r[k + 3],
                punished: r[k + 4],
                ..Node::default()
            };
            for pair in r[k + 5..].chunks(2) {
                let (w, c) = (pair[0] as u32, pair[1] as u64);
                nd.counts.insert(w, c);
                nd.total += c;
                let better = match nd.best {
                    None => true,
                    Some(b) => c > nd.counts[&b] || (c == nd.counts[&b] && w < b),
                };
                if better {
                    nd.best = Some(w);
                }
                if k == 0 {
                    m.ranks.entry(c).or_default().insert(w);
                }
            }
            m.tables[k].insert(ctx, nd);
        }
        m.clock = v.num("clock", 0.0) as u64;
        m.sentences = v.num("sentences", 0.0) as u64;
        for s in v
            .get("sources")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if let Some(a) = s.as_array() {
                if a.len() == 3 {
                    m.sources.push((
                        a[0].as_str().unwrap_or("").to_string(),
                        a[1].as_f64().unwrap_or(0.0) as usize,
                        a[2].as_f64().unwrap_or(0.0) as usize,
                    ));
                }
            }
        }
        m.history = v.get("history").and_then(Json::as_array).cloned().unwrap_or_default();
        Ok(m)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let mut f = fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

pub fn load_chain(path: &str) -> Result<Chain, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if path.ends_with(".gz") || bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    Chain::from_json(&parse(&text)?)
}
