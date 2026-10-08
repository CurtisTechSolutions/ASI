//! The primed radix pair over the tokenizer's codes: a count tree and a reward tree sharing one address
//! space whose nodes are every prefix of a context code. See ../DESIGN.md sections 3 and 4.

use crate::nn::Rng;
use crate::tokenizer::Codes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The outcome that closes a text, and the outcome alphabet: the bytes and END.
pub const END: usize = 256;
pub const OUT: usize = 257;

/// The outcome count of English as the repository's phonetic tokenizer spells it: 24 consonants and 15
/// vowels at three stress levels (39 phonemes without stress, 88 emitted symbols with boundaries and pauses).
pub const ENGLISH_PHONES: usize = 69;

/// Numbers the context nodes: every prefix of a code, the empty prefix being the root.
#[derive(Clone, Debug)]
pub struct Address {
    pub radices: Vec<usize>,
    pub out: usize,
    pub bases: Vec<usize>,
    pub n: usize,
}

impl Address {
    pub fn new(radices: &[usize], out: usize, cell_ceiling: usize) -> Result<Address, String> {
        if radices.is_empty() {
            return Err("a code needs at least one symbol".into());
        }
        let mut bases = vec![0usize; radices.len() + 2];
        let mut width = 1usize;
        for (l, &r) in radices.iter().enumerate() {
            if r < 1 {
                return Err(format!("symbol {l} has radix {r}"));
            }
            bases[l + 1] = bases[l] + width;
            width = width.saturating_mul(r);
            if width > cell_ceiling {
                return Err(format!("level {} alone needs {width} nodes, above the ceiling", l + 1));
            }
        }
        let n = bases[radices.len()] + width;
        bases[radices.len() + 1] = n;
        if n.saturating_mul(out) > cell_ceiling {
            return Err(format!("{n} context nodes x {out} outcomes = {} cells, above the ceiling of {cell_ceiling}", n * out));
        }
        Ok(Address { radices: radices.to_vec(), out, bases, n })
    }
    pub fn d(&self) -> usize {
        self.radices.len()
    }
    pub fn cells(&self) -> usize {
        self.n * self.out
    }
    /// The node of the first l symbols of code.
    pub fn of(&self, code: &[i32], l: usize) -> usize {
        let mut v = 0usize;
        for j in 0..l {
            v = v * self.radices[j] + code[j] as usize;
        }
        self.bases[l] + v
    }
    /// The node of every prefix of code, root first, full code last.
    pub fn chain(&self, code: &[i32], out: &mut Vec<usize>) {
        out.clear();
        out.push(0);
        let mut v = 0usize;
        for (j, &r) in self.radices.iter().enumerate() {
            v = v * r + code[j] as usize;
            out.push(self.bases[j + 1] + v);
        }
    }
    pub fn level(&self, node: usize) -> usize {
        self.bases.partition_point(|&b| b <= node) - 1
    }
    pub fn code(&self, node: usize) -> Vec<i32> {
        let l = self.level(node);
        let mut v = node - self.bases[l];
        let mut out = vec![0i32; l];
        for j in (0..l).rev() {
            out[j] = (v % self.radices[j]) as i32;
            v /= self.radices[j];
        }
        out
    }
    pub fn parent(&self, node: usize) -> usize {
        let l = self.level(node);
        if l == 0 {
            0
        } else {
            self.bases[l - 1] + (node - self.bases[l]) / self.radices[l - 1]
        }
    }
    pub fn cell(&self, node: usize, x: usize) -> usize {
        node * self.out + x
    }
    pub fn valid(&self, code: &[i32]) -> bool {
        code.len() == self.radices.len() && code.iter().zip(&self.radices).all(|(&c, &r)| c >= 0 && (c as usize) < r)
    }
}

/// Counts, under every prefix of a position's code, the outcome that followed. Written only by reading.
#[derive(Clone, Debug)]
pub struct CountTree {
    pub addr: Arc<Address>,
    pub cnt: Vec<i64>,
    pub ctx: Vec<i64>,
    pub texts: usize,
    pub units: usize,
    pub version: u64,
    pub alpha: f64,
    pub smoothing: f64,
    chain: Vec<usize>,
}

impl CountTree {
    pub fn new(addr: Arc<Address>, alpha: f64, smoothing: f64) -> CountTree {
        CountTree { cnt: vec![0; addr.cells()], ctx: vec![0; addr.n], addr, texts: 0, units: 0, version: 0, alpha, smoothing, chain: Vec::new() }
    }
    pub fn observe(&mut self, codes: &Codes, outcomes: &[usize]) {
        for (i, &x) in outcomes.iter().enumerate() {
            self.addr.chain(codes.at(i), &mut self.chain);
            for &node in &self.chain {
                self.cnt[node * self.addr.out + x] += 1;
                self.ctx[node] += 1;
            }
        }
        self.texts += 1;
        self.units += outcomes.len();
        self.version += 1;
    }
    pub fn children(&self, node: usize) -> &[i64] {
        let start = self.addr.cell(node, 0);
        &self.cnt[start..start + self.addr.out]
    }
    /// (count + s) / (ctx + s * out) per outcome; all zero when nothing was read and s == 0.
    pub fn share(&self, node: usize, out: &mut Vec<f64>) {
        out.clear();
        let denom = self.ctx[node] as f64 + self.smoothing * self.addr.out as f64;
        for &c in self.children(node) {
            out.push(if denom <= 0.0 { 0.0 } else { (c as f64 + self.smoothing) / denom });
        }
    }
    /// ctx / (ctx + alpha).
    pub fn own(&self, node: usize) -> f64 {
        let c = self.ctx[node] as f64;
        if c <= 0.0 {
            0.0
        } else {
            c / (c + self.alpha)
        }
    }
    /// The depth-weighted share of a text's chain nodes this tree has met.
    pub fn coverage(&mut self, codes: &Codes) -> f64 {
        let (mut seen, mut of) = (0.0, 0.0);
        for i in 0..codes.len() {
            self.addr.chain(codes.at(i), &mut self.chain);
            for l in 1..self.chain.len() {
                let w = l as f64;
                of += w;
                if self.ctx[self.chain[l]] > 0 {
                    seen += w;
                }
            }
        }
        if of == 0.0 {
            0.0
        } else {
            seen / of
        }
    }
    pub fn nonzero(&self, mut f: impl FnMut(usize, i64)) {
        for (i, &c) in self.cnt.iter().enumerate() {
            if c != 0 {
                f(i, c);
            }
        }
    }
}

/// What outcomes earned and what they were punished for, kept apart. Written only by judged outcomes.
#[derive(Clone, Debug)]
pub struct RewardTree {
    pub addr: Arc<Address>,
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
    pub judged: usize,
    pub rewards_total: f64,
    pub penalties_total: f64,
    pub version: u64,
    chain: Vec<usize>,
}

impl RewardTree {
    pub fn new(addr: Arc<Address>) -> RewardTree {
        RewardTree { plus: vec![0.0; addr.cells()], minus: vec![0.0; addr.cells()], addr, judged: 0, rewards_total: 0.0, penalties_total: 0.0, version: 0, chain: Vec::new() }
    }
    /// Adds amount (negative: a punishment) to the cells of a judged text's steps from position skip on,
    /// at every level (rungs "all") or the full code only ("final"). Returns the cells credited.
    pub fn credit(&mut self, codes: &Codes, outcomes: &[usize], amount: f64, rungs: &str, skip: usize) -> Result<usize, String> {
        if rungs != "all" && rungs != "final" {
            return Err(format!("rungs must be all or final, got {rungs:?}"));
        }
        if amount == 0.0 {
            return Ok(0);
        }
        let a = amount.abs();
        let mut n = 0;
        for i in skip..outcomes.len() {
            let x = outcomes[i];
            if x >= self.addr.out {
                return Err(format!("outcome {x} outside 0..{}", self.addr.out - 1));
            }
            self.addr.chain(codes.at(i), &mut self.chain);
            let target = if amount > 0.0 { &mut self.plus } else { &mut self.minus };
            if rungs == "all" {
                for &node in &self.chain {
                    target[node * self.addr.out + x] += a;
                }
                n += self.chain.len();
            } else {
                target[self.chain[self.chain.len() - 1] * self.addr.out + x] += a;
                n += 1;
            }
        }
        if amount > 0.0 {
            self.rewards_total += a * n as f64;
        } else {
            self.penalties_total += a * n as f64;
        }
        self.judged += 1;
        self.version += 1;
        Ok(n)
    }
    pub fn invert(&mut self) {
        std::mem::swap(&mut self.plus, &mut self.minus);
        std::mem::swap(&mut self.rewards_total, &mut self.penalties_total);
        self.version += 1;
    }
    pub fn reward(&self, cell: usize) -> f64 {
        self.plus[cell] - self.minus[cell]
    }
    pub fn penalty(&self, cell: usize) -> f64 {
        self.minus[cell]
    }
    pub fn rewards(&self, node: usize, out: &mut Vec<f64>) {
        let start = self.addr.cell(node, 0);
        out.clear();
        out.extend((start..start + self.addr.out).map(|i| self.plus[i] - self.minus[i]));
    }
    pub fn penalties(&self, node: usize, out: &mut Vec<f64>) {
        let start = self.addr.cell(node, 0);
        out.clear();
        out.extend_from_slice(&self.minus[start..start + self.addr.out]);
    }
    pub fn nonzero(&self, mut f: impl FnMut(usize, f64, f64)) {
        for i in 0..self.plus.len() {
            if self.plus[i] != 0.0 || self.minus[i] != 0.0 {
                f(i, self.plus[i], self.minus[i]);
            }
        }
    }
}

/// The numbers a prediction reads.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub alpha: f64,
    pub floor: f64,
    pub smoothing: f64,
    pub share_scale: f64,
    pub reward_scale: f64,
    pub merit_scale: f64,
    pub penalty_scale: f64,
    pub strength: f64,
    #[serde(default)]
    pub outcomes: usize,
    pub rungs: String,
    pub backoff: String,
    pub cell_ceiling: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            alpha: 2.0,
            floor: 0.02,
            smoothing: 0.0,
            share_scale: 1.0,
            reward_scale: 1.0,
            merit_scale: 1.0,
            penalty_scale: 1.0,
            strength: 1.0,
            outcomes: ENGLISH_PHONES,
            rungs: "all".into(),
            backoff: "all".into(),
            cell_ceiling: 16_777_216,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        if self.alpha < 0.0 || self.floor < 0.0 || self.floor >= 1.0 || self.smoothing < 0.0 {
            return Err("alpha >= 0, 0 <= floor < 1 and smoothing >= 0".into());
        }
        if self.rungs != "all" && self.rungs != "final" {
            return Err(format!("rungs must be all or final, got {:?}", self.rungs));
        }
        if !["all", "deepest", "none"].contains(&self.backoff.as_str()) {
            return Err(format!("backoff must be all, deepest or none, got {:?}", self.backoff));
        }
        if self.cell_ceiling < 1 {
            return Err("cell_ceiling must be positive".into());
        }
        if self.outcomes < 1 {
            return Err(format!("outcomes must be at least 1, got {}", self.outcomes));
        }
        Ok(())
    }
}

/// A text scored: the model's belief and the reward tree's readings of its steps.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Score {
    pub bits: f64,
    pub mean_reward: f64,
    pub worst_penalty: f64,
    pub units: usize,
    pub per_unit: Vec<UnitScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitScore {
    pub unit: String,
    pub bits: f64,
    pub reward: f64,
    pub penalty: f64,
}

/// A walk: the outcomes it chose and what they cost.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PathResult {
    pub text: String,
    pub full_text: String,
    pub units: Vec<usize>,
    pub cost: f64,
    pub step_costs: Vec<f64>,
    pub traversal: String,
    pub mode: String,
    pub reached_end: bool,
    pub peak: f64,
    #[serde(skip)]
    pub bytes: Vec<u8>,
}

/// Names an outcome: printable bytes as themselves, the rest escaped, END as </s>.
pub fn symbol(x: usize) -> String {
    match x {
        END => "</s>".into(),
        10 => "\\n".into(),
        9 => "\\t".into(),
        32..=126 => (x as u8 as char).to_string(),
        _ => format!("\\x{x:02x}"),
    }
}

/// A text's outcomes: its bytes then END.
pub fn outcomes(text: &[u8]) -> Vec<usize> {
    let mut out: Vec<usize> = text.iter().map(|&b| b as usize).collect();
    out.push(END);
    out
}

/// A distribution from scores in place; -inf scores get 0; all -inf gives the uniform.
pub fn softmax(scores: &mut [f64]) {
    let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if top == f64::NEG_INFINITY {
        let u = 1.0 / scores.len() as f64;
        scores.iter_mut().for_each(|s| *s = u);
        return;
    }
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = if *s == f64::NEG_INFINITY { 0.0 } else { (*s - top).exp() };
        sum += *s;
    }
    scores.iter_mut().for_each(|s| *s /= sum);
}

/// The two trees over one address space.
#[derive(Clone, Debug)]
pub struct Pair {
    pub addr: Arc<Address>,
    pub count: CountTree,
    pub reward: RewardTree,
    pub settings: Settings,
    share_buf: Vec<f64>,
    rew_buf: Vec<f64>,
    chain_buf: Vec<usize>,
}

impl Pair {
    pub fn new(radices: &[usize], settings: Settings) -> Result<Pair, String> {
        settings.validate()?;
        let addr = Arc::new(Address::new(radices, OUT, settings.cell_ceiling)?);
        Ok(Pair {
            count: CountTree::new(addr.clone(), settings.alpha, settings.smoothing),
            reward: RewardTree::new(addr.clone()),
            addr,
            settings,
            share_buf: Vec::new(),
            rew_buf: Vec::new(),
            chain_buf: Vec::new(),
        })
    }
    pub fn configure(&mut self, settings: Settings) -> Result<(), String> {
        settings.validate()?;
        self.count.alpha = settings.alpha;
        self.count.smoothing = settings.smoothing;
        self.settings = settings;
        Ok(())
    }

    /// The step scores of a node's outcomes under a traversal.
    pub fn scores(&mut self, node: usize, traversal: &str, out: &mut Vec<f64>) -> Result<(), String> {
        if traversal != "reward" && traversal != "punishment" {
            return Err(format!("traversal must be reward or punishment, got {traversal:?}"));
        }
        let s = self.settings.clone();
        self.count.share(node, &mut self.share_buf);
        out.clear();
        let read_rewards = s.rungs == "all" || self.addr.level(node) == self.addr.d();
        if traversal == "reward" {
            let rew = if read_rewards && s.reward_scale != 0.0 {
                self.reward.rewards(node, &mut self.rew_buf);
                true
            } else {
                false
            };
            for (j, &sh) in self.share_buf.iter().enumerate() {
                let mut m = if sh > 0.0 { s.share_scale * sh.ln() } else { f64::NEG_INFINITY };
                if rew {
                    m += s.reward_scale * self.rew_buf[j];
                }
                out.push(m);
            }
            return Ok(());
        }
        let pen = if read_rewards && s.penalty_scale != 0.0 {
            self.reward.penalties(node, &mut self.rew_buf);
            true
        } else {
            false
        };
        for (j, &sh) in self.share_buf.iter().enumerate() {
            let mut m = if sh > 0.0 { s.merit_scale * s.share_scale * sh.ln() } else { f64::NEG_INFINITY };
            if pen {
                m -= s.penalty_scale * self.rew_buf[j];
            }
            out.push(m);
        }
        Ok(())
    }

    /// A node's own next-outcome distribution.
    pub fn q(&mut self, node: usize, traversal: &str) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(self.addr.out);
        self.scores(node, traversal, &mut out)?;
        softmax(&mut out);
        Ok(out)
    }

    /// The next-outcome distribution of a context, floor included.
    pub fn fold(&mut self, code: &[i32], traversal: &str, backoff: &str) -> Result<Vec<f64>, String> {
        let backoff = if backoff.is_empty() { self.settings.backoff.clone() } else { backoff.to_string() };
        if !["all", "deepest", "none"].contains(&backoff.as_str()) {
            return Err(format!("backoff must be all, deepest or none, got {backoff:?}"));
        }
        if !self.addr.valid(code) {
            return Err(format!("code {code:?} does not fit radices {:?}", self.addr.radices));
        }
        let n = self.addr.out;
        let uniform = 1.0 / n as f64;
        let mut chain = std::mem::take(&mut self.chain_buf);
        self.addr.chain(code, &mut chain);
        let mut p = vec![uniform; n];
        match backoff.as_str() {
            "none" => {
                let deepest = chain[chain.len() - 1];
                if self.count.ctx[deepest] > 0 {
                    p = self.q(deepest, traversal)?;
                }
            }
            "deepest" => {
                let deepest = chain[chain.len() - 1];
                let o = self.count.own(deepest);
                let q = self.q(deepest, traversal)?;
                for j in 0..n {
                    p[j] = o * q[j] + (1.0 - o) * uniform;
                }
            }
            _ => {
                for &node in &chain {
                    let o = self.count.own(node);
                    if o == 0.0 {
                        continue;
                    }
                    let q = self.q(node, traversal)?;
                    for j in 0..n {
                        p[j] = o * q[j] + (1.0 - o) * p[j];
                    }
                }
            }
        }
        self.chain_buf = chain;
        let f = self.settings.floor;
        p.iter_mut().for_each(|v| *v = (1.0 - f) * *v + f * uniform);
        Ok(p)
    }

    /// Prices a text whose positions have the given codes.
    pub fn score_text(&mut self, codes: &Codes, outcomes: &[usize], traversal: &str, backoff: &str) -> Result<Score, String> {
        let mut out = Score::default();
        let (mut bits_sum, mut rew_sum, mut worst) = (0.0, 0.0, 0.0f64);
        for (i, &x) in outcomes.iter().enumerate() {
            let p = self.fold(codes.at(i), traversal, backoff)?;
            let bits = -p[x].log2();
            let cell = self.addr.cell(self.addr.of(codes.at(i), self.addr.d()), x);
            let (r, pen) = (self.reward.reward(cell), self.reward.penalty(cell));
            bits_sum += bits;
            rew_sum += r;
            worst = worst.max(pen);
            out.per_unit.push(UnitScore { unit: symbol(x), bits, reward: r, penalty: pen });
        }
        let n = (outcomes.len() as f64).max(1.0);
        out.bits = bits_sum / n;
        out.mean_reward = rew_sum / n;
        out.worst_penalty = worst;
        out.units = outcomes.len().saturating_sub(1);
        Ok(out)
    }

    pub fn memory_bytes(&self) -> usize {
        24 * self.addr.cells() + 8 * self.addr.n
    }

    /// Continues a prefix one outcome at a time from the exact fold: the most probable outcome (no rng),
    /// or a sample at a temperature. Stops at END, after `units` outcomes, or with to_end after max_units.
    #[allow(clippy::too_many_arguments)]
    pub fn walk(
        &mut self,
        prefix: &[u8],
        encode: &dyn Fn(&[u8]) -> Vec<i32>,
        units: usize,
        traversal: &str,
        mut rng: Option<&mut Rng>,
        temperature: f64,
        to_end: bool,
        backoff: &str,
        max_units: usize,
    ) -> Result<PathResult, String> {
        let temperature = if temperature <= 0.0 { 1.0 } else { temperature };
        let max_units = if max_units == 0 { 4096 } else { max_units };
        let mut res = PathResult { traversal: traversal.into(), mode: if rng.is_some() { "sample".into() } else { "greedy".into() }, ..Default::default() };
        let mut buf = prefix.to_vec();
        let mut step = 0;
        loop {
            if to_end {
                if step >= max_units {
                    break;
                }
            } else if step >= units {
                break;
            }
            let p = self.fold(&encode(&buf), traversal, backoff)?;
            let x = match rng.as_deref_mut() {
                None => {
                    let mut x = 0;
                    for (j, &v) in p.iter().enumerate() {
                        if v > p[x] {
                            x = j;
                        }
                    }
                    x
                }
                Some(r) => sample(&p, temperature, r),
            };
            if step == 0 {
                res.peak = p.iter().cloned().fold(0.0, f64::max);
            }
            let cost = -p[x].ln();
            res.cost += cost;
            res.step_costs.push(cost);
            res.units.push(x);
            if x == END {
                res.reached_end = true;
                break;
            }
            buf.push(x as u8);
            step += 1;
        }
        res.text = String::from_utf8_lossy(&buf[prefix.len()..]).into_owned();
        res.full_text = String::from_utf8_lossy(&buf).into_owned();
        res.bytes = buf;
        Ok(res)
    }
}

fn sample(p: &[f64], temperature: f64, rng: &mut Rng) -> usize {
    let w: Vec<f64> = p.iter().map(|&v| if v > 0.0 { v.powf(1.0 / temperature) } else { 0.0 }).collect();
    let sum: f64 = w.iter().sum();
    let r = rng.f64() * sum;
    let mut acc = 0.0;
    for (j, &v) in w.iter().enumerate() {
        acc += v;
        if r < acc {
            return j;
        }
    }
    w.iter().rposition(|&v| v > 0.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_oracle() {
        for radices in [vec![2], vec![3, 4], vec![6, 4, 4], vec![2, 3, 5, 2]] {
            let a = Address::new(&radices, OUT, 1 << 30).unwrap();
            let mut seen = vec![false; a.n];
            fn walk(a: &Address, code: Vec<i32>, seen: &mut Vec<bool>) {
                let l = code.len();
                let node = a.of(&code, l);
                assert!(!seen[node], "duplicate node {node}");
                seen[node] = true;
                assert_eq!(a.level(node), l);
                assert_eq!(a.code(node), code);
                if l > 0 {
                    assert_eq!(a.parent(node), a.of(&code, l - 1));
                }
                if l < a.radices.len() {
                    for x in 0..a.radices[l] {
                        let mut c = code.clone();
                        c.push(x as i32);
                        walk(a, c, seen);
                    }
                } else {
                    let mut chain = Vec::new();
                    a.chain(&code, &mut chain);
                    assert_eq!(chain.len(), l + 1);
                    assert_eq!(chain[0], 0);
                    assert_eq!(chain[l], node);
                }
            }
            walk(&a, vec![], &mut seen);
            assert!(seen.iter().all(|&s| s), "unreached nodes for {radices:?}");
            assert_eq!(a.parent(0), 0);
        }
        assert!(Address::new(&[16, 16, 16], OUT, 1000).is_err());
    }

    #[test]
    fn symbols_and_softmax() {
        assert_eq!(symbol(10), "\\n");
        assert_eq!(symbol(0), "\\x00");
        assert_eq!(symbol(b'a' as usize), "a");
        assert_eq!(symbol(END), "</s>");
        let mut s = vec![f64::NEG_INFINITY, 0.0, 0.0];
        softmax(&mut s);
        assert_eq!(s, vec![0.0, 0.5, 0.5]);
        let mut u = vec![f64::NEG_INFINITY; 4];
        softmax(&mut u);
        assert_eq!(u, vec![0.25; 4]);
    }
}
