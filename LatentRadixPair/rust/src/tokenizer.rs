//! The trained context tokenizer: a feed-forward autoencoder that compresses the last `window` bytes of
//! context into a short code of small integers (the symbols the primed tree is addressed by) and decodes a
//! code back into the bytes it stood for. See ../DESIGN.md section 2.

use crate::files;
use crate::nn::{self, Adam, Dense, Mat, Param, Rng};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// The input symbol for positions before a text starts, and the input and output alphabet size.
pub const PAD: usize = 256;
pub const SYMBOLS: usize = 257;
/// The file format name.
pub const FORMAT: &str = "latent-tokenizer";
/// How many of a window's newest bytes the tail accuracy checks.
pub const TAIL_BYTES: usize = 4;

/// The tokenizer's shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub window: usize,
    pub embed: usize,
    pub enc_hidden: usize,
    pub dec_hidden: usize,
    pub out_embed: usize,
    pub levels: Vec<Vec<usize>>,
    pub noise: f64,
    pub recency: f64,
    pub predict: f64,
    pub start_share: f64,
}

impl Default for Config {
    /// 16 bytes of context into three symbols of radix 16, the newest byte's reconstruction weighing 1 and
    /// each older byte 0.6 of the next, predicting the next byte weighing four times the reconstruction.
    fn default() -> Config {
        Config {
            window: 16,
            embed: 16,
            enc_hidden: 256,
            dec_hidden: 256,
            out_embed: 32,
            levels: vec![vec![4, 4], vec![4, 4], vec![4, 4]],
            noise: 0.1,
            recency: 0.6,
            predict: 4.0,
            start_share: 0.05,
        }
    }
}

impl Config {
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
    pub fn dims(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }
    pub fn radix(&self, s: usize) -> usize {
        self.levels[s].iter().product()
    }
    pub fn radices(&self) -> Vec<usize> {
        (0..self.depth()).map(|s| self.radix(s)).collect()
    }
    pub fn bits(&self) -> f64 {
        (0..self.depth()).map(|s| (self.radix(s) as f64).log2()).sum()
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.window < 1 || self.embed < 1 || self.enc_hidden < 1 || self.dec_hidden < 1 || self.out_embed < 1 {
            return Err("window, embed, enc_hidden, dec_hidden and out_embed must be positive".into());
        }
        if self.levels.is_empty() {
            return Err("at least one code symbol is needed".into());
        }
        for (s, dims) in self.levels.iter().enumerate() {
            if dims.is_empty() {
                return Err(format!("symbol {s} has no dimensions"));
            }
            if dims.iter().any(|&l| l < 2) {
                return Err(format!("symbol {s}: every dimension needs at least 2 levels"));
            }
        }
        if self.noise < 0.0 || self.noise >= 1.0 || self.start_share < 0.0 || self.start_share > 1.0 {
            return Err("0 <= noise < 1 and 0 <= start_share <= 1".into());
        }
        if self.recency <= 0.0 || self.recency > 1.0 {
            return Err("0 < recency <= 1".into());
        }
        if self.predict < 0.0 {
            return Err("predict must be >= 0".into());
        }
        Ok(())
    }
    /// Reconstruction weights of the window's positions, oldest first, mean 1.
    pub fn weights(&self) -> Vec<f32> {
        let mut w: Vec<f64> = (0..self.window).map(|p| self.recency.powi((self.window - 1 - p) as i32)).collect();
        let sum: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v *= self.window as f64 / sum);
        w.into_iter().map(|v| v as f32).collect()
    }
}

/// Reads "4,4:4,4:4,4" (symbols by ':', dimensions by ',') or "16:16:16".
pub fn parse_levels(text: &str) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::new();
    for sym in text.trim().split(':') {
        let mut dims = Vec::new();
        for d in sym.split(',') {
            let v: usize = d.trim().parse().map_err(|_| format!("levels: {d:?} is not a level count of at least 2"))?;
            if v < 2 {
                return Err(format!("levels: {d:?} is not a level count of at least 2"));
            }
            dims.push(v);
        }
        out.push(dims);
    }
    Ok(out)
}

/// The inverse of parse_levels.
pub fn levels_string(levels: &[Vec<usize>]) -> String {
    levels
        .iter()
        .map(|d| d.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join(":")
}

/// One line of the training log.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stat {
    pub step: usize,
    pub loss_bits: f64,
    #[serde(default)]
    pub next_bits: Vec<f64>,
    pub accuracy: Vec<f64>,
    #[serde(default)]
    pub tail: Vec<f64>,
    pub train_bits: f64,
    pub lr: f64,
    pub seconds: f64,
}

/// Training options.
#[derive(Clone, Debug)]
pub struct Options {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub seed: u64,
    pub eval_every: usize,
    pub eval_windows: usize,
    pub clip: f64,
    pub quantize: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { steps: 2000, batch: 256, lr: 3e-3, seed: 1, eval_every: 100, eval_windows: 512, clip: 5.0, quantize: true }
    }
}

/// The codes of consecutive positions, `d` symbols each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Codes {
    pub d: usize,
    pub data: Vec<i32>,
}

impl Codes {
    pub fn len(&self) -> usize {
        if self.d == 0 {
            0
        } else {
            self.data.len() / self.d
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn at(&self, i: usize) -> &[i32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

/// The trained encoder and decoder.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub config: Config,
    pub seed: i64,
    pub steps: usize,
    pub corpus: usize,
    pub stats: Vec<Stat>,
    pub emb: Param,
    pub enc1: Dense,
    pub enc2: Dense,
    pub enc_out: Dense,
    pub dec1: Dense,
    pub dec2: Dense,
    pub dec_out: Dense,
    pub unembed: Dense,
    pub next_head: Dense,
}

/// One batch's activations and gradients, reused across steps.
#[derive(Default)]
struct Act {
    n: usize,
    windows: Vec<i32>,
    targets: Vec<usize>,
    next: Vec<usize>,
    depth: Vec<usize>,
    weights: Vec<f32>,
    codes: Vec<i32>,
    x: Mat,
    h1: Mat,
    a1: Mat,
    h2: Mat,
    a2: Mat,
    z: Mat,
    th: Mat,
    q: Mat,
    u: Mat,
    g1: Mat,
    b1: Mat,
    g2: Mat,
    b2: Mat,
    o: Mat,
    logits: Mat,
    next_logits: Mat,
    dlogits: Mat,
    dnext: Mat,
    dnb2: Mat,
    do_: Mat,
    db2: Mat,
    dg2: Mat,
    db1: Mat,
    dg1: Mat,
    du: Mat,
    dz: Mat,
    da2: Mat,
    dh2: Mat,
    da1: Mat,
    dh1: Mat,
    dx: Mat,
}

/// The `window` input symbols before position `pos` of `text` (PAD before its start).
fn window(text: &[u8], pos: usize, w: usize, out: &mut [i32]) {
    for p in 0..w {
        let i = pos as isize - w as isize + p as isize;
        out[p] = if i < 0 { PAD as i32 } else { text[i as usize] as i32 };
    }
}

impl Tokenizer {
    pub fn new(config: Config, seed: i64) -> Result<Tokenizer, String> {
        config.validate()?;
        let mut rng = Rng::new(seed as u64);
        let mut emb = Mat::new(SYMBOLS, config.embed);
        for v in emb.data.iter_mut() {
            *v = (rng.normal() * 0.5) as f32;
        }
        let c = &config;
        let t = Tokenizer {
            emb: Param { name: "emb".into(), w: emb, g: Mat::new(SYMBOLS, c.embed) },
            enc1: Dense::new("enc1", c.window * c.embed, c.enc_hidden, &mut rng, 1.0),
            enc2: Dense::new("enc2", c.enc_hidden, c.enc_hidden, &mut rng, 1.0),
            enc_out: Dense::new("enc_out", c.enc_hidden, c.dims(), &mut rng, 1.0),
            dec1: Dense::new("dec1", c.dims() + c.depth(), c.dec_hidden, &mut rng, 1.0),
            dec2: Dense::new("dec2", c.dec_hidden, c.dec_hidden, &mut rng, 1.0),
            dec_out: Dense::new("dec_out", c.dec_hidden, c.window * c.out_embed, &mut rng, 1.0),
            unembed: Dense::new("unembed", c.out_embed, SYMBOLS, &mut rng, 1.0),
            next_head: Dense::new("next", c.dec_hidden, SYMBOLS, &mut rng, 1.0),
            config,
            seed,
            steps: 0,
            corpus: 0,
            stats: Vec::new(),
        };
        Ok(t)
    }

    pub fn depth(&self) -> usize {
        self.config.depth()
    }
    pub fn radices(&self) -> Vec<usize> {
        self.config.radices()
    }

    /// Every trainable matrix, in a fixed order.
    pub fn params(&self) -> Vec<&Param> {
        let mut out = vec![&self.emb];
        for d in [&self.enc1, &self.enc2, &self.enc_out, &self.dec1, &self.dec2, &self.dec_out, &self.unembed, &self.next_head] {
            out.extend(d.params());
        }
        out
    }
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut out = vec![&mut self.emb];
        for d in [
            &mut self.enc1,
            &mut self.enc2,
            &mut self.enc_out,
            &mut self.dec1,
            &mut self.dec2,
            &mut self.dec_out,
            &mut self.unembed,
            &mut self.next_head,
        ] {
            out.extend(d.params_mut());
        }
        out
    }
    pub fn param_count(&self) -> usize {
        self.params().iter().map(|p| p.w.data.len()).sum()
    }

    fn embed(&self, a: &mut Act) {
        let (w, e) = (self.config.window, self.config.embed);
        a.x.resize(a.n, w * e);
        for i in 0..a.n {
            let row = a.x.row_mut(i);
            for p in 0..w {
                let s = a.windows[i * w + p] as usize;
                row[p * e..(p + 1) * e].copy_from_slice(&self.emb.w.data[s * e..(s + 1) * e]);
            }
        }
    }

    /// Runs the encoder on a.x and quantises into a.th (tanh), a.q (quantised, or tanh when quantize is
    /// false) and a.codes.
    fn encode_act(&self, a: &mut Act, quantize: bool) {
        self.enc1.forward(&a.x, &mut a.h1);
        nn::silu(&a.h1, &mut a.a1);
        self.enc2.forward(&a.a1, &mut a.h2);
        nn::silu(&a.h2, &mut a.a2);
        self.enc_out.forward(&a.a2, &mut a.z);
        let (d, dims) = (self.depth(), self.config.dims());
        a.th.resize(a.n, dims);
        a.q.resize(a.n, dims);
        a.codes.clear();
        a.codes.resize(a.n * d, 0);
        for i in 0..a.n {
            let mut j = 0;
            for (s, levels) in self.config.levels.iter().enumerate() {
                let (mut id, mut mul) = (0usize, 1usize);
                for &l in levels {
                    let v = (a.z.data[i * dims + j] as f64).tanh() as f32;
                    a.th.data[i * dims + j] = v;
                    let scaled = (v + 1.0) / 2.0 * (l - 1) as f32;
                    let r = ((scaled as f64 + 0.5).floor() as i64).clamp(0, l as i64 - 1) as usize;
                    a.q.data[i * dims + j] = if quantize { r as f32 / (l - 1) as f32 * 2.0 - 1.0 } else { v };
                    id += r * mul;
                    mul *= l;
                    j += 1;
                }
                a.codes[i * d + s] = id as i32;
            }
        }
    }

    /// Builds a.u from a.q and a.depth: a symbol's dimensions when visible, else zeros, plus a flag.
    fn decoder_input(&self, a: &mut Act) {
        let (d, dims) = (self.depth(), self.config.dims());
        a.u.resize(a.n, dims + d);
        for i in 0..a.n {
            let mut j = 0;
            for (s, levels) in self.config.levels.iter().enumerate() {
                if s < a.depth[i] {
                    for k in 0..levels.len() {
                        a.u.data[i * (dims + d) + j + k] = a.q.data[i * dims + j + k];
                    }
                    a.u.data[i * (dims + d) + dims + s] = 1.0;
                }
                j += levels.len();
            }
        }
    }

    /// Runs the decoder on a.u into a.logits [n*window x SYMBOLS] and a.next_logits.
    fn decode_act(&self, a: &mut Act) {
        self.dec1.forward(&a.u, &mut a.g1);
        nn::silu(&a.g1, &mut a.b1);
        self.dec2.forward(&a.b1, &mut a.g2);
        nn::silu(&a.g2, &mut a.b2);
        self.dec_out.forward(&a.b2, &mut a.o);
        a.o.rows = a.n * self.config.window; // the same numbers, one row per position
        a.o.cols = self.config.out_embed;
        self.unembed.forward(&a.o, &mut a.logits);
        self.next_head.forward(&a.b2, &mut a.next_logits);
    }

    /// The whole backward pass from a.dlogits and a.dnext, accumulating parameter gradients.
    fn backward(&mut self, a: &mut Act) {
        let (w, eo, e) = (self.config.window, self.config.out_embed, self.config.embed);
        self.unembed.backward(&a.o, &a.dlogits, Some(&mut a.do_));
        a.do_.rows = a.n; // back to one row per window
        a.do_.cols = w * eo;
        self.dec_out.backward(&a.b2, &a.do_, Some(&mut a.db2));
        if self.config.predict > 0.0 {
            self.next_head.backward(&a.b2, &a.dnext, Some(&mut a.dnb2));
            for (d, v) in a.db2.data.iter_mut().zip(&a.dnb2.data) {
                *d += v;
            }
        }
        nn::silu_backward(&a.g2, &a.db2, &mut a.dg2);
        self.dec2.backward(&a.b1, &a.dg2, Some(&mut a.db1));
        nn::silu_backward(&a.g1, &a.db1, &mut a.dg1);
        self.dec1.backward(&a.u, &a.dg1, Some(&mut a.du));
        let (d, dims) = (self.depth(), self.config.dims());
        a.dz.resize(a.n, dims);
        for i in 0..a.n {
            let mut j = 0;
            for (s, levels) in self.config.levels.iter().enumerate() {
                for _ in levels {
                    if s < a.depth[i] {
                        let th = a.th.data[i * dims + j];
                        a.dz.data[i * dims + j] = a.du.data[i * (dims + d) + j] * (1.0 - th * th);
                    }
                    j += 1;
                }
            }
        }
        self.enc_out.backward(&a.a2, &a.dz, Some(&mut a.da2));
        nn::silu_backward(&a.h2, &a.da2, &mut a.dh2);
        self.enc2.backward(&a.a1, &a.dh2, Some(&mut a.da1));
        nn::silu_backward(&a.h1, &a.da1, &mut a.dh1);
        self.enc1.backward(&a.x, &a.dh1, Some(&mut a.dx));
        for i in 0..a.n {
            for p in 0..w {
                let s = a.windows[i * w + p] as usize;
                let src = &a.dx.data[i * w * e + p * e..i * w * e + (p + 1) * e];
                for (g, v) in self.emb.g.data[s * e..(s + 1) * e].iter_mut().zip(src) {
                    *g += v;
                }
            }
        }
    }

    /// A forward pass on a batch: returns the loss (nats: recency-weighted reconstruction plus predict
    /// times the next byte's), the exact-byte hits and the next byte's loss alone; with grad, fills the
    /// gradients of the logits.
    fn forward_loss(&self, a: &mut Act, quantize: bool, grad: bool) -> (f64, usize, f64) {
        self.embed(a);
        self.encode_act(a, quantize);
        self.decoder_input(a);
        self.decode_act(a);
        let w = self.config.window;
        if a.weights.len() != a.n * w {
            let ws = self.config.weights();
            a.weights = (0..a.n).flat_map(|_| ws.iter().copied()).collect();
        }
        let (loss, hits) =
            nn::softmax_cross_entropy(&mut a.logits, &a.targets, Some(&a.weights), if grad { Some(&mut a.dlogits) } else { None });
        let (next, _) = nn::softmax_cross_entropy(&mut a.next_logits, &a.next, None, if grad { Some(&mut a.dnext) } else { None });
        if grad && self.config.predict > 0.0 && self.config.predict != 1.0 {
            let s = self.config.predict as f32;
            a.dnext.data.iter_mut().for_each(|v| *v *= s);
        }
        (loss + self.config.predict * next, hits, next)
    }

    /// After forward_loss: the rows whose last `tail` bytes were all reconstructed exactly.
    fn tail_hits(&self, a: &Act, tail: usize) -> usize {
        let w = self.config.window;
        (0..a.n)
            .filter(|&i| {
                (w - tail..w).all(|p| {
                    let row = a.logits.row(i * w + p);
                    let mut arg = 0;
                    for (j, &v) in row.iter().enumerate() {
                        if v > row[arg] {
                            arg = j;
                        }
                    }
                    arg == a.targets[i * w + p]
                })
            })
            .count()
    }

    /// The code of every position 0..=len(text): the context before each byte and after the last one.
    pub fn encode_all(&self, text: &[u8]) -> Codes {
        const CHUNK: usize = 4096;
        let n = text.len() + 1;
        let (w, d) = (self.config.window, self.depth());
        let mut out = Codes { d, data: vec![0; n * d] };
        let mut a = Act::default();
        let mut lo = 0;
        while lo < n {
            let hi = (lo + CHUNK).min(n);
            a.n = hi - lo;
            a.windows.clear();
            a.windows.resize(a.n * w, 0);
            for i in lo..hi {
                window(text, i, w, &mut a.windows[(i - lo) * w..(i - lo + 1) * w]);
            }
            self.embed(&mut a);
            self.encode_act(&mut a, true);
            out.data[lo * d..hi * d].copy_from_slice(&a.codes);
            lo = hi;
        }
        out
    }

    /// The code of the context that ends a text (its last `window` bytes).
    pub fn encode(&self, context: &[u8]) -> Vec<i32> {
        let mut a = Act::default();
        a.n = 1;
        a.windows = vec![0; self.config.window];
        window(context, context.len(), self.config.window, &mut a.windows);
        self.embed(&mut a);
        self.encode_act(&mut a, true);
        a.codes
    }

    /// The unquantised, tanh-bounded latent of a context.
    pub fn latent(&self, context: &[u8]) -> Vec<f32> {
        let mut a = Act::default();
        a.n = 1;
        a.windows = vec![0; self.config.window];
        window(context, context.len(), self.config.window, &mut a.windows);
        self.embed(&mut a);
        self.encode_act(&mut a, false);
        a.th.data
    }

    /// The grid values of a code: the decoder's view of it.
    fn quantised(&self, code: &[i32], q: &mut [f32]) {
        let mut j = 0;
        for (s, levels) in self.config.levels.iter().enumerate() {
            let mut id = code[s] as usize;
            for &l in levels {
                let r = id % l;
                id /= l;
                q[j] = r as f32 / (l - 1) as f32 * 2.0 - 1.0;
                j += 1;
            }
        }
    }

    /// Reconstructs the window a code stands for from its first `known` symbols (0 or more than the depth
    /// means all): `window` symbols, bytes or PAD.
    pub fn decode_symbols(&self, code: &[i32], known: usize) -> Result<Vec<usize>, String> {
        let d = self.depth();
        if code.len() != d {
            return Err(format!("a code has {d} symbols, got {}", code.len()));
        }
        for (s, &c) in code.iter().enumerate() {
            if c < 0 || c as usize >= self.config.radix(s) {
                return Err(format!("symbol {s} is {c}, outside 0..{}", self.config.radix(s) - 1));
            }
        }
        let known = if known == 0 || known > d { d } else { known };
        let mut a = Act::default();
        a.n = 1;
        a.q.resize(1, self.config.dims());
        let mut q = vec![0f32; self.config.dims()];
        self.quantised(code, &mut q);
        a.q.data.copy_from_slice(&q);
        a.depth = vec![known];
        self.decoder_input(&mut a);
        self.decode_act(&mut a);
        Ok((0..self.config.window)
            .map(|p| {
                let row = a.logits.row(p);
                let mut arg = 0;
                for (j, &v) in row.iter().enumerate() {
                    if v > row[arg] {
                        arg = j;
                    }
                }
                arg
            })
            .collect())
    }

    /// decode_symbols rendered as text, PAD as a middle dot.
    pub fn decode(&self, code: &[i32], known: usize) -> Result<String, String> {
        Ok(render(&self.decode_symbols(code, known)?))
    }

    /// Fits the tokenizer to texts; `log` sees every evaluation line.
    pub fn train(&mut self, texts: &[Vec<u8>], o: &Options, mut log: impl FnMut(&Stat)) -> Result<Vec<Stat>, String> {
        if o.steps == 0 || o.batch == 0 || o.lr <= 0.0 {
            return Err("steps, batch and lr must be positive".into());
        }
        let total: usize = texts.iter().map(|t| t.len()).sum();
        if total == 0 {
            return Err("no training text".into());
        }
        let d = self.depth();
        let w = self.config.window;
        let mut rng = Rng::new(o.seed);
        let mut train = Sampler::new(texts, w, self.config.start_share, Rng::new(o.seed ^ 0xA5A5));
        let mut eval = Sampler::new(texts, w, self.config.start_share, Rng::new(o.seed.wrapping_add(7919)));
        let eval_n = if o.eval_windows == 0 { 256 } else { o.eval_windows };
        let mut eval_act = Act::default();
        eval.fill(&mut eval_act, eval_n, 0.0, d, false);
        let eval_targets = eval_act.targets.clone();
        let tail = TAIL_BYTES.min(w);
        let mut adam = Adam::new(&self.params());
        let mut a = Act::default();
        let warm = (o.steps / 20).max(10);
        let mut stats = Vec::new();
        let started = Instant::now();
        let mut run_loss = 0.0;
        for step in 1..=o.steps {
            let lr = if step <= warm {
                o.lr * step as f64 / warm as f64
            } else {
                let progress = (step - warm) as f64 / (o.steps - warm + 1) as f64;
                o.lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
            };
            let _ = &mut rng;
            train.fill(&mut a, o.batch, self.config.noise, d, true);
            let (loss, _, _) = self.forward_loss(&mut a, o.quantize, true);
            self.backward(&mut a);
            let mut params = self.params_mut();
            nn::clip(&mut params, o.clip);
            adam.step(&mut params, lr);
            self.steps += 1;
            run_loss = if run_loss == 0.0 { loss } else { 0.95 * run_loss + 0.05 * loss };
            if (o.eval_every > 0 && step % o.eval_every == 0) || step == o.steps {
                let (mut acc, mut tails, mut nexts) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
                let mut eval_loss = 0.0;
                for k in 1..=d {
                    eval_act.depth.iter_mut().for_each(|v| *v = k);
                    eval_act.targets.copy_from_slice(&eval_targets);
                    let (l, hits, next) = self.forward_loss(&mut eval_act, o.quantize, false);
                    acc[k - 1] = hits as f64 / (eval_n * w) as f64;
                    tails[k - 1] = self.tail_hits(&eval_act, tail) as f64 / eval_n as f64;
                    nexts[k - 1] = next / std::f64::consts::LN_2;
                    if k == d {
                        eval_loss = (l - self.config.predict * next) / std::f64::consts::LN_2;
                    }
                }
                let s = Stat {
                    step: self.steps,
                    loss_bits: eval_loss,
                    next_bits: nexts,
                    accuracy: acc,
                    tail: tails,
                    train_bits: run_loss / std::f64::consts::LN_2,
                    lr,
                    seconds: started.elapsed().as_secs_f64(),
                };
                log(&s);
                stats.push(s);
            }
        }
        self.corpus += total;
        self.stats.extend(stats.iter().cloned());
        Ok(stats)
    }

    /// Compares the analytic gradient with central finite differences on a tiny batch without the
    /// rounding; returns the largest relative error found.
    pub fn grad_check(&mut self, texts: &[Vec<u8>], seed: u64, eps: f64) -> f64 {
        let mut s = Sampler::new(texts, self.config.window, self.config.start_share, Rng::new(seed));
        let mut a = Act::default();
        s.fill(&mut a, 4, 0.0, self.depth(), false);
        let targets = a.targets.clone();
        {
            let mut params = self.params_mut();
            nn::zero_grads(&mut params);
        }
        a.targets.copy_from_slice(&targets);
        self.forward_loss(&mut a, false, true);
        self.backward(&mut a);
        let grads: Vec<Vec<f32>> = self.params().iter().map(|p| p.g.data.clone()).collect();
        let count = grads.len();
        let mut worst = 0.0f64;
        for k in 0..count {
            let len = grads[k].len();
            let stride = len / 12 + 1;
            let mut i = 0;
            while i < len {
                let old = self.params_mut()[k].w.data[i];
                self.params_mut()[k].w.data[i] = old + eps as f32;
                a.targets.copy_from_slice(&targets);
                let lp = self.forward_loss(&mut a, false, false).0;
                self.params_mut()[k].w.data[i] = old - eps as f32;
                a.targets.copy_from_slice(&targets);
                let lm = self.forward_loss(&mut a, false, false).0;
                self.params_mut()[k].w.data[i] = old;
                let num = (lp - lm) / (2.0 * eps);
                let ana = grads[k][i] as f64;
                let rel = (num - ana).abs() / (num.abs() + ana.abs() + 1e-2);
                worst = worst.max(rel);
                i += stride;
            }
        }
        worst
    }

    /// Describes the tokenizer.
    pub fn info(&self) -> serde_json::Value {
        let last = self.stats.last().cloned().unwrap_or_default();
        serde_json::json!({
            "window": self.config.window, "depth": self.depth(), "radices": self.radices(),
            "levels": levels_string(&self.config.levels), "code_bits": self.config.bits(),
            "params": self.param_count(), "steps": self.steps, "corpus_bytes": self.corpus, "seed": self.seed,
            "loss_bits": last.loss_bits, "next_bits": last.next_bits, "accuracy": last.accuracy,
            "tail_accuracy": last.tail, "recency": self.config.recency, "predict": self.config.predict,
        })
    }

    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        let mut weights = BTreeMap::new();
        for p in self.params() {
            weights.insert(p.name.clone(), encode_mat(&p.w));
        }
        let f = FileJson {
            format: FORMAT.into(),
            version: 1,
            config: self.config.clone(),
            seed: self.seed,
            steps: self.steps,
            corpus_bytes: self.corpus,
            stats: self.stats.clone(),
            weights,
        };
        serde_json::to_vec(&f).map_err(|e| e.to_string())
    }

    pub fn from_json(data: &[u8]) -> Result<Tokenizer, String> {
        let f: FileJson = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        if f.format != FORMAT {
            return Err(format!("not a {FORMAT} file (format {:?})", f.format));
        }
        let mut t = Tokenizer::new(f.config, f.seed)?;
        t.steps = f.steps;
        t.corpus = f.corpus_bytes;
        t.stats = f.stats;
        for p in t.params_mut() {
            let w = f.weights.get(&p.name).ok_or_else(|| format!("weight {} missing", p.name))?;
            decode_mat(w, &mut p.w).map_err(|e| format!("weight {}: {e}", p.name))?;
        }
        Ok(t)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        files::write_file(path, &self.to_json()?).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<Tokenizer, String> {
        let data = files::read_file(path).map_err(|e| format!("{path}: {e}"))?;
        Tokenizer::from_json(&data)
    }
}

/// Window symbols as text, PAD as a middle dot.
pub fn render(syms: &[usize]) -> String {
    let mut out = Vec::with_capacity(syms.len());
    for &s in syms {
        if s == PAD {
            out.extend_from_slice("\u{b7}".as_bytes());
        } else {
            out.push(s as u8);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Draws training windows from texts, each weighted by its length.
struct Sampler<'a> {
    texts: Vec<&'a [u8]>,
    cum: Vec<usize>,
    total: usize,
    w: usize,
    rng: Rng,
    starts: f64,
}

impl<'a> Sampler<'a> {
    fn new(texts: &'a [Vec<u8>], w: usize, starts: f64, rng: Rng) -> Sampler<'a> {
        let mut s = Sampler { texts: Vec::new(), cum: Vec::new(), total: 0, w, rng, starts };
        for t in texts {
            if t.is_empty() {
                continue;
            }
            s.texts.push(t);
            s.total += t.len() + 1;
            s.cum.push(s.total);
        }
        s
    }
    fn draw(&mut self) -> (&'a [u8], usize) {
        let r = self.rng.below(self.total);
        let mut k = 0;
        while self.cum[k] <= r {
            k += 1;
        }
        let text = self.texts[k];
        if self.rng.f64() < self.starts {
            let lim = self.w.min(text.len() + 1);
            return (text, self.rng.below(lim));
        }
        (text, self.rng.below(text.len() + 1))
    }
    fn fill(&mut self, a: &mut Act, n: usize, noise: f64, d: usize, corrupt: bool) {
        let w = self.w;
        a.n = n;
        a.windows.clear();
        a.windows.resize(n * w, 0);
        a.targets.clear();
        a.targets.resize(n * w, 0);
        a.depth.clear();
        a.depth.resize(n, 0);
        a.next.clear();
        a.next.resize(n, 0);
        for i in 0..n {
            let (text, pos) = self.draw();
            window(text, pos, w, &mut a.windows[i * w..(i + 1) * w]);
            for p in 0..w {
                a.targets[i * w + p] = a.windows[i * w + p] as usize;
            }
            a.next[i] = if pos < text.len() { text[pos] as usize } else { PAD };
            if corrupt && noise > 0.0 {
                let level = self.rng.f64() * noise;
                for p in 0..w {
                    if a.windows[i * w + p] != PAD as i32 && self.rng.f64() < level {
                        a.windows[i * w + p] = self.rng.below(256) as i32;
                    }
                }
            }
            a.depth[i] = 1 + self.rng.below(d);
        }
    }
}

#[derive(Serialize, Deserialize)]
struct WeightJson {
    rows: usize,
    cols: usize,
    data: String,
}

#[derive(Serialize, Deserialize)]
struct FileJson {
    format: String,
    version: u32,
    config: Config,
    seed: i64,
    steps: usize,
    corpus_bytes: usize,
    #[serde(default)]
    stats: Vec<Stat>,
    weights: BTreeMap<String, WeightJson>,
}

fn encode_mat(m: &Mat) -> WeightJson {
    let mut buf = Vec::with_capacity(4 * m.data.len());
    for v in &m.data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    WeightJson { rows: m.rows, cols: m.cols, data: base64::engine::general_purpose::STANDARD.encode(buf) }
}

fn decode_mat(w: &WeightJson, into: &mut Mat) -> Result<(), String> {
    let buf = base64::engine::general_purpose::STANDARD.decode(&w.data).map_err(|e| e.to_string())?;
    if w.rows != into.rows || w.cols != into.cols || buf.len() != 4 * into.data.len() {
        return Err(format!("weight is {}x{} ({} bytes), want {}x{}", w.rows, w.cols, buf.len(), into.rows, into.cols));
    }
    for (i, v) in into.data.iter_mut().enumerate() {
        *v = f32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Config {
        Config {
            window: 6,
            embed: 4,
            enc_hidden: 12,
            dec_hidden: 12,
            out_embed: 6,
            levels: vec![vec![3, 3], vec![4], vec![2, 2]],
            noise: 0.1,
            recency: 0.7,
            predict: 0.5,
            start_share: 0.1,
        }
    }

    fn corpus() -> Vec<Vec<u8>> {
        vec![
            b"the cat sat on the mat. the cat sat on the log. the dog ate the bone. a cat and a dog.".to_vec(),
            b"the bird sang on the wire. a dog and a cat sat. the mat was on the floor. the log was on the fire.".to_vec(),
            vec![0, 1, 2, 3, 255, 254, 253, 128, 127, 10, 13, 9, 0, 0, 0, 200],
        ]
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut tok = Tokenizer::new(tiny(), 5).unwrap();
        let worst = tok.grad_check(&corpus(), 3, 1e-2);
        assert!(worst <= 2e-2, "worst relative gradient error {worst}");
    }

    #[test]
    fn codes_are_in_range_and_deterministic() {
        let tok = Tokenizer::new(tiny(), 1).unwrap();
        for text in corpus() {
            let codes = tok.encode_all(&text);
            assert_eq!(codes.len(), text.len() + 1);
            assert_eq!(codes.d, 3);
            for i in 0..codes.len() {
                let c = codes.at(i);
                for (s, &v) in c.iter().enumerate() {
                    assert!(v >= 0 && (v as usize) < tok.config.radix(s));
                }
                assert_eq!(tok.encode(&text[..i]), c, "position {i}");
            }
        }
        assert_eq!(tok.radices(), vec![9, 4, 4]);
        assert!((tok.config.bits() - 7.17).abs() < 0.01);
    }

    #[test]
    fn training_lowers_the_loss_and_decodes() {
        let mut tok = Tokenizer::new(tiny(), 2).unwrap();
        let o = Options { steps: 150, batch: 32, eval_every: 50, eval_windows: 64, ..Options::default() };
        let stats = tok.train(&corpus(), &o, |_| {}).unwrap();
        assert_eq!(stats.len(), 3);
        assert_eq!(tok.steps, 150);
        let (first, last) = (&stats[0], &stats[2]);
        assert!(last.loss_bits < first.loss_bits && last.loss_bits < 7.5, "loss {} -> {}", first.loss_bits, last.loss_bits);
        assert!(last.next_bits[2] < first.next_bits[2], "next bits did not fall");
        assert!(last.accuracy[2] >= last.accuracy[0]);
        let w = tok.config.weights();
        assert!(w[5] > w[0] && w[5] >= 1.0);
        let code = tok.encode(b"the cat sat on the ");
        for known in 1..=3 {
            let syms = tok.decode_symbols(&code, known).unwrap();
            assert_eq!(syms.len(), 6);
            assert_eq!(tok.decode(&code, known).unwrap().chars().count(), 6);
        }
        assert!(tok.decode_symbols(&[99, 0, 0], 0).is_err());
        assert!(render(&[PAD, b'a' as usize]).contains("\u{b7}a"));
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut tok = Tokenizer::new(tiny(), 9).unwrap();
        let o = Options { steps: 20, batch: 16, eval_every: 10, eval_windows: 32, ..Options::default() };
        tok.train(&corpus(), &o, |_| {}).unwrap();
        let dir = std::env::temp_dir().join(format!("latentpair-tok-{}", std::process::id()));
        let path = dir.join("tok.json.gz");
        tok.save(path.to_str().unwrap()).unwrap();
        let back = Tokenizer::load(path.to_str().unwrap()).unwrap();
        assert_eq!(back.steps, 20);
        assert_eq!(back.corpus, tok.corpus);
        assert_eq!(back.stats.len(), 2);
        for text in corpus() {
            assert_eq!(tok.encode_all(&text), back.encode_all(&text));
        }
        for (p, q) in tok.params().iter().zip(back.params()) {
            assert_eq!(p.w.data, q.w.data, "weight {}", p.name);
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn levels_parse_and_print() {
        for s in ["4,4:4,4:4,4", "16:16:16", "8:5,5"] {
            assert_eq!(levels_string(&parse_levels(s).unwrap()), s);
        }
        assert!(parse_levels("4:1").is_err());
        assert!(Config { levels: vec![], ..Config::default() }.validate().is_err());
    }
}
