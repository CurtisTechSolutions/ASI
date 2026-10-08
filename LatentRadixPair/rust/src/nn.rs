//! The numeric core: row-major f32 matrices, the three products a dense layer needs (rows split across
//! threads with rayon), SiLU, a weighted softmax cross-entropy, Adam and a small deterministic RNG.

use rayon::prelude::*;

/// A dense row-major f32 matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let c = self.cols;
        &mut self.data[i * c..(i + 1) * c]
    }
    pub fn zero(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }
    /// Makes the matrix rows x cols, reusing its storage, and zeroes it.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.data.clear();
        self.data.resize(rows * cols, 0.0);
    }
}

/// Rows below this are done on the calling thread.
const PARALLEL_MIN_ROWS: usize = 64;

fn chunk_rows(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    ((n + threads - 1) / threads).max(1)
}

/// C = A · B for A [n x k] and B [k x m].
pub fn matmul(a: &Mat, b: &Mat, c: &mut Mat) {
    assert_eq!(a.cols, b.rows, "matmul shape");
    let (n, k, m) = (a.rows, a.cols, b.cols);
    c.resize(n, m);
    let body = |rows: &mut [f32], lo: usize| {
        for (r, ci) in rows.chunks_mut(m).enumerate() {
            let ai = a.row(lo + r);
            for (p, &av) in ai.iter().enumerate().take(k) {
                if av == 0.0 {
                    continue;
                }
                let bp = &b.data[p * m..(p + 1) * m];
                for (cj, &bj) in ci.iter_mut().zip(bp) {
                    *cj += av * bj;
                }
            }
        }
    };
    if n < PARALLEL_MIN_ROWS {
        body(&mut c.data, 0);
    } else {
        let chunk = chunk_rows(n);
        c.data.par_chunks_mut(chunk * m).enumerate().for_each(|(ci, rows)| body(rows, ci * chunk));
    }
}

/// C = A · Bᵀ for A [n x k] and B [m x k].
pub fn matmul_abt(a: &Mat, b: &Mat, c: &mut Mat) {
    assert_eq!(a.cols, b.cols, "matmul_abt shape");
    let (n, k, m) = (a.rows, a.cols, b.rows);
    c.resize(n, m);
    let body = |rows: &mut [f32], lo: usize| {
        for (r, ci) in rows.chunks_mut(m).enumerate() {
            let ai = a.row(lo + r);
            for (j, cj) in ci.iter_mut().enumerate() {
                let bj = &b.data[j * k..(j + 1) * k];
                let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
                let mut p = 0;
                while p + 4 <= k {
                    s0 += ai[p] * bj[p];
                    s1 += ai[p + 1] * bj[p + 1];
                    s2 += ai[p + 2] * bj[p + 2];
                    s3 += ai[p + 3] * bj[p + 3];
                    p += 4;
                }
                while p < k {
                    s0 += ai[p] * bj[p];
                    p += 1;
                }
                *cj = s0 + s1 + s2 + s3;
            }
        }
    };
    if n < PARALLEL_MIN_ROWS {
        body(&mut c.data, 0);
    } else {
        let chunk = chunk_rows(n);
        c.data.par_chunks_mut(chunk * m).enumerate().for_each(|(ci, rows)| body(rows, ci * chunk));
    }
}

/// C += Aᵀ · B for A [n x k] and B [n x m], C [k x m].
pub fn matmul_atb_add(a: &Mat, b: &Mat, c: &mut Mat) {
    assert!(a.rows == b.rows && c.rows == a.cols && c.cols == b.cols, "matmul_atb_add shape");
    let (n, k, m) = (a.rows, a.cols, b.cols);
    let body = |rows: &mut [f32], lo: usize| {
        for (r, ci) in rows.chunks_mut(m).enumerate() {
            let i = lo + r;
            for row in 0..n {
                let av = a.data[row * k + i];
                if av == 0.0 {
                    continue;
                }
                let br = &b.data[row * m..(row + 1) * m];
                for (cj, &bj) in ci.iter_mut().zip(br) {
                    *cj += av * bj;
                }
            }
        }
    };
    if k < PARALLEL_MIN_ROWS {
        body(&mut c.data, 0);
    } else {
        let chunk = chunk_rows(k);
        c.data.par_chunks_mut(chunk * m).enumerate().for_each(|(ci, rows)| body(rows, ci * chunk));
    }
}

/// A weight matrix with its gradient.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub w: Mat,
    pub g: Mat,
}

/// y = x · W + b with W [inp x out].
#[derive(Clone, Debug)]
pub struct Dense {
    pub inp: usize,
    pub out: usize,
    pub w: Param,
    pub b: Param,
}

impl Dense {
    /// W ~ N(0, scale² · 2/inp), b = 0.
    pub fn new(name: &str, inp: usize, out: usize, rng: &mut Rng, scale: f64) -> Dense {
        let mut w = Mat::new(inp, out);
        let std = scale * (2.0 / inp as f64).sqrt();
        for v in w.data.iter_mut() {
            *v = (rng.normal() * std) as f32;
        }
        Dense {
            inp,
            out,
            w: Param { name: format!("{name}.w"), w, g: Mat::new(inp, out) },
            b: Param { name: format!("{name}.b"), w: Mat::new(1, out), g: Mat::new(1, out) },
        }
    }
    pub fn forward(&self, x: &Mat, y: &mut Mat) {
        matmul(x, &self.w.w, y);
        let b = &self.b.w.data;
        for i in 0..y.rows {
            for (v, bv) in y.row_mut(i).iter_mut().zip(b) {
                *v += bv;
            }
        }
    }
    /// dW += xᵀ · dy, db += Σ dy, and dx = dy · Wᵀ when asked.
    pub fn backward(&mut self, x: &Mat, dy: &Mat, dx: Option<&mut Mat>) {
        matmul_atb_add(x, dy, &mut self.w.g);
        for i in 0..dy.rows {
            for (g, v) in self.b.g.data.iter_mut().zip(dy.row(i)) {
                *g += v;
            }
        }
        if let Some(dx) = dx {
            matmul_abt(dy, &self.w.w, dx);
        }
    }
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w, &mut self.b]
    }
    pub fn params(&self) -> Vec<&Param> {
        vec![&self.w, &self.b]
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x as f64).exp() as f32)
}

/// y = x · σ(x).
pub fn silu(x: &Mat, y: &mut Mat) {
    y.resize(x.rows, x.cols);
    y.data.iter_mut().zip(&x.data).for_each(|(yv, &v)| *yv = v * sigmoid(v));
}

/// dx = dy · SiLU'(x), SiLU'(x) = σ(x)(1 + x(1 - σ(x))).
pub fn silu_backward(x: &Mat, dy: &Mat, dx: &mut Mat) {
    dx.resize(x.rows, x.cols);
    for i in 0..x.data.len() {
        let v = x.data[i];
        let s = sigmoid(v);
        dx.data[i] = dy.data[i] * s * (1.0 + v * (1.0 - s));
    }
}

/// Turns logits [n x c] into probabilities in place and returns the weighted mean negative log
/// probability of the targets (nats; no weights means uniform) and the argmax hits; dlogits gets the
/// gradient of that mean.
pub fn softmax_cross_entropy(logits: &mut Mat, targets: &[usize], weights: Option<&[f32]>, mut dlogits: Option<&mut Mat>) -> (f64, usize) {
    let (n, c) = (logits.rows, logits.cols);
    if let Some(d) = dlogits.as_deref_mut() {
        d.resize(n, c);
    }
    let total: f64 = match weights {
        None => n as f64,
        Some(w) => w[..n].iter().map(|&v| v as f64).sum(),
    };
    let mut loss = 0.0;
    let mut hits = 0;
    for i in 0..n {
        let mut inv = (1.0 / total) as f32;
        if let Some(w) = weights {
            inv *= w[i];
        }
        let row = logits.row_mut(i);
        let (mut top, mut arg) = (row[0], 0);
        for (j, &v) in row.iter().enumerate() {
            if v > top {
                top = v;
                arg = j;
            }
        }
        let mut sum = 0f64;
        for v in row.iter_mut() {
            let e = ((*v - top) as f64).exp();
            *v = e as f32;
            sum += e;
        }
        let t = targets[i];
        if arg == t {
            hits += 1;
        }
        for v in row.iter_mut() {
            *v /= sum as f32;
        }
        let p = (row[t] as f64).max(1e-30);
        loss -= inv as f64 * p.ln();
        if let Some(d) = dlogits.as_deref_mut() {
            let drow = d.row_mut(i);
            for (dj, &pj) in drow.iter_mut().zip(row.iter()) {
                *dj = pj * inv;
            }
            drow[t] -= inv;
        }
    }
    (loss, hits)
}

/// Adam (Kingma & Ba) with bias correction, over parameters given in a fixed order.
pub struct Adam {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: u32,
}

impl Adam {
    pub fn new(params: &[&Param]) -> Adam {
        Adam {
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            m: params.iter().map(|p| vec![0.0; p.w.data.len()]).collect(),
            v: params.iter().map(|p| vec![0.0; p.w.data.len()]).collect(),
            t: 0,
        }
    }
    /// Applies the accumulated gradients at learning rate lr and clears them.
    pub fn step(&mut self, params: &mut [&mut Param], lr: f64) {
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t as i32);
        let c2 = 1.0 - self.beta2.powi(self.t as i32);
        let (b1, b2) = (self.beta1 as f32, self.beta2 as f32);
        for (k, p) in params.iter_mut().enumerate() {
            let (m, v) = (&mut self.m[k], &mut self.v[k]);
            for i in 0..p.w.data.len() {
                let gi = p.g.data[i];
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let mh = m[i] as f64 / c1;
                let vh = v[i] as f64 / c2;
                p.w.data[i] -= (lr * mh / (vh.sqrt() + self.eps)) as f32;
                p.g.data[i] = 0.0;
            }
        }
    }
}

/// Scales the gradients down to a global norm of at most limit; returns the norm before clipping.
pub fn clip(params: &mut [&mut Param], limit: f64) -> f64 {
    let sum: f64 = params.iter().flat_map(|p| p.g.data.iter()).map(|&g| g as f64 * g as f64).sum();
    let norm = sum.sqrt();
    if limit > 0.0 && norm > limit {
        let s = (limit / norm) as f32;
        for p in params.iter_mut() {
            p.g.data.iter_mut().for_each(|g| *g *= s);
        }
    }
    norm
}

pub fn zero_grads(params: &mut [&mut Param]) {
    params.iter_mut().for_each(|p| p.g.zero());
}

/// A small deterministic generator (xorshift64*), enough for initialisation, sampling and corruption.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15, spare: None };
        if r.state == 0 {
            r.state = 0x2545_F491_4F6C_DD1D;
        }
        for _ in 0..8 {
            r.next_u64();
        }
        r
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.f64() * n as f64) as usize % n.max(1)
    }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.f64().max(1e-300), self.f64());
        let r = (-2.0 * u1.ln()).sqrt();
        let (a, b) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some(r * b);
        r * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(m: &mut Mat, rng: &mut Rng) {
        m.data.iter_mut().for_each(|v| *v = rng.normal() as f32);
    }

    #[test]
    fn products_match_the_triple_loop() {
        let mut rng = Rng::new(1);
        for (n, k, m) in [(3, 5, 7), (130, 17, 9), (70, 200, 33)] {
            let (mut a, mut b) = (Mat::new(n, k), Mat::new(k, m));
            fill(&mut a, &mut rng);
            fill(&mut b, &mut rng);
            let mut c = Mat::default();
            matmul(&a, &b, &mut c);
            let mut bt = Mat::new(m, k);
            for i in 0..k {
                for j in 0..m {
                    bt.data[j * k + i] = b.data[i * m + j];
                }
            }
            let mut c2 = Mat::default();
            matmul_abt(&a, &bt, &mut c2);
            let mut bn = Mat::new(n, m);
            fill(&mut bn, &mut rng);
            let mut c3 = Mat::new(k, m);
            matmul_atb_add(&a, &bn, &mut c3);
            for i in 0..n {
                for j in 0..m {
                    let want: f64 = (0..k).map(|p| a.data[i * k + p] as f64 * b.data[p * m + j] as f64).sum();
                    assert!((c.data[i * m + j] as f64 - want).abs() < 1e-3, "matmul {n}x{k}x{m} at {i},{j}");
                    assert!((c2.data[i * m + j] as f64 - want).abs() < 1e-3, "matmul_abt {n}x{k}x{m} at {i},{j}");
                }
            }
            for i in 0..k {
                for j in 0..m {
                    let want: f64 = (0..n).map(|r| a.data[r * k + i] as f64 * bn.data[r * m + j] as f64).sum();
                    assert!((c3.data[i * m + j] as f64 - want).abs() < 1e-3, "matmul_atb_add at {i},{j}");
                }
            }
        }
    }

    #[test]
    fn dense_gradients_match_finite_differences() {
        let mut rng = Rng::new(2);
        let mut l1 = Dense::new("l1", 5, 7, &mut rng, 1.0);
        let mut l2 = Dense::new("l2", 7, 4, &mut rng, 1.0);
        let mut x = Mat::new(6, 5);
        fill(&mut x, &mut rng);
        let targets = [0usize, 1, 2, 3, 1, 2];
        let weights = [1f32, 0.5, 2.0, 1.0, 0.25, 1.0];
        let loss = |l1: &Dense, l2: &Dense| {
            let (mut h, mut a, mut y) = (Mat::default(), Mat::default(), Mat::default());
            l1.forward(&x, &mut h);
            silu(&h, &mut a);
            l2.forward(&a, &mut y);
            softmax_cross_entropy(&mut y, &targets, Some(&weights), None).0
        };
        let (mut h, mut a, mut y, mut dy, mut da, mut dh, mut dx) =
            (Mat::default(), Mat::default(), Mat::default(), Mat::default(), Mat::default(), Mat::default(), Mat::default());
        l1.forward(&x, &mut h);
        silu(&h, &mut a);
        l2.forward(&a, &mut y);
        softmax_cross_entropy(&mut y, &targets, Some(&weights), Some(&mut dy));
        l2.backward(&a, &dy, Some(&mut da));
        silu_backward(&h, &da, &mut dh);
        l1.backward(&x, &dh, Some(&mut dx));
        let grads: Vec<(String, Vec<f32>)> =
            l1.params().into_iter().chain(l2.params()).map(|p| (p.name.clone(), p.g.data.clone())).collect();
        let mut checked = 0;
        for which in 0..4 {
            let len = grads[which].1.len();
            for i in (0..len).step_by(3) {
                let eps = 1e-2f32;
                let get = |l1: &mut Dense, l2: &mut Dense| -> *mut f32 {
                    let p: &mut Param = match which {
                        0 => &mut l1.w,
                        1 => &mut l1.b,
                        2 => &mut l2.w,
                        _ => &mut l2.b,
                    };
                    &mut p.w.data[i]
                };
                let ptr = get(&mut l1, &mut l2);
                let old = unsafe { *ptr };
                unsafe { *ptr = old + eps };
                let lp = loss(&l1, &l2);
                unsafe { *ptr = old - eps };
                let lm = loss(&l1, &l2);
                unsafe { *ptr = old };
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads[which].1[i] as f64;
                assert!((num - ana).abs() <= 1e-2 * (num.abs() + ana.abs()) + 2e-3, "{}[{i}]: analytic {ana} numeric {num}", grads[which].0);
                checked += 1;
            }
        }
        assert!(checked >= 20);
    }

    #[test]
    fn adam_finds_a_minimum() {
        let mut p = Param { name: "p".into(), w: Mat::new(1, 3), g: Mat::new(1, 3) };
        p.w.data.copy_from_slice(&[3.0, -2.0, 0.5]);
        let mut adam = Adam::new(&[&p]);
        for _ in 0..2000 {
            for i in 0..3 {
                p.g.data[i] = 2.0 * (p.w.data[i] - i as f32);
            }
            adam.step(&mut [&mut p], 0.01);
        }
        for i in 0..3 {
            assert!((p.w.data[i] - i as f32).abs() < 1e-2, "param {i} = {}", p.w.data[i]);
        }
    }

    #[test]
    fn rng_is_deterministic_and_uniform() {
        let (mut a, mut b) = (Rng::new(5), Rng::new(5));
        assert_eq!(a.next_u64(), b.next_u64());
        let mut rng = Rng::new(9);
        let mean: f64 = (0..10000).map(|_| rng.f64()).sum::<f64>() / 10000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
        let below: usize = (0..1000).map(|_| rng.below(7)).max().unwrap();
        assert!(below < 7);
    }
}
