//! The radix model's training backend: the one-hop local learning rule over a
//! CSR export of the graph.
//!
//! For a transition `p -> c` observed in training the loss is the negative
//! log-softmax of the edge score `w * f_p(z_p) * f_c(z_c)` among all children
//! of `p`.  Gradients touch only `w` on `p`'s out-edges, the states `z` and the
//! activation parameters `(a, b, h, k)` of `p` and its children - nothing
//! propagates deeper, so the vanishing gradient simply does not arise.
//!
//! This is `PythonBackend` of `radixnet/backend.py`, not a reinterpretation of
//! it: the gradients are accumulated in the same order, cached per node and
//! batch the same way, scaled, clipped element-wise and applied once per batch,
//! and `b` is kept `>= MIN_B`.  Every sum runs left to right as Python's does,
//! so a model trained here and one trained in Python hold the same doubles.
//!
//! Python has a second backend, `backend_torch.py`, which runs the same rule on
//! a GPU.  It is deliberately not here: it needs torch, and a crate with no
//! dependencies cannot have it.  [`describe_backends`] says so rather than
//! pretending.

use crate::json::Json;

/// The lower bound applied to every node's `b` after each update.
pub const MIN_B: f64 = 1e-3;

/// What this port calls its backend in `stats`, the file and `/api/status`.
pub const NAME: &str = "rust";
/// The one device it runs on.
pub const DEVICE: &str = "cpu";

/// The graph's edges in compressed-sparse-row form, node-id order.
///
/// `indptr[p]..indptr[p + 1]` are `p`'s out-edges; `indices[j]` is the child,
/// `edge_ids[j]` the graph's edge id and `weights[j]` its weight.
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub edge_ids: Vec<usize>,
    pub weights: Vec<f64>,
    /// Edge id -> CSR position (`usize::MAX` for an edge not in the CSR).
    pub edge_pos: Vec<usize>,
}

/// The per-node learnable values in node-id order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeParams {
    pub z: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub h: Vec<f64>,
    pub k: Vec<f64>,
}

/// The mutable state of one epoch: the weights, the node parameters and the
/// gradient accumulators, which stay zero between steps (only what a batch
/// touched is written and reset, so a step costs `O(batch * fan-out)`).
pub struct State {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    pub weights: Vec<f64>,
    pub params: NodeParams,
    gw: Vec<f64>,
    gz: Vec<f64>,
    ga: Vec<f64>,
    gb: Vec<f64>,
    gh: Vec<f64>,
    gk: Vec<f64>,
}

/// One node's activation and its partials, cached for a batch: `(f, df/dz,
/// df/da, df/db, df/dh)`.
type Facts = [f64; 5];

impl State {
    /// Copies the CSR weights and node parameters into a mutable state.
    pub fn prepare(csr: &Csr, params: &NodeParams) -> Result<State, String> {
        let n = params.z.len();
        if csr.indptr.len() != n + 1 {
            return Err(format!(
                "CSR has {} rows but params has {n} nodes",
                csr.indptr.len().saturating_sub(1)
            ));
        }
        Ok(State {
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
            weights: csr.weights.clone(),
            params: params.clone(),
            gw: vec![0.0; csr.indices.len()],
            gz: vec![0.0; n],
            ga: vec![0.0; n],
            gb: vec![0.0; n],
            gh: vec![0.0; n],
            gk: vec![0.0; n],
        })
    }

    /// `(f, df/dz, df/da, df/db, df/dh)` of node `i`, as `_accumulate` inlines it.
    fn facts(&self, i: usize) -> Facts {
        let p = &self.params;
        crate::activation::partials(p.z[i], p.a[i], p.b[i], p.h[i], p.k[i])
    }

    /// Sums the batch loss and its gradients into the accumulators; returns
    /// `(loss sum, touched nodes, touched parents)` in first-seen order.
    fn accumulate(&mut self, parents: &[usize], positions: &[usize]) -> Result<(f64, Vec<usize>, Vec<usize>), String> {
        if parents.len() != positions.len() {
            return Err("parents and positions must have the same length".to_string());
        }
        // node -> its facts for this batch; `touched` is the order they were first computed in
        let mut cache: std::collections::HashMap<usize, Facts> = std::collections::HashMap::new();
        let mut touched: Vec<usize> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut in_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut total = 0.0;
        let mut scores: Vec<f64> = Vec::new();
        let mut facts: Vec<Facts> = Vec::new();
        let mut ex: Vec<f64> = Vec::new();
        for (&p, &t) in parents.iter().zip(positions) {
            let (lo, hi) = (self.indptr[p], self.indptr[p + 1]);
            if t < lo || t >= hi {
                return Err(format!("position {t} is not an out-edge of node {p}"));
            }
            if hi - lo == 1 {
                // a single child: the softmax is 1, and the loss and every gradient 0
                continue;
            }
            let pp = match cache.get(&p) {
                Some(f) => *f,
                None => {
                    let f = self.facts(p);
                    cache.insert(p, f);
                    touched.push(p);
                    f
                }
            };
            let fp = pp[0];
            if in_rows.insert(p) {
                rows.push(p);
            }
            scores.clear();
            facts.clear();
            let mut m = f64::NEG_INFINITY;
            for j in lo..hi {
                let c = self.indices[j];
                let cc = match cache.get(&c) {
                    Some(f) => *f,
                    None => {
                        let f = self.facts(c);
                        cache.insert(c, f);
                        touched.push(c);
                        f
                    }
                };
                let s = self.weights[j] * fp * cc[0];
                scores.push(s);
                facts.push(cc);
                if s > m {
                    m = s;
                }
            }
            ex.clear();
            let mut ssum = 0.0;
            for &s in &scores {
                let v = (s - m).exp();
                ex.push(v);
                ssum += v;
            }
            total += m + ssum.ln() - scores[t - lo];
            let inv = 1.0 / ssum;
            let mut gfp = 0.0;
            for idx in 0..hi - lo {
                let j = lo + idx;
                let mut g = ex[idx] * inv;
                if j == t {
                    g -= 1.0;
                }
                let cc = facts[idx];
                let wj = self.weights[j];
                let fc = cc[0];
                self.gw[j] += g * fp * fc;
                gfp += g * wj * fc;
                let gfc = g * wj * fp;
                let c = self.indices[j];
                self.gz[c] += gfc * cc[1];
                self.ga[c] += gfc * cc[2];
                self.gb[c] += gfc * cc[3];
                self.gh[c] += gfc * cc[4];
                self.gk[c] += gfc;
            }
            self.gz[p] += gfp * pp[1];
            self.ga[p] += gfp * pp[2];
            self.gb[p] += gfp * pp[3];
            self.gh[p] += gfp * pp[4];
            self.gk[p] += gfp;
        }
        Ok((total, touched, rows))
    }

    /// One mini-batch gradient step; returns the mean loss *before* the update.
    ///
    /// Gradients are summed over the batch, divided by its size, clipped
    /// element-wise to `[-clip, clip]` and applied once; `b` is kept `>=
    /// MIN_B`.  An empty batch is a no-op returning 0.
    pub fn step(
        &mut self,
        parents: &[usize],
        positions: &[usize],
        lr: f64,
        act_lr: f64,
        clip: f64,
    ) -> Result<f64, String> {
        let n = parents.len();
        if n == 0 {
            return Ok(0.0);
        }
        let (total, touched, rows) = self.accumulate(parents, positions)?;
        let scale = 1.0 / n as f64;
        let clipped = |g: f64| -> f64 {
            if g > clip {
                clip
            } else if g < -clip {
                -clip
            } else {
                g
            }
        };
        for p in rows {
            for j in self.indptr[p]..self.indptr[p + 1] {
                let g = clipped(self.gw[j] * scale);
                self.weights[j] -= lr * g;
                self.gw[j] = 0.0;
            }
        }
        let params = &mut self.params;
        for c in touched {
            let g = clipped(self.gz[c] * scale);
            params.z[c] -= lr * g;
            self.gz[c] = 0.0;
            let g = clipped(self.ga[c] * scale);
            params.a[c] -= act_lr * g;
            self.ga[c] = 0.0;
            let g = clipped(self.gb[c] * scale);
            let nb = params.b[c] - act_lr * g;
            params.b[c] = if nb > MIN_B { nb } else { MIN_B };
            self.gb[c] = 0.0;
            let g = clipped(self.gh[c] * scale);
            params.h[c] -= act_lr * g;
            self.gh[c] = 0.0;
            let g = clipped(self.gk[c] * scale);
            params.k[c] -= act_lr * g;
            self.gk[c] = 0.0;
        }
        Ok(total * scale)
    }

    /// The mean batch loss at the current parameters, nothing updated.
    pub fn loss(&mut self, parents: &[usize], positions: &[usize]) -> Result<f64, String> {
        if parents.is_empty() {
            return Ok(0.0);
        }
        let (total, touched, rows) = self.accumulate(parents, positions)?;
        self.reset(&touched, &rows);
        Ok(total / parents.len() as f64)
    }

    /// The mean (unclipped) gradients of the batch loss, nothing applied:
    /// `(loss, [(csr position, dw)], [(node, [dz, da, db, dh, dk])])`.
    #[allow(clippy::type_complexity)]
    pub fn gradients(
        &mut self,
        parents: &[usize],
        positions: &[usize],
    ) -> Result<(f64, Vec<(usize, f64)>, Vec<(usize, [f64; 5])>), String> {
        if parents.is_empty() {
            return Ok((0.0, Vec::new(), Vec::new()));
        }
        let (total, touched, rows) = self.accumulate(parents, positions)?;
        let scale = 1.0 / parents.len() as f64;
        let mut dw = Vec::new();
        for &p in &rows {
            for j in self.indptr[p]..self.indptr[p + 1] {
                dw.push((j, self.gw[j] * scale));
            }
        }
        let dn = touched
            .iter()
            .map(|&c| {
                (
                    c,
                    [
                        self.gz[c] * scale,
                        self.ga[c] * scale,
                        self.gb[c] * scale,
                        self.gh[c] * scale,
                        self.gk[c] * scale,
                    ],
                )
            })
            .collect();
        self.reset(&touched, &rows);
        Ok((total * scale, dw, dn))
    }

    fn reset(&mut self, touched: &[usize], rows: &[usize]) {
        for &p in rows {
            for j in self.indptr[p]..self.indptr[p + 1] {
                self.gw[j] = 0.0;
            }
        }
        for &c in touched {
            self.gz[c] = 0.0;
            self.ga[c] = 0.0;
            self.gb[c] = 0.0;
            self.gh[c] = 0.0;
            self.gk[c] = 0.0;
        }
    }

    /// The weights (CSR order) and node parameters to write back.
    pub fn finalize(self) -> (Vec<f64>, NodeParams) {
        (self.weights, self.params)
    }
}

/// The backend a `--backend` / `"backend"` name asks for.
///
/// `auto` and `python` are this port of Python's backend (the same rule, the
/// same numbers); `rust` names it directly.  `torch` needs torch and a GPU and
/// is not in the port.
pub fn resolve(name: &str) -> Result<&'static str, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "python" | "rust" => Ok(NAME),
        "torch" => Err(
            "the torch backend is not in the Rust port (it needs torch and a GPU); --backend auto runs the \
             Python backend's learning rule, ported"
                .to_string(),
        ),
        other => Err(format!(
            "unknown backend {other:?}; expected 'auto', 'python' or 'torch'"
        )),
    }
}

/// Which backends this server can train with - honestly: the Python backend's
/// rule, ported, and no torch.
pub fn describe_backends() -> Json {
    Json::obj([
        ("python", Json::Bool(false)),
        ("torch", Json::Bool(false)),
        ("cuda", Json::Bool(false)),
        ("mps", Json::Bool(false)),
        ("rust", Json::Bool(true)),
        ("default", Json::str(NAME)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two nodes feeding three children, as a CSR.
    fn tiny() -> (Csr, NodeParams) {
        let csr = Csr {
            indptr: vec![0, 3, 5, 5, 5, 5],
            indices: vec![2, 3, 4, 3, 4],
            edge_ids: vec![0, 1, 2, 3, 4],
            weights: vec![0.9, 1.2, 0.7, 1.1, 0.6],
            edge_pos: vec![0, 1, 2, 3, 4],
        };
        let params = NodeParams {
            z: vec![0.3, -1.2, 2.1, -0.4, 0.9],
            a: vec![-1.0; 5],
            b: vec![1.0 / 3.0; 5],
            h: vec![0.0; 5],
            k: vec![0.0; 5],
        };
        (csr, params)
    }

    #[test]
    fn the_gradients_match_finite_differences() {
        let (csr, params) = tiny();
        let batch_p = [0usize, 1, 0];
        let batch_q = [1usize, 4, 2];
        let mut state = State::prepare(&csr, &params).unwrap();
        let (loss, dw, dn) = state.gradients(&batch_p, &batch_q).unwrap();
        let eps = 1e-6;
        let loss_at = |csr: &Csr, params: &NodeParams| -> f64 {
            State::prepare(csr, params).unwrap().loss(&batch_p, &batch_q).unwrap()
        };
        assert!((loss - loss_at(&csr, &params)).abs() < 1e-15);
        for (j, g) in dw {
            let (mut up, mut down) = (csr.clone(), csr.clone());
            up.weights[j] += eps;
            down.weights[j] -= eps;
            let fd = (loss_at(&up, &params) - loss_at(&down, &params)) / (2.0 * eps);
            assert!((g - fd).abs() < 1e-6, "w[{j}]: {g} vs {fd}");
        }
        for (c, grads) in dn {
            for (which, g) in grads.iter().enumerate() {
                let bump = |params: &NodeParams, d: f64| {
                    let mut p = params.clone();
                    match which {
                        0 => p.z[c] += d,
                        1 => p.a[c] += d,
                        2 => p.b[c] += d,
                        3 => p.h[c] += d,
                        _ => p.k[c] += d,
                    }
                    p
                };
                let fd = (loss_at(&csr, &bump(&params, eps)) - loss_at(&csr, &bump(&params, -eps))) / (2.0 * eps);
                assert!((g - fd).abs() < 1e-6, "node {c} param {which}: {g} vs {fd}");
            }
        }
    }

    #[test]
    fn a_step_lowers_the_loss_and_keeps_b_positive() {
        let (csr, params) = tiny();
        let mut state = State::prepare(&csr, &params).unwrap();
        let before = state.loss(&[0, 1], &[1, 4]).unwrap();
        for _ in 0..20 {
            state.step(&[0, 1], &[1, 4], 0.5, 0.05, 5.0).unwrap();
        }
        let after = state.loss(&[0, 1], &[1, 4]).unwrap();
        assert!(after < before, "{after} >= {before}");
        assert!(state.params.b.iter().all(|&b| b >= MIN_B));
        // a single-child row contributes nothing
        assert_eq!(state.step(&[], &[], 0.5, 0.05, 5.0).unwrap(), 0.0);
        assert!(state.step(&[0], &[4], 0.5, 0.05, 5.0).is_err());
    }

    #[test]
    fn torch_is_named_as_missing() {
        assert_eq!(resolve("auto").unwrap(), NAME);
        assert_eq!(resolve("python").unwrap(), NAME);
        assert!(resolve("torch").unwrap_err().contains("torch"));
        assert!(resolve("tpu").is_err());
    }
}
