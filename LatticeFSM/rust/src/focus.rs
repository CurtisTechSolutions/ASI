//! The learned focus: a function of the input that says which node of the central vertical vector a run starts from.
//!
//! `focus(text) = sigmoid(w · x(text))`, where `x` is the bias, the text's length as `n / (n + 6)`, each symbol's
//! share of it, and the symbol it starts and ends with, one-hot: `2 + 3A` features. The weights start at a bias of
//! `-3` and nothing else, so an untrained learner's focus is `0.047`, the top node. A run that learns draws
//! `f = mu + explore · N(0, 1)` and starts from the node the clipped draw picks; when it is credited `c`,
//! `w += rate · (c − baseline) · (f − mu) / explore² · mu(1 − mu) · x` and the baseline moves 5 % toward `c`: the
//! gradient of a Gaussian policy through the sigmoid, the unclipped draw learned from. The same as
//! `../latticefsm/focus.py`.

use crate::json::Json;

/// The bias an untrained learner starts with: sigmoid(-3) = 0.047, the top node.
pub const FOCUS_BIAS: f64 = -3.0;
const LIMIT: f64 = 8.0;

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `focus(text) = sigmoid(w · x(text))`, learned from credit.
#[derive(Clone, Debug, PartialEq)]
pub struct FocusLearner {
    pub alphabet: Vec<String>,
    pub weights: Vec<f64>,
    pub rate: f64,
    pub explore: f64,
    /// The running mean of the credit the learner has seen.
    pub baseline: f64,
}

/// A draw the learner made for a run: the unclipped focus, its mean, and the features - what credit learns from.
pub type FocusDraw = (f64, f64, Vec<f64>);

impl FocusLearner {
    pub fn new(alphabet: Vec<String>, rate: f64, explore: f64) -> FocusLearner {
        let n = 2 + 3 * alphabet.len();
        let mut weights = vec![0.0; n];
        weights[0] = FOCUS_BIAS;
        FocusLearner {
            alphabet,
            weights,
            rate,
            explore,
            baseline: 0.0,
        }
    }

    /// The bias, `n / (n + 6)`, each symbol's share, the first symbol and the last, one-hot; `symbols` are indices.
    pub fn features(&self, symbols: &[usize]) -> Vec<f64> {
        let a = self.alphabet.len();
        let n = symbols.len();
        let mut x = vec![0.0; 2 + 3 * a];
        x[0] = 1.0;
        x[1] = n as f64 / (n as f64 + 6.0);
        for &s in symbols {
            x[2 + s] += 1.0 / n as f64;
        }
        if let (Some(&first), Some(&last)) = (symbols.first(), symbols.last()) {
            x[2 + a + first] = 1.0;
            x[2 + 2 * a + last] = 1.0;
        }
        x
    }

    pub fn mean(&self, x: &[f64]) -> f64 {
        sigmoid(self.weights.iter().zip(x).map(|(w, xi)| w * xi).sum())
    }

    /// The focus the learner gives this input, without exploring.
    pub fn predict(&self, symbols: &[usize]) -> f64 {
        self.mean(&self.features(symbols))
    }

    /// The draw (unclipped), its mean and the features: the mean plus noise when `explore`, else the mean.
    pub fn choose(&self, symbols: &[usize], gauss: impl FnOnce() -> f64, explore: bool) -> FocusDraw {
        let x = self.features(symbols);
        let mu = self.mean(&x);
        let f = if explore && self.explore > 0.0 {
            mu + gauss() * self.explore
        } else {
            mu
        };
        (f, mu, x)
    }

    pub fn clip(f: f64) -> f64 {
        f.clamp(0.0, 1.0)
    }

    /// Move the weights with `credit` against the baseline, then move the baseline.
    pub fn learn(&mut self, credit: f64, draw: &FocusDraw) {
        if credit == 0.0 || self.explore <= 0.0 {
            return;
        }
        let (f, mu, x) = draw;
        let advantage = credit - self.baseline;
        self.baseline += 0.05 * (credit - self.baseline);
        let g = self.rate * advantage * (f - mu) / (self.explore * self.explore) * mu * (1.0 - mu);
        for (w, xi) in self.weights.iter_mut().zip(x) {
            *w = (*w + g * xi).clamp(-LIMIT, LIMIT);
        }
    }

    /// Keep the weights on the symbols they belong to when two symbols trade places.
    pub fn swap_symbols(&mut self, a: usize, b: usize) {
        let n = self.alphabet.len();
        self.alphabet.swap(a, b);
        for base in [2, 2 + n, 2 + 2 * n] {
            self.weights.swap(base + a, base + b);
        }
    }

    pub fn to_json(&self) -> Json {
        Json::object()
            .with("alphabet", Json::strings(&self.alphabet))
            .with("weights", Json::numbers(&self.weights))
            .with("rate", self.rate.into())
            .with("explore", self.explore.into())
            .with("baseline", self.baseline.into())
    }

    pub fn from_json(v: &Json) -> Result<FocusLearner, String> {
        let alphabet: Vec<String> = v
            .get("alphabet")
            .and_then(Json::as_array)
            .ok_or("focus_learner.alphabet")?
            .iter()
            .filter_map(|s| s.as_str().map(str::to_string))
            .collect();
        let weights: Vec<f64> = v
            .get("weights")
            .and_then(Json::as_array)
            .ok_or("focus_learner.weights")?
            .iter()
            .filter_map(Json::as_f64)
            .collect();
        if weights.len() != 2 + 3 * alphabet.len() {
            return Err("a focus learner has 2 + 3A weights".to_string());
        }
        Ok(FocusLearner {
            alphabet,
            weights,
            rate: v.num("rate", 0.05),
            explore: v.num("explore", 0.15),
            baseline: v.num("baseline", 0.0),
        })
    }
}
