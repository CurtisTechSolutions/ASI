//! The two networks as one output path: the positive model writes, the
//! negative one vetoes (`radixnet/duo.py`, `go/radixnet/duo.go`).
//!
//! Being ported: the filter's settings are here, the filter itself follows.

use crate::http::Server;
use crate::service::Service;

/// How strictly the negative network filters the positive one's output.
/// `None` is "unset" (the negative model's own thresholds) and, for `ratio`
/// and `peak`, "off".
#[derive(Clone, Debug, PartialEq)]
pub struct FilterConfig {
    pub threshold: Option<f64>,
    pub min_coverage: Option<f64>,
    /// The log-odds (negative minus positive, per character) at or above which
    /// a candidate is rejected; `None` turns the rule off.
    pub ratio: Option<f64>,
    /// The evidence on a single transition at or above which a candidate is
    /// rejected; `None` turns the rule off.
    pub peak: Option<f64>,
    /// How many candidates are drawn per wanted output (0 = 3).
    pub over_sample: usize,
    /// Also drops candidates the negative network only finds suspect.
    pub strict: bool,
    /// How many blamed fragments a verdict reports (0 = 3).
    pub spans: usize,
    /// Blames what this filter rejects (off: the tutor supplies the negatives,
    /// the filter only applies them).
    pub learn: bool,
    /// Recorded when `learn` is on.
    pub reason: String,
}

impl Default for FilterConfig {
    /// The negative model's own thresholds, the ratio rule at 0, no peak rule,
    /// three candidates per wanted text.
    fn default() -> FilterConfig {
        FilterConfig {
            threshold: None,
            min_coverage: None,
            ratio: Some(0.0),
            peak: None,
            over_sample: 3,
            strict: false,
            spans: 3,
            learn: false,
            reason: "filtered".to_string(),
        }
    }
}

impl FilterConfig {
    /// Rejects settings the filter cannot run with.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("threshold", self.threshold),
            ("min_coverage", self.min_coverage),
            ("ratio", self.ratio),
            ("peak", self.peak),
        ] {
            if let Some(v) = value {
                if !v.is_finite() {
                    return Err(format!("{name} must be a finite number or unset"));
                }
            }
        }
        if let Some(c) = self.min_coverage {
            if !(0.0..=1.0).contains(&c) {
                return Err(format!("min_coverage must lie in [0, 1], got {c}"));
            }
        }
        Ok(())
    }

    /// Candidates drawn per wanted output.
    pub fn over_sample(&self) -> usize {
        if self.over_sample == 0 {
            3
        } else {
            self.over_sample
        }
    }

    /// Blamed fragments a verdict reports.
    pub fn spans(&self) -> usize {
        if self.spans == 0 {
            3
        } else {
            self.spans
        }
    }
}

/// The negative network's routes: `GET /api/negative`, and `POST
/// /api/negative/{blame,clear,judge,filter,forget,settings,reset,save}`.
pub fn routes(_server: &mut Server<Service>) {}
