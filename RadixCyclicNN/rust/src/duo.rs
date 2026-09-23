//! The two networks as one output path: the positive model writes, the
//! negative one vetoes (`radixnet/duo.py`, `go/radixnet/duo.go`).
//!
//! This is the GAN at output time rather than at training time.  The finished
//! pair works together on every answer:
//!
//! 1. the positive model (the count / reward model) proposes candidates - it
//!    is the generator, and the only one of the two that can write;
//! 2. the negative model judges each candidate - it is the discriminator, and
//!    the only one of the two that knows what going wrong looks like;
//! 3. what survives is returned, what does not comes back with the reason it
//!    was dropped, the fragment to blame and who said so.
//!
//! Three signals decide, and any one of them is enough to reject: **blame**
//! (the negative network's own verdict: risk against its threshold, behind
//! the coverage gate, so text the tutor has never failed is never vetoed on a
//! hunch), **peak** (off by default: the evidence on a single transition) and
//! **ratio** (log P_negative - log P_positive per unit, the classic
//! discriminator logit of two generative models).
//!
//! The server puts the pair in the way of every answer it hands out
//! ([`Service::guard`] and the predict, generate and converse routes), and
//! serves the negative network itself under `/api/negative`.

use std::sync::Arc;

use crate::beam::Prediction;
use crate::cli::verdict_json;
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::model::{GenerateOptions, Model, PredictOptions};
use crate::negative::{BlameOptions, BlamedSpan, JudgeOptions, NegativeOptions, ReasonRow};
use crate::search::PathResult;
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

/// The pair's verdict on one text.
#[derive(Clone, Debug)]
pub struct FilterVerdict {
    pub text: String,
    /// `pass`, `suspect` or `reject`.
    pub decision: String,
    /// What rejected it (`blame`, `peak`, `ratio`, `suspect`), `None` when
    /// nothing did.
    pub rule: Option<String>,
    pub risk: f64,
    pub peak: f64,
    pub coverage: f64,
    pub blame: f64,
    /// Blamed transitions (the coverage gate needs at least one).
    pub blamed: usize,
    pub threshold: f64,
    pub min_coverage: f64,
    pub peak_threshold: Option<f64>,
    pub ratio: f64,
    pub ratio_threshold: Option<f64>,
    /// Log-probability per unit under the positive model, then the negative.
    pub positive: f64,
    pub negative: f64,
    pub reasons: Vec<ReasonRow>,
    pub spans: Vec<BlamedSpan>,
    pub why: String,
}

impl FilterVerdict {
    /// The verdict as Python's `NegativeFilter.judge` returns it.
    pub fn to_json(&self) -> Json {
        let opt = |v: Option<f64>| v.map(Json::Num).unwrap_or(Json::Null);
        Json::obj([
            ("text", Json::str(self.text.clone())),
            ("decision", Json::str(self.decision.clone())),
            (
                "rule",
                match &self.rule {
                    Some(rule) => Json::str(rule.clone()),
                    None => Json::Null,
                },
            ),
            ("risk", Json::Num(self.risk)),
            ("peak", Json::Num(self.peak)),
            ("coverage", Json::Num(self.coverage)),
            ("blame", Json::Num(self.blame)),
            ("threshold", Json::Num(self.threshold)),
            ("min_coverage", Json::Num(self.min_coverage)),
            ("peak_threshold", opt(self.peak_threshold)),
            ("ratio", Json::Num(self.ratio)),
            ("ratio_threshold", opt(self.ratio_threshold)),
            ("positive", Json::Num(self.positive)),
            ("negative", Json::Num(self.negative)),
            (
                "reasons",
                Json::Arr(
                    self.reasons
                        .iter()
                        .map(|r| {
                            Json::obj([
                                ("reason", Json::str(r.reason.clone())),
                                ("blame", Json::Num(r.blame)),
                                ("share", Json::Num(r.share)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "spans",
                verdict_json(&crate::negative::Verdict {
                    spans: self.spans.clone(),
                    ..Default::default()
                })
                .at("spans")
                .clone(),
            ),
            ("why", Json::str(self.why.clone())),
        ])
    }

    fn rejected(&self) -> bool {
        self.decision == "reject"
    }
}

/// What a batch of candidates came to.
#[derive(Clone, Debug, Default)]
pub struct FilterOutcome {
    /// The texts handed back: every survivor of a judged batch, the `count`
    /// cleanest of a generated one.
    pub texts: Vec<String>,
    /// The walks behind `texts`, in the same order (generation only; they
    /// never go on the wire).
    pub results: Vec<PathResult>,
    /// Every survivor, in the order offered.
    pub kept: Vec<String>,
    pub verdicts: Vec<FilterVerdict>,
    pub candidates: usize,
    pub asked: usize,
    pub rate: Option<f64>,
}

impl FilterOutcome {
    /// The verdicts that rejected.
    pub fn rejected(&self) -> Vec<&FilterVerdict> {
        self.verdicts.iter().filter(|v| v.rejected()).collect()
    }

    /// `{"texts", "kept", "rejected", "verdicts", "candidates", "asked",
    /// "rate"}`, as the filter route answers.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("texts", Json::strs(self.texts.clone())),
            ("kept", Json::strs(self.kept.clone())),
            (
                "rejected",
                Json::Arr(self.rejected().iter().map(|v| v.to_json()).collect()),
            ),
            (
                "verdicts",
                Json::Arr(self.verdicts.iter().map(|v| v.to_json()).collect()),
            ),
            ("candidates", Json::Int(self.candidates as i64)),
            ("asked", Json::Int(self.asked as i64)),
            ("rate", self.rate.map(Json::Num).unwrap_or(Json::Null)),
        ])
    }
}

/// A continuation of a prefix through the pair.
#[derive(Clone, Debug)]
pub struct FilterPrediction {
    pub prefix: String,
    /// The best surviving text (prefix included); `None` when every
    /// candidate was rejected.
    pub text: Option<String>,
    pub kept: Vec<String>,
    pub verdicts: Vec<FilterVerdict>,
    pub candidates: usize,
    /// What the negative network predicts goes wrong from here.
    pub warning: Option<String>,
}

impl FilterPrediction {
    pub fn to_json(&self) -> Json {
        let text = |t: &Option<String>| t.clone().map(Json::str).unwrap_or(Json::Null);
        Json::obj([
            ("prefix", Json::str(self.prefix.clone())),
            ("text", text(&self.text)),
            ("kept", Json::strs(self.kept.clone())),
            (
                "rejected",
                Json::Arr(
                    self.verdicts
                        .iter()
                        .filter(|v| v.rejected())
                        .map(|v| v.to_json())
                        .collect(),
                ),
            ),
            (
                "verdicts",
                Json::Arr(self.verdicts.iter().map(|v| v.to_json()).collect()),
            ),
            ("candidates", Json::Int(self.candidates as i64)),
            ("warning", text(&self.warning)),
        ])
    }
}

/// The positive model's output, filtered by the negative one.
///
/// `positive` writes and is never changed here; `negative` judges (which
/// counts its judgements) and, with `learn`, is taught what it rejected.
pub struct Filter<'a> {
    pub positive: &'a mut Model,
    pub negative: &'a mut Model,
    pub config: FilterConfig,
}

impl<'a> Filter<'a> {
    /// Pairs a positive model with a negative one.
    pub fn new(positive: &'a mut Model, negative: &'a mut Model, config: FilterConfig) -> Result<Filter<'a>, String> {
        if !negative.is_negative() {
            return Err("the negative model must be a negative network".to_string());
        }
        if positive.is_negative() {
            return Err("the positive and negative models must be two different networks".to_string());
        }
        config.validate()?;
        Ok(Filter {
            positive,
            negative,
            config,
        })
    }

    /// Whether the negative half has anything to say: a failure has been
    /// blamed and not fully cleared.  An empty negative network vetoes
    /// nothing, so a filter around one costs the over-sampling and buys
    /// nothing - this is what the output paths ask before putting the pair in
    /// the way.
    pub fn ready(&self) -> bool {
        ready(self.negative)
    }

    /// The pair's verdict on one text: the negative network's blame, the
    /// likelihood ratio, and the decision.
    pub fn judge(&mut self, text: &str) -> FilterVerdict {
        let cfg = &self.config;
        let verdict = self
            .negative
            .judge(
                text,
                &JudgeOptions {
                    threshold: cfg.threshold,
                    min_coverage: cfg.min_coverage,
                    spans: cfg.spans(),
                },
            )
            .unwrap_or_default();
        let positive = self.positive.score(text).per_char;
        let negative = self.negative.score(text).per_char;
        let ratio = negative - positive;
        let gate = verdict.coverage >= verdict.min_coverage && verdict.blamed > 0;
        let mut rule: Option<&str> = if verdict.verdict == "reject" {
            Some("blame")
        } else if cfg.peak.is_some_and(|p| verdict.peak >= p) && verdict.blamed > 0 {
            // a single fragment the tutor has corrected is enough, whatever the rest of the text is
            Some("peak")
        } else if gate && cfg.ratio.is_some_and(|r| ratio >= r) {
            Some("ratio")
        } else {
            None
        };
        let mut decision = if rule.is_some() {
            "reject".to_string()
        } else {
            verdict.verdict.clone()
        };
        if decision == "suspect" && cfg.strict {
            decision = "reject".to_string();
            rule = Some("suspect");
        }
        let why = filter_why(&decision, rule, &verdict, ratio);
        FilterVerdict {
            text: text.to_string(),
            decision,
            rule: rule.map(str::to_string),
            risk: verdict.risk,
            peak: verdict.peak,
            coverage: verdict.coverage,
            blame: verdict.blame,
            blamed: verdict.blamed,
            threshold: verdict.threshold,
            min_coverage: verdict.min_coverage,
            peak_threshold: cfg.peak,
            ratio,
            ratio_threshold: cfg.ratio,
            positive,
            negative,
            reasons: verdict.reasons,
            spans: verdict.spans,
            why,
        }
    }

    /// Judges every text; with `learn` the rejected ones are blamed as new
    /// failures.
    pub fn filter(&mut self, texts: &[String]) -> Result<FilterOutcome, String> {
        let verdicts: Vec<FilterVerdict> = texts.iter().map(|t| self.judge(t)).collect();
        let kept: Vec<String> = verdicts
            .iter()
            .filter(|v| !v.rejected())
            .map(|v| v.text.clone())
            .collect();
        let rejected: Vec<String> = verdicts
            .iter()
            .filter(|v| v.rejected())
            .map(|v| v.text.clone())
            .collect();
        self.learn(&rejected, "rejected by the filter")?;
        Ok(FilterOutcome {
            texts: kept.clone(),
            results: Vec::new(),
            rate: (!texts.is_empty()).then(|| kept.len() as f64 / texts.len() as f64),
            kept,
            verdicts,
            candidates: texts.len(),
            asked: texts.len(),
        })
    }

    /// With `learn` on, blames what the pair refused (the texts long enough
    /// to hold a gram) under the configured reason.
    pub fn learn(&mut self, refused: &[String], note: &str) -> Result<(), String> {
        if refused.is_empty() || !self.config.learn {
            return Ok(());
        }
        let enc = self.negative.encoding();
        let blamed: Vec<String> = refused.iter().filter(|t| !enc.encode(t).is_empty()).cloned().collect();
        self.negative.blame(
            &blamed,
            &BlameOptions {
                reason: self.config.reason.clone(),
                source: "filter".to_string(),
                note: note.to_string(),
                ..Default::default()
            },
        )?;
        Ok(())
    }

    /// Writes through the pair: the positive model over-samples, the negative
    /// one vetoes, and the `count` survivors with the least blame come back
    /// (fewer when the filter vetoed too much - that is information, not an
    /// error).
    pub fn generate(&mut self, count: usize, o: &GenerateOptions) -> Result<FilterOutcome, String> {
        if count == 0 {
            return Ok(FilterOutcome::default());
        }
        let asked = count * self.config.over_sample();
        let results = self.positive.generate(&GenerateOptions {
            count: asked,
            ..o.clone()
        })?;
        let mut candidates: Vec<String> = Vec::new();
        let mut walks: Vec<PathResult> = Vec::new();
        for result in results {
            if result.text.is_empty() || candidates.contains(&result.text) {
                continue;
            }
            candidates.push(result.text.clone());
            walks.push(result);
        }
        let mut outcome = self.filter(&candidates)?;
        outcome.asked = asked;
        let mut keepers: Vec<&FilterVerdict> = outcome.verdicts.iter().filter(|v| !v.rejected()).collect();
        // cleanest first; a stable sort keeps the model's own order among equally clean texts
        keepers.sort_by(|a, b| a.risk.partial_cmp(&b.risk).unwrap_or(std::cmp::Ordering::Equal));
        let survivors: Vec<String> = keepers.iter().take(count).map(|v| v.text.clone()).collect();
        outcome.results = survivors
            .iter()
            .filter_map(|t| candidates.iter().position(|c| c == t).map(|i| walks[i].clone()))
            .collect();
        outcome.texts = survivors;
        Ok(outcome)
    }

    /// A finished prediction with the vetoed continuations taken out of it,
    /// and a verdict on each.  The search has already run, so this re-ranks
    /// what it offered rather than asking for more; when nothing survives the
    /// prediction is the prefix alone (`expanded` still reports the search).
    pub fn rank(&mut self, prefix: &str, result: &Prediction) -> (Prediction, Vec<FilterVerdict>) {
        let offered: Vec<PathResult> = if result.top.is_empty() {
            vec![result.best.clone()]
        } else {
            result.top.clone()
        };
        let enc = self.negative.encoding();
        let mut verdicts = Vec::with_capacity(offered.len());
        let mut kept: Vec<PathResult> = Vec::new();
        for candidate in offered {
            let verdict = self.judge(&enc.join(&[prefix, &candidate.text]));
            if !verdict.rejected() {
                kept.push(candidate);
            }
            verdicts.push(verdict);
        }
        let mut best = kept.first().cloned().unwrap_or_else(|| PathResult {
            full_text: prefix.to_string(),
            ..Default::default()
        });
        best.expanded = result.best.expanded;
        let mut ranked = result.clone();
        ranked.best = best;
        ranked.top = kept;
        (ranked, verdicts)
    }

    /// Continues a prefix through the pair: the positive model's top-K
    /// continuations minus the vetoed ones, and what the negative network
    /// expects to go wrong from here.
    pub fn predict(&mut self, prefix: &str, o: &PredictOptions) -> Result<FilterPrediction, String> {
        let found = self.positive.predict(
            prefix,
            &PredictOptions {
                mode: "beam".to_string(),
                ..o.clone()
            },
        )?;
        let offered = if found.top.is_empty() {
            vec![found.best.clone()]
        } else {
            found.top.clone()
        };
        let mut candidates: Vec<String> = Vec::new();
        for result in &offered {
            if !result.text.is_empty() && !candidates.contains(&result.text) {
                candidates.push(result.text.clone());
            }
        }
        let enc = self.negative.encoding();
        let joined: Vec<String> = candidates.iter().map(|t| enc.join(&[prefix, t])).collect();
        let outcome = self.filter(&joined)?;
        let warning = self.negative.predict(
            prefix,
            &PredictOptions {
                length: o.length,
                k: 1,
                max_length: o.max_length,
                ..Default::default()
            },
        )?;
        Ok(FilterPrediction {
            prefix: prefix.to_string(),
            text: outcome.verdicts.iter().find(|v| !v.rejected()).map(|v| v.text.clone()),
            kept: outcome.kept,
            verdicts: outcome.verdicts,
            candidates: candidates.len(),
            warning: (!warning.best.text.is_empty()).then(|| enc.join(&[prefix, &warning.best.text])),
        })
    }

    /// What the pair is made of, for a status line.
    pub fn describe(&self) -> Json {
        describe(self.positive, self.negative, &self.config)
    }
}

/// Whether a negative network has anything to veto with.
pub fn ready(negative: &Model) -> bool {
    negative.g.neg.as_ref().is_some_and(|n| n.total_blame > 0.0)
}

/// `{"positive", "negative", "config"}` - what a pair is made of.
pub fn describe(positive: &Model, negative: &Model, cfg: &FilterConfig) -> Json {
    let neg = negative.neg.as_ref();
    let threshold = cfg
        .threshold
        .or(neg.map(|n| n.threshold))
        .map(Json::Num)
        .unwrap_or(Json::Null);
    let min_coverage = cfg
        .min_coverage
        .or(neg.map(|n| n.min_coverage))
        .map(Json::Num)
        .unwrap_or(Json::Null);
    let opt = |v: Option<f64>| v.map(Json::Num).unwrap_or(Json::Null);
    Json::obj([
        ("positive", labelled(positive, "Count / reward (Rust)")),
        ("negative", labelled(negative, "Negative network (Rust)")),
        (
            "config",
            Json::obj([
                ("threshold", threshold),
                ("min_coverage", min_coverage),
                ("ratio", opt(cfg.ratio)),
                ("peak", opt(cfg.peak)),
                ("over_sample", Json::Int(cfg.over_sample() as i64)),
                ("strict", Json::Bool(cfg.strict)),
                ("learn", Json::Bool(cfg.learn)),
            ]),
        ),
    ])
}

/// A model's stats with its kind and label first, as `describe` shows them.
fn labelled(model: &Model, label: &str) -> Json {
    let mut pairs = vec![
        ("kind".to_string(), Json::str(model.kind())),
        ("label".to_string(), Json::str(label)),
    ];
    if let Json::Obj(stats) = crate::report::stats(model) {
        pairs.extend(stats.into_iter().filter(|(k, _)| k != "kind"));
    }
    Json::Obj(pairs)
}

/// The one sentence a decision rests on.
fn filter_why(decision: &str, rule: Option<&str>, verdict: &crate::negative::Verdict, ratio: f64) -> String {
    use crate::negative::python_repr;
    if decision == "pass" {
        return verdict.why.clone();
    }
    match rule {
        Some("peak") => {
            let (at, reason) = match verdict.spans.first() {
                Some(span) => (
                    format!(" at {}", python_repr(&span.fragment)),
                    if span.reason.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", span.reason)
                    },
                ),
                None => (String::new(), String::new()),
            };
            format!(
                "it carries {:.2} of blame on one fragment{at}{reason}; rejected",
                verdict.peak
            )
        }
        Some("ratio") => {
            let reason = match verdict.reasons.first() {
                Some(r) => format!(", mostly {}", python_repr(&r.reason)),
                None => String::new(),
            };
            format!(
                "it reads {ratio:.2} nats/char more like known failure than like the training data{reason}; rejected"
            )
        }
        Some("suspect") => verdict
            .why
            .replace("; below the threshold, kept", "; rejected (strict)"),
        _ => verdict.why.clone(),
    }
}

// -- the server's negative network ---------------------------------------------------------------

impl Service {
    /// Where the negative network is saved: `<stem>.negative<ext>` beside the
    /// model file (Python's `model_path_for("negative")`).
    pub fn negative_path(&self) -> String {
        if self.model_path.is_empty() {
            return String::new();
        }
        crate::cli::negative_path(&self.model_path)
    }

    /// Runs `f` on the server's negative network: loaded from its file the
    /// first time, else created empty.  Takes the negative network's lock
    /// only - pair it with [`Service::with_model`] *outside* it, never inside.
    pub fn with_negative<T>(&self, f: impl FnOnce(&mut Model) -> T) -> Result<T, ApiError> {
        let mut slot = self.negative.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(self.load_negative()?);
        }
        Ok(f(slot.as_mut().expect("a negative network")))
    }

    fn load_negative(&self) -> Result<Model, ApiError> {
        let path = self.negative_path();
        let mut model = if !path.is_empty() && std::path::Path::new(&path).is_file() {
            let m = Model::load(&path)?;
            if !m.is_negative() {
                return Err(ApiError::bad_request(format!(
                    "{path} holds a {} model, not a negative one",
                    m.kind()
                )));
            }
            crate::log_info!("negative", "negative model loaded from {path}");
            m
        } else {
            Model::new_negative(
                self.seed,
                &NegativeOptions {
                    encoding: self.active_encoding(),
                    ..Default::default()
                },
            )?
        };
        model.workers = self.workers;
        model.g.workers = self.workers;
        Ok(model)
    }

    /// Whether there is a negative network worth guarding with: one in memory,
    /// or one saved beside the model - never created here (an answer is not
    /// the place to bring one into being) - that has been taught a failure.
    pub fn guard_ready(&self) -> bool {
        {
            let slot = self.negative.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(neg) = slot.as_ref() {
                return ready(neg);
            }
        }
        let path = self.negative_path();
        if path.is_empty() || !std::path::Path::new(&path).is_file() {
            return false;
        }
        self.with_negative(|n| ready(n)).unwrap_or(false)
    }

    /// Runs `f` on the pair guarding the running model, or returns `None`
    /// when there is nothing to guard with (see [`Service::guard_ready`]).
    /// Holds both locks, the model's first, for as long as `f` runs.
    pub fn guard<T>(&self, f: impl FnOnce(&mut Filter) -> T) -> Result<Option<T>, ApiError> {
        if !self.guard_ready() {
            return Ok(None);
        }
        let config = self.guard.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut out = None;
        let mut failure = None;
        self.with_model(|positive| {
            let outcome = self.with_negative(|negative| {
                let mut pair = Filter::new(positive, negative, config)?;
                Ok::<T, String>(f(&mut pair))
            });
            match outcome {
                Ok(Ok(value)) => out = Some(value),
                Ok(Err(why)) => failure = Some(ApiError::bad_request(why)),
                Err(err) => failure = Some(err),
            }
        });
        match failure {
            Some(err) => Err(err),
            None => Ok(out),
        }
    }
}

/// What the guard did, for the caller to show: the vetoes, with the reason
/// and the fragment behind each (Python's `_guard_report`).
pub fn guard_report(pair: &Filter, verdicts: &[FilterVerdict], extra: Vec<(&str, Json)>) -> Json {
    let rejected: Vec<Json> = verdicts.iter().filter(|v| v.rejected()).map(|v| v.to_json()).collect();
    let mut pairs = vec![
        ("on".to_string(), Json::Bool(true)),
        ("vetoed".to_string(), Json::Int(rejected.len() as i64)),
        ("rejected".to_string(), Json::Arr(rejected)),
        (
            "verdicts".to_string(),
            Json::Arr(verdicts.iter().map(|v| v.to_json()).collect()),
        ),
        ("negative".to_string(), crate::report::stats(pair.negative)),
        ("config".to_string(), pair.describe().at("config").clone()),
    ];
    pairs.extend(extra.into_iter().map(|(k, v)| (k.to_string(), v)));
    Json::Obj(pairs)
}

// -- the routes ---------------------------------------------------------------------------------

/// The texts of `texts` (a list) and `text` (one per line), refused when
/// there are none.
fn negative_texts(r: &Request, what: &str) -> Result<Vec<String>, ApiError> {
    let mut texts = r.texts("texts");
    texts.extend(r.texts("text"));
    if texts.is_empty() {
        return Err(ApiError::bad_request(format!(
            "give the {what} as 'texts' (a list) or 'text' (one per line)"
        )));
    }
    Ok(texts)
}

/// A number that may be absent (`null` or missing) and must be >= `minimum`.
fn maybe_number(r: &Request, key: &str, minimum: Option<f64>) -> Result<Option<f64>, ApiError> {
    match r.body.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(v) => {
            let n = v
                .as_f64()
                .ok_or_else(|| ApiError::bad_request(format!("{key:?} must be a number")))?;
            if let Some(min) = minimum {
                if n < min {
                    return Err(ApiError::bad_request(format!("{key:?} must be >= {min}")));
                }
            }
            Ok(Some(n))
        }
    }
}

fn at_least(r: &Request, key: &str, fallback: f64, minimum: f64) -> Result<f64, ApiError> {
    let n = r.number(key, fallback)?;
    if n < minimum {
        return Err(ApiError::bad_request(format!("{key:?} must be >= {minimum}")));
    }
    Ok(n)
}

fn count_at_least(r: &Request, key: &str, fallback: usize, minimum: usize) -> Result<usize, ApiError> {
    let n = r.usize(key, fallback)?;
    if n < minimum {
        return Err(ApiError::bad_request(format!("{key:?} must be >= {minimum}")));
    }
    Ok(n)
}

fn negative_result(m: &mut Model, records: Vec<Json>, extra: Vec<(&str, Json)>) -> Json {
    let mut pairs = vec![
        ("records".to_string(), Json::Arr(records)),
        (
            "reasons".to_string(),
            Json::Arr(m.reasons().iter().map(|r| r.to_json()).collect()),
        ),
        ("stats".to_string(), crate::report::stats(m)),
    ];
    pairs.extend(extra.into_iter().map(|(k, v)| (k.to_string(), v)));
    Json::Obj(pairs)
}

fn weights_json(m: &Model) -> Json {
    let mut pairs = vec![("function".to_string(), Json::str("blame"))];
    for (name, value) in m.g.negative_weight_config() {
        pairs.push((name.to_string(), Json::Num(value)));
    }
    Json::Obj(pairs)
}

fn settings_json(m: &Model) -> Json {
    let neg = m.neg.as_ref();
    Json::obj([
        ("threshold", neg.map(|n| Json::Num(n.threshold)).unwrap_or(Json::Null)),
        (
            "min_coverage",
            neg.map(|n| Json::Num(n.min_coverage)).unwrap_or(Json::Null),
        ),
    ])
}

fn status(svc: &Arc<Service>, _r: &Request) -> Answer {
    let path = svc.negative_path();
    svc.with_negative(|m| {
        Json::obj([
            (
                "path",
                if path.is_empty() {
                    Json::Null
                } else {
                    Json::str(path.clone())
                },
            ),
            ("active", Json::Bool(false)),
            ("stats", crate::report::stats(m)),
            ("reasons", Json::Arr(m.reasons().iter().map(|r| r.to_json()).collect())),
            ("journal", Json::Arr(m.recent(20).iter().map(|e| e.to_json()).collect())),
            ("weights", weights_json(m)),
            ("settings", settings_json(m)),
        ])
    })
}

fn blame(svc: &Arc<Service>, r: &Request) -> Answer {
    let texts = negative_texts(r, "failed texts")?;
    let reason = r.text("reason", "unspecified");
    let severity = at_least(r, "severity", 1.0, 0.0)?;
    let opts = BlameOptions {
        reason: reason.clone(),
        severity,
        source: r.text("source", "api"),
        note: r.text("note", ""),
        epochs: count_at_least(r, "epochs", 1, 1)?,
        no_compress: false,
    };
    svc.with_negative(|m| -> Result<Json, String> {
        let records = m.blame(&texts, &opts)?;
        Ok(negative_result(
            m,
            records.iter().map(|r| r.to_json()).collect(),
            vec![
                ("texts", Json::Int(texts.len() as i64)),
                ("reason", Json::str(reason.clone())),
                ("severity", Json::Num(severity)),
            ],
        ))
    })?
    .map_err(ApiError::bad_request)
}

fn clear(svc: &Arc<Service>, r: &Request) -> Answer {
    let texts = negative_texts(r, "passed texts")?;
    let weight = at_least(r, "weight", 1.0, 0.0)?;
    let epochs = count_at_least(r, "epochs", 1, 1)?;
    svc.with_negative(|m| -> Result<Json, String> {
        let records = m.clear(&texts, weight, epochs)?;
        let last = |key: &str| {
            records
                .last()
                .and_then(|r| r.extra_value(key))
                .cloned()
                .unwrap_or(Json::Int(0))
        };
        let (matched, unmatched) = (last("matched"), last("unmatched"));
        Ok(negative_result(
            m,
            records.iter().map(|r| r.to_json()).collect(),
            vec![
                ("texts", Json::Int(texts.len() as i64)),
                ("matched", matched),
                ("unmatched", unmatched),
            ],
        ))
    })?
    .map_err(ApiError::bad_request)
}

fn judge(svc: &Arc<Service>, r: &Request) -> Answer {
    let texts = negative_texts(r, "texts to judge")?;
    let opts = JudgeOptions {
        threshold: maybe_number(r, "threshold", Some(0.0))?,
        min_coverage: maybe_number(r, "min_coverage", Some(0.0))?,
        spans: r.usize("spans", 5)?,
    };
    svc.with_negative(|m| {
        let verdicts: Vec<Json> = texts
            .iter()
            .filter_map(|t| m.judge(t, &opts))
            .map(|v| verdict_json(&v))
            .collect();
        Json::obj([("verdicts", Json::Arr(verdicts)), ("stats", crate::report::stats(m))])
    })
}

/// The filter settings of a request body (Python's `FilterConfig` fields).
fn filter_config(r: &Request) -> Result<FilterConfig, ApiError> {
    let config = FilterConfig {
        threshold: maybe_number(r, "threshold", Some(0.0))?,
        min_coverage: maybe_number(r, "min_coverage", Some(0.0))?,
        ratio: if r.flag("no_ratio", false) {
            None
        } else {
            Some(r.number("ratio", 0.0)?)
        },
        peak: maybe_number(r, "peak", Some(0.0))?,
        over_sample: count_at_least(r, "over_sample", 3, 1)?,
        strict: r.flag("strict", false),
        spans: r.usize("spans", 3)?,
        learn: r.flag("learn", false),
        reason: r.text("reason", "filtered"),
    };
    config.validate()?;
    Ok(config)
}

fn filter(svc: &Arc<Service>, r: &Request) -> Answer {
    let config = filter_config(r)?;
    let mut texts = r.texts("texts");
    texts.extend(r.texts("text"));
    let count = r.usize("count", 3)?;
    let generate = GenerateOptions {
        mode: r.text("mode", "sample"),
        max_length: r.usize("max_length", 60)?,
        temperature: at_least(r, "temperature", 1.0, 0.0)?,
        prefix: r.text("prefix", ""),
        seed: match r.body.get("seed") {
            None | Some(Json::Null) => None,
            Some(v) => Some(
                v.as_i64()
                    .ok_or_else(|| ApiError::bad_request("\"seed\" must be an integer"))?,
            ),
        },
        step_penalty: at_least(r, "step_penalty", 0.0, 0.0)?,
        beam: r.usize("beam", 0)?,
        ..Default::default()
    };
    let mut failure = None;
    let mut out = Json::Null;
    svc.with_model(|positive| {
        let answer = svc.with_negative(|negative| -> Result<Json, String> {
            let mut pair = Filter::new(positive, negative, config)?;
            let outcome = if texts.is_empty() {
                pair.generate(count, &generate)?
            } else {
                pair.filter(&texts)?
            };
            let Json::Obj(mut pairs) = outcome.to_json() else {
                unreachable!()
            };
            pairs.push(("pair".to_string(), pair.describe()));
            Ok(Json::Obj(pairs))
        });
        match answer {
            Ok(Ok(doc)) => out = doc,
            Ok(Err(why)) => failure = Some(ApiError::bad_request(why)),
            Err(err) => failure = Some(err),
        }
    });
    match failure {
        Some(err) => Err(err),
        None => Ok(out),
    }
}

fn forget(svc: &Arc<Service>, r: &Request) -> Answer {
    let reason = match r.body.get("reason") {
        None | Some(Json::Null) => None,
        Some(v) => v.as_str().map(str::to_string),
    };
    let factor = at_least(r, "factor", 0.0, 0.0)?;
    svc.with_negative(|m| -> Result<Json, String> {
        let result = m.g.forget(reason.as_deref(), factor)?;
        m.g.prepare();
        Ok(negative_result(
            m,
            Vec::new(),
            vec![
                ("reason", Json::str(result.reason)),
                ("edges", Json::Int(result.edges as i64)),
                ("blame_removed", Json::Num(result.blame_removed)),
            ],
        ))
    })?
    .map_err(ApiError::bad_request)
}

fn settings(svc: &Arc<Service>, r: &Request) -> Answer {
    let threshold = maybe_number(r, "threshold", Some(0.0))?;
    let min_coverage = maybe_number(r, "min_coverage", Some(0.0))?;
    let mut scales: Vec<(String, f64)> = Vec::new();
    for name in ["share_scale", "blame_scale", "clear_scale"] {
        if let Some(v) = maybe_number(r, name, None)? {
            scales.push((name.to_string(), v));
        }
    }
    svc.with_negative(|m| -> Result<Json, String> {
        {
            let neg = m.neg.as_mut().expect("a negative network");
            if let Some(t) = threshold {
                neg.threshold = t;
            }
            if let Some(c) = min_coverage {
                neg.min_coverage = c;
            }
        }
        if !scales.is_empty() {
            m.g.configure_negative(&scales)?;
        }
        Ok(Json::obj([
            ("settings", settings_json(m)),
            ("weights", weights_json(m)),
            ("stats", crate::report::stats(m)),
        ]))
    })?
    .map_err(ApiError::bad_request)
}

fn reset(svc: &Arc<Service>, r: &Request) -> Answer {
    let seed = match r.body.get("seed") {
        None | Some(Json::Null) => svc.seed,
        Some(v) => v
            .as_i64()
            .ok_or_else(|| ApiError::bad_request("\"seed\" must be an integer"))?,
    };
    let mut fresh = Model::new_negative(
        seed,
        &NegativeOptions {
            encoding: svc.active_encoding(),
            ..Default::default()
        },
    )?;
    fresh.workers = svc.workers;
    fresh.g.workers = svc.workers;
    let stats = crate::report::stats(&fresh);
    *svc.negative.lock().unwrap_or_else(|e| e.into_inner()) = Some(fresh);
    Ok(Json::obj([
        ("stats", stats),
        ("reasons", Json::Arr(Vec::new())),
        ("journal", Json::Arr(Vec::new())),
    ]))
}

fn save(svc: &Arc<Service>, r: &Request) -> Answer {
    let given = r.text("path", "");
    let target = if given.is_empty() { svc.negative_path() } else { given };
    if target.is_empty() {
        return Err(ApiError::bad_request(
            "no 'path' given and the server was started without a model path",
        ));
    }
    let target = std::path::absolute(&target)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(target);
    svc.with_negative(|m| m.save(&target))?.map_err(ApiError::bad_request)?;
    let bytes = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
    Ok(Json::obj([
        ("path", Json::str(target)),
        ("bytes", Json::Int(bytes as i64)),
    ]))
}

/// The negative network's routes: `GET /api/negative`, and `POST
/// /api/negative/{blame,clear,judge,filter,forget,settings,reset,save}`.
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/negative", status);
    server.route("POST", "/api/negative/blame", blame);
    server.route("POST", "/api/negative/clear", clear);
    server.route("POST", "/api/negative/judge", judge);
    server.route("POST", "/api/negative/filter", filter);
    server.route("POST", "/api/negative/forget", forget);
    server.route("POST", "/api/negative/settings", settings);
    server.route("POST", "/api/negative/reset", reset);
    server.route("POST", "/api/negative/save", save);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TrainOptions;
    use crate::GraphOptions;

    fn pair() -> (Model, Model) {
        let mut positive = Model::new(1, GraphOptions::default()).unwrap();
        let texts: Vec<String> = ["the cat sat on the mat", "the dog sat on the log", "a cat and a dog"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        positive
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut negative = Model::new_negative(1, &NegativeOptions::default()).unwrap();
        negative
            .blame(
                &["the the the the cat".to_string()],
                &BlameOptions {
                    reason: "repetition".to_string(),
                    severity: 2.0,
                    ..Default::default()
                },
            )
            .unwrap();
        (positive, negative)
    }

    #[test]
    fn a_known_failure_is_vetoed_and_new_text_is_not() {
        let (mut positive, mut negative) = pair();
        let mut filter = Filter::new(&mut positive, &mut negative, FilterConfig::default()).unwrap();
        assert!(filter.ready());
        let outcome = filter
            .filter(&["the the the the cat".to_string(), "an unseen line".to_string()])
            .unwrap();
        let decisions: Vec<&str> = outcome.verdicts.iter().map(|v| v.decision.as_str()).collect();
        assert_eq!(decisions, vec!["reject", "pass"]);
        assert_eq!(outcome.verdicts[0].rule.as_deref(), Some("blame"));
        assert_eq!(outcome.kept, vec!["an unseen line".to_string()]);
        assert_eq!(outcome.rate, Some(0.5));
        let doc = outcome.to_json();
        assert!(doc.at("verdicts").as_array()[1].at("rule").is_null());
    }

    #[test]
    fn a_learning_filter_blames_what_it_rejects() {
        let (mut positive, mut negative) = pair();
        let before = negative.neg.as_ref().unwrap().failures_total.value;
        let config = FilterConfig {
            learn: true,
            ..Default::default()
        };
        let mut filter = Filter::new(&mut positive, &mut negative, config).unwrap();
        filter.filter(&["the the the the cat".to_string()]).unwrap();
        assert_eq!(negative.neg.as_ref().unwrap().failures_total.value, before + 1);
        assert_eq!(negative.recent(1)[0].source, "filter");
    }

    #[test]
    fn generation_over_samples_and_ranking_keeps_the_search_order() {
        let (mut positive, mut negative) = pair();
        let mut filter = Filter::new(&mut positive, &mut negative, FilterConfig::default()).unwrap();
        let outcome = filter
            .generate(
                2,
                &GenerateOptions {
                    mode: "beam".to_string(),
                    max_length: 30,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(outcome.asked, 6);
        assert!(outcome.texts.len() <= 2);
        assert_eq!(outcome.texts.len(), outcome.results.len());
        let found = filter
            .positive
            .predict(
                "the ",
                &PredictOptions {
                    length: 8,
                    k: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let (ranked, verdicts) = filter.rank("the ", &found);
        assert_eq!(verdicts.len(), found.top.len());
        assert_eq!(ranked.best.expanded, found.best.expanded);
    }

    #[test]
    fn the_pair_needs_two_different_networks() {
        let (mut positive, mut negative) = pair();
        let mut other = Model::new(0, GraphOptions::default()).unwrap();
        assert!(Filter::new(&mut positive, &mut other, FilterConfig::default()).is_err());
        let bad = FilterConfig {
            min_coverage: Some(2.0),
            ..Default::default()
        };
        assert!(Filter::new(&mut positive, &mut negative, bad).is_err());
    }
}
