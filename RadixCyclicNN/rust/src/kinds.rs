//! The model kinds: which there are, what each is called, how a new one is
//! made, and how the commands and routes every kind answers hand a call to the
//! kind's own code.
//!
//! Python's `radixnet` has four kinds of model over one self-compressing
//! graph (`radixnet/model.py`'s `model_classes`): `radix`, the sine-activation
//! network ([`crate::radix`]); `count`, the count / reward model this port
//! started as; `negative`, the failures only ([`crate::negative`]); and
//! `resonant`, the phase model ([`crate::resonance`]).  Words are not a kind -
//! they are an encoding - so a word model is any of them over `word:n:s`.
//!
//! Python's default kind is `radix`.  **This port's default is `count`**: it
//! was built as the count model's port, and a `radixnet` run with no `--kind`
//! keeps doing what it always did.  Everything else follows Python - a loaded
//! file's own kind wins over `--kind`, the default file follows the kind, and
//! each kind answers `train`, `predict`, `generate`, `score`, `feedback`,
//! `2nrl`, `invert`, `compress`, `info` and `weights` the way Python's does.
//!
//! The dispatch lives here rather than in every caller so that the CLI
//! (`src/bin/radixnet.rs`) and the server ([`crate::service`]) cannot answer a
//! kind differently.

use crate::encoding::Encoding;
use crate::json::Json;
use crate::model::{EpochRecord, Model};
use crate::radix::{Feedback, TrainConfig};

/// A model kind as Python's `model_kinds()` lists it.
pub struct KindInfo {
    pub kind: &'static str,
    pub label: &'static str,
    pub description: &'static str,
}

/// Every kind, in Python's order.
pub const KINDS: [KindInfo; 4] = [
    KindInfo {
        kind: "radix",
        label: "RadixNet (sine activation)",
        description: "edge weights and per-node sine activations learned by the one-hop rule; Dijkstra prediction; \
                      2NRL inverts the network",
    },
    KindInfo {
        kind: "count",
        label: "Count / reward",
        description: "edge weight = the edge's share of its node's traversals, all time and inside a sliding window, \
                      plus rewards - penalties; no learning rate; beam prediction with the top-K and bottom-K \
                      continuations",
    },
    KindInfo {
        kind: "negative",
        label: "Negative / blame",
        description: "the failures only: every edge keeps the blame it collected, how often it failed and the \
                      tutor's reasons; it judges text instead of writing it and filters the positive model's output",
    },
    KindInfo {
        kind: "resonant",
        label: "Resonant (phase)",
        description: "an analog phase rides along the walk: edges learn the phase at which they fire and how \
                      coherently, and a phase-locked cycle hands the decision to the metacognitive layer",
    },
];

/// The kind a new model is when nothing says otherwise - this port's, not
/// Python's (which is `radix`).
pub const DEFAULT_KIND: &str = "count";

/// The score-function settings each kind's `weights` command takes
/// (Python's `_WEIGHT_OPTIONS`); the other two kinds have none there.
pub const COUNT_WEIGHTS: [&str; 6] = [
    "count_scale",
    "global_scale",
    "window_scale",
    "reward_scale",
    "path_scale",
    "window",
];
pub const RESONANT_WEIGHTS: [&str; 7] = [
    "buckets",
    "period",
    "kick_scale",
    "resonance_scale",
    "amp_scale",
    "reward_scale",
    "concentration",
];

/// Every score-function setting a request body may carry, in the order
/// Python's `_weight_options` reads them.
pub const WEIGHT_FIELDS: [&str; 12] = [
    "count_scale",
    "global_scale",
    "window_scale",
    "path_scale",
    "window",
    "buckets",
    "period",
    "kick_scale",
    "resonance_scale",
    "amp_scale",
    "concentration",
    "reward_scale",
];

/// `[{"kind", "label", "description", "units"}]` - Python's `model_kinds()`.
pub fn kinds_json() -> Json {
    Json::Arr(
        KINDS
            .iter()
            .map(|k| {
                Json::obj([
                    ("kind", Json::str(k.kind)),
                    ("label", Json::str(k.label)),
                    ("description", Json::str(k.description)),
                    ("units", Json::str("chars")),
                ])
            })
            .collect(),
    )
}

/// What a kind is called.
pub fn label(kind: &str) -> &'static str {
    KINDS.iter().find(|k| k.kind == kind).map(|k| k.label).unwrap_or("")
}

/// A kind by name (`model_class`): trimmed and lower-cased, `""` the
/// default; anything else is Python's error.
pub fn parse_kind(name: &str) -> Result<&'static str, String> {
    let key = name.trim().to_lowercase();
    if key.is_empty() {
        return Ok(DEFAULT_KIND);
    }
    KINDS.iter().find(|k| k.kind == key).map(|k| k.kind).ok_or_else(|| {
        format!(
            "unknown model kind {}; expected one of: {}",
            crate::negative::python_repr(name),
            KINDS.iter().map(|k| k.kind).collect::<Vec<_>>().join(", ")
        )
    })
}

/// The `weights` options of a kind, `None` for a kind without a weight
/// function the command can change.
pub fn cli_weight_names(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "count" => Some(&COUNT_WEIGHTS),
        "resonant" => Some(&RESONANT_WEIGHTS),
        _ => None,
    }
}

/// A fresh model of `kind`, with the score-function `options` its
/// constructor takes - Python's `cls(seed=..., encoding=..., **options)`,
/// refusals worded as Python's are.
pub fn new_model(kind: &str, seed: i64, encoding: Encoding, options: &[(String, f64)]) -> Result<Model, String> {
    let unexpected = |class: &str, allowed: &[&str]| -> Result<(), String> {
        match options.iter().find(|(name, _)| !allowed.contains(&name.as_str())) {
            Some((name, _)) => Err(format!(
                "{class}.__init__() got an unexpected keyword argument {}",
                crate::negative::python_repr(name)
            )),
            None => Ok(()),
        }
    };
    let get = |name: &str| options.iter().find(|(n, _)| n == name).map(|(_, v)| *v);
    match parse_kind(kind)? {
        "radix" => {
            if !options.is_empty() {
                let mut names: Vec<&str> = options.iter().map(|(n, _)| n.as_str()).collect();
                names.sort_unstable();
                return Err(format!(
                    "weight options ({}) do not apply to the radix model",
                    names.join(", ")
                ));
            }
            Model::new_radix(seed, encoding)
        }
        "negative" => {
            unexpected("NegativeNet", &["share_scale", "blame_scale", "clear_scale"])?;
            let base = crate::negative::NegativeOptions::default();
            Model::new_negative(
                seed,
                &crate::negative::NegativeOptions {
                    share_scale: get("share_scale").unwrap_or(base.share_scale),
                    blame_scale: get("blame_scale").unwrap_or(base.blame_scale),
                    clear_scale: get("clear_scale").unwrap_or(base.clear_scale),
                    encoding,
                },
            )
        }
        "resonant" => {
            unexpected("ResonantNet", &RESONANT_WEIGHTS)?;
            let base = crate::resonance::ResonantOptions::default();
            let buckets = match get("buckets") {
                Some(v) if v < 1.0 => return Err(format!("buckets must be >= 1, got {}", v as i64)),
                Some(v) => v as usize,
                None => base.buckets,
            };
            let o = crate::resonance::ResonantOptions {
                buckets,
                period: get("period").unwrap_or(base.period),
                kick_scale: get("kick_scale").unwrap_or(base.kick_scale),
                resonance_scale: get("resonance_scale").unwrap_or(base.resonance_scale),
                amp_scale: get("amp_scale").unwrap_or(base.amp_scale),
                reward_scale: get("reward_scale").unwrap_or(base.reward_scale),
                concentration: get("concentration").unwrap_or(base.concentration),
            };
            Model::new_resonant(seed, &o, encoding)
        }
        _ => {
            unexpected(
                "CountRewardNet",
                &["count_scale", "reward_scale", "global_scale", "window_scale", "window"],
            )?;
            let base = crate::GraphOptions::default();
            let window = match get("window") {
                Some(v) if v < 1.0 => return Err(format!("window must be >= 1, got {}", v as i64)),
                Some(v) => v as usize,
                None => base.window,
            };
            Model::new(
                seed,
                crate::GraphOptions {
                    count_scale: get("count_scale").unwrap_or(base.count_scale),
                    reward_scale: get("reward_scale").unwrap_or(base.reward_scale),
                    global_scale: get("global_scale").unwrap_or(base.global_scale),
                    window_scale: get("window_scale").unwrap_or(base.window_scale),
                    window,
                    encoding,
                    ..base
                },
            )
        }
    }
}

/// The model's score function (`weight_config()`), `null` for the sine
/// model, whose weights are learned rather than computed.
pub fn weight_config(model: &Model) -> Json {
    if model.g.is_radix() {
        return Json::Null;
    }
    if let Some(r) = model.g.res.as_ref() {
        return r.weight_config(model.g.reward_scale);
    }
    if model.is_negative() {
        let mut pairs = vec![("function".to_string(), Json::str("blame"))];
        for (name, value) in model.g.negative_weight_config() {
            pairs.push((name.to_string(), Json::Num(value)));
        }
        pairs.push(("smoothing".to_string(), Json::Num(crate::weights::SMOOTHING)));
        return Json::Obj(pairs);
    }
    crate::report::weight_config(&model.g)
}

/// Changes the model's score function (`configure_weights`) and returns the
/// new one; the sine model has none to change.
pub fn configure_weights(model: &mut Model, options: &[(String, f64)]) -> Result<Json, String> {
    let kind = model.kind();
    match kind {
        "radix" => Err(format!(
            "the {kind} model has no configurable weight function (select another kind first)"
        )),
        "resonant" => model.g.configure_resonant(options),
        "negative" => {
            model.g.configure_negative(options)?;
            model.g.recompute_weights();
            Ok(weight_config(model))
        }
        _ => {
            for (name, _) in options {
                if !COUNT_WEIGHTS.contains(&name.as_str()) {
                    return Err(format!("unknown weight option {}", crate::negative::python_repr(name)));
                }
            }
            for (name, value) in options {
                if name != "window" && !value.is_finite() {
                    return Err(format!(
                        "{name} must be a finite number, got {}",
                        crate::json::py_repr(*value)
                    ));
                }
                crate::report::configure_weight(&mut model.g, name, *value)?;
            }
            model.g.invalidate();
            model.g.recompute_weights();
            Ok(weight_config(model))
        }
    }
}

/// `stats()` of any kind.
pub fn stats(model: &Model) -> Json {
    crate::report::stats(model)
}

/// The `meta` block as the kind's file writes it: the sine model keeps no
/// reward counters, the phase model adds the cycles its walks met.
pub fn meta_json(model: &Model) -> Json {
    if model.g.is_radix() {
        crate::radix::meta_json(model)
    } else if model.is_resonant() {
        crate::resonance::meta_json(model)
    } else {
        model.meta.to_json()
    }
}

/// How a training run goes, for every kind: each reads what applies to it.
#[derive(Clone, Debug)]
pub struct TrainSettings {
    /// The sine model's config; its `epochs` and `auto_compress` are every
    /// kind's.
    pub config: TrainConfig,
    /// The count model's chunk size (0: the default).
    pub chunk_size: usize,
    /// What a negative network is taught the texts failed for.
    pub reason: String,
    pub severity: f64,
    pub source: String,
    pub note: String,
}

impl Default for TrainSettings {
    fn default() -> TrainSettings {
        TrainSettings {
            config: TrainConfig::default(),
            chunk_size: 0,
            reason: String::new(),
            severity: 1.0,
            source: String::new(),
            note: String::new(),
        }
    }
}

/// Trains `model` on `texts` the way its kind trains: the sine model by
/// gradient descent, the count and phase models by counting, the negative
/// network by blaming (training it *is* blaming).  `on_epoch` sees every
/// record; returning `false` stops the run where the kind can stop.
pub fn train(
    model: &mut Model,
    texts: &[String],
    s: &TrainSettings,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    s.config.validate()?;
    match model.kind() {
        "radix" => model.radix_train(texts, &s.config, None, on_epoch),
        "resonant" => model.resonant_train(texts, s.config.epochs, s.config.auto_compress, None, on_epoch),
        "negative" => {
            let o = crate::negative::BlameOptions {
                reason: s.reason.clone(),
                severity: s.severity,
                source: s.source.clone(),
                note: s.note.clone(),
                epochs: s.config.epochs,
                no_compress: !s.config.auto_compress,
            };
            model.blame_with(texts, &o, on_epoch)
        }
        _ => {
            let opts = crate::model::TrainOptions {
                epochs: s.config.epochs,
                auto_compress: s.config.auto_compress,
                chunk_size: s.chunk_size,
                phase: None,
            };
            let records = model.train(texts, &opts)?;
            for record in &records {
                if !on_epoch(record) {
                    break;
                }
            }
            Ok(records)
        }
    }
}

/// What feedback does with the texts it is given: `"2nrl"` (both), `"reward"`
/// (good only), `"punish"` (bad only).
pub fn feedback_action(good: &[String], bad: &[String]) -> Option<&'static str> {
    match (good.is_empty(), bad.is_empty()) {
        (false, false) => Some("2nrl"),
        (false, true) => Some("reward"),
        (true, false) => Some("punish"),
        (true, true) => None,
    }
}

/// 2NRL on any kind - `(negative, positive)`: the sine model trains on the
/// bad texts, inverts and fine-tunes on the good ones; the count and phase
/// models punish and then count and reward; the negative network blames the
/// bad texts and clears the good ones.
pub fn two_nrl(
    model: &mut Model,
    bad: &[String],
    good: &[String],
    o: &Feedback,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<(Vec<EpochRecord>, Vec<EpochRecord>), String> {
    match model.kind() {
        "radix" => model.radix_two_nrl(bad, good, o, on_epoch),
        "resonant" => model.resonant_two_nrl(
            bad,
            good,
            o.neg_epochs,
            o.pos_epochs,
            o.strength,
            o.bad_weights.as_deref(),
            o.good_weights.as_deref(),
            on_epoch,
        ),
        "negative" => {
            let negative = negative_blame(model, bad, o, None, "", on_epoch)?;
            let positive = negative_clear(model, good, o, on_epoch)?;
            model.meta.twonrl_runs.add(1);
            Ok((negative, positive))
        }
        _ if o.bad_weights.is_some() || o.good_weights.is_some() => {
            let negative = match &o.bad_weights {
                None => model.punish(bad, o.neg_epochs, o.strength)?,
                Some(w) => count_weighted(model, bad, w, "bad_weights", o.neg_epochs, o.strength, false)?,
            };
            let positive = match &o.good_weights {
                None => model.reward(good, o.pos_epochs, o.strength)?,
                Some(w) => count_weighted(model, good, w, "weights", o.pos_epochs, o.strength, true)?,
            };
            model.meta.twonrl_runs.add(1);
            Ok((negative, positive))
        }
        _ => model.two_nrl(bad, good, o.neg_epochs, o.pos_epochs, o.strength),
    }
}

/// The count model's rated passes: one per group of equally weighted texts
/// (heaviest first), the reward or penalty scaled by the weight, and every
/// record - the history's too - carrying `weight`, as Python stamps them.
fn count_weighted(
    model: &mut Model,
    texts: &[String],
    weights: &[f64],
    name: &str,
    epochs: usize,
    strength: f64,
    positive: bool,
) -> Result<Vec<EpochRecord>, String> {
    let mut records = Vec::new();
    for (weight, group) in crate::radix::weight_groups(texts, weights, name)? {
        let amount = strength.abs() * weight;
        let mut done = if positive {
            model.reward(&group, epochs, amount)?
        } else {
            model.punish(&group, epochs, amount)?
        };
        let first = model.history.len() - done.len();
        for (i, record) in done.iter_mut().enumerate() {
            record.extra.push(("weight".to_string(), Json::Num(weight)));
            model.history[first + i] = record.clone();
        }
        records.extend(done);
    }
    Ok(records)
}

/// A thumbs up on any kind.
pub fn reward(
    model: &mut Model,
    good: &[String],
    o: &Feedback,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    match model.kind() {
        "radix" => model.radix_reward(good, o, on_epoch),
        "resonant" => model.resonant_reward(good, o.pos_epochs, o.strength, o.good_weights.as_deref(), on_epoch),
        "negative" => negative_clear(model, good, o, on_epoch),
        _ => match &o.good_weights {
            None => model.reward(good, o.pos_epochs, o.strength),
            Some(w) => count_weighted(model, good, w, "weights", o.pos_epochs, o.strength, true),
        },
    }
}

/// A thumbs down on any kind.
pub fn punish(
    model: &mut Model,
    bad: &[String],
    o: &Feedback,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    match model.kind() {
        "radix" => model.radix_punish(bad, o, on_epoch),
        "resonant" => model.resonant_punish(bad, o.neg_epochs, o.strength, o.bad_weights.as_deref(), on_epoch),
        "negative" => negative_blame(model, bad, o, Some("thumbs-down"), "feedback", on_epoch),
        _ => match &o.bad_weights {
            None => model.punish(bad, o.neg_epochs, o.strength),
            Some(w) => count_weighted(model, bad, w, "weights", o.neg_epochs, o.strength, false),
        },
    }
}

/// The negative network's thumbs down: blame, `strength` the severity, a
/// rating scaling it per text (one pass per rated text, the records carrying
/// `weight`).
fn negative_blame(
    model: &mut Model,
    bad: &[String],
    o: &Feedback,
    reason: Option<&str>,
    source: &str,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    let options = |severity: f64| crate::negative::BlameOptions {
        reason: reason.unwrap_or("").to_string(),
        severity,
        source: source.to_string(),
        epochs: o.neg_epochs,
        ..Default::default()
    };
    match &o.bad_weights {
        None => model.blame_with(bad, &options(o.strength), on_epoch),
        Some(weights) => {
            let mut records = Vec::new();
            for (text, &weight) in rated(bad, weights, "weights")? {
                let mut blamed =
                    model.blame_with(std::slice::from_ref(text), &options(o.strength * weight), &mut |_| true)?;
                for record in blamed.iter_mut() {
                    record.extra.push(("weight".to_string(), Json::Num(weight)));
                    on_epoch(record);
                }
                records.extend(blamed);
            }
            Ok(records)
        }
    }
}

/// The negative network's thumbs up: clearing, never learning *from* a
/// correct text.
fn negative_clear(
    model: &mut Model,
    good: &[String],
    o: &Feedback,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    match &o.good_weights {
        None => model.clear_with(good, o.strength, o.pos_epochs, on_epoch),
        Some(weights) => {
            let mut records = Vec::new();
            for (text, &weight) in rated(good, weights, "weights")? {
                let mut cleared = model.clear_with(
                    std::slice::from_ref(text),
                    o.strength * weight,
                    o.pos_epochs,
                    &mut |_| true,
                )?;
                for record in cleared.iter_mut() {
                    record.extra.push(("weight".to_string(), Json::Num(weight)));
                    on_epoch(record);
                }
                records.extend(cleared);
            }
            Ok(records)
        }
    }
}

/// Python's `_rated`: the texts paired with their weights, the zero weights
/// dropped.
fn rated<'a>(texts: &'a [String], weights: &'a [f64], name: &str) -> Result<Vec<(&'a String, &'a f64)>, String> {
    if weights.len() != texts.len() {
        return Err(format!(
            "{name} has {} entries for {} texts",
            weights.len(),
            texts.len()
        ));
    }
    let mut out = Vec::new();
    for (text, weight) in texts.iter().zip(weights) {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "{name} must be finite and >= 0, got {}",
                crate::json::py_repr(*weight)
            ));
        }
        if *weight > 0.0 {
            out.push((text, weight));
        }
    }
    Ok(out)
}

/// Marks out of 10 (`"8,10,6"`) as one weight per text - `--good-ratings`.
pub fn marks(raw: &str, texts: usize, option: &str) -> Result<Option<Vec<f64>>, String> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let mut out = Vec::new();
    for part in raw.replace(' ', "").split(',').filter(|p| !p.is_empty()) {
        let mark: f64 = part.parse().map_err(|_| {
            format!(
                "{option} must be a comma-separated list of marks out of 10: could not convert string to float: \
                 {}",
                crate::negative::python_repr(part)
            )
        })?;
        out.push(mark / 10.0);
    }
    if out.len() != texts {
        return Err(format!("{option} has {} marks for {texts} texts", out.len()));
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_is_listed_and_named_as_python_names_it() {
        let listed = kinds_json();
        let names: Vec<&str> = listed.as_array().iter().filter_map(|k| k.at("kind").as_str()).collect();
        assert_eq!(names, vec!["radix", "count", "negative", "resonant"]);
        assert_eq!(parse_kind(" Resonant ").unwrap(), "resonant");
        assert_eq!(parse_kind("").unwrap(), DEFAULT_KIND);
        let err = parse_kind("sine").unwrap_err();
        assert_eq!(
            err,
            "unknown model kind 'sine'; expected one of: radix, count, negative, resonant"
        );
    }

    #[test]
    fn a_new_model_takes_the_options_its_kind_has() {
        let enc = Encoding::default();
        for kind in ["radix", "count", "negative", "resonant"] {
            assert_eq!(new_model(kind, 1, enc, &[]).unwrap().kind(), kind);
        }
        let m = new_model("resonant", 1, enc, &[("buckets".to_string(), 12.0)]).unwrap();
        assert_eq!(weight_config(&m).at("buckets").as_i64(), Some(12));
        let err = new_model("radix", 1, enc, &[("window".to_string(), 3.0)])
            .err()
            .unwrap();
        assert_eq!(err, "weight options (window) do not apply to the radix model");
        let err = new_model("count", 1, enc, &[("buckets".to_string(), 3.0)])
            .err()
            .unwrap();
        assert_eq!(
            err,
            "CountRewardNet.__init__() got an unexpected keyword argument 'buckets'"
        );
        assert!(weight_config(&new_model("radix", 1, enc, &[]).unwrap()).is_null());
    }

    #[test]
    fn every_kind_trains_and_takes_feedback() {
        let texts: Vec<String> = ["the cat sat on the mat", "the dog sat on the log"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for kind in ["radix", "count", "negative", "resonant"] {
            let mut m = new_model(kind, 3, Encoding::default(), &[]).unwrap();
            let s = TrainSettings {
                config: TrainConfig {
                    epochs: 2,
                    ..Default::default()
                },
                ..Default::default()
            };
            let records = train(&mut m, &texts, &s, &mut |_| true).unwrap();
            assert!(!records.is_empty(), "{kind}");
            let o = Feedback::two_nrl();
            let (negative, positive) = two_nrl(&mut m, &texts[..1], &texts[1..], &o, &mut |_| true).unwrap();
            assert!(!negative.is_empty() || !positive.is_empty(), "{kind}");
        }
    }

    #[test]
    fn marks_are_tenths() {
        assert_eq!(marks("8, 10", 2, "--good-ratings").unwrap(), Some(vec![0.8, 1.0]));
        assert!(marks("8", 2, "--good-ratings").is_err());
        assert_eq!(marks("", 2, "--good-ratings").unwrap(), None);
    }
}
