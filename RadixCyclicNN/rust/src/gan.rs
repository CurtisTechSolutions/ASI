//! The self-upgrade loop: the model is the generator, a second network the
//! discriminator (`radixnet/gan.py`, `go/radixnet/gan.go`,
//! `go/server/evolve.go`).
//!
//! Every generation:
//!
//! 1. the generator samples `samples` fakes (its own seeded RNG, so a run is
//!    the same run every time);
//! 2. the discriminator - a second count model on the generator's encoding -
//!    runs 2NRL over the fakes (bad) and a sample of the real corpus (good),
//!    then scores both sides *per character*, so long and short texts compare;
//! 3. the fakes it scores below the real texts are the **failures**, and how
//!    far below (the gap, in nats per character) is how bad each one is;
//! 4. the blatant mode decides what the generator does with them - `none`:
//!    the worst half of the fakes is ordinary 2NRL garbage; `fail_invert`:
//!    every failure is penalised in proportion to how bad it is; `activation`
//!    / `state`: the count model has no activation to flip, so every edge of a
//!    failed path loses reward in proportion to its gap, and the blatant ones
//!    (past the margin) leave the 2NRL garbage set.
//!
//! With `blame`, the discriminator is also the **negative network's tutor**:
//! every failure is blamed (`discriminator`, or `blatant` past the margin) by
//! how far below the real texts it landed, and the real texts clear blame
//! (through [`crate::blame::teach`]).
//!
//! It is Python's loop to the number: the real texts are drawn the way
//! `random.Random(seed).sample` draws them, the means are `fsum` means, and
//! `invert_paths` penalises at Python's strength (2, so a full failure costs
//! an edge 4) - so an evolve run here and one in Python write the same
//! generator, the same discriminator and the same negative network.  (Go
//! draws with its own generator and penalises at half that.)

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::blame::{teach, Fault, TeachOptions, TeachReport};
use crate::checkpoint::{Checkpoints, Schedule};
use crate::cli::{negative_stats, Ctx};
use crate::encoding::Encoding;
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::model::{EpochRecord, GenerateOptions, Model};
use crate::mt19937::Mt19937;
use crate::negative::NegativeOptions;
use crate::report::stats;
use crate::service::Service;
use crate::GraphOptions;

/// What the loop's lines are filed under.
const LOG: &str = "evolve";

/// How failed fakes drive the generator's update.
pub const BLATANT_MODES: &[&str] = &["none", "fail_invert", "activation", "state"];

/// The strength `invert_paths` penalises a failed path at on the count model
/// (Python's default): a failure of amount 1 costs each edge `2 * 2 = 4`.
const INVERT_STRENGTH: f64 = 2.0;

/// The loop's hyper-parameters (Python's `EvolveConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct EvolveConfig {
    /// Fakes generated per generation.
    pub samples: usize,
    /// Real corpus texts drawn per generation.
    pub real_per_generation: usize,
    /// Units per fake.
    pub max_length: usize,
    pub temperature: f64,
    /// The generator's 2NRL epochs.
    pub neg_epochs: usize,
    pub pos_epochs: usize,
    /// Accepted for the interface's sake; the count model has no learning rate.
    pub neg_lr: f64,
    pub pos_lr: f64,
    /// The discriminator's 2NRL epochs.
    pub disc_neg_epochs: usize,
    pub disc_pos_epochs: usize,
    /// Accepted for the interface's sake; the count model has no batches.
    pub batch_size: usize,
    /// Checkpoint the generator every N generations (0 = off).
    pub checkpoint_every: usize,
    pub seed: i64,
    /// One of [`BLATANT_MODES`].
    pub blatant_mode: String,
    /// The per-character gap past which a failure is blatant.
    pub blatant_margin: f64,
    /// `fail_invert`: the largest multiplier a failure's penalty gets.
    pub blatant_boost: f64,
    /// The reward / penalty of one 2NRL pass (Go's knob; 1 is Python's).
    pub strength: f64,
}

impl Default for EvolveConfig {
    /// Python's defaults.
    fn default() -> EvolveConfig {
        EvolveConfig {
            samples: 8,
            real_per_generation: 8,
            max_length: 40,
            temperature: 1.0,
            neg_epochs: 1,
            pos_epochs: 1,
            neg_lr: 0.05,
            pos_lr: 0.01,
            disc_neg_epochs: 1,
            disc_pos_epochs: 1,
            batch_size: 32,
            checkpoint_every: 0,
            seed: 0,
            blatant_mode: "none".to_string(),
            blatant_margin: 1.0,
            blatant_boost: 4.0,
            strength: 1.0,
        }
    }
}

impl EvolveConfig {
    /// Refuses values the loop cannot run with; `max_length` is in the units
    /// of `encoding`, and a fake has to hold one gram of it.
    pub fn validate(&self, encoding: &Encoding) -> Result<(), String> {
        if !BLATANT_MODES.contains(&self.blatant_mode.as_str()) {
            return Err(format!(
                "blatant_mode must be one of {}, got {:?}",
                BLATANT_MODES.join(", "),
                self.blatant_mode
            ));
        }
        // NaN fails every one of these, as it fails Python's `not (x >= 0)`
        if self.blatant_margin.is_nan() || self.blatant_margin < 0.0 {
            return Err(format!("blatant_margin must be >= 0, got {}", self.blatant_margin));
        }
        if self.blatant_boost.is_nan() || self.blatant_boost < 1.0 {
            return Err(format!("blatant_boost must be >= 1, got {}", self.blatant_boost));
        }
        if self.samples < 1 {
            return Err(format!("samples must be >= 1, got {}", self.samples));
        }
        if self.real_per_generation < 1 {
            return Err(format!(
                "real_per_generation must be >= 1, got {}",
                self.real_per_generation
            ));
        }
        if self.max_length < encoding.n {
            return Err(format!("max_length must be >= {}, got {}", encoding.n, self.max_length));
        }
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err(format!("temperature must be >= 0, got {}", self.temperature));
        }
        if self.batch_size < 1 {
            return Err(format!("batch_size must be >= 1, got {}", self.batch_size));
        }
        if !self.strength.is_finite() || self.strength < 0.0 {
            return Err(format!("strength must be >= 0, got {}", self.strength));
        }
        Ok(())
    }

    /// The configuration in Python's `dataclasses.asdict` order, then `strength`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("samples", Json::Int(self.samples as i64)),
            ("real_per_generation", Json::Int(self.real_per_generation as i64)),
            ("max_length", Json::Int(self.max_length as i64)),
            ("temperature", Json::Num(self.temperature)),
            ("neg_epochs", Json::Int(self.neg_epochs as i64)),
            ("pos_epochs", Json::Int(self.pos_epochs as i64)),
            ("neg_lr", Json::Num(self.neg_lr)),
            ("pos_lr", Json::Num(self.pos_lr)),
            ("disc_neg_epochs", Json::Int(self.disc_neg_epochs as i64)),
            ("disc_pos_epochs", Json::Int(self.disc_pos_epochs as i64)),
            ("batch_size", Json::Int(self.batch_size as i64)),
            ("checkpoint_every", Json::Int(self.checkpoint_every as i64)),
            ("seed", Json::Int(self.seed)),
            ("blatant_mode", Json::str(self.blatant_mode.clone())),
            ("blatant_margin", Json::Num(self.blatant_margin)),
            ("blatant_boost", Json::Num(self.blatant_boost)),
            ("strength", Json::Num(self.strength)),
        ])
    }
}

// -- Python's arithmetic ------------------------------------------------------------------------

/// `random.Random._randbelow`: `getrandbits(k)` until it lands below `n`
/// (one 32-bit word per draw, which covers every corpus below 2^32 texts).
fn randbelow(rng: &mut Mt19937, n: usize) -> usize {
    let k = usize::BITS - n.leading_zeros();
    loop {
        let r = (rng.next_u32() >> (32 - k.min(32))) as usize;
        if r < n {
            return r;
        }
    }
}

/// `random.Random.sample(range(n), k)`, draw for draw: a pool for a small
/// population, a set of the chosen for a large one - as CPython chooses.
fn sample_indices(rng: &mut Mt19937, n: usize, k: usize) -> Vec<usize> {
    let mut setsize = 21usize;
    if k > 5 {
        setsize += 4usize.pow(((k * 3) as f64).log(4.0).ceil() as u32);
    }
    let mut out = Vec::with_capacity(k);
    if n <= setsize {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = randbelow(rng, n - i);
            out.push(pool[j]);
            pool[j] = pool[n - i - 1];
        }
    } else {
        let mut chosen = std::collections::HashSet::with_capacity(k);
        for _ in 0..k {
            let mut j = randbelow(rng, n);
            while chosen.contains(&j) {
                j = randbelow(rng, n);
            }
            chosen.insert(j);
            out.push(j);
        }
    }
    out
}

/// `statistics.fmean`: the exact sum over the count; `None` for no values.
fn fmean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| crate::fsum::fsum(values) / values.len() as f64)
}

/// `statistics.median`: the middle value, or the mean of the two middles.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    match n {
        0 => 0.0,
        _ if n % 2 == 1 => sorted[n / 2],
        _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// `round(x, 3)`: the nearest three-decimal number, as the decimal string rounds it.
fn round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

// -- what the loop asks of the count model ---------------------------------------------------------

/// `[(weight, texts)]`: the texts of equal (three-decimal) weight together,
/// heaviest first; a weight of 0 drops its text (Python's `_weight_groups`).
pub fn weight_groups(texts: &[String], weights: &[f64], name: &str) -> Result<Vec<(f64, Vec<String>)>, String> {
    if weights.len() != texts.len() {
        return Err(format!(
            "{name} has {} entries for {} texts",
            weights.len(),
            texts.len()
        ));
    }
    let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
    for (text, &weight) in texts.iter().zip(weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!("{name} must be finite and >= 0, got {weight}"));
        }
        if weight > 0.0 {
            let key = round3(weight);
            match groups.iter_mut().find(|(w, _)| *w == key) {
                Some((_, group)) => group.push(text.clone()),
                None => groups.push((key, vec![text.clone()])),
            }
        }
    }
    groups.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(groups)
}

/// 2NRL with a mark per failure: each group of equally bad texts is punished
/// at `strength * weight` (its records carry the weight), then the good ones
/// are counted and rewarded - the count model's `two_nrl(bad_weights=...)`.
pub fn two_nrl_weighted(
    model: &mut Model,
    bad: &[String],
    weights: &[f64],
    good: &[String],
    neg_epochs: usize,
    pos_epochs: usize,
    strength: f64,
) -> Result<(Vec<EpochRecord>, Vec<EpochRecord>), String> {
    let mut negative = Vec::new();
    for (weight, group) in weight_groups(bad, weights, "bad_weights")? {
        let mut records = model.punish(&group, neg_epochs, strength * weight)?;
        // the weight goes into the history too: Python tags the one record it
        // keeps in both places, and the pass's records are the history's last
        let first = model.history.len() - records.len().min(model.history.len());
        for (record, kept) in records.iter_mut().zip(&mut model.history[first..]) {
            record.extra.push(("weight".to_string(), Json::Num(weight)));
            kept.extra.push(("weight".to_string(), Json::Num(weight)));
        }
        negative.extend(records);
    }
    let positive = model.reward(good, pos_epochs, strength)?;
    model.meta.twonrl_runs.add(1);
    Ok((negative, positive))
}

/// What [`invert_paths`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct InvertOutcome {
    pub texts: usize,
    /// Edges penalised.
    pub flipped: usize,
    pub amount_mean: f64,
}

impl InvertOutcome {
    /// `{"texts", "flipped", "unit": "edges", "mode": "penalty", "amount_mean"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("texts", Json::Int(self.texts as i64)),
            ("flipped", Json::Int(self.flipped as i64)),
            ("unit", Json::str("edges")),
            ("mode", Json::str("penalty")),
            ("amount_mean", Json::Num(self.amount_mean)),
        ])
    }
}

/// The count model's inversion of failed paths: there is no activation to
/// flip, so every edge of a text's path loses `strength * 2 * amount` reward
/// (an edge on several paths takes the largest).  A text the structure cannot
/// walk yet is registered first.
pub fn invert_paths(
    model: &mut Model,
    texts: &[String],
    amounts: &[f64],
    strength: f64,
) -> Result<InvertOutcome, String> {
    let enc = model.encoding();
    let texts: Vec<&String> = texts.iter().filter(|t| enc.len(t) >= enc.n).collect();
    if amounts.len() != texts.len() {
        return Err(format!(
            "amounts has {} entries for {} texts",
            amounts.len(),
            texts.len()
        ));
    }
    if let Some(bad) = amounts.iter().find(|a| !(0.0..=1.0).contains(*a)) {
        return Err(format!("amounts must lie in [0, 1], got {bad}"));
    }
    let mut paths: Vec<Vec<usize>> = Vec::new();
    for text in &texts {
        let grams = enc.encode(text);
        if grams.is_empty() {
            continue;
        }
        let path = match model.g.node_path(&grams) {
            Some(path) => Some(path),
            None => {
                model.g.observe(&grams, false)?;
                model.g.node_path(&grams)
            }
        };
        if let Some(path) = path.filter(|p| !p.is_empty()) {
            paths.push(path);
        }
    }
    let mut penalties: Vec<(usize, f64)> = Vec::new();
    for (path, &amount) in paths.iter().zip(amounts) {
        let penalty = strength.abs() * 2.0 * amount;
        if penalty <= 0.0 {
            continue;
        }
        for pair in path.windows(2) {
            if let Some(e) = model.g.edge(pair[0], pair[1]) {
                match penalties.iter_mut().find(|(edge, _)| *edge == e) {
                    Some((_, most)) => *most = most.max(penalty),
                    None => penalties.push((e, penalty)),
                }
            }
        }
    }
    let mut touched = 0usize;
    for (e, penalty) in &penalties {
        touched += model.g.add_reward(&[*e], -penalty);
        model.meta.penalties_total += penalty;
    }
    if touched > 0 {
        model.meta.feedback_passes.add(1);
    }
    let applied: Vec<f64> = amounts.iter().copied().filter(|&a| a > 0.0).collect();
    Ok(InvertOutcome {
        texts: texts.len(),
        flipped: touched,
        amount_mean: if applied.is_empty() {
            0.0
        } else {
            applied.iter().sum::<f64>() / applied.len() as f64
        },
    })
}

// -- the loop ----------------------------------------------------------------------------------------

/// The loop: a corpus of real texts, the discriminator, and every generation's record.
pub struct Evolver {
    pub config: EvolveConfig,
    /// The real texts, those too short to hold one gram left out.
    pub corpus: Vec<String>,
    pub discriminator: Model,
    /// Generations run so far.
    pub generation: usize,
    pub history: Vec<Json>,
    rng: Mt19937,
}

/// What the generator's half of a generation leaves for the negative
/// network's: the record so far and what the critic made of the fakes.
pub struct Pending {
    record: Vec<(String, Json)>,
    fakes: Vec<String>,
    scores: Vec<f64>,
    real_mean: Option<f64>,
    real: Vec<String>,
    started: Instant,
}

impl Evolver {
    /// Builds the loop over `corpus`; without a `discriminator` a fresh count
    /// model is made on the generator's encoding (two networks on different
    /// encodings judge different grams), seeded one past the loop's seed.
    pub fn new(
        generator: &Model,
        corpus: &[String],
        discriminator: Option<Model>,
        config: EvolveConfig,
    ) -> Result<Evolver, String> {
        if generator.is_negative() {
            return Err("the evolve loop improves the count model, not the negative network".to_string());
        }
        let enc = generator.encoding();
        config.validate(&enc)?;
        let corpus: Vec<String> = corpus.iter().filter(|t| !enc.encode(t).is_empty()).cloned().collect();
        if corpus.is_empty() {
            return Err(format!(
                "corpus needs at least one text of {}+ {}",
                enc.n,
                enc.units_name()
            ));
        }
        let discriminator = match discriminator {
            Some(d) => d,
            None => {
                let mut fresh = Model::new(
                    config.seed + 1,
                    GraphOptions {
                        encoding: enc,
                        ..Default::default()
                    },
                )?;
                fresh.workers = generator.workers;
                fresh.g.workers = generator.workers;
                fresh
            }
        };
        Ok(Evolver {
            rng: Mt19937::new(config.seed),
            config,
            corpus,
            discriminator,
            generation: 0,
            history: Vec::new(),
        })
    }

    /// This generation's fakes: samples of the generator, falling back to its
    /// cheapest text when every sample came out shorter than one gram.
    fn fakes(&self, generator: &mut Model) -> Result<Vec<String>, String> {
        let cfg = &self.config;
        let enc = generator.encoding();
        let sampled = generator.generate(&GenerateOptions {
            max_length: cfg.max_length,
            mode: "sample".to_string(),
            temperature: cfg.temperature,
            count: cfg.samples,
            ..Default::default()
        })?;
        let mut fakes: Vec<String> = sampled
            .into_iter()
            .map(|r| r.text)
            .filter(|t| !enc.encode(t).is_empty())
            .collect();
        if fakes.is_empty() {
            let best = generator.generate(&GenerateOptions {
                max_length: cfg.max_length,
                mode: "dijkstra".to_string(),
                ..Default::default()
            })?;
            if let Some(text) = best.into_iter().next().map(|r| r.text) {
                if !enc.encode(&text).is_empty() {
                    fakes.push(text);
                }
            }
        }
        Ok(fakes)
    }

    /// The generator's half of one generation: fakes, the critic's lesson and
    /// marks, and the generator's update.  [`Evolver::finish`] completes it.
    pub fn step(&mut self, generator: &mut Model) -> Result<Pending, String> {
        let started = Instant::now();
        let cfg = self.config.clone();
        let fakes = self.fakes(generator)?;
        let want = cfg.real_per_generation.min(self.corpus.len());
        let real: Vec<String> = sample_indices(&mut self.rng, self.corpus.len(), want)
            .into_iter()
            .map(|i| self.corpus[i].clone())
            .collect();
        // the critic learns real from fake, then marks both
        let disc = &mut self.discriminator;
        disc.two_nrl(&fakes, &real, cfg.disc_neg_epochs, cfg.disc_pos_epochs, cfg.strength)?;
        let scores: Vec<f64> = fakes.iter().map(|f| disc.score(f).per_char).collect();
        let real_scores: Vec<f64> = real.iter().map(|r| disc.score(r).per_char).collect();
        let (fake_mean, real_mean) = (fmean(&scores), fmean(&real_scores));
        let mut worst: Vec<String> = Vec::new();
        if !fakes.is_empty() {
            let cut = median(&scores);
            worst = fakes
                .iter()
                .zip(&scores)
                .filter(|(_, &s)| s <= cut)
                .map(|(f, _)| f.clone())
                .collect();
            if worst.is_empty() {
                worst.push(fakes[0].clone());
            }
        }
        // the failures: fakes the critic scored below the real texts, each with its gap
        let (mut failures, mut gaps, mut blatant) = (Vec::new(), Vec::new(), Vec::new());
        let mode = cfg.blatant_mode.as_str();
        if let Some(real_mean) = real_mean.filter(|_| mode != "none" && cfg.blatant_margin > 0.0) {
            for (fake, score) in fakes.iter().zip(&scores) {
                let gap = real_mean - score;
                if gap <= 0.0 {
                    continue;
                }
                failures.push(fake.clone());
                gaps.push(gap);
                if gap > cfg.blatant_margin {
                    blatant.push(fake.clone());
                }
            }
        }
        let (mut twonrl, mut flipped) = (false, 0usize);
        let (mut boost_mean, mut boost_max) = (0.0f64, 0.0f64);
        let positive: Vec<EpochRecord>;
        if mode == "fail_invert" && !failures.is_empty() {
            // blatantly fail on purpose: the worse the fake, the harder it is pushed away
            let weights: Vec<f64> = gaps
                .iter()
                .map(|gap| cfg.blatant_boost.min(1.0 + gap / cfg.blatant_margin))
                .collect();
            let (_, pos) = two_nrl_weighted(
                generator,
                &failures,
                &weights,
                &real,
                cfg.neg_epochs,
                cfg.pos_epochs,
                cfg.strength,
            )?;
            positive = pos;
            worst = failures.clone();
            twonrl = true;
            boost_mean = fmean(&weights).unwrap_or(0.0);
            boost_max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        } else {
            if mode == "fail_invert" {
                worst.clear(); // nothing scored below the real texts: no failure to train on
            }
            if (mode == "activation" || mode == "state") && !failures.is_empty() {
                // the local variant: only the failed paths move, the worse the more
                let amounts: Vec<f64> = gaps
                    .iter()
                    .map(|gap| 1.0f64.min(gap / (2.0 * cfg.blatant_margin)))
                    .collect();
                let outcome = invert_paths(generator, &failures, &amounts, INVERT_STRENGTH * cfg.strength)?;
                flipped = outcome.flipped;
                boost_mean = outcome.amount_mean;
                boost_max = amounts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                worst.retain(|w| !blatant.contains(w));
            }
            if !worst.is_empty() {
                let (_, pos) = generator.two_nrl(&worst, &real, cfg.neg_epochs, cfg.pos_epochs, cfg.strength)?;
                positive = pos;
                twonrl = true;
            } else {
                // nothing left for the negative pass: only the fine-tune pass
                positive = generator.reward(&real, cfg.pos_epochs, cfg.strength)?;
            }
        }
        self.generation += 1;
        let g = &generator.g;
        let maybe = |x: Option<f64>| x.map(Json::Num).unwrap_or(Json::Null);
        let record = vec![
            ("generation".to_string(), Json::Int(self.generation as i64)),
            ("fake_score_mean".to_string(), maybe(fake_mean)),
            ("real_score_mean".to_string(), maybe(real_mean)),
            (
                "gap".to_string(),
                maybe(fake_mean.zip(real_mean).map(|(fake, real)| real - fake)),
            ),
            ("gen_loss".to_string(), maybe(positive.last().map(|r| r.loss))),
            ("nodes".to_string(), Json::Int(g.num_nodes() as i64)),
            ("edges".to_string(), Json::Int(g.num_edges() as i64)),
            ("compression_ratio".to_string(), Json::Num(g.compression_ratio())),
            (
                "sample".to_string(),
                Json::str(fakes.first().cloned().unwrap_or_default()),
            ),
            ("fakes".to_string(), Json::Int(fakes.len() as i64)),
            ("worst".to_string(), Json::Int(worst.len() as i64)),
            ("failures".to_string(), Json::Int(failures.len() as i64)),
            ("blatant".to_string(), Json::Int(blatant.len() as i64)),
            ("flipped".to_string(), Json::Int(flipped as i64)),
            ("boost_mean".to_string(), Json::Num(boost_mean)),
            ("boost_max".to_string(), Json::Num(boost_max)),
            ("twonrl".to_string(), Json::Bool(twonrl)),
            ("mode".to_string(), Json::str(cfg.blatant_mode.clone())),
        ];
        Ok(Pending {
            record,
            fakes,
            scores,
            real_mean,
            real,
            started,
        })
    }

    /// The negative network's half: the critic is its tutor - every fake it
    /// scored below the real texts is blamed by how far below, and the real
    /// texts clear blame.  Then the record is complete, and kept.
    pub fn finish(
        &mut self,
        pending: Pending,
        negative: Option<&mut Model>,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Json, String> {
        let taught = match negative {
            Some(negative) => self.teach_negative(negative, &pending, stop)?,
            None => None,
        };
        let mut record = pending.record;
        record.push((
            "seconds".to_string(),
            Json::Num(pending.started.elapsed().as_secs_f64()),
        ));
        if let Some(report) = taught {
            record.push(("negative_blamed".to_string(), Json::Int(report.blamed as i64)));
            record.push(("negative_edges".to_string(), Json::Int(report.edges as i64)));
            record.push((
                "negative_reasons".to_string(),
                Json::Obj(
                    report
                        .reasons
                        .iter()
                        .map(|(r, n)| (r.clone(), Json::Int(*n as i64)))
                        .collect(),
                ),
            ));
        }
        let record = Json::Obj(record);
        crate::log_info!(
            LOG,
            "generation {}: fake {} real {} gap {}, {} failure(s), sample {:?}",
            self.generation,
            fmt_mark(record.at("fake_score_mean")),
            fmt_mark(record.at("real_score_mean")),
            fmt_mark(record.at("gap")),
            record.at("failures").as_i64().unwrap_or(0),
            record.at("sample").as_str().unwrap_or("")
        );
        self.history.push(record.clone());
        Ok(record)
    }

    fn teach_negative(
        &self,
        negative: &mut Model,
        pending: &Pending,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<TeachReport>, String> {
        let Some(real_mean) = pending.real_mean else {
            return Ok(None);
        };
        let cfg = &self.config;
        let margin = if cfg.blatant_margin > 0.0 {
            cfg.blatant_margin
        } else {
            1.0
        };
        let faults: Vec<Fault> = pending
            .fakes
            .iter()
            .zip(&pending.scores)
            .filter_map(|(fake, score)| {
                let gap = real_mean - score;
                (gap > 0.0).then(|| {
                    Fault::new(
                        fake,
                        if gap > cfg.blatant_margin {
                            "blatant"
                        } else {
                            "discriminator"
                        },
                        (gap / margin).clamp(0.25, 2.0),
                        &format!("the discriminator scored it {gap:.3} per char below the real texts"),
                        "evolve",
                    )
                })
            })
            .collect();
        if faults.is_empty() {
            return Ok(Some(TeachReport::default()));
        }
        let options = TeachOptions {
            stop: Some(stop),
            ..Default::default()
        };
        teach(negative, &faults, &pending.real, &options).map(Some)
    }

    /// One whole generation; returns (and keeps) its record.
    pub fn run_generation(&mut self, generator: &mut Model, negative: Option<&mut Model>) -> Result<Json, String> {
        let pending = self.step(generator)?;
        self.finish(pending, negative, &|| false)
    }

    /// `generations` generations (0: until `stop` says so), checked before
    /// every one; with a manager the generator is checkpointed every
    /// `config.checkpoint_every` generations (tag `gen`).  `progress` sees
    /// every record as it is made.
    pub fn run(
        &mut self,
        generator: &mut Model,
        mut negative: Option<&mut Model>,
        generations: usize,
        manager: Option<&Checkpoints>,
        stop: &(dyn Fn() -> bool + Sync),
        progress: &mut dyn FnMut(&Json),
    ) -> Result<Vec<Json>, String> {
        let every = self.config.checkpoint_every;
        let mut records = Vec::new();
        while generations == 0 || records.len() < generations {
            if stop() {
                break;
            }
            let pending = self.step(generator)?;
            let record = self.finish(pending, negative.as_deref_mut(), stop)?;
            progress(&record);
            if let Some(manager) = manager.filter(|_| every > 0 && self.generation % every == 0) {
                manager.save(generator, self.generation as i64, "gen", Some(record.clone()))?;
            }
            records.push(record);
        }
        Ok(records)
    }
}

/// A mark for a log line: three decimals, or `-` when there is none.
fn fmt_mark(value: &Json) -> String {
    value
        .as_f64()
        .map(|x| format!("{x:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

// -- the command -------------------------------------------------------------------------------------

/// The loop's settings from the command line, over Python's defaults.
fn config_from_args(ctx: &Ctx, every: usize) -> Result<EvolveConfig, String> {
    let a = &ctx.args;
    let d = EvolveConfig::default();
    Ok(EvolveConfig {
        samples: a.usize("samples", d.samples)?,
        real_per_generation: a.usize("real-per-generation", d.real_per_generation)?,
        max_length: a.usize("max-length", d.max_length)?,
        temperature: a.float("temperature", d.temperature)?,
        neg_epochs: a.usize("neg-epochs", d.neg_epochs)?,
        pos_epochs: a.usize("pos-epochs", d.pos_epochs)?,
        neg_lr: a.float("neg-lr", d.neg_lr)?,
        pos_lr: a.float("pos-lr", d.pos_lr)?,
        disc_neg_epochs: a.usize("disc-neg-epochs", d.disc_neg_epochs)?,
        disc_pos_epochs: a.usize("disc-pos-epochs", d.disc_pos_epochs)?,
        batch_size: a.usize("batch-size", d.batch_size)?,
        checkpoint_every: every,
        seed: ctx.seed,
        blatant_mode: a.str("blatant-mode", &d.blatant_mode),
        blatant_margin: a.float("blatant-margin", d.blatant_margin)?,
        blatant_boost: a.float("blatant-boost", d.blatant_boost)?,
        strength: a.float("strength", d.strength)?,
    })
}

/// `{"path", "bytes"}` of a file just written.
fn saved_doc(path: &str) -> Json {
    let bytes = std::fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
    Json::obj([("path", Json::str(path)), ("bytes", Json::Int(bytes))])
}

/// The negative network's reason table as Python reports it: `{"reason",
/// "blame", "fails", "fails_resets", "edges", "share"}`, where `edges` counts
/// the live edges that carry the reason.
fn reason_rows(model: &Model) -> Json {
    let rows = model.reasons();
    let mut edges = vec![0i64; rows.iter().map(|r| r.id + 1).max().unwrap_or(0)];
    for e in (0..model.g.num_edge_ids()).filter(|&e| model.g.is_edge_alive(e)) {
        for held in model.g.edge_reasons(e) {
            if let Some(n) = edges.get_mut(held.id) {
                *n += 1;
            }
        }
    }
    Json::Arr(
        rows.iter()
            .map(|r| {
                Json::obj([
                    ("reason", Json::str(r.reason.clone())),
                    ("blame", Json::Num(r.blame)),
                    ("fails", Json::Int(r.fails)),
                    ("fails_resets", Json::Int(r.fails_resets)),
                    ("edges", Json::Int(edges.get(r.id).copied().unwrap_or(0))),
                    ("share", Json::Num(r.share)),
                ])
            })
            .collect(),
    )
}

/// `radixnet evolve --data FILE [--generations N] ...`: the self-upgrade loop
/// over a real corpus, then the generator saved (and the discriminator with
/// `--discriminator`, the negative network with `--blame`).
///
/// `--generations 0` runs until the process is stopped.  The standard library
/// cannot catch Ctrl-C, so such a run saves after every generation instead of
/// once at the end: an interrupt loses the generation in flight and nothing
/// else.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let data = ctx
        .args
        .get("data")
        .ok_or("evolve needs --data FILE: the real corpus the critic learns from")?
        .to_string();
    let corpus = crate::cli::read_named(&ctx.args, "data")?;
    if corpus.is_empty() {
        return Err(format!("no corpus texts found in {data}"));
    }
    let schedule = Schedule::from_args(&ctx.args)?;
    let origin = if Path::new(&ctx.model_path).exists() {
        Json::obj([
            ("kind", Json::str("model")),
            ("path", Json::str(ctx.model_path.clone())),
        ])
    } else {
        Json::obj([("kind", Json::str("new")), ("path", Json::Null)])
    };
    let mut generator = ctx.open(false)?;
    let disc_path = ctx.args.get("discriminator").map(str::to_string);
    let discriminator = match &disc_path {
        Some(path) if Path::new(path).exists() => {
            let mut d = Model::load(path)?;
            d.workers = ctx.workers;
            d.g.workers = ctx.workers;
            Some(d)
        }
        _ => None,
    };
    let config = config_from_args(ctx, schedule.every)?;
    let mut negative = if ctx.args.on("blame") {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let mut evolver = Evolver::new(&generator, &corpus, discriminator, config)?;
    let generations = ctx.args.usize("generations", 0)?;
    let out = ctx.out.clone().unwrap_or_else(|| ctx.model_path.clone());
    let neg_path = ctx.negative_path();
    crate::log_info!(
        LOG,
        "{} over {} real text(s), {} fake(s) and {} real per generation, blatant mode {}",
        if generations == 0 {
            "until stopped (saving after every generation)".to_string()
        } else {
            format!("{generations} generation(s)")
        },
        evolver.corpus.len(),
        evolver.config.samples,
        evolver.config.real_per_generation,
        evolver.config.blatant_mode
    );
    let manager = schedule.manager();
    let save_all = |generator: &mut Model, evolver: &mut Evolver, negative: &mut Option<Model>| {
        generator.save(&out)?;
        if let Some(path) = &disc_path {
            evolver.discriminator.save(path)?;
        }
        if let Some(neg) = negative.as_mut() {
            neg.save(&neg_path)?;
        }
        Ok::<(), String>(())
    };
    let mut records = Vec::new();
    while generations == 0 || records.len() < generations {
        let record = evolver.run_generation(&mut generator, negative.as_mut())?;
        if let Some(manager) = manager {
            if evolver.generation % schedule.every == 0 {
                manager.save(&mut generator, evolver.generation as i64, "gen", Some(record.clone()))?;
            }
        }
        records.push(record);
        if generations == 0 {
            save_all(&mut generator, &mut evolver, &mut negative)?;
        }
    }
    save_all(&mut generator, &mut evolver, &mut negative)?;
    let negative_doc = match negative.as_mut() {
        Some(neg) => Json::obj([
            ("path", Json::str(neg_path.clone())),
            ("saved", saved_doc(&neg_path)),
            ("reasons", reason_rows(neg)),
            ("stats", negative_stats(neg)),
        ]),
        None => Json::Null,
    };
    let discriminator_doc = match &disc_path {
        Some(path) => Json::obj([
            ("path", Json::str(path.clone())),
            ("saved", saved_doc(path)),
            ("stats", stats(&evolver.discriminator)),
        ]),
        None => Json::Null,
    };
    let checkpoint_dir = match &schedule.manager {
        Some(manager) => Json::str(manager.directory()),
        None => Json::Null,
    };
    ctx.emit(Json::obj([
        ("negative", negative_doc),
        ("model", origin),
        ("out", Json::str(out.clone())),
        ("corpus_texts", Json::Int(evolver.corpus.len() as i64)),
        ("config", evolver.config.to_json()),
        ("generations", Json::Arr(records)),
        ("generation", Json::Int(evolver.generation as i64)),
        ("interrupted", Json::Bool(false)),
        ("saved", saved_doc(&out)),
        ("discriminator", discriminator_doc),
        ("checkpoint_dir", checkpoint_dir),
        ("stats", stats(&generator)),
    ]));
    Ok(())
}

// -- the server ----------------------------------------------------------------------------------

/// What the server keeps between evolve runs: the discriminator (reused by a
/// run of the same model with the same seed, as Python's is) and the record of
/// every generation run so far.  Both belong to the model they were run on:
/// when another model takes its place, they are forgotten.
#[derive(Default)]
pub struct State {
    kept: Mutex<Kept>,
}

#[derive(Default)]
struct Kept {
    discriminator: Option<Model>,
    seed: Option<i64>,
    /// Which model the discriminator and history belong to.
    owner: Option<String>,
    history: Vec<Json>,
}

impl State {
    /// Reads the `serve` command's flags for this area (it has none).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    fn kept(&self) -> std::sync::MutexGuard<'_, Kept> {
        self.kept.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Drops the discriminator and the history: the model they belonged to is gone.
    pub(crate) fn forget(&self) {
        *self.kept() = Kept::default();
    }
}

/// Who a model is, for as long as it is the running one: when it was made,
/// from which seed, reading which encoding.  A load, a reset or a restore
/// makes another one.
fn owner_of(model: &Model) -> String {
    format!("{}|{}|{}", model.meta.created, model.meta.seed, model.encoding())
}

/// The negative network the server holds, loaded (or made) the first time an
/// evolve run blames with it: from beside the model file, else empty, on the
/// running model's encoding.
fn ensure_negative(svc: &Service) -> Result<(), ApiError> {
    // the encoding first: the model lock is never taken under the negative one here
    let encoding = svc.active_encoding();
    let mut slot = svc.negative.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_some() {
        return Ok(());
    }
    let path = crate::cli::negative_path(&svc.model_path);
    let mut model = if !svc.model_path.is_empty() && Path::new(&path).is_file() {
        let found = Model::load(&path)?;
        if !found.is_negative() {
            return Err(ApiError::bad_request(format!(
                "{path} holds a {} model, not a negative one",
                found.kind()
            )));
        }
        crate::log_info!(LOG, "negative model loaded from {path}");
        found
    } else {
        Model::new_negative(
            svc.seed,
            &NegativeOptions {
                encoding,
                ..Default::default()
            },
        )?
    };
    model.workers = svc.workers;
    model.g.workers = svc.workers;
    *slot = Some(model);
    Ok(())
}

/// The loop's settings from a request body, over Python's defaults (the seed
/// is the loop's own, 0 unless given).
fn config_from_request(r: &Request) -> Result<EvolveConfig, ApiError> {
    let d = EvolveConfig::default();
    let config = EvolveConfig {
        samples: r.usize("samples", d.samples)?,
        real_per_generation: r.usize("real_per_generation", d.real_per_generation)?,
        max_length: r.usize("max_length", d.max_length)?,
        temperature: r.number("temperature", d.temperature)?,
        neg_epochs: r.usize("neg_epochs", d.neg_epochs)?,
        pos_epochs: r.usize("pos_epochs", d.pos_epochs)?,
        neg_lr: r.number("neg_lr", d.neg_lr)?,
        pos_lr: r.number("pos_lr", d.pos_lr)?,
        disc_neg_epochs: r.usize("disc_neg_epochs", d.disc_neg_epochs)?,
        disc_pos_epochs: r.usize("disc_pos_epochs", d.disc_pos_epochs)?,
        batch_size: r.usize("batch_size", d.batch_size)?,
        checkpoint_every: r.usize("checkpoint_every", d.checkpoint_every)?,
        seed: r.body.at("seed").as_i64().unwrap_or(d.seed),
        blatant_mode: r.text("blatant_mode", &d.blatant_mode),
        blatant_margin: r.number("blatant_margin", d.blatant_margin)?,
        blatant_boost: r.number("blatant_boost", d.blatant_boost)?,
        strength: r.number("strength", d.strength)?,
    };
    Ok(config)
}

/// `POST /api/evolve/start`: the loop as a background job (202), polled on
/// `/api/job`; `generations` 0 (or absent) runs until stopped.
fn start(svc: &Arc<Service>, r: &Request) -> Answer {
    let corpus = svc.texts_of(r, "corpus", "corpus_text", "corpus_files")?;
    if corpus.is_empty() {
        return Err(ApiError::bad_request(
            "missing field 'corpus' (list of strings), 'corpus_text' (string, one text per line) \
             or 'corpus_files' (list of upload names)",
        ));
    }
    let config = config_from_request(r)?;
    let generations = r.usize("generations", 0)?;
    let blame = r.flag("blame", false);
    let every = config.checkpoint_every;
    if every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    svc.ensure_idle()?;
    if blame {
        ensure_negative(svc)?;
    }
    let (owner, evolver) = svc.with_model(|m| -> Result<(String, Evolver), String> {
        let owner = owner_of(m);
        let mut evolver = Evolver::new(m, &corpus, None, config.clone())?;
        // the kept discriminator takes the fresh one's place only once nothing
        // can fail any more, so a refused run never loses it
        let mut kept = svc.evolve.kept();
        if kept.owner.as_deref() != Some(owner.as_str()) {
            *kept = Kept::default();
        }
        if kept.seed == Some(config.seed) {
            if let Some(discriminator) = kept.discriminator.take() {
                evolver.discriminator = discriminator;
            }
        }
        Ok((owner, evolver))
    })?;
    let corpus_len = evolver.corpus.len();
    svc.start_job("evolve");
    let worker = Arc::clone(svc);
    let seed = config.seed;
    std::thread::spawn(move || {
        let mut evolver = evolver;
        let stop = || worker.stopping();
        let outcome = (|| -> Result<Vec<Json>, String> {
            let manager = crate::checkpoint::manager(&worker).filter(|_| every > 0);
            let mut records = Vec::new();
            while generations == 0 || records.len() < generations {
                if worker.stopping() {
                    break;
                }
                // the model lock once per generation, so the frontend's reads get in between
                let pending = worker.with_model(|m| evolver.step(m))?;
                let record = if blame {
                    let mut negative = worker.negative.lock().unwrap_or_else(|e| e.into_inner());
                    evolver.finish(pending, negative.as_mut(), &stop)?
                } else {
                    evolver.finish(pending, None, &stop)?
                };
                worker.job_progress(record.clone());
                worker.evolve.kept().history.push(record.clone());
                if let Some(manager) = manager.filter(|_| evolver.generation % every == 0) {
                    let step = evolver.generation as i64;
                    worker.with_model(|m| manager.save(m, step, "gen", Some(record.clone())))?;
                }
                records.push(record);
            }
            Ok(records)
        })();
        {
            let mut kept = worker.evolve.kept();
            kept.discriminator = Some(evolver.discriminator);
            kept.seed = Some(seed);
            kept.owner = Some(owner);
        }
        if outcome.is_ok() {
            worker.autosave();
        }
        worker.finish_job(outcome);
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("config", config.to_json()),
        ("generations", Json::Int(generations as i64)),
        ("corpus", Json::Int(corpus_len as i64)),
    ])))
}

/// `POST /api/evolve/stop`: asks the evolve job to stop after the generation
/// in flight; 409 while another kind of job runs, 404 when none was started.
fn stop(svc: &Arc<Service>, _r: &Request) -> Answer {
    let job = svc.job_json();
    let kind = job.at("type").as_str().unwrap_or("");
    if !job.is_null() && kind != "evolve" && job.at("state").as_str() == Some("running") {
        return Err(ApiError::conflict(format!(
            "the running job ({}) is a {kind} job; use POST /api/job/stop",
            job.at("id").as_str().unwrap_or("")
        )));
    }
    if job.is_null() || kind != "evolve" {
        return Err(ApiError::not_found("no evolve job has been started"));
    }
    svc.request_stop();
    Ok(svc.job_json())
}

/// `GET /api/evolve/history`: the generation records of every evolve run of
/// the running model.
fn history(svc: &Arc<Service>, _r: &Request) -> Answer {
    let owner = svc.with_model(|m| owner_of(m));
    let kept = svc.evolve.kept();
    let history = if kept.owner.as_deref() == Some(owner.as_str()) {
        kept.history.clone()
    } else {
        Vec::new()
    };
    Ok(Json::obj([("history", Json::Arr(history))]))
}

/// This area's routes:
/// `POST /api/evolve/start`
/// `POST /api/evolve/stop`
/// `GET /api/evolve/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/evolve/start", start);
    server.route("POST", "/api/evolve/stop", stop);
    server.route("GET", "/api/evolve/history", history);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TrainOptions;

    const CORPUS: [&str; 5] = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a bird flew over the hill",
        "the cat chased the mouse",
        "rain fell on the quiet town",
    ];

    fn corpus() -> Vec<String> {
        CORPUS.iter().map(|s| s.to_string()).collect()
    }

    fn generator() -> Model {
        let mut m = Model::new(3, GraphOptions::default()).unwrap();
        m.train(
            &corpus(),
            &TrainOptions {
                epochs: 3,
                ..Default::default()
            },
        )
        .unwrap();
        m
    }

    fn small(mode: &str) -> EvolveConfig {
        EvolveConfig {
            samples: 4,
            real_per_generation: 3,
            max_length: 24,
            blatant_mode: mode.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn the_draws_are_pythons() {
        // random.Random(0).sample(range(10), 3) == [6, 9, 0];
        // random.Random(1).sample(range(100), 7) == [17, 72, 97, 8, 32, 15, 63]
        assert_eq!(sample_indices(&mut Mt19937::new(0), 10, 3), vec![6, 9, 0]);
        assert_eq!(
            sample_indices(&mut Mt19937::new(1), 100, 7),
            vec![17, 72, 97, 8, 32, 15, 63]
        );
        assert_eq!(sample_indices(&mut Mt19937::new(5), 3, 3), vec![2, 1, 0]);
        assert_eq!(
            sample_indices(&mut Mt19937::new(7), 30, 8),
            vec![10, 4, 12, 20, 1, 2, 17, 3]
        );
        // a population larger than the pool CPython would copy: the set of the chosen
        assert_eq!(sample_indices(&mut Mt19937::new(3), 1000, 4), vec![243, 606, 557, 133]);
        assert_eq!(
            sample_indices(&mut Mt19937::new(9), 500, 12),
            vec![237, 313, 191, 136, 70, 95, 443, 346, 3, 173, 257, 459]
        );
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(fmean(&[0.1, 0.2, 0.3]), Some(0.6 / 3.0));
        assert_eq!(fmean(&[]), None);
        assert_eq!(round3(1.23456), 1.235);
    }

    #[test]
    fn the_defaults_validate_and_the_nonsense_does_not() {
        let enc = Encoding::default();
        assert!(EvolveConfig::default().validate(&enc).is_ok());
        for broken in [
            EvolveConfig {
                samples: 0,
                ..Default::default()
            },
            EvolveConfig {
                real_per_generation: 0,
                ..Default::default()
            },
            EvolveConfig {
                max_length: 2,
                ..Default::default()
            },
            EvolveConfig {
                temperature: -1.0,
                ..Default::default()
            },
            EvolveConfig {
                blatant_mode: "sideways".to_string(),
                ..Default::default()
            },
            EvolveConfig {
                blatant_boost: 0.5,
                ..Default::default()
            },
            EvolveConfig {
                blatant_margin: -1.0,
                ..Default::default()
            },
        ] {
            assert!(broken.validate(&enc).is_err(), "{broken:?}");
        }
        let keys: Vec<String> = match EvolveConfig::default().to_json() {
            Json::Obj(pairs) => pairs.into_iter().map(|(k, _)| k).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys.first().map(String::as_str), Some("samples"));
        assert!(keys.contains(&"blatant_boost".to_string()));
    }

    #[test]
    fn a_loop_needs_a_count_model_and_a_corpus() {
        let g = generator();
        assert!(Evolver::new(&g, &["ab".to_string()], None, small("none")).is_err());
        let neg = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        assert!(Evolver::new(&neg, &corpus(), None, small("none")).is_err());
        let loop_ = Evolver::new(&g, &corpus(), None, small("none")).unwrap();
        assert_eq!(loop_.discriminator.meta.seed, 1, "seeded one past the loop's seed");
        assert_eq!(loop_.discriminator.encoding(), g.encoding());
    }

    #[test]
    fn one_generation_reports_what_python_reports() {
        let mut g = generator();
        let mut loop_ = Evolver::new(&g, &corpus(), None, small("none")).unwrap();
        let record = loop_.run_generation(&mut g, None).unwrap();
        let keys: Vec<String> = match &record {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.clone()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            keys,
            [
                "generation",
                "fake_score_mean",
                "real_score_mean",
                "gap",
                "gen_loss",
                "nodes",
                "edges",
                "compression_ratio",
                "sample",
                "fakes",
                "worst",
                "failures",
                "blatant",
                "flipped",
                "boost_mean",
                "boost_max",
                "twonrl",
                "mode",
                "seconds"
            ]
        );
        assert_eq!(record.at("generation").as_i64(), Some(1));
        assert!(record.at("fakes").as_i64().unwrap() > 0);
        assert_eq!(record.at("twonrl").as_bool(), Some(true));
        assert!(loop_.discriminator.g.num_nodes() > 0, "the critic learned something");
        assert_eq!(loop_.history.len(), 1);
    }

    #[test]
    fn the_same_seed_is_the_same_run() {
        let run = || {
            let mut g = generator();
            let mut loop_ = Evolver::new(&g, &corpus(), None, small("activation")).unwrap();
            let records = loop_.run(&mut g, None, 3, None, &|| false, &mut |_| {}).unwrap();
            records
                .into_iter()
                .map(|r| match r {
                    Json::Obj(pairs) => Json::Obj(pairs.into_iter().filter(|(k, _)| k != "seconds").collect()),
                    other => other,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn fail_invert_punishes_the_worst_hardest() {
        let mut g = generator();
        let mut loop_ = Evolver::new(&g, &corpus(), None, small("fail_invert")).unwrap();
        let record = loop_.run_generation(&mut g, None).unwrap();
        if record.at("failures").as_i64().unwrap() > 0 {
            assert!(record.at("boost_max").as_f64().unwrap() >= 1.0);
            assert_eq!(record.at("twonrl").as_bool(), Some(true));
            assert!(
                g.history.iter().any(|h| h.extra_value("weight").is_some()),
                "the weight is in the history"
            );
        } else {
            assert_eq!(record.at("worst").as_i64(), Some(0));
        }
    }

    #[test]
    fn the_local_modes_move_only_the_failed_paths() {
        for mode in ["activation", "state"] {
            let mut g = generator();
            let mut loop_ = Evolver::new(&g, &corpus(), None, small(mode)).unwrap();
            let record = loop_.run_generation(&mut g, None).unwrap();
            assert_eq!(record.at("mode").as_str(), Some(mode));
            if record.at("failures").as_i64().unwrap() > 0 {
                assert!(
                    record.at("flipped").as_i64().unwrap() > 0,
                    "{mode}: failures move edges"
                );
            }
        }
    }

    #[test]
    fn weighted_2nrl_groups_by_the_mark() {
        let texts: Vec<String> = ["aaa bbb", "ccc ddd", "eee fff"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let groups = weight_groups(&texts, &[1.0, 2.0004, 0.0], "w").unwrap();
        assert_eq!(
            groups,
            vec![(2.0, vec!["ccc ddd".to_string()]), (1.0, vec!["aaa bbb".to_string()])]
        );
        assert!(weight_groups(&texts, &[1.0], "w").is_err());
        assert!(weight_groups(&texts, &[1.0, -1.0, 1.0], "w").is_err());
        let mut g = generator();
        let (neg, pos) = two_nrl_weighted(&mut g, &texts, &[2.0, 1.0, 1.0], &corpus(), 1, 1, 1.0).unwrap();
        assert_eq!(neg.len(), 2);
        assert_eq!(neg[0].extra_value("weight").and_then(|w| w.as_f64()), Some(2.0));
        assert_eq!(neg[0].reward, -2.0, "the heavier failure is punished harder");
        assert_eq!(pos.len(), 1);
        assert_eq!(g.meta.twonrl_runs.value, 1);
    }

    #[test]
    fn a_failed_path_loses_reward() {
        let mut g = generator();
        let texts = vec!["the cat sat on the mat".to_string(), "never seen before".to_string()];
        let before = g.g.total_reward().1;
        let outcome = invert_paths(&mut g, &texts, &[1.0, 0.5], 2.0).unwrap();
        assert_eq!(outcome.texts, 2);
        assert!(outcome.flipped > 0);
        assert!(g.g.total_reward().1 < before);
        assert_eq!(outcome.amount_mean, 0.75);
        assert!(invert_paths(&mut g, &texts, &[1.0], 2.0).is_err());
        assert!(invert_paths(&mut g, &texts, &[1.5, 0.0], 2.0).is_err());
    }

    #[test]
    fn the_critic_teaches_the_negative_network() {
        let mut g = generator();
        let mut neg = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let config = EvolveConfig {
            blatant_margin: 0.01,
            ..small("none")
        };
        let mut loop_ = Evolver::new(&g, &corpus(), None, config).unwrap();
        let record = loop_.run_generation(&mut g, Some(&mut neg)).unwrap();
        assert!(record.get("negative_blamed").is_some());
        let blamed = record.at("negative_blamed").as_i64().unwrap();
        if blamed > 0 {
            assert!(neg.g.num_edges() > 0);
            assert!(neg
                .reasons()
                .iter()
                .any(|r| r.reason == "blatant" || r.reason == "discriminator"));
        }
    }

    #[test]
    fn checkpoints_every_n_generations_and_a_stop() {
        let dir = std::env::temp_dir().join(format!("radixnet-gan-ckpt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let manager = Checkpoints::new(&dir.to_string_lossy(), 5).unwrap();
        let mut g = generator();
        let config = EvolveConfig {
            checkpoint_every: 2,
            ..small("none")
        };
        let mut loop_ = Evolver::new(&g, &corpus(), None, config).unwrap();
        let mut seen = 0;
        let records = loop_
            .run(&mut g, None, 4, Some(&manager), &|| false, &mut |_| seen += 1)
            .unwrap();
        assert_eq!((records.len(), seen), (4, 4));
        let names: Vec<String> = manager.list().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["ckpt-gen-000002.json.gz", "ckpt-gen-000004.json.gz"]);
        assert_eq!(manager.latest().unwrap().metrics.at("generation").as_i64(), Some(4));
        let stopped = loop_.run(&mut g, None, 0, None, &|| true, &mut |_| {}).unwrap();
        assert!(stopped.is_empty(), "a stop before the first generation runs none");
    }

    #[test]
    fn the_routes_run_a_job_and_keep_its_history() {
        let svc = Arc::new(Service::new(generator(), String::new(), 0, 1));
        let refused = start(
            &svc,
            &Request::json("POST", "/", Json::obj([("generations", Json::Int(1))])),
        );
        assert_eq!(refused.unwrap_err().status, 400, "no corpus");
        let bad = start(
            &svc,
            &Request::json(
                "POST",
                "/",
                Json::obj([
                    ("corpus", Json::strs(corpus())),
                    ("blatant_mode", Json::str("sideways")),
                ]),
            ),
        );
        assert_eq!(bad.unwrap_err().status, 400);
        let none = stop(&svc, &Request::json("POST", "/", Json::Null));
        assert_eq!(none.unwrap_err().status, 404);
        let body = Json::obj([
            ("corpus", Json::strs(corpus())),
            ("generations", Json::Int(2)),
            ("samples", Json::Int(3)),
            ("real_per_generation", Json::Int(2)),
            ("max_length", Json::Int(20)),
            ("blame", Json::Bool(true)),
        ]);
        let answer = start(&svc, &Request::json("POST", "/", body)).unwrap();
        assert_eq!(answer.at("__status").as_i64(), Some(202));
        for _ in 0..2000 {
            if svc.job_json().at("state").as_str() != Some("running") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let job = svc.job_json();
        assert_eq!(job.at("state").as_str(), Some("finished"), "{}", job.render(0));
        assert_eq!(job.at("history").as_array().len(), 2);
        let kept = history(&svc, &Request::json("GET", "/", Json::Null)).unwrap();
        assert_eq!(kept.at("history").as_array().len(), 2);
        assert!(kept.at("history").as_array()[0].get("negative_blamed").is_some());
        assert!(
            svc.negative.lock().unwrap().is_some(),
            "blame loaded the negative network"
        );
        let stopped = stop(&svc, &Request::json("POST", "/", Json::Null)).unwrap();
        assert_eq!(stopped.at("type").as_str(), Some("evolve"));
        // the critic is kept for the next run of this model and seed, and a
        // refused run does not lose it
        let learned = |svc: &Service| svc.evolve.kept().discriminator.as_ref().map(|d| d.g.num_edges());
        let edges = learned(&svc).expect("the discriminator is kept");
        assert!(edges > 0);
        let refused = start(
            &svc,
            &Request::json("POST", "/", Json::obj([("corpus", Json::strs(["ab".to_string()]))])),
        );
        assert_eq!(refused.unwrap_err().status, 400);
        assert_eq!(learned(&svc), Some(edges));
        // another model takes the running one's place: its history is not this one's
        svc.with_model(|m| *m = Model::new(9, GraphOptions::default()).unwrap());
        let other = history(&svc, &Request::json("GET", "/", Json::Null)).unwrap();
        assert!(other.at("history").as_array().is_empty());
    }
}
