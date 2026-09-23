//! The negative network feeding itself: an LLM reviewer on a loop - the
//! Negative tab, automatic (`radixnet/critic.py`, `go/radixnet/critic.go`,
//! `go/server/critic.go`).
//!
//! Every other tutor hands the negative network its failures as a side effect
//! of doing something else - the English tutor marks a sentence, the sandbox
//! rejects a program, the recall tutor finds a misremembered waveform.  The
//! Negative tab itself was the one place where somebody had to *type* a
//! failure in.  This loop closes that gap.  One round is:
//!
//! 1. the **positive model** writes `count` texts of its own (a stochastic
//!    walk, optionally continuing a prefix);
//! 2. an **LLM reviewer** - a local Ollama model by default, ChatGPT when the
//!    provider says so - marks each one out of 10 and writes a one-sentence
//!    critique ([`crate::review::review_texts`]);
//! 3. the failures **blame** the negative network, the critique picking the
//!    reason and the mark the severity, and the passes **clear** blame off the
//!    fragments they share with known failures
//!    ([`crate::review::teach_reviews`]).
//!
//! Then it goes round again.  Nothing is invented: every failure still comes
//! from something outside the network that looked at an output and said it
//! was wrong, and why.
//!
//! # The positive model is only read from
//!
//! Nothing here trains, rewards or inverts it, so the loop can run beside
//! anything else teaching it.  Sampling does move the model's generator on, so
//! a seeded run advances the seed by the round rather than reusing one, which
//! keeps a run reproducible and its rounds different.
//!
//! # Where the two networks live
//!
//! [`Networks`] is the one thing the CLI and the server do differently: the
//! command owns both models for the length of the run, while the server keeps
//! each behind its own lock and takes it for a step at a time - the positive
//! model to write, the negative one to learn - so the frontend's reads get in
//! between, and nothing is locked while the reviewer thinks, which is most of
//! a round.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::blame::{TeachOptions, TeachReport};
use crate::cli::{negative_path, negative_stats, Ctx};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::llm::fields::Fields;
use crate::llm::{format_g, new_client, normalise_provider, seconds, LlmClient, CHATGPT, DEFAULT_PROVIDER, PROVIDERS};
use crate::model::Model;
use crate::negative::NegativeOptions;
use crate::review::{review_texts, sample_texts, summarise_reviews, teach_reviews, Review, DEFAULT_BATCH};
use crate::service::Service;

/// What this module's own lines are filed under.
const LOG: &str = "critic";

/// How the reviewer is run, and what it is told.
#[derive(Clone, Debug, PartialEq)]
pub struct CriticConfig {
    /// Rounds to run; 0 keeps going until something stops it.
    pub rounds: usize,
    /// Texts the model writes per round, for the reviewer to mark.
    pub count: usize,
    /// Continue this instead of writing from scratch.
    pub prefix: String,
    pub max_length: usize,
    pub temperature: f64,
    /// The pass mark out of 10: below it a text is a failure and is blamed.
    pub threshold: f64,
    /// What the reviewer is told the texts are meant to be (its yardstick).
    pub context: String,
    /// `ollama` or `chatgpt`.
    pub provider: String,
    /// The reviewer's model; `""` is the client's own default.
    pub reviewer_model: String,
    /// Let the texts the reviewer passed take blame off what they share with
    /// known failures.
    pub clear_passes: bool,
    /// Blame epochs per round.
    pub epochs: usize,
    /// The seed of the first round's sampling; later rounds advance it.
    pub seed: Option<i64>,
}

impl Default for CriticConfig {
    /// The Python defaults: three rounds of eight texts of sixty characters,
    /// a pass at 6, reviewed by the local provider.
    fn default() -> CriticConfig {
        CriticConfig {
            rounds: 3,
            count: 8,
            prefix: String::new(),
            max_length: 60,
            temperature: 1.0,
            threshold: 6.0,
            context: String::new(),
            provider: DEFAULT_PROVIDER.to_string(),
            reviewer_model: String::new(),
            clear_passes: true,
            epochs: 1,
            seed: None,
        }
    }
}

impl CriticConfig {
    /// Refuses settings the loop cannot run with.
    pub fn validate(&self) -> Result<(), String> {
        if self.count < 1 {
            return Err("count must be >= 1".to_string());
        }
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        if !(0.0..=10.0).contains(&self.threshold) {
            return Err("threshold must lie in [0, 10]".to_string());
        }
        if normalise_provider(&self.provider).is_err() {
            return Err(format!("provider must be one of: {}", PROVIDERS.join(", ")));
        }
        Ok(())
    }

    /// The settings as a document (`dataclasses.asdict` of Python's config).
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("rounds", Json::Int(self.rounds as i64)),
            ("count", Json::Int(self.count as i64)),
            ("prefix", Json::str(self.prefix.clone())),
            ("max_length", Json::Int(self.max_length as i64)),
            ("temperature", Json::Num(self.temperature)),
            ("threshold", Json::Num(self.threshold)),
            ("context", Json::str(self.context.clone())),
            ("provider", Json::str(self.provider.clone())),
            ("reviewer_model", Json::str(self.reviewer_model.clone())),
            ("clear_passes", Json::Bool(self.clear_passes)),
            ("epochs", Json::Int(self.epochs as i64)),
            ("seed", self.seed.map(Json::Int).unwrap_or(Json::Null)),
        ])
    }

    /// The settings of a `POST /api/negative/auto` body, defaults filled in;
    /// every bad one is a 400 before a job starts.
    pub fn from_fields(f: &Fields) -> Result<CriticConfig, ApiError> {
        let d = CriticConfig::default();
        let raw = match f.text("provider")?.filter(|p| !p.is_empty()) {
            Some(p) => Some(p),
            None => f.text("reviewer_provider")?.filter(|p| !p.is_empty()),
        };
        let provider = match raw {
            Some(raw) => normalise_provider(&raw)
                .map_err(|_| {
                    ApiError::bad_request(format!(
                        "'provider' must be one of {} (got {})",
                        PROVIDERS.join(", "),
                        crate::negative::python_repr(&raw)
                    ))
                })?
                .to_string(),
            None => d.provider,
        };
        let reviewer_model = match f.text("reviewer_model")?.filter(|m| !m.is_empty()) {
            Some(m) => m,
            None => f.text("model")?.unwrap_or_default(),
        };
        let config = CriticConfig {
            rounds: f.count_or("rounds", d.rounds, 0)?,
            count: f.count_or("count", d.count, 1)?,
            prefix: f.text_or("prefix", &d.prefix)?,
            max_length: f.count_or("max_length", d.max_length, 0)?,
            temperature: f.number_or("temperature", d.temperature, Some(0.0))?,
            threshold: f.number_or("threshold", d.threshold, Some(0.0))?,
            context: f.text_or("context", &d.context)?,
            provider,
            reviewer_model,
            clear_passes: f.flag("clear_passes", d.clear_passes)?,
            epochs: f.count_or("epochs", d.epochs, 0)?,
            seed: f.integer("seed", None)?,
        };
        config.validate().map_err(ApiError::bad_request)?;
        Ok(config)
    }

    /// The settings of `radixnet negative auto`; the seed is the global
    /// `--seed` (0 when it is not given, as Python's `effective_seed`).
    pub fn from_args(ctx: &Ctx) -> Result<CriticConfig, String> {
        let args = &ctx.args;
        let d = CriticConfig::default();
        let whole = |name: &str, default: usize| -> Result<usize, String> {
            let value = args.int(name, default as i64)?;
            if value < 0 {
                return Err(format!("--{name} must be >= 0, got {value}"));
            }
            Ok(value as usize)
        };
        let provider = args.str("provider", DEFAULT_PROVIDER);
        let config = CriticConfig {
            rounds: whole("rounds", d.rounds)?,
            count: whole("count", d.count)?,
            prefix: args.str("prefix", ""),
            max_length: whole("max-length", d.max_length)?,
            temperature: crate::llm::nonneg_flag(ctx, "temperature", d.temperature)?,
            threshold: crate::llm::nonneg_flag(ctx, "threshold", d.threshold)?,
            context: args.str("context", ""),
            provider: normalise_provider(&provider)?.to_string(),
            reviewer_model: args.str("reviewer-model", ""),
            clear_passes: !args.on("no-clear"),
            epochs: whole("epochs", d.epochs)?,
            seed: Some(ctx.seed),
        };
        config.validate()?;
        Ok(config)
    }
}

/// Where the loop's two networks live: owned by a command for the length of
/// the run, or behind the server's locks, taken a step at a time.
pub trait Networks {
    /// The positive model writes the round's texts.
    fn sample(
        &mut self,
        count: usize,
        prefix: &str,
        max_length: usize,
        temperature: f64,
        seed: Option<i64>,
    ) -> Result<Vec<String>, String>;

    /// The negative network learns from the verdicts; the report, and its
    /// statistics afterwards.
    fn teach(
        &mut self,
        reviews: &[Review],
        threshold: f64,
        clear_passes: bool,
        epochs: usize,
    ) -> Result<(TeachReport, Json), String>;
}

/// The two networks a command holds for the whole run.
pub struct Owned<'m> {
    pub model: &'m mut Model,
    pub negative: &'m mut Model,
    /// Saves the negative network here after every round - what a run with
    /// no end is stopped by killing, and it must not lose its rounds.
    pub save_each_round: Option<String>,
}

impl Networks for Owned<'_> {
    fn sample(
        &mut self,
        count: usize,
        prefix: &str,
        max_length: usize,
        temperature: f64,
        seed: Option<i64>,
    ) -> Result<Vec<String>, String> {
        sample_texts(self.model, count, prefix, max_length, temperature, seed)
    }

    fn teach(
        &mut self,
        reviews: &[Review],
        threshold: f64,
        clear_passes: bool,
        epochs: usize,
    ) -> Result<(TeachReport, Json), String> {
        let o = TeachOptions {
            epochs,
            ..Default::default()
        };
        let report = teach_reviews(self.negative, reviews, threshold, clear_passes, "critic", &o)?;
        if let Some(path) = &self.save_each_round {
            self.negative.save(path)?;
        }
        Ok((report, negative_stats(self.negative)))
    }
}

/// The two networks behind the server's locks: each is held for one step.
struct Shared<'s> {
    svc: &'s Service,
}

impl Networks for Shared<'_> {
    fn sample(
        &mut self,
        count: usize,
        prefix: &str,
        max_length: usize,
        temperature: f64,
        seed: Option<i64>,
    ) -> Result<Vec<String>, String> {
        self.svc
            .with_model(|m| sample_texts(m, count, prefix, max_length, temperature, seed))
    }

    fn teach(
        &mut self,
        reviews: &[Review],
        threshold: f64,
        clear_passes: bool,
        epochs: usize,
    ) -> Result<(TeachReport, Json), String> {
        let o = TeachOptions {
            epochs,
            ..Default::default()
        };
        with_negative(self.svc, |negative| {
            let report = teach_reviews(negative, reviews, threshold, clear_passes, "critic", &o)?;
            Ok((report, negative_stats(negative)))
        })
        .map_err(|err| err.message)?
    }
}

/// The reviewer on a loop, keeping the negative network fed.
pub struct Critic<'a> {
    client: &'a dyn LlmClient,
    pub config: CriticConfig,
    /// The rounds run so far.
    pub round_no: usize,
    /// Every round record, in order.
    pub history: Vec<Json>,
}

impl<'a> Critic<'a> {
    /// The loop, once its settings are known to be sound.
    pub fn new(client: &'a dyn LlmClient, config: CriticConfig) -> Result<Critic<'a>, String> {
        config.validate()?;
        Ok(Critic {
            client,
            config,
            round_no: 0,
            history: Vec::new(),
        })
    }

    /// This round's sampling seed: the configured one advanced by the round.
    fn seed(&self) -> Option<i64> {
        self.config.seed.map(|seed| seed + self.round_no as i64)
    }

    /// Writes, reviews and blames once, and returns the round's record.
    pub fn run_round(&mut self, nets: &mut dyn Networks) -> Result<Json, String> {
        self.round_no += 1;
        let started = Instant::now();
        let seed = self.seed();
        let cfg = &self.config;
        // the model writes first, under whatever lock guards it
        let samples = nets.sample(cfg.count, &cfg.prefix, cfg.max_length, cfg.temperature, seed)?;
        // only the reviewer's thinking happens with nothing held
        let reviews = review_texts(
            self.client,
            &samples,
            &cfg.context,
            &cfg.reviewer_model,
            cfg.threshold,
            DEFAULT_BATCH,
        )?;
        let reviewer = if cfg.reviewer_model.is_empty() {
            self.client.model()
        } else {
            cfg.reviewer_model.as_str()
        };
        let review = summarise_reviews("model", reviewer, cfg.threshold, samples, reviews);
        let (taught, stats) = nets.teach(&review.reviews, cfg.threshold, cfg.clear_passes, cfg.epochs)?;
        let taught_doc = taught.to_json();
        let record = Json::obj([
            ("kind", Json::str("round")),
            ("round", Json::Int(self.round_no as i64)),
            ("reviewer", Json::str(review.model.clone())),
            ("threshold", Json::Num(cfg.threshold)),
            ("texts", Json::Int(review.texts.len() as i64)),
            (
                "reviews",
                Json::Arr(review.reviews.iter().map(Review::to_json).collect()),
            ),
            ("mean_rating", review.mean_rating.map(Json::Num).unwrap_or(Json::Null)),
            ("pass_rate", review.pass_rate.map(Json::Num).unwrap_or(Json::Null)),
            ("passed", Json::Int(review.good.len() as i64)),
            ("failed", Json::Int(review.bad.len() as i64)),
            ("blamed", Json::Int(taught.blamed as i64)),
            ("cleared", Json::Int(taught.cleared as i64)),
            ("unmatched", Json::Int(taught.unmatched as i64)),
            ("edges", Json::Int(taught.edges as i64)),
            ("reasons", taught_doc.at("reasons").clone()),
            ("severity_mean", Json::Num(taught.severity_mean)),
            ("stats", stats),
            ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ]);
        crate::log_info!(
            LOG,
            "round {}: {}/{} failed, blamed {} over {} edge(s), cleared {}",
            self.round_no,
            review.bad.len(),
            review.texts.len(),
            taught.blamed,
            taught.edges,
            taught.cleared
        );
        self.history.push(record.clone());
        Ok(record)
    }

    /// Runs the configured rounds - or, with 0, until `stop` says so - asking
    /// `stop` before every round, so stopping finishes the round it is in
    /// rather than abandoning a half-taught one.  The records are the rounds,
    /// then one report card; `progress` sees each as it happens.
    pub fn run(
        &mut self,
        nets: &mut dyn Networks,
        stop: &dyn Fn() -> bool,
        progress: &mut dyn FnMut(&Json),
    ) -> Result<Vec<Json>, String> {
        let limit = self.config.rounds;
        let mut records = Vec::new();
        while limit == 0 || records.len() < limit {
            if stop() {
                break;
            }
            let record = self.run_round(nets)?;
            progress(&record);
            records.push(record);
        }
        let card = report_card(&records);
        progress(&card);
        records.push(card);
        Ok(records)
    }
}

/// What a run of rounds came to: how much was reviewed, blamed and cleared,
/// and why.  `mean_rating` is over the rounds that produced one, and `trend`
/// is the last round's mean mark minus the first's - positive when the
/// reviewer is marking the output better than it did at the start.
pub fn report_card(records: &[Json]) -> Json {
    let rounds: Vec<&Json> = records
        .iter()
        .filter(|r| r.at("kind").as_str() == Some("round"))
        .collect();
    let numbers = |key: &str| -> Vec<f64> {
        rounds
            .iter()
            .filter_map(|r| match r.at(key) {
                Json::Int(n) => Some(*n as f64),
                Json::Num(x) => Some(*x),
                _ => None,
            })
            .collect()
    };
    let ratings = numbers("mean_rating");
    let rates = numbers("pass_rate");
    let total = |key: &str| -> i64 { rounds.iter().map(|r| r.at(key).as_i64().unwrap_or(0)).sum() };
    let mut reasons: Vec<(String, i64)> = Vec::new();
    for record in &rounds {
        if let Json::Obj(pairs) = record.at("reasons") {
            for (reason, count) in pairs {
                let count = count.as_i64().unwrap_or(0);
                match reasons.iter_mut().find(|(r, _)| r == reason) {
                    Some(slot) => slot.1 += count,
                    None => reasons.push((reason.clone(), count)),
                }
            }
        }
    }
    reasons.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mean = |values: &[f64]| -> Json {
        if values.is_empty() {
            Json::Null
        } else {
            Json::Num(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    Json::obj([
        ("kind", Json::str("report")),
        ("rounds", Json::Int(rounds.len() as i64)),
        ("reviewed", Json::Int(total("texts"))),
        ("blamed", Json::Int(total("blamed"))),
        ("cleared", Json::Int(total("cleared"))),
        ("edges", Json::Int(total("edges"))),
        ("mean_rating", mean(&ratings)),
        ("pass_rate", mean(&rates)),
        (
            "trend",
            if ratings.len() > 1 {
                Json::Num(ratings[ratings.len() - 1] - ratings[0])
            } else {
                Json::Null
            },
        ),
        (
            "reasons",
            Json::Obj(reasons.into_iter().map(|(r, n)| (r, Json::Int(n))).collect()),
        ),
        (
            "stats",
            rounds.last().map(|r| r.at("stats").clone()).unwrap_or(Json::Null),
        ),
    ])
}

// -- the command line ---------------------------------------------------------------------------

/// `radixnet negative auto`: the Negative tab without anyone typing a failure
/// into it.  `--rounds` (0: until the process is stopped - the negative network
/// is then saved after every round), `--count`, `--prefix`, `--max-length`,
/// `--temperature`, `--threshold`, `--context`, `--provider`,
/// `--reviewer-model`, `--url`, `--timeout`, `--epochs`, `--no-clear`, and
/// `--out` for where the negative network is saved (default: `--negative`).
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let config = CriticConfig::from_args(ctx)?;
    if normalise_provider(&config.provider)? == CHATGPT && !crate::chatgpt::key_configured() {
        return Err(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT review, or use \
             --provider ollama"
                .to_string(),
        );
    }
    let timeout = seconds(crate::llm::nonneg_flag(ctx, "timeout", 0.0)?);
    let client = new_client(&config.provider, &args.str("url", ""), &config.reviewer_model, timeout)?;
    let mut model = ctx.open(true)?;
    let mut negative = ctx.open_negative(false)?;
    let out = args.str("out", &ctx.negative_path());
    crate::log_info!(
        LOG,
        "reviewer {}: {} at {}; {} round(s) x {} text(s) of {} chars, pass at {}/10",
        config.provider,
        client.model(),
        client.url(),
        if config.rounds == 0 {
            "until stopped".to_string()
        } else {
            config.rounds.to_string()
        },
        config.count,
        config.max_length,
        format_g(config.threshold)
    );
    let mut critic = Critic::new(client.as_ref(), config.clone())?;
    let records = {
        let mut nets = Owned {
            model: &mut model,
            negative: &mut negative,
            save_each_round: (config.rounds == 0).then(|| out.clone()),
        };
        critic.run(&mut nets, &|| false, &mut |_| {})?
    };
    negative.save(&out)?;
    let card = records.last().cloned().unwrap_or(Json::Null);
    ctx.emit(Json::obj([
        ("config", config.to_json()),
        ("url", Json::str(client.url())),
        ("reviewer", Json::str(client.model())),
        ("records", Json::Arr(records)),
        ("report", card),
        ("interrupted", Json::Bool(false)),
        (
            "negative",
            Json::obj([
                ("path", Json::str(out.clone())),
                ("saved", crate::llm::saved(&out)),
                ("reasons", crate::review::reason_rows(&negative)),
                ("stats", negative_stats(&mut negative)),
            ]),
        ),
    ]));
    Ok(())
}

// -- the server ---------------------------------------------------------------------------------

/// What the server keeps for this area: the round and report records of
/// every automatic run, oldest first.
#[derive(Default)]
pub struct State {
    history: Mutex<Vec<Json>>,
}

impl State {
    /// Reads the `serve` command's flags for this area (it has none).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    /// Every record so far.
    pub fn history(&self) -> Vec<Json> {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The report at the end of the last run, `None` before one has finished.
    pub fn card(&self) -> Option<Json> {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        history
            .iter()
            .rev()
            .find(|r| r.at("kind").as_str() == Some("report"))
            .cloned()
    }

    fn push(&self, record: Json) {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).push(record);
    }
}

/// Runs `f` on the server's negative network, loading it on first use from
/// beside the model file (`model.count.json` -> `model.count.negative.json`),
/// or creating an empty one in the running model's encoding.
///
/// It lives here rather than on [`Service`] because the service is shared by
/// every area being ported; the negative routes (`duo.rs`) and this loop are
/// both callers.  The model's encoding is read *before* the negative's lock is
/// taken, so the two locks are never held in the other order.
pub(crate) fn with_negative<T>(svc: &Service, f: impl FnOnce(&mut Model) -> T) -> Result<T, ApiError> {
    let encoding = svc.active_encoding();
    let mut slot = svc.negative.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        let path = (!svc.model_path.is_empty()).then(|| negative_path(&svc.model_path));
        let mut model = match path.filter(|p| std::path::Path::new(p).is_file()) {
            Some(path) => {
                let model = Model::load(&path)?;
                if !model.is_negative() {
                    return Err(ApiError::bad_request(format!(
                        "{path} holds a {} model, not a negative one",
                        model.kind()
                    )));
                }
                crate::log_info!(LOG, "negative model loaded from {path}");
                model
            }
            None => Model::new_negative(
                svc.seed,
                &NegativeOptions {
                    encoding,
                    ..Default::default()
                },
            )?,
        };
        model.workers = svc.workers;
        model.g.workers = svc.workers;
        *slot = Some(model);
    }
    Ok(f(slot.as_mut().expect("loaded above")))
}

/// This area's routes:
/// `POST /api/negative/auto`
/// `GET /api/negative/auto/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/negative/auto", auto_route);
    server.route("GET", "/api/negative/auto/history", history_route);
}

/// Starts a `critic` job: the model writes, the LLM reviews, the failures
/// blame the negative network - `{rounds (0 = until stopped), count, prefix,
/// max_length, temperature, threshold, context, provider: ollama|chatgpt,
/// reviewer_model, url, timeout, clear_passes, epochs, seed}`.  Answers 202
/// with the job, the settings and the reviewer; the rounds follow on
/// `/api/job` and `/api/negative/auto/history`.
fn auto_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = CriticConfig::from_fields(&f)?;
    let url = f.text("url")?;
    let timeout = f.number("timeout", Some(1.0))?.and_then(seconds);
    let client = svc
        .llm
        .client(&config.provider, url.as_deref(), Some(&config.reviewer_model), timeout)?;
    svc.ensure_idle()?;
    // loaded now, so a file that is not a negative network is this request's 400
    with_negative(svc, |_| ())?;
    let (client_url, reviewer) = (client.url().to_string(), client.model().to_string());
    svc.start_job("critic");
    let worker = Arc::clone(svc);
    let settings = config.clone();
    std::thread::spawn(move || {
        let outcome = Critic::new(client.as_ref(), settings).and_then(|mut critic| {
            let mut nets = Shared { svc: &worker };
            critic.run(&mut nets, &|| worker.stopping(), &mut |record| {
                worker.critic.push(record.clone());
                worker.job_progress(record.clone());
            })
        });
        worker.finish_job(outcome);
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("config", config.to_json()),
        ("url", Json::str(client_url)),
        ("reviewer", Json::str(reviewer)),
    ])))
}

/// The round and report records of every automatic run.
fn history_route(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(Json::obj([("history", Json::Arr(svc.critic.history()))]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmOptions};

    /// Marks every text 2 - "it repeats the same word over and over" - except
    /// the ones `pass` names, which get a 9.
    struct Scripted {
        pass: &'static str,
        prompts: Mutex<Vec<String>>,
    }

    impl LlmClient for Scripted {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://scripted"
        }
        fn model(&self) -> &str {
            "scripted"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, _o: &LlmOptions) -> Result<String, LlmError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            let reviews: Vec<String> = prompt
                .lines()
                .filter_map(|l| l.strip_prefix('['))
                .filter_map(|l| l.split_once("] "))
                .map(|(i, text)| {
                    let rating = if !self.pass.is_empty() && text.contains(self.pass) {
                        9
                    } else {
                        2
                    };
                    format!(
                        "{{\"index\": {i}, \"rating\": {rating}, \"critique\": \"it repeats the same word over and \
                         over\"}}"
                    )
                })
                .collect();
            Ok(format!("{{\"reviews\": [{}]}}", reviews.join(",")))
        }
        fn chat(&self, _m: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn scripted(pass: &'static str) -> Scripted {
        Scripted {
            pass,
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn trained() -> Model {
        let corpus: Vec<String> = [
            "good sentences about the cat",
            "good sentences about the dog",
            "xxxx xxxx xxxx xxxx",
            "zzzz zzzz zzzz zzzz",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut model = Model::new(5, crate::GraphOptions::default()).unwrap();
        model
            .train(
                &corpus,
                &crate::TrainOptions {
                    epochs: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    fn config(rounds: usize) -> CriticConfig {
        CriticConfig {
            rounds,
            count: 4,
            max_length: 24,
            seed: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn the_defaults_review_with_ollama_and_nonsense_is_refused() {
        let d = CriticConfig::default();
        assert_eq!((d.provider.as_str(), d.rounds, d.clear_passes), ("ollama", 3, true));
        assert!(d.validate().is_ok());
        for bad in [
            CriticConfig {
                count: 0,
                ..Default::default()
            },
            CriticConfig {
                temperature: -1.0,
                ..Default::default()
            },
            CriticConfig {
                threshold: 11.0,
                ..Default::default()
            },
            CriticConfig {
                threshold: -1.0,
                ..Default::default()
            },
            CriticConfig {
                provider: "gemini".to_string(),
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        let doc = config(2).to_json();
        assert_eq!(doc.at("rounds").as_i64(), Some(2));
        assert_eq!(doc.at("seed").as_i64(), Some(1));
    }

    #[test]
    fn a_round_writes_reviews_and_blames() {
        let mut model = trained();
        let mut negative = Model::new_negative(5, &NegativeOptions::default()).unwrap();
        let before = model.g.num_nodes();
        let reviewer = scripted("");
        let mut critic = Critic::new(&reviewer, config(1)).unwrap();
        let mut nets = Owned {
            model: &mut model,
            negative: &mut negative,
            save_each_round: None,
        };
        let record = critic.run_round(&mut nets).unwrap();
        assert_eq!(record.at("kind").as_str(), Some("round"));
        assert_eq!(record.at("round").as_i64(), Some(1));
        assert_eq!(record.at("texts").as_i64(), Some(4));
        assert_eq!(
            record.at("failed").as_i64(),
            Some(4),
            "the scripted reviewer fails everything"
        );
        assert_eq!(record.at("blamed").as_i64(), Some(4));
        assert_eq!(record.at("reasons").render(0), "{\"repetition\":4}");
        assert_eq!(record.at("reviewer").as_str(), Some("scripted"));
        assert!(negative.neg.as_ref().unwrap().blame_total > 0.0);
        assert_eq!(model.g.num_nodes(), before, "the positive model is only read from");
        let log = &negative.neg.as_ref().unwrap().log;
        assert!(log.iter().all(|e| e.source == "critic"));
        assert!(log.iter().all(|e| e.note == "it repeats the same word over and over"));
    }

    #[test]
    fn what_it_passes_is_not_blamed_and_seeded_rounds_differ() {
        let mut model = trained();
        let mut negative = Model::new_negative(5, &NegativeOptions::default()).unwrap();
        let reviewer = scripted(" ");
        let mut critic = Critic::new(&reviewer, config(2)).unwrap();
        let mut seen = Vec::new();
        let records = {
            let mut nets = Owned {
                model: &mut model,
                negative: &mut negative,
                save_each_round: None,
            };
            critic
                .run(&mut nets, &|| false, &mut |r| {
                    seen.push(r.at("kind").as_str().unwrap_or("").to_string())
                })
                .unwrap()
        };
        assert_eq!(seen, vec!["round", "round", "report"]);
        assert_eq!(records.len(), 3);
        let prompts = reviewer.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        assert_ne!(prompts[0], prompts[1], "the seed advances with the round");
        let card = &records[2];
        assert_eq!(card.at("rounds").as_i64(), Some(2));
        assert_eq!(card.at("reviewed").as_i64(), Some(8));
    }

    #[test]
    fn a_stop_ends_the_run_between_rounds() {
        let mut model = trained();
        let mut negative = Model::new_negative(5, &NegativeOptions::default()).unwrap();
        let reviewer = scripted("");
        let mut critic = Critic::new(&reviewer, config(0)).unwrap();
        let rounds = std::cell::Cell::new(0);
        let records = {
            let mut nets = Owned {
                model: &mut model,
                negative: &mut negative,
                save_each_round: None,
            };
            critic
                .run(&mut nets, &|| rounds.get() >= 2, &mut |r| {
                    if r.at("kind").as_str() == Some("round") {
                        rounds.set(rounds.get() + 1)
                    }
                })
                .unwrap()
        };
        assert_eq!(records.len(), 3, "two rounds and the report of a run with no end");
    }

    #[test]
    fn the_report_card_adds_up() {
        let round = |texts: i64, mean: f64, rate: f64, blamed: i64, reasons: &str| {
            let mut doc = crate::json::parse(&format!(
                "{{\"kind\": \"round\", \"texts\": {texts}, \"blamed\": {blamed}, \"cleared\": 1, \"edges\": 3, \
                 \"reasons\": {reasons}, \"stats\": {{\"x\": {texts}}}}}"
            ))
            .unwrap();
            if let Json::Obj(pairs) = &mut doc {
                pairs.push(("mean_rating".to_string(), Json::Num(mean)));
                pairs.push(("pass_rate".to_string(), Json::Num(rate)));
            }
            doc
        };
        let records = vec![
            round(2, 3.0, 0.0, 2, "{\"b\": 1, \"a\": 1}"),
            round(2, 7.0, 1.0, 0, "{\"a\": 1}"),
            Json::obj([("kind", Json::str("report"))]),
        ];
        let card = report_card(&records);
        assert_eq!(card.at("rounds").as_i64(), Some(2));
        assert_eq!(card.at("reviewed").as_i64(), Some(4));
        assert_eq!(card.at("blamed").as_i64(), Some(2));
        assert_eq!(card.at("mean_rating").as_f64(), Some(5.0));
        assert_eq!(card.at("pass_rate").as_f64(), Some(0.5));
        assert_eq!(card.at("trend").as_f64(), Some(4.0));
        assert_eq!(card.at("reasons").render(0), "{\"a\":2,\"b\":1}");
        assert_eq!(card.at("stats").render(0), "{\"x\":2}");
        let empty = report_card(&[]);
        assert_eq!(empty.at("rounds").as_i64(), Some(0));
        assert!(empty.at("trend").is_null() && empty.at("stats").is_null());
        assert!(
            report_card(&records[..1]).at("trend").is_null(),
            "one round has no trend"
        );
    }

    #[test]
    fn a_request_body_is_read_as_python_reads_it() {
        let body = crate::json::parse(
            "{\"rounds\": 1, \"count\": 3, \"provider\": \"OpenAI\", \"model\": \"gpt-x\", \"seed\": 4}",
        )
        .unwrap();
        let config = CriticConfig::from_fields(&Fields::new(&body)).unwrap();
        assert_eq!(
            (config.provider.as_str(), config.reviewer_model.as_str(), config.seed),
            ("chatgpt", "gpt-x", Some(4))
        );
        for bad in [
            "{\"count\": 0}",
            "{\"provider\": \"gemini\"}",
            "{\"threshold\": 11}",
            "{\"rounds\": -1}",
        ] {
            let body = crate::json::parse(bad).unwrap();
            let err = CriticConfig::from_fields(&Fields::new(&body)).unwrap_err();
            assert_eq!(err.status, 400, "{bad}");
        }
        let body = crate::json::parse("{\"provider\": \"gemini\"}").unwrap();
        assert_eq!(
            CriticConfig::from_fields(&Fields::new(&body)).unwrap_err().message,
            "'provider' must be one of ollama, chatgpt (got 'gemini')"
        );
    }

    #[test]
    fn the_server_loop_takes_each_lock_for_a_step() {
        let dir = std::env::temp_dir().join(format!("radixnet-critic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.count.json").to_string_lossy().into_owned();
        let svc = Service::new(trained(), path.clone(), 5, 1);
        let reviewer = scripted("");
        let mut critic = Critic::new(&reviewer, config(1)).unwrap();
        let mut nets = Shared { svc: &svc };
        let records = critic.run(&mut nets, &|| false, &mut |_| {}).unwrap();
        assert_eq!(records.len(), 2);
        let blame = with_negative(&svc, |n| n.neg.as_ref().unwrap().blame_total).unwrap();
        assert!(blame > 0.0);
        assert!(
            !std::path::Path::new(&negative_path(&path)).exists(),
            "the server keeps it in memory until something saves it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
