//! The English tutor from the command line (`radixnet tutor`) and over HTTP
//! (`/api/tutor/*`): the `cmd_tutor` of `radixnet/cli.py` and the tutor
//! routes of `radixnet/api.py`, speaking their JSON.
//!
//! Both build the same [`TutorTrainer`] and differ only in where the networks
//! live ([`Networks`]): the command owns the model - and, with `--blame`, the
//! negative network - for the length of the run and saves them at the end;
//! the server runs the lessons as a `tutor` job that takes each lock for a
//! step at a time and never while the teacher thinks, and keeps every record
//! for `GET /api/tutor/history` and for the plan that follows it.

use std::sync::{Arc, Mutex};

use crate::cli::{negative_stats, Ctx};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::llm::fields::{py_repr_of, Fields};
use crate::llm::{new_client, normalise_provider, LlmClient, CHATGPT, DEFAULT_PROVIDER, PROVIDERS};
use crate::negative::python_repr;
use crate::plan::{plan_lessons, PlanRequest, DEFAULT_PLAN_LESSONS, LEVELS, UPGRADE_STEPS, WORDS_LADDER};
use crate::service::Service;

use super::trainer::{Networks, TutorConfig, TutorTrainer};
use super::{default_tutor_model, report_card, Exercise, ERROR_TYPES, MODES, TWONRL_PER};

/// What this module's own lines are filed under.
const LOG: &str = "tutor";

// -- the command line -----------------------------------------------------------------------------

/// A record as the command reports it while it runs: a marked sentence, a
/// round, a report card, a plan, the next batch or a note - on stderr, so the
/// one document on stdout stays whole.
fn say(record: &Json) {
    let text = |key: &str| record.at(key).as_str().unwrap_or("").to_string();
    let mark = |key: &str| match record.at(key).as_f64() {
        Some(x) => format!("{x:.2}"),
        None => "-".to_string(),
    };
    match record.at("kind").as_str() {
        Some("lesson") => {
            let passed = record.at("passed").as_bool() == Some(true);
            crate::log_info!(
                LOG,
                "{} {} {}/10 {:?}",
                text("exercise"),
                if passed { "pass" } else { "fail" },
                mark("score"),
                text("sentence")
            );
            if !passed && !text("correction").is_empty() {
                crate::log_info!(LOG, "    correct: {:?} ({})", text("correction"), text("error"));
            }
        }
        Some("round") => crate::log_info!(
            LOG,
            "round {}: {}/{} passed, mean {} (grammar {}), weakest: {} -> {}",
            record.at("round").as_i64().unwrap_or(0),
            record.at("passed").as_i64().unwrap_or(0),
            record.at("lessons").as_i64().unwrap_or(0),
            mark("mean_score"),
            mark("mean_grammar"),
            record.at("weakest").to_strings().join(", "),
            record.at("action").as_str().unwrap_or("nothing to learn")
        ),
        Some("report") => crate::log_info!(
            LOG,
            "report card: {}/{} passed, mean {}",
            record.at("passed").as_i64().unwrap_or(0),
            record.at("lessons").as_i64().unwrap_or(0),
            mark("mean_score")
        ),
        Some("plan") => crate::log_info!(
            LOG,
            "lesson plan ({}): {} lesson(s) at {} level, drilling {}",
            text("source"),
            record.at("lessons").as_array().len(),
            text("level"),
            record.at("targets").to_strings().join(", ")
        ),
        Some("batch") => crate::log_info!(
            LOG,
            "batch {} ({}): {} level, openings of {} words - {}",
            record.at("batch").as_i64().unwrap_or(0),
            text("step"),
            text("level"),
            text("words"),
            text("brief")
        ),
        Some("note") => crate::log_warn!(LOG, "note: {}", text("message")),
        _ => {}
    }
}

/// A number flag that may be left out, but must be at least `minimum` when given.
fn optional_count(ctx: &Ctx, name: &str, minimum: i64) -> Result<Option<usize>, String> {
    match ctx.args.get(name) {
        Some(_) => Ok(Some(crate::llm::count_flag(ctx, name, 0, minimum)?)),
        None => Ok(None),
    }
}

/// The settings of `radixnet tutor`, flag for flag Python's.
fn config_from_args(ctx: &Ctx) -> Result<TutorConfig, String> {
    let args = &ctx.args;
    let d = TutorConfig::default();
    let count = |name: &str, default: usize, minimum: i64| crate::llm::count_flag(ctx, name, default as i64, minimum);
    let number = |name: &str, default: f64| crate::llm::nonneg_flag(ctx, name, default);
    let provider = args
        .get("tutor-provider")
        .or_else(|| args.get("provider"))
        .unwrap_or(DEFAULT_PROVIDER)
        .to_string();
    let provider = normalise_provider(&provider)?.to_string();
    let grader_provider = match args.get("grader-provider").filter(|p| !p.is_empty()) {
        Some(p) => normalise_provider(p)?.to_string(),
        None => provider.clone(),
    };
    let tutor_model = match args.get("tutor-model").filter(|m| !m.is_empty()) {
        Some(m) => m.to_string(),
        None => default_tutor_model(&provider),
    };
    let strength = if args.get("strength").is_some() {
        Some(number("strength", 1.0)?)
    } else {
        None
    };
    let mut config = TutorConfig {
        topic: args.str("topic", &d.topic),
        rounds: count("rounds", d.rounds, 1)?,
        exercises: count("exercises", d.exercises, 1)?,
        attempts: count("attempts", d.attempts, 1)?,
        focus: args.get("focus").map(str::to_string),
        level: args.str("level", &d.level),
        words: args.str("words", &d.words),
        brief: args.str("brief", ""),
        tutor_provider: provider,
        tutor_model,
        grader_provider,
        grader_model: args.get("grader-model").map(str::to_string),
        mode: args.str("mode", &d.mode),
        length: count("length", d.length, 0)?,
        max_length: count("max-length", d.max_length, 1)?,
        temperature: number("temperature", d.temperature)?,
        to_end: !args.on("no-to-end"),
        beam: optional_count(ctx, "beam", 1)?,
        threshold: number("threshold", d.threshold)?,
        grammar_weight: number("grammar-weight", d.grammar_weight)?,
        batch: count("batch", d.batch, 1)?,
        adapt: !args.on("no-adapt"),
        drills: count("drills", d.drills, 0)?,
        variants: count("variants", d.variants, 0)?,
        variant_weight: number("variant-weight", d.variant_weight)?,
        plan: count("plan", 0, 0)?,
        batches: count("batches", d.batches, 0)?,
        teach_answer: !args.on("no-teach-answer"),
        learn: !args.on("dry-run"),
        twonrl_per: args.str("twonrl-per", &d.twonrl_per),
        diff_corrections: !args.on("no-diff-corrections"),
        keep_weight: number("keep-weight", d.keep_weight)?,
        min_weight: number("min-weight", d.min_weight)?,
        neg_epochs: count("neg-epochs", d.neg_epochs, 0)?,
        pos_epochs: count("pos-epochs", d.pos_epochs, 0)?,
        neg_lr: number("neg-lr", d.neg_lr)?,
        pos_lr: number("pos-lr", d.pos_lr)?,
        batch_size: count("batch-size", d.batch_size, 1)?,
        strength,
        replay: !args.on("no-replay"),
        replay_limit: count("replay-limit", d.replay_limit, 0)?,
        checkpoint_every: 0,
    };
    config.resolve()?;
    Ok(config)
}

/// The teacher, and the marker when it is another client (another provider
/// or another endpoint); `None` means the teacher marks its own.
type Clients = (Box<dyn LlmClient>, Option<Box<dyn LlmClient>>);

/// `radixnet tutor`: automated English lessons.  The teacher writes the
/// openings, the network completes them, the teacher marks every sentence,
/// and the marks decide what the network learns; `--blame` also teaches the
/// negative network why each failure failed, `--plan N` plans the next N
/// lessons from the final report card and `--batches N` (0: until stopped)
/// runs batch after batch, each taught to the plan the one before led to.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let schedule = crate::checkpoint::Schedule::from_args(args)?;
    let mut config = config_from_args(ctx)?;
    config.checkpoint_every = schedule.every;
    if (config.tutor_provider == CHATGPT || config.grader_provider == CHATGPT) && !crate::chatgpt::key_configured() {
        return Err(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT teach, or use \
                    --tutor-provider ollama"
                .to_string(),
        );
    }
    config.validate()?;
    let timeout = crate::llm::timeout_flag(ctx)?;
    let client = new_client(
        &config.tutor_provider,
        &args.str("url", ""),
        &config.tutor_model,
        timeout,
    )?;
    let grader_url = args.str("grader-url", "");
    let grader: Option<Box<dyn LlmClient>> =
        if config.grader_provider == config.tutor_provider && grader_url.trim().is_empty() {
            None
        } else {
            Some(new_client(
                &config.grader_provider,
                &grader_url,
                &config.resolved_grader_model(),
                timeout,
            )?)
        };
    let (client, grader): Clients = (client, grader);
    let origin = if std::path::Path::new(&ctx.model_path).exists() {
        Json::obj([
            ("kind", Json::str("model")),
            ("path", Json::str(ctx.model_path.clone())),
        ])
    } else {
        Json::obj([("kind", Json::str("new")), ("path", Json::Null)])
    };
    let mut model = ctx.open(false)?;
    let mut negative = if args.on("blame") {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let out = ctx.out.clone().unwrap_or_else(|| ctx.model_path.clone());
    let marker: &dyn LlmClient = grader.as_deref().unwrap_or(client.as_ref());
    crate::log_info!(
        LOG,
        "{} round(s) x {} exercise(s) about {:?}; teacher {}: {} at {}, marked by {}: {} at {}, pass at {}/10",
        config.rounds,
        config.exercises,
        config.topic,
        config.tutor_provider,
        config.tutor_model,
        client.url(),
        config.grader_provider,
        config.resolved_grader_model(),
        marker.url(),
        crate::llm::format_g(config.threshold)
    );
    // the settings as the run left them: an auto run applies every plan to
    // them (the brief, the level, the openings, the pass mark), and Python
    // reports the config its trainer changed
    let (records, lessons, config) = {
        let mut trainer = TutorTrainer::new(client.as_ref(), marker, config.clone())?
            .with_progress(say)
            .with_checkpoints(schedule.manager());
        let mut nets = Networks::Owned {
            model: &mut model,
            negative: negative.as_mut(),
        };
        let records = trainer.run(&mut nets)?;
        (records, trainer.lessons, trainer.config)
    };
    let saved = if config.learn {
        model.save(&out)?;
        crate::log_info!(LOG, "saved {out}");
        crate::llm::saved(&out)
    } else {
        Json::Null
    };
    let card = report_card(&lessons);
    // the plan that says what comes *after* the run: the one the last batch's card led to
    let last_batch = records
        .iter()
        .filter(|r| r.at("kind").as_str() == Some("report"))
        .filter_map(|r| r.at("batch").as_i64())
        .max()
        .unwrap_or(1);
    let plan = records
        .iter()
        .rev()
        .find(|r| r.at("kind").as_str() == Some("plan") && r.at("batch").as_i64().unwrap_or(1) == last_batch)
        .cloned()
        .unwrap_or(Json::Null);
    let negative_doc = match negative.as_mut() {
        Some(negative) => {
            let path = ctx.negative_path();
            negative.save(&path)?;
            Json::obj([
                ("path", Json::str(path.clone())),
                ("saved", crate::llm::saved(&path)),
                ("reasons", crate::review::reason_rows(negative)),
                ("stats", negative_stats(negative)),
            ])
        }
        None => Json::Null,
    };
    let doc = Json::obj([
        ("model", origin),
        ("out", if config.learn { Json::str(out) } else { Json::Null }),
        ("config", config.to_json()),
        ("records", Json::Arr(records)),
        ("lessons", Json::Arr(lessons.iter().map(|l| l.to_json()).collect())),
        ("report", card),
        ("plan", plan),
        ("interrupted", Json::Bool(false)),
        ("saved", saved),
        ("stats", crate::report::stats(&model)),
        ("negative", negative_doc),
    ]);
    if let Some(path) = args.get("report") {
        let keep = ["config", "records", "lessons", "report", "plan"];
        let report = Json::Obj(keep.iter().map(|k| (k.to_string(), doc.at(k).clone())).collect());
        std::fs::write(path, report.render(2)).map_err(|err| format!("cannot write {path}: {err}"))?;
        crate::log_info!(LOG, "wrote report to {path}");
    }
    ctx.emit(doc);
    Ok(())
}

// -- the server -----------------------------------------------------------------------------------

/// What the server keeps for this area: the lesson, round, report, plan,
/// batch and note records of every tutor run, oldest first.
#[derive(Default)]
pub struct State {
    history: Mutex<Vec<Json>>,
}

impl State {
    /// Reads the `serve` command's flags for this area (it has none of its own:
    /// the teacher is the LLM area's).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    /// Every record so far.
    pub fn history(&self) -> Vec<Json> {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The report card at the end of the last run, `None` before one has been
    /// marked.
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

/// This area's routes:
/// `GET /api/tutor`
/// `POST /api/tutor/start`
/// `POST /api/tutor/lesson`
/// `POST /api/tutor/plan`
/// `GET /api/tutor/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/tutor", describe);
    server.route("POST", "/api/tutor/start", start);
    server.route("POST", "/api/tutor/lesson", lesson);
    server.route("POST", "/api/tutor/plan", plan);
    server.route("GET", "/api/tutor/history", history);
}

/// A provider field and its alias: `ollama` | `chatgpt` (`openai`, `gpt`, ...
/// too), `default` when neither is given.
fn provider_field(f: &Fields, name: &str, alias: Option<&str>, default: &str) -> Result<String, ApiError> {
    let mut raw = f.text(name)?.filter(|p| !p.is_empty());
    if raw.is_none() {
        if let Some(alias) = alias {
            raw = f.text(alias)?.filter(|p| !p.is_empty());
        }
    }
    let Some(raw) = raw else {
        return Ok(default.to_string());
    };
    normalise_provider(&raw).map(str::to_string).map_err(|_| {
        ApiError::bad_request(format!(
            "'{name}' must be one of {} (got {})",
            PROVIDERS.join(", "),
            python_repr(&raw)
        ))
    })
}

/// A string field that counts only when it is not empty (Python's `f.text(name, None) or ...`).
fn nonempty(f: &Fields, name: &str) -> Result<Option<String>, ApiError> {
    Ok(f.text(name)?.filter(|t| !t.is_empty()))
}

/// The tutoring settings of a request body, defaults from [`TutorConfig`] and
/// the teacher from the server's own - read in Python's order, so the first
/// bad field is the one refused.
fn config_from_fields(svc: &Service, f: &Fields) -> Result<TutorConfig, ApiError> {
    let d = TutorConfig::default();
    let tutor_provider = provider_field(f, "tutor_provider", Some("provider"), &d.tutor_provider)?;
    let grader_provider = provider_field(f, "grader_provider", None, &tutor_provider)?;
    let mut grader_model = nonempty(f, "grader_model")?;
    if grader_model.is_none() && grader_provider != tutor_provider {
        grader_model = Some(svc.llm.tutor_model(&grader_provider));
    }
    let count = |name: &str, default: usize, minimum: usize| f.count_or(name, default, minimum);
    let number = |name: &str, default: f64| f.number_or(name, default, Some(0.0));
    let topic = f.text_or("topic", &d.topic)?;
    let rounds = count("rounds", d.rounds, 1)?;
    let exercises = count("exercises", d.exercises, 1)?;
    let attempts = count("attempts", d.attempts, 1)?;
    let focus = nonempty(f, "focus")?;
    let level = f.text_or("level", &d.level)?;
    let words = f.text_or("words", &d.words)?;
    let brief = f.text_or("brief", &d.brief)?;
    let tutor_model = match nonempty(f, "tutor_model")? {
        Some(m) => m,
        None => match nonempty(f, "model")? {
            Some(m) => m,
            None => svc.llm.tutor_model(&tutor_provider),
        },
    };
    let mut config = TutorConfig {
        topic,
        rounds,
        exercises,
        attempts,
        focus,
        level,
        words,
        brief,
        tutor_provider,
        tutor_model,
        grader_provider,
        grader_model,
        mode: f.text_or("mode", &d.mode)?.trim().to_lowercase(),
        length: count("length", d.length, 0)?,
        max_length: count("max_length", d.max_length, 1)?,
        temperature: number("temperature", d.temperature)?,
        to_end: f.flag("to_end", d.to_end)?,
        beam: f.integer("beam", Some(1))?.map(|b| b as usize),
        threshold: number("threshold", d.threshold)?,
        grammar_weight: number("grammar_weight", d.grammar_weight)?,
        batch: count("batch", d.batch, 1)?,
        adapt: f.flag("adapt", d.adapt)?,
        drills: count("drills", d.drills, 0)?,
        variants: count("variants", d.variants, 0)?,
        variant_weight: number("variant_weight", d.variant_weight)?,
        plan: count("plan", d.plan, 0)?,
        batches: count("batches", d.batches, 0)?,
        teach_answer: f.flag("teach_answer", d.teach_answer)?,
        learn: f.flag("learn", d.learn)?,
        twonrl_per: f.text_or("twonrl_per", &d.twonrl_per)?.trim().to_lowercase(),
        diff_corrections: f.flag("diff_corrections", d.diff_corrections)?,
        keep_weight: number("keep_weight", d.keep_weight)?,
        min_weight: number("min_weight", d.min_weight)?,
        neg_epochs: count("neg_epochs", d.neg_epochs, 0)?,
        pos_epochs: count("pos_epochs", d.pos_epochs, 0)?,
        neg_lr: number("neg_lr", d.neg_lr)?,
        pos_lr: number("pos_lr", d.pos_lr)?,
        batch_size: count("batch_size", d.batch_size, 1)?,
        strength: f.number("strength", Some(0.0))?,
        replay: f.flag("replay", d.replay)?,
        replay_limit: count("replay_limit", d.replay_limit, 0)?,
        checkpoint_every: count("checkpoint_every", d.checkpoint_every, 0)?,
    };
    config.resolve().map_err(ApiError::bad_request)?;
    config.validate().map_err(ApiError::bad_request)?;
    Ok(config)
}

/// The teacher and the marker of a request (`url` / `grader_url` / `timeout`
/// override the server's defaults); one client unless the providers or the
/// endpoints differ.
fn clients(svc: &Service, f: &Fields, config: &TutorConfig) -> Result<Clients, ApiError> {
    let timeout = f.number("timeout", Some(1.0))?.and_then(crate::llm::seconds);
    let url = f.text("url")?;
    let client = svc.llm.client(
        &config.tutor_provider,
        url.as_deref(),
        Some(&config.tutor_model),
        timeout,
    )?;
    let grader_url = nonempty(f, "grader_url")?;
    if config.grader_provider == config.tutor_provider && grader_url.is_none() {
        return Ok((client, None));
    }
    let grader = svc.llm.client(
        &config.grader_provider,
        grader_url.as_deref(),
        Some(&config.resolved_grader_model()),
        timeout,
    )?;
    Ok((client, Some(grader)))
}

/// `GET /api/tutor`: the teachers on offer, the marking vocabulary and every default.
fn describe(svc: &Arc<Service>, _r: &Request) -> Answer {
    let llm = &svc.llm;
    Ok(Json::obj([
        ("url", Json::str(llm.ollama_url.clone())),
        ("model", Json::str(llm.tutor_model(DEFAULT_PROVIDER))),
        ("env_model", Json::str(default_tutor_model(DEFAULT_PROVIDER))),
        (
            "providers",
            Json::obj([
                (
                    "ollama",
                    Json::obj([
                        ("url", Json::str(llm.ollama_url.clone())),
                        ("model", Json::str(llm.tutor_model("ollama"))),
                        ("configured", Json::Bool(true)),
                    ]),
                ),
                (
                    "chatgpt",
                    Json::obj([
                        ("url", Json::str(llm.chatgpt_url.clone())),
                        ("model", Json::str(llm.tutor_model(CHATGPT))),
                        ("configured", Json::Bool(crate::chatgpt::key_configured())),
                    ]),
                ),
            ]),
        ),
        ("error_types", Json::strs(ERROR_TYPES.iter().copied())),
        ("modes", Json::strs(MODES.iter().copied())),
        ("twonrl_per", Json::strs(TWONRL_PER.iter().copied())),
        ("levels", Json::strs(LEVELS.iter().copied())),
        ("words_ladder", Json::strs(WORDS_LADDER.iter().copied())),
        ("upgrade_steps", Json::strs(UPGRADE_STEPS.iter().copied())),
        ("plan_lessons", Json::Int(DEFAULT_PLAN_LESSONS as i64)),
        ("defaults", TutorConfig::default().to_json()),
    ]))
}

/// `POST /api/tutor/start`: rounds of prefix -> completion -> grade -> 2NRL as
/// a `tutor` job; with `blame`, every failed sentence also teaches the
/// negative network why it failed.  Answers 202 with the job, the settings and
/// the teacher's endpoint; the records follow on `/api/job` and
/// `/api/tutor/history`.
fn start(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = config_from_fields(svc, &f)?;
    let (client, grader) = clients(svc, &f, &config)?;
    let blame = f.flag("blame", false)?;
    if config.checkpoint_every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    svc.ensure_idle()?;
    if blame {
        // loaded now, so a file that is not a negative network is this request's 400
        svc.ensure_negative()?;
    }
    let url = client.url().to_string();
    svc.start_job("tutor");
    let worker = Arc::clone(svc);
    let settings = config.clone();
    std::thread::spawn(move || {
        let marker: &dyn LlmClient = grader.as_deref().unwrap_or(client.as_ref());
        let every = settings.checkpoint_every;
        let outcome = TutorTrainer::new(client.as_ref(), marker, settings).and_then(|trainer| {
            let mut trainer = trainer
                .with_stop(|| worker.stopping())
                .with_progress(|record| {
                    worker.tutor.push(record.clone());
                    worker.job_progress(record.clone());
                })
                .with_checkpoints(if every > 0 {
                    crate::checkpoint::manager(&worker)
                } else {
                    None
                });
            trainer.run(&mut Networks::Shared {
                svc: &worker,
                negative: blame,
            })
        });
        worker.finish_job(outcome.map(|_| worker.job_records()));
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("config", config.to_json()),
        ("url", Json::str(url)),
    ])))
}

/// `GET /api/tutor/history`: the records of every tutor run.
fn history(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(Json::obj([("history", Json::Arr(svc.tutor.history()))]))
}

/// `POST /api/tutor/lesson`: one round of lessons without training - the
/// exercises (or the given `prefixes`), the network's completions, their
/// grades and the report card.  The search runs under the model lock, the
/// teacher's calls do not.
fn lesson(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = config_from_fields(svc, &f)?;
    let (client, grader) = clients(svc, &f, &config)?;
    let given = f.texts_optional("prefixes", "prefix")?;
    let marker: &dyn LlmClient = grader.as_deref().unwrap_or(client.as_ref());
    let trainer = TutorTrainer::new(client.as_ref(), marker, config.clone()).map_err(ApiError::bad_request)?;
    let exercises: Vec<Exercise> = if given.is_empty() {
        trainer.set_exercises(1)?
    } else {
        given
            .iter()
            .take(config.exercises)
            .enumerate()
            .map(|(i, prefix)| Exercise {
                id: format!("e{}", i + 1),
                prefix: prefix.clone(),
                focus: config.focus.clone().unwrap_or_default(),
                answer: String::new(),
            })
            .collect()
    };
    let mut lessons = svc.with_model(|m| -> Result<Vec<super::Lesson>, String> {
        let mut out = Vec::new();
        for exercise in &exercises {
            for attempt in 0..config.attempts {
                out.push(trainer.complete(m, exercise, attempt)?);
            }
        }
        Ok(out)
    })?;
    trainer.grade(&mut lessons)?;
    Ok(Json::obj([
        (
            "source",
            Json::str(if given.is_empty() {
                config.tutor_provider.as_str()
            } else {
                "given"
            }),
        ),
        ("model", Json::str(client.model())),
        ("url", Json::str(client.url())),
        ("config", config.to_json()),
        (
            "exercises",
            Json::Arr(exercises.iter().map(Exercise::to_json).collect()),
        ),
        ("lessons", Json::Arr(lessons.iter().map(|l| l.to_json()).collect())),
        ("report", report_card(&lessons)),
    ]))
}

/// `report`: a JSON object field, `None` when it was not given.
fn mapping(f: &Fields, name: &str) -> Result<Option<Json>, ApiError> {
    match f.get(name) {
        None => Ok(None),
        Some(card @ Json::Obj(_)) => Ok(Some(card.clone())),
        Some(other) => Err(ApiError::bad_request(format!(
            "'{name}' must be an object (got {} {})",
            json_type(other),
            py_repr_of(other)
        ))),
    }
}

/// The JSON type of a value, as the refusals name it.
fn json_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Int(_) | Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Arr(_) => "array",
        Json::Obj(_) => "object",
    }
}

/// `POST /api/tutor/plan`: the lessons to run next - the teacher turns a
/// report card (`report`, else the card at the end of the last run) into a
/// syllabus of its weakest points, with the brief and the step up for the
/// next batch.
fn plan(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = config_from_fields(svc, &f)?;
    let (client, _grader) = clients(svc, &f, &config)?;
    let card = match mapping(&f, "report")? {
        Some(card) => card,
        None => svc.tutor.card().ok_or_else(|| {
            ApiError::bad_request("no report card yet: run some lessons first, or send one as 'report'")
        })?,
    };
    let fallback = if config.plan > 0 {
        config.plan
    } else {
        DEFAULT_PLAN_LESSONS
    };
    let count = f.count_or("count", fallback, 1)?;
    let planned = plan_lessons(
        client.as_ref(),
        &card,
        &PlanRequest {
            topic: config.topic.clone(),
            level: config.level.clone(),
            words: config.words.clone(),
            threshold: config.threshold,
            count,
            exercises: config.exercises as i64,
            drills: config.drills as i64,
            model: config.tutor_model.clone(),
            temperature: 0.0,
        },
    )?;
    Ok(Json::obj([
        ("plan", planned.to_json()),
        ("source", Json::str(planned.source.clone())),
        ("provider", Json::str(config.tutor_provider.clone())),
        ("model", Json::str(client.model())),
        ("url", Json::str(client.url())),
        ("report", card),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parse_args;

    fn ctx(argv: &[&str]) -> Ctx {
        let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        let (command, args) = parse_args(&argv).unwrap();
        Ctx {
            command,
            args,
            model_path: "m.count.json".to_string(),
            encoding: Default::default(),
            seed: 0,
            workers: 1,
            json: true,
            out: None,
        }
    }

    #[test]
    fn the_command_reads_python_s_flags() {
        let c = config_from_args(&ctx(&[
            "tutor",
            "--topic",
            "animals",
            "--rounds",
            "2",
            "--mode",
            "beam",
            "--threshold",
            "9.5",
            "--dry-run",
            "--plan",
            "2",
            "--provider",
            "OpenAI",
            "--beam",
            "4",
            "--strength",
            "0.5",
        ]))
        .unwrap();
        assert_eq!((c.topic.as_str(), c.rounds, c.mode.as_str()), ("animals", 2, "beam"));
        assert_eq!((c.threshold, c.learn, c.plan), (9.5, false, 2));
        assert_eq!(
            (c.tutor_provider.as_str(), c.grader_provider.as_str()),
            ("chatgpt", "chatgpt")
        );
        assert_eq!((c.beam, c.strength), (Some(4), Some(0.5)));
        assert!(config_from_args(&ctx(&["tutor", "--rounds", "0"])).is_err());
        assert!(config_from_args(&ctx(&["tutor", "--beam", "0"])).is_err());
        assert!(config_from_args(&ctx(&["tutor", "--provider", "bard"])).is_err());
    }

    #[test]
    fn a_request_body_is_read_as_python_reads_it() {
        let svc = Service::new(crate::Model::new(0, Default::default()).unwrap(), String::new(), 0, 1);
        let body = crate::json::parse(
            "{\"topic\": \"animals\", \"mode\": \" BEAM \", \"grader_provider\": \"gpt\", \"model\": \"m\", \
             \"threshold\": 9}",
        )
        .unwrap();
        let config = config_from_fields(&svc, &Fields::new(&body)).unwrap();
        assert_eq!((config.mode.as_str(), config.tutor_model.as_str()), ("beam", "m"));
        assert_eq!(config.grader_provider, "chatgpt");
        assert_eq!(config.grader_model.as_deref(), Some(svc.llm.chatgpt_model.as_str()));
        assert_eq!(config.threshold, 9.0);
        for (bad, words) in [
            ("{\"topic\": \" \"}", "topic"),
            ("{\"mode\": \"nope\"}", "mode"),
            ("{\"twonrl_per\": \"hourly\"}", "twonrl_per"),
            ("{\"threshold\": 99}", "threshold"),
            ("{\"exercises\": 0}", "'exercises' must be >= 1 (got 0)"),
            (
                "{\"tutor_provider\": \"bard\"}",
                "'tutor_provider' must be one of ollama, chatgpt (got 'bard')",
            ),
            ("{\"strength\": -1}", "'strength' must be >= 0.0 (got -1)"),
        ] {
            let body = crate::json::parse(bad).unwrap();
            let err = config_from_fields(&svc, &Fields::new(&body)).unwrap_err();
            assert_eq!(err.status, 400, "{bad}");
            assert!(err.message.contains(words), "{bad}: {}", err.message);
        }
        let body = crate::json::parse("{\"report\": \"a card\"}").unwrap();
        assert_eq!(
            mapping(&Fields::new(&body), "report").unwrap_err().message,
            "'report' must be an object (got string 'a card')"
        );
    }
}
