//! The HTTP API of the Rust port: the same JSON contract as the Python and Go
//! servers, so `frontend/dist` runs against it unchanged.
//!
//! What it serves is what this crate *has* - the model, both traversals, the
//! word alphabet, the judged paths, the graph and the model file - and it says
//! so: the kinds it offers are `count` and `word`, and a request for the
//! negative network, a tutor or an LLM is answered with the one line that says
//! which server to ask instead.  A frontend tab that needs one of those is
//! hidden when `engine` is `rust` (`frontend/src/App.jsx`).
//!
//! One model, one lock.  A training run holds it for as long as it takes, so a
//! second request waits rather than reading a half-built graph; the job it
//! reports is the one the frontend polls.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::clock::utc_now;
use crate::encoding::{parse_encoding, Encoding, Unit, BACK_LABEL, END_LABEL, START_LABEL};
use crate::graph::{END, FIRST, START};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::model::{GenerateOptions, Model, PredictOptions, TrainOptions};
use crate::penalty::{resolve_traversal, DEFAULT_TRAVERSAL, TRAVERSALS};
use crate::report::{configure_weight, node_rows, path_rows, split_texts, stats, weight_config};
use crate::search::PathResult;
use crate::GraphOptions;

/// What this server can be asked for and what it answers with.
pub const ENGINE: &str = "rust";

/// What the service's own lines are filed under.
const LOG: &str = "model";

/// A background job: one training run at a time, polled by the frontend.
#[derive(Default)]
pub struct Job {
    pub id: String,
    pub kind: String,
    pub state: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub error: Option<String>,
    pub records: Vec<Json>,
}

impl Job {
    fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("id".to_string(), Json::str(self.id.clone())),
            ("type".to_string(), Json::str(self.kind.clone())),
            ("state".to_string(), Json::str(self.state.clone())),
            ("started_at".to_string(), Json::str(self.started_at.clone())),
            (
                "progress".to_string(),
                self.records.last().cloned().unwrap_or(Json::Null),
            ),
            ("history".to_string(), Json::Arr(self.records.clone())),
        ];
        pairs.push((
            "finished_at".to_string(),
            match &self.finished_at {
                Some(at) => Json::str(at.clone()),
                None => Json::Null,
            },
        ));
        pairs.push((
            "error".to_string(),
            match &self.error {
                Some(message) => Json::str(message.clone()),
                None => Json::Null,
            },
        ));
        Json::Obj(pairs)
    }
}

/// The service: one model behind one lock, plus where its files live.
pub struct Service {
    model: Mutex<Model>,
    /// The encodings that were running earlier, kept as they were left.
    ///
    /// Changing encoding parks the model that was running rather than dropping
    /// it, so a switch away and back does not lose unsaved training - which is
    /// what the Python service does for a kind (`ModelService._parked`).
    parked: Mutex<Vec<(String, Model)>>,
    /// The encoding this server was started on: the one whose file is
    /// `model_path` rather than a name derived from it.
    born_with: Encoding,
    job: Mutex<Option<Job>>,
    stop: AtomicBool,
    jobs: AtomicUsize,
    pub model_path: String,
    pub upload_dir: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub seed: i64,
    pub workers: usize,
    // (read by the areas being ported - the allow goes when they land)
    #[allow(dead_code)]
    /// The negative network this server guards its output paths with, loaded
    /// from beside the model file the first time something needs it.
    pub(crate) negative: Mutex<Option<Model>>,
    #[allow(dead_code)]
    /// How strictly the negative network guards the output paths.
    pub(crate) guard: Mutex<crate::duo::FilterConfig>,
    // what each area keeps between requests; an area's routes own its state
    pub(crate) llm: crate::llm::State,
    pub(crate) tools: crate::tools::State,
    pub(crate) checkpoints: crate::checkpoint::State,
    pub(crate) evolve: crate::gan::State,
    pub(crate) tutor: crate::tutor::State,
    pub(crate) critic: crate::critic::State,
    pub(crate) chat: crate::chat::State,
    pub(crate) codegen: crate::codegen::State,
    pub(crate) agent: crate::agent::State,
    /// Every route the server answers, filled in by [`build`]: `/api/status`
    /// reports it, and the frontend shows a tab when its route is there.
    routes: std::sync::OnceLock<Vec<String>>,
}

impl Service {
    /// The service around a model already loaded (or created).
    pub fn new(model: Model, model_path: String, seed: i64, workers: usize) -> Service {
        let born_with = model.encoding();
        Service {
            model: Mutex::new(model),
            parked: Mutex::new(Vec::new()),
            born_with,
            job: Mutex::new(None),
            stop: AtomicBool::new(false),
            jobs: AtomicUsize::new(0),
            model_path,
            upload_dir: None,
            checkpoint_dir: None,
            seed,
            workers,
            negative: Mutex::new(None),
            guard: Mutex::new(crate::duo::FilterConfig::default()),
            llm: Default::default(),
            tools: Default::default(),
            checkpoints: Default::default(),
            evolve: Default::default(),
            tutor: Default::default(),
            critic: Default::default(),
            chat: Default::default(),
            codegen: Default::default(),
            agent: Default::default(),
            routes: std::sync::OnceLock::new(),
        }
    }

    /// Reads the `serve` command's flags into every area's state.
    pub fn configure(&mut self, args: &crate::cli::Args) -> Result<(), String> {
        self.llm.configure(args)?;
        self.tools.configure(args)?;
        self.checkpoints.configure(args)?;
        self.evolve.configure(args)?;
        self.tutor.configure(args)?;
        self.critic.configure(args)?;
        self.chat.configure(args)?;
        self.codegen.configure(args)?;
        self.agent.configure(args)?;
        Ok(())
    }

    /// Runs `f` on the running model, holding its lock for as long as `f` runs.
    ///
    /// A long job takes the lock once per step rather than once for the run,
    /// so the frontend's reads get in between.
    pub(crate) fn with_model<T>(&self, f: impl FnOnce(&mut Model) -> T) -> T {
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut model)
    }

    /// Opens a job of `kind` and hands back its id; the caller finishes it.
    pub(crate) fn start_job(&self, kind: &str) -> String {
        let id = format!("{kind}-{}", self.jobs.fetch_add(1, Ordering::Relaxed) + 1);
        let mut job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        *job = Some(Job {
            id: id.clone(),
            kind: kind.to_string(),
            state: "running".to_string(),
            started_at: utc_now(),
            ..Default::default()
        });
        self.stop.store(false, Ordering::Relaxed);
        crate::log_info!(LOG, "{id} started");
        id
    }

    /// Adds one record to the running job, so `/api/job` shows it as progress
    /// before the job is done.
    #[allow(dead_code)]
    pub(crate) fn job_progress(&self, record: Json) {
        let mut job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = job.as_mut() {
            if job.state == "running" {
                job.records.push(record);
            }
        }
    }

    /// The records the running job has reported so far.
    #[allow(dead_code)]
    pub(crate) fn job_records(&self) -> Vec<Json> {
        let job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        job.as_ref().map(|j| j.records.clone()).unwrap_or_default()
    }

    /// Whether `/api/job/stop` (or an area's own stop route) asked the running
    /// job to stop; a job checks between its steps.
    #[allow(dead_code)]
    pub(crate) fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Asks the running job to stop.
    pub(crate) fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Closes the running job with what the work came back with.
    pub(crate) fn finish_job(&self, outcome: Result<Vec<Json>, String>) {
        let mut job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = job.as_mut() else { return };
        job.finished_at = Some(utc_now());
        match outcome {
            Ok(records) => {
                crate::log_info!(LOG, "{} finished, {} record(s)", job.id, records.len());
                job.state = "finished".to_string();
                job.records = records;
            }
            Err(message) => {
                crate::log_error!(LOG, "{} failed: {message}", job.id);
                job.state = "failed".to_string();
                job.error = Some(message);
            }
        }
    }

    /// Saves the active kind to its own file, and says whether it landed.
    pub(crate) fn autosave(&self) -> bool {
        let path = self.model_path_for(self.active_encoding());
        if path.is_empty() {
            return false;
        }
        match self.with_model(|m| m.save(&path)) {
            Ok(()) => {
                crate::log_info!(LOG, "saved {path}");
                true
            }
            Err(why) => {
                crate::log_error!(LOG, "cannot save {path}: {why}");
                false
            }
        }
    }

    /// Refuses a second job while one is running, the way the Python service does.
    pub(crate) fn ensure_idle(&self) -> Result<(), ApiError> {
        let job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        match job.as_ref() {
            Some(job) if job.state == "running" => Err(ApiError::conflict(format!(
                "a {} job is already running; stop it first",
                job.kind
            ))),
            _ => Ok(()),
        }
    }

    pub(crate) fn job_json(&self) -> Json {
        let job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        match job.as_ref() {
            Some(job) => job.to_json(),
            None => Json::Null,
        }
    }

    /// The kinds this server offers: the count model, and the same model over words.
    fn kinds(&self) -> Json {
        Json::Arr(vec![Json::obj([
            ("kind", Json::str("count")),
            ("label", Json::str("Count / reward (Rust)")),
            ("units", Json::str("chars")),
            (
                "description",
                Json::str(
                    "edge weight = the edge's share of its node's traversals, all time and inside a sliding window, \
                     plus rewards - penalties; no learning rate; beam prediction with the top-K and bottom-K \
                     continuations (Rust implementation)",
                ),
            ),
        ])])
    }

    /// The one kind this server runs, with what the running model counts in.
    ///
    /// Words are not a kind - they are an encoding - so the units follow the
    /// encoding while the kind stays `count`.
    fn active_kind(&self) -> (&'static str, &'static str, &'static str) {
        let words = self.with_model(|m| m.encoding().unit == Unit::Words);
        if words {
            (KIND_COUNT.0, KIND_COUNT.1, "words")
        } else {
            KIND_COUNT
        }
    }

    /// The encoding the running model was built with.
    pub(crate) fn active_encoding(&self) -> Encoding {
        self.with_model(|m| m.encoding())
    }

    /// The encoding a bare unit name asks for.
    ///
    /// `{"kind": "word"}` names a unit and not the other two dials, so they
    /// come from the model that unit last had - a parked one, else the one the
    /// server was started on, else the default n and stride.  Which is what
    /// makes selecting `word`, then `count`, then `word` again come back to
    /// the same model rather than to a third one.
    fn encoding_for_unit(&self, unit: Unit) -> Encoding {
        let parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        for (name, _) in parked.iter() {
            if let Ok(enc) = parse_encoding(name) {
                if enc.unit == unit {
                    return enc;
                }
            }
        }
        if self.born_with.unit == unit {
            return self.born_with;
        }
        let default = Encoding::default();
        Encoding {
            unit,
            n: default.n,
            stride: default.stride,
        }
    }

    /// Where a model of `encoding` is saved by default.
    ///
    /// The server's own path for the encoding it was started on, and
    /// `<stem>.<unit><ext>` for the other - so switching to words and saving
    /// does not land on the character model's file.
    pub(crate) fn model_path_for(&self, encoding: Encoding) -> String {
        let path = self.model_path.as_str();
        if path.is_empty() {
            return String::new();
        }
        if encoding == self.born_with {
            return path.to_string();
        }
        let (stem, ext) = split_extension(path);
        let stem = match stem.rsplit_once('.') {
            // .count / .word in the name is the other encoding's; swap it
            Some((head, tail)) if tail == "count" || tail == "word" => head,
            _ => stem,
        };
        let tag = if encoding.unit == Unit::Words { "word" } else { "count" };
        format!("{stem}.{tag}{ext}")
    }

    /// The encodings this server has a model for right now, active one first.
    fn in_memory(&self) -> Vec<String> {
        let mut out = vec![self.active_encoding().to_string()];
        let parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        out.extend(parked.iter().map(|(enc, _)| enc.clone()));
        out
    }

    /// Makes `encoding` the running model and says where it came from.
    ///
    /// An encoding is fixed for a model's life, so *changing* it means a
    /// different model.  The one that was running is parked with its unsaved
    /// work rather than dropped, and the new one is whichever comes first of
    /// the parked model of that encoding, its own file, and a fresh one -
    /// which is what lets the frontend switch to words and back without losing
    /// a training run.
    fn select_encoding(&self, encoding: Encoding, seed: i64) -> Result<&'static str, ApiError> {
        self.ensure_idle()?;
        let active = self.active_encoding();
        if encoding == active {
            return Ok("active");
        }
        let key = encoding.to_string();
        let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        let (origin, mut wanted) = match parked.iter().position(|(name, _)| *name == key) {
            Some(at) => ("memory", parked.remove(at).1),
            None => {
                // the default path is a convention, not a claim: a file there
                // built under another encoding is simply not this one's, so it
                // is left alone rather than refused
                let path = self.model_path_for(encoding);
                let on_disk = if path.is_empty() || !std::path::Path::new(&path).is_file() {
                    None
                } else {
                    Model::load(&path).ok().filter(|m| m.encoding() == encoding)
                };
                if let Some(found) = on_disk {
                    ("file", found)
                } else {
                    (
                        "new",
                        Model::new(
                            seed,
                            GraphOptions {
                                encoding,
                                ..Default::default()
                            },
                        )?,
                    )
                }
            }
        };
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        wanted.workers = model.workers;
        wanted.g.workers = model.workers;
        let previous = std::mem::replace(&mut *model, wanted);
        crate::log_info!(LOG, "encoding {active} -> {encoding} ({origin}); {active} parked");
        parked.push((active.to_string(), previous));
        Ok(origin)
    }
}

/// The one kind this server runs: the id, the label and what it counts in by
/// default.  The units follow the running model's encoding.
const KIND_COUNT: (&str, &str, &str) = ("count", "Count / reward (Rust)", "chars");

/// `("model.count", ".json")`, with `.json.gz` kept whole.
fn split_extension(path: &str) -> (&str, &str) {
    let Some(dot) = path.rfind('.') else {
        return (path, "");
    };
    if &path[dot..] == ".gz" {
        if let Some(inner) = path[..dot].rfind('.') {
            return (&path[..inner], &path[inner..]);
        }
    }
    (&path[..dot], &path[dot..])
}

impl Service {
    /// The texts of a request field, the way the Python service reads them
    /// (`api._texts_and_files`): `<name>` as a list, `<name>_text` as one text
    /// per line, and `<name>_files` as the uploads to read - all three added
    /// together, because a client may give any combination.
    ///
    /// `train` names its three `texts` / `text` / `files`, which is the one
    /// place the pattern is spelt differently; `one` and `files` say so.
    pub(crate) fn texts_of(&self, r: &Request, list: &str, one: &str, files: &str) -> Result<Vec<String>, ApiError> {
        let mut texts = r.texts(list);
        texts.extend(r.texts(one));
        for name in r.texts(files) {
            texts.extend(self.upload_texts(&name, r.flag("whole_file", false))?);
        }
        Ok(texts)
    }

    /// One upload as training texts: the whole file as a single text, or a text
    /// per non-empty line.
    pub(crate) fn upload_texts(&self, name: &str, whole: bool) -> Result<Vec<String>, ApiError> {
        let dir = self
            .upload_dir
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("no upload directory is configured"))?;
        let path = safe_upload(dir, name)?;
        let content = std::fs::read_to_string(&path)
            .map_err(|err| ApiError::bad_request(format!("cannot read upload {name:?}: {err}")))?;
        if whole {
            return Ok(vec![content]);
        }
        Ok(split_texts(&content, "lines", 0))
    }
}

/// An upload's path, with a name that tries to leave the directory refused -
/// the same plain-name rule the upload and delete routes write by.
pub(crate) fn safe_upload(dir: &str, name: &str) -> Result<std::path::PathBuf, ApiError> {
    if !plain_name(name) {
        return Err(ApiError::bad_request(format!("{name:?} is not an upload name")));
    }
    Ok(std::path::Path::new(dir).join(name))
}

/// A bare file name: no directory, no climbing out of one.
fn plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && !name.contains("..")
}

// -- the routes ---------------------------------------------------------------------------------

fn health(svc: &Arc<Service>, _r: &Request) -> Answer {
    let _ = svc;
    Ok(Json::obj([
        ("status", Json::str("ok")),
        ("version", Json::str(env!("CARGO_PKG_VERSION"))),
        ("engine", Json::str(ENGINE)),
    ]))
}

fn status(svc: &Arc<Service>, _r: &Request) -> Answer {
    let (kind, label, units) = svc.active_kind();
    let mut doc = svc.with_model(|m| stats(m));
    let pairs = match &mut doc {
        Json::Obj(pairs) => pairs,
        _ => return Ok(doc),
    };
    pairs.push(("kind".to_string(), Json::str(kind)));
    pairs.push(("model_label".to_string(), Json::str(label)));
    pairs.push(("units".to_string(), Json::str(units)));
    pairs.push(("kinds".to_string(), svc.kinds()));
    pairs.push(("job".to_string(), svc.job_json()));
    pairs.push((
        "routes".to_string(),
        Json::strs(svc.routes.get().cloned().unwrap_or_default()),
    ));
    pairs.push((
        "backends".to_string(),
        Json::obj([
            ("python", Json::Bool(false)),
            ("torch", Json::Bool(false)),
            ("cuda", Json::Bool(false)),
            ("mps", Json::Bool(false)),
            ("rust", Json::Bool(true)),
            ("default", Json::str(ENGINE)),
        ]),
    ));
    pairs.push(("model_path".to_string(), Json::str(svc.model_path.clone())));
    pairs.push((
        "checkpoint_dir".to_string(),
        match &svc.checkpoint_dir {
            Some(dir) => Json::str(dir.clone()),
            None => Json::Null,
        },
    ));
    pairs.push((
        "upload_dir".to_string(),
        match &svc.upload_dir {
            Some(dir) => Json::str(dir.clone()),
            None => Json::Null,
        },
    ));
    // the LLM providers' defaults, which the Ollama tab reads its own from
    pairs.extend(svc.llm.status_fields());
    pairs.push(("engine".to_string(), Json::str(ENGINE)));
    pairs.push(("workers".to_string(), Json::Int(svc.workers as i64)));
    pairs.push(("counting".to_string(), Json::str("exact")));
    Ok(doc)
}

fn model_info(svc: &Arc<Service>, _r: &Request) -> Answer {
    let (kind, label, units) = svc.active_kind();
    let weights = svc.with_model(|m| weight_config(&m.g));
    let enc = svc.active_encoding();
    let chars = Encoding::default();
    let words = Encoding::new(Unit::Words, enc.n, enc.stride).expect("a valid word encoding");
    Ok(Json::obj([
        ("kind", Json::str(kind)),
        ("label", Json::str(label)),
        ("units", Json::str(units)),
        ("kinds", svc.kinds()),
        ("encoding", Json::str(enc.to_string())),
        ("unit", Json::str(enc.unit.name())),
        ("ngram", Json::Int(enc.n as i64)),
        ("stride", Json::Int(enc.stride as i64)),
        ("model_path", Json::str(svc.model_path_for(enc))),
        (
            "paths",
            Json::obj([
                ("count", Json::str(svc.model_path_for(chars))),
                ("word", Json::str(svc.model_path_for(words))),
            ]),
        ),
        ("in_memory", Json::strs(svc.in_memory())),
        ("weights", weights),
        ("engine", Json::str(ENGINE)),
    ]))
}

/// `POST /api/model/select`: the kind, and - the dial being what words are now
/// - the encoding.
///
/// Words are not a kind, so `{"kind": "word"}` is read as a request for a word
/// *encoding* and answered by switching to one; the model that was running is
/// parked with its unsaved work. `{"encoding": "word:2:1"}` says it directly.
fn model_select(svc: &Arc<Service>, r: &Request) -> Answer {
    let spec = r.text("encoding", "");
    let wanted = if !spec.is_empty() {
        parse_encoding(&spec)?
    } else {
        let kind = r.text("kind", KIND_COUNT.0).to_ascii_lowercase();
        if kind.is_empty() {
            return Err(ApiError::bad_request("'kind' must not be empty"));
        }
        match kind.as_str() {
            "count" | "char" | "chars" => svc.encoding_for_unit(Unit::Chars),
            "word" | "words" => svc.encoding_for_unit(Unit::Words),
            other => {
                return Err(ApiError::bad_request(format!(
                    "the Rust server runs the count / reward model only, over any encoding \
                     (kind {other:?} is served by the Python server)"
                )))
            }
        }
    };
    let origin = svc.select_encoding(wanted, svc.seed)?;
    let mut doc = model_info(svc, r)?;
    if let Json::Obj(pairs) = &mut doc {
        pairs.push(("origin".to_string(), Json::str(origin)));
        pairs.push(("stats".to_string(), svc.with_model(|m| stats(m))));
    }
    Ok(doc)
}

fn model_weights(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let mut options: Vec<(String, f64)> = Vec::new();
    if let Json::Obj(pairs) = &r.body {
        for (key, value) in pairs {
            if let Some(number) = value.as_f64() {
                options.push((key.clone(), number));
            }
        }
    }
    let out = svc.with_model(|m| -> Result<Json, String> {
        for (key, value) in &options {
            configure_weight(&mut m.g, key, *value)?;
        }
        m.g.recompute_weights();
        Ok(Json::obj([("weights", weight_config(&m.g)), ("stats", stats(m))]))
    })?;
    Ok(out)
}

/// The options every search endpoint reads: the traversal and its two scales.
fn traversal_of(r: &Request) -> Result<(String, f64, f64), ApiError> {
    let name = resolve_traversal(&r.text("traversal", DEFAULT_TRAVERSAL))?;
    Ok((
        name.to_string(),
        r.number("penalty_scale", 1.0)?,
        r.number("merit_scale", 1.0)?,
    ))
}

fn path_json(result: &PathResult) -> Json {
    let mut pairs = vec![
        ("text".to_string(), Json::str(result.text.clone())),
        ("continuation".to_string(), Json::str(result.text.clone())),
        ("full_text".to_string(), Json::str(result.full_text.clone())),
        ("cost".to_string(), Json::Num(result.cost)),
        ("probability".to_string(), Json::Num(result.probability())),
        ("path".to_string(), Json::strs(result.labels.clone())),
        (
            "node_ids".to_string(),
            Json::ints(result.node_ids.iter().map(|&n| n as i64)),
        ),
        ("step_costs".to_string(), Json::nums(result.step_costs.clone())),
        ("reached_end".to_string(), Json::Bool(result.reached_end)),
    ];
    if result.punish != 0.0 {
        pairs.push(("punish".to_string(), Json::Num(result.punish)));
    }
    Json::Obj(pairs)
}

fn predict(svc: &Arc<Service>, r: &Request) -> Answer {
    let (traversal, penalty_scale, merit_scale) = traversal_of(r)?;
    let max_length = r.body.at("max_length").as_i64();
    let opts = PredictOptions {
        length: r.usize("length", 20)?,
        mode: r.text("mode", "beam"),
        k: r.usize("k", 5)?,
        beam: r.usize("beam", 0)?,
        step_penalty: r.number("step_penalty", 0.0)?,
        temperature: r.number("temperature", 1.0)?,
        to_end: r.flag("to_end", false),
        max_length: max_length.filter(|v| *v >= 0).map(|v| v as usize),
        traversal,
        penalty_scale,
        merit_scale,
    };
    let prefix = r.text("prefix", "");
    let (kind, ..) = svc.active_kind();
    let found = svc.with_model(|m| m.predict(&prefix, &opts))?;
    Ok(Json::obj([
        ("prefix", Json::str(prefix)),
        ("kind", Json::str(kind)),
        ("continuation", Json::str(found.best.text.clone())),
        ("full_text", Json::str(found.best.full_text.clone())),
        ("cost", Json::Num(found.best.cost)),
        ("probability", Json::Num(found.best.probability())),
        ("step_costs", Json::nums(found.best.step_costs.clone())),
        ("path", Json::strs(found.best.labels.clone())),
        ("node_ids", Json::ints(found.best.node_ids.iter().map(|&n| n as i64))),
        ("expanded", Json::Int(found.expanded as i64)),
        ("reached_end", Json::Bool(found.best.reached_end)),
        ("mode", Json::str(found.mode.clone())),
        ("traversal", Json::str(found.traversal.clone())),
        ("k", Json::Int(found.k as i64)),
        ("beam", Json::Int(found.beam as i64)),
        ("top", Json::Arr(found.top.iter().map(path_json).collect())),
        ("bottom", Json::Arr(found.bottom.iter().map(path_json).collect())),
        ("guard", Json::Null),
    ]))
}

fn generate(svc: &Arc<Service>, r: &Request) -> Answer {
    let (traversal, penalty_scale, merit_scale) = traversal_of(r)?;
    let seeded = r.body.get("seed").and_then(|v| v.as_i64());
    let opts = GenerateOptions {
        max_length: r.usize("max_length", 60)?,
        mode: r.text("mode", "sample"),
        temperature: r.number("temperature", 1.0)?,
        count: r.usize("count", 1)?,
        seed: seeded,
        prefix: r.text("prefix", ""),
        step_penalty: r.number("step_penalty", 0.0)?,
        beam: r.usize("beam", 0)?,
        traversal,
        penalty_scale,
        merit_scale,
    };
    let samples = svc.with_model(|m| m.generate(&opts))?;
    Ok(Json::obj([
        ("samples", Json::Arr(samples.iter().map(path_json).collect())),
        ("guard", Json::Null),
    ]))
}

fn score(svc: &Arc<Service>, r: &Request) -> Answer {
    let texts = match r.body.get("text").and_then(|v| v.as_str()) {
        // one text is scored whole, newlines and all: `text` is the text here,
        // not a list of them (the Python server scores it the same way)
        Some(one) => vec![one.to_string()],
        None => r.texts("texts"),
    };
    if texts.is_empty() {
        return Err(ApiError::bad_request("give 'text' or 'texts' to score"));
    }
    let results: Vec<Json> = svc.with_model(|m| {
        texts
            .iter()
            .map(|text| {
                let s = m.score(text);
                Json::obj([
                    ("text", Json::str(text.clone())),
                    ("log_prob", Json::Num(s.log_prob)),
                    ("per_char", Json::Num(s.per_char)),
                    ("chars", Json::Int(s.chars as i64)),
                    ("transitions", Json::Int(s.transitions as i64)),
                    ("unknown_transitions", Json::Int(s.unknown_transitions as i64)),
                ])
            })
            .collect()
    });
    let n = results.len().max(1) as f64;
    let mean = |key: &str| -> f64 { results.iter().filter_map(|r| r.at(key).as_f64()).sum::<f64>() / n };
    let (log_prob, per_char) = (mean("log_prob"), mean("per_char"));
    let first = results.first().cloned().unwrap_or(Json::Null);
    let mut doc = match first {
        Json::Obj(pairs) => pairs,
        _ => Vec::new(),
    };
    doc.push(("results".to_string(), Json::Arr(results)));
    doc.push(("count".to_string(), Json::Int(n as i64)));
    doc.push(("mean_log_prob".to_string(), Json::Num(log_prob)));
    doc.push(("mean_per_char".to_string(), Json::Num(per_char)));
    Ok(Json::Obj(doc))
}

fn feedback(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let good = svc.texts_of(r, "good", "good_text", "good_files")?;
    let bad = svc.texts_of(r, "bad", "bad_text", "bad_files")?;
    if good.is_empty() && bad.is_empty() {
        return Err(ApiError::bad_request(
            "nothing to learn from: give 'good' (thumbs up) and / or 'bad' (thumbs down)",
        ));
    }
    let strength = r.number("strength", 1.0)?;
    let epochs = r.usize("epochs", 1)?;
    let out = svc.with_model(|m| -> Result<Json, String> {
        let mut records: Vec<Json> = Vec::new();
        if !bad.is_empty() {
            records.extend(m.punish(&bad, epochs, strength)?.iter().map(|r| r.to_json()));
        }
        if !good.is_empty() {
            records.extend(m.reward(&good, epochs, strength)?.iter().map(|r| r.to_json()));
        }
        Ok(Json::obj([
            ("good", Json::Int(good.len() as i64)),
            ("bad", Json::Int(bad.len() as i64)),
            ("strength", Json::Num(strength)),
            ("records", Json::Arr(records)),
            ("stats", stats(m)),
        ]))
    })?;
    Ok(out)
}

fn two_nrl(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let bad = svc.texts_of(r, "bad", "bad_text", "bad_files")?;
    let good = svc.texts_of(r, "good", "good_text", "good_files")?;
    if bad.is_empty() && good.is_empty() {
        return Err(ApiError::bad_request("2NRL needs 'bad' and / or 'good' texts"));
    }
    let strength = r.number("strength", 1.0)?;
    let neg_epochs = r.usize("neg_epochs", 2)?;
    let pos_epochs = r.usize("pos_epochs", 3)?;
    svc.start_job("two_nrl");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| m.two_nrl(&bad, &good, neg_epochs, pos_epochs, strength));
        worker.finish_job(outcome.map(|(negative, positive)| {
            // one list, the negative phase first, because a job's records are a
            // list of epochs and 2NRL is two phases of them
            negative.iter().chain(positive.iter()).map(|r| r.to_json()).collect()
        }));
    });
    Ok(accepted(Json::obj([("job", svc.job_json())])))
}

fn invert(svc: &Arc<Service>, _r: &Request) -> Answer {
    svc.ensure_idle()?;
    Ok(svc.with_model(|m| {
        m.invert();
        Json::obj([("inverted", Json::Bool(m.g.inverted)), ("stats", stats(m))])
    }))
}

fn compress(svc: &Arc<Service>, _r: &Request) -> Answer {
    svc.ensure_idle()?;
    Ok(svc.with_model(|m| {
        let merges = m.g.compress();
        Json::obj([("merges", Json::Int(merges as i64)), ("stats", stats(m))])
    }))
}

fn train(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let texts = svc.texts_of(r, "texts", "text", "files")?;
    if texts.is_empty() {
        return Err(ApiError::bad_request(
            "missing field 'texts' (list of strings), 'text' (string, one text per line) \
             or 'files' (list of upload names)",
        ));
    }
    let opts = TrainOptions {
        epochs: r.usize("epochs", 5)?,
        auto_compress: !r.flag("no_compress", false),
        chunk_size: r.usize("chunk", 0)?,
        phase: None,
    };
    svc.start_job("train");
    // the run holds the model for as long as it takes, which is what makes a
    // second request wait; the answer goes out now, and the frontend follows the
    // run on /api/job - the same 202 the Python and Go servers answer with
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| m.train(&texts, &opts));
        if outcome.is_ok() {
            worker.autosave();
        }
        worker.finish_job(outcome.map(|records| records.iter().map(|r| r.to_json()).collect()));
    });
    Ok(accepted(Json::obj([("job", svc.job_json())])))
}

fn job(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(svc.job_json())
}

fn job_stop(svc: &Arc<Service>, _r: &Request) -> Answer {
    svc.request_stop();
    Ok(svc.job_json())
}

fn history(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(svc.with_model(|m| Json::obj([("history", Json::Arr(m.history.iter().map(|r| r.to_json()).collect()))])))
}

fn graph_view(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 150)?;
    Ok(svc.with_model(|m| {
        m.g.prepare();
        let mut real: Vec<usize> = (FIRST..m.g.num_node_ids()).filter(|&i| m.g.is_alive(i)).collect();
        real.sort_by(|&a, &b| m.g.node_count(b).value.cmp(&m.g.node_count(a).value).then(a.cmp(&b)));
        real.truncate(limit);
        real.sort_unstable();
        let mut ids = vec![START, END];
        ids.extend(real);
        let nodes: Vec<Json> = ids
            .iter()
            .map(|&i| {
                Json::obj([
                    ("id", Json::Int(i as i64)),
                    ("label", Json::str(m.g.label(i).to_string())),
                    ("count", Json::Int(m.g.node_count(i).value)),
                    ("count_resets", Json::Int(m.g.node_count(i).resets)),
                    ("activation", Json::Num(1.0)),
                ])
            })
            .collect();
        let chosen: Vec<usize> = ids.clone();
        let mut edges: Vec<Json> = Vec::new();
        for &p in &ids {
            for cc in m.g.child_costs_from(p, None) {
                if !chosen.contains(&cc.child) {
                    continue;
                }
                edges.push(Json::obj([
                    ("src", Json::Int(p as i64)),
                    ("dst", Json::Int(cc.child as i64)),
                    ("edge", Json::Int(cc.edge as i64)),
                    ("cost", Json::Num(cc.cost)),
                    ("weight", Json::Num(m.g.edge_w[cc.edge])),
                    ("count", Json::Int(m.g.edge_traversals(cc.edge).value)),
                    ("reward", Json::Num(m.g.edge_reward[cc.edge])),
                    ("punish", Json::Num(cc.punish)),
                ]));
            }
        }
        Json::obj([
            ("nodes", Json::Arr(nodes)),
            ("edges", Json::Arr(edges)),
            ("limit", Json::Int(limit as i64)),
            ("total_nodes", Json::Int(m.g.num_nodes() as i64)),
        ])
    }))
}

fn paths(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 50)?;
    Ok(svc.with_model(|m| {
        let totals = m.g.path_totals();
        Json::obj([
            ("paths", path_rows(&m.g, limit, None)),
            (
                "totals",
                Json::obj([
                    ("contexts", Json::Int(totals.contexts as i64)),
                    ("judged", Json::Int(totals.judged as i64)),
                    ("seen", Json::Int(totals.seen)),
                    ("correct", Json::Int(totals.correct)),
                    ("incorrect", Json::Int(totals.incorrect)),
                ]),
            ),
            ("limit", Json::Int(limit as i64)),
            ("path_scale", Json::Num(m.g.path_scale)),
        ])
    }))
}

fn nodes(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 20)?;
    let wanted = r.query("node").map(|s| s.to_string());
    let out = svc.with_model(|m| -> Result<Json, ApiError> {
        m.g.prepare();
        let node = match &wanted {
            None => None,
            Some(text) => {
                // a label is text on every encoding: the node itself, or a gram it holds
                let found = (0..m.g.num_node_ids())
                    .find(|&i| m.g.is_alive(i) && m.g.label(i) == text)
                    .or_else(|| m.g.lookup(text).map(|loc| loc.node));
                Some(found.ok_or_else(|| {
                    ApiError::not_found(format!(
                        "no node labelled {text:?}: give a node label, or one of its grams"
                    ))
                })?)
            }
        };
        let totals = m.g.path_totals();
        Ok(Json::obj([
            ("nodes", node_rows(&m.g, limit, node)),
            ("limit", Json::Int(limit as i64)),
            (
                "node",
                match &wanted {
                    Some(text) => Json::str(text.clone()),
                    None => Json::Null,
                },
            ),
            ("total_nodes", Json::Int(m.g.num_nodes() as i64)),
            (
                "totals",
                Json::obj([
                    ("contexts", Json::Int(totals.contexts as i64)),
                    ("judged", Json::Int(totals.judged as i64)),
                    ("seen", Json::Int(totals.seen)),
                    ("correct", Json::Int(totals.correct)),
                    ("incorrect", Json::Int(totals.incorrect)),
                ]),
            ),
        ]))
    })?;
    Ok(out)
}

fn words(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 50)?;
    let out = svc.with_model(|m| -> Result<Json, ApiError> {
        let enc = m.encoding();
        if enc.unit != Unit::Words {
            return Err(ApiError::bad_request(format!(
                "this model counts in {}, so it has no words to list; a word alphabet needs a word encoding \
                 (--encoding word:{}:{})",
                enc.units_name(),
                enc.n,
                enc.stride
            )));
        }
        let all = m.top_words(0);
        let vocabulary = all.len();
        let rows = if limit > 0 {
            &all[..limit.min(all.len())]
        } else {
            &all[..]
        };
        Ok(Json::obj([
            ("words", Json::Arr(rows.iter().map(|row| row.to_json()).collect())),
            ("limit", Json::Int(limit as i64)),
            ("vocabulary", Json::Int(vocabulary as i64)),
            ("units", Json::str(m.units())),
            ("encoding", Json::str(enc.to_string())),
        ]))
    })?;
    Ok(out)
}

fn encoding(svc: &Arc<Service>, _r: &Request) -> Answer {
    let enc = svc.active_encoding();
    Ok(Json::obj([
        ("encoding", Json::str(enc.to_string())),
        ("unit", Json::str(enc.unit.name())),
        ("units", Json::str(enc.units_name())),
        ("window", Json::Int(enc.n as i64)),
        ("ngram", Json::Int(enc.n as i64)),
        ("stride", Json::Int(enc.stride as i64)),
        ("overlap", Json::Int(enc.overlap() as i64)),
        ("sliding", Json::Bool(enc.sliding())),
        ("start_label", Json::str(START_LABEL)),
        ("end_label", Json::str(END_LABEL)),
        ("back_label", Json::str(BACK_LABEL)),
        // the dial is chosen when a model is created and fixed for its life:
        // changing it means a different model, which POST /api/reset makes
        ("configurable", Json::Bool(true)),
        ("fixed_for_life", Json::Bool(true)),
        (
            "note",
            Json::str(format!(
                "Text goes in as {} of {} {}, stride {}, and comes back out of the (possibly compressed) node \
                 labels along a path. The encoding is part of the model - fixed when it is created - so changing \
                 it means a new model (POST /api/reset with an encoding).",
                if enc.sliding() { "overlapping windows" } else { "groups" },
                enc.n,
                enc.units_name(),
                enc.stride,
            )),
        ),
    ]))
}

fn encoding_preview(svc: &Arc<Service>, r: &Request) -> Answer {
    let text = r.text("text", "");
    let (kind, _, units) = svc.active_kind();
    Ok(svc.with_model(|m| {
        let enc = m.encoding();
        let grams = enc.encode(&text);
        let decoded = enc.decode_grams(grams.iter().map(|g| g.as_str()));
        let unknown: Vec<String> = grams.iter().filter(|g| m.g.lookup(g).is_none()).cloned().collect();
        let walked = if grams.is_empty() { None } else { m.g.node_path(&grams) };
        let path = match walked {
            None => Json::obj([
                ("known", Json::Bool(false)),
                (
                    "reason",
                    Json::str(if grams.is_empty() {
                        format!("the text is shorter than one gram ({} {})", enc.n, enc.units_name())
                    } else if !unknown.is_empty() {
                        format!("{} of the {} grams have never been seen", unknown.len(), grams.len())
                    } else {
                        "every gram is known, but the structure cannot walk the whole text from START to END as it \
                     stands"
                            .to_string()
                    }),
                ),
                ("labels", Json::strs(Vec::<String>::new())),
                ("node_ids", Json::ints(Vec::<i64>::new())),
                ("decoded", Json::str("")),
                ("nodes", Json::Int(0)),
                ("compressed", Json::Int(0)),
            ]),
            Some(walked) => {
                let labels: Vec<String> = walked.iter().map(|&n| m.g.label(n).to_string()).collect();
                let real: Vec<String> = walked
                    .iter()
                    .filter(|&&n| n != START && n != END)
                    .map(|&n| m.g.label(n).to_string())
                    .collect();
                let compressed = real.iter().filter(|l| enc.len(l) > enc.n).count();
                let borrowed: Vec<&str> = real.iter().map(|s| s.as_str()).collect();
                Json::obj([
                    ("known", Json::Bool(true)),
                    ("reason", Json::Null),
                    ("labels", Json::strs(labels)),
                    ("node_ids", Json::ints(walked.iter().map(|&n| n as i64))),
                    ("decoded", Json::str(enc.decode_path(&borrowed, 0, true))),
                    ("nodes", Json::Int(real.len() as i64)),
                    ("compressed", Json::Int(compressed as i64)),
                ])
            }
        };
        Json::obj([
            ("encoding", Json::str(enc.to_string())),
            ("window", Json::Int(enc.n as i64)),
            ("ngram", Json::Int(enc.n as i64)),
            ("stride", Json::Int(enc.stride as i64)),
            ("overlap", Json::Int(enc.overlap() as i64)),
            ("configurable", Json::Bool(true)),
            ("units", Json::str(units)),
            ("kind", Json::str(kind)),
            ("text", Json::str(text.clone())),
            ("chars", Json::Int(enc.len(&text) as i64)),
            ("windows", Json::strs(grams.clone())),
            ("grams", Json::strs(grams.clone())),
            ("count", Json::Int(grams.len() as i64)),
            ("decoded", Json::str(decoded.clone())),
            ("round_trip", Json::Bool(decoded == enc.normalize(&text))),
            ("unknown_windows", Json::strs(unknown.clone())),
            ("unknown_grams", Json::strs(unknown)),
            ("path", path),
        ])
    }))
}

fn save(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    // the active kind's own file, so saving a word model never lands on the
    // count model's - the rule the Python service saves by
    let path = r.text("path", &svc.model_path_for(svc.active_encoding()));
    svc.with_model(|m| m.save(&path))?;
    let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
    Ok(Json::obj([
        ("saved", Json::str(path)),
        ("bytes", Json::Int(bytes as i64)),
    ]))
}

fn load(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let path = r.text("path", &svc.model_path_for(svc.active_encoding()));
    let loaded = Model::load(&path)?;
    svc.with_model(|m| {
        let workers = m.workers;
        *m = loaded;
        m.workers = workers;
        m.g.workers = workers;
    });
    Ok(Json::obj([
        ("loaded", Json::str(path)),
        ("stats", svc.with_model(|m| stats(m))),
    ]))
}

/// `POST /api/reset`: a fresh model, optionally under a different encoding.
///
/// The encoding is fixed for a model's life, so this is where it is chosen -
/// the same place the Python and Go servers choose it.  `{"encoding":
/// "word:3:1"}`, or the three dials separately as `unit` / `ngram` / `stride`.
fn reset(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let seed = r.body.at("seed").as_i64().unwrap_or(svc.seed);
    let active = svc.active_encoding();
    let spec = r.text("encoding", "");
    let mut encoding = if spec.is_empty() {
        active
    } else {
        parse_encoding(&spec)?
    };
    if let Some(name) = r.body.get("unit").and_then(|v| v.as_str()) {
        encoding.unit = Unit::parse(name)
            .ok_or_else(|| ApiError::bad_request(format!("unit must be char or word, got {name:?}")))?;
    }
    if let Some(n) = r.body.get("ngram").and_then(|v| v.as_i64()) {
        encoding.n = n.max(0) as usize;
    }
    if let Some(stride) = r.body.get("stride").and_then(|v| v.as_i64()) {
        encoding.stride = stride.max(0) as usize;
    }
    encoding.validate()?;
    let fresh = Model::new(
        seed,
        GraphOptions {
            encoding,
            ..Default::default()
        },
    )?;
    svc.with_model(|m| {
        let workers = m.workers;
        *m = fresh;
        m.workers = workers;
        m.g.workers = workers;
    });
    let (kind, _, units) = svc.active_kind();
    Ok(Json::obj([
        ("reset", Json::Bool(true)),
        ("kind", Json::str(kind)),
        ("units", Json::str(units)),
        ("encoding", Json::str(encoding.to_string())),
        ("stats", svc.with_model(|m| stats(m))),
    ]))
}

fn uploads(svc: &Arc<Service>, _r: &Request) -> Answer {
    let Some(dir) = &svc.upload_dir else {
        return Ok(Json::obj([("uploads", Json::Arr(Vec::new()))]));
    };
    let mut rows: Vec<Json> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let lines = std::fs::read_to_string(&path)
                .map(|text| text.lines().filter(|l| !l.trim().is_empty()).count())
                .unwrap_or(0);
            rows.push(Json::obj([
                ("name", Json::str(name)),
                ("bytes", Json::Int(bytes as i64)),
                ("lines", Json::Int(lines as i64)),
            ]));
        }
    }
    Ok(Json::obj([("uploads", Json::Arr(rows))]))
}

fn upload(svc: &Arc<Service>, r: &Request) -> Answer {
    let Some(dir) = &svc.upload_dir else {
        return Err(ApiError::bad_request("no upload directory is configured"));
    };
    let name = r.text("name", "");
    let content = r.text("content", "");
    if !plain_name(&name) {
        return Err(ApiError::bad_request("give a plain file name for the upload"));
    }
    std::fs::create_dir_all(dir).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let path = std::path::Path::new(dir).join(&name);
    std::fs::write(&path, content.as_bytes()).map_err(|err| ApiError::bad_request(err.to_string()))?;
    uploads(svc, r)
}

fn upload_delete(svc: &Arc<Service>, r: &Request) -> Answer {
    let Some(dir) = &svc.upload_dir else {
        return Err(ApiError::bad_request("no upload directory is configured"));
    };
    let name = r.text("name", "");
    if !plain_name(&name) {
        return Err(ApiError::bad_request("give a plain file name"));
    }
    let _ = std::fs::remove_file(std::path::Path::new(dir).join(&name));
    uploads(svc, r)
}

fn schedule(svc: &Arc<Service>, _r: &Request) -> Answer {
    let _ = svc;
    // the count model has no learning rate to schedule, and says so rather than 404ing
    Ok(Json::obj([
        ("schedules", Json::Arr(Vec::new())),
        (
            "note",
            Json::str("the count / reward model learns by counting: there is no learning rate to schedule"),
        ),
    ]))
}

fn traversals(svc: &Arc<Service>, _r: &Request) -> Answer {
    let _ = svc;
    Ok(Json::obj([
        ("traversals", Json::strs(TRAVERSALS)),
        ("default", Json::str(DEFAULT_TRAVERSAL)),
    ]))
}

/// Every route this server answers, in the order the frontend meets them.
pub fn build(service: Arc<Service>, frontend: Option<String>) -> Server<Service> {
    let mut server = Server::new(service);
    server.route("GET", "/api/health", health);
    server.route("GET", "/api/status", status);
    server.route("GET", "/api/model", model_info);
    server.route("POST", "/api/model/select", model_select);
    server.route("POST", "/api/model/weights", model_weights);
    server.route("POST", "/api/predict", predict);
    server.route("POST", "/api/generate", generate);
    server.route("POST", "/api/score", score);
    server.route("POST", "/api/feedback", feedback);
    server.route("POST", "/api/2nrl", two_nrl);
    server.route("POST", "/api/invert", invert);
    server.route("POST", "/api/compress", compress);
    server.route("POST", "/api/train", train);
    server.route("GET", "/api/job", job);
    server.route("POST", "/api/job/stop", job_stop);
    server.route("GET", "/api/history", history);
    server.route("GET", "/api/graph", graph_view);
    server.route("GET", "/api/paths", paths);
    server.route("GET", "/api/nodes", nodes);
    server.route("GET", "/api/words", words);
    server.route("GET", "/api/encoding", encoding);
    server.route("POST", "/api/encoding/preview", encoding_preview);
    server.route("POST", "/api/save", save);
    server.route("POST", "/api/load", load);
    server.route("POST", "/api/reset", reset);
    server.route("GET", "/api/uploads", uploads);
    server.route("POST", "/api/uploads", upload);
    server.route("POST", "/api/uploads/delete", upload_delete);
    server.route("GET", "/api/schedule", schedule);
    server.route("GET", "/api/traversals", traversals);
    // every other area answers its own routes
    crate::duo::routes(&mut server);
    crate::critic::routes(&mut server);
    crate::dialogue::routes(&mut server);
    crate::checkpoint::routes(&mut server);
    crate::gan::routes(&mut server);
    crate::ollama::routes(&mut server);
    crate::chatgpt::routes(&mut server);
    crate::tools::routes(&mut server);
    crate::vision::routes(&mut server);
    crate::speech::routes(&mut server);
    crate::tutor::routes(&mut server);
    crate::chat::routes(&mut server);
    crate::codegen::routes(&mut server);
    crate::agent::routes(&mut server);
    let _ = server.state().routes.set(server.routes());
    if let Some(dir) = frontend {
        server.frontend(dir);
    }
    server
}
