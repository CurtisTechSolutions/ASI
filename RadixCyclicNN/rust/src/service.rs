//! The HTTP API of the Rust port: the same JSON contract as the Python and Go
//! servers, so `frontend/dist` runs against it unchanged.
//!
//! The model's own routes are here; every other area of the crate answers its
//! own (`crate::duo::routes`, `crate::tutor::routes`, ...), wired in [`build`].
//! `/api/status` lists every route the server has, and the frontend shows a
//! tab when the route it needs is in that list (`frontend/src/App.jsx`).
//!
//! It runs every kind Python's service runs - `radix`, `count`, `negative`
//! and `resonant` ([`crate::kinds`]) - and a word model is a count model over a
//! word encoding.  Selecting another kind parks the running model with its
//! unsaved work, as Python's `ModelService` parks it; the negative network,
//! when it is not the running model, is the one the guard filters with.
//!
//! One model, one lock.  A training run holds it for as long as it takes, so a
//! second request waits rather than reading a half-built graph; the job it
//! reports is the one the frontend polls.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::clock::utc_now;
use crate::encoding::{parse_encoding, Encoding, Unit, BACK_LABEL, END_LABEL, START_LABEL, THINK_LABEL};
use crate::graph::{END, START};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::kinds;
use crate::model::{GenerateOptions, Model, PredictOptions, SearchTuning};
use crate::penalty::{resolve_traversal, DEFAULT_TRAVERSAL, TRAVERSALS};
use crate::radix::{Feedback, TrainConfig};
use crate::report::{node_rows, path_rows, split_texts, stats};
use crate::search::{PathResult, SamplingFilter};
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
    /// The kind this server was started on, whose file `model_path` is.
    born_kind: &'static str,
    /// Whether the negative network is the running model (selected as a
    /// kind).  The negative routes then reach it through the model's lock,
    /// and the guard stands down: a network cannot filter itself.
    pub(crate) negative_active: AtomicBool,
    job: Mutex<Option<Job>>,
    stop: AtomicBool,
    jobs: AtomicUsize,
    pub model_path: String,
    pub upload_dir: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub seed: i64,
    pub workers: usize,
    /// The negative network this server guards its output paths with, loaded
    /// from beside the model file the first time something needs it.
    pub(crate) negative: Mutex<Option<Model>>,
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
    /// When this server started: what `/v1/models` reports as every model's `created`.
    pub(crate) born: i64,
}

impl Service {
    /// The service around a model already loaded (or created).
    pub fn new(model: Model, model_path: String, seed: i64, workers: usize) -> Service {
        let born_with = model.encoding();
        let born_kind = model.kind();
        let negative_active = AtomicBool::new(model.is_negative());
        Service {
            model: Mutex::new(model),
            parked: Mutex::new(Vec::new()),
            born_with,
            born_kind,
            negative_active,
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
            born: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
        }
    }

    /// How strictly the negative network guards the output paths, as set.
    pub(crate) fn guard_config(&self) -> crate::duo::FilterConfig {
        self.guard.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Runs `f` on the model a request's `model` names (`crate::assistant::kind_of_id`):
    /// the running one for `""`, `radixnet` or its own kind, else the model of
    /// that kind kept in memory - which is what `partner` does for
    /// `/api/converse` - and a 404 for a kind that is not here.  Takes the
    /// running model's lock, or the parked models', for as long as `f` runs.
    pub(crate) fn with_voice<T>(&self, name: &str, f: impl FnOnce(&mut Model) -> T) -> Result<T, ApiError> {
        let kind = crate::assistant::kind_of_id(name);
        let (active, active_words) = self.with_model(|m| (m.kind(), m.encoding().unit == Unit::Words));
        if kind.is_empty() || kind == active || (kind == "word" && active == "count" && active_words) {
            return Ok(self.with_model(f));
        }
        let found = if kind == "word" {
            let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            parked
                .iter_mut()
                .find(|(_, m)| m.kind() == "count" && m.encoding().unit == Unit::Words)
                .map(|(_, m)| f(m))
        } else {
            self.with_parked_kind(&kind, f)
        };
        found.ok_or_else(|| {
            ApiError::with_status(
                404,
                format!(
                    "the model {} is not in memory; the models here are {} (select a kind once to load it)",
                    crate::negative::python_repr(name),
                    self.model_ids().join(", ")
                ),
            )
        })
    }

    /// The ids of every model in memory, the running one first.
    pub(crate) fn model_ids(&self) -> Vec<String> {
        self.models_json()
            .at("data")
            .as_array()
            .iter()
            .filter_map(|m| m.at("id").as_str().map(str::to_string))
            .collect()
    }

    /// `GET /v1/models`: the models in memory as an OpenAI model list, `radixnet-<kind>` each.
    pub(crate) fn models_json(&self) -> Json {
        let describe = |m: &Model, active: bool| {
            Json::obj([
                ("id", Json::str(crate::assistant::model_id(m))),
                ("object", Json::str("model")),
                ("created", Json::Int(self.born)),
                ("owned_by", Json::str("radixnet")),
                ("kind", Json::str(m.kind())),
                ("label", Json::str(kinds::label(m.kind()))),
                ("encoding", Json::str(m.encoding().to_string())),
                ("units", Json::str(m.encoding().units_name())),
                ("active", Json::Bool(active)),
            ])
        };
        let mut data = vec![self.with_model(|m| describe(m, true))];
        {
            let parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            data.extend(parked.iter().map(|(_, m)| describe(m, false)));
        }
        if !self.negative_active.load(std::sync::atomic::Ordering::Relaxed) {
            let slot = self.negative.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(negative) = slot.as_ref() {
                data.push(describe(negative, false));
            }
        }
        Json::obj([("object", Json::str("list")), ("data", Json::Arr(data))])
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
    pub(crate) fn job_progress(&self, record: Json) {
        let mut job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = job.as_mut() {
            if job.state == "running" {
                job.records.push(record);
            }
        }
    }

    /// The records the running job has reported so far.
    #[allow(dead_code)] // for a job route that reports its records before it is done
    pub(crate) fn job_records(&self) -> Vec<Json> {
        let job = self.job.lock().unwrap_or_else(|e| e.into_inner());
        job.as_ref().map(|j| j.records.clone()).unwrap_or_default()
    }

    /// Whether `/api/job/stop` (or an area's own stop route) asked the running
    /// job to stop; a job checks between its steps.
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
            // the three ends Python's and Go's jobs have: `done`, `stopped` when
            // it was asked to stop and did, `error` - what the frontend reads
            Ok(records) => {
                let stopped = self.stop.load(Ordering::Relaxed);
                crate::log_info!(
                    LOG,
                    "{} {}, {} record(s)",
                    job.id,
                    if stopped { "stopped" } else { "done" },
                    records.len()
                );
                job.state = if stopped { "stopped" } else { "done" }.to_string();
                job.records = records;
            }
            Err(message) => {
                crate::log_error!(LOG, "{} failed: {message}", job.id);
                job.state = "error".to_string();
                job.error = Some(message);
            }
        }
    }

    /// Saves the active kind to its own file, and says whether it landed.
    pub(crate) fn autosave(&self) -> bool {
        let path = self.active_path();
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

    /// The kinds this server runs - every kind Python's does (`model_kinds()`).
    fn kinds(&self) -> Json {
        kinds::kinds_json()
    }

    /// The running model's kind, its label and what it counts in.
    ///
    /// Words are not a kind - they are an encoding - so the units follow the
    /// encoding while the kind stays what it is.
    fn active_kind(&self) -> (&'static str, &'static str, &'static str) {
        let (kind, words) = self.with_model(|m| (m.kind(), m.encoding().unit == Unit::Words));
        (kind, kinds::label(kind), if words { "words" } else { "chars" })
    }

    /// Where a model of `kind` (and `encoding`, for a count model) is saved
    /// by default: `model_path` for the kind and encoding the server was
    /// started on, `<stem>.<kind><ext>` for another kind - Python's
    /// `model_path_for` - and the `.count` / `.word` pair for the count model.
    pub(crate) fn path_for_kind(&self, kind: &str, encoding: Encoding) -> String {
        if self.model_path.is_empty() {
            return String::new();
        }
        if kind == "count" && self.born_kind == "count" {
            return self.model_path_for(encoding);
        }
        if kind == self.born_kind && (kind != "count" || encoding == self.born_with) {
            return self.model_path.clone();
        }
        if kind == "negative" {
            return self.negative_path(); // the one the guard reads, beside the model
        }
        let (stem, ext) = split_extension(&self.model_path);
        let tag = match kind {
            "count" if encoding.unit == Unit::Words => "word",
            other => other,
        };
        format!("{stem}.{tag}{ext}")
    }

    /// The running model's own file.
    pub(crate) fn active_path(&self) -> String {
        let (kind, encoding) = self.with_model(|m| (m.kind(), m.encoding()));
        self.path_for_kind(kind, encoding)
    }

    /// Runs `f` on a positive model kept in memory while the negative network
    /// is the running one - the born kind's first, then radix, count and
    /// resonant (Python's `positive_model`); `None` when none is parked.
    /// Takes the parked models' lock, then whatever `f` takes.
    pub(crate) fn with_parked_positive<T>(&self, f: impl FnOnce(&mut Model) -> T) -> Option<T> {
        let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        let order = [self.born_kind, "radix", "count", "resonant"];
        let at = order
            .iter()
            .find_map(|kind| parked.iter().position(|(_, m)| m.kind() == *kind && !m.is_negative()))?;
        Some(f(&mut parked[at].1))
    }

    /// Runs `f` on the parked model of `kind` (a count model under any
    /// encoding); `None` when there is none.  Takes the parked models' lock,
    /// then whatever `f` takes.
    pub(crate) fn with_parked_kind<T>(&self, kind: &str, f: impl FnOnce(&mut Model) -> T) -> Option<T> {
        let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        let at = parked.iter().position(|(_, m)| m.kind() == kind)?;
        Some(f(&mut parked[at].1))
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

    /// The models this server has in memory right now, active one first: a
    /// count model by its encoding, any other by its kind - and the negative
    /// network the guard holds, once it has been loaded.
    fn in_memory(&self) -> Vec<String> {
        let mut out = vec![self.with_model(|m| park_key(m))];
        {
            let parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            out.extend(parked.iter().map(|(key, _)| key.clone()));
        }
        let guard_loaded = self.negative.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        if guard_loaded && !out.iter().any(|k| k == "negative") {
            out.push("negative".to_string());
        }
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
    fn select(&self, target: Target, seed: i64) -> Result<&'static str, ApiError> {
        self.ensure_idle()?;
        let active = self.with_model(|m| park_key(m));
        let key = match target {
            Target::Count(encoding) => encoding.to_string(),
            Target::Kind(kind) => kind.to_string(),
        };
        if key == active {
            return Ok("active");
        }
        // no two locks are ever held here: each is taken, used and let go
        let (origin, mut wanted) = if let Target::Kind("negative") = target {
            // the one negative network: the guard's, else its file, else a fresh one
            let held = self.negative.lock().unwrap_or_else(|e| e.into_inner()).take();
            match held {
                Some(model) => ("memory", model),
                None => {
                    let path = self.negative_path();
                    let on_disk = !path.is_empty() && std::path::Path::new(&path).is_file();
                    (
                        if on_disk { "file" } else { "new" },
                        self.load_negative(self.active_encoding())?,
                    )
                }
            }
        } else {
            let found = {
                let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
                parked
                    .iter()
                    .position(|(name, _)| *name == key)
                    .map(|at| parked.remove(at).1)
            };
            match (found, target) {
                (Some(model), _) => ("memory", model),
                (None, Target::Count(encoding)) => {
                    // the default path is a convention, not a claim: a file there
                    // built under another encoding is simply not this one's, so it
                    // is left alone rather than refused
                    let path = self.path_for_kind("count", encoding);
                    let on_disk = if path.is_empty() || !std::path::Path::new(&path).is_file() {
                        None
                    } else {
                        Model::load(&path)
                            .ok()
                            .filter(|m| m.encoding() == encoding && m.kind() == "count")
                    };
                    match on_disk {
                        Some(found) => ("file", found),
                        None => (
                            "new",
                            Model::new(
                                seed,
                                GraphOptions {
                                    encoding,
                                    ..Default::default()
                                },
                            )?,
                        ),
                    }
                }
                (None, Target::Kind(kind)) => {
                    let path = self.path_for_kind(kind, Encoding::default());
                    if !path.is_empty() && std::path::Path::new(&path).is_file() {
                        let found = Model::load(&path)?;
                        if found.kind() != kind {
                            return Err(ApiError::bad_request(format!(
                                "{path} holds a {} model, not a {kind} one",
                                found.kind()
                            )));
                        }
                        ("file", found)
                    } else {
                        ("new", kinds::new_model(kind, seed, Encoding::default(), &[])?)
                    }
                }
            }
        };
        wanted.workers = self.workers;
        wanted.g.workers = self.workers;
        let now_negative = wanted.is_negative();
        let previous = self.with_model(|m| std::mem::replace(m, wanted));
        self.negative_active.store(now_negative, Ordering::Relaxed);
        crate::log_info!(LOG, "model {active} -> {key} ({origin}); {active} kept in memory");
        self.keep(previous);
        Ok(origin)
    }

    /// Makes `model` the running model (a reset or a load): a model of
    /// another kind that was running is kept in memory, one of the same kind is
    /// replaced - Python's `_replace_model`.
    pub(crate) fn install(&self, mut model: Model) {
        model.workers = self.workers;
        model.g.workers = self.workers;
        let key = park_key(&model);
        let now_negative = model.is_negative();
        let previous = self.with_model(|m| std::mem::replace(m, model));
        self.negative_active.store(now_negative, Ordering::Relaxed);
        if now_negative {
            // there is one negative network: the new one replaces the guard's
            *self.negative.lock().unwrap_or_else(|e| e.into_inner()) = None;
        } else {
            let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            parked.retain(|(name, _)| *name != key);
        }
        if previous.kind() != self.with_model(|m| m.kind()) {
            self.keep(previous);
        }
    }

    /// Keeps a model that stopped running: the negative network where the
    /// guard finds it, any other among the parked ones.
    fn keep(&self, model: Model) {
        if model.is_negative() {
            *self.negative.lock().unwrap_or_else(|e| e.into_inner()) = Some(model);
        } else {
            let key = park_key(&model);
            let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            parked.retain(|(name, _)| *name != key);
            parked.push((key, model));
        }
    }
}

/// What `POST /api/model/select` asks for: a count model over an encoding
/// (the word model is one), or a kind.
#[derive(Clone, Copy)]
enum Target {
    Count(Encoding),
    Kind(&'static str),
}

/// The name a model is kept under: its encoding for a count model, its kind
/// for any other.
fn park_key(m: &Model) -> String {
    if m.kind() == "count" {
        m.encoding().to_string()
    } else {
        m.kind().to_string()
    }
}

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
        // a ZIP archive is kept whole and its text entries unpacked here (D-034)
        if let Some(texts) = crate::source::upload_texts(&path, whole, self.workers)
            .map_err(|err| ApiError::bad_request(format!("{name}: {err}")))?
        {
            return Ok(texts);
        }
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
    // the replay buffer the model keeps (../../SPEC-SearchAndTraining.md section 4)
    let replay = svc.with_model(|m| match &m.replay {
        Some(b) => Json::obj([
            ("size", Json::Int(b.size as i64)),
            ("texts", Json::Int(b.len() as i64)),
            ("seen", Json::Int(b.seen)),
        ]),
        None => Json::Null,
    });
    pairs.push(("replay".to_string(), replay));
    pairs.push(("kinds".to_string(), svc.kinds()));
    pairs.push(("job".to_string(), svc.job_json()));
    pairs.push((
        "routes".to_string(),
        Json::strs(svc.routes.get().cloned().unwrap_or_default()),
    ));
    // honestly: the radix model's learning rule runs in Rust here; torch is not ported
    pairs.push(("backends".to_string(), crate::backend::describe_backends()));
    pairs.push(("model_path".to_string(), Json::str(svc.active_path())));
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
    // the tools the network may call, and the server's settings for them
    pairs.push(("tools".to_string(), crate::tools::status_json(svc)));
    pairs.push(("engine".to_string(), Json::str(ENGINE)));
    pairs.push(("workers".to_string(), Json::Int(svc.workers as i64)));
    pairs.push(("counting".to_string(), Json::str("exact")));
    Ok(doc)
}

fn model_info(svc: &Arc<Service>, _r: &Request) -> Answer {
    let (kind, label, units) = svc.active_kind();
    let weights = svc.with_model(|m| kinds::weight_config(m));
    let enc = svc.active_encoding();
    let chars = Encoding::default();
    let words = Encoding::new(Unit::Words, enc.n, enc.stride).expect("a valid word encoding");
    // every kind's default file (Python's `paths`), and the word model's
    let mut paths: Vec<(String, Json)> = kinds::KINDS
        .iter()
        .map(|k| {
            let path = if k.kind == "count" {
                svc.path_for_kind("count", chars)
            } else {
                svc.path_for_kind(k.kind, enc)
            };
            (k.kind.to_string(), Json::str(path))
        })
        .collect();
    paths.push(("word".to_string(), Json::str(svc.path_for_kind("count", words))));
    Ok(Json::obj([
        ("kind", Json::str(kind)),
        ("label", Json::str(label)),
        ("units", Json::str(units)),
        ("kinds", svc.kinds()),
        ("encoding", Json::str(enc.to_string())),
        ("unit", Json::str(enc.unit.name())),
        ("ngram", Json::Int(enc.n as i64)),
        ("stride", Json::Int(enc.stride as i64)),
        ("model_path", Json::str(svc.active_path())),
        ("paths", Json::Obj(paths)),
        ("in_memory", Json::strs(svc.in_memory())),
        ("weights", weights),
        ("engine", Json::str(ENGINE)),
    ]))
}

/// `POST /api/model/select`: the kind, and - the dial being what words are now
/// - the encoding.
///
/// `{"kind": "radix" | "count" | "negative" | "resonant"}` makes that kind the
/// running model: the one kept in memory, else its file, else a fresh one,
/// with the model that was running parked with its unsaved work (Python's
/// `select_kind`).  Words are not a kind, so `{"kind": "word"}` is read as a
/// request for a word *encoding* of the count model;
/// `{"encoding": "word:2:1"}` says it directly.
fn model_select(svc: &Arc<Service>, r: &Request) -> Answer {
    let spec = r.text("encoding", "");
    let target = if !spec.is_empty() {
        Target::Count(parse_encoding(&spec)?)
    } else {
        let kind = r.text("kind", kinds::DEFAULT_KIND).trim().to_ascii_lowercase();
        if kind.is_empty() {
            return Err(ApiError::bad_request("'kind' must not be empty"));
        }
        match kind.as_str() {
            "count" | "char" | "chars" => Target::Count(svc.encoding_for_unit(Unit::Chars)),
            "word" | "words" => Target::Count(svc.encoding_for_unit(Unit::Words)),
            other => Target::Kind(kinds::parse_kind(other).map_err(ApiError::bad_request)?),
        }
    };
    let origin = svc.select(target, svc.seed)?;
    let mut doc = model_info(svc, r)?;
    if let Json::Obj(pairs) = &mut doc {
        pairs.push(("origin".to_string(), Json::str(origin)));
        pairs.push(("stats".to_string(), svc.with_model(|m| stats(m))));
    }
    Ok(doc)
}

/// The score-function settings a request body carries - every kind's, as
/// Python's `_weight_options` reads them; the model refuses the ones it does
/// not have.
fn weight_options(r: &Request) -> Result<Vec<(String, f64)>, ApiError> {
    let mut options = Vec::new();
    for name in kinds::WEIGHT_FIELDS {
        match r.body.get(name) {
            None | Some(Json::Null) => {}
            Some(value) => {
                let number = value
                    .as_f64()
                    .ok_or_else(|| ApiError::bad_request(format!("'{name}' must be a number")))?;
                options.push((name.to_string(), number));
            }
        }
    }
    Ok(options)
}

/// `POST /api/model/weights`: the running model's score function, changed
/// (`configure_weights`); the sine model has none.
fn model_weights(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let options = weight_options(r)?;
    let out = svc.with_model(|m| -> Result<Json, String> {
        let weights = kinds::configure_weights(m, &options)?;
        Ok(Json::obj([("weights", weights), ("stats", stats(m))]))
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

/// The sampling filters and the beam's diversity of a request
/// (`../../SPEC-SearchAndTraining.md` sections 1-2): each is off at its
/// default, so a request that names none of them searches as it always did.
fn search_fields(r: &Request) -> Result<SearchTuning, ApiError> {
    let tuning = SearchTuning {
        filter: SamplingFilter {
            top_k: r.usize("top_k", 0)?,
            top_p: r.number("top_p", 1.0)?,
            min_p: r.number("min_p", 0.0)?,
        },
        diversity: r.number("diversity", 0.0)?,
        ..Default::default()
    };
    tuning.check()?;
    Ok(tuning)
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
    let tuning = search_fields(r)?;
    let max_length = r.body.at("max_length").as_i64();
    let opts = PredictOptions {
        length: r.usize("length", 20)?,
        // Python's default; the count model's dijkstra is its beam
        mode: r.text("mode", "dijkstra"),
        k: r.usize("k", 5)?,
        beam: r.usize("beam", 0)?,
        step_penalty: r.number("step_penalty", 0.0)?,
        temperature: r.number("temperature", 1.0)?,
        to_end: r.flag("to_end", false),
        max_length: max_length.filter(|v| *v >= 0).map(|v| v as usize),
        traversal,
        penalty_scale,
        merit_scale,
        top_k: tuning.filter.top_k,
        top_p: tuning.filter.top_p,
        min_p: tuning.filter.min_p,
        diversity: tuning.diversity,
        origin: START,
    };
    let prefix = r.text("prefix", "");
    let (kind, ..) = svc.active_kind();
    // the negative network guards the answer: the best continuation it does
    // not veto is the one that comes back, and when it vetoes every one the
    // continuation is empty and `guard` says why (`guard: false` turns it off;
    // `provenance: false` keeps the vetoes' reasons out of the answer)
    let provenance = crate::duo::maybe_flag(r, "provenance")?;
    let guarded = if r.flag("guard", true) {
        svc.guard(provenance, |pair| -> Result<(crate::beam::Prediction, Json), String> {
            let found = pair.positive.predict(&prefix, &opts)?;
            let (ranked, verdicts) = pair.rank(&prefix, &found);
            let kept = verdicts.iter().filter(|v| v.decision != "reject").count();
            let report = crate::duo::guard_report(
                pair,
                &verdicts,
                vec![
                    ("candidates", Json::Int(verdicts.len() as i64)),
                    ("kept", Json::Int(kept as i64)),
                ],
            );
            Ok((ranked, report))
        })?
    } else {
        None
    };
    let (found, guard) = match guarded {
        Some(outcome) => {
            let (found, report) = outcome?;
            (found, report)
        }
        None => (svc.with_model(|m| m.predict(&prefix, &opts))?, Json::Null),
    };
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
        ("guard", guard),
    ]))
}

fn generate(svc: &Arc<Service>, r: &Request) -> Answer {
    let (traversal, penalty_scale, merit_scale) = traversal_of(r)?;
    let tuning = search_fields(r)?;
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
        top_k: tuning.filter.top_k,
        top_p: tuning.filter.top_p,
        min_p: tuning.filter.min_p,
        diversity: tuning.diversity,
    };
    // the negative network guards the texts: the model is asked for
    // `over_sample` times as many and what the negative half recognises as
    // failure never reaches the answer (fewer come back when it vetoed a lot;
    // `provenance: false` reports how many were vetoed, not which or why)
    let provenance = crate::duo::maybe_flag(r, "provenance")?;
    let guarded = if r.flag("guard", true) {
        svc.guard(
            provenance,
            |pair| -> Result<(Vec<crate::search::PathResult>, Json), String> {
                let outcome = pair.generate(opts.count, &opts)?;
                let report = crate::duo::guard_report(
                    pair,
                    &outcome.verdicts,
                    vec![
                        ("candidates", Json::Int(outcome.candidates as i64)),
                        ("kept", Json::Int(outcome.kept.len() as i64)),
                        ("asked", Json::Int(outcome.asked as i64)),
                        ("rate", outcome.rate.map(Json::Num).unwrap_or(Json::Null)),
                    ],
                );
                Ok((outcome.results, report))
            },
        )?
    } else {
        None
    };
    let (samples, guard) = match guarded {
        Some(outcome) => outcome?,
        None => (svc.with_model(|m| m.generate(&opts))?, Json::Null),
    };
    Ok(Json::obj([
        ("samples", Json::Arr(samples.iter().map(path_json).collect())),
        ("guard", guard),
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

/// `POST /api/feedback`: rated texts as a background job (202), polled on
/// `/api/job` - Python's `_r_feedback` / `start_feedback`, for every kind.
///
/// Both sets: 2NRL; only `good`: a thumbs up; only `bad`: a thumbs down -
/// each through the model's own kind ([`crate::kinds`]): the sine model
/// learns at `neg_lr` / `pos_lr` (0.5 / 0.1) in batches of 4 (rated sets are
/// small), the count and phase models reward and penalise by `strength`.
/// `<side>_weights` (shares) or `<side>_ratings` (marks out of 10) turn the
/// thumbs into ratings.
fn feedback(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let good = svc.texts_of(r, "good", "good_text", "good_files")?;
    let bad = svc.texts_of(r, "bad", "bad_text", "bad_files")?;
    let Some(action) = kinds::feedback_action(&good, &bad) else {
        return Err(ApiError::bad_request(
            "give 'good' (thumbs up) and/or 'bad' (thumbs down) texts: lists, good_text / bad_text (one per \
             line) or good_files / bad_files (upload names)",
        ));
    };
    let mut o = feedback_options(r, Feedback::rated())?;
    o.good_weights = ratings(r, "good", good.len())?;
    o.bad_weights = ratings(r, "bad", bad.len())?;
    validate_feedback(&o)?;
    let weights = |w: &Option<Vec<f64>>| w.as_ref().map(|w| Json::nums(w.clone())).unwrap_or(Json::Null);
    let (good_weights, bad_weights) = (weights(&o.good_weights), weights(&o.bad_weights));
    let (good_count, bad_count) = (good.len(), bad.len());
    svc.start_job("feedback");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| {
            let mut watch = |record: &crate::model::EpochRecord| {
                worker.job_progress(record.to_json());
                !worker.stopping()
            };
            match action {
                "2nrl" => kinds::two_nrl(m, &bad, &good, &o, &mut watch),
                "reward" => kinds::reward(m, &good, &o, &mut watch).map(|records| (Vec::new(), records)),
                _ => kinds::punish(m, &bad, &o, &mut watch).map(|records| (records, Vec::new())),
            }
        });
        worker.finish_job(
            outcome.map(|(negative, positive)| negative.iter().chain(positive.iter()).map(|r| r.to_json()).collect()),
        );
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("action", Json::str(action)),
        ("good", Json::Int(good_count as i64)),
        ("bad", Json::Int(bad_count as i64)),
        ("good_weights", good_weights),
        ("bad_weights", bad_weights),
    ])))
}

/// `POST /api/2nrl`: 2NRL as a background job (202), through the model's own
/// kind, over Python's defaults for every kind (`start_two_nrl`: 3 and 3
/// epochs, rates 0.05 and 0.01).
fn two_nrl(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let bad = svc.texts_of(r, "bad", "bad_text", "bad_files")?;
    let good = svc.texts_of(r, "good", "good_text", "good_files")?;
    if bad.is_empty() && good.is_empty() {
        return Err(ApiError::bad_request("2NRL needs 'bad' and / or 'good' texts"));
    }
    let mut o = feedback_options(r, Feedback::two_nrl())?;
    o.good_weights = ratings(r, "good", good.len())?;
    o.bad_weights = ratings(r, "bad", bad.len())?;
    validate_feedback(&o)?;
    svc.start_job("two_nrl");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = worker.with_model(|m| {
            let mut watch = |record: &crate::model::EpochRecord| {
                worker.job_progress(record.to_json());
                !worker.stopping()
            };
            kinds::two_nrl(m, &bad, &good, &o, &mut watch)
        });
        worker.finish_job(outcome.map(|(negative, positive)| {
            // one list, the negative phase first, because a job's records are a
            // list of epochs and 2NRL is two phases of them
            negative.iter().chain(positive.iter()).map(|r| r.to_json()).collect()
        }));
    });
    Ok(accepted(Json::obj([("job", svc.job_json())])))
}

/// Refuses feedback settings no phase can train with before the job starts,
/// as Python's `TrainConfig(...).validate()` of both phases does.
fn validate_feedback(o: &Feedback) -> Result<(), ApiError> {
    let base = TrainConfig::default();
    for (epochs, lr, act_lr) in [
        (o.neg_epochs, o.neg_lr, base.act_lr),
        (o.pos_epochs, o.pos_lr, o.pos_lr / 10.0),
    ] {
        TrainConfig {
            epochs,
            lr,
            act_lr,
            batch_size: o.batch_size.unwrap_or(base.batch_size),
            clip: o.clip.unwrap_or(base.clip),
            ..base.clone()
        }
        .validate()
        .map_err(ApiError::bad_request)?;
    }
    Ok(())
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

/// The training settings of a request body over Python's `TrainConfig`
/// defaults (Python's `_train_config`): every kind's, each kind reading what
/// applies to it - the routes that start a training job share it.
pub(crate) fn train_config(r: &Request) -> Result<TrainConfig, ApiError> {
    let base = TrainConfig::default();
    let text = |name: &str| Some(r.text(name, "").trim().to_string()).filter(|s| !s.is_empty());
    let config = TrainConfig {
        epochs: r.usize("epochs", base.epochs)?,
        lr: r.number("lr", base.lr)?,
        act_lr: r.number("act_lr", base.act_lr)?,
        lr_schedule: text("lr_schedule"),
        act_lr_schedule: text("act_lr_schedule"),
        reverse_schedule: r.flag("reverse_schedule", false),
        batch_size: r.usize("batch_size", base.batch_size)?,
        clip: r.number("clip", base.clip)?,
        auto_compress: r.flag("auto_compress", !r.flag("no_compress", false)),
        shuffle: r.flag("shuffle", base.shuffle),
        checkpoint_every: r.usize("checkpoint_every", 0)?,
        plan: plan_fields(r)?,
        ..base
    };
    config.validate().map_err(ApiError::bad_request)?;
    Ok(config)
}

/// How a training run walks its texts (`../../SPEC-SearchAndTraining.md`
/// sections 3-6): `order`, `curriculum`, `replay`, `replay_size`, `patience`
/// and `min_delta`, each off when the request leaves it out.  A value out of
/// range is a 400, before any job starts.
fn plan_fields(r: &Request) -> Result<crate::training::Plan, ApiError> {
    let order = r.text("order", "").trim().to_ascii_lowercase();
    let plan = crate::training::Plan {
        order: if order.is_empty() { "corpus".to_string() } else { order },
        curriculum: r.number("curriculum", 1.0)?,
        replay: r.number("replay", 0.0)?,
        // absent (or null) leaves the model's buffer as it is; 0 drops it
        replay_size: match r.body.get("replay_size") {
            None | Some(Json::Null) => None,
            Some(_) => Some(r.usize("replay_size", 0)?),
        },
        patience: r.usize("patience", 0)?,
        min_delta: r.number("min_delta", 0.0)?,
    };
    plan.check()?;
    Ok(plan)
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
    let config = train_config(r)?;
    let settings = kinds::TrainSettings {
        config,
        chunk_size: r.usize("chunk", 0)?,
        reason: r.text("reason", ""),
        severity: r.number("severity", 1.0)?,
        source: r.text("source", "api"),
        note: r.text("note", ""),
    };
    // checkpoints every N epochs, into the server's --checkpoint-dir
    let every = r.usize("checkpoint_every", 0)?;
    if every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    svc.start_job("train");
    // the run holds the model for as long as it takes, which is what makes a
    // second request wait; the answer goes out now, and the frontend follows the
    // run on /api/job - the same 202 the Python and Go servers answer with
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = crate::checkpoint::train_job(&worker, &texts, &settings, every);
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

/// `GET /api/graph`: the `limit` most visited nodes (ties: the lowest id)
/// plus START and END, and the edges among them - Python's `_graph_view`, key
/// for key, for every kind: the sine model's node parameters, the count
/// model's shares and window, the phase model's advances and locks.
fn graph_view(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 150)?;
    Ok(svc.with_model(|m| {
        m.g.prepare();
        let g = &m.g;
        let kind = m.kind();
        // START and END lead; the rest are ranked from id 2, BACK included, as Python ranks them
        let mut real: Vec<usize> = (END + 1..g.num_node_ids()).filter(|&i| g.is_alive(i)).collect();
        if limit < real.len() {
            // `nsmallest(limit, real, key=(-count, id))`: the most visited, ties to the lowest id
            real.sort_by(|&a, &b| {
                let (ca, cb) = (g.node_count(a), g.node_count(b));
                if cb.less(ca) {
                    std::cmp::Ordering::Less
                } else if ca.less(cb) {
                    std::cmp::Ordering::Greater
                } else {
                    a.cmp(&b)
                }
            });
            real.truncate(limit);
            real.sort_unstable();
        }
        let mut ids = vec![START, END];
        ids.extend(real);
        let resonant = g.res.as_ref();
        let nodes: Vec<Json> = ids
            .iter()
            .map(|&i| {
                let (z, a, b, h, k) = g.node_parameters(i);
                let mut pairs = vec![
                    ("id".to_string(), Json::Int(i as i64)),
                    ("label".to_string(), Json::str(g.label(i).to_string())),
                    ("count".to_string(), Json::Int(g.node_count(i).value)),
                    ("count_resets".to_string(), Json::Int(g.node_count(i).resets)),
                    ("activation".to_string(), Json::Num(g.activation_of(i))),
                    ("z".to_string(), Json::Num(z)),
                    ("a".to_string(), Json::Num(a)),
                    ("b".to_string(), Json::Num(b)),
                    ("h".to_string(), Json::Num(h)),
                    ("k".to_string(), Json::Num(k)),
                ];
                if let Some(res) = resonant {
                    pairs.push(("advance".to_string(), Json::Int(res.advance[i] as i64)));
                }
                Json::Obj(pairs)
            })
            .collect();
        let has_reward = kind == "count" || kind == "resonant";
        let mut edges: Vec<(usize, usize, Json)> = Vec::new();
        for &p in &ids {
            let shares: Vec<(usize, f64, f64)> = match kind {
                "count" => g.shares(p).into_iter().map(|(_, e, x, y)| (e, x, y)).collect(),
                "resonant" => {
                    let res = resonant.expect("a phase graph");
                    g.children_of(p)
                        .into_iter()
                        .map(|t| (t.e, res.coherence(t.e), res.mu(t.e)))
                        .collect()
                }
                _ => Vec::new(),
            };
            for cc in g.child_costs_from(p, None) {
                if !ids.contains(&cc.child) {
                    continue;
                }
                let e = cc.edge;
                let mut pairs = vec![
                    ("source".to_string(), Json::Int(p as i64)),
                    ("target".to_string(), Json::Int(cc.child as i64)),
                    ("weight".to_string(), Json::Num(g.edge_w[e])),
                    ("count".to_string(), Json::Int(g.edge_traversals(e).value)),
                    ("count_resets".to_string(), Json::Int(g.edge_traversals(e).resets)),
                    ("prob".to_string(), Json::Num((-cc.cost).exp())),
                    ("cost".to_string(), Json::Num(cc.cost)),
                ];
                if has_reward {
                    pairs.push(("reward".to_string(), Json::Num(g.edge_reward[e])));
                }
                let (first, second) = shares
                    .iter()
                    .find(|(edge, _, _)| *edge == e)
                    .map(|&(_, x, y)| (x, y))
                    .unwrap_or((0.0, 0.0));
                if kind == "count" {
                    pairs.push(("share".to_string(), Json::Num(first)));
                    pairs.push(("recent_share".to_string(), Json::Num(second)));
                    pairs.push(("recent_count".to_string(), Json::Int(g.window_edge_count[e])));
                } else if kind == "resonant" {
                    pairs.push(("coherence".to_string(), Json::Num(first)));
                    pairs.push(("mu".to_string(), Json::Num(second)));
                }
                edges.push((p, cc.child, Json::Obj(pairs)));
            }
        }
        edges.sort_by_key(|(p, c, _)| (*p, *c));
        let mut view = vec![
            ("nodes".to_string(), Json::Arr(nodes)),
            (
                "edges".to_string(),
                Json::Arr(edges.into_iter().map(|(_, _, e)| e).collect()),
            ),
            ("limit".to_string(), Json::Int(limit as i64)),
            ("total_nodes".to_string(), Json::Int(g.num_nodes() as i64)),
            ("total_edges".to_string(), Json::Int(g.num_edges() as i64)),
        ];
        if has_reward {
            view.push(("total_traversals".to_string(), Json::Int(g.total_traversals().value)));
            view.push((
                "total_traversals_resets".to_string(),
                Json::Int(g.total_traversals().resets),
            ));
        }
        if kind == "count" {
            view.push(("window_traversals".to_string(), Json::Int(g.window_traversals() as i64)));
            view.push(("window".to_string(), Json::Int(g.window_size as i64)));
        }
        if let Some(res) = resonant {
            view.push(("buckets".to_string(), Json::Int(res.buckets as i64)));
        }
        Json::Obj(view)
    }))
}

/// The count model keeps the judged paths and the node ratios; the other
/// kinds say so, in Python's words.
fn count_only(svc: &Arc<Service>, what: &str) -> Result<(), ApiError> {
    let kind = svc.with_model(|m| m.kind());
    if kind == "count" {
        return Ok(());
    }
    Err(ApiError::bad_request(format!("the {kind} model does not count {what}")))
}

fn paths(svc: &Arc<Service>, r: &Request) -> Answer {
    let limit = r.query_usize("limit", 50)?;
    count_only(svc, "paths")?;
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
    count_only(svc, "node ratios")?;
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
        ("think_label", Json::str(THINK_LABEL)),
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
    let path = r.text("path", &svc.active_path());
    svc.with_model(|m| m.save(&path))?;
    let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
    Ok(Json::obj([
        ("saved", Json::str(path)),
        ("bytes", Json::Int(bytes as i64)),
    ]))
}

fn load(svc: &Arc<Service>, r: &Request) -> Answer {
    svc.ensure_idle()?;
    let path = r.text("path", &svc.active_path());
    let loaded = Model::load(&path)?;
    svc.install(loaded);
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
    // the kind (the running one unless the body names another) and its score function
    let kind = match r.body.get("kind").and_then(|v| v.as_str()).map(str::trim) {
        Some(name) if !name.is_empty() => kinds::parse_kind(name).map_err(ApiError::bad_request)?,
        _ => svc.active_kind().0,
    };
    let options = weight_options(r)?;
    let fresh = kinds::new_model(kind, seed, encoding, &options).map_err(ApiError::bad_request)?;
    svc.install(fresh);
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
        // in name order, as the Python service lists them
        let mut entries: Vec<std::fs::DirEntry> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            // a file still on its way in (a `.part`) is not an upload yet
            if !path.is_file() || name.ends_with(".part") {
                continue;
            }
            // an archive reports the lines of its text entries, not its bytes read as text
            if let Some(record) = crate::source::archive_record(&path, svc.workers) {
                rows.push(record);
                continue;
            }
            // a text file, counted a line at a time, whatever its size
            if let Ok(record) = crate::multipart::record(&path) {
                rows.push(record);
            }
        }
    }
    Ok(Json::obj([("uploads", Json::Arr(rows))]))
}

/// `POST /api/uploads`: every form Python's `_r_upload` takes - JSON `{name,
/// content | content_base64}` or `{files: [...]}`, `multipart/form-data`, or a
/// raw body named by `?name=` - answered with the records of what was stored.
/// The route streams: the body is read as it arrives and a multipart part or
/// a raw body goes straight to disk, so an archive may be of any size
/// (`multipart::store_stream`, D-035).
fn upload(svc: &Arc<Service>, r: &Request, body: &mut dyn std::io::Read) -> Answer {
    if svc.upload_dir.is_none() {
        return Err(ApiError::bad_request("no upload directory is configured"));
    }
    crate::multipart::store_stream(svc, r, body)
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

/// `GET /api/schedule`: what a learning-rate schedule expression may use -
/// variables, functions, helpers, presets (Python's `describe`).
fn schedule(svc: &Arc<Service>, _r: &Request) -> Answer {
    let _ = svc;
    Ok(crate::schedule::describe())
}

/// `POST /api/schedule/preview`: the per-epoch rates a pair of schedule
/// expressions gives - the graph the frontend draws while one is typed.
fn schedule_preview(svc: &Arc<Service>, r: &Request) -> Answer {
    let _ = svc;
    let base = TrainConfig::default();
    let text = |name: &str| Some(r.text(name, "").trim().to_string()).filter(|s| !s.is_empty());
    let (lr_schedule, act_lr_schedule) = (text("lr_schedule"), text("act_lr_schedule"));
    let epochs = r.usize("epochs", base.epochs)?;
    let lr = r.number("lr", base.lr)?;
    let act_lr = r.number("act_lr", base.act_lr)?;
    let reverse = r.flag("reverse_schedule", false);
    let points = crate::schedule::preview_points(
        lr_schedule.as_deref(),
        act_lr_schedule.as_deref(),
        epochs as i64,
        lr,
        act_lr,
        reverse,
    )
    .map_err(ApiError::bad_request)?;
    let name = |s: &Option<String>| s.clone().map(Json::str).unwrap_or(Json::Null);
    Ok(Json::obj([
        ("lr_schedule", name(&lr_schedule)),
        ("act_lr_schedule", name(&act_lr_schedule)),
        ("epochs", Json::Int(epochs as i64)),
        ("lr", Json::Num(lr)),
        ("act_lr", Json::Num(act_lr)),
        ("reverse_schedule", Json::Bool(reverse)),
        ("points", Json::Arr(points.iter().map(|p| p.to_json()).collect())),
    ]))
}

/// A feedback route's settings over its defaults: the epochs and strength
/// every kind reads, the learning rates and batch settings the radix model
/// reads (Python's `_train_overrides`).
fn feedback_options(r: &Request, base: Feedback) -> Result<Feedback, ApiError> {
    let present = |name: &str| !matches!(r.body.get(name), None | Some(Json::Null));
    Ok(Feedback {
        neg_epochs: r.usize("neg_epochs", base.neg_epochs)?,
        pos_epochs: r.usize("pos_epochs", base.pos_epochs)?,
        neg_lr: r.number("neg_lr", base.neg_lr)?,
        pos_lr: r.number("pos_lr", base.pos_lr)?,
        strength: r.number("strength", base.strength)?,
        batch_size: if present("batch_size") {
            Some(r.usize("batch_size", 1)?.max(1))
        } else {
            base.batch_size
        },
        auto_compress: if present("auto_compress") {
            Some(r.flag("auto_compress", true))
        } else {
            base.auto_compress
        },
        clip: if present("clip") {
            Some(r.number("clip", 5.0)?)
        } else {
            base.clip
        },
        shuffle: if present("shuffle") {
            Some(r.flag("shuffle", true))
        } else {
            base.shuffle
        },
        ..base
    })
}

/// One side's per-text weights: `<side>_weights` (shares of 1) or
/// `<side>_ratings` (marks out of 10), never both - Python's `_ratings`.
fn ratings(r: &Request, side: &str, texts: usize) -> Result<Option<Vec<f64>>, ApiError> {
    let read = |key: &str, scale: f64| -> Result<Option<Vec<f64>>, ApiError> {
        match r.body.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(Json::Arr(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    let v = item
                        .as_f64()
                        .ok_or_else(|| ApiError::bad_request(format!("'{key}' must be a list of numbers")))?;
                    out.push(v / scale);
                }
                if out.len() != texts {
                    return Err(ApiError::bad_request(format!(
                        "'{key}' has {} entries for {texts} texts",
                        out.len()
                    )));
                }
                Ok(Some(out))
            }
            Some(_) => Err(ApiError::bad_request(format!("'{key}' must be a list of numbers"))),
        }
    };
    let weights = read(&format!("{side}_weights"), 1.0)?;
    let marks = read(&format!("{side}_ratings"), 10.0)?;
    if weights.is_some() && marks.is_some() {
        return Err(ApiError::bad_request(format!(
            "give '{side}_weights' or '{side}_ratings', not both"
        )));
    }
    Ok(weights.or(marks))
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
    server.stream_route("POST", "/api/uploads", upload);
    server.route("POST", "/api/uploads/delete", upload_delete);
    server.route("GET", "/api/schedule", schedule);
    server.route("POST", "/api/schedule/preview", schedule_preview);
    server.route("GET", "/api/traversals", traversals);
    // every other area answers its own routes
    crate::duo::routes(&mut server);
    crate::critic::routes(&mut server);
    crate::dialogue::routes(&mut server);
    crate::thinking::routes(&mut server);
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
    crate::assistant::routes(&mut server);
    let _ = server.state().routes.set(server.routes());
    if let Some(dir) = frontend {
        server.frontend(dir);
    }
    server
}
