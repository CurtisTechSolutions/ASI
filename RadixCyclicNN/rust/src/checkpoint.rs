//! Checkpoints in the Python `CheckpointManager` layout (`radixnet/checkpoint.py`,
//! `go/server/checkpoints.go`).
//!
//! A checkpoint is an ordinary model file, named `ckpt-<tag>-<step:06d>.json.gz`
//! inside one directory.  `latest.json` points at the one written last, and
//! `index.json` keeps every record (step, tag, metrics, time, size) so a
//! listing never opens a model file.  The layout is the contract, not a
//! convenience: a directory written here is listed, restored and continued by
//! Python and by Go, and theirs by this port - so the three can share one
//! training run's history.
//!
//! **The files win.**  The index is reconciled with the directory on every
//! read: a record whose file is gone is dropped, and a checkpoint file nobody
//! indexed (copied in by hand, left by a crash) is recorded from its name and
//! its file time.  **Keep-N pruning** removes the oldest by `(step, saved_at,
//! name)` after every save, and never the one just written.
//!
//! Training checkpoints every N epochs (`--checkpoint-every`, the train route's
//! `checkpoint_every`), and the evolve loop every N generations
//! ([`crate::gan`]).  A run that checkpoints trains an epoch at a time
//! ([`train`]), and ends where the same run without checkpoints ends - bit for
//! bit, which is what makes a checkpoint a place to resume *from*.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::{Args, Ctx};
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::model::{EpochRecord, Model, TrainOptions};
use crate::report::stats;
use crate::service::Service;

/// What this module's lines are filed under.
const LOG: &str = "checkpoint";

const PREFIX: &str = "ckpt-";
const SUFFIXES: [&str; 2] = [".json.gz", ".json"];
const LATEST: &str = "latest.json";
const INDEX: &str = "index.json";

/// How many checkpoints a directory keeps unless told otherwise.
pub const DEFAULT_KEEP: usize = 5;

/// Where the `checkpoints` command looks unless told otherwise.
pub const DEFAULT_DIR: &str = "checkpoints";

/// One checkpoint: the record `index.json` keeps and the routes report.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub name: String,
    /// The file's absolute path.
    pub path: String,
    pub step: i64,
    pub tag: String,
    /// What the run reported when the checkpoint was taken (the epoch or
    /// generation record), or `Null`.
    pub metrics: Json,
    /// `datetime.now(timezone.utc).isoformat(timespec="microseconds")`.
    pub saved_at: String,
    pub bytes: i64,
}

impl Record {
    /// `{"name", "path", "step", "tag", "metrics", "saved_at", "bytes"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("name", Json::str(self.name.clone())),
            ("path", Json::str(self.path.clone())),
            ("step", Json::Int(self.step)),
            ("tag", Json::str(self.tag.clone())),
            ("metrics", self.metrics.clone()),
            ("saved_at", Json::str(self.saved_at.clone())),
            ("bytes", Json::Int(self.bytes)),
        ])
    }

    /// A record as a file holds it; `name` is the key it was filed under.
    fn from_json(name: &str, doc: &Json) -> Option<Record> {
        if !matches!(doc, Json::Obj(_)) {
            return None;
        }
        let number = |key: &str| {
            let v = doc.at(key);
            v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)).unwrap_or(0)
        };
        Some(Record {
            name: doc.at("name").as_str().unwrap_or(name).to_string(),
            path: doc.at("path").as_str().unwrap_or("").to_string(),
            step: number("step"),
            tag: doc.at("tag").as_str().unwrap_or("").to_string(),
            metrics: doc.at("metrics").clone(),
            saved_at: doc.at("saved_at").as_str().unwrap_or("").to_string(),
            bytes: number("bytes"),
        })
    }

    fn sort_key(&self) -> (i64, &str, &str) {
        (self.step, &self.saved_at, &self.name)
    }
}

/// Whether a tag may name a checkpoint: letters, digits, `_` and `-`,
/// starting with a letter or a digit.
pub fn valid_tag(tag: &str) -> bool {
    let mut chars = tag.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// `ckpt-<tag>-<step>.json[.gz]` taken apart: `(tag, step)`, or `None` for
/// any other name.  The step has at least six digits; the tag may itself hold
/// dashes, so it is everything before the last one.
fn parse_name(name: &str) -> Option<(String, i64)> {
    let rest = name.strip_prefix(PREFIX)?;
    let stem = SUFFIXES.iter().find_map(|suffix| rest.strip_suffix(suffix))?;
    let (tag, step) = stem.rsplit_once('-')?;
    if step.len() < 6 || !step.bytes().all(|b| b.is_ascii_digit()) || !valid_tag(tag) {
        return None;
    }
    Some((tag.to_string(), step.parse().ok()?))
}

/// An ISO-8601 UTC time with `digits` fractional digits of the second
/// (`isoformat(timespec="microseconds")` is 6, `"milliseconds"` 3).
pub(crate) fn iso_time(time: SystemTime, digits: usize) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let whole = crate::clock::iso8601(since.as_secs() as i64);
    let fraction = format!("{:09}", since.subsec_nanos());
    let cut = digits.min(9);
    match whole.strip_suffix("+00:00") {
        Some(head) if cut > 0 => format!("{head}.{}+00:00", &fraction[..cut]),
        _ => whole,
    }
}

/// `json.dumps(doc, indent=2, sort_keys=True)`: every object's keys in order,
/// and anything outside ASCII escaped - the bytes Python writes these two
/// small files with.
fn python_pretty(doc: &Json) -> String {
    fn sorted(doc: &Json) -> Json {
        match doc {
            Json::Obj(pairs) => {
                let mut pairs: Vec<(String, Json)> = pairs.iter().map(|(k, v)| (k.clone(), sorted(v))).collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                Json::Obj(pairs)
            }
            Json::Arr(items) => Json::Arr(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    let mut out = String::new();
    for ch in sorted(doc).render(2).chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// A checkpoint directory: save / list / restore, with keep-N pruning.
///
/// Every operation holds one lock, so a job checkpointing while a request
/// lists the directory never sees half an index.
pub struct Checkpoints {
    dir: PathBuf,
    keep: usize,
    /// Checkpoints are gzipped (`.json.gz`) unless this is off.
    pub compress: bool,
    lock: Mutex<()>,
}

impl Checkpoints {
    /// The manager of `dir` (created when missing); `keep` must be at least 1.
    pub fn new(dir: &str, keep: usize) -> Result<Checkpoints, String> {
        if keep < 1 {
            return Err(format!("keep must be >= 1, got {keep}"));
        }
        let dir = std::path::absolute(dir).map_err(|err| format!("{dir}: {err}"))?;
        std::fs::create_dir_all(&dir).map_err(|err| format!("cannot create {}: {err}", dir.display()))?;
        Ok(Checkpoints {
            dir,
            keep,
            compress: true,
            lock: Mutex::new(()),
        })
    }

    /// The directory, as an absolute path.
    pub fn directory(&self) -> String {
        self.dir.to_string_lossy().into_owned()
    }

    /// How many checkpoints survive a save.
    pub fn keep(&self) -> usize {
        self.keep
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn read_json(&self, name: &str) -> Option<Json> {
        let doc = crate::file::read_document(&self.path(name).to_string_lossy()).ok()?;
        matches!(doc, Json::Obj(_)).then_some(doc)
    }

    fn write_json(&self, name: &str, doc: &Json) -> Result<(), String> {
        crate::file::write_atomic(&self.path(name).to_string_lossy(), python_pretty(doc).as_bytes())
    }

    fn write_index(&self, index: &[Record]) -> Result<(), String> {
        let doc = Json::Obj(index.iter().map(|r| (r.name.clone(), r.to_json())).collect());
        self.write_json(INDEX, &doc)
    }

    /// A record for a checkpoint file nobody indexed, from its name and file time.
    fn record_from_file(&self, name: &str) -> Option<Record> {
        let (tag, step) = parse_name(name)?;
        let meta = std::fs::metadata(self.path(name)).ok().filter(|m| m.is_file())?;
        Some(Record {
            name: name.to_string(),
            path: self.path(name).to_string_lossy().into_owned(),
            step,
            tag,
            metrics: Json::Null,
            saved_at: meta.modified().map(|t| iso_time(t, 6)).unwrap_or_default(),
            bytes: meta.len() as i64,
        })
    }

    /// The index, reconciled with the directory: the files win.
    fn index(&self) -> Vec<Record> {
        let mut index: Vec<Record> = Vec::new();
        let mut changed = false;
        if let Some(Json::Obj(stored)) = self.read_json(INDEX) {
            for (name, doc) in &stored {
                match Record::from_json(name, doc) {
                    Some(mut record) if self.path(name).is_file() => {
                        record.path = self.path(name).to_string_lossy().into_owned();
                        index.push(record);
                    }
                    _ => changed = true,
                }
            }
        }
        let mut names: Vec<String> = std::fs::read_dir(&self.dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        for name in names {
            if index.iter().any(|r| r.name == name)
                || !name.starts_with(PREFIX)
                || !SUFFIXES.iter().any(|s| name.ends_with(s))
            {
                continue;
            }
            if let Some(record) = self.record_from_file(&name) {
                index.push(record);
                changed = true;
            }
        }
        if changed {
            if let Err(err) = self.write_index(&index) {
                crate::log_warn!(LOG, "cannot rewrite {}: {err}", self.path(INDEX).display());
            }
        }
        index
    }

    fn sorted(mut records: Vec<Record>) -> Vec<Record> {
        records.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        records
    }

    /// Drops the oldest checkpoints until `keep` remain, never `protect`.
    fn prune(&self, index: &mut Vec<Record>, protect: &str) {
        let mut records = Checkpoints::sorted(index.clone());
        while records.len() > self.keep {
            let Some(at) = records.iter().position(|r| r.name != protect) else {
                break;
            };
            let victim = records.remove(at);
            index.retain(|r| r.name != victim.name);
            let _ = std::fs::remove_file(self.path(&victim.name));
            crate::log_debug!(LOG, "pruned {}", victim.name);
        }
    }

    /// Writes `ckpt-<tag>-<step:06d>.json.gz`, points `latest.json` at it and
    /// prunes.  Saving the same `(tag, step)` twice overwrites the first.
    pub fn save(&self, model: &mut Model, step: i64, tag: &str, metrics: Option<Json>) -> Result<Record, String> {
        if !valid_tag(tag) {
            return Err(format!(
                "invalid checkpoint tag {tag:?} (letters, digits, '_' and '-' only)"
            ));
        }
        if step < 0 {
            return Err(format!("step must be >= 0, got {step}"));
        }
        let suffix = if self.compress { ".json.gz" } else { ".json" };
        let name = format!("{PREFIX}{tag}-{step:06}{suffix}");
        let path = self.path(&name);
        let _held = self.guard();
        model.save(&path.to_string_lossy())?;
        let bytes = std::fs::metadata(&path).map(|m| m.len() as i64).unwrap_or(0);
        let record = Record {
            name: name.clone(),
            path: path.to_string_lossy().into_owned(),
            step,
            tag: tag.to_string(),
            metrics: metrics.unwrap_or(Json::Null),
            saved_at: iso_time(SystemTime::now(), 6),
            bytes,
        };
        let mut index = self.index();
        index.retain(|r| r.name != name);
        index.push(record.clone());
        self.write_json(LATEST, &record.to_json())?;
        self.prune(&mut index, &name);
        self.write_index(&index)?;
        crate::log_info!(LOG, "saved {name} ({bytes} bytes)");
        Ok(record)
    }

    /// Every checkpoint, oldest first by `(step, saved_at, name)`.
    pub fn list(&self) -> Vec<Record> {
        let _held = self.guard();
        Checkpoints::sorted(self.index())
    }

    /// The record `latest.json` points at, while its file is there.
    pub fn latest(&self) -> Option<Record> {
        let _held = self.guard();
        let doc = self.read_json(LATEST)?;
        let name = doc.at("name").as_str()?.to_string();
        let path = self.path(&name);
        if !path.is_file() {
            return None;
        }
        let mut record = Record::from_json(&name, &doc)?;
        record.path = path.to_string_lossy().into_owned();
        Some(record)
    }

    /// The model `latest.json` points at, or `None`.
    pub fn load_latest(&self) -> Result<Option<Model>, String> {
        match self.latest() {
            Some(record) => Model::load(&record.path).map(Some),
            None => Ok(None),
        }
    }

    /// The absolute path of a checkpoint given its name, its bare stem, or a
    /// path; a name with the other suffix (`.json` for a `.json.gz` file, or
    /// the other way round) resolves too.
    pub fn resolve(&self, name_or_path: &str) -> Result<String, String> {
        let stem = SUFFIXES
            .iter()
            .find_map(|s| name_or_path.strip_suffix(s))
            .unwrap_or(name_or_path);
        let mut candidates = vec![PathBuf::from(name_or_path), self.path(name_or_path)];
        candidates.extend(SUFFIXES.iter().map(|s| self.path(&format!("{stem}{s}"))));
        for candidate in candidates {
            if candidate.is_file() {
                let absolute = std::path::absolute(&candidate).unwrap_or(candidate);
                return Ok(absolute.to_string_lossy().into_owned());
            }
        }
        Err(format!("no checkpoint {name_or_path:?} in {}", self.directory()))
    }

    /// A checkpoint by name, stem or path, of whatever kind it holds.
    pub fn load(&self, name_or_path: &str) -> Result<Model, String> {
        Model::load(&self.resolve(name_or_path)?)
    }

    /// Removes a checkpoint; `latest.json` moves to the newest one left.  False
    /// when the name is absent, outside the directory, or not a checkpoint
    /// (`latest.json` and `index.json` cannot be deleted this way).
    pub fn delete(&self, name: &str) -> bool {
        let Ok(path) = self.resolve(name) else { return false };
        let path = PathBuf::from(path);
        if path.parent() != Some(self.dir.as_path()) {
            return false;
        }
        let Some(target) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            return false;
        };
        if parse_name(&target).is_none() {
            return false;
        }
        let _held = self.guard();
        let mut index = self.index();
        index.retain(|r| r.name != target);
        if std::fs::remove_file(&path).is_err() {
            return false;
        }
        let latest = self.read_json(LATEST);
        if latest.as_ref().and_then(|d| d.at("name").as_str()) == Some(target.as_str()) {
            match Checkpoints::sorted(index.clone()).last() {
                Some(newest) => {
                    let _ = self.write_json(LATEST, &newest.to_json());
                }
                None => {
                    let _ = std::fs::remove_file(self.path(LATEST));
                }
            }
        }
        let _ = self.write_index(&index);
        true
    }
}

// -- training that checkpoints ---------------------------------------------------------------------

/// Trains `model` over `texts` and checkpoints it every `every` epochs
/// (tag `epoch`, the step its lifetime epoch count, the epoch's record as the
/// metrics) - Python's `train(checkpoint_manager=...)`.
///
/// Without a manager (or with `every` 0) this is [`Model::train`] itself.
/// With one it trains an epoch at a time, and the model it ends with is the
/// model one call would have ended with: a later epoch finds the structure
/// already built and compressed, so only the texts it would count a second
/// time are put back.  `on_epoch` sees every record after its checkpoint;
/// returning `false` stops the run there (a job's stop button).
pub fn train(
    model: &mut Model,
    texts: &[String],
    opts: &TrainOptions,
    manager: Option<&Checkpoints>,
    every: usize,
    on_epoch: &mut dyn FnMut(&EpochRecord) -> bool,
) -> Result<Vec<EpochRecord>, String> {
    let Some(manager) = manager.filter(|_| every > 0 && opts.epochs > 0) else {
        return model.train(texts, opts);
    };
    let one = TrainOptions {
        epochs: 1,
        ..opts.clone()
    };
    let mut records = Vec::with_capacity(opts.epochs);
    for epoch in 0..opts.epochs {
        // a run counts the texts it reads once, however many epochs it takes
        let counted = (model.meta.trained_texts, model.meta.trained_chars);
        let mut done = model.train(texts, &one)?;
        if epoch > 0 {
            (model.meta.trained_texts, model.meta.trained_chars) = counted;
        }
        let Some(record) = done.pop() else { break };
        if record.epoch % every as i64 == 0 {
            manager.save(model, record.epoch, "epoch", Some(record.to_json()))?;
        }
        let go_on = on_epoch(&record);
        records.push(record);
        if !go_on {
            break;
        }
    }
    Ok(records)
}

/// The checkpoint options a training command takes: `--checkpoint-dir`,
/// `--checkpoint-every` (every epoch or generation when a directory is given
/// and this is not) and `--keep`.
pub struct Schedule {
    pub manager: Option<Checkpoints>,
    pub every: usize,
}

impl Schedule {
    /// Reads the three flags, refusing `--checkpoint-every` without a directory.
    pub fn from_args(args: &Args) -> Result<Schedule, String> {
        let manager = match args.get("checkpoint-dir") {
            Some(dir) if !dir.is_empty() => Some(Checkpoints::new(dir, args.usize("keep", DEFAULT_KEEP)?)?),
            _ => None,
        };
        let every = match args.get("checkpoint-every") {
            None => usize::from(manager.is_some()),
            Some(_) => {
                let every = args.usize("checkpoint-every", 0)?;
                if every > 0 && manager.is_none() {
                    return Err("--checkpoint-every requires --checkpoint-dir".to_string());
                }
                every
            }
        };
        Ok(Schedule { manager, every })
    }

    /// The manager, when checkpoints are on.
    pub fn manager(&self) -> Option<&Checkpoints> {
        self.manager.as_ref().filter(|_| self.every > 0)
    }

    /// `checkpoint_dir` and `checkpoints` of a command's answer.
    pub fn report(&self) -> [(String, Json); 2] {
        match &self.manager {
            Some(m) => [
                ("checkpoint_dir".to_string(), Json::str(m.directory())),
                (
                    "checkpoints".to_string(),
                    Json::Arr(m.list().iter().map(Record::to_json).collect()),
                ),
            ],
            None => [
                ("checkpoint_dir".to_string(), Json::Null),
                ("checkpoints".to_string(), Json::Arr(Vec::new())),
            ],
        }
    }

    /// `train --resume`: the latest checkpoint's model, or `None` (with a
    /// note) when the directory has none.
    pub fn resume(&self, ctx: &Ctx) -> Result<Option<Model>, String> {
        let Some(manager) = &self.manager else {
            return Err("--resume requires --checkpoint-dir".to_string());
        };
        let Some(record) = manager.latest() else {
            crate::log_warn!(LOG, "no checkpoint to resume from in {}", manager.directory());
            return Ok(None);
        };
        let mut model = Model::load(&record.path)?;
        model.workers = ctx.workers;
        model.g.workers = ctx.workers;
        crate::log_info!(LOG, "resuming from {}", record.name);
        Ok(Some(model))
    }
}

// -- the command -------------------------------------------------------------------------------------

/// `radixnet checkpoints [--dir DIR] [--restore NAME|latest] [--out PATH]`:
/// lists a checkpoint directory, or restores one into a model file.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let dir = ctx.args.str("dir", DEFAULT_DIR);
    if !Path::new(&dir).is_dir() {
        return Err(format!("checkpoint directory not found: {dir}"));
    }
    let manager = Checkpoints::new(&dir, ctx.args.usize("keep", DEFAULT_KEEP)?)?;
    let records = manager.list();
    let latest = manager.latest();
    let Some(wanted) = ctx.args.get("restore") else {
        if ctx.out.is_some() {
            return Err("--out requires --restore NAME".to_string());
        }
        ctx.emit(Json::obj([
            ("directory", Json::str(manager.directory())),
            ("checkpoints", Json::Arr(records.iter().map(Record::to_json).collect())),
            ("latest", latest.map(|r| r.to_json()).unwrap_or(Json::Null)),
        ]));
        return Ok(());
    };
    let record = if wanted == "latest" {
        latest.ok_or_else(|| format!("no latest checkpoint in {}", manager.directory()))?
    } else {
        let path = manager.resolve(wanted)?;
        records.into_iter().find(|r| r.path == path).unwrap_or_else(|| Record {
            name: Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            path: path.clone(),
            step: 0,
            tag: String::new(),
            metrics: Json::Null,
            saved_at: String::new(),
            bytes: 0,
        })
    };
    let mut model = Model::load(&record.path)?;
    let out = ctx.save(&mut model)?;
    let bytes = std::fs::metadata(&out).map(|m| m.len() as i64).unwrap_or(0);
    // a record Python did not have in its index is just a name and a path
    let restored = if record.saved_at.is_empty() {
        Json::obj([
            ("name", Json::str(record.name.clone())),
            ("path", Json::str(record.path.clone())),
        ])
    } else {
        record.to_json()
    };
    ctx.emit(Json::obj([
        ("directory", Json::str(manager.directory())),
        ("restored", restored),
        ("out", Json::str(out.clone())),
        (
            "saved",
            Json::obj([("path", Json::str(out)), ("bytes", Json::Int(bytes))]),
        ),
        ("stats", stats(&model)),
    ]));
    Ok(())
}

// -- the server ----------------------------------------------------------------------------------

/// What the server keeps for checkpoints: the manager of `--checkpoint-dir`.
#[derive(Default)]
pub struct State {
    manager: OnceLock<Option<Checkpoints>>,
}

impl State {
    /// Reads the `serve` command's `--checkpoint-dir` and `--keep`.
    pub fn configure(&mut self, args: &Args) -> Result<(), String> {
        if let Some(dir) = args.get("checkpoint-dir").filter(|d| !d.is_empty()) {
            let manager = Checkpoints::new(dir, args.usize("keep", DEFAULT_KEEP)?)?;
            self.manager = OnceLock::from(Some(manager));
        }
        Ok(())
    }
}

/// The server's checkpoint manager: the configured one, else one over
/// `svc.checkpoint_dir` made the first time it is needed.
pub(crate) fn manager(svc: &Service) -> Option<&Checkpoints> {
    svc.checkpoints
        .manager
        .get_or_init(|| {
            let dir = svc.checkpoint_dir.as_deref().filter(|d| !d.is_empty())?;
            Checkpoints::new(dir, DEFAULT_KEEP)
                .map_err(|err| crate::log_error!(LOG, "{err}"))
                .ok()
        })
        .as_ref()
}

/// The manager, or the 400 Python answers without one; `what` names the
/// option that needed it.
pub(crate) fn require<'a>(svc: &'a Service, what: &str) -> Result<&'a Checkpoints, ApiError> {
    manager(svc).ok_or_else(|| {
        ApiError::bad_request(format!(
            "{what} requires a checkpoint directory; start the server with --checkpoint-dir"
        ))
    })
}

/// The train route's work when it checkpoints: [`train`] under the model
/// lock, every epoch reported to the job as it finishes.
pub(crate) fn train_job(
    svc: &Service,
    texts: &[String],
    opts: &TrainOptions,
    every: usize,
) -> Result<Vec<EpochRecord>, String> {
    let manager = manager(svc).filter(|_| every > 0);
    svc.with_model(|m| {
        train(m, texts, opts, manager, every, &mut |record| {
            if manager.is_some() {
                svc.job_progress(record.to_json());
            }
            !svc.stopping()
        })
    })
}

/// `GET /api/checkpoints`: every record, and the one `latest.json` points at.
fn list_route(svc: &Arc<Service>, _r: &Request) -> Answer {
    let Some(manager) = manager(svc) else {
        return Ok(Json::obj([
            ("checkpoints", Json::Arr(Vec::new())),
            ("latest", Json::Null),
        ]));
    };
    Ok(Json::obj([
        (
            "checkpoints",
            Json::Arr(manager.list().iter().map(Record::to_json).collect()),
        ),
        ("latest", manager.latest().map(|r| r.to_json()).unwrap_or(Json::Null)),
    ]))
}

/// `POST /api/checkpoints/save {tag}`: checkpoints the running model - the
/// step its lifetime epoch count, the metrics its last record.
fn save_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let manager = require(svc, "checkpoints")?;
    let tag = r.text("tag", "manual");
    let tag = if tag.is_empty() { "manual".to_string() } else { tag };
    let record = svc.with_model(|m| {
        let metrics = m.history.last().map(|h| h.to_json());
        manager.save(m, m.meta.epochs_total.value, &tag, metrics)
    })?;
    Ok(record.to_json())
}

/// `POST /api/checkpoints/restore {name}`: makes a checkpoint the running
/// model; the evolve loop's discriminator and history go with the model they
/// belonged to.
fn restore_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let manager = require(svc, "checkpoints")?;
    let name = r.text("name", "");
    if name.is_empty() {
        return Err(ApiError::bad_request("'name' must not be empty"));
    }
    svc.ensure_idle()?;
    let path = manager.resolve(&name).map_err(ApiError::not_found)?;
    let loaded = Model::load(&path)?;
    let doc = svc.with_model(|m| {
        let workers = m.workers;
        *m = loaded;
        m.workers = workers;
        m.g.workers = workers;
        stats(m)
    });
    svc.evolve.forget();
    crate::log_info!(LOG, "restored {path}");
    Ok(doc)
}

/// This area's routes:
/// `GET /api/checkpoints`
/// `POST /api/checkpoints/save`
/// `POST /api/checkpoints/restore`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/checkpoints", list_route);
    server.route("POST", "/api/checkpoints/save", save_route);
    server.route("POST", "/api/checkpoints/restore", restore_route);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphOptions;

    fn temp_dir(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("radixnet-ckpt-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_string_lossy().into_owned()
    }

    fn texts() -> Vec<String> {
        [
            "the cat sat on the mat",
            "the dog sat on the log",
            "a bird flew over the hill",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn trained() -> Model {
        let mut m = Model::new(1, GraphOptions::default()).unwrap();
        m.train(
            &texts(),
            &TrainOptions {
                epochs: 1,
                ..Default::default()
            },
        )
        .unwrap();
        m
    }

    #[test]
    fn names_tags_and_times() {
        assert_eq!(parse_name("ckpt-epoch-000003.json.gz"), Some(("epoch".to_string(), 3)));
        assert_eq!(
            parse_name("ckpt-my-tag-1234567.json"),
            Some(("my-tag".to_string(), 1_234_567))
        );
        assert_eq!(parse_name("ckpt-epoch-003.json"), None, "six digits at least");
        assert_eq!(parse_name("ckpt--000003.json"), None);
        assert_eq!(parse_name("latest.json"), None);
        assert!(valid_tag("gen") && valid_tag("a_b-1") && !valid_tag("-x") && !valid_tag("") && !valid_tag("a b"));
        let t = UNIX_EPOCH + std::time::Duration::new(1_000_000_000, 123_456_789);
        assert_eq!(iso_time(t, 6), "2001-09-09T01:46:40.123456+00:00");
        assert_eq!(iso_time(t, 3), "2001-09-09T01:46:40.123+00:00");
        assert_eq!(iso_time(t, 0), "2001-09-09T01:46:40+00:00");
        let doc = Json::obj([("b", Json::Int(1)), ("a", Json::str("é"))]);
        assert_eq!(python_pretty(&doc), "{\n  \"a\": \"\\u00e9\",\n  \"b\": 1\n}");
    }

    #[test]
    fn save_list_latest_and_prune() {
        let dir = temp_dir("prune");
        let manager = Checkpoints::new(&dir, 2).unwrap();
        let mut m = trained();
        for step in 1..=4 {
            let record = manager
                .save(&mut m, step, "epoch", Some(Json::obj([("loss", Json::Num(0.5))])))
                .unwrap();
            assert_eq!(record.name, format!("ckpt-epoch-{step:06}.json.gz"));
            assert!(record.bytes > 0);
        }
        let names: Vec<String> = manager.list().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["ckpt-epoch-000003.json.gz", "ckpt-epoch-000004.json.gz"]);
        assert_eq!(manager.latest().unwrap().name, "ckpt-epoch-000004.json.gz");
        assert!(!Path::new(&dir).join("ckpt-epoch-000001.json.gz").exists());
        // a save of an older step is still the one kept: never prune the latest
        manager.save(&mut m, 0, "manual", None).unwrap();
        let names: Vec<String> = manager.list().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["ckpt-manual-000000.json.gz", "ckpt-epoch-000004.json.gz"]);
        assert!(manager.save(&mut m, 1, "bad tag", None).is_err());
        assert!(manager.save(&mut m, -1, "epoch", None).is_err());
        assert!(Checkpoints::new(&dir, 0).is_err());
    }

    #[test]
    fn the_files_win_over_the_index() {
        let dir = temp_dir("reconcile");
        let manager = Checkpoints::new(&dir, 5).unwrap();
        let mut m = trained();
        manager.save(&mut m, 1, "epoch", None).unwrap();
        // a file copied in by hand is recorded; one removed by hand is dropped
        std::fs::copy(
            Path::new(&dir).join("ckpt-epoch-000001.json.gz"),
            Path::new(&dir).join("ckpt-hand-000007.json.gz"),
        )
        .unwrap();
        std::fs::write(Path::new(&dir).join("ckpt-nonsense.json"), "{}").unwrap();
        std::fs::remove_file(Path::new(&dir).join("ckpt-epoch-000001.json.gz")).unwrap();
        let records = manager.list();
        assert_eq!(records.len(), 1);
        assert_eq!((records[0].tag.as_str(), records[0].step), ("hand", 7));
        assert!(records[0].metrics.is_null());
        assert!(manager.latest().is_none(), "latest.json points at a file that is gone");
        let index = crate::file::read_document(&format!("{dir}/index.json")).unwrap();
        assert!(index.get("ckpt-hand-000007.json.gz").is_some());
    }

    #[test]
    fn resolve_load_and_delete() {
        let dir = temp_dir("resolve");
        let manager = Checkpoints::new(&dir, 5).unwrap();
        let mut m = trained();
        manager.save(&mut m, 1, "epoch", None).unwrap();
        manager.save(&mut m, 2, "epoch", None).unwrap();
        let full = manager.resolve("ckpt-epoch-000001.json.gz").unwrap();
        assert_eq!(manager.resolve("ckpt-epoch-000001").unwrap(), full);
        assert_eq!(
            manager.resolve("ckpt-epoch-000001.json").unwrap(),
            full,
            "either suffix"
        );
        assert_eq!(manager.resolve(&full).unwrap(), full);
        assert!(manager.resolve("nope").is_err());
        let loaded = manager.load("ckpt-epoch-000002").unwrap();
        assert_eq!(loaded.g.num_nodes(), m.g.num_nodes());
        assert!(!manager.delete("latest.json"));
        assert!(!manager.delete("index.json"));
        assert!(manager.delete("ckpt-epoch-000002"));
        assert_eq!(
            manager.latest().unwrap().name,
            "ckpt-epoch-000001.json.gz",
            "latest moves back"
        );
        assert!(manager.delete("ckpt-epoch-000001.json.gz"));
        assert!(manager.latest().is_none());
        assert!(manager.list().is_empty());
    }

    #[test]
    fn a_run_that_checkpoints_ends_where_one_that_does_not_ends() {
        let dir = temp_dir("train");
        let manager = Checkpoints::new(&dir, 10).unwrap();
        let opts = TrainOptions {
            epochs: 3,
            ..Default::default()
        };
        let mut plain = Model::new(1, GraphOptions::default()).unwrap();
        let expected = plain.train(&texts(), &opts).unwrap();
        let mut stepped = Model::new(1, GraphOptions::default()).unwrap();
        stepped.meta.created = plain.meta.created.clone();
        let mut seen = 0;
        let got = train(&mut stepped, &texts(), &opts, Some(&manager), 2, &mut |_| {
            seen += 1;
            true
        })
        .unwrap();
        assert_eq!(seen, 3);
        let strip = |r: &EpochRecord| {
            let mut r = r.clone();
            r.seconds = 0.0;
            r.to_json()
        };
        assert_eq!(
            got.iter().map(strip).collect::<Vec<_>>(),
            expected.iter().map(strip).collect::<Vec<_>>()
        );
        let (mut a, mut b) = (plain.to_doc(), stepped.to_doc());
        // the history's timings are the one thing two runs never share
        for doc in [&mut a, &mut b] {
            if let Json::Obj(pairs) = doc {
                pairs.retain(|(k, _)| k != "history" && k != "saved_at");
            }
        }
        assert_eq!(a, b, "the same model, byte for byte");
        let names: Vec<String> = manager.list().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["ckpt-epoch-000002.json.gz"], "every second epoch");
        let first = manager.list()[0].metrics.clone();
        assert_eq!(first.at("epoch").as_i64(), Some(2));
        // and a stop after the first epoch is honoured
        let mut stopped = Model::new(1, GraphOptions::default()).unwrap();
        let got = train(&mut stopped, &texts(), &opts, Some(&manager), 1, &mut |_| false).unwrap();
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn the_checkpoint_options_of_a_command() {
        let dir = temp_dir("schedule");
        let args = |items: &[&str]| {
            let argv: Vec<String> = items.iter().map(|s| s.to_string()).collect();
            crate::cli::parse_args(&argv).unwrap().1
        };
        let off = Schedule::from_args(&args(&["train"])).unwrap();
        assert!(off.manager().is_none() && off.every == 0);
        let on = Schedule::from_args(&args(&["train", "--checkpoint-dir", &dir])).unwrap();
        assert_eq!(on.every, 1, "every epoch when a directory is given");
        assert!(on.manager().is_some());
        let every =
            Schedule::from_args(&args(&["train", "--checkpoint-dir", &dir, "--checkpoint-every", "3"])).unwrap();
        assert_eq!(every.every, 3);
        assert!(Schedule::from_args(&args(&["train", "--checkpoint-every", "2"])).is_err());
        let zero = Schedule::from_args(&args(&["train", "--checkpoint-every", "0"])).unwrap();
        assert!(zero.manager().is_none());
    }

    #[test]
    fn the_routes_answer_the_python_json() {
        let dir = temp_dir("routes");
        let mut svc = Service::new(trained(), String::new(), 0, 1);
        svc.checkpoint_dir = Some(dir.clone());
        let svc = Arc::new(svc);
        let empty = list_route(&svc, &Request::json("GET", "/api/checkpoints", Json::Null)).unwrap();
        assert_eq!(empty.at("checkpoints").as_array().len(), 0);
        assert!(empty.at("latest").is_null());
        let saved = save_route(&svc, &Request::json("POST", "/", Json::obj([("tag", Json::str(""))]))).unwrap();
        assert_eq!(saved.at("name").as_str(), Some("ckpt-manual-000001.json.gz"));
        assert_eq!(saved.at("step").as_i64(), Some(1));
        assert_eq!(saved.at("metrics").at("epoch").as_i64(), Some(1));
        let listing = list_route(&svc, &Request::json("GET", "/", Json::Null)).unwrap();
        assert_eq!(listing.at("latest").at("name"), saved.at("name"));
        let before = svc.with_model(|m| m.g.num_nodes());
        svc.with_model(|m| {
            m.train(
                &["an entirely new sentence".to_string()],
                &TrainOptions {
                    epochs: 1,
                    ..Default::default()
                },
            )
            .unwrap()
        });
        let restored = restore_route(
            &svc,
            &Request::json("POST", "/", Json::obj([("name", Json::str("ckpt-manual-000001"))])),
        )
        .unwrap();
        assert_eq!(restored.at("nodes").as_i64(), Some(before as i64));
        let missing = restore_route(
            &svc,
            &Request::json("POST", "/", Json::obj([("name", Json::str("nope"))])),
        );
        assert_eq!(missing.unwrap_err().status, 404);
        let blank = restore_route(&svc, &Request::json("POST", "/", Json::obj([("name", Json::str(""))])));
        assert_eq!(blank.unwrap_err().status, 400);
        let none = Arc::new(Service::new(trained(), String::new(), 0, 1));
        let refused = save_route(&none, &Request::json("POST", "/", Json::Null)).unwrap_err();
        assert!(refused.message.contains("--checkpoint-dir"), "{}", refused.message);
    }
}
