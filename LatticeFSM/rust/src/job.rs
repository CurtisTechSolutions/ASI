//! Background jobs: training that takes a while runs on its own thread, reporting how far it has got, and can be
//! stopped; the server answers everything else meanwhile.
//!
//! The model code reports through this thread's job, when there is one (`tick`, `phase`, `stopped`) - a few atomic
//! operations, and nothing at all on a thread without a job (the command line, the tests).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::json::Json;

/// How far a job has got.
#[derive(Default)]
pub struct Progress {
    pub sequences: AtomicU64,
    pub tokens: AtomicU64,
    pub round: AtomicU64,
    pub stop: AtomicBool,
    /// What it is doing ("training", "drilling", ...) and on what (a file, an archive's member).
    pub phase: Mutex<String>,
    pub label: Mutex<String>,
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Progress>>> = const { RefCell::new(None) };
}

/// Run `f` with `p` as this thread's job.
pub fn with_progress<T>(p: Arc<Progress>, f: impl FnOnce() -> T) -> T {
    CURRENT.with(|c| *c.borrow_mut() = Some(p));
    let out = f();
    CURRENT.with(|c| *c.borrow_mut() = None);
    out
}

fn current<T>(f: impl FnOnce(&Progress) -> T) -> Option<T> {
    CURRENT.with(|c| c.borrow().as_ref().map(|p| f(p)))
}

/// Whether this thread runs a job.
pub fn in_job() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// One more sequence of `tokens` learned, from `label`.
pub fn tick(label: &str, tokens: usize) {
    current(|p| {
        p.sequences.fetch_add(1, Ordering::Relaxed);
        p.tokens.fetch_add(tokens as u64, Ordering::Relaxed);
        if let Ok(mut l) = p.label.try_lock() {
            if l.as_str() != label {
                l.clear();
                l.push_str(label);
            }
        }
    });
}

/// What the job is doing now, and (for rounds) which round.
pub fn phase(name: &str, round: u64) {
    current(|p| {
        if let Ok(mut ph) = p.phase.lock() {
            ph.clear();
            ph.push_str(name);
        }
        p.round.store(round, Ordering::Relaxed);
    });
}

/// Whether this thread's job has been asked to stop.
pub fn stopped() -> bool {
    current(|p| p.stop.load(Ordering::Relaxed)).unwrap_or(false)
}

/// The message a stopped job's work returns.
pub const STOPPED: &str = "stopped on request";

/// A job: what it is, how far it has got, and how it ended.
pub struct Job {
    pub id: u64,
    pub kind: String,
    pub model: String,
    pub progress: Arc<Progress>,
    pub started: Instant,
    pub started_at: f64,
    pub finished: Option<f64>,
    pub state: String,
    pub result: Option<Json>,
    pub error: Option<String>,
}

impl Job {
    pub fn new(id: u64, kind: &str, model: &str) -> Job {
        Job {
            id,
            kind: kind.to_string(),
            model: model.to_string(),
            progress: Arc::new(Progress::default()),
            started: Instant::now(),
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            finished: None,
            state: "running".to_string(),
            result: None,
            error: None,
        }
    }

    pub fn running(&self) -> bool {
        self.state == "running"
    }

    pub fn to_json(&self) -> Json {
        let p = &self.progress;
        let elapsed = match self.finished {
            Some(f) => f - self.started_at,
            None => self.started.elapsed().as_secs_f64(),
        };
        let tokens = p.tokens.load(Ordering::Relaxed);
        Json::object()
            .with("id", (self.id as f64).into())
            .with("kind", self.kind.as_str().into())
            .with("model", self.model.as_str().into())
            .with("state", self.state.as_str().into())
            .with("started_at", self.started_at.into())
            .with("elapsed", elapsed.into())
            .with(
                "progress",
                Json::object()
                    .with(
                        "phase",
                        p.phase.lock().map(|s| s.clone()).unwrap_or_default().as_str().into(),
                    )
                    .with(
                        "label",
                        p.label.lock().map(|s| s.clone()).unwrap_or_default().as_str().into(),
                    )
                    .with("sequences", (p.sequences.load(Ordering::Relaxed) as f64).into())
                    .with("tokens", (tokens as f64).into())
                    .with("round", (p.round.load(Ordering::Relaxed) as f64).into())
                    .with(
                        "tokens_per_second",
                        if elapsed > 0.0 {
                            (tokens as f64 / elapsed).into()
                        } else {
                            0.0.into()
                        },
                    )
                    .with("stopping", p.stop.load(Ordering::Relaxed).into()),
            )
            .with("result", self.result.clone().unwrap_or(Json::Null))
            .with("error", self.error.as_deref().map(Json::from).unwrap_or(Json::Null))
    }
}
