//! The HTTP API and the page that uses it (`DESIGN.md` §16).
//!
//! `gamelattice serve` holds one [`Body`] reading a language and one [`NimBody`] behind locks, and answers JSON
//! under `/api/`; anything else is served from the prebuilt React frontend (`../frontend/dist`), which shows the
//! stage game naming a state step by step, the lattice it names states of, the encoder's economy, Nim, and the
//! measured results.  The server is `latticefsm::http` as it is - the sibling's dependency-free HTTP/1.1 - and its
//! rules are the sibling's (`LatticeFSM/rust/src/server.rs`): an unknown path outside `/api/` is one of the
//! single-page app's own routes and gets `index.html`; a path that tries to climb out of the directory is
//! refused; without a frontend directory the root answers a JSON 404 that says so.
//!
//! **Locks.** Outside a job every request takes the lock it needs only when it is free, and answers 409
//! `{busy: true, error, job}` when it is not, so nothing waits behind a long training run.  **Jobs.** A `POST`
//! with `"job": true` runs on its own thread and answers `{job}` at once; `GET /api/job` reports its progress
//! (and, for training, the held-out curve so far), its result or its error; `POST /api/job/stop` asks it to stop
//! at the next episode or game, keeping what it learned.  One job at a time: a second is refused with 409.
//! Everything is deterministic given the seed.
//!
//! | route | does |
//! |---|---|
//! | `GET /api/health` | the version, the language, the lattice's shape, the encoder's settings |
//! | `GET /api/status` | everything in brief: settings, episodes, diversity, wealth, thinks, the lattice's stats, agreement, Nim, the job |
//! | `GET /api/languages` | the four languages |
//! | `POST /api/new` | `{language?, states?, window?, n?, seed?, encoder_sets_state?, forward_lessons?, punish?, whole_run?, label_set?, nim?}`: a fresh body |
//! | `POST /api/train` | `{episodes?, max_length?, job?}`: random strings, every step credited; held-out accuracy every tenth on a fixed 300 |
//! | `POST /api/classify` | `{text}`: quiet - the verdicts and the naming trace |
//! | `POST /api/run` | `{text}`: one learning episode on `text` - the verdicts and the trace, Shapley values included |
//! | `GET /api/matrix?symbol=a` | the lattice's `S × S` slice for one symbol (the shape `latticefsm serve` answers) |
//! | `GET /api/edge?source=&symbol=&target=` | one edge, every field |
//! | `GET /api/table` | the greedy transition table and the accepting states |
//! | `GET /api/encoder` | the graph, the named-state histogram, diversity, wealth, thinks, the richest nodes |
//! | `GET /api/nim` | the Nim body in brief: settings, games, `I(state; nim-sum)`, states used, agreement |
//! | `POST /api/nim/new` | `{states?, window?, seed?, grade?}`: a fresh Nim body |
//! | `POST /api/nim/babble` | `{games?, job?}`: random games; the forward model learns |
//! | `POST /api/nim/play` | `{games?, first?, job?}`: games against a random opponent; per game and in total |
//! | `GET /api/results` | `results/e1.json` and `e2.json` as they are, when present |
//! | `GET /api/job`, `POST /api/job/stop` | the running or last job; ask it to stop |

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use latticefsm::http::{self, Handler, Request, Response};
use latticefsm::job::{self, Job};
use latticefsm::json::{parse, Json};
use latticefsm::languages::{examples, language, random_string, Language, LANGUAGES};
use latticefsm::machine::Machine;
use latticefsm::rng::Rng;
use radixnet::FIRST;

use crate::body::{Body, BodySettings, Classification, GameRecord, NimBody, NimSettings, TraceSeat, TraceStep};
use crate::VERSION;

/// How many held-out examples every accuracy is measured on.
pub const TEST_SIZE: usize = 300;

/// How a busy lock's refusal begins.
const BUSY: &str = "busy:";

/// The body that reads a language, with what measures it: the fixed held-out set and the generator the training
/// strings come from (both drawn from the seed, as `gamelattice train` draws them), and the curve so far.
pub struct LanguageBody {
    pub body: Body,
    pub language: Language,
    pub test: Vec<(String, bool)>,
    pub rng: Rng,
    /// `[episodes, encoder accuracy, lattice accuracy]` at every measurement since the body was made.
    pub history: Vec<[f64; 3]>,
}

impl LanguageBody {
    pub fn new(lang: Language, settings: BodySettings) -> Result<LanguageBody, String> {
        let mut rng = Rng::new(settings.seed.wrapping_mul(1000).wrapping_add(17));
        let test = examples(&lang, TEST_SIZE, &mut rng, 6);
        Ok(LanguageBody {
            body: Body::for_language(&lang, settings)?,
            language: lang,
            test,
            rng,
            history: Vec::new(),
        })
    }

    /// Held-out accuracy of the encoder's verdict and of the lattice's own; nothing moves (the agreement count a
    /// quiet read notes is put back).
    pub fn accuracy(&mut self) -> (f64, f64) {
        let agreement = self.body.agreement.clone();
        let acc = self.body.accuracy(&self.test);
        self.body.agreement = agreement;
        acc
    }
}

/// What `serve` starts with.
#[derive(Clone, Debug)]
pub struct ServeSettings {
    pub language: String,
    pub body: BodySettings,
    pub nim: NimSettings,
    pub frontend: Option<PathBuf>,
    pub results: PathBuf,
}

impl Default for ServeSettings {
    fn default() -> ServeSettings {
        ServeSettings {
            language: "even-b".to_string(),
            body: BodySettings::default(),
            nim: NimSettings::default(),
            frontend: None,
            results: PathBuf::from("../results"),
        }
    }
}

/// The two bodies behind the server, the job slot, and where the frontend and the results are.
pub struct Service {
    pub body: Mutex<LanguageBody>,
    pub nim: Mutex<NimBody>,
    /// The background job, running or the last one (`/api/job`).
    pub job: Mutex<Option<Job>>,
    pub next_job: AtomicU64,
    /// The held-out curve of the training job running now, readable while the body is locked by it.
    pub curve: Mutex<Vec<[f64; 3]>>,
    pub frontend: Option<PathBuf>,
    pub results: PathBuf,
}

impl Service {
    pub fn new(settings: ServeSettings) -> Result<Arc<Service>, String> {
        let lang = language(&settings.language)?;
        Ok(Arc::new(Service {
            body: Mutex::new(LanguageBody::new(lang, settings.body)?),
            nim: Mutex::new(NimBody::new(settings.nim)?),
            job: Mutex::new(None),
            next_job: AtomicU64::new(0),
            curve: Mutex::new(Vec::new()),
            frontend: settings.frontend,
            results: settings.results,
        }))
    }

    /// The handler the HTTP server calls.
    pub fn handler(self: &Arc<Service>) -> Arc<Handler> {
        let service = Arc::clone(self);
        Arc::new(move |req: &Request| service.route(req))
    }

    /// A lock: waited for on a job's own thread; anywhere else taken only when it is free, so a request to a body
    /// a job is training is answered at once (`busy: ...`) instead of waiting behind it.
    fn take<'a, T>(&self, m: &'a Mutex<T>, what: &str) -> Result<MutexGuard<'a, T>, String> {
        if job::in_job() {
            return m.lock().map_err(|_| format!("{what}'s lock is poisoned"));
        }
        match m.try_lock() {
            Ok(g) => Ok(g),
            Err(TryLockError::WouldBlock) => {
                let doing = self
                    .job
                    .lock()
                    .ok()
                    .and_then(|j| {
                        j.as_ref()
                            .filter(|j| j.running())
                            .map(|j| format!("job {} ({} {})", j.id, j.model, j.kind))
                    })
                    .unwrap_or_else(|| "another request".to_string());
                Err(format!(
                    "{BUSY} {what} is in use by {doing} - try again when it is done (GET /api/job), or stop it (POST /api/job/stop)"
                ))
            }
            Err(TryLockError::Poisoned(_)) => Err(format!("{what}'s lock is poisoned")),
        }
    }

    fn job_json(&self) -> Json {
        self.job
            .lock()
            .ok()
            .and_then(|j| j.as_ref().map(Job::to_json))
            .unwrap_or(Json::Null)
    }

    /// A route's result as a response: 409 with the job for a busy lock, 404 and 405 for routes that are not
    /// there, 400 for every other error.
    fn answer(&self, r: Result<Json, String>) -> Response {
        match r {
            Ok(v) => Response::ok(&v),
            Err(e) if e.starts_with(BUSY) => Response::json(
                409,
                &Json::object()
                    .with("error", e.as_str().into())
                    .with("busy", true.into())
                    .with("job", self.job_json()),
            ),
            Err(e) if e == "no such route" => Response::error(404, &e),
            Err(e) if e == "method not allowed" => Response::error(405, &e),
            Err(e) => Response::error(400, &e),
        }
    }

    /// `POST` with `"job": true`: the same request on its own thread; `{job}` at once, its progress, result or
    /// error at `GET /api/job`.  One job at a time.
    fn start_job(self: &Arc<Self>, req: &Request) -> Response {
        let action = req.path.trim_start_matches("/api/");
        let (model, kind) = match action.split_once('/') {
            Some((m, k)) => (m.to_string(), k.to_string()),
            None => ("body".to_string(), action.to_string()),
        };
        let mut slot = match self.job.lock() {
            Ok(s) => s,
            Err(_) => return Response::error(500, "the job's lock is poisoned"),
        };
        if let Some(j) = slot.as_ref().filter(|j| j.running()) {
            return Response::json(
                409,
                &Json::object()
                    .with(
                        "error",
                        format!("{BUSY} job {} ({} {}) is still running", j.id, j.model, j.kind)
                            .as_str()
                            .into(),
                    )
                    .with("busy", true.into())
                    .with("job", j.to_json()),
            );
        }
        let id = self.next_job.fetch_add(1, Ordering::Relaxed) + 1;
        let new = Job::new(id, &kind, &model);
        let progress = Arc::clone(&new.progress);
        let summary = new.to_json();
        *slot = Some(new);
        drop(slot);
        if let Ok(mut c) = self.curve.lock() {
            c.clear();
        }
        let body = match &req.body {
            Json::Object(items) => Json::Object(items.iter().filter(|(k, _)| k != "job").cloned().collect()),
            other => other.clone(),
        };
        let inner = Request {
            method: req.method.clone(),
            path: req.path.clone(),
            query: req.query.clone(),
            body,
        };
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            let stop = Arc::clone(&progress);
            // a panic must not leave the job "running" for ever: it ends as failed, with the panic's message
            let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                job::with_progress(progress, || me.route(&inner))
            }))
            .unwrap_or_else(|e| {
                let msg = e
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "it panicked".to_string());
                Response::error(500, &format!("the job failed: {msg}"))
            });
            let parsed = parse(&String::from_utf8_lossy(&response.body)).unwrap_or(Json::Null);
            if let Ok(mut slot) = me.job.lock() {
                if let Some(j) = slot.as_mut().filter(|j| j.id == id) {
                    j.finished = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0),
                    );
                    if response.status == 200 {
                        // a stopped job that kept what it learned answers 200 with `stopped: true`
                        j.state = if parsed.bool_or("stopped", false) {
                            "stopped".to_string()
                        } else {
                            "done".to_string()
                        };
                        j.result = Some(parsed);
                    } else {
                        let msg = parsed.str_or("error", "failed").to_string();
                        j.state = if stop.stop.load(Ordering::Relaxed) || msg.contains(job::STOPPED) {
                            "stopped".to_string()
                        } else {
                            "failed".to_string()
                        };
                        j.error = Some(msg);
                    }
                }
            }
        });
        Response::ok(&Json::object().with("job", summary))
    }

    /// `GET /api/job` (the running or last job, with the held-out curve of a training job), `POST /api/job/stop`.
    fn job_route(&self, req: &Request) -> Response {
        let mut slot = match self.job.lock() {
            Ok(s) => s,
            Err(_) => return Response::error(500, "the job's lock is poisoned"),
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/api/job") => Response::ok(&match slot.as_ref() {
                Some(j) => {
                    let curve = self.curve.lock().map(|c| curve_json(&c)).unwrap_or(Json::Null);
                    j.to_json().with("curve", curve)
                }
                None => Json::Null,
            }),
            ("POST", "/api/job/stop") => match slot.as_mut().filter(|j| j.running()) {
                Some(j) => {
                    j.progress.stop.store(true, Ordering::Relaxed);
                    Response::ok(&j.to_json())
                }
                None => Response::error(400, "no job is running"),
            },
            ("GET", _) | ("POST", _) => Response::error(404, "no such route"),
            _ => Response::error(405, "method not allowed"),
        }
    }

    pub fn route(self: &Arc<Self>, req: &Request) -> Response {
        if !req.path.starts_with("/api/") {
            return self.serve_static(req);
        }
        if req.path == "/api/job" || req.path.starts_with("/api/job/") {
            return self.job_route(req);
        }
        if req.method == "POST" && req.body.bool_or("job", false) && !job::in_job() {
            return self.start_job(req);
        }
        let b = &req.body;
        let result = match (req.method.as_str(), &req.path[5..]) {
            ("GET", "health") => self.with_body(|lb| Ok(health(lb))),
            ("GET", "status") => Ok(self.status()),
            ("GET", "languages") => Ok(languages()),
            ("POST", "new") => self.new_body(b),
            ("POST", "train") => self.with_body(|lb| self.train(lb, b)),
            ("POST", "classify") => self.with_body(|lb| {
                let text = text_of(b, lb)?;
                let (c, steps) = lb.body.trace(&text);
                Ok(read_json(lb, &text, &c, &steps, false))
            }),
            ("POST", "run") => self.with_body(|lb| {
                let text = text_of(b, lb)?;
                let (c, steps) = lb.body.run_traced(&text);
                Ok(read_json(lb, &text, &c, &steps, true))
            }),
            ("GET", "matrix") => self.with_body(|lb| matrix(&lb.body.lattice.machine, req.param("symbol"))),
            ("GET", "edge") => self.with_body(|lb| edge(&lb.body.lattice.machine, req)),
            ("GET", "table") => self.with_body(|lb| Ok(table(&lb.body.lattice.machine))),
            ("GET", "encoder") => self.with_body(|lb| Ok(encoder(lb, req.param("top")))),
            ("POST", "nim/new") => self.take(&self.nim, "the Nim body").and_then(|mut nim| {
                *nim = NimBody::new(nim_settings(&nim.settings, b))?;
                Ok(nim_info(&nim))
            }),
            ("POST", "nim/babble") => self
                .take(&self.nim, "the Nim body")
                .and_then(|mut nim| nim_babble(&mut nim, b)),
            ("POST", "nim/play") => self
                .take(&self.nim, "the Nim body")
                .and_then(|mut nim| nim_play(&mut nim, b)),
            ("GET", "nim") => self.take(&self.nim, "the Nim body").map(|nim| nim_info(&nim)),
            ("GET", "results") => Ok(self.results()),
            ("GET", _) | ("POST", _) => Err("no such route".to_string()),
            _ => Err("method not allowed".to_string()),
        };
        self.answer(result)
    }

    fn with_body(&self, f: impl FnOnce(&mut LanguageBody) -> Result<Json, String>) -> Result<Json, String> {
        let mut g = self.take(&self.body, "the body")?;
        f(&mut g)
    }

    /// Everything in brief; a body a job holds reads `{busy: true}`.
    fn status(&self) -> Json {
        let busy = || Json::object().with("busy", true.into());
        let body = self.body.try_lock().ok().map(|lb| body_status(&lb));
        let nim = self.nim.try_lock().ok().map(|n| nim_info(&n));
        Json::object()
            .with("version", VERSION.into())
            .with("body", body.unwrap_or_else(busy))
            .with("nim", nim.unwrap_or_else(busy))
            .with("job", self.job_json())
    }

    /// `POST /api/new`: a fresh body; whatever is not given is the body's now.  With `nim`, a fresh Nim body too.
    fn new_body(&self, b: &Json) -> Result<Json, String> {
        let mut lb = self.take(&self.body, "the body")?;
        let lang = match b.get("language").and_then(Json::as_str) {
            Some(name) => language(name)?,
            None => lb.language,
        };
        let settings = body_settings(&lb.body.settings, b);
        let fresh = LanguageBody::new(lang, settings)?;
        if let Some(n) = b.get("nim").filter(|n| matches!(n, Json::Object(_))) {
            let mut nim = self.take(&self.nim, "the Nim body")?;
            *nim = NimBody::new(nim_settings(&nim.settings, n))?;
        }
        *lb = fresh;
        Ok(body_status(&lb))
    }

    /// `POST /api/train`: `episodes` random strings, each step credited by the oracle; held-out accuracy before,
    /// every tenth of the way and after.  As a job it reports each episode and the curve, and stops on request
    /// keeping what it learned.
    fn train(&self, lb: &mut LanguageBody, b: &Json) -> Result<Json, String> {
        let episodes = b.num("episodes", 1000.0).max(0.0) as usize;
        let max_length = b.num("max_length", 6.0).clamp(0.0, 64.0) as usize;
        let before = lb.accuracy();
        if lb.history.is_empty() {
            lb.history.push([lb.body.episodes as f64, before.0, before.1]);
        }
        let mut curve = vec![[lb.body.episodes as f64, before.0, before.1]];
        self.publish(&curve);
        let every = (episodes / 10).max(1);
        let mut stopped = false;
        job::phase("training", 0);
        for ep in 1..=episodes {
            if job::stopped() {
                stopped = true;
                break;
            }
            let s = random_string(&mut lb.rng, max_length);
            lb.body.run(&s, true);
            job::tick(lb.language.name, s.chars().count());
            if ep % every == 0 || ep == episodes {
                let (enc, lat) = lb.accuracy();
                let point = [lb.body.episodes as f64, enc, lat];
                curve.push(point);
                lb.history.push(point);
                self.publish(&curve);
                job::phase("training", (ep / every) as u64);
            }
        }
        let after = lb.accuracy();
        if stopped {
            let point = [lb.body.episodes as f64, after.0, after.1];
            curve.push(point);
            lb.history.push(point);
            self.publish(&curve);
        }
        Ok(Json::object()
            .with("language", lb.language.name.into())
            .with("episodes", episodes.into())
            .with("stopped", stopped.into())
            .with("test", lb.test.len().into())
            .with("before", accuracy_json(before))
            .with("after", accuracy_json(after))
            .with("curve", curve_json(&curve))
            .with("history", curve_json(&lb.history))
            .with("status", body_status(lb)))
    }

    fn publish(&self, curve: &[[f64; 3]]) {
        if job::in_job() {
            if let Ok(mut c) = self.curve.lock() {
                *c = curve.to_vec();
            }
        }
    }

    /// `GET /api/results`: the experiments' files as they were written, `null` where one is missing.
    fn results(&self) -> Json {
        let mut out = Json::object().with("dir", self.results.display().to_string().as_str().into());
        let mut missing: Vec<String> = Vec::new();
        for name in ["e1", "e2"] {
            let path = self.results.join(format!("{name}.json"));
            match std::fs::read_to_string(&path) {
                Ok(text) => match parse(&text) {
                    Ok(doc) => out.set(name, doc),
                    Err(e) => {
                        out.set(name, Json::Null);
                        missing.push(format!("{}: {e}", path.display()));
                    }
                },
                Err(_) => {
                    out.set(name, Json::Null);
                    missing.push(path.display().to_string());
                }
            }
        }
        out.with("missing", Json::strings(&missing))
    }

    fn serve_static(&self, req: &Request) -> Response {
        let Some(root) = &self.frontend else {
            return Response::error(
                404,
                "no frontend directory is being served: build ../frontend (make frontend-build) or pass --frontend-dir",
            );
        };
        if req.method != "GET" && req.method != "HEAD" {
            return Response::error(405, "method not allowed");
        }
        let relative = req.path.trim_start_matches('/');
        let mut file = safe_join(root, relative);
        if file.as_ref().is_none_or(|p| p.is_dir()) {
            file = safe_join(root, "index.html");
        }
        match file.and_then(|path| std::fs::read(&path).ok().map(|bytes| (path, bytes))) {
            Some((path, bytes)) => Response {
                status: 200,
                content_type: content_type(&path),
                body: bytes,
                attachment: None,
            },
            // a single-page app: an unknown path is one of its routes, not a missing file
            None => match std::fs::read(root.join("index.html")) {
                Ok(bytes) => Response {
                    status: 200,
                    content_type: "text/html; charset=utf-8",
                    body: bytes,
                    attachment: None,
                },
                Err(_) => Response::error(404, "not found"),
            },
        }
    }
}

/// `root/relative`, or `None` when the path tries to leave the directory.
pub fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
    if relative.is_empty() {
        return Some(root.join("index.html"));
    }
    let candidate = Path::new(relative);
    for part in candidate.components() {
        match part {
            Component::Normal(_) => {}
            // "..", a root, a prefix: anything that could climb out
            _ => return None,
        }
    }
    Some(root.join(candidate))
}

pub fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "map" => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Serve on `addr` until killed (or `max_requests`, for tests).
pub fn serve(settings: ServeSettings, addr: &str, max_requests: Option<usize>) -> Result<(), String> {
    let service = Service::new(settings)?;
    http::serve(addr, service.handler(), max_requests)
}

/// Where the built frontend is, when nothing was asked for: `frontend/dist` under the working directory, or
/// `../frontend/dist` (the checkout's layout, run from `rust/`), whichever holds an `index.html`.
pub fn default_frontend_dir() -> Option<PathBuf> {
    ["frontend/dist", "../frontend/dist"]
        .iter()
        .map(PathBuf::from)
        .find(|dir| dir.join("index.html").is_file())
}

/// Where the experiments wrote, when nothing was asked for: `results` or `../results`, whichever exists.
pub fn default_results_dir() -> PathBuf {
    ["results", "../results"]
        .iter()
        .map(PathBuf::from)
        .find(|dir| dir.is_dir())
        .unwrap_or_else(|| PathBuf::from("../results"))
}

// ---- the body's settings and readouts --------------------------------------------------------------------------

/// `settings` with whatever `b` gives: `states`, `window`, `n`, `seed` and the arm toggles.
pub fn body_settings(settings: &BodySettings, b: &Json) -> BodySettings {
    let mut s = settings.clone();
    s.encoder.states = b.num("states", s.encoder.states as f64).max(2.0) as usize;
    s.encoder.window = b.num("window", s.encoder.window as f64).max(1.0) as usize;
    s.encoder.n = b.num("n", s.encoder.n as f64).max(1.0) as usize;
    s.seed = b.num("seed", s.seed as f64).max(0.0) as u64;
    s.encoder_sets_state = b.bool_or("encoder_sets_state", s.encoder_sets_state);
    s.forward_lessons = b.bool_or("forward_lessons", s.forward_lessons);
    s.punish = b.bool_or("punish", s.punish);
    s.whole_run = b.bool_or("whole_run", s.whole_run);
    s.label_set = b.bool_or("label_set", s.label_set);
    s
}

/// `settings` with whatever `b` gives: `states`, `window`, `seed`, `grade`.
pub fn nim_settings(settings: &NimSettings, b: &Json) -> NimSettings {
    let mut s = settings.clone();
    s.encoder.states = b.num("states", s.encoder.states as f64).max(2.0) as usize;
    s.encoder.window = b.num("window", s.encoder.window as f64).max(1.0) as usize;
    s.repertoire.window = s.encoder.window.max(1);
    s.seed = b.num("seed", s.seed as f64).max(0.0) as u64;
    s.grade = b.bool_or("grade", s.grade);
    s
}

fn settings_json(lb: &LanguageBody) -> Json {
    let s = &lb.body.settings;
    let g = &s.encoder.game;
    Json::object()
        .with("language", lb.language.name.into())
        .with("states", s.encoder.states.into())
        .with("window", s.encoder.window.into())
        .with("n", s.encoder.n.into())
        .with("seed", (s.seed as f64).into())
        .with("encoder_sets_state", s.encoder_sets_state.into())
        .with("forward_lessons", s.forward_lessons.into())
        .with("punish", s.punish.into())
        .with("whole_run", s.whole_run.into())
        .with("label_set", s.label_set.into())
        .with("reserve", g.reserve.into())
        .with("seats", g.seats.into())
        .with("iters", g.iters.into())
        .with("permutations", g.permutations.into())
        .with("margin_tau", g.margin_tau.into())
        .with("expl_tau", g.expl_tau.into())
}

fn shape(m: &Machine) -> Json {
    Json::Array(vec![
        m.n_states().into(),
        m.alphabet().len().into(),
        m.n_states().into(),
    ])
}

fn health(lb: &LanguageBody) -> Json {
    let m = &lb.body.lattice.machine;
    Json::object()
        .with("ok", true.into())
        .with("version", VERSION.into())
        .with("language", lb.language.name.into())
        .with("shape", shape(m))
        .with("alphabet", Json::strings(m.alphabet()))
        .with(
            "accepting",
            Json::Array(m.accepting().into_iter().map(Json::from).collect()),
        )
        .with("encoder", settings_json(lb))
}

fn body_status(lb: &LanguageBody) -> Json {
    let body = &lb.body;
    let (mean, gini) = body.encoder.wealth_stats();
    let last = lb.history.last();
    Json::object()
        .with("settings", settings_json(lb))
        .with("shape", shape(&body.lattice.machine))
        .with("episodes", (body.episodes as f64).into())
        .with("lessons", (body.lessons as f64).into())
        .with("diversity", body.encoder.diversity().into())
        .with(
            "wealth",
            Json::object().with("mean", mean.into()).with("gini", gini.into()),
        )
        .with("thinks", (body.encoder.thinks as f64).into())
        .with("namings", (body.encoder.namings as f64).into())
        .with("agreement", body.agreement.rate().into())
        .with("agreement_steps", (body.agreement.steps as f64).into())
        .with("graph_nodes", body.encoder.graph.num_nodes().into())
        .with(
            "accuracy",
            match last {
                Some(p) => Json::object()
                    .with("episodes", p[0].into())
                    .with("encoder", p[1].into())
                    .with("lattice", p[2].into()),
                None => Json::Null,
            },
        )
        .with("lattice", body.lattice.machine.stats())
}

fn languages() -> Json {
    Json::Array(
        LANGUAGES
            .iter()
            .map(|l| {
                Json::object()
                    .with("name", l.name.into())
                    .with("description", l.description.into())
                    .with(
                        "accepting",
                        Json::Array(l.accepting.iter().map(|&s| s.into()).collect()),
                    )
                    .with("min_states", l.min_states.into())
            })
            .collect(),
    )
}

fn accuracy_json((encoder, lattice): (f64, f64)) -> Json {
    Json::object()
        .with("encoder", encoder.into())
        .with("lattice", lattice.into())
}

fn curve_json(curve: &[[f64; 3]]) -> Json {
    Json::Array(curve.iter().map(|p| Json::numbers(p)).collect())
}

/// The text of a classify or run: every character must be a symbol of the lattice's alphabet.
fn text_of(b: &Json, lb: &LanguageBody) -> Result<String, String> {
    let text = b.str_or("text", "").to_string();
    let m = &lb.body.lattice.machine;
    for ch in text.chars() {
        m.symbol(&ch.to_string())
            .map_err(|_| format!("{ch:?} is not a symbol of the alphabet {}", m.alphabet().join("")))?;
    }
    Ok(text)
}

fn classification_json(c: &Classification) -> Json {
    Json::object()
        .with("encoder", c.encoder.into())
        .with("lattice", c.lattice.into())
        .with("named", Json::Array(c.named.iter().map(|&s| s.into()).collect()))
        .with(
            "predicted",
            Json::Array(c.predicted.iter().map(|&s| s.into()).collect()),
        )
        .with("lattice_bits", c.lattice_bits.into())
        .with("encoder_bits", c.encoder_bits.into())
        .with("triggers", c.triggers.into())
}

fn seat_json(s: &TraceSeat, states: usize) -> Json {
    Json::object()
        .with("node", s.node.into())
        .with("bucket", s.bucket.into())
        .with("label", s.label.as_str().into())
        .with("seated", s.seated.into())
        .with("bid", s.bid.into())
        .with("reputation", s.reputation.into())
        .with("wealth", s.wealth.into())
        .with("belief", Json::numbers(&s.belief))
        .with(
            "action",
            match s.action {
                Some(a) if a < states => a.into(),
                Some(_) => "abstain".into(),
                None => Json::Null,
            },
        )
        .with("phi", s.phi.map(Json::from).unwrap_or(Json::Null))
}

fn step_json(i: usize, s: &TraceStep, states: usize) -> Json {
    Json::object()
        .with("index", i.into())
        .with("symbol", s.symbol.as_str().into())
        .with("prev", s.prev.into())
        .with(
            "seats",
            Json::Array(s.seats.iter().map(|x| seat_json(x, states)).collect()),
        )
        .with("aggregate", Json::numbers(&s.aggregate))
        .with("named", s.named.into())
        .with("margin", s.margin.into())
        .with("price", s.price.into())
        .with("bidders", s.bidders.into())
        .with("seated", s.seats.iter().filter(|x| x.seated).count().into())
        .with("no_bidders", s.no_bidders.into())
        .with("exploitability", s.exploitability.into())
        .with("cycle", s.cycle.into())
        .with("converged", s.converged.into())
        .with("predicted", s.predicted.into())
        .with("probability", s.probability.into())
        .with("agree", (s.predicted == s.named).into())
        .with("triggers", Json::Array(s.triggers.iter().map(|&t| t.into()).collect()))
        .with("learned", s.learned.into())
}

/// A classify or a run: the verdicts, whether the string is in the language, and the trace.
fn read_json(lb: &LanguageBody, text: &str, c: &Classification, steps: &[TraceStep], learned: bool) -> Json {
    let m = &lb.body.lattice.machine;
    let states = m.n_states();
    Json::object()
        .with("text", text.into())
        .with("learned", learned.into())
        .with("member", (lb.language.member)(text).into())
        .with("states", states.into())
        .with("start", m.start.into())
        .with(
            "accepting",
            Json::Array(m.accepting().into_iter().map(Json::from).collect()),
        )
        .with("classification", classification_json(c))
        .with(
            "trace",
            Json::Array(steps.iter().enumerate().map(|(i, s)| step_json(i, s, states)).collect()),
        )
        .with("episodes", (lb.body.episodes as f64).into())
}

/// `GET /api/encoder`: the graph, what it has named, its economy, and the `top` richest nodes.
fn encoder(lb: &LanguageBody, top: Option<&str>) -> Json {
    let enc = &lb.body.encoder;
    let top = top.and_then(|t| t.parse::<usize>().ok()).unwrap_or(20);
    let (mean, gini) = enc.wealth_stats();
    let mut nodes: Vec<usize> = (FIRST..enc.graph.num_node_ids())
        .filter(|&n| enc.plays(n) > 0.0)
        .collect();
    nodes.sort_by(|&a, &b| enc.wealth(b).partial_cmp(&enc.wealth(a)).unwrap().then(a.cmp(&b)));
    let players = nodes.len();
    let richest: Vec<Json> = nodes
        .into_iter()
        .take(top)
        .map(|n| {
            Json::object()
                .with("node", n.into())
                .with("label", enc.graph.label(n).into())
                .with("wealth", enc.wealth(n).into())
                .with("reputation", enc.reputation(n).into())
                .with("plays", enc.plays(n).into())
        })
        .collect();
    Json::object()
        .with(
            "graph",
            Json::object()
                .with("nodes", enc.graph.num_nodes().into())
                .with("edges", enc.graph.num_edges().into())
                .with("compression_ratio", enc.graph.compression_ratio().into()),
        )
        .with("states", enc.states().into())
        .with(
            "named",
            Json::numbers(&enc.named.iter().map(|&c| c as f64).collect::<Vec<_>>()),
        )
        .with("namings", (enc.namings as f64).into())
        .with("diversity", enc.diversity().into())
        .with("max_diversity", (enc.states() as f64).log2().into())
        .with(
            "wealth",
            Json::object().with("mean", mean.into()).with("gini", gini.into()),
        )
        .with("thinks", (enc.thinks as f64).into())
        .with("players", players.into())
        .with("top", Json::Array(richest))
}

// ---- the lattice, as latticefsm serve shows it -------------------------------------------------------------------

/// The `S × S` slice for one symbol: per cell the probability in its row, `seen`, width and net - the shape
/// `latticefsm serve` answers, so its matrix view reads it as it is.
pub fn matrix(m: &Machine, symbol: Option<&str>) -> Result<Json, String> {
    let a = match symbol {
        Some(s) => m.symbol(s)?,
        None => 0,
    };
    let (clock, life, stim) = (m.clock, m.life, m.stimulation());
    let rows: Vec<Json> = (0..m.n_states())
        .map(|s| {
            let probs = m.probabilities(s, a, None, None);
            let edges: Vec<Json> = m
                .lattice
                .row(s, a)
                .iter()
                .zip(probs.iter())
                .map(|(e, p)| {
                    Json::object()
                        .with("target", e.target.into())
                        .with("probability", (*p).into())
                        .with("seen", (e.seen as f64).into())
                        .with("width", e.width_at(clock, life).into())
                        .with("net", e.net().into())
                        .with("recent", e.recent_at(clock, life).into())
                        .with("log_weight", e.log_weight(clock, life, stim).into())
                })
                .collect();
            Json::object()
                .with("source", s.into())
                .with("edges", Json::Array(edges))
        })
        .collect();
    Ok(Json::object()
        .with("symbol", m.alphabet()[a].as_str().into())
        .with("symbol_index", a.into())
        .with("alphabet", Json::strings(m.alphabet()))
        .with("stimulation", stim.into())
        .with("state", m.state.into())
        .with(
            "accepting",
            Json::Array(m.accepting().into_iter().map(Json::from).collect()),
        )
        .with("rows", Json::Array(rows)))
}

fn edge(m: &Machine, req: &Request) -> Result<Json, String> {
    let n = |k: &str| {
        req.param(k)
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or(format!("{k} is required"))
    };
    let (source, target) = (n("source")?, n("target")?);
    let symbol = match req.param("symbol") {
        Some(s) => match s.parse::<usize>() {
            Ok(i) => i,
            Err(_) => m.symbol(s)?,
        },
        None => return Err("symbol is required".to_string()),
    };
    m.lattice.offset(source, symbol, target)?;
    Ok(m.edge(source, symbol, target)
        .describe(m.clock, m.life, m.stimulation()))
}

/// The greedy transition table and the accepting states.
pub fn table(m: &Machine) -> Json {
    let t = m.transition_table();
    Json::object()
        .with("alphabet", Json::strings(m.alphabet()))
        .with("start", m.start.into())
        .with(
            "accepting",
            Json::Array(m.accepting().into_iter().map(Json::from).collect()),
        )
        .with(
            "table",
            Json::Array(
                t.iter()
                    .map(|row| Json::Array(row.iter().map(|&t| t.into()).collect()))
                    .collect(),
            ),
        )
}

// ---- Nim ---------------------------------------------------------------------------------------------------------

fn nim_info(nim: &NimBody) -> Json {
    let (bits, used) = nim.information();
    let s = &nim.settings;
    Json::object()
        .with(
            "settings",
            Json::object()
                .with("states", s.encoder.states.into())
                .with("window", s.encoder.window.into())
                .with("seed", (s.seed as f64).into())
                .with("grade", s.grade.into()),
        )
        .with("games", (nim.games as f64).into())
        .with("information_bits", bits.into())
        .with("states_used", used.into())
        .with("agreement", nim.agreement.rate().into())
        .with("diversity", nim.encoder.diversity().into())
        .with("graph_nodes", nim.encoder.graph.num_nodes().into())
}

fn record_json(i: usize, r: &GameRecord, first: bool) -> Json {
    Json::object()
        .with("game", i.into())
        .with("first", first.into())
        .with("won", r.won.into())
        .with("plies", r.plies.into())
        .with("our_moves", r.our_moves.into())
        .with("first_try", r.first_try.into())
        .with("refusals", r.refusals.into())
        .with("abstained", r.abstained.into())
        .with("optimal", r.optimal.into())
        .with("lessons", r.lessons.into())
}

/// The totals of a set of games: rates per game and per move of ours.
fn totals_json(records: &[GameRecord]) -> Json {
    let games = records.len().max(1) as f64;
    let moves = records.iter().map(|r| r.our_moves).sum::<usize>().max(1) as f64;
    let sum = |f: fn(&GameRecord) -> usize| records.iter().map(f).sum::<usize>() as f64;
    Json::object()
        .with("games", records.len().into())
        .with("won", sum(|r| usize::from(r.won)).into())
        .with("win_rate", (sum(|r| usize::from(r.won)) / games).into())
        .with("our_moves", sum(|r| r.our_moves).into())
        .with("legal_first_try", (sum(|r| r.first_try) / moves).into())
        .with("refusals_per_move", (sum(|r| r.refusals) / moves).into())
        .with("abstained_per_move", (sum(|r| r.abstained) / moves).into())
        .with("optimal_rate", (sum(|r| r.optimal) / moves).into())
        .with("lessons", sum(|r| r.lessons).into())
}

/// Play `games`, the first of each alternating unless `first` says (babbling: `babble::babble_nim`, a game at a
/// time); a job ticks once a game and stops on request.
fn nim_games(nim: &mut NimBody, b: &Json, babble: bool, default: usize) -> (Vec<(GameRecord, bool)>, bool) {
    let games = b.num("games", default as f64).max(0.0) as usize;
    let fixed = b.get("first").and_then(Json::as_bool);
    let mut out = Vec::with_capacity(games);
    job::phase(if babble { "babbling" } else { "playing" }, 0);
    for i in 0..games {
        if job::stopped() {
            return (out, true);
        }
        let first = fixed.unwrap_or(i % 2 == 0);
        let r = nim.play(first, babble);
        job::tick("nim", r.plies);
        out.push((r, first));
    }
    (out, false)
}

fn nim_report(nim: &NimBody, played: &[(GameRecord, bool)], stopped: bool) -> Json {
    let records: Vec<GameRecord> = played.iter().map(|(r, _)| r.clone()).collect();
    let (bits, used) = nim.information();
    Json::object()
        .with("stopped", stopped.into())
        .with("totals", totals_json(&records))
        .with("information_bits", bits.into())
        .with("states_used", used.into())
        .with("agreement", nim.agreement.rate().into())
        .with(
            "records",
            Json::Array(
                played
                    .iter()
                    .enumerate()
                    .map(|(i, (r, first))| record_json(i + 1, r, *first))
                    .collect(),
            ),
        )
        .with("nim", nim_info(nim))
}

fn nim_babble(nim: &mut NimBody, b: &Json) -> Result<Json, String> {
    let (played, stopped) = nim_games(nim, b, true, 200);
    Ok(nim_report(nim, &played, stopped))
}

fn nim_play(nim: &mut NimBody, b: &Json) -> Result<Json, String> {
    let (played, stopped) = nim_games(nim, b, false, 20);
    Ok(nim_report(nim, &played, stopped))
}
