//! The HTTP API and the page that uses it.
//!
//! `latticefsm serve` holds one machine behind a lock and answers JSON under
//! `/api/`; anything else is served from the prebuilt React frontend
//! (`../frontend/dist`, the directory `Service::frontend` is given), which
//! draws the matrix one symbol-slice at a time, runs strings at a stimulation
//! of your choosing, rewards and punishes them, teaches a language, and lets
//! time pass.  An unknown path under it is one of the single-page app's own
//! routes and gets `index.html`; without a frontend directory the root
//! answers a JSON 404 that says so.
//!
//! | route | does |
//! |---|---|
//! | `GET /api/health` | the version and the machine's shape |
//! | `GET /api/stats` | [`Machine::stats`] |
//! | `GET /api/matrix?symbol=a` | the `S × S` slice for one symbol: probabilities and a summary of every edge |
//! | `GET /api/edge?source=&symbol=&target=` | one edge, every field |
//! | `GET /api/table` | the greedy transition table |
//! | `GET /api/languages` | the languages `train` knows |
//! | `POST /api/run` | `{text, stimulation?, temperature?, quiet?}`: a run, traversed unless quiet |
//! | `POST /api/credit` | `{amount}`: reward (positive) or punish (negative) the last run |
//! | `POST /api/teach` | `{source, symbol, target, amount}`: one edge, traversed and credited |
//! | `POST /api/train` | `{language, episodes?}`: random strings credited by the language's verdict |
//! | `POST /api/tick` | `{ticks}`: time passes |
//! | `POST /api/stimulate` | `{amount}` or `{level}`: raise the stimulation, or set it |
//! | `POST /api/new` | `{states?, alphabet?, accepting?, life?, ...}`: a fresh machine (13 × 13 × 13 by default) |
//! | `POST /api/compress` | `{precision?, budget?}`: fold the matrix into its central node; the code, its loss, and its rebuild shell by shell |
//! | `GET /api/core` | the code the central node holds, measured against the machine now |
//! | `POST /api/expand` | `{shells?}`: replace the machine with the code's rebuild, from the central node outward |
//! | `POST /api/core/run` | `{text, from_middle?}`: the greedy walk read straight from the code |
//! | `POST /api/core/save`, `/api/core/load` | `{path}`: write or read the code |
//! | `POST /api/compression` | `{every?, precision?, rebuild?}`: compress automatically every N transitions |
//! | `POST /api/focus` | `{focus?: number or null, learn?, text?}`: the node of the central vertical vector runs start from, or a focus learned from the input |
//! | `POST /api/skip` | `{skip?, margin?, rearrange_every?}`: skips when a two-edge path is more efficient; the rearrangement schedule |
//! | `POST /api/rearrange` | `{full?}` or `{axis, i, j}`: let the nodes rearrange themselves toward the centre, or swap two |
//! | `POST /api/save` | `{path}`: write the machine |
//! | `POST /api/load` | `{path}`: read a machine in place of the old |

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::compress::{compress, expansion, fidelity, load_core, Precision};
use crate::experiment::teach_language;
use crate::http::{self, Handler, Request, Response};
use crate::json::Json;
use crate::languages::{examples, language, LANGUAGES};
use crate::machine::{load_machine, Machine, Settings, Walk, DEFAULT_ALPHABET, DEFAULT_STATES};
use crate::rng::Rng;
use crate::VERSION;

/// The machine behind the server, and where the prebuilt frontend lives.
pub struct Service {
    pub machine: Mutex<Machine>,
    pub frontend: Option<PathBuf>,
}

impl Service {
    pub fn new(machine: Machine) -> Arc<Service> {
        Arc::new(Service {
            machine: Mutex::new(machine),
            frontend: None,
        })
    }

    /// The same, serving the static files of `dir` (the built frontend) at every path outside `/api/`.
    pub fn with_frontend(machine: Machine, dir: Option<PathBuf>) -> Arc<Service> {
        Arc::new(Service {
            machine: Mutex::new(machine),
            frontend: dir,
        })
    }

    /// The handler the HTTP server calls.
    pub fn handler(self: &Arc<Service>) -> Arc<Handler> {
        let service = Arc::clone(self);
        Arc::new(move |req: &Request| service.route(req))
    }

    pub fn route(&self, req: &Request) -> Response {
        if !req.path.starts_with("/api/") {
            return self.serve_static(req);
        }
        let mut m = match self.machine.lock() {
            Ok(m) => m,
            Err(_) => return Response::error(500, "the machine's lock is poisoned"),
        };
        let result = match (req.method.as_str(), &req.path[5..]) {
            ("GET", "health") => Ok(Json::object()
                .with("ok", true.into())
                .with("version", VERSION.into())
                .with("shape", m.stats().get("shape").cloned().unwrap_or(Json::Null))),
            ("GET", "stats") => Ok(m.stats()),
            ("GET", "matrix") => matrix(&m, req.param("symbol")),
            ("GET", "edge") => edge(&m, req),
            ("GET", "table") => Ok(table(&m)),
            ("GET", "languages") => Ok(Json::Array(
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
            )),
            ("POST", "run") => run(&mut m, &req.body),
            ("POST", "compress") => compress_route(&mut m, &req.body),
            ("GET", "core") => core_route(&m),
            ("POST", "expand") => expand_route(&mut m, &req.body),
            ("POST", "core/run") => match &m.core {
                Some(core) => core
                    .walk(
                        req.body.str_or("text", ""),
                        req.body.bool_or("from_middle", false),
                        req.body.get("focus").and_then(Json::as_f64),
                        req.body.get("skip").and_then(Json::as_bool),
                    )
                    .map(|(states, accepted)| {
                        Json::object()
                            .with("states", Json::Array(states.into_iter().map(Json::from).collect()))
                            .with("accepted", accepted.into())
                            .with("from_middle", req.body.bool_or("from_middle", false).into())
                    }),
                None => Err("nothing has been compressed yet: POST /api/compress first".to_string()),
            },
            ("POST", "core/save") => match &m.core {
                Some(core) => {
                    let path = req.body.str_or("path", "core.json.gz").to_string();
                    core.save(&path)
                        .map(|_| Json::object().with("saved", path.as_str().into()))
                }
                None => Err("nothing has been compressed yet: POST /api/compress first".to_string()),
            },
            ("POST", "core/load") => load_core(req.body.str_or("path", "core.json.gz")).map(|core| {
                let summary = core.summary();
                m.core = Some(core);
                Json::object().with("summary", summary)
            }),
            ("POST", "compression") => compression_route(&mut m, &req.body),
            ("POST", "focus") => focus_route(&mut m, &req.body),
            ("POST", "skip") => skip_route(&mut m, &req.body),
            ("POST", "rearrange") => rearrange_route(&mut m, &req.body),
            ("POST", "credit") => {
                let amount = req.body.num("amount", 1.0);
                let credited = m.credit(amount);
                Ok(Json::object()
                    .with("credited", credited.into())
                    .with("amount", amount.into())
                    .with("stats", m.stats()))
            }
            ("POST", "teach") => teach(&mut m, &req.body),
            ("POST", "train") => train(&mut m, &req.body),
            ("POST", "tick") => {
                m.tick(req.body.num("ticks", 1.0) as i64);
                Ok(m.stats())
            }
            ("POST", "stimulate") => {
                let r = match req.body.get("level").and_then(Json::as_f64) {
                    Some(level) => m.set_stimulation(level).map(|_| level),
                    None => Ok(m.stimulate(req.body.num("amount", 1.0))),
                };
                r.map(|level| {
                    Json::object()
                        .with("stimulation", level.into())
                        .with("stats", m.stats())
                })
            }
            ("POST", "new") => new_machine(&req.body).map(|fresh| {
                *m = fresh;
                m.stats()
            }),
            ("POST", "save") => {
                let path = req.body.str_or("path", "machine.json.gz").to_string();
                m.save(&path)
                    .map(|_| Json::object().with("saved", path.as_str().into()))
            }
            ("POST", "load") => {
                let path = req.body.str_or("path", "machine.json.gz").to_string();
                load_machine(&path).map(|loaded| {
                    *m = loaded;
                    m.stats()
                })
            }
            ("GET", _) | ("POST", _) => Err("no such route".to_string()),
            _ => return Response::error(405, "method not allowed"),
        };
        match result {
            Ok(v) => Response::ok(&v),
            Err(msg) if msg == "no such route" => Response::error(404, &msg),
            Err(msg) => Response::error(400, &msg),
        }
    }
}

impl Service {
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
            },
            // a single-page app: an unknown path is one of its routes, not a missing file
            None => match std::fs::read(root.join("index.html")) {
                Ok(bytes) => Response {
                    status: 200,
                    content_type: "text/html; charset=utf-8",
                    body: bytes,
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

fn matrix(m: &Machine, symbol: Option<&str>) -> Result<Json, String> {
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

fn table(m: &Machine) -> Json {
    let t = m.transition_table();
    Json::object()
        .with("alphabet", Json::strings(m.alphabet()))
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

fn run(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let text = body.str_or("text", "");
    let quiet = body.bool_or("quiet", false);
    let walk = Walk {
        stimulation: body.get("stimulation").and_then(Json::as_f64),
        temperature: body.get("temperature").and_then(Json::as_f64),
        quiet,
        from_middle: body.bool_or("from_middle", false),
        focus: body.get("focus").and_then(Json::as_f64),
        skip: body.get("skip").and_then(Json::as_bool),
    };
    let symbols = m.tokenize(text)?;
    let run = m.walk(&symbols, walk)?;
    Ok(run
        .to_json()
        .with("quiet", quiet.into())
        .with("from_middle", walk.from_middle.into())
        .with("focus", m.last_focus.map(Json::from).unwrap_or(Json::Null))
        .with("stats", m.stats()))
}

fn focus_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    match body.get("focus") {
        Some(Json::Null) => m.set_focus(None)?,
        Some(Json::Number(f)) => m.set_focus(Some(*f))?,
        Some(_) => return Err("focus is a number in [0, 1] or null".to_string()),
        None => {}
    }
    if let Some(learn) = body.get("learn").and_then(Json::as_bool) {
        m.learn_focus = learn;
    }
    let predicted = match body.get("text").and_then(Json::as_str) {
        Some(text) => {
            let symbols = m.tokenize(text)?;
            let f = m.focus_learner.predict(&symbols);
            Json::object()
                .with("text", text.into())
                .with("focus", f.into())
                .with("state", m.focus_state(f).into())
        }
        None => Json::Null,
    };
    Ok(Json::object().with("learned", predicted).with("stats", m.stats()))
}

fn skip_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    if let Some(skip) = body.get("skip").and_then(Json::as_bool) {
        m.skip = skip;
    }
    if let Some(margin) = body.get("margin").and_then(Json::as_f64) {
        if margin < 0.0 {
            return Err("margin must be >= 0".to_string());
        }
        m.skip_margin = margin;
    }
    if let Some(every) = body.get("rearrange_every").and_then(Json::as_f64) {
        m.rearrange_every = every.max(0.0) as u64;
    }
    Ok(m.stats())
}

fn rearrange_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let swaps: Vec<Json> = if let Some(axis) = body.get("axis").and_then(Json::as_str) {
        let (i, j) = (body.num("i", -1.0), body.num("j", -1.0));
        if i < 0.0 || j < 0.0 {
            return Err("a swap needs i and j".to_string());
        }
        match axis {
            "states" => m.swap_states(i as usize, j as usize)?,
            "symbols" => m.swap_symbols(i as usize, j as usize)?,
            other => return Err(format!("axis is states or symbols, got {other:?}")),
        }
        vec![Json::Array(vec![axis.into(), i.into(), j.into()])]
    } else {
        m.rearrange(body.bool_or("full", false))
            .into_iter()
            .map(|(axis, i, j)| Json::Array(vec![axis.into(), i.into(), j.into()]))
            .collect()
    };
    Ok(Json::object()
        .with("swaps", Json::Array(swaps))
        .with("stats", m.stats()))
}

/// The code the machine holds, measured against the machine: its summary, its loss, its rebuild shell by shell.
fn core_report(m: &Machine) -> Result<Json, String> {
    let core = m
        .core
        .as_ref()
        .ok_or("nothing has been compressed yet: POST /api/compress first")?;
    if core.shape != m.lattice.shape() {
        return Ok(Json::object()
            .with("summary", core.summary())
            .with("fidelity", Json::Null)
            .with("expansion", Json::Null));
    }
    let rebuilt = core.decompress(None)?;
    Ok(Json::object()
        .with("summary", core.summary())
        .with("fidelity", fidelity(m, &rebuilt))
        .with("expansion", expansion(m, core)?)
        .with("compressed_at", m.last_compressed.into()))
}

fn compress_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let precision = body
        .get("precision")
        .and_then(Json::as_str)
        .map(Precision::parse)
        .transpose()?;
    let budget = body.get("budget").and_then(Json::as_f64).map(|b| b.max(0.0) as usize);
    let core = compress(m, precision, budget);
    m.core = Some(core);
    m.compressions += 1;
    m.since_compression = 0;
    m.last_compressed = m.clock;
    Ok(core_report(m)?.with("stats", m.stats()))
}

fn core_route(m: &Machine) -> Result<Json, String> {
    Ok(core_report(m)?.with("stats", m.stats()))
}

fn expand_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let core = m
        .core
        .take()
        .ok_or("nothing has been compressed yet: POST /api/compress first")?;
    let shells = body.get("shells").and_then(Json::as_f64).map(|k| k.max(0.0) as usize);
    let rebuilt = core.decompress(shells);
    match rebuilt {
        Ok(mut fresh) => {
            fresh.core = Some(core);
            *m = fresh;
            Ok(Json::object()
                .with("shells", shells.map(Json::from).unwrap_or(Json::Null))
                .with("stats", m.stats()))
        }
        Err(e) => {
            m.core = Some(core);
            Err(e)
        }
    }
}

fn compression_route(m: &mut Machine, body: &Json) -> Result<Json, String> {
    if let Some(every) = body.get("every").and_then(Json::as_f64) {
        m.compress_every = every.max(0.0) as u64;
    }
    if let Some(p) = body.get("precision").and_then(Json::as_str) {
        m.compress_precision = Precision::parse(p)?;
    }
    if let Some(r) = body.get("rebuild").and_then(Json::as_bool) {
        m.compress_rebuild = r;
    }
    Ok(m.stats())
}

fn teach(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let source = body.num("source", 0.0) as usize;
    let target = body.num("target", 0.0) as usize;
    let symbol = match body.get("symbol") {
        Some(Json::String(s)) => m.symbol(s)?,
        Some(Json::Number(n)) => *n as usize,
        _ => return Err("symbol is required".to_string()),
    };
    let amount = body.num("amount", 1.0);
    let (clock, life, stim) = (m.clock + 1, m.life, m.stimulation());
    let e = m.teach(source, symbol, target, amount)?.describe(clock, life, stim);
    Ok(Json::object().with("edge", e).with("stats", m.stats()))
}

fn train(m: &mut Machine, body: &Json) -> Result<Json, String> {
    let lang = language(body.str_or("language", "even-b"))?;
    let episodes = body.num("episodes", 4000.0).max(0.0) as usize;
    let max_length = body.num("max_length", 6.0) as usize;
    for &s in lang.accepting {
        if s >= m.n_states() {
            return Err(format!(
                "{} needs an accepting state {s}; the machine has {} states",
                lang.name,
                m.n_states()
            ));
        }
    }
    if body.bool_or("set_accepting", true) {
        for st in m.lattice.states.iter_mut() {
            st.accepting = lang.accepting.contains(&st.index);
        }
    }
    let mut rng = Rng::new(m.seed ^ (m.clock as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let test = examples(&lang, 200, &mut rng, max_length);
    let before = m.accuracy(&test, 0.0);
    let every = (episodes / 10).max(1);
    let curve = teach_language(m, &lang, episodes, &mut rng, max_length, false, 1.0, Some(&test), every);
    let after = m.accuracy(&test, 0.0);
    Ok(Json::object()
        .with("language", lang.name.into())
        .with("episodes", episodes.into())
        .with("before", before.into())
        .with("after", after.into())
        .with(
            "curve",
            Json::Array(
                curve
                    .iter()
                    .map(|(ep, acc)| Json::numbers(&[*ep as f64, *acc]))
                    .collect(),
            ),
        )
        .with("table", table(m))
        .with("stats", m.stats()))
}

/// A machine from `{states?, alphabet?, accepting?, start?, life?, baseline?, temperature?, discount?, seed?}`;
/// without `states` and `alphabet`, the default 13 × 13 × 13.
pub fn new_machine(body: &Json) -> Result<Machine, String> {
    let states = body.num("states", DEFAULT_STATES as f64) as usize;
    let alphabet: Vec<String> = match body.get("alphabet") {
        Some(Json::String(s)) => s
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_string())
            .collect(),
        Some(Json::Array(items)) => items.iter().filter_map(|s| s.as_str().map(str::to_string)).collect(),
        _ => DEFAULT_ALPHABET.chars().map(|c| c.to_string()).collect(),
    };
    let accepting: Vec<usize> = body
        .get("accepting")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_f64().map(|n| n as usize)).collect())
        .unwrap_or_default();
    let d = Settings::default();
    let settings = Settings {
        start: body.num("start", 0.0) as usize,
        life: body.num("life", d.life),
        baseline: body.num("baseline", d.baseline),
        calm: body.get("calm").and_then(Json::as_f64),
        temperature: body.num("temperature", d.temperature),
        discount: body.num("discount", d.discount),
        trace: body.num("trace", d.trace),
        use_widening: body.num("use_widening", d.use_widening),
        reward_widening: body.num("reward_widening", d.reward_widening),
        punish_narrowing: body.num("punish_narrowing", d.punish_narrowing),
        prototype: d.prototype,
        seed: body.num("seed", 1.0) as u64,
        compress_every: body.num("compress_every", 0.0).max(0.0) as u64,
        compress_precision: Precision::parse(body.str_or("compress_precision", "exact"))?,
        compress_rebuild: body.bool_or("compress_rebuild", false),
        focus: body.get("focus").and_then(Json::as_f64),
        learn_focus: body.bool_or("learn_focus", false),
        focus_rate: body.num("focus_rate", d.focus_rate),
        focus_explore: body.num("focus_explore", d.focus_explore),
        skip: body.bool_or("skip", false),
        skip_margin: body.num("skip_margin", 0.0),
        rearrange_every: body.num("rearrange_every", 0.0).max(0.0) as u64,
    };
    Machine::new(states, &alphabet, &accepting, settings)
}

/// Serve `machine` on `addr`, with the frontend built at `frontend` when there is one.
pub fn serve(
    machine: Machine,
    addr: &str,
    frontend: Option<PathBuf>,
    max_requests: Option<usize>,
) -> Result<(), String> {
    let service = Service::with_frontend(machine, frontend);
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
