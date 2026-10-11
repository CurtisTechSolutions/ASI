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
//! | `GET /api/cube` | the whole matrix at once, for the 3D view: every cell's probability, seen, width and net, row-major |
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
//! | `GET /api/ngram` | the N-gram model the server also holds (trigrams of words trained on `data/corpus` at start): stats, history |
//! | `POST /api/ngram/new` | `{order?, unit?, stimulation?}`: a fresh model (word, char, sound or base64) |
//! | `POST /api/ngram/train` | `{data?, held_out?, predict?, self_credit?}`: train on files or directories (default `data/corpus`), each token predicted before it is learned; how well, and the held-out measure |
//! | `POST /api/ngram/text` | `{text, predict?, self_credit?}`: train on pasted text, a line a sequence, predicting as it learns |
//! | `POST /api/ngram/predict` | `{text, k?}`: the likeliest next tokens after `text` |
//! | `POST /api/ngram/generate` | `{text, max_tokens?, temperature?, seed?}`: `text` continued |
//! | `POST /api/ngram/credit` | `{text, amount}`: reward (positive) or punish the edges a text takes |
//! | `GET /api/ngram/ollama` | `?url=&model=`: is Ollama there, which models |
//! | `POST /api/ngram/autotrain` | `{data?, ollama?, model?, url?, topics?, rounds?, lines?, continuations?, judge?, patience?, fresh?, predict?, self_credit?}` |
//! | `POST /api/ngram/save`, `/api/ngram/load` | `{path}`: write or read the model |
//! | `GET /api/chain` | the chain the server holds (8 nodes of letters, confidence 0.9, trained on `data/corpus` at start): stats per depth, sources, history |
//! | `POST /api/chain/new` | `{nodes?, unit?, confidence?, lower?}`: a fresh chain |
//! | `POST /api/{chain,ngram}/train` | `{data?, uploads?, held_out?, predict?, self_credit?, until?, max_rounds?}`: train on server paths and uploads, every token predicted first; with `until`, drilled to N of N (or the most possible); the held-out measure |
//! | `POST /api/{chain,ngram}/text` | `{text, predict?, self_credit?, until?, max_rounds?}`: train on pasted text, a line a sequence |
//! | `POST /api/{chain,ngram}/predict` | `{text, k?}`: the next tokens; the chain's input nodes and walk too |
//! | `POST /api/{chain,ngram}/generate` | `{text, max_tokens?, temperature?, seed?, count?}`: `count` samples (the chain's walk depths with them) |
//! | `POST /api/{chain,ngram}/credit`, `/feedback` | `{text, amount}`, `{good, bad, amount?}`: reward or punish |
//! | `POST /api/{chain,ngram}/score` | `{text}`: every token's probability (and walk depth), perplexity, bits |
//! | `POST /api/{chain,ngram}/teach` | `{text, reply, max_rounds?, threshold?}`: teach `reply` after `text` until every token of it is right; the new continuation |
//! | `POST /api/chain/confidence`, `/sweep` | `{confidence}`, `{data?, uploads?, held_out?, thresholds?}`: set the threshold; accuracy and depth at each threshold on held-out lines |
//! | `GET /api/chain/graph` | `?limit=`: the most-seen node contexts: guess, confidence |
//! | `POST /api/{chain,ngram}/save`, `/load` | `{path}`: write or read the model |
//! | `GET /api/uploads`, `POST /api/uploads`, `POST /api/uploads/delete` | `{name, content_base64 \| content, append?}`, `{name}`: the uploads (any file; big ones in pieces), written under `data/uploads` or `$LATTICEFSM_UPLOADS` |
//! | `GET /api/checkpoints`, `POST /api/checkpoints/{save,restore,delete,upload}`, `GET /api/checkpoints/download?name=` | `{model, tag?}`, `{name}`, `{name, content_base64, append?}`: checkpoints of any model in `checkpoints/` or `$LATTICEFSM_CHECKPOINTS` |
//! | `POST /api/converse` | `{model, opening, turns?, max_tokens?, temperature?}`: the model talking to itself |
//! | `POST /api/chat/llm` | `{model, topic, turns?, pass_mark?, learn?, ollama_model?, url?}`: Ollama talks with the model and marks its replies; 502 without Ollama |
//! | `POST /api/ollama/corpus` | `{prompt, lines?, ollama_model?, url?}`: sentences Ollama writes |
//! | `GET /api/status` | every model in brief, uploads, checkpoints |
//! | `GET /api/network` | the network the server also holds (`enc>dec` until another is made): its blocks and summary |
//! | `POST /api/network/new` | `{arch?, alphabet?, seed?, accepting?, credit?, layer_discount?, temperature?, discount?, life?}` |
//! | `POST /api/network/talk` | `{text}`: the network's reply - English from a translator, yes / no from a classifier - and its trace |
//! | `POST /api/network/teach` | `{text, reply, times?}`: teach it that `reply` is the answer to `text` |
//! | `POST /api/network/fit` | `{task?, epochs?}` (plural, past), `{language, episodes?}`, or `{data, epochs?}` (a labels file) |
//! | `POST /api/network/save`, `/api/network/load` | `{path}`: write or read the network |

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::autotrain::Ollama;
use crate::chain::{load_chain, Chain};
use crate::compress::{compress, expansion, fidelity, load_core, Precision};
use crate::experiment::teach_language;
use crate::http::{self, Handler, Request, Response};
use crate::json::Json;
use crate::languages::{examples, language, LANGUAGES};
use crate::machine::{load_machine, Machine, Settings, Walk, DEFAULT_ALPHABET, DEFAULT_STATES};
use crate::network::{load_network, Credit, Merge, Network, Step, Vote, LAYER_DISCOUNT};
use crate::ngram::{load_ngram, Evaluation, NGram};
use crate::rng::Rng;
use crate::routes::{
    chain_route, chat_llm, checkpoint_model, checkpoint_name, checkpoint_path, checkpoint_upload, checkpoints_dir,
    checkpoints_list, converse, default_chain, lm_route, ollama_corpus, uploads_route, Lm,
};
use crate::VERSION;

/// The machine behind the server, the network beside it (made on first use), and where the prebuilt frontend lives.
pub struct Service {
    pub machine: Mutex<Machine>,
    pub network: Mutex<Option<Network>>,
    pub ngram: Mutex<Option<NGram>>,
    pub chain: Mutex<Option<Chain>>,
    pub frontend: Option<PathBuf>,
}

impl Service {
    pub fn new(machine: Machine) -> Arc<Service> {
        Arc::new(Service {
            machine: Mutex::new(machine),
            network: Mutex::new(None),
            ngram: Mutex::new(None),
            chain: Mutex::new(None),
            frontend: None,
        })
    }

    /// The same, serving the static files of `dir` (the built frontend) at every path outside `/api/`.
    pub fn with_frontend(machine: Machine, dir: Option<PathBuf>) -> Arc<Service> {
        Arc::new(Service {
            machine: Mutex::new(machine),
            network: Mutex::new(None),
            ngram: Mutex::new(None),
            chain: Mutex::new(None),
            frontend: dir,
        })
    }

    /// Run `f` on the language model a body names (`model`: chain, the default, or ngram).
    fn with_lm_result(&self, b: &Json, f: impl FnOnce(&mut dyn Lm) -> Result<Json, String>) -> Result<Json, String> {
        if b.str_or("model", "chain") == "ngram" {
            let mut g = self
                .ngram
                .lock()
                .map_err(|_| "the N-gram model's lock is poisoned".to_string())?;
            f(g.get_or_insert_with(default_ngram))
        } else {
            let mut g = self
                .chain
                .lock()
                .map_err(|_| "the chain's lock is poisoned".to_string())?;
            f(g.get_or_insert_with(default_chain))
        }
    }

    fn with_lm(&self, b: &Json, f: impl FnOnce(&mut dyn Lm) -> Result<Json, String>) -> Response {
        answer(self.with_lm_result(b, f))
    }

    /// Every model in brief, and how many uploads and checkpoints there are.
    fn status(&self) -> Json {
        let chain = self.chain.lock().ok().map(|mut g| {
            let m = g.get_or_insert_with(default_chain);
            Json::object()
                .with("nodes", m.nodes.into())
                .with("unit", m.unit.as_str().into())
                .with("confidence", m.confidence.into())
                .with("tokens", (m.clock as f64).into())
                .with("contexts", m.tables.iter().map(|t| t.len()).sum::<usize>().into())
                .with("vocab", (m.vocab.len() - 3).into())
        });
        let ngram = self.ngram.lock().ok().map(|mut g| {
            let m = g.get_or_insert_with(default_ngram);
            Json::object()
                .with("order", m.order.into())
                .with("unit", m.unit.as_str().into())
                .with("tokens", (m.clock as f64).into())
                .with("ngrams", m.n_grams().into())
                .with("vocab", (m.vocab.len() - 3).into())
        });
        let machine = self.machine.lock().ok().map(|m| {
            let st = m.stats();
            Json::object()
                .with("shape", st.get("shape").cloned().unwrap_or(Json::Null))
                .with("clock", st.get("clock").cloned().unwrap_or(Json::Null))
        });
        let uploads = uploads_route("GET", "", &Json::Null)
            .ok()
            .and_then(|j| j.get("uploads").and_then(Json::as_array).map(Vec::len))
            .unwrap_or(0);
        let checkpoints = checkpoints_list()
            .get("checkpoints")
            .and_then(Json::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        Json::object()
            .with("version", VERSION.into())
            .with("chain", chain.unwrap_or(Json::Null))
            .with("ngram", ngram.unwrap_or(Json::Null))
            .with("machine", machine.unwrap_or(Json::Null))
            .with("uploads", uploads.into())
            .with("checkpoints", checkpoints.into())
    }

    /// Save a model to a file, or read one in place of the model of its kind.
    fn save_model(&self, model: &str, path: &str) -> Result<(), String> {
        match model {
            "chain" => self
                .chain
                .lock()
                .map_err(poisoned)?
                .get_or_insert_with(default_chain)
                .save(path),
            "ngram" => self
                .ngram
                .lock()
                .map_err(poisoned)?
                .get_or_insert_with(default_ngram)
                .save(path),
            "machine" => self.machine.lock().map_err(poisoned)?.save(path),
            "network" => match self.network.lock().map_err(poisoned)?.as_ref() {
                Some(n) => n.save(path),
                None => Err("there is no network yet: make one first".to_string()),
            },
            other => Err(format!("no model {other:?}")),
        }
    }

    fn load_model(&self, model: &str, path: &str) -> Result<Json, String> {
        Ok(match model {
            "chain" => {
                let mut g = self.chain.lock().map_err(poisoned)?;
                Lm::info(g.insert(load_chain(path)?))
            }
            "ngram" => {
                let mut g = self.ngram.lock().map_err(poisoned)?;
                Lm::info(g.insert(load_ngram(path)?))
            }
            "machine" => {
                let mut g = self.machine.lock().map_err(poisoned)?;
                *g = load_machine(path)?;
                g.stats()
            }
            "network" => {
                let mut g = self.network.lock().map_err(poisoned)?;
                network_info(g.insert(load_network(path)?))
            }
            other => return Err(format!("no model {other:?}")),
        })
    }

    /// `/api/checkpoints`: list, save, restore, delete, download, upload.
    fn checkpoints_route(&self, req: &Request) -> Response {
        let action = req.path["/api/checkpoints".len()..].trim_start_matches('/');
        let b = &req.body;
        if (req.method.as_str(), action) == ("GET", "download") {
            let name = req.param("name").unwrap_or("");
            return match checkpoint_path(name).and_then(|p| std::fs::read(&p).map_err(|e| format!("{name}: {e}"))) {
                Ok(bytes) => Response::file(
                    if name.ends_with(".gz") {
                        "application/gzip"
                    } else {
                        "application/json"
                    },
                    bytes,
                    name,
                ),
                Err(e) => Response::error(404, &e),
            };
        }
        answer(match (req.method.as_str(), action) {
            ("GET", "") => Ok(checkpoints_list()),
            ("POST", "save") => (|| {
                let model = b.str_or("model", "chain");
                let name = checkpoint_name(model, b.str_or("tag", "manual"))?;
                let dir = checkpoints_dir();
                std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                self.save_model(model, &dir.join(&name).to_string_lossy())?;
                Ok(Json::object().with("name", name.as_str().into()))
            })(),
            ("POST", "restore") => (|| {
                let name = b.str_or("name", "");
                let model = checkpoint_model(name).ok_or_else(|| format!("which model {name:?} holds is not known"))?;
                let info = self.load_model(model, &checkpoint_path(name)?.to_string_lossy())?;
                Ok(Json::object().with("model", model.into()).with("info", info))
            })(),
            ("POST", "delete") => (|| {
                let name = b.str_or("name", "");
                std::fs::remove_file(checkpoint_path(name)?).map_err(|e| format!("{name}: {e}"))?;
                Ok(Json::object().with("deleted", name.into()))
            })(),
            ("POST", "upload") => checkpoint_upload(b),
            ("GET", _) | ("POST", _) => Err("no such route".to_string()),
            _ => Err("method not allowed".to_string()),
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
        if req.path == "/api/network" || req.path.starts_with("/api/network/") {
            return self.network_route(req);
        }
        if req.path == "/api/ngram" || req.path.starts_with("/api/ngram/") {
            let mut guard = match self.ngram.lock() {
                Ok(g) => g,
                Err(_) => return Response::error(500, "the N-gram model's lock is poisoned"),
            };
            let m = guard.get_or_insert_with(default_ngram);
            return answer(ngram_route(m, req));
        }
        if req.path == "/api/chain" || req.path.starts_with("/api/chain/") {
            let mut guard = match self.chain.lock() {
                Ok(g) => g,
                Err(_) => return Response::error(500, "the chain's lock is poisoned"),
            };
            let m = guard.get_or_insert_with(default_chain);
            let action = req.path["/api/chain".len()..].trim_start_matches('/');
            return answer(chain_route(m, &req.method, action, req));
        }
        if req.path == "/api/uploads" || req.path.starts_with("/api/uploads/") {
            let action = req.path["/api/uploads".len()..].trim_start_matches('/');
            return answer(uploads_route(&req.method, action, &req.body));
        }
        if req.path == "/api/checkpoints" || req.path.starts_with("/api/checkpoints/") {
            return self.checkpoints_route(req);
        }
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/api/status") => return answer(Ok(self.status())),
            ("POST", "/api/converse") => return self.with_lm(&req.body, |m| Ok(converse(m, &req.body))),
            ("POST", "/api/chat/llm") => {
                return match self.with_lm_result(&req.body, |m| chat_llm(m, &req.body)) {
                    Ok(v) => Response::ok(&v),
                    Err(e) => Response::error(502, &e),
                }
            }
            ("POST", "/api/ollama/corpus") => {
                return match ollama_corpus(&req.body) {
                    Ok(v) => Response::ok(&v),
                    Err(e) => Response::error(502, &e),
                }
            }
            _ => {}
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
            ("GET", "cube") => Ok(cube(&m)),
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
    fn network_route(&self, req: &Request) -> Response {
        let mut guard = match self.network.lock() {
            Ok(g) => g,
            Err(_) => return Response::error(500, "the network's lock is poisoned"),
        };
        if guard.is_none() {
            match new_network(&Json::object()) {
                Ok(n) => *guard = Some(n),
                Err(e) => return Response::error(500, &e),
            }
        }
        let net = guard.as_mut().expect("made above");
        let result = match (req.method.as_str(), &req.path[5..]) {
            ("GET", "network") => Ok(network_info(net)),
            ("POST", "network/new") => new_network(&req.body).map(|fresh| {
                *net = fresh;
                network_info(net)
            }),
            ("POST", "network/talk") => talk(net, req.body.str_or("text", "")),
            ("POST", "network/teach") => network_teach(net, &req.body),
            ("POST", "network/fit") => network_fit(net, &req.body),
            ("POST", "network/save") => {
                let path = req.body.str_or("path", "network.json.gz").to_string();
                net.save(&path)
                    .map(|_| Json::object().with("saved", path.as_str().into()))
            }
            ("POST", "network/load") => load_network(req.body.str_or("path", "network.json.gz")).map(|loaded| {
                *net = loaded;
                network_info(net)
            }),
            ("GET", _) | ("POST", _) => Err("no such route".to_string()),
            _ => return Response::error(405, "method not allowed"),
        };
        match result {
            Ok(v) => Response::ok(&v),
            Err(msg) if msg == "no such route" => Response::error(404, &msg),
            Err(msg) => Response::error(400, &msg),
        }
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

/// The whole matrix at once: per cell `(s, a, t)`, row-major, its probability in its row, `seen`, width and net;
/// with the shape, the symbols, the central node, the focus node, each cell's shell and where the machine is.
fn cube(m: &Machine) -> Json {
    use crate::geometry::{center, focus_node, shell};
    let shape = m.lattice.shape();
    let (clock, life) = (m.clock, m.life);
    let (mut prob, mut seen, mut width, mut net, mut shells) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in 0..shape.0 {
        for a in 0..shape.1 {
            let probs = m.probabilities(s, a, None, None);
            for (e, p) in m.lattice.row(s, a).iter().zip(probs) {
                prob.push(p);
                seen.push(e.seen as f64);
                width.push(e.width_at(clock, life));
                net.push(e.net());
                shells.push(shell(shape, s, a, e.target) as f64);
            }
        }
    }
    let triple = |(x, y, z): (usize, usize, usize)| Json::Array(vec![x.into(), y.into(), z.into()]);
    Json::object()
        .with("shape", triple(shape))
        .with("symbols", Json::strings(m.alphabet()))
        .with("center", triple(center(shape)))
        .with("focus", m.focus().map(Json::from).unwrap_or(Json::Null))
        .with("focus_node", triple(focus_node(shape, m.focus())))
        .with("state", m.state.into())
        .with(
            "accepting",
            Json::Array(m.accepting().into_iter().map(Json::from).collect()),
        )
        .with("probability", Json::numbers(&prob))
        .with("seen", Json::numbers(&seen))
        .with("width", Json::numbers(&width))
        .with("net", Json::numbers(&net))
        .with("shell", Json::numbers(&shells))
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

/// A network from `{arch?, alphabet?, seed?, accepting?, credit?, layer_discount?, temperature?, discount?, life?}`;
/// without `arch`, `enc>dec`: English in, English out.
pub fn new_network(body: &Json) -> Result<Network, String> {
    let alphabet: Vec<String> = body
        .str_or("alphabet", DEFAULT_ALPHABET)
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_string())
        .collect();
    let d = Settings::default();
    let settings = Settings {
        temperature: body.num("temperature", d.temperature),
        discount: body.num("discount", d.discount),
        life: body.num("life", d.life),
        ..d
    };
    let mut net = Network::from_architecture(
        body.str_or("arch", "enc>dec"),
        &alphabet,
        body.num("seed", 1.0) as u64,
        Merge::parse(body.str_or("merge", "concat"))?,
        Vote::parse(body.str_or("vote", "majority"))?,
        &settings,
    )?;
    let accepting: Option<Vec<usize>> = body
        .get("accepting")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_f64().map(|n| n as usize)).collect());
    let accepting = accepting.or_else(|| net.decoder().is_none().then(|| vec![0]));
    net.compile(
        accepting.as_deref(),
        body.num("layer_discount", LAYER_DISCOUNT),
        1.0,
        1.0,
        Credit::parse(body.str_or("credit", "shared"))?,
    )?;
    Ok(net)
}

fn network_info(net: &Network) -> Json {
    let layers: Vec<Json> = net
        .layers()
        .iter()
        .map(|l| {
            let (s, a, t) = l.machine.lattice.shape();
            Json::object()
                .with("name", l.name.as_str().into())
                .with("shape", Json::Array(vec![s.into(), a.into(), t.into()]))
                .with("touched", l.machine.lattice.touched().count().into())
                .with("decoder", l.is_decoder().into())
        })
        .collect();
    Json::object()
        .with("architecture", net.architecture.as_str().into())
        .with("reads", net.encoder().map_or("symbols", |e| e.source.kind()).into())
        .with("translator", (net.decoder().is_some()).into())
        .with("layers", Json::Array(layers))
        .with("words", net.words.len().into())
        .with("examples", (net.examples as f64).into())
        .with("epochs", (net.epochs as f64).into())
        .with("credit", net.credit.as_str().into())
        .with("summary", net.summary().into())
}

fn step_json(s: &Step) -> Json {
    Json::object()
        .with("layer", s.layer.as_str().into())
        .with("read", s.read.as_str().into())
        .with("states", Json::Array(s.states.iter().map(|&x| x.into()).collect()))
        .with("wrote", s.wrote.as_str().into())
        .with("accepted", s.accepted.into())
        .with("english", s.english.as_deref().map(Json::from).unwrap_or(Json::Null))
}

/// The network's reply to `text`: what a translator writes, in English, or a classifier's yes or no; and its trace.
fn talk(net: &mut Network, text: &str) -> Result<Json, String> {
    let (steps, output, accepted) = net.trace(text)?;
    let translator = net.decoder().is_some();
    let mut j = Json::object()
        .with("text", text.into())
        .with("translator", translator.into())
        .with("accepted", accepted.into())
        .with("trace", Json::Array(steps.iter().map(step_json).collect()));
    if translator {
        let sounds = net.sounds(text)?;
        j = j
            .with("reply", net.translate(text)?.into())
            .with("sounds", Json::strings(&sounds));
    } else {
        j = j
            .with("reply", if accepted { "yes" } else { "no" }.into())
            .with("output", output.into());
    }
    Ok(j)
}

/// `{text, reply, times?, until?, max_rounds?}`: a translator taught the pair, a classifier the verdict (yes / no);
/// then its new reply.  With `until`, round after round (each `times` teachings) until a whole round is right - N of
/// N - and the network's answer is the reply (or `max_rounds`); `right` / `of` are then the last round's.
fn network_teach(net: &mut Network, body: &Json) -> Result<Json, String> {
    let text = body.str_or("text", "").to_string();
    let reply = body.str_or("reply", "").to_string();
    let times = (body.num("times", 1.0) as usize).clamp(1, 1000);
    let until = body.bool_or("until", false);
    let max_rounds = if until {
        (body.num("max_rounds", 200.0) as usize).clamp(1, 100_000)
    } else {
        1
    };
    let threshold = crate::routes::threshold_of(body);
    let translator = net.decoder().is_some();
    let label = if translator {
        None
    } else {
        Some(crate::media::verdict(&reply)?)
    };
    let (mut right, mut of, mut rounds, mut reached) = (0, 0, 0, false);
    let mut history = Vec::new();
    while rounds < max_rounds {
        (right, of) = (0, 0);
        for _ in 0..times {
            let (r, n) = match label {
                None => net.teach_pair(&text, &reply)?,
                Some(l) => (net.teach(&text, l)? as usize, 1),
            };
            right += r;
            of += n;
        }
        rounds += 1;
        history.push(right as f64);
        // N of N, and the answer it now gives is the one taught
        let answers = match label {
            None => net.sounds(&text)? == net.target_sounds(&reply),
            Some(l) => net.predict(std::slice::from_ref(&text))?[0] == l,
        };
        // the goal: `threshold` of a round right - and, at all of it, the answer it gives is the one taught
        if right >= crate::ngram::goal_of(threshold, of) && (threshold < 1.0 || answers) {
            reached = true;
            break;
        }
    }
    let after = talk(net, &text)?;
    Ok(Json::object()
        .with("right", right.into())
        .with("of", of.into())
        .with("need", crate::ngram::goal_of(threshold, of).into())
        .with("rounds", rounds.into())
        .with("reached", reached.into())
        .with("history", Json::numbers(&history))
        .with("after", after))
}

/// Examples taught and held out.
type Split = (Vec<(String, bool)>, Vec<(String, bool)>);

/// `{task, epochs?}` for a translator, `{language, episodes?}` or `{data, epochs?}` for a classifier.
fn network_fit(net: &mut Network, body: &Json) -> Result<Json, String> {
    if let Some(task) = body.get("task").and_then(Json::as_str) {
        let pairs = crate::phonetic::task(task)?;
        let (train, test) = crate::phonetic::split(&pairs, 4);
        let mut h = net.fit_pairs(&train, body.num("epochs", 20.0) as usize, Some(&test))?;
        // until: epochs on and on until every taught pair is exactly right
        let max_epochs = (body.num("max_epochs", 200.0) as usize).clamp(1, 100_000);
        let threshold = crate::routes::threshold_of(body);
        while body.bool_or("until", false)
            && h.exact.last().is_some_and(|&e| e < threshold)
            && h.exact.len() < max_epochs
        {
            let more = net.fit_pairs(&train, 1, Some(&test))?;
            h.sounds.extend(more.sounds);
            h.exact.extend(more.exact);
            h.val_exact.extend(more.val_exact);
        }
        let reached = h.exact.last().is_some_and(|&e| e >= threshold);
        let mut shown = Vec::new();
        for (x, y) in train.iter().take(3).chain(test.iter().take(3)) {
            shown.push(Json::strings(&[x.clone(), net.translate(x)?, y.clone()]));
        }
        return Ok(Json::object()
            .with("sounds", Json::numbers(&h.sounds))
            .with("exact", Json::numbers(&h.exact))
            .with("held_out", Json::numbers(&h.val_exact))
            .with("examples", Json::Array(shown))
            .with("reached", reached.into())
            .with("network", network_info(net)));
    }
    let (train, test): Split = if let Some(path) = body.get("data").and_then(Json::as_str) {
        let files = net
            .encoder()
            .is_some_and(|e| !matches!(e.source.kind(), "phonetic" | "text"));
        let rows = crate::media::read_labels(path, files)?;
        let mut tr = Vec::new();
        let mut te = Vec::new();
        for (i, (x, y)) in rows.into_iter().enumerate() {
            let e = (x, crate::media::verdict(&y)?);
            if i % 5 == 4 {
                te.push(e)
            } else {
                tr.push(e)
            }
        }
        (tr, te)
    } else {
        let lang = language(body.str_or("language", "even-b"))?;
        let mut rng = Rng::new(body.num("seed", 1.0) as u64);
        let tr = examples(&lang, body.num("episodes", 2000.0) as usize, &mut rng, 6);
        let te = examples(&lang, 200, &mut rng, 6);
        (tr, te)
    };
    let epochs = (body.num("epochs", 1.0) as usize).max(1);
    let until = body.bool_or("until", false);
    let threshold = crate::routes::threshold_of(body);
    let max_epochs = (body.num("max_epochs", 200.0) as usize).clamp(epochs, 100_000);
    let mut held = Vec::new();
    let mut taught = Vec::new();
    let mut reached = false;
    while taught.len() < if until { max_epochs } else { epochs } {
        taught.push(net.fit(&train, 1)?[0]);
        held.push(net.evaluate(&test)?);
        // until: every taught example right while taught, and by the quiet pass after
        if until
            && taught.len() >= epochs
            && taught[taught.len() - 1] >= threshold
            && net.evaluate(&train)? >= threshold
        {
            reached = true;
            break;
        }
    }
    if !until {
        reached = taught.last().is_some_and(|&t| t >= threshold);
    }
    Ok(Json::object()
        .with("taught", Json::numbers(&taught))
        .with("held_out", Json::numbers(&held))
        .with("reached", reached.into())
        .with("network", network_info(net)))
}

/// The model the server starts with: trigrams of words, trained on `data/corpus` (or `../data/corpus`) when there is
/// one, every tenth line held out.
pub fn default_ngram() -> NGram {
    let mut m = NGram::new(3, "word", true, 1.0).expect("valid");
    for dir in ["data/corpus", "../data/corpus"] {
        if std::path::Path::new(dir).is_dir() {
            if let Ok((_, test)) = m.train_paths(&[dir.to_string()], 10) {
                let ev = m.evaluate(&test, 500);
                m.history
                    .push(eval_json(&ev).with("round", 0usize.into()).with("source", dir.into()));
            }
            break;
        }
    }
    m
}

fn eval_json(ev: &Evaluation) -> Json {
    Json::object()
        .with("tokens", ev.tokens.into())
        .with("perplexity", ev.perplexity.map(Json::from).unwrap_or(Json::Null))
        .with("bits", ev.bits.map(Json::from).unwrap_or(Json::Null))
        .with("accuracy", ev.accuracy.map(Json::from).unwrap_or(Json::Null))
        .with("top5", ev.top5.map(Json::from).unwrap_or(Json::Null))
}

fn ngram_info(m: &mut NGram) -> Json {
    m.stats().with("history", Json::Array(m.history.clone())).with(
        "sources",
        Json::Array(
            m.sources
                .iter()
                .map(|(s, c, n)| {
                    Json::object()
                        .with("source", s.as_str().into())
                        .with("lines", (*c).into())
                        .with("tokens", (*n).into())
                })
                .collect(),
        ),
    )
}

fn paths_of(body: &Json) -> Vec<String> {
    match body.get("data") {
        Some(Json::String(s)) => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        Some(Json::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => vec![if std::path::Path::new("data/corpus").is_dir() {
            "data/corpus"
        } else {
            "../data/corpus"
        }
        .to_string()],
    }
}

pub(crate) fn ngram_predict(m: &mut NGram, body: &Json) -> Json {
    let text = body.str_or("text", "");
    let k = (body.num("k", 10.0) as usize).clamp(1, 200);
    let context: Vec<String> = {
        let toks = m.prompt_tokens(text);
        let ctx = m.context(&toks);
        ctx.iter().map(|&i| m.vocab[i as usize].clone()).collect()
    };
    let next: Vec<Json> = m
        .predict(text, k)
        .into_iter()
        .map(|(t, p)| Json::object().with("token", t.as_str().into()).with("p", p.into()))
        .collect();
    Json::object()
        .with("text", text.into())
        .with("unit", m.unit.as_str().into())
        .with("context", Json::strings(&context))
        .with("next", Json::Array(next))
}

fn poisoned<T>(_: T) -> String {
    "a lock is poisoned".to_string()
}

/// A route's result as a response: 404 and 405 for the routes that are not there, 400 for every other error.
fn answer(r: Result<Json, String>) -> Response {
    match r {
        Ok(v) => Response::ok(&v),
        Err(msg) if msg == "no such route" => Response::error(404, &msg),
        Err(msg) if msg == "method not allowed" => Response::error(405, &msg),
        Err(msg) => Response::error(400, &msg),
    }
}

fn ngram_route(m: &mut NGram, req: &Request) -> Result<Json, String> {
    let b = &req.body;
    let action = req.path["/api/ngram".len()..].trim_start_matches('/');
    if let Some(r) = lm_route(m, &req.method, action, req) {
        return r;
    }
    match (req.method.as_str(), &req.path[5..]) {
        ("POST", "ngram/new") => {
            *m = NGram::new(
                b.num("order", 3.0) as usize,
                b.str_or("unit", "word"),
                b.bool_or("lower", true),
                b.num("stimulation", 1.0),
            )?;
            Ok(ngram_info(m))
        }
        ("GET", "ngram/ollama") => {
            let client = Ollama::new(req.param("url"), req.param("model"));
            Ok(match client.models() {
                Ok(models) => Json::object()
                    .with("available", true.into())
                    .with("url", client.url.as_str().into())
                    .with("model", client.model.as_str().into())
                    .with("models", Json::strings(&models)),
                Err(e) => Json::object()
                    .with("available", false.into())
                    .with("url", client.url.as_str().into())
                    .with("model", client.model.as_str().into())
                    .with("error", e.into()),
            })
        }
        ("POST", "ngram/autotrain") => {
            let client = b.bool_or("ollama", false).then(|| {
                Ollama::new(
                    b.get("url").and_then(Json::as_str),
                    b.get("model").and_then(Json::as_str),
                )
            });
            let topics: Vec<String> = b
                .str_or("topics", "")
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            let o = crate::autotrain::Options {
                data: paths_of(b),
                held_out: b.num("held_out", 10.0) as usize,
                topics,
                rounds: (b.num("rounds", 3.0) as usize).min(50),
                lines: b.num("lines", 20.0) as usize,
                continuations: b.num("continuations", 20.0) as usize,
                judge: b.num("judge", 10.0) as usize,
                patience: b.num("patience", 2.0) as usize,
                keep: b.get("keep").and_then(Json::as_str).map(String::from),
                predict: b.bool_or("predict", true),
                self_credit: b.num("self_credit", 0.0),
                ..Default::default()
            };
            if b.bool_or("fresh", true) {
                // the corpus is trained again: start from an empty model of the same shape, not on top of the old one
                let (order, unit, lower, stim) = (m.order, m.unit.clone(), m.lower, m.stimulation);
                *m = NGram::new(order, &unit, lower, stim)?;
            }
            let mut log = Vec::new();
            let records = crate::autotrain::autotrain(m, &o, client.as_ref(), &mut |l| log.push(l.to_string()))?;
            Ok(Json::object()
                .with("rounds", Json::Array(records))
                .with("log", Json::strings(&log))
                .with("model", ngram_info(m)))
        }
        ("POST", "ngram/load") => {
            *m = load_ngram(b.str_or("path", "ngram.json.gz"))?;
            Ok(ngram_info(m))
        }
        ("GET", _) | ("POST", _) => Err("no such route".to_string()),
        _ => Err("method not allowed".to_string()),
    }
}
