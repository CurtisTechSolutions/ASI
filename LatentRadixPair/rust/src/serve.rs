//! The JSON API and the static frontend behind `latentpair serve`.
//!
//! One thread answers requests; the model sits behind a mutex. Training a new tokenizer runs on its own
//! thread and swaps the finished model in, while progress is polled from the page.

use crate::model::Model;
use crate::pair::{self, Settings};
use crate::tokenizer::{self, Config, Options, Stat, Tokenizer};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Response, Server};

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_JS: &str = include_str!("../web/app.js");
const STYLE_CSS: &str = include_str!("../web/style.css");

struct State {
    model: Model,
    path: String,
}

#[derive(Default)]
struct Progress {
    running: bool,
    done: bool,
    error: Option<String>,
    stats: Vec<Stat>,
    config: Option<Config>,
    params: usize,
}

type Shared = Arc<Mutex<State>>;
type Prog = Arc<Mutex<Progress>>;

/// Serves the model at addr until the process ends.
pub fn run(model: Model, path: String, addr: &str) -> Result<(), String> {
    let server = Server::http(addr).map_err(|e| format!("listen on {addr}: {e}"))?;
    let state: Shared = Arc::new(Mutex::new(State { model, path }));
    let progress: Prog = Arc::new(Mutex::new(Progress::default()));
    eprintln!("latentpair: serving http://{addr}/");
    for mut request in server.incoming_requests() {
        let mut body = String::new();
        if request.body_length().unwrap_or(0) > 0 {
            request.as_reader().read_to_string(&mut body).ok();
        }
        let url = request.url().split('?').next().unwrap_or("/").to_string();
        let response = match (request.method(), url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => text_response(200, INDEX_HTML, "text/html; charset=utf-8"),
            (Method::Get, "/app.js") => text_response(200, APP_JS, "application/javascript; charset=utf-8"),
            (Method::Get, "/style.css") => text_response(200, STYLE_CSS, "text/css; charset=utf-8"),
            (Method::Get, "/api/info") => json_result(info(&state, &progress)),
            (Method::Get, "/api/tokenizer/progress") => json_result(Ok(progress_json(&progress))),
            (Method::Post, p) => json_result(post(p, &body, &state, &progress)),
            _ => text_response(404, "not found", "text/plain"),
        };
        let _ = request.respond(response);
    }
    Ok(())
}

fn text_response(status: u16, body: &str, content_type: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", content_type).unwrap())
}

fn json_result(r: Result<Value, String>) -> Response<std::io::Cursor<Vec<u8>>> {
    match r {
        Ok(v) => text_response(200, &v.to_string(), "application/json"),
        Err(e) => text_response(400, &json!({ "error": e }).to_string(), "application/json"),
    }
}

fn parse<T: for<'a> Deserialize<'a>>(body: &str) -> Result<T, String> {
    serde_json::from_str(if body.is_empty() { "{}" } else { body }).map_err(|e| format!("bad request: {e}"))
}

fn info(state: &Shared, progress: &Prog) -> Result<Value, String> {
    let s = state.lock().unwrap();
    let mut v = s.model.info();
    v["model_path"] = json!(s.path);
    let p = progress.lock().unwrap();
    v["training"] = if p.running {
        json!(format!("step {}", p.stats.last().map_or(0, |s| s.step)))
    } else {
        Value::Null
    };
    Ok(v)
}

fn progress_json(progress: &Prog) -> Value {
    let p = progress.lock().unwrap();
    json!({ "running": p.running, "done": p.done, "error": p.error, "stats": p.stats, "config": p.config, "params": p.params })
}

#[derive(Deserialize)]
#[serde(default)]
struct PredictReq {
    prefix: String,
    length: usize,
    mode: String,
    traversal: String,
    backoff: String,
    temperature: f64,
    to_end: bool,
}
impl Default for PredictReq {
    fn default() -> Self {
        PredictReq { prefix: String::new(), length: 32, mode: "greedy".into(), traversal: "reward".into(), backoff: String::new(), temperature: 1.0, to_end: false }
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct FoldReq {
    prefix: String,
    traversal: String,
    backoff: String,
    top: usize,
}
impl Default for FoldReq {
    fn default() -> Self {
        FoldReq { prefix: String::new(), traversal: "reward".into(), backoff: String::new(), top: 10 }
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct TextsReq {
    texts: Vec<String>,
    limit: usize,
}

#[derive(Deserialize)]
#[serde(default)]
struct JudgeReq {
    kind: String,
    text: String,
    bad: String,
    prefix: String,
    strength: f64,
    outcomes: usize,
    read: bool,
}
impl Default for JudgeReq {
    fn default() -> Self {
        JudgeReq { kind: "reward".into(), text: String::new(), bad: String::new(), prefix: String::new(), strength: 0.0, outcomes: 0, read: true }
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct ScoreReq {
    text: String,
    traversal: String,
    backoff: String,
}
impl Default for ScoreReq {
    fn default() -> Self {
        ScoreReq { text: String::new(), traversal: "reward".into(), backoff: String::new() }
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct TrainTokReq {
    texts: Vec<String>,
    steps: usize,
    batch: usize,
    lr: f64,
    window: usize,
    levels: String,
    recency: f64,
    predict: f64,
    noise: f64,
    seed: u64,
    read: bool,
}
impl Default for TrainTokReq {
    fn default() -> Self {
        let c = Config::default();
        TrainTokReq { texts: vec![], steps: 400, batch: 256, lr: 3e-3, window: c.window, levels: tokenizer::levels_string(&c.levels), recency: c.recency, predict: c.predict, noise: c.noise, seed: 1, read: true }
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SaveReq {
    path: String,
}

fn post(path: &str, body: &str, state: &Shared, progress: &Prog) -> Result<Value, String> {
    match path {
        "/api/predict" => {
            let r: PredictReq = parse(body)?;
            let mut s = state.lock().unwrap();
            let res = s.model.predict(&r.prefix, r.length, &r.mode, &r.traversal, r.temperature, r.to_end, &r.backoff, 4096)?;
            let mut v = serde_json::to_value(&res).map_err(|e| e.to_string())?;
            v["unit_names"] = json!(res.units.iter().map(|&u| pair::symbol(u)).collect::<Vec<_>>());
            v["bits_per_unit"] = json!(res.cost / std::f64::consts::LN_2 / res.units.len().max(1) as f64);
            Ok(v)
        }
        "/api/fold" => {
            let r: FoldReq = parse(body)?;
            let mut s = state.lock().unwrap();
            fold_json(&mut s.model, &r.prefix, &r.traversal, &r.backoff, r.top)
        }
        "/api/decode" => {
            let r: FoldReq = parse(body)?;
            let s = state.lock().unwrap();
            let code = s.model.code(&r.prefix);
            let decodes: Vec<Value> = (1..=s.model.tok.depth())
                .map(|k| json!({ "known": k, "code": &code[..k], "text": s.model.tok.decode(&code, k).unwrap_or_default() }))
                .collect();
            Ok(json!({ "code": code, "decodes": decodes }))
        }
        "/api/read" => {
            let r: TextsReq = parse(body)?;
            if r.texts.is_empty() {
                return Err("no texts".into());
            }
            let mut s = state.lock().unwrap();
            Ok(s.model.train(&r.texts))
        }
        "/api/judge" => {
            let r: JudgeReq = parse(body)?;
            let mut s = state.lock().unwrap();
            let mut watch: Vec<usize> = Vec::new();
            for t in [&r.text, &r.bad] {
                if let Some(&b) = t.as_bytes().first() {
                    if !watch.contains(&(b as usize)) {
                        watch.push(b as usize);
                    }
                }
            }
            let before = s.model.fold(&r.prefix, "reward", "")?;
            let before_pen = s.model.fold(&r.prefix, "punishment", "")?;
            let record = match r.kind.as_str() {
                "reward" => s.model.reward(&[r.text.clone()], r.strength, None, r.read, &r.prefix, r.outcomes)?,
                "punish" => s.model.punish(&[r.text.clone()], r.strength, None, &r.prefix, r.outcomes)?,
                "two_nrl" => s.model.two_nrl(&[r.bad.clone()], &[r.text.clone()], r.strength, &r.prefix, r.outcomes)?,
                k => return Err(format!("kind must be reward, punish or two_nrl, got {k:?}")),
            };
            let after = s.model.fold(&r.prefix, "reward", "")?;
            let after_pen = s.model.fold(&r.prefix, "punishment", "")?;
            let changes: Vec<Value> = watch
                .iter()
                .flat_map(|&x| {
                    vec![
                        json!({ "unit": format!("{} reward", pair::symbol(x)), "before": before[x], "after": after[x] }),
                        json!({ "unit": format!("{} punishment", pair::symbol(x)), "before": before_pen[x], "after": after_pen[x] }),
                    ]
                })
                .collect();
            Ok(json!({ "record": record, "changes": changes }))
        }
        "/api/score" => {
            let r: ScoreReq = parse(body)?;
            let mut s = state.lock().unwrap();
            let sc = s.model.score(&r.text, &r.traversal, &r.backoff)?;
            serde_json::to_value(sc).map_err(|e| e.to_string())
        }
        "/api/classes" => {
            let r: TextsReq = parse(body)?;
            if r.texts.is_empty() {
                return Err("no texts".into());
            }
            let s = state.lock().unwrap();
            let texts: Vec<Vec<u8>> = r.texts.iter().map(|t| t.as_bytes().to_vec()).collect();
            Ok(classes_json(&s.model.tok, &texts))
        }
        "/api/settings" => {
            let settings: Settings = parse(body)?;
            let mut s = state.lock().unwrap();
            s.model.pair.configure(settings)?;
            Ok(s.model.info())
        }
        "/api/save" => {
            let r: SaveReq = parse(body)?;
            let s = state.lock().unwrap();
            let path = if r.path.is_empty() { s.path.clone() } else { r.path };
            s.model.save(&path)?;
            Ok(json!({ "saved": path }))
        }
        "/api/tokenizer/train" => {
            let r: TrainTokReq = parse(body)?;
            if r.texts.is_empty() {
                return Err("no texts".into());
            }
            {
                let mut p = progress.lock().unwrap();
                if p.running {
                    return Err("a tokenizer is already training".into());
                }
                *p = Progress { running: true, ..Default::default() };
            }
            let mut cfg = Config { window: r.window, recency: r.recency, predict: r.predict, noise: r.noise, ..Config::default() };
            cfg.levels = tokenizer::parse_levels(&r.levels)?;
            cfg.validate()?;
            let settings = state.lock().unwrap().model.pair.settings.clone();
            let (state, progress) = (state.clone(), progress.clone());
            std::thread::spawn(move || {
                let outcome = (|| -> Result<(), String> {
                    let texts: Vec<Vec<u8>> = r.texts.iter().map(|t| t.as_bytes().to_vec()).collect();
                    let mut tok = Tokenizer::new(cfg.clone(), r.seed as i64)?;
                    {
                        let mut p = progress.lock().unwrap();
                        p.config = Some(cfg.clone());
                        p.params = tok.param_count();
                    }
                    let o = Options { steps: r.steps, batch: r.batch, lr: r.lr, seed: r.seed, eval_every: (r.steps / 20).max(1), ..Options::default() };
                    let prog = progress.clone();
                    tok.train(&texts, &o, |s| prog.lock().unwrap().stats.push(s.clone()))?;
                    let mut model = Model::prime(tok, settings, r.seed as i64)?;
                    if r.read {
                        model.train_bytes(&texts);
                    }
                    state.lock().unwrap().model = model;
                    Ok(())
                })();
                let mut p = progress.lock().unwrap();
                p.running = false;
                p.done = outcome.is_ok();
                p.error = outcome.err();
            });
            Ok(json!({ "started": true }))
        }
        _ => Err(format!("no such endpoint {path}")),
    }
}

/// A context's code with each level's reading and the top outcomes of the fold.
pub fn fold_json(model: &mut Model, prefix: &str, traversal: &str, backoff: &str, top: usize) -> Result<Value, String> {
    let code = model.code(prefix);
    let mut chain = Vec::new();
    model.pair.addr.chain(&code, &mut chain);
    let levels: Vec<Value> = chain
        .iter()
        .enumerate()
        .map(|(l, &node)| {
            json!({ "level": l, "node": node, "seen": model.pair.count.ctx[node], "own": model.pair.count.own(node),
                    "decode": if l == 0 { String::new() } else { model.tok.decode(&code, l).unwrap_or_default() } })
        })
        .collect();
    let p = model.pair.fold(&code, traversal, backoff)?;
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
    let entropy: f64 = -p.iter().filter(|&&v| v > 0.0).map(|&v| v * v.log2()).sum::<f64>();
    let top: Vec<Value> = idx.iter().take(top.max(1)).map(|&x| json!({ "unit": pair::symbol(x), "x": x, "p": p[x] })).collect();
    Ok(json!({ "code": code, "levels": levels, "top": top, "entropy_bits": entropy }))
}

/// What each value of the first code symbol stands for over some texts.
pub fn classes_json(tok: &Tokenizer, texts: &[Vec<u8>]) -> Value {
    let radix = tok.config.radix(0);
    let mut counts = vec![0usize; radix];
    let mut examples: Vec<Option<Vec<u8>>> = vec![None; radix];
    let mut total = 0;
    for text in texts {
        let codes = tok.encode_all(text);
        for i in 0..codes.len() {
            let c = codes.at(i)[0] as usize;
            counts[c] += 1;
            total += 1;
            if examples[c].is_none() || (i % 97 == 0 && i >= tok.config.window) {
                let lo = i.saturating_sub(tok.config.window);
                examples[c] = Some(text[lo..i].to_vec());
            }
        }
    }
    let classes: Vec<Value> = (0..radix)
        .map(|c| {
            let mut code = vec![0i32; tok.depth()];
            code[0] = c as i32;
            json!({ "symbol": c, "count": counts[c], "share": counts[c] as f64 / total.max(1) as f64,
                    "prototype": tok.decode(&code, 1).unwrap_or_default(),
                    "example": examples[c].as_ref().map(|e| String::from_utf8_lossy(e).into_owned()).unwrap_or_default() })
        })
        .collect();
    json!({ "radix": radix, "total": total, "classes": classes })
}
