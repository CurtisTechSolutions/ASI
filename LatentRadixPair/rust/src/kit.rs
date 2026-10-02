//! The ModelKit HTTP API (ModelKit/frontend, RadixCyclicNN DESIGN.md section 12) spoken by this model, so
//! ModelKit's React app can drive a LatentRadixPair server: `/api/status` says `engine: "rust"` and lists
//! the routes answered, and the app shows the tabs those routes serve (Train, Predict, Generate, Score,
//! 2NRL, Checkpoints, Model settings, Graph) and hides the rest.
//!
//! Long runs (train, 2NRL) are jobs: one at a time, polled at `GET /api/job`, stoppable at
//! `POST /api/job/stop`; the job thread takes the model lock one text at a time so the page stays live.

use crate::model::{self, Model};
use crate::pair::{self, PathResult};
use crate::tokenizer;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The model kind this server reports.
pub const KIND: &str = "latent";
pub const LABEL: &str = "LatentRadixPair";

/// What the server holds besides the model.
pub struct Kit {
    pub job: Mutex<Option<Job>>,
    pub stop: AtomicBool,
    pub next_job: AtomicU64,
    pub checkpoint_dir: Option<String>,
    pub inverted: Mutex<bool>,
    pub last_loss: Mutex<Option<f64>>,
    pub twonrl_runs: AtomicU64,
    pub epochs_total: AtomicU64,
    /// Every epoch record of every train and 2NRL job, for `GET /api/history`.
    pub training: Mutex<Vec<Value>>,
}

impl Kit {
    pub fn new(checkpoint_dir: Option<String>) -> Kit {
        Kit {
            job: Mutex::new(None),
            stop: AtomicBool::new(false),
            next_job: AtomicU64::new(1),
            checkpoint_dir,
            inverted: Mutex::new(false),
            last_loss: Mutex::new(None),
            twonrl_runs: AtomicU64::new(0),
            epochs_total: AtomicU64::new(0),
            training: Mutex::new(Vec::new()),
        }
    }
}

/// A background run: train or 2NRL.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: u64,
    pub kind: String,
    pub state: String, // running | done | error | stopped
    pub progress: Option<Value>,
    pub history: Vec<Value>,
    pub error: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
}

impl Job {
    pub fn to_json(&self) -> Value {
        json!({ "id": self.id, "type": self.kind, "state": self.state, "progress": self.progress, "history": self.history,
                "error": self.error, "started_at": self.started_at, "finished_at": self.finished_at })
    }
}

/// The routes this server answers, as the app reads them from `/api/status`.
pub fn routes(checkpoints: bool) -> Vec<String> {
    let mut r: Vec<&str> = vec![
        "GET /api/health", "GET /api/status", "GET /api/model", "POST /api/model/select", "POST /api/train", "GET /api/job",
        "POST /api/job/stop", "POST /api/predict", "POST /api/generate", "POST /api/score", "POST /api/2nrl", "POST /api/feedback",
        "POST /api/invert", "POST /api/save", "POST /api/load", "POST /api/reset", "GET /api/history", "GET /api/graph",
        "GET /api/encoding", "POST /api/encoding/preview", "GET /api/model/attention", "GET /api/model/window", "GET /api/uploads",
    ];
    if checkpoints {
        r.extend(["GET /api/checkpoints", "POST /api/checkpoints/save", "POST /api/checkpoints/restore"]);
    }
    r.into_iter().map(String::from).collect()
}

fn kinds(m: &Model) -> Value {
    json!([{ "kind": KIND, "label": LABEL, "description": format!(
        "a trained context tokenizer under a primed radix pair: the last {} bytes compressed into a {:.0}-bit code of {} symbols, bytes as units",
        m.tok.config.window, m.tok.config.bits(), m.tok.depth()) }])
}

/// The model's numbers, as the status bar and the stats answers read them.
pub fn stats(m: &Model, kit: &Kit) -> Value {
    let (mut edges, mut used) = (0usize, 0usize);
    m.pair.count.nonzero(|_, _| edges += 1);
    for &c in &m.pair.count.ctx {
        if c > 0 {
            used += 1;
        }
    }
    let bits = m.tok.config.bits();
    json!({
        "kind": KIND, "model_label": LABEL, "kinds": kinds(m), "units": "chars",
        "nodes": m.pair.addr.n, "edges": edges, "trigrams": used,
        "compression_ratio": (m.tok.config.window as f64 * 8.0) / bits.max(1e-9),
        "inverted": *kit.inverted.lock().unwrap(), "backend": "cpu", "device": "cpu",
        "trained_texts": m.pair.count.texts, "epochs_total": kit.epochs_total.load(Ordering::Relaxed),
        "twonrl_runs": kit.twonrl_runs.load(Ordering::Relaxed), "last_loss": *kit.last_loss.lock().unwrap(),
        "encoding": format!("latent code {} over {} bytes", tokenizer::levels_string(&m.tok.config.levels), m.tok.config.window),
        "code_bits": bits, "window": m.tok.config.window, "radices": m.pair.addr.radices,
        "read_units": m.pair.count.units, "judged_texts": m.pair.reward.judged,
    })
}

fn now() -> String {
    model::now_rfc3339()
}

/// The texts of a train-like body: `texts`, or `text` split on newlines with blank lines dropped.
fn texts_of(v: &Value, list: &str, single: &str) -> Vec<String> {
    let mut out: Vec<String> = v[list].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
    if let Some(t) = v[single].as_str() {
        out.extend(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from));
    }
    out
}

/// Bits per byte of some texts under the model (a sample of 32 at most, for a loss line).
fn loss_of(m: &mut Model, texts: &[String]) -> f64 {
    let (mut bits, mut n) = (0.0, 0usize);
    for t in texts.iter().take(32) {
        if let Ok(s) = m.score(t, "reward", "") {
            bits += s.bits * (t.len() + 1) as f64;
            n += t.len() + 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        bits / n as f64
    }
}

fn path_json(prefix: &str, r: &PathResult) -> Value {
    let names: Vec<String> = r.units.iter().map(|&u| pair::symbol(u)).collect();
    json!({
        "prefix": prefix, "continuation": r.text, "full_text": r.full_text, "text": r.text, "cost": r.cost,
        "probability": (-r.cost).exp(), "step_costs": r.step_costs, "path": names, "node_ids": r.units,
        "unit_names": names, "units": r.units, "expanded": r.units.len(), "reached_end": r.reached_end,
        "mode": r.mode, "traversal": r.traversal, "peak": r.peak,
        "bits_per_unit": r.cost / std::f64::consts::LN_2 / r.units.len().max(1) as f64,
    })
}

/// A beam over the fold: the k cheapest continuations, `width` outcomes tried per step.
fn beam(m: &mut Model, prefix: &[u8], k: usize, width: usize, max_len: usize, to_end: bool, traversal: &str) -> Vec<PathResult> {
    struct B {
        bytes: Vec<u8>,
        cost: f64,
        steps: Vec<f64>,
        units: Vec<usize>,
        done: bool,
    }
    let k = k.max(1);
    let width = width.max(1);
    let mut frontier = vec![B { bytes: prefix.to_vec(), cost: 0.0, steps: vec![], units: vec![], done: false }];
    let mut finished: Vec<B> = Vec::new();
    for _ in 0..max_len.max(1) {
        let mut next: Vec<B> = Vec::new();
        for b in frontier.drain(..) {
            if b.done {
                finished.push(b);
                continue;
            }
            let code = m.tok.encode(&b.bytes);
            let Ok(p) = m.pair.fold(&code, traversal, "") else { continue };
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.sort_by(|&a, &c| p[c].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &x in idx.iter().take(width) {
                if p[x] <= 0.0 {
                    continue;
                }
                let cost = -p[x].ln();
                let mut nb = B { bytes: b.bytes.clone(), cost: b.cost + cost, steps: b.steps.clone(), units: b.units.clone(), done: false };
                nb.steps.push(cost);
                nb.units.push(x);
                if x == pair::END {
                    nb.done = true;
                } else {
                    nb.bytes.push(x as u8);
                }
                next.push(nb);
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
        next.truncate(k);
        frontier = next;
        if frontier.iter().all(|b| b.done) {
            finished.extend(frontier.drain(..));
            break;
        }
    }
    let _ = to_end;
    finished.extend(frontier.drain(..));
    finished.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
    finished.truncate(k);
    finished
        .into_iter()
        .map(|b| PathResult {
            text: String::from_utf8_lossy(&b.bytes[prefix.len()..]).into_owned(),
            full_text: String::from_utf8_lossy(&b.bytes).into_owned(),
            units: b.units,
            cost: b.cost,
            step_costs: b.steps,
            traversal: traversal.into(),
            mode: "beam".into(),
            reached_end: b.done,
            peak: 0.0,
            bytes: b.bytes,
        })
        .collect()
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Predict {
    prefix: String,
    length: usize,
    mode: String,
    to_end: bool,
    temperature: f64,
    k: usize,
    beam: usize,
    traversal: String,
    backoff: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Generate {
    count: usize,
    max_length: usize,
    mode: String,
    prefix: String,
    temperature: f64,
    traversal: String,
}

fn parse<T: for<'a> Deserialize<'a>>(body: &str) -> Result<T, String> {
    serde_json::from_str(if body.is_empty() { "{}" } else { body }).map_err(|e| format!("bad request: {e}"))
}

fn value(body: &str) -> Result<Value, String> {
    serde_json::from_str(if body.is_empty() { "{}" } else { body }).map_err(|e| format!("bad request: {e}"))
}

/// Answers a ModelKit route, or None when the path is not one of them.
pub fn handle(method: &str, path: &str, query: &str, body: &str, state: &Arc<Mutex<(Model, String)>>, kit: &Arc<Kit>) -> Option<Result<Value, String>> {
    let r = match (method, path) {
        ("GET", "/api/health") => Ok(json!({ "ok": true, "version": format!("latentpair {}", env!("CARGO_PKG_VERSION")), "engine": "rust" })),
        ("GET", "/api/status") => {
            let s = state.lock().unwrap();
            let mut v = stats(&s.0, kit);
            v["engine"] = json!("rust");
            v["workers"] = json!(rayon::current_num_threads());
            v["routes"] = json!(routes(kit.checkpoint_dir.is_some()));
            v["job"] = kit.job.lock().unwrap().as_ref().map(Job::to_json).unwrap_or(Value::Null);
            v["backends"] = json!({});
            v["model_path"] = json!(s.1);
            v["checkpoint_dir"] = json!(kit.checkpoint_dir);
            Ok(v)
        }
        ("GET", "/api/model") => {
            let s = state.lock().unwrap();
            let c = &s.0.tok.config;
            Ok(json!({ "kind": KIND, "label": LABEL, "kinds": kinds(&s.0), "weights": Value::Null, "encoding": "latent",
                "unit": "byte", "window": c.window, "ngram": c.window, "stride": 1, "overlap": c.window.saturating_sub(1),
                "start_label": "<pad>", "end_label": "</s>", "back_label": "", "think_label": Value::Null,
                "levels": tokenizer::levels_string(&c.levels), "code_bits": c.bits(), "params": s.0.tok.param_count() }))
        }
        ("POST", "/api/model/select") => {
            let v = match value(body) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            let kind = v["kind"].as_str().unwrap_or("");
            if kind != KIND {
                return Some(Err(format!("this server holds one model kind, {KIND:?}; {kind:?} is not it")));
            }
            let s = state.lock().unwrap();
            Ok(json!({ "kind": KIND, "label": LABEL, "origin": "active", "model_path": s.1, "stats": stats(&s.0, kit) }))
        }
        ("GET", "/api/job") => Ok(kit.job.lock().unwrap().as_ref().map(Job::to_json).unwrap_or(Value::Null)),
        ("POST", "/api/job/stop") => {
            kit.stop.store(true, Ordering::Relaxed);
            Ok(kit.job.lock().unwrap().as_ref().map(Job::to_json).unwrap_or(Value::Null))
        }
        ("POST", "/api/train") => start_train(body, state, kit),
        ("POST", "/api/2nrl") => start_2nrl(body, state, kit),
        ("POST", "/api/predict") => (|| {
            let p: Predict = parse(body)?;
            let traversal = if p.traversal.is_empty() { "reward" } else { &p.traversal };
            let length = if p.length == 0 { 20 } else { p.length };
            let mut s = state.lock().unwrap();
            let m = &mut s.0;
            let mode = match p.mode.as_str() {
                "sample" => "sample",
                _ => "greedy",
            };
            let r = m.predict(&p.prefix, length, mode, traversal, if p.temperature <= 0.0 { 1.0 } else { p.temperature }, p.to_end, &p.backoff, 4096)?;
            let mut v = path_json(&p.prefix, &r);
            if p.mode == "beam" {
                let k = if p.k == 0 { 5 } else { p.k };
                let width = if p.beam == 0 { 5 } else { p.beam };
                let rows = beam(m, p.prefix.as_bytes(), k, width, length, p.to_end, traversal);
                v["top"] = json!(rows.iter().map(|r| path_json(&p.prefix, r)).collect::<Vec<_>>());
                v["bottom"] = json!([]);
                v["k"] = json!(k);
                v["beam"] = json!(width);
                v["mode"] = json!("beam");
            }
            v["guard"] = Value::Null;
            Ok(v)
        })(),
        ("POST", "/api/generate") => (|| {
            let g: Generate = parse(body)?;
            let count = g.count.clamp(1, 64);
            let max_len = if g.max_length == 0 { 60 } else { g.max_length.min(4096) };
            let traversal = if g.traversal.is_empty() { "reward" } else { &g.traversal };
            let mut s = state.lock().unwrap();
            let m = &mut s.0;
            let samples: Vec<Value> = match g.mode.as_str() {
                "beam" => beam(m, g.prefix.as_bytes(), count, 5, max_len, true, traversal).iter().map(|r| path_json(&g.prefix, r)).collect(),
                "sample" => (0..count)
                    .map(|_| m.predict(&g.prefix, max_len, "sample", traversal, if g.temperature <= 0.0 { 1.0 } else { g.temperature }, true, "", max_len))
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .map(|r| path_json(&g.prefix, r))
                    .collect(),
                _ => vec![path_json(&g.prefix, &m.predict(&g.prefix, max_len, "greedy", traversal, 1.0, true, "", max_len)?)],
            };
            Ok(json!({ "samples": samples, "guard": Value::Null, "mode": g.mode }))
        })(),
        ("POST", "/api/score") => (|| {
            let v = value(body)?;
            let text = v["text"].as_str().unwrap_or("");
            let traversal = v["traversal"].as_str().unwrap_or("reward");
            let backoff = v["backoff"].as_str().unwrap_or("");
            let mut s = state.lock().unwrap();
            let sc = s.0.score(text, traversal, backoff)?;
            let n = (text.len() + 1) as f64;
            let log_prob = -sc.bits * std::f64::consts::LN_2 * n;
            let unknown = {
                let codes = s.0.tok.encode_all(text.as_bytes());
                let d = s.0.pair.addr.d();
                (0..codes.len()).filter(|&i| s.0.pair.count.ctx[s.0.pair.addr.of(codes.at(i), d)] == 0).count()
            };
            let mut out = serde_json::to_value(&sc).map_err(|e| e.to_string())?;
            out["log_prob"] = json!(log_prob);
            out["per_char"] = json!(log_prob / n);
            out["chars"] = json!(text.len());
            out["transitions"] = json!(text.len() + 1);
            out["unknown_transitions"] = json!(unknown);
            Ok(out)
        })(),
        ("POST", "/api/feedback") => (|| {
            let v = value(body)?;
            let good = texts_of(&v, "good", "good_text");
            let bad = texts_of(&v, "bad", "bad_text");
            if good.is_empty() && bad.is_empty() {
                return Err("feedback needs good or bad texts".into());
            }
            let strength = v["strength"].as_f64().unwrap_or(0.0);
            let outcomes = v["outcomes"].as_u64().unwrap_or(0) as usize;
            let mut s = state.lock().unwrap();
            let action = if !good.is_empty() && !bad.is_empty() { "2nrl" } else if !good.is_empty() { "reward" } else { "punish" };
            let record = match action {
                "2nrl" => s.0.two_nrl(&bad, &good, strength, "", outcomes)?,
                "reward" => s.0.reward(&good, strength, None, true, "", outcomes)?,
                _ => s.0.punish(&bad, strength, None, "", outcomes)?,
            };
            let id = kit.next_job.fetch_add(1, Ordering::Relaxed);
            let job = Job { id, kind: "feedback".into(), state: "done".into(), progress: Some(record.clone()), history: vec![record], error: None, started_at: now(), finished_at: Some(now()) };
            Ok(json!({ "job": job.to_json(), "action": action, "good": good.len(), "bad": bad.len() }))
        })(),
        ("POST", "/api/invert") => {
            let mut s = state.lock().unwrap();
            s.0.pair.reward.invert();
            let mut inv = kit.inverted.lock().unwrap();
            *inv = !*inv;
            drop(inv);
            Ok(stats(&s.0, kit))
        }
        ("POST", "/api/save") => (|| {
            let v = value(body)?;
            let s = state.lock().unwrap();
            let path = v["path"].as_str().filter(|p| !p.is_empty()).map(String::from).unwrap_or_else(|| s.1.clone());
            s.0.save(&path)?;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            Ok(json!({ "saved": path, "path": path, "bytes": bytes }))
        })(),
        ("POST", "/api/load") => (|| {
            let v = value(body)?;
            let path = v["path"].as_str().filter(|p| !p.is_empty()).ok_or("load needs a path")?.to_string();
            let m = Model::load(&path)?;
            let mut s = state.lock().unwrap();
            s.0 = m;
            s.1 = path;
            Ok(stats(&s.0, kit))
        })(),
        ("POST", "/api/reset") => (|| {
            let v = value(body)?;
            let seed = v["seed"].as_i64().unwrap_or(0);
            let mut s = state.lock().unwrap();
            let settings = s.0.pair.settings.clone();
            let tok = s.0.tok.clone();
            s.0 = Model::prime(tok, settings, seed)?;
            *kit.inverted.lock().unwrap() = false;
            Ok(stats(&s.0, kit))
        })(),
        // the training history the Train tab tables: epoch records of the jobs run here (the model's own
        // call history is at /api/info)
        ("GET", "/api/history") => Ok(json!({ "history": *kit.training.lock().unwrap() })),
        ("GET", "/api/uploads") => Ok(json!({ "uploads": [], "upload_dir": Value::Null, "note": "this server keeps no upload directory: paste texts, or read files with the embedded app" })),
        ("POST", "/api/uploads") | ("POST", "/api/uploads/delete") => Err("this server keeps no upload directory".into()),
        ("GET", "/api/graph") => Ok(graph(&state.lock().unwrap().0, query_limit(query, 150))),
        ("GET", "/api/encoding") => {
            let s = state.lock().unwrap();
            Ok(encoding(&s.0))
        }
        ("POST", "/api/encoding/preview") => (|| {
            let v = value(body)?;
            let text = v["text"].as_str().unwrap_or("");
            let s = state.lock().unwrap();
            let codes = s.0.tok.encode_all(text.as_bytes());
            let windows: Vec<String> = (0..codes.len()).map(|i| format!("{:?}", codes.at(i))).collect();
            let mut out = encoding(&s.0);
            out["chars"] = json!(text.len());
            out["windows"] = json!(windows);
            out["count"] = json!(codes.len());
            out["decoded"] = json!(s.0.decode(text, 0).unwrap_or_default());
            out["round_trip"] = json!(false);
            out["unknown_windows"] = json!([]);
            out["kind"] = json!(KIND);
            out["path"] = json!({ "known": true, "reason": "every context has a node: the tree is primed", "labels": windows, "node_ids": (0..codes.len()).map(|i| s.0.pair.addr.of(codes.at(i), s.0.pair.addr.d())).collect::<Vec<_>>() });
            Ok(out)
        })(),
        ("GET", "/api/model/attention") => Ok(json!({ "kind": KIND, "attention": { "on": false, "applies": false, "blur": Value::Null, "weights": Value::Null, "unit": "byte", "units": "bytes", "default_blur": 0.0, "note": "the latent model has no attention band: a verdict credits the code's levels" } })),
        ("GET", "/api/model/window") => {
            let s = state.lock().unwrap();
            let w = s.0.tok.config.window;
            Ok(json!({ "kind": KIND, "window": { "on": false, "applies": false, "size": w, "top": w, "floor": w, "auto": false, "sizes": [w], "next": Value::Null, "unit": "byte", "units": "bytes", "ngram": w, "longer": Value::Null, "longest": w, "nodes": s.0.pair.addr.n, "heavy": Value::Null, "default_top": w, "default_floor": w, "note": "the window is the tokenizer's: fixed at training time" } }))
        }
        ("POST", "/api/model/attention") | ("POST", "/api/model/window") | ("POST", "/api/model/window/step") | ("POST", "/api/model/weights") => {
            Err("the latent model has no attention band, dynamic window or edge weights to set".into())
        }
        ("GET", "/api/checkpoints") => checkpoints_list(kit),
        ("POST", "/api/checkpoints/save") => checkpoint_save(body, state, kit),
        ("POST", "/api/checkpoints/restore") => checkpoint_restore(body, state, kit),
        _ => return None,
    };
    Some(r)
}

/// `limit=N` from a query string.
fn query_limit(query: &str, default: usize) -> usize {
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix("limit=").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(default)
}

/// The top context nodes by how often they were contexts, with their parent links.
pub fn graph(m: &Model, limit: usize) -> Value {
    let mut nodes: Vec<usize> = (0..m.pair.addr.n).filter(|&n| m.pair.count.ctx[n] > 0).collect();
    nodes.sort_by(|&a, &b| m.pair.count.ctx[b].cmp(&m.pair.count.ctx[a]));
    nodes.truncate(limit.max(2));
    if !nodes.contains(&0) {
        nodes.insert(0, 0);
    }
    let chosen: std::collections::HashSet<usize> = nodes.iter().copied().collect();
    let node_json: Vec<Value> = nodes
        .iter()
        .map(|&n| {
            let level = m.pair.addr.level(n);
            let code = m.pair.addr.code(n);
            let label = if level == 0 { "root".to_string() } else { m.tok.decode(&pad_code(&code, m.tok.depth()), level).unwrap_or_default().trim_start_matches('\u{b7}').to_string() };
            json!({ "id": n, "label": format!("{} {:?}", format!("{:?}", code), label), "count": m.pair.count.ctx[n], "level": level, "visits": m.pair.count.ctx[n] })
        })
        .collect();
    let edges: Vec<Value> = nodes
        .iter()
        .filter(|&&n| n != 0)
        .filter_map(|&n| {
            let p = m.pair.addr.parent(n);
            if !chosen.contains(&p) {
                return None;
            }
            let (cn, cp) = (m.pair.count.ctx[n] as f64, m.pair.count.ctx[p] as f64);
            let prob = if cp > 0.0 { cn / cp } else { 0.0 };
            Some(json!({ "source": p, "target": n, "weight": cn, "count": m.pair.count.ctx[n], "prob": prob, "cost": if prob > 0.0 { -prob.ln() } else { Value::Null.as_f64().unwrap_or(99.0) } }))
        })
        .collect();
    json!({ "nodes": node_json, "edges": edges })
}

fn pad_code(code: &[i32], depth: usize) -> Vec<i32> {
    let mut out = code.to_vec();
    out.resize(depth, 0);
    out
}

fn encoding(m: &Model) -> Value {
    let c = &m.tok.config;
    json!({ "window": c.window, "stride": 1, "overlap": c.window.saturating_sub(1), "start_label": "<pad>", "end_label": "</s>",
            "back_label": "", "think_label": Value::Null, "configurable": false, "unit": "byte",
            "note": format!("the trained tokenizer reads the last {} bytes and compresses them into a {:.0}-bit code ({}); the trees are keyed by that code", c.window, c.bits(), tokenizer::levels_string(&c.levels)) })
}

/// Starts a job unless one runs; the closure gets the shared state and the kit.
fn start_job(kind: &str, kit: &Arc<Kit>, work: impl FnOnce(Arc<Kit>, u64) -> Result<(), String> + Send + 'static) -> Result<Value, String> {
    let mut slot = kit.job.lock().unwrap();
    if slot.as_ref().map_or(false, |j| j.state == "running") {
        return Err("a job is already running; stop it or wait".into());
    }
    let id = kit.next_job.fetch_add(1, Ordering::Relaxed);
    let job = Job { id, kind: kind.into(), state: "running".into(), progress: None, history: vec![], error: None, started_at: now(), finished_at: None };
    *slot = Some(job.clone());
    drop(slot);
    kit.stop.store(false, Ordering::Relaxed);
    let kit2 = kit.clone();
    std::thread::spawn(move || {
        let outcome = work(kit2.clone(), id);
        let mut slot = kit2.job.lock().unwrap();
        if let Some(j) = slot.as_mut() {
            if j.id == id {
                j.state = match &outcome {
                    Ok(()) if kit2.stop.load(Ordering::Relaxed) => "stopped".into(),
                    Ok(()) => "done".into(),
                    Err(_) => "error".into(),
                };
                j.error = outcome.err();
                j.finished_at = Some(now());
            }
        }
    });
    Ok(json!({ "job": job.to_json() }))
}

fn record(kit: &Kit, id: u64, r: Value) {
    kit.training.lock().unwrap().push(r.clone());
    let mut slot = kit.job.lock().unwrap();
    if let Some(j) = slot.as_mut() {
        if j.id == id {
            j.progress = Some(r.clone());
            j.history.push(r);
        }
    }
}

fn start_train(body: &str, state: &Arc<Mutex<(Model, String)>>, kit: &Arc<Kit>) -> Result<Value, String> {
    let v = value(body)?;
    let texts = texts_of(&v, "texts", "text");
    if texts.is_empty() {
        return Err("train needs texts (`texts` or `text`; this server has no upload directory)".into());
    }
    let epochs = v["epochs"].as_u64().unwrap_or(1).clamp(1, 1000) as usize;
    let state = state.clone();
    start_job("train", kit, move |kit, id| {
        let started = Instant::now();
        for epoch in 1..=epochs {
            for t in &texts {
                if kit.stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                state.lock().unwrap().0.train(std::slice::from_ref(t));
            }
            kit.epochs_total.fetch_add(1, Ordering::Relaxed);
            let (loss, nodes, edges) = {
                let mut s = state.lock().unwrap();
                let loss = loss_of(&mut s.0, &texts);
                let st = stats(&s.0, &kit);
                (loss, st["trigrams"].clone(), st["edges"].clone())
            };
            *kit.last_loss.lock().unwrap() = Some(loss);
            record(&kit, id, json!({ "epoch": epoch, "loss": loss, "perplexity": loss.exp2(), "nodes": nodes, "edges": edges, "texts": texts.len(), "seconds": started.elapsed().as_secs_f64() }));
        }
        Ok(())
    })
}

fn start_2nrl(body: &str, state: &Arc<Mutex<(Model, String)>>, kit: &Arc<Kit>) -> Result<Value, String> {
    let v = value(body)?;
    let bad = texts_of(&v, "bad", "bad_text");
    let good = texts_of(&v, "good", "good_text");
    if bad.is_empty() && good.is_empty() {
        return Err("2nrl needs bad or good texts".into());
    }
    let neg = v["neg_epochs"].as_u64().unwrap_or(1).clamp(0, 1000) as usize;
    let pos = v["pos_epochs"].as_u64().unwrap_or(1).clamp(0, 1000) as usize;
    let strength = v["strength"].as_f64().unwrap_or(0.0);
    let outcomes = v["outcomes"].as_u64().unwrap_or(0) as usize;
    let state = state.clone();
    start_job("2nrl", kit, move |kit, id| {
        let started = Instant::now();
        for (phase, epochs, texts, sign) in [("negative", neg, &bad, -1.0), ("positive", pos, &good, 1.0)] {
            for epoch in 1..=epochs {
                for t in texts {
                    if kit.stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let mut s = state.lock().unwrap();
                    if sign < 0.0 {
                        s.0.punish(std::slice::from_ref(t), strength, None, "", outcomes)?;
                    } else {
                        s.0.reward(std::slice::from_ref(t), strength, None, true, "", outcomes)?;
                    }
                }
                let (loss, nodes, edges) = {
                    let mut s = state.lock().unwrap();
                    let loss = loss_of(&mut s.0, &good);
                    let st = stats(&s.0, &kit);
                    (loss, st["trigrams"].clone(), st["edges"].clone())
                };
                record(&kit, id, json!({ "phase": phase, "epoch": epoch, "loss": loss, "perplexity": loss.exp2(), "nodes": nodes, "edges": edges, "seconds": started.elapsed().as_secs_f64() }));
            }
        }
        kit.twonrl_runs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })
}

fn checkpoint_dir(kit: &Kit) -> Result<&str, String> {
    kit.checkpoint_dir.as_deref().ok_or_else(|| "no checkpoint directory: start the server with --checkpoint-dir".into())
}

fn checkpoints_list(kit: &Kit) -> Result<Value, String> {
    let dir = checkpoint_dir(kit)?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut list: Vec<(std::time::SystemTime, Value)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".json.gz") {
            continue;
        }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        let secs = modified.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        list.push((modified, json!({ "name": name, "bytes": meta.len(), "modified": secs, "path": entry.path().to_string_lossy() })));
    }
    list.sort_by(|a, b| b.0.cmp(&a.0));
    let latest = list.first().map(|(_, v)| v.clone());
    Ok(json!({ "checkpoints": list.into_iter().map(|(_, v)| v).collect::<Vec<_>>(), "latest": latest, "checkpoint_dir": dir }))
}

fn checkpoint_save(body: &str, state: &Arc<Mutex<(Model, String)>>, kit: &Kit) -> Result<Value, String> {
    let dir = checkpoint_dir(kit)?.to_string();
    let v = value(body)?;
    let tag: String = v["tag"].as_str().unwrap_or("checkpoint").chars().filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
    let tag = if tag.is_empty() { "checkpoint".to_string() } else { tag };
    let stamp = now().replace([':', '-'], "");
    let name = format!("{tag}-{stamp}.json.gz");
    let path = format!("{dir}/{name}");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    state.lock().unwrap().0.save(&path)?;
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    Ok(json!({ "name": name, "path": path, "bytes": bytes, "tag": tag }))
}

fn checkpoint_restore(body: &str, state: &Arc<Mutex<(Model, String)>>, kit: &Kit) -> Result<Value, String> {
    let dir = checkpoint_dir(kit)?.to_string();
    let v = value(body)?;
    let name = v["name"].as_str().unwrap_or("");
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return Err("restore needs the name of a checkpoint".into());
    }
    let m = Model::load(&format!("{dir}/{name}"))?;
    let mut s = state.lock().unwrap();
    s.0 = m;
    Ok(stats(&s.0, kit))
}
