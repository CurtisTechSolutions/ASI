//! The HTTP API (`DESIGN.md` §16): the routes over a real socket, against one served body and one Nim body.
#![allow(clippy::field_reassign_with_default)]

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use gamelattice::server::{safe_join, ServeSettings, Service};
use latticefsm::http::{self, request};
use latticefsm::json::{parse, Json};

/// A results directory of this test's own, holding a small `e1.json` and no `e2.json`.
fn results_dir() -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    let dir = std::env::temp_dir().join(format!("gamelattice-results-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("e1.json"),
        r#"{"experiment":"E1","summary":{"game":{"encoder":0.96,"lattice":0.95}},"rows":[]}"#,
    )
    .unwrap();
    dir
}

/// Serve `requests` requests on an ephemeral port, with no frontend directory.
fn serve(requests: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let mut s = ServeSettings::default();
    s.results = results_dir();
    let service = Service::new(s).unwrap();
    let handler = service.handler();
    thread::spawn(move || http::serve_on(listener, Arc::clone(&handler), Some(requests)));
    addr
}

fn post(addr: &str, path: &str, body: &str) -> (u16, Json) {
    request(addr, "POST", path, Some(&parse(body).unwrap())).unwrap()
}

fn get(addr: &str, path: &str) -> (u16, Json) {
    request(addr, "GET", path, None).unwrap()
}

#[test]
fn the_routes_answer() {
    let addr = serve(16);
    let (status, health) = get(&addr, "/api/health");
    assert_eq!(status, 200);
    assert_eq!(health.str_or("language", ""), "even-b");
    assert_eq!(health.get("shape").unwrap().dump(), "[13,2,13]");
    assert_eq!(health.get("encoder").unwrap().num("window", 0.0), 8.0);

    let (status, langs) = get(&addr, "/api/languages");
    assert_eq!(status, 200);
    assert_eq!(langs.as_array().unwrap().len(), 4);

    let (status, fresh) = post(
        &addr,
        "/api/new",
        r#"{"language":"mod3-a","states":5,"window":4,"seed":2}"#,
    );
    assert_eq!(status, 200);
    assert_eq!(fresh.get("settings").unwrap().str_or("language", ""), "mod3-a");
    assert_eq!(fresh.get("shape").unwrap().dump(), "[5,2,5]");
    assert_eq!(fresh.num("episodes", -1.0), 0.0);

    let (status, trained) = post(&addr, "/api/train", r#"{"episodes":60}"#);
    assert_eq!(status, 200);
    assert_eq!(trained.num("episodes", 0.0), 60.0);
    assert_eq!(trained.num("test", 0.0), 300.0);
    let curve = trained.get("curve").unwrap().as_array().unwrap();
    assert_eq!(curve.len(), 11, "before, and every tenth of the way");
    assert_eq!(curve.last().unwrap().dump().split(',').next().unwrap(), "[60");
    let after = trained.get("after").unwrap();
    assert!((0.0..=1.0).contains(&after.num("encoder", -1.0)));
    assert!((0.0..=1.0).contains(&after.num("lattice", -1.0)));

    // classify is quiet: twice the same answer, and the status unchanged
    let (_, status_before) = get(&addr, "/api/status");
    let (status, a) = post(&addr, "/api/classify", r#"{"text":"abaa"}"#);
    assert_eq!(status, 200);
    let (_, b) = post(&addr, "/api/classify", r#"{"text":"abaa"}"#);
    assert_eq!(a.dump(), b.dump(), "two classifies answer the same");
    let (_, status_after) = get(&addr, "/api/status");
    assert_eq!(
        status_before.get("body").unwrap().dump(),
        status_after.get("body").unwrap().dump(),
        "a classify moves nothing"
    );
    assert!(!a.bool_or("learned", true));
    assert_eq!(a.num("states", 0.0), 5.0);
    let trace = a.get("trace").unwrap().as_array().unwrap();
    assert_eq!(trace.len(), 4);
    let named = a
        .get("classification")
        .unwrap()
        .get("named")
        .unwrap()
        .as_array()
        .unwrap();
    for (i, step) in trace.iter().enumerate() {
        assert_eq!(step.str_or("symbol", ""), &"abaa"[i..i + 1]);
        assert_eq!(step.get("aggregate").unwrap().as_array().unwrap().len(), 5);
        assert_eq!(step.num("named", -1.0), named[i].as_f64().unwrap());
        for key in [
            "margin",
            "price",
            "bidders",
            "predicted",
            "probability",
            "exploitability",
        ] {
            assert!(step.get(key).and_then(Json::as_f64).is_some(), "{key}");
        }
        assert!(step.get("triggers").unwrap().as_array().is_some());
        for seat in step.get("seats").unwrap().as_array().unwrap() {
            assert!(["current", "previous", "older", "child"].contains(&seat.str_or("bucket", "")));
            assert_eq!(seat.get("belief").unwrap().as_array().unwrap().len(), 6);
            assert_eq!(seat.get("phi"), Some(&Json::Null), "phi is known only after learning");
        }
    }
    let (status, bad) = post(&addr, "/api/classify", r#"{"text":"abc"}"#);
    assert_eq!(status, 400, "{}", bad.dump());

    let (status, m) = get(&addr, "/api/matrix?symbol=b");
    assert_eq!(status, 200);
    assert_eq!(m.str_or("symbol", ""), "b");
    let rows = m.get("rows").unwrap().as_array().unwrap();
    assert_eq!(rows.len(), 5);
    let cells = rows[0].get("edges").unwrap().as_array().unwrap();
    assert_eq!(cells.len(), 5);
    let total: f64 = cells.iter().map(|c| c.num("probability", 0.0)).sum();
    assert!((total - 1.0).abs() < 1e-9, "a row is a distribution: {total}");
    for key in ["seen", "net", "width"] {
        assert!(cells[0].get(key).and_then(Json::as_f64).is_some(), "{key}");
    }

    let (status, nim) = post(&addr, "/api/nim/play", r#"{"games":3}"#);
    assert_eq!(status, 200);
    assert_eq!(nim.get("records").unwrap().as_array().unwrap().len(), 3);
    let totals = nim.get("totals").unwrap();
    assert_eq!(totals.num("games", 0.0), 3.0);
    for key in ["win_rate", "legal_first_try", "refusals_per_move", "optimal_rate"] {
        assert!(totals.get(key).and_then(Json::as_f64).is_some(), "{key}");
    }
    assert!(nim.get("information_bits").and_then(Json::as_f64).is_some());

    let (status, results) = get(&addr, "/api/results");
    assert_eq!(status, 200);
    assert_eq!(results.get("e1").unwrap().str_or("experiment", ""), "E1");
    assert_eq!(results.get("e2"), Some(&Json::Null));
    assert_eq!(results.get("missing").unwrap().as_array().unwrap().len(), 1);

    let (status, nope) = get(&addr, "/api/nope");
    assert_eq!(status, 404);
    assert_eq!(nope.str_or("error", ""), "no such route");
}

#[test]
fn without_a_frontend_the_root_is_a_json_404() {
    let addr = serve(2);
    let (status, page) = get(&addr, "/");
    assert_eq!(status, 404);
    assert!(page.str_or("error", "").contains("no frontend directory"));
    let (status, _) = get(&addr, "/some/spa/route");
    assert_eq!(status, 404);
}

#[test]
fn a_learning_run_moves_the_body_and_carries_phi() {
    let addr = serve(3);
    let (status, run) = post(&addr, "/api/run", r#"{"text":"abba"}"#);
    assert_eq!(status, 200);
    assert!(run.bool_or("learned", false));
    assert_eq!(run.num("episodes", 0.0), 1.0);
    let seated: Vec<&Json> = run
        .get("trace")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s.get("seats").unwrap().as_array().unwrap().iter())
        .filter(|s| s.bool_or("seated", false))
        .collect();
    assert!(!seated.is_empty());
    assert!(seated.iter().all(|s| s.get("phi").and_then(Json::as_f64).is_some()));
    let (_, enc) = get(&addr, "/api/encoder");
    assert!(enc.get("graph").unwrap().num("nodes", 0.0) > 0.0);
    assert_eq!(enc.get("named").unwrap().as_array().unwrap().len(), 13);
    let (_, table) = get(&addr, "/api/table");
    assert_eq!(table.get("table").unwrap().as_array().unwrap().len(), 13);
}

#[test]
fn training_runs_as_a_job_and_a_second_is_refused() {
    let addr = serve(200);
    let (status, started) = post(&addr, "/api/train", r#"{"episodes":3000,"job":true}"#);
    assert_eq!(status, 200);
    let job = started.get("job").unwrap();
    assert_eq!(job.str_or("kind", ""), "train");
    assert_eq!(job.str_or("model", ""), "body");
    let id = job.num("id", 0.0);
    let (status, second) = post(&addr, "/api/nim/play", r#"{"games":2,"job":true}"#);
    assert_eq!(status, 409);
    assert!(second.bool_or("busy", false));
    // the job's thread holds the body a moment after it starts: a classify then answers busy, at once
    let mut saw_busy = false;
    for _ in 0..100 {
        let (status, r) = post(&addr, "/api/classify", r#"{"text":"ab"}"#);
        if status == 409 {
            assert!(r.bool_or("busy", false));
            saw_busy = true;
            break;
        }
        let (_, j) = get(&addr, "/api/job");
        if j.str_or("state", "") != "running" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (_, stopped) = post(&addr, "/api/job/stop", "{}");
    let mut last = Json::Null;
    for _ in 0..400 {
        let (_, j) = get(&addr, "/api/job");
        if j.str_or("state", "running") != "running" {
            last = j;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(last.num("id", 0.0), id);
    let state = last.str_or("state", "");
    assert!(state == "stopped" || state == "done", "{state}: {}", stopped.dump());
    assert!(last.get("curve").unwrap().as_array().is_some());
    if saw_busy {
        assert_eq!(state, "stopped", "a stop asked while it ran stops it");
        assert!(last.get("result").unwrap().bool_or("stopped", false));
    }
    // the body answers again
    let (status, _) = post(&addr, "/api/classify", r#"{"text":"ab"}"#);
    assert_eq!(status, 200);
}

#[test]
fn a_path_cannot_climb_out_of_the_frontend() {
    let root = PathBuf::from("/srv/dist");
    assert_eq!(safe_join(&root, ""), Some(root.join("index.html")));
    assert_eq!(safe_join(&root, "assets/app.js"), Some(root.join("assets/app.js")));
    assert_eq!(safe_join(&root, "../secret"), None);
    assert_eq!(safe_join(&root, "/etc/passwd"), None);
}
