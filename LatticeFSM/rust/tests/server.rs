//! The HTTP API: every route, over a real socket, against one served machine.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use latticefsm::http::{self, request};
use latticefsm::json::{parse, Json};
use latticefsm::machine::{Machine, Settings};
use latticefsm::server::{content_type, safe_join, Service};

/// A stand-in for the built frontend: an index and one asset, in a directory of this test's own.
fn fake_dist() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("latticefsm-dist-{}-{}", std::process::id(), rand_suffix()));
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::write(dir.join("index.html"), "<!doctype html><title>LatticeFSM</title>").unwrap();
    std::fs::write(dir.join("assets/app.js"), "console.log('hi')").unwrap();
    dir
}

fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64
}

fn serve(requests: usize) -> String {
    serve_with(requests, Some(fake_dist()))
}

fn serve_with(requests: usize, frontend: Option<std::path::PathBuf>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let m = Machine::over(4, "ab", &[0], Settings::default()).unwrap();
    let service = Service::with_frontend(m, frontend);
    let handler = service.handler();
    thread::spawn(move || http::serve_on(listener, Arc::clone(&handler), Some(requests)));
    addr
}

#[test]
fn every_route_answers() {
    let addr = serve(17);
    let (status, health) = request(&addr, "GET", "/api/health", None).unwrap();
    assert_eq!(status, 200);
    assert_eq!(health.get("shape").unwrap().dump(), "[4,2,4]");

    let (status, page) = request(&addr, "GET", "/", None).unwrap();
    assert_eq!(status, 200);
    assert!(page.as_str().unwrap().contains("<title>LatticeFSM</title>"));

    let body = parse(r#"{"text":"abab","stimulation":2}"#).unwrap();
    let (status, run) = request(&addr, "POST", "/api/run", Some(&body)).unwrap();
    assert_eq!(status, 200);
    assert_eq!(run.get("transitions").unwrap().as_array().unwrap().len(), 4);
    assert_eq!(run.get("stats").unwrap().num("clock", 0.0), 4.0);

    let (_, credit) = request(
        &addr,
        "POST",
        "/api/credit",
        Some(&Json::object().with("amount", 1.0.into())),
    )
    .unwrap();
    assert_eq!(credit.num("credited", 0.0), 4.0);

    let quiet = parse(r#"{"text":"ab","quiet":true}"#).unwrap();
    let (_, q) = request(&addr, "POST", "/api/run", Some(&quiet)).unwrap();
    assert_eq!(q.get("stats").unwrap().num("clock", 0.0), 4.0);

    let train = parse(r#"{"language":"even-b","episodes":600}"#).unwrap();
    let (status, t) = request(&addr, "POST", "/api/train", Some(&train)).unwrap();
    assert_eq!(status, 200);
    assert!(t.num("after", 0.0) > 0.9, "{}", t.num("after", 0.0));
    assert_eq!(t.get("curve").unwrap().as_array().unwrap().len(), 10);

    let (_, matrix) = request(&addr, "GET", "/api/matrix?symbol=b", None).unwrap();
    assert_eq!(matrix.str_or("symbol", ""), "b");
    assert_eq!(matrix.get("rows").unwrap().as_array().unwrap().len(), 4);

    let (_, edge) = request(&addr, "GET", "/api/edge?source=0&symbol=a&target=0", None).unwrap();
    assert!(edge.num("seen", 0.0) > 0.0);

    let (_, table) = request(&addr, "GET", "/api/table", None).unwrap();
    assert_eq!(table.get("table").unwrap().as_array().unwrap().len(), 4);

    let (_, langs) = request(&addr, "GET", "/api/languages", None).unwrap();
    assert_eq!(langs.as_array().unwrap().len(), 4);

    let (_, teach) = request(
        &addr,
        "POST",
        "/api/teach",
        Some(&parse(r#"{"source":1,"symbol":"b","target":2,"amount":-1}"#).unwrap()),
    )
    .unwrap();
    let taught = teach.get("edge").unwrap();
    assert!(taught.num("punished", 0.0) >= 1.0); // training may have punished it already; this adds one
    assert_eq!(
        taught.num("last_punished", 0.0),
        teach.get("stats").unwrap().num("clock", -1.0)
    );

    let (_, ticked) = request(
        &addr,
        "POST",
        "/api/tick",
        Some(&Json::object().with("ticks", 500.0.into())),
    )
    .unwrap();
    assert!(ticked.num("clock", 0.0) > 500.0);

    let (_, stim) = request(
        &addr,
        "POST",
        "/api/stimulate",
        Some(&Json::object().with("amount", 2.0.into())),
    )
    .unwrap();
    assert_eq!(stim.num("stimulation", 0.0), 3.0);
    let (_, stim) = request(
        &addr,
        "POST",
        "/api/stimulate",
        Some(&Json::object().with("level", 0.5.into())),
    )
    .unwrap();
    assert_eq!(stim.num("stimulation", 0.0), 0.5);

    let (status, default) = request(&addr, "POST", "/api/new", Some(&Json::object())).unwrap();
    assert_eq!(status, 200);
    assert_eq!(default.get("shape").unwrap().dump(), "[13,13,13]");
    assert_eq!(default.num("edges", 0.0), 2197.0);

    let (status, fresh) = request(
        &addr,
        "POST",
        "/api/new",
        Some(&parse(r#"{"states":3,"alphabet":"xyz","accepting":[2]}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(status, 200);
    assert_eq!(fresh.get("shape").unwrap().dump(), "[3,3,3]");
}

#[test]
fn errors_are_reported() {
    let addr = serve(4);
    let (status, _) = request(&addr, "GET", "/api/nothing", None).unwrap();
    assert_eq!(status, 404);
    let (status, _) = request(&addr, "DELETE", "/api/stats", None).unwrap();
    assert_eq!(status, 405);
    let (status, err) = request(&addr, "POST", "/api/run", Some(&parse(r#"{"text":"abq"}"#).unwrap())).unwrap();
    assert_eq!(status, 400);
    assert!(err.str_or("error", "").contains("not in the alphabet"));
    let (status, _) = request(&addr, "GET", "/elsewhere", None).unwrap();
    assert_eq!(status, 200); // a single-page app's own route: index.html
}

#[test]
fn the_frontend_is_served_from_its_directory() {
    let addr = serve(4);
    let (status, asset) = request(&addr, "GET", "/assets/app.js", None).unwrap();
    assert_eq!(status, 200);
    assert_eq!(asset.as_str().unwrap(), "console.log('hi')");
    let (status, _) = request(&addr, "GET", "/index.html", None).unwrap();
    assert_eq!(status, 200);
    let (status, climbed) = request(&addr, "GET", "/../Cargo.toml", None).unwrap();
    assert_eq!(status, 200); // refused as a path, answered as the app's index
    assert!(climbed.as_str().unwrap().contains("<title>LatticeFSM</title>"));

    let without = serve_with(2, None);
    let (status, err) = request(&without, "GET", "/", None).unwrap();
    assert_eq!(status, 404);
    assert!(err.str_or("error", "").contains("no frontend directory"));
    let (status, _) = request(&without, "GET", "/api/health", None).unwrap();
    assert_eq!(status, 200);
}

#[test]
fn static_paths_are_safe_and_typed() {
    let root = std::path::Path::new("/srv/dist");
    assert_eq!(safe_join(root, ""), Some(root.join("index.html")));
    assert_eq!(safe_join(root, "assets/app.js"), Some(root.join("assets/app.js")));
    assert_eq!(safe_join(root, "../Cargo.toml"), None);
    assert_eq!(safe_join(root, "/etc/passwd"), None);
    assert_eq!(
        content_type(std::path::Path::new("a/index.html")),
        "text/html; charset=utf-8"
    );
    assert_eq!(
        content_type(std::path::Path::new("a/app.js")),
        "text/javascript; charset=utf-8"
    );
    assert_eq!(
        content_type(std::path::Path::new("a/app.css")),
        "text/css; charset=utf-8"
    );
    assert_eq!(
        content_type(std::path::Path::new("a/x.bin")),
        "application/octet-stream"
    );
}

#[test]
fn compression_routes_answer() {
    let addr = serve(12);
    let (_, none) = request(&addr, "GET", "/api/core", None).unwrap();
    assert!(none.str_or("error", "").contains("nothing has been compressed"));
    request(
        &addr,
        "POST",
        "/api/train",
        Some(&parse(r#"{"language":"even-b","episodes":400}"#).unwrap()),
    )
    .unwrap();
    let (status, c) = request(&addr, "POST", "/api/compress", Some(&Json::object())).unwrap();
    assert_eq!(status, 200);
    assert!(c.get("fidelity").unwrap().bool_or("lossless", false));
    assert_eq!(c.get("summary").unwrap().str_or("precision", ""), "exact");
    assert_eq!(c.get("expansion").unwrap().as_array().unwrap().len(), 3); // a 4-state machine has 3 shells
    let (_, c16) = request(
        &addr,
        "POST",
        "/api/compress",
        Some(&parse(r#"{"precision":"float16"}"#).unwrap()),
    )
    .unwrap();
    assert!(c16.get("summary").unwrap().num("bytes", 0.0) < c.get("summary").unwrap().num("bytes", 0.0));
    let (_, walk) = request(
        &addr,
        "POST",
        "/api/core/run",
        Some(&parse(r#"{"text":"ab","from_middle":true}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(walk.get("states").unwrap().as_array().unwrap()[0].as_f64(), Some(2.0));
    let (_, run) = request(
        &addr,
        "POST",
        "/api/run",
        Some(&parse(r#"{"text":"ab","from_middle":true,"quiet":true}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(run.get("states").unwrap().as_array().unwrap()[0].as_f64(), Some(2.0));
    let (_, sched) = request(
        &addr,
        "POST",
        "/api/compression",
        Some(&parse(r#"{"every":5,"precision":"float32","rebuild":true}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(
        (sched.num("compress_every", 0.0), sched.str_or("compress_precision", "")),
        (5.0, "float32")
    );
    let (_, after) = request(
        &addr,
        "POST",
        "/api/run",
        Some(&parse(r#"{"text":"abababab"}"#).unwrap()),
    )
    .unwrap();
    assert!(after.get("stats").unwrap().num("compressions", 0.0) >= 3.0);
    let (status, expanded) = request(&addr, "POST", "/api/expand", Some(&parse(r#"{"shells":1}"#).unwrap())).unwrap();
    assert_eq!(status, 200);
    assert!(expanded.get("stats").unwrap().num("touched", 99.0) <= 2.0);
    let (status, _) = request(&addr, "GET", "/api/core", None).unwrap();
    assert_eq!(status, 200);
}

#[test]
fn query_strings_decode() {
    assert_eq!(
        http::parse_query("a=1&b=x%20y&c=p+q&flag"),
        vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "x y".to_string()),
            ("c".to_string(), "p q".to_string()),
            ("flag".to_string(), String::new()),
        ]
    );
}
