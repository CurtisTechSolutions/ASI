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
fn walk_routes_answer() {
    let addr = serve(10);
    let (_, f) = request(&addr, "POST", "/api/focus", Some(&parse(r#"{"focus":1.0}"#).unwrap())).unwrap();
    assert_eq!(f.get("stats").unwrap().num("origin", -1.0), 3.0); // a 4-state machine: the bottom node is 3
    let (_, run) = request(
        &addr,
        "POST",
        "/api/run",
        Some(&parse(r#"{"text":"ab","quiet":true}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(run.get("states").unwrap().as_array().unwrap()[0].as_f64(), Some(3.0));
    let (status, _) = request(&addr, "POST", "/api/focus", Some(&parse(r#"{"focus":7}"#).unwrap())).unwrap();
    assert_eq!(status, 400);
    let (_, f) = request(
        &addr,
        "POST",
        "/api/focus",
        Some(&parse(r#"{"focus":null,"learn":true,"text":"ab"}"#).unwrap()),
    )
    .unwrap();
    assert!(f.get("learned").unwrap().num("focus", 1.0) < 0.1);
    let (_, sk) = request(
        &addr,
        "POST",
        "/api/skip",
        Some(&parse(r#"{"skip":true,"margin":0.5,"rearrange_every":7}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(
        (
            sk.bool_or("skip", false),
            sk.num("skip_margin", 0.0),
            sk.num("rearrange_every", 0.0)
        ),
        (true, 0.5, 7.0)
    );
    request(
        &addr,
        "POST",
        "/api/train",
        Some(&parse(r#"{"language":"even-b","episodes":200}"#).unwrap()),
    )
    .unwrap();
    let (_, re) = request(
        &addr,
        "POST",
        "/api/rearrange",
        Some(&parse(r#"{"full":true}"#).unwrap()),
    )
    .unwrap();
    assert!(re.get("stats").unwrap().num("swaps", 0.0) > 0.0);
    let (_, sw) = request(
        &addr,
        "POST",
        "/api/rearrange",
        Some(&parse(r#"{"axis":"states","i":0,"j":3}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(sw.get("swaps").unwrap().as_array().unwrap().len(), 1);
    let (status, _) = request(
        &addr,
        "POST",
        "/api/rearrange",
        Some(&parse(r#"{"axis":"rows","i":0,"j":3}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(status, 400);
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

#[test]
fn the_network_talks_and_learns() {
    let addr = serve(9);
    let (status, info) = request(&addr, "GET", "/api/network", None).unwrap();
    assert_eq!(status, 200);
    assert_eq!(info.str_or("architecture", ""), "enc>dec");
    assert!(info.bool_or("translator", false));

    let talk = parse(r#"{"text":"cat"}"#).unwrap();
    let (_, reply) = request(&addr, "POST", "/api/network/talk", Some(&talk)).unwrap();
    assert_eq!(reply.str_or("reply", "?"), "");
    assert_eq!(reply.get("trace").unwrap().as_array().unwrap().len(), 2);

    let teach = parse(r#"{"text":"cat","reply":"cats","times":6}"#).unwrap();
    let (status, taught) = request(&addr, "POST", "/api/network/teach", Some(&teach)).unwrap();
    assert_eq!(status, 200);
    assert_eq!(taught.get("after").unwrap().str_or("reply", ""), "cats");

    let fit = parse(r#"{"task":"plural","epochs":6}"#).unwrap();
    let (_, fitted) = request(&addr, "POST", "/api/network/fit", Some(&fit)).unwrap();
    assert!(
        fitted
            .get("exact")
            .unwrap()
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.5
    );
    let (_, reply) = request(
        &addr,
        "POST",
        "/api/network/talk",
        Some(&parse(r#"{"text":"dog"}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(reply.str_or("reply", ""), "dogs");
    let decoder = &reply.get("trace").unwrap().as_array().unwrap()[1];
    assert_eq!(decoder.str_or("english", ""), "dogs");

    let new = parse(r#"{"arch":"text>3"}"#).unwrap();
    let (_, info) = request(&addr, "POST", "/api/network/new", Some(&new)).unwrap();
    assert!(!info.bool_or("translator", true));
    let teach = parse(r#"{"text":"hello","reply":"yes","times":2}"#).unwrap();
    let (status, taught) = request(&addr, "POST", "/api/network/teach", Some(&teach)).unwrap();
    assert_eq!(status, 200);
    assert!(["yes", "no"].contains(&taught.get("after").unwrap().str_or("reply", "")));
    let bad = parse(r#"{"arch":"13>"}"#).unwrap();
    let (status, _) = request(&addr, "POST", "/api/network/new", Some(&bad)).unwrap();
    assert_eq!(status, 400);
}

#[test]
fn the_ngram_model_predicts_and_learns() {
    let addr = serve(9);
    let (status, info) = request(&addr, "GET", "/api/ngram", None).unwrap();
    assert_eq!(status, 200);
    assert_eq!(info.str_or("unit", ""), "word");
    let lines = parse(r#"{"text":"the cat sat on the mat\nthe cat ate the fish"}"#).unwrap();
    let (_, trained) = request(&addr, "POST", "/api/ngram/text", Some(&lines)).unwrap();
    assert!(trained.num("trained", 0.0) >= 13.0);
    // every token was predicted before it was learned
    let learned = trained.get("learned").unwrap();
    assert_eq!(learned.num("predicted", 0.0), trained.num("trained", 0.0));
    let acc = learned.num("accuracy", -1.0);
    assert!((0.0..=1.0).contains(&acc));
    assert!(learned.num("perplexity", 0.0) > 1.0);
    let quiet = parse(r#"{"text":"the dog sat","predict":false}"#).unwrap();
    let (_, t) = request(&addr, "POST", "/api/ngram/text", Some(&quiet)).unwrap();
    assert_eq!(t.get("learned").unwrap().num("predicted", -1.0), 0.0);
    let (_, p) = request(
        &addr,
        "POST",
        "/api/ngram/predict",
        Some(&parse(r#"{"text":"the cat","k":3}"#).unwrap()),
    )
    .unwrap();
    let next = p.get("next").unwrap().as_array().unwrap();
    assert_eq!(next.len(), 3);
    assert!(next[0].num("p", 0.0) >= next[1].num("p", 0.0));
    let (_, g) = request(
        &addr,
        "POST",
        "/api/ngram/generate",
        Some(&parse(r#"{"text":"the cat","max_tokens":5}"#).unwrap()),
    )
    .unwrap();
    assert!(!g.get("tokens").unwrap().as_array().unwrap().is_empty());
    let (_, c) = request(
        &addr,
        "POST",
        "/api/ngram/credit",
        Some(&parse(r#"{"text":"the cat sat","amount":1}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(c.num("credited", 0.0), 4.0);
    let (_, fresh) = request(
        &addr,
        "POST",
        "/api/ngram/new",
        Some(&parse(r#"{"order":4,"unit":"base64"}"#).unwrap()),
    )
    .unwrap();
    assert_eq!(fresh.num("order", 0.0), 4.0);
    let (_, o) = request(&addr, "GET", "/api/ngram/ollama?url=127.0.0.1:9", None).unwrap();
    assert!(!o.bool_or("available", true));
    let (status, _) = request(&addr, "POST", "/api/ngram/new", Some(&parse(r#"{"order":0}"#).unwrap())).unwrap();
    assert_eq!(status, 400);
}

fn post(addr: &str, path: &str, body: &str) -> (u16, Json) {
    request(addr, "POST", path, Some(&parse(body).unwrap())).unwrap()
}

#[test]
fn the_language_models_uploads_and_checkpoints() {
    let base = std::env::temp_dir().join(format!("latticefsm-files-{}-{}", std::process::id(), rand_suffix()));
    std::env::set_var("LATTICEFSM_UPLOADS", base.join("uploads"));
    std::env::set_var("LATTICEFSM_CHECKPOINTS", base.join("checkpoints"));
    let addr = serve(40);
    // the chain: one input to a node, the walk, the next inputs
    let (status, p) = post(&addr, "/api/chain/predict", r#"{"text":"the bra","k":3}"#);
    assert_eq!(status, 200);
    assert_eq!(p.get("inputs").unwrap().as_array().unwrap().len(), 7);
    assert!(!p.get("walk").unwrap().as_array().unwrap().is_empty());
    assert_eq!(p.get("next").unwrap().as_array().unwrap().len(), 3);
    let (_, g) = post(
        &addr,
        "/api/chain/generate",
        r#"{"text":"the brain","count":2,"max_tokens":20}"#,
    );
    let samples = g.get("samples").unwrap().as_array().unwrap();
    assert_eq!(samples.len(), 2);
    assert!(samples[0].get("depths").is_some());
    let (_, s) = post(&addr, "/api/chain/score", r#"{"text":"the cat"}"#);
    assert_eq!(s.get("tokens").unwrap().as_array().unwrap().len(), 8);
    assert!(s.num("perplexity", 0.0) > 1.0);
    // teaching a reply until every token of it is right
    let (_, t) = post(
        &addr,
        "/api/chain/teach",
        r#"{"text":"hello","reply":"hi there.","threshold":1}"#,
    );
    assert!(t.bool_or("reached", false));
    assert_eq!(t.num("right", 0.0), t.num("of", -1.0));
    assert_eq!(t.str_or("completion", "").trim(), "hi there.");
    let (_, t) = post(
        &addr,
        "/api/ngram/teach",
        r#"{"text":"hello","reply":"hi there .","threshold":0.5}"#,
    );
    assert!(t.bool_or("reached", false));
    assert_eq!(t.num("need", 0.0), 2.0); // half of "hi there ." (three tokens) and its end
                                         // uploads, in chunks, and training on them until N of N
    let (_, u) = post(
        &addr,
        "/api/uploads",
        r#"{"name":"notes/a.txt","content_base64":"dGhlIGNhdCBzYXQK"}"#,
    );
    assert_eq!(u.num("bytes", 0.0), 12.0);
    let (_, u) = post(
        &addr,
        "/api/uploads",
        r#"{"name":"notes/a.txt","content":"the cat ate\n","append":true}"#,
    );
    assert_eq!(u.num("bytes", 0.0), 24.0);
    let (_, list) = request(&addr, "GET", "/api/uploads", None).unwrap();
    let ups = list.get("uploads").unwrap().as_array().unwrap();
    assert_eq!(ups[0].str_or("name", ""), "notes/a.txt");
    assert_eq!(ups[0].str_or("kind", ""), "text");
    let (_, tr) = post(
        &addr,
        "/api/ngram/train",
        r#"{"uploads":["notes/a.txt"],"held_out":0,"until":true}"#,
    );
    assert_eq!(tr.num("trained", 0.0), 8.0);
    assert!(tr.bool_or("reached", false));
    let (status, _) = post(&addr, "/api/ngram/train", r#"{"uploads":["nope.txt"]}"#);
    assert_eq!(status, 400);
    let (_, sw) = post(&addr, "/api/chain/sweep", r#"{"thresholds":[0,1]}"#);
    let rows = sw.get("rows").unwrap().as_array().unwrap();
    assert_eq!(rows[0].num("depth", -1.0), 0.0);
    assert!(rows[1].num("depth", 0.0) > 1.0);
    let (_, gr) = request(&addr, "GET", "/api/chain/graph?limit=3", None).unwrap();
    assert_eq!(gr.get("nodes").unwrap().as_array().unwrap().len(), 3);
    // checkpoints: save, list, download, restore, delete
    let (_, saved) = post(&addr, "/api/checkpoints/save", r#"{"model":"chain","tag":"t1"}"#);
    let name = saved.str_or("name", "").to_string();
    assert!(name.starts_with("chain--t1--"), "{name}");
    let (_, list) = request(&addr, "GET", "/api/checkpoints", None).unwrap();
    assert_eq!(list.str_or("latest", ""), name);
    let (status, _) = request(&addr, "GET", &format!("/api/checkpoints/download?name={name}"), None).unwrap();
    assert_eq!(status, 200);
    let (status, r) = post(&addr, "/api/checkpoints/restore", &format!(r#"{{"name":"{name}"}}"#));
    assert_eq!(status, 200);
    assert_eq!(r.str_or("model", ""), "chain");
    let (status, _) = post(&addr, "/api/checkpoints/restore", r#"{"name":"../etc/passwd"}"#);
    assert_eq!(status, 400);
    let (_, st) = request(&addr, "GET", "/api/status", None).unwrap();
    assert_eq!(st.num("checkpoints", 0.0), 1.0);
    assert_eq!(st.num("uploads", 0.0), 1.0);
    let (status, _) = post(&addr, "/api/checkpoints/delete", &format!(r#"{{"name":"{name}"}}"#));
    assert_eq!(status, 200);
    let (_, c) = post(
        &addr,
        "/api/converse",
        r#"{"model":"chain","opening":"the brain is","turns":2}"#,
    );
    assert_eq!(c.get("turns").unwrap().as_array().unwrap().len(), 3);
    let (status, _) = post(&addr, "/api/chat/llm", r#"{"url":"127.0.0.1:9"}"#);
    assert_eq!(status, 502);
    let (_, d) = post(&addr, "/api/uploads/delete", r#"{"name":"notes/a.txt"}"#);
    assert_eq!(d.str_or("deleted", ""), "notes/a.txt");
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn the_guard_vetoes_what_it_was_told_failed() {
    let addr = serve(12);
    let (_, g) = post(&addr, "/api/chain/generate", r#"{"text":"the brain","max_tokens":30}"#);
    let reply = g.str_or("completion", "").trim().to_string();
    let blame = Json::object()
        .with("text", reply.as_str().into())
        .with("reason", "repeats itself".into())
        .with("severity", 2.0.into());
    let (status, b) = request(&addr, "POST", "/api/guard/blame", Some(&blame)).unwrap();
    assert_eq!(status, 200);
    assert!(b.num("blamed", 0.0) > 0.0);
    let judge = Json::object().with("texts", Json::strings(&[reply.clone(), "a lattice".to_string()]));
    let (_, j) = request(&addr, "POST", "/api/guard/judge", Some(&judge)).unwrap();
    let v = j.get("verdicts").unwrap().as_array().unwrap();
    assert_eq!(v[0].str_or("verdict", ""), "reject");
    assert_eq!(v[1].str_or("verdict", ""), "pass");
    // generating with the guard: the blamed greedy reply is vetoed, a sampled one served
    let (_, gg) = post(
        &addr,
        "/api/chain/generate",
        r#"{"text":"the brain","max_tokens":30,"guard":true}"#,
    );
    let summary = gg.get("guard").unwrap();
    assert!(summary.num("vetoed", 0.0) >= 1.0);
    assert!(!summary.bool_or("all_vetoed", true));
    let first = &gg.get("samples").unwrap().as_array().unwrap()[0];
    assert_ne!(first.get("guard").unwrap().str_or("verdict", ""), "reject");
    let (_, f) = post(
        &addr,
        "/api/ngram/feedback",
        r#"{"good":["the cat"],"bad":["buy now buy now"],"guard":true}"#,
    );
    assert_eq!(f.num("blamed", 0.0), 5.0);
    let (_, t) = post(&addr, "/api/network/talk", r#"{"text":"cat","guard":true}"#);
    assert!(t.get("guard").is_some());
    let (_, st) = request(&addr, "GET", "/api/guard", None).unwrap();
    assert_eq!(st.num("blamed", 0.0), 2.0);
    let (_, fo) = post(&addr, "/api/guard/forget", r#"{"reason":"repeats itself"}"#);
    assert!(fo.num("forgotten", 0.0) > 0.0);
    let (_, r) = post(&addr, "/api/guard/reset", "{}");
    assert_eq!(r.num("transitions", -1.0), 0.0);
}

fn get(addr: &str, path: &str) -> (u16, Json) {
    request(addr, "GET", path, None).unwrap()
}

/// Poll `GET /api/job` until job `id` is no longer running.
fn wait_for_job(addr: &str, id: f64) -> Json {
    for _ in 0..2000 {
        let (_, j) = get(addr, "/api/job");
        assert_eq!(j.num("id", -1.0), id);
        if j.str_or("state", "") != "running" {
            return j;
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("job {id} never ended");
}

#[test]
fn training_runs_as_a_job_and_the_server_answers_meanwhile() {
    let dir = std::env::temp_dir().join(format!("latticefsm-jobs-{}-{}", std::process::id(), rand_suffix()));
    std::fs::create_dir_all(&dir).unwrap();
    let big = dir.join("big.txt");
    let line = "the quick brown fox jumps over the lazy dog and the cat sat on the mat\n";
    std::fs::write(&big, line.repeat(60_000)).unwrap();
    let addr = serve(4000);
    assert_eq!(get(&addr, "/api/job").1, Json::Null);
    // a short job, start to finish: its result is what the plain request would have answered
    let (status, started) = post(
        &addr,
        "/api/chain/text",
        r#"{"text":"the cat sat\nthe dog sat","job":true}"#,
    );
    assert_eq!(status, 200);
    let job = started.get("job").unwrap();
    assert_eq!(job.str_or("model", ""), "chain");
    assert_eq!(job.str_or("kind", ""), "text");
    let done = wait_for_job(&addr, job.num("id", 0.0));
    assert_eq!(done.str_or("state", ""), "done");
    let result = done.get("result").unwrap();
    assert!(result.num("trained", 0.0) > 0.0);
    assert!(result.get("model").is_some());
    // a long one: the server answers meanwhile, the model it trains answers busy (409), a second job is refused
    let body = Json::object()
        .with("data", big.to_string_lossy().as_ref().into())
        .with("job", true.into());
    let (status, started) = request(&addr, "POST", "/api/chain/train", Some(&body)).unwrap();
    assert_eq!(status, 200);
    let id = started.get("job").unwrap().num("id", 0.0);
    assert_eq!(id, 2.0);
    // the job's thread takes the chain's lock a moment after it starts: wait for that (or for the job to end)
    let mut s = Json::Null;
    for _ in 0..500 {
        let (status, now) = get(&addr, "/api/status");
        assert_eq!(status, 200);
        assert_eq!(now.get("job").unwrap().num("id", 0.0), id);
        let held = now.get("chain").unwrap().get("busy") == Some(&Json::Bool(true));
        let ended = now.get("job").unwrap().str_or("state", "") != "running";
        s = now;
        if held || ended {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(2));
    }
    let running = s.get("job").unwrap();
    if running.str_or("state", "") == "running" {
        assert_eq!(s.get("chain").unwrap().get("busy"), Some(&Json::Bool(true)));
        let (status, busy) = post(&addr, "/api/chain/predict", r#"{"text":"the","k":3}"#);
        assert_eq!(status, 409);
        assert_eq!(busy.get("busy"), Some(&Json::Bool(true)));
        assert!(busy.str_or("error", "").starts_with("busy:"));
        assert_eq!(busy.get("job").unwrap().num("id", 0.0), id);
        let (status, _) = post(&addr, "/api/ngram/text", r#"{"text":"another model","job":true}"#);
        assert_eq!(status, 409);
        // the other models answer
        let (status, _) = post(&addr, "/api/ngram/predict", r#"{"text":"the","k":3}"#);
        assert_eq!(status, 200);
        // it reports how far it has got, and stops when asked, keeping what it learned
        let (status, stopping) = post(&addr, "/api/job/stop", "{}");
        assert_eq!(status, 200);
        assert_eq!(
            stopping.get("progress").unwrap().get("stopping"),
            Some(&Json::Bool(true))
        );
    }
    let ended = wait_for_job(&addr, id);
    let tokens = ended.get("progress").unwrap().num("tokens", 0.0);
    if ended.str_or("state", "") == "stopped" {
        assert!(ended.str_or("error", "").contains("stopped"));
        assert!(tokens < 60_000.0 * 73.0);
    } else {
        assert_eq!(ended.str_or("state", ""), "done");
    }
    let (status, info) = get(&addr, "/api/chain");
    assert_eq!(status, 200);
    assert!(info.num("tokens", 0.0) >= tokens);
    // nothing to stop now
    assert_eq!(post(&addr, "/api/job/stop", "{}").0, 400);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_reply_that_is_no_answer_is_drawn_again_hotter() {
    let addr = serve(40);
    // a model that has learned nothing ends every reply at once: nothing at every rung, and it says so
    post(&addr, "/api/ngram/new", r#"{"order":2,"unit":"word"}"#);
    let (status, r) = post(
        &addr,
        "/api/ngram/generate",
        r#"{"text":"hi","auto_temperature":true,"max_temperature":0.4,"tries":2}"#,
    );
    assert_eq!(status, 200);
    let auto = r.get("samples").unwrap().as_array().unwrap()[0]
        .get("auto")
        .unwrap()
        .clone();
    assert_eq!(auto.get("found"), Some(&Json::Bool(false)));
    assert_eq!(auto.str_or("why", ""), "nothing");
    let attempts = auto.get("attempts").unwrap().as_array().unwrap();
    let temps: Vec<f64> = attempts.iter().map(|a| a.num("temperature", -1.0)).collect();
    assert_eq!(temps, vec![0.0, 0.2, 0.2, 0.4, 0.4]);
    assert!(auto.str_or("note", "").starts_with("no answer up to 0.4"));
    // one that goes round: the greedy reply loops (a context comes back), a hotter one gets out
    post(
        &addr,
        "/api/ngram/text",
        r#"{"text":"a b a b a b a b\na b c d e f g\na b a b a b a b"}"#,
    );
    let (_, r) = post(
        &addr,
        "/api/ngram/generate",
        r#"{"text":"a","max_tokens":12,"auto_temperature":true}"#,
    );
    let sample = r.get("samples").unwrap().as_array().unwrap()[0].clone();
    let auto = sample.get("auto").unwrap();
    let attempts = auto.get("attempts").unwrap().as_array().unwrap();
    assert_eq!(attempts[0].str_or("why", ""), "a loop");
    assert_eq!(attempts[0].num("temperature", -1.0), 0.0);
    if auto.get("found") == Some(&Json::Bool(true)) {
        assert!(auto.num("temperature", 0.0) > 0.0);
        assert_eq!(sample.num("temperature", -1.0), auto.num("temperature", 0.0));
    }
    // without it the reply is the greedy one, and says nothing of a ladder
    let (_, plain) = post(&addr, "/api/ngram/generate", r#"{"text":"a","max_tokens":12}"#);
    let first = plain.get("samples").unwrap().as_array().unwrap()[0].clone();
    assert!(first.get("auto").is_none());
    assert_eq!(first.get("completion"), attempts[0].get("completion"));
    // guarded too: every sample still carries the guard's verdict
    let (status, g) = post(
        &addr,
        "/api/ngram/generate",
        r#"{"text":"a","max_tokens":12,"auto_temperature":true,"guard":true}"#,
    );
    assert_eq!(status, 200);
    assert!(g.get("samples").unwrap().as_array().unwrap()[0].get("guard").is_some());
    assert!(g.get("guard").unwrap().num("judged", 0.0) >= 1.0);
}
