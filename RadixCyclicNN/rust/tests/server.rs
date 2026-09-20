//! The HTTP API end to end: a real server on a real port, answering the same
//! JSON contract the Python and Go servers answer (`../../DESIGN.md` §12).
//!
//! No HTTP client here either - the requests are written out, which is all a
//! test needs and keeps the crate's "no dependencies" rule where the library
//! keeps it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use radixnet::json::{parse, Json};
use radixnet::service::{build, Service};
use radixnet::{GraphOptions, Model, TrainOptions};

/// A port nothing is listening on, taken and released so the server can have it.
fn free_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("a free port");
    let port = listener.local_addr().expect("an address").port();
    drop(listener);
    port
}

fn trained(words: bool) -> Model {
    let texts: Vec<String> = [
        "the cat sat on the mat",
        "the cat sat on the log",
        "the dog sat on the mat",
        "a bird flew over the hill",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut model = Model::new(
        0,
        GraphOptions {
            words,
            ..Default::default()
        },
    )
    .expect("a model");
    model
        .train(
            &texts,
            &TrainOptions {
                epochs: 2,
                ..Default::default()
            },
        )
        .expect("a trained model");
    model
}

/// Starts a server on its own thread and hands back the port it answers on.
fn serve(model: Model, path: &str) -> u16 {
    let port = free_port();
    let service = Arc::new(Service::new(model, path.to_string(), 0, 1));
    std::thread::spawn(move || {
        let _ = build(service, None).serve("127.0.0.1", port);
    });
    // the listener is up as soon as `serve` binds; give the thread a moment to get there
    for _ in 0..200 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    port
}

/// One request; returns `(status, body)`.
fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, Json) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("the server");
    let payload = body.unwrap_or("");
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\
         Content-Type: application/json\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).expect("the request head");
    stream.write_all(payload.as_bytes()).expect("the request body");
    let mut answer = String::new();
    stream.read_to_string(&mut answer).expect("the answer");
    let status: u16 = answer
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = answer.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    (status, parse(body).unwrap_or(Json::Null))
}

fn get(port: u16, path: &str) -> (u16, Json) {
    request(port, "GET", path, None)
}

fn post(port: u16, path: &str, body: &str) -> (u16, Json) {
    request(port, "POST", path, Some(body))
}

#[test]
fn the_model_endpoints_answer_the_same_contract() {
    let port = serve(trained(false), "/tmp/radixnet-test-server.json");

    let (status, health) = get(port, "/api/health");
    assert_eq!(status, 200);
    assert_eq!(health.at("status").as_str(), Some("ok"));
    assert_eq!(health.at("engine").as_str(), Some("rust"));

    let (status, stats) = get(port, "/api/status");
    assert_eq!(status, 200);
    assert_eq!(stats.at("kind").as_str(), Some("count"));
    assert_eq!(stats.at("units").as_str(), Some("chars"));
    assert_eq!(stats.at("engine").as_str(), Some("rust"));
    assert!(stats.at("nodes").as_i64().unwrap_or(0) > 3);
    assert_eq!(stats.at("kinds").as_array().len(), 2);

    let (status, model) = get(port, "/api/model");
    assert_eq!(status, 200);
    assert!(model.at("weights").get("global_scale").is_some());

    let (status, found) = post(port, "/api/predict", r#"{"prefix":"the cat","length":8,"k":2}"#);
    assert_eq!(status, 200);
    assert!(found.at("full_text").as_str().unwrap_or("").starts_with("the cat"));
    assert_eq!(found.at("traversal").as_str(), Some("reward"));
    assert_eq!(found.at("top").as_array().len(), 2);

    let (status, samples) = post(port, "/api/generate", r#"{"count":2,"max_length":20,"mode":"beam"}"#);
    assert_eq!(status, 200);
    assert_eq!(samples.at("samples").as_array().len(), 2);

    let (status, scored) = post(port, "/api/score", r#"{"text":"the cat sat on the mat"}"#);
    assert_eq!(status, 200);
    assert!(scored.at("log_prob").as_f64().unwrap_or(1.0) < 0.0);
    assert_eq!(scored.at("count").as_i64(), Some(1));

    for path in [
        "/api/graph?limit=5",
        "/api/paths?limit=3",
        "/api/nodes?limit=2",
        "/api/history",
    ] {
        let (status, _) = get(port, path);
        assert_eq!(status, 200, "{path}");
    }
}

#[test]
fn the_traversals_are_all_three_and_each_one_answers() {
    let port = serve(trained(false), "/tmp/radixnet-test-traversals.json");
    let (_, named) = get(port, "/api/traversals");
    assert_eq!(
        named.at("traversals").to_strings(),
        vec!["reward", "punishment", "least-punished"]
    );

    post(
        port,
        "/api/feedback",
        r#"{"bad":["the cat sat on the mat"],"strength":3}"#,
    );
    let mut costs = Vec::new();
    for traversal in ["reward", "punishment", "least-punished"] {
        let body = format!(r#"{{"prefix":"the cat","length":10,"traversal":"{traversal}","merit_scale":0}}"#);
        let (status, found) = post(port, "/api/predict", &body);
        assert_eq!(status, 200, "{traversal}");
        assert_eq!(found.at("traversal").as_str(), Some(traversal));
        assert!(!found.at("full_text").as_str().unwrap_or("").is_empty(), "{traversal}");
        costs.push(found.at("cost").as_f64().unwrap_or(0.0));
    }
    // the punishment traversal prices the steps differently, whether or not the argmax moves on
    // a corpus this small; the least-punished one leaves the prices alone
    assert_ne!(
        costs[0], costs[1],
        "the punishment traversal priced nothing differently"
    );
    assert_eq!(costs[0], costs[2], "the least-punished traversal changed a price");
}

#[test]
fn a_word_model_serves_its_alphabet() {
    let port = serve(trained(true), "/tmp/radixnet-test-words.json");
    let (status, stats) = get(port, "/api/status");
    assert_eq!(status, 200);
    assert_eq!(stats.at("kind").as_str(), Some("word"));
    assert_eq!(stats.at("units").as_str(), Some("words"));

    let (status, words) = get(port, "/api/words?limit=3");
    assert_eq!(status, 200);
    assert_eq!(words.at("units").as_str(), Some("words"));
    assert!(words.at("vocabulary").as_i64().unwrap_or(0) > 3);
    let rows = words.at("words").as_array();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].at("word").as_str(), Some("the"));

    let (status, found) = post(port, "/api/predict", r#"{"prefix":"the cat sat on","length":2}"#);
    assert_eq!(status, 200);
    let text = found.at("continuation").as_str().unwrap_or("");
    assert!(text == "the mat" || text == "the log", "{text:?}");
}

#[test]
fn what_it_does_not_serve_says_so() {
    let port = serve(trained(false), "/tmp/radixnet-test-refusals.json");
    // a word model's vocabulary on a character model
    let (status, error) = get(port, "/api/words");
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("vocabulary"));

    // a kind this server was not started with
    let (status, error) = post(port, "/api/model/select", r#"{"kind":"word"}"#);
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("--kind word"));

    // a kind no Rust server runs
    let (status, error) = post(port, "/api/model/select", r#"{"kind":"resonant"}"#);
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("Python server"));

    // and a route that is not here at all
    let (status, error) = get(port, "/api/tutor/report");
    assert_eq!(status, 404);
    assert!(error.at("error").as_str().unwrap_or("").contains("no route"));
}
