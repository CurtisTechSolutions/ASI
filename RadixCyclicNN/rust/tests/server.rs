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
use radixnet::{Encoding, GraphOptions, Model, TrainOptions, Unit};

/// A port nothing is listening on, taken and released so the server can have it.
fn free_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("a free port");
    let port = listener.local_addr().expect("an address").port();
    drop(listener);
    port
}

/// A trained model in the given encoding: `Unit::Chars` is the character
/// trigram every other test uses, `Unit::Words` the word trigram.
fn trained(unit: Unit) -> Model {
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
            encoding: Encoding {
                unit,
                ..Default::default()
            },
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

/// A directory of this test's own, made fresh: the model files a server writes
/// and the uploads it reads live here, so no two tests tread on each other.
fn temp_dir(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("radixnet-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a test directory");
    dir.to_string_lossy().into_owned()
}

/// Starts a server on its own thread and hands back the port it answers on.
fn serve(model: Model, path: &str) -> u16 {
    let port = free_port();
    let mut service = Service::new(model, path.to_string(), 0, 1);
    // the uploads live beside the model, so `files` has something to read
    let uploads = std::path::Path::new(path)
        .parent()
        .map(|dir| dir.join("uploads"))
        .unwrap_or_else(|| std::path::PathBuf::from("uploads"));
    let _ = std::fs::create_dir_all(&uploads);
    service.upload_dir = Some(uploads.to_string_lossy().into_owned());
    let service = Arc::new(service);
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
    let port = serve(
        trained(Unit::Chars),
        &format!("{}/model.count.json", temp_dir("server")),
    );

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
    // one kind, because a word model is this same model under a word
    // encoding and not a kind of its own
    assert_eq!(stats.at("kinds").as_array().len(), 1);

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
    let port = serve(
        trained(Unit::Chars),
        &format!("{}/model.count.json", temp_dir("traversals")),
    );
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

/// Characters or words is the encoding, chosen when a model is made, not a
/// kind to switch between: `POST /api/reset` is where the frontend picks it,
/// and `GET /api/encoding` is where it reads it back.
#[test]
fn the_encoding_is_chosen_when_a_model_is_made() {
    let dir = temp_dir("encoding");
    let port = serve(trained(Unit::Chars), &format!("{dir}/model.count.json"));

    let (status, dial) = get(port, "/api/encoding");
    assert_eq!(status, 200);
    assert_eq!(dial.at("encoding").as_str(), Some("char:3:1"));
    assert_eq!(dial.at("unit").as_str(), Some("char"));
    assert_eq!(dial.at("ngram").as_i64(), Some(3));
    assert_eq!(dial.at("stride").as_i64(), Some(1));
    assert_eq!(dial.at("configurable").as_bool(), Some(true));

    // a fresh model in any encoding the dial can name
    let (status, reset) = post(port, "/api/reset", r#"{"encoding":"word:2:1"}"#);
    assert_eq!(status, 200);
    assert_eq!(reset.at("encoding").as_str(), Some("word:2:1"));
    assert_eq!(reset.at("stats").at("units").as_str(), Some("words"));

    let (_, dial) = get(port, "/api/encoding");
    assert_eq!(dial.at("unit").as_str(), Some("word"));
    assert_eq!(dial.at("overlap").as_i64(), Some(1));

    // the three dials on their own do the same job
    let (status, reset) = post(port, "/api/reset", r#"{"unit":"char","ngram":4,"stride":4}"#);
    assert_eq!(status, 200);
    assert_eq!(reset.at("encoding").as_str(), Some("char:4:4"));
    let (_, dial) = get(port, "/api/encoding");
    assert_eq!(dial.at("overlap").as_i64(), Some(0), "groups share nothing");

    // and it is a working model: trainable, and counted in its own units
    let (status, reset) = post(port, "/api/reset", r#"{"encoding":"word:3:1"}"#);
    assert_eq!(status, 200);
    assert_eq!(reset.at("stats").at("units").as_str(), Some("words"));
    let (status, _) = post(
        port,
        "/api/train",
        r#"{"texts":["the cat sat on the mat","the cat sat on the log"],"epochs":3}"#,
    );
    assert_eq!(status, 202, "training the word model");
    for _ in 0..200 {
        if get(port, "/api/job").1.at("state").as_str() != Some("running") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let (_, stats) = get(port, "/api/status");
    assert_eq!(stats.at("units").as_str(), Some("words"));
    assert!(
        stats.at("nodes").as_i64().unwrap_or(0) > 3,
        "the word model was not trained"
    );
    // the kind the server runs never changes with the encoding
    assert_eq!(stats.at("kind").as_str(), Some("count"));

    // one model, one file: saving lands on the path the server was given
    let (status, saved) = post(port, "/api/save", "{}");
    assert_eq!(status, 200);
    assert!(saved.at("saved").as_str().unwrap_or("").ends_with("model.count.json"));
}

/// Every way the Python contract lets a client name its texts: a list, one text
/// per line, and the uploads to read - all three, added together.
#[test]
fn the_texts_of_a_request_may_come_three_ways() {
    let dir = temp_dir("texts");
    let port = serve(trained(Unit::Chars), &format!("{dir}/model.count.json"));

    let (status, _) = post(
        port,
        "/api/uploads",
        r#"{"name":"corpus.txt","content":"the owl flew over the hill
the fox ran up the hill"}"#,
    );
    assert_eq!(status, 200);

    for body in [
        r#"{"texts":["the cat sat on the mat"],"epochs":1}"#,
        r#"{"text":"the cat sat on the mat
the cat ate the rat","epochs":1}"#,
        r#"{"files":["corpus.txt"],"epochs":1}"#,
    ] {
        let (status, _) = post(port, "/api/train", body);
        assert_eq!(status, 202, "{body}");
        for _ in 0..200 {
            if get(port, "/api/job").1.at("state").as_str() != Some("running") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(get(port, "/api/job").1.at("state").as_str(), Some("finished"), "{body}");
    }

    // feedback names its three good / good_text / good_files, and reads them the same way
    let (status, learned) = post(
        port,
        "/api/feedback",
        r#"{"good_text":"the cat sat on the mat","bad_text":"the cat ate the rat","strength":2}"#,
    );
    assert_eq!(status, 200);
    assert_eq!(learned.at("good").as_i64(), Some(1));
    assert_eq!(learned.at("bad").as_i64(), Some(1));

    // an upload name that tries to leave the directory is refused
    let (status, error) = post(port, "/api/train", r#"{"files":["../secrets.txt"],"epochs":1}"#);
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("upload name"));

    // and a request naming none of the three says which three
    let (status, error) = post(port, "/api/train", r#"{"epochs":1}"#);
    assert_eq!(status, 400);
    let says = error.at("error").as_str().unwrap_or("");
    for field in ["'texts'", "'text'", "'files'"] {
        assert!(says.contains(field), "{says}");
    }
}

#[test]
fn a_word_model_serves_its_alphabet() {
    let port = serve(trained(Unit::Words), &format!("{}/model.word.json", temp_dir("words")));
    let (status, stats) = get(port, "/api/status");
    assert_eq!(status, 200);
    assert_eq!(stats.at("kind").as_str(), Some("count"));
    assert_eq!(stats.at("units").as_str(), Some("words"));

    let (status, words) = get(port, "/api/words?limit=3");
    assert_eq!(status, 200);
    assert_eq!(words.at("units").as_str(), Some("words"));
    assert_eq!(words.at("encoding").as_str(), Some("word:3:1"));
    assert!(words.at("vocabulary").as_i64().unwrap_or(0) > 3);
    let rows = words.at("words").as_array();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].at("word").as_str(), Some("the"));
    // the id is the rank, most read first, because nothing records the order
    // the words were first read
    assert_eq!(rows[0].at("id").as_i64(), Some(0));
    assert_eq!(rows[0].at("grams").as_i64(), rows[0].at("trigrams").as_i64());

    let (status, found) = post(port, "/api/predict", r#"{"prefix":"the cat sat on","length":2}"#);
    assert_eq!(status, 200);
    let text = found.at("continuation").as_str().unwrap_or("");
    assert!(text == "the mat" || text == "the log", "{text:?}");
}

#[test]
fn what_it_does_not_serve_says_so() {
    let port = serve(
        trained(Unit::Chars),
        &format!("{}/model.count.json", temp_dir("refusals")),
    );
    // a word alphabet on a character model
    let (status, error) = get(port, "/api/words");
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("word encoding"));

    // a kind no Rust server runs
    let (status, error) = post(port, "/api/model/select", r#"{"kind":"resonant"}"#);
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("Python server"));

    // and a route that is not here at all
    let (status, error) = get(port, "/api/tutor/report");
    assert_eq!(status, 404);
    assert!(error.at("error").as_str().unwrap_or("").contains("no route"));
}
