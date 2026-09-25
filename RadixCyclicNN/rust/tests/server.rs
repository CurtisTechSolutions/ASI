//! The HTTP API end to end: a real server on a real port, answering the same
//! JSON contract the Python and Go servers answer (`../../DESIGN.md` §12).
//!
//! No HTTP client here either - the requests are written out, which is all a
//! test needs and keeps the crate's "no dependencies" rule where the library
//! keeps it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use radixnet::encoding::{Encoding, Unit};
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
    let encoding = if words {
        Encoding::new(Unit::Words, 3, 1).expect("a word encoding")
    } else {
        Encoding::default()
    };
    let mut model = Model::new(
        0,
        GraphOptions {
            encoding,
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
    send(port, method, path, "application/json", body.unwrap_or("").as_bytes())
}

/// One request with any body: `(status, body)`.
fn send(port: u16, method: &str, path: &str, content_type: &str, body: &[u8]) -> (u16, Json) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("the server");
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\
         Content-Type: {content_type}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).expect("the request head");
    // a server that refuses the body unread may have answered and hung up already
    let _ = stream.write_all(body);
    answer(stream)
}

/// A request that announces its body with `Expect: 100-continue` and sends
/// it only once the server has nodded, as curl does with a large file.
fn send_expecting(port: u16, path: &str, content_type: &str, body: &[u8]) -> (u16, Json) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("the server");
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\
         Content-Type: {content_type}\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).expect("the request head");
    // the nod, before a byte of the body goes
    let mut nod = Vec::new();
    let mut byte = [0u8; 1];
    while !nod.ends_with(b"\r\n\r\n") {
        assert_eq!(
            stream.read(&mut byte).expect("the nod"),
            1,
            "the connection closed before the nod"
        );
        nod.push(byte[0]);
    }
    assert!(
        nod.starts_with(b"HTTP/1.1 100 Continue"),
        "{}",
        String::from_utf8_lossy(&nod)
    );
    stream.write_all(body).expect("the request body");
    answer(stream)
}

/// A request whose body stops short of the length it declared: the client
/// hangs up after `body`.
fn send_short(port: u16, path: &str, content_type: &str, body: &[u8], declared: usize) -> (u16, Json) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("the server");
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {declared}\r\n\
         Content-Type: {content_type}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).expect("the request head");
    stream.write_all(body).expect("the request body");
    stream.shutdown(std::net::Shutdown::Write).expect("half a close");
    answer(stream)
}

/// The status and the JSON body of what the server wrote back.
fn answer(mut stream: TcpStream) -> (u16, Json) {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&chunk[..n]),
            // a reset after the answer arrived is the server hanging up on what it did not read
            Err(err) if !raw.is_empty() => {
                eprintln!("the connection ended after the answer: {err}");
                break;
            }
            Err(err) => panic!("no answer: {err}"),
        }
    }
    let answer = String::from_utf8_lossy(&raw);
    let status: u16 = answer
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = answer.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    (status, parse(body).unwrap_or(Json::Null))
}

/// Waits for the running job to end; `true` when it ended done.
fn job_done(port: u16) -> bool {
    for _ in 0..6000 {
        match get(port, "/api/job").1.at("state").as_str() {
            Some("running") => std::thread::sleep(std::time::Duration::from_millis(10)),
            Some("done") => return true,
            _ => return false,
        }
    }
    false
}

/// A browser's `FormData` with one file in it.
fn multipart_body(boundary: &str, filename: &str, data: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(data);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// A ZIP archive of stored (uncompressed) members, written out the plain way:
/// the local headers, the central directory and the end record, with the
/// CRC-32 the reader checks each member against.
fn stored_zip(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in members {
        let crc = radixnet::gzip::crc32(data);
        let offset = out.len() as u32;
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version, flags, stored, time, date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0x00, 0x08, 0, 0, 0, 0, 0, 0]); // made by, needed, UTF-8 names, stored, time, date
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0u8; 12]); // extra, comment, disk, attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let directory = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&directory.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

/// Lines enough to fill `bytes`, numbered so that no two are alike.
fn lines_of(bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes + 64);
    let mut i = 0usize;
    while out.len() < bytes {
        out.extend_from_slice(format!("the cat number {i} sat on mat {}\n", (i * 7) % 61).as_bytes());
        i += 1;
    }
    out
}

fn get(port: u16, path: &str) -> (u16, Json) {
    request(port, "GET", path, None)
}

fn post(port: u16, path: &str, body: &str) -> (u16, Json) {
    request(port, "POST", path, Some(body))
}

#[test]
fn the_model_endpoints_answer_the_same_contract() {
    let port = serve(trained(false), &format!("{}/model.count.json", temp_dir("server")));

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
    // every kind Python's service runs; words are an encoding, not a kind
    let kinds: Vec<String> = stats
        .at("kinds")
        .as_array()
        .iter()
        .filter_map(|k| k.at("kind").as_str().map(str::to_string))
        .collect();
    assert_eq!(kinds, vec!["radix", "count", "negative", "resonant"]);
    assert_eq!(stats.at("backends").at("torch").as_bool(), Some(false));
    assert_eq!(stats.at("backends").at("rust").as_bool(), Some(true));

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
    let port = serve(trained(false), &format!("{}/model.count.json", temp_dir("traversals")));
    let (_, named) = get(port, "/api/traversals");
    assert_eq!(
        named.at("traversals").to_strings(),
        vec!["reward", "punishment", "least-punished"]
    );

    let (status, _) = post(
        port,
        "/api/feedback",
        r#"{"bad":["the cat sat on the mat"],"strength":3}"#,
    );
    assert_eq!(status, 202);
    // the thumbs down is a job: the prices are read once it is done
    for _ in 0..200 {
        if get(port, "/api/job").1.at("state").as_str() != Some("running") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
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

/// Switching to a word encoding is what the frontend's model selector does, and
/// it has to be reachable from a server started on characters - otherwise the
/// Words tab can never appear. An encoding is fixed for a model's life, so the
/// switch is to a *different* model; the one that was running is parked with
/// its unsaved work rather than dropped, which is what the Python service does
/// for a kind.
#[test]
fn the_server_switches_between_the_two_encodings() {
    let dir = temp_dir("switch");
    let port = serve(trained(false), &format!("{dir}/model.count.json"));

    let (status, chosen) = post(port, "/api/model/select", r#"{"kind":"word"}"#);
    assert_eq!(status, 200);
    assert_eq!(chosen.at("kind").as_str(), Some("count"), "the kind never changes");
    assert_eq!(chosen.at("units").as_str(), Some("words"));
    assert_eq!(chosen.at("encoding").as_str(), Some("word:3:1"));
    // nothing was saved under that name yet, so it is a fresh model
    assert_eq!(chosen.at("origin").as_str(), Some("new"));
    assert_eq!(chosen.at("in_memory").to_strings(), vec!["word:3:1", "char:3:1"]);
    // each encoding has its own default file, derived from the one the server was given
    assert!(chosen
        .at("paths")
        .at("word")
        .as_str()
        .unwrap_or("")
        .ends_with("model.word.json"));
    assert!(chosen
        .at("paths")
        .at("count")
        .as_str()
        .unwrap_or("")
        .ends_with("model.count.json"));

    // and it is a working model: trainable, and counted in words
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
    let trained_words = stats.at("nodes").as_i64().unwrap_or(0);
    assert!(
        trained_words > 3,
        "the word model was not trained: {trained_words} nodes"
    );

    // the character model comes back as it was left, and so does the word one
    let (_, back) = post(port, "/api/model/select", r#"{"kind":"count"}"#);
    assert_eq!(back.at("encoding").as_str(), Some("char:3:1"));
    assert_eq!(back.at("origin").as_str(), Some("memory"));
    assert_eq!(back.at("units").as_str(), Some("chars"));
    let (_, again) = post(port, "/api/model/select", r#"{"kind":"word"}"#);
    assert_eq!(again.at("origin").as_str(), Some("memory"));
    assert_eq!(
        again.at("stats").at("nodes").as_i64(),
        Some(trained_words),
        "the parked word model lost its training"
    );

    // saving follows the active kind, so a word model never lands on the count model's file
    let (status, saved) = post(port, "/api/save", "{}");
    assert_eq!(status, 200);
    assert!(saved.at("saved").as_str().unwrap_or("").ends_with("model.word.json"));
}

/// Every way the Python contract lets a client name its texts: a list, one text
/// per line, and the uploads to read - all three, added together.
#[test]
fn the_texts_of_a_request_may_come_three_ways() {
    let dir = temp_dir("texts");
    let port = serve(trained(false), &format!("{dir}/model.count.json"));

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
        assert_eq!(get(port, "/api/job").1.at("state").as_str(), Some("done"), "{body}");
    }

    // feedback names its three good / good_text / good_files, and reads them the same way;
    // it is a job, as Python's is
    let (status, learned) = post(
        port,
        "/api/feedback",
        r#"{"good_text":"the cat sat on the mat","bad_text":"the cat ate the rat","strength":2}"#,
    );
    assert_eq!(status, 202);
    assert_eq!(learned.at("good").as_i64(), Some(1));
    assert_eq!(learned.at("bad").as_i64(), Some(1));
    assert_eq!(learned.at("action").as_str(), Some("2nrl"));
    assert_eq!(learned.at("job").at("type").as_str(), Some("feedback"));
    for _ in 0..200 {
        if get(port, "/api/job").1.at("state").as_str() != Some("running") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(get(port, "/api/job").1.at("state").as_str(), Some("done"));

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
    let port = serve(trained(true), &format!("{}/model.word.json", temp_dir("words")));
    let (status, stats) = get(port, "/api/status");
    assert_eq!(status, 200);
    assert_eq!(stats.at("kind").as_str(), Some("count"));
    assert_eq!(stats.at("units").as_str(), Some("words"));
    assert_eq!(stats.at("encoding").as_str(), Some("word:3:1"));

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
    let port = serve(trained(false), &format!("{}/model.count.json", temp_dir("refusals")));
    // a word alphabet on a character model
    let (status, error) = get(port, "/api/words");
    assert_eq!(status, 400);
    let says = error.at("error").as_str().unwrap_or("");
    assert!(
        says.contains("counts in chars") && says.contains("word encoding"),
        "{says}"
    );

    // a kind no server runs, in Python's words
    let (status, error) = post(port, "/api/model/select", r#"{"kind":"sine"}"#);
    assert_eq!(status, 400);
    assert!(error
        .at("error")
        .as_str()
        .unwrap_or("")
        .contains("unknown model kind 'sine'; expected one of: radix, count, negative, resonant"));
    // and what only the count model keeps
    post(port, "/api/model/select", r#"{"kind":"radix"}"#);
    let (status, error) = get(port, "/api/paths");
    assert_eq!(status, 400);
    assert_eq!(error.at("error").as_str(), Some("the radix model does not count paths"));

    // and a route that is not here at all
    let (status, error) = get(port, "/api/tutor/report");
    assert_eq!(status, 404);
    assert!(error.at("error").as_str().unwrap_or("").contains("no route"));
}

/// Every kind is selectable, as it is on the Python server: the model that was
/// running is parked with its unsaved work, a reset makes a fresh one of the
/// kind asked for (with its score function), and each answers the routes the
/// frontend's tabs use.
#[test]
fn every_kind_is_selected_trained_and_reset() {
    let dir = temp_dir("kinds");
    let port = serve(trained(false), &format!("{dir}/model.count.json"));
    let texts = r#"{"texts":["the cat sat on the mat","the dog sat on the log"],"epochs":2}"#;
    for kind in ["radix", "resonant", "negative"] {
        let (status, chosen) = post(port, "/api/model/select", &format!(r#"{{"kind":"{kind}"}}"#));
        assert_eq!(status, 200, "{kind}: {chosen:?}");
        assert_eq!(chosen.at("kind").as_str(), Some(kind));
        assert_eq!(chosen.at("origin").as_str(), Some("new"));
        let expected = format!("{dir}/model.count.{kind}.json");
        assert_eq!(chosen.at("model_path").as_str(), Some(expected.as_str()), "{kind}");
        let (status, _) = post(port, "/api/train", texts);
        assert_eq!(status, 202, "{kind}");
        for _ in 0..500 {
            if get(port, "/api/job").1.at("state").as_str() != Some("running") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(get(port, "/api/job").1.at("state").as_str(), Some("done"), "{kind}");
        let (_, status_doc) = get(port, "/api/status");
        assert_eq!(status_doc.at("kind").as_str(), Some(kind));
        assert!(
            status_doc.at("epochs_total").as_i64().unwrap_or(0) > 0 || kind == "negative",
            "{kind}"
        );
        let (status, graph) = get(port, "/api/graph?limit=5");
        assert_eq!(status, 200, "{kind}");
        assert!(graph.at("edges").as_array().iter().all(|e| e.get("source").is_some()));
    }
    // the count model comes back as it was left
    let (_, back) = post(port, "/api/model/select", r#"{"kind":"count"}"#);
    assert_eq!(back.at("origin").as_str(), Some("memory"));
    assert_eq!(back.at("kind").as_str(), Some("count"));
    let memory = back.at("in_memory").to_strings();
    for key in ["char:3:1", "radix", "resonant", "negative"] {
        assert!(memory.iter().any(|k| k == key), "{key} not in {memory:?}");
    }
    let (_, again) = post(port, "/api/model/select", r#"{"kind":"resonant"}"#);
    assert_eq!(again.at("origin").as_str(), Some("memory"));
    assert!(again.at("stats").at("epochs_total").as_i64().unwrap_or(0) > 0);

    // a reset takes the kind and its score function; the wrong settings are refused as Python refuses them
    let (status, reset) = post(
        port,
        "/api/reset",
        r#"{"kind":"resonant","buckets":12,"kick_scale":0.5}"#,
    );
    assert_eq!(status, 200, "{reset:?}");
    let (_, model) = get(port, "/api/model");
    assert_eq!(model.at("weights").at("buckets").as_i64(), Some(12));
    assert_eq!(model.at("weights").at("kick_scale").as_f64(), Some(0.5));
    let (status, error) = post(port, "/api/reset", r#"{"kind":"radix","window":5}"#);
    assert_eq!(status, 400);
    assert_eq!(
        error.at("error").as_str(),
        Some("weight options (window) do not apply to the radix model")
    );
    let (status, weights) = post(port, "/api/model/weights", r#"{"resonance_scale":2}"#);
    assert_eq!(status, 200, "{weights:?}");
    assert_eq!(weights.at("weights").at("resonance_scale").as_f64(), Some(2.0));
    post(port, "/api/model/select", r#"{"kind":"radix"}"#);
    let (status, error) = post(port, "/api/model/weights", r#"{"reward_scale":2}"#);
    assert_eq!(status, 400);
    assert!(error
        .at("error")
        .as_str()
        .unwrap_or("")
        .contains("no configurable weight function"));

    // the schedule language the Train tab previews
    let (_, schedule) = get(port, "/api/schedule");
    assert!(!schedule.at("presets").as_array().is_empty());
    let (status, preview) = post(
        port,
        "/api/schedule/preview",
        r#"{"lr_schedule":"linear(lr0, 4 * lr0)","epochs":3,"lr":0.1}"#,
    );
    assert_eq!(status, 200);
    let lrs: Vec<f64> = preview
        .at("points")
        .as_array()
        .iter()
        .filter_map(|p| p.at("lr").as_f64())
        .collect();
    assert_eq!(lrs.len(), 3);
    assert!((lrs[2] - 0.4).abs() < 1e-12);
    let (status, error) = post(port, "/api/schedule/preview", r#"{"lr_schedule":"lr0 +"}"#);
    assert_eq!(status, 400);
    assert!(error.at("error").as_str().unwrap_or("").contains("invalid syntax"));
}

/// `reverse` reads every text of a run backwards - the inline texts and the
/// uploads alike - so the model learns what comes before: its continuation of
/// a text's end, read backwards, is the whole text.  Anything but a boolean is
/// refused, as the Python and Go servers refuse it.
#[test]
fn a_reversed_run_learns_what_comes_before() {
    let dir = temp_dir("reverse");
    let empty = Model::new(0, GraphOptions::default()).expect("a model");
    let port = serve(empty, &format!("{dir}/model.count.json"));

    let (status, error) = post(port, "/api/train", r#"{"texts":["the cat sat"],"reverse":"yes"}"#);
    assert_eq!(status, 400);
    assert!(error
        .at("error")
        .as_str()
        .unwrap_or("")
        .contains("'reverse' must be a boolean"));

    let (status, _) = post(
        port,
        "/api/uploads",
        r#"{"name":"fox.txt","content":"quick brown fox\n"}"#,
    );
    assert_eq!(status, 200);
    let (status, _) = post(
        port,
        "/api/train",
        r#"{"texts":["lazy dog sleeps"],"files":["fox.txt"],"reverse":true,"epochs":2}"#,
    );
    assert_eq!(status, 202);
    for _ in 0..200 {
        if get(port, "/api/job").1.at("state").as_str() != Some("running") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(get(port, "/api/job").1.at("state").as_str(), Some("done"));

    let enc = Encoding::default();
    for (end, whole) in [("fox", "quick brown fox"), ("sleeps", "lazy dog sleeps")] {
        let body = format!(
            r#"{{"prefix":"{}","length":1,"to_end":true,"guard":false}}"#,
            enc.reverse(end)
        );
        let (status, found) = post(port, "/api/predict", &body);
        assert_eq!(status, 200, "{found:?}");
        let full = found.at("full_text").as_str().unwrap_or("").to_string();
        assert_eq!(enc.reverse(&full), whole, "backwards from {end:?}");
    }
}

/// The uploads route streams its body to disk rather than reading it whole,
/// so an archive is not held to the cap the JSON routes keep - the old 16 MiB
/// limit, which one source tree passes at once (D-035).  Multipart, a raw
/// body and the JSON form all take an archive larger than that, the listing
/// describes it, it trains, a body cut short leaves nothing behind, and the
/// cap still holds where a body is read whole.
#[test]
fn an_upload_of_any_size_streams_to_disk() {
    let dir = temp_dir("big-uploads");
    let port = serve(trained(false), &format!("{dir}/model.count.json"));
    let uploads = std::path::Path::new(&dir).join("uploads");
    let over_the_cap = radixnet::http::MAX_BODY + (1 << 20);

    let big = lines_of(over_the_cap);
    let big_lines = big.iter().filter(|&&b| b == b'\n').count();
    let zip = stored_zip(&[
        ("corpus/big.txt", &big),
        ("corpus/small.txt", b"a bird sang in the tree\n"),
        ("corpus/.DS_Store", b"\0\0"),
    ]);
    assert!(zip.len() > radixnet::http::MAX_BODY);

    // as multipart, what the browser sends
    let (status, doc) = send(
        port,
        "POST",
        "/api/uploads",
        "multipart/form-data; boundary=q",
        &multipart_body("q", "corpus.zip", &zip),
    );
    assert_eq!(status, 200, "{doc:?}");
    let record = doc.at("uploads").as_array()[0].clone();
    assert_eq!(record.at("name").as_str(), Some("corpus.zip"));
    assert_eq!(record.at("archive").as_bool(), Some(true));
    assert_eq!(record.at("bytes").as_i64(), Some(zip.len() as i64));
    assert_eq!(record.at("files").as_i64(), Some(2));
    assert_eq!(record.at("lines").as_i64(), Some(big_lines as i64 + 1));
    assert_eq!(record.at("skipped").as_i64(), Some(1));
    assert_eq!(record.at("replaced").as_bool(), Some(false));
    let archive = doc.at("archives").as_array()[0].clone();
    assert_eq!(archive.at("extracted").as_i64(), Some(2));
    assert_eq!(archive.at("entries").as_i64(), Some(3));
    assert_eq!(archive.at("bytes").as_i64(), Some(zip.len() as i64));
    assert_eq!(
        archive.at("skipped").as_array()[0].at("reason").as_str(),
        Some("system file")
    );
    assert_eq!(
        std::fs::read(uploads.join("corpus.zip")).unwrap(),
        zip,
        "kept whole, byte for byte"
    );

    // as a raw body, announced with Expect: 100-continue as curl announces a large file
    let (status, doc) = send_expecting(port, "/api/uploads?name=raw", "application/zip", &zip);
    assert_eq!(status, 200, "{doc:?}");
    assert_eq!(doc.at("uploads").as_array()[0].at("name").as_str(), Some("raw.zip"));
    assert_eq!(std::fs::read(uploads.join("raw.zip")).unwrap(), zip);

    // and as the JSON form, which carries the file inline and is read whole
    let json = format!(
        r#"{{"name":"inline.zip","content_base64":"{}"}}"#,
        radixnet::multipart::base64::encode(&zip)
    );
    assert!(json.len() > radixnet::http::MAX_BODY);
    let (status, doc) = post(port, "/api/uploads", &json);
    assert_eq!(status, 200, "{doc:?}");
    assert_eq!(doc.at("uploads").as_array()[0].at("files").as_i64(), Some(2));
    assert_eq!(std::fs::read(uploads.join("inline.zip")).unwrap(), zip);

    // a text file over the cap streams too, converted on the way: the byte-order mark goes
    let mut text = vec![0xef, 0xbb, 0xbf];
    text.extend_from_slice(&big);
    let (status, doc) = send(port, "POST", "/api/uploads?name=big.txt", "text/plain", &text);
    assert_eq!(status, 200, "{doc:?}");
    let record = doc.at("uploads").as_array()[0].clone();
    assert_eq!(record.at("lines").as_i64(), Some(big_lines as i64));
    assert_eq!(record.at("chars").as_i64(), Some(big.len() as i64));
    assert_eq!(record.at("bytes").as_i64(), Some(big.len() as i64));
    assert_eq!(std::fs::read(uploads.join("big.txt")).unwrap(), big);

    // the listing describes them all, and nothing half-written is in it
    let (status, listing) = get(port, "/api/uploads");
    assert_eq!(status, 200);
    let rows = listing.at("uploads").as_array();
    let names: Vec<&str> = rows.iter().filter_map(|u| u.at("name").as_str()).collect();
    assert_eq!(names, vec!["big.txt", "corpus.zip", "inline.zip", "raw.zip"]);
    let listed = rows.iter().find(|u| u.at("name").as_str() == Some("big.txt")).unwrap();
    assert_eq!(listed.at("lines").as_i64(), Some(big_lines as i64));
    assert_eq!(listed.at("chars").as_i64(), Some(big.len() as i64));
    assert!(listed.get("modified").is_some());
    let listed = rows
        .iter()
        .find(|u| u.at("name").as_str() == Some("corpus.zip"))
        .unwrap();
    assert_eq!(listed.at("lines").as_i64(), Some(big_lines as i64 + 1));

    // and an archive that came in this way trains, every line of every entry
    // (a small one: the big one takes a debug build most of a minute)
    let small = stored_zip(&[
        ("a/one.txt", b"the cat sat on the mat\nthe cat ran to the door\n"),
        ("two.txt", b"the dog sat on the log\n"),
    ]);
    let (status, doc) = send(
        port,
        "POST",
        "/api/uploads",
        "multipart/form-data; boundary=q",
        &multipart_body("q", "small.zip", &small),
    );
    assert_eq!(status, 200, "{doc:?}");
    let before = get(port, "/api/status").1.at("trained_texts").as_i64().unwrap_or(0);
    let (status, _) = post(port, "/api/train", r#"{"files":["small.zip"],"epochs":1}"#);
    assert_eq!(status, 202);
    assert!(job_done(port));
    let after = get(port, "/api/status").1.at("trained_texts").as_i64().unwrap_or(0);
    assert_eq!(after - before, 3);

    // a body shorter than its Content-Length is refused, and leaves nothing behind
    let (status, doc) = send_short(
        port,
        "/api/uploads?name=short.zip",
        "application/zip",
        &zip[..4096],
        zip.len(),
    );
    assert_eq!(status, 400, "{doc:?}");
    assert_eq!(doc.at("error").as_str(), Some("incomplete request body"));
    let left: Vec<String> = std::fs::read_dir(&uploads)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !left.iter().any(|n| n == "short.zip" || n.ends_with(".part")),
        "{left:?}"
    );

    // an archive that is not one, and one with nothing to train on, are refused the same way
    let (status, doc) = send(
        port,
        "POST",
        "/api/uploads?name=broken.zip",
        "application/zip",
        b"PK\x03\x04 and nothing else",
    );
    assert_eq!(status, 400);
    assert!(
        doc.at("error")
            .as_str()
            .unwrap_or("")
            .starts_with("broken.zip: not a valid ZIP archive"),
        "{doc:?}"
    );
    let empty = stored_zip(&[("only.png", b"\x89PNG\r\n\x1a\n\0\0")]);
    let (status, doc) = send(port, "POST", "/api/uploads?name=empty.zip", "application/zip", &empty);
    assert_eq!(status, 400);
    assert_eq!(
        doc.at("error").as_str(),
        Some("empty.zip holds no text files to train on (only.png: binary)")
    );
    assert!(!uploads.join("broken.zip").exists() && !uploads.join("empty.zip").exists());

    // a chunked body has no length to stream by: refused as the Python server refuses it
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("the server");
    stream
        .write_all(
            b"POST /api/uploads?name=chunked.txt HTTP/1.1\r\nHost: localhost\r\n\
              Transfer-Encoding: chunked\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n\
              5\r\nhello\r\n0\r\n\r\n",
        )
        .expect("the request");
    let (status, doc) = answer(stream);
    assert_eq!(status, 411, "{doc:?}");
    assert!(
        doc.at("error").as_str().unwrap_or("").contains("Content-Length"),
        "{doc:?}"
    );
    assert!(!uploads.join("chunked.txt").exists());

    // the cap still holds where a body is read whole: a JSON route
    let padding = "x".repeat(radixnet::http::MAX_BODY);
    let (status, doc) = post(port, "/api/predict", &format!(r#"{{"prefix":"{padding}"}}"#));
    assert_eq!(status, 400);
    assert!(
        doc.at("error").as_str().unwrap_or("").contains("larger than"),
        "{doc:?}"
    );
}
