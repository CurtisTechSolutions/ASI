//! Predicting the next thing: Kneser-Ney, the adaptive edges, base64, the file, and autotrain against a stand-in
//! Ollama that answers over a real socket with chunked replies.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread;

use latticefsm::autotrain::{autotrain, parse_judgements, parse_lines, Ollama, Options};
use latticefsm::b64;
use latticefsm::json::{parse, Json};
use latticefsm::ngram::{load_ngram, predict_now, tokenize, NGram};
use latticefsm::rng::Rng;

fn toy(order: usize) -> NGram {
    let mut m = NGram::new(order, "word", true, 1.0).unwrap();
    let lines: Vec<String> = [
        "the cat sat on the mat",
        "the cat ate the fish",
        "a dog sat on the log",
        "the dog ate the bone .",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    m.train_lines(&lines, None);
    m
}

#[test]
fn tokens_and_distributions() {
    assert_eq!(
        tokenize("Don't stop, OK?", "word", true),
        ["don't", "stop", ",", "ok", "?"]
    );
    let mut m = toy(3);
    for text in ["", "the", "the cat", "never seen"] {
        let ctx = m.context(&m.tokens(text));
        let d = m.distribution(&ctx);
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{text}");
        for (w, &p) in d.iter().enumerate() {
            assert_eq!(p, m.prob(&ctx, w as u32));
        }
    }
    assert_eq!(m.predict("the cat", 1)[0].0, "sat");
    assert_eq!(m.complete("the dog", 10, 0.0, &mut Rng::new(1)), "ate the fish");
}

#[test]
fn credit_moves_the_prediction() {
    let mut m = toy(3);
    let ctx = m.context(&m.tokens("the cat"));
    let ate = m.vocab.iter().position(|t| t == "ate").unwrap() as u32;
    let before = m.prob(&ctx, ate);
    m.credit_text("the cat ate", 3.0);
    assert!(m.prob(&ctx, ate) > before);
    m.credit_text("the cat ate", -10.0);
    assert!(m.prob(&ctx, ate) < before);
    let path = std::env::temp_dir().join(format!("latticefsm-ngram-{}.json.gz", std::process::id()));
    m.save(path.to_str().unwrap()).unwrap();
    let mut back = load_ngram(path.to_str().unwrap()).unwrap();
    assert_eq!(back.to_json().dump(), m.to_json().dump());
    assert_eq!(back.predict("the", 4), m.predict("the", 4));
}

#[test]
fn base64_everything() {
    let mut rng = Rng::new(3);
    for n in 0..40 {
        let data: Vec<u8> = (0..n).map(|_| (rng.random() * 256.0) as u8).collect();
        assert_eq!(b64::decode(&b64::encode(&data)), data);
        for k in 0..=n {
            assert!(b64::encode(&data).starts_with(&b64::encode_prefix(&data[..k])));
        }
    }
    assert_eq!(b64::encode(b"Hello"), "SGVsbG8=");
    assert_eq!(b64::encode_prefix(b"Hell"), "SGVsb");
    assert_eq!(b64::decode("SGV sbG8 !"), b"Hello");
    let mut m = NGram::new(6, "base64", true, 1.0).unwrap();
    let lines: Vec<String> = (0..30).map(|i| format!("the cat number {i} sat on the mat")).collect();
    m.train_lines(&lines, None);
    assert!(m.complete("the cat", 40, 0.0, &mut Rng::new(1)).starts_with("the cat"));
    let bytes = m.generate_bytes(b"the", 80, 0.0, &mut Rng::new(1)).unwrap();
    assert!(bytes.starts_with(b"the"));
}

#[test]
fn parsing_what_the_llm_says() {
    assert_eq!(
        parse_lines("Here they are:\n1. One.\n- Two\n\n3) \"Three\"", None),
        ["One.", "Two", "Three"]
    );
    assert_eq!(
        parse_judgements("ok: [{\"id\": 2, \"ok\": false}, {\"id\": 1, \"ok\": true}]", 3),
        [Some(true), Some(false), None]
    );
    assert_eq!(parse_judgements("1: yes\n2. no", 2), [Some(true), Some(false)]);
}

/// A stand-in Ollama on a socket: sentences, continuations and judgements, every reply chunked.
fn fake_ollama(requests: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    thread::spawn(move || {
        for stream in listener.incoming().take(requests) {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            let mut length = 0;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h == "\r\n" {
                    break;
                }
                if let Some(v) = h.to_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let reply = if first.starts_with("GET /api/tags") {
                Json::object().with(
                    "models",
                    Json::Array(vec![Json::object().with("name", "fake-model:latest".into())]),
                )
            } else {
                let prompt = parse(&String::from_utf8(body).unwrap())
                    .unwrap()
                    .str_or("prompt", "")
                    .to_string();
                let text = if prompt.contains("natural English sentences") {
                    "Sure:\n1. The garden is green.\n2. The garden is quiet.".to_string()
                } else if prompt.starts_with("Continue") {
                    "\"and then it rains.\"".to_string()
                } else {
                    "[{\"id\": 1, \"ok\": true}, {\"id\": 2, \"ok\": false}]".to_string()
                };
                Json::object().with("response", text.into()).with("done", true.into())
            };
            let data = reply.dump().into_bytes();
            let mut out =
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            for c in data.chunks(16) {
                out.extend(format!("{:x}\r\n", c.len()).as_bytes());
                out.extend(c);
                out.extend(b"\r\n");
            }
            out.extend(b"0\r\n\r\n");
            stream.write_all(&out).unwrap();
        }
    });
    addr
}

#[test]
fn autotrain_with_a_stand_in_ollama() {
    let addr = fake_ollama(1 + 2 * (2 + 2 + 1));
    let client = Ollama::new(Some(&addr), Some("fake-model"));
    assert_eq!(client.models().unwrap(), ["fake-model:latest"]);
    let mut m = NGram::new(3, "word", true, 1.0).unwrap();
    let o = Options {
        data: vec!["../data/corpus/sentences.txt".to_string()],
        held_out: 5,
        topics: vec!["gardens".to_string()],
        rounds: 2,
        lines: 2,
        continuations: 2,
        judge: 2,
        patience: 9,
        ..Options::default()
    };
    let mut log = Vec::new();
    let recs = autotrain(&mut m, &o, Some(&client), &mut |l| log.push(l.to_string())).unwrap();
    assert_eq!(recs.len(), 3);
    assert_eq!(recs[1].num("ollama_lines", 0.0), 4.0);
    assert_eq!(recs[1].num("continuations", 0.0), 2.0);
    assert_eq!(recs[1].num("approved", 0.0), 1.0);
    assert!(m.vocab.iter().any(|t| t == "garden"));
    assert!(m.credited_edges() > 0);
    assert_eq!(log.len(), 3);
    assert!(Ollama::new(Some("127.0.0.1:9"), None).models().is_err());
}

fn corpus_lines() -> Vec<String> {
    let text = std::fs::read_to_string("../data/corpus/prose.txt").unwrap();
    text.lines().map(String::from).collect()
}

/// The likeliest id by the whole distribution, ties to the lower id, never `<s>`.
fn argmax(m: &mut NGram, ctx: &[u32]) -> u32 {
    let d = m.distribution(ctx);
    let mut best = 1;
    for (w, &p) in d.iter().enumerate().skip(1) {
        if p > d[best] {
            best = w;
        }
    }
    best as u32
}

#[test]
fn predicting_while_learning() {
    let lines = corpus_lines();
    for (unit, order) in [("word", 3), ("word", 1), ("char", 5)] {
        let mut plain = NGram::new(order, unit, true, 1.0).unwrap();
        plain.train_lines(&lines, None);
        let mut learner = NGram::new(order, unit, true, 1.0).unwrap();
        learner.predict = true;
        let n = learner.train_lines(&lines, None);
        // predicting first changes nothing that is learned
        assert_eq!(learner.to_json().dump(), plain.to_json().dump(), "{unit} {order}");
        let (predicted, acc, ppl, bits) = learner.reset_online();
        // every token of the first PREDICT_ALL, a thinning sample after
        assert_eq!(predicted, (0..n as u64).filter(|&i| predict_now(i)).count() as u64);
        assert!(acc.unwrap() > 0.0 && acc.unwrap() < 1.0);
        assert!(ppl.unwrap() > 1.0 && (bits.unwrap() - ppl.unwrap().log2()).abs() < 1e-9);
        assert_eq!(learner.reset_online().0, 0);
        // the first guess, found without the whole distribution, is its argmax, on the tables kept up to date
        for line in lines.iter().take(40) {
            let toks = learner.tokens(line);
            for i in 0..toks.len() {
                let ctx = learner.context(&toks[..i]);
                assert_eq!(learner.top(&ctx), argmax(&mut learner, &ctx), "{unit} {order} {line}");
            }
        }
    }
}

#[test]
fn crediting_its_own_guesses() {
    let lines = corpus_lines();
    let mut m = NGram::new(3, "word", true, 1.0).unwrap();
    m.predict = true;
    m.self_credit = 0.5;
    m.train_lines(&lines, None);
    assert!(m.stats().num("credited_edges", 0.0) > 0.0);
    // the guesses after a credited context still match the full distribution's
    for line in lines.iter().take(20) {
        let toks = m.tokens(line);
        for i in 0..toks.len() {
            let ctx = m.context(&toks[..i]);
            assert_eq!(m.top(&ctx), argmax(&mut m, &ctx));
        }
    }
}
