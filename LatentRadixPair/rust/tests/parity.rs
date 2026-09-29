//! The Rust port loads the Go implementation's files and reproduces its outputs.

use latentpair::model::Model;
use latentpair::tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    codes: Vec<CodeCase>,
    decodes: Vec<DecodeCase>,
    folds: Vec<FoldCase>,
    predictions: Vec<PredCase>,
    scores: Vec<ScoreCase>,
    info: serde_json::Value,
    coverage: std::collections::HashMap<String, f64>,
}
#[derive(Deserialize)]
struct CodeCase {
    bytes: Vec<u8>,
    codes: Vec<i32>,
}
#[derive(Deserialize)]
struct DecodeCase {
    context: String,
    known: usize,
    code: Vec<i32>,
    symbols: Vec<usize>,
}
#[derive(Deserialize)]
struct FoldCase {
    prefix: String,
    traversal: String,
    backoff: String,
    probs: Vec<f64>,
}
#[derive(Deserialize)]
struct PredCase {
    prefix: String,
    length: usize,
    traversal: String,
    backoff: String,
    to_end: bool,
    units: Vec<usize>,
    text: String,
    cost: f64,
    reached_end: bool,
    peak: f64,
}
#[derive(Deserialize)]
struct ScoreCase {
    text: String,
    traversal: String,
    bits: f64,
    mean_reward: f64,
    worst_penalty: f64,
    units: usize,
    per_unit: Vec<Vec<f64>>,
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn testdata(name: &str) -> String {
    format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn expected() -> Expected {
    serde_json::from_slice(&std::fs::read(testdata("expected.json")).unwrap()).unwrap()
}

#[test]
fn tokenizer_codes_and_decodes_match_go() {
    let tok = Tokenizer::load(&testdata("tokenizer.json")).unwrap();
    let e = expected();
    let (mut total, mut wrong) = (0, 0);
    for c in &e.codes {
        let got = tok.encode_all(&c.bytes);
        assert_eq!(got.data.len(), c.codes.len());
        for i in 0..got.len() {
            total += 1;
            if got.at(i) != &c.codes[i * got.d..(i + 1) * got.d] {
                wrong += 1;
            }
        }
    }
    assert!(wrong * 200 <= total, "{wrong} of {total} codes differ from Go");
    assert_eq!(wrong, 0, "{wrong} of {total} codes differ from Go (a rounding-boundary flip; investigate)");
    for d in &e.decodes {
        assert_eq!(tok.encode(d.context.as_bytes()), d.code, "code of {:?}", d.context);
        assert_eq!(tok.decode_symbols(&d.code, d.known).unwrap(), d.symbols, "decode of {:?} known {}", d.context, d.known);
    }
}

#[test]
fn model_folds_predictions_and_scores_match_go() {
    let mut m = Model::load(&testdata("model.json.gz")).unwrap();
    let e = expected();
    let info = m.info();
    for key in ["nodes", "cells", "nonzero_counts", "nonzero_rewards", "history"] {
        assert_eq!(info[key], e.info[key], "info {key}");
    }
    assert_eq!(info["read"]["units"], e.info["read"]["units"]);
    assert_eq!(info["judged"]["texts"], e.info["judged"]["texts"]);
    assert!(close(info["judged"]["rewards_total"].as_f64().unwrap(), e.info["judged"]["rewards_total"].as_f64().unwrap(), 1e-9));
    for f in &e.folds {
        let got = m.fold(&f.prefix, &f.traversal, &f.backoff).unwrap();
        assert_eq!(got.len(), f.probs.len());
        for (j, (&a, &b)) in got.iter().zip(&f.probs).enumerate() {
            assert!(close(a, b, 1e-9), "fold {:?} {} {} [{j}]: {a} vs {b}", f.prefix, f.traversal, f.backoff);
        }
    }
    for p in &e.predictions {
        let got = m.predict(&p.prefix, p.length, "greedy", &p.traversal, 0.0, p.to_end, &p.backoff, if p.to_end { 30 } else { 0 }).unwrap();
        assert_eq!(got.units, p.units, "predict {:?} {} {} to_end {}", p.prefix, p.traversal, p.backoff, p.to_end);
        assert_eq!(got.text, p.text);
        assert_eq!(got.reached_end, p.reached_end);
        assert!(close(got.cost, p.cost, 1e-9), "cost {} vs {}", got.cost, p.cost);
        assert!(close(got.peak, p.peak, 1e-9));
    }
    for s in &e.scores {
        let got = m.score(&s.text, &s.traversal, "").unwrap();
        assert_eq!(got.units, s.units, "score {:?}", s.text);
        assert!(close(got.bits, s.bits, 1e-9), "bits of {:?}: {} vs {}", s.text, got.bits, s.bits);
        assert!(close(got.mean_reward, s.mean_reward, 1e-9));
        assert!(close(got.worst_penalty, s.worst_penalty, 1e-9));
        assert_eq!(got.per_unit.len(), s.per_unit.len());
        for (u, pu) in got.per_unit.iter().zip(&s.per_unit) {
            assert!(close(u.bits, pu[0], 1e-9) && close(u.reward, pu[1], 1e-9) && close(u.penalty, pu[2], 1e-9));
        }
    }
    for (text, &want) in &e.coverage {
        assert!(close(m.coverage(text), want, 1e-9), "coverage of {text:?}");
    }
}

#[test]
fn a_model_saved_by_rust_reloads_and_still_matches() {
    let m = Model::load(&testdata("model.json.gz")).unwrap();
    let dir = std::env::temp_dir().join(format!("latentpair-parity-{}", std::process::id()));
    let path = dir.join("m.json.gz");
    m.save(path.to_str().unwrap()).unwrap();
    let mut back = Model::load(path.to_str().unwrap()).unwrap();
    let e = expected();
    for f in e.folds.iter().take(6) {
        let got = back.fold(&f.prefix, &f.traversal, &f.backoff).unwrap();
        for (&a, &b) in got.iter().zip(&f.probs) {
            assert!(close(a, b, 1e-9));
        }
    }
    assert_eq!(back.history.len(), m.history.len());
    assert_eq!(back.created, m.created);
    std::fs::remove_dir_all(dir).ok();
}
