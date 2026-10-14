//! The chain: one input to a node, the output node predicts the next, and the walk back through the input nodes
//! stops as soon as it has gathered enough confidence.

use latticefsm::chain::{load_chain, Chain};
use latticefsm::json::Json;
use latticefsm::ngram::{corpus_sequences, predict_now, split_lines, Sequence, PREDICT_ALL};
use latticefsm::rng::Rng;

fn corpus(unit: &str) -> (Vec<Sequence>, Vec<Sequence>) {
    let seqs: Vec<Sequence> = corpus_sequences(&["../data/corpus/prose.txt".to_string()], unit, true)
        .unwrap()
        .into_iter()
        .flat_map(|(_, s)| s)
        .collect();
    split_lines(&seqs, 10)
}

fn lines(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn abc_fills_nodes_one_to_three_and_node_four_predicts() {
    let mut m = Chain::new(4, "char", true, 1.0).unwrap();
    m.train_lines(&lines(&["abcd", "abcd", "abce", "abcd"]), None);
    let e = m.explain(&m.tokens("abc"), 3, None);
    let inputs: Vec<&str> = e
        .get("inputs")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert_eq!(inputs, ["a", "b", "c"]);
    let walk = e.get("walk").unwrap().as_array().unwrap();
    let nodes: Vec<f64> = walk.iter().map(|s| s.num("node", 0.0)).collect();
    assert_eq!(nodes, [4.0, 3.0, 2.0, 1.0]);
    assert_eq!(walk[0].get("input"), Some(&Json::Null));
    assert_eq!(walk[1].str_or("input", ""), "c");
    assert_eq!(e.get("next").unwrap().as_array().unwrap()[0].str_or("token", ""), "d");
}

#[test]
fn the_more_confident_the_shorter_the_walk() {
    let (train, test) = corpus("char");
    let mut m = Chain::new(8, "char", true, 0.9).unwrap();
    m.predict = false;
    m.train_sequences(&train, None);
    let sweep = m.sweep(&test, &[0.0, 0.3, 0.5, 0.7, 0.9, 1.0]);
    let depths: Vec<f64> = sweep.iter().map(|(_, e)| e.depth.unwrap()).collect();
    assert_eq!(depths[0], 0.0);
    assert!(depths.windows(2).all(|w| w[0] <= w[1]), "{depths:?}");
    assert!(sweep[5].1.accuracy > sweep[0].1.accuracy);
    let before = m.evaluate(&test, None).depth.unwrap();
    m.train_sequences(&train, None);
    assert!(m.evaluate(&test, None).depth.unwrap() < before);
    // reward shortens the walks of what it rewards; punishment lengthens them
    let text = "the brain is a directed cyclic graph";
    let mut ids = m.history_ids(&m.tokens(text));
    ids.push(1);
    let walked = |m: &Chain| (1..ids.len()).map(|i| m.walk(&ids[..i], None).unwrap()).sum::<usize>();
    let start = walked(&m);
    assert!(m.credit_text(text, 2.0) > 0);
    let rewarded = walked(&m);
    assert!(rewarded < start);
    m.credit_text(text, -6.0);
    assert!(walked(&m) > rewarded);
}

#[test]
fn predicting_while_learning_changes_nothing_learned() {
    let (train, _) = corpus("word");
    let mut plain = Chain::new(4, "word", true, 0.9).unwrap();
    plain.predict = false;
    plain.train_sequences(&train, None);
    let mut learner = Chain::new(4, "word", true, 0.9).unwrap();
    let n = learner.train_sequences(&train, None);
    assert_eq!(plain.to_json().dump(), learner.to_json().dump());
    let online = learner.reset_online();
    assert_eq!(online.predicted as usize, n);
    assert!(online.accuracy.unwrap() > 0.0 && online.depth.unwrap() > 0.0);
    assert_eq!(plain.reset_online().predicted, 0);
}

#[test]
fn a_long_run_predicts_a_thinning_sample() {
    // every position of the first PREDICT_ALL, then every 2nd of the next as many, every 3rd, ...
    assert!((0..PREDICT_ALL).all(predict_now));
    assert!(predict_now(PREDICT_ALL) && !predict_now(PREDICT_ALL + 1) && predict_now(PREDICT_ALL + 2));
    assert!(predict_now(3 * PREDICT_ALL) && !predict_now(2 * PREDICT_ALL + 1) && predict_now(2 * PREDICT_ALL + 2));
    let line = "the quick brown fox jumps over the lazy dog";
    let text: Vec<String> = (0..3000).map(|i| format!("{line} {i}")).collect();
    let mut plain = Chain::new(6, "char", true, 0.9).unwrap();
    plain.predict = false;
    plain.train_lines(&text, None);
    let mut learner = Chain::new(6, "char", true, 0.9).unwrap();
    let n = learner.train_lines(&text, None) as u64;
    assert!(n > 2 * PREDICT_ALL);
    // what is learned is the same, and the sample is exactly the positions predict_now picks
    assert_eq!(plain.to_json().dump(), learner.to_json().dump());
    let online = learner.reset_online();
    assert_eq!(online.predicted, (0..n).filter(|&i| predict_now(i)).count() as u64);
    assert!(online.predicted < n);
    // a reset starts the count again: every position of a short run is predicted
    let short = learner.train_lines(&lines(&["the cat sat"]), None) as u64;
    assert_eq!(learner.reset_online().predicted, short);
}

#[test]
fn the_file_and_generation() {
    let (train, _) = corpus("char");
    let mut m = Chain::new(5, "char", true, 0.8).unwrap();
    m.train_sequences(&train[..40], None);
    m.credit_text("the brain", 1.5);
    let path = std::env::temp_dir().join(format!("latticefsm-chain-{}.json.gz", std::process::id()));
    m.save(path.to_str().unwrap()).unwrap();
    let back = load_chain(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.to_json().dump(), m.to_json().dump());
    let t = m.tokens("the bra");
    assert_eq!(back.predict_tokens(&t, 5), m.predict_tokens(&t, 5));
    let (toks, depths) = m.continue_tokens(&t, 20, 0.0, &mut Rng::new(1));
    assert!(!toks.is_empty() && depths.len() >= toks.len());
    assert!(Chain::new(1, "char", true, 0.5).is_err());
    assert!(Chain::new(4, "char", true, 1.5).is_err());
}
