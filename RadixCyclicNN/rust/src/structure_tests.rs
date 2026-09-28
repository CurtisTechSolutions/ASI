//! Splits and merges must not change what the count model hangs on its edges -
//! the recent shares, the verdicts and the prices - and the bottom beam of the
//! least-punished traversal finds the most punished paths (D-091).

use std::sync::atomic::Ordering;

use crate::correct::CorrectOptions;
use crate::graph::{Graph, FIRST, START};
use crate::model::PredictOptions;
use crate::paths::PathRow;
use crate::{GraphOptions, Model, TrainOptions};

const T1: &str = "abcdefghij klm"; // shares "abcdefghij " with T2, then parts
const T2: &str = "abcdefghij xyz";
const T3R: &str = "mnop tuv"; // shares nothing with the other two
const T3W: &str = "mnop tuw";

fn texts(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|s| s.to_string()).collect()
}

fn model(window: Option<usize>) -> Model {
    let mut opts = GraphOptions::default();
    if let Some(window) = window {
        opts.window = window;
    }
    let mut m = Model::new(1, opts).unwrap();
    m.workers = 1;
    m.g.workers = 1;
    m
}

fn train(m: &mut Model, lines: &[&str], epochs: usize, auto_compress: bool) {
    m.train(
        &texts(lines),
        &TrainOptions {
            epochs,
            auto_compress,
            ..Default::default()
        },
    )
    .unwrap();
}

fn node(g: &Graph, label: &str) -> usize {
    (FIRST..g.labels.len())
        .find(|&n| g.alive[n] && g.labels[n] == label)
        .unwrap_or_else(|| panic!("no node {label:?}"))
}

/// One text trained fifty times and a text that parts from it mid-node once:
/// the fork, and (probability, traversals, windowed traversals) of each way
/// on, by the first three units of the child's label.
fn fork(auto_compress: bool, window: Option<usize>) -> (Model, usize, Vec<(String, f64, i64, i64)>) {
    let mut m = model(window);
    train(&mut m, &["abcdefghij klmnop"], 50, auto_compress);
    train(&mut m, &["abcdefghij qrstuv"], 1, auto_compress);
    let forks: Vec<usize> = (FIRST..m.g.labels.len())
        .filter(|&n| m.g.alive[n] && m.g.degree(n) == 2)
        .collect();
    assert_eq!(forks.len(), 1, "one fork: {forks:?}");
    m.g.prepare();
    let mut ways: Vec<(String, f64, i64, i64)> =
        m.g.child_costs(forks[0])
            .into_iter()
            .map(|cc| {
                (
                    m.g.labels[cc.child][..3].to_string(),
                    (-cc.cost).exp(),
                    m.g.edge_count[cc.edge].load(Ordering::Relaxed),
                    m.g.window_edge_count[cc.edge],
                )
            })
            .collect();
    ways.sort_by(|a, b| a.0.cmp(&b.0));
    (m, forks[0], ways)
}

fn judged_pair() -> Model {
    let mut m = model(None);
    train(&mut m, &[T1, T1, T1, T1, T1, T3R], 1, true);
    for _ in 0..3 {
        // the model wrote xyz, the teacher wrote klm: blame the xyz step
        m.correct(T2, T1, &CorrectOptions::default()).unwrap();
    }
    let light = CorrectOptions {
        strength: 0.3,
        ..Default::default()
    };
    m.correct(T3W, T3R, &light).unwrap(); // a light blame on the tuw step
    m
}

/// Texts whose shared prefix the dynamic window has halved and holds apart.
fn halved(lines: &[&str]) -> Model {
    let mut m = model(None);
    train(&mut m, lines, 2, true);
    m.configure_window(Some(true), Some(8), Some(8), Some(8), Some(false))
        .unwrap();
    let step = m.window_step(1, true).unwrap();
    assert_eq!(step.splits, 1);
    m
}

fn window_off(m: &mut Model) {
    m.configure_window(Some(false), None, None, None, None).unwrap();
}

type Walk = (String, f64, f64);

fn ranking(m: &mut Model, traversal: &str, k: usize) -> (Vec<Walk>, Vec<Walk>) {
    let opts = PredictOptions {
        length: 0,
        to_end: true,
        max_length: Some(40),
        k,
        traversal: traversal.to_string(),
        ..Default::default()
    };
    let p = m.predict("", &opts).unwrap();
    let rows = |results: &[crate::search::PathResult]| -> Vec<Walk> {
        results
            .iter()
            .map(|r| {
                (
                    r.text.clone(),
                    (r.punish * 1e6).round() / 1e6,
                    (r.cost * 1e9).round() / 1e9,
                )
            })
            .collect()
    };
    (rows(&p.top), rows(&p.bottom))
}

fn rankings(m: &mut Model) -> Vec<(String, Vec<Walk>)> {
    ["reward", "least-punished"]
        .iter()
        .map(|t| (t.to_string(), ranking(m, t, 4).0))
        .collect()
}

#[test]
fn a_split_hands_the_bridge_its_window_history() {
    let (m, _, compressed) = fork(true, None);
    let (_, _, plain) = fork(false, None);
    // the bridge carries the fifty windowed traversals of the node it came out of, as the chain's
    // edge does, so the compressed graph and the chain price the new branch alike
    assert_eq!(compressed.len(), 2);
    for (c, p) in compressed.iter().zip(plain.iter()) {
        assert_eq!(c.0, p.0);
        assert_eq!((c.2, c.3), (p.2, p.3), "{}: (traversals, windowed)", c.0);
        assert!(
            (c.1 - p.1).abs() < 1e-12,
            "{}: probability {} compressed, {} on the chain",
            c.0,
            c.1,
            p.1
        );
    }
    assert_eq!((compressed[0].2, compressed[0].3), (50, 50), "the bridge");
    assert!(
        compressed[1].1 < 0.05,
        "one traversal in fifty-one is not a fifth of the way on: {}",
        compressed[1].1
    );
    m.g.check_invariants(&[], false).unwrap();
}

#[test]
fn the_rewritten_window_keeps_its_size() {
    let (m, _, ways) = fork(true, Some(60));
    assert_eq!(m.g.window_traversals(), 60);
    let sum: i64 = m.g.window_edge_count.iter().sum();
    assert_eq!(sum, 60, "every event in the window is counted once");
    assert!(ways[0].3 > 0, "the bridge holds part of the window: {ways:?}");
    assert!(ways[1].1 < 0.06, "the new branch: {ways:?}");
}

#[test]
fn a_split_hands_the_bridge_no_verdict() {
    let mut m = judged_pair();
    let before = rankings(&mut m);
    assert_eq!(before[1].1[0].0, T1);
    assert_eq!(before[1].1[1].0, T3R);
    let totals = m.g.path_totals();
    assert_eq!(m.g.split_window(8).unwrap(), 1, "one split of the shared prefix");
    let a = node(&m.g, "abcdefg");
    let bridge = m.g.children[a].edges[0];
    assert_eq!(m.g.edge_paths(bridge), (0, 0, 0), "a forced step carries no verdict");
    assert_eq!(
        m.g.path_totals(),
        totals,
        "the split moved the verdicts, it did not add any"
    );
    // the same walks, the same blame, the same prices - a split is invisible to both traversals
    assert_eq!(rankings(&mut m), before);
    m.g.check_invariants(&[], false).unwrap();
}

#[test]
fn a_merge_keeps_the_verdicts_and_the_prices() {
    let mut m = halved(&[T1, T2]);
    m.correct(T2, T1, &CorrectOptions::default()).unwrap(); // verdicts on the fork, which the first half now calls
    m.punish(&texts(&[T1]), 1, 0.5).unwrap();
    let a = node(&m.g, "abcdefg");
    let b = m.g.children[a].order[0];
    let rows = |g: &Graph, prev: usize, parent: usize| -> Vec<(usize, PathRow)> {
        let mut out: Vec<(usize, PathRow)> = g
            .paths
            .iter()
            .filter(|(key, _)| key.prev == prev && g.edge_parent[key.edge] == parent)
            .map(|(key, row)| (key.edge, *row))
            .collect();
        out.sort_by_key(|(e, _)| *e);
        out
    };
    let kept = rows(&m.g, a, b);
    assert_eq!(kept.len(), 2, "both ways out of the fork were judged");
    let before = rankings(&mut m);
    window_off(&mut m);
    assert_eq!(m.g.compress(), 1);
    // the fork's contexts are keyed by the merged node's caller now, counters intact
    assert_eq!(rows(&m.g, START, a), kept);
    assert_eq!(rankings(&mut m), before, "a merge changed what the model predicts");
    m.g.check_invariants(&[], true).unwrap();
}

#[test]
fn a_merge_carries_what_the_forced_step_alone_was_taught() {
    let mut m = halved(&[T1]);
    let a = m.g.children[START].order[0]; // the first half of the only text, held apart from the second
    let bridge = m.g.children[a].edges[0];
    let into = m.g.children[START].edges[0];
    m.g.add_reward(&[bridge], -1.0); // blame the forced step alone
    assert_eq!(ranking(&mut m, "least-punished", 1).0[0].1, 1.0);
    let ledger = m.g.total_reward();
    window_off(&mut m);
    assert_eq!(m.g.compress(), 1);
    assert_eq!(
        m.g.edge_reward[into], -1.0,
        "the blame moved onto the step into the merged node"
    );
    assert_eq!(m.g.total_reward(), ledger);
    assert_eq!(ranking(&mut m, "least-punished", 1).0[0].1, 1.0);

    // a pass that penalised every edge of the path moves nothing: the step into the node explains it
    let mut m = halved(&[T1]);
    m.punish(&texts(&[T1]), 1, 0.5).unwrap();
    let into = m.g.children[START].edges[0];
    let prices = ranking(&mut m, "reward", 1);
    window_off(&mut m);
    assert_eq!(m.g.compress(), 1);
    assert_eq!(
        m.g.edge_reward[into], -0.5,
        "the step into the merged node keeps its own penalty"
    );
    assert_eq!(ranking(&mut m, "reward", 1), prices);
}

#[test]
fn the_bottom_beam_finds_the_most_punished_paths() {
    let mut m = judged_pair();
    let (top, bottom) = ranking(&mut m, "least-punished", 1);
    assert_eq!(top[0].0, T1);
    assert_eq!(bottom.len(), 1, "{bottom:?}");
    assert_eq!(bottom[0].0, T2, "the walk the corrections blamed three times over");
    assert!(bottom[0].1 > 4.0, "its blame: {bottom:?}");
    let (_, by_cost) = ranking(&mut m, "reward", 1);
    assert_eq!(by_cost[0].0, T2, "by cost it is the dearest walk too");
    let (_, two) = ranking(&mut m, "least-punished", 2);
    let names: Vec<&str> = two.iter().map(|w| w.0.as_str()).collect();
    assert_eq!(names, vec![T2, T3W], "most punished first");
}
