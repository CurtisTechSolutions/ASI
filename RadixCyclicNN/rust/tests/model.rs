//! The model end to end: the structure it builds, what it predicts, and the
//! two traversals.

use radixnet::model::PredictOptions;
use radixnet::{GraphOptions, Model, TrainOptions, Traversal};

fn texts(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|s| s.to_string()).collect()
}

fn trained(lines: &[&str], epochs: usize) -> Model {
    let mut model = Model::new(0, GraphOptions::default()).unwrap();
    model.workers = 1;
    model.g.workers = 1;
    model
        .train(
            &texts(lines),
            &TrainOptions {
                epochs,
                ..Default::default()
            },
        )
        .unwrap();
    model
}

const CORPUS: &[&str] = &[
    "the cat sat on the mat",
    "the cat sat on the log",
    "the dog sat on the mat",
    "a bird flew over the hill",
    "the quick brown fox jumps over the lazy dog",
];

#[test]
fn training_keeps_every_invariant_and_every_text() {
    let model = trained(CORPUS, 3);
    model.g.check_invariants(&texts(CORPUS), true).expect("invariants");
    assert!(model.g.num_nodes() > 3, "the sentinels alone are not a graph");
    assert!(model.g.compression_ratio() > 1.0, "nothing was compressed");
}

#[test]
fn a_second_pass_changes_the_counts_but_not_the_structure() {
    let mut model = trained(CORPUS, 1);
    let (nodes, edges, trigrams) = (model.g.num_nodes(), model.g.num_edges(), model.g.num_trigrams());
    let before = model.g.total_traversals().float();
    model
        .train(
            &texts(CORPUS),
            &TrainOptions {
                epochs: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        (model.g.num_nodes(), model.g.num_edges(), model.g.num_trigrams()),
        (nodes, edges, trigrams)
    );
    assert!(model.g.total_traversals().float() > before);
}

#[test]
fn prediction_continues_the_prefix_it_was_given() {
    let mut model = trained(CORPUS, 5);
    let found = model
        .predict(
            "the cat sat on the ",
            &PredictOptions {
                length: 3,
                k: 3,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        found.best.full_text.starts_with("the cat sat on the "),
        "{:?}",
        found.best.full_text
    );
    assert!(!found.top.is_empty());
    for path in &found.top {
        assert!(path.cost.is_finite());
    }
    // the corpus continues that prefix with "mat" or "log" - or with "cat"
    // again, because the graph is cyclic and "the " leads back into "the cat"
    let tail = &found.best.full_text["the cat sat on the ".len()..];
    assert!(["mat", "log", "cat"].iter().any(|w| tail.starts_with(w)), "{tail:?}");
}

#[test]
fn a_trained_text_scores_better_than_gibberish() {
    let mut model = trained(CORPUS, 5);
    let known = model.score("the cat sat on the mat");
    let junk = model.score("qzx wvq kjh gfd");
    assert!(known.log_prob > junk.log_prob, "{known:?} vs {junk:?}");
    assert_eq!(known.unknown_transitions, 0);
    assert!(junk.unknown_transitions > 0);
}

#[test]
fn generation_from_the_start_sentinel_writes_a_whole_text() {
    let mut model = trained(CORPUS, 5);
    let out = model
        .generate(&radixnet::GenerateOptions {
            mode: "dijkstra".into(),
            max_length: 60,
            count: 1,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(!out[0].text.is_empty());
}

#[test]
fn the_least_punished_walk_leaves_a_step_that_was_judged_wrong() {
    // "mat" is rewarded five times over and punished once; "log" is never
    // judged at all.  The reward search takes the rewarded step; the
    // least-punished one will not touch it, however cheap the rewards made it.
    let mut model = trained(&["the cat sat on the mat", "the cat sat on the log"], 3);
    let good = texts(&["the cat sat on the mat"]);
    model.reward(&good, 1, 5.0).unwrap();
    model.punish(&good, 1, 1.0).unwrap();

    let prefix = "the cat sat on the ";
    let opts = |traversal: &str| PredictOptions {
        length: 6,
        k: 3,
        traversal: traversal.to_string(),
        ..Default::default()
    };
    let by_reward = model.predict(prefix, &opts("reward")).unwrap();
    let by_blame = model.predict(prefix, &opts("least-punished")).unwrap();

    assert_eq!(by_reward.best.full_text, "the cat sat on the mat");
    assert_eq!(by_blame.best.full_text, "the cat sat on the log");
    // and it is not that the blamed walk became cheap: it is dearer by an order
    // of magnitude, and taken anyway
    assert!(by_blame.best.cost > by_reward.best.cost);
    assert_eq!(by_blame.best.punish, 0.0);
    // the rewarded walk is still on offer, ranked behind everything unpunished
    let blamed = by_blame
        .top
        .iter()
        .find(|r| r.full_text == "the cat sat on the mat")
        .expect("still offered");
    assert!(blamed.punish > 0.0);
}

#[test]
fn with_nothing_punished_the_two_traversals_agree() {
    let mut model = trained(CORPUS, 5);
    for prefix in ["the ", "the cat", "a bird flew", "over the "] {
        let opts = |traversal: &str| PredictOptions {
            length: 8,
            k: 3,
            traversal: traversal.to_string(),
            ..Default::default()
        };
        let a = model.predict(prefix, &opts("reward")).unwrap();
        let b = model.predict(prefix, &opts("least-punished")).unwrap();
        assert_eq!(a.best.full_text, b.best.full_text, "prefix {prefix:?}");
        assert_eq!(a.best.cost, b.best.cost, "prefix {prefix:?}");
        assert_eq!(a.expanded, b.expanded, "prefix {prefix:?}");
    }
}

#[test]
fn punishment_is_not_bought_off_by_a_reward() {
    let mut model = trained(&["the cat sat on the mat", "the cat sat on the log"], 3);
    let good = texts(&["the cat sat on the mat"]);
    model.punish(&good, 1, 1.0).unwrap();
    model.reward(&good, 1, 50.0).unwrap();
    model.g.prepare();
    // the edge's own reward is now hugely positive, so the edge term is zero -
    // and the step is still punished, because the walk through it was judged
    // wrong once and that is not a thing a reward undoes
    let punished: Vec<f64> = model
        .g
        .path_contexts(0)
        .into_iter()
        .filter(|(_, row)| row.incorrect > 0)
        .map(|(key, _)| model.g.step_punishment(Some(key.prev), key.edge))
        .collect();
    assert!(!punished.is_empty(), "the punished pass recorded no context");
    assert!(
        punished.iter().all(|&p| p > 0.0),
        "a reward bought off the blame: {punished:?}"
    );
}

#[test]
fn the_traversal_names_round_trip() {
    for (name, want) in [
        ("", Traversal::Reward),
        ("reward", Traversal::Reward),
        ("least-punished", Traversal::LeastPunished),
        ("LEAST_PUNISHED", Traversal::LeastPunished),
        ("blame", Traversal::LeastPunished),
    ] {
        assert_eq!(radixnet::parse_traversal(name).unwrap(), want, "{name:?}");
    }
    assert!(radixnet::parse_traversal("sideways").is_err());
    assert_eq!(
        radixnet::parse_traversal(Traversal::LeastPunished.name()).unwrap(),
        Traversal::LeastPunished
    );
}

#[test]
fn the_workers_do_not_change_the_answer() {
    let corpus: Vec<String> = (0..500)
        .map(|i| format!("{} number {i}", CORPUS[i % CORPUS.len()]))
        .collect();
    let run = |workers: usize| {
        let mut model = Model::new(0, GraphOptions::default()).unwrap();
        model.workers = workers;
        model.g.workers = workers;
        let records = model
            .train(
                &corpus,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        let found = model
            .predict(
                "number 1",
                &PredictOptions {
                    length: 6,
                    k: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        (
            model.g.num_nodes(),
            model.g.num_edges(),
            model.g.num_trigrams(),
            records.last().unwrap().loss,
            found.best.full_text.clone(),
        )
    };
    assert_eq!(run(1), run(4));
}
