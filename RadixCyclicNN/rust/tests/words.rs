//! The word n-gram model: the same graph over an alphabet whose symbols are
//! words (`../../SPEC-WordNGrams.md` §10 is the table these follow).

use radixnet::model::{GenerateOptions, PredictOptions};
use radixnet::words::{WordRow, WORD_UNITS};
use radixnet::{GraphOptions, Model, TrainOptions};

const CORPUS: &[&str] = &[
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ate the bone in the garden",
    "a bird sang in the garden",
];

fn texts(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|s| s.to_string()).collect()
}

fn trained(lines: &[&str], epochs: usize) -> Model {
    let mut model = Model::new(
        0,
        GraphOptions {
            words: true,
            ..Default::default()
        },
    )
    .unwrap();
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

#[test]
fn a_word_model_is_the_count_model_over_words() {
    let model = trained(CORPUS, 3);
    assert_eq!(model.kind(), "word");
    assert_eq!(model.units(), WORD_UNITS);
    assert_eq!(model.format(), "radixnet-word");
    // the invariants are the character model's, over the symbols the graph holds
    let symbols: Vec<String> = CORPUS.iter().map(|t| model.g.symbols_of(t)).collect();
    model.g.check_invariants(&symbols, true).expect("invariants");
    assert!(model.g.compression_ratio() > 1.0);
}

#[test]
fn a_repeated_phrase_becomes_one_node() {
    let model = trained(&["the cat sat on the mat", "a dog sat on the mat"], 1);
    let labels: Vec<String> = (0..model.g.num_node_ids())
        .filter(|&n| model.g.is_alive(n))
        .map(|n| model.g.text_of(model.g.label(n)))
        .collect();
    assert!(labels.iter().any(|l| l == "sat on the mat"), "{labels:?}");
    assert!(labels.iter().any(|l| l == "the cat sat on"), "{labels:?}");
}

#[test]
fn a_prediction_is_words_and_lengths_are_counted_in_them() {
    let mut model = trained(CORPUS, 3);
    let found = model
        .predict(
            "the cat sat on",
            &PredictOptions {
                length: 2,
                k: 3,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        found.best.text == "the mat" || found.best.text == "the rug",
        "{:?}",
        found.best.text
    );
    assert_eq!(found.best.full_text, format!("the cat sat on {}", found.best.text));
    for label in &found.best.labels {
        assert!(
            !label.chars().any(|c| c as u32 >= 0x0100),
            "a label is words: {label:?}"
        );
    }
    // the prefix comes back normalised: a word model's round trip costs its whitespace
    let spaced = model
        .predict(
            "the   cat\tsat on",
            &PredictOptions {
                length: 2,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(spaced.best.full_text.starts_with("the cat sat on "));
}

#[test]
fn a_generated_text_is_whole_and_spaced() {
    let mut model = trained(CORPUS, 3);
    let results = model
        .generate(&GenerateOptions {
            count: 2,
            mode: "beam".to_string(),
            max_length: 12,
            ..Default::default()
        })
        .unwrap();
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(r.text, r.full_text);
        assert!(!r.text.contains("  "), "{:?}", r.text);
        assert_eq!(r.text, model.normalise(&r.text));
    }
}

#[test]
fn an_unread_word_is_unknown_and_scores_are_per_word() {
    let mut model = trained(CORPUS, 3);
    let known = model.score("the cat sat on the mat");
    let unknown = model.score("the cat sat on the flurb");
    assert_eq!(known.chars, 6, "chars counts words");
    assert_eq!(known.unknown_transitions, 0);
    assert!(unknown.unknown_transitions > 0);
    assert!(unknown.log_prob < known.log_prob);
    // two different unread words are the same symbol
    let a = model.score("the qux sat");
    let b = model.score("the quux sat");
    assert_eq!(a.log_prob, b.log_prob);
    assert_eq!(
        model.g.vocab.as_ref().unwrap().id("qux"),
        0,
        "predicting never grows it"
    );
}

#[test]
fn the_alphabet_is_read_back_in_order() {
    let mut model = trained(CORPUS, 2);
    let words: Vec<String> = model.g.vocab.as_ref().unwrap().words().to_vec();
    let doc = model.to_doc();
    assert_eq!(doc.at("format").as_str(), Some("radixnet-word"));
    assert_eq!(doc.at("graph").at("units").as_str(), Some(WORD_UNITS));
    assert_eq!(doc.at("graph").at("vocabulary").to_strings(), words);
    let mut again = Model::from_doc(&doc).unwrap();
    assert_eq!(again.g.vocab.as_ref().unwrap().words(), words.as_slice());
    assert_eq!(
        again
            .predict("the cat sat on", &Default::default())
            .unwrap()
            .best
            .full_text,
        model
            .predict("the cat sat on", &Default::default())
            .unwrap()
            .best
            .full_text
    );
    assert_eq!(
        again.score("the dog sat on the mat").log_prob,
        model.score("the dog sat on the mat").log_prob
    );
}

#[test]
fn a_count_reader_refuses_a_word_file() {
    let mut model = trained(CORPUS, 1);
    let mut doc = model.to_doc();
    // a count document with a word graph is refused, and the other way round
    if let radixnet::Json::Obj(pairs) = &mut doc {
        for (key, value) in pairs.iter_mut() {
            if key == "format" {
                *value = radixnet::Json::str("radixnet-count");
            }
        }
    }
    assert!(Model::from_doc(&doc).is_err());

    let mut plain = Model::new(0, GraphOptions::default()).unwrap();
    plain
        .train(
            &texts(CORPUS),
            &TrainOptions {
                epochs: 1,
                ..Default::default()
            },
        )
        .unwrap();
    let mut doc = plain.to_doc();
    if let radixnet::Json::Obj(pairs) = &mut doc {
        for (key, value) in pairs.iter_mut() {
            if key == "format" {
                *value = radixnet::Json::str("radixnet-word");
            }
        }
    }
    assert!(Model::from_doc(&doc).is_err());
}

#[test]
fn the_most_read_words_are_what_the_graph_holds() {
    let mut model = trained(CORPUS, 2);
    let rows: Vec<WordRow> = model.top_words(3);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].word, "the");
    assert!(rows.iter().all(|r| r.trigrams > 0));
    assert_eq!(model.meta.trained_chars.value, 32, "trained_chars counts words");
}
