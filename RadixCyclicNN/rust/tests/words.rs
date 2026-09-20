//! The word view of a word encoding: the alphabet the graph has read, and what
//! every number of the model is counted in (`../../SPEC-WordNGrams.md`).
//!
//! A word encoding has no vocabulary object - a gram is text - so the alphabet
//! is derived from the gram index, which is the one count compression cannot
//! change.  The Rust twin of `go/radixnet/words_test.go`.

use radixnet::model::{GenerateOptions, PredictOptions};
use radixnet::words::{word_rows, WordRow, CHAR_UNITS, WORD_UNITS};
use radixnet::{Encoding, GraphOptions, Model, TrainOptions, Unit};

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

fn word_encoding(n: usize) -> Encoding {
    Encoding {
        unit: Unit::Words,
        n,
        stride: 1,
    }
}

fn trained(lines: &[&str], epochs: usize) -> Model {
    let mut model = Model::new(
        0,
        GraphOptions {
            encoding: word_encoding(3),
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
    // one kind and one format whatever the encoding: what makes this a word
    // model is its dial, not a class of its own
    assert_eq!(model.kind(), "count");
    assert_eq!(model.format(), "radixnet-count");
    assert_eq!(model.units(), WORD_UNITS);
    // the invariants are the character model's, over the units the graph holds
    model.g.check_invariants(&texts(CORPUS), true).expect("invariants");
    assert!(model.g.compression_ratio() > 1.0);
}

#[test]
fn a_repeated_phrase_becomes_one_node() {
    let model = trained(&["the cat sat on the mat", "a dog sat on the mat"], 1);
    let labels: Vec<String> = (0..model.g.num_node_ids())
        .filter(|&n| model.g.is_alive(n))
        .map(|n| model.g.label(n).to_string())
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
    // the prefix is joined to the continuation the way the encoding joins
    // units - one space between words - and is otherwise handed back as it was
    // given, which is what the Python implementation does
    let spaced = model
        .predict(
            "the   cat\tsat on",
            &PredictOptions {
                length: 2,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(spaced.best.full_text, format!("the   cat\tsat on {}", spaced.best.text));
    assert_eq!(spaced.best.text, found.best.text, "the words located the same path");
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
    // a word the graph has never read has no gram in the index, so it is
    // charged what any unknown transition is charged - and reading it never
    // grows anything, because there is nothing to grow
    let before = model.g.num_trigrams();
    let a = model.score("the qux sat");
    assert_eq!(model.g.num_trigrams(), before, "scoring grew the graph");
    assert!(a.unknown_transitions > 0);
}

#[test]
fn the_encoding_is_read_back_with_the_file() {
    let mut model = trained(CORPUS, 2);
    let doc = model.to_doc();
    assert_eq!(doc.at("format").as_str(), Some("radixnet-count"));
    let block = doc.at("graph").at("encoding");
    assert_eq!(block.at("unit").as_str(), Some("word"));
    assert_eq!(block.at("n").as_i64(), Some(3));
    assert_eq!(block.at("stride").as_i64(), Some(1));
    let mut again = Model::from_doc(&doc).unwrap();
    assert_eq!(again.g.enc, word_encoding(3));
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
fn an_ordinary_file_says_nothing_about_the_encoding() {
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
    let doc = plain.to_doc();
    assert_eq!(doc.at("format").as_str(), Some("radixnet-count"));
    // the character trigram of stride 1 is what a file leaves unwritten, so an
    // ordinary document is byte for byte what it always was
    assert!(doc.at("graph").get("encoding").is_none());
    assert!(doc.at("graph").get("vocabulary").is_none());
}

#[test]
fn units_name_follows_the_encoding() {
    assert_eq!(Encoding::default().units_name(), CHAR_UNITS);
    assert_eq!(word_encoding(3).units_name(), WORD_UNITS);
    // and a model says so in its stats, because a per-word number read as
    // per-character is read wrong
    let model = trained(CORPUS, 1);
    let stats = radixnet::report::stats(&model);
    assert_eq!(stats.at("units").as_str(), Some(WORD_UNITS));
    let plain = Model::new(0, GraphOptions::default()).unwrap();
    assert_eq!(radixnet::report::stats(&plain).at("units").as_str(), Some(CHAR_UNITS));
}

#[test]
fn the_vocabulary_is_what_the_grams_are_made_of() {
    let enc = word_encoding(3);
    let counts = enc.vocabulary(["the cat sat", "cat sat on"]);
    for (word, want) in [("the", 1), ("cat", 2), ("sat", 2), ("on", 1)] {
        assert_eq!(counts.get(word).copied(), Some(want), "{word:?} in {counts:?}");
    }
    assert_eq!(counts.len(), 4);
    // a character encoding counts characters, which is what its units are
    assert_eq!(Encoding::default().vocabulary(["the"]).len(), 3);
}

#[test]
fn the_most_read_words_are_what_the_graph_holds() {
    let mut model = trained(CORPUS, 2);
    let rows: Vec<WordRow> = model.top_words(3);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].word, "the");
    // the id is the rank, because the graph has no vocabulary to number from
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.id, i);
        assert!(row.grams > 0);
        assert_eq!(row.trigrams, row.grams, "the two counts disagree: {row:?}");
    }
    for pair in rows.windows(2) {
        assert!(pair[1].grams <= pair[0].grams, "not most read first: {rows:?}");
    }
    // limit 0 is the whole alphabet, and every word of the corpus is in it
    let all = model.top_words(0);
    for word in "the cat sat on mat rug dog ate bone in garden a bird sang".split(' ') {
        assert!(
            all.iter().any(|r| r.word == word),
            "{word:?} was read but is not listed"
        );
    }
    // a character model has no alphabet of words
    let plain = Model::new(0, GraphOptions::default()).unwrap();
    assert!(word_rows(&plain.g.enc, plain.g.gram_index().iter().map(String::as_str)).is_empty());
    assert_eq!(model.meta.trained_chars.value, 32, "trained_chars counts words");
}
