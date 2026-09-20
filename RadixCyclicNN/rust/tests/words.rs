//! A word encoding end to end: the same model over an alphabet of words.
//!
//! Words are not a kind - they are the encoding's `unit` dial - so nothing here
//! is a different model. What is checked is that the graph, the search, the
//! counters and the file all measure in *words* when told to, and that a word
//! model's file is one Python and Go read.

use radixnet::encoding::{Encoding, Unit};
use radixnet::{GenerateOptions, GraphOptions, Model, PredictOptions, TrainOptions};

const CORPUS: [&str; 6] = [
    "the cat sat on the mat",
    "the cat sat on the log",
    "the dog sat on the mat",
    "a bird flew over the hill",
    "the fox ran up the hill",
    "the owl flew over the barn",
];

fn word_model(n: usize) -> Model {
    let mut model = Model::new(
        0,
        GraphOptions {
            encoding: Encoding::new(Unit::Words, n, 1).expect("a word encoding"),
            ..Default::default()
        },
    )
    .expect("a model");
    let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    model
        .train(
            &texts,
            &TrainOptions {
                epochs: 3,
                ..Default::default()
            },
        )
        .expect("a trained model");
    model
}

#[test]
fn a_word_model_is_the_count_model_over_words() {
    let model = word_model(3);
    assert_eq!(model.kind(), "count", "words are an encoding, not a kind");
    assert_eq!(model.units(), "words");
    assert_eq!(model.encoding().to_string(), "word:3:1");
    // a label is text like any other; a gram of three words holds two spaces
    let labels: Vec<String> = (radixnet::FIRST..model.g.num_node_ids())
        .filter(|&n| model.g.is_alive(n))
        .map(|n| model.g.label(n).to_string())
        .collect();
    assert!(!labels.is_empty());
    for label in &labels {
        assert!(!label.contains("  "), "{label:?} carries the original spacing");
    }
    model.g.check_invariants(&[], false).expect("the invariants hold");
}

#[test]
fn a_prediction_is_words_and_lengths_are_counted_in_them() {
    let mut model = word_model(3);
    let found = model
        .predict(
            "the cat sat on",
            &PredictOptions {
                length: 2,
                k: 3,
                ..Default::default()
            },
        )
        .expect("a prediction");
    // two WORDS on, not two characters
    assert!(
        matches!(
            found.best.full_text.as_str(),
            "the cat sat on the mat" | "the cat sat on the log"
        ),
        "{:?}",
        found.best.full_text
    );
    assert_eq!(found.best.text.split_whitespace().count(), 2);
}

#[test]
fn a_generated_text_is_whole_and_spaced() {
    let mut model = word_model(3);
    let texts = model
        .generate(&GenerateOptions {
            count: 3,
            max_length: 12,
            mode: "beam".to_string(),
            ..Default::default()
        })
        .expect("generated texts");
    assert_eq!(texts.len(), 3);
    for one in &texts {
        assert!(!one.text.is_empty());
        assert!(!one.text.contains("  "), "{:?} has a double space", one.text);
        assert!(one.text.split_whitespace().count() <= 12);
    }
}

#[test]
fn an_unread_word_is_an_unknown_transition_and_scores_are_per_word() {
    let mut model = word_model(3);
    let known = model.score("the cat sat on the mat");
    assert_eq!(known.chars, 6, "a six-word text is six units long");
    assert_eq!(known.unknown_transitions, 0);
    assert!(known.log_prob < 0.0);

    let unread = model.score("qqzz never read this");
    assert!(
        unread.unknown_transitions > 0,
        "an unread word is an unknown transition"
    );
    assert!(unread.log_prob < known.log_prob);
}

#[test]
fn the_most_read_words_are_what_the_grams_hold() {
    let mut model = word_model(3);
    let rows = model.top_words(5);
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].word, "the", "the corpus says 'the' most often");
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.id, i, "the id is the rank");
    }
    for pair in rows.windows(2) {
        assert!(pair[0].grams >= pair[1].grams, "not most read first: {rows:?}");
    }
    // limit 0 is the whole alphabet
    let all = model.top_words(0);
    assert!(all.len() > rows.len());
    for word in ["the", "cat", "sat", "mat", "hill", "barn"] {
        assert!(
            all.iter().any(|r| r.word == word),
            "{word:?} was read but is not listed"
        );
    }
}

#[test]
fn the_file_carries_the_encoding_and_reads_back() {
    let dir = std::env::temp_dir().join(format!("radixnet-words-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a test directory");
    let path = dir.join("model.word.json");
    let path = path.to_str().expect("a path");

    let mut model = word_model(3);
    let before = model.predict("the cat sat on", &PredictOptions::default()).unwrap();
    model.save(path).expect("the model saves");

    let doc = radixnet::file::read_document(path).expect("the file reads");
    // a word model is still the count format: the encoding is what changed
    assert_eq!(doc.at("format").as_str(), Some(radixnet::MODEL_FORMAT));
    assert_eq!(doc.at("kind").as_str(), Some("count"));
    let encoding = doc.at("graph").at("encoding");
    assert_eq!(encoding.at("unit").as_str(), Some("word"));
    assert_eq!(encoding.at("n").as_i64(), Some(3));
    assert_eq!(encoding.at("stride").as_i64(), Some(1));

    let mut again = Model::load(path).expect("the model loads");
    assert_eq!(again.encoding().to_string(), "word:3:1");
    assert_eq!(again.units(), "words");
    let after = again.predict("the cat sat on", &PredictOptions::default()).unwrap();
    assert_eq!(before.best.full_text, after.best.full_text);
    assert_eq!(before.best.cost, after.best.cost);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_character_model_leaves_the_file_as_it_always_was() {
    let dir = std::env::temp_dir().join(format!("radixnet-chars-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a test directory");
    let path = dir.join("model.count.json");
    let path = path.to_str().expect("a path");

    let mut model = Model::new(0, GraphOptions::default()).expect("a model");
    let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    model.train(&texts, &TrainOptions::default()).expect("a trained model");
    model.save(path).expect("the model saves");

    // the default encoding is never written, so an ordinary file is byte for
    // byte what it always was - and Python, which reads that one, still can
    let doc = radixnet::file::read_document(path).expect("the file reads");
    assert!(
        doc.at("graph").get("encoding").is_none(),
        "the default encoding was written"
    );
    assert_eq!(model.units(), "chars");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_other_dials_work_too() {
    // groups of four letters: no overlap at all, so the graph is a chain
    let mut model = Model::new(
        0,
        GraphOptions {
            encoding: Encoding::new(Unit::Chars, 4, 4).expect("a grouping encoding"),
            ..Default::default()
        },
    )
    .expect("a model");
    let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    model
        .train(
            &texts,
            &TrainOptions {
                epochs: 2,
                ..Default::default()
            },
        )
        .expect("a trained model");
    assert_eq!(model.units(), "chars");
    assert_eq!(model.encoding().overlap(), 0);
    model.g.check_invariants(&[], false).expect("the invariants hold");
    // and a word bigram
    let mut bigram = word_model(2);
    assert_eq!(bigram.encoding().to_string(), "word:2:1");
    assert!(!bigram.top_words(0).is_empty());
    bigram.g.check_invariants(&[], false).expect("the invariants hold");
}
