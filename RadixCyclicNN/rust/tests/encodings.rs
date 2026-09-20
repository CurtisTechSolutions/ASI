//! Every encoding, end to end: the graph, the model, and the walk back out.
//!
//! Two dials compose here, and the point of these tests is that they do:
//!
//! * `GraphOptions::words` says what one **symbol** is - a character, or a
//!   word given a code point of its own (`SPEC-WordNGrams.md`);
//! * `Encoding { n, stride }` says how many symbols a **gram** holds and how
//!   far apart consecutive grams start.
//!
//! So a word bigram is `words: true` with `n: 2`, groups of five letters are
//! `words: false` with `n: 5, stride: 5`, and neither dial has to know about
//! the other.

use radixnet::model::PredictOptions;
use radixnet::{Encoding, GraphOptions, Model, TrainOptions, Traversal};

/// `(words, encoding)`: every combination the tests drive end to end.
fn every_encoding() -> Vec<(bool, Encoding)> {
    let chars = |n: usize, stride: usize| (false, Encoding { n, stride });
    let words = |n: usize, stride: usize| (true, Encoding { n, stride });
    vec![
        chars(3, 1), // the trigram: the default
        chars(1, 1),
        chars(2, 1),
        chars(5, 1), // any n, sliding
        chars(4, 4),
        chars(5, 5), // groups of letters
        chars(6, 3), // groups that half-overlap
        words(1, 1),
        words(2, 1),
        words(3, 1),
        words(2, 2), // word groups
    ]
}

const CORPUS: &[&str] = &[
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "a quick brown fox jumps over the lazy dog",
];

fn texts() -> Vec<String> {
    CORPUS.iter().map(|s| s.to_string()).collect()
}

fn trained(words: bool, enc: Encoding, epochs: usize) -> Model {
    let mut model = Model::new(
        1,
        GraphOptions {
            encoding: enc,
            words,
            ..Default::default()
        },
    )
    .unwrap();
    model.workers = 1;
    model.g.workers = 1;
    model
        .train(
            &texts(),
            &TrainOptions {
                epochs,
                ..Default::default()
            },
        )
        .unwrap();
    model
}

/// The symbols a model walks a text as: its characters, or its words.
fn symbols(model: &mut Model, text: &str) -> String {
    if model.is_words() {
        model.symbols(text, false)
    } else {
        text.to_string()
    }
}

#[test]
fn the_defaults_and_what_does_not_validate() {
    assert!(Encoding::default().is_default());
    assert_eq!(Encoding::default().overlap(), radixnet::OVERLAP);
    assert_eq!(Encoding { n: 4, stride: 4 }.overlap(), 0);
    assert!(!Encoding { n: 4, stride: 4 }.sliding());
    for bad in [
        Encoding { n: 0, stride: 1 },
        Encoding { n: 3, stride: 0 },
        Encoding { n: 2, stride: 3 },
    ] {
        assert!(bad.validate().is_err(), "{bad} must not validate");
        assert!(
            Model::new(
                1,
                GraphOptions {
                    encoding: bad,
                    ..Default::default()
                }
            )
            .is_err(),
            "{bad} must not build a model"
        );
    }
    for (_, enc) in every_encoding() {
        assert!(enc.validate().is_ok(), "{enc}");
    }
}

#[test]
fn a_spec_parses_into_the_two_dials_and_prints_back() {
    let cases = [
        ("", false, "char:3:1"),
        ("trigram", false, "char:3:1"),
        ("char:4", false, "char:4:1"),
        ("chars:4:4", false, "char:4:4"),
        ("char:5:groups", false, "char:5:5"),
        ("letters:7:2", false, "char:7:2"),
        ("word", true, "word:1:1"),
        ("word:2", true, "word:2:1"),
        ("word-trigram", true, "word:3:1"),
        ("WORD:2:2", true, "word:2:2"),
    ];
    for (spec, want_words, want) in cases {
        let (words, enc) = Encoding::parse_spec(spec).unwrap_or_else(|e| panic!("{spec}: {e}"));
        assert_eq!(words, want_words, "{spec}: what a symbol is");
        assert_eq!(enc.spec(words), want, "{spec}");
        // and the printed spec parses back to the same pair
        assert_eq!(
            Encoding::parse_spec(&enc.spec(words)).unwrap(),
            (words, enc),
            "{spec} round trip"
        );
    }
    for bad in ["rune:3", "char:x", "char:3:y", "char:2:3", "char:0", "a:b:c:d"] {
        assert!(Encoding::parse_spec(bad).is_err(), "{bad} must not parse");
    }
}

#[test]
fn a_text_encodes_into_the_grams_it_should() {
    let cases: Vec<(Encoding, &str, Vec<&str>)> = vec![
        (Encoding { n: 3, stride: 1 }, "hello", vec!["hel", "ell", "llo"]),
        (Encoding { n: 1, stride: 1 }, "abc", vec!["a", "b", "c"]),
        (Encoding { n: 4, stride: 4 }, "abcdefghij", vec!["abcd", "efgh"]),
        (Encoding { n: 5, stride: 5 }, "abcdefghij", vec!["abcde", "fghij"]),
        (Encoding { n: 4, stride: 2 }, "abcdef", vec!["abcd", "cdef"]),
        (Encoding { n: 3, stride: 1 }, "hi", vec![]),
    ];
    for (enc, text, want) in cases {
        let got: Vec<String> = enc.encode(text).iter().map(|g| g.to_string()).collect();
        assert_eq!(got, want, "{enc}.encode({text:?})");
    }
}

#[test]
fn a_word_model_encodes_whole_words() {
    let mut model = trained(true, Encoding { n: 2, stride: 1 }, 1);
    // the symbols are code points, one per word, so the grams are word pairs
    let text = "the cat sat";
    let syms = symbols(&mut model, text);
    assert_eq!(syms.chars().count(), 3, "three words, three symbols");
    let grams = model.g.enc.encode(&syms);
    assert_eq!(grams.len(), 2, "three words make two bigrams");
    assert_eq!(model.words(&grams[0].to_string()), "the cat");
    assert_eq!(model.words(&grams[1].to_string()), "cat sat");
}

#[test]
fn normalize_is_what_comes_back() {
    for (enc, text, want) in [
        (Encoding { n: 3, stride: 1 }, "hello", "hello"),
        (Encoding { n: 4, stride: 4 }, "abcdefghij", "abcdefgh"),
        (Encoding { n: 4, stride: 4 }, "abc", ""),
    ] {
        assert_eq!(enc.normalize(text), want, "{enc}.normalize({text:?})");
        let grams = enc.encode(text);
        if !grams.is_empty() {
            assert_eq!(enc.decode_grams(&grams), want, "{enc} decode_grams");
        }
    }
}

#[test]
fn the_graph_round_trips_under_every_encoding() {
    for (words, enc) in every_encoding() {
        let mut model = trained(words, enc, 2);
        let walked: Vec<String> = texts().iter().map(|t| symbols(&mut model, t)).collect();
        model
            .g
            .check_invariants(&walked, false)
            .unwrap_or_else(|e| panic!("{enc} words={words}: invariants: {e}"));
        model.g.compress();
        model
            .g
            .check_invariants(&walked, true)
            .unwrap_or_else(|e| panic!("{enc} words={words}: invariants after compression: {e}"));
        for text in &walked {
            let grams = enc.encode(text);
            if grams.is_empty() {
                continue;
            }
            let path = model
                .g
                .node_path(&grams)
                .unwrap_or_else(|| panic!("{enc} words={words}: {text:?} does not walk"));
            let labels: Vec<&str> = path[1..path.len() - 1].iter().map(|&id| model.g.label(id)).collect();
            assert_eq!(
                enc.decode_path(&labels, 0, true),
                enc.normalize(text),
                "{enc} words={words}"
            );
        }
    }
}

#[test]
fn training_predicting_and_scoring_work_under_every_encoding() {
    for (words, enc) in every_encoding() {
        let mut model = trained(words, enc, 2);
        assert!(model.g.num_trigrams() > 0, "{enc} words={words}: nothing was learned");
        assert_eq!(model.g.enc, enc);
        assert_eq!(model.is_words(), words);
        let found = model
            .predict(
                "the cat sat",
                &PredictOptions {
                    length: 3,
                    k: 2,
                    traversal: Traversal::Reward,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{enc} words={words}: predict: {e}"));
        assert!(!found.top.is_empty(), "{enc} words={words}: no continuation");
        let known = model.score(CORPUS[0]);
        let noise = model.score("qzx wqzj vbn qzx wqzj vbn");
        assert!(
            known.unknown_transitions <= noise.unknown_transitions,
            "{enc} words={words}: a trained text is stranger than noise"
        );
    }
}

#[test]
fn a_word_model_walks_in_whole_words() {
    let mut model = trained(true, Encoding { n: 2, stride: 1 }, 3);
    let corpus = CORPUS.join(" ");
    let found = model
        .predict(
            "the cat sat",
            &PredictOptions {
                length: 3,
                k: 3,
                traversal: Traversal::Reward,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!found.top.is_empty());
    for result in &found.top {
        for word in result.text.split_whitespace() {
            assert!(corpus.contains(word), "predicted {word:?}, not a word of the corpus");
        }
    }
    assert!(
        found.best.full_text.starts_with("the cat sat"),
        "{:?}",
        found.best.full_text
    );
}

#[test]
fn a_group_encoding_has_no_overlap() {
    let enc = Encoding { n: 4, stride: 4 };
    assert_eq!(enc.overlap(), 0);
    let lines = vec!["abcdefghijkl".to_string(), "abcdmnopijkl".to_string()];
    let mut model = Model::new(
        1,
        GraphOptions {
            encoding: enc,
            ..Default::default()
        },
    )
    .unwrap();
    model
        .train(
            &lines,
            &TrainOptions {
                epochs: 2,
                ..Default::default()
            },
        )
        .unwrap();
    model.g.check_invariants(&lines, false).unwrap();
    for gram in model.g.trigrams() {
        assert_eq!(gram.to_string().chars().count(), 4, "{gram} is not four characters");
    }
    model.g.compress();
    model.g.check_invariants(&lines, true).unwrap();
}

/// The three implementations build the *same* graph from the same corpus, in
/// every encoding: the sample corpus, seed 1, two epochs.
#[test]
fn the_structure_is_the_one_the_other_two_build() {
    let path = std::path::Path::new("../data/sample_corpus.txt");
    let Ok(data) = std::fs::read_to_string(path) else {
        eprintln!("sample corpus not found; skipped");
        return;
    };
    let corpus: Vec<String> = data
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    // (spec, nodes, edges, grams) - as Python and Go report them
    let expected = [
        ("char:3:1", 469, 824, 715),
        ("char:5:1", 274, 432, 1144),
        ("char:4:4", 153, 262, 323),
        ("char:5:5", 111, 196, 274),
        ("char:6:3", 129, 223, 467),
        ("word:1:1", 147, 274, 203),
        ("word:2:1", 112, 199, 261),
        ("word:3:1", 71, 131, 239),
        ("word:2:2", 77, 141, 155),
    ];
    for (spec, nodes, edges, grams) in expected {
        let (words, enc) = Encoding::parse_spec(spec).unwrap();
        let mut model = Model::new(
            1,
            GraphOptions {
                encoding: enc,
                words,
                ..Default::default()
            },
        )
        .unwrap();
        model.workers = 1;
        model.g.workers = 1;
        model
            .train(
                &corpus,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            (model.g.num_nodes(), model.g.num_edges(), model.g.num_trigrams()),
            (nodes, edges, grams),
            "{spec}: this port disagrees with Python and Go about the structure"
        );
    }
}
