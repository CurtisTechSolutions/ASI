//! Every encoding, end to end: the graph, the model, and the walk back out.
//!
//! Three dials compose here, and the point of these tests is that they do:
//!
//! * `Encoding::unit` says what one **unit** of text is - a character, or a
//!   whitespace word (`SPEC-WordNGrams.md`);
//! * `n` says how many units a **gram** holds;
//! * `stride` says how far apart consecutive grams start.
//!
//! So a word bigram is `unit: Words, n: 2`, groups of five letters are
//! `unit: Chars, n: 5, stride: 5`, and the graph below them never has to know
//! which it is holding.

use radixnet::model::PredictOptions;
use radixnet::{parse_encoding, Encoding, GraphOptions, Model, TrainOptions, Unit};

/// Every combination the tests drive end to end.
fn every_encoding() -> Vec<Encoding> {
    let chars = |n: usize, stride: usize| Encoding {
        unit: Unit::Chars,
        n,
        stride,
    };
    let words = |n: usize, stride: usize| Encoding {
        unit: Unit::Words,
        n,
        stride,
    };
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

fn trained(enc: Encoding, epochs: usize) -> Model {
    let mut model = Model::new(
        1,
        GraphOptions {
            encoding: enc,
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

#[test]
fn the_defaults_and_what_does_not_validate() {
    assert!(Encoding::default().is_default());
    assert_eq!(Encoding::default().overlap(), radixnet::OVERLAP);
    assert_eq!(parse_encoding("char:4:4").unwrap().overlap(), 0);
    assert!(!parse_encoding("char:4:4").unwrap().sliding());
    for bad in [
        Encoding {
            unit: Unit::Chars,
            n: 0,
            stride: 1,
        },
        Encoding {
            unit: Unit::Chars,
            n: 3,
            stride: 0,
        },
        Encoding {
            unit: Unit::Words,
            n: 2,
            stride: 3,
        },
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
    for enc in every_encoding() {
        assert!(enc.validate().is_ok(), "{enc}");
    }
}

#[test]
fn a_spec_parses_into_the_three_dials_and_prints_back() {
    let cases = [
        ("", "char:3:1"),
        ("trigram", "char:3:1"),
        ("char:4", "char:4:1"),
        ("chars:4:4", "char:4:4"),
        ("char:5:groups", "char:5:5"),
        ("letters:7:2", "char:7:2"),
        ("word", "word:1:1"),
        ("word:2", "word:2:1"),
        ("word-trigram", "word:3:1"),
        ("WORD:2:2", "word:2:2"),
    ];
    for (spec, want) in cases {
        let enc = parse_encoding(spec).unwrap_or_else(|e| panic!("{spec}: {e}"));
        assert_eq!(enc.to_string(), want, "{spec}");
        // and the printed spec parses back to the same encoding
        assert_eq!(parse_encoding(&enc.to_string()).unwrap(), enc, "{spec} round trip");
    }
    for bad in ["rune:3", "char:x", "char:3:y", "char:2:3", "char:0", "a:b:c:d"] {
        assert!(parse_encoding(bad).is_err(), "{bad} must not parse");
    }
}

#[test]
fn a_text_encodes_into_the_grams_it_should() {
    let cases: Vec<(&str, &str, Vec<&str>)> = vec![
        ("char:3:1", "hello", vec!["hel", "ell", "llo"]),
        ("char:1:1", "abc", vec!["a", "b", "c"]),
        ("char:4:4", "abcdefghij", vec!["abcd", "efgh"]),
        ("char:5:5", "abcdefghij", vec!["abcde", "fghij"]),
        ("char:4:2", "abcdef", vec!["abcd", "cdef"]),
        ("char:3:1", "hi", vec![]),
        ("word:2:1", "the cat sat on", vec!["the cat", "cat sat", "sat on"]),
        ("word:2:2", "the cat sat on", vec!["the cat", "sat on"]),
        ("word:3:1", "the  cat\tsat", vec!["the cat sat"]),
        ("word:2:1", "alone", vec![]),
    ];
    for (spec, text, want) in cases {
        let enc = parse_encoding(spec).unwrap();
        let got: Vec<String> = enc.encode(text).iter().map(|g| g.to_string()).collect();
        assert_eq!(got, want, "{enc}.encode({text:?})");
    }
}

#[test]
fn a_word_model_encodes_whole_words() {
    let model = trained(parse_encoding("word:2").unwrap(), 1);
    // a gram is text like any other, made of two whole words
    let enc = model.g.enc;
    let text = "the cat sat";
    assert_eq!(enc.len(text), 3, "three words, three units");
    let grams = enc.encode(text);
    assert_eq!(grams.len(), 2, "three words make two bigrams");
    assert_eq!(grams[0].to_string(), "the cat");
    assert_eq!(grams[1].to_string(), "cat sat");
}

#[test]
fn normalize_is_what_comes_back() {
    for (spec, text, want) in [
        ("char:3:1", "hello", "hello"),
        ("char:4:4", "abcdefghij", "abcdefgh"),
        ("char:4:4", "abc", ""),
        // a word encoding's round trip costs the original whitespace and
        // nothing else, and drops the tail no whole gram covers
        ("word:2:1", "the  cat\tsat", "the cat sat"),
        ("word:2:2", "the cat sat", "the cat"),
    ] {
        let enc = parse_encoding(spec).unwrap();
        assert_eq!(enc.normalize(text), want, "{enc}.normalize({text:?})");
        let grams = enc.encode(text);
        if !grams.is_empty() {
            assert_eq!(
                enc.decode_grams(grams.iter().map(String::as_str)),
                want,
                "{enc} decode_grams"
            );
        }
    }
}

#[test]
fn the_graph_round_trips_under_every_encoding() {
    for enc in every_encoding() {
        let mut model = trained(enc, 2);
        let walked = texts();
        model
            .g
            .check_invariants(&walked, false)
            .unwrap_or_else(|e| panic!("{enc}: invariants: {e}"));
        model.g.compress();
        model
            .g
            .check_invariants(&walked, true)
            .unwrap_or_else(|e| panic!("{enc}: invariants after compression: {e}"));
        for text in &walked {
            let grams = enc.encode(text);
            if grams.is_empty() {
                continue;
            }
            let path = model
                .g
                .node_path(&grams)
                .unwrap_or_else(|| panic!("{enc}: {text:?} does not walk"));
            let labels: Vec<&str> = path[1..path.len() - 1].iter().map(|&id| model.g.label(id)).collect();
            assert_eq!(enc.decode_path(&labels, 0, true), enc.normalize(text), "{enc}");
        }
    }
}

#[test]
fn training_predicting_and_scoring_work_under_every_encoding() {
    for enc in every_encoding() {
        let mut model = trained(enc, 2);
        assert!(model.g.num_trigrams() > 0, "{enc}: nothing was learned");
        assert_eq!(model.g.enc, enc);
        assert_eq!(model.units(), enc.units_name());
        let found = model
            .predict(
                "the cat sat",
                &PredictOptions {
                    length: 3,
                    k: 2,
                    traversal: "reward".to_string(),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{enc}: predict: {e}"));
        assert!(!found.top.is_empty(), "{enc}: no continuation");
        let known = model.score(CORPUS[0]);
        let noise = model.score("qzx wqzj vbn qzx wqzj vbn");
        assert!(
            known.unknown_transitions <= noise.unknown_transitions,
            "{enc}: a trained text is stranger than noise"
        );
    }
}

#[test]
fn a_word_model_walks_in_whole_words() {
    let mut model = trained(parse_encoding("word:2").unwrap(), 3);
    let corpus = CORPUS.join(" ");
    let found = model
        .predict(
            "the cat sat",
            &PredictOptions {
                length: 3,
                k: 3,
                traversal: "reward".to_string(),
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
    let enc = parse_encoding("char:4:groups").unwrap();
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
        ("char:3:1", 470, 824, 715),
        ("char:5:1", 275, 432, 1144),
        ("char:4:4", 154, 262, 323),
        ("char:5:5", 112, 196, 274),
        ("char:6:3", 130, 223, 467),
        ("word:1:1", 148, 274, 203),
        ("word:2:1", 113, 199, 261),
        ("word:3:1", 72, 131, 239),
        ("word:2:2", 78, 141, 155),
    ];
    for (spec, nodes, edges, grams) in expected {
        let enc = parse_encoding(spec).unwrap();
        let mut model = Model::new(
            1,
            GraphOptions {
                encoding: enc,
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
