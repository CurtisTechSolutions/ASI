//! Every encoding, end to end: the graph, the model, and the walk back out.
//!
//! The unit tests in `src/encoding.rs` cover the encoder and decoder on their
//! own; this drives the whole model with each of them, so that "n-grams of any
//! size, groups of letters and words" is a claim about the model and not only
//! about the encoder.

use radixnet::model::PredictOptions;
use radixnet::{Encoding, GraphOptions, Model, TrainOptions, Traversal, Unit};

/// The encodings the tests drive end to end.
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
    assert_eq!(
        Encoding {
            n: 4,
            stride: 4,
            ..Default::default()
        }
        .overlap(),
        0
    );
    assert!(!Encoding {
        n: 4,
        stride: 4,
        ..Default::default()
    }
    .sliding());
    for bad in [
        Encoding {
            n: 0,
            stride: 1,
            unit: Unit::Chars,
        },
        Encoding {
            n: 3,
            stride: 0,
            unit: Unit::Chars,
        },
        Encoding {
            n: 2,
            stride: 3,
            unit: Unit::Chars,
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
fn a_spec_parses_and_prints_back() {
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
        let enc = Encoding::parse(spec).unwrap_or_else(|e| panic!("{spec}: {e}"));
        assert_eq!(enc.to_string(), want, "{spec}");
        assert_eq!(Encoding::parse(&enc.to_string()).unwrap(), enc, "{spec} round trip");
    }
    for bad in ["rune:3", "char:x", "char:3:y", "char:2:3", "char:0", "a:b:c:d"] {
        assert!(Encoding::parse(bad).is_err(), "{bad} must not parse");
    }
}

#[test]
fn a_text_encodes_into_the_grams_it_should() {
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
    let cases: Vec<(Encoding, &str, Vec<&str>)> = vec![
        (chars(3, 1), "hello", vec!["hel", "ell", "llo"]),
        (chars(1, 1), "abc", vec!["a", "b", "c"]),
        (chars(4, 4), "abcdefghij", vec!["abcd", "efgh"]),
        (chars(5, 5), "abcdefghij", vec!["abcde", "fghij"]),
        (chars(4, 2), "abcdef", vec!["abcd", "cdef"]),
        (chars(3, 1), "hi", vec![]),
        (words(2, 1), "the cat sat down", vec!["the cat", "cat sat", "sat down"]),
        (words(3, 1), "the cat sat down", vec!["the cat sat", "cat sat down"]),
        (words(2, 2), "the cat sat down here", vec!["the cat", "sat down"]),
        (words(2, 1), "  the   cat  ", vec!["the cat"]),
        (words(2, 1), "alone", vec![]),
    ];
    for (enc, text, want) in cases {
        let got: Vec<String> = enc.encode(text).iter().map(|g| g.to_string()).collect();
        assert_eq!(got, want, "{enc}.encode({text:?})");
    }
}

#[test]
fn normalize_is_what_comes_back() {
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
    for (enc, text, want) in [
        (chars(3, 1), "hello", "hello"),
        (chars(4, 4), "abcdefghij", "abcdefgh"),
        (chars(4, 4), "abc", ""),
        (words(2, 1), "  the  cat   sat ", "the cat sat"),
        (words(2, 2), "the cat sat down here", "the cat sat down"),
    ] {
        assert_eq!(enc.normalize(text), want, "{enc}.normalize({text:?})");
        let grams = enc.encode(text);
        if !grams.is_empty() {
            assert_eq!(enc.decode_grams(&grams), want, "{enc} decode_grams");
        }
    }
}

#[test]
fn a_word_prefix_stops_at_a_word_boundary() {
    let word = Encoding {
        unit: Unit::Words,
        n: 2,
        stride: 1,
    };
    assert!(!word.has_unit_prefix("the cat", "the ca"));
    assert!(word.has_unit_prefix("the cat", "the"));
    assert!(word.has_unit_prefix("the cat", "the cat"));
    assert!(Encoding::default().has_unit_prefix("the cat", "the ca"));
}

#[test]
fn the_graph_round_trips_under_every_encoding() {
    for enc in every_encoding() {
        let mut model = trained(enc, 2);
        model
            .g
            .check_invariants(&texts(), false)
            .unwrap_or_else(|e| panic!("{enc}: invariants before compression: {e}"));
        model.g.compress();
        model
            .g
            .check_invariants(&texts(), true)
            .unwrap_or_else(|e| panic!("{enc}: invariants after compression: {e}"));
        for text in texts() {
            let grams = enc.encode(&text);
            if grams.is_empty() {
                continue;
            }
            let path = model
                .g
                .node_path(&grams)
                .unwrap_or_else(|| panic!("{enc}: {text:?} does not walk through the graph"));
            let labels: Vec<&str> = path[1..path.len() - 1].iter().map(|&id| model.g.label(id)).collect();
            assert_eq!(enc.decode_path(&labels, 0, true), enc.normalize(&text), "{enc}");
        }
    }
}

#[test]
fn training_predicting_and_scoring_work_under_every_encoding() {
    for enc in every_encoding() {
        let mut model = trained(enc, 2);
        assert!(model.g.num_trigrams() > 0, "{enc}: nothing was learned");
        assert_eq!(model.encoding(), enc);
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
    let enc = Encoding {
        unit: Unit::Words,
        n: 2,
        stride: 1,
    };
    let mut model = trained(enc, 3);
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
    // the prefix and the continuation are joined by the unit, not glued together
    assert!(
        found.best.full_text.starts_with("the cat sat "),
        "{:?}",
        found.best.full_text
    );
    for id in 3..model.g.num_node_ids() {
        let label = model.g.label(id);
        if !label.is_empty() {
            assert_eq!(label, label.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
}

#[test]
fn a_group_encoding_has_no_overlap() {
    let enc = Encoding {
        unit: Unit::Chars,
        n: 4,
        stride: 4,
    };
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
    for (gram, _) in model.g.index_entries() {
        assert_eq!(gram.to_string().chars().count(), 4, "{gram} is not four characters");
    }
    model.g.compress();
    model.g.check_invariants(&lines, true).unwrap();
}

/// The three implementations build the *same* graph from the same corpus, in
/// every encoding.
///
/// Python and Go check that against each other by trading model files
/// (`tests/test_go_parity.py::TestGoEncodingParity`); this crate writes no
/// model file, so the numbers those two agree on are pinned here instead. The
/// sample corpus, seed 1, two epochs.
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
    // (encoding, nodes, edges, grams) - as Python and Go report them
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
        let enc = Encoding::parse(spec).unwrap();
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
        model
            .g
            .check_invariants(&corpus, false)
            .unwrap_or_else(|e| panic!("{spec}: {e}"));
    }
}
