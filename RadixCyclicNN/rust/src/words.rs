//! The word view of a word encoding: what alphabet the graph has actually read.
//!
//! Under `--encoding word:n:s` a gram is n words rather than n characters
//! ([`crate::encoding`]), and a label is text like any other.  There is no
//! vocabulary object, and that is the point of the design: a gram is text, so
//! the alphabet the graph knows is whatever its grams are made of.  It grows as
//! training reads new words and there is nothing to freeze, prune or learn.
//!
//! What is here is the *reporting* side of that - `radixnet words`,
//! `GET /api/words` and the frontend's Words tab.  The counting itself is
//! [`crate::encoding::vocabulary`], so the three implementations list one
//! alphabet in one order.

use crate::json::Json;

/// One word of the alphabet: how many grams hold it, and its rank in the listing.
///
/// There is no id to give a word - nothing records the order the words were
/// first read - so `id` is the row's rank, which is what a reader can check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordRow {
    pub word: String,
    pub id: usize,
    /// How many of the graph's grams hold this word: what the graph actually
    /// knows about it, and the one count that survives compression.
    pub grams: usize,
}

impl WordRow {
    /// The row as the API and the CLI print it.
    ///
    /// `trigrams` is `grams` under the name a word-model client knew it by, so
    /// a reader written against the old word kind still finds its number.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("word", Json::str(self.word.clone())),
            ("id", Json::Int(self.id as i64)),
            ("grams", Json::Int(self.grams as i64)),
            ("trigrams", Json::Int(self.grams as i64)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{Encoding, Unit};
    use crate::{GraphOptions, Model, TrainOptions};

    fn word_model(n: usize, texts: &[&str]) -> Model {
        let mut model = Model::new(
            0,
            GraphOptions {
                encoding: Encoding::new(Unit::Words, n, 1).unwrap(),
                ..Default::default()
            },
        )
        .expect("a model");
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .expect("a trained model");
        model
    }

    #[test]
    fn the_alphabet_is_most_read_first() {
        let mut model = word_model(
            3,
            &[
                "the cat sat on the mat",
                "the cat sat on the log",
                "the dog sat on the mat",
            ],
        );
        let rows = model.top_words(4);
        assert_eq!(rows.len(), 4);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.id, i, "the id is the rank");
        }
        for pair in rows.windows(2) {
            assert!(pair[0].grams >= pair[1].grams, "not most read first: {rows:?}");
        }
        // every word of the corpus is in the whole listing
        let all = model.top_words(0);
        for word in ["the", "cat", "sat", "on", "mat", "log", "dog"] {
            assert!(all.iter().any(|r| r.word == word), "{word:?} is missing from {all:?}");
        }
    }

    #[test]
    fn a_character_model_has_no_words() {
        let mut model = Model::new(0, GraphOptions::default()).unwrap();
        model
            .train(&["the cat sat on the mat".to_string()], &TrainOptions::default())
            .unwrap();
        assert!(model.top_words(5).is_empty());
    }

    #[test]
    fn a_row_carries_its_count_under_both_names() {
        let row = WordRow {
            word: "the".into(),
            id: 0,
            grams: 14,
        };
        let doc = row.to_json();
        assert_eq!(doc.at("word").as_str(), Some("the"));
        assert_eq!(doc.at("grams").as_i64(), Some(14));
        assert_eq!(doc.at("trigrams").as_i64(), Some(14));
    }
}
