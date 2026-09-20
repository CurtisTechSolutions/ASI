//! The word view of a word encoding: what alphabet the graph has actually read.
//!
//! Under `--encoding word:n:s` a gram is n words rather than n characters
//! ([`crate::encoding`]), and a label is text like any other.  There is no
//! vocabulary object to consult, and that is the point of the design
//! (`../../SPEC-WordNGrams.md`): a gram is text, so the alphabet the graph
//! knows is whatever its grams are made of.  It grows as training reads new
//! words and there is nothing to freeze, prune or learn.
//!
//! What is here is the *reporting* side of that - `radixnet words`,
//! `GET /api/words` and the frontend's Words tab - plus the one thing every
//! number of a model needs beside it: what it is counted in.
//!
//! The Rust twin of Python's `Encoding.units_name` / `word_rows` and Go's
//! `words.go`.

use crate::encoding::{Encoding, Unit};

/// What a word encoding counts in, in its file and in every report.
pub const WORD_UNITS: &str = "words";
/// What every other encoding counts in.
pub const CHAR_UNITS: &str = "chars";

/// The whole tokeniser: maximal runs of code points the Unicode `White_Space`
/// property does not cover.  Punctuation stays attached to the word it touches
/// and case is kept (`"mat."` and `"mat"` are two words, `"The"` and `"the"`
/// are two words), because every refinement of that rule is a step towards a
/// vocabulary that has to be designed, versioned and defended.
pub fn split_words(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

/// One word of the alphabet: how many grams hold it, and its rank in the
/// listing.
///
/// There is no id to give a word - the graph has no vocabulary, so nothing
/// records the order the words were first read - so `id` is the row's rank,
/// which is what a reader can actually check.  `trigrams` is `grams` under the
/// name a word-model client knew it by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordRow {
    pub word: String,
    pub id: usize,
    pub grams: usize,
    pub trigrams: usize,
}

/// The alphabet as the API and the CLI list it: most read first, ties
/// alphabetically.  A model that does not count in words has none.
///
/// The order is the Python implementation's, word for word: three ports asked
/// for the same alphabet hand back the same rows in the same order.
pub fn word_rows<'a, I: IntoIterator<Item = &'a str>>(enc: &Encoding, grams: I) -> Vec<WordRow> {
    if enc.unit != Unit::Words {
        return Vec::new();
    }
    let counts = enc.vocabulary(grams);
    let mut rows: Vec<WordRow> = counts
        .into_iter()
        .map(|(word, grams)| WordRow {
            word,
            id: 0,
            grams,
            trigrams: grams,
        })
        .collect();
    // a map iterates by the race, and a listing has to be one answer
    rows.sort_by(|a, b| b.grams.cmp(&a.grams).then_with(|| a.word.cmp(&b.word)));
    for (i, row) in rows.iter_mut().enumerate() {
        row.id = i;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_is_the_unicode_property() {
        let words: Vec<&str> = split_words("The mat. the mat").collect();
        assert_eq!(words, vec!["The", "mat.", "the", "mat"]);
        // Python calls U+001C..U+001F whitespace; the Unicode property does not
        assert_eq!(split_words("a\u{1c}b").collect::<Vec<_>>(), vec!["a\u{1c}b"]);
        assert_eq!(split_words("a\tb\nc\r\nd\u{b}e\u{c}f\u{a0}g\u{85}h").count(), 8);
        assert_eq!(split_words("   ").count(), 0);
    }

    #[test]
    fn the_alphabet_is_whatever_the_grams_are_made_of() {
        let enc = Encoding {
            unit: Unit::Words,
            n: 2,
            stride: 1,
        };
        let grams: Vec<String> = enc
            .encode("the cat sat on the mat")
            .iter()
            .map(|g| g.to_string())
            .collect();
        let rows = word_rows(&enc, grams.iter().map(String::as_str));
        // the bigrams are "the cat", "cat sat", "sat on", "on the", "the mat":
        // "the" is in three of them, every other word in one or two.  Most
        // read first, ties alphabetically.
        assert_eq!(rows[0].word, "the");
        assert_eq!(rows[0].id, 0);
        assert_eq!(rows[0].grams, 3);
        assert_eq!(rows[0].trigrams, rows[0].grams);
        let listed: Vec<&str> = rows.iter().map(|r| r.word.as_str()).collect();
        assert_eq!(listed, vec!["the", "cat", "on", "sat", "mat"]);
        assert!(word_rows(&Encoding::default(), grams.iter().map(String::as_str)).is_empty());
    }

    #[test]
    fn a_number_says_what_it_counts() {
        assert_eq!(Encoding::default().units_name(), CHAR_UNITS);
        assert_eq!(
            Encoding {
                unit: Unit::Words,
                n: 3,
                stride: 1
            }
            .units_name(),
            WORD_UNITS
        );
    }
}
