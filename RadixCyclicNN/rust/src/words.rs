//! The word alphabet: the same model over symbols that are words.
//!
//! `../../SPEC-WordNGrams.md`: not one of the graph's structural rules mentions
//! a *character*.  A label is a sequence of symbols, an edge exists where two
//! labels overlap by [`OVERLAP`](crate::encoding::OVERLAP) symbols, the index
//! maps a [`WINDOW`](crate::encoding::WINDOW)-symbol key to `(node, offset)`.
//! A word n-gram model is therefore this model over an alphabet whose symbols
//! are words, and a symbol is one code point:
//!
//! ```text
//! id 0          -> U+0100          the unknown word
//! id i (i >= 1) -> U+0100 + i      skipping the surrogate block D800..DFFF
//! ```
//!
//! Starting past Latin-1 keeps every label out of the range text, sentinels and
//! punctuation live in; skipping the surrogates keeps every label a string all
//! three implementations can hold and write.  A [`Trigram`](crate::Trigram)
//! packs three code points of 21 bits, which is every one of them, so the
//! representation this port trains on needs no change at all.

use std::collections::HashMap;

/// The code point of word 0.
pub const WORD_BASE: u32 = 0x0100;
/// The surrogate block, which no word may land in.
pub const SURROGATE_LO: u32 = 0xD800;
pub const SURROGATE_HI: u32 = 0xDFFF;
const SURROGATES: u32 = SURROGATE_HI - SURROGATE_LO + 1;
/// 1 111 808: every code point above Latin-1 that is not a surrogate.
pub const MAX_WORDS: usize = (0x110000 - WORD_BASE - SURROGATES) as usize;
/// Word 0: a word the model has never read maps to it at prediction and
/// scoring time, and two different unread words are the same symbol.
pub const UNKNOWN_WORD: &str = "<unk>";
/// That word's id.
pub const UNKNOWN_ID: usize = 0;
/// What a word model counts in, in its file and in every report.
pub const WORD_UNITS: &str = "words";
/// What every other kind counts in.
pub const CHAR_UNITS: &str = "chars";

/// The code point that carries a word id.
pub fn word_symbol(id: usize) -> Result<char, String> {
    if id >= MAX_WORDS {
        return Err(format!("word id must lie in [0, {MAX_WORDS}), got {id}"));
    }
    let mut code = WORD_BASE + id as u32;
    if code >= SURROGATE_LO {
        code += SURROGATES;
    }
    char::from_u32(code).ok_or_else(|| format!("U+{code:04X} is not a character"))
}

/// The inverse of [`word_symbol`]; an error for a code point that carries no word.
pub fn symbol_word(symbol: char) -> Result<usize, String> {
    let code = symbol as u32;
    if code < WORD_BASE || (SURROGATE_LO..=SURROGATE_HI).contains(&code) {
        return Err(format!("U+{code:04X} is not a word symbol"));
    }
    let id = if code > SURROGATE_HI {
        code - WORD_BASE - SURROGATES
    } else {
        code - WORD_BASE
    };
    Ok(id as usize)
}

/// The whole tokeniser: maximal runs of code points the Unicode `White_Space`
/// property does not cover.  Punctuation stays attached to the word it touches
/// and case is kept (`"mat."` and `"mat"` are two words, `"The"` and `"the"`
/// are two words), because every refinement of that rule is a step towards a
/// vocabulary that has to be designed, versioned and defended.
pub fn split_words(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

/// A word model's alphabet: a two-way map between words and single code points.
///
/// Id 0 is always [`UNKNOWN_WORD`]; every other word is in it because training
/// read it, in the order it was first read.  Nothing is frozen, pruned or
/// learned - there is no tokeniser here and no training run before the training
/// run.
#[derive(Clone, Debug)]
pub struct Vocabulary {
    words: Vec<String>,
    ids: HashMap<String, usize>,
}

impl Default for Vocabulary {
    fn default() -> Vocabulary {
        Vocabulary::new()
    }
}

impl Vocabulary {
    /// The unknown word and nothing else.
    pub fn new() -> Vocabulary {
        let mut ids = HashMap::new();
        ids.insert(UNKNOWN_WORD.to_string(), UNKNOWN_ID);
        Vocabulary {
            words: vec![UNKNOWN_WORD.to_string()],
            ids,
        }
    }

    /// Rebuilds a vocabulary from a file's list, which is in id order.
    pub fn from_list(words: &[String]) -> Result<Vocabulary, String> {
        let mut v = Vocabulary::new();
        if words.is_empty() {
            return Ok(v);
        }
        if words[0] != UNKNOWN_WORD {
            return Err(format!("a vocabulary starts with {UNKNOWN_WORD:?}, got {:?}", words[0]));
        }
        for word in &words[1..] {
            if v.ids.contains_key(word) {
                return Err(format!("the vocabulary holds {word:?} twice"));
            }
            v.add(word)?;
        }
        Ok(v)
    }

    /// How many words it holds, the unknown word included.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Never: a vocabulary always holds the unknown word.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The words in id order, as the model file carries them.
    pub fn words(&self) -> &[String] {
        &self.words
    }

    /// The id of a word, or [`UNKNOWN_ID`] when it has never been read.
    pub fn id(&self, word: &str) -> usize {
        self.ids.get(word).copied().unwrap_or(UNKNOWN_ID)
    }

    /// The id of a word, giving it the next one if it is new.
    pub fn add(&mut self, word: &str) -> Result<usize, String> {
        if let Some(&id) = self.ids.get(word) {
            return Ok(id);
        }
        if self.words.len() >= MAX_WORDS {
            return Err(format!("a word model holds at most {MAX_WORDS} words"));
        }
        let id = self.words.len();
        self.words.push(word.to_string());
        self.ids.insert(word.to_string(), id);
        Ok(id)
    }

    /// `text` as the graph's symbols, one code point per word.  `grow` gives an
    /// unread word the next id (training); without it an unread word is
    /// [`UNKNOWN_WORD`], whose windows are not in the index, so the search and
    /// the score charge it what they charge any unknown transition.
    pub fn encode(&mut self, text: &str, grow: bool) -> String {
        let mut out = String::new();
        for word in split_words(text) {
            let id = if grow {
                // the cap is 1 111 808 words; past it everything is the unknown word
                self.add(word).unwrap_or(UNKNOWN_ID)
            } else {
                self.id(word)
            };
            out.push(word_symbol(id).unwrap_or('\u{0100}'));
        }
        out
    }

    /// [`Vocabulary::encode`] without growing, so it needs no `&mut`.
    pub fn encode_known(&self, text: &str) -> String {
        let mut out = String::new();
        for word in split_words(text) {
            out.push(word_symbol(self.id(word)).unwrap_or('\u{0100}'));
        }
        out
    }

    /// The words a symbol string stands for, joined by single spaces.  A symbol
    /// this vocabulary has no word for decodes to [`UNKNOWN_WORD`]: a file is
    /// checked when it is loaded, so past that point this cannot happen.
    pub fn decode(&self, symbols: &str) -> String {
        let mut out = String::new();
        for symbol in symbols.chars() {
            if !out.is_empty() {
                out.push(' ');
            }
            match symbol_word(symbol) {
                Ok(id) if id < self.words.len() => out.push_str(&self.words[id]),
                _ => out.push_str(UNKNOWN_WORD),
            }
        }
        out
    }
}

/// One word of the alphabet with how much of the graph holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordRow {
    pub word: String,
    pub id: usize,
    pub trigrams: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_id_round_trips_and_the_surrogates_are_skipped() {
        for id in [0usize, 1, 2, 1000, 55_039, 55_040, 55_041, MAX_WORDS - 1] {
            assert_eq!(symbol_word(word_symbol(id).unwrap()).unwrap(), id);
        }
        assert_eq!(word_symbol(0).unwrap(), '\u{0100}');
        assert_eq!(word_symbol(MAX_WORDS - 1).unwrap() as u32, 0x10FFFF);
        assert!(word_symbol(MAX_WORDS).is_err());
        assert_eq!(MAX_WORDS, 1_111_808);
        let below = word_symbol((SURROGATE_LO - WORD_BASE - 1) as usize).unwrap() as u32;
        let above = word_symbol((SURROGATE_LO - WORD_BASE) as usize).unwrap() as u32;
        assert_eq!((below, above), (SURROGATE_LO - 1, SURROGATE_HI + 1));
        for ch in ['a', ' ', '<', '\u{00ff}'] {
            assert!(symbol_word(ch).is_err());
        }
    }

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
    fn a_vocabulary_grows_in_reading_order() {
        let mut v = Vocabulary::new();
        let symbols = v.encode("the cat sat on the mat", true);
        assert_eq!(v.len(), 6);
        assert_eq!(v.decode(&symbols), "the cat sat on the mat");
        assert_eq!(v.decode(&v.encode_known("the qux sat")), "the <unk> sat");
        assert_eq!(v.len(), 6);
        let back = Vocabulary::from_list(v.words()).unwrap();
        assert_eq!(back.words(), v.words());
        assert!(Vocabulary::from_list(&["the".to_string()]).is_err());
        assert!(Vocabulary::from_list(&[UNKNOWN_WORD.to_string(), "a".into(), "a".into()]).is_err());
    }
}
