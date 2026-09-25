//! How a text becomes the grams the graph is built from, and how it comes back.
//!
//! [`Encoding`] is three dials - what one *unit* of text is, how many units a
//! gram holds, and how far apart consecutive grams start:
//!
//! * `Encoding::default()` is the character trigram of stride 1, the one every
//!   model was born with (`"hello"` -> `["hel", "ell", "llo"]`);
//! * `n = 5` is a sliding window of five characters, and any n works;
//! * `n = 4, stride = 4` is *tokenisation*: non-overlapping groups of four
//!   letters;
//! * `unit = Words, n = 2` is the word bigram, `n = 3` the word trigram.
//!
//! Two consecutive grams share [`Encoding::overlap`] units, which is what lets
//! the graph chain them: node A ends with the units node B starts with.  With
//! `stride == n` nothing is shared and the graph is a plain chain of groups.
//!
//! This mirrors `radixnet/encoding.py` and `go/radixnet/encoding.go` exactly,
//! down to the spec strings `parse_encoding` reads and the names it accepts.
//!
//! # A gram is text
//!
//! Earlier releases of this port packed a trigram into a `u64` - three code
//! points of 21 bits - which is why its encoding pass allocated nothing.  A
//! gram of *any* n over *any* unit does not fit in an integer, so the index is
//! keyed by the gram itself, as the other two implementations key it.  What
//! that costs is measured rather than guessed: `bench/RESULTS.md`.

use std::fmt;

/// The sliding-window length of the default encoding.
pub const WINDOW: usize = 3;
/// The number of units two consecutive labels of the default encoding share.
pub const OVERLAP: usize = WINDOW - 1;

/// The labels of the three sentinel nodes.
pub const START_LABEL: &str = "<s>";
pub const END_LABEL: &str = "</s>";
/// The third sentinel: where the graph has learned a walk goes round.
pub const BACK_LABEL: &str = "<back>";
/// The fourth sentinel: where the graph has learned to stop and think, and
/// where its thoughts begin.
pub const THINK_LABEL: &str = "<think>";

/// What one position of a text is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Unit {
    /// One Unicode code point: the encoding the model was born with.
    #[default]
    Chars,
    /// One whitespace-delimited word.  The layout is not kept - a word
    /// encoding keeps the words, and a round trip writes single spaces.
    Words,
}

impl Unit {
    /// The name the spec strings and the model file use.
    pub fn name(self) -> &'static str {
        match self {
            Unit::Chars => "char",
            Unit::Words => "word",
        }
    }

    /// What a model under this unit counts in.
    ///
    /// Every length, count and score is in these, so the CLI, the API and the
    /// frontend carry it beside the number: a per-word number read as
    /// per-character is read wrong.
    pub fn units_name(self) -> &'static str {
        match self {
            Unit::Chars => "chars",
            Unit::Words => "words",
        }
    }

    /// The unit a name means, or `None`.
    pub fn parse(name: &str) -> Option<Unit> {
        match name {
            "char" | "chars" | "character" | "characters" | "letter" | "letters" => Some(Unit::Chars),
            "word" | "words" => Some(Unit::Words),
            _ => None,
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a text becomes grams, and how a walk's labels become text again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Encoding {
    pub unit: Unit,
    /// Units per gram: the n of the n-gram.  Always `>= 1`.
    pub n: usize,
    /// Units between two consecutive grams: 1 slides the window, `n` cuts the
    /// text into non-overlapping groups.  Always `1 ..= n`.
    pub stride: usize,
}

impl Default for Encoding {
    /// The character trigram with stride 1: what every model used before the
    /// encoding became a choice, and the only encoding a file leaves unwritten.
    fn default() -> Encoding {
        Encoding {
            unit: Unit::Chars,
            n: WINDOW,
            stride: 1,
        }
    }
}

impl Encoding {
    /// An encoding, checked.
    pub fn new(unit: Unit, n: usize, stride: usize) -> Result<Encoding, String> {
        let e = Encoding { unit, n, stride };
        e.validate()?;
        Ok(e)
    }

    /// What is wrong with this encoding, if anything.
    pub fn validate(&self) -> Result<(), String> {
        if self.n < 1 {
            return Err(format!("n must be >= 1, got {}", self.n));
        }
        if self.stride < 1 {
            return Err(format!("stride must be >= 1, got {}", self.stride));
        }
        if self.stride > self.n {
            return Err(format!(
                "stride {} must be <= n {}: a larger stride would skip units",
                self.stride, self.n
            ));
        }
        Ok(())
    }

    /// How many units two consecutive grams share: `n - stride`.
    pub fn overlap(&self) -> usize {
        self.n - self.stride
    }

    /// Whether this is the character trigram of stride 1 - the encoding a model
    /// file leaves unwritten.
    pub fn is_default(&self) -> bool {
        *self == Encoding::default()
    }

    /// Whether consecutive grams overlap at all.
    pub fn sliding(&self) -> bool {
        self.stride < self.n
    }

    /// What a model under this encoding counts in.
    pub fn units_name(&self) -> &'static str {
        self.unit.units_name()
    }

    /// The human form: `3-character grams, stride 1 (sliding)`.
    pub fn describe(&self) -> String {
        let unit = match self.unit {
            Unit::Chars => "character",
            Unit::Words => "word",
        };
        let kind = if self.sliding() { "sliding" } else { "groups" };
        format!("{}-{unit} grams, stride {} ({kind})", self.n, self.stride)
    }

    // -- units ---------------------------------------------------------------

    /// The text cut into units, indexed so that slicing by unit is O(1).
    pub fn units(&self, text: &str) -> Units {
        match self.unit {
            Unit::Chars => Units::chars(text),
            Unit::Words => Units::words(text),
        }
    }

    /// How many units a text holds.
    pub fn len(&self, text: &str) -> usize {
        match self.unit {
            Unit::Chars => text.chars().count(),
            Unit::Words => text.split_whitespace().count(),
        }
    }

    /// Whether a text holds no units at all.
    pub fn is_empty(&self, text: &str) -> bool {
        self.len(text) == 0
    }

    /// Units `[from, to)` of a text.
    pub fn slice(&self, text: &str, from: usize, to: usize) -> String {
        self.units(text).slice(from, to)
    }

    /// The first `n` units of a text.
    pub fn truncate(&self, text: &str, n: usize) -> String {
        self.units(text).slice(0, n)
    }

    /// Glues unit-aligned pieces: nothing between characters, one space between
    /// words.  Empty pieces are dropped, so a label contributing no unit adds
    /// no separator.
    pub fn join(&self, parts: &[&str]) -> String {
        if self.unit != Unit::Words {
            return parts.concat();
        }
        let mut out = String::new();
        for part in parts {
            if part.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(part);
        }
        out
    }

    /// Whether `text` starts with `prefix` **on a unit boundary**.
    ///
    /// For words that means whole words: `"the ca"` is not a prefix of
    /// `"the cat sat"`, `"the cat"` is.
    pub fn has_unit_prefix(&self, text: &str, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true;
        }
        if self.unit != Unit::Words {
            return text.starts_with(prefix);
        }
        text == prefix || (text.starts_with(prefix) && text.as_bytes().get(prefix.len()) == Some(&b' '))
    }

    /// The text read backwards, unit by unit: its last character first, or
    /// its last word - what a run with `reverse` trains on
    /// (`../../SPEC-SearchAndTraining.md` section 9).
    ///
    /// Characters are reversed code point by code point, so reversing twice
    /// gives the text back; a word encoding reverses the order of the words
    /// and writes them with single spaces - the layout it keeps anyway - each
    /// word's letters in their order.  Python's `Encoding.reverse`, character
    /// for character.
    pub fn reverse(&self, text: &str) -> String {
        match self.unit {
            Unit::Chars => text.chars().rev().collect(),
            Unit::Words => text.split_whitespace().rev().collect::<Vec<_>>().join(" "),
        }
    }

    // -- encoder -------------------------------------------------------------

    /// The grams of a text: `n` units each, `stride` units apart; empty when it
    /// holds fewer than `n` units.
    ///
    /// The tail that does not fill a whole gram is dropped, exactly as the
    /// trigram encoding drops the last two characters of a text.
    pub fn encode(&self, text: &str) -> Vec<String> {
        self.encode_units(&self.units(text))
    }

    /// [`Encoding::encode`] over an already-cut text.
    pub fn encode_units(&self, u: &Units) -> Vec<String> {
        let n = u.len();
        if n < self.n {
            return Vec::new();
        }
        let count = (n - self.n) / self.stride + 1;
        (0..count)
            .map(|i| u.slice(i * self.stride, i * self.stride + self.n))
            .collect()
    }

    /// How many units of a text its grams actually cover.
    pub fn covered(&self, units: usize) -> usize {
        if units < self.n {
            return 0;
        }
        (units - self.n) / self.stride * self.stride + self.n
    }

    /// The text this encoding can represent - what a round trip returns.
    ///
    /// For the default encoding that is the text itself; a word encoding loses
    /// the original spacing, and a grouping one the tail that does not fill a
    /// group.
    pub fn normalize(&self, text: &str) -> String {
        let u = self.units(text);
        let covered = self.covered(u.len());
        if covered == 0 {
            return String::new();
        }
        u.slice(0, covered)
    }

    // -- decoder -------------------------------------------------------------

    /// Decodes node labels in path order into text.
    ///
    /// Every label after the first contributes the units beyond the overlap;
    /// the first contributes `label[start_offset..]` when `include_context`,
    /// else `label[start_offset + n..]` - the deterministic remainder of a
    /// compressed node after the matched gram.
    pub fn decode_path(&self, labels: &[&str], start_offset: usize, include_context: bool) -> String {
        let first_cut = if include_context {
            start_offset
        } else {
            start_offset + self.n
        };
        let mut pieces: Vec<String> = Vec::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            let u = self.units(label);
            let cut = if i == 0 { first_cut } else { self.overlap() };
            pieces.push(u.slice(cut, u.len()));
        }
        let refs: Vec<&str> = pieces.iter().map(|s| s.as_str()).collect();
        self.join(&refs)
    }

    /// The text a run of consecutive grams spells.
    ///
    /// Each gram after the first contributes the `stride` units the previous
    /// one did not.
    pub fn decode_grams<'a>(&self, grams: impl IntoIterator<Item = &'a str>) -> String {
        let mut pieces: Vec<String> = Vec::new();
        for (i, gram) in grams.into_iter().enumerate() {
            if i == 0 {
                pieces.push(gram.to_string());
            } else {
                let u = self.units(gram);
                pieces.push(u.slice(self.overlap(), u.len()));
            }
        }
        let refs: Vec<&str> = pieces.iter().map(|s| s.as_str()).collect();
        self.join(&refs)
    }
}

impl fmt::Display for Encoding {
    /// The compact spec [`parse_encoding`] reads back: `char:3:1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.unit, self.n, self.stride)
    }
}

/// Reads a spec: `unit[:n[:stride]]`.
///
/// `unit` is char / chars / character / word / words, `n` defaults to
/// [`WINDOW`] and `stride` to 1 (a sliding window); `unit:n:n` and the
/// shorthand `unit:n:groups` are the non-overlapping form.  A few names for the
/// common ones are accepted too: `trigram`, `bigram`, `word-bigram`,
/// `word-trigram`.
pub fn parse_encoding(spec: &str) -> Result<Encoding, String> {
    let s = spec.trim().to_ascii_lowercase();
    if s.is_empty() {
        return Ok(Encoding::default());
    }
    match s.as_str() {
        "default" | "trigram" | "trigrams" => return Ok(Encoding::default()),
        "bigram" | "bigrams" => return Encoding::new(Unit::Chars, 2, 1),
        "word" | "words" | "unigram-words" | "word-unigram" => return Encoding::new(Unit::Words, 1, 1),
        "word-bigram" | "word-bigrams" | "bigram-words" => return Encoding::new(Unit::Words, 2, 1),
        "word-trigram" | "word-trigrams" | "trigram-words" => return Encoding::new(Unit::Words, 3, 1),
        _ => {}
    }
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() > 3 {
        return Err(format!("encoding {spec:?}: expected unit[:n[:stride]]"));
    }
    let unit = Unit::parse(parts[0])
        .ok_or_else(|| format!("encoding {spec:?}: unit must be char or word, got {:?}", parts[0]))?;
    let mut n = WINDOW;
    if let Some(text) = parts.get(1).filter(|p| !p.is_empty()) {
        n = text
            .parse()
            .map_err(|_| format!("encoding {spec:?}: n must be a number, got {text:?}"))?;
    }
    let mut stride = 1;
    if let Some(text) = parts.get(2).filter(|p| !p.is_empty()) {
        stride = match *text {
            "groups" | "group" | "blocks" | "block" => n,
            "sliding" | "slide" => 1,
            other => other
                .parse()
                .map_err(|_| format!("encoding {spec:?}: stride must be a number, got {other:?}"))?,
        };
    }
    Encoding::new(unit, n, stride).map_err(|why| format!("encoding {spec:?}: {why}"))
}

/// A text cut into units, indexed so that slicing by unit is O(1).
///
/// That is what the graph does all day - every split, every merge, every
/// re-index of a label walks its grams - so the text is cut once and sliced
/// many times.  `starts` holds the byte offset of every unit plus the end.
#[derive(Clone, Debug)]
pub struct Units {
    text: String,
    starts: Vec<usize>,
    /// The bytes between two units: 0 for characters, 1 for the space between words.
    sep: usize,
}

impl Units {
    /// One unit per code point; the text is kept as it is.
    fn chars(text: &str) -> Units {
        let mut starts: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
        starts.push(text.len());
        Units {
            text: text.to_string(),
            starts,
            sep: 0,
        }
    }

    /// One unit per whitespace-delimited word; the text is rewritten with
    /// single spaces, which is the layout a word encoding keeps.
    fn words(text: &str) -> Units {
        let fields: Vec<&str> = text.split_whitespace().collect();
        if fields.is_empty() {
            return Units {
                text: String::new(),
                starts: vec![0],
                sep: 1,
            };
        }
        let mut out = String::with_capacity(text.len());
        let mut starts = Vec::with_capacity(fields.len() + 1);
        for (i, field) in fields.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            starts.push(out.len());
            out.push_str(field);
        }
        starts.push(out.len());
        Units {
            text: out,
            starts,
            sep: 1,
        }
    }

    /// The number of units.
    pub fn len(&self) -> usize {
        self.starts.len() - 1
    }

    /// Whether the text holds no units.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole (normalised) text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Units `[from, to)` as text; out-of-range bounds are clamped and an empty
    /// range is `""`.
    pub fn slice(&self, from: usize, to: usize) -> String {
        let n = self.len();
        let to = to.min(n);
        if from >= to {
            return String::new();
        }
        let mut end = self.starts[to];
        if to < n {
            end -= self.sep; // the separator before unit `to` belongs to neither side
        }
        self.text[self.starts[from]..end].to_string()
    }

    /// One unit.
    pub fn at(&self, i: usize) -> String {
        self.slice(i, i + 1)
    }
}

/// The alphabet of a word encoding: every word its grams are made of, and how
/// many grams hold each.
///
/// A word encoding has no vocabulary object to consult - a gram is text - so
/// the alphabet the graph has read is whatever its grams are made of.  The gram
/// index is the one count that survives compression, so that is what is counted.
pub fn vocabulary<'a>(encoding: &Encoding, grams: impl IntoIterator<Item = &'a str>) -> Vec<(String, usize)> {
    use crate::hash::{map, Map};
    let mut counts: Map<String, usize> = map();
    for gram in grams {
        let u = encoding.units(gram);
        for i in 0..u.len() {
            *counts.entry(u.at(i)).or_insert(0) += 1;
        }
    }
    let mut rows: Vec<(String, usize)> = counts.into_iter().collect();
    // most read first, ties alphabetically - the Python and Go order, word for
    // word, so three implementations list one alphabet
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_character_trigram() {
        let e = Encoding::default();
        assert_eq!(e.to_string(), "char:3:1");
        assert!(e.is_default());
        assert!(e.sliding());
        assert_eq!(e.overlap(), OVERLAP);
        assert_eq!(e.units_name(), "chars");
        assert_eq!(e.encode("hello"), vec!["hel", "ell", "llo"]);
        assert_eq!(e.normalize("hello"), "hello");
    }

    #[test]
    fn any_n_and_any_stride() {
        let five = Encoding::new(Unit::Chars, 5, 1).unwrap();
        assert_eq!(five.encode("hello!"), vec!["hello", "ello!"]);
        let groups = Encoding::new(Unit::Chars, 4, 4).unwrap();
        assert_eq!(groups.encode("hello there"), vec!["hell", "o th"]);
        assert!(!groups.sliding());
        assert_eq!(groups.overlap(), 0);
        // the tail that does not fill a group is dropped, and normalize says so
        assert_eq!(groups.normalize("hello there"), "hello th");
        // a stride larger than n would skip units
        assert!(Encoding::new(Unit::Chars, 3, 4).is_err());
        assert!(Encoding::new(Unit::Chars, 0, 1).is_err());
    }

    #[test]
    fn a_word_encoding_counts_words() {
        let e = Encoding::new(Unit::Words, 2, 1).unwrap();
        assert_eq!(e.to_string(), "word:2:1");
        assert_eq!(e.units_name(), "words");
        assert_eq!(e.encode("the cat sat"), vec!["the cat", "cat sat"]);
        assert_eq!(e.len("the cat sat"), 3);
        // the layout is not kept: a round trip writes single spaces
        assert_eq!(e.normalize("the   cat\tsat"), "the cat sat");
        assert_eq!(e.join(&["the cat", "sat"]), "the cat sat");
        assert_eq!(e.join(&["", "sat"]), "sat");
        // a prefix is whole words
        assert!(e.has_unit_prefix("the cat sat", "the cat"));
        assert!(!e.has_unit_prefix("the cat sat", "the ca"));
        assert!(Encoding::default().has_unit_prefix("the cat sat", "the ca"));
    }

    #[test]
    fn the_specs_it_reads() {
        for (spec, want) in [
            ("", Encoding::default()),
            ("trigram", Encoding::default()),
            ("char:3:1", Encoding::default()),
            ("CHAR", Encoding::default()),
            ("char:5", Encoding::new(Unit::Chars, 5, 1).unwrap()),
            ("char:4:groups", Encoding::new(Unit::Chars, 4, 4).unwrap()),
            ("word:2:1", Encoding::new(Unit::Words, 2, 1).unwrap()),
            ("word-trigram", Encoding::new(Unit::Words, 3, 1).unwrap()),
            ("words", Encoding::new(Unit::Words, 1, 1).unwrap()),
        ] {
            assert_eq!(parse_encoding(spec).unwrap(), want, "{spec}");
        }
        for spec in ["sideways", "char:x", "char:3:x", "a:b:c:d", "char:3:9"] {
            assert!(parse_encoding(spec).is_err(), "{spec} was accepted");
        }
        // every spec round-trips through its own string form
        for e in [Encoding::default(), Encoding::new(Unit::Words, 2, 1).unwrap()] {
            assert_eq!(parse_encoding(&e.to_string()).unwrap(), e);
        }
    }

    #[test]
    fn units_slice_by_unit() {
        let e = Encoding::new(Unit::Words, 3, 1).unwrap();
        let u = e.units("the  cat   sat on");
        assert_eq!(u.len(), 4);
        assert_eq!(u.text(), "the cat sat on");
        assert_eq!(u.slice(1, 3), "cat sat");
        assert_eq!(u.at(0), "the");
        assert_eq!(u.slice(2, 99), "sat on"); // clamped
        assert_eq!(u.slice(3, 1), ""); // empty
        let c = Encoding::default().units("héllo");
        assert_eq!(c.len(), 5); // code points, not bytes
        assert_eq!(c.at(1), "é");
    }

    #[test]
    fn the_grams_spell_the_text_again() {
        for (e, text) in [
            (Encoding::default(), "hello there"),
            (Encoding::new(Unit::Chars, 5, 1).unwrap(), "hello there"),
            (Encoding::new(Unit::Words, 2, 1).unwrap(), "the cat sat on the mat"),
            (Encoding::new(Unit::Words, 3, 1).unwrap(), "the cat sat on the mat"),
        ] {
            let grams = e.encode(text);
            let back = e.decode_grams(grams.iter().map(|s| s.as_str()));
            assert_eq!(back, e.normalize(text), "{e}");
        }
    }

    #[test]
    fn the_alphabet_is_what_the_grams_are_made_of() {
        let e = Encoding::new(Unit::Words, 3, 1).unwrap();
        let rows = vocabulary(&e, ["the cat sat", "cat sat on"]);
        assert_eq!(
            rows,
            vec![
                ("cat".to_string(), 2),
                ("sat".to_string(), 2),
                ("on".to_string(), 1),
                ("the".to_string(), 1),
            ]
        );
    }
}
