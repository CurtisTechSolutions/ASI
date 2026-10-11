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
//! * `unit = Words, n = 2` is the word bigram, `n = 3` the word trigram;
//! * `unit = Base64` is the trigram of *base64 symbols*: anything - a text,
//!   an image, a video, a recording - is read as its bytes and written in
//!   base64 ([`base64_text`]), the graph learns n-grams of those symbols, and
//!   what the model says is decoded back into bytes ([`base64_bytes`]).
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

use crate::json::Json;

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
    /// One *sound*: a phoneme (`DH`, `AH0`, `K`), the `#` between two words or
    /// a pause, as the phonetic tokenizer (the sibling PhoneticTokenizer crate)
    /// writes them.  `"the cat"` is the units `DH AH0 # K AE1 T`, and a label is
    /// that text, so a model file stays readable.
    Phones,
    /// One syllable (`K.AE1.T`), a `#` or a pause.
    Syllables,
    /// One *acoustic unit*: a sound learned from audio (`q17`) by the phonetic
    /// tokenizer's codebook, with nothing written down.  A recording is heard
    /// as a text of units (`phonetic::hear_audio`), and that text is what the
    /// graph is built over; as text the units are tokens taken as they come,
    /// like words, and what the model says is spoken through the vocoder.
    Acoustic,
    /// One *base64 symbol*: six bits of the bytes of anything - a text, an
    /// image, a video, a recording.  Content becomes base64 where it enters
    /// ([`base64_text`]): a text as its UTF-8 bytes, a file as the bytes it
    /// holds, written in the 64 symbols `A-Z a-z 0-9 + /` with no padding.
    /// The graph is built over those symbols exactly as over characters, so
    /// the n-gram dials mean what they always do (`base64:4:4` is one gram per
    /// three bytes), and what the model says is decoded back into bytes
    /// ([`base64_bytes`]).  A text of the graph is already base64: the
    /// conversion happens once, where the content comes in, never inside the
    /// model.
    Base64,
}

impl Unit {
    /// The name the spec strings and the model file use.
    pub fn name(self) -> &'static str {
        match self {
            Unit::Chars => "char",
            Unit::Words => "word",
            Unit::Phones => "phone",
            Unit::Syllables => "syllable",
            Unit::Acoustic => "acoustic",
            Unit::Base64 => "base64",
        }
    }

    /// Whether the units are sounds rather than letters or words.
    pub fn phonetic(self) -> bool {
        matches!(self, Unit::Phones | Unit::Syllables)
    }

    /// Whether the units are whitespace-separated tokens taken as they come
    /// (words, acoustic units).
    pub fn tokens(self) -> bool {
        matches!(self, Unit::Words | Unit::Acoustic)
    }

    /// Whether a text slices into these units code point by code point:
    /// characters, and base64 symbols (Python's `CHARACTER_UNITS`).
    ///
    /// Every place that cuts, measures, joins or reverses a text letter by
    /// letter - or that has no words to list because a unit is a letter - asks
    /// this rather than naming the character unit, so the symbols take the
    /// path the characters take, wherever that is.
    pub fn characters(self) -> bool {
        matches!(self, Unit::Chars | Unit::Base64)
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
            Unit::Phones => "phones",
            Unit::Syllables => "syllables",
            Unit::Acoustic => "units",
            Unit::Base64 => "symbols",
        }
    }

    /// The unit a name means, or `None`.
    pub fn parse(name: &str) -> Option<Unit> {
        match name {
            "char" | "chars" | "character" | "characters" | "letter" | "letters" => Some(Unit::Chars),
            "word" | "words" => Some(Unit::Words),
            "phone" | "phones" | "phoneme" | "phonemes" | "sound" | "sounds" => Some(Unit::Phones),
            "syllable" | "syllables" | "syl" => Some(Unit::Syllables),
            "acoustic" | "acoustics" | "audio" | "unit" | "units" => Some(Unit::Acoustic),
            "base64" | "b64" => Some(Unit::Base64),
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
        if self.unit.phonetic() {
            crate::phonetic::tokenizer(self.unit)?; // a phonetic unit needs its tokenizer: better refused now than mid-run
        }
        if self.unit == Unit::Acoustic {
            crate::phonetic::acoustic_tokenizer()?; // and the acoustic unit its codebook, to hear recordings and be heard
        }
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
            Unit::Phones => "phone",
            Unit::Syllables => "syllable",
            Unit::Acoustic => "unit",
            Unit::Base64 => "symbol",
        };
        let kind = if self.sliding() { "sliding" } else { "groups" };
        format!("{}-{unit} grams, stride {} ({kind})", self.n, self.stride)
    }

    // -- units ---------------------------------------------------------------

    /// The text cut into units, indexed so that slicing by unit is O(1).
    pub fn units(&self, text: &str) -> Units {
        match self.unit {
            // a base64 symbol is one code point, as a character is
            Unit::Chars | Unit::Base64 => Units::chars(text),
            Unit::Words | Unit::Acoustic => Units::words(text),
            // the text read through the tokenizer: a word becomes its sounds, punctuation a
            // pause, the gap between two words a #.  Text that is already sounds passes
            // through unchanged, so a label cuts into the units it was made of.
            Unit::Phones | Unit::Syllables => Units::words(&crate::phonetic::text(self.unit, text)),
        }
    }

    /// How many units a text holds.
    pub fn len(&self, text: &str) -> usize {
        match self.unit {
            Unit::Chars | Unit::Base64 => text.chars().count(),
            Unit::Words | Unit::Acoustic => text.split_whitespace().count(),
            Unit::Phones | Unit::Syllables => crate::phonetic::text(self.unit, text).split_whitespace().count(),
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
        if self.unit.characters() {
            return parts.concat();
        }
        let mut out = String::new();
        for part in parts {
            // a piece given as text joins as the sounds it makes, so a joined text is all sounds
            let sounds;
            let part: &str = if self.unit.phonetic() {
                sounds = crate::phonetic::text(self.unit, part);
                sounds.as_str()
            } else {
                part
            };
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
        if self.unit.characters() {
            return text.starts_with(prefix);
        }
        let sounds;
        let prefix = if self.unit.phonetic() {
            // a prefix given as text is looked for as the sounds it makes
            sounds = crate::phonetic::text(self.unit, prefix);
            if sounds.is_empty() {
                return true;
            }
            sounds.as_str()
        } else {
            prefix
        };
        text == prefix || (text.starts_with(prefix) && text.as_bytes().get(prefix.len()) == Some(&b' '))
    }

    /// The words a phonetic text spells (`"DH AH0 # K AE1 T"` -> `"the cat"`); a
    /// text given in words is read as the sounds it makes first, so it spells
    /// itself back.  A character or word encoding returns the text as it is.
    pub fn spell(&self, text: &str) -> String {
        if !self.unit.phonetic() {
            return text.to_string();
        }
        crate::phonetic::spell(self.unit, text)
    }

    /// What `tail` - the last units of `whole` - spells, as the part of
    /// `spell(whole)` it wrote.  A continuation spelled on its own loses the
    /// space before its first word, and cannot finish a word the text before
    /// it began, so the whole text is spelled and what the text before the
    /// tail spells is cut off its front: `spell_tail("DH AH0 # K AE1 T # S AE1
    /// T", "# S AE1 T")` is `" sat"`, and the two pieces join back into
    /// `spell(whole)`.  When the tail finished a word the head began, the cut
    /// falls back to where that word starts.  A tail that does not end `whole`
    /// is spelled on its own.  A character or word encoding returns the tail
    /// as it is.  Python's `Encoding.spell_tail`, character for character.
    pub fn spell_tail(&self, whole: &str, tail: &str) -> String {
        if !self.unit.phonetic() {
            return tail.to_string();
        }
        let Some(before) = whole.strip_suffix(tail) else {
            return self.spell(tail);
        };
        let spelled = self.spell(whole);
        let mut head = self.spell(before);
        if !spelled.starts_with(&head) {
            // the tail finished a word the head began: cut where that word starts
            let common: usize = spelled
                .chars()
                .zip(head.chars())
                .take_while(|(a, b)| a == b)
                .map(|(a, _)| a.len_utf8())
                .sum();
            let cut = spelled[..common].rfind(' ').map_or(0, |i| i + 1);
            head = spelled[..cut].to_string();
        }
        spelled[head.len()..].to_string()
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
            // base64 symbols too: the symbols backwards, not the bytes
            Unit::Chars | Unit::Base64 => text.chars().rev().collect(),
            // words, sounds or acoustic units: the units in reverse order (a piece given as
            // text is first turned into its sounds, as `join` does)
            _ => self
                .join(&[text])
                .split_whitespace()
                .rev()
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    // -- content: what comes in, and what goes out ----------------------------

    /// Whether the units are base64 symbols - six bits of the bytes of any
    /// content ([`Unit::Base64`]).
    pub fn is_base64(&self) -> bool {
        self.unit == Unit::Base64
    }

    /// Content - the bytes of a file - as the text this encoding reads.
    ///
    /// Base64 writes the bytes in its symbols, whatever they hold
    /// ([`base64_text`]); every other unit reads them as UTF-8 text, and bytes
    /// that are not UTF-8 are no text it can read.  Python's
    /// `Encoding.content_text`.
    pub fn content_text(&self, data: &[u8]) -> Result<String, String> {
        if self.is_base64() {
            return Ok(base64_text(data));
        }
        std::str::from_utf8(data)
            .map(str::to_string)
            .map_err(|err| err.to_string())
    }

    /// A text given in plain words - a sentence to train on, to score or to
    /// blame - as this encoding reads it.
    ///
    /// Base64 writes its UTF-8 bytes in symbols; every other unit takes it as
    /// it is.  Python's `Encoding.text_input`.
    pub fn text_input(&self, text: &str) -> String {
        if self.is_base64() {
            return base64_text(text.as_bytes());
        }
        text.to_string()
    }

    /// A prefix given in plain words, as this encoding asks it.
    ///
    /// Base64 asks the symbols its UTF-8 bytes fix ([`base64_prefix`]), so the
    /// walk writes on from the last whole symbol; every other unit takes it as
    /// it is.  Python's `Encoding.prefix_input`.
    pub fn prefix_input(&self, text: &str) -> String {
        if self.is_base64() {
            return base64_prefix(text.as_bytes());
        }
        text.to_string()
    }

    /// What a text of this encoding holds, as bytes: base64 symbols decoded
    /// ([`base64_bytes`]), any other text as its UTF-8.  Python's
    /// `Encoding.content_bytes`.
    pub fn content_bytes(&self, text: &str) -> Vec<u8> {
        if self.is_base64() {
            return base64_bytes(text);
        }
        text.as_bytes().to_vec()
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
/// `unit` is char / chars / character / word / words, a sound (phone,
/// syllable, acoustic) or base64 / b64; `n` defaults to [`WINDOW`] and
/// `stride` to 1 (a sliding window); `unit:n:n` and the shorthand
/// `unit:n:groups` are the non-overlapping form, so `base64:4:4` is one gram
/// per three bytes of any content.  A few names for the common ones are
/// accepted too: `trigram`, `bigram`, `word-bigram`, `word-trigram`.
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
    let unit = Unit::parse(parts[0]).ok_or_else(|| {
        format!(
            "encoding {spec:?}: unit must be char, word, phone, syllable, acoustic or base64, got {:?}",
            parts[0]
        )
    })?;
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

// -- the base64 unit: content in, content out -----------------------------------

/// The 64 symbols of the base64 unit: RFC 4648's standard alphabet, in its
/// order.
pub const BASE64_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Bytes as base64 symbols: the standard alphabet, no padding (3 bytes -> 4
/// symbols, 1 -> 2, 2 -> 3).
///
/// This is the encoder of the base64 unit: a text is its UTF-8 bytes, a file
/// the bytes it holds, and either becomes the text a model over base64
/// symbols is built from.  Written out here rather than borrowed from the kit,
/// whose own base64 pads (the model never depends on the kit).  Python's
/// `base64_text`.
pub fn base64_text(data: &[u8]) -> String {
    let alphabet = BASE64_ALPHABET.as_bytes();
    let mut out = String::with_capacity((data.len() * 8).div_ceil(6));
    for group in data.chunks(3) {
        // the group's bytes as 24 bits, a missing byte as zeros; one symbol
        // per six bits the bytes reach into - two for one byte, three for two
        let bits = group
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &b)| acc | (u32::from(b) << (16 - 8 * i)));
        for i in 0..=group.len() {
            out.push(char::from(alphabet[((bits >> (18 - 6 * i)) & 63) as usize]));
        }
    }
    out
}

/// The symbols every continuation of `data` shares: the first
/// `floor(8 * len / 6)` of its base64.
///
/// The last symbol of a text's base64 also holds bits of the bytes that would
/// follow it, so a prefix given as content is asked as the symbols it fixes:
/// `b"the c"` -> `"dGhlIG"`, and the walk writes on from there - the bits of
/// `c` that the next symbol carries included.  Python's `base64_prefix`.
pub fn base64_prefix(data: &[u8]) -> String {
    let mut symbols = base64_text(data);
    symbols.truncate(data.len() * 8 / 6); // every symbol is one byte of ASCII
    symbols
}

/// The six bits a base64 symbol stands for, or `None` for anything outside
/// the 64.
fn symbol_value(byte: u8) -> Option<u32> {
    match byte {
        b'A'..=b'Z' => Some(u32::from(byte - b'A')),
        b'a'..=b'z' => Some(u32::from(byte - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(byte - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Base64 symbols back to bytes, whatever was written: the decoder of the
/// base64 unit, which never fails.
///
/// Anything outside the 64 symbols is skipped (padding, whitespace, a stray
/// character, the URL-safe `-` and `_`), a last symbol that cannot complete a
/// byte is dropped, and the bits a final partial group has over are ignored -
/// so a predicted or generated text always decodes, into `floor(6 * symbols /
/// 8)` bytes (`"YR"` is `b"a"`).  Python's `base64_bytes`.
pub fn base64_bytes(text: &str) -> Vec<u8> {
    // a character outside ASCII is no symbol, and none of its bytes is one either
    let mut values: Vec<u32> = text.bytes().filter_map(symbol_value).collect();
    if values.len() % 4 == 1 {
        values.pop();
    }
    let mut out = Vec::with_capacity(values.len() * 3 / 4);
    for group in values.chunks(4) {
        let bits = group
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &v)| acc | (v << (18 - 6 * i)));
        // four symbols make three bytes, three make two, two make one
        for i in 0..group.len() * 6 / 8 {
            out.push((bits >> (16 - 8 * i)) as u8);
        }
    }
    out
}

/// Whether bytes read as UTF-8 text - strictly, as Python's decoder reads
/// them: a surrogate or an overlong form is no text.  What decides whether a
/// file is cut into lines or taken whole.  Python's `is_utf8`.
pub fn is_utf8(data: &[u8]) -> bool {
    std::str::from_utf8(data).is_ok()
}

/// The bytes a blank line of a text file is made of: ASCII whitespace.
const ASCII_SPACE: &[u8] = b" \t\n\r\x0b\x0c";

/// The texts a file gives a model of base64 symbols, each in its symbols
/// ([`base64_text`]).
///
/// A file that reads as UTF-8 text gives one text per line - split on `\n`,
/// each line's bytes as they are (a `\r` stays, a byte-order mark stays), a
/// line of nothing but ASCII whitespace skipped - unless `whole_file`.
/// Anything else (an image, a video, a recording, an archive), and any file
/// taken whole, is one text: every byte it holds, so the decoded text is the
/// file again.  An empty file gives nothing.  Python's `base64_file_texts`.
pub fn base64_file_texts(data: &[u8], whole_file: bool) -> Vec<String> {
    if data.is_empty() {
        return Vec::new();
    }
    if !whole_file && is_utf8(data) {
        return data
            .split(|&b| b == b'\n')
            .filter(|line| line.iter().any(|b| !ASCII_SPACE.contains(b)))
            .map(base64_text)
            .collect();
    }
    vec![base64_text(data)]
}

/// What a text of a model of base64 symbols holds: `decoded`,
/// `decoded_continuation` and `decoded_bytes`.
///
/// `decoded` is the whole text - the prefix and what the walk wrote after it -
/// decoded ([`base64_bytes`]) and read as UTF-8, a byte sequence that is not
/// text showing as U+FFFD: one per maximal invalid subpart, the Unicode
/// Standard's practice, which is what `String::from_utf8_lossy` writes and
/// what Python's `decode("utf-8", "replace")` writes.  `decoded_bytes` is how
/// many bytes that is.  `decoded_continuation` is what the model wrote that
/// `prefix` - the content the walk was asked to continue, as bytes - did not
/// already hold: everything after the bytes the two have in common.  A prefix
/// that ends inside a symbol leaves its last byte to the walk, which may write
/// it differently (asked `the d` it may say `the log`), so the continuation
/// starts where the model's bytes and the prefix's part.
///
/// Every other encoding gets no pairs, as a model that is not sounds gets no
/// `spelled` ([`crate::phonetic::spelled_prediction`]): the fields are there
/// only when they say something.  Python's `decoded_prediction`.
pub fn decoded_prediction(enc: Encoding, full_text: &str, prefix: &[u8]) -> Vec<(String, Json)> {
    if !enc.is_base64() {
        return Vec::new();
    }
    let whole = base64_bytes(full_text);
    let same = whole.iter().zip(prefix).take_while(|(a, b)| a == b).count();
    vec![
        (
            "decoded".to_string(),
            Json::str(String::from_utf8_lossy(&whole).into_owned()),
        ),
        (
            "decoded_continuation".to_string(),
            Json::str(String::from_utf8_lossy(&whole[same..]).into_owned()),
        ),
        ("decoded_bytes".to_string(), Json::Int(whole.len() as i64)),
    ]
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

    // -- the base64 unit (Python's tests/test_base64_unit.py) ------------------------

    /// A tiny binary blob that is not UTF-8: the head of a PNG and some bytes
    /// that are no text - what an image or a recording looks like to the unit.
    const BLOB: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1,
    ];

    fn b64() -> Encoding {
        Encoding::new(Unit::Base64, 3, 1).unwrap()
    }

    /// Bytes nobody chose: a small linear congruential generator, so the
    /// round trips run over every value a byte takes without a dependency.
    fn noise(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect()
    }

    fn decoded(pairs: &[(String, Json)], key: &str) -> Json {
        pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or(Json::Null)
    }

    #[test]
    fn the_base64_dial_is_parsed_named_and_written() {
        for spec in ["base64", "b64", "BASE64:3:1", " base64 ", "b64:3:sliding"] {
            assert_eq!(parse_encoding(spec).unwrap(), b64(), "{spec}");
        }
        let groups = parse_encoding("base64:4:4").unwrap();
        assert_eq!((groups.unit, groups.n, groups.stride), (Unit::Base64, 4, 4));
        assert_eq!(groups.to_string(), "base64:4:4");
        assert_eq!(parse_encoding(&groups.to_string()).unwrap(), groups);
        assert_eq!(groups.describe(), "4-symbol grams, stride 4 (groups)");
        assert_eq!(b64().describe(), "3-symbol grams, stride 1 (sliding)");
        assert_eq!(groups.units_name(), "symbols");
        assert_eq!(Unit::Base64.name(), "base64");
        assert!(groups.is_base64() && !groups.is_default());
        assert!(!Encoding::default().is_base64());
        assert!(Unit::Base64.characters() && Unit::Chars.characters());
        assert!(!Unit::Words.characters() && !Unit::Phones.characters() && !Unit::Acoustic.characters());
        assert!(!Unit::Base64.phonetic() && !Unit::Base64.tokens());
        let refused = parse_encoding("base65").unwrap_err();
        assert!(refused.contains("base64"), "{refused}");
    }

    /// The symbols slice like characters: one code point, one unit.
    #[test]
    fn a_symbol_is_one_unit() {
        let e = b64();
        let text = "dGhlIGNhdA";
        assert_eq!(e.units(text).len(), 10);
        assert_eq!(e.units(text).text(), text);
        assert_eq!(e.len(text), 10);
        assert_eq!(e.slice(text, 2, 5), "hlI");
        assert_eq!(e.join(&["dGh", "", "lIG"]), "dGhlIG");
        assert!(e.has_unit_prefix("dGhlIG", "dGh"));
        assert!(!e.has_unit_prefix("dGhlIG", "dGx"));
        assert_eq!(e.reverse("abc+/"), "/+cba");
        assert_eq!(e.encode("dGhlIA"), vec!["dGh", "Ghl", "hlI", "lIA"]);
        let groups = parse_encoding("base64:4:4").unwrap();
        assert_eq!(groups.encode("dGhlIGNhdA"), vec!["dGhl", "IGNh"]);
        assert_eq!(groups.normalize("dGhlIGNhdA"), "dGhlIGNh");
        assert_eq!(e.spell("dGhl"), "dGhl", "symbols are their own spelling");
    }

    #[test]
    fn bytes_become_symbols_without_padding() {
        assert_eq!(base64_text(b""), "");
        assert_eq!(base64_text(b"a"), "YQ");
        assert_eq!(base64_text(b"ab"), "YWI");
        assert_eq!(base64_text(b"abc"), "YWJj");
        assert_eq!(base64_text(b"the cat"), "dGhlIGNhdA");
        assert_eq!(base64_text(&[0xff, 0xfe, 0xfd]), "//79");
        assert_eq!(base64_text(&[0xfb, 0xff]), "+/8");
        // every byte value, against the standard alphabet with the padding taken off
        let every: Vec<u8> = (0..=255).collect();
        assert!(base64_text(&every).starts_with("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4v"));
        assert!(base64_text(&every).ends_with("8PHy8/T19vf4+fr7/P3+/w"));
        assert!(base64_text(BLOB).chars().all(|c| BASE64_ALPHABET.contains(c)));
        assert_eq!(BASE64_ALPHABET.len(), 64);
    }

    #[test]
    fn any_bytes_come_back_from_their_symbols() {
        for size in (0..40).chain([255, 1000, 4097]) {
            let data = noise(7 + size as u32, size);
            let symbols = base64_text(&data);
            assert_eq!(symbols.len(), (size * 8).div_ceil(6), "{size} bytes");
            assert_eq!(base64_bytes(&symbols), data, "{size} bytes");
        }
    }

    /// A predicted text always decodes: padding, whitespace and stray
    /// characters are skipped, a dangling symbol is dropped, and the bits a
    /// last partial group has over are ignored.
    #[test]
    fn the_decoder_takes_whatever_was_written() {
        assert_eq!(base64_bytes(""), b"");
        assert_eq!(base64_bytes("YQ=="), b"a");
        assert_eq!(base64_bytes("YW\nJj \t"), b"abc");
        assert_eq!(
            base64_bytes("Y-W_J.j"),
            b"abc",
            "the URL-safe symbols are not this alphabet's"
        );
        assert_eq!(base64_bytes("YWJjZ"), b"abc", "one symbol cannot make a byte");
        assert_eq!(base64_bytes("Y"), b"");
        assert_eq!(base64_bytes("YR"), b"a", "R's low four bits are left over");
        assert_eq!(base64_bytes("YWJ"), b"ab");
        assert_eq!(
            base64_bytes("Y\u{e9}W\u{1f600}Jj"),
            b"abc",
            "a character outside ASCII is no symbol"
        );
        assert_eq!(base64_bytes("//79"), [0xff, 0xfe, 0xfd]);
    }

    #[test]
    fn a_prefix_is_the_symbols_its_bytes_fix() {
        assert_eq!(base64_prefix(b""), "");
        assert_eq!(base64_prefix(b"t"), "d");
        assert_eq!(base64_prefix(b"th"), "dG");
        assert_eq!(base64_prefix(b"the"), "dGhl");
        assert_eq!(base64_prefix(b"the d"), "dGhlIG");
        assert_eq!(base64_prefix(b"the cat"), "dGhlIGNhd");
        for size in 0..30 {
            let data = noise(3 + size as u32, size);
            let prefix = base64_prefix(&data);
            assert_eq!(prefix.len(), size * 8 / 6, "{size} bytes");
            // every continuation of the bytes shares it
            for tail in [&b""[..], b"\x00", b"\xff", b"xyz"] {
                let longer = [&data[..], tail].concat();
                assert!(base64_text(&longer).starts_with(&prefix), "{size} bytes + {tail:?}");
            }
        }
    }

    #[test]
    fn utf8_is_read_strictly() {
        assert!(is_utf8("the cat \u{20ac}".as_bytes()));
        assert!(is_utf8(b""));
        assert!(!is_utf8(BLOB));
        assert!(!is_utf8(b"\xed\xa0\x80"), "a surrogate is not UTF-8");
        assert!(!is_utf8(b"\xc0\xaf"), "nor is an overlong form");
        assert!(!is_utf8(b"\xf0\x90\x80"), "nor a sequence cut short");
    }

    #[test]
    fn a_text_file_is_one_text_per_line() {
        let data = b"one\r\n  \ntwo\n\n\tthree \n";
        let texts = base64_file_texts(data, false);
        assert_eq!(
            texts,
            vec![base64_text(b"one\r"), base64_text(b"two"), base64_text(b"\tthree ")]
        );
        let back: Vec<Vec<u8>> = texts.iter().map(|t| base64_bytes(t)).collect();
        assert_eq!(back, vec![b"one\r".to_vec(), b"two".to_vec(), b"\tthree ".to_vec()]);
        // a byte-order mark is a byte of the first line like any other
        assert_eq!(
            base64_file_texts(b"\xef\xbb\xbfhi\n", false),
            vec![base64_text(b"\xef\xbb\xbfhi")]
        );
    }

    #[test]
    fn anything_else_is_one_text_of_every_byte() {
        assert_eq!(base64_file_texts(BLOB, false), vec![base64_text(BLOB)]);
        assert_eq!(base64_bytes(&base64_file_texts(BLOB, false)[0]), BLOB);
        let data = b"one\ntwo\n\n";
        assert_eq!(base64_file_texts(data, true), vec![base64_text(data)]);
        assert_eq!(base64_bytes(&base64_file_texts(data, true)[0]), data);
        // nothing comes from nothing; a blank file taken whole is its bytes
        assert!(base64_file_texts(b"", false).is_empty());
        assert!(base64_file_texts(b"", true).is_empty());
        assert!(base64_file_texts(b" \n\t\n", false).is_empty());
        assert_eq!(base64_file_texts(b" \n\t\n", true), vec![base64_text(b" \n\t\n")]);
    }

    #[test]
    fn content_comes_in_and_goes_out() {
        let e = b64();
        assert_eq!(e.text_input("the cat"), "dGhlIGNhdA");
        assert_eq!(e.prefix_input("the cat"), "dGhlIGNhd");
        assert_eq!(e.content_text(BLOB).unwrap(), base64_text(BLOB));
        assert_eq!(e.content_bytes(&base64_text(BLOB)), BLOB);
        // every other unit takes text as it is, and gives its UTF-8 back
        let plain = Encoding::default();
        assert_eq!(plain.text_input("the cat"), "the cat");
        assert_eq!(plain.prefix_input("the cat"), "the cat");
        assert_eq!(plain.content_text("caf\u{e9}".as_bytes()).unwrap(), "caf\u{e9}");
        assert!(plain.content_text(BLOB).is_err(), "bytes that are no text");
        assert_eq!(plain.content_bytes("caf\u{e9}"), "caf\u{e9}".as_bytes());
    }

    #[test]
    fn a_prediction_decodes() {
        let e = b64();
        assert_eq!(
            Json::Obj(decoded_prediction(e, "dGhlIGNhdCBzYXQ", b"the cat")),
            Json::obj([
                ("decoded", Json::str("the cat sat")),
                ("decoded_continuation", Json::str(" sat")),
                ("decoded_bytes", Json::Int(11)),
            ])
        );
        // no prefix: all of it was written
        let all = decoded_prediction(e, "dGhlIGNhdA", b"");
        assert_eq!(decoded(&all, "decoded_continuation").as_str(), Some("the cat"));
        // every other encoding adds nothing
        assert!(decoded_prediction(Encoding::default(), "the cat sat", b"the cat").is_empty());
    }

    /// Asked `the d`, the symbols fix 4 bytes and half of the fifth; a walk
    /// that finishes it as `l` said `the log`, and the continuation is
    /// everything after the bytes the two share.
    #[test]
    fn the_last_byte_of_a_prefix_is_the_walks_to_finish() {
        let e = b64();
        let prefix = base64_prefix(b"the d");
        assert_eq!(prefix, "dGhlIG");
        let out = decoded_prediction(e, &format!("{prefix}xvZw"), b"the d");
        assert_eq!(decoded(&out, "decoded").as_str(), Some("the log"));
        assert_eq!(decoded(&out, "decoded_continuation").as_str(), Some("log"));
        let out = decoded_prediction(e, &base64_text(b"the dog"), b"the d");
        assert_eq!(decoded(&out, "decoded").as_str(), Some("the dog"));
        assert_eq!(decoded(&out, "decoded_continuation").as_str(), Some("og"));
    }

    /// One U+FFFD per maximal invalid subpart - the Unicode Standard's
    /// practice, Python's `errors="replace"`, and what the Go port writes too.
    #[test]
    fn bytes_that_are_not_text_read_as_replacement_characters() {
        for (data, text) in [
            (&b"ok\xe2\x82A\xff"[..], "ok\u{fffd}A\u{fffd}"), // a truncated sequence is one subpart
            (b"\xed\xa0\x80", "\u{fffd}\u{fffd}\u{fffd}"),    // a surrogate: three
            (b"\xf0\x90\x80", "\u{fffd}"),                    // a four-byte sequence cut short at the end
            (b"\xc0\xaf", "\u{fffd}\u{fffd}"),                // an overlong form: two
            (b"\xf4\x90\x80\x80", "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"), // past U+10FFFF: four
            (b"\xe2\x82\xac", "\u{20ac}"),
        ] {
            let out = decoded_prediction(b64(), &base64_text(data), b"");
            assert_eq!(decoded(&out, "decoded").as_str(), Some(text), "{data:?}");
            assert_eq!(decoded(&out, "decoded_bytes").as_i64(), Some(data.len() as i64));
        }
    }
}
