//! How a text becomes the grams the graph is built from, and how it comes back.
//!
//! [`Encoding`] is three dials - what one *unit* of text is, how many units a
//! gram holds, and how far apart consecutive grams start:
//!
//! * `Encoding::default()` is the one the model was born with: character
//!   trigrams, stride 1 (`"hello"` -> `["hel", "ell", "llo"]`).
//! * `Encoding { n: 5, .. }` is a sliding window of five characters, and any n
//!   works.
//! * `Encoding { n: 4, stride: 4, .. }` is *tokenisation*: non-overlapping
//!   groups of four letters.
//! * `Encoding { unit: Unit::Words, n: 2, stride: 1 }` is the word bigram.
//!
//! A gram is a [`Gram`]: up to three characters packed into one `u64` - the
//! representation choice this port makes and the reason the default encoding's
//! pass allocates nothing at all - or, for anything longer and for words, the
//! text itself.  The other two implementations carry every gram as a string.

use std::borrow::Cow;

/// The default n of the n-gram: the trigram the model was born with.
pub const WINDOW: usize = 3;
/// The overlap that goes with it (stride 1).  A graph's own overlap is
/// `graph.enc.overlap()`.
pub const OVERLAP: usize = WINDOW - 1;

/// Bits one code point takes in a packed gram: Unicode ends at `U+10FFFF`.
const BITS: u32 = 21;
/// The most characters that fit in one `u64` at [`BITS`] each.
pub const MAX_PACKED: usize = 3;

/// The labels of the three sentinel nodes.
pub const START_LABEL: &str = "<s>";
pub const END_LABEL: &str = "</s>";
/// The third sentinel: where the graph has learned a walk goes round.
pub const BACK_LABEL: &str = "<back>";

/// A window of [`WINDOW`] characters, packed 21 bits per code point.
///
/// Unicode ends at `U+10FFFF`, which is 21 bits, so three characters fit in 63
/// bits: a trigram is one `Copy` integer - hashable, comparable and free of any
/// allocation - rather than a string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct Trigram(pub u64);

impl Trigram {
    /// Packs three characters.
    #[inline]
    pub fn pack(a: char, b: char, c: char) -> Trigram {
        Trigram(a as u64 | (b as u64) << 21 | (c as u64) << 42)
    }

    /// Packs the first three characters of `s`, or `None` when it is shorter.
    pub fn parse(s: &str) -> Option<Trigram> {
        let mut it = s.chars();
        let (a, b, c) = (it.next()?, it.next()?, it.next()?);
        Some(Trigram::pack(a, b, c))
    }

    /// The three characters back out.
    #[inline]
    pub fn chars(self) -> [char; 3] {
        let one = |shift: u32| char::from_u32(((self.0 >> shift) & 0x1f_ffff) as u32).unwrap_or('\u{fffd}');
        [one(0), one(21), one(42)]
    }

    /// Appends the trigram to a string.
    pub fn write_to(self, out: &mut String) {
        for ch in self.chars() {
            out.push(ch);
        }
    }
}

impl std::fmt::Display for Trigram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::with_capacity(3);
        self.write_to(&mut s);
        f.write_str(&s)
    }
}

// -- what a unit is ------------------------------------------------------------

/// What one position of a text is: a character, or a whitespace word.
///
/// It is the atom everything downstream counts in - the n of the n-gram, the
/// stride, a node's label length, the length of a prediction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Unit {
    /// One unit is one character (one code point).  `"hello"` is 5 units.
    #[default]
    Chars,
    /// One unit is one whitespace-delimited word; the text is normalised to
    /// single spaces on the way in - a word encoding keeps the words, not the
    /// layout.
    Words,
}

impl Unit {
    /// The name the other two implementations write into a model file.
    pub fn name(self) -> &'static str {
        match self {
            Unit::Chars => "char",
            Unit::Words => "word",
        }
    }

    /// Bytes written between two units of a label.
    fn sep(self) -> usize {
        match self {
            Unit::Chars => 0,
            Unit::Words => 1,
        }
    }
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// -- the encoding ---------------------------------------------------------------

/// How a text becomes grams, and how the labels of a walk become text again.
///
/// Fixed for a graph's life: every label, every index key and every offset is
/// measured in the units of the encoding that built it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Encoding {
    pub unit: Unit,
    /// Units per gram: the n of the n-gram, any `n >= 1`.
    pub n: usize,
    /// Units between two consecutive grams.  `1` slides the window, so grams
    /// share `n - 1` units - which is what lets the graph chain them; `n` cuts
    /// the text into groups that share nothing.
    pub stride: usize,
}

impl Default for Encoding {
    fn default() -> Self {
        Encoding {
            unit: Unit::Chars,
            n: WINDOW,
            stride: 1,
        }
    }
}

impl Encoding {
    /// Fails with what is wrong, if anything.
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

    /// Units two consecutive grams share: `n - stride`.
    pub fn overlap(&self) -> usize {
        self.n - self.stride
    }

    /// Do consecutive grams overlap at all?
    pub fn sliding(&self) -> bool {
        self.stride < self.n
    }

    /// The character trigram of stride 1 - what a model file leaves unwritten.
    pub fn is_default(&self) -> bool {
        *self == Encoding::default()
    }

    /// Can a gram of this encoding be packed into a `u64`?
    fn packs(&self) -> bool {
        self.unit == Unit::Chars && self.n <= MAX_PACKED
    }

    /// The human form: `"2-word grams, stride 1 (sliding)"`.
    pub fn describe(&self) -> String {
        let unit = if self.unit == Unit::Words { "word" } else { "character" };
        let kind = if self.sliding() { "sliding" } else { "groups" };
        format!("{}-{unit} grams, stride {} ({kind})", self.n, self.stride)
    }

    /// Reads a spec: `unit[:n[:stride]]`, or one of the names below.
    ///
    /// `"char:3:1"` is the default, `"char:5:groups"` (or `"char:5:5"`)
    /// non-overlapping groups of five letters, `"word:2"` the word bigram.
    pub fn parse(spec: &str) -> Result<Encoding, String> {
        let text = spec.trim().to_lowercase();
        match text.as_str() {
            "" | "default" | "trigram" | "trigrams" => return Ok(Encoding::default()),
            "bigram" | "bigrams" => {
                return Ok(Encoding {
                    n: 2,
                    ..Encoding::default()
                })
            }
            "word" | "words" | "word-unigram" => {
                return Ok(Encoding {
                    unit: Unit::Words,
                    n: 1,
                    stride: 1,
                })
            }
            "word-bigram" | "word-bigrams" => {
                return Ok(Encoding {
                    unit: Unit::Words,
                    n: 2,
                    stride: 1,
                })
            }
            "word-trigram" | "word-trigrams" => {
                return Ok(Encoding {
                    unit: Unit::Words,
                    n: 3,
                    stride: 1,
                })
            }
            _ => {}
        }
        let parts: Vec<&str> = text.split(':').collect();
        if parts.len() > 3 {
            return Err(format!("encoding {spec:?}: expected unit[:n[:stride]]"));
        }
        let unit = match parts[0] {
            "char" | "chars" | "character" | "characters" | "letter" | "letters" => Unit::Chars,
            "word" | "words" => Unit::Words,
            other => return Err(format!("encoding {spec:?}: unit must be char or word, got {other:?}")),
        };
        let mut n = WINDOW;
        if parts.len() > 1 && !parts[1].is_empty() {
            n = parts[1]
                .parse()
                .map_err(|_| format!("encoding {spec:?}: n must be a number, got {:?}", parts[1]))?;
        }
        let mut stride = 1;
        if parts.len() > 2 && !parts[2].is_empty() {
            stride = match parts[2] {
                "groups" | "group" | "blocks" | "block" => n,
                "sliding" | "slide" => 1,
                other => other
                    .parse()
                    .map_err(|_| format!("encoding {spec:?}: stride must be a number, got {other:?}"))?,
            };
        }
        let enc = Encoding { unit, n, stride };
        enc.validate().map_err(|e| format!("encoding {spec:?}: {e}"))?;
        Ok(enc)
    }

    // -- units -------------------------------------------------------------

    /// The text cut into this encoding's units, indexed so that slicing by
    /// unit is O(1) - which is what the graph does all day.
    pub fn units<'a>(&self, text: &'a str) -> Units<'a> {
        match self.unit {
            Unit::Chars => {
                let mut starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
                starts.push(text.len());
                Units {
                    text: Cow::Borrowed(text),
                    starts,
                    sep: Unit::Chars.sep(),
                }
            }
            Unit::Words => {
                let fields: Vec<&str> = text.split_whitespace().collect();
                let mut joined = String::with_capacity(text.len());
                let mut starts = Vec::with_capacity(fields.len() + 1);
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        joined.push(' ');
                    }
                    starts.push(joined.len());
                    joined.push_str(field);
                }
                starts.push(joined.len());
                Units {
                    text: Cow::Owned(joined),
                    starts,
                    sep: Unit::Words.sep(),
                }
            }
        }
    }

    /// How many units a text holds.
    pub fn len(&self, text: &str) -> usize {
        match self.unit {
            Unit::Chars => text.chars().count(),
            Unit::Words => text.split_whitespace().count(),
        }
    }

    /// Units `[from, to)` of a text; `to = None` means "to the end".
    pub fn slice(&self, text: &str, from: usize, to: Option<usize>) -> String {
        match self.unit {
            Unit::Chars => char_slice(text, from, to).to_string(),
            Unit::Words => self.units(text).slice(from, to).to_string(),
        }
    }

    /// Glues unit-aligned pieces: nothing between characters, one space
    /// between words.  Empty pieces are dropped, so a label contributing no
    /// unit adds no separator.
    pub fn join(&self, parts: &[&str]) -> String {
        if self.unit != Unit::Words {
            return parts.concat();
        }
        let kept: Vec<&str> = parts.iter().copied().filter(|p| !p.is_empty()).collect();
        kept.join(" ")
    }

    /// The first `n` units of a text.
    pub fn truncate(&self, text: &str, n: usize) -> String {
        self.slice(text, 0, Some(n))
    }

    /// Does `text` start with `prefix` *on a unit boundary*?  For words that
    /// means whole words: `"the ca"` is not a prefix of `"the cat sat"`.
    pub fn has_unit_prefix(&self, text: &str, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true;
        }
        if self.unit != Unit::Words {
            return text.starts_with(prefix);
        }
        text == prefix || (text.starts_with(prefix) && text.as_bytes().get(prefix.len()) == Some(&b' '))
    }

    // -- grams ---------------------------------------------------------------

    /// The gram at unit offset `at` of an already-cut text.
    pub fn gram(&self, units: &Units<'_>, at: usize) -> Gram {
        let text = units.slice(at, Some(at + self.n));
        if self.packs() {
            Gram::packed(text)
        } else {
            Gram::Text(Box::new(text.to_string()))
        }
    }

    /// A gram read from text, or `None` when it is not `n` units long.
    pub fn gram_of(&self, text: &str) -> Option<Gram> {
        if self.len(text) != self.n {
            return None;
        }
        Some(if self.packs() {
            Gram::packed(text)
        } else {
            Gram::Text(Box::new(text.to_string()))
        })
    }

    /// How many units of a text its grams cover: everything for a sliding
    /// window, everything but the ragged tail for groups.
    pub fn covered(&self, units: usize) -> usize {
        if units < self.n {
            return 0;
        }
        (units - self.n) / self.stride * self.stride + self.n
    }

    /// Does gram `x` overlap gram `y` - do the last `n - stride` units of the
    /// one open the other?  Free of any allocation where both are packed,
    /// which is where a training pass asks it.
    pub fn overlaps(&self, x: &Gram, y: &Gram) -> bool {
        let ov = self.overlap();
        if ov == 0 {
            return true; // groups share nothing: any two chain
        }
        if let (Gram::Packed { bits: xb, .. }, Gram::Packed { bits: yb, .. }) = (x, y) {
            // the units above the stride of one against the units below the
            // overlap of the other, as plain integers
            return (xb >> (BITS * self.stride as u32)) == (yb & mask(ov));
        }
        let (xs, ys) = (x.to_string(), y.to_string());
        self.slice(&xs, self.stride, None) == self.slice(&ys, 0, Some(ov))
    }

    /// The grams of a text: `n` units each, `stride` units apart.  Empty when
    /// the text holds fewer than `n` units.
    ///
    /// The tail that does not fill a whole gram is dropped, exactly as the
    /// trigram encoding drops the last two characters; [`Encoding::normalize`]
    /// says what is left.
    pub fn encode(&self, text: &str) -> Vec<Gram> {
        if self.packs() {
            // the hot path of a training pass: a ring of characters packed
            // straight into the gram, so the pass allocates nothing at all
            let mut window = ['\0'; MAX_PACKED];
            let mut filled = 0usize;
            let mut since = 0usize;
            let mut out = Vec::new();
            for ch in text.chars() {
                if filled < self.n {
                    window[filled] = ch;
                    filled += 1;
                } else {
                    window.copy_within(1..self.n, 0);
                    window[self.n - 1] = ch;
                }
                if filled == self.n {
                    if since == 0 {
                        out.push(Gram::of_chars(&window[..self.n]));
                        since = self.stride;
                    }
                    since -= 1;
                }
            }
            return out;
        }
        if self.unit == Unit::Chars {
            // more characters than fit in a `u64`: a ring of byte offsets, so
            // only the gram itself is allocated
            let mut offsets: Vec<usize> = vec![0; self.n];
            let (mut filled, mut since) = (0usize, 0usize);
            let mut out = Vec::new();
            for (bi, ch) in text.char_indices() {
                if filled < self.n {
                    offsets[filled] = bi;
                    filled += 1;
                } else {
                    offsets.copy_within(1..self.n, 0);
                    offsets[self.n - 1] = bi;
                }
                if filled == self.n {
                    if since == 0 {
                        out.push(Gram::Text(Box::new(text[offsets[0]..bi + ch.len_utf8()].to_string())));
                        since = self.stride;
                    }
                    since -= 1;
                }
            }
            return out;
        }
        let units = self.units(text);
        let n = units.len();
        if n < self.n {
            return Vec::new();
        }
        let mut out = Vec::with_capacity((n - self.n) / self.stride + 1);
        let mut at = 0;
        while at + self.n <= n {
            out.push(self.gram(&units, at));
            at += self.stride;
        }
        out
    }

    /// The text this encoding can represent - what a round trip through the
    /// graph returns, and so what a round trip is checked against.
    pub fn normalize(&self, text: &str) -> String {
        let units = self.units(text);
        let covered = self.covered(units.len());
        if covered == 0 {
            return String::new();
        }
        units.slice(0, Some(covered)).to_string()
    }

    // -- decoder --------------------------------------------------------------

    /// Decodes node labels in path order into text.  Every label after the
    /// first contributes its part beyond the overlap; the first contributes
    /// `label[start_offset..]` when `include_context`, else
    /// `label[start_offset + n..]` (the deterministic remainder of a
    /// compressed node after the matched gram).  Offsets are in units.
    pub fn decode_path(&self, labels: &[&str], start_offset: usize, include_context: bool) -> String {
        let first_cut = if include_context {
            start_offset
        } else {
            start_offset + self.n
        };
        if self.unit == Unit::Chars {
            let mut out = String::new();
            for (i, label) in labels.iter().enumerate() {
                let cut = if i == 0 { first_cut } else { self.overlap() };
                out.push_str(char_slice(label, cut, None));
            }
            return out;
        }
        let parts: Vec<String> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let cut = if i == 0 { first_cut } else { self.overlap() };
                self.slice(label, cut, None)
            })
            .collect();
        self.join(&parts.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// Decodes raw grams in order - the inverse of [`Encoding::encode`], up to
    /// the tail it dropped.
    pub fn decode_grams(&self, grams: &[Gram]) -> String {
        let texts: Vec<String> = grams.iter().map(Gram::to_string).collect();
        let parts: Vec<String> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if i == 0 {
                    t.clone()
                } else {
                    self.slice(t, self.overlap(), None)
                }
            })
            .collect();
        self.join(&parts.iter().map(String::as_str).collect::<Vec<_>>())
    }
}

impl std::fmt::Display for Encoding {
    /// The compact spec [`Encoding::parse`] reads back: `"char:3:1"`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.unit, self.n, self.stride)
    }
}

// -- units ----------------------------------------------------------------------

/// A text cut into units, indexed so that slicing by unit is O(1).
///
/// `starts` holds the byte offset of every unit plus the end of the text.  For
/// words the text is the normalised (single-spaced) form, which is why it may
/// be owned.
pub struct Units<'a> {
    text: Cow<'a, str>,
    starts: Vec<usize>,
    sep: usize,
}

impl<'a> Units<'a> {
    /// The number of units.
    pub fn len(&self) -> usize {
        self.starts.len() - 1
    }

    /// Is the text empty of units?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole (normalised) text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Units `[from, to)` as text; `to = None` means "to the end".  Bounds are
    /// clamped and an empty range is `""`.
    pub fn slice(&self, from: usize, to: Option<usize>) -> &str {
        let n = self.len();
        let to = to.map_or(n, |t| t.min(n));
        if from >= to {
            return "";
        }
        // the separator before unit `to` belongs to neither side
        let end = if to < n {
            self.starts[to] - self.sep
        } else {
            self.starts[to]
        };
        &self.text[self.starts[from]..end]
    }
}

// -- a gram ---------------------------------------------------------------------

/// One gram: the key the graph's index is built on.
///
/// Up to [`MAX_PACKED`] characters live in a `u64` at [`BITS`] each - `Copy`,
/// hashable and free of any allocation, which is what keeps the default
/// encoding's pass allocation-free.  Anything longer, and every word gram, is
/// the text itself.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Gram {
    /// `n` characters packed [`BITS`] bits each.  `n` rides along so that two
    /// grams of different lengths never collide, which costs nothing: the
    /// variant is padded to the pointer either way.
    Packed { bits: u64, n: u8 },
    /// The gram as text: more than [`MAX_PACKED`] characters, or words.  The
    /// string is behind one thin pointer rather than a fat one, so a `Gram` is
    /// 16 bytes and not 24 - the index holds one per gram in the corpus, and
    /// a training pass walks the whole of it.
    Text(Box<String>),
}

/// The low `n` units of a packed gram.
#[inline]
fn mask(n: usize) -> u64 {
    if n == 0 {
        0
    } else {
        (1u64 << (BITS * n as u32)) - 1
    }
}

/// Hashing a packed gram writes the one `u64` and nothing else - what hashing
/// the index key cost before it could also be a string.  A packed gram and a
/// text one may collide, which is what `Eq` is for; inside one graph only one
/// of the two ever occurs, because the encoding is fixed.
impl std::hash::Hash for Gram {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Gram::Packed { bits, .. } => state.write_u64(*bits),
            Gram::Text(s) => s.hash(state),
        }
    }
}

impl Gram {
    /// Packs a text of at most [`MAX_PACKED`] characters.
    fn packed(text: &str) -> Gram {
        let mut bits = 0u64;
        let mut n = 0u8;
        for (i, ch) in text.chars().enumerate() {
            bits |= (ch as u64) << (BITS * i as u32);
            n = i as u8 + 1;
        }
        Gram::Packed { bits, n }
    }

    /// Packs characters straight, without going through a string.
    #[inline]
    fn of_chars(chars: &[char]) -> Gram {
        let mut bits = 0u64;
        for (i, ch) in chars.iter().enumerate() {
            bits |= (*ch as u64) << (BITS * i as u32);
        }
        Gram::Packed {
            bits,
            n: chars.len() as u8,
        }
    }

    /// The characters back out of a packed gram.
    pub fn chars(&self) -> Vec<char> {
        match self {
            Gram::Packed { bits, n } => (0..*n as u32)
                .map(|i| char::from_u32(((bits >> (BITS * i)) & 0x1f_ffff) as u32).unwrap_or('\u{fffd}'))
                .collect(),
            Gram::Text(s) => s.chars().collect(),
        }
    }

    /// Appends the gram to a string.
    pub fn write_to(&self, out: &mut String) {
        match self {
            Gram::Packed { .. } => out.extend(self.chars()),
            Gram::Text(s) => out.push_str(s),
        }
    }
}

impl std::fmt::Display for Gram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gram::Packed { .. } => {
                let mut s = String::with_capacity(MAX_PACKED);
                self.write_to(&mut s);
                f.write_str(&s)
            }
            Gram::Text(s) => f.write_str(s),
        }
    }
}

impl From<Trigram> for Gram {
    fn from(t: Trigram) -> Gram {
        Gram::Packed {
            bits: t.0,
            n: MAX_PACKED as u8,
        }
    }
}

/// The overlapping trigrams of `text` under the default encoding.
pub fn encode(text: &str) -> Vec<Gram> {
    Encoding::default().encode(text)
}

/// Decodes node labels under the default encoding.
pub fn decode_path(labels: &[&str], start_offset: usize, include_context: bool) -> String {
    Encoding::default().decode_path(labels, start_offset, include_context)
}

/// The character (code point) length of a string - the unit the model measures
/// text in, on all three implementations.
#[inline]
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `s[from..to]` in characters; `to = None` means "to the end".
pub fn char_slice(s: &str, from: usize, to: Option<usize>) -> &str {
    let mut start = s.len();
    let mut end = s.len();
    for (i, (bi, _)) in s.char_indices().enumerate() {
        if i == from {
            start = bi;
        }
        if let Some(to) = to {
            if i == to {
                end = bi;
                break;
            }
        }
    }
    if from == 0 {
        start = 0;
    }
    if start > end {
        return "";
    }
    &s[start..end]
}

/// The first `n` characters of `s`.
pub fn truncate_chars(s: &str, n: usize) -> &str {
    char_slice(s, 0, Some(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_of_three() {
        let grams = encode("hello");
        let words: Vec<String> = grams.iter().map(|t| t.to_string()).collect();
        assert_eq!(words, vec!["hel", "ell", "llo"]);
        assert!(encode("hi").is_empty());
    }

    #[test]
    fn trigrams_survive_the_packing() {
        for text in ["abc", "  a", "é€語", "\u{10ffff}ab"] {
            let t = Trigram::parse(text).unwrap();
            assert_eq!(t.to_string(), text);
        }
    }

    #[test]
    fn a_path_decodes_back_to_its_text() {
        assert_eq!(decode_path(&["hel", "ell", "llo"], 0, true), "hello");
        assert_eq!(decode_path(&["hello"], 0, false), "lo");
        assert_eq!(char_slice("héllo", 1, Some(3)), "él");
        assert_eq!(truncate_chars("héllo", 2), "hé");
    }
}
