//! The trigram encoding and its inverse.
//!
//! A text goes into the graph through a sliding window of three characters,
//! stride 1: `"hello"` -> `["hel", "ell", "llo"]`.  Every window is a
//! [`Trigram`], three code points packed into one `u64` - the Go port carries
//! the same windows as freshly allocated strings, which is the one
//! representation choice this port makes differently, and the reason its
//! encoding pass allocates nothing at all.

/// The sliding-window length of the encoding.
pub const WINDOW: usize = 3;
/// The number of characters two consecutive labels share.
pub const OVERLAP: usize = WINDOW - 1;

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

/// The overlapping windows of `text` (stride 1); empty when the text is
/// shorter than [`WINDOW`] characters.
pub fn encode(text: &str) -> Vec<Trigram> {
    let n = char_len(text);
    if n < WINDOW {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - WINDOW + 1);
    let mut window = ['\0'; WINDOW];
    let mut filled = 0usize;
    for ch in text.chars() {
        if filled < WINDOW {
            window[filled] = ch;
            filled += 1;
        } else {
            window[0] = window[1];
            window[1] = window[2];
            window[2] = ch;
        }
        if filled == WINDOW {
            out.push(Trigram::pack(window[0], window[1], window[2]));
        }
    }
    out
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

/// Decodes node labels in path order into text.  Every label after the first
/// contributes its part beyond the overlap; the first contributes
/// `label[start_offset..]` when `include_context`, else
/// `label[start_offset + WINDOW..]` (the deterministic remainder of a
/// compressed node after the matched window).
pub fn decode_path(labels: &[&str], start_offset: usize, include_context: bool) -> String {
    let first_cut = if include_context {
        start_offset
    } else {
        start_offset + WINDOW
    };
    let mut out = String::new();
    for (i, label) in labels.iter().enumerate() {
        if i == 0 {
            out.push_str(char_slice(label, first_cut, None));
        } else {
            out.push_str(char_slice(label, OVERLAP, None));
        }
    }
    out
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
