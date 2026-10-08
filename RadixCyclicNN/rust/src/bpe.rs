//! The traditional LLM tokenizer: byte-level byte-pair encoding, the way GPT-2
//! and its descendants read text - the Rust twin of `radixnet/bpe.py`, token
//! for token (`tests/tokens_fixture.json` holds the three ports to it).
//!
//! A text is cut into pre-tokens by GPT-2's pattern (a word with the space
//! before it, a run of digits, a run of punctuation, a run of whitespace, an
//! English contraction), each pre-token's UTF-8 bytes are written in GPT-2's
//! byte alphabet (`Ġ` for the space, `Ċ` for the newline) and merged pairwise
//! by a ranked list of merges - the pair with the best rank anywhere in the
//! pre-token is merged everywhere it occurs, left to right - until no pair of
//! the list is left.  Every byte is a token of its own, so nothing is
//! unreadable.  Ids 0-255 are the bytes, the merged tokens follow in rank
//! order, and `<|endoftext|>` closes the vocabulary.  A space is put before
//! every text, and [`BpeTokenizer::decode`] takes it away again.
//!
//! The *text form* is what a graph over tokens is built from: a token that is
//! a space and printable ASCII is written bare (`"Ġcat"` is `cat`), any other
//! glued on with `⁀` (`"ing"` is `⁀ing`), units joined by single spaces.  It
//! is idempotent - any run of its units reads back as itself - which is what
//! lets a label cut into the tokens it was made of (`../SPEC-Tokens.md`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// The mark of a glued token in the text form: `⁀` (U+2040 CHARACTER TIE),
/// outside the byte alphabet, so no token holds it.
pub const GLUE: char = '\u{2040}';

/// The special token that closes the vocabulary.
pub const ENDOFTEXT: &str = "<|endoftext|>";

/// The environment variable that names a merges file to read instead of the bundled one.
pub const ENV: &str = "RADIXNET_TOKENIZER";

/// The merges that ship with the package: `radixnet/data/merges.txt`.
pub const BUNDLED_MERGES: &str = include_str!("../../radixnet/data/merges.txt");

// -- the byte alphabet ----------------------------------------------------------

fn byte_chars() -> &'static ([char; 256], [i16; 324]) {
    static TABLES: OnceLock<([char; 256], [i16; 324])> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut chars = ['\0'; 256];
        let mut back = [-1i16; 324];
        let mut moved = 0u32;
        for (b, slot) in chars.iter_mut().enumerate() {
            let keep = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
            let c = if keep {
                b as u32
            } else {
                moved += 1;
                255 + moved
            };
            *slot = char::from_u32(c).expect("the byte alphabet is made of real characters");
            back[c as usize] = b as i16;
        }
        (chars, back)
    })
}

/// The character a byte is written as: `byte_char(0x20) == 'Ġ'`.
pub fn byte_char(b: u8) -> char {
    byte_chars().0[b as usize]
}

/// The byte a character of the byte alphabet stands for.
pub fn char_byte(c: char) -> Option<u8> {
    let back = &byte_chars().1;
    let code = c as usize;
    if code < back.len() && back[code] >= 0 {
        Some(back[code] as u8)
    } else {
        None
    }
}

/// Bytes written in the byte alphabet: `b" cat"` -> `"Ġcat"`.
pub fn byte_text(data: &[u8]) -> String {
    data.iter().map(|&b| byte_char(b)).collect()
}

/// The bytes a string of the byte alphabet stands for; `None` when a character is not in it.
pub fn text_bytes(text: &str) -> Option<Vec<u8>> {
    text.chars().map(char_byte).collect()
}

// -- the pre-tokenizer ----------------------------------------------------------

/// The classes of the pre-tokenizer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Space,
    Digit,
    Punct,
    Letter,
}

impl Class {
    /// The one-letter name the fixture uses.
    pub fn letter(self) -> &'static str {
        match self {
            Class::Space => "S",
            Class::Digit => "D",
            Class::Punct => "P",
            Class::Letter => "L",
        }
    }
}

/// Where the punctuation class lies: every ASCII character that is not a
/// letter, a digit or whitespace, and a fixed list of Unicode blocks -
/// `radixnet/bpe.py`'s `_PUNCT_RANGES`, range for range.
const PUNCT_RANGES: [(u32, u32); 20] = [
    (0x0000, 0x002F),
    (0x003A, 0x0040),
    (0x005B, 0x0060),
    (0x007B, 0x00BF),
    (0x00D7, 0x00D7),
    (0x00F7, 0x00F7),
    (0x2000, 0x206F),
    (0x20A0, 0x20CF),
    (0x2190, 0x2BFF),
    (0x2E00, 0x2E7F),
    (0x3001, 0x3003),
    (0x3008, 0x3011),
    (0x3014, 0x301F),
    (0xFE00, 0xFE0F),
    (0xFF01, 0xFF0F),
    (0xFF1A, 0xFF20),
    (0xFF3B, 0xFF40),
    (0xFF5B, 0xFF65),
    (0x1F000, 0x1FAFF),
    (0xE0020, 0xE007F),
];

/// The Unicode `White_Space` property, written out: the 25 code points the
/// three ports agree are space (`char::is_whitespace` is the same set).
fn is_white_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{85}' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}' | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// The class of one character: space, digit, punctuation, or letter
/// (everything else, every script's letters included).
pub fn char_class(c: char) -> Class {
    if is_white_space(c) {
        return Class::Space;
    }
    if c.is_ascii_digit() {
        return Class::Digit;
    }
    let code = c as u32;
    for &(lo, hi) in &PUNCT_RANGES {
        if code < lo {
            break;
        }
        if code <= hi {
            return Class::Punct;
        }
    }
    Class::Letter
}

/// How many characters of `'s 't 're 've 'm 'll 'd` (either case) start at `rs[i]`, or 0.
fn contraction(rs: &[char], i: usize) -> usize {
    if rs[i] != '\'' || i + 1 >= rs.len() {
        return 0;
    }
    let a = rs[i + 1].to_ascii_lowercase();
    if matches!(a, 's' | 't' | 'm' | 'd') {
        return 2;
    }
    if i + 2 < rs.len() {
        let b = rs[i + 2].to_ascii_lowercase();
        if matches!((a, b), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return 3;
        }
    }
    0
}

/// GPT-2's pre-tokens: `'s|'t|'re|'ve|'m|'ll|'d| ?L+| ?D+| ?P+|\s+(?!\S)|\s+`
/// over the classes above.  The pieces join back into the text exactly.
pub fn pretokenize(text: &str) -> Vec<String> {
    let rs: Vec<char> = text.chars().collect();
    let n = rs.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let k = contraction(&rs, i);
        if k > 0 {
            out.push(rs[i..i + k].iter().collect());
            i += k;
            continue;
        }
        let j = if rs[i] == ' ' && i + 1 < n && char_class(rs[i + 1]) != Class::Space {
            i + 1
        } else {
            i
        };
        let c = char_class(rs[j]);
        if c != Class::Space {
            let mut k = j + 1;
            while k < n && char_class(rs[k]) == c {
                k += 1;
            }
            out.push(rs[i..k].iter().collect());
            i = k;
            continue;
        }
        let mut k = i + 1;
        while k < n && char_class(rs[k]) == Class::Space {
            k += 1;
        }
        if k < n && k - i >= 2 {
            k -= 1; // leave the last whitespace character to what follows
        }
        out.push(rs[i..k].iter().collect());
        i = k;
    }
    out
}

// -- UTF-8 ---------------------------------------------------------------------------

/// Bytes read as UTF-8, a broken sequence as one U+FFFD for each maximal
/// broken piece (the Unicode practice Python follows).  Unless `last`, a valid
/// but unfinished sequence at the end is handed back unread.
pub fn decode_utf8(data: &[u8], last: bool) -> (String, Vec<u8>) {
    let mut out = String::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let c = data[i];
        if c < 0x80 {
            out.push(c as char);
            i += 1;
            continue;
        }
        let (need, lo, hi) = match c {
            0xC2..=0xDF => (1, 0x80, 0xBF),
            0xE0 => (2, 0xA0, 0xBF),
            0xE1..=0xEC | 0xEE | 0xEF => (2, 0x80, 0xBF),
            0xED => (2, 0x80, 0x9F),
            0xF0 => (3, 0x90, 0xBF),
            0xF1..=0xF3 => (3, 0x80, 0xBF),
            0xF4 => (3, 0x80, 0x8F),
            _ => {
                // a continuation byte where none belongs, or a byte UTF-8 never uses
                out.push(char::REPLACEMENT_CHARACTER);
                i += 1;
                continue;
            }
        };
        let mut k = 1;
        while k <= need {
            if i + k >= data.len() {
                if !last {
                    return (out, data[i..].to_vec());
                }
                break;
            }
            let d = data[i + k];
            let ok = if k == 1 {
                (lo..=hi).contains(&d)
            } else {
                (0x80..=0xBF).contains(&d)
            };
            if !ok {
                break;
            }
            k += 1;
        }
        if k <= need {
            // broken after k - 1 good continuation bytes: one replacement for all of them
            out.push(char::REPLACEMENT_CHARACTER);
            i += k;
            continue;
        }
        out.push_str(std::str::from_utf8(&data[i..i + need + 1]).expect("checked above"));
        i += need + 1;
    }
    (out, Vec::new())
}

// -- the tokenizer ----------------------------------------------------------------

const CACHE_LIMIT: usize = 1 << 16;

/// A byte-level BPE tokenizer: ranked merges over the byte alphabet, ids, and
/// the text form.  Safe to share between threads.
pub struct BpeTokenizer {
    /// id -> token, in the byte alphabet: the 256 bytes, then the merged tokens.
    pub vocab: Vec<String>,
    /// The merges, by rank.
    pub merges: Vec<(String, String)>,
    pub special: Vec<String>,
    pub note: String,
    ids: HashMap<String, usize>,
    ranks: HashMap<(String, String), usize>,
    special_at: HashMap<String, usize>,
    words: Mutex<HashMap<String, Vec<String>>>,
    bare: Mutex<HashMap<String, bool>>,
}

impl BpeTokenizer {
    /// A tokenizer over merges (in rank order) and special tokens.
    pub fn new(merges: &[(String, String)], special: &[&str], note: &str) -> Result<BpeTokenizer, String> {
        let mut t = BpeTokenizer {
            vocab: (0..=255u8).map(|b| byte_char(b).to_string()).collect(),
            merges: Vec::new(),
            special: Vec::new(),
            note: note.to_string(),
            ids: HashMap::new(),
            ranks: HashMap::new(),
            special_at: HashMap::new(),
            words: Mutex::new(HashMap::new()),
            bare: Mutex::new(HashMap::new()),
        };
        for (i, tok) in t.vocab.iter().enumerate() {
            t.ids.insert(tok.clone(), i);
        }
        for (left, right) in merges {
            t.add_merge(left, right)?;
        }
        for s in special {
            if t.special_at.contains_key(*s) {
                return Err(format!("the special token {s:?} is listed twice"));
            }
            t.special_at.insert(s.to_string(), t.special.len());
            t.special.push(s.to_string());
        }
        Ok(t)
    }

    fn add_merge(&mut self, left: &str, right: &str) -> Result<(), String> {
        let pair = (left.to_string(), right.to_string());
        let number = self.merges.len() + 1;
        if self.ranks.contains_key(&pair) {
            return Err(format!("merge {number} ({left} {right}) is listed twice"));
        }
        for side in [left, right] {
            if !self.ids.contains_key(side) {
                return Err(format!(
                    "merge {number} ({left} {right}): {side:?} is not a token yet - a merge joins two tokens the \
                     bytes and the merges before it made"
                ));
            }
        }
        self.ranks.insert(pair.clone(), self.merges.len());
        self.merges.push(pair);
        let joined = format!("{left}{right}");
        if !self.ids.contains_key(&joined) {
            self.ids.insert(joined.clone(), self.vocab.len());
            self.vocab.push(joined);
        }
        Ok(())
    }

    /// Reads a merges file: GPT-2's `merges.txt` - lines of `left right`, `#`
    /// starting a comment line and `#note:` lines kept as the note.
    pub fn parse(text: &str) -> Result<BpeTokenizer, String> {
        let mut merges = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        for (number, raw) in text.split('\n').enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                if let Some(note) = rest.strip_prefix("note:") {
                    notes.push(note.trim().to_string());
                }
                continue;
            }
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                return Err(format!(
                    "line {}: a merge is two tokens separated by one space, got {line:?}",
                    number + 1
                ));
            }
            for side in &parts {
                if text_bytes(side).is_none() {
                    return Err(format!(
                        "line {}: {side:?} is not written in the byte alphabet",
                        number + 1
                    ));
                }
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }
        BpeTokenizer::new(&merges, &[ENDOFTEXT], &notes.join("\n")).map_err(|e| format!("merges: {e}"))
    }

    /// Reads a merges file from disk.
    pub fn load(path: &str) -> Result<BpeTokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        BpeTokenizer::parse(&text)
    }

    /// The tokenizer over the merges that ship with the package.
    pub fn bundled() -> Result<BpeTokenizer, String> {
        BpeTokenizer::parse(BUNDLED_MERGES)
    }

    /// Every id: the bytes, the merged tokens and the special tokens.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special.len()
    }

    /// The id of a token in the byte alphabet, or of a special token.
    pub fn token_id(&self, token: &str) -> Option<usize> {
        if let Some(&id) = self.ids.get(token) {
            return Some(id);
        }
        self.special_at.get(token).map(|at| self.vocab.len() + at)
    }

    /// The token an id stands for (a special token as itself).
    pub fn id_token(&self, id: usize) -> Result<&str, String> {
        if id < self.vocab.len() {
            return Ok(&self.vocab[id]);
        }
        self.special
            .get(id - self.vocab.len())
            .map(|s| s.as_str())
            .ok_or_else(|| format!("id {id} is outside a vocabulary of {}", self.vocab_size()))
    }

    /// GPT-2's `bpe` over one pre-token's byte characters: the pair with the
    /// lowest rank is merged everywhere it occurs, left to right and without
    /// overlap, again and again.
    fn merge_word(&self, mut parts: Vec<String>) -> Vec<String> {
        // one key, reused, so looking a pair up allocates nothing new
        let mut key = (String::new(), String::new());
        while parts.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..parts.len() - 1 {
                key.0.clear();
                key.0.push_str(&parts[i]);
                key.1.clear();
                key.1.push_str(&parts[i + 1]);
                if let Some(&rank) = self.ranks.get(&key) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((at, _)) = best else { break };
            let (first, second) = (parts[at].clone(), parts[at + 1].clone());
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == first && parts[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut parts[i]));
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
    }

    fn word(&self, pretoken: &str) -> Vec<String> {
        if let Some(found) = self.words.lock().unwrap_or_else(|e| e.into_inner()).get(pretoken) {
            return found.clone();
        }
        let parts: Vec<String> = pretoken.bytes().map(|b| byte_char(b).to_string()).collect();
        let found = self.merge_word(parts);
        let mut cache = self.words.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(pretoken.to_string(), found.clone());
        found
    }

    fn tokens_of(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for p in pretokenize(text) {
            out.extend(self.word(&p));
        }
        out
    }

    /// The tokens of a text in the byte alphabet, read with a space before it:
    /// `"the cat."` -> `[Ġthe, Ġcat, .]`.  A special token in the text is its characters.
    pub fn tokens(&self, text: &str) -> Vec<String> {
        self.tokens_of(&format!(" {text}"))
    }

    /// The ids of a text; `allowed_special` turns `<|endoftext|>` in it into its id.
    pub fn encode(&self, text: &str, allowed_special: bool) -> Vec<usize> {
        let framed = format!(" {text}");
        let mut ids = Vec::new();
        let add = |part: &str, ids: &mut Vec<usize>| {
            for tok in self.tokens_of(part) {
                ids.push(self.ids[&tok]);
            }
        };
        if !allowed_special || self.special.is_empty() {
            add(&framed, &mut ids);
            return ids;
        }
        let mut specials: Vec<&String> = self.special.iter().collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let mut at = 0;
        loop {
            // the leftmost special, the longest where two start together
            let mut next: Option<(usize, &String)> = None;
            for s in &specials {
                if let Some(k) = framed[at..].find(s.as_str()) {
                    if next.is_none_or(|(n, _)| at + k < n) {
                        next = Some((at + k, s));
                    }
                }
            }
            let Some((start, which)) = next else { break };
            add(&framed[at..start], &mut ids);
            ids.push(self.vocab.len() + self.special_at[which.as_str()]);
            at = start + which.len();
        }
        add(&framed[at..], &mut ids);
        ids
    }

    /// The bytes a token stands for; a special token is its own text.
    pub fn token_bytes(&self, token: &str) -> Result<Vec<u8>, String> {
        if self.special_at.contains_key(token) {
            return Ok(token.as_bytes().to_vec());
        }
        text_bytes(token).ok_or_else(|| format!("{token:?} is not a token of the byte alphabet"))
    }

    /// The bytes of a run of ids, the leading space the encoder put there taken away.
    pub fn decode_bytes(&self, ids: &[usize]) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(self.token_bytes(self.id_token(id)?)?);
        }
        if out.first() == Some(&b' ') {
            out.remove(0);
        }
        Ok(out)
    }

    /// The text of a run of ids: the inverse of [`BpeTokenizer::encode`],
    /// exactly.  A run cut inside a character decodes what is broken as
    /// U+FFFD, one for each maximal broken piece - Python's rule.
    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        Ok(decode_utf8(&self.decode_bytes(ids)?, true).0)
    }

    // -- the text form ---------------------------------------------------------

    /// A token as the text form writes it: `"Ġcat"` -> `cat`, `"ing"` -> `⁀ing`.
    pub fn render(token: &str) -> String {
        if let Some(rest) = token.strip_prefix('Ġ') {
            if !rest.is_empty() && rest.bytes().all(|b| (b'!'..=b'~').contains(&b)) {
                return rest.to_string();
            }
        }
        format!("{GLUE}{token}")
    }

    /// The token a piece of the text form stands for.  A glued piece is the
    /// token after the mark, if the vocabulary has it; a bare piece is
    /// printable ASCII and stands for the token of a space and itself - but only
    /// when that is what the tokenizer makes of it as text, so the two readings agree.
    pub fn read(&self, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        if let Some(rest) = piece.strip_prefix(GLUE) {
            return self.ids.contains_key(rest).then(|| rest.to_string());
        }
        let known = self.bare.lock().unwrap_or_else(|e| e.into_inner()).get(piece).copied();
        let ok = match known {
            Some(ok) => ok,
            None => {
                let token = format!("Ġ{piece}");
                let ok = piece.bytes().all(|b| (b'!'..=b'~').contains(&b)) && {
                    let tokens = self.tokens_of(&format!(" {piece}"));
                    tokens.len() == 1 && tokens[0] == token
                };
                let mut cache = self.bare.lock().unwrap_or_else(|e| e.into_inner());
                if cache.len() >= CACHE_LIMIT {
                    cache.clear();
                }
                cache.insert(piece.to_string(), ok);
                ok
            }
        };
        ok.then(|| format!("Ġ{piece}"))
    }

    /// A text as the units of the text form: `"The cat sat."` -> `[The, cat,
    /// sat, ⁀.]`.  Text that already is the text form - single spaces between
    /// pieces that each read - is read piece by piece; anything else as text.
    pub fn units(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut units = Vec::new();
        for piece in text.split(' ') {
            match self.read(piece) {
                Some(tok) => units.push(BpeTokenizer::render(&tok)),
                None => return self.tokens(text).iter().map(|t| BpeTokenizer::render(t)).collect(),
            }
        }
        units
    }

    /// The text form of a text: its units joined by single spaces.  Idempotent.
    pub fn text(&self, text: &str) -> String {
        self.units(text).join(" ")
    }

    /// The ids of a text given as text or in the text form.
    pub fn token_ids(&self, text: &str) -> Vec<usize> {
        self.units(text).iter().map(|u| self.ids[&unit_token(u)]).collect()
    }

    /// The bytes of a text's units, the leading space kept: what a walk said, byte by byte.
    pub fn bytes_of(&self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for u in self.units(text) {
            out.extend(text_bytes(&unit_token(&u)).unwrap_or_default());
        }
        out
    }

    /// What a text of the text form says: `"The cat sat ⁀."` -> `"The cat sat."`.
    pub fn spell(&self, text: &str) -> String {
        self.decode(&self.token_ids(text)).unwrap_or_else(|_| text.to_string())
    }
}

fn unit_token(unit: &str) -> String {
    match unit.strip_prefix(GLUE) {
        Some(rest) => rest.to_string(),
        None => format!("Ġ{unit}"),
    }
}

static DEFAULT: OnceLock<Result<BpeTokenizer, String>> = OnceLock::new();

/// The tokenizer the token unit reads through: the merges file
/// `RADIXNET_TOKENIZER` names, else the bundled merges - read once and kept.  A
/// model's tokens mean nothing without the merges that made them: train and
/// predict with the same file.
pub fn default_tokenizer() -> Result<&'static BpeTokenizer, String> {
    DEFAULT
        .get_or_init(|| match std::env::var(ENV) {
            Ok(path) if !path.is_empty() => BpeTokenizer::load(&path)
                .map_err(|e| format!("the token unit cannot read its merges ({ENV}={path}): {e}")),
            _ => BpeTokenizer::bundled()
                .map_err(|e| format!("the token unit cannot read its merges (the bundled merges): {e}")),
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// The units of a text through the default tokenizer.
pub fn units(text: &str) -> Vec<String> {
    default_tokenizer()
        .expect("Encoding::validate refused the encoding before any text could reach here")
        .units(text)
}

/// The text form of a text through the default tokenizer.
pub fn text(text: &str) -> String {
    units(text).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{parse, Json};

    fn fixture() -> Json {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/tokens_fixture.json");
        parse(&std::fs::read_to_string(path).expect("the fixture")).expect("the fixture is JSON")
    }

    fn strings(j: &Json) -> Vec<String> {
        j.to_strings()
    }

    fn ids(j: &Json) -> Vec<usize> {
        j.to_i64s().into_iter().map(|i| i as usize).collect()
    }

    fn check_side(name: &str, tok: &BpeTokenizer, side: &Json) {
        assert_eq!(
            tok.vocab_size() as i64,
            side.at("vocab_size").as_i64().unwrap(),
            "{name}"
        );
        assert_eq!(tok.merges.len() as i64, side.at("merges").as_i64().unwrap(), "{name}");
        for c in side.at("cases").as_array() {
            let text = c.at("text").as_str().unwrap();
            assert_eq!(
                pretokenize(&format!(" {text}")),
                strings(c.at("pretokens")),
                "{name} {text:?}"
            );
            assert_eq!(tok.tokens(text), strings(c.at("tokens")), "{name} {text:?}");
            assert_eq!(tok.encode(text, false), ids(c.at("ids")), "{name} {text:?}");
            assert_eq!(tok.encode(text, true), ids(c.at("special_ids")), "{name} {text:?}");
            let units = strings(c.at("units"));
            assert_eq!(tok.units(text), units, "{name} {text:?}");
            let form = units.join(" ");
            assert_eq!(tok.units(&form), strings(c.at("reread")), "{name} {form:?}");
            assert_eq!(tok.spell(&form), c.at("spelled").as_str().unwrap(), "{name} {form:?}");
            assert_eq!(tok.decode(&ids(c.at("ids"))).unwrap(), text, "{name}");
        }
        for r in side.at("reads").as_array() {
            let text = r.at("text").as_str().unwrap();
            assert_eq!(tok.units(text), strings(r.at("units")), "{name} read {text:?}");
            assert_eq!(
                tok.spell(text),
                r.at("spelled").as_str().unwrap(),
                "{name} read {text:?}"
            );
        }
        for b in side.at("broken").as_array() {
            assert_eq!(
                tok.decode(&ids(b.at("ids"))).unwrap(),
                b.at("text").as_str().unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn the_fixture_python_wrote() {
        let f = fixture();
        assert_eq!(f.at("glue").as_str().unwrap(), GLUE.to_string());
        for pair in f.at("classes").as_array() {
            let pair = pair.as_array();
            let c = char::from_u32(pair[0].as_i64().unwrap() as u32).unwrap();
            assert_eq!(char_class(c).letter(), pair[1].as_str().unwrap(), "U+{:04X}", c as u32);
        }
        check_side("bundled", &BpeTokenizer::bundled().unwrap(), f.at("bundled"));
        let small = f.at("small");
        check_side(
            "small",
            &BpeTokenizer::parse(small.at("merges_text").as_str().unwrap()).unwrap(),
            small,
        );
    }

    #[test]
    fn the_byte_alphabet() {
        assert_eq!(byte_char(b' '), 'Ġ');
        assert_eq!(byte_char(b'\n'), 'Ċ');
        assert_eq!(byte_char(b'a'), 'a');
        let all: Vec<u8> = (0..=255).collect();
        assert_eq!(text_bytes(&byte_text(&all)).unwrap(), all);
        assert_eq!(text_bytes("x\u{2040}"), None);
    }

    #[test]
    fn an_unfinished_character_waits() {
        assert_eq!(
            decode_utf8(b"a\xe2\x82", false),
            ("a".to_string(), b"\xe2\x82".to_vec())
        );
        assert_eq!(decode_utf8(b"\xe2\x82\xac", false), ("€".to_string(), Vec::new()));
        assert_eq!(decode_utf8(b"a\xe2\x82", true).0, "a\u{FFFD}");
    }

    #[test]
    fn bad_merges_are_refused() {
        for text in ["a", "a b c", "a  b", "Ġth e\n", "a b\na b\n"] {
            assert!(BpeTokenizer::parse(text).is_err(), "{text:?}");
        }
        let tok = BpeTokenizer::parse("#version: 0.2\nĠ t\nĠt h\nĠth e\n").unwrap();
        assert_eq!(tok.tokens("the"), vec!["Ġthe"]);
    }
}
