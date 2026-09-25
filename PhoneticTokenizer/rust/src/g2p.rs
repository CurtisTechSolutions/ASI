//! Grapheme to phoneme (the port of g2p.py): the transcriber, its memory, and the respelling.

use std::collections::HashMap;

use crate::lexicon::Lexicon;
use crate::numbers::{number_words, split_alphanumeric};
use crate::phones::{
    base, is_consonant, is_sibilant, is_stressed, is_vowel, past_suffix, plural_suffix, stress_of,
    strings, strip_stress, with_stress,
};
use crate::rules::letter_to_sound;

const SUFFIXES: [(&str, &[&str]); 11] = [
    ("ness", &["N", "AH0", "S"]),
    ("ment", &["M", "AH0", "N", "T"]),
    ("less", &["L", "AH0", "S"]),
    ("ful", &["F", "AH0", "L"]),
    ("able", &["AH0", "B", "AH0", "L"]),
    ("ish", &["IH0", "SH"]),
    ("ly", &["L", "IY0"]),
    ("er", &["ER0"]),
    ("est", &["IH0", "S", "T"]),
    ("ing", &["IH0", "NG"]),
    ("y", &["IY0"]),
];

const PREFIXES: [(&str, &[&str]); 19] = [
    ("under", &["AH2", "N", "D", "ER0"]),
    ("inter", &["IH2", "N", "T", "ER0"]),
    ("super", &["S", "UW2", "P", "ER0"]),
    ("multi", &["M", "AH2", "L", "T", "IY0"]),
    ("semi", &["S", "EH2", "M", "IY0"]),
    ("over", &["OW2", "V", "ER0"]),
    ("anti", &["AE2", "N", "T", "IY0"]),
    ("auto", &["AO2", "T", "OW0"]),
    ("micro", &["M", "AY2", "K", "R", "OW0"]),
    ("non", &["N", "AA2", "N"]),
    ("out", &["AW2", "T"]),
    ("pre", &["P", "R", "IY2"]),
    ("dis", &["D", "IH0", "S"]),
    ("mis", &["M", "IH0", "S"]),
    ("sub", &["S", "AH0", "B"]),
    ("un", &["AH0", "N"]),
    ("re", &["R", "IY0"]),
    ("de", &["D", "IY0"]),
    ("co", &["K", "OW0"]),
];

const CONTRACTIONS: [(&str, &[&str]); 6] = [
    ("n't", &["N", "T"]),
    ("'ll", &["L"]),
    ("'re", &["R"]),
    ("'ve", &["V"]),
    ("'d", &["D"]),
    ("'m", &["M"]),
];

/// Words to sounds, through the lexicon, the morphology and the rules; and sounds back to words.
pub struct Transcriber {
    pub lexicon: Lexicon,
    pub use_rules: bool,
    pub remember: bool,
    /// Words the lexicon did not hold, with the sounds they were given.
    pub memory: HashMap<String, Vec<String>>,
    memory_order: Vec<String>,
    /// How often each spelled word has been transcribed: the tie-breaker among homophones.
    pub counts: HashMap<String, usize>,
}

fn trim_quotes(w: &str) -> &str {
    w.trim_matches(|c| {
        matches!(
            c,
            '\'' | '"' | '`' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}'
        )
    })
}

fn is_joiner(c: char) -> bool {
    matches!(c, '-' | '/' | '_' | '.')
}

fn is_upper_word(s: &str) -> bool {
    let mut has = false;
    for c in s.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            has = true;
        }
    }
    has
}

impl Transcriber {
    pub fn new(lexicon: Lexicon) -> Transcriber {
        Transcriber {
            lexicon,
            use_rules: true,
            remember: true,
            memory: HashMap::new(),
            memory_order: Vec::new(),
            counts: HashMap::new(),
        }
    }

    /// The sounds of one spelled word (no spaces); empty if it has none.
    pub fn word(&mut self, word: &str) -> Vec<String> {
        self.explain(word).0
    }

    /// The sounds of a word and where they came from: lexicon, memory, number, joined, morphology,
    /// letters, rules or none.
    pub fn explain(&mut self, word: &str) -> (Vec<String>, &'static str) {
        let raw = word;
        let w = trim_quotes(word).to_lowercase();
        if w.is_empty() {
            return (Vec::new(), "none");
        }
        *self.counts.entry(w.clone()).or_insert(0) += 1;
        if let Some(found) = self.lexicon.lookup(&w) {
            return (found, "lexicon");
        }
        if let Some(m) = self.memory.get(&w) {
            return (m.clone(), "memory");
        }
        let (phones, how) = self.derive(&w, raw);
        if !phones.is_empty() && self.remember && how != "none" {
            if !self.memory.contains_key(&w) {
                self.memory_order.push(w.clone());
            }
            self.memory.insert(w, phones.clone());
        }
        (phones, how)
    }

    fn derive(&mut self, w: &str, raw: &str) -> (Vec<String>, &'static str) {
        if w.chars().any(|c| c.is_ascii_digit()) {
            if let Some(words) = number_words(w) {
                let parts: Vec<String> = words.split(' ').map(|s| s.to_string()).collect();
                return (self.join(&parts), "number");
            }
            return (self.join(&split_alphanumeric(w)), "number");
        }
        let mut w = w.to_string();
        if w.chars().any(is_joiner) {
            let parts: Vec<String> = w
                .split(is_joiner)
                .filter(|p| !p.is_empty())
                .map(|p| p.to_string())
                .collect();
            if parts.len() > 1 {
                return (self.join(&parts), "joined");
            }
            match parts.first() {
                Some(p) => w = p.clone(),
                None => return (Vec::new(), "none"),
            }
        }
        if let Some(phones) = self.contraction(&w) {
            return (phones, "morphology");
        }
        if let Some(phones) = self.morphology(&w) {
            return (phones, "morphology");
        }
        if !w.chars().any(|c| c.is_ascii_lowercase()) {
            return (Vec::new(), "none");
        }
        let raw_len = raw.chars().count();
        if !w.chars().any(|c| "aeiouy".contains(c))
            || (is_upper_word(raw) && (2..=5).contains(&raw_len))
        {
            return (self.letters(&w), "letters");
        }
        if !self.use_rules {
            return (Vec::new(), "none");
        }
        let cleaned: String = w
            .chars()
            .filter(|c| c.is_ascii_lowercase() || *c == '\'')
            .collect();
        (letter_to_sound(&cleaned), "rules")
    }

    fn join(&mut self, parts: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for p in parts {
            out.extend(self.explain(p).0);
        }
        out
    }

    fn letters(&self, w: &str) -> Vec<String> {
        let mut out = Vec::new();
        for c in w.chars() {
            if !c.is_alphabetic() {
                continue;
            }
            let found = self.lexicon.pronunciations(&c.to_string());
            let named = found
                .iter()
                .find(|p| p.iter().any(|x| stress_of(x) == Some(1)));
            match (named, found.first()) {
                (Some(p), _) => out.extend(p.iter().cloned()),
                (None, Some(p)) => out.extend(p.iter().cloned()),
                (None, None) => out.extend(letter_to_sound(&c.to_string())),
            }
        }
        out
    }

    fn stem(&self, stem: &str) -> Option<Vec<String>> {
        if let Some(found) = self.lexicon.lookup(stem) {
            return Some(found);
        }
        self.memory.get(stem).cloned()
    }

    fn contraction(&self, w: &str) -> Option<Vec<String>> {
        if !w.contains('\'') {
            return None;
        }
        for ending in ["'s", "s'"] {
            if w.ends_with(ending) && w.len() > 2 {
                let mut stem = w[..w.len() - 2].to_string();
                if ending == "s'" {
                    stem.push('s');
                }
                if let Some(mut phones) = self.stem(&stem) {
                    if ending == "'s" {
                        phones.extend(plural_suffix(&phones.clone()));
                    }
                    return Some(phones);
                }
            }
        }
        for (ending, sounds) in CONTRACTIONS {
            if w.ends_with(ending) && w.len() > ending.len() {
                let Some(mut phones) = self.stem(&w[..w.len() - ending.len()]) else {
                    continue;
                };
                let last = phones
                    .last()
                    .map(|p| base(p).to_string())
                    .unwrap_or_default();
                if ending == "n't" && last == "N" {
                    phones.push("T".to_string());
                    return Some(phones);
                }
                if !last.is_empty() && is_vowel(&last) && ending != "n't" {
                    phones.extend(strings(sounds));
                    return Some(phones);
                }
                if ending == "'d" && (last == "T" || last == "D") {
                    phones.extend(strings(&["AH0", "D"]));
                    return Some(phones);
                }
                if ending == "'d" && !last.is_empty() && !is_vowel(&last) {
                    phones.push("D".to_string());
                    return Some(phones);
                }
                phones.push("AH0".to_string());
                phones.extend(strings(sounds));
                return Some(phones);
            }
        }
        None
    }

    fn stems(stem: &str, suffix_starts_with_vowel: bool) -> Vec<String> {
        let mut candidates = vec![stem.to_string()];
        if suffix_starts_with_vowel {
            candidates.push(format!("{stem}e"));
            let b = stem.as_bytes();
            if b.len() >= 3
                && b[b.len() - 1] == b[b.len() - 2]
                && b"bdfgklmnprstz".contains(&b[b.len() - 1])
            {
                candidates.push(stem[..stem.len() - 1].to_string());
            }
            if let Some(short) = stem.strip_suffix('i') {
                candidates.push(format!("{short}y"));
            }
        }
        candidates
    }

    fn morphology(&self, w: &str) -> Option<Vec<String>> {
        if w.len() < 4 {
            return None;
        }
        if w.ends_with("ies") && w.len() > 4 {
            if let Some(mut s) = self.stem(&format!("{}y", &w[..w.len() - 3])) {
                s.extend(plural_suffix(&s.clone()));
                return Some(s);
            }
        }
        if w.ends_with("es") && w.len() > 3 {
            if let Some(mut s) = self.stem(&w[..w.len() - 2]) {
                if s.last().map(|p| is_sibilant(base(p))).unwrap_or(false) {
                    s.extend(strings(&["IH0", "Z"]));
                } else {
                    s.extend(plural_suffix(&s.clone()));
                }
                return Some(s);
            }
        }
        if w.ends_with('s') && !w.ends_with("ss") {
            if let Some(mut s) = self.stem(&w[..w.len() - 1]) {
                s.extend(plural_suffix(&s.clone()));
                return Some(s);
            }
        }
        if w.ends_with("ed") && w.len() > 4 {
            for candidate in Self::stems(&w[..w.len() - 2], true) {
                if let Some(mut s) = self.stem(&candidate) {
                    s.extend(past_suffix(&s.clone()));
                    return Some(s);
                }
            }
        }
        for (suffix, sounds) in SUFFIXES {
            if w.ends_with(suffix) && w.len() - suffix.len() >= 3 {
                let vowel_first = "aeiouy".contains(suffix.chars().next().unwrap());
                for candidate in Self::stems(&w[..w.len() - suffix.len()], vowel_first) {
                    if let Some(mut s) = self.stem(&candidate) {
                        if suffix == "ly" && s.last().map(|p| p == "L").unwrap_or(false) {
                            s.extend(strings(&sounds[1..])); // real-ly, full-y: one L, as the dictionary says
                        } else {
                            s.extend(strings(sounds));
                        }
                        return Some(s);
                    }
                }
            }
        }
        for (prefix, sounds) in PREFIXES {
            if w.starts_with(prefix) && w.len() - prefix.len() >= 3 {
                if let Some(s) = self.stem(&w[prefix.len()..]) {
                    let mut out = strings(sounds);
                    out.extend(s);
                    return Some(out);
                }
            }
        }
        // a compound of two known words: the cut nearest the middle wins
        let mut best: Option<(f64, Vec<String>)> = None;
        let half = w.len() as f64 / 2.0;
        for cut in 3..w.len().saturating_sub(2) {
            if !w.is_char_boundary(cut) {
                continue;
            }
            let (Some(left), Some(right)) = (self.stem(&w[..cut]), self.stem(&w[cut..])) else {
                continue;
            };
            let demoted: Vec<String> = right
                .iter()
                .map(|p| {
                    if stress_of(p) == Some(1) {
                        with_stress(p, 2)
                    } else {
                        p.clone()
                    }
                })
                .collect();
            let balance = (cut as f64 - half).abs();
            if best.as_ref().map(|(b, _)| balance < *b).unwrap_or(true) {
                let mut phones = left;
                phones.extend(demoted);
                best = Some((balance, phones));
            }
        }
        best.map(|(_, p)| p)
    }

    /// A word that sounds like `phones`: a remembered one, a lexicon one, or a respelling.
    pub fn spell(&mut self, phones: &[String]) -> String {
        if phones.is_empty() {
            return String::new();
        }
        let key = phones.join(" ");
        let mut candidates: Vec<String> = self
            .memory_order
            .iter()
            .filter(|w| self.memory[*w].join(" ") == key)
            .cloned()
            .collect();
        if candidates.is_empty() {
            candidates = self.lexicon.spellings(phones, true);
        }
        if candidates.is_empty() {
            let loose = strip_stress(phones).join(" ");
            candidates = self
                .memory_order
                .iter()
                .filter(|w| strip_stress(&self.memory[*w]).join(" ") == loose)
                .cloned()
                .collect();
            if candidates.is_empty() {
                candidates = self.lexicon.spellings(phones, false);
            }
        }
        if candidates.is_empty() {
            return respell(phones);
        }
        let counts = &self.counts;
        let mut best = candidates[0].clone();
        for w in &candidates[1..] {
            let (ca, cb) = (
                counts.get(w).copied().unwrap_or(0),
                counts.get(&best).copied().unwrap_or(0),
            );
            let (la, lb) = (w.chars().count(), best.chars().count());
            if (ca, std::cmp::Reverse(la), std::cmp::Reverse(w.clone()))
                > (cb, std::cmp::Reverse(lb), std::cmp::Reverse(best.clone()))
            {
                best = w.clone();
            }
        }
        best
    }
}

// -- respelling --------------------------------------------------------------------

fn consonant_spelling(c: &str) -> &'static str {
    match c {
        "B" => "b",
        "CH" => "ch",
        "D" => "d",
        "DH" => "th",
        "F" => "f",
        "G" => "g",
        "HH" => "h",
        "JH" => "j",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ng",
        "P" => "p",
        "R" => "r",
        "S" => "s",
        "SH" => "sh",
        "T" => "t",
        "TH" => "th",
        "V" => "v",
        "W" => "w",
        "Y" => "y",
        "Z" => "z",
        "ZH" => "zh",
        _ => "",
    }
}

fn vowel_spelling(v: &str) -> &'static str {
    match v {
        "AA" => "o",
        "AE" => "a",
        "AH" => "u",
        "AO" => "aw",
        "AW" => "ow",
        "AY" => "i",
        "EH" => "e",
        "ER" => "er",
        "EY" => "ay",
        "IH" => "i",
        "IY" => "ee",
        "OW" => "o",
        "OY" => "oy",
        "UH" => "oo",
        "UW" => "oo",
        _ => "",
    }
}

fn tense_magic(v: &str) -> Option<&'static str> {
    match v {
        "EY" => Some("a"),
        "AY" => Some("i"),
        "OW" => Some("o"),
        "UW" => Some("u"),
        _ => None,
    }
}

const FRONT: [&str; 5] = ["EH", "IY", "IH", "EY", "AY"];
const MAGIC_OK: &str = "bdfgklmnprstvz";

/// An invented spelling that reads back as `phones`: K AE1 T -> cat, M EY1 K -> make.
pub fn respell(phones: &[String]) -> String {
    let n = phones.len();
    let mut out: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let p = phones[i].as_str();
        let b = base(p);
        let nxt = if i + 1 < n {
            Some(phones[i + 1].as_str())
        } else {
            None
        };
        let nxt_b = nxt.map(base).unwrap_or("");
        let after = if i + 2 < n { base(&phones[i + 2]) } else { "" };
        let nxt_is_vowel = is_vowel(nxt_b) && !nxt_b.is_empty();
        let after_is_vowel = !after.is_empty() && is_vowel(after);
        let prev = if i > 0 { base(&phones[i - 1]) } else { "" };
        if is_consonant(b) {
            if b == "K" {
                let soft =
                    (nxt_is_vowel && !FRONT.contains(&nxt_b)) || matches!(nxt_b, "L" | "R" | "Y");
                out.push((if soft { "c" } else { "k" }).to_string());
                if nxt.is_none()
                    && i > 0
                    && matches!(prev, "AE" | "EH" | "IH" | "AH")
                    && stress_of(&phones[i - 1]) != Some(0)
                {
                    *out.last_mut().unwrap() = "ck".to_string();
                }
            } else if b == "Y" && nxt_b == "UW" {
                out.push(String::new());
            } else {
                out.push(consonant_spelling(b).to_string());
            }
            if i > 0
                && matches!(prev, "AE" | "EH" | "IH" | "AH" | "UH")
                && is_stressed(&phones[i - 1])
                && nxt_is_vowel
                && matches!(
                    b,
                    "B" | "D" | "G" | "L" | "M" | "N" | "P" | "R" | "S" | "T" | "Z"
                )
            {
                let last = out.last().unwrap().clone();
                *out.last_mut().unwrap() = format!("{last}{last}");
            }
            continue;
        }
        let magic = is_consonant(nxt_b)
            && after.is_empty()
            && consonant_spelling(nxt_b).len() == 1
            && MAGIC_OK.contains(consonant_spelling(nxt_b));
        let spelling: String = if b == "UW" && i > 0 && prev == "Y" {
            (if magic { "u\0" } else { "u" }).to_string()
        } else if b == "IY" && nxt.is_none() && stress_of(p) == Some(0) && i > 0 {
            "y".to_string()
        } else if b == "AY" && nxt.is_none() {
            (if i > 0 { "y" } else { "i" }).to_string()
        } else if b == "EY" && nxt.is_none() {
            "ay".to_string()
        } else if b == "OW" && nxt.is_none() {
            "o".to_string()
        } else if tense_magic(b).is_some() && magic {
            format!("{}\0", tense_magic(b).unwrap())
        } else if tense_magic(b).is_some() && is_consonant(nxt_b) && after_is_vowel && b != "IY" {
            tense_magic(b).unwrap().to_string()
        } else if b == "AH" && stress_of(p) == Some(0) {
            (if nxt.is_none() || i == 0 { "a" } else { "u" }).to_string()
        } else {
            vowel_spelling(b).to_string()
        };
        out.push(spelling);
    }
    let mut text = out.concat();
    if let Some(i) = text.find('\0') {
        let rest: String = text[i + 1..].to_string();
        let mut chars = rest.chars();
        let first: String = chars.next().map(|c| c.to_string()).unwrap_or_default();
        let tail: String = chars.collect();
        text = format!("{}{}e{}", &text[..i], first, tail).replace('\0', "");
    }
    text
}
