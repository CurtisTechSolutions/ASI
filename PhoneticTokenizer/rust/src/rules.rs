//! Letter-to-sound rules (the port of rules.py): the same rule table as the
//! Python package, read from the same data file, and the same context engine.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::phones::{base, is_vowel, stress_of};

/// The rule table: the very file the Python package reads.
pub const RULES_TEXT: &str = include_str!("../../phonetok/data/rules.lts");

/// One rewrite rule, `LEFT[MATCH]RIGHT=PHONES`.
#[derive(Clone, Debug)]
pub struct Rule {
    pub left: String,
    pub matched: String,
    pub right: String,
    pub phones: Vec<String>,
    pub source: String,
}

pub fn parse_rule(text: &str) -> Result<Rule, String> {
    let open = text
        .find('[')
        .ok_or_else(|| format!("not a rule: {text:?}"))?;
    let close = text[open..]
        .find(']')
        .ok_or_else(|| format!("not a rule: {text:?}"))?
        + open;
    let eq = text[close..]
        .find('=')
        .ok_or_else(|| format!("not a rule: {text:?}"))?
        + close;
    let matched = &text[open + 1..close];
    if matched.is_empty() {
        return Err(format!("not a rule: {text:?}"));
    }
    Ok(Rule {
        left: text[..open].to_string(),
        matched: matched.to_string(),
        right: text[close + 1..eq].to_string(),
        phones: text[eq + 1..]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect(),
        source: text.to_string(),
    })
}

/// The rules grouped by the first letter of their match, in the order they are tried.
pub fn rules() -> &'static HashMap<u8, Vec<Rule>> {
    static RULES: OnceLock<HashMap<u8, Vec<Rule>>> = OnceLock::new();
    RULES.get_or_init(|| {
        let mut table: HashMap<u8, Vec<Rule>> = HashMap::new();
        for line in RULES_TEXT.lines() {
            if line.trim().is_empty() || line.starts_with(';') {
                continue;
            }
            let rule = parse_rule(line).expect("a rule line");
            table
                .entry(rule.matched.as_bytes()[0])
                .or_default()
                .push(rule);
        }
        table
    })
}

pub fn rule_count() -> usize {
    rules().values().map(|v| v.len()).sum()
}

const VOWEL_LETTERS: &[u8] = b"AEIOUY";
const VOICED_LETTERS: &[u8] = b"BDGJLMNRVWZ";
const FRONT_LETTERS: &[u8] = b"EIY";
const SIBILANT_ONE: &[u8] = b"SCGZXJ";
const SIBILANT_TWO: [&[u8]; 2] = [b"CH", b"SH"];
const U_PLAIN_ONE: &[u8] = b"TSRDLZNJ";
const U_PLAIN_TWO: [&[u8]; 3] = [b"TH", b"CH", b"SH"];
const SUFFIXES: [&[u8]; 6] = [b"ING", b"ELY", b"ER", b"ES", b"ED", b"E"];

fn is_vowel_letter(ch: u8) -> bool {
    VOWEL_LETTERS.contains(&ch)
}

fn is_consonant_letter(ch: u8) -> bool {
    ch.is_ascii_uppercase() && !is_vowel_letter(ch)
}

fn starts_with_any(word: &[u8], pos: usize, options: &[&[u8]]) -> Option<usize> {
    options
        .iter()
        .find(|o| word[pos..].starts_with(o))
        .map(|o| o.len())
}

fn ends_with_any(word: &[u8], pos: usize, options: &[&[u8]]) -> Option<usize> {
    options
        .iter()
        .find(|o| pos >= o.len() && word[pos - o.len()..pos] == ***o)
        .map(|o| o.len())
}

fn match_right(word: &[u8], pos: usize, pattern: &[u8], k: usize) -> bool {
    if k == pattern.len() {
        return true;
    }
    let sym = pattern[k];
    match sym {
        b'#' => {
            let mut j = pos;
            while j < word.len() && is_vowel_letter(word[j]) {
                j += 1;
                if match_right(word, j, pattern, k + 1) {
                    return true;
                }
            }
            return false;
        }
        b':' => {
            let mut j = pos;
            loop {
                if match_right(word, j, pattern, k + 1) {
                    return true;
                }
                if j < word.len() && is_consonant_letter(word[j]) {
                    j += 1;
                } else {
                    return false;
                }
            }
        }
        _ => {}
    }
    if pos >= word.len() {
        return false;
    }
    let ch = word[pos];
    match sym {
        b'^' => is_consonant_letter(ch) && match_right(word, pos + 1, pattern, k + 1),
        b'.' => VOICED_LETTERS.contains(&ch) && match_right(word, pos + 1, pattern, k + 1),
        b'+' => FRONT_LETTERS.contains(&ch) && match_right(word, pos + 1, pattern, k + 1),
        b'&' => {
            if let Some(n) = starts_with_any(word, pos, &SIBILANT_TWO) {
                if match_right(word, pos + n, pattern, k + 1) {
                    return true;
                }
            }
            SIBILANT_ONE.contains(&ch) && match_right(word, pos + 1, pattern, k + 1)
        }
        b'@' => {
            if let Some(n) = starts_with_any(word, pos, &U_PLAIN_TWO) {
                if match_right(word, pos + n, pattern, k + 1) {
                    return true;
                }
            }
            U_PLAIN_ONE.contains(&ch) && match_right(word, pos + 1, pattern, k + 1)
        }
        b'%' => {
            for suffix in SUFFIXES {
                if word[pos..].starts_with(suffix) {
                    let rest = &word[pos + suffix.len()..];
                    if (rest == b" " || rest == b"S ")
                        && match_right(word, word.len(), pattern, k + 1)
                    {
                        return true;
                    }
                }
            }
            false
        }
        _ => ch == sym && match_right(word, pos + 1, pattern, k + 1),
    }
}

fn match_left(word: &[u8], pos: usize, pattern: &[u8], k: usize) -> bool {
    if k == 0 {
        return true;
    }
    let sym = pattern[k - 1];
    match sym {
        b'#' => {
            let mut j = pos;
            while j > 0 && is_vowel_letter(word[j - 1]) {
                j -= 1;
                if match_left(word, j, pattern, k - 1) {
                    return true;
                }
            }
            return false;
        }
        b':' => {
            let mut j = pos;
            loop {
                if match_left(word, j, pattern, k - 1) {
                    return true;
                }
                if j > 0 && is_consonant_letter(word[j - 1]) {
                    j -= 1;
                } else {
                    return false;
                }
            }
        }
        _ => {}
    }
    if pos == 0 {
        return false;
    }
    let ch = word[pos - 1];
    match sym {
        b'^' => is_consonant_letter(ch) && match_left(word, pos - 1, pattern, k - 1),
        b'.' => VOICED_LETTERS.contains(&ch) && match_left(word, pos - 1, pattern, k - 1),
        b'+' => FRONT_LETTERS.contains(&ch) && match_left(word, pos - 1, pattern, k - 1),
        b'&' => {
            if let Some(n) = ends_with_any(word, pos, &SIBILANT_TWO) {
                if match_left(word, pos - n, pattern, k - 1) {
                    return true;
                }
            }
            SIBILANT_ONE.contains(&ch) && match_left(word, pos - 1, pattern, k - 1)
        }
        b'@' => {
            if let Some(n) = ends_with_any(word, pos, &U_PLAIN_TWO) {
                if match_left(word, pos - n, pattern, k - 1) {
                    return true;
                }
            }
            U_PLAIN_ONE.contains(&ch) && match_left(word, pos - 1, pattern, k - 1)
        }
        _ => ch == sym && match_left(word, pos - 1, pattern, k - 1),
    }
}

/// Sounds out a word with the rules alone: phones without stress (schwas excepted).
pub fn apply_rules(word: &str, mut trace: Option<&mut Vec<Rule>>) -> Vec<String> {
    let text = format!(" {} ", word.to_uppercase());
    let bytes = text.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut pos = 1;
    let end = bytes.len() - 1;
    while pos < end {
        let ch = bytes[pos];
        let Some(candidates) = rules().get(&ch) else {
            pos += 1;
            continue;
        };
        let mut fired = false;
        for rule in candidates {
            let m = rule.matched.as_bytes();
            if !bytes[pos..].starts_with(m) {
                continue;
            }
            if !match_right(bytes, pos + m.len(), rule.right.as_bytes(), 0) {
                continue;
            }
            if !match_left(bytes, pos, rule.left.as_bytes(), rule.left.len()) {
                continue;
            }
            out.extend(rule.phones.iter().cloned());
            if let Some(t) = trace.as_deref_mut() {
                t.push(rule.clone());
            }
            pos += m.len();
            fired = true;
            break;
        }
        if !fired {
            pos += 1;
        }
    }
    out
}

// -- stress ------------------------------------------------------------------------

const PREFIXES: [&str; 36] = [
    "a", "ab", "ac", "ad", "af", "ag", "al", "ap", "ar", "as", "at", "be", "com", "con", "cor",
    "de", "dis", "em", "en", "ex", "for", "im", "in", "inter", "ir", "mis", "ob", "per", "pre",
    "pro", "re", "sub", "sup", "sur", "trans", "un",
];

const PENULT_SUFFIXES: [(&str, usize); 35] = [
    ("tion", 1),
    ("sion", 1),
    ("cian", 1),
    ("ic", 1),
    ("ics", 1),
    ("ical", 2),
    ("ity", 2),
    ("ities", 2),
    ("ify", 2),
    ("ial", 1),
    ("ual", 1),
    ("eous", 1),
    ("ious", 1),
    ("ian", 1),
    ("ience", 1),
    ("ient", 1),
    ("ish", 1),
    ("ive", 1),
    ("ia", 1),
    ("ium", 1),
    ("ular", 2),
    ("itis", 1),
    ("ology", 2),
    ("ogy", 2),
    ("ographer", 2),
    ("ography", 2),
    ("ometer", 2),
    ("osis", 1),
    ("atic", 1),
    ("ency", 2),
    ("ancy", 2),
    ("ent", 1),
    ("ant", 1),
    ("ator", 3),
    ("ators", 3),
];

const FINAL_STRESS_SUFFIXES: [&str; 11] = [
    "ee", "eer", "ese", "ette", "esque", "ique", "oon", "ain", "een", "ine", "aire",
];
const TENSE: [&str; 8] = ["EY", "IY", "AY", "OW", "UW", "AW", "OY", "AO"];

fn reduced(b: &str) -> &str {
    match b {
        "AE" | "EH" | "AA" | "UH" | "AO" => "AH",
        other => other,
    }
}

/// Gives the rules' output for `word` its stress digits (see rules.py's assign_stress).
pub fn assign_stress(word: &str, phones: &[String]) -> Vec<String> {
    let vowels: Vec<usize> = (0..phones.len())
        .filter(|&i| is_vowel(&phones[i]))
        .collect();
    let mut out: Vec<String> = phones.to_vec();
    if vowels.is_empty() {
        return out;
    }
    let free: Vec<usize> = vowels
        .iter()
        .copied()
        .filter(|&i| stress_of(&phones[i]).is_none())
        .collect();
    if free.is_empty() {
        if !vowels.iter().any(|&i| stress_of(&phones[i]) == Some(1)) {
            let i = vowels[0];
            out[i] = format!("{}1", base(&phones[i]));
        }
        return out;
    }
    let primary = if free.len() > 1 {
        pick_primary(&word.to_lowercase(), &vowels, &free)
    } else {
        free[0]
    };
    let primary_syllable = vowels.iter().position(|&v| v == primary).unwrap_or(0);
    for &i in &free {
        let b = base(&phones[i]);
        out[i] = if i == primary {
            format!("{b}1")
        } else if (i == vowels[0] && primary_syllable >= 2) || (i > primary && TENSE.contains(&b)) {
            format!("{b}2") // a first syllable far from the stress, or a tense vowel after it, keeps a secondary one
        } else {
            format!("{}0", reduced(b))
        };
    }
    out
}

fn pick_primary(word: &str, vowels: &[usize], free: &[usize]) -> usize {
    let n_syl = vowels.len();
    for suffix in FINAL_STRESS_SUFFIXES {
        if word.ends_with(suffix) && n_syl >= 2 {
            return free[free.len() - 1];
        }
    }
    for (suffix, back) in PENULT_SUFFIXES {
        if word.ends_with(suffix) {
            if vowels.len() > back {
                let idx = vowels.len() - 1 - back;
                let candidate = vowels[idx];
                if free.contains(&candidate) {
                    return candidate;
                }
                if let Some(&nearer) = free.iter().rfind(|&&i| i <= candidate) {
                    return nearer;
                }
            }
            break;
        }
    }
    if n_syl >= 2 {
        let mut sorted: Vec<&str> = PREFIXES.to_vec();
        // stable, longest first
        for i in 1..sorted.len() {
            let mut j = i;
            while j > 0 && sorted[j].len() > sorted[j - 1].len() {
                sorted.swap(j, j - 1);
                j -= 1;
            }
        }
        for prefix in sorted {
            if word.starts_with(prefix) && word.len() > prefix.len() + 2 {
                let later: Vec<usize> = free.iter().copied().filter(|&i| i > vowels[0]).collect();
                if !later.is_empty() && n_syl <= 3 {
                    return later[0];
                }
                break;
            }
        }
    }
    if n_syl >= 4 {
        let candidate = vowels[vowels.len() - 3];
        if free.contains(&candidate) {
            return candidate;
        }
    }
    free[0]
}

/// Phones with stress for a spelled word, from the rules alone.
pub fn letter_to_sound(word: &str) -> Vec<String> {
    assign_stress(word, &apply_rules(word, None))
}
