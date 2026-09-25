//! The sound inventory (the port of phones.py): the 39 ARPAbet phonemes, the
//! stress digit a vowel carries, the fixed alphabet of ids, the articulatory
//! features, the sonority scale and the IPA.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The 15 vowel phonemes, in ARPAbet order.
pub const VOWELS: [&str; 15] = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
];

/// The 24 consonant phonemes, in ARPAbet order.
pub const CONSONANTS: [&str; 24] = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T",
    "TH", "V", "W", "Y", "Z", "ZH",
];

pub const BOUNDARY: &str = "#";
pub const PAUSE_SHORT: &str = ",";
pub const PAUSE_FULL: &str = ".";
pub const PAUSE_QUESTION: &str = "?";
pub const PAUSES: [&str; 3] = [PAUSE_SHORT, PAUSE_FULL, PAUSE_QUESTION];
pub const PAD: &str = "<pad>";
pub const UNK: &str = "<unk>";
pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";
pub const SPECIALS: [&str; 4] = [PAD, UNK, BOS, EOS];

pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const BOS_ID: usize = 2;
pub const EOS_ID: usize = 3;
pub const BOUNDARY_ID: usize = 4;

/// The fixed alphabet in id order: the specials, the boundary, the pauses, then every ARPAbet
/// symbol as cmudict.symbols lists them: 92 symbols.
pub fn symbols() -> &'static Vec<String> {
    static SYMBOLS: OnceLock<Vec<String>> = OnceLock::new();
    SYMBOLS.get_or_init(|| {
        let mut out: Vec<String> = SPECIALS.iter().map(|s| s.to_string()).collect();
        out.push(BOUNDARY.to_string());
        out.extend(PAUSES.iter().map(|s| s.to_string()));
        for v in VOWELS {
            out.push(v.to_string());
            for d in ["0", "1", "2"] {
                out.push(format!("{v}{d}"));
            }
        }
        out.extend(CONSONANTS.iter().map(|s| s.to_string()));
        out
    })
}

/// Symbol -> id.
pub fn symbol_id(symbol: &str) -> Option<usize> {
    static IDS: OnceLock<HashMap<String, usize>> = OnceLock::new();
    IDS.get_or_init(|| {
        symbols()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect()
    })
    .get(symbol)
    .copied()
}

/// The phoneme without its stress digit.
pub fn base(phone: &str) -> &str {
    match phone.as_bytes().last() {
        Some(b'0'..=b'2') => &phone[..phone.len() - 1],
        _ => phone,
    }
}

/// The stress digit of a vowel, or `None` for a consonant or a bare vowel.
pub fn stress_of(phone: &str) -> Option<u8> {
    match phone.as_bytes().last() {
        Some(d @ b'0'..=b'2') => Some(d - b'0'),
        _ => None,
    }
}

pub fn is_vowel(phone: &str) -> bool {
    VOWELS.contains(&base(phone))
}

pub fn is_consonant(phone: &str) -> bool {
    CONSONANTS.contains(&phone)
}

/// Is this an ARPAbet symbol: a consonant, or a vowel with or without a stress digit?
pub fn is_phone(token: &str) -> bool {
    is_vowel(token) || is_consonant(token)
}

pub fn is_stressed(phone: &str) -> bool {
    matches!(stress_of(phone), Some(1) | Some(2))
}

pub fn strip_stress(phones: &[String]) -> Vec<String> {
    phones.iter().map(|p| base(p).to_string()).collect()
}

/// A vowel with its stress set; a consonant is left alone.
pub fn with_stress(phone: &str, stress: u8) -> String {
    let b = base(phone);
    if VOWELS.contains(&b) {
        format!("{b}{stress}")
    } else {
        phone.to_string()
    }
}

pub fn check_phones(phones: &[String]) -> Result<(), String> {
    for p in phones {
        if !is_phone(p) {
            return Err(format!("{p:?} is not an ARPAbet phone"));
        }
    }
    Ok(())
}

// -- classes and articulation ----------------------------------------------------

pub const STOPS: [&str; 6] = ["B", "D", "G", "K", "P", "T"];
pub const AFFRICATES: [&str; 2] = ["CH", "JH"];
pub const FRICATIVES: [&str; 8] = ["DH", "F", "S", "SH", "TH", "V", "Z", "ZH"];
pub const NASALS: [&str; 3] = ["M", "N", "NG"];
pub const LIQUIDS: [&str; 2] = ["L", "R"];
pub const GLIDES: [&str; 2] = ["W", "Y"];
pub const SIBILANTS: [&str; 6] = ["S", "Z", "SH", "ZH", "CH", "JH"];
pub const VOICED_CONSONANTS: [&str; 15] = [
    "B", "D", "DH", "G", "JH", "L", "M", "N", "NG", "R", "V", "W", "Y", "Z", "ZH",
];
const LABIAL: [&str; 6] = ["B", "P", "M", "F", "V", "W"];
const CORONAL: [&str; 13] = [
    "T", "D", "S", "Z", "N", "L", "R", "TH", "DH", "SH", "ZH", "CH", "JH",
];
const DORSAL: [&str; 5] = ["K", "G", "NG", "W", "Y"];
const STRIDENT: [&str; 8] = ["S", "Z", "SH", "ZH", "CH", "JH", "F", "V"];

pub fn is_sibilant(c: &str) -> bool {
    SIBILANTS.contains(&c)
}

pub fn is_voiced_consonant(c: &str) -> bool {
    VOICED_CONSONANTS.contains(&c)
}

pub fn is_sonorant(c: &str) -> bool {
    NASALS.contains(&c) || LIQUIDS.contains(&c) || GLIDES.contains(&c)
}

pub const MAX_SONORITY: u32 = 10;

/// Where a phone sits on the sonority scale, 0 (a voiceless stop) to 10 (a low vowel).
pub fn sonority(phone: &str) -> u32 {
    let b = base(phone);
    let voiced = VOICED_CONSONANTS.contains(&b);
    if STOPS.contains(&b) {
        return if voiced { 1 } else { 0 };
    }
    if AFFRICATES.contains(&b) {
        return if voiced { 2 } else { 1 };
    }
    if FRICATIVES.contains(&b) {
        return if voiced { 3 } else { 2 };
    }
    match b {
        "HH" => 2,
        "M" | "N" | "NG" => 4,
        "L" => 5,
        "R" => 6,
        "W" | "Y" => 7,
        "IY" | "IH" | "UW" | "UH" | "ER" => 8,
        "EY" | "EH" | "OW" | "AO" | "AH" | "OY" => 9,
        "AE" | "AA" | "AY" | "AW" => 10,
        _ => panic!("not a phone: {phone}"),
    }
}

/// The 22 dimensions of `features`, in order.
pub const FEATURE_NAMES: [&str; 22] = [
    "syllabic",
    "consonantal",
    "sonorant",
    "voice",
    "nasal",
    "continuant",
    "strident",
    "lateral",
    "rhotic",
    "labial",
    "coronal",
    "dorsal",
    "glottal",
    "high",
    "low",
    "back",
    "round",
    "tense",
    "glide_y",
    "glide_w",
    "stress",
    "sonority",
];
pub const FEATURE_DIM: usize = 22;

fn vowel_space(v: &str) -> (f64, f64, f64, f64, &'static str) {
    match v {
        "IY" => (1.0, 0.0, 0.0, 1.0, ""),
        "IH" => (0.8, 0.2, 0.0, 0.0, ""),
        "EY" => (0.6, 0.0, 0.0, 1.0, "Y"),
        "EH" => (0.5, 0.2, 0.0, 0.0, ""),
        "AE" => (0.2, 0.2, 0.0, 0.0, ""),
        "AA" => (0.0, 0.9, 0.0, 1.0, ""),
        "AO" => (0.3, 1.0, 1.0, 1.0, ""),
        "OW" => (0.6, 1.0, 1.0, 1.0, "W"),
        "UH" => (0.8, 0.9, 1.0, 0.0, ""),
        "UW" => (1.0, 1.0, 1.0, 1.0, ""),
        "AH" => (0.4, 0.6, 0.0, 0.0, ""),
        "ER" => (0.5, 0.5, 0.0, 1.0, ""),
        "AY" => (0.1, 0.5, 0.0, 1.0, "Y"),
        "AW" => (0.1, 0.5, 0.0, 1.0, "W"),
        "OY" => (0.3, 1.0, 1.0, 1.0, "Y"),
        _ => unreachable!(),
    }
}

fn b2f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// The articulatory feature vector of a phone, stress included.
pub fn features(phone: &str) -> Result<[f64; FEATURE_DIM], String> {
    let b = base(phone);
    if !is_vowel(b) && !is_consonant(b) {
        return Err(format!("{phone:?} is not an ARPAbet phone"));
    }
    let stress = match stress_of(phone) {
        Some(1) => 1.0,
        Some(2) => 0.5,
        _ => 0.0,
    };
    let son = sonority(b) as f64 / MAX_SONORITY as f64;
    if is_vowel(b) {
        let (height, back, round, tense, glide) = vowel_space(b);
        return Ok([
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            b2f(b == "ER"),
            0.0,
            0.0,
            0.0,
            0.0,
            height,
            1.0 - height,
            back,
            round,
            tense,
            b2f(glide == "Y"),
            b2f(glide == "W"),
            stress,
            son,
        ]);
    }
    let continuant =
        FRICATIVES.contains(&b) || b == "HH" || LIQUIDS.contains(&b) || GLIDES.contains(&b);
    Ok([
        0.0,
        b2f(!(GLIDES.contains(&b) || b == "HH")),
        b2f(is_sonorant(b)),
        b2f(VOICED_CONSONANTS.contains(&b)),
        b2f(NASALS.contains(&b)),
        b2f(continuant),
        b2f(STRIDENT.contains(&b)),
        b2f(b == "L"),
        b2f(b == "R"),
        b2f(LABIAL.contains(&b)),
        b2f(CORONAL.contains(&b)),
        b2f(DORSAL.contains(&b)),
        b2f(b == "HH"),
        b2f(b == "Y" || b == "W"),
        0.0,
        b2f(matches!(b, "W" | "K" | "G" | "NG")),
        b2f(b == "W"),
        0.0,
        0.0,
        0.0,
        0.0,
        son,
    ])
}

/// How differently two phones are made: the Euclidean distance of their feature vectors.
pub fn distance(a: &str, b: &str) -> f64 {
    let (fa, fb) = (features(a).expect("a phone"), features(b).expect("a phone"));
    fa.iter()
        .zip(fb.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

pub fn similarity(a: &str, b: &str) -> f64 {
    1.0 / (1.0 + distance(a, b))
}

// -- the IPA ---------------------------------------------------------------------

/// ARPAbet -> IPA, one entry per phoneme.
pub fn ipa(phoneme: &str) -> &'static str {
    match phoneme {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => "ʌ",
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "EH" => "ɛ",
        "ER" => "ɝ",
        "EY" => "eɪ",
        "IH" => "ɪ",
        "IY" => "i",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "UH" => "ʊ",
        "UW" => "u",
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        _ => "",
    }
}

/// Phones as one IPA string, a stress mark before a stressed vowel when `stress` is true.
pub fn to_ipa(phones: &[String], stress: bool) -> String {
    let mut out = String::new();
    for p in phones {
        let b = base(p);
        let s = stress_of(p);
        if is_vowel(b) {
            if stress && s == Some(1) {
                out.push('ˈ');
            } else if stress && s == Some(2) {
                out.push('ˌ');
            }
            match (b, s) {
                ("AH", Some(0)) => out.push('ə'),
                ("ER", Some(0)) => out.push('ɚ'),
                _ => out.push_str(ipa(b)),
            }
        } else if is_consonant(b) {
            out.push_str(ipa(b));
        } else {
            out.push_str(p);
        }
    }
    out
}

// -- the possessive / plural / past suffixes --------------------------------------

/// The -s / -'s ending after these phones.
pub fn plural_suffix(phones: &[String]) -> Vec<String> {
    let Some(last) = phones.last() else {
        return vec!["Z".into()];
    };
    let last = base(last);
    if is_sibilant(last) {
        vec!["IH0".into(), "Z".into()]
    } else if is_vowel(last) || is_voiced_consonant(last) {
        vec!["Z".into()]
    } else {
        vec!["S".into()]
    }
}

/// The -ed ending.
pub fn past_suffix(phones: &[String]) -> Vec<String> {
    let Some(last) = phones.last() else {
        return vec!["D".into()];
    };
    let last = base(last);
    if last == "T" || last == "D" {
        vec!["IH0".into(), "D".into()]
    } else if is_vowel(last) || is_voiced_consonant(last) {
        vec!["D".into()]
    } else {
        vec!["T".into()]
    }
}

/// A helper used across the crate: owned strings from string slices.
pub fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}
