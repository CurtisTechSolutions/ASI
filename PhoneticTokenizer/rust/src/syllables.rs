//! Syllables (the port of syllables.py): maximal onset over the closed list of
//! English onsets, a stressed lax vowel closing its syllable, rhyme and alliteration.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::phones::{base, is_stressed, is_vowel, sonority, stress_of, strip_stress, CONSONANTS};

const DOUBLE_ONSETS: &str = "P L|P R|P Y|B L|B R|B Y|T R|T W|T Y|D R|D W|D Y|K L|K R|K W|K Y|G L|G R|G W|G Y|\
F L|F R|F Y|V Y|TH R|TH W|TH Y|SH R|SH M|SH N|SH L|SH W|S P|S T|S K|S M|S N|S L|S W|S F|M Y|N Y|L Y|HH Y";
const TRIPLE_ONSETS: &str = "S P L|S P R|S P Y|S T R|S T Y|S K L|S K R|S K W|S K Y";

/// Every consonant cluster that may start an English syllable, the empty one included, joined by spaces.
pub fn onsets() -> &'static HashSet<String> {
    static ONSETS: OnceLock<HashSet<String>> = OnceLock::new();
    ONSETS.get_or_init(|| {
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::new());
        for c in CONSONANTS {
            if c != "NG" {
                set.insert(c.to_string());
            }
        }
        for s in DOUBLE_ONSETS.split('|').chain(TRIPLE_ONSETS.split('|')) {
            set.insert(s.to_string());
        }
        set
    })
}

pub const LAX_VOWELS: [&str; 5] = ["AE", "AH", "EH", "IH", "UH"];
const APPENDIX: [&str; 5] = ["S", "Z", "T", "D", "TH"];

/// May this consonant cluster start a syllable?  The empty cluster may.
pub fn is_legal_onset(cluster: &[String]) -> bool {
    onsets().contains(&cluster.join(" "))
}

/// May this cluster close a syllable: does its sonority fall away from the vowel?
pub fn is_legal_coda(cluster: &[String]) -> bool {
    if cluster.is_empty() {
        return true;
    }
    if cluster.iter().any(|c| c == "HH" || c == "W" || c == "Y") || cluster.len() > 4 {
        return false;
    }
    let mut core: Vec<&str> = cluster.iter().map(|s| s.as_str()).collect();
    while core.len() > 1 && APPENDIX.contains(core.last().unwrap()) {
        core.pop();
    }
    core.windows(2).all(|w| sonority(w[0]) >= sonority(w[1]))
}

/// One syllable: onset + nucleus + coda.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Syllable {
    pub onset: Vec<String>,
    pub nucleus: String,
    pub coda: Vec<String>,
}

impl Syllable {
    /// 0, 1 or 2 (0 for a vowel without a digit and for a consonant nucleus).
    pub fn stress(&self) -> u8 {
        stress_of(&self.nucleus).unwrap_or(0)
    }

    pub fn phones(&self) -> Vec<String> {
        let mut out = self.onset.clone();
        out.push(self.nucleus.clone());
        out.extend(self.coda.iter().cloned());
        out
    }

    pub fn rime(&self) -> Vec<String> {
        let mut out = vec![self.nucleus.clone()];
        out.extend(self.coda.iter().cloned());
        out
    }

    pub fn is_open(&self) -> bool {
        self.coda.is_empty()
    }

    /// The syllable as one token: its phones joined by dots.
    pub fn text(&self) -> String {
        self.phones().join(".")
    }
}

fn split_cluster(cluster: &[String]) -> usize {
    (0..=cluster.len())
        .find(|&keep| is_legal_onset(&cluster[keep..]))
        .unwrap_or(cluster.len())
}

/// Breaks a word's phones into syllables by maximal onset; `close_lax` applies the stressed-lax-vowel rule.
pub fn syllabify(phones: &[String], close_lax: bool) -> Vec<Syllable> {
    if phones.is_empty() {
        return Vec::new();
    }
    let vowels: Vec<usize> = (0..phones.len())
        .filter(|&i| is_vowel(&phones[i]))
        .collect();
    if vowels.is_empty() {
        let mut peak = 0;
        for i in 1..phones.len() {
            if sonority(&phones[i]) > sonority(&phones[peak]) {
                peak = i;
            }
        }
        return vec![Syllable {
            onset: phones[..peak].to_vec(),
            nucleus: phones[peak].clone(),
            coda: phones[peak + 1..].to_vec(),
        }];
    }
    let mut out = Vec::new();
    let mut onset_start = 0;
    for (k, &vi) in vowels.iter().enumerate() {
        let onset = phones[onset_start..vi].to_vec();
        let coda;
        if k + 1 < vowels.len() {
            let cluster = &phones[vi + 1..vowels[k + 1]];
            let mut keep = split_cluster(cluster);
            if close_lax
                && keep == 0
                && !cluster.is_empty()
                && LAX_VOWELS.contains(&base(&phones[vi]))
                && is_stressed(&phones[vi])
                && is_legal_onset(&cluster[1..])
            {
                keep = 1;
            }
            coda = cluster[..keep].to_vec();
            onset_start = vi + 1 + keep;
        } else {
            coda = phones[vi + 1..].to_vec();
        }
        out.push(Syllable {
            onset,
            nucleus: phones[vi].clone(),
            coda,
        });
    }
    out
}

/// One per vowel, or one for a word with none.
pub fn syllable_count(phones: &[String]) -> usize {
    let n = phones.iter().filter(|p| is_vowel(p)).count();
    if n == 0 && !phones.is_empty() {
        1
    } else {
        n
    }
}

/// The index of the syllable carrying primary stress (the first secondary one, else the last), if any.
pub fn stressed_syllable(syllables: &[Syllable]) -> Option<usize> {
    if syllables.is_empty() {
        return None;
    }
    if let Some(i) = syllables.iter().position(|s| s.stress() == 1) {
        return Some(i);
    }
    if let Some(i) = syllables.iter().position(|s| s.stress() == 2) {
        return Some(i);
    }
    Some(syllables.len() - 1)
}

fn tail(syllables: &[Syllable], i: usize) -> Vec<String> {
    let mut out: Vec<String> = syllables[i..].iter().flat_map(|s| s.phones()).collect();
    out.drain(..syllables[i].onset.len());
    strip_stress(&out)
}

/// Do two words share their sounds from the last stressed vowel on?
pub fn rhymes(a: &[String], b: &[String]) -> bool {
    let (sa, sb) = (syllabify(a, true), syllabify(b, true));
    match (stressed_syllable(&sa), stressed_syllable(&sb)) {
        (Some(ia), Some(ib)) => tail(&sa, ia) == tail(&sb, ib),
        _ => false,
    }
}

/// Do two words start with the same consonant sound(s)?
pub fn alliterates(a: &[String], b: &[String]) -> bool {
    let (sa, sb) = (syllabify(a, true), syllabify(b, true));
    match (sa.first(), sb.first()) {
        (Some(x), Some(y)) => !x.onset.is_empty() && x.onset == y.onset,
        _ => false,
    }
}
