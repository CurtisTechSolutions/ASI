//! English in and out of a network, through the sounds it is made of.
//!
//! The phonetic tokenizer of this repository (`../../PhoneticTokenizer/rust`, the `phonetok` crate) reads English
//! as ARPAbet phonemes - `"cats"` is `K AE T S` - and spells phonemes back as English.  The network's encoder and
//! decoder (`network.rs`) put it at the two ends of a network.  The sounds are the 39 phonemes without their stress
//! digits (`stress` keeps them), the word boundary `#` and the pauses `, . ?`: 43 symbols.  The tokenizer is the
//! portable one - the core lexicon and the letter-to-sound rules, plus the file `PHONETOK_LEXICON` names - the
//! same the Python package uses, so the two ports hear every text alike.  `latticefsm/phonetic.py` is the same.

use phonetok::phones::{symbols, BOUNDARY, PAUSES, SPECIALS};
use phonetok::{Level, Lexicon, Tokenizer};

/// The phonetic tokenizer at the phoneme level over the portable lexicon, with or without the vowels' stress.
pub fn tokenizer(stress: bool) -> Tokenizer {
    let lexicon = Lexicon::portable().unwrap_or_else(|_| Lexicon::core());
    let mut t = Tokenizer::new(Level::Phoneme, lexicon);
    t.stress = stress;
    t
}

/// The sounds, in a fixed order: `#`, the pauses `, . ?`, then the phonemes (alphabetical; with `stress`, every
/// vowel as `AA AA0 AA1 AA2`, the tokenizer's own id order).
pub fn inventory(stress: bool) -> Vec<String> {
    if stress {
        return symbols()
            .iter()
            .filter(|s| !SPECIALS.contains(&s.as_str()))
            .cloned()
            .collect();
    }
    let mut phonemes: Vec<String> = symbols()
        .iter()
        .filter(|s| !SPECIALS.contains(&s.as_str()) && *s != BOUNDARY && !PAUSES.contains(&s.as_str()))
        .filter(|s| !s.ends_with(|c: char| c.is_ascii_digit()))
        .cloned()
        .collect();
    phonemes.sort();
    let mut out = vec![BOUNDARY.to_string()];
    out.extend(PAUSES.iter().map(|s| s.to_string()));
    out.extend(phonemes);
    out
}

fn pairs(a: &'static str, b: &'static str) -> Vec<(String, String)> {
    let (xs, ys): (Vec<&str>, Vec<&str>) = (a.split_whitespace().collect(), b.split_whitespace().collect());
    assert_eq!(xs.len(), ys.len());
    xs.into_iter()
        .zip(ys)
        .map(|(x, y)| (x.to_string(), y.to_string()))
        .collect()
}

/// Nouns and their plurals: `-s` after a voiceless sound, `-z` after a voiced one, `-iz` after a hiss.
pub fn plurals() -> Vec<(String, String)> {
    pairs(
        "cat dog bus hat cup bed pig box rose sock map bag fox kiss pen car bird book ship tree cake bell lamp star \
         fish dish wish bench horse key ring frog bat hand rock sea door song boat bike nose glass page road kite drum",
        "cats dogs buses hats cups beds pigs boxes roses socks maps bags foxes kisses pens cars birds books ships \
         trees cakes bells lamps stars fishes dishes wishes benches horses keys rings frogs bats hands rocks seas \
         doors songs boats bikes noses glasses pages roads kites drums",
    )
}

/// Verbs and their past tenses: `-t` after a voiceless sound, `-d` after a voiced one, `-id` after t or d.
pub fn past() -> Vec<(String, String)> {
    pairs(
        "walk play want jump call need talk look help cook kiss wash fill rain love open start end paint wait pull \
         push dance miss clean kick smile park",
        "walked played wanted jumped called needed talked looked helped cooked kissed washed filled rained loved \
         opened started ended painted waited pulled pushed danced missed cleaned kicked smiled parked",
    )
}

/// A task by name: `plural` or `past`.
pub fn task(name: &str) -> Result<Vec<(String, String)>, String> {
    match name {
        "plural" => Ok(plurals()),
        "past" => Ok(past()),
        other => Err(format!("no task {other:?}: plural or past")),
    }
}

/// English pairs, input and target.
pub type Pairs = Vec<(String, String)>;

/// Every `every`-th pair held out for testing, the rest for teaching.
pub fn split(pairs: &[(String, String)], every: usize) -> (Pairs, Pairs) {
    let train = pairs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % every != every - 1)
        .map(|(_, p)| p.clone())
        .collect();
    let test = pairs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % every == every - 1)
        .map(|(_, p)| p.clone())
        .collect();
    (train, test)
}
