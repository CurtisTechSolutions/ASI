//! The phonetic units of the encoding read text through the phonetic tokenizer
//! (the sibling PhoneticTokenizer crate, the Rust port of the phonetok package).
//!
//! One tokenizer per unit, made on first use and kept: what it remembers of the
//! words it sounded out is what lets a prediction be spelled back.  The lexicon
//! is the *portable* one - the bundled core plus the file `PHONETOK_LEXICON`
//! names - the lexicon the Python and Go ports read too, so that a model whose
//! symbols are sounds means the same sounds in every port.  Training reads from
//! many threads, so the tokenizer sits behind a mutex.

use std::sync::{Mutex, OnceLock};

use phonetok::lexicon::Lexicon;
use phonetok::tokenizer::{Level, Tokenizer};

use crate::encoding::Unit;

type Bridge = Result<Mutex<Tokenizer>, String>;

static PHONES: OnceLock<Bridge> = OnceLock::new();
static SYLLABLES: OnceLock<Bridge> = OnceLock::new();

fn make(unit: Unit) -> Bridge {
    let lexicon =
        Lexicon::portable().map_err(|e| format!("the {} unit needs the phonetic lexicon: {e}", unit.name()))?;
    let level = if unit == Unit::Syllables {
        Level::Syllable
    } else {
        Level::Phoneme
    };
    Ok(Mutex::new(Tokenizer::new(level, lexicon)))
}

/// The tokenizer a phonetic unit reads through, or the reason it could not be made.
pub fn tokenizer(unit: Unit) -> Result<&'static Mutex<Tokenizer>, String> {
    let cell = if unit == Unit::Syllables { &SYLLABLES } else { &PHONES };
    cell.get_or_init(|| make(unit)).as_ref().map_err(|e| e.clone())
}

/// The text as the sounds it is made of, joined by single spaces: the text form
/// of the tokenizer, which is idempotent.
pub fn text(unit: Unit, text: &str) -> String {
    let tok = tokenizer(unit).expect("Encoding::validate refused the encoding before any text could reach here");
    let mut tok = tok.lock().unwrap_or_else(|e| e.into_inner());
    tok.text(text)
}

/// The words a phonetic text spells: `"DH AH0 # K AE1 T"` -> `"the cat"`.
pub fn spell(unit: Unit, text: &str) -> String {
    let Ok(tok) = tokenizer(unit) else {
        return text.to_string();
    };
    let mut tok = tok.lock().unwrap_or_else(|e| e.into_inner());
    let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
    tok.decode(&tokens)
}
