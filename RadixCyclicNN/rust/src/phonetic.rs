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

use phonetok::acoustic::{AcousticTokenizer, Codebook};
use phonetok::lexicon::Lexicon;
use phonetok::tokenizer::{Level, Tokenizer};

use crate::encoding::{Encoding, Unit};

// -- the acoustic units ------------------------------------------------------------

static ACOUSTIC: OnceLock<Result<AcousticTokenizer, String>> = OnceLock::new();

/// The acoustic units' tokenizer over its codebook - the bundled one, or the
/// file `PHONETOK_CODEBOOK` names - made once; a model's units mean nothing
/// without the codebook that made them, so the two travel together.
pub fn acoustic_tokenizer() -> Result<&'static AcousticTokenizer, String> {
    ACOUSTIC
        .get_or_init(|| {
            let book = match std::env::var("PHONETOK_CODEBOOK") {
                Ok(path) if !path.is_empty() => Codebook::load(&path)
                    .map_err(|e| format!("the acoustic unit's codebook (PHONETOK_CODEBOOK): {e}"))?,
                _ => Codebook::bundled().map_err(|e| format!("the acoustic unit needs its codebook: {e}"))?,
            };
            AcousticTokenizer::new(book, true)
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// A WAV file's bytes as a text of acoustic units - `"q2 q28 q55 ..."`, runs
/// collapsed.  One recording is one utterance, and so one text.
pub fn hear_audio(data: &[u8]) -> Result<String, String> {
    Ok(acoustic_tokenizer()?.listen_wav(data)?.text())
}

/// Whether a file is a WAV, by its name: something to hear rather than read.
pub fn is_audio_file(path: &str) -> bool {
    path.to_lowercase().ends_with(".wav")
}

/// The sample rate a model is heard at: `rate` for the synthesizer, the
/// codebook's for acoustic units.
pub fn output_rate(enc: Encoding, rate: u32) -> Result<u32, String> {
    if enc.unit != Unit::Acoustic {
        return Ok(rate);
    }
    Ok(acoustic_tokenizer()?.book.analysis.rate)
}

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
