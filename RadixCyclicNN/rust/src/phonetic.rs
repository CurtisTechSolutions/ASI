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

use crate::codegen::set;
use crate::encoding::{Encoding, Unit};
use crate::json::Json;

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

/// The words a phonetic text spells: `"DH AH0 # K AE1 T"` -> `"the cat"`.  A
/// text given in words is read as the sounds it makes first, so it spells
/// itself back, and so does a text that mixes the two.
pub fn spell(unit: Unit, text: &str) -> String {
    let Ok(tok) = tokenizer(unit) else {
        return text.to_string();
    };
    let mut tok = tok.lock().unwrap_or_else(|e| e.into_inner());
    let sounds = tok.text(text);
    let tokens: Vec<String> = sounds.split_whitespace().map(|s| s.to_string()).collect();
    tok.decode(&tokens)
}

// -- what a model of sounds said, in words ------------------------------------------

/// The words a prediction of a model of sounds spells: `spelled`, the whole
/// text read back as English, and `spelled_continuation`, the part of it the
/// continuation wrote ([`Encoding::spell_tail`]), so the prefix's words are
/// `spelled` without it.  Every other encoding is its own spelling and gets no
/// pairs: the fields are there only when they say something.  Python's
/// `spelled_prediction`.
pub fn spelled_prediction(enc: Encoding, full_text: &str, continuation: &str) -> Vec<(String, Json)> {
    if !enc.unit.phonetic() {
        return Vec::new();
    }
    vec![
        ("spelled".to_string(), Json::str(enc.spell(full_text))),
        (
            "spelled_continuation".to_string(),
            Json::str(enc.spell_tail(full_text, continuation)),
        ),
    ]
}

/// What a reader - a teacher, a reviewer, a partner - is shown of a text a
/// model wrote: a model of sounds is shown the English its sounds spell, so
/// an LLM marks, corrects and answers words rather than `DH.AH0 # K.AE1.T`;
/// every other text is shown as it is.  What the reader writes back stays in
/// words: a model of sounds reads it as the sounds it makes.  Python's
/// `reader_text`.
pub fn reader_text(enc: Encoding, text: &str) -> String {
    if !enc.unit.phonetic() {
        return text.to_string();
    }
    enc.spell(text)
}

/// A completion of `prefix` by a model of sounds, as a reader is shown it:
/// `(prefix, continuation)` in words, the sentence being the two joined.
/// When the sounds kept the prefix's words, the prefix is the one given -
/// capitals, punctuation and trailing space as they were - and the
/// continuation is the words after it (a pause that attaches to the prefix's
/// last word stays on it); when the continuation finished a word the prefix
/// began, both are cut from the words of the whole
/// ([`Encoding::spell_tail`]).  Any other encoding is its own spelling.
/// Python's `spelled_completion`.
pub fn spelled_completion(enc: Encoding, prefix: &str, full_text: &str, continuation: &str) -> (String, String) {
    if !enc.unit.phonetic() {
        return (prefix.to_string(), continuation.to_string());
    }
    let tail = enc.spell_tail(full_text, continuation);
    let whole = enc.spell(full_text);
    let head = whole.strip_suffix(tail.as_str()).unwrap_or("").to_string();
    let words = |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_lowercase).collect() };
    if words(&head) != words(prefix) {
        return (head, tail);
    }
    if tail.chars().next().is_none_or(char::is_whitespace) {
        // a word of its own after the prefix, which keeps its trailing space
        return (prefix.to_string(), tail.trim_start().to_string());
    }
    (prefix.trim_end().to_string(), tail)
}

/// A thought's record ([`crate::thinking::Thought::to_json`]) with `spelled`
/// beside its text, and its questions' too; any other encoding, the record as
/// it is.  Python's `spelled_thought`.
pub fn spelled_thought(enc: Encoding, thought: Json) -> Json {
    let Json::Obj(mut pairs) = thought else { return thought };
    if !enc.unit.phonetic() {
        return Json::Obj(pairs);
    }
    let doc = Json::Obj(pairs.clone());
    let questions = doc
        .at("questions")
        .as_array()
        .iter()
        .map(|q| spelled_thought(enc, q.clone()))
        .collect();
    set(
        &mut pairs,
        "spelled",
        Json::str(enc.spell(doc.at("text").as_str().unwrap_or(""))),
    );
    set(&mut pairs, "questions", Json::Arr(questions));
    Json::Obj(pairs)
}

/// A turn's record ([`crate::dialogue::Turn::to_json`]) with the words it
/// spells beside its sounds: `spelled` for the whole line, `spelled_reply` for
/// the part the search added after the context it picked up, and the thought
/// of a rethink spelled too.  Any other encoding: the record as it is.
/// Python's `spelled_turn`.
pub fn spelled_turn(enc: Encoding, turn: Json) -> Json {
    let Json::Obj(mut pairs) = turn else { return turn };
    if !enc.unit.phonetic() {
        return Json::Obj(pairs);
    }
    let doc = Json::Obj(pairs.clone());
    let text = doc.at("text").as_str().unwrap_or("");
    set(&mut pairs, "spelled", Json::str(enc.spell(text)));
    set(
        &mut pairs,
        "spelled_reply",
        Json::str(enc.spell_tail(text, doc.at("reply").as_str().unwrap_or(""))),
    );
    if let Json::Obj(mut rethink) = doc.at("rethink").clone() {
        let thought = doc.at("rethink").at("thought");
        if !thought.is_null() {
            set(&mut rethink, "thought", spelled_thought(enc, thought.clone()));
            set(&mut pairs, "rethink", Json::Obj(rethink));
        }
    }
    Json::Obj(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::parse_encoding;

    fn enc(spec: &str) -> Encoding {
        parse_encoding(spec).unwrap()
    }

    #[test]
    fn a_text_in_words_spells_itself_back() {
        let (phones, syllables) = (enc("phone:3:1"), enc("syllable:2:1"));
        assert_eq!(phones.spell("DH AH0 # K AE1 T # S AE1 T ."), "the cat sat.");
        assert_eq!(phones.spell("the cat sat"), "the cat sat");
        assert_eq!(phones.spell("the K AE1 T sat"), "the cat sat");
        assert_eq!(syllables.spell("B.AH1.T ER0 # K.AH1.P"), "butter cup");
        assert_eq!(syllables.spell("the K.AE1.T sat"), "the cat sat");
        assert_eq!(enc("char:3:1").spell("DH AH0"), "DH AH0");
    }

    /// Python's `test_a_continuation_is_spelled_as_the_part_it_wrote`.
    #[test]
    fn a_continuation_is_spelled_as_the_part_it_wrote() {
        for spec in ["phone:3:1", "syllable:2:1"] {
            let e = enc(spec);
            let whole = e.join(&["the cat sat on the mat. hello"]);
            for prefix in ["", "the", "the cat", "the cat sat on the mat"] {
                let head = e.join(&[prefix]);
                let tail = &whole[head.len()..];
                assert_eq!(
                    e.spell(&head) + &e.spell_tail(&whole, tail),
                    "the cat sat on the mat. hello",
                    "{spec} after {prefix:?}"
                );
            }
            assert_eq!(e.spell_tail(&whole, ""), "");
        }
        let phones = enc("phone:3:1");
        phones.units("the ca"); // the tokenizer reads "ca" as K AH1, and remembers it
        assert_eq!(
            phones.spell_tail("DH AH0 # K AH1 T # S AE1 T", "T # S AE1 T"),
            "cut sat"
        );
        assert_eq!(phones.spell_tail("DH AH0 # K AE1 T", "S AE1 T"), "sat");
        assert_eq!(enc("char:3:1").spell_tail("the cat sat", " sat"), " sat");
    }

    #[test]
    fn the_records_carry_the_words_they_spell() {
        let phones = enc("phone:3:1");
        let whole = "DH AH0 # K AE1 T # S AE1 T";
        assert_eq!(
            Json::Obj(spelled_prediction(phones, whole, "# S AE1 T")),
            Json::obj([
                ("spelled", Json::str("the cat sat")),
                ("spelled_continuation", Json::str(" sat"))
            ])
        );
        assert!(spelled_prediction(enc("char:3:1"), "the cat", "cat").is_empty());
        let question = Json::obj([("text", Json::str("S AE1 T .")), ("questions", Json::Arr(Vec::new()))]);
        let thought = spelled_thought(
            phones,
            Json::obj([
                ("text", Json::str("DH AH0 # K AE1 T")),
                ("questions", Json::Arr(vec![question])),
            ]),
        );
        assert_eq!(thought.at("spelled").as_str(), Some("the cat"));
        assert_eq!(
            thought.at("questions").as_array()[0].at("spelled").as_str(),
            Some("sat.")
        );
        let turn = Json::obj([
            ("index", Json::Int(1)),
            ("text", Json::str(whole)),
            ("context", Json::str("the cat")),
            ("reply", Json::str("# S AE1 T")),
            (
                "rethink",
                Json::obj([
                    ("kind", Json::str("repeat")),
                    (
                        "thought",
                        Json::obj([("text", Json::str("K AE1 T")), ("questions", Json::Arr(Vec::new()))]),
                    ),
                ]),
            ),
        ]);
        let spelled = spelled_turn(phones, turn.clone());
        assert_eq!(spelled.at("spelled").as_str(), Some("the cat sat"));
        assert_eq!(spelled.at("spelled_reply").as_str(), Some(" sat"));
        assert_eq!(spelled.at("rethink").at("kind").as_str(), Some("repeat"));
        assert_eq!(spelled.at("rethink").at("thought").at("spelled").as_str(), Some("cat"));
        // the keys a record already had keep their place, as Python's dict update keeps them
        let Json::Obj(pairs) = &spelled else {
            panic!("an object")
        };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "index",
                "text",
                "context",
                "reply",
                "rethink",
                "spelled",
                "spelled_reply"
            ]
        );
        assert_eq!(spelled_turn(enc("char:3:1"), turn.clone()), turn);
    }
}
