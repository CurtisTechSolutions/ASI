//! The phonetic tokenizer (the port of tokenizer.py): text in, the sounds it is
//! made of out, at one of four levels; and back.

use std::collections::HashMap;

use crate::g2p::Transcriber;
use crate::json::{self, Json};
use crate::lexicon::Lexicon;
use crate::mt::Mt;
use crate::phones::{
    features, is_phone, is_vowel, strip_stress, symbols, to_ipa, BOUNDARY, FEATURE_DIM, PAUSES,
    PAUSE_FULL, PAUSE_QUESTION, PAUSE_SHORT, SPECIALS, UNK, UNK_ID,
};
use crate::phonotactics::{well_formed, Phonotactics};
use crate::syllables::{alliterates, rhymes, syllabify, Syllable};

/// What one token is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Phoneme,
    Constituent,
    Syllable,
    Word,
}

impl Level {
    pub const ALL: [Level; 4] = [
        Level::Phoneme,
        Level::Constituent,
        Level::Syllable,
        Level::Word,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Phoneme => "phoneme",
            Level::Constituent => "constituent",
            Level::Syllable => "syllable",
            Level::Word => "word",
        }
    }

    pub fn parse(name: &str) -> Result<Level, String> {
        match name {
            "phoneme" => Ok(Level::Phoneme),
            "constituent" => Ok(Level::Constituent),
            "syllable" => Ok(Level::Syllable),
            "word" => Ok(Level::Word),
            _ => Err(format!(
                "level must be one of phoneme, constituent, syllable, word, got {name:?}"
            )),
        }
    }
}

/// The tokens every level's vocabulary starts with, in id order.
pub fn fixed() -> Vec<String> {
    let mut out: Vec<String> = SPECIALS.iter().map(|s| s.to_string()).collect();
    out.push(BOUNDARY.to_string());
    out.extend(PAUSES.iter().map(|s| s.to_string()));
    out
}

const TRAILING: &str = ".,;:!?…\"')]}»’”";
const LEADING: &str = "(\"'[{«‘“…";

fn pause_of(c: char) -> Option<&'static str> {
    match c {
        '.' | '!' | '…' => Some(PAUSE_FULL),
        '?' => Some(PAUSE_QUESTION),
        ',' | ';' | ':' | ')' | ']' | '}' => Some(PAUSE_SHORT),
        _ => None,
    }
}

fn pause_strength(p: &str) -> u8 {
    match p {
        PAUSE_SHORT => 1,
        PAUSE_FULL => 2,
        _ => 3,
    }
}

/// One token: its text, what kind of thing it is, the phones it stands for, and its word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub kind: &'static str,
    pub phones: Vec<String>,
    pub word: i64,
}

fn is_cluster_body(body: &str) -> bool {
    if body.is_empty() {
        return false;
    }
    body.split('.').all(|part| {
        let b = part.as_bytes();
        let letters = b.iter().take_while(|c| c.is_ascii_uppercase()).count();
        (1..=2).contains(&letters)
            && (b.len() == letters
                || (b.len() == letters + 1 && (b'0'..=b'2').contains(&b[letters])))
    })
}

/// What a token string is: `(kind, phones)`, or `None` if it is not a phonetic token.
pub fn parse_token(text: &str) -> Option<(&'static str, Vec<String>)> {
    if text.is_empty() {
        return None;
    }
    if SPECIALS.contains(&text) {
        return Some(("special", Vec::new()));
    }
    if text == BOUNDARY {
        return Some(("boundary", Vec::new()));
    }
    if PAUSES.contains(&text) {
        return Some(("pause", Vec::new()));
    }
    if is_phone(text) {
        return Some((
            if is_vowel(text) { "nucleus" } else { "phone" },
            vec![text.to_string()],
        ));
    }
    let onset = text.ends_with('-') && !text.starts_with('-');
    let coda = text.starts_with('-') && !text.ends_with('-');
    let body = if onset {
        &text[..text.len() - 1]
    } else if coda {
        &text[1..]
    } else {
        text
    };
    if !is_cluster_body(body) {
        return None;
    }
    let phones: Vec<String> = body.split('.').map(|s| s.to_string()).collect();
    if !phones.iter().all(|p| is_phone(p)) {
        return None;
    }
    let vowels = phones.iter().filter(|p| is_vowel(p)).count();
    if onset || coda {
        if vowels > 0 {
            return None;
        }
        return Some((if onset { "onset" } else { "coda" }, phones));
    }
    if phones.len() == 1 {
        return Some((if vowels == 1 { "nucleus" } else { "phone" }, phones));
    }
    Some((if vowels <= 1 { "syllable" } else { "word" }, phones))
}

/// Token strings <-> ids: the fixed head, then whatever is read, in order.
#[derive(Clone, Debug)]
pub struct Vocab {
    pub tokens: Vec<String>,
    ids: HashMap<String, usize>,
    pub frozen: bool,
}

impl Vocab {
    pub fn new(tokens: &[String], frozen: bool) -> Vocab {
        let mut v = Vocab {
            tokens: Vec::new(),
            ids: HashMap::new(),
            frozen,
        };
        for t in tokens {
            v.add(t);
        }
        v
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn has(&self, token: &str) -> bool {
        self.ids.contains_key(token)
    }

    pub fn add(&mut self, token: &str) -> usize {
        if let Some(&i) = self.ids.get(token) {
            return i;
        }
        self.ids.insert(token.to_string(), self.tokens.len());
        self.tokens.push(token.to_string());
        self.tokens.len() - 1
    }

    /// The id of a token; a new token gets the next id if `grow` and the vocabulary is not frozen.
    pub fn id(&mut self, token: &str, grow: bool) -> usize {
        if let Some(&i) = self.ids.get(token) {
            return i;
        }
        if grow && !self.frozen {
            return self.add(token);
        }
        UNK_ID
    }

    pub fn token(&self, i: usize) -> &str {
        self.tokens.get(i).map(|s| s.as_str()).unwrap_or(UNK)
    }
}

/// One row of `explain`.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub word: String,
    pub phones: Vec<String>,
    pub how: &'static str,
    pub syllables: Vec<String>,
    pub ipa: String,
    pub score: Option<f64>,
}

/// Text as sounds, at one of four levels.
pub struct Tokenizer {
    pub level: Level,
    pub stress: bool,
    pub boundaries: bool,
    pub pauses: bool,
    pub transcriber: Transcriber,
    pub vocab: Vocab,
    phonotactics: Option<Phonotactics>,
}

impl Tokenizer {
    pub fn new(level: Level, lexicon: Lexicon) -> Tokenizer {
        let vocab = if level == Level::Phoneme {
            Vocab::new(symbols(), true)
        } else {
            Vocab::new(&fixed(), false)
        };
        Tokenizer {
            level,
            stress: true,
            boundaries: true,
            pauses: true,
            transcriber: Transcriber::new(lexicon),
            vocab,
            phonotactics: None,
        }
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.transcriber.lexicon
    }

    /// The sound habits, fitted on the lexicon the first time they are needed.
    pub fn phonotactics(&mut self) -> &Phonotactics {
        if self.phonotactics.is_none() {
            self.phonotactics = Some(Phonotactics::from_lexicon(&self.transcriber.lexicon, false));
        }
        self.phonotactics.as_ref().unwrap()
    }

    pub fn describe(&self) -> String {
        format!(
            "{} tokens, stress {}, {} ids, lexicon of {}",
            self.level.as_str(),
            if self.stress { "kept" } else { "dropped" },
            self.vocab.len(),
            self.lexicon().describe()
        )
    }

    /// The text cut into `("word", spelling)`, `("sounds", token)`, `("boundary", "#")`, `("pause", token)`
    /// and `("punct", mark)` pieces: the first three kinds of token stay as they are, wherever they are.
    pub fn pieces(&self, text: &str) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        for raw in text.split_whitespace() {
            if let Some((kind, _)) = parse_token(raw) {
                match kind {
                    "boundary" => out.push(("boundary", raw.to_string())),
                    "pause" => out.push(("pause", raw.to_string())),
                    "special" => {}
                    _ => out.push(("sounds", raw.to_string())),
                }
                continue;
            }
            if matches!(raw, "-" | "--" | "—" | "–") {
                out.push(("punct", PAUSE_SHORT.to_string()));
                continue;
            }
            let mut word: Vec<char> = raw
                .trim_start_matches(|c| LEADING.contains(c))
                .chars()
                .collect();
            let mut pause: Option<&'static str> = None;
            while let Some(&last) = word.last() {
                if !TRAILING.contains(last) {
                    break;
                }
                if let Some(mark) = pause_of(last) {
                    if pause
                        .map(|p| pause_strength(mark) > pause_strength(p))
                        .unwrap_or(true)
                    {
                        pause = Some(mark);
                    }
                }
                word.pop();
            }
            if !word.is_empty() {
                out.push(("word", word.into_iter().collect()));
            }
            if let Some(p) = pause {
                out.push(("punct", p.to_string()));
            }
        }
        out
    }

    fn stressed(&self, phones: Vec<String>) -> Vec<String> {
        if self.stress {
            phones
        } else {
            strip_stress(&phones)
        }
    }

    /// The phones of each word of the text, punctuation dropped.
    pub fn transcribe(&mut self, text: &str) -> Vec<Vec<String>> {
        let mut words: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for (kind, piece) in self.pieces(text) {
            match kind {
                "word" => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                    let phones = self.transcriber.word(&piece);
                    if !phones.is_empty() {
                        words.push(self.stressed(phones));
                    }
                }
                "sounds" => current.extend(parse_token(&piece).map(|(_, p)| p).unwrap_or_default()),
                _ => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                }
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    fn word_tokens(&self, phones: Vec<String>, word: i64) -> Vec<Token> {
        match self.level {
            Level::Phoneme => phones
                .into_iter()
                .map(|p| Token {
                    kind: if is_vowel(&p) { "nucleus" } else { "phone" },
                    phones: vec![p.clone()],
                    text: p,
                    word,
                })
                .collect(),
            Level::Word => vec![Token {
                text: phones.join("."),
                kind: "word",
                phones,
                word,
            }],
            Level::Syllable | Level::Constituent => {
                let mut out = Vec::new();
                for s in syllabify(&phones, true) {
                    if self.level == Level::Syllable {
                        out.push(Token {
                            text: s.text(),
                            kind: "syllable",
                            phones: s.phones(),
                            word,
                        });
                        continue;
                    }
                    if !s.onset.is_empty() {
                        out.push(Token {
                            text: format!("{}-", s.onset.join(".")),
                            kind: "onset",
                            phones: s.onset.clone(),
                            word,
                        });
                    }
                    out.push(Token {
                        text: s.nucleus.clone(),
                        kind: "nucleus",
                        phones: vec![s.nucleus.clone()],
                        word,
                    });
                    if !s.coda.is_empty() {
                        out.push(Token {
                            text: format!("-{}", s.coda.join(".")),
                            kind: "coda",
                            phones: s.coda.clone(),
                            word,
                        });
                    }
                }
                out
            }
        }
    }

    /// The tokens of a text at this level, boundaries and pauses included.
    #[allow(unused_assignments)] // the flush macro's last assignments are read by the next flush, or by nobody
    pub fn tokenize(&mut self, text: &str) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::new();
        let mut word_index: i64 = 0;
        let mut current: Vec<String> = Vec::new();
        let mut last = "";
        let pieces = self.pieces(text);
        macro_rules! flush {
            () => {
                if !current.is_empty() {
                    if last == "word" && self.boundaries {
                        out.push(Token {
                            text: BOUNDARY.to_string(),
                            kind: "boundary",
                            phones: Vec::new(),
                            word: -1,
                        });
                    }
                    let phones = self.stressed(std::mem::take(&mut current));
                    out.extend(self.word_tokens(phones, word_index));
                    word_index += 1;
                    last = "word";
                }
                current.clear();
            };
        }
        for (kind, piece) in pieces {
            match kind {
                "word" => {
                    flush!();
                    let phones = self.transcriber.word(&piece);
                    if !phones.is_empty() {
                        current = phones;
                        flush!();
                    }
                }
                "sounds" => current.extend(parse_token(&piece).map(|(_, p)| p).unwrap_or_default()),
                "boundary" => {
                    // a boundary given as a token stays, wherever it is
                    flush!();
                    out.push(Token {
                        text: BOUNDARY.to_string(),
                        kind: "boundary",
                        phones: Vec::new(),
                        word: -1,
                    });
                    last = "boundary";
                }
                "pause" => {
                    // so does a pause given as a token
                    flush!();
                    out.push(Token {
                        text: piece.clone(),
                        kind: "pause",
                        phones: Vec::new(),
                        word: -1,
                    });
                    last = "pause";
                }
                _ => {
                    // punctuation: one pause, after something, never two in a row
                    flush!();
                    if self.pauses && last != "pause" && !out.is_empty() {
                        out.push(Token {
                            text: piece.clone(),
                            kind: "pause",
                            phones: Vec::new(),
                            word: -1,
                        });
                        last = "pause";
                    }
                }
            }
        }
        flush!();
        let _ = (word_index, last);
        out
    }

    /// The token strings.
    pub fn tokens(&mut self, text: &str) -> Vec<String> {
        self.tokenize(text).into_iter().map(|t| t.text).collect()
    }

    /// The text form: the tokens joined by single spaces.
    pub fn text(&mut self, text: &str) -> String {
        self.tokens(text).join(" ")
    }

    /// The ids of the tokens; an unseen token grows the vocabulary (`grow`) or is `<unk>`.
    pub fn encode(&mut self, text: &str, grow: bool) -> Vec<usize> {
        let tokens = self.tokens(text);
        tokens.iter().map(|t| self.vocab.id(t, grow)).collect()
    }

    /// Token strings back into `Token`s; an error for one that is not phonetic.
    pub fn parse(&self, tokens: &[String]) -> Result<Vec<Token>, String> {
        let mut out: Vec<Token> = Vec::new();
        let mut word: i64 = 0;
        for text in tokens {
            let Some((kind, phones)) = parse_token(text) else {
                return Err(format!("{text:?} is not a phonetic token"));
            };
            match kind {
                "boundary" | "pause" => {
                    if out
                        .last()
                        .map(|t| t.kind != "boundary" && t.kind != "pause")
                        .unwrap_or(false)
                    {
                        word += 1;
                    }
                    out.push(Token {
                        text: text.clone(),
                        kind,
                        phones: Vec::new(),
                        word: -1,
                    });
                }
                "special" => out.push(Token {
                    text: text.clone(),
                    kind,
                    phones: Vec::new(),
                    word: -1,
                }),
                _ => out.push(Token {
                    text: text.clone(),
                    kind,
                    phones,
                    word,
                }),
            }
        }
        Ok(out)
    }

    /// Token strings grouped back into the phones of each word.
    pub fn words_of(&self, tokens: &[String]) -> Vec<Vec<String>> {
        let mut words = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for t in tokens {
            let Some((kind, phones)) = parse_token(t) else {
                continue;
            };
            match kind {
                "boundary" | "pause" | "special" => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                }
                _ => current.extend(phones),
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    pub fn tokens_of_ids(&self, ids: &[usize]) -> Vec<String> {
        ids.iter()
            .map(|&i| self.vocab.token(i).to_string())
            .collect()
    }

    /// The sounds spelled back as words, with the pauses as punctuation.
    pub fn decode(&mut self, tokens: &[String]) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for t in tokens {
            let Some((kind, phones)) = parse_token(t) else {
                continue;
            };
            match kind {
                "boundary" | "pause" | "special" => {
                    if !current.is_empty() {
                        let word = self.transcriber.spell(&current);
                        out.push(word);
                        current.clear();
                    }
                    if kind == "pause" {
                        if let Some(last) = out.last_mut() {
                            last.push_str(t);
                        }
                    }
                }
                _ => current.extend(phones),
            }
        }
        if !current.is_empty() {
            let word = self.transcriber.spell(&current);
            out.push(word);
        }
        out.join(" ")
    }

    pub fn decode_ids(&mut self, ids: &[usize]) -> String {
        let tokens = self.tokens_of_ids(ids);
        self.decode(&tokens)
    }

    /// The articulatory features of a token: a phone's own, the mean over a cluster's or a syllable's
    /// phones, zeros for a boundary, a pause or a special.
    pub fn features_of(&self, token: &str) -> Vec<f64> {
        let mut out = vec![0.0; FEATURE_DIM];
        let Some((_, phones)) = parse_token(token) else {
            return out;
        };
        if phones.is_empty() {
            return out;
        }
        for p in &phones {
            let f = features(p).unwrap_or([0.0; FEATURE_DIM]);
            for i in 0..FEATURE_DIM {
                out[i] += f[i];
            }
        }
        for x in out.iter_mut() {
            *x /= phones.len() as f64;
        }
        out
    }

    /// The phones of one word.
    pub fn pronounce(&mut self, word: &str) -> Vec<String> {
        let phones = self.transcriber.word(word);
        self.stressed(phones)
    }

    /// The text in the IPA, a stress mark before each stressed syllable.
    pub fn ipa(&mut self, text: &str) -> String {
        let mut words = Vec::new();
        for phones in self.transcribe(text) {
            let mut s = String::new();
            for syl in syllabify(&phones, true) {
                match syl.stress() {
                    1 => s.push('ˈ'),
                    2 => s.push('ˌ'),
                    _ => {}
                }
                s.push_str(&to_ipa(&syl.phones(), false));
            }
            words.push(s);
        }
        words.join(" ")
    }

    pub fn syllables(&mut self, word: &str) -> Vec<Syllable> {
        let phones = self.pronounce(word);
        syllabify(&phones, true)
    }

    pub fn rhymes_with(&mut self, a: &str, b: &str) -> bool {
        let (pa, pb) = (self.pronounce(a), self.pronounce(b));
        rhymes(&pa, &pb)
    }

    pub fn alliterates_with(&mut self, a: &str, b: &str) -> bool {
        let (pa, pb) = (self.pronounce(a), self.pronounce(b));
        alliterates(&pa, &pb)
    }

    pub fn affinity(&mut self, a: &str, b: &str) -> f64 {
        self.phonotactics().affinity(a, b)
    }

    pub fn score(&mut self, word: &str) -> f64 {
        let phones = self.pronounce(word);
        self.phonotactics().score(&phones)
    }

    pub fn well_formed(&mut self, word: &str) -> bool {
        let phones = self.pronounce(word);
        well_formed(&phones)
    }

    /// A new, pronounceable word (spelled as it sounds), built from the sound habits with a seeded RNG.
    pub fn coin(&mut self, seed: u64, syllables: usize) -> String {
        let mut rng = Mt::new(seed);
        let avoid = self.lexicon().items();
        let phones = self
            .phonotactics()
            .build(&mut rng, syllables, 1, 3, &avoid, 50);
        if phones.is_empty() {
            return String::new();
        }
        self.transcriber.spell(&phones)
    }

    /// A portmanteau of two words at the joint where their sounds meet best.
    pub fn blend(&mut self, a: &str, b: &str) -> Result<String, String> {
        let (pa, pb) = (self.pronounce(a), self.pronounce(b));
        let (phones, _, _) = self.phonotactics().blend(&pa, &pb)?;
        Ok(self.transcriber.spell(&phones))
    }

    /// One row per word: the spelling, its phones, where they came from, the syllables and the IPA.
    pub fn explain(&mut self, text: &str) -> Vec<Explanation> {
        let mut rows = Vec::new();
        for (kind, piece) in self.pieces(text) {
            if kind != "word" {
                continue;
            }
            let (phones, how) = self.transcriber.explain(&piece);
            let phones = self.stressed(phones);
            let syllables: Vec<String> =
                syllabify(&phones, true).iter().map(|s| s.text()).collect();
            let ipa = self.ipa(&piece);
            let score = if phones.is_empty() {
                None
            } else {
                Some(round3(self.phonotactics().score(&phones)))
            };
            rows.push(Explanation {
                word: piece,
                phones,
                how,
                syllables,
                ipa,
                score,
            });
        }
        rows
    }

    // -- persistence -----------------------------------------------------------------

    /// Everything learned by reading: the grown vocabulary, the memory, the counts.
    pub fn to_json(&self) -> Json {
        let mut memory: Vec<(String, Json)> = self
            .transcriber
            .memory
            .iter()
            .map(|(w, p)| (w.clone(), Json::strs(p)))
            .collect();
        memory.sort_by(|a, b| a.0.cmp(&b.0));
        let mut counts: Vec<(String, Json)> = self
            .transcriber
            .counts
            .iter()
            .map(|(w, c)| (w.clone(), Json::Num(*c as f64)))
            .collect();
        counts.sort_by(|a, b| a.0.cmp(&b.0));
        Json::obj(vec![
            ("format", Json::str("phonetok")),
            ("version", Json::Num(1.0)),
            ("level", Json::str(self.level.as_str())),
            ("stress", Json::Bool(self.stress)),
            ("boundaries", Json::Bool(self.boundaries)),
            ("pauses", Json::Bool(self.pauses)),
            ("vocab", Json::strs(&self.vocab.tokens)),
            ("memory", Json::Obj(memory)),
            ("counts", Json::Obj(counts)),
        ])
    }

    pub fn load_json(&mut self, d: &Json) -> Result<(), String> {
        if d.get("format").as_str() != Some("phonetok") {
            return Err("not a phonetok tokenizer document".to_string());
        }
        if let Some(level) = d.get("level").as_str() {
            if level != self.level.as_str() {
                return Err(format!(
                    "the file holds a {level} tokenizer, this one is {}",
                    self.level.as_str()
                ));
            }
        }
        self.stress = d.get("stress").as_bool().unwrap_or(self.stress);
        self.boundaries = d.get("boundaries").as_bool().unwrap_or(self.boundaries);
        self.pauses = d.get("pauses").as_bool().unwrap_or(self.pauses);
        if self.level != Level::Phoneme {
            let mut tokens = d.get("vocab").strings();
            if tokens.is_empty() {
                tokens = fixed();
            }
            self.vocab = Vocab::new(&tokens, false);
        }
        for (w, p) in d.get("memory").as_obj() {
            self.transcriber.memory.insert(w.clone(), p.strings());
        }
        for (w, c) in d.get("counts").as_obj() {
            *self.transcriber.counts.entry(w.clone()).or_insert(0) +=
                c.as_f64().unwrap_or(0.0) as usize;
        }
        Ok(())
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_json().to_pretty() + "\n").map_err(|e| format!("{path}: {e}"))
    }

    pub fn load(path: &str, lexicon: Lexicon) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let d = json::parse(&text)?;
        let level = Level::parse(d.get("level").as_str().unwrap_or("phoneme"))?;
        let mut t = Tokenizer::new(level, lexicon);
        t.load_json(&d)?;
        Ok(t)
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0 + if x < 0.0 { -0.5 } else { 0.5 }).trunc() / 1000.0
}
