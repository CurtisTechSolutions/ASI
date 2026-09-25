//! The pronouncing lexicon (the port of lexicon.py): CMU-format entries,
//! layered from the bundled core and the file named by `PHONETOK_LEXICON`.

use std::collections::HashMap;

use crate::phones::{check_phones, strip_stress};

/// The bundled core lexicon: the very file the Python package reads.
pub const CORE_DICT: &str = include_str!("../../phonetok/data/core.dict");

/// The environment variable that names a full dictionary file.
pub const ENV_LEXICON: &str = "PHONETOK_LEXICON";

pub type Entry = (String, Vec<String>);

/// One line of a CMU-format file as `(word, phones)`; `None` for a comment or a blank.
pub fn parse_entry(line: &str) -> Option<Entry> {
    if line.starts_with(";;;") {
        return None;
    }
    let line = line.split('#').next().unwrap_or("").trim();
    let mut parts = line.split_whitespace();
    let word = parts.next()?.to_lowercase();
    let phones: Vec<String> = parts.map(|p| p.to_uppercase()).collect();
    if phones.is_empty() {
        return None;
    }
    let word = match (word.ends_with(')'), word.rfind('(')) {
        (true, Some(i)) => word[..i].to_string(),
        _ => word,
    };
    Some((word, phones))
}

/// Every entry of a CMU-format text.
pub fn read_entries(text: &str) -> Vec<Entry> {
    text.lines().filter_map(parse_entry).collect()
}

/// The bundled core lexicon's entries.
pub fn core_entries() -> Vec<Entry> {
    read_entries(CORE_DICT)
}

/// A word -> pronunciations map and its inverse.  Every method lower-cases the word it is given.
#[derive(Clone, Debug, Default)]
pub struct Lexicon {
    entries: HashMap<String, Vec<Vec<String>>>,
    order: Vec<String>,
    reverse: Option<HashMap<String, Vec<String>>>,
    loose: Option<HashMap<String, Vec<String>>>,
    pub sources: Vec<String>,
}

impl Lexicon {
    pub fn new() -> Lexicon {
        Lexicon::default()
    }

    /// The bundled core lexicon.
    pub fn core() -> Lexicon {
        let mut lex = Lexicon::new();
        lex.extend(core_entries());
        lex.sources.push("the core lexicon".to_string());
        lex
    }

    /// A CMU-format file.
    pub fn load(path: &str) -> Result<Lexicon, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lex = Lexicon::new();
        lex.extend(read_entries(&text));
        lex.sources.push(format!("file {path}"));
        Ok(lex)
    }

    /// The lexicon every port builds alike: the core, plus the `PHONETOK_LEXICON` file if it is set.
    pub fn portable() -> Result<Lexicon, String> {
        let mut lex = Lexicon::core();
        if let Ok(path) = std::env::var(ENV_LEXICON) {
            if !path.is_empty() {
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                lex.extend(read_entries(&text));
                lex.sources.push(format!("file {path}"));
            }
        }
        Ok(lex)
    }

    /// Adds entries; a pronunciation the word already has is not added twice.  Returns how many were new.
    pub fn extend(&mut self, entries: Vec<Entry>) -> usize {
        let mut added = 0;
        for (word, phones) in entries {
            let word = word.to_lowercase();
            match self.entries.get_mut(&word) {
                None => {
                    self.entries.insert(word.clone(), vec![phones]);
                    self.order.push(word);
                    added += 1;
                }
                Some(known) => {
                    if !known.contains(&phones) {
                        known.push(phones);
                        added += 1;
                    }
                }
            }
        }
        if added > 0 {
            self.reverse = None;
            self.loose = None;
        }
        added
    }

    /// Adds a pronunciation; `first` makes it the one `lookup` returns.
    pub fn add(&mut self, word: &str, phones: &[String], first: bool) -> Result<(), String> {
        check_phones(phones)?;
        let word = word.to_lowercase();
        let phones = phones.to_vec();
        if !self.entries.contains_key(&word) {
            self.order.push(word.clone());
        }
        let known = self.entries.entry(word).or_default();
        known.retain(|k| *k != phones);
        if first {
            known.insert(0, phones);
        } else {
            known.push(phones);
        }
        self.reverse = None;
        self.loose = None;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn has(&self, word: &str) -> bool {
        self.entries.contains_key(&word.to_lowercase())
    }

    /// The first pronunciation of a word.
    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        self.entries
            .get(&word.to_lowercase())
            .and_then(|v| v.first().cloned())
    }

    /// Every pronunciation of a word; empty if it is unknown.
    pub fn pronunciations(&self, word: &str) -> Vec<Vec<String>> {
        self.entries
            .get(&word.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// The words in the order they were first added.
    pub fn words(&self) -> &[String] {
        &self.order
    }

    /// Every (word, pronunciation) pair, alternatives included, in first-added order.
    pub fn items(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        for w in &self.order {
            for p in &self.entries[w] {
                out.push((w.clone(), p.clone()));
            }
        }
        out
    }

    fn build_reverse(&mut self) {
        let mut exact: HashMap<String, Vec<String>> = HashMap::new();
        let mut loose: HashMap<String, Vec<String>> = HashMap::new();
        for w in &self.order {
            for p in &self.entries[w] {
                exact.entry(p.join(" ")).or_default().push(w.clone());
                loose
                    .entry(strip_stress(p).join(" "))
                    .or_default()
                    .push(w.clone());
            }
        }
        self.reverse = Some(exact);
        self.loose = Some(loose);
    }

    /// The words pronounced exactly like `phones` (homophones), in the order they were added;
    /// with `stress` false the stress digits are ignored on both sides.
    pub fn spellings(&mut self, phones: &[String], stress: bool) -> Vec<String> {
        if self.reverse.is_none() {
            self.build_reverse();
        }
        if stress {
            self.reverse
                .as_ref()
                .unwrap()
                .get(&phones.join(" "))
                .cloned()
                .unwrap_or_default()
        } else {
            self.loose
                .as_ref()
                .unwrap()
                .get(&strip_stress(phones).join(" "))
                .cloned()
                .unwrap_or_default()
        }
    }

    pub fn describe(&self) -> String {
        let src = if self.sources.is_empty() {
            "nowhere".to_string()
        } else {
            self.sources.join(", ")
        };
        format!("{} words from {}", self.entries.len(), src)
    }
}
