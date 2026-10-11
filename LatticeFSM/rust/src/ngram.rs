//! Predict the next thing: an N-gram model over the lattice's adaptive edges.
//!
//! The same model as `latticefsm/ngram.py`, number for number: a finite state machine whose states are the last
//! `n - 1` tokens, kept sparse - an edge exists once `(context, token)` has been seen - each edge with its count and,
//! once credited, a record (width, rewarded, punished, last).  The next-token distribution is interpolated
//! Kneser-Ney (the highest order from raw counts, each lower order from continuation counts, Ney's discount per
//! order, a uniform floor), and every credited edge out of a context weighs in by `width ^ stimulation`,
//! renormalised.  Counts are integers and every probability is computed in the Python port's order, so the two
//! ports give the same probabilities, the same greedy texts and the same measurements, and share their files.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use crate::gzip;
use crate::json::{parse, Json};
use crate::rng::Rng;

pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";
pub const UNK: &str = "<unk>";
pub const UNITS: [&str; 4] = ["word", "char", "sound", "base64"];
pub const TEXT_EXTENSIONS: [&str; 7] = [".txt", ".md", ".text", ".jsonl", ".json", ".csv", ".tsv"];
const FORMAT: &str = "latticefsm-ngram";
const WIDTH_MIN: f64 = 0.05;
const WIDTH_MAX: f64 = 20.0;
const NO_SPACE_BEFORE: &str = ".,!?;:)]}%'\"";
const NO_SPACE_AFTER: &str = "([{$\"";

// ---- tokens ------------------------------------------------------------------------------------------------------

thread_local! {
    /// The phonetic tokenizer the `sound` unit reads and spells with: one per thread, no stress, no memory of the
    /// words it has sounded out - a word's sounds never depend on what was read before it (as in the Python port).
    static SOUNDS: std::cell::RefCell<Option<phonetok::Tokenizer>> = const { std::cell::RefCell::new(None) };
}

fn with_sounds<T>(f: impl FnOnce(&mut phonetok::Tokenizer) -> T) -> T {
    SOUNDS.with(|cell| {
        let mut slot = cell.borrow_mut();
        let t = slot.get_or_insert_with(|| {
            let mut t = crate::phonetic::tokenizer(false);
            t.transcriber.remember = false;
            t
        });
        f(t)
    })
}

fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// One line as tokens, as `ngram.tokenize` does: words (an apostrophe between letters inside), every other
/// non-space character alone; or characters (a space as `_`); or the phonetic tokenizer's sounds.
pub fn tokenize(line: &str, unit: &str, lower: bool) -> Vec<String> {
    if unit == "base64" {
        // the letters of the line's UTF-8 in base64, case kept: bytes are bytes
        return crate::b64::encode(line.as_bytes()).chars().map(String::from).collect();
    }
    let line = if lower { line.to_lowercase() } else { line.to_string() };
    match unit {
        "char" => line
            .trim()
            .chars()
            .map(|c| {
                if c.is_whitespace() {
                    "_".to_string()
                } else {
                    c.to_string()
                }
            })
            .collect(),
        "sound" => {
            let known = crate::phonetic::inventory(false);
            // numerals other than 0-9 (², ½, Ⅳ) are dropped: the tokenizer cannot say them
            let line: String = line
                .chars()
                .map(|c| if c.is_numeric() && !c.is_ascii_digit() { ' ' } else { c })
                .collect();
            with_sounds(|t| t.tokens(&line))
                .into_iter()
                .filter(|t| known.contains(t))
                .collect()
        }
        _ => {
            let chars: Vec<char> = line.chars().collect();
            let (mut out, mut i, n) = (Vec::new(), 0, chars.len());
            while i < n {
                let c = chars[i];
                if c.is_whitespace() {
                    i += 1;
                } else if word_char(c) {
                    let mut j = i + 1;
                    while j < n
                        && (word_char(chars[j])
                            || (chars[j] == '\''
                                && j + 1 < n
                                && chars[j + 1].is_alphabetic()
                                && word_char(chars[j - 1])))
                    {
                        j += 1;
                    }
                    out.push(chars[i..j].iter().collect());
                    i = j;
                } else {
                    out.push(c.to_string());
                    i += 1;
                }
            }
            out
        }
    }
}

/// Tokens written back as text: words with spaces (none before punctuation), characters run together, sounds
/// spelled as English.
pub fn detokenize(tokens: &[String], unit: &str) -> String {
    let toks: Vec<&String> = tokens
        .iter()
        .filter(|t| ![BOS, EOS, UNK].contains(&t.as_str()))
        .collect();
    match unit {
        "base64" => {
            let joined: String = toks.iter().map(|t| t.as_str()).collect();
            String::from_utf8_lossy(&crate::b64::decode(&joined)).into_owned()
        }
        "char" => toks
            .iter()
            .map(|t| if t.as_str() == "_" { " " } else { t.as_str() })
            .collect(),
        "sound" => {
            let owned: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
            with_sounds(|t| t.decode(&owned))
        }
        _ => {
            let mut out = String::new();
            for t in toks {
                let before = t.chars().count() == 1 && NO_SPACE_BEFORE.contains(t.as_str());
                let after = out.chars().last().is_some_and(|c| NO_SPACE_AFTER.contains(c));
                if !out.is_empty() && !(before || after) {
                    out.push(' ');
                }
                out.push_str(t);
            }
            out
        }
    }
}

// ---- corpora -----------------------------------------------------------------------------------------------------

/// Lines as Python's `str.splitlines` splits them.
pub fn split_text_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(std::mem::take(&mut cur));
            }
            '\n' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                out.push(std::mem::take(&mut cur))
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn json_text(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        Json::Object(pairs) => {
            for key in ["text", "content", "sentence", "body"] {
                if let Some(Json::String(s)) = v.get(key) {
                    return s.clone();
                }
            }
            pairs
                .iter()
                .filter_map(|(_, x)| x.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        Json::Array(items) => items.iter().map(json_text).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

fn walk(dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if p.is_dir() {
            dirs.push((name, p));
        } else if !name.starts_with('.')
            && name != "README.md"
            && TEXT_EXTENSIONS.iter().any(|x| name.to_lowercase().ends_with(x))
        {
            files.push((name, p));
        }
    }
    files.sort();
    dirs.sort();
    out.extend(files.into_iter().map(|(_, p)| p.to_string_lossy().into_owned()));
    for (_, d) in dirs {
        walk(&d, out)?;
    }
    Ok(())
}

/// Every text file under `paths` as (path, lines), as `ngram.corpus_lines` reads them.
pub fn corpus_lines(paths: &[String]) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut files = Vec::new();
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, &mut files)?;
        } else if path.exists() {
            files.push(p.clone());
        } else {
            return Err(format!("{p}: no such file or directory"));
        }
    }
    let mut out = Vec::new();
    for f in files {
        let raw = fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
        let raw = if f.ends_with(".gz") {
            gzip::decompress(&raw)?
        } else {
            raw
        };
        let text = String::from_utf8_lossy(&raw).into_owned();
        let lower = f.to_lowercase();
        let lines: Vec<String> = if lower.ends_with(".jsonl") {
            let mut v = Vec::new();
            for l in split_text_lines(&text) {
                let l = l.trim();
                if l.is_empty() {
                    continue;
                }
                match parse(l) {
                    Ok(j) => v.extend(split_text_lines(&json_text(&j))),
                    Err(_) => v.push(l.to_string()),
                }
            }
            v
        } else if lower.ends_with(".json") {
            match parse(&text) {
                Ok(j) => split_text_lines(&json_text(&j)),
                Err(_) => split_text_lines(&text),
            }
        } else {
            split_text_lines(&text)
        };
        out.push((
            f,
            lines
                .iter()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
        ));
    }
    Ok(out)
}

/// A sequence of tokens.
pub type Sequence = Vec<String>;

/// Every file under `paths` as (path, token sequences): for `base64` every file of any kind, its bytes 76 letters a
/// sequence; for the other units every text file line by line, each line tokenized, empty ones dropped.
pub fn corpus_sequences(paths: &[String], unit: &str, lower: bool) -> Result<Vec<(String, Vec<Sequence>)>, String> {
    if unit == "base64" {
        return Ok(crate::b64::file_sequences(paths, 76)?
            .into_iter()
            .map(|(p, seqs)| (p, seqs.iter().map(|s| s.chars().map(String::from).collect()).collect()))
            .collect());
    }
    Ok(corpus_lines(paths)?
        .into_iter()
        .map(|(p, lines)| {
            (
                p,
                lines
                    .iter()
                    .map(|l| tokenize(l, unit, lower))
                    .filter(|q| !q.is_empty())
                    .collect(),
            )
        })
        .collect())
}

/// Every `every`-th line held out, the rest to train on.
pub fn split_lines<T: Clone>(lines: &[T], every: usize) -> (Vec<T>, Vec<T>) {
    if every <= 1 {
        return (lines.to_vec(), Vec::new());
    }
    let train = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| i % every != every - 1)
        .map(|(_, l)| l.clone())
        .collect();
    let test = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| i % every == every - 1)
        .map(|(_, l)| l.clone())
        .collect();
    (train, test)
}

// ---- the model ---------------------------------------------------------------------------------------------------

type Row = HashMap<u32, u64>;

/// One order's Kneser-Ney table: counts by context, their totals, how many distinct tokens, the discount.
struct Table {
    rows: HashMap<Vec<u32>, Row>,
    totals: HashMap<Vec<u32>, u64>,
    distinct: HashMap<Vec<u32>, u64>,
    discount: f64,
}

/// What `evaluate` measured.
#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    pub tokens: usize,
    pub perplexity: Option<f64>,
    pub bits: Option<f64>,
    pub accuracy: Option<f64>,
    pub top5: Option<f64>,
    pub ranked: usize,
}

/// An N-gram model of order `order` over `unit` tokens: sparse adaptive edges, interpolated Kneser-Ney.
pub struct NGram {
    pub order: usize,
    pub unit: String,
    pub lower: bool,
    pub stimulation: f64,
    pub reward_widening: f64,
    pub punish_narrowing: f64,
    pub vocab: Vec<String>,
    index: HashMap<String, u32>,
    /// The highest order's counts: context (`order - 1` ids) -> next id -> how many times.
    pub grams: HashMap<Vec<u32>, Row>,
    /// Edges with a record beyond their count: context -> id -> [width, rewarded, punished, last].
    pub edges: HashMap<Vec<u32>, BTreeMap<u32, [f64; 4]>>,
    pub clock: u64,
    pub sentences: u64,
    pub sources: Vec<(String, usize, usize)>,
    pub history: Vec<Json>,
    tables: Option<Vec<Table>>,
}

impl NGram {
    pub fn new(order: usize, unit: &str, lower: bool, stimulation: f64) -> Result<NGram, String> {
        if !(1..=12).contains(&order) {
            return Err(format!("order is 1 to 12, got {order}"));
        }
        if !UNITS.contains(&unit) {
            return Err(format!("unit must be one of {UNITS:?}, got {unit:?}"));
        }
        let vocab: Vec<String> = [BOS, EOS, UNK].iter().map(|s| s.to_string()).collect();
        let index = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        Ok(NGram {
            order,
            unit: unit.to_string(),
            lower,
            stimulation,
            reward_widening: 0.2,
            punish_narrowing: 0.2,
            vocab,
            index,
            grams: HashMap::new(),
            edges: HashMap::new(),
            clock: 0,
            sentences: 0,
            sources: Vec::new(),
            history: Vec::new(),
            tables: None,
        })
    }

    pub fn tokens(&self, text: &str) -> Vec<String> {
        tokenize(text, &self.unit, self.lower)
    }

    fn id(&mut self, token: &str, grow: bool) -> u32 {
        if let Some(&i) = self.index.get(token) {
            return i;
        }
        if !grow {
            return 2;
        }
        let i = self.vocab.len() as u32;
        self.vocab.push(token.to_string());
        self.index.insert(token.to_string(), i);
        i
    }

    fn lookup(&self, token: &str) -> u32 {
        self.index.get(token).copied().unwrap_or(2)
    }

    /// The last `order - 1` tokens as ids (unknown ones as `<unk>`), padded with `<s>`.
    pub fn context(&self, tokens: &[String]) -> Vec<u32> {
        let mut ids = vec![0u32; self.order - 1];
        ids.extend(tokens.iter().map(|t| self.lookup(t)));
        ids[ids.len() - (self.order - 1)..].to_vec()
    }

    // ---- training ------------------------------------------------------------------------------------------------

    /// One sequence: every (context, next) counted, `</s>` last.  Returns how many were counted.
    pub fn train_tokens(&mut self, tokens: &[String]) -> usize {
        let k = self.order - 1;
        let mut ids = vec![0u32; k];
        for t in tokens {
            let i = self.id(t, true);
            ids.push(i);
        }
        ids.push(1);
        for i in k..ids.len() {
            let ctx = ids[i - k..i].to_vec();
            *self.grams.entry(ctx.clone()).or_default().entry(ids[i]).or_insert(0) += 1;
            self.clock += 1;
            if let Some(rec) = self.edges.get_mut(&ctx).and_then(|r| r.get_mut(&ids[i])) {
                rec[3] = self.clock as f64;
            }
        }
        self.sentences += 1;
        self.tables = None;
        ids.len() - k
    }

    /// Every line one sequence; returns the tokens counted.
    pub fn train_lines(&mut self, lines: &[String], source: Option<&str>) -> usize {
        let (mut n, mut count) = (0, 0);
        for line in lines {
            let toks = self.tokens(line);
            if !toks.is_empty() {
                n += self.train_tokens(&toks);
                count += 1;
            }
        }
        if let Some(s) = source {
            self.sources.push((s.to_string(), count, n));
        }
        n
    }

    /// Every sequence of tokens trained on; returns the tokens counted.
    pub fn train_sequences(&mut self, seqs: &[Sequence], source: Option<&str>) -> usize {
        let (mut n, mut count) = (0, 0);
        for toks in seqs {
            if !toks.is_empty() {
                n += self.train_tokens(toks);
                count += 1;
            }
        }
        if let Some(s) = source {
            self.sources.push((s.to_string(), count, n));
        }
        n
    }

    /// What a prompt is read as to be continued: its tokens - for `base64` only the letters whose bits are all known.
    pub fn prompt_tokens(&self, text: &str) -> Sequence {
        if self.unit == "base64" {
            return crate::b64::encode_prefix(text.as_bytes())
                .chars()
                .map(String::from)
                .collect();
        }
        self.tokens(text)
    }

    /// Train on every file under `paths` (`corpus_sequences`), every `held_out`-th sequence kept back and returned.
    pub fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String> {
        let (mut total, mut kept) = (0, Vec::new());
        for (path, lines) in corpus_sequences(paths, &self.unit, self.lower)? {
            let (train, test) = if held_out > 0 {
                split_lines(&lines, held_out)
            } else {
                (lines, Vec::new())
            };
            total += self.train_sequences(&train, Some(&path));
            kept.extend(test);
        }
        Ok((total, kept))
    }

    // ---- Kneser-Ney ----------------------------------------------------------------------------------------------

    fn derive(&mut self) {
        if self.tables.is_some() {
            return;
        }
        let mut tables: Vec<Option<Table>> = (0..=self.order).map(|_| None).collect();
        let mut table = self.grams.clone();
        for k in (1..=self.order).rev() {
            let totals: HashMap<Vec<u32>, u64> = table.iter().map(|(h, r)| (h.clone(), r.values().sum())).collect();
            let distinct: HashMap<Vec<u32>, u64> = table.iter().map(|(h, r)| (h.clone(), r.len() as u64)).collect();
            let (mut n1, mut n2) = (0u64, 0u64);
            for r in table.values() {
                for &c in r.values() {
                    n1 += (c == 1) as u64;
                    n2 += (c == 2) as u64;
                }
            }
            let mut d = if n1 > 0 && n2 > 0 {
                n1 as f64 / (n1 + 2 * n2) as f64
            } else {
                0.75
            };
            if !(d > 0.0 && d < 1.0) {
                d = 0.75;
            }
            let lower = if k > 1 {
                let mut lower: HashMap<Vec<u32>, Row> = HashMap::new();
                for (h, r) in &table {
                    let dst = lower.entry(h[1..].to_vec()).or_default();
                    for &w in r.keys() {
                        *dst.entry(w).or_insert(0) += 1;
                    }
                }
                Some(lower)
            } else {
                None
            };
            tables[k] = Some(Table {
                rows: table,
                totals,
                distinct,
                discount: d,
            });
            if let Some(l) = lower {
                table = l;
            } else {
                table = HashMap::new();
            }
        }
        // index 0 is unused; keep the vector dense for indexing by order
        self.tables = Some(tables.into_iter().map(|t| t.unwrap_or_else(empty_table)).collect());
    }

    fn floor(&self, w: u32) -> f64 {
        if w == 0 {
            0.0
        } else {
            1.0 / (self.vocab.len() - 1) as f64
        }
    }

    /// `P_KN(w | ctx)`: interpolated Kneser-Ney, from the uniform floor up through every order.
    pub fn kn(&mut self, ctx: &[u32], w: u32) -> f64 {
        self.derive();
        let mut p = self.floor(w);
        let tables = self.tables.as_ref().expect("derived");
        for (k, t) in tables.iter().enumerate().skip(1) {
            let h: &[u32] = if k > 1 { &ctx[ctx.len() - (k - 1)..] } else { &[] };
            if let Some(row) = t.rows.get(h) {
                if !row.is_empty() {
                    let tot = t.totals[h] as f64;
                    let c = row.get(&w).copied().unwrap_or(0) as f64;
                    let d = t.discount;
                    p = (c - d).max(0.0) / tot + (d * t.distinct[h] as f64 / tot) * p;
                }
            }
        }
        p
    }

    fn boost(&mut self, ctx: &[u32]) -> Option<(BTreeMap<u32, [f64; 4]>, f64)> {
        let recs = self.edges.get(ctx)?.clone();
        if recs.is_empty() || self.stimulation == 0.0 {
            return None;
        }
        let mut z = 1.0;
        for (&w, rec) in &recs {
            z += self.kn(ctx, w) * (rec[0].powf(self.stimulation) - 1.0);
        }
        Some((recs, z))
    }

    /// `P(w | ctx)`: Kneser-Ney, every credited edge out of `ctx` weighted by its width ^ stimulation.
    pub fn prob(&mut self, ctx: &[u32], w: u32) -> f64 {
        let boost = self.boost(ctx);
        let p = self.kn(ctx, w);
        match boost {
            None => p,
            Some((recs, z)) => {
                let q = match recs.get(&w) {
                    Some(rec) => p * rec[0].powf(self.stimulation),
                    None => p,
                };
                q / z
            }
        }
    }

    /// `P(w | ctx)` for every id (`<s>` has 0), computed order by order as `prob` computes it.
    pub fn distribution(&mut self, ctx: &[u32]) -> Vec<f64> {
        self.derive();
        let floor = 1.0 / (self.vocab.len() - 1) as f64;
        let mut out = vec![floor; self.vocab.len()];
        out[0] = 0.0;
        {
            let tables = self.tables.as_ref().expect("derived");
            for (k, t) in tables.iter().enumerate().skip(1) {
                let h: &[u32] = if k > 1 { &ctx[ctx.len() - (k - 1)..] } else { &[] };
                if let Some(row) = t.rows.get(h) {
                    if !row.is_empty() {
                        let tot = t.totals[h] as f64;
                        let d = t.discount;
                        let f = d * t.distinct[h] as f64 / tot;
                        for p in out.iter_mut() {
                            *p *= f;
                        }
                        for (&w, &c) in row {
                            out[w as usize] += (c as f64 - d).max(0.0) / tot; // x + y == y + x: the same number prob gives
                        }
                    }
                }
            }
        }
        if let Some((recs, z)) = self.boost(ctx) {
            for (w, p) in out.iter_mut().enumerate() {
                *p = match recs.get(&(w as u32)) {
                    Some(rec) => *p * rec[0].powf(self.stimulation) / z,
                    None => *p / z,
                };
            }
        }
        out
    }

    fn ranked(dist: &[f64], k: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (1..dist.len()).collect();
        ids.sort_by(|&a, &b| {
            dist[b]
                .partial_cmp(&dist[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        ids.truncate(k);
        ids
    }

    /// The `k` likeliest next tokens after `text`, likeliest first (ties: the token seen first).
    pub fn predict(&mut self, text: &str, k: usize) -> Vec<(String, f64)> {
        let toks = self.prompt_tokens(text);
        self.predict_tokens(&toks, k)
    }

    pub fn predict_tokens(&mut self, toks: &[String], k: usize) -> Vec<(String, f64)> {
        let ctx = self.context(toks);
        let dist = self.distribution(&ctx);
        NGram::ranked(&dist, k)
            .into_iter()
            .map(|w| (self.vocab[w].clone(), dist[w]))
            .collect()
    }

    /// Tokens continuing `prompt` until `</s>` or `max_tokens`: the likeliest each time at temperature 0, else drawn
    /// from `P ^ (1 / temperature)` (this port's generator).
    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64, rng: &mut Rng) -> Vec<String> {
        let toks = self.prompt_tokens(prompt);
        self.continue_tokens(&toks, max_tokens, temperature, rng)
    }

    /// Tokens continuing the tokens `toks`, as `generate` does (never `<s>` or `<unk>`).
    pub fn continue_tokens(
        &mut self,
        toks: &[String],
        max_tokens: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for _ in 0..max_tokens {
            let all: Vec<String> = toks.iter().chain(out.iter()).cloned().collect();
            let ctx = self.context(&all);
            let mut dist = self.distribution(&ctx);
            dist[0] = 0.0;
            dist[2] = 0.0;
            let w = if temperature <= 0.0 {
                let mut best = 0;
                for i in 1..dist.len() {
                    if dist[i] > dist[best] {
                        best = i;
                    }
                }
                best
            } else {
                let weights: Vec<f64> = dist
                    .iter()
                    .map(|&p| if p > 0.0 { p.powf(1.0 / temperature) } else { 0.0 })
                    .collect();
                let total: f64 = weights.iter().sum();
                let r = rng.random() * total;
                let (mut acc, mut pick) = (0.0, 1);
                for (i, q) in weights.iter().enumerate() {
                    acc += q;
                    if r < acc {
                        pick = i;
                        break;
                    }
                }
                pick
            };
            if w == 1 {
                break;
            }
            out.push(self.vocab[w].clone());
        }
        out
    }

    /// `prompt` continued, written back as text (for `base64`, the prompt's letters and the new ones decoded together).
    pub fn complete(&mut self, prompt: &str, max_tokens: usize, temperature: f64, rng: &mut Rng) -> String {
        let toks = self.generate(prompt, max_tokens, temperature, rng);
        if self.unit == "base64" {
            let mut all = self.prompt_tokens(prompt);
            all.extend(toks);
            return detokenize(&all, &self.unit);
        }
        detokenize(&toks, &self.unit)
    }

    /// A `base64` model continuing `prompt` (the start of a file, or nothing), decoded to bytes; at a learned
    /// sequence's end the next one begins afresh, until `max_tokens` letters or the padding of a file's end.
    pub fn generate_bytes(
        &mut self,
        prompt: &[u8],
        max_tokens: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> Result<Vec<u8>, String> {
        if self.unit != "base64" {
            return Err("bytes come from a base64 model".to_string());
        }
        let start: Sequence = crate::b64::encode_prefix(prompt).chars().map(String::from).collect();
        let mut out = start.clone();
        let mut chunk = start.clone();
        while out.len() - start.len() < max_tokens {
            let more = self.continue_tokens(&chunk, max_tokens - (out.len() - start.len()), temperature, rng);
            if more.is_empty() {
                break;
            }
            let ended = more.iter().any(|t| t == "=");
            out.extend(more);
            if ended {
                break;
            }
            chunk = Vec::new();
        }
        let joined: String = out.iter().map(|t| t.as_str()).collect();
        Ok(crate::b64::decode(&joined))
    }

    // ---- measuring -----------------------------------------------------------------------------------------------

    /// Perplexity, bits a token, and the next-token accuracy (first guess, first five) over the first `max_ranked`
    /// positions, on held-out lines.  Nothing moves.
    pub fn evaluate(&mut self, seqs: &[Sequence], max_ranked: usize) -> Evaluation {
        let (mut logp, mut n, mut ranked, mut top1, mut top5) = (0.0f64, 0usize, 0usize, 0usize, 0usize);
        for toks in seqs {
            if toks.is_empty() {
                continue;
            }
            let mut ids: Vec<u32> = toks.iter().map(|t| self.lookup(t)).collect();
            ids.push(1);
            let mut hist: Vec<String> = Vec::new();
            for (j, &w) in ids.iter().enumerate() {
                let ctx = self.context(&hist);
                let p = self.prob(&ctx, w);
                logp += p.max(1e-300).ln();
                n += 1;
                if ranked < max_ranked {
                    let dist = self.distribution(&ctx);
                    let order = NGram::ranked(&dist, 5);
                    top1 += (order[0] == w as usize) as usize;
                    top5 += order.contains(&(w as usize)) as usize;
                    ranked += 1;
                }
                hist.push(if j < toks.len() {
                    toks[j].clone()
                } else {
                    EOS.to_string()
                });
            }
        }
        if n == 0 {
            return Evaluation::default();
        }
        Evaluation {
            tokens: n,
            perplexity: Some((-logp / n as f64).exp()),
            bits: Some(-logp / n as f64 / 2f64.ln()),
            accuracy: (ranked > 0).then(|| top1 as f64 / ranked as f64),
            top5: (ranked > 0).then(|| top5 as f64 / ranked as f64),
            ranked,
        }
    }

    // ---- credit --------------------------------------------------------------------------------------------------

    /// One edge credited: rewarded and widened, or punished and narrowed, as a lattice edge is.
    pub fn credit(&mut self, ctx: &[u32], w: u32, amount: f64) {
        let clock = self.clock as f64;
        let (rw, pn) = (self.reward_widening, self.punish_narrowing);
        let rec = self
            .edges
            .entry(ctx.to_vec())
            .or_default()
            .entry(w)
            .or_insert([1.0, 0.0, 0.0, clock]);
        if amount > 0.0 {
            rec[0] = WIDTH_MAX.min(rec[0] * (1.0 + rw * amount));
            rec[1] += amount;
        } else if amount < 0.0 {
            rec[0] = WIDTH_MIN.max(rec[0] / (1.0 + pn * -amount));
            rec[2] -= amount;
        }
    }

    /// Every edge a text takes, `</s>` included, credited `amount`.  Returns how many.
    pub fn credit_text(&mut self, text: &str, amount: f64) -> usize {
        let mut toks = self.tokens(text);
        toks.push(EOS.to_string());
        let mut hist: Vec<String> = Vec::new();
        let mut n = 0;
        let last = toks.len() - 1;
        for (j, t) in toks.iter().enumerate() {
            let w = if j == last { 1 } else { self.lookup(t) };
            if w != 2 {
                let ctx = self.context(&hist);
                self.credit(&ctx, w, amount);
                n += 1;
            }
            hist.push(t.clone());
        }
        n
    }

    // ---- describing ----------------------------------------------------------------------------------------------

    pub fn n_grams(&self) -> usize {
        self.grams.values().map(|r| r.len()).sum()
    }

    pub fn credited_edges(&self) -> usize {
        self.edges.values().map(|r| r.len()).sum()
    }

    pub fn discounts(&mut self) -> Vec<f64> {
        self.derive();
        self.tables
            .as_ref()
            .expect("derived")
            .iter()
            .skip(1)
            .map(|t| t.discount)
            .collect()
    }

    pub fn stats(&mut self) -> Json {
        let discounts = self.discounts();
        Json::object()
            .with("order", self.order.into())
            .with("unit", self.unit.as_str().into())
            .with("lower", self.lower.into())
            .with("vocab", (self.vocab.len() - 3).into())
            .with("contexts", self.grams.len().into())
            .with("ngrams", self.n_grams().into())
            .with("tokens", (self.clock as f64).into())
            .with("sentences", (self.sentences as f64).into())
            .with("stimulation", self.stimulation.into())
            .with("credited_edges", self.credited_edges().into())
            .with("discounts", Json::numbers(&discounts))
            .with("sources", self.sources.len().into())
    }

    // ---- the file ------------------------------------------------------------------------------------------------

    pub fn to_json(&self) -> Json {
        let mut grams: Vec<Vec<u64>> = self
            .grams
            .iter()
            .flat_map(|(ctx, r)| {
                r.iter().map(move |(&w, &c)| {
                    let mut v: Vec<u64> = ctx.iter().map(|&x| x as u64).collect();
                    v.push(w as u64);
                    v.push(c);
                    v
                })
            })
            .collect();
        grams.sort();
        let mut edges: Vec<Vec<f64>> = self
            .edges
            .iter()
            .flat_map(|(ctx, r)| {
                r.iter().map(move |(&w, rec)| {
                    let mut v: Vec<f64> = ctx.iter().map(|&x| x as f64).collect();
                    v.push(w as f64);
                    v.extend_from_slice(rec);
                    v
                })
            })
            .collect();
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Json::object()
            .with("format", FORMAT.into())
            .with("version", 1.0.into())
            .with("order", self.order.into())
            .with("unit", self.unit.as_str().into())
            .with("lower", self.lower.into())
            .with("stimulation", self.stimulation.into())
            .with("reward_widening", self.reward_widening.into())
            .with("punish_narrowing", self.punish_narrowing.into())
            .with("clock", (self.clock as f64).into())
            .with("sentences", (self.sentences as f64).into())
            .with("vocab", Json::strings(&self.vocab))
            .with(
                "grams",
                Json::Array(
                    grams
                        .iter()
                        .map(|g| Json::numbers(&g.iter().map(|&x| x as f64).collect::<Vec<_>>()))
                        .collect(),
                ),
            )
            .with("edges", Json::Array(edges.iter().map(|e| Json::numbers(e)).collect()))
            .with(
                "sources",
                Json::Array(
                    self.sources
                        .iter()
                        .map(|(s, c, n)| Json::Array(vec![s.as_str().into(), (*c).into(), (*n).into()]))
                        .collect(),
                ),
            )
            .with("history", Json::Array(self.history.clone()))
    }

    pub fn from_json(v: &Json) -> Result<NGram, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let mut m = NGram::new(
            v.num("order", 3.0) as usize,
            v.str_or("unit", "word"),
            v.bool_or("lower", true),
            v.num("stimulation", 1.0),
        )?;
        m.reward_widening = v.num("reward_widening", 0.2);
        m.punish_narrowing = v.num("punish_narrowing", 0.2);
        m.vocab = v
            .get("vocab")
            .and_then(Json::as_array)
            .ok_or("no vocab")?
            .iter()
            .map(|s| s.as_str().map(String::from).ok_or("the vocab is strings"))
            .collect::<Result<_, _>>()?;
        if m.vocab.len() < 3 || m.vocab[0] != BOS || m.vocab[1] != EOS || m.vocab[2] != UNK {
            return Err("the vocabulary does not start <s> </s> <unk>".to_string());
        }
        m.index = m.vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let k = m.order - 1;
        let nums = |row: &Json| -> Vec<f64> {
            row.as_array()
                .map(|a| a.iter().filter_map(Json::as_f64).collect())
                .unwrap_or_default()
        };
        for row in v
            .get("grams")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let r = nums(row);
            if r.len() != k + 2 {
                return Err("a gram row of the wrong length".to_string());
            }
            let ctx: Vec<u32> = r[..k].iter().map(|&x| x as u32).collect();
            m.grams.entry(ctx).or_default().insert(r[k] as u32, r[k + 1] as u64);
        }
        for row in v
            .get("edges")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let r = nums(row);
            if r.len() != k + 5 {
                return Err("an edge row of the wrong length".to_string());
            }
            let ctx: Vec<u32> = r[..k].iter().map(|&x| x as u32).collect();
            m.edges
                .entry(ctx)
                .or_default()
                .insert(r[k] as u32, [r[k + 1], r[k + 2], r[k + 3], r[k + 4]]);
        }
        m.clock = v.num("clock", 0.0) as u64;
        m.sentences = v.num("sentences", 0.0) as u64;
        for s in v
            .get("sources")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if let Some(a) = s.as_array() {
                if a.len() == 3 {
                    m.sources.push((
                        a[0].as_str().unwrap_or("").to_string(),
                        a[1].as_f64().unwrap_or(0.0) as usize,
                        a[2].as_f64().unwrap_or(0.0) as usize,
                    ));
                }
            }
        }
        m.history = v.get("history").and_then(Json::as_array).cloned().unwrap_or_default();
        Ok(m)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let mut f = fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

fn empty_table() -> Table {
    Table {
        rows: HashMap::new(),
        totals: HashMap::new(),
        distinct: HashMap::new(),
        discount: 0.75,
    }
}

pub fn load_ngram(path: &str) -> Result<NGram, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if path.ends_with(".gz") || bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    NGram::from_json(&parse(&text)?)
}
