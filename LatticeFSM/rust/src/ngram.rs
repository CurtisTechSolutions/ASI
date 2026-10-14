//! Predict the next thing: an N-gram model over the lattice's adaptive edges.
//!
//! The same model as `latticefsm/ngram.py`, number for number: a finite state machine whose states are the last
//! `n - 1` tokens, kept sparse - an edge exists once `(context, token)` has been seen - each edge with its count and,
//! once credited, a record (width, rewarded, punished, last).  The next-token distribution is interpolated
//! Kneser-Ney (the highest order from raw counts, each lower order from continuation counts, Ney's discount per
//! order, a uniform floor), and every credited edge out of a context weighs in by `width ^ stimulation`,
//! renormalised.  Counts are integers and every probability is computed in the Python port's order, so the two
//! ports give the same probabilities, the same greedy texts and the same measurements, and share their files.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};

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
                if !(out.is_empty() || before || after) {
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

/// Every text document under `paths` as (label, lines): files, directories walked in name order, and the members
/// of archives (`corpus.rs`: text, `.jsonl` / `.json`, HTML / XML, docx / odt / epub, zip, tar, gz), its non-empty
/// lines, trimmed.
pub fn corpus_lines(paths: &[String]) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    crate::corpus::text_documents(paths, &mut |_, label, kind, r| {
        let mut lines = Vec::new();
        crate::corpus::text_lines(kind, r, &mut |l| {
            lines.push(l);
            Ok(())
        })?;
        out.push((label.to_string(), lines));
        Ok(())
    })?;
    Ok(out)
}

/// A sequence of tokens.
pub type Sequence = Vec<String>;

/// Every sequence under `paths`, one at a time, handed to `f(path, label, tokens)`: for `base64` every file's bytes
/// (archives' members') 57 bytes - 76 letters - a sequence; for the other units every non-empty line of every text
/// document, tokenized in bounded batches on the CPU workers. Callbacks remain in
/// corpus order; the whole corpus is never held in memory.
pub fn corpus_stream(
    paths: &[String],
    unit: &str,
    lower: bool,
    f: &mut dyn FnMut(&str, &str, Sequence) -> Result<(), String>,
) -> Result<(), String> {
    if unit == "base64" {
        return crate::corpus::byte_documents(paths, &mut |path, label, r| {
            crate::corpus::read_chunks(r, crate::b64::CHUNK_BYTES, &mut |chunk| {
                f(
                    path,
                    label,
                    crate::b64::encode(chunk).chars().map(String::from).collect(),
                )
            })
        });
    }
    crate::corpus::text_documents(paths, &mut |path, label, kind, r| {
        let mut lines = Vec::new();
        let mut bytes = 0usize;
        let mut flush = |lines: &mut Vec<String>| -> Result<(), String> {
            let sequences = tokenize_lines(lines, unit, lower);
            lines.clear();
            for toks in sequences.into_iter().filter(|t| !t.is_empty()) {
                f(path, label, toks)?;
            }
            Ok(())
        };
        crate::corpus::text_lines(kind, r, &mut |line| {
            bytes += line.len();
            lines.push(line);
            if lines.len() >= 256 || bytes >= 1 << 20 {
                flush(&mut lines)?;
                bytes = 0;
            }
            Ok(())
        })?;
        flush(&mut lines)
    })
}

/// Tokenize independent lines concurrently, retaining their original order.
pub(crate) fn tokenize_lines(lines: &[String], unit: &str, lower: bool) -> Vec<Sequence> {
    let workers = if lines.iter().map(String::len).sum::<usize>() < 8192 {
        1
    } else {
        0
    };
    crate::parallel::parallel_map(lines, workers, |line| tokenize(line, unit, lower))
}

/// Every document under `paths` as (label, token sequences): `corpus_stream`, gathered.
pub fn corpus_sequences(paths: &[String], unit: &str, lower: bool) -> Result<Vec<(String, Vec<Sequence>)>, String> {
    let mut out: Vec<(String, Vec<Sequence>)> = Vec::new();
    corpus_stream(paths, unit, lower, &mut |_, label, toks| {
        match out.last_mut() {
            Some((l, seqs)) if l == label => seqs.push(toks),
            _ => out.push((label.to_string(), vec![toks])),
        }
        Ok(())
    })?;
    Ok(out)
}

/// Predicting while training: every token of the first this many since the last `reset_online`, then a thinning
/// sample - every 2nd of the next as many, every 3rd of the next, ... - as `ngram.PREDICT_ALL`.
pub const PREDICT_ALL: u64 = 50_000;

/// Whether the `seen`-th position (from 0) since the last reset is predicted while training.
pub fn predict_now(seen: u64) -> bool {
    seen % (1 + seen / PREDICT_ALL) == 0
}

/// How much of the goal teaching until N of N stops at, by default: 95% of the tokens (of the most possible, when
/// training), so it need not be exact.
pub const THRESHOLD: f64 = 0.95;

/// The right guesses that meet a goal of `n` at `threshold` (a share; `1`: all of them).
pub fn goal_of(threshold: f64, n: usize) -> usize {
    if threshold >= 1.0 {
        return n;
    }
    (threshold * n as f64 - 1e-9).ceil().max(0.0) as usize
}

/// Held-out sequences kept at most while training on a corpus; past it, every sequence is trained on.
pub const MAX_HELD_OUT: usize = 10000;

/// Stream every sequence under `paths` into `learn(tokens) -> counted`, every `held_out`-th sequence of each document
/// kept back instead (up to `MAX_HELD_OUT`); one (path, sequences, counted) pushed onto `sources` per file (an
/// archive is one).  Returns (counted, the held-out sequences).
pub fn train_corpus(
    paths: &[String],
    unit: &str,
    lower: bool,
    held_out: usize,
    learn: &mut dyn FnMut(&[String]) -> usize,
    sources: &mut Vec<(String, usize, usize)>,
) -> Result<(usize, Vec<Sequence>), String> {
    let (mut total, mut kept) = (0usize, Vec::new());
    let (mut path, mut label): (Option<String>, Option<String>) = (None, None);
    let (mut count, mut n, mut index) = (0usize, 0usize, 0usize);
    crate::job::phase("training", 0);
    corpus_stream(paths, unit, lower, &mut |p, lab, toks| {
        if crate::job::stopped() {
            return Err(crate::job::STOPPED.to_string());
        }
        crate::job::tick(lab, toks.len());
        if path.as_deref() != Some(p) {
            if let Some(old) = path.take() {
                sources.push((old, count, n));
            }
            path = Some(p.to_string());
            count = 0;
            n = 0;
        }
        if label.as_deref() != Some(lab) {
            label = Some(lab.to_string());
            index = 0;
        }
        if held_out > 1 && index % held_out == held_out - 1 && kept.len() < MAX_HELD_OUT {
            kept.push(toks);
        } else {
            let c = learn(&toks);
            n += c;
            total += c;
            count += 1;
        }
        index += 1;
        Ok(())
    })?;
    if let Some(old) = path {
        sources.push((old, count, n));
    }
    Ok((total, kept))
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

/// One order's Kneser-Ney table, kept current a count at a time: counts by context, their totals, how many distinct
/// tokens each context has, and the count-of-counts the discount is read from.
struct Order {
    rows: HashMap<Vec<u32>, Row>,
    totals: HashMap<Vec<u32>, u64>,
    distinct: HashMap<Vec<u32>, u64>,
    n1: u64,
    n2: u64,
}

impl Order {
    fn new(rows: HashMap<Vec<u32>, Row>) -> Order {
        let totals = rows.iter().map(|(h, r)| (h.clone(), r.values().sum())).collect();
        let distinct = rows.iter().map(|(h, r)| (h.clone(), r.len() as u64)).collect();
        let (mut n1, mut n2) = (0u64, 0u64);
        for r in rows.values() {
            for &c in r.values() {
                n1 += (c == 1) as u64;
                n2 += (c == 2) as u64;
            }
        }
        Order {
            rows,
            totals,
            distinct,
            n1,
            n2,
        }
    }

    /// Ney's `n1 / (n1 + 2 n2)`; 0.75 when that is undefined or outside (0, 1).
    fn discount(&self) -> f64 {
        let d = if self.n1 > 0 && self.n2 > 0 {
            self.n1 as f64 / (self.n1 + 2 * self.n2) as f64
        } else {
            0.75
        };
        if d > 0.0 && d < 1.0 {
            d
        } else {
            0.75
        }
    }
}

/// One more `(h, w)` in order `k`'s table; a pair new to order `k` is one more continuation for order `k - 1`.  The
/// order-1 buckets (count -> ids) follow when they exist.
fn add_to(tables: &mut [Order], ranks: &mut Option<BTreeMap<u64, BTreeSet<u32>>>, k: usize, h: &[u32], w: u32) {
    let t = &mut tables[k];
    let row = t.rows.entry(h.to_vec()).or_default();
    let old = *row.get(&w).unwrap_or(&0);
    row.insert(w, old + 1);
    *t.totals.entry(h.to_vec()).or_insert(0) += 1;
    if old == 0 {
        *t.distinct.entry(h.to_vec()).or_insert(0) += 1;
    }
    t.n1 = t.n1 + (old + 1 == 1) as u64 - (old == 1) as u64;
    t.n2 = t.n2 + (old + 1 == 2) as u64 - (old == 2) as u64;
    if k == 1 {
        if let Some(r) = ranks.as_mut() {
            if old > 0 {
                if let Some(b) = r.get_mut(&old) {
                    b.remove(&w);
                    if b.is_empty() {
                        r.remove(&old);
                    }
                }
            }
            r.entry(old + 1).or_default().insert(w);
        }
    }
    if old == 0 && k > 1 {
        add_to(tables, ranks, k - 1, &h[1..], w);
    }
}

/// What `teach` did: the reply's tokens (and its end) guessed right after it, of how many, in how many rounds,
/// whether all were, and the right guesses before each round and after the last.
#[derive(Clone, Debug, PartialEq)]
pub struct Teaching {
    pub right: usize,
    pub of: usize,
    pub need: usize,
    pub rounds: usize,
    pub reached: bool,
    pub history: Vec<usize>,
}

impl Teaching {
    pub fn to_json(&self) -> crate::json::Json {
        crate::json::Json::object()
            .with("right", self.right.into())
            .with("of", self.of.into())
            .with("need", self.need.into())
            .with("rounds", self.rounds.into())
            .with("reached", self.reached.into())
            .with(
                "history",
                crate::json::Json::numbers(&self.history.iter().map(|&x| x as f64).collect::<Vec<_>>()),
            )
    }
}

/// What `drill` did: the first guesses right after each round, the last of them, of how many, the most possible, and
/// whether that was reached.
#[derive(Clone, Debug, PartialEq)]
pub struct Drill {
    pub rounds: Vec<usize>,
    pub right: usize,
    pub of: usize,
    pub best: usize,
    pub need: usize,
    pub reached: bool,
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
    /// Train by first predicting every token from what has been learned so far (`train_tokens`); not saved.
    pub predict: bool,
    /// While predicting: reward a right first guess and punish a wrong one by this much; not saved.
    pub self_credit: f64,
    /// With `self_credit`: reward right guesses too (the default), or only punish wrong ones.
    pub self_reward: bool,
    /// While predicting: [right first guesses, tokens predicted, sum of ln P(true token)].
    pub online: (u64, u64, f64),
    /// Positions considered for predicting since the last `reset_online`.
    pub seen: u64,
    tables: Option<Vec<Order>>,
    ranks: Option<BTreeMap<u64, BTreeSet<u32>>>,
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
            predict: false,
            self_credit: 0.0,
            self_reward: true,
            online: (0, 0, 0.0),
            seen: 0,
            tables: None,
            ranks: None,
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
    ///
    /// With `predict`, every token is first predicted from what has been learned so far - the first guess (`top`) and
    /// the probability given to the true token are added to `online` - and then learned; with `self_credit` as well,
    /// the guess is rewarded when right and punished when wrong before the true token is counted.
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
            let w = ids[i];
            let sample = self.predict && predict_now(self.seen);
            if self.predict {
                self.seen += 1;
            }
            if sample {
                let guess = self.top(&ctx);
                let p = self.prob(&ctx, w);
                self.online.0 += (guess == w) as u64;
                self.online.1 += 1;
                self.online.2 += p.max(1e-300).ln();
                let c = self.self_credit;
                if c != 0.0 && (self.self_reward || guess != w) {
                    if guess == w {
                        self.credit(&ctx, w, c);
                    } else {
                        self.credit(&ctx, guess, -c);
                    }
                }
            }
            *self.grams.entry(ctx.clone()).or_default().entry(w).or_insert(0) += 1;
            if let Some(tables) = self.tables.as_mut() {
                let n = self.order;
                add_to(tables, &mut self.ranks, n, &ctx, w);
            }
            self.clock += 1;
            if let Some(rec) = self.edges.get_mut(&ctx).and_then(|r| r.get_mut(&w)) {
                rec[3] = self.clock as f64;
            }
        }
        self.sentences += 1;
        ids.len() - k
    }

    /// What predicting while training measured since the last reset: (tokens predicted, first guesses right as a
    /// share, perplexity, bits a token); and a fresh start.
    pub fn reset_online(&mut self) -> (u64, Option<f64>, Option<f64>, Option<f64>) {
        let (right, n, logp) = self.online;
        self.online = (0, 0, 0.0);
        self.seen = 0;
        if n == 0 {
            return (0, None, None, None);
        }
        let nf = n as f64;
        (
            n,
            Some(right as f64 / nf),
            Some((-logp / nf).exp()),
            Some(-logp / nf / 2f64.ln()),
        )
    }

    /// The model's first guess after `ctx`: the likeliest next id (ties: the lower id; never `<s>`), found without the
    /// whole distribution - a token in none of the context's rows above order 1 and not credited there has
    /// `P = Λ · p_1` with one `Λ` for all of them, so the order-1 ranking decides among those.
    pub fn top(&mut self, ctx: &[u32]) -> u32 {
        self.derive();
        if self.ranks.is_none() {
            let mut r: BTreeMap<u64, BTreeSet<u32>> = BTreeMap::new();
            if let Some(row) = self.tables.as_ref().expect("derived")[1].rows.get(&Vec::new()) {
                for (&w, &c) in row {
                    r.entry(c).or_default().insert(w);
                }
            }
            self.ranks = Some(r);
        }
        let floor = 1.0 / (self.vocab.len() - 1) as f64;
        let mut finalists: Vec<u32> = Vec::new();
        {
            // the levels this context reaches, lowest order first, each with its row and the constants `kn` uses
            struct Level<'a> {
                row: &'a Row,
                tot: f64,
                d: f64,
                lam: f64,
            }
            let tables = self.tables.as_ref().expect("derived");
            let mut levels: Vec<Level> = Vec::with_capacity(self.order);
            for (k, t) in tables.iter().enumerate().skip(1) {
                let h: &[u32] = if k > 1 { &ctx[ctx.len() - (k - 1)..] } else { &[] };
                if let Some(row) = t.rows.get(h) {
                    if !row.is_empty() {
                        let tot = t.totals[h] as f64;
                        let d = t.discount();
                        levels.push(Level {
                            row,
                            tot,
                            d,
                            lam: d * t.distinct[h] as f64 / tot,
                        });
                    }
                }
            }
            // a token seen at level j (and none above it) ends as its Kneser-Ney value there times the same factor
            // for every such token, so only the best of each level (and its near-ties) can win
            let partial = |w: u32, upto: usize| -> f64 {
                let mut p = floor;
                for lv in &levels[..=upto] {
                    let c = lv.row.get(&w).copied().unwrap_or(0) as f64;
                    p = (c - lv.d).max(0.0) / lv.tot + lv.lam * p;
                }
                p
            };
            // credited words are weighed by their widths, so they are scored exactly below, never ranked here
            let credited = if self.stimulation != 0.0 {
                self.edges.get(ctx)
            } else {
                None
            };
            let mut scored: Vec<(u32, f64)> = Vec::new();
            for j in (1..levels.len()).rev() {
                scored.clear();
                let mut best = f64::NEG_INFINITY;
                for &w in levels[j].row.keys() {
                    if w == 0
                        || levels[j + 1..].iter().any(|l| l.row.contains_key(&w))
                        || credited.is_some_and(|r| r.contains_key(&w))
                    {
                        continue;
                    }
                    let v = partial(w, j);
                    if v > best {
                        best = v;
                    }
                    scored.push((w, v));
                }
                let near = best - best.abs() * 1e-9;
                finalists.extend(scored.iter().filter(|(_, v)| *v >= near).map(|(w, _)| *w));
            }
            // the best of the rest by the order-1 ranking: not in a row above it, nor credited (scored exactly below)
            let in_rows = |w: u32| {
                levels.iter().skip(1).any(|l| l.row.contains_key(&w)) || credited.is_some_and(|r| r.contains_key(&w))
            };
            let mut outside = None;
            'search: for (_, bucket) in self.ranks.as_ref().expect("built").iter().rev() {
                for &w in bucket {
                    if w != 0 && !in_rows(w) {
                        outside = Some(w);
                        break 'search;
                    }
                }
            }
            if outside.is_none() {
                let seen = tables[1].rows.get(&Vec::new());
                outside =
                    (1..self.vocab.len() as u32).find(|&w| !in_rows(w) && !seen.is_some_and(|r| r.contains_key(&w)));
            }
            finalists.extend(outside);
        }
        if self.stimulation != 0.0 {
            if let Some(recs) = self.edges.get(ctx) {
                finalists.extend(recs.keys().filter(|&&w| w != 0));
            }
        }
        finalists.sort_unstable();
        finalists.dedup();
        let mut best = (0u32, f64::NEG_INFINITY);
        for w in finalists {
            let p = self.prob(ctx, w);
            if p > best.1 {
                best = (w, p);
            }
        }
        best.0
    }

    /// Every context of `seqs` with what follows it and how often, and how many tokens there are.
    fn targets(&self, seqs: &[Sequence]) -> (HashMap<Vec<u32>, HashMap<u32, usize>>, usize) {
        let mut counts: HashMap<Vec<u32>, HashMap<u32, usize>> = HashMap::new();
        let mut n = 0;
        for seq in seqs {
            for j in 0..=seq.len() {
                let w = if j == seq.len() { 1 } else { self.lookup(&seq[j]) };
                *counts.entry(self.context(&seq[..j])).or_default().entry(w).or_insert(0) += 1;
                n += 1;
            }
        }
        (counts, n)
    }

    /// How many tokens of `seqs` (each sequence and its `</s>`) any first guess could get right, of how many: in every
    /// context only the token that follows it most often.
    pub fn ceiling(&self, seqs: &[Sequence]) -> (usize, usize) {
        let (counts, n) = self.targets(seqs);
        (counts.values().map(|r| r.values().copied().max().unwrap_or(0)).sum(), n)
    }

    /// Correct the model on `seqs` (already learned) round after round until its first guess in every context is the
    /// token that follows it most often, or `max_rounds`, or `patience` rounds without a gain - as `NGram.drill`.
    pub fn drill(&mut self, seqs: &[Sequence], max_rounds: usize, patience: usize, threshold: f64) -> Drill {
        let (counts, n) = self.targets(seqs);
        let mut targets: Vec<(Vec<u32>, u32)> = counts
            .iter()
            .map(|(c, r)| {
                let t = r
                    .iter()
                    .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                    .map(|(&w, _)| w)
                    .unwrap_or(1);
                (c.clone(), t)
            })
            .collect();
        targets.sort();
        let best_possible: usize = counts.values().map(|r| r.values().copied().max().unwrap_or(0)).sum();
        let need = goal_of(threshold, best_possible);
        let mut streak: HashMap<Vec<u32>, i32> = HashMap::new();
        let mut rounds = Vec::new();
        let (mut best, mut stale) = (None, 0);
        loop {
            let mut right = 0;
            for seq in seqs {
                for j in 0..=seq.len() {
                    let w = if j == seq.len() { 1 } else { self.lookup(&seq[j]) };
                    let c = self.context(&seq[..j]);
                    right += (self.top(&c) == w) as usize;
                }
            }
            rounds.push(right);
            crate::job::phase("drilling", rounds.len() as u64);
            if best.is_none_or(|b| right > b) {
                best = Some(right);
                stale = 0;
            } else {
                stale += 1;
            }
            if right >= need || rounds.len() >= max_rounds || stale >= patience || crate::job::stopped() {
                return Drill {
                    right,
                    of: n,
                    best: best_possible,
                    need,
                    reached: right >= need,
                    rounds,
                };
            }
            for (c, t) in &targets {
                let g = self.top(c);
                if g != *t {
                    let k = streak.entry(c.clone()).or_insert(0);
                    let a = 2f64.powi(*k);
                    *k += 1;
                    if g != 2 {
                        self.credit(c, g, -a);
                    }
                    self.credit(c, *t, a);
                }
            }
        }
    }

    /// Every line one sequence; returns the tokens counted.
    pub fn train_lines(&mut self, lines: &[String], source: Option<&str>) -> usize {
        let mut total = 0;
        let mut count = 0;
        for batch in lines.chunks(256) {
            for toks in tokenize_lines(batch, &self.unit, self.lower) {
                if !toks.is_empty() {
                    total += self.train_tokens(&toks);
                    count += 1;
                }
            }
        }
        if let Some(s) = source {
            self.sources.push((s.to_string(), count, total));
        }
        total
    }

    /// Every sequence of tokens trained on; returns the tokens counted.
    pub fn train_sequences(&mut self, seqs: &[Sequence], source: Option<&str>) -> usize {
        let (mut n, mut count) = (0, 0);
        for toks in seqs {
            if !toks.is_empty() {
                if crate::job::stopped() {
                    break;
                }
                n += self.train_tokens(toks);
                crate::job::tick("", toks.len());
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

    /// Train on every sequence under `paths`, streamed (`train_corpus`), every `held_out`-th sequence of each
    /// document kept back and returned.
    pub fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String> {
        let (unit, lower) = (self.unit.clone(), self.lower);
        let mut sources = std::mem::take(&mut self.sources);
        let r = train_corpus(
            paths,
            &unit,
            lower,
            held_out,
            &mut |t| self.train_tokens(t),
            &mut sources,
        );
        self.sources = sources;
        r
    }

    // ---- Kneser-Ney ----------------------------------------------------------------------------------------------

    fn derive(&mut self) {
        if self.tables.is_some() {
            return;
        }
        let mut tables: Vec<Option<Order>> = (0..=self.order).map(|_| None).collect();
        let mut table = self.grams.clone();
        for k in (1..=self.order).rev() {
            let lower = (k > 1).then(|| {
                let mut lower: HashMap<Vec<u32>, Row> = HashMap::new();
                for (h, r) in &table {
                    let dst = lower.entry(h[1..].to_vec()).or_default();
                    for &w in r.keys() {
                        *dst.entry(w).or_insert(0) += 1;
                    }
                }
                lower
            });
            tables[k] = Some(Order::new(table));
            table = lower.unwrap_or_default();
        }
        // index 0 is unused; keep the vector dense for indexing by order
        self.tables = Some(
            tables
                .into_iter()
                .map(|t| t.unwrap_or_else(|| Order::new(HashMap::new())))
                .collect(),
        );
        self.ranks = None;
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
        self.kn_ready(ctx, w)
    }

    // Derived tables are immutable for the duration of a measurement.
    fn kn_ready(&self, ctx: &[u32], w: u32) -> f64 {
        let mut p = self.floor(w);
        let tables = self.tables.as_ref().expect("derived");
        for (k, t) in tables.iter().enumerate().skip(1) {
            let h: &[u32] = if k > 1 { &ctx[ctx.len() - (k - 1)..] } else { &[] };
            if let Some(row) = t.rows.get(h) {
                if !row.is_empty() {
                    let tot = t.totals[h] as f64;
                    let c = row.get(&w).copied().unwrap_or(0) as f64;
                    let d = t.discount();
                    p = (c - d).max(0.0) / tot + (d * t.distinct[h] as f64 / tot) * p;
                }
            }
        }
        p
    }

    fn boost(&self, ctx: &[u32]) -> Option<(BTreeMap<u32, [f64; 4]>, f64)> {
        let recs = self.edges.get(ctx)?.clone();
        if recs.is_empty() || self.stimulation == 0.0 {
            return None;
        }
        let mut z = 1.0;
        for (&w, rec) in &recs {
            z += self.kn_ready(ctx, w) * (rec[0].powf(self.stimulation) - 1.0);
        }
        Some((recs, z))
    }

    /// `P(w | ctx)`: Kneser-Ney, every credited edge out of `ctx` weighted by its width ^ stimulation.
    pub fn prob(&mut self, ctx: &[u32], w: u32) -> f64 {
        self.derive();
        self.prob_ready(ctx, w)
    }

    fn prob_ready(&self, ctx: &[u32], w: u32) -> f64 {
        let boost = self.boost(ctx);
        let p = self.kn_ready(ctx, w);
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
        self.distribution_ready(ctx)
    }

    fn distribution_ready(&self, ctx: &[u32]) -> Vec<f64> {
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
                        let d = t.discount();
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
        self.derive();
        let (mut logp, mut n, mut ranked, mut top1, mut top5) = (0.0f64, 0usize, 0usize, 0usize, 0usize);
        for batch in seqs.chunks(256) {
            let mut offset = n;
            let jobs: Vec<_> = batch
                .iter()
                .map(|toks| {
                    let start = offset;
                    if !toks.is_empty() {
                        offset += toks.len() + 1;
                    }
                    (toks, start)
                })
                .collect();
            let workers = if offset - n < 256 { 1 } else { 0 };
            let measured = crate::parallel::parallel_map(&jobs, workers, |&(toks, start)| {
                if toks.is_empty() {
                    return Vec::new();
                }
                let mut ids: Vec<u32> = toks.iter().map(|t| self.lookup(t)).collect();
                ids.push(1);
                let mut hist: Vec<String> = Vec::new();
                ids.iter()
                    .enumerate()
                    .map(|(j, &w)| {
                        let ctx = self.context(&hist);
                        let logp = self.prob_ready(&ctx, w).max(1e-300).ln();
                        let (first, five) = if start + j < max_ranked {
                            let dist = self.distribution_ready(&ctx);
                            let order = NGram::ranked(&dist, 5);
                            (order[0] == w as usize, order.contains(&(w as usize)))
                        } else {
                            (false, false)
                        };
                        hist.push(if j < toks.len() {
                            toks[j].clone()
                        } else {
                            EOS.to_string()
                        });
                        (logp, first, five)
                    })
                    .collect::<Vec<_>>()
            });
            // Fold individual positions in corpus order, retaining the exact
            // floating-point sum and the global max_ranked cutoff.
            for (p, first, five) in measured.into_iter().flatten() {
                logp += p;
                n += 1;
                if ranked < max_ranked {
                    top1 += first as usize;
                    top5 += five as usize;
                    ranked += 1;
                }
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
    /// Every token of `toks` and the `</s>` after it, with the probability the model gives it as it reads them.
    pub fn score_tokens(&mut self, toks: &[String]) -> Vec<(String, f64)> {
        (0..=toks.len())
            .map(|j| {
                let w = if j == toks.len() { 1 } else { self.lookup(&toks[j]) };
                let c = self.context(&toks[..j]);
                let t = if j == toks.len() {
                    EOS.to_string()
                } else {
                    toks[j].clone()
                };
                (t, self.prob(&c, w))
            })
            .collect()
    }

    /// Teach the model that `reply` follows `text` (tokens), round after round, until its first guess is right on every
    /// token of the reply and on the `</s>` after it - its greedy continuation of `text` is then `reply` - or
    /// `max_rounds` rounds have passed.  A round narrows every wrong guess's edge (credit -1 the first round a position
    /// is wrong, doubling every round it stays wrong) and then learns `text + reply` as one sequence, as
    /// `NGram.teach` does.
    pub fn teach(&mut self, text: &[String], reply: &[String], max_rounds: usize, threshold: f64) -> Teaching {
        let of = reply.len() + 1;
        let need = goal_of(threshold, of);
        let mut history = Vec::new();
        let mut streak = vec![0i32; of];
        let mut rounds = 0;
        let mut all: Vec<String> = text.to_vec();
        all.extend_from_slice(reply);
        loop {
            let mut guesses = Vec::with_capacity(of);
            for i in 0..of {
                let c = self.context(&all[..text.len() + i]);
                let w = if i == reply.len() { 1 } else { self.lookup(&reply[i]) };
                let g = self.top(&c);
                guesses.push((c, w, g));
            }
            let right = guesses.iter().filter(|(_, w, g)| w == g).count();
            history.push(right);
            crate::job::phase("teaching", history.len() as u64);
            if right >= need || rounds >= max_rounds || crate::job::stopped() {
                return Teaching {
                    right,
                    of,
                    need,
                    rounds,
                    reached: right >= need,
                    history,
                };
            }
            for (i, (c, w, g)) in guesses.iter().enumerate() {
                if w == g {
                    streak[i] = 0;
                    continue;
                }
                if *g != 2 {
                    self.credit(c, *g, -(2f64.powi(streak[i])));
                }
                streak[i] += 1;
            }
            let predict = std::mem::replace(&mut self.predict, false);
            self.train_tokens(&all);
            self.predict = predict;
            rounds += 1;
        }
    }

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
            .map(|t| t.discount())
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
