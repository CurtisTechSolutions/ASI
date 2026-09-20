//! How fast this build counts and predicts.
//!
//! The synthetic corpus is structured the way the Python and Go benchmarks
//! build theirs - verbatim lines, lines spliced at a word boundary, lines with
//! a word or two swapped - so the graph it builds has the same shape: unary
//! chains to compress, shared prefixes to branch on and realistic fan-out.  The
//! sentences are not held to theirs, because each language's benchmark measures
//! throughput rather than numbers.
//!
//! The cross-language comparison does not use it at all: `--texts` and
//! `--prefixes` hand this port and the Go one the *same* corpus and the *same*
//! prefixes, so neither is timed on work the other did not do (see
//! `../../bench/`).

use std::time::Instant;

use crate::encoding::char_len;
use crate::graph::GraphOptions;
use crate::json::Json;
use crate::model::{Model, PredictOptions, TrainOptions};
use crate::search::Traversal;

/// Benchmark defaults, matching the Python and Go modules.
pub const BENCH_DEFAULT_CHARS: usize = 20_000;
pub const BENCH_DEFAULT_EPOCHS: usize = 2;
pub const BENCH_PREDICT_LENGTH: usize = 20;
const BENCH_MIN_PREFIX: usize = 3;
const BENCH_MAX_PREFIX: usize = 12;
const BENCH_MIN_PREDICTS: usize = 100;
const BENCH_MAX_PREDICTS: usize = 20_000;

const FALLBACK_LINES: &[&str] = &[
    "the cat sat on the mat",
    "the dog sat on the log",
    "a bird flew over the hill",
    "the cat chased the mouse",
    "the quick brown fox jumps over the lazy dog",
    "rain fell on the quiet town",
    "she walked to the river at dawn",
    "the old man read a book by the fire",
    "children played in the summer field",
    "a train passed through the empty station",
];

/// SplitMix64 - the benchmark's own generator, so a synthetic corpus is
/// reproducible without pulling in a crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: i64) -> SplitMix64 {
        SplitMix64(seed as u64)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// The non-blank lines of at least three characters of a sample corpus, falling
/// back to a built-in list when the file cannot be read.
pub fn bench_lines(path: &str) -> Vec<String> {
    let lines: Vec<String> = std::fs::read_to_string(path)
        .map(|data| {
            data.lines()
                .map(|line| line.trim().to_string())
                .filter(|line| char_len(line) >= 3)
                .collect()
        })
        .unwrap_or_default();
    if lines.is_empty() {
        return FALLBACK_LINES.iter().map(|s| s.to_string()).collect();
    }
    lines
}

/// Builds deterministic training texts totalling at least `chars` characters.
pub fn synthetic_corpus(chars: usize, seed: i64, base: &[String]) -> Result<Vec<String>, String> {
    let lines: Vec<&String> = base.iter().filter(|line| char_len(line) >= 3).collect();
    if lines.is_empty() {
        return Err("need at least one base line of >= 3 characters".to_string());
    }
    let mut vocab: Vec<&str> = lines.iter().flat_map(|line| line.split_whitespace()).collect();
    vocab.sort_unstable();
    vocab.dedup();

    let mut rng = SplitMix64::new(seed);
    let mut texts = Vec::new();
    let mut total = 0usize;
    while total < chars {
        let roll = rng.unit();
        let mut text = if roll < 0.4 || vocab.len() < 2 {
            lines[rng.below(lines.len())].to_string()
        } else if roll < 0.7 {
            let left: Vec<&str> = lines[rng.below(lines.len())].split_whitespace().collect();
            let right: Vec<&str> = lines[rng.below(lines.len())].split_whitespace().collect();
            let cut = 1 + rng.below(left.len());
            let from = rng.below(right.len());
            let mut parts: Vec<&str> = left[..cut].to_vec();
            parts.extend_from_slice(&right[from..]);
            parts.join(" ")
        } else {
            let mut parts: Vec<&str> = lines[rng.below(lines.len())].split_whitespace().collect();
            let swaps = rng.below(2) + 1;
            for _ in 0..swaps {
                let at = rng.below(parts.len());
                parts[at] = vocab[rng.below(vocab.len())];
            }
            parts.join(" ")
        };
        if char_len(&text) < 3 {
            text = lines[rng.below(lines.len())].to_string();
        }
        total += char_len(&text);
        texts.push(text);
    }
    Ok(texts)
}

/// Cuts `count` prefixes (3-12 characters) out of the corpus, so every
/// prediction starts at a node the trained graph knows.
pub fn bench_prefixes(texts: &[String], count: usize, seed: i64) -> Result<Vec<String>, String> {
    if count > 0 && texts.is_empty() {
        return Err("cannot cut prefixes out of an empty corpus".to_string());
    }
    let mut rng = SplitMix64::new(seed + 1);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let runes: Vec<char> = texts[rng.below(texts.len())].chars().collect();
        let high = BENCH_MAX_PREFIX.min(runes.len());
        let mut length = BENCH_MIN_PREFIX;
        if high > BENCH_MIN_PREFIX {
            length += rng.below(high - BENCH_MIN_PREFIX + 1);
        }
        let start = if runes.len() > length {
            rng.below(runes.len() - length + 1)
        } else {
            0
        };
        out.push(runes[start..start + length].iter().collect());
    }
    Ok(out)
}

/// How many alive edges carry a penalty - how much blame the least-punished
/// traversal has to steer by.
fn punished_edges(g: &crate::graph::Graph) -> usize {
    (0..g.num_edge_ids())
        .filter(|&e| g.is_edge_alive(e) && g.edge_reward(e) < 0.0)
        .count()
}

/// One benchmark run.
#[derive(Clone, Debug)]
pub struct BenchOptions {
    pub chars: usize,
    pub epochs: usize,
    pub seed: i64,
    /// 0: `chars / 2`, clamped to `[100, 20000]`
    pub predictions: usize,
    /// The sample corpus the sentences are built from.
    pub corpus_path: String,
    pub workers: usize,
    /// The corpus itself, instead of a synthetic one.
    pub texts: Vec<String>,
    /// The prefixes to predict, instead of ones cut out of the corpus.
    pub prefixes: Vec<String>,
    /// The search the prediction phase times.
    pub traversal: Traversal,
    /// Writes every prediction's continuation to this file, one per line - how
    /// the comparison counts the answers the two traversals disagree on.
    pub dump: String,
    /// Punishes every Nth text of the corpus (one pass, strength 1) before the
    /// predictions are timed - 0 leaves the model as training left it.  A model
    /// nothing was ever punished for has no blame to walk by, so the
    /// least-punished traversal would have nothing to do and both searches
    /// would measure the same thing.
    pub punish_every: usize,
}

impl Default for BenchOptions {
    fn default() -> BenchOptions {
        BenchOptions {
            chars: BENCH_DEFAULT_CHARS,
            epochs: BENCH_DEFAULT_EPOCHS,
            seed: 0,
            predictions: 0,
            corpus_path: "data/sample_corpus.txt".to_string(),
            workers: 0,
            texts: Vec::new(),
            prefixes: Vec::new(),
            traversal: Traversal::Reward,
            dump: String::new(),
            punish_every: 0,
        }
    }
}

/// Trains on a corpus, then predicts, and reports the rates - the same keys the
/// Python and Go benchmarks report.
pub fn run_benchmark(o: &BenchOptions) -> Result<Json, String> {
    if o.chars < 1 && o.texts.is_empty() {
        return Err(format!("chars must be >= 1, got {}", o.chars));
    }
    if o.epochs < 1 {
        return Err(format!("epochs must be >= 1, got {}", o.epochs));
    }
    let mut predictions = o.predictions;
    if !o.prefixes.is_empty() {
        predictions = o.prefixes.len();
    }
    if predictions == 0 {
        predictions = (o.chars / 2).clamp(BENCH_MIN_PREDICTS, BENCH_MAX_PREDICTS);
    }
    let texts = if o.texts.is_empty() {
        synthetic_corpus(o.chars, o.seed, &bench_lines(&o.corpus_path))?
    } else {
        o.texts.clone()
    };
    let total_chars: usize = texts.iter().map(|t| char_len(t)).sum();

    let mut model = Model::new(o.seed, GraphOptions::default())?;
    model.workers = o.workers;
    model.g.workers = o.workers;
    let started = Instant::now();
    // compression is part of the work a training pass does, so timing it
    // without would flatter the result
    let records = model.train(
        &texts,
        &TrainOptions {
            epochs: o.epochs,
            auto_compress: true,
            ..Default::default()
        },
    )?;
    let train_seconds = started.elapsed().as_secs_f64();

    let mut punished = 0usize;
    if o.punish_every > 0 {
        let bad: Vec<String> = texts
            .iter()
            .skip(o.punish_every - 1)
            .step_by(o.punish_every)
            .cloned()
            .collect();
        punished = bad.len();
        model.punish(&bad, 1, 1.0)?;
    }

    let prefixes = if o.prefixes.is_empty() {
        bench_prefixes(&texts, predictions, o.seed)?
    } else {
        o.prefixes.clone()
    };
    model.g.prepare();
    let mut expansions = 0usize;
    let mut sample = None;
    let mut dump: Vec<String> = if o.dump.is_empty() {
        Vec::new()
    } else {
        Vec::with_capacity(prefixes.len())
    };
    let opts = PredictOptions {
        length: BENCH_PREDICT_LENGTH,
        mode: "beam".into(),
        k: 1,
        traversal: o.traversal,
        ..Default::default()
    };
    let started = Instant::now();
    for (i, prefix) in prefixes.iter().enumerate() {
        let found = model.predict(prefix, &opts)?;
        expansions += found.expanded;
        if !o.dump.is_empty() {
            dump.push(found.best.text.replace('\n', " "));
        }
        if i == 0 {
            sample = Some(found);
        }
    }
    let predict_seconds = started.elapsed().as_secs_f64();
    if !o.dump.is_empty() {
        std::fs::write(&o.dump, dump.join("\n") + "\n").map_err(|err| format!("cannot write {}: {err}", o.dump))?;
    }

    let transitions: i64 = records.iter().map(|r| r.transitions).sum();
    let g = &model.g;
    let rate = |count: f64, seconds: f64| if seconds <= 0.0 { 0.0 } else { count / seconds };
    let mut out = vec![
        ("backend".to_string(), Json::str("rust")),
        ("seed".to_string(), Json::Int(o.seed)),
        ("chars".to_string(), Json::Int(total_chars as i64)),
        ("texts".to_string(), Json::Int(texts.len() as i64)),
        ("epochs".to_string(), Json::Int(o.epochs as i64)),
        ("train_seconds".to_string(), Json::Num(train_seconds)),
        ("transitions".to_string(), Json::Int(transitions)),
        (
            "transitions_per_sec".to_string(),
            Json::Num(rate(transitions as f64, train_seconds)),
        ),
        (
            "chars_per_sec".to_string(),
            Json::Num(rate((total_chars * o.epochs) as f64, train_seconds)),
        ),
        ("predict_count".to_string(), Json::Int(predictions as i64)),
        ("predict_seconds".to_string(), Json::Num(predict_seconds)),
        (
            "predictions_per_sec".to_string(),
            Json::Num(rate(predictions as f64, predict_seconds)),
        ),
        ("dijkstra_expansions".to_string(), Json::Int(expansions as i64)),
        (
            "dijkstra_expansions_per_sec".to_string(),
            Json::Num(rate(expansions as f64, predict_seconds)),
        ),
        ("traversal".to_string(), Json::str(o.traversal.name())),
        ("punished_texts".to_string(), Json::Int(punished as i64)),
        ("punished_edges".to_string(), Json::Int(punished_edges(g) as i64)),
        ("edge_reward_negative".to_string(), Json::Num(g.total_reward().1)),
        ("nodes".to_string(), Json::Int(g.num_nodes() as i64)),
        ("edges".to_string(), Json::Int(g.num_edges() as i64)),
        ("trigrams".to_string(), Json::Int(g.num_trigrams() as i64)),
        ("compression_ratio".to_string(), Json::Num(g.compression_ratio())),
        ("workers".to_string(), Json::Int(o.workers as i64)),
        ("counting".to_string(), Json::str(model.counting())),
        (
            "loss".to_string(),
            Json::Num(records.last().map(|r| r.loss).unwrap_or(0.0)),
        ),
    ];
    if let Some(sample) = sample {
        out.push((
            "sample_prediction".to_string(),
            Json::obj([
                ("prefix", Json::str(prefixes[0].clone())),
                ("continuation", Json::str(sample.best.text.clone())),
                ("cost", Json::Num(sample.best.cost)),
                ("reached_end", Json::Bool(sample.best.reached_end)),
            ]),
        ));
    }
    Ok(Json::Obj(out))
}
