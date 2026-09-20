//! `radixnet` - the CLI of the Rust count / reward model.
//!
//! The same commands as `python -m radixnet` and `go/cmd/radixnet-count`, over
//! the same model file, so the three can be pointed at one model and compared
//! (`../../../tests/test_rust_parity.py` does exactly that).
//!
//! ```text
//! radixnet [--model PATH] [--encoding SPEC] [--json] [--seed N] [--workers N] [--out PATH] <command> [flags]
//!
//!   train      count one traversal of every text's path per epoch
//!   predict    continue a prefix: the K likeliest and the K least likely
//!   generate   whole texts, by beam, by the single cheapest path, or sampled
//!   score      the log-probability of a text under the model
//!   feedback   thumbs up / down: reward or punish whole texts
//!   2nrl       punish the bad texts, then count and reward the good ones
//!   invert     flip the sign of every reward
//!   compress   merge every unary chain
//!   weights    read or change the weight function's scales and window
//!   paths      what the judged walks did, per context
//!   nodes      one node against its neighbours
//!   words      the word model's alphabet, most read first
//!   info       the model's statistics
//!   serve      the HTTP API and the prebuilt frontend
//!   version    the port's version
//! ```

use std::process::ExitCode;

use radixnet::encoding::{parse_encoding, Unit};
use radixnet::file::read_document;
use radixnet::json::Json;
use radixnet::log::{self, Level};
use radixnet::model::{GenerateOptions, Model, PredictOptions, TrainOptions};
use radixnet::penalty::{resolve_traversal, DEFAULT_TRAVERSAL};
use radixnet::report::{node_rows, path_rows, split_texts, stats};
use radixnet::search::PathResult;
use radixnet::service::Service;
use radixnet::{Graph, GraphOptions};

const USAGE: &str = "usage: radixnet [--model PATH] [--encoding SPEC] [--json] [--seed N] [--workers N] \
     [--out PATH] <command>\n\
     commands: train predict generate score feedback 2nrl invert compress weights paths nodes words info serve \
     version\n\
     --encoding unit[:n[:stride]] of a NEW model - what one unit is (char | word), how many units a gram holds \
     and how far apart\n\
     consecutive grams start (1 = the sliding window, n = non-overlapping groups).  char:3:1 is the default, \
     char:5:5 groups of five\n\
     letters, word:2:1 the word bigram, word:3:1 the word trigram; the names trigram | bigram | word-bigram | \
     word-trigram work too.\n\
     --units / --ngram / --stride set the three dials separately.  A loaded file's own encoding always wins, and \
     is fixed for its life.\n\
     --log SPEC (or $RADIXNET_LOG) sets what is written to stderr: a level (error | warn | info | debug | trace | \
     off), optionally\n\
     per target - 'info', 'warn,http=debug', 'info,train=trace'.  --verbose is --log debug and --quiet is --log \
     off.  Nothing is ever\n\
     written to stdout, which is where --json puts its document.  Targets: http, model, train.";

/// The default `--model` per unit, so a word model never overwrites a
/// character model's file.
const DEFAULT_COUNT_MODEL: &str = "model.count.json";
const DEFAULT_WORD_MODEL: &str = "model.word.json";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("radixnet: {err}");
            ExitCode::FAILURE
        }
    }
}

/// The command line: global flags, a command, and that command's flags.
struct Args {
    flags: Vec<(String, String)>,
    switches: Vec<String>,
}

impl Args {
    /// A flag's value, or `None`.
    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
    fn str(&self, name: &str, fallback: &str) -> String {
        self.get(name).unwrap_or(fallback).to_string()
    }
    fn on(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }
    fn int(&self, name: &str, fallback: i64) -> Result<i64, String> {
        match self.get(name) {
            Some(text) => text
                .parse()
                .map_err(|_| format!("--{name} needs a number, got {text:?}")),
            None => Ok(fallback),
        }
    }
    fn usize(&self, name: &str, fallback: usize) -> Result<usize, String> {
        Ok(self.int(name, fallback as i64)?.max(0) as usize)
    }
    fn float(&self, name: &str, fallback: f64) -> Result<f64, String> {
        match self.get(name) {
            Some(text) => text
                .parse()
                .map_err(|_| format!("--{name} needs a number, got {text:?}")),
            None => Ok(fallback),
        }
    }
}

/// Flags that take no value.
const SWITCHES: [&str; 9] = [
    "json",
    "no-compress",
    "to-end",
    "seeded",
    "quiet",
    "verbose",
    "resume",
    "exact",
    "no-guard",
];

fn parse_args(argv: &[String]) -> Result<(String, Args), String> {
    let mut command = String::new();
    let mut args = Args {
        flags: Vec::new(),
        switches: Vec::new(),
    };
    let mut i = 0;
    while i < argv.len() {
        let item = &argv[i];
        if let Some(name) = item.strip_prefix("--") {
            let (name, inline) = match name.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (name.to_string(), None),
            };
            if SWITCHES.contains(&name.as_str()) && inline.is_none() {
                args.switches.push(name);
            } else {
                let value = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        argv.get(i).cloned().ok_or_else(|| format!("--{name} needs a value"))?
                    }
                };
                args.flags.push((name, value));
            }
        } else if command.is_empty() {
            command = item.clone();
        } else {
            return Err(format!("unexpected argument {item:?}\n{USAGE}"));
        }
        i += 1;
    }
    Ok((command, args))
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let (command, args) = parse_args(&argv)?;
    let json_mode = args.on("json");
    // stderr only, so --json keeps one document on stdout and `serve` keeps the
    // protocol there; a server defaults to info because one that says nothing
    // while it runs cannot be debugged
    log::init(if command == "serve" { Level::Info } else { Level::Warn });
    if args.on("verbose") {
        log::set_level(Level::Debug);
    }
    if args.on("quiet") {
        log::set_level(Level::Off);
    }
    if let Some(spec) = args.get("log") {
        log::configure(spec)?;
    }
    // the kind is the one this port has; the *encoding* is the dial - what one
    // unit is, how many units a gram holds, and how far apart grams start
    match args.str("kind", "count").as_str() {
        "" | "count" => {}
        "word" => {
            return Err(
                "words are an encoding, not a kind: use --encoding word:3:1 (or --units word) instead of --kind word"
                    .to_string(),
            )
        }
        other => return Err(format!("unknown model kind {other:?}; this port has 'count'")),
    }
    let mut encoding = parse_encoding(&args.str("encoding", ""))?;
    if let Some(name) = args.get("units") {
        encoding.unit = Unit::parse(name).ok_or_else(|| format!("--units must be char or word, got {name:?}"))?;
    }
    if args.get("ngram").is_some() {
        encoding.n = args.usize("ngram", encoding.n)?;
    }
    if args.get("stride").is_some() {
        encoding.stride = args.usize("stride", encoding.stride)?;
    }
    encoding.validate()?;
    let model_path = args.str(
        "model",
        if encoding.unit == Unit::Words {
            DEFAULT_WORD_MODEL
        } else {
            DEFAULT_COUNT_MODEL
        },
    );
    let out_path = args.get("out").map(str::to_string);
    let seed = args.int("seed", 0)?;
    let workers = args.usize("workers", 0)?;

    let open = |must_exist: bool| -> Result<Model, String> {
        if std::path::Path::new(&model_path).exists() {
            let mut m = Model::load(&model_path)?;
            m.workers = workers;
            m.g.workers = workers;
            return Ok(m);
        }
        if must_exist {
            return Err(format!("no model at {model_path}"));
        }
        let mut m = Model::new(
            seed,
            GraphOptions {
                encoding,
                ..Default::default()
            },
        )?;
        m.workers = workers;
        m.g.workers = workers;
        Ok(m)
    };
    let save = |model: &mut Model| -> Result<String, String> {
        let path = out_path.clone().unwrap_or_else(|| model_path.clone());
        model.save(&path)?;
        Ok(path)
    };
    let emit = |doc: Json| {
        if json_mode {
            println!("{}", doc.render(0));
        } else {
            println!("{}", doc.render(2));
        }
    };

    match command.as_str() {
        "train" => {
            let mut model = open(false)?;
            let texts = read_texts(&args)?;
            let opts = TrainOptions {
                epochs: args.usize("epochs", 5)?,
                auto_compress: !args.on("no-compress"),
                chunk_size: args.usize("chunk", 0)?,
                phase: None,
            };
            let records = model.train(&texts, &opts)?;
            let saved = save(&mut model)?;
            emit(Json::obj([
                ("texts", Json::Int(texts.len() as i64)),
                ("split", Json::str(args.str("split", "lines"))),
                (
                    "chunk",
                    Json::Int(opts.chunk_size.max(radixnet::model::DEFAULT_CHUNK_SIZE) as i64),
                ),
                ("workers", Json::Int(workers as i64)),
                ("counting", Json::str(model.counting())),
                ("records", Json::Arr(records.iter().map(|r| r.to_json()).collect())),
                ("saved", Json::str(saved)),
                ("stats", stats(&model)),
            ]));
        }
        "predict" => {
            let mut model = open(true)?;
            let prefix = args.str("prefix", "");
            let max_length = args.int("max-length", -1)?;
            let opts = PredictOptions {
                length: args.usize("length", 20)?,
                mode: args.str("mode", "beam"),
                k: args.usize("k", 5)?,
                beam: args.usize("beam", 0)?,
                step_penalty: args.float("step-penalty", 0.0)?,
                temperature: args.float("temperature", 1.0)?,
                to_end: args.on("to-end"),
                max_length: (max_length >= 0).then_some(max_length as usize),
                traversal: resolve_traversal(&args.str("traversal", DEFAULT_TRAVERSAL))?.to_string(),
                penalty_scale: args.float("penalty-scale", 1.0)?,
                merit_scale: args.float("merit-scale", 1.0)?,
            };
            let found = model.predict(&prefix, &opts)?;
            let mut doc = vec![
                ("prefix".to_string(), Json::str(prefix)),
                ("kind".to_string(), Json::str(model.kind())),
                ("continuation".to_string(), Json::str(found.best.text.clone())),
                ("full_text".to_string(), Json::str(found.best.full_text.clone())),
                ("cost".to_string(), Json::Num(found.best.cost)),
                ("probability".to_string(), Json::Num(found.best.probability())),
                ("step_costs".to_string(), Json::nums(found.best.step_costs.clone())),
                ("path".to_string(), Json::strs(found.best.labels.clone())),
                (
                    "node_ids".to_string(),
                    Json::ints(found.best.node_ids.iter().map(|&n| n as i64)),
                ),
                ("expanded".to_string(), Json::Int(found.expanded as i64)),
                ("reached_end".to_string(), Json::Bool(found.best.reached_end)),
                ("mode".to_string(), Json::str(found.mode.clone())),
                ("k".to_string(), Json::Int(found.k as i64)),
                ("beam".to_string(), Json::Int(found.beam as i64)),
                ("top".to_string(), Json::Arr(found.top.iter().map(path_json).collect())),
                (
                    "bottom".to_string(),
                    Json::Arr(found.bottom.iter().map(path_json).collect()),
                ),
            ];
            doc.push(("traversal".to_string(), Json::str(found.traversal.clone())));
            emit(Json::Obj(doc));
        }
        "generate" => {
            let mut model = open(true)?;
            let seeded = args.on("seeded");
            let opts = GenerateOptions {
                max_length: args.usize("max-length", 60)?,
                mode: args.str("mode", "sample"),
                temperature: args.float("temperature", 1.0)?,
                count: args.usize("count", 1)?,
                seed: seeded.then_some(seed),
                prefix: args.str("prefix", ""),
                step_penalty: args.float("step-penalty", 0.0)?,
                beam: args.usize("beam", 0)?,
                traversal: resolve_traversal(&args.str("traversal", DEFAULT_TRAVERSAL))?.to_string(),
                penalty_scale: args.float("penalty-scale", 1.0)?,
                merit_scale: args.float("merit-scale", 1.0)?,
            };
            let samples = model.generate(&opts)?;
            emit(Json::obj([
                ("samples", Json::Arr(samples.iter().map(path_json).collect())),
                ("count", Json::Int(samples.len() as i64)),
                ("mode", Json::str(opts.mode.clone())),
                ("prefix", Json::str(opts.prefix.clone())),
                ("max_length", Json::Int(opts.max_length as i64)),
                ("temperature", Json::Num(opts.temperature)),
            ]));
        }
        "score" => {
            let mut model = open(true)?;
            let texts = match args.get("text") {
                Some(text) => vec![text.to_string()],
                None => read_texts(&args)?,
            };
            let results: Vec<Json> = texts
                .iter()
                .map(|text| {
                    let s = model.score(text);
                    Json::obj([
                        ("text", Json::str(text.clone())),
                        ("log_prob", Json::Num(s.log_prob)),
                        ("per_char", Json::Num(s.per_char)),
                        ("chars", Json::Int(s.chars as i64)),
                        ("transitions", Json::Int(s.transitions as i64)),
                        ("unknown_transitions", Json::Int(s.unknown_transitions as i64)),
                    ])
                })
                .collect();
            let n = results.len().max(1) as f64;
            let mean = |key: &str| results.iter().map(|r| r.at(key).as_f64().unwrap_or(0.0)).sum::<f64>() / n;
            emit(Json::obj([
                ("results", Json::Arr(results.clone())),
                ("count", Json::Int(results.len() as i64)),
                ("mean_log_prob", Json::Num(mean("log_prob"))),
                ("mean_per_char", Json::Num(mean("per_char"))),
            ]));
        }
        "feedback" => {
            // 2NRL when both kinds are given, reward on good alone, punish on
            // bad alone - and the same epoch defaults Python and Go use
            let mut model = open(true)?;
            let strength = args.float("strength", 1.0)?;
            let neg_epochs = args.usize("neg-epochs", 2)?;
            let pos_epochs = args.usize("pos-epochs", 3)?;
            let good = read_named(&args, "good")?;
            let bad = read_named(&args, "bad")?;
            if good.is_empty() && bad.is_empty() {
                return Err("nothing to learn from: give --good (thumbs up) and / or --bad (thumbs down)".to_string());
            }
            let (action, negative, positive) = match (good.is_empty(), bad.is_empty()) {
                (false, false) => {
                    let (negative, positive) = model.two_nrl(&bad, &good, neg_epochs, pos_epochs, strength)?;
                    ("2nrl", negative, positive)
                }
                (false, true) => ("reward", Vec::new(), model.reward(&good, pos_epochs, strength)?),
                _ => ("punish", model.punish(&bad, neg_epochs, strength)?, Vec::new()),
            };
            let inverted = model.g.inverted;
            let saved = save(&mut model)?;
            emit(Json::obj([
                ("action", Json::str(action)),
                ("good_texts", Json::Int(good.len() as i64)),
                ("bad_texts", Json::Int(bad.len() as i64)),
                ("negative", Json::Arr(negative.iter().map(|r| r.to_json()).collect())),
                ("positive", Json::Arr(positive.iter().map(|r| r.to_json()).collect())),
                ("inverted", Json::Bool(inverted)),
                ("saved", Json::str(saved)),
                ("stats", stats(&model)),
            ]));
        }
        "2nrl" => {
            let mut model = open(true)?;
            let bad = read_named(&args, "bad")?;
            let good = read_named(&args, "good")?;
            // the standalone command's defaults are 3 and 3; `feedback` uses 2 and 3
            let (negative, positive) = model.two_nrl(
                &bad,
                &good,
                args.usize("neg-epochs", 3)?,
                args.usize("pos-epochs", 3)?,
                args.float("strength", 1.0)?,
            )?;
            let inverted = model.g.inverted;
            let saved = save(&mut model)?;
            emit(Json::obj([
                ("negative", Json::Arr(negative.iter().map(|r| r.to_json()).collect())),
                ("positive", Json::Arr(positive.iter().map(|r| r.to_json()).collect())),
                ("inverted", Json::Bool(inverted)),
                ("saved", Json::str(saved)),
                ("stats", stats(&model)),
            ]));
        }
        "invert" => {
            let mut model = open(true)?;
            model.invert();
            let saved = save(&mut model)?;
            emit(Json::obj([("saved", Json::str(saved)), ("stats", stats(&model))]));
        }
        "compress" => {
            let mut model = open(true)?;
            let merged = model.g.compress();
            let doc = Json::obj([
                ("merged", Json::Int(merged as i64)),
                ("nodes", Json::Int(model.g.num_nodes() as i64)),
                ("edges", Json::Int(model.g.num_edges() as i64)),
                ("trigrams", Json::Int(model.g.num_trigrams() as i64)),
                ("compression_ratio", Json::Num(model.g.compression_ratio())),
                ("saved", Json::str(save(&mut model)?)),
            ]);
            emit(doc);
        }
        "weights" => {
            let mut model = open(true)?;
            let mut changed = false;
            for (flag, apply) in weight_flags() {
                if let Some(text) = args.get(flag) {
                    let value: f64 = text.parse().map_err(|_| format!("--{flag} needs a number"))?;
                    if !value.is_finite() {
                        return Err(format!("--{flag} must be a finite number"));
                    }
                    apply(&mut model.g, value)?;
                    changed = true;
                }
            }
            if changed {
                model.g.invalidate();
                model.g.recompute_weights();
                let saved = save(&mut model)?;
                emit(Json::obj([("saved", Json::str(saved)), ("stats", stats(&model))]));
            } else {
                emit(Json::obj([("stats", stats(&model))]));
            }
        }
        "paths" => {
            let model = open(true)?;
            let node = args.get("node").map(|n| n.parse::<usize>().unwrap_or(usize::MAX));
            let mut doc = match path_rows(&model.g, args.usize("limit", 20)?, node) {
                Json::Obj(pairs) => pairs,
                other => vec![("paths".to_string(), other)],
            };
            doc.push(("stats".to_string(), stats(&model)));
            emit(Json::Obj(doc));
        }
        "nodes" => {
            let model = open(true)?;
            let node = args.get("node").map(|n| n.parse::<usize>().unwrap_or(usize::MAX));
            emit(Json::obj([
                ("nodes", node_rows(&model.g, args.usize("limit", 20)?, node)),
                ("stats", stats(&model)),
            ]));
        }
        "words" => {
            let mut model = open(true)?;
            let enc = model.encoding();
            if enc.unit != Unit::Words {
                return Err(format!(
                    "{model_path} counts in {}, so it has no words to list; a word alphabet needs a word encoding \
                     (train a new model with --encoding word:{}:{})",
                    enc.units_name(),
                    enc.n,
                    enc.stride
                ));
            }
            let all = model.top_words(0);
            let vocabulary = all.len();
            let limit = args.usize("limit", 20)?;
            let rows = if limit > 0 {
                &all[..limit.min(all.len())]
            } else {
                &all[..]
            };
            emit(Json::obj([
                ("words", Json::Arr(rows.iter().map(|r| r.to_json()).collect())),
                ("vocabulary", Json::Int(vocabulary as i64)),
                ("units", Json::str(model.units())),
                ("encoding", Json::str(enc.to_string())),
                ("stats", stats(&model)),
            ]));
        }
        "serve" => {
            // the HTTP API and, when it is there, the prebuilt frontend beside it
            let model = open(false)?;
            let host = args.str("host", "127.0.0.1");
            let port = args.usize("port", 8000)? as u16;
            let frontend = args.str("frontend-dir", "frontend/dist");
            let mut service = Service::new(model, model_path.clone(), seed, workers);
            service.upload_dir = args.get("upload-dir").map(str::to_string);
            service.checkpoint_dir = args.get("checkpoint-dir").map(str::to_string);
            let frontend = if std::path::Path::new(&frontend).is_dir() {
                Some(frontend)
            } else {
                None
            };
            // through the logger, so a run has one channel rather than two
            radixnet::log_info!(
                "http",
                "{}",
                match &frontend {
                    Some(dir) => format!("serving the frontend from {dir}"),
                    None => "no frontend directory: the API alone".to_string(),
                }
            );
            radixnet::service::build(std::sync::Arc::new(service), frontend).serve(&host, port)?;
        }
        "info" => {
            let model = open(true)?;
            let tail = model.history.len().saturating_sub(args.usize("history", 5)?);
            emit(Json::obj([
                ("model", Json::str(model_path.clone())),
                ("stats", stats(&model)),
                ("meta", model.meta.to_json()),
                ("history_len", Json::Int(model.history.len() as i64)),
                (
                    "history",
                    Json::Arr(model.history[tail..].iter().map(|r| r.to_json()).collect()),
                ),
            ]));
        }
        "check" => {
            // the structural invariants, over a corpus when one is given
            let model = open(true)?;
            let texts = read_named(&args, "data")?;
            model.g.check_invariants(&texts, false)?;
            emit(Json::obj([
                ("checked", Json::Int(texts.len() as i64)),
                ("stats", stats(&model)),
            ]));
        }
        "version" => emit(Json::obj([
            ("version", Json::str(env!("CARGO_PKG_VERSION"))),
            ("backend", Json::str("rust")),
            ("format", Json::str(radixnet::MODEL_FORMAT)),
            ("format_version", Json::Int(radixnet::MODEL_FORMAT_VERSION)),
        ])),
        "" => return Err(format!("no command\n{USAGE}")),
        other => return Err(format!("unknown command {other:?}\n{USAGE}")),
    }
    Ok(())
}

type WeightSetter = (&'static str, fn(&mut Graph, f64) -> Result<(), String>);

/// The weight function's knobs, as the `weights` command sets them.
fn weight_flags() -> Vec<WeightSetter> {
    vec![
        ("count-scale", |g, v| {
            g.count_scale = v;
            Ok(())
        }),
        ("global-scale", |g, v| {
            g.global_scale = v;
            Ok(())
        }),
        ("window-scale", |g, v| {
            g.window_scale = v;
            Ok(())
        }),
        ("reward-scale", |g, v| {
            g.reward_scale = v;
            Ok(())
        }),
        ("path-scale", |g, v| {
            g.path_scale = v;
            Ok(())
        }),
        ("window", |g, v| {
            if v < 1.0 {
                return Err(format!("window must be >= 1, got {v}"));
            }
            g.set_window(v as usize);
            Ok(())
        }),
    ]
}

/// One walk, as the CLI reports it.
fn path_json(result: &PathResult) -> Json {
    let mut pairs = vec![
        ("text".to_string(), Json::str(result.text.clone())),
        ("labels".to_string(), Json::strs(result.labels.clone())),
        (
            "node_ids".to_string(),
            Json::ints(result.node_ids.iter().map(|&n| n as i64)),
        ),
        ("cost".to_string(), Json::Num(result.cost)),
        ("step_costs".to_string(), Json::nums(result.step_costs.clone())),
        ("expanded".to_string(), Json::Int(result.expanded as i64)),
        ("reached_end".to_string(), Json::Bool(result.reached_end)),
        ("full_text".to_string(), Json::str(result.full_text.clone())),
        ("probability".to_string(), Json::Num(result.probability())),
    ];
    if result.punish != 0.0 {
        pairs.push(("punish".to_string(), Json::Num(result.punish)));
    }
    Json::Obj(pairs)
}

/// The texts of `--data` (or of a `--text`), split as `--split` asks.
fn read_texts(args: &Args) -> Result<Vec<String>, String> {
    if let Some(text) = args.get("text") {
        return Ok(vec![text.to_string()]);
    }
    read_named(args, "data")
}

/// The texts of one named file flag; an absent flag reads as nothing.
fn read_named(args: &Args, flag: &str) -> Result<Vec<String>, String> {
    let Some(path) = args.get(flag) else {
        return Ok(Vec::new());
    };
    let content = if path.ends_with(".gz") {
        let bytes = std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))?;
        String::from_utf8(radixnet::gzip::decompress(&bytes)?).map_err(|err| format!("{path}: {err}"))?
    } else {
        std::fs::read_to_string(path).map_err(|err| format!("cannot read {path}: {err}"))?
    };
    let unit = args.str("split", "lines");
    let page_lines = args.usize("page-lines", 0).unwrap_or(0);
    Ok(split_texts(&content, &unit, page_lines))
}

/// Reads a document without building a model - what `--json` callers use to
/// check a file this port wrote.
#[allow(dead_code)]
fn peek(path: &str) -> Result<Json, String> {
    read_document(path)
}
