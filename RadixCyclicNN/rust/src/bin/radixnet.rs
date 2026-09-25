//! `radixnet` - the CLI of the Rust port.
//!
//! The same commands as `python -m radixnet` and `go/cmd/radixnet-count`, over
//! the same model file, so the three can be pointed at one model and compared
//! (`../../../tests/test_rust_parity.py` does exactly that).  `--kind radix |
//! count | negative | resonant` picks the kind of a *new* model - count is this
//! port's default, radix Python's - and a loaded file's own kind always wins
//! ([`radixnet::kinds`]).
//!
//! ```text
//! radixnet [--model PATH] [--kind KIND] [--encoding SPEC] [--json] [--seed N] [--workers N] [--out PATH] <command>
//!
//!   train      train on texts the way the kind learns (counting, gradient descent, blame)
//!   predict    continue a prefix: the K likeliest and the K least likely
//!   generate   whole texts, by beam, by the single cheapest path, or sampled
//!   score      the log-probability of a text under the model
//!   feedback   thumbs up / down: reward or punish whole texts
//!   2nrl       punish the bad texts, then count and reward the good ones
//!   invert     flip the sign of every reward
//!   compress   merge every unary chain
//!   weights    read or change the weight function's scales and window
//!   schedule   preview a learning-rate schedule (the radix model's)
//!   paths      what the judged walks did, per context
//!   nodes      one node against its neighbours
//!   words      the word model's alphabet, most read first
//!   info       the model's statistics
//!   serve      the HTTP API and the prebuilt frontend
//!   version    the port's version
//! ```

use std::process::ExitCode;

use radixnet::cli::{
    dispatch, negative_stats, parse_args, path_json, read_named, read_texts, verdict_json, Ctx, COMMANDS,
};
use radixnet::duo::{guard_report, Filter, FilterConfig};
use radixnet::encoding::{parse_encoding, Unit};
use radixnet::file::read_document;
use radixnet::json::Json;
use radixnet::kinds::TrainSettings;
use radixnet::log::{self, Level};
use radixnet::model::{GenerateOptions, Model, PredictOptions, SearchTuning};
use radixnet::negative::{BlameOptions as NegBlameOptions, JudgeOptions};
use radixnet::penalty::{resolve_traversal, DEFAULT_TRAVERSAL};
use radixnet::radix::{Feedback, TrainConfig};
use radixnet::report::{node_rows, path_rows, stats};
use radixnet::search::SamplingFilter;
use radixnet::service::Service;
use radixnet::training::Plan;

const USAGE: &str = "usage: radixnet [--model PATH] [--kind KIND] [--encoding SPEC] [--json] [--seed N] \
     [--workers N] [--out PATH] <command>\n\
     commands: train predict generate score feedback 2nrl negative invert compress weights schedule paths nodes \
     words info \
     serve version\n\
     --kind radix | count | negative | resonant: the algorithm of a NEW model (count is this port's default); a \
     loaded file's own kind always wins.\n\
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
     written to stdout, which is where --json puts its document.  Targets: http, model, train.\n\
     predict / generate: --top-k K, --top-p P, --min-p P narrow what a sampled step draws from (off at 0 / 1 / 0); \
     --diversity X spreads\n\
     the beam's K continuations apart (0 = off).  train: --order corpus | shortest-first | longest-first | \
     shuffle, --curriculum C (the\n\
     first epoch walks the first C of the ordered texts, the last all of them; 1 = off), --replay R (every epoch \
     rehearses R times as\n\
     many texts from the model's replay buffer; 0 = off), --replay-size N (the buffer's capacity; 0 drops it), \
     --patience N and\n\
     --min-delta X (stop after N full epochs without the loss improving by X; 0 = off).  \
     ../SPEC-SearchAndTraining.md has the rules.";

/// The default `--model` per unit, so a word model never overwrites a
/// character model's file.
/// The usage text: the model's own commands, then every area's.
fn usage() -> String {
    let mut text = USAGE.to_string();
    text.push_str("\nmore commands:");
    for (name, _, help) in COMMANDS {
        text.push_str(&format!("\n  {name:<12} {help}"));
    }
    text
}

const DEFAULT_COUNT_MODEL: &str = "model.count.json";
const DEFAULT_WORD_MODEL: &str = "model.word.json";

/// The default `--model` of the other kinds, Python's: one kind never
/// overwrites another's default file.
fn default_model(kind: &str, words: bool) -> &'static str {
    match kind {
        "radix" => "model.json",
        "negative" => "model.negative.json",
        "resonant" => "model.resonant.json",
        _ if words => DEFAULT_WORD_MODEL,
        _ => DEFAULT_COUNT_MODEL,
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("radixnet: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
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
    // the kind of a new model (count unless --kind says otherwise); words are
    // not one - they are the *encoding*, the dial of what one unit is, how many
    // units a gram holds, and how far apart grams start
    let kind =
        match args.str("kind", "").trim().to_lowercase().as_str() {
            "word" | "words" => return Err(
                "words are an encoding, not a kind: use --encoding word:3:1 (or --units word) instead of --kind word"
                    .to_string(),
            ),
            other => radixnet::kinds::parse_kind(other)?,
        };
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
    let model_path = args.str("model", default_model(kind, encoding.unit == Unit::Words));
    let out_path = args.get("out").map(str::to_string);
    let seed = args.int("seed", 0)?;
    let workers = args.usize("workers", 0)?;

    let ctx = Ctx {
        command: command.clone(),
        args: args.clone(),
        model_path: model_path.clone(),
        encoding,
        seed,
        workers,
        json: json_mode,
        out: out_path.clone(),
    };
    let open = |must_exist: bool| ctx.open(must_exist);
    let save = |model: &mut Model| ctx.save(model);
    let emit = |doc: Json| ctx.emit(doc);

    match command.as_str() {
        "train" => {
            // --checkpoint-dir / --checkpoint-every / --keep, and --resume from the latest
            let schedule = radixnet::checkpoint::Schedule::from_args(&args)?;
            let resumed = if args.on("resume") {
                schedule.resume(&ctx)?
            } else {
                None
            };
            let mut model = match resumed {
                Some(model) => model,
                None => open(false)?,
            };
            let texts = read_texts(&args)?;
            // every kind's settings; each kind reads what applies to it
            let settings = TrainSettings {
                config: train_config(&args, args.usize("epochs", 5)?)?,
                chunk_size: args.usize("chunk", 0)?,
                reason: args.str("reason", ""),
                severity: args.float("severity", 1.0)?,
                source: args.str("source", ""),
                note: args.str("note", ""),
            };
            let records = radixnet::checkpoint::train(
                &mut model,
                &texts,
                &settings,
                schedule.shared(),
                schedule.every,
                &mut |_| true,
            )?;
            let saved = save(&mut model)?;
            let mut doc = Json::obj([
                ("kind", Json::str(model.kind())),
                ("texts", Json::Int(texts.len() as i64)),
                ("split", Json::str(args.str("split", "lines"))),
                (
                    "chunk",
                    Json::Int(settings.chunk_size.max(radixnet::model::DEFAULT_CHUNK_SIZE) as i64),
                ),
                ("workers", Json::Int(workers as i64)),
                ("counting", Json::str(model.counting())),
                ("config", settings.config.to_json()),
                ("records", Json::Arr(records.iter().map(|r| r.to_json()).collect())),
                ("saved", Json::str(saved)),
                ("stats", stats(&model)),
            ]);
            if let Json::Obj(pairs) = &mut doc {
                pairs.extend(schedule.report());
            }
            emit(doc);
        }
        "predict" => {
            let mut model = open(true)?;
            let prefix = args.str("prefix", "");
            let max_length = args.int("max-length", -1)?;
            // Python's default is dijkstra; the count model's dijkstra is its beam
            let default_mode = if model.kind() == "count" { "beam" } else { "dijkstra" };
            let tuning = search_tuning(&args)?;
            let opts = PredictOptions {
                length: args.usize("length", 20)?,
                mode: args.str("mode", default_mode),
                k: args.usize("k", 5)?,
                beam: args.usize("beam", 0)?,
                step_penalty: args.float("step-penalty", 0.0)?,
                temperature: args.float("temperature", 1.0)?,
                to_end: args.on("to-end"),
                max_length: (max_length >= 0).then_some(max_length as usize),
                traversal: resolve_traversal(&args.str("traversal", DEFAULT_TRAVERSAL))?.to_string(),
                penalty_scale: args.float("penalty-scale", 1.0)?,
                merit_scale: args.float("merit-scale", 1.0)?,
                top_k: tuning.filter.top_k,
                top_p: tuning.filter.top_p,
                min_p: tuning.filter.min_p,
                diversity: tuning.diversity,
                origin: radixnet::START,
            };
            let mut found = model.predict(&prefix, &opts)?;
            // the guard re-ranks what the search already offered: the best
            // continuation the negative network does not veto
            let mut guard = Json::Null;
            if let Some((mut negative, config)) = ctx.open_guard()? {
                let mut pair = Filter::new(&mut model, &mut negative, config)?;
                let (ranked, verdicts) = pair.rank(&prefix, &found);
                let kept = verdicts.iter().filter(|v| v.decision != "reject").count();
                guard = guard_report(
                    &pair,
                    &verdicts,
                    vec![
                        ("candidates", Json::Int(verdicts.len() as i64)),
                        ("kept", Json::Int(kept as i64)),
                    ],
                );
                found = ranked;
            }
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
            doc.push(("guard".to_string(), guard));
            emit(Json::Obj(doc));
        }
        "generate" => {
            let mut model = open(true)?;
            let seeded = args.on("seeded");
            let tuning = search_tuning(&args)?;
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
                top_k: tuning.filter.top_k,
                top_p: tuning.filter.top_p,
                min_p: tuning.filter.min_p,
                diversity: tuning.diversity,
            };
            // the pair: the model over-samples, the negative network vetoes,
            // the cleanest survivors come back
            let (samples, guard) = match ctx.open_guard()? {
                Some((mut negative, config)) => {
                    let mut pair = Filter::new(&mut model, &mut negative, config)?;
                    let outcome = pair.generate(opts.count, &opts)?;
                    let report = guard_report(
                        &pair,
                        &outcome.verdicts,
                        vec![
                            ("candidates", Json::Int(outcome.candidates as i64)),
                            ("asked", Json::Int(outcome.asked as i64)),
                            ("kept", Json::Int(outcome.kept.len() as i64)),
                            ("rate", outcome.rate.map(Json::Num).unwrap_or(Json::Null)),
                        ],
                    );
                    (outcome.results, report)
                }
                None => (model.generate(&opts)?, Json::Null),
            };
            emit(Json::obj([
                ("samples", Json::Arr(samples.iter().map(path_json).collect())),
                ("count", Json::Int(samples.len() as i64)),
                ("mode", Json::str(opts.mode.clone())),
                ("prefix", Json::str(opts.prefix.clone())),
                ("max_length", Json::Int(opts.max_length as i64)),
                ("temperature", Json::Num(opts.temperature)),
                ("traversal", Json::str(opts.traversal.clone())),
                ("guard", guard),
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
            let good = read_named(&args, "good")?;
            let bad = read_named(&args, "bad")?;
            if good.is_empty() && bad.is_empty() {
                return Err("nothing to learn from: give --good (thumbs up) and / or --bad (thumbs down)".to_string());
            }
            // rated sets are small: the feedback defaults, and a mark out of 10 per text
            let mut o = feedback_options(&args, Feedback::rated())?;
            o.good_weights = radixnet::kinds::marks(&args.str("good-ratings", ""), good.len(), "--good-ratings")?;
            o.bad_weights = radixnet::kinds::marks(&args.str("bad-ratings", ""), bad.len(), "--bad-ratings")?;
            let action = radixnet::kinds::feedback_action(&good, &bad).unwrap_or("punish");
            let (negative, positive) = match action {
                "2nrl" => radixnet::kinds::two_nrl(&mut model, &bad, &good, &o, &mut |_| true)?,
                "reward" => (
                    Vec::new(),
                    radixnet::kinds::reward(&mut model, &good, &o, &mut |_| true)?,
                ),
                _ => (
                    radixnet::kinds::punish(&mut model, &bad, &o, &mut |_| true)?,
                    Vec::new(),
                ),
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
            let o = feedback_options(&args, Feedback::two_nrl())?;
            let (negative, positive) = radixnet::kinds::two_nrl(&mut model, &bad, &good, &o, &mut |_| true)?;
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
            let kind = model.kind();
            let Some(mine) = radixnet::kinds::cli_weight_names(kind) else {
                return Err(format!(
                    "{model_path} holds a {kind} model; the weight function belongs to the count and resonant \
                     models (--kind count / --kind resonant)"
                ));
            };
            let flag = |name: &str| format!("--{}", name.replace('_', "-"));
            let mut stray: Vec<&str> = radixnet::kinds::WEIGHT_FIELDS
                .iter()
                .copied()
                .filter(|name| !mine.contains(name) && args.get(&flag(name)[2..]).is_some())
                .collect();
            stray.sort_unstable();
            if !stray.is_empty() {
                return Err(format!(
                    "{} do(es) not apply to the {kind} model; it takes {}",
                    stray.iter().map(|n| flag(n)).collect::<Vec<_>>().join(", "),
                    mine.iter().map(|n| flag(n)).collect::<Vec<_>>().join(", ")
                ));
            }
            let mut changes: Vec<(String, f64)> = Vec::new();
            for name in mine {
                if let Some(text) = args.get(&flag(name)[2..]) {
                    let value: f64 = text.parse().map_err(|_| format!("{} needs a number", flag(name)))?;
                    if !value.is_finite() {
                        return Err(format!("{} must be a finite number", flag(name)));
                    }
                    changes.push((name.to_string(), value));
                }
            }
            let mut saved = Json::Null;
            if !changes.is_empty() {
                radixnet::kinds::configure_weights(&mut model, &changes)?;
                let path = save(&mut model)?;
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                saved = Json::obj([("path", Json::str(path)), ("bytes", Json::Int(bytes as i64))]);
            }
            emit(Json::obj([
                ("weights", radixnet::kinds::weight_config(&model)),
                (
                    "changed",
                    Json::Obj(changes.iter().map(|(k, v)| (k.clone(), Json::Num(*v))).collect()),
                ),
                ("saved", saved),
                ("stats", stats(&model)),
            ]));
        }
        "schedule" => {
            // the radix model's learning rates as a graph function of the epoch
            let lr_schedule = args.get("lr-schedule").map(str::to_string).filter(|s| !s.is_empty());
            let act_lr_schedule = args
                .get("act-lr-schedule")
                .map(str::to_string)
                .filter(|s| !s.is_empty());
            if lr_schedule.is_none() && act_lr_schedule.is_none() {
                let info = radixnet::schedule::describe();
                emit(Json::obj([
                    ("presets", info.at("presets").clone()),
                    ("variables", info.at("variables").clone()),
                    ("functions", info.at("functions").clone()),
                    ("helpers", info.at("helpers").clone()),
                ]));
            } else {
                let epochs = args.usize("epochs", 10)?;
                let lr = args.float("lr", TrainConfig::default().lr)?;
                let act_lr = args.float("act-lr", TrainConfig::default().act_lr)?;
                let reverse = args.on("reverse-schedule");
                let points = radixnet::schedule::preview_points(
                    lr_schedule.as_deref(),
                    act_lr_schedule.as_deref(),
                    epochs as i64,
                    lr,
                    act_lr,
                    reverse,
                )?;
                let text = |s: &Option<String>| s.clone().map(Json::str).unwrap_or(Json::Null);
                emit(Json::obj([
                    ("lr_schedule", text(&lr_schedule)),
                    ("act_lr_schedule", text(&act_lr_schedule)),
                    ("epochs", Json::Int(epochs as i64)),
                    ("lr", Json::Num(lr)),
                    ("act_lr", Json::Num(act_lr)),
                    ("reverse_schedule", Json::Bool(reverse)),
                    ("points", Json::Arr(points.iter().map(|p| p.to_json()).collect())),
                ]));
            }
        }
        "paths" => {
            let model = open(true)?;
            if model.kind() != "count" {
                return Err(format!(
                    "{model_path} holds a {} model; path counters belong to the count model",
                    model.kind()
                ));
            }
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
            if model.kind() != "count" {
                return Err(format!(
                    "{model_path} holds a {} model; node ratios belong to the count model",
                    model.kind()
                ));
            }
            let node = args.get("node").map(|n| n.parse::<usize>().unwrap_or(usize::MAX));
            emit(Json::obj([
                ("nodes", node_rows(&model.g, args.usize("limit", 20)?, node)),
                ("stats", stats(&model)),
            ]));
        }
        "negative" => {
            // the negative model lives beside --model (model.count.json ->
            // model.count.negative.json) unless --negative says otherwise
            let path = ctx.negative_path();
            let open_negative = |must_exist: bool| ctx.open_negative(must_exist);
            match args.action() {
                "blame" => {
                    let texts = read_texts(&args)?;
                    if texts.is_empty() {
                        return Err("give the failed text with --text or --data".to_string());
                    }
                    let mut model = open_negative(false)?;
                    let records = model.blame(
                        &texts,
                        &NegBlameOptions {
                            reason: args.str("reason", "unspecified"),
                            severity: args.float("severity", 1.0)?,
                            source: args.str("source", "cli"),
                            note: args.str("note", ""),
                            epochs: args.usize("epochs", 1)?,
                            no_compress: args.on("no-compress"),
                        },
                    )?;
                    let saved = args.str("out", &path);
                    model.save(&saved)?;
                    emit(Json::obj([
                        ("blamed", Json::Int(texts.len() as i64)),
                        (
                            "reason",
                            Json::str(radixnet::negative::clean_reason(&args.str("reason", "unspecified"))),
                        ),
                        ("severity", Json::Num(args.float("severity", 1.0)?)),
                        ("saved", Json::str(saved)),
                        ("records", Json::Arr(records.iter().map(|r| r.to_json()).collect())),
                        ("stats", negative_stats(&mut model)),
                    ]));
                }
                "clear" => {
                    let texts = read_texts(&args)?;
                    if texts.is_empty() {
                        return Err("give the passed text with --text or --data".to_string());
                    }
                    let mut model = open_negative(true)?;
                    let edges = model.clear_text(&texts, args.float("weight", 1.0)?)?;
                    let saved = args.str("out", &path);
                    model.save(&saved)?;
                    emit(Json::obj([
                        ("cleared", Json::Int(texts.len() as i64)),
                        ("edges", Json::Int(edges as i64)),
                        ("saved", Json::str(saved)),
                        ("stats", negative_stats(&mut model)),
                    ]));
                }
                "why" => {
                    let texts = read_texts(&args)?;
                    if texts.is_empty() {
                        return Err("give a text to judge with --text or --data".to_string());
                    }
                    let mut model = open_negative(true)?;
                    let opts = JudgeOptions {
                        threshold: args.get("threshold").map(|t| t.parse().unwrap_or(f64::NAN)),
                        min_coverage: args.get("min-coverage").map(|t| t.parse().unwrap_or(f64::NAN)),
                        spans: args.usize("spans", 5)?,
                    };
                    let verdicts: Vec<Json> = texts
                        .iter()
                        .filter_map(|t| model.judge(t, &opts))
                        .map(|v| verdict_json(&v))
                        .collect();
                    emit(Json::obj([
                        ("verdicts", Json::Arr(verdicts.clone())),
                        ("count", Json::Int(verdicts.len() as i64)),
                        ("stats", negative_stats(&mut model)),
                    ]));
                }
                "reasons" => {
                    let mut model = open_negative(true)?;
                    let limit = args.usize("limit", 20)?;
                    let rows: Vec<Json> = model
                        .reasons()
                        .iter()
                        .take(if limit == 0 { usize::MAX } else { limit })
                        .map(|r| {
                            Json::obj([
                                ("reason", Json::str(r.reason.clone())),
                                ("blame", Json::Num(r.blame)),
                                ("fails", Json::Int(r.fails)),
                                ("share", Json::Num(r.share)),
                            ])
                        })
                        .collect();
                    let journal: Vec<Json> = model
                        .recent(limit)
                        .iter()
                        .map(|e| {
                            Json::obj([
                                ("at", Json::str(e.at.clone())),
                                ("text", Json::str(e.text.clone())),
                                ("reason", Json::str(e.reason.clone())),
                                ("severity", Json::Num(e.severity)),
                                ("source", Json::str(e.source.clone())),
                                ("note", Json::str(e.note.clone())),
                            ])
                        })
                        .collect();
                    emit(Json::obj([
                        ("reasons", Json::Arr(rows)),
                        ("journal", Json::Arr(journal)),
                        ("stats", negative_stats(&mut model)),
                    ]));
                }
                "forget" => {
                    let mut model = open_negative(true)?;
                    let reason = args.get("reason").map(|s| s.to_string());
                    let result = model.g.forget(reason.as_deref(), args.float("factor", 0.0)?)?;
                    let saved = args.str("out", &path);
                    model.save(&saved)?;
                    emit(Json::obj([
                        ("reason", Json::str(result.reason)),
                        ("edges", Json::Int(result.edges as i64)),
                        ("blame_removed", Json::Num(result.blame_removed)),
                        ("saved", Json::str(saved)),
                        ("stats", negative_stats(&mut model)),
                    ]));
                }
                "settings" => {
                    let mut model = open_negative(true)?;
                    let threshold = args.get("threshold").map(|t| t.parse().unwrap_or(f64::NAN));
                    let min_coverage = args.get("min-coverage").map(|t| t.parse().unwrap_or(f64::NAN));
                    model.set_judgement(threshold, min_coverage)?;
                    let saved = args.str("out", &path);
                    model.save(&saved)?;
                    let neg = model.neg.as_ref().expect("a negative model");
                    emit(Json::obj([
                        ("threshold", Json::Num(neg.threshold)),
                        ("min_coverage", Json::Num(neg.min_coverage)),
                        ("saved", Json::str(saved)),
                    ]));
                }
                "filter" => {
                    // the pair at work: the positive model writes, the
                    // negative one vetoes - or the given texts are judged
                    let mut negative = open_negative(true)?;
                    let mut positive = open(true)?;
                    let config = FilterConfig {
                        threshold: args.maybe_float("threshold")?,
                        min_coverage: args.maybe_float("min-coverage")?,
                        ratio: if args.on("no-ratio") {
                            None
                        } else {
                            Some(args.float("ratio", 0.0)?)
                        },
                        peak: args.maybe_float("peak")?,
                        over_sample: args.usize("over-sample", 3)?.max(1),
                        strict: args.on("strict"),
                        spans: args.usize("spans", 3)?,
                        learn: args.on("learn"),
                        reason: args.str("reason", "filtered"),
                    };
                    let given = read_texts(&args)?;
                    let mut pair = Filter::new(&mut positive, &mut negative, config)?;
                    let outcome = if given.is_empty() {
                        pair.generate(
                            args.usize("count", 3)?,
                            &GenerateOptions {
                                max_length: args.usize("max-length", 60)?,
                                mode: args.str("mode", "sample"),
                                temperature: args.float("temperature", 1.0)?,
                                prefix: args.str("prefix", ""),
                                seed: args.get("seed").map(|_| seed),
                                step_penalty: args.float("step-penalty", 0.0)?,
                                beam: args.usize("beam", 0)?,
                                ..Default::default()
                            },
                        )?
                    } else {
                        pair.filter(&given)?
                    };
                    let learn = pair.config.learn;
                    let described = pair.describe();
                    let Json::Obj(mut doc) = outcome.to_json() else { unreachable!() };
                    if learn {
                        negative.save(&path)?;
                        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        doc.push((
                            "saved".to_string(),
                            Json::obj([("path", Json::str(path.clone())), ("bytes", Json::Int(bytes as i64))]),
                        ));
                    }
                    doc.push(("pair".to_string(), described));
                    emit(Json::Obj(doc));
                }
                // the model writes, an LLM reviews, the failures blame (critic.rs)
                "auto" => radixnet::critic::cli(&ctx)?,
                "" => {
                    return Err(
                        "negative needs an action: blame, clear, why, reasons, forget, settings, filter, auto".to_string(),
                    )
                }
                other => {
                    return Err(format!(
                        "unknown negative action {other:?}; expected blame, clear, why, reasons, forget, settings, filter or auto"
                    ))
                }
            }
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
            service.configure(&args)?;
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
                ("meta", radixnet::kinds::meta_json(&model)),
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
        "" => return Err(format!("no command\n{}", usage())),
        other => match dispatch(&ctx) {
            Some(outcome) => outcome?,
            None => return Err(format!("unknown command {other:?}\n{}", usage())),
        },
    }
    Ok(())
}

/// The radix model's training settings from the flags (`--lr`, `--act-lr`,
/// the schedules, `--batch-size`, `--no-compress`, `--backend`), checked as
/// Python checks them; the other kinds read only `epochs` and compression.
fn train_config(args: &radixnet::cli::Args, epochs: usize) -> Result<TrainConfig, String> {
    radixnet::backend::resolve(&args.str("backend", "auto"))?;
    let base = TrainConfig::default();
    let text = |flag: &str| args.get(flag).map(str::to_string).filter(|s| !s.trim().is_empty());
    let cfg = TrainConfig {
        epochs,
        lr: args.float("lr", base.lr)?,
        act_lr: args.float("act-lr", base.act_lr)?,
        batch_size: args.usize("batch-size", base.batch_size)?,
        auto_compress: !args.on("no-compress"),
        lr_schedule: text("lr-schedule"),
        act_lr_schedule: text("act-lr-schedule"),
        reverse_schedule: args.on("reverse-schedule"),
        plan: plan_flags(args)?,
        ..base
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A count flag that refuses a negative value rather than reading it as 0 -
/// `--replay-size -1` must not drop a buffer.
fn count_flag(args: &radixnet::cli::Args, name: &str, fallback: usize) -> Result<usize, String> {
    let value = args.int(name, fallback as i64)?;
    if value < 0 {
        return Err(format!("--{name} must be >= 0, got {value}"));
    }
    Ok(value as usize)
}

/// The sampling filters and the beam's diversity (`--top-k`, `--top-p`,
/// `--min-p`, `--diversity`; `../SPEC-SearchAndTraining.md` sections 1-2),
/// each off by default.
fn search_tuning(args: &radixnet::cli::Args) -> Result<SearchTuning, String> {
    let tuning = SearchTuning {
        filter: SamplingFilter {
            top_k: count_flag(args, "top-k", 0)?,
            top_p: args.float("top-p", 1.0)?,
            min_p: args.float("min-p", 0.0)?,
        },
        diversity: args.float("diversity", 0.0)?,
        ..Default::default()
    };
    tuning.check()?;
    Ok(tuning)
}

/// How a training run walks its texts (`--order`, `--curriculum`,
/// `--replay`, `--replay-size`, `--patience`, `--min-delta`; the spec's
/// sections 3-6), each off by default.
fn plan_flags(args: &radixnet::cli::Args) -> Result<Plan, String> {
    let plan = Plan {
        order: args.str("order", "corpus"),
        curriculum: args.float("curriculum", 1.0)?,
        replay: args.float("replay", 0.0)?,
        replay_size: match args.get("replay-size") {
            Some(_) => Some(count_flag(args, "replay-size", 0)?),
            None => None,
        },
        patience: count_flag(args, "patience", 0)?,
        min_delta: args.float("min-delta", 0.0)?,
    };
    plan.check()?;
    Ok(plan)
}

/// A feedback command's settings over the command's defaults: the epochs and
/// strength every kind reads, the learning rates and batch size the radix
/// model reads.
fn feedback_options(args: &radixnet::cli::Args, base: Feedback) -> Result<Feedback, String> {
    radixnet::backend::resolve(&args.str("backend", "auto"))?;
    Ok(Feedback {
        neg_epochs: args.usize("neg-epochs", base.neg_epochs)?,
        pos_epochs: args.usize("pos-epochs", base.pos_epochs)?,
        neg_lr: args.float("neg-lr", base.neg_lr)?,
        pos_lr: args.float("pos-lr", base.pos_lr)?,
        strength: args.float("strength", base.strength)?,
        batch_size: match args.get("batch-size") {
            Some(_) => Some(args.usize("batch-size", 1)?),
            None => base.batch_size,
        },
        ..base
    })
}

/// Reads a document without building a model - what `--json` callers use to
/// check a file this port wrote.
#[allow(dead_code)]
fn peek(path: &str) -> Result<Json, String> {
    read_document(path)
}
