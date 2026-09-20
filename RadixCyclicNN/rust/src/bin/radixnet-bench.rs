//! `radixnet-bench` - how fast the Rust port counts and predicts.
//!
//! ```text
//! radixnet-bench [--chars N] [--epochs N] [--predictions N] [--seed N]
//!                [--workers N] [--traversal reward|least-punished]
//!                [--punish-every N] [--texts FILE] [--prefixes FILE] [--data FILE]
//!                [--dump FILE] [--json]
//! ```
//!
//! `--texts` / `--prefixes` are how the cross-language comparison runs it: both
//! ports are handed the same corpus and the same prefixes, so neither is timed
//! on work the other did not do.

use std::process::ExitCode;

use radixnet::bench::{run_benchmark, BenchOptions, BENCH_DEFAULT_CHARS, BENCH_DEFAULT_EPOCHS};
use radixnet::encoding::Encoding;
use radixnet::parallel::effective_workers;
use radixnet::search::parse_traversal;

const USAGE: &str = "usage: radixnet-bench [--chars N] [--epochs N] [--predictions N] [--seed N]\n\
                     \x20                     [--workers N] [--traversal reward|least-punished]\n\
                     \x20                     [--punish-every N] [--texts FILE] [--prefixes FILE]\n\
                     \x20                     [--data FILE] [--dump FILE] [--json]\n\
                     \x20                     [--encoding SPEC | --units char|word --ngram N --stride N]\n\
                     \n\
                     the encoding is how a text becomes grams: --units says what one unit is, --ngram how many\n\
                     units a gram holds, --stride how far apart two grams start (1 = the sliding window, n =\n\
                     non-overlapping groups of n).  --encoding sets all three: char:3:1 (the default, and what\n\
                     the Python and Go benchmarks measure), char:5:5 (groups of five letters), word:2:1.";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("radixnet-bench: {err}");
            ExitCode::FAILURE
        }
    }
}

/// One text per line, blank lines dropped.
fn file_lines(path: &str) -> Result<Vec<String>, String> {
    if path.trim().is_empty() {
        return Ok(Vec::new());
    }
    let data = std::fs::read_to_string(path).map_err(|err| format!("cannot read {path}: {err}"))?;
    let lines: Vec<String> = data
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.to_string())
        .collect();
    if lines.is_empty() {
        return Err(format!("{path} holds no usable line"));
    }
    Ok(lines)
}

fn run() -> Result<(), String> {
    let mut o = BenchOptions {
        chars: BENCH_DEFAULT_CHARS,
        epochs: BENCH_DEFAULT_EPOCHS,
        ..Default::default()
    };
    let mut json_mode = false;
    let (mut texts_path, mut prefixes_path) = (String::new(), String::new());

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--json" => json_mode = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--chars" => o.chars = value(&mut i)?.parse().map_err(|_| "chars must be a number")?,
            "--epochs" => o.epochs = value(&mut i)?.parse().map_err(|_| "epochs must be a number")?,
            "--predictions" => o.predictions = value(&mut i)?.parse().map_err(|_| "predictions must be a number")?,
            "--seed" => o.seed = value(&mut i)?.parse().map_err(|_| "seed must be a number")?,
            "--workers" => o.workers = value(&mut i)?.parse().map_err(|_| "workers must be a number")?,
            "--traversal" => o.traversal = parse_traversal(&value(&mut i)?)?,
            "--encoding" => o.encoding = Encoding::parse(&value(&mut i)?)?,
            "--units" => o.encoding.unit = Encoding::parse(&value(&mut i)?)?.unit,
            "--ngram" => {
                let n = value(&mut i)?.parse().map_err(|_| "ngram must be a number")?;
                // a bare --ngram keeps a sliding encoding sliding, and grows a group
                let stride = if o.encoding.sliding() {
                    o.encoding.stride.min(n)
                } else {
                    n
                };
                o.encoding = Encoding {
                    n,
                    stride,
                    ..o.encoding
                };
                o.encoding.validate()?;
            }
            "--stride" => {
                let stride = value(&mut i)?.parse().map_err(|_| "stride must be a number")?;
                o.encoding = Encoding { stride, ..o.encoding };
                o.encoding.validate()?;
            }
            "--punish-every" => o.punish_every = value(&mut i)?.parse().map_err(|_| "punish-every must be a number")?,
            "--dump" => o.dump = value(&mut i)?,
            "--texts" => texts_path = value(&mut i)?,
            "--prefixes" => prefixes_path = value(&mut i)?,
            "--data" => o.corpus_path = value(&mut i)?,
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }
    o.texts = file_lines(&texts_path)?;
    o.prefixes = file_lines(&prefixes_path)?;

    let pool = effective_workers(o.workers);
    if o.texts.is_empty() {
        eprintln!(
            "benchmark: {} chars, {} epoch(s), {} traversal, exact counting, {pool} worker(s)",
            o.chars, o.epochs, o.traversal
        );
    } else {
        eprintln!(
            "benchmark: {} text(s) from {texts_path}, {} epoch(s), {} traversal, exact counting, {pool} worker(s)",
            o.texts.len(),
            o.epochs,
            o.traversal
        );
    }
    let result = run_benchmark(&o)?;
    if json_mode {
        println!("{}", result.render(0));
        return Ok(());
    }
    if let radixnet::json::Json::Obj(pairs) = &result {
        let mut rows: Vec<(&String, &radixnet::json::Json)> = pairs
            .iter()
            .filter(|(k, _)| k != "sample_prediction")
            .map(|(k, v)| (k, v))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));
        for (key, value) in rows {
            let text = match value {
                radixnet::json::Json::Num(x) if *x >= 1000.0 => format!("{x:.0}"),
                radixnet::json::Json::Num(x) => format!("{x:.4}"),
                radixnet::json::Json::Str(s) => s.clone(),
                other => other.render(0),
            };
            println!("{key:<28} {text}");
        }
        if let Some((_, radixnet::json::Json::Obj(sample))) = pairs.iter().find(|(k, _)| k == "sample_prediction") {
            let field = |name: &str| {
                sample
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.render(0))
                    .unwrap_or_else(|| "null".to_string())
            };
            println!();
            println!(
                "{:<28} {} -> {}",
                "sample_prediction",
                field("prefix"),
                field("continuation")
            );
        }
    }
    Ok(())
}
