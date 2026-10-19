//! Exercise the public CLI, including EOS counting and warmup exclusion.
use std::process::Command;

use latticefsm::chain::Chain;
use latticefsm::json::{parse, Json};
use latticefsm::ngram::NGram;

fn invoke(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_latticefsm"))
        .arg("benchmark")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn counts_generated_tokens_not_requested_tokens_or_warmup() {
    let dir = std::env::temp_dir().join(format!("latticefsm-benchmark-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lines = vec!["alpha beta gamma".to_string(); 20];
    let mut ngram = NGram::new(3, "word", true, 1.0).unwrap();
    ngram.train_lines(&lines, None);
    let mut chain = Chain::new(3, "word", true, 1.0).unwrap();
    chain.train_lines(&lines, None);
    for model in ["ngram", "chain"] {
        let file = dir.join(format!("{model}.json"));
        let path = file.to_str().unwrap();
        if model == "ngram" {
            ngram.save(path).unwrap();
        } else {
            chain.save(path).unwrap();
        }
        let before = std::fs::read(path).unwrap();
        for (prompt, limit, expected) in [
            ("alpha", "100", 8.0),
            ("alpha", "1", 4.0),
            ("alpha beta gamma", "100", 0.0),
        ] {
            let output = invoke(&[
                "--model",
                model,
                "--load",
                path,
                "--text",
                prompt,
                "--runs",
                "4",
                "--warmup",
                "3",
                "--max-tokens",
                limit,
                "--json",
            ]);
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            let data = parse(&String::from_utf8(output.stdout).unwrap()).unwrap();
            let row = &data.as_array().unwrap()[0];
            assert_eq!(row.num("tokens", -1.0), expected);
            assert_eq!(row.str_or("unit", ""), "word");
            let seconds = row.num("seconds", 0.0);
            let rate = row.num("tokens_per_second", -1.0);
            assert!(seconds > 0.0 && seconds.is_finite());
            assert!(rate.is_finite());
            assert!((rate * seconds - expected).abs() < 1e-6);
        }
        assert_eq!(
            before,
            std::fs::read(path).unwrap(),
            "benchmark must not save over the model"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validates_benchmark_options_before_reading_data() {
    for args in [
        vec!["--runs", "0"],
        vec!["--runs", "1.5"],
        vec!["--warmup", "-1"],
        vec!["--max-tokens", "0"],
        vec!["--model", "other"],
        vec!["--temperature", "NaN"],
        vec!["--temperature", "-1"],
        vec!["--seed", "no"],
        vec!["--load", "unused.json"],
    ] {
        let out = invoke(&args);
        assert!(!out.status.success(), "accepted {args:?}");
        assert!(!out.stderr.is_empty());
    }
}

#[test]
fn benchmarks_both_models_from_a_corpus_and_rejects_empty_data() {
    let dir = std::env::temp_dir().join(format!("latticefsm-benchmark-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("corpus.txt");
    std::fs::write(&file, "alpha beta gamma\n".repeat(20)).unwrap();
    let args = [
        "--data",
        file.to_str().unwrap(),
        "--runs",
        "2",
        "--warmup",
        "0",
        "--text",
        "alpha",
        "--json",
    ];
    let out = invoke(&args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let data = parse(&String::from_utf8(out.stdout).unwrap()).unwrap();
    let rows = data.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.num("tokens", 0.0) == 4.0));
    assert_eq!(rows[0].get("model"), Some(&Json::from("chain")));
    std::fs::write(&file, "").unwrap();
    assert!(!invoke(&args).status.success());
    std::fs::remove_dir_all(dir).unwrap();
}
