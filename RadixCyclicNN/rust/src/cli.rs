//! The command line's plumbing, in the library so every module can own its
//! own commands: the flags, the context a command runs in (which model, which
//! encoding, where it is saved, how the answer is printed), and the table that
//! sends a command name to the module that answers it.
//!
//! `src/bin/radixnet.rs` keeps the model's own commands (`train`, `predict`,
//! ...) and hands every other name to [`dispatch`]; a module that adds a
//! command adds one row to [`COMMANDS`] and a `pub fn cli(ctx: &Ctx)`.

use crate::encoding::Encoding;
use crate::json::Json;
use crate::model::Model;
use crate::negative::{NegativeOptions, Verdict};
use crate::report::{split_texts, stats};
use crate::search::PathResult;

/// The command line: global flags, a command, and that command's flags.
#[derive(Clone, Debug, Default)]
pub struct Args {
    pub flags: Vec<(String, String)>,
    pub switches: Vec<String>,
    /// Bare words after the command, for the commands that take an action
    /// (`negative blame`, `tutor lesson`, ...).
    pub rest: Vec<String>,
}

impl Args {
    /// A flag's value, or `None`.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
    pub fn str(&self, name: &str, fallback: &str) -> String {
        self.get(name).unwrap_or(fallback).to_string()
    }
    /// A switch: `--name`, or `--name=true` / `--name=1` (and `--name=false`
    /// turns it off again).
    pub fn on(&self, name: &str) -> bool {
        match self.get(name) {
            Some(value) => matches!(value.to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on"),
            None => self.switches.iter().any(|s| s == name),
        }
    }
    /// Every value of a repeatable flag, in the order given.
    pub fn all(&self, name: &str) -> Vec<String> {
        self.flags
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .collect()
    }
    /// The action word after the command, or `""`.
    pub fn action(&self) -> &str {
        self.rest.first().map(|s| s.as_str()).unwrap_or("")
    }
    pub fn int(&self, name: &str, fallback: i64) -> Result<i64, String> {
        match self.get(name) {
            Some(text) => text
                .parse()
                .map_err(|_| format!("--{name} needs a number, got {text:?}")),
            None => Ok(fallback),
        }
    }
    pub fn usize(&self, name: &str, fallback: usize) -> Result<usize, String> {
        Ok(self.int(name, fallback as i64)?.max(0) as usize)
    }
    pub fn float(&self, name: &str, fallback: f64) -> Result<f64, String> {
        match self.get(name) {
            Some(text) => text
                .parse()
                .map_err(|_| format!("--{name} needs a number, got {text:?}")),
            None => Ok(fallback),
        }
    }
    /// A number that may be left out: `None` when the flag is absent.
    pub fn maybe_float(&self, name: &str) -> Result<Option<f64>, String> {
        match self.get(name) {
            Some(_) => Ok(Some(self.float(name, 0.0)?)),
            None => Ok(None),
        }
    }
}

/// Flags that take no value - every boolean flag of the Python CLI and the Go
/// one, so a command line written for either parses here.
pub const SWITCHES: &[&str] = &[
    "2nrl",
    "allow-private",
    "allow-repeats",
    "allow-word-repeats",
    "blame",
    "browser",
    "dry-run",
    "exact",
    "json",
    "json-answer",
    "learn",
    "learn-thinking",
    "lenient",
    "listen-back",
    "no-adapt",
    "no-avoid",
    "no-blame",
    "no-clear",
    "no-compress",
    "no-count",
    "no-diff-corrections",
    "no-fallback-teacher",
    "no-guard",
    "no-headless",
    "no-judge",
    "no-learn",
    "no-model",
    "no-network-isolation",
    "no-questions",
    "no-ratio",
    "no-replay",
    "no-solve",
    "no-teach",
    "no-teach-answer",
    "no-teach-partner",
    "no-think",
    "no-think-questions",
    "no-to-end",
    "no-waveform",
    "normalise",
    "offline",
    "pair",
    "parallel-parts",
    "python-tool",
    "quiet",
    "read-reward",
    "resume",
    "reverse-schedule",
    "sample-first",
    "save",
    "seeded",
    "shared-token",
    "strict",
    "to-end",
    "train",
    "verbose",
    "whole-file",
    "with-answers",
];

/// Flags whose value may be left out, with what a bare one means (Python's
/// `nargs="?"` and `const`): `--plan` alone plans the default three lessons.
pub const OPTIONAL_VALUES: &[(&str, &str)] = &[("plan", "3")];

/// Splits a command line into the command and its flags.
pub fn parse_args(argv: &[String]) -> Result<(String, Args), String> {
    let mut command = String::new();
    let mut args = Args::default();
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
                let bare = OPTIONAL_VALUES.iter().find(|(flag, _)| *flag == name);
                let value = match (inline, bare) {
                    (Some(v), _) => v,
                    // a bare optional flag: nothing follows, or another flag does
                    (None, Some((_, meaning))) if argv.get(i + 1).is_none_or(|next| next.starts_with("--")) => {
                        meaning.to_string()
                    }
                    (None, _) => {
                        i += 1;
                        argv.get(i).cloned().ok_or_else(|| format!("--{name} needs a value"))?
                    }
                };
                args.flags.push((name, value));
            }
        } else if command.is_empty() {
            command = item.clone();
        } else {
            args.rest.push(item.clone());
        }
        i += 1;
    }
    Ok((command, args))
}

/// Everything a command runs with: its flags and the global ones, resolved.
pub struct Ctx {
    pub command: String,
    pub args: Args,
    /// `--model`, or the default file of the encoding's unit.
    pub model_path: String,
    /// The encoding a *new* model is built with (a loaded file's own wins).
    pub encoding: Encoding,
    pub seed: i64,
    pub workers: usize,
    /// `--json`: one document on stdout, rendered compactly.
    pub json: bool,
    /// `--out`: where a changed model is saved instead of `--model`.
    pub out: Option<String>,
}

impl Ctx {
    /// The model at `--model`; a fresh one when there is none and `must_exist`
    /// is false.
    pub fn open(&self, must_exist: bool) -> Result<Model, String> {
        // `--kind` is the kind of a NEW model (count unless it says otherwise)
        let wanted = self.args.str("kind", "");
        if std::path::Path::new(&self.model_path).exists() {
            let mut m = Model::load(&self.model_path)?;
            m.workers = self.workers;
            m.g.workers = self.workers;
            if !wanted.trim().is_empty() && crate::kinds::parse_kind(&wanted).is_ok_and(|k| k != m.kind()) {
                // the file's own kind wins, as it does in Python
                crate::log_warn!(
                    "model",
                    "note: {} holds a {} model; --kind {} applies to new models only",
                    self.model_path,
                    m.kind(),
                    wanted.trim()
                );
            }
            return Ok(m);
        }
        if must_exist {
            return Err(format!("no model at {}", self.model_path));
        }
        let mut m = crate::kinds::new_model(crate::kinds::parse_kind(&wanted)?, self.seed, self.encoding, &[])?;
        m.workers = self.workers;
        m.g.workers = self.workers;
        Ok(m)
    }

    /// Saves a changed model to `--out`, else back to `--model`; says where.
    pub fn save(&self, model: &mut Model) -> Result<String, String> {
        let path = self.out.clone().unwrap_or_else(|| self.model_path.clone());
        model.save(&path)?;
        Ok(path)
    }

    /// Prints the command's answer: compact under `--json`, indented otherwise.
    pub fn emit(&self, doc: Json) {
        if self.json {
            println!("{}", doc.render(0));
        } else {
            println!("{}", doc.render(2));
        }
    }

    /// The texts of every `--text` and of `--data`, split as `--split` asks.
    pub fn texts(&self) -> Result<Vec<String>, String> {
        read_texts(&self.args)
    }

    /// Where the negative network lives: `--negative`, else beside `--model`.
    pub fn negative_path(&self) -> String {
        self.args.str("negative", &negative_path(&self.model_path))
    }

    /// The negative network at [`Ctx::negative_path`]; a fresh one when there
    /// is none and `must_exist` is false.
    pub fn open_negative(&self, must_exist: bool) -> Result<Model, String> {
        let path = self.negative_path();
        if std::path::Path::new(&path).exists() {
            let mut m = Model::load(&path)?;
            if !m.is_negative() {
                return Err(format!("{path} holds a {} model, not a negative one", m.kind()));
            }
            m.workers = self.workers;
            m.g.workers = self.workers;
            return Ok(m);
        }
        if must_exist {
            return Err(format!(
                "no negative model at {path} (teach it one first with \
                 `radixnet negative blame --text '...' --reason gibberish`)"
            ));
        }
        let mut m = Model::new_negative(
            self.seed,
            &NegativeOptions {
                encoding: self.encoding,
                ..Default::default()
            },
        )?;
        m.workers = self.workers;
        m.g.workers = self.workers;
        Ok(m)
    }
}

impl Ctx {
    /// The negative network guarding the output of `--model`, with the filter
    /// settings of `--threshold`, `--min-coverage` and `--over-sample`; `None`
    /// when `--no-guard` was given, when there is no negative network beside
    /// the model, or when the one there has never been taught a failure and
    /// so would veto nothing (Python's `open_guard`).
    pub fn open_guard(&self) -> Result<Option<(Model, crate::duo::FilterConfig)>, String> {
        if self.args.on("no-guard") {
            return Ok(None);
        }
        let path = self.negative_path();
        if !std::path::Path::new(&path).is_file() {
            return Ok(None);
        }
        let negative = self.open_negative(true)?;
        if !crate::duo::ready(&negative) {
            return Ok(None);
        }
        let config = crate::duo::FilterConfig {
            threshold: self.args.maybe_float("threshold")?,
            min_coverage: self.args.maybe_float("min-coverage")?,
            over_sample: self.args.usize("over-sample", 3)?,
            ..Default::default()
        };
        if let Some(g) = negative.g.neg.as_ref() {
            crate::log_info!(
                "negative",
                "guard: {path} ({} blame over {} reasons)",
                crate::json::py_repr(g.total_blame),
                g.reason_names.len()
            );
        }
        Ok(Some((negative, config)))
    }
}

/// A command a module answers.
pub type Command = fn(&Ctx) -> Result<(), String>;

/// Every command outside the model's own, with the module that answers it and
/// the line `radixnet --help` prints for it.
pub const COMMANDS: &[(&str, Command, &str)] = &[
    ("converse", crate::dialogue::cli, "the model converses with itself"),
    (
        "think",
        crate::thinking::cli,
        "the model thinks: one thought from the THINK sentinel, questioning itself where it learned to",
    ),
    (
        "chat",
        crate::chat::cli,
        "an LLM converses with the model and marks every reply",
    ),
    (
        "correct",
        crate::correct::cli,
        "teach one correction: only what changed moves",
    ),
    (
        "evolve",
        crate::gan::cli,
        "GAN-style self-upgrading loop (generator vs discriminator)",
    ),
    ("checkpoints", crate::checkpoint::cli, "list checkpoints or restore one"),
    (
        "image",
        crate::vision::cli,
        "images as text: encode, decode, and the recall tutor",
    ),
    (
        "speech",
        crate::speech::cli,
        "the waveform behind one unique token: teach, decode, the recall tutor",
    ),
    (
        "ollama",
        crate::ollama::cli,
        "a local Ollama LLM: models, a corpus from a prompt, the adversarial review, thoughts",
    ),
    (
        "chatgpt",
        crate::chatgpt::cli,
        "query ChatGPT (OpenAI): check the key and models, ask",
    ),
    (
        "tutor",
        crate::tutor::cli,
        "automated English lessons: the teacher writes the exercise and marks it",
    ),
    (
        "codegen",
        crate::codegen::cli,
        "generate programs, run them in a sandbox, judge them",
    ),
    (
        "tools",
        crate::tools::cli,
        "list the external tools, or call one directly",
    ),
    ("agent", crate::agent::cli, "solve tasks with tools"),
    (
        "explore",
        crate::agent::explore_cli,
        "let the network browse on its own",
    ),
    (
        "mcp",
        crate::mcp::cli,
        "serve the tools and the network over the Model Context Protocol",
    ),
    ("bench", crate::bench::cli, "throughput benchmark"),
];

/// Runs a command from [`COMMANDS`]; `None` when no module answers the name.
pub fn dispatch(ctx: &Ctx) -> Option<Result<(), String>> {
    COMMANDS
        .iter()
        .find(|(name, _, _)| *name == ctx.command)
        .map(|(_, run, _)| run(ctx))
}

/// What a module says for a command it does not answer yet.
pub fn not_ported(what: &str) -> String {
    format!("{what} is not in the Rust port yet")
}

/// The texts of every `--text` and of `--data`.
pub fn read_texts(args: &Args) -> Result<Vec<String>, String> {
    let mut texts = args.all("text");
    texts.extend(read_named(args, "data")?);
    Ok(texts)
}

/// The texts of one named file flag, split as `--split` asks; an absent flag
/// reads as nothing.
pub fn read_named(args: &Args, flag: &str) -> Result<Vec<String>, String> {
    let Some(path) = args.get(flag) else {
        return Ok(Vec::new());
    };
    let unit = args.str("split", "lines");
    let page_lines = args.usize("page-lines", 0).unwrap_or(0);
    // a ZIP archive (by its magic, whatever its name) is read entry by entry
    if crate::zip::is_zip_file(path) {
        let workers = args.usize("workers", 0).unwrap_or(0);
        let source = crate::source::source_for_file(std::path::Path::new(path), &unit, page_lines, workers);
        return crate::source::collect_texts(source.as_ref()).map_err(|err| format!("{path}: {err}"));
    }
    let content = read_file_text(path)?;
    Ok(split_texts(&content, &unit, page_lines))
}

/// A text file, gunzipped when its name ends in `.gz`.
pub fn read_file_text(path: &str) -> Result<String, String> {
    if path.ends_with(".gz") {
        let bytes = std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))?;
        String::from_utf8(crate::gzip::decompress(&bytes)?).map_err(|err| format!("{path}: {err}"))
    } else {
        std::fs::read_to_string(path).map_err(|err| format!("cannot read {path}: {err}"))
    }
}

/// The negative model beside a model file: `model.count.json` ->
/// `model.count.negative.json`, with `.gz` kept on the end.
pub fn negative_path(model: &str) -> String {
    let (stem, ext) = match model.strip_suffix(".gz") {
        Some(head) => match head.rfind('.') {
            Some(at) => (&head[..at], format!("{}.gz", &head[at..])),
            None => (head, ".gz".to_string()),
        },
        None => match model.rfind('.') {
            Some(at) => (&model[..at], model[at..].to_string()),
            None => (model, String::new()),
        },
    };
    if stem.ends_with(".negative") {
        return model.to_string();
    }
    format!("{stem}.negative{ext}")
}

/// A verdict as the CLI and the API print it.
pub fn verdict_json(v: &Verdict) -> Json {
    Json::obj([
        ("text", Json::str(v.text.clone())),
        ("chars", Json::Int(v.chars as i64)),
        ("transitions", Json::Int(v.transitions as i64)),
        ("known", Json::Int(v.known as i64)),
        ("blamed", Json::Int(v.blamed as i64)),
        ("coverage", Json::Num(v.coverage)),
        ("blame", Json::Num(v.blame)),
        ("risk", Json::Num(v.risk)),
        ("peak", Json::Num(v.peak)),
        ("per_char", Json::Num(v.per_char)),
        ("threshold", Json::Num(v.threshold)),
        ("min_coverage", Json::Num(v.min_coverage)),
        ("verdict", Json::str(v.verdict.clone())),
        (
            "reasons",
            Json::Arr(
                v.reasons
                    .iter()
                    .map(|r| {
                        Json::obj([
                            ("reason", Json::str(r.reason.clone())),
                            ("blame", Json::Num(r.blame)),
                            ("share", Json::Num(r.share)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "spans",
            Json::Arr(
                v.spans
                    .iter()
                    .map(|s| {
                        Json::obj([
                            ("start", Json::Int(s.start as i64)),
                            ("end", Json::Int(s.end as i64)),
                            ("fragment", Json::str(s.fragment.clone())),
                            ("blame", Json::Num(s.blame)),
                            ("fails", Json::Int(s.fails)),
                            ("clear", Json::Num(s.clear)),
                            ("reason", Json::str(s.reason.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("why", Json::str(v.why.clone())),
    ])
}

/// The statistics of a negative model, as the CLI and the API report them
/// (Python's `NegativeNet.stats`).
pub fn negative_stats(model: &mut Model) -> Json {
    stats(model)
}

/// One walk, as the CLI reports it.
pub fn path_json(result: &PathResult) -> Json {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flags_switches_and_actions_come_apart() {
        let (command, args) = parse_args(&argv(&[
            "negative",
            "blame",
            "--text",
            "a b",
            "--json",
            "--epochs=3",
            "--strict=false",
            "--text",
            "c",
        ]))
        .unwrap();
        assert_eq!(command, "negative");
        assert_eq!(args.action(), "blame");
        assert_eq!(args.all("text"), vec!["a b", "c"]);
        assert!(args.on("json"));
        assert!(!args.on("strict"), "--strict=false turns it off");
        assert_eq!(args.usize("epochs", 1).unwrap(), 3);
        assert_eq!(args.maybe_float("threshold").unwrap(), None);
        assert!(parse_args(&argv(&["train", "--epochs"])).is_err());
        // a bare --plan means the default three lessons, as Python's const does
        let (_, args) = parse_args(&argv(&["tutor", "--plan", "--rounds", "2"])).unwrap();
        assert_eq!((args.get("plan"), args.get("rounds")), (Some("3"), Some("2")));
        let (_, args) = parse_args(&argv(&["tutor", "--plan"])).unwrap();
        assert_eq!(args.get("plan"), Some("3"));
        let (_, args) = parse_args(&argv(&["tutor", "--plan", "5"])).unwrap();
        assert_eq!(args.get("plan"), Some("5"));
    }

    #[test]
    fn the_thinking_switches_take_no_value() {
        // a switch that took a value would swallow the flag after it: --no-think would read "--think-depth"
        let (_, args) = parse_args(&argv(&["converse", "--no-think", "--think-depth", "0"])).unwrap();
        assert!(args.on("no-think"));
        assert_eq!(args.get("think-depth"), Some("0"));
        let (_, args) = parse_args(&argv(&[
            "ollama",
            "think",
            "--with-answers",
            "--no-questions",
            "--prompt",
            "the sea",
        ]))
        .unwrap();
        assert!(args.on("with-answers") && args.on("no-questions"));
        assert_eq!((args.action(), args.get("prompt")), ("think", Some("the sea")));
        let (_, args) = parse_args(&argv(&[
            "tutor",
            "--learn-thinking",
            "--no-think-questions",
            "--think",
            "high",
        ]))
        .unwrap();
        assert!(args.on("learn-thinking") && args.on("no-think-questions"));
        assert_eq!(args.get("think"), Some("high"));
        for switch in [
            "learn-thinking",
            "no-questions",
            "no-think",
            "no-think-questions",
            "with-answers",
        ] {
            assert!(SWITCHES.contains(&switch), "{switch} is not a switch");
        }
    }

    #[test]
    fn the_negative_network_lives_beside_the_model() {
        assert_eq!(negative_path("model.count.json"), "model.count.negative.json");
        assert_eq!(negative_path("m.json.gz"), "m.negative.json.gz");
        assert_eq!(negative_path("m.negative.json"), "m.negative.json");
        assert_eq!(negative_path("plain"), "plain.negative");
    }

    #[test]
    fn every_command_has_one_home() {
        let mut names: Vec<&str> = COMMANDS.iter().map(|(n, _, _)| *n).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "a command is listed twice");
    }
}
