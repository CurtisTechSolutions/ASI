//! The command line's plumbing, in the library so every module can own its
//! own commands: the flags, the context a command runs in (which model, which
//! encoding, where it is saved, how the answer is printed), and the table that
//! sends a command name to the module that answers it.
//!
//! `src/bin/radixnet.rs` keeps the model's own commands (`train`, `predict`,
//! ...) and hands every other name to [`dispatch`]; a module that adds a
//! command adds one row to [`COMMANDS`] and a `pub fn cli(ctx: &Ctx)`.
//! The commands over the model's own modules that the binary does not keep -
//! `correct`, `attention` and `window` - are the submodules here.

pub mod attention;
pub mod correct;
pub mod window;

use radixnet::encoding::{base64_file_texts, base64_prefix, Encoding};
use radixnet::file::read_file_text;
use radixnet::json::Json;
use radixnet::model::Model;
use radixnet::negative::{NegativeOptions, Verdict};
use radixnet::report::{split_texts, stats};
use radixnet::search::PathResult;

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

/// `--auto-temperature T` on `predict`, `generate`, `converse` and `talk`: the
/// ceiling a search with no answer climbs to - nothing said, no end reached
/// with `--to-end`, or everything vetoed - walking again up a ladder of
/// temperatures, the asked one and then doubling up to T, three walks a rung,
/// until one answers (`../../../RadixCyclicNN/SPEC-SearchAndTraining.md`
/// section 10).  0, the default, is off; 2 is a good ceiling: hotter walks end no more
/// often and say nothing more often.  Refused, in Python's words, unless it is
/// a finite number `>= 0`.
pub fn auto_temperature(args: &Args) -> Result<f64, String> {
    let Some(text) = args.get("auto-temperature") else {
        return Ok(0.0);
    };
    let value: f64 = text.trim().parse().map_err(|_| {
        format!(
            "argument --auto-temperature: expected a number, got {}",
            radixnet::negative::python_repr(text)
        )
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "argument --auto-temperature: must be a finite number >= 0, got {text}"
        ));
    }
    Ok(value)
}

/// Flags that take no value - every boolean flag of the Python CLI and the Go
/// one, so a command line written for either parses here.
pub const SWITCHES: &[&str] = &[
    "2nrl",
    "allow-private",
    "allow-repeats",
    "allow-word-repeats",
    "auto",
    "blame",
    "browser",
    "correct",
    "dry-run",
    "exact",
    "json",
    "json-answer",
    "learn",
    "learn-thinking",
    "lenient",
    "listen-back",
    "manual",
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
    "no-learn-reply",
    "no-model",
    "no-network-isolation",
    "no-provenance",
    "no-questions",
    "no-ratio",
    "no-replay",
    "no-solve",
    "no-speak",
    "no-teach",
    "no-teach-answer",
    "no-teach-partner",
    "no-think",
    "no-think-questions",
    "no-to-end",
    "no-train",
    "no-waveform",
    "normalise",
    "off",
    "offline",
    "on",
    "pair",
    "parallel-parts",
    "play",
    "python-tool",
    "quiet",
    "raw",
    "read-reward",
    "resume",
    "reverse",
    "reverse-schedule",
    "sample-first",
    "save",
    "seeded",
    "shared-token",
    "stream",
    "strict",
    "to-end",
    "train",
    "verbose",
    "whole-file",
    "with-answers",
];

/// Flags whose value may be left out, with what a bare one means (Python's
/// `nargs="?"` and `const`): `--plan` alone plans the default three lessons,
/// `--step` alone takes one step of the dynamic window.
pub const OPTIONAL_VALUES: &[(&str, &str)] = &[("plan", "3"), ("step", "1")];

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
            if !wanted.trim().is_empty() && radixnet::kinds::parse_kind(&wanted).is_ok_and(|k| k != m.kind()) {
                // the file's own kind wins, as it does in Python
                radixnet::log_warn!(
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
        let mut m = radixnet::kinds::new_model(radixnet::kinds::parse_kind(&wanted)?, self.seed, self.encoding, &[])?;
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
    /// settings of `--threshold`, `--min-coverage` and `--over-sample` (and
    /// `--no-provenance`: veto without saying why); `None` when `--no-guard`
    /// was given, when there is no negative network beside the model, or when
    /// the one there has never been taught a failure and so would veto
    /// nothing (Python's `open_guard`).
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
            provenance: !self.args.on("no-provenance"),
            ..Default::default()
        };
        if let Some(g) = negative.g.neg.as_ref() {
            radixnet::log_info!(
                "negative",
                "guard: {path} ({} blame over {} reasons)",
                radixnet::json::py_repr(g.total_blame),
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
        "talk",
        crate::assistant::cli,
        "talk to the model in today's format: messages in, a reply out, the thinking first, streamed",
    ),
    (
        "chat",
        crate::chat::cli,
        "an LLM converses with the model and marks every reply",
    ),
    (
        "correct",
        crate::cli::correct::cli,
        "teach one correction: only what changed moves",
    ),
    (
        "attention",
        crate::cli::attention::cli,
        "the attention band: where inside a gram a correction's blame and credit land",
    ),
    (
        "window",
        crate::cli::window::cli,
        "the dynamic window: a ladder of node sizes, halving from 32 to 4 and back up",
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
    // a recording is heard as one text of acoustic units: one utterance, one text
    if radixnet::phonetic::is_audio_file(path) {
        let data = std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))?;
        return Ok(vec![
            radixnet::phonetic::hear_audio(&data).map_err(|err| format!("{path}: {err}"))?
        ]);
    }
    let unit = args.str("split", "lines");
    let page_lines = args.usize("page-lines", 0).unwrap_or(0);
    // a ZIP archive (by its magic, whatever its name) is read entry by entry
    if radixnet::zip::is_zip_file(path) {
        let workers = args.usize("workers", 0).unwrap_or(0);
        let source = radixnet::source::source_for_file(std::path::Path::new(path), &unit, page_lines, workers);
        return radixnet::source::collect_texts(source.as_ref()).map_err(|err| format!("{path}: {err}"));
    }
    let content = read_file_text(path)?;
    Ok(split_texts(&content, &unit, page_lines))
}

// -- a model of base64 symbols: content in, content out --------------------------------------------

/// Whether every file is to be read as one text: `--whole-file`, or this
/// port's own way of saying it, `--split file`.
pub fn whole_file(args: &Args) -> bool {
    args.on("whole-file")
        || matches!(
            args.str("split", "lines").trim().to_ascii_lowercase().as_str(),
            "file" | "whole" | "whole-file"
        )
}

/// A file's bytes as they are; a missing one is `data file not found: PATH`
/// (Python's `read_file_bytes`).
pub fn read_file_bytes(path: &str) -> Result<Vec<u8>, String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("data file not found: {path}"));
    }
    std::fs::read(path).map_err(|err| format!("{path}: {err}"))
}

/// The texts a model of base64 symbols reads from every file a flag names
/// (`--data notes.txt --data photo.png`), each file as the bytes it holds,
/// written in its symbols ([`radixnet::encoding::base64_file_texts`]): a
/// UTF-8 text file one text per line, anything else - an image, a video, a
/// recording, an archive - one text of every byte, and every file one text
/// with `whole_file`.  Nothing is unpacked or heard first: a ZIP or a WAV is
/// just bytes.
///
/// An absent flag reads as nothing; files that give no text at all are
/// refused, `no {what} texts found in ...` - Python's `read_texts(paths,
/// whole_file, what, encoding)`.
pub fn read_symbols(args: &Args, flag: &str, whole_file: bool, what: &str) -> Result<Vec<String>, String> {
    let paths = args.all(flag);
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let mut texts = Vec::new();
    for path in &paths {
        texts.extend(base64_file_texts(&read_file_bytes(path)?, whole_file));
    }
    if texts.is_empty() {
        return Err(format!("no {what} texts found in {}", paths.join(", ")));
    }
    Ok(texts)
}

/// The texts of one named file flag as a model of `enc` reads them: a model
/// of base64 symbols reads every file's bytes ([`read_symbols`]), each file
/// one text when `--whole-file` (or `--split file`) says so; any other reads
/// the file as text, split as `--split` asks ([`read_named`]).
pub fn read_named_as(args: &Args, flag: &str, enc: Encoding, what: &str) -> Result<Vec<String>, String> {
    if enc.is_base64() {
        return read_symbols(args, flag, whole_file(args), what);
    }
    read_named(args, flag)
}

/// The texts of every `--text` and of `--data` as a model of `enc` reads
/// them: a model of base64 symbols learns the UTF-8 bytes of every `--text`
/// and the bytes of every `--data` file - each file one text with
/// `--whole-file` - all written in its symbols; any other model reads them as
/// [`read_texts`] does.
pub fn read_texts_as(args: &Args, enc: Encoding, what: &str) -> Result<Vec<String>, String> {
    if !enc.is_base64() {
        return read_texts(args);
    }
    let mut texts: Vec<String> = args.all("text").iter().map(|t| enc.text_input(t)).collect();
    texts.extend(read_symbols(args, "data", whole_file(args), what)?);
    Ok(texts)
}

/// The prefix a model is asked, and the content it stands for, as bytes
/// (Python's `_prefix_of`).
///
/// `--prefix-file` gives a prefix as the bytes of a file, `--prefix` as text
/// (its UTF-8 bytes): one or the other, and `required` says whether the
/// command needs one at all (`predict` does; `generate` starts from START).
/// A model of base64 symbols is asked the symbols those bytes fix
/// ([`radixnet::encoding::base64_prefix`]), so its walk writes on from the
/// last whole symbol; any other model reads the file as UTF-8 text and takes
/// `--prefix` as it is.
pub fn prefix_of(args: &Args, enc: Encoding, required: bool) -> Result<(String, Vec<u8>), String> {
    let (text, file) = (args.get("prefix"), args.get("prefix-file"));
    if text.is_some() && file.is_some() {
        return Err("argument --prefix-file: not allowed with argument --prefix".to_string());
    }
    if let Some(path) = file.filter(|p| !p.is_empty()) {
        let data = read_file_bytes(path)?;
        if enc.is_base64() {
            return Ok((base64_prefix(&data), data));
        }
        let asked = enc
            .content_text(&data)
            .map_err(|err| format!("{path} is not valid UTF-8 text: {err}"))?;
        return Ok((asked, data));
    }
    if required && text.is_none() && file.is_none() {
        return Err("one of the arguments --prefix --prefix-file is required".to_string());
    }
    let text = text.unwrap_or("");
    Ok((enc.prefix_input(text), text.as_bytes().to_vec()))
}

/// What a document adds about the prefix it was asked (Python's
/// `_prefix_doc`): the file it came from, and the symbols a model of base64
/// symbols was asked.
pub fn prefix_doc(args: &Args, enc: Encoding, asked: &str) -> Vec<(String, Json)> {
    let mut doc = Vec::new();
    if let Some(path) = args.get("prefix-file").filter(|p| !p.is_empty()) {
        doc.push(("prefix_file".to_string(), Json::str(path)));
    }
    if enc.is_base64() {
        doc.push(("prefix_symbols".to_string(), Json::str(asked)));
    }
    doc
}

/// `os.path.splitext`: `("out", ".png")` for `out.png`.  Only the last part
/// of a path has an extension, and a name's leading dots are not one
/// (`.hidden` has none).
pub fn split_ext(path: &str) -> (&str, &str) {
    let name = path.rfind('/').map_or(0, |at| at + 1);
    match path.rfind('.') {
        Some(dot) if dot > name && path[name..dot].bytes().any(|b| b != b'.') => (&path[..dot], &path[dot..]),
        _ => (path, ""),
    }
}

/// `--decode-to PATH` on predict and generate: what each text holds, written
/// as bytes - the output decoder of a model of base64 symbols (Python's
/// `_decode_to`).
///
/// One text is written to `PATH`; several to `PATH` numbered before its
/// extension (`out-1.png`, `out-2.png`, ...).  A model of base64 symbols
/// writes the decoded bytes - an image, a video, a text again - and any other
/// model its text as UTF-8.  Answers `[{"path", "bytes"}]`, a row per file;
/// each write is said on the log, never on stdout, which is the document's.
pub fn decode_to(path: &str, enc: Encoding, texts: &[String]) -> Result<Json, String> {
    let (root, ext) = split_ext(path);
    let mut written = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let target = if texts.len() == 1 {
            path.to_string()
        } else {
            format!("{root}-{}{ext}", i + 1)
        };
        let data = enc.content_bytes(text);
        std::fs::write(&target, &data).map_err(|err| format!("{target}: {err}"))?;
        radixnet::log_info!("model", "wrote {target} ({} bytes)", data.len());
        written.push(Json::obj([
            ("path", Json::str(target)),
            ("bytes", Json::Int(data.len() as i64)),
        ]));
    }
    Ok(Json::Arr(written))
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
    fn the_ceiling_is_a_finite_number_no_smaller_than_zero() {
        let ceiling = |flags: &[&str]| {
            let (_, args) = parse_args(&argv(flags)).unwrap();
            auto_temperature(&args)
        };
        assert_eq!(ceiling(&["predict"]), Ok(0.0), "off unless asked");
        assert_eq!(ceiling(&["predict", "--auto-temperature", "2"]), Ok(2.0));
        assert_eq!(ceiling(&["converse", "--auto-temperature=0.5"]), Ok(0.5));
        assert_eq!(
            ceiling(&["talk", "--auto-temperature", "-1"]),
            Err("argument --auto-temperature: must be a finite number >= 0, got -1".to_string())
        );
        assert_eq!(
            ceiling(&["generate", "--auto-temperature", "hot"]),
            Err("argument --auto-temperature: expected a number, got 'hot'".to_string())
        );
        assert!(ceiling(&["predict", "--auto-temperature", "inf"]).is_err());
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
        // --reverse is a switch: the flag after it keeps its own value
        let (_, args) = parse_args(&argv(&["train", "--reverse", "--data", "corpus.txt"])).unwrap();
        assert!(args.on("reverse"));
        assert_eq!(args.get("data"), Some("corpus.txt"));
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

    // -- a model of base64 symbols (Python's tests/test_base64.py) --------------------------

    /// The head of a PNG and some bytes that are no text.
    const BLOB: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1,
    ];

    /// A directory of the test's own, made fresh.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("radixnet-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base64() -> Encoding {
        radixnet::encoding::parse_encoding("base64").unwrap()
    }

    #[test]
    fn an_extension_is_split_off_as_python_splits_it() {
        for (path, root, ext) in [
            ("out.png", "out", ".png"),
            ("out", "out", ""),
            ("dir/out.tar.gz", "dir/out.tar", ".gz"),
            (".hidden", ".hidden", ""),
            ("dir.d/out", "dir.d/out", ""),
            ("a/.b.c", "a/.b", ".c"),
            ("..x", "..x", ""),
            ("a..b", "a.", ".b"),
            ("dir/", "dir/", ""),
        ] {
            assert_eq!(split_ext(path), (root, ext), "{path}");
        }
    }

    #[test]
    fn a_model_of_base64_symbols_reads_files_as_their_bytes() {
        let dir = scratch("symbols");
        let text = dir.join("corpus.txt");
        std::fs::write(&text, "the cat\n\n  \nthe dog\r\n").unwrap();
        let blob = dir.join("image.png");
        std::fs::write(&blob, BLOB).unwrap();
        let (text, blob) = (text.to_string_lossy().into_owned(), blob.to_string_lossy().into_owned());
        let (_, args) = parse_args(&argv(&["train", "--data", &text, "--data", &blob, "--text", "hi"])).unwrap();
        let read = read_texts_as(&args, base64(), "training").unwrap();
        let b64 = radixnet::encoding::base64_text;
        assert_eq!(read, vec![b64(b"hi"), b64(b"the cat"), b64(b"the dog\r"), b64(BLOB)]);
        // each file one text, every byte of it
        let (_, args) = parse_args(&argv(&["train", "--data", &text, "--whole-file"])).unwrap();
        assert_eq!(
            read_texts_as(&args, base64(), "training").unwrap(),
            vec![b64(b"the cat\n\n  \nthe dog\r\n")]
        );
        // a text model reads the text as it always did
        let (_, args) = parse_args(&argv(&["score", "--data", &text])).unwrap();
        assert_eq!(
            read_named_as(&args, "data", Encoding::default(), "scorable").unwrap(),
            vec!["the cat", "the dog"]
        );
        // a missing file, and files with no text in them, say so
        let (_, args) = parse_args(&argv(&["score", "--data", "/no/such/file"])).unwrap();
        assert_eq!(
            read_named_as(&args, "data", base64(), "scorable").unwrap_err(),
            "data file not found: /no/such/file"
        );
        let blank = dir.join("blank.txt").to_string_lossy().into_owned();
        std::fs::write(&blank, " \n\n").unwrap();
        let (_, args) = parse_args(&argv(&["2nrl", "--bad", &blank])).unwrap();
        assert_eq!(
            read_named_as(&args, "bad", base64(), "bad").unwrap_err(),
            format!("no bad texts found in {blank}")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_prefix_is_text_or_a_file_and_never_both() {
        let dir = scratch("prefix");
        let head = dir.join("head.bin").to_string_lossy().into_owned();
        std::fs::write(&head, &BLOB[..8]).unwrap();
        let ask = |items: &[&str], enc: Encoding, required: bool| {
            let (_, args) = parse_args(&argv(items)).unwrap();
            prefix_of(&args, enc, required)
        };
        // the symbols the bytes fix, and the bytes themselves
        let (asked, content) = ask(&["predict", "--prefix-file", &head], base64(), true).unwrap();
        assert_eq!(asked, radixnet::encoding::base64_prefix(&BLOB[..8]));
        assert_eq!(content, &BLOB[..8]);
        let (asked, content) = ask(&["predict", "--prefix", "the d"], base64(), true).unwrap();
        assert_eq!((asked.as_str(), content.as_slice()), ("dGhlIG", &b"the d"[..]));
        // a text model takes text as it is, and a file only when it is text
        let (asked, _) = ask(&["predict", "--prefix", "the d"], Encoding::default(), true).unwrap();
        assert_eq!(asked, "the d");
        let refused = ask(&["predict", "--prefix-file", &head], Encoding::default(), true).unwrap_err();
        assert!(refused.contains("is not valid UTF-8 text"), "{refused}");
        // one or the other; predict needs one, generate starts from nothing
        assert!(ask(&["predict", "--prefix", "a", "--prefix-file", &head], base64(), false).is_err());
        assert!(ask(&["predict"], base64(), true).is_err());
        assert_eq!(
            ask(&["generate"], base64(), false).unwrap(),
            (String::new(), Vec::new())
        );
        let (_, args) = parse_args(&argv(&["predict", "--prefix-file", &head])).unwrap();
        let keys: Vec<String> = prefix_doc(&args, base64(), "iVBORw0KGg")
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(keys, ["prefix_file", "prefix_symbols"]);
        assert_eq!(
            prefix_doc(&args, Encoding::default(), "x").len(),
            1,
            "no symbols but the base64 model's"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_to_writes_what_each_text_holds() {
        let dir = scratch("decode-to");
        let out = dir.join("out.png").to_string_lossy().into_owned();
        let symbols = radixnet::encoding::base64_text(BLOB);
        let one = decode_to(&out, base64(), std::slice::from_ref(&symbols)).unwrap();
        assert_eq!(one.as_array()[0].at("path").as_str(), Some(out.as_str()));
        assert_eq!(one.as_array()[0].at("bytes").as_i64(), Some(BLOB.len() as i64));
        assert_eq!(std::fs::read(&out).unwrap(), BLOB);
        // several are numbered before the extension; a text model writes its text
        let two = decode_to(&out, Encoding::default(), &["a".to_string(), "caf\u{e9}".to_string()]).unwrap();
        let paths: Vec<&str> = two.as_array().iter().filter_map(|w| w.at("path").as_str()).collect();
        let numbered = |i: usize| dir.join(format!("out-{i}.png")).to_string_lossy().into_owned();
        assert_eq!(paths, [numbered(1), numbered(2)]);
        assert_eq!(std::fs::read(numbered(2)).unwrap(), "caf\u{e9}".as_bytes());
        assert_eq!(two.as_array()[1].at("bytes").as_i64(), Some(5));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
