//! `latticefsm <command>` - every command the machine has.
//!
//!     demo        a machine learns a language by reward and punishment, is stimulated, and lets time pass
//!     train       teach a language by credit over random strings, and save the machine
//!     run         read a string from a saved machine, traversing it (or --quiet), and save what that did
//!     teach       traverse one edge deliberately and credit it
//!     accuracy    how many random strings of a language a saved machine classifies right (nothing moves)
//!     table       the greedy transition table of a saved machine (nothing moves)
//!     stats       the size, the memory, the clock and the stimulation of a machine
//!     tick        let time pass
//!     stimulate   raise the stimulation
//!     compress    fold the matrix into its central node (exact, float32 or float16) and save the code
//!     expand      rebuild a machine from a code, from the central node outward (all shells, or --shells k)
//!     core-run    walk a string straight from a code, decoding only the cells the walk reaches
//!     network     machines connected into a network: new, fit, evaluate, predict, translate, summary
//!     ngram       predict the next thing: an N-gram model trained on a corpus or with Ollama
//!     chain       one input to a node, the output node predicts the next: walks only as deep as confidence needs
//!     benchmark   generation tokens per second for the chain and N-gram models
//!     experiment  the learning, stimulation, adaptation and compression experiments
//!     serve       the HTTP API and the React frontend (frontend/dist) on --port
//!
//! Every command but `train`, `demo`, `experiment` and `serve` takes `--load` (a machine file written by
//! `train` or by any command's `--save`); `serve` and `train` take it too, and start fresh without it.

use std::env;
use std::process;

use latticefsm::autotemp;
use latticefsm::autotrain::{self, Ollama};
use latticefsm::b64;
use latticefsm::chain::{load_chain, Chain};
use latticefsm::compress::{compress, expansion, fidelity, load_core, Core, Precision};
use latticefsm::experiment::{self, teach_language};
use latticefsm::json::Json;
use latticefsm::languages::{all_strings, examples, language, LANGUAGES};
use latticefsm::machine::{load_machine, Machine, Mode, Settings, Walk, DEFAULT_ALPHABET, DEFAULT_STATES};
use latticefsm::media;
use latticefsm::network::{load_network, Credit, Merge, Network, Vote, LAYER_DISCOUNT};
use latticefsm::ngram::{corpus_sequences, detokenize, load_ngram, split_lines, NGram};
use latticefsm::phonetic;
use latticefsm::rng::Rng;
use latticefsm::server;

struct Args {
    command: String,
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Args {
        let mut argv = env::args().skip(1);
        let command = argv.next().unwrap_or_else(|| "help".to_string());
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let rest: Vec<String> = argv.collect();
        let mut i = 0;
        while i < rest.len() {
            let a = &rest[i];
            // a short flag (`-k 5`) is read as the long one of the same name
            let short =
                (a.len() == 2 && a.starts_with('-') && a.as_bytes()[1].is_ascii_alphabetic()).then(|| format!("-{a}"));
            let a = short.as_ref().unwrap_or(a);
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    flags.push((k.to_string(), v.to_string()));
                } else if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                    flags.push((name.to_string(), rest[i + 1].clone()));
                    i += 1;
                } else {
                    flags.push((name.to_string(), "true".to_string()));
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Args {
            command,
            positional,
            flags,
        }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn num(&self, name: &str, default: f64) -> f64 {
        self.get(name).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    fn str(&self, name: &str, default: &str) -> String {
        self.get(name).unwrap_or(default).to_string()
    }
}

const HELP: &str = "latticefsm - a finite state machine over a dense 3D matrix of adaptive edges

usage: latticefsm <command> [options]

commands
  demo        learn a language, be stimulated, let time pass: the model in one screen
  train       teach --language over --episodes random strings; --save the machine
  run TEXT    read TEXT from the start state and traverse it (--quiet: ask only); --credit R rewards (R<0 punishes);
              --from-middle starts from the middle state, the central node's
  compress    fold the matrix into its central node; --precision exact|float32|float16 or --budget BYTES;
              --core FILE saves the code
  expand      --core FILE [--shells K]: rebuild a machine from the central node outward; --save it
  core-run TEXT --core FILE [--from-middle]: the greedy walk read straight from the code
  focus       --level F (0..1) | --none | --learn on|off [--text T]: the node of the central vertical vector runs start
              from - none or 0 the top, 1 the bottom - or a focus learned from the input
  skip on|off [--margin M]: let runs skip a node when a two-edge path is more efficient
  rearrange   [--full] | --swap states|symbols I J: let the nodes rearrange themselves toward the centre, or swap two
  teach S SYM T [--amount R]   traverse one edge deliberately and credit it
  accuracy    the share of --tests random strings of --language classified right (nothing moves)
  table       the greedy transition table (nothing moves)
  stats       size, memory, clock and stimulation
  tick        let --ticks pass
  stimulate   raise the stimulation by --amount (or set --level)
  network ACTION   machines connected into a network (latticefsm/network.py is the same):
              new        --arch 13>(13|13)>13 (> in a row, (a|b) side by side, 13f final state only, enc / dec
                         English in / out through the phonetic tokenizer)  --alphabet ab  --accepting 0
                         --credit shared|own  --merge concat|interleave  --vote majority|any|all
                         --layer-discount 0.8  --seed 1, and the machine options; --save FILE
              fit        --language L --episodes N [--epochs E]: teach a language; or --task plural|past
                         [--pairs FILE.tsv] [--epochs 20]: English to English pairs (every 4th held out)
              evaluate   accuracy on every string up to --max-length (6), or exact translations of --task
              predict TEXT...    each string's verdict, layer by layer
              translate TEXT...  English in, English out
              summary    the blocks, their matrices and what they write
              encode INPUT...    what the encoder turns each input (a text or a file) into
              fit / evaluate --data FILE.tsv   input<TAB>verdict lines (1/0, yes/no), paths relative to the
                         file, every 5th held out: with --arch text>13, audio>13, image>13, diff>13 or video>13
                         (text, a WAV, a PNG / PPM / BMP, a directory of frames or a video file through ffmpeg)
              make-data  --kind text|image|audio|video --out DIR [--count 40]: a small synthetic labelled set
  ngram ACTION     predict the next thing (latticefsm/ngram.py is the same model, number for number):
              train      --data FILES/DIRS (data/corpus) --order 3 --unit word|char|sound|base64 --held-out 10 --save F;
                         every token is predicted before it is learned (--no-predict: only count; --self-credit R:
                         reward a right first guess, punish a wrong one)
              autotrain  the corpus, then --ollama rounds: sentences on --topics (default: the corpus's own words),
                         continuations of its contexts, a judge's credit on its texts; --rounds 3 --lines 20
                         --continuations 20 --judge 10 --patience 2 --keep FILE --save F --model M --url U
              predict TEXT... [-k 10]   the likeliest next tokens        generate TEXT... [--temperature 0]
                         [--auto-temperature [--max-temperature 0.8] [--tries 3]]: a reply that is nothing, no
                         words or a loop drawn again, 0.2 hotter each rung, until it is an answer
              evaluate --data ...       perplexity and next-token accuracy on held-out lines
              stats | credit TEXT --amount R | ollama (is it there, which models)
  chain ACTION     one input to a node (\"abc\": a in node 1, b in node 2, c in node 3), the output node predicts the
                   next, walking back only until it has gathered --confidence (latticefsm/chain.py, number for number):
              train      --data ... --nodes 8 --unit char|word|sound|base64 --confidence 0.9 --held-out 10 --save F
                         (--no-predict, --self-credit R as for ngram)
              predict TEXT... [-k 10]   the input nodes, the walk and the likeliest next inputs
              generate TEXT... [--max-tokens 40]   the continuation and how deep every walk went
                         [--auto-temperature ...] as for ngram
              evaluate | sweep [--thresholds 0,0.5,0.9,1]   accuracy and depth on held-out lines
              stats | credit TEXT --amount R   (without --load the data is learned first; --set-confidence T)
              teach TEXT REPLY [--max-rounds 100] [--threshold 0.95]; train --until [--max-rounds 20] [--threshold]
  guard ACTION [TEXT...] --load F --save F   the negative network: blame TEXT --reason R --severity S | clear TEXT |
              judge TEXT (reject / suspect / pass, why) | forget --reason R | reset | stats
  corpus PATH... [--unit text|base64] [--show N]   what a corpus is read as: every document (zip, tar, tar.gz,
              gz, html, xml, docx, odt, epub, jsonl opened) and its lines, or its base64 sequences
  base64 encode|decode INPUT... [--out FILE]   anything (a file, or a text) as base64, or base64 back to bytes;
              ngram --unit base64 learns any file - text, image, audio, video - letter by letter
  benchmark   --model all|chain|ngram --data FILES/DIRS --unit word|char|sound|base64
              --text \"the brain is\" --runs 100 --warmup 3 --max-tokens 128 --temperature 0 --seed 1
              --order 3 --nodes 8 --confidence 0.9; --load FILE for one model; --json for raw results
  experiment  --which learning|stimulation|adaptation|compression|walk|network|all, --out DIR
  serve       the HTTP API and the frontend on --port (default 8000); --frontend-dir DIR (default: frontend/dist
              under the working directory, or ../frontend/dist)

machine options (train, demo, serve without --load)
  --states N (13)  --alphabet abcdefghijklm  (a 13 x 13 x 13 matrix)  --accepting 0,2  --start 0  --life 1000
  --baseline 1  --calm LIFE
  --temperature 1  --discount 0.8  --seed 1  --use-widening 0.01  --reward-widening 0.2  --punish-narrowing 0.2
  --compress-every N (0: never)  --compress-precision exact|float32|float16  --compress-rebuild
  --focus F (0..1)  --learn-focus  --skip  --skip-margin M  --rearrange-every N (0: never)
  --mode own|explore  how a row's edges are weighed: own (each edge's own weighting) or explore (least traversed
                      and most rewarded first: rewarded - punished + E sqrt(ln(1+N)/(1+seen)), N the row's traversals)
  --equal-reward      a reward lands on every edge of the run in full (punishment is still discounted from the end)
  --explore E         explore's bonus for a little-traversed edge (1.414, UCB1's sqrt 2; 0: the most rewarded only)
common options
  --load FILE  --save FILE (.json or .json.gz)  --stimulation X (run)  --temperature T (run, accuracy)
  --language even-b|contains-aa|ends-ab|mod3-a  --episodes N  --max-length 6  --tests 300
";

fn settings(a: &Args) -> Result<Settings, String> {
    let d = Settings::default();
    Ok(Settings {
        start: a.num("start", 0.0) as usize,
        life: a.num("life", d.life),
        baseline: a.num("baseline", d.baseline),
        calm: a.get("calm").and_then(|v| v.parse().ok()),
        temperature: a.num("temperature", d.temperature),
        discount: a.num("discount", d.discount),
        trace: a.num("trace", d.trace),
        use_widening: a.num("use-widening", d.use_widening),
        reward_widening: a.num("reward-widening", d.reward_widening),
        punish_narrowing: a.num("punish-narrowing", d.punish_narrowing),
        prototype: d.prototype,
        seed: a.num("seed", 1.0) as u64,
        compress_every: a.num("compress-every", 0.0).max(0.0) as u64,
        compress_precision: a
            .get("compress-precision")
            .map(Precision::parse)
            .transpose()
            .ok()
            .flatten()
            .unwrap_or(Precision::Exact),
        compress_rebuild: a.has("compress-rebuild"),
        focus: a.get("focus").and_then(|v| v.parse().ok()),
        learn_focus: a.has("learn-focus"),
        focus_rate: a.num("focus-rate", d.focus_rate),
        focus_explore: a.num("focus-explore", d.focus_explore),
        skip: a.has("skip"),
        skip_margin: a.num("skip-margin", 0.0),
        rearrange_every: a.num("rearrange-every", 0.0).max(0.0) as u64,
        mode: Mode::parse(&a.str("mode", "own"))?,
        explore: a.num("explore", d.explore),
        equal_reward: a.has("equal-reward"),
    })
}

fn fresh(a: &Args) -> Result<Machine, String> {
    let states = a.num("states", DEFAULT_STATES as f64) as usize;
    let alphabet: Vec<String> = a
        .str("alphabet", DEFAULT_ALPHABET)
        .chars()
        .map(|c| c.to_string())
        .collect();
    let accepting: Vec<usize> = match a.get("accepting") {
        Some(list) => list
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().unwrap_or(0))
            .collect(),
        None => match a.get("language") {
            Some(name) => language(name)?.accepting.to_vec(),
            None => vec![0],
        },
    };
    Machine::new(states, &alphabet, &accepting, settings(a)?)
}

fn machine(a: &Args) -> Result<Machine, String> {
    match a.get("load") {
        Some(path) => load_machine(path),
        None => fresh(a),
    }
}

fn save_if_asked(m: &Machine, a: &Args) -> Result<(), String> {
    if let Some(path) = a.get("save") {
        m.save(path)?;
        eprintln!("saved {path}");
    }
    Ok(())
}

fn main() {
    let a = Args::parse();
    let result = match a.command.as_str() {
        "help" | "--help" | "-h" => {
            print!("{HELP}");
            Ok(())
        }
        "demo" => demo(&a),
        "train" => train(&a),
        "run" => run(&a),
        "compress" => compress_cmd(&a),
        "expand" => expand_cmd(&a),
        "core-run" => core_run(&a),
        "focus" => focus_cmd(&a),
        "skip" => skip_cmd(&a),
        "rearrange" => rearrange_cmd(&a),
        "teach" => teach(&a),
        "network" => network_cmd(&a),
        "base64" => base64_cmd(&a),
        "ngram" => ngram_cmd(&a),
        "chain" => chain_cmd(&a),
        "benchmark" => benchmark_cmd(&a),
        "corpus" => corpus_cmd(&a),
        "guard" => guard_cmd(&a),
        "accuracy" => accuracy(&a),
        "table" => table(&a),
        "stats" => machine(&a).map(|m| println!("{}", m.stats().pretty(1))),
        "tick" => machine(&a).and_then(|mut m| {
            m.tick(a.num("ticks", 1.0) as i64);
            println!("clock {}  stimulation {:.3}", m.clock, m.stimulation());
            save_if_asked(&m, &a)
        }),
        "stimulate" => machine(&a).and_then(|mut m| {
            match a.get("level") {
                Some(level) => m.set_stimulation(level.parse().map_err(|_| "bad --level")?)?,
                None => {
                    m.stimulate(a.num("amount", 1.0));
                }
            }
            println!("stimulation {:.3}", m.stimulation());
            save_if_asked(&m, &a)
        }),
        "experiment" => {
            let which: Vec<String> = a.str("which", "all").split(',').map(str::to_string).collect();
            experiment::run(
                &which,
                a.get("out"),
                a.num("episodes", 4000.0) as usize,
                a.num("life", 1000.0),
            )
            .map(|recs| {
                for r in recs {
                    println!("{}\n", experiment::tables(&r));
                }
            })
        }
        "serve" => machine(&a).and_then(|m| {
            let addr = format!("{}:{}", a.str("host", "127.0.0.1"), a.num("port", 8000.0) as u16);
            let frontend = match a.get("frontend-dir") {
                Some(dir) => Some(std::path::PathBuf::from(dir)),
                None => server::default_frontend_dir(),
            };
            eprintln!(
                "latticefsm serving http://{addr}/  (machine {}; frontend {})",
                m.stats().get("shape").map(Json::dump).unwrap_or_default(),
                frontend
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none: /api/ only".to_string())
            );
            server::serve(m, &addr, frontend, None)
        }),
        other => Err(format!("unknown command {other:?}; try --help")),
    };
    if let Err(msg) = result {
        eprintln!("error: {msg}");
        process::exit(1);
    }
}

fn train(a: &Args) -> Result<(), String> {
    let name = a.str("language", "even-b");
    let lang = language(&name)?;
    let mut m = machine(a)?;
    for st in m.lattice.states.iter_mut() {
        st.accepting = lang.accepting.contains(&st.index);
    }
    let mut rng = Rng::new(a.num("seed", 1.0) as u64);
    let max_length = a.num("max-length", 6.0) as usize;
    let test = examples(&lang, a.num("tests", 300.0) as usize, &mut rng, max_length);
    let episodes = a.num("episodes", 4000.0) as usize;
    let before = m.accuracy(&test, 0.0);
    let curve = teach_language(
        &mut m,
        &lang,
        episodes,
        &mut rng,
        max_length,
        a.has("quiet"),
        1.0,
        Some(&test),
        (episodes / 8).max(1),
    );
    println!(
        "{} ({}): {} states, {} episodes{}",
        lang.name,
        lang.description,
        m.n_states(),
        episodes,
        if a.has("quiet") { ", quietly" } else { "" }
    );
    println!("  accuracy before {before:.3}");
    for (ep, acc) in &curve {
        println!("  after {ep:>6}: {acc:.3}");
    }
    print_table(&m);
    save_if_asked(&m, a)
}

fn print_core(core: &Core, m: &Machine) -> Result<(), String> {
    let s = core.summary();
    let f = fidelity(m, &core.decompress(None)?);
    let label = s
        .get("center_label")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let shape: Vec<String> = s
        .get("shape")
        .and_then(Json::as_array)
        .map(|v| v.iter().map(Json::dump).collect())
        .unwrap_or_default();
    println!(
        "the {} matrix folded into its central node ({}, {}, {}), {}:",
        shape.join(" x "),
        label.first().map(Json::dump).unwrap_or_default(),
        label.get(1).and_then(Json::as_str).unwrap_or("?"),
        label.get(2).map(Json::dump).unwrap_or_default(),
        s.str_or("precision", "")
    );
    println!(
        "  {} bytes for {} touched edges of {} (the dense matrix is {} bytes: {:.1}x smaller), plus a {}-byte record of the rest",
        s.num("bytes", 0.0),
        s.num("touched", 0.0),
        s.num("cells", 0.0),
        s.num("dense_bytes", 0.0),
        s.num("ratio", 0.0),
        s.num("record_bytes", 0.0)
    );
    println!(
        "  loss: {}; largest relative error {:.2e}; KL mean {:.2e}, max {:.2e}; greedy choices changed {} of {}",
        if f.bool_or("lossless", false) {
            "none - every field of every edge comes back bit for bit"
        } else {
            "lossy"
        },
        f.num("max_relative_error", 0.0),
        f.num("mean_kl", 0.0),
        f.num("max_kl", 0.0),
        f.num("greedy_changed", 0.0),
        f.num("rows", 0.0)
    );
    println!("  from the central node outward:");
    for row in s
        .get("shell_table")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default()
    {
        println!(
            "    shell {}: {:>4} cells, {:>4} touched, {:>6} bytes",
            row.num("shell", 0.0),
            row.num("cells", 0.0),
            row.num("touched", 0.0),
            row.num("bytes", 0.0)
        );
    }
    Ok(())
}

fn compress_cmd(a: &Args) -> Result<(), String> {
    let m = machine(a)?;
    let precision = a.get("precision").map(Precision::parse).transpose()?;
    let budget = a
        .get("budget")
        .map(|b| b.parse::<usize>().map_err(|_| "bad --budget"))
        .transpose()?;
    let core = compress(&m, precision, budget);
    print_core(&core, &m)?;
    if a.has("expansion") {
        println!("  rebuilt from the central node outward, a shell at a time:");
        for row in expansion(&m, &core)?.as_array().cloned().unwrap_or_default() {
            println!(
                "    {} shells: {:>4} cells, {:>4} touched edges restored, {} greedy choices changed",
                row.num("shells", 0.0),
                row.num("cells", 0.0),
                row.num("touched", 0.0),
                row.num("greedy_changed", 0.0)
            );
        }
    }
    if let Some(path) = a.get("core") {
        core.save(path)?;
        eprintln!("saved {path}");
    }
    Ok(())
}

fn expand_cmd(a: &Args) -> Result<(), String> {
    let core = load_core(a.get("core").ok_or("expand needs --core FILE")?)?;
    let shells = a
        .get("shells")
        .map(|k| k.parse::<usize>().map_err(|_| "bad --shells"))
        .transpose()?;
    let m = core.decompress(shells)?;
    let st = m.stats();
    println!(
        "rebuilt from the central node outward, {}: {} of {} edges written, clock {}",
        match shells {
            None => "every shell".to_string(),
            Some(1) => "1 shell".to_string(),
            Some(k) => format!("{k} shells"),
        },
        st.num("touched", 0.0),
        st.num("edges", 0.0),
        st.num("clock", 0.0)
    );
    print_table(&m);
    save_if_asked(&m, a)
}

fn focus_line(m: &Machine) -> String {
    if m.learn_focus {
        return "focus learned from the input: each run starts from the node of the central vertical vector it picks"
            .to_string();
    }
    match m.focus() {
        None => format!("no focus: runs start from the start state, {}", m.start),
        Some(f) => {
            let (s, a, t) = latticefsm::geometry::focus_node(m.lattice.shape(), Some(f));
            format!(
                "focus {f}: runs start from node ({s}, {}, {t}) of the central vertical vector, state {s}",
                m.alphabet()[a]
            )
        }
    }
}

fn focus_cmd(a: &Args) -> Result<(), String> {
    let mut m = machine(a)?;
    if a.has("none") {
        m.set_focus(None)?;
    } else if let Some(level) = a.get("level") {
        m.set_focus(Some(level.parse().map_err(|_| "bad --level")?))?;
    }
    match a.get("learn") {
        Some("on") => m.learn_focus = true,
        Some("off") => m.learn_focus = false,
        Some(other) => return Err(format!("--learn is on or off, got {other:?}")),
        None => {}
    }
    println!("{}", focus_line(&m));
    if let (true, Some(text)) = (m.learn_focus, a.get("text")) {
        let symbols = m.tokenize(text)?;
        let f = m.focus_learner.predict(&symbols);
        println!(
            "  the learner reads {text:?} as focus {f:.3}: state {}",
            m.focus_state(f)
        );
    }
    save_if_asked(&m, a)
}

fn skip_cmd(a: &Args) -> Result<(), String> {
    let mut m = machine(a)?;
    match a.positional.first().map(String::as_str) {
        Some("on") => m.skip = true,
        Some("off") => m.skip = false,
        _ => return Err("skip needs on or off".to_string()),
    }
    if let Some(margin) = a.get("margin") {
        let margin: f64 = margin.parse().map_err(|_| "bad --margin")?;
        if margin < 0.0 {
            return Err("--margin must be >= 0".to_string());
        }
        m.skip_margin = margin;
    }
    println!(
        "skips {}, margin {}; {} taken so far",
        if m.skip { "on" } else { "off" },
        m.skip_margin,
        m.skips
    );
    save_if_asked(&m, a)
}

fn rearrange_cmd(a: &Args) -> Result<(), String> {
    let mut m = machine(a)?;
    if let Some(kind) = a.get("swap") {
        let (i, j) = (
            a.positional.first().ok_or("--swap needs I J")?,
            a.positional.get(1).ok_or("--swap needs I J")?,
        );
        match kind {
            "states" => m.swap_states(i.parse().map_err(|_| "bad state")?, j.parse().map_err(|_| "bad state")?)?,
            "symbols" => {
                let (x, y) = (m.symbol(i)?, m.symbol(j)?);
                m.swap_symbols(x, y)?
            }
            other => return Err(format!("--swap states|symbols, got {other:?}")),
        }
        println!("swapped {kind} {i} and {j}");
    } else {
        let full = a.has("full");
        let made = m.rearrange(full);
        println!(
            "{} swaps toward the centre{}",
            made.len(),
            if full { " (until settled)" } else { " (one pass)" }
        );
    }
    let order: Vec<String> = m.lattice.state_ids.iter().map(|i| i.to_string()).collect();
    println!(
        "  states, top to bottom: [{}]  (start state {})",
        order.join(", "),
        m.start
    );
    println!("  symbols, in order: {}", m.alphabet().concat());
    save_if_asked(&m, a)
}

fn core_run(a: &Args) -> Result<(), String> {
    let core = load_core(a.get("core").ok_or("core-run needs --core FILE")?)?;
    let text = a.positional.first().ok_or("core-run needs a string")?;
    let from_middle = a.has("from-middle");
    let (states, accepted) = core.run(text, from_middle)?;
    let path: Vec<String> = states.iter().map(|s| s.to_string()).collect();
    println!(
        "{}{}  ({}), read straight from the code",
        if from_middle { "from the middle: " } else { "" },
        path.join(" -> "),
        if accepted { "accepted" } else { "rejected" }
    );
    Ok(())
}

fn run(a: &Args) -> Result<(), String> {
    let text = a
        .positional
        .first()
        .cloned()
        .or_else(|| a.get("text").map(str::to_string))
        .ok_or("run needs a string")?;
    let mut m = machine(a)?;
    let quiet = a.has("quiet");
    let stimulation = a.get("stimulation").and_then(|v| v.parse().ok());
    let temperature = a.get("temperature").and_then(|v| v.parse().ok());
    let times = a.num("times", 1.0) as usize;
    for _ in 0..times.max(1) {
        let symbols = m.tokenize(&text)?;
        let focus = if a.has("load") {
            a.get("focus")
                .map(|f| f.parse::<f64>().map_err(|_| "bad --focus"))
                .transpose()?
        } else {
            None
        };
        let walk = Walk {
            stimulation,
            temperature,
            quiet,
            from_middle: a.has("from-middle"),
            focus,
            skip: if a.has("skip") { Some(true) } else { None },
            mode: a.get("mode").map(Mode::parse).transpose()?,
        };
        let r = m.walk(&symbols, walk)?;
        let steps: Vec<String> = r
            .transitions
            .iter()
            .map(|t| match t.skipped {
                None => format!("{} -{}({:.2})-> {}", t.source, t.symbol, t.probability, t.target),
                Some(k) => format!(
                    "{} ={}({:.2})=> {} [skipping {}]",
                    t.source, t.symbol, t.probability, t.target, k
                ),
            })
            .collect();
        println!(
            "{}{}  => {} ({}), log p {:.3}, stimulation {:.2}",
            if quiet { "quietly: " } else { "" },
            if steps.is_empty() {
                "(empty)".to_string()
            } else {
                steps.join("  ")
            },
            r.final_state,
            if r.accepted { "accepted" } else { "rejected" },
            r.log_probability(),
            m.stimulation()
        );
        if let Some(credit) = a.get("credit") {
            let amount: f64 = credit.parse().map_err(|_| "bad --credit")?;
            let n = m.credit(amount);
            println!("  credited {n} edges with {amount:+}");
        }
    }
    save_if_asked(&m, a)
}

fn teach(a: &Args) -> Result<(), String> {
    if a.positional.len() < 3 {
        return Err("teach needs SOURCE SYMBOL TARGET".to_string());
    }
    let mut m = machine(a)?;
    let source: usize = a.positional[0].parse().map_err(|_| "bad source")?;
    let symbol = m.symbol(&a.positional[1])?;
    let target: usize = a.positional[2].parse().map_err(|_| "bad target")?;
    let amount = a.num("amount", 1.0);
    let (life, stim) = (m.life, m.stimulation());
    let e = m.teach(source, symbol, target, amount)?;
    let clock = e.last_seen + 1;
    println!("{}", e.describe(clock, life, stim).pretty(1));
    save_if_asked(&m, a)
}

fn accuracy(a: &Args) -> Result<(), String> {
    let lang = language(&a.str("language", "even-b"))?;
    let mut m = machine(a)?;
    let mut rng = Rng::new(a.num("seed", 1.0) as u64 + 1000);
    let test = examples(
        &lang,
        a.num("tests", 300.0) as usize,
        &mut rng,
        a.num("max-length", 6.0) as usize,
    );
    let temperature = a.num("temperature", 0.0);
    println!(
        "{} on {} strings: {:.3}",
        lang.name,
        test.len(),
        m.accuracy(&test, temperature)
    );
    Ok(())
}

fn table(a: &Args) -> Result<(), String> {
    let m = machine(a)?;
    print_table(&m);
    Ok(())
}

fn print_table(m: &Machine) {
    let t = m.transition_table();
    let acc = m.accepting();
    // a symbol no edge was ever traversed on is a row of ties: listed once, not thirteen times
    let used: Vec<usize> = (0..m.alphabet().len())
        .filter(|&a| m.lattice.edges.iter().any(|e| e.symbol == a && e.touched()))
        .collect();
    let unused: Vec<&str> = (0..m.alphabet().len())
        .filter(|a| !used.contains(a))
        .map(|a| m.alphabet()[a].as_str())
        .collect();
    println!("  greedy table (accepting: {:?}):", acc);
    for (s, row) in t.iter().enumerate() {
        let cells: Vec<String> = used
            .iter()
            .map(|&a| format!("{} -> {}", m.alphabet()[a], row[a]))
            .collect();
        println!(
            "    {:>2}{}: {}",
            s,
            if acc.contains(&s) { "*" } else { " " },
            cells.join("   ")
        );
    }
    if !unused.is_empty() && !used.is_empty() {
        println!(
            "    (never traversed on {}: every next state still ties)",
            unused.join(" ")
        );
    }
}

fn demo(a: &Args) -> Result<(), String> {
    let name = a.str("language", "even-b");
    let lang = language(&name)?;
    let mut m = fresh(a)?;
    for st in m.lattice.states.iter_mut() {
        st.accepting = lang.accepting.contains(&st.index);
    }
    let mut rng = Rng::new(m.seed);
    let test = examples(&lang, 300, &mut rng, 6);
    println!(
        "a machine of {} states over {:?}: {} edges, each a record of its own; life {} ticks, baseline stimulation {}",
        m.n_states(),
        m.alphabet(),
        m.lattice.len(),
        m.life,
        m.baseline
    );
    println!(
        "\nlearning {:?} ({}) by reward and punishment over random strings:",
        lang.name, lang.description
    );
    println!("  accuracy before: {:.3}", m.accuracy(&test, 0.0));
    let episodes = a.num("episodes", 4000.0) as usize;
    for (ep, acc) in teach_language(
        &mut m,
        &lang,
        episodes,
        &mut rng,
        6,
        false,
        1.0,
        Some(&test),
        (episodes / 4).max(1),
    ) {
        println!("  after {ep:>5} episodes: {acc:.3}");
    }
    print_table(&m);
    let s = m.stats();
    println!(
        "  clock {}, {} of {} edges touched, widest channel {:.2}, narrowest {:.2}",
        s.num("clock", 0.0),
        s.num("touched", 0.0),
        s.num("edges", 0.0),
        s.num("widest", 0.0),
        s.num("narrowest", 0.0)
    );

    println!("\nstimulation prefers the wide channel.  A fresh fork of three edges, widths 4, 1 and 0.25:");
    let mut fork = Machine::over(
        3,
        "a",
        &[],
        Settings {
            life: m.life,
            ..Settings::default()
        },
    )?;
    for (t, w) in [4.0, 1.0, 0.25].into_iter().enumerate() {
        fork.edge_mut(0, 0, t).set_width(w, 0);
    }
    for level in [0.0, 1.0, 2.0, 4.0] {
        let p: Vec<String> = fork
            .probabilities(0, 0, Some(level), None)
            .iter()
            .map(|p| format!("{p:.3}"))
            .collect();
        println!("  stimulation {level:<4}: [{}]", p.join(", "));
    }
    let (src, sym) = (m.start, 0);
    let widths: Vec<String> = m
        .lattice
        .row(src, sym)
        .iter()
        .map(|e| format!("{:.2}", e.width_at(m.clock, m.life)))
        .collect();
    println!(
        "  (the taught machine's row ({src}, {}) has widths [{}]: use and reward have already widened its habit)",
        m.alphabet()[sym],
        widths.join(", ")
    );
    m.stimulate(3.0);
    println!(
        "\nstimulated by +3: level {:.2}; a life of silence later:",
        m.stimulation()
    );
    m.tick(m.life as i64);
    println!(
        "  level {:.2}, the widest channel now {:.2}; accuracy still {:.3}",
        m.stimulation(),
        m.stats().num("widest", 0.0),
        m.accuracy(&test, 0.0)
    );
    m.tick(4 * m.life as i64);
    println!("  four more lives: level {:.2}, widest {:.2}; accuracy {:.3} - the verdicts never fade, the widths and traces do", m.stimulation(), m.stats().num("widest", 0.0), m.accuracy(&test, 0.0));
    let core = compress(&m, None, None);
    let rebuilt = core.decompress(None)?;
    let f = fidelity(&m, &rebuilt);
    let half = compress(&m, Some(Precision::Float16), None);
    let fh = fidelity(&m, &half.decompress(None)?);
    let (cs, ca, ct) = core.center();
    println!(
        "\nfolded into its central node ({cs}, {}, {ct}): {} bytes exact, {} - {:.1}x smaller than the dense matrix",
        m.alphabet()[ca],
        core.bytes(),
        if f.bool_or("lossless", false) {
            "lossless"
        } else {
            "lossy"
        },
        core.summary().num("ratio", 0.0)
    );
    println!(
        "  float16: {} bytes, max KL {:.1e}, {} greedy choices changed",
        half.bytes(),
        fh.num("max_kl", 0.0),
        fh.num("greedy_changed", 0.0)
    );
    let (from_middle, accepted) = core.run("abba", true)?;
    let path: Vec<String> = from_middle.iter().map(|s| s.to_string()).collect();
    println!(
        "  'abba' walked straight from the code, from the middle state outward: {} ({})",
        path.join(" -> "),
        if accepted { "accepted" } else { "rejected" }
    );
    let other = LANGUAGES
        .iter()
        .find(|l| l.name != lang.name && l.accepting == lang.accepting)
        .copied();
    if let Some(other) = other {
        println!(
            "\nretaught {:?} ({}) on the same machine, {episodes} episodes:",
            other.name, other.description
        );
        let test2 = examples(&other, 300, &mut rng, 6);
        println!("  accuracy before: {:.3}", m.accuracy(&test2, 0.0));
        for (ep, acc) in teach_language(
            &mut m,
            &other,
            episodes,
            &mut rng,
            6,
            false,
            1.0,
            Some(&test2),
            (episodes / 2).max(1),
        ) {
            println!("  after {ep:>5} episodes: {acc:.3}");
        }
        println!("  and the first language now: {:.3}", m.accuracy(&test, 0.0));
    }
    save_if_asked(&m, a)
}

fn network(a: &Args) -> Result<Network, String> {
    if let Some(path) = a.get("load") {
        return load_network(path);
    }
    let alphabet: Vec<String> = a
        .str("alphabet", DEFAULT_ALPHABET)
        .chars()
        .map(|c| c.to_string())
        .collect();
    let mut net = Network::from_architecture(
        &a.str("arch", "13>13"),
        &alphabet,
        a.num("seed", 1.0) as u64,
        Merge::parse(&a.str("merge", "concat"))?,
        Vote::parse(&a.str("vote", "majority"))?,
        &settings(a)?,
    )?;
    let accepting: Vec<usize> = match a.get("accepting") {
        Some(list) => list
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().map_err(|_| format!("bad accepting state {s:?}")))
            .collect::<Result<_, _>>()?,
        None => language(&a.str("language", "even-b"))?.accepting.to_vec(),
    };
    let accepting = if net.decoder().is_some() && a.get("accepting").is_none() {
        None
    } else {
        Some(accepting)
    };
    net.compile(
        accepting.as_deref(),
        a.num("layer-discount", LAYER_DISCOUNT),
        1.0,
        1.0,
        Credit::parse(&a.str("credit", "shared"))?,
    )?;
    Ok(net)
}

fn network_accuracy(
    net: &mut Network,
    lang: &latticefsm::languages::Language,
    max_length: usize,
) -> Result<(f64, usize), String> {
    let strings = all_strings(max_length);
    let examples: Vec<(String, bool)> = strings.iter().map(|s| (s.clone(), (lang.member)(s))).collect();
    Ok((net.evaluate(&examples)?, examples.len()))
}

/// Whether the network's inputs are files (an audio, image, diff or video encoder), so paths in a data file are
/// read relative to it.
fn reads_files(net: &Network) -> bool {
    net.encoder()
        .is_some_and(|e| !matches!(e.source.kind(), "phonetic" | "text"))
}

fn pairs(a: &Args, net: &Network) -> Result<(String, Vec<(String, String)>), String> {
    if let Some(path) = a.get("pairs") {
        return Ok((path.to_string(), media::read_labels(path, reads_files(net))?));
    }
    let name = a.str("task", "plural");
    Ok((name.clone(), phonetic::task(&name)?))
}

/// A data file's name, its taught examples and its held-out ones.
type Data = (String, Vec<(String, bool)>, Vec<(String, bool)>);

fn data(a: &Args, net: &Network) -> Result<Data, String> {
    let path = a.get("data").expect("asked for");
    let rows = media::read_labels(path, reads_files(net))?;
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, (x, y)) in rows.into_iter().enumerate() {
        let example = (x, media::verdict(&y)?);
        if i % 5 == 4 {
            test.push(example);
        } else {
            train.push(example);
        }
    }
    Ok((path.to_string(), train, test))
}

fn network_cmd(a: &Args) -> Result<(), String> {
    let action = a.positional.first().map(String::as_str).unwrap_or("summary");
    let texts: Vec<String> = a.positional.iter().skip(1).cloned().collect();
    if action == "make-data" {
        let path = media::make_dataset(
            &a.str("kind", "image"),
            &a.str("out", "data"),
            a.num("count", 40.0) as usize,
            a.num("seed", 1.0) as u64,
        )?;
        println!("wrote {path}");
        return Ok(());
    }
    let mut net = network(a)?;
    let max_length = a.num("max-length", 6.0) as usize;
    let by_pairs = a.has("task") || a.has("pairs");
    let by_data = a.has("data");
    match action {
        "new" | "summary" => println!("{}", net.summary()),
        "encode" => {
            for x in &texts {
                let symbols = net.tokenize(x)?;
                let names: Vec<&str> = symbols.iter().map(|&i| net.alphabet[i].as_str()).collect();
                println!("{x}: {} symbols", names.len());
                println!("  {}", names.join(" "));
            }
        }
        "fit" if by_data => {
            let (path, train, test) = data(a, &net)?;
            let epochs = a.num("epochs", 10.0) as usize;
            println!(
                "{path}: {} examples taught x {epochs} epochs, {} held out",
                train.len(),
                test.len()
            );
            println!(
                "  accuracy before: taught {:.6}, held out {:.6}",
                net.evaluate(&train)?,
                net.evaluate(&test)?
            );
            for _ in 0..epochs {
                let h = net.fit(&train, 1)?;
                let (taught, held) = (net.evaluate(&train)?, net.evaluate(&test)?);
                println!(
                    "  epoch {}: right while taught {:.6}, taught {taught:.6}, held out {held:.6}",
                    net.epochs, h[0]
                );
            }
            println!("{}", net.summary());
        }
        "evaluate" if by_data => {
            let (path, train, test) = data(a, &net)?;
            println!(
                "accuracy {path}: taught {:.6}, held out {:.6}",
                net.evaluate(&train)?,
                net.evaluate(&test)?
            );
        }
        "fit" if by_pairs => {
            let (name, all) = pairs(a, &net)?;
            let (train, test) = phonetic::split(&all, 4);
            let epochs = a.num("epochs", 20.0) as usize;
            println!(
                "{name}: {} pairs taught x {epochs} epochs, {} held out",
                train.len(),
                test.len()
            );
            println!(
                "  exact before: taught {:.6}, held out {:.6}",
                net.evaluate_pairs(&train)?,
                net.evaluate_pairs(&test)?
            );
            for _ in 0..epochs {
                let h = net.fit_pairs(&train, 1, Some(&test))?;
                println!(
                    "  epoch {}: sounds right while taught {:.6}, exact {:.6}, held out {:.6}",
                    net.epochs, h.sounds[0], h.exact[0], h.val_exact[0]
                );
            }
            println!("{}", net.summary());
        }
        "fit" => {
            let lang = language(&a.str("language", "even-b"))?;
            let episodes = a.num("episodes", 4000.0) as usize;
            let epochs = a.num("epochs", 1.0) as usize;
            let mut rng = Rng::new(a.num("seed", 1.0) as u64);
            let train = examples(&lang, episodes, &mut rng, max_length);
            let (acc, n) = network_accuracy(&mut net, &lang, max_length)?;
            println!(
                "{} ({}): {episodes} strings x {epochs} epochs",
                lang.name, lang.description
            );
            println!("  accuracy on every string up to {max_length} symbols ({n}): before {acc:.6}");
            for _ in 0..epochs {
                let h = net.fit(&train, 1)?;
                let (acc, _) = network_accuracy(&mut net, &lang, max_length)?;
                println!("  epoch {}: right while taught {:.6}, after {acc:.6}", net.epochs, h[0]);
            }
            println!("{}", net.summary());
        }
        "evaluate" if by_pairs => {
            let (name, all) = pairs(a, &net)?;
            let (train, test) = phonetic::split(&all, 4);
            println!(
                "exact {name}: taught {:.6}, held out {:.6}",
                net.evaluate_pairs(&train)?,
                net.evaluate_pairs(&test)?
            );
        }
        "evaluate" => {
            let lang = language(&a.str("language", "even-b"))?;
            let (acc, n) = network_accuracy(&mut net, &lang, max_length)?;
            println!(
                "accuracy {} on every string up to {max_length} symbols ({n}): {acc:.6}",
                lang.name
            );
        }
        "predict" => {
            let texts = if texts.is_empty() { vec![String::new()] } else { texts };
            for text in &texts {
                let (steps, _, accepted) = net.trace(text)?;
                let shown = if text.is_empty() { "(empty)" } else { text.as_str() };
                println!("{shown}: {}", if accepted { "accepted" } else { "rejected" });
                for st in steps {
                    let states: Vec<String> = st.states.iter().map(|s| s.to_string()).collect();
                    println!(
                        "  {} read {}, walked {}, wrote {}{}{}",
                        st.layer,
                        if st.read.is_empty() { "-" } else { &st.read },
                        if states.is_empty() {
                            "-".to_string()
                        } else {
                            states.join(" ")
                        },
                        if st.wrote.is_empty() { "-" } else { &st.wrote },
                        st.english.map(|e| format!(" = \"{e}\"")).unwrap_or_default(),
                        if st.accepted { " (accepting)" } else { "" }
                    );
                }
            }
        }
        "translate" => {
            for text in &texts {
                let sounds = net.sounds(text)?;
                println!("{text} -> {}  ({})", net.translate(text)?, sounds.join(" "));
            }
        }
        other => {
            return Err(format!(
                "network {other:?}: new, fit, evaluate, predict, translate or summary"
            ))
        }
    }
    if matches!(action, "new" | "fit" | "summary") {
        if let Some(path) = a.get("save") {
            net.save(path)?;
            eprintln!("saved {path}");
        }
    }
    Ok(())
}

fn pct(threshold: f64) -> String {
    format!("{:.0}%", threshold * 100.0)
}

/// `--until`: after the first pass (round 1), the model drilled on what it learned until `threshold` of every token
/// that can be right is (a context the data continues in more than one way can only be guessed one way), as
/// `cli._train_until` does.
fn report_until(first: (u64, u64), best_possible: usize, threshold: f64, d: &latticefsm::ngram::Drill) {
    let (right, of) = first;
    println!(
        "round 1: {right} of {of} predicted right before learning (the most possible: {best_possible}; the goal: {}, {})",
        d.need,
        pct(threshold)
    );
    for (k, r) in d.rounds.iter().enumerate() {
        println!("round {}: {r} of {} right", k + 2, d.of);
    }
    let (rnd, right, of) = (d.rounds.len() + 1, d.right, d.of);
    if right == of {
        println!("{of} of {of} reached in {rnd} rounds");
    } else if right >= best_possible {
        println!(
            "{right} of {of} reached in {rnd} rounds: the most possible - the rest ({}) come after contexts the data \
             continues in more than one way",
            of - right
        );
    } else if d.reached {
        println!(
            "{right} of {of} reached in {rnd} rounds: the goal of {} ({} of the {best_possible} possible)",
            d.need,
            pct(threshold)
        );
    } else {
        println!(
            "stopped after {rnd} rounds at {right} of {of} (the goal: {} of the {best_possible} possible)",
            d.need
        );
    }
}

/// The training sequences of `data` (every `held_out`-th of each document kept back, as training does).
fn training_sequences(data: &[String], unit: &str, lower: bool, held_out: usize) -> Result<Vec<Vec<String>>, String> {
    let mut seqs = Vec::new();
    latticefsm::ngram::train_corpus(
        data,
        unit,
        lower,
        held_out,
        &mut |t| {
            seqs.push(t.to_vec());
            0
        },
        &mut Vec::new(),
    )?;
    Ok(seqs)
}

fn print_teaching(text: &str, reply: &str, t: &latticefsm::ngram::Teaching, threshold: f64, now: String) {
    let h: Vec<String> = t.history.iter().map(|x| x.to_string()).collect();
    let goal = if t.need < t.of {
        format!("; the goal {} ({})", t.need, pct(threshold))
    } else {
        String::new()
    };
    println!(
        "taught {} after {}: {} of {} right after {} rounds{}{goal}; right before each round: {}",
        py_repr(reply),
        py_repr(text),
        t.right,
        t.of,
        t.rounds,
        if t.reached {
            ""
        } else {
            " (short of the goal: --max-rounds)"
        },
        h.join(" ")
    );
    println!("now: {now}");
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".to_string())
}

fn ngram_cmd(a: &Args) -> Result<(), String> {
    let action = a.positional.first().map(String::as_str).unwrap_or("stats");
    let texts: Vec<String> = a.positional.iter().skip(1).cloned().collect();
    if action == "ollama" {
        let client = Ollama::new(a.get("url"), a.get("model"));
        let names = client.models()?;
        let mut line = format!("Ollama at {}: {} models", client.url, names.len());
        for n in &names {
            line.push_str(&format!("\n  {n}"));
        }
        println!("{line}");
        let pulled = names
            .iter()
            .any(|n| n == &client.model || n.split(':').next() == Some(client.model.as_str()));
        let note = if pulled {
            String::new()
        } else {
            format!(" (not pulled: `ollama pull {}`)", client.model)
        };
        println!("autotrain would use {}{note}", client.model);
        return Ok(());
    }
    let mut m = match a.get("load") {
        Some(p) if std::path::Path::new(p).exists() => load_ngram(p)?,
        Some(p) if !matches!(action, "train" | "autotrain") => return Err(format!("no model at {p}")),
        _ => NGram::new(
            a.num("order", 3.0) as usize,
            &a.str("unit", "word"),
            !a.has("keep-case"),
            a.num("stimulation", 1.0),
        )?,
    };
    let data: Vec<String> = match a.get("data") {
        Some(d) => d.split(',').map(String::from).collect(),
        None => vec!["data/corpus".to_string()],
    };
    let held_out = a.num("held-out", 10.0) as usize;
    m.predict = !a.has("no-predict") || a.has("until");
    m.self_credit = a.num("self-credit", 0.0);
    match action {
        "train" => {
            let (_, test) = m.train_paths(&data, held_out)?;
            let first = (m.online.0, m.online.1);
            let (learned, lacc, lppl, _) = m.reset_online();
            let st = m.stats();
            let what = if m.unit == "base64" {
                "base64 letters".to_string()
            } else {
                format!("{}s", m.unit)
            };
            println!(
                "{}-grams of {what}: {} tokens, {} distinct, {} n-grams in {} contexts, from {} sources",
                m.order,
                m.clock,
                m.vocab.len() - 3,
                m.n_grams(),
                m.grams.len(),
                st.num("sources", 0.0)
            );
            if learned > 0 {
                println!(
                    "while training: predicted {learned} tokens before learning each, first guess right {}, perplexity {}",
                    fmt_opt(lacc, 3),
                    fmt_opt(lppl, 2)
                );
            }
            if a.has("until") {
                let seqs = training_sequences(&data, &m.unit.clone(), m.lower, held_out)?;
                let (best_possible, _) = m.ceiling(&seqs);
                let rounds = (a.num("max-rounds", 20.0) as usize).max(2) - 1;
                let threshold = a.num("threshold", 0.95);
                let d = m.drill(&seqs, rounds, rounds, threshold);
                report_until(first, best_possible, threshold, &d);
            }
            if !test.is_empty() {
                let ev = m.evaluate(&test, 2000);
                println!(
                    "held out ({} lines, {} tokens): perplexity {}, {} bits a token, next-token accuracy {}, top 5 {}",
                    test.len(),
                    ev.tokens,
                    fmt_opt(ev.perplexity, 2),
                    fmt_opt(ev.bits, 3),
                    fmt_opt(ev.accuracy, 3),
                    fmt_opt(ev.top5, 3)
                );
            }
        }
        "autotrain" => {
            let client = a.has("ollama").then(|| Ollama::new(a.get("url"), a.get("model")));
            let o = autotrain::Options {
                data: data.clone(),
                held_out,
                topics: a
                    .get("topics")
                    .map(|t| {
                        t.split(',')
                            .map(|x| x.trim().to_string())
                            .filter(|x| !x.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
                rounds: a.num("rounds", 3.0) as usize,
                lines: a.num("lines", 20.0) as usize,
                continuations: a.num("continuations", 20.0) as usize,
                judge: a.num("judge", 10.0) as usize,
                patience: a.num("patience", 2.0) as usize,
                keep: a.get("keep").map(String::from),
                save: a.get("save").map(String::from),
                seed: a.num("seed", 1.0) as u64,
                predict: !a.has("no-predict"),
                self_credit: a.num("self-credit", 0.0),
                ..autotrain::Options::default()
            };
            autotrain::autotrain(&mut m, &o, client.as_ref(), &mut |line| println!("{line}"))?;
        }
        "predict" => {
            let texts = if texts.is_empty() { vec![String::new()] } else { texts };
            for text in &texts {
                if text.is_empty() {
                    println!("at the start:");
                } else {
                    println!("after {}:", py_repr(text));
                }
                for (tok, p) in m.predict(text, a.num("k", 10.0) as usize) {
                    println!("  {tok:<16} {p:.6}");
                }
            }
        }
        "generate" => {
            let texts = if texts.is_empty() { vec![String::new()] } else { texts };
            let mut rng = Rng::new(a.num("seed", 1.0) as u64);
            if let Some(out) = a.get("out") {
                if m.unit != "base64" {
                    return Err("--out writes bytes: it needs a base64 model (--unit base64)".to_string());
                }
                let start: Vec<u8> = match a.get("prefix-from") {
                    Some(f) => {
                        let mut d = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
                        d.truncate(a.num("prefix-bytes", 64.0) as usize);
                        d
                    }
                    None => texts[0].as_bytes().to_vec(),
                };
                let bytes = m.generate_bytes(
                    &start,
                    a.num("max-tokens", 30.0) as usize,
                    a.num("temperature", 0.0),
                    &mut rng,
                )?;
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
                println!("wrote {} bytes to {out}", bytes.len());
                return Ok(());
            }
            for text in &texts {
                if m.unit == "base64" {
                    println!(
                        "{}",
                        m.complete(
                            text,
                            a.num("max-tokens", 30.0) as usize,
                            a.num("temperature", 0.0),
                            &mut rng
                        )
                    );
                    continue;
                }
                let max = a.num("max-tokens", 30.0) as usize;
                let (toks, note) = if a.has("auto-temperature") {
                    let start = m.prompt_tokens(text);
                    let (k, unit) = (m.order - 1, m.unit.clone());
                    let c = autotemp::climb(
                        &start,
                        k,
                        a.num("temperature", 0.0),
                        a.num("max-temperature", autotemp::MAX_TEMPERATURE),
                        a.num("tries", autotemp::TRIES as f64) as usize,
                        a.num("seed", 1.0) as u64,
                        |t, seed| {
                            let toks = m.continue_tokens(&start, max, t, &mut Rng::new(seed));
                            (toks.clone(), detokenize(&toks, &unit), toks)
                        },
                        |_, _| None,
                    );
                    let note = autotemp::climb_note(c.found, c.temperature, c.why, &c.tried);
                    (c.pick, Some(note))
                } else {
                    (m.generate(text, max, a.num("temperature", 0.0), &mut rng), None)
                };
                let sep = if !text.is_empty() && !toks.is_empty() && m.unit == "word" {
                    " "
                } else {
                    ""
                };
                println!("{text}{sep}{}", detokenize(&toks, &m.unit));
                if let Some(note) = note {
                    println!("  (auto-temperature: {note})");
                }
            }
        }
        "evaluate" => {
            let mut test = Vec::new();
            for (_, lines) in corpus_sequences(&data, &m.unit.clone(), m.lower)? {
                if held_out > 0 {
                    test.extend(split_lines(&lines, held_out).1);
                } else {
                    test.extend(lines);
                }
            }
            let ev = m.evaluate(&test, 2000);
            println!(
                "{} lines, {} tokens: perplexity {}, bits {}, accuracy {}, top 5 {}",
                test.len(),
                ev.tokens,
                fmt_opt(ev.perplexity, 6),
                fmt_opt(ev.bits, 6),
                fmt_opt(ev.accuracy, 6),
                fmt_opt(ev.top5, 6)
            );
        }
        "stats" => println!("{}", m.stats().pretty(1)),
        "credit" => {
            let amount = a.num("amount", 1.0);
            for text in &texts {
                println!(
                    "credited {} edges of {} with {}",
                    m.credit_text(text, amount),
                    py_repr(text),
                    py_signed(amount)
                );
            }
        }
        "teach" => {
            if texts.len() != 2 {
                return Err("teach TEXT REPLY: the text and the reply that should follow it".to_string());
            }
            let (text, reply) = (&texts[0], &texts[1]);
            let threshold = a.num("threshold", 0.95);
            let t = m.teach(
                &m.tokens(text),
                &m.tokens(reply),
                a.num("max-rounds", 100.0) as usize,
                threshold,
            );
            let now = m.complete(text, 2 * t.of + 10, 0.0, &mut Rng::new(1));
            print_teaching(text, reply, &t, threshold, now);
        }
        other => {
            return Err(format!(
                "ngram {other:?}: train, autotrain, predict, generate, evaluate, stats, credit, teach or ollama"
            ))
        }
    }
    if matches!(action, "train" | "credit" | "teach") {
        if let Some(path) = a.get("save") {
            m.save(path)?;
            eprintln!("saved {path}");
        }
    }
    Ok(())
}

fn guard_cmd(a: &Args) -> Result<(), String> {
    use latticefsm::guard::{clean_reason, load_guard, Guard};
    let action = a.positional.first().map(String::as_str).unwrap_or("stats");
    let texts: Vec<String> = a.positional.iter().skip(1).cloned().collect();
    let mut g = match a.get("load") {
        Some(p) if std::path::Path::new(p).exists() => load_guard(p)?,
        _ => Guard::new(
            &a.str("unit", "word"),
            a.num("order", 3.0) as usize,
            true,
            a.num("threshold", 1.0),
            a.num("min-coverage", 0.5),
        )?,
    };
    let reason = a.str("reason", "");
    match action {
        "blame" => {
            let severity = a.num("severity", 1.0);
            for text in &texts {
                let n = g.blame(text, &reason, severity);
                println!(
                    "blamed {n} transitions of {} ({}, {severity:.2})",
                    py_repr(text),
                    py_repr(&clean_reason(&reason))
                );
            }
        }
        "clear" => {
            for text in &texts {
                println!(
                    "cleared {} transitions of {}",
                    g.clear(text, a.num("amount", 1.0)),
                    py_repr(text)
                );
            }
        }
        "judge" => {
            for text in &texts {
                let j = g.judge(text, None, None, 5);
                println!(
                    "{}: {}: risk {:.3}, coverage {:.3}, peak {:.3}",
                    j.str_or("verdict", ""),
                    py_repr(text),
                    j.num("risk", 0.0),
                    j.num("coverage", 0.0),
                    j.num("peak", 0.0)
                );
                println!("  {}", j.str_or("why", ""));
                let reasons = j.get("reasons").and_then(Json::as_array).cloned().unwrap_or_default();
                if !reasons.is_empty() {
                    let parts: Vec<String> = reasons
                        .iter()
                        .map(|r| format!("{} {:.0}%", r.str_or("reason", ""), r.num("share", 0.0) * 100.0))
                        .collect();
                    println!("  reasons: {}", parts.join(", "));
                }
                for sp in j.get("spans").and_then(Json::as_array).cloned().unwrap_or_default() {
                    let toks: Vec<String> = sp
                        .get("tokens")
                        .and_then(Json::as_array)
                        .map(|v| v.iter().map(|t| t.as_str().unwrap_or("").to_string()).collect())
                        .unwrap_or_default();
                    let end = if sp.str_or("next", "") == "</s>" { " </s>" } else { "" };
                    println!(
                        "  '{}{end}' ({:.2}, {})",
                        toks.join(" "),
                        sp.num("evidence", 0.0),
                        sp.str_or("reason", "")
                    );
                }
            }
        }
        "forget" => {
            let n = g.forget(&reason);
            println!("forgot {}: {n} transitions", py_repr(&clean_reason(&reason)));
        }
        "reset" => {
            g.reset();
            println!("the guard is empty");
        }
        "stats" => {}
        other => return Err(format!("guard {other:?}: blame, clear, judge, forget, reset or stats")),
    }
    if action == "stats" {
        let st = g.stats();
        println!(
            "a guard over {}s, order {}, threshold {:.2}, min coverage {:.2}: {} transitions, {} blamed, {} cleared, {} \
             judged, {} rejected",
            g.unit,
            g.order,
            g.threshold,
            g.min_coverage,
            g.edges.len(),
            g.blamed,
            g.cleared,
            g.judgements,
            g.rejected
        );
        for r in st.get("reasons").and_then(Json::as_array).cloned().unwrap_or_default() {
            println!("  {}: {:.2}", r.str_or("reason", ""), r.num("blame", 0.0));
        }
    }
    if let Some(path) = a.get("save") {
        g.save(path)?;
        eprintln!("saved {path}");
    }
    Ok(())
}

fn corpus_cmd(a: &Args) -> Result<(), String> {
    use latticefsm::archive::Crc32;
    let paths = a.positional.clone();
    if paths.is_empty() {
        return Err("corpus PATH...: the files or directories to read".to_string());
    }
    let mut docs = 0;
    if a.str("unit", "text") == "base64" {
        for (label, seqs) in b64::file_sequences(&paths, 76)? {
            let text: String = seqs.concat();
            let mut crc = Crc32::default();
            crc.update(text.as_bytes());
            println!(
                "{label}: {} sequences, {} letters, crc {:08x}",
                seqs.len(),
                text.len(),
                crc.value()
            );
            docs += 1;
        }
    } else {
        let show = a.num("show", 0.0) as usize;
        for (label, lines) in latticefsm::ngram::corpus_lines(&paths)? {
            let mut crc = Crc32::default();
            crc.update(lines.join("\n").as_bytes());
            println!("{label}: {} lines, crc {:08x}", lines.len(), crc.value());
            for l in lines.iter().take(show) {
                println!("  {l}");
            }
            docs += 1;
        }
    }
    println!("{docs} documents");
    Ok(())
}

// The timed loop uses the same continuation methods as `ngram/chain generate`.
// Setup, prompt tokenization, warmup and printing are outside the measurement.
fn benchmark_cmd(a: &Args) -> Result<(), String> {
    let count = |name: &str, default: usize, minimum: usize| -> Result<usize, String> {
        match a.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= minimum)
                .ok_or_else(|| format!("--{name} must be an integer >= {minimum}")),
        }
    };
    let kind = a.str("model", "all");
    if !["all", "chain", "ngram"].contains(&kind.as_str()) {
        return Err("--model must be all, chain or ngram".into());
    }
    if kind == "all" && a.has("load") {
        return Err("--load requires --model chain or ngram".into());
    }
    let runs = count("runs", 100, 1)?;
    let warmup = count("warmup", 3, 0)?;
    let max_tokens = count("max-tokens", 128, 1)?;
    runs.checked_mul(max_tokens).ok_or("runs * max-tokens is too large")?;
    let temperature: f64 = a
        .get("temperature")
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid --temperature")?;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("--temperature must be finite and >= 0".into());
    }
    let seed: u64 = a.get("seed").unwrap_or("1").parse().map_err(|_| "invalid --seed")?;
    let text = a.str("text", "the brain is");
    let unit = a.str("unit", "word");
    let data: Vec<String> = a.str("data", "data/corpus").split(',').map(String::from).collect();
    type Generate = Box<dyn FnMut(&mut Rng) -> usize>;
    let mut reports = Vec::new();
    for model in ["chain", "ngram"] {
        if kind != "all" && kind != model {
            continue;
        }
        let (model_unit, vocab, mut generate): (String, usize, Generate) = if model == "chain" {
            let m = match a.get("load") {
                Some(path) => load_chain(path)?,
                None => {
                    let confidence = a
                        .get("confidence")
                        .unwrap_or("0.9")
                        .parse()
                        .map_err(|_| "invalid --confidence")?;
                    let mut m = Chain::new(count("nodes", 8, 2)?, &unit, true, confidence)?;
                    m.predict = false;
                    m.train_paths(&data, 10)?;
                    m
                }
            };
            // Both variants keep their model and pre-tokenized prompt inside the closure.
            let prompt = m.prompt_tokens(&text);
            (
                m.unit.clone(),
                m.vocab.len(),
                Box::new(move |rng| {
                    std::hint::black_box(m.continue_tokens(&prompt, max_tokens, temperature, rng))
                        .0
                        .len()
                }),
            )
        } else {
            let mut m = match a.get("load") {
                Some(path) => load_ngram(path)?,
                None => {
                    let mut m = NGram::new(count("order", 3, 1)?, &unit, true, 1.0)?;
                    m.predict = false;
                    m.train_paths(&data, 10)?;
                    m
                }
            };
            let prompt = m.prompt_tokens(&text);
            (
                m.unit.clone(),
                m.vocab.len(),
                Box::new(move |rng| {
                    std::hint::black_box(m.continue_tokens(&prompt, max_tokens, temperature, rng)).len()
                }),
            )
        };
        if vocab <= 3 {
            return Err(format!(
                "{model}: no training tokens; supply a nonempty corpus or model"
            ));
        }
        let mut rng = Rng::new(seed);
        for _ in 0..warmup {
            generate(&mut rng);
        }
        // Warmup must not consume the measured run's random stream.
        let mut rng = Rng::new(seed);
        let start = std::time::Instant::now();
        let tokens: usize = (0..runs).map(|_| generate(&mut rng)).sum();
        let seconds = start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
        let rate = tokens as f64 / seconds;
        if !a.has("json") {
            println!("{model} ({model_unit}, {vocab} vocabulary entries): {tokens} generated tokens in {seconds:.6} s = {rate:.2} tokens/s ({runs} runs, {warmup} warmup)");
        }
        reports.push(
            Json::object()
                .with("model", model.into())
                .with("unit", model_unit.into())
                .with("vocabulary", vocab.into())
                .with("runs", runs.into())
                .with("warmup", warmup.into())
                .with("max_tokens", max_tokens.into())
                .with("temperature", temperature.into())
                .with("seed", seed.to_string().into())
                .with("prompt", text.clone().into())
                .with("tokens", tokens.into())
                .with("seconds", seconds.into())
                .with("tokens_per_second", rate.into()),
        );
    }
    if a.has("json") {
        println!("{}", Json::Array(reports).pretty(2));
    }
    Ok(())
}

fn chain_test(m: &Chain, data: &[String], held_out: usize) -> Result<Vec<Vec<String>>, String> {
    let mut test = Vec::new();
    for (_, seqs) in corpus_sequences(data, &m.unit, m.lower)? {
        if held_out > 0 {
            test.extend(split_lines(&seqs, held_out).1);
        } else {
            test.extend(seqs);
        }
    }
    Ok(test)
}

fn chain_cmd(a: &Args) -> Result<(), String> {
    let action = a.positional.first().map(String::as_str).unwrap_or("stats");
    let texts: Vec<String> = a.positional.iter().skip(1).cloned().collect();
    let data: Vec<String> = match a.get("data") {
        Some(d) => d.split(',').filter(|p| !p.is_empty()).map(String::from).collect(),
        None => vec!["data/corpus".to_string()],
    };
    let held_out = a.num("held-out", 10.0) as usize;
    let mut m = match a.get("load") {
        Some(p) if std::path::Path::new(p).exists() => load_chain(p)?,
        Some(p) if action != "train" => return Err(format!("no chain at {p}")),
        _ => {
            let mut m = Chain::new(
                a.num("nodes", 8.0) as usize,
                &a.str("unit", "char"),
                !a.has("keep-case"),
                a.num("confidence", 0.9),
            )?;
            if action != "train" {
                // nothing to load: learn the data first (every --held-out-th line kept back), quietly
                m.predict = false;
                m.train_paths(&data, held_out)?;
            }
            m
        }
    };
    if a.has("set-confidence") {
        m.confidence = a.num("set-confidence", m.confidence);
    }
    m.predict = !a.has("no-predict") || a.has("until");
    m.self_credit = a.num("self-credit", 0.0);
    let what = if m.unit == "base64" {
        "base64 letters".to_string()
    } else {
        format!("{}s", m.unit)
    };
    let st = |m: &Chain, key: &str| m.stats().num(key, 0.0) as u64;
    match action {
        "train" => {
            let (_, test) = m.train_paths(&data, held_out)?;
            let first = (m.online.0, m.online.1);
            let learned = m.reset_online();
            println!(
                "a chain of {} nodes ({} input nodes and the output node) over {what}: {} inputs, {} distinct, {} node \
                 contexts, from {} sources",
                m.nodes,
                m.depth(),
                m.clock,
                m.vocab.len() - 3,
                st(&m, "contexts"),
                m.sources.len()
            );
            if learned.predicted > 0 {
                println!(
                    "while training: predicted {} inputs before learning each, first guess right {}, perplexity {}, {} \
                     input nodes traversed",
                    learned.predicted,
                    fmt_opt(learned.accuracy, 3),
                    fmt_opt(learned.perplexity, 2),
                    fmt_opt(learned.depth, 2)
                );
            }
            if a.has("until") {
                let seqs = training_sequences(&data, &m.unit.clone(), m.lower, held_out)?;
                let (best_possible, _) = m.ceiling(&seqs);
                let rounds = (a.num("max-rounds", 20.0) as usize).max(2) - 1;
                let threshold = a.num("threshold", 0.95);
                let d = m.drill(&seqs, rounds, rounds, threshold);
                report_until(first, best_possible, threshold, &d);
            }
            if !test.is_empty() {
                let ev = m.evaluate(&test, None);
                println!(
                    "held out ({} lines, {} inputs) at confidence {:.2}: perplexity {}, accuracy {}, {} of {} input \
                     nodes traversed",
                    test.len(),
                    ev.tokens,
                    m.confidence,
                    fmt_opt(ev.perplexity, 2),
                    fmt_opt(ev.accuracy, 3),
                    fmt_opt(ev.depth, 2),
                    m.depth()
                );
            }
        }
        "predict" => {
            let k = a.num("k", 10.0) as usize;
            for text in if texts.is_empty() {
                vec![String::new()]
            } else {
                texts.clone()
            } {
                let toks = m.prompt_tokens(&text);
                let e = m.explain(&toks, k, None);
                if text.is_empty() {
                    println!("at the start:");
                } else {
                    println!("after {}:", py_repr(&text));
                }
                let inputs: Vec<String> = e
                    .get("inputs")
                    .and_then(Json::as_array)
                    .map(|v| v.iter().map(|t| t.as_str().unwrap_or("\u{b7}").to_string()).collect())
                    .unwrap_or_default();
                println!("  input nodes: {}", inputs.join(" "));
                let walk = e.get("walk").and_then(Json::as_array).cloned().unwrap_or_default();
                let steps: Vec<String> = walk
                    .iter()
                    .map(|s| {
                        let input = s.get("input").and_then(Json::as_str).map(py_repr);
                        format!(
                            "node {} {} {:.3}",
                            s.num("node", 0.0),
                            input.unwrap_or_else(|| "(output)".to_string()),
                            s.num("confidence", 0.0)
                        )
                    })
                    .collect();
                let gathered = walk.last().map(|s| s.num("gathered", 0.0)).unwrap_or(0.0);
                let shown = if steps.is_empty() {
                    "(nothing learned)".to_string()
                } else {
                    steps.join(" > ")
                };
                println!(
                    "  walk: {shown}; {} of {} input nodes, confidence gathered {gathered:.3}",
                    e.num("depth", 0.0),
                    m.depth()
                );
                for n in e.get("next").and_then(Json::as_array).cloned().unwrap_or_default() {
                    println!("  {:<16} {:.6}", n.str_or("token", ""), n.num("p", 0.0));
                }
            }
        }
        "generate" => {
            let mut rng = Rng::new(a.num("seed", 1.0) as u64);
            for text in if texts.is_empty() {
                vec![String::new()]
            } else {
                texts.clone()
            } {
                let start = m.prompt_tokens(&text);
                let max = a.num("max-tokens", 40.0) as usize;
                let (mut note, k) = (None, m.nodes - 1);
                let (toks, depths) = if a.has("auto-temperature") {
                    let c = autotemp::climb(
                        &start,
                        k,
                        a.num("temperature", 0.0),
                        a.num("max-temperature", autotemp::MAX_TEMPERATURE),
                        a.num("tries", autotemp::TRIES as f64) as usize,
                        a.num("seed", 1.0) as u64,
                        |t, seed| {
                            let (toks, depths) = m.continue_tokens(&start, max, t, &mut Rng::new(seed));
                            (toks.clone(), detokenize(&toks, &m.unit), (toks, depths))
                        },
                        |_, _| None,
                    );
                    note = Some(autotemp::climb_note(c.found, c.temperature, c.why, &c.tried));
                    c.pick
                } else {
                    m.continue_tokens(&start, max, a.num("temperature", 0.0), &mut rng)
                };
                let line = if m.unit == "base64" {
                    let mut all = start.clone();
                    all.extend(toks);
                    detokenize(&all, &m.unit)
                } else {
                    let sep = if !text.is_empty() && !toks.is_empty() && m.unit == "word" {
                        " "
                    } else {
                        ""
                    };
                    format!("{text}{sep}{}", detokenize(&toks, &m.unit))
                };
                let mean = if depths.is_empty() {
                    0.0
                } else {
                    depths.iter().sum::<usize>() as f64 / depths.len() as f64
                };
                let ds: Vec<String> = depths.iter().map(|d| d.to_string()).collect();
                println!(
                    "{line}\n  ({} walks, {mean:.2} input nodes traversed on average: {})",
                    depths.len(),
                    ds.join(" ")
                );
                if let Some(note) = note {
                    println!("  (auto-temperature: {note})");
                }
            }
        }
        "evaluate" => {
            let test = chain_test(&m, &data, held_out)?;
            let ev = m.evaluate(&test, None);
            let ds: Vec<String> = ev.depths.iter().map(|d| d.to_string()).collect();
            println!(
                "{} lines, {} inputs at confidence {:.2}: perplexity {}, bits {}, accuracy {}, input nodes traversed {}; \
                 walks by depth: {}",
                test.len(),
                ev.tokens,
                m.confidence,
                fmt_opt(ev.perplexity, 6),
                fmt_opt(ev.bits, 6),
                fmt_opt(ev.accuracy, 6),
                fmt_opt(ev.depth, 6),
                ds.join(" ")
            );
        }
        "sweep" => {
            let test = chain_test(&m, &data, held_out)?;
            let thresholds: Vec<f64> = a
                .str("thresholds", "0,0.3,0.5,0.7,0.8,0.9,0.95,1")
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse::<f64>().map_err(|_| format!("not a number: {t}")))
                .collect::<Result<_, _>>()?;
            println!(
                "{} held-out lines; the less confidence a walk needs, the fewer of the {} input nodes it traverses:",
                test.len(),
                m.depth()
            );
            for (t, ev) in m.sweep(&test, &thresholds) {
                println!(
                    "  confidence {t:.2}: accuracy {}, perplexity {}, {} input nodes traversed",
                    fmt_opt(ev.accuracy, 4),
                    fmt_opt(ev.perplexity, 3),
                    fmt_opt(ev.depth, 3)
                );
            }
        }
        "stats" => {
            let s = m.stats();
            println!(
                "a chain of {} nodes over {what}, confidence {:.2}: {} inputs, {} distinct, {} node contexts, {} credited",
                m.nodes,
                m.confidence,
                m.clock,
                m.vocab.len() - 3,
                s.num("contexts", 0.0),
                s.num("credited", 0.0)
            );
            for d in s.get("depths").and_then(Json::as_array).cloned().unwrap_or_default() {
                println!(
                    "  depth {}: {} contexts, tested {} times, right {}",
                    d.num("depth", 0.0),
                    d.num("contexts", 0.0),
                    d.num("tested", 0.0),
                    fmt_opt(d.get("accuracy").and_then(Json::as_f64), 3)
                );
            }
        }
        "credit" => {
            let amount = a.num("amount", 1.0);
            for text in &texts {
                println!(
                    "credited {} nodes on the walks of {} with {}",
                    m.credit_text(text, amount),
                    py_repr(text),
                    py_signed(amount)
                );
            }
        }
        "teach" => {
            if texts.len() != 2 {
                return Err("teach TEXT REPLY: the text and the reply that should follow it".to_string());
            }
            let (text, reply) = (&texts[0], &texts[1]);
            let threshold = a.num("threshold", 0.95);
            let t = m.teach(
                &m.tokens(text),
                &m.tokens(reply),
                a.num("max-rounds", 100.0) as usize,
                threshold,
            );
            let now = m.complete(text, 2 * t.of + 10, 0.0, &mut Rng::new(1));
            print_teaching(text, reply, &t, threshold, now);
        }
        other => return Err(format!("unknown chain action {other:?}")),
    }
    if let Some(path) = a.get("save") {
        if matches!(action, "train" | "credit" | "teach") {
            m.save(path)?;
            eprintln!("saved {path}");
        }
    }
    Ok(())
}

/// A string the way Python's `repr` writes a simple one.
fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// A float the way Python's `{:+}` writes it.
fn py_signed(x: f64) -> String {
    let body = if x == x.trunc() && x.abs() < 1e16 {
        format!("{:.1}", x.abs())
    } else {
        format!("{}", x.abs())
    };
    format!("{}{body}", if x < 0.0 { "-" } else { "+" })
}

fn base64_cmd(a: &Args) -> Result<(), String> {
    use std::io::{Read as _, Write as _};
    let action = a.positional.first().map(String::as_str).unwrap_or("encode");
    let inputs: Vec<String> = a.positional.iter().skip(1).cloned().collect();
    let inputs = if inputs.is_empty() {
        vec!["-".to_string()]
    } else {
        inputs
    };
    for item in &inputs {
        let data = if item == "-" {
            let mut d = Vec::new();
            std::io::stdin().read_to_end(&mut d).map_err(|e| e.to_string())?;
            d
        } else if std::path::Path::new(item).is_file() {
            std::fs::read(item).map_err(|e| format!("{item}: {e}"))?
        } else {
            item.as_bytes().to_vec()
        };
        match action {
            "encode" => {
                let text = b64::encode(&data);
                match a.get("out") {
                    Some(out) => {
                        std::fs::write(out, format!("{text}\n")).map_err(|e| format!("{out}: {e}"))?;
                        eprintln!("wrote {} letters ({} bytes encoded) to {out}", text.len(), data.len());
                    }
                    None => println!("{text}"),
                }
            }
            "decode" => {
                let raw = b64::decode(&String::from_utf8_lossy(&data));
                match a.get("out") {
                    Some(out) => {
                        std::fs::write(out, &raw).map_err(|e| format!("{out}: {e}"))?;
                        eprintln!("wrote {} bytes to {out}", raw.len());
                    }
                    None => {
                        let mut text = String::from_utf8_lossy(&raw).into_owned();
                        if !raw.ends_with(b"\n") {
                            text.push('\n');
                        }
                        std::io::stdout()
                            .write_all(text.as_bytes())
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
            other => return Err(format!("base64 {other:?}: encode or decode")),
        }
    }
    Ok(())
}
