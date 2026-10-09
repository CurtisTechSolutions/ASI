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
//!     experiment  the learning, stimulation, adaptation and compression experiments
//!     serve       the HTTP API and the React frontend (frontend/dist) on --port
//!
//! Every command but `train`, `demo`, `experiment` and `serve` takes `--load` (a machine file written by
//! `train` or by any command's `--save`); `serve` and `train` take it too, and start fresh without it.

use std::env;
use std::process;

use latticefsm::compress::{compress, expansion, fidelity, load_core, Core, Precision};
use latticefsm::experiment::{self, teach_language};
use latticefsm::json::Json;
use latticefsm::languages::{examples, language, LANGUAGES};
use latticefsm::machine::{load_machine, Machine, Settings, Walk, DEFAULT_ALPHABET, DEFAULT_STATES};
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
  experiment  --which learning|stimulation|adaptation|compression|all, --out DIR
  serve       the HTTP API and the frontend on --port (default 8000); --frontend-dir DIR (default: frontend/dist
              under the working directory, or ../frontend/dist)

machine options (train, demo, serve without --load)
  --states N (13)  --alphabet abcdefghijklm  (a 13 x 13 x 13 matrix)  --accepting 0,2  --start 0  --life 1000
  --baseline 1  --calm LIFE
  --temperature 1  --discount 0.8  --seed 1  --use-widening 0.01  --reward-widening 0.2  --punish-narrowing 0.2
  --compress-every N (0: never)  --compress-precision exact|float32|float16  --compress-rebuild
  --focus F (0..1)  --learn-focus  --skip  --skip-margin M  --rearrange-every N (0: never)
common options
  --load FILE  --save FILE (.json or .json.gz)  --stimulation X (run)  --temperature T (run, accuracy)
  --language even-b|contains-aa|ends-ab|mod3-a  --episodes N  --max-length 6  --tests 300
";

fn settings(a: &Args) -> Settings {
    let d = Settings::default();
    Settings {
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
    }
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
    Machine::new(states, &alphabet, &accepting, settings(a))
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
