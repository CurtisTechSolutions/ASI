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
//!     experiment  the learning, stimulation and adaptation experiments
//!     serve       the HTTP API and the page on --port
//!
//! Every command but `train`, `demo`, `experiment` and `serve` takes `--load` (a machine file written by
//! `train` or by any command's `--save`); `serve` and `train` take it too, and start fresh without it.

use std::env;
use std::process;

use latticefsm::experiment::{self, teach_language};
use latticefsm::json::Json;
use latticefsm::languages::{examples, language, LANGUAGES};
use latticefsm::machine::{load_machine, Machine, Settings};
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
  run TEXT    read TEXT from the start state and traverse it (--quiet: ask only); --credit R rewards (R<0 punishes)
  teach S SYM T [--amount R]   traverse one edge deliberately and credit it
  accuracy    the share of --tests random strings of --language classified right (nothing moves)
  table       the greedy transition table (nothing moves)
  stats       size, memory, clock and stimulation
  tick        let --ticks pass
  stimulate   raise the stimulation by --amount (or set --level)
  experiment  --which learning|stimulation|adaptation|all, --out DIR
  serve       the HTTP API and the page on --port (default 8000)

machine options (train, demo, serve without --load)
  --states N (4)  --alphabet ab  --accepting 0,2  --start 0  --life 1000  --baseline 1  --calm LIFE
  --temperature 1  --discount 0.8  --seed 1  --use-widening 0.01  --reward-widening 0.2  --punish-narrowing 0.2
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
    }
}

fn fresh(a: &Args) -> Result<Machine, String> {
    let states = a.num("states", 4.0) as usize;
    let alphabet: Vec<String> = a.str("alphabet", "ab").chars().map(|c| c.to_string()).collect();
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
            eprintln!(
                "latticefsm serving http://{addr}/  (machine {:?})",
                m.stats().get("shape").map(Json::dump)
            );
            server::serve(m, &addr, None)
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
    let episodes = a.num("episodes", 2000.0) as usize;
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
        let r = m.run_text(&text, stimulation, temperature, quiet)?;
        let steps: Vec<String> = r
            .transitions
            .iter()
            .map(|t| format!("{} -{}({:.2})-> {}", t.source, t.symbol, t.probability, t.target))
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
    println!("  greedy table (accepting: {:?}):", acc);
    for (s, row) in t.iter().enumerate() {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(a, t)| format!("{} -> {}", m.alphabet()[a], t))
            .collect();
        println!(
            "    {}{}: {}",
            s,
            if acc.contains(&s) { "*" } else { " " },
            cells.join("   ")
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
    let episodes = a.num("episodes", 2000.0) as usize;
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
