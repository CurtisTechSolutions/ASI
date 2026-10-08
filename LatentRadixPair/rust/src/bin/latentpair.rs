//! The latentpair command: train the tokenizer, prime, read, judge, predict, serve the frontend.

use latentpair::model::Model;
use latentpair::pair::{self, Settings};
use latentpair::serve;
use latentpair::tokenizer::{self, Config, Options, Stat, Tokenizer};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

const USAGE: &str = "latentpair: a trained context tokenizer under a primed radix pair.

  latentpair tokenizer train --out TOK [options] FILES...   train the tokenizer on files
  latentpair tokenizer info --tokenizer TOK                 describe a tokenizer
  latentpair tokenizer encode --tokenizer TOK --text T      the code of a context (--all: every position)
  latentpair tokenizer decode --tokenizer TOK --text T      what its code decodes to, one symbol at a time
  latentpair tokenizer classes --tokenizer TOK FILES...     what each first symbol stands for in files
  latentpair prime --tokenizer TOK --out MODEL [settings]   prime the two trees for a tokenizer
  latentpair train --model MODEL [--out MODEL] FILES...     read files into the count tree
  latentpair reward --model MODEL --text T [--prefix P]     credit an outcome (also read)
  latentpair punish --model MODEL --text T [--prefix P]     charge an outcome
  latentpair judge --model MODEL --good T --bad T           two-sided judgement
  latentpair predict --model MODEL --prefix P [--length N]  continue a prefix
  latentpair fold --model MODEL --prefix P                  the next-byte distribution and the code's levels
  latentpair score --model MODEL --text T                   bits per byte and the reward readings of a text
  latentpair info --model MODEL                             describe a model
  latentpair demo [--steps N] FILES...                      the whole loop on some files
  latentpair bench --tokenizer TOK FILES...                 held-out bits per byte against raw byte contexts
  latentpair serve --model MODEL [--addr 127.0.0.1:8080]    the web frontend and JSON API (ModelKit's API too);
                   [--web DIR] [--checkpoint-dir DIR]        --web serves a directory of static files instead, such as
                                                             ModelKit's frontend/dist; --checkpoint-dir enables its checkpoints

Flags take --name value; --text - reads standard input. Judgements take --strength and --outcomes
(the potential outcomes of the space the verdict comes from; a verdict is worth strength/outcomes).
Settings for prime and bench: --alpha --floor --smoothing --share-scale --reward-scale --merit-scale
--penalty-scale --strength --outcomes --rungs --backoff --cell-ceiling. Tokenizer shape for train and
demo: --window --embed --enc-hidden --dec-hidden --out-embed --levels --noise --recency --predict
--start-share; training: --steps --batch --lr --seed --eval-every.
";

/// Flags (--name value, or a bare --name for a boolean) and positional arguments.
struct Args {
    flags: HashMap<String, String>,
    positional: Vec<String>,
}

const BOOLEANS: [&str; 5] = ["all", "no-read", "to-end", "json", "help"];

impl Args {
    fn parse(argv: &[String]) -> Args {
        let mut a = Args { flags: HashMap::new(), positional: Vec::new() };
        let mut i = 0;
        while i < argv.len() {
            let arg = &argv[i];
            if let Some(name) = arg.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    a.flags.insert(k.to_string(), v.to_string());
                } else if BOOLEANS.contains(&name) {
                    a.flags.insert(name.to_string(), "true".into());
                } else if i + 1 < argv.len() {
                    a.flags.insert(name.to_string(), argv[i + 1].clone());
                    i += 1;
                } else {
                    a.flags.insert(name.to_string(), "true".into());
                }
            } else {
                a.positional.push(arg.clone());
            }
            i += 1;
        }
        a
    }
    fn get(&self, name: &str, default: &str) -> String {
        self.flags.get(name).cloned().unwrap_or_else(|| default.to_string())
    }
    fn f64(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.flags.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{name} wants a number, got {v:?}")),
        }
    }
    fn usize(&self, name: &str, default: usize) -> Result<usize, String> {
        match self.flags.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{name} wants a whole number, got {v:?}")),
        }
    }
    fn has(&self, name: &str) -> bool {
        self.flags.get(name).map_or(false, |v| v == "true")
    }
    fn text(&self, name: &str) -> Result<String, String> {
        let v = self.get(name, "");
        if v == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).map_err(|e| e.to_string())?;
            return Ok(s);
        }
        Ok(v)
    }
}

fn read_files(paths: &[String]) -> Result<Vec<Vec<u8>>, String> {
    if paths.is_empty() {
        return Err("no files given".into());
    }
    paths.iter().map(|p| std::fs::read(p).map_err(|e| format!("{p}: {e}"))).collect()
}

fn settings_from(a: &Args) -> Result<Settings, String> {
    let d = Settings::default();
    Ok(Settings {
        alpha: a.f64("alpha", d.alpha)?,
        floor: a.f64("floor", d.floor)?,
        smoothing: a.f64("smoothing", d.smoothing)?,
        share_scale: a.f64("share-scale", d.share_scale)?,
        reward_scale: a.f64("reward-scale", d.reward_scale)?,
        merit_scale: a.f64("merit-scale", d.merit_scale)?,
        penalty_scale: a.f64("penalty-scale", d.penalty_scale)?,
        strength: a.f64("strength", d.strength)?,
        outcomes: a.usize("outcomes", d.outcomes)?,
        rungs: a.get("rungs", &d.rungs),
        backoff: a.get("backoff", &d.backoff),
        cell_ceiling: a.usize("cell-ceiling", d.cell_ceiling)?,
    })
}

fn config_from(a: &Args) -> Result<Config, String> {
    let d = Config::default();
    let cfg = Config {
        window: a.usize("window", d.window)?,
        embed: a.usize("embed", d.embed)?,
        enc_hidden: a.usize("enc-hidden", d.enc_hidden)?,
        dec_hidden: a.usize("dec-hidden", d.dec_hidden)?,
        out_embed: a.usize("out-embed", d.out_embed)?,
        levels: tokenizer::parse_levels(&a.get("levels", &tokenizer::levels_string(&d.levels)))?,
        noise: a.f64("noise", d.noise)?,
        recency: a.f64("recency", d.recency)?,
        predict: a.f64("predict", d.predict)?,
        start_share: a.f64("start-share", d.start_share)?,
    };
    cfg.validate()?;
    Ok(cfg)
}

fn options_from(a: &Args) -> Result<Options, String> {
    let d = Options::default();
    Ok(Options {
        steps: a.usize("steps", d.steps)?,
        batch: a.usize("batch", d.batch)?,
        lr: a.f64("lr", d.lr)?,
        seed: a.usize("seed", d.seed as usize)? as u64,
        eval_every: a.usize("eval-every", d.eval_every)?,
        eval_windows: a.usize("eval-windows", d.eval_windows)?,
        ..d
    })
}

fn log_stat(s: &Stat) {
    println!(
        "step {:6}  recon {:.3} bits  next {}  train {:.3}  exact {}  tail {}  lr {:.1e}  {:.0}s",
        s.step, s.loss_bits, fmt_list(&s.next_bits), s.train_bits, fmt_list(&s.accuracy), fmt_list(&s.tail), s.lr, s.seconds
    );
}

fn fmt_list(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join("/")
}

fn fmt_code(code: &[i32]) -> String {
    format!("[{}]", code.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "))
}

fn print_json(v: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "help" || argv[0] == "--help" || argv[0] == "-h" {
        print!("{USAGE}");
        std::process::exit(if argv.is_empty() { 2 } else { 0 });
    }
    let a = Args::parse(&argv[1..]);
    let result = match argv[0].as_str() {
        "tokenizer" => tokenizer_cmd(&a),
        "prime" => prime_cmd(&a),
        "train" => train_cmd(&a),
        "reward" | "punish" => judge_one_cmd(&argv[0], &a),
        "judge" => judge_cmd(&a),
        "predict" => predict_cmd(&a),
        "fold" => fold_cmd(&a),
        "score" => score_cmd(&a),
        "info" => info_cmd(&a),
        "demo" => demo_cmd(&a),
        "bench" => bench_cmd(&a),
        "serve" => serve_cmd(&a),
        other => Err(format!("unknown command {other:?}\n{USAGE}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn tokenizer_cmd(a: &Args) -> Result<(), String> {
    let sub = a.positional.first().cloned().unwrap_or_default();
    let files = a.positional.iter().skip(1).cloned().collect::<Vec<_>>();
    match sub.as_str() {
        "train" => {
            let cfg = config_from(a)?;
            let o = options_from(a)?;
            let out = a.get("out", "tokenizer.json.gz");
            let texts = read_files(&files)?;
            let mut tok = Tokenizer::new(cfg.clone(), o.seed as i64)?;
            let total: usize = texts.iter().map(|t| t.len()).sum();
            println!(
                "tokenizer: window {}, code {} ({:.1} bits), {} parameters; corpus {} files, {} bytes",
                cfg.window, tokenizer::levels_string(&cfg.levels), cfg.bits(), tok.param_count(), texts.len(), total
            );
            tok.train(&texts, &o, log_stat)?;
            tok.save(&out)?;
            println!("saved {out}");
            Ok(())
        }
        "info" => {
            let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
            print_json(&tok.info());
            Ok(())
        }
        "encode" => {
            let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
            let t = a.text("text")?;
            if !a.has("all") {
                println!("{}", fmt_code(&tok.encode(t.as_bytes())));
                return Ok(());
            }
            let codes = tok.encode_all(t.as_bytes());
            for i in 0..codes.len() {
                let next = if i < t.len() { pair::symbol(t.as_bytes()[i] as usize) } else { "</s>".into() };
                println!("{i:4}  {:<12} next {next}", fmt_code(codes.at(i)));
            }
            Ok(())
        }
        "decode" => {
            let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
            let t = a.text("text")?;
            show_decode(&tok, t.as_bytes())
        }
        "classes" => {
            let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
            let texts = read_files(&files)?;
            let v = serve::classes_json(&tok, &texts);
            println!("first symbol over {} positions (radix {}):", v["total"], v["radix"]);
            for c in v["classes"].as_array().unwrap() {
                println!("  {:3}  {:6.2}%  prototype {:?}  example {:?}", c["symbol"], 100.0 * c["share"].as_f64().unwrap_or(0.0), c["prototype"].as_str().unwrap_or(""), c["example"].as_str().unwrap_or(""));
            }
            Ok(())
        }
        _ => Err("tokenizer needs a subcommand: train, info, encode, decode, classes".into()),
    }
}

fn show_decode(tok: &Tokenizer, ctx: &[u8]) -> Result<(), String> {
    let code = tok.encode(ctx);
    let shown = &ctx[ctx.len().saturating_sub(tok.config.window)..];
    println!("context {:?} -> code {}", String::from_utf8_lossy(shown), fmt_code(&code));
    for k in 1..=tok.depth() {
        println!("  {k} symbol(s) {:<12} -> {:?}", fmt_code(&code[..k]), tok.decode(&code, k)?);
    }
    Ok(())
}

fn prime_cmd(a: &Args) -> Result<(), String> {
    let s = settings_from(a)?;
    let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
    let out = a.get("out", "model.json.gz");
    let m = Model::prime(tok, s, a.usize("seed", 1)? as i64)?;
    m.save(&out)?;
    println!(
        "primed {} context nodes x {} outcomes = {} cells ({:.1} MB) for codes {:?}; saved {out}",
        m.pair.addr.n, pair::OUT, m.pair.addr.cells(), m.pair.memory_bytes() as f64 / 1e6, m.pair.addr.radices
    );
    Ok(())
}

fn load_model(a: &Args) -> Result<(Model, String), String> {
    let path = a.get("model", "model.json.gz");
    let out = a.get("out", &path);
    Ok((Model::load(&path)?, out))
}

fn train_cmd(a: &Args) -> Result<(), String> {
    let (mut m, out) = load_model(a)?;
    let text = a.text("text")?;
    let texts = if !text.is_empty() { vec![text.into_bytes()] } else { read_files(&a.positional)? };
    let started = Instant::now();
    let r = m.train_bytes(&texts);
    println!("read {} texts, {} units in {:.2}s; saving {out}", r["texts"], r["units"], started.elapsed().as_secs_f64());
    m.save(&out)
}

fn judge_one_cmd(kind: &str, a: &Args) -> Result<(), String> {
    let (mut m, out) = load_model(a)?;
    let t = a.text("text")?;
    let prefix = a.get("prefix", "");
    let strength = a.f64("strength", 0.0)?;
    let outcomes = a.usize("outcomes", 0)?;
    let r = if kind == "reward" {
        m.reward(&[t], strength, None, !a.has("no-read"), &prefix, outcomes)?
    } else {
        m.punish(&[t], strength, None, &prefix, outcomes)?
    };
    print_json(&r);
    m.save(&out)
}

fn judge_cmd(a: &Args) -> Result<(), String> {
    let (mut m, out) = load_model(a)?;
    let r = m.two_nrl(&[a.get("bad", "")], &[a.get("good", "")], a.f64("strength", 0.0)?, &a.get("prefix", ""), a.usize("outcomes", 0)?)?;
    print_json(&r);
    m.save(&out)
}

fn predict_cmd(a: &Args) -> Result<(), String> {
    let path = a.get("model", "model.json.gz");
    let mut m = Model::load(&path)?;
    let prefix = a.text("prefix")?;
    let mode = a.get("mode", "greedy");
    let r = m.predict(&prefix, a.usize("length", 32)?, &mode, &a.get("traversal", "reward"), a.f64("temperature", 1.0)?, a.has("to-end"), &a.get("backoff", ""), 4096)?;
    if a.has("json") {
        print_json(&serde_json::to_value(&r).unwrap());
    } else {
        println!("{prefix}|{}", r.text);
        println!(
            "({} units, cost {:.3} nats, {:.2} bits/unit, reached end: {}, first-step peak {:.3})",
            r.units.len(), r.cost, r.cost / std::f64::consts::LN_2 / r.units.len().max(1) as f64, r.reached_end, r.peak
        );
    }
    if mode == "sample" {
        m.save(&path)?;
    }
    Ok(())
}

fn fold_cmd(a: &Args) -> Result<(), String> {
    let mut m = Model::load(&a.get("model", "model.json.gz"))?;
    show_fold(&mut m, &a.text("prefix")?, &a.get("traversal", "reward"), &a.get("backoff", ""), a.usize("top", 8)?)
}

fn show_fold(m: &mut Model, prefix: &str, traversal: &str, backoff: &str, top: usize) -> Result<(), String> {
    let v = serve::fold_json(m, prefix, traversal, backoff, top)?;
    println!("context {prefix:?} -> code {}", fmt_code(&m.code(prefix)));
    for l in v["levels"].as_array().unwrap() {
        let d = if l["level"] == 0 { "(root: every context)".to_string() } else { format!("{:?}", l["decode"].as_str().unwrap_or("")) };
        println!("  level {} node {:<8} seen {:<8} own {:.3}  decodes to {d}", l["level"], l["node"], l["seen"], l["own"].as_f64().unwrap_or(0.0));
    }
    print!("  next ({traversal}):");
    for t in v["top"].as_array().unwrap() {
        print!("  {} {:.3}", t["unit"].as_str().unwrap_or(""), t["p"].as_f64().unwrap_or(0.0));
    }
    println!();
    Ok(())
}

fn score_cmd(a: &Args) -> Result<(), String> {
    let mut m = Model::load(&a.get("model", "model.json.gz"))?;
    let s = m.score(&a.text("text")?, &a.get("traversal", "reward"), &a.get("backoff", ""))?;
    if a.has("json") {
        print_json(&serde_json::to_value(&s).unwrap());
    } else {
        println!("{:.3} bits/unit over {} units, mean reward {:.3}, worst penalty {:.3}", s.bits, s.units, s.mean_reward, s.worst_penalty);
    }
    Ok(())
}

fn info_cmd(a: &Args) -> Result<(), String> {
    let m = Model::load(&a.get("model", "model.json.gz"))?;
    print_json(&m.info());
    Ok(())
}

fn demo_cmd(a: &Args) -> Result<(), String> {
    let texts = read_files(&a.positional)?;
    let cfg = config_from(a)?;
    let mut o = options_from(a)?;
    if a.flags.get("steps").is_none() {
        o.steps = 400;
    }
    o.eval_every = (o.steps / 4).max(1);
    let seed = o.seed as i64;
    let mut tok = Tokenizer::new(cfg.clone(), seed)?;
    let total: usize = texts.iter().map(|t| t.len()).sum();
    println!(
        "== tokenizer: {} parameters, window {}, code {} ({:.1} bits); {} files, {} bytes, {} steps",
        tok.param_count(), cfg.window, tokenizer::levels_string(&cfg.levels), cfg.bits(), texts.len(), total, o.steps
    );
    tok.train(&texts, &o, log_stat)?;
    let mut m = Model::prime(tok, Settings::default(), seed)?;
    println!("\n== primed {} context nodes x {} outcomes ({:.1} MB); reading the files", m.pair.addr.n, pair::OUT, m.pair.memory_bytes() as f64 / 1e6);
    let started = Instant::now();
    let r = m.train_bytes(&texts);
    println!("read {} units in {:.2}s", r["units"], started.elapsed().as_secs_f64());
    let sample = &texts[0][..texts[0].len().min(400)];
    let mut end = sample.len().saturating_sub(1);
    while end > 0 && sample[end] != b' ' {
        end -= 1;
    }
    let ctx_bytes = &sample[..=end.min(sample.len().saturating_sub(1))];
    let ctx_bytes = &ctx_bytes[ctx_bytes.len().saturating_sub(60)..];
    let ctx = String::from_utf8_lossy(ctx_bytes).into_owned();
    println!("\n== what a code stands for");
    show_decode(&m.tok, ctx.as_bytes())?;
    println!("\n== the fold after that context");
    show_fold(&mut m, &ctx, "reward", "", 6)?;
    println!("\n== greedy continuation");
    let p = m.predict(&ctx, 40, "greedy", "reward", 0.0, false, "", 0)?;
    println!("{ctx:?} -> {:?} ({:.2} bits/unit)", p.text, p.cost / std::f64::consts::LN_2 / p.units.len().max(1) as f64);
    println!("\n== verdicts in outcome units");
    let before = m.fold(&ctx, "reward", "")?;
    let first: String = p.text.chars().take(4).collect();
    let alt = "zzz";
    m.reward(&[alt.into()], 1.0, None, false, &ctx, 0)?;
    let english = m.fold(&ctx, "reward", "")?;
    m.reward(&[alt.into()], 1.0, None, false, &ctx, 2)?;
    let game = m.fold(&ctx, "reward", "")?;
    let z = b'z' as usize;
    println!(
        "rewarded {alt:?} after the context at strength 1: P(\"z\") {:.4}; as an English verdict (1/{} per rung) {:.4}; as a two-outcome verdict (1/2 per rung) {:.4}",
        before[z], m.pair.settings.outcomes, english[z], game[z]
    );
    if let Some(&f) = first.as_bytes().first() {
        m.punish(&[first.clone()], 1.0, None, &ctx, 2)?;
        let pen = m.fold(&ctx, "punishment", "")?;
        println!("punished {first:?} after the context as a two-outcome verdict: P({:?}) {:.4} before, {:.4} after", pair::symbol(f as usize), before[f as usize], pen[f as usize]);
    }
    let p2 = m.predict(&ctx, 12, "greedy", "punishment", 0.0, false, "", 0)?;
    println!("punishment traversal now continues {ctx:?} -> {:?}", p2.text);
    let out = a.get("out", "");
    if !out.is_empty() {
        m.save(&out)?;
        println!("\nsaved {out}");
    }
    Ok(())
}

/// A raw-byte baseline folding exactly like the pair over the last k bytes, k = 0..order.
struct Ngram {
    order: usize,
    levels: Vec<HashMap<Vec<u8>, (usize, HashMap<usize, usize>)>>,
    alpha: f64,
    floor: f64,
}

impl Ngram {
    fn new(order: usize, alpha: f64, floor: f64) -> Ngram {
        Ngram { order, levels: (0..=order).map(|_| HashMap::new()).collect(), alpha, floor }
    }
    fn observe(&mut self, text: &[u8]) {
        for (i, &x) in pair::outcomes(text).iter().enumerate() {
            for k in 0..=self.order.min(i) {
                let e = self.levels[k].entry(text[i - k..i].to_vec()).or_insert_with(|| (0, HashMap::new()));
                e.0 += 1;
                *e.1.entry(x).or_insert(0) += 1;
            }
        }
    }
    fn prob(&self, text: &[u8], i: usize, x: usize) -> f64 {
        let mut p = 1.0 / pair::OUT as f64;
        for k in 0..=self.order.min(i) {
            let Some(n) = self.levels[k].get(&text[i - k..i]) else { break };
            let own = n.0 as f64 / (n.0 as f64 + self.alpha);
            p = own * (*n.1.get(&x).unwrap_or(&0) as f64) / n.0 as f64 + (1.0 - own) * p;
        }
        (1.0 - self.floor) * p + self.floor / pair::OUT as f64
    }
    fn contexts(&self) -> usize {
        (1..=self.order).map(|k| self.levels[k].len()).sum()
    }
}

fn bench_cmd(a: &Args) -> Result<(), String> {
    let tok = Tokenizer::load(&a.get("tokenizer", "tokenizer.json.gz"))?;
    let holdout = a.usize("holdout", 5)?.max(2);
    let max_order = a.usize("max-order", 3)?;
    let s = settings_from(a)?;
    let texts = read_files(&a.positional)?;
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, t) in texts.into_iter().enumerate() {
        if (i + 1) % holdout == 0 {
            test.push(t);
        } else {
            train.push(t);
        }
    }
    if train.is_empty() || test.is_empty() {
        return Err("need at least one training and one held-out file".into());
    }
    let train_bytes: usize = train.iter().map(|t| t.len()).sum();
    let test_bytes: usize = test.iter().map(|t| t.len()).sum();
    println!(
        "train {} files ({train_bytes} bytes), held out {} files ({test_bytes} bytes); alpha {:.1} floor {:.3} smoothing {:.2}",
        train.len(), test.len(), s.alpha, s.floor, s.smoothing
    );
    let mut m = Model::prime(tok.clone(), s.clone(), 1)?;
    let started = Instant::now();
    m.train_bytes(&train);
    let read_secs = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (mut bits, mut units) = (0.0, 0usize);
    for t in &test {
        let text = String::from_utf8_lossy(t).into_owned();
        let sc = m.score(&text, "reward", "")?;
        bits += sc.bits * (t.len() + 1) as f64;
        units += t.len() + 1;
    }
    let score_secs = started.elapsed().as_secs_f64();
    let used = m.pair.count.ctx.iter().filter(|&&c| c > 0).count();
    println!("\n{:<34} {:>10} {:>12} {:>10}", "context", "contexts", "bits/byte", "bits");
    println!("{:<34} {used:>10} {:>12.3} {:>10.1}", format!("latent code {:?} ({:.0} bits)", tok.radices(), tok.config.bits()), bits / units as f64, tok.config.bits());
    for order in 0..=max_order {
        let mut g = Ngram::new(order, s.alpha, s.floor);
        for t in &train {
            g.observe(t);
        }
        let mut b = 0.0;
        for t in &test {
            for (i, &x) in pair::outcomes(t).iter().enumerate() {
                b -= g.prob(t, i, x).log2();
            }
        }
        println!("{:<34} {:>10} {:>12.3} {:>10}", format!("raw bytes, last {order}"), g.contexts(), b / units as f64, 8 * order);
    }
    print!("\nthroughput: read {:.0} bytes/s, score {:.0} bytes/s", train_bytes as f64 / read_secs, test_bytes as f64 / score_secs);
    let sample = &train[0][..train[0].len().min(200_000)];
    let started = Instant::now();
    tok.encode_all(sample);
    print!(", encode {:.0} bytes/s", (sample.len() + 1) as f64 / started.elapsed().as_secs_f64());
    let started = Instant::now();
    let mut n = 0;
    while started.elapsed().as_secs_f64() < 1.0 {
        m.predict("the ", 20, "greedy", "reward", 0.0, false, "", 0)?;
        n += 20;
    }
    println!(", predict {:.0} bytes/s", n as f64 / started.elapsed().as_secs_f64());
    Ok(())
}

fn serve_cmd(a: &Args) -> Result<(), String> {
    let path = a.get("model", "model.json.gz");
    let m = Model::load(&path)?;
    let opt = |name: &str| {
        let v = a.get(name, "");
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    };
    serve::run(m, path, serve::ServeOptions { addr: a.get("addr", "127.0.0.1:8080"), web: opt("web"), checkpoint_dir: opt("checkpoint-dir") })
}
