//! `gamelattice <command>`:
//!
//!     demo        a body learns a language: the encoder names the lattice's states, the oracle credits each step
//!     train       the same over --episodes random strings, printing held-out accuracy
//!     nim         babble, then play Nim against a random opponent, printing legality and optimal moves
//!     experiment  E1 (the four languages against the plain oracle lattice) and E2 (Nim by window), to --out
//!     serve       the HTTP API and the built frontend (`DESIGN.md` §16): a body for --language and a Nim body,
//!                 on --host:--port, the frontend from --frontend-dir, the results from --results-dir
//!
//! Everything is deterministic given --seed.

#![allow(clippy::field_reassign_with_default)]

use std::env;
use std::process;

use gamelattice::babble::babble_nim;
use gamelattice::body::{Body, BodySettings, NimBody, NimSettings};
use gamelattice::experiment::{self, E1Settings, E2Settings};
use gamelattice::server::{self, ServeSettings};
use latticefsm::languages::{examples, language};
use latticefsm::rng::Rng;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    arg(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn usage() -> ! {
    eprintln!(
        "usage: gamelattice <demo|train|nim|experiment|serve> [--language L] [--episodes N] [--states S] [--window W] \
         [--n N] [--seed K] [--plain] [--games N] [--babble N] [--which e1|e2|all] [--seeds a,b,c] [--out DIR] \
         [--arms a,b] [--host H] [--port P] [--frontend-dir DIR] [--results-dir DIR]"
    );
    process::exit(2)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(command) = args.first().cloned() else { usage() };
    let seed: u64 = num(&args, "--seed", 1);
    match command.as_str() {
        "demo" | "train" => {
            let lang_name = arg(&args, "--language").unwrap_or_else(|| "even-b".to_string());
            let lang = language(&lang_name).unwrap_or_else(|e| {
                eprintln!("{e}");
                process::exit(2)
            });
            let episodes: usize = num(&args, "--episodes", if command == "demo" { 1000 } else { 4000 });
            let mut settings = BodySettings::default();
            settings.encoder.states = num(&args, "--states", 13);
            settings.encoder.window = num(&args, "--window", 8);
            settings.encoder.n = num(&args, "--n", 1);
            settings.encoder_sets_state = !flag(&args, "--plain");
            settings.forward_lessons = flag(&args, "--lessons");
            settings.seed = seed;
            let mut rng = Rng::new(seed.wrapping_mul(1000).wrapping_add(17));
            let test = examples(&lang, 300, &mut rng, 6);
            let mut body = Body::for_language(&lang, settings).unwrap_or_else(|e| {
                eprintln!("{e}");
                process::exit(1)
            });
            let every = (episodes / 10).max(1);
            for ep in 1..=episodes {
                let s = latticefsm::languages::random_string(&mut rng, 6);
                body.run(&s, true);
                if ep % every == 0 || ep == episodes {
                    let (enc, lat) = body.accuracy(&test);
                    println!(
                        "episode {ep:>5}: encoder {enc:.3}  lattice {lat:.3}  agreement {:.2}  diversity {:.2} bits  thinks {}",
                        body.agreement.rate(),
                        body.encoder.diversity(),
                        body.encoder.thinks
                    );
                }
            }
            for s in ["", "ab", "abab", "bb", "aab", "babab"] {
                let c = body.classify(s);
                println!(
                    "{s:>6?}: encoder {} lattice {} (named {:?}; in the language: {})",
                    c.encoder,
                    c.lattice,
                    c.named,
                    (lang.member)(s)
                );
            }
        }
        "nim" => {
            let mut settings = NimSettings::default();
            settings.seed = seed;
            settings.encoder.states = num(&args, "--states", 13);
            settings.encoder.window = num(&args, "--window", 8);
            settings.repertoire.window = settings.encoder.window.max(1);
            let games: usize = num(&args, "--games", 400);
            let babble: usize = num(&args, "--babble", 200);
            let mut body = NimBody::new(settings).unwrap_or_else(|e| {
                eprintln!("{e}");
                process::exit(1)
            });
            let b = babble_nim(&mut body, babble);
            let lessons: usize = b.iter().map(|r| r.lessons).sum();
            println!(
                "babbled {babble} games: {lessons} lessons on the forward model, I = {:.3} bits",
                body.information().0
            );
            let mut won = 0;
            let mut first = 0;
            let mut moves = 0;
            let mut optimal = 0;
            for i in 0..games {
                let r = body.play(i % 2 == 0, false);
                won += usize::from(r.won);
                first += r.first_try;
                moves += r.our_moves;
                optimal += r.optimal;
                if (i + 1) % (games / 5).max(1) == 0 {
                    println!(
                        "game {:>4}: wins {:.2}  legal@1 {:.3}  optimal {:.2}  I {:.3} bits  agreement {:.2}",
                        i + 1,
                        won as f64 / (i + 1) as f64,
                        first as f64 / moves.max(1) as f64,
                        optimal as f64 / moves.max(1) as f64,
                        body.information().0,
                        body.agreement.rate()
                    );
                }
            }
        }
        "experiment" => {
            let which = arg(&args, "--which").unwrap_or_else(|| "all".to_string());
            let out = arg(&args, "--out").unwrap_or_else(|| "../results".to_string());
            let seeds: Vec<u64> = arg(&args, "--seeds")
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_else(|| vec![1, 2, 3]);
            let mut log = |line: &str| println!("{line}");
            if which == "e1" || which == "all" {
                let mut e = E1Settings::default();
                e.episodes = num(&args, "--episodes", 4000);
                e.states = num(&args, "--states", 13);
                e.window = num(&args, "--window", 8);
                e.n = num(&args, "--n", 1);
                e.seeds = seeds.clone();
                e.arms = arg(&args, "--arms")
                    .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                    .unwrap_or_default();
                let doc = experiment::e1(&e, &mut log);
                match experiment::write(&out, "e1", &doc) {
                    Ok(p) => println!("wrote {p}"),
                    Err(e) => eprintln!("{e}"),
                }
            }
            if which == "e2" || which == "all" {
                let mut e = E2Settings::default();
                e.games = num(&args, "--games", 400);
                e.babble_games = num(&args, "--babble", 200);
                e.states = num(&args, "--states", 13);
                e.seeds = seeds;
                if let Some(w) = arg(&args, "--windows") {
                    e.windows = w.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
                let doc = experiment::e2(&e, &mut log);
                match experiment::write(&out, "e2", &doc) {
                    Ok(p) => println!("wrote {p}"),
                    Err(e) => eprintln!("{e}"),
                }
            }
        }
        "serve" => {
            let mut s = ServeSettings::default();
            s.language = arg(&args, "--language").unwrap_or_else(|| "even-b".to_string());
            s.body.encoder.states = num(&args, "--states", 13);
            s.body.encoder.window = num(&args, "--window", 8);
            s.body.encoder.n = num(&args, "--n", 1);
            s.body.seed = seed;
            s.nim.encoder.states = s.body.encoder.states;
            s.nim.encoder.window = s.body.encoder.window;
            s.nim.repertoire.window = s.nim.encoder.window.max(1);
            s.nim.seed = seed;
            s.frontend = arg(&args, "--frontend-dir")
                .map(std::path::PathBuf::from)
                .or_else(server::default_frontend_dir);
            s.results = arg(&args, "--results-dir")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(server::default_results_dir);
            let addr = format!(
                "{}:{}",
                arg(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string()),
                num::<u16>(&args, "--port", 8000)
            );
            eprintln!(
                "gamelattice serving http://{addr}/  ({} on {} states, window {}, seed {seed}; frontend {}; results {})",
                s.language,
                s.body.encoder.states,
                s.body.encoder.window,
                s.frontend
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none: /api/ only".to_string()),
                s.results.display()
            );
            if let Err(e) = server::serve(s, &addr, None) {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
        _ => usage(),
    }
}
