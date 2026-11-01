//! The measurement plan (`DESIGN.md` §17): each experiment names the number it must beat and its controls,
//! and writes `results/<name>.json`.

use std::fs;
use std::time::Instant;

use latticefsm::json::Json;
use latticefsm::languages::{examples, LANGUAGES};
use latticefsm::rng::Rng;

use crate::babble::babble_nim;
use crate::body::{Body, BodySettings, NimBody, NimSettings};

/// The numbers E1 is measured against (`LatticeFSM/README.md`, "Credit by an oracle"): a 13-state lattice
/// alone over 4 000 episodes, the mean over the four languages.
pub const E1_OWN_WHOLE_RUN: f64 = 0.894;
pub const E1_OWN_ORACLE: f64 = 0.924;
pub const E1_EXPLORE_ORACLE: f64 = 0.944;

#[derive(Clone, Debug)]
pub struct E1Settings {
    pub episodes: usize,
    pub seeds: Vec<u64>,
    pub states: usize,
    pub window: usize,
    pub n: usize,
    pub test: usize,
    pub max_length: usize,
    /// Which arms to run; empty for all.
    pub arms: Vec<String>,
}

impl Default for E1Settings {
    fn default() -> E1Settings {
        E1Settings {
            episodes: 4000,
            seeds: vec![1, 2, 3],
            states: 13,
            window: 8,
            n: 1,
            test: 300,
            max_length: 6,
            arms: Vec::new(),
        }
    }
}

/// The arms of E1: the game against the plain oracle lattice, and the controls of `DESIGN.md` §17.
pub fn e1_arms(e: &E1Settings) -> Vec<(String, BodySettings)> {
    let mut base = BodySettings::default();
    base.encoder.states = e.states;
    base.encoder.window = e.window;
    base.encoder.n = e.n;
    let mut arms: Vec<(String, BodySettings)> = Vec::new();
    let mut plain = base.clone();
    plain.encoder_sets_state = false;
    arms.push(("plain-oracle".to_string(), plain));
    arms.push(("game".to_string(), base.clone()));
    let mut b5 = base.clone();
    b5.credit.baseline = 0.5;
    arms.push(("game-baseline-0.5".to_string(), b5));
    let mut b1 = base.clone();
    b1.credit.baseline = 1.0;
    arms.push(("game-baseline-1".to_string(), b1));
    let mut rewards = base.clone();
    rewards.punish = false;
    arms.push(("game-rewards-only".to_string(), rewards));
    let mut whole = base.clone();
    whole.whole_run = true;
    arms.push(("game-whole-run".to_string(), whole));
    let mut lessons = base.clone();
    lessons.forward_lessons = true;
    arms.push(("game-lessons".to_string(), lessons));
    if e.arms.is_empty() {
        arms
    } else {
        arms.into_iter().filter(|(n, _)| e.arms.contains(n)).collect()
    }
}

/// E1: does the stage game earn its cost?  The four languages, each arm, each seed; held-out accuracy of the
/// encoder's verdict and of the lattice's own, with the readouts the design asks for beside them.
pub fn e1(e: &E1Settings, log: &mut dyn FnMut(&str)) -> Json {
    let started = Instant::now();
    let mut rows: Vec<Json> = Vec::new();
    let mut summary = Json::object();
    for (arm, settings) in e1_arms(e) {
        let mut arm_enc = Vec::new();
        let mut arm_lat = Vec::new();
        for lang in LANGUAGES.iter() {
            for &seed in &e.seeds {
                let mut s = settings.clone();
                s.seed = seed;
                let mut rng = Rng::new(seed.wrapping_mul(1000).wrapping_add(17));
                let test = examples(lang, e.test, &mut rng, e.max_length);
                let mut body = Body::for_language(lang, s).expect("a body");
                let t0 = Instant::now();
                body.teach(e.episodes, &mut rng, e.max_length);
                let (enc, lat) = body.accuracy(&test);
                let (mean_wealth, gini) = body.encoder.wealth_stats();
                let bits = {
                    let mut lb = 0.0;
                    let mut eb = 0.0;
                    for (s, _) in test.iter().take(50) {
                        let c = body.classify(s);
                        lb += c.lattice_bits;
                        eb += c.encoder_bits;
                    }
                    (lb / 50.0, eb / 50.0)
                };
                let touched = body.lattice.machine.lattice.touched().count();
                let mut row = Json::object();
                row.set("arm", Json::String(arm.clone()));
                row.set("language", Json::String(lang.name.to_string()));
                row.set("seed", Json::Number(seed as f64));
                row.set("encoder_accuracy", Json::Number(enc));
                row.set("lattice_accuracy", Json::Number(lat));
                row.set("agreement", Json::Number(body.agreement.rate()));
                row.set("diversity_bits", Json::Number(body.encoder.diversity()));
                row.set("mean_wealth", Json::Number(mean_wealth));
                row.set("gini", Json::Number(gini));
                row.set("thinks", Json::Number(body.encoder.thinks as f64));
                row.set("lessons", Json::Number(body.lessons as f64));
                row.set("lattice_bits_per_symbol", Json::Number(bits.0));
                row.set("encoder_bits_per_symbol", Json::Number(bits.1));
                row.set("touched_edges", Json::Number(touched as f64));
                row.set("graph_nodes", Json::Number(body.encoder.graph.num_nodes() as f64));
                row.set("seconds", Json::Number(t0.elapsed().as_secs_f64()));
                log(&format!(
                    "e1 {arm:>18} {:<12} seed {seed}: encoder {enc:.3} lattice {lat:.3} agree {:.2} diversity {:.2} bits gini {gini:.2} ({:.1}s)",
                    lang.name,
                    body.agreement.rate(),
                    body.encoder.diversity(),
                    t0.elapsed().as_secs_f64()
                ));
                arm_enc.push(enc);
                arm_lat.push(lat);
                rows.push(row);
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let mut s = Json::object();
        s.set("encoder", Json::Number(mean(&arm_enc)));
        s.set("lattice", Json::Number(mean(&arm_lat)));
        summary.set(&arm, s);
    }
    let mut out = Json::object();
    out.set("experiment", Json::String("E1".to_string()));
    out.set(
        "question",
        Json::String("does the stage game earn its cost on the four languages?".to_string()),
    );
    let mut beat = Json::object();
    beat.set("own-whole-run", Json::Number(E1_OWN_WHOLE_RUN));
    beat.set("own-oracle", Json::Number(E1_OWN_ORACLE));
    beat.set("explore-oracle", Json::Number(E1_EXPLORE_ORACLE));
    out.set("to_beat", beat);
    out.set("episodes", Json::Number(e.episodes as f64));
    out.set("states", Json::Number(e.states as f64));
    out.set("window", Json::Number(e.window as f64));
    out.set(
        "seeds",
        Json::numbers(&e.seeds.iter().map(|s| *s as f64).collect::<Vec<_>>()),
    );
    out.set("summary", summary);
    out.set("rows", Json::Array(rows));
    out.set("seconds", Json::Number(started.elapsed().as_secs_f64()));
    out
}

#[derive(Clone, Debug)]
pub struct E2Settings {
    pub babble_games: usize,
    pub games: usize,
    pub seeds: Vec<u64>,
    pub windows: Vec<usize>,
    pub states: usize,
}

impl Default for E2Settings {
    fn default() -> E2Settings {
        E2Settings {
            babble_games: 200,
            games: 400,
            seeds: vec![1, 2, 3],
            windows: vec![1, 4, 8, 16],
            states: 13,
        }
    }
}

/// E2: the window as memory, on Nim.  Babbling first, then play against a random opponent; per window width:
/// first-try legality, refusals a move, wins, optimal moves, `I(named state; nim-sum)` in bits, coverage.
pub fn e2(e: &E2Settings, log: &mut dyn FnMut(&str)) -> Json {
    let started = Instant::now();
    let mut rows = Vec::new();
    for &w in &e.windows {
        for &seed in &e.seeds {
            let mut s = NimSettings::default();
            s.seed = seed;
            s.encoder.window = w;
            s.encoder.states = e.states;
            s.repertoire.window = w.max(1);
            let mut body = NimBody::new(s).expect("a nim body");
            let t0 = Instant::now();
            let babbled = babble_nim(&mut body, e.babble_games);
            let babble_lessons: usize = babbled.iter().map(|r| r.lessons).sum();
            let (mi_after_babble, _) = body.information();
            let mut won = 0;
            let mut first = 0;
            let mut moves = 0;
            let mut refusals = 0;
            let mut optimal = 0;
            let mut abstained = 0;
            let mut late_won = 0;
            let mut late_optimal = 0;
            let mut late_moves = 0;
            for i in 0..e.games {
                let r = body.play(i % 2 == 0, false);
                won += usize::from(r.won);
                first += r.first_try;
                moves += r.our_moves;
                refusals += r.refusals;
                optimal += r.optimal;
                abstained += r.abstained;
                if i >= e.games / 2 {
                    late_won += usize::from(r.won);
                    late_optimal += r.optimal;
                    late_moves += r.our_moves;
                }
            }
            let (mi, coverage) = body.information();
            let mut row = Json::object();
            row.set("window", Json::Number(w as f64));
            row.set("seed", Json::Number(seed as f64));
            row.set("legal_first_try", Json::Number(first as f64 / moves.max(1) as f64));
            row.set("refusals_per_move", Json::Number(refusals as f64 / moves.max(1) as f64));
            row.set(
                "abstained_per_move",
                Json::Number(abstained as f64 / moves.max(1) as f64),
            );
            row.set("win_rate", Json::Number(won as f64 / e.games.max(1) as f64));
            row.set(
                "win_rate_second_half",
                Json::Number(late_won as f64 / (e.games - e.games / 2).max(1) as f64),
            );
            row.set("optimal_rate", Json::Number(optimal as f64 / moves.max(1) as f64));
            row.set(
                "optimal_rate_second_half",
                Json::Number(late_optimal as f64 / late_moves.max(1) as f64),
            );
            row.set("information_bits", Json::Number(mi));
            row.set("information_bits_after_babble", Json::Number(mi_after_babble));
            row.set("states_used", Json::Number(coverage as f64));
            row.set("agreement", Json::Number(body.agreement.rate()));
            row.set("babble_lessons", Json::Number(babble_lessons as f64));
            row.set("diversity_bits", Json::Number(body.encoder.diversity()));
            row.set("graph_nodes", Json::Number(body.encoder.graph.num_nodes() as f64));
            row.set("seconds", Json::Number(t0.elapsed().as_secs_f64()));
            log(&format!(
                "e2 W={w:<2} seed {seed}: legal@1 {:.3} refusals/move {:.2} wins {:.2} (late {:.2}) optimal {:.2} (late {:.2}) I {:.3} bits over {} states agree {:.2} ({:.1}s)",
                first as f64 / moves.max(1) as f64,
                refusals as f64 / moves.max(1) as f64,
                won as f64 / e.games.max(1) as f64,
                late_won as f64 / (e.games - e.games / 2).max(1) as f64,
                optimal as f64 / moves.max(1) as f64,
                late_optimal as f64 / late_moves.max(1) as f64,
                mi,
                coverage,
                body.agreement.rate(),
                t0.elapsed().as_secs_f64()
            ));
            rows.push(row);
        }
    }
    let mut out = Json::object();
    out.set("experiment", Json::String("E2".to_string()));
    out.set(
        "question",
        Json::String(
            "is the walk window a memory? Nim: legality, optimal moves and I(state; nim-sum) by window".to_string(),
        ),
    );
    out.set("random_opponent_win_rate_to_beat", Json::Number(0.5));
    out.set("random_legal_first_try", Json::Number(0.0));
    out.set("rows", Json::Array(rows));
    out.set("seconds", Json::Number(started.elapsed().as_secs_f64()));
    out
}

pub fn write(out_dir: &str, name: &str, doc: &Json) -> Result<String, String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let path = format!("{}/{}.json", out_dir.trim_end_matches('/'), name);
    fs::write(&path, doc.pretty(2)).map_err(|e| e.to_string())?;
    Ok(path)
}
