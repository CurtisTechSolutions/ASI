//! The three experiments: a language is learned by credit, stimulation prefers the wide channel, an edge adapts.
//!
//! Each returns a JSON record; [`run`] runs the ones asked for and writes them, and [`tables`] renders a
//! record as the markdown `README.md` quotes.  Everything is deterministic given the seed.
//!
//! * [`learning`] - for each language and machine size, the machine runs random strings, is rewarded when it
//!   ends in the right kind of state and punished when it does not, and its accuracy on held-out strings is
//!   measured as it goes.  The control runs the same strings quietly and credits them the same way: credit
//!   without a traversal lands on nothing.
//! * [`stimulation`] - a fork of three channels of different widths, the probability of each against the
//!   stimulation level; then a surge of stimulation relaxing on the clock.
//! * [`adaptation`] - two edges from one prototype living different lives: one used and rewarded, one used and
//!   punished.  Use widens, punishment narrows, the verdict never fades, the width and the trace do, and the
//!   two weighting functions end up different.

use std::fs;

use crate::compress::{compress, fidelity, Precision};
use crate::edge::Weighting;
use crate::json::Json;
use crate::languages::{examples, language, random_string, Language, ALPHABET, LANGUAGES};
use crate::machine::{Machine, Settings, BASELINE, DISCOUNT, LIFE};
use crate::rng::Rng;

pub const EXPERIMENTS: [&str; 4] = ["learning", "stimulation", "adaptation", "compression"];

fn alphabet() -> Vec<String> {
    ALPHABET.iter().map(|s| s.to_string()).collect()
}

/// Run `episodes` random strings through `machine` and credit each by whether it was classified right.
/// Returns `(episode, accuracy on test)` every `every` episodes when a test set is given.  Quiet, the runs
/// traverse nothing and the credit has nothing to land on.
#[allow(clippy::too_many_arguments)]
pub fn teach_language(
    machine: &mut Machine,
    lang: &Language,
    episodes: usize,
    rng: &mut Rng,
    max_length: usize,
    quiet: bool,
    reward: f64,
    test: Option<&[(String, bool)]>,
    every: usize,
) -> Vec<(usize, f64)> {
    let mut curve = Vec::new();
    for ep in 1..=episodes {
        let s = random_string(rng, max_length);
        let symbols = machine.tokenize(&s).expect("over the alphabet");
        let result = machine.run(&symbols, None, None, quiet);
        if result.accepted == (lang.member)(&s) {
            machine.reward(reward);
        } else {
            machine.punish(reward);
        }
        if let Some(test) = test {
            if every > 0 && ep % every == 0 {
                curve.push((ep, machine.accuracy(test, 0.0)));
            }
        }
    }
    curve
}

pub struct LearningOptions {
    pub episodes: usize,
    pub sizes: Vec<usize>,
    pub seeds: Vec<u64>,
    pub life: f64,
    pub discount: f64,
    pub max_length: usize,
    pub tests: usize,
}

impl Default for LearningOptions {
    fn default() -> LearningOptions {
        LearningOptions {
            episodes: 4000,
            sizes: vec![3, 4, 6, 13],
            seeds: vec![1, 2, 3, 4, 5],
            life: LIFE,
            discount: DISCOUNT,
            max_length: 6,
            tests: 300,
        }
    }
}

/// One arm of the learning experiment across the seeds: credited, or the quiet control.
struct Arm {
    name: &'static str,
    records: Vec<Json>,
    before: Vec<f64>,
    after: Vec<f64>,
    touched: Vec<usize>,
}

impl Arm {
    fn new(name: &'static str) -> Arm {
        Arm {
            name,
            records: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
            touched: Vec::new(),
        }
    }
}

fn curve_json(curve: &[(usize, f64)]) -> Json {
    Json::Array(
        curve
            .iter()
            .map(|(ep, acc)| Json::numbers(&[*ep as f64, *acc]))
            .collect(),
    )
}

pub fn learning(o: &LearningOptions) -> Json {
    let mut rows = Vec::new();
    let n = o.seeds.len() as f64;
    for lang in LANGUAGES.iter() {
        for &states in &o.sizes {
            let mut arms = [Arm::new("credited"), Arm::new("quiet")];
            for &seed in &o.seeds {
                for arm in arms.iter_mut() {
                    let quiet = arm.name == "quiet";
                    let mut rng = Rng::new(seed);
                    let test = examples(lang, o.tests, &mut rng, o.max_length);
                    let settings = Settings {
                        life: o.life,
                        discount: o.discount,
                        seed,
                        ..Settings::default()
                    };
                    let mut m = Machine::new(states, &alphabet(), lang.accepting, settings).expect("a machine");
                    let before = m.accuracy(&test, 0.0);
                    let every = (o.episodes / 8).max(1);
                    let curve = teach_language(
                        &mut m,
                        lang,
                        o.episodes,
                        &mut rng,
                        o.max_length,
                        quiet,
                        1.0,
                        Some(&test),
                        every,
                    );
                    let after = m.accuracy(&test, 0.0);
                    let touched = m.stats().num("touched", 0.0) as usize;
                    arm.records.push(
                        Json::object()
                            .with("seed", (seed as f64).into())
                            .with("before", before.into())
                            .with("after", after.into())
                            .with("curve", curve_json(&curve))
                            .with("touched", touched.into())
                            .with("clock", m.clock.into()),
                    );
                    arm.before.push(before);
                    arm.after.push(after);
                    arm.touched.push(touched);
                }
            }
            let credited = &arms[0].after;
            let quiet = &arms[1].after;
            rows.push(
                Json::object()
                    .with("language", lang.name.into())
                    .with("description", lang.description.into())
                    .with("min_states", lang.min_states.into())
                    .with("states", states.into())
                    .with("before", (arms[0].before.iter().sum::<f64>() / n).into())
                    .with("after_mean", (credited.iter().sum::<f64>() / n).into())
                    .with("after_min", credited.iter().cloned().fold(1.0, f64::min).into())
                    .with("after_max", credited.iter().cloned().fold(0.0, f64::max).into())
                    .with("solved", credited.iter().filter(|&&a| a >= 0.999).count().into())
                    .with("quiet_after", (quiet.iter().sum::<f64>() / n).into())
                    .with(
                        "quiet_touched",
                        arms[1].touched.iter().cloned().max().unwrap_or(0).into(),
                    )
                    .with(
                        "arms",
                        Json::object()
                            .with("credited", Json::Array(arms[0].records.clone()))
                            .with("quiet", Json::Array(arms[1].records.clone())),
                    ),
            );
        }
    }
    Json::object()
        .with("experiment", "learning".into())
        .with(
            "settings",
            Json::object()
                .with("episodes", o.episodes.into())
                .with("sizes", Json::Array(o.sizes.iter().map(|&s| s.into()).collect()))
                .with(
                    "seeds",
                    Json::Array(o.seeds.iter().map(|&s| (s as f64).into()).collect()),
                )
                .with("life", o.life.into())
                .with("discount", o.discount.into())
                .with("max_length", o.max_length.into())
                .with("tests", o.tests.into()),
        )
        .with("rows", Json::Array(rows))
}

pub struct StimulationOptions {
    pub widths: Vec<f64>,
    pub levels: Vec<f64>,
    pub life: f64,
    pub baseline: f64,
    pub surge: f64,
    pub waits: Vec<f64>,
}

impl Default for StimulationOptions {
    fn default() -> StimulationOptions {
        StimulationOptions {
            widths: vec![4.0, 1.0, 0.25],
            levels: vec![0.0, 0.5, 1.0, 2.0, 4.0],
            life: LIFE,
            baseline: BASELINE,
            surge: 3.0,
            waits: vec![0.0, 0.5, 1.0, 2.0, 4.0],
        }
    }
}

fn fork(o: &StimulationOptions) -> Machine {
    let settings = Settings {
        life: o.life,
        baseline: o.baseline,
        ..Settings::default()
    };
    let mut m = Machine::over(o.widths.len(), "a", &[], settings).expect("a machine");
    for (t, &w) in o.widths.iter().enumerate() {
        m.edge_mut(0, 0, t).set_width(w, 0);
    }
    m
}

pub fn stimulation(o: &StimulationOptions) -> Json {
    let m = fork(o);
    let curve: Vec<Json> = o
        .levels
        .iter()
        .map(|&level| {
            Json::object().with("stimulation", level.into()).with(
                "probabilities",
                Json::numbers(&m.probabilities(0, 0, Some(level), None)),
            )
        })
        .collect();
    let relax: Vec<Json> = o
        .waits
        .iter()
        .map(|&lives| {
            let mut m2 = fork(o);
            m2.stimulate(o.surge);
            m2.tick((lives * o.life) as i64);
            let widths: Vec<f64> = m2
                .lattice
                .row(0, 0)
                .iter()
                .map(|e| e.width_at(m2.clock, m2.life))
                .collect();
            Json::object()
                .with("lives", lives.into())
                .with("stimulation", m2.stimulation().into())
                .with("probabilities", Json::numbers(&m2.probabilities(0, 0, None, None)))
                .with("widths", Json::numbers(&widths))
        })
        .collect();
    Json::object()
        .with("experiment", "stimulation".into())
        .with(
            "settings",
            Json::object()
                .with("widths", Json::numbers(&o.widths))
                .with("levels", Json::numbers(&o.levels))
                .with("life", o.life.into())
                .with("baseline", o.baseline.into())
                .with("surge", o.surge.into())
                .with("waits", Json::numbers(&o.waits)),
        )
        .with("curve", Json::Array(curve))
        .with("relaxation", Json::Array(relax))
}

/// Two edges, one life each: rewarded on every use, and punished on every use.
pub fn adaptation(uses: usize, life: f64, waits: &[f64]) -> Json {
    let settings = Settings {
        life,
        ..Settings::default()
    };
    let mut m = Machine::over(3, "a", &[], settings).expect("a machine");
    let mut stages = Vec::new();
    let snapshot = |m: &Machine, label: &str| -> Json {
        let (clock, life, stim) = (m.clock, m.life, m.stimulation());
        Json::object()
            .with("stage", label.into())
            .with("clock", clock.into())
            .with("probabilities", Json::numbers(&m.probabilities(0, 0, None, None)))
            .with(
                "edges",
                Json::object()
                    .with("rewarded", m.edge(0, 0, 1).describe(clock, life, stim))
                    .with("punished", m.edge(0, 0, 2).describe(clock, life, stim))
                    .with("untouched", m.edge(0, 0, 0).describe(clock, life, stim)),
            )
    };
    stages.push(snapshot(&m, "fresh"));
    for i in 1..=uses {
        m.teach(0, 0, 1, 1.0).expect("edge");
        m.teach(0, 0, 2, -1.0).expect("edge");
        if i == 1 || i == 5 || i == uses {
            stages.push(snapshot(&m, &format!("after {i} uses each")));
        }
    }
    let end = m.clock;
    for &lives in waits.iter().skip(1) {
        m.tick(end + (lives * life) as i64 - m.clock);
        let label = format!("{} {} later", lives, if lives == 1.0 { "life" } else { "lives" });
        stages.push(snapshot(&m, &label));
    }
    Json::object()
        .with("experiment", "adaptation".into())
        .with(
            "settings",
            Json::object()
                .with("uses", uses.into())
                .with("life", life.into())
                .with("waits", Json::numbers(waits))
                .with("prototype", Weighting::default().to_json()),
        )
        .with("stages", Json::Array(stages))
}

// ---- rendering ----------------------------------------------------------------------------------------------------

/// Default 13 × 13 × 13 machines folded into their central node: fresh, and taught each language, at each
/// precision; one rebuilt from the central node outward, a shell at a time; and every language taught again with
/// the matrix folded into its central node every `every` transitions, rebuilt from the code each time.
pub fn compression(episodes: usize, seed: u64, life: f64, tests: usize, every: u64) -> Result<Json, String> {
    let (mut rows, mut expansion, mut periodic) = (Vec::new(), Vec::new(), Vec::new());
    let mut labels: Vec<(&str, Option<Language>)> = vec![("fresh", None)];
    labels.extend(LANGUAGES.iter().map(|l| (l.name, Some(*l))));
    for (label, lang) in labels {
        let accepting: &[usize] = lang.as_ref().map(|l| l.accepting).unwrap_or(&[0]);
        let mut m = Machine::default_shape(
            accepting,
            Settings {
                life,
                seed,
                ..Settings::default()
            },
        )?;
        let test = match &lang {
            Some(l) => examples(l, tests, &mut Rng::new(seed + 1000), 6),
            None => Vec::new(),
        };
        if let Some(l) = &lang {
            teach_language(&mut m, l, episodes, &mut Rng::new(seed), 6, false, 1.0, None, 0);
        }
        for p in Precision::ALL {
            let core = compress(&m, Some(p), None);
            let mut r = core.decompress(None)?;
            let f = fidelity(&m, &r);
            let s = core.summary();
            let (acc, acc_r) = if test.is_empty() {
                (Json::Null, Json::Null)
            } else {
                (m.accuracy(&test, 0.0).into(), r.accuracy(&test, 0.0).into())
            };
            rows.push(
                Json::object()
                    .with("machine", label.into())
                    .with("precision", p.name().into())
                    .with("touched", s.get("touched").cloned().unwrap_or(Json::Null))
                    .with("bytes", s.get("bytes").cloned().unwrap_or(Json::Null))
                    .with("dense_bytes", s.get("dense_bytes").cloned().unwrap_or(Json::Null))
                    .with("ratio", s.get("ratio").cloned().unwrap_or(Json::Null))
                    .with("record_bytes", s.get("record_bytes").cloned().unwrap_or(Json::Null))
                    .with("lossless", f.get("lossless").cloned().unwrap_or(Json::Null))
                    .with(
                        "max_relative_error",
                        f.get("max_relative_error").cloned().unwrap_or(Json::Null),
                    )
                    .with("mean_kl", f.get("mean_kl").cloned().unwrap_or(Json::Null))
                    .with("max_kl", f.get("max_kl").cloned().unwrap_or(Json::Null))
                    .with("greedy_changed", f.get("greedy_changed").cloned().unwrap_or(Json::Null))
                    .with("accuracy", acc)
                    .with("accuracy_rebuilt", acc_r),
            );
            if label == "even-b" && p == Precision::Exact {
                let table = core.shell_table();
                let shells = table.as_array().cloned().unwrap_or_default();
                for k in 1..=shells.len() {
                    let mut part = core.decompress(Some(k))?;
                    let pf = fidelity(&m, &part);
                    expansion.push(
                        Json::object()
                            .with("shells", k.into())
                            .with(
                                "cells",
                                shells[..k].iter().map(|x| x.num("cells", 0.0)).sum::<f64>().into(),
                            )
                            .with(
                                "touched",
                                shells[..k].iter().map(|x| x.num("touched", 0.0)).sum::<f64>().into(),
                            )
                            .with(
                                "greedy_changed",
                                pf.get("greedy_changed").cloned().unwrap_or(Json::Null),
                            )
                            .with("max_kl", pf.get("max_kl").cloned().unwrap_or(Json::Null))
                            .with("accuracy", part.accuracy(&test, 0.0).into()),
                    );
                }
            }
        }
    }
    for lang in LANGUAGES.iter() {
        let test = examples(lang, tests, &mut Rng::new(seed + 1000), 6);
        let mut arms: Vec<(String, Option<Precision>)> = vec![("never".to_string(), None)];
        arms.extend(Precision::ALL.iter().map(|&p| (format!("every {every}"), Some(p))));
        for (label, precision) in arms {
            let settings = Settings {
                life,
                seed,
                compress_every: if precision.is_some() { every } else { 0 },
                compress_precision: precision.unwrap_or(Precision::Exact),
                compress_rebuild: precision.is_some(),
                ..Settings::default()
            };
            let mut m = Machine::default_shape(lang.accepting, settings)?;
            teach_language(&mut m, lang, episodes, &mut Rng::new(seed), 6, false, 1.0, None, 0);
            periodic.push(
                Json::object()
                    .with("language", lang.name.into())
                    .with("compression", label.into())
                    .with(
                        "precision",
                        precision.map(|p| Json::from(p.name())).unwrap_or(Json::Null),
                    )
                    .with("compressions", (m.compressions as f64).into())
                    .with(
                        "bytes",
                        m.core.as_ref().map(|c| Json::from(c.bytes())).unwrap_or(Json::Null),
                    )
                    .with("accuracy", m.accuracy(&test, 0.0).into()),
            );
        }
    }
    Ok(Json::object()
        .with("experiment", "compression".into())
        .with(
            "settings",
            Json::object()
                .with("episodes", episodes.into())
                .with("seed", (seed as f64).into())
                .with("life", life.into())
                .with("tests", tests.into())
                .with("every", (every as f64).into())
                .with("shape", Json::numbers(&[13.0, 13.0, 13.0])),
        )
        .with("rows", Json::Array(rows))
        .with("expansion", Json::Array(expansion))
        .with("periodic", Json::Array(periodic)))
}

fn compression_table(r: &Json) -> String {
    let s = r.get("settings").cloned().unwrap_or(Json::object());
    let mut lines = vec![
        format!(
            "compression: default 13 x 13 x 13 machines, fresh and taught each language for {} episodes, folded into \
             the central node (6, g, 6) at each precision",
            s.num("episodes", 0.0)
        ),
        String::new(),
        "| machine | precision | touched | bytes | smaller than dense | lossless | max relative error | max KL | \
         greedy changed | accuracy: original → rebuilt |"
            .to_string(),
        "|---|---|---|---|---|---|---|---|---|---|".to_string(),
    ];
    for row in arr(r, "rows") {
        let acc = match (
            row.get("accuracy").and_then(Json::as_f64),
            row.get("accuracy_rebuilt").and_then(Json::as_f64),
        ) {
            (Some(a), Some(b)) => format!("{a:.3} → {b:.3}"),
            _ => "–".to_string(),
        };
        lines.push(format!(
            "| {} | {} | {} | {} | {:.1}× | {} | {:.1e} | {:.1e} | {} | {} |",
            row.str_or("machine", ""),
            row.str_or("precision", ""),
            row.num("touched", 0.0),
            row.num("bytes", 0.0),
            row.num("ratio", 0.0),
            if row.bool_or("lossless", false) { "yes" } else { "no" },
            row.num("max_relative_error", 0.0),
            row.num("max_kl", 0.0),
            row.num("greedy_changed", 0.0),
            acc
        ));
    }
    lines.push(String::new());
    lines.push("even-b, exact, rebuilt from the central node outward".to_string());
    lines.push(String::new());
    lines.push("| shells rebuilt | cells | touched edges restored | greedy changed | accuracy |".to_string());
    lines.push("|---|---|---|---|---|".to_string());
    for row in arr(r, "expansion") {
        lines.push(format!(
            "| {} | {} | {} | {} | {:.3} |",
            row.num("shells", 0.0),
            row.num("cells", 0.0),
            row.num("touched", 0.0),
            row.num("greedy_changed", 0.0),
            row.num("accuracy", 0.0)
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "every language taught again, the matrix folded into its central node every {} transitions and rebuilt \
         from the code each time",
        s.num("every", 0.0)
    ));
    lines.push(String::new());
    lines.push("| language | compression | precision | compressions | last code, bytes | accuracy |".to_string());
    lines.push("|---|---|---|---|---|---|".to_string());
    for row in arr(r, "periodic") {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {:.3} |",
            row.str_or("language", ""),
            row.str_or("compression", ""),
            row.str_or("precision", "–"),
            row.num("compressions", 0.0),
            row.get("bytes")
                .and_then(Json::as_f64)
                .map(|b| b.to_string())
                .unwrap_or_else(|| "–".to_string()),
            row.num("accuracy", 0.0)
        ));
    }
    lines.join("\n")
}

pub fn tables(record: &Json) -> String {
    match record.str_or("experiment", "") {
        "learning" => learning_table(record),
        "stimulation" => stimulation_table(record),
        "compression" => compression_table(record),
        _ => adaptation_table(record),
    }
}

fn arr<'a>(v: &'a Json, key: &str) -> &'a [Json] {
    v.get(key).and_then(Json::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn learning_table(r: &Json) -> String {
    let s = r.get("settings").cloned().unwrap_or(Json::object());
    let seeds = arr(&s, "seeds").len();
    let mut lines = vec![
        format!(
            "learning: {} episodes, strings up to {} symbols, {} seeds, {} held-out strings; the quiet arm runs the \
             same strings without traversing",
            s.num("episodes", 0.0),
            s.num("max_length", 0.0),
            seeds,
            s.num("tests", 0.0)
        ),
        String::new(),
        "| language | states (min) | before | after: mean | min | max | solved | quiet after |".to_string(),
        "|---|---|---|---|---|---|---|---|".to_string(),
    ];
    for row in arr(r, "rows") {
        lines.push(format!(
            "| {} ({}) | {} ({}) | {:.2} | **{:.2}** | {:.2} | {:.2} | {}/{} | {:.2} |",
            row.str_or("language", ""),
            row.str_or("description", ""),
            row.num("states", 0.0),
            row.num("min_states", 0.0),
            row.num("before", 0.0),
            row.num("after_mean", 0.0),
            row.num("after_min", 0.0),
            row.num("after_max", 0.0),
            row.num("solved", 0.0),
            seeds,
            row.num("quiet_after", 0.0)
        ));
    }
    lines.join("\n")
}

fn probs_cells(c: &Json) -> String {
    arr(c, "probabilities")
        .iter()
        .map(|p| format!("{:.3}", p.as_f64().unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn stimulation_table(r: &Json) -> String {
    let s = r.get("settings").cloned().unwrap_or(Json::object());
    let widths = arr(&s, "widths");
    let head = widths
        .iter()
        .map(|w| format!("width {}", w.as_f64().unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(" | ");
    let rule = "---|".repeat(widths.len());
    let mut lines = vec![
        format!(
            "stimulation: a fork of {} channels; probability of each by stimulation level",
            widths.len()
        ),
        String::new(),
        format!("| stimulation | {head} |"),
        format!("|---|{rule}"),
    ];
    for c in arr(r, "curve") {
        lines.push(format!("| {} | {} |", c.num("stimulation", 0.0), probs_cells(c)));
    }
    lines.push(String::new());
    lines.push(format!(
        "a surge of +{} over the baseline {}, relaxing on the clock",
        s.num("surge", 0.0),
        s.num("baseline", 0.0)
    ));
    lines.push(String::new());
    lines.push(format!("| lives since | stimulation | {head} |"));
    lines.push(format!("|---|---|{rule}"));
    for c in arr(r, "relaxation") {
        lines.push(format!(
            "| {} | {:.2} | {} |",
            c.num("lives", 0.0),
            c.num("stimulation", 0.0),
            probs_cells(c)
        ));
    }
    lines.join("\n")
}

fn adaptation_table(r: &Json) -> String {
    let mut lines = vec![
        "adaptation: two edges from one prototype, one rewarded on every use and one punished".to_string(),
        String::new(),
        "| stage | clock | P(rewarded) | P(punished) | P(untouched) | width + | width − | net + | net − | recent + | \
         bias + | bias − |"
            .to_string(),
        "|---|---|---|---|---|---|---|---|---|---|---|---|".to_string(),
    ];
    let stages = arr(r, "stages");
    for st in stages {
        let p = arr(st, "probabilities");
        let pn = |i: usize| p.get(i).and_then(Json::as_f64).unwrap_or(0.0);
        let edges = st.get("edges").cloned().unwrap_or(Json::object());
        let g = edges.get("rewarded").cloned().unwrap_or(Json::object());
        let b = edges.get("punished").cloned().unwrap_or(Json::object());
        let bias = |e: &Json| arr(e, "weighting").first().and_then(Json::as_f64).unwrap_or(0.0);
        lines.push(format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:+.2} | {:+.2} | {:.2} | {:+.2} | {:+.2} |",
            st.str_or("stage", ""),
            st.num("clock", 0.0),
            pn(1),
            pn(2),
            pn(0),
            g.num("width", 0.0),
            b.num("width", 0.0),
            g.num("net", 0.0),
            b.num("net", 0.0),
            g.num("recent", 0.0),
            bias(&g),
            bias(&b)
        ));
    }
    if let Some(last) = stages.last() {
        let edges = last.get("edges").cloned().unwrap_or(Json::object());
        let w = |k: &str| {
            arr(&edges.get(k).cloned().unwrap_or(Json::object()), "weighting")
                .iter()
                .map(|v| format!("{:.3}", v.as_f64().unwrap_or(0.0)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(String::new());
        lines.push("the two weighting functions at the end (bias, seen, recent, net, age, width, rate):".to_string());
        lines.push(format!("  rewarded: [{}]", w("rewarded")));
        lines.push(format!("  punished: [{}]", w("punished")));
    }
    lines.join("\n")
}

// ---- running -------------------------------------------------------------------------------------------------------

/// Run the experiments named (or `all`) and write each as `<out>/<name>_results.json` when `out` is given.
pub fn run(which: &[String], out: Option<&str>, episodes: usize, life: f64) -> Result<Vec<Json>, String> {
    let names: Vec<String> = if which.iter().any(|w| w == "all") {
        EXPERIMENTS.iter().map(|s| s.to_string()).collect()
    } else {
        which.to_vec()
    };
    let mut records = Vec::new();
    for name in &names {
        let rec = match name.as_str() {
            "learning" => learning(&LearningOptions {
                episodes,
                life,
                ..LearningOptions::default()
            }),
            "stimulation" => stimulation(&StimulationOptions {
                life,
                ..StimulationOptions::default()
            }),
            "adaptation" => adaptation(20, life, &[0.0, 1.0, 4.0]),
            "compression" => compression(episodes, 1, life, 300, 500)?,
            other => {
                return Err(format!(
                    "no experiment named {other:?}; choose from {EXPERIMENTS:?} or all"
                ))
            }
        };
        if let Some(dir) = out {
            fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            let path = format!("{dir}/{name}_results.json");
            fs::write(&path, rec.pretty(1)).map_err(|e| format!("{path}: {e}"))?;
        }
        records.push(rec);
    }
    Ok(records)
}

/// A language by name, for the CLI.
pub fn language_by_name(name: &str) -> Result<Language, String> {
    language(name)
}
