//! The machine: the matrix, the edge, credit, stimulation, time, persistence, learning.

use latticefsm::edge::{Edge, Weighting, COEFFICIENT_LIMIT, WIDTH_MAX, WIDTH_MIN, WIDTH_REST};
use latticefsm::experiment::{self, teach_language, LearningOptions, StimulationOptions};
use latticefsm::json::parse;
use latticefsm::languages::{examples, language};
use latticefsm::lattice::Lattice;
use latticefsm::machine::{load_machine, Machine, Settings};
use latticefsm::rng::Rng;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn machine(states: usize, alphabet: &str, accepting: &[usize]) -> Machine {
    Machine::over(states, alphabet, accepting, Settings::default()).unwrap()
}

#[test]
fn the_matrix_is_dense_and_three_dimensional() {
    let alphabet: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let lat = Lattice::new(4, &alphabet, Weighting::default()).unwrap();
    assert_eq!(lat.shape(), (4, 3, 4));
    assert_eq!(lat.len(), 48);
    for s in 0..4 {
        for a in 0..3 {
            for t in 0..4 {
                let e = lat.edge(s, a, t);
                assert_eq!((e.source, e.symbol, e.target), (s, a, t));
                assert!(!e.touched());
            }
        }
    }
    assert_eq!(
        lat.row(2, 1).iter().map(|e| e.target).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(lat.leaving(1).len(), 12);
    assert_eq!(lat.arriving(3).count(), 12);
    assert!(lat.arriving(3).all(|e| e.target == 3));
    assert!(lat.offset(4, 0, 0).is_err());
    assert!(lat.symbol_index("z").is_err());
    assert!(Lattice::new(0, &alphabet, Weighting::default()).is_err());
    assert!(Lattice::new(2, &["a".to_string(), "a".to_string()], Weighting::default()).is_err());
}

#[test]
fn every_edge_has_its_own_weighting() {
    let mut m = machine(2, "a", &[]);
    m.edge_mut(0, 0, 0).weighting.bias = 3.0;
    assert_eq!(m.edge(0, 0, 1).weighting.bias, 0.0);
    assert_eq!(m.lattice.prototype.bias, 0.0);
}

#[test]
fn an_edge_traverses_credits_fades_and_clips() {
    let mut e = Edge::new(0, 0, 0, Weighting::default());
    assert_eq!(e.features_at(0, 1000.0), [0.0; 5]);
    assert_eq!(e.log_weight(0, 1000.0, 3.0), 0.0);
    e.traverse(10, 100.0, 1.0, 0.0);
    assert_eq!((e.seen, e.first_seen, e.last_seen), (1, 10, 10));
    assert!(close(e.recent_at(110, 100.0), 0.5));
    assert!(close(e.age_at(110, 100.0), 0.5));
    e.traverse(110, 100.0, 1.0, 0.5);
    assert!(close(e.recent_at(110, 100.0), 1.5));
    assert!(close(e.width_at(110, 100.0), 1.5));
    assert!(close(e.width_at(210, 100.0), 1.25));
    let f = e.features;
    e.credit(1.0, 111, 100.0, 1.0, 0.0);
    assert!(close(e.rewarded, 1.0) && e.last_rewarded == 111 && e.last_punished == -1);
    assert!(close(e.net(), 1.0 / 3.0));
    assert!(close(e.weighting.bias, 0.1));
    assert!(close(e.weighting.seen, 0.1 * f[0]));
    assert!(close(e.weighting.age, 0.1 * f[3]));
    e.credit(-2.0, 112, 100.0, 0.0, 1.0);
    assert!(close(e.punished, 2.0) && e.last_punished == 112);
    assert!(close(e.weighting.bias, -0.1));
    e.set_width(1e9, 0);
    assert_eq!(e.width, WIDTH_MAX);
    e.set_width(0.0, 0);
    assert_eq!(e.width, WIDTH_MIN);
    let mut w = Weighting {
        rate: 100.0,
        ..Weighting::default()
    };
    w.adapt(1.0, &[1.0; 5]);
    assert_eq!(w.bias, COEFFICIENT_LIMIT);
    assert_eq!(w.width, COEFFICIENT_LIMIT);
    let fresh = Edge::new(0, 0, 0, Weighting::default());
    assert_eq!(fresh.width_at(10_000, 100.0), WIDTH_REST);
    let back = Edge::from_json(&parse(&e.to_json().dump()).unwrap()).unwrap();
    assert_eq!(back, e);
}

#[test]
fn a_run_traverses_one_edge_per_symbol_and_a_quiet_run_moves_nothing() {
    let mut m = machine(3, "ab", &[0]);
    let symbols = m.tokenize("abba").unwrap();
    let r = m.run(&symbols, None, None, false);
    assert_eq!(r.transitions.len(), 4);
    assert_eq!(m.clock, 4);
    assert_eq!(m.path.len(), 4);
    assert_eq!(r.states()[0], 0);
    assert_eq!(r.accepted, r.final_state == 0);
    assert_eq!(m.lattice.edges.iter().map(|e| e.seen).sum::<u64>(), 4);
    assert!(close(r.log_probability(), 4.0 * (1.0f64 / 3.0).ln()));
    let before = m.to_json();
    let q = m.run(&symbols, None, None, true);
    assert_eq!(q.transitions.len(), 4);
    let mut after = m.to_json();
    after.set("rng", before.get("rng").unwrap().clone());
    assert_eq!(after, before);
    assert_eq!(m.credit(1.0), 4); // the last traversed run is still the path
    assert!(m.tokenize("abq").is_err());
}

#[test]
fn credit_is_discounted_back_from_the_end() {
    let mut m = Machine::over(
        3,
        "ab",
        &[],
        Settings {
            discount: 0.5,
            ..Settings::default()
        },
    )
    .unwrap();
    let symbols = m.tokenize("aba").unwrap();
    m.run(&symbols, None, None, false);
    let path = m.path.clone();
    assert_eq!(m.reward(2.0), 3);
    assert!(close(m.lattice.edges[path[2]].rewarded, 2.0));
    assert!(close(m.lattice.edges[path[1]].rewarded, 1.0));
    assert!(close(m.lattice.edges[path[0]].rewarded, 0.5));
    assert_eq!(m.punish(1.0), 3);
    assert!(close(m.lattice.edges[path[0]].punished, 0.25));
    assert_eq!(m.credits, 2);
    let mut one = machine(1, "a", &[]);
    let s = one.tokenize("aaa").unwrap();
    one.run(&s, None, None, false);
    one.reward(1.0);
    assert!(close(one.edge(0, 0, 0).rewarded, 1.0 + 0.8 + 0.64));
}

#[test]
fn reward_moves_probability_toward_the_path() {
    let mut m = Machine::over(
        3,
        "a",
        &[],
        Settings {
            seed: 3,
            ..Settings::default()
        },
    )
    .unwrap();
    let r = m.run(&[0], None, None, false);
    let t = r.transitions[0].target;
    let p0 = m.probabilities(0, 0, None, None)[t];
    m.reward(1.0);
    assert!(m.probabilities(0, 0, None, None)[t] > p0);
    m.run(&[0], None, None, false);
    let t2 = m.lattice.edges[m.path[0]].target;
    let before = m.probabilities(0, 0, None, None)[t2];
    m.punish(1.0);
    assert!(m.probabilities(0, 0, None, None)[t2] < before);
}

#[test]
fn teach_is_one_traversal_and_one_credit() {
    let mut m = machine(3, "ab", &[]);
    let e = m.teach(1, 1, 2, -0.5).unwrap();
    assert_eq!((e.seen, e.punished), (1, 0.5));
    assert_eq!(m.clock, 1);
    assert_eq!(m.state, 0);
    assert!(m.teach(5, 0, 0, 1.0).is_err());
}

#[test]
fn stimulation_prefers_the_wide_channel_and_relaxes_on_the_clock() {
    let mut m = Machine::over(
        3,
        "a",
        &[],
        Settings {
            life: 100.0,
            ..Settings::default()
        },
    )
    .unwrap();
    m.edge_mut(0, 0, 0).set_width(4.0, 0);
    m.edge_mut(0, 0, 2).set_width(0.25, 0);
    let wide: Vec<f64> = [0.0, 0.5, 1.0, 2.0, 4.0]
        .iter()
        .map(|&s| m.probabilities(0, 0, Some(s), None)[0])
        .collect();
    assert!(close(wide[0], 1.0 / 3.0));
    assert!(wide.windows(2).all(|w| w[0] < w[1]));
    assert!(close(m.probabilities(0, 0, Some(1.0), None)[0], 4.0 / 5.25));
    assert!(wide[4] > 0.99);
    assert_eq!(m.stimulation(), 1.0);
    assert_eq!(m.stimulate(3.0), 4.0);
    m.tick(100);
    assert!(close(m.stimulation(), 2.5));
    m.stimulate(-10.0);
    assert_eq!(m.stimulation(), 0.0);
    assert!(m.set_stimulation(-1.0).is_err());
    let mut calm = Machine::over(
        2,
        "a",
        &[],
        Settings {
            life: 100.0,
            calm: Some(10.0),
            ..Settings::default()
        },
    )
    .unwrap();
    calm.stimulate(1.0);
    calm.tick(10);
    assert!(close(calm.stimulation(), 1.5));
    let r = m.run(&[0], Some(5.0), Some(0.0), false);
    assert_eq!(r.transitions[0].target, 0);
    assert_eq!(r.transitions[0].stimulation, 5.0);
}

#[test]
fn a_custom_weight_function_replaces_the_edges_own() {
    let mut m = machine(3, "a", &[]);
    m.weight_fn = Some(Box::new(
        |e: &Edge, _clock: i64, _life: f64, _stim: f64| if e.target == 1 { 10.0 } else { 0.0 },
    ));
    m.edge_mut(0, 0, 0).weighting.bias = 100.0;
    assert!(m.probabilities(0, 0, None, None)[1] > 0.99);
}

#[test]
fn time_fades_traces_and_widths_but_not_verdicts() {
    let mut m = Machine::over(
        2,
        "a",
        &[],
        Settings {
            life: 100.0,
            ..Settings::default()
        },
    )
    .unwrap();
    m.teach(0, 0, 1, 1.0).unwrap();
    let e = m.edge(0, 0, 1).clone();
    assert_eq!(e.recent_at(e.last_seen, 100.0), 1.0);
    let w = e.width_at(m.clock, 100.0);
    assert!(w > 1.0);
    m.tick(1000);
    assert!(e.recent_at(m.clock, 100.0) < 0.01);
    assert!(e.width_at(m.clock, 100.0) - 1.0 < (w - 1.0) * 0.001);
    assert_eq!((e.rewarded, e.seen), (1.0, 1));
}

#[test]
fn invalid_settings_are_refused() {
    assert!(Machine::over(
        2,
        "a",
        &[],
        Settings {
            life: 0.0,
            ..Settings::default()
        }
    )
    .is_err());
    assert!(Machine::over(
        2,
        "a",
        &[],
        Settings {
            baseline: -1.0,
            ..Settings::default()
        }
    )
    .is_err());
    assert!(Machine::over(
        2,
        "a",
        &[],
        Settings {
            discount: 1.5,
            ..Settings::default()
        }
    )
    .is_err());
    assert!(Machine::over(
        2,
        "a",
        &[],
        Settings {
            temperature: -1.0,
            ..Settings::default()
        }
    )
    .is_err());
    assert!(Machine::over(
        2,
        "a",
        &[],
        Settings {
            start: 5,
            ..Settings::default()
        }
    )
    .is_err());
    assert!(Machine::over(2, "a", &[7], Settings::default()).is_err());
}

#[test]
fn a_language_is_learned_by_credit_and_not_quietly() {
    let lang = language("even-b").unwrap();
    for quiet in [false, true] {
        let mut rng = Rng::new(1);
        let mut m = Machine::over(3, "ab", lang.accepting, Settings::default()).unwrap();
        let test = examples(&lang, 200, &mut rng, 6);
        teach_language(&mut m, &lang, 1500, &mut rng, 6, quiet, 1.0, None, 0);
        let acc = m.accuracy(&test, 0.0);
        if quiet {
            assert!(acc < 0.75, "{acc}");
            assert_eq!(m.stats().num("touched", 1.0), 0.0);
        } else {
            assert!(acc > 0.95, "{acc}");
            let t = m.transition_table();
            assert_eq!(t[0][0], 0);
            assert_ne!(t[0][1], 0);
        }
    }
}

#[test]
fn the_experiments_say_what_the_readme_says() {
    let stim = experiment::stimulation(&StimulationOptions::default());
    let curve: Vec<f64> = stim
        .get("curve")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.get("probabilities").unwrap().as_array().unwrap()[0].as_f64().unwrap())
        .collect();
    assert!(close(curve[0], 1.0 / 3.0));
    assert!(curve.windows(2).all(|w| w[0] < w[1]));
    let adapt = experiment::adaptation(5, 100.0, &[0.0, 1.0]);
    let stages = adapt.get("stages").unwrap().as_array().unwrap();
    let last = stages.last().unwrap().get("edges").unwrap();
    assert!(last.get("rewarded").unwrap().num("net", 0.0) > 0.0);
    assert!(last.get("punished").unwrap().num("net", 0.0) < 0.0);
    assert_ne!(
        last.get("rewarded").unwrap().get("weighting"),
        last.get("punished").unwrap().get("weighting")
    );
    assert!(experiment::tables(&adapt).contains("| stage |"));
    let learn = experiment::learning(&LearningOptions {
        episodes: 300,
        sizes: vec![3],
        seeds: vec![1],
        tests: 50,
        ..LearningOptions::default()
    });
    assert_eq!(learn.get("rows").unwrap().as_array().unwrap().len(), 4);
    assert!(experiment::tables(&learn).contains("| language |"));
}

#[test]
fn a_machine_round_trips_through_its_file() {
    let lang = language("ends-ab").unwrap();
    let mut m = Machine::over(
        3,
        "ab",
        lang.accepting,
        Settings {
            life: 500.0,
            seed: 7,
            ..Settings::default()
        },
    )
    .unwrap();
    let mut rng = Rng::new(7);
    teach_language(&mut m, &lang, 200, &mut rng, 6, false, 1.0, None, 0);
    m.stimulate(2.0);
    m.tick(50);
    let dir = std::env::temp_dir().join(format!("latticefsm-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["m.json", "m.json.gz"] {
        let path = dir.join(name).to_string_lossy().to_string();
        m.save(&path).unwrap();
        let mut back = load_machine(&path).unwrap();
        assert_eq!(back.to_json(), m.to_json());
        assert_eq!(back.stats(), m.stats());
        assert_eq!(back.transition_table(), m.transition_table());
        let s = m.tokenize("abab").unwrap();
        assert_eq!(
            back.run(&s, None, None, false).states(),
            m.run(&s, None, None, false).states()
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let fresh = machine(5, "abc", &[]);
    assert_eq!(
        fresh
            .to_json()
            .get("lattice")
            .unwrap()
            .get("edges")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        0
    );
    let mut foreign = fresh.to_json();
    foreign.set("rng", parse(r#"{"mersenne":[3,[1,2,3],null]}"#).unwrap());
    let back = Machine::from_json(&foreign).unwrap();
    assert_eq!(back.rng, Rng::new(1));
}
