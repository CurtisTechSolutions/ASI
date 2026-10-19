//! Where a run starts and how it moves: focus on the central vertical vector, the learned focus, skips, swaps and
//! the nodes rearranging themselves.

use latticefsm::compress::compress;
use latticefsm::experiment::teach_language;
use latticefsm::focus::FocusLearner;
use latticefsm::geometry::{central_vertical, check_focus, focus_index, focus_node};
use latticefsm::json::parse;
use latticefsm::languages::language;
use latticefsm::machine::{Machine, Settings, Walk};
use latticefsm::rng::Rng;

fn taught(episodes: usize, settings: Settings) -> Machine {
    let lang = language("even-b").unwrap();
    let mut m = Machine::default_shape(lang.accepting, settings).unwrap();
    teach_language(&mut m, &lang, episodes, &mut Rng::new(2), 6, false, 1.0, None, 0);
    m
}

fn edges(m: &Machine) -> Vec<String> {
    m.lattice.edges.iter().map(|e| e.to_json().dump()).collect()
}

#[test]
fn focus_picks_a_node_of_the_central_vertical_vector() {
    let shape = (13, 13, 13);
    assert_eq!(central_vertical(shape), (0..13).map(|s| (s, 6, 6)).collect::<Vec<_>>());
    for (f, k) in [
        (None, 0),
        (Some(0.0), 0),
        (Some(0.07), 0),
        (Some(1.0 / 13.0), 1),
        (Some(0.5), 6),
        (Some(0.92), 11),
        (Some(12.0 / 13.0), 12),
        (Some(1.0), 12),
    ] {
        assert_eq!(focus_index(13, f), k, "{f:?}");
        assert_eq!(focus_node(shape, f), (k, 6, 6));
    }
    for bad in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
        assert!(check_focus(bad).is_err(), "{bad}");
    }
}

#[test]
fn runs_start_from_the_focus_node() {
    let mut m = Machine::default_shape(&[0], Settings::default()).unwrap();
    let s = m.tokenize("ab").unwrap();
    assert_eq!(m.run(&s, None, None, false).states()[0], 0);
    m.set_focus(Some(1.0)).unwrap();
    assert_eq!(m.origin(), 12);
    assert_eq!(m.run(&s, None, None, false).states()[0], 12);
    assert_eq!(m.run(&s, None, None, true).states()[0], 12);
    let r = m
        .walk(
            &s,
            Walk {
                focus: Some(0.5),
                ..Walk::default()
            },
        )
        .unwrap();
    assert_eq!(r.states()[0], 6);
    assert!(m
        .walk(
            &s,
            Walk {
                focus: Some(3.0),
                ..Walk::default()
            }
        )
        .is_err());
    assert!(m.set_focus(Some(-1.0)).is_err());
    assert_eq!(m.stats().get("focus_node").unwrap().dump(), "[12,6,6]");
    let back = Machine::from_json(&parse(&m.to_json().dump()).unwrap()).unwrap();
    assert_eq!(back.focus(), Some(1.0));
    assert!(Machine::default_shape(
        &[0],
        Settings {
            focus: Some(2.0),
            ..Settings::default()
        }
    )
    .is_err());
}

#[test]
fn a_learned_focus_reads_the_input_and_learns_from_credit() {
    let mut m = Machine::default_shape(
        &[0],
        Settings {
            learn_focus: true,
            ..Settings::default()
        },
    )
    .unwrap();
    let s = m.tokenize("abba").unwrap();
    assert_eq!(m.focus_learner.features(&s).len(), 2 + 3 * 13);
    assert!((m.focus_learner.predict(&s) - 1.0 / (1.0 + 3f64.exp())).abs() < 1e-12);
    assert_eq!(m.run(&s, None, None, true).states()[0], 0);
    let before = m.focus_learner.weights.clone();
    m.run(&s, None, None, false);
    assert!(m.last_focus.is_some());
    m.reward(1.0);
    assert_ne!(m.focus_learner.weights, before);
    for (credit, up) in [(1.0, true), (-1.0, false)] {
        let mut fl = FocusLearner::new(vec!["a".into(), "b".into()], 0.1, 0.2);
        let x = fl.features(&[0, 1]);
        let mu = fl.mean(&x);
        fl.learn(credit, &(mu + 0.3, mu, x.clone()));
        assert_eq!(fl.mean(&x) > mu, up);
    }
    let trained = taught(
        600,
        Settings {
            learn_focus: true,
            ..Settings::default()
        },
    );
    assert!(trained.focus_learner.weights.iter().all(|w| (-8.0..=8.0).contains(w)));
    let back = Machine::from_json(&parse(&trained.to_json().dump()).unwrap()).unwrap();
    assert_eq!(back.focus_learner, trained.focus_learner);
    assert!(back.learn_focus);
}

#[test]
fn a_skip_takes_the_more_efficient_two_edge_path() {
    let mut m = Machine::over(
        3,
        "ab",
        &[],
        Settings {
            skip: true,
            temperature: 0.0,
            ..Settings::default()
        },
    )
    .unwrap();
    m.edge_mut(0, 0, 1).weighting.bias = 1.0;
    m.edge_mut(0, 0, 2).weighting.bias = 0.9;
    m.edge_mut(2, 1, 0).weighting.bias = 6.0;
    let s = m.tokenize("ab").unwrap();
    let plain = m
        .walk(
            &s,
            Walk {
                quiet: true,
                skip: Some(false),
                ..Walk::default()
            },
        )
        .unwrap();
    assert_eq!(plain.transitions.len(), 2);
    assert_eq!(plain.transitions[0].target, 1);
    let r = m.run(&s, None, None, false);
    assert_eq!(r.transitions.len(), 1);
    let t = &r.transitions[0];
    assert_eq!(
        (t.source, t.symbol.as_str(), t.target, t.skipped),
        (0, "ab", 0, Some(2))
    );
    let path: Vec<(usize, usize)> = m
        .path
        .iter()
        .map(|&o| (m.lattice.edges[o].source, m.lattice.edges[o].target))
        .collect();
    assert_eq!(path, vec![(0, 2), (2, 0)]);
    assert_eq!(m.lattice.states[2].visits, 0);
    assert_eq!((m.skips, m.clock), (1, 2));
    assert_eq!(m.reward(1.0), 2);
    m.skip_margin = 100.0;
    assert!(m.run(&s, None, None, true).transitions[0].skipped.is_none());
}

#[test]
fn without_skips_nothing_changes_and_skips_are_saved() {
    assert_eq!(
        edges(&taught(500, Settings::default())),
        edges(&taught(
            500,
            Settings {
                skip: false,
                ..Settings::default()
            }
        ))
    );
    let m = taught(
        600,
        Settings {
            skip: true,
            ..Settings::default()
        },
    );
    assert!(m.skips > 0);
    let back = Machine::from_json(&parse(&m.to_json().dump()).unwrap()).unwrap();
    assert_eq!((back.skip, back.skips), (true, m.skips));
    let core = compress(&m, None, None);
    let mut q = taught(
        600,
        Settings {
            skip: true,
            ..Settings::default()
        },
    );
    for text in ["abba", "aab", "bbbb"] {
        let s = q.tokenize(text).unwrap();
        let r = q.run(&s, None, Some(0.0), true);
        assert_eq!(core.run(text, false).unwrap().0, r.states(), "{text}");
    }
}

#[test]
fn a_swap_is_a_relabelling_and_rearranging_moves_the_busy_nodes_inward() {
    let mut m = taught(800, Settings::default());
    let before: Vec<f64> = m
        .lattice
        .edges
        .iter()
        .map(|e| e.log_weight(m.clock, m.life, 1.0))
        .collect();
    let old = m.lattice.clone();
    m.swap_states(0, 12).unwrap();
    m.swap_symbols(0, 6).unwrap();
    assert_eq!((m.lattice.state_ids[0], m.lattice.state_ids[12], m.start), (12, 0, 12));
    assert_eq!((m.alphabet()[0].as_str(), m.alphabet()[6].as_str()), ("g", "a"));
    let ps = |x: usize| {
        if x == 0 {
            12
        } else if x == 12 {
            0
        } else {
            x
        }
    };
    let pa = |x: usize| {
        if x == 0 {
            6
        } else if x == 6 {
            0
        } else {
            x
        }
    };
    for (i, e) in old.edges.iter().enumerate() {
        let moved = m.lattice.edge(ps(e.source), pa(e.symbol), ps(e.target));
        assert_eq!(moved.log_weight(m.clock, m.life, 1.0), before[i]);
    }
    let mut m = taught(800, Settings::default());
    assert!(!m.rearrange(true).is_empty());
    assert!(m.rearrange(true).is_empty());
    for load in [m.lattice.state_load(), m.lattice.symbol_load()] {
        let c = load.len() / 2;
        assert!((0..c).all(|k| load[k] <= load[k + 1]), "{load:?}");
        assert!((c..load.len() - 1).all(|k| load[k] >= load[k + 1]), "{load:?}");
    }
    let table = compress(&m, None, None).shell_table();
    let inner: f64 = table.as_array().unwrap()[..5]
        .iter()
        .map(|r| r.num("touched", 0.0))
        .sum();
    assert!(inner > 0.0);
    assert_eq!(edges(&compress(&m, None, None).decompress(None).unwrap()), edges(&m));
}

#[test]
fn a_rearrangement_waits_for_the_end_of_a_run() {
    let mut m = Machine::default_shape(
        &[0],
        Settings {
            rearrange_every: 3,
            skip: true,
            ..Settings::default()
        },
    )
    .unwrap();
    for _ in 0..30 {
        let s = m.tokenize("abababab").unwrap();
        let r = m.run(&s, None, None, false);
        // every edge on the path is where its coordinates say it is
        for &o in &m.path {
            let e = &m.lattice.edges[o];
            assert_eq!(m.lattice.offset(e.source, e.symbol, e.target).unwrap(), o);
        }
        if r.accepted {
            m.reward(1.0);
        } else {
            m.punish(1.0);
        }
    }
    assert!(m.swaps > 0);
    let back = Machine::from_json(&parse(&m.to_json().dump()).unwrap()).unwrap();
    assert_eq!(
        (back.rearrange_every, back.swaps, back.lattice.state_ids.clone()),
        (3, m.swaps, m.lattice.state_ids.clone())
    );
    // a lesson that sets a rearrangement off returns the taught edge where it now sits
    m.since_rearrange = 2;
    let seen_before = m.lattice.edges.iter().map(|e| e.seen).sum::<u64>();
    let e = m.teach(0, 0, 1, 1.0).unwrap().clone();
    assert_eq!(m.lattice.edge(e.source, e.symbol, e.target), &e);
    assert_eq!(m.lattice.edges.iter().map(|e| e.seen).sum::<u64>(), seen_before + 1);
}
