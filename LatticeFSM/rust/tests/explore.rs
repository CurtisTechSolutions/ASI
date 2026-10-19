//! The explore traversal mode: the least traversed and most rewarded edge of a row first.  The same cases as
//! `TestExploreMode` in the Python tests.

use latticefsm::edge::{Edge, Weighting, EXPLORE};
use latticefsm::experiment::teach_language;
use latticefsm::languages::{examples, language};
use latticefsm::machine::{Machine, Mode, Settings, Walk};
use latticefsm::rng::Rng;

fn explore(c: f64) -> Settings {
    Settings {
        mode: Mode::Explore,
        explore: c,
        ..Settings::default()
    }
}

#[test]
fn an_edges_explore_weight() {
    let mut e = Edge::new(0, 0, 1, Weighting::default());
    // nothing traversed in the row: no bonus, and nothing earned
    assert_eq!(e.explore_log_weight(0, EXPLORE), 0.0);
    (e.seen, e.rewarded, e.punished) = (4, 3.0, 1.0);
    let want = 2.0 + EXPLORE * (11f64.ln() / 5.0).sqrt();
    assert!((e.explore_log_weight(10, EXPLORE) - want).abs() < 1e-12);
    assert_eq!(e.explore_log_weight(10, 0.0), 2.0);
    // the less traversed of two equally rewarded edges weighs more
    let mut f = Edge::new(0, 0, 2, Weighting::default());
    (f.seen, f.rewarded, f.punished) = (1, 3.0, 1.0);
    assert!(f.explore_log_weight(10, EXPLORE) > e.explore_log_weight(10, EXPLORE));
}

#[test]
fn the_machine_weighs_its_rows_so() {
    let mut m = Machine::over(3, "ab", &[0], explore(EXPLORE)).unwrap();
    assert_eq!(m.mode, Mode::Explore);
    let w = m.log_weights(0, 0, None);
    assert!(w.iter().all(|&x| x == w[0]), "untried: every edge alike");
    m.teach(0, 0, 1, 2.0).unwrap(); // rewarded
    m.teach(0, 0, 2, -1.0).unwrap(); // punished
    let w = m.log_weights(0, 0, None);
    assert!(
        w[1] > w[0] && w[0] > w[2],
        "rewarded beats untried beats punished: {w:?}"
    );
    // with no bonus the untried edge has only its (zero) earnings
    m.explore = 0.0;
    assert_eq!(m.log_weights(0, 0, None), vec![0.0, 2.0, -1.0]);
    // the own mode weighs the same edges by their own functions
    m.mode = Mode::Own;
    let own: Vec<f64> = m
        .lattice
        .row(0, 0)
        .iter()
        .map(|e| e.log_weight(m.clock, m.life, m.stimulation()))
        .collect();
    assert_eq!(m.log_weights(0, 0, None), own);
    assert!(Mode::parse("wander").is_err());
    assert!(Machine::over(3, "ab", &[0], explore(-1.0)).is_err());
}

#[test]
fn a_run_can_take_a_mode_of_its_own() {
    let mut m = Machine::over(3, "ab", &[0], Settings::default()).unwrap();
    m.teach(0, 0, 2, 3.0).unwrap();
    let walk = Walk {
        temperature: Some(0.0),
        quiet: true,
        mode: Some(Mode::Explore),
        ..Walk::default()
    };
    assert_eq!(m.walk(&[0], walk).unwrap().final_state, 2);
    assert_eq!(m.mode, Mode::Own, "the machine's own mode is back");
}

#[test]
fn the_mode_is_saved() {
    let m = Machine::over(3, "ab", &[0], explore(0.5)).unwrap();
    let mut j = m.to_json();
    let back = Machine::from_json(&j).unwrap();
    assert_eq!((back.mode, back.explore), (Mode::Explore, 0.5));
    // a file from before the mode reads as the own mode
    if let latticefsm::json::Json::Object(items) = &mut j {
        for (k, v) in items.iter_mut() {
            if k == "settings" {
                if let latticefsm::json::Json::Object(st) = v {
                    st.retain(|(k, _)| k != "mode" && k != "explore");
                }
            }
        }
    }
    let old = Machine::from_json(&j).unwrap();
    assert_eq!((old.mode, old.explore), (Mode::Own, EXPLORE));
}

#[test]
fn it_learns_a_language() {
    let lang = language("even-b").unwrap();
    let test = examples(&lang, 300, &mut Rng::new(7), 6);
    let mut m = Machine::over(6, "ab", lang.accepting, explore(EXPLORE)).unwrap();
    teach_language(&mut m, &lang, 4000, &mut Rng::new(1), 6, false, 1.0, None, 0);
    assert!(m.accuracy(&test, 0.0) >= 0.95);
}

#[test]
fn a_reward_can_be_the_whole_paths() {
    for equal in [false, true] {
        let settings = Settings {
            equal_reward: equal,
            temperature: 0.0,
            ..Settings::default()
        };
        let mut m = Machine::over(3, "ab", &[0], settings).unwrap();
        let symbols = m.tokenize("aba").unwrap();
        m.run(&symbols, None, None, false);
        let path = m.path.clone();
        assert_eq!(path.len(), 3);
        m.reward(1.0);
        let got: Vec<f64> = path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
        let want = if equal {
            vec![1.0, 1.0, 1.0]
        } else {
            vec![0.8 * 0.8, 0.8, 1.0]
        };
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "equal {equal}: {got:?}");
        }
        // punishment is discounted back from the end either way
        m.run(&symbols, None, None, false);
        let path = m.path.clone();
        let before: Vec<f64> = path.iter().map(|&i| m.lattice.edges[i].punished).collect();
        m.punish(1.0);
        let got: Vec<f64> = path
            .iter()
            .zip(&before)
            .map(|(&i, b)| m.lattice.edges[i].punished - b)
            .collect();
        for (g, w) in got.iter().zip(&[0.64, 0.8, 1.0]) {
            assert!((g - w).abs() < 1e-12, "equal {equal}: {got:?}");
        }
    }
    let m = Machine::over(
        3,
        "ab",
        &[0],
        Settings {
            equal_reward: true,
            ..Settings::default()
        },
    )
    .unwrap();
    assert!(Machine::from_json(&m.to_json()).unwrap().equal_reward);
}
