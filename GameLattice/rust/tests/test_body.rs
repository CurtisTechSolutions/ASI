//! The design choices, as tests (`DESIGN.md` §15).
#![allow(clippy::field_reassign_with_default)]

use gamelattice::body::{Body, BodySettings};
use gamelattice::credit::{self, CreditSettings};
use gamelattice::decoders::readout::VerdictReadout;
use gamelattice::decoders::{check, Decoder};
use gamelattice::efference::{Efference, Return};
use gamelattice::forward;
use gamelattice::lattice::{GameLattice, Kind};
use gamelattice::select::{Release, Selection};
use gamelattice::verdict::{Source, Verdict};
use latticefsm::languages::{examples, language};
use latticefsm::machine::Settings;
use latticefsm::rng::Rng;

fn lattice(states: usize) -> GameLattice {
    let alphabet: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    GameLattice::new("t", Kind::Act, states, &alphabet, &[0], Settings::default()).unwrap()
}

#[test]
fn classify_is_quiet_and_deterministic() {
    let lang = language("even-b").unwrap();
    let mut body = Body::for_language(&lang, BodySettings::default()).unwrap();
    let mut rng = Rng::new(1);
    body.teach(200, &mut rng, 6);
    let before = body.lattice.machine.to_json().dump();
    let namings = body.encoder.namings;
    let a = body.classify("abab");
    let b = body.classify("abab");
    assert_eq!(a, b);
    assert_eq!(body.lattice.machine.to_json().dump(), before, "nothing moved");
    assert_eq!(body.encoder.namings, namings);
}

#[test]
fn the_plain_control_and_the_game_both_learn_even_b() {
    let lang = language("even-b").unwrap();
    let mut rng = Rng::new(5);
    let test = examples(&lang, 100, &mut rng, 6);
    for plain in [true, false] {
        let mut s = BodySettings::default();
        s.encoder_sets_state = !plain;
        s.seed = 1;
        let mut body = Body::for_language(&lang, s).unwrap();
        body.teach(3000, &mut rng, 6);
        let (enc, _) = body.accuracy(&test);
        assert!(enc > 0.9, "plain {plain}: {enc}");
    }
}

#[test]
fn the_encoder_does_not_collapse_to_one_state() {
    let lang = language("mod3-a").unwrap();
    let mut body = Body::for_language(&lang, BodySettings::default()).unwrap();
    let mut rng = Rng::new(9);
    body.teach(800, &mut rng, 6);
    assert!(body.encoder.diversity() > 1.0, "diversity {}", body.encoder.diversity());
}

#[test]
fn a_forward_lesson_moves_the_row_and_agreement_moves_nothing() {
    let mut l = lattice(5);
    let m = &mut l.machine;
    let (t, _) = forward::predict(m, 0, 0);
    let before = m.to_json().dump();
    assert!(!forward::learn(m, 0, 0, t, t, 1.0).unwrap(), "agreement is no lesson");
    assert_eq!(m.to_json().dump(), before);
    let actual = (t + 1) % 5;
    assert!(forward::learn(m, 0, 0, t, actual, 1.0).unwrap());
    let (t2, _) = forward::predict(m, 0, 0);
    assert_eq!(t2, actual, "the lesson taught the actual target");
}

#[test]
fn reafference_is_credited_exactly_zero() {
    let mut l = lattice(4);
    let m = &mut l.machine;
    let symbols = m.tokenize("abab").unwrap();
    m.run(&symbols, None, None, false);
    let verdicts: Vec<Verdict> = (0..4)
        .map(|i| Verdict::sharp(1.0, "right", Source::Oracle, i))
        .collect();
    let returns = vec![
        Return::Reafferent,
        Return::Exafferent,
        Return::Reafferent,
        Return::Exafferent,
    ];
    let before: Vec<f64> = m.path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
    credit::land(m, &verdicts, &returns, CreditSettings::default());
    let after: Vec<f64> = m.path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
    assert_eq!(after[0], before[0], "a reafferent step earns nothing");
    assert_eq!(after[2], before[2]);
    assert!(
        after[1] > before[1] && after[3] > before[3],
        "an exafferent step is credited"
    );
}

#[test]
fn efference_sorts_what_returns() {
    let mut e = Efference::default();
    e.copy("x", "ear", 0, 2, None);
    let (r, _) = e.observe("x", "ear", |a, b| a == b);
    assert_eq!(r, Return::Reafferent);
    let (r, _) = e.observe("y", "ear", |a, b| a == b);
    assert_eq!(r, Return::Exafferent, "no copy: the world spoke");
    e.copy("x", "ear", 0, 2, None);
    let (r, _) = e.observe("z", "ear", |a, b| a == b);
    assert_eq!(r, Return::Error);
    e.copy("x", "ear", 10, 2, None);
    assert_eq!(e.expire(20), 1, "a copy whose sensation never came is an error");
}

#[test]
fn a_sharp_verdict_lands_on_one_edge_and_a_dull_one_never_crosses_the_run() {
    let mut l = lattice(4);
    let m = &mut l.machine;
    let symbols = m.tokenize("aab").unwrap();
    m.run(&symbols, None, None, false);
    let before: Vec<f64> = m.path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
    credit::land(
        m,
        &[Verdict::sharp(1.0, "right", Source::Oracle, 1)],
        &[],
        CreditSettings::default(),
    );
    let after: Vec<f64> = m.path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
    assert_eq!(after[0], before[0]);
    assert_eq!(after[2], before[2]);
    assert!(
        (after[1] - before[1] - 1.0).abs() < 1e-12,
        "in full, on that edge alone"
    );
    // a baseline of 1 subtracts the edge's own expectation
    let settings = CreditSettings { baseline: 1.0 };
    let net = m.lattice.edges[m.path[1]].net();
    credit::land(m, &[Verdict::sharp(net, "right", Source::Oracle, 1)], &[], settings);
    let again: Vec<f64> = m.path.iter().map(|&i| m.lattice.edges[i].rewarded).collect();
    assert!(
        (again[1] - after[1]).abs() < 1e-12,
        "a perfectly predicted verdict moves the edge by zero"
    );
}

#[test]
fn a_decoder_must_name_the_sense_that_reads_it_back() {
    let good = VerdictReadout {
        sense: "text".to_string(),
    };
    assert!(check(&good).is_ok());
    let bad = VerdictReadout { sense: " ".to_string() };
    assert!(check(&bad).is_err());
    assert_eq!(good.output_type(), "verdict");
}

#[test]
fn the_release_rule_abstains_on_a_photo_finish_and_below_the_reserve() {
    let r = Release::default();
    assert_eq!(r.select(&[0.5, 0.49, 0.01]), Selection::PhotoFinish);
    assert_eq!(r.select(&[0.005, 0.005]), Selection::Abstain);
    assert_eq!(r.select(&[0.1, 0.8, 0.1]), Selection::Released(1));
}

#[test]
fn a_topographic_layer_keeps_its_symbol_axis() {
    let mut l = lattice(6);
    l.topographic = true;
    assert!(l.swap_symbols(0, 1).is_err());
    l.topographic = false;
    assert!(l.swap_symbols(0, 1).is_ok());
}
