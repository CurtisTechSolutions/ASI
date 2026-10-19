//! Credit by an oracle: every step judged by the right answer for that step alone, rewarded when right and - unless
//! only rewards are given - punished when wrong.  The same cases as `TestOracleCredit` in the Python tests.

use latticefsm::experiment::{oracle_amounts, teach_language_by, Lesson};
use latticefsm::languages::{examples, language};
use latticefsm::machine::{Machine, Settings};
use latticefsm::rng::Rng;

#[test]
fn each_step_gets_its_own_credit() {
    let settings = Settings {
        temperature: 0.0,
        ..Settings::default()
    };
    let mut m = Machine::over(3, "ab", &[0], settings).unwrap();
    let symbols = m.tokenize("aba").unwrap();
    m.run(&symbols, None, None, false);
    let path = m.path.clone();
    assert_eq!(m.credit_steps(&[1.0, -1.0, 0.0]), 2);
    let e = |i: usize| &m.lattice.edges[path[i]];
    assert_eq!((e(0).rewarded, e(0).punished), (1.0, 0.0));
    assert_eq!((e(1).rewarded, e(1).punished), (0.0, 1.0));
    assert_eq!(
        (e(2).rewarded, e(2).punished),
        (0.0, 0.0),
        "nothing for the step given nothing"
    );
}

#[test]
fn the_oracle_judges_every_prefix() {
    // even-b: accepting at 0 means "an even number of b's so far"
    let lang = language("even-b").unwrap();
    let settings = Settings {
        temperature: 0.0,
        ..Settings::default()
    };
    let mut m = Machine::over(2, "ab", lang.accepting, settings).unwrap();
    // teach the right machine outright: a keeps the state, b flips it
    for (s, a, t) in [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)] {
        m.teach(s, a, t, 5.0).unwrap();
    }
    let s = "abba";
    let run = m.run(&m.tokenize(s).unwrap(), None, None, true);
    let all = Lesson {
        oracle: true,
        ..Lesson::default()
    };
    assert_eq!(oracle_amounts(&m, &lang, s, &run, &all), vec![1.0; 4]);
    // the wrong machine: every step lands in state 1, right only where the prefix has an odd number of b's
    let mut wrong = Machine::over(
        2,
        "ab",
        lang.accepting,
        Settings {
            temperature: 0.0,
            ..Settings::default()
        },
    )
    .unwrap();
    for (s, a) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        wrong.teach(s, a, 1, 5.0).unwrap();
    }
    let run = wrong.run(&wrong.tokenize(s).unwrap(), None, None, true);
    // prefixes a, ab, abb, abba: b-counts 0, 1, 2, 2 - state 1 (not accepting) is right only for "ab"
    assert_eq!(
        oracle_amounts(&wrong, &lang, s, &run, &all),
        vec![-1.0, 1.0, -1.0, -1.0]
    );
    let rewards_only = Lesson {
        oracle: true,
        punish: false,
        ..Lesson::default()
    };
    assert_eq!(
        oracle_amounts(&wrong, &lang, s, &run, &rewards_only),
        vec![0.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn it_learns_a_language_by_the_oracle() {
    let lang = language("even-b").unwrap();
    let test = examples(&lang, 300, &mut Rng::new(7), 6);
    let mut m = Machine::over(6, "ab", lang.accepting, Settings::default()).unwrap();
    let lesson = Lesson {
        oracle: true,
        ..Lesson::default()
    };
    teach_language_by(&mut m, &lang, 4000, &mut Rng::new(1), 6, lesson, None, 0);
    assert!(m.accuracy(&test, 0.0) >= 0.95);
}
