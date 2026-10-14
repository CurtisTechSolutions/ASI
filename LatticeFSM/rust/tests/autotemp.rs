//! Auto-temperature: a reply that is no answer - nothing, no words, a loop - is drawn again a little hotter until it
//! is one.  The same cases as `TestAutoTemperature` in the Python tests.

use latticefsm::autotemp::{
    attempts, climb, climb_note, ladder, loop_period, repeats_context, why_draw, why_no_answer,
};

fn chars(s: &str) -> Vec<String> {
    s.chars().map(String::from).collect()
}

#[test]
fn what_is_no_answer() {
    let ab = "ab".repeat(10);
    let cases: [(&str, Option<&str>); 13] = [
        ("", Some("nothing")),
        ("   ", Some("nothing")),
        (" . ", Some("no words")),
        ("............", Some("no words")),
        ("?!", Some("no words")),
        (
            "is a directed cycle is recorded in a brain is a directed cycle is recorded in a",
            Some("a loop"),
        ),
        (
            "the context would be surprised the context would be surprised the context wo",
            Some("a loop"),
        ),
        (&ab, Some("a loop")),
        ("the cat sat on the mat", None),
        ("hello world", None),
        ("the brain is a directed cyclic graph can be.", None),
        ("a b c d e f g h i j k l m n o p", None),
        ("x", None),
    ];
    for (text, why) in cases {
        assert_eq!(why_no_answer(text), why, "{text}");
    }
    assert_eq!(loop_period(&ab), Some(2));
    assert_eq!(
        loop_period("is a directed cycle is recorded in a brain is a directed cycle is recorded in a"),
        Some(43)
    );
    assert_eq!(loop_period("the cat sat on the mat"), None);
}

#[test]
fn a_greedy_loop_is_a_context_come_back() {
    assert!(repeats_context(&chars("abcabc"), 0, 2));
    // the same context in the prompt is not the model's doing: what it drew after it may differ
    assert!(!repeats_context(&chars("abcabc"), 3, 2));
    assert!(!repeats_context(&chars("abcd"), 0, 2));
    // contexts still reaching back to <s> are their own; with no context at all, two draws are a loop
    assert!(!repeats_context(&chars("ab"), 0, 5));
    assert!(repeats_context(&chars("aa"), 0, 0));
    // greedy: the context decides, not the text; sampled: the text
    assert_eq!(why_draw("bcabc", 0.0, &chars("abcabc"), 1, 2), Some("a loop"));
    assert_eq!(why_draw("bcabc", 0.5, &chars("abcabc"), 1, 2), None);
    assert_eq!(why_draw("", 0.0, &chars("a"), 1, 2), Some("nothing"));
}

#[test]
fn the_ladder() {
    assert_eq!(ladder(0.0, 0.8), vec![0.0, 0.2, 0.4, 0.6, 0.8]);
    assert_eq!(ladder(0.5, 0.8), vec![0.5, 0.7]);
    assert_eq!(ladder(1.0, 0.8), vec![1.0]);
    assert_eq!(ladder(-1.0, 0.2), vec![0.0, 0.2]);
    assert_eq!(
        attempts(0.0, 0.4, 2),
        vec![(0.0, 0), (0.2, 7919), (0.2, 15838), (0.4, 23757), (0.4, 31676)]
    );
}

#[test]
fn climbing_until_an_answer() {
    let mut drawn = Vec::new();
    let c = climb(
        &[],
        1,
        0.0,
        0.8,
        2,
        10,
        |t, seed| {
            drawn.push((t, seed));
            let text = if t >= 0.4 {
                "an answer"
            } else if t == 0.0 {
                ""
            } else {
                "."
            };
            (chars(text), text.to_string(), text)
        },
        |_, _| None,
    );
    assert!(c.found);
    assert_eq!((c.pick, c.temperature, c.seed), ("an answer", 0.4, 10 + 3 * 7919));
    assert_eq!(
        drawn,
        vec![(0.0, 10), (0.2, 10 + 7919), (0.2, 10 + 2 * 7919), (0.4, 10 + 3 * 7919)]
    );
    assert_eq!(
        climb_note(c.found, c.temperature, c.why, &c.tried),
        "found at temperature 0.4: 0 nothing, 0.2 no words, 0.2 no words, 0.4 an answer"
    );
    // nothing is an answer: the least bad is kept (a loop says something), the coolest of equals
    let ab = "ab".repeat(10);
    let c = climb(
        &[],
        1,
        0.0,
        0.4,
        1,
        1,
        |t, _| (Vec::new(), if t == 0.0 { String::new() } else { ab.clone() }, t),
        |_, _| None,
    );
    assert!(!c.found);
    assert_eq!((c.pick, c.why), (0.2, Some("a loop")));
    assert!(climb_note(c.found, c.temperature, c.why, &c.tried)
        .starts_with("no answer up to 0.4 (kept the one at 0.2: a loop)"));
    // a check (the guard) can turn an answer down
    let c = climb(
        &[],
        1,
        0.0,
        0.4,
        1,
        1,
        |t, _| (Vec::new(), "fine words".to_string(), t),
        |_, t| (*t < 0.4).then_some("rejected by the guard"),
    );
    assert_eq!((c.found, c.pick), (true, 0.4));
}
