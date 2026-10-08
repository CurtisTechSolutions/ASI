//! The regular languages the machine is taught, and how examples of them are drawn.
//!
//! A language is a membership function over strings of `a` and `b` and the accepting states a machine is
//! given to learn it with (the start state is `0`).  Each is small enough that a few states suffice and the
//! machine's credit has to find a transition table, not a lookup.

use crate::rng::Rng;

pub const ALPHABET: [&str; 2] = ["a", "b"];

#[derive(Clone, Copy, Debug)]
pub struct Language {
    pub name: &'static str,
    pub description: &'static str,
    pub member: fn(&str) -> bool,
    /// Which states accept.  `0` is the start, so a language containing the empty string accepts at `0`.
    pub accepting: &'static [usize],
    /// The smallest deterministic machine that recognises it.
    pub min_states: usize,
}

pub const LANGUAGES: [Language; 4] = [
    Language {
        name: "even-b",
        description: "an even number of b's",
        member: |s| s.matches('b').count() % 2 == 0,
        accepting: &[0],
        min_states: 2,
    },
    Language {
        name: "contains-aa",
        description: "two a's in a row somewhere",
        member: |s| s.contains("aa"),
        accepting: &[2],
        min_states: 3,
    },
    Language {
        name: "ends-ab",
        description: "ends with ab",
        member: |s| s.ends_with("ab"),
        accepting: &[2],
        min_states: 3,
    },
    Language {
        name: "mod3-a",
        description: "a multiple of three a's",
        member: |s| s.matches('a').count() % 3 == 0,
        accepting: &[0],
        min_states: 3,
    },
];

pub fn language(name: &str) -> Result<Language, String> {
    LANGUAGES.iter().find(|l| l.name == name).copied().ok_or_else(|| {
        format!(
            "{name:?} is not one of {:?}",
            LANGUAGES.iter().map(|l| l.name).collect::<Vec<_>>()
        )
    })
}

/// A string of length 0 to `max_length` drawn uniformly over the alphabet.
pub fn random_string(rng: &mut Rng, max_length: usize) -> String {
    let n = rng.randint(0, max_length);
    (0..n).map(|_| *rng.choice(&ALPHABET)).collect()
}

/// `count` strings, each with its membership.
pub fn examples(lang: &Language, count: usize, rng: &mut Rng, max_length: usize) -> Vec<(String, bool)> {
    (0..count)
        .map(|_| {
            let s = random_string(rng, max_length);
            let m = (lang.member)(&s);
            (s, m)
        })
        .collect()
}
