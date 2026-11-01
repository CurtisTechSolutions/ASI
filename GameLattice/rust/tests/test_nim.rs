//! The Nim referee and the action decoder.

use gamelattice::babble::babble_nim;
use gamelattice::body::{NimBody, NimSettings};
use gamelattice::decoders::taste::{Swallow, Taste};
use gamelattice::referee::{Nim, NimMove};

#[test]
fn nim_is_the_solved_game_uge_describes() {
    let s = Nim::initial();
    assert_eq!(Nim::legal(s).len(), 15);
    assert_eq!(Nim::action_count(), 21);
    assert_eq!(Nim::state_codes(), 384);
    assert_eq!(Nim::nim_sum(s), 3 ^ 5 ^ 7);
    let best = Nim::optimal(s);
    assert!(best.iter().all(|&m| Nim::nim_sum(Nim::apply(s, m)) == 0));
    assert!(
        Nim::refuse(s, NimMove { heap: 0, take: 4 }, 0).is_some(),
        "the referee refuses"
    );
    assert!(Nim::refuse(s, NimMove { heap: 0, take: 3 }, 0).is_none());
    for i in 0..Nim::action_count() {
        assert_eq!(NimMove::from_index(i).index(), i);
    }
}

#[test]
fn babbling_teaches_the_forward_model_and_credits_no_repertoire() {
    let mut body = NimBody::new(NimSettings::default()).unwrap();
    let records = babble_nim(&mut body, 20);
    assert!(
        records.iter().map(|r| r.lessons).sum::<usize>() > 0,
        "lessons on the act lattice"
    );
    assert_eq!(
        body.decoder.proposer.emissions, 0,
        "no repertoire spoke during babbling"
    );
    let r = body.play(true, false);
    assert!(r.our_moves > 0);
}

#[test]
fn taste_spits_the_empty_and_the_blamed_and_swallows_the_clean() {
    let mut t = Taste::new("text");
    t.blamed.push("Traceback".to_string());
    assert_eq!(t.taste("   ").gate, Swallow::Spit);
    assert_eq!(t.taste("Traceback (most recent call last)").gate, Swallow::Spit);
    assert_eq!(t.taste("the cat sat on the mat").gate, Swallow::Swallow);
    assert_eq!(
        t.taste("abcdefghijklmnopabcdefghijklmnopabcdefghijklmnop").gate,
        Swallow::Caution
    );
}
