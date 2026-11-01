//! Babbling (`DESIGN.md` §8.3): before any teacher, an effector emits random symbols of its alphabet, reads the
//! return through its own sense, and the act lattice's rows learn by lessons; nothing credits the decoder.

use crate::body::{GameRecord, NimBody};

/// `games` random games of Nim: the forward model learns the rules' dynamics from the referee's replay.
pub fn babble_nim(body: &mut NimBody, games: usize) -> Vec<GameRecord> {
    (0..games).map(|i| body.play(i % 2 == 0, true)).collect()
}
