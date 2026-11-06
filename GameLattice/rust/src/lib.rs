//! gamelattice - a RadixCyclicNN whose step is a game, feeding a LatticeFSM whose every state is one of the
//! encoder's outputs, read out by decoders that are separate from the lattice - and the loop that copies every
//! output back in through the sense that reads it.
//!
//! `../DESIGN.md` is the contract this crate is implemented against; its section numbers are cited throughout.
//! The two models it combines are the sibling crates: [`radixnet`] (the gram graph, its compression, its
//! cycles and sentinels) and [`latticefsm`] (the dense matrix of adaptive edges, the clock, credit per step).
//! Nothing in either changes; this crate wraps them.
//!
//! ```
//! use gamelattice::body::{Body, BodySettings};
//! use latticefsm::languages::language;
//!
//! let mut body = Body::for_language(&language("even-b").unwrap(), BodySettings::default()).unwrap();
//! let mut rng = latticefsm::rng::Rng::new(1);
//! body.teach(500, &mut rng, 6);                   // random strings, every step credited by the oracle
//! let verdict = body.classify("abab");            // quiet: the encoder's final named state, accepting or not
//! assert_eq!(verdict.named.len(), 4);
//! ```

#![allow(clippy::field_reassign_with_default)]

pub mod babble;
pub mod body;
pub mod clock;
pub mod credit;
pub mod decoders;
pub mod efference;
pub mod encoder;
pub mod experiment;
pub mod forward;
pub mod game;
pub mod gate;
pub mod lattice;
pub mod referee;
pub mod select;
pub mod server;
pub mod verdict;

pub use body::{Body, BodySettings, Classification};
pub use encoder::{EncoderSettings, GameGraph, Named};
pub use game::{Equilibrium, GameSettings, Payoff, Seat};
pub use lattice::{GameLattice, Kind};
pub use verdict::{Source, Verdict};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
