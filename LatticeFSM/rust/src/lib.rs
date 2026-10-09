//! latticefsm - a finite state machine over a dense 3D matrix of adaptive edges.
//!
//! The matrix is `states × symbols × states` - 13 × 13 × 13 unless asked otherwise
//! ([`machine::DEFAULT_STATES`], [`machine::DEFAULT_ALPHABET`]) - and every cell is an [`edge::Edge`]:
//! a dense record of how often and when the transition was traversed, what it
//! was rewarded and punished, how wide the channel is, and the coefficients of
//! its own adaptive weighting function.  A [`machine::Machine`] walks it as a
//! finite state machine: in a state, reading a symbol, it draws the next state
//! from the row of edges by their weights - and the wider an edge the easier
//! it is to take, the more so the more *stimulated* the machine is.
//!
//! Standard library only.  `../DESIGN.md` is the specification shared with the
//! Python package beside this crate; `../README.md` says what it measured.
//!
//! ```
//! use latticefsm::machine::{Machine, Settings};
//!
//! let mut m = Machine::over(3, "ab", &[0], Settings::default()).unwrap();
//! let symbols = m.tokenize("abab").unwrap();
//! let run = m.run(&symbols, None, None, false);
//! if run.accepted { m.reward(1.0); } else { m.punish(1.0); }
//! ```

pub mod compress;
pub mod edge;
pub mod experiment;
pub mod focus;
pub mod geometry;
pub mod gzip;
pub mod http;
pub mod json;
pub mod languages;
pub mod lattice;
pub mod machine;
pub mod rng;
pub mod server;

pub use edge::{Edge, Weighting};
pub use lattice::{Lattice, State};
pub use machine::{load_machine, Machine, Run, Settings, Transition, DEFAULT_ALPHABET, DEFAULT_STATES};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
