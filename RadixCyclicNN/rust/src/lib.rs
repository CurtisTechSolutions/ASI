//! `radixnet` - the Rust port of the **count / reward model** of
//! RadixCyclicNN: a self-compressing cyclic trigram graph whose edge weights
//! are a dual frequency function of traversal counts (all time and inside a
//! sliding window) plus rewards, with beam-search prediction (top-K **and**
//! bottom-K continuations), generation, scoring and 2NRL feedback.
//!
//! It is the same model as `radixnet/` (Python) and `go/radixnet` (Go), built
//! to answer one question - *how fast is this in Rust?* - and to carry the
//! traversal the other two did not have: the walk that follows the **least
//! punished** step rather than the best rewarded one (see
//! `../SPEC-LeastPunished.md` and [`search::Traversal`]).
//!
//! # No dependencies
//!
//! The standard library only, like the other two implementations: the hash
//! ([`hash`]), the Mersenne Twister ([`mt19937`]), the exact float sum
//! ([`fsum`]) and the worker pool ([`parallel`]) are written out here.
//!
//! # What is here, and what is not
//!
//! The model itself is: the graph and its structural operations, the weight
//! function, the path contexts, both traversals, training, prediction,
//! generation, scoring and the benchmark.  The things around it that the Go
//! port grew - the negative network, the tutor, the agent, the LLM clients, the
//! HTTP server and the JSON model file format - are not ported; a model file is
//! read and written by the Python and Go implementations.
//!
//! ```
//! use radixnet::{GraphOptions, Model, PredictOptions, Traversal, TrainOptions};
//!
//! let mut model = Model::new(0, GraphOptions::default()).unwrap();
//! let texts: Vec<String> = ["the cat sat on the mat", "the cat sat on the log"]
//!     .iter()
//!     .map(|s| s.to_string())
//!     .collect();
//! model.train(&texts, &TrainOptions { epochs: 2, ..Default::default() }).unwrap();
//!
//! let found = model
//!     .predict("the cat", &PredictOptions { length: 8, k: 2, traversal: Traversal::Reward, ..Default::default() })
//!     .unwrap();
//! assert!(found.best.full_text.starts_with("the cat"));
//! ```

pub mod beam;
pub mod bench;
pub mod counter;
pub mod encoding;
pub mod fsum;
pub mod graph;
pub mod hash;
pub mod json;
pub mod model;
pub mod mt19937;
pub mod parallel;
pub mod paths;
pub mod search;
pub mod weights;

pub use beam::{BeamOptions, Prediction};
pub use counter::Counter;
pub use encoding::{encode, Trigram, OVERLAP, WINDOW};
pub use graph::{Graph, GraphOptions, Transition, BACK, END, FIRST, START};
pub use model::{EpochRecord, GenerateOptions, Meta, Model, PredictOptions, Score, TrainOptions};
pub use paths::{PathKey, PathOutcome, PathRow};
pub use search::{least_punished, onward, parse_traversal, PathResult, Traversal};
pub use weights::ChildCost;
