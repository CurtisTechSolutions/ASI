//! `radixnet` - the Rust port of the **count / reward model** of
//! RadixCyclicNN: a self-compressing cyclic n-gram graph whose edge weights
//! are a dual frequency function of traversal counts (all time and inside a
//! sliding window) plus rewards, with beam-search prediction (top-K **and**
//! bottom-K continuations), generation, scoring and 2NRL feedback.
//!
//! It is the same model as `radixnet/` (Python) and `go/radixnet` (Go), built
//! to answer one question - *how fast is this in Rust?* - and it answers to all
//! three traversals they do: `reward`, the walk that follows what the model
//! believes; `punishment`, the same graph priced by the penalties alone
//! ([`penalty`]); and `least-punished`, the walk that ranks by the blame a path
//! carries before it looks at the cost at all ([`search::Traversal`],
//! `../SPEC-LeastPunished.md`).
//!
//! # No dependencies
//!
//! The standard library only, like the other two implementations: the hash
//! ([`hash`]), the Mersenne Twister ([`mt19937`]), the exact float sum
//! ([`fsum`]) and the worker pool ([`parallel`]) are written out here.
//!
//! # A gram is text
//!
//! Earlier releases packed a trigram into a `u64` - three code points of 21
//! bits - which is why the encoding pass allocated nothing.  A gram of any n
//! over any unit does not fit in an integer, so the index is keyed by the gram
//! itself, as the other two key it.  `bench/RESULTS.md` says what that cost.
//!
//! # What is here, and what is not
//!
//! The model itself is here in full: the graph and its structural operations,
//! the weight function, the path contexts, all three traversals, training,
//! prediction, generation, scoring, the character and **word** alphabets, the
//! JSON model file (byte for byte what Python and Go read and write), the
//! benchmark, the CLI (`src/bin/radixnet.rs`) and the HTTP API the frontend talks to
//! ([`http`], [`service`]).
//!
//! What is **not** here is everything the Python package grew around the model:
//! the negative network, the tutor and the other teaching loops, the agent and
//! its tools, the LLM clients, images and speech, and MCP.  The LLM clients in
//! particular need HTTPS, which the no-dependency rule rules out.
//!
//! ```
//! use radixnet::{GraphOptions, Model, PredictOptions, TrainOptions, REWARD};
//!
//! let mut model = Model::new(0, GraphOptions::default()).unwrap();
//! let texts: Vec<String> = ["the cat sat on the mat", "the cat sat on the log"]
//!     .iter()
//!     .map(|s| s.to_string())
//!     .collect();
//! model.train(&texts, &TrainOptions { epochs: 2, ..Default::default() }).unwrap();
//!
//! let found = model
//!     .predict("the cat", &PredictOptions { length: 8, k: 2, traversal: REWARD.to_string(), ..Default::default() })
//!     .unwrap();
//! assert!(found.best.full_text.starts_with("the cat"));
//! ```

pub mod agent;
pub mod beam;
pub mod bench;
pub mod blame;
pub mod calc;
pub mod chat;
pub mod chatgpt;
pub mod checkpoint;
pub mod cli;
pub mod clock;
pub mod codegen;
pub mod correct;
pub mod counter;
pub mod critic;
pub mod dialogue;
pub mod diff;
pub mod duo;
pub mod encoding;
pub mod fetch;
pub mod file;
pub mod fsum;
pub mod gan;
pub mod graph;
pub mod gzip;
pub mod hash;
pub mod http;
pub mod json;
pub mod llm;
pub mod log;
pub mod mcp;
pub mod model;
pub mod mt19937;
pub mod multipart;
pub mod negative;
pub mod ollama;
pub mod parallel;
pub mod paths;
pub mod penalty;
pub mod plan;
pub mod recall;
pub mod report;
pub mod review;
pub mod search;
pub mod service;
pub mod source;
pub mod speech;
pub mod toolbox;
pub mod tools;
pub mod tutor;
pub mod vision;
pub mod web;
pub mod weights;
pub mod words;
pub mod zip;

pub use beam::{BeamOptions, Prediction};
pub use counter::Counter;
pub use encoding::{parse_encoding, Encoding, Unit, Units, OVERLAP, WINDOW};
pub use file::{MODEL_FORMAT, MODEL_FORMAT_VERSION};
pub use graph::{Graph, GraphOptions, Transition, BACK, END, FIRST, START};
pub use json::Json;
pub use model::{EpochRecord, GenerateOptions, Meta, Model, PredictOptions, Score, TrainOptions};
pub use paths::{PathKey, PathOutcome, PathRow};
pub use penalty::{
    resolve_traversal, traversal_costs, PenaltyCosts, DEFAULT_TRAVERSAL, LEAST_PUNISHED, PUNISHMENT, REWARD, TRAVERSALS,
};
pub use search::{least_punished, onward, parse_traversal, PathResult, Traversal};
pub use weights::ChildCost;
pub use words::WordRow;
