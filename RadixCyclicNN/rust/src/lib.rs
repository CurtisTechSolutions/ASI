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
//! # What is here
//!
//! All of the Python package: the model of every kind - the count / reward
//! model, the sine-activation `radix` model ([`radix`]), the phase model
//! ([`resonance`]) and the negative network ([`negative`], [`duo`]) - with the
//! JSON model files byte for byte what Python writes, and around it every
//! teaching loop (the tutor, the chat, the critic, evolve, the recall tutor),
//! the LLM clients ([`llm`], [`ollama`], [`chatgpt`]), the agent and its tools,
//! code generation, images and speech, MCP, the CLI ([`cli`], `src/bin/radixnet.rs`) and the
//! HTTP API the frontend talks to ([`http`], [`service`]).  `rust/README.md`
//! lists every module, and what is deliberately not ported (torch, the Stable
//! Diffusion encoder, local Whisper) and why.
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
pub mod assistant;
pub mod attention;
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
pub mod phonetic;
pub mod plan;
pub mod recall;
pub mod report;
pub mod review;
pub mod search;
pub mod service;
pub mod source;
pub mod speech;
pub mod thinking;
pub mod toolbox;
pub mod tools;
pub mod training;
pub mod tutor;
pub mod vision;
pub mod voice;
pub mod web;
pub mod weights;
pub mod words;
pub mod zip;
// the sine-activation and phase models (radix / resonant) and what they stand on
pub mod activation;
pub mod backend;
pub mod blake2b;
pub mod dijkstra;
pub mod kinds;
pub mod metacog;
pub mod phasesearch;
pub mod pyheap;
pub mod radix;
pub mod resonance;
pub mod schedule;
// Chrome over WebDriver behind the web tools (`--browser`)
pub mod browser;

pub use beam::{BeamOptions, Prediction};
pub use counter::Counter;
pub use encoding::{parse_encoding, Encoding, Unit, Units, OVERLAP, WINDOW};
pub use file::{MODEL_FORMAT, MODEL_FORMAT_VERSION};
pub use graph::{Graph, GraphOptions, Transition, BACK, END, FIRST, START, THINK};
pub use json::Json;
pub use model::{EpochRecord, GenerateOptions, Meta, Model, PredictOptions, Score, TrainOptions};
pub use paths::{PathKey, PathOutcome, PathRow};
pub use penalty::{
    resolve_traversal, traversal_costs, PenaltyCosts, DEFAULT_TRAVERSAL, LEAST_PUNISHED, PUNISHMENT, REWARD, TRAVERSALS,
};
pub use search::{least_punished, onward, parse_traversal, PathResult, Traversal};
pub use weights::ChildCost;
pub use words::WordRow;
