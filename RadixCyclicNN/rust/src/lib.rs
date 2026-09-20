//! `radixnet` - the Rust port of the **count / reward model** of
//! RadixCyclicNN: a self-compressing cyclic n-gram graph whose edge weights
//! are a dual frequency function of traversal counts (all time and inside a
//! sliding window) plus rewards, with beam-search prediction (top-K **and**
//! bottom-K continuations), generation, scoring and 2NRL feedback.
//!
//! How a text becomes those grams is two dials that compose: [`Encoding`] says
//! how many symbols a gram holds (`n`) and how far apart grams start
//! (`stride`), and `GraphOptions::words` says whether one symbol is a
//! character or a whole word ([`words`], `../SPEC-WordNGrams.md`).  So
//! character trigrams are the default, `n: 5, stride: 5` is groups of five
//! letters, and `words: true, n: 2` is the word bigram.
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
//! let mut model = Model::new(0, GraphOptions::default()).unwrap();   // character trigrams
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
//!
//! // the same corpus in word bigrams: `words` says a symbol is a word,
//! // `n` says a gram holds two of them
//! use radixnet::Encoding;
//! let opts = GraphOptions { words: true, encoding: Encoding { n: 2, stride: 1 }, ..Default::default() };
//! let mut model = Model::new(0, opts).unwrap();
//! model.train(&texts, &TrainOptions { epochs: 2, ..Default::default() }).unwrap();
//! assert_eq!(model.g.enc.spec(model.is_words()), "word:2:1");
//! ```

pub mod beam;
pub mod bench;
pub mod clock;
pub mod counter;
pub mod encoding;
pub mod file;
pub mod fsum;
pub mod graph;
pub mod gzip;
pub mod hash;
pub mod http;
pub mod json;
pub mod model;
pub mod mt19937;
pub mod parallel;
pub mod paths;
pub mod penalty;
pub mod report;
pub mod search;
pub mod service;
pub mod weights;
pub mod words;

pub use beam::{BeamOptions, Prediction};
pub use counter::Counter;
pub use encoding::{encode, Encoding, Gram, Trigram, OVERLAP, WINDOW};
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
pub use words::{split_words, symbol_word, word_symbol, Vocabulary, WordRow, MAX_WORDS, UNKNOWN_WORD};
