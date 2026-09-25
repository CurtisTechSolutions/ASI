//! phonetok - a phonetic tokenizer: text as the sounds it is made of.
//!
//! The Rust port of the Python package of the same name (../phonetok), module
//! for module, reading the same two data files (the core lexicon and the
//! letter-to-sound rules), so that the three ports turn the same text into the
//! same sounds, token for token.  No dependencies.

pub mod acoustic;
pub mod g2p;
pub mod json;
pub mod lexicon;
pub mod mt;
pub mod numbers;
pub mod phones;
pub mod phonotactics;
pub mod rules;
pub mod syllables;
pub mod synth;
pub mod tokenizer;

pub use acoustic::{AcousticTokenizer, Codebook, Vocoder};
pub use g2p::{respell, Transcriber};
pub use lexicon::Lexicon;
pub use phonotactics::{violations, well_formed, Phonotactics};
pub use rules::letter_to_sound;
pub use syllables::{syllabify, Syllable};
pub use synth::{Synthesizer, VoiceSettings};
pub use tokenizer::{parse_token, Level, Token, Tokenizer, Vocab};
