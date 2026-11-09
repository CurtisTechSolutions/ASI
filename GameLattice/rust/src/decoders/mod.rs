//! The decoders (`DESIGN.md` §6): separate from the lattice, typed by output, each naming the sense that reads
//! its output back, producing it by one of four mechanisms - a readout, repertoires on edges, propose-and-replay
//! through a referee, or a gate.

pub mod action;
pub mod readout;
pub mod repertoire;
pub mod taste;

use latticefsm::json::Json;

use crate::lattice::GameLattice;
use crate::verdict::Verdict;

/// What a decoder produces (`DESIGN.md` §6.2).
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Verdict(bool),
    Label(usize),
    States(Vec<usize>),
    Symbols(Vec<String>),
    Action(String),
    Distribution(Vec<f64>),
    Value(f64),
    Trace(Json),
    Node(usize),
    /// Nothing, with why.
    Abstain(String),
}

impl Output {
    pub fn to_json(&self) -> Json {
        match self {
            Output::Verdict(b) => Json::Bool(*b),
            Output::Label(l) => Json::Number(*l as f64),
            Output::States(s) => Json::Array(s.iter().map(|x| Json::Number(*x as f64)).collect()),
            Output::Symbols(s) => Json::strings(s),
            Output::Action(a) => Json::String(a.clone()),
            Output::Distribution(p) => Json::numbers(p),
            Output::Value(v) => Json::Number(*v),
            Output::Trace(t) => t.clone(),
            Output::Node(n) => Json::Number(*n as f64),
            Output::Abstain(why) => Json::object().with("abstain", Json::String(why.clone())),
        }
    }
}

/// The protocol (`DESIGN.md` §6.1).  A decoder that names no sense to read it back is refused by `check`.
pub trait Decoder {
    fn output_type(&self) -> &'static str;
    /// The sense that reads this output back.
    fn reads_back(&self) -> &str;
    fn read(&mut self, lattice: &GameLattice) -> Output;
    /// A verdict on the last output.  Returns how many of the decoder's own records moved.
    fn credit(&mut self, verdict: &Verdict) -> usize;
}

/// Refuse a decoder that names no feedback sense: grasp without touch is clumsy.
pub fn check(decoder: &dyn Decoder) -> Result<(), String> {
    if decoder.reads_back().trim().is_empty() {
        Err(format!(
            "a {} decoder must name the sense that reads it back",
            decoder.output_type()
        ))
    } else {
        Ok(())
    }
}
