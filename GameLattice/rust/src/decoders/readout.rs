//! Readouts (`DESIGN.md` §6.3): nothing to learn in the decoder; the lattice is read.

use latticefsm::json::Json;

use super::{Decoder, Output};
use crate::lattice::GameLattice;
use crate::verdict::Verdict;

/// Accept or reject: the state the run ended in.
pub struct VerdictReadout {
    pub sense: String,
}

impl Decoder for VerdictReadout {
    fn output_type(&self) -> &'static str {
        "verdict"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        Output::Verdict(lattice.machine.lattice.states[lattice.machine.state].accepting)
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}

/// One of `k`: a state-to-label map.
pub struct LabelReadout {
    pub sense: String,
    pub labels: Vec<usize>,
}

impl Decoder for LabelReadout {
    fn output_type(&self) -> &'static str {
        "label"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        Output::Label(self.labels.get(lattice.machine.state).copied().unwrap_or(0))
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}

/// The states the run visited.
pub struct StateStream {
    pub sense: String,
}

impl Decoder for StateStream {
    fn output_type(&self) -> &'static str {
        "state stream"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        let m = &lattice.machine;
        Output::States(m.path.iter().map(|&i| m.lattice.edges[i].target).collect())
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}

/// A scalar: the mean net credit per traversal along the run - how sure the lattice was.
pub struct ValueReadout {
    pub sense: String,
}

impl Decoder for ValueReadout {
    fn output_type(&self) -> &'static str {
        "value"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        let m = &lattice.machine;
        if m.path.is_empty() {
            return Output::Value(0.0);
        }
        let v: f64 = m.path.iter().map(|&i| m.lattice.edges[i].net()).sum::<f64>() / m.path.len() as f64;
        Output::Value(v)
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}

/// Why: the run's edges with their records, and the reasons that landed on them.
pub struct TraceReadout {
    pub sense: String,
}

impl Decoder for TraceReadout {
    fn output_type(&self) -> &'static str {
        "trace"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        let m = &lattice.machine;
        let (clock, life, stim) = (m.clock, m.life, m.stimulation());
        let edges: Vec<Json> = m
            .path
            .iter()
            .map(|&i| {
                let e = &m.lattice.edges[i];
                let mut j = e.describe(clock, life, stim);
                if let Some(r) = lattice.reasons.of((e.source, e.symbol, e.target)) {
                    let mut rj = Json::object();
                    for (k, v) in r {
                        rj.set(k, Json::Number(*v));
                    }
                    j.set("reasons", rj);
                }
                j
            })
            .collect();
        Output::Trace(Json::object().with("edges", Json::Array(edges)))
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}

/// The honest channel (`DESIGN.md` §6.2 expression): a readout of the gauges the credit uses, uncredited by
/// design - which is what makes it honest.
pub struct ExpressionReadout {
    pub sense: String,
}

impl Decoder for ExpressionReadout {
    fn output_type(&self) -> &'static str {
        "expression"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        let m = &lattice.machine;
        let mut j = Json::object();
        j.set("stimulation", Json::Number(m.stimulation()));
        j.set("clock", Json::Number(m.clock as f64));
        j.set("runs", Json::Number(m.runs as f64));
        j.set("credits", Json::Number(m.credits as f64));
        j.set("state", Json::Number(m.state as f64));
        let touched = m.lattice.touched().count();
        j.set("touched", Json::Number(touched as f64));
        Output::Trace(j)
    }
    fn credit(&mut self, _: &Verdict) -> usize {
        0
    }
}
