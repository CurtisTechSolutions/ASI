//! The lattice (`DESIGN.md` §5): one class, three kinds, over [`latticefsm::Machine`] as it is.
//!
//! A *sense* lattice reads an organ's receptor symbols; an *act* lattice reads the act the decoder just
//! emitted, so its row `(s, a) -> t` is the forward model; the *association* lattice reads the sense lattices'
//! written states.  Beside the machine: the kind, a topographic flag (which refuses `rearrange` and
//! `swap_symbols` on the symbol axis), an abstain floor, and the reasons the verdicts on its edges carried.

use std::collections::BTreeMap;

use latticefsm::json::Json;
use latticefsm::machine::{Machine, Settings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sense,
    Act,
    Association,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Sense => "sense",
            Kind::Act => "act",
            Kind::Association => "association",
        }
    }
}

/// Reasons by edge: what the verdicts that landed there said, so a reflex can say why it withdrew and
/// `forget` can take one rule back (`DESIGN.md` §5.5).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reasons {
    pub table: BTreeMap<(usize, usize, usize), BTreeMap<String, f64>>,
}

impl Reasons {
    pub fn blame(&mut self, edge: (usize, usize, usize), reason: &str, amount: f64) {
        *self
            .table
            .entry(edge)
            .or_default()
            .entry(reason.to_string())
            .or_insert(0.0) += amount.abs();
    }

    /// Take `factor` of `reason`'s blame out of every edge; an edge left with none is dropped.
    pub fn forget(&mut self, reason: &str, factor: f64) -> usize {
        let mut touched = 0;
        for reasons in self.table.values_mut() {
            if let Some(b) = reasons.get_mut(reason) {
                *b *= 1.0 - factor.clamp(0.0, 1.0);
                touched += 1;
                if *b <= 1e-12 {
                    reasons.remove(reason);
                }
            }
        }
        self.table.retain(|_, r| !r.is_empty());
        touched
    }

    pub fn of(&self, edge: (usize, usize, usize)) -> Option<&BTreeMap<String, f64>> {
        self.table.get(&edge)
    }

    pub fn to_json(&self) -> Json {
        let rows: Vec<Json> = self
            .table
            .iter()
            .map(|((s, a, t), reasons)| {
                let mut r = Json::object();
                for (k, v) in reasons {
                    r.set(k, Json::Number(*v));
                }
                Json::Array(vec![
                    Json::Number(*s as f64),
                    Json::Number(*a as f64),
                    Json::Number(*t as f64),
                    r,
                ])
            })
            .collect();
        Json::Array(rows)
    }
}

pub struct GameLattice {
    pub name: String,
    pub kind: Kind,
    pub machine: Machine,
    /// A surface's adjacency lives beside the lattice; with this set the symbol axis never rearranges.
    pub topographic: bool,
    /// A row whose best target has less than this probability abstains (0: never).
    pub abstain_floor: f64,
    pub reasons: Reasons,
}

impl GameLattice {
    pub fn new(
        name: &str,
        kind: Kind,
        states: usize,
        alphabet: &[String],
        accepting: &[usize],
        settings: Settings,
    ) -> Result<GameLattice, String> {
        let machine = Machine::new(states, alphabet, accepting, settings)?;
        Ok(GameLattice {
            name: name.to_string(),
            kind,
            machine,
            topographic: false,
            abstain_floor: 0.0,
            reasons: Reasons::default(),
        })
    }

    pub fn states(&self) -> usize {
        self.machine.n_states()
    }

    /// Let the nodes rearrange themselves toward the centre - on the state axis only when topographic.
    pub fn rearrange(&mut self, full: bool) -> usize {
        if self.topographic {
            // the symbol axis is a map: only the states may move; latticefsm rearranges both axes, so the
            // symbol swaps it made are undone
            let swaps = self.machine.rearrange(full);
            let mut undone = 0;
            for (axis, i, j) in swaps.iter().rev() {
                if *axis == "symbols" {
                    let _ = self.machine.swap_symbols(*i, *j);
                    undone += 1;
                }
            }
            swaps.len() - undone
        } else {
            self.machine.rearrange(full).len()
        }
    }

    pub fn swap_symbols(&mut self, a: usize, b: usize) -> Result<(), String> {
        if self.topographic {
            return Err(format!(
                "{}: a topographic layer's symbol axis is a map and does not swap",
                self.name
            ));
        }
        self.machine.swap_symbols(a, b)
    }

    /// Whether the row `(source, symbol)` abstains: its best target is below the floor.
    pub fn abstains(&self, source: usize, symbol: usize) -> bool {
        if self.abstain_floor <= 0.0 {
            return false;
        }
        let p = self.machine.probabilities(source, symbol, None, None);
        p.iter().cloned().fold(0.0, f64::max) < self.abstain_floor
    }

    pub fn to_json(&self) -> Json {
        let mut j = Json::object();
        j.set("name", Json::String(self.name.clone()));
        j.set("kind", Json::String(self.kind.name().to_string()));
        j.set("topographic", Json::Bool(self.topographic));
        j.set("abstain_floor", Json::Number(self.abstain_floor));
        j.set("reasons", self.reasons.to_json());
        j.set("machine", self.machine.to_json());
        j
    }
}
