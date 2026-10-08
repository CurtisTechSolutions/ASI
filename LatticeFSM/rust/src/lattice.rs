//! The 3D matrix: one [`Edge`] for every `(source, symbol, target)`.
//!
//! The matrix is dense.  A machine of `S` states over an alphabet of `A`
//! symbols holds exactly `S · A · S` edges from the moment it is made, every
//! one a full record, and none is ever added or removed.  `lattice.edge(s, a, t)`
//! is the edge; `lattice.row(s, a)` is the fibre of `S` edges the machine
//! chooses among when it is in `s` and reads `a`.

use crate::edge::{Edge, Weighting};
use crate::json::Json;

/// One state of the machine: a cell's worth of bookkeeping beside the edges that leave it.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub index: usize,
    pub accepting: bool,
    pub visits: u64,
    pub last_visited: i64,
}

impl State {
    pub fn to_json(&self) -> Json {
        Json::Array(vec![
            self.index.into(),
            self.accepting.into(),
            (self.visits as f64).into(),
            self.last_visited.into(),
        ])
    }

    pub fn from_json(v: &Json) -> Result<State, String> {
        let l = v.as_array().ok_or("a state is a list")?;
        if l.len() != 4 {
            return Err("a state is a list of four values".to_string());
        }
        Ok(State {
            index: l[0].as_f64().ok_or("state index")? as usize,
            accepting: l[1].as_bool().ok_or("state accepting")?,
            visits: l[2].as_f64().ok_or("state visits")? as u64,
            last_visited: l[3].as_f64().ok_or("state last_visited")? as i64,
        })
    }
}

/// The dense `S × A × S` matrix of edges, with the alphabet and the states around it.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub n_states: usize,
    pub symbols: Vec<String>,
    /// The weighting every edge starts from; each adapts its own copy.
    pub prototype: Weighting,
    pub states: Vec<State>,
    /// Row-major: `edges[(s · A + a) · S + t]` is `(s, a, t)`.
    pub edges: Vec<Edge>,
}

impl Lattice {
    pub fn new(states: usize, alphabet: &[String], prototype: Weighting) -> Result<Lattice, String> {
        if states < 1 {
            return Err(format!("a machine needs at least one state, got {states}"));
        }
        if alphabet.is_empty() {
            return Err("the alphabet must be a non-empty sequence of distinct symbols".to_string());
        }
        for (i, s) in alphabet.iter().enumerate() {
            if alphabet[..i].contains(s) {
                return Err(format!("the alphabet repeats {s:?}"));
            }
        }
        let a = alphabet.len();
        let mut edges = Vec::with_capacity(states * a * states);
        for s in 0..states {
            for sym in 0..a {
                for t in 0..states {
                    edges.push(Edge::new(s, sym, t, prototype.clone()));
                }
            }
        }
        Ok(Lattice {
            n_states: states,
            symbols: alphabet.to_vec(),
            prototype,
            states: (0..states)
                .map(|i| State {
                    index: i,
                    accepting: false,
                    visits: 0,
                    last_visited: -1,
                })
                .collect(),
            edges,
        })
    }

    pub fn n_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// `(states, symbols, states)`: the three dimensions of the matrix.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_states, self.symbols.len(), self.n_states)
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn symbol_index(&self, symbol: &str) -> Result<usize, String> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .ok_or_else(|| format!("{symbol:?} is not in the alphabet {:?}", self.symbols))
    }

    pub fn offset(&self, source: usize, symbol: usize, target: usize) -> Result<usize, String> {
        let (s, a, _) = self.shape();
        if source >= s || target >= s || symbol >= a {
            return Err(format!(
                "({source}, {symbol}, {target}) is outside the matrix {:?}",
                self.shape()
            ));
        }
        Ok((source * a + symbol) * s + target)
    }

    pub fn edge(&self, source: usize, symbol: usize, target: usize) -> &Edge {
        &self.edges[self.offset(source, symbol, target).expect("edge index")]
    }

    pub fn edge_mut(&mut self, source: usize, symbol: usize, target: usize) -> &mut Edge {
        let i = self.offset(source, symbol, target).expect("edge index");
        &mut self.edges[i]
    }

    /// The `S` edges leaving `source` on `symbol`: what the machine chooses among.
    pub fn row(&self, source: usize, symbol: usize) -> &[Edge] {
        let start = self.offset(source, symbol, 0).expect("row index");
        &self.edges[start..start + self.n_states]
    }

    /// Every edge leaving `source`, on any symbol.
    pub fn leaving(&self, source: usize) -> &[Edge] {
        let start = self.offset(source, 0, 0).expect("row index");
        &self.edges[start..start + self.symbols.len() * self.n_states]
    }

    /// Every edge arriving at `target`, on any symbol, from any state.
    pub fn arriving(&self, target: usize) -> impl Iterator<Item = &Edge> {
        self.edges.iter().skip(target).step_by(self.n_states)
    }

    /// The edges anything was ever written to.
    pub fn touched(&self) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(|e| e.touched())
    }

    /// The matrix as JSON: the shape, the alphabet, the states, and only the edges that were written to.
    pub fn to_json(&self) -> Json {
        Json::object()
            .with("states", self.n_states.into())
            .with("alphabet", Json::strings(&self.symbols))
            .with("prototype", self.prototype.to_json())
            .with(
                "state_records",
                Json::Array(self.states.iter().map(State::to_json).collect()),
            )
            .with("edges", Json::Array(self.touched().map(Edge::to_json).collect()))
    }

    pub fn from_json(v: &Json) -> Result<Lattice, String> {
        let states = v.get("states").and_then(Json::as_f64).ok_or("lattice.states")? as usize;
        let alphabet: Vec<String> = v
            .get("alphabet")
            .and_then(Json::as_array)
            .ok_or("lattice.alphabet")?
            .iter()
            .map(|s| s.as_str().map(str::to_string).ok_or("a symbol is a string".to_string()))
            .collect::<Result<_, _>>()?;
        let prototype = Weighting::from_json(v.get("prototype").ok_or("lattice.prototype")?)?;
        let mut lattice = Lattice::new(states, &alphabet, prototype)?;
        for s in v
            .get("state_records")
            .and_then(Json::as_array)
            .ok_or("lattice.state_records")?
        {
            let state = State::from_json(s)?;
            if state.index >= states {
                return Err(format!("state {} is outside the machine", state.index));
            }
            let index = state.index;
            lattice.states[index] = state;
        }
        for e in v.get("edges").and_then(Json::as_array).ok_or("lattice.edges")? {
            let edge = Edge::from_json(e)?;
            let i = lattice.offset(edge.source, edge.symbol, edge.target)?;
            lattice.edges[i] = edge;
        }
        Ok(lattice)
    }
}
