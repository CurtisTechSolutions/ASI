//! The finite state machine that walks the matrix.
//!
//! A machine is in one state.  It reads a symbol, looks along the row of edges
//! `lattice.row(state, symbol)` - one edge per possible next state - weighs
//! each by its own weighting function and the machine's stimulation, draws
//! one, traverses it, and is in that state.  A string of symbols is a run; the
//! run ends in a state that is accepting or not.  Everything else here feeds
//! back into that: the clock (every traversal is a tick), credit (`reward` and
//! `punish` land on the current run, discounted back from its end), the
//! stimulation (a level that multiplies the log-width term of every weight and
//! relaxes toward its baseline on the clock), and quiet runs (a draw without a
//! traversal: a measurement).  `DESIGN.md` §5-§8.

use std::fs;
use std::io::{Read, Write};

use crate::compress::{compress, Core, Precision};
use crate::edge::{Edge, Weighting};
use crate::gzip;
use crate::json::{parse, Json};
use crate::lattice::Lattice;
use crate::rng::Rng;

/// Ticks for a trace, an age, a width's excess or the stimulation's excess to fade by a half.
pub const LIFE: f64 = 1_000.0;
/// The stimulation a machine rests at: width counts in proportion.
pub const BASELINE: f64 = 1.0;
/// How much of a run's credit each step further back from its end receives.
pub const DISCOUNT: f64 = 0.8;
/// The states of a machine made without saying how many: with [`DEFAULT_ALPHABET`], a 13 × 13 × 13 matrix.
pub const DEFAULT_STATES: usize = 13;
/// The alphabet of a machine made without saying which: thirteen symbols, `a` to `m`.  The languages are over
/// `a` and `b`, so a default machine can be taught any of them; the other eleven symbols' edges wait unused.
pub const DEFAULT_ALPHABET: &str = "abcdefghijklm";

const FORMAT: &str = "latticefsm-machine";
const FORMAT_VERSION: f64 = 1.0;

/// A function that weighs an edge in place of its own weighting: `(edge, clock, life, stimulation) -> log-weight`.
pub type WeightFn = Box<dyn Fn(&Edge, i64, f64, f64) -> f64 + Send + Sync>;

/// One step the machine took (or, quietly, would have taken).
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub source: usize,
    pub symbol: String,
    pub target: usize,
    /// The probability the step had among the row's options under the stimulation it was taken at.
    pub probability: f64,
    pub stimulation: f64,
    pub clock: i64,
}

impl Transition {
    pub fn to_json(&self) -> Json {
        Json::object()
            .with("source", self.source.into())
            .with("symbol", self.symbol.as_str().into())
            .with("target", self.target.into())
            .with("probability", self.probability.into())
            .with("stimulation", self.stimulation.into())
            .with("clock", self.clock.into())
    }
}

/// A string read from the start state: the steps, where it ended, and whether that is accepting.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub transitions: Vec<Transition>,
    pub final_state: usize,
    pub accepted: bool,
}

impl Run {
    /// The states visited, the start first.
    pub fn states(&self) -> Vec<usize> {
        match self.transitions.first() {
            None => vec![self.final_state],
            Some(first) => {
                let mut v = vec![first.source];
                v.extend(self.transitions.iter().map(|t| t.target));
                v
            }
        }
    }

    pub fn log_probability(&self) -> f64 {
        self.transitions.iter().map(|t| t.probability.ln()).sum()
    }

    pub fn to_json(&self) -> Json {
        Json::object()
            .with(
                "transitions",
                Json::Array(self.transitions.iter().map(Transition::to_json).collect()),
            )
            .with(
                "states",
                Json::Array(self.states().into_iter().map(Json::from).collect()),
            )
            .with("final", self.final_state.into())
            .with("accepted", self.accepted.into())
            .with("log_probability", self.log_probability().into())
    }
}

/// The settings a machine is made with.
#[derive(Clone, Debug)]
pub struct Settings {
    pub start: usize,
    pub life: f64,
    pub baseline: f64,
    /// Ticks for the stimulation's excess over the baseline to fade by a half; `None` is `life`.
    pub calm: Option<f64>,
    pub temperature: f64,
    pub discount: f64,
    pub trace: f64,
    pub use_widening: f64,
    pub reward_widening: f64,
    pub punish_narrowing: f64,
    pub prototype: Weighting,
    pub seed: u64,
    /// Fold the matrix into its central node every this many transitions (traversals); 0 never.
    pub compress_every: u64,
    /// How the automatic compression packs the edges' numbers.
    pub compress_precision: Precision,
    /// Rebuild the matrix from the code after every automatic compression, so a lossy precision's loss is applied.
    pub compress_rebuild: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            start: 0,
            life: LIFE,
            baseline: BASELINE,
            calm: None,
            temperature: 1.0,
            discount: DISCOUNT,
            trace: 1.0,
            use_widening: 0.01,
            reward_widening: 0.2,
            punish_narrowing: 0.2,
            prototype: Weighting::default(),
            seed: 1,
            compress_every: 0,
            compress_precision: Precision::Exact,
            compress_rebuild: false,
        }
    }
}

/// A finite state machine over a dense 3D matrix of adaptive edges.
pub struct Machine {
    pub lattice: Lattice,
    pub start: usize,
    pub life: f64,
    pub baseline: f64,
    pub calm: f64,
    pub temperature: f64,
    pub discount: f64,
    pub trace: f64,
    pub use_widening: f64,
    pub reward_widening: f64,
    pub punish_narrowing: f64,
    pub weight_fn: Option<WeightFn>,
    pub seed: u64,
    pub rng: Rng,
    /// Every traversal, plus time let pass.
    pub clock: i64,
    pub state: usize,
    /// The edges of the current run, by offset, in order: what credit lands on.
    pub path: Vec<usize>,
    stimulation: f64,
    stimulation_stamp: i64,
    pub runs: u64,
    pub credits: u64,
    /// Fold the matrix into its central node every this many transitions; 0 never.
    pub compress_every: u64,
    pub compress_precision: Precision,
    pub compress_rebuild: bool,
    /// The latest code the central node holds, from [`Machine::compress_now`].
    pub core: Option<Core>,
    /// How many times the matrix has been folded into its central node.
    pub compressions: u64,
    /// Transitions since the last compression.
    pub since_compression: u64,
    /// The clock at the last compression; `-1` never.
    pub last_compressed: i64,
}

impl Machine {
    pub fn new(states: usize, alphabet: &[String], accepting: &[usize], settings: Settings) -> Result<Machine, String> {
        if !(settings.life.is_finite() && settings.life > 0.0) {
            return Err(format!(
                "life must be a positive number of ticks, got {}",
                settings.life
            ));
        }
        if settings.baseline < 0.0 {
            return Err(format!("baseline stimulation must be >= 0, got {}", settings.baseline));
        }
        if !(0.0..=1.0).contains(&settings.discount) {
            return Err(format!("discount must be in [0, 1], got {}", settings.discount));
        }
        if settings.temperature < 0.0 {
            return Err(format!("temperature must be >= 0, got {}", settings.temperature));
        }
        let mut lattice = Lattice::new(states, alphabet, settings.prototype.clone())?;
        if settings.start >= states {
            return Err(format!(
                "start state {} is not one of 0..{}",
                settings.start,
                states - 1
            ));
        }
        for &s in accepting {
            if s >= states {
                return Err(format!("accepting state {s} is not one of 0..{}", states - 1));
            }
            lattice.states[s].accepting = true;
        }
        lattice.states[settings.start].visits += 1;
        lattice.states[settings.start].last_visited = 0;
        Ok(Machine {
            lattice,
            start: settings.start,
            life: settings.life,
            baseline: settings.baseline,
            calm: settings.calm.unwrap_or(settings.life),
            temperature: settings.temperature,
            discount: settings.discount,
            trace: settings.trace,
            use_widening: settings.use_widening,
            reward_widening: settings.reward_widening,
            punish_narrowing: settings.punish_narrowing,
            weight_fn: None,
            seed: settings.seed,
            compress_every: settings.compress_every,
            compress_precision: settings.compress_precision,
            compress_rebuild: settings.compress_rebuild,
            core: None,
            compressions: 0,
            since_compression: 0,
            last_compressed: -1,
            rng: Rng::new(settings.seed),
            clock: 0,
            state: settings.start,
            path: Vec::new(),
            stimulation: settings.baseline,
            stimulation_stamp: 0,
            runs: 0,
            credits: 0,
        })
    }

    /// The default machine: [`DEFAULT_STATES`] states over [`DEFAULT_ALPHABET`], a 13 × 13 × 13 matrix.
    pub fn default_shape(accepting: &[usize], settings: Settings) -> Result<Machine, String> {
        Machine::over(DEFAULT_STATES, DEFAULT_ALPHABET, accepting, settings)
    }

    /// A machine over a one-character-per-symbol alphabet, e.g. `"ab"`.
    pub fn over(states: usize, alphabet: &str, accepting: &[usize], settings: Settings) -> Result<Machine, String> {
        let symbols: Vec<String> = alphabet.chars().map(|c| c.to_string()).collect();
        Machine::new(states, &symbols, accepting, settings)
    }

    // ---- shape and lookups ---------------------------------------------------------------------------------------

    pub fn n_states(&self) -> usize {
        self.lattice.n_states
    }

    pub fn alphabet(&self) -> &[String] {
        &self.lattice.symbols
    }

    /// The middle state: the source and target of the matrix's central node (`geometry::center`).
    pub fn center_state(&self) -> usize {
        self.n_states() / 2
    }

    pub fn accepting(&self) -> Vec<usize> {
        self.lattice
            .states
            .iter()
            .filter(|s| s.accepting)
            .map(|s| s.index)
            .collect()
    }

    pub fn symbol(&self, symbol: &str) -> Result<usize, String> {
        self.lattice.symbol_index(symbol)
    }

    pub fn edge(&self, source: usize, symbol: usize, target: usize) -> &Edge {
        self.lattice.edge(source, symbol, target)
    }

    pub fn edge_mut(&mut self, source: usize, symbol: usize, target: usize) -> &mut Edge {
        self.lattice.edge_mut(source, symbol, target)
    }

    /// Split a string into symbols: one per character when every symbol is one character, else whitespace-separated.
    pub fn tokenize(&self, text: &str) -> Result<Vec<usize>, String> {
        let single = self.lattice.symbols.iter().all(|s| s.chars().count() == 1);
        if single {
            text.chars()
                .filter(|c| !c.is_whitespace())
                .map(|c| self.symbol(&c.to_string()))
                .collect()
        } else {
            text.split_whitespace().map(|s| self.symbol(s)).collect()
        }
    }

    // ---- stimulation ---------------------------------------------------------------------------------------------

    /// The level now: the baseline plus whatever excess is left, halved every `calm` ticks.
    pub fn stimulation(&self) -> f64 {
        let excess = self.stimulation - self.baseline;
        if excess == 0.0 {
            self.baseline
        } else {
            self.baseline + excess * 0.5f64.powf((self.clock - self.stimulation_stamp) as f64 / self.calm)
        }
    }

    pub fn set_stimulation(&mut self, level: f64) -> Result<(), String> {
        if level < 0.0 {
            return Err(format!("stimulation must be >= 0, got {level}"));
        }
        self.stimulation = level;
        self.stimulation_stamp = self.clock;
        Ok(())
    }

    /// Raise (or, negative, lower) the stimulation by `amount` from where it is now; returns the new level.
    /// The stimulation as the machine file writes it: `[level, stamp]`.
    pub fn to_json_stimulation(&self) -> Json {
        Json::numbers(&[self.stimulation, self.stimulation_stamp as f64])
    }

    pub fn stimulate(&mut self, amount: f64) -> f64 {
        let level = (self.stimulation() + amount).max(0.0);
        self.set_stimulation(level).expect("non-negative");
        level
    }

    // ---- weighing and choosing -----------------------------------------------------------------------------------

    /// The log-weight of every edge in the row `(source, symbol)`, by target.
    pub fn log_weights(&self, source: usize, symbol: usize, stimulation: Option<f64>) -> Vec<f64> {
        let stim = stimulation.unwrap_or_else(|| self.stimulation());
        let row = self.lattice.row(source, symbol);
        match &self.weight_fn {
            Some(f) => row.iter().map(|e| f(e, self.clock, self.life, stim)).collect(),
            None => row.iter().map(|e| e.log_weight(self.clock, self.life, stim)).collect(),
        }
    }

    /// The row's log-weights as a distribution over targets: softmax at `temperature`; greedy at zero.
    pub fn probabilities(
        &self,
        source: usize,
        symbol: usize,
        stimulation: Option<f64>,
        temperature: Option<f64>,
    ) -> Vec<f64> {
        softmax(
            &self.log_weights(source, symbol, stimulation),
            temperature.unwrap_or(self.temperature),
        )
    }

    /// Draw a target from the row, returning it and the probability it had.
    pub fn choose(
        &mut self,
        source: usize,
        symbol: usize,
        stimulation: Option<f64>,
        temperature: Option<f64>,
    ) -> (usize, f64) {
        let probs = self.probabilities(source, symbol, stimulation, temperature);
        let r = self.rng.random();
        let mut acc = 0.0;
        for (t, p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return (t, *p);
            }
        }
        let t = argmax(&probs);
        (t, probs[t])
    }

    // ---- moving --------------------------------------------------------------------------------------------------

    /// Back to the start state with an empty path.  Not a tick.
    pub fn reset(&mut self) {
        self.state = self.start;
        self.path.clear();
    }

    /// Read one symbol: choose a next state from the current one, traverse the edge, and be there.  Quiet, the
    /// choice is made and reported and nothing moves: not the state, not the edge, not the clock.
    pub fn step(
        &mut self,
        symbol: usize,
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Transition {
        let stim = stimulation.unwrap_or_else(|| self.stimulation());
        let source = self.state;
        let (target, p) = self.choose(source, symbol, Some(stim), temperature);
        let t = Transition {
            source,
            symbol: self.lattice.symbols[symbol].clone(),
            target,
            probability: p,
            stimulation: stim,
            clock: self.clock,
        };
        if !quiet {
            self.traverse(source, symbol, target);
            self.state = target;
        }
        t
    }

    fn traverse(&mut self, source: usize, symbol: usize, target: usize) {
        let i = self.lattice.offset(source, symbol, target).expect("edge index");
        let (clock, life, trace, widen) = (self.clock, self.life, self.trace, self.use_widening);
        self.lattice.edges[i].traverse(clock, life, trace, widen);
        self.path.push(i);
        self.clock += 1;
        let s = &mut self.lattice.states[target];
        s.visits += 1;
        s.last_visited = self.clock;
        self.after_transition();
    }

    /// Count a transition, and fold the matrix into its central node when `compress_every` of them have passed.
    fn after_transition(&mut self) {
        self.since_compression += 1;
        if self.compress_every > 0 && self.since_compression >= self.compress_every {
            self.compress_now();
        }
    }

    /// Fold the matrix into its central node now, at `compress_precision`; with `compress_rebuild`, rebuild the
    /// matrix from the code, so the code's loss (none, at `exact`) is applied.  The path's offsets stay valid.
    pub fn compress_now(&mut self) -> &Core {
        let core = compress(self, Some(self.compress_precision), None);
        if self.compress_rebuild {
            core.rebuild_into(&mut self.lattice)
                .expect("a code rebuilds the machine it was made from");
        }
        self.compressions += 1;
        self.since_compression = 0;
        self.last_compressed = self.clock;
        self.core = Some(core);
        self.core.as_ref().expect("just set")
    }

    /// Read a string from the start state.  Quiet, nothing moves.
    pub fn run(&mut self, symbols: &[usize], stimulation: Option<f64>, temperature: Option<f64>, quiet: bool) -> Run {
        self.run_from(self.start, symbols, stimulation, temperature, quiet)
    }

    /// Read a string from the middle state - the central node's - and walk outward from there.
    pub fn run_from_middle(
        &mut self,
        symbols: &[usize],
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Run {
        self.run_from(self.center_state(), symbols, stimulation, temperature, quiet)
    }

    /// Read a string from `origin`.  Quiet, nothing moves.
    pub fn run_from(
        &mut self,
        origin: usize,
        symbols: &[usize],
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Run {
        let origin = origin.min(self.n_states() - 1);
        let stim = stimulation.unwrap_or_else(|| self.stimulation());
        if quiet {
            let mut state = origin;
            let mut transitions = Vec::with_capacity(symbols.len());
            for &sym in symbols {
                let (target, p) = self.choose(state, sym, Some(stim), temperature);
                transitions.push(Transition {
                    source: state,
                    symbol: self.lattice.symbols[sym].clone(),
                    target,
                    probability: p,
                    stimulation: stim,
                    clock: self.clock,
                });
                state = target;
            }
            return Run {
                transitions,
                final_state: state,
                accepted: self.lattice.states[state].accepting,
            };
        }
        self.reset();
        self.state = origin;
        self.runs += 1;
        let transitions = symbols
            .iter()
            .map(|&sym| self.step(sym, Some(stim), temperature, false))
            .collect();
        Run {
            transitions,
            final_state: self.state,
            accepted: self.lattice.states[self.state].accepting,
        }
    }

    /// The same, from a string of text (`tokenize`).
    pub fn run_text(
        &mut self,
        text: &str,
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Result<Run, String> {
        let symbols = self.tokenize(text)?;
        Ok(self.run(&symbols, stimulation, temperature, quiet))
    }

    /// Let time pass: the clock advances with nothing traversed.
    pub fn tick(&mut self, ticks: i64) {
        self.clock += ticks.max(0);
    }

    // ---- credit --------------------------------------------------------------------------------------------------

    /// Credit the current run: its last edge receives `amount`, each earlier one `discount` times the next.
    /// Returns how many edges were credited.
    pub fn credit(&mut self, amount: f64) -> usize {
        if self.path.is_empty() || amount == 0.0 {
            return 0;
        }
        self.credits += 1;
        let mut share = amount;
        let (clock, life, widen, narrow) = (self.clock, self.life, self.reward_widening, self.punish_narrowing);
        for &i in self.path.iter().rev() {
            self.lattice.edges[i].credit(share, clock, life, widen, narrow);
            share *= self.discount;
            if share.abs() < 1e-12 {
                break;
            }
        }
        self.path.len()
    }

    pub fn reward(&mut self, amount: f64) -> usize {
        self.credit(amount.abs())
    }

    pub fn punish(&mut self, amount: f64) -> usize {
        self.credit(-amount.abs())
    }

    /// Traverse one edge deliberately and credit it `amount`: a lesson rather than an experience.
    pub fn teach(&mut self, source: usize, symbol: usize, target: usize, amount: f64) -> Result<&Edge, String> {
        let i = self.lattice.offset(source, symbol, target)?;
        let (clock, life, trace, widen) = (self.clock, self.life, self.trace, self.use_widening);
        self.lattice.edges[i].traverse(clock, life, trace, widen);
        self.clock += 1;
        let (clock, rw, pn) = (self.clock, self.reward_widening, self.punish_narrowing);
        self.lattice.edges[i].credit(amount, clock, life, rw, pn);
        self.after_transition();
        Ok(&self.lattice.edges[i])
    }

    // ---- measurements (quiet) ------------------------------------------------------------------------------------

    /// Whether the greedy (or, at a temperature, a sampled) quiet run ends in an accepting state.
    pub fn accepts(&mut self, symbols: &[usize], temperature: f64) -> bool {
        self.run(symbols, None, Some(temperature), true).accepted
    }

    /// The share of `(string, accept?)` examples the quiet run classifies correctly.
    pub fn accuracy(&mut self, examples: &[(String, bool)], temperature: f64) -> f64 {
        if examples.is_empty() {
            return 0.0;
        }
        let mut right = 0usize;
        for (text, accept) in examples {
            let symbols = self.tokenize(text).expect("examples are over the alphabet");
            if self.accepts(&symbols, temperature) == *accept {
                right += 1;
            }
        }
        right as f64 / examples.len() as f64
    }

    /// The deterministic machine the greedy walk is: `table[state][symbol] = state`.
    pub fn transition_table(&self) -> Vec<Vec<usize>> {
        (0..self.n_states())
            .map(|s| {
                (0..self.lattice.n_symbols())
                    .map(|a| argmax(&self.probabilities(s, a, None, Some(0.0))))
                    .collect()
            })
            .collect()
    }

    pub fn stats(&self) -> Json {
        let touched: Vec<&Edge> = self.lattice.touched().collect();
        let seen: Vec<&&Edge> = touched.iter().filter(|e| e.seen > 0).collect();
        let (clock, life) = (self.clock, self.life);
        let widths: Vec<f64> = touched.iter().map(|e| e.width_at(clock, life)).collect();
        let (s, a, _) = self.lattice.shape();
        Json::object()
            .with("shape", Json::numbers(&[s as f64, a as f64, s as f64]))
            .with("alphabet", Json::strings(&self.lattice.symbols))
            .with("edges", self.lattice.len().into())
            .with("touched", touched.len().into())
            .with("traversed", seen.len().into())
            .with("clock", self.clock.into())
            .with("runs", (self.runs as f64).into())
            .with("credits", (self.credits as f64).into())
            .with("stimulation", self.stimulation().into())
            .with("baseline", self.baseline.into())
            .with("life", self.life.into())
            .with("temperature", self.temperature.into())
            .with("state", self.state.into())
            .with("start", self.start.into())
            .with(
                "accepting",
                Json::Array(self.accepting().into_iter().map(Json::from).collect()),
            )
            .with("total_seen", seen.iter().map(|e| e.seen as f64).sum::<f64>().into())
            .with("total_rewarded", touched.iter().map(|e| e.rewarded).sum::<f64>().into())
            .with("total_punished", touched.iter().map(|e| e.punished).sum::<f64>().into())
            .with("widest", widths.iter().cloned().fold(1.0, f64::max).into())
            .with("narrowest", widths.iter().cloned().fold(1.0, f64::min).into())
            .with(
                "center",
                Json::numbers(&[
                    self.center_state() as f64,
                    (self.lattice.n_symbols() / 2) as f64,
                    self.center_state() as f64,
                ]),
            )
            .with("compress_every", (self.compress_every as f64).into())
            .with("compress_precision", self.compress_precision.name().into())
            .with("compress_rebuild", self.compress_rebuild.into())
            .with("compressions", (self.compressions as f64).into())
            .with("since_compression", (self.since_compression as f64).into())
            .with("last_compressed", self.last_compressed.into())
            .with(
                "core_bytes",
                self.core
                    .as_ref()
                    .map(|c| c.bytes() as f64)
                    .map(Json::from)
                    .unwrap_or(Json::Null),
            )
    }

    // ---- persistence ---------------------------------------------------------------------------------------------

    /// The settings as the machine file writes them.
    pub fn settings_json(&self) -> Json {
        Json::object()
            .with("start", self.start.into())
            .with("life", self.life.into())
            .with("baseline", self.baseline.into())
            .with("calm", self.calm.into())
            .with("temperature", self.temperature.into())
            .with("discount", self.discount.into())
            .with("trace", self.trace.into())
            .with("use_widening", self.use_widening.into())
            .with("reward_widening", self.reward_widening.into())
            .with("punish_narrowing", self.punish_narrowing.into())
            .with("seed", (self.seed as f64).into())
            .with("compress_every", (self.compress_every as f64).into())
            .with("compress_precision", self.compress_precision.name().into())
            .with("compress_rebuild", self.compress_rebuild.into())
    }

    pub fn to_json(&self) -> Json {
        let state = self.rng.state();
        Json::object()
            .with("format", FORMAT.into())
            .with("version", FORMAT_VERSION.into())
            .with("settings", self.settings_json())
            .with("clock", self.clock.into())
            .with("state", self.state.into())
            .with("runs", (self.runs as f64).into())
            .with("credits", (self.credits as f64).into())
            .with("compressions", (self.compressions as f64).into())
            .with("since_compression", (self.since_compression as f64).into())
            .with("last_compressed", self.last_compressed.into())
            .with(
                "stimulation",
                Json::numbers(&[self.stimulation, self.stimulation_stamp as f64]),
            )
            .with(
                "rng",
                Json::object().with("xoshiro256", Json::strings(&state.map(|w| w.to_string()))),
            )
            .with("lattice", self.lattice.to_json())
    }

    /// A machine from its JSON.  A file written by the Python port carries that port's generator state under
    /// `rng`; it is not this generator's, so the machine is reseeded from its seed.
    pub fn from_json(v: &Json) -> Result<Machine, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let lattice = Lattice::from_json(v.get("lattice").ok_or("no lattice")?)?;
        let st = v.get("settings").ok_or("no settings")?;
        let seed = st.num("seed", 1.0) as u64;
        let mut rng = Rng::new(seed);
        if let Some(words) = v.get("rng").and_then(|r| r.get("xoshiro256")).and_then(Json::as_array) {
            if words.len() == 4 {
                let mut s = [0u64; 4];
                for (i, w) in words.iter().enumerate() {
                    s[i] = w.as_str().and_then(|t| t.parse().ok()).ok_or("bad rng state")?;
                }
                rng = Rng::from_state(s);
            }
        }
        let stim = v.get("stimulation").and_then(Json::as_array).ok_or("no stimulation")?;
        if stim.len() != 2 {
            return Err("stimulation is [level, stamp]".to_string());
        }
        let start = st.num("start", 0.0) as usize;
        if start >= lattice.n_states {
            return Err(format!("start state {start} is outside the machine"));
        }
        Ok(Machine {
            start,
            life: st.num("life", LIFE),
            baseline: st.num("baseline", BASELINE),
            calm: st.num("calm", st.num("life", LIFE)),
            temperature: st.num("temperature", 1.0),
            discount: st.num("discount", DISCOUNT),
            trace: st.num("trace", 1.0),
            use_widening: st.num("use_widening", 0.01),
            reward_widening: st.num("reward_widening", 0.2),
            punish_narrowing: st.num("punish_narrowing", 0.2),
            weight_fn: None,
            seed,
            rng,
            clock: v.num("clock", 0.0) as i64,
            state: (v.num("state", start as f64) as usize).min(lattice.n_states - 1),
            path: Vec::new(),
            stimulation: stim[0].as_f64().ok_or("stimulation level")?,
            stimulation_stamp: stim[1].as_f64().ok_or("stimulation stamp")? as i64,
            runs: v.num("runs", 0.0) as u64,
            credits: v.num("credits", 0.0) as u64,
            compress_every: st.num("compress_every", 0.0) as u64,
            compress_precision: Precision::parse(st.str_or("compress_precision", "exact"))?,
            compress_rebuild: st.bool_or("compress_rebuild", false),
            core: None,
            compressions: v.num("compressions", 0.0) as u64,
            since_compression: v.num("since_compression", 0.0) as u64,
            last_compressed: v.num("last_compressed", -1.0) as i64,
            lattice,
        })
    }

    /// Write the machine to `path` (`.json`, or `.json.gz` to gzip it).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let mut f = fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

/// Read a machine from `path` (`.json` or `.json.gz`).
pub fn load_machine(path: &str) -> Result<Machine, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if path.ends_with(".gz") || bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    Machine::from_json(&parse(&text)?)
}

pub fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    if temperature == 0.0 {
        let best = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let winners = logits.iter().filter(|&&v| v == best).count() as f64;
        return logits
            .iter()
            .map(|&v| if v == best { 1.0 / winners } else { 0.0 })
            .collect();
    }
    let scaled: Vec<f64> = logits.iter().map(|v| v / temperature).collect();
    let top = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|v| v / total).collect()
}

pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}
