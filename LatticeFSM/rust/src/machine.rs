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
use crate::focus::{FocusDraw, FocusLearner};
use crate::geometry::{check_focus, focus_index, focus_node};

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
    /// A skip: the node passed through without stopping, the step reading two symbols (`symbol` holds both).
    pub skipped: Option<usize>,
}

impl Transition {
    pub fn to_json(&self) -> Json {
        Json::object()
            .with("skipped", self.skipped.map(Json::from).unwrap_or(Json::Null))
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
    /// A number in `[0, 1]`: the node of the central vertical vector runs start from; `None` the start state.
    pub focus: Option<f64>,
    /// Read the focus off the input through a learner, from credit.
    pub learn_focus: bool,
    pub focus_rate: f64,
    pub focus_explore: f64,
    /// Skip a node when a two-edge path is more efficient than the step the machine would take.
    pub skip: bool,
    pub skip_margin: f64,
    /// Let the nodes rearrange themselves every this many transitions; 0 never.
    pub rearrange_every: u64,
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
            focus: None,
            learn_focus: false,
            focus_rate: 0.05,
            focus_explore: 0.15,
            skip: false,
            skip_margin: 0.0,
            rearrange_every: 0,
        }
    }
}

/// How one run goes: every field `None` / `false` is the machine's own way.
#[derive(Clone, Copy, Debug, Default)]
pub struct Walk {
    pub stimulation: Option<f64>,
    pub temperature: Option<f64>,
    pub quiet: bool,
    /// Start from the middle state, the central node's.
    pub from_middle: bool,
    /// Start from the node this focus picks, for this run only.
    pub focus: Option<f64>,
    /// Skip or not for this run, instead of the machine's `skip`.
    pub skip: Option<bool>,
}

enum Move {
    Step(usize, f64),
    Skip(usize, usize, f64),
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
    focus: Option<f64>,
    pub learn_focus: bool,
    pub focus_learner: FocusLearner,
    last_focus_draw: Option<FocusDraw>,
    /// The focus the latest run started from, when it had one.
    pub last_focus: Option<f64>,
    pub skip: bool,
    pub skip_margin: f64,
    pub skips: u64,
    pub rearrange_every: u64,
    pub swaps: u64,
    pub since_rearrange: u64,
    rearrange_due: bool,
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
        if let Some(f) = settings.focus {
            check_focus(f)?;
        }
        if settings.skip_margin < 0.0 {
            return Err(format!("skip_margin must be >= 0, got {}", settings.skip_margin));
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
            focus: settings.focus,
            learn_focus: settings.learn_focus,
            focus_learner: FocusLearner::new(alphabet.to_vec(), settings.focus_rate, settings.focus_explore),
            last_focus_draw: None,
            last_focus: None,
            skip: settings.skip,
            skip_margin: settings.skip_margin,
            skips: 0,
            rearrange_every: settings.rearrange_every,
            swaps: 0,
            since_rearrange: 0,
            rearrange_due: false,
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

    /// The focus, a number in `[0, 1]`, or `None`: which node of the central vertical vector runs start from.
    pub fn focus(&self) -> Option<f64> {
        self.focus
    }

    pub fn set_focus(&mut self, focus: Option<f64>) -> Result<(), String> {
        if let Some(f) = focus {
            check_focus(f)?;
        }
        self.focus = focus;
        Ok(())
    }

    /// The state `focus` starts a run from: its node on the central vertical vector.
    pub fn focus_state(&self, focus: f64) -> usize {
        focus_index(self.n_states(), Some(focus))
    }

    /// Where a run starts: the focus node's state when there is a focus, else `start`.
    pub fn origin(&self) -> usize {
        match self.focus {
            Some(f) => self.focus_state(f),
            None => self.start,
        }
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
            skipped: None,
        };
        if !quiet {
            self.traverse(source, symbol, target);
            self.state = target;
            self.settle_rearrange();
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
        self.since_rearrange += 1;
        if self.rearrange_every > 0 && self.since_rearrange >= self.rearrange_every {
            self.rearrange_due = true;
        }
        if self.compress_every > 0 && self.since_compression >= self.compress_every {
            self.compress_now();
        }
    }

    // ---- rearranging ---------------------------------------------------------------------------------------------

    /// States `i` and `j` trade places in the matrix: a relabelling - every edge, the two state records, the start
    /// state, the current state and the run's path go with them - so the machine walks as before from its start
    /// state; what changes is which one the focus and the middle pick, and which shell each sits in.
    pub fn swap_states(&mut self, i: usize, j: usize) -> Result<(), String> {
        let n = self.n_states();
        if i >= n || j >= n {
            return Err(format!("states {i} and {j} are not both in 0..{}", n - 1));
        }
        if i == j {
            return Ok(());
        }
        let path: Vec<(usize, usize, usize)> = self
            .path
            .iter()
            .map(|&o| {
                (
                    self.lattice.edges[o].source,
                    self.lattice.edges[o].symbol,
                    self.lattice.edges[o].target,
                )
            })
            .collect();
        self.lattice.swap_states(i, j);
        let perm = |x: usize| {
            if x == i {
                j
            } else if x == j {
                i
            } else {
                x
            }
        };
        self.path = path
            .into_iter()
            .map(|(s, a, t)| self.lattice.offset(perm(s), a, perm(t)).expect("in range"))
            .collect();
        self.start = perm(self.start);
        self.state = perm(self.state);
        self.swaps += 1;
        Ok(())
    }

    /// Symbols `a` and `b` trade places: their slices and their labels; strings read the same as before.
    pub fn swap_symbols(&mut self, a: usize, b: usize) -> Result<(), String> {
        let n = self.lattice.n_symbols();
        if a >= n || b >= n {
            return Err(format!("symbols {a} and {b} are not both in 0..{}", n - 1));
        }
        if a == b {
            return Ok(());
        }
        let path: Vec<(usize, usize, usize)> = self
            .path
            .iter()
            .map(|&o| {
                (
                    self.lattice.edges[o].source,
                    self.lattice.edges[o].symbol,
                    self.lattice.edges[o].target,
                )
            })
            .collect();
        self.lattice.swap_symbols(a, b);
        self.focus_learner.swap_symbols(a, b);
        let perm = |x: usize| {
            if x == a {
                b
            } else if x == b {
                a
            } else {
                x
            }
        };
        self.path = path
            .into_iter()
            .map(|(s, y, t)| self.lattice.offset(s, perm(y), t).expect("in range"))
            .collect();
        self.swaps += 1;
        Ok(())
    }

    /// Let the nodes rearrange themselves toward the centre: one pass of neighbour swaps on the state axis and the
    /// symbol axis (`full`: passes until none is left).  On each axis, from the outside in on each side, a pair
    /// swaps when the node farther from the centre is busier.  Returns the swaps made, as `(axis, i, j)`.
    pub fn rearrange(&mut self, full: bool) -> Vec<(&'static str, usize, usize)> {
        let mut done = Vec::new();
        loop {
            let mut made = Vec::new();
            let mut load = self.lattice.state_load();
            for (i, j) in inward_pairs(load.len()) {
                if load[i] > load[j] {
                    self.swap_states(i, j).expect("in range");
                    load.swap(i, j);
                    made.push(("states", i.min(j), i.max(j)));
                }
            }
            let mut load = self.lattice.symbol_load();
            for (i, j) in inward_pairs(load.len()) {
                if load[i] > load[j] {
                    self.swap_symbols(i, j).expect("in range");
                    load.swap(i, j);
                    made.push(("symbols", i.min(j), i.max(j)));
                }
            }
            let settled = made.is_empty();
            done.extend(made);
            if !full || settled {
                break;
            }
        }
        self.since_rearrange = 0;
        done
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

    /// Read a string the machine's own way: from its learned focus's node, its focus node, or its start state, with
    /// its skips.  Quiet, nothing moves.
    pub fn run(&mut self, symbols: &[usize], stimulation: Option<f64>, temperature: Option<f64>, quiet: bool) -> Run {
        self.walk(
            symbols,
            Walk {
                stimulation,
                temperature,
                quiet,
                ..Walk::default()
            },
        )
        .expect("a walk without a focus of its own has nothing to refuse")
    }

    /// Read a string from the middle state - the central node's - and walk outward from there.
    pub fn run_from_middle(
        &mut self,
        symbols: &[usize],
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Run {
        self.walk(
            symbols,
            Walk {
                stimulation,
                temperature,
                quiet,
                from_middle: true,
                ..Walk::default()
            },
        )
        .expect("a walk without a focus of its own has nothing to refuse")
    }

    /// Read a string from `origin`, the machine's skips and nothing learned about the focus.  Quiet, nothing moves.
    pub fn run_from(
        &mut self,
        origin: usize,
        symbols: &[usize],
        stimulation: Option<f64>,
        temperature: Option<f64>,
        quiet: bool,
    ) -> Run {
        let walk = Walk {
            stimulation,
            temperature,
            quiet,
            ..Walk::default()
        };
        self.walk_from(origin.min(self.n_states() - 1), None, symbols, walk)
    }

    /// Read a string as `walk` says.  It starts from the middle state (`from_middle`), else the node `walk.focus`
    /// picks, else - with `learn_focus` - the node the learned focus reads off the input, else [`Machine::origin`].
    /// With skips on, two symbols are read in one move whenever that is more efficient (`decide`).
    pub fn walk(&mut self, symbols: &[usize], walk: Walk) -> Result<Run, String> {
        if let Some(f) = walk.focus {
            check_focus(f)?;
        }
        let mut learned = None;
        let origin = if walk.from_middle {
            self.center_state()
        } else if let Some(f) = walk.focus {
            self.focus_state(f)
        } else if self.learn_focus {
            let explore = !walk.quiet;
            let draw = {
                let rng = &mut self.rng;
                self.focus_learner.choose(symbols, || rng.gauss(), explore)
            };
            let origin = self.focus_state(FocusLearner::clip(draw.0));
            learned = Some(draw);
            origin
        } else {
            self.origin()
        };
        Ok(self.walk_from(origin, learned, symbols, walk))
    }

    fn walk_from(&mut self, origin: usize, learned: Option<FocusDraw>, symbols: &[usize], walk: Walk) -> Run {
        let stim = walk.stimulation.unwrap_or_else(|| self.stimulation());
        let do_skip = walk.skip.unwrap_or(self.skip);
        if !walk.quiet {
            self.reset();
            self.state = origin;
            self.runs += 1;
            self.last_focus = match &learned {
                Some(d) => Some(FocusLearner::clip(d.0)),
                None => walk.focus.or(self.focus),
            };
            self.last_focus_draw = learned;
        }
        let mut state = origin;
        let mut transitions = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            let a = symbols[i];
            let next = if do_skip && i + 1 < symbols.len() {
                Some(symbols[i + 1])
            } else {
                None
            };
            let clock = self.clock;
            match self.decide(state, a, next, stim, walk.temperature) {
                Move::Step(target, p) => {
                    if !walk.quiet {
                        let e = self.lattice.offset(state, a, target).expect("edge index");
                        self.traverse_offset(e, true);
                        self.state = target;
                    }
                    transitions.push(Transition {
                        source: state,
                        symbol: self.lattice.symbols[a].clone(),
                        target,
                        probability: p,
                        stimulation: stim,
                        clock,
                        skipped: None,
                    });
                    i += 1;
                }
                Move::Skip(mid, target, p) => {
                    let b = next.expect("a skip reads two symbols");
                    if !walk.quiet {
                        let e1 = self.lattice.offset(state, a, mid).expect("edge index");
                        self.traverse_offset(e1, false);
                        let e2 = self.lattice.offset(mid, b, target).expect("edge index");
                        self.traverse_offset(e2, true);
                        self.state = target;
                        self.skips += 1;
                    }
                    transitions.push(Transition {
                        source: state,
                        symbol: format!("{}{}", self.lattice.symbols[a], self.lattice.symbols[b]),
                        target,
                        probability: p,
                        stimulation: stim,
                        clock,
                        skipped: Some(mid),
                    });
                    i += 2;
                }
            }
            state = if walk.quiet {
                transitions.last().expect("a move").target
            } else {
                self.state
            };
        }
        let accepted = self.lattice.states[state].accepting;
        if !walk.quiet {
            self.settle_rearrange();
        }
        Run {
            transitions,
            final_state: if walk.quiet { state } else { self.state },
            accepted,
        }
    }

    /// A rearrangement that fell due during a run or a lesson happens when it is over, so nothing the run holds -
    /// the symbols still to read, a skip's target - refers to a position that moved under it.
    fn settle_rearrange(&mut self) {
        if self.rearrange_due {
            self.rearrange_due = false;
            self.rearrange(false);
        }
    }

    /// The next move from `state`: a step on `a`, or a skip on `a` then `b` past the node between - when the best
    /// two-edge path beats the path the step begins (the step, then the best edge after it) by more than
    /// `skip_margin` in summed log-probability, at the run's temperature, else the machine's, else 1.
    fn decide(&mut self, state: usize, a: usize, b: Option<usize>, stim: f64, temperature: Option<f64>) -> Move {
        let (t1, p1) = self.choose(state, a, Some(stim), temperature);
        let Some(b) = b else {
            return Move::Step(t1, p1);
        };
        let temp = match temperature {
            Some(t) if t > 0.0 => t,
            _ => {
                if self.temperature > 0.0 {
                    self.temperature
                } else {
                    1.0
                }
            }
        };
        let pa = self.probabilities(state, a, Some(stim), Some(temp));
        let (mut best_t, mut best_u, mut best_v, mut step_v) = (0, 0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (t, &pt) in pa.iter().enumerate() {
            let pb = self.probabilities(t, b, Some(stim), Some(temp));
            let u = argmax(&pb);
            let v = pt.max(1e-300).ln() + pb[u].max(1e-300).ln();
            if v > best_v {
                (best_t, best_u, best_v) = (t, u, v);
            }
            if t == t1 {
                step_v = v;
            }
        }
        if best_v > step_v + self.skip_margin {
            let pb = self.probabilities(best_t, b, Some(stim), Some(temp));
            return Move::Skip(best_t, best_u, pa[best_t] * pb[best_u]);
        }
        Move::Step(t1, p1)
    }

    fn traverse_offset(&mut self, i: usize, visit: bool) {
        let (clock, life, trace, widen) = (self.clock, self.life, self.trace, self.use_widening);
        self.lattice.edges[i].traverse(clock, life, trace, widen);
        self.path.push(i);
        self.clock += 1;
        if visit {
            let target = self.lattice.edges[i].target;
            let s = &mut self.lattice.states[target];
            s.visits += 1;
            s.last_visited = self.clock;
        }
        self.after_transition();
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
        if let Some(draw) = self.last_focus_draw.clone() {
            self.focus_learner.learn(amount, &draw);
        }
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
        // the edge's states and symbol by identity, to find it again if a rearrangement moves it
        let (sid, tid, label) = (
            self.lattice.state_ids[source],
            self.lattice.state_ids[target],
            self.lattice.symbols[symbol].clone(),
        );
        self.settle_rearrange();
        let pos = |id: usize, ids: &[usize]| ids.iter().position(|&x| x == id).expect("a permutation");
        let (s2, t2) = (pos(sid, &self.lattice.state_ids), pos(tid, &self.lattice.state_ids));
        let a2 = self.lattice.symbol_index(&label)?;
        let at = self.lattice.offset(s2, a2, t2)?;
        Ok(&self.lattice.edges[at])
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
            .with("focus", self.focus.map(Json::from).unwrap_or(Json::Null))
            .with("focus_node", {
                let (s, a, t) = focus_node(self.lattice.shape(), self.focus);
                Json::numbers(&[s as f64, a as f64, t as f64])
            })
            .with("origin", self.origin().into())
            .with("learn_focus", self.learn_focus.into())
            .with("last_focus", self.last_focus.map(Json::from).unwrap_or(Json::Null))
            .with("skip", self.skip.into())
            .with("skip_margin", self.skip_margin.into())
            .with("skips", (self.skips as f64).into())
            .with("rearrange_every", (self.rearrange_every as f64).into())
            .with("swaps", (self.swaps as f64).into())
            .with(
                "state_order",
                Json::Array(self.lattice.state_ids.iter().map(|&i| i.into()).collect()),
            )
            .with("symbol_order", Json::strings(&self.lattice.symbols))
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
            .with("focus", self.focus.map(Json::from).unwrap_or(Json::Null))
            .with("learn_focus", self.learn_focus.into())
            .with("skip", self.skip.into())
            .with("skip_margin", self.skip_margin.into())
            .with("rearrange_every", (self.rearrange_every as f64).into())
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
            .with("skips", (self.skips as f64).into())
            .with("swaps", (self.swaps as f64).into())
            .with("since_rearrange", (self.since_rearrange as f64).into())
            .with("focus_learner", self.focus_learner.to_json())
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
            focus: match st.get("focus").and_then(Json::as_f64) {
                Some(f) => Some(check_focus(f)?),
                None => None,
            },
            learn_focus: st.bool_or("learn_focus", false),
            focus_learner: match v.get("focus_learner") {
                Some(fl) if !matches!(fl, Json::Null) => FocusLearner::from_json(fl)?,
                _ => FocusLearner::new(lattice.symbols.clone(), 0.05, 0.15),
            },
            last_focus_draw: None,
            last_focus: None,
            skip: st.bool_or("skip", false),
            skip_margin: st.num("skip_margin", 0.0),
            skips: v.num("skips", 0.0) as u64,
            rearrange_every: st.num("rearrange_every", 0.0) as u64,
            swaps: v.num("swaps", 0.0) as u64,
            since_rearrange: v.num("since_rearrange", 0.0) as u64,
            rearrange_due: false,
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

/// One pass's neighbour pairs toward the centre of an axis of `n` nodes, as `(outer, inner)`: from the outside in on
/// the left side, then from the outside in on the right.
fn inward_pairs(n: usize) -> Vec<(usize, usize)> {
    let c = n / 2;
    let mut pairs: Vec<(usize, usize)> = (1..=c).map(|k| (k - 1, k)).collect();
    if n >= 2 {
        pairs.extend((c..n - 1).rev().map(|k| (k + 1, k)));
    }
    pairs
}
