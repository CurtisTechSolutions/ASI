//! The organism wired (`DESIGN.md` §2, §9): a text sense read by the GameRadix encoder, naming the states of a
//! sense lattice credited by the oracle per step; and, for a game, the same encoder over the move tape naming
//! the states of an act lattice whose row is the forward model, with an action decoder proposing moves the
//! referee replays.
//!
//! A string can also be read with its trace (`Body::trace`, quiet; `Body::run_traced`, learning): per character, the
//! window's candidates with their bids and the slots they played, the auction, the equilibrium, the aggregate,
//! the named state beside the lattice's prediction, and what would have asked the meta-player (`DESIGN.md` §4.3,
//! §6.2's *trace* output, §16).

use latticefsm::languages::{random_string, Language};
use latticefsm::machine::Settings;
use latticefsm::rng::Rng;

use crate::clock::SharedClock;
use crate::credit::{self, CreditSettings};
use crate::decoders::action::ActionDecoder;
use crate::decoders::repertoire::{RepertoireSettings, SymbolDecoder};
use crate::efference::{Efference, Return};
use crate::encoder::{bucket_name, EncoderSettings, GameGraph, Named, Window, BUCKETS};
use crate::forward::{self, Agreement};
use crate::game::{Payoff, Seat};
use crate::lattice::{GameLattice, Kind};
use crate::referee::{LanguageOracle, Nim, NimMove, NimState};
use crate::verdict::Verdict;

#[derive(Clone, Debug)]
pub struct BodySettings {
    pub encoder: EncoderSettings,
    pub lattice: Settings,
    pub credit: CreditSettings,
    /// The encoder's named state is where the lattice continues from (the design); off, the lattice continues
    /// from its own prediction and the encoder only names beside it (the plain-lattice control).
    pub encoder_sets_state: bool,
    /// The lattice's row is taught by a lesson when its prediction and the next named state disagree
    /// (`DESIGN.md` §8.2); off, the lattice is credited by the oracle alone.
    pub forward_lessons: bool,
    /// The oracle punishes a wrong step (`-reward`); off, rewards only - the measured-worse control.
    pub punish: bool,
    /// Credit the whole run by its final answer, discounted (the LatticeFSM default), instead of per step.
    pub whole_run: bool,
    /// The training label is the whole set of right states (`Payoff::Oracle`, right-and-rare among them)
    /// instead of one state: the lattice's prediction when it is right, else the right state its row ranks
    /// highest (`DESIGN.md` §4.4: the state whose edge was credited).
    pub label_set: bool,
    pub seed: u64,
}

impl Default for BodySettings {
    fn default() -> BodySettings {
        BodySettings {
            encoder: EncoderSettings::default(),
            lattice: Settings::default(),
            credit: CreditSettings::default(),
            encoder_sets_state: true,
            forward_lessons: false,
            punish: true,
            whole_run: false,
            label_set: false,
            seed: 1,
        }
    }
}

/// What a string came to: the encoder's final named state, the lattice's own final prediction, and the walk.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification {
    pub encoder: bool,
    pub lattice: bool,
    pub named: Vec<usize>,
    pub predicted: Vec<usize>,
    /// Mean `-log2 p` of the lattice's chosen edges: its bits per symbol.
    pub lattice_bits: f64,
    /// Mean entropy in bits of the encoder's aggregates.
    pub encoder_bits: f64,
    pub triggers: usize,
}

/// One candidate of a position's window as the trace shows it: seated or not, its bid, its belief over the `S`
/// states plus abstain (keyed by its bucket and the lattice's previous state), and - seated - the slot it played
/// and, once a learning step settled it, its Shapley value.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSeat {
    pub node: usize,
    /// Where it sits in the window: `current`, `previous`, `older` or `child`.
    pub bucket: &'static str,
    /// The node's gram (`graph.label`).
    pub label: String,
    pub seated: bool,
    pub bid: f64,
    pub reputation: f64,
    pub wealth: f64,
    /// `S + 1` slots; the last is abstain.
    pub belief: Vec<f64>,
    /// The slot it played (`S` is abstain); `None` when it was not seated.
    pub action: Option<usize>,
    /// Its Shapley value; `None` on a quiet step or an unseated candidate.
    pub phi: Option<f64>,
}

/// One character of a read: the stage game that named a state, and the lattice's prediction beside it.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceStep {
    pub symbol: String,
    /// The state the lattice was in: the row's source, and the key of every belief.
    pub prev: usize,
    /// The seated seats in bid order, then the candidates that were not seated.
    pub seats: Vec<TraceSeat>,
    pub aggregate: Vec<f64>,
    pub named: usize,
    pub margin: f64,
    pub price: f64,
    pub bidders: usize,
    pub no_bidders: bool,
    pub exploitability: f64,
    pub cycle: usize,
    pub converged: bool,
    /// The lattice's own target from `prev` on this symbol, and the probability it had.
    pub predicted: usize,
    pub probability: f64,
    /// Why the meta-player would be asked (`Named::triggered`): `cycle`, `photo finish`, `no bidders`,
    /// `unconverged`; empty when nothing would.
    pub triggers: Vec<&'static str>,
    /// Whether the step was credited (a learning read).
    pub learned: bool,
}

pub struct Body {
    pub encoder: GameGraph,
    pub lattice: GameLattice,
    pub oracle: LanguageOracle,
    pub clock: SharedClock,
    pub efference: Efference,
    pub agreement: Agreement,
    pub settings: BodySettings,
    pub lessons: u64,
    pub episodes: u64,
}

impl Body {
    /// A body that reads one of the four languages: a text sense over `a` and `b`, an encoder naming `S`
    /// states, a sense lattice of `S` states whose accepting set is the language's.
    pub fn for_language(lang: &Language, settings: BodySettings) -> Result<Body, String> {
        let mut enc = settings.encoder.clone();
        enc.seed = settings.seed;
        let states = enc.states;
        let mut ls = settings.lattice.clone();
        ls.seed = settings.seed;
        let alphabet: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let mut lattice = GameLattice::new("text", Kind::Sense, states, &alphabet, lang.accepting, ls)?;
        // measurements are greedy and draw nothing: of tied targets the lowest is taken (network.py's rule)
        lattice.machine.greedy_first = true;
        Ok(Body {
            encoder: GameGraph::new(enc)?,
            lattice,
            oracle: LanguageOracle {
                language: *lang,
                reward: 1.0,
                punish: settings.punish,
            },
            clock: SharedClock::default(),
            efference: Efference::default(),
            agreement: Agreement::default(),
            settings,
            lessons: 0,
            episodes: 0,
        })
    }

    fn accepts(&self, state: usize) -> bool {
        self.lattice.machine.lattice.states[state].accepting
    }

    /// Read a string: the lattice steps from the state the encoder named, the encoder names the next state
    /// from the walk window, and - learning - every step is credited by the oracle.  Quiet, nothing moves.
    pub fn run(&mut self, text: &str, learn: bool) -> Classification {
        self.read(text, learn, false).0
    }

    /// Quiet, with the trace: what `classify` answers and, per character, the game that named the state.  Nothing
    /// moves - not an edge, not a belief, not the machine's state or path, not the agreement count.
    pub fn trace(&mut self, text: &str) -> (Classification, Vec<TraceStep>) {
        let state = self.lattice.machine.state;
        let path = self.lattice.machine.path.clone();
        let agreement = self.agreement.clone();
        let out = self.read(text, false, true);
        self.lattice.machine.state = state;
        self.lattice.machine.path = path;
        self.agreement = agreement;
        out
    }

    /// One learning episode on `text`, with the trace (each seat's Shapley value included).
    pub fn run_traced(&mut self, text: &str) -> (Classification, Vec<TraceStep>) {
        self.read(text, true, true)
    }

    /// The window's candidates at one naming, seated first (as `named.seats` holds them), then the rest.
    fn trace_seats(&self, window: &Window, named: &Named) -> Vec<TraceSeat> {
        let mut out: Vec<TraceSeat> = Vec::with_capacity(window.nodes.len());
        for (k, seat) in named.seats.iter().enumerate() {
            let (node, bucket) = (seat.id / BUCKETS, seat.id % BUCKETS);
            out.push(TraceSeat {
                node,
                bucket: bucket_name(bucket),
                label: self.encoder.graph.label(node).to_string(),
                seated: true,
                bid: seat.bid(),
                reputation: seat.reputation,
                wealth: self.encoder.wealth(node),
                belief: seat.belief.clone(),
                action: named.actions.get(k).copied(),
                phi: None,
            });
        }
        for &(node, bucket) in &window.nodes {
            if named.seated.contains(&(node * BUCKETS + bucket)) {
                continue;
            }
            let belief = self.encoder.belief(node, bucket, named.prev);
            let reputation = self.encoder.reputation(node);
            let bid = Seat {
                id: node * BUCKETS + bucket,
                belief: belief.clone(),
                reputation,
                voice: 1.0,
                regret: Vec::new(),
            }
            .bid();
            out.push(TraceSeat {
                node,
                bucket: bucket_name(bucket),
                label: self.encoder.graph.label(node).to_string(),
                seated: false,
                bid,
                reputation,
                wealth: self.encoder.wealth(node),
                belief,
                action: None,
                phi: None,
            });
        }
        out
    }

    /// `run`, recording the trace when `record`.
    fn read(&mut self, text: &str, learn: bool, record: bool) -> (Classification, Vec<TraceStep>) {
        let mut steps: Vec<TraceStep> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let nodes = self.encoder.walk(text, learn);
        let m = &mut self.lattice.machine;
        m.reset();
        let mut s = m.start;
        let mut named_states = Vec::with_capacity(chars.len());
        let mut predicted = Vec::with_capacity(chars.len());
        let mut lattice_verdicts: Vec<Verdict> = Vec::new();
        let mut lattice_bits = 0.0;
        let mut encoder_bits = 0.0;
        let mut triggers = 0;
        let mut last_t = s;
        for (i, ch) in chars.iter().enumerate() {
            let a = self
                .lattice
                .machine
                .symbol(&ch.to_string())
                .expect("a symbol of the alphabet");
            self.lattice.machine.state = s;
            let temperature = if learn { None } else { Some(0.0) };
            let t = self.lattice.machine.step(a, None, temperature, !learn);
            lattice_bits -= t.probability.max(1e-12).log2();
            let window = self.encoder.window(&nodes, i);
            // learning, the oracle's label is known before the game: every state that accepts exactly when the
            // prefix is in the language is right (an exact potential game); at play the seats play their beliefs
            let prefix: String = chars[..=i].iter().collect();
            let payoff = if learn {
                let member = self.oracle.member(&prefix);
                let right: Vec<bool> = (0..self.lattice.states())
                    .map(|st| self.accepts(st) == member)
                    .collect();
                if self.settings.label_set {
                    Payoff::Oracle { right }
                } else if right[t.target] {
                    Payoff::Correctness { y: t.target }
                } else {
                    // the right state the lattice's own row ranks highest: what the oracle would have credited
                    let p = self.lattice.machine.probabilities(s, a, None, Some(1.0));
                    let mut best = None;
                    for (st, ok) in right.iter().enumerate() {
                        if *ok && best.map(|b: usize| p[st] > p[b]).unwrap_or(true) {
                            best = Some(st);
                        }
                    }
                    Payoff::Correctness {
                        y: best.unwrap_or(t.target),
                    }
                }
            } else {
                Payoff::Belief
            };
            let mut named: Named = if learn {
                self.encoder.name(&window, s, payoff.clone())
            } else {
                self.encoder.name_quietly(&window, s, payoff.clone())
            };
            encoder_bits += crate::game::entropy_bits(&named.aggregate);
            if named.triggered(&self.settings.encoder.game) {
                triggers += 1;
            }
            let mut step = record.then(|| {
                let g = &self.settings.encoder.game;
                let mut why = Vec::new();
                if named.cycle > 0 {
                    why.push("cycle");
                }
                if named.margin < g.margin_tau {
                    why.push("photo finish");
                }
                if named.no_bidders {
                    why.push("no bidders");
                }
                if !named.converged && named.exploitability > g.expl_tau {
                    why.push("unconverged");
                }
                TraceStep {
                    symbol: ch.to_string(),
                    prev: s,
                    seats: self.trace_seats(&window, &named),
                    aggregate: named.aggregate.clone(),
                    named: named.state,
                    margin: named.margin,
                    price: named.price,
                    bidders: named.bidders,
                    no_bidders: named.no_bidders,
                    exploitability: named.exploitability,
                    cycle: named.cycle,
                    converged: named.converged,
                    predicted: t.target,
                    probability: t.probability,
                    triggers: why,
                    learned: learn,
                }
            });
            let next = if self.settings.encoder_sets_state {
                named.state
            } else {
                t.target
            };
            if learn {
                lattice_verdicts.push(self.oracle.judge(&prefix, self.accepts(t.target), i));
                self.encoder.settle(&mut named, payoff);
                if let Some(step) = step.as_mut() {
                    for (k, seat) in step.seats.iter_mut().filter(|x| x.seated).enumerate() {
                        seat.phi = named.phi.get(k).copied();
                    }
                }
                if named.cycle > 0 {
                    self.encoder.think_at(nodes[i]);
                }
                if self.settings.forward_lessons && self.settings.encoder_sets_state {
                    if let Ok(true) = forward::learn(&mut self.lattice.machine, s, a, t.target, next, 1.0) {
                        self.lessons += 1;
                    }
                }
            }
            self.agreement.note(t.target == named.state);
            named_states.push(named.state);
            predicted.push(t.target);
            last_t = t.target;
            s = next;
            steps.extend(step);
        }
        if learn {
            self.episodes += 1;
            if self.settings.whole_run {
                let right = self.accepts(s) == self.oracle.member(text);
                let amount = if right {
                    1.0
                } else if self.settings.punish {
                    -1.0
                } else {
                    0.0
                };
                self.lattice.machine.credit(amount);
            } else {
                let returns: Vec<Return> = Vec::new();
                credit::land(
                    &mut self.lattice.machine,
                    &lattice_verdicts,
                    &returns,
                    self.settings.credit,
                );
            }
            self.clock.stepped(chars.len() as i64);
        }
        let n = chars.len().max(1) as f64;
        let classification = Classification {
            encoder: self.accepts(s),
            lattice: self.accepts(last_t),
            named: named_states,
            predicted,
            lattice_bits: lattice_bits / n,
            encoder_bits: encoder_bits / n,
            triggers,
        };
        (classification, steps)
    }

    /// Quiet: the verdict on a string, nothing moved.
    pub fn classify(&mut self, text: &str) -> Classification {
        self.run(text, false)
    }

    /// `episodes` random strings of the language, learned.
    pub fn teach(&mut self, episodes: usize, rng: &mut Rng, max_length: usize) {
        for _ in 0..episodes {
            let s = random_string(rng, max_length);
            self.run(&s, true);
        }
    }

    /// Accuracy of the encoder's verdict and of the lattice's own on labelled examples; nothing moves.
    pub fn accuracy(&mut self, examples: &[(String, bool)]) -> (f64, f64) {
        if examples.is_empty() {
            return (0.0, 0.0);
        }
        let mut enc = 0;
        let mut lat = 0;
        for (s, label) in examples {
            let c = self.classify(s);
            if c.encoder == *label {
                enc += 1;
            }
            if c.lattice == *label {
                lat += 1;
            }
        }
        (enc as f64 / examples.len() as f64, lat as f64 / examples.len() as f64)
    }
}

// ---- a body that plays Nim ---------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NimSettings {
    pub encoder: EncoderSettings,
    pub lattice: Settings,
    pub repertoire: RepertoireSettings,
    /// Our moves of a won game are credited as right at the game's end (the grade, later); off, only the
    /// referee's refusals credit the decoder.
    pub grade: bool,
    pub seed: u64,
}

impl Default for NimSettings {
    fn default() -> NimSettings {
        let mut encoder = EncoderSettings::default();
        encoder.n = 1;
        NimSettings {
            encoder,
            lattice: Settings::default(),
            repertoire: RepertoireSettings::default(),
            grade: true,
            seed: 1,
        }
    }
}

/// One game's record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameRecord {
    pub won: bool,
    pub plies: usize,
    pub our_moves: usize,
    pub first_try: usize,
    pub refusals: usize,
    pub abstained: usize,
    pub optimal: usize,
    pub lessons: usize,
}

pub struct NimBody {
    pub encoder: GameGraph,
    pub lattice: GameLattice,
    pub decoder: ActionDecoder,
    pub efference: Efference,
    pub agreement: Agreement,
    pub settings: NimSettings,
    rng: Rng,
    /// `(named state, nim-sum is zero)` counts: for `I(state; truth)`.
    pub joint: Vec<[u64; 2]>,
    pub games: u64,
}

impl NimBody {
    pub fn new(settings: NimSettings) -> Result<NimBody, String> {
        let mut enc = settings.encoder.clone();
        enc.seed = settings.seed;
        let mut encoding_enc = enc.clone();
        // the tape is words: one move a unit
        encoding_enc.n = 1;
        let states = enc.states;
        let mut ls = settings.lattice.clone();
        ls.seed = settings.seed;
        let alphabet = Nim::alphabet();
        let lattice = GameLattice::new("nim", Kind::Act, states, &alphabet, &[], ls)?;
        let mut rs = settings.repertoire.clone();
        rs.seed = settings.seed;
        let proposer = SymbolDecoder::new("nim", alphabet, rs);
        Ok(NimBody {
            encoder: GameGraph::new_words(encoding_enc)?,
            lattice,
            decoder: ActionDecoder::new(proposer),
            efference: Efference::default(),
            agreement: Agreement::default(),
            rng: Rng::new(settings.seed.wrapping_mul(31).wrapping_add(3)),
            settings,
            joint: vec![[0, 0]; states],
            games: 0,
        })
    }

    /// Name the state of a history (the tape of moves so far, as words), learning the tape's structure.
    fn name(&mut self, tape: &[String], prev: usize) -> Named {
        let text = tape.join(" ");
        let nodes = if tape.is_empty() {
            vec![radixnet::START]
        } else {
            self.encoder.walk(&text, true)
        };
        let window = self.encoder.window(&nodes, nodes.len() - 1);
        self.encoder.name(&window, prev, Payoff::Belief)
    }

    /// One game against a random opponent (we move first when `first`).  Babbling: our moves are random too,
    /// and only the forward model learns.
    pub fn play(&mut self, first: bool, babble: bool) -> GameRecord {
        let mut state = Nim::initial();
        let mut tape: Vec<String> = Vec::new();
        let mut rec = GameRecord::default();
        let mut s = self.lattice.machine.start;
        self.lattice.machine.reset();
        let mut ours = first;
        let mut our_emissions: Vec<(crate::decoders::repertoire::Emission, usize)> = Vec::new();
        let mut step = 0;
        while !Nim::is_terminal(state) {
            let mut first_refused = false;
            let mv: NimMove = if ours {
                rec.our_moves += 1;
                let chosen = if babble {
                    Nim::random_move(state, &mut self.rng)
                } else {
                    let st = state;
                    let proposal = self.decoder.propose(&self.lattice, step, |symbol, step| {
                        Nim::refuse(st, NimMove::from_index(symbol), step)
                    });
                    rec.refusals += proposal.refusals.len();
                    first_refused = !proposal.refusals.is_empty();
                    if proposal.first_try {
                        rec.first_try += 1;
                    }
                    match proposal.accepted {
                        Some(sym) => {
                            if let Some(e) = proposal.emission {
                                our_emissions.push((e, sym));
                            }
                            Some(NimMove::from_index(sym))
                        }
                        None => {
                            rec.abstained += 1;
                            Nim::random_move(state, &mut self.rng)
                        }
                    }
                };
                let mv = chosen.expect("a legal move exists");
                if Nim::optimal(state).contains(&mv) {
                    rec.optimal += 1;
                }
                mv
            } else {
                Nim::random_move(state, &mut self.rng).expect("a legal move exists")
            };
            // the act enters the act lattice as its symbol: the forward model predicts the next state
            let a = self
                .lattice
                .machine
                .symbol(&mv.symbol())
                .expect("a move of the alphabet");
            self.lattice.machine.state = s;
            let t = self.lattice.machine.step(a, None, None, false);
            self.efference
                .copy(&mv.symbol(), "nim", self.lattice.machine.clock, 2, Some(t.target));
            // the world replays it
            let before = state;
            state = Nim::apply(state, mv);
            tape.push(mv.symbol());
            let next_state = state;
            // the encoder reads the replayed history and names where we are now (exafferent: the world's position)
            let mut named = self.name(&tape, s);
            let (ret, _) = self.efference.observe(&mv.symbol(), "nim", |copy, got| copy == got);
            debug_assert_eq!(ret, Return::Reafferent);
            self.agreement.note(t.target == named.state);
            if forward::learn(&mut self.lattice.machine, s, a, t.target, named.state, 1.0).unwrap_or(false) {
                rec.lessons += 1;
            }
            self.lattice.machine.state = named.state;
            self.joint[named.state][usize::from(Nim::nim_sum(next_state) == 0)] += 1;
            // the encoder's label (DESIGN.md 4.4): the state whose edge was credited - the forward model's
            // prediction when the act that followed from it was accepted first try; a refusal for a
            // belief-level reason charges the seats that named the state it arrived in
            if !babble {
                if ours && first_refused {
                    let wrong = s;
                    let mut prev_named = Named {
                        state: wrong,
                        ..named.clone()
                    };
                    let _ = &mut prev_named;
                }
                let payoff = if ours && first_refused {
                    Payoff::Refused { named: named.state }
                } else {
                    Payoff::Correctness { y: t.target }
                };
                self.encoder.settle(&mut named, payoff);
            }
            let _ = before;
            s = next_state_marker(&self.lattice);
            ours = !ours;
            step += 1;
            rec.plies += 1;
        }
        rec.won = Nim::winner(state) == Some(if first { 0 } else { 1 });
        if !babble && self.settings.grade && rec.won {
            // the grade, later: a won game credits every move of ours as right, once more
            for (e, sym) in &our_emissions {
                self.decoder
                    .proposer
                    .settle_emission(e, Payoff::Correctness { y: *sym });
            }
        }
        self.games += 1;
        rec
    }

    /// `I(named state; nim-sum is zero)` in bits over everything named so far, and the states' coverage.
    pub fn information(&self) -> (f64, usize) {
        let total: u64 = self.joint.iter().map(|r| r[0] + r[1]).sum();
        if total == 0 {
            return (0.0, 0);
        }
        let n = total as f64;
        let px: Vec<f64> = self.joint.iter().map(|r| (r[0] + r[1]) as f64 / n).collect();
        let py = [
            self.joint.iter().map(|r| r[0]).sum::<u64>() as f64 / n,
            self.joint.iter().map(|r| r[1]).sum::<u64>() as f64 / n,
        ];
        let mut mi = 0.0;
        for (x, r) in self.joint.iter().enumerate() {
            for (y, &c) in r.iter().enumerate() {
                if c > 0 {
                    let pxy = c as f64 / n;
                    mi += pxy * (pxy / (px[x] * py[y])).log2();
                }
            }
        }
        (mi, px.iter().filter(|p| **p > 0.0).count())
    }
}

fn next_state_marker(lattice: &GameLattice) -> usize {
    lattice.machine.state
}

impl GameGraph {
    /// An encoder whose text sense reads words - one move of a tape a unit.
    pub fn new_words(settings: EncoderSettings) -> Result<GameGraph, String> {
        GameGraph::with_unit(settings, radixnet::Unit::Words)
    }
}

impl NimState {
    pub fn describe(&self) -> String {
        format!("{:?} to move {}", self.heaps, self.mover)
    }
}
