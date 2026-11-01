//! The GameRadix encoder (`DESIGN.md` §4): a RadixCyclicNN whose step is a stage game among the nodes of the
//! walk window, naming a state of the lattice.
//!
//! The gram graph is [`radixnet::Graph`] as it is - its reading, splitting, merging, compression, cycles and
//! sentinels.  What a node holds beside its gram is here: a belief over the lattice's `S` states plus abstain,
//! *conditioned on the state the lattice was in* (so the window carries memory the one-token gram cannot), its
//! cumulative regret, its wealth, and the discounted counts its reputation is read from.  No weight is shared
//! between nodes and no gradient crosses one.

use latticefsm::rng::Rng;
use radixnet::{Encoding, Graph, GraphOptions, Unit, BACK, END, FIRST, START, THINK};

use crate::game::{self, aggregate, allocate, softmax, Equilibrium, GameSettings, Payoff, Seat};

#[derive(Clone, Debug, PartialEq)]
pub struct EncoderSettings {
    /// `S`: the lattice's states the encoder names.
    pub states: usize,
    /// `W`: how many positions back the window reaches (the seats are the distinct nodes in it, plus the
    /// current node's children).
    pub window: usize,
    /// The gram size of the text sense (`char:n:1`); 1 reads one unit a node, as the lattice does.
    pub n: usize,
    /// The REINFORCE step on a seat's logits, scaled by its Shapley value.
    pub lr: f64,
    /// Logits are clipped to `[-clip, clip]` so no node becomes certain (`COEFFICIENT_LIMIT` is 8 on an edge).
    pub clip: f64,
    /// Hits and plays decay by it before every update, so reputation is recent.
    pub gamma_rep: f64,
    pub game: GameSettings,
    pub seed: u64,
    /// A newborn node's logits are drawn uniformly in `[-noise, noise]`.
    pub noise: f64,
    /// A newborn node's abstain logit: below zero it speaks until taught otherwise.
    pub abstain_prior: f64,
    pub fallback: Fallback,
}

impl Default for EncoderSettings {
    fn default() -> EncoderSettings {
        EncoderSettings {
            states: 13,
            window: 8,
            n: 1,
            lr: 0.1,
            clip: 8.0,
            gamma_rep: 0.999,
            game: GameSettings::default(),
            seed: 1,
            noise: 0.1,
            abstain_prior: -1.0,
            fallback: Fallback::Reputation,
        }
    }
}

/// What one naming produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Named {
    pub state: usize,
    /// The seated aggregate over the `S` states.
    pub aggregate: Vec<f64>,
    /// The seats (`node * BUCKETS + bucket`), highest bid first.
    pub seated: Vec<usize>,
    /// Per seat (`node * BUCKETS + bucket`), the slot it played (`S` is abstain).
    pub actions: Vec<usize>,
    pub exploitability: f64,
    pub cycle: usize,
    pub converged: bool,
    pub price: f64,
    pub bidders: usize,
    /// `top1 - top2` of the aggregate.
    pub margin: f64,
    /// Nobody cleared the reserve: the state fell back to the lattice's own.
    pub no_bidders: bool,
    /// Per seated node, its Shapley value once settled; empty until then.
    pub phi: Vec<f64>,
    /// The state the lattice was in when this was named.
    pub prev: usize,
    /// The seats as they played, and the profile they played: what `settle` credits against.
    pub seats: Vec<Seat>,
    pub profile: Vec<Vec<f64>>,
}

impl Named {
    /// Whether the meta-player would be asked (`DESIGN.md` §10.4): a cycle, a photo finish, nobody bidding,
    /// or an unconverged solve.
    pub fn triggered(&self, g: &GameSettings) -> bool {
        self.cycle > 0
            || self.no_bidders
            || self.margin < g.margin_tau
            || (!self.converged && self.exploitability > g.expl_tau)
    }
}

/// Where a seat sits in the window: the current gram's node, the one before it, an older one, or a child of
/// the current node.  A node's belief is keyed by its bucket as well as by the lattice's previous state, so a
/// node seated as *current* can say something different from the same node seated as *older*.
pub const BUCKETS: usize = 4;
pub const CURRENT: usize = 0;
pub const PREVIOUS: usize = 1;
pub const OLDER: usize = 2;
pub const CHILD: usize = 3;

/// The nodes a position's walk window seats, each with its bucket.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Window {
    pub nodes: Vec<(usize, usize)>,
}

/// What a naming falls back to when nobody speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The lattice's own previous state: nothing moves.
    Prev,
    /// The highest-reputation seat's own belief, as GTMNN's meta-player does at its depth cap
    /// (`TRUST_REPUTATION`).
    Reputation,
}

pub struct GameGraph {
    pub graph: Graph,
    pub settings: EncoderSettings,
    encoding: Encoding,
    /// `node x bucket x prev-state x (S + 1)`.
    logits: Vec<f64>,
    regret: Vec<f64>,
    wealth: Vec<f64>,
    hits: Vec<f64>,
    plays: Vec<f64>,
    rng: Rng,
    /// How often each state was named: the collapse guard.
    pub named: Vec<u64>,
    pub namings: u64,
    pub thinks: u64,
}

impl GameGraph {
    pub fn new(settings: EncoderSettings) -> Result<GameGraph, String> {
        GameGraph::with_unit(settings, Unit::Chars)
    }

    /// An encoder over any unit of the radix encoding (characters by default; words for a tape of moves).
    pub fn with_unit(settings: EncoderSettings, unit: Unit) -> Result<GameGraph, String> {
        if settings.states < 2 {
            return Err("an encoder names at least two states".to_string());
        }
        let encoding = Encoding::new(unit, settings.n, 1)?;
        let graph = Graph::new(
            settings.seed as i64,
            GraphOptions {
                encoding,
                ..GraphOptions::default()
            },
        )?;
        let mut g = GameGraph {
            graph,
            encoding,
            rng: Rng::new(settings.seed.wrapping_mul(7919).wrapping_add(13)),
            logits: Vec::new(),
            regret: Vec::new(),
            wealth: Vec::new(),
            hits: Vec::new(),
            plays: Vec::new(),
            named: vec![0; settings.states],
            namings: 0,
            thinks: 0,
            settings,
        };
        g.ensure();
        Ok(g)
    }

    pub fn states(&self) -> usize {
        self.settings.states
    }

    fn slots(&self) -> usize {
        self.settings.states + 1
    }

    fn ensure(&mut self) {
        let n = self.graph.num_node_ids();
        let per = BUCKETS * self.settings.states * self.slots();
        if self.logits.len() < n * per {
            // a newborn node bids freely: a small random lean toward some states, abstain disfavoured
            // (GTMNN DESIGN 17.2: newborns are neither advantaged nor protected, and reputation 0.5 is enough
            // to win a seat on an input the incumbents are unsure about)
            let from = self.logits.len();
            self.logits.resize(n * per, 0.0);
            self.regret.resize(n * per, 0.0);
            let slots = self.slots();
            for i in from..n * per {
                self.logits[i] = if i % slots == slots - 1 {
                    self.settings.abstain_prior
                } else {
                    (self.rng.random() - 0.5) * 2.0 * self.settings.noise
                };
            }
        }
        if self.wealth.len() < n {
            self.wealth.resize(n, 0.0);
            self.hits.resize(n, 0.0);
            self.plays.resize(n, 0.0);
        }
    }

    #[inline]
    fn at(&self, node: usize, bucket: usize, prev: usize) -> usize {
        ((node * BUCKETS + bucket) * self.settings.states + prev) * self.slots()
    }

    pub fn reputation(&self, node: usize) -> f64 {
        (self.hits[node] + 1.0) / (self.plays[node] + 2.0)
    }

    pub fn wealth(&self, node: usize) -> f64 {
        self.wealth[node]
    }

    /// A node's belief over the slots, seated in `bucket`, given the lattice was in `prev`.
    pub fn belief(&self, node: usize, bucket: usize, prev: usize) -> Vec<f64> {
        let a = self.at(node, bucket, prev);
        softmax(&self.logits[a..a + self.slots()])
    }

    /// Read a text into the graph (observing its structure and, with `learn`, its counts) and return, per unit of
    /// the text, the node that unit's gram lives in - `START` for the units before the first whole gram and for a
    /// gram the graph does not know when not learning.
    pub fn walk(&mut self, text: &str, learn: bool) -> Vec<usize> {
        let units = self.encoding.len(text);
        let grams = self.encoding.encode(text);
        if learn && !grams.is_empty() {
            let _ = self.graph.observe(&grams, true);
            self.ensure();
        }
        let lead = self.settings.n.saturating_sub(1).min(units);
        let mut nodes = vec![START; lead];
        nodes.extend(
            grams
                .iter()
                .map(|gram| self.graph.lookup(gram).map(|l| l.node).unwrap_or(START)),
        );
        nodes.resize(units, START);
        nodes
    }

    /// The seats at position `i` of a walk: the distinct nodes of the last `W` positions, each in the bucket of
    /// its nearest occurrence (current, previous, older), and the current node's children; never `END`, `BACK`
    /// or `THINK`.
    pub fn window(&self, nodes: &[usize], i: usize) -> Window {
        let mut out: Vec<(usize, usize)> = Vec::new();
        let last = i.min(nodes.len() - 1);
        let from = (last + 1).saturating_sub(self.settings.window);
        for (back, &n) in nodes[from..=last].iter().rev().enumerate() {
            if (n == START || n >= FIRST) && !out.iter().any(|(m, _)| *m == n) {
                out.push((n, back.min(OLDER)));
            }
        }
        let current = nodes[last];
        let children: Vec<(usize, usize)> = self
            .graph
            .children_pairs(current)
            .into_iter()
            .map(|(c, _)| c)
            .filter(|&c| c >= FIRST && c != END && c != BACK && c != THINK && !out.iter().any(|(m, _)| *m == c))
            .map(|c| (c, CHILD))
            .collect();
        out.extend(children);
        Window { nodes: out }
    }

    fn seat(&self, node: usize, bucket: usize, prev: usize) -> Seat {
        let a = self.at(node, bucket, prev);
        Seat {
            id: node * BUCKETS + bucket,
            belief: self.belief(node, bucket, prev),
            reputation: self.reputation(node),
            voice: 1.0,
            regret: self.regret[a..a + self.slots()].to_vec(),
        }
    }

    /// Play the stage game of one position and name a state: the seats are sold, the game is solved under
    /// `payoff` (`Belief` when no label is known yet, which is the usual case: the verdict arrives after the
    /// naming and `settle` credits it), and the aggregate's argmax is the state.  `prev` is the state the
    /// lattice was in; it is also what the naming falls back to when nobody bids.
    pub fn name(&mut self, window: &Window, prev: usize, payoff: Payoff) -> Named {
        self.name_as(window, prev, payoff, false)
    }

    /// `name`, and quiet: the game is played and reported and nothing is written - not a regret, not a
    /// price, not the count of what was named.
    pub fn name_quietly(&mut self, window: &Window, prev: usize, payoff: Payoff) -> Named {
        self.name_as(window, prev, payoff, true)
    }

    fn name_as(&mut self, window: &Window, prev: usize, payoff: Payoff, quiet: bool) -> Named {
        self.ensure();
        let states = self.settings.states;
        let g = self.settings.game.clone();
        let candidates: Vec<Seat> = window.nodes.iter().map(|&(n, b)| self.seat(n, b, prev)).collect();
        let alloc = allocate(&candidates, g.seats, g.reserve);
        let seats: Vec<Seat> = alloc.seated.iter().map(|&i| candidates[i].clone()).collect();
        let seated: Vec<usize> = seats.iter().map(|s| s.id).collect();
        let mut regrets: Vec<Vec<f64>> = seats.iter().map(|s| s.regret.clone()).collect();
        let eq: Equilibrium = game::solve(&seats, &payoff, &g, &mut regrets);
        let members: Vec<usize> = (0..seats.len()).collect();
        let agg = aggregate(&seats, &eq.profile, &members, states, g.eps);
        let no_bidders = seats.is_empty();
        let all_abstain = !no_bidders && eq.actions.iter().all(|&a| a == states);
        let state = if no_bidders || all_abstain {
            match (self.settings.fallback, candidates.is_empty()) {
                (Fallback::Reputation, false) => {
                    // the highest-reputation candidate's own belief, abstain aside
                    let best = candidates
                        .iter()
                        .max_by(|x, y| x.reputation.partial_cmp(&y.reputation).unwrap().then(y.id.cmp(&x.id)))
                        .unwrap();
                    argmax_low(&best.belief[..states])
                }
                _ => prev,
            }
        } else {
            argmax_low(&agg)
        };
        let margin = game::margin(&agg);
        if !quiet {
            let slots = self.slots();
            for (k, &seat) in seated.iter().enumerate() {
                let a = self.at(seat / BUCKETS, seat % BUCKETS, prev);
                self.regret[a..a + slots].copy_from_slice(&regrets[k]);
                self.wealth[seat / BUCKETS] -= alloc.price;
            }
            self.named[state] += 1;
            self.namings += 1;
        }
        Named {
            state,
            aggregate: agg,
            seated,
            actions: eq.actions,
            exploitability: eq.exploitability,
            cycle: eq.cycle,
            converged: eq.converged,
            price: alloc.price,
            bidders: alloc.bidders,
            margin,
            no_bidders: no_bidders || all_abstain,
            phi: Vec::new(),
            prev,
            seats,
            profile: eq.profile,
        }
    }

    /// The verdict arrived: credit the seats of a naming.  With `Correctness { y }` each seat takes its Shapley
    /// value against `v(C) = log P_C(y)` as the advantage of one step on its own logits, is paid its utility,
    /// and its reputation moves; with `Refused { named }` every seat that named it is charged `-lam` (§4.4);
    /// `Belief` settles nothing.
    pub fn settle(&mut self, named: &mut Named, payoff: Payoff) {
        let states = self.settings.states;
        let g = self.settings.game.clone();
        let prev = named.prev;
        if named.seats.is_empty() {
            return;
        }
        match &payoff {
            Payoff::Correctness { .. } | Payoff::Oracle { .. } => {
                let right = payoff.right_mask(states).expect("a payoff with a label");
                let phi = game::shapley(&named.seats, &named.profile, states, &right, &g, &mut self.rng);
                let mut loads = vec![0usize; states + 1];
                for &a in &named.actions {
                    loads[a] += 1;
                }
                for (k, &seat) in named.seated.iter().enumerate() {
                    let (node, bucket) = (seat / BUCKETS, seat % BUCKETS);
                    let u = game::utility(&payoff, &named.seats[k], named.actions[k], &loads, &g);
                    self.wealth[node] += u;
                    self.hits[node] *= self.settings.gamma_rep;
                    self.plays[node] *= self.settings.gamma_rep;
                    if named.actions[k] != states {
                        self.plays[node] += 1.0;
                        if right[named.actions[k]] {
                            self.hits[node] += 1.0;
                        }
                    }
                    self.learn(node, bucket, prev, named.actions[k], phi[k]);
                }
                named.phi = phi;
            }
            Payoff::Refused { named: wrong } => {
                let wrong = *wrong;
                for (k, &seat) in named.seated.iter().enumerate() {
                    let (node, bucket) = (seat / BUCKETS, seat % BUCKETS);
                    if named.actions[k] == wrong {
                        self.wealth[node] -= g.lam;
                        self.hits[node] *= self.settings.gamma_rep;
                        self.plays[node] *= self.settings.gamma_rep;
                        self.plays[node] += 1.0;
                        self.learn(node, bucket, prev, named.actions[k], -g.lam);
                    }
                }
            }
            Payoff::Belief => {}
        }
    }

    /// One REINFORCE step on a seat's logits with `phi` as the advantage: `d log pi(slot) = onehot - pi`.
    fn learn(&mut self, node: usize, bucket: usize, prev: usize, slot: usize, phi: f64) {
        if phi == 0.0 {
            return;
        }
        let a = self.at(node, bucket, prev);
        let p = self.belief(node, bucket, prev);
        let (lr, clip) = (self.settings.lr, self.settings.clip);
        for (k, pk) in p.iter().enumerate() {
            let grad = if k == slot { 1.0 - pk } else { -pk };
            let x = &mut self.logits[a + k];
            *x = (*x + lr * phi * grad).clamp(-clip, clip);
        }
    }

    /// Where the graph learned to stop and think: a cycle at `node` (`graph.observe_think`).
    pub fn think_at(&mut self, node: usize) {
        if node >= FIRST && self.graph.observe_think(node, 1.0).is_ok() {
            self.thinks += 1;
        }
    }

    /// The entropy in bits of the states named so far: a collapse guard (0 means one state always).
    pub fn diversity(&self) -> f64 {
        let total: u64 = self.named.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let p: Vec<f64> = self.named.iter().map(|&c| c as f64 / total as f64).collect();
        game::entropy_bits(&p)
    }

    /// Mean wealth and its Gini coefficient over the nodes that ever played: the specialisation readout.
    pub fn wealth_stats(&self) -> (f64, f64) {
        let mut w: Vec<f64> = (FIRST..self.wealth.len())
            .filter(|&i| self.plays[i] > 0.0)
            .map(|i| self.wealth[i])
            .collect();
        if w.is_empty() {
            return (0.0, 0.0);
        }
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = w.len() as f64;
        let shift = w[0].min(0.0);
        let shifted: Vec<f64> = w.iter().map(|x| x - shift).collect();
        let total: f64 = shifted.iter().sum();
        if total <= 0.0 {
            return (mean, 0.0);
        }
        let mut acc = 0.0;
        for (i, x) in shifted.iter().enumerate() {
            acc += (2.0 * (i as f64 + 1.0) - n - 1.0) * x;
        }
        (mean, acc / (n * total))
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }
}

fn argmax_low(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
