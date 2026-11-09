//! Repertoires on edges (`DESIGN.md` §6.3): each edge of an act lattice carries at most `K` output symbols plus
//! abstain; the last few edges of the walk are the seats; they play the congestion game over the output
//! alphabet; sampled Shapley credits each edge.  `GTMNN/DESIGN.md` §7.5's repertoire, re-pointed at edges.

use std::collections::BTreeMap;

use latticefsm::rng::Rng;

use super::{Decoder, Output};
use crate::game::{self, aggregate, allocate, softmax, GameSettings, Payoff, Seat};
use crate::lattice::GameLattice;
use crate::verdict::Verdict;

#[derive(Clone, Debug, PartialEq)]
pub struct RepertoireSettings {
    /// Symbols an edge may name.
    pub k: usize,
    /// How many of the walk's last edges are seated.
    pub window: usize,
    /// Every this many games an edge's repertoire becomes its `K` most credited symbols.
    pub repertoire_every: u64,
    pub lr: f64,
    pub clip: f64,
    pub gamma_rep: f64,
    pub game: GameSettings,
    pub seed: u64,
    /// A newborn repertoire's logits are drawn uniformly in `[-noise, noise]`: it leans somewhere, so it bids.
    pub noise: f64,
    /// A newborn repertoire's abstain logit.
    pub abstain_prior: f64,
}

impl Default for RepertoireSettings {
    fn default() -> RepertoireSettings {
        RepertoireSettings {
            k: 4,
            window: 8,
            repertoire_every: 512,
            lr: 0.1,
            clip: 8.0,
            gamma_rep: 0.999,
            game: GameSettings::default(),
            seed: 1,
            noise: 1.0,
            abstain_prior: -1.0,
        }
    }
}

/// What one edge holds.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeRepertoire {
    /// Global symbol ids, at most `K`.
    pub symbols: Vec<usize>,
    /// One logit per repertoire entry, plus abstain last.
    pub logits: Vec<f64>,
    /// Over the whole alphabet plus abstain.
    pub regret: Vec<f64>,
    pub wealth: f64,
    pub hits: f64,
    pub plays: f64,
    /// Credit received per symbol: what the next repertoire is chosen from.
    pub tally: Vec<f64>,
    pub games: u64,
}

/// The last naming: what `credit` settles.
#[derive(Clone, Debug, PartialEq)]
pub struct Emission {
    pub symbol: Option<usize>,
    pub aggregate: Vec<f64>,
    pub seated: Vec<usize>,
    pub actions: Vec<usize>,
    pub seats: Vec<Seat>,
    pub profile: Vec<Vec<f64>>,
    pub exploitability: f64,
    pub cycle: usize,
    pub bidders: usize,
}

pub struct SymbolDecoder {
    pub sense: String,
    pub alphabet: Vec<String>,
    pub settings: RepertoireSettings,
    pub edges: BTreeMap<usize, EdgeRepertoire>,
    pub last: Option<Emission>,
    rng: Rng,
    pub emissions: u64,
    pub refusals: u64,
}

impl SymbolDecoder {
    pub fn new(sense: &str, alphabet: Vec<String>, settings: RepertoireSettings) -> SymbolDecoder {
        SymbolDecoder {
            sense: sense.to_string(),
            alphabet,
            rng: Rng::new(settings.seed.wrapping_mul(104729).wrapping_add(7)),
            settings,
            edges: BTreeMap::new(),
            last: None,
            emissions: 0,
            refusals: 0,
        }
    }

    fn v(&self) -> usize {
        self.alphabet.len()
    }

    fn fresh(&mut self) -> EdgeRepertoire {
        let v = self.v();
        let k = self.settings.k.min(v);
        let mut symbols: Vec<usize> = Vec::with_capacity(k);
        while symbols.len() < k {
            let s = self.rng.randint(0, v - 1);
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
        symbols.sort_unstable();
        let mut logits: Vec<f64> = (0..k)
            .map(|_| (self.rng.random() - 0.5) * 2.0 * self.settings.noise)
            .collect();
        logits.push(self.settings.abstain_prior);
        EdgeRepertoire {
            symbols,
            logits,
            regret: vec![0.0; v + 1],
            wealth: 0.0,
            hits: 0.0,
            plays: 0.0,
            tally: vec![0.0; v],
            games: 0,
        }
    }

    fn repertoire(&mut self, edge: usize) -> &mut EdgeRepertoire {
        if !self.edges.contains_key(&edge) {
            let r = self.fresh();
            self.edges.insert(edge, r);
        }
        self.edges.get_mut(&edge).unwrap()
    }

    /// The seat an edge is: its belief over the alphabet plus abstain, zero outside its repertoire.
    fn seat(&self, edge: usize) -> Seat {
        let v = self.v();
        let r = &self.edges[&edge];
        let p = softmax(&r.logits);
        let mut belief = vec![0.0; v + 1];
        for (i, &s) in r.symbols.iter().enumerate() {
            belief[s] = p[i];
        }
        belief[v] = p[r.symbols.len()];
        Seat {
            id: edge,
            belief,
            reputation: (r.hits + 1.0) / (r.plays + 2.0),
            voice: 1.0,
            regret: r.regret.clone(),
        }
    }

    /// The seated edges of a path: the last `window` distinct edges.
    pub fn seats_of(&self, path: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        for &e in path.iter().rev() {
            if !out.contains(&e) {
                out.push(e);
            }
            if out.len() >= self.settings.window {
                break;
            }
        }
        out
    }

    /// Play the game over the alphabet on the lattice's current path and return the aggregate (over the
    /// alphabet) and the emission record; `None` symbol when nobody bids.
    pub fn emit(&mut self, path: &[usize]) -> &Emission {
        let v = self.v();
        let g = self.settings.game.clone();
        let edges = self.seats_of(path);
        for &e in &edges {
            self.repertoire(e);
        }
        let candidates: Vec<Seat> = edges.iter().map(|&e| self.seat(e)).collect();
        let alloc = allocate(&candidates, g.seats, g.reserve);
        let seats: Vec<Seat> = alloc.seated.iter().map(|&i| candidates[i].clone()).collect();
        let seated: Vec<usize> = seats.iter().map(|s| s.id).collect();
        let mut regrets: Vec<Vec<f64>> = seats.iter().map(|s| s.regret.clone()).collect();
        let eq = game::solve(&seats, &Payoff::Belief, &g, &mut regrets);
        for (k, &e) in seated.iter().enumerate() {
            let r = self.edges.get_mut(&e).unwrap();
            r.regret = regrets[k].clone();
            r.wealth -= alloc.price;
            r.games += 1;
        }
        let members: Vec<usize> = (0..seats.len()).collect();
        let agg = aggregate(&seats, &eq.profile, &members, v, g.eps);
        let symbol = if seats.is_empty() { None } else { Some(argmax_low(&agg)) };
        self.emissions += 1;
        self.last = Some(Emission {
            symbol,
            aggregate: agg,
            seated,
            actions: eq.actions,
            seats,
            profile: eq.profile,
            exploitability: eq.exploitability,
            cycle: eq.cycle,
            bidders: alloc.bidders,
        });
        self.last.as_ref().unwrap()
    }

    fn learn(&mut self, edge: usize, slot_symbol: usize, phi: f64) {
        if phi == 0.0 {
            return;
        }
        let (lr, clip, v) = (self.settings.lr, self.settings.clip, self.v());
        let r = self.edges.get_mut(&edge).unwrap();
        let p = softmax(&r.logits);
        let local = if slot_symbol == v {
            r.symbols.len()
        } else {
            match r.symbols.iter().position(|&s| s == slot_symbol) {
                Some(i) => i,
                None => return,
            }
        };
        for (k, pk) in p.iter().enumerate() {
            let grad = if k == local { 1.0 - pk } else { -pk };
            r.logits[k] = (r.logits[k] + lr * phi * grad).clamp(-clip, clip);
        }
    }

    /// The world said: `Correctness { y }` when the right symbol is known (the emitted one, when it was
    /// rewarded), `Refused { named }` when an emitted symbol was refused.
    pub fn settle(&mut self, payoff: Payoff) {
        let Some(last) = self.last.clone() else { return };
        self.settle_emission(&last, payoff);
    }

    /// Settle any emission, not only the last: the grade arrives after the game for every move of it.
    pub fn settle_emission(&mut self, last: &Emission, payoff: Payoff) {
        if last.seats.is_empty() {
            return;
        }
        let v = self.v();
        let g = self.settings.game.clone();
        let every = self.settings.repertoire_every;
        match &payoff {
            Payoff::Correctness { .. } | Payoff::Oracle { .. } => {
                let right = payoff.right_mask(v).expect("a payoff with a label");
                let phi = game::shapley(&last.seats, &last.profile, v, &right, &g, &mut self.rng);
                let mut loads = vec![0usize; v + 1];
                for &a in &last.actions {
                    loads[a] += 1;
                }
                for (k, &e) in last.seated.iter().enumerate() {
                    let u = game::utility(&payoff, &last.seats[k], last.actions[k], &loads, &g);
                    let gamma = self.settings.gamma_rep;
                    {
                        let r = self.edges.get_mut(&e).unwrap();
                        r.wealth += u;
                        r.hits *= gamma;
                        r.plays *= gamma;
                        if last.actions[k] != v {
                            r.plays += 1.0;
                            if right[last.actions[k]] {
                                r.hits += 1.0;
                            }
                        }
                        for (sym, is_right) in right.iter().enumerate() {
                            if *is_right {
                                r.tally[sym] += phi[k].max(0.0);
                            }
                        }
                    }
                    self.learn(e, last.actions[k], phi[k]);
                    self.refresh(e, every);
                }
            }
            Payoff::Refused { named } => {
                let named = *named;
                self.refusals += 1;
                for (k, &e) in last.seated.iter().enumerate() {
                    if last.actions[k] == named {
                        let gamma = self.settings.gamma_rep;
                        {
                            let r = self.edges.get_mut(&e).unwrap();
                            r.wealth -= g.lam;
                            r.hits *= gamma;
                            r.plays *= gamma;
                            r.plays += 1.0;
                        }
                        self.learn(e, named, -g.lam);
                    }
                }
            }
            Payoff::Belief => {}
        }
    }

    /// Every `every` games an edge's repertoire becomes its `K` most credited symbols; a symbol entering or
    /// leaving starts from a fresh logit.
    fn refresh(&mut self, edge: usize, every: u64) {
        let k = self.settings.k.min(self.v());
        let r = self.edges.get_mut(&edge).unwrap();
        if every == 0 || r.games % every != 0 || r.games == 0 {
            return;
        }
        let mut ranked: Vec<usize> = (0..r.tally.len()).collect();
        ranked.sort_by(|&a, &b| r.tally[b].partial_cmp(&r.tally[a]).unwrap().then(a.cmp(&b)));
        let mut next: Vec<usize> = ranked.into_iter().take(k).collect();
        next.sort_unstable();
        if next != r.symbols {
            let mut logits = vec![0.0; next.len() + 1];
            for (i, &s) in next.iter().enumerate() {
                if let Some(j) = r.symbols.iter().position(|&x| x == s) {
                    logits[i] = r.logits[j];
                }
            }
            logits[next.len()] = r.logits[r.symbols.len()];
            r.symbols = next;
            r.logits = logits;
        }
    }

    pub fn symbol_name(&self, s: usize) -> &str {
        &self.alphabet[s]
    }
}

impl Decoder for SymbolDecoder {
    fn output_type(&self) -> &'static str {
        "symbol stream"
    }
    fn reads_back(&self) -> &str {
        &self.sense
    }
    fn read(&mut self, lattice: &GameLattice) -> Output {
        let path = lattice.machine.path.clone();
        let e = self.emit(&path);
        match e.symbol {
            Some(s) => Output::Symbols(vec![self.alphabet[s].clone()]),
            None => Output::Abstain("no bidders".to_string()),
        }
    }
    fn credit(&mut self, verdict: &Verdict) -> usize {
        let Some(last) = &self.last else { return 0 };
        let Some(symbol) = last.symbol else { return 0 };
        let n = last.seated.len();
        if verdict.amount > 0.0 {
            self.settle(Payoff::Correctness { y: symbol });
        } else if verdict.amount < 0.0 {
            self.settle(Payoff::Refused { named: symbol });
        }
        n
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
