//! The stage game (`DESIGN.md` §4.3, §10): seats sold in a uniform-price auction with a reserve, a congestion
//! game over the lattice's states solved by regret matching, credit divided by sampled Shapley values.
//!
//! Written out from `GTMNN/DESIGN.md` §8-12, which has no code of its own; `CyclicCortex/cortex/routing.py`
//! (bid, allocate at the highest losing bid, settle `base / n` or `-penalty`) is the running precedent.
//! Nothing here knows what a seat *is*: the encoder seats graph nodes (§4.3), a decoder seats edges (§6.3).

use latticefsm::rng::Rng;

/// The slot a seat uses to decline: index `S` of every `S + 1`-slot vector.
pub fn abstain_slot(states: usize) -> usize {
    states
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameSettings {
    /// The base reward for a correct call, split among everyone who made it.
    pub b: f64,
    /// The penalty for a wrong non-abstaining call.  Rewards-only measured worse in every LatticeFSM arm.
    pub lam: f64,
    /// Bids below it are not seated (`cortex/routing.py` uses 0.05; GTMNN 0.01).
    pub reserve: f64,
    /// Seats sold per stage game.
    pub seats: usize,
    /// Regret-matching iterations.
    pub iters: usize,
    /// Shapley permutations, in antithetic pairs; never an exact mode.
    pub permutations: usize,
    /// The smoothing that keeps every aggregate strictly positive so `log P` is finite.
    pub eps: f64,
    /// How much of a seat's cumulative regret survives into the next stage game.
    pub gamma_regret: f64,
    /// A photo finish: `top1 - top2` below it escalates.
    pub margin_tau: f64,
    /// Exploitability above it, unconverged, escalates.
    pub expl_tau: f64,
}

impl Default for GameSettings {
    fn default() -> GameSettings {
        GameSettings {
            b: 1.0,
            lam: 0.5,
            reserve: 0.01,
            seats: 16,
            iters: 64,
            permutations: 32,
            eps: 1e-3,
            gamma_regret: 0.95,
            margin_tau: 0.05,
            expl_tau: 0.05,
        }
    }
}

/// A player as the game sees it: who it is, what it believes, how loud it speaks.
#[derive(Clone, Debug, PartialEq)]
pub struct Seat {
    /// The node (or the edge) this seat is; opaque to the game.
    pub id: usize,
    /// Its own belief over the `S + 1` slots (the last is abstain); rows sum to 1.
    pub belief: Vec<f64>,
    /// `(hits + 1) / (plays + 2)`.
    pub reputation: f64,
    /// `|f(z)|` of a node's sine, 1 for an edge: how loudly it speaks in the aggregate.
    pub voice: f64,
    /// Its cumulative regret as it arrives, warm-started from its last game.
    pub regret: Vec<f64>,
}

impl Seat {
    /// `reputation * (q_max - q_abstain)`, floored at zero: how much better than abstaining the seat thinks its
    /// best call is, discounted by how often it is right.
    pub fn bid(&self) -> f64 {
        let n = self.belief.len();
        let q_abstain = self.belief[n - 1];
        let q_max = self.belief[..n - 1].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (self.reputation * (q_max - q_abstain)).max(0.0)
    }
}

/// Who was seated and at what price.
#[derive(Clone, Debug, PartialEq)]
pub struct Allocation {
    /// Indices into the candidates, highest bid first.
    pub seated: Vec<usize>,
    pub bids: Vec<f64>,
    /// The highest losing bid; 0 when fewer candidates cleared the reserve than there are seats.
    pub price: f64,
    /// How many candidates bid above the reserve.
    pub bidders: usize,
}

/// Seats sold at the highest losing bid (Vickrey: a seat cannot change what it pays by changing its bid, so
/// bidding its true value is dominant).  Ties by reputation, then by id, so the allocation is deterministic.
pub fn allocate(candidates: &[Seat], m: usize, reserve: f64) -> Allocation {
    let mut order: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.bid()))
        .filter(|(_, b)| *b > reserve)
        .collect();
    order.sort_by(|(i, a), (j, b)| {
        b.partial_cmp(a)
            .unwrap()
            .then(
                candidates[*j]
                    .reputation
                    .partial_cmp(&candidates[*i].reputation)
                    .unwrap(),
            )
            .then(candidates[*i].id.cmp(&candidates[*j].id))
    });
    let bidders = order.len();
    let price = if order.len() > m { order[m].1 } else { 0.0 };
    let seated: Vec<usize> = order.iter().take(m).map(|(i, _)| *i).collect();
    let bids: Vec<f64> = order.iter().take(m).map(|(_, b)| *b).collect();
    Allocation {
        seated,
        bids,
        price,
        bidders,
    }
}

/// How a seat is paid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Payoff {
    /// The label is known: `b / n_y` for naming it, `-lam` for naming anything else, 0 for abstaining.
    /// Identical across players and non-increasing in load: an exact potential game, which cannot cycle.
    Correctness { y: usize },
    /// An oracle says which states are right for this step (several may be: every state that accepts exactly
    /// when the prefix is in the language): `b / n_a` for naming a right one, `-lam` for a wrong one, 0 for
    /// abstaining.  Exact potential too, and right-and-rare spreads the seats over the right states.
    Oracle { right: Vec<bool> },
    /// The label is unknown: each seat scores by its own belief.  Player-specific; it may cycle.
    Belief,
    /// The act that followed was refused for a belief-level reason: `-lam` for every seat that named the state
    /// the refusal arrived in, 0 for the rest (§4.4).
    Refused { named: usize },
}

/// The payoff to `seat` for `slot` given the loads (how many seats chose each slot).
pub fn utility(payoff: &Payoff, seat: &Seat, slot: usize, loads: &[usize], g: &GameSettings) -> f64 {
    let abstain = seat.belief.len() - 1;
    if slot == abstain {
        return 0.0;
    }
    match payoff {
        Payoff::Correctness { y } => {
            if slot == *y {
                g.b / loads[slot].max(1) as f64
            } else {
                -g.lam
            }
        }
        Payoff::Oracle { right } => {
            if right.get(slot).copied().unwrap_or(false) {
                g.b / loads[slot].max(1) as f64
            } else {
                -g.lam
            }
        }
        Payoff::Belief => {
            let q = seat.belief[slot];
            q * g.b / loads[slot].max(1) as f64 - (1.0 - q) * g.lam
        }
        Payoff::Refused { named } => {
            if slot == *named {
                -g.lam
            } else {
                0.0
            }
        }
    }
}

impl Payoff {
    /// Which slots are right, when the payoff knows: one-hot for a label, the oracle's mask, none otherwise.
    pub fn right_mask(&self, states: usize) -> Option<Vec<bool>> {
        match self {
            Payoff::Correctness { y } => {
                let mut m = vec![false; states];
                if *y < states {
                    m[*y] = true;
                }
                Some(m)
            }
            Payoff::Oracle { right } => Some(right.clone()),
            _ => None,
        }
    }

    pub fn is_right(&self, slot: usize, states: usize) -> bool {
        self.right_mask(states)
            .map(|m| m.get(slot).copied().unwrap_or(false))
            .unwrap_or(false)
    }
}

/// The outcome of solving one stage game.
#[derive(Clone, Debug, PartialEq)]
pub struct Equilibrium {
    /// Per seat, the time-averaged mixed strategy over its slots.
    pub profile: Vec<Vec<f64>>,
    /// Per seat, a pure slot: the argmax of its profile, ties to the lowest.
    pub actions: Vec<usize>,
    pub iterations: usize,
    pub converged: bool,
    /// 0 when none; else the length of the best-response cycle found after the solve.
    pub cycle: usize,
    /// `max` over seats of (best-response utility - realised utility): 0 at a Nash equilibrium.
    pub exploitability: f64,
}

fn loads_of(actions: &[usize], slots: usize) -> Vec<usize> {
    let mut loads = vec![0usize; slots];
    for &a in actions {
        loads[a] += 1;
    }
    loads
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

/// Regret matching plus (Hart & Mas-Colell), per seat over its slots; the time-average profile is returned
/// (the convergence result is about the average, not the last iterate).  `regrets` are the seats' cumulative
/// regrets, warm-started and written back decayed, so a seat arrives at a new game knowing which of its slots
/// have been under-played.
pub fn solve(seats: &[Seat], payoff: &Payoff, g: &GameSettings, regrets: &mut [Vec<f64>]) -> Equilibrium {
    let n = seats.len();
    if n == 0 {
        return Equilibrium {
            profile: Vec::new(),
            actions: Vec::new(),
            iterations: 0,
            converged: true,
            cycle: 0,
            exploitability: 0.0,
        };
    }
    let slots = seats[0].belief.len();
    for r in regrets.iter_mut() {
        for x in r.iter_mut() {
            *x *= g.gamma_regret;
        }
    }
    // a slot the seat gives no belief to is outside its repertoire and is never played
    let strategy_of = |r: &[f64], prior: &[f64]| -> Vec<f64> {
        let pos: Vec<f64> = r
            .iter()
            .zip(prior)
            .map(|(x, p)| if *p > 0.0 { x.max(0.0) } else { 0.0 })
            .collect();
        let total: f64 = pos.iter().sum();
        if total > 0.0 {
            pos.iter().map(|x| x / total).collect()
        } else {
            prior.to_vec()
        }
    };
    let mut strategies: Vec<Vec<f64>> = (0..n).map(|i| strategy_of(&regrets[i], &seats[i].belief)).collect();
    let mut average: Vec<Vec<f64>> = vec![vec![0.0; slots]; n];
    let mut iterations = 0;
    for _ in 0..g.iters {
        iterations += 1;
        let actions: Vec<usize> = strategies.iter().map(|s| argmax_low(s)).collect();
        let loads = loads_of(&actions, slots);
        for i in 0..n {
            let realised = utility(payoff, &seats[i], actions[i], &loads, g);
            for a in 0..slots {
                // counterfactual: this seat alone moves to `a`
                let mut l = loads.clone();
                l[actions[i]] -= 1;
                l[a] += 1;
                let u = utility(payoff, &seats[i], a, &l, g);
                if seats[i].belief[a] > 0.0 {
                    regrets[i][a] = (regrets[i][a] + (u - realised)).max(0.0);
                }
            }
            strategies[i] = strategy_of(&regrets[i], &seats[i].belief);
            for a in 0..slots {
                average[i][a] += strategies[i][a];
            }
        }
    }
    for row in average.iter_mut() {
        for x in row.iter_mut() {
            *x /= iterations as f64;
        }
    }
    let actions: Vec<usize> = average.iter().map(|s| argmax_low(s)).collect();
    let exploitability = exploitability_of(seats, payoff, &actions, g);
    let cycle = best_response_cycle(seats, payoff, &actions, g, 32);
    Equilibrium {
        profile: average,
        actions,
        iterations,
        converged: exploitability <= g.expl_tau,
        cycle,
        exploitability,
    }
}

/// `max` over seats of the gain a unilateral move would bring: zero at a Nash equilibrium.
pub fn exploitability_of(seats: &[Seat], payoff: &Payoff, actions: &[usize], g: &GameSettings) -> f64 {
    let slots = seats[0].belief.len();
    let loads = loads_of(actions, slots);
    let mut worst: f64 = 0.0;
    for (i, seat) in seats.iter().enumerate() {
        let realised = utility(payoff, seat, actions[i], &loads, g);
        for a in 0..slots {
            if seat.belief[a] <= 0.0 {
                continue;
            }
            let mut l = loads.clone();
            l[actions[i]] -= 1;
            l[a] += 1;
            let gain = utility(payoff, seat, a, &l, g) - realised;
            if gain > worst {
                worst = gain;
            }
        }
    }
    worst
}

/// Best-response dynamics from `start`, round-robin, pushed once per completed sweep: 0 when it settles,
/// else the cycle length - the game's property, not the solver's failure (Milchtaich; `GTMNN/DESIGN.md` §11.1).
pub fn best_response_cycle(seats: &[Seat], payoff: &Payoff, start: &[usize], g: &GameSettings, sweeps: usize) -> usize {
    let slots = seats[0].belief.len();
    let mut actions = start.to_vec();
    let mut seen: Vec<Vec<usize>> = vec![actions.clone()];
    for _ in 0..sweeps {
        let mut moved = false;
        for i in 0..seats.len() {
            let mut loads = loads_of(&actions, slots);
            loads[actions[i]] -= 1;
            let mut best = actions[i];
            let mut best_u = {
                let mut l = loads.clone();
                l[best] += 1;
                utility(payoff, &seats[i], best, &l, g)
            };
            for a in 0..slots {
                if seats[i].belief[a] <= 0.0 {
                    continue;
                }
                let mut l = loads.clone();
                l[a] += 1;
                let u = utility(payoff, &seats[i], a, &l, g);
                if u > best_u + 1e-9 {
                    best = a;
                    best_u = u;
                }
            }
            if best != actions[i] {
                actions[i] = best;
                moved = true;
            }
        }
        if !moved {
            return 0;
        }
        if let Some(first) = seen.iter().position(|s| *s == actions) {
            return seen.len() - first;
        }
        seen.push(actions.clone());
    }
    0
}

/// `P_C(state)`: the aggregate of a coalition over the states (abstain contributes nothing), strictly
/// positive everywhere by `eps`.
pub fn aggregate(seats: &[Seat], profile: &[Vec<f64>], members: &[usize], states: usize, eps: f64) -> Vec<f64> {
    let mut score = vec![eps / states as f64; states];
    for &i in members {
        let w = seats[i].reputation * seats[i].voice;
        for s in 0..states {
            score[s] += w * profile[i][s];
        }
    }
    let total: f64 = score.iter().sum();
    score.iter().map(|x| x / total).collect()
}

/// `v(C) = log P_C(right)`, the mass the coalition's aggregate puts on the right states; the empty coalition
/// is exactly uniform.
pub fn coalition_value(
    seats: &[Seat],
    profile: &[Vec<f64>],
    members: &[usize],
    states: usize,
    right: &[bool],
    eps: f64,
) -> f64 {
    let p = aggregate(seats, profile, members, states, eps);
    p.iter()
        .zip(right)
        .filter(|(_, r)| **r)
        .map(|(x, _)| x)
        .sum::<f64>()
        .max(1e-300)
        .ln()
}

/// Shapley values by sampled permutations in antithetic pairs (each permutation and its reverse).  Efficiency,
/// `sum(phi) = v(N) - v(empty)`, is exact for every permutation because marginal contributions telescope.
/// There is no exact mode: `2^n` is not a per-step cost anywhere (`DESIGN.md` §4.3).
pub fn shapley(
    seats: &[Seat],
    profile: &[Vec<f64>],
    states: usize,
    right: &[bool],
    g: &GameSettings,
    rng: &mut Rng,
) -> Vec<f64> {
    let n = seats.len();
    let mut phi = vec![0.0; n];
    if n == 0 {
        return phi;
    }
    let pairs = (g.permutations / 2).max(1);
    let mut order: Vec<usize> = (0..n).collect();
    for _ in 0..pairs {
        // a Fisher-Yates shuffle from the crate's own generator
        for i in (1..n).rev() {
            let j = rng.randint(0, i);
            order.swap(i, j);
        }
        for pass in 0..2 {
            let perm: Vec<usize> = if pass == 0 {
                order.clone()
            } else {
                order.iter().rev().cloned().collect()
            };
            let mut members: Vec<usize> = Vec::with_capacity(n);
            let mut before = coalition_value(seats, profile, &members, states, right, g.eps);
            for &i in &perm {
                members.push(i);
                let after = coalition_value(seats, profile, &members, states, right, g.eps);
                phi[i] += after - before;
                before = after;
            }
        }
    }
    let draws = (pairs * 2) as f64;
    for x in phi.iter_mut() {
        *x /= draws;
    }
    phi
}

/// The margin `top1 - top2` of a distribution.
pub fn margin(p: &[f64]) -> f64 {
    let mut top1 = f64::NEG_INFINITY;
    let mut top2 = f64::NEG_INFINITY;
    for &x in p {
        if x > top1 {
            top2 = top1;
            top1 = x;
        } else if x > top2 {
            top2 = x;
        }
    }
    if top2 == f64::NEG_INFINITY {
        top1
    } else {
        top1 - top2
    }
}

/// Entropy in bits.
pub fn entropy_bits(p: &[f64]) -> f64 {
    -p.iter().filter(|x| **x > 0.0).map(|x| x * x.log2()).sum::<f64>()
}

/// A uniform belief over `S + 1` slots.
pub fn uniform(states: usize) -> Vec<f64> {
    vec![1.0 / (states + 1) as f64; states + 1]
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}
