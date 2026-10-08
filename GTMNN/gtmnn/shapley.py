"""Credit assignment. This module replaces backpropagation.

Shapley (1953) proved that exactly ONE function satisfies efficiency, symmetry,
null player and linearity. Uniqueness is the point: there is no tuning knob here
and no alternative worth comparing against, because any other credit rule
violates one of four properties that are individually non-negotiable.

  efficiency    sum_i phi_i = v(N) - v(0)   credit is conserved; the learning
                signal handed to the population is exactly the log-likelihood it
                earned over a uniform guess. Nothing invented, nothing leaked.
  symmetry      interchangeable micros get identical credit -- no tie broken by
                id, insertion order, or which one the loop reached first.
  null player   a micro that changed nothing takes EXACTLY zero gradient step.
  linearity     credit from two games adds, so a batch of stage games can be
                accumulated before stepping without distorting anyone's share.
"""
import math, random
from array import array
from dataclasses import dataclass


@dataclass
class ShapleyResult:
    phi: array
    permutations: int
    total: float
    v_full: float
    v_empty: float
    efficiency_error: float

    def to_dict(self):
        return {"permutations": self.permutations, "total": round(self.total, 9),
                "v_full": round(self.v_full, 9), "v_empty": round(self.v_empty, 9),
                "efficiency_error": self.efficiency_error,
                "phi": [round(x, 9) for x in self.phi]}


def _contributions(game, profile, pool):
    """Per seat, the (symbol, weight) pairs it adds to the aggregate's score."""
    out = []
    for seat, i in enumerate(game.seats):
        rho = pool.reputation(i)
        row = profile[seat]
        pairs = []
        for s, sym in enumerate(game.ctx.slot_symbol[seat]):
            if sym is None: continue
            w = rho * row[s]
            if w > 0.0: pairs.append((sym, w))
        out.append(pairs)
    return out


def coalition_value(game, profile, pool, y, eps=1e-3):
    """v(S) = log P_S(y), where P_S is `aggregate` restricted to the seats in S.

    v(empty) = log(1/|V|) exactly -- the smoothing makes the empty coalition
    uniform, which is what keeps every v(S) finite."""
    V = game.ctx.alphabet_size
    contrib = _contributions(game, profile, pool)
    base = eps / V

    def v(S):
        score_y, total = base, eps
        for seat in S:
            for sym, w in contrib[seat]:
                total += w
                if sym == y: score_y += w
        return math.log(score_y) - math.log(total)
    return v


def shapley_values(game, profile, pool, y, permutations=32, rng=None,
                   antithetic=True, truncate=0.0, eps=1e-3):
    """Monte-Carlo over permutations, with v maintained INCREMENTALLY.

    Efficiency is exact per permutation, not merely in expectation: the marginal
    contributions along a single permutation telescope, so sum_i (cur - prev) =
    v(N) - v(0) for ANY number of permutations including one. Only the sampling
    variance in the split between micros falls with more permutations; the total
    never moves.

    Antithetic pairs (pi, reverse(pi)) are both valid permutations, so efficiency
    survives them exactly, and the negative correlation between a seat's position
    in one and the other cuts the variance of phi_i at no cost.

    Truncation saves time on games a handful of seats already decide, but it
    BREAKS efficiency -- efficiency_error becomes non-zero and is reported rather
    than hidden. Off by default."""
    rng = rng or random.Random(0)
    n = len(game.seats)
    V = game.ctx.alphabet_size
    phi = array("d", [0.0]) * n
    if n == 0:
        return ShapleyResult(phi, 0, 0.0, 0.0, 0.0, 0.0)

    contrib = _contributions(game, profile, pool)
    base = eps / V
    v_empty = math.log(base) - math.log(eps)          # = log(1/V)

    score_y, total = base, eps
    for pairs in contrib:
        for sym, w in pairs:
            total += w
            if sym == y: score_y += w
    v_full = math.log(score_y) - math.log(total)

    order = list(range(n))
    draws = []
    while len(draws) < permutations:
        rng.shuffle(order)
        draws.append(list(order))
        if antithetic and len(draws) < permutations:
            draws.append(list(reversed(order)))

    for perm in draws:
        sy, tot = base, eps
        prev = v_empty
        for seat in perm:
            for sym, w in contrib[seat]:
                tot += w
                if sym == y: sy += w
            cur = math.log(sy) - math.log(tot)
            phi[seat] += cur - prev
            prev = cur
            if truncate > 0.0 and abs(cur - v_full) < truncate:
                break
    m = float(len(draws))
    for i in range(n): phi[i] /= m
    tot_phi = sum(phi)
    return ShapleyResult(phi, len(draws), tot_phi, v_full, v_empty,
                         abs(tot_phi - (v_full - v_empty)))


def shapley_exact(game, profile, pool, y, eps=1e-3):
    """Closed-form subset-weighted sum. O(2^M) -- for tests and tiny games only."""
    from itertools import combinations
    n = len(game.seats)
    if n > 12: raise ValueError(f"exact Shapley over {n} seats is 2^{n} coalitions")
    v = coalition_value(game, profile, pool, y, eps)
    fact = [1.0] * (n + 1)
    for i in range(1, n + 1): fact[i] = fact[i - 1] * i
    phi = array("d", [0.0]) * n
    others = list(range(n))
    for i in range(n):
        rest = [j for j in others if j != i]
        for k in range(len(rest) + 1):
            w = fact[k] * fact[n - k - 1] / fact[n]
            for S in combinations(rest, k):
                phi[i] += w * (v(set(S) | {i}) - v(set(S)))
    return phi


def banzhaf_values(game, profile, pool, y, samples=64, rng=None, eps=1e-3):
    """Marginal contribution over uniformly random SUBSETS rather than
    permutations. Cheaper per sample and sometimes a better RANKING of
    influence, but it does not satisfy efficiency -- the values do not sum to
    v(N) - v(0) and would silently rescale the population's learning rate with
    the population's size. For analysis; never the learning coefficient."""
    rng = rng or random.Random(0)
    n = len(game.seats)
    v = coalition_value(game, profile, pool, y, eps)
    phi = array("d", [0.0]) * n
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        acc = 0.0
        for _ in range(samples):
            S = {j for j in rest if rng.random() < 0.5}
            acc += v(S | {i}) - v(S)
        phi[i] = acc / samples
    return phi


def counterfactual_marginal(game, profile, pool, y, seat, slot, eps=1e-3):
    """What seat i would have contributed, last in, had it played `slot` instead.

    The null-player axiom is correct and it creates a blind spot: a micro that
    abstains when it does not hold y has changed nothing, so phi is exactly 0
    and it takes exactly no step -- it is never taught that abstaining was
    RIGHT. Measured: q_abstain FALLS over training (0.201 -> 0.163) because the
    only signal a micro ever receives is "push toward y" in the contexts where
    it holds y, and normalisation drags ABSTAIN down with everything else. At
    inference it then names one of its four symbols in every context.

    Regret matching (Hart & Mas-Colell 2000) supplies the missing term: the
    advantage of an action is its payoff minus the payoff of the best
    alternative. With Shapley as the payoff, an abstainer's advantage is
    0 - marginal(playing its best wrong symbol) > 0, which pushes it toward
    ABSTAIN, and a holder's is phi - 0 = phi, unchanged.

    This is the LAST-IN marginal to the full coalition, not a Shapley average
    over orderings: one aggregate recomputation, O(V), exact. That is a
    deliberate approximation -- the counterfactual is a single alternative
    action, not a player whose fair share needs settling."""
    V = game.ctx.alphabet_size
    base = eps / V
    contrib = _contributions(game, profile, pool)
    score_y, total = base, eps
    for k, pairs in enumerate(contrib):
        if k == seat: continue
        for sym, w in pairs:
            total += w
            if sym == y: score_y += w
    v_without = math.log(score_y) - math.log(total)
    sym = game.ctx.slot_symbol[seat][slot]
    if sym is None: return 0.0                      # abstaining contributes nothing
    rho = pool.reputation(game.seats[seat])
    total += rho
    if sym == y: score_y += rho
    return (math.log(score_y) - math.log(total)) - v_without


def regret_advantages(game, profile, pool, y, phi, eps=1e-3):
    """phi_i minus the last-in marginal of seat i's best alternative action.

    For a seat on a symbol, the alternative is ABSTAIN (marginal 0) and the
    advantage is phi_i. For a seat on ABSTAIN, the alternative is its most
    believed symbol, and the advantage is -marginal(that), positive when the
    symbol is wrong and negative when the micro was holding y and stayed quiet.
    """
    K = pool.K
    out = array("d", [0.0]) * len(game.seats)
    for seat, i in enumerate(game.seats):
        row = profile[seat]
        on_abstain = row[K] >= max(row[s] for s in range(K)) if K else True
        if not on_abstain:
            out[seat] = phi[seat]
            continue
        best, bs = None, -1.0
        for s in range(K):
            if game.ctx.slot_symbol[seat][s] is None: continue
            if not game.playable(seat)[s]: continue
            if row[s] > bs or (best is None): best, bs = s, row[s]
        if best is None:
            out[seat] = phi[seat]; continue
        out[seat] = phi[seat] - counterfactual_marginal(game, profile, pool, y, seat, best, eps)
    return out
