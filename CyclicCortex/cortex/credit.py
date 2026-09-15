"""Shapley credit over REGIONS.

GTMNN divided credit among 4096 micros by Monte-Carlo Shapley because exact
enumeration of 2^4096 coalitions is absurd. CyclicCortex has a dozen regions, and
that changes the arithmetic: 2^12 = 4096 coalitions is a loop, not a sampling
problem. Exact Shapley becomes the default and Monte-Carlo the fallback, which
was the specific benefit claimed for moving from micros to regions.

The coalition value is the ensemble's log-likelihood of the correct answer, so
Shapley's efficiency axiom means the credits sum to exactly the log-likelihood
the whole set of regions earned over a uniform guess -- nothing invented, nothing
lost.
"""
import math, random
from itertools import combinations

def coverage(region, game):
    """Fraction of a game's mechanics this region's vocabulary can encode. A
    region with zero coverage encodes the game as all-zeros and is not merely
    bad at it -- it cannot represent it at all."""
    spec = set(game.spec)
    if not spec: return 0.0
    return len(spec & set(region.vocab.slot)) / len(spec)

def ensemble(regions, cortex, game, state, mv, weights=None):
    """Reputation-weighted mean of (p_valid, grade) across regions.

    Regions share no weights, so this is the ONLY channel through which one
    region's learning can reach another's game."""
    num_v = num_g = den = 0.0
    for i, r in enumerate(regions):
        if r.net is None: continue
        w = (weights[i] if weights else r.reputation()) * coverage(r, game)
        if w <= 0.0: continue
        pv, gr = r.net.predict(cortex.encode(r, game, state, mv))
        num_v += w * pv; num_g += w * gr; den += w
    if den == 0.0: return 0.5, 0.0
    return num_v / den, num_g / den

def value_fn(cortex, game, samples, eps=1e-3):
    """v(S) = mean log-likelihood the coalition assigns to the true legality.
    v({}) = log(0.5): knowing nothing, a coin. Cached by coalition."""
    cache = {}
    base = math.log(0.5)
    def v(S):
        key = frozenset(r.id for r in S)
        if key in cache: return cache[key]
        if not S: cache[key] = base; return base
        tot = 0.0
        for st, mv, legal in samples:
            pv, _ = ensemble(list(S), cortex, game, st, mv)
            p = min(1.0 - eps, max(eps, pv))
            tot += math.log(p if legal else 1.0 - p)
        out = tot / max(1, len(samples))
        cache[key] = out
        return out
    return v, base

def shapley(cortex, game, samples, regions=None, exact_max=12, permutations=64,
            rng=None):
    """Exact below `exact_max` regions, Monte-Carlo above.

    Exact: phi_i = sum over S not containing i of |S|!(n-|S|-1)!/n! * (v(S+i)-v(S)).
    MC: average marginal contribution along random permutations, which satisfies
    efficiency EXACTLY per permutation because the contributions telescope.
    """
    regions = regions or [r for r in cortex.regions if r.net is not None]
    n = len(regions)
    if n == 0: return {}, 0.0, 0.0
    v, base = value_fn(cortex, game, samples)
    full = v(set(regions))
    phi = {r.id: 0.0 for r in regions}
    if n <= exact_max:
        fact = [math.factorial(k) for k in range(n + 1)]
        for i, r in enumerate(regions):
            others = [x for x in regions if x is not r]
            for k in range(n):
                w = fact[k] * fact[n - k - 1] / fact[n]
                for S in combinations(others, k):
                    phi[r.id] += w * (v(set(S) | {r}) - v(set(S)))
        method = "exact"
    else:
        rng = rng or random.Random(0)
        order = list(regions)
        for _ in range(permutations):
            rng.shuffle(order)
            acc = set(); prev = base
            for r in order:
                acc.add(r); cur = v(acc)
                phi[r.id] += (cur - prev) / permutations
                prev = cur
        method = "mc"
    total = sum(phi.values())
    return {"phi": phi, "total": total, "v_full": full, "v_empty": base,
            "efficiency_error": abs(total - (full - base)), "method": method,
            "regions": n}
