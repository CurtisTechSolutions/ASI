"""Metacognition: a second population playing a different game over the
STATISTICS of the deliberation that just failed.

    "When we encounter a cycle, we use metacognition or another part of the
     brain instead."  -- Research/CyclesAreAFeature.md

The meta-player never sees the text. It sees only how the argument went, which
is what makes this metacognition rather than a second opinion. Nothing about it
is a special case: same MicroPool, same auction, same congestion game, same
Shapley credit, same evolution -- the identical machinery pointed at the
network's own deliberation. That is the claim being made.
"""
import math, random
from array import array
from dataclasses import dataclass, field

from gtmnn import game as game_mod
from gtmnn import equilibrium as eq_mod

META_F = 16
TRUST_EQUILIBRIUM, TRUST_REPUTATION, TRUST_COALITION, TRUST_CYCLE_AVERAGE, META_ABSTAIN = 0, 1, 2, 3, 4
POLICY_NAMES = ["TRUST_EQUILIBRIUM", "TRUST_REPUTATION", "TRUST_COALITION",
                "TRUST_CYCLE_AVERAGE", "ABSTAIN"]


@dataclass
class MetaResult:
    distribution: array
    policy: int
    policy_name: str
    triggered: str
    depth: int
    inner: object = None

    def to_dict(self):
        return {"policy": self.policy, "policy_name": self.policy_name,
                "triggered": self.triggered, "depth": self.depth,
                "inner": self.inner.to_dict() if self.inner else None}


def triggered(eq, alloc, agg, m, margin_tau=0.05, expl_tau=0.05):
    """Which trigger fired, or "" when none did -- the common case, and the
    reason the meta-player costs nothing on inputs the population finds easy."""
    if eq.cycle: return "cycle"
    if alloc.bidders < m: return "bidders"
    if not eq.converged and eq.exploitability > expl_tau: return "exploitability"
    order = sorted(agg, reverse=True)
    if len(order) > 1 and order[0] - order[1] < margin_tau: return "margin"
    return ""


def meta_features(eq, alloc, game, agg, pool, depth, iters=64, max_depth=2, out=None):
    """Sixteen summary statistics of the deliberation. L2-normalised, exactly as
    FeatureHasher does and for the same reason."""
    V = len(agg)
    n = len(game.seats) or 1
    H = game_mod.entropy(agg) / (math.log(V) if V > 1 else 1.0)
    order = sorted(agg, reverse=True)
    margin = (order[0] - order[1]) if len(order) > 1 else order[0]
    nabs = sum(1 for s, a in enumerate(eq.actions)
               if game.ctx.slot_symbol[s][a] is None)
    reps = [pool.reputation(i) for i in game.seats] or [0.5]
    mrep = sum(reps) / len(reps)
    vrep = (sum((r - mrep) ** 2 for r in reps) / len(reps)) ** 0.5
    loads = game.loads(eq.actions)
    maxload = max(loads[:-1]) if len(loads) > 1 else 0
    distinct = sum(1 for c in loads[:-1] if c > 0)
    S = pool.S
    mreg = sum(abs(pool.regret[i * S + s]) for i in game.seats for s in range(S)) / (n * S)
    mbid = (sum(alloc.bids) / len(alloc.bids)) if alloc.bids else 0.0

    v = out if out is not None else array("d", [0.0]) * META_F
    vals = [H, margin, min(eq.cycle, 16) / 16.0, min(max(eq.exploitability, 0.0), 1.0),
            nabs / n, alloc.bidders / max(1, pool.count_alive()), alloc.price,
            mrep, vrep, eq.iterations / max(1, iters), mreg, maxload / n,
            distinct / n, 1.0 if eq.converged else 0.0, mbid, depth / max(1, max_depth)]
    for i in range(META_F): v[i] = vals[i]
    norm = sum(x * x for x in v) ** 0.5
    if norm > 0.0:
        for i in range(META_F): v[i] /= norm
    return v


def _from_reputation(game, pool, agg):
    """The single highest-reputation seat's own q, mapped to the alphabet."""
    V = len(agg)
    out = array("d", [1e-3 / V]) * V
    if not game.seats: return _norm(out)
    best = max(range(len(game.seats)), key=lambda s: pool.reputation(game.seats[s]))
    q = game.ctx.q[best]
    for s, sym in enumerate(game.ctx.slot_symbol[best]):
        if sym is not None: out[sym] += q[s]
    return _norm(out)


def _from_point(sym, V, eps=1e-3):
    out = array("d", [eps / V]) * V
    if sym is not None: out[sym] += 1.0
    return _norm(out)


def _norm(v):
    t = sum(v)
    if t > 0.0:
        for i in range(len(v)): v[i] /= t
    return v


def apply_policy(policy, eq, alloc, game, agg, pool, coalitions=None, cycle_profiles=None):
    V = len(agg)
    if policy == TRUST_EQUILIBRIUM: return agg
    if policy == TRUST_REPUTATION: return _from_reputation(game, pool, agg)
    if policy == TRUST_COALITION:
        if coalitions:
            from gtmnn.coalition import joint_action
            sym = joint_action(coalitions[0], game.seats, eq.profile, game.ctx.slot_symbol)
            return _from_point(sym, V)
        return agg
    if policy == TRUST_CYCLE_AVERAGE:
        # A best-response cycle has no pure equilibrium to report, but the
        # time-average over the cycle is exactly the object fictitious play
        # converges to on such games, and it is a genuine mixed strategy rather
        # than an arbitrary point on the loop. When the reasoning goes round in a
        # circle, the answer is the circle's centre.
        if cycle_profiles:
            out = array("d", [0.0]) * V
            for d in cycle_profiles:
                for c in range(V): out[c] += d[c]
            return _norm(out)
        return agg
    return _norm(array("d", [1.0 / V]) * V)      # META_ABSTAIN: I do not know either


def resolve(eq, alloc, game, agg, pool, meta_pool, depth=0, max_depth=2, rng=None,
            trigger="", iters=64, m=16, coalitions=None, cycle_profiles=None,
            solver="regret_plus"):
    """Build and solve a stage game on the META population over the deliberation's
    own statistics, then apply the winning policy.

    The recursion cap is a hard stop, not a soft preference. A system that can
    always escalate is a system that can fail to answer."""
    rng = rng or random.Random(0)
    feat = meta_features(eq, alloc, game, agg, pool, depth, iters=iters, max_depth=max_depth)
    mg, malloc, _ = game_mod.build(meta_pool, feat, len(POLICY_NAMES), None,
                                   m=m, charge=False)
    if not mg.seats:
        return MetaResult(agg, TRUST_EQUILIBRIUM, POLICY_NAMES[TRUST_EQUILIBRIUM],
                          trigger, depth)
    meq = eq_mod.solve(mg, solver, iters=iters, rng=rng, pool=meta_pool)
    magg = game_mod.aggregate(mg, meq.profile, meta_pool)

    inner = None
    if depth < max_depth:
        t2 = triggered(meq, malloc, magg, m)
        if t2:
            inner = resolve(meq, malloc, mg, magg, meta_pool, meta_pool,
                            depth + 1, max_depth, rng, t2, iters, m)
            magg = inner.distribution
    else:
        # At the cap the fallback is DETERMINISTIC, not another vote.
        policy = TRUST_REPUTATION
        return MetaResult(apply_policy(policy, eq, alloc, game, agg, pool,
                                       coalitions, cycle_profiles),
                          policy, POLICY_NAMES[policy], trigger, depth, inner)

    policy = max(range(len(POLICY_NAMES)), key=lambda c: magg[c]) if len(magg) >= len(POLICY_NAMES) else TRUST_EQUILIBRIUM
    dist = apply_policy(policy, eq, alloc, game, agg, pool, coalitions, cycle_profiles)
    return MetaResult(dist, policy, POLICY_NAMES[policy], trigger, depth, inner)
