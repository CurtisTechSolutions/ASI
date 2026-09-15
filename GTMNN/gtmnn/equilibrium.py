"""Solving the stage game.

`exploitability` is the honest convergence measure: zero exactly at a Nash
equilibrium. "The solver ran 64 iterations" says nothing; "exploitability 0.004"
says the profile is within 0.004 of nobody wanting to move.
"""
import math, random
from array import array
from dataclasses import dataclass, field

GAMMA_REGRET = 0.95
ETA_REPLICATOR = 0.1


@dataclass
class Equilibrium:
    profile: list
    actions: list
    iterations: int
    converged: bool
    cycle: int = 0
    exploitability: float = 0.0
    potential: object = None
    solver: str = "regret_plus"

    def to_dict(self):
        return {"iterations": self.iterations, "converged": self.converged,
                "cycle": self.cycle, "exploitability": round(self.exploitability, 9),
                "potential": self.potential, "solver": self.solver,
                "actions": list(self.actions)}


class CycleDetector:
    """Hashes each completed sweep's pure profile. A returning profile in a
    DETERMINISTIC dynamic means the whole future is that loop, so the solver
    stops immediately rather than confirming it.

    Push once per completed SWEEP, never per seat move. This is not a detail:
    mid-sweep the profile is a half-updated state that is not a point of the
    dynamic, and hashing those states multiplies every reported cycle length by
    the number of seats that move within the loop. Two-player rock-paper-scissors
    is the worked example -- per sweep it has period 3, per move it reports 6."""

    def __init__(self, max_history=256):
        self.seen = {}
        self.sweep = 0
        self.max_history = max_history

    def push(self, actions):
        key = tuple(actions)
        first = self.seen.get(key)
        if first is not None:
            return self.sweep - first
        if len(self.seen) < self.max_history:
            self.seen[key] = self.sweep
        self.sweep += 1
        return 0


def _uniform(n):
    return array("d", [1.0 / n]) * n


def _uniform_over(ok):
    """Uniform over the PLAYABLE slots only."""
    k = sum(1 for v in ok if v)
    return array("d", [(1.0 / k if v else 0.0) for v in ok]) if k else _uniform(len(ok))


def _argmax(row):
    best, bi = -1e18, 0
    for i, v in enumerate(row):
        if v > best + 1e-15: best, bi = v, i
    return bi


def exploitability(game, actions):
    """max over seats of (best-response utility - realised utility). 0 at Nash."""
    us = game.utilities(actions)
    loads = game.loads(actions)
    worst = 0.0
    for seat in range(len(actions)):
        _, bu = game.best_response(seat, actions, loads)
        d = bu - us[seat]
        if d > worst: worst = d
    return worst


def solve(game, method="regret_plus", iters=64, rng=None, tol=1e-6, pool=None):
    rng = rng or random.Random(0)
    fn = {"best_response": _best_response, "fictitious": _fictitious,
          "regret": _regret, "regret_plus": _regret_plus,
          "replicator": _replicator}.get(method)
    if fn is None: raise ValueError(f"unknown solver {method!r}")
    eq = fn(game, iters, rng, tol, pool)
    eq.solver = method
    eq.exploitability = exploitability(game, eq.actions)
    eq.potential = game.potential(eq.actions)
    return eq


def _pure(actions, nslots):
    out = []
    for a, k in zip(actions, nslots):
        row = array("d", [0.0]) * k
        row[a] = 1.0
        out.append(row)
    return out


def _slot_counts(game):
    return [len(r) for r in game.ctx.slot_symbol]


def _best_response(game, iters, rng, tol, pool):
    """Round-robin; each seat moves if that strictly improves its utility.

    On an exact potential game this terminates by the finite improvement
    property, and we ASSERT the potential strictly increased on every accepted
    deviation. If that fires, the payoff function has lied about being an exact
    potential, and finding that out at the moment it happens is worth the
    arithmetic."""
    nslots = _slot_counts(game)
    n = len(game.seats)
    actions = [k - 1 for k in nslots]            # everyone starts on ABSTAIN
    exact = game.payoff.is_exact_potential()
    det = CycleDetector()
    converged, cycle, it = False, 0, 0
    for it in range(1, iters + 1):
        moved = False
        for seat in range(n):
            loads = game.loads(actions)
            us = game.slot_utilities(seat, actions, loads)
            u_now = us[actions[seat]]
            s, u = max(enumerate(us), key=lambda t: (t[1], -t[0]))
            if s != actions[seat] and u > u_now + tol:
                if exact:
                    before = game.potential(actions)
                    actions[seat] = s
                    after = game.potential(actions)
                    assert after > before - 1e-9, (
                        f"{game.payoff.name} claims exact potential but Phi fell "
                        f"{before} -> {after} on an improving deviation")
                else:
                    actions[seat] = s
                moved = True
        if not moved:
            converged = True
            break
        if not exact:
            cycle = det.push(actions)
            if cycle: break
    return Equilibrium(_pure(actions, nslots), actions, it, converged, cycle)


def _fictitious(game, iters, rng, tol, pool):
    """Each seat best-responds to the time-average rather than the last profile."""
    nslots = _slot_counts(game)
    n = len(game.seats)
    counts = [array("d", [0.0]) * k for k in nslots]
    actions = [k - 1 for k in nslots]
    for it in range(1, iters + 1):
        for seat in range(n):
            s, _ = game.best_response(seat, actions)
            actions[seat] = s
        for seat in range(n): counts[seat][actions[seat]] += 1.0
    profile = []
    for c in counts:
        t = sum(c) or 1.0
        profile.append(array("d", [v / t for v in c]))
    acts = [_argmax(r) for r in profile]
    return Equilibrium(profile, acts, iters, False, 0)


def _regret_core(game, iters, rng, tol, pool, plus):
    """Hart & Mas-Colell regret matching, per seat over its K+1 slots.

    The returned profile is the TIME-AVERAGE of p^t, not the final iterate: the
    convergence result is about the average, and returning the last iterate is a
    common and silent error.

    `regret_plus` clamps cumulative regrets at zero, discarding the memory of
    actions that have been bad for a long time. It converges several times faster
    and is the default."""
    nslots = _slot_counts(game)
    n = len(game.seats)
    ok = [game.playable(seat) for seat in range(n)]
    R = []
    for seat, k in enumerate(nslots):
        row = array("d", [0.0]) * k
        if pool is not None:                     # warm start, decayed
            i = game.seats[seat]
            S = pool.S
            for s in range(min(k, S)): row[s] = pool.regret[i * S + s] * GAMMA_REGRET
        R.append(row)
    avg = [array("d", [0.0]) * k for k in nslots]
    p = [_uniform_over(ok[seat]) for seat in range(n)]
    actions = [k - 1 for k in nslots]

    for it in range(1, iters + 1):
        for seat in range(n):
            r = p[seat]
            x = rng.random(); acc = 0.0; pick = len(r) - 1
            for s, v in enumerate(r):
                acc += v
                if x <= acc: pick = s; break
            actions[seat] = pick
        loads = game.loads(actions)
        for seat in range(n):
            us = game.slot_utilities(seat, actions, loads)
            u_now = us[actions[seat]]
            for s in range(nslots[seat]):
                if not ok[seat][s]: R[seat][s] = 0.0; continue
                R[seat][s] += us[s] - u_now
                if plus and R[seat][s] < 0.0: R[seat][s] = 0.0
        for seat in range(n):
            tot = 0.0
            for s, v in enumerate(R[seat]):
                if ok[seat][s] and v > 0.0: tot += v
            row = p[seat]
            if tot > 0.0:
                for s in range(nslots[seat]):
                    v = R[seat][s] if (ok[seat][s] and R[seat][s] > 0.0) else 0.0
                    row[s] = v / tot
            else:
                u = _uniform_over(ok[seat])
                for s in range(nslots[seat]): row[s] = u[s]
            for s in range(nslots[seat]): avg[seat][s] += row[s]

    profile = []
    for seat in range(n):
        t = sum(avg[seat]) or 1.0
        profile.append(array("d", [v / t for v in avg[seat]]))
    if pool is not None:
        S = pool.S
        for seat, i in enumerate(game.seats):
            for s in range(min(nslots[seat], S)): pool.regret[i * S + s] = R[seat][s]
    acts = [_argmax(r) for r in profile]
    return Equilibrium(profile, acts, iters, True, 0)


def _regret(game, iters, rng, tol, pool): return _regret_core(game, iters, rng, tol, pool, False)
def _regret_plus(game, iters, rng, tol, pool): return _regret_core(game, iters, rng, tol, pool, True)


def _replicator(game, iters, rng, tol, pool):
    """p(a) <- p(a) * [1 + eta(u(a) - ubar)], renormalised.

    Not used for inference -- it is the dynamic under which "evolutionarily
    stable" is defined, and evolve.py uses it to test whether a mutant strategy
    can invade the incumbents."""
    nslots = _slot_counts(game)
    n = len(game.seats)
    ok = [game.playable(seat) for seat in range(n)]
    p = [_uniform_over(ok[seat]) for seat in range(n)]
    for it in range(1, iters + 1):
        actions = [_argmax(r) for r in p]
        loads = game.loads(actions)
        for seat in range(n):
            us = game.slot_utilities(seat, actions, loads)
            ubar = sum(p[seat][s] * us[s] for s in range(nslots[seat]))
            tot = 0.0
            for s in range(nslots[seat]):
                v = p[seat][s] * (1.0 + ETA_REPLICATOR * (us[s] - ubar)) if ok[seat][s] else 0.0
                p[seat][s] = v if v > 0.0 else 0.0
                tot += p[seat][s]
            if tot > 0.0:
                for s in range(nslots[seat]): p[seat][s] /= tot
            else:
                u = _uniform_over(ok[seat])
                for s in range(nslots[seat]): p[seat][s] = u[s]
    acts = [_argmax(r) for r in p]
    return Equilibrium(p, acts, iters, True, 0)
