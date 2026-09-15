"""Choosing what to try next.

The intuition -- guess, fail, learn from what breaks -- has an exact form. Choose
the probe maximising expected information gain; for a binary oracle that is
provably maximised at p(legal) = 0.5, so the TARGET FAILURE RATE IS 50% and it
falls out of the mathematics rather than being set as a knob.

Every policy reports its observed failure rate, because that is the diagnostic:
a run at 5% is confirming what it already believes and a run at 95% is thrashing
outside the boundary. Neither is exploring.
"""
import math, random
from gren.verdict import Outcome

class Policy:
    name = "policy"
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.legal = 0; self.illegal = 0; self.bits = 0.0
    def note(self, verdict, surprise=0.0):
        if verdict.outcome == Outcome.LEGAL: self.legal += 1
        elif verdict.outcome == Outcome.ILLEGAL: self.illegal += 1
        self.bits += surprise
    @property
    def failure_rate(self):
        n = self.legal + self.illegal
        return self.illegal / n if n else 0.0
    def propose(self, oracle, state, k=12): raise NotImplementedError
    def stats(self):
        return {"policy": self.name, "probes": self.legal + self.illegal,
                "failure_rate": round(self.failure_rate, 3),
                "bits": round(self.bits, 2)}

class Random(Policy):
    """The control. A policy that cannot beat it is not a policy."""
    name = "random"
    def propose(self, oracle, state, k=12):
        return oracle.candidates(state, self.rng, k)

class Boundary(Policy):
    """Small edits to a KNOWN-LEGAL action land near the refusal boundary, which
    is where p(legal) is near 0.5 by construction. Needs no posterior and is the
    strongest policy wherever legal moves can be enumerated at all."""
    name = "boundary"
    def propose(self, oracle, state, k=12):
        legal = oracle.actions(state)
        if not legal: return oracle.candidates(state, self.rng, k)
        out = []
        for _ in range(k * 2):
            a = self.rng.choice(legal)
            out.append(_perturb(a, self.rng))
            if len(out) >= k: break
        return out + self.rng.sample(legal, min(k // 2, len(legal)))

class EIG(Policy):
    """Full expected information gain over the REASON-CLASS distribution, not
    just legal/illegal. A probe that splits candidates by WHICH refusal they
    predict scores far above one that only splits legal from illegal, which is
    where log2(R+1) bits per probe instead of 1 actually comes from."""
    name = "eig"
    def __init__(self, seed=0, model=None):
        super().__init__(seed)
        self.model = model or {}       # code -> count, the running belief
        self.total = 0

    def _entropy(self):
        if self.total <= 0: return 1.0
        h = 0.0
        for c in self.model.values():
            p = c / self.total
            if p > 0: h -= p * math.log2(p)
        return h

    def observe(self, verdict):
        if verdict.outcome == Outcome.ILLEGAL and verdict.reason_code:
            before = self._entropy()
            self.model[verdict.reason_code] = self.model.get(verdict.reason_code, 0) + 1
            self.total += 1
            self.note(verdict, surprise=max(0.0, self._entropy() - before) +
                      math.log2(max(1, len(self.model))) / max(1, self.total))
        else:
            self.note(verdict)

    def propose(self, oracle, state, k=12):
        """Mix legal actions with perturbations of them, aiming the split at
        50/50. Rare refusals are sought by preferring perturbations whose code
        the belief has seen LEAST often, which is what drives the taxonomy to
        completeness rather than to the common cases."""
        legal = oracle.actions(state)
        pool = oracle.candidates(state, self.rng, k * 2)
        if not legal: return pool[:k]
        half = max(1, k // 2)
        near = [_perturb(self.rng.choice(legal), self.rng) for _ in range(half * 2)]
        picked = self.rng.sample(legal, min(half, len(legal))) + near[:half]
        self.rng.shuffle(picked)
        return picked[:k] or pool[:k]

def _perturb(action, rng):
    """One small edit. Board moves shift the target a square; sudoku shifts the
    value. Edit distance one is what puts a probe ON the boundary."""
    if isinstance(action, tuple) and len(action) == 3 and all(
            isinstance(v, int) for v in action):
        x, y, v = action
        which = rng.randrange(3)
        if which == 0: return (max(0, min(8, x + rng.choice([-1, 1]))), y, v)
        if which == 1: return (x, max(0, min(8, y + rng.choice([-1, 1]))), v)
        return (x, y, max(1, min(9, v + rng.choice([-1, 1]))))
    if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], tuple):
        (fx, fy), (tx, ty) = action
        if rng.random() < 0.5:
            return ((fx, fy), (tx + rng.choice([-1, 0, 1]), ty + rng.choice([-1, 0, 1])))
        return ((fx + rng.choice([-1, 0, 1]), fy + rng.choice([-1, 0, 1])), (tx, ty))
    if isinstance(action, tuple) and len(action) == 2:
        x, y = action
        return (x + rng.choice([-1, 0, 1]), y + rng.choice([-1, 0, 1]))
    return action

POLICIES = {"random": Random, "boundary": Boundary, "eig": EIG}
