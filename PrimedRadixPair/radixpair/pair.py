"""The rungs: the count tree and the reward tree read together, at every context length.

Both trees are complete over the same ``(codec, L)``, so a sequence has one id
in each and the rung between equal nodes is that id read twice.  At a context
``c`` the count tree says how often each step was taken (its smoothed
*share*) and the reward tree what it earned, and the rung adds them the way
``RadixCyclicNN``'s count model adds them inside one weight (D-022, without the
recency window)::

    merit(x|c)  = share_scale * log share(x|c)
    reward:     score = merit + reward_scale * (plus - minus)        q_c = softmax(score)
    punishment: score = merit_scale * merit - penalty_scale * minus  q_c = softmax(score)

The **fold** is the exact next-unit distribution: stay at ``c`` with ``own(c)``
and answer from ``q_c``, or fall to ``c[1:]`` with ``1 - own(c)`` and decide
again; at the root, falling means answering uniformly; the floor is mixed in
last.  With ``smoothing = 0`` and no rewards it is ``FilterBankRadix``'s
section 5.3 term for term (``DESIGN.md`` section 9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields

from .address import Address
from .codec import Codec
from .count import DEFAULT_ALPHA, DEFAULT_SMOOTHING, NODE_CEILING, CountTree
from .reward import RUNGS, RewardTree

__all__ = ["BACKOFFS", "RadixPair", "Score", "Settings", "TRAVERSALS", "softmax"]

TRAVERSALS = ("reward", "punishment")
BACKOFFS = ("all", "deepest", "none")
DEFAULT_FLOOR = 0.02


@dataclass
class Settings:
    alpha: float = DEFAULT_ALPHA
    floor: float = DEFAULT_FLOOR
    smoothing: float = DEFAULT_SMOOTHING
    share_scale: float = 1.0
    reward_scale: float = 1.0
    merit_scale: float = 1.0
    penalty_scale: float = 1.0
    strength: float = 1.0
    rungs: str = "all"
    backoff: str = "all"
    step_penalty: float = 0.0
    max_expansions: int = 200_000
    node_ceiling: int = NODE_CEILING

    def validate(self) -> "Settings":
        if self.alpha < 0 or not (0.0 <= self.floor < 1.0) or self.smoothing < 0:
            raise ValueError("alpha >= 0, 0 <= floor < 1 and smoothing >= 0")
        for name in ("share_scale", "reward_scale", "merit_scale", "penalty_scale"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if self.rungs not in RUNGS:
            raise ValueError(f"rungs must be one of {RUNGS}, got {self.rungs!r}")
        if self.backoff not in BACKOFFS:
            raise ValueError(f"backoff must be one of {BACKOFFS}, got {self.backoff!r}")
        if self.step_penalty < 0 or self.max_expansions < 1:
            raise ValueError("step_penalty >= 0 and max_expansions >= 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        return cls(**d).validate()

    def replace(self, **changes) -> "Settings":
        d = self.to_dict()
        d.update({k: v for k, v in changes.items() if v is not None})
        return Settings.from_dict(d)


@dataclass
class Score:
    """A text scored: the model's belief, and the reward tree's own readings of its steps."""

    bits: float
    mean_reward: float
    worst_penalty: float
    units: int
    per_unit: list[tuple[str, float, float, float]] = field(default_factory=list)   # (unit, bits, reward, penalty)

    def to_dict(self) -> dict:
        return {"bits": self.bits, "mean_reward": self.mean_reward, "worst_penalty": self.worst_penalty,
                "units": self.units, "per_unit": [list(row) for row in self.per_unit]}


def softmax(scores: Sequence[float]) -> list[float]:
    """A distribution from scores; ``-inf`` scores get 0; all ``-inf`` gives the uniform."""
    top = max(scores)
    if top == -math.inf:
        n = len(scores)
        return [1.0 / n] * n
    exps = [math.exp(s - top) if s > -math.inf else 0.0 for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


class RadixPair:
    def __init__(self, codec: Codec, L: int, settings: Settings | None = None) -> None:
        self.codec = codec
        self.L = int(L)
        self.settings = (settings or Settings()).validate()
        self.address = Address(codec.R, self.L)
        self.count = CountTree(codec, self.L, self.address, alpha=self.settings.alpha,
                               smoothing=self.settings.smoothing, node_ceiling=self.settings.node_ceiling)
        self.reward = RewardTree(codec, self.L, self.address)
        self.emits = codec.emits()
        self.index = {x: j for j, x in enumerate(self.emits)}
        self._cache: dict = {}
        self._cache_key: tuple = ()

    # -- settings --------------------------------------------------------------------------

    @property
    def D(self) -> int:
        return self.L - 1

    @property
    def R_out(self) -> int:
        return len(self.emits)

    def configure(self, **changes) -> Settings:
        """Change the settings a prediction reads; ``rungs`` and ``node_ceiling`` are the model's and stay."""
        for fixed in ("rungs", "node_ceiling"):
            if fixed in changes and changes[fixed] is not None and changes[fixed] != getattr(self.settings, fixed):
                raise ValueError(f"{fixed} is decided at priming and cannot be changed")
        self.settings = self.settings.replace(**changes)
        self.count.alpha = self.settings.alpha
        self.count.smoothing = self.settings.smoothing
        self._cache.clear()
        return self.settings

    # -- the rung ----------------------------------------------------------------------------

    def _fresh(self) -> None:
        key = (self.count.version, self.reward.version, self.settings.smoothing, self.settings.share_scale,
               self.settings.reward_scale, self.settings.merit_scale, self.settings.penalty_scale)
        if key != self._cache_key:
            self._cache.clear()
            self._cache_key = key

    def scores(self, i: int, l: int, traversal: str = "reward") -> list[float]:
        """The step scores of node ``i``'s children under a traversal, in ``emits`` order."""
        if traversal not in TRAVERSALS:
            raise ValueError(f"traversal must be one of {TRAVERSALS}, got {traversal!r}")
        s = self.settings
        share = self.count.share(i, l)
        merit = [s.share_scale * math.log(p) if p > 0.0 else -math.inf for p in share]
        read_rewards = s.rungs == "all" or l == self.D
        if traversal == "reward":
            if not read_rewards or s.reward_scale == 0.0:
                return merit
            rew = self.reward.rewards(i, l)
            k = s.reward_scale
            return [m + k * r for m, r in zip(merit, rew)]
        if not read_rewards or s.penalty_scale == 0.0:
            return [s.merit_scale * m for m in merit]
        pen = self.reward.penalties(i, l)
        return [s.merit_scale * m - s.penalty_scale * p for m, p in zip(merit, pen)]

    def q(self, i: int, l: int, traversal: str = "reward") -> list[float]:
        """``softmax(scores)`` - the rung's answer at one level; cached until either tree changes."""
        self._fresh()
        key = (i, traversal)
        hit = self._cache.get(key)
        if hit is None:
            hit = softmax(self.scores(i, l, traversal))
            self._cache[key] = hit
        return hit

    def own(self, i: int, l: int) -> float:
        return self.count.own(i, l)

    # -- contexts ------------------------------------------------------------------------------

    def context_node(self, context: Sequence[int]) -> tuple[int, int]:
        """The node of the last ``<= D`` units of a context, and its level."""
        c = list(context)[-self.D:] if self.D > 0 else []
        return self.address.of(c), len(c)

    def levels(self, context: Sequence[int]) -> list[tuple[int, int]]:
        """``(node, level)`` for the context and every shorter suffix, deepest first, down to the root."""
        c = list(context)[-self.D:] if self.D > 0 else []
        out = []
        for j in range(len(c) + 1):
            suffix = c[j:]
            out.append((self.address.of(suffix), len(suffix)))
        return out

    # -- the fold --------------------------------------------------------------------------------

    def fold(self, context: Sequence[int], traversal: str = "reward", backoff: str | None = None) -> list[float]:
        """The exact next-unit distribution from a context, over ``emits``, floor included."""
        s = self.settings
        backoff = backoff or s.backoff
        if backoff not in BACKOFFS:
            raise ValueError(f"backoff must be one of {BACKOFFS}, got {backoff!r}")
        n = self.R_out
        uniform = 1.0 / n
        levels = self.levels(context)
        if backoff == "none":
            i, l = levels[0]
            P = self.q(i, l, traversal) if self.count.ctx(i, l) > 0 else [uniform] * n
        elif backoff == "deepest":
            i, l = levels[0]
            o = self.own(i, l)
            qc = self.q(i, l, traversal)
            P = [o * a + (1.0 - o) * uniform for a in qc]
        else:
            P = [uniform] * n
            for i, l in reversed(levels):      # the root first, the deepest last
                o = self.own(i, l)
                if o == 0.0:
                    continue                    # nothing read here: fall straight through
                qc = self.q(i, l, traversal)
                P = [o * a + (1.0 - o) * b for a, b in zip(qc, P)]
        f = s.floor
        return [(1.0 - f) * p + f * uniform for p in P]

    def probability(self, context: Sequence[int], x: int, traversal: str = "reward", backoff: str | None = None) -> float:
        return self.fold(context, traversal, backoff)[self.index[x]]

    # -- scoring --------------------------------------------------------------------------------------

    def score(self, text: str, traversal: str = "reward", backoff: str | None = None) -> Score:
        """Bits per unit under the model's belief, and the reward tree's readings of every step."""
        ids = self.codec.padded(text)
        bits_sum = 0.0
        rew_sum = 0.0
        worst = 0.0
        rows: list[tuple[str, float, float, float]] = []
        idx = self.index
        for t in range(1, len(ids)):
            x = ids[t]
            if x not in idx:
                raise ValueError(f"{self.codec.symbol(x)!r} (id {x}) cannot be emitted")
            context = ids[max(0, t - self.D):t] if self.D > 0 else []
            P = self.fold(context, traversal, backoff)
            bits = -math.log2(P[idx[x]])
            node, l = self.context_node(context)
            step = self.address.append(node, l, x)
            r = self.reward.reward(step)
            p = self.reward.penalty(step)
            bits_sum += bits
            rew_sum += r
            worst = max(worst, p)
            rows.append((self.codec.symbol(x), bits, r, p))
        n = max(1, len(ids) - 1)
        return Score(bits=bits_sum / n, mean_reward=rew_sum / n, worst_penalty=worst, units=len(ids) - 2, per_unit=rows)

    def memory_bytes(self) -> int:
        return self.count.memory_bytes() + self.reward.memory_bytes()

    def __repr__(self) -> str:
        return f"RadixPair({self.codec.describe()}, L={self.L}, N={self.address.N:,})"
