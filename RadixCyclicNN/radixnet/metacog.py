"""MetaLayer - the part of the brain that takes over when the walk meets a cycle.

``Research/CyclesAreAFeature.md``: *the brain is a directed cyclic graph, where
cycles are a feature, not a bug; when we encounter a cycle we use metacognition
or another part of the brain instead.*  In :mod:`radixnet.resonance` that is
taken literally.  The ordinary edge costs decide every step that goes somewhere
new; the moment a walk is about to re-enter a state it has already been in
**at the same phase** - a phase-locked cycle, the kind that would repeat for
ever - the search stops asking the graph and asks this layer instead.

A cycle is named by a *signature*: the first trigram of the node being re-entered
and how many steps ago it was last seen, e.g. ``"lo :2"``.  Every signature
carries a score for each of three actions:

``ride``
    go round again.  Right for ``"aaa"``, ``"lol lol lol"``, ``"----"``,
    indentation, any honest repetition.
``escape``
    take the cheapest child that does *not* close the cycle.
``abort``
    stop here - walk to END.

The scores are learned exactly the way everything else in this package is: by
counting what the real corpus did.  While training walks a text and passes
through a cycle signature, whatever the text actually did next scores ``+1``.
At prediction the layer contributes ``-log softmax(scores)`` to the cost of the
matching move, so a cycle the corpus rides is cheap to ride and a cycle it never
rides is expensive.  Unseen signatures fall back to a global prior learned over
every cycle the model has ever seen, and an untrained layer is uniform - it
costs ``log 3`` whatever the walk does, which changes no ranking.

:meth:`invert` negates every score (2NRL: what was ridden is now escaped), and
:meth:`reward` / :meth:`punish` move one signature the way thumbs up / down move
an edge.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = ["ACTIONS", "RIDE", "ESCAPE", "ABORT", "MetaLayer", "cycle_signature"]

RIDE, ESCAPE, ABORT = "ride", "escape", "abort"
ACTIONS: tuple[str, str, str] = (RIDE, ESCAPE, ABORT)

_MAX_LOOP = 8
"""Loop lengths above this share one signature: past a point a loop is just 'a long one'."""

_LOG_ACTIONS = math.log(len(ACTIONS))


def cycle_signature(label: str, loop_len: int) -> str:
    """Name of a cycle: the re-entered node's first trigram and the loop's length.

    ``loop_len`` is how many steps ago the state was last visited (``1`` is a
    self-loop).  Lengths above :data:`_MAX_LOOP` are clamped so that rare long
    loops share one signature instead of each getting its own.  The trigram -
    not the node id - keys the layer, so a signature survives compression,
    saving and loading.
    """
    return f"{label[:3]}:{min(max(1, int(loop_len)), _MAX_LOOP)}"


class MetaLayer:
    """Learned ``ride`` / ``escape`` / ``abort`` policy per cycle signature.

    ``scores`` maps a signature to three floats (one per action in
    :data:`ACTIONS`); ``prior`` holds the same three summed over every
    signature and is what an unseen cycle is judged by.  Scores are plain
    floats rather than counts so that feedback can push them negative.
    """

    __slots__ = ("scores", "prior", "smoothing", "observed")

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = float(smoothing)
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        self.scores: dict[str, list[float]] = {}
        self.prior: list[float] = [0.0, 0.0, 0.0]
        self.observed = 0

    # -- learning ------------------------------------------------------------

    def observe(self, signature: str, action: str, amount: float = 1.0) -> None:
        """Record that a real text resolved ``signature`` by doing ``action``."""
        idx = self._index(action)
        row = self.scores.get(signature)
        if row is None:
            row = self.scores[signature] = [0.0, 0.0, 0.0]
        row[idx] += amount
        self.prior[idx] += amount
        self.observed += 1

    def reward(self, signature: str, action: str, amount: float = 1.0) -> None:
        """Thumbs up: make ``action`` more likely for this cycle."""
        self.observe(signature, action, amount)

    def punish(self, signature: str, action: str, amount: float = 1.0) -> None:
        """Thumbs down: make ``action`` less likely for this cycle."""
        self.observe(signature, action, -amount)

    def invert(self) -> None:
        """2NRL: negate every score, so what the layer preferred it now avoids."""
        for row in self.scores.values():
            row[0], row[1], row[2] = -row[0], -row[1], -row[2]
        self.prior = [-v for v in self.prior]

    # -- the policy ----------------------------------------------------------

    def log_policy(self, signature: str) -> list[float]:
        """``[log P(ride), log P(escape), log P(abort)]`` for a signature.

        The signature's scores when it has been seen, the global prior
        otherwise.  A score is evidence, not a logit, so it goes through a
        sign-preserving ``log1p`` first::

            logit(s) = sign(s) * log(1 + |s| / smoothing)

        With the default ``smoothing = 1`` that makes ``exp(logit) = 1 +
        score``, so the softmax is exactly add-one smoothing over the counts -
        ten observations are confident, a hundred more so, and one is barely
        an opinion.  Negative scores (a punished or inverted action) push the
        other way by the same amount, and an untrained layer is uniform, which
        costs ``log 3`` whatever the walk does and so changes no ranking.
        """
        row = self.scores.get(signature)
        if row is None:
            row = self.prior
        s = self.smoothing or 1.0
        logits = [math.copysign(math.log1p(abs(v) / s), v) for v in row]
        m = max(logits)
        log_total = m + math.log(math.fsum(math.exp(v - m) for v in logits))
        return [v - log_total for v in logits]

    def cost(self, signature: str, action: str) -> float:
        """``-log P(action | signature)`` - what the layer adds to a move's cost."""
        return -self.log_policy(signature)[self._index(action)]

    def costs(self, signature: str) -> dict[str, float]:
        """``{action: -log P}`` for all three actions."""
        return {a: -lp for a, lp in zip(ACTIONS, self.log_policy(signature))}

    def decide(self, signature: str) -> str:
        """The action the layer would take on its own (the cheapest one)."""
        lp = self.log_policy(signature)
        return ACTIONS[max(range(len(ACTIONS)), key=lp.__getitem__)]

    # -- reporting -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scores)

    def totals(self) -> dict[str, float]:
        """Summed score per action over every signature."""
        return dict(zip(ACTIONS, self.prior))

    def top(self, n: int = 5) -> list[dict]:
        """The ``n`` most-observed signatures with their policy, for tables and the API."""
        rows = sorted(self.scores.items(), key=lambda kv: -sum(abs(v) for v in kv[1]))[: max(0, n)]
        out: list[dict] = []
        for sig, row in rows:
            out.append({
                "signature": sig,
                "scores": dict(zip(ACTIONS, row)),
                "decision": self.decide(sig),
                "weight": sum(abs(v) for v in row),
            })
        return out

    def stats(self) -> dict:
        """``{"signatures", "observed", "totals", "decision"}`` - what the status bar shows."""
        return {
            "signatures": len(self.scores),
            "observed": self.observed,
            "totals": self.totals(),
            "decision": self.decide("") if self.observed else RIDE,
        }

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (signatures with an all-zero row are dropped)."""
        return {
            "smoothing": self.smoothing,
            "observed": self.observed,
            "actions": list(ACTIONS),
            "prior": list(self.prior),
            "scores": {sig: list(row) for sig, row in self.scores.items() if any(row)},
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "MetaLayer":
        """Rebuild a layer; ``None`` or a missing document gives a fresh one."""
        layer = cls(smoothing=float((d or {}).get("smoothing", 1.0)))
        if not d:
            return layer
        layer.observed = int(d.get("observed", 0))
        prior = d.get("prior")
        if isinstance(prior, Iterable) and not isinstance(prior, (str, bytes)):
            values = [float(v) for v in prior]
            if len(values) == len(ACTIONS):
                layer.prior = values
        for sig, row in (d.get("scores") or {}).items():
            values = [float(v) for v in row]
            if len(values) == len(ACTIONS):
                layer.scores[str(sig)] = values
        return layer

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _index(action: str) -> int:
        try:
            return ACTIONS.index(action)
        except ValueError:
            raise ValueError(f"unknown action {action!r}; expected one of: {', '.join(ACTIONS)}") from None

    def __repr__(self) -> str:
        return f"MetaLayer(signatures={len(self.scores)}, observed={self.observed})"
