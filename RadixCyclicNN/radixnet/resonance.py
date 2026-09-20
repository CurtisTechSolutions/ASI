"""ResonantNet - the phase model: an analog carrier on the same self-compressing graph.

``Research/SineWaveActivationFunction.md``: *a brain operates like an analog
computer, hence sine waves are the standard of information encoding/decoding.*
:class:`~radixnet.model.RadixNet` takes that as a **pointwise** sine - every node
passes its state through ``-sin(z / 3)``.  This model takes the other half of
the idea: a sine wave has a **phase**, phases **add** along a path, and two
signals that meet in phase reinforce while two that meet in antiphase cancel.

A walk therefore carries one more number than the node it is standing on: the
phase ``psi`` of the signal.  Every node advances it, every edge has learned a
phase at which it likes to fire, and an edge is cheap exactly when the walk
arrives in phase with it.

The phase
---------

Phase is kept as one of ``buckets`` positions on a ring (a P-state ring
counter, so the model is a finite automaton over ``(node, phase)`` rather than
an approximation of a continuous one).  Each trigram advances it by a fixed
integer::

    advance(t) = round(buckets / period + kick_scale * trigram_phase(t) * buckets / TAU)

- the first term is a **clock**: with the default ``period == buckets`` one
  character is one bucket, so the phase says *where in the rhythm we are* -
  how far into a line, a word, an indent;
- the second is a **kick**: a stable hash of the trigram itself, so the phase
  also carries *what has been said*.  ``kick_scale = 0`` (the default) leaves a
  pure clock, dense and quickly learned; turning it up turns the phase into a
  rolling signature of the whole path - real long-range context on a
  three-character graph - at the price of much sparser statistics per phase.

Because the advance is defined per **trigram** and summed, a node's advance is
the sum over the trigrams its label covers.  That makes the phase exactly
invariant under the radix split and merge (a split and a merge move trigrams
between labels but never change the multiset) and identical to the phase
computed from the emitted text alone - :meth:`ResonantGraph.text_bucket`.

The edges
---------

An edge does not learn a phase *offset*; it learns the phases at which it was
actually taken, as a circular mean.  Each traversal at bucket ``b`` adds the
unit vector ``(cos, sin)`` of that bucket to the edge's accumulator, which gives

``mu``
    the mean phase at which this transition fires, and
``coherence`` ``r = |accumulator| / (weight + concentration)``
    how consistently: ``r -> 1`` when the edge only ever fires at one phase,
    ``r -> 0`` when it fires at all of them.  ``concentration`` shrinks a thinly
    observed edge towards 0, so one traversal is not mistaken for certainty.

``r`` is free confidence: it says how *context-dependent* a transition is,
measured, with nothing added to the model to measure it.  It gates the
resonance, so the score of an edge is::

    score(p -> c | psi) = amplitude + resonance_scale * r * cos(psi - mu)

with ``amplitude = amp_scale * log(the edge's smoothed share of its parent's
traversals) + reward_scale * reward``.  An incoherent edge falls back to plain
frequency; a coherent one is cheap in phase and dear out of phase.  Children are
drawn by a softmax over those scores, so ``P(child | parent, phase)`` - the
first model here whose distribution depends on more than the parent.

Learning is counting: a traversal, a unit vector, a reward.  There is no
gradient, no learning rate and no ordering effect.  ``invert()`` negates the
accumulators, which rotates every ``mu`` by ``pi``: what resonated now cancels.
That is 2NRL's inversion, done in one line and with an exact meaning.

Cycles
------

``Research/CyclesAreAFeature.md``: *cycles are a feature, not a bug; when we
encounter a cycle we use metacognition or another part of the brain instead.*
The phase is what makes that actionable.  Coming back to a node at a **new**
phase is progress - the signal has moved on.  Coming back at the **same** phase
is a true loop that would repeat for ever.  So the search watches for a repeated
``(node, phase)`` on the path it is building and, when it finds one, stops
asking the graph and asks :class:`~radixnet.metacog.MetaLayer` whether to ride
the loop, escape it or stop (:mod:`radixnet.phasesearch`).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_B, DEFAULT_H
from .backend import get_backend
from .counter import CyclicCounter
from .beam import Prediction, default_beam
from .encoding import WINDOW, Decoder, Encoder
from .graph import BACK, END, FIRST, START, RadixCyclicGraph
from .metacog import ABORT, ESCAPE, RIDE, MetaLayer, cycle_signature
from .model import (
    MODEL_FORMAT_VERSION,
    GraphModel,
    ProgressFn,
    TrainConfig,
    _resolve_config,
    _utc_now,
    _weight_groups,
    _whole_text,
)
from .phasesearch import phase_beam, phase_dijkstra, phase_kbest, phase_walk
from .search import PathResult

__all__ = ["RESONANT_MODEL_FORMAT", "TAU", "ResonantGraph", "ResonantNet", "trigram_phase"]

RESONANT_MODEL_FORMAT = "radixnet-resonant"
TAU = 2.0 * math.pi
_W = WINDOW
_OV = WINDOW - 1
_MAX_LOG_PPL = 700.0
_LOG_UNKNOWN = math.log(1e-6)

_phase_cache: dict[str, float] = {}


def trigram_phase(trigram: str) -> float:
    """The trigram's own phase in ``[0, TAU)`` - a stable hash, identical in every process.

    ``hash()`` is randomised per process and would make a saved model read
    differently after a restart, so the phase comes from a BLAKE2b digest of
    the trigram's UTF-8 bytes.
    """
    value = _phase_cache.get(trigram)
    if value is None:
        digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big") / 2.0**64 * TAU
        _phase_cache[trigram] = value
    return value


class ResonantGraph(RadixCyclicGraph):
    """A :class:`RadixCyclicGraph` that carries a phase and scores edges by resonance.

    Per node: ``advance`` - how many buckets traversing it moves the phase, the
    sum of its trigrams' advances (so splits and merges leave it unchanged).
    Per edge: the circular accumulator ``(edge_cx, edge_cy)`` of the phases at
    which the edge fired with its total weight ``edge_cw``, giving ``mu`` and
    ``coherence``, plus ``edge_reward``.  Activations are the constant 1 on
    every node (``a = 0, k = 1``), so the base class's phase-free scores,
    probabilities, costs, Dijkstra, sampling, splits and merges keep working
    and read as the model's **phase-marginal**: what it believes before the
    phase is taken into account.
    """

    SMOOTHING = 0.5
    """Add-``s`` smoothing of an edge's share of its parent's traversals."""

    def __init__(
        self,
        seed: int = 0,
        buckets: int = 8,
        period: float = 0.0,
        kick_scale: float = 0.0,
        resonance_scale: float = 1.0,
        amp_scale: float = 1.0,
        reward_scale: float = 1.0,
        concentration: float = 2.0,
    ) -> None:
        self.buckets = int(buckets)
        if self.buckets < 1:
            raise ValueError(f"buckets must be >= 1, got {buckets}")
        self.period = float(period) if period else float(self.buckets)
        if self.period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self.kick_scale = float(kick_scale)
        self.resonance_scale = float(resonance_scale)
        self.amp_scale = float(amp_scale)
        self.reward_scale = float(reward_scale)
        self.concentration = float(concentration)
        if self.concentration < 0:
            raise ValueError(f"concentration must be >= 0, got {concentration}")
        self.advance: list[int] = []
        self.edge_cx: list[float] = []
        self.edge_cy: list[float] = []
        self.edge_cw: list[float] = []
        self.edge_reward: list[float] = []
        self.total_traversals = CyclicCounter()
        self._adv_cache: dict[str, int] = {}
        self._phase_costs: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
        self._phase_cache_version = -1
        super().__init__(seed)

    # -- the phase -----------------------------------------------------------

    def trigram_advance(self, trigram: str) -> int:
        """Buckets one trigram moves the phase: the clock plus ``kick_scale`` times its own phase."""
        value = self._adv_cache.get(trigram)
        if value is None:
            clock = self.buckets / self.period
            kick = self.kick_scale * trigram_phase(trigram) * self.buckets / TAU
            value = int(round(clock + kick)) % self.buckets
            self._adv_cache[trigram] = value
        return value

    def label_advance(self, label: str) -> int:
        """Buckets a label moves the phase: the sum over the trigrams it covers.

        Summing per trigram is what makes the phase survive the radix
        operations - a split and a merge move trigrams between labels but never
        change which trigrams exist - and makes a walk's phase equal
        :meth:`text_bucket` of the text it emitted.
        """
        if len(label) < _W:
            return 0
        adv = self.trigram_advance
        return sum(adv(label[i : i + _W]) for i in range(len(label) - _OV)) % self.buckets

    def node_advance(self, node: int, label: str) -> int:
        """:meth:`label_advance` of a real node; 0 for START and END.

        The sentinels emit no characters, so they must not move the phase -
        and their labels (``"<s>"``, ``"</s>"``, ``"<back>"``) are long enough
        to look like ordinary ones.
        """
        return 0 if node < FIRST else self.label_advance(label)

    def text_bucket(self, text: str, start: int = 0) -> int:
        """Phase of a piece of text - the bucket a walk that emitted it would carry."""
        if len(text) < _W:
            return start % self.buckets
        adv = self.trigram_advance
        total = sum(adv(text[i : i + _W]) for i in range(len(text) - _OV))
        return (start + total) % self.buckets

    def bucket_phase(self, bucket: int) -> float:
        """The angle in ``[0, TAU)`` a bucket stands for."""
        return TAU * (bucket % self.buckets) / self.buckets

    def step_bucket(self, bucket: int, node: int) -> int:
        """The phase after traversing ``node`` from ``bucket``."""
        return (bucket + self.advance[node]) % self.buckets

    def _refresh_advance(self, node: int) -> None:
        self.advance[node] = self.node_advance(node, self.labels[node])

    # -- edge statistics -----------------------------------------------------

    def edge_mu(self, e: int) -> float:
        """Mean phase at which the edge fires (0 when it never has)."""
        x, y = self.edge_cx[e], self.edge_cy[e]
        return math.atan2(y, x) % TAU if (x or y) else 0.0

    def edge_coherence(self, e: int) -> float:
        """How consistently the edge fires at one phase, in ``[0, 1]``.

        The resultant length of the circular mean, shrunk by ``concentration``
        so that a single traversal (whose resultant length is exactly 1) reads
        as an opinion rather than a certainty.
        """
        w = self.edge_cw[e]
        if w <= 0.0:
            return 0.0
        return math.hypot(self.edge_cx[e], self.edge_cy[e]) / (w + self.concentration)

    def resonance(self, e: int, bucket: int) -> float:
        """``coherence * cos(phase - mu)`` - how well a walk at ``bucket`` meets the edge."""
        r = self.edge_coherence(e)
        if r <= 0.0:
            return 0.0
        return r * math.cos(self.bucket_phase(bucket) - self.edge_mu(e))

    def record_traversal(self, e: int, bucket: int | None, amount: float = 1.0) -> None:
        """Count one traversal of ``e`` into its circular accumulator, at phase ``bucket``.

        ``bucket = None`` counts the traversal without a phase: the weight of
        the mean grows but the vector does not, so the edge's coherence *falls*
        towards 0 and it competes on its share alone.  That is what a traversal
        whose phase is unknown honestly says - it fires, but nothing was learned
        about when.
        """
        if bucket is not None:
            angle = self.bucket_phase(bucket)
            self.edge_cx[e] += amount * math.cos(angle)
            self.edge_cy[e] += amount * math.sin(angle)
        self.edge_cw[e] += abs(amount)
        self.edge_count[e] += 1
        self.total_traversals += 1
        self.traversals += 1   # what bounds every counter, and so decides when carry_counters() has work

    def add_reward(self, edge_ids: Iterable[int], amount: float) -> int:
        """Move the reward of every alive edge in ``edge_ids`` by ``amount``; returns how many moved."""
        moved = 0
        for e in edge_ids:
            if 0 <= e < len(self.edge_reward) and self.edge_alive[e]:
                self.edge_reward[e] += amount
                moved += 1
        if moved:
            self.recompute_weights()
        return moved

    def sharpen(self, edge_ids: Iterable[int], factor: float) -> int:
        """Scale the circular accumulators: ``> 1`` locks an edge to its phase, ``< 1`` decoheres it.

        The resultant length can never exceed the accumulated weight, so
        sharpening saturates at ``coherence = w / (w + concentration)``.
        """
        if factor < 0:
            raise ValueError(f"factor must be >= 0, got {factor}")
        changed = 0
        for e in edge_ids:
            if not (0 <= e < len(self.edge_cx)) or not self.edge_alive[e]:
                continue
            x, y = self.edge_cx[e] * factor, self.edge_cy[e] * factor
            cap = self.edge_cw[e]
            length = math.hypot(x, y)
            if cap > 0.0 and length > cap:
                x, y = x * cap / length, y * cap / length
            self.edge_cx[e], self.edge_cy[e] = x, y
            changed += 1
        if changed:
            self.version += 1
        return changed

    def rotate(self, edge_ids: Iterable[int], turns: float) -> int:
        """Rotate the mean phase of edges by ``turns`` of a half-circle (``1`` = antiphase)."""
        angle = math.pi * float(turns)
        ca, sa = math.cos(angle), math.sin(angle)
        changed = 0
        for e in edge_ids:
            if not (0 <= e < len(self.edge_cx)) or not self.edge_alive[e]:
                continue
            x, y = self.edge_cx[e], self.edge_cy[e]
            self.edge_cx[e], self.edge_cy[e] = x * ca - y * sa, x * sa + y * ca
            changed += 1
        if changed:
            self.version += 1
        return changed

    # -- weights -------------------------------------------------------------

    def edge_amplitude(self, e: int, parent_total: float, degree: int) -> float:
        """The phase-free part of an edge's score: its smoothed share of the parent, plus rewards."""
        s = self.SMOOTHING
        share = (self._edge_traversals_f(e) + s) / (parent_total + s * max(1, degree))
        return self.amp_scale * math.log(share) + self.reward_scale * self.edge_reward[e]

    def recompute_weights(self) -> None:
        """Rewrite every ``edge_w`` from the counts and rewards (O(E)); bumps ``version``."""
        ew = self.edge_w
        traversals = self._edge_traversals_f
        for p, ch in enumerate(self.children):
            if not ch or not self.alive[p]:
                continue
            total = 0.0
            for e in ch.values():
                total += traversals(e)
            deg = len(ch)
            for e in ch.values():
                ew[e] = self.edge_amplitude(e, total, deg)
        self.version += 1

    def shares(self, p: int) -> list[tuple[int, int, float, float]]:
        """``[(child, edge, coherence, mu)]`` for the graph view."""
        return [(c, e, self.edge_coherence(e), self.edge_mu(e)) for c, e in self.children[p].items()]

    def configure(self, **options: float) -> dict:
        """Change the score function at run time; returns :meth:`weight_config`.

        Accepts ``buckets``, ``period``, ``kick_scale``, ``resonance_scale``,
        ``amp_scale``, ``reward_scale`` and ``concentration``.  Changing
        ``buckets``, ``period`` or ``kick_scale`` redefines the phase itself,
        so every node's advance is recomputed; what the edges already learned
        is kept (their means simply refer to the new ring).
        """
        rephase = False
        for name in ("buckets", "period", "kick_scale", "resonance_scale", "amp_scale", "reward_scale", "concentration"):
            if name not in options or options[name] is None:
                continue
            value = options.pop(name)
            if name == "buckets":
                value = int(value)
                if value < 1:
                    raise ValueError(f"buckets must be >= 1, got {value}")
            else:
                value = float(value)
                if name == "period" and value <= 0:
                    raise ValueError(f"period must be > 0, got {value}")
                if name == "concentration" and value < 0:
                    raise ValueError(f"concentration must be >= 0, got {value}")
            if getattr(self, name) != value:
                setattr(self, name, value)
                rephase = rephase or name in ("buckets", "period", "kick_scale")
        if options:
            raise ValueError(f"unknown weight option(s): {', '.join(sorted(options))}")
        if rephase:
            self._adv_cache.clear()
            for n, ok in enumerate(self.alive):
                if ok:
                    self._refresh_advance(n)
        self.recompute_weights()
        return self.weight_config()

    def weight_config(self) -> dict:
        """The score function's current settings."""
        return {
            "buckets": self.buckets,
            "period": self.period,
            "kick_scale": self.kick_scale,
            "resonance_scale": self.resonance_scale,
            "amp_scale": self.amp_scale,
            "reward_scale": self.reward_scale,
            "concentration": self.concentration,
        }

    # -- phase-aware scores --------------------------------------------------

    def child_scores_at(self, p: int, bucket: int) -> list[tuple[int, int, float]]:
        """``[(child, edge, amplitude + resonance_scale * coherence * cos(phase - mu))]``."""
        angle = self.bucket_phase(bucket)
        rs = self.resonance_scale
        ew = self.edge_w
        cx, cy, cw = self.edge_cx, self.edge_cy, self.edge_cw
        conc = self.concentration
        cos = math.cos
        atan2 = math.atan2
        hypot = math.hypot
        out: list[tuple[int, int, float]] = []
        for c, e in self.children[p].items():
            score = ew[e]
            if rs:
                w = cw[e]
                if w > 0.0:
                    x, y = cx[e], cy[e]
                    r = hypot(x, y) / (w + conc)
                    if r > 0.0:
                        score += rs * r * cos(angle - atan2(y, x))
            out.append((c, e, score))
        return out

    def child_costs_at(self, p: int, bucket: int) -> list[tuple[int, int, float]]:
        """``[(child, edge, -log softmax)]`` at one phase, cached until ``version`` changes."""
        if self._phase_cache_version != self.version:
            self._phase_costs.clear()
            self._phase_cache_version = self.version
        key = (p, bucket % self.buckets)
        costs = self._phase_costs.get(key)
        if costs is None:
            items = self.child_scores_at(p, bucket)
            if items:
                m = max(s for _, _, s in items)
                lse = m + math.log(math.fsum(math.exp(s - m) for _, _, s in items))
                costs = [(c, e, lse - s) for c, e, s in items]
            else:
                costs = []
            self._phase_costs[key] = costs
        return costs

    def child_probs_at(self, p: int, bucket: int) -> list[tuple[int, float]]:
        """``[(child, P(child | parent, phase))]``."""
        return [(c, math.exp(-cost)) for c, _e, cost in self.child_costs_at(p, bucket)]

    # -- graph hooks ---------------------------------------------------------

    def _new_node(self, label, z=None, a=None, b=DEFAULT_B, h=DEFAULT_H, k=0.0, count=0, count_resets=0) -> int:
        # a = 0 and k = 1 make f(z) = 1: the base class's scores reduce to edge_w, the phase-marginal
        nid = super()._new_node(
            label, z=0.0 if z is None else z, a=0.0, b=b, h=h, k=1.0, count=count, count_resets=count_resets
        )
        self.advance.append(self.node_advance(nid, label))
        return nid

    def _new_edge(self, p: int, c: int, count: int = 0, count_resets: int = 0) -> int:
        e = super()._new_edge(p, c, count, count_resets)
        self.edge_cx.append(0.0)
        self.edge_cy.append(0.0)
        self.edge_cw.append(0.0)
        self.edge_reward.append(0.0)
        self.edge_w[e] = 0.0  # recompute_weights() gives it its real value once the pass is over
        return e

    def split(self, node_id: int, i: int) -> tuple[int, int]:
        a_id, b_id = super().split(node_id, i)
        self._refresh_advance(a_id)
        self._refresh_advance(b_id)
        return a_id, b_id

    def merge_child(self, p: int) -> bool:
        merged = super().merge_child(p)
        if merged:
            self._refresh_advance(p)
        return merged

    def observe_back(self, p: int, went: int | None = None, instead: int | None = None,
                     amount: float = 1.0) -> int:
        """As :meth:`RadixCyclicGraph.observe_back`, learned the way this model learns everything.

        The sine model nudges the weights directly; here a weight is a *function* of the counts, the rewards
        and the phase, so nothing may be written to ``edge_w`` by hand - the next
        :meth:`recompute_weights` would erase it.  Going round is taught by counting the ``BACK`` edge and what
        to do instead by a penalty on the step it looped through and a reward on the step it took after backing
        up.  The hand-over is counted **without a phase**: a voice that backed out of a repeat
        (:func:`radixnet.dialogue.backtrack`) walked outside this model's search and cannot say which phase it
        was in, so the edge competes on its share rather than pretending to a phase it never learned.
        """
        e = super().observe_back(p, amount=0.0)  # no weight is nudged by hand here
        self.record_traversal(e, None, amount or 1.0)
        self.add_reward([e], amount)
        for child, sign in ((went, -1.0), (instead, 1.0)):
            edge = self.children[p].get(child) if child is not None and child != BACK else None
            if edge is not None:
                self.add_reward([edge], sign * amount)
        self.recompute_weights()
        return e

    def back_probability(self, p: int) -> float:
        """``P(hand over | p)`` - the share ``p``'s ``BACK`` edge takes of its children, 0 when it has none."""
        cost = self.back_cost(p)
        return 0.0 if cost is None else math.exp(-cost)

    def observe_onward(self, p: int, amount: float = 1.0) -> bool:
        """The opposite lesson to :meth:`observe_back`: a walk at ``p`` faced a cycle and carried on.

        ``observe_back`` is one-directional by design - nothing calls it except a
        voice that already backed out - so a hand-over estimate fed from it alone
        can only ever rise.  That is fine for experience, which is rare, and
        wrong for observation, which is not: a corpus declines cycles constantly,
        and teaching only the declines saturates ``BACK`` until it vetoes nodes
        the model needs (a sample run drove ``"he "`` to ``P(BACK) = 0.79``,
        which took ``"the sun"`` out of reach).  Counting the rides too makes the
        estimate a *balance* - how often walks at this node went round and had to
        stop, against how often going round was right - which is what a share of
        the node's probability is supposed to mean.  Returns ``False`` when
        ``p`` has no ``BACK`` edge to push back on.
        """
        e = self.children[p].get(BACK) if FIRST <= p < len(self.children) else None
        if e is None or amount <= 0:
            return False
        self.add_reward([e], -amount)
        return True

    def invert(self) -> None:
        """Rotate every edge's mean phase by ``pi`` and negate every reward.

        What arrived in phase now arrives in antiphase: the exact meaning of
        2NRL's inversion in a model whose scores are cosines.
        """
        for e, ok in enumerate(self.edge_alive):
            if ok:
                self.edge_cx[e] = -self.edge_cx[e]
                self.edge_cy[e] = -self.edge_cy[e]
                self.edge_reward[e] = -self.edge_reward[e]
        self.inverted = not self.inverted
        self.recompute_weights()

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        cx: list[float] = []
        cy: list[float] = []
        cw: list[float] = []
        reward: list[float] = []
        for old, ok in enumerate(self.alive):
            if ok:
                for _c, e in self.children[old].items():
                    cx.append(self.edge_cx[e])
                    cy.append(self.edge_cy[e])
                    cw.append(self.edge_cw[e])
                    reward.append(self.edge_reward[e])
        d["edges"].update(cx=cx, cy=cy, cw=cw, reward=reward)
        d["weights"] = {
            **self.weight_config(),
            "kind": "resonant",
            "total_traversals": self.total_traversals.value,
            "total_traversals_resets": self.total_traversals.resets,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResonantGraph":
        g = super().from_dict(d)
        weights = d.get("weights") or {}
        g.buckets = max(1, int(weights.get("buckets", 8)))
        g.period = float(weights.get("period", 0.0)) or float(g.buckets)
        g.kick_scale = float(weights.get("kick_scale", 0.0))
        g.resonance_scale = float(weights.get("resonance_scale", 1.0))
        g.amp_scale = float(weights.get("amp_scale", 1.0))
        g.reward_scale = float(weights.get("reward_scale", 1.0))
        g.concentration = float(weights.get("concentration", 2.0))
        g.total_traversals = CyclicCounter.from_pair(
            weights.get("total_traversals", 0), weights.get("total_traversals_resets", 0)
        )
        edges = d.get("edges", {})
        n = len(g.edge_w)
        for name, key in (("edge_cx", "cx"), ("edge_cy", "cy"), ("edge_cw", "cw"), ("edge_reward", "reward")):
            values = edges.get(key)
            if values is None:
                setattr(g, name, [0.0] * n)
                continue
            if len(values) != n:
                raise ValueError(f"edge {key} array has an inconsistent length")
            setattr(g, name, [float(v) for v in values])
        g.a = [0.0] * len(g.labels)
        g.k = [1.0] * len(g.labels)
        g._adv_cache = {}
        g._phase_costs = {}
        g._phase_cache_version = -1
        g.advance = [g.node_advance(n, lab) if ok else 0 for n, (lab, ok) in enumerate(zip(g.labels, g.alive))]
        g.recompute_weights()
        return g

    def __repr__(self) -> str:
        return (
            f"ResonantGraph(nodes={self.num_nodes()}, edges={self.num_edges()}, "
            f"buckets={self.buckets}, kick_scale={self.kick_scale}, inverted={self.inverted})"
        )


class ResonantNet(GraphModel):
    """The phase model (see the module docstring).

    ``train`` walks every text carrying its phase and counts each traversal
    into the edge's circular accumulator, so an edge learns *at which phase*
    it fires and how consistently; the cycle decisions the text made train the
    metacognitive layer beside it.  ``predict`` searches
    ``(node, chars, phase)``.  There is no gradient and no learning rate: a
    :class:`TrainConfig`'s ``lr`` / ``act_lr`` / ``batch_size`` are accepted
    and ignored, and the magnitude of feedback is ``strength`` (default 1: one
    unit of reward multiplies an edge's odds by ``e`` and pulls its phase lock
    tighter by half).
    """

    kind = "resonant"
    format = RESONANT_MODEL_FORMAT
    label = "Resonant (phase)"
    description = (
        "an analog phase rides along the walk: edges learn the phase at which they fire and how coherently, "
        "and a phase-locked cycle hands the decision to the metacognitive layer"
    )

    def __init__(
        self,
        seed: int = 0,
        backend: str = "auto",
        device: str | None = None,
        buckets: int = 8,
        period: float = 0.0,
        kick_scale: float = 0.0,
        resonance_scale: float = 1.0,
        amp_scale: float = 1.0,
        reward_scale: float = 1.0,
        concentration: float = 2.0,
        teach_back: bool = False,
        back_strength: float = 0.25,
        back_ceiling: float = 0.10,
    ) -> None:
        self.seed = int(seed)
        self.teach_back = bool(teach_back)
        """Whether a corpus declining a cycle also teaches the node's ``BACK`` edge (off; see :meth:`_teach_cycle`)."""
        self.back_strength = float(back_strength)
        """How loudly it does - one whole traversal's worth at 1.0."""
        self.back_ceiling = float(back_ceiling)
        """The share observation may push ``BACK`` to; above it, only experience speaks (see :meth:`_teach_cycle`)."""
        if self.back_strength < 0:
            raise ValueError(f"back_strength must be >= 0, got {back_strength}")
        if not (0.0 <= self.back_ceiling <= 1.0):
            raise ValueError(f"back_ceiling must lie in [0, 1], got {back_ceiling}")
        self.graph = ResonantGraph(
            seed=self.seed, buckets=buckets, period=period, kick_scale=kick_scale,
            resonance_scale=resonance_scale, amp_scale=amp_scale, reward_scale=reward_scale,
            concentration=concentration,
        )
        self.metacog = MetaLayer()
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
        # nothing numeric runs on a device; backend / device are accepted for interface parity with RadixNet
        self.backend = get_backend("python", None)
        self.history: list[dict] = []
        self.meta: dict = self._new_meta(self.seed)

    @staticmethod
    def _new_meta(seed: int) -> dict:
        meta = GraphModel._new_meta(seed)
        meta.update(rewards_total=0.0, penalties_total=0.0, feedback_passes=0, cycles_seen=0)
        return meta

    # -- walking a text with its phase ---------------------------------------

    def _walk(self, text: str, *, learn: bool, count: bool, reward: float, strength: float) -> dict:
        """Walk one text through the structure carrying its phase.

        Counts each traversal into its edge's circular accumulator at the phase
        the walk was in (``count``), moves the edge's reward (``reward``), and
        teaches the metacognitive layer what the text did at every cycle it had
        the option to close (``learn``).  Returns what the pass did.
        """
        graph = self.graph
        grams = self.encoder.encode(text)
        if not grams:
            return {"transitions": 0, "cycles": 0, "cost": 0.0}
        traced = graph._trace(grams)
        if traced is None:
            graph.observe_sequence(grams, count=False)
            traced = graph._trace(grams)
            if traced is None:  # pragma: no cover - observe_sequence guarantees walkability
                raise RuntimeError("internal error: observed sequence is not walkable")
        transitions, path = traced
        buckets = graph.buckets
        advance = graph.advance
        seen: dict[tuple[int, int], int] = {}
        bucket = 0
        cycles = 0
        total_cost = 0.0
        edges: list[int] = []
        for step, (p, e) in enumerate(transitions):
            child = path[step + 1]
            costs = {c: cost for c, _e, cost in graph.child_costs_at(p, bucket)}
            total_cost += costs.get(child, -_LOG_UNKNOWN)
            if learn:
                cycles += self._teach_cycle(p, bucket, child, seen, step, strength)
            if count:
                graph.record_traversal(e, bucket)
            edges.append(e)
            seen[(p, bucket)] = step
            bucket = (bucket + advance[child]) % buckets
        if reward:
            graph.add_reward(edges, reward)
        return {
            "transitions": len(transitions),
            "cycles": cycles,
            "cost": total_cost / max(1, len(transitions)),
            "edges": edges,
        }

    def _teach_cycle(self, p: int, bucket: int, taken: int, seen: dict, step: int, strength: float) -> int:
        """Teach the layer what the text did where a phase-locked cycle was on offer.

        A cycle decision only exists when one of ``p``'s children would land on
        a ``(node, phase)`` the walk has already been in.  When it does, the
        text either rode that child (``ride``), went to END (``abort``) or took
        something else (``escape``) - and that is the observation.

        A decision *not* to ride is also the lesson ``BACK`` (section 24's
        sentinel) is for, so with ``teach_back`` it is passed on: a walk reached
        ``p``, could have gone round, and did not.  That is the same thing a
        voice reports when it catches itself repeating and backs out
        (:func:`radixnet.dialogue.backtrack`), arrived at by observation rather
        than by experience, so it is taught the same way - the child that would
        have looped gets dearer, the one the text took instead gets cheaper, and
        ``p``'s hand-over estimate goes up.  It rolls the layer's per-cycle
        memory up into the node-level reflex, which is what primes the layer at
        cycles it has never seen.
        """
        graph = self.graph
        advance = graph.advance
        buckets = graph.buckets
        loops: dict[int, int] = {}
        for c in graph.children[p]:
            if c < FIRST:
                continue  # a sentinel is not a node to loop through
            first = seen.get((c, (bucket + advance[c]) % buckets))
            if first is not None:
                loops[c] = step + 1 - first
        if not loops:
            return 0
        target = min(loops, key=loops.__getitem__)
        signature = cycle_signature(graph.labels[target], loops[target])
        action = RIDE if taken in loops else (ABORT if taken == END else ESCAPE)
        self.metacog.observe(signature, action, strength)
        if self.teach_back and p >= FIRST and strength > 0:
            amount = strength * self.back_strength
            if action is RIDE:
                graph.observe_onward(p, amount)  # going round was right here: push the hand-over back down
            elif graph.back_probability(p) < self.back_ceiling:
                graph.observe_back(p, went=target, instead=None if taken == END else taken, amount=amount)
        return 1

    # -- passes over data ----------------------------------------------------

    def _passes(
        self,
        texts: Iterable[str] | str,
        cfg: TrainConfig,
        *,
        count: bool,
        reward: float,
        strength: float,
        phase: str | None,
        learn: bool = True,
        sharpen: float = 1.0,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """One routine behind ``train`` / ``reward`` / ``punish``: N epochs of walks over ``texts``."""
        cleaned, skipped = self._clean_texts(texts)
        graph = self.graph
        records: list[dict] = []
        if cleaned:
            # register everything structurally (and compress) before the first counting pass, so
            # every epoch - the first included - walks exactly the same transitions
            self._observe(cleaned, count=False)
            if cfg.auto_compress:
                graph.compress()
            self._observe(cleaned, count=False)
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            transitions = 0
            cycles = 0
            cost = 0.0
            touched: set[int] = set()
            for text in cleaned:
                done = self._walk(text, learn=learn, count=count, reward=reward, strength=strength)
                transitions += done["transitions"]
                cycles += done["cycles"]
                cost += done["cost"] * done["transitions"]
                touched.update(done.get("edges") or ())
            if sharpen != 1.0 and touched:
                graph.sharpen(touched, sharpen)
            if count or reward or sharpen != 1.0:
                graph.recompute_weights()
            graph.carry_counters()  # the epoch is over: wrap whatever reached the limit
            loss = cost / max(1, transitions)
            record = {
                "epoch": epoch,
                "loss": loss,
                "perplexity": math.exp(min(loss, _MAX_LOG_PPL)),
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
                "trigrams": graph.num_trigrams(),
                "compression_ratio": graph.compression_ratio(),
                "merges": 0,
                "transitions": transitions,
                "traversed": transitions if count else 0,
                "reward": reward * transitions,
                "cycles": cycles,
                "signatures": len(self.metacog),
                "seconds": time.perf_counter() - started,
                "skipped_short": skipped,
            }
            if phase:
                record["phase"] = phase
            self.history.append(record)
            records.append(record)
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self, epoch, "epoch", record)
            if stop_event is not None and stop_event.is_set():
                break
        self.meta["cycles_seen"] += sum(r["cycles"] for r in records)
        return records

    def train(
        self,
        texts: Iterable[str] | str,
        config: TrainConfig | None = None,
        *,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        phase: str | None = None,
        **overrides,
    ) -> list[dict]:
        """Learn from texts: count every traversal at its phase, and the cycle decisions beside it."""
        cfg = _resolve_config(config, overrides)
        cleaned, _ = self._clean_texts(texts)
        records = self._passes(
            texts, cfg, count=True, reward=0.0, strength=1.0, phase=phase,
            checkpoint_manager=checkpoint_manager, progress=progress, stop_event=stop_event,
        )
        self.meta["epochs_total"] += len(records)
        self.meta["trained_chars"] += sum(len(t) for t in cleaned)
        self.meta["trained_texts"] += len(cleaned)
        return records

    def _feedback_passes(
        self,
        texts: Iterable[str] | str,
        cfg: TrainConfig,
        *,
        count: bool,
        reward: float,
        strength: float,
        phase: str,
        sharpen_rate: float = 0.0,
        weights: Sequence[float] | None = None,
        name: str = "weights",
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """:meth:`_passes` over all the texts at once, or once per group of equally weighted texts.

        ``sharpen_rate`` is how far one unit of ``strength`` moves the phase
        lock: ``+0.5`` tightens it (a reward), ``-0.5`` scrambles it (a
        penalty), ``0`` leaves it alone.  ``weights`` (one per text, ``>= 0``)
        scales everything a pass does - the reward, the strength and with it
        the sharpening - so a text rated 9 out of 10 is treated nine tenths as
        hard as a perfect one.  Texts of equal weight share a pass (heaviest
        first) and their records carry ``"weight"``; a weight of 0 is skipped.
        """
        if weights is None:
            return self._passes(
                texts, cfg, count=count, reward=reward, strength=strength, phase=phase,
                sharpen=max(0.0, 1.0 + sharpen_rate * strength),
                checkpoint_manager=checkpoint_manager, progress=progress, stop_event=stop_event,
            )
        records: list[dict] = []
        for weight, group in _weight_groups(texts, weights, name):
            if stop_event is not None and stop_event.is_set():
                break
            group_records = self._passes(
                group, cfg, count=count, reward=reward * weight, strength=strength * weight, phase=phase,
                sharpen=max(0.0, 1.0 + sharpen_rate * strength * weight),
                checkpoint_manager=checkpoint_manager, stop_event=stop_event,
            )
            for record in group_records:
                record["weight"] = weight
                if progress is not None:
                    progress(record)
            records.extend(group_records)
        return records

    def reward(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs up: count the texts' paths, reward their edges and lock their phases tighter.

        ``weights`` (one per text, ``>= 0``) turns the thumbs up into a
        rating: a text of weight ``w`` is rewarded by ``w * strength`` and its
        phases are locked in proportion, so a text rated 9 out of 10 keeps
        nine tenths of what a perfect one would.  ``lr`` is accepted for
        interface parity with RadixNet and ignored.
        """
        base = abs(1.0 if strength is None else float(strength))
        records = self._feedback_passes(
            texts, _resolve_config(None, {**overrides, "epochs": epochs}),
            count=True, reward=base, strength=base, phase="positive", sharpen_rate=0.5, weights=weights,
            progress=progress, checkpoint_manager=checkpoint_manager, stop_event=stop_event,
        )
        self.meta["rewards_total"] += sum(r["reward"] for r in records)
        self.meta["feedback_passes"] += 1
        return records

    def punish(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs down: penalise the texts' edges and *decohere* them - their phase lock is scrambled.

        A penalty makes a path unlikely; decohering it also takes away the
        context in which it was right, which is the phase model's own way of
        forgetting.  The traversals are not counted.  ``weights`` rates the
        failures the way :meth:`reward` rates the successes: the worse a text,
        the larger its penalty and the harder its phases are scrambled.  ``lr``
        is accepted for interface parity with RadixNet and ignored.
        """
        base = abs(1.0 if strength is None else float(strength))
        records = self._feedback_passes(
            texts, _resolve_config(None, {**overrides, "epochs": epochs}),
            count=False, reward=-base, strength=base, phase="negative", sharpen_rate=-0.5, weights=weights,
            progress=progress, checkpoint_manager=checkpoint_manager, stop_event=stop_event,
        )
        self.meta["penalties_total"] += sum(-r["reward"] for r in records)
        self.meta["feedback_passes"] += 1
        return records

    def two_nrl(
        self,
        bad: Iterable[str] | str,
        good: Iterable[str] | str,
        neg_epochs: int = 3,
        pos_epochs: int = 3,
        neg_lr: float | None = None,
        pos_lr: float | None = None,
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
        strength: float | None = 1.0,
        bad_weights: Sequence[float] | None = None,
        good_weights: Sequence[float] | None = None,
        **overrides,
    ) -> dict:
        """2NRL: learn the garbage, invert, then fine-tune on the good data.

        Inversion is exact and cheap here - :meth:`ResonantGraph.invert`
        rotates every edge's mean phase by ``pi`` and negates every reward, so
        the phases the garbage just locked in now *cancel* rather than
        reinforce, and the metacognitive layer flips with them.  The
        fine-tuning pass then relocks the phases on the real texts.

        ``bad_weights`` (the evolve loop's per-failure boost) runs the negative
        phase once per distinct weight, heaviest first, at ``weight *
        strength``; ``good_weights`` rates the positive phase the same way - a
        rating, not a thumbs up.  ``neg_lr`` / ``pos_lr`` are accepted for
        interface parity with RadixNet and ignored; the magnitude per pass is
        ``strength``.  If ``stop_event`` is set after the negative phase the
        network is still inverted (so it never stays in the garbage-favouring
        state) but the positive phase is skipped.  With a
        ``checkpoint_manager`` one checkpoint tagged ``"2nrl"`` is written at
        the end (step = number of 2NRL runs).
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs")
        base = abs(1.0 if strength is None else float(strength))
        # the garbage is *learned*, phases and all; the inversion below is what turns that into avoidance
        negative = self._feedback_passes(
            bad, _resolve_config(None, {**overrides, "epochs": neg_epochs}),
            count=True, reward=0.0, strength=base, phase="negative",
            weights=bad_weights, name="bad_weights",
            progress=progress, checkpoint_manager=checkpoint_manager, stop_event=stop_event,
        )
        self.invert()
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            positive = self._feedback_passes(
                good, _resolve_config(None, {**overrides, "epochs": pos_epochs}),
                count=True, reward=base, strength=base, phase="positive",
                weights=good_weights, name="good_weights",
                progress=progress, checkpoint_manager=checkpoint_manager, stop_event=stop_event,
            )
        self.meta["twonrl_runs"] += 1
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, self.meta["twonrl_runs"], "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    def invert(self) -> None:
        """Rotate every edge's phase lock into antiphase and flip the metacognitive layer with it."""
        self.graph.invert()
        self.metacog.invert()

    def invert_paths(self, texts: Iterable[str] | str, mode: str = "activation", amounts=None, **_options) -> dict:
        """Failures: make the paths of ``texts`` unlikely locally, each by ``amounts[i]``.

        ``activation`` rotates the edges of a path by ``amount`` half-circles -
        a full amount puts them in antiphase, so where they used to resonate
        they now cancel; ``state`` decoheres them instead (``amount = 1``
        erases the phase lock entirely and the edges fall back to plain
        frequency).  Both are local and structural: nothing else moves.
        """
        mode = (mode or "activation").lower()
        if mode not in ("activation", "state"):
            raise ValueError(f"unknown mode {mode!r}; expected 'activation' or 'state'")
        cleaned, _ = self._clean_texts(texts)
        values = self._amounts(cleaned, amounts)
        flipped = 0
        used: list[float] = []
        for text, amount in zip(cleaned, values):
            if amount <= 0:
                continue
            done = self._walk(text, learn=False, count=False, reward=0.0, strength=0.0)
            edges = done.get("edges") or []
            if not edges:
                continue
            if mode == "activation":
                flipped += self.graph.rotate(edges, amount)
            else:
                flipped += self.graph.sharpen(edges, max(0.0, 1.0 - amount))
            used.append(amount)
        if flipped:
            self.graph.recompute_weights()
        return {
            "texts": len(cleaned),
            "flipped": flipped,
            "unit": "edges",
            "mode": mode,
            "amount_mean": (math.fsum(used) / len(used)) if used else 0.0,
        }

    # -- prediction ----------------------------------------------------------

    def _search(
        self,
        prefix: str,
        length: int,
        mode: str,
        k: int,
        beam: int | None,
        step_penalty: float,
        temperature: float,
        to_end: bool,
        max_length: int | None,
        rng=None,
    ) -> Prediction:
        """The shared search hook, routed through :meth:`predict`.

        ``GraphModel.generate(mode="sample")`` and
        :func:`radixnet.dialogue.converse` reach the search through here; the
        base implementation walks the phase-free graph, which would ignore both
        the phase and the metacognitive layer.
        """
        return self.predict(
            prefix, length=length, mode=mode, step_penalty=step_penalty, temperature=temperature,
            to_end=to_end, max_length=max_length, k=max(1, k), beam=beam, rng=rng,
        )

    def predict(
        self,
        prefix: str,
        length: int = 20,
        mode: str = "kbest",
        step_penalty: float = 0.0,
        temperature: float = 1.0,
        to_end: bool = False,
        max_length: int | None = None,
        k: int = 5,
        beam: int | None = None,
        rng=None,
    ) -> Prediction:
        """Continue ``prefix`` over the phase-unrolled graph.

        * ``"kbest"`` (the default) is Dijkstra with ``k`` labels per state
          instead of one: the ``k`` cheapest walks, *exactly*, and - because
          every label is a distinct walk that can be read back to its own path -
          with the metacognitive layer running on the cycles it meets.  ``k = 1``
          is ``"dijkstra"`` to the expansion.
        * ``"dijkstra"`` is the single cheapest walk, one label per state.  One
          label cannot say which walk reached it, so the layer does not run.
        * ``"beam"`` keeps the ``beam`` cheapest partial walks per step and is
          the only mode that also returns the ``k`` *least* likely
          continuations, which a k-best search cannot: in a cyclic graph the
          worst walk is unboundedly bad, so "worst" needs a frontier's bound
          rather than a goal count.
        * ``"sample"`` is one stochastic walk, layer included.
        """
        mode = (mode or "kbest").lower()
        if mode not in ("kbest", "beam", "dijkstra", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'kbest', 'beam', 'dijkstra' or 'sample'")
        self._check_predict_args(prefix, length, max_length, k, beam)
        graph = self.graph
        node, offset, lead = self._prefix_start(prefix)
        bucket = graph.text_bucket(prefix) if prefix else 0
        want = max(0, length - len(lead))
        cap: int | None
        if mode == "sample":
            cap = max_length if max_length is not None else length
            max_chars = max(0, cap - len(lead))
        elif max_length is None and length == 0:
            cap, max_chars = 0, 0      # "emit nothing" stays empty even without a cap
        elif max_length is None:
            cap, max_chars = None, None  # no limit: the whole path is returned
        else:
            cap = max(length, max_length)
            max_chars = max(want, cap - len(lead))
        width = 0
        if mode == "kbest":
            found, expanded = phase_kbest(
                graph, self.metacog, node, offset, bucket, min_chars=want, k=k, max_chars=max_chars,
                step_penalty=step_penalty, to_end=to_end,
            )
            top, bottom = found, []
            best = top[0] if top else None
        elif mode == "dijkstra":
            best = phase_dijkstra(
                graph, node, offset, bucket, min_chars=want, max_chars=max_chars,
                step_penalty=step_penalty, to_end=to_end,
            )
            top, bottom, expanded = [best], [], best.expanded
        elif mode == "beam":
            top, bottom, expanded = phase_beam(
                graph, self.metacog, node, offset, bucket, min_chars=want, k=k, beam=beam,
                max_chars=max_chars, step_penalty=step_penalty, to_end=to_end,
            )
            width = default_beam(k) if beam is None else int(beam)
            best = top[0] if top else None
        else:
            walk = phase_walk(
                graph, self.metacog, node, offset, bucket, max_chars=max_chars,
                temperature=temperature, rng=rng,
            )
            top, bottom, expanded, best = [walk], [], walk.expanded, walk
        for result in top + bottom:
            if lead:
                result.text = lead + result.text if cap is None else (lead + result.text)[:cap]
            result.full_text = prefix + result.text
        if best is None:
            best = PathResult(
                text=lead if cap is None else lead[: cap or 0],
                labels=[graph.labels[node]], node_ids=[node],
            )
            best.full_text = prefix + best.text
        return Prediction(
            text=best.text, labels=list(best.labels), node_ids=list(best.node_ids), cost=best.cost,
            step_costs=list(best.step_costs), expanded=expanded, reached_end=best.reached_end,
            full_text=best.full_text, top=top, bottom=bottom, k=k, beam=width, mode=mode,
        )

    def _edge_log_prob_at(self, p: int, offset: int, c: int, bucket: int) -> float | None:
        """``log P(c | p, phase)`` for the edge ``p -> c`` taken from ``p``'s trigram at ``offset``.

        ``None`` when ``p`` is not positioned at its last trigram (a transition
        out of the middle of a compressed node is impossible) or the edge does
        not exist - the phase-aware twin of :meth:`GraphModel._edge_log_prob`.
        """
        if p != START and offset + _W != len(self.graph.labels[p]):
            return None
        for child, _edge, cost in self.graph.child_costs_at(p, bucket):
            if child == c:
                return -cost
        return None

    def generate(
        self,
        max_length: int = 60,
        mode: str = "kbest",
        temperature: float = 1.0,
        count: int = 1,
        seed: int | None = None,
        prefix: str = "",
        step_penalty: float = 0.0,
        beam: int | None = None,
    ) -> list[PathResult]:
        """Whole texts from the prediction search; ``"kbest"`` (the default) returns the exact ``count`` cheapest.

        Generation asks the search for the ``count`` most likely *complete*
        texts, which is precisely what :func:`~radixnet.phasesearch.phase_kbest`
        answers exactly - and for a fraction of what a beam of the same ``k``
        costs, because it stops as soon as it has ``count`` finished walks.  The
        other modes are :meth:`GraphModel.generate`'s.
        """
        mode = (mode or "kbest").lower()
        if mode != "kbest":
            return super().generate(
                max_length=max_length, mode=mode, temperature=temperature, count=count, seed=seed,
                prefix=prefix, step_penalty=step_penalty, beam=beam,
            )
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if count == 0:
            return []
        found = self.predict(
            prefix, length=0, mode="kbest", k=count, to_end=True, max_length=max_length,
            step_penalty=step_penalty,
        )
        return [_whole_text(result) for result in found.top]

    def score(self, text: str) -> dict:
        """Log-probability of ``text`` under the model, phase included.

        The text's trigrams are walked ``START -> ... -> END`` through the
        structure carrying the phase, and every transition contributes
        ``log P(child | parent, phase)``.  Steps inside a compressed node are
        deterministic (cost 0) and are not counted as transitions, but they
        still move the phase on - a node's advance is the sum of its trigrams'.
        An unknown trigram, a missing edge or a transition that would need a
        split contributes ``log(UNKNOWN_PROB)`` and is counted in
        ``unknown_transitions``, so an unseen text scores *worse* than a
        trained one rather than looking deterministic.  Scoring never changes
        the model.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        grams = self.encoder.encode(text)
        chars = len(text)
        if not grams:
            return {"log_prob": 0.0, "per_char": 0.0, "chars": chars, "transitions": 0, "unknown_transitions": 0}
        graph = self.graph
        index = graph.trigram_index
        buckets = graph.buckets
        advance_of = graph.trigram_advance
        log_prob = 0.0
        transitions = 0
        unknown = 0
        node = START
        offset = 0
        bucket = 0        # the phase of the text consumed so far
        lost = False      # position unknown after an unknown trigram
        for t in grams:
            loc = index.get(t)
            if loc is None:
                transitions += 1
                unknown += 1
                log_prob += _LOG_UNKNOWN
                lost = True
            else:
                n, o = loc
                if not lost and n == node and o == offset + 1:
                    offset = o  # deterministic in-node step: no edge, no transition
                    bucket = (bucket + advance_of(t)) % buckets
                    continue
                transitions += 1
                lp = None if lost or o != 0 else self._edge_log_prob_at(node, offset, n, bucket)
                if lp is None:
                    unknown += 1
                    log_prob += _LOG_UNKNOWN
                else:
                    log_prob += lp
                node, offset, lost = n, o, False
            bucket = (bucket + advance_of(t)) % buckets
        transitions += 1
        lp = None if lost else self._edge_log_prob_at(node, offset, END, bucket)
        if lp is None:
            unknown += 1
            log_prob += _LOG_UNKNOWN
        else:
            log_prob += lp
        return {
            "log_prob": log_prob,
            "per_char": log_prob / max(1, chars),
            "chars": chars,
            "transitions": transitions,
            "unknown_transitions": unknown,
        }

    # -- configuration and reporting -----------------------------------------

    def configure_weights(self, **options: float) -> dict:
        """Change the score function at run time (see :meth:`ResonantGraph.configure`)."""
        return self.graph.configure(**options)

    def weight_config(self) -> dict:
        """The score function's current settings."""
        return self.graph.weight_config()

    def coherence(self) -> dict:
        """How phase-locked the model is overall: the mean and the best edge coherence."""
        graph = self.graph
        values = [graph.edge_coherence(e) for e, ok in enumerate(graph.edge_alive) if ok]
        if not values:
            return {"mean": 0.0, "max": 0.0, "edges": 0}
        return {"mean": math.fsum(values) / len(values), "max": max(values), "edges": len(values)}

    def stats(self) -> dict:
        """Everything the status bar, the CLI ``info`` table and ``/api/status`` show."""
        graph = self.graph
        coherence = self.coherence()
        rewards = 0.0
        penalties = 0.0
        for e, ok in enumerate(graph.edge_alive):
            if not ok:
                continue
            value = graph.edge_reward[e]
            if value > 0:
                rewards += value
            elif value < 0:
                penalties -= value
        return {
            "kind": self.kind,
            "nodes": graph.num_nodes(),
            "edges": graph.num_edges(),
            "trigrams": graph.num_trigrams(),
            "compression_ratio": graph.compression_ratio(),
            "inverted": graph.inverted,
            "backend": self.backend.name,
            "device": self.backend.device,
            "epochs_total": self.meta["epochs_total"],
            "trained_chars": self.meta["trained_chars"],
            "trained_texts": self.meta["trained_texts"],
            "twonrl_runs": self.meta["twonrl_runs"],
            "history_len": len(self.history),
            "last_loss": self.history[-1]["loss"] if self.history else None,
            "total_traversals": graph.total_traversals.value,
            "total_traversals_resets": graph.total_traversals.resets,
            "rewards_total": rewards,
            "penalties_total": penalties,
            "feedback_passes": self.meta["feedback_passes"],
            "buckets": graph.buckets,
            "coherence_mean": coherence["mean"],
            "coherence_max": coherence["max"],
            "cycles_seen": self.meta["cycles_seen"],
            "meta": self.metacog.stats(),
            "teach_back": self.teach_back,
            "back_strength": self.back_strength,
            "back_ceiling": self.back_ceiling,
            "weights": self.weight_config(),
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable document (``format`` names the kind for :func:`load_model`)."""
        return {
            "format": RESONANT_MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "meta": self.meta,
            "history": self.history,
            "backend": self.backend.name,
            "metacog": self.metacog.to_dict(),
            "cycles": {
                "teach_back": self.teach_back,
                "back_strength": self.back_strength,
                "back_ceiling": self.back_ceiling,
            },
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None) -> "ResonantNet":
        """Inverse of :meth:`to_dict`."""
        if d.get("format") != RESONANT_MODEL_FORMAT:
            raise ValueError(f"not a {RESONANT_MODEL_FORMAT} model document")
        model = cls(seed=int(d.get("meta", {}).get("seed", 0)), backend=backend, device=device)
        model.graph = ResonantGraph.from_dict(d["graph"])
        model.metacog = MetaLayer.from_dict(d.get("metacog"))
        cycles = d.get("cycles") or {}
        model.teach_back = bool(cycles.get("teach_back", False))
        model.back_strength = float(cycles.get("back_strength", 0.25))
        model.back_ceiling = float(cycles.get("back_ceiling", 0.10))
        model.history = list(d.get("history") or [])
        model.meta = {**model._new_meta(model.seed), **(d.get("meta") or {})}
        return model

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"ResonantNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"buckets={g.buckets}, signatures={len(self.metacog)}, inverted={g.inverted})"
        )
