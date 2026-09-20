"""The punishment traversal: a walk priced by what the network was punished for, not by what it was rewarded for.

Every search in this package - Dijkstra, the two beams, the k-best walk, the
sampler, and their phase-unrolled twins in :mod:`radixnet.phasesearch` - reads
the graph through one funnel: ``[(child, edge, cost)]`` for a node, with
``cost = -log P(child | parent)``.  Which cost function fills that list is the
**traversal**, and there are two of them:

``"reward"`` (the default)
    what the model believes.  The count / reward model's probability carries
    ``exp(reward_scale * reward)``, so a path the tutor rewarded is cheap and
    the search *follows the rewards*.

``"punishment"``
    what the model was punished for.  The rewards are taken out of the score
    altogether and only the **penalties** price the step, so the cheapest path
    is the one that accumulated the **least punishment**.  Nothing the network
    was praised for makes a step cheaper here; only what it was corrected for
    makes one dearer.

Why the two differ
------------------

Rewards and penalties are not symmetric evidence.  A reward says *this was
good once*; a penalty says *this was wrong, and here is the correction*.  The
first is an invitation to repeat a success and pulls the search towards
whatever the tutor happened to praise; the second is a boundary, and a walk
that respects every boundary it has been taught is not the same walk as one
that chases every reward it has been given.  A model whose rewards are sparse
(a handful of thumbs up) but whose penalties are dense (a tutor that corrected
a thousand sentences) has far more to say in the second currency than in the
first, and this traversal is how it gets to say it.

The split
---------

The traversal rests on one hook, :meth:`RadixCyclicGraph.child_evidence`, which
every model kind implements in its own currency and which splits an edge's
evidence in two:

``merit``
    what speaks *for* the step with every reward taken out of it - frequency,
    structure, resonance: what the corpus did, not what a judge said about it;
``penalty >= 0``
    what speaks *against* it - the punishment the edge carries.

The step's score is then ``merit_scale * merit - penalty_scale * penalty`` and
the cost the usual ``-log softmax`` over the parent's children, so costs stay
non-negative (Dijkstra keeps working), ``exp(-cost)`` is still a path's
probability, and the numbers stay comparable with the reward traversal's.
``merit_scale = 0`` is the pure form: nothing but the punishment decides, and
among equally unpunished children the walk is indifferent.

Where each model keeps its punishments
--------------------------------------

===============================  =======================================  ==================================
model                            merit                                    penalty
===============================  =======================================  ==================================
:class:`~radixnet.model.RadixNet`        the positive part of ``w * f_p * f_c``   the negative part of it
:class:`~radixnet.countnet.CountRewardNet`  the dual frequency function, rewards out  ``reward_scale * max(0, -reward)``
:class:`~radixnet.resonance.ResonantNet`    amplitude and resonance, rewards out      ``reward_scale * max(0, -reward)``
:class:`~radixnet.negative.NegativeNet`     ``log(1 + cleared text)``                 ``log(1 + net blame)``
===============================  =======================================  ==================================

The sine model keeps no separate ledger of its punishments: 2NRL trains a
failure in and then inverts it, so what a punishment leaves behind *is* a
negative score on the edges of the path - which is why its penalty is read
straight off the score.  The count / reward model and the phase model keep
``edge_reward``, whose negative half is the punishment and whose positive half
this traversal drops.  The negative network is nothing but punishment, so its
currency is blame against cleared text.  A judged path context
(:meth:`CountRewardGraph.path_term`) splits the same way: the part of it that
says *this step was wrong here* is a penalty, the part that says *this step was
right here* is merit.
"""

from __future__ import annotations

import math

__all__ = [
    "DEFAULT_TRAVERSAL",
    "TRAVERSALS",
    "PenaltyCosts",
    "PhasePenaltyCosts",
    "phase_traversal_costs",
    "resolve_traversal",
    "traversal_costs",
    "LEAST_PUNISHED",
]

TRAVERSALS = ("reward", "punishment", "least-punished")
"""The traversals a search can run: follow the rewards, avoid the punishments, or rank a walk by the blame on it.

The first two are **cost functions** and live here.  The third is a different
kind of thing - a *ranking*, which orders a walk by the punishment on its worst
step before its cost, and which only the count model can run because only it
keeps the judged paths that ranking reads (:mod:`radixnet.search`,
``../SPEC-LeastPunished.md``).  It is named here so one flag offers all three
and no search has to guess which registry a name came from; the cost functions
below simply do not apply to it.
"""

LEAST_PUNISHED = "least-punished"
"""The ranking traversal, priced in the count model's own currency rather than in a cost."""

DEFAULT_TRAVERSAL = "reward"
"""What every search runs unless told otherwise - the behaviour of every release before this option existed."""


def resolve_traversal(traversal: str | None) -> str:
    """Normalise a traversal name (``None`` / ``""`` = the default); raises ``ValueError`` for anything else."""
    name = (traversal or DEFAULT_TRAVERSAL).lower()
    if name not in TRAVERSALS:
        raise ValueError(f"unknown traversal {name!r}; expected {' or '.join(repr(t) for t in TRAVERSALS)}")
    return name


def _softmax_costs(items: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """``[(child, edge, score)] -> [(child, edge, -log softmax(score))]``, numerically stable."""
    if not items:
        return []
    m = max(s for _c, _e, s in items)
    lse = m + math.log(math.fsum(math.exp(s - m) for _c, _e, s in items))
    return [(c, e, lse - s) for c, e, s in items]


class _Costs:
    """Shared plumbing of the two punishment cost functions: the scales and a cache tied to ``graph.version``."""

    __slots__ = ("graph", "penalty_scale", "merit_scale", "_cache", "_version")

    def __init__(self, graph, penalty_scale: float = 1.0, merit_scale: float = 1.0) -> None:
        if penalty_scale < 0:
            raise ValueError(f"penalty_scale must be >= 0, got {penalty_scale}")
        if merit_scale < 0:
            raise ValueError(f"merit_scale must be >= 0, got {merit_scale}")
        self.graph = graph
        self.penalty_scale = float(penalty_scale)
        self.merit_scale = float(merit_scale)
        self._cache: dict = {}
        self._version = -1

    def _fresh(self) -> dict:
        """The cache, emptied when the graph moved under it."""
        if self._version != self.graph.version:
            self._cache.clear()
            self._version = int(self.graph.version)
        return self._cache

    def _price(self, evidence: list[tuple[int, int, float, float]]) -> list[tuple[int, int, float]]:
        merit_scale, penalty_scale = self.merit_scale, self.penalty_scale
        return _softmax_costs([
            (c, e, merit_scale * merit - penalty_scale * penalty) for c, e, merit, penalty in evidence
        ])


class PenaltyCosts(_Costs):
    """``child_costs`` for the punishment traversal: the least punished child is the cheapest.

    Call it exactly as :meth:`RadixCyclicGraph.child_costs` - ``(parent)`` or
    ``(parent, prev)`` - and hand it to any search as its ``costs``.  The
    result is cached per ``(parent, prev)`` until the graph's ``version``
    changes, like the graph's own cost cache.
    """

    def __call__(self, p: int, prev: int | None = None) -> list[tuple[int, int, float]]:
        graph = self.graph
        cache = self._fresh()
        if prev is not None and p not in graph.nodes_with_paths():
            prev = None  # the context makes no difference here, so one entry serves every caller
        key = (p, prev)
        costs = cache.get(key)
        if costs is None:
            costs = cache[key] = self._price(graph.child_evidence(p, prev))
        return costs


class PhasePenaltyCosts(_Costs):
    """``child_costs_at`` for the punishment traversal over the phase-unrolled graph (:class:`ResonantGraph`)."""

    def __call__(self, p: int, bucket: int) -> list[tuple[int, int, float]]:
        graph = self.graph
        cache = self._fresh()
        key = (p, bucket % graph.buckets)
        costs = cache.get(key)
        if costs is None:
            costs = cache[key] = self._price(graph.child_evidence_at(p, key[1]))
        return costs


def traversal_costs(
    graph, traversal: str | None = DEFAULT_TRAVERSAL, penalty_scale: float = 1.0, merit_scale: float = 1.0
):
    """The cost function a search should read the graph through, or ``None`` for the model's own.

    ``None`` is what the reward traversal returns: the searches then call
    ``graph.child_costs`` directly and nothing about them changes.
    """
    if resolve_traversal(traversal) in ("reward", LEAST_PUNISHED):
        return None  # the least-punished traversal reads the blame itself, not through a cost function
    return PenaltyCosts(graph, penalty_scale, merit_scale)


def phase_traversal_costs(
    graph, traversal: str | None = DEFAULT_TRAVERSAL, penalty_scale: float = 1.0, merit_scale: float = 1.0
):
    """:func:`traversal_costs` for the phase-unrolled searches: ``(node, bucket) -> costs``, or ``None``."""
    if resolve_traversal(traversal) in ("reward", LEAST_PUNISHED):
        return None
    return PhasePenaltyCosts(graph, penalty_scale, merit_scale)
