"""Shortest-path prediction (Dijkstra) and stochastic sampling over the graph.

Prediction is a search over the *depth-unrolled* graph: a state is
``(node_id, chars_emitted)``.  Moving over edge ``p -> c`` costs
``-log softmax(children of p)[c] + step_penalty`` (always ``>= 0``) and emits
``len(label_c) - 2`` characters (the part of ``c`` that does not overlap
``p``).  The start node emits its deterministic remainder after the matched
trigram; START and END emit nothing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from heapq import heappop, heappush

from .encoding import WINDOW, Decoder
from .graph import BACK, END, FIRST, START, RadixCyclicGraph

__all__ = ["CostFn", "PathResult", "dijkstra_predict", "sample_walk"]

CostFn = Callable[..., list[tuple[int, int, float]]]
"""What a search reads the graph through: ``(parent[, prev]) -> [(child, edge, cost)]``.

:meth:`RadixCyclicGraph.child_costs` is the default one; the punishment
traversal (:mod:`radixnet.penalty`) hands in a
:class:`~radixnet.penalty.PenaltyCosts` instead, and every search behaves
exactly as it always did - it is only reading a different cost function.
"""

_W = WINDOW      # the default n; a graph's own n is graph.encoding.n
_OV = WINDOW - 1  # and its own overlap graph.encoding.overlap


@dataclass(slots=True)
class PathResult:
    """Outcome of a prediction / generation walk.

    ``text`` is the emitted continuation, ``labels`` / ``node_ids`` the path
    (sentinels included), ``cost`` the summed edge cost (``== sum(step_costs)``),
    ``expanded`` the number of Dijkstra expansions (or sampler steps),
    ``reached_end`` whether the path ends at END.  ``full_text`` is filled in
    by ``RadixNet`` (prefix + text).
    """

    text: str = ""
    labels: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    cost: float = 0.0
    step_costs: list[float] = field(default_factory=list)
    expanded: int = 0
    reached_end: bool = False
    full_text: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "text": self.text,
            "labels": list(self.labels),
            "node_ids": list(self.node_ids),
            "cost": self.cost,
            "step_costs": list(self.step_costs),
            "expanded": self.expanded,
            "reached_end": self.reached_end,
            "full_text": self.full_text,
        }


def onward(costs: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """The children a walk may actually take, given what the model has learned about going round.

    ``BACK`` is not a continuation - it emits nothing and no text passes through it - so it never appears in a
    path.  But it *competes* with the real children for probability, and when it is the cheapest of them the
    model's most likely next step at this node is to stop rather than carry on: the walk hands over, which here
    means the branch offers nothing and the search goes on with its others (:meth:`RadixCyclicGraph.observe_back`).
    """
    onward = [item for item in costs if item[0] != BACK]
    if len(onward) == len(costs):
        return costs
    back = min(cost for c, _e, cost in costs if c == BACK)
    return [] if all(cost >= back for _c, _e, cost in onward) else onward


def _start_emission(graph: RadixCyclicGraph, start_node: int, start_offset: int) -> int:
    """Units emitted by the start node (its remainder after the matched gram)."""
    if start_node < 0 or start_node >= len(graph.labels) or not graph.alive[start_node]:
        raise ValueError(f"start node {start_node} is not alive")
    if start_node < FIRST:
        return 0
    remainder = graph.label_len(start_node) - (start_offset + graph.encoding.n)
    if start_offset < 0 or remainder < 0:
        raise ValueError(
            f"start_offset {start_offset} out of range for label {graph.labels[start_node]!r}"
        )
    return remainder


def _build_result(
    graph: RadixCyclicGraph,
    node_ids: list[int],
    step_costs: list[float],
    start_offset: int,
    max_chars: int | None,
    expanded: int,
    include_context: bool | None,
) -> PathResult:
    labels = [graph.labels[n] for n in node_ids]
    start_node = node_ids[0]
    if include_context is None:
        # From START there is no matched context to strip: emit the first node in full.
        include_context = start_node == START
    offset = 0 if start_node < FIRST else start_offset
    # sentinels are stripped by id: a real node may carry the label "<s>" or "</s>"
    real = [lab for n, lab in zip(node_ids, labels) if n >= FIRST]
    text = graph.encoding.decode_path(real, offset, include_context, skip_sentinels=False)
    if max_chars is not None and max_chars >= 0:
        text = graph.encoding.truncate(text, max_chars)
    return PathResult(
        text=text,
        labels=labels,
        node_ids=node_ids,
        cost=math.fsum(step_costs),
        step_costs=step_costs,
        expanded=expanded,
        reached_end=node_ids[-1] == END,
    )


def dijkstra_predict(
    graph: RadixCyclicGraph,
    start_node: int,
    start_offset: int,
    min_chars: int,
    max_chars: int | None = None,
    step_penalty: float = 0.0,
    to_end: bool = False,
    max_expansions: int = 200_000,
    include_context: bool | None = None,
    costs: CostFn | None = None,
) -> PathResult:
    """Cheapest path from ``(start_node, start_offset)`` emitting ``>= min_chars``.

    ``max_chars`` is an optional hard cap: states with ``chars_emitted >=
    max_chars`` are not expanded and the text is truncated to it; ``None``
    (the default) means no limit - the whole cheapest path is returned.  Goal: with
    ``to_end`` the END node; otherwise the first popped state with at least
    ``min_chars`` emitted (Dijkstra pops in cost order, so it is the cheapest
    such path) - reaching END earlier also counts.  If no goal is reachable
    (dead end or ``max_expansions``), the popped state with the most emitted
    characters (ties: lowest cost) is returned; this never raises for a valid
    start.  ``include_context`` defaults to ``True`` from START and ``False``
    otherwise (see :meth:`Decoder.decode_path`); the text is truncated to
    ``max_chars``.  ``costs`` replaces the graph's own cost function: it is
    how the punishment traversal walks the least punished path instead of the
    most rewarded one (:mod:`radixnet.penalty`).
    """
    if step_penalty < 0:
        raise ValueError("step_penalty must be >= 0 (Dijkstra needs non-negative costs)")
    if max_chars is not None and max_chars < min_chars:
        max_chars = min_chars
    labels = graph.labels
    child_costs = graph.child_costs if costs is None else costs
    overlap = graph.encoding.overlap
    push = heappush
    pop = heappop
    inf = math.inf
    start_chars = _start_emission(graph, start_node, start_offset)
    # a model that counts paths prices a step by the node the walk came from, so the state has to carry it -
    # but only where it makes a difference, which keeps the search as small as it was everywhere else
    context = graph.nodes_with_paths()
    start_from = START if start_node == START else -1
    start_key = (start_from if start_node in context else -1, start_node, start_chars)
    best: dict[tuple[int, int, int], float] = {start_key: 0.0}
    best_get = best.get
    prev: dict[tuple[int, int, int], tuple[tuple[int, int, int], float]] = {}
    heap = [(0.0, 0, start_from, start_node, start_chars)]
    tie = 0
    expanded = 0
    goal: tuple[int, int, int] | None = None
    fallback = start_key
    fb_chars = start_chars
    fb_cost = 0.0
    while heap:
        cost, _, came_from, node, chars = pop(heap)
        key = (came_from if node in context else -1, node, chars)
        if cost > best[key]:
            continue  # stale entry
        expanded += 1
        if node == END or (not to_end and chars >= min_chars):
            goal = key
            break
        if chars > fb_chars or (chars == fb_chars and cost < fb_cost):
            fallback, fb_chars, fb_cost = key, chars, cost
        if expanded >= max_expansions:
            break
        if max_chars is not None and chars >= max_chars:
            continue
        for c, _e, ec in onward(child_costs(node, came_from if came_from >= 0 else None)):
            nchars = chars if c == END else chars + graph.label_len(c) - overlap
            step = ec + step_penalty
            ncost = cost + step
            nkey = (node if c in context else -1, c, nchars)
            if ncost < best_get(nkey, inf):
                best[nkey] = ncost
                prev[nkey] = (key, step)
                tie += 1
                push(heap, (ncost, tie, node, c, nchars))
    end_key = goal if goal is not None else fallback
    node_ids: list[int] = []
    step_costs: list[float] = []
    key = end_key
    while True:
        node_ids.append(key[1])
        link = prev.get(key)
        if link is None:
            break
        step_costs.append(link[1])
        key = link[0]
    node_ids.reverse()
    step_costs.reverse()
    return _build_result(graph, node_ids, step_costs, start_offset, max_chars, expanded, include_context)


def sample_walk(
    graph: RadixCyclicGraph,
    start_node: int,
    start_offset: int,
    max_chars: int | None,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    stop_at_end: bool = True,
    include_context: bool | None = None,
    costs: CostFn | None = None,
) -> PathResult:
    """Stochastic walk sampling each child from ``softmax(scores / temperature)``.

    Stops at END (if ``stop_at_end``), once ``max_chars`` characters were
    emitted (``None`` = no limit: only END or a dead end stops the walk), or
    at a node without children.  ``temperature == 0`` is greedy
    (argmax); negative temperatures are rejected.  ``rng`` defaults to the
    graph's own seeded generator.  ``cost`` / ``step_costs`` are the model's
    ``-log p`` of each chosen edge (temperature 1), comparable with
    :func:`dijkstra_predict`.  ``costs`` replaces the graph's own cost
    function, which is how this walk samples what the network was *not*
    punished for (:mod:`radixnet.penalty`).
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if rng is None:
        rng = graph.rng
    labels = graph.labels
    child_costs = graph.child_costs if costs is None else costs
    overlap = graph.encoding.overlap
    exp = math.exp
    node = start_node
    chars = _start_emission(graph, start_node, start_offset)
    node_ids = [node]
    step_costs: list[float] = []
    steps = 0
    came_from = START if start_node == START else None
    while True:
        if (node == END and stop_at_end) or (max_chars is not None and chars >= max_chars):
            break
        options = onward(child_costs(node, came_from))  # a node the model expects to go round offers nothing
        if not options:
            break
        if temperature == 0 or len(options) == 1:
            pick = min(options, key=lambda item: item[2])
        else:
            inv_t = 1.0 / temperature
            lowest = min(cst for _, _, cst in options)
            weights = [exp(-(cst - lowest) * inv_t) for _, _, cst in options]
            r = rng.random() * math.fsum(weights)
            pick = options[-1]
            acc = 0.0
            for item, wgt in zip(options, weights):
                acc += wgt
                if r < acc:
                    pick = item
                    break
        c, _e, cst = pick
        step_costs.append(cst)
        node_ids.append(c)
        if c != END:
            chars += graph.label_len(c) - overlap
        came_from = node
        node = c
        steps += 1
    return _build_result(graph, node_ids, step_costs, start_offset, max_chars, steps, include_context)
