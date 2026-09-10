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
from dataclasses import dataclass, field
from heapq import heappop, heappush

from .encoding import WINDOW, Decoder
from .graph import END, START, RadixCyclicGraph

__all__ = ["PathResult", "dijkstra_predict", "sample_walk"]

_W = WINDOW
_OV = WINDOW - 1
_DECODER = Decoder(_W)


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


def _start_emission(graph: RadixCyclicGraph, start_node: int, start_offset: int) -> int:
    """Characters emitted by the start node (its remainder after the matched trigram)."""
    if start_node < 0 or start_node >= len(graph.labels) or not graph.alive[start_node]:
        raise ValueError(f"start node {start_node} is not alive")
    if start_node == START or start_node == END:
        return 0
    remainder = len(graph.labels[start_node]) - (start_offset + _W)
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
    max_chars: int,
    expanded: int,
    include_context: bool | None,
) -> PathResult:
    labels = [graph.labels[n] for n in node_ids]
    start_node = node_ids[0]
    if include_context is None:
        # From START there is no matched context to strip: emit the first node in full.
        include_context = start_node == START
    offset = 0 if start_node == START or start_node == END else start_offset
    # sentinels are stripped by id: a real node may carry the label "<s>" or "</s>"
    real = [lab for n, lab in zip(node_ids, labels) if n != START and n != END]
    text = _DECODER.decode_path(real, offset, include_context, skip_sentinels=False)
    if max_chars >= 0:
        text = text[:max_chars]
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
    max_chars: int,
    step_penalty: float = 0.0,
    to_end: bool = False,
    max_expansions: int = 200_000,
    include_context: bool | None = None,
) -> PathResult:
    """Cheapest path from ``(start_node, start_offset)`` emitting ``>= min_chars``.

    States with ``chars_emitted >= max_chars`` are not expanded.  Goal: with
    ``to_end`` the END node; otherwise the first popped state with at least
    ``min_chars`` emitted (Dijkstra pops in cost order, so it is the cheapest
    such path) - reaching END earlier also counts.  If no goal is reachable
    (dead end or ``max_expansions``), the popped state with the most emitted
    characters (ties: lowest cost) is returned; this never raises for a valid
    start.  ``include_context`` defaults to ``True`` from START and ``False``
    otherwise (see :meth:`Decoder.decode_path`); the text is truncated to
    ``max_chars``.
    """
    if step_penalty < 0:
        raise ValueError("step_penalty must be >= 0 (Dijkstra needs non-negative costs)")
    if max_chars < min_chars:
        max_chars = min_chars
    labels = graph.labels
    child_costs = graph.child_costs
    push = heappush
    pop = heappop
    inf = math.inf
    start_chars = _start_emission(graph, start_node, start_offset)
    start_key = (start_node, start_chars)
    best: dict[tuple[int, int], float] = {start_key: 0.0}
    best_get = best.get
    prev: dict[tuple[int, int], tuple[tuple[int, int], float]] = {}
    heap = [(0.0, 0, start_node, start_chars)]
    tie = 0
    expanded = 0
    goal: tuple[int, int] | None = None
    fallback = start_key
    fb_chars = start_chars
    fb_cost = 0.0
    while heap:
        cost, _, node, chars = pop(heap)
        key = (node, chars)
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
        if chars >= max_chars:
            continue
        for c, _e, ec in child_costs(node):
            nchars = chars if c == END else chars + len(labels[c]) - _OV
            step = ec + step_penalty
            ncost = cost + step
            nkey = (c, nchars)
            if ncost < best_get(nkey, inf):
                best[nkey] = ncost
                prev[nkey] = (key, step)
                tie += 1
                push(heap, (ncost, tie, c, nchars))
    end_key = goal if goal is not None else fallback
    node_ids: list[int] = []
    step_costs: list[float] = []
    key = end_key
    while True:
        node_ids.append(key[0])
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
    max_chars: int,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    stop_at_end: bool = True,
    include_context: bool | None = None,
) -> PathResult:
    """Stochastic walk sampling each child from ``softmax(scores / temperature)``.

    Stops at END (if ``stop_at_end``), once ``max_chars`` characters were
    emitted, or at a node without children.  ``temperature == 0`` is greedy
    (argmax); negative temperatures are rejected.  ``rng`` defaults to the
    graph's own seeded generator.  ``cost`` / ``step_costs`` are the model's
    ``-log p`` of each chosen edge (temperature 1), comparable with
    :func:`dijkstra_predict`.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if rng is None:
        rng = graph.rng
    labels = graph.labels
    child_costs = graph.child_costs
    exp = math.exp
    node = start_node
    chars = _start_emission(graph, start_node, start_offset)
    node_ids = [node]
    step_costs: list[float] = []
    steps = 0
    while True:
        if (node == END and stop_at_end) or chars >= max_chars:
            break
        costs = child_costs(node)
        if not costs:
            break
        if temperature == 0 or len(costs) == 1:
            pick = min(costs, key=lambda item: item[2])
        else:
            inv_t = 1.0 / temperature
            lowest = min(cst for _, _, cst in costs)
            weights = [exp(-(cst - lowest) * inv_t) for _, _, cst in costs]
            r = rng.random() * math.fsum(weights)
            pick = costs[-1]
            acc = 0.0
            for item, wgt in zip(costs, weights):
                acc += wgt
                if r < acc:
                    pick = item
                    break
        c, _e, cst = pick
        step_costs.append(cst)
        node_ids.append(c)
        if c != END:
            chars += len(labels[c]) - _OV
        node = c
        steps += 1
    return _build_result(graph, node_ids, step_costs, start_offset, max_chars, steps, include_context)
