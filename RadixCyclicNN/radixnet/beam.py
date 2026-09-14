"""Beam search over the graph: the K best *and* the K worst continuations in one prediction.

A state is a partial path ``(cost, chars_emitted, node, entry)`` where
``entry`` indexes a parent-pointer table, so every partial path is a distinct
text.  The *top* beam keeps the ``beam`` cheapest partial paths per step, the
*bottom* beam the ``beam`` most expensive ones.  A path is complete at END,
once it emitted ``min_chars`` characters (unless ``to_end``), or at a cap.
``beam_predict`` returns the ``k`` cheapest complete paths, the ``k`` most
expensive complete paths that are not among them, and the number of
expansions.  Costs are the same as Dijkstra's (``-log softmax`` per edge plus
``step_penalty``), so ``exp(-cost)`` is a path's probability.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .encoding import WINDOW
from .graph import END, RadixCyclicGraph
from .search import PathResult, _build_result, _start_emission

__all__ = ["Prediction", "beam_predict", "default_beam", "path_probability"]

_OV = WINDOW - 1


@dataclass(slots=True)
class Prediction(PathResult):
    """A :class:`PathResult` (the best path) plus the top-K and bottom-K paths of the same search."""

    top: list[PathResult] = field(default_factory=list)
    bottom: list[PathResult] = field(default_factory=list)
    k: int = 0
    beam: int = 0
    mode: str = "beam"

    def to_dict(self) -> dict:
        d = PathResult.to_dict(self)
        d.update(
            top=[p.to_dict() for p in self.top],
            bottom=[p.to_dict() for p in self.bottom],
            k=self.k,
            beam=self.beam,
            mode=self.mode,
        )
        return d


def default_beam(k: int) -> int:
    """Beam width used when none is given: wide enough for ``k`` distinct paths per side."""
    return max(4 * max(1, k), 16)


def path_probability(result: PathResult) -> float:
    """``exp(-cost)`` of a path (0 for an infinite cost)."""
    return math.exp(-result.cost) if math.isfinite(result.cost) else 0.0


def _run_beam(
    graph: RadixCyclicGraph,
    start_node: int,
    start_chars: int,
    min_chars: int,
    cap: int | None,
    k: int,
    width: int,
    step_penalty: float,
    to_end: bool,
    max_steps: int,
    max_expansions: int,
    worst: bool,
) -> tuple[list[tuple[float, list[int], list[float]]], int]:
    """One beam: the ``k`` cheapest (or, with ``worst``, dearest) complete paths as ``(cost, node_ids, step_costs)``."""
    labels = graph.labels
    child_costs = graph.child_costs
    entries: list[tuple[int, int, float]] = [(start_node, -1, 0.0)]  # entry -> (node, parent, step cost)
    sign = -1.0 if worst else 1.0  # heap keys: the k-th best finished path sits at the heap top

    def complete(node: int, chars: int) -> bool:
        if node == END:
            return True
        if cap is not None and chars >= cap:
            return True
        return not to_end and chars >= min_chars

    done: list[tuple[float, int]] = []  # bounded heap of (sign * -cost, entry): pops the worst kept path first

    def offer(cost: float, entry: int) -> None:
        if k == 0:
            return
        key = (-cost * sign, entry)
        if len(done) < k:
            heapq.heappush(done, key)
        elif key[0] > done[0][0]:
            heapq.heapreplace(done, key)

    start = (0.0, start_chars, start_node, 0)
    frontier: list[tuple[float, int, int, int]] = []
    if complete(start_node, start_chars):
        offer(0.0, 0)
    else:
        frontier.append(start)
    fallback = frontier
    expanded = 0
    steps = 0
    while frontier and steps < max_steps and expanded < max_expansions:
        steps += 1
        candidates: list[tuple[float, int, int, int]] = []
        for cost, chars, node, entry in frontier:
            expanded += 1
            for c, _e, ec in child_costs(node):
                nchars = chars if c == END else chars + len(labels[c]) - _OV
                step = ec + step_penalty
                ncost = cost + step
                child_entry = len(entries)
                entries.append((c, entry, step))
                if complete(c, nchars):
                    offer(ncost, child_entry)
                else:
                    candidates.append((ncost, nchars, c, child_entry))
            if expanded >= max_expansions:
                break
        if candidates:
            fallback = candidates
        frontier = heapq.nlargest(width, candidates) if worst else heapq.nsmallest(width, candidates)
        # costs only grow along a path: once k paths finished and every partial one is already
        # dearer than the k-th cheapest, the best side is settled
        if not worst and k > 0 and len(done) == k and frontier and frontier[0][0] >= -done[0][0]:
            break

    def path_of(entry: int) -> tuple[list[int], list[float]]:
        node_ids: list[int] = []
        step_costs: list[float] = []
        while entry >= 0:
            node, parent, step = entries[entry]
            node_ids.append(node)
            if parent >= 0:
                step_costs.append(step)
            entry = parent
        node_ids.reverse()
        step_costs.reverse()
        return node_ids, step_costs

    if done:
        finished = sorted(((-key * sign, entry) for key, entry in done), key=lambda item: (item[0] * sign, item[1]))
    elif fallback:
        # nothing completed within the limits: the surviving partial paths, most characters first
        pick = heapq.nsmallest(k, fallback, key=lambda s: (-s[1], s[0] * sign, s[3]))
        finished = [(s[0], s[3]) for s in pick]
    else:
        finished = []
    return [(cost, *path_of(entry)) for cost, entry in finished], expanded


def beam_predict(
    graph: RadixCyclicGraph,
    start_node: int,
    start_offset: int,
    min_chars: int,
    k: int = 5,
    beam: int | None = None,
    max_chars: int | None = None,
    step_penalty: float = 0.0,
    to_end: bool = False,
    max_steps: int | None = None,
    max_expansions: int = 200_000,
    include_context: bool | None = None,
) -> tuple[list[PathResult], list[PathResult], int]:
    """``(top, bottom, expanded)``: the ``k`` cheapest and the ``k`` most expensive complete paths.

    ``top`` is sorted by rising cost, ``bottom`` by falling cost, and no path
    appears in both.  With ``to_end`` and no ``max_chars`` the bottom side is
    capped at twice the length of the longest top path (plus a margin), so the
    "worst" paths stay comparable instead of cycling for hundreds of steps.
    When nothing completes within ``max_steps`` rounds (default ``min_chars +
    50``, or 500 with ``to_end``) or ``max_expansions``, the surviving partial
    paths are returned instead (``reached_end`` is ``False`` on them).  A start
    that already satisfies the goal gives one complete path (the start itself).
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if step_penalty < 0:
        raise ValueError("step_penalty must be >= 0")
    width = default_beam(k) if beam is None else int(beam)
    if width < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")
    if max_chars is not None and max_chars < min_chars:
        max_chars = min_chars
    if max_steps is None:
        max_steps = 500 if to_end else min_chars + 50
    start_chars = _start_emission(graph, start_node, start_offset)
    if k == 0:
        return [], [], 0
    common = (graph, start_node, start_chars, min_chars)
    limits = (k, width, step_penalty, to_end, max_steps, max_expansions)
    best, expanded = _run_beam(*common, max_chars, *limits, False)
    bottom_cap = max_chars
    if bottom_cap is None and to_end:
        longest = max((len(graph.labels[n]) for _, ids, _ in best for n in ids[1:]), default=0)
        emitted = max((sum(len(graph.labels[n]) - _OV for n in ids[1:] if n != END) for _, ids, _ in best), default=0)
        bottom_cap = max(2 * emitted + longest + 8, min_chars, 16)
    worst, expanded_worst = _run_beam(*common, bottom_cap, *limits, True)
    expanded += expanded_worst
    seen = {tuple(ids) for _, ids, _ in best}
    top = [_build_result(graph, ids, steps, start_offset, max_chars, expanded, include_context) for _, ids, steps in best]
    bottom = [
        _build_result(graph, ids, steps, start_offset, bottom_cap, expanded, include_context)
        for _, ids, steps in worst
        if tuple(ids) not in seen
    ]
    return top, bottom, expanded
