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
from .graph import END, START, RadixCyclicGraph
from .search import (
    LEAST_PUNISHED,
    CostFn,
    PathResult,
    _build_result,
    _start_emission,
    least_punished,
    onward,
    parse_traversal,
)

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
    traversal: str = "reward"
    """Which traversal wrote this: ``"reward"``, ``"punishment"`` (:mod:`radixnet.penalty` prices the steps)
    or ``"least-punished"`` (``../SPEC-LeastPunished.md`` ranks the walks)."""

    def to_dict(self) -> dict:
        d = PathResult.to_dict(self)
        d.update(
            top=[p.to_dict() for p in self.top],
            bottom=[p.to_dict() for p in self.bottom],
            k=self.k,
            beam=self.beam,
            mode=self.mode,
            traversal=self.traversal,
        )
        if self.traversal and self.traversal != "reward":
            d["traversal"] = self.traversal
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
    traversal: str = "reward",
    costs: CostFn | None = None,
) -> tuple[list[tuple[float, list[int], list[float], float]], int]:
    """One beam: the ``k`` cheapest (or, with ``worst``, dearest) complete paths
    as ``(cost, node_ids, step_costs, punish)``.

    ``costs`` replaces the graph's own cost function, which is how the
    punishment traversal prices a step (:mod:`radixnet.penalty`).  Under
    ``traversal="least-punished"`` "cheapest" instead reads as "least punished,
    and cheapest among those": a path is ranked by its *worst* step first and by
    its summed cost only where two paths carry the same worst step.
    """
    labels = graph.labels
    blamed = traversal == LEAST_PUNISHED
    child_costs = graph.child_steps if blamed else (graph.child_costs if costs is None else costs)
    entries: list[tuple[int, int, float]] = [(start_node, -1, 0.0)]  # entry -> (node, parent, step cost)
    sign = -1.0 if worst else 1.0  # heap keys: the k-th best finished path sits at the heap top

    def complete(node: int, chars: int) -> bool:
        if node == END:
            return True
        if cap is not None and chars >= cap:
            return True
        return not to_end and chars >= min_chars

    # a bounded heap of (sign * -punish, sign * -cost, entry): the worst kept path
    # is at the top, and under the reward traversal the first component is 0.0 on
    # every path, so the whole comparison is the cost order the beam has always used
    done: list[tuple[float, float, int]] = []

    def offer(punish: float, cost: float, entry: int) -> None:
        if k == 0:
            return
        key = (-punish * sign if blamed else 0.0, -cost * sign, entry)
        if len(done) < k:
            heapq.heappush(done, key)
        elif key[:2] > done[0][:2]:
            heapq.heapreplace(done, key)

    start = (0.0, 0.0, start_chars, start_node, 0)
    frontier: list[tuple[float, float, int, int, int]] = []
    if complete(start_node, start_chars):
        offer(0.0, 0.0, 0)
    else:
        frontier.append(start)
    fallback = frontier
    expanded = 0
    steps = 0
    while frontier and steps < max_steps and expanded < max_expansions:
        steps += 1
        candidates: list[tuple[float, float, int, int, int]] = []
        for punish, cost, chars, node, entry in frontier:
            expanded += 1
            parent_entry = entries[entry][1]
            # where the walk came from: a model that counts paths prices the next step by it
            prev = entries[parent_entry][0] if parent_entry >= 0 else (START if node == START else None)
            children = onward(child_costs(node, prev))
            if blamed:
                children = least_punished(children)
            for item in children:
                c, ec = item[0], item[2]
                nchars = chars if c == END else chars + len(labels[c]) - _OV
                step = ec + step_penalty
                ncost = cost + step
                npunish = max(punish, item[3]) if blamed else 0.0  # a path is as punished as its worst step
                child_entry = len(entries)
                entries.append((c, entry, step))
                if complete(c, nchars):
                    offer(npunish, ncost, child_entry)
                else:
                    candidates.append((npunish, ncost, nchars, c, child_entry))
            if expanded >= max_expansions:
                break
        if candidates:
            fallback = candidates
        frontier = heapq.nlargest(width, candidates) if worst else heapq.nsmallest(width, candidates)
        # neither number shrinks along a path - a cost is a sum of costs and a punishment the
        # worst step so far - so once k paths finished and the best partial one already ranks
        # below the k-th of them, the best side is settled
        if not worst and k > 0 and len(done) == k and frontier:
            head = (-frontier[0][0] * sign if blamed else 0.0, -frontier[0][1] * sign)
            if head <= tuple(done[0][:2]):
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
        # 0.0 - x rather than -x: a negative zero reads as a punishment that is not there
        kept = [(0.0 - punish * sign, 0.0 - cost * sign, entry) for punish, cost, entry in done]
        finished = sorted(kept, key=lambda item: (item[0] * sign if blamed else 0.0, item[1] * sign, item[2]))
    elif fallback:
        # nothing completed within the limits: the surviving partial paths, most characters first
        pick = heapq.nsmallest(
            k, fallback, key=lambda s: (-s[2], s[0] * sign if blamed else 0.0, s[1] * sign, s[4])
        )
        finished = [(s[0], s[1], s[4]) for s in pick]
    else:
        finished = []
    return [(cost, *path_of(entry), punish) for punish, cost, entry in finished], expanded


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
    traversal: str = "reward",
    costs: CostFn | None = None,
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
    ``costs`` replaces the graph's own cost function: with the punishment
    traversal's (:mod:`radixnet.penalty`) ``top`` is the ``k`` *least punished*
    continuations and ``bottom`` the ``k`` most punished ones.
    ``traversal="least-punished"`` does something else again: it ranks a path by
    the blame on its worst step before its cost, and lets a node offer only the
    children it has the least against (``../SPEC-LeastPunished.md``).
    """
    traversal = parse_traversal(traversal)
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
    best, expanded = _run_beam(*common, max_chars, *limits, False, traversal, costs)
    bottom_cap = max_chars
    if bottom_cap is None and to_end:
        longest = max((len(graph.labels[n]) for _, ids, _, _ in best for n in ids[1:]), default=0)
        emitted = max(
            (sum(len(graph.labels[n]) - _OV for n in ids[1:] if n != END) for _, ids, _, _ in best), default=0
        )
        bottom_cap = max(2 * emitted + longest + 8, min_chars, 16)
    worst, expanded_worst = _run_beam(*common, bottom_cap, *limits, True, traversal, costs)
    expanded += expanded_worst
    seen = {tuple(ids) for _, ids, _, _ in best}

    def built(rows: list, cap: int | None) -> list[PathResult]:
        out = []
        for _cost, ids, steps, punish in rows:
            result = _build_result(graph, ids, steps, start_offset, cap, expanded, include_context)
            result.punish = punish
            out.append(result)
        return out

    top = built(best, max_chars)
    bottom = built([row for row in worst if tuple(row[1]) not in seen], bottom_cap)
    return top, bottom, expanded
