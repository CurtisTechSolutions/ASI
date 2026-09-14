"""Search over the phase-unrolled graph, with the metacognitive handoff on cycles.

A state here is ``(node, chars_emitted, phase_bucket)`` rather than
:mod:`radixnet.search`'s ``(node, chars_emitted)``.  The extra coordinate is
what :class:`~radixnet.resonance.ResonantGraph` scores edges against, and it is
a plain function of the state: entering node ``c`` moves the phase on by
``graph.advance[c]`` buckets.  Costs are ``-log softmax`` over the children *at
that phase* (plus ``step_penalty``), so ``exp(-cost)`` is still a path's
probability and the results stay comparable with the other two models'.

Because the phase is part of the state, the cost of an edge depends on the
whole path that led to it - a three-character graph with memory - and yet the
search stays exact: the product graph is finite (``buckets`` times bigger), so
optimal substructure holds and :func:`phase_dijkstra` is a true shortest path
over it.  ``kick_scale = 0`` makes the phase a function of the emitted length
alone, and the product collapses back to the ordinary unrolled graph at no
extra cost.

Cycles
------

Revisiting a node at a *different* phase is progress; revisiting it at the
*same* phase is a loop that would repeat for ever.  The two searches that carry
a path - :func:`phase_beam` and :func:`phase_walk` - watch for the second kind
and hand the decision to a :class:`~radixnet.metacog.MetaLayer`: the cost of
closing the loop takes the layer's ``ride`` cost, every other child takes its
``escape`` cost and END takes ``abort``.  A cycle the corpus rides stays cheap
to ride, one it never rides becomes expensive, and neither is forbidden.
:func:`phase_dijkstra` keeps its exactness instead: it prices the phase but not
the path's history, so it runs without the layer.
"""

from __future__ import annotations

import math
import random
from heapq import heappop, heappush

from .beam import default_beam
from .encoding import WINDOW
from .graph import END
from .metacog import ABORT, ESCAPE, RIDE, cycle_signature
from .search import PathResult, _build_result, _start_emission

__all__ = ["phase_beam", "phase_dijkstra", "phase_walk", "start_bucket"]

_OV = WINDOW - 1


def start_bucket(graph, start_node: int, start_offset: int, prefix: str = "") -> int:
    """Phase a prediction starts at: the phase of the text already behind it.

    A node's advance is the sum of its trigrams', so the phase after a piece of
    text is a function of the text alone (:meth:`ResonantGraph.text_bucket`) -
    no walk needed, and exactly what a walk that emitted it would carry.  What
    the start node has *not* emitted yet (the part of its label before the
    matched trigram) is already inside ``prefix``, so nothing is added for it.
    """
    return graph.text_bucket(prefix) if prefix else 0


def _meta_costs(meta, label: str, loop_len: int) -> dict[str, float]:
    """``{action: extra cost}`` from the metacognitive layer for one cycle."""
    if meta is None:
        return {RIDE: 0.0, ESCAPE: 0.0, ABORT: 0.0}
    return meta.costs(cycle_signature(label, loop_len))


def phase_dijkstra(
    graph,
    start_node: int,
    start_offset: int,
    start_phase: int,
    min_chars: int,
    max_chars: int | None = None,
    step_penalty: float = 0.0,
    to_end: bool = False,
    max_expansions: int = 200_000,
    include_context: bool | None = None,
) -> PathResult:
    """Cheapest path over ``(node, chars, phase)`` emitting at least ``min_chars``.

    The same contract as :func:`radixnet.search.dijkstra_predict` - goal, cap,
    fallback and ``include_context`` all behave identically - with the phase
    added to the state and to the costs.  No metacognition: this is the exact
    mode, and an exact search cannot depend on which path reached a state.
    """
    if step_penalty < 0:
        raise ValueError("step_penalty must be >= 0 (Dijkstra needs non-negative costs)")
    if max_chars is not None and max_chars < min_chars:
        max_chars = min_chars
    labels = graph.labels
    advance = graph.advance
    buckets = graph.buckets
    child_costs_at = graph.child_costs_at
    inf = math.inf
    start_chars = _start_emission(graph, start_node, start_offset)
    start_key = (start_node, start_chars, start_phase % buckets)
    best: dict[tuple[int, int, int], float] = {start_key: 0.0}
    best_get = best.get
    prev: dict[tuple[int, int, int], tuple[tuple[int, int, int], float]] = {}
    heap = [(0.0, 0, start_key)]
    tie = 0
    expanded = 0
    goal: tuple[int, int, int] | None = None
    fallback, fb_chars, fb_cost = start_key, start_chars, 0.0
    while heap:
        cost, _, key = heappop(heap)
        if cost > best[key]:
            continue  # stale entry
        node, chars, phase = key
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
        for c, _e, ec in child_costs_at(node, phase):
            nchars = chars if c == END else chars + len(labels[c]) - _OV
            step = ec + step_penalty
            ncost = cost + step
            nkey = (c, nchars, phase if c == END else (phase + advance[c]) % buckets)
            if ncost < best_get(nkey, inf):
                best[nkey] = ncost
                prev[nkey] = (key, step)
                tie += 1
                heappush(heap, (ncost, tie, nkey))
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


class _Entry:
    """One partial path in a beam: its tail state, its parent and what it has seen."""

    __slots__ = ("node", "chars", "phase", "cost", "parent", "step", "seen", "depth")

    def __init__(self, node, chars, phase, cost, parent, step, seen, depth):
        self.node = node
        self.chars = chars
        self.phase = phase
        self.cost = cost
        self.parent = parent
        self.step = step
        self.seen = seen              # frozenset of (node, phase) already on this path
        self.depth = depth            # position of each state, for the loop length

    def path(self) -> tuple[list[int], list[float]]:
        nodes: list[int] = []
        steps: list[float] = []
        entry: "_Entry | None" = self
        while entry is not None:
            nodes.append(entry.node)
            if entry.parent is not None:
                steps.append(entry.step)
            entry = entry.parent
        nodes.reverse()
        steps.reverse()
        return nodes, steps


def _expand(graph, meta, entry: _Entry, step_penalty: float):
    """Children of a beam entry as ``(child, edge, step_cost, phase, action)``.

    ``action`` is the metacognitive action the move stands for when this entry
    faces a phase-locked cycle, and ``None`` when it does not.  The layer's
    cost is added to the edge's, so a cycle the corpus rides stays cheap and
    one it never rides is dear - neither is ruled out.
    """
    node, phase = entry.node, entry.phase
    advance = graph.advance
    buckets = graph.buckets
    labels = graph.labels
    raw = graph.child_costs_at(node, phase)
    extra: dict[str, float] | None = None
    loops: dict[int, int] = {}
    if meta is not None and entry.seen:
        here = entry.depth.get((node, phase), 0)
        for c, _e, _cost in raw:
            if c == END:
                continue
            nphase = (phase + advance[c]) % buckets
            first = entry.depth.get((c, nphase))
            if first is not None:
                loops[c] = here + 1 - first  # how many steps the loop would close over
        if loops:
            target = min(loops, key=loops.__getitem__)  # the tightest loop names the cycle
            extra = _meta_costs(meta, labels[target], loops[target])
    out = []
    for c, e, cost in raw:
        nphase = phase if c == END else (phase + advance[c]) % buckets
        action = None
        if extra is not None:
            action = RIDE if c in loops else (ABORT if c == END else ESCAPE)
            cost += extra[action]
        out.append((c, e, cost + step_penalty, nphase, action))
    return out


def phase_beam(
    graph,
    meta,
    start_node: int,
    start_offset: int,
    start_phase: int,
    min_chars: int,
    k: int = 5,
    beam: int | None = None,
    max_chars: int | None = None,
    step_penalty: float = 0.0,
    to_end: bool = False,
    max_steps: int = 4_000,
    include_context: bool | None = None,
) -> tuple[list[PathResult], list[PathResult], int]:
    """The ``k`` cheapest and the ``k`` dearest complete paths, with metacognition on cycles.

    Mirrors :func:`radixnet.beam.beam_predict` - a top beam of the cheapest
    partial paths and a bottom beam of the dearest, both bounded by ``beam`` -
    over the phase-unrolled graph.  Every entry carries the ``(node, phase)``
    states already on its path, so a cycle is detected per path and priced by
    ``meta``.
    """
    width = default_beam(k) if beam is None else int(beam)
    if width < 1:
        raise ValueError(f"beam must be >= 1, got {width}")
    if max_chars is not None and max_chars < min_chars:
        max_chars = min_chars
    labels = graph.labels
    start_chars = _start_emission(graph, start_node, start_offset)
    phase0 = start_phase % graph.buckets
    root = _Entry(start_node, start_chars, phase0, 0.0, None, 0.0, frozenset(), {})
    root.seen = frozenset({(start_node, phase0)})
    root.depth = {(start_node, phase0): 0}
    top_beam = [root]
    bottom_beam = [root]
    top_done: list[tuple[float, int, _Entry]] = []
    bottom_done: list[tuple[float, int, _Entry]] = []
    seq = 0
    expanded = 0
    depth = 0
    limit_chars = max_chars
    bottom_limit = max_chars
    while (top_beam or bottom_beam) and depth < max_steps:
        depth += 1
        nxt_top: list[_Entry] = []
        nxt_bottom: list[_Entry] = []
        for side, source, sink in ((0, top_beam, nxt_top), (1, bottom_beam, nxt_bottom)):
            cap_chars = limit_chars if side == 0 else bottom_limit
            for entry in source:
                if entry.node == END:
                    continue
                if cap_chars is not None and entry.chars >= cap_chars:
                    continue
                expanded += 1
                for c, _e, step, nphase, _action in _expand(graph, meta, entry, step_penalty):
                    nchars = entry.chars if c == END else entry.chars + len(labels[c]) - _OV
                    if cap_chars is not None and nchars > cap_chars and c != END:
                        continue
                    key = (c, nphase)
                    child = _Entry(
                        c, nchars, nphase, entry.cost + step, entry, step,
                        entry.seen | {key}, {**entry.depth, key: depth},
                    )
                    complete = c == END or (not to_end and nchars >= min_chars)
                    if complete:
                        seq += 1
                        done = top_done if side == 0 else bottom_done
                        score = child.cost if side == 0 else -child.cost
                        if len(done) < max(1, k):
                            heappush(done, (-score, seq, child))
                        elif -score > done[0][0]:
                            heappop(done)
                            heappush(done, (-score, seq, child))
                        if c == END:
                            continue
                    sink.append(child)
        nxt_top.sort(key=lambda e: e.cost)
        nxt_bottom.sort(key=lambda e: -e.cost)
        top_beam = nxt_top[:width]
        bottom_beam = nxt_bottom[:width]
        if bottom_limit is None and len(top_done) >= max(1, k):
            # no cap and running to the end: keep the dearest paths roughly as long as the cheapest,
            # so "least likely" stays comparable instead of cycling for hundreds of steps
            bottom_limit = 2 * max(e.chars for _s, _q, e in top_done)
        if len(top_done) >= max(1, k) and top_beam:
            worst_kept = max(-s for s, _, _ in top_done)
            if top_beam[0].cost >= worst_kept:
                top_beam = []  # nothing left can beat the k-th finished path
        if not top_beam and not bottom_beam:
            break
    def _results(done):
        out = []
        for _score, _seq, entry in done:
            nodes, steps = entry.path()
            out.append(_build_result(graph, nodes, steps, start_offset, max_chars, expanded, include_context))
        return out

    top = sorted(_results(top_done), key=lambda r: r.cost)[: max(0, k)]
    taken = {r.text for r in top}
    bottom = [r for r in sorted(_results(bottom_done), key=lambda r: -r.cost) if r.text not in taken][: max(0, k)]
    return top, bottom, expanded


def phase_walk(
    graph,
    meta,
    start_node: int,
    start_offset: int,
    start_phase: int,
    max_chars: int | None,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    stop_at_end: bool = True,
    include_context: bool | None = None,
) -> PathResult:
    """A stochastic walk over the phase-unrolled graph, metacognition included.

    Samples each child from ``softmax(-cost / temperature)`` where the cost is
    the phase-aware one plus, at a phase-locked cycle, the layer's price for
    riding, escaping or aborting.  ``temperature == 0`` is greedy.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if rng is None:
        rng = graph.rng
    labels = graph.labels
    node = start_node
    phase = start_phase % graph.buckets
    chars = _start_emission(graph, start_node, start_offset)
    node_ids = [node]
    step_costs: list[float] = []
    entry = _Entry(node, chars, phase, 0.0, None, 0.0, frozenset({(node, phase)}), {(node, phase): 0})
    steps = 0
    while True:
        if (node == END and stop_at_end) or (max_chars is not None and chars >= max_chars):
            break
        options = _expand(graph, meta, entry, 0.0)
        if not options:
            break
        if temperature == 0 or len(options) == 1:
            pick = min(options, key=lambda item: item[2])
        else:
            inv_t = 1.0 / temperature
            lowest = min(cst for _c, _e, cst, _p, _a in options)
            weights = [math.exp(-(cst - lowest) * inv_t) for _c, _e, cst, _p, _a in options]
            r = rng.random() * math.fsum(weights)
            pick = options[-1]
            acc = 0.0
            for item, wgt in zip(options, weights):
                acc += wgt
                if r < acc:
                    pick = item
                    break
        c, _e, cst, nphase, _action = pick
        step_costs.append(cst)
        node_ids.append(c)
        if c != END:
            chars += len(labels[c]) - _OV
        steps += 1
        key = (c, nphase)
        entry = _Entry(c, chars, nphase, entry.cost + cst, None, cst, entry.seen | {key}, {**entry.depth, key: steps})
        node, phase = c, nphase
    return _build_result(graph, node_ids, step_costs, start_offset, max_chars, steps, include_context)
