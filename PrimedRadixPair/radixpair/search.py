"""Walks over the pair: the cheapest path (Dijkstra), the greedy walk and the sampled walk.

A walk's state is ``(node, level, emitted)``.  From a context node it may
*emit* a unit (down one level, at the cost the rung sets), *fall* to the
context without its oldest unit (at ``-log(1 - own)``, the count tree's price
for not trusting this context), or - at level ``L``, where a node has no
children - *shift* to the same sequence without its oldest unit at no cost:
the window slides.  The costs are the fold's terms without the floor, so a
path's cost is one choice of how far to fall at every step and never below the
un-floored fold's cost of the same units.  ``greedy`` walks the exact fold one
unit at a time instead, and is the model's default: a cheapest single path on a
primed tree can pay for a rare unit whose unread context then falls to the root
for nothing (``DESIGN.md`` section 10).
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .pair import BACKOFFS, RadixPair, TRAVERSALS

__all__ = ["MODES", "PathResult", "dijkstra", "greedy"]

MODES = ("dijkstra", "greedy", "sample")


@dataclass
class PathResult:
    text: str
    units: list[int]
    node_ids: list[int]
    hops: list[tuple[str, int]]
    cost: float
    step_costs: list[float]
    traversal: str
    expanded: int
    reached_end: bool
    mode: str = "dijkstra"
    full_text: str = ""
    fallback: bool = False
    symbols: list[str] = field(default_factory=list)
    spelled: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text, "full_text": self.full_text, "spelled": self.spelled, "units": list(self.units),
            "symbols": list(self.symbols),
            "node_ids": list(self.node_ids), "hops": [[kind, i] for kind, i in self.hops], "cost": self.cost,
            "step_costs": list(self.step_costs), "traversal": self.traversal, "mode": self.mode,
            "expanded": self.expanded, "reached_end": self.reached_end, "fallback": self.fallback,
        }


def _finish(pair: RadixPair, units: list[int], node_ids: list[int], hops: list[tuple[str, int]], cost: float,
            step_costs: list[float], traversal: str, expanded: int, reached_end: bool, mode: str,
            fallback: bool = False) -> PathResult:
    codec = pair.codec
    return PathResult(
        text=codec.decode(units), units=units, node_ids=node_ids, hops=hops, cost=cost, step_costs=step_costs,
        traversal=traversal, expanded=expanded, reached_end=reached_end, mode=mode, fallback=fallback,
        symbols=[codec.symbol(u) for u in units],
    )


def dijkstra(pair: RadixPair, context: Sequence[int], min_units: int, max_units: int | None = None,
             to_end: bool = False, traversal: str = "reward", step_penalty: float | None = None,
             max_expansions: int | None = None) -> PathResult:
    """The cheapest walk from a context that emits ``min_units`` units, or reaches the end mark."""
    if traversal not in TRAVERSALS:
        raise ValueError(f"traversal must be one of {TRAVERSALS}, got {traversal!r}")
    s = pair.settings
    penalty = s.step_penalty if step_penalty is None else float(step_penalty)
    cap = s.max_expansions if max_expansions is None else int(max_expansions)
    if min_units < 0 or (max_units is not None and max_units < min_units and not to_end):
        raise ValueError("min_units >= 0 and max_units >= min_units")
    address, count, codec = pair.address, pair.count, pair.codec
    emits, end, L, D = pair.emits, codec.end, pair.L, pair.D
    R_out = pair.R_out
    start_node, start_level = pair.context_node(context)
    start = (start_node, 0)
    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[tuple[int, int] | None, str, int | None, float]] = {start: (None, "start", None, 0.0)}
    level_of = {start_node: start_level}
    heap = [(0.0, 0, start_node, 0)]
    tie = 0
    expanded = 0
    goal = None
    most: tuple[int, float, tuple[int, int]] | None = None
    while heap:
        cost, _, node, e = heapq.heappop(heap)
        state = (node, e)
        if cost > best.get(state, math.inf):
            continue
        l = level_of[node]
        if most is None or (e, -cost) > (most[0], -most[1]):
            most = (e, cost, state)
        if not to_end and e >= min_units and (e > 0 or min_units == 0):
            goal = state
            break
        if expanded >= cap:
            break
        expanded += 1
        if max_units is not None and e >= max_units:
            continue

        def relax(to_node: int, to_level: int, to_e: int, hop: str, unit: int | None, c: float) -> None:
            nonlocal tie
            to_state = (to_node, to_e)
            new = cost + c
            if new < best.get(to_state, math.inf):
                best[to_state] = new
                parent[to_state] = (state, hop, unit, c)
                level_of[to_node] = to_level
                tie += 1
                heapq.heappush(heap, (new, tie, to_node, to_e))

        if l == L:
            relax(address.drop_oldest(node, L), D, e, "shift", None, 0.0)
            continue
        own = count.own(node, l)
        if own > 0.0 or l == 0:
            q = pair.q(node, l, traversal)
            uniform = (1.0 - own) / R_out if l == 0 else 0.0
            for j, x in enumerate(emits):
                p = own * q[j] + uniform
                if p <= 0.0:
                    continue
                c = -math.log(p) + penalty
                child = address.append(node, l, x)
                if x == end:
                    end_state = (child, e)
                    new = cost + c
                    if new < best.get(end_state, math.inf):
                        best[end_state] = new
                        parent[end_state] = (state, "emit", x, c)
                        level_of[child] = l + 1
                        if to_end or e >= min_units:
                            goal = end_state
                            break
                    continue
                relax(child, l + 1, e + 1, "emit", x, c)
            if goal is not None:
                break
        if l >= 1 and own < 1.0:
            relax(address.drop_oldest(node, l), l - 1, e, "fall", None, -math.log(1.0 - own) if own > 0.0 else 0.0)
    fallback = goal is None
    if goal is None:
        goal = most[2] if most is not None else start
    reached_end = parent[goal][2] == end if goal in parent else False
    # reconstruct
    units: list[int] = []
    node_ids: list[int] = []
    hops: list[tuple[str, int]] = []
    step_costs: list[float] = []
    pending = 0.0
    chain = []
    state = goal
    while state is not None:
        prev, hop, unit, c = parent[state]
        chain.append((state, hop, unit, c))
        state = prev
    chain.reverse()
    for state, hop, unit, c in chain:
        if hop == "start":
            continue
        hops.append((hop, state[0]))
        if hop == "emit":
            if unit != end:
                units.append(unit)
                node_ids.append(state[0])
            step_costs.append(pending + c)
            pending = 0.0
        else:
            pending += c
    total = best[goal]
    return _finish(pair, units, node_ids, hops, total, step_costs, traversal, expanded, reached_end, "dijkstra", fallback)


def greedy(pair: RadixPair, context: Sequence[int], units: int, traversal: str = "reward",
           rng: random.Random | None = None, temperature: float = 0.0, to_end: bool = False,
           backoff: str | None = None) -> PathResult:
    """One unit at a time from the exact fold: the argmax at temperature 0, a sample above it."""
    if traversal not in TRAVERSALS:
        raise ValueError(f"traversal must be one of {TRAVERSALS}, got {traversal!r}")
    if backoff is not None and backoff not in BACKOFFS:
        raise ValueError(f"backoff must be one of {BACKOFFS}, got {backoff!r}")
    if temperature < 0.0:
        raise ValueError("temperature >= 0")
    if temperature > 0.0 and rng is None:
        rng = random.Random(0)
    ctx = list(context)
    emits, end, D = pair.emits, pair.codec.end, pair.D
    out: list[int] = []
    node_ids: list[int] = []
    hops: list[tuple[str, int]] = []
    step_costs: list[float] = []
    cost = 0.0
    reached_end = False
    steps = 0
    limit = units if units > 0 else (10 ** 9 if to_end else 0)
    while steps < limit:
        window = ctx[-D:] if D > 0 else []
        P = pair.fold(window, traversal, backoff)
        if temperature == 0.0:
            j = max(range(len(P)), key=lambda k: (P[k], -emits[k]))
        else:
            logits = [math.log(p) / temperature for p in P]
            top = max(logits)
            w = [math.exp(v - top) for v in logits]
            r = rng.random() * sum(w)
            acc = 0.0
            j = len(w) - 1
            for k, v in enumerate(w):
                acc += v
                if r < acc:
                    j = k
                    break
        x = emits[j]
        node, l = pair.context_node(window)
        step = pair.address.append(node, l, x)
        c = -math.log(P[j])
        cost += c
        step_costs.append(c)
        hops.append(("emit", step))
        if x == end:
            reached_end = True
            break
        out.append(x)
        node_ids.append(step)
        ctx.append(x)
        steps += 1
    mode = "greedy" if temperature == 0.0 else "sample"
    return _finish(pair, out, node_ids, hops, cost, step_costs, traversal, steps, reached_end, mode)
