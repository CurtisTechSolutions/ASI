"""Shortest-path prediction and stochastic sampling over the tree.

The cyclic graph has to search a *depth-unrolled* copy of itself: its state is
``(node, chars_emitted)``, because a node can be reached again and only the
characters emitted on the way tell the second visit from the first.  A tree
needs none of that.  Every node is reached by exactly one path, so the state
is the node, no node is ever pushed twice, no visited set exists, and the
search ends when the subtree does - "keep going until there is nowhere to go"
is a complete algorithm here.  There is no expansion budget because there is
nothing for one to guard against.

Moving over an edge ``p -> c`` costs ``-log softmax(children of p)[c] +
step_penalty`` (always ``>= 0``, the cyclic graph's cost) and emits
``len(label_c) - overlap`` characters; an END leaf emits nothing.  The start
node emits its deterministic remainder after the matched gram.

Because there are no cycles there are no negative cycles, so a cost may be
negative here: ``costs="signal"`` walks by the raw edge signal
``-(w * f_p * f_c)`` and ``step_penalty`` may be below zero.  Dijkstra's early
stop is only valid for non-negative costs, so under signed costs the search is
exhaustive over the subtree instead - still finite, still one visit per node.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush

from .tree import RadixTree

__all__ = ["COSTS", "PathResult", "cheapest_path", "sample_walk"]

COSTS = ("logprob", "signal")
"""What a step may cost: the non-negative ``-log P`` or the signed edge signal."""


@dataclass(slots=True)
class PathResult:
    """Outcome of a prediction / generation walk.

    ``text`` is the emitted continuation, ``labels`` / ``node_ids`` the path
    (sentinels included), ``cost`` the summed step cost, ``expanded`` how many
    nodes the search visited, ``reached_end`` whether the path ends at an END
    leaf, ``legs`` how many times the walk had to re-enter the tree (always 1
    on a tree of every suffix), ``full_text`` the prefix and the continuation.
    """

    text: str = ""
    labels: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    cost: float = 0.0
    step_costs: list[float] = field(default_factory=list)
    expanded: int = 0
    reached_end: bool = False
    full_text: str = ""
    legs: int = 1
    spelled: str = ""
    """``text`` spelled back into words when the units are sounds; ``text`` itself otherwise."""
    full_spelled: str = ""
    """``full_text`` spelled back into words when the units are sounds; ``full_text`` itself otherwise."""

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
            "legs": self.legs,
            "spelled": self.spelled,
            "full_spelled": self.full_spelled,
        }


def check_costs(costs: str) -> None:
    """``ValueError`` for a cost kind the search does not know."""
    if costs not in COSTS:
        raise ValueError(f"costs must be one of {', '.join(COSTS)}, got {costs!r}")


def _start_emission(tree: RadixTree, node: int, offset: int) -> int:
    """Characters the start node emits on its own: its remainder after the matched gram."""
    if node < 0 or node >= len(tree.labels) or not tree.alive[node]:
        raise ValueError(f"start node {node} is not alive")
    if not tree.is_real(node):
        return 0
    remainder = tree.label_len(node) - (offset + tree.encoding.n)
    if offset < 0 or remainder < 0:
        raise ValueError(f"start offset {offset} out of range for label {tree.labels[node]!r}")
    return remainder


def _build_result(
    tree: RadixTree,
    node_ids: list[int],
    step_costs: list[float],
    start_offset: int,
    max_chars: int | None,
    expanded: int,
    include_context: bool | None,
) -> PathResult:
    labels = [tree.labels[i] for i in node_ids]
    start = node_ids[0]
    if include_context is None:
        include_context = not tree.is_real(start)
    offset = start_offset if tree.is_real(start) else 0
    real = [tree.labels[i] for i in node_ids if tree.is_real(i)]
    text = tree.encoding.decode_path(real, offset, include_context)
    if max_chars is not None and max_chars >= 0:
        text = tree.encoding.truncate(text, max_chars)
    return PathResult(
        text=text,
        labels=labels,
        node_ids=node_ids,
        cost=math.fsum(step_costs),
        step_costs=step_costs,
        expanded=expanded,
        reached_end=bool(node_ids) and tree.is_end(node_ids[-1]),
    )


def cheapest_path(
    tree: RadixTree,
    start_node: int,
    start_offset: int,
    min_chars: int,
    max_chars: int | None = None,
    step_penalty: float = 0.0,
    to_end: bool = False,
    include_context: bool | None = None,
    costs: str = "logprob",
) -> PathResult:
    """Cheapest path from ``(start_node, start_offset)`` emitting ``>= min_chars``.

    Goal: with ``to_end`` an END leaf; otherwise the first node on a path with
    at least ``min_chars`` emitted (reaching END earlier also counts).  A path
    stops at its first goal.  ``max_chars`` is an optional hard cap: nodes
    with that many characters emitted are not expanded and the text is cut to
    it.  If no goal is reachable (the window ran out first) the visited node
    with the most characters emitted (ties: lowest cost) is returned; this
    never raises for a valid start.  ``include_context`` defaults to true
    from a sentinel and false from a real node (the continuation only).

    Under non-negative costs the search is Dijkstra and stops at the first
    goal it pops; under ``costs="signal"`` or a negative ``step_penalty`` it
    walks the whole subtree and keeps the cheapest goal it saw.
    """
    check_costs(costs)
    if min_chars < 0:
        raise ValueError(f"min_chars must be >= 0, got {min_chars}")
    if max_chars is not None and max_chars < min_chars:
        max_chars = min_chars
    nonneg = costs == "logprob" and step_penalty >= 0
    is_end = tree.is_end
    child_costs = tree.child_costs
    label_len = tree.label_len
    overlap = tree.encoding.overlap
    push = heappush
    pop = heappop
    start_chars = _start_emission(tree, start_node, start_offset)
    prev: dict[int, tuple[int, float]] = {}
    heap = [(0.0, 0, start_node, start_chars)]
    tie = 0
    expanded = 0
    goal: int | None = None
    goal_cost = math.inf
    fallback = start_node
    fb_chars = start_chars
    fb_cost = 0.0
    while heap:
        cost, _, node, chars = pop(heap)
        expanded += 1
        if is_end(node) or (not to_end and chars >= min_chars):
            if nonneg:
                goal = node
                break
            if cost < goal_cost:
                goal, goal_cost = node, cost
            continue  # a path stops at its first goal
        if chars > fb_chars or (chars == fb_chars and cost < fb_cost):
            fallback, fb_chars, fb_cost = node, chars, cost
        if max_chars is not None and chars >= max_chars:
            continue
        for c, ec in child_costs(node, costs):
            nchars = chars if is_end(c) else chars + label_len(c) - overlap
            step = ec + step_penalty
            prev[c] = (node, step)
            tie += 1
            push(heap, (cost + step, tie, c, nchars))
    end = goal if goal is not None else fallback
    node_ids: list[int] = []
    step_costs: list[float] = []
    node = end
    while True:
        node_ids.append(node)
        link = prev.get(node)
        if link is None or node == start_node:
            break
        step_costs.append(link[1])
        node = link[0]
    node_ids.reverse()
    step_costs.reverse()
    return _build_result(tree, node_ids, step_costs, start_offset, max_chars, expanded, include_context)


def sample_walk(
    tree: RadixTree,
    start_node: int,
    start_offset: int,
    max_chars: int | None,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    stop_at_end: bool = True,
    include_context: bool | None = None,
    costs: str = "logprob",
) -> PathResult:
    """Stochastic walk sampling each child from ``softmax(-cost / temperature)``.

    Stops at an END leaf (if ``stop_at_end``), once ``max_chars`` characters
    were emitted (``None``: only END or a leaf stops it), or at a node without
    options.  ``temperature == 0`` is greedy; negative temperatures are
    rejected.  ``rng`` defaults to the tree's own seeded generator.
    ``step_costs`` are the model's costs of each chosen edge (``-log p`` under
    ``"logprob"`` at temperature 1), comparable with :func:`cheapest_path`.
    """
    check_costs(costs)
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if rng is None:
        rng = tree.rng
    is_end = tree.is_end
    child_costs = tree.child_costs
    label_len = tree.label_len
    overlap = tree.encoding.overlap
    exp = math.exp
    node = start_node
    chars = _start_emission(tree, start_node, start_offset)
    node_ids = [node]
    step_costs: list[float] = []
    steps = 0
    while True:
        if (is_end(node) and stop_at_end) or (max_chars is not None and chars >= max_chars):
            break
        options = child_costs(node, costs)
        if not options:
            break
        if temperature == 0 or len(options) == 1:
            pick = min(options, key=lambda item: item[1])
        else:
            inv_t = 1.0 / temperature
            lowest = min(cst for _, cst in options)
            weights = [exp(-(cst - lowest) * inv_t) for _, cst in options]
            r = rng.random() * math.fsum(weights)
            pick = options[-1]
            acc = 0.0
            for item, wgt in zip(options, weights):
                acc += wgt
                if r < acc:
                    pick = item
                    break
        c, cst = pick
        step_costs.append(cst)
        node_ids.append(c)
        if not is_end(c):
            chars += label_len(c) - overlap
        node = c
        steps += 1
    return _build_result(tree, node_ids, step_costs, start_offset, max_chars, steps + 1, include_context)
