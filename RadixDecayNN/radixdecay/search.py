"""The cheapest path over the tree, by shares of ``seen``, and what a walk returns.

Costs are ``-log share``: non-negative, so Dijkstra stops at the first goal
it pops.  Every node is reached by exactly one path, so the state is the node,
nothing is pushed twice, and the search ends when the subtree does - the
acyclic sibling's search, with the shares where its softmax was.  The search
itself is a measurement: it visits without arriving.  Whether the path it
returns is then *traversed* is the model's decision (:mod:`radixdecay.model`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from heapq import heappop, heappush

from .tree import DecayTree

__all__ = ["PathResult", "cheapest_path"]


@dataclass(slots=True)
class PathResult:
    """Outcome of a walk: the text, the path, its cost, how many nodes were visited or queries made, whether it
    reached END, the whole text, its words when the units are sounds, and what the walk added to ``seen``."""

    text: str = ""
    labels: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    cost: float = 0.0
    step_costs: list[float] = field(default_factory=list)
    expanded: int = 0
    reached_end: bool = False
    full_text: str = ""
    spelled: str = ""
    full_spelled: str = ""
    traversals: int = 0
    """How many traversals the walk made - what it added to ``seen`` (0 for a quiet walk)."""

    def to_dict(self) -> dict:
        return {
            "text": self.text, "labels": list(self.labels), "node_ids": list(self.node_ids), "cost": self.cost,
            "step_costs": list(self.step_costs), "expanded": self.expanded, "reached_end": self.reached_end,
            "full_text": self.full_text, "spelled": self.spelled, "full_spelled": self.full_spelled,
            "traversals": self.traversals,
        }


def start_emission(tree: DecayTree, node: int, offset: int) -> int:
    """Units the start node emits on its own: its remainder after the matched gram."""
    if node < 0 or node >= len(tree.labels) or not tree.alive[node]:
        raise ValueError(f"start node {node} is not alive")
    if not tree.is_real(node):
        return 0
    remainder = tree.label_len(node) - (offset + tree.encoding.n)
    if offset < 0 or remainder < 0:
        raise ValueError(f"start offset {offset} out of range for label {tree.labels[node]!r}")
    return remainder


def build_result(
    tree: DecayTree, node_ids: list[int], step_costs: list[float], start_offset: int, max_units: int | None,
    expanded: int, include_context: bool | None,
) -> PathResult:
    labels = [tree.labels[i] for i in node_ids]
    start = node_ids[0]
    if include_context is None:
        include_context = not tree.is_real(start)
    offset = start_offset if tree.is_real(start) else 0
    real = [tree.labels[i] for i in node_ids if tree.is_real(i)]
    text = tree.encoding.decode_path(real, offset, include_context)
    if max_units is not None and max_units >= 0:
        text = tree.encoding.truncate(text, max_units)
    return PathResult(
        text=text, labels=labels, node_ids=node_ids, cost=math.fsum(step_costs), step_costs=step_costs,
        expanded=expanded, reached_end=bool(node_ids) and tree.is_end(node_ids[-1]),
    )


def cheapest_path(
    tree: DecayTree,
    start_node: int,
    start_offset: int,
    min_units: int,
    max_units: int | None = None,
    to_end: bool = False,
    include_context: bool | None = None,
) -> PathResult:
    """Cheapest path from ``(start_node, start_offset)`` emitting ``>= min_units`` units, or to an END leaf with
    ``to_end``; a path stops at its first goal.  If no goal is reachable, the visited node with the most units
    emitted (ties: lowest cost) comes back.  Visits nothing: no ``seen`` moves."""
    if min_units < 0:
        raise ValueError(f"min_units must be >= 0, got {min_units}")
    if max_units is not None and max_units < min_units:
        max_units = min_units
    is_end = tree.is_end
    child_costs = tree.child_costs
    label_len = tree.label_len
    overlap = tree.encoding.overlap
    start_units = start_emission(tree, start_node, start_offset)
    prev: dict[int, tuple[int, float]] = {}
    heap = [(0.0, 0, start_node, start_units)]
    tie = 0
    expanded = 0
    goal: int | None = None
    fallback, fb_units, fb_cost = start_node, start_units, 0.0
    while heap:
        cost, _, node, units = heappop(heap)
        expanded += 1
        if is_end(node) or (not to_end and units >= min_units):
            goal = node
            break
        if units > fb_units or (units == fb_units and cost < fb_cost):
            fallback, fb_units, fb_cost = node, units, cost
        if max_units is not None and units >= max_units:
            continue
        for c, ec in child_costs(node):
            nunits = units if is_end(c) else units + label_len(c) - overlap
            prev[c] = (node, ec)
            tie += 1
            heappush(heap, (cost + ec, tie, c, nunits))
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
    return build_result(tree, node_ids, step_costs, start_offset, max_units, expanded, include_context)
