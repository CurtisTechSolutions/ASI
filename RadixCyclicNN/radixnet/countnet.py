"""CountRewardNet - the count / reward model: a second algorithm on the same self-compressing graph.

Every edge keeps several numbers: ``count`` - how many times a training pass
traversed the path through it, all time -, its traversals inside a *sliding
window* of the last ``window`` traversals seen anywhere in the graph, and
``reward`` - the sum of the rewards (+) and penalties (-) it received from
feedback (thumbs up / down, 2NRL, the code-generation judge, the adversarial
review ...).  The graph keeps the global totals (``total_traversals``,
``window_traversals``).  The edge weight is a *dual frequency function*:
each count is compared against the node the edge leaves, as a ratio, once
over the whole history and once inside the window::

    R_all    = (count + 0.5) / (traversals leaving the node + 0.5 * children)
    R_recent = the same ratio inside the sliding window
    weight   = global_scale * log(R_all) + window_scale * log(R_recent)
             + reward_scale * reward   (+ count_scale * log(1 + count), off by default)

with ``global_scale = window_scale = 0.5`` by default (the geometric mean of
the two shares: when history and the window agree, the probability is the
share itself),

and a parent's children are drawn by a softmax over those weights: every
node's activation is the constant 1 (``a = 0, k = 1`` in the sine
parameters), so the graph's edge score ``w * f_parent * f_child`` is the
weight itself and ``P(child | parent) ∝ R_all ** global_scale * R_recent **
window_scale * exp(reward_scale * reward)``.  Every count here is a *cyclic
counter* (:mod:`radixnet.counter`): it wraps back to 0 at ``COUNTER_LIMIT`` and
counts the wrap in ``<name>_resets``, so no total ever outgrows the integer or
the JSON number that carries it.  There is no gradient and no
learning rate: training counts traversals (and slides the window), feedback
moves rewards, ``invert`` flips the sign of every reward, and prediction is a
beam search that returns the K most likely *and* the K least likely
continuations of one prefix in a single call (:class:`~radixnet.beam.Prediction`).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_B, DEFAULT_H
from .backend import get_backend
from .beam import Prediction
from . import diff
from .counter import CyclicCounter
from .encoding import WINDOW, Decoder, Encoder
from .graph import BACK, END, START, RadixCyclicGraph
from .model import (
    MODEL_FORMAT_VERSION,
    GraphModel,
    ProgressFn,
    TrainConfig,
    _resolve_config,
    _utc_now,
    _weight_groups,
    carry_meta,
    meta_add,
    meta_stats,
)

__all__ = ["COUNT_MODEL_FORMAT", "CountRewardGraph", "CountRewardNet"]

COUNT_MODEL_FORMAT = "radixnet-count"
_W = WINDOW
_MAX_LOG_PPL = 700.0


def _walks(transitions: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Cut a pass's transitions back into one walk per text: every text starts at START, nothing else does."""
    walks: list[list[tuple[int, int]]] = []
    for step in transitions:
        if step[0] == START or not walks:
            walks.append([])
        walks[-1].append(step)
    return walks


def _side_totals(rows: Sequence[dict]) -> dict:
    """What a whole in- or out-side of a node did, over the rows in the order they are reported."""
    seen = correct = incorrect = path_seen = 0
    reward = 0.0
    for row in rows:
        seen += row["seen"]
        reward += row["reward"]
        path_seen += row["path_seen"]
        correct += row["correct"]
        incorrect += row["incorrect"]
    judged = correct + incorrect
    return {
        "edges": len(rows), "seen": seen, "reward": reward, "path_seen": path_seen,
        "correct": correct, "incorrect": incorrect,
        "correct_ratio": (correct / judged) if judged else None,
    }


def _by_amount(rewards: dict[int, float]) -> dict[float, list[int]]:
    """Group edges by the reward they are owed (first-seen order, so the graph moves once per amount)."""
    groups: dict[float, list[int]] = {}
    for edge, amount in rewards.items():
        groups.setdefault(amount, []).append(edge)
    return groups


class CountRewardGraph(RadixCyclicGraph):
    """A :class:`RadixCyclicGraph` whose edge weights come from a *dual frequency function* plus rewards.

    Every edge keeps several numbers: ``edge_count`` (all-time traversals),
    ``window_edge_count`` (traversals inside the sliding window of the last
    ``window`` traversals seen anywhere in the graph) and ``edge_reward``;
    the graph keeps the global totals ``total_traversals`` and
    ``window_traversals``.  The two frequencies are compared against the
    *node* the edge leaves: with ``C_p`` / ``W_p`` the all-time / windowed
    traversals leaving parent ``p`` over its ``deg`` children and the
    smoothing ``s = 0.5``::

        R_all    = (count + s) / (C_p + s * deg)          # the edge's share of the node's traversals, all time
        R_recent = (window_count + s) / (W_p + s * deg)   # the same share inside the sliding window
        weight   = count_scale * log(1 + count) + global_scale * log(R_all)
                 + window_scale * log(R_recent) + reward_scale * reward

    so ``P(child | parent) ∝ R_all ** global_scale * R_recent ** window_scale * exp(reward_scale * reward)``
    (times ``(1 + count) ** count_scale``, off by default): what a node did
    over its whole life and what it did recently, each as a ratio, decide
    together.  The default scales ``0.5`` / ``0.5`` make that the geometric
    mean of the two shares - when they agree the probability *is* the share;
    raise one to trust history or recency more.  Activations are the constant 1 on every node, so the base
    class's scores, probabilities, costs, Dijkstra / sampling walks, splits
    and merges all work unchanged; only how a weight comes about differs.
    """

    SMOOTHING = 0.5

    def __init__(
        self,
        seed: int = 0,
        count_scale: float = 0.0,
        reward_scale: float = 1.0,
        global_scale: float = 0.5,
        window_scale: float = 0.5,
        path_scale: float = 1.0,
        window: int = 10_000,
    ) -> None:
        self.count_scale = float(count_scale)
        self.reward_scale = float(reward_scale)
        self.global_scale = float(global_scale)
        self.window_scale = float(window_scale)
        self.path_scale = float(path_scale)
        self.window = int(window)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.edge_reward: list[float] = []
        self.window_edge_count: list[int] = []
        self._window: deque[int] = deque()
        self.total_traversals = CyclicCounter()
        self.edge_parent: list[int] = []  # which node an edge leaves: splits and merges move edges about
        # (the node before the edge's parent, edge) -> [seen, correct, incorrect]: what a *path* did,
        # not what an edge did.  Born when a path is judged, kept up to date by every later traversal.
        self.paths: dict[tuple[int, int], list[int]] = {}
        self._by_edge: dict[int, set[int]] = {}  # edge -> the nodes that called it
        self._by_prev: dict[int, set[int]] = {}  # node -> the edges it called
        self._path_parents: set[int] | None = None  # nodes whose costs depend on where the walk came from
        self._ctx_cache: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
        self._ctx_version = -1
        super().__init__(seed)

    # -- the tracked numbers -------------------------------------------------

    @property
    def window_traversals(self) -> int:
        """Traversals currently inside the sliding window."""
        return len(self._window)

    def record_traversals(self, transitions: Iterable[tuple[int, int]]) -> int:
        """Count traversals of the ``(parent, edge)`` transitions: all time, in the window and globally."""
        window = self._window
        wcount = self.window_edge_count
        limit = self.window
        n = 0
        for _p, e in transitions:
            window.append(e)
            wcount[e] += 1
            n += 1
            if len(window) > limit:
                old = window.popleft()
                if old < len(wcount) and wcount[old] > 0:
                    wcount[old] -= 1
        self.total_traversals += n
        return n

    # -- unique paths: which node called which -------------------------------

    def record_path(
        self, transitions: Sequence[tuple[int, int]], outcome: bool | None = None, create: bool | None = None
    ) -> int:
        """Count one walk of a whole text, step by step, in its own context.

        ``transitions`` are the ``(parent, edge)`` steps of a single text in
        order; the context of step *k* is the parent of step *k-1* (START for
        the first step, which has none).  ``outcome`` says what the walk was
        judged to be - ``True`` correct, ``False`` incorrect, ``None`` not
        judged - and is what separates a *path* counter from an edge counter:
        an edge is right in one sentence and wrong in another, a path is not.
        ``create`` (default: whenever the walk was judged) adds contexts never
        seen before; an unjudged pass only updates the ones a judgement
        created, so bulk training cannot fill the table with the second-order
        counts of a whole corpus.
        """
        if not transitions:
            return 0
        if create is None:
            create = outcome is not None
        if not create and not self.paths:
            return 0
        alive = self.edge_alive
        touched = 0
        prev = START
        for index, (parent, edge) in enumerate(transitions):
            if index:
                prev = transitions[index - 1][0]
            if edge < 0 or edge >= len(alive) or not alive[edge]:
                continue
            row = self._path_row(prev, edge, create, parent)
            if row is None:
                continue
            row[0] += 1
            if outcome is True:
                row[1] += 1
            elif outcome is False:
                row[2] += 1
            touched += 1
        if touched and outcome is not None:
            self.version += 1  # the contexts that moved make their node's costs stale
            self._ctx_cache.clear()
        return touched

    def _path_row(self, prev: int, edge: int, create: bool, parent: int | None = None) -> list[int] | None:
        """The counters of one context, created on demand (and indexed both ways so a split can find them)."""
        key = (prev, edge)
        row = self.paths.get(key)
        if row is not None or not create:
            return row
        row = self.paths[key] = [0, 0, 0]
        self._by_edge.setdefault(edge, set()).add(prev)
        self._by_prev.setdefault(prev, set()).add(edge)
        self._path_parents = None
        return row

    def _drop_path(self, prev: int, edge: int) -> list[int] | None:
        row = self.paths.pop((prev, edge), None)
        if row is None:
            return None
        for index, key, value in ((self._by_edge, edge, prev), (self._by_prev, prev, edge)):
            holder = index.get(key)
            if holder is not None:
                holder.discard(value)
                if not holder:
                    del index[key]
        self._path_parents = None
        return row

    def _add_path(self, prev: int, edge: int, row: Sequence[int], parent: int | None = None) -> None:
        """Add one context's counters into another (a split or a merge moved the step)."""
        into = self._path_row(prev, edge, True, parent)
        into[0] += row[0]
        into[1] += row[1]
        into[2] += row[2]

    def mark_steps(self, steps: Sequence[tuple[int, int]], outcome: bool) -> int:
        """Judge single ``(prev, edge)`` steps - what a diff blames or teaches - rather than a whole walk."""
        marked = 0
        for prev, edge in steps:
            if edge < 0 or edge >= len(self.edge_alive) or not self.edge_alive[edge]:
                continue
            row = self._path_row(int(prev), int(edge), True)
            row[0] += 1
            row[1 if outcome else 2] += 1
            marked += 1
        if marked:
            self.version += 1
            self._ctx_cache.clear()
        return marked

    def path_stats(self, prev: int, edge: int) -> dict | None:
        """``{"prev","edge","seen","correct","incorrect","correct_ratio","seen_ratio","term"}`` of one context."""
        row = self.paths.get((int(prev), int(edge)))
        if row is None:
            return None
        seen, correct, incorrect = row
        judged = correct + incorrect
        traversals = self.edge_count[edge] if 0 <= edge < len(self.edge_count) else 0
        return {
            "prev": int(prev), "edge": int(edge), "seen": seen, "correct": correct, "incorrect": incorrect,
            "correct_ratio": (correct / judged) if judged else None,
            "seen_ratio": (seen / traversals) if traversals else None,
            "term": self.path_term(prev, edge),
        }

    def path_term(self, prev: int, edge: int) -> float:
        """``log((correct + s) / (incorrect + s))`` of a context: zero until a path is judged, and symmetric."""
        row = self.paths.get((int(prev), int(edge)))
        if row is None or (row[1] == 0 and row[2] == 0):
            return 0.0
        s = self.SMOOTHING
        return math.log((row[1] + s) / (row[2] + s))

    def path_totals(self) -> dict:
        """How much of the graph has been judged as paths rather than as edges."""
        seen = correct = incorrect = judged = 0
        for row in self.paths.values():
            seen += row[0]
            correct += row[1]
            incorrect += row[2]
            if row[1] or row[2]:
                judged += 1
        return {"contexts": len(self.paths), "judged": judged, "seen": seen, "correct": correct, "incorrect": incorrect}

    def path_contexts(self, limit: int = 0, node: int | None = None) -> list[dict]:
        """Every context (``node``: only the steps leaving that node), most judged first; ``limit`` 0 = all."""
        out = []
        for prev, edge in self.paths:
            if node is not None and self.edge_parent[edge] != node:
                continue
            stats = self.path_stats(prev, edge)
            if stats is not None:
                out.append(stats)
        out.sort(key=lambda s: (-(s["correct"] + s["incorrect"]), -s["seen"], s["prev"], s["edge"]))
        return out[:limit] if limit > 0 else out

    def nodes_with_paths(self) -> set[int]:
        """The nodes whose out-edges carry a context: everywhere else a step costs what its edge costs."""
        if self._path_parents is None:
            parents = self._path_parents = set()
            for edge in self._by_edge:
                if 0 <= edge < len(self.edge_parent):
                    parents.add(self.edge_parent[edge])
        return self._path_parents

    # -- a node against its neighbours ---------------------------------------

    def edge_paths(self, edge: int) -> tuple[int, int, int]:
        """``(seen, correct, incorrect)`` of one edge, summed over every caller that reached it."""
        seen = correct = incorrect = 0
        for prev in self._by_edge.get(edge, ()):
            row = self.paths.get((prev, edge))
            if row is not None:
                seen += row[0]
                correct += row[1]
                incorrect += row[2]
        return seen, correct, incorrect

    def _side_rows(self, pairs: Sequence[tuple[int, int]]) -> list[dict]:
        """One row per neighbour of a node: its share of the side's traffic and of the side's reward."""
        pairs = sorted(pairs)  # by neighbour id, so both languages add the shares up in the same order
        traversals = {e: self._edge_traversals_f(e) for _n, e in pairs}
        total = 0.0
        for _n, e in pairs:  # an explicit left-to-right sum: the Go port adds them in the same order
            total += traversals[e]
        mass = 0.0
        for _n, e in pairs:
            mass += abs(float(self.edge_reward[e]))
        rows = []
        for n, e in pairs:
            path_seen, correct, incorrect = self.edge_paths(e)
            judged = correct + incorrect
            reward = float(self.edge_reward[e])
            rows.append({
                "node": n,
                "label": self.labels[n] if 0 <= n < len(self.labels) else "",
                "edge": e,
                "seen": self.edge_count[e],
                "seen_resets": self.edge_count_resets.get(e, 0),
                "seen_ratio": (traversals[e] / total) if total else 0.0,
                "reward": reward,
                "reward_ratio": (reward / mass) if mass else 0.0,
                "path_seen": path_seen,
                "path_ratio": (path_seen / traversals[e]) if traversals[e] else None,
                "correct": correct,
                "incorrect": incorrect,
                "correct_ratio": (correct / judged) if judged else None,
            })
        rows.sort(key=lambda r: (-r["seen"], -r["reward"], r["node"]))
        return rows

    def node_ratios(self, node: int) -> dict | None:
        """One node against the nodes around it, or ``None`` when it is not a live node.

        ``from`` is a row per previous node - the edge that arrives here - and
        ``to`` a row per next node, the edge that leaves.  Within a side,
        ``seen_ratio`` is that edge's share of the side's traversals and
        ``reward_ratio`` its share of the side's reward *magnitude*, signed, so
        a penalty reads as a negative share of the pressure on the node and the
        two sides can be compared without the signs cancelling out.
        ``path_ratio`` is how much of the edge's traffic a judged context has
        been watching, and ``correct`` / ``incorrect`` are what those contexts
        made of it, summed over every caller.

        The denominators are the side's own, not the node's visits: a node is
        entered without an in-edge whenever a text starts on it, so ``visits``
        can be larger than everything ``from`` adds up to.
        """
        if not (0 <= node < len(self.alive)) or not self.alive[node]:
            return None
        rows_in = self._side_rows(list(self.parents[node].items()))
        rows_out = self._side_rows(list(self.children[node].items()))
        return {
            "node": node,
            "label": self.labels[node],
            "visits": self.count[node],
            "visit_resets": self.count_resets.get(node, 0),
            "from": rows_in,
            "to": rows_out,
            "in_totals": _side_totals(rows_in),
            "out_totals": _side_totals(rows_out),
        }

    def node_ratio_rows(self, limit: int = 0, node: int | None = None) -> list[dict]:
        """:meth:`node_ratios` of the most visited nodes (``node``: only that one); ``limit`` 0 = all."""
        if node is not None:
            one = self.node_ratios(node)
            return [one] if one is not None else []
        order = [i for i in range(len(self.alive)) if self.alive[i]]
        order.sort(key=lambda i: (-self.node_count(i), i))
        if limit > 0:
            order = order[:limit]
        return [self.node_ratios(i) for i in order]

    def child_costs(self, p: int, prev: int | None = None) -> list[tuple[int, int, float]]:
        """``[(child, edge, -log prob)]`` of ``p``'s out-edges, as seen by a walk that arrived from ``prev``.

        Without a judged context the costs are the edge costs the base class
        caches.  Where a path *has* been judged, its context adds
        ``path_scale * log((correct + s) / (incorrect + s))`` to that edge's
        weight before the softmax - so the same edge is cheap for the walk
        that was right here and dear for the one that was wrong, which is the
        whole point of counting paths instead of edges.
        """
        if prev is None or self.path_scale == 0.0 or p not in self.nodes_with_paths():
            return super().child_costs(p)
        if self._ctx_version != self.version:
            self._ctx_cache.clear()
            self._ctx_version = self.version
        cached = self._ctx_cache.get((prev, p))
        if cached is not None:
            return cached
        items = []
        for c, e in self.children[p].items():
            items.append((c, e, self.edge_w[e] + self.path_scale * self.path_term(prev, e)))
        if items:
            m = max(w for _c, _e, w in items)
            lse = m + math.log(math.fsum(math.exp(w - m) for _c, _e, w in items))
            costs = [(c, e, lse - w) for c, e, w in items]
        else:
            costs = []
        self._ctx_cache[(prev, p)] = costs
        return costs

    # -- keeping the contexts honest through splits and merges ----------------

    def split(self, node_id: int, i: int) -> tuple[int, int]:
        """Split, then follow the contexts: ``q -> P -> c`` becomes ``q -> A -> B -> c``.

        Every out-edge of the old node now leaves ``B``, whose only way in is
        ``A``, so a context ``(q, e)`` of a moved edge becomes ``(A, e)``; the
        new edge ``A -> B`` inherits ``(q, A->B)``, the step ``q`` now calls.
        """
        moving = list(self.children[node_id].values()) if self.alive[node_id] else []
        a_id, b_id = super().split(node_id, i)
        for edge in moving:
            self.edge_parent[edge] = b_id
        if not self.paths or not moving:
            self._path_parents = None
            return a_id, b_id
        bridge = self.children[a_id].get(b_id)
        for edge in moving:
            for prev in list(self._by_edge.get(edge, ())):
                row = self._drop_path(prev, edge)
                if row is None:
                    continue
                self._add_path(a_id, edge, row, b_id)
                if bridge is not None:
                    self._add_path(prev, bridge, row, a_id)
        self._path_parents = None
        self._ctx_cache.clear()
        return a_id, b_id

    def merge_child(self, p: int) -> bool:
        """Merge, then follow the contexts: the chain was unary, so what it knew was never a choice.

        The edge ``p -> c`` dies with its contexts, and so do the contexts of
        ``c``'s out-edges, which said "having come to ``c`` from ``p``" - a
        step nobody could avoid.  Contexts that arrive *through* ``c`` are
        re-keyed to ``p``, which is what the merged node is called from now on.
        """
        child = dying = None
        moved: list[int] = []
        if 0 <= p < len(self.labels) and self.alive[p] and len(self.children[p]) == 1:
            child = next(iter(self.children[p]))
            dying = self.children[p][child]
            moved = list(self.children[child].values())
        if not super().merge_child(p):
            return False
        for edge in moved:
            self.edge_parent[edge] = p
        if self.paths:
            for prev in list(self._by_edge.get(dying, ())):
                self._drop_path(prev, dying)  # the edge is gone
            for edge in moved:
                self._drop_path(p, edge)  # "arrived from p" was the only way to arrive
            for edge in list(self._by_prev.get(child, ())):
                row = self._drop_path(child, edge)
                if row is not None:
                    self._add_path(p, edge, row)
            self._ctx_cache.clear()
        self._path_parents = None
        return True

    def observe_sequence(self, trigrams, count: bool = True) -> list[tuple[int, int]]:
        transitions = super().observe_sequence(trigrams, count)
        if count and transitions:
            self.record_traversals(transitions)
            self.recompute_weights()
        return transitions

    def observe_back(self, p: int, went: int | None = None, instead: int | None = None,
                     amount: float = 1.0) -> int:
        """As :meth:`RadixCyclicGraph.observe_back`, learned the way this model learns everything.

        The sine model nudges the weights directly; here a weight is a *function* of the counts and the
        rewards, so going round is taught by the ``BACK`` edge's count and what to do instead by a penalty on
        the step it looped through and a reward on the step it took after backing up - the same rewards 2NRL
        moves (:meth:`add_reward`).
        """
        e = super().observe_back(p, amount=0.0)  # no weight is nudged by hand here
        self.record_traversals([(p, e)])
        self.add_reward([e], amount)  # the hand-over itself, learned the way this model learns everything
        for child, sign in ((went, -1.0), (instead, 1.0)):
            edge = self.children[p].get(child) if child is not None and child != BACK else None
            if edge is not None:
                self.add_reward([edge], sign * amount)
        self.recompute_weights()
        return e

    def configure(self, **options: float) -> dict:
        """Change scales / the window size (``count_scale``, ``global_scale``, ``window_scale``, ``reward_scale``,
        ``window``) and recompute every weight; returns :meth:`weight_config`."""
        for name, value in options.items():
            if name not in ("count_scale", "global_scale", "window_scale", "reward_scale", "path_scale", "window"):
                raise ValueError(f"unknown weight option {name!r}")
            if value is None:
                continue
            if name == "window":
                size = int(value)
                if size < 1:
                    raise ValueError(f"window must be >= 1, got {value}")
                self.window = size
                while len(self._window) > size:
                    old = self._window.popleft()
                    if old < len(self.window_edge_count) and self.window_edge_count[old] > 0:
                        self.window_edge_count[old] -= 1
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{name} must be a finite number, got {value}")
                setattr(self, name, number)
        self._ctx_cache.clear()
        self._ctx_version = -1
        self.recompute_weights()
        return self.weight_config()

    def weight_config(self) -> dict:
        return {
            "function": "dual-frequency",
            "count_scale": self.count_scale,
            "global_scale": self.global_scale,
            "window_scale": self.window_scale,
            "reward_scale": self.reward_scale,
            "path_scale": self.path_scale,
            "window": self.window,
            "smoothing": self.SMOOTHING,
        }

    # -- the weight function -------------------------------------------------

    def edge_weight(
        self, count: float, reward: float, parent_total: float, degree: int, window_count: float, window_total: float
    ) -> float:
        """The dual frequency function for one edge (see the class docstring)."""
        s = self.SMOOTHING
        count = max(0.0, float(count))
        window_count = max(0.0, float(window_count))
        degree = max(1, int(degree))
        r_all = (count + s) / (max(0.0, float(parent_total)) + s * degree)
        r_recent = (window_count + s) / (max(0.0, float(window_total)) + s * degree)
        return (
            self.count_scale * math.log1p(count)
            + self.global_scale * math.log(r_all)
            + self.window_scale * math.log(r_recent)
            + self.reward_scale * float(reward)
        )

    def shares(self, p: int) -> list[tuple[int, int, float, float]]:
        """``[(child, edge, share_all, share_recent)]`` of ``p``'s edges: each edge's ratio of the node's traversals."""
        edges = list(self.children[p].items())
        recent = sum(self.window_edge_count[e] for _c, e in edges)
        traversals = {e: self._edge_traversals_f(e) for _c, e in edges}
        total = 0.0
        for _c, e in edges:  # an explicit left-to-right sum: the Go port adds them in the same order
            total += traversals[e]
        return [
            (c, e, traversals[e] / total if total else 0.0, self.window_edge_count[e] / recent if recent else 0.0)
            for c, e in edges
        ]

    def recompute_weights(self) -> None:
        """Write the dual frequency weight to every alive edge (after counts, rewards or scales changed)."""
        ew, er, wc = self.edge_w, self.edge_reward, self.window_edge_count
        weight = self.edge_weight
        traversals = self._edge_traversals_f
        for p, ch in enumerate(self.children):
            if not ch or not self.alive[p]:
                continue
            edges = list(ch.values())
            degree = len(edges)
            counts = [traversals(e) for e in edges]
            total = 0.0
            for count in counts:  # an explicit left-to-right sum, as in the Go port
                total += count
            recent = sum(wc[e] for e in edges)
            for e, count in zip(edges, counts):
                ew[e] = weight(count, er[e], total, degree, wc[e], recent)
        self.version += 1

    def add_reward(self, edge_ids: Iterable[int], amount: float) -> int:
        """Add ``amount`` (negative = penalty) to the reward of every listed alive edge; returns how many."""
        er = self.edge_reward
        alive = self.edge_alive
        touched = 0
        for e in edge_ids:
            if 0 <= e < len(er) and alive[e]:
                er[e] += amount
                touched += 1
        if touched:
            self.recompute_weights()
        return touched

    def total_reward(self) -> tuple[float, float]:
        """``(sum of positive rewards, sum of negative rewards)`` over alive edges."""
        pos = neg = 0.0
        for e, ok in enumerate(self.edge_alive):
            if ok:
                r = self.edge_reward[e]
                if r > 0:
                    pos += r
                elif r < 0:
                    neg += r
        return pos, neg

    # -- construction overrides ----------------------------------------------

    def _new_node(self, label, z=None, a=None, b=DEFAULT_B, h=DEFAULT_H, k=0.0, count=0, count_resets=0) -> int:
        # a = 0 and k = 1 make f(z) = 1 whatever z is: scores reduce to the edge weight
        return super()._new_node(
            label, z=0.0 if z is None else z, a=0.0, b=b, h=h, k=1.0, count=count, count_resets=count_resets
        )

    def _new_edge(self, p: int, c: int, count: int = 0, count_resets: int = 0) -> int:
        e = super()._new_edge(p, c, count, count_resets)
        self.edge_reward.append(0.0)
        self.window_edge_count.append(0)
        self.edge_parent.append(p)
        self.edge_w[e] = 0.0  # recompute_weights() gives it its real value once the pass is over
        return e

    def invert(self) -> None:
        """Flip the sign of every reward (what was rewarded is now penalised and vice versa)."""
        er = self.edge_reward
        for e, ok in enumerate(self.edge_alive):
            if ok:
                er[e] = -er[e]
        self.inverted = not self.inverted
        self.recompute_weights()

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        rewards: list[float] = []
        new_index: dict[int, int] = {}
        for old, ok in enumerate(self.alive):
            if ok:
                for _c, e in self.children[old].items():
                    new_index[e] = len(rewards)
                    rewards.append(self.edge_reward[e])
        d["edges"]["reward"] = rewards
        d["weights"] = {
            **self.weight_config(),
            "kind": "count-reward",
            "total_traversals": self.total_traversals.value,
            "total_traversals_resets": self.total_traversals.resets,
            "window_events": [new_index[e] for e in self._window if e in new_index],
        }
        node_index: dict[int, int] = {}
        for old, ok in enumerate(self.alive):
            if ok:
                node_index[old] = len(node_index)
        rows = []
        for (prev, edge), row in self.paths.items():
            if edge in new_index and prev in node_index:
                rows.append((node_index[prev], new_index[edge], row))
        rows.sort()  # the two implementations keep their tables in different orders; the file has one
        d["paths"] = {
            "prev": [r[0] for r in rows],
            "edge": [r[1] for r in rows],
            "seen": [r[2][0] for r in rows],
            "correct": [r[2][1] for r in rows],
            "incorrect": [r[2][2] for r in rows],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CountRewardGraph":
        g = super().from_dict(d)
        weights = d.get("weights") or {}
        legacy = "global_scale" not in weights  # files written before the dual frequency function
        g.count_scale = float(weights.get("count_scale", 1.0 if legacy else 0.0))
        g.reward_scale = float(weights.get("reward_scale", 1.0))
        g.global_scale = float(weights.get("global_scale", 0.0 if legacy else 0.5))
        g.window_scale = float(weights.get("window_scale", 0.0 if legacy else 0.5))
        g.window = max(1, int(weights.get("window", 10_000)))
        g.path_scale = float(weights.get("path_scale", 1.0))
        g.total_traversals = CyclicCounter.from_pair(
            weights.get("total_traversals", 0), weights.get("total_traversals_resets", 0)
        )
        g.edge_parent = [-1] * len(g.edge_w)
        for node, ok in enumerate(g.alive):
            if ok:
                for _child, e in g.children[node].items():
                    g.edge_parent[e] = node
        paths = d.get("paths") or {}
        prevs, edges = paths.get("prev") or [], paths.get("edge") or []
        seen, correct, incorrect = paths.get("seen") or [], paths.get("correct") or [], paths.get("incorrect") or []
        for i, (prev, edge) in enumerate(zip(prevs, edges)):
            if not (0 <= edge < len(g.edge_w)) or not (0 <= prev < len(g.labels)):
                continue
            row = g._path_row(int(prev), int(edge), True)
            row[0] = int(seen[i]) if i < len(seen) else 0
            row[1] = int(correct[i]) if i < len(correct) else 0
            row[2] = int(incorrect[i]) if i < len(incorrect) else 0
        rewards = d.get("edges", {}).get("reward")
        n = len(g.edge_w)
        if rewards is None:
            g.edge_reward = [0.0] * n
        else:
            if len(rewards) != n:
                raise ValueError("edge reward array has an inconsistent length")
            g.edge_reward = [float(v) for v in rewards]
        g.window_edge_count = [0] * n
        g._window = deque()
        for e in weights.get("window_events", []):
            e = int(e)
            if 0 <= e < n:
                g._window.append(e)
                g.window_edge_count[e] += 1
        g.a = [0.0] * len(g.labels)
        g.k = [1.0] * len(g.labels)
        g.recompute_weights()
        return g


class CountRewardNet(GraphModel):
    """The count / reward model (see the module docstring).

    ``train`` counts traversals, ``reward`` / ``punish`` move rewards,
    ``two_nrl`` penalises the bad texts and then counts + rewards the good
    ones (no inversion), ``predict`` returns top-K and bottom-K continuations.
    Learning rates in a :class:`TrainConfig` are accepted and ignored; the
    magnitude of feedback is ``strength`` (default 1: one unit of reward
    multiplies an edge's odds by ``e``).
    """

    kind = "count"
    format = COUNT_MODEL_FORMAT
    label = "Count / reward"
    description = (
        "edge weight = the edge's share of its node's traversals, all time and inside a sliding window, "
        "plus rewards - penalties; no learning rate; beam prediction with the top-K and bottom-K continuations"
    )

    def __init__(
        self,
        seed: int = 0,
        backend: str = "auto",
        device: str | None = None,
        count_scale: float = 0.0,
        reward_scale: float = 1.0,
        global_scale: float = 0.5,
        window_scale: float = 0.5,
        window: int = 10_000,
    ) -> None:
        self.seed = int(seed)
        self.graph = CountRewardGraph(
            seed=self.seed, count_scale=count_scale, reward_scale=reward_scale, global_scale=global_scale,
            window_scale=window_scale, window=window,
        )
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
        # no numeric learning rule runs, so the backend is only reported (python / cpu); backend / device are accepted
        # for interface parity with RadixNet
        self.backend = get_backend("python", None)
        self.history: list[dict] = []
        self.meta: dict = self._new_meta(self.seed)

    @staticmethod
    def _new_meta(seed: int) -> dict:
        meta = GraphModel._new_meta(seed)
        meta.update(rewards_total=0.0, penalties_total=0.0, feedback_passes=0, feedback_passes_resets=0)
        return meta

    # -- passes over data ----------------------------------------------------

    def _mean_cost(self, transitions: list[tuple[int, int]]) -> float:
        """Mean ``-log P`` of the transitions under the current weights."""
        if not transitions:
            return 0.0
        child_costs = self.graph.child_costs
        cache: dict[int, dict[int, float]] = {}
        total = 0.0
        for p, e in transitions:
            costs = cache.get(p)
            if costs is None:
                costs = cache[p] = {edge: cost for _c, edge, cost in child_costs(p)}
            total += costs.get(e, 0.0)
        return total / len(transitions)

    def _passes(
        self,
        texts: Iterable[str] | str,
        cfg: TrainConfig,
        *,
        count: bool,
        reward: float,
        phase: str | None,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """``cfg.epochs`` passes over ``texts``: each traverses (``count``) and / or rewards (``reward``) every path."""
        texts, skipped_short = self._clean_texts(texts)
        graph = self.graph
        meta = self.meta
        records: list[dict] = []
        grams = [self.encoder.encode(t) for t in texts]
        if count:
            meta_add(meta, "trained_texts", len(texts))
            meta_add(meta, "trained_chars", sum(len(t) for t in texts))
        # build the structure first (no counting) and compress it, so every pass - the first included - walks
        # the same transitions: steps inside a compressed node are deterministic and never counted
        self._observe_grams(grams, False)
        pending_merges = graph.compress() if cfg.auto_compress else 0
        outcome = None if not reward else reward > 0  # a rewarded path was judged correct, a penalised one wrong
        for _ in range(cfg.epochs):
            t0 = time.perf_counter()
            transitions = self._observe_grams(grams, count)
            edges = [e for _, e in transitions]
            for walk in _walks(transitions):  # what each text did, in its own context
                graph.record_path(walk, outcome, create=outcome is not None)
            if reward:
                graph.add_reward(edges, reward)
                meta_add(meta, "feedback_passes", 1)
                if reward > 0:
                    meta["rewards_total"] += reward * len(edges)
                else:
                    meta["penalties_total"] += -reward * len(edges)
            if count or reward:
                graph.recompute_weights()
            loss = self._mean_cost(transitions)
            merges = (graph.compress() if cfg.auto_compress else 0) + pending_merges
            pending_merges = 0
            graph.carry_counters()  # the epoch is over: wrap whatever reached the limit
            epoch = meta_add(meta, "epochs_total", 1)
            record = {
                "epoch": epoch,
                "loss": loss,
                "perplexity": math.exp(min(loss, _MAX_LOG_PPL)),
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
                "trigrams": graph.num_trigrams(),
                "compression_ratio": graph.compression_ratio(),
                "merges": merges,
                "transitions": len(transitions),
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "traversed": count,
                "reward": reward,
            }
            if phase is not None:
                record["phase"] = phase
            self.history.append(record)
            records.append(record)
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self, epoch, "epoch", record)
            if stop_event is not None and stop_event.is_set():
                break
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
        """Count one traversal of every text's path per epoch (structure is built on demand, as in RadixNet).

        The loss is the mean ``-log P`` of the transitions after the pass;
        ``lr`` / ``act_lr`` / ``batch_size`` in the config are ignored.
        """
        cfg = _resolve_config(config, overrides)
        return self._passes(
            texts, cfg, count=True, reward=0.0, phase=phase, checkpoint_manager=checkpoint_manager,
            progress=progress, stop_event=stop_event,
        )

    def reward(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs up: ``epochs`` passes that traverse *and* reward (``+strength``) every path of ``texts``.

        ``weights`` (one per text, ``>= 0``) turns the thumbs up into a
        rating: each path is rewarded by ``weight * strength``, so a text
        rated 9 out of 10 adds nine tenths of what a perfect one adds.  Texts
        of equal weight share a pass (heaviest first) and their records carry
        ``"weight"``; a weight of 0 is skipped.
        """
        base = abs(1.0 if strength is None else float(strength))
        if weights is None:
            cfg = _resolve_config(None, {"epochs": epochs, **overrides})
            return self._passes(
                texts, cfg, count=True, reward=base, phase="positive", progress=progress, stop_event=stop_event,
            )
        return self._weighted_passes(
            texts, weights, epochs=epochs, count=True, reward=base, phase="positive",
            progress=progress, stop_event=stop_event, **overrides,
        )

    def punish(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs down: ``epochs`` passes that penalise (``-strength``) every path of ``texts``; no traversal is counted.

        ``weights`` rates the failures the way :meth:`reward` rates the
        successes: each path is penalised by ``weight * strength``.
        """
        base = abs(1.0 if strength is None else float(strength))
        if weights is None:
            cfg = _resolve_config(None, {"epochs": epochs, **overrides})
            return self._passes(
                texts, cfg, count=False, reward=-base, phase="negative", progress=progress, stop_event=stop_event,
            )
        return self._weighted_passes(
            texts, weights, epochs=epochs, count=False, reward=-base, phase="negative",
            progress=progress, stop_event=stop_event, **overrides,
        )

    def _weighted_passes(
        self,
        texts: Iterable[str] | str,
        weights: Sequence[float],
        *,
        epochs: int,
        count: bool,
        reward: float,
        phase: str,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        **overrides,
    ) -> list[dict]:
        """One pass per group of equally weighted texts, the reward / penalty scaled by the weight."""
        cfg = _resolve_config(None, {"epochs": epochs, **overrides})
        records: list[dict] = []
        for weight, group in _weight_groups(texts, weights, "weights"):
            if stop_event is not None and stop_event.is_set():
                break
            group_records = self._passes(
                group, cfg, count=count, reward=reward * weight, phase=phase, stop_event=stop_event,
            )
            for record in group_records:
                record["weight"] = weight
                if progress is not None:
                    progress(record)
            records.extend(group_records)
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
        """2NRL for the count model: penalise ``bad`` (``neg_epochs`` passes), then count + reward ``good``.

        Nothing is inverted: a penalty already makes a path unlikely.
        ``neg_lr`` / ``pos_lr`` are accepted for interface parity and ignored;
        the magnitude per pass is ``strength``, scaled per text by
        ``bad_weights`` when given (the worse a failure, the larger its
        penalty) and by ``good_weights`` (the better a text, the larger its
        reward - a rating, not a thumbs up).
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs")
        base = 1.0 if strength is None else float(strength)
        if bad_weights is None:
            negative = self.punish(bad, epochs=neg_epochs, strength=base, progress=progress, stop_event=stop_event, **overrides)
        else:
            negative = []
            for weight, group in _weight_groups(bad, bad_weights, "bad_weights"):
                if stop_event is not None and stop_event.is_set():
                    break
                records = self.punish(group, epochs=neg_epochs, strength=base * weight, stop_event=stop_event, **overrides)
                for record in records:
                    record["weight"] = weight
                    if progress is not None:
                        progress(record)
                negative.extend(records)
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            positive = self.reward(
                good, epochs=pos_epochs, strength=strength, weights=good_weights, progress=progress,
                stop_event=stop_event, **overrides,
            )
        runs = meta_add(self.meta, "twonrl_runs", 1)
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, runs, "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    def invert(self) -> None:
        """Flip the sign of every reward."""
        self.graph.invert()

    def configure_weights(self, **options: float) -> dict:
        """Change the dual frequency function's scales / window (see :meth:`CountRewardGraph.configure`)."""
        return self.graph.configure(**options)

    def weight_config(self) -> dict:
        return self.graph.weight_config()

    def invert_paths(
        self, texts: Iterable[str] | str, mode: str = "activation", amounts=None, strength: float = 2.0, **options
    ) -> dict:
        """Failures: there is no activation to flip here, so every edge of a text's path loses ``strength * 2 * amount``
        reward (amount 1, the default, is a full ``2 * strength`` penalty; the worse the text, the larger).

        Returns ``{"texts", "flipped", "unit": "edges", "mode": "penalty", "amount_mean"}``.
        """
        texts, _ = self._clean_texts(texts)
        values = self._amounts(texts, amounts)
        graph = self.graph
        penalties: dict[int, float] = {}
        for path, amount in zip(self._paths_of(texts), values):
            penalty = abs(float(strength)) * 2.0 * amount
            if penalty <= 0:
                continue
            for p, c in zip(path, path[1:]):
                e = graph.children[p].get(c)
                if e is not None:
                    penalties[e] = max(penalties.get(e, 0.0), penalty)
        touched = 0
        for e, penalty in penalties.items():
            touched += graph.add_reward([e], -penalty)
            self.meta["penalties_total"] += penalty
        if touched:
            meta_add(self.meta, "feedback_passes", 1)
        applied = [v for v in values if v > 0]
        return {
            "texts": len(texts), "flipped": touched, "unit": "edges", "mode": "penalty",
            "amount_mean": sum(applied) / len(applied) if applied else 0.0,
        }

    # -- learning from a correction ------------------------------------------

    def correct(
        self,
        wrong: str,
        right: str,
        *,
        strength: float | None = 1.0,
        weight: float = 1.0,
        reward: float = 1.0,
        keep: float = 0.0,
        count: bool = True,
    ) -> dict:
        """Teach one correction: move only the trigram nodes the two sentences disagree on.

        ``wrong`` is what the network wrote, ``right`` what the teacher wrote
        instead.  The two are aligned character by character
        (:mod:`radixnet.diff`) and every step of either path is charged with
        the characters it adds, so:

        * the steps of ``wrong`` that added a character the teacher struck out
          or replaced lose ``strength * weight`` of reward - and *only* those:
          the words both sentences agree on keep what they earned;
        * the steps of ``right`` that wrote what the teacher put there instead
          gain ``strength * reward``; the rest of the correction earns ``keep``
          times as much, and ``keep`` is 0 by default - a whole path is
          rewarded when the *output* was correct (:meth:`reward`), not when it
          had to be corrected;
        * ``count`` traverses the correction once, as a training pass does,
          because a corrected sentence is correct English whatever changed.

        An edge both sentences walk over a changed span - the network wrote the
        right characters by another route - is rewarded, never penalised.
        Returns what moved: ``{"edits", "changes", "penalised", "rewarded",
        "kept", "penalty", "reward", "loss", "wrong_chars", "right_chars"}``.
        """
        base = abs(1.0 if strength is None else float(strength))
        wrong, right = str(wrong or ""), str(right or "")
        changes = diff.summary(wrong, right, limit=0)
        wrong_spans, right_spans = diff.changed_spans(wrong, right)
        result = {
            "edits": len(changes), "changes": changes[:8],
            "penalised": 0, "rewarded": 0, "kept": 0, "penalty": 0.0, "reward": 0.0, "loss": None,
            "wrong_chars": sum(hi - lo for lo, hi in wrong_spans),
            "right_chars": sum(hi - lo for lo, hi in right_spans),
            "marked_correct": 0, "marked_incorrect": 0,
        }
        graph = self.graph
        wrong_grams = self.encoder.encode(wrong) if len(wrong) >= _W else []
        right_grams = self.encoder.encode(right) if len(right) >= _W else []
        if not wrong_grams and not right_grams:
            return result
        # both sentences join the structure before either is measured: observing one can split a node the
        # other's path runs through, and the split moves the very edge a penalty was meant for
        self._observe_grams([g for g in (wrong_grams, right_grams) if g], False)
        penalties: dict[int, float] = {}
        rewards: dict[int, float] = {}
        fixed: set[int] = set()
        blamed_steps: list[tuple[int, int]] = []
        taught_steps: list[tuple[int, int]] = []
        if wrong_grams and wrong_spans and base * weight > 0:
            blamed_steps = self._steps_over(wrong_grams, len(wrong), wrong_spans)
            for _prev, edge in blamed_steps:
                penalties[edge] = -base * float(weight)
        if right_grams:
            transitions = graph.observe_sequence(right_grams, count)
            if count:
                self.meta["trained_texts"] += 1
                self.meta["trained_chars"] += len(right)
                graph.record_path(transitions, None, create=False)  # the correction's own traffic
            if right_spans:
                taught_steps = self._steps_over(right_grams, len(right), right_spans)
                fixed = {edge for _prev, edge in taught_steps}
            for _p, e in transitions:
                amount = base * float(reward) * (1.0 if e in fixed else float(keep))
                if amount > 0:
                    rewards[e] = amount
            result["loss"] = self._mean_cost(transitions)
        # one call per distinct amount: add_reward recomputes the weights, and a correction moves
        # at most three of them (the penalty, the fix, and what the rest of the correction keeps)
        blamed = [e for e in penalties if e not in rewards]  # the teacher wrote it too: it is not the mistake
        if blamed:
            penalty = -base * float(weight)
            result["penalised"] = graph.add_reward(blamed, penalty)
            result["penalty"] = -penalty * result["penalised"]
        for amount, edges in _by_amount(rewards).items():
            touched = graph.add_reward(edges, amount)
            result["rewarded" if all(e in fixed for e in edges) else "kept"] += touched
            result["reward"] += amount * touched
        # the counters follow the reward: what was blamed is a wrong path here, what was taught a right one
        result["marked_incorrect"] = graph.mark_steps([s for s in blamed_steps if s[1] not in rewards], False)
        result["marked_correct"] = graph.mark_steps(taught_steps, True)
        if result["penalised"] or result["rewarded"] or result["kept"]:
            self.meta["feedback_passes"] += 1
            self.meta["rewards_total"] += result["reward"]
            self.meta["penalties_total"] += result["penalty"]
            graph.recompute_weights()
        return result

    def paths(self, limit: int = 20, node: int | None = None) -> list[dict]:
        """The judged paths: ``[{"prev","edge","seen","correct","incorrect","correct_ratio","seen_ratio","term"}]``."""
        return self.graph.path_contexts(limit=limit, node=node)

    def node_ratios(self, limit: int = 20, node: int | None = None) -> list[dict]:
        """Each node against its neighbours: ``[{"node","label","visits","from","to","in_totals","out_totals"}]``."""
        return self.graph.node_ratio_rows(limit=limit, node=node)

    # -- prediction ----------------------------------------------------------

    def predict(
        self,
        prefix: str,
        length: int = 20,
        mode: str = "beam",
        k: int = 5,
        beam: int | None = None,
        step_penalty: float = 0.0,
        temperature: float = 1.0,
        to_end: bool = False,
        max_length: int | None = None,
    ) -> Prediction:
        """Continue ``prefix``: the ``k`` most likely and the ``k`` least likely continuations in one search.

        ``"beam"`` (``"dijkstra"`` is accepted as an alias) runs the beam
        search of :mod:`radixnet.beam`; the result *is* the best path (a
        :class:`~radixnet.search.PathResult`) and carries ``top`` / ``bottom``.
        ``"sample"`` draws one stochastic walk (``top = [it]``).  ``length``,
        ``to_end``, ``max_length`` and ``step_penalty`` mean what they mean for
        :meth:`RadixNet.predict`.
        """
        self._check_predict_args(prefix, length, max_length, k, beam)
        mode = (mode or "beam").lower()
        if mode == "dijkstra":
            mode = "beam"
        if mode not in ("beam", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'beam', 'dijkstra' or 'sample'")
        return self._search(prefix, length, mode, k, beam, step_penalty, temperature, to_end, max_length)

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict:
        g = self.graph
        meta = self.meta
        pos, neg = g.total_reward()
        paths = g.path_totals()
        return {
            "kind": self.kind,
            "nodes": g.num_nodes(),
            "edges": g.num_edges(),
            "trigrams": g.num_trigrams(),
            "compression_ratio": g.compression_ratio(),
            "inverted": g.inverted,
            "backend": self.backend.name,
            "device": self.backend.device,
            **meta_stats(meta, "epochs_total", "trained_chars", "trained_texts", "twonrl_runs", "feedback_passes"),
            "history_len": len(self.history),
            "last_loss": self.history[-1]["loss"] if self.history else None,
            "rewards_total": meta["rewards_total"],
            "penalties_total": meta["penalties_total"],
            "path_contexts": paths["contexts"],
            "path_judged": paths["judged"],
            "path_seen": paths["seen"],
            "path_correct": paths["correct"],
            "path_incorrect": paths["incorrect"],
            "edge_reward_positive": pos,
            "edge_reward_negative": neg,
            "count_scale": g.count_scale,
            "reward_scale": g.reward_scale,
            "global_scale": g.global_scale,
            "window_scale": g.window_scale,
            "path_scale": g.path_scale,
            "window": g.window,
            "total_traversals": g.total_traversals.value,
            "total_traversals_resets": g.total_traversals.resets,
            "window_traversals": g.window_traversals,
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": COUNT_MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "kind": self.kind,
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None) -> "CountRewardNet":
        if not isinstance(d, dict) or d.get("format") != COUNT_MODEL_FORMAT:
            raise ValueError(f"not a {COUNT_MODEL_FORMAT} model document")
        version = int(d.get("version", 1))
        if version > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {COUNT_MODEL_FORMAT} model version {version}")
        graph = CountRewardGraph.from_dict(d["graph"])
        model = cls(seed=graph.seed, backend=backend, device=device)
        model.graph = graph
        model.history = [dict(r) for r in d.get("history", [])]
        meta = cls._new_meta(graph.seed)
        meta.update(d.get("meta") or {})
        model.meta = carry_meta(meta)
        return model

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"CountRewardNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, inverted={g.inverted})"
        )
