"""RadixTree - the acyclic sibling of ``RadixCyclicNN``'s graph.

The one structural difference from ``radixnet.graph.RadixCyclicGraph`` is what
a node *is*.  There, a node is a gram: a trigram lives in exactly one node, so
the second occurrence of ``"aaa"`` is an edge back to the first and the graph
has cycles.  Here, a node is a **context** - the sequence of grams on its path
from the root - so the second ``"aaa"`` is a different node from the first,
every edge runs from a node to a strictly deeper one, and the structure is a
rooted tree.  A tree cannot contain a cycle; :meth:`RadixTree.check_invariants`
proves it after every operation the tests make.

Everything else is the cyclic graph's, node for node:

* a node holds a label of ``n`` or more characters and therefore one or more
  grams; a run of nodes with one child each is stored as **one node** (path
  compression, the radix rule) and a transition observed into or out of the
  middle of such a run **splits** it again (:meth:`split`, :meth:`merge_child`);
* every node owns a sine activation ``f(z) = a * sin(b * (z - h)) + k`` and
  its in-edge a weight ``w``; an edge ``p -> c`` scores ``w_c * f_p * f_c``
  and the children of a node compete in a softmax over those scores;
* new nodes and weights are drawn from the same ranges, with the same seeded
  generator, and are negated while the tree is inverted.

What the tree holds is **every suffix** of every text (bounded by ``depth``
symbols when a bound is given), the way a suffix tree does: a text of ``L``
grams is inserted once from START - ``START -> g0 -> g1 -> ... -> END`` - and
once from the root for every start position, so a walk from the root along the
last few grams of any prefix lands on the deepest context the corpus has seen.
Without the bound every root path ends in an END leaf; with it a path may end
where its window ran out.

Sentinels: node ``0`` is the ROOT (the empty context, no label), node ``1`` is
START (a child of the root: the context *at the beginning of a text*), and
every context a text ended in has its own END leaf - a leaf, not a shared node,
because a shared END would have many parents and the structure would stop being
a tree.  Sentinels are told apart by :attr:`kind`, never by label.

Storage is flat parallel lists indexed by node id; ``children[p]`` maps a real
child's first gram to its id, ``end_leaf[p]`` is the END leaf under ``p`` (or
``-1``).  Nodes removed by a merge are tombstoned (ids are never reused) and
compacted only by :meth:`to_dict`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K
from .encoding import END_LABEL, START_LABEL, Encoding, _piece

__all__ = [
    "FIRST", "KIND_END", "KIND_REAL", "KIND_ROOT", "KIND_START", "ROOT", "START",
    "RadixTree", "W_HIGH", "W_LOW", "Z_RANGE",
]

ROOT, START = 0, 1
"""The two fixed nodes: the empty context, and the context at the beginning of a text."""

FIRST = START + 1
"""The first node id that is a real node or an END leaf."""

KIND_ROOT, KIND_START, KIND_END, KIND_REAL = 0, 1, 2, 3
"""What a node is: the root, START, an END leaf, or a real node holding grams."""

Z_RANGE = 4.5
"""New node state ``z`` is drawn uniformly from ``[-Z_RANGE, Z_RANGE]`` (``radixnet.graph.Z_RANGE``)."""

W_LOW, W_HIGH = 0.5, 1.5
"""New in-edge weights are drawn uniformly from ``[W_LOW, W_HIGH]`` (negated while inverted)."""

_TREE_FORMAT = "radixtree-tree"
_TREE_FORMAT_VERSION = 1


class RadixTree:
    """Nodes, the suffix insertion, the radix split / merge, and the scores over a node's children."""

    def __init__(self, seed: int = 0, encoding: Encoding | None = None, depth: int | None = None) -> None:
        self.seed = int(seed)
        self.encoding = encoding if encoding is not None else Encoding()
        self.encoding.validate()
        if depth is not None and int(depth) < 2:
            raise ValueError(f"depth must be >= 2 symbols (a context and what follows it) or None, got {depth}")
        self.depth: int | None = None if depth is None else int(depth)
        """How many symbols (START counts as one, END does not) a root path may hold; ``None`` is unbounded."""
        self.rng = random.Random(self.seed)
        self.labels: list[str] = []
        self.kind: list[int] = []
        self.parent: list[int] = []
        self.children: list[dict[str, int]] = []
        self.end_leaf: list[int] = []
        self.count: list[int] = []
        self.w: list[float] = []
        self.z: list[float] = []
        self.a: list[float] = []
        self.b: list[float] = []
        self.h: list[float] = []
        self.k: list[float] = []
        self.alive: list[bool] = []
        self.grams: set[str] = set()
        """Every distinct gram ever observed - what the compression ratio counts."""
        self.inverted = False
        self.version = 0
        """Bumped on any structural or parameter change (the cost caches key on it)."""
        self.structure_version = 0
        """Bumped only on node creation, split and merge."""
        self._n_real = 0
        self._n_ends = 0
        self._cost_cache: dict[int, list[tuple[int, float]]] = {}
        self._cost_version = -1
        self._act_cache: list[float] = []
        self._act_version = -1
        self._n = self.encoding.n
        self._ov = self.encoding.overlap
        self._view = self.encoding.view
        self._last = self.encoding.last_unit
        self._new_node("", KIND_ROOT, -1, w=0.0)
        self._new_node(START_LABEL, KIND_START, ROOT)

    # -- construction --------------------------------------------------------

    def _new_node(
        self,
        label: str,
        kind: int,
        parent: int,
        z: float | None = None,
        a: float | None = None,
        b: float = DEFAULT_B,
        h: float = DEFAULT_H,
        k: float = DEFAULT_K,
        count: int = 0,
        w: float | None = None,
    ) -> int:
        """Allocate a node under ``parent`` (not linked into its children); ``z`` and ``w`` random unless given."""
        if z is None:
            z = self.rng.uniform(-Z_RANGE, Z_RANGE)
        if a is None:
            a = -DEFAULT_A if self.inverted else DEFAULT_A
        if w is None:
            w = self.rng.uniform(W_LOW, W_HIGH)
            if self.inverted:
                w = -w
        nid = len(self.labels)
        self.labels.append(label)
        self.kind.append(kind)
        self.parent.append(parent)
        self.children.append({})
        self.end_leaf.append(-1)
        self.count.append(count)
        self.w.append(w)
        self.z.append(z)
        self.a.append(a)
        self.b.append(b)
        self.h.append(h)
        self.k.append(k)
        self.alive.append(True)
        if kind == KIND_REAL:
            self._n_real += 1
        elif kind == KIND_END:
            self._n_ends += 1
        self.version += 1
        self.structure_version += 1
        return nid

    def _new_real(self, label: str, parent: int) -> int:
        """A real node holding ``label``, linked under ``parent`` by its first gram."""
        nid = self._new_node(label, KIND_REAL, parent)
        self.children[parent][self.encoding.first_gram(label)] = nid
        return nid

    def _new_end(self, parent: int) -> int:
        """The END leaf of ``parent`` (which must not have one yet)."""
        nid = self._new_node(END_LABEL, KIND_END, parent)
        self.end_leaf[parent] = nid
        return nid

    # -- sizes and kinds -----------------------------------------------------

    def is_real(self, i: int) -> bool:
        """A node holding grams (not the root, START or an END leaf)."""
        return self.kind[i] == KIND_REAL

    def is_end(self, i: int) -> bool:
        """An END leaf."""
        return self.kind[i] == KIND_END

    def held(self, i: int) -> int:
        """How many grams node ``i`` holds (0 for the root, START and END leaves)."""
        return len(self._view(self.labels[i])) - self._ov if self.kind[i] == KIND_REAL else 0

    def label_len(self, i: int) -> int:
        """The length of a real node's label in the encoding's units (0 for sentinels)."""
        return len(self._view(self.labels[i])) if self.kind[i] == KIND_REAL else 0

    def first_gram(self, i: int) -> str:
        """The gram a real node is keyed by under its parent."""
        return self.encoding.first_gram(self.labels[i])

    def num_nodes(self) -> int:
        """Alive real nodes - the ones holding grams; the sentinels are not counted."""
        return self._n_real

    def num_ends(self) -> int:
        """Alive END leaves: one per context a text has ended in."""
        return self._n_ends

    def num_edges(self) -> int:
        """Alive learnable edges: every real node and every END leaf has exactly one in-edge (a tree)."""
        return self._n_real + self._n_ends

    def num_grams(self) -> int:
        """Distinct grams observed."""
        return len(self.grams)

    def compression_ratio(self) -> float:
        """Distinct grams per real node - the cyclic graph's metric, on a structure where a gram may live in many nodes."""
        return len(self.grams) / max(1, self._n_real)

    def label_chars(self) -> int:
        """Units held in real labels (characters, or sounds): the tree's real size."""
        return sum(self.label_len(i) for i in range(len(self.labels)) if self.alive[i] and self.kind[i] == KIND_REAL)

    def alive_nodes(self) -> list[int]:
        """Ids of alive nodes in increasing order (the root and START first)."""
        return [i for i, ok in enumerate(self.alive) if ok]

    def real_nodes(self) -> list[int]:
        """Ids of alive real nodes in increasing order."""
        return [i for i, ok in enumerate(self.alive) if ok and self.kind[i] == KIND_REAL]

    def depth_of(self, i: int) -> int:
        """Edges from the root to node ``i``."""
        d = 0
        while i != ROOT:
            i = self.parent[i]
            d += 1
        return d

    def max_depth(self) -> int:
        """The deepest node, in edges from the root."""
        best = 0
        for i in self.alive_nodes():
            d = self.depth_of(i)
            if d > best:
                best = d
        return best

    def child_items(self, p: int) -> list[int]:
        """The children a walk may take from ``p``: its real children, then its END leaf.

        START is the root's child too, but never an option: a walk begins there,
        it does not go there.
        """
        kids = list(self.children[p].values())
        e = self.end_leaf[p]
        if e >= 0:
            kids.append(e)
        return kids

    def num_children(self, p: int) -> int:
        """How many options a walk has at ``p``."""
        return len(self.children[p]) + (1 if self.end_leaf[p] >= 0 else 0)

    def branches(self) -> int:
        """Nodes with more than one option - the only places a decision is made."""
        return sum(1 for i in self.alive_nodes() if self.num_children(i) > 1)

    # -- activations, scores, probabilities, costs ---------------------------

    def activation_of(self, i: int) -> float:
        """``f_i(z_i)`` of one node."""
        return self.a[i] * math.sin(self.b[i] * (self.z[i] - self.h[i])) + self.k[i]

    def _activations(self) -> list[float]:
        """Activation of every node id, cached per ``version``."""
        if self._act_version != self.version:
            sin = math.sin
            self._act_cache = [
                ai * sin(bi * (zi - hi)) + ki for zi, ai, bi, hi, ki in zip(self.z, self.a, self.b, self.h, self.k)
            ]
            self._act_version = self.version
        return self._act_cache

    def child_scores(self, p: int) -> list[tuple[int, float]]:
        """``[(child, w_c * f_p * f_c)]`` over the options at ``p``."""
        acts = self._activations()
        fp = acts[p]
        w = self.w
        return [(c, w[c] * fp * acts[c]) for c in self.child_items(p)]

    def child_probs(self, p: int) -> list[tuple[int, float]]:
        """Numerically stable softmax over :meth:`child_scores`."""
        scores = self.child_scores(p)
        if not scores:
            return []
        m = max(s for _, s in scores)
        exps = [(c, math.exp(s - m)) for c, s in scores]
        total = sum(v for _, v in exps)
        return [(c, v / total) for c, v in exps]

    def child_costs(self, p: int, costs: str = "logprob") -> list[tuple[int, float]]:
        """``[(child, cost)]`` over the options at ``p``.

        ``"logprob"`` is ``-log softmax`` - non-negative, the cyclic graph's
        cost, cached by ``version``.  ``"signal"`` is the negated edge signal
        ``-(w_c * f_p * f_c)`` itself, which may be negative: a cost only a
        structure without cycles can walk by, since a negative cycle would make
        the cheapest path ``-inf`` and there is none here.
        """
        if costs == "signal":
            return [(c, -s) for c, s in self.child_scores(p)]
        if costs != "logprob":
            raise ValueError(f"costs must be 'logprob' or 'signal', got {costs!r}")
        if self._cost_version != self.version:
            self._cost_cache = {}
            self._cost_version = self.version
        hit = self._cost_cache.get(p)
        if hit is None:
            scores = self.child_scores(p)
            if scores:
                m = max(s for _, s in scores)
                lse = m + math.log(sum(math.exp(s - m) for _, s in scores))
                hit = [(c, lse - s) for c, s in scores]
            else:
                hit = []
            self._cost_cache[p] = hit
        return hit

    def log_prob(self, p: int, c: int) -> float | None:
        """``log P(c | p)`` for an option ``c`` at ``p``, ``None`` when ``c`` is not one."""
        for child, cost in self.child_costs(p):
            if child == c:
                return -cost
        return None

    # -- structural operations -----------------------------------------------

    def split(self, node: int, i: int) -> tuple[int, int]:
        """Split a real node between its grams ``i - 1`` and ``i``.

        ``A`` keeps ``node``'s id, its in-edge and ``label[:i + n - 1]``; ``B``
        is a new node with ``label[i:]`` that inherits ``A``'s children, END
        leaf, state, activation parameters and count, and gets a fresh in-edge
        weight - exactly the cyclic graph's split.  Returns ``(A, B)``.
        """
        if node < 0 or node >= len(self.labels) or not self.alive[node]:
            raise ValueError(f"node {node} is not alive")
        if self.kind[node] != KIND_REAL:
            raise ValueError("cannot split a sentinel node")
        held = self.held(node)
        if i < 1 or i >= held:
            raise ValueError(f"split index {i} out of range 1..{held - 1} for label {self.labels[node]!r}")
        view = self._view(self.labels[node])
        n, ov = self._n, self._ov
        b = self._new_node(
            _piece(view, i), KIND_REAL, node,
            z=self.z[node], a=self.a[node], b=self.b[node], h=self.h[node], k=self.k[node], count=self.count[node],
        )
        self.children[b] = self.children[node]
        for c in self.children[b].values():
            self.parent[c] = b
        e = self.end_leaf[node]
        self.end_leaf[b] = e
        if e >= 0:
            self.parent[e] = b
        self.children[node] = {_piece(view, i, i + n): b}
        self.end_leaf[node] = -1
        self.labels[node] = _piece(view, 0, i + ov)
        self.version += 1
        self.structure_version += 1
        return node, b

    def merge_child(self, p: int) -> bool:
        """Merge ``p``'s single real child ``c`` into ``p`` when the chain is unary.

        Conditions: ``p`` is a real node with exactly one option, and that
        option is a real node (an END leaf is never merged away).  ``p`` takes
        the label ``labels[p] + labels[c][n - 1:]``, ``c``'s children and END
        leaf, and the larger of the two counts; ``c`` is tombstoned.

        The merge preserves the network function, as the cyclic graph's does:
        the merged node keeps the activation of whichever endpoint has the
        larger ``|f|`` and the edges on the other side are rescaled by the
        ratio of the two activations, so every remaining edge score
        ``w * f_parent * f_child`` - and every probability and cost - is
        unchanged.  The edge ``p -> c`` itself was the only option at ``p``,
        probability 1 and cost 0, so nothing is lost by removing it.
        """
        if p < FIRST or p >= len(self.labels) or not self.alive[p] or self.kind[p] != KIND_REAL:
            return False
        ch = self.children[p]
        if len(ch) != 1 or self.end_leaf[p] >= 0:
            return False
        c = next(iter(ch.values()))
        fp = self.activation_of(p)
        fc = self.activation_of(c)
        w = self.w
        if abs(fp) >= abs(fc):
            ratio = fc / fp if fp != 0.0 else 1.0  # fp == 0 implies fc == 0: every score stays 0
            for g in self.child_items(c):
                w[g] *= ratio
        else:
            w[p] *= fp / fc
            self.z[p], self.a[p], self.b[p], self.h[p], self.k[p] = self.z[c], self.a[c], self.b[c], self.h[c], self.k[c]
        self.labels[p] = self.encoding.join_units(self.labels[p], _piece(self._view(self.labels[c]), self._ov))
        self.children[p] = self.children[c]
        for g in self.children[p].values():
            self.parent[g] = p
        e = self.end_leaf[c]
        self.end_leaf[p] = e
        if e >= 0:
            self.parent[e] = p
        if self.count[c] > self.count[p]:
            self.count[p] = self.count[c]
        self.labels[c] = ""
        self.children[c] = {}
        self.end_leaf[c] = -1
        self.parent[c] = -1
        self.alive[c] = False
        self._n_real -= 1
        self.version += 1
        self.structure_version += 1
        return True

    def compress(self) -> int:
        """Merge every unary chain until none remains; returns the merge count.

        Observation never leaves one: a real node is created holding a whole
        remainder, or cut at a branch, so a tree built by
        :meth:`observe_sequence` alone is compressed already, at any depth, and
        this returns 0.  It is :meth:`split` - the operation a dynamic window
        would be built on - that leaves chains for it to merge.
        """
        merges = 0
        alive = self.alive
        kind = self.kind
        merge = self.merge_child
        while True:
            done = 0
            for p in range(FIRST, len(self.labels)):
                if alive[p] and kind[p] == KIND_REAL:
                    while merge(p):
                        done += 1
            if done == 0:
                return merges
            merges += done

    # -- observation ---------------------------------------------------------

    def _windows(self, grams: Sequence[str]) -> list[tuple[int, list[str], bool]]:
        """``(origin, symbols, ended)`` for every window of a text: from START, then from the root at every position."""
        depth = self.depth
        total = len(grams)
        out: list[tuple[int, list[str], bool]] = []
        if depth is None:
            out.append((START, list(grams), True))
            for s in range(total):
                out.append((ROOT, list(grams[s:]), True))
        else:
            out.append((START, list(grams[: depth - 1]), total <= depth - 1))
            for s in range(total):
                out.append((ROOT, list(grams[s : s + depth]), s + depth >= total))
        return out

    def observe_sequence(self, grams: Sequence[str], count: bool = True) -> list[tuple[int, int]]:
        """Register a text's grams: every window of it, from START and from the root.

        Returns the ``(parent, child)`` transitions the windows took, in order -
        exactly what the learning rule trains on.  A step inside a compressed
        node is deterministic and is not a transition.  ``count`` also bumps
        the visit counters.  Consecutive grams must overlap as
        :meth:`Encoding.encode` produces them.
        """
        if not grams:
            return []
        self.encoding.check_grams(grams)
        self.grams.update(grams)
        windows = self._windows(grams)
        before = self.structure_version
        transitions: list[tuple[int, int]] = []
        for origin, symbols, ended in windows:
            transitions.extend(self._insert(origin, symbols, ended, count))
        if self.structure_version != before:
            # a later window may have split a node an earlier transition named: read them again against the
            # structure as it now stands (every window is a root path now, so nothing is created or counted)
            transitions = []
            for origin, symbols, ended in windows:
                transitions.extend(self._insert(origin, symbols, ended, False))
        return transitions

    def _insert(self, node: int, symbols: list[str], ended: bool, count: bool) -> list[tuple[int, int]]:
        """Radix insertion of one window under ``node``; returns its transitions."""
        counts = self.count
        labels = self.labels
        children = self.children
        view = self._view
        ov = self._ov
        tails = [self._last(g) for g in symbols]
        trans: list[tuple[int, int]] = []
        if count:
            counts[node] += 1
        pos = 0
        total = len(symbols)
        while True:
            if pos == total:
                if ended:
                    leaf = self.end_leaf[node]
                    if leaf < 0:
                        leaf = self._new_end(node)
                    if count:
                        counts[leaf] += 1
                    trans.append((node, leaf))
                return trans
            g = symbols[pos]
            nxt = children[node].get(g)
            if nxt is None:
                leaf = self._new_real(self.encoding.decode_grams(symbols[pos:]), node)
                if count:
                    counts[leaf] += 1
                trans.append((node, leaf))
                if ended:
                    e = self._new_end(leaf)
                    if count:
                        counts[e] += 1
                    trans.append((leaf, e))
                return trans
            lv = view(labels[nxt])
            held = len(lv) - ov
            remaining = total - pos
            limit = held if held < remaining else remaining
            i = 1
            while i < limit and lv[i + ov] == tails[pos + i]:
                i += 1
            if i < held:
                if i == remaining:
                    # the window ends inside this run
                    if not ended:
                        if count:
                            counts[nxt] += 1
                        trans.append((node, nxt))
                        return trans
                    self.split(nxt, i)  # the text ends here: END must hang off this exact context
                else:
                    self.split(nxt, i)  # diverged inside the run
                held = i
            if count:
                counts[nxt] += 1
            trans.append((node, nxt))
            node = nxt
            pos += held

    # -- walking -------------------------------------------------------------

    def walk(self, origin: int, grams: Sequence[str]) -> tuple[int, int] | None:
        """Where ``grams`` lead from ``origin``: ``(node, offset)``, or ``None``.

        ``offset`` is the index of the last matched gram inside the node's
        label; ``offset == held(node) - 1`` means the walk landed *on* the node,
        a smaller offset that it stopped inside a run, where the continuation
        is fixed.  An empty walk lands on the origin (offset ``-1``).
        """
        node = origin
        offset = -1
        pos = 0
        total = len(grams)
        labels = self.labels
        children = self.children
        view = self._view
        ov = self._ov
        tails = [self._last(g) for g in grams]
        while pos < total:
            nxt = children[node].get(grams[pos])
            if nxt is None:
                return None
            lv = view(labels[nxt])
            held = len(lv) - ov
            remaining = total - pos
            limit = held if held < remaining else remaining
            i = 1
            while i < limit and lv[i + ov] == tails[pos + i]:
                i += 1
            if i < limit:
                return None
            if remaining <= held:
                return (nxt, remaining - 1)
            node = nxt
            offset = held - 1
            pos += held
        return (node, offset)

    def at_end(self, node: int, offset: int) -> bool:
        """Whether ``(node, offset)`` stands on the node's last gram - where its options begin."""
        return offset == self.held(node) - 1

    def trace(self, grams: Sequence[str], origin: int = START) -> tuple[list[tuple[int, int]], list[int]] | None:
        """``(transitions, node_path)`` of a whole text through the tree, or ``None``.

        The path runs ``[origin, ..., END leaf]``.  ``None`` means the text is
        not a root path of the tree as it stands: a gram is missing, or it
        ends inside a run.  Nothing is created or counted.
        """
        node = origin
        pos = 0
        total = len(grams)
        labels = self.labels
        children = self.children
        view = self._view
        ov = self._ov
        tails = [self._last(g) for g in grams]
        transitions: list[tuple[int, int]] = []
        path = [origin]
        while pos < total:
            nxt = children[node].get(grams[pos])
            if nxt is None:
                return None
            lv = view(labels[nxt])
            held = len(lv) - ov
            if held > total - pos:
                return None
            i = 1
            while i < held and lv[i + ov] == tails[pos + i]:
                i += 1
            if i < held:
                return None
            transitions.append((node, nxt))
            path.append(nxt)
            node = nxt
            pos += held
        leaf = self.end_leaf[node]
        if leaf < 0:
            return None
        transitions.append((node, leaf))
        path.append(leaf)
        return transitions, path

    def node_path(self, grams: Sequence[str], origin: int = START) -> list[int] | None:
        """Node ids ``[START, n0, ..., END leaf]`` a text visits, or ``None`` when it is not a root path."""
        traced = self.trace(grams, origin)
        return None if traced is None else traced[1]

    def symbols_of(self, node: int, offset: int) -> int:
        """How many symbols a located context holds: START counts as one, every gram up to ``offset`` as one."""
        if self.kind[node] == KIND_REAL:
            total = offset + 1
        else:
            total = 1 if node == START else 0
        p = self.parent[node]
        while p > ROOT:
            total += self.held(p) if self.kind[p] == KIND_REAL else 1
            p = self.parent[p]
        return total

    def usable(self, node: int, offset: int, min_count: int = 1) -> bool:
        """Whether a located context can answer: seen ``min_count`` times, with something to continue with."""
        if self.count[node] < min_count:
            return False
        if not self.at_end(node, offset):
            return True
        return bool(self.children[node]) or self.end_leaf[node] >= 0

    def locate(
        self, grams: Sequence[str], min_count: int = 1, ending: bool = False, anchored: bool = True
    ) -> tuple[int, int, int] | None:
        """The deepest usable context of a history: ``(node, offset, symbols)``, or ``None``.

        Tries the history from START first (the whole of it, when it fits under
        the depth), then its suffixes from the root, longest first - the way
        a suffix tree answers for a prefix it has never seen whole: drop the
        oldest gram and ask again.  ``symbols`` is how many symbols the context
        holds, START included.  ``None`` when not even the last gram is known.
        ``anchored=False`` skips the walk from START: for a window cut out of
        a longer history, which does not begin a text however many texts begin
        with those grams.

        With a bound, a context of exactly ``depth`` symbols stands at the end
        of its window: it can never have seen a next gram, only whether texts
        ended there.  So it is consulted for END alone (``ending=True``) and a
        gram is asked of a context of at most ``depth - 1`` symbols - what a
        window can hold, and one more.
        """
        total = len(grams)
        depth = self.depth
        room = None if depth is None else (depth if ending else depth - 1)
        if anchored and (room is None or total + 1 <= room):
            loc = self.walk(START, grams)
            if loc is not None and self.usable(loc[0], loc[1], min_count):
                return (loc[0], loc[1], total + 1)
        longest = total if room is None else min(total, room)
        for length in range(longest, 0, -1):
            loc = self.walk(ROOT, grams[total - length :])
            if loc is not None and self.usable(loc[0], loc[1], min_count):
                return (loc[0], loc[1], length)
        return None

    # -- inversion -----------------------------------------------------------

    def invert(self) -> None:
        """Negate every in-edge weight and every node's activation (``a`` and ``k``); toggle ``inverted``.

        Both ``a`` and ``k`` move, as in the cyclic graph: negating the
        amplitude alone leaves ``-f + 2k``, the negation only while ``k`` is 0,
        and ``k`` is learned.  With both negated every edge signal is exactly
        negated, every ranking reverses exactly, and two inversions are the
        identity.
        """
        self.w = [-v for v in self.w]
        self.a = [-v for v in self.a]
        self.k = [-v for v in self.k]
        self.inverted = not self.inverted
        self.version += 1

    def flip_nodes(self, nodes: Iterable[int] | dict[int, float], mode: str = "activation", amount: float = 1.0) -> int:
        """Move the activation (``a`` and ``k``) or the state ``z`` of nodes toward their negation: ``value *= 1 - 2 * amount``.

        Amount 1 is a full sign flip, 0.5 zeroes the value, a smaller amount
        attenuates it; ``nodes`` may be a ``{node: amount}`` mapping.  A
        node's activation sign enters the score of every edge into and out of
        it, so flipping every other node of a path flips every edge of the
        path - and in a tree that parity always exists (a tree is bipartite),
        which is the one thing the cyclic graph cannot promise.  An END leaf
        may be flipped: it belongs to one context, so its sign is that path's
        alone, where the cyclic graph's END is shared by every text.  The root,
        START, dead and unknown ids and zero amounts are ignored; returns how
        many nodes changed.
        """
        if mode not in ("activation", "state"):
            raise ValueError(f"mode must be 'activation' or 'state', got {mode!r}")
        targets = (self.a, self.k) if mode == "activation" else (self.z,)
        items = nodes.items() if isinstance(nodes, dict) else ((n, amount) for n in set(nodes))
        changed = 0
        for node, amt in items:
            amt = float(amt)
            if node < FIRST or node >= len(self.a) or not self.alive[node] or amt <= 0:
                continue
            scale = 1.0 - 2.0 * amt
            for target in targets:
                target[node] *= scale
            changed += 1
        if changed:
            self.version += 1
        return changed

    # -- invariants ----------------------------------------------------------

    def check_invariants(self, texts: Iterable[str] | None = None, compressed: bool = False) -> None:
        """Assert every structural invariant; ``texts`` adds the suffix property for what was observed.

        The ones that say *no cycles*: every alive node but the root has one
        alive parent that lists it; following parents from any node reaches
        the root; and a walk down from the root visits every alive node
        exactly once - the parents and the children agree on one tree.
        """
        labels, kind, parent, children, end_leaf, alive = (
            self.labels, self.kind, self.parent, self.children, self.end_leaf, self.alive,
        )
        n, ov = self._n, self._ov
        view = self._view
        total = len(labels)
        assert alive[ROOT] and kind[ROOT] == KIND_ROOT and parent[ROOT] == -1 and labels[ROOT] == ""
        assert alive[START] and kind[START] == KIND_START and parent[START] == ROOT and labels[START] == START_LABEL
        assert START not in children[ROOT].values() and end_leaf[ROOT] < 0
        real = ends = 0
        for i in range(total):
            if not alive[i]:
                assert labels[i] == "" and not children[i] and end_leaf[i] < 0 and parent[i] == -1, i
                continue
            if kind[i] == KIND_REAL:
                real += 1
                assert len(view(labels[i])) >= n, (i, labels[i])
            elif kind[i] == KIND_END:
                ends += 1
                assert labels[i] == END_LABEL and not children[i] and end_leaf[i] < 0, i
            assert self.count[i] >= 0
            if i == ROOT:
                continue
            p = parent[i]
            assert 0 <= p < total and alive[p], (i, p)
            if kind[i] == KIND_REAL:
                assert children[p].get(self.encoding.first_gram(labels[i])) == i, (i, p)
                if kind[p] == KIND_REAL and ov:
                    vp, vi = view(labels[p]), view(labels[i])
                    assert vp[len(vp) - ov :] == vi[:ov], (p, i)
            elif kind[i] == KIND_END:
                assert end_leaf[p] == i, (i, p)
            else:
                assert i == START
            # following parents reaches the root: no node is its own ancestor
            seen = {i}
            q = p
            while q != ROOT:
                assert q not in seen, f"cycle through node {q}"
                seen.add(q)
                q = parent[q]
        assert real == self._n_real and ends == self._n_ends, (real, self._n_real, ends, self._n_ends)
        # one walk down from the root meets every alive node exactly once
        visited: set[int] = set()
        stack = [ROOT]
        while stack:
            p = stack.pop()
            assert p not in visited, f"node {p} reached twice"
            visited.add(p)
            kids = list(children[p].values())
            if end_leaf[p] >= 0:
                kids.append(end_leaf[p])
            if p == ROOT:
                kids.append(START)
            for c in kids:
                assert parent[c] == p, (c, p)
                stack.append(c)
        assert visited == set(self.alive_nodes()), "the parents and the children disagree"
        for i in range(FIRST, total):
            if alive[i] and kind[i] == KIND_REAL:
                v = view(labels[i])
                for j in range(len(v) - ov):
                    assert _piece(v, j, j + n) in self.grams, (i, _piece(v, j, j + n))
        if compressed:
            for i in range(FIRST, total):
                if alive[i] and kind[i] == KIND_REAL:
                    assert not (len(children[i]) == 1 and end_leaf[i] < 0), f"unary chain at node {i}"
        if self.depth is None:
            for i in range(total):
                if alive[i] and kind[i] != KIND_END:
                    kids = self.child_items(i)
                    assert self.count[i] == sum(self.count[c] for c in kids), (i, self.count[i], kids)
        if texts is not None:
            enc = self.encoding
            for text in texts:
                grams = enc.encode(text)
                if not grams:
                    continue
                for origin, symbols, ended in self._windows(grams):
                    loc = self.walk(origin, symbols)
                    assert loc is not None, (text, origin, symbols)
                    if ended:
                        assert self.at_end(*loc) and end_leaf[loc[0]] >= 0, (text, origin, symbols, loc)
                if self.depth is None or len(grams) + 1 <= self.depth:
                    assert self.node_path(grams) is not None, text

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot; dead nodes are compacted (ids remapped), the root and START keep 0 and 1."""
        ids = self.alive_nodes()
        remap = {old: new for new, old in enumerate(ids)}
        state = self.rng.getstate()
        return {
            "format": _TREE_FORMAT,
            "format_version": _TREE_FORMAT_VERSION,
            "seed": self.seed,
            "encoding": self.encoding.to_dict(),
            "depth": self.depth,
            "inverted": self.inverted,
            "version": self.version,
            "structure_version": self.structure_version,
            "grams": sorted(self.grams),
            "rng_state": [state[0], list(state[1]), state[2]],
            "nodes": {
                "labels": [self.labels[i] for i in ids],
                "kind": [self.kind[i] for i in ids],
                "parent": [remap[self.parent[i]] if self.parent[i] >= 0 else -1 for i in ids],
                "count": [self.count[i] for i in ids],
                "w": [self.w[i] for i in ids],
                "z": [self.z[i] for i in ids],
                "a": [self.a[i] for i in ids],
                "b": [self.b[i] for i in ids],
                "h": [self.h[i] for i in ids],
                "k": [self.k[i] for i in ids],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RadixTree":
        """Inverse of :meth:`to_dict`."""
        if not isinstance(d, dict) or d.get("format") != _TREE_FORMAT:
            raise ValueError(f"not a {_TREE_FORMAT} document")
        if int(d.get("format_version", 1)) > _TREE_FORMAT_VERSION:
            raise ValueError(f"unsupported {_TREE_FORMAT} format version {d.get('format_version')}")
        tree = cls(seed=int(d.get("seed", 0)), encoding=Encoding.from_dict(d.get("encoding")), depth=d.get("depth"))
        nodes = d["nodes"]
        labels = list(nodes["labels"])
        kinds = [int(x) for x in nodes["kind"]]
        parents = [int(x) for x in nodes["parent"]]
        total = len(labels)
        if total < FIRST or kinds[ROOT] != KIND_ROOT or kinds[START] != KIND_START:
            raise ValueError("a tree document holds the root and START first")
        tree.labels = labels
        tree.kind = kinds
        tree.parent = parents
        tree.count = [int(x) for x in nodes["count"]]
        tree.w = [float(x) for x in nodes["w"]]
        tree.z = [float(x) for x in nodes["z"]]
        tree.a = [float(x) for x in nodes["a"]]
        tree.b = [float(x) for x in nodes["b"]]
        tree.h = [float(x) for x in nodes["h"]]
        tree.k = [float(x) for x in nodes["k"]]
        tree.alive = [True] * total
        tree.children = [{} for _ in range(total)]
        tree.end_leaf = [-1] * total
        first_gram = tree.encoding.first_gram
        real = ends = 0
        for i in range(FIRST, total):
            p = parents[i]
            if kinds[i] == KIND_REAL:
                tree.children[p][first_gram(labels[i])] = i
                real += 1
            elif kinds[i] == KIND_END:
                tree.end_leaf[p] = i
                ends += 1
            else:
                raise ValueError(f"node {i} has kind {kinds[i]}")
        tree._n_real = real
        tree._n_ends = ends
        tree.grams = set(d.get("grams") or [])
        tree.inverted = bool(d.get("inverted", False))
        tree.version = int(d.get("version", 0))
        tree.structure_version = int(d.get("structure_version", 0))
        state = d.get("rng_state")
        if state:
            tree.rng.setstate((int(state[0]), tuple(int(x) for x in state[1]), state[2]))
        return tree

    def __repr__(self) -> str:
        return (
            f"RadixTree(seed={self.seed}, depth={self.depth}, nodes={self.num_nodes()}, ends={self.num_ends()}, "
            f"grams={self.num_grams()}, inverted={self.inverted})"
        )
