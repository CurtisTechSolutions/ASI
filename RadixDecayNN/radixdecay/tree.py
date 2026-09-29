"""DecayTree - the radix tree of every suffix, remembering what it has read and said, and forgetting the rest.

The structure is ``RadixAcyclicNN``'s: a node is a context, the path of grams
from the root; every text is inserted from START and from the root at every
position, the way a suffix tree holds a text; a run of nodes with one child
each is one node (path compression), and a transition into the middle of a
run splits it.  A tree cannot contain a cycle, and
:meth:`DecayTree.check_invariants` proves it after every operation the tests
make.

What is new is the memory.  A node holds nothing but ``seen``, and ``seen``
is what this model *is*:

* **every traversal adds one.**  A training window that passes through a
  context is a traversal; so is a query that takes that context as its next
  token.  The model counts what it has read and what it has said alike.
* **it decays on the model's own clock.**  Time is the number of traversals
  anywhere in the tree (:attr:`traversals`), and a ``seen`` of one fades to a
  half after :attr:`life` traversals (``decay="half-life"``), or to nothing
  (``decay="linear"``: one traversal's worth is lost every ``life``).  The
  decay is applied lazily - a node keeps the value it was last settled at and
  the clock reading it was settled at, and :meth:`seen` reads the two through
  the elapsed time - so reading ``seen`` never changes it, the value depends
  only on how much has happened since, and there is no sweep.
* **shares are the probabilities.**  At a branch each option's share of the
  ``seen`` among the options is its probability; nothing is learned by a
  gradient.

There are no weights and no activations.  Sentinels: node ``0`` is the ROOT
(the empty context), node ``1`` is START (the context at the beginning of a
text), and every context a text ended in has its own END leaf.  Nodes removed
by a merge are tombstoned and compacted by :meth:`to_dict`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .encoding import END_LABEL, START_LABEL, Encoding, _piece

__all__ = [
    "DECAYS", "FIRST", "KIND_END", "KIND_REAL", "KIND_ROOT", "KIND_START", "LIFE", "MIN_SEEN", "ROOT", "START",
    "DecayTree",
]

ROOT, START = 0, 1
"""The two fixed nodes: the empty context, and the context at the beginning of a text."""

FIRST = START + 1
"""The first node id that is a real node or an END leaf."""

KIND_ROOT, KIND_START, KIND_END, KIND_REAL = 0, 1, 2, 3
"""What a node is: the root, START, an END leaf, or a real node holding grams."""

LIFE = 10_000
"""Traversals it takes a lone visit to fade to a half (``half-life``) or to nothing (``linear``): the cyclic count
model's window, ``RadixCyclicNN/DECISIONS.md`` D-022, borrowed for its units - how far back the model calls recent."""

DECAYS = ("half-life", "linear", "none")
"""How ``seen`` fades with elapsed traversals: ``value * 0.5 ** (elapsed / life)``, ``value - elapsed / life`` floored
at 0, or not at all."""

MIN_SEEN = 0.5
"""How much of a visit a context must still hold to be consulted: below half a visit it is forgotten."""

_TREE_FORMAT = "radixdecay-tree"
_TREE_FORMAT_VERSION = 1


class DecayTree:
    """Nodes, the suffix insertion, the radix split / merge, and the decaying ``seen`` that is the memory."""

    def __init__(
        self,
        encoding: Encoding | None = None,
        depth: int | None = None,
        life: float = LIFE,
        decay: str = "half-life",
        min_seen: float = MIN_SEEN,
    ) -> None:
        self.encoding = encoding if encoding is not None else Encoding()
        self.encoding.validate()
        if depth is not None and int(depth) < 2:
            raise ValueError(f"depth must be >= 2 symbols (a context and what follows it) or None, got {depth}")
        if decay not in DECAYS:
            raise ValueError(f"decay must be one of {DECAYS}, got {decay!r}")
        if not (life > 0) or not math.isfinite(life):
            raise ValueError(f"life must be a positive number of traversals, got {life}")
        if min_seen < 0:
            raise ValueError(f"min_seen must be >= 0, got {min_seen}")
        self.depth: int | None = None if depth is None else int(depth)
        """How many symbols (START counts as one, END does not) a root path may hold; ``None`` is unbounded."""
        self.life = float(life)
        self.decay = decay
        self.min_seen = float(min_seen)
        self.traversals = 0
        """The clock: every traversal anywhere, read or said."""
        self.labels: list[str] = []
        self.kind: list[int] = []
        self.parent: list[int] = []
        self.children: list[dict[str, int]] = []
        self.end_leaf: list[int] = []
        self.value: list[float] = []
        """``seen`` as it was last settled."""
        self.stamp: list[int] = []
        """The clock reading each node was last settled at."""
        self.alive: list[bool] = []
        self.grams: set[str] = set()
        """Every distinct gram ever read."""
        self.structure_version = 0
        self._totals: dict[int, tuple[tuple[int, int], float]] = {}
        """``option_total`` per node, keyed by the clock and structure it was computed under."""
        self._n_real = 0
        self._n_ends = 0
        self._n = self.encoding.n
        self._ov = self.encoding.overlap
        self._view = self.encoding.view
        self._last = self.encoding.last_unit
        self._new_node("", KIND_ROOT, -1)
        self._new_node(START_LABEL, KIND_START, ROOT)

    # -- construction --------------------------------------------------------

    def _new_node(self, label: str, kind: int, parent: int, value: float = 0.0, stamp: int | None = None) -> int:
        nid = len(self.labels)
        self.labels.append(label)
        self.kind.append(kind)
        self.parent.append(parent)
        self.children.append({})
        self.end_leaf.append(-1)
        self.value.append(float(value))
        self.stamp.append(self.traversals if stamp is None else int(stamp))
        self.alive.append(True)
        if kind == KIND_REAL:
            self._n_real += 1
        elif kind == KIND_END:
            self._n_ends += 1
        self.structure_version += 1
        return nid

    def _new_real(self, label: str, parent: int) -> int:
        nid = self._new_node(label, KIND_REAL, parent)
        self.children[parent][self.encoding.first_gram(label)] = nid
        return nid

    def _new_end(self, parent: int) -> int:
        nid = self._new_node(END_LABEL, KIND_END, parent)
        self.end_leaf[parent] = nid
        return nid

    # -- the memory ----------------------------------------------------------

    def seen(self, i: int) -> float:
        """What node ``i`` holds now: its settled value, faded by the traversals since.  Reading it changes nothing."""
        v = self.value[i]
        if v <= 0.0:
            return 0.0
        elapsed = self.traversals - self.stamp[i]
        if elapsed <= 0 or self.decay == "none":
            return v
        if self.decay == "half-life":
            return v * 0.5 ** (elapsed / self.life)
        w = v - elapsed / self.life
        return w if w > 0.0 else 0.0

    def settle(self, i: int) -> float:
        """Write node ``i``'s faded value down at the current reading; returns it."""
        v = self.seen(i)
        self.value[i] = v
        self.stamp[i] = self.traversals
        return v

    def touch(self, i: int, amount: float = 1.0) -> float:
        """A traversal arrives at node ``i``: what it holds is settled, ``amount`` is added, and the clock ticks.

        Reading and saying both come through here; nothing else adds to
        ``seen``.  Returns the node's ``seen`` after the visit.
        """
        v = self.seen(i) + amount
        self.value[i] = v
        self.stamp[i] = self.traversals
        self.traversals += 1
        return v

    def tick(self, traversals: int = 1) -> None:
        """Let time pass: advance the clock without arriving anywhere - what happens to this tree while another
        model, or the world, is busy.  Every ``seen`` fades accordingly."""
        if traversals < 0:
            raise ValueError(f"traversals must be >= 0, got {traversals}")
        self.traversals += int(traversals)

    def total_seen(self) -> float:
        """The sum of ``seen`` over every alive node: how much the tree remembers, in visits."""
        return math.fsum(self.seen(i) for i in range(len(self.labels)) if self.alive[i])

    def remembered(self, min_seen: float | None = None) -> int:
        """How many real nodes still hold at least ``min_seen`` (the tree's own by default) of a visit."""
        floor = self.min_seen if min_seen is None else min_seen
        return sum(1 for i in range(len(self.labels)) if self.alive[i] and self.kind[i] == KIND_REAL and self.seen(i) >= floor)

    # -- sizes and kinds -----------------------------------------------------

    def is_real(self, i: int) -> bool:
        return self.kind[i] == KIND_REAL

    def is_end(self, i: int) -> bool:
        return self.kind[i] == KIND_END

    def held(self, i: int) -> int:
        """How many grams node ``i`` holds (0 for the root, START and END leaves)."""
        return len(self._view(self.labels[i])) - self._ov if self.kind[i] == KIND_REAL else 0

    def label_len(self, i: int) -> int:
        """The length of a real node's label in the encoding's units (0 for sentinels)."""
        return len(self._view(self.labels[i])) if self.kind[i] == KIND_REAL else 0

    def first_gram(self, i: int) -> str:
        return self.encoding.first_gram(self.labels[i])

    def num_nodes(self) -> int:
        """Alive real nodes; the sentinels are not counted."""
        return self._n_real

    def num_ends(self) -> int:
        return self._n_ends

    def num_grams(self) -> int:
        return len(self.grams)

    def label_chars(self) -> int:
        """Units held in real labels: the tree's real size."""
        return sum(self.label_len(i) for i in range(len(self.labels)) if self.alive[i] and self.kind[i] == KIND_REAL)

    def alive_nodes(self) -> list[int]:
        return [i for i, ok in enumerate(self.alive) if ok]

    def real_nodes(self) -> list[int]:
        return [i for i, ok in enumerate(self.alive) if ok and self.kind[i] == KIND_REAL]

    def depth_of(self, i: int) -> int:
        d = 0
        while i != ROOT:
            i = self.parent[i]
            d += 1
        return d

    def max_depth(self) -> int:
        return max((self.depth_of(i) for i in self.alive_nodes()), default=0)

    def child_items(self, p: int) -> list[int]:
        """The options at ``p``: its real children, then its END leaf.  START is never an option."""
        kids = list(self.children[p].values())
        e = self.end_leaf[p]
        if e >= 0:
            kids.append(e)
        return kids

    def num_children(self, p: int) -> int:
        return len(self.children[p]) + (1 if self.end_leaf[p] >= 0 else 0)

    def branches(self) -> int:
        """Nodes with more than one option."""
        return sum(1 for i in self.alive_nodes() if self.num_children(i) > 1)

    # -- shares, probabilities, costs ----------------------------------------

    def option_total(self, p: int) -> float:
        """The ``seen`` the options at ``p`` hold between them - cached until anything moves (the clock or the
        structure), so a quiet walk that asks a wide node again and again computes it once."""
        key = (self.traversals, self.structure_version)
        hit = self._totals.get(p)
        if hit is not None and hit[0] == key:
            return hit[1]
        total = math.fsum(self.seen(c) for c in self.child_items(p))
        self._totals[p] = (key, total)
        return total

    def has_options(self, p: int) -> bool:
        """Whether any option at ``p`` still holds something."""
        return self.option_total(p) > 0.0

    def shares(self, p: int) -> list[tuple[int, float]]:
        """``[(option, share)]`` at ``p``: each option's part of the ``seen`` the options hold between them.

        An option that has faded to nothing is not offered; a node whose
        options have all faded offers nothing, and a walk backs off from it.
        """
        total = self.option_total(p)
        if total <= 0.0:
            return []
        out: list[tuple[int, float]] = []
        for c in self.child_items(p):
            s = self.seen(c)
            if s > 0.0:
                out.append((c, s / total))
        return out

    def child_costs(self, p: int) -> list[tuple[int, float]]:
        """``[(option, -log share)]``: non-negative, what the cheapest path walks by."""
        return [(c, -math.log(s)) for c, s in self.shares(p)]

    def log_share(self, p: int, c: int) -> float | None:
        """``log`` of ``c``'s share at ``p``, ``None`` when ``c`` is not on offer there."""
        if c < FIRST or c >= len(self.labels) or not self.alive[c] or self.parent[c] != p:
            return None
        s = self.seen(c)
        if s <= 0.0:
            return None
        total = self.option_total(p)
        return math.log(s / total) if total > 0.0 else None

    # -- structural operations -----------------------------------------------

    def split(self, node: int, i: int) -> tuple[int, int]:
        """Split a real node between its grams ``i - 1`` and ``i``; the deep half inherits the children, the END
        leaf, and the node's ``seen`` (value and stamp) - ``RadixAcyclicNN``'s split, without the parameters."""
        if node < 0 or node >= len(self.labels) or not self.alive[node]:
            raise ValueError(f"node {node} is not alive")
        if self.kind[node] != KIND_REAL:
            raise ValueError("cannot split a sentinel node")
        held = self.held(node)
        if i < 1 or i >= held:
            raise ValueError(f"split index {i} out of range 1..{held - 1} for label {self.labels[node]!r}")
        view = self._view(self.labels[node])
        n, ov = self._n, self._ov
        b = self._new_node(_piece(view, i), KIND_REAL, node, value=self.value[node], stamp=self.stamp[node])
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
        self.structure_version += 1
        return node, b

    def merge_child(self, p: int) -> bool:
        """Merge ``p``'s single real child into ``p`` when the chain is unary (an END leaf is never merged away).
        The merged node keeps the larger of the two ``seen``, settled now."""
        if p < FIRST or p >= len(self.labels) or not self.alive[p] or self.kind[p] != KIND_REAL:
            return False
        ch = self.children[p]
        if len(ch) != 1 or self.end_leaf[p] >= 0:
            return False
        c = next(iter(ch.values()))
        sp, sc = self.settle(p), self.seen(c)
        if sc > sp:
            self.value[p] = sc
        self.labels[p] = self.encoding.join_units(self.labels[p], _piece(self._view(self.labels[c]), self._ov))
        self.children[p] = self.children[c]
        for g in self.children[p].values():
            self.parent[g] = p
        e = self.end_leaf[c]
        self.end_leaf[p] = e
        if e >= 0:
            self.parent[e] = p
        self.labels[c] = ""
        self.children[c] = {}
        self.end_leaf[c] = -1
        self.parent[c] = -1
        self.alive[c] = False
        self.value[c] = 0.0
        self._n_real -= 1
        self.structure_version += 1
        return True

    def compress(self) -> int:
        """Merge every unary chain; 0 on a tree built by reading alone (only :meth:`split` leaves chains)."""
        merges = 0
        while True:
            done = 0
            for p in range(FIRST, len(self.labels)):
                if self.alive[p] and self.kind[p] == KIND_REAL:
                    while self.merge_child(p):
                        done += 1
            if done == 0:
                return merges
            merges += done

    # -- reading -------------------------------------------------------------

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

    def read(self, grams: Sequence[str]) -> int:
        """Read a text's grams: every window of it, from START and from the root, every node it passes through
        visited (:meth:`touch`).  Returns the traversals made."""
        if not grams:
            return 0
        self.encoding.check_grams(grams)
        self.grams.update(grams)
        before = self.traversals
        for origin, symbols, ended in self._windows(grams):
            self._insert(origin, symbols, ended)
        return self.traversals - before

    def _insert(self, node: int, symbols: list[str], ended: bool) -> None:
        """Radix insertion of one window under ``node``, visiting every node on the way."""
        labels = self.labels
        children = self.children
        view = self._view
        ov = self._ov
        tails = [self._last(g) for g in symbols]
        self.touch(node)
        pos = 0
        total = len(symbols)
        while True:
            if pos == total:
                if ended:
                    leaf = self.end_leaf[node]
                    if leaf < 0:
                        leaf = self._new_end(node)
                    self.touch(leaf)
                return
            g = symbols[pos]
            nxt = children[node].get(g)
            if nxt is None:
                leaf = self._new_real(self.encoding.decode_grams(symbols[pos:]), node)
                self.touch(leaf)
                if ended:
                    self.touch(self._new_end(leaf))
                return
            lv = view(labels[nxt])
            held = len(lv) - ov
            remaining = total - pos
            limit = held if held < remaining else remaining
            i = 1
            while i < limit and lv[i + ov] == tails[pos + i]:
                i += 1
            if i < held:
                if i == remaining:
                    if not ended:
                        self.touch(nxt)
                        return
                    self.split(nxt, i)
                else:
                    self.split(nxt, i)
                held = i
            self.touch(nxt)
            node = nxt
            pos += held

    # -- walking -------------------------------------------------------------

    def walk(self, origin: int, grams: Sequence[str]) -> tuple[int, int] | None:
        """Where ``grams`` lead from ``origin``: ``(node, offset)`` - the index of the last matched gram inside the
        node's label, ``held - 1`` when the walk landed on the node - or ``None``."""
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
        return offset == self.held(node) - 1

    def node_path(self, grams: Sequence[str], origin: int = START) -> list[int] | None:
        """Node ids ``[START, ..., END leaf]`` a whole text visits, or ``None`` when it is not a root path."""
        node = origin
        pos = 0
        total = len(grams)
        labels = self.labels
        children = self.children
        view = self._view
        ov = self._ov
        tails = [self._last(g) for g in grams]
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
            path.append(nxt)
            node = nxt
            pos += held
        leaf = self.end_leaf[node]
        if leaf < 0:
            return None
        path.append(leaf)
        return path

    def symbols_of(self, node: int, offset: int) -> int:
        """Symbols a located context holds: START counts as one, every gram up to ``offset`` as one."""
        total = offset + 1 if self.kind[node] == KIND_REAL else (1 if node == START else 0)
        p = self.parent[node]
        while p > ROOT:
            total += self.held(p) if self.kind[p] == KIND_REAL else 1
            p = self.parent[p]
        return total

    def usable(self, node: int, offset: int) -> bool:
        """Whether a located context can answer: it still holds ``min_seen`` of a visit, and has something to
        continue with - a run to finish, or an option that has not faded to nothing."""
        if self.seen(node) < self.min_seen:
            return False
        if not self.at_end(node, offset):
            return True
        return self.has_options(node)

    def locate(
        self, grams: Sequence[str], ending: bool = False, anchored: bool = True, longest: int | None = None
    ) -> tuple[int, int, int] | None:
        """The deepest usable context of a history: ``(node, offset, symbols)``, or ``None``.

        The history from START first (the whole of it, when the window can
        hold it), then its suffixes from the root, longest first.  Under a
        bound a context of exactly ``depth`` symbols is consulted for END alone
        (``ending``).  ``anchored=False`` skips the walk from START, for a
        window cut out of a longer history; ``longest`` considers no suffix of
        more grams than that (and then not the walk from START either), for a
        walk backing off from a context it has just left.
        """
        total = len(grams)
        depth = self.depth
        room = None if depth is None else (depth if ending else depth - 1)
        if longest is not None and longest < total:
            anchored = False
        if anchored and (room is None or total + 1 <= room):
            loc = self.walk(START, grams)
            if loc is not None and self.usable(loc[0], loc[1]):
                return (loc[0], loc[1], total + 1)
        longest = total if longest is None else min(total, longest)
        if room is not None and room < longest:
            longest = room
        for length in range(longest, 0, -1):
            loc = self.walk(ROOT, grams[total - length :])
            if loc is not None and self.usable(loc[0], loc[1]):
                return (loc[0], loc[1], length)
        return None

    # -- invariants ----------------------------------------------------------

    def check_invariants(self, texts: Iterable[str] | None = None, compressed: bool = False) -> None:
        """Assert every structural invariant - the ones that say *no cycles* included - and that the memory is sane."""
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
            assert self.value[i] >= 0.0 and self.stamp[i] <= self.traversals, (i, self.value[i], self.stamp[i])
            if kind[i] == KIND_REAL:
                real += 1
                assert len(view(labels[i])) >= n, (i, labels[i])
            elif kind[i] == KIND_END:
                ends += 1
                assert labels[i] == END_LABEL and not children[i] and end_leaf[i] < 0, i
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
            seen_ids = {i}
            q = p
            while q != ROOT:
                assert q not in seen_ids, f"cycle through node {q}"
                seen_ids.add(q)
                q = parent[q]
        assert real == self._n_real and ends == self._n_ends
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
        """JSON-serialisable snapshot; dead nodes compacted, the root and START first.  A saved file holds settled
        values: what the tree remembered at the moment it was written, and the clock it was written at."""
        ids = self.alive_nodes()
        remap = {old: new for new, old in enumerate(ids)}
        return {
            "format": _TREE_FORMAT,
            "format_version": _TREE_FORMAT_VERSION,
            "encoding": self.encoding.to_dict(),
            "depth": self.depth,
            "life": self.life,
            "decay": self.decay,
            "min_seen": self.min_seen,
            "traversals": self.traversals,
            "structure_version": self.structure_version,
            "grams": sorted(self.grams),
            "nodes": {
                "labels": [self.labels[i] for i in ids],
                "kind": [self.kind[i] for i in ids],
                "parent": [remap[self.parent[i]] if self.parent[i] >= 0 else -1 for i in ids],
                "seen": [self.seen(i) for i in ids],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecayTree":
        """Inverse of :meth:`to_dict`: every node arrives settled at the file's clock."""
        if not isinstance(d, dict) or d.get("format") != _TREE_FORMAT:
            raise ValueError(f"not a {_TREE_FORMAT} document")
        if int(d.get("format_version", 1)) > _TREE_FORMAT_VERSION:
            raise ValueError(f"unsupported {_TREE_FORMAT} format version {d.get('format_version')}")
        tree = cls(
            encoding=Encoding.from_dict(d.get("encoding")), depth=d.get("depth"), life=float(d.get("life", LIFE)),
            decay=str(d.get("decay", "half-life")), min_seen=float(d.get("min_seen", MIN_SEEN)),
        )
        nodes = d["nodes"]
        labels = list(nodes["labels"])
        kinds = [int(x) for x in nodes["kind"]]
        parents = [int(x) for x in nodes["parent"]]
        total = len(labels)
        if total < FIRST or kinds[ROOT] != KIND_ROOT or kinds[START] != KIND_START:
            raise ValueError("a tree document holds the root and START first")
        clock = int(d.get("traversals", 0))
        tree.traversals = clock
        tree.labels = labels
        tree.kind = kinds
        tree.parent = parents
        tree.value = [float(x) for x in nodes["seen"]]
        tree.stamp = [clock] * total
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
        tree.structure_version = int(d.get("structure_version", 0))
        return tree

    def __repr__(self) -> str:
        return (
            f"DecayTree(depth={self.depth}, decay={self.decay!r}, life={self.life:g}, nodes={self.num_nodes()}, "
            f"ends={self.num_ends()}, grams={self.num_grams()}, traversals={self.traversals})"
        )
