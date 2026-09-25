"""RadixCyclicGraph - the self-compressing cyclic graph.

Nodes carry a label of at least three characters; a node with label length
``L`` *contains* the ``L - 2`` trigrams ``label[i:i+3]`` and every trigram of
the training data lives in exactly one node (``trigram_index``).  Two
radix-tree operations shape the graph:

* :meth:`RadixCyclicGraph.split` cuts a compressed node when a transition is
  observed *into* or *out of* its middle;
* :meth:`RadixCyclicGraph.merge_child` / :meth:`RadixCyclicGraph.compress`
  glue unary chains (``p`` has one child ``c``, ``c`` has one parent ``p``)
  back into a single node - path compression.

Repeated trigrams create cycles (``"aaaa"`` gives a self-loop); that is a
feature, not an error.  Node ids ``0`` (START), ``1`` (END), ``2`` (BACK) and
``3`` (THINK) are sentinels.

Storage is flat parallel lists indexed by node id / edge id; per-node dicts
hold the edges for O(1) lookup.  Removed nodes and edges are tombstoned (ids
are never reused) and compacted only by :meth:`to_dict`.

Every counter here - node and edge visit counts, the traversal total, the
version stamps - is a *cyclic counter* (:mod:`radixnet.counter`): it wraps back
to 0 at :data:`~radixnet.counter.COUNTER_LIMIT` and counts the wrap as a reset,
so nothing grows out of an ``int64`` or out of a JSON number.  The wrapping
itself happens in :meth:`RadixCyclicGraph.carry_counters`, never in the
counting loops.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K
from .attention import AttentionBand
from .backend import CSR, NodeParams
from .counter import COUNTER_LIMIT, CyclicCounter, as_float, carry_series, total
from .encoding import (
    BACK_LABEL, CHARS, END_LABEL, START_LABEL, THINK_LABEL, WINDOW, Decoder, Encoder, Encoding, _piece,
)

__all__ = ["START", "END", "BACK", "THINK", "FIRST", "RadixCyclicGraph", "Z_RANGE", "W_LOW", "W_HIGH"]

START, END, BACK, THINK = 0, 1, 2, 3
"""The sentinels.  START and END are where a text begins and ends - observed in the corpus, like everything
else.  BACK is where the graph has learned that a walk *goes round*: nothing in a corpus says so, so its edges
are taught by the voices that caught themselves repeating (:func:`radixnet.dialogue.backtrack`).  An edge
``p -> BACK`` competes for probability with ``p``'s real children, so the more often walks through ``p`` had to
be backed out of, the likelier the search is to hand over there instead of carrying on.

THINK is where the graph *stops to think*, and it faces both ways.  An edge ``p -> THINK`` is taught by
experience - an event at ``p`` made the model think there (:func:`radixnet.thinking.think`) - and, like BACK's,
competes with ``p``'s real children for probability.  Its out-edges are how thoughts begin: a thought is a text
whose walk starts at THINK instead of START (:meth:`RadixCyclicGraph.observe_sequence` with ``origin=THINK``),
observed from an LLM's thinking the way texts are observed from a corpus.  Neither sentinel is a continuation:
no text passes through them, and the search never walks into them (:func:`radixnet.search.onward`)."""

FIRST = THINK + 1
"""The first node id that is not a sentinel."""

ORIGINS = (START, THINK)
"""The sentinels a walk may begin at: START for a text, THINK for a thought."""

_NO_NODES: frozenset[int] = frozenset()

_W = WINDOW          # the default n of the n-gram
_OV = WINDOW - 1     # the overlap that goes with it (stride 1)
# A graph is built in its own Encoding (``self.encoding``): ``_n`` and ``_ov``
# below are that encoding's n and overlap, and are what the code reads.  These
# two module constants are only the defaults a graph gets when none is given.

Z_RANGE = 4.5
"""New node state ``z`` is drawn uniformly from ``[-Z_RANGE, Z_RANGE]``."""
W_LOW, W_HIGH = 0.5, 1.5
"""New edge weights are drawn uniformly from ``[W_LOW, W_HIGH]`` (negated when inverted)."""

BACK_Z = Z_RANGE
"""BACK's state, fixed at the far edge of the range instead of drawn - the strongest activation a state in
range can give (``|f|`` within 0.3 % of ``|a|``).

Every other node's ``z`` is random, and BACK is not every other node: it is never trained (no text passes
through it) and it must not move the random stream, or adding the sentinel would have changed every seeded
model ever written.  A fixed, firmly non-zero activation is also what lets an edge into it carry a learned
weight at all: the cost of an edge is a softmax over ``w * f(p) * f(c)``, so a sentinel activating at zero
could never be learned toward."""

THINK_Z = -Z_RANGE
"""THINK's state, fixed at the *other* edge of the range: as firmly non-zero as BACK's, drawn from no stream, and
distinguishable from it - the activation a walk hands over to when it thinks is the mirror of the one it hands
over to when it goes round."""

_GRAPH_FORMAT = "radixnet-graph"
_GRAPH_FORMAT_VERSION = 4   # 2 added the counter reset fields; 3 the BACK sentinel and 4 the THINK sentinel
                            # (older files gain an unvisited one on load, and their node ids shift up by one)


def _with_sentinel(d: dict, at: int, label: str, z: float, since: int) -> dict:
    """A graph document with the sentinel ``label`` at node id ``at``: format ``since`` as it is, older upgraded.

    Files written before the sentinel existed have the sentinels before it and then their real nodes, so it is
    inserted at ``at`` and every node id from there up shifts by one.  It arrives unvisited and with no edges: a
    model that has never caught itself repeating has nothing to say about where it goes round, and one that has
    never thought has nothing to say about where it stops to think, or how a thought begins.
    """
    if int(d.get("format_version", 1)) >= since:
        return d
    nodes = dict(d["nodes"])
    edges = dict(d["edges"])
    labels = list(nodes["labels"])
    if len(labels) < at or (len(labels) > at and labels[at] == label):
        return d
    nodes["labels"] = labels[:at] + [label] + labels[at:]
    # the sentinel takes START's activation parameters, whatever kind of model wrote the file (the count
    # model's a = 0, k = 1 make every activation 1; the sine model's are the defaults), and its own fixed state
    for key, blank in (("z", z), ("a", None), ("b", None), ("h", None), ("k", None),
                       ("count", 0), ("count_resets", 0)):
        if key in nodes and nodes[key]:
            values = list(nodes[key])
            nodes[key] = values[:at] + [values[START] if blank is None else blank] + values[at:]
    shift = lambda i: i + 1 if i >= at else i  # noqa: E731 - one expression, used twice below
    edges["src"] = [shift(int(i)) for i in edges["src"]]
    edges["dst"] = [shift(int(i)) for i in edges["dst"]]
    return {**d, "nodes": nodes, "edges": edges, "format_version": since}


def _with_back(d: dict) -> dict:
    """A graph document with the BACK sentinel in it: format 3 as it is, anything older upgraded."""
    return _with_sentinel(d, BACK, BACK_LABEL, BACK_Z, 3)


def _with_think(d: dict) -> dict:
    """A graph document with the THINK sentinel in it: format 4 as it is, anything older upgraded (BACK first)."""
    return _with_sentinel(_with_back(d), THINK, THINK_LABEL, THINK_Z, _GRAPH_FORMAT_VERSION)




class RadixCyclicGraph:
    """Nodes, edges, trigram index and the radix split / merge operations."""

    def __init__(self, seed: int = 0, encoding: Encoding | None = None) -> None:
        self.seed = int(seed)
        self.encoding = encoding if encoding is not None else Encoding()
        """How this graph turns text into grams and its labels back into text.

        Fixed here: every label, every index key and every offset below is
        measured in this encoding's units.  It travels with the model file."""
        self.encoding.validate()
        self.attention = AttentionBand()
        """Where inside a gram a correction's blame and credit land (:mod:`radixnet.attention`).

        Off by default - each changed unit is charged to the step that wrote
        it.  Unlike the encoding it changes nothing the graph holds, so it can
        be switched at any time; it travels with the model file while it is on."""
        self.rng = random.Random(self.seed)
        self.labels: list[str] = []
        self.z: list[float] = []
        self.a: list[float] = []
        self.b: list[float] = []
        self.h: list[float] = []
        self.k: list[float] = []
        self.count: list[int] = []
        self.count_resets: dict[int, int] = {}
        self.alive: list[bool] = []
        self.children: list[dict[int, int]] = []
        self.parents: list[dict[int, int]] = []
        self.edge_w: list[float] = []
        self.edge_count: list[int] = []
        self.edge_count_resets: dict[int, int] = {}
        self.edge_alive: list[bool] = []
        self.trigram_index: dict[str, tuple[int, int]] = {}
        self.inverted = False
        self.traversals = CyclicCounter()
        self.version = CyclicCounter()
        self.structure_version = CyclicCounter()
        self._n_alive_nodes = 0
        self._n_alive_edges = 0
        self._cost_cache: dict[int, list[tuple[int, int, float]]] = {}
        self._cost_cache_version = -1
        self._act_cache: list[float] = []
        self._act_cache_version = -1
        self._n_is_chars = self.encoding.unit == CHARS
        self._n = self.encoding.n
        self._ov = self.encoding.overlap
        self._stride = self.encoding.stride
        self._decoder = Decoder(encoding=self.encoding)
        self._encoder = Encoder(encoding=self.encoding)
        self._new_node(START_LABEL)
        self._new_node(END_LABEL)
        self._new_node(BACK_LABEL, z=BACK_Z)
        self._new_node(THINK_LABEL, z=THINK_Z)

    # -- counters ------------------------------------------------------------

    @staticmethod
    def _set_resets(resets: dict[int, int], key: int, value: int) -> None:
        """Store one reset count, keeping the map sparse (missing means 'never wrapped')."""
        if value:
            resets[key] = value
        else:
            resets.pop(key, None)

    def node_count(self, i: int) -> int:
        """How often node ``i`` was visited, exactly, across every reset of its counter."""
        return total(self.count[i], self.count_resets.get(i, 0))

    def edge_traversals(self, e: int) -> int:
        """How often edge ``e`` was traversed, exactly, across every reset of its counter."""
        return total(self.edge_count[e], self.edge_count_resets.get(e, 0))

    def _edge_traversals_f(self, e: int) -> float:
        """:meth:`edge_traversals` as a float - what the weight function (and the Go port) sums."""
        return as_float(self.edge_count[e], self.edge_count_resets.get(e, 0))

    def carry_counters(self, force: bool = False) -> int:
        """Set every counter that reached ``COUNTER_LIMIT`` back to 0, counting the reset; returns how many wrapped.

        This is the *only* place the visit counters wrap, so the counting loops
        stay plain increments and the Go port can count into a raw ``int64``
        from several goroutines.  Call it at a safe point - the end of an
        epoch, before a save.  While the graph has not seen
        ``COUNTER_LIMIT`` increments no counter can have reached the limit
        (:attr:`traversals` counts them all and so bounds every single one), so
        the sweep is skipped after one comparison; ``force`` runs it anyway.
        """
        if not force and not self.traversals.resets:
            return 0
        return carry_series(self.count, self.count_resets) + carry_series(self.edge_count, self.edge_count_resets)

    # -- construction helpers ------------------------------------------------

    def _new_node(
        self,
        label: str,
        z: float | None = None,
        a: float | None = None,
        b: float = DEFAULT_B,
        h: float = DEFAULT_H,
        k: float = DEFAULT_K,
        count: int = 0,
        count_resets: int = 0,
    ) -> int:
        """Allocate a node (not indexed); ``z`` random and ``a`` default unless given."""
        if z is None:
            z = self.rng.uniform(-Z_RANGE, Z_RANGE)
        if a is None:
            a = -DEFAULT_A if self.inverted else DEFAULT_A
        nid = len(self.labels)
        self.labels.append(label)
        self.z.append(z)
        self.a.append(a)
        self.b.append(b)
        self.h.append(h)
        self.k.append(k)
        self.count.append(count)
        if count_resets:
            self.count_resets[nid] = count_resets
        self.alive.append(True)
        self.children.append({})
        self.parents.append({})
        self._n_alive_nodes += 1
        self.version += 1
        self.structure_version += 1
        return nid

    def _new_edge(self, p: int, c: int, count: int = 0, count_resets: int = 0) -> int:
        """Allocate edge ``p -> c`` with a fresh random weight; ``p -> c`` must not exist."""
        w = self.rng.uniform(W_LOW, W_HIGH)
        if self.inverted:
            w = -w
        e = len(self.edge_w)
        self.edge_w.append(w)
        self.edge_count.append(count)
        if count_resets:
            self.edge_count_resets[e] = count_resets
        self.edge_alive.append(True)
        self.children[p][c] = e
        self.parents[c][p] = e
        self._n_alive_edges += 1
        self.version += 1
        self.structure_version += 1
        return e

    def label_len(self, node: int) -> int:
        """The length of a node's label in the encoding's units (characters by default)."""
        label = self.labels[node]
        # the encoding's own split, not ``str.split()``: a length that
        # disagrees with the units view is a split index the graph cannot
        # honour
        return len(label) if self._n_is_chars else self.encoding.length(label)

    def _create_trigram_node(self, trigram: str) -> int:
        """Create the node for an unknown gram and index it."""
        if self.encoding.length(trigram) != self._n:
            raise ValueError(f"expected a gram of {self._n} {self.encoding.unit}s, got {trigram!r}")
        nid = self._new_node(trigram)
        self.trigram_index[trigram] = (nid, 0)
        return nid

    # -- sizes ---------------------------------------------------------------

    def num_nodes(self) -> int:
        """Alive nodes including the sentinels."""
        return self._n_alive_nodes

    def num_edges(self) -> int:
        """Alive edges."""
        return self._n_alive_edges

    def num_trigrams(self) -> int:
        """Distinct trigrams stored in the graph."""
        return len(self.trigram_index)

    def symbols_of(self, text: str) -> str:
        """Text as *this graph's symbols*, the inverse of :meth:`text_of`, for anything that looks a label up.

        The identity here, and the word model's encoder in a word graph; an
        unread word comes back as the unknown symbol.
        """
        return text

    def text_of(self, label: str) -> str:
        """A label as *text*, for anything that reports one.

        The identity here: a character model's symbols are the text.  A word
        model's symbols are code points standing for words
        (``../SPEC-WordNGrams.md``), and its graph decodes them back before a
        label is shown to anyone.
        """
        return label

    def compression_ratio(self) -> float:
        """Trigrams per real (non-sentinel) node."""
        return len(self.trigram_index) / max(1, self._n_alive_nodes - FIRST)

    def alive_nodes(self) -> list[int]:
        """Ids of alive nodes in increasing order (the sentinels first)."""
        return [i for i, ok in enumerate(self.alive) if ok]

    # -- lookup --------------------------------------------------------------

    def lookup(self, trigram: str) -> tuple[int, int] | None:
        """``(node_id, offset)`` of a trigram or ``None``."""
        return self.trigram_index.get(trigram)

    def get_or_create(self, trigram: str) -> tuple[int, int]:
        """Look a trigram up, creating a fresh 3-char node if it is unknown."""
        loc = self.trigram_index.get(trigram)
        if loc is not None:
            return loc
        return (self._create_trigram_node(trigram), 0)

    def activation_of(self, node_id: int) -> float:
        """``f_i(z_i)`` of one node."""
        return self.a[node_id] * math.sin(self.b[node_id] * (self.z[node_id] - self.h[node_id])) + self.k[node_id]

    def _activations(self) -> list[float]:
        """Activation of every node id, cached per ``version``."""
        if self._act_cache_version != self.version:
            sin = math.sin
            self._act_cache = [
                ai * sin(bi * (zi - hi)) + ki
                for zi, ai, bi, hi, ki in zip(self.z, self.a, self.b, self.h, self.k)
            ]
            self._act_cache_version = self.version
        return self._act_cache

    # -- structural operations -----------------------------------------------

    def split(self, node_id: int, i: int) -> tuple[int, int]:
        """Split a node between its trigrams ``i - 1`` and ``i``.

        ``A`` keeps ``node_id`` with ``label[:i + 2]``; ``B`` is a new node
        with ``label[i:]`` that inherits ``A``'s out-edges (edge ids kept),
        state, activation parameters and count.  ``A`` gets a single new edge
        ``A -> B`` (``edge_count = count[A]``).  Returns ``(A, B)``.
        """
        if node_id < FIRST:
            raise ValueError("cannot split a sentinel node")
        if node_id < 0 or node_id >= len(self.labels) or not self.alive[node_id]:
            raise ValueError(f"node {node_id} is not alive")
        enc = self.encoding
        view = enc.units(self.labels[node_id])
        label = self.labels[node_id]
        length = len(view)
        stride, ov, n = self._stride, self._ov, self._n
        if i < stride or i > length - n or i % stride:
            raise ValueError(
                f"split index {i} out of range {stride}..{length - n} "
                f"(a multiple of the stride {stride}) for label {label!r}"
            )
        a_id = node_id
        a_resets = self.count_resets.get(a_id, 0)
        b_id = self._new_node(
            _piece(view, i),
            z=self.z[a_id], a=self.a[a_id], b=self.b[a_id], h=self.h[a_id], k=self.k[a_id],
            count=self.count[a_id], count_resets=a_resets,
        )
        ch_a = self.children[a_id]
        ch_b = self.children[b_id]
        parents = self.parents
        for c, e in ch_a.items():
            ch_b[c] = e
            pc = parents[c]
            del pc[a_id]
            pc[b_id] = e
        ch_a.clear()
        self._new_edge(a_id, b_id, count=self.count[a_id], count_resets=a_resets)
        index = self.trigram_index
        for j in range(i, length - n + 1, stride):
            index[_piece(view, j, j + n)] = (b_id, j - i)
        self.labels[a_id] = _piece(view, 0, i + ov)
        return (a_id, b_id)

    def merge_child(self, p: int) -> bool:
        """Merge ``p``'s single child ``c`` into ``p`` if the chain is unary.

        Conditions: neither is a sentinel, ``p != c``, ``p`` has exactly one
        child and ``c`` exactly one parent.  ``p`` takes label
        ``labels[p] + labels[c][2:]``, ``c``'s out-edges (ids kept), the
        maximum of the two counts and ``c``'s trigrams; ``c`` and the edge
        ``p -> c`` are tombstoned.  Returns ``True`` if a merge happened.

        The merge preserves the network function: the merged node keeps the
        activation of whichever endpoint has the larger ``|f|`` and the edges
        on the other side are rescaled by the ratio of the two activations
        (``<= 1`` in magnitude, so weights never grow) so that every edge
        score ``w * f_parent * f_child`` - and therefore every child
        probability and cost - is unchanged.  A cycle edge ``c -> p`` (which
        becomes the self-loop ``p -> p``) is covered by the same rescale.
        """
        if p < FIRST or p >= len(self.labels) or not self.alive[p]:
            return False
        ch = self.children[p]
        if len(ch) != 1:
            return False
        c = next(iter(ch))
        if c == p or c < FIRST:
            return False
        pc = self.parents[c]
        if len(pc) != 1:
            return False
        enc = self.encoding
        lp_view, lc_view = enc.units(self.labels[p]), enc.units(self.labels[c])
        lc = self.labels[c]
        shift = len(lp_view) - self._ov
        ew = self.edge_w
        fp = self.activation_of(p)
        fc = self.activation_of(c)
        e = ch.pop(c)
        pc.clear()
        self.edge_alive[e] = False
        self._n_alive_edges -= 1
        parents = self.parents
        if abs(fp) >= abs(fc):
            # keep p's activation; c's out-edges carried f_c, they now carry f_p
            out_ratio = fc / fp if fp != 0.0 else 1.0  # fp == 0 implies fc == 0: all scores stay 0
        else:
            # adopt c's activation; p's in-edges carried f_p, they now carry f_c
            out_ratio = 1.0
            in_ratio = fp / fc
            for e2 in parents[p].values():
                ew[e2] *= in_ratio
            self.z[p], self.a[p], self.b[p], self.h[p], self.k[p] = (
                self.z[c], self.a[c], self.b[c], self.h[c], self.k[c]
            )
        cc = self.children[c]
        for target, e2 in cc.items():
            if target == c:
                target = p
            pt = parents[target]
            pt.pop(c, None)
            pt[p] = e2
            ch[target] = e2
            ew[e2] *= out_ratio
        cc.clear()
        index = self.trigram_index
        for j in range(0, len(lc_view) - self._n + 1, self._stride):
            index[_piece(lc_view, j, j + self._n)] = (p, shift + j)
        self.labels[p] = enc.join(_piece(lp_view, 0), _piece(lc_view, self._ov))
        self.labels[c] = ""
        if self.node_count(c) > self.node_count(p):
            self.count[p] = self.count[c]
            self._set_resets(self.count_resets, p, self.count_resets.get(c, 0))
        self.alive[c] = False
        self._n_alive_nodes -= 1
        self.version += 1
        self.structure_version += 1
        return True

    def compress(self) -> int:
        """Merge every unary chain until none remains; returns the merge count."""
        merges = 0
        alive = self.alive
        merge = self.merge_child
        while True:
            done = 0
            for p in range(FIRST, len(self.labels)):
                if alive[p]:
                    while merge(p):
                        done += 1
            if done == 0:
                return merges
            merges += done

    def observe_sequence(
        self, trigrams: Sequence[str], count: bool = True, origin: int = START
    ) -> list[tuple[int, int]]:
        """Register a training sequence ``START -> t0 -> ... -> tn -> END``.

        Splits nodes so that every transition either stays inside a compressed
        node (deterministic, no edge) or runs from the *last* trigram of one
        node to the *first* trigram of another over an edge that is created on
        demand.  Returns the ``(parent_id, edge_id)`` transitions in order -
        exactly what the backend trains on.  ``count`` also bumps the node and
        edge visit counters.  Consecutive trigrams must overlap by two
        characters (``x[1:] == y[:2]``) as produced by :class:`Encoder`.

        ``origin`` is the sentinel the sequence begins at: ``START`` for a
        text, ``THINK`` for a *thought* - the same structure, the same
        counting, the same edge into END, only the first edge leaves the other
        sentinel (:data:`ORIGINS`).
        """
        if not trigrams:
            return []
        if origin not in ORIGINS:
            raise ValueError(f"a sequence begins at START or THINK, not at node {origin}")
        index = self.trigram_index
        labels = self.labels
        children = self.children
        counts = self.count
        ecounts = self.edge_count
        index_get = index.get
        split = self.split
        new_edge = self._new_edge
        create = self._create_trigram_node
        did_split = False
        transitions: list[tuple[int, int]] = []

        enc = self.encoding
        stride, ov, n = self._stride, self._ov, self._n
        llen = self.label_len
        x = trigrams[0]
        loc = index_get(x)
        if loc is None:
            px, ox = create(x), 0
        else:
            px, ox = loc
        if ox != 0:
            px = split(px, ox)[1]
            ox = 0
            did_split = True
        e = children[origin].get(px)
        if e is None:
            e = new_edge(origin, px)
        transitions.append((origin, e))
        if count:
            counts[origin] += 1
            counts[px] += 1
            ecounts[e] += 1

        for idx in range(1, len(trigrams)):
            y = trigrams[idx]
            loc = index_get(y)
            if loc is None:
                py, oy = create(y), 0
            else:
                py, oy = loc
            if py == px and oy == ox + stride:
                ox = oy
                x = y
                continue
            if enc.piece(x, stride) != enc.piece(y, 0, ov):
                raise ValueError(f"grams {x!r} -> {y!r} do not overlap")
            if ox + n < llen(px):
                split(px, ox + stride)
                did_split = True
                py, oy = index[y]
            if oy != 0:
                py = split(py, oy)[1]
                oy = 0
                did_split = True
                px, ox = index[x]
            e = children[px].get(py)
            if e is None:
                e = new_edge(px, py)
            transitions.append((px, e))
            if count:
                counts[py] += 1
                ecounts[e] += 1
            px, ox, x = py, 0, y

        if ox + n < llen(px):
            split(px, ox + stride)
            did_split = True
        e = children[px].get(END)
        if e is None:
            e = new_edge(px, END)
        transitions.append((px, e))
        if count:
            counts[END] += 1
            ecounts[e] += 1

        if count:
            # every transition bumped one node counter and one edge counter, plus the origin's:
            # the total bounds each of them and so decides when carry_counters() has work
            self.traversals += 2 * len(transitions) + 1

        if did_split:
            # a later split may have moved an out-edge recorded earlier in this
            # sequence to the new B node; re-derive the transitions structurally
            traced = self._trace(trigrams, origin)
            if traced is None:
                raise RuntimeError("internal error: observed sequence is not walkable")
            transitions = traced[0]
        return transitions

    def _trace(
        self, trigrams: Sequence[str], origin: int = START
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        """Walk a sequence through the structure without modifying it.

        Returns ``(transitions, node_path)`` (``node_path`` starts with the
        ``origin`` - START for a text, THINK for a thought - and ends with
        END) or ``None`` when a trigram is unknown, an edge is missing or a
        split would be required.
        """
        if not trigrams:
            return None
        index = self.trigram_index
        children = self.children
        stride, n = self._stride, self._n
        llen = self.label_len
        loc = index.get(trigrams[0])
        if loc is None or loc[1] != 0:
            return None
        px, ox = loc
        e = children[origin].get(px)
        if e is None:
            return None
        transitions = [(origin, e)]
        path = [origin, px]
        for idx in range(1, len(trigrams)):
            loc = index.get(trigrams[idx])
            if loc is None:
                return None
            py, oy = loc
            if py == px and oy == ox + stride:
                ox = oy
                continue
            if oy != 0 or ox + n != llen(px):
                return None
            e = children[px].get(py)
            if e is None:
                return None
            transitions.append((px, e))
            path.append(py)
            px, ox = py, 0
        if ox + n != llen(px):
            return None
        e = children[px].get(END)
        if e is None:
            return None
        transitions.append((px, e))
        path.append(END)
        return transitions, path

    def trace(
        self, trigrams: Sequence[str], origin: int = START
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        """``(transitions, node_path)`` of a sequence through the current structure, or ``None``.

        Like :meth:`observe_sequence` without the observing: nothing is
        created, split or counted, so ``None`` means the structure cannot
        represent the sequence as it stands (see :meth:`node_path`).
        """
        return self._trace(trigrams, origin)

    def node_path(self, trigrams: Sequence[str], origin: int = START) -> list[int] | None:
        """Node ids ``[START, n0, ..., END]`` visited by a sequence, or ``None``.

        ``None`` means the sequence is not representable by the current
        structure without a split (unknown trigram, missing edge, or a
        transition into / out of the middle of a compressed node).  A thought
        is traced from ``origin=THINK``.
        """
        traced = self._trace(trigrams, origin)
        return None if traced is None else traced[1]

    def invert(self) -> None:
        """Negate every alive edge weight and every node's activation; toggle ``inverted``.

        A node's activation is ``f(x) = a * sin(b * (x - h)) + k``, so negating
        it means negating **both** ``a`` and ``k``: flipping the amplitude
        alone leaves ``-a * sin(u) + k``, which is ``-f(x) + 2k`` and equals
        ``-f(x)`` only while ``k`` is 0.  ``k`` is learned (``df/dk`` is 1, so
        every training step moves it), so by the time 2NRL inverts anything it
        is not 0 and the difference is real: the edge signal
        ``w * f_p * f_c`` would not change sign cleanly and the softmax over a
        node's children would not reverse.  With both negated every edge signal
        is exactly negated, the ranking reverses exactly, and two inversions
        are the identity.
        """
        ew = self.edge_w
        for e, ok in enumerate(self.edge_alive):
            if ok:
                ew[e] = -ew[e]
        self.a = [-v for v in self.a]
        self.k = [-v for v in self.k]
        self.inverted = not self.inverted
        self.version += 1

    def flip_nodes(self, nodes: Iterable[int] | dict[int, float], mode: str = "activation", amount: float = 1.0) -> int:
        """Move the activation amplitude ``a`` (``mode="activation"``) or the state ``z`` (``"state"``) of nodes toward
        their negation: ``value *= 1 - 2 * amount``.

        ``amount`` 1 is a full sign flip, 0.5 zeroes the value (the node's
        transitions become neutral), a small amount only attenuates it - so
        the update can grow with how bad a path was.  ``nodes`` may be a
        ``{node: amount}`` mapping for per-node amounts.  A node's activation
        sign enters the score ``w * f_p * f_c`` of every edge into and out of
        it, so flipping every other node of a path makes that path's
        transitions as unlikely as they were likely - a local counterpart of
        :meth:`invert`, which flips the whole network.  ``"state"`` scales the
        trained node value instead (the same thing while ``h`` and ``k`` are
        0).  Sentinels, dead and unknown ids and zero amounts are ignored;
        returns how many nodes changed.
        """
        if mode not in ("activation", "state"):
            raise ValueError(f"mode must be 'activation' or 'state', got {mode!r}")
        # "activation" scales the whole output of the unit, which is `a * sin(u) + k`,
        # so both the amplitude and the offset move: at amount 1 that is an exact
        # negation and at 0.5 the node really does go neutral (f = 0).  Scaling `a`
        # alone would leave f = k at 0.5 and -f + 2k at 1.  "state" scales the node's
        # trained value instead, which is a different operation (see the docstring).
        targets = (self.a, self.k) if mode == "activation" else (self.z,)
        alive = self.alive
        items = nodes.items() if isinstance(nodes, dict) else ((n, amount) for n in set(nodes))
        changed = 0
        for n, amt in items:
            amt = float(amt)
            if n < FIRST or n >= len(self.a) or not alive[n] or amt <= 0:
                continue
            scale = 1.0 - 2.0 * amt
            for target in targets:
                target[n] *= scale
            changed += 1
        if changed:
            self.version += 1
        return changed

    # -- scores / probabilities / costs --------------------------------------

    def child_scores(self, p: int) -> list[tuple[int, float]]:
        """``[(child_id, w * f_p * f_c)]`` over ``p``'s out-edges."""
        acts = self._activations()
        fp = acts[p]
        ew = self.edge_w
        return [(c, ew[e] * fp * acts[c]) for c, e in self.children[p].items()]

    def child_probs(self, p: int) -> list[tuple[int, float]]:
        """Numerically stable softmax over :meth:`child_scores`."""
        scores = self.child_scores(p)
        if not scores:
            return []
        m = max(s for _, s in scores)
        exps = [(c, math.exp(s - m)) for c, s in scores]
        total = sum(v for _, v in exps)
        return [(c, v / total) for c, v in exps]

    def child_evidence(self, p: int, prev: int | None = None) -> list[tuple[int, int, float, float]]:
        """``[(child, edge, merit, penalty)]`` over ``p``'s out-edges: what speaks *for* each step and what against.

        The split the punishment traversal walks on (:mod:`radixnet.penalty`).
        ``merit`` is the evidence for the step with every reward taken out of
        it; ``penalty >= 0`` is the punishment the edge carries.  The sine
        model keeps no separate ledger of its punishments - 2NRL trains a
        failure in and then inverts it, so what a punishment leaves behind *is*
        a negative score on the edges of the path - which is why here the
        negative part of the score ``w * f_p * f_c`` is the penalty and the
        positive part the merit.  ``prev``, the node the walk arrived from, is
        what a model that counts paths prices the step by; ignored here.
        """
        acts = self._activations()
        fp = acts[p]
        ew = self.edge_w
        out: list[tuple[int, int, float, float]] = []
        for c, e in self.children[p].items():
            score = ew[e] * fp * acts[c]
            out.append((c, e, max(0.0, score), max(0.0, -score)))
        return out

    def nudge_edge(self, p: int, c: int, amount: float) -> bool:
        """Move the weight of ``p -> c`` so the transition gets ``amount`` *likelier* (negative: dearer).

        The cost of an edge is a softmax over ``w * f(p) * f(c)``, so which way the weight has to move depends
        on the sign of the two activations it sits between.  Returns ``False`` when there is no such edge.
        """
        e = self.children[p].get(c) if FIRST <= p < len(self.children) else None
        if e is None or not amount:
            return False
        acts = self._activations()
        pull = acts[p] * acts[c]
        if pull > 0:
            self.edge_w[e] += amount
        elif pull < 0:
            self.edge_w[e] -= amount
        else:
            return False
        self.version += 1
        return True

    def observe_back(self, p: int, went: int | None = None, instead: int | None = None,
                     amount: float = 1.0) -> int:
        """Teach what a voice learned by backing out of a repeat at ``p``.

        Nothing in a corpus says where a walk loops, so this is the one thing the graph learns from
        *experience* rather than from observation - a voice that caught itself repeating and had to back up
        (:func:`radixnet.dialogue.backtrack`).  Three things are taught at once, and all three are ordinary
        learned quantities:

        * ``p -> BACK`` is created on first use and bumped like any observed transition, its weight moving to
          make the transition likelier.  It competes with ``p``'s real children for probability, so every
          hand-over raises the model's own estimate that walks through ``p`` go round - and once that estimate
          beats the real children, the search hands over there by itself, wherever it is walking.
        * ``went`` - the child the walk was about to loop through - gets ``amount`` *dearer*.
        * ``instead`` - the child the voice took after backing up - gets ``amount`` *cheaper*.

        The first is where it goes round; the other two are what to do instead.  Returns the ``BACK`` edge id.
        """
        if p < FIRST or p >= len(self.labels) or not self.alive[p]:
            raise ValueError(f"node {p} is not a real node to go back from")
        if amount < 0:
            raise ValueError(f"amount must be >= 0, got {amount}")
        e = self.children[p].get(BACK)
        if e is None:
            e = self._new_edge(p, BACK)
        self.count[BACK] += 1
        self.edge_count[e] += 1
        self.traversals += 1
        self.version += 1
        self.nudge_edge(p, BACK, amount)
        if went is not None and went != BACK:
            self.nudge_edge(p, went, -amount)
        if instead is not None and instead != BACK:
            self.nudge_edge(p, instead, amount)
        return e

    def back_cost(self, p: int) -> float | None:
        """What the model thinks a walk arriving at ``p`` costs to go round, or ``None`` when it has no idea.

        ``None`` for a node that was never backed out of; otherwise the cost of its ``BACK`` edge, to compare
        with the costs of its real children - when it is the cheapest of them the model's most likely next step
        is to *stop*, which is what the search acts on.
        """
        for c, _e, cost in self.child_costs(p):
            if c == BACK:
                return cost
        return None

    def observe_think(self, p: int, amount: float = 1.0) -> int:
        """Teach that something at ``p`` made the model stop and think.

        The twin of :meth:`observe_back`, learned the same way - from experience, never from a corpus: an event
        at ``p`` (a voice catching itself repeating, a question asked about the text that ends here, a thought
        questioning itself) called for a thought, and the model remembers where (:func:`radixnet.thinking.think`).
        ``p -> THINK`` is created on first use and bumped like any observed transition, its weight moving to
        make the transition likelier; it competes with ``p``'s real children for probability, so the oftener
        walks through ``p`` had to stop and think, the likelier a thought passing through ``p`` is to question
        itself there.  Nothing is taught about what to do instead - that is the thought's business, and what it
        hands over to when it stops (BACK, or nothing).  Returns the ``THINK`` edge id.
        """
        if p < FIRST or p >= len(self.labels) or not self.alive[p]:
            raise ValueError(f"node {p} is not a real node to think at")
        if amount < 0:
            raise ValueError(f"amount must be >= 0, got {amount}")
        e = self.children[p].get(THINK)
        if e is None:
            e = self._new_edge(p, THINK)
        self.count[THINK] += 1
        self.edge_count[e] += 1
        self.traversals += 1
        self.version += 1
        self.nudge_edge(p, THINK, amount)
        return e

    def think_cost(self, p: int) -> float | None:
        """What the model thinks it costs to stop and think at ``p``, or ``None`` when it never had to.

        The cost of ``p``'s ``THINK`` edge, to compare with the costs of its real children: when it is the
        cheapest of them the model's most likely next step is to question what it is doing, which is what a
        thought passing through ``p`` acts on (:func:`radixnet.thinking.think`).
        """
        for c, _e, cost in self.child_costs(p):
            if c == THINK:
                return cost
        return None

    def thinks_at(self, p: int) -> bool:
        """Whether the model has learned to stop and think at ``p``: its THINK edge is the cheapest way on."""
        costs = self.child_costs(p)
        thinking = [cost for c, _e, cost in costs if c == THINK]
        if not thinking:
            return False
        return all(cost >= thinking[0] for c, _e, cost in costs if c != THINK and c != BACK)

    def nodes_with_paths(self) -> set[int]:
        """Nodes whose costs depend on where the walk came from; empty unless the model counts paths."""
        return _NO_NODES

    def edge_punishment(self, e: int) -> float:
        """What the model has been taught *against* one edge.

        A graph with no record of failure has nothing against any of them, so
        this is 0 here; :class:`~radixnet.countnet.CountRewardGraph` overrides
        it with the penalty side of the edge's reward (see
        ``../SPEC-LeastPunished.md``).
        """
        return 0.0

    def step_punishment(self, prev: int | None, e: int) -> float:
        """The punishment of one step taken from ``prev`` (0 without a record of failure)."""
        return 0.0

    def child_steps(self, p: int, prev: int | None = None) -> list[tuple[int, int, float, float]]:
        """:meth:`child_costs` with the punishment of each step beside its cost.

        The least-punished traversal reads the fourth element; every other
        search reads the first three, which is why the two are one call rather
        than two.
        """
        return [(c, e, cost, self.step_punishment(prev, e)) for c, e, cost in self.child_costs(p, prev)]

    def child_costs(self, p: int, prev: int | None = None) -> list[tuple[int, int, float]]:
        """``[(child_id, edge_id, -log softmax prob)]``, cached until ``version`` changes.

        ``prev`` is the node the walk arrived from, which a model that counts
        paths (:class:`~radixnet.countnet.CountRewardGraph`) uses to price the
        same edge differently in different contexts; here it is ignored.
        """
        if self._cost_cache_version != self.version:
            self._cost_cache.clear()
            self._cost_cache_version = self.version
        costs = self._cost_cache.get(p)
        if costs is None:
            acts = self._activations()
            fp = acts[p]
            ew = self.edge_w
            items = [(c, e, ew[e] * fp * acts[c]) for c, e in self.children[p].items()]
            if items:
                m = max(s for _, _, s in items)
                lse = m + math.log(sum(math.exp(s - m) for _, _, s in items))
                costs = [(c, e, lse - s) for c, e, s in items]
            else:
                costs = []
            self._cost_cache[p] = costs
        return costs

    # -- backend interchange -------------------------------------------------

    def to_csr(self) -> CSR:
        """CSR over *all* node ids (dead nodes have empty rows)."""
        indptr = [0]
        indices: list[int] = []
        edge_ids: list[int] = []
        weights: list[float] = []
        edge_pos: dict[int, int] = {}
        ew = self.edge_w
        for ch in self.children:
            for c, e in ch.items():
                edge_pos[e] = len(indices)
                indices.append(c)
                edge_ids.append(e)
                weights.append(ew[e])
            indptr.append(len(indices))
        return CSR(indptr, indices, edge_ids, weights, edge_pos)

    def apply_csr_weights(self, csr: CSR, weights: Sequence[float]) -> None:
        """Write weights (CSR order) back to the edges."""
        edge_ids = csr.edge_ids
        if len(weights) != len(edge_ids):
            raise ValueError(f"expected {len(edge_ids)} weights, got {len(weights)}")
        ew = self.edge_w
        for e, w in zip(edge_ids, weights):
            ew[e] = w
        self.version += 1

    def node_params(self) -> NodeParams:
        """Copies of ``z, a, b, h, k`` in node-id order."""
        return NodeParams(list(self.z), list(self.a), list(self.b), list(self.h), list(self.k))

    def apply_node_params(self, params: NodeParams) -> None:
        """Write node parameters back (length must match the node count)."""
        if len(params) != len(self.labels):
            raise ValueError(f"expected params for {len(self.labels)} nodes, got {len(params)}")
        self.z = list(params.z)
        self.a = list(params.a)
        self.b = list(params.b)
        self.h = list(params.h)
        self.k = list(params.k)
        self.version += 1

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot with dead nodes/edges compacted away.

        Node ids are remapped (the four sentinels first, then alive nodes in id order);
        the trigram index is rebuilt from the labels on load.  The RNG state is
        included so training continues reproducibly after a reload.
        """
        self.carry_counters()  # a saved file always holds a wrapped reading
        remap: dict[int, int] = {}
        for old, ok in enumerate(self.alive):
            if ok:
                remap[old] = len(remap)
        src: list[int] = []
        dst: list[int] = []
        ew: list[float] = []
        ec: list[int] = []
        er: list[int] = []
        for old, new in remap.items():
            for c, e in self.children[old].items():
                src.append(new)
                dst.append(remap[c])
                ew.append(self.edge_w[e])
                ec.append(self.edge_count[e])
                er.append(self.edge_count_resets.get(e, 0))
        order = list(remap)
        nr = [self.count_resets.get(i, 0) for i in order]
        state = self.rng.getstate()
        return {
            "format": _GRAPH_FORMAT,
            "format_version": _GRAPH_FORMAT_VERSION,
            # written only when it is not the character trigram of stride 1, so an
            # ordinary file is byte for byte what it always was
            **({} if self.encoding.is_default() else {"encoding": self.encoding.to_dict()}),
            # the attention band, likewise only while it is on: a file without it was written with it off
            **({"attention": self.attention.to_dict()} if self.attention.on else {}),
            "seed": self.seed,
            "inverted": self.inverted,
            "version": self.version.value,
            "version_resets": self.version.resets,
            "structure_version": self.structure_version.value,
            "structure_version_resets": self.structure_version.resets,
            "traversals": self.traversals.value,
            "traversals_resets": self.traversals.resets,
            "nodes": {
                "labels": [self.labels[i] for i in order],
                "z": [self.z[i] for i in order],
                "a": [self.a[i] for i in order],
                "b": [self.b[i] for i in order],
                "h": [self.h[i] for i in order],
                "k": [self.k[i] for i in order],
                "count": [self.count[i] for i in order],
                # the reset counts ride along only once something has actually wrapped
                **({"count_resets": nr} if any(nr) else {}),
            },
            "edges": {"src": src, "dst": dst, "w": ew, "count": ec, **({"count_resets": er} if any(er) else {})},
            "rng_state": [state[0], list(state[1]), state[2]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RadixCyclicGraph":
        """Inverse of :meth:`to_dict`."""
        if d.get("format") != _GRAPH_FORMAT:
            raise ValueError(f"not a {_GRAPH_FORMAT} document")
        d = _with_think(d)
        nodes = d["nodes"]
        edges = d["edges"]
        labels = list(nodes["labels"])
        n = len(labels)
        if n < FIRST or labels[START] != START_LABEL or labels[END] != END_LABEL or labels[BACK] != BACK_LABEL \
                or labels[THINK] != THINK_LABEL:
            raise ValueError("graph document is missing the START/END/BACK/THINK sentinels")
        g = cls(seed=int(d.get("seed", 0)), encoding=Encoding.from_dict(d.get("encoding")))
        g.attention = AttentionBand.from_dict(d.get("attention"))
        enc = g.encoding
        g.inverted = bool(d.get("inverted", False))
        g.labels = labels
        g.z = [float(v) for v in nodes["z"]]
        g.a = [float(v) for v in nodes["a"]]
        g.b = [float(v) for v in nodes["b"]]
        g.h = [float(v) for v in nodes["h"]]
        g.k = [float(v) for v in nodes["k"]]
        g.count = [int(v) for v in nodes["count"]]
        node_resets = nodes.get("count_resets") or []
        g.count_resets = {i: int(v) for i, v in enumerate(node_resets) if int(v)}
        if not (len(g.z) == len(g.a) == len(g.b) == len(g.h) == len(g.k) == len(g.count) == n):
            raise ValueError("node arrays have inconsistent lengths")
        if node_resets and len(node_resets) != n:
            raise ValueError("node arrays have inconsistent lengths")
        g.alive = [True] * n
        g.children = [{} for _ in range(n)]
        g.parents = [{} for _ in range(n)]
        g._n_alive_nodes = n
        index: dict[str, tuple[int, int]] = {}
        for nid in range(FIRST, n):
            view = enc.units(labels[nid])
            if len(view) < enc.n:
                raise ValueError(f"node {nid} label {labels[nid]!r} is shorter than {enc.n} {enc.unit}s")
            for o in range(0, len(view) - enc.n + 1, enc.stride):
                t = _piece(view, o, o + enc.n)
                if t in index:
                    raise ValueError(f"gram {t!r} appears in two nodes")
                index[t] = (nid, o)
        g.trigram_index = index
        src, dst = edges["src"], edges["dst"]
        ew, ec = edges["w"], edges["count"]
        if not (len(src) == len(dst) == len(ew) == len(ec)):
            raise ValueError("edge arrays have inconsistent lengths")
        g.edge_w = [float(v) for v in ew]
        g.edge_count = [int(v) for v in ec]
        edge_resets = edges.get("count_resets") or []
        if edge_resets and len(edge_resets) != len(src):
            raise ValueError("edge arrays have inconsistent lengths")
        g.edge_count_resets = {i: int(v) for i, v in enumerate(edge_resets) if int(v)}
        g.edge_alive = [True] * len(src)
        for e, (p, c) in enumerate(zip(src, dst)):
            if not (0 <= p < n and 0 <= c < n) or c in g.children[p]:
                raise ValueError(f"invalid or duplicate edge {p} -> {c}")
            g.children[p][c] = e
            g.parents[c][p] = e
        g._n_alive_edges = len(src)
        state = d.get("rng_state")
        if state is not None:
            g.rng.setstate((state[0], tuple(state[1]), state[2]))
        g.version = CyclicCounter.from_pair(d.get("version", 0), d.get("version_resets", 0))
        g.structure_version = CyclicCounter.from_pair(
            d.get("structure_version", 0), d.get("structure_version_resets", 0)
        )
        if "traversals" in d:
            g.traversals = CyclicCounter.from_pair(d["traversals"], d.get("traversals_resets", 0))
        else:  # a format 1 file counted into unbounded integers: their sum bounds every one of them
            g.traversals = CyclicCounter(sum(g.count) + sum(g.edge_count))
        g.carry_counters(force=True)  # normalise whatever the file carried, however it was written
        return g

    # -- debugging -----------------------------------------------------------

    def check_invariants(self, texts: Iterable[str] | None = None, compressed: bool = False) -> None:
        """Raise ``AssertionError`` if any structural invariant is violated.

        ``texts`` (optional) are checked for the structural round trip (their
        trigrams must walk through the graph and decode back to the text);
        ``compressed=True`` additionally asserts that no unary chain remains.
        """
        labels, alive = self.labels, self.alive
        children, parents = self.children, self.parents
        enc, n_gram, ov = self.encoding, self._n, self._ov
        n = len(labels)
        assert (
            len(self.z) == len(self.a) == len(self.b) == len(self.h) == len(self.k)
            == len(self.count) == len(alive) == len(children) == len(parents) == n
        ), "node arrays have inconsistent lengths"
        assert len(self.edge_w) == len(self.edge_count) == len(self.edge_alive), (
            "edge arrays have inconsistent lengths"
        )
        for name, values, resets in (
            ("node", self.count, self.count_resets), ("edge", self.edge_count, self.edge_count_resets)
        ):
            assert all(v >= 0 for v in values), f"a {name} counter went negative"
            assert all(0 < r < COUNTER_LIMIT and 0 <= i < len(values) for i, r in resets.items()), (
                f"a {name} counter reset count is out of range or belongs to no counter"
            )
        assert n >= FIRST and alive[START] and alive[END] and alive[BACK] and alive[THINK], (
            "sentinels must exist and be alive"
        )
        assert labels[:FIRST] == [START_LABEL, END_LABEL, BACK_LABEL, THINK_LABEL], "sentinel labels changed"
        assert not parents[START], "START must not have parents"
        assert not children[END], "END must not have children"
        assert not children[BACK], "BACK must not have children"
        assert sum(alive) == self._n_alive_nodes, "alive node counter is stale"
        seen: dict[int, tuple[int, int]] = {}
        for p in range(n):
            if not alive[p]:
                assert not children[p] and not parents[p], f"dead node {p} still has edges"
                continue
            if p >= FIRST:
                assert self.label_len(p) >= n_gram, f"node {p} label {labels[p]!r} shorter than {n_gram}"
            for c, e in children[p].items():
                assert 0 <= e < len(self.edge_w), f"edge id {e} out of range on {p}->{c}"
                assert self.edge_alive[e], f"dead edge {e} referenced by {p}->{c}"
                assert e not in seen, f"edge {e} listed twice ({seen.get(e)} and {(p, c)})"
                seen[e] = (p, c)
                assert alive[c], f"edge {p}->{c} points at dead node {c}"
                assert c != START and p != END and p != BACK, (
                    f"edge {p}->{c} touches a sentinel illegally"
                )
                assert parents[c].get(p) == e, f"edge {p}->{c} ({e}) missing from parents[{c}]"
                if p >= FIRST and c >= FIRST and ov:
                    tail = enc.piece(labels[p], self.label_len(p) - ov)
                    assert tail == enc.piece(labels[c], 0, ov), (
                        f"edge {p}->{c} violates the gram overlap: {labels[p]!r} -> {labels[c]!r}"
                    )
            for q, e in parents[p].items():
                assert children[q].get(p) == e, f"parents[{p}] lists {q} ({e}) but children[{q}] does not"
        assert len(seen) == sum(self.edge_alive) == self._n_alive_edges, "alive edge bookkeeping is stale"
        index = self.trigram_index
        expected = 0
        for p in range(FIRST, n):
            if not alive[p]:
                continue
            view = enc.units(labels[p])
            for o in range(0, len(view) - n_gram + 1, enc.stride):
                t = _piece(view, o, o + n_gram)
                assert index.get(t) == (p, o), f"gram {t!r} of node {p}@{o} indexed as {index.get(t)}"
                expected += 1
            rest = (len(view) - n_gram) % enc.stride
            assert not rest, (
                f"node {p} label {labels[p]!r} holds {len(view)} units, {rest} past its last whole gram"
            )
        assert len(index) == expected, f"trigram_index has {len(index)} entries, expected {expected}"
        for t, (p, o) in index.items():
            assert alive[p], f"gram {t!r} maps to dead node {p}"
            assert enc.piece(labels[p], o, o + n_gram) == t, (
                f"gram {t!r} indexed at {p}@{o} but label is {labels[p]!r}"
            )
        if compressed:
            for p in range(FIRST, n):
                if alive[p] and len(children[p]) == 1:
                    c = next(iter(children[p]))
                    assert c == p or c < FIRST or len(parents[c]) != 1, (
                        f"unary chain {p}->{c} survived compress ({labels[p]!r} -> {labels[c]!r})"
                    )
        if texts is not None:
            for text in texts:
                grams = self._encoder.encode(text)
                if not grams:
                    continue
                path = self.node_path(grams)
                assert path is not None, f"text {text!r} is not walkable through the graph"
                # path[0] is START and path[-1] is END; real nodes may be labelled like a sentinel
                decoded = self._decoder.decode_path(
                    [labels[i] for i in path[1:-1]], 0, True, skip_sentinels=False
                )
                # what comes back is what the encoding can represent: the text itself
                # under a sliding character window, its words under a word encoding,
                # everything but the ragged tail under a grouping one
                want = enc.normalize(text)
                assert decoded == want, f"round trip of {want!r} gave {decoded!r}"

    def __repr__(self) -> str:
        return (
            f"RadixCyclicGraph(nodes={self.num_nodes()}, edges={self.num_edges()}, "
            f"trigrams={self.num_trigrams()}, inverted={self.inverted})"
        )
