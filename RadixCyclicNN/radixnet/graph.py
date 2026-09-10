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
feature, not an error.  Node ids ``0`` (START) and ``1`` (END) are sentinels.

Storage is flat parallel lists indexed by node id / edge id; per-node dicts
hold the edges for O(1) lookup.  Removed nodes and edges are tombstoned (ids
are never reused) and compacted only by :meth:`to_dict`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K
from .backend import CSR, NodeParams
from .encoding import END_LABEL, START_LABEL, WINDOW, Decoder, Encoder

__all__ = ["START", "END", "RadixCyclicGraph", "Z_RANGE", "W_LOW", "W_HIGH"]

START, END = 0, 1

_W = WINDOW          # trigram length
_OV = WINDOW - 1     # overlap between consecutive labels

Z_RANGE = 4.5
"""New node state ``z`` is drawn uniformly from ``[-Z_RANGE, Z_RANGE]``."""
W_LOW, W_HIGH = 0.5, 1.5
"""New edge weights are drawn uniformly from ``[W_LOW, W_HIGH]`` (negated when inverted)."""

_GRAPH_FORMAT = "radixnet-graph"
_GRAPH_FORMAT_VERSION = 1


class RadixCyclicGraph:
    """Nodes, edges, trigram index and the radix split / merge operations."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.labels: list[str] = []
        self.z: list[float] = []
        self.a: list[float] = []
        self.b: list[float] = []
        self.h: list[float] = []
        self.k: list[float] = []
        self.count: list[int] = []
        self.alive: list[bool] = []
        self.children: list[dict[int, int]] = []
        self.parents: list[dict[int, int]] = []
        self.edge_w: list[float] = []
        self.edge_count: list[int] = []
        self.edge_alive: list[bool] = []
        self.trigram_index: dict[str, tuple[int, int]] = {}
        self.inverted = False
        self.version = 0
        self.structure_version = 0
        self._n_alive_nodes = 0
        self._n_alive_edges = 0
        self._cost_cache: dict[int, list[tuple[int, int, float]]] = {}
        self._cost_cache_version = -1
        self._act_cache: list[float] = []
        self._act_cache_version = -1
        self._decoder = Decoder(_W)
        self._encoder = Encoder(_W)
        self._new_node(START_LABEL)
        self._new_node(END_LABEL)

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
        self.alive.append(True)
        self.children.append({})
        self.parents.append({})
        self._n_alive_nodes += 1
        self.version += 1
        self.structure_version += 1
        return nid

    def _new_edge(self, p: int, c: int, count: int = 0) -> int:
        """Allocate edge ``p -> c`` with a fresh random weight; ``p -> c`` must not exist."""
        w = self.rng.uniform(W_LOW, W_HIGH)
        if self.inverted:
            w = -w
        e = len(self.edge_w)
        self.edge_w.append(w)
        self.edge_count.append(count)
        self.edge_alive.append(True)
        self.children[p][c] = e
        self.parents[c][p] = e
        self._n_alive_edges += 1
        self.version += 1
        self.structure_version += 1
        return e

    def _create_trigram_node(self, trigram: str) -> int:
        """Create the node for an unknown trigram and index it."""
        if len(trigram) != _W:
            raise ValueError(f"expected a {_W}-character trigram, got {trigram!r}")
        nid = self._new_node(trigram)
        self.trigram_index[trigram] = (nid, 0)
        return nid

    # -- sizes ---------------------------------------------------------------

    def num_nodes(self) -> int:
        """Alive nodes including START/END."""
        return self._n_alive_nodes

    def num_edges(self) -> int:
        """Alive edges."""
        return self._n_alive_edges

    def num_trigrams(self) -> int:
        """Distinct trigrams stored in the graph."""
        return len(self.trigram_index)

    def compression_ratio(self) -> float:
        """Trigrams per real (non-sentinel) node."""
        return len(self.trigram_index) / max(1, self._n_alive_nodes - 2)

    def alive_nodes(self) -> list[int]:
        """Ids of alive nodes in increasing order (START and END first)."""
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
        if node_id == START or node_id == END:
            raise ValueError("cannot split a sentinel node")
        if node_id < 0 or node_id >= len(self.labels) or not self.alive[node_id]:
            raise ValueError(f"node {node_id} is not alive")
        label = self.labels[node_id]
        length = len(label)
        if i < 1 or i > length - _W:
            raise ValueError(f"split index {i} out of range 1..{length - _W} for label {label!r}")
        a_id = node_id
        b_id = self._new_node(
            label[i:],
            z=self.z[a_id], a=self.a[a_id], b=self.b[a_id], h=self.h[a_id], k=self.k[a_id],
            count=self.count[a_id],
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
        self._new_edge(a_id, b_id, count=self.count[a_id])
        index = self.trigram_index
        for j in range(i, length - _OV):
            index[label[j : j + _W]] = (b_id, j - i)
        self.labels[a_id] = label[: i + _OV]
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
        if p == START or p == END or p < 0 or p >= len(self.labels) or not self.alive[p]:
            return False
        ch = self.children[p]
        if len(ch) != 1:
            return False
        c = next(iter(ch))
        if c == p or c == START or c == END:
            return False
        pc = self.parents[c]
        if len(pc) != 1:
            return False
        lp = self.labels[p]
        lc = self.labels[c]
        shift = len(lp) - _OV
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
        for j in range(len(lc) - _OV):
            index[lc[j : j + _W]] = (p, shift + j)
        self.labels[p] = lp + lc[_OV:]
        self.labels[c] = ""
        self.count[p] = max(self.count[p], self.count[c])
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
            for p in range(2, len(self.labels)):
                if alive[p]:
                    while merge(p):
                        done += 1
            if done == 0:
                return merges
            merges += done

    def observe_sequence(self, trigrams: Sequence[str], count: bool = True) -> list[tuple[int, int]]:
        """Register a training sequence ``START -> t0 -> ... -> tn -> END``.

        Splits nodes so that every transition either stays inside a compressed
        node (deterministic, no edge) or runs from the *last* trigram of one
        node to the *first* trigram of another over an edge that is created on
        demand.  Returns the ``(parent_id, edge_id)`` transitions in order -
        exactly what the backend trains on.  ``count`` also bumps the node and
        edge visit counters.  Consecutive trigrams must overlap by two
        characters (``x[1:] == y[:2]``) as produced by :class:`Encoder`.
        """
        if not trigrams:
            return []
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
        e = children[START].get(px)
        if e is None:
            e = new_edge(START, px)
        transitions.append((START, e))
        if count:
            counts[START] += 1
            counts[px] += 1
            ecounts[e] += 1

        for idx in range(1, len(trigrams)):
            y = trigrams[idx]
            loc = index_get(y)
            if loc is None:
                py, oy = create(y), 0
            else:
                py, oy = loc
            if py == px and oy == ox + 1:
                ox = oy
                x = y
                continue
            if x[1:] != y[:_OV]:
                raise ValueError(f"trigrams {x!r} -> {y!r} do not overlap")
            if ox + _W < len(labels[px]):
                split(px, ox + 1)
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

        if ox + _W < len(labels[px]):
            split(px, ox + 1)
            did_split = True
        e = children[px].get(END)
        if e is None:
            e = new_edge(px, END)
        transitions.append((px, e))
        if count:
            counts[END] += 1
            ecounts[e] += 1

        if did_split:
            # a later split may have moved an out-edge recorded earlier in this
            # sequence to the new B node; re-derive the transitions structurally
            traced = self._trace(trigrams)
            if traced is None:
                raise RuntimeError("internal error: observed sequence is not walkable")
            transitions = traced[0]
        return transitions

    def _trace(self, trigrams: Sequence[str]) -> tuple[list[tuple[int, int]], list[int]] | None:
        """Walk a sequence through the structure without modifying it.

        Returns ``(transitions, node_path)`` (``node_path`` starts with START
        and ends with END) or ``None`` when a trigram is unknown, an edge is
        missing or a split would be required.
        """
        if not trigrams:
            return None
        index = self.trigram_index
        labels = self.labels
        children = self.children
        loc = index.get(trigrams[0])
        if loc is None or loc[1] != 0:
            return None
        px, ox = loc
        e = children[START].get(px)
        if e is None:
            return None
        transitions = [(START, e)]
        path = [START, px]
        for idx in range(1, len(trigrams)):
            loc = index.get(trigrams[idx])
            if loc is None:
                return None
            py, oy = loc
            if py == px and oy == ox + 1:
                ox = oy
                continue
            if oy != 0 or ox + _W != len(labels[px]):
                return None
            e = children[px].get(py)
            if e is None:
                return None
            transitions.append((px, e))
            path.append(py)
            px, ox = py, 0
        if ox + _W != len(labels[px]):
            return None
        e = children[px].get(END)
        if e is None:
            return None
        transitions.append((px, e))
        path.append(END)
        return transitions, path

    def node_path(self, trigrams: Sequence[str]) -> list[int] | None:
        """Node ids ``[START, n0, ..., END]`` visited by a sequence, or ``None``.

        ``None`` means the sequence is not representable by the current
        structure without a split (unknown trigram, missing edge, or a
        transition into / out of the middle of a compressed node).
        """
        traced = self._trace(trigrams)
        return None if traced is None else traced[1]

    def invert(self) -> None:
        """Flip every alive edge weight and every node's ``a``; toggle ``inverted``."""
        ew = self.edge_w
        for e, ok in enumerate(self.edge_alive):
            if ok:
                ew[e] = -ew[e]
        self.a = [-v for v in self.a]
        self.inverted = not self.inverted
        self.version += 1

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

    def child_costs(self, p: int) -> list[tuple[int, int, float]]:
        """``[(child_id, edge_id, -log softmax prob)]``, cached until ``version`` changes."""
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

        Node ids are remapped (START=0, END=1, then alive nodes in id order);
        the trigram index is rebuilt from the labels on load.  The RNG state is
        included so training continues reproducibly after a reload.
        """
        remap: dict[int, int] = {}
        for old, ok in enumerate(self.alive):
            if ok:
                remap[old] = len(remap)
        src: list[int] = []
        dst: list[int] = []
        ew: list[float] = []
        ec: list[int] = []
        for old, new in remap.items():
            for c, e in self.children[old].items():
                src.append(new)
                dst.append(remap[c])
                ew.append(self.edge_w[e])
                ec.append(self.edge_count[e])
        order = list(remap)
        state = self.rng.getstate()
        return {
            "format": _GRAPH_FORMAT,
            "format_version": _GRAPH_FORMAT_VERSION,
            "seed": self.seed,
            "inverted": self.inverted,
            "version": self.version,
            "structure_version": self.structure_version,
            "nodes": {
                "labels": [self.labels[i] for i in order],
                "z": [self.z[i] for i in order],
                "a": [self.a[i] for i in order],
                "b": [self.b[i] for i in order],
                "h": [self.h[i] for i in order],
                "k": [self.k[i] for i in order],
                "count": [self.count[i] for i in order],
            },
            "edges": {"src": src, "dst": dst, "w": ew, "count": ec},
            "rng_state": [state[0], list(state[1]), state[2]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RadixCyclicGraph":
        """Inverse of :meth:`to_dict`."""
        if d.get("format") != _GRAPH_FORMAT:
            raise ValueError(f"not a {_GRAPH_FORMAT} document")
        nodes = d["nodes"]
        edges = d["edges"]
        labels = list(nodes["labels"])
        n = len(labels)
        if n < 2 or labels[START] != START_LABEL or labels[END] != END_LABEL:
            raise ValueError("graph document is missing the START/END sentinels")
        g = cls(seed=int(d.get("seed", 0)))
        g.inverted = bool(d.get("inverted", False))
        g.labels = labels
        g.z = [float(v) for v in nodes["z"]]
        g.a = [float(v) for v in nodes["a"]]
        g.b = [float(v) for v in nodes["b"]]
        g.h = [float(v) for v in nodes["h"]]
        g.k = [float(v) for v in nodes["k"]]
        g.count = [int(v) for v in nodes["count"]]
        if not (len(g.z) == len(g.a) == len(g.b) == len(g.h) == len(g.k) == len(g.count) == n):
            raise ValueError("node arrays have inconsistent lengths")
        g.alive = [True] * n
        g.children = [{} for _ in range(n)]
        g.parents = [{} for _ in range(n)]
        g._n_alive_nodes = n
        index: dict[str, tuple[int, int]] = {}
        for nid in range(2, n):
            label = labels[nid]
            if len(label) < _W:
                raise ValueError(f"node {nid} label {label!r} is shorter than {_W}")
            for o in range(len(label) - _OV):
                t = label[o : o + _W]
                if t in index:
                    raise ValueError(f"trigram {t!r} appears in two nodes")
                index[t] = (nid, o)
        g.trigram_index = index
        src, dst = edges["src"], edges["dst"]
        ew, ec = edges["w"], edges["count"]
        if not (len(src) == len(dst) == len(ew) == len(ec)):
            raise ValueError("edge arrays have inconsistent lengths")
        g.edge_w = [float(v) for v in ew]
        g.edge_count = [int(v) for v in ec]
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
        g.version = int(d.get("version", 0))
        g.structure_version = int(d.get("structure_version", 0))
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
        n = len(labels)
        assert (
            len(self.z) == len(self.a) == len(self.b) == len(self.h) == len(self.k)
            == len(self.count) == len(alive) == len(children) == len(parents) == n
        ), "node arrays have inconsistent lengths"
        assert len(self.edge_w) == len(self.edge_count) == len(self.edge_alive), (
            "edge arrays have inconsistent lengths"
        )
        assert n >= 2 and alive[START] and alive[END], "sentinels must exist and be alive"
        assert labels[START] == START_LABEL and labels[END] == END_LABEL, "sentinel labels changed"
        assert not parents[START], "START must not have parents"
        assert not children[END], "END must not have children"
        assert sum(alive) == self._n_alive_nodes, "alive node counter is stale"
        seen: dict[int, tuple[int, int]] = {}
        for p in range(n):
            if not alive[p]:
                assert not children[p] and not parents[p], f"dead node {p} still has edges"
                continue
            if p > END:
                assert len(labels[p]) >= _W, f"node {p} label {labels[p]!r} shorter than {_W}"
            for c, e in children[p].items():
                assert 0 <= e < len(self.edge_w), f"edge id {e} out of range on {p}->{c}"
                assert self.edge_alive[e], f"dead edge {e} referenced by {p}->{c}"
                assert e not in seen, f"edge {e} listed twice ({seen.get(e)} and {(p, c)})"
                seen[e] = (p, c)
                assert alive[c], f"edge {p}->{c} points at dead node {c}"
                assert c != START and p != END, f"edge {p}->{c} touches a sentinel illegally"
                assert parents[c].get(p) == e, f"edge {p}->{c} ({e}) missing from parents[{c}]"
                if p > END and c > END:
                    assert labels[p][-_OV:] == labels[c][:_OV], (
                        f"edge {p}->{c} violates window overlap: {labels[p]!r} -> {labels[c]!r}"
                    )
            for q, e in parents[p].items():
                assert children[q].get(p) == e, f"parents[{p}] lists {q} ({e}) but children[{q}] does not"
        assert len(seen) == sum(self.edge_alive) == self._n_alive_edges, "alive edge bookkeeping is stale"
        index = self.trigram_index
        expected = 0
        for p in range(2, n):
            if not alive[p]:
                continue
            label = labels[p]
            for o in range(len(label) - _OV):
                t = label[o : o + _W]
                assert index.get(t) == (p, o), f"trigram {t!r} of node {p}@{o} indexed as {index.get(t)}"
                expected += 1
        assert len(index) == expected, f"trigram_index has {len(index)} entries, expected {expected}"
        for t, (p, o) in index.items():
            assert alive[p], f"trigram {t!r} maps to dead node {p}"
            assert labels[p][o : o + _W] == t, f"trigram {t!r} indexed at {p}@{o} but label is {labels[p]!r}"
        if compressed:
            for p in range(2, n):
                if alive[p] and len(children[p]) == 1:
                    c = next(iter(children[p]))
                    assert c == p or c <= END or len(parents[c]) != 1, (
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
                assert decoded == text, f"round trip of {text!r} gave {decoded!r}"

    def __repr__(self) -> str:
        return (
            f"RadixCyclicGraph(nodes={self.num_nodes()}, edges={self.num_edges()}, "
            f"trigrams={self.num_trigrams()}, inverted={self.inverted})"
        )
