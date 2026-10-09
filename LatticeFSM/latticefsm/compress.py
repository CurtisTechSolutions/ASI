"""Compress the matrix into its central node, from the outside in; rebuild it from the middle outward.

The code the central node holds (:class:`Core`) is the whole matrix with the
least loss there is, laid out in :func:`geometry.center_out` order:

* **the structure** - one bit per cell, from the central node outward:
  whether anything was ever written to that edge.  An untouched edge is the
  prototype's, so a bit is all it costs, and it comes back exactly;
* **the touched edges' fields**, in the same order: the six integers of an
  edge (``seen`` and its five clock stamps) and its fifteen numbers (the
  trace, the verdict, the width, six weighting coefficients - the rate is the
  prototype's - and five features), packed little-endian;
* **the rest of the machine** - settings, clock, stimulation, the states,
  the counters, the prototype - as a small record, losslessly.

``precision="exact"`` packs the numbers as float64 and the integers at their
width: rebuilding gives back every field of every edge bit for bit, so the
loss is zero.  ``float32`` and ``float16`` pack the fifteen numbers smaller;
:func:`fidelity` measures what that costs in behaviour, and with a
``budget`` (bytes) :func:`compress` takes the least lossy precision that fits.

Because the code runs from the centre out, rebuilding can stop at any shell
(:meth:`Core.decompress`, ``shells=k``): the cells within ``k`` shells of the
central node come back exactly and the rest as the prototype.  The machine
can also be walked straight from the code, decoding only the cells the walk
reaches (:meth:`Core.probabilities`, :meth:`Core.run`), from the start state
or from the middle.

Low-rank codes were measured against this one (``DESIGN.md`` §11): a Tucker
decomposition of the cube, and a PCA of the touched edges' fields across
their channels.  At the loss where a rebuilt machine still behaves as the
original, both are larger than this exact code.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, fields

from .edge import Edge, Weighting
from .geometry import center, center_out, shell_sizes, shells
from .lattice import Lattice, State
from .machine import Machine, _softmax

__all__ = ["FLOATS", "INTS", "MAX_KL", "PRECISIONS", "Core", "compress", "fidelity", "load_core"]

INTS = ("seen", "first_seen", "last_seen", "last_rewarded", "last_punished", "width_stamp")
"""An edge's integers, in the order the code packs them."""

FLOATS = ("recent", "rewarded", "punished", "width", "w_bias", "w_seen", "w_recent", "w_net", "w_age", "w_width",
          "f_seen", "f_recent", "f_net", "f_age", "f_width")
"""An edge's numbers, in the order the code packs them."""

PRECISIONS = {"exact": ("d", 8), "float32": ("f", 4), "float16": ("e", 2)}
"""How the fifteen numbers are packed: struct code and bytes, least lossy first."""

MAX_KL = 1e-3
"""The largest KL divergence (nats) a row may move by for :func:`fidelity` to call the behaviour preserved."""

TIE = 1e-6
"""Two log-weights this close are a tie: a rebuilt greedy choice among the original's tied best is not a change."""

_EDGE_FIELDS = tuple(f.name for f in fields(Edge))
_HALF_MAX = 65504.0
_FLOAT_MAX = 3.4028234663852886e38
_FORMAT = "latticefsm-core"
_FORMAT_VERSION = 1


def _ints(e: Edge) -> list[int]:
    return [e.seen, e.first_seen, e.last_seen, e.last_rewarded, e.last_punished, e.width_stamp]


def _floats(e: Edge) -> list[float]:
    w = e.weighting
    return [e.recent, e.rewarded, e.punished, e.width, w.bias, w.seen, w.recent, w.net, w.age, w.width, *e.features]


def _clamp(x: float, precision: str) -> float:
    limit = {"exact": math.inf, "float32": _FLOAT_MAX, "float16": _HALF_MAX}[precision]
    return max(-limit, min(limit, x))


@dataclass
class Core:
    """The whole matrix, compressed into its central node."""

    shape: tuple[int, int, int]
    alphabet: list[str]
    precision: str
    int_width: int
    """4 or 8: the integers are packed as int32 when every one fits, else int64."""
    touched: list[bool]
    """One bit per cell, in centre-out order."""
    ints: bytes
    floats: bytes
    machine: dict
    """Everything that is not an edge: settings, clock, stimulation, states, counters, prototype."""

    def __post_init__(self) -> None:
        self.order = center_out(self.shape)
        self._position: dict[int, int] | None = None

    # ---- geometry and size -----------------------------------------------------------------------------------------

    @property
    def center(self) -> tuple[int, int, int]:
        return center(self.shape)

    @property
    def n_touched(self) -> int:
        return sum(self.touched)

    @property
    def bytes(self) -> int:
        """What the edges cost: the bitmap, the integers and the numbers.  The machine record is besides."""
        return (len(self.touched) + 7) // 8 + len(self.ints) + len(self.floats)

    def shell_table(self) -> list[dict]:
        """Per shell, from the central node out: cells, touched edges, and the bytes the code spends there."""
        sizes = shell_sizes(self.shape)
        fbytes = PRECISIONS[self.precision][1]
        out, i = [], 0
        for k, n in enumerate(sizes):
            hits = sum(self.touched[i:i + n])
            out.append({"shell": k, "cells": n, "touched": hits,
                        "bytes": (n + 7) // 8 + hits * (len(INTS) * self.int_width + len(FLOATS) * fbytes)})
            i += n
        return out

    def summary(self) -> dict:
        S, A, _ = self.shape
        cells = S * A * S
        dense = cells * (len(INTS) * 8 + len(FLOATS) * 8)
        c = self.center
        return {
            "shape": list(self.shape), "center": list(c), "center_label": [c[0], self.alphabet[c[1]], c[2]],
            "shells": shells(self.shape), "precision": self.precision, "touched": self.n_touched, "cells": cells,
            "bytes": self.bytes, "dense_bytes": dense, "ratio": dense / max(1, self.bytes),
            "record_bytes": len(json.dumps(self.machine, separators=(",", ":"))),
            "shell_table": self.shell_table(),
        }

    # ---- reading the code back -------------------------------------------------------------------------------------

    def _prototype(self) -> Weighting:
        return Weighting.from_list(self.machine["lattice_prototype"])

    def _decode(self, cell: int, k: int) -> Edge:
        """The edge at matrix offset ``cell``, the ``k``-th touched edge of the code."""
        S, A, _ = self.shape
        s, rest = divmod(cell, A * S)
        a, t = divmod(rest, S)
        ic = "<i" if self.int_width == 4 else "<q"
        fc, fb = PRECISIONS[self.precision]
        iv = struct.unpack_from("<" + ic[1] * len(INTS), self.ints, k * len(INTS) * self.int_width)
        fv = struct.unpack_from("<" + fc * len(FLOATS), self.floats, k * len(FLOATS) * fb)
        p = self._prototype()
        e = Edge(s, a, t, weighting=Weighting(fv[4], fv[5], fv[6], fv[7], fv[8], fv[9], p.rate))
        e.seen, e.first_seen, e.last_seen, e.last_rewarded, e.last_punished, e.width_stamp = iv
        e.recent, e.rewarded, e.punished, e.width = fv[0], fv[1], fv[2], fv[3]
        e.features = tuple(fv[10:15])
        return e

    def _positions(self) -> dict[int, int]:
        if self._position is None:
            self._position, k = {}, 0
            for cell, hit in zip(self.order, self.touched):
                if hit:
                    self._position[cell] = k
                    k += 1
        return self._position

    def edge(self, s: int, a: int, t: int) -> Edge:
        """One edge, decoded on its own; an untouched cell is the prototype's edge."""
        S, A, _ = self.shape
        cell = (s * A + a) * S + t
        k = self._positions().get(cell)
        if k is None:
            return Edge(s, a, t, weighting=self._prototype().copy())
        return self._decode(cell, k)

    def decompress(self, shells: int | None = None) -> Machine:
        """The machine rebuilt from the central node outward; with ``shells=k``, only the first ``k`` shells (the
        central node is shell 0) - every cell beyond them comes back as the prototype's edge."""
        S, A, _ = self.shape
        lattice = Lattice(S, self.alphabet, self._prototype())
        lattice.states = [State.from_list(v) for v in self.machine["states"]]
        limit = len(self.order)
        if shells is not None:
            limit = sum(shell_sizes(self.shape)[:max(0, shells)])
        k = 0
        for i, (cell, hit) in enumerate(zip(self.order, self.touched)):
            if i >= limit:
                break
            if hit:
                lattice.edges[cell] = self._decode(cell, k)
                k += 1
        return Machine._assemble(lattice, self.machine)

    def rebuild_into(self, lattice: Lattice) -> None:
        """Put the code's edges back into ``lattice`` (of the same shape), in place: every edge object keeps its
        identity and takes the code's fields - a touched one decoded, an untouched one the prototype's."""
        if lattice.shape != self.shape:
            raise ValueError(f"the code is {self.shape}, the matrix {lattice.shape}")
        proto = self._prototype()
        positions = self._positions()
        S, A, _ = self.shape
        for cell in self.order:
            k = positions.get(cell)
            if k is not None:
                new = self._decode(cell, k)
            else:
                s, rest = divmod(cell, A * S)
                new = Edge(s, rest // S, rest % S, weighting=proto.copy())
            old = lattice.edges[cell]
            for name in _EDGE_FIELDS:
                setattr(old, name, getattr(new, name))

    @property
    def stimulation(self) -> float:
        st = self.machine["settings"]
        value, stamp = self.machine["stimulation"]
        excess = value - st["baseline"]
        if excess == 0:
            return st["baseline"]
        return st["baseline"] + excess * 0.5 ** ((self.machine["clock"] - stamp) / st["calm"])

    def probabilities(self, s: int, symbol: int | str, stimulation: float | None = None,
                      temperature: float | None = None) -> list[float]:
        """A row's distribution, read straight from the code: only that row's cells are decoded."""
        a = self.alphabet.index(symbol) if isinstance(symbol, str) else symbol
        st = self.machine["settings"]
        clock, life = self.machine["clock"], st["life"]
        stim = self.stimulation if stimulation is None else stimulation
        logits = [self.edge(s, a, t).log_weight(clock, life, stim) for t in range(self.shape[2])]
        return _softmax(logits, st["temperature"] if temperature is None else temperature)

    def run(self, text: str, from_middle: bool = False) -> tuple[list[int], bool]:
        """The greedy walk of ``text``, read straight from the code, from the start state or (``from_middle``) the
        middle state: the states it passes through, and whether the last accepts."""
        state = self.center[0] if from_middle else self.machine["settings"]["start"]
        states = [state]
        accepting = {s[0] for s in self.machine["states"] if s[1]}
        for ch in text:
            if ch.isspace():
                continue
            probs = self.probabilities(state, ch, temperature=0.0)
            state = max(range(len(probs)), key=probs.__getitem__)
            states.append(state)
        return states, state in accepting

    # ---- persistence -----------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": _FORMAT, "version": _FORMAT_VERSION, "shape": list(self.shape), "alphabet": list(self.alphabet),
            "center": list(self.center), "order": "center-out", "ints_fields": list(INTS),
            "floats_fields": list(FLOATS), "precision": self.precision, "int_width": self.int_width,
            "touched": _bits_to_hex(self.touched), "ints": base64.b64encode(self.ints).decode("ascii"),
            "floats": base64.b64encode(self.floats).decode("ascii"), "machine": self.machine,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Core:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        if list(d.get("ints_fields", INTS)) != list(INTS) or list(d.get("floats_fields", FLOATS)) != list(FLOATS):
            raise ValueError("the core's fields are not this version's")
        if d.get("precision") not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        S, A, T = d["shape"]
        return cls((S, A, T), list(d["alphabet"]), d["precision"], int(d["int_width"]),
                   _hex_to_bits(d["touched"], S * A * T), base64.b64decode(d["ints"]), base64.b64decode(d["floats"]),
                   d["machine"])

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)


def load_core(path: str) -> Core:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return Core.from_dict(json.load(f))


def _bits_to_hex(bits: Sequence[bool]) -> str:
    out = []
    for i in range(0, len(bits), 4):
        nib = 0
        for j, b in enumerate(bits[i:i + 4]):
            nib |= int(bool(b)) << (3 - j)
        out.append("0123456789abcdef"[nib])
    return "".join(out)


def _hex_to_bits(text: str, n: int) -> list[bool]:
    bits = []
    for ch in text:
        nib = int(ch, 16)
        bits.extend(bool(nib >> (3 - j) & 1) for j in range(4))
    return bits[:n]


def _machine_record(m: Machine) -> dict:
    return {
        "settings": m.settings_dict(), "clock": m.clock, "state": m.state, "runs": m.runs, "credits": m.credits,
        "compressions": m.compressions, "since_compression": m.since_compression,
        "last_compressed": m.last_compressed, "stimulation": [m._stimulation, m._stimulation_stamp],
        "states": [s.to_list() for s in m.lattice.states], "lattice_prototype": m.lattice.prototype.to_list(),
    }


# ---- compressing ----------------------------------------------------------------------------------------------------

def _encode(machine: Machine, precision: str) -> Core:
    lat = machine.lattice
    order = center_out(lat.shape)
    touched = [lat.edges[c].touched for c in order]
    edges = [lat.edges[c] for c, hit in zip(order, touched) if hit]
    all_ints = [x for e in edges for x in _ints(e)]
    int_width = 4 if all(-2**31 <= x < 2**31 for x in all_ints) else 8
    ints = struct.pack("<" + ("i" if int_width == 4 else "q") * len(all_ints), *all_ints)
    fc = PRECISIONS[precision][0]
    floats = struct.pack("<" + fc * (len(edges) * len(FLOATS)),
                         *[_clamp(x, precision) for e in edges for x in _floats(e)])
    return Core(lat.shape, list(machine.alphabet), precision, int_width, touched, ints, floats,
                _machine_record(machine))


def compress(machine: Machine, precision: str | None = None, budget: int | None = None) -> Core:
    """The matrix, folded into its central node with the least loss.

    ``precision`` packs the numbers ``exact`` (the default: no loss at all), ``float32`` or ``float16``.  With a
    ``budget`` in bytes and no precision, the least lossy precision whose code fits; if none fits, ``float16``."""
    if precision is not None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)}, got {precision!r}")
        return _encode(machine, precision)
    if budget is None:
        return _encode(machine, "exact")
    core = None
    for name in PRECISIONS:
        core = _encode(machine, name)
        if core.bytes <= budget:
            return core
    return core


# ---- loss -----------------------------------------------------------------------------------------------------------

def fidelity(original: Machine, restored: Machine) -> dict:
    """How far ``restored`` is from ``original``.

    In behaviour: per row, the KL divergence of its distribution (at the original's stimulation; temperature 1
    when the original's is 0) and whether the greedy choice is still one of the original's tied best.  In state:
    the largest relative error over every number of every edge, and how many edges differ at all."""
    S, A = original.n_states, len(original.alphabet)
    stim = original.stimulation
    temp = original.temperature if original.temperature > 0 else 1.0
    kls, agree = [], 0
    for s in range(S):
        for a in range(A):
            lo = original.log_weights(s, a, stim)
            lr = restored.log_weights(s, a, stim)
            p, q = _softmax(lo, temp), _softmax(lr, temp)
            kls.append(max(0.0, sum(pi * (math.log(pi) - math.log(max(qi, 1e-300))) for pi, qi in zip(p, q) if pi > 0)))
            best = max(lo)
            choice = max(range(S), key=lambda t: (lr[t], -t))
            agree += lo[choice] >= best - TIE * (1.0 + abs(best))
    rows = S * A
    worst, differ = 0.0, 0
    for eo, er in zip(original.lattice.edges, restored.lattice.edges):
        if _ints(eo) != _ints(er) or _floats(eo) != _floats(er):
            differ += 1
        for x, y in zip(_floats(eo), _floats(er)):
            if x != y:
                worst = max(worst, abs(x - y) / max(abs(x), 1e-12))
    return {
        "rows": rows, "mean_kl": sum(kls) / rows, "max_kl": max(kls), "greedy_agreement": agree / rows,
        "greedy_changed": rows - agree, "preserved": agree == rows and max(kls) <= MAX_KL,
        "edges_differing": differ, "max_relative_error": worst, "lossless": differ == 0,
    }
