"""The count tree: how often every sequence of ``1..L`` units occurred, written by reading alone.

One ``array('q')`` of ``N`` counts, allocated once at priming and never
resized.  ``observe`` counts every substring of length ``1..L`` of a padded
text once per occurrence with the rolling code of ``DESIGN.md`` section 6.3 -
``L`` multiplications and ``L`` increments per unit, no lookup, no allocation.
It is the only thing that writes ``cnt``.  ``ctx``, ``share`` and ``own`` read
a context's children as one slice of the array.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence

from .address import Address, node_count
from .codec import Codec

__all__ = ["CountTree", "DEFAULT_ALPHA", "DEFAULT_SMOOTHING", "NODE_CEILING"]

DEFAULT_ALPHA = 2.0
"""``own(c) = ctx / (ctx + ALPHA)`` - FilterBankRadix's constant."""
DEFAULT_SMOOTHING = 0.5
"""Jeffreys: ``share = (count + s) / (ctx + s * R')``; 0 is the raw frequency."""
NODE_CEILING = 4_194_304
"""``prime`` refuses more nodes than this unless told otherwise (DESIGN.md section 16)."""


class CountTree:
    def __init__(self, codec: Codec, L: int, address: Address | None = None, alpha: float = DEFAULT_ALPHA,
                 smoothing: float = DEFAULT_SMOOTHING, node_ceiling: int | None = NODE_CEILING) -> None:
        self.codec = codec
        self.L = int(L)
        self.address = address if address is not None else Address(codec.R, self.L)
        N = self.address.N
        if node_ceiling is not None and N > node_ceiling:
            raise ValueError(
                f"R={codec.R} at L={self.L} is {N:,} nodes, above the ceiling of {node_ceiling:,}; "
                f"a smaller vocabulary, a smaller L, or --ceiling to raise it"
            )
        self.cnt = array("q", bytes(8 * N))
        self.alpha = float(alpha)
        self.smoothing = float(smoothing)
        self.texts = 0
        self.units = 0
        self.unk = 0
        self.version = 0
        self._emits = codec.emits()
        self._excluded = [x for x in range(codec.R) if x not in set(self._emits)]

    @property
    def N(self) -> int:
        return self.address.N

    # -- writing ----------------------------------------------------------------------

    def observe(self, ids: Sequence[int]) -> int:
        """Count every substring of length ``1..L`` of a padded text; returns the increments made."""
        R, L, bases = self.address.R, self.address.L, self.address.bases
        cnt = self.cnt
        codes = [0] * (L + 1)
        n = 0
        unk = self.codec.unk
        unks = 0
        for t, x in enumerate(ids, 1):
            if not 0 <= x < R:
                raise ValueError(f"unit id {x} outside 0..{R - 1}")
            if x == unk:
                unks += 1
            m = L if t >= L else t
            for l in range(m, 0, -1):
                c = codes[l - 1] * R + x
                codes[l] = c
                cnt[bases[l] + c] += 1
            n += m
        self.texts += 1
        self.units += max(0, len(ids) - 2)
        self.unk += unks
        self.version += 1
        return n

    def observe_text(self, text: str) -> int:
        return self.observe(self.codec.padded(text))

    # -- reading ------------------------------------------------------------------------

    def count(self, i: int) -> int:
        return self.cnt[i]

    def children(self, i: int, l: int) -> list[int]:
        """The counts of the steps out of node ``i``, in ``emits()`` order."""
        start, _ = self.address.block(i, l)
        cnt = self.cnt
        return [cnt[start + x] for x in self._emits]

    def ctx(self, i: int, l: int) -> int:
        """How often ``i`` was a context: the sum over its emittable children."""
        start, stop = self.address.block(i, l)
        total = sum(self.cnt[start:stop])
        cnt = self.cnt
        for x in self._excluded:
            total -= cnt[start + x]
        return total

    def share(self, i: int, l: int) -> list[float]:
        """``(count + s) / (ctx + s * R')`` per emittable child; all zero when nothing was read and ``s == 0``."""
        kids = self.children(i, l)
        s = self.smoothing
        total = sum(kids)
        denom = total + s * len(kids)
        if denom <= 0.0:
            return [0.0] * len(kids)
        return [(c + s) / denom for c in kids]

    def own(self, i: int, l: int) -> float:
        c = self.ctx(i, l)
        return c / (c + self.alpha) if c > 0 else 0.0

    def nonzero(self) -> Iterator[tuple[int, int]]:
        for i, c in enumerate(self.cnt):
            if c:
                yield i, c

    def memory_bytes(self) -> int:
        return 8 * self.N

    def __repr__(self) -> str:
        return f"CountTree(R={self.address.R}, L={self.L}, N={self.N:,}, texts={self.texts}, units={self.units})"


def nodes_for(codec: Codec, L: int) -> int:
    return node_count(codec.R, L)
