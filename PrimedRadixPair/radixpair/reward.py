"""The reward tree: what every step earned, written by outcomes alone - and a punishment is a negative reward.

Two ``array('d')`` of ``N``: ``plus``, the rewards a step received, and
``minus``, the penalties, both ``>= 0`` and kept apart.  ``credit`` is the
count tree's loop with a float and a sign: a positive amount goes to ``plus``,
a negative one to ``minus``, at every context length of every step
(``rungs="all"``) or at the deepest only (``rungs="final"``).  What a step is
worth is its net, ``reward(i) = plus - minus``; ``penalty(i) = minus`` is what
the punishment traversal reads, and the reason the two are not one number
(``RadixCyclicNN/SPEC-LeastPunished.md`` section 1).  ``credit`` and
``invert`` are the only things that write either array.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence

from .address import Address
from .codec import Codec

__all__ = ["RewardTree", "RUNGS"]

RUNGS = ("all", "final")


class RewardTree:
    def __init__(self, codec: Codec, L: int, address: Address | None = None) -> None:
        self.codec = codec
        self.L = int(L)
        self.address = address if address is not None else Address(codec.R, self.L)
        N = self.address.N
        self.plus = array("d", bytes(8 * N))
        self.minus = array("d", bytes(8 * N))
        self.judged = 0
        self.rewards_total = 0.0
        self.penalties_total = 0.0
        self.version = 0
        self._emits = codec.emits()

    @property
    def N(self) -> int:
        return self.address.N

    # -- writing ----------------------------------------------------------------------

    def credit(self, ids: Sequence[int], amount: float, rungs: str = "all", skip: int = 0) -> int:
        """Credit a padded text by a signed amount at every level (or the deepest); returns the entries written.

        ``skip`` positions at the start are context only: their substrings roll
        the codes but are not credited - what the model was *given* is not
        what it *produced*.
        """
        if rungs not in RUNGS:
            raise ValueError(f"rungs must be one of {RUNGS}, got {rungs!r}")
        amount = float(amount)
        if amount == 0.0:
            return 0
        R, L, bases = self.address.R, self.address.L, self.address.bases
        target = self.plus if amount > 0 else self.minus
        a = abs(amount)
        codes = [0] * (L + 1)
        n = 0
        skip = max(0, int(skip))
        if rungs == "all":
            for t, x in enumerate(ids, 1):
                if not 0 <= x < R:
                    raise ValueError(f"unit id {x} outside 0..{R - 1}")
                m = L if t >= L else t
                if t <= skip:
                    for l in range(m, 0, -1):
                        codes[l] = codes[l - 1] * R + x
                    continue
                for l in range(m, 0, -1):
                    c = codes[l - 1] * R + x
                    codes[l] = c
                    target[bases[l] + c] += a
                n += m
        else:
            base = bases[L]
            for t, x in enumerate(ids, 1):
                if not 0 <= x < R:
                    raise ValueError(f"unit id {x} outside 0..{R - 1}")
                m = L if t >= L else t
                for l in range(m, 0, -1):
                    c = codes[l - 1] * R + x
                    codes[l] = c
                if t >= L and t > skip:
                    target[base + codes[L]] += a
                    n += 1
        if amount > 0:
            self.rewards_total += a * n
        else:
            self.penalties_total += a * n
        self.judged += 1
        self.version += 1
        return n

    def invert(self) -> None:
        """Every reward becomes a penalty of the same size and every penalty a reward; an involution."""
        self.plus, self.minus = self.minus, self.plus
        self.rewards_total, self.penalties_total = self.penalties_total, self.rewards_total
        self.version += 1

    # -- reading ------------------------------------------------------------------------

    def reward(self, i: int) -> float:
        return self.plus[i] - self.minus[i]

    def penalty(self, i: int) -> float:
        return self.minus[i]

    def rewards(self, i: int, l: int) -> list[float]:
        """The net reward of every step out of node ``i``, in ``emits()`` order."""
        start, _ = self.address.block(i, l)
        plus, minus = self.plus, self.minus
        return [plus[start + x] - minus[start + x] for x in self._emits]

    def penalties(self, i: int, l: int) -> list[float]:
        start, _ = self.address.block(i, l)
        minus = self.minus
        return [minus[start + x] for x in self._emits]

    def nonzero(self) -> Iterator[tuple[int, float, float]]:
        plus, minus = self.plus, self.minus
        for i in range(self.N):
            p, m = plus[i], minus[i]
            if p or m:
                yield i, p, m

    def memory_bytes(self) -> int:
        return 16 * self.N

    def __repr__(self) -> str:
        return (f"RewardTree(R={self.address.R}, L={self.L}, N={self.N:,}, judged={self.judged}, "
                f"rewards={self.rewards_total:g}, penalties={self.penalties_total:g})")
