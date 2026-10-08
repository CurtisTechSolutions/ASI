"""The arithmetic of a primed tree: every sequence of ``0..L`` units over ``R`` ids has one address.

A complete ``R``-ary tree needs no pointers.  Sequences are numbered by length,
then lexicographically by unit id, the first unit most significant::

    base(l) = (R**l - 1) // (R - 1)      # sequences shorter than l == the id of the first of length l
    code(s) = sum(s[i] * R**(l - 1 - i))  # the path as a number in base R
    id(s)   = base(l) + code(s)           # 0 <= id < N,  N = base(L + 1)

``id`` is a bijection from the sequences of length ``0..L`` onto ``0..N-1``, so
the flat arrays of the two trees have no holes, and every move a walk makes -
append a unit, drop the oldest, drop the newest - is integer arithmetic on an
id (``DESIGN.md`` section 6).  ``check.py`` primes a small tree by literal
brute force and asserts that this arithmetic is the same tree.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence

__all__ = ["Address", "node_count"]


def node_count(R: int, L: int) -> int:
    """``N`` for a vocabulary of ``R`` ids and sequences up to length ``L``."""
    if R < 2 or L < 1:
        raise ValueError(f"a primed tree needs R >= 2 and L >= 1, got R={R}, L={L}")
    return (R ** (L + 1) - 1) // (R - 1)


class Address:
    """``(R, L)`` bound once; every method is integer arithmetic on ids."""

    __slots__ = ("R", "L", "N", "bases", "powers")

    def __init__(self, R: int, L: int) -> None:
        if R < 2 or L < 1:
            raise ValueError(f"a primed tree needs R >= 2 and L >= 1, got R={R}, L={L}")
        self.R = int(R)
        self.L = int(L)
        self.powers = [self.R ** l for l in range(self.L + 2)]
        self.bases = [(self.R ** l - 1) // (self.R - 1) for l in range(self.L + 2)]
        self.N = self.bases[self.L + 1]

    # -- sequences <-> ids -------------------------------------------------------

    def of(self, seq: Sequence[int]) -> int:
        """The id of a sequence of ``0..L`` unit ids."""
        l = len(seq)
        if l > self.L:
            raise ValueError(f"a sequence of {l} units is longer than L={self.L}")
        R = self.R
        c = 0
        for x in seq:
            if not 0 <= x < R:
                raise ValueError(f"unit id {x} outside 0..{R - 1}")
            c = c * R + x
        return self.bases[l] + c

    def seq(self, i: int) -> tuple[int, ...]:
        """The sequence an id spells."""
        l = self.level(i)
        k = i - self.bases[l]
        R = self.R
        out = []
        for _ in range(l):
            out.append(k % R)
            k //= R
        out.reverse()
        return tuple(out)

    def level(self, i: int) -> int:
        """The length of the sequence ``i`` spells (``0`` for the root)."""
        if not 0 <= i < self.N:
            raise ValueError(f"id {i} outside 0..{self.N - 1}")
        return bisect_right(self.bases, i) - 1

    # -- the moves ------------------------------------------------------------------

    def append(self, i: int, l: int, x: int) -> int:
        """The child ``s.x`` of the node ``i`` at level ``l`` - the step ``x`` after ``s``."""
        if l >= self.L:
            raise ValueError(f"level {l} has no children (L={self.L})")
        return self.bases[l + 1] + (i - self.bases[l]) * self.R + x

    def drop_oldest(self, i: int, l: int) -> int:
        """``s[1:]``: the sequence without its oldest unit - the shift, and the fall."""
        if l < 1:
            raise ValueError("the root has nothing to drop")
        return self.bases[l - 1] + (i - self.bases[l]) % self.powers[l - 1]

    def drop_newest(self, i: int, l: int) -> int:
        """``s[:-1]``: the tree's parent of ``i``."""
        if l < 1:
            raise ValueError("the root has no parent")
        return self.bases[l - 1] + (i - self.bases[l]) // self.R

    def newest(self, i: int, l: int) -> int:
        """The last unit of the sequence ``i`` spells."""
        if l < 1:
            raise ValueError("the root has no units")
        return (i - self.bases[l]) % self.R

    def oldest(self, i: int, l: int) -> int:
        """The first unit of the sequence ``i`` spells."""
        if l < 1:
            raise ValueError("the root has no units")
        return (i - self.bases[l]) // self.powers[l - 1]

    def block(self, i: int, l: int) -> tuple[int, int]:
        """``(start, stop)`` of the contiguous block of ``i``'s children: entry ``x`` is ``start + x``."""
        if l >= self.L:
            raise ValueError(f"level {l} has no children (L={self.L})")
        start = self.bases[l + 1] + (i - self.bases[l]) * self.R
        return start, start + self.R

    # -- the rolling code -----------------------------------------------------------

    def substrings(self, ids: Sequence[int]) -> Iterator[tuple[int, int, int]]:
        """``(position, level, id)`` for every substring of length ``1..L`` ending at every position.

        The reference for the hot loops of the two trees: the last ``l`` units
        ending at position ``t`` are the last ``l - 1`` units ending at ``t - 1``
        followed by ``u_t``, so the codes roll (section 6.3).  Positions count
        from 1.
        """
        R, L, bases = self.R, self.L, self.bases
        codes = [0] * (L + 1)
        for t, x in enumerate(ids, 1):
            if not 0 <= x < R:
                raise ValueError(f"unit id {x} outside 0..{R - 1}")
            m = L if t >= L else t
            for l in range(m, 0, -1):
                c = codes[l - 1] * R + x
                codes[l] = c
                yield t, l, bases[l] + c

    def __repr__(self) -> str:
        return f"Address(R={self.R}, L={self.L}, N={self.N})"
