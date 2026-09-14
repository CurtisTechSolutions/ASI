"""Cyclic counters - the odometer every growing integer in the model runs on.

A number that only ever counts up - traversals, node and edge visit counts,
epochs, trained characters, version stamps - eventually leaves the range of
whatever holds it: 64 bits in the Go port, and, long before that, the 53-bit
mantissa of the JSON number that carries it through a model file, the HTTP API
and the frontend.  So no counter in the model is an unbounded integer.  Each
one is a **two-digit odometer** in base :data:`COUNTER_LIMIT`::

    total = resets * COUNTER_LIMIT + value        with 0 <= value < COUNTER_LIMIT

``value`` counts up as before; the moment it reaches the limit it is **set back
to 0** and ``resets`` - the number of times that happened - goes up by one.
Nothing is lost: ``total`` is the exact number of events, ``value`` is what the
odometer reads now and ``resets`` is how often it went round.  Cycles are a
feature here too.

:data:`COUNTER_LIMIT` is ``10 ** 15`` because

* it is exactly representable as a ``float64`` (``< 2 ** 53``), so a counter
  survives a model file, a JSON API response and a JavaScript ``Number``
  unchanged - which an unbounded integer does not;
* it leaves four orders of magnitude of head room under a 64-bit integer
  (``~9.2 * 10 ** 18``), so a whole epoch of increments can land on a counter
  before the next :func:`carry` without ever overflowing the Go port's
  ``int64``;
* it is a round decimal: one quadrillion events per turn of the odometer.

``resets`` wraps at the same limit, so the pair itself cycles after ``10 ** 30``
events - the odometer's full turn, unreachable by several orders of magnitude
at any rate matter can produce.

Counting stays cheap because wrapping is **not** done per increment.  Hot loops
add to a raw integer and a :func:`carry` sweep at a safe point (the end of an
epoch, before a save) moves whatever crossed the limit into the resets - see
:meth:`radixnet.graph.RadixCyclicGraph.carry_counters`.
"""

from __future__ import annotations

__all__ = ["COUNTER_LIMIT", "CyclicCounter", "as_float", "carry", "carry_series", "total"]

COUNTER_LIMIT = 1_000_000_000_000_000
"""Every counter wraps back to 0 here (``10 ** 15``) and counts the wrap as a reset."""


def carry(value: int, resets: int = 0) -> tuple[int, int]:
    """Normalise a raw ``(value, resets)`` pair into ``0 <= value < COUNTER_LIMIT``.

    The whole turns of the odometer move from ``value`` into ``resets``, which
    wraps at the same limit::

        carry(COUNTER_LIMIT + 7, 2) == (7, 3)
    """
    turns, value = divmod(int(value), COUNTER_LIMIT)
    return value, (int(resets) + turns) % COUNTER_LIMIT


def total(value: int, resets: int = 0) -> int:
    """The exact number of events an odometer reading stands for."""
    return int(resets) * COUNTER_LIMIT + int(value)


def as_float(value: int, resets: int = 0) -> float:
    """:func:`total` as a float, in the one order the Go port also uses (identical bits)."""
    return float(resets) * COUNTER_LIMIT + float(value)


def carry_series(values: list[int], resets: dict[int, int]) -> int:
    """Wrap every entry of a parallel counter list in place; returns how many wrapped.

    ``values`` is the odometer reading per node / edge id, ``resets`` holds the
    reset counts of the ids that ever wrapped (absent means 0), so the common
    case - nothing has come near the limit - costs one comparison per entry and
    stores nothing.
    """
    wrapped = 0
    for i, v in enumerate(values):
        if v >= COUNTER_LIMIT or v < 0:
            values[i], r = carry(v, resets.get(i, 0))
            if r:
                resets[i] = r
            else:
                resets.pop(i, None)
            wrapped += 1
    return wrapped


class CyclicCounter:
    """One scalar odometer: ``value`` in ``[0, COUNTER_LIMIT)`` plus the ``resets``.

    Immutable, so ``counter += 1`` rebinds instead of mutating a reading some
    cache is holding on to.  It compares and hashes like the integer it stands
    for, which keeps ``version`` stamps and ``!=`` cache checks exact across a
    reset - the point of keeping the resets at all.
    """

    __slots__ = ("value", "resets")

    def __init__(self, value: int = 0, resets: int = 0) -> None:
        v, r = carry(value, resets)
        object.__setattr__(self, "value", v)
        object.__setattr__(self, "resets", r)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; use += to get a new reading")

    # -- readings ------------------------------------------------------------

    @property
    def total(self) -> int:
        """The exact number of events counted: ``resets * COUNTER_LIMIT + value``."""
        return total(self.value, self.resets)

    def as_float(self) -> float:
        """:attr:`total` as a float (the Go port computes it the same way)."""
        return as_float(self.value, self.resets)

    def bumped(self, n: int = 1) -> "CyclicCounter":
        """A new reading ``n`` events later, wrapping as often as it takes."""
        return CyclicCounter(self.value + int(n), self.resets)

    # -- (im)mutation --------------------------------------------------------

    def __add__(self, n: int) -> "CyclicCounter":
        return self.bumped(n)

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: object) -> int:
        return self.total - (other.total if isinstance(other, CyclicCounter) else int(other))  # type: ignore[arg-type]

    # -- comparison, as the integer it stands for ----------------------------

    @staticmethod
    def _other(other: object) -> int | None:
        if isinstance(other, CyclicCounter):
            return other.total
        if isinstance(other, int):
            return other
        return None

    def __eq__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total == o

    def __ne__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total != o

    def __lt__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total < o

    def __le__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total <= o

    def __gt__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total > o

    def __ge__(self, other: object) -> bool:
        o = self._other(other)
        return NotImplemented if o is None else self.total >= o

    def __hash__(self) -> int:
        return hash(self.total)

    def __bool__(self) -> bool:
        return bool(self.value or self.resets)

    def __int__(self) -> int:
        return self.total

    def __index__(self) -> int:
        return self.total

    def __float__(self) -> float:
        return self.as_float()

    # immutable, so a copy of a reading may as well be the reading itself
    def __copy__(self) -> "CyclicCounter":
        return self

    def __deepcopy__(self, memo: dict) -> "CyclicCounter":
        return self

    def __reduce__(self) -> tuple:
        return (type(self), (self.value, self.resets))

    def __repr__(self) -> str:
        return f"CyclicCounter(value={self.value}, resets={self.resets})"

    def __str__(self) -> str:
        return str(self.value) if not self.resets else f"{self.value} (+{self.resets} resets)"

    # -- serialisation -------------------------------------------------------

    def to_pair(self) -> tuple[int, int]:
        """``(value, resets)`` - the two JSON numbers a model file stores."""
        return self.value, self.resets

    @classmethod
    def from_pair(cls, value: object, resets: object = 0) -> "CyclicCounter":
        """Read a counter back from a model file; a missing ``resets`` means it never wrapped."""
        return cls(int(value or 0), int(resets or 0))
