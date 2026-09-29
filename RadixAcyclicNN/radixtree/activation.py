"""The parametric sine, and its partial derivatives.

Every node of the tree owns an activation ``f(x) = a * sin(b * (x - h)) + k``;
at the defaults ``a = -1, b = 1/3, h = k = 0`` it is exactly the author's
``-sin(x / 3)``.  All four parameters are learnable per node, which is what
"update the activation function itself" means in this family of models.

Same formula, same partials and same defaults as
``RadixCyclicNN/radixnet/activation.py``: this directory is standalone
(standard library only, no cross-package imports), so the reference
implementation is repeated here rather than imported, the way
``FilterBankRadix`` repeats it.
"""

from __future__ import annotations

import math

__all__ = [
    "DEFAULT_A", "DEFAULT_B", "DEFAULT_H", "DEFAULT_K", "MIN_B",
    "SineActivation", "edge_signal", "sine_activation", "sine_derivative", "sine_partials",
]

DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0

MIN_B = 1e-3
"""Lower bound applied to every node's ``b`` after each update (``radixnet.backend.MIN_B``)."""


def sine_activation(x: float, a: float = DEFAULT_A, b: float = DEFAULT_B, h: float = DEFAULT_H, k: float = DEFAULT_K) -> float:
    """Return ``a * sin(b * (x - h)) + k`` (default: ``-sin(x / 3)``)."""
    return a * math.sin(b * (x - h)) + k


def sine_derivative(x: float, a: float = DEFAULT_A, b: float = DEFAULT_B, h: float = DEFAULT_H, k: float = DEFAULT_K) -> float:
    """Return ``df/dx = a * b * cos(b * (x - h))``."""
    return a * b * math.cos(b * (x - h))


def sine_partials(
    x: float, a: float = DEFAULT_A, b: float = DEFAULT_B, h: float = DEFAULT_H, k: float = DEFAULT_K
) -> tuple[float, float, float, float, float, float]:
    """Return ``(f, df/dx, df/da, df/db, df/dh, df/dk)`` at ``x``.

    With ``u = b * (x - h)``::

        f     = a * sin(u) + k
        df/dx = a * b * cos(u)
        df/da = sin(u)
        df/db = a * (x - h) * cos(u)
        df/dh = -a * b * cos(u)
        df/dk = 1
    """
    d = x - h
    u = b * d
    su = math.sin(u)
    cu = math.cos(u)
    abc = a * b * cu
    return (a * su + k, abc, su, a * d * cu, -abc, 1.0)


class SineActivation:
    """Convenience object wrapping one parameter set (not used in hot loops)."""

    __slots__ = ("a", "b", "h", "k")

    def __init__(self, a: float = DEFAULT_A, b: float = DEFAULT_B, h: float = DEFAULT_H, k: float = DEFAULT_K) -> None:
        self.a = float(a)
        self.b = float(b)
        self.h = float(h)
        self.k = float(k)

    def __call__(self, x: float) -> float:
        return sine_activation(x, self.a, self.b, self.h, self.k)

    def derivative(self, x: float) -> float:
        return sine_derivative(x, self.a, self.b, self.h, self.k)

    def partials(self, x: float) -> tuple[float, float, float, float, float, float]:
        return sine_partials(x, self.a, self.b, self.h, self.k)

    def inverted(self) -> "SineActivation":
        """The negation of this activation: ``a -> -a`` and ``k -> -k``.

        Negating the amplitude alone leaves ``-a * sin(u) + k = -f(x) + 2k``,
        the negation only while ``k`` is 0.  The offset is part of the unit's
        output and is negated with it, so ``inverted().inverted()`` is the
        identity and ``inverted()(x) == -self(x)`` exactly.
        """
        return SineActivation(-self.a, self.b, self.h, -self.k)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "h": self.h, "k": self.k}

    @classmethod
    def from_dict(cls, d: dict) -> "SineActivation":
        return cls(d.get("a", DEFAULT_A), d.get("b", DEFAULT_B), d.get("h", DEFAULT_H), d.get("k", DEFAULT_K))

    def __repr__(self) -> str:
        return f"SineActivation(a={self.a!r}, b={self.b!r}, h={self.h!r}, k={self.k!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SineActivation):
            return NotImplemented
        return (self.a, self.b, self.h, self.k) == (other.a, other.b, other.h, other.k)

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.h, self.k))


def edge_signal(w: float, fp: float, fc: float) -> float:
    """Signal carried by edge ``p -> c``: weight times ``f_p`` times ``f_c``."""
    return w * fp * fc
