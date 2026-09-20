"""The parametric sine, and its partial derivatives.

``f(x) = a * sin(b * (x - h)) + k``, at the defaults ``a = -1, b = 1/3,
h = k = 0`` exactly the author's ``-sin(x / 3)``.  All four parameters are
learnable wherever this function is used, which in this architecture is two
different places:

* every **filter unit** in layer 1 (:mod:`fbradix.filter`), where the sign of
  ``f`` is one bit of the routing address;
* every **node** of every expert tree (:mod:`fbradix.tree`), where ``f`` scales
  the edges into and out of that node.

Same formula and same partials as ``RadixCyclicNN/radixnet/activation.py`` and
``Experiments/SBNN_RNN_ActivationFunction/main.py``; this directory is
standalone (standard library only, no cross-package imports), so the reference
implementation is repeated rather than imported.

Why a *periodic* unit is load-bearing here and not merely inherited: the sign
of a monotone unit cuts its input axis in two, so ``m`` monotone units address
at most the cells of an arrangement of ``m`` hyperplanes.  The sign of a sine
unit is a **square wave** - it cuts the same axis into unboundedly many
alternating bands - so one unit can separate inputs that are not linearly
separable, and ``m`` of them can reach all ``2^m`` addresses along a single
projection.  ``DESIGN.md`` section 4 makes that argument properly, and
``fbradix.experiment``'s ``monotone`` arm is the control that tests it.
"""

from __future__ import annotations

import math

DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0

MIN_B = 1e-3
"""Lower bound applied to ``b`` after every update (radixnet's ``backend.MIN_B``)."""

DEAD_EPS = 0.01
"""``|a*b| <= DEAD_EPS`` means the unit is dead: ``|f'| = |a*b*cos(u)| <= |a*b|``.

The threshold, and the reason it is a statement about the ``(a, b)`` axis
rather than the input axis, are `Experiments/ActivationFunctionTest/` finding 4.
"""


def sine(x: float, a: float, b: float, h: float, k: float) -> float:
    """Return ``a * sin(b * (x - h)) + k``."""
    return a * math.sin(b * (x - h)) + k


def sine_partials(
    x: float, a: float, b: float, h: float, k: float
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


def tanh_partials(
    x: float, a: float, b: float, h: float, k: float
) -> tuple[float, float, float, float, float, float]:
    """The monotone control: ``g(x) = a * tanh(b * (x - h)) + k``, same knobs.

    Parameter-matched to :func:`sine_partials` down to the argument order, so
    the ``monotone`` arm of the experiment differs from the ``learned`` arm in
    the shape of one function and in nothing else.  With ``t = tanh(b(x-h))``
    and ``s = 1 - t^2``::

        dg/dx = a * b * s      dg/da = t      dg/db = a * (x-h) * s
        dg/dh = -a * b * s     dg/dk = 1
    """
    d = x - h
    t = math.tanh(b * d)
    s = 1.0 - t * t
    abs_ = a * b * s
    return (a * t + k, abs_, t, a * d * s, -abs_, 1.0)


def is_dead(a: float, b: float, eps: float = DEAD_EPS) -> bool:
    """True when ``|f'| <= eps`` at *every* input, i.e. the unit cannot move."""
    return abs(a * b) <= eps
