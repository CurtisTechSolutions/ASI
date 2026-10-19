"""The matrix as a cube: its central node, its shells, and the order that visits it from the middle outward.

A matrix of ``S`` states over ``A`` symbols is a ``S x A x S`` block of cells.
Its **central node** is the cell ``(S // 2, A // 2, S // 2)`` - ``(6, g, 6)``
on the default 13 x 13 x 13 machine, the one cell with as many cells on every
side as on the other.  A cell's **shell** is its Chebyshev distance from the
centre, ``max(|s - 6|, |a - 6|, |t - 6|)``: shell 0 is the central node, shell
1 the 26 cells around it, and so on out to shell 6, the 866 cells of the
cube's surface.

:func:`center_out` lists every cell from the middle outward - shell by shell,
in ``(s, a, t)`` order within a shell - and is the order the compressed code
is laid out in and rebuilt in (``compress.py``).

**The central vertical vector** is the column of cells through the central
node along the source axis - the axis the matrix is drawn with top to
bottom: ``(s, A // 2, S // 2)`` for ``s`` from 0 (the top) to ``S - 1`` (the
bottom), ``(0..12, g, 6)`` on the default machine.  **Focus** is a number in
``[0, 1]`` (:data:`FOCUS_RANGE`) that picks one node of that vector, and so
the state a walk starts from: the range is cut into ``S`` equal bands, the
lowest band (and no focus at all) the top node, the highest the bottom
node, ``0.5`` the central node (:func:`focus_index`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "FOCUS_RANGE", "center", "center_out", "central_vertical", "check_focus", "focus_index", "focus_node", "shell",
    "shell_sizes", "shells",
]

FOCUS_RANGE = (0.0, 1.0)
"""The finite range focus lives in: 0 is none, 1 is full."""


def center(shape: Sequence[int]) -> tuple[int, int, int]:
    """The central node of a ``(S, A, S)`` matrix."""
    S, A, T = shape
    return (S // 2, A // 2, T // 2)


def shell(shape: Sequence[int], s: int, a: int, t: int) -> int:
    """How many shells out from the central node the cell ``(s, a, t)`` lies."""
    cs, ca, ct = center(shape)
    return max(abs(s - cs), abs(a - ca), abs(t - ct))


def shells(shape: Sequence[int]) -> int:
    """How many shells the matrix has, the central node's included: 7 for 13 x 13 x 13."""
    S, A, T = shape
    cs, ca, ct = center(shape)
    return 1 + max(cs, S - 1 - cs, ca, A - 1 - ca, ct, T - 1 - ct)


def center_out(shape: Sequence[int]) -> list[int]:
    """Every cell's matrix offset (``(s * A + a) * S + t``), from the central node outward."""
    S, A, T = shape
    cells = [(shell(shape, s, a, t), s, a, t) for s in range(S) for a in range(A) for t in range(T)]
    cells.sort()
    return [(s * A + a) * T + t for _, s, a, t in cells]


def shell_sizes(shape: Sequence[int]) -> list[int]:
    """How many cells each shell holds, from the centre out: ``[1, 26, 98, 218, 386, 602, 866]`` for 13 x 13 x 13."""
    S, A, T = shape
    sizes = [0] * shells(shape)
    for s in range(S):
        for a in range(A):
            for t in range(T):
                sizes[shell(shape, s, a, t)] += 1
    return sizes


def central_vertical(shape: Sequence[int]) -> list[tuple[int, int, int]]:
    """The central vertical vector, top to bottom: ``(s, A // 2, S // 2)`` for every source state ``s``."""
    S, A, T = shape
    _, ca, ct = center(shape)
    return [(s, ca, ct) for s in range(S)]


def check_focus(focus: float | None) -> float | None:
    """``focus`` if it is ``None`` or a finite number in :data:`FOCUS_RANGE`; anything else is refused."""
    if focus is None:
        return None
    if isinstance(focus, bool) or not isinstance(focus, (int, float)):
        raise ValueError(f"focus must be a number in {list(FOCUS_RANGE)} or None, got {focus!r}")
    lo, hi = FOCUS_RANGE
    if not math.isfinite(focus) or not lo <= focus <= hi:
        raise ValueError(f"focus must be a finite number in {list(FOCUS_RANGE)}, got {focus!r}")
    return float(focus)


def focus_index(n: int, focus: float | None) -> int:
    """Which of the ``n`` nodes of the central vertical vector ``focus`` picks, 0 the top: the range cut into ``n``
    equal bands, ``floor(focus * n)`` and the top of the range in the last band.  No focus is the top node."""
    focus = check_focus(focus)
    if focus is None:
        return 0
    lo, hi = FOCUS_RANGE
    return min(n - 1, int(math.floor((focus - lo) / (hi - lo) * n)))


def focus_node(shape: Sequence[int], focus: float | None) -> tuple[int, int, int]:
    """The node of the central vertical vector that ``focus`` picks: the walk starts from its source state."""
    return central_vertical(shape)[focus_index(shape[0], focus)]
