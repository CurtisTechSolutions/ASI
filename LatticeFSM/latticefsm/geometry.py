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
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["center", "center_out", "shell", "shell_sizes", "shells"]


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
