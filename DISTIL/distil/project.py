"""Projecting the embedding space onto a plane you can look at.

`GREN/DESIGN.md` 21 calls this "the plane of existence made literal, and what the
frontend renders". This is that, for the memory layer: 512 dimensions reduced to
two so a person can see which memories sit together.

Principal components rather than MDS. The two agree here -- classical MDS on a
Euclidean distance matrix *is* PCA -- and PCA costs one pass over the vectors
where MDS builds an n x n matrix first, which at ten thousand traces is a hundred
million entries to hold for a picture.

Power iteration, in the standard library, because the rest of this package is.
Deflation gives the second component. Both are deterministic: the starting vector
is fixed rather than random, so the same store always produces the same picture
and a moved point means the memory moved, not the renderer.

**What the axes mean: nothing.** They are the two directions of greatest variance
and they have no names. Distance between points is meaningful, direction is not,
and the sign of either axis is arbitrary -- which is why `orient` pins it instead
of leaving a picture that flips between reloads.
"""
from __future__ import annotations

import math

from .vector import Vector, dot, norm


def _centre(vectors: list[Vector]) -> tuple[list[Vector], Vector]:
    dims = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]
    return [[x - m for x, m in zip(v, mean)] for v in vectors], mean


def _top_component(rows: list[Vector], iterations: int = 64) -> Vector:
    """Leading eigenvector of the covariance, by power iteration.

    Never forms the covariance matrix: `X^T X v` is two passes over the rows, so
    the cost is O(n*d) per iteration rather than O(d^2) in memory. At 512 dims
    that is the difference between a few thousand floats and a quarter of a
    million.
    """
    dims = len(rows[0])
    # A fixed, non-degenerate start. Random would make the picture move between
    # runs for no reason, and a uniform vector can be exactly orthogonal to the
    # component being sought.
    v = [math.sin(i * 0.7 + 1.0) for i in range(dims)]
    scale = norm(v)
    v = [x / scale for x in v]
    for _ in range(iterations):
        acc = [0.0] * dims
        for row in rows:
            weight = dot(row, v)
            if weight:
                for i, x in enumerate(row):
                    acc[i] += weight * x
        length = norm(acc)
        if length < 1e-12:
            return v                 # no variance left in this direction
        nxt = [x / length for x in acc]
        if sum(abs(a - b) for a, b in zip(nxt, v)) < 1e-9:
            return nxt               # converged
        v = nxt
    return v


def _deflate(rows: list[Vector], component: Vector) -> list[Vector]:
    return [[x - dot(row, component) * c for x, c in zip(row, component)] for row in rows]


def _orthogonalise(v: Vector, against: Vector) -> Vector:
    """Strip out the part of `v` that lies along `against`.

    Deflation leaves rows with no variance in the first direction, so when there
    is no second direction at all -- every point on one line -- power iteration
    hits its "no variance left" branch and hands back its own starting vector,
    which is not orthogonal to anything. Projecting onto that put the x
    coordinate back on the y axis and turned a straight line into a diagonal.

    Returning zeros when nothing survives is the honest answer: there is no
    second axis here, so every point sits at y = 0 and the picture is the line
    the data actually is.
    """
    overlap = dot(v, against)
    out = [x - overlap * a for x, a in zip(v, against)]
    length = norm(out)
    if length < 1e-9:
        return [0.0] * len(v)
    return [x / length for x in out]


def project(vectors: list[Vector]) -> list[tuple[float, float]]:
    """The two leading principal components, as (x, y) in [-1, 1].

    Fewer than three vectors have no plane to project onto, so they are laid out
    rather than decomposed -- a straight line of points is the honest picture of
    "not enough data to have a shape".
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        return [(0.0, 0.0)]
    if len(vectors) == 2:
        return [(-0.6, 0.0), (0.6, 0.0)]

    rows, _ = _centre(vectors)
    first = _top_component(rows)
    second = _orthogonalise(_top_component(_deflate(rows, first)), first)
    points = [(dot(row, first), dot(row, second)) for row in rows]
    return _fit(orient(points))


def orient(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pin the arbitrary sign of each axis.

    An eigenvector is only defined up to sign, so the same data can come back
    mirrored on either axis between runs. Fixing the sign by the skew of the
    projection makes the picture stable, which matters because a person reading
    it will remember where things were.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    flip_x = -1.0 if sum(x ** 3 for x in xs) < 0 else 1.0
    flip_y = -1.0 if sum(y ** 3 for y in ys) < 0 else 1.0
    return [(x * flip_x, y * flip_y) for x, y in points]


def _fit(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Scale into [-1, 1] on both axes, preserving aspect.

    Scaling each axis independently would stretch a tight cluster to fill the
    frame and make two points look far apart because nothing else varies. One
    scale keeps distances comparable, which is the only thing the picture is
    entitled to claim.
    """
    span = max((max(abs(x), abs(y)) for x, y in points), default=0.0)
    if span < 1e-12:
        return [(0.0, 0.0) for _ in points]
    return [(round(x / span, 5), round(y / span, 5)) for x, y in points]
