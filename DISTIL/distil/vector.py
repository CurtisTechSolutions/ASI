"""Vectors, in the standard library.

No numpy. Every other directory in this repository runs on a bare interpreter
and this one does too -- a memory layer that needs a C toolchain to recall a
fact is not a memory layer, it is a deployment problem.

Vectors are plain `list[float]`, always L2-normalised on the way in, so cosine
similarity collapses to a dot product and recall costs one multiply-add per
dimension per trace. At 512 dims and 10k traces that is 5M flops -- tens of
milliseconds in pure Python, which is under the latency of the API call that
produced the query in the first place. The exact-search ceiling is ~100k traces;
past that the answer is an index, not a faster loop (DESIGN 4.5).
"""
from __future__ import annotations

import math

Vector = list[float]


def zeros(dims: int) -> Vector:
    return [0.0] * dims


def dot(a: Vector, b: Vector) -> float:
    # zip stops at the shorter one; mismatched dims are a bug upstream, so say so
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def normalise(a: Vector) -> Vector:
    n = norm(a)
    if n == 0.0:
        return list(a)          # the zero vector has no direction; leave it alone
    return [x / n for x in a]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine of pre-normalised vectors. Falls back to the full formula when
    either side is not unit length, because silently returning a dot product
    for an unnormalised vector is how similarity scores stop being comparable."""
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    if abs(na - 1.0) < 1e-9 and abs(nb - 1.0) < 1e-9:
        return dot(a, b)
    return dot(a, b) / (na * nb)


def add_scaled(target: Vector, other: Vector, scale: float) -> Vector:
    return [t + scale * o for t, o in zip(target, other)]


def centroid(vectors: list[Vector]) -> Vector:
    """The mean direction of a set. Used for cluster summaries, where the
    interesting quantity is where a group of memories points, not where any one
    of them does."""
    if not vectors:
        return []
    dims = len(vectors[0])
    acc = zeros(dims)
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return normalise([x / len(vectors) for x in acc])


def spread(vectors: list[Vector]) -> float:
    """Mean cosine distance from the centroid: 0 for a tight cluster, ->1 for a
    diffuse one. This is the signal `explore.py` mines for gaps -- a region of
    memory with high spread and low grade is a subject the system has opinions
    about but no competence in, which is exactly where an experiment pays."""
    if len(vectors) < 2:
        return 0.0
    c = centroid(vectors)
    return sum(1.0 - cosine(c, v) for v in vectors) / len(vectors)
