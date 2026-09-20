"""Finite-difference checks of every gradient in the architecture.

The numbers this repository reports are only worth reading if the derivatives
are right, so both learning rules are checked against central differences and
the worst error is printed before any run that asks for it
(``python3 -m fbradix.cli check``), the same rule
``Experiments/SBNN_RNN_ActivationFunction/main.py`` and
``Experiments/ActivationFunctionTest/self_building_sinewave.py`` follow.

There are exactly two rules to check:

* :func:`check_tree` - the one-hop softmax over a node's children, in all six
  parameters it touches (``w`` on the edges, and ``z, a, b, h, k`` on the
  parent and every child);
* :func:`check_filter` - the routing hinge, in the projection ``v`` and the
  four activation parameters of the unit being pushed;
* :func:`check_pair` - the argmax hinge, in both units it touches at once.
"""

from __future__ import annotations

import math

from .filter import ActivationFilter, features
from .tree import RadixTreeNet

EPS = 1e-6


def _batch_loss(tree: RadixTreeNet, batch) -> float:
    """The mean one-hop loss of ``batch`` at the tree's current parameters."""
    total = 0.0
    for p, target in batch:
        kids = list(tree.kids[p].values())
        if len(kids) < 2:
            continue
        fp = tree.f(p)
        scores = [tree.w[c] * fp * tree.f(c) for c in kids]
        m = max(scores)
        total += m + math.log(sum(math.exp(s - m) for s in scores)) - scores[kids.index(target)]
    return total / len(batch)


def check_tree(texts=None, depth: int = 4, seed: int = 1, n: int = 24) -> float:
    """Max ``|analytic - numeric|`` over every parameter the one-hop rule touches.

    The analytic gradient is read out of :meth:`RadixTreeNet.step` itself - one
    step at a tiny rate, with ``act_lr = lr`` so every parameter moves at the
    same rate, and the parameter delta divided back out - so what is checked is
    the code that trains, not a second implementation of the same formula.
    """
    texts = texts or [
        "the cat sat on the mat", "the dog sat on the log",
        "the cat ran to the mat", "a dog and a cat",
    ]
    ref = RadixTreeNet(depth=depth, alphabet=64, seed=seed)
    for t in texts:
        ref.insert_text(t)
    batch = ref.plan(texts)[:n]
    if not batch:
        return 0.0
    moved = RadixTreeNet(depth=depth, alphabet=64, seed=seed)
    for t in texts:
        moved.insert_text(t)
    before = {p: list(getattr(moved, p)) for p in ("w", "z", "a", "b", "h", "k")}
    lr = 1e-3
    moved.step(batch, lr, lr)
    worst = 0.0
    for name, old in before.items():
        new = getattr(moved, name)
        for i, (o, nv) in enumerate(zip(old, new)):
            analytic = (o - nv) / lr
            if abs(analytic) < 1e-12:
                continue
            arr = getattr(ref, name)
            keep = arr[i]
            arr[i] = keep + EPS
            up = _batch_loss(ref, batch)
            arr[i] = keep - EPS
            down = _batch_loss(ref, batch)
            arr[i] = keep
            worst = max(worst, abs((up - down) / (2 * EPS) - analytic))
    return worst


def check_filter(kind: str = "sine", seed: int = 0, bits: int = 3) -> float:
    """Max ``|analytic - numeric|`` over the routing hinge's five parameters."""
    filt = ActivationFilter(bits=bits, seed=seed, kind=kind)
    texts = ["def main(): return 42", "# a paragraph of ordinary prose, written out",
             '{"a": 1, "b": [2, 3]}', "func Foo(x int) error { return nil }"]
    worst = 0.0
    for text in texts:
        x = features(text)
        for j in range(bits):
            for y in (1.0, -1.0):
                g = filt.hinge_grads(x, j, y)
                if g is None:
                    continue
                for i in range(filt.dim):
                    keep = filt.v[j][i]
                    filt.v[j][i] = keep + EPS
                    up = filt.hinge_loss(x, j, y)
                    filt.v[j][i] = keep - EPS
                    down = filt.hinge_loss(x, j, y)
                    filt.v[j][i] = keep
                    worst = max(worst, abs((up - down) / (2 * EPS) - g["v"][i]))
                for name in ("a", "b", "h", "k"):
                    arr = getattr(filt, name)
                    keep = arr[j]
                    arr[j] = keep + EPS
                    up = filt.hinge_loss(x, j, y)
                    arr[j] = keep - EPS
                    down = filt.hinge_loss(x, j, y)
                    arr[j] = keep
                    worst = max(worst, abs((up - down) / (2 * EPS) - g[name]))
    return worst


def check_pair(kind: str = "sine", seed: int = 0, bits: int = 4) -> float:
    """Max ``|analytic - numeric|`` over the argmax hinge, for both units."""
    filt = ActivationFilter(bits=bits, seed=seed, kind=kind, code="argmax")
    texts = ["def main(): return 42", "# ordinary prose, written out at length",
             '{"a": 1, "b": [2, 3]}', "func Foo(x int) error { return nil }"]
    worst = 0.0
    for text in texts:
        x = features(text)
        for up in range(bits):
            for down in range(bits):
                if up == down:
                    continue
                g = filt.pair_grads(x, up, down)
                if g is None:
                    continue
                for side in ("up", "down"):
                    j = g[side]["unit"]
                    for i in range(filt.dim):
                        keep = filt.v[j][i]
                        filt.v[j][i] = keep + EPS
                        hi = filt.pair_loss(x, up, down)
                        filt.v[j][i] = keep - EPS
                        lo = filt.pair_loss(x, up, down)
                        filt.v[j][i] = keep
                        worst = max(worst, abs((hi - lo) / (2 * EPS) - g[side]["v"][i]))
                    for name in ("a", "b", "h", "k"):
                        arr = getattr(filt, name)
                        keep = arr[j]
                        arr[j] = keep + EPS
                        hi = filt.pair_loss(x, up, down)
                        arr[j] = keep - EPS
                        lo = filt.pair_loss(x, up, down)
                        arr[j] = keep
                        worst = max(worst, abs((hi - lo) / (2 * EPS) - g[side][name]))
    return worst


def check_all(verbose: bool = True) -> dict:
    """Both rules, both filter shapes; returns the worst error of each."""
    out = {
        "tree_one_hop": check_tree(),
        "filter_hinge_sine": check_filter("sine"),
        "filter_hinge_monotone": check_filter("monotone"),
        "filter_argmax_hinge": check_pair("sine"),
    }
    if verbose:
        for name, err in out.items():
            print(f"  gradient check  {name:22s} max error {err:.2e}")
    return out
