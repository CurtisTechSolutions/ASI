"""Finite-difference check of the one learning rule in the architecture.

The numbers this directory reports are only worth reading if the derivatives
are right, so the one-hop rule is checked against central differences before
any run that asks for it (``python3 -m radixtree check``), the way
``FilterBankRadix`` and the experiments in ``Experiments/`` check theirs.

The analytic gradient is read out of :meth:`RadixTreeNet.step` itself - one
step at a tiny rate, with ``act_lr = lr`` so every parameter moves at the same
rate and a clip too wide to bite, and the parameter delta divided back out - so
what is checked is the code that trains, not a second implementation of the
same formula.  :func:`check_gradients` does the same for
:meth:`RadixTreeNet.gradients_of`, the unclipped reading the tests use.
"""

from __future__ import annotations

from .model import RadixTreeNet

__all__ = ["EPS", "TEXTS", "check_all", "check_gradients", "check_step"]

EPS = 1e-6

TEXTS = (
    "the cat sat on the mat", "the dog sat on the log",
    "the cat ran to the mat", "a dog and a cat",
)


def _batch(model: RadixTreeNet, texts, n: int) -> list[tuple[int, int]]:
    """The first ``n`` branching transitions of ``texts``, in the tree as it stands."""
    transitions, _ = model._observe(list(texts), count=False)
    out = [(p, c) for p, c in transitions if model.tree.num_children(p) > 1]
    return out[:n]


def _fresh(texts, depth, seed) -> RadixTreeNet:
    model = RadixTreeNet(seed=seed, depth=depth)
    model.train(texts, epochs=0)
    return model


def check_step(texts=TEXTS, depth: int | None = None, seed: int = 1, n: int = 40) -> float:
    """Max ``|analytic - numeric|`` over every parameter one :meth:`RadixTreeNet.step` moved."""
    ref = _fresh(texts, depth, seed)
    batch = _batch(ref, texts, n)
    if not batch:
        return 0.0
    moved = _fresh(texts, depth, seed)
    before = {name: list(getattr(moved.tree, name)) for name in ("w", "z", "a", "b", "h", "k")}
    lr = 1e-3
    moved.step(batch, lr, lr, clip=1e9)
    worst = 0.0
    for name, old in before.items():
        new = getattr(moved.tree, name)
        arr = getattr(ref.tree, name)
        for i, (o, nv) in enumerate(zip(old, new)):
            analytic = (o - nv) / lr
            if abs(analytic) < 1e-12:
                continue
            keep = arr[i]
            arr[i] = keep + EPS
            ref.tree.version += 1
            up = ref.loss_of(batch)
            arr[i] = keep - EPS
            ref.tree.version += 1
            down = ref.loss_of(batch)
            arr[i] = keep
            ref.tree.version += 1
            worst = max(worst, abs((up - down) / (2 * EPS) - analytic))
    return worst


def check_gradients(texts=TEXTS, depth: int | None = None, seed: int = 1, n: int = 40) -> float:
    """Max ``|analytic - numeric|`` over every gradient :meth:`RadixTreeNet.gradients_of` reports."""
    model = _fresh(texts, depth, seed)
    batch = _batch(model, texts, n)
    if not batch:
        return 0.0
    grads = model.gradients_of(batch)
    worst = 0.0
    for name in ("w", "z", "a", "b", "h", "k"):
        arr = getattr(model.tree, name)
        for i, analytic in grads[name].items():
            keep = arr[i]
            arr[i] = keep + EPS
            model.tree.version += 1
            up = model.loss_of(batch)
            arr[i] = keep - EPS
            model.tree.version += 1
            down = model.loss_of(batch)
            arr[i] = keep
            model.tree.version += 1
            worst = max(worst, abs((up - down) / (2 * EPS) - analytic))
    return worst


def check_all(verbose: bool = True) -> dict:
    """The rule through :meth:`step` and through :meth:`gradients_of`, unbounded and at a bounded depth."""
    out = {
        "step_every_suffix": check_step(),
        "step_depth_4": check_step(depth=4),
        "gradients_every_suffix": check_gradients(),
        "gradients_depth_4": check_gradients(depth=4),
    }
    if verbose:
        for name, err in out.items():
            print(f"  gradient check  {name:24s} max error {err:.2e}")
    return out
