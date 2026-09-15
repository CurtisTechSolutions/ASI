"""Correctness proofs for the sine network and the 2NRL inversion.

Run with ``python3 test_invert.py``.  Covers the analytic gradients of the
learnable activation, the exactness of :meth:`SineNet.invert`, and the
parity finding: the *literal* "flip everything" rule is a no-op on a
feed-forward stack.
"""

from __future__ import annotations

import numpy as np

from sinenet import ACT_PARAMS, SineNet

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def net(seed: int = 0, sizes=(4, 24, 2)) -> SineNet:
    return SineNet(list(sizes), np.random.default_rng(seed))


def randomise_act(n: SineNet, seed: int = 7) -> None:
    """Move a, b, h, k off their defaults - h and k in particular, so the
    activation is no longer odd and the inversion cannot rely on it."""
    rng = np.random.default_rng(seed)
    for layer in n.layers:
        layer.a += rng.normal(0, 0.3, layer.a.shape)
        layer.b += rng.normal(0, 0.05, layer.b.shape)
        layer.h += rng.normal(0, 0.4, layer.h.shape)
        layer.k += rng.normal(0, 0.3, layer.k.shape)
        layer.bias += rng.normal(0, 0.2, layer.bias.shape)


def test_gradients() -> None:
    """Analytic gradients vs central finite differences."""
    print("\ngradients (analytic vs finite difference)")
    n = net(1)
    randomise_act(n, 3)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 4))
    target = rng.normal(size=(8, 2))

    def loss_of(model: SineNet) -> float:
        return float(((model.forward(x, train=False) - target) ** 2).sum())

    out = n.forward(x)
    grads = n.backward(2.0 * (out - target))

    eps, worst, worst_name = 1e-6, 0.0, ""
    for li, layer in enumerate(n.layers):
        for name in layer.params:
            arr = getattr(layer, name)
            flat = arr.reshape(-1)
            for idx in range(0, flat.size, max(1, flat.size // 4)):
                original = flat[idx]
                flat[idx] = original + eps
                hi = loss_of(n)
                flat[idx] = original - eps
                lo = loss_of(n)
                flat[idx] = original
                numeric = (hi - lo) / (2 * eps)
                analytic = grads[li][name].reshape(-1)[idx]
                err = abs(numeric - analytic) / max(1.0, abs(numeric))
                if err > worst:
                    worst, worst_name = err, f"layer{li}.{name}[{idx}]"
    check("every analytic gradient matches finite differences",
          worst < 1e-5, f"max rel. err {worst:.2e} at {worst_name}")


def test_invert_exact() -> None:
    """invert() negates the output exactly, with defaults and with h, k != 0."""
    print("\ninversion is an exact negation")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 4))

    n = net(11)
    before = n.forward(x, train=False)
    n.invert()
    after = n.forward(x, train=False)
    err = np.abs(after + before).max()
    check("defaults (h = k = 0): out_inverted == -out", err < 1e-12, f"max |err| {err:.2e}")

    n = net(12)
    randomise_act(n, 9)
    before = n.forward(x, train=False)
    n.invert()
    after = n.forward(x, train=False)
    err = np.abs(after + before).max()
    check("trained a, b, h, k all non-zero: out_inverted == -out",
          err < 1e-12, f"max |err| {err:.2e}")

    for break_at in range(2):
        n = net(13)
        randomise_act(n, 4)
        before = n.forward(x, train=False)
        n.invert(parity_break=break_at)
        err = np.abs(n.forward(x, train=False) + before).max()
        check(f"parity break on layer {break_at} negates the output",
              err < 1e-12, f"max |err| {err:.2e}")

    n = net(14)
    randomise_act(n, 6)
    before = n.forward(x, train=False)
    n.invert()
    n.invert()
    err = np.abs(n.forward(x, train=False) - before).max()
    check("inverting twice is the identity", err < 1e-12, f"max |err| {err:.2e}")


def test_literal_is_noop() -> None:
    """The finding: flipping *everything* cancels itself on a feed-forward stack."""
    print("\nthe parity finding: the literal flip is a no-op")
    rng = np.random.default_rng(21)
    x = rng.normal(size=(64, 4))

    n = net(15)                      # h = k = 0, the research defaults
    before = n.forward(x, train=False)
    n.invert_literal()
    err = np.abs(n.forward(x, train=False) - before).max()
    check("flip every W, bias and amplitude -> network unchanged (odd activation)",
          err < 1e-12, f"max |err| {err:.2e}")

    for depth, sizes in (("3 layers", (4, 16, 16, 2)), ("4 layers", (4, 16, 16, 16, 2))):
        n = net(16, sizes)
        before = n.forward(x, train=False)
        n.invert_literal()
        err = np.abs(n.forward(x, train=False) - before).max()
        check(f"still a no-op at {depth}", err < 1e-12, f"max |err| {err:.2e}")

    n = net(17)
    before = n.forward(x, train=False)
    n.invert_literal()
    after = n.forward(x, train=False)
    same = (after.argmax(1) == before.argmax(1)).mean()
    check("literal flip leaves every action choice unchanged", same == 1.0,
          f"{same:.0%} identical argmax")


def test_decision_flip() -> None:
    """What the inversion is for: every preference reverses."""
    print("\nevery win becomes a loss")
    rng = np.random.default_rng(31)
    x = rng.normal(size=(2000, 4))

    n = net(18)
    randomise_act(n, 2)
    before = n.forward(x, train=False)
    n.invert()
    after = n.forward(x, train=False)
    flipped = (after.argmax(1) != before.argmax(1)).mean()
    check("argmax becomes argmin on every state", flipped == 1.0,
          f"{flipped:.0%} of 2000 states reversed")

    # Softmax policy over two actions: the probabilities swap exactly.
    def policy(v, tau=0.2):
        logits = v / tau
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    err = np.abs(policy(after) - policy(before)[:, ::-1]).max()
    check("the softmax policy's two action probabilities swap",
          err < 1e-12, f"max |err| {err:.2e}")


def test_hidden_preserved() -> None:
    """The inversion re-reads the learned features, it does not destroy them."""
    print("\nwhat the negative phase learned survives")
    rng = np.random.default_rng(41)
    x = rng.normal(size=(32, 4))

    n = net(19)
    randomise_act(n, 8)
    hidden_before = n.layers[0].forward(x, train=False).copy()
    n.invert()                                   # parity break on the last layer
    hidden_after = n.layers[0].forward(x, train=False)
    err = np.abs(hidden_after - hidden_before).max()
    check("hidden layer output is bit-for-bit identical after inverting",
          err < 1e-12, f"max |err| {err:.2e}")

    n = net(20)
    moved = {p: False for p in ACT_PARAMS}
    randomise_act(n, 5)
    snapshot = {p: [getattr(l, p).copy() for l in n.layers] for p in ACT_PARAMS}
    n.invert()
    for p in ACT_PARAMS:
        moved[p] = any(not np.allclose(getattr(l, p), snapshot[p][i])
                       for i, l in enumerate(n.layers))
    check("the inversion does touch the activation parameters",
          moved["a"] and moved["h"] and moved["k"],
          f"changed: {[p for p in ACT_PARAMS if moved[p]]}")


if __name__ == "__main__":
    print("2NRL sine network - correctness proofs")
    test_gradients()
    test_invert_exact()
    test_literal_is_noop()
    test_decision_flip()
    test_hidden_preserved()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)
    print("all proofs hold")
