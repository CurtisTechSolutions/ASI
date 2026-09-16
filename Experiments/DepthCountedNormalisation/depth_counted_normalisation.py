"""Depth-counted normalisation: is the vanishing gradient a measurement?

Standard library only - no numpy, no torch - so it runs anywhere with nothing
installed, the same rule RadixCyclicNN follows.

    python3 depth_counted_normalisation.py            # full run (~8 min)
    python3 depth_counted_normalisation.py --quick     # ~1 min


The claim under test
--------------------
Back-propagation multiplies.  A gradient reaching a layer `n` products away from
the loss has been through `n` of them, and each scales it by roughly the same
factor `c`, so it arrives at about `c**n` of the size it began with.

Nothing is *lost* there.  `c**n` is invertible.  The magnitude was not destroyed,
it was **encoded** - and the key to the code is `n`, which is free: it is the
depth, known from the architecture before any data is seen.  The vanishing
gradient looks like information loss only because we throw the key away.

So: count the multiplications, divide the decay back out, and see what happens.


What is measured, in order
--------------------------
1. **The premise.**  Does the magnitude really decay like `c**n`?  Fit
   `log||g_n|| = a + b*n` across the layers; report `c = exp(b)` and R^2.  A high
   R^2 means `n` is a *sufficient statistic* for the decay: knowing the depth
   tells you the size, so the decay is fully predictable and fully removable.

2. **The correction.**  Rescale each layer's update by `1 / c**(alpha * n)`:

       alpha = 0  ->  divide by 1      ->  plain SGD, decay left in
       alpha = 1  ->  divide by c**n   ->  decay entirely undone

   The count is what makes that dial exist.  Without `n` there is no dial, only
   the two ends.

3. **Depth.**  Where does the correction help, and where does it stop helping?


Two controls, both of which can sink the claim
----------------------------------------------
* `normalised` divides each layer's gradient by its own measured norm.  That
  equalises the layers too, and it uses **no count at all**.  If it matches the
  best `alpha`, the count bought nothing.

* **Total step size is held constant.**  After any per-layer reallocation the
  whole gradient is rescaled so its global norm matches the uncorrected one.
  Without this, "counted" is just a larger learning rate for deep layers and the
  comparison measures step size rather than the idea.  Every arm therefore takes
  the same size of step and differs *only in how it spreads that step across
  layers*.

`adam` is the reference for what the field does instead: a per-parameter running
RMS, which is an *empirical* estimate of the same scale the count gives
*structurally*, for free, at step zero, with no state and no warm-up.
"""

from __future__ import annotations

import argparse
import math
import random

SIGMOID_MAX_SLOPE = 0.25
"""max |d/dx sigmoid(x)|, at x = 0.  The a-priori value of `c` when the linear
maps are scaled to be roughly norm-preserving: each backward hop multiplies by
W^T (about 1) and by f' (at most this)."""


def sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

class DeepMLP:
    """`depth` hidden sigmoid layers of `width` units, then one linear unit."""

    def __init__(self, n_in: int, width: int, depth: int, seed: int) -> None:
        rng = random.Random(seed)
        self.sizes = [n_in] + [width] * depth + [1]
        self.W: list[list[list[float]]] = []
        self.b: list[list[float]] = []
        for i in range(len(self.sizes) - 1):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            # Xavier: the linear part stays roughly norm-preserving, so the
            # sigmoid derivative is the contraction that is left over.
            s = math.sqrt(6.0 / (fan_in + fan_out))
            self.W.append([[rng.uniform(-s, s) for _ in range(fan_in)] for _ in range(fan_out)])
            self.b.append([0.0] * fan_out)

    @property
    def n_layers(self) -> int:
        return len(self.W)

    def depth_of(self, layer: int) -> int:
        """Multiplications between this layer's weights and the loss.

        The last layer is 0 hops away, each layer below is one product further.
        This is the `n` of `c**n`, and it is known from the architecture alone.
        """
        return self.n_layers - 1 - layer

    def forward(self, x):
        acts, zs, a = [x], [], x
        last = self.n_layers - 1
        for i, (Wi, bi) in enumerate(zip(self.W, self.b)):
            z = [sum(w * v for w, v in zip(row, a)) + bias for row, bias in zip(Wi, bi)]
            zs.append(z)
            a = z if i == last else [sigmoid(v) for v in z]  # linear head
            acts.append(a)
        return acts, zs

    def grads(self, xs, ys):
        """Full-batch MSE gradients.  Returns (dW, db, loss)."""
        dW = [[[0.0] * len(row) for row in Wi] for Wi in self.W]
        db = [[0.0] * len(bi) for bi in self.b]
        loss = 0.0
        for x, y in zip(xs, ys):
            acts, zs = self.forward(x)
            pred = acts[-1][0]
            loss += (pred - y) ** 2
            delta = [2.0 * (pred - y)]
            for layer in range(self.n_layers - 1, -1, -1):
                a_prev = acts[layer]
                for j, d in enumerate(delta):
                    db[layer][j] += d
                    row = dW[layer][j]
                    for k, av in enumerate(a_prev):
                        row[k] += d * av
                if layer == 0:
                    break
                z_below = zs[layer - 1]
                Wl = self.W[layer]
                new = []
                for k in range(len(a_prev)):
                    acc = 0.0
                    for j, d in enumerate(delta):
                        acc += Wl[j][k] * d
                    s = sigmoid(z_below[k])
                    new.append(acc * s * (1.0 - s))
                delta = new
        n = float(len(xs))
        return ([[[v / n for v in r] for r in Wi] for Wi in dW],
                [[v / n for v in bi] for bi in db], loss / n)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def frob(M) -> float:
    return math.sqrt(sum(v * v for row in M for v in row))


def fit_line(xs, ys):
    """Least squares `y = a + b x`.  Returns (a, b, R^2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1.0 - ss_res / ss_tot if ss_tot else 1.0)


def estimate_c(net, dW) -> float:
    """Fit one scalar `c` to the current per-layer gradient norms.

    ONE number for the whole network, not one per layer - that is what keeps
    this distinct from `normalised`, which uses every layer's own measured norm.
    """
    pts = [(net.depth_of(l), frob(dW[l])) for l in range(net.n_layers)]
    pts = [(d, m) for d, m in pts if m > 0.0]
    if len(pts) < 2:
        return 1.0
    _, b, _ = fit_line([float(d) for d, _ in pts], [math.log(m) for _, m in pts])
    return math.exp(b)


def make_task(n: int, seed: int):
    """A smooth two-input regression target.  Needs a nonlinearity, nothing more.

    Predicting the mean scores about 0.18, so that is the do-nothing baseline.
    """
    rng = random.Random(seed + 9001)
    xs, ys = [], []
    for _ in range(n):
        x0, x1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        xs.append([x0, x1])
        ys.append(0.5 * math.sin(math.pi * x0) + 0.5 * x1 * x1)
    return xs, ys


# --------------------------------------------------------------------------
# the update rules
# --------------------------------------------------------------------------

def reallocate(net, dW, db, rule, alpha, c_fixed):
    """Spread the step across layers, holding the TOTAL step size constant.

    Returns (dW, db) rescaled.  Every rule here changes only the *distribution*
    of the step over layers; the global norm is restored afterwards so no arm
    gets a bigger step than another.
    """
    L = net.n_layers
    norms = [frob(dW[l]) for l in range(L)]

    if rule == "plain":
        f = [1.0] * L
    elif rule == "counted":              # c re-fitted from the current gradients
        c = estimate_c(net, dW)
        f = [1.0 / max(1e-30, c ** (alpha * net.depth_of(l))) for l in range(L)]
    elif rule == "counted-fixed":        # c measured once, at initialisation
        f = [1.0 / max(1e-30, c_fixed ** (alpha * net.depth_of(l))) for l in range(L)]
    elif rule == "normalised":           # the control: no count, measured norms
        f = [1.0 / (norms[l] + 1e-12) for l in range(L)]
    else:
        raise ValueError(rule)

    total_before = math.sqrt(sum(n * n for n in norms))
    total_after = math.sqrt(sum((norms[l] * f[l]) ** 2 for l in range(L)))
    g = (total_before / total_after) if total_after > 0 else 1.0
    return ([[[v * f[l] * g for v in r] for r in dW[l]] for l in range(L)],
            [[v * f[l] * g for v in db[l]] for l in range(L)])


def train(net, xs, ys, *, rule, alpha, c_fixed, lr, steps):
    """One run.  Returns (final MSE, how far the bottom layer moved from init)."""
    W0 = [[list(r) for r in Wi] for Wi in net.W]
    m = [[[0.0] * len(r) for r in Wi] for Wi in net.W]
    v = [[[0.0] * len(r) for r in Wi] for Wi in net.W]
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(1, steps + 1):
        dW, db, loss = net.grads(xs, ys)
        if not math.isfinite(loss):
            return float("inf"), 0.0

        if rule == "adam":
            sW, sb = [], db
            for l in range(net.n_layers):
                out = []
                for j, row in enumerate(dW[l]):
                    o = []
                    for k, g in enumerate(row):
                        m[l][j][k] = b1 * m[l][j][k] + (1 - b1) * g
                        v[l][j][k] = b2 * v[l][j][k] + (1 - b2) * g * g
                        o.append((m[l][j][k] / (1 - b1 ** step)) /
                                 (math.sqrt(v[l][j][k] / (1 - b2 ** step)) + eps))
                    out.append(o)
                sW.append(out)
        else:
            sW, sb = reallocate(net, dW, db, rule, alpha, c_fixed)

        for l in range(net.n_layers):
            for j, row in enumerate(sW[l]):
                for k, g in enumerate(row):
                    net.W[l][j][k] -= lr * g
            for j, g in enumerate(sb[l]):
                net.b[l][j] -= lr * g

    _, _, loss = net.grads(xs, ys)
    moved = frob([[a - b for a, b in zip(ra, rb)] for ra, rb in zip(net.W[0], W0[0])])
    return loss, moved


def run_arm(label, *, rule, alpha, lr, steps, width, depth, seeds):
    losses, moves = [], []
    for seed in seeds:
        net = DeepMLP(2, width, depth, seed)
        xs, ys = make_task(64, seed)
        c_fixed = estimate_c(net, net.grads(xs, ys)[0])
        l, mv = train(net, xs, ys, rule=rule, alpha=alpha, c_fixed=c_fixed, lr=lr, steps=steps)
        losses.append(l)
        moves.append(mv)
    ok = [l for l in losses if math.isfinite(l)]
    if not ok:
        return label, float("inf"), 0.0, 0.0, len(losses)
    mean = sum(ok) / len(ok)
    sd = (sum((l - mean) ** 2 for l in ok) / len(ok)) ** 0.5
    return label, mean, sd, sum(moves) / len(moves), len(losses) - len(ok)


def table(rows, head="arm"):
    print(f"  {head:<34}{'final MSE':>12}{'std':>10}{'bottom moved':>15}{'diverged':>10}")
    for label, mean, sd, mv, bad in rows:
        shown = "inf" if not math.isfinite(mean) else f"{mean:.6f}"
        print(f"  {label:<34}{shown:>12}{sd:>10.6f}{mv:>15.3e}{bad:>10}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="depth-counted normalisation")
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--depth", type=int, default=4, help="depth for the alpha sweep")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.seeds = 400, 2
    seeds = list(range(a.seeds))
    BASELINE = 0.1796  # MSE of predicting the mean

    print(f"deep sigmoid MLP, width {a.width}, linear head, full-batch MSE")
    print(f"{a.steps} steps, lr {a.lr}, {a.seeds} seeds")
    print(f"predicting the mean scores {BASELINE} - anything near it has not learned\n")

    print("=" * 79)
    print("PART 1 - the premise: does the gradient decay like c**n?")
    print("=" * 79)
    print(f"  {'depth':>7}{'c (fitted)':>14}{'R^2':>10}{'total decay':>16}")
    for d in (2, 4, 6, 8, 12):
        per = {}
        for s in seeds:
            net = DeepMLP(2, a.width, d, s)
            xs, ys = make_task(64, s)
            dW, _, _ = net.grads(xs, ys)
            for l in range(net.n_layers):
                per.setdefault(net.depth_of(l), []).append(frob(dW[l]))
        ds = sorted(per)
        mean = [sum(per[k]) / len(per[k]) for k in ds]
        pts = [(k, m) for k, m in zip(ds, mean) if m > 0]
        _, b, r2 = fit_line([float(k) for k, _ in pts], [math.log(m) for _, m in pts])
        print(f"  {d:>7}{math.exp(b):>14.4f}{r2:>10.4f}{mean[-1] / mean[0]:>16.3e}")
    print(f"\n  a-priori value for sigmoid (max |f'|) = {SIGMOID_MAX_SLOPE}")
    print("  High R^2 means the count n predicts the decay: n is all you need.")

    print()
    print("=" * 79)
    print(f"PART 2 - the correction at depth {a.depth}: update /= c**(alpha * n)")
    print("=" * 79)
    print("  Total step size is identical in every arm; only its spread differs.\n")
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        tag = f"counted  alpha={alpha:.2f}" + ("  (= plain SGD)" if alpha == 0 else "")
        rows.append(run_arm(tag, rule="counted", alpha=alpha, lr=a.lr, steps=a.steps,
                            width=a.width, depth=a.depth, seeds=seeds))
    rows.append(run_arm("counted-fixed  alpha=1.00", rule="counted-fixed", alpha=1.0,
                        lr=a.lr, steps=a.steps, width=a.width, depth=a.depth, seeds=seeds))
    rows.append(run_arm("normalised  (NO count)", rule="normalised", alpha=0.0, lr=a.lr,
                        steps=a.steps, width=a.width, depth=a.depth, seeds=seeds))
    rows.append(run_arm("adam", rule="adam", alpha=0.0, lr=0.01, steps=a.steps,
                        width=a.width, depth=a.depth, seeds=seeds))
    table(rows)

    print()
    print("=" * 79)
    print("PART 3 - across depth: where does counting help?")
    print("=" * 79)
    print(f"  {'depth':>6}{'plain':>12}{'counted a=1':>14}{'normalised':>13}{'adam':>12}   verdict")
    for d in (2, 3, 4, 5, 6):
        got = {}
        for tag, rule, alpha, lr in (("plain", "counted", 0.0, a.lr),
                                     ("counted", "counted", 1.0, a.lr),
                                     ("norm", "normalised", 0.0, a.lr),
                                     ("adam", "adam", 0.0, 0.01)):
            got[tag] = run_arm(tag, rule=rule, alpha=alpha, lr=lr, steps=a.steps,
                               width=a.width, depth=d, seeds=seeds)[1]
        # a difference smaller than this is a tie, not a result
        margin = 0.95
        sgd_family = [got["plain"], got["counted"], got["norm"]]
        if min(sgd_family) > BASELINE * 0.8:
            verdict = "SGD family all stall"
        elif max(sgd_family) - min(sgd_family) < 0.01:
            verdict = "tie"
        elif got["counted"] < margin * min(got["plain"], got["norm"]):
            verdict = "counting wins"
        elif got["counted"] < margin * got["plain"]:
            verdict = "helps, control better"
        else:
            verdict = "no help"
        print(f"  {d:>6}{got['plain']:>12.4f}{got['counted']:>14.4f}"
              f"{got['norm']:>13.4f}{got['adam']:>12.4f}   {verdict}")

    print()
    print("=" * 79)
    print("PART 4 - verdict")
    print("=" * 79)
    counted = [r for r in rows if r[0].startswith("counted ")]
    best = min(counted, key=lambda r: r[1])
    plain = next(r for r in counted if "plain SGD" in r[0])
    norm = next(r for r in rows if r[0].startswith("normalised"))
    fixed = next(r for r in rows if r[0].startswith("counted-fixed"))
    print(f"  best counted arm      : {best[0].strip()}  ->  {best[1]:.6f}")
    print(f"  plain SGD             : {plain[1]:.6f}")
    print(f"  normalised (no count) : {norm[1]:.6f}")
    print(f"  counted, c fixed once : {fixed[1]:.6f}   (sensitivity to a stale c)")
    print()
    print(f"  counting beats plain SGD            : {best[1] < plain[1]}")
    print(f"  counting beats the count-free control: {best[1] < norm[1]}")
    print(f"  re-fitting c matters                : {best[1] < fixed[1]}")


if __name__ == "__main__":
    main()
