"""Does reading Adam's moments as two interfering sine waves buy anything?

Standard library only - no numpy, no torch - so it runs anywhere with nothing
installed, the same rule the rest of `Experiments/` follows.

    python3 test_two_sine_optimizer.py            # full run (~1 min 40 s)
    python3 test_two_sine_optimizer.py --quick    # ~10 s
    python3 test_two_sine_optimizer.py --check    # algebra + gradient checks only


What is under test
------------------
`two_sine_optimizer.py` keeps three EMAs instead of Adam's two and combines the
two resulting signal-to-noise ratios as two sine waves rather than as one
fraction. The sum of the waves is a beat - a carrier times an envelope - and the
envelope closes when the fast and slow timescales disagree, which is the state of
a parameter bouncing across a ravine instead of descending it.

That predicts something specific and falsifiable: **the brake should widen the
band of learning rates that still converge**, because the failure it damps is
exactly the oscillation that a too-large learning rate causes. So the thing to
measure is not "best loss" - it is the whole learning-rate grid.

Two statistics do that, and part 4 reports both:

* **the band count** (part 2) - how many learning rates in a fixed grid reach a
  fixed absolute target. Easy to read, and no arm can game an absolute bar. But
  it is a threshold over seed-averages, and thresholds are brittle: an arm that
  lands just the wrong side of one loses a whole point. It moves around with the
  seed set, so it is the secondary number.
* **the paired comparison** (part 4C) - every arm runs on the same seeds, which
  fix the initial weights and the batch order, so each (problem, lr, seed) cell
  is a matched pair. Counting wins within pairs removes the between-seed variance
  the band count leaves in, and a sign test says whether the split is thin. This
  is the number to read first.


Four controls, every one of which can sink it
---------------------------------------------
Each holds the state and the timescales fixed and changes only the combining
rule, so anything that separates them is the rule and not the extra memory.

* `two-ema-linear` - the same three EMAs, averaged without a sine anywhere.
  If it ties, the second EMA won and the waves did nothing.
* `one-sine` - one wave, on the fast ratio only. If it ties, the second wave
  did nothing.
* `sign-gate` - the fast ratio, hard-zeroed when the moments disagree in sign.
  If it ties, the smooth envelope did nothing an `if` could not do.
* `slow-adam` - Adam with `beta1 = 0.99`. Rules out "it is just a longer memory".

`adam` is the reference. Every arm is swept over the same learning-rate grid and
reported at its own best, because TwoSine's step is `pi/2` times Adam's near zero
and a shared learning rate would measure that constant instead of the idea.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_sine_optimizer import RULES, TwoSine, build  # noqa: E402

ARMS = ("two-sine", "adam", "two-ema-linear", "one-sine", "sign-gate",
        "slow-adam", "two-sine-full")
assert set(ARMS) == set(RULES), "every rule the optimiser offers must be an arm here"

LR_GRID = (3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)


# ---------------------------------------------------------------------------
# problems with a known answer
# ---------------------------------------------------------------------------

class Quadratic:
    """`f(x) = 0.5 * sum(lambda_i * x_i^2)`, minimum 0 at the origin.

    The eigenvalues are log-spaced over `cond`, so the stiff directions want a
    small step and the flat ones want a large one. That conflict is the whole
    reason adaptive optimisers exist, and the stiff directions are where a
    too-large learning rate turns into sign-flipping oscillation - the exact
    failure the envelope is supposed to catch.
    """

    name = "ill-conditioned quadratic"
    optimum = 0.0

    def __init__(self, n: int = 20, cond: float = 1000.0) -> None:
        self.n = n
        self.lam = [cond ** (i / (n - 1)) for i in range(n)]

    def start(self, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.uniform(0.5, 1.5) for _ in range(self.n)]

    def value_grad(self, x):
        f = 0.5 * sum(l * xi * xi for l, xi in zip(self.lam, x))
        return f, [l * xi for l, xi in zip(self.lam, x)]


class Rosenbrock:
    """The standard banana. Minimum 0 at all-ones; a curved, narrow valley."""

    name = "Rosenbrock"
    optimum = 0.0

    def __init__(self, n: int = 2) -> None:
        self.n = n

    def start(self, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [-1.2 + 0.1 * rng.uniform(-1, 1) if i % 2 == 0 else 1.0 + 0.1 * rng.uniform(-1, 1)
                for i in range(self.n)]

    def value_grad(self, x):
        f, g = 0.0, [0.0] * self.n
        for i in range(self.n - 1):
            a, b = x[i], x[i + 1]
            t = b - a * a
            f += 100.0 * t * t + (1.0 - a) ** 2
            g[i] += -400.0 * a * t - 2.0 * (1.0 - a)
            g[i + 1] += 200.0 * t
        return f, g


class EnvelopeLog:
    """Accumulates the brake diagnostics over a run."""

    def __init__(self) -> None:
        self.total = self.engaged = 0.0
        self.n = 0
        self.lo = 1.0

    def add(self, opt) -> None:
        if not isinstance(opt, TwoSine):
            return
        _, mean, lo, frac = opt.diagnostics()
        self.total += mean
        self.engaged += frac
        self.lo = min(self.lo, lo)
        self.n += 1

    def report(self) -> tuple[float, float, float]:
        if not self.n:
            return (float("nan"),) * 3
        return self.total / self.n, self.lo, self.engaged / self.n


def run_function(problem, rule, lr, steps, seed, *, track=False):
    """Optimise a closed-form function. Returns (f at the LAST iterate, log).

    The last iterate, not the best one seen. Best-seen would pay an arm for
    passing near the optimum on its way past, and an arm that overshoots
    violently touches low values on the way through - which is precisely the
    behaviour this experiment is trying to detect rather than reward.
    """
    x = problem.start(seed)
    opt = build(rule, len(x), lr)
    log = EnvelopeLog()
    for _ in range(steps):
        f, g = problem.value_grad(x)
        if not math.isfinite(f) or f > 1e12:
            return float("inf"), log
        opt.step(x, g)
        if track:
            log.add(opt)
    f, _ = problem.value_grad(x)
    return (f if math.isfinite(f) else float("inf")), log


# ---------------------------------------------------------------------------
# a real network: the repository's own sine activation, learned per unit
# ---------------------------------------------------------------------------

class SineMLP:
    """`n_in -> H -> 1`, hidden units `f(z) = a*sin(b*(z-h)) + k`.

    The activation and its derivatives are section 8 of
    `Research/SineWaveActivationFunction.md`, initialised to that paper's
    `a=-1, b=1/3, h=0, k=0`. All four are learnable per unit, and the output
    head is linear.

    This is deliberately the hardest bed in the repository for an optimiser: the
    paper's own section 7 predicts that the periodicity gives the loss surface
    many local minima and that the optimiser will have to fight them. An
    optimiser paper should be tested where the architecture is known to be
    awkward, not where it is easy.

    Parameters are held in one flat list so any optimiser here can drive it.
    """

    def __init__(self, n_in: int, hidden: int, seed: int) -> None:
        rng = random.Random(seed)
        self.n_in, self.H = n_in, hidden
        s = math.sqrt(6.0 / (n_in + hidden))
        p = []
        for _ in range(hidden):
            p += [rng.uniform(-s, s) for _ in range(n_in)]     # W1 row
        p += [0.0] * hidden                                    # b1
        p += [-1.0] * hidden                                   # a
        p += [1.0 / 3.0] * hidden                              # b
        p += [0.0] * hidden                                    # h
        p += [0.0] * hidden                                    # k
        s2 = math.sqrt(6.0 / (hidden + 1))
        p += [rng.uniform(-s2, s2) for _ in range(hidden)]     # W2
        p += [0.0]                                             # b2
        self.params = p
        H, n = hidden, n_in
        self.o_W1, self.o_b1 = 0, H * n
        self.o_a, self.o_b = H * n + H, H * n + 2 * H
        self.o_h, self.o_k = H * n + 3 * H, H * n + 4 * H
        self.o_W2, self.o_b2 = H * n + 5 * H, H * n + 6 * H
        self.n_params = H * n + 6 * H + 1

    def loss_grad(self, xs, ys):
        """Mean squared error and its gradient w.r.t. every parameter."""
        p, H, n = self.params, self.H, self.n_in
        g = [0.0] * self.n_params
        loss = 0.0
        for x, y in zip(xs, ys):
            z, u, act = [], [], []
            for j in range(H):
                zj = p[self.o_b1 + j] + sum(p[self.o_W1 + j * n + t] * x[t] for t in range(n))
                uj = p[self.o_b + j] * (zj - p[self.o_h + j])
                z.append(zj)
                u.append(uj)
                act.append(p[self.o_a + j] * math.sin(uj) + p[self.o_k + j])
            pred = p[self.o_b2] + sum(p[self.o_W2 + j] * act[j] for j in range(H))
            d = pred - y
            loss += d * d
            e = 2.0 * d
            g[self.o_b2] += e
            for j in range(H):
                aj, bj = p[self.o_a + j], p[self.o_b + j]
                cos_u = math.cos(u[j])
                g[self.o_W2 + j] += e * act[j]
                da = e * p[self.o_W2 + j]                      # dL/d act_j
                g[self.o_a + j] += da * math.sin(u[j])         # df/da = sin u
                g[self.o_k + j] += da                          # df/dk = 1
                g[self.o_b + j] += da * aj * (z[j] - p[self.o_h + j]) * cos_u
                g[self.o_h + j] += da * (-aj * bj * cos_u)     # = -df/dz
                dz = da * aj * bj * cos_u                      # df/dz
                g[self.o_b1 + j] += dz
                for t in range(n):
                    g[self.o_W1 + j * n + t] += dz * x[t]
        m = float(len(xs))
        return loss / m, [v / m for v in g]


def make_task(n: int, seed: int):
    """A smooth two-input regression target - the same one the depth experiment
    in `../DepthCountedNormalisation/` uses, so the two are comparable.

    Predicting the mean scores about 0.18; anything near that has not learned.
    """
    rng = random.Random(seed + 9001)
    xs, ys = [], []
    for _ in range(n):
        x0, x1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        xs.append([x0, x1])
        ys.append(0.5 * math.sin(math.pi * x0) + 0.5 * x1 * x1)
    return xs, ys


def run_mlp(rule, lr, *, seed, steps, hidden=8, batch=16, n_data=64, track=False):
    """Mini-batch training. Returns (final full-batch MSE, mean envelope).

    Mini-batch on purpose: the two timescales can only disagree when the
    gradient carries noise, so a full-batch test would be the one setting in
    which the claim cannot show up either way.
    """
    net = SineMLP(2, hidden, seed)
    xs, ys = make_task(n_data, seed)
    rng = random.Random(seed + 77)
    opt = build(rule, net.n_params, lr)
    log = EnvelopeLog()
    for _ in range(steps):
        idx = [rng.randrange(n_data) for _ in range(batch)]
        _, g = net.loss_grad([xs[i] for i in idx], [ys[i] for i in idx])
        if any(not math.isfinite(v) for v in g):
            return float("inf"), log
        opt.step(net.params, g)
        if any(not math.isfinite(v) for v in net.params):
            return float("inf"), log
        if track:
            log.add(opt)
    loss, _ = net.loss_grad(xs, ys)
    return (loss if math.isfinite(loss) else float("inf")), log


# ---------------------------------------------------------------------------
# sweeping and reporting
# ---------------------------------------------------------------------------

def sweep(run, rule, lrs, seeds):
    """Every seed's score at every learning rate. Returns `{lr: [score per seed]}`.

    Per seed, not averaged, because the comparison that follows is paired: the
    same seed gives every arm the same initial weights and the same mini-batch
    order, so comparing arms within a seed removes the variance that a
    threshold count over averages leaves in.
    """
    return {lr: [run(rule, lr, seed=sd)[0] for sd in seeds] for lr in lrs}


def mean_of(vals):
    """Mean, or inf if any seed diverged - one divergence condemns the setting."""
    return sum(vals) / len(vals) if all(math.isfinite(v) for v in vals) else float("inf")


def means(cells):
    return {lr: mean_of(v) for lr, v in cells.items()}


def band(scores, target):
    """How many learning rates in the grid reach `target`. The robustness count."""
    return sum(1 for v in scores.values() if v <= target)


MARGIN = 1.25
"""A cell counts as a win only at this factor. Anything closer is a tie - two
optimisers within 25% of each other on one seed have not been distinguished."""


def paired(a_cells, b_cells):
    """Win/loss/tie over every (learning rate, seed) cell. `a` is the subject.

    Both arms saw the same seed, so each cell is a matched pair. Divergence
    counts as a loss to anything finite and a tie against another divergence.
    """
    w = l = t = 0
    for lr, a_vals in a_cells.items():
        for a_v, b_v in zip(a_vals, b_cells[lr]):
            fa, fb = math.isfinite(a_v), math.isfinite(b_v)
            if not fa and not fb:
                t += 1
            elif not fb:
                w += 1
            elif not fa:
                l += 1
            elif a_v * MARGIN < b_v:
                w += 1
            elif b_v * MARGIN < a_v:
                l += 1
            else:
                t += 1
    return w, l, t


def sign_test(w: int, l: int) -> float:
    """Two-sided exact binomial p-value on the decisive cells, null = 50/50.

    It is optimistic: cells at different learning rates on the same problem and
    seed share a trajectory and are not fully independent, so the true p is
    larger than this. It is here to stop a 57-35 split being read as decisive
    when it is thin, not to certify one.
    """
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def table(rows, cols, title, fmt="{:.3e}"):
    print(f"  {title:<18}" + "".join(f"{c:>12}" for c in cols))
    for label, vals in rows:
        cells = "".join(f"{('inf' if not math.isfinite(v) else fmt.format(v)):>12}" for v in vals)
        print(f"  {label:<18}{cells}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def checks() -> bool:
    ok = True

    def claim(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'pass' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    rng = random.Random(0)
    # 1. the beat identity - the product form is derived, not imposed
    worst = 0.0
    for _ in range(20000):
        rf, rs = rng.uniform(-1, 1), rng.uniform(-1, 1)
        step, car, env = TwoSine.combine(rf, rs)
        direct = 0.5 * (math.sin(math.pi / 2 * rf) + math.sin(math.pi / 2 * rs))
        worst = max(worst, abs(step - direct), abs(step - car * env))
    claim("sum-to-product: (sin a + sin b)/2 == sin(mean) * cos(half-diff)",
          worst < 1e-12, f"max err {worst:.2e}")

    # 2. bounded step and a brake that can only brake
    lo_e, hi_e, hi_s = 2.0, -2.0, 0.0
    for _ in range(20000):
        rf, rs = rng.uniform(-3, 3), rng.uniform(-3, 3)
        step, _, env = TwoSine.combine(rf, rs)
        lo_e, hi_e, hi_s = min(lo_e, env), max(hi_e, env), max(hi_s, abs(step))
    claim("|step| <= 1", hi_s <= 1.0 + 1e-12, f"max {hi_s:.6f}")
    claim("envelope in [0, 1] - clipped, it can only brake",
          -1e-12 <= lo_e and hi_e <= 1.0 + 1e-12, f"[{lo_e:.6f}, {hi_e:.6f}]")

    # the clip is what buys that guarantee, and this is what it costs to drop it:
    # unclipped, |phi_f - phi_s| can pass pi, the envelope goes negative, and the
    # step reverses rather than braking. A different optimiser, not a relaxed one.
    lo_full = min(TwoSine.combine(a / 10.0, b / 10.0, True)[2]
                  for a in range(-30, 31) for b in range(-30, 31))
    claim("full_wave drops that guarantee - envelope can reverse the step",
          lo_full < -0.5, f"min envelope {lo_full:.4f}")

    # 3. full agreement passes the signal, full opposition kills it
    claim("agreement (r, r) -> envelope 1", abs(TwoSine.combine(0.7, 0.7)[2] - 1.0) < 1e-12)
    claim("opposition (1, -1) -> step 0", abs(TwoSine.combine(1.0, -1.0)[0]) < 1e-12)

    # 4. reduces to Adam at small signal-to-noise
    worst = 0.0
    for r in (1e-4, 1e-3, 1e-2):
        step = TwoSine.combine(r, r)[0]
        worst = max(worst, abs(step / ((math.pi / 2) * r) - 1.0))
    claim("small-SNR limit -> (pi/2) * Adam's rule", worst < 1e-3, f"max rel err {worst:.2e}")

    # 5. equal timescales collapse the beat onto the single wave
    o1 = build("two-sine", 3, 0.01, beta_slow=0.9)
    o2 = build("one-sine", 3, 0.01, beta_slow=0.9)
    p1, p2 = [0.5, -0.3, 0.1], [0.5, -0.3, 0.1]
    for _ in range(30):
        g = [rng.uniform(-1, 1) for _ in range(3)]
        o1.step(p1, list(g))
        o2.step(p2, list(g))
    d = max(abs(a - b) for a, b in zip(p1, p2))
    claim("beta_slow == beta_fast collapses TwoSine onto one-sine", d < 1e-12, f"max diff {d:.2e}")

    # 6. the network's analytic gradient against finite differences
    net = SineMLP(2, 5, 3)
    xs, ys = make_task(12, 3)
    _, g = net.loss_grad(xs, ys)
    worst = 0.0
    for i in rng.sample(range(net.n_params), 18):
        h, keep = 1e-6, net.params[i]
        net.params[i] = keep + h
        fp, _ = net.loss_grad(xs, ys)
        net.params[i] = keep - h
        fm, _ = net.loss_grad(xs, ys)
        net.params[i] = keep
        num = (fp - fm) / (2 * h)
        worst = max(worst, abs(num - g[i]) / max(1.0, abs(num)))
    claim("SineMLP backprop == finite differences", worst < 1e-5, f"max rel err {worst:.2e}")

    print()
    print("  all checks passed" if ok else "  SOME CHECKS FAILED")
    return ok


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="two-sine-wave optimiser")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--mlp-steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true", help="checks only, then exit")
    a = ap.parse_args()

    print("=" * 79)
    print("PART 0 - checks: is the algebra what the docstring says it is?")
    print("=" * 79)
    if not checks():
        sys.exit(1)
    if a.check:
        return

    if a.quick:
        a.steps, a.mlp_steps, a.seeds = 400, 300, 3
    seeds = list(range(a.seeds))
    print()
    print(f"{a.seeds} seeds, {a.steps} steps on the closed-form problems, "
          f"{a.mlp_steps} on the network")
    print(f"learning-rate grid: {', '.join(str(l) for l in LR_GRID)}")
    print("every arm is reported at its own best learning rate\n")

    quad, rosen = Quadratic(), Rosenbrock()
    MLP_TARGET = 1e-2
    targets = {quad.name: 1e-3, rosen.name: 1e-2}
    cells = {}          # (problem, rule) -> {lr: [score per seed]}

    print("=" * 79)
    print("PART 1 - closed-form problems, best over the learning-rate grid")
    print("=" * 79)
    for prob in (quad, rosen):
        print(f"\n  {prob.name} (minimum {prob.optimum}, value at the last iterate)")
        rows = []
        for rule in ARMS:
            c = sweep(lambda r, lr, seed: run_function(prob, r, lr, a.steps, seed),
                      rule, LR_GRID, seeds)
            cells[(prob.name, rule)] = c
            m = means(c)
            best_lr = min(m, key=lambda k: m[k])
            rows.append((rule, m[best_lr], best_lr))
        print(f"  {'arm':<18}{'best f':>12}{'at lr':>9}")
        for label, val, lr in rows:
            shown = "inf" if not math.isfinite(val) else f"{val:.3e}"
            print(f"  {label:<18}{shown:>12}{lr:>9g}")

    print()
    print("=" * 79)
    print("PART 2 - the claim: how many learning rates still converge?")
    print("=" * 79)
    print("  A fixed absolute target, the same for every arm. The count is out of")
    print(f"  {len(LR_GRID)} learning rates spanning {LR_GRID[0]} to {LR_GRID[-1]},")
    print("  largest on the left. A wider band is an optimiser that needs less tuning.\n")
    for prob in (quad, rosen):
        t = targets[prob.name]
        print(f"  {prob.name}, target f <= {t:g}")
        for rule in ARMS:
            m = means(cells[(prob.name, rule)])
            marks = "".join("#" if m[lr] <= t else "." for lr in LR_GRID)
            print(f"    {rule:<18}{band(m, t):>3}/{len(LR_GRID)}   {marks}")
        print(f"    {'':18}     {''.join(('%g' % l)[0] for l in LR_GRID)}   <- lr, first digit")
        print()

    print("=" * 79)
    print("PART 3 - a real network: sine activation, learnable per unit, mini-batch")
    print("=" * 79)
    print("  2 -> 8 sine units -> 1, batch 16 of 64. Predicting the mean scores 0.1796,")
    print("  so anything at or above that has not learned.\n")
    mlp_best = {}
    for rule in ARMS:
        c = sweep(lambda r, lr, seed: run_mlp(r, lr, seed=seed, steps=a.mlp_steps),
                  rule, LR_GRID, seeds)
        cells[("network", rule)] = c
        m = means(c)
        mlp_best[rule] = min(m, key=lambda k: m[k])
    print(f"  learning-rate sweep (mean MSE over {a.seeds} seeds)")
    table([(r, [means(cells[("network", r)])[lr] for lr in LR_GRID]) for r in ARMS],
          [f"{l:g}" for l in LR_GRID], "arm \\ lr")

    print()
    print(f"  learning rates reaching MSE <= {MLP_TARGET:g}")
    for rule in ARMS:
        m = means(cells[("network", rule)])
        marks = "".join("#" if m[lr] <= MLP_TARGET else "." for lr in LR_GRID)
        print(f"    {rule:<18}{band(m, MLP_TARGET):>3}/{len(LR_GRID)}   {marks}")
    print(f"    {'':18}     {''.join(('%g' % l)[0] for l in LR_GRID)}   <- lr, first digit")

    print()
    print(f"  at each arm's own best learning rate, {a.seeds} seeds")
    print(f"  {'arm':<18}{'lr':>7}{'mean MSE':>12}{'std':>11}{'worst':>12}"
          f"{'env mean':>10}{'env min':>9}{'engaged':>9}")
    final = {}
    for rule in ARMS:
        lr = mlp_best[rule]
        vals, logs = [], []
        for sd in seeds:
            v, lg = run_mlp(rule, lr, seed=sd, steps=a.mlp_steps, track=True)
            vals.append(v)
            logs.append(lg.report())
        good = [v for v in vals if math.isfinite(v)]
        mean = sum(good) / len(good) if good else float("inf")
        sd_ = (sum((v - mean) ** 2 for v in good) / len(good)) ** 0.5 if good else 0.0
        final[rule] = mean
        shown = "inf" if not math.isfinite(mean) else f"{mean:.6f}"
        worst = f"{max(good):.6f}" if good else "inf"
        if rule.startswith("two-sine"):
            diag = (f"{sum(l[0] for l in logs) / len(logs):>10.4f}"
                    f"{min(l[1] for l in logs):>9.4f}"
                    f"{sum(l[2] for l in logs) / len(logs):>9.1%}")
        else:
            diag = f"{'-':>10}{'-':>9}{'-':>9}"
        print(f"  {rule:<18}{lr:>7g}{shown:>12}{sd_:>11.6f}{worst:>12}{diag}")
    print()
    print("  'env' is cos((phi_f - phi_s)/2), the brake. 'engaged' is the share of")
    print("  (parameter, step) pairs where it fell below 0.99. An engaged share of 0")
    print("  would mean the brake never bit and the two-wave form is Adam with state.")

    print()
    print("=" * 79)
    print("PART 4 - verdict")
    print("=" * 79)
    print("  A. best loss on the network, each arm at its own best learning rate")
    for rule in ARMS:
        print(f"     {rule:<18}{final[rule]:.6f}")

    print()
    print("  B. width of the converging learning-rate band")
    print(f"     {'arm':<18}{'quadratic':>11}{'rosenbrock':>12}{'network':>10}{'total':>9}")
    totals = {}
    for rule in ARMS:
        b = [band(means(cells[(quad.name, rule)]), targets[quad.name]),
             band(means(cells[(rosen.name, rule)]), targets[rosen.name]),
             band(means(cells[("network", rule)]), MLP_TARGET)]
        totals[rule] = sum(b)
        print(f"     {rule:<18}{b[0]:>8}/{len(LR_GRID)}{b[1]:>9}/{len(LR_GRID)}"
              f"{b[2]:>7}/{len(LR_GRID)}{sum(b):>6}/{3 * len(LR_GRID)}")
    best = max(totals.values())
    print(f"     widest: {', '.join(r for r, t in totals.items() if t == best)} "
          f"({best}/{3 * len(LR_GRID)})")

    n_cells = 3 * len(LR_GRID) * a.seeds
    print()
    print(f"  C. two-sine against each control, paired over all {n_cells} "
          f"(problem, lr, seed) cells")
    print(f"     Same seed means the same initial weights and the same batch order, so")
    print(f"     each cell is a matched pair. A win needs a factor of {MARGIN}; closer is a tie.")
    print("     p is a two-sided sign test on the decisive cells - see sign_test().")
    print(f"     {'against':<18}{'win':>6}{'loss':>6}{'tie':>6}   reading")
    subject = {lr: [v for prob in (quad.name, rosen.name, "network")
                    for v in cells[(prob, "two-sine")][lr]] for lr in LR_GRID}
    for ctrl in ARMS:
        if ctrl == "two-sine":
            continue
        other = {lr: [v for prob in (quad.name, rosen.name, "network")
                      for v in cells[(prob, ctrl)][lr]] for lr in LR_GRID}
        w, l, t = paired(subject, other)
        pv = sign_test(w, l)
        if pv >= 0.05:
            reading = f"not distinguished (p={pv:.3f})"
        elif w > l:
            reading = f"two-sine ahead (p={pv:.3f})"
        else:
            reading = f"{ctrl} ahead (p={pv:.3f})"
        print(f"     {ctrl:<18}{w:>6}{l:>6}{t:>6}   {reading}")

    print()
    print("  Read C first: it is the only part of this with the seeds paired. B is a")
    print("  threshold count over means and moves around with the seed set. And read")
    print("  both against the controls rather than against adam - if two-ema-linear is")
    print("  not distinguished from two-sine, the third EMA did the work and the")
    print("  interference did nothing; if one-sine is not, the second wave did nothing;")
    print("  if sign-gate is not, an `if` would have done.")


if __name__ == "__main__":
    main()
