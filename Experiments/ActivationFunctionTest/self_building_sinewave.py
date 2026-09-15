#!/usr/bin/env python3
"""A self-building network, measured on the activation function.

Standard library only - no TensorFlow, no PyTorch, no gymnasium - so it runs
anywhere with nothing installed, the same rule ``RadixCyclicNN`` and
``Experiments/DepthCountedNormalisation`` follow.  The other scripts in this directory need a
deep-learning framework and a gym; this one needs a Python.

    python3 self_building_sinewave.py                  # full run
    python3 self_building_sinewave.py --quick          # one seed, fewer epochs
    python3 self_building_sinewave.py --check          # gradient checks, then exit
    python3 self_building_sinewave.py --init shared    # same init for every arm


What is under test
------------------
A *self-building* network starts too small and grows its hidden layer whenever
learning stalls - the rule in ``Experiments/SBNN_RNN_ActivationFunction/main.py``.  That
turns a question about an activation function into a question you can count:

    **how many units does the network have to build, and how many of the ones
    it built are still alive when it stops?**

Those are different numbers, and the gap between them is the point.  A unit
whose derivative has gone to zero everywhere on the data is a unit the network
paid to build and can no longer train.  Growth cannot recover it - growth adds
*new* units next to the dead one.  So a self-building network wearing a
saturating activation does not merely learn slower; it builds capacity it
cannot use, stalls again, and builds more.

``Research/SineWaveActivationFunction.md`` section 6 claims the sine unit never
permanently dies: ``|f'(x)| = |a*b*cos(b*(x-h))|`` returns to ``|a*b|`` every
half period, so the dead fraction stays at about 2% of the input range however
large the pre-activations get, while sigmoid's grows toward 100% and ReLU's is
a fixed 50%.  That claim was made about a function on its own.  Here it is put
inside a network that is allowed to respond to its own failure to learn, which
is where it should cost something.

Five arms, identical in every respect except the shape of one unit:

    sigmoid      1 / (1 + exp(-(z - h)))          h learnable, nothing else
    tanh         tanh(z - h)                      h learnable, nothing else
    relu         max(0, z - h)                    h learnable, nothing else
    sine-fixed   -sin((z - h) / 3)                h learnable, nothing else
    sine          a * sin(b * (z - h)) + k        a, b, h, k all learnable

``tanh`` is here because section 9 of the paper asks for it by name: it is the
strongest of the classical squashing functions and the fairest opponent for a
bounded periodic one.

``sine-fixed`` is the parameter-matched control.  It has *exactly* the knob
count of the baselines - one weight, one bias-shaped parameter, one output
weight per unit - so what it shows is the shape of the function and not the size
of the model, and the ``sine`` against ``sine-fixed`` gap is what *learning the
activation* is worth with the shape held fixed.  ``sine`` is the full unit from
the paper, and it is the one with an advantage to declare: four parameters per
unit against one.

Every arm has the same bias.  It is written ``z - h`` in all five because
``df/dh = -df/dz`` exactly (section 8), so the sine's phase parameter *is* the
per-unit input bias - the same knob the baselines get, living inside the
nonlinearity instead of in front of it.  Writing it the same way in all five
arms keeps the comparison honest and the code symmetric.

Confounds I would rather name myself
------------------------------------
* ``a`` is degenerate with the output weight and ``k`` with the output bias:
  ``v * (a*sin(u) + k)`` can be rewritten by moving scale into ``v``.  The full
  ``sine`` arm therefore carries redundant parameters, which is why
  ``sine-fixed`` exists and why it is the one to quote.
* Each arm is initialised the way its own literature says to (He for ReLU,
  Xavier for sigmoid and tanh, and the paper's ``Z_RANGE = 4.5`` for the sine,
  matched to the quarter period ``3*pi/2 = 4.712`` so a fresh unit starts on the
  monotone arm of its own wave).  That is the fair comparison - each arm gets
  the initialisation designed for it - but it does mean the arms do not start
  from the same weights.  ``--init shared`` puts every arm on the sine's
  initialisation instead, so the result can be checked for being an artifact of
  the draw.
* Three tasks, deliberately not all friendly.  ``steps`` is a staircase, which
  is the shape a sigmoid *is* one step of; ``chirp`` is a swept sine, which
  suits a periodic unit; ``spline`` is a smooth curve through fixed random
  knots and suits neither.  Read all three.
* Single-hidden-layer regression on 1-D data.  Nothing here says anything about
  depth, and section 11 of the paper does not claim it does.

The measured numbers, and what came out of the controls, are in this directory's
``README.md``.  Run it and they print themselves.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time

# --------------------------------------------------------------------------
# The activation under test
# --------------------------------------------------------------------------

DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0
"""Defaults from ``RadixCyclicNN/radixnet/activation.py``: exactly ``-sin(x/3)``."""

QUARTER_PERIOD = 3.0 * math.pi / 2.0
"""``(2*pi/b)/4`` with ``b = 1/3``: 4.712, the far end of the monotone arm."""

Z_RANGE = 4.5
"""``RadixCyclicNN/radixnet/graph.py``.  4.5 < 4.712, with 0.21 to spare."""

DEAD = 0.01
"""A unit is dead when ``|df/dz|`` is under this at every training input.

The same threshold the paper's section 6 table uses.  Dead is not "small" - it
is "cannot be trained by any sample in the data", which for a self-building
network is the difference between a unit and a bill.
"""


def sine(z, a, b, h, k):
    """``f(z) = a*sin(b*(z-h)) + k``."""
    return a * math.sin(b * (z - h)) + k


def sine_partials(z, a, b, h, k):
    """``(f, df/dz, df/da, df/db, df/dh, df/dk)`` - the six of section 8."""
    d = z - h
    u = b * d
    su = math.sin(u)
    cu = math.cos(u)
    abc = a * b * cu
    return (a * su + k, abc, su, a * d * cu, -abc, 1.0)


def sigmoid(z):
    if z < -50.0:
        return 0.0
    if z > 50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

_KNOTS = [0.35, -0.62, 0.81, -0.14, -0.93, 0.47, 0.22, -0.70, 0.55]
"""Fixed knots for the ``spline`` task - hard-coded so every arm, every seed and
every run sees the identical target.  The only randomness in this experiment is
the weight initialisation."""


def _spline(x):
    """Catmull-Rom through ``_KNOTS`` laid out over ``[-1.2, 1.2]``."""
    n = len(_KNOTS)
    span = 2.4 / (n - 1)
    pos = (x + 1.2) / span
    i = min(max(int(math.floor(pos)), 0), n - 2)
    t = pos - i
    p0 = _KNOTS[max(i - 1, 0)]
    p1 = _KNOTS[i]
    p2 = _KNOTS[i + 1]
    p3 = _KNOTS[min(i + 2, n - 1)]
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def _steps(x):
    """A staircase.  Section 3: a sigmoid is one step of a rotated sine wave,
    so this is the task built out of the baseline's own shape."""
    if x < -0.45:
        return -0.8
    if x < 0.25:
        return 0.1
    return 0.9


def _chirp(x):
    """A swept sine: instantaneous frequency rising from 0.8 to 2.4 cycles per
    unit, 1.6 cycles of phase across the domain.  This one suits a periodic unit
    and is reported as such."""
    u = (x + 1.0) / 2.0
    return math.sin(2.0 * math.pi * (0.8 + 0.8 * u) * u)


TASKS = {"spline": (_spline, 5e-3), "steps": (_steps, 1e-2), "chirp": (_chirp, 1e-3)}
"""``name -> (target function, the train MSE that counts as solved)``.

The target is per task and not global because the tasks have different error
floors - the staircase's is set by its own discontinuity, which no bounded
smooth function fits exactly.  Each one sits just above the floor a 48-unit
network of any kind reaches, so ``solved@`` asks *how many units does this arm
need to get to the same place*, rather than who first crosses a line that three
of the arms could never cross.  ``--target`` overrides all three.
"""


def make_data(task, n_train=48, n_test=193):
    """Training grid, and a test grid offset from it - the test points fall
    *between* training points, so the score is interpolation and not recall."""
    f = TASKS[task][0]
    xs = [-1.0 + 2.0 * i / (n_train - 1) for i in range(n_train)]
    ys = [f(x) for x in xs]
    step = 2.0 / (n_test - 1)
    txs = [-1.0 + step * (i + 0.5) for i in range(n_test - 1)]
    tys = [f(x) for x in txs]
    return xs, ys, txs, tys


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

ARMS = ("sigmoid", "tanh", "relu", "sine-fixed", "sine")


class SelfBuildingNet:
    """``1 -> H -> 1`` with a linear output head, and ``H`` grown on demand.

    The head is linear on purpose.  Section 9 of the paper names the squashed
    output head in ``sinewave.py`` as a confound that flatters the sine, and
    points at the DQN pair in this directory as the version that removes it.
    This script removes it the same way: the activation appears in the hidden
    layer, where it is the thing being measured, and nowhere else.
    """

    PATIENCE = 60
    """Epochs of no progress that count as a stall."""
    STALL_TOL = 0.02
    """Under 2% improvement across the window is a stall.  The test is *relative*:
    an absolute threshold fires at different times for arms sitting at different
    loss scales, and then the arms are running different experiments."""
    MAX_HIDDEN = 48
    GROWTH = 4

    def __init__(self, kind, hidden, rng, lr=0.02, init="native", act_lr_ratio=0.1):
        self.kind = kind
        self.rng = rng
        self.lr = lr
        self.act_lr = lr * act_lr_ratio
        # Section 8: the activation is the shape of the space the weights are
        # searching in.  Move the space as fast as the search and both diverge.
        # ``--act-lr`` exposes the ratio because that claim is testable, and the
        # ``deaf`` column is what tests it.

        self.learn_act = kind == "sine"
        self.periodic = kind in ("sine", "sine-fixed")

        if init == "shared":
            self.w_scale = Z_RANGE
        elif kind == "relu":
            self.w_scale = math.sqrt(2.0)            # He, fan_in = 1
        elif self.periodic:
            self.w_scale = Z_RANGE                   # the paper's rule
        else:
            self.w_scale = math.sqrt(6.0 / 2.0)      # Xavier, fan_in = fan_out = 1

        self.w, self.h, self.a, self.b, self.k, self.v = [], [], [], [], [], []
        self.c = 0.0
        self.hidden = 0
        self._add_units(hidden)

        # Adam, the optimiser the other scripts in this directory use.  It also
        # removes a confound plain SGD would leave in: the arms do not produce
        # gradients of the same size (the sine's df/dz carries a factor of
        # b = 1/3, ReLU's is 1), and under SGD that difference is an unequal
        # learning rate rather than a property of the activation.  Adam's
        # per-parameter normalisation gives every arm the same step size, so
        # what is left to differ is the shape of the function.
        self.t = 0
        self.m = {n: [0.0] * hidden for n in ("w", "h", "a", "b", "k", "v")}
        self.u = {n: [0.0] * hidden for n in ("w", "h", "a", "b", "k", "v")}
        self.mc = 0.0
        self.uc = 0.0

        # growth control
        self.history = []

    def _add_units(self, n):
        rng = self.rng
        s = self.w_scale
        for _ in range(n):
            w = rng.uniform(-s, s)
            self.w.append(w)
            # Put the unit's operating point somewhere inside the data range,
            # so a fresh unit has something to say about the inputs it sees.
            self.h.append(rng.uniform(-abs(w), abs(w)))
            self.a.append(DEFAULT_A)
            self.b.append(DEFAULT_B)
            self.k.append(DEFAULT_K)
            self.v.append(rng.uniform(-0.1, 0.1))
        self.hidden += n

    def unit(self, z, i):
        """``(f, df/dz)`` for unit ``i`` at pre-activation ``z``."""
        d = z - self.h[i]
        if self.kind == "sine":
            u = self.b[i] * d
            return self.a[i] * math.sin(u) + self.k[i], self.a[i] * self.b[i] * math.cos(u)
        if self.kind == "sine-fixed":
            # the same formula, a/b/k pinned at the defaults so only h is free
            u = DEFAULT_B * d
            return DEFAULT_A * math.sin(u) + DEFAULT_K, DEFAULT_A * DEFAULT_B * math.cos(u)
        if self.kind == "sigmoid":
            s = sigmoid(d)
            return s, s * (1.0 - s)
        if self.kind == "tanh":
            t = math.tanh(d)
            return t, 1.0 - t * t
        if self.kind == "relu":
            return (d, 1.0) if d > 0.0 else (0.0, 0.0)
        raise ValueError(self.kind)

    def predict(self, x):
        y = self.c
        for i in range(self.hidden):
            f, _ = self.unit(self.w[i] * x, i)
            y += self.v[i] * f
        return y

    def mse(self, xs, ys):
        return sum((self.predict(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

    def step(self, xs, ys):
        """One full-batch gradient step.  Returns the loss before the step."""
        n = self.hidden
        gw = [0.0] * n
        gh = [0.0] * n
        ga = [0.0] * n
        gb = [0.0] * n
        gk = [0.0] * n
        gv = [0.0] * n
        gc = 0.0
        loss = 0.0
        scale = 2.0 / len(xs)

        for x, y in zip(xs, ys):
            pred = self.c
            cache = []
            for i in range(n):
                z = self.w[i] * x
                if self.learn_act:
                    f, dz, da, db, _, dk = sine_partials(
                        z, self.a[i], self.b[i], self.h[i], self.k[i]
                    )
                    cache.append((f, dz, da, db, dk))
                else:
                    f, dz = self.unit(z, i)
                    cache.append((f, dz, 0.0, 0.0, 0.0))
                pred += self.v[i] * f
            err = pred - y
            loss += err * err
            dy = scale * err
            gc += dy
            for i in range(n):
                f, dz, da, db, dk = cache[i]
                gv[i] += dy * f
                df = dy * self.v[i]
                if self.learn_act:
                    ga[i] += df * da
                    gb[i] += df * db
                    gk[i] += df * dk
                gw[i] += df * dz * x
                gh[i] -= df * dz          # df/dh = -df/dz, section 8, in all five arms

        self.t += 1
        self._adam("w", self.w, gw, self.lr)
        self._adam("h", self.h, gh, self.lr)
        self._adam("v", self.v, gv, self.lr)
        self.mc, self.uc, dc = _adam_delta(_clip(gc), self.mc, self.uc, self.t, self.lr)
        self.c -= dc
        if self.learn_act:
            self._adam("a", self.a, ga, self.act_lr)
            self._adam("b", self.b, gb, self.act_lr)
            self._adam("k", self.k, gk, self.act_lr)

        return loss / len(xs)

    def _adam(self, name, params, grads, lr):
        m, u = self.m[name], self.u[name]
        t = self.t
        for i, g in enumerate(grads):
            m[i], u[i], d = _adam_delta(_clip(g), m[i], u[i], t, lr)
            params[i] -= d

    def maybe_grow(self, loss):
        """Grow when the loss has stopped moving - the self-building rule.

        New units arrive with a near-zero output weight, so the network's
        function barely moves at the moment it grows: growth adds capacity
        without throwing away what has already been learned.
        """
        self.history.append(loss)
        if self.hidden >= self.MAX_HIDDEN or len(self.history) < self.PATIENCE:
            return False
        window = self.history[-self.PATIENCE:]
        first = window[0]
        improvement = (first - window[-1]) / max(abs(first), 1e-12)
        if improvement >= self.STALL_TOL:
            return False
        self._add_units(self.GROWTH)
        for d in (self.m, self.u):
            for lst in d.values():
                lst.extend([0.0] * self.GROWTH)
        self.history.clear()
        return True

    def liveness(self, xs):
        """``(dead units, deaf fraction, max |z|)`` over the training set.

        *Dead* counts units whose ``|df/dz|`` is under :data:`DEAD` at **every**
        training input - not "saturated on this sample" but unreachable by the
        whole data set, a unit that was built and can never be trained again.

        *Deaf* is the fraction of ``(unit, sample)`` pairs under the same
        threshold: how much of the training signal the layer cannot feel.  It is
        the quantity in the paper's section 6 table, except that the table
        computes it over a range chosen by hand and this computes it over the
        pre-activations the network actually produced - which is the number that
        decides whether the claim matters in practice.

        ``max |z|`` is the range discipline of section 11.2: how far the
        pre-activations travelled from the arm they were initialised on.
        """
        dead = 0
        deaf = 0
        zmax = 0.0
        for i in range(self.hidden):
            peak = 0.0
            for x in xs:
                z = self.w[i] * x
                zmax = max(zmax, abs(z))
                d = abs(self.unit(z, i)[1])
                if d < DEAD:
                    deaf += 1
                peak = max(peak, d)
            if peak < DEAD:
                dead += 1
        return dead, deaf / (self.hidden * len(xs)), zmax


def _clip(g, lim=5.0):
    return lim if g > lim else (-lim if g < -lim else g)


_B1, _B2, _EPS = 0.9, 0.999, 1e-8


def _adam_delta(g, m, u, t, lr):
    """One Adam update: returns the new moments and the step to subtract."""
    m = _B1 * m + (1.0 - _B1) * g
    u = _B2 * u + (1.0 - _B2) * g * g
    mh = m / (1.0 - _B1 ** t)
    uh = u / (1.0 - _B2 ** t)
    return m, u, lr * mh / (math.sqrt(uh) + _EPS)


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

class Result:
    """What one training run built, and what state it was in when it stopped."""

    __slots__ = ("test_mse", "train_mse", "built", "dead", "deaf", "zmax",
                 "to_target", "units_at_target")


def run(kind, task, seed, epochs, target, init, hidden0=4, lr=0.02, act_lr_ratio=0.1):
    """Train one arm on one task from one seed, and measure what it built."""
    xs, ys, txs, tys = make_data(task)
    net = SelfBuildingNet(kind, hidden0, random.Random(seed), lr=lr, init=init,
                          act_lr_ratio=act_lr_ratio)

    to_target = None
    units_at_target = None

    for epoch in range(1, epochs + 1):
        loss = net.step(xs, ys)
        if to_target is None and loss <= target:
            to_target = epoch
            units_at_target = net.hidden
        if to_target is None:
            # Self-building stops once the network is good enough; a network
            # that has met the target has finished building.
            net.maybe_grow(loss)

    r = Result()
    r.train_mse = net.mse(xs, ys)
    r.test_mse = net.mse(txs, tys)
    r.built = net.hidden
    r.dead, r.deaf, r.zmax = net.liveness(xs)
    r.to_target = to_target
    r.units_at_target = units_at_target
    return r


# --------------------------------------------------------------------------
# Gradient checks - the numbers above are only worth reading if these pass
# --------------------------------------------------------------------------

def check(verbose=True):
    eps = 1e-6
    worst = 0.0
    params = [
        (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K),
        (-0.7, 0.9, 1.3, -0.4),
        (2.0, 0.05, -2.0, 0.75),
    ]
    xs = [-12.0, -4.5, -1.0, 0.0, 0.3, 2.5, 9.0, 15.0]

    # section 4: the initialisation range sits inside the monotone arm
    assert Z_RANGE < QUARTER_PERIOD, "Z_RANGE must stay on one arm of the wave"

    # the default is exactly -sin(x/3)
    for x in xs:
        worst = max(worst, abs(sine(x, DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K) + math.sin(x / 3.0)))

    for (a, b, h, k) in params:
        for x in xs:
            f, dz, da, db, dh, dk = sine_partials(x, a, b, h, k)
            worst = max(worst, abs(f - sine(x, a, b, h, k)))
            # five central finite differences
            fd = [
                (sine(x + eps, a, b, h, k) - sine(x - eps, a, b, h, k)) / (2 * eps),
                (sine(x, a + eps, b, h, k) - sine(x, a - eps, b, h, k)) / (2 * eps),
                (sine(x, a, b + eps, h, k) - sine(x, a, b - eps, h, k)) / (2 * eps),
                (sine(x, a, b, h + eps, k) - sine(x, a, b, h - eps, k)) / (2 * eps),
                (sine(x, a, b, h, k + eps) - sine(x, a, b, h, k - eps)) / (2 * eps),
            ]
            for got, want in zip((dz, da, db, dh, dk), fd):
                worst = max(worst, abs(got - want))
            # section 8: df/dh = -df/dx, exactly
            worst = max(worst, abs(dh + dz))
            # section 4: negating amplitude and offset together negates the unit
            worst = max(worst, abs(sine(x, -a, b, h, -k) + sine(x, a, b, h, k)))

    # every arm's df/dz against a finite difference of its own forward pass
    for kind in ARMS:
        net = SelfBuildingNet(kind, 3, random.Random(7))
        for i in range(net.hidden):
            for z in (-3.0, -0.4, 0.7, 2.2, 6.0):
                if kind == "relu" and abs(z - net.h[i]) < 1e-3:
                    continue                      # the kink has no derivative
                _, dz = net.unit(z, i)
                fd = (net.unit(z + eps, i)[0] - net.unit(z - eps, i)[0]) / (2 * eps)
                worst = max(worst, abs(dz - fd))

    ok = worst < 1e-5
    if verbose:
        print(f"gradient check: {'ok' if ok else 'FAILED'} (max error {worst:.2e})")
    if not ok:
        raise AssertionError(f"gradient check failed, max error {worst:.2e}")
    return worst


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _pm(values, fmt="{:.4f}"):
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.fmean(values))} +/- {fmt.format(statistics.pstdev(values))}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=sorted(TASKS) + ["all"], default="all")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--target", type=float, default=None,
                   help="override every task's own solved-at threshold")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--init", choices=("native", "shared"), default="native")
    p.add_argument("--act-lr", type=float, default=0.1, dest="act_lr",
                   help="activation-parameter learning rate, as a fraction of --lr "
                        "(section 8 says a tenth; raise it and watch 'deaf')")
    p.add_argument("--quick", action="store_true", help="one seed, 1000 epochs")
    p.add_argument("--check", action="store_true", help="run the gradient checks and exit")
    args = p.parse_args()

    if args.check:
        check()
        return

    if args.quick:
        args.seeds = 1
        args.epochs = 1000

    check()
    tasks = sorted(TASKS) if args.task == "all" else [args.task]
    started = time.time()

    print(
        f"\nself-building network, 1 -> H -> 1, linear head, Adam"
        f"\nstart H=4, +4 units per stall of {SelfBuildingNet.PATIENCE} epochs, "
        f"max H={SelfBuildingNet.MAX_HIDDEN}"
        f"\nseeds={args.seeds}  epochs={args.epochs}  lr={args.lr}  init={args.init}"
        f"  act-lr={args.act_lr} x lr"
    )

    summary = {}
    for task in tasks:
        target = args.target if args.target else TASKS[task][1]
        print(f"\n== {task}  (solved = train MSE <= {target:g}) "
              + "=" * max(0, 52 - len(task)))
        head = (f"{'activation':<12}  {'test MSE':>19}  {'built':>12}  {'dead':>11}  "
                f"{'deaf':>6}  {'|z|max':>6}  {'solved@':>13}")
        print(head)
        print("-" * len(head))
        for kind in ARMS:
            rs = [run(kind, task, seed, args.epochs, target, args.init,
                      lr=args.lr, act_lr_ratio=args.act_lr)
                  for seed in range(args.seeds)]
            test = [r.test_mse for r in rs]
            built = [float(r.built) for r in rs]
            dead = [float(r.dead) for r in rs]
            deaf = [100.0 * r.deaf for r in rs]
            zmax = [r.zmax for r in rs]
            solved = [r for r in rs if r.to_target is not None]
            at = _pm([float(r.units_at_target) for r in solved], "{:.1f}") if solved else "-"
            if solved and len(solved) < len(rs):
                at += f"[{len(solved)}/{len(rs)}]"
            print(
                f"{kind:<12}  {_pm(test, '{:.2e}'):>19}  {_pm(built, '{:.1f}'):>12}  "
                f"{_pm(dead, '{:.1f}'):>11}  {statistics.fmean(deaf):>5.1f}%  "
                f"{statistics.fmean(zmax):>6.1f}  {at:>13}"
            )
            summary[(task, kind)] = (
                statistics.fmean(test), statistics.fmean(built),
                statistics.fmean(dead), statistics.fmean(deaf),
                None if not solved else statistics.fmean([float(r.units_at_target)
                                                          for r in solved]),
                len(solved) == len(rs),
            )

    print(f"""
{'=' * 88}
built    hidden units at the end - what the network decided it had to build.
dead     of those, units with |df/dz| < {DEAD} at *every* training input: built, then
         lost.  Growth cannot recover one; it can only add another beside it.
deaf     fraction of (unit, sample) pairs under the same threshold - the section 6
         dead fraction, measured on the pre-activations the network really produced.
         Half of ReLU's is its own gating and not a pathology; the 'dead' column is
         the part of it that cannot be undone.
|z|max   largest pre-activation reached (section 11.2, range discipline).
solved@  units built when the train MSE first reached the task's target; '-' is never,
         and [n/{args.seeds}] is how many seeds got there when only some did.
""")

    for task in tasks:
        rows = {k: summary[(task, k)] for k in ARMS}
        best = min(ARMS, key=lambda k: rows[k][0])
        done = [k for k in ARMS if rows[k][5]]
        cheap = min(done, key=lambda k: rows[k][4]) if done else None
        live = min(ARMS, key=lambda k: rows[k][3])
        line = (f"{task:<7} best fit: {best} ({rows[best][0]:.1e})"
                f"   least deaf: {live} ({rows[live][3]:.1f}%)")
        if cheap is not None:
            line += f"   fewest units to target: {cheap} ({rows[cheap][4]:.1f})"
        print(line)
    print(f"\n({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
