"""TwoSine: Adam, but the two moments are read as two sine waves that interfere.

Standard library only - no numpy, no torch - so it runs anywhere with nothing
installed, the same rule `Research/experiments/` and `RadixCyclicNN/` follow.

    python3 two_sine_optimizer.py            # self-checks on the algebra
    python3 test_two_sine_optimizer.py       # the benchmark


The idea
--------
`Research/SineWaveActivationFunction.md` replaces the sigmoid with a wave on the
argument that a sigmoid already *is* one - one single step of one, with
everything before and after thrown away. This applies the same move one level
up, to the optimiser.

Adam's update is `m / sqrt(v)`: a first moment over a root second moment. That
ratio is a **signal-to-noise ratio**. It is near `+/-1` when the recent gradients
all point the same way and near `0` when they cancel, so Adam's step magnitude is
already a saturating, odd, bounded function of consistency - a sigmoid-shaped
response in disguise, clipped at `+/-1` and flat forever after.

So take two of them, at two different timescales, and let them interfere.

Keep three EMAs per parameter instead of Adam's two:

    m_f  fast first moment   beta 0.9    ~10 steps of memory
    m_s  slow first moment   beta 0.99   ~100 steps
    v    second moment       beta 0.999  ~1000 steps, the shared scale

Two signal-to-noise ratios come out of that, one per timescale, both measured
against the same scale `sqrt(v)`:

    r_f = m_f / (sqrt(v) + eps)        r_s = m_s / (sqrt(v) + eps)

Map each to a phase - saturated agreement is a quarter turn:

    phi = (pi/2) * clip(r, -1, 1)

and take the two waves' mean. That is where the two sine waves become one
product, by an identity and not by a choice:

    step = (sin(phi_f) + sin(phi_s)) / 2
         = sin((phi_f + phi_s) / 2) * cos((phi_f - phi_s) / 2)
           \_______carrier________/   \_______envelope_______/

**This is a beat.** Two waves of nearby frequency sum to a carrier at their mean
modulated by an envelope at their difference; it is why two mistuned piano
strings throb. Here the two "frequencies" are the two timescales, and the beat
splits the update into the two questions an optimiser actually has:

* **carrier** `sin((phi_f + phi_s)/2)` - which way, and how sure, averaged over
  both timescales. This is the signal.
* **envelope** `cos((phi_f - phi_s)/2)` - how much the two timescales *agree*.
  It is `1` when they say the same thing and falls to `0` when they are fully
  opposed, which is exactly the state of a parameter that has just reversed
  against its own trend: the signature of bouncing across a ravine instead of
  descending it.

Because the clip holds `|phi_f - phi_s| <= pi`, the envelope lies in `[0, 1]` and
can only ever brake - never reverse, never amplify. Nothing here is a schedule:
the brake is read off the gradients, engages after an overshoot and releases once
the timescales re-agree.

What it costs, plainly: one extra EMA per parameter. Three slots against Adam's
two, +50% optimiser state.


Reducing to Adam
----------------
For small `r`, `sin((pi/2) r) -> (pi/2) r`, so

    step -> (pi/2) * (r_f + r_s) / 2

which is Adam's rule on the mean of the two moments, times `pi/2`. TwoSine is a
strict generalisation: it is Adam wherever Adam is confident of little, and
departs from it exactly where the SNR is large enough for the curvature of the
wave to matter. `test_two_sine_optimizer.py --check` verifies this numerically.

The `pi/2` is a real factor - at equal `lr` TwoSine steps ~1.57x further than
Adam near zero - so every comparison in the test script sweeps the learning rate
per arm and reports each arm at its own best. Comparing at one shared `lr` would
measure that constant and nothing else.


Past the quarter turn
---------------------
`clip(r, -1, 1)` is a choice, and it is the one thing here the source paper's
argument is against: it throws away everything past the first quarter of the
wave. `full_wave=True` removes the clip, letting `|r| > 1` push the phase past
`pi/2` so that the step comes *back down* - an over-confident direction is
distrusted rather than trusted maximally.

It also removes the guarantee above. Without the clip `|phi_f - phi_s|` can
exceed `pi`, so the envelope goes **negative** and *reverses* the step rather
than braking it; the benchmark measures it reaching `-0.55`. That is a different
optimiser, not a relaxed one, which is why the default is `False`.

Measured, it changes nothing in 113 of 120 cells - `|r| > 1` is rare enough that
the clip almost never binds - and where it does bind it is worse. For this
application the rest of the wave is not being thrown away: there is nothing out
there to keep. See `../Experiments/README.md` for the numbers.
"""

from __future__ import annotations

import math

__all__ = ["TwoSine", "Adam", "Ablation", "RULES", "build"]


class TwoSine:
    """Two sine waves, one per timescale, combined by interference.

    Parameters and gradients are flat lists of floats; `step` updates in place.
    """

    def __init__(self, n: int, lr: float = 0.01, *, beta_fast: float = 0.9,
                 beta_slow: float = 0.99, beta_sq: float = 0.999,
                 eps: float = 1e-8, full_wave: bool = False) -> None:
        if not 0.0 <= beta_fast < 1.0 or not 0.0 <= beta_slow < 1.0 or not 0.0 <= beta_sq < 1.0:
            raise ValueError("betas must lie in [0, 1)")
        self.lr = lr
        self.b_f, self.b_s, self.b_v = beta_fast, beta_slow, beta_sq
        self.eps = eps
        self.full_wave = full_wave
        self.t = 0
        self.m_f = [0.0] * n
        self.m_s = [0.0] * n
        self.v = [0.0] * n

    # -- the rule, isolated so the checks can call it on plain numbers --------
    @staticmethod
    def combine(r_f: float, r_s: float, full_wave: bool = False):
        """`(step, carrier, envelope)` from the two signal-to-noise ratios.

        `step` is exactly `(sin(phi_f) + sin(phi_s)) / 2` and exactly
        `carrier * envelope`; the two agree by the sum-to-product identity.
        """
        if not full_wave:
            r_f = 1.0 if r_f > 1.0 else (-1.0 if r_f < -1.0 else r_f)
            r_s = 1.0 if r_s > 1.0 else (-1.0 if r_s < -1.0 else r_s)
        phi_f = (math.pi / 2.0) * r_f
        phi_s = (math.pi / 2.0) * r_s
        carrier = math.sin(0.5 * (phi_f + phi_s))
        envelope = math.cos(0.5 * (phi_f - phi_s))
        return carrier * envelope, carrier, envelope

    def step(self, params: list[float], grads: list[float]) -> None:
        self.t += 1
        c_f = 1.0 - self.b_f ** self.t          # bias corrections
        c_s = 1.0 - self.b_s ** self.t
        c_v = 1.0 - self.b_v ** self.t
        for i, g in enumerate(grads):
            self.m_f[i] = self.b_f * self.m_f[i] + (1.0 - self.b_f) * g
            self.m_s[i] = self.b_s * self.m_s[i] + (1.0 - self.b_s) * g
            self.v[i] = self.b_v * self.v[i] + (1.0 - self.b_v) * g * g
            scale = math.sqrt(self.v[i] / c_v) + self.eps
            r_f = (self.m_f[i] / c_f) / scale
            r_s = (self.m_s[i] / c_s) / scale
            params[i] -= self.lr * self.combine(r_f, r_s, self.full_wave)[0]

    ENGAGED = 0.99
    """An envelope below this counts as the brake having engaged."""

    def diagnostics(self) -> tuple[float, float, float, float]:
        """`(mean |carrier|, mean envelope, min envelope, fraction engaged)`.

        The envelope is the whole claim, and its *mean* is the wrong statistic
        for it: the brake is meant to be transient, biting for a few steps after
        a parameter reverses against its own trend and letting go again, so an
        average over every parameter and every step washes it out. The minimum
        and the engaged fraction are what say whether it ever bit at all. If the
        fraction is zero the optimiser is Adam with extra state, and the
        benchmark should say so.
        """
        c_f = 1.0 - self.b_f ** max(self.t, 1)
        c_s = 1.0 - self.b_s ** max(self.t, 1)
        c_v = 1.0 - self.b_v ** max(self.t, 1)
        car = env = 0.0
        lo, hit = 1.0, 0
        for i in range(len(self.v)):
            scale = math.sqrt(self.v[i] / c_v) + self.eps
            _, c, e = self.combine((self.m_f[i] / c_f) / scale,
                                   (self.m_s[i] / c_s) / scale, self.full_wave)
            car += abs(c)
            env += e
            lo = min(lo, e)
            hit += e < self.ENGAGED
        n = max(len(self.v), 1)
        return car / n, env / n, lo, hit / n


class Adam:
    """Kingma & Ba, 2014. The reference every arm is measured against."""

    def __init__(self, n: int, lr: float = 0.001, *, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = [0.0] * n
        self.v = [0.0] * n

    def step(self, params: list[float], grads: list[float]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            params[i] -= self.lr * (self.m[i] / c1) / (math.sqrt(self.v[i] / c2) + self.eps)

    def diagnostics(self) -> tuple[float, float, float, float]:
        return (float("nan"),) * 4


class Ablation:
    """The controls, each of which can sink the claim.

    Every one holds the state and the timescales fixed and changes only how the
    two ratios are combined, so whatever separates them is the combining rule
    and not the extra memory.

    `two-ema-linear`  mean of the two ratios, clipped. No sine anywhere. If this
                      ties TwoSine, the second EMA bought the win and the waves
                      bought nothing.
    `one-sine`        a single wave on the fast ratio, two state slots. If this
                      ties, the *second* wave bought nothing.
    `sign-gate`       the fast ratio, hard-gated to zero when the two moments
                      disagree in sign. If this ties, the smooth envelope bought
                      nothing over an `if`.
    `slow-adam`       Adam with beta1 = beta_slow. Rules out "it is just a
                      longer first moment".
    """

    def __init__(self, n: int, lr: float = 0.01, *, rule: str, beta_fast: float = 0.9,
                 beta_slow: float = 0.99, beta_sq: float = 0.999, eps: float = 1e-8) -> None:
        if rule not in ("two-ema-linear", "one-sine", "sign-gate", "slow-adam"):
            raise ValueError(f"unknown rule {rule!r}")
        self.rule, self.lr, self.eps = rule, lr, eps
        self.b_f, self.b_s, self.b_v = beta_fast, beta_slow, beta_sq
        self.t = 0
        self.m_f = [0.0] * n
        self.m_s = [0.0] * n
        self.v = [0.0] * n

    def step(self, params: list[float], grads: list[float]) -> None:
        self.t += 1
        c_f = 1.0 - self.b_f ** self.t
        c_s = 1.0 - self.b_s ** self.t
        c_v = 1.0 - self.b_v ** self.t
        half_pi = math.pi / 2.0
        for i, g in enumerate(grads):
            self.m_f[i] = self.b_f * self.m_f[i] + (1.0 - self.b_f) * g
            self.m_s[i] = self.b_s * self.m_s[i] + (1.0 - self.b_s) * g
            self.v[i] = self.b_v * self.v[i] + (1.0 - self.b_v) * g * g
            scale = math.sqrt(self.v[i] / c_v) + self.eps
            mf, ms = self.m_f[i] / c_f, self.m_s[i] / c_s
            r_f, r_s = mf / scale, ms / scale
            cf = 1.0 if r_f > 1.0 else (-1.0 if r_f < -1.0 else r_f)
            cs = 1.0 if r_s > 1.0 else (-1.0 if r_s < -1.0 else r_s)
            if self.rule == "two-ema-linear":
                s = 0.5 * (cf + cs)
            elif self.rule == "one-sine":
                s = math.sin(half_pi * cf)
            elif self.rule == "sign-gate":
                s = cf if (mf >= 0.0) == (ms >= 0.0) else 0.0
            else:                                   # slow-adam
                s = cs
            params[i] -= self.lr * s

    def diagnostics(self) -> tuple[float, float, float, float]:
        return (float("nan"),) * 4


RULES = ("two-sine", "two-sine-full", "adam", "two-ema-linear", "one-sine",
         "sign-gate", "slow-adam")


def build(rule: str, n: int, lr: float, **kw):
    """One constructor for every arm, so the harness never special-cases."""
    if rule == "two-sine":
        return TwoSine(n, lr, **kw)
    if rule == "two-sine-full":
        return TwoSine(n, lr, full_wave=True, **kw)
    if rule == "adam":
        return Adam(n, lr, eps=kw.get("eps", 1e-8))
    return Ablation(n, lr, rule=rule, **kw)
