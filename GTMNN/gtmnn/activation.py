"""The parametric sine, a*sin(b*(x-h)) + k, learnable per neuron.

Same contract as RadixCyclicNN/radixnet/activation.py so a fix in one transfers
to the other by inspection -- with one deliberate divergence: DEFAULT_B is 1.0
here, not 1/3.

DESIGN.md 5.1 has the measurement. Reaching the first peak of sin(b*z) needs
|z| = pi/(2b), which at b=1/3 is 4.71. Features are L2-normalised so |x| ~ 1,
and with weights in [-0.5, 0.5] over R=8 inputs the pre-activation sits near
zero. Every neuron would operate in the LINEAR region of its sine, and a
population of near-linear units is a linear model however wide it is. Measured
on the task most favourable to a sine, b=1/3 is 17x worse than tanh and b=1.0
is 6x better.

The general rule: b ~ pi / (2 * E|z|). The initialisation is the trap, not the
parameterisation -- df/db = a(x-h)cos(b(x-h)) is small exactly when x is small,
so the gradient that would fix a bad frequency is suppressed by the same
condition that makes it wrong.
"""
import math

DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K = -1.0, 1.0, 0.0, 0.0


def sine_activation(x, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K):
    return a * math.sin(b * (x - h)) + k


def sine_derivative(x, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K):
    return a * b * math.cos(b * (x - h))


def sine_partials(x, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K):
    """(f, df/dx, df/da, df/db, df/dh, df/dk) -- exactly six floats.

    One call, because every one of them shares sin(u) and cos(u) and the hot
    loop computes all six at once or none of them."""
    d = x - h
    u = b * d
    s, c = math.sin(u), math.cos(u)
    return (a * s + k, a * b * c, s, a * d * c, -a * b * c, 1.0)


class SineActivation:
    """Convenience object for tests and docs. Never used in a hot loop -- the
    pool reads the four parameters out of its flat arrays directly."""
    __slots__ = ("a", "b", "h", "k")

    def __init__(self, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K):
        self.a, self.b, self.h, self.k = float(a), float(b), float(h), float(k)

    def __call__(self, x): return sine_activation(x, self.a, self.b, self.h, self.k)
    def derivative(self, x): return sine_derivative(x, self.a, self.b, self.h, self.k)
    def partials(self, x): return sine_partials(x, self.a, self.b, self.h, self.k)
    def inverted(self): return SineActivation(-self.a, self.b, self.h, self.k)

    def to_dict(self): return {"a": self.a, "b": self.b, "h": self.h, "k": self.k}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("a", DEFAULT_A), d.get("b", DEFAULT_B),
                   d.get("h", DEFAULT_H), d.get("k", DEFAULT_K))

    def __repr__(self):
        return f"SineActivation(a={self.a}, b={self.b}, h={self.h}, k={self.k})"


def suggested_b(mean_abs_z):
    """b ~ pi / (2 E|z|): put the first peak inside the actual operating range.

    DESIGN.md 5.1. Exposed so `bench` can report whether the realised spread of
    pre-activations matches what DEFAULT_B assumes, rather than assuming it."""
    return math.pi / (2.0 * mean_abs_z) if mean_abs_z > 1e-12 else DEFAULT_B
