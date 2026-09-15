"""A small feed-forward network built on the sine-wave activation function.

The activation is the one from ``Research/SineWaveActivationFunction.md``::

    f(x) = a * sin(b * (x - h)) + k          defaults a = -1, b = 1/3, h = k = 0

and, as in ``RadixCyclicNN/radixnet/activation.py``, all four parameters are
**learnable per neuron**: learning acts on the activation itself rather than
fighting the vanishing gradient through a fixed non-linearity.

The point of this module is the 2NRL operation :meth:`SineNet.invert` - the
step that turns "every win into a loss and every loss into a win".

Inverting a feed-forward stack
------------------------------
In the radix graph an edge's score is a *product*, ``w * f(z_p) * f(z_c)``.
``graph.invert()`` flips every edge weight ``w`` and every node amplitude
``a``; the two amplitude flips cancel inside the product and the edge score
flips exactly **once**.  The parity works out for free.

A feed-forward stack *composes* instead of multiplying, and there the same
literal rule cancels itself.  With ``h = k = 0`` the sine activation is odd,
so for one layer::

    z' = (-W)x + (-bias) = -z
    f'(z') = (-a) * sin(b * (-z)) = a * sin(b * z) = f(z)        <- unchanged

Every layer cancels and the network is returned unmodified: flipping
*everything* is a no-op (:meth:`invert_literal` demonstrates it).  The
inversion has to break parity exactly once.

:meth:`invert` does that.  Writing ``s`` for the sign of the signal entering
a layer, ``t`` for the sign we force onto its pre-activation and ``sigma``
for the sign of its output, a layer transforms exactly as::

    W' = t*s*W      bias' = t*bias
    t = +1:   a' =  sigma*a    h' =  h    k' = sigma*k
    t = -1:   a' = -sigma*a    h' = -h    k' = sigma*k

Choosing ``t = -s`` flips every weight matrix (the research's rule), and
choosing ``sigma = -1`` from the parity-break layer onwards negates the
output exactly once.  With the break on the last layer this reduces to:

* every layer flips ``W``, ``bias`` and the phase ``h``,
* every layer flips its amplitude ``a`` **except the last**,
* the last layer flips its offset ``k``.

The result is exact to machine precision, ``net_inverted(x) == -net(x)``, so
``argmax`` becomes ``argmin``: every action the network preferred it now
rejects.  The hidden activations are bit-for-bit unchanged - the inversion
does not damage what the negative phase learned, it only re-reads it with
the opposite sign.
"""

from __future__ import annotations

import numpy as np

DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0

WEIGHT_PARAMS = ("W", "bias")
ACT_PARAMS = ("a", "b", "h", "k")


class SineDense:
    """Dense layer followed by a per-neuron ``a * sin(b * (x - h)) + k``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 a: float = DEFAULT_A, b: float = DEFAULT_B,
                 h: float = DEFAULT_H, k: float = DEFAULT_K) -> None:
        # Glorot, widened by the activation's own slope |a*b| = 1/3 at the
        # origin so the forward signal does not shrink layer over layer.
        limit = np.sqrt(6.0 / (n_in + n_out)) / abs(a * b)
        self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.bias = np.zeros(n_out)
        self.a = np.full(n_out, float(a))
        self.b = np.full(n_out, float(b))
        self.h = np.full(n_out, float(h))
        self.k = np.full(n_out, float(k))
        self._cache: dict | None = None

    @property
    def params(self) -> tuple[str, ...]:
        return WEIGHT_PARAMS + ACT_PARAMS

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        z = x @ self.W + self.bias
        d = z - self.h
        u = self.b * d
        y = self.a * np.sin(u) + self.k
        if train:
            self._cache = {"x": x, "d": d, "u": u}
        return y

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Gradients for ``dL/dy``; returns ``(dL/dx, {param: grad})``.

        Partials of ``f = a*sin(b*(x-h)) + k`` (``u = b*(x-h)``)::

            df/dx = a*b*cos(u)   df/da = sin(u)   df/db = a*(x-h)*cos(u)
            df/dh = -a*b*cos(u)  df/dk = 1
        """
        c = self._cache
        if c is None:
            raise RuntimeError("backward() called without a training forward()")
        x, d, u = c["x"], c["d"], c["u"]
        cos_u, sin_u = np.cos(u), np.sin(u)
        slope = self.a * self.b * cos_u          # df/dz
        dz = dy * slope
        grads = {
            "W": x.T @ dz,
            "bias": dz.sum(axis=0),
            "a": (dy * sin_u).sum(axis=0),
            "b": (dy * self.a * d * cos_u).sum(axis=0),
            "h": (dy * -slope).sum(axis=0),
            "k": dy.sum(axis=0),
        }
        return dz @ self.W.T, grads


class SineNet:
    """A stack of :class:`SineDense` layers, and the 2NRL inversion."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, **act) -> None:
        self.sizes = list(sizes)
        self.layers = [SineDense(sizes[i], sizes[i + 1], rng, **act)
                       for i in range(len(sizes) - 1)]
        self.inverted = False

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, dy: np.ndarray) -> list[dict[str, np.ndarray]]:
        grads: list[dict[str, np.ndarray]] = []
        for layer in reversed(self.layers):
            dy, g = layer.backward(dy)
            grads.append(g)
        grads.reverse()
        return grads

    # ---------------------------------------------------------------- 2NRL

    def invert(self, parity_break: int = -1) -> None:
        """2NRL step 2: negate the network's output exactly once.

        Flips every weight matrix and every bias (the research's rule) and
        every activation amplitude *except* the parity-break layer, whose
        un-flipped amplitude lets the negation through to the output.  See
        the module docstring for the derivation.  Exact: after this call
        ``forward(x) == -forward_before(x)`` to machine precision.
        """
        n = len(self.layers)
        j = parity_break % n
        s = 1.0                                   # sign of the incoming signal
        for i, layer in enumerate(self.layers):
            t = -s                                # forces W' = t*s*W = -W
            sigma = -1.0 if i >= j else 1.0       # sign of this layer's output
            layer.W *= t * s
            layer.bias *= t
            if t > 0:
                layer.a *= sigma
                layer.k *= sigma
            else:
                layer.a *= -sigma
                layer.h *= -1.0
                layer.k *= sigma
            s = sigma
        self.inverted = not self.inverted

    def invert_literal(self) -> None:
        """The literal reading: flip every weight, bias and amplitude.

        Kept to demonstrate that on a feed-forward stack with an odd
        activation (``h = k = 0``) this is a **no-op** - see the module
        docstring.  Not used by the experiment.
        """
        for layer in self.layers:
            layer.W *= -1.0
            layer.bias *= -1.0
            layer.a *= -1.0

    # ----------------------------------------------------------- utilities

    def copy(self) -> "SineNet":
        clone = object.__new__(SineNet)
        clone.sizes = list(self.sizes)
        clone.inverted = self.inverted
        clone.layers = []
        for layer in self.layers:
            new = object.__new__(SineDense)
            for name in layer.params:
                setattr(new, name, getattr(layer, name).copy())
            new._cache = None
            clone.layers.append(new)
        return clone

    def act_report(self) -> list[dict[str, float]]:
        """Mean learned activation parameters per layer (how far they moved)."""
        return [{name: float(np.mean(getattr(layer, name))) for name in ACT_PARAMS}
                for layer in self.layers]


class Adam:
    """Adam with a separate, smaller learning rate for the activation parameters.

    Mirrors ``TrainConfig``: ``lr`` for weights and biases, ``act_lr`` for
    ``a, b, h, k`` (the default there is ``lr / 10``, which is what 2NRL's
    positive phase uses).
    """

    def __init__(self, net: SineNet, lr: float, act_lr: float | None = None,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 clip: float = 5.0) -> None:
        self.net = net
        self.lr = lr
        self.act_lr = lr / 10.0 if act_lr is None else act_lr
        self.beta1, self.beta2, self.eps, self.clip = beta1, beta2, eps, clip
        self.reset()

    def reset(self) -> None:
        """Drop the moment estimates (used after an inversion flips the signs)."""
        self.t = 0
        self.m = [{p: np.zeros_like(getattr(l, p)) for p in l.params} for l in self.net.layers]
        self.v = [{p: np.zeros_like(getattr(l, p)) for p in l.params} for l in self.net.layers]

    def step(self, grads: list[dict[str, np.ndarray]]) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i, layer in enumerate(self.net.layers):
            for name, g in grads[i].items():
                g = np.clip(g, -self.clip, self.clip)
                m = self.m[i][name] = self.beta1 * self.m[i][name] + (1 - self.beta1) * g
                v = self.v[i][name] = self.beta2 * self.v[i][name] + (1 - self.beta2) * g * g
                lr = self.act_lr if name in ACT_PARAMS else self.lr
                setattr(layer, name,
                        getattr(layer, name) - lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps))
