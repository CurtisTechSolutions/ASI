"""A Self-Building Neural Network on the sine-wave activation, with the 2NRL inversion.

Two of the repository's papers meet in this file.

``Research/SineWaveActivationFunction.md`` gives the activation::

    f(x) = a * sin(b * (x - h)) + k          defaults a = -1, b = 1/3, h = k = 0

with all four parameters **learnable per neuron**, and
``Research/2NRL.md`` §4.3 gives the operator that makes 2NRL possible at all:
an inversion that is exact, cheap, involutive and order-reversing.  The sine's
sign parameter is what makes negating a unit a *parameter* change rather than
a structural one.

The network part is the one from ``TwoNRL_CartPole/sinenet.py`` - same layer,
same parity argument, same Adam - plus the thing that makes it *self-building*:
:meth:`SineNet.grow_hidden`.

Growth, and why it is an identity here
--------------------------------------
``SBNN_RNN_ActivationFunction/main.py`` grows by copying the old weights into
the top-left corner of larger matrices and drawing the new rows *and* columns
from ``N(0, 0.1)``, so a growth step **perturbs** the function the network
computes.  That is fine when growth is the only thing happening.  It is not
fine here, for two reasons that are specific to 2NRL:

1. Phase 1 spends its whole budget building a representation of the failure.
   Phase 2 negates *that*.  A growth step that perturbs the function damages
   what there is to negate, and the damage is not something phase 3 was
   designed to repair.
2. The inversion is exact only if every path from input to output flips sign
   exactly once.  A new unit that already contributes to the output has to be
   correct under that bookkeeping the instant it appears.

So growth here is **identity-preserving**, in ``GREN/gren/sbnn.py``'s sense: a
new hidden unit enters with random *incoming* weights (so it is not born
symmetry-locked to its siblings) and **zero outgoing** weights, so it
contributes exactly nothing until it has learned something.  The network
computes the same function for every input the instant after growing, and
zeros stay zeros under the sign flips, so the inversion stays exact.
``test_sbnn.py`` asserts both.

Inverting a feed-forward stack: the parity correction
-----------------------------------------------------
In the radix graph an edge's score is a *product*, ``w * f(z_p) * f(z_c)``, so
flipping every weight and every amplitude flips the score exactly once.  A
feed-forward stack **composes** instead, and there the same literal rule
cancels itself: with ``h = k = 0`` the sine is odd, so per layer
``z' = -z`` and ``f'(z') = (-a) sin(b(-z)) = f(z)`` - unchanged.  Flipping
everything is a no-op (:meth:`invert_literal` demonstrates it).

:meth:`invert` breaks parity exactly once.  Writing ``s`` for the sign
entering a layer, ``t`` for the sign forced onto its pre-activation and
``sigma`` for its output sign::

    W' = t*s*W      bias' = t*bias
    t = +1:   a' =  sigma*a    h' =  h    k' = sigma*k
    t = -1:   a' = -sigma*a    h' = -h    k' = sigma*k

``t = -s`` flips every weight matrix (the research's rule) and ``sigma = -1``
from the break layer on negates the output once.  The result is exact to
machine precision: ``net_inverted(x) == -net(x)``, so the move the failure
policy most wanted is the one it now least wants.
"""

from __future__ import annotations

import numpy as np

DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0

WEIGHT_PARAMS = ("W", "bias")
ACT_PARAMS = ("a", "b", "h", "k")
MIN_B = 1e-3          # radixnet/activation.py's floor: b must stay a frequency


class SineDense:
    """Dense layer followed by a per-neuron ``a * sin(b * (x - h)) + k``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 a: float = DEFAULT_A, b: float = DEFAULT_B,
                 h: float = DEFAULT_H, k: float = DEFAULT_K) -> None:
        self.defaults = (float(a), float(b), float(h), float(k))
        self.W = self._init_weights(n_in, n_out, rng)
        self.bias = np.zeros(n_out)
        self.a = np.full(n_out, float(a))
        self.b = np.full(n_out, float(b))
        self.h = np.full(n_out, float(h))
        self.k = np.full(n_out, float(k))
        self._cache: dict | None = None

    def _init_weights(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        # Glorot, widened by the activation's own slope |a*b| = 1/3 at the
        # origin so the forward signal does not shrink layer over layer.
        a, b = self.defaults[0], self.defaults[1]
        limit = np.sqrt(6.0 / (n_in + n_out)) / abs(a * b)
        return rng.uniform(-limit, limit, size=(n_in, n_out))

    @property
    def params(self) -> tuple[str, ...]:
        return WEIGHT_PARAMS + ACT_PARAMS

    @property
    def n_in(self) -> int:
        return self.W.shape[0]

    @property
    def n_out(self) -> int:
        return self.W.shape[1]

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

    # ----------------------------------------------------------- growth

    def grow_out(self, n: int, rng: np.random.Generator) -> None:
        """Add ``n`` neurons to this layer: random incoming weights, fresh waves.

        A new unit's wave enters at the defaults, so a unit built on a stall is
        born as exactly ``-sin(x/3)`` and learns its own ``a, b, h, k`` from
        there - the same rule as ``SBNN_RNN_ActivationFunction``.
        """
        a, b, h, k = self.defaults
        limit = np.sqrt(6.0 / (self.n_in + self.n_out + n)) / abs(a * b)
        self.W = np.hstack([self.W, rng.uniform(-limit, limit, size=(self.n_in, n))])
        self.bias = np.concatenate([self.bias, np.zeros(n)])
        self.a = np.concatenate([self.a, np.full(n, a)])
        self.b = np.concatenate([self.b, np.full(n, b)])
        self.h = np.concatenate([self.h, np.full(n, h)])
        self.k = np.concatenate([self.k, np.full(n, k)])
        self._cache = None

    def grow_in(self, n: int) -> None:
        """Accept ``n`` more inputs, with **zero** weight on every one of them.

        This is what makes growth an identity: whatever the new units upstream
        compute, this layer ignores it until training moves these weights off
        zero.  Zero also survives the inversion's sign flips unchanged.
        """
        self.W = np.vstack([self.W, np.zeros((n, self.n_out))])
        self._cache = None


class SineNet:
    """A stack of :class:`SineDense` layers: the SBNN, and the 2NRL inversion."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, **act) -> None:
        self.sizes = list(sizes)
        self.layers = [SineDense(sizes[i], sizes[i + 1], rng, **act)
                       for i in range(len(sizes) - 1)]
        self.inverted = False
        self.growth_log: list[dict] = []

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

    def score(self, x: np.ndarray) -> np.ndarray:
        """Forward with a single output unit, returned flat. No training cache."""
        return self.forward(x, train=False)[:, 0]

    # ---------------------------------------------------------------- 2NRL

    def invert(self, parity_break: int = -1) -> None:
        """2NRL step 2: negate the network's output exactly once.

        Exact: after this call ``forward(x) == -forward_before(x)`` to machine
        precision, so ``argmax`` becomes ``argmin``.  It is an involution, and
        the hidden activations are bit-for-bit unchanged - the inversion does
        not damage what the negative phase learned, it re-reads it with the
        opposite sign.
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
        activation (``h = k = 0``) this is a **no-op**.  Not used by the
        experiment.
        """
        for layer in self.layers:
            layer.W *= -1.0
            layer.bias *= -1.0
            layer.a *= -1.0

    # -------------------------------------------------------------- growth

    def grow_hidden(self, n: int = 4, rng: np.random.Generator | None = None,
                    layer: int = 0, max_hidden: int = 256) -> int:
        """Widen one hidden layer by ``n`` units, preserving the function exactly.

        Returns the number of units actually added (0 if the cap is reached).
        ``layer`` indexes the hidden layer, so ``0`` is the first one.
        """
        if rng is None:
            rng = np.random.default_rng()
        if layer >= len(self.layers) - 1:
            raise ValueError("there is no hidden layer at index %d" % layer)
        current = self.layers[layer].n_out
        n = min(n, max_hidden - current)
        if n <= 0:
            return 0
        self.layers[layer].grow_out(n, rng)
        self.layers[layer + 1].grow_in(n)
        self.sizes[layer + 1] = self.layers[layer].n_out
        self.growth_log.append({"layer": layer, "added": n, "to": self.sizes[layer + 1]})
        return n

    @property
    def hidden_sizes(self) -> list[int]:
        return self.sizes[1:-1]

    def n_params(self) -> int:
        return sum(int(getattr(l, p).size) for l in self.layers for p in l.params)

    # ----------------------------------------------------------- utilities

    def copy(self) -> "SineNet":
        clone = object.__new__(SineNet)
        clone.sizes = list(self.sizes)
        clone.inverted = self.inverted
        clone.growth_log = list(self.growth_log)
        clone.layers = []
        for layer in self.layers:
            new = object.__new__(SineDense)
            new.defaults = layer.defaults
            for name in layer.params:
                setattr(new, name, getattr(layer, name).copy())
            new._cache = None
            clone.layers.append(new)
        return clone

    def act_report(self) -> list[dict[str, float]]:
        """Mean learned activation parameters per layer (how far they moved)."""
        return [{name: float(np.mean(getattr(layer, name))) for name in ACT_PARAMS}
                for layer in self.layers]

    def save(self, path: str) -> None:
        blob = {"sizes": np.array(self.sizes), "inverted": np.array([self.inverted])}
        for i, layer in enumerate(self.layers):
            for name in layer.params:
                blob[f"{i}.{name}"] = getattr(layer, name)
        np.savez(path, **blob)

    @staticmethod
    def load(path: str) -> "SineNet":
        blob = np.load(path)
        sizes = [int(v) for v in blob["sizes"]]
        net = SineNet(sizes, np.random.default_rng(0))
        net.inverted = bool(blob["inverted"][0])
        for i, layer in enumerate(net.layers):
            for name in layer.params:
                setattr(layer, name, blob[f"{i}.{name}"].copy())
        return net


class Adam:
    """Adam with a separate, smaller learning rate for the activation parameters.

    Mirrors ``TrainConfig``: ``lr`` for weights and biases, ``act_lr`` for
    ``a, b, h, k`` (the default there is ``lr / 10``, which is what 2NRL's
    positive phase uses).  :meth:`resync` grows the moment estimates alongside
    the network, so a growth step does not throw away the optimiser state of
    the units that were already there.
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

    def resync(self) -> None:
        """Resize the moments to the network's current shape, keeping what fits."""
        for i, layer in enumerate(self.net.layers):
            for name in layer.params:
                target = getattr(layer, name)
                for store in (self.m, self.v):
                    old = store[i][name]
                    if old.shape == target.shape:
                        continue
                    new = np.zeros_like(target)
                    idx = tuple(slice(0, s) for s in old.shape)
                    new[idx] = old
                    store[i][name] = new

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
            # b is a frequency; radixnet/activation.py floors it so a unit
            # cannot collapse into a constant.
            np.clip(layer.b, MIN_B, None, out=layer.b)
