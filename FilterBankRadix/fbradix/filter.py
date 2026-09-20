"""Layer 1: the activation filter.

The whole of layer 1 is ``m`` activation units read for their **sign**::

    z_j = v_j . x                       one projection of the segment's features
    f_j = a_j sin(b_j (z_j - h_j)) + k_j    the unit's response
    s_j = [f_j > 0]                     one bit of the routing address

and the address ``r = sum_j s_j 2^j`` names one of the ``2^m`` expert trees in
:mod:`fbradix.bank`.  That is the ``sign`` code.  There is a second,
``argmax``, where the address is simply the unit with the largest response and
``m`` units address ``m`` experts - see :meth:`ActivationFilter.address` and
section 4.5 of ``DESIGN.md``, because which of the two is used turns out to
matter more than anything else in layer 1.  Layer 1 computes no representation that layer 2 consumes:
it computes an **address**, and the only thing that crosses the boundary is
which tree the text is handed to.  That is what "filter" means here - the
network is filtering its input into channels, not transforming it.

The address is a binary number read most-significant-last, which makes it a
*radix-2 prefix* over the units.  A tree of experts keyed by that prefix is the
same structure as the character radix trees it selects between, one level up -
the architecture is a radix tree of radix trees.

**The input.**  :func:`features` is a 16-bucket histogram of character
codepoints (bucket ``ord(c) >> 3``, so bucket 6 is the digits, bucket 5 is
``()*+,-./`` and buckets 12-15 are the lowercase letters), plus a constant
1.0 so each unit has a bias.  It is deliberately generic: no feature says
"this looks like code", and nothing in it is derived from the source labels
the experiment scores routing against.

**The update rule.**  There is no label saying which expert *should* have
received a segment, so the filter learns from the counterfactual:

1. find ``j*``, the bit whose response is closest to zero - the address's
   least-confident bit, and so the cheapest one to change;
2. compare the chosen expert with the neighbour across that single boundary,
   by how many bits each spends on the segment;
3. push ``f_j*(x)`` across zero when the neighbour was better, and away from
   zero when it was not, by a hinge whose step is scaled by the size of the
   difference.

Everything in that rule is local - one unit, one segment, one hinge, analytic
partials of the sine - and nothing back-propagates through the expert trees.
The advantage is the only thing the trees report back, and it is a number, not
a gradient.  ``DESIGN.md`` sections 5-7 are the argument; :meth:`hinge_grads`
is checked against finite differences by ``tests/test_fbradix.py``.
"""

from __future__ import annotations

import math
import random

from .activation import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_K,
    MIN_B,
    is_dead,
    sine_partials,
    tanh_partials,
)

NBUCKET = 16
"""Codepoint buckets: ``ord(c) >> 3`` folded into 16, covering ASCII exactly."""

DIM = NBUCKET + 1
"""Feature dimension: the histogram plus a constant 1.0 (the per-unit bias)."""

TAU = 0.25
"""Hinge margin: a bit is only left alone once ``|f_j| >= TAU`` on the right side."""

CLIP = 5.0
"""Element-wise gradient clip, as in ``radixnet.backend.PythonBackend.step``."""

ACT_RATE = 0.1
"""The activation parameters move at a tenth of the projection's rate.

Section 8 of `Research/SineWaveActivationFunction.md` asks for the tenth by
name, and `Experiments/ActivationFunctionTest/` finding 4 measured what happens
without it: at equal rates the units die along ``(a, b)`` instead of along
``x``.  A dead filter unit is a bit of the address stuck at a constant, so here
the cost of losing that tenth is paid in addressable experts.
"""

PARTIALS = {"sine": sine_partials, "monotone": tanh_partials}


def features(text: str) -> list[float]:
    """Return the 17-dim feature vector of ``text``: 16 codepoint buckets + bias.

    Each bucket holds the *fraction* of characters that fall in it, scaled by
    :data:`NBUCKET` so the entries average 1.0 and the projection starts at
    O(1) whatever the segment's length.  Empty text gives the zero histogram.
    """
    counts = [0] * NBUCKET
    n = 0
    for ch in text:
        counts[(ord(ch) >> 3) % NBUCKET] += 1
        n += 1
    if n == 0:
        return [0.0] * NBUCKET + [1.0]
    scale = float(NBUCKET) / n
    out = [c * scale for c in counts]
    out.append(1.0)
    return out


class ActivationFilter:
    """``m`` learnable units whose responses spell the address of an expert tree."""

    __slots__ = ("bits", "dim", "kind", "code", "rng", "v", "a", "b", "h", "k",
                 "act_rate", "updates", "flips")

    def __init__(
        self,
        bits: int = 3,
        dim: int = DIM,
        seed: int = 0,
        kind: str = "sine",
        spread: float = 1.0,
        act_rate: float = ACT_RATE,
        code: str = "sign",
    ) -> None:
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        if kind not in PARTIALS:
            raise ValueError(f"kind must be one of {sorted(PARTIALS)}, got {kind!r}")
        if code not in ("sign", "argmax"):
            raise ValueError(f"code must be 'sign' or 'argmax', got {code!r}")
        self.bits = int(bits)
        self.dim = int(dim)
        self.kind = kind
        self.code = code
        """How the responses become an address.

        ``sign``   the sign pattern of ``m`` units, so ``2^m`` addresses.
        ``argmax`` the largest response of ``m`` units, so ``m`` addresses.

        The difference is not a detail of encoding.  A sign-bit address makes
        every bit an independent **dichotomy of the data**: a partition is
        reachable only if it factors into ``m`` binary questions the filter can
        answer, and the joint answer is about as good as the product of the
        parts.  Measured on this corpus (``make probe``, three seeds): the
        three ways to split the four registers two-against-two are answerable
        at 0.797, 0.711 and 0.636 by one unit each, their two best multiply to
        0.57, and the sign code's supervised ceiling over all 24 assignments is
        0.693 - the same neighbourhood.  ``argmax`` has no such constraint:
        each expert owns a unit and the partition never has to factor.  It
        costs ``K`` units instead of ``log2 K``, and reaches 0.723, so the two
        codes end up close together for different reasons - and in the bank
        itself they are 2.887 against 2.908 bits/char, inside the spread.
        """
        self.act_rate = float(act_rate)
        """Rate of ``a, b, h, k`` relative to ``v``; 0 holds the wave fixed.

        Holding it fixed is the control the ``fixedwave`` arm runs, and it is
        worth two different things.  A filter unit can **die** along ``(a, b)``
        - a dead unit is a bit of the address stuck at a constant, so the bank
        silently loses half of its experts - and freezing the wave makes that
        impossible.  But freezing it also strands a *periodic* unit: measured,
        a sine filter with its wave fixed cannot be fitted to an argmax address
        at all (0.243 against a chance of 0.25), because the response wraps and
        the hinge pushes it into the next lobe - 100% of its units end up past
        their first peak - while the same filter with the wave free reaches
        0.723 and lowers ``b`` from 0.333 to 0.176.  What learning the wave
        mostly does in layer 1 is flatten it until it is monotone over the
        data's range, and in the bank it is worth 0.18 bits/char.
        """
        self.rng = random.Random(seed)
        s = spread / math.sqrt(dim)
        self.v = [[self.rng.uniform(-s, s) for _ in range(dim)] for _ in range(bits)]
        self.a = [DEFAULT_A] * bits
        self.b = [DEFAULT_B] * bits
        self.h = [DEFAULT_H] * bits
        self.k = [DEFAULT_K] * bits
        self.updates = 0
        self.flips = 0

    # -- forward -------------------------------------------------------------

    @property
    def n_experts(self) -> int:
        """``2 ** bits`` under the sign code, ``bits`` under argmax."""
        return (1 << self.bits) if self.code == "sign" else self.bits

    def project(self, x: list[float], j: int) -> float:
        """``z_j = v_j . x``."""
        vj = self.v[j]
        return sum(vi * xi for vi, xi in zip(vj, x))

    def responses(self, x: list[float]) -> list[float]:
        """``f_j(x)`` for every unit."""
        f = PARTIALS[self.kind]
        return [
            f(self.project(x, j), self.a[j], self.b[j], self.h[j], self.k[j])[0]
            for j in range(self.bits)
        ]

    def address(self, x: list[float]) -> tuple[int, list[float]]:
        """Return ``(route, responses)`` under this filter's :attr:`code`.

        ``sign``: bit ``j`` of the address is ``f_j > 0``.
        ``argmax``: the address *is* the index of the largest response.
        """
        resp = self.responses(x)
        if self.code == "argmax":
            best = 0
            for j in range(1, len(resp)):
                if resp[j] > resp[best]:
                    best = j
            return best, resp
        route = 0
        for j, fj in enumerate(resp):
            if fj > 0.0:
                route |= 1 << j
        return route, resp

    def route(self, x: list[float]) -> int:
        """The address alone."""
        return self.address(x)[0]

    @staticmethod
    def weakest_bit(resp: list[float]) -> int:
        """The bit nearest its own boundary - the cheapest one to change."""
        j = 0
        best = abs(resp[0])
        for i in range(1, len(resp)):
            v = abs(resp[i])
            if v < best:
                best = v
                j = i
        return j

    def calibrate(self, xs: list[list[float]]) -> list[float]:
        """Centre each unit on its own input: shift ``k_j`` so the median
        response is 0, and the bit therefore splits the data in half.

        Unsupervised (it reads the features, never the labels) and done once,
        before any training.  Without it a unit can start with every segment on
        one side of its boundary, which costs the address that bit outright:
        the bank would be born with half of its experts unreachable.  The
        frozen arm is calibrated too - it is part of drawing the filter, not
        part of learning it.
        """
        shifts = []
        for j in range(self.bits):
            vals = sorted(self.responses(x)[j] for x in xs)
            if not vals:
                shifts.append(0.0)
                continue
            n = len(vals)
            med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
            self.k[j] -= med
            shifts.append(med)
        return shifts

    # -- the hinge -----------------------------------------------------------

    def hinge_loss(self, x: list[float], j: int, y: float) -> float:
        """``max(0, TAU - y * f_j(x))``: zero once bit ``j`` sits ``TAU`` clear of
        the boundary on the side ``y`` asks for."""
        f = PARTIALS[self.kind]
        z = self.project(x, j)
        fj = f(z, self.a[j], self.b[j], self.h[j], self.k[j])[0]
        return max(0.0, TAU - y * fj)

    def hinge_grads(self, x: list[float], j: int, y: float) -> dict | None:
        """Gradients of :meth:`hinge_loss` w.r.t. ``v_j, a, b, h, k``.

        ``None`` when the hinge is inactive (the bit is already where it should
        be, with margin), which is also when every gradient is zero.
        """
        f = PARTIALS[self.kind]
        z = self.project(x, j)
        fj, dfdz, dfda, dfdb, dfdh, dfdk = f(z, self.a[j], self.b[j], self.h[j], self.k[j])
        if y * fj >= TAU:
            return None
        return {
            "loss": TAU - y * fj,
            "v": [-y * dfdz * xi for xi in x],
            "a": -y * dfda,
            "b": -y * dfdb,
            "h": -y * dfdh,
            "k": -y * dfdk,
        }

    def pair_loss(self, x: list[float], up: int, down: int) -> float:
        """``max(0, TAU - (f_up - f_down))``: the argmax hinge.

        The sign hinge asks a unit to be positive; this one asks unit ``up`` to
        beat unit ``down`` by a margin, which is the only thing an argmax
        address cares about.
        """
        f = PARTIALS[self.kind]
        fu = f(self.project(x, up), self.a[up], self.b[up], self.h[up], self.k[up])[0]
        fd = f(self.project(x, down), self.a[down], self.b[down], self.h[down], self.k[down])[0]
        return max(0.0, TAU - (fu - fd))

    def pair_grads(self, x: list[float], up: int, down: int) -> dict | None:
        """Gradients of :meth:`pair_loss` for both units; ``None`` if inactive.

        ``dL/df_up = -1`` and ``dL/df_down = +1``, each chained through its own
        unit's partials - so this is two one-unit updates in opposite
        directions, and still local.
        """
        f = PARTIALS[self.kind]
        pu = f(self.project(x, up), self.a[up], self.b[up], self.h[up], self.k[up])
        pd = f(self.project(x, down), self.a[down], self.b[down], self.h[down], self.k[down])
        if pu[0] - pd[0] >= TAU:
            return None
        out = {"loss": TAU - (pu[0] - pd[0])}
        for name, idx, part, sign in (("up", up, pu, -1.0), ("down", down, pd, 1.0)):
            out[name] = {
                "unit": idx,
                "v": [sign * part[1] * xi for xi in x],
                "a": sign * part[2],
                "b": sign * part[3],
                "h": sign * part[4],
                "k": sign * part[5],
            }
        return out

    def push_pair(self, x: list[float], up: int, down: int, lr: float) -> bool:
        """One argmax-hinge step: raise unit ``up``, lower unit ``down``."""
        g = self.pair_grads(x, up, down)
        self.updates += 1
        if g is None:
            return False
        for side in ("up", "down"):
            part = g[side]
            j = part["unit"]
            vj = self.v[j]
            for i, gi in enumerate(part["v"]):
                vj[i] -= lr * _clip(gi)
            alr = lr * self.act_rate
            self.a[j] -= alr * _clip(part["a"])
            nb = self.b[j] - alr * _clip(part["b"])
            self.b[j] = nb if nb > MIN_B else MIN_B
            self.h[j] -= alr * _clip(part["h"])
            self.k[j] -= alr * _clip(part["k"])
        self.flips += 1
        return True

    def push(self, x: list[float], j: int, y: float, lr: float) -> bool:
        """One hinge step on unit ``j`` towards sign ``y``.  True if it moved.

        ``v`` moves at ``lr``; ``a, b, h, k`` at :data:`ACT_RATE` of it, and
        ``b`` is floored at :data:`MIN_B` exactly as the expert trees floor it.
        """
        g = self.hinge_grads(x, j, y)
        self.updates += 1
        if g is None:
            return False
        vj = self.v[j]
        for i, gi in enumerate(g["v"]):
            vj[i] -= lr * _clip(gi)
        alr = lr * self.act_rate
        self.a[j] -= alr * _clip(g["a"])
        nb = self.b[j] - alr * _clip(g["b"])
        self.b[j] = nb if nb > MIN_B else MIN_B
        self.h[j] -= alr * _clip(g["h"])
        self.k[j] -= alr * _clip(g["k"])
        self.flips += 1
        return True

    # -- health --------------------------------------------------------------

    def dead_units(self) -> list[int]:
        """Units with ``|a*b| <= DEAD_EPS``: a response that cannot move, so a
        bit of the address stuck at whatever constant it holds."""
        return [j for j in range(self.bits) if is_dead(self.a[j], self.b[j])]

    def amplitude_frequency(self) -> list[float]:
        """``|a_j * b_j|`` per unit - the bound on ``|f'|`` that finding 4 tracks."""
        return [abs(self.a[j] * self.b[j]) for j in range(self.bits)]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable state (the RNG is not part of it)."""
        return {
            "bits": self.bits,
            "dim": self.dim,
            "kind": self.kind,
            "code": self.code,
            "act_rate": self.act_rate,
            "v": [list(row) for row in self.v],
            "a": list(self.a),
            "b": list(self.b),
            "h": list(self.h),
            "k": list(self.k),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationFilter":
        """Inverse of :meth:`to_dict`."""
        f = cls(bits=d["bits"], dim=d["dim"], kind=d.get("kind", "sine"),
                act_rate=d.get("act_rate", ACT_RATE), code=d.get("code", "sign"))
        f.v = [list(row) for row in d["v"]]
        f.a = list(d["a"])
        f.b = list(d["b"])
        f.h = list(d["h"])
        f.k = list(d["k"])
        return f

    def __repr__(self) -> str:
        return (
            f"ActivationFilter(bits={self.bits}, kind={self.kind!r}, "
            f"code={self.code!r}, experts={self.n_experts}, "
            f"dead={len(self.dead_units())})"
        )


def _clip(g: float, c: float = CLIP) -> float:
    """Element-wise clip to ``[-c, c]``."""
    return c if g > c else (-c if g < -c else g)
