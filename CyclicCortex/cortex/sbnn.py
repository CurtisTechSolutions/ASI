"""The Self-Building Neural Network. One per region.

Grows hidden units on a loss PLATEAU, and grows inputs/outputs when a joining
game brings vocabulary the region lacks.

The growth invariant: immediately after growing hidden units or inputs, the
network computes EXACTLY what it computed before, for every input. New hidden
units enter with zero OUTGOING weight; new inputs enter with zero INCOMING
weight. So growth is free at the moment it happens and the network only changes
as the new capacity earns its way in. Asserted in tests/test_sbnn.py.

Two output heads per GREN/DESIGN.md 22.6:
    p_valid   is this action legal?   (sigmoid, trained from the rules)
    grade     how good is it?         (linear, trained from outcome)
"""
import math, random

class SBNN:
    def __init__(self, ni, nh=24, seed=0, act="tanh", b=1.0, a=-1.0, h=0.0, k=0.0):
        self.rng = random.Random(seed)
        self.ni, self.nh, self.act, self.b = ni, nh, act, b
        self.a, self.h, self.k = a, h, k
        s = 1.0 / math.sqrt(max(1, ni))
        self.W1 = [[self.rng.uniform(-s, s) for _ in range(nh)] for _ in range(ni)]
        self.b1 = [0.0] * nh
        self.Wv = [self.rng.uniform(-1, 1) / math.sqrt(nh) for _ in range(nh)]
        self.bv = 0.0
        self.Wg = [self.rng.uniform(-1, 1) / math.sqrt(nh) for _ in range(nh)]
        self.bg = 0.0
        self.loss = 1.0; self.best = None; self.since = 0; self.seen = 0
        self.warmup = 400; self.patience = 1200; self.min_improve = 0.01
        self.max_hidden = 256; self.growth = 4
        self.log = []

    # ---------------------------------------------------------------- forward
    def f(self, z):
        """``f(z) = a*sin(b*(z-h)) + k`` -- Research/SineWaveActivationFunction.md 4.

        The author's formula written out in full, so all four knobs of the wave
        are present rather than folded into a constant. At the defaults
        ``a=-1, h=0, k=0`` this is ``-sin(b*z)``, which is exactly what this net
        computed before; only ``b`` diverges from the paper's 1/3, deliberately
        and for a measured reason (DESIGN.md 8, Experiments/NeuralCompression/FINDINGS.md 5).
        """
        if self.act == "tanh": return math.tanh(z)
        return self.a * math.sin(self.b * (z - self.h)) + self.k
    def df(self, z, fz):
        """``df/dz = a*b*cos(b*(z-h))``; ``fz`` is ``f(z)``, which only tanh reuses."""
        if self.act == "tanh": return 1.0 - fz * fz
        return self.a * self.b * math.cos(self.b * (z - self.h))

    def forward(self, x):
        zh = list(self.b1)
        for i in range(min(len(x), self.ni)):
            xi = x[i]
            if xi == 0.0: continue
            Wi = self.W1[i]
            for h in range(self.nh): zh[h] += Wi[h] * xi
        ah = [self.f(z) for z in zh]
        zv = self.bv + sum(self.Wv[h] * ah[h] for h in range(self.nh))
        gr = self.bg + sum(self.Wg[h] * ah[h] for h in range(self.nh))
        pv = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, zv))))
        return zh, ah, pv, gr

    def predict(self, x):
        _, _, pv, gr = self.forward(x)
        return pv, gr

    # --------------------------------------------------------------- learning
    def step(self, x, valid=None, grade=None, lr=0.05, clip=5.0):
        """Either head may be None: validity and grade are trained from different
        signals arriving at different times, and are never required together."""
        zh, ah, pv, gr = self.forward(x)
        dv = (pv - valid) if valid is not None else 0.0
        dg = (gr - grade) if grade is not None else 0.0
        for d in (dv, dg):
            if d > clip or d < -clip: pass
        dv = max(-clip, min(clip, dv)); dg = max(-clip, min(clip, dg))
        gh = [0.0] * self.nh
        for h in range(self.nh):
            gh[h] = (dv * self.Wv[h] + dg * self.Wg[h]) * self.df(zh[h], ah[h])
            if valid is not None: self.Wv[h] -= lr * dv * ah[h]
            if grade is not None: self.Wg[h] -= lr * dg * ah[h]
        if valid is not None: self.bv -= lr * dv
        if grade is not None: self.bg -= lr * dg
        for h in range(self.nh): self.b1[h] -= lr * gh[h]
        for i in range(min(len(x), self.ni)):
            xi = x[i]
            if xi == 0.0: continue
            Wi = self.W1[i]
            for h in range(self.nh): Wi[h] -= lr * gh[h] * xi
        sq = dv * dv + dg * dg
        self._note(sq)
        return sq

    def _note(self, sq):
        self.loss = 0.99 * self.loss + 0.01 * sq
        self.seen += 1
        if self.best is None or self.loss < self.best * (1.0 - self.min_improve):
            self.best = self.loss; self.since = 0
        else:
            self.since += 1

    def plateaued(self):
        """Past warmup and the running loss has stopped improving. NOT a
        gradient-magnitude test: Experiments/NeuralCompression/FINDINGS.md 11 measured that
        magnitude cannot separate stuck from converged from still starting."""
        return self.seen > self.warmup and self.since > self.patience

    # ----------------------------------------------------------------- growth
    def grow_hidden(self, k=None):
        k = k or self.growth
        if self.nh + k > self.max_hidden: return 0
        s = 1.0 / math.sqrt(max(1, self.ni))
        for i in range(self.ni):
            self.W1[i].extend(self.rng.uniform(-s, s) for _ in range(k))
        self.b1.extend([0.0] * k)
        self.Wv.extend([0.0] * k)          # zero OUTGOING: the identity
        self.Wg.extend([0.0] * k)
        self.nh += k
        self.since = 0; self.best = self.loss
        self.log.append({"grew": "hidden", "by": k, "to": self.nh, "at": self.seen})
        return k

    def grow_inputs(self, k):
        if k <= 0: return 0
        for _ in range(k):
            self.W1.append([0.0] * self.nh)   # zero INCOMING: the identity
        self.ni += k
        self.log.append({"grew": "inputs", "by": k, "to": self.ni, "at": self.seen})
        return k

    def maybe_grow(self):
        if self.plateaued(): return self.grow_hidden()
        return 0

    # ---------------------------------------------------------- serialisation
    def to_dict(self):
        return {"ni": self.ni, "nh": self.nh, "act": self.act,
                "a": self.a, "b": self.b, "h": self.h, "k": self.k,
                "W1": self.W1, "b1": self.b1, "Wv": self.Wv, "bv": self.bv,
                "Wg": self.Wg, "bg": self.bg,
                "loss": self.loss, "best": self.best, "since": self.since,
                "seen": self.seen, "log": self.log,
                "warmup": self.warmup, "patience": self.patience,
                "min_improve": self.min_improve, "max_hidden": self.max_hidden,
                "growth": self.growth}

    @classmethod
    def from_dict(cls, d):
        # a/h/k default to the paper's values so checkpoints written before the
        # formula was spelled out still load to the same function.
        n = cls(d["ni"], d["nh"], act=d["act"], b=d["b"],
                a=d.get("a", -1.0), h=d.get("h", 0.0), k=d.get("k", 0.0))
        for k in ("W1","b1","Wv","bv","Wg","bg","loss","best","since","seen","log",
                  "warmup","patience","min_improve","max_hidden","growth"):
            setattr(n, k, d[k])
        return n

    def shape(self): return {"ni": self.ni, "nh": self.nh, "params":
                             self.ni * self.nh + self.nh + 2 * self.nh + 2}
