"""A Self-Building Neural Network whose INPUT and OUTPUT layers grow by discovery.

GREN starts knowing nothing: no features, and no refusal kinds at all. Both
vocabularies are discovered by probing, so both layers grow.

  inputs   a new FEATURE NAME appears (a new oracle's feature language, §14)
  outputs  a new REFUSAL CODE is seen for the first time
  hidden   the running loss plateaus (NeuralCompression/FINDINGS.md §11 measured
           that gradient magnitude cannot separate stuck from converged from
           still-starting; a plateau can)

All three growths are IDENTITIES -- the network computes exactly what it computed
before, for every input, the instant after growing:

  hidden   new units enter with zero OUTGOING weight
  inputs   new inputs enter with zero INCOMING weight
  outputs  a softmax is the awkward case, and the obvious fix is wrong. A new
           unit with zero weight and zero bias takes exp(0)=1 of the mass and
           disturbs every existing class. Giving it a large NEGATIVE bias is not
           enough either: a softmax cares only about RELATIVE logits, and after
           training the existing logits may sit far below any fixed constant, so
           the new class becomes the maximum and takes everything (measured:
           0.244 of the distribution moved). It enters with ZERO weights and a
           bias MARGIN below the largest existing bias, so its logit is constant
           and provably negligible whatever the input.

The head predicts a distribution over {LEGAL} u {known refusal codes}, which is
what makes the real expected-information-gain objective computable (§17.3)
rather than approximated by a histogram.
"""
import math, random

MARGIN = 40.0        # how far below the existing logits a new class enters

class GrowingSBNN:
    def __init__(self, nh=24, seed=0, act="tanh", b=1.0):
        self.rng = random.Random(seed)
        self.nh, self.act, self.b = nh, act, b
        self.ni = 0; self.no = 0
        self.inputs = {}             # feature name -> index
        self.outputs = {}            # label (code or "LEGAL") -> index
        self.out_names = []
        self.W1 = []                              # ni x nh, grows by input
        self.b1 = [0.0] * nh
        self.W2 = [[] for _ in range(nh)]         # nh x no, grows by output
        self.b2 = []
        self.loss = 1.0; self.best = None; self.since = 0; self.seen = 0
        self.warmup = 300; self.patience = 900; self.min_improve = 0.01
        self.max_hidden = 256; self.growth = 4
        self.log = []

    # ----------------------------------------------------------------- growth
    def _preserving(self):
        """Identity-preserving growth is right for GROWTH and wrong for
        INITIALISATION. Zero-weight rows and columns are what make growth free,
        but a network whose W1 AND W2 both start at zero is symmetry-locked:
        ah = tanh(0) = 0 zeroes the W2 update, which zeroes the hidden gradient,
        which zeroes the W1 update. Nothing but the biases can ever move, and the
        loss sits exactly where it started (measured: 0.7003, forever).

        So during warmup, when there is no learned function to protect, new rows
        and columns are initialised randomly. After it, they enter as identities.
        """
        return self.seen > self.warmup

    def feature_index(self, name):
        i = self.inputs.get(name)
        if i is None:
            i = self.ni
            self.inputs[name] = i
            if self._preserving():
                self.W1.append([0.0] * self.nh)          # identity
            else:
                s = 1.0 / math.sqrt(max(1, self.ni + 1))
                self.W1.append([self.rng.uniform(-s, s) for _ in range(self.nh)])
            self.ni += 1
            self.log.append({"grew": "input", "name": name, "to": self.ni})
        return i

    def label_index(self, label):
        i = self.outputs.get(label)
        if i is None:
            i = self.no
            self.outputs[label] = i
            self.out_names.append(label)
            if self._preserving():
                for h in range(self.nh): self.W2[h].append(0.0)      # identity
            else:
                s = 1.0 / math.sqrt(max(1, self.nh))
                for h in range(self.nh): self.W2[h].append(self.rng.uniform(-s, s))
            # The margin exists to PRESERVE a learned distribution, so it
            # applies only once there is one. Labels discovered during warmup
            # enter on equal footing; applying the margin from the first step
            # instead started the second label 40 logits below the first and
            # cost ~0.5 accuracy on a trivially separable task, because `seen`
            # is already 1 by the time the second label appears.
            self.b2.append((max(self.b2) - MARGIN)
                           if (self.b2 and self._preserving()) else 0.0)
            self.no += 1
            self.log.append({"grew": "output", "name": label, "to": self.no})
        return i

    def grow_hidden(self, k=None):
        """Always an identity: a new hidden unit's OUTGOING weights are zero, so
        it contributes nothing until it learns, and its incoming weights are
        random so it is not born symmetry-locked to its siblings."""
        k = k or self.growth
        if self.nh + k > self.max_hidden: return 0
        s = 1.0 / math.sqrt(max(1, self.ni))
        for row in self.W1: row.extend(self.rng.uniform(-s, s) for _ in range(k))
        self.b1.extend([0.0] * k)
        for _ in range(k): self.W2.append([0.0] * self.no)   # zero OUTGOING
        self.nh += k
        self.since = 0; self.best = self.loss
        self.log.append({"grew": "hidden", "by": k, "to": self.nh, "at": self.seen})
        return k

    def plateaued(self):
        return self.seen > self.warmup and self.since > self.patience

    def maybe_grow(self):
        return self.grow_hidden() if self.plateaued() else 0

    # ---------------------------------------------------------------- forward
    def _f(self, z):
        return math.tanh(z) if self.act == "tanh" else -math.sin(self.b * z)
    def _df(self, z, a):
        return 1.0 - a*a if self.act == "tanh" else -math.cos(self.b * z) * self.b

    def _vec(self, feats, grow=False):
        x = [0.0] * self.ni
        for name, v in feats.items():
            i = self.inputs.get(name)
            if i is None:
                if not grow: continue
                i = self.feature_index(name)
                x.append(0.0)
            x[i] = v
        return x

    def forward(self, feats, grow=False):
        x = self._vec(feats, grow)
        zh = list(self.b1)
        for i, xi in enumerate(x):
            if xi == 0.0: continue
            Wi = self.W1[i]
            for h in range(self.nh): zh[h] += Wi[h] * xi
        ah = [self._f(z) for z in zh]
        if self.no == 0: return x, zh, ah, []
        zo = list(self.b2)
        for h in range(self.nh):
            a = ah[h]
            if a == 0.0: continue
            W = self.W2[h]
            for o in range(self.no): zo[o] += W[o] * a
        m = max(zo); e = [math.exp(v - m) for v in zo]; t = sum(e) or 1.0
        return x, zh, ah, [v / t for v in e]

    def predict(self, feats):
        """Distribution over {LEGAL} u known codes. Unknown features are ignored
        rather than erroring -- a game may exercise something this net has never
        been shown, and dropping it is correct."""
        _, _, _, p = self.forward(feats, grow=False)
        return {n: p[i] for n, i in self.outputs.items()} if p else {}

    def p_legal(self, feats):
        p = self.predict(feats)
        return p.get("LEGAL", 0.5) if p else 0.5

    # --------------------------------------------------------------- learning
    def step(self, feats, label, lr=0.05, clip=5.0):
        """One cross-entropy step. Growing on demand: an unseen feature adds an
        input, an unseen label adds an output, and neither disturbs what the
        network already computes."""
        for name in feats: self.feature_index(name)
        t = self.label_index(label)
        x, zh, ah, p = self.forward(feats, grow=True)
        if not p: return 0.0
        d = [p[o] - (1.0 if o == t else 0.0) for o in range(self.no)]
        d = [max(-clip, min(clip, v)) for v in d]
        gh = [0.0] * self.nh
        for h in range(self.nh):
            W = self.W2[h]; acc = 0.0
            for o in range(self.no):
                acc += W[o] * d[o]
                W[o] -= lr * d[o] * ah[h]
            gh[h] = acc * self._df(zh[h], ah[h])
        for o in range(self.no): self.b2[o] -= lr * d[o]
        for h in range(self.nh): self.b1[h] -= lr * gh[h]
        for i, xi in enumerate(x):
            if xi == 0.0: continue
            Wi = self.W1[i]
            for h in range(self.nh): Wi[h] -= lr * gh[h] * xi
        loss = -math.log(max(1e-12, p[t]))
        self.seen += 1
        self.loss = 0.99 * self.loss + 0.01 * loss
        if self.best is None or self.loss < self.best * (1.0 - self.min_improve):
            self.best = self.loss; self.since = 0
        else: self.since += 1
        return loss

    def shape(self):
        return {"ni": self.ni, "nh": self.nh, "no": self.no,
                "params": self.ni*self.nh + self.nh + self.nh*self.no + self.no}

    def to_dict(self):
        return {"nh": self.nh, "ni": self.ni, "no": self.no, "act": self.act, "b": self.b,
                "inputs": self.inputs, "outputs": self.outputs, "out_names": self.out_names,
                "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "loss": self.loss, "seen": self.seen, "log": self.log}

    @classmethod
    def from_dict(cls, d):
        n = cls(nh=d["nh"], act=d["act"], b=d["b"])
        for k in ("ni","no","inputs","outputs","out_names","W1","b1","W2","b2",
                  "loss","seen","log"): setattr(n, k, d[k])
        return n
