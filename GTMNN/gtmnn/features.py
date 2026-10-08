"""Alphabet, signed feature hashing, and receptive fields.

The hash MUST NOT be Python's hash(): that is salted per process
(PYTHONHASHSEED), so a saved model would predict differently after a restart.
This is a correctness requirement, not a performance one, and the test suite
asserts it across a subprocess boundary.
"""
import random
from array import array

UNK, END = 0, 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a(data, seed=0):
    """64-bit FNV-1a, mixed with the seed. Deterministic across processes."""
    h = (FNV_OFFSET ^ (seed * FNV_PRIME)) & MASK64
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


class Alphabet:
    """Symbol ids. 0 is UNK, 1 is END, real characters start at 2.

    `fit` is incremental and never reorders existing ids, so an alphabet may be
    extended after training without invalidating a saved model -- a micro's
    repertoire holds symbol ids, and renumbering them would silently repoint
    every repertoire slot in the population."""

    def __init__(self, symbols=()):
        self.symbols = ["<unk>", "<end>"]
        self.index = {"<unk>": UNK, "<end>": END}
        for s in symbols: self.add(s)

    def add(self, ch):
        i = self.index.get(ch)
        if i is not None: return i
        i = len(self.symbols)
        self.symbols.append(ch); self.index[ch] = i
        return i

    def fit(self, texts):
        for t in texts:
            for ch in t: self.add(ch)
        return self

    def __len__(self): return len(self.symbols)
    def encode(self, ch): return self.index.get(ch, UNK)
    def decode(self, i): return self.symbols[i] if 0 <= i < len(self.symbols) else "<unk>"

    def to_dict(self): return {"symbols": self.symbols}

    @classmethod
    def from_dict(cls, d):
        a = cls()
        a.symbols = list(d["symbols"])
        a.index = {s: i for i, s in enumerate(a.symbols)}
        return a


class FeatureHasher:
    """The last `context` characters -> a fixed `dim`-dimensional vector.

    No vocabulary, no growth, no lookup table. Positional offsets are part of
    the token, so "ab" one character back and "ab" five characters back are
    different features.

    L2 normalisation is what makes the pre-activation scale PREDICTABLE: values
    land in roughly [-1, 1], a micro sums R of them through weights in
    [-0.5, 0.5], so E|z| is known in advance -- which is precisely what lets the
    sine frequency be initialised against it (activation.py). It is not a reason
    to sit near zero: that is the sine's linear region and a population of
    near-linear units is a linear model."""

    def __init__(self, dim=256, context=12, ngrams=(1, 2, 3), seed=0, decay=0.9):
        self.dim, self.context = int(dim), int(context)
        self.ngrams, self.seed, self.decay = tuple(ngrams), int(seed), float(decay)

    def transform(self, context, out=None):
        if out is None: out = array("d", [0.0]) * self.dim
        else:
            for i in range(self.dim): out[i] = 0.0
        ctx = context[-self.context:] if self.context else ""
        L = len(ctx)
        dim, seed, decay = self.dim, self.seed, self.decay
        for n in self.ngrams:
            if n > L: continue
            # o = 0 is the n-gram ending at the last character.
            for o in range(0, L - n + 1):
                end = L - o
                tok = f"{n}:{o}:{ctx[end - n:end]}"
                hv = fnv1a(tok.encode("utf-8"), seed)
                out[hv % dim] += (1.0 if (hv >> 32) & 1 else -1.0) * (decay ** o)
        norm = 0.0
        for v in out: norm += v * v
        if norm > 0.0:
            norm = norm ** 0.5
            for i in range(dim): out[i] /= norm
        return out

    def to_dict(self):
        return {"dim": self.dim, "context": self.context, "ngrams": list(self.ngrams),
                "seed": self.seed, "decay": self.decay}

    @classmethod
    def from_dict(cls, d):
        return cls(d["dim"], d["context"], tuple(d["ngrams"]), d["seed"], d.get("decay", 0.9))


def sample_receptive_field(dim, size, rng):
    """`size` DISTINCT indices in [0, dim), sorted ascending."""
    return sorted(rng.sample(range(dim), min(size, dim)))


def coverage(pool, dim=None):
    """How many living micros read each feature index.

    A feature no living micro reads is a blind spot. With N=4096, R=8, F=256 the
    chance of one at birth is ~3e-57; the logic exists for EVOLVED populations,
    where culling can strip coverage from a region the corpus stopped
    exercising."""
    dim = dim or pool.F
    out = [0] * dim
    R = pool.R
    for i in range(pool.n):
        if not pool.alive[i]: continue
        base = i * R
        for r in range(R):
            idx = pool.rf[base + r]
            if 0 <= idx < dim: out[idx] += 1
    return out


def coverage_gaps(pool, dim=None):
    return sum(1 for c in coverage(pool, dim) if c == 0)
