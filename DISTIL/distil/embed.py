"""Text -> vector. The layer the memory is built out of.

Two backends, and the choice between them is not a fallback hierarchy so much as
an admission about what is actually being retrieved.

`HashEmbedder` is lexical: the signed hashing trick over words and character
4-grams, sublinear term frequency, inverse document frequency learned online.
It has no idea that "compile" and "build" are related. What it does have is
determinism, zero latency, zero cost, no network, and exact reproducibility
across processes and machines -- and for the traffic this system actually
stores (identifiers, error strings, tool names, file paths, stack frames) the
lexical match IS the right match. `TypeError: unhashable type` should retrieve
by those exact tokens, not by vibe.

`ProviderEmbedder` is semantic, costs a round trip, and earns it on prose.

The default is `HashEmbedder` wrapped so a provider can take over per-call.
Both produce unit vectors of the same width so the two are mixable in one
store -- though mixing them inside one *comparison* is meaningless, which is
why every trace records the embedder that produced it and recall refuses to
score across backends (DESIGN 4.4).

On hashing: Python's builtin `hash()` for strings is salted per process, so a
vector written on Monday would not match the same text embedded on Tuesday.
Every hash here is blake2b. This is the single most important line in the file.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .vector import Vector, normalise, zeros

DIMS = 512
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def _h(token: str, salt: str = "") -> int:
    """Stable 64-bit digest. blake2b because it is in the standard library, it is
    fast, and unlike `hash()` it does not change between interpreter runs."""
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=salt.encode()[:16])
    return int.from_bytes(d.digest(), "big")


def tokenise(text: str) -> list[str]:
    """Words, numbers and punctuation as separate tokens, plus character 4-grams
    of every long word. The n-grams are what make `ValueErrpr` land near
    `ValueError`: a typo, a truncation or an unseen identifier still shares most
    of its substrings with the thing it is a variant of."""
    lowered = text.lower()
    words = _WORD.findall(lowered)
    grams: list[str] = []
    for w in words:
        if len(w) > 5:
            grams.extend("#" + w[i:i + 4] for i in range(len(w) - 3))
    return words + grams


class Vocabulary:
    """Document frequencies, learned online.

    IDF is usually computed over a fixed corpus. There is no fixed corpus here --
    the corpus is whatever the system has lived through so far -- so the weights
    move as it learns. Old vectors are not rewritten when they move: re-embedding
    the whole store on every observation costs O(n) per write for a second-order
    correction to a ranking. The drift is bounded (DESIGN 4.3) and the store
    records the vocabulary epoch so a rebuild is always available.
    """

    def __init__(self) -> None:
        self.docs = 0
        self.df: Counter[str] = Counter()
        self.epoch = 0

    def observe(self, tokens: list[str]) -> None:
        self.docs += 1
        self.epoch += 1
        for t in set(tokens):
            self.df[t] += 1

    def idf(self, token: str) -> float:
        # smoothed idf, never negative, never zero -- an unseen token is maximally
        # informative rather than an error
        return math.log((1.0 + self.docs) / (1.0 + self.df.get(token, 0))) + 1.0

    def to_json(self) -> dict:
        return {"docs": self.docs, "epoch": self.epoch, "df": dict(self.df)}

    @classmethod
    def from_json(cls, d: dict) -> "Vocabulary":
        v = cls()
        v.docs = d.get("docs", 0)
        v.epoch = d.get("epoch", 0)
        v.df = Counter(d.get("df", {}))
        return v


class HashEmbedder:
    """Signed hashing trick. One pass, no training, no allocation beyond the
    output vector.

    At 512 dimensions the collision noise floor sits around cosine 0.17 for
    short texts, while genuine lexical matches land at 0.25 and above. That gap
    is workable but it is narrow, and callers comparing *one or two words*
    against a longer description should not rely on cosine alone (`frame.py`
    matches surface forms exactly for this reason). Widening DIMS pushes the
    floor down linearly in cost; provider embeddings remove the issue entirely.

    Signed because unsigned hashing makes every collision constructive: two
    unrelated tokens landing in the same bucket always inflate the similarity.
    With a sign bit drawn from an independent digest, collisions cancel in
    expectation, so the error is zero-mean instead of a systematic bias toward
    "everything is related to everything" (Weinberger et al.'s result, and it
    matters more here than in classification because the scores are compared to
    each other rather than thresholded).
    """

    name = "hash-v1"

    def __init__(self, dims: int = DIMS, vocab: Vocabulary | None = None) -> None:
        self.dims = dims
        self.vocab = vocab or Vocabulary()

    def embed(self, text: str, learn: bool = True) -> Vector:
        tokens = tokenise(text or "")
        if not tokens:
            return zeros(self.dims)
        if learn:
            self.vocab.observe(tokens)
        counts = Counter(tokens)
        v = zeros(self.dims)
        for token, tf in counts.items():
            weight = (1.0 + math.log(tf)) * self.vocab.idf(token)   # sublinear tf
            digest = _h(token)
            bucket = digest % self.dims
            sign = 1.0 if (_h(token, "sign") >> 63) & 1 else -1.0
            v[bucket] += sign * weight
        return normalise(v)

    def embed_batch(self, texts: list[str], learn: bool = True) -> list[Vector]:
        return [self.embed(t, learn=learn) for t in texts]


class ProviderEmbedder:
    """Semantic embeddings from whatever model is reachable.

    Degrades rather than raises: if the provider is down, times out, or returns
    the wrong width, the call falls through to the lexical embedder and the
    resulting trace is tagged with the backend that actually produced it. A
    memory layer that stops remembering because a service is unreachable is
    worse than one that remembers lexically for an hour.
    """

    def __init__(self, provider, fallback: HashEmbedder | None = None, dims: int = DIMS) -> None:
        self.provider = provider
        self.fallback = fallback or HashEmbedder(dims)
        self.dims = dims
        self.failures = 0

    @property
    def name(self) -> str:
        return f"{getattr(self.provider, 'name', 'provider')}-embed"

    def embed(self, text: str, learn: bool = True) -> Vector:
        self.fallback.embed(text, learn=learn)      # keep the lexical vocabulary warm
        try:
            raw = self.provider.embed([text])[0]
        except Exception:
            self.failures += 1
            return self.fallback.embed(text, learn=False)
        return normalise(_fit(raw, self.dims))

    def embed_batch(self, texts: list[str], learn: bool = True) -> list[Vector]:
        return [self.embed(t, learn=learn) for t in texts]


def _fit(raw: list[float], dims: int) -> Vector:
    """Force a provider vector to the store's width.

    Truncation would throw away the tail; zero-padding would leave most of the
    space empty. Folding by addition -- position i of the output accumulates
    every input position congruent to i -- is a random projection with the same
    collision-cancellation argument as the hashing trick, and it preserves inner
    products in expectation. Same-width vectors take the identity path.
    """
    if len(raw) == dims:
        return [float(x) for x in raw]
    out = zeros(dims)
    for i, x in enumerate(raw):
        out[i % dims] += float(x)
    return out
