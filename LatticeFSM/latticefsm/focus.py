"""The learned focus: a function of the input that says which node of the central vertical vector a run starts from.

Focus is a number in ``[0, 1]`` (``geometry.py``).  A machine with
``learn_focus`` does not take it as a setting: it reads it off the input,
through a :class:`FocusLearner` -

    focus(text) = sigmoid(w . x(text))

- where ``x`` is the bias, the text's length as ``n / (n + 6)``, the share of
each symbol in it, and which symbol it starts and ends with: ``2 + 3A``
features, 41 on the default machine.  The weights start at a bias of
``-3`` and nothing else, so an untrained learner's focus is ``0.047``: the top
node, as with no focus at all.

It learns from the machine's credit, as the edges do.  A run that learns
explores: it draws ``f = mu + explore * N(0, 1)`` and starts from the node
``f`` picks, clipped to the range.  When the run is credited ``c``,

    w += rate * (c - baseline) * (f - mu) / explore^2 * mu * (1 - mu) * x
    baseline += 0.05 * (c - baseline)

- the gradient of a Gaussian policy's log-likelihood through the sigmoid,
with the credit measured against its running mean: a run credited better
than usual pulls the mean toward the focus it drew, a worse one pushes it
away.  The unclipped draw is what is learned from, so the range's edges do
not bias it.  Every weight is clipped to ``[-8, 8]``.  A quiet run reads the
mean and does not explore.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["FOCUS_BIAS", "FocusLearner"]

FOCUS_BIAS = -3.0
"""The bias an untrained learner starts with: sigmoid(-3) = 0.047, the top node."""

_LIMIT = 8.0


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class FocusLearner:
    """``focus(text) = sigmoid(w . x(text))``, learned from credit."""

    alphabet: list[str]
    weights: list[float] = field(default_factory=list)
    rate: float = 0.05
    explore: float = 0.15
    baseline: float = 0.0
    """The running mean of the credit the learner has seen."""

    def __post_init__(self) -> None:
        n = 2 + 3 * len(self.alphabet)
        if not self.weights:
            self.weights = [FOCUS_BIAS] + [0.0] * (n - 1)
        if len(self.weights) != n:
            raise ValueError(f"a focus learner over {len(self.alphabet)} symbols has {n} weights, got {len(self.weights)}")
        if self.rate < 0 or self.explore < 0:
            raise ValueError("rate and explore must be >= 0")
        self._index = {s: i for i, s in enumerate(self.alphabet)}

    def features(self, symbols: Sequence[str]) -> list[float]:
        """The bias, ``n / (n + 6)``, each symbol's share, the first symbol and the last, one-hot."""
        A = len(self.alphabet)
        n = len(symbols)
        x = [1.0, n / (n + 6.0)] + [0.0] * (3 * A)
        for s in symbols:
            x[2 + self._index[s]] += 1.0 / n
        if n:
            x[2 + A + self._index[symbols[0]]] = 1.0
            x[2 + 2 * A + self._index[symbols[-1]]] = 1.0
        return x

    def mean(self, x: Sequence[float]) -> float:
        return _sigmoid(sum(w * xi for w, xi in zip(self.weights, x)))

    def predict(self, symbols: Sequence[str]) -> float:
        """The focus the learner gives this input, without exploring."""
        return self.mean(self.features(symbols))

    def choose(self, symbols: Sequence[str], gauss, explore: bool) -> tuple[float, float, list[float]]:
        """The draw (unclipped), its mean, and the features: the mean plus noise when ``explore``, else the mean.
        The node it picks is :func:`clip` of the draw."""
        x = self.features(symbols)
        mu = self.mean(x)
        f = mu + gauss() * self.explore if explore and self.explore > 0 else mu
        return f, mu, x

    @staticmethod
    def clip(f: float) -> float:
        return min(1.0, max(0.0, f))

    def learn(self, credit: float, f: float, mu: float, x: Sequence[float]) -> None:
        """Move the weights with ``credit`` against the baseline: toward the draw ``f`` when it did better than
        usual, away when worse; then move the baseline."""
        if credit == 0.0 or self.explore <= 0:
            return
        advantage = credit - self.baseline
        self.baseline += 0.05 * (credit - self.baseline)
        g = self.rate * advantage * (f - mu) / (self.explore * self.explore) * mu * (1.0 - mu)
        self.weights = [max(-_LIMIT, min(_LIMIT, w + g * xi)) for w, xi in zip(self.weights, x)]

    def swap_symbols(self, a: int, b: int) -> None:
        """Keep the learner's weights on the symbols they belong to when two symbols trade places."""
        A = len(self.alphabet)
        self.alphabet[a], self.alphabet[b] = self.alphabet[b], self.alphabet[a]
        for base in (2, 2 + A, 2 + 2 * A):
            self.weights[base + a], self.weights[base + b] = self.weights[base + b], self.weights[base + a]
        self._index = {s: i for i, s in enumerate(self.alphabet)}

    def to_dict(self) -> dict:
        return {"alphabet": list(self.alphabet), "weights": list(self.weights), "rate": self.rate,
                "explore": self.explore, "baseline": self.baseline}

    @classmethod
    def from_dict(cls, d: dict) -> FocusLearner:
        return cls(list(d["alphabet"]), [float(w) for w in d["weights"]], float(d["rate"]), float(d["explore"]),
                   float(d.get("baseline", 0.0)))
