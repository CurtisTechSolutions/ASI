"""A Q-learning network over the plane: one node per hertz, one strip at a time.

The picture is already the right shape for a network to read.  A **vertical
one-pixel strip** of it is a single moment of sound - every frequency at that
instant, one pixel each - so a strip taken from a plane whose rows are spaced
one hertz apart has exactly one pixel per node.  Two hundred hertz of range is
two hundred nodes.

Each node owns its own frequency and nothing else.  It sees the strip, and it
answers one question: *how loud will I be in the next strip?*

    strip t                      strip t+1
    ┌───┐                        ┌───┐
    │ ░ │  node 199 (199 Hz) ──▶ │ ░ │   its own next level
    │ █ │  node 198 (198 Hz) ──▶ │ █ │
    │ █ │        ...             │ ▓ │
    │ ▒ │  node   1 (1 Hz)   ──▶ │ ▒ │
    │ ░ │  node   0 (0 Hz)   ──▶ │ ░ │
    └───┘                        └───┘

That is a reinforcement learning problem, not a regression: the node is in a
**state** (how loud it is now, and what its neighbours are doing), it takes an
**action** (the level it claims comes next), and it earns a **reward** for
being close.  Q-learning then does what it always does::

    Q(s, a) <- Q(s, a) + lr * (reward + discount * max Q(s', a') - Q(s, a))

The ``discount`` term is why this is worth doing rather than fitting a curve:
a node is rewarded not only for the next strip but for the value of the state
it lands in, so it learns runs that hold together over time - a note that keeps
sounding, a harmonic that decays at the rate its fundamental decays - instead
of guessing each strip on its own.

Every node keeps its own table, so a node is genuinely a small independent
learner about one frequency.  What ties them together is the *context* part of
the state: a node also sees a summary of its neighbours in the same strip, so
a harmonic can notice that the band around it has gone quiet.

Once trained, :meth:`QNet.generate` runs the whole thing forward from a seed
strip and writes new strips - which is a new plane, which
:func:`audioimage.codec.decode` turns into sound.

What this does not do
---------------------
**It does not beat "the next strip is the same as this one".**  That baseline
scores 0.026 mean level error on a melody; the best arrangement measured here
scores 0.030, and an untrained network with :attr:`QConfig.warm_start` scores
exactly the baseline because holding *is* the baseline.  Training moves it the
wrong way.

That is not a bug to be tuned out, and six arrangements were measured before
writing it down:

======================================  ===============
change                                  mean level error
======================================  ===============
baseline: next strip = this strip        0.026
naming the level outright, discount 0.85 0.195
naming the change instead                0.063
...and holding when a state is unseen    0.033
...and 2 or 4 strips of history          0.034 - 0.035
...and pooling what nodes learn          0.043
...and an evidence threshold             0.030
======================================  ===============

The reason is in the data: **77% of node-steps do not change at all**, so a
predictor is graded almost entirely on noticing the 23% that do - and *when* a
node changes is decided by things a node cannot see.  A note stops because the
player stopped, which is visible in the envelope and in the rest of the
spectrum, not in one node's own level and a local average.  Adding temporal
history does not help because the missing information is not in this node's
past; it is in the other nodes' present.

So the honest use of this module today is :meth:`QNet.generate`, where there is
no baseline to lose to - persistence generates one frozen strip forever - and
where the discount finally means something, because the network is feeding on
its own output and its action really does decide the next state.  Even there,
nodes sample independently, so what comes out is textured rather than musical.
Making it musical needs coupling between nodes, which is the next thing to try
and is not here yet.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import os
import random
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import __version__

__all__ = ["QConfig", "QNet", "QLearnError", "plane_strips", "strips_plane"]

Strip = list[float]
"""One column of the plane: an intensity in ``[0, 1]`` per node, lowest frequency first."""


class QLearnError(ValueError):
    """A shape that does not fit the network, or a file that is not one of its models."""


@dataclass
class QConfig:
    """What the network is and how it learns."""

    f_min: int = 0
    f_max: int = 400
    """The frequency range.  One node per hertz, so this is also the node count."""
    levels: int = 16
    """How many loudness levels a pixel is rounded to - the size of the action space."""
    context: int = 6
    """How many neighbours either side a node can see.  0 makes every node blind to the rest."""
    context_buckets: int = 4
    """How coarsely that neighbourhood is summarised."""
    action_mode: str = "delta"
    """``delta``: a node names *how much it changes*.  ``level``: it names the level outright.

    ``delta`` is the default because 77% of node-steps do not change at all, so
    "stay as I am" should be one action a node can learn, not a different
    absolute answer in every state.  Measured on a melody, switching from
    ``level`` to ``delta`` halved the error (0.119 -> 0.063).
    """
    span: int = 3
    """With ``delta``, how far a node may move in one strip: actions are -span..+span."""
    learning_rate: float = 0.2
    discount: float = 0.0
    """How much a node cares about the state it lands in.

    Zero by default, and that is a statement about the problem rather than a
    tuning choice.  While *training* on a recording, a node's action does not
    affect what the next strip is - the recording does - so the discounted
    term is the same for every action it could have taken, and all it adds to
    the comparison is noise.  Measured: 0.85 scored 0.195, zero scored 0.119.

    It earns its place in :meth:`QNet.generate`, where the network feeds on its
    own output and the action really does decide the next state.  Set it when
    you care more about what the network dreams than about what it predicts.
    """
    epsilon: float = 0.2
    """How often it tries something other than its best guess, while training."""
    warm_start: float = 0.05
    """The value given to "no change" everywhere before training.

    Without it, a state the network has never met has an all-zero row, the
    argmax returns the first action, and the node confidently predicts the
    biggest possible drop - so an untrained network says *silence* rather than
    *carry on as you are*.  Measured: the warm start took the error from 0.063
    to 0.033, which is most of the distance to the persistence baseline.
    """
    epsilon_decay: float = 0.92
    epsilon_min: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        if self.f_max <= self.f_min:
            raise QLearnError(f"f_max must be above f_min, got {self.f_min}..{self.f_max}")
        if self.nodes > 1_000_000:
            raise QLearnError(f"{self.nodes} nodes is more than this will hold")
        if self.levels < 2:
            raise QLearnError(f"levels must be at least 2, got {self.levels}")
        if self.action_mode not in ("delta", "level"):
            raise QLearnError(f"action_mode must be delta or level, got {self.action_mode!r}")
        if self.span < 1:
            raise QLearnError(f"span must be >= 1, got {self.span}")
        if self.context < 0:
            raise QLearnError(f"context must be >= 0, got {self.context}")
        if self.context_buckets < 1:
            raise QLearnError(f"context_buckets must be >= 1, got {self.context_buckets}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise QLearnError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if not 0.0 <= self.discount < 1.0:
            raise QLearnError(f"discount must be in [0, 1), got {self.discount}")
        for name in ("epsilon", "epsilon_min"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise QLearnError(f"{name} must be in [0, 1], got {value}")

    @property
    def nodes(self) -> int:
        """One per hertz of range."""
        return int(self.f_max - self.f_min)

    @property
    def actions(self) -> int:
        """How many choices a node has in a strip."""
        return 2 * self.span + 1 if self.action_mode == "delta" else self.levels

    @property
    def hold(self) -> int:
        """The action that means "do not change" - the one an unseen state falls back to."""
        return self.span if self.action_mode == "delta" else 0

    @property
    def buckets(self) -> int:
        """The neighbourhood summary is only used when a node can see one."""
        return self.context_buckets if self.context > 0 else 1

    @property
    def states(self) -> int:
        """``own level x neighbourhood bucket``."""
        return self.levels * self.buckets

    @property
    def table_size(self) -> int:
        """How many numbers the whole network is."""
        return self.nodes * self.states * self.actions

    def to_dict(self) -> dict[str, Any]:
        """The settings, for a saved model."""
        return {
            "f_min": self.f_min, "f_max": self.f_max, "levels": self.levels,
            "action_mode": self.action_mode, "span": self.span,
            "context": self.context, "context_buckets": self.context_buckets,
            "learning_rate": self.learning_rate, "discount": self.discount,
            "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min, "warm_start": self.warm_start, "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# planes and strips
# ---------------------------------------------------------------------------


def plane_strips(plane: Sequence[Sequence[float]]) -> list[Strip]:
    """A plane (image row order, top first) -> its strips, each lowest frequency first.

    The plane is stored the way a picture is - row 0 at the top, which is the
    *highest* frequency - so every strip is flipped on the way out.  A node's
    index is then its frequency, which is the whole point of the arrangement.
    """
    rows = list(plane)
    if not rows:
        return []
    height = len(rows)
    width = len(rows[0])
    return [[rows[height - 1 - r][t] for r in range(height)] for t in range(width)]


def strips_plane(strips: Sequence[Strip]) -> list[list[float]]:
    """Strips -> a plane in image row order.  The inverse of :func:`plane_strips`."""
    if not strips:
        return []
    height = len(strips[0])
    return [[strip[height - 1 - r] for strip in strips] for r in range(height)]


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------


class QNet:
    """One Q-table per node; a node is one hertz.

    The table is ``nodes x states x actions`` of 32-bit floats, held in one flat
    buffer.  numpy runs the whole strip at once when it is installed; without
    it the same arithmetic runs node by node and gives identical results.
    """

    def __init__(self, config: QConfig | None = None) -> None:
        self.config = config or QConfig()
        self.trained_strips = 0
        self.epochs = 0
        self._rng = random.Random(self.config.seed)
        self._epsilon = self.config.epsilon
        self._np: Any = None
        cfg = self.config
        try:
            import numpy as np

            self._np = np
            self._q = np.zeros((cfg.nodes, cfg.states, cfg.actions), dtype=np.float32)
        except Exception:
            self._q = array("f", bytes(4 * cfg.table_size))
        self._warm()

    def _warm(self) -> None:
        """Give "no change" a head start, so an unseen state holds rather than drops."""
        cfg = self.config
        if cfg.warm_start == 0.0:
            return
        if self._np is not None:
            self._q[:, :, cfg.hold] = cfg.warm_start
            return
        for node in range(cfg.nodes):
            for state in range(cfg.states):
                self._q[(node * cfg.states + state) * cfg.actions + cfg.hold] = cfg.warm_start

    # -- properties --------------------------------------------------------

    @property
    def nodes(self) -> int:
        """How many nodes - one per hertz of the configured range."""
        return self.config.nodes

    @property
    def backend(self) -> str:
        return "numpy" if self._np is not None else "python"

    @property
    def epsilon(self) -> float:
        """How much it is still exploring."""
        return self._epsilon

    def describe(self) -> dict[str, Any]:
        """A summary for reports and for the page."""
        cfg = self.config
        return {
            "nodes": self.nodes,
            "hz_per_node": 1,
            "range": [cfg.f_min, cfg.f_max],
            "levels": cfg.levels,
            "states": cfg.states,
            "actions": cfg.actions,
            "action_mode": cfg.action_mode,
            "table_size": cfg.table_size,
            "memory_bytes": cfg.table_size * 4,
            "epochs": self.epochs,
            "trained_strips": self.trained_strips,
            "epsilon": round(self._epsilon, 5),
            "backend": self.backend,
            "visited": self.visited(),
        }

    def visited(self) -> float:
        """The fraction of the table that has ever been written to."""
        if self.config.table_size == 0:
            return 0.0
        if self._np is not None:
            return float((self._q != 0).sum()) / self.config.table_size
        return sum(1 for v in self._q if v != 0.0) / self.config.table_size

    # -- quantising --------------------------------------------------------

    def level_of(self, intensity: float) -> int:
        """An intensity in ``[0, 1]`` -> one of ``levels`` buckets."""
        n = self.config.levels
        if intensity <= 0.0:
            return 0
        if intensity >= 1.0:
            return n - 1
        return min(n - 1, int(intensity * n))

    def intensity_of(self, level: int) -> float:
        """A bucket -> the intensity at its centre."""
        return (level + 0.5) / self.config.levels

    def _check(self, strip: Sequence[float]) -> None:
        if len(strip) != self.nodes:
            raise QLearnError(
                f"a strip must be {self.nodes} pixels tall for {self.config.f_min}-{self.config.f_max} Hz, got {len(strip)}"
            )

    # -- the state of every node in one strip ------------------------------

    def _levels(self, strip: Sequence[float]) -> list[int]:
        self._check(strip)
        return [self.level_of(v) for v in strip]

    def _states(self, levels: Sequence[int]) -> list[int]:
        """``own level * buckets + neighbourhood bucket`` for every node.

        The neighbourhood is a running mean over ``+/- context`` rows, taken
        from a prefix sum so the whole strip costs one pass however wide the
        window is.
        """
        cfg = self.config
        n = len(levels)
        if cfg.context <= 0:
            return list(levels)
        prefix = [0] * (n + 1)
        for i, level in enumerate(levels):
            prefix[i + 1] = prefix[i] + level
        buckets = cfg.buckets
        top = cfg.levels - 1
        out = [0] * n
        for i in range(n):
            lo = max(0, i - cfg.context)
            hi = min(n, i + cfg.context + 1)
            mean = (prefix[hi] - prefix[lo]) / (hi - lo)
            bucket = min(buckets - 1, int(mean / (top + 1e-9) * buckets)) if top else 0
            out[i] = levels[i] * buckets + bucket
        return out

    # -- reading the table -------------------------------------------------

    def _best(self, states: Sequence[int]) -> list[int]:
        """The highest-valued action for every node, ties going to the lowest level."""
        cfg = self.config
        if self._np is not None:
            np = self._np
            rows = self._q[np.arange(self.nodes), np.asarray(states, dtype=np.intp)]
            return rows.argmax(axis=1).tolist()
        actions = cfg.actions
        out = [0] * self.nodes
        q = self._q
        for node in range(self.nodes):
            base = (node * cfg.states + states[node]) * actions
            best = 0
            best_v = q[base]
            for a in range(1, actions):
                v = q[base + a]
                if v > best_v:
                    best, best_v = a, v
            out[node] = best
        return out

    def _max_q(self, states: Sequence[int]) -> list[float]:
        """The value of the best action in each node's next state."""
        cfg = self.config
        if self._np is not None:
            np = self._np
            rows = self._q[np.arange(self.nodes), np.asarray(states, dtype=np.intp)]
            return rows.max(axis=1).tolist()
        actions = cfg.actions
        q = self._q
        out = [0.0] * self.nodes
        for node in range(self.nodes):
            base = (node * cfg.states + states[node]) * actions
            out[node] = max(q[base : base + actions])
        return out

    def _sample(self, states: Sequence[int], temperature: float) -> list[int]:
        """Pick an action per node, softly - higher temperature, looser grip."""
        if temperature <= 0.0:
            return self._best(states)
        cfg = self.config
        rng = self._rng
        out = [0] * self.nodes
        if self._np is not None:
            np = self._np
            rows = self._q[np.arange(self.nodes), np.asarray(states, dtype=np.intp)].astype(np.float64)
            rows = rows - rows.max(axis=1, keepdims=True)
            weights = np.exp(rows / temperature)
            totals = weights.sum(axis=1, keepdims=True)
            probabilities = np.divide(weights, totals, out=np.full_like(weights, 1.0 / cfg.actions), where=totals > 0)
            picks = probabilities.cumsum(axis=1)
            draws = np.array([rng.random() for _ in range(self.nodes)])[:, None]
            return (picks < draws).sum(axis=1).clip(0, cfg.actions - 1).tolist()
        q = self._q
        for node in range(self.nodes):
            base = (node * cfg.states + states[node]) * cfg.actions
            row = [q[base + a] for a in range(cfg.actions)]
            top = max(row)
            weights = [math.exp((v - top) / temperature) for v in row]
            total = sum(weights)
            if total <= 0:
                out[node] = rng.randrange(cfg.actions)
                continue
            draw = rng.random() * total
            running = 0.0
            for a, w in enumerate(weights):
                running += w
                if running >= draw:
                    out[node] = a
                    break
            else:  # pragma: no cover - float drift on the last bucket
                out[node] = cfg.actions - 1
        return out

    def _apply(self, levels: Sequence[int], actions: Sequence[int]) -> list[int]:
        """Turn each node's chosen action into the level it is claiming."""
        cfg = self.config
        if cfg.action_mode == "level":
            return list(actions)
        top = cfg.levels - 1
        span = cfg.span
        return [min(top, max(0, levels[i] + actions[i] - span)) for i in range(len(levels))]

    # -- learning ----------------------------------------------------------

    def _update(
        self,
        states: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        next_best: Sequence[float],
    ) -> None:
        """One Bellman step for every node at once."""
        cfg = self.config
        lr, discount = cfg.learning_rate, cfg.discount
        if self._np is not None:
            np = self._np
            index = np.arange(self.nodes)
            s = np.asarray(states, dtype=np.intp)
            a = np.asarray(actions, dtype=np.intp)
            target = np.asarray(rewards, dtype=np.float32) + discount * np.asarray(next_best, dtype=np.float32)
            current = self._q[index, s, a]
            self._q[index, s, a] = current + lr * (target - current)
            return
        q = self._q
        actions_n = cfg.actions
        for node in range(self.nodes):
            i = (node * cfg.states + states[node]) * actions_n + actions[node]
            current = q[i]
            q[i] = current + lr * (rewards[node] + discount * next_best[node] - current)

    def train_pair(self, strip: Sequence[float], nxt: Sequence[float], explore: bool = True) -> dict[str, float]:
        """Learn from one strip and the strip that actually followed it."""
        cfg = self.config
        levels = self._levels(strip)
        targets = self._levels(nxt)
        states = self._states(levels)
        next_states = self._states(targets)

        chosen = self._best(states)
        if explore and self._epsilon > 0.0:
            rng = self._rng
            span = cfg.actions
            for node in range(self.nodes):
                if rng.random() < self._epsilon:
                    chosen[node] = rng.randrange(span)

        claimed = self._apply(levels, chosen)
        top = cfg.levels - 1
        rewards = [1.0 - abs(claimed[i] - targets[i]) / top for i in range(self.nodes)]
        # the discount is zero unless the caller asked for it: while training on
        # a recording the action cannot change what comes next (see QConfig)
        future = self._max_q(next_states) if cfg.discount else [0.0] * self.nodes
        self._update(states, chosen, rewards, future)

        exact = sum(1 for i in range(self.nodes) if claimed[i] == targets[i])
        return {
            "reward": sum(rewards) / self.nodes,
            "exact": exact / self.nodes,
            "error": sum(abs(claimed[i] - targets[i]) for i in range(self.nodes)) / (self.nodes * top),
        }

    def train(
        self,
        strips: Sequence[Strip],
        epochs: int = 4,
        progress: Callable[[int, int, dict[str, float]], None] | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, float]]:
        """Walk the strips left to right, ``epochs`` times over.

        Exploration decays after every pass, so early epochs wander and later
        ones settle.  Returns one summary per epoch.
        """
        if len(strips) < 2:
            raise QLearnError(f"training needs at least two strips, got {len(strips)}")
        if epochs < 1:
            raise QLearnError(f"epochs must be >= 1, got {epochs}")
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            totals = {"reward": 0.0, "exact": 0.0, "error": 0.0}
            pairs = 0
            for t in range(len(strips) - 1):
                if stop is not None and stop():
                    break
                stats = self.train_pair(strips[t], strips[t + 1])
                for key in totals:
                    totals[key] += stats[key]
                pairs += 1
                self.trained_strips += 1
            summary = {key: (value / pairs if pairs else 0.0) for key, value in totals.items()}
            summary["epsilon"] = self._epsilon
            summary["epoch"] = epoch + 1
            history.append(summary)
            self.epochs += 1
            self._epsilon = max(self.config.epsilon_min, self._epsilon * self.config.epsilon_decay)
            if progress is not None:
                progress(epoch + 1, epochs, summary)
            if stop is not None and stop():
                break
        return history

    # -- using it ----------------------------------------------------------

    def predict(self, strip: Sequence[float], temperature: float = 0.0) -> Strip:
        """The strip this one says comes next."""
        levels = self._levels(strip)
        states = self._states(levels)
        actions = self._best(states) if temperature <= 0 else self._sample(states, temperature)
        return [self.intensity_of(level) for level in self._apply(levels, actions)]

    def evaluate(self, strips: Sequence[Strip]) -> dict[str, float]:
        """How well it predicts, without learning anything from the attempt."""
        if len(strips) < 2:
            raise QLearnError(f"evaluating needs at least two strips, got {len(strips)}")
        top = self.config.levels - 1
        exact = 0.0
        error = 0.0
        for t in range(len(strips) - 1):
            levels = self._levels(strips[t])
            got = self._apply(levels, self._best(self._states(levels)))
            want = self._levels(strips[t + 1])
            exact += sum(1 for i in range(self.nodes) if got[i] == want[i]) / self.nodes
            error += sum(abs(got[i] - want[i]) for i in range(self.nodes)) / (self.nodes * top)
        pairs = len(strips) - 1
        return {"exact": exact / pairs, "error": error / pairs, "reward": 1.0 - error / pairs}

    def generate(
        self,
        seed: Sequence[Strip],
        steps: int = 64,
        temperature: float = 0.0,
        keep_seed: bool = True,
    ) -> list[Strip]:
        """Run forward from ``seed``, writing ``steps`` new strips.

        Each strip it produces becomes the state it reads next, so the network
        is listening to itself - which is what makes this generation rather
        than prediction.  ``temperature`` above 0 samples instead of always
        taking the best action, which is what stops it settling into a drone.
        """
        if not seed:
            raise QLearnError("generating needs at least one strip to start from")
        if steps < 1:
            raise QLearnError(f"steps must be >= 1, got {steps}")
        out = [list(s) for s in seed] if keep_seed else []
        current = list(seed[-1])
        for _ in range(steps):
            current = self.predict(current, temperature)
            out.append(list(current))
        return out

    # -- files -------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> str:
        """Write the model as one gzipped JSON file."""
        if self._np is not None:
            raw = self._q.astype(self._np.float32).tobytes()
        else:
            raw = self._q.tobytes()
        document = {
            "tool": f"audioimage {__version__}",
            "kind": "qnet",
            "v": 1,
            "config": self.config.to_dict(),
            "epochs": self.epochs,
            "trained_strips": self.trained_strips,
            "epsilon": self._epsilon,
            "q": base64.b64encode(gzip.compress(raw, 6)).decode("ascii"),
        }
        parent = os.path.dirname(os.path.abspath(str(path)))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(document, fh)
        return str(path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "QNet":
        """Read a model back."""
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                document = json.load(fh)
        except (OSError, ValueError, EOFError) as exc:
            raise QLearnError(f"that is not a saved network: {exc}") from None
        if document.get("kind") != "qnet":
            raise QLearnError("that file is not a qnet model")
        try:
            net = cls(QConfig(**document["config"]))
        except TypeError as exc:
            raise QLearnError(f"that model was saved by a different version: {exc}") from None
        raw = gzip.decompress(base64.b64decode(document["q"]))
        if len(raw) != net.config.table_size * 4:
            raise QLearnError(
                f"the saved table is {len(raw) // 4} numbers, but these settings need {net.config.table_size}"
            )
        if net._np is not None:
            net._q = net._np.frombuffer(raw, dtype=net._np.float32).reshape(net._q.shape).copy()
        else:
            net._q = array("f")
            net._q.frombytes(raw)
        net.epochs = int(document.get("epochs", 0))
        net.trained_strips = int(document.get("trained_strips", 0))
        net._epsilon = float(document.get("epsilon", net.config.epsilon))
        return net

    def __repr__(self) -> str:
        cfg = self.config
        return f"QNet(nodes={self.nodes} [{cfg.f_min}-{cfg.f_max}Hz], levels={cfg.levels}, epochs={self.epochs})"
