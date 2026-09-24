"""How a training run walks its texts: the order, the curriculum, replay and early stopping.

``../SPEC-SearchAndTraining.md`` §3-6 is the contract; the Go
(``go/radixnet/training.go``) and Rust (``rust/src/training.rs``) ports keep it
number for number, which is why every rule below says how it rounds and how a
tie breaks.

A :class:`TrainingPlan` is built once per ``train`` call.  It says, epoch by
epoch, which of the run's texts to walk and in what order, which texts of the
model's :class:`ReplayBuffer` to rehearse after them, and whether the run
should stop.  Everything is off by default, and a plan that is off walks
every text in corpus order every epoch - the pass the model always had.

Nothing here draws a random number: the shuffle and the replay buffer order
their texts by SplitMix64 keys of the model's seed (:func:`shuffle_key`,
:func:`replay_key`), so the model's own generator - and every edge weight it
draws - is where it would have been.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

__all__ = [
    "ORDERS",
    "EarlyStop",
    "ReplayBuffer",
    "TrainingPlan",
    "check_plan",
    "paced",
    "replay_count",
    "replay_key",
    "shuffle_key",
    "splitmix64",
]

ORDERS = ("corpus", "shortest-first", "longest-first", "shuffle")
"""The orders a run can walk its texts in; ``"corpus"`` is the one it always had."""

_MASK = (1 << 64) - 1
_SHUFFLE_SALT = 0x53485546464C45  # "SHUFFLE"
_REPLAY_SALT = 0x5245504C4159  # "REPLAY"


def splitmix64(x: int) -> int:
    """The SplitMix64 finaliser on an unsigned 64-bit integer (wrapping arithmetic)."""
    z = (x + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def shuffle_key(seed: int, epoch: int, i: int) -> int:
    """Where text ``i`` goes in the shuffled order of epoch number ``epoch`` (smallest first)."""
    return splitmix64(splitmix64(splitmix64((seed & _MASK) ^ _SHUFFLE_SALT) ^ (epoch & _MASK)) ^ (i & _MASK))


def replay_key(seed: int, g: int) -> int:
    """The priority of the ``g``-th text ever offered to a replay buffer (the smallest are kept)."""
    return splitmix64(splitmix64((seed & _MASK) ^ _REPLAY_SALT) ^ (g & _MASK))


def check_plan(
    order: str = "corpus",
    curriculum: float = 1.0,
    replay: float = 0.0,
    replay_size: int | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
) -> None:
    """``ValueError`` for a training setting out of range (``../SPEC-SearchAndTraining.md`` §7)."""
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; expected one of: {', '.join(ORDERS)}")
    if not (0.0 < curriculum <= 1.0):
        raise ValueError(f"curriculum must lie in (0, 1], got {curriculum}")
    if not (replay >= 0.0) or not math.isfinite(replay):
        raise ValueError(f"replay must be a finite number >= 0, got {replay}")
    if replay_size is not None and replay_size < 0:
        raise ValueError(f"replay_size must be >= 0, got {replay_size}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    if not (min_delta >= 0.0) or not math.isfinite(min_delta):
        raise ValueError(f"min_delta must be a finite number >= 0, got {min_delta}")


def paced(n: int, curriculum: float, j: int, epochs: int) -> int:
    """How many of the list epoch ``j`` (from 0) of ``epochs`` walks: the first ``c`` of it, growing to all."""
    if n <= 0:
        return 0
    if curriculum >= 1.0 or epochs <= 1:
        return n
    frac = curriculum + ((1.0 - curriculum) * j) / (epochs - 1)
    return min(n, max(1, math.ceil(frac * n)))


def replay_count(replay: float, n: int, pool: int) -> int:
    """How many buffered texts an epoch rehearses: ``replay`` of the run's ``n`` texts, rounded half up."""
    if replay <= 0.0 or pool <= 0:
        return 0
    return min(pool, math.floor(replay * n + 0.5))


class ReplayBuffer:
    """A bounded, uniform sample of every text a model was ever trained on.

    The ``g``-th text ever offered gets the priority :func:`replay_key` of the
    model's seed and ``g``; the buffer keeps the ``size`` entries with the
    smallest ``(priority, g)``.  That is bottom-k sampling: whatever order the
    corpus came in, every text ever offered is equally likely to be here, and
    no random number is drawn.
    """

    __slots__ = ("seed", "size", "seen", "_items")

    def __init__(self, size: int, seed: int, seen: int = 0, entries: Sequence[tuple[int, str]] = ()) -> None:
        if size < 0:
            raise ValueError(f"replay_size must be >= 0, got {size}")
        self.seed = int(seed)
        self.size = int(size)
        self.seen = int(seen)
        items = sorted((replay_key(self.seed, int(g)), int(g), str(t)) for g, t in entries)
        self._items: list[tuple[int, int, str]] = items[: self.size]

    def __len__(self) -> int:
        return len(self._items)

    def texts(self) -> list[str]:
        """The kept texts, in priority order - the order an epoch rehearses them in."""
        return [t for _k, _g, t in self._items]

    def offer(self, texts: Sequence[str]) -> None:
        """Offer ``texts`` in order; each is kept while it is among the ``size`` smallest priorities."""
        items = self._items
        for text in texts:
            g = self.seen
            self.seen += 1
            if self.size == 0:
                continue
            item = (replay_key(self.seed, g), g, text)
            if len(items) < self.size:
                bisect.insort(items, item)
            elif item[:2] < items[-1][:2]:
                items.pop()
                bisect.insort(items, item)

    def resize(self, size: int) -> None:
        """A new capacity; a smaller one drops the entries with the largest priorities."""
        if size < 0:
            raise ValueError(f"replay_size must be >= 0, got {size}")
        self.size = int(size)
        del self._items[self.size :]

    def to_dict(self) -> dict:
        """The file's ``replay`` block: ``{size, seen, index, texts}``, entries in priority order."""
        return {
            "size": self.size,
            "seen": self.seen,
            "index": [g for _k, g, _t in self._items],
            "texts": [t for _k, _g, t in self._items],
        }

    @classmethod
    def from_dict(cls, d: dict, seed: int) -> "ReplayBuffer":
        """Read a ``replay`` block; the priorities are recomputed from ``seed``, not trusted."""
        index = [int(g) for g in d.get("index") or []]
        texts = [str(t) for t in d.get("texts") or []]
        if len(index) != len(texts):
            raise ValueError(f"replay block has {len(index)} indices for {len(texts)} texts")
        return cls(int(d.get("size", len(texts))), seed, int(d.get("seen", len(texts))), list(zip(index, texts)))


class EarlyStop:
    """Stop a run that has stopped improving: ``patience`` epochs without a loss ``min_delta`` below the best."""

    __slots__ = ("patience", "min_delta", "best", "stale")

    def __init__(self, patience: int = 0, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.stale = 0

    def update(self, loss: float) -> bool:
        """Take one epoch's loss; ``True`` when the run should stop after it."""
        if loss < self.best - self.min_delta:
            self.best = loss
            self.stale = 0
        else:
            self.stale += 1
        return self.patience > 0 and self.stale >= self.patience


class TrainingPlan:
    """Which texts each epoch of one ``train`` call walks, what it rehearses, and when it stops.

    ``texts`` are the run's usable texts (the ones too short for a gram
    already dropped), ``lengths`` their lengths in the encoding's units.
    ``buffer`` is the model's replay buffer as the run begins - read, never
    written: :meth:`finish` hands back the buffer the model keeps afterwards.
    """

    def __init__(
        self,
        texts: Sequence[str],
        lengths: Sequence[int],
        *,
        seed: int,
        epochs: int,
        order: str = "corpus",
        curriculum: float = 1.0,
        replay: float = 0.0,
        replay_size: int | None = None,
        patience: int = 0,
        min_delta: float = 0.0,
        buffer: ReplayBuffer | None = None,
    ) -> None:
        check_plan(order, curriculum, replay, replay_size, patience, min_delta)
        self.texts = list(texts)
        self.n = len(self.texts)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.order = order
        self.curriculum = float(curriculum)
        self.replay_size = replay_size
        self._buffer = buffer
        self.pool: list[str] = buffer.texts() if buffer is not None else []
        self.rehearse = replay_count(float(replay), self.n, len(self.pool))
        self._stop = EarlyStop(patience, min_delta)
        if order == "shortest-first":
            self._base = sorted(range(self.n), key=lambda i: (lengths[i], i))
        elif order == "longest-first":
            self._base = sorted(range(self.n), key=lambda i: (-lengths[i], i))
        else:
            self._base = list(range(self.n))

    @property
    def plain(self) -> bool:
        """Whether every epoch walks every text in corpus order and rehearses nothing - the pass it always was."""
        return self.order == "corpus" and self.curriculum >= 1.0 and self.rehearse == 0

    def replayed(self) -> list[str]:
        """The buffered texts the structure pass observes after the run's own: the whole pool, when any is rehearsed.

        They were trained before, so observing them is a no-op on a graph that
        still holds them; observing all of them, in priority order, keeps the
        structure pass one list long in every port.
        """
        return list(self.pool) if self.rehearse else []

    def walked(self, j: int) -> int:
        """How many of the run's texts epoch ``j`` walks."""
        return paced(self.n, self.curriculum, j, self.epochs)

    def epoch(self, j: int, number: int) -> tuple[list[int], list[str]]:
        """Epoch ``j`` (from 0), whose record will carry epoch ``number``: ``(text indices, rehearsed texts)``."""
        if self.order == "shuffle":
            order = sorted(range(self.n), key=lambda i: (shuffle_key(self.seed, number, i), i))
        else:
            order = self._base
        chosen = order[: self.walked(j)]
        if not self.rehearse:
            return chosen, []
        size = len(self.pool)
        start = j * self.rehearse
        return chosen, [self.pool[(start + i) % size] for i in range(self.rehearse)]

    def stop(self, j: int, loss: float) -> bool:
        """Whether the run stops after epoch ``j``, given its loss; an epoch inside the curriculum does not count."""
        if self.walked(j) < self.n:
            return False
        return self._stop.update(loss)

    def finish(self) -> ReplayBuffer | None:
        """The replay buffer the model keeps after this run: resized if asked, with the run's texts offered."""
        buffer = self._buffer
        if self.replay_size is not None:
            if self.replay_size == 0:
                return None
            if buffer is None:
                buffer = ReplayBuffer(self.replay_size, self.seed)
            else:
                buffer.resize(self.replay_size)
        if buffer is not None:
            buffer.offer(self.texts)
        return buffer
