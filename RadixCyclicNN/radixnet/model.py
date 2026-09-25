"""RadixNet - training, prediction, generation, scoring, 2NRL and persistence.

The model ties the pieces together: :class:`~radixnet.encoding.Encoder`
turns text into trigrams, :class:`~radixnet.graph.RadixCyclicGraph` stores
the self-compressing structure and its parameters, a
:class:`~radixnet.backend.Backend` runs the one-hop local learning rule over
CSR mini-batches and :mod:`radixnet.search` finds the cheapest (Dijkstra) or a
sampled continuation.

2NRL (:meth:`RadixNet.two_nrl`) is *Double-Negative Reinforcement Learning*,
the author's scheme: train on bad / garbage data, :meth:`RadixNet.invert` the
network (every edge weight and every activation amplitude flips sign, so what
was likely becomes unlikely) and fine-tune on correct data with a smaller
learning rate.  The two negatives of the name are the first two steps - trained
**on** the failures, then negated - and the fine-tune is the positive one.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from . import attention as attention_band
from .attention import DEFAULT_BLUR, AttentionBand
from .backend import Backend, get_backend
from .counter import CyclicCounter
from .encoding import WINDOW, Decoder, Encoder, Encoding, _piece
from .graph import BACK, END, FIRST, ORIGINS, START, THINK, W_HEAVY, RadixCyclicGraph
from .beam import Prediction, beam_predict, default_beam
from .penalty import DEFAULT_TRAVERSAL, resolve_traversal, traversal_costs
from .search import LEAST_PUNISHED, REWARD, PathResult, check_sampling, dijkstra_predict, parse_traversal, sample_walk
from .schedule import preview_points
from .training import ReplayBuffer, TrainingPlan, check_plan
from .window import DEFAULT_FLOOR, DEFAULT_TOP, DynamicWindow, check_ladder

__all__ = [
    "GraphModel",
    "MODEL_FORMAT",
    "MODEL_FORMAT_VERSION",
    "RadixNet",
    "TrainConfig",
    "UNKNOWN_PROB",
    "load_model",
    "model_class",
    "model_classes",
    "model_from_dict",
    "model_kinds",
    "new_model",
]

MODEL_FORMAT = "radixnet"
MODEL_FORMAT_VERSION = 1

UNKNOWN_PROB = 1e-6
"""Probability charged by :meth:`RadixNet.score` for an unknown trigram or a missing edge."""

_LOG_UNKNOWN = math.log(UNKNOWN_PROB)
_W = WINDOW       # the default n of the n-gram; a model's own is self.encoding.n
_OV = WINDOW - 1  # and its own overlap self.encoding.overlap
_MAX_LOG_PPL = 700.0  # exp() overflows above ~709; a mean -log softmax never gets there

ProgressFn = Callable[[dict], None]


def _utc_now(timespec: str = "seconds") -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def write_bytes_atomic(path: str, data: bytes, use_gzip: bool = False) -> None:
    """Write ``data`` to ``path`` via a temporary file + ``os.replace``.

    A reader never sees a half-written file, and a crash mid-write leaves the
    previous version intact.  ``use_gzip`` compresses the payload.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            if use_gzip:
                with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                    gz.write(data)
            else:
                fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json_file(path: str) -> dict:
    """Load a JSON document, transparently gunzipping it when it carries the gzip magic.

    The content decides, not the suffix: a plain-JSON file that merely ends in
    ``.gz`` loads as well.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


@dataclass
class TrainConfig:
    """Hyper-parameters of one :meth:`RadixNet.train` call."""

    epochs: int = 5
    lr: float = 0.05
    act_lr: float = 0.005
    """Learning rate of the per-node activation parameters ``a, b, h, k``."""
    batch_size: int = 256
    """Transitions per backend step."""
    clip: float = 5.0
    auto_compress: bool = True
    """Run ``graph.compress()`` after every epoch."""
    shuffle: bool = True
    """Shuffle the transitions every epoch (with the model's seeded RNG)."""
    checkpoint_every: int = 0
    """Checkpoint every N epochs when a manager is given (0 = off)."""
    verbose: bool = False
    lr_schedule: str | None = None
    """Graph function of the epoch for ``lr`` (see :mod:`radixnet.schedule`), e.g. ``"linear(lr0, 4 * lr0)"``."""
    act_lr_schedule: str | None = None
    """Graph function of the epoch for ``act_lr``; may use ``lr`` (the epoch's learning rate), e.g. ``"lr / 10"``."""
    reverse_schedule: bool = False
    """Play the schedules backwards: the last epoch's rates come first (a ramp up becomes a ramp down)."""
    order: str = "corpus"
    """How an epoch walks the texts: ``corpus``, ``shortest-first``, ``longest-first`` or ``shuffle``
    (``../SPEC-SearchAndTraining.md`` §3)."""
    curriculum: float = 1.0
    """The share of the ordered texts the first epoch walks, growing to all of them by the last (1 = off)."""
    replay: float = 0.0
    """How many texts of the model's replay buffer each epoch rehearses, as a share of the run's texts (0 = off)."""
    replay_size: int | None = None
    """The replay buffer's capacity from this run on (``None`` leaves the model's alone, 0 drops it)."""
    patience: int = 0
    """Stop after this many full epochs without the loss improving by ``min_delta`` (0 = off)."""
    min_delta: float = 0.0
    """How much an epoch's loss must fall below the best so far to count as an improvement."""
    reverse: bool = False
    """Read every text backwards, in the encoding's units: its last character (word) first, so the model
    learns what comes *before* (``../SPEC-SearchAndTraining.md`` §9).  Off: the texts as given."""

    def rates(self) -> list[tuple[float, float]]:
        """``(lr, act_lr)`` for every epoch of this config, schedules applied (constant when none are set)."""
        if self.lr_schedule is None and self.act_lr_schedule is None:
            return [(self.lr, self.act_lr)] * self.epochs
        points = preview_points(
            self.lr_schedule, self.act_lr_schedule, self.epochs, self.lr, self.act_lr, reverse=bool(self.reverse_schedule)
        )
        return [(p["lr"], p["act_lr"]) for p in points]

    def validate(self) -> None:
        """Raise ``ValueError`` for values that cannot be trained with."""
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        for name in ("lr_schedule", "act_lr_schedule"):
            expression = getattr(self, name)
            if expression is not None and not isinstance(expression, str):
                raise ValueError(f"{name} must be a string expression or None, got {type(expression).__name__}")
        if self.lr_schedule is not None or self.act_lr_schedule is not None:
            self.rates()  # parses both expressions and evaluates every epoch: a ScheduleError is a ValueError
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        for name in ("lr", "act_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {value}")
        if not (self.clip > 0):
            raise ValueError(f"clip must be > 0, got {self.clip}")
        if self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must be >= 0, got {self.checkpoint_every}")
        check_plan(self.order, self.curriculum, self.replay, self.replay_size, self.patience, self.min_delta)

    def to_dict(self) -> dict:
        """Plain-dict form (JSON-serialisable)."""
        return dataclasses.asdict(self)


_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _whole_text(result: PathResult, prefix: str, encoding: Encoding | None = None) -> PathResult:
    """A generated result is a whole text: ``text`` becomes prefix + continuation (``full_text`` already is).

    The two are joined the way the encoding joins units - nothing between
    characters, a space between words.
    """
    result.full_text = (encoding or Encoding()).join(prefix, result.text)
    result.text = result.full_text
    return result


def _weight_groups(
    texts: Iterable[str] | str, weights: Sequence[float], name: str = "weights"
) -> list[tuple[float, list[str]]]:
    """``[(weight, texts)]`` grouping texts of equal (3-decimal) weight, heaviest first; zero weights are dropped."""
    items = [texts] if isinstance(texts, str) else list(texts)
    values = [float(w) for w in weights]
    if len(values) != len(items):
        raise ValueError(f"{name} has {len(values)} entries for {len(items)} texts")
    groups: dict[float, list[str]] = {}
    for text, weight in zip(items, values):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {weight}")
        if weight > 0:
            groups.setdefault(round(weight, 3), []).append(text)
    return sorted(groups.items(), key=lambda item: -item[0])


def _touches(lo: int, hi: int, spans: Sequence[tuple[int, int]]) -> bool:
    """Does the half-open range ``[lo, hi)`` meet any of the changed ``spans``?

    An empty span is an insertion point: the characters belong on the other
    side, so the step that walked straight past the position is the one at
    fault.
    """
    for start, end in spans:
        if end == start:
            end = start + 1
        if start < hi and lo < end:
            return True
    return False


def _resolve_config(config: TrainConfig | None, overrides: dict) -> TrainConfig:
    """Merge keyword overrides into ``config`` (or the defaults) and validate."""
    cfg = TrainConfig() if config is None else config
    if overrides:
        unknown = sorted(set(overrides) - _CONFIG_FIELDS)
        if unknown:
            raise TypeError(f"unknown train option(s): {', '.join(unknown)}")
        cfg = dataclasses.replace(cfg, **overrides)
    cfg.validate()
    return cfg



META_COUNTERS = (
    "epochs_total", "trained_chars", "trained_texts", "twonrl_runs", "feedback_passes", "path_inversions",
    "failures_total", "cleared_total", "judgements", "rejected",
)
"""Lifetime counters of :attr:`GraphModel.meta` that wrap: each keeps its resets in ``<name>_resets``."""

META_COUNTER_MAPS = ("sources",)
"""Lifetime counters of :attr:`GraphModel.meta` kept per key (``{name: count}``), resets in ``<name>_resets``."""


def meta_add(meta: dict, key: str, delta: int) -> int:
    """Add ``delta`` to a lifetime counter, wrapping it like every other counter; returns the new reading.

    The counter is stored as the two JSON numbers a cyclic counter needs:
    ``meta[key]`` - back to 0 whenever it reaches
    :data:`~radixnet.counter.COUNTER_LIMIT` - and ``meta[key + "_resets"]``,
    how often that happened (see :mod:`radixnet.counter`).
    """
    counter = CyclicCounter.from_pair(meta.get(key, 0), meta.get(f"{key}_resets", 0)).bumped(delta)
    meta[key], meta[f"{key}_resets"] = counter.to_pair()
    return counter.value


def meta_counter(meta: dict, key: str) -> CyclicCounter:
    """One lifetime counter as a :class:`~radixnet.counter.CyclicCounter` reading."""
    return CyclicCounter.from_pair(meta.get(key, 0), meta.get(f"{key}_resets", 0))


def meta_add_keyed(meta: dict, key: str, name: str, delta: int) -> int:
    """Add to one entry of a ``{name: count}`` lifetime counter map, wrapping it; returns the new reading.

    The resets live in a mirror map ``meta[key + "_resets"]`` and, as
    everywhere else, only the names that ever wrapped appear in it.
    """
    resets = meta.setdefault(f"{key}_resets", {})
    counts = meta.setdefault(key, {})
    counter = CyclicCounter.from_pair(counts.get(name, 0), resets.get(name, 0)).bumped(delta)
    counts[name] = counter.value
    if counter.resets:
        resets[name] = counter.resets
    else:
        resets.pop(name, None)
    return counter.value


def carry_meta(meta: dict) -> dict:
    """Wrap every lifetime counter of ``meta`` in place - what a file loaded from disk goes through.

    A file written before the counters were cyclic (or by hand) can carry a
    plain integer over the limit; after this every counter reads as an
    odometer, exactly as it would have if it had been counted here.
    """
    for key in META_COUNTERS:
        if key in meta or f"{key}_resets" in meta:
            meta[key], meta[f"{key}_resets"] = meta_counter(meta, key).to_pair()
    for key in META_COUNTER_MAPS:
        counts = meta.get(key)
        if isinstance(counts, dict):
            for name in list(counts):
                meta_add_keyed(meta, key, name, 0)
    return meta


def meta_stats(meta: dict, *keys: str) -> dict:
    """``{key: value, key + "_resets": resets}`` for the named lifetime counters - what :meth:`GraphModel.stats` reports."""
    out: dict = {}
    for key in keys:
        counter = meta_counter(meta, key)
        out[key], out[f"{key}_resets"] = counter.to_pair()
    return out


class GraphModel:
    """What every model kind shares: the graph, text encoding, prefix location, sampling, scoring, persistence.

    Concrete kinds (:class:`RadixNet`, :class:`~radixnet.countnet.CountRewardNet`)
    differ in how edges get their weights (``train`` / ``reward`` / ``punish`` /
    ``two_nrl`` / ``invert``) and in what ``predict`` returns; everything below
    only needs ``graph``, ``encoder`` and ``decoder``.  ``kind`` names the
    algorithm in files, the API and the CLI.
    """

    kind = "graph"
    label = "graph model"
    description = ""
    units = "chars"
    """What a *new* model of this kind counts in; the encoding decides for a live one."""
    takes_corrections = False
    """Does this kind learn from a diff (``correct``)?  Only such a kind has anything for the attention band
    (:mod:`radixnet.attention`) to spread, so only such a kind accepts one."""

    @property
    def counts_in(self) -> str:
        """What this model counts in - ``"chars"``, or ``"words"`` under a word encoding."""
        return self.encoding.units_name
    """What this kind counts in: its symbols, as a reader of ``length``, ``max_length`` or ``per_char`` needs
    them named.  Characters everywhere but the word model (``../SPEC-WordNGrams.md``)."""

    graph: RadixCyclicGraph
    encoder: Encoder
    decoder: Decoder
    backend: Backend
    history: list[dict]
    meta: dict
    seed: int
    replay: ReplayBuffer | None = None
    """The replay buffer: a uniform sample of every text this model was trained on, rehearsed by a run with
    ``replay > 0`` - or ``None``, and then the file does not mention it (``../SPEC-SearchAndTraining.md`` §4)."""

    def _read(self, texts: Iterable[str] | str, cfg: TrainConfig) -> Iterable[str] | str:
        """The texts of one ``train`` call as ``cfg`` reads them: each backwards, unit by unit, with ``reverse``.

        Every kind calls this first, so everything after it - the short texts
        dropped, the plan, the structure pass, the counters, the replay buffer -
        sees the reversed texts, exactly as if they had been given reversed.
        Anything that is not a string is left for the kind to refuse.
        """
        if not cfg.reverse:
            return texts
        items = [texts] if isinstance(texts, str) else texts
        return [self.encoding.reverse(t) if isinstance(t, str) else t for t in items]

    def _plan(self, texts: list[str], cfg: TrainConfig) -> TrainingPlan:
        """How one ``train`` call walks ``texts``: the order, the curriculum, the rehearsal, the stop."""
        return TrainingPlan(
            texts, [self.encoding.length(t) for t in texts], seed=self.seed, epochs=cfg.epochs,
            order=cfg.order, curriculum=cfg.curriculum, replay=cfg.replay, replay_size=cfg.replay_size,
            patience=cfg.patience, min_delta=cfg.min_delta, buffer=self.replay,
        )

    def replay_summary(self) -> dict | None:
        """The replay buffer at a glance - ``{size, texts, seen}`` - or ``None`` when the model keeps none."""
        if self.replay is None:
            return None
        return {"size": self.replay.size, "texts": len(self.replay), "seen": self.replay.seen}

    def _with_replay(self, doc: dict) -> dict:
        """A model document with the ``replay`` block at its end - only when there is a buffer."""
        if self.replay is not None:
            doc["replay"] = self.replay.to_dict()
        return doc

    def _read_replay(self, d: dict) -> None:
        """The ``replay`` block of a model document, if it has one."""
        block = d.get("replay")
        self.replay = ReplayBuffer.from_dict(block, self.seed) if isinstance(block, dict) else None

    @staticmethod
    def _new_meta(seed: int) -> dict:
        meta = {"created": _utc_now(), "seed": seed}
        for key in ("epochs_total", "trained_chars", "trained_texts", "twonrl_runs"):
            meta[key], meta[f"{key}_resets"] = 0, 0
        return meta

    # -- the algorithm-specific part (implemented by every kind) -------------

    def train(self, texts, config=None, *, checkpoint_manager=None, progress=None, stop_event=None, phase=None, **overrides):
        raise NotImplementedError

    def predict(self, prefix: str, **options):
        raise NotImplementedError

    def two_nrl(self, bad, good, **options) -> dict:
        raise NotImplementedError

    def reward(self, texts, **options) -> list[dict]:
        raise NotImplementedError

    def punish(self, texts, **options) -> list[dict]:
        raise NotImplementedError

    def invert(self) -> None:
        raise NotImplementedError

    def invert_paths(self, texts, mode: str = "activation", amounts=None, **options) -> dict:
        """Failures: make the paths of ``texts`` unlikely locally, each by ``amounts[i]`` (0..1, default 1 = fully).

        See the kinds; returns ``{"texts", "flipped", "unit", "mode", "amount_mean"}``.
        """
        raise NotImplementedError

    @staticmethod
    def _amounts(texts: list[str], amounts) -> list[float]:
        """Per-text update amounts in ``[0, 1]``: one number for all, a sequence aligned with ``texts``, or 1."""
        if amounts is None:
            return [1.0] * len(texts)
        if isinstance(amounts, (int, float)):
            values = [float(amounts)] * len(texts)
        else:
            values = [float(v) for v in amounts]
            if len(values) != len(texts):
                raise ValueError(f"amounts has {len(values)} entries for {len(texts)} texts")
        for v in values:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"amounts must lie in [0, 1], got {v}")
        return values

    def _steps_over(self, grams: list[str], length: int, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """The steps of a traced text that wrote a character inside one of ``spans``, as ``(prev node, edge)``.

        Every step is charged with the characters it adds to the text: the
        first with the whole of its node's label, a later one with everything
        past the two characters it overlaps its parent by, and the step into
        END with the position just past the last character - where a sentence
        that stopped too early went wrong.  Used to move only the nodes a
        correction's diff (:mod:`radixnet.diff`) marks as changed.
        """
        graph = self.graph
        path = graph.node_path(grams)
        if not path or len(path) < 2:
            return []
        labels = graph.labels
        children = graph.children
        overlap = graph.encoding.overlap
        out: list[tuple[int, int]] = []
        position = 0  # gram index of the node being entered
        for index in range(1, len(path)):
            node = path[index]
            prev = path[index - 2] if index >= 2 else START  # who called the step: START begins every walk
            edge = children[path[index - 1]].get(node)
            if node == END:
                if edge is not None and _touches(length, length + 1, spans):
                    out.append((prev, edge))
                break
            size = graph.label_len(node)
            lo = 0 if index == 1 else position + overlap
            if edge is not None and _touches(lo, position + size, spans):
                out.append((prev, edge))
            position += size - overlap
        return out

    def _charged_steps(
        self, grams: list[str], length: int, spans: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int, float, bool]]:
        """The steps of a traced text a correction charges, as ``(prev node, edge, charge, focus)`` in path order.

        With the attention band off these are exactly :meth:`_steps_over`'s
        steps, each charged 1 and each the focus.  With it on, every changed
        unit is shared out over the grams that see it (:func:`radixnet.attention.spread`)
        and a step is charged what the grams of its node collected - capped at
        one, since a step is one decision - and is the *focus* when one of them
        sees a changed unit most sharply.  The step into END answers for the
        position after the last unit either way.  ``length`` is the text's
        length in the encoding's units.
        """
        band = self.graph.attention
        if not band.on:
            return [(prev, edge, 1.0, True) for prev, edge in self._steps_over(grams, length, spans)]
        graph = self.graph
        path = graph.node_path(grams)
        if not path or len(path) < 2:
            return []
        enc = graph.encoding
        n, stride = enc.n, enc.stride
        shared = attention_band.spread(n, stride, len(grams), length, spans, band.weights(n))
        children = graph.children
        out: list[tuple[int, int, float, bool]] = []
        gram = 0  # the text's first gram inside the node being entered
        for index in range(1, len(path)):
            node = path[index]
            prev = path[index - 2] if index >= 2 else START
            edge = children[path[index - 1]].get(node)
            if node == END:
                if edge is not None and shared.end:
                    out.append((prev, edge, 1.0, True))
                break
            held = (graph.label_len(node) - n) // stride + 1  # the grams a node of this length holds
            charge = 0.0
            focus = False
            for g in range(gram, min(gram + held, len(grams))):
                charge += shared.shares[g]
                focus = focus or shared.focus[g]
            gram += held
            if edge is not None and charge > 0.0:
                out.append((prev, edge, min(1.0, charge), focus))
        return out

    # -- the attention band: where a correction lands --------------------------

    def attention_config(self) -> dict:
        """The attention band as the API, the CLI and the frontend show it.

        ``weights`` is the band over one gram of this model's encoding (``None``
        while it is off); ``applies`` says whether this kind is ever corrected,
        i.e. whether the band has anything to spread.
        """
        band = self.graph.attention
        enc = self.encoding
        return {
            "on": band.on,
            "blur": band.blur,
            "weights": band.weights(enc.n),
            "ngram": enc.n,
            "stride": enc.stride,
            "unit": enc.unit,
            "units": enc.units_name,
            "applies": self.takes_corrections,
            "default_blur": DEFAULT_BLUR,
        }

    def configure_attention(self, *, on: bool | None = None, blur: float | None = None) -> dict:
        """Switch the band on (at ``blur``, else the blur it had, else :data:`~radixnet.attention.DEFAULT_BLUR`)
        or off; returns :meth:`attention_config`.

        ``blur`` alone switches it on at that blur; ``on=False`` switches it off
        whatever ``blur`` says.  A kind that is never corrected refuses rather
        than keeping a setting that could not do anything.
        """
        if not self.takes_corrections:
            raise ValueError(
                f"the {self.kind} model is never corrected, so an attention band would have nothing to spread; "
                "it belongs to the kinds that learn from a diff (count, negative)"
            )
        current = self.graph.attention
        if on is False:
            self.graph.attention = AttentionBand()
        elif on is True or blur is not None:
            if blur is None:
                blur = current.blur if current.on else DEFAULT_BLUR
            self.graph.attention = AttentionBand(blur)
        return self.attention_config()

    def attention_preview(self, wrong: str, right: str, blur: float | None = None) -> dict:
        """Where one correction's charges land, gram by gram - under the writer rule and under a band.

        The band shown is ``blur`` when given, else the model's own, else the
        default one: a preview is how to see a blur before applying it, so it
        never needs the band to be on and never changes anything.  Needs only
        the encoding, not the graph.
        """
        from . import diff  # local import, as the negative network does: only corrections need the alignment

        wrong, right = str(wrong or ""), str(right or "")
        band = self.graph.attention
        shown = AttentionBand(blur if blur is not None else (band.blur if band.on else DEFAULT_BLUR))
        enc = self.encoding
        weights = shown.weights(enc.n)
        wrong_spans, right_spans = diff.changed_spans(wrong, right, enc)
        return {
            "attention": self.attention_config(),
            "blur": shown.blur,
            "weights": weights,
            "changes": diff.summary(wrong, right, limit=0, encoding=enc),
            "wrong": attention_band.preview(enc, weights, wrong, wrong_spans),
            "right": attention_band.preview(enc, weights, right, right_spans),
        }

    # -- the dynamic window: a ladder of node sizes, halving down and back up ------

    def window_config(self) -> dict:
        """The dynamic window as the API, the CLI and the frontend show it (:mod:`radixnet.window`).

        ``sizes`` is the ladder (empty while off), ``next`` the size after the
        current one, ``longer`` how many real nodes a step would halve (``None``
        while off) and ``longest`` the longest label in the graph, both in the
        encoding's units; ``heavy`` is the bridge weight of the sine model
        (``None`` for a kind whose bridge is heavy by its count).
        """
        graph = self.graph
        window = graph.dynamic_window
        enc = self.encoding
        longer, longest = graph.longer_than(window.size if window.on else 0)
        return {
            "on": window.on,
            "top": window.top,
            "floor": window.floor,
            "size": window.size,
            "auto": window.auto,
            "sizes": window.sizes(),
            "next": window.next_size(),
            "unit": enc.unit,
            "units": enc.units_name,
            "ngram": enc.n,
            "longer": longer if window.on else None,
            "longest": longest,
            "nodes": graph.num_nodes() - FIRST,
            "heavy": W_HEAVY if graph.learns_weights else None,
            "default_top": DEFAULT_TOP,
            "default_floor": DEFAULT_FLOOR,
        }

    def configure_window(
        self, *, on: bool | None = None, top: int | None = None, floor: int | None = None, size: int | None = None,
        auto: bool | None = None,
    ) -> dict:
        """Switch the window on or off, or move its ladder; returns :meth:`window_config`.

        ``on=False`` switches it off, whatever else is given: the graph stays as
        the last step left it, and compression is unbounded again.  ``on=True``
        or any setting switches it on - at the values given over the ones it
        had, else the defaults (``32`` down to ``4``, standing at the top,
        stepping at the end of every epoch).  A new top or floor keeps the
        size on the ladder: above the top it becomes the top, below the floor
        the floor.  ``ValueError`` for a size that is not a power of two, a
        floor over the top or a size off the ladder.  Nothing here touches the
        graph: only a step does (:meth:`window_step`).
        """
        current = self.graph.dynamic_window
        if on is False:
            self.graph.dynamic_window = DynamicWindow()
            return self.window_config()
        if on is None and top is None and floor is None and size is None and auto is None:
            return self.window_config()
        top = (current.top if current.on else DEFAULT_TOP) if top is None else int(top)
        floor = (current.floor if current.on else DEFAULT_FLOOR) if floor is None else int(floor)
        auto = (current.auto if current.on else True) if auto is None else bool(auto)
        check_ladder(top, floor)
        if size is None:
            size = current.size if current.on else top
            size = min(top, max(floor, size))  # a new top or floor keeps the size on the ladder
        self.graph.dynamic_window = DynamicWindow(top, floor, int(size), auto)
        return self.window_config()

    def window_step(self, steps: int = 1, compress: bool = True) -> dict:
        """Step the ladder ``steps`` times: merge what fits the window, halve what does not, move the window.

        Each step compresses the graph (within the current size), halves every
        node longer than the size (:meth:`RadixCyclicGraph.split_window`) and
        moves the window down the ladder - back to the top from the floor.
        ``compress=False`` leaves the merging to a caller that has just done
        it.  Returns what happened: the sizes applied, the merges, the splits
        and the graph's size before and after, with :meth:`window_config` as
        ``window``.  ``ValueError`` while the window is off.
        """
        graph = self.graph
        if not graph.dynamic_window.on:
            raise ValueError("the dynamic window is off: switch it on first (window --on)")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        nodes_before, edges_before = graph.num_nodes(), graph.num_edges()
        applied: list[int] = []
        merges = splits = 0
        for _ in range(steps):
            window = graph.dynamic_window
            if compress:
                merges += graph.compress()
            splits += graph.split_window(window.size)
            applied.append(window.size)
            graph.dynamic_window = window.advanced()
        return {
            "steps": steps,
            "sizes": applied,
            "from": applied[0],
            "to": graph.dynamic_window.size,
            "merges": merges,
            "splits": splits,
            "nodes_before": nodes_before,
            "nodes_after": graph.num_nodes(),
            "edges_before": edges_before,
            "edges_after": graph.num_edges(),
            "window": self.window_config(),
        }

    def _window_epoch(self, compress: bool = False) -> dict | None:
        """The window's automatic step at the end of a training epoch, or ``None`` when it is off or stepped by hand.

        Returns ``{"window": the size applied, "splits", "merges"}``; the
        epoch's record carries the first two, and its ``merges`` the third.
        Every kind's loop has compressed the graph just before, so the step
        only halves and moves - a kind that does not compress per epoch (the
        phase model) asks for the merging here with ``compress=True``.
        """
        window = self.graph.dynamic_window
        if not (window.on and window.auto):
            return None
        done = self.window_step(1, compress=compress)
        return {"window": done["from"], "splits": done["splits"], "merges": done["merges"]}

    def _paths_of(self, texts: list[str]) -> list[list[int]]:
        """Node paths (sentinels included) of texts, registering a text structurally when it cannot be walked yet."""
        graph = self.graph
        paths: list[list[int]] = []
        for text in texts:
            grams = self.encoder.encode(text)
            if not grams:
                continue
            path = graph.node_path(grams)
            if path is None:
                graph.observe_sequence(grams, count=False)
                path = graph.node_path(grams)
            if path:
                paths.append(path)
        return paths

    def stats(self) -> dict:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None):
        raise NotImplementedError

    # -- texts and structure -------------------------------------------------

    def _clean_texts(self, texts: Iterable[str] | str) -> tuple[list[str], int]:
        """Normalise the ``texts`` argument; returns ``(usable, skipped_short)``.

        A text too short to hold one gram of this model's encoding is skipped."""
        items = [texts] if isinstance(texts, str) else list(texts)
        kept: list[str] = []
        skipped = 0
        for t in items:
            if not isinstance(t, str):
                raise TypeError(f"texts must be strings, got {type(t).__name__}")
            if self.encoding.length(t) < self.encoding.n:
                skipped += 1
            else:
                kept.append(t)
        return kept, skipped

    def _observe_grams(
        self, grams: list[list[str]], count: bool, origin: int = START
    ) -> list[tuple[int, int]]:
        """Register encoded texts structurally; returns the ``(parent_id, edge_id)`` transitions.

        A text observed later can split a node that an earlier text's
        transition points at (the edge moves to the new node), so whenever the
        first pass changed the structure a second, non-counting pass re-derives
        every transition from the final structure - that pass never splits.
        ``origin`` is the sentinel every sequence begins at (START; THINK for
        thoughts).
        """
        graph = self.graph
        observe = graph.observe_sequence
        before = graph.structure_version
        transitions: list[tuple[int, int]] = []
        extend = transitions.extend
        for g in grams:
            extend(observe(g, count, origin))
        if graph.structure_version != before:
            transitions = []
            extend = transitions.extend
            for g in grams:
                extend(observe(g, False, origin))
        return transitions

    def _observe(self, texts: list[str], count: bool, origin: int = START) -> tuple[list[tuple[int, int]], int]:
        """Register every text structurally; returns ``(transitions, structure_version)``."""
        transitions = self._observe_grams([self.encoder.encode(t) for t in texts], count, origin)
        return transitions, self.graph.structure_version

    # -- locating a prefix ---------------------------------------------------

    @property
    def encoding(self) -> Encoding:
        """How this model turns text into grams and back - its graph's :class:`Encoding`."""
        return self.graph.encoding

    def _adopt(self, graph: RadixCyclicGraph) -> None:
        """Take a loaded graph *and read text the way it does*.

        A model built to hold a file's graph is constructed before the graph is
        read, so its encoder and decoder are the default ones; a graph in any
        other encoding would then be fed grams it has never seen.
        """
        self.graph = graph
        self.encoder = Encoder(encoding=graph.encoding)
        self.decoder = Decoder(encoding=graph.encoding)

    def _best_trigram(self, key: str) -> tuple[int, int] | None:
        """Most-visited ``(node, offset)`` holding a trigram that starts with ``key``."""
        count = self.graph.node_count
        best: tuple[int, int] | None = None
        best_rank: tuple[int, int, int] | None = None
        prefixed = self.encoding.has_unit_prefix
        for t, (node, off) in self.graph.trigram_index.items():
            if prefixed(t, key):
                rank = (-count(node), node, off)
                if best_rank is None or rank < best_rank:
                    best, best_rank = (node, off), rank
        return best

    def _best_node_with_prefix(self, prefix: str) -> int | None:
        """Most-visited real node whose label starts with ``prefix``."""
        g = self.graph
        best: int | None = None
        best_count = -1
        for node in range(FIRST, len(g.labels)):
            if g.alive[node] and self.encoding.has_unit_prefix(g.labels[node], prefix):
                c = g.node_count(node)
                if c > best_count:
                    best, best_count = node, c
        return best

    def _locate(self, prefix: str) -> tuple[int, int, int]:
        """``(node, offset, matched)``: where ``prefix`` ends in the graph.

        ``matched`` is how many characters of the located trigram (or, for a
        short prefix, of the node label) the prefix actually covers; the rest
        of that trigram is a guessed continuation.  ``(START, 0, 0)`` if
        nothing matches.
        """
        enc = self.encoding
        view = enc.units(prefix)
        n = len(view)
        if n == 0:
            return START, 0, 0
        if n >= enc.n:
            # the gram the prefix ends on, then - when the stride skips it - the
            # last gram of the prefix's own grid
            loc = self.graph.lookup(_piece(view, n - enc.n, n))
            if loc is None:
                aligned = (n - enc.n) // enc.stride * enc.stride
                if aligned != n - enc.n:
                    loc = self.graph.lookup(_piece(view, aligned, aligned + enc.n))
            if loc is not None:
                return loc[0], loc[1], enc.n
            for m in (enc.n - 1, 1):
                if not 1 <= m < enc.n:
                    continue
                found = self._best_trigram(_piece(view, n - m, n))
                if found is not None:
                    return found[0], found[1], m
            return START, 0, 0
        node = self._best_node_with_prefix(prefix)
        if node is not None:
            return node, 0, n
        return START, 0, 0

    def locate(self, prefix: str) -> tuple[int, int]:
        """Node and trigram offset where a prediction for ``prefix`` starts.

        Exact match of the last three characters first; otherwise the
        most-visited node holding a trigram that starts with the last two
        (then the last one) characters; a prefix shorter than three characters
        matches the most-visited node whose label starts with it.  Falls back
        to ``(START, 0)``.
        """
        node, offset, _ = self._locate(prefix)
        return node, offset

    def _prefix_start(self, prefix: str) -> tuple[int, int, str]:
        """``(node, offset, lead)`` for a prediction: where the prefix ends and the unmatched rest of that trigram.

        When only a partial trigram of the prefix could be matched, ``lead`` is
        the remainder of the located trigram - a guessed opening of the
        continuation that every predicted path starts with.
        """
        enc = self.encoding
        node, offset, matched = self._locate(prefix)
        lead = "" if node == START or matched >= enc.n else enc.piece(
            self.graph.labels[node], offset + matched, offset + enc.n
        )
        return node, offset, lead

    def _walk_start(self, prefix: str, origin: int = START) -> tuple[int, int, str]:
        """Where a walk begins: at the end of ``prefix``, or - with no prefix - at the ``origin`` sentinel.

        ``origin=THINK`` is a thought (:mod:`radixnet.thinking`): the same
        search from the other sentinel, through the openings the model learned
        for its thoughts rather than for its texts.  A prefix wins over the
        origin, because a located prefix already says where the walk stands.
        """
        if origin not in ORIGINS:
            raise ValueError(f"a walk begins at START or THINK, not at node {origin}")
        if not prefix and origin != START:
            return origin, 0, ""
        return self._prefix_start(prefix)

    # -- the prediction search (shared by every kind) ------------------------

    @staticmethod
    def _check_predict_args(prefix: str, length: int, max_length: int | None, k: int, beam: int | None) -> None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if beam is not None and beam < 1:
            raise ValueError(f"beam must be >= 1, got {beam}")

    def _search(
        self,
        prefix: str,
        length: int,
        mode: str,
        k: int,
        beam: int | None,
        step_penalty: float,
        temperature: float,
        to_end: bool,
        max_length: int | None,
        rng: random.Random | None = None,
        traversal: str = DEFAULT_TRAVERSAL,
        penalty_scale: float = 1.0,
        merit_scale: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        diversity: float = 0.0,
        origin: int = START,
    ) -> Prediction:
        """The prediction search from where ``prefix`` ends: ``"beam"`` (the ``k`` most and least likely
        continuations) or ``"sample"`` (one stochastic walk).  The result *is* the best path and carries ``top`` /
        ``bottom``; ``length``, ``to_end`` and ``max_length`` follow :meth:`RadixNet.predict`.  ``traversal``
        picks the cost function the search reads the graph through
        (``"punishment"``, :mod:`radixnet.penalty`) or, with ``"least-punished"``,
        what a walk is *ranked* by (``../SPEC-LeastPunished.md``).  ``top_k`` / ``top_p`` / ``min_p`` narrow what
        a sampled step draws from and ``diversity`` spreads the top beam out
        (``../SPEC-SearchAndTraining.md`` §1-2); each is off at its default.  ``origin=THINK`` with an empty
        prefix walks a *thought* (:mod:`radixnet.thinking`)."""
        graph = self.graph
        enc = self.encoding
        traversal = resolve_traversal(traversal)
        costs = traversal_costs(graph, traversal, penalty_scale, merit_scale)
        node, offset, lead = self._walk_start(prefix, origin)
        # the lead is the unmatched rest of the located gram: a length in the encoding's units, as
        # `length` and `max_length` are - its words under a word encoding, not its characters
        lead_len = enc.length(lead)
        want = max(0, length - lead_len)
        cap: int | None
        if mode == "beam":
            if max_length is None and length == 0:
                cap, max_chars = 0, 0
            elif max_length is None:
                cap, max_chars = None, None
            else:
                cap = max(length, max_length)
                max_chars = max(want, cap - lead_len)
            top, bottom, expanded = beam_predict(
                graph, node, offset, min_chars=want, k=k, beam=beam, max_chars=max_chars,
                step_penalty=step_penalty, to_end=to_end, costs=costs, traversal=traversal, diversity=diversity,
            )
            width = default_beam(k) if beam is None else int(beam)
        else:
            cap = max_length if max_length is not None else length
            walk = sample_walk(
                graph, node, offset, max_chars=max(0, cap - lead_len), temperature=temperature, rng=rng,
                costs=costs, traversal=traversal, top_k=top_k, top_p=top_p, min_p=min_p,
            )
            top, bottom, expanded, width = [walk], [], walk.expanded, 0
        for result in top + bottom:
            if lead:
                joined = enc.join(lead, result.text)
                result.text = joined if cap is None else enc.truncate(joined, cap)
            result.full_text = enc.join(prefix, result.text)
        best = top[0] if top else PathResult(
            text=lead if cap is None else enc.truncate(lead, cap or 0), labels=[graph.labels[node]], node_ids=[node]
        )
        if not top:
            best.full_text = enc.join(prefix, best.text)
        return Prediction(
            text=best.text, labels=list(best.labels), node_ids=list(best.node_ids), cost=best.cost,
            step_costs=list(best.step_costs), expanded=expanded, reached_end=best.reached_end, full_text=best.full_text,
            top=top, bottom=bottom, k=k, beam=width, mode=mode, traversal=traversal,
        )

    # -- generation and scoring ----------------------------------------------

    def generate(
        self,
        max_length: int = 60,
        mode: str = "sample",
        temperature: float = 1.0,
        count: int = 1,
        seed: int | None = None,
        prefix: str = "",
        step_penalty: float = 0.0,
        beam: int | None = None,
        traversal: str = DEFAULT_TRAVERSAL,
        penalty_scale: float = 1.0,
        merit_scale: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        diversity: float = 0.0,
    ) -> list[PathResult]:
        """Generate whole texts with the prediction search, from START or continuing ``prefix``.

        * ``"beam"`` - the prediction search run to the end of a text: the
          ``count`` most likely distinct complete texts (each at most
          ``max_length`` characters), most likely first.  ``beam`` is the
          beam width (default ``max(4 * count, 16)``), ``step_penalty`` the
          extra cost per edge.
        * ``"sample"`` - ``count`` stochastic walks (``seed`` gives a private
          RNG; otherwise the model's seeded RNG is consumed).
        * ``"dijkstra"`` - the single cheapest complete text (one entry).

        Every result's ``text`` (and ``full_text``) is the whole text, prefix
        included; ``probability`` is ``exp(-cost)``.  ``traversal`` picks what
        the search is looking for - ``"reward"`` or ``"punishment"``, the text
        that accumulated the least punishment (:meth:`predict`).  ``top_k`` /
        ``top_p`` / ``min_p`` narrow what a sampled step draws from and
        ``diversity`` spreads the beam's texts apart (``../SPEC-SearchAndTraining.md``).
        """
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        mode = (mode or "sample").lower()
        if mode not in ("beam", "dijkstra", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'beam', 'dijkstra' or 'sample'")
        if count == 0:
            return []
        check_sampling(top_k, top_p, min_p)
        if not (diversity >= 0.0):
            raise ValueError(f"diversity must be >= 0, got {diversity}")
        walk_options = {"traversal": traversal, "penalty_scale": penalty_scale, "merit_scale": merit_scale}
        if mode == "sample":
            rng = random.Random(seed) if seed is not None else None
            results: list[PathResult] = []
            for _ in range(count):
                walk = self._search(
                    prefix, max_length, "sample", 0, None, 0.0, temperature, False, max_length, rng=rng,
                    top_k=top_k, top_p=top_p, min_p=min_p, **walk_options,
                )
                results.append(_whole_text(walk, prefix, self.encoding))
            return results
        if mode == "dijkstra":
            best = self.predict(
                prefix, length=0, mode="dijkstra", to_end=True, max_length=max_length,
                step_penalty=step_penalty, **walk_options,
            )
            return [_whole_text(best, prefix, self.encoding)]
        found = self.predict(
            prefix, length=0, mode="beam", k=count, beam=beam, to_end=True, max_length=max_length,
            step_penalty=step_penalty, diversity=diversity, **walk_options,
        )
        return [_whole_text(result, prefix, self.encoding) for result in found.top]

    def converse(self, opening: str = "", turns: int = 6, **options):
        """The model converses with itself: two voices, each reply the prediction search picking up the end of
        the previous line (:func:`radixnet.dialogue.converse`; ``partner=`` lets another model answer)."""
        from .dialogue import converse

        return converse(self, opening, turns, **options)

    def _edge_log_prob(self, p: int, offset: int, c: int) -> float | None:
        """``log P(c | p)`` for the edge ``p -> c`` taken from ``p``'s trigram at ``offset``.

        ``None`` when ``p`` is not positioned at its last trigram (a transition
        out of the middle of a compressed node is impossible) or the edge does
        not exist.
        """
        g = self.graph
        if p != START and offset + g.encoding.n != g.label_len(p):
            return None
        for child, _edge, cost in g.child_costs(p):
            if child == c:
                return -cost
        return None

    def score(self, text: str) -> dict:
        """Log-probability of ``text`` under the model.

        The text's trigrams are walked ``START -> ... -> END`` through the
        structure; every edge contributes ``log softmax`` of its score, steps
        inside a compressed node are deterministic (cost 0) and are not
        counted as transitions.  An unknown trigram, a missing edge or a
        transition that would need a split contributes ``log(UNKNOWN_PROB)``
        and is counted in ``unknown_transitions``.

        ``chars`` and ``per_char`` are in the encoding's **units**: characters
        under the default encoding, words under a word one.  ``units`` says
        which, because a per-word number read as per-character is read wrong.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        grams = self.encoder.encode(text)
        chars = self.encoding.length(text)
        units = self.encoding.units_name
        if not grams:
            return {"log_prob": 0.0, "per_char": 0.0, "chars": chars, "transitions": 0,
                    "unknown_transitions": 0, "units": units}
        index = self.graph.trigram_index
        log_prob = 0.0
        transitions = 0
        unknown = 0
        node = START
        offset = 0
        lost = False  # position unknown after an unknown trigram
        for t in grams:
            loc = index.get(t)
            if loc is None:
                transitions += 1
                unknown += 1
                log_prob += _LOG_UNKNOWN
                lost = True
                continue
            n, o = loc
            if not lost and n == node and o == offset + 1:
                offset = o  # deterministic in-node step
                continue
            transitions += 1
            lp = None if lost or o != 0 else self._edge_log_prob(node, offset, n)
            if lp is None:
                unknown += 1
                log_prob += _LOG_UNKNOWN
            else:
                log_prob += lp
            node, offset, lost = n, o, False
        transitions += 1
        lp = None if lost else self._edge_log_prob(node, offset, END)
        if lp is None:
            unknown += 1
            log_prob += _LOG_UNKNOWN
        else:
            log_prob += lp
        return {
            "log_prob": log_prob,
            "per_char": log_prob / max(1, chars),
            "chars": chars,
            "transitions": transitions,
            "unknown_transitions": unknown,
            "units": units,
        }

    # -- structure -----------------------------------------------------------

    def compress(self) -> int:
        """Merge all unary chains; returns the number of merges."""
        return self.graph.compress()

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the model as JSON (gzip when ``path`` ends with ``.gz``), atomically."""
        payload = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        write_bytes_atomic(path, payload, use_gzip=path.endswith(".gz"))

    @classmethod
    def load(cls, path: str, backend: str = "auto", device: str | None = None) -> "GraphModel":
        """Load a model written by :meth:`save` (of this class; :func:`load_model` detects the kind)."""
        return cls.from_dict(read_json_file(path), backend=backend, device=device)

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"{type(self).__name__}(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, inverted={g.inverted})"
        )


class RadixNet(GraphModel):
    """The self-compressing cyclic-graph network.

    ``graph`` holds structure and parameters, ``backend`` runs the learning
    rule, ``history`` collects one record per trained epoch and ``meta``
    keeps lifetime counters.  All records are plain JSON-serialisable dicts.
    """

    kind = "radix"
    format = MODEL_FORMAT
    label = "RadixNet (sine activation)"
    description = (
        "edge weights and per-node sine activations learned by the one-hop rule; Dijkstra prediction; "
        "2NRL inverts the network"
    )

    def __init__(
        self, seed: int = 0, backend: str = "auto", device: str | None = None,
        encoding: Encoding | None = None,
    ) -> None:
        self.seed = int(seed)
        self.graph = RadixCyclicGraph(seed=self.seed, encoding=encoding)
        self.encoder = Encoder(encoding=self.graph.encoding)
        self.decoder = Decoder(encoding=self.graph.encoding)
        self.backend: Backend = get_backend(backend, device)
        self.history: list[dict] = []
        self.meta: dict = self._new_meta(self.seed)

    # -- training ------------------------------------------------------------


    def train(
        self,
        texts: Iterable[str] | str,
        config: TrainConfig | None = None,
        *,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        phase: str | None = None,
        origin: int = START,
        **overrides,
    ) -> list[dict]:
        """Train on ``texts`` (one string or an iterable of strings).

        Every text is first registered structurally (splitting compressed
        nodes and creating edges as needed); then, per epoch, the graph is
        exported as CSR, the transitions are shuffled with the model's seeded
        RNG and fed to the backend in mini-batches, the updated weights and
        node parameters are written back and (with ``auto_compress``) unary
        chains are merged.  With ``auto_compress`` the graph is also
        compressed once *before* the first epoch, so every epoch's loss is
        measured over the same kind of transitions (steps inside a compressed
        node emit nothing and are never counted).  Because a merge moves edge
        ids, the transitions are re-derived (``count=False``) whenever the
        structure changed.

        Keyword ``overrides`` replace individual :class:`TrainConfig` fields.
        ``phase`` (optional) is stamped on every record - :meth:`two_nrl` uses
        it.  Returns the epoch records (also appended to :attr:`history`).

        The texts are walked the way ``cfg`` says - the order, the curriculum,
        the rehearsal of the replay buffer and the early stop of
        ``../SPEC-SearchAndTraining.md`` - unless the call carries a ``phase``:
        a pass stamped ``"negative"`` or ``"positive"`` is feedback
        (:meth:`reward`, :meth:`punish`, :meth:`two_nrl`), walks every text in
        corpus order, and neither reads the replay buffer nor offers it a text -
        a punished text is not one to rehearse.

        ``origin`` is the sentinel every text's walk begins at: START, or
        THINK to train the texts as *thoughts* (:mod:`radixnet.thinking`).
        """
        cfg = _resolve_config(config, overrides)
        if origin not in ORIGINS:
            raise ValueError(f"a text begins at START or THINK, not at node {origin}")
        texts, skipped_short = self._clean_texts(self._read(texts, cfg))
        plan = self._plan(texts, cfg) if phase is None else None
        rehearsed = plan.replayed() if plan is not None else []
        graph = self.graph
        backend = self.backend
        rng = graph.rng
        meta = self.meta
        records: list[dict] = []

        transitions, observed_version = self._observe(texts, count=True, origin=origin)
        if rehearsed:
            self._observe(rehearsed, count=False, origin=origin)  # a rehearsed text is not new: nothing to count
            transitions, observed_version = self._observe(texts, count=False, origin=origin)
        meta_add(meta, "trained_texts", len(texts))
        meta_add(meta, "trained_chars", sum(self.encoding.length(t) for t in texts))
        pending_merges = graph.compress() if cfg.auto_compress else 0

        parents_all: list[int] = []
        positions_all: list[int] = []
        arrays_version = -1
        walked = texts
        rates = cfg.rates()  # (lr, act_lr) per epoch: the schedules are graph functions of the epoch
        for k in range(cfg.epochs):
            t0 = time.perf_counter()
            lr, act_lr = rates[k]
            if plan is not None and not plan.plain:
                # this epoch's texts: the order, the curriculum and the rehearsal (../SPEC-SearchAndTraining.md)
                chosen, again = plan.epoch(k, meta_counter(meta, "epochs_total").bumped(1).value)
                walked = [texts[i] for i in chosen] + again
                transitions, observed_version = self._observe(walked, count=False, origin=origin)
                arrays_version = -1
            elif graph.structure_version != observed_version:
                # a merge (or an external structural change) moved edge ids
                transitions, observed_version = self._observe(walked, count=False, origin=origin)
            csr = graph.to_csr()
            state = backend.prepare(csr, graph.node_params())
            if arrays_version != observed_version:
                edge_pos = csr.edge_pos
                parents_all = [p for p, _ in transitions]
                positions_all = [edge_pos[e] for _, e in transitions]
                arrays_version = observed_version
            n = len(parents_all)
            if cfg.shuffle and n > 1:
                order = list(range(n))
                rng.shuffle(order)
                parents = [parents_all[i] for i in order]
                positions = [positions_all[i] for i in order]
            else:
                parents, positions = parents_all, positions_all
            loss_sum = 0.0
            bs = cfg.batch_size
            for start in range(0, n, bs):
                batch_p = parents[start : start + bs]
                batch_q = positions[start : start + bs]
                loss_sum += backend.step(state, batch_p, batch_q, lr, act_lr, cfg.clip) * len(batch_p)
            weights, params = backend.finalize(state)
            graph.apply_csr_weights(csr, weights)
            graph.apply_node_params(params)
            merges = (graph.compress() if cfg.auto_compress else 0) + pending_merges
            pending_merges = 0
            stepped = self._window_epoch()  # the dynamic window's step, after the compression it rides on
            loss = loss_sum / n if n else 0.0
            graph.carry_counters()  # the epoch is over: wrap whatever reached the limit
            epoch = meta_add(meta, "epochs_total", 1)
            record = {
                "epoch": epoch,
                "loss": loss,
                "perplexity": math.exp(min(loss, _MAX_LOG_PPL)),
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
                "trigrams": graph.num_trigrams(),
                "compression_ratio": graph.compression_ratio(),
                "merges": merges,
                **({"splits": stepped["splits"], "window": stepped["window"]} if stepped else {}),
                "transitions": n,
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "lr": lr,
                "act_lr": act_lr,
            }
            if phase is not None:
                record["phase"] = phase
            stopping = plan is not None and plan.stop(k, loss)
            if stopping:
                record["early_stop"] = True
            self.history.append(record)
            records.append(record)
            if cfg.verbose:
                tag = f"[{phase}] " if phase else ""
                print(
                    f"{tag}epoch {epoch}: loss={loss:.4f} ppl={record['perplexity']:.2f} "
                    f"nodes={record['nodes']} edges={record['edges']} "
                    f"ratio={record['compression_ratio']:.2f} merges={merges} "
                    f"transitions={n} ({record['seconds']:.2f}s)",
                    file=sys.stderr,
                )
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self, epoch, "epoch", record)
            if stopping or (stop_event is not None and stop_event.is_set()):
                break
        if plan is not None:
            self.replay = plan.finish()
        return records

    # -- inference -----------------------------------------------------------

    def predict(
        self,
        prefix: str,
        length: int = 20,
        mode: str = "dijkstra",
        step_penalty: float = 0.0,
        temperature: float = 1.0,
        to_end: bool = False,
        max_length: int | None = None,
        k: int = 5,
        beam: int | None = None,
        traversal: str = DEFAULT_TRAVERSAL,
        penalty_scale: float = 1.0,
        merit_scale: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        diversity: float = 0.0,
    ) -> PathResult:
        """Continue ``prefix``.

        ``"dijkstra"`` returns the cheapest path emitting at least ``length``
        characters (or reaching END; ``to_end`` forces END).  There is no
        limit on the emitted characters unless ``max_length`` is given, in
        which case the continuation is capped there.  ``"beam"`` runs the
        beam search of :mod:`radixnet.beam` with the same goal and returns a
        :class:`~radixnet.beam.Prediction`: the best path plus the ``k`` most
        likely (``top``) and the ``k`` least likely (``bottom``) continuations
        - the search :meth:`generate` builds on.  ``"sample"`` walks
        stochastically until END or ``length`` (or ``max_length``) characters.
        ``result.text`` is the continuation only; ``result.full_text`` is
        ``prefix + text``.  When only a partial trigram of the prefix could be
        matched, the unmatched remainder of that trigram opens the
        continuation; a cap, when given, applies to the continuation as a
        whole (lead included).

        ``traversal`` is *what* the search is looking for, as opposed to
        ``mode``, which is how it looks: ``"reward"`` (the default) walks the
        model's own distribution, rewards and all, and ``"punishment"`` prices
        every step by the punishment it carries instead, so the cheapest path
        is the one that accumulated the **least punishment**
        (:mod:`radixnet.penalty`).  ``penalty_scale`` weighs the punishments
        and ``merit_scale`` what the corpus did; ``merit_scale = 0`` is the
        pure form, where nothing but the punishments decides.  Both are
        ignored by the reward traversal.

        ``"least-punished"`` is the count model's own traversal and is refused
        here: it ranks a walk by the blame on its **worst step**, which needs
        the judged paths this model does not keep (``../SPEC-LeastPunished.md``).

        ``top_k`` / ``top_p`` / ``min_p`` narrow what a sampled step draws from
        and ``diversity`` spreads the beam out (``../SPEC-SearchAndTraining.md``);
        Dijkstra, which returns one path, takes none of them.
        """
        if parse_traversal(traversal) == LEAST_PUNISHED:
            raise ValueError(
                f"the {traversal!r} traversal belongs to the count model; this model keeps no record of failure "
                "to rank a walk by (its punishments price a step instead: --traversal punishment)"
            )
        self._check_predict_args(prefix, length, max_length, k, beam)
        mode = (mode or "dijkstra").lower()
        if mode not in ("dijkstra", "beam", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'dijkstra', 'beam' or 'sample'")
        traversal = resolve_traversal(traversal)
        if mode != "dijkstra":
            return self._search(
                prefix, length, mode, k, beam, step_penalty, temperature, to_end, max_length,
                traversal=traversal, penalty_scale=penalty_scale, merit_scale=merit_scale,
                top_k=top_k, top_p=top_p, min_p=min_p, diversity=diversity,
            )
        graph = self.graph
        node, offset, lead = self._prefix_start(prefix)
        lead_len = self.encoding.length(lead)  # in units, as `length` is
        want = max(0, length - lead_len)
        cap: int | None
        if max_length is None and length == 0:
            cap = 0  # "emit nothing" stays empty even without a cap
            max_chars = 0
        elif max_length is None:
            cap = None  # no limit: the whole cheapest path is returned
            max_chars = None
        else:
            cap = max(length, max_length)
            max_chars = max(want, cap - lead_len)
        result = dijkstra_predict(
            graph, node, offset, min_chars=want, max_chars=max_chars,
            step_penalty=step_penalty, to_end=to_end,
            costs=traversal_costs(graph, traversal, penalty_scale, merit_scale),
        )
        if lead:
            joined = self.encoding.join(lead, result.text)
            result.text = joined if cap is None else self.encoding.truncate(joined, cap)
        result.full_text = self.encoding.join(prefix, result.text)
        return result


    # -- structural operations -----------------------------------------------

    def invert(self) -> None:
        """Flip every edge weight and activation amplitude (``graph.invert()``)."""
        self.graph.invert()

    def invert_paths(self, texts: Iterable[str] | str, mode: str = "activation", amounts=None, **options) -> dict:
        """Failures: move the activation (``a``) or the trained node value (``z``) of the nodes on their paths toward
        their negation - fully (amount 1: a sign flip) or partly (``amounts``: the worse the text, the larger).

        The local counterpart of 2NRL's global :meth:`invert`: only nodes the
        failed texts run through change, so those transitions become unlikely
        without touching the rest of the network - no negative training pass,
        no inversion of everything learned so far.  An edge's score
        ``w * f_p * f_c`` only changes sign when exactly one of its endpoints
        flips, so every *other* node of a path is updated (the parity that
        covers the most edges); a node shared by several texts takes the
        largest amount.  Amount 1 negates a value, 0.5 zeroes it (the path
        becomes neutral), smaller amounts attenuate it.  A text the structure
        cannot walk yet is registered first (nothing counted).  Two full
        flips of the same node cancel out, so use it for texts that are
        clearly wrong.  Returns ``{"texts", "flipped", "unit": "nodes",
        "mode", "amount_mean"}``.
        """
        texts, _ = self._clean_texts(texts)
        values = self._amounts(texts, amounts)
        chosen: dict[int, float] = {}
        for path, amount in zip(self._paths_of(texts), values):
            real = [n for n in path if n >= FIRST]
            if not real or amount <= 0:
                continue
            best: tuple[int, set[int]] | None = None
            for parity in (0, 1):
                candidate = {n for i, n in enumerate(real) if i % 2 == parity}
                state = set(chosen) | candidate
                gain = sum(1 for a, b in zip(path, path[1:]) if (a in state) != (b in state))
                if best is None or gain > best[0]:
                    best = (gain, candidate)
            for n in best[1]:
                chosen[n] = max(chosen.get(n, 0.0), amount)
        flipped = self.graph.flip_nodes(chosen, mode)
        meta_add(self.meta, "path_inversions", flipped)
        applied = [v for v in values if v > 0]
        return {
            "texts": len(texts), "flipped": flipped, "unit": "nodes", "mode": mode,
            "amount_mean": sum(applied) / len(applied) if applied else 0.0,
        }


    def two_nrl(
        self,
        bad: Iterable[str] | str,
        good: Iterable[str] | str,
        neg_epochs: int = 3,
        pos_epochs: int = 3,
        neg_lr: float = 0.05,
        pos_lr: float = 0.01,
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
        strength: float | None = None,
        bad_weights: Sequence[float] | None = None,
        good_weights: Sequence[float] | None = None,
        **overrides,
    ) -> dict:
        """2NRL: train on ``bad``, invert, fine-tune on ``good``.

        ``bad_weights`` (one per bad text, ``>= 0``) scale the negative
        phase per text: a text of weight ``w`` is trained with ``w * neg_lr``
        for weights and states and ``w * act_lr`` for the activation
        parameters, so the worse a failure the harder the model is pushed to
        reproduce it - to *blatantly fail on purpose* - before the inversion
        turns that into avoidance.  ``good_weights`` does the same for the
        positive phase (``w * pos_lr``, ``w * pos_lr / 10``): a rating, not a
        thumbs up, so a text rated 9 out of 10 is learned nine tenths as hard
        as a perfect one and a barely-acceptable text barely moves the
        network.  Texts of equal weight share a pass, heaviest first, and
        their records carry ``"weight"``.  ``strength`` is accepted for
        interface parity with the count / reward model and ignored here.

        The negative phase uses ``neg_lr``; the positive phase uses ``pos_lr``
        for weights and states and ``pos_lr / 10`` for the activation
        parameters.  Other :class:`TrainConfig` fields (``batch_size``,
        ``auto_compress``, ...) can be given as keyword ``overrides`` and apply
        to both phases.  Records carry ``"phase"`` (``"negative"`` /
        ``"positive"``).  If ``stop_event`` is set after the negative phase the
        network is still inverted (so it never stays in the garbage-favouring
        state) but the positive phase is skipped.  With a
        ``checkpoint_manager`` one checkpoint tagged ``"2nrl"`` is written at
        the end (step = number of 2NRL runs).
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(
                f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs/neg_lr/pos_lr"
            )
        if bad_weights is None:
            negative = self.train(
                bad, epochs=neg_epochs, lr=neg_lr, progress=progress,
                checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="negative", **overrides,
            )
        else:
            negative = []
            act_lr = overrides.pop("act_lr", None)
            base_act_lr = TrainConfig.act_lr if act_lr is None else act_lr
            for weight, group in _weight_groups(bad, bad_weights, "bad_weights"):
                if stop_event is not None and stop_event.is_set():
                    break
                records = self.train(
                    group, epochs=neg_epochs, lr=neg_lr * weight, act_lr=base_act_lr * weight,
                    checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="negative", **overrides,
                )
                for record in records:
                    record["weight"] = weight
                    if progress is not None:
                        progress(record)
                negative.extend(records)
        self.invert()
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            if good_weights is None:
                positive = self.train(
                    good, epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, progress=progress,
                    checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="positive", **overrides,
                )
            else:
                for weight, group in _weight_groups(good, good_weights, "good_weights"):
                    if stop_event is not None and stop_event.is_set():
                        break
                    records = self.train(
                        group, epochs=pos_epochs, lr=pos_lr * weight, act_lr=pos_lr / 10 * weight,
                        checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="positive", **overrides,
                    )
                    for record in records:
                        record["weight"] = weight
                        if progress is not None:
                            progress(record)
                    positive.extend(records)
        runs = meta_add(self.meta, "twonrl_runs", 1)
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, runs, "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    def reward(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 3,
        lr: float = 0.1,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        strength: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs up: a positive-phase pass over ``texts`` (``act_lr = lr / 10``); records carry ``phase="positive"``.

        ``weights`` (one per text, ``>= 0``) turns the thumbs up into a
        rating: a text of weight ``w`` is learned with ``w * lr`` and
        ``w * lr / 10``, so how good a text is decides how much of it the
        network keeps.  Texts of equal weight share a pass (heaviest first)
        and their records carry ``"weight"``; a weight of 0 is skipped.
        """
        if weights is None:
            return self.train(
                texts, epochs=epochs, lr=lr, act_lr=lr / 10, phase="positive", progress=progress,
                stop_event=stop_event, **overrides,
            )
        return self._weighted_passes(
            texts, weights, "weights", epochs=epochs, lr=lr, act_lr=lr / 10, phase="positive",
            progress=progress, stop_event=stop_event, **overrides,
        )

    def punish(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 2,
        lr: float = 0.5,
        weights: Sequence[float] | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        strength: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs down: a negative-phase pass over ``texts``, then the network is inverted (unless stopped).

        ``weights`` rates the failures the way :meth:`reward` rates the
        successes: the worse a text, the larger its share of ``lr``.
        """
        if weights is None:
            records = self.train(
                texts, epochs=epochs, lr=lr, phase="negative", progress=progress, stop_event=stop_event, **overrides,
            )
        else:
            act_lr = overrides.pop("act_lr", TrainConfig.act_lr)
            records = self._weighted_passes(
                texts, weights, "weights", epochs=epochs, lr=lr, act_lr=act_lr, phase="negative",
                progress=progress, stop_event=stop_event, **overrides,
            )
        if stop_event is None or not stop_event.is_set():
            self.invert()
        return records

    def _weighted_passes(
        self,
        texts: Iterable[str] | str,
        weights: Sequence[float],
        name: str,
        *,
        epochs: int,
        lr: float,
        act_lr: float,
        phase: str,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        **overrides,
    ) -> list[dict]:
        """One pass per group of equally weighted texts, the learning rates scaled by the weight."""
        records: list[dict] = []
        for weight, group in _weight_groups(texts, weights, name):
            if stop_event is not None and stop_event.is_set():
                break
            group_records = self.train(
                group, epochs=epochs, lr=lr * weight, act_lr=act_lr * weight, phase=phase,
                stop_event=stop_event, **overrides,
            )
            for record in group_records:
                record["weight"] = weight
                if progress is not None:
                    progress(record)
            records.extend(group_records)
        return records

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict:
        """Size, state and lifetime counters (JSON-serialisable)."""
        g = self.graph
        meta = self.meta
        return {
            "kind": self.kind,
            "nodes": g.num_nodes(),
            "edges": g.num_edges(),
            "trigrams": g.num_trigrams(),
            "grams": g.num_trigrams(),
            "encoding": str(self.encoding),
            "unit": self.encoding.unit,
            "units": self.encoding.units_name,
            "ngram": self.encoding.n,
            "stride": self.encoding.stride,
            "compression_ratio": g.compression_ratio(),
            "dynamic_window": g.dynamic_window.size,
            "inverted": g.inverted,
            "backend": self.backend.name,
            "device": self.backend.device,
            **meta_stats(meta, "epochs_total", "trained_chars", "trained_texts", "twonrl_runs"),
            "history_len": len(self.history),
            "last_loss": self.history[-1]["loss"] if self.history else None,
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (graph, history, meta, and the replay buffer when there is one)."""
        return self._with_replay({
            "format": MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "backend": self.backend.name,
            "graph": self.graph.to_dict(),
        })

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None) -> "RadixNet":
        """Inverse of :meth:`to_dict`; the backend is chosen afresh (not the saved name)."""
        if not isinstance(d, dict) or d.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a {MODEL_FORMAT} model document")
        version = int(d.get("version", 1))
        if version > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {MODEL_FORMAT} model version {version}")
        graph = RadixCyclicGraph.from_dict(d["graph"])
        model = cls(seed=graph.seed, backend=backend, device=device)
        model._adopt(graph)
        model.history = [dict(r) for r in d.get("history", [])]
        meta = cls._new_meta(graph.seed)
        meta.update(d.get("meta") or {})
        model.meta = carry_meta(meta)
        model._read_replay(d)
        return model


    def __repr__(self) -> str:
        g = self.graph
        return (
            f"RadixNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, inverted={g.inverted}, backend={self.backend.name!r})"
        )


# ---------------------------------------------------------------------------
# model kinds
# ---------------------------------------------------------------------------


def model_classes() -> dict[str, type[GraphModel]]:
    """``{kind: class}`` of every model kind (``"radix"``, ``"count"``, ``"word"``, ``"negative"``, ``"resonant"``)."""
    from .countnet import CountRewardNet  # local imports: they all build on this module
    from .negative import NegativeNet
    from .resonance import ResonantNet

    return {
        RadixNet.kind: RadixNet,
        CountRewardNet.kind: CountRewardNet,
        NegativeNet.kind: NegativeNet,
        ResonantNet.kind: ResonantNet,
    }


def model_kinds() -> list[dict]:
    """``[{"kind", "label", "description", "units"}]`` for menus and help texts."""
    return [
        {"kind": c.kind, "label": c.label, "description": c.description, "units": c.units}
        for c in model_classes().values()
    ]


def model_class(kind: str | None) -> type[GraphModel]:
    """The class of a model kind (``None`` / ``""`` = ``"radix"``); ``ValueError`` for an unknown kind."""
    key = (kind or RadixNet.kind).strip().lower()
    classes = model_classes()
    if key not in classes:
        raise ValueError(f"unknown model kind {kind!r}; expected one of: {', '.join(classes)}")
    return classes[key]


def new_model(
    kind: str | None = None, seed: int = 0, backend: str = "auto", device: str | None = None,
    encoding: Encoding | None = None,
) -> GraphModel:
    """A fresh model of ``kind``, in ``encoding`` (the default is character trigrams)."""
    return model_class(kind)(seed=seed, backend=backend, device=device, encoding=encoding)


def model_from_dict(d: dict, backend: str = "auto", device: str | None = None) -> GraphModel:
    """Rebuild a model of whatever kind a document holds (``format`` decides)."""
    if not isinstance(d, dict):
        raise ValueError("not a radixnet model document")
    fmt = d.get("format")
    for cls in model_classes().values():
        if fmt == getattr(cls, "format", None):
            return cls.from_dict(d, backend=backend, device=device)
    expected = ", ".join(str(getattr(cls, "format", cls.kind)) for cls in model_classes().values())
    raise ValueError(f"not a radixnet model document (format {fmt!r}; expected one of: {expected})")


def load_model(path: str, backend: str = "auto", device: str | None = None) -> GraphModel:
    """Load a model file of any kind (see :func:`model_from_dict`)."""
    return model_from_dict(read_json_file(path), backend=backend, device=device)
