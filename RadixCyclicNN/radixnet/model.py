"""RadixNet - training, prediction, generation, scoring, 2NRL and persistence.

The model ties the pieces together: :class:`~radixnet.encoding.Encoder`
turns text into trigrams, :class:`~radixnet.graph.RadixCyclicGraph` stores
the self-compressing structure and its parameters, a
:class:`~radixnet.backend.Backend` runs the one-hop local learning rule over
CSR mini-batches and :mod:`radixnet.search` finds the cheapest (Dijkstra) or a
sampled continuation.

2NRL (:meth:`RadixNet.two_nrl`) is the author's two-phase scheme: train on
bad / garbage data, :meth:`RadixNet.invert` the network (every edge weight
and every activation amplitude flips sign, so what was likely becomes
unlikely) and fine-tune on correct data with a smaller learning rate.
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

from .backend import Backend, get_backend
from .encoding import WINDOW, Decoder, Encoder
from .graph import END, START, RadixCyclicGraph
from .beam import Prediction, beam_predict, default_beam
from .search import PathResult, dijkstra_predict, sample_walk
from .schedule import preview_points

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
_W = WINDOW
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

    def to_dict(self) -> dict:
        """Plain-dict form (JSON-serialisable)."""
        return dataclasses.asdict(self)


_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _whole_text(result: PathResult, prefix: str) -> PathResult:
    """A generated result is a whole text: ``text`` becomes ``prefix + continuation`` (``full_text`` already is)."""
    result.full_text = prefix + result.text
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

    graph: RadixCyclicGraph
    encoder: Encoder
    decoder: Decoder
    backend: Backend
    history: list[dict]
    meta: dict
    seed: int

    @staticmethod
    def _new_meta(seed: int) -> dict:
        return {
            "created": _utc_now(),
            "seed": seed,
            "epochs_total": 0,
            "trained_chars": 0,
            "trained_texts": 0,
            "twonrl_runs": 0,
        }

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

    @staticmethod
    def _clean_texts(texts: Iterable[str] | str) -> tuple[list[str], int]:
        """Normalise the ``texts`` argument; returns ``(usable, skipped_short)``."""
        items = [texts] if isinstance(texts, str) else list(texts)
        kept: list[str] = []
        skipped = 0
        for t in items:
            if not isinstance(t, str):
                raise TypeError(f"texts must be strings, got {type(t).__name__}")
            if len(t) < _W:
                skipped += 1
            else:
                kept.append(t)
        return kept, skipped

    def _observe_grams(self, grams: list[list[str]], count: bool) -> list[tuple[int, int]]:
        """Register encoded texts structurally; returns the ``(parent_id, edge_id)`` transitions.

        A text observed later can split a node that an earlier text's
        transition points at (the edge moves to the new node), so whenever the
        first pass changed the structure a second, non-counting pass re-derives
        every transition from the final structure - that pass never splits.
        """
        graph = self.graph
        observe = graph.observe_sequence
        before = graph.structure_version
        transitions: list[tuple[int, int]] = []
        extend = transitions.extend
        for g in grams:
            extend(observe(g, count))
        if graph.structure_version != before:
            transitions = []
            extend = transitions.extend
            for g in grams:
                extend(observe(g, False))
        return transitions

    def _observe(self, texts: list[str], count: bool) -> tuple[list[tuple[int, int]], int]:
        """Register every text structurally; returns ``(transitions, structure_version)``."""
        transitions = self._observe_grams([self.encoder.encode(t) for t in texts], count)
        return transitions, self.graph.structure_version

    # -- locating a prefix ---------------------------------------------------

    def _best_trigram(self, key: str) -> tuple[int, int] | None:
        """Most-visited ``(node, offset)`` holding a trigram that starts with ``key``."""
        count = self.graph.count
        best: tuple[int, int] | None = None
        best_rank: tuple[int, int, int] | None = None
        for t, (node, off) in self.graph.trigram_index.items():
            if t.startswith(key):
                rank = (-count[node], node, off)
                if best_rank is None or rank < best_rank:
                    best, best_rank = (node, off), rank
        return best

    def _best_node_with_prefix(self, prefix: str) -> int | None:
        """Most-visited real node whose label starts with ``prefix``."""
        g = self.graph
        best: int | None = None
        best_count = -1
        for node in range(2, len(g.labels)):
            if g.alive[node] and g.labels[node].startswith(prefix):
                c = g.count[node]
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
        n = len(prefix)
        if n == 0:
            return START, 0, 0
        if n >= _W:
            loc = self.graph.lookup(prefix[-_W:])
            if loc is not None:
                return loc[0], loc[1], _W
            for m in (_W - 1, 1):
                found = self._best_trigram(prefix[-m:])
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
        node, offset, matched = self._locate(prefix)
        lead = "" if node == START or matched >= _W else self.graph.labels[node][offset + matched : offset + _W]
        return node, offset, lead

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
    ) -> Prediction:
        """The prediction search from where ``prefix`` ends: ``"beam"`` (the ``k`` most and least likely
        continuations) or ``"sample"`` (one stochastic walk).  The result *is* the best path and carries ``top`` /
        ``bottom``; ``length``, ``to_end`` and ``max_length`` follow :meth:`RadixNet.predict`."""
        graph = self.graph
        node, offset, lead = self._prefix_start(prefix)
        want = max(0, length - len(lead))
        cap: int | None
        if mode == "beam":
            if max_length is None and length == 0:
                cap, max_chars = 0, 0
            elif max_length is None:
                cap, max_chars = None, None
            else:
                cap = max(length, max_length)
                max_chars = max(want, cap - len(lead))
            top, bottom, expanded = beam_predict(
                graph, node, offset, min_chars=want, k=k, beam=beam, max_chars=max_chars,
                step_penalty=step_penalty, to_end=to_end,
            )
            width = default_beam(k) if beam is None else int(beam)
        else:
            cap = max_length if max_length is not None else length
            walk = sample_walk(graph, node, offset, max_chars=max(0, cap - len(lead)), temperature=temperature, rng=rng)
            top, bottom, expanded, width = [walk], [], walk.expanded, 0
        for result in top + bottom:
            if lead:
                result.text = lead + result.text if cap is None else (lead + result.text)[:cap]
            result.full_text = prefix + result.text
        best = top[0] if top else PathResult(text=lead if cap is None else lead[: cap or 0], labels=[graph.labels[node]], node_ids=[node])
        if not top:
            best.full_text = prefix + best.text
        return Prediction(
            text=best.text, labels=list(best.labels), node_ids=list(best.node_ids), cost=best.cost,
            step_costs=list(best.step_costs), expanded=expanded, reached_end=best.reached_end, full_text=best.full_text,
            top=top, bottom=bottom, k=k, beam=width, mode=mode,
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
        included; ``probability`` is ``exp(-cost)``.
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
        if mode == "sample":
            rng = random.Random(seed) if seed is not None else None
            results: list[PathResult] = []
            for _ in range(count):
                walk = self._search(prefix, max_length, "sample", 0, None, 0.0, temperature, False, max_length, rng=rng)
                results.append(_whole_text(walk, prefix))
            return results
        if mode == "dijkstra":
            best = self.predict(prefix, length=0, mode="dijkstra", to_end=True, max_length=max_length, step_penalty=step_penalty)
            return [_whole_text(best, prefix)]
        found = self.predict(
            prefix, length=0, mode="beam", k=count, beam=beam, to_end=True, max_length=max_length,
            step_penalty=step_penalty,
        )
        return [_whole_text(result, prefix) for result in found.top]

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
        if p != START and offset + _W != len(g.labels[p]):
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
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        grams = self.encoder.encode(text)
        chars = len(text)
        if not grams:
            return {"log_prob": 0.0, "per_char": 0.0, "chars": chars, "transitions": 0, "unknown_transitions": 0}
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

    def __init__(self, seed: int = 0, backend: str = "auto", device: str | None = None) -> None:
        self.seed = int(seed)
        self.graph = RadixCyclicGraph(seed=self.seed)
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
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
        """
        cfg = _resolve_config(config, overrides)
        texts, skipped_short = self._clean_texts(texts)
        graph = self.graph
        backend = self.backend
        rng = graph.rng
        meta = self.meta
        records: list[dict] = []

        transitions, observed_version = self._observe(texts, count=True)
        meta["trained_texts"] += len(texts)
        meta["trained_chars"] += sum(len(t) for t in texts)
        pending_merges = graph.compress() if cfg.auto_compress else 0

        parents_all: list[int] = []
        positions_all: list[int] = []
        arrays_version = -1
        rates = cfg.rates()  # (lr, act_lr) per epoch: the schedules are graph functions of the epoch
        for k in range(cfg.epochs):
            t0 = time.perf_counter()
            lr, act_lr = rates[k]
            if graph.structure_version != observed_version:
                # a merge (or an external structural change) moved edge ids
                transitions, observed_version = self._observe(texts, count=False)
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
            loss = loss_sum / n if n else 0.0
            meta["epochs_total"] += 1
            epoch = meta["epochs_total"]
            record = {
                "epoch": epoch,
                "loss": loss,
                "perplexity": math.exp(min(loss, _MAX_LOG_PPL)),
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
                "trigrams": graph.num_trigrams(),
                "compression_ratio": graph.compression_ratio(),
                "merges": merges,
                "transitions": n,
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "lr": lr,
                "act_lr": act_lr,
            }
            if phase is not None:
                record["phase"] = phase
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
            if stop_event is not None and stop_event.is_set():
                break
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
        """
        self._check_predict_args(prefix, length, max_length, k, beam)
        mode = (mode or "dijkstra").lower()
        if mode not in ("dijkstra", "beam", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'dijkstra', 'beam' or 'sample'")
        if mode != "dijkstra":
            return self._search(prefix, length, mode, k, beam, step_penalty, temperature, to_end, max_length)
        graph = self.graph
        node, offset, lead = self._prefix_start(prefix)
        want = max(0, length - len(lead))
        cap: int | None
        if max_length is None and length == 0:
            cap = 0  # "emit nothing" stays empty even without a cap
            max_chars = 0
        elif max_length is None:
            cap = None  # no limit: the whole cheapest path is returned
            max_chars = None
        else:
            cap = max(length, max_length)
            max_chars = max(want, cap - len(lead))
        result = dijkstra_predict(
            graph, node, offset, min_chars=want, max_chars=max_chars,
            step_penalty=step_penalty, to_end=to_end,
        )
        if lead:
            result.text = lead + result.text if cap is None else (lead + result.text)[:cap]
        result.full_text = prefix + result.text
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
            real = [n for n in path if n > END]
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
        self.meta["path_inversions"] = self.meta.get("path_inversions", 0) + flipped
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
        self.meta["twonrl_runs"] += 1
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, self.meta["twonrl_runs"], "2nrl", last)
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
            "compression_ratio": g.compression_ratio(),
            "inverted": g.inverted,
            "backend": self.backend.name,
            "device": self.backend.device,
            "epochs_total": meta["epochs_total"],
            "trained_chars": meta["trained_chars"],
            "trained_texts": meta["trained_texts"],
            "twonrl_runs": meta["twonrl_runs"],
            "history_len": len(self.history),
            "last_loss": self.history[-1]["loss"] if self.history else None,
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (graph, history, meta)."""
        return {
            "format": MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "backend": self.backend.name,
            "graph": self.graph.to_dict(),
        }

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
        model.graph = graph
        model.history = [dict(r) for r in d.get("history", [])]
        meta = cls._new_meta(graph.seed)
        meta.update(d.get("meta") or {})
        model.meta = meta
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
    """``{kind: class}`` of every model kind (``"radix"`` and ``"count"``)."""
    from .countnet import CountRewardNet  # local import: countnet builds on this module

    return {RadixNet.kind: RadixNet, CountRewardNet.kind: CountRewardNet}


def model_kinds() -> list[dict]:
    """``[{"kind", "label", "description"}]`` for menus and help texts."""
    return [{"kind": c.kind, "label": c.label, "description": c.description} for c in model_classes().values()]


def model_class(kind: str | None) -> type[GraphModel]:
    """The class of a model kind (``None`` / ``""`` = ``"radix"``); ``ValueError`` for an unknown kind."""
    key = (kind or RadixNet.kind).strip().lower()
    classes = model_classes()
    if key not in classes:
        raise ValueError(f"unknown model kind {kind!r}; expected one of: {', '.join(classes)}")
    return classes[key]


def new_model(kind: str | None = None, seed: int = 0, backend: str = "auto", device: str | None = None) -> GraphModel:
    """A fresh model of ``kind``."""
    return model_class(kind)(seed=seed, backend=backend, device=device)


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
