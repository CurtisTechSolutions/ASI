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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from .backend import Backend, get_backend
from .encoding import WINDOW, Decoder, Encoder
from .graph import END, START, RadixCyclicGraph
from .search import PathResult, dijkstra_predict, sample_walk
from .schedule import preview_points

__all__ = ["TrainConfig", "RadixNet", "UNKNOWN_PROB", "MODEL_FORMAT", "MODEL_FORMAT_VERSION"]

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

    def rates(self) -> list[tuple[float, float]]:
        """``(lr, act_lr)`` for every epoch of this config, schedules applied (constant when none are set)."""
        if self.lr_schedule is None and self.act_lr_schedule is None:
            return [(self.lr, self.act_lr)] * self.epochs
        points = preview_points(self.lr_schedule, self.act_lr_schedule, self.epochs, self.lr, self.act_lr)
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


class RadixNet:
    """The self-compressing cyclic-graph network.

    ``graph`` holds structure and parameters, ``backend`` runs the learning
    rule, ``history`` collects one record per trained epoch and ``meta``
    keeps lifetime counters.  All records are plain JSON-serialisable dicts.
    """

    def __init__(self, seed: int = 0, backend: str = "auto", device: str | None = None) -> None:
        self.seed = int(seed)
        self.graph = RadixCyclicGraph(seed=self.seed)
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
        self.backend: Backend = get_backend(backend, device)
        self.history: list[dict] = []
        self.meta: dict = self._new_meta(self.seed)

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

    # -- training ------------------------------------------------------------

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

    def _observe(self, texts: list[str], count: bool) -> tuple[list[tuple[int, int]], int]:
        """Register every text structurally; returns ``(transitions, structure_version)``.

        A text observed later can split a node that an earlier text's
        transition points at (the edge moves to the new node), so whenever the
        first pass changed the structure a second, non-counting pass re-derives
        every transition from the final structure - that pass never splits.
        """
        graph = self.graph
        observe = graph.observe_sequence
        grams = [self.encoder.encode(t) for t in texts]
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
        return transitions, graph.structure_version

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
    ) -> PathResult:
        """Continue ``prefix``.

        ``"dijkstra"`` returns the cheapest path emitting at least ``length``
        characters (or reaching END; ``to_end`` forces END).  There is no
        limit on the emitted characters unless ``max_length`` is given, in
        which case the continuation is capped there.  ``"sample"`` walks
        stochastically until END or ``length`` (or ``max_length``) characters.
        ``result.text`` is the continuation only; ``result.full_text`` is
        ``prefix + text``.  When only a partial trigram of the prefix could be
        matched, the unmatched remainder of that trigram opens the
        continuation; a cap, when given, applies to the continuation as a
        whole (lead included).
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        mode = (mode or "dijkstra").lower()
        if mode not in ("dijkstra", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'dijkstra' or 'sample'")
        graph = self.graph
        node, offset, matched = self._locate(prefix)
        lead = "" if node == START or matched >= _W else graph.labels[node][offset + matched : offset + _W]
        want = max(0, length - len(lead))
        cap: int | None
        if mode == "dijkstra":
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
        else:
            cap = max_length if max_length is not None else length
            result = sample_walk(graph, node, offset, max_chars=max(0, cap - len(lead)), temperature=temperature)
        if lead:
            result.text = lead + result.text if cap is None else (lead + result.text)[:cap]
        result.full_text = prefix + result.text
        return result

    def generate(
        self,
        max_length: int = 60,
        mode: str = "sample",
        temperature: float = 1.0,
        count: int = 1,
        seed: int | None = None,
    ) -> list[PathResult]:
        """Generate texts from START.

        ``"sample"`` draws ``count`` stochastic walks (``seed`` gives a private
        RNG; otherwise the model's seeded RNG is consumed); ``"dijkstra"``
        returns the single cheapest path to END, so the list has one entry.
        """
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        mode = (mode or "sample").lower()
        if mode not in ("dijkstra", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'dijkstra' or 'sample'")
        if count == 0:
            return []
        graph = self.graph
        if mode == "dijkstra":
            result = dijkstra_predict(graph, START, 0, min_chars=0, max_chars=max_length, to_end=True)
            result.full_text = result.text
            return [result]
        rng = random.Random(seed) if seed is not None else None
        results: list[PathResult] = []
        for _ in range(count):
            result = sample_walk(graph, START, 0, max_chars=max_length, temperature=temperature, rng=rng)
            result.full_text = result.text
            results.append(result)
        return results

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

    # -- structural operations -----------------------------------------------

    def invert(self) -> None:
        """Flip every edge weight and activation amplitude (``graph.invert()``)."""
        self.graph.invert()

    def compress(self) -> int:
        """Merge all unary chains; returns the number of merges."""
        return self.graph.compress()

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
        **overrides,
    ) -> dict:
        """2NRL: train on ``bad``, invert, fine-tune on ``good``.

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
        negative = self.train(
            bad, epochs=neg_epochs, lr=neg_lr, progress=progress,
            checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="negative", **overrides,
        )
        self.invert()
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            positive = self.train(
                good, epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, progress=progress,
                checkpoint_manager=checkpoint_manager, stop_event=stop_event, phase="positive", **overrides,
            )
        self.meta["twonrl_runs"] += 1
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, self.meta["twonrl_runs"], "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict:
        """Size, state and lifetime counters (JSON-serialisable)."""
        g = self.graph
        meta = self.meta
        return {
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

    def save(self, path: str) -> None:
        """Write the model as JSON (gzip when ``path`` ends with ``.gz``), atomically."""
        payload = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        write_bytes_atomic(path, payload, use_gzip=path.endswith(".gz"))

    @classmethod
    def load(cls, path: str, backend: str = "auto", device: str | None = None) -> "RadixNet":
        """Load a model written by :meth:`save`."""
        return cls.from_dict(read_json_file(path), backend=backend, device=device)

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"RadixNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, inverted={g.inverted}, backend={self.backend.name!r})"
        )
