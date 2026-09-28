"""RadixTreeNet - training, prediction, scoring, 2NRL and persistence over a :class:`RadixTree`.

The model ties the pieces together the way ``radixnet.model.RadixNet`` does:
:class:`~radixtree.encoding.Encoding` turns text into grams,
:class:`~radixtree.tree.RadixTree` stores the structure and its parameters,
the **one-hop local rule** below trains them, and :mod:`radixtree.search`
finds the cheapest or a sampled continuation.

The learning rule is the cyclic graph's, unchanged.  For a transition
``p -> c`` observed in training the loss is the negative log-softmax of the
edge score ``w_c * f_p(z_p) * f_c(z_c)`` among all options at ``p``; gradients
touch only the in-edge weights of ``p``'s children, the states ``z`` and the
activation parameters ``(a, b, h, k)`` of ``p`` and its children.  Nothing
propagates deeper.  What differs is where the transitions come from: a text is
a transition at every context it passes through - from START, and from the
root at every position - so a corpus position trains every context that
predicts it.

2NRL (:meth:`RadixTreeNet.two_nrl`) is the author's *Double-Negative
Reinforcement Learning*: train on bad data, :meth:`RadixTreeNet.invert` the
network (every weight and every activation flips sign, so what was likely
becomes unlikely) and fine-tune on correct data with a smaller learning rate.
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
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from .activation import MIN_B
from .encoding import Encoding
from .search import PathResult, cheapest_path, check_costs, sample_walk
from .tree import ROOT, START, RadixTree

__all__ = [
    "MAX_LEGS", "MODEL_FORMAT", "MODEL_FORMAT_VERSION", "UNKNOWN_PROB",
    "RadixTreeNet", "TrainConfig", "load_model", "read_json_file", "write_bytes_atomic",
]

MODEL_FORMAT = "radixtree"
MODEL_FORMAT_VERSION = 1

UNKNOWN_PROB = 1e-6
"""Probability charged by :meth:`RadixTreeNet.score` for a gram the context has never been followed by -
``radixnet.model.UNKNOWN_PROB``, so the two models' scores are read on the same scale."""

MAX_LEGS = 64
"""How many times a walk may re-enter a bounded tree before it stops - the clock a bounded window needs."""

MODES = ("dijkstra", "sample", "slide")
"""How a prediction walks: the cheapest path, a sampled path, or one token per query of the tree."""

MAX_SLIDE = 2000
"""Characters a token-by-token walk to END may emit before it is stopped: a window that forgets can go round -
the cycle is back, outside the tree - and this is its clock."""

_LOG_UNKNOWN = math.log(UNKNOWN_PROB)
_MAX_LOG_PPL = 700.0

ProgressFn = Callable[[dict], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes_atomic(path: str, data: bytes, use_gzip: bool = False) -> None:
    """Write ``data`` to ``path`` via a temporary file and ``os.replace``; a reader never sees a half-written file."""
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
    """Load a JSON document, gunzipping it when it carries the gzip magic (the content decides, not the suffix)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


@dataclass
class TrainConfig:
    """Hyper-parameters of one :meth:`RadixTreeNet.train` call - the cyclic model's defaults."""

    epochs: int = 5
    lr: float = 0.05
    act_lr: float = 0.005
    """Learning rate of the per-node activation parameters ``a, b, h, k``."""
    batch_size: int = 256
    """Transitions per gradient step."""
    clip: float = 5.0
    auto_compress: bool = True
    """Merge unary chains before the first epoch and after every epoch (a no-op on a tree of every suffix)."""
    shuffle: bool = True
    """Shuffle the transitions every epoch (with the model's seeded RNG)."""
    verbose: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` for values that cannot be trained with."""
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        for name in ("lr", "act_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {value}")
        if not (self.clip > 0):
            raise ValueError(f"clip must be > 0, got {self.clip}")

    def to_dict(self) -> dict:
        """Plain-dict form (JSON-serialisable)."""
        return dataclasses.asdict(self)


_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _resolve_config(config: TrainConfig | None, overrides: dict) -> TrainConfig:
    cfg = TrainConfig() if config is None else config
    if overrides:
        unknown = sorted(set(overrides) - _CONFIG_FIELDS)
        if unknown:
            raise TypeError(f"unknown train option(s): {', '.join(unknown)}")
        cfg = dataclasses.replace(cfg, **overrides)
    cfg.validate()
    return cfg


class RadixTreeNet:
    """The radix tree network: a :class:`RadixTree`, the one-hop rule, and the searches over it.

    ``depth`` bounds how many symbols a root path may hold (``None``, the
    default, is the tree of every whole suffix); ``min_count`` is how often a
    context must have been seen before a prediction or a score trusts it (1:
    the deepest known context always decides).
    """

    kind = "radix-tree"
    label = "RadixTreeNet (sine activation, no cycles)"

    def __init__(self, seed: int = 0, depth: int | None = None, encoding: Encoding | None = None, min_count: int = 1) -> None:
        if min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {min_count}")
        self.seed = int(seed)
        self.tree = RadixTree(seed=self.seed, encoding=encoding, depth=depth)
        self.min_count = int(min_count)
        self._acc: tuple[list[float], ...] | None = None
        self.history: list[dict] = []
        self.meta: dict = {
            "created": _utc_now(), "seed": self.seed, "epochs_total": 0, "trained_chars": 0,
            "trained_texts": 0, "twonrl_runs": 0, "path_inversions": 0,
        }

    @property
    def encoding(self) -> Encoding:
        """The encoding the tree was built in."""
        return self.tree.encoding

    @property
    def depth(self) -> int | None:
        """The bound on a root path, in symbols (``None``: every whole suffix)."""
        return self.tree.depth

    # -- texts ---------------------------------------------------------------

    def _clean_texts(self, texts: Iterable[str] | str) -> tuple[list[str], int]:
        """Strings only, the ones too short for a single gram dropped (and counted)."""
        items = [texts] if isinstance(texts, str) else list(texts)
        enc = self.encoding
        n = enc.n
        kept: list[str] = []
        skipped = 0
        for t in items:
            if not isinstance(t, str):
                raise TypeError("texts must be strings")
            if enc.length(t) >= n:
                kept.append(t)
            else:
                skipped += 1
        return kept, skipped

    def _observe(self, texts: list[str], count: bool) -> tuple[list[tuple[int, int]], int]:
        """Every transition of every window of every text, and the structure version it was read at."""
        tree = self.tree
        encode = self.encoding.encode
        grams_of = [encode(text) for text in texts]
        before = tree.structure_version
        transitions: list[tuple[int, int]] = []
        for grams in grams_of:
            if grams:
                transitions.extend(tree.observe_sequence(grams, count=count))
        if tree.structure_version != before:
            # a later text may have split a node an earlier text's transition named: read every transition again
            # against the structure as it now stands (every window is a root path now, so this changes nothing)
            transitions = []
            for grams in grams_of:
                if grams:
                    transitions.extend(tree.observe_sequence(grams, count=False))
        return transitions, tree.structure_version

    # -- the one-hop rule ----------------------------------------------------

    def _accumulate(self, batch: list[tuple[int, int]]) -> tuple[float, list[int]]:
        """The summed loss of a batch, its (unscaled) gradients left in the accumulators, and the nodes they touch.

        The batch is grouped by parent first: the loss and the gradient of the
        one-hop rule are sums over transitions, so a parent that occurs ``m``
        times with its children as targets ``n_c`` times contributes
        ``m * logsumexp - sum(n_c * score_c)`` and ``m * q_c - n_c`` at every
        child - one softmax per parent per batch instead of one per
        transition, and the same numbers to the last bit of arithmetic order.
        Gradients accumulate in the flat per-node arrays of ``self._acc``
        (``gw, gz, ga, gb, gh, gk``), kept between calls; the caller reads the
        touched entries and must reset them (:meth:`_reset`), so a batch costs
        its parents' fan-out and nothing of the tree's size
        (``radixnet.backend`` keeps its accumulators the same way).
        """
        tree = self.tree
        z, a, b, h, k, w = tree.z, tree.a, tree.b, tree.h, tree.k, tree.w
        child_items = tree.child_items
        sin, cos, exp, log = math.sin, math.cos, math.exp, math.log
        total_nodes = len(z)
        acc = self._acc
        if acc is None or len(acc[0]) != total_nodes:
            acc = self._acc = tuple([0.0] * total_nodes for _ in range(6))
        gw, gz, ga, gb, gh, gk = acc
        groups: dict[int, dict[int, int]] = {}
        for p, target in batch:
            targets = groups.get(p)
            if targets is None:
                targets = groups[p] = {}
            targets[target] = targets.get(target, 0) + 1
        parts: dict[int, tuple[float, float, float, float, float]] = {}
        touched: list[int] = []
        total = 0.0
        for p, targets in groups.items():
            kids = child_items(p)
            if len(kids) < 2:
                continue  # a softmax over one option is 1: no loss, no gradient
            for c in targets:
                if c not in kids:
                    raise ValueError(f"node {c} is not an option at node {p}")
            pp = parts.get(p)
            if pp is None:
                d = z[p] - h[p]
                u = b[p] * d
                su, cu = sin(u), cos(u)
                pp = parts[p] = (a[p] * su + k[p], a[p] * b[p] * cu, su, a[p] * d * cu, -a[p] * b[p] * cu)
                touched.append(p)
            fp = pp[0]
            facts = []
            scores = []
            for c in kids:
                cc = parts.get(c)
                if cc is None:
                    d = z[c] - h[c]
                    u = b[c] * d
                    su, cu = sin(u), cos(u)
                    cc = parts[c] = (a[c] * su + k[c], a[c] * b[c] * cu, su, a[c] * d * cu, -a[c] * b[c] * cu)
                    touched.append(c)
                facts.append(cc)
                scores.append(w[c] * fp * cc[0])
            m = max(scores)
            ex = [exp(s - m) for s in scores]
            ssum = sum(ex)
            times = sum(targets.values())
            lse = m + log(ssum)
            inv = times / ssum
            gfp = 0.0
            for idx, c in enumerate(kids):
                hits = targets.get(c, 0)
                if hits:
                    total += hits * (lse - scores[idx])
                    g = ex[idx] * inv - hits
                else:
                    g = ex[idx] * inv
                cc = facts[idx]
                fc = cc[0]
                wc = w[c]
                gw[c] += g * fp * fc
                gfp += g * wc * fc
                gfc = g * wc * fp
                gz[c] += gfc * cc[1]
                ga[c] += gfc * cc[2]
                gb[c] += gfc * cc[3]
                gh[c] += gfc * cc[4]
                gk[c] += gfc
            gz[p] += gfp * pp[1]
            ga[p] += gfp * pp[2]
            gb[p] += gfp * pp[3]
            gh[p] += gfp * pp[4]
            gk[p] += gfp
        return total, touched

    def _reset(self, touched: list[int]) -> None:
        """Zero the accumulator entries a batch touched."""
        gw, gz, ga, gb, gh, gk = self._acc
        for i in touched:
            gw[i] = gz[i] = ga[i] = gb[i] = gh[i] = gk[i] = 0.0

    def step(self, batch: list[tuple[int, int]], lr: float, act_lr: float, clip: float = 5.0) -> float:
        """One mini-batch of the one-hop rule; returns the mean loss *before* the update.

        Gradients are summed over the batch, divided by its size, clipped
        element-wise to ``[-clip, clip]`` and applied once: ``w`` and ``z`` at
        ``lr``, ``a, b, h, k`` at ``act_lr``; ``b`` is kept ``>= MIN_B``.
        """
        n = len(batch)
        if n == 0:
            return 0.0
        total, touched = self._accumulate(batch)
        gw, gz, ga, gb, gh, gk = self._acc
        tree = self.tree
        scale = 1.0 / n
        neg = -clip
        w, z, a, b, h, k = tree.w, tree.z, tree.a, tree.b, tree.h, tree.k
        for c in touched:
            g = gw[c] * scale
            w[c] -= lr * (clip if g > clip else neg if g < neg else g)
            g = gz[c] * scale
            z[c] -= lr * (clip if g > clip else neg if g < neg else g)
            g = ga[c] * scale
            a[c] -= act_lr * (clip if g > clip else neg if g < neg else g)
            g = gb[c] * scale
            nb = b[c] - act_lr * (clip if g > clip else neg if g < neg else g)
            b[c] = nb if nb > MIN_B else MIN_B
            g = gh[c] * scale
            h[c] -= act_lr * (clip if g > clip else neg if g < neg else g)
            g = gk[c] * scale
            k[c] -= act_lr * (clip if g > clip else neg if g < neg else g)
            gw[c] = gz[c] = ga[c] = gb[c] = gh[c] = gk[c] = 0.0
        tree.version += 1
        return total * scale

    def loss_of(self, batch: list[tuple[int, int]]) -> float:
        """Mean batch loss at the current parameters (no update)."""
        if not batch:
            return 0.0
        total, touched = self._accumulate(batch)
        self._reset(touched)
        return total / len(batch)

    def gradients_of(self, batch: list[tuple[int, int]]) -> dict:
        """Mean (unclipped) gradients of the batch loss without applying them, keyed ``w, z, a, b, h, k``."""
        out: dict = {"loss": 0.0, "w": {}, "z": {}, "a": {}, "b": {}, "h": {}, "k": {}}
        if not batch:
            return out
        total, touched = self._accumulate(batch)
        scale = 1.0 / len(batch)
        out["loss"] = total * scale
        for name, grads in zip(("w", "z", "a", "b", "h", "k"), self._acc):
            out[name] = {i: grads[i] * scale for i in touched if grads[i] != 0.0}
        self._reset(touched)
        return out

    # -- training ------------------------------------------------------------

    def train(
        self,
        texts: Iterable[str] | str,
        config: TrainConfig | None = None,
        *,
        phase: str | None = None,
        progress: ProgressFn | None = None,
        **overrides,
    ) -> list[dict]:
        """Train on ``texts`` (one string or an iterable of strings).

        Every text is first registered structurally - every window of it, from
        START and from the root - and its transitions collected; then, per
        epoch, the transitions are shuffled with the model's seeded RNG and fed
        to :meth:`step` in mini-batches.  With ``auto_compress`` unary chains
        are merged before the first epoch and after every one, and because a
        merge moves nodes the transitions are re-derived whenever the
        structure changed.  Keyword ``overrides`` replace individual
        :class:`TrainConfig` fields; ``phase`` is stamped on every record.
        Returns the epoch records (also appended to :attr:`history`).
        """
        cfg = _resolve_config(config, overrides)
        texts, skipped_short = self._clean_texts(texts)
        tree = self.tree
        rng = tree.rng
        meta = self.meta
        records: list[dict] = []
        transitions, observed = self._observe(texts, count=True)
        meta["trained_texts"] += len(texts)
        meta["trained_chars"] += sum(self.encoding.length(t) for t in texts)
        pending_merges = tree.compress() if cfg.auto_compress else 0
        if tree.structure_version != observed:
            transitions, observed = self._observe(texts, count=False)  # a merge moved nodes
        for _ in range(cfg.epochs):
            t0 = time.perf_counter()
            if tree.structure_version != observed:
                transitions, observed = self._observe(texts, count=False)
            n = len(transitions)
            if cfg.shuffle and n > 1:
                order = list(range(n))
                rng.shuffle(order)
                walked = [transitions[i] for i in order]
            else:
                walked = transitions
            loss_sum = 0.0
            bs = cfg.batch_size
            for start in range(0, n, bs):
                batch = walked[start : start + bs]
                loss_sum += self.step(batch, cfg.lr, cfg.act_lr, cfg.clip) * len(batch)
            merges = (tree.compress() if cfg.auto_compress else 0) + pending_merges
            pending_merges = 0
            loss = loss_sum / n if n else 0.0
            meta["epochs_total"] += 1
            record = {
                "epoch": meta["epochs_total"],
                "loss": loss,
                "perplexity": math.exp(min(loss, _MAX_LOG_PPL)),
                "nodes": tree.num_nodes(),
                "ends": tree.num_ends(),
                "edges": tree.num_edges(),
                "grams": tree.num_grams(),
                "compression_ratio": tree.compression_ratio(),
                "merges": merges,
                "transitions": n,
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "lr": cfg.lr,
                "act_lr": cfg.act_lr,
            }
            if phase is not None:
                record["phase"] = phase
            self.history.append(record)
            records.append(record)
            if cfg.verbose:
                tag = f"[{phase}] " if phase else ""
                print(
                    f"{tag}epoch {record['epoch']}: loss={loss:.4f} ppl={record['perplexity']:.2f} "
                    f"nodes={record['nodes']} ends={record['ends']} grams={record['grams']} "
                    f"ratio={record['compression_ratio']:.2f} merges={merges} transitions={n} "
                    f"({record['seconds']:.2f}s)",
                    file=sys.stderr,
                )
            if progress is not None:
                progress(record)
        return records

    # -- inference -----------------------------------------------------------

    def _start(self, prefix: str) -> tuple[int, int, bool, int]:
        """Where a prediction for ``prefix`` begins: ``(node, offset, include_context, cut)``.

        The deepest usable context of the prefix, continued from the matched
        gram; a prefix too short for a gram matches the most visited child of
        START whose label begins with it, and ``cut`` is how many characters
        of that label the prefix already covers.  Nothing known: START, a new
        text.
        """
        tree = self.tree
        grams = self.encoding.encode(prefix)
        if grams:
            loc = tree.locate(grams, self.min_count)
            if loc is not None:
                return loc[0], loc[1], False, 0
            return START, -1, True, 0
        if prefix:
            enc = self.encoding
            best = -1
            for c in tree.children[START].values():
                if enc.has_unit_prefix(tree.labels[c], prefix) and (best < 0 or tree.count[c] > tree.count[best]):
                    best = c
            if best >= 0:
                return best, 0, True, enc.length(prefix)
        return START, -1, True, 0

    @staticmethod
    def _check_predict_args(
        prefix: str, length: int, max_length: int | None, mode: str, max_legs: int, window: int | None
    ) -> None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        if max_legs < 1:
            raise ValueError(f"max_legs must be >= 1, got {max_legs}")
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 grams or None, got {window}")

    def context(self, grams: list[str], window: int | None = None) -> tuple[int, int] | None:
        """One query of the tree: the deepest usable context of the last ``window`` grams (all of them: ``None``).

        A window cut out of a longer history is never matched from START - it
        does not begin a text - while a history the window still holds whole is.
        """
        tree = self.tree
        if window is not None and len(grams) > window:
            loc = tree.locate(grams[len(grams) - window :], self.min_count, anchored=False)
        else:
            loc = tree.locate(grams, self.min_count)
        return None if loc is None else (loc[0], loc[1])

    def next_token(
        self, node: int, offset: int, temperature: float = 0.0, rng: random.Random | None = None, costs: str = "logprob"
    ) -> tuple[int, str | None, float]:
        """The token that follows a located context: ``(node, gram, cost)``, with ``gram`` ``None`` for END.

        Inside a compressed run the next gram is the run's, at cost 0; at a
        node's end it is drawn from the options - the cheapest at temperature
        0, a sample from ``softmax(-cost / temperature)`` otherwise.
        """
        tree = self.tree
        if offset < tree.held(node) - 1:
            return node, self.encoding.gram_at(tree.labels[node], offset + 1), 0.0
        options = tree.child_costs(node, costs)
        if not options:
            return node, None, 0.0
        if temperature == 0 or len(options) == 1:
            child, cost = min(options, key=lambda item: item[1])
        else:
            if rng is None:
                rng = tree.rng
            inv_t = 1.0 / temperature
            lowest = min(cst for _, cst in options)
            weights = [math.exp(-(cst - lowest) * inv_t) for _, cst in options]
            r = rng.random() * math.fsum(weights)
            child, cost = options[-1]
            acc = 0.0
            for item, wgt in zip(options, weights):
                acc += wgt
                if r < acc:
                    child, cost = item
                    break
        if tree.is_end(child):
            return child, None, cost
        return child, tree.first_gram(child), cost

    def _slide(
        self,
        prefix: str,
        length: int,
        window: int | None,
        temperature: float,
        rng: random.Random | None,
        to_end: bool,
        max_length: int | None,
        costs: str,
    ) -> PathResult:
        """Token by token: query the tree once, take the token it gives, feed it back, query again.

        Every step is one :meth:`context` of what has been said so far and one
        :meth:`next_token` from it.  With ``window=None`` the query is the
        whole history and the walk follows the deepest path the tree has, as
        the cheapest path does inside a run; with a window of ``k`` grams the
        tree is asked about the last ``k`` alone and decides afresh at every
        token, whatever it said before them.  Stops at END, at ``length``
        characters (unless ``to_end``), at ``max_length``, or when the tree
        knows nothing about the context.
        """
        tree = self.tree
        enc = self.encoding
        n = enc.n
        emitted = ""
        n_emitted = 0
        labels: list[str] = []
        node_ids: list[int] = []
        step_costs: list[float] = []
        queries = 0
        reached_end = False
        grams = enc.encode(prefix)
        if grams:
            loc = self.context(grams, window)
            cut = n - 1 if loc is not None else 0  # the next gram overlaps the located one; its new unit is emitted
            if loc is None:
                loc = (START, -1)  # nothing known: a new text begins, and the prefix is not its context
                grams = []
        else:
            node, offset, _include, cut = self._start(prefix)
            loc = (node, offset)
            grams = []
        if to_end and max_length is None:
            max_length = max(length, MAX_SLIDE)
        while True:
            if max_length is not None and n_emitted >= max_length:
                break
            if not to_end and n_emitted >= length:
                break
            node, offset = loc
            queries += 1
            child, gram, cost = self.next_token(node, offset, temperature, rng, costs)
            labels.append(tree.labels[child])
            node_ids.append(child)
            step_costs.append(cost)
            if gram is None:
                reached_end = tree.is_end(child)
                break
            piece = enc.piece_of(gram, cut) if cut else gram
            cut = n - 1
            units = len(enc.view(piece))
            if max_length is not None and n_emitted + units > max_length:
                piece = enc.truncate(piece, max_length - n_emitted)
                units = max_length - n_emitted
            emitted = enc.join_units(emitted, piece)
            n_emitted += units
            grams.append(gram)
            loc = self.context(grams, window)
            if loc is None:
                break
        return self._spelled(PathResult(
            text=emitted, labels=labels, node_ids=node_ids, cost=math.fsum(step_costs), step_costs=step_costs,
            expanded=queries, reached_end=reached_end, full_text=enc.join(prefix, emitted), legs=1,
        ))

    def predict(
        self,
        prefix: str,
        length: int = 20,
        mode: str = "dijkstra",
        step_penalty: float = 0.0,
        temperature: float = 1.0,
        to_end: bool = False,
        max_length: int | None = None,
        costs: str = "logprob",
        seed: int | None = None,
        max_legs: int = MAX_LEGS,
        window: int | None = None,
    ) -> PathResult:
        """Continue ``prefix``.

        ``mode="dijkstra"`` returns the cheapest continuation of at least
        ``length`` characters (or to an END leaf with ``to_end``);
        ``mode="sample"`` walks stochastically at ``temperature`` (0 is
        greedy), stopping after ``length`` characters or at END;
        ``mode="slide"`` goes token by token - one query of the tree per
        token, from the last ``window`` grams of everything said so far (all
        of it: ``None``), the token it gives fed back for the next query, the
        most likely token at temperature 0 and a sample above it
        (:meth:`_slide`).  ``max_length`` caps the text.  ``costs`` is ``"logprob"`` (the cyclic graph's
        non-negative cost) or ``"signal"`` (the signed edge signal, legal only
        because there are no cycles); ``step_penalty`` may be negative for the
        same reason.  ``seed`` makes a sampled walk reproducible on its own.

        On a tree of every suffix a walk is one leg: it ends at END or at
        ``length``.  With a bounded ``depth`` a path can run out where its
        window did; the walk then **re-enters** the tree at the deepest
        context of what it has said so far and goes on, at most ``max_legs``
        times.  The tree has no cycles; the re-entry is a loop *outside* it,
        and ``max_legs`` is its clock.
        """
        self._check_predict_args(prefix, length, max_length, mode, max_legs, window)
        check_costs(costs)
        tree = self.tree
        rng = random.Random(seed) if seed is not None else tree.rng
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if mode == "slide":
            return self._slide(prefix, length, window, temperature, rng, to_end, max_length, costs)
        enc = self.encoding
        node, offset, include_context, cut = self._start(prefix)
        emitted = ""
        n_emitted = 0
        labels: list[str] = []
        node_ids: list[int] = []
        step_costs: list[float] = []
        expanded = 0
        legs = 0
        reached_end = False
        while True:
            need = length - n_emitted
            if need < 0:
                need = 0
            cap = None if max_length is None else max(0, max_length - n_emitted)
            if mode == "dijkstra":
                leg = cheapest_path(
                    tree, node, offset, min_chars=need, max_chars=cap, step_penalty=step_penalty,
                    to_end=to_end, include_context=include_context, costs=costs,
                )
            else:
                limit = cap if to_end else (need if cap is None else min(need, cap))
                leg = sample_walk(
                    tree, node, offset, max_chars=limit, temperature=temperature, rng=rng,
                    stop_at_end=True, include_context=include_context, costs=costs,
                )
            text = enc.piece_of(leg.text, cut) if cut else leg.text
            cut = 0
            legs += 1
            emitted = enc.join_units(emitted, text)
            n_emitted += len(enc.view(text))
            labels.extend(leg.labels)
            node_ids.extend(leg.node_ids)
            step_costs.extend(leg.step_costs)
            expanded += leg.expanded
            if leg.reached_end:
                reached_end = True
                break
            if not to_end and n_emitted >= length:
                break
            if cap is not None and n_emitted >= max_length:
                break
            if not text or legs >= max_legs:
                break
            loc = tree.locate(enc.encode(enc.join(prefix, emitted)), self.min_count)
            if loc is None:
                break
            node, offset, include_context = loc[0], loc[1], False
        if max_length is not None:
            emitted = enc.truncate(emitted, max_length)
        return self._spelled(PathResult(
            text=emitted, labels=labels, node_ids=node_ids, cost=math.fsum(step_costs), step_costs=step_costs,
            expanded=expanded, reached_end=reached_end, full_text=enc.join(prefix, emitted), legs=legs,
        ))

    def _spelled(self, result: PathResult) -> PathResult:
        """A result with its words: the continuation and the whole text spelled back when the units are sounds."""
        enc = self.encoding
        result.full_spelled = enc.spell(result.full_text)
        result.spelled = enc.spell_tail(result.full_text, result.text)
        return result

    def generate(
        self,
        max_length: int = 40,
        count: int = 1,
        mode: str = "sample",
        temperature: float = 1.0,
        seed: int | None = None,
        step_penalty: float = 0.0,
        to_end: bool = False,
        costs: str = "logprob",
        window: int | None = None,
    ) -> list[PathResult]:
        """``count`` texts from START: sampled walks by default, the one cheapest path with ``mode="dijkstra"``,
        or token-by-token walks with ``mode="slide"`` (one text at temperature 0, which is deterministic)."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if mode == "dijkstra" or (mode == "slide" and temperature == 0):
            count = min(count, 1)
        out: list[PathResult] = []
        for i in range(count):
            out.append(self.predict(
                "", length=max_length, mode=mode, step_penalty=step_penalty, temperature=temperature,
                to_end=to_end, max_length=max_length, costs=costs, seed=None if seed is None else seed + i,
                window=window,
            ))
        return out

    def next_token_accuracy(self, texts: Iterable[str] | str, window: int | None = None) -> dict:
        """How often the token-by-token walk's most likely token is the text's next one, over texts.

        Every position of every text is one query (:meth:`context` of the
        text so far, over the last ``window`` grams) and one decision at
        temperature 0; END counts as a token.  ``known`` is how often the
        actual token was among the context's options at all - the tail the
        deepest window may not hold even when its top choice is right.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        enc = self.encoding
        tree = self.tree
        correct = known = total = 0
        for text in items:
            grams = enc.encode(text)
            if not grams:
                continue
            history: list[str] = []
            for actual in grams + [None]:
                loc = self.context(history, window) if history else (START, -1)
                if loc is not None and tree.count[loc[0]] < self.min_count:
                    loc = None
                total += 1
                if loc is not None:
                    node, offset = loc
                    _child, gram, _cost = self.next_token(node, offset)
                    correct += gram == actual
                    if offset < tree.held(node) - 1:
                        known += enc.gram_at(tree.labels[node], offset + 1) == actual
                    elif actual is None:
                        known += tree.end_leaf[node] >= 0
                    else:
                        known += actual in tree.children[node]
                if actual is not None:
                    history.append(actual)
        return {
            "window": window, "tokens": total,
            "accuracy": correct / total if total else 0.0, "known": known / total if total else 0.0,
        }

    # -- scoring -------------------------------------------------------------

    def _relocate(self, history: list[str], ending: bool = False) -> tuple[int, int]:
        """The deepest usable context of ``history``, or the root (the empty context)."""
        if history:
            loc = self.tree.locate(history, self.min_count, ending)
            if loc is not None:
                return loc[0], loc[1]
        return ROOT, -1

    def score(self, text: str) -> dict:
        """Log-probability of ``text`` under the model.

        The text is walked from START through the deepest context it has at
        every step: along the tree while the context continues, and from the
        longest known suffix of what was read so far whenever it does not - a
        window that ran out, a context seen fewer than ``min_count`` times, or
        a gram the context was never followed by.  Every edge taken contributes
        ``log softmax`` of its score; a step inside a compressed node is
        deterministic (cost 0) and not a transition; a gram the deepest context
        was never followed by contributes ``log(UNKNOWN_PROB)`` and is counted
        in ``unknown_transitions`` - the cyclic model's convention, so the two
        read on one scale.  The context never peeks at the answer: a shorter
        context that would have known the gram does not rescue a deeper one
        that did not.  Under a bound, a context at the window's limit is asked
        about END alone (:meth:`RadixTree.locate`), so a text the tree was
        trained on never misses, at any depth.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tree = self.tree
        enc = self.encoding
        grams = enc.encode(text)
        chars = enc.length(text)
        units = enc.units_name
        if not grams:
            return {"log_prob": 0.0, "per_char": 0.0, "chars": chars, "transitions": 0, "unknown_transitions": 0, "units": units}
        labels = tree.labels
        children = tree.children
        count = tree.count
        held = tree.held
        view = enc.view
        last = enc.last_unit
        ov = enc.overlap
        min_count = self.min_count
        end_leaf = tree.end_leaf
        log_prob = 0.0
        transitions = 0
        unknown = 0
        node, offset = START, -1
        history: list[str] = []

        depth = tree.depth
        symbols_of = tree.symbols_of

        def settle(node: int, offset: int, ending: bool = False) -> tuple[int, int]:
            """The context the next symbol is read from: the current one while it is trusted and has something
            to offer - and, for a gram, room in its window to have seen one - else the deepest usable suffix of
            what was read so far."""
            if (
                count[node] < min_count
                or (offset == held(node) - 1 and not (children[node] or end_leaf[node] >= 0))
                or (not ending and depth is not None and symbols_of(node, offset) >= depth)
            ):
                return self._relocate(history, ending)
            return node, offset

        for g in grams:
            node, offset = settle(node, offset)
            ok = False
            if offset < held(node) - 1:
                if view(labels[node])[offset + 1 + ov] == last(g):
                    offset += 1
                    ok = True
            else:
                c = children[node].get(g)
                if c is not None:
                    lp = tree.log_prob(node, c)
                    if lp is not None:
                        log_prob += lp
                        transitions += 1
                        node, offset = c, 0
                        ok = True
            history.append(g)
            if not ok:
                transitions += 1
                unknown += 1
                log_prob += _LOG_UNKNOWN
                node, offset = self._relocate(history)
        node, offset = settle(node, offset, ending=True)
        transitions += 1
        lp = tree.log_prob(node, end_leaf[node]) if offset == held(node) - 1 and end_leaf[node] >= 0 else None
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

    def bits_per_char(self, texts: Iterable[str] | str) -> dict:
        """``-log2 P`` per character over texts, with the miss rate: ``{bits_per_char, chars, transitions, unknown_transitions, miss_rate}``."""
        items = [texts] if isinstance(texts, str) else list(texts)
        total = 0.0
        chars = transitions = unknown = 0
        for t in items:
            s = self.score(t)
            total -= s["log_prob"]
            chars += s["chars"]
            transitions += s["transitions"]
            unknown += s["unknown_transitions"]
        return {
            "bits_per_char": total / math.log(2.0) / chars if chars else 0.0,
            "chars": chars,
            "transitions": transitions,
            "unknown_transitions": unknown,
            "miss_rate": unknown / transitions if transitions else 0.0,
            "units": self.encoding.units_name,
        }

    # -- inversion and 2NRL --------------------------------------------------

    def invert(self) -> None:
        """Flip every in-edge weight and every activation (``tree.invert()``)."""
        self.tree.invert()

    def _paths_of(self, texts: list[str]) -> list[list[int]]:
        """Node paths (sentinels included) of texts, registering a text structurally when it is not a root path yet."""
        tree = self.tree
        encode = self.encoding.encode
        paths: list[list[int]] = []
        for text in texts:
            grams = encode(text)
            if not grams:
                continue
            path = tree.node_path(grams)
            if path is None:
                tree.observe_sequence(grams, count=False)
                path = tree.node_path(grams)
            if path:
                paths.append(path)
        return paths

    def invert_paths(self, texts: Iterable[str] | str, mode: str = "activation", amounts=None) -> dict:
        """Failures: move the activation (``a``, ``k``) or the state ``z`` of the nodes on their paths toward their
        negation - fully (amount 1: a sign flip) or partly (``amounts``: one per text, the worse the larger).

        The local counterpart of :meth:`invert`: only nodes the failed texts run
        through change.  An edge's score ``w * f_p * f_c`` changes sign when
        exactly one of its endpoints flips, so every *other* node of a path is
        flipped, the path's own END leaf included; a node shared by several
        texts takes the largest amount.  On a tree every path is simple, the
        tree is bipartite and the END leaf belongs to this context alone, so
        for one text the parity flips **every** edge of its path - the
        operation the cyclic graph can only do best-effort, exact here.
        Returns ``{"texts", "flipped", "unit": "nodes", "mode", "amount_mean"}``.
        """
        texts, _ = self._clean_texts(texts)
        if amounts is None:
            values = [1.0] * len(texts)
        else:
            values = [float(v) for v in amounts]
            if len(values) != len(texts):
                raise ValueError(f"amounts has {len(values)} entries for {len(texts)} texts")
            for v in values:
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"amounts must be finite and >= 0, got {v}")
        tree = self.tree
        chosen: dict[int, float] = {}
        for path, amount in zip(self._paths_of(texts), values):
            real = [i for i in path if i != START]  # the real nodes and the path's own END leaf
            if not real or amount <= 0:
                continue
            best: tuple[int, set[int]] | None = None
            for parity in (0, 1):
                candidate = {i for j, i in enumerate(real) if j % 2 == parity}
                state = set(chosen) | candidate
                gain = sum(1 for x, y in zip(path, path[1:]) if (x in state) != (y in state))
                if best is None or gain > best[0]:
                    best = (gain, candidate)
            for i in best[1]:
                chosen[i] = max(chosen.get(i, 0.0), amount)
        flipped = tree.flip_nodes(chosen, mode)
        self.meta["path_inversions"] += flipped
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
        **overrides,
    ) -> dict:
        """2NRL: train on ``bad``, invert, fine-tune on ``good``.

        The negative phase uses ``neg_lr``; the positive phase ``pos_lr`` for
        weights and states and ``pos_lr / 10`` for the activation parameters.
        Other :class:`TrainConfig` fields can be given as keyword ``overrides``
        and apply to both phases.  Records carry ``"phase"``.
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs/neg_lr/pos_lr")
        negative = self.train(bad, epochs=neg_epochs, lr=neg_lr, progress=progress, phase="negative", **overrides)
        self.invert()
        positive = self.train(
            good, epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, progress=progress, phase="positive", **overrides
        )
        self.meta["twonrl_runs"] += 1
        return {"negative": negative, "positive": positive, "inverted": self.tree.inverted}

    # -- structure, stats, persistence ---------------------------------------

    def compress(self) -> int:
        """Merge all unary chains; returns the number of merges (0 on a tree of every suffix)."""
        return self.tree.compress()

    def stats(self) -> dict:
        """Size, shape, state and lifetime counters (JSON-serialisable)."""
        t = self.tree
        return {
            "kind": self.kind,
            "nodes": t.num_nodes(),
            "ends": t.num_ends(),
            "edges": t.num_edges(),
            "grams": t.num_grams(),
            "label_chars": t.label_chars(),
            "branches": t.branches(),
            "max_depth": t.max_depth(),
            "compression_ratio": t.compression_ratio(),
            "encoding": str(self.encoding),
            "unit": self.encoding.unit,
            "units": self.encoding.units_name,
            "ngram": self.encoding.n,
            "depth": self.depth,
            "min_count": self.min_count,
            "inverted": t.inverted,
            "epochs_total": self.meta["epochs_total"],
            "trained_chars": self.meta["trained_chars"],
            "trained_texts": self.meta["trained_texts"],
            "twonrl_runs": self.meta["twonrl_runs"],
            "path_inversions": self.meta["path_inversions"],
            "history_len": len(self.history),
            "last_loss": self.history[-1]["loss"] if self.history else None,
        }

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (tree, history, meta)."""
        return {
            "format": MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "kind": self.kind,
            "saved_at": _utc_now(),
            "min_count": self.min_count,
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "tree": self.tree.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RadixTreeNet":
        """Inverse of :meth:`to_dict`."""
        if not isinstance(d, dict) or d.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a {MODEL_FORMAT} model document")
        if int(d.get("version", 1)) > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {MODEL_FORMAT} model version {d.get('version')}")
        tree = RadixTree.from_dict(d["tree"])
        model = cls.__new__(cls)
        model.seed = tree.seed
        model.tree = tree
        model.min_count = int(d.get("min_count", 1))
        model._acc = None
        model.history = [dict(r) for r in d.get("history") or []]
        meta = dict(d.get("meta") or {})
        for key, blank in (("epochs_total", 0), ("trained_chars", 0), ("trained_texts", 0), ("twonrl_runs", 0), ("path_inversions", 0)):
            meta.setdefault(key, blank)
        meta.setdefault("seed", tree.seed)
        meta.setdefault("created", _utc_now())
        model.meta = meta
        return model

    def save(self, path: str) -> None:
        """Write the model as JSON (gzip when ``path`` ends with ``.gz``), atomically."""
        payload = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        write_bytes_atomic(path, payload, use_gzip=path.endswith(".gz"))

    @classmethod
    def load(cls, path: str) -> "RadixTreeNet":
        """Load a model written by :meth:`save`."""
        return cls.from_dict(read_json_file(path))

    def __repr__(self) -> str:
        t = self.tree
        return (
            f"RadixTreeNet(seed={self.seed}, depth={self.depth}, nodes={t.num_nodes()}, ends={t.num_ends()}, "
            f"grams={t.num_grams()}, inverted={t.inverted})"
        )


def load_model(path: str) -> RadixTreeNet:
    """Load a model file written by :meth:`RadixTreeNet.save`."""
    return RadixTreeNet.load(path)
