"""NegativeNet - the negative network: the failures, and *why* they were failures.

A copy of the network that keeps only its **negative portions**.  It is the
same self-compressing cyclic graph (same trigram window, same splits and
merges, same Dijkstra / beam searches), but nothing in it comes from correct
data: **every node and every edge exists because something went wrong there**,
and every edge remembers *why* - the reasons the tutor gave, with how much
blame each one carries.

The tutor supplies the negatives
-------------------------------
The negative network never invents failures.  They arrive from the tutor -
the Ollama reviewer that rates and critiques the network's own output, the
code-generation teacher / judge with its sandbox errors and verdicts, the
evolve loop's discriminator, a thumbs-down in the frontend - and every one
of them carries the tutor's reason (:mod:`radixnet.tutor` turns a critique
into a reason tag and a severity).  Training on anything else is a category
error: :meth:`NegativeNet.train` *is* a blame pass.

What an edge remembers
----------------------
Every edge ``p -> c`` keeps

* ``blame``  - the summed severity of the failures that ran through it,
* ``fails``  - how many failing texts ran through it,
* ``clear``  - how much *cleared* (tutor-passed) text ran through it, and
* ``reasons`` - ``{reason: blame}``, the tutor's reasons split by how much
  blame each contributed (at most :data:`MAX_EDGE_REASONS` of them).

The *net evidence* against an edge is ``max(0, blame - clear)``: blame and
clearing cancel, so an edge the tutor's passes cross as often as its failures
do carries no verdict at all - which is how common fragments stay out of the
judgement.

The weight function (the negative distribution)
-----------------------------------------------
There is no learning rate and no gradient.  Like the count / reward model
every node's activation is the constant 1, so an edge's score *is* its
weight, and the weight is the edge's share of the failure mass leaving its
parent::

    net    = max(0, blame - clear_scale * clear)
    R_bad  = (net + s) / (net leaving the parent + s * children)
    weight = share_scale * log(R_bad) + blame_scale * log(1 + net)

with ``s = 0.5``.  A softmax over the children therefore reads
``P(child | parent) ∝ R_bad * (1 + net) ** blame_scale``: the network models
**how text goes wrong**.  Its beam
prediction returns the most likely ways to fail from a prefix (and, as the
bottom-K, the least likely ones), and :meth:`NegativeNet.judge` walks a text
through it and reports how much of it is built out of known failure, which
reasons those failures carried and which fragments are to blame.

That judgement is the filter: :mod:`radixnet.duo` pairs this network with
the positive one so that generated text is scored by the generator and vetoed
by the critic - the GAN, at output time.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_B, DEFAULT_H
from .backend import get_backend
from .beam import Prediction
from .encoding import WINDOW, Decoder, Encoder
from .graph import END, START, RadixCyclicGraph
from .model import (
    MODEL_FORMAT_VERSION,
    GraphModel,
    ProgressFn,
    TrainConfig,
    _resolve_config,
    _utc_now,
)

__all__ = [
    "MAX_EDGE_REASONS",
    "MAX_LOG_ENTRIES",
    "NEGATIVE_MODEL_FORMAT",
    "UNSPECIFIED_REASON",
    "NegativeGraph",
    "NegativeNet",
]

NEGATIVE_MODEL_FORMAT = "radixnet-negative"
UNSPECIFIED_REASON = "unspecified"
"""Reason recorded when a failure arrives without one (the tutor always gives one)."""

MAX_EDGE_REASONS = 8
"""Reasons kept per edge; the smallest is dropped when a ninth appears (the totals keep it)."""

MAX_LOG_ENTRIES = 200
"""Entries kept in the journal of failures (the newest win)."""

_W = WINDOW
_MAX_LOG_PPL = 700.0
_LOG_TEXT_CHARS = 160


def _clean_reason(reason: str | None) -> str:
    """A reason tag: trimmed, lower-cased, collapsed to one line, never empty."""
    text = " ".join(str(reason or "").split()).strip().lower()
    return text[:60] or UNSPECIFIED_REASON


class NegativeGraph(RadixCyclicGraph):
    """A :class:`RadixCyclicGraph` whose edge weights come from blame (see the module docstring).

    Node activations are the constant 1 (``a = 0, k = 1``), so the base
    class's scores, probabilities, costs, Dijkstra / sampling walks, splits
    and merges all work unchanged and an edge's score is its weight.  The
    side arrays (``edge_blame``, ``edge_fails``, ``edge_clear``,
    ``edge_reasons``) are indexed by edge id like ``edge_w``, so they survive
    splits and merges exactly as the counts do.
    """

    SMOOTHING = 0.5

    def __init__(
        self,
        seed: int = 0,
        share_scale: float = 1.0,
        blame_scale: float = 0.0,
        clear_scale: float = 1.0,
    ) -> None:
        self.share_scale = float(share_scale)
        self.blame_scale = float(blame_scale)
        self.clear_scale = float(clear_scale)
        self.edge_blame: list[float] = []
        self.edge_fails: list[int] = []
        self.edge_clear: list[float] = []
        self.edge_reasons: list[dict[int, float]] = []
        self.reasons: list[str] = []
        self.reason_ids: dict[str, int] = {}
        self.reason_blame: list[float] = []
        self.reason_fails: list[int] = []
        self.total_blame = 0.0
        self.total_fails = 0
        self.total_clear = 0.0
        super().__init__(seed)

    # -- reasons -------------------------------------------------------------

    def reason_id(self, reason: str | None) -> int:
        """Id of a reason tag, registering it on first sight."""
        label = _clean_reason(reason)
        rid = self.reason_ids.get(label)
        if rid is None:
            rid = len(self.reasons)
            self.reason_ids[label] = rid
            self.reasons.append(label)
            self.reason_blame.append(0.0)
            self.reason_fails.append(0)
        return rid

    def reason_label(self, rid: int) -> str:
        """The tag of a reason id (``"unspecified"`` for an unknown id)."""
        return self.reasons[rid] if 0 <= rid < len(self.reasons) else UNSPECIFIED_REASON

    def reason_table(self) -> list[dict]:
        """``[{"reason", "blame", "fails", "edges", "share"}]``, heaviest blame first."""
        edges = [0] * len(self.reasons)
        for e, ok in enumerate(self.edge_alive):
            if ok:
                for rid in self.edge_reasons[e]:
                    if 0 <= rid < len(edges):
                        edges[rid] += 1
        total = sum(self.reason_blame) or 1.0
        table = [
            {
                "reason": label,
                "blame": self.reason_blame[rid],
                "fails": self.reason_fails[rid],
                "edges": edges[rid],
                "share": self.reason_blame[rid] / total,
            }
            for rid, label in enumerate(self.reasons)
        ]
        table.sort(key=lambda row: (-row["blame"], row["reason"]))
        return table

    def edge_reason_table(self, e: int) -> list[dict]:
        """``[{"reason", "blame"}]`` of one edge, heaviest first."""
        if not (0 <= e < len(self.edge_reasons)):
            return []
        rows = [{"reason": self.reason_label(rid), "blame": blame} for rid, blame in self.edge_reasons[e].items()]
        rows.sort(key=lambda row: (-row["blame"], row["reason"]))
        return rows

    # -- the tracked numbers -------------------------------------------------

    def record_failure(self, edge_ids: Iterable[int], severity: float, reason: str | None) -> int:
        """Blame every listed alive edge by ``severity`` for ``reason``; returns how many were blamed."""
        amount = abs(float(severity))
        if not math.isfinite(amount):
            raise ValueError(f"severity must be finite, got {severity}")
        rid = self.reason_id(reason)
        blame, fails, reasons, alive = self.edge_blame, self.edge_fails, self.edge_reasons, self.edge_alive
        touched = 0
        for e in edge_ids:
            if not (0 <= e < len(blame)) or not alive[e]:
                continue
            blame[e] += amount
            fails[e] += 1
            if amount:
                by_reason = reasons[e]
                by_reason[rid] = by_reason.get(rid, 0.0) + amount
                if len(by_reason) > MAX_EDGE_REASONS:
                    # the weakest of the *others* goes: a reason just added carries the least blame of
                    # all by construction and would otherwise evict itself
                    weakest = min((r for r in by_reason if r != rid), key=lambda r: by_reason[r])
                    del by_reason[weakest]
            touched += 1
        if touched:
            self.total_blame += amount * touched
            self.total_fails += touched
            self.reason_blame[rid] += amount * touched
            self.reason_fails[rid] += 1
            self.recompute_weights()
        return touched

    def record_clear(self, edge_ids: Iterable[int], weight: float = 1.0) -> int:
        """Credit every listed alive edge with ``weight`` of *cleared* text (the tutor passed it); returns the count."""
        amount = abs(float(weight))
        if not math.isfinite(amount):
            raise ValueError(f"weight must be finite, got {weight}")
        clear, alive = self.edge_clear, self.edge_alive
        touched = 0
        for e in edge_ids:
            if not (0 <= e < len(clear)) or not alive[e]:
                continue
            clear[e] += amount
            touched += 1
        if touched:
            self.total_clear += amount * touched
            self.recompute_weights()
        return touched

    def evidence(self, e: int) -> float:
        """Net evidence against an edge: ``max(0, blame - clear_scale * clear)`` (0 for an unknown or dead edge).

        Blame and clearing cancel: an edge the tutor's *passes* cross as often
        as its failures do is not evidence of anything, however busy it is -
        which is what keeps a common fragment out of the verdict.
        """
        if not (0 <= e < len(self.edge_blame)) or not self.edge_alive[e]:
            return 0.0
        return max(0.0, self.edge_blame[e] - self.clear_scale * self.edge_clear[e])

    def forget(self, reason: str | None = None, factor: float = 0.0) -> dict:
        """Scale blame down (``factor`` 0 = forget it entirely, 0.5 = halve it), for one reason or all of them.

        Use it when the tutor was wrong, or to let old failures fade.
        Returns ``{"reason", "edges", "blame_removed"}``.
        """
        scale = float(factor)
        if not (0.0 <= scale < 1.0):
            raise ValueError(f"factor must lie in [0, 1), got {factor}")
        rid = self.reason_ids.get(_clean_reason(reason)) if reason is not None else None
        if reason is not None and rid is None:
            return {"reason": _clean_reason(reason), "edges": 0, "blame_removed": 0.0}
        removed = 0.0
        touched = 0
        for e, ok in enumerate(self.edge_alive):
            if not ok:
                continue
            by_reason = self.edge_reasons[e]
            if rid is None:
                drop = self.edge_blame[e] * (1.0 - scale)
                if drop:
                    self.edge_blame[e] -= drop
                    self.edge_reasons[e] = {r: v * scale for r, v in by_reason.items()} if scale else {}
                    removed += drop
                    touched += 1
            elif rid in by_reason:
                drop = by_reason[rid] * (1.0 - scale)
                self.edge_blame[e] = max(0.0, self.edge_blame[e] - drop)
                if scale:
                    by_reason[rid] *= scale
                else:
                    del by_reason[rid]
                removed += drop
                touched += 1
        if rid is None:
            self.reason_blame = [v * scale for v in self.reason_blame]
            self.total_blame *= scale
        else:
            self.total_blame = max(0.0, self.total_blame - self.reason_blame[rid] * (1.0 - scale))
            self.reason_blame[rid] *= scale
        if touched:
            self.recompute_weights()
        return {
            "reason": _clean_reason(reason) if reason is not None else "*",
            "edges": touched,
            "blame_removed": removed,
        }

    # -- the weight function -------------------------------------------------

    def edge_weight(self, evidence: float, parent_evidence: float, degree: int) -> float:
        """The blame weight of one edge from its net evidence (see the class docstring)."""
        s = self.SMOOTHING
        evidence = max(0.0, float(evidence))
        degree = max(1, int(degree))
        r_bad = (evidence + s) / (max(0.0, float(parent_evidence)) + s * degree)
        return self.share_scale * math.log(r_bad) + self.blame_scale * math.log1p(evidence)

    def recompute_weights(self) -> None:
        """Write the blame weight to every alive edge (after blame, clearing or scales changed)."""
        ew = self.edge_w
        weight = self.edge_weight
        evidence = self.evidence
        for p, ch in enumerate(self.children):
            if not ch or not self.alive[p]:
                continue
            edges = list(ch.values())
            degree = len(edges)
            net = {e: evidence(e) for e in edges}
            total = sum(net.values())
            for e in edges:
                ew[e] = weight(net[e], total, degree)
        self.version += 1

    def configure(self, **options: float) -> dict:
        """Change ``share_scale`` / ``blame_scale`` / ``clear_scale`` and recompute every weight."""
        for name, value in options.items():
            if name not in ("share_scale", "blame_scale", "clear_scale"):
                raise ValueError(f"unknown weight option {name!r}")
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number, got {value}")
            setattr(self, name, number)
        self.recompute_weights()
        return self.weight_config()

    def weight_config(self) -> dict:
        return {
            "function": "blame",
            "share_scale": self.share_scale,
            "blame_scale": self.blame_scale,
            "clear_scale": self.clear_scale,
            "smoothing": self.SMOOTHING,
        }

    # -- construction overrides ----------------------------------------------

    def _new_node(self, label, z=None, a=None, b=DEFAULT_B, h=DEFAULT_H, k=0.0, count=0) -> int:
        # a = 0 and k = 1 make f(z) = 1 whatever z is: scores reduce to the edge weight
        return super()._new_node(label, z=0.0 if z is None else z, a=0.0, b=b, h=h, k=1.0, count=count)

    def _new_edge(self, p: int, c: int, count: int = 0) -> int:
        e = super()._new_edge(p, c, count)
        self.edge_blame.append(0.0)
        self.edge_fails.append(0)
        self.edge_clear.append(0.0)
        self.edge_reasons.append({})
        self.edge_w[e] = 0.0  # recompute_weights() gives it its real value once the pass is over
        return e

    def merge_child(self, p: int) -> bool:
        """Merge a unary chain unless its edge carries evidence: what went wrong stays an edge.

        Compression turns a transition into a deterministic step *inside* a
        node, and a step inside a node has no edge to carry blame - a
        correction that blamed a single transition would be compressed away
        and forgotten.  Everything the tutor never ruled on still merges, so
        the structure stays a radix tree everywhere it can.
        """
        children = self.children[p] if 0 <= p < len(self.children) else {}
        if len(children) == 1:
            edge = next(iter(children.values()))
            if self.edge_blame[edge] > 0 or self.edge_clear[edge] > 0:
                return False
        return super().merge_child(p)

    def invert(self) -> None:
        """Swap blame and clearing: what the tutor rejected becomes what it accepted, and back.

        The inverse of a network of failures is a network of the text that
        passed - the local counterpart of :meth:`RadixCyclicGraph.invert` for
        a model whose weights are evidence rather than learned numbers.
        """
        self.edge_blame, self.edge_clear = [float(v) for v in self.edge_clear], [float(v) for v in self.edge_blame]
        self.total_blame, self.total_clear = self.total_clear, self.total_blame
        self.inverted = not self.inverted
        self.recompute_weights()

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        blame: list[float] = []
        fails: list[int] = []
        clear: list[float] = []
        reasons: list[list[list[float]]] = []
        for old, ok in enumerate(self.alive):
            if not ok:
                continue
            for _c, e in self.children[old].items():
                blame.append(self.edge_blame[e])
                fails.append(self.edge_fails[e])
                clear.append(self.edge_clear[e])
                reasons.append([[rid, value] for rid, value in sorted(self.edge_reasons[e].items())])
        d["edges"].update(blame=blame, fails=fails, clear=clear, reasons=reasons)
        d["weights"] = {
            **self.weight_config(),
            "kind": "negative",
            "total_blame": self.total_blame,
            "total_fails": self.total_fails,
            "total_clear": self.total_clear,
            "reasons": {
                "labels": list(self.reasons),
                "blame": list(self.reason_blame),
                "fails": list(self.reason_fails),
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeGraph":
        g = super().from_dict(d)
        weights = d.get("weights") or {}
        g.share_scale = float(weights.get("share_scale", 1.0))
        g.blame_scale = float(weights.get("blame_scale", 0.0))
        g.clear_scale = float(weights.get("clear_scale", 1.0))
        g.total_blame = float(weights.get("total_blame", 0.0))
        g.total_fails = int(weights.get("total_fails", 0))
        g.total_clear = float(weights.get("total_clear", 0.0))
        registry = weights.get("reasons") or {}
        g.reasons = [_clean_reason(label) for label in registry.get("labels", [])]
        g.reason_ids = {label: rid for rid, label in enumerate(g.reasons)}
        g.reason_blame = [float(v) for v in registry.get("blame", [])]
        g.reason_fails = [int(v) for v in registry.get("fails", [])]
        while len(g.reason_blame) < len(g.reasons):
            g.reason_blame.append(0.0)
        while len(g.reason_fails) < len(g.reasons):
            g.reason_fails.append(0)
        edges = d.get("edges", {})
        n = len(g.edge_w)
        g.edge_blame = _float_array(edges.get("blame"), n, "blame")
        g.edge_clear = _float_array(edges.get("clear"), n, "clear")
        g.edge_fails = [int(v) for v in _float_array(edges.get("fails"), n, "fails")]
        stored = edges.get("reasons")
        if stored is None:
            g.edge_reasons = [{} for _ in range(n)]
        else:
            if len(stored) != n:
                raise ValueError("edge reason array has an inconsistent length")
            g.edge_reasons = [
                {int(rid): float(value) for rid, value in entry if 0 <= int(rid) < len(g.reasons)} for entry in stored
            ]
        g.a = [0.0] * len(g.labels)
        g.k = [1.0] * len(g.labels)
        g.recompute_weights()
        return g


def _float_array(values, n: int, name: str) -> list[float]:
    """A per-edge float array from a document (zeros when absent); length must match."""
    if values is None:
        return [0.0] * n
    if len(values) != n:
        raise ValueError(f"edge {name} array has an inconsistent length")
    return [float(v) for v in values]


class NegativeNet(GraphModel):
    """The negative network: trained on failures only, and it remembers why (see the module docstring).

    * :meth:`blame` (and therefore :meth:`train` / :meth:`punish`) is the only
      thing that builds structure - a node or edge exists here because a
      failure ran through it.
    * :meth:`clear` (and therefore :meth:`reward`) walks text the tutor
      *passed* through the structure that already exists and credits those
      edges, which damps their evidence; it never creates anything, so the
      network stays a network of failures.
    * :meth:`correct` is the sharpest lesson of all: when the tutor writes the
      sentence out correctly, only the characters it changed are blamed and
      the rest of the correction clears blame.
    * :meth:`judge` is the filter: how much of a text is built out of known
      failure, which reasons those failures carried, and which fragments are
      to blame.
    * :meth:`predict` continues a prefix the way failures continue it: the
      top-K most likely ways to go wrong, the bottom-K least likely.

    There is no learning rate; the magnitude of a lesson is its ``severity``
    (1 = one ordinary failure).
    """

    kind = "negative"
    format = NEGATIVE_MODEL_FORMAT
    label = "Negative / blame"
    description = (
        "the failures only: every edge keeps the blame it collected, how often it failed and the tutor's reasons; "
        "it judges text instead of writing it and filters the positive model's output"
    )

    def __init__(
        self,
        seed: int = 0,
        backend: str = "auto",
        device: str | None = None,
        share_scale: float = 1.0,
        blame_scale: float = 0.0,
        clear_scale: float = 1.0,
        threshold: float = 1.0,
        min_coverage: float = 0.5,
    ) -> None:
        self.seed = int(seed)
        self.graph = NegativeGraph(
            seed=self.seed, share_scale=share_scale, blame_scale=blame_scale, clear_scale=clear_scale
        )
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
        # no numeric learning rule runs, so the backend is only reported (python / cpu); backend / device are
        # accepted for interface parity with RadixNet
        self.backend = get_backend("python", None)
        self.history: list[dict] = []
        self.log: list[dict] = []
        self.threshold = float(threshold)
        """Default risk (blame per transition) at which :meth:`judge` rejects a text."""
        self.min_coverage = float(min_coverage)
        """Default share of a text's transitions that must be known failures before it can be rejected."""
        self.meta: dict = self._new_meta(self.seed)

    @staticmethod
    def _new_meta(seed: int) -> dict:
        meta = GraphModel._new_meta(seed)
        meta.update(
            failures_total=0, blame_total=0.0, cleared_total=0, judgements=0, rejected=0, sources={},
        )
        return meta

    # -- walking texts -------------------------------------------------------

    def _mean_cost(self, transitions: list[tuple[int, int]]) -> float:
        """Mean ``-log P`` of the transitions under the current weights."""
        if not transitions:
            return 0.0
        child_costs = self.graph.child_costs
        cache: dict[int, dict[int, float]] = {}
        total = 0.0
        for p, e in transitions:
            costs = cache.get(p)
            if costs is None:
                costs = cache[p] = {edge: cost for _c, edge, cost in child_costs(p)}
            total += costs.get(e, 0.0)
        return total / len(transitions)

    def _register(self, grams: list[list[str]]) -> list[list[tuple[int, int]]]:
        """Register every encoded failure structurally and return its ``(parent, edge)`` transitions.

        Two passes: a text registered later can split a node an earlier one
        pointed at, so the transitions are re-derived once the structure has
        settled (the second pass never splits again).
        """
        graph = self.graph
        for g in grams:
            if g:
                graph.observe_sequence(g, False)
        return [graph.observe_sequence(g, False) if g else [] for g in grams]

    def _shared_edges(self, text: str) -> list[tuple[int, int]]:
        """The ``(parent, edge)`` transitions ``text`` shares with the failure structure (nothing is created).

        Cleared text is ordinary, correct text: it will not walk a graph of
        failures end to end, so every crossing it *does* share counts and the
        rest is simply skipped.
        """
        return [(c["parent"], c["edge"]) for c in self.crossings(text) if c["edge"] is not None]

    def _pass(
        self,
        texts: Iterable[str] | str,
        cfg: TrainConfig,
        *,
        blame: bool,
        amount: float,
        reason: str,
        source: str,
        note: str,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager=None,
    ) -> list[dict]:
        """``cfg.epochs`` passes that blame (``blame``) or clear (``not blame``) every path of ``texts``."""
        texts, skipped_short = self._clean_texts(texts)
        graph = self.graph
        meta = self.meta
        grams = [self.encoder.encode(t) for t in texts]
        records: list[dict] = []
        if blame:
            meta["trained_texts"] += len(texts)
            meta["trained_chars"] += sum(len(t) for t in texts)
            if source:
                meta["sources"][source] = meta["sources"].get(source, 0) + len(texts)
            for text in texts:
                self._note(text, reason=reason, severity=amount, source=source, note=note)
        if blame:
            # settle the structure once so every epoch walks the same transitions
            self._register(grams)
        pending_merges = graph.compress() if (blame and cfg.auto_compress) else 0
        for _ in range(max(0, cfg.epochs)):
            t0 = time.perf_counter()
            per_text = self._register(grams) if blame else [self._shared_edges(t) for t in texts]
            touched = 0
            for transitions in per_text:
                if not transitions:
                    continue
                edges = [e for _p, e in transitions]
                if blame:
                    touched += graph.record_failure(edges, amount, reason)
                else:
                    touched += graph.record_clear(edges, amount)
            flat = [t for transitions in per_text for t in transitions]
            matched = sum(1 for transitions in per_text if transitions)
            if blame:
                meta["failures_total"] += matched
                meta["blame_total"] += amount * touched
            else:
                meta["cleared_total"] += matched
            loss = self._mean_cost(flat)
            merges = (graph.compress() if (blame and cfg.auto_compress) else 0) + pending_merges
            pending_merges = 0
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
                "transitions": len(flat),
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "phase": "negative" if blame else "clear",
                "texts": len(texts),
                "matched": matched,
                "unmatched": len(texts) - matched,
                "edges_touched": touched,
                "reason": reason if blame else None,
                "severity": amount if blame else None,
                "source": source or None,
            }
            self.history.append(record)
            records.append(record)
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self, epoch, "epoch", record)
            if stop_event is not None and stop_event.is_set():
                break
        return records

    def _note(self, text: str, *, reason: str, severity: float, source: str, note: str) -> None:
        """Append one failure to the journal (oldest entries drop out)."""
        self.log.append({
            "at": _utc_now(),
            "text": text[:_LOG_TEXT_CHARS],
            "reason": reason,
            "severity": severity,
            "source": source or "",
            "note": " ".join(str(note or "").split())[:300],
        })
        if len(self.log) > MAX_LOG_ENTRIES:
            del self.log[: len(self.log) - MAX_LOG_ENTRIES]

    # -- learning (negative data only) ---------------------------------------

    def blame(
        self,
        texts: Iterable[str] | str,
        *,
        reason: str | None = None,
        severity: float = 1.0,
        source: str = "",
        note: str = "",
        epochs: int = 1,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager=None,
        **overrides,
    ) -> list[dict]:
        """Learn a failure: register every text's path and blame its edges by ``severity`` for ``reason``.

        ``reason`` is the tutor's verdict (``"gibberish"``, ``"wrong-output"``,
        ``"thumbs-down"`` ...), ``note`` its own words, ``source`` who said so
        (``"review"``, ``"codegen"``, ``"evolve"``, ``"frontend"``).  This is
        the only operation that adds structure to the negative network.
        """
        cfg = _resolve_config(None, {"epochs": epochs, **overrides})
        return self._pass(
            texts, cfg, blame=True, amount=abs(float(severity)), reason=_clean_reason(reason), source=source,
            note=note, progress=progress, stop_event=stop_event, checkpoint_manager=checkpoint_manager,
        )

    def clear(
        self,
        texts: Iterable[str] | str,
        *,
        weight: float = 1.0,
        epochs: int = 1,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        **overrides,
    ) -> list[dict]:
        """The tutor passed these texts: credit the edges they share with known failures, creating nothing.

        An edge seen in cleared text is weaker evidence (its net blame is
        ``blame / (1 + clear)``), so a fragment that shows up in good and bad
        text alike stops carrying the verdict on its own.  Texts the negative
        structure cannot walk are counted as ``unmatched`` and change nothing.
        """
        cfg = _resolve_config(None, {"epochs": epochs, **overrides})
        return self._pass(
            texts, cfg, blame=False, amount=abs(float(weight)), reason="", source="", note="",
            progress=progress, stop_event=stop_event,
        )

    def train(
        self,
        texts: Iterable[str] | str,
        config: TrainConfig | None = None,
        *,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        phase: str | None = None,
        reason: str | None = None,
        severity: float = 1.0,
        source: str = "",
        note: str = "",
        **overrides,
    ) -> list[dict]:
        """Training this network **is** blaming: every text given here is a failure (see :meth:`blame`).

        ``lr`` / ``act_lr`` / ``batch_size`` in the config are accepted and
        ignored; ``epochs`` repeats the blame pass.
        """
        cfg = _resolve_config(config, overrides)
        return self._pass(
            texts, cfg, blame=True, amount=abs(float(severity)), reason=_clean_reason(reason), source=source,
            note=note, progress=progress, stop_event=stop_event, checkpoint_manager=checkpoint_manager,
        )

    def punish(self, texts: Iterable[str] | str, *, epochs: int = 1, strength: float | None = 1.0,
               lr: float | None = None, reason: str | None = "thumbs-down", source: str = "feedback",
               note: str = "", progress: ProgressFn | None = None, stop_event: threading.Event | None = None,
               **overrides) -> list[dict]:
        """Thumbs down: :meth:`blame` with ``strength`` as the severity."""
        return self.blame(
            texts, reason=reason, severity=1.0 if strength is None else float(strength), source=source, note=note,
            epochs=epochs, progress=progress, stop_event=stop_event, **overrides,
        )

    def reward(self, texts: Iterable[str] | str, *, epochs: int = 1, strength: float | None = 1.0,
               lr: float | None = None, progress: ProgressFn | None = None,
               stop_event: threading.Event | None = None, **overrides) -> list[dict]:
        """Thumbs up: :meth:`clear` - the negative network never learns *from* correct text, it only lets go of blame."""
        return self.clear(
            texts, weight=1.0 if strength is None else float(strength), epochs=epochs, progress=progress,
            stop_event=stop_event, **overrides,
        )

    def two_nrl(
        self,
        bad: Iterable[str] | str,
        good: Iterable[str] | str,
        neg_epochs: int = 3,
        pos_epochs: int = 3,
        neg_lr: float | None = None,
        pos_lr: float | None = None,
        progress: ProgressFn | None = None,
        checkpoint_manager=None,
        stop_event: threading.Event | None = None,
        strength: float | None = 1.0,
        bad_weights: Sequence[float] | None = None,
        reason: str | None = None,
        source: str = "",
        note: str = "",
        **overrides,
    ) -> dict:
        """2NRL for the negative network: blame ``bad``, then clear ``good``.

        Nothing is inverted - blame already makes a path the *likely* one here
        and the model is never asked to write text.  ``neg_lr`` / ``pos_lr``
        are accepted for interface parity and ignored; ``bad_weights`` scales
        the severity per text (the worse the failure, the heavier the blame).
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs")
        base = 1.0 if strength is None else float(strength)
        negative: list[dict] = []
        if bad_weights is None:
            negative = self.blame(
                bad, reason=reason, severity=base, source=source, note=note, epochs=neg_epochs, progress=progress,
                stop_event=stop_event, **overrides,
            )
        else:
            items = [bad] if isinstance(bad, str) else list(bad)
            weights = [float(w) for w in bad_weights]
            if len(weights) != len(items):
                raise ValueError(f"bad_weights has {len(weights)} entries for {len(items)} texts")
            for text, weight in zip(items, weights):
                if stop_event is not None and stop_event.is_set():
                    break
                if weight <= 0:
                    continue
                records = self.blame(
                    [text], reason=reason, severity=base * weight, source=source, note=note, epochs=neg_epochs,
                    stop_event=stop_event, **overrides,
                )
                for record in records:
                    record["weight"] = weight
                    if progress is not None:
                        progress(record)
                negative.extend(records)
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            positive = self.clear(good, weight=base, epochs=pos_epochs, progress=progress, stop_event=stop_event, **overrides)
        self.meta["twonrl_runs"] += 1
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, self.meta["twonrl_runs"], "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    def invert(self) -> None:
        """Swap blame and clearing (see :meth:`NegativeGraph.invert`)."""
        self.graph.invert()

    def invert_paths(
        self, texts: Iterable[str] | str, mode: str = "activation", amounts=None, strength: float = 1.0,
        reason: str | None = "blatant", **options
    ) -> dict:
        """Failures from the evolve loop: blame every path by ``strength * amount`` (the worse the fake, the more).

        Returns ``{"texts", "flipped", "unit": "edges", "mode": "blame", "amount_mean"}``.
        """
        texts, _ = self._clean_texts(texts)
        values = self._amounts(texts, amounts)
        touched = 0
        for text, amount in zip(texts, values):
            if amount <= 0:
                continue
            records = self.blame(
                [text], reason=reason, severity=abs(float(strength)) * amount, source="evolve", epochs=1,
            )
            touched += records[-1]["edges_touched"] if records else 0
        applied = [v for v in values if v > 0]
        return {
            "texts": len(texts), "flipped": touched, "unit": "edges", "mode": "blame",
            "amount_mean": sum(applied) / len(applied) if applied else 0.0,
        }

    def correct(
        self,
        wrong: str,
        right: str,
        *,
        reason: str | None = None,
        severity: float = 1.0,
        source: str = "tutor",
        note: str = "",
        clear: float = 1.0,
    ) -> dict:
        """Learn one correction: blame only the characters the tutor changed, and clear what it kept.

        ``wrong`` is the sentence the network wrote, ``right`` the sentence the
        teacher wrote instead.  The two are aligned character by character
        (:mod:`radixnet.diff`) and

        * only the steps of ``wrong`` that wrote a character the teacher struck
          out or replaced are blamed - the words both sentences agree on are
          not the mistake, so they carry no verdict;
        * the correction clears blame wherever the failure structure already
          knows it (nothing is created: a correction is correct English, and
          this is a network of failures), which also lifts the blame off the
          words the wrong sentence shares with it;
        * an edge the correction walks too is never blamed - the network wrote
          the right characters by another route.

        Returns ``{"changes", "edits", "blamed", "cleared", "reason",
        "severity", "wrong_chars", "right_chars"}``.
        """
        from . import diff  # local import: the alignment is only needed for corrections

        wrong, right = str(wrong or ""), str(right or "")
        tag = _clean_reason(reason)
        amount = abs(float(severity))
        changes = diff.summary(wrong, right, limit=0)
        wrong_spans, right_spans = diff.changed_spans(wrong, right)
        result = {
            "changes": changes[:8], "edits": len(changes), "blamed": 0, "cleared": 0, "reason": tag,
            "severity": amount, "phase": "correction",
            "wrong_chars": sum(hi - lo for lo, hi in wrong_spans),
            "right_chars": sum(hi - lo for lo, hi in right_spans),
        }
        if len(wrong) < _W:
            return result
        grams = self.encoder.encode(wrong)
        self._register([grams])  # the failure joins the structure; the correction never does
        cleared = {e for _p, e in self._shared_edges(right)} if len(right) >= _W else set()
        blamed = [e for e in self._steps_over(grams, len(wrong), wrong_spans) if e not in cleared]
        if blamed and amount:
            result["blamed"] = self.graph.record_failure(blamed, amount, tag)
            self.meta["failures_total"] += 1
            self.meta["blame_total"] += amount * result["blamed"]
            self.meta["trained_texts"] += 1
            self.meta["trained_chars"] += len(wrong)
            if source:
                self.meta["sources"][source] = self.meta["sources"].get(source, 0) + 1
            self._note(wrong, reason=tag, severity=amount, source=source, note=note)
        if cleared and clear:
            result["cleared"] = self.graph.record_clear(cleared, clear)
            self.meta["cleared_total"] += 1
        return result

    def forget(self, reason: str | None = None, factor: float = 0.0) -> dict:
        """Drop (or fade) the blame of one reason, or of everything - the tutor can be wrong too."""
        return self.graph.forget(reason, factor)

    def configure_weights(self, **options: float) -> dict:
        """Change the blame weight function's scales (see :meth:`NegativeGraph.configure`)."""
        return self.graph.configure(**options)

    def weight_config(self) -> dict:
        return self.graph.weight_config()

    # -- prediction ----------------------------------------------------------

    def predict(
        self,
        prefix: str,
        length: int = 20,
        mode: str = "beam",
        k: int = 5,
        beam: int | None = None,
        step_penalty: float = 0.0,
        temperature: float = 1.0,
        to_end: bool = False,
        max_length: int | None = None,
    ) -> Prediction:
        """How a prefix goes wrong: the ``k`` most likely continuations *among the known failures* (and the ``k``
        least likely, as ``bottom``).

        Same search as the count / reward model (``"dijkstra"`` is an alias of
        ``"beam"``, ``"sample"`` draws one walk); the distribution it searches
        is the failure distribution, so a prediction here is a warning, not a
        suggestion.
        """
        self._check_predict_args(prefix, length, max_length, k, beam)
        mode = (mode or "beam").lower()
        if mode == "dijkstra":
            mode = "beam"
        if mode not in ("beam", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'beam', 'dijkstra' or 'sample'")
        return self._search(prefix, length, mode, k, beam, step_penalty, temperature, to_end, max_length)

    # -- the filter ----------------------------------------------------------

    def crossings(self, text: str) -> list[dict]:
        """Every transition of ``text`` through the negative structure, in order.

        ``{"index", "start", "end", "fragment", "parent", "edge", "blame",
        "fails", "clear", "evidence", "reason", "reasons"}`` per entry; ``edge`` is
        ``None`` where the negative network has never been there (an unknown
        trigram, a missing edge, or a junction it cannot represent), which is
        the common case for text that never failed.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        graph = self.graph
        grams = self.encoder.encode(text)
        if not grams:
            return []
        index = graph.trigram_index
        labels = graph.labels
        children = graph.children
        crossings: list[dict] = []
        node, offset, lost = START, 0, False

        def crossing(i: int, edge: int | None) -> dict:
            start = max(0, i - 1)
            end = min(len(text), i + _W)
            entry = {
                "index": i, "start": start, "end": end, "fragment": text[start:end], "parent": node, "edge": edge,
                "blame": 0.0, "fails": 0, "clear": 0.0, "evidence": 0.0, "reason": None, "reasons": [],
            }
            if edge is not None:
                reasons = graph.edge_reason_table(edge)
                entry.update(
                    blame=graph.edge_blame[edge], fails=graph.edge_fails[edge], clear=graph.edge_clear[edge],
                    evidence=graph.evidence(edge), reasons=reasons,
                    reason=reasons[0]["reason"] if reasons else None,
                )
            return entry

        def edge_from(n: int) -> int | None:
            if lost or (node != START and offset + _W != len(labels[node])):
                return None
            return children[node].get(n)

        for i, gram in enumerate(grams):
            loc = index.get(gram)
            if loc is None:
                crossings.append(crossing(i, None))
                lost = True
                continue
            n, o = loc
            if not lost and n == node and o == offset + 1:
                offset = o  # deterministic step inside a compressed node: no edge, nothing to blame
                continue
            crossings.append(crossing(i, None if o != 0 else edge_from(n)))
            node, offset, lost = n, o, False
        crossings.append(crossing(len(grams), edge_from(END)))
        return crossings

    def judge(
        self,
        text: str,
        *,
        threshold: float | None = None,
        min_coverage: float | None = None,
        spans: int = 5,
    ) -> dict:
        """Why this text looks like a failure - the negative network's verdict.

        Walks ``text`` through the negative structure and reports

        * ``blame`` - the summed net evidence (``max(0, blame - clear)``) of
          the transitions it shares with known failures, and ``risk`` - that
          sum over *all* of the text's transitions, so repeating a failure
          blamed once scores about 1 (more where the same fragment failed
          several times over) and sharing a third of one's transitions with it
          scores about 0.33; compression moves both numbers together, so the
          ratio does not depend on it,
        * ``coverage`` - the share of its transitions that are known failures,
          and ``peak`` - the most evidence any single transition carries (a
          correction blames one transition, so a fragment the teacher keeps
          rewriting shows up here long before ``risk`` moves),
        * ``reasons`` - the tutor's reasons behind that blame, heaviest first,
        * ``spans`` - the worst fragments, with their character range and the
          reason each one carries,
        * ``verdict`` - ``"reject"`` (enough coverage and ``risk`` at or above
          ``threshold``), ``"suspect"`` (known failures, but below it) or
          ``"pass"``, and ``why``, one sentence saying so.

        The defaults reject a text that carries, on average, a whole failure's
        worth of net evidence per transition (``threshold = 1``) over at least
        half of it (``min_coverage = 0.5``) - repeating something the tutor
        hated, in other words, rather than merely sharing fragments with it.

        A text the network has never seen fail scores 0 and passes: this is a
        filter, not a censor.  Defaults come from ``self.threshold`` /
        ``self.min_coverage``.
        """
        limit = self.threshold if threshold is None else float(threshold)
        floor = self.min_coverage if min_coverage is None else float(min_coverage)
        crossings = self.crossings(text)
        chars = len(text)
        total = len(crossings)
        blamed = [c for c in crossings if c["evidence"] > 0]
        known = sum(1 for c in crossings if c["edge"] is not None)
        blame = sum(c["evidence"] for c in blamed)
        coverage = len(blamed) / total if total else 0.0
        risk = blame / max(1, total)
        by_reason: dict[str, float] = {}
        for entry in blamed:
            share = entry["evidence"] / entry["blame"] if entry["blame"] else 0.0
            if entry["reasons"]:
                for row in entry["reasons"]:
                    by_reason[row["reason"]] = by_reason.get(row["reason"], 0.0) + row["blame"] * share
            else:
                by_reason[UNSPECIFIED_REASON] = by_reason.get(UNSPECIFIED_REASON, 0.0) + entry["evidence"]
        reasons = [{"reason": name, "blame": value} for name, value in by_reason.items()]
        reasons.sort(key=lambda row: (-row["blame"], row["reason"]))
        weight_total = sum(row["blame"] for row in reasons) or 1.0
        for row in reasons:
            row["share"] = row["blame"] / weight_total
        ranked = sorted(blamed, key=lambda c: -c["evidence"])
        worst = ranked[: max(0, spans)]
        peak = ranked[0]["evidence"] if ranked else 0.0
        enough = coverage >= floor and blamed
        if enough and risk >= limit:
            verdict = "reject"
        elif blamed:
            verdict = "suspect"
        else:
            verdict = "pass"
        self.meta["judgements"] += 1
        if verdict == "reject":
            self.meta["rejected"] += 1
        return {
            "text": text,
            "chars": chars,
            "transitions": total,
            "known": known,
            "blamed": len(blamed),
            "coverage": coverage,
            "blame": blame,
            "risk": risk,
            "peak": peak,
            "per_char": blame / max(1, chars),
            "threshold": limit,
            "min_coverage": floor,
            "verdict": verdict,
            "reasons": reasons,
            "spans": [
                {
                    "start": c["start"], "end": c["end"], "fragment": c["fragment"], "blame": c["evidence"],
                    "fails": c["fails"], "clear": c["clear"], "reason": c["reason"],
                }
                for c in worst
            ],
            "why": self._why(verdict, len(blamed), total, risk, reasons, worst),
        }

    @staticmethod
    def _why(verdict: str, blamed: int, total: int, risk: float, reasons: list[dict], worst: list[dict]) -> str:
        """One sentence: what the verdict rests on."""
        if not blamed:
            return "nothing here has failed before"
        lead = f"{blamed} of {total} transitions are known failures (risk {risk:.2f})"
        reason = f", mostly {reasons[0]['reason']!r}" if reasons else ""
        where = f", worst at {worst[0]['fragment']!r}" if worst else ""
        tail = {"reject": "; rejected", "suspect": "; below the threshold, kept"}.get(verdict, "")
        return lead + reason + where + tail

    # -- introspection -------------------------------------------------------

    def reasons(self) -> list[dict]:
        """Everything the tutor has blamed, heaviest first: ``[{"reason", "blame", "fails", "edges", "share"}]``."""
        return self.graph.reason_table()

    def recent(self, limit: int = 20) -> list[dict]:
        """The newest journal entries (what was blamed, for what reason, in whose words), newest first."""
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        return [dict(entry) for entry in reversed(self.log[-limit:])] if limit else []

    def stats(self) -> dict:
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
            "failures_total": meta["failures_total"],
            "blame_total": meta["blame_total"],
            "cleared_total": meta["cleared_total"],
            "judgements": meta["judgements"],
            "rejected": meta["rejected"],
            "sources": dict(meta["sources"]),
            "edge_blame_total": g.total_blame,
            "edge_clear_total": g.total_clear,
            "reason_count": len(g.reasons),
            "top_reasons": g.reason_table()[:5],
            "threshold": self.threshold,
            "min_coverage": self.min_coverage,
            "share_scale": g.share_scale,
            "blame_scale": g.blame_scale,
            "clear_scale": g.clear_scale,
            "log_entries": len(self.log),
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": NEGATIVE_MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "kind": self.kind,
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "log": [dict(entry) for entry in self.log],
            "filter": {"threshold": self.threshold, "min_coverage": self.min_coverage},
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None) -> "NegativeNet":
        if not isinstance(d, dict) or d.get("format") != NEGATIVE_MODEL_FORMAT:
            raise ValueError(f"not a {NEGATIVE_MODEL_FORMAT} model document")
        version = int(d.get("version", 1))
        if version > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {NEGATIVE_MODEL_FORMAT} model version {version}")
        graph = NegativeGraph.from_dict(d["graph"])
        model = cls(seed=graph.seed, backend=backend, device=device)
        model.graph = graph
        model.history = [dict(r) for r in d.get("history", [])]
        model.log = [dict(entry) for entry in d.get("log", [])][-MAX_LOG_ENTRIES:]
        settings = d.get("filter") or {}
        model.threshold = float(settings.get("threshold", model.threshold))
        model.min_coverage = float(settings.get("min_coverage", model.min_coverage))
        meta = cls._new_meta(graph.seed)
        meta.update(d.get("meta") or {})
        meta["sources"] = dict(meta.get("sources") or {})
        model.meta = meta
        return model

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"NegativeNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"blame={g.total_blame:.3g}, reasons={len(g.reasons)})"
        )
