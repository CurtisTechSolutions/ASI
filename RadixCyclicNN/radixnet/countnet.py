"""CountRewardNet - the count / reward model: a second algorithm on the same self-compressing graph.

Every edge keeps two numbers: ``count`` - how many times a training pass
traversed the path through it - and ``reward`` - the sum of the rewards (+)
and penalties (-) it received from feedback (thumbs up / down, 2NRL, the
code-generation judge, the adversarial review ...).  The edge weight is a
fixed function of the two::

    weight = count_scale * log(1 + count) + reward_scale * reward

and a parent's children are drawn by a softmax over those weights: every
node's activation is the constant 1 (``a = 0, k = 1`` in the sine
parameters), so the graph's edge score ``w * f_parent * f_child`` is the
weight itself and ``P(child | parent) ∝ (1 + count) ** count_scale *
exp(reward_scale * reward)``.  There is no gradient and no learning rate:
training counts traversals, feedback moves rewards, ``invert`` flips the
sign of every reward, and prediction is a beam search that returns the K most
likely *and* the K least likely continuations of one prefix in a single call
(:class:`~radixnet.beam.Prediction`).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Sequence

from .activation import DEFAULT_B, DEFAULT_H
from .backend import get_backend
from .beam import Prediction, beam_predict, default_beam
from .encoding import WINDOW, Decoder, Encoder
from .graph import RadixCyclicGraph
from .model import (
    MODEL_FORMAT_VERSION,
    GraphModel,
    ProgressFn,
    TrainConfig,
    _resolve_config,
    _utc_now,
    _weight_groups,
)
from .search import PathResult, sample_walk

__all__ = ["COUNT_MODEL_FORMAT", "CountRewardGraph", "CountRewardNet"]

COUNT_MODEL_FORMAT = "radixnet-count"
_W = WINDOW
_MAX_LOG_PPL = 700.0


class CountRewardGraph(RadixCyclicGraph):
    """A :class:`RadixCyclicGraph` whose edge weights are ``count_scale * log(1 + count) + reward_scale * reward``.

    Activations are the constant 1 on every node, so the base class's scores,
    probabilities, costs, Dijkstra / sampling walks, splits and merges all
    work unchanged; only how a weight comes about differs.  ``edge_reward``
    is a parallel list indexed by edge id like ``edge_w`` / ``edge_count``.
    """

    def __init__(self, seed: int = 0, count_scale: float = 1.0, reward_scale: float = 1.0) -> None:
        self.count_scale = float(count_scale)
        self.reward_scale = float(reward_scale)
        self.edge_reward: list[float] = []
        super().__init__(seed)

    # -- the weight function -------------------------------------------------

    def weight(self, count: float, reward: float) -> float:
        """The edge weight function of the two tracked numbers."""
        return self.count_scale * math.log1p(max(0.0, float(count))) + self.reward_scale * float(reward)

    def recompute_weights(self) -> None:
        """Write ``weight(count, reward)`` to every alive edge (after counts or scales changed)."""
        ew, ec, er = self.edge_w, self.edge_count, self.edge_reward
        weight = self.weight
        for e, ok in enumerate(self.edge_alive):
            if ok:
                ew[e] = weight(ec[e], er[e])
        self.version += 1

    def add_reward(self, edge_ids: Iterable[int], amount: float) -> int:
        """Add ``amount`` (negative = penalty) to the reward of every listed alive edge; returns how many."""
        ew, ec, er = self.edge_w, self.edge_count, self.edge_reward
        alive = self.edge_alive
        weight = self.weight
        touched = 0
        for e in edge_ids:
            if 0 <= e < len(er) and alive[e]:
                er[e] += amount
                ew[e] = weight(ec[e], er[e])
                touched += 1
        if touched:
            self.version += 1
        return touched

    def total_reward(self) -> tuple[float, float]:
        """``(sum of positive rewards, sum of negative rewards)`` over alive edges."""
        pos = neg = 0.0
        for e, ok in enumerate(self.edge_alive):
            if ok:
                r = self.edge_reward[e]
                if r > 0:
                    pos += r
                elif r < 0:
                    neg += r
        return pos, neg

    # -- construction overrides ----------------------------------------------

    def _new_node(self, label, z=None, a=None, b=DEFAULT_B, h=DEFAULT_H, k=0.0, count=0) -> int:
        # a = 0 and k = 1 make f(z) = 1 whatever z is: scores reduce to the edge weight
        return super()._new_node(label, z=0.0 if z is None else z, a=0.0, b=b, h=h, k=1.0, count=count)

    def _new_edge(self, p: int, c: int, count: int = 0) -> int:
        e = super()._new_edge(p, c, count)
        self.edge_reward.append(0.0)
        self.edge_w[e] = self.weight(count, 0.0)
        return e

    def invert(self) -> None:
        """Flip the sign of every reward (what was rewarded is now penalised and vice versa)."""
        er = self.edge_reward
        for e, ok in enumerate(self.edge_alive):
            if ok:
                er[e] = -er[e]
        self.inverted = not self.inverted
        self.recompute_weights()

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        rewards: list[float] = []
        for old, ok in enumerate(self.alive):
            if ok:
                for _c, e in self.children[old].items():
                    rewards.append(self.edge_reward[e])
        d["edges"]["reward"] = rewards
        d["weights"] = {"kind": "count-reward", "count_scale": self.count_scale, "reward_scale": self.reward_scale}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CountRewardGraph":
        g = super().from_dict(d)
        weights = d.get("weights") or {}
        g.count_scale = float(weights.get("count_scale", 1.0))
        g.reward_scale = float(weights.get("reward_scale", 1.0))
        rewards = d.get("edges", {}).get("reward")
        n = len(g.edge_w)
        if rewards is None:
            g.edge_reward = [0.0] * n
        else:
            if len(rewards) != n:
                raise ValueError("edge reward array has an inconsistent length")
            g.edge_reward = [float(v) for v in rewards]
        g.a = [0.0] * len(g.labels)
        g.k = [1.0] * len(g.labels)
        g.recompute_weights()
        return g


class CountRewardNet(GraphModel):
    """The count / reward model (see the module docstring).

    ``train`` counts traversals, ``reward`` / ``punish`` move rewards,
    ``two_nrl`` penalises the bad texts and then counts + rewards the good
    ones (no inversion), ``predict`` returns top-K and bottom-K continuations.
    Learning rates in a :class:`TrainConfig` are accepted and ignored; the
    magnitude of feedback is ``strength`` (default 1: one unit of reward
    multiplies an edge's odds by ``e``).
    """

    kind = "count"
    format = COUNT_MODEL_FORMAT
    label = "Count / reward"
    description = (
        "edge weight = log(1 + traversals) + rewards - penalties; no learning rate; "
        "beam prediction with the top-K and bottom-K continuations"
    )

    def __init__(
        self,
        seed: int = 0,
        backend: str = "auto",
        device: str | None = None,
        count_scale: float = 1.0,
        reward_scale: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.graph = CountRewardGraph(seed=self.seed, count_scale=count_scale, reward_scale=reward_scale)
        self.encoder = Encoder(_W)
        self.decoder = Decoder(_W)
        # no numeric learning rule runs, so the backend is only reported (python / cpu); backend / device are accepted
        # for interface parity with RadixNet
        self.backend = get_backend("python", None)
        self.history: list[dict] = []
        self.meta: dict = self._new_meta(self.seed)

    @staticmethod
    def _new_meta(seed: int) -> dict:
        meta = GraphModel._new_meta(seed)
        meta.update(rewards_total=0.0, penalties_total=0.0, feedback_passes=0)
        return meta

    # -- passes over data ----------------------------------------------------

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

    def _passes(
        self,
        texts: Iterable[str] | str,
        cfg: TrainConfig,
        *,
        count: bool,
        reward: float,
        phase: str | None,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """``cfg.epochs`` passes over ``texts``: each traverses (``count``) and / or rewards (``reward``) every path."""
        texts, skipped_short = self._clean_texts(texts)
        graph = self.graph
        meta = self.meta
        records: list[dict] = []
        grams = [self.encoder.encode(t) for t in texts]
        if count:
            meta["trained_texts"] += len(texts)
            meta["trained_chars"] += sum(len(t) for t in texts)
        # build the structure first (no counting) and compress it, so every pass - the first included - walks
        # the same transitions: steps inside a compressed node are deterministic and never counted
        self._observe_grams(grams, False)
        pending_merges = graph.compress() if cfg.auto_compress else 0
        for _ in range(cfg.epochs):
            t0 = time.perf_counter()
            transitions = self._observe_grams(grams, count)
            edges = [e for _, e in transitions]
            if reward:
                graph.add_reward(edges, reward)
                meta["feedback_passes"] += 1
                if reward > 0:
                    meta["rewards_total"] += reward * len(edges)
                else:
                    meta["penalties_total"] += -reward * len(edges)
            if count:
                graph.recompute_weights()
            loss = self._mean_cost(transitions)
            merges = (graph.compress() if cfg.auto_compress else 0) + pending_merges
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
                "transitions": len(transitions),
                "seconds": time.perf_counter() - t0,
                "skipped_short": skipped_short,
                "traversed": count,
                "reward": reward,
            }
            if phase is not None:
                record["phase"] = phase
            self.history.append(record)
            records.append(record)
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self, epoch, "epoch", record)
            if stop_event is not None and stop_event.is_set():
                break
        return records

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
        """Count one traversal of every text's path per epoch (structure is built on demand, as in RadixNet).

        The loss is the mean ``-log P`` of the transitions after the pass;
        ``lr`` / ``act_lr`` / ``batch_size`` in the config are ignored.
        """
        cfg = _resolve_config(config, overrides)
        return self._passes(
            texts, cfg, count=True, reward=0.0, phase=phase, checkpoint_manager=checkpoint_manager,
            progress=progress, stop_event=stop_event,
        )

    def reward(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs up: ``epochs`` passes that traverse *and* reward (``+strength``) every path of ``texts``."""
        cfg = _resolve_config(None, {"epochs": epochs, **overrides})
        return self._passes(
            texts, cfg, count=True, reward=abs(1.0 if strength is None else float(strength)), phase="positive",
            progress=progress, stop_event=stop_event,
        )

    def punish(
        self,
        texts: Iterable[str] | str,
        *,
        epochs: int = 1,
        strength: float | None = 1.0,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        lr: float | None = None,
        **overrides,
    ) -> list[dict]:
        """Thumbs down: ``epochs`` passes that penalise (``-strength``) every path of ``texts``; no traversal is counted."""
        cfg = _resolve_config(None, {"epochs": epochs, **overrides})
        return self._passes(
            texts, cfg, count=False, reward=-abs(1.0 if strength is None else float(strength)), phase="negative",
            progress=progress, stop_event=stop_event,
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
        **overrides,
    ) -> dict:
        """2NRL for the count model: penalise ``bad`` (``neg_epochs`` passes), then count + reward ``good``.

        Nothing is inverted: a penalty already makes a path unlikely.
        ``neg_lr`` / ``pos_lr`` are accepted for interface parity and ignored;
        the magnitude per pass is ``strength``, scaled per text by
        ``bad_weights`` when given (the worse a failure, the larger its penalty).
        """
        reserved = sorted({"epochs", "lr", "act_lr"} & set(overrides))
        if reserved:
            raise TypeError(f"two_nrl sets {', '.join(reserved)} per phase; use neg_epochs/pos_epochs")
        base = 1.0 if strength is None else float(strength)
        if bad_weights is None:
            negative = self.punish(bad, epochs=neg_epochs, strength=base, progress=progress, stop_event=stop_event, **overrides)
        else:
            negative = []
            for weight, group in _weight_groups(bad, bad_weights):
                if stop_event is not None and stop_event.is_set():
                    break
                records = self.punish(group, epochs=neg_epochs, strength=base * weight, stop_event=stop_event, **overrides)
                for record in records:
                    record["weight"] = weight
                    if progress is not None:
                        progress(record)
                negative.extend(records)
        positive: list[dict] = []
        if not (stop_event is not None and stop_event.is_set()):
            positive = self.reward(good, epochs=pos_epochs, strength=strength, progress=progress, stop_event=stop_event, **overrides)
        self.meta["twonrl_runs"] += 1
        if checkpoint_manager is not None:
            last = positive[-1] if positive else (negative[-1] if negative else None)
            checkpoint_manager.save(self, self.meta["twonrl_runs"], "2nrl", last)
        return {"negative": negative, "positive": positive, "inverted": self.graph.inverted}

    def invert(self) -> None:
        """Flip the sign of every reward."""
        self.graph.invert()

    def invert_paths(
        self, texts: Iterable[str] | str, mode: str = "activation", amounts=None, strength: float = 2.0, **options
    ) -> dict:
        """Failures: there is no activation to flip here, so every edge of a text's path loses ``strength * 2 * amount``
        reward (amount 1, the default, is a full ``2 * strength`` penalty; the worse the text, the larger).

        Returns ``{"texts", "flipped", "unit": "edges", "mode": "penalty", "amount_mean"}``.
        """
        texts, _ = self._clean_texts(texts)
        values = self._amounts(texts, amounts)
        graph = self.graph
        penalties: dict[int, float] = {}
        for path, amount in zip(self._paths_of(texts), values):
            penalty = abs(float(strength)) * 2.0 * amount
            if penalty <= 0:
                continue
            for p, c in zip(path, path[1:]):
                e = graph.children[p].get(c)
                if e is not None:
                    penalties[e] = max(penalties.get(e, 0.0), penalty)
        touched = 0
        for e, penalty in penalties.items():
            touched += graph.add_reward([e], -penalty)
            self.meta["penalties_total"] += penalty
        self.meta["feedback_passes"] += 1 if touched else 0
        applied = [v for v in values if v > 0]
        return {
            "texts": len(texts), "flipped": touched, "unit": "edges", "mode": "penalty",
            "amount_mean": sum(applied) / len(applied) if applied else 0.0,
        }

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
        """Continue ``prefix``: the ``k`` most likely and the ``k`` least likely continuations in one search.

        ``"beam"`` (``"dijkstra"`` is accepted as an alias) runs the beam
        search of :mod:`radixnet.beam`; the result *is* the best path (a
        :class:`~radixnet.search.PathResult`) and carries ``top`` / ``bottom``.
        ``"sample"`` draws one stochastic walk (``top = [it]``).  ``length``,
        ``to_end``, ``max_length`` and ``step_penalty`` mean what they mean for
        :meth:`RadixNet.predict`.
        """
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
        mode = (mode or "beam").lower()
        if mode == "dijkstra":
            mode = "beam"
        if mode not in ("beam", "sample"):
            raise ValueError(f"unknown mode {mode!r}; expected 'beam', 'dijkstra' or 'sample'")
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
            walk = sample_walk(graph, node, offset, max_chars=max(0, cap - len(lead)), temperature=temperature)
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

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict:
        g = self.graph
        meta = self.meta
        pos, neg = g.total_reward()
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
            "rewards_total": meta["rewards_total"],
            "penalties_total": meta["penalties_total"],
            "feedback_passes": meta["feedback_passes"],
            "edge_reward_positive": pos,
            "edge_reward_negative": neg,
            "count_scale": g.count_scale,
            "reward_scale": g.reward_scale,
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": COUNT_MODEL_FORMAT,
            "version": MODEL_FORMAT_VERSION,
            "saved_at": _utc_now(),
            "kind": self.kind,
            "meta": dict(self.meta),
            "history": [dict(r) for r in self.history],
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, backend: str = "auto", device: str | None = None) -> "CountRewardNet":
        if not isinstance(d, dict) or d.get("format") != COUNT_MODEL_FORMAT:
            raise ValueError(f"not a {COUNT_MODEL_FORMAT} model document")
        version = int(d.get("version", 1))
        if version > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {COUNT_MODEL_FORMAT} model version {version}")
        graph = CountRewardGraph.from_dict(d["graph"])
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
            f"CountRewardNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, inverted={g.inverted})"
        )
