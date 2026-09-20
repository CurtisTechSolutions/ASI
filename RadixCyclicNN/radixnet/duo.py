"""The two networks as one output path: the positive model writes, the negative one vetoes.

This is the GAN at *output* time rather than at training time.  In
:mod:`radixnet.gan` the generator and the discriminator take turns improving
each other; here the finished pair works together on every answer:

1. the **positive** model (a :class:`~radixnet.model.RadixNet` or the count /
   reward model) proposes candidates - it is the generator, and the only one
   of the two that can write;
2. the **negative** model (:class:`~radixnet.negative.NegativeNet`, built from
   nothing but the tutor's failures) judges each candidate - it is the
   discriminator, and the only one of the two that knows what going wrong
   looks like;
3. what survives is returned, what does not comes back with the reason it was
   dropped, the fragment to blame and who said so.

Two signals decide, and either one is enough to reject (both only after the
candidate passes a coverage gate, so text the tutor has never failed is never
vetoed on a hunch):

* **blame** - ``risk``, the net evidence per transition that
  :meth:`~radixnet.negative.NegativeNet.judge` walks out of the failure
  structure, against ``threshold``;
* **peak** (off by default) - the evidence on a *single* transition, for
  vetoing a candidate that carries one fragment the tutor has already
  corrected even though the rest of it is fine;
* **ratio** - ``log P_negative(text) - log P_positive(text)`` per character,
  the classic discriminator logit of two generative models: how much more the
  candidate looks like known failure than like the text the positive model
  was trained on.

Nothing here trains by default.  The negatives come from the tutor
(:mod:`radixnet.blame`, :mod:`radixnet.tutor`); :meth:`NegativeFilter.teach` is the way in, and
``config.learn`` (off) is the deliberate exception that lets the filter blame
its own rejects.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .beam import Prediction
from .encoding import WINDOW
from .model import GraphModel
from .negative import NegativeNet
from .search import PathResult
from . import blame

__all__ = ["FilterConfig", "NegativeFilter"]

_W = WINDOW


@dataclass
class FilterConfig:
    """How strictly the negative network filters the positive one's output."""

    threshold: float | None = None
    """Risk (net blame per transition) at or above which a candidate is rejected; ``None`` = the negative model's own."""
    min_coverage: float | None = None
    """Share of a candidate's transitions that must be known failures before any rule may reject it."""
    ratio: float | None = 0.0
    """Log-odds (negative minus positive, per character) at or above which a candidate is rejected; ``None`` = off."""
    peak: float | None = None
    """Evidence on a *single* transition at or above which a candidate is rejected (``None`` = off).

    The blame rule weighs a text as a whole, which is right for gibberish and
    too forgiving for a one-word mistake: a correction blames the handful of
    characters the teacher changed and nothing else.  Set this to veto any
    candidate carrying a fragment the tutor has already corrected (1 = once).
    """
    over_sample: int = 3
    """Candidates drawn per wanted output, so the filter has something to choose from."""
    strict: bool = False
    """Also drop candidates the negative network only finds *suspect*."""
    spans: int = 3
    """Blamed fragments reported per verdict."""
    learn: bool = False
    """Blame what this filter rejects (off: the tutor supplies the negatives, the filter only applies them)."""
    reason: str = "filtered"
    """Reason recorded when ``learn`` is on."""

    def validate(self) -> None:
        """Raise ``ValueError`` for values the filter cannot run with."""
        for name in ("threshold", "min_coverage", "ratio", "peak"):
            value = getattr(self, name)
            if value is not None and value != value:  # NaN
                raise ValueError(f"{name} must be a number or None")
        if self.min_coverage is not None and not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must lie in [0, 1], got {self.min_coverage}")
        if self.over_sample < 1:
            raise ValueError(f"over_sample must be >= 1, got {self.over_sample}")
        if self.spans < 0:
            raise ValueError(f"spans must be >= 0, got {self.spans}")


class NegativeFilter:
    """The positive model's output, filtered by the negative one (see the module docstring).

    ``positive`` writes and is never modified here; ``negative`` judges.  The
    pair is stateless apart from the two models, so the same filter can be
    used from several places at once.
    """

    def __init__(self, positive: GraphModel, negative: NegativeNet, config: FilterConfig | None = None) -> None:
        if not isinstance(negative, NegativeNet):
            raise TypeError(f"the negative model must be a NegativeNet, got {type(negative).__name__}")
        if positive is negative:
            raise ValueError("the positive and negative models must be two different networks")
        self.config = FilterConfig() if config is None else config
        self.config.validate()
        self.positive = positive
        self.negative = negative

    @property
    def ready(self) -> bool:
        """Whether the negative half has anything to say: a failure has been blamed and not fully cleared.

        An empty negative network vetoes nothing, so a filter around one costs
        the over-sampling and buys nothing - this is what the output paths ask
        before putting the pair in the way (see :meth:`radixnet.api.ModelService.guard`).
        """
        return self.negative.graph.total_blame > 0.0

    # -- judging -------------------------------------------------------------

    def judge(self, text: str) -> dict:
        """The pair's verdict on one text: the negative network's blame, the likelihood ratio, and the decision.

        ``{"text", "decision": "pass" | "suspect" | "reject", "rule", "risk",
        "coverage", "blame", "ratio", "positive", "negative", "reasons",
        "spans", "why"}``.  ``rule`` names what rejected it (``"blame"``,
        ``"ratio"``) and is ``None`` when nothing did.
        """
        cfg = self.config
        verdict = self.negative.judge(
            text, threshold=cfg.threshold, min_coverage=cfg.min_coverage, spans=cfg.spans
        )
        positive = self.positive.score(text)
        negative = self.negative.score(text)
        ratio = negative["per_char"] - positive["per_char"]
        gate = verdict["coverage"] >= verdict["min_coverage"] and verdict["blamed"] > 0
        rule: str | None = None
        if verdict["verdict"] == "reject":
            rule = "blame"
        elif cfg.peak is not None and verdict["peak"] >= cfg.peak and verdict["blamed"] > 0:
            rule = "peak"  # a single fragment the tutor has corrected is enough, whatever the rest of the text is
        elif gate and cfg.ratio is not None and ratio >= cfg.ratio:
            rule = "ratio"
        decision = "reject" if rule else verdict["verdict"]
        if decision == "suspect" and cfg.strict:
            decision, rule = "reject", "suspect"
        return {
            "text": text,
            "decision": decision,
            "rule": rule,
            "risk": verdict["risk"],
            "peak": verdict["peak"],
            "coverage": verdict["coverage"],
            "blame": verdict["blame"],
            "threshold": verdict["threshold"],
            "min_coverage": verdict["min_coverage"],
            "peak_threshold": cfg.peak,
            "ratio": ratio,
            "ratio_threshold": cfg.ratio,
            "positive": positive["per_char"],
            "negative": negative["per_char"],
            "reasons": verdict["reasons"],
            "spans": verdict["spans"],
            "why": self._why(decision, rule, verdict, ratio),
        }

    @staticmethod
    def _why(decision: str, rule: str | None, verdict: dict, ratio: float) -> str:
        """One sentence: what the decision rests on."""
        if decision == "pass":
            return verdict["why"]
        if rule == "peak":
            span = verdict["spans"][0] if verdict["spans"] else None
            where = f" at {span['fragment']!r}" if span else ""
            reason = f" ({span['reason']})" if span and span["reason"] else ""
            return f"it carries {verdict['peak']:.2f} of blame on one fragment{where}{reason}; rejected"
        if rule == "ratio":
            reason = f", mostly {verdict['reasons'][0]['reason']!r}" if verdict["reasons"] else ""
            return (
                f"it reads {ratio:.2f} nats/char more like known failure than like the training data"
                f"{reason}; rejected"
            )
        if rule == "suspect":
            return verdict["why"].replace("; below the threshold, kept", "; rejected (strict)")
        return verdict["why"]

    def filter(self, texts: Iterable[str] | str) -> dict:
        """Judge every text; returns ``{"kept", "rejected", "verdicts", "accepted", "rate"}``.

        ``kept`` / ``rejected`` are the texts themselves (in input order),
        ``verdicts`` the full judgements.  With ``config.learn`` the rejected
        texts are blamed as new failures.
        """
        items = [texts] if isinstance(texts, str) else [str(t) for t in texts]
        verdicts = [self.judge(t) for t in items]
        kept = [v["text"] for v in verdicts if v["decision"] != "reject"]
        rejected = [v["text"] for v in verdicts if v["decision"] == "reject"]
        if rejected and self.config.learn:
            self.negative.blame(
                [t for t in rejected if len(t) >= _W], reason=self.config.reason, source="filter",
                note="rejected by the filter",
            )
        return {
            "kept": kept,
            "rejected": rejected,
            "verdicts": verdicts,
            "accepted": len(kept),
            "rate": len(kept) / len(items) if items else None,
        }

    def rank(self, prefix: str, result: PathResult) -> tuple[PathResult, list[dict]]:
        """A finished prediction with the vetoed continuations taken out of it, and a verdict on each.

        The search has already run, so this re-ranks what it offered rather
        than asking for more: a guarded prediction costs one judgement per
        continuation and no second search.  A :class:`~radixnet.beam.Prediction`
        keeps its surviving ``top`` in the search's own order and the best of
        them becomes the prediction itself; when none survives, the prediction
        is the prefix and nothing else (``expanded`` still reports the search
        that was run).
        """
        offered = list(result.top) if isinstance(result, Prediction) else [result]
        verdicts: list[dict] = []
        kept: list[PathResult] = []
        for candidate in offered:
            verdict = self.judge(prefix + candidate.text)
            verdicts.append(verdict)
            if verdict["decision"] != "reject":
                kept.append(candidate)
        best = kept[0] if kept else PathResult(full_text=prefix)
        carried = {f.name: getattr(best, f.name) for f in dataclasses.fields(PathResult) if f.name != "expanded"}
        if isinstance(result, Prediction):
            return dataclasses.replace(result, top=kept, **carried), verdicts
        return dataclasses.replace(result, **carried), verdicts

    # -- output --------------------------------------------------------------

    def generate(
        self,
        count: int = 1,
        *,
        over_sample: int | None = None,
        mode: str = "sample",
        max_length: int = 60,
        temperature: float = 1.0,
        prefix: str = "",
        seed: int | None = None,
        step_penalty: float = 0.0,
        beam: int | None = None,
        traversal: str = "reward",
    ) -> dict:
        """Generate through the pair: over-sample from the positive model, keep what the negative one allows.

        Returns ``{"texts", "results", "kept", "rejected", "verdicts",
        "candidates", "asked", "rate"}``; ``texts`` are the ``count``
        surviving candidates with the least blame (the cleanest first, and
        among equally clean ones the positive model's own order - the veto
        re-ranks as little as it can), ``results`` the walks behind them, and
        ``rejected`` carries every dropped candidate with the reason it was
        dropped.  Fewer than ``count`` texts come back when the filter vetoed
        too much - that is information, not an error.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return {"texts": [], "kept": [], "rejected": [], "verdicts": [], "candidates": 0, "asked": 0,
                    "rate": None, "results": []}
        factor = self.config.over_sample if over_sample is None else int(over_sample)
        if factor < 1:
            raise ValueError(f"over_sample must be >= 1, got {factor}")
        asked = count * factor
        results = self.positive.generate(
            max_length=max_length, mode=mode, temperature=temperature, count=asked, seed=seed, prefix=prefix,
            step_penalty=step_penalty, beam=beam, traversal=traversal,
        )
        candidates: list[str] = []
        paths: dict[str, Any] = {}
        for result in results:
            text = result.text
            if text and text not in paths:
                candidates.append(text)
                paths[text] = result
        outcome = self.filter(candidates)
        keepers = [v for v in outcome["verdicts"] if v["decision"] != "reject"]
        keepers.sort(key=lambda v: v["risk"])  # cleanest first; a stable sort keeps the model's own order among equals
        survivors = keepers[:count]
        return {
            "texts": [v["text"] for v in survivors],
            "results": [paths[v["text"]] for v in survivors],  # the walks behind them: cost, probability, path
            "kept": outcome["kept"],
            "rejected": [v for v in outcome["verdicts"] if v["decision"] == "reject"],
            "verdicts": outcome["verdicts"],
            "candidates": len(candidates),
            "asked": asked,
            "rate": outcome["rate"],
        }

    def predict(
        self,
        prefix: str,
        length: int = 20,
        *,
        k: int = 5,
        beam: int | None = None,
        step_penalty: float = 0.0,
        to_end: bool = False,
        max_length: int | None = None,
        traversal: str = "reward",
    ) -> dict:
        """Continue ``prefix`` through the pair: the positive model's top-K continuations, minus the vetoed ones.

        Returns ``{"prefix", "text", "kept", "rejected", "verdicts",
        "candidates", "warning"}``; ``text`` is the best surviving
        continuation (``None`` when every one of them was rejected) and
        ``warning`` is what the negative network predicts goes wrong from
        here, whether or not anything was rejected.
        """
        found = self.positive.predict(
            prefix, length=length, mode="beam", k=k, beam=beam, step_penalty=step_penalty, to_end=to_end,
            max_length=max_length, traversal=traversal,
        )
        candidates: list[str] = []
        for result in getattr(found, "top", None) or [found]:
            text = result.text
            if text and text not in candidates:
                candidates.append(text)
        outcome = self.filter([prefix + text for text in candidates])
        keepers = [v for v in outcome["verdicts"] if v["decision"] != "reject"]
        warning = self.negative.predict(prefix, length=length, k=1, max_length=max_length)
        return {
            "prefix": prefix,
            "text": keepers[0]["text"] if keepers else None,
            "kept": outcome["kept"],
            "rejected": [v for v in outcome["verdicts"] if v["decision"] == "reject"],
            "verdicts": outcome["verdicts"],
            "candidates": len(candidates),
            "warning": (prefix + warning.text) if warning.text else None,
        }

    def converse(
        self,
        opening: str = "",
        turns: int = 6,
        *,
        partner: GraphModel | None = None,
        **options: Any,
    ) -> dict:
        """Converse through the pair: every candidate reply the negative network refuses is left unsaid.

        The conversation is the positive model's (:func:`radixnet.dialogue.converse`);
        the filter only supplies the veto, so a turn whose every candidate is
        vetoed falls back exactly as a dead end does - a shorter context, then
        a fresh text - and the conversation stops when there is nothing left
        that may be said.

        Returns ``{"turns", "rejected", "verdicts", "vetoed"}``: the turns as
        spoken, a verdict on every distinct candidate the voices considered,
        and the ones the veto stopped with the reason.
        """
        verdicts: list[dict] = []
        seen: dict[str, bool] = {}  # the same candidate can be offered again after a shorter context

        def veto(text: str) -> bool:
            if text not in seen:
                verdict = self.judge(text)
                seen[text] = verdict["decision"] == "reject"
                verdicts.append(verdict)
            return seen[text]

        spoken = self.positive.converse(opening, turns, partner=partner, veto=veto, **options)
        rejected = [v for v in verdicts if v["decision"] == "reject"]
        if rejected and self.config.learn:
            self.negative.blame(
                [v["text"] for v in rejected if len(v["text"]) >= _W], reason=self.config.reason, source="filter",
                note="vetoed in conversation",
            )
        return {
            "turns": spoken,
            "rejected": rejected,
            "verdicts": verdicts,
            "vetoed": sum(turn.vetoed for turn in spoken),
        }

    # -- the tutor -----------------------------------------------------------

    def teach(self, reviews, *, threshold: float = 6.0, **options) -> dict:
        """Hand a tutor's review of some texts to the negative half (see :func:`radixnet.blame.teach_reviews`)."""
        return blame.teach_reviews(self.negative, reviews, threshold=threshold, **options)

    def describe(self) -> dict:
        """What the pair is made of, for a status line."""
        return {
            "positive": {"kind": self.positive.kind, "label": type(self.positive).label, **self.positive.stats()},
            "negative": {"kind": self.negative.kind, "label": type(self.negative).label, **self.negative.stats()},
            "config": {
                "threshold": self.config.threshold if self.config.threshold is not None else self.negative.threshold,
                "min_coverage": (
                    self.config.min_coverage if self.config.min_coverage is not None else self.negative.min_coverage
                ),
                "ratio": self.config.ratio,
                "peak": self.config.peak,
                "over_sample": self.config.over_sample,
                "strict": self.config.strict,
                "learn": self.config.learn,
            },
        }

    def __repr__(self) -> str:
        return f"NegativeFilter(positive={self.positive!r}, negative={self.negative!r}, config={self.config!r})"
