"""The negative network feeding itself: an LLM reviewer on a loop (`the Negative tab, automatic`).

Every other tutor in this project hands the negative network its failures as a
side effect of doing something else - the English tutor marks a sentence
(:mod:`radixnet.tutor`), the sandbox rejects a program (:mod:`radixnet.codegen`),
the recall tutor finds a misremembered waveform (:mod:`radixnet.recall`).  The
Negative tab itself was the one place where somebody had to *type* a failure in
by hand.

This module closes that loop.  One round is:

1. the **positive model** writes ``count`` texts of its own (a stochastic walk,
   optionally continuing a prefix);
2. an **LLM reviewer** - a local Ollama model by default, ChatGPT with
   ``provider="chatgpt"`` - marks each one out of 10, passes or fails it and
   writes a one-sentence critique (:func:`radixnet.ollama.adversarial_review`);
3. the failures **blame** the negative network, with the critique picking the
   reason and the mark setting the severity, and the passes **clear** blame off
   the fragments they share with known failures
   (:func:`radixnet.blame.teach_reviews`).

Then it does it again.  Nothing is invented: every failure still arrives from
something outside the network that looked at an output and said it was wrong,
and why - the only change is that nobody has to sit there doing it.

The loop touches the *negative* network only.  It never trains, rewards or
inverts the positive model, so it can be left running beside anything else that
is teaching it.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from .llm import DEFAULT_PROVIDER, PROVIDERS, normalise_provider

__all__ = ["Critic", "CriticConfig", "report_card"]

ProgressFn = Callable[[dict], Any]


@dataclasses.dataclass
class CriticConfig:
    """How the reviewer is run, and what it is told."""

    rounds: int = 3
    """Rounds to run; 0 means keep going until something stops it."""
    count: int = 8
    """Texts the model writes per round, for the reviewer to mark."""
    prefix: str = ""
    """Continue this prefix instead of writing from scratch."""
    max_length: int = 60
    temperature: float = 1.0
    threshold: float = 6.0
    """Pass mark out of 10: below it the text is a failure and is blamed."""
    context: str = ""
    """What the reviewer is told the texts are meant to be (its yardstick)."""
    provider: str = DEFAULT_PROVIDER
    reviewer_model: str = ""
    """The reviewer's model name; empty means the client's own default."""
    clear_passes: bool = True
    """Let the texts the reviewer passed take blame off what they share with known failures."""
    epochs: int = 1
    """Blame epochs per round."""
    seed: int | None = None
    """Seed of the first round's sampling; later rounds advance it, so rounds differ."""

    def validate(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be >= 0 (0 = until stopped)")
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.max_length < 0:
            raise ValueError("max_length must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.threshold <= 10:
            raise ValueError("threshold must lie in [0, 10]")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")
        if normalise_provider(self.provider) not in PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Critic:
    """The reviewer on a loop, keeping the negative network fed.

    ``model`` is the positive model whose output is reviewed, ``negative`` the
    :class:`~radixnet.negative.NegativeNet` that learns from the verdicts, and
    ``client`` any :class:`~radixnet.llm.LLMClient` (an Ollama or ChatGPT one).
    ``external`` wraps the slow LLM call so a server can release its model lock
    around it, exactly as the English tutor does.
    """

    def __init__(
        self,
        model: Any,
        negative: Any,
        client: Any,
        config: CriticConfig | None = None,
        external: Callable[[], Any] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("a positive model to review is required")
        if negative is None:
            raise ValueError("a negative network to teach is required")
        self.model = model
        self.negative = negative
        self.client = client
        self.config = config or CriticConfig()
        self.config.validate()
        self._external = external or nullcontext
        self.history: list[dict] = []
        self.round_no = 0

    # -- one round -----------------------------------------------------------

    def _seed(self) -> int | None:
        """This round's sampling seed: the configured one advanced by the round, so rounds differ."""
        if self.config.seed is None:
            return None
        return int(self.config.seed) + self.round_no

    def run_round(self) -> dict:
        """Write, review, blame - one round, returning its record."""
        from . import blame as blame_module
        from .ollama import review_texts, sample_texts, summarise_reviews

        cfg = self.config
        self.round_no += 1
        started = time.perf_counter()
        # The model writes first, under whatever lock the caller holds: sampling
        # walks the graph, so it must not run while another request may mutate it.
        samples = sample_texts(
            self.model, cfg.count, prefix=cfg.prefix, max_length=cfg.max_length,
            temperature=cfg.temperature, seed=self._seed(),
        )
        with self._external():  # only the reviewer's thinking happens outside the lock
            reviews = review_texts(
                self.client, samples, context=cfg.context or None,
                model=cfg.reviewer_model or None, threshold=cfg.threshold,
            )
        review = summarise_reviews(
            "model", cfg.reviewer_model or self.client.model, cfg.threshold, samples, reviews,
        )
        taught = blame_module.teach_reviews(
            self.negative, review, threshold=cfg.threshold, clear_passes=cfg.clear_passes,
            source="critic", epochs=cfg.epochs,
        )
        record = {
            "kind": "round",
            "round": self.round_no,
            "reviewer": review.get("model"),
            "threshold": float(cfg.threshold),
            "texts": len(review.get("texts") or []),
            "reviews": review.get("reviews") or [],
            "mean_rating": review.get("mean_rating"),
            "pass_rate": review.get("pass_rate"),
            "passed": len(review.get("good") or []),
            "failed": len(review.get("bad") or []),
            "blamed": taught["blamed"],
            "cleared": taught["cleared"],
            "unmatched": taught["unmatched"],
            "edges": taught["edges"],
            "reasons": taught["reasons"],
            "severity_mean": taught["severity_mean"],
            "stats": self.negative.stats(),
            "seconds": time.perf_counter() - started,
        }
        self.history.append(record)
        return record

    # -- the loop ------------------------------------------------------------

    def run(
        self,
        rounds: int | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """Run ``rounds`` rounds (``None`` uses the config; 0 or ``None`` there means until stopped).

        The stop event is checked before every round, so a loop left running
        ends cleanly after the round it is in.  The records are the round
        records, followed by one ``report`` record summarising them.
        """
        limit = self.config.rounds if rounds is None else rounds
        if limit is not None and limit < 0:
            raise ValueError(f"rounds must be >= 0 or None, got {limit}")
        forever = not limit
        records: list[dict] = []
        while forever or len(records) < limit:
            if stop_event is not None and stop_event.is_set():
                break
            record = self.run_round()
            records.append(record)
            if progress is not None:
                progress(record)
        card = report_card(records)
        records.append(card)
        if progress is not None:
            progress(card)
        return records

    def __repr__(self) -> str:
        return f"Critic(round={self.round_no}, config={self.config!r})"


def report_card(records: list[dict]) -> dict:
    """What a run of rounds came to: how much was reviewed, blamed and cleared, and why.

    ``mean_rating`` is over the rounds that produced one, and ``trend`` is the
    last round's mean rating minus the first's - positive when the reviewer is
    marking the model's output better than it did at the start.
    """
    rounds = [r for r in records if r.get("kind") == "round"]
    ratings = [float(r["mean_rating"]) for r in rounds if isinstance(r.get("mean_rating"), (int, float))]
    rates = [float(r["pass_rate"]) for r in rounds if isinstance(r.get("pass_rate"), (int, float))]
    reasons: dict[str, int] = {}
    for record in rounds:
        for reason, count in (record.get("reasons") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return {
        "kind": "report",
        "rounds": len(rounds),
        "reviewed": sum(int(r.get("texts") or 0) for r in rounds),
        "blamed": sum(int(r.get("blamed") or 0) for r in rounds),
        "cleared": sum(int(r.get("cleared") or 0) for r in rounds),
        "edges": sum(int(r.get("edges") or 0) for r in rounds),
        "mean_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "pass_rate": (sum(rates) / len(rates)) if rates else None,
        "trend": (ratings[-1] - ratings[0]) if len(ratings) > 1 else None,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "stats": rounds[-1].get("stats") if rounds else None,
    }
