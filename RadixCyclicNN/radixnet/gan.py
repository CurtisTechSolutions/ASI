"""GAN-style self-upgrading loop (``Evolver``).

The *generator* is the model being improved; the *discriminator* is a second
:class:`RadixNet` that learns to tell real corpus text from generated text
via 2NRL (garbage = the generator's samples, correct = real samples).  Each
generation the discriminator scores the fakes, the worst half becomes the
generator's own 2NRL garbage and real samples are its fine-tune pass - the
generator keeps upgrading itself against a critic that keeps upgrading too.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import blame as blame_module
from .encoding import WINDOW
from .model import GraphModel, RadixNet
from .negative import NegativeNet

__all__ = ["BLATANT_MODES", "EvolveConfig", "Evolver"]

ProgressFn = Callable[[dict], None]
BLATANT_MODES = ("none", "fail_invert", "activation", "state")


@dataclass
class EvolveConfig:
    """Hyper-parameters of the evolution loop."""

    samples: int = 8
    """Fakes generated per generation."""
    real_per_generation: int = 8
    max_length: int = 40
    temperature: float = 1.0
    neg_epochs: int = 1
    pos_epochs: int = 1
    neg_lr: float = 0.05
    pos_lr: float = 0.01
    disc_neg_epochs: int = 1
    disc_pos_epochs: int = 1
    batch_size: int = 32
    """Transitions per backend step in every 2NRL phase (small batches learn faster per epoch)."""
    checkpoint_every: int = 0
    """Checkpoint the generator every N generations (0 = off)."""
    seed: int = 0
    blatant_mode: str = "none"
    """How failed fakes (scored below the real texts by the discriminator) drive the generator's update:

    * ``"none"``: the worst half of the fakes is ordinary 2NRL garbage (uniform ``neg_lr``).
    * ``"fail_invert"``: *train on failures, blatantly fail on purpose, then invert the model* - every failed fake
      is trained on in the negative phase with its learning rates (weights, states and the activation parameters)
      multiplied by ``min(blatant_boost, 1 + g / blatant_margin)`` where ``g`` is how far (per-char log-prob) below
      the real texts it scored, so the worse the response the more the activation functions update; then the whole
      model is inverted (what it now does confidently becomes what it confidently avoids) and fine-tuned on real
      texts.  The count model penalises the failures with the same multipliers instead (no inversion).
    * ``"activation"`` / ``"state"``: the local variant - no negative pass; every other node on a failed fake's path
      has its activation amplitude (or trained node value) moved toward its negation by ``min(1, g / (2 *
      blatant_margin))`` (a slight attenuation for a slightly worse fake, a neutralised path at the margin, a full
      inversion at twice the margin).  Fakes beyond the margin are *blatant* and leave the 2NRL garbage set, so when
      every bad fake was blatant the generation skips the negative pass and the global inversion altogether.
    """
    blatant_margin: float = 1.0
    """Per-char log-prob below the real texts at which a fake counts as blatant (``fail_invert``: the multiplier
    reaches 2 here; local modes: the path is neutralised here and fully inverted at twice this)."""
    blatant_boost: float = 4.0
    """``fail_invert``: the largest learning-rate multiplier a failure can get."""

    def validate(self) -> None:
        """Raise ``ValueError`` for values the loop cannot run with."""
        if self.blatant_mode not in BLATANT_MODES:
            raise ValueError(f"blatant_mode must be one of {', '.join(BLATANT_MODES)}, got {self.blatant_mode!r}")
        if not (self.blatant_margin >= 0):
            raise ValueError(f"blatant_margin must be >= 0, got {self.blatant_margin}")
        if not (self.blatant_boost >= 1):
            raise ValueError(f"blatant_boost must be >= 1, got {self.blatant_boost}")
        if self.samples < 1:
            raise ValueError(f"samples must be >= 1, got {self.samples}")
        if self.real_per_generation < 1:
            raise ValueError(f"real_per_generation must be >= 1, got {self.real_per_generation}")
        if self.max_length < WINDOW:
            raise ValueError(f"max_length must be >= {WINDOW}, got {self.max_length}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        for name in ("neg_epochs", "pos_epochs", "disc_neg_epochs", "disc_pos_epochs", "checkpoint_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


class Evolver:
    """Run generations of generator / discriminator 2NRL against ``corpus``."""

    def __init__(
        self,
        generator: GraphModel,
        corpus: Iterable[str],
        discriminator: GraphModel | None = None,
        config: EvolveConfig | None = None,
        negative: NegativeNet | None = None,
    ) -> None:
        self.config = EvolveConfig() if config is None else config
        self.config.validate()
        self.generator = generator
        self.corpus = [t for t in corpus if isinstance(t, str) and len(t) >= WINDOW]
        if not self.corpus:
            raise ValueError(f"corpus needs at least one text of {WINDOW}+ characters")
        if discriminator is None:
            # the same kind as the generator (RadixNet, or the count / reward model)
            discriminator = type(generator)(
                seed=self.config.seed + 1, backend=generator.backend.name, device=generator.backend.device
            )
        self.discriminator = discriminator
        self.negative = negative
        """Optional :class:`~radixnet.negative.NegativeNet`: here the discriminator is the critic, so every fake it
        scores below the real texts is blamed (reason ``"discriminator"``, or ``"blatant"`` past the margin) and the
        real texts clear blame."""
        self.rng = random.Random(self.config.seed)
        self.generation = 0
        self.history: list[dict] = []

    def _fakes(self) -> list[str]:
        cfg = self.config
        fakes = [
            r.text
            for r in self.generator.generate(
                max_length=cfg.max_length, mode="sample", temperature=cfg.temperature, count=cfg.samples
            )
        ]
        fakes = [f for f in fakes if len(f) >= WINDOW]
        if not fakes:
            text = self.generator.generate(max_length=cfg.max_length, mode="dijkstra")[0].text
            if len(text) >= WINDOW:
                fakes = [text]
        return fakes

    def run_generation(self) -> dict:
        """One generation; returns (and records) its summary dict."""
        cfg = self.config
        t0 = time.perf_counter()
        fakes = self._fakes()
        real = self.rng.sample(self.corpus, min(cfg.real_per_generation, len(self.corpus)))
        disc = self.discriminator
        disc.two_nrl(
            bad=fakes, good=real, neg_epochs=cfg.disc_neg_epochs, pos_epochs=cfg.disc_pos_epochs,
            batch_size=cfg.batch_size,
        )
        fake_scores = [disc.score(f)["per_char"] for f in fakes]
        real_scores = [disc.score(r)["per_char"] for r in real]
        fake_mean = statistics.fmean(fake_scores) if fake_scores else None
        real_mean = statistics.fmean(real_scores) if real_scores else None
        worst: list[str] = []
        if fakes:
            cut = statistics.median(fake_scores)
            worst = [f for f, s in zip(fakes, fake_scores) if s <= cut] or [fakes[0]]
        # failures: fakes the critic scores below the real texts, each with its gap g (per-char log-prob)
        blatant: list[str] = []
        failures: list[str] = []
        gaps: list[float] = []
        flipped = 0
        boost_mean = 0.0
        boost_max = 0.0
        mode = cfg.blatant_mode
        if mode != "none" and real_mean is not None and cfg.blatant_margin > 0:
            for fake, score in zip(fakes, fake_scores):
                gap = real_mean - score
                if gap <= 0:
                    continue
                failures.append(fake)
                gaps.append(gap)
                if gap > cfg.blatant_margin:
                    blatant.append(fake)
        twonrl = False
        if mode == "fail_invert" and not failures:
            worst = []  # nothing scored below the real texts: no failure to train on, so no negative pass
        if mode == "fail_invert" and failures:
            # train on the failures - blatantly fail on purpose: the worse the fake, the larger its learning rates -
            # then invert the model and fine-tune on real texts
            weights = [min(cfg.blatant_boost, 1.0 + gap / cfg.blatant_margin) for gap in gaps]
            result = self.generator.two_nrl(
                bad=failures, good=real, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs,
                neg_lr=cfg.neg_lr, pos_lr=cfg.pos_lr, batch_size=cfg.batch_size, bad_weights=weights,
            )
            positive = result["positive"]
            worst = failures
            twonrl = True
            boost_mean = statistics.fmean(weights)
            boost_max = max(weights)
        else:
            if mode in ("activation", "state") and failures:
                # the local variant: move every other node of a failed path toward its negation, the worse the more
                amounts = [min(1.0, gap / (2.0 * cfg.blatant_margin)) for gap in gaps]
                outcome = self.generator.invert_paths(failures, mode=mode, amounts=amounts)
                flipped = outcome["flipped"]
                boost_mean = outcome["amount_mean"]
                boost_max = max(amounts)
                blatant_set = set(blatant)
                worst = [f for f in worst if f not in blatant_set]
            if worst:
                result = self.generator.two_nrl(
                    bad=worst, good=real, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs,
                    neg_lr=cfg.neg_lr, pos_lr=cfg.pos_lr, batch_size=cfg.batch_size,
                )
                positive = result["positive"]
                twonrl = True
            else:
                # nothing left for the negative pass: only the fine-tune pass on real texts, no inversion
                positive = self.generator.reward(real, epochs=cfg.pos_epochs, lr=cfg.pos_lr, batch_size=cfg.batch_size)
        negative_report = self._teach_negative(fakes, fake_scores, real_mean, real)
        self.generation += 1
        g = self.generator.graph
        record = {
            "generation": self.generation,
            "fake_score_mean": fake_mean,
            "real_score_mean": real_mean,
            "gap": (real_mean - fake_mean) if fake_mean is not None and real_mean is not None else None,
            "gen_loss": positive[-1]["loss"] if positive else None,
            "nodes": g.num_nodes(),
            "edges": g.num_edges(),
            "compression_ratio": g.compression_ratio(),
            "sample": fakes[0] if fakes else "",
            "fakes": len(fakes),
            "worst": len(worst),
            "failures": len(failures),
            "blatant": len(blatant),
            "flipped": flipped,
            "boost_mean": boost_mean,
            "boost_max": boost_max,
            "twonrl": twonrl,
            "mode": mode,
            "seconds": time.perf_counter() - t0,
        }
        if negative_report is not None:
            record["negative_blamed"] = negative_report["blamed"]
            record["negative_edges"] = negative_report["edges"]
            record["negative_reasons"] = negative_report["reasons"]
        self.history.append(record)
        return record

    def _teach_negative(
        self, fakes: list[str], fake_scores: list[float], real_mean: float | None, real: list[str]
    ) -> dict | None:
        """Hand this generation's failures to the negative network: the discriminator is its tutor.

        Every fake the critic scored below the real texts is blamed by how far
        below it landed (``"blatant"`` past ``blatant_margin``, else
        ``"discriminator"``); the real texts clear blame.  Returns ``None``
        when no negative network is attached.
        """
        if self.negative is None or real_mean is None:
            return None
        cfg = self.config
        margin = cfg.blatant_margin if cfg.blatant_margin > 0 else 1.0
        faults = []
        for fake, score in zip(fakes, fake_scores):
            gap = real_mean - score
            if gap <= 0:
                continue
            faults.append({
                "text": fake,
                "reason": "blatant" if gap > cfg.blatant_margin else "discriminator",
                "severity": max(0.25, min(2.0, gap / margin)),
                "note": f"the discriminator scored it {gap:.3f} per char below the real texts",
                "source": "evolve",
            })
        if not faults:
            return {"blamed": 0, "edges": 0, "reasons": {}, "cleared": 0, "unmatched": 0, "severity_mean": 0.0,
                    "records": []}
        return blame_module.teach(self.negative, faults, real)

    def run(
        self,
        generations: int | None,
        checkpoint_manager=None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """Run ``generations`` generations (``None``: until ``stop_event`` is set).

        The stop event is checked before every generation.  With a
        ``checkpoint_manager`` the generator is checkpointed every
        ``config.checkpoint_every`` generations (tag ``"gen"``).
        """
        if generations is not None and generations < 0:
            raise ValueError(f"generations must be >= 0 or None, got {generations}")
        every = self.config.checkpoint_every
        records: list[dict] = []
        while generations is None or len(records) < generations:
            if stop_event is not None and stop_event.is_set():
                break
            record = self.run_generation()
            records.append(record)
            if progress is not None:
                progress(record)
            if checkpoint_manager is not None and every and self.generation % every == 0:
                checkpoint_manager.save(self.generator, self.generation, "gen", record)
        return records

    def __repr__(self) -> str:
        return f"Evolver(generation={self.generation}, corpus={len(self.corpus)}, config={self.config!r})"
