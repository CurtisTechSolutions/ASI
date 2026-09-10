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

from .encoding import WINDOW
from .model import RadixNet

__all__ = ["EvolveConfig", "Evolver"]

ProgressFn = Callable[[dict], None]


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

    def validate(self) -> None:
        """Raise ``ValueError`` for values the loop cannot run with."""
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
        generator: RadixNet,
        corpus: Iterable[str],
        discriminator: RadixNet | None = None,
        config: EvolveConfig | None = None,
    ) -> None:
        self.config = EvolveConfig() if config is None else config
        self.config.validate()
        self.generator = generator
        self.corpus = [t for t in corpus if isinstance(t, str) and len(t) >= WINDOW]
        if not self.corpus:
            raise ValueError(f"corpus needs at least one text of {WINDOW}+ characters")
        if discriminator is None:
            discriminator = RadixNet(
                seed=self.config.seed + 1, backend=generator.backend.name, device=generator.backend.device
            )
        self.discriminator = discriminator
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
        worst: list[str] = []
        if fakes:
            cut = statistics.median(fake_scores)
            worst = [f for f, s in zip(fakes, fake_scores) if s <= cut] or [fakes[0]]
        result = self.generator.two_nrl(
            bad=worst, good=real, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs,
            neg_lr=cfg.neg_lr, pos_lr=cfg.pos_lr, batch_size=cfg.batch_size,
        )
        positive = result["positive"]
        fake_mean = statistics.fmean(fake_scores) if fake_scores else None
        real_mean = statistics.fmean(real_scores) if real_scores else None
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
            "seconds": time.perf_counter() - t0,
        }
        self.history.append(record)
        return record

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
