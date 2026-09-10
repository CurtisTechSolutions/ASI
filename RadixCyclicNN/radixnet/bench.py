"""Throughput benchmarks for the radixnet core.

Two things are timed on a deterministic synthetic corpus:

* **training** - :meth:`RadixNet.train` end to end (structural observation,
  compression and ``epochs`` backend epochs), reported as transitions/sec
  and chars/sec;
* **prediction** - :meth:`RadixNet.predict` in Dijkstra mode from prefixes
  cut out of the corpus, reported as predictions/sec and Dijkstra
  expansions/sec.

The corpus is derived from ``data/sample_corpus.txt`` with a seeded RNG (see
:func:`synthetic_corpus`), so the same ``(chars, seed)`` always produces the
same texts, the same graph and the same search problems - numbers from
different runs, backends and code revisions are directly comparable.

Run directly::

    python -m radixnet.bench --chars 20000 --epochs 2 --backend python [--json]

or through the CLI (``python -m radixnet bench``), which calls
:func:`run_benchmark`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Sequence

from .model import RadixNet

__all__ = [
    "DATA_FILE",
    "DEFAULT_CHARS",
    "DEFAULT_EPOCHS",
    "PREDICT_LENGTH",
    "RESULT_KEYS",
    "load_base_lines",
    "synthetic_corpus",
    "prediction_prefixes",
    "run_benchmark",
    "format_report",
    "main",
]

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_corpus.txt"
)
"""Base lines of the synthetic corpus (a built-in fallback is used if it is missing)."""

DEFAULT_CHARS = 20_000
DEFAULT_EPOCHS = 2
PREDICT_LENGTH = 20
"""``length`` passed to :meth:`RadixNet.predict` for every timed prediction."""

MIN_PREDICTIONS = 100
MAX_PREDICTIONS = 20_000
"""Bounds of the default prediction count (``chars // 2`` clamped into this range)."""

_WARMUP_CHARS = 400
"""Size of the throwaway corpus used to initialise the backend before timing."""

_PREFIX_MAX_LEN = 12
"""Longest prefix cut out of a corpus text for the prediction benchmark."""

RESULT_KEYS = (
    "backend",
    "device",
    "chars",
    "texts",
    "epochs",
    "train_seconds",
    "transitions",
    "transitions_per_sec",
    "chars_per_sec",
    "predict_count",
    "predict_seconds",
    "predictions_per_sec",
    "dijkstra_expansions_per_sec",
    "nodes",
    "edges",
    "compression_ratio",
    "sample_prediction",
)
"""Keys every :func:`run_benchmark` result carries (in this order, extras follow)."""

_FALLBACK_LINES = (
    "the quick brown fox jumps over the lazy dog",
    "the cat sat on the mat",
    "the dog chased the cat around the garden",
    "a bird in the hand is worth two in the bush",
    "all that glitters is not gold",
    "the early bird catches the worm",
    "actions speak louder than words",
    "every cloud has a silver lining",
    "practice makes perfect",
    "better late than never",
    "a journey of a thousand miles begins with a single step",
    "where there is a will there is a way",
)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def load_base_lines(path: str = DATA_FILE) -> list[str]:
    """Non-blank lines (>= 3 characters) of the sample corpus.

    Falls back to a small built-in sentence list when the file cannot be
    read, so the benchmark also works for an installed package without the
    ``data/`` directory.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [line.strip() for line in fh]
    except OSError:
        lines = list(_FALLBACK_LINES)
    lines = [line for line in lines if len(line) >= 3]
    return lines or list(_FALLBACK_LINES)


def synthetic_corpus(chars: int, seed: int = 0, lines: Sequence[str] | None = None) -> list[str]:
    """Deterministic, structured training texts totalling at least ``chars`` characters.

    Each text is built from the base ``lines`` (default: :func:`load_base_lines`)
    by one of three recipes chosen with a seeded RNG:

    * a verbatim line (repetition -> unary chains -> compression),
    * two lines spliced at a word boundary (shared prefixes and suffixes ->
      branching nodes and cycles),
    * a line with one or two words replaced by other vocabulary words (local
      branching with realistic fan-out).

    The same ``(chars, seed, lines)`` always yields the same list.  Every text
    has at least three characters.  ``chars == 0`` gives an empty list.
    """
    if chars < 0:
        raise ValueError(f"chars must be >= 0, got {chars}")
    base = [line for line in (load_base_lines() if lines is None else lines) if len(line) >= 3]
    if not base:
        raise ValueError("need at least one base line of >= 3 characters")
    rng = random.Random(seed)
    vocab = sorted({word for line in base for word in line.split()})
    texts: list[str] = []
    total = 0
    while total < chars:
        roll = rng.random()
        if roll < 0.4 or len(vocab) < 2:
            text = rng.choice(base)
        elif roll < 0.7:
            left = rng.choice(base).split()
            right = rng.choice(base).split()
            text = " ".join(left[: rng.randint(1, len(left))] + right[rng.randrange(len(right)) :])
        else:
            words = rng.choice(base).split()
            for _ in range(rng.randint(1, 2)):
                words[rng.randrange(len(words))] = rng.choice(vocab)
            text = " ".join(words)
        if len(text) < 3:
            text = rng.choice(base)
        texts.append(text)
        total += len(text)
    return texts


def prediction_prefixes(texts: Sequence[str], count: int, seed: int = 0) -> list[str]:
    """``count`` deterministic prefixes: substrings (3-12 chars) of random ``texts``.

    Because every prefix occurs in the training data, its last trigram is
    known to the trained graph and the prediction starts at a real node.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count and not texts:
        raise ValueError("cannot cut prefixes out of an empty corpus")
    rng = random.Random(seed + 1)
    prefixes: list[str] = []
    for _ in range(count):
        text = rng.choice(texts)
        n = len(text)
        length = rng.randint(3, min(_PREFIX_MAX_LEN, n))
        start = rng.randint(0, n - length)
        prefixes.append(text[start : start + length])
    return prefixes


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def _say(quiet: bool, message: str) -> None:
    """Progress line on stderr unless ``quiet``."""
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _rate(count: float, seconds: float) -> float:
    """``count / seconds`` guarded against a zero-length interval."""
    return count / seconds if seconds > 0.0 else 0.0


def _warm_up(backend: str, device: str | None, seed: int) -> None:
    """Exercise the backend and the search once on a throwaway model.

    Removes one-off costs (module imports, device initialisation, kernel
    compilation) from the timed sections.
    """
    model = RadixNet(seed=seed, backend=backend, device=device)
    texts = synthetic_corpus(_WARMUP_CHARS, seed)
    model.train(texts, epochs=1)
    model.predict(texts[0][:3], length=PREDICT_LENGTH)


def run_benchmark(
    chars: int = DEFAULT_CHARS,
    epochs: int = DEFAULT_EPOCHS,
    backend: str = "python",
    device: str | None = None,
    seed: int = 0,
    quiet: bool = True,
    predictions: int | None = None,
) -> dict:
    """Train on a synthetic corpus of about ``chars`` characters, then predict.

    Returns a JSON-serialisable dict with the :data:`RESULT_KEYS`:

    * ``train_seconds`` - wall-clock time of one :meth:`RadixNet.train` call
      (observation + compression + ``epochs`` epochs);
    * ``transitions`` - transitions stepped through the backend over all
      epochs; ``transitions_per_sec = transitions / train_seconds``;
    * ``chars_per_sec = chars * epochs / train_seconds`` (every epoch
      consumes the whole corpus);
    * ``predict_count`` Dijkstra predictions of :data:`PREDICT_LENGTH`
      characters from corpus prefixes, timed as ``predict_seconds``;
      ``predictions_per_sec`` and ``dijkstra_expansions_per_sec`` follow;
    * ``nodes``, ``edges``, ``compression_ratio`` of the trained graph and a
      ``sample_prediction`` (prefix, continuation, cost, reached_end).

    Extra keys: ``seed``, ``trigrams``, ``mean_fanout``, ``loss_first``,
    ``loss_last``, ``epoch_seconds`` (sum of the per-epoch times, i.e. the
    backend share of ``train_seconds``), ``backend_transitions_per_sec``
    (``transitions / epoch_seconds``) and ``dijkstra_expansions``.

    ``predictions`` defaults to ``chars // 2`` clamped to
    ``[MIN_PREDICTIONS, MAX_PREDICTIONS]``.  A throwaway model is trained
    first so backend start-up costs are not timed.  ``quiet=False`` prints
    progress to stderr.
    """
    if chars < 1:
        raise ValueError(f"chars must be >= 1, got {chars}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if predictions is None:
        predictions = max(MIN_PREDICTIONS, min(MAX_PREDICTIONS, chars // 2))
    elif predictions < 1:
        raise ValueError(f"predictions must be >= 1, got {predictions}")

    texts = synthetic_corpus(chars, seed)
    total_chars = sum(len(t) for t in texts)
    _say(quiet, f"corpus: {len(texts)} texts, {total_chars} chars (seed {seed})")

    _warm_up(backend, device, seed)
    model = RadixNet(seed=seed, backend=backend, device=device)
    _say(quiet, f"backend: {model.backend.name} on {model.backend.device}; training {epochs} epoch(s)")

    def progress(record: dict) -> None:
        _say(
            quiet,
            f"  epoch {record['epoch']}: loss={record['loss']:.4f} nodes={record['nodes']} "
            f"edges={record['edges']} ratio={record['compression_ratio']:.2f} "
            f"transitions={record['transitions']} ({record['seconds']:.3f}s)",
        )

    t0 = time.perf_counter()
    records = model.train(texts, epochs=epochs, progress=progress if not quiet else None)
    train_seconds = time.perf_counter() - t0
    transitions = sum(r["transitions"] for r in records)
    epoch_seconds = sum(r["seconds"] for r in records)

    prefixes = prediction_prefixes(texts, predictions, seed)
    _say(quiet, f"predicting: {predictions} Dijkstra predictions of {PREDICT_LENGTH} chars")
    predict = model.predict
    predict(prefixes[0], length=PREDICT_LENGTH)  # warm the activation / cost caches
    t0 = time.perf_counter()
    first = predict(prefixes[0], length=PREDICT_LENGTH)
    expansions = first.expanded
    for prefix in prefixes[1:]:
        expansions += predict(prefix, length=PREDICT_LENGTH).expanded
    predict_seconds = time.perf_counter() - t0

    graph = model.graph
    nodes = graph.num_nodes()
    edges = graph.num_edges()
    return {
        "backend": model.backend.name,
        "device": model.backend.device,
        "chars": total_chars,
        "texts": len(texts),
        "epochs": epochs,
        "train_seconds": train_seconds,
        "transitions": transitions,
        "transitions_per_sec": _rate(transitions, train_seconds),
        "chars_per_sec": _rate(total_chars * epochs, train_seconds),
        "predict_count": predictions,
        "predict_seconds": predict_seconds,
        "predictions_per_sec": _rate(predictions, predict_seconds),
        "dijkstra_expansions_per_sec": _rate(expansions, predict_seconds),
        "nodes": nodes,
        "edges": edges,
        "compression_ratio": graph.compression_ratio(),
        "sample_prediction": {
            "prefix": prefixes[0],
            "continuation": first.text,
            "cost": first.cost,
            "reached_end": first.reached_end,
        },
        "seed": seed,
        "trigrams": graph.num_trigrams(),
        "mean_fanout": edges / max(1, nodes - 1),  # END has no out-edges
        "loss_first": records[0]["loss"],
        "loss_last": records[-1]["loss"],
        "epoch_seconds": epoch_seconds,
        "backend_transitions_per_sec": _rate(transitions, epoch_seconds),
        "dijkstra_expansions": expansions,
    }


# --------------------------------------------------------------------------
# reporting / entry point
# --------------------------------------------------------------------------


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if abs(value) >= 1000.0 else f"{value:.4f}"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def format_report(result: dict) -> str:
    """Readable two-column table of a :func:`run_benchmark` result.

    Nested dicts are shown with dotted keys (``sample_prediction.prefix``).
    """
    rows: list[tuple[str, str]] = []
    for key, value in result.items():
        if isinstance(value, dict):
            rows.extend((f"{key}.{sub}", _format_value(v)) for sub, v in value.items())
        else:
            rows.append((key, _format_value(value)))
    width = max((len(key) for key, _ in rows), default=0)
    return "\n".join(f"{key:<{width}}  {text}" for key, text in rows)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m radixnet.bench",
        description="Measure radixnet training and Dijkstra prediction throughput on a synthetic corpus.",
    )
    parser.add_argument("--chars", type=_positive_int, default=DEFAULT_CHARS,
                        help=f"characters of synthetic training text (default {DEFAULT_CHARS})")
    parser.add_argument("--epochs", type=_positive_int, default=DEFAULT_EPOCHS,
                        help=f"training epochs to time (default {DEFAULT_EPOCHS})")
    parser.add_argument("--backend", choices=("auto", "python", "torch"), default="python",
                        help="training backend (default python)")
    parser.add_argument("--device", default=None, help="torch device override (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the corpus, the model and the prefixes")
    parser.add_argument("--predictions", type=_positive_int, default=None,
                        help="number of timed predictions (default: chars // 2, clamped to "
                             f"{MIN_PREDICTIONS}..{MAX_PREDICTIONS})")
    parser.add_argument("--json", action="store_true", help="print the result as one JSON document")
    parser.add_argument("--verbose", action="store_true", help="print progress to stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        result = run_benchmark(
            chars=args.chars,
            epochs=args.epochs,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
            quiet=not args.verbose,
            predictions=args.predictions,
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
