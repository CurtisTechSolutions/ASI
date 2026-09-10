"""Command-line interface: ``python -m radixnet <command> [...]`` / ``radixnet``.

Global options - ``--model``, ``--backend``, ``--device``, ``--seed`` and
``--json`` - go before the command (they are accepted after it as well).
Every command prints human-readable tables by default and, with ``--json``,
exactly one JSON document on stdout; progress lines and notices then go to
stderr so stdout stays machine-readable.

Exit codes: ``0`` success, ``1`` error (one-line message on stderr: missing
files, bad arguments, library errors), ``130`` aborted by a second Ctrl-C.

``train``, ``2nrl`` and ``evolve`` run their work in a worker thread.  The
first Ctrl-C sets the library's ``stop_event`` so the current epoch or
generation completes and the model is saved; a second Ctrl-C aborts without
saving.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, TextIO, TypeVar

from . import __version__
from .checkpoint import CheckpointManager
from .gan import EvolveConfig, Evolver
from .model import RadixNet, TrainConfig

__all__ = ["main", "build_parser", "CliError", "EXIT_OK", "EXIT_ERROR", "EXIT_ABORTED"]

PROG = "radixnet"
DEFAULT_MODEL = "model.json"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_FRONTEND_DIR = os.path.join("frontend", "dist")
BACKENDS = ("auto", "python", "torch")
MODES = ("dijkstra", "sample")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ABORTED = 130

_T = TypeVar("_T")


class CliError(Exception):
    """A user-facing error: printed as one line on stderr, exit code 1."""


# errors turned into "radixnet: error: ..." + exit 1; genuine bugs still get a traceback
_HANDLED_ERRORS = (CliError, OSError, ValueError, TypeError, LookupError, ImportError, RuntimeError, ArithmeticError)


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def fmt(value: Any) -> str:
    """Render one cell: ``None`` -> ``-``, bools -> yes/no, floats to 4 digits."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        magnitude = abs(value)
        if magnitude and (magnitude >= 1e6 or magnitude < 1e-3):
            return f"{value:.4g}"
        return f"{value:.4f}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def quote(text: str) -> str:
    """Text in double quotes with whitespace escaped (JSON string syntax)."""
    return json.dumps(text, ensure_ascii=False)


def clip(text: str, width: int) -> str:
    """Single-line preview of ``text`` at most ``width`` characters long."""
    flat = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    return flat if len(flat) <= width else flat[: max(0, width - 3)] + "..."


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Fixed-width text table; numeric columns are right-aligned."""
    cells = [[fmt(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    right = [bool(rows) and all(row[i] is None or _is_number(row[i]) for row in rows) for i in range(len(headers))]

    def line(parts: Sequence[str]) -> str:
        return "  ".join(p.rjust(w) if r else p.ljust(w) for p, w, r in zip(parts, widths, right)).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths)] + [line(row) for row in cells])


def render_pairs(items: Sequence[tuple[str, Any]]) -> str:
    """Two-column key / value listing."""
    width = max((len(k) for k, _ in items), default=0)
    return "\n".join(f"{k.ljust(width)}  {fmt(v)}" for k, v in items)


def flatten(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Nested dicts -> ``[("a.b", value), ...]`` in insertion order."""
    if not isinstance(data, dict):
        return [(prefix or "value", data)]
    items: list[tuple[str, Any]] = []
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(flatten(value, name))
        else:
            items.append((name, value))
    return items


class Console:
    """Output channel honouring ``--json``.

    Human-readable text goes to stdout.  With ``--json`` stdout is reserved
    for the single final document (:meth:`emit`), so progress lines and
    notices go to stderr.  Everything is flushed immediately so progress is
    visible even when piped.
    """

    __slots__ = ("json_mode", "stdout", "stderr")

    def __init__(self, json_mode: bool, stdout: TextIO | None = None, stderr: TextIO | None = None) -> None:
        self.json_mode = json_mode
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr

    def say(self, text: str = "") -> None:
        """Human-mode line (suppressed with ``--json``)."""
        if not self.json_mode:
            print(text, file=self.stdout, flush=True)

    def note(self, text: str) -> None:
        """Notice on stderr (always shown)."""
        print(text, file=self.stderr, flush=True)

    def progress(self, human: str, compact: str) -> None:
        """Progress: table rows on stdout, or one compact line on stderr with ``--json``."""
        if self.json_mode:
            print(compact, file=self.stderr, flush=True)
        else:
            print(human, file=self.stdout, flush=True)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.say(render_table(headers, rows))

    def pairs(self, items: Sequence[tuple[str, Any]]) -> None:
        self.say(render_pairs(items))

    def emit(self, doc: Any) -> None:
        """The one JSON document of ``--json`` mode."""
        print(json.dumps(doc, indent=2, ensure_ascii=False, default=str), file=self.stdout, flush=True)


# --------------------------------------------------------------------------
# streaming progress printers (train / 2nrl / evolve)
# --------------------------------------------------------------------------

# (name, width, align); width 0 = free-form last column
_Column = tuple[str, int, str]

EPOCH_COLUMNS: tuple[_Column, ...] = (
    ("epoch", 5, ">"), ("loss", 9, ">"), ("ppl", 10, ">"), ("nodes", 7, ">"), ("edges", 7, ">"),
    ("trigrams", 8, ">"), ("ratio", 6, ">"), ("merges", 6, ">"), ("transitions", 11, ">"), ("seconds", 8, ">"),
)
PHASE_COLUMN: _Column = ("phase", 8, "<")
GENERATION_COLUMNS: tuple[_Column, ...] = (
    ("gen", 5, ">"), ("fake", 8, ">"), ("real", 8, ">"), ("gap", 8, ">"), ("gen_loss", 8, ">"),
    ("nodes", 7, ">"), ("edges", 7, ">"), ("ratio", 6, ">"), ("seconds", 8, ">"), ("sample", 0, "<"),
)
EPOCH_HEADERS = tuple(name for name, _, _ in EPOCH_COLUMNS)


def epoch_row(record: dict) -> list[Any]:
    """Table cells of one epoch record in :data:`EPOCH_COLUMNS` order."""
    return [
        record.get("epoch"), record.get("loss"), record.get("perplexity"), record.get("nodes"),
        record.get("edges"), record.get("trigrams"), record.get("compression_ratio"), record.get("merges"),
        record.get("transitions"), record.get("seconds"),
    ]


def _compact(prefix: str, pairs: Sequence[tuple[str, Any]]) -> str:
    return prefix + " " + " ".join(f"{k}={fmt(v)}" for k, v in pairs)


class _RowPrinter:
    """Streams fixed-width table rows through a :class:`Console` as records arrive.

    The header is printed together with the first row, so nothing appears
    when there are no records; with ``--json`` every record becomes one
    ``key=value`` line on stderr instead.
    """

    __slots__ = ("console", "columns", "seen", "_started")

    def __init__(self, console: Console, columns: Sequence[_Column]) -> None:
        self.console = console
        self.columns = tuple(columns)
        self.seen = 0
        self._started = False

    def _line(self, cells: Sequence[str]) -> str:
        parts = []
        for (_, width, align), cell in zip(self.columns, cells):
            if width == 0:
                parts.append(cell)
            else:
                parts.append(cell.rjust(width) if align == ">" else cell.ljust(width))
        return "  ".join(parts).rstrip()

    def _header(self) -> str:
        names = [name for name, _, _ in self.columns]
        rule = ["-" * (width or len(name)) for name, width, _ in self.columns]
        return self._line(names) + "\n" + self._line(rule)

    def row(self, cells: Sequence[Any], compact: str) -> None:
        text = self._line([fmt(c) for c in cells])
        if not self._started:
            self._started = True
            text = self._header() + "\n" + text
        self.seen += 1
        self.console.progress(text, compact)


class EpochPrinter(_RowPrinter):
    """``progress`` callback for :meth:`RadixNet.train` / :meth:`RadixNet.two_nrl`."""

    __slots__ = ("phased",)

    def __init__(self, console: Console, phased: bool = False) -> None:
        columns = (PHASE_COLUMN,) + EPOCH_COLUMNS if phased else EPOCH_COLUMNS
        super().__init__(console, columns)
        self.phased = phased

    def __call__(self, record: dict) -> None:
        cells = epoch_row(record)
        phase = record.get("phase")
        if self.phased:
            cells = [phase or "-"] + cells
        pairs = list(zip(EPOCH_HEADERS, epoch_row(record)))
        prefix = f"epoch {record.get('epoch')}" + (f" [{phase}]" if phase else "")
        self.row(cells, _compact(prefix, pairs[1:]))


class GenerationPrinter(_RowPrinter):
    """``progress`` callback for :meth:`Evolver.run`."""

    __slots__ = ()

    def __init__(self, console: Console) -> None:
        super().__init__(console, GENERATION_COLUMNS)

    def __call__(self, record: dict) -> None:
        values = [
            record.get("generation"), record.get("fake_score_mean"), record.get("real_score_mean"),
            record.get("gap"), record.get("gen_loss"), record.get("nodes"), record.get("edges"),
            record.get("compression_ratio"), record.get("seconds"),
        ]
        sample = quote(clip(str(record.get("sample", "")), 40))
        pairs = list(zip([c[0] for c in GENERATION_COLUMNS], values))
        self.row(values + [sample], _compact(f"generation {record.get('generation')}", pairs[1:] + [("sample", sample)]))


# --------------------------------------------------------------------------
# Ctrl-C handling for long-running work
# --------------------------------------------------------------------------


def _wait(done: threading.Event) -> None:
    # Event.wait with a timeout stays interruptible by Ctrl-C on every platform and,
    # unlike Thread.join(timeout) on Python 3.11, is not confused by the interrupt.
    while not done.wait(0.2):
        pass


def run_interruptible(
    work: Callable[[], _T], stop_event: threading.Event, console: Console, unit: str
) -> tuple[_T, bool]:
    """Run ``work`` in a worker thread; Ctrl-C asks it to stop cleanly.

    The first ``KeyboardInterrupt`` sets ``stop_event`` (checked by the
    library after every epoch / generation) and waits for the current
    ``unit`` to finish; a second one propagates so the caller aborts without
    saving.  Returns ``(result, interrupted)``; an exception raised by
    ``work`` is re-raised here.
    """
    outcome: list[tuple[bool, Any]] = []
    done = threading.Event()

    def target() -> None:
        try:
            outcome.append((True, work()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            outcome.append((False, exc))
        finally:
            done.set()

    worker = threading.Thread(target=target, name="radixnet-worker", daemon=True)
    worker.start()
    interrupted = False
    try:
        _wait(done)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        console.note(f"interrupted: finishing the current {unit}, then saving (Ctrl-C again aborts without saving)")
        _wait(done)
    worker.join()
    ok, value = outcome[0]
    if not ok:
        raise value
    return value, interrupted


# --------------------------------------------------------------------------
# shared command helpers
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Origin:
    """Where a command's model came from: ``new``, ``model`` (file) or ``checkpoint``."""

    kind: str
    path: str | None = None
    detail: str = ""

    def describe(self) -> str:
        if self.kind == "new":
            return f"new model ({self.detail})"
        return f"{self.kind} {self.path}" + (f" ({self.detail})" if self.detail else "")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path}


def effective_seed(args: argparse.Namespace) -> int:
    return 0 if args.seed is None else int(args.seed)


def backend_label(model: RadixNet) -> str:
    return f"{model.backend.name} ({model.backend.device})"


def read_text_file(path: str) -> str:
    """UTF-8 file content; a missing or undecodable file is a :class:`CliError`."""
    if not os.path.isfile(path):
        raise CliError(f"data file not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise CliError(f"{path} is not valid UTF-8 text: {exc}") from exc


def read_texts(paths: Sequence[str], whole_file: bool = False, what: str = "training") -> list[str]:
    """Texts from files: one per non-blank line, or one per file with ``whole_file``.

    Lines are kept verbatim (only blank ones are dropped); in whole-file mode
    trailing line terminators are stripped.  Raises :class:`CliError` when
    nothing usable was found.
    """
    texts: list[str] = []
    for path in paths:
        content = read_text_file(path)
        if whole_file:
            text = content.rstrip("\r\n")
            if text:
                texts.append(text)
        else:
            texts.extend(line for line in content.split("\n") if line.strip())
    if not texts:
        raise CliError(f"no {what} texts found in {', '.join(paths)}")
    return texts


def open_model(
    args: argparse.Namespace, console: Console, *, required: bool, manager: CheckpointManager | None = None
) -> tuple[RadixNet, Origin]:
    """Load ``--model`` (or the latest checkpoint of ``manager``), else create a fresh model.

    ``required`` turns a missing model file into an error (inference and
    maintenance commands); training commands start from a new model instead.
    """
    backend, device = args.backend, args.device
    if manager is not None:
        record = manager.latest()
        if record is not None:
            model = RadixNet.load(record["path"], backend=backend, device=device)
            return model, Origin("checkpoint", record["path"], f"{model.meta['epochs_total']} epochs trained")
        console.note(f"note: no checkpoint to resume from in {manager.directory}")
    path = args.model
    if os.path.isfile(path):
        model = RadixNet.load(path, backend=backend, device=device)
        return model, Origin("model", path, f"{model.meta['epochs_total']} epochs trained")
    if required:
        raise CliError(f"model file not found: {path} (train one first with `{PROG} train --data FILE`)")
    seed = effective_seed(args)
    return RadixNet(seed=seed, backend=backend, device=device), Origin("new", None, f"seed {seed}")


def save_model(model: RadixNet, path: str) -> dict:
    model.save(path)
    return {"path": path, "bytes": os.path.getsize(path)}


def checkpoint_manager(args: argparse.Namespace) -> CheckpointManager | None:
    if not args.checkpoint_dir:
        return None
    return CheckpointManager(args.checkpoint_dir, keep=args.keep)


def checkpoint_every(args: argparse.Namespace, manager: CheckpointManager | None) -> int:
    """``--checkpoint-every`` with its default: every step when a directory is given, else off."""
    if args.checkpoint_every is None:
        return 1 if manager is not None else 0
    if args.checkpoint_every and manager is None:
        raise CliError("--checkpoint-every requires --checkpoint-dir")
    return args.checkpoint_every


def _resolve_frontend_dir(path: str | None) -> str:
    """``--frontend-dir``: the given path, else ``frontend/dist`` here or next to the package."""
    if path is not None:
        return path
    if os.path.isdir(DEFAULT_FRONTEND_DIR):
        return DEFAULT_FRONTEND_DIR
    packaged = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
    return packaged if os.path.isdir(packaged) else DEFAULT_FRONTEND_DIR


def _finish_training(console: Console, model: RadixNet, out: str, interrupted: bool, printer: _RowPrinter, unit: str) -> dict:
    """Common tail of train / 2nrl / evolve: note an early stop, save, report."""
    if interrupted:
        console.note(f"stopped after {printer.seen} {unit}(s); saving")
    saved = save_model(model, out)
    console.say()
    console.say(f"saved {saved['path']} ({saved['bytes']} bytes)")
    return saved


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace, console: Console) -> dict:
    texts = read_texts(args.data, args.whole_file)
    manager = checkpoint_manager(args)
    if args.resume and manager is None:
        raise CliError("--resume requires --checkpoint-dir")
    every = checkpoint_every(args, manager)
    model, origin = open_model(args, console, required=False, manager=manager if args.resume else None)
    config = TrainConfig(
        epochs=args.epochs, lr=args.lr, act_lr=args.act_lr, batch_size=args.batch_size,
        auto_compress=not args.no_compress, checkpoint_every=every,
    )
    out = args.out or args.model
    chars = sum(len(t) for t in texts)
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("data", f"{len(texts)} texts, {chars} chars from {', '.join(args.data)}"),
        ("config", f"epochs={config.epochs} lr={config.lr} act_lr={config.act_lr} batch={config.batch_size} "
                   f"compress={'yes' if config.auto_compress else 'no'}"),
        ("checkpoints", f"{manager.directory} every {every} epoch(s), keep {manager.keep}" if manager else "off"),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console)
    stop = threading.Event()
    records, interrupted = run_interruptible(
        lambda: model.train(texts, config, checkpoint_manager=manager, progress=printer, stop_event=stop),
        stop, console, "epoch",
    )
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    checkpoints = manager.list() if manager else []
    if manager:
        console.say(f"{len(checkpoints)} checkpoint(s) in {manager.directory}")
    return {
        "model": origin.to_dict(),
        "out": out,
        "texts": len(texts),
        "chars": chars,
        "config": config.to_dict(),
        "records": records,
        "interrupted": interrupted,
        "saved": saved,
        "checkpoint_dir": manager.directory if manager else None,
        "checkpoints": checkpoints,
        "stats": model.stats(),
    }


def cmd_predict(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    result = model.predict(
        args.prefix, length=args.length, mode=args.mode, step_penalty=args.step_penalty,
        temperature=args.temperature, to_end=args.to_end, max_length=args.max_length,
    )
    console.pairs([
        ("prefix", quote(args.prefix)),
        ("continuation", quote(result.text)),
        ("full text", quote(result.full_text)),
        ("cost", result.cost),
        ("reached end", result.reached_end),
        ("expanded", result.expanded),
        ("mode", args.mode),
    ])
    console.say()
    console.say("path:")
    offset = len(result.labels) - len(result.step_costs)
    rows = [
        [i, node, result.step_costs[i - offset] if i >= offset else None, quote(label)]
        for i, (node, label) in enumerate(zip(result.node_ids, result.labels))
    ]
    console.table(("step", "node", "cost", "label"), rows)
    return {
        "prefix": args.prefix,
        "continuation": result.text,
        "full_text": result.full_text,
        "cost": result.cost,
        "step_costs": list(result.step_costs),
        "path": list(result.labels),
        "node_ids": list(result.node_ids),
        "expanded": result.expanded,
        "reached_end": result.reached_end,
        "mode": args.mode,
    }


def cmd_generate(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    results = model.generate(
        max_length=args.max_length, mode=args.mode, temperature=args.temperature, count=args.count, seed=args.seed,
    )
    rows = [[i + 1, r.cost, r.reached_end, quote(clip(r.text, 100))] for i, r in enumerate(results)]
    console.table(("#", "cost", "end", "text"), rows)
    return {
        "samples": [r.to_dict() for r in results],
        "count": len(results),
        "mode": args.mode,
        "max_length": args.max_length,
        "temperature": args.temperature,
    }


def cmd_score(args: argparse.Namespace, console: Console) -> dict:
    texts = [args.text] if args.text is not None else read_texts([args.data], what="scorable")
    model, _ = open_model(args, console, required=True)
    results = [{"text": t, **model.score(t)} for t in texts]
    mean_log_prob = sum(r["log_prob"] for r in results) / len(results)
    mean_per_char = sum(r["per_char"] for r in results) / len(results)
    rows = [
        [r["log_prob"], r["per_char"], r["chars"], r["transitions"], r["unknown_transitions"], quote(clip(r["text"], 60))]
        for r in results
    ]
    console.table(("log_prob", "per_char", "chars", "transitions", "unknown", "text"), rows)
    if len(results) > 1:
        console.say()
        console.say(f"{len(results)} texts: mean log_prob {fmt(mean_log_prob)}, mean per_char {fmt(mean_per_char)}")
    return {"results": results, "count": len(results), "mean_log_prob": mean_log_prob, "mean_per_char": mean_per_char}


def cmd_two_nrl(args: argparse.Namespace, console: Console) -> dict:
    bad = read_texts([args.bad], what="bad")
    good = read_texts([args.good], what="good")
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("bad", f"{len(bad)} texts from {args.bad}"),
        ("good", f"{len(good)} texts from {args.good}"),
        ("negative", f"epochs={args.neg_epochs} lr={args.neg_lr}"),
        ("positive", f"epochs={args.pos_epochs} lr={args.pos_lr} act_lr={args.pos_lr / 10}"),
        ("batch", args.batch_size),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console, phased=True)
    stop = threading.Event()
    result, interrupted = run_interruptible(
        lambda: model.two_nrl(
            bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
            pos_lr=args.pos_lr, progress=printer, stop_event=stop, batch_size=args.batch_size,
        ),
        stop, console, "epoch",
    )
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    console.say(f"inverted: {fmt(result['inverted'])}")
    return {
        "model": origin.to_dict(),
        "out": out,
        "bad_texts": len(bad),
        "good_texts": len(good),
        "negative": result["negative"],
        "positive": result["positive"],
        "inverted": result["inverted"],
        "interrupted": interrupted,
        "saved": saved,
        "stats": model.stats(),
    }


def cmd_invert(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    was = model.graph.inverted
    model.invert()
    out = args.out or args.model
    saved = save_model(model, out)
    console.pairs([("inverted", f"{fmt(was)} -> {fmt(model.graph.inverted)}"), ("saved", f"{out} ({saved['bytes']} bytes)")])
    return {"inverted": model.graph.inverted, "was_inverted": was, "out": out, "saved": saved, "stats": model.stats()}


def cmd_compress(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    before = model.stats()
    merges = model.compress()
    out = args.out or args.model
    saved = save_model(model, out)
    stats = model.stats()
    console.pairs([
        ("merges", merges),
        ("nodes", f"{before['nodes']} -> {stats['nodes']}"),
        ("edges", f"{before['edges']} -> {stats['edges']}"),
        ("compression ratio", f"{fmt(before['compression_ratio'])} -> {fmt(stats['compression_ratio'])}"),
        ("saved", f"{out} ({saved['bytes']} bytes)"),
    ])
    return {
        "merges": merges,
        "nodes_before": before["nodes"],
        "nodes_after": stats["nodes"],
        "edges_before": before["edges"],
        "edges_after": stats["edges"],
        "out": out,
        "saved": saved,
        "stats": stats,
    }


def cmd_evolve(args: argparse.Namespace, console: Console) -> dict:
    corpus = read_texts([args.data], what="corpus")
    manager = checkpoint_manager(args)
    every = checkpoint_every(args, manager)
    generator, origin = open_model(args, console, required=False)
    discriminator: RadixNet | None = None
    disc_origin: Origin | None = None
    if args.discriminator:
        if os.path.isfile(args.discriminator):
            discriminator = RadixNet.load(args.discriminator, backend=args.backend, device=args.device)
            disc_origin = Origin("model", args.discriminator, f"{discriminator.meta['epochs_total']} epochs trained")
        else:
            disc_origin = Origin("new", None, f"seed {effective_seed(args) + 1}, saved to {args.discriminator}")
    config = EvolveConfig(
        samples=args.samples, real_per_generation=args.real_per_generation, max_length=args.max_length,
        temperature=args.temperature, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs,
        neg_lr=args.neg_lr, pos_lr=args.pos_lr, disc_neg_epochs=args.disc_neg_epochs,
        disc_pos_epochs=args.disc_pos_epochs, batch_size=args.batch_size, checkpoint_every=every,
        seed=effective_seed(args),
    )
    evolver = Evolver(generator, corpus, discriminator, config)
    generations = args.generations or None
    out = args.out or args.model
    console.pairs([
        ("generator", origin.describe()),
        ("discriminator", disc_origin.describe() if disc_origin else "fresh (not saved)"),
        ("backend", backend_label(generator)),
        ("corpus", f"{len(evolver.corpus)} texts from {args.data}"),
        ("generations", "until Ctrl-C" if generations is None else generations),
        ("config", f"samples={config.samples} real={config.real_per_generation} max_length={config.max_length} "
                   f"temperature={config.temperature} batch={config.batch_size}"),
        ("checkpoints", f"{manager.directory} every {every} generation(s), keep {manager.keep}" if manager else "off"),
        ("output", out),
    ])
    console.say()
    printer = GenerationPrinter(console)
    stop = threading.Event()
    records, interrupted = run_interruptible(
        lambda: evolver.run(generations, checkpoint_manager=manager, progress=printer, stop_event=stop),
        stop, console, "generation",
    )
    saved = _finish_training(console, generator, out, interrupted, printer, "generation")
    disc_saved = None
    if args.discriminator:
        disc_saved = save_model(evolver.discriminator, args.discriminator)
        console.say(f"saved discriminator {args.discriminator} ({disc_saved['bytes']} bytes)")
    return {
        "model": origin.to_dict(),
        "out": out,
        "corpus_texts": len(evolver.corpus),
        "config": dataclasses.asdict(config),
        "generations": records,
        "generation": evolver.generation,
        "interrupted": interrupted,
        "saved": saved,
        "discriminator": {"path": args.discriminator, "saved": disc_saved, "stats": evolver.discriminator.stats()}
        if args.discriminator else None,
        "checkpoint_dir": manager.directory if manager else None,
        "stats": generator.stats(),
    }


def cmd_info(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    stats = model.stats()
    meta = dict(model.meta)
    tail = model.history[-args.tail:] if args.tail > 0 else []
    console.pairs([
        ("model", args.model),
        ("backend", backend_label(model)),
        ("nodes", stats["nodes"]),
        ("edges", stats["edges"]),
        ("trigrams", stats["trigrams"]),
        ("compression ratio", stats["compression_ratio"]),
        ("inverted", stats["inverted"]),
        ("epochs total", stats["epochs_total"]),
        ("trained", f"{stats['trained_texts']} texts, {stats['trained_chars']} chars"),
        ("2NRL runs", stats["twonrl_runs"]),
        ("last loss", stats["last_loss"]),
        ("seed", meta.get("seed")),
        ("created", meta.get("created")),
    ])
    if tail:
        console.say()
        console.say(f"last {len(tail)} of {len(model.history)} epoch record(s):")
        console.table(("phase",) + EPOCH_HEADERS, [[r.get("phase") or "-"] + epoch_row(r) for r in tail])
    return {"model": args.model, "stats": stats, "meta": meta, "history_len": len(model.history), "history_tail": tail}


def cmd_checkpoints(args: argparse.Namespace, console: Console) -> dict:
    if not os.path.isdir(args.dir):
        raise CliError(f"checkpoint directory not found: {args.dir}")
    manager = CheckpointManager(args.dir)
    records = manager.list()
    latest = manager.latest()
    latest_name = latest["name"] if latest else None
    if args.restore is None:
        if args.out:
            raise CliError("--out requires --restore NAME")
        console.say(f"{len(records)} checkpoint(s) in {manager.directory}")
        if records:
            console.table(
                ("name", "tag", "step", "saved_at", "bytes", "loss", "latest"),
                [
                    [r["name"], r["tag"], r["step"], r.get("saved_at"), r.get("bytes"),
                     (r.get("metrics") or {}).get("loss"), "*" if r["name"] == latest_name else ""]
                    for r in records
                ],
            )
        return {"directory": manager.directory, "checkpoints": records, "latest": latest}
    if args.restore == "latest":
        if latest is None:
            raise CliError(f"no latest checkpoint in {manager.directory}")
        record = latest
    else:
        try:
            path = manager.resolve(args.restore)
        except FileNotFoundError as exc:
            raise CliError(str(exc)) from None
        record = next((r for r in records if r["path"] == path), None) or {"name": os.path.basename(path), "path": path}
    model = RadixNet.load(record["path"], backend=args.backend, device=args.device)
    out = args.out or args.model
    saved = save_model(model, out)
    stats = model.stats()
    console.pairs([
        ("restored", record["name"]),
        ("saved", f"{out} ({saved['bytes']} bytes)"),
        ("nodes", stats["nodes"]),
        ("edges", stats["edges"]),
        ("epochs total", stats["epochs_total"]),
    ])
    return {"directory": manager.directory, "restored": record, "out": out, "saved": saved, "stats": stats}


def cmd_bench(args: argparse.Namespace, console: Console) -> dict:
    try:
        from . import bench
    except ImportError as exc:
        raise CliError(f"benchmark module unavailable: {exc}") from exc
    console.say(f"benchmark: {args.chars} chars, {args.epochs} epoch(s), backend {args.backend}"
                + (f" on {args.device}" if args.device else ""))
    result = bench.run_benchmark(chars=args.chars, epochs=args.epochs, backend=args.backend, device=args.device)
    console.say()
    console.pairs(flatten(result))
    return result


def cmd_serve(args: argparse.Namespace, console: Console) -> None:
    try:
        from . import api
    except ImportError as exc:
        raise CliError(f"API module unavailable: {exc}") from exc
    frontend_dir = _resolve_frontend_dir(args.frontend_dir)
    doc = {
        "host": args.host,
        "port": args.port,
        "url": f"http://{args.host}:{args.port}/",
        "model": args.model,
        "model_exists": os.path.isfile(args.model),
        "checkpoint_dir": args.checkpoint_dir,
        "frontend_dir": frontend_dir,
        "frontend_built": os.path.isfile(os.path.join(frontend_dir, "index.html")),
        "backend": args.backend,
        "device": args.device,
        "seed": effective_seed(args),
        "version": __version__,
    }
    console.pairs([
        ("serving", doc["url"]),
        ("model", args.model + ("" if doc["model_exists"] else " (new; created on first save)")),
        ("checkpoints", args.checkpoint_dir),
        ("frontend", frontend_dir + ("" if doc["frontend_built"] else " (not built: run `npm install && npm run build` in frontend/)")),
        ("backend", args.backend + (f" on {args.device}" if args.device else "")),
    ])
    console.say("press Ctrl-C to stop")
    if console.json_mode:
        console.emit(doc)  # the server blocks, so the document goes out first
    api.run_server(
        host=args.host, port=args.port, model_path=args.model, checkpoint_dir=args.checkpoint_dir,
        frontend_dir=frontend_dir, backend=args.backend, device=args.device, seed=effective_seed(args),
    )
    return None


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit with code 1 (not 2)."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults (except None / False ones) and keep description line breaks."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        default = action.default
        if default is None or default is False or default is argparse.SUPPRESS or not action.option_strings:
            return action.help
        return super()._get_help_string(action)


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value

    return parse


def _float_at_least(minimum: float) -> Callable[[str], float]:
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
        if not math.isfinite(value) or value < minimum:
            raise argparse.ArgumentTypeError(f"must be a finite number >= {minimum:g}, got {text}")
        return value

    return parse


nonneg_int = _int_at_least(0)
pos_int = _int_at_least(1)
nonneg_float = _float_at_least(0.0)


def _add_global_options(parser: argparse.ArgumentParser, top_level: bool) -> None:
    """Global options; sub-command copies use SUPPRESS defaults so they never override the top level."""

    def default(value: Any) -> Any:
        return value if top_level else argparse.SUPPRESS

    group = parser.add_argument_group("global options")
    group.add_argument("--model", metavar="PATH", default=default(DEFAULT_MODEL),
                       help="model file to load / save (gzip when the name ends with .gz)")
    group.add_argument("--backend", choices=BACKENDS, default=default("auto"),
                       help="training backend; auto = torch when a GPU (cuda / mps) is available, else python")
    group.add_argument("--device", metavar="DEV", default=default(None),
                       help="backend device (cpu, cuda, mps); default: the backend's own choice")
    group.add_argument("--seed", type=int, metavar="N", default=default(None),
                       help="RNG seed for a newly created model (default 0) and for `generate` sampling")
    group.add_argument("--json", action="store_true", default=default(False),
                       help="print one JSON document on stdout instead of tables (progress goes to stderr)")


def _add_checkpoint_options(parser: argparse.ArgumentParser, unit: str) -> None:
    group = parser.add_argument_group("checkpoint options")
    group.add_argument("--checkpoint-dir", metavar="DIR", help="write rotated checkpoints into DIR")
    group.add_argument("--checkpoint-every", type=nonneg_int, metavar="N",
                       help=f"checkpoint every N {unit}s (default: every {unit} when --checkpoint-dir is given, else off)")
    group.add_argument("--keep", type=pos_int, default=5, help="checkpoints to keep in --checkpoint-dir")


def _add_two_nrl_options(parser: argparse.ArgumentParser, neg_epochs: int, pos_epochs: int, batch_size: int) -> None:
    group = parser.add_argument_group("2NRL options")
    group.add_argument("--neg-epochs", type=nonneg_int, default=neg_epochs, help="epochs of the negative (garbage) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=pos_epochs, help="epochs of the positive (fine-tune) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.05, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.01,
                       help="learning rate of the positive phase (activation parameters use a tenth of it)")
    group.add_argument("--batch-size", type=pos_int, default=batch_size, help="transitions per backend step in both phases")


def build_parser() -> argparse.ArgumentParser:
    """The complete argument parser (``main`` uses it; useful for docs and tests)."""
    parser = _Parser(
        prog=PROG,
        description="RadixCyclicNN - a self-compressing cyclic-graph neural network (radix-tree idea) "
                    "with a sine activation, Dijkstra prediction, 2NRL and GAN-style self-upgrading.",
        epilog=(
            "examples:\n"
            f"  {PROG} train --data data/sample_corpus.txt --epochs 5 --checkpoint-dir checkpoints\n"
            f"  {PROG} predict --prefix 'the quick brown' --length 20\n"
            f"  {PROG} 2nrl --bad data/sample_garbage.txt --good data/sample_corpus.txt\n"
            f"  {PROG} evolve --data data/sample_corpus.txt --generations 0   # Ctrl-C stops and saves\n"
            f"  {PROG} --json info\n"
            f"  {PROG} serve --port 8000\n"
            "\nglobal options go before the command (they are accepted after it as well)."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    _add_global_options(parser, top_level=True)
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands", required=True)

    def command(name: str, help_text: str, description: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=description, formatter_class=_HelpFormatter)
        _add_global_options(p, top_level=False)
        return p

    # train ----------------------------------------------------------------
    p = command(
        "train", "train on text files (incremental: --model is loaded first when it exists)",
        "Train the network.  The model file is loaded first when it exists, so repeated runs keep\n"
        "learning; per-epoch loss / perplexity / size rows are printed as each epoch finishes.\n"
        "Small corpora learn visibly only with small batches and a larger learning rate\n"
        "(e.g. --batch-size 1..8 --lr 0.5..1.0).  Ctrl-C stops after the current epoch and saves.",
    )
    p.add_argument("--data", nargs="+", required=True, metavar="FILE",
                   help="training text files, one text per line (blank lines are skipped)")
    p.add_argument("--whole-file", action="store_true", help="treat each file as a single text")
    p.add_argument("--epochs", type=nonneg_int, default=TrainConfig.epochs, help="training epochs")
    p.add_argument("--lr", type=nonneg_float, default=TrainConfig.lr, help="learning rate for edge weights and node states")
    p.add_argument("--act-lr", type=nonneg_float, default=TrainConfig.act_lr,
                   help="learning rate for the activation parameters a, b, h, k")
    p.add_argument("--batch-size", type=pos_int, default=TrainConfig.batch_size, help="transitions per backend step")
    p.add_argument("--no-compress", action="store_true", help="do not merge unary chains after each epoch")
    _add_checkpoint_options(p, "epoch")
    p.add_argument("--resume", action="store_true",
                   help="start from the latest checkpoint in --checkpoint-dir (falls back to --model)")
    p.add_argument("--out", metavar="PATH", help="where to save the trained model (default: --model)")
    p.set_defaults(handler=cmd_train)

    # predict --------------------------------------------------------------
    p = command(
        "predict", "continue a prefix (shortest path or sampling)",
        "Continue --prefix.  dijkstra returns the cheapest path (cost = -log P per edge + step penalty)\n"
        "emitting at least --length characters or reaching the end; sample walks stochastically.",
    )
    p.add_argument("--prefix", required=True, metavar="TEXT", help="text to continue")
    p.add_argument("--length", type=nonneg_int, default=20, help="characters to emit (dijkstra: minimum)")
    p.add_argument("--max-length", type=nonneg_int, metavar="N", help="hard cap on emitted characters (default: 2 x --length)")
    p.add_argument("--mode", choices=MODES, default="dijkstra", help="search strategy")
    p.add_argument("--to-end", action="store_true", help="dijkstra: cheapest path all the way to the end of a text")
    p.add_argument("--step-penalty", type=nonneg_float, default=0.0, help="dijkstra: extra cost per edge (prefers short paths)")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="sample: softmax temperature (0 = greedy)")
    p.set_defaults(handler=cmd_predict)

    # generate -------------------------------------------------------------
    p = command(
        "generate", "generate texts from scratch",
        "Generate texts from the start node: sample draws --count stochastic walks (--seed makes them\n"
        "reproducible); dijkstra returns the single cheapest complete text.",
    )
    p.add_argument("--count", type=nonneg_int, default=1, help="number of samples (sample mode)")
    p.add_argument("--max-length", type=nonneg_int, default=60, help="maximum characters per sample")
    p.add_argument("--mode", choices=MODES, default="sample", help="generation strategy")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="softmax temperature (0 = greedy)")
    p.set_defaults(handler=cmd_generate)

    # score ----------------------------------------------------------------
    p = command(
        "score", "log-probability of texts",
        "Log-probability of one text (--text) or of every line of a file (--data): unknown trigrams\n"
        "and missing edges are charged log(1e-6) and counted as unknown transitions.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", metavar="TEXT", help="a single text")
    source.add_argument("--data", metavar="FILE", help="file with one text per line")
    p.set_defaults(handler=cmd_score)

    # 2nrl -----------------------------------------------------------------
    p = command(
        "2nrl", "2NRL: train on garbage, invert, fine-tune on correct data",
        "2NRL: (1) train on --bad, (2) invert the network (all edge weights and activation amplitudes\n"
        "flip sign, so what was likely becomes unlikely), (3) fine-tune on --good with a smaller\n"
        "learning rate.  A missing --model starts from a new network.",
    )
    p.add_argument("--bad", required=True, metavar="FILE", help="garbage / wrong texts, one per line")
    p.add_argument("--good", required=True, metavar="FILE", help="correct texts, one per line")
    _add_two_nrl_options(p, neg_epochs=3, pos_epochs=3, batch_size=TrainConfig.batch_size)
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_two_nrl)

    # invert / compress ----------------------------------------------------
    p = command("invert", "invert the network and save", "Flip every edge weight and activation amplitude, then save.")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_invert)

    p = command("compress", "merge unary chains and save",
                "Merge every unary chain (radix-tree path compression) and save; predictions are unchanged.")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_compress)

    # evolve ---------------------------------------------------------------
    p = command(
        "evolve", "GAN-style self-upgrading loop (generator vs discriminator)",
        "Constantly self-upgrade: each generation the model generates samples, a discriminator\n"
        "learns real-vs-fake via 2NRL, the worst fakes become the model's own 2NRL garbage and real\n"
        "corpus samples are its fine-tune pass.  --generations 0 runs until Ctrl-C, which stops after\n"
        "the current generation and saves the model (and --discriminator when given).",
    )
    p.add_argument("--data", required=True, metavar="FILE", help="real corpus, one text per line")
    p.add_argument("--generations", type=nonneg_int, default=0, metavar="N", help="generations to run (0 = forever)")
    p.add_argument("--samples", type=pos_int, default=EvolveConfig.samples, help="fakes generated per generation")
    p.add_argument("--real-per-generation", type=pos_int, default=EvolveConfig.real_per_generation,
                   help="real corpus texts used per generation")
    p.add_argument("--max-length", type=pos_int, default=EvolveConfig.max_length, help="maximum characters per fake")
    p.add_argument("--temperature", type=nonneg_float, default=EvolveConfig.temperature, help="sampling temperature")
    p.add_argument("--discriminator", metavar="PATH", help="discriminator model file (loaded when it exists, saved at the end)")
    p.add_argument("--disc-neg-epochs", type=nonneg_int, default=EvolveConfig.disc_neg_epochs,
                   help="discriminator negative-phase epochs")
    p.add_argument("--disc-pos-epochs", type=nonneg_int, default=EvolveConfig.disc_pos_epochs,
                   help="discriminator positive-phase epochs")
    _add_two_nrl_options(p, neg_epochs=EvolveConfig.neg_epochs, pos_epochs=EvolveConfig.pos_epochs,
                         batch_size=EvolveConfig.batch_size)
    _add_checkpoint_options(p, "generation")
    p.add_argument("--out", metavar="PATH", help="where to save the generator (default: --model)")
    p.set_defaults(handler=cmd_evolve)

    # info -----------------------------------------------------------------
    p = command("info", "model statistics and training history tail", "Print the model's statistics and its last epoch records.")
    p.add_argument("--tail", type=nonneg_int, default=10, help="history records to show")
    p.set_defaults(handler=cmd_info)

    # checkpoints ----------------------------------------------------------
    p = command(
        "checkpoints", "list checkpoints or restore one",
        "List the checkpoints of a directory, or restore one into a model file:\n"
        "--restore NAME accepts a checkpoint name, its bare stem, a path, or `latest`.",
    )
    p.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, metavar="DIR", help="checkpoint directory")
    p.add_argument("--restore", metavar="NAME", help="checkpoint to restore")
    p.add_argument("--out", metavar="PATH", help="where to write the restored model (default: --model)")
    p.set_defaults(handler=cmd_checkpoints)

    # bench ----------------------------------------------------------------
    p = command(
        "bench", "throughput benchmark",
        "Measure training transitions/sec and chars/sec, predictions/sec and Dijkstra expansions/sec\n"
        "on a synthetic corpus with the selected backend.",
    )
    p.add_argument("--chars", type=pos_int, default=20000, help="characters of synthetic training text")
    p.add_argument("--epochs", type=pos_int, default=3, help="training epochs to time")
    p.set_defaults(handler=cmd_bench)

    # serve ----------------------------------------------------------------
    p = command(
        "serve", "start the HTTP JSON API (and serve the React frontend)",
        "Start the API server.  --model is loaded when it exists (a new model otherwise) and the\n"
        "built frontend is served from --frontend-dir when present.  Ctrl-C stops the server.",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address")
    p.add_argument("--port", type=nonneg_int, default=8000, help="TCP port")
    p.add_argument("--frontend-dir", metavar="DIR",
                   help="built frontend directory (default: frontend/dist here, or the one next to the package)")
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, metavar="DIR", help="checkpoint directory of the API")
    p.set_defaults(handler=cmd_serve)
    return parser


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _configure_streams() -> None:
    """Never crash on characters the console cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and getattr(stream, "errors", None) == "strict":
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def _describe(exc: BaseException) -> str:
    if isinstance(exc, CliError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the exit code (never raises ``SystemExit`` itself)."""
    _configure_streams()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help / --version (0) or a usage error (1)
        return exc.code if isinstance(exc.code, int) else (EXIT_OK if exc.code is None else EXIT_ERROR)
    console = Console(bool(args.json))
    try:
        doc = args.handler(args, console)
    except KeyboardInterrupt:
        console.note(f"{PROG}: aborted")
        return EXIT_ABORTED
    except _HANDLED_ERRORS as exc:
        console.note(f"{PROG}: error: {_describe(exc)}")
        return EXIT_ERROR
    if console.json_mode and doc is not None:
        console.emit(doc)
    return EXIT_OK
