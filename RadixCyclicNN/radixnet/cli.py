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
    ("epoch", 5, ">"), ("loss", 9, ">"), ("ppl", 10, ">"), ("lr", 7, ">"), ("act_lr", 7, ">"), ("nodes", 7, ">"),
    ("edges", 7, ">"), ("trigrams", 8, ">"), ("ratio", 6, ">"), ("merges", 6, ">"), ("transitions", 11, ">"),
    ("seconds", 8, ">"),
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
        record.get("epoch"), record.get("loss"), record.get("perplexity"), record.get("lr"), record.get("act_lr"),
        record.get("nodes"), record.get("edges"), record.get("trigrams"), record.get("compression_ratio"),
        record.get("merges"), record.get("transitions"), record.get("seconds"),
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


PROBLEM_COLUMNS: tuple[_Column, ...] = (
    ("phase", 7, "<"), ("round", 5, ">"), ("problem", 12, "<"), ("attempts", 8, ">"), ("result", 7, "<"),
    ("by", 6, "<"), ("action", 6, "<"), ("neg_loss", 8, ">"), ("pos_loss", 8, ">"), ("seconds", 8, ">"),
)


class ProblemPrinter(_RowPrinter):
    """``progress`` callback for :meth:`CodeGenTrainer.run`: attempts as notes, problems as rows."""

    __slots__ = ("attempts",)

    def __init__(self, console: Console) -> None:
        super().__init__(console, PROBLEM_COLUMNS)
        self.attempts = 0

    def __call__(self, record: dict) -> None:
        kind = record.get("kind")
        if kind == "attempt":
            self.attempts += 1
            status = "correct" if record.get("correct") else ("runs but rejected" if record.get("runs") else "error")
            detail = record.get("error") or (record.get("issues") or [""])[0]
            score = f" score={fmt(record.get('score'))}" if record.get("score") is not None else ""
            self.console.note(
                f"    {record.get('problem')} attempt {record.get('attempt')} [{record.get('source')}] {status}{score}"
                + (f": {clip(str(detail), 80)}" if detail else "")
            )
        elif kind == "problem":
            values = [
                record.get("phase"), record.get("round"), clip(str(record.get("problem")), 12), record.get("attempts"),
                "correct" if record.get("correct") else "failed", record.get("solved_by") or "-",
                record.get("action") or "-", record.get("neg_loss"), record.get("pos_loss"), record.get("seconds"),
            ]
            pairs = list(zip([c[0] for c in PROBLEM_COLUMNS], values))
            self.row(values, _compact(f"problem {record.get('problem')}", pairs))
        elif kind == "round":
            learned = f", 2NRL per round: {record.get('action')}" if record.get("action") else ""
            self.console.note(
                f"round {record.get('round')} [{record.get('phase')}]: {record.get('solved')}/{record.get('problems')} solved, "
                f"{record.get('model_solved')} by the model{learned}"
            )


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
        lr_schedule=args.lr_schedule, act_lr_schedule=args.act_lr_schedule,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    out = args.out or args.model
    chars = sum(len(t) for t in texts)
    schedule_note = ""
    if config.lr_schedule or config.act_lr_schedule:
        rates = config.rates()
        schedule_note = (
            f" lr_schedule={config.lr_schedule or '-'} act_lr_schedule={config.act_lr_schedule or '-'}"
            + (f" (lr {fmt(rates[0][0])} -> {fmt(rates[-1][0])}, act_lr {fmt(rates[0][1])} -> {fmt(rates[-1][1])})" if rates else "")
        )
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("data", f"{len(texts)} texts, {chars} chars from {', '.join(args.data)}"),
        ("config", f"epochs={config.epochs} lr={config.lr} act_lr={config.act_lr} batch={config.batch_size} "
                   f"compress={'yes' if config.auto_compress else 'no'}" + schedule_note),
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


def cmd_schedule(args: argparse.Namespace, console: Console) -> dict:
    """Preview learning-rate schedules: a table of the rates per epoch and a bar graph of the shape."""
    from .schedule import ScheduleError, describe, preview_points

    info = describe()
    if not args.lr_schedule and not args.act_lr_schedule:
        console.say("Presets (use them as --lr-schedule / --act-lr-schedule expressions):")
        console.table(("preset", "lr", "act_lr", "description"),
                      [[p["name"], p["lr"], p["act_lr"], p["description"]] for p in info["presets"]])
        console.say()
        console.say("variables: " + ", ".join(info["variables"]) + " · constants: " + ", ".join(info["constants"]))
        console.say("functions: " + ", ".join(info["functions"]))
        for line in info["helpers"]:
            console.say("  " + line)
        return {"presets": info["presets"], "variables": info["variables"], "functions": info["functions"], "helpers": info["helpers"]}
    try:
        points = preview_points(args.lr_schedule, args.act_lr_schedule, args.epochs, args.lr, args.act_lr)
    except ScheduleError as exc:
        raise CliError(str(exc)) from exc
    width = 30
    top_lr = max((p["lr"] for p in points), default=0.0) or 1.0
    top_act = max((p["act_lr"] for p in points), default=0.0) or 1.0
    rows = [
        [p["epoch"], p["lr"], "#" * max(1, round(width * p["lr"] / top_lr)) if p["lr"] > 0 else "",
         p["act_lr"], "#" * max(1, round(width * p["act_lr"] / top_act)) if p["act_lr"] > 0 else ""]
        for p in points
    ]
    console.pairs([
        ("lr", f"{args.lr} -> {args.lr_schedule or 'constant'}"),
        ("act_lr", f"{args.act_lr} -> {args.act_lr_schedule or 'constant'}"),
        ("epochs", args.epochs),
    ])
    console.say()
    console.table(("epoch", "lr", "lr graph", "act_lr", "act_lr graph"), rows)
    return {
        "lr_schedule": args.lr_schedule, "act_lr_schedule": args.act_lr_schedule, "epochs": args.epochs,
        "lr": args.lr, "act_lr": args.act_lr, "points": points,
    }


def cmd_feedback(args: argparse.Namespace, console: Console) -> dict:
    """Learn from rated texts: 2NRL when both kinds are given, reward on good alone, punish on bad alone."""
    good = list(args.good_text or [])
    bad = list(args.bad_text or [])
    if args.good:
        good += read_texts([args.good], what="good")
    if args.bad:
        bad += read_texts([args.bad], what="bad")
    good = [t for t in good if t.strip()]
    bad = [t for t in bad if t.strip()]
    if not good and not bad:
        raise CliError("nothing to learn from: give --good / --good-text (thumbs up) and/or --bad / --bad-text (thumbs down)")
    action = "2nrl" if good and bad else ("reward" if good else "punish")
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("thumbs up", f"{len(good)} texts"),
        ("thumbs down", f"{len(bad)} texts"),
        ("action", {"2nrl": "2NRL: bad -> invert -> good", "reward": "reward: train on the good texts",
                    "punish": "punish: train on the bad texts, then invert"}[action]),
        ("negative", f"epochs={args.neg_epochs} lr={args.neg_lr}"),
        ("positive", f"epochs={args.pos_epochs} lr={args.pos_lr} act_lr={args.pos_lr / 10}"),
        ("batch", args.batch_size),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console, phased=True)
    stop = threading.Event()

    def work() -> dict:
        if action == "2nrl":
            return model.two_nrl(
                bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
                pos_lr=args.pos_lr, progress=printer, stop_event=stop, batch_size=args.batch_size,
            )
        if action == "reward":
            records = model.train(
                good, epochs=args.pos_epochs, lr=args.pos_lr, act_lr=args.pos_lr / 10, batch_size=args.batch_size,
                progress=lambda rec: printer({**rec, "phase": "positive"}), stop_event=stop,
            )
            return {"negative": [], "positive": [{**r, "phase": "positive"} for r in records], "inverted": model.graph.inverted}
        records = model.train(
            bad, epochs=args.neg_epochs, lr=args.neg_lr, batch_size=args.batch_size,
            progress=lambda rec: printer({**rec, "phase": "negative"}), stop_event=stop,
        )
        if not stop.is_set():
            model.invert()
        return {"negative": [{**r, "phase": "negative"} for r in records], "positive": [], "inverted": model.graph.inverted}

    result, interrupted = run_interruptible(work, stop, console, "epoch")
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    console.say(f"action: {action}; inverted: {fmt(result['inverted'])}")
    return {
        "model": origin.to_dict(), "out": out, "action": action, "good_texts": len(good), "bad_texts": len(bad),
        "negative": result["negative"], "positive": result["positive"], "inverted": result["inverted"],
        "interrupted": interrupted, "saved": saved, "stats": model.stats(),
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


def cmd_codegen(args: argparse.Namespace, console: Console) -> dict:
    from .codegen import PHASES, CodeGenConfig, CodeGenTrainer, Sandbox, load_problems
    from .ollama import OllamaClient, OllamaError

    try:
        problems = load_problems(args.problems)
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot load problems from {args.problems}: {exc}") from exc
    phases = PHASES if args.phase == "both" else (args.phase,)
    manager = checkpoint_manager(args)
    config = CodeGenConfig(
        teacher_model=args.teacher_model or CodeGenConfig().teacher_model, judge_model=args.judge_model, phases=phases,
        rounds=args.rounds, teacher_attempts=args.teacher_attempts, model_attempts=args.model_attempts,
        first_attempt_dijkstra=not args.sample_first, temperature=args.temperature, max_length=args.max_length,
        strictness=args.strictness, use_judge=not args.no_judge, fallback_teacher=not args.no_fallback_teacher,
        twonrl_per=args.twonrl_per, replay=not args.no_replay, teacher_prompt=args.teacher_prompt,
        model_prompt=args.model_prompt if args.model_prompt is not None else CodeGenConfig().model_prompt,
        neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr, pos_lr=args.pos_lr,
        batch_size=args.batch_size, checkpoint_every=checkpoint_every(args, manager),
    )
    try:
        config.validate()
        client = OllamaClient(args.url, config.teacher_model, args.timeout)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    sandbox = Sandbox(timeout=args.sandbox_timeout, memory_mb=args.memory_mb, isolate_network=not args.no_network_isolation)
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("problems", f"{len(problems)} from {args.problems}"),
        ("phases", " -> ".join(phases) + f", {config.rounds} round(s)"),
        ("teacher", f"{config.teacher_model} at {client.url}"),
        ("judge", (config.judge_model or config.teacher_model) if config.use_judge else "off (sandbox, expected output and tests only)"),
        ("attempts", f"teacher={config.teacher_attempts} model={config.model_attempts}"
                     + ("" if config.fallback_teacher else ", no teacher fallback")),
        ("sandbox", f"timeout={sandbox.timeout:g}s memory={sandbox.memory_mb}MB network="
                    + ("isolated" if sandbox.network_isolated else "NOT isolated")),
        ("2NRL", f"per {config.twonrl_per}: negative epochs={config.neg_epochs} lr={config.neg_lr}, positive epochs={config.pos_epochs} "
                 f"lr={config.pos_lr}, batch={config.batch_size}, replay={'on' if config.replay else 'off'}, {config.strictness}"),
        ("output", out),
    ])
    console.say()
    printer = ProblemPrinter(console)
    stop = threading.Event()
    trainer = CodeGenTrainer(model, client, sandbox, config)
    try:
        records, interrupted = run_interruptible(
            lambda: trainer.run(problems, progress=printer, stop_event=stop, checkpoint_manager=manager),
            stop, console, "problem",
        )
    except OllamaError as exc:
        raise CliError(str(exc)) from exc
    saved = _finish_training(console, model, out, interrupted, printer, "problem")
    problem_records = [r for r in records if r.get("kind") == "problem"]
    solved = sum(1 for r in problem_records if r["correct"])
    by_model = sum(1 for r in problem_records if r["model_solved"])
    console.say(
        f"{solved}/{len(problem_records)} problem runs solved ({by_model} by the model); "
        f"{len(trainer.solved)} of {len(problems)} problems have a correct solution"
    )
    doc = {
        "model": origin.to_dict(), "out": out, "problems": [p.to_dict() for p in problems], "config": config.to_dict(),
        "records": records, "attempts": printer.attempts, "solved": solved, "model_solved": by_model,
        "solutions": trainer.solved, "interrupted": interrupted, "saved": saved, "stats": model.stats(),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({k: doc[k] for k in ("problems", "config", "records", "solutions", "solved", "model_solved")}, fh, indent=2)
        console.say(f"wrote report to {args.report}")
    return doc


def _ollama_client(args: argparse.Namespace) -> Any:
    from .ollama import OllamaClient

    try:
        return OllamaClient(args.url, args.ollama_model, args.timeout)
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def cmd_ollama_models(args: argparse.Namespace, console: Console) -> dict:
    from .ollama import OllamaError

    client = _ollama_client(args)
    try:
        models = client.models()
    except OllamaError as exc:
        raise CliError(str(exc)) from exc
    console.pairs([("ollama", client.url), ("default model", client.model)])
    console.say()
    if models:
        console.table(("name", "bytes", "modified"), [[m.get("name"), m.get("size"), m.get("modified_at")] for m in models])
    else:
        console.say("no models installed (pull one with `ollama pull llama3.2`)")
    return {"url": client.url, "model": client.model, "models": models}


def cmd_ollama_corpus(args: argparse.Namespace, console: Console) -> dict:
    from .ollama import OllamaError, corpus_from_prompt

    client = _ollama_client(args)
    console.pairs([
        ("ollama", f"{client.model} at {client.url}"),
        ("prompt", quote(clip(args.prompt, 60))),
        ("style", args.style),
        ("lines", args.lines),
    ])
    try:
        texts = corpus_from_prompt(client, args.prompt, lines=args.lines, style=args.style, model=client.model)
    except OllamaError as exc:
        raise CliError(str(exc)) from exc
    if not texts:
        raise CliError(f"Ollama model {client.model!r} returned no usable lines")
    console.say()
    for i, line in enumerate(texts, 1):
        console.say(f"{i:3d}  {line}")
    doc: dict[str, Any] = {
        "url": client.url, "model": client.model, "prompt": args.prompt, "style": args.style,
        "lines": len(texts), "texts": texts, "out": None, "trained": None,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(texts) + "\n")
        console.say()
        console.say(f"wrote {len(texts)} lines to {args.out}")
        doc["out"] = args.out
    if args.train:
        model, origin = open_model(args, console, required=False)
        target = args.model_out or args.model
        console.say()
        console.pairs([
            ("model", origin.describe()),
            ("backend", backend_label(model)),
            ("training", f"epochs={args.epochs} lr={args.lr} batch={args.batch_size}"),
            ("output", target),
        ])
        console.say()
        printer = EpochPrinter(console)
        stop = threading.Event()
        records, interrupted = run_interruptible(
            lambda: model.train(
                texts, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, progress=printer, stop_event=stop,
            ),
            stop, console, "epoch",
        )
        saved = _finish_training(console, model, target, interrupted, printer, "epoch")
        doc["trained"] = {
            "model": origin.to_dict(), "out": target, "epochs": records, "interrupted": interrupted,
            "saved": saved, "stats": model.stats(),
        }
    return doc


def cmd_ollama_review(args: argparse.Namespace, console: Console) -> dict:
    from .ollama import OllamaError, adversarial_review

    client = _ollama_client(args)
    texts: list[str] | None = None
    if args.text:
        texts = list(args.text)
    elif args.data:
        texts = read_texts([args.data], what="reviewable")
    model = origin = None
    if texts is None or args.apply_two_nrl:
        model, origin = open_model(args, console, required=texts is None)
    if texts is not None:
        source = f"{len(texts)} given texts"
    else:
        source = f"{args.count} samples from {origin.describe()}" + (f" continuing {quote(args.prefix)}" if args.prefix else "")
    console.pairs([("ollama", f"{client.model} at {client.url}"), ("source", source), ("threshold", args.threshold)])
    try:
        result = adversarial_review(
            model, client, count=args.count, prefix=args.prefix, max_length=args.max_length,
            temperature=args.temperature, texts=texts, threshold=args.threshold, context=args.context,
            ollama_model=client.model, seed=args.seed,
        )
    except OllamaError as exc:
        raise CliError(str(exc)) from exc
    console.say()
    rows = [[r["rating"], r["verdict"], quote(clip(r["text"], 50)), clip(r["critique"], 60)] for r in result["reviews"]]
    console.table(("rating", "verdict", "text", "critique"), rows)
    console.say()
    console.say(
        f"{len(result['reviews'])} texts: mean rating {fmt(result['mean_rating'])}, pass rate {fmt(result['pass_rate'])}, "
        f"{len(result['bad'])} bad / {len(result['good'])} good"
    )
    doc: dict[str, Any] = dict(result)
    doc["two_nrl"] = None
    if args.apply_two_nrl:
        bad = list(result["bad"])
        good = list(result["good"])
        if args.good:
            good.extend(read_texts([args.good], what="good"))
        if not bad:
            raise CliError("nothing failed the review, so there is no garbage for the negative phase")
        if not good:
            raise CliError("no text passed the review and no --good file was given for the positive phase")
        out = args.out or args.model
        console.say()
        console.pairs([
            ("2NRL", f"{len(bad)} bad -> invert -> {len(good)} good"),
            ("negative", f"epochs={args.neg_epochs} lr={args.neg_lr}"),
            ("positive", f"epochs={args.pos_epochs} lr={args.pos_lr} act_lr={args.pos_lr / 10}"),
            ("output", out),
        ])
        console.say()
        printer = EpochPrinter(console, phased=True)
        stop = threading.Event()
        two, interrupted = run_interruptible(
            lambda: model.two_nrl(
                bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
                pos_lr=args.pos_lr, progress=printer, stop_event=stop, batch_size=args.batch_size,
            ),
            stop, console, "epoch",
        )
        saved = _finish_training(console, model, out, interrupted, printer, "epoch")
        doc["two_nrl"] = {
            "bad_texts": len(bad), "good_texts": len(good), "negative": two["negative"], "positive": two["positive"],
            "inverted": two["inverted"], "interrupted": interrupted, "saved": saved, "stats": model.stats(),
        }
    return doc


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
        "upload_dir": args.upload_dir,
        "frontend_dir": frontend_dir,
        "frontend_built": os.path.isfile(os.path.join(frontend_dir, "index.html")),
        "backend": args.backend,
        "device": args.device,
        "seed": effective_seed(args),
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "version": __version__,
    }
    console.pairs([
        ("serving", doc["url"]),
        ("model", args.model + ("" if doc["model_exists"] else " (new; created on first save)")),
        ("checkpoints", args.checkpoint_dir),
        ("uploads", args.upload_dir),
        ("frontend", frontend_dir + ("" if doc["frontend_built"] else " (not built: run `npm install && npm run build` in frontend/)")),
        ("backend", args.backend + (f" on {args.device}" if args.device else "")),
        ("ollama", f"{args.ollama_model or '$RADIXNET_OLLAMA_MODEL'} at {args.ollama_url or '$OLLAMA_HOST'}"),
    ])
    console.say("press Ctrl-C to stop")
    if console.json_mode:
        console.emit(doc)  # the server blocks, so the document goes out first
    api.run_server(
        host=args.host, port=args.port, model_path=args.model, checkpoint_dir=args.checkpoint_dir,
        frontend_dir=frontend_dir, backend=args.backend, device=args.device, seed=effective_seed(args),
        upload_dir=args.upload_dir, ollama_url=args.ollama_url, ollama_model=args.ollama_model,
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
    p.add_argument("--lr-schedule", metavar="EXPR",
                   help="graph function of the epoch for the learning rate, e.g. 'linear(lr0, 4 * lr0)', 'lr0 * 1.25 ** i', "
                        "'warmup(lr0 / 10, lr0, 3)' (variables: epoch, i, epochs, t, lr0; see `radixnet schedule --help`)")
    p.add_argument("--act-lr-schedule", metavar="EXPR",
                   help="graph function of the epoch for the activation learning rate; may use lr (the epoch's rate), e.g. 'lr / 10'")
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
    p.add_argument("--max-length", type=nonneg_int, metavar="N",
                   help="optional hard cap on emitted characters (default: no limit; dijkstra returns the whole cheapest path)")
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

    # schedule -------------------------------------------------------------
    p = command(
        "schedule", "preview a learning-rate schedule (graph function of the epoch)",
        "Evaluate --lr-schedule / --act-lr-schedule expressions for --epochs epochs and print the rate\n"
        "of every epoch with a bar graph of the shape.  Without expressions the presets, variables and\n"
        "functions are listed.  Variables: epoch (1-based), i (0-based), epochs, t (0 at the first epoch,\n"
        "1 at the last), lr0 (the base rate), act_lr0, and - for the activation schedule - lr (the\n"
        "epoch's learning rate).  Helpers: linear(a, b), geometric(a, b), cosine(a, b),\n"
        "step(a, factor, every), warmup(a, b, n); plus sin cos exp log sqrt min max clamp ... and pi, e.",
    )
    p.add_argument("--lr-schedule", metavar="EXPR", help="learning-rate expression, e.g. 'linear(lr0, 4 * lr0)'")
    p.add_argument("--act-lr-schedule", metavar="EXPR", help="activation learning-rate expression, e.g. 'lr / 10'")
    p.add_argument("--epochs", type=nonneg_int, default=10, help="epochs to preview")
    p.add_argument("--lr", type=nonneg_float, default=TrainConfig.lr, help="base learning rate (lr0)")
    p.add_argument("--act-lr", type=nonneg_float, default=TrainConfig.act_lr, help="base activation learning rate (act_lr0)")
    p.set_defaults(handler=cmd_schedule)

    # feedback -------------------------------------------------------------
    p = command(
        "feedback", "learn from rated texts (thumbs up / thumbs down) with 2NRL",
        "Thumbs up (--good / --good-text) are correct texts, thumbs down (--bad / --bad-text) garbage.  Both:\n"
        "2NRL (train on the bad texts, invert, fine-tune on the good ones).  Only good: reward (a positive-phase\n"
        "pass).  Only bad: punish (a negative-phase pass, then the network is inverted so those texts become\n"
        "unlikely).  The same rule the frontend's Generate tab uses for its ratings.",
    )
    p.add_argument("--good", metavar="FILE", help="thumbs-up texts, one per line")
    p.add_argument("--bad", metavar="FILE", help="thumbs-down texts, one per line")
    p.add_argument("--good-text", action="append", metavar="TEXT", help="a thumbs-up text (repeatable)")
    p.add_argument("--bad-text", action="append", metavar="TEXT", help="a thumbs-down text (repeatable)")
    group = p.add_argument_group("2NRL options")
    group.add_argument("--neg-epochs", type=nonneg_int, default=2, help="epochs of the negative (punish) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=3, help="epochs of the positive (reward) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.5, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.1, help="learning rate of the positive phase (activation parameters use a tenth)")
    group.add_argument("--batch-size", type=pos_int, default=4, help="transitions per backend step (rated sets are small)")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_feedback)

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

    # codegen --------------------------------------------------------------
    from .codegen import DEFAULT_TEACHER_MODEL as codegen_default_model

    p = command(
        "codegen", "generate Python programs, run them in a sandbox, judge them with Ollama, reward / punish with 2NRL",
        "Semi-supervised code generation over a list of problems.  Phase `teacher`: an Ollama model writes\n"
        "each solution, the sandbox runs it, the teacher fixes failures and the judge (the LLM plus an\n"
        "objective PEP 8 / naming check) confirms it; the network learns question + answer with every wrong\n"
        "attempt as 2NRL garbage before the correct one.  Phase `model`: the network writes the solutions\n"
        "itself; errors and rejected programs are punished (negative phase), correct ones rewarded (positive\n"
        "phase), the teacher supplies the answer when the network never succeeds.  Ctrl-C stops after the\n"
        "current problem and saves.",
    )
    p.add_argument("--problems", required=True, metavar="FILE",
                   help="one prompt per line, or .json / .jsonl objects {id, prompt, tests, expected_output}")
    p.add_argument("--phase", choices=("both", "teacher", "model"), default="both",
                   help="teacher: Ollama writes the solutions; model: the network writes them; both: teacher, then model")
    p.add_argument("--rounds", type=pos_int, default=1, help="passes over the problem list (each pass runs the chosen phases)")
    p.add_argument("--teacher-model", metavar="NAME",
                   help=f"Ollama model that writes, fixes and judges (default: $RADIXNET_CODEGEN_MODEL or {codegen_default_model})")
    p.add_argument("--judge-model", metavar="NAME", help="a different Ollama model for judging (default: the teacher model)")
    p.add_argument("--url", metavar="URL", help="Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
    p.add_argument("--timeout", type=_float_at_least(1.0), metavar="SECONDS", help="seconds to wait for one Ollama answer (default: 120)")
    p.add_argument("--teacher-attempts", type=pos_int, default=3, help="programs the teacher may try per problem (first + fixes)")
    p.add_argument("--model-attempts", type=pos_int, default=4, help="programs the network may try per problem before the teacher steps in")
    p.add_argument("--sample-first", action="store_true",
                   help="sample the network's first attempt too (default: the first attempt is the cheapest path)")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="sampling temperature of the network's attempts")
    p.add_argument("--max-length", type=pos_int, default=800, help="characters the network may generate per attempt")
    p.add_argument("--strictness", choices=("strict", "lenient"), default="strict",
                   help="strict: runs + task + PEP 8 + naming; lenient: runs + task")
    p.add_argument("--no-judge", action="store_true", help="no LLM judge: correctness from the sandbox, expected output and tests only")
    p.add_argument("--no-fallback-teacher", action="store_true", help="in the model phase, never ask the teacher for the correct answer")
    p.add_argument("--twonrl-per", choices=("problem", "round"), default="problem",
                   help="apply 2NRL after every problem, or once per round over all attempts")
    p.add_argument("--no-replay", action="store_true", help="do not add earlier correct solutions to every positive phase")
    p.add_argument("--teacher-prompt", metavar="TEXT", help="extra instructions for the teacher (phase 1 prompt)")
    p.add_argument("--model-prompt", metavar="TEMPLATE",
                   help="prefix template the network continues into code; {problem} is the prompt (default: '{problem}' then a newline)")
    p.add_argument("--sandbox-timeout", type=_float_at_least(0.1), default=10.0, help="seconds a program may run")
    p.add_argument("--memory-mb", type=nonneg_int, default=256, help="memory limit of a program (0 = unlimited)")
    p.add_argument("--no-network-isolation", action="store_true", help="do not run programs in a separate network namespace")
    group = p.add_argument_group("2NRL options")
    group.add_argument("--neg-epochs", type=nonneg_int, default=2, help="epochs of the negative (punish) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=3, help="epochs of the positive (reward) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.5, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.1, help="learning rate of the positive phase (activation parameters use a tenth)")
    group.add_argument("--batch-size", type=pos_int, default=4, help="transitions per backend step")
    _add_checkpoint_options(p, "problem")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.add_argument("--report", metavar="FILE", help="write a JSON report (problems, config, records, solutions)")
    p.set_defaults(handler=cmd_codegen)

    # ollama ---------------------------------------------------------------
    from .ollama import DEFAULT_MODEL as ollama_default_model, DEFAULT_URL as ollama_default_url

    p = command(
        "ollama", "hook the network into a local Ollama LLM: corpus from a prompt, adversarial review",
        "Talk to an Ollama server (https://ollama.com).  `corpus` turns a prompt into training lines, correct\n"
        "or deliberately garbage (the two halves of 2NRL), and can train on them; `review` lets the LLM\n"
        "adversarially rate the network's own samples (or given texts) from 0 to 10 and, with --2nrl,\n"
        "feeds the failed ones back as garbage and the passed ones as correct data.\n"
        f"Usage: {PROG} [global options] ollama [--url URL] [--ollama-model NAME] <action> [options]",
    )
    p.add_argument("--url", metavar="URL", help=f"Ollama base URL (default: $OLLAMA_HOST or {ollama_default_url})")
    p.add_argument("--ollama-model", metavar="NAME",
                   help=f"Ollama model name (default: $RADIXNET_OLLAMA_MODEL or {ollama_default_model})")
    p.add_argument("--timeout", type=_float_at_least(1.0), metavar="SECONDS",
                   help="seconds to wait for one Ollama answer (default: 120)")
    actions = p.add_subparsers(dest="action", metavar="<action>", title="actions", required=True)

    a = actions.add_parser("models", help="list the models installed in Ollama",
                           description="List the models installed in the Ollama server.", formatter_class=_HelpFormatter)
    a.set_defaults(handler=cmd_ollama_models)

    a = actions.add_parser(
        "corpus", help="turn a prompt into training lines (optionally train on them)",
        description="Ask the LLM for lines of text about --prompt: one short sentence per line, either correct\n"
                    "(--style good) or deliberately wrong (--style garbage, for the 2NRL negative phase).\n"
                    "--out writes them to a file, --train trains the model on them and saves it.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--prompt", required=True, metavar="TEXT", help="what the lines should be about / how they should read")
    a.add_argument("--lines", type=pos_int, default=20, help="number of lines to ask for")
    a.add_argument("--style", choices=("good", "garbage"), default="good",
                   help="correct text, or deliberately wrong text for the 2NRL negative phase")
    a.add_argument("--out", metavar="FILE", help="also write the lines to FILE (one per line; usable as --data / --bad / --good)")
    a.add_argument("--train", action="store_true", help="train the model on the lines and save it")
    a.add_argument("--epochs", type=nonneg_int, default=10, help="training epochs with --train")
    a.add_argument("--lr", type=nonneg_float, default=0.5, help="learning rate with --train (a prompt corpus is small, so it is high)")
    a.add_argument("--batch-size", type=pos_int, default=4, help="transitions per backend step with --train")
    a.add_argument("--model-out", metavar="PATH", help="where to save the trained model (default: --model)")
    a.set_defaults(handler=cmd_ollama_corpus)

    a = actions.add_parser(
        "review", help="adversarial LLM review / rating of the network's output",
        description="The LLM plays the harsh critic: every sample the network generates (or every given text)\n"
                    "gets a rating from 0 to 10, a pass/fail verdict against --threshold and a one-sentence\n"
                    "critique.  With --2nrl the failed texts become the negative phase and the passed texts\n"
                    "(plus --good) the positive phase of a 2NRL pass, after which the model is saved.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--count", type=pos_int, default=8, help="samples to draw from the model")
    a.add_argument("--prefix", default="", metavar="TEXT", help="continue this prefix instead of generating from scratch")
    a.add_argument("--max-length", type=nonneg_int, default=60, help="characters per sample")
    a.add_argument("--temperature", type=nonneg_float, default=1.0, help="sampling temperature")
    a.add_argument("--text", action="append", metavar="TEXT", help="review this text instead of sampling (repeatable)")
    a.add_argument("--data", metavar="FILE", help="review the texts of FILE (one per line) instead of sampling")
    a.add_argument("--threshold", type=nonneg_float, default=6.0, help="ratings at or above this pass")
    a.add_argument("--context", metavar="TEXT", help="extra context for the reviewer (e.g. what the model was trained on)")
    a.add_argument("--2nrl", dest="apply_two_nrl", action="store_true",
                   help="afterwards run 2NRL: failed texts as garbage, passed texts (+ --good) as correct data, then save")
    a.add_argument("--good", metavar="FILE", help="extra correct texts for the 2NRL positive phase (one per line)")
    _add_two_nrl_options(a, neg_epochs=3, pos_epochs=3, batch_size=4)
    a.add_argument("--out", metavar="PATH", help="where to save the model after --2nrl (default: --model)")
    a.set_defaults(handler=cmd_ollama_review)

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
    p.add_argument("--upload-dir", default="uploads", metavar="DIR",
                   help="directory for training files uploaded through the API / frontend (default: uploads)")
    p.add_argument("--ollama-url", metavar="URL",
                   help=f"Ollama base URL for the /api/ollama endpoints (default: $OLLAMA_HOST or {ollama_default_url})")
    p.add_argument("--ollama-model", metavar="NAME",
                   help=f"default Ollama model for the /api/ollama endpoints (default: $RADIXNET_OLLAMA_MODEL or {ollama_default_model})")
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
