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

from . import __version__, diff
from .archive import zip_texts_from_file
from .checkpoint import CheckpointManager
from .gan import BLATANT_MODES, EvolveConfig, Evolver
from .beam import Prediction, path_probability
from .dialogue import DEFAULT_SPEAKERS, transcript
from .model import GraphModel, RadixNet, TrainConfig, load_model, model_class, model_kinds
from .speech import ASR_BACKENDS as SPEECH_BACKENDS
from .speech import DEFAULT_RATE as SPEECH_RATE
from .speech import RECORDERS as SPEECH_RECORDERS
from .speech import SPEECH_TOKEN as SPEECH_TOKEN_HELP

__all__ = ["main", "build_parser", "CliError", "EXIT_OK", "EXIT_ERROR", "EXIT_ABORTED"]

PROG = "radixnet"
DEFAULT_MODEL = "model.json"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_FRONTEND_DIR = os.path.join("frontend", "dist")
BACKENDS = ("auto", "python", "torch")
MODES = ("dijkstra", "sample")
PREDICT_MODES = ("dijkstra", "beam", "sample")
KINDS = ("radix", "count", "negative")
DEFAULT_COUNT_MODEL = "model.count.json"
DEFAULT_NEGATIVE_MODEL = "model.negative.json"
DEFAULT_KIND_MODELS = {"count": DEFAULT_COUNT_MODEL, "negative": DEFAULT_NEGATIVE_MODEL}

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
    ("fails", 5, ">"), ("blatant", 7, ">"), ("boost", 6, ">"),
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
            record.get("gap"), record.get("gen_loss"), record.get("failures"), record.get("blatant"),
            record.get("boost_mean"), record.get("nodes"), record.get("edges"), record.get("compression_ratio"),
            record.get("seconds"),
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


LESSON_COLUMNS: tuple[_Column, ...] = (
    ("round", 5, ">"), ("exercise", 8, "<"), ("score", 6, ">"), ("gram", 6, ">"), ("spell", 6, ">"),
    ("flu", 6, ">"), ("mark", 4, "<"), ("error", 12, "<"), ("sentence", 0, "<"),
)


class LessonPrinter(_RowPrinter):
    """``progress`` callback for :meth:`TutorTrainer.run`: lessons as rows, rounds and the report card as notes."""

    __slots__ = ("lessons", "corrections")

    def __init__(self, console: Console) -> None:
        super().__init__(console, LESSON_COLUMNS)
        self.lessons = 0
        self.corrections = 0

    def __call__(self, record: dict) -> None:
        kind = record.get("kind")
        if kind == "lesson":
            self.lessons += 1
            values = [
                record.get("round"), clip(str(record.get("exercise")), 8), record.get("score"), record.get("grammar"),
                record.get("spelling"), record.get("fluency"), "pass" if record.get("passed") else "fail",
                clip(str(record.get("error") or "-"), 12),
            ]
            sentence = quote(clip(str(record.get("sentence", "")), 46))
            pairs = list(zip([c[0] for c in LESSON_COLUMNS], values + [sentence]))
            self.row(values + [sentence], _compact(f"lesson {record.get('exercise')}", pairs[2:]))
            if not record.get("passed"):
                self.corrections += 1
                correction, comment = record.get("correction"), record.get("comment")
                if correction:
                    self.console.note(f"    correct: {quote(clip(str(correction), 80))}")
                if comment:
                    self.console.note(f"    teacher: {clip(str(comment), 100)}")
        elif kind == "round":
            weakest = ", ".join(record.get("weakest") or []) or "nothing"
            learned = record.get("action") or "nothing to learn"
            self.console.note(
                f"round {record.get('round')}: {record.get('passed')}/{record.get('lessons')} passed, "
                f"mean {fmt(record.get('mean_score'))} (grammar {fmt(record.get('mean_grammar'))}), "
                f"weakest: {weakest} -> {learned} (bad={record.get('bad')}, good={record.get('good')})"
            )
        elif kind == "report":
            self.console.note(
                f"report card: {record.get('passed')}/{record.get('lessons')} passed over {record.get('rounds')} round(s), "
                f"mean {fmt(record.get('mean_score'))}"
            )
        elif kind == "plan":
            self.console.note(
                f"lesson plan ({record.get('source')}): {len(record.get('lessons') or [])} lesson(s) at "
                f"{record.get('level')} level, drilling {', '.join(record.get('targets') or []) or 'nothing in particular'}"
            )
        elif kind == "note":
            self.console.note(f"note: {record.get('message')}")


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


def backend_label(model: GraphModel) -> str:
    return f"{model.backend.name} ({model.backend.device})"


def kind_label(model: GraphModel) -> str:
    return f"{model.kind} ({type(model).label})"


def effective_kind(args: argparse.Namespace) -> str:
    return getattr(args, "kind", None) or "radix"


def read_text_file(path: str) -> str:
    """UTF-8 file content; a missing or undecodable file is a :class:`CliError`."""
    if not os.path.isfile(path):
        raise CliError(f"data file not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise CliError(f"{path} is not valid UTF-8 text: {exc}") from exc


def _is_zip_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        return fh.read(4) in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def read_texts(paths: Sequence[str], whole_file: bool = False, what: str = "training") -> list[str]:
    """Texts from files: one per non-blank line, or one per file with ``whole_file``.

    A ``.zip`` archive contributes every text entry it holds (unpacked in
    memory; directories, metadata, binary, nested-archive and empty entries
    are skipped).  Lines are kept verbatim (only blank ones are dropped); in
    whole-file mode trailing line terminators are stripped and an archive
    entry counts as one file.  Raises :class:`CliError` when nothing usable
    was found.
    """
    texts: list[str] = []
    for path in paths:
        if _is_zip_file(path):
            try:
                contents = [entry.text for entry in zip_texts_from_file(path)]
            except ValueError as exc:
                raise CliError(f"{path}: {exc}") from exc
        else:
            contents = [read_text_file(path)]
        for content in contents:
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
) -> tuple[GraphModel, Origin]:
    """Load ``--model`` (or the latest checkpoint of ``manager``), else create a fresh model of ``--kind``.

    ``required`` turns a missing model file into an error (inference and
    maintenance commands); training commands start from a new model instead.
    A file's own kind wins over ``--kind`` (a note says so when they differ).
    """
    backend, device = args.backend, args.device
    wanted = getattr(args, "kind", None)
    if manager is not None:
        record = manager.latest()
        if record is not None:
            model = load_model(record["path"], backend=backend, device=device)
            return model, Origin("checkpoint", record["path"], f"{model.kind}, {model.meta['epochs_total']} epochs trained")
        console.note(f"note: no checkpoint to resume from in {manager.directory}")
    path = args.model
    if os.path.isfile(path):
        model = load_model(path, backend=backend, device=device)
        if wanted and wanted != model.kind:
            console.note(f"note: {path} holds a {model.kind} model; --kind {wanted} applies to new models only")
        return model, Origin("model", path, f"{model.kind}, {model.meta['epochs_total']} epochs trained")
    if required:
        raise CliError(f"model file not found: {path} (train one first with `{PROG} train --data FILE`)")
    seed = effective_seed(args)
    kind = effective_kind(args)
    model = model_class(kind)(seed=seed, backend=backend, device=device)
    return model, Origin("new", None, f"seed {seed}, kind {kind}")


def save_model(model: GraphModel, path: str) -> dict:
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


def _finish_training(console: Console, model: GraphModel, out: str, interrupted: bool, printer: _RowPrinter, unit: str) -> dict:
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
        lr_schedule=args.lr_schedule, act_lr_schedule=args.act_lr_schedule, reverse_schedule=args.reverse_schedule,
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
            + (" (reversed)" if config.reverse_schedule else "")
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
    mode = args.mode
    options = dict(
        length=args.length, mode=mode, step_penalty=args.step_penalty, temperature=args.temperature,
        to_end=args.to_end, max_length=args.max_length,
    )
    options.update(k=args.k, beam=args.beam)
    result = model.predict(args.prefix, **options)
    console.pairs([
        ("model", kind_label(model)),
        ("prefix", quote(args.prefix)),
        ("continuation", quote(result.text)),
        ("full text", quote(result.full_text)),
        ("cost", result.cost),
        ("probability", path_probability(result)),
        ("reached end", result.reached_end),
        ("expanded", result.expanded),
        ("mode", mode),
    ])
    console.say()
    console.say("path:")
    offset = len(result.labels) - len(result.step_costs)
    rows = [
        [i, node, result.step_costs[i - offset] if i >= offset else None, quote(label)]
        for i, (node, label) in enumerate(zip(result.node_ids, result.labels))
    ]
    console.table(("step", "node", "cost", "label"), rows)
    doc = {
        "prefix": args.prefix,
        "kind": model.kind,
        "continuation": result.text,
        "full_text": result.full_text,
        "cost": result.cost,
        "probability": path_probability(result),
        "step_costs": list(result.step_costs),
        "path": list(result.labels),
        "node_ids": list(result.node_ids),
        "expanded": result.expanded,
        "reached_end": result.reached_end,
        "mode": mode,
    }
    if isinstance(result, Prediction):
        for title, paths in (("top", result.top), ("bottom", result.bottom)):
            console.say()
            console.say(f"{title} {len(paths)} continuation(s) (k={result.k}, beam={result.beam}):")
            console.table(
                ("#", "cost", "prob", "end", "continuation"),
                [[i + 1, r.cost, path_probability(r), r.reached_end, quote(clip(r.text, 80))] for i, r in enumerate(paths)],
            )
        doc.update(
            k=result.k, beam=result.beam,
            top=[{**r.to_dict(), "probability": path_probability(r)} for r in result.top],
            bottom=[{**r.to_dict(), "probability": path_probability(r)} for r in result.bottom],
        )
    return doc


def cmd_generate(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    results = model.generate(
        max_length=args.max_length, mode=args.mode, temperature=args.temperature, count=args.count, seed=args.seed,
        prefix=args.prefix, step_penalty=args.step_penalty, beam=args.beam,
    )
    rows = [[i + 1, r.cost, path_probability(r), r.reached_end, quote(clip(r.text, 100))] for i, r in enumerate(results)]
    console.table(("#", "cost", "prob", "end", "text"), rows)
    return {
        "samples": [{**r.to_dict(), "probability": path_probability(r)} for r in results],
        "count": len(results),
        "mode": args.mode,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "temperature": args.temperature,
    }


def cmd_converse(args: argparse.Namespace, console: Console) -> dict:
    model, _ = open_model(args, console, required=True)
    partner = None
    if args.partner:
        if not os.path.isfile(args.partner):
            raise CliError(f"partner model file not found: {args.partner}")
        partner = load_model(args.partner, backend=args.backend, device=args.device)
    speakers = [name.strip() for name in args.speakers.split(",") if name.strip()] or list(DEFAULT_SPEAKERS)
    turns = model.converse(
        args.opening, args.turns, mode=args.mode, max_length=args.max_length, context=args.context,
        temperature=args.temperature, k=args.k, beam=args.beam, step_penalty=args.step_penalty, seed=args.seed,
        speakers=speakers, partner=partner, avoid_repeats=not args.allow_repeats,
    )
    for turn in turns:
        flags = [f for f, on in (("given", turn.given), ("new topic", turn.fresh and not turn.given), ("repeat", turn.repeat)) if on]
        console.say(f"{turn.speaker}: {turn.text}")
        detail = f"    cost {fmt(turn.cost)}  p {fmt(turn.probability)}"
        if turn.context:
            detail += f"  picked up {quote(turn.context)}"
        if flags:
            detail += f"  [{', '.join(flags)}]"
        console.say(detail)
    if not turns:
        console.say("(nothing to say: train the model first)")
    return {
        "turns": [t.to_dict() for t in turns],
        "count": len(turns),
        "speakers": speakers,
        "mode": args.mode,
        "opening": args.opening,
        "kind": model.kind,
        "partner_kind": partner.kind if partner is not None else None,
        "transcript": transcript(turns),
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


def _phase_pairs(model: GraphModel, args: argparse.Namespace) -> list[tuple[str, Any]]:
    """Console lines describing the negative / positive phases for the model's kind."""
    if model.kind == "count":
        return [
            ("negative", f"{args.neg_epochs} pass(es): reward -= {args.strength} on every edge of the bad paths"),
            ("positive", f"{args.pos_epochs} pass(es): traversal counted and reward += {args.strength} on the good paths"),
        ]
    return [
        ("negative", f"epochs={args.neg_epochs} lr={args.neg_lr}"),
        ("positive", f"epochs={args.pos_epochs} lr={args.pos_lr} act_lr={args.pos_lr / 10}"),
        ("batch", args.batch_size),
    ]


def cmd_two_nrl(args: argparse.Namespace, console: Console) -> dict:
    bad = read_texts([args.bad], what="bad")
    good = read_texts([args.good], what="good")
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("kind", kind_label(model)),
        ("backend", backend_label(model)),
        ("bad", f"{len(bad)} texts from {args.bad}"),
        ("good", f"{len(good)} texts from {args.good}"),
        *_phase_pairs(model, args),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console, phased=True)
    stop = threading.Event()
    result, interrupted = run_interruptible(
        lambda: model.two_nrl(
            bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
            pos_lr=args.pos_lr, progress=printer, stop_event=stop, batch_size=args.batch_size, strength=args.strength,
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
        points = preview_points(
            args.lr_schedule, args.act_lr_schedule, args.epochs, args.lr, args.act_lr, reverse=args.reverse_schedule,
        )
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
        ("reversed", args.reverse_schedule),
    ])
    console.say()
    console.table(("epoch", "lr", "lr graph", "act_lr", "act_lr graph"), rows)
    return {
        "lr_schedule": args.lr_schedule, "act_lr_schedule": args.act_lr_schedule, "epochs": args.epochs,
        "lr": args.lr, "act_lr": args.act_lr, "reverse_schedule": args.reverse_schedule, "points": points,
    }


def cmd_image_info(args: argparse.Namespace, console: Console) -> dict:
    from .vision import describe

    info = describe()
    console.pairs([
        ("pillow", info["pillow"]),
        ("torch", info["torch"]),
        ("diffusers", info["diffusers"]),
        ("sd vae", info["sd_model"] + (" (loaded)" if info["sd_loaded"] else "")),
        ("sd error", info["sd_error"] or "-"),
        ("auto picks", info["auto"] or "nothing: pip install pillow"),
        ("default size", info["default_size"]),
        ("text format", info["text_format"]),
    ])
    return info


def cmd_image_encode(args: argparse.Namespace, console: Console) -> dict:
    from .vision import VisionError, encode_image

    if not os.path.isfile(args.file):
        raise CliError(f"image file not found: {args.file}")
    with open(args.file, "rb") as fh:
        data = fh.read()
    try:
        result = encode_image(data, size=args.size, encoder=args.encoder)
    except VisionError as exc:
        raise CliError(str(exc)) from exc
    console.pairs([
        ("file", args.file),
        ("encoder", result["encoder"]),
        ("size", f"{result['width']}x{result['height']} (source {result['source_size'][0]}x{result['source_size'][1]})"),
        ("latent", " x ".join(str(v) for v in result["latent_shape"]) + f" = {result['bytes']} bytes"),
        ("text", f"{result['chars']} chars"),
    ])
    console.say()
    console.say(result["text"])
    result["file"] = args.file
    result["out"] = None
    result["trained"] = None
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result["text"] + "\n")
        result["out"] = args.out
        console.say()
        console.say(f"text written to {args.out}")
    if args.train:
        model, origin = open_model(args, console, required=False)
        out = args.model_out or args.model
        console.say()
        console.say(f"training {origin.describe()} on the encoded text: epochs={args.epochs} lr={args.lr} batch={args.batch_size}")
        records = model.train([result["text"]], epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        saved = save_model(model, out)
        result["trained"] = {"epochs": len(records), "loss": records[-1]["loss"] if records else None, "saved": saved, "kind": model.kind}
        console.say(f"trained {len(records)} epoch(s); saved to {out}")
    return result


def cmd_image_decode(args: argparse.Namespace, console: Console) -> dict:
    from .vision import VisionError, decode_text

    text = args.text if args.text is not None else read_text_file(args.data)
    try:
        result = decode_text(text, encoder=args.encoder)
    except VisionError as exc:
        raise CliError(str(exc)) from exc
    with open(args.out, "wb") as fh:
        fh.write(result.pop("png"))
    result["out"] = args.out
    console.pairs([
        ("encoder", result["encoder"]),
        ("size", f"{result['width']}x{result['height']}"),
        ("repaired", result["repaired"]),
        ("written", args.out),
    ])
    return result


def cmd_speech_info(args: argparse.Namespace, console: Console) -> dict:
    from .speech import describe

    info = describe()
    console.pairs([
        ("faster-whisper", info["faster_whisper"]),
        ("whisper", info["whisper"]),
        ("whisper model", info["whisper_model"]),
        ("transcription server", info["server_url"] or "-" ),
        ("auto picks", info["auto"] or "nothing installed: pass --text (or what the browser dictated)"),
        ("ffmpeg", info["ffmpeg"]),
        ("recorders", ", ".join(info["recorders"]) or "-"),
        ("codecs", ", ".join(info["codecs"])),
        ("default rate", f"{info['default_rate']} Hz"),
        ("token", f"{info['token']} (unique: {info['token_example']})"),
        ("text format", info["text_format"]),
        ("audio formats", info["formats"]),
    ])
    return info


def read_audio_file(path: str) -> bytes:
    if not os.path.isfile(path):
        raise CliError(f"audio file not found: {path}")
    with open(path, "rb") as fh:
        return fh.read()


def cmd_speech_transcribe(args: argparse.Namespace, console: Console) -> dict:
    """Speech to text only: what the audio says, nothing trained."""
    from .speech import SpeechError, transcribe

    data = read_audio_file(args.file)
    try:
        result = transcribe(
            data, backend=args.backend, text=args.text or "", language=args.language,
            model=args.asr_model, url=args.asr_url,
        )
    except SpeechError as exc:
        raise CliError(str(exc)) from exc
    console.pairs([
        ("file", args.file),
        ("backend", result["backend"]),
        ("model", result["model"] or "-"),
        ("language", result["language"] or "-"),
        ("took", f"{result['seconds']:.2f}s"),
    ])
    console.say()
    console.say(result["transcript"] or "(nothing was recognised)")
    result["file"] = args.file
    result["out"] = None
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result["transcript"] + "\n")
        result["out"] = args.out
        console.say()
        console.say(f"transcript written to {args.out}")
    return result


def _teach_from_audio(args: argparse.Namespace, console: Console, data: bytes, source: str) -> dict:
    """The shared flow of `speech teach` and `speech listen`: transcribe, encode the waveform, train."""
    from .speech import SpeechError, teach

    try:
        result = teach(
            data, transcript=args.text or "", backend=args.backend, language=args.language,
            asr_model=args.asr_model, asr_url=args.asr_url, rate=args.rate, codec=args.codec,
            normalise=args.normalise, waveform=not args.no_waveform, pair=args.pair,
            token=args.token, unique=not args.shared_token,
        )
    except SpeechError as exc:
        raise CliError(str(exc)) from exc
    audio = result["audio"]
    console.pairs([
        ("source", source),
        ("token", result["token"]),
        ("transcript", result["transcript"] or "-"),
        ("transcribed by", result["asr"]["backend"] or "-"),
        ("transcription error", result["asr"]["error"] or "-"),
        ("waveform", f"{audio['codec']} {audio['rate']} Hz, {audio['seconds']:.2f}s, {audio['bytes']} bytes"
                     if audio else "-"),
        ("texts", f"{len(result['texts'])} ({result['chars']} chars)"),
    ])
    if result["asr"]["error"]:
        console.note(f"note: not transcribed ({result['asr']['error']}); only the waveform is learned")
    for text in result["texts"]:
        console.say()
        console.say(text if len(text) <= 200 else text[:200] + f"… (+{len(text) - 200} chars)")
    result["source"] = source
    result["out"] = None
    result["saved_audio"] = None
    result["trained"] = None
    if getattr(args, "save", None):
        with open(args.save, "wb") as fh:
            fh.write(data)
        result["saved_audio"] = args.save
        console.say()
        console.say(f"audio written to {args.save}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(result["texts"]) + "\n")
        result["out"] = args.out
        console.say()
        console.say(f"{len(result['texts'])} text(s) written to {args.out}")
    if args.train:
        if not result["texts"]:
            raise CliError("nothing to train on: no transcript and no waveform")
        model, origin = open_model(args, console, required=False)
        out = args.model_out or args.model
        console.say()
        console.say(f"teaching {origin.describe()}: epochs={args.epochs} lr={args.lr} batch={args.batch_size}")
        records = model.train(result["texts"], epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        saved = save_model(model, out)
        result["trained"] = {
            "epochs": len(records), "loss": records[-1]["loss"] if records else None,
            "texts": len(result["texts"]), "saved": saved, "kind": model.kind,
        }
        console.say(f"trained {len(records)} epoch(s) on {len(result['texts'])} text(s); saved to {out}")
    return result


def cmd_speech_teach(args: argparse.Namespace, console: Console) -> dict:
    return _teach_from_audio(args, console, read_audio_file(args.file), args.file)


def cmd_speech_listen(args: argparse.Namespace, console: Console) -> dict:
    """Talk to the model: record from the microphone, then teach it what was said and how it sounded."""
    from .speech import SpeechError, record

    console.note(f"recording {args.seconds:g}s from the microphone - speak now…")
    try:
        data = record(seconds=args.seconds, rate=args.record_rate, recorder=args.recorder)
    except SpeechError as exc:
        raise CliError(str(exc)) from exc
    console.note("recording finished")
    return _teach_from_audio(args, console, data, f"microphone ({args.seconds:g}s)")


def cmd_speech_decode(args: argparse.Namespace, console: Console) -> dict:
    """An encoded - or predicted - waveform text back into a WAV file, so it can be listened to."""
    from .speech import SpeechError, decode_text

    text = args.text if args.text is not None else read_text_file(args.data)
    try:
        result = decode_text(text, codec=args.codec)
    except SpeechError as exc:
        raise CliError(str(exc)) from exc
    with open(args.out, "wb") as fh:
        fh.write(result.pop("wav"))
    result["out"] = args.out
    console.pairs([
        ("codec", result["codec"]),
        ("rate", f"{result['rate']} Hz x {result['channels']}"),
        ("length", f"{result['seconds']:.2f}s ({result['samples']} samples)"),
        ("repaired", result["repaired"]),
        ("written", args.out),
    ])
    return result


def cmd_weights(args: argparse.Namespace, console: Console) -> dict:
    model, origin = open_model(args, console, required=True)
    if model.kind != "count":
        raise CliError(f"{args.model} holds a {model.kind} model; the weight function belongs to the count model (--kind count)")
    options = {
        "count_scale": args.count_scale, "global_scale": args.global_scale, "window_scale": args.window_scale,
        "reward_scale": args.reward_scale, "window": args.window,
    }
    changes = {k: v for k, v in options.items() if v is not None}
    saved = None
    if changes:
        try:
            model.configure_weights(**changes)
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        saved = save_model(model, args.out or args.model)
    config = model.weight_config()
    stats = model.stats()
    console.pairs([
        ("model", origin.describe()),
        ("function", "global_scale * log(R_all) + window_scale * log(R_recent) + reward_scale * reward + count_scale * log(1 + count)"),
        ("count_scale", config["count_scale"]),
        ("global_scale", config["global_scale"]),
        ("window_scale", config["window_scale"]),
        ("reward_scale", config["reward_scale"]),
        ("window", f"{config['window']} traversals ({stats['window_traversals']} inside now)"),
        ("total traversals", stats["total_traversals"]),
        ("changed", ", ".join(f"{k}={v}" for k, v in changes.items()) if changes else "nothing"),
        ("saved", saved["path"] if saved else "-"),
    ])
    return {"weights": config, "changed": changes, "saved": saved, "stats": stats}


def _marks(raw: str | None, texts: list[str], option: str) -> list[float] | None:
    """``--good-ratings 8,10,6`` -> one weight per text (mark / 10), in the order the texts were collected."""
    if not raw:
        return None
    try:
        marks = [float(part) for part in raw.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise CliError(f"{option} must be a comma-separated list of marks out of 10: {exc}") from exc
    if len(marks) != len(texts):
        raise CliError(f"{option} has {len(marks)} mark(s) for {len(texts)} text(s)")
    if any(not (0.0 <= mark <= 10.0) for mark in marks):
        raise CliError(f"{option} marks must lie between 0 and 10")
    return [mark / 10.0 for mark in marks]


def _marks_label(raw: str | None) -> str:
    return ", ".join(part for part in (raw or "").replace(" ", "").split(",") if part)


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
    good_weights = _marks(args.good_ratings, good, "--good-ratings")
    bad_weights = _marks(args.bad_ratings, bad, "--bad-ratings")
    action = "2nrl" if good and bad else ("reward" if good else "punish")
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    if model.kind == "count":
        wording = {"2nrl": "2NRL: penalise the bad paths, then count + reward the good ones",
                   "reward": "reward: count + reward the good paths", "punish": "punish: penalise the bad paths"}
    else:
        wording = {"2nrl": "2NRL: bad -> invert -> good", "reward": "reward: train on the good texts",
                   "punish": "punish: train on the bad texts, then invert"}
    console.pairs([
        ("model", origin.describe()),
        ("kind", kind_label(model)),
        ("backend", backend_label(model)),
        ("thumbs up", f"{len(good)} texts" + (f", marks {_marks_label(args.good_ratings)}/10" if good_weights else "")),
        ("thumbs down", f"{len(bad)} texts" + (f", marks {_marks_label(args.bad_ratings)}/10" if bad_weights else "")),
        ("action", wording[action]),
        *_phase_pairs(model, args),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console, phased=True)
    stop = threading.Event()

    def work() -> dict:
        if action == "2nrl":
            return model.two_nrl(
                bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
                pos_lr=args.pos_lr, progress=printer, stop_event=stop, batch_size=args.batch_size, strength=args.strength,
                bad_weights=bad_weights, good_weights=good_weights,
            )
        if action == "reward":
            records = model.reward(
                good, epochs=args.pos_epochs, lr=args.pos_lr, batch_size=args.batch_size, strength=args.strength,
                weights=good_weights, progress=printer, stop_event=stop,
            )
            return {"negative": [], "positive": records, "inverted": model.graph.inverted}
        records = model.punish(
            bad, epochs=args.neg_epochs, lr=args.neg_lr, batch_size=args.batch_size, strength=args.strength,
            weights=bad_weights, progress=printer, stop_event=stop,
        )
        return {"negative": records, "positive": [], "inverted": model.graph.inverted}

    result, interrupted = run_interruptible(work, stop, console, "epoch")
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    console.say(f"action: {action}; inverted: {fmt(result['inverted'])}")
    return {
        "model": origin.to_dict(), "out": out, "action": action, "good_texts": len(good), "bad_texts": len(bad),
        "good_weights": good_weights, "bad_weights": bad_weights,
        "negative": result["negative"], "positive": result["positive"], "inverted": result["inverted"],
        "interrupted": interrupted, "saved": saved, "stats": model.stats(),
    }


def cmd_correct(args: argparse.Namespace, console: Console) -> dict:
    """Teach one correction: only the trigram nodes the two sentences disagree on move."""
    wrong, right = str(args.wrong or ""), str(args.right or "")
    if not wrong.strip() and not right.strip():
        raise CliError("nothing to correct: give --wrong (what the network wrote) and --right (what it should say)")
    changes = diff.summary(wrong, right, limit=0)
    if args.dry_run:
        console.pairs([("wrong", wrong), ("right", right), ("changes", len(changes))])
        _print_changes(console, changes)
        return {"wrong": wrong, "right": right, "changes": changes, "dry_run": True}
    model, origin = open_model(args, console, required=True)
    if not hasattr(model, "correct"):
        raise CliError(f"the {kind_label(model)} model cannot learn from a diff; use feedback instead")
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("kind", kind_label(model)),
        ("wrong", wrong),
        ("right", right),
        ("weights", f"penalty {args.weight}, reward {args.reward}, keep {args.keep}, strength {args.strength}"),
        ("output", out),
    ])
    moved = model.correct(
        wrong, right, strength=args.strength, weight=args.weight, reward=args.reward, keep=args.keep,
        count=not args.no_count,
    )
    negative = blamed = None
    if args.blame:
        negative, neg_origin = open_negative(args, console, required=False)
        console.pairs([("negative model", f"{neg_origin.describe()} -> {negative_path(args)}")])
        blamed = negative.correct(
            wrong, right, reason=args.reason, severity=args.weight * args.strength, source="cli",
            note=args.note,
        )
    _print_changes(console, changes)
    console.say(
        f"moved: {moved['penalised']} step(s) penalised, {moved['rewarded']} taught, "
        f"{moved['kept']} kept at {args.keep}"
    )
    if blamed is not None:
        console.say(
            f"negative network: {blamed['blamed']} step(s) blamed for {blamed['reason']}, {blamed['cleared']} cleared"
        )
    saved = save_model(model, out)
    console.say(f"saved {out} ({saved['bytes']} bytes)")
    return {
        "model": origin.to_dict(), "out": out, "wrong": wrong, "right": right, **moved,
        "changes": changes, "saved": saved, "stats": model.stats(),
        "negative": None if negative is None else {"blamed": blamed, **_save_negative(console, negative, negative_path(args))},
    }


def _print_changes(console: Console, changes: list[dict]) -> None:
    if not changes:
        console.say("the two sentences are the same: nothing to teach")
        return
    console.say()
    console.table(
        ["change", "the network wrote", "the teacher wrote"],
        [[change["op"], change["wrong"] or "-", change["right"] or "-"] for change in changes],
    )
# --------------------------------------------------------------------------
# the negative network (the failures, and why)
# --------------------------------------------------------------------------


def negative_path(args: argparse.Namespace) -> str:
    """``--negative``, else the negative model beside ``--model`` (``model.json`` -> ``model.negative.json``)."""
    path = getattr(args, "negative", None)
    if path:
        return path
    root, ext = os.path.splitext(args.model)
    if ext == ".gz":
        root, inner = os.path.splitext(root)
        ext = inner + ext
    return args.model if root.endswith(".negative") else f"{root}.negative{ext}"


def open_negative(args: argparse.Namespace, console: Console, *, required: bool) -> tuple[Any, Origin]:
    """Load the negative network from :func:`negative_path`, else create an empty one."""
    from .negative import NegativeNet

    path = negative_path(args)
    if os.path.isfile(path):
        model = load_model(path, backend=args.backend, device=args.device)
        if model.kind != "negative":
            raise CliError(f"{path} holds a {model.kind} model, not a negative one")
        g = model.graph
        return model, Origin("model", path, f"{fmt(g.total_blame)} blame over {len(g.reasons)} reasons")
    if required:
        raise CliError(
            f"negative model file not found: {path} "
            f"(teach it first with `{PROG} negative blame --text '...' --reason gibberish`)"
        )
    seed = effective_seed(args)
    return NegativeNet(seed=seed), Origin("new", None, f"seed {seed}")


def _negative_texts(args: argparse.Namespace, what: str) -> list[str]:
    texts = list(getattr(args, "text", None) or [])
    if getattr(args, "data", None):
        texts += read_texts([args.data], what=what)
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise CliError(f"no texts to {what}: give --text TEXT (repeatable) or --data FILE")
    return texts


def _reason_rows(model: Any, limit: int = 10) -> list[list[Any]]:
    return [[r["reason"], r["blame"], r["fails"], r["edges"], r["share"]] for r in model.reasons()[:limit]]


def cmd_negative_blame(args: argparse.Namespace, console: Console) -> dict:
    """Teach the negative network a failure: its reason, its severity and the tutor's own words."""
    texts = _negative_texts(args, "blame")
    model, origin = open_negative(args, console, required=False)
    out = args.out or negative_path(args)
    console.pairs([
        ("negative model", origin.describe()),
        ("failures", f"{len(texts)} texts"),
        ("reason", args.reason),
        ("severity", args.severity),
        ("source", args.source),
        ("note", clip(args.note, 60) if args.note else "-"),
        ("epochs", args.epochs),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console)
    stop = threading.Event()
    records, interrupted = run_interruptible(
        lambda: model.blame(
            texts, reason=args.reason, severity=args.severity, source=args.source, note=args.note,
            epochs=args.epochs, progress=printer, stop_event=stop,
        ),
        stop, console, "epoch",
    )
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    console.say()
    console.table(("reason", "blame", "fails", "edges", "share"), _reason_rows(model))
    return {
        "model": origin.to_dict(), "out": out, "texts": len(texts), "reason": args.reason,
        "severity": args.severity, "source": args.source, "records": records, "interrupted": interrupted,
        "saved": saved, "reasons": model.reasons(), "stats": model.stats(),
    }


def cmd_negative_clear(args: argparse.Namespace, console: Console) -> dict:
    """The tutor passed these texts: credit the edges they share with known failures, creating nothing."""
    texts = _negative_texts(args, "clear")
    model, origin = open_negative(args, console, required=True)
    out = args.out or negative_path(args)
    console.pairs([
        ("negative model", origin.describe()),
        ("cleared", f"{len(texts)} texts"),
        ("weight", args.weight),
        ("epochs", args.epochs),
        ("output", out),
    ])
    console.say()
    printer = EpochPrinter(console)
    stop = threading.Event()
    records, interrupted = run_interruptible(
        lambda: model.clear(texts, weight=args.weight, epochs=args.epochs, progress=printer, stop_event=stop),
        stop, console, "epoch",
    )
    saved = _finish_training(console, model, out, interrupted, printer, "epoch")
    last = records[-1] if records else {}
    console.say(f"matched {last.get('matched', 0)} of {len(texts)} texts ({last.get('edges_touched', 0)} edges cleared)")
    return {
        "model": origin.to_dict(), "out": out, "texts": len(texts), "weight": args.weight, "records": records,
        "interrupted": interrupted, "saved": saved, "stats": model.stats(),
    }


def _print_verdict(console: Console, verdict: dict, spans: bool = True) -> None:
    """One judgement as a block: the sentence, the reasons and the fragments to blame."""
    console.pairs([
        ("text", quote(clip(verdict["text"], 70))),
        ("verdict", verdict.get("decision") or verdict["verdict"]),
        ("risk", verdict["risk"]),
        ("coverage", verdict["coverage"]),
        ("blame", verdict["blame"]),
        *([("ratio", verdict["ratio"])] if "ratio" in verdict else []),
        ("why", verdict["why"]),
    ])
    if verdict["reasons"]:
        console.say()
        console.table(("reason", "blame", "share"), [[r["reason"], r["blame"], r["share"]] for r in verdict["reasons"]])
    if spans and verdict["spans"]:
        console.say()
        console.table(
            ("at", "fragment", "blame", "fails", "reason"),
            [[f"{s['start']}..{s['end']}", quote(s["fragment"]), s["blame"], s["fails"], s["reason"]] for s in verdict["spans"]],
        )


def cmd_negative_why(args: argparse.Namespace, console: Console) -> dict:
    """Why the negative network thinks a text went wrong: the reasons, and the fragments carrying them."""
    texts = _negative_texts(args, "judge")
    model, origin = open_negative(args, console, required=True)
    console.pairs([("negative model", origin.describe()), ("texts", len(texts))])
    verdicts = []
    for text in texts:
        verdict = model.judge(text, threshold=args.threshold, min_coverage=args.min_coverage, spans=args.spans)
        verdicts.append(verdict)
        console.say()
        _print_verdict(console, verdict)
    return {"model": origin.to_dict(), "verdicts": verdicts, "stats": model.stats()}


def cmd_negative_filter(args: argparse.Namespace, console: Console) -> dict:
    """The pair at work: the positive model writes, the negative one vetoes (or judge given texts)."""
    from .duo import FilterConfig, NegativeFilter

    negative, neg_origin = open_negative(args, console, required=True)
    positive, pos_origin = open_model(args, console, required=True)
    config = FilterConfig(
        threshold=args.threshold, min_coverage=args.min_coverage, ratio=None if args.no_ratio else args.ratio,
        peak=args.peak, over_sample=args.over_sample, strict=args.strict, spans=args.spans, learn=args.learn,
    )
    pair = NegativeFilter(positive, negative, config)
    given = list(args.text or [])
    if args.data:
        given += read_texts([args.data], what="filterable")
    console.pairs([
        ("positive model", f"{pos_origin.describe()} [{kind_label(positive)}]"),
        ("negative model", neg_origin.describe()),
        ("source", f"{len(given)} given texts" if given else f"{args.count} generated ({args.mode}, x{args.over_sample})"),
        ("threshold", f"risk >= {fmt(config.threshold if config.threshold is not None else negative.threshold)}"
                      f", coverage >= {fmt(config.min_coverage if config.min_coverage is not None else negative.min_coverage)}"),
        ("ratio", "off" if config.ratio is None else f">= {fmt(config.ratio)} nats/char"),
        ("peak", "off" if config.peak is None else f">= {fmt(config.peak)} blame on one fragment"),
        ("strict", config.strict),
        ("learn", config.learn),
    ])
    if given:
        outcome = pair.filter(given)
        doc: dict[str, Any] = {
            "texts": outcome["kept"], "kept": outcome["kept"], "verdicts": outcome["verdicts"],
            "rejected": [v for v in outcome["verdicts"] if v["decision"] == "reject"],
            "candidates": len(given), "asked": len(given), "rate": outcome["rate"],
        }
    else:
        doc = pair.generate(
            count=args.count, mode=args.mode, max_length=args.max_length, temperature=args.temperature,
            prefix=args.prefix, seed=args.seed, step_penalty=args.step_penalty,
        )
    console.say()
    console.table(
        ("decision", "rule", "risk", "peak", "ratio", "reason", "text"),
        [
            [v["decision"], v["rule"] or "-", v["risk"], v["peak"], v["ratio"],
             v["reasons"][0]["reason"] if v["reasons"] else "-", quote(clip(v["text"], 46))]
            for v in doc["verdicts"]
        ],
    )
    console.say()
    returned = doc.get("texts") or []
    console.say(
        f"{len(doc['verdicts'])} candidates: {len(doc['kept'])} passed the filter, {len(doc['rejected'])} vetoed"
        + (f" (acceptance {fmt(doc['rate'])})" if doc.get("rate") is not None else "")
        + (f"; returning the {len(returned)} cleanest" if len(returned) < len(doc["kept"]) else "")
    )
    for text in returned:
        console.say(f"  {quote(text)}")
    for verdict in doc["rejected"]:
        console.say(f"  vetoed: {quote(clip(verdict['text'], 60))} - {verdict['why']}")
    if config.learn:
        saved = save_model(negative, negative_path(args))
        console.say(f"saved {saved['path']} ({saved['bytes']} bytes)")
        doc["saved"] = saved
    doc["pair"] = pair.describe()
    return doc


def cmd_negative_reasons(args: argparse.Namespace, console: Console) -> dict:
    """Everything the tutor has blamed, and the journal of what it said."""
    model, origin = open_negative(args, console, required=True)
    stats = model.stats()
    console.pairs([
        ("negative model", origin.describe()),
        ("failures", stats["failures_total"]),
        ("blame", stats["edge_blame_total"]),
        ("cleared", stats["cleared_total"]),
        ("nodes / edges", f"{stats['nodes']} / {stats['edges']}"),
        ("judgements", f"{stats['judgements']} ({stats['rejected']} rejected)"),
        ("sources", ", ".join(f"{k}={v}" for k, v in sorted(stats["sources"].items())) or "-"),
        ("filter", f"risk >= {fmt(stats['threshold'])}, coverage >= {fmt(stats['min_coverage'])}"),
    ])
    console.say()
    console.table(("reason", "blame", "fails", "edges", "share"), _reason_rows(model, limit=args.limit))
    journal = model.recent(args.log)
    if journal:
        console.say()
        console.table(
            ("when", "reason", "severity", "source", "text", "the tutor said"),
            [[e["at"], e["reason"], e["severity"], e["source"] or "-", quote(clip(e["text"], 34)), clip(e["note"], 44)]
             for e in journal],
        )
    return {"model": origin.to_dict(), "reasons": model.reasons(), "journal": journal, "stats": stats}


def cmd_negative_forget(args: argparse.Namespace, console: Console) -> dict:
    """The tutor can be wrong too: drop (or fade) the blame behind one reason, or all of it."""
    model, origin = open_negative(args, console, required=True)
    out = args.out or negative_path(args)
    result = model.forget(args.reason, factor=args.factor)
    saved = save_model(model, out)
    console.pairs([
        ("negative model", origin.describe()),
        ("reason", result["reason"]),
        ("keeping", f"{fmt(args.factor * 100)}% of its blame"),
        ("edges", result["edges"]),
        ("blame removed", result["blame_removed"]),
        ("saved", f"{saved['path']} ({saved['bytes']} bytes)"),
    ])
    console.say()
    console.table(("reason", "blame", "fails", "edges", "share"), _reason_rows(model))
    return {"model": origin.to_dict(), "out": out, **result, "saved": saved, "stats": model.stats()}


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
    discriminator: GraphModel | None = None
    disc_origin: Origin | None = None
    if args.discriminator:
        if os.path.isfile(args.discriminator):
            discriminator = load_model(args.discriminator, backend=args.backend, device=args.device)
            disc_origin = Origin("model", args.discriminator, f"{discriminator.meta['epochs_total']} epochs trained")
        else:
            disc_origin = Origin("new", None, f"seed {effective_seed(args) + 1}, saved to {args.discriminator}")
    config = EvolveConfig(
        samples=args.samples, real_per_generation=args.real_per_generation, max_length=args.max_length,
        temperature=args.temperature, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs,
        neg_lr=args.neg_lr, pos_lr=args.pos_lr, disc_neg_epochs=args.disc_neg_epochs,
        disc_pos_epochs=args.disc_pos_epochs, batch_size=args.batch_size, checkpoint_every=every,
        seed=effective_seed(args),
        blatant_mode=args.blatant_mode, blatant_margin=args.blatant_margin, blatant_boost=args.blatant_boost,
    )
    negative = neg_origin = None
    if args.blame:
        negative, neg_origin = open_negative(args, console, required=False)
    evolver = Evolver(generator, corpus, discriminator, config, negative=negative)
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
        *([("negative model", f"{neg_origin.describe()} -> {negative_path(args)}")] if negative is not None else []),
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
    negative_doc = _save_negative(console, negative, negative_path(args)) if negative is not None else None
    return {
        "negative": negative_doc,
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
        ("kind", kind_label(model)),
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
        *([("rewards", f"+{fmt(stats['rewards_total'])} / -{fmt(stats['penalties_total'])} over "
                       f"{stats['feedback_passes']} feedback pass(es); edge rewards +{fmt(stats['edge_reward_positive'])} "
                       f"/ {fmt(stats['edge_reward_negative'])}"),
           ("traversals", f"{stats['total_traversals']} total, {stats['window_traversals']} inside the sliding window "
                          f"of {stats['window']}"),
           ("weights", f"global_scale={fmt(stats['global_scale'])} window_scale={fmt(stats['window_scale'])} "
                       f"reward_scale={fmt(stats['reward_scale'])} count_scale={fmt(stats['count_scale'])}")]
          if model.kind == "count" else []),
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
    model = load_model(record["path"], backend=args.backend, device=args.device)
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
    from .chatgpt import api_key_configured
    from .codegen import PHASES, CodeGenConfig, CodeGenTrainer, Sandbox, default_teacher_model, load_problems
    from .llm import LLMError, make_client

    try:
        problems = load_problems(args.problems)
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot load problems from {args.problems}: {exc}") from exc
    phases = PHASES if args.phase == "both" else (args.phase,)
    manager = checkpoint_manager(args)
    judge_provider = args.judge_provider or args.teacher_provider
    config = CodeGenConfig(
        teacher_provider=args.teacher_provider, judge_provider=judge_provider,
        teacher_model=args.teacher_model or default_teacher_model(args.teacher_provider),
        judge_model=args.judge_model, phases=phases,
        rounds=args.rounds, teacher_attempts=args.teacher_attempts, model_attempts=args.model_attempts,
        first_attempt_dijkstra=not args.sample_first, temperature=args.temperature, max_length=args.max_length,
        strictness=args.strictness, use_judge=not args.no_judge, fallback_teacher=not args.no_fallback_teacher,
        twonrl_per=args.twonrl_per, replay=not args.no_replay, teacher_prompt=args.teacher_prompt,
        model_prompt=args.model_prompt if args.model_prompt is not None else CodeGenConfig().model_prompt,
        neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr, pos_lr=args.pos_lr,
        batch_size=args.batch_size, checkpoint_every=checkpoint_every(args, manager),
    )
    providers = {config.teacher_provider} | ({config.judge_provider} if config.use_judge else set())
    if "chatgpt" in providers and not api_key_configured():
        raise CliError(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT tutor, "
            "or use --teacher-provider ollama"
        )
    try:
        config.validate()
        client = make_client(config.teacher_provider, args.url, config.teacher_model, args.timeout)
        if config.judge_provider == config.teacher_provider and not args.judge_url:
            judge_client = client
        else:
            judge_client = make_client(config.judge_provider, args.judge_url, config.resolved_judge_model, args.timeout)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    sandbox = Sandbox(timeout=args.sandbox_timeout, memory_mb=args.memory_mb, isolate_network=not args.no_network_isolation)
    model, origin = open_model(args, console, required=False)
    negative = neg_origin = None
    if args.blame:
        negative, neg_origin = open_negative(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("problems", f"{len(problems)} from {args.problems}"),
        ("phases", " -> ".join(phases) + f", {config.rounds} round(s)"),
        ("teacher", f"{config.teacher_provider}: {config.teacher_model} at {client.url}"),
        ("judge", f"{config.judge_provider}: {config.resolved_judge_model} at {judge_client.url}" if config.use_judge
                  else "off (sandbox, expected output and tests only)"),
        ("attempts", f"teacher={config.teacher_attempts} model={config.model_attempts}"
                     + ("" if config.fallback_teacher else ", no teacher fallback")),
        ("sandbox", f"timeout={sandbox.timeout:g}s memory={sandbox.memory_mb}MB network="
                    + ("isolated" if sandbox.network_isolated else "NOT isolated")),
        ("2NRL", f"per {config.twonrl_per}: negative epochs={config.neg_epochs} lr={config.neg_lr}, positive epochs={config.pos_epochs} "
                 f"lr={config.pos_lr}, batch={config.batch_size}, replay={'on' if config.replay else 'off'}, {config.strictness}"),
        *([("negative model", f"{neg_origin.describe()} -> {negative_path(args)}")] if negative is not None else []),
        ("output", out),
    ])
    console.say()
    printer = ProblemPrinter(console)
    stop = threading.Event()
    trainer = CodeGenTrainer(model, client, sandbox, config, judge_client=judge_client)
    trainer = CodeGenTrainer(model, client, sandbox, config, negative=negative)
    try:
        records, interrupted = run_interruptible(
            lambda: trainer.run(problems, progress=printer, stop_event=stop, checkpoint_manager=manager),
            stop, console, "problem",
        )
    except LLMError as exc:
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
    if negative is not None:
        doc["negative"] = _save_negative(console, negative, negative_path(args))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({k: doc[k] for k in ("problems", "config", "records", "solutions", "solved", "model_solved")}, fh, indent=2)
        console.say(f"wrote report to {args.report}")
    return doc


def cmd_tutor(args: argparse.Namespace, console: Console) -> dict:
    from .chatgpt import api_key_configured
    from .llm import LLMError, make_client
    from .tutor import TutorConfig, TutorTrainer, default_tutor_model, report_card

    manager = checkpoint_manager(args)
    grader_provider = args.grader_provider or args.tutor_provider
    config = TutorConfig(
        topic=args.topic, rounds=args.rounds, exercises=args.exercises, attempts=args.attempts, focus=args.focus,
        level=args.level, words=args.words, brief=args.brief or "",
        tutor_provider=args.tutor_provider, grader_provider=grader_provider,
        tutor_model=args.tutor_model or default_tutor_model(args.tutor_provider),
        grader_model=args.grader_model, mode=args.mode, length=args.length, max_length=args.max_length,
        temperature=args.temperature, to_end=not args.no_to_end, beam=args.beam, threshold=args.threshold,
        grammar_weight=args.grammar_weight, batch=args.batch, adapt=not args.no_adapt, drills=args.drills,
        plan=args.plan or 0,
        teach_answer=not args.no_teach_answer, learn=not args.dry_run, twonrl_per=args.twonrl_per,
        diff_corrections=not args.no_diff_corrections, keep_weight=args.keep_weight, min_weight=args.min_weight, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr,
        pos_lr=args.pos_lr, batch_size=args.batch_size, strength=args.strength, replay=not args.no_replay,
        replay_limit=args.replay_limit, checkpoint_every=checkpoint_every(args, manager),
    )
    if "chatgpt" in (config.tutor_provider, config.grader_provider) and not api_key_configured():
        raise CliError(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT teach, "
            "or use --tutor-provider ollama"
        )
    try:
        config.validate()
        client = make_client(config.tutor_provider, args.url, config.tutor_model, args.timeout)
        if config.grader_provider == config.tutor_provider and not args.grader_url:
            grader_client = client
        else:
            grader_client = make_client(config.grader_provider, args.grader_url, config.resolved_grader_model, args.timeout)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    model, origin = open_model(args, console, required=False)
    out = args.out or args.model
    console.pairs([
        ("model", origin.describe()),
        ("backend", backend_label(model)),
        ("topic", config.topic + (f", drilling {config.focus}" if config.focus else "")),
        ("lessons", f"{config.rounds} round(s) x {config.exercises} exercise(s) x {config.attempts} attempt(s)"),
        ("teacher", f"{config.tutor_provider}: {config.tutor_model} at {client.url}"),
        ("marker", f"{config.grader_provider}: {config.resolved_grader_model} at {grader_client.url}"),
        ("completion", f"{config.mode}, length={config.length}, max={config.max_length}"
                       + (f", temperature={fmt(config.temperature)}" if config.mode == "sample" else "")),
        ("marking", f"pass at {fmt(config.threshold)}/10, grammar weight {fmt(config.grammar_weight)}, "
                    f"{config.batch} per call" + (", adapting to the weakest points" if config.adapt else "")),
        ("corrections", "from the diff with what the network wrote: only what changed moves"
                        f" (the rest keeps {fmt(config.keep_weight)})" if config.diff_corrections
                        else "as whole sentences (--no-diff-corrections)"),
        ("2NRL", "off (--dry-run: the grades are reported, nothing is trained)" if args.dry_run else
                 f"per {config.twonrl_per}: negative epochs={config.neg_epochs} lr={config.neg_lr}, positive "
                 f"epochs={config.pos_epochs} lr={config.pos_lr}, batch={config.batch_size}, "
                 f"garbage weight {fmt(config.min_weight)}..1"),
        ("brief", clip(config.brief, 100) if config.brief else "none (--brief TEXT: the last plan's prompt)"),
        ("plan", f"the teacher plans the next {config.plan} lesson(s) from the final report card"
                 if config.plan else "no lesson plan (--plan N)"),
        ("output", "not saved (--dry-run)" if args.dry_run else out),
    ])
    console.say()
    printer = LessonPrinter(console)
    stop = threading.Event()
    trainer = TutorTrainer(model, client, config, grader_client=grader_client)
    negative = neg_origin = None
    if args.blame:
        negative, neg_origin = open_negative(args, console, required=False)
        console.pairs([("negative model", f"{neg_origin.describe()} -> {negative_path(args)}")])
        console.say()
    trainer = TutorTrainer(model, client, config, negative=negative)
    try:
        records, interrupted = run_interruptible(
            lambda: trainer.run(progress=printer, stop_event=stop, checkpoint_manager=manager), stop, console, "round",
        )
    except LLMError as exc:
        raise CliError(str(exc)) from exc
    saved = None
    if args.dry_run:
        if interrupted:
            console.note(f"stopped after {printer.seen} lesson(s)")
        console.say()
    else:
        saved = _finish_training(console, model, out, interrupted, printer, "lesson")
    card = report_card(trainer.lessons)
    console.say()
    console.pairs([
        ("lessons", f"{card['passed']}/{card['lessons']} passed"
                    + (f" ({100 * card['pass_rate']:.0f}%)" if card["pass_rate"] is not None else "")),
        ("mean score", f"{fmt(card['mean_score'])}/10"),
        ("grammar", fmt(card["mean_grammar"])),
        ("spelling", fmt(card["mean_spelling"])),
        ("fluency", fmt(card["mean_fluency"])),
        ("mistakes", ", ".join(f"{name} x{count}" for name, count in card["errors"].items()) or "none"),
        ("weakest", ", ".join(card["weakest"]) or "-"),
    ])
    plan = next((record for record in reversed(records) if record.get("kind") == "plan"), None)
    if plan is not None:
        upgrade = plan.get("upgrade") or {}
        console.say()
        console.say(f"lesson plan ({plan['source']}): {plan['summary']}")
        console.table(
            ("#", "focus", "fixes", "topic", "exercises", "drills", "why"),
            [
                [i, lesson["focus"] or "-", lesson["targets"], lesson["topic"] or "-", lesson["exercises"],
                 lesson["drills"], clip(str(lesson["why"] or "-"), 60)]
                for i, lesson in enumerate(plan["lessons"], 1)
            ],
        )
        console.pairs([
            ("step up", f"{upgrade.get('step', '-')}: {upgrade.get('note', '-')}"),
            ("brief", plan.get("prompt") or "-"),
        ])
        first = plan["lessons"][0]
        console.say(
            f"teach the next batch: {PROG} tutor --topic {quote(first['topic'] or config.topic)}"
            + f" --level {upgrade.get('level', plan['level'])} --words {quote(str(upgrade.get('words', config.words)))}"
            + f" --threshold {float(upgrade.get('threshold', config.threshold)):g}"
            + f" --exercises {first['exercises']} --drills {upgrade.get('drills', config.drills)}"
            + f" --brief {quote(plan.get('prompt') or '')}"
        )
    doc = {
        "model": origin.to_dict(), "out": None if args.dry_run else out, "config": config.to_dict(),
        "records": records, "lessons": [lesson.to_dict() for lesson in trainer.lessons], "report": card,
        "plan": plan, "interrupted": interrupted, "saved": saved, "stats": model.stats(),
        "negative": _save_negative(console, negative, negative_path(args)) if negative is not None else None,
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({k: doc[k] for k in ("config", "records", "lessons", "report", "plan")}, fh, indent=2)
        console.say(f"wrote report to {args.report}")
    return doc
def _save_negative(console: Console, negative: Any, path: str) -> dict:
    """Save the negative network and print what the tutor taught it."""
    saved = save_model(negative, path)
    console.say()
    console.say(f"negative model: {saved['path']} ({saved['bytes']} bytes)")
    console.table(("reason", "blame", "fails", "edges", "share"), _reason_rows(negative))
    return {"path": path, "saved": saved, "reasons": negative.reasons(), "stats": negative.stats()}


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
    doc["negative"] = None
    if args.blame:
        from . import blame

        negative, neg_origin = open_negative(args, console, required=False)
        console.say()
        console.pairs([("negative model", neg_origin.describe()), ("blaming", f"{len(result['bad'])} failed texts")])
        report = blame.teach_reviews(negative, result, threshold=args.threshold, source="review")
        console.say(
            f"blamed {report['blamed']} texts over {report['edges']} edges "
            f"(mean severity {fmt(report['severity_mean'])}), cleared {report['cleared']} of {report['passed']} passed"
        )
        doc["negative"] = {
            "blamed": report["blamed"], "cleared": report["cleared"], "edges": report["edges"],
            "reasons": report["reasons"], "lessons": report["lessons"],
            **_save_negative(console, negative, negative_path(args)),
        }
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


def _chatgpt_client(args: argparse.Namespace) -> Any:
    from .chatgpt import ChatGPTClient, api_key_configured

    if not api_key_configured():
        raise CliError(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE, a file holding it) and try again"
        )
    try:
        return ChatGPTClient(args.url, args.chatgpt_model, args.timeout)
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def cmd_chatgpt_models(args: argparse.Namespace, console: Console) -> dict:
    from .chatgpt import ChatGPTError

    client = _chatgpt_client(args)
    try:
        models = client.models()
    except ChatGPTError as exc:
        raise CliError(str(exc)) from exc
    console.pairs([("endpoint", client.url), ("default model", client.model), ("models", len(models))])
    console.say()
    if models:
        console.table(("name", "owner", "created"), [[m["name"], m.get("owned_by"), _epoch(m.get("created"))] for m in models])
    else:
        console.say("the key has access to no models")
    return {"url": client.url, "model": client.model, "models": models}


def cmd_chatgpt_ask(args: argparse.Namespace, console: Console) -> dict:
    from .chatgpt import ChatGPTError

    client = _chatgpt_client(args)
    console.pairs([("model", f"{client.model} at {client.url}"), ("prompt", quote(clip(args.prompt, 60)))])
    console.say()
    try:
        answer = client.generate(
            args.prompt, system=args.system, json_mode=args.json_answer, options={"temperature": args.temperature}
        )
    except ChatGPTError as exc:
        raise CliError(str(exc)) from exc
    console.say(answer.strip())
    return {"url": client.url, "model": client.model, "prompt": args.prompt, "answer": answer}


def _epoch(value: Any) -> Any:
    """A unix timestamp as a date (anything else unchanged)."""
    if isinstance(value, (int, float)) and value > 0:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m-%d")
    return value


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
        "kind": effective_kind(args),
        "checkpoint_dir": args.checkpoint_dir,
        "upload_dir": args.upload_dir,
        "frontend_dir": frontend_dir,
        "frontend_built": os.path.isfile(os.path.join(frontend_dir, "index.html")),
        "backend": args.backend,
        "device": args.device,
        "seed": effective_seed(args),
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "chatgpt_url": args.chatgpt_url,
        "chatgpt_model": args.chatgpt_model,
        "chatgpt_configured": _chatgpt_configured(),
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
        ("chatgpt", f"{args.chatgpt_model or '$RADIXNET_OPENAI_MODEL'} at {args.chatgpt_url or '$OPENAI_BASE_URL'}"
                    + (" (key set)" if doc["chatgpt_configured"] else " (no OPENAI_API_KEY: ChatGPT tutoring is off)")),
    ])
    console.say("press Ctrl-C to stop")
    if console.json_mode:
        console.emit(doc)  # the server blocks, so the document goes out first
    api.run_server(
        host=args.host, port=args.port, model_path=args.model, checkpoint_dir=args.checkpoint_dir,
        frontend_dir=frontend_dir, backend=args.backend, device=args.device, seed=effective_seed(args),
        upload_dir=args.upload_dir, ollama_url=args.ollama_url, ollama_model=args.ollama_model,
        chatgpt_url=args.chatgpt_url, chatgpt_model=args.chatgpt_model, kind=getattr(args, "kind", None),
    )
    return None


def _chatgpt_configured() -> bool:
    from .chatgpt import api_key_configured

    return api_key_configured()


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
    group.add_argument("--kind", choices=KINDS, default=default(None),
                       help="algorithm of a NEW model: radix = the sine-activation network (default), count = the count / "
                            "reward model (edge weight = log(1 + traversals) + rewards, top-K / bottom-K prediction), "
                            "negative = the negative network (failures only, blamed with the tutor's reasons); a "
                            "loaded file's own kind always wins.  With --kind count / negative the default --model is "
                            f"{DEFAULT_COUNT_MODEL} / {DEFAULT_NEGATIVE_MODEL}")
    group.add_argument("--json", action="store_true", default=default(False),
                       help="print one JSON document on stdout instead of tables (progress goes to stderr)")


def _add_negative_option(parser: argparse.ArgumentParser, top_level: bool) -> None:
    """``--negative PATH``; the action copies suppress their default so the group's value survives."""
    parser.add_argument(
        "--negative", metavar="PATH", default=None if top_level else argparse.SUPPRESS,
        help=f"negative model file (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)",
    )


def _add_checkpoint_options(parser: argparse.ArgumentParser, unit: str) -> None:
    group = parser.add_argument_group("checkpoint options")
    group.add_argument("--checkpoint-dir", metavar="DIR", help="write rotated checkpoints into DIR")
    group.add_argument("--checkpoint-every", type=nonneg_int, metavar="N",
                       help=f"checkpoint every N {unit}s (default: every {unit} when --checkpoint-dir is given, else off)")
    group.add_argument("--keep", type=pos_int, default=5, help="checkpoints to keep in --checkpoint-dir")


def _add_asr_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("transcription options")
    group.add_argument("--backend", choices=SPEECH_BACKENDS, default="auto",
                       help="speech-to-text backend (auto: a given transcript, then faster-whisper, whisper, "
                            "a configured server)")
    group.add_argument("--language", metavar="CODE", help="language hint for the backend, e.g. en")
    group.add_argument("--asr-model", metavar="NAME",
                       help="Whisper model size / the server's model name ($RADIXNET_WHISPER_MODEL, $RADIXNET_ASR_MODEL)")
    group.add_argument("--asr-url", metavar="URL",
                       help="OpenAI-compatible /v1/audio/transcriptions endpoint ($RADIXNET_ASR_URL)")


def _add_speech_teach_options(parser: argparse.ArgumentParser) -> None:
    """Everything `speech teach` and `speech listen` share: the transcript, the waveform, the token, training."""
    _add_asr_options(parser)
    parser.add_argument("--text", metavar="TEXT",
                        help="the transcript (what you said); skips the transcription backends")
    group = parser.add_argument_group("waveform options")
    group.add_argument("--rate", type=pos_int, default=SPEECH_RATE,
                       help="resample the waveform to this many samples per second before quantising")
    group.add_argument("--codec", choices=("auto", "mu", "pcm8"), default="auto",
                       help="waveform quantisation (auto = mu-law)")
    group.add_argument("--normalise", action="store_true", help="scale a quiet recording up to full range first")
    group.add_argument("--no-waveform", action="store_true", help="learn the transcript only, not the sound")
    group.add_argument("--pair", action="store_true",
                       help="also learn one text of the waveform followed by its transcript (sound -> words)")
    group = parser.add_argument_group("token options")
    group.add_argument("--token", metavar="TEXT", help="use this token instead of <speech:digest>")
    group.add_argument("--shared-token", action="store_true",
                       help=f"use the plain {SPEECH_TOKEN_HELP} for every utterance instead of a unique one")
    parser.add_argument("--out", metavar="PATH", help="write the texts (one per line) to this file")
    group = parser.add_argument_group("training options")
    group.add_argument("--train", action="store_true", help="train the model on the texts, then save it")
    group.add_argument("--epochs", type=nonneg_int, default=3, help="training epochs with --train")
    group.add_argument("--lr", type=nonneg_float, default=0.5, help="learning rate with --train")
    group.add_argument("--batch-size", type=pos_int, default=8, help="batch size with --train")
    group.add_argument("--model-out", metavar="PATH", help="where to save the model with --train (default: --model)")


def _add_two_nrl_options(parser: argparse.ArgumentParser, neg_epochs: int, pos_epochs: int, batch_size: int) -> None:
    group = parser.add_argument_group("2NRL options")
    group.add_argument("--neg-epochs", type=nonneg_int, default=neg_epochs, help="epochs of the negative (garbage) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=pos_epochs, help="epochs of the positive (fine-tune) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.05, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.01,
                       help="learning rate of the positive phase (activation parameters use a tenth of it)")
    group.add_argument("--batch-size", type=pos_int, default=batch_size, help="transitions per backend step in both phases")
    group.add_argument("--strength", type=nonneg_float, default=1.0,
                       help="count model: reward / penalty added to every edge of a path per pass (radix: ignored)")


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
                   help="training text files, one text per line (blank lines are skipped); a .zip archive contributes "
                        "every text file inside it")
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
    p.add_argument("--reverse-schedule", action="store_true",
                   help="play the schedules backwards: the last epoch's rates first (a ramp up becomes a ramp down)")
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
    p.add_argument("--mode", choices=PREDICT_MODES, default="dijkstra",
                   help="search strategy: dijkstra = the exact cheapest path (the count model treats it as beam), "
                        "beam = the top-K and bottom-K continuations, sample = one stochastic walk")
    p.add_argument("--k", type=nonneg_int, default=5, help="beam: continuations per side (top K and bottom K)")
    p.add_argument("--beam", type=pos_int, metavar="N", help="beam: beam width (default: max(4k, 16))")
    p.add_argument("--to-end", action="store_true", help="dijkstra: cheapest path all the way to the end of a text")
    p.add_argument("--step-penalty", type=nonneg_float, default=0.0, help="dijkstra: extra cost per edge (prefers short paths)")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="sample: softmax temperature (0 = greedy)")
    p.set_defaults(handler=cmd_predict)

    # generate -------------------------------------------------------------
    p = command(
        "generate", "generate whole texts with the prediction search",
        "Generate texts from the start node, or continuing --prefix: beam runs the prediction search to the\n"
        "end of a text and returns the --count most likely complete texts; sample draws --count stochastic\n"
        "walks (--seed makes them reproducible); dijkstra returns the single cheapest complete text.",
    )
    p.add_argument("--count", type=nonneg_int, default=1, help="texts to generate (beam: the K most likely)")
    p.add_argument("--max-length", type=nonneg_int, default=60, help="maximum characters per text")
    p.add_argument("--mode", choices=("beam", "sample", "dijkstra"), default="sample", help="generation strategy")
    p.add_argument("--prefix", default="", metavar="TEXT", help="start every text with this (default: from START)")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="sample: softmax temperature (0 = greedy)")
    p.add_argument("--step-penalty", type=nonneg_float, default=0.0, help="beam / dijkstra: extra cost per edge")
    p.add_argument("--beam", type=pos_int, metavar="N", help="beam: beam width (default: max(4 * count, 16))")
    p.set_defaults(handler=cmd_generate)

    # converse -------------------------------------------------------------
    p = command(
        "converse", "the model converses with itself",
        "Two voices take turns; every reply is the prediction search picking up the last words of the\n"
        "previous line (--context characters, at a word boundary) and continuing them to the end of a\n"
        "text.  beam speaks the most likely continuation the conversation has not heard yet; sample draws\n"
        "stochastic walks.  When nothing follows, the context loses a word at a time and finally the voice\n"
        "changes the subject with a fresh text.  --partner FILE lets a second model answer.",
    )
    p.add_argument("--opening", default="", metavar="TEXT", help="the first line, spoken as given (default: a fresh text)")
    p.add_argument("--turns", type=nonneg_int, default=6, help="turns to generate")
    p.add_argument("--mode", choices=("beam", "sample"), default="beam", help="how a reply is found")
    p.add_argument("--max-length", type=nonneg_int, default=60, help="characters a reply may add to its context")
    p.add_argument("--context", type=nonneg_int, default=12, help="characters of the previous line a reply picks up")
    p.add_argument("--k", type=pos_int, default=5, help="candidates considered per turn (beam: the K most likely)")
    p.add_argument("--beam", type=pos_int, metavar="N", help="beam: beam width (default: max(4 * k, 16))")
    p.add_argument("--temperature", type=nonneg_float, default=1.0, help="sample: softmax temperature (0 = greedy)")
    p.add_argument("--step-penalty", type=nonneg_float, default=0.0, help="beam: extra cost per edge")
    p.add_argument("--speakers", default=",".join(DEFAULT_SPEAKERS), metavar="A,B", help="names of the voices")
    p.add_argument("--partner", metavar="FILE", help="a second model file that speaks the second voice")
    p.add_argument("--allow-repeats", action="store_true", help="do not skip continuations the conversation already heard")
    p.set_defaults(handler=cmd_converse)

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
    p.add_argument("--reverse-schedule", action="store_true", help="play the schedules backwards (last epoch first)")
    p.add_argument("--epochs", type=nonneg_int, default=10, help="epochs to preview")
    p.add_argument("--lr", type=nonneg_float, default=TrainConfig.lr, help="base learning rate (lr0)")
    p.add_argument("--act-lr", type=nonneg_float, default=TrainConfig.act_lr, help="base activation learning rate (act_lr0)")
    p.set_defaults(handler=cmd_schedule)

    # image ----------------------------------------------------------------
    p = command(
        "image", "images as text: the Stable Diffusion VAE run backwards, quantised, base64-encoded",
        "Encode an image into a text the network can train on and predict: the Stable Diffusion VAE's\n"
        "encoder (the inverse of image generation) turns it into a 4 x H/8 x W/8 latent, every number\n"
        "becomes one signed byte and the bytes become base64 - `img:sd:128x128:AAAA...`.  `decode` runs the\n"
        "forward process again (text -> latent -> VAE decoder -> PNG).  Needs pillow; the sd encoder also\n"
        "needs torch + diffusers and the VAE weights ($RADIXNET_SD_VAE, default stabilityai/sd-vae-ft-mse);\n"
        "without them a thumbnail stand-in with the same 8x reduction is used (--encoder tiny / auto).",
    )
    actions = p.add_subparsers(dest="action", metavar="<action>", title="actions", required=True)
    a = actions.add_parser("info", help="which encoders are available", description="Report the image encoders and their dependencies.",
                           formatter_class=_HelpFormatter)
    a.set_defaults(handler=cmd_image_info)
    a = actions.add_parser(
        "encode", help="encode an image file as text (optionally train on it)",
        description="Encode FILE as text.  --out writes the text to a file; --train trains the model on it and saves.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("file", metavar="FILE", help="image file (PNG, JPEG, WebP, ... whatever Pillow reads)")
    a.add_argument("--size", type=pos_int, default=128, help="resize to SIZE x SIZE (a multiple of 8) before encoding")
    a.add_argument("--encoder", choices=("auto", "sd", "tiny"), default="auto", help="the encoder (auto = sd when it loads)")
    a.add_argument("--out", metavar="PATH", help="write the encoded text to this file")
    a.add_argument("--train", action="store_true", help="train the model on the encoded text, then save it")
    a.add_argument("--epochs", type=nonneg_int, default=3, help="training epochs with --train")
    a.add_argument("--lr", type=nonneg_float, default=0.5, help="learning rate with --train (one long text, so it is high)")
    a.add_argument("--batch-size", type=pos_int, default=8, help="batch size with --train")
    a.add_argument("--model-out", metavar="PATH", help="where to save the model with --train (default: --model)")
    a.set_defaults(handler=cmd_image_encode)
    a = actions.add_parser(
        "decode", help="decode an encoded or predicted text back to a PNG",
        description="Turn `img:<encoder>:<w>x<h>:<base64>` back into an image (a cut-off tail is padded).",
        formatter_class=_HelpFormatter,
    )
    source = a.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", metavar="TEXT", help="the encoded text")
    source.add_argument("--data", metavar="FILE", help="a file holding the encoded text")
    a.add_argument("--encoder", choices=("sd", "tiny"), help="override the encoder named in the text")
    a.add_argument("--out", required=True, metavar="PNG", help="where to write the image")
    a.set_defaults(handler=cmd_image_decode)

    # speech ---------------------------------------------------------------
    p = command(
        "speech", "teach the model by talking to it: the transcript and the waveform behind one unique token",
        "Speech to text plus the sound itself.  One utterance becomes two texts that start with the\n"
        "same unique token - `<speech:9f2a1c7d> the cat sat on the mat` and\n"
        "`<speech:9f2a1c7d> aud:mu:8000x1:<base64>` - so the words and the waveform leave the same node\n"
        "of the graph.  The transcript comes from faster-whisper / openai-whisper / an OpenAI-compatible\n"
        "/v1/audio/transcriptions server, or from --text (what the browser's Speech tab dictated, or what\n"
        "you know you said); the waveform is mixed to mono, resampled and quantised to one mu-law byte per\n"
        "sample.  `listen` records from the microphone first, `decode` turns a predicted waveform back into\n"
        "a WAV file.  WAV is read directly; other formats need ffmpeg.",
    )
    actions = p.add_subparsers(dest="action", metavar="<action>", title="actions", required=True)
    a = actions.add_parser("info", help="which transcription backends, recorders and codecs are available",
                           description="Report the speech backends and their dependencies.", formatter_class=_HelpFormatter)
    a.set_defaults(handler=cmd_speech_info)

    a = actions.add_parser(
        "transcribe", help="speech to text: print what an audio file says",
        description="Transcribe FILE and print the words (nothing is trained; use `teach` for that).",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("file", metavar="FILE", help="audio file (WAV directly; MP3, M4A, WebM, Ogg, FLAC ... need ffmpeg)")
    _add_asr_options(a)
    a.add_argument("--text", metavar="TEXT", help="a transcript you already have (used instead of a backend)")
    a.add_argument("--out", metavar="PATH", help="write the transcript to this file")
    a.set_defaults(handler=cmd_speech_transcribe)

    a = actions.add_parser(
        "teach", help="teach the model an audio file: the transcript and the waveform",
        description="Transcribe FILE, encode its waveform, and (with --train) learn both texts.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("file", metavar="FILE", help="audio file (WAV directly; other formats need ffmpeg)")
    _add_speech_teach_options(a)
    a.set_defaults(handler=cmd_speech_teach)

    a = actions.add_parser(
        "listen", help="record from the microphone, then teach the model what was said",
        description="Record --seconds of audio (arecord, rec / sox or ffmpeg), then run the `teach` flow on it.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--seconds", type=nonneg_float, default=5.0, help="how long to record")
    a.add_argument("--record-rate", type=pos_int, default=16000, help="sample rate to record at (before resampling)")
    a.add_argument("--recorder", choices=SPEECH_RECORDERS, help="force one recorder (default: the first installed)")
    a.add_argument("--save", metavar="WAV", help="also keep the recording in this file")
    _add_speech_teach_options(a)
    a.set_defaults(handler=cmd_speech_listen)

    a = actions.add_parser(
        "decode", help="turn an encoded or predicted waveform text back into a WAV file",
        description="Turn `aud:<codec>:<rate>x<channels>:<base64>` back into audio (a cut-off tail is padded).",
        formatter_class=_HelpFormatter,
    )
    source = a.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", metavar="TEXT", help="the encoded text")
    source.add_argument("--data", metavar="FILE", help="a file holding the encoded text")
    a.add_argument("--codec", choices=("mu", "pcm8"), help="override the codec named in the text")
    a.add_argument("--out", required=True, metavar="WAV", help="where to write the audio")
    a.set_defaults(handler=cmd_speech_decode)

    # weights --------------------------------------------------------------
    p = command(
        "weights", "count model: show or change the dual frequency weight function",
        "The count / reward model weighs an edge by its share of its node's traversals - all time and inside a\n"
        "sliding window of the last --window traversals - plus its rewards:\n"
        "  weight = global_scale * log(R_all) + window_scale * log(R_recent) + reward_scale * reward\n"
        "         (+ count_scale * log(1 + count), off by default).\n"
        "Without options the current function and the tracked totals are shown; with options the model is\n"
        "changed, every weight recomputed and the model saved.",
    )
    p.add_argument("--count-scale", type=float, metavar="X", help="weight of log(1 + traversals)")
    p.add_argument("--global-scale", type=float, metavar="X", help="weight of the all-time share log(R_all)")
    p.add_argument("--window-scale", type=float, metavar="X", help="weight of the sliding-window share log(R_recent)")
    p.add_argument("--reward-scale", type=float, metavar="X", help="weight of the rewards")
    p.add_argument("--window", type=pos_int, metavar="N", help="traversals the sliding window remembers")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_weights)

    # feedback -------------------------------------------------------------
    p = command(
        "feedback", "learn from rated texts (thumbs up / thumbs down) with 2NRL",
        "Thumbs up (--good / --good-text) are correct texts, thumbs down (--bad / --bad-text) garbage.  Both:\n"
        "2NRL (train on the bad texts, invert, fine-tune on the good ones).  Only good: reward (a positive-phase\n"
        "pass).  Only bad: punish (a negative-phase pass, then the network is inverted so those texts become\n"
        "unlikely).  The count / reward model penalises / rewards the rated paths by --strength instead (no\n"
        "inversion).  A rating is more than a thumb: --good-ratings / --bad-ratings give a mark out of 10 per\n"
        "text and the network learns each one in proportion to it.  The same rule the frontend's Generate tab\n"
        "uses for its ratings.",
    )
    p.add_argument("--good", metavar="FILE", help="thumbs-up texts, one per line")
    p.add_argument("--bad", metavar="FILE", help="thumbs-down texts, one per line")
    p.add_argument("--good-text", action="append", metavar="TEXT", help="a thumbs-up text (repeatable)")
    p.add_argument("--bad-text", action="append", metavar="TEXT", help="a thumbs-down text (repeatable)")
    p.add_argument("--good-ratings", metavar="MARKS",
                   help="how good each thumbs-up text is: comma-separated marks out of 10, one per text "
                        "(--good-text first, then the lines of --good); 10 = the full learning rate, 0 skips it")
    p.add_argument("--bad-ratings", metavar="MARKS",
                   help="how bad each thumbs-down text is: comma-separated marks out of 10, one per text")
    group = p.add_argument_group("2NRL options")
    group.add_argument("--neg-epochs", type=nonneg_int, default=2, help="epochs of the negative (punish) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=3, help="epochs of the positive (reward) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.5, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.1, help="learning rate of the positive phase (activation parameters use a tenth)")
    group.add_argument("--batch-size", type=pos_int, default=4, help="transitions per backend step (rated sets are small)")
    group.add_argument("--strength", type=nonneg_float, default=1.0,
                       help="count model: reward / penalty added to every edge of a rated path per pass (radix: ignored)")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_feedback)

    # correct --------------------------------------------------------------
    p = command(
        "correct", "teach one correction: only what changed moves",
        "Align what the network wrote (--wrong) with what it should have written (--right) character by\n"
        "character and move only the trigram nodes the two disagree on: the steps that wrote the struck-out\n"
        "characters are penalised, the steps that write the teacher's version are rewarded, and the words both\n"
        "sentences share keep what they earned.  --keep gives the rest of the correction a smaller reward (1\n"
        "is the old whole-sentence thumbs up, 0 teaches the fix alone).  This is what the tutor does with\n"
        "every correction its teacher writes; --dry-run only shows the alignment.",
    )
    p.add_argument("--wrong", metavar="TEXT", required=True, help="what the network wrote")
    p.add_argument("--right", metavar="TEXT", required=True, help="what it should have written")
    p.add_argument("--strength", type=nonneg_float, default=1.0, help="magnitude of one unit of feedback")
    p.add_argument("--weight", type=nonneg_float, default=1.0, help="how bad the attempt was: the penalty is strength x weight")
    p.add_argument("--reward", type=nonneg_float, default=1.0, help="what the correction is worth")
    p.add_argument("--keep", type=nonneg_float, default=0.25, help="what the unchanged part of the correction still earns")
    p.add_argument("--no-count", action="store_true", help="do not traverse the correction (it is counted by default)")
    p.add_argument("--dry-run", action="store_true", help="show the alignment without touching the model")
    p.add_argument("--blame", action="store_true",
                   help="also teach the negative network: the same diff, blaming only the characters you changed")
    p.add_argument("--reason", default="corrected", metavar="TAG", help="reason recorded with --blame")
    p.add_argument("--note", default="", metavar="TEXT", help="your own words, kept in the negative network's journal")
    p.add_argument("--negative", metavar="PATH",
                   help=f"negative model file for --blame (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.set_defaults(handler=cmd_correct)
    # negative -------------------------------------------------------------
    p = command(
        "negative", "the negative network: failures, why they failed, and the filter",
        "A copy of the network that keeps only its negative portions: every node and edge in it exists\n"
        "because something went wrong there, and every edge remembers why - the reasons the tutor gave\n"
        "(the Ollama reviewer, the code judge, a thumbs down, the evolve discriminator) with the blame\n"
        "each one carries.  `blame` teaches it a failure, `clear` lets text the tutor passed take blame\n"
        "back off the fragments it shares, `why` explains a text, `reasons` lists what the tutor said\n"
        "and `filter` runs the pair: the positive model writes, the negative one vetoes.\n"
        "The negative model lives beside --model (model.json -> model.negative.json) unless --negative\n"
        "says otherwise; it is also an ordinary model kind, so `--kind negative train` blames as well.",
    )
    _add_negative_option(p, top_level=True)
    actions = p.add_subparsers(dest="action", metavar="<action>", title="actions", required=True)

    a = actions.add_parser(
        "blame", help="learn a failure: the text, its reason and its severity",
        description="Teach the negative network that these texts went wrong.  --reason is the tutor's verdict\n"
                    "(gibberish, repetition, wrong-output, thumbs-down, ...), --note its own words (kept in the\n"
                    "journal) and --severity how badly it failed (1 = one ordinary failure).  This is the only\n"
                    "operation that adds structure to the negative network.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--text", action="append", metavar="TEXT", help="a failed text (repeatable)")
    a.add_argument("--data", metavar="FILE", help="failed texts, one per line")
    a.add_argument("--reason", default="unspecified", metavar="TAG", help="why it failed (the tutor's verdict)")
    a.add_argument("--severity", type=nonneg_float, default=1.0, help="how heavily to blame it (1 = one ordinary failure)")
    a.add_argument("--source", default="cli", metavar="NAME", help="who says so (review, codegen, evolve, frontend, cli)")
    a.add_argument("--note", default="", metavar="TEXT", help="the tutor's own words, kept in the journal")
    a.add_argument("--epochs", type=pos_int, default=1, help="blame passes over the texts")
    a.add_argument("--out", metavar="PATH", help="where to save the negative model (default: --negative)")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_blame)

    a = actions.add_parser(
        "clear", help="the tutor passed these texts: take blame off what they share",
        description="Credit the edges these texts share with known failures.  Net evidence is blame minus\n"
                    "clearing, so a fragment that shows up in good and bad output alike stops carrying the\n"
                    "verdict.  Nothing is created: text the failure structure cannot walk simply does not match.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--text", action="append", metavar="TEXT", help="a passed text (repeatable)")
    a.add_argument("--data", metavar="FILE", help="passed texts, one per line")
    a.add_argument("--weight", type=nonneg_float, default=1.0, help="how much blame one pass cancels")
    a.add_argument("--epochs", type=pos_int, default=1, help="clearing passes over the texts")
    a.add_argument("--out", metavar="PATH", help="where to save the negative model (default: --negative)")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_clear)

    a = actions.add_parser(
        "why", help="why a text looks like a failure (reasons and the fragments to blame)",
        description="Walk texts through the failure structure: how much of each is built out of known failure\n"
                    "(risk and coverage), which reasons that blame carries and which fragments carry it.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--text", action="append", metavar="TEXT", help="a text to judge (repeatable)")
    a.add_argument("--data", metavar="FILE", help="texts to judge, one per line")
    a.add_argument("--threshold", type=nonneg_float, metavar="RISK", help="reject at this blame per transition (default: the model's)")
    a.add_argument("--min-coverage", type=nonneg_float, metavar="SHARE", help="known-failing share needed before rejecting")
    a.add_argument("--spans", type=nonneg_int, default=5, help="blamed fragments to show")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_why)

    a = actions.add_parser(
        "filter", help="the pair: the positive model writes, the negative one vetoes",
        description="The GAN at output time.  The positive model (--model) over-samples candidates, the negative\n"
                    "one judges each of them, and what survives is printed with what was dropped and why.  Two\n"
                    "signals reject: blame (risk over the threshold) and the likelihood ratio (the candidate reads\n"
                    "more like known failure than like the training data).  With --text / --data the given texts\n"
                    "are judged instead of generating any.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--count", type=pos_int, default=3, help="texts wanted out of the filter")
    a.add_argument("--prefix", default="", metavar="TEXT", help="continue this prefix")
    a.add_argument("--max-length", type=nonneg_int, default=60, help="characters per candidate")
    a.add_argument("--mode", choices=("sample", "beam", "dijkstra"), default="sample", help="how the positive model writes")
    a.add_argument("--temperature", type=nonneg_float, default=1.0, help="sampling temperature")
    a.add_argument("--step-penalty", type=nonneg_float, default=0.0, help="extra cost per edge (longer texts cost more)")
    a.add_argument("--over-sample", type=pos_int, default=3, help="candidates drawn per wanted text")
    a.add_argument("--text", action="append", metavar="TEXT", help="judge this text instead of generating (repeatable)")
    a.add_argument("--data", metavar="FILE", help="judge the texts of FILE (one per line) instead of generating")
    a.add_argument("--threshold", type=nonneg_float, metavar="RISK", help="reject at this blame per transition (default: the model's)")
    a.add_argument("--min-coverage", type=nonneg_float, metavar="SHARE", help="known-failing share needed before rejecting")
    a.add_argument("--ratio", type=float, default=0.0, metavar="NATS",
                   help="reject when the candidate reads this much more like failure than like the training data")
    a.add_argument("--no-ratio", action="store_true", help="judge by blame alone (turn the likelihood ratio off)")
    a.add_argument("--peak", type=nonneg_float, metavar="BLAME",
                   help="reject a candidate carrying this much blame on a single fragment, whatever the rest of it is "
                        "(1 = a fragment the tutor corrected once); off by default")
    a.add_argument("--strict", action="store_true", help="also drop candidates the negative network only finds suspect")
    a.add_argument("--spans", type=nonneg_int, default=3, help="blamed fragments per verdict")
    a.add_argument("--learn", action="store_true",
                   help="blame what the filter rejects (off by default: the tutor supplies the negatives)")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_filter)

    a = actions.add_parser(
        "reasons", help="what the tutor has blamed, and the journal of what it said",
        description="The reason table (blame, failures and edges per reason) and the newest journal entries.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--limit", type=pos_int, default=20, help="reasons to list")
    a.add_argument("--log", type=nonneg_int, default=10, help="journal entries to show")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_reasons)

    a = actions.add_parser(
        "forget", help="drop or fade the blame behind a reason (the tutor can be wrong)",
        description="Remove the blame one reason contributed (or all of it with no --reason).  --factor keeps a\n"
                    "share of it instead of dropping it entirely, which is how old failures fade.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--reason", metavar="TAG", help="the reason to forget (default: every reason)")
    a.add_argument("--factor", type=nonneg_float, default=0.0, help="share of the blame to keep (0 = forget it, 0.5 = halve it)")
    a.add_argument("--out", metavar="PATH", help="where to save the negative model (default: --negative)")
    _add_negative_option(a, top_level=False)
    a.set_defaults(handler=cmd_negative_forget)

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
    p.add_argument("--blame", action="store_true",
                   help="teach the negative network from this loop: every fake the discriminator scores below the "
                        "real texts is blamed (reason discriminator / blatant), the real texts clear blame")
    p.add_argument("--negative", metavar="PATH",
                   help=f"negative model file for --blame (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)")
    p.add_argument("--disc-neg-epochs", type=nonneg_int, default=EvolveConfig.disc_neg_epochs,
                   help="discriminator negative-phase epochs")
    p.add_argument("--disc-pos-epochs", type=nonneg_int, default=EvolveConfig.disc_pos_epochs,
                   help="discriminator positive-phase epochs")
    _add_two_nrl_options(p, neg_epochs=EvolveConfig.neg_epochs, pos_epochs=EvolveConfig.pos_epochs,
                         batch_size=EvolveConfig.batch_size)
    group = p.add_argument_group("failures")
    group.add_argument("--blatant-mode", choices=BLATANT_MODES, default=EvolveConfig.blatant_mode,
                       help="how failed fakes drive the update: none = the worst half is uniform 2NRL garbage; "
                            "fail_invert = train on every failure with learning rates scaled by how bad it is (blatantly "
                            "fail on purpose), then invert the model and fine-tune on real texts; activation / state = "
                            "no negative pass, move the activation amplitude / trained value of every other node on a "
                            "failed path toward its negation, the worse the more (blatant fakes skip 2NRL entirely)")
    group.add_argument("--blatant-margin", type=nonneg_float, default=EvolveConfig.blatant_margin, metavar="NATS",
                       help="per-char log-prob below the real texts at which a fake is blatant (fail_invert: the "
                            "learning-rate multiplier reaches 2 here)")
    group.add_argument("--blatant-boost", type=nonneg_float, default=EvolveConfig.blatant_boost, metavar="X",
                       help="fail_invert: the largest learning-rate multiplier a failure can get")
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
    from .chatgpt import DEFAULT_MODEL as chatgpt_default_model, DEFAULT_URL as chatgpt_default_url
    from .codegen import DEFAULT_TEACHER_MODEL as codegen_default_model
    from .llm import DEFAULT_PROVIDER, PROVIDERS

    p = command(
        "codegen", "generate Python programs, run them in a sandbox, judge them with an LLM, reward / punish with 2NRL",
        "Semi-supervised code generation over a list of problems.  Phase `teacher`: the tutor (a local\n"
        "Ollama model, or ChatGPT with --teacher-provider chatgpt and $OPENAI_API_KEY set) writes\n"
        "each solution, the sandbox runs it, the tutor fixes failures and the judge (the LLM plus an\n"
        "objective PEP 8 / naming check) confirms it; the network learns question + answer with every wrong\n"
        "attempt as 2NRL garbage before the correct one.  Phase `model`: the network writes the solutions\n"
        "itself; errors and rejected programs are punished (negative phase), correct ones rewarded (positive\n"
        "phase), the teacher supplies the answer when the network never succeeds.  Ctrl-C stops after the\n"
        "current problem and saves.",
    )
    p.add_argument("--problems", required=True, metavar="FILE",
                   help="one prompt per line, or .json / .jsonl objects {id, prompt, tests, expected_output}")
    p.add_argument("--phase", choices=("both", "teacher", "model"), default="both",
                   help="teacher: the tutor writes the solutions; model: the network writes them; both: teacher, then model")
    p.add_argument("--rounds", type=pos_int, default=1, help="passes over the problem list (each pass runs the chosen phases)")
    p.add_argument("--teacher-provider", "--provider", dest="teacher_provider", choices=PROVIDERS, default=DEFAULT_PROVIDER,
                   help="who tutors: a local Ollama model, or ChatGPT (needs $OPENAI_API_KEY; prompts and programs go to OpenAI)")
    p.add_argument("--teacher-model", metavar="NAME",
                   help=f"model that writes, fixes and judges (default: $RADIXNET_CODEGEN_MODEL or {codegen_default_model} "
                        f"for ollama, $RADIXNET_OPENAI_MODEL or {chatgpt_default_model} for chatgpt)")
    p.add_argument("--judge-provider", choices=PROVIDERS, help="judge with the other provider (default: the tutor's)")
    p.add_argument("--judge-model", metavar="NAME", help="a different model for judging (default: the teacher model)")
    p.add_argument("--url", metavar="URL",
                   help="the tutor's base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434 for ollama, "
                        "$OPENAI_BASE_URL or https://api.openai.com/v1 for chatgpt)")
    p.add_argument("--judge-url", metavar="URL", help="base URL of the judge's provider (default: the same as --url / its own default)")
    p.add_argument("--timeout", type=_float_at_least(1.0), metavar="SECONDS", help="seconds to wait for one LLM answer (default: 120)")
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
    p.add_argument("--blame", action="store_true",
                   help="teach the negative network why the rejected programs were rejected (the sandbox, the style "
                        "checker and the judge are the tutor)")
    p.add_argument("--negative", metavar="PATH",
                   help=f"negative model file for --blame (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)")
    p.set_defaults(handler=cmd_codegen)

    # tutor ------------------------------------------------------------------
    from .tutor import (
        DEFAULT_PLAN_LESSONS,
        DEFAULT_TUTOR_MODEL as tutor_default_model,
        MODES as TUTOR_MODES,
        TWONRL_PER as TUTOR_TWONRL_PER,
    )

    p = command(
        "tutor", "automated English lessons: the teacher writes the prefix, the network completes it and is marked",
        "The prediction process run without a human at the keyboard.  Each round the teacher (a local\n"
        "Ollama model, or ChatGPT with --tutor-provider chatgpt and $OPENAI_API_KEY set) writes\n"
        "sentence openings about a topic (each drilling one point of grammar, each with its own model\n"
        "answer), the network completes them with the prediction search, and the same LLM marks every\n"
        "sentence as an English teacher: grammar, spelling and fluency out of 10, the worst mistake named,\n"
        "one line of teaching and the sentence written out correctly.  Failed sentences become 2NRL garbage\n"
        "- weighted by how bad the mark was - and the corrections the fine-tune pass, so the network is\n"
        "taught the English it got wrong.  With --adapt (the default) the next round drills the mistakes\n"
        "the last one made.  Ctrl-C stops after the current round and saves.",
    )
    p.add_argument("--topic", default="everyday life", metavar="TEXT", help="what the sentences are about")
    p.add_argument("--rounds", type=pos_int, default=3, help="lesson rounds (each writes, completes and marks a new set of exercises)")
    p.add_argument("--exercises", type=pos_int, default=5, help="sentence openings per round")
    p.add_argument("--attempts", type=pos_int, default=1, help="completions the network writes per exercise (the first in --mode, the rest sampled)")
    p.add_argument("--focus", metavar="TEXT", help="pin every exercise to one point of grammar, e.g. 'past tense'")
    p.add_argument("--level", default="beginner", metavar="TEXT", help="how hard the exercises are (beginner, intermediate, ...)")
    p.add_argument("--words", default="3 to 6", metavar="TEXT", help="how many words a prefix has")
    p.add_argument("--brief", metavar="TEXT",
                   help="what this batch of lessons is being taught to: the prompt the last report card led to "
                        "(the plan's 'brief'), handed to the teacher with every set of exercises")
    p.add_argument("--tutor-provider", "--provider", dest="tutor_provider", choices=PROVIDERS, default=DEFAULT_PROVIDER,
                   help="who teaches: a local Ollama model, or ChatGPT (needs $OPENAI_API_KEY; the lessons go to OpenAI)")
    p.add_argument("--tutor-model", metavar="NAME",
                   help=f"model that sets and marks the exercises (default: $RADIXNET_TUTOR_MODEL or {tutor_default_model} "
                        f"for ollama, $RADIXNET_OPENAI_MODEL or {chatgpt_default_model} for chatgpt)")
    p.add_argument("--grader-provider", choices=PROVIDERS, help="mark with the other provider (default: the teacher's)")
    p.add_argument("--grader-model", metavar="NAME", help="a different model for the marking (default: the tutor model)")
    p.add_argument("--url", metavar="URL",
                   help="the teacher's base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434 for ollama, "
                        "$OPENAI_BASE_URL or https://api.openai.com/v1 for chatgpt)")
    p.add_argument("--grader-url", metavar="URL", help="base URL of the marker's provider (default: the same as --url / its own default)")
    p.add_argument("--timeout", type=_float_at_least(1.0), metavar="SECONDS", help="seconds to wait for one LLM answer (default: 120)")
    group = p.add_argument_group("completion options")
    group.add_argument("--mode", choices=TUTOR_MODES, default="dijkstra", help="how the network completes a prefix")
    group.add_argument("--length", type=nonneg_int, default=20, help="characters the completion should reach")
    group.add_argument("--max-length", type=pos_int, default=80, help="cap on the completion")
    group.add_argument("--temperature", type=nonneg_float, default=1.0, help="sampling temperature (--mode sample and the extra attempts)")
    group.add_argument("--no-to-end", action="store_true", help="stop at --length instead of finishing the sentence at END")
    group.add_argument("--beam", type=pos_int, metavar="N", help="beam width (--mode beam)")
    group = p.add_argument_group("marking options")
    group.add_argument("--threshold", type=nonneg_float, default=6.0, help="mark out of 10 a sentence must reach to pass")
    group.add_argument("--grammar-weight", type=nonneg_float, default=0.6,
                       help="share of the mark that is grammar; the rest is spelling and fluency")
    group.add_argument("--batch", type=pos_int, default=10, help="sentences marked in one Ollama call")
    group.add_argument("--no-adapt", action="store_true", help="do not drill the previous round's weakest points")
    group.add_argument("--drills", type=nonneg_int, default=0,
                       help="extra correct example sentences per round, added to the fine-tune pass")
    group.add_argument("--plan", type=nonneg_int, nargs="?", const=DEFAULT_PLAN_LESSONS, metavar="N",
                       help="hand the report card at the end back to the teacher and print the next N lessons it "
                            f"plans - one point of grammar each, worst mistake first (0 = off, default N: "
                            f"{DEFAULT_PLAN_LESSONS})")
    group.add_argument("--no-teach-answer", action="store_true",
                       help="a failed lesson learns only the correction, not the teacher's own model answer")
    group.add_argument("--dry-run", action="store_true", help="set and mark the exercises but train nothing and save nothing")
    group = p.add_argument_group("what a correction teaches")
    group.add_argument("--no-diff-corrections", action="store_true",
                       help="learn a correction as two whole sentences (the old way) instead of from its diff "
                            "with what the network wrote")
    group.add_argument("--keep-weight", type=nonneg_float, default=0.25,
                       help="what the unchanged part of a correction still earns: 0 teaches the fix alone, "
                            "1 rewards the whole corrected sentence")
    group = p.add_argument_group("2NRL options")
    group.add_argument("--twonrl-per", choices=TUTOR_TWONRL_PER, default="round", help="learn once per round, or after every lesson")
    group.add_argument("--min-weight", type=nonneg_float, default=0.25,
                       help="negative-phase weight of a near miss (a hopeless sentence always weighs 1)")
    group.add_argument("--neg-epochs", type=nonneg_int, default=2, help="epochs of the negative (garbage) phase")
    group.add_argument("--pos-epochs", type=nonneg_int, default=3, help="epochs of the positive (correction) phase")
    group.add_argument("--neg-lr", type=nonneg_float, default=0.5, help="learning rate of the negative phase")
    group.add_argument("--pos-lr", type=nonneg_float, default=0.1, help="learning rate of the positive phase (activation parameters use a tenth)")
    group.add_argument("--batch-size", type=pos_int, default=4, help="transitions per backend step")
    group.add_argument("--strength", type=nonneg_float, help="count model: reward / penalty per pass (radix: ignored)")
    group.add_argument("--no-replay", action="store_true", help="do not keep teaching earlier corrections")
    group.add_argument("--replay-limit", type=nonneg_int, default=64, help="corrections kept for the replay (0 = no limit)")
    _add_checkpoint_options(p, "round")
    p.add_argument("--out", metavar="PATH", help="where to save the model (default: --model)")
    p.add_argument("--report", metavar="FILE", help="write a JSON report (config, records, lessons, report card)")
    p.add_argument("--blame", action="store_true",
                   help="teach the negative network why each failed sentence failed: the mistake the teacher named "
                        "is the reason, its mark the severity, and only the characters it corrected are blamed")
    p.add_argument("--negative", metavar="PATH",
                   help=f"negative model file for --blame (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)")
    p.set_defaults(handler=cmd_tutor)

    # chatgpt ---------------------------------------------------------------
    p = command(
        "chatgpt", "query ChatGPT (OpenAI): check the key and models, ask one question",
        "Talk to OpenAI's API, the hosted alternative to a local Ollama tutor.  `models` lists what the\n"
        "key may use (and is the quickest check that tutoring will work), `ask` sends one prompt.\n"
        "The key comes from $OPENAI_API_KEY (or $OPENAI_API_KEY_FILE) and is never printed or stored;\n"
        "$OPENAI_BASE_URL points the client at any OpenAI-compatible server.\n"
        f"Usage: {PROG} [global options] chatgpt [--url URL] [--chatgpt-model NAME] <action> [options]",
    )
    p.add_argument("--url", metavar="URL", help=f"OpenAI base URL (default: $OPENAI_BASE_URL or {chatgpt_default_url})")
    p.add_argument("--chatgpt-model", metavar="NAME",
                   help=f"model name (default: $RADIXNET_OPENAI_MODEL or {chatgpt_default_model})")
    p.add_argument("--timeout", type=_float_at_least(1.0), metavar="SECONDS",
                   help="seconds to wait for one ChatGPT answer (default: 120)")
    actions = p.add_subparsers(dest="action", metavar="<action>", title="actions", required=True)

    a = actions.add_parser("models", help="list the models the API key can use",
                           description="List the models the OpenAI API key has access to.", formatter_class=_HelpFormatter)
    a.set_defaults(handler=cmd_chatgpt_models)

    a = actions.add_parser(
        "ask", help="send one prompt and print the answer",
        description="Send one prompt to ChatGPT and print the answer: a quick check that the key, the model\n"
                    "and the network path all work before pointing the code-generation tutor at it.",
        formatter_class=_HelpFormatter,
    )
    a.add_argument("--prompt", required=True, metavar="TEXT", help="the question to ask")
    a.add_argument("--system", metavar="TEXT", help="system instruction sent before the prompt")
    a.add_argument("--temperature", type=nonneg_float, default=0.7, help="sampling temperature (dropped when the model refuses it)")
    a.add_argument("--json", dest="json_answer", action="store_true", help="ask for a JSON answer")
    a.set_defaults(handler=cmd_chatgpt_ask)

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
    a.add_argument("--blame", action="store_true",
                   help="teach the negative network what failed and why: the reviewer's critique becomes the reason, "
                        "its rating the severity, and the passed texts clear blame")
    a.add_argument("--negative", metavar="PATH",
                   help=f"negative model file for --blame (default: {DEFAULT_NEGATIVE_MODEL}, i.e. beside --model)")
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
    p.add_argument("--chatgpt-url", metavar="URL",
                   help=f"OpenAI base URL for the ChatGPT tutor (default: $OPENAI_BASE_URL or {chatgpt_default_url})")
    p.add_argument("--chatgpt-model", metavar="NAME",
                   help=f"default ChatGPT model (default: $RADIXNET_OPENAI_MODEL or {chatgpt_default_model}); "
                        "the key always comes from the server's $OPENAI_API_KEY")
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
    kind_default = DEFAULT_KIND_MODELS.get(getattr(args, "kind", None) or "")
    if kind_default and args.model == DEFAULT_MODEL:
        args.model = kind_default  # another kind does not overwrite the radix default file
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
