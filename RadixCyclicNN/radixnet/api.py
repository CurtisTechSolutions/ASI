"""HTTP JSON API and static file serving for RadixNet (standard library only).

Architecture
------------
* :class:`ModelService` owns one :class:`~radixnet.model.RadixNet` behind an
  ``RLock``.  Short operations (predict, generate, score, graph, status, ...)
  run synchronously under the lock.  Long operations - ``train``, ``2nrl`` and
  the GAN-style ``evolve`` loop - run in a daemon thread as a :class:`Job`
  (one at a time; a second request is refused with 409) and hold the lock
  while they mutate the model.  At every epoch / generation boundary the job
  hands the lock to waiting readers, so the model stays readable while it
  trains and a reader never sees a half-merged graph.
* :class:`ApiHandler` (``BaseHTTPRequestHandler``) parses requests, validates
  JSON bodies (400 with a helpful message), dispatches to the service, adds
  CORS headers to every response and serves the built React frontend
  (``frontend/dist``) - or a built-in help page when it is not built.
* :func:`create_server` / :func:`run_server` wire both onto a
  ``ThreadingHTTPServer``.

HTTP status codes: 200 for synchronous results, 202 for accepted background
jobs, 204 for ``OPTIONS`` preflights, 400 for bad input, 404 for unknown
endpoints / files, 405 for a wrong method (with ``Allow``), 409 while a job is
running, 411 for chunked bodies (``Content-Length`` is required), 413 for
oversized bodies and 500 for unexpected failures.  Errors are
always ``{"error": "message"}``.  See DESIGN.md section 12 for the endpoint
table.
"""

from __future__ import annotations

import contextlib
import heapq
import html
import itertools
import json
import math
import mimetypes
import os
import sys
import email.parser
import email.policy
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__
from .backend import describe_backends
from .checkpoint import CheckpointManager
from .gan import EvolveConfig, Evolver
from .graph import END, START, RadixCyclicGraph
from .model import RadixNet, TrainConfig
from .ollama import (
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
    DEFAULT_URL as OLLAMA_DEFAULT_URL,
    STYLES as OLLAMA_STYLES,
    OllamaClient,
    OllamaError,
    adversarial_review,
    corpus_from_prompt,
    normalise_url,
    sample_texts,
)
from .search import PathResult

__all__ = [
    "ApiError",
    "Job",
    "ModelService",
    "ApiHandler",
    "RadixNetHTTPServer",
    "create_server",
    "run_server",
    "MAX_BODY_BYTES",
    "DEFAULT_GRAPH_LIMIT",
]

MAX_BODY_BYTES = 64 * 1024 * 1024
"""Largest accepted request body (a training corpus can be big)."""

DEFAULT_GRAPH_LIMIT = 150
"""Default number of top nodes returned by ``GET /api/graph``."""

_READER_YIELD_SECONDS = 1.0
"""Longest time a job waits at a boundary for queued readers before resuming."""

_CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type, Accept"),
    ("Access-Control-Max-Age", "86400"),
)

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"
_HTML_CONTENT_TYPE = "text/html; charset=utf-8"

_MIME_TYPES = {
    ".html": _HTML_CONTENT_TYPE,
    ".htm": _HTML_CONTENT_TYPE,
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": _JSON_CONTENT_TYPE,
    ".map": _JSON_CONTENT_TYPE,
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".wasm": "application/wasm",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# errors and jobs
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Failure reported to the client as ``{"error": message}`` with ``status``."""

    __slots__ = ("status", "message")

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message


class Job:
    """One asynchronous operation (``train`` / ``2nrl`` / ``evolve``).

    ``state`` is ``running`` until the worker finishes: ``done`` (ran to
    completion), ``stopped`` (a stop was requested) or ``error`` (``error``
    holds the message).  ``progress`` is the latest record, ``history`` all
    of them.  ``to_dict()`` is the JSON status document of DESIGN.md
    section 12 plus ``stop_requested``.
    """

    __slots__ = (
        "id", "type", "state", "progress", "history", "error",
        "started_at", "finished_at", "stop_event", "thread", "_lock",
    )

    def __init__(self, job_id: str, kind: str) -> None:
        self.id = job_id
        self.type = kind
        self.state = "running"
        self.progress: dict | None = None
        self.history: list[dict] = []
        self.error: str | None = None
        self.started_at = _utc_now()
        self.finished_at: str | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.state == "running"

    def record(self, rec: dict) -> None:
        """Progress callback target: remember one epoch / generation record."""
        with self._lock:
            self.history.append(rec)
            self.progress = rec

    def finish(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.finished_at = _utc_now()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "type": self.type,
                "state": self.state,
                "progress": self.progress,
                "history": list(self.history),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "stop_requested": self.stop_event.is_set(),
            }


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


class ModelService:
    """The model, its lock, background jobs, checkpoints and persistence.

    Every public method is safe to call from any thread.  Readers use
    :meth:`session`; methods that change the model use :meth:`mutating`, which
    refuses with 409 while a job runs.  ``model_path`` / ``checkpoint_dir``
    are absolute paths (or ``None``); when ``model_path`` exists it is loaded
    at construction.
    """

    def __init__(
        self,
        model_path: str | None = None,
        checkpoint_dir: str | None = None,
        backend: str = "auto",
        device: str | None = None,
        seed: int = 0,
        keep: int = 5,
        quiet: bool = False,
        upload_dir: str | None = None,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
    ) -> None:
        self.model_path = os.path.abspath(model_path) if model_path else None
        self.checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else None
        self.upload_dir = os.path.abspath(upload_dir) if upload_dir else None
        self._upload_lock = threading.Lock()
        self.ollama_url = normalise_url(ollama_url or OLLAMA_DEFAULT_URL)
        self.ollama_model = (ollama_model or OLLAMA_DEFAULT_MODEL).strip() or OLLAMA_DEFAULT_MODEL
        self.backend_name = backend
        self.device = device
        self.seed = int(seed)
        self.quiet = bool(quiet)
        self.backends = describe_backends()
        self._checkpoints = CheckpointManager(self.checkpoint_dir, keep=keep) if self.checkpoint_dir else None
        self._lock = threading.RLock()
        self._readers = threading.Condition(self._lock)
        self._waiting = 0
        self._waiting_lock = threading.Lock()
        self._job: Job | None = None
        self._job_ids = itertools.count(1)
        self._evolve_history: list[dict] = []
        self._discriminator: RadixNet | None = None
        self._discriminator_seed: int | None = None
        if self.model_path and os.path.isfile(self.model_path):
            self.model = RadixNet.load(self.model_path, backend=backend, device=device)
            self.loaded_from: str | None = self.model_path
        else:
            self.model = RadixNet(seed=self.seed, backend=backend, device=device)
            self.loaded_from = None

    # -- locking -------------------------------------------------------------

    @contextlib.contextmanager
    def session(self) -> Iterator[RadixNet]:
        """Hold the model lock.

        A running job owns the lock while it works and hands it over at its
        next epoch / generation boundary to whoever is queued here, so a
        reader blocks for at most one epoch.
        """
        with self._waiting_lock:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._waiting_lock:
                self._waiting -= 1
        try:
            yield self.model
        finally:
            self._readers.notify_all()
            self._lock.release()

    @contextlib.contextmanager
    def mutating(self) -> Iterator[RadixNet]:
        """Like :meth:`session` but refuses (409) while a job is running."""
        self._ensure_idle()
        with self.session() as model:
            self._ensure_idle()
            yield model

    def _yield_to_readers(self) -> None:
        """Called by the job thread (holding the lock) between epochs.

        Releases the lock until every queued reader had its turn (bounded by
        ``_READER_YIELD_SECONDS``), then re-acquires it.
        """
        if self._waiting == 0:
            return
        deadline = time.monotonic() + _READER_YIELD_SECONDS
        while self._waiting > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._readers.wait(remaining)

    # -- jobs ----------------------------------------------------------------

    @property
    def job(self) -> Job | None:
        """The current or most recent job (``None`` before the first one)."""
        return self._job

    @property
    def discriminator(self) -> RadixNet | None:
        """Discriminator kept across evolve runs of the same model and seed."""
        return self._discriminator

    def _ensure_idle(self) -> None:
        job = self._job
        if job is not None and job.running:
            raise ApiError(
                409, f"a {job.type} job ({job.id}) is running; wait for it to finish or POST /api/job/stop"
            )

    def _progress(self, job: Job, sink: list[dict] | None = None) -> Callable[[dict], None]:
        def on_record(rec: dict) -> None:
            job.record(rec)
            if sink is not None:
                sink.append(rec)
            self._yield_to_readers()

        return on_record

    def _start_job(self, kind: str, work: Callable[[Job], None]) -> dict:
        self._ensure_idle()
        with self.session():
            self._ensure_idle()
            job = Job(f"{kind}-{next(self._job_ids)}", kind)
            thread = threading.Thread(
                target=self._run_job, args=(job, work), name=f"radixnet-{job.id}", daemon=True
            )
            job.thread = thread
            self._job = job
            thread.start()
            return job.to_dict()

    def _run_job(self, job: Job, work: Callable[[Job], None]) -> None:
        with self._lock:
            state, error = "error", "job thread terminated unexpectedly"
            try:
                work(job)
                state = "stopped" if job.stop_event.is_set() else "done"
                error = None
            except Exception as exc:  # noqa: BLE001 - reported as the job's error
                error = f"{type(exc).__name__}: {exc}"
                self._log_exception(f"{job.type} job {job.id} failed")
            finally:
                job.finish(state, error)

    def job_status(self) -> dict | None:
        """Status of the current / last job, ``None`` before the first one."""
        job = self._job
        return job.to_dict() if job is not None else None

    def stop_job(self) -> dict:
        """Request the current job to stop (idempotent); returns its status."""
        job = self._job
        if job is None:
            raise ApiError(404, "no job has been started")
        job.stop_event.set()
        return job.to_dict()

    def stop_evolve(self) -> dict:
        """Stop the evolve job; 409 if a job of another type is running."""
        job = self._job
        if job is not None and job.type != "evolve" and job.running:
            raise ApiError(409, f"the running job ({job.id}) is a {job.type} job; use POST /api/job/stop")
        if job is None or job.type != "evolve":
            raise ApiError(404, "no evolve job has been started")
        job.stop_event.set()
        return job.to_dict()

    def start_train(self, texts: list[str], config: TrainConfig) -> dict:
        """Start a ``train`` job; returns its status."""
        config.validate()
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None

        def work(job: Job) -> None:
            self.model.train(
                texts, config, checkpoint_manager=manager, progress=self._progress(job),
                stop_event=job.stop_event,
            )

        return self._start_job("train", work)

    def start_two_nrl(
        self,
        bad: list[str],
        good: list[str],
        neg_epochs: int = 3,
        pos_epochs: int = 3,
        neg_lr: float = 0.05,
        pos_lr: float = 0.01,
        **overrides: Any,
    ) -> dict:
        """Start a ``2nrl`` job (``overrides``: batch_size, auto_compress, clip, shuffle)."""
        TrainConfig(epochs=neg_epochs, lr=neg_lr, **overrides).validate()
        TrainConfig(epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, **overrides).validate()

        def work(job: Job) -> None:
            self.model.two_nrl(
                bad, good, neg_epochs=neg_epochs, pos_epochs=pos_epochs, neg_lr=neg_lr, pos_lr=pos_lr,
                progress=self._progress(job), stop_event=job.stop_event, **overrides,
            )

        return self._start_job("2nrl", work)

    def start_evolve(self, corpus: list[str], generations: int | None, config: EvolveConfig) -> dict:
        """Start an ``evolve`` job (``generations`` ``None`` / ``0`` = until stopped).

        The discriminator survives across runs of the same model with the same
        ``config.seed``; its records are appended to :meth:`evolve_history`.
        """
        if not generations:
            generations = None
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        self._ensure_idle()
        with self.session() as model:
            self._ensure_idle()
            disc = self._discriminator if self._discriminator_seed == config.seed else None
            evolver = Evolver(model, corpus, discriminator=disc, config=config)
            self._discriminator = evolver.discriminator
            self._discriminator_seed = config.seed
            sink = self._evolve_history

            def work(job: Job) -> None:
                evolver.run(
                    generations, checkpoint_manager=manager, progress=self._progress(job, sink),
                    stop_event=job.stop_event,
                )

            return self._start_job("evolve", work)

    # -- synchronous operations ----------------------------------------------

    def status(self) -> dict:
        with self.session() as model:
            stats = model.stats()
        job = self._job
        stats.update(
            job=job.to_dict() if job is not None else None,
            backends=self.backends,
            model_path=self.model_path,
            checkpoint_dir=self.checkpoint_dir,
            upload_dir=self.upload_dir,
            ollama={"url": self.ollama_url, "model": self.ollama_model},
        )
        return stats

    def predict(self, prefix: str, **options: Any) -> dict:
        with self.session() as model:
            result = model.predict(prefix, **options)
        return {
            "prefix": prefix,
            "continuation": result.text,
            "full_text": result.full_text,
            "cost": result.cost,
            "step_costs": list(result.step_costs),
            "path": list(result.labels),
            "node_ids": list(result.node_ids),
            "expanded": result.expanded,
            "reached_end": result.reached_end,
        }

    def generate(self, **options: Any) -> dict:
        with self.session() as model:
            results = model.generate(**options)
        return {"samples": [_sample_dict(r) for r in results]}

    def score(self, text: str) -> dict:
        with self.session() as model:
            return model.score(text)

    def invert(self) -> dict:
        with self.mutating() as model:
            model.invert()
            return model.stats()

    def compress(self) -> dict:
        with self.mutating() as model:
            merges = model.compress()
            return {"merges": merges, **model.stats()}

    def history(self) -> dict:
        with self.session() as model:
            return {"history": list(model.history)}

    def evolve_history(self) -> dict:
        with self.session():
            return {"history": list(self._evolve_history)}

    def graph(self, limit: int = DEFAULT_GRAPH_LIMIT) -> dict:
        if limit < 0:
            raise ApiError(400, f"'limit' must be >= 0 (got {limit})")
        with self.session() as model:
            return _graph_view(model.graph, limit)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | None = None) -> dict:
        """Write the model (default: ``model_path``); allowed while a job runs
        because it snapshots the model at an epoch boundary."""
        target = path or self.model_path
        if not target:
            raise ApiError(400, "no 'path' given and the server was started without a model path")
        target = os.path.abspath(target)
        with self.session() as model:
            model.save(target)
        return {"path": target, "bytes": os.path.getsize(target)}

    def load(self, path: str) -> dict:
        """Replace the model with the one at ``path`` (parsed outside the lock)."""
        self._ensure_idle()
        model = RadixNet.load(path, backend=self.backend_name, device=self.device)
        return self._replace_model(model)

    def reset(self, seed: int | None = None) -> dict:
        """Replace the model with a fresh one (``seed`` defaults to the server seed)."""
        self._ensure_idle()
        model = RadixNet(seed=self.seed if seed is None else seed, backend=self.backend_name, device=self.device)
        return self._replace_model(model)

    def _replace_model(self, model: RadixNet) -> dict:
        with self.mutating():
            self.model = model
            self._discriminator = None
            self._discriminator_seed = None
            self._evolve_history = []
            return model.stats()

    def _require_checkpoints(self, what: str = "checkpoints") -> CheckpointManager:
        if self._checkpoints is None:
            raise ApiError(400, f"{what} requires a checkpoint directory; start the server with --checkpoint-dir")
        return self._checkpoints

    def checkpoints(self) -> dict:
        manager = self._checkpoints
        if manager is None:
            return {"checkpoints": [], "latest": None}
        return {"checkpoints": manager.list(), "latest": manager.latest()}

    def save_checkpoint(self, tag: str = "manual") -> dict:
        """Checkpoint the model (step = total epochs, metrics = last record)."""
        manager = self._require_checkpoints()
        with self.session() as model:
            metrics = model.history[-1] if model.history else None
            return manager.save(model, model.meta["epochs_total"], tag, metrics)

    def restore_checkpoint(self, name: str) -> dict:
        manager = self._require_checkpoints()
        self._ensure_idle()
        model = manager.load(name, backend=self.backend_name, device=self.device)
        return self._replace_model(model)

    # -- uploads (training corpora kept on the server) -----------------------

    def _require_uploads(self) -> str:
        if not self.upload_dir:
            raise ApiError(400, "uploads are disabled: start the server with --upload-dir (create_server(upload_dir=...))")
        os.makedirs(self.upload_dir, exist_ok=True)
        return self.upload_dir

    def _upload_path(self, name: str) -> str:
        return os.path.join(self._require_uploads(), sanitize_upload_name(name))

    def uploads(self) -> dict:
        """Every uploaded file with its size and non-blank line count."""
        root = self._require_uploads()
        with self._upload_lock:
            names = sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n)) and not n.endswith(".part"))
            records = [_upload_record(root, name) for name in names]
        return {"uploads": records, "upload_dir": root}

    def upload(self, name: str, content: str) -> dict:
        """Store ``content`` as the upload ``name`` (UTF-8; an existing file of that name is replaced)."""
        path = self._upload_path(name)
        with self._upload_lock:
            replaced = os.path.isfile(path)
            part = path + ".part"
            with open(part, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)
            os.replace(part, path)
            record = _upload_record(os.path.dirname(path), os.path.basename(path))
        record["replaced"] = replaced
        self._log(f"upload {record['name']} ({record['bytes']} bytes, {record['lines']} lines)")
        return record

    def delete_upload(self, name: str) -> dict:
        path = self._upload_path(name)
        with self._upload_lock:
            if not os.path.isfile(path):
                raise ApiError(404, f"no upload named {os.path.basename(path)!r}")
            os.remove(path)
        return {"deleted": os.path.basename(path)}

    def upload_texts(self, names: list[str], whole_file: bool = False) -> list[str]:
        """Training texts read from uploads: one per non-blank line, or each file as one text."""
        texts: list[str] = []
        for name in names:
            path = self._upload_path(name)
            with self._upload_lock:
                if not os.path.isfile(path):
                    raise ApiError(404, f"no upload named {os.path.basename(path)!r}")
                with open(path, encoding="utf-8", errors="replace") as fh:
                    blob = fh.read()
            if whole_file:
                if blob.strip():
                    texts.append(blob)
            else:
                texts.extend(line for line in blob.splitlines() if line.strip())
        return texts

    # -- Ollama (local LLM) --------------------------------------------------

    def ollama_client(self, url: str | None = None, model: str | None = None, timeout: float | None = None) -> OllamaClient:
        """A client for the request's Ollama overrides, falling back to the server defaults."""
        try:
            return OllamaClient(url or self.ollama_url, model or self.ollama_model, timeout)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc

    def sample_texts(
        self, count: int, prefix: str = "", max_length: int = 60, temperature: float = 1.0, seed: int | None = None
    ) -> list[str]:
        """``count`` stochastic texts from the model (held under the lock only while sampling)."""
        with self.session() as model:
            return sample_texts(model, count, prefix=prefix, max_length=max_length, temperature=temperature, seed=seed)

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop a running job and wait (bounded) for its thread to finish."""
        job = self._job
        if job is None or not job.running:
            return
        job.stop_event.set()
        self._log(f"stopping {job.type} job {job.id} ...")
        thread = job.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _log(self, message: str) -> None:
        if not self.quiet:
            sys.stderr.write(message + "\n")

    def _log_exception(self, message: str) -> None:
        if not self.quiet:
            sys.stderr.write(f"{message}\n{traceback.format_exc()}")


def _sample_dict(result: PathResult) -> dict:
    return {
        "text": result.text,
        "cost": result.cost,
        "path": list(result.labels),
        "node_ids": list(result.node_ids),
        "step_costs": list(result.step_costs),
        "reached_end": result.reached_end,
    }


def _graph_view(graph: RadixCyclicGraph, limit: int) -> dict:
    """Top-``limit`` alive nodes by visit count (ties: lowest id) plus START/END, and the edges among them."""
    alive = graph.alive
    count = graph.count
    real = [i for i in range(2, len(graph.labels)) if alive[i]]
    top = heapq.nsmallest(limit, real, key=lambda i: (-count[i], i)) if limit < len(real) else real
    ids = [START, END, *sorted(top)]
    chosen = set(ids)
    labels, z, a, b, h, k = graph.labels, graph.z, graph.a, graph.b, graph.h, graph.k
    nodes = [
        {
            "id": i, "label": labels[i], "count": count[i], "activation": graph.activation_of(i),
            "z": z[i], "a": a[i], "b": b[i], "h": h[i], "k": k[i],
        }
        for i in ids
    ]
    edge_w = graph.edge_w
    edge_count = graph.edge_count
    edges = []
    for p in ids:
        for c, e, cost in graph.child_costs(p):
            if c in chosen:
                edges.append(
                    {
                        "source": p, "target": c, "weight": edge_w[e], "count": edge_count[e],
                        "prob": math.exp(-cost), "cost": cost,
                    }
                )
    edges.sort(key=lambda d: (d["source"], d["target"]))
    return {
        "nodes": nodes, "edges": edges, "limit": limit,
        "total_nodes": graph.num_nodes(), "total_edges": graph.num_edges(),
    }


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

_MISSING = object()


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def parse_body(raw: bytes) -> dict:
    """Decode a JSON object body (empty body -> ``{}``); 400 for anything else."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise ApiError(400, f"request body must be UTF-8 encoded JSON ({exc.reason})") from exc
    except ValueError as exc:
        raise ApiError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise ApiError(400, f"JSON body must be an object, not {_json_type(data)}")
    return data


class Fields:
    """Typed access to a JSON body; every problem is an :class:`ApiError` 400.

    A JSON ``null`` counts as "not given": optional fields fall back to their
    default and required fields report a missing field.
    """

    __slots__ = ("_body",)

    def __init__(self, body: dict) -> None:
        self._body = body

    def _lookup(self, name: str) -> Any:
        value = self._body.get(name)
        return _MISSING if value is None else value

    def present(self, name: str) -> bool:
        return self._lookup(name) is not _MISSING

    @staticmethod
    def _default(name: str, default: Any, kind: str) -> Any:
        if default is _MISSING:
            raise ApiError(400, f"missing field '{name}' ({kind})")
        return default

    @staticmethod
    def _bad(name: str, expected: str, value: Any) -> ApiError:
        return ApiError(400, f"'{name}' must be {expected} (got {_json_type(value)} {value!r})")

    def text(self, name: str, default: Any = _MISSING) -> Any:
        value = self._lookup(name)
        if value is _MISSING:
            return self._default(name, default, "string")
        if not isinstance(value, str):
            raise self._bad(name, "a string", value)
        return value

    def flag(self, name: str, default: Any = _MISSING) -> Any:
        value = self._lookup(name)
        if value is _MISSING:
            return self._default(name, default, "boolean")
        if not isinstance(value, bool):
            raise self._bad(name, "a boolean", value)
        return value

    def integer(self, name: str, default: Any = _MISSING, minimum: int | None = None) -> Any:
        value = self._lookup(name)
        if value is _MISSING:
            return self._default(name, default, "integer")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._bad(name, "an integer", value)
        if isinstance(value, float):
            if not value.is_integer():
                raise self._bad(name, "an integer", value)
            value = int(value)
        if minimum is not None and value < minimum:
            raise ApiError(400, f"'{name}' must be >= {minimum} (got {value})")
        return value

    def number(self, name: str, default: Any = _MISSING, minimum: float | None = None) -> Any:
        value = self._lookup(name)
        if value is _MISSING:
            return self._default(name, default, "number")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise self._bad(name, "a finite number", value)
        if minimum is not None and value < minimum:
            raise ApiError(400, f"'{name}' must be >= {minimum} (got {value})")
        return float(value)

    def texts(self, list_name: str, text_name: str) -> list[str]:
        """``list_name`` (list of strings) or ``text_name`` (one text per line, blank lines dropped)."""
        value = self._lookup(list_name)
        if value is not _MISSING:
            if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
                raise self._bad(list_name, "a list of strings", value)
            texts = list(value)
        else:
            blob = self._lookup(text_name)
            if blob is _MISSING:
                raise ApiError(
                    400, f"missing field '{list_name}' (list of strings) or '{text_name}' (string, one text per line)"
                )
            if not isinstance(blob, str):
                raise self._bad(text_name, "a string", blob)
            texts = [line for line in blob.splitlines() if line.strip()]
        if not texts:
            raise ApiError(400, f"'{list_name}' contains no texts")
        return texts

    def texts_optional(self, list_name: str, text_name: str) -> list[str]:
        """Like :meth:`texts`, but absent or empty inputs give ``[]`` instead of 400."""
        if not self.present(list_name) and not self.present(text_name):
            return []
        try:
            return self.texts(list_name, text_name)
        except ApiError as exc:
            if exc.message.endswith("contains no texts"):
                return []
            raise

    def names(self, name: str) -> list[str]:
        """An optional list of non-empty strings (upload names); absent -> ``[]``."""
        value = self._lookup(name)
        if value is _MISSING:
            return []
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise self._bad(name, "a list of non-empty strings", value)
        return list(value)

    def upload_files(self) -> list[tuple[str, str]]:
        """``{name, content}`` or ``{files: [{name, content}, ...]}`` -> ``[(name, content), ...]``."""
        files = self._lookup("files")
        if files is _MISSING:
            return [(self.text("name"), self.text("content"))]
        if not isinstance(files, list) or not files:
            raise self._bad("files", "a non-empty list of {name, content} objects", files)
        result = []
        for i, item in enumerate(files):
            if not isinstance(item, dict):
                raise ApiError(400, f"'files[{i}]' must be an object with 'name' and 'content'")
            entry = Fields(item)
            result.append((entry.text("name"), entry.text("content")))
        return result


_UPLOAD_NAME_CHARS = frozenset("._- +@()")
MAX_UPLOAD_NAME = 128


def sanitize_upload_name(name: str) -> str:
    """A safe file name inside the upload directory (base name only, no traversal, no odd characters)."""
    base = os.path.basename(str(name).replace("\\", "/")).strip()
    base = "".join(ch if ch.isalnum() or ch in _UPLOAD_NAME_CHARS else "_" for ch in base).strip(" .")
    if not base or base in (".", ".."):
        raise ApiError(400, f"invalid upload name {name!r}")
    return base[:MAX_UPLOAD_NAME]


def _upload_record(root: str, name: str) -> dict:
    path = os.path.join(root, name)
    info = os.stat(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        blob = fh.read()
    return {
        "name": name,
        "bytes": info.st_size,
        "chars": len(blob),
        "lines": sum(1 for line in blob.splitlines() if line.strip()),
        "modified": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(timespec="milliseconds"),
    }


def _decode_text(data: bytes) -> str:
    """Bytes of an uploaded text file -> ``str`` (UTF-8, BOM dropped, undecodable bytes replaced)."""
    return data.decode("utf-8-sig", errors="replace")


def _multipart_files(body: bytes, content_type: str) -> list[dict]:
    """File parts of a ``multipart/form-data`` body (``curl -F file=@corpus.txt``)."""
    prologue = b"Content-Type: " + content_type.encode("latin-1", errors="replace") + b"\r\nMIME-Version: 1.0\r\n\r\n"
    try:
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(prologue + body)
    except Exception as exc:  # noqa: BLE001 - the parser's errors are not worth distinguishing
        raise ApiError(400, f"malformed multipart/form-data body: {exc}") from exc
    if not message.is_multipart():
        raise ApiError(400, "malformed multipart/form-data body (missing boundary?)")
    files = []
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue  # plain form fields are ignored
        payload = part.get_payload(decode=True) or b""
        files.append({"name": filename, "content": _decode_text(payload)})
    if not files:
        raise ApiError(400, "multipart body contains no file parts (use -F file=@corpus.txt)")
    return files


def _upload_body(body: bytes, content_type: str, query: dict[str, list[str]]) -> dict:
    """Body of ``POST /api/uploads`` in any accepted form, normalised to the JSON form.

    * ``application/json``: ``{name, content}`` or ``{files: [{name, content}, ...]}``
    * ``multipart/form-data``: every part with a file name
    * anything else (``text/plain``, ``--data-binary @file``): the raw body, named by ``?name=``
    """
    kind = content_type.split(";", 1)[0].strip().lower()
    names = [n for n in query.get("name", []) if n.strip()]
    if kind == "multipart/form-data":
        return {"files": _multipart_files(body, content_type)}
    if kind == "application/json" or (kind == "" and not names):
        return parse_body(body)
    if not names:
        raise ApiError(400, "raw uploads need a ?name=<file name> query parameter (or send JSON {name, content})")
    return {"files": [{"name": names[0], "content": _decode_text(body)}]}


def _texts_and_files(
    svc: ModelService, fields: Fields, list_name: str, text_name: str, files_name: str, whole_file: bool = False
) -> list[str]:
    """Texts given inline and/or read from uploaded files; at least one text is required."""
    names = fields.names(files_name)
    texts = fields.texts_optional(list_name, text_name)
    if names:
        texts.extend(svc.upload_texts(names, whole_file=whole_file))
    if not texts:
        if names:
            raise ApiError(400, f"'{files_name}' contains no texts")
        if fields.present(list_name) or fields.present(text_name):
            raise ApiError(400, f"'{list_name}' contains no texts")
        raise ApiError(
            400,
            f"missing field '{list_name}' (list of strings), '{text_name}' (string, one text per line)"
            f" or '{files_name}' (list of upload names)",
        )
    return texts


def _train_overrides(fields: Fields) -> dict:
    """Optional :class:`TrainConfig` fields shared by the 2NRL phases (only those given)."""
    overrides: dict[str, Any] = {}
    if fields.present("batch_size"):
        overrides["batch_size"] = fields.integer("batch_size", minimum=1)
    if fields.present("auto_compress"):
        overrides["auto_compress"] = fields.flag("auto_compress")
    if fields.present("clip"):
        overrides["clip"] = fields.number("clip")
    if fields.present("shuffle"):
        overrides["shuffle"] = fields.flag("shuffle")
    return overrides


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

RouteFn = Callable[[ModelService, Fields, dict[str, list[str]]], tuple[int, Any]]


def _r_health(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, {"ok": True, "version": __version__}


def _r_status(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.status()


def _train_config(f: Fields) -> TrainConfig:
    """The optional training settings of a request body, defaults from :class:`TrainConfig`."""
    return TrainConfig(
        epochs=f.integer("epochs", TrainConfig.epochs, minimum=0),
        lr=f.number("lr", TrainConfig.lr, minimum=0.0),
        act_lr=f.number("act_lr", TrainConfig.act_lr, minimum=0.0),
        batch_size=f.integer("batch_size", TrainConfig.batch_size, minimum=1),
        clip=f.number("clip", TrainConfig.clip),
        auto_compress=f.flag("auto_compress", TrainConfig.auto_compress),
        shuffle=f.flag("shuffle", TrainConfig.shuffle),
        checkpoint_every=f.integer("checkpoint_every", TrainConfig.checkpoint_every, minimum=0),
    )


def _r_train(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    texts = _texts_and_files(svc, f, "texts", "text", "files", whole_file=f.flag("whole_file", False))
    return 202, {"job": svc.start_train(texts, _train_config(f))}


def _r_job(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.job_status()


def _r_job_stop(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.stop_job()


def _r_predict(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    prefix = f.text("prefix")
    return 200, svc.predict(
        prefix,
        length=f.integer("length", 20, minimum=0),
        mode=f.text("mode", "dijkstra"),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        to_end=f.flag("to_end", False),
        max_length=f.integer("max_length", None, minimum=0),
    )


def _r_generate(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.generate(
        count=f.integer("count", 1, minimum=0),
        max_length=f.integer("max_length", 60, minimum=0),
        mode=f.text("mode", "sample"),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        seed=f.integer("seed", None),
    )


def _r_score(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.score(f.text("text"))


def _r_two_nrl(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    whole_file = f.flag("whole_file", False)
    bad = _texts_and_files(svc, f, "bad", "bad_text", "bad_files", whole_file=whole_file)
    good = _texts_and_files(svc, f, "good", "good_text", "good_files", whole_file=whole_file)
    job = svc.start_two_nrl(
        bad, good,
        neg_epochs=f.integer("neg_epochs", 3, minimum=0),
        pos_epochs=f.integer("pos_epochs", 3, minimum=0),
        neg_lr=f.number("neg_lr", 0.05, minimum=0.0),
        pos_lr=f.number("pos_lr", 0.01, minimum=0.0),
        **_train_overrides(f),
    )
    return 202, {"job": job}


def _r_invert(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.invert()


def _r_compress(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.compress()


def _r_evolve_start(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    corpus = _texts_and_files(svc, f, "corpus", "corpus_text", "corpus_files", whole_file=f.flag("whole_file", False))
    generations = f.integer("generations", None, minimum=0)
    d = EvolveConfig
    config = EvolveConfig(
        samples=f.integer("samples", d.samples, minimum=1),
        real_per_generation=f.integer("real_per_generation", d.real_per_generation, minimum=1),
        max_length=f.integer("max_length", d.max_length, minimum=0),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        neg_epochs=f.integer("neg_epochs", d.neg_epochs, minimum=0),
        pos_epochs=f.integer("pos_epochs", d.pos_epochs, minimum=0),
        neg_lr=f.number("neg_lr", d.neg_lr, minimum=0.0),
        pos_lr=f.number("pos_lr", d.pos_lr, minimum=0.0),
        disc_neg_epochs=f.integer("disc_neg_epochs", d.disc_neg_epochs, minimum=0),
        disc_pos_epochs=f.integer("disc_pos_epochs", d.disc_pos_epochs, minimum=0),
        batch_size=f.integer("batch_size", d.batch_size, minimum=1),
        checkpoint_every=f.integer("checkpoint_every", d.checkpoint_every, minimum=0),
        seed=f.integer("seed", d.seed),
    )
    return 202, {"job": svc.start_evolve(corpus, generations, config)}


def _r_evolve_stop(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.stop_evolve()


def _r_evolve_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.evolve_history()


def _r_save(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.save(f.text("path", None))


def _r_load(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    path = f.text("path")
    if not path:
        raise ApiError(400, "'path' must not be empty")
    return 200, svc.load(path)


def _r_reset(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.reset(f.integer("seed", None))


def _r_checkpoints(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.checkpoints()


def _r_checkpoint_save(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.save_checkpoint(f.text("tag", "manual") or "manual")


def _r_checkpoint_restore(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    name = f.text("name")
    if not name:
        raise ApiError(400, "'name' must not be empty")
    return 200, svc.restore_checkpoint(name)


def _r_graph(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    raw = q.get("limit", [str(DEFAULT_GRAPH_LIMIT)])[-1]
    try:
        limit = int(raw)
    except ValueError:
        raise ApiError(400, f"query parameter 'limit' must be an integer (got {raw!r})") from None
    return 200, svc.graph(limit)


def _r_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.history()


def _r_uploads(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.uploads()


def _r_upload(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 201, {"uploads": [svc.upload(name, content) for name, content in f.upload_files()]}


def _r_upload_delete(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.delete_upload(f.text("name"))


def _ollama_client(svc: ModelService, f: Fields) -> OllamaClient:
    return svc.ollama_client(f.text("url", None), f.text("model", None), f.number("timeout", None, minimum=1.0))


def _r_ollama_models(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    url = (q.get("url") or [None])[0]
    client = svc.ollama_client(url or None)
    error = None
    try:
        models = client.models()
    except OllamaError as exc:
        models, error = [], str(exc)
    return 200, {
        "available": error is None,
        "url": client.url,
        "model": client.model,
        "models": [
            {"name": m.get("name"), "size": m.get("size"), "modified_at": m.get("modified_at"), "details": m.get("details")}
            for m in models
        ],
        "error": error,
    }


def _r_ollama_corpus(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    prompt = f.text("prompt")
    lines = f.integer("lines", 20, minimum=1)
    style = f.text("style", "good").strip().lower()
    if style not in OLLAMA_STYLES:
        raise ApiError(400, f"'style' must be one of {', '.join(OLLAMA_STYLES)} (got {style!r})")
    train = f.flag("train", False)
    client = _ollama_client(svc, f)
    if train:
        svc._ensure_idle()  # do not spend an LLM call on a request that cannot start a job
    try:
        texts = corpus_from_prompt(client, prompt, lines=lines, style=style, model=client.model)
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    if not texts:
        raise ApiError(502, f"Ollama model {client.model!r} returned no usable lines")
    result: dict[str, Any] = {
        "prompt": prompt, "style": style, "model": client.model, "url": client.url,
        "lines": len(texts), "texts": texts, "upload": None, "job": None,
    }
    save_as = f.text("save_as", None)
    if save_as:
        result["upload"] = svc.upload(save_as, "\n".join(texts) + "\n")
    if train:
        result["job"] = svc.start_train(texts, _train_config(f))
        return 202, result
    return 200, result


def _r_ollama_review(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    given = f.texts_optional("texts", "text")
    apply = f.text("apply", "none").strip().lower()
    if apply not in ("none", "2nrl"):
        raise ApiError(400, f"'apply' must be 'none' or '2nrl' (got {apply!r})")
    client = _ollama_client(svc, f)
    threshold = f.number("threshold", 6.0, minimum=0.0)
    if apply == "2nrl":
        svc._ensure_idle()
    if given:
        samples, source = given, "given"
    else:
        samples = svc.sample_texts(
            f.integer("count", 8, minimum=1), prefix=f.text("prefix", ""),
            max_length=f.integer("max_length", 60, minimum=0), temperature=f.number("temperature", 1.0, minimum=0.0),
            seed=f.integer("seed", None),
        )
        source = "model"
    try:
        result = adversarial_review(
            None, client, texts=samples, threshold=threshold, context=f.text("context", None), ollama_model=client.model,
        )
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    result["source"] = source
    result["url"] = client.url
    result["job"] = None
    if apply != "2nrl":
        return 200, result
    bad = list(result["bad"])
    good = list(result["good"]) + f.texts_optional("good", "good_text")
    good_files = f.names("good_files")
    if good_files:
        good.extend(svc.upload_texts(good_files, whole_file=f.flag("whole_file", False)))
    if not bad:
        raise ApiError(400, "nothing failed the review, so there is no garbage for the negative phase")
    if not good:
        raise ApiError(400, "no text passed the review and no good texts were given for the positive phase")
    result["job"] = svc.start_two_nrl(
        bad, good,
        neg_epochs=f.integer("neg_epochs", 3, minimum=0),
        pos_epochs=f.integer("pos_epochs", 3, minimum=0),
        neg_lr=f.number("neg_lr", 0.05, minimum=0.0),
        pos_lr=f.number("pos_lr", 0.01, minimum=0.0),
        **_train_overrides(f),
    )
    return 202, result


_ENDPOINTS: tuple[tuple[str, str, RouteFn, str], ...] = (
    ("GET", "/api/health", _r_health, "liveness check and package version"),
    ("GET", "/api/status", _r_status, "model stats, current job, backend availability, paths"),
    ("POST", "/api/train", _r_train, "start a training job: {texts | text, epochs, lr, act_lr, batch_size, auto_compress}"),
    ("GET", "/api/job", _r_job, "status of the current / last job"),
    ("POST", "/api/job/stop", _r_job_stop, "ask the running job to stop"),
    ("POST", "/api/predict", _r_predict, "continue a prefix: {prefix, length, mode, to_end, step_penalty, temperature, max_length (optional cap; default none)}"),
    ("POST", "/api/generate", _r_generate, "sample texts from START: {count, max_length, mode, temperature}"),
    ("POST", "/api/score", _r_score, "log-probability of a text: {text}"),
    ("POST", "/api/2nrl", _r_two_nrl, "start a 2NRL job: {bad | bad_text, good | good_text, neg_epochs, pos_epochs, neg_lr, pos_lr}"),
    ("POST", "/api/invert", _r_invert, "invert the network"),
    ("POST", "/api/compress", _r_compress, "merge unary chains"),
    ("POST", "/api/evolve/start", _r_evolve_start, "start the GAN-style loop: {corpus | corpus_text, generations, samples, ...}"),
    ("POST", "/api/evolve/stop", _r_evolve_stop, "stop the evolve job"),
    ("GET", "/api/evolve/history", _r_evolve_history, "generation records of all evolve runs"),
    ("POST", "/api/save", _r_save, "save the model: {path} (default: the server's model path)"),
    ("POST", "/api/load", _r_load, "load a model file: {path}"),
    ("POST", "/api/reset", _r_reset, "replace the model with a fresh one: {seed}"),
    ("GET", "/api/checkpoints", _r_checkpoints, "list checkpoints and the latest pointer"),
    ("POST", "/api/checkpoints/save", _r_checkpoint_save, "write a checkpoint: {tag}"),
    ("POST", "/api/checkpoints/restore", _r_checkpoint_restore, "restore a checkpoint: {name}"),
    ("GET", "/api/graph", _r_graph, "top nodes by visit count and the edges among them (?limit=150)"),
    ("GET", "/api/history", _r_history, "the model's training history"),
    ("GET", "/api/uploads", _r_uploads, "uploaded training files (name, bytes, lines)"),
    ("POST", "/api/uploads", _r_upload, "upload text files: JSON {name, content} | {files: [...]}, multipart/form-data, or a raw body with ?name="),
    ("POST", "/api/uploads/delete", _r_upload_delete, "delete an uploaded file: {name}"),
    ("GET", "/api/ollama/models", _r_ollama_models, "models installed in Ollama (?url= overrides the server default); never fails"),
    ("POST", "/api/ollama/corpus", _r_ollama_corpus,
     "training lines from a prompt: {prompt, lines, style: good|garbage, model, url, save_as, train, epochs, lr, batch_size}"),
    ("POST", "/api/ollama/review", _r_ollama_review,
     "adversarial LLM review of the model's samples or {texts}: {count, prefix, max_length, threshold, apply: none|2nrl, good, good_files}"),
)

_ROUTES: dict[str, dict[str, RouteFn]] = {}
for _method, _path, _fn, _doc in _ENDPOINTS:
    _ROUTES.setdefault(_path, {})[_method] = _fn
del _method, _path, _fn, _doc


# ---------------------------------------------------------------------------
# static files and the built-in page
# ---------------------------------------------------------------------------


def _resolve_static(root: str, url_path: str) -> str | None:
    """Absolute path of the file ``url_path`` names under ``root`` (``None`` if absent).

    Raises ``PermissionError`` when the path escapes ``root``: ``..``
    segments (encoded or not), backslashes, NUL bytes, or symlinks that
    resolve outside the directory.  Directories resolve to ``index.html``.
    """
    raw = unquote(url_path)
    if "\x00" in raw:
        raise PermissionError(url_path)
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." or "\\" in p for p in parts):
        raise PermissionError(url_path)
    candidate = os.path.join(root, *parts) if parts else root
    try:
        real = os.path.realpath(candidate)
    except OSError:
        return None
    if real != root and not real.startswith(root + os.sep):
        raise PermissionError(url_path)
    if os.path.isdir(real):
        real = os.path.join(real, "index.html")
    return real if os.path.isfile(real) else None


def _is_spa_route(url_path: str) -> bool:
    """Paths without a file extension are client-side routes (served index.html)."""
    return "." not in url_path.rsplit("/", 1)[-1]


def _content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _MIME_TYPES.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def _cache_control(rel_path: str) -> str:
    if rel_path.startswith("assets/") and not rel_path.endswith(".html"):
        return "public, max-age=31536000, immutable"  # Vite emits content-hashed file names
    return "no-cache"


def _help_page(frontend_dir: str | None) -> str:
    """Built-in HTML shown when the frontend is not built."""
    where = f" in <code>{html.escape(frontend_dir)}</code>" if frontend_dir else ""
    rows = []
    for method, path, _fn, doc in _ENDPOINTS:
        target = f'<a href="{html.escape(path)}">{html.escape(path)}</a>' if method == "GET" else html.escape(path)
        rows.append(f"<tr><td><code>{method}</code></td><td><code>{target}</code></td><td>{html.escape(doc)}</td></tr>")
    table_rows = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadixCyclicNN API</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1f2937; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.95em; }}
pre {{ background: #f3f4f6; padding: 0.75rem 1rem; border-radius: 6px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
</style>
</head>
<body>
<h1>RadixCyclicNN API <small>v{html.escape(__version__)}</small></h1>
<p>The JSON API is running, but the React frontend is not built{where}.</p>
<h2>Build the frontend</h2>
<pre>cd frontend
npm install
npm run build                      # writes frontend/dist
python -m radixnet serve --frontend-dir frontend/dist</pre>
<p>Then reload this page. During development <code>npm run dev</code> proxies <code>/api</code> to this server.</p>
<h2>Endpoints</h2>
<p>All bodies and responses are JSON; errors are <code>{{"error": "message"}}</code>.
Long operations (train, 2nrl, evolve) return a job to poll at <code>GET /api/job</code>.</p>
<table>
<thead><tr><th>method</th><th>path</th><th>description</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler and server
# ---------------------------------------------------------------------------


class ApiHandler(BaseHTTPRequestHandler):
    """Request handler: JSON API under ``/api/``, static frontend elsewhere."""

    protocol_version = "HTTP/1.1"
    server_version = f"radixnet/{__version__}"
    timeout = 60.0  # idle keep-alive connections are closed after this
    error_content_type = _JSON_CONTENT_TYPE
    error_message_format = '{"error": "HTTP %(code)d: %(explain)s"}\n'

    server: "RadixNetHTTPServer"

    # -- entry points --------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def _dispatch(self, method: str) -> None:
        t0 = time.perf_counter()
        status = 500
        try:
            status = self._handle(method)
        except ApiError as exc:
            status = self._send_json(exc.status, {"error": exc.message}, method)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - last resort, keep the server alive
            self._log_exception(f"unhandled error handling {method} {self.path}")
            with contextlib.suppress(OSError):
                status = self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"}, method)
        finally:
            self._log_line(method, status, t0)

    def _handle(self, method: str) -> int:
        if self.path.startswith(("http://", "https://")):  # absolute-form target (proxies)
            url = urlsplit(self.path)
            path, query = url.path, url.query
        else:
            path, _, query = self.path.partition("?")
        path = path or "/"
        body = self._read_body()
        if method == "OPTIONS":
            return self._send(204, b"", None, method)
        if path == "/api" or path.startswith("/api/"):
            return self._handle_api(method, path.rstrip("/") or "/api", query, body)
        if method in ("GET", "HEAD"):
            return self._handle_static(method, path)
        return self._send_json(405, {"error": f"method {method} is not allowed for {path}"}, method, allow="GET, HEAD, OPTIONS")

    # -- API -----------------------------------------------------------------

    def _handle_api(self, method: str, path: str, query: str, body: bytes) -> int:
        route = _ROUTES.get(path)
        if route is None:
            return self._send_json(404, {"error": f"unknown API endpoint {path}"}, method)
        lookup = "GET" if method == "HEAD" else method
        fn = route.get(lookup)
        if fn is None:
            allow = ", ".join(sorted(route) + ["OPTIONS"])
            return self._send_json(405, {"error": f"method {method} is not allowed for {path}; use {allow}"}, method, allow=allow)
        service = self.server.service
        try:
            params = parse_qs(query, keep_blank_values=True)
            if lookup == "POST" and path == "/api/uploads":
                fields = Fields(_upload_body(body, self.headers.get("Content-Type", ""), params))
            else:
                fields = Fields(parse_body(body) if lookup == "POST" else {})
            status, payload = fn(service, fields, params)
        except ApiError as exc:
            status, payload = exc.status, {"error": exc.message}
        except FileNotFoundError as exc:
            status, payload = 404, {"error": _os_error_text(exc)}
        except (ValueError, TypeError) as exc:
            status, payload = 400, {"error": str(exc) or type(exc).__name__}
        except OSError as exc:
            status, payload = 400, {"error": _os_error_text(exc)}
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged here
            self._log_exception(f"{method} {path} failed")
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        return self._send_json(status, payload, method)

    # -- static files ----------------------------------------------------------

    def _handle_static(self, method: str, path: str) -> int:
        root = self.server.frontend_root()
        if root is None:
            if path in ("/", "/index.html"):
                return self._send(200, self.server.help_page, _HTML_CONTENT_TYPE, method, cache="no-cache")
            return self._send_json(404, {"error": f"not found: {path}"}, method)
        try:
            target = _resolve_static(root, path)
        except PermissionError:  # escapes the frontend directory: never fall back
            return self._send_json(404, {"error": f"not found: {path}"}, method)
        if target is None:
            if not _is_spa_route(path):
                return self._send_json(404, {"error": f"not found: {path}"}, method)
            target = os.path.join(root, "index.html")
        try:
            with open(target, "rb") as fh:
                data = fh.read()
        except OSError:
            return self._send_json(404, {"error": f"not found: {path}"}, method)
        rel = os.path.relpath(target, root).replace(os.sep, "/")
        return self._send(200, data, _content_type(target), method, cache=_cache_control(rel))

    # -- plumbing ------------------------------------------------------------

    def _read_body(self) -> bytes:
        """Consume the request body (always, so keep-alive connections stay in sync)."""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True
            raise ApiError(411, "chunked request bodies are not supported; send a Content-Length header")
        raw = self.headers.get("Content-Length")
        if raw is None:
            return b""
        try:
            length = int(raw)
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            raise ApiError(400, f"invalid Content-Length header {raw!r}") from None
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            raise ApiError(413, f"request body too large ({length} bytes; the limit is {MAX_BODY_BYTES})")
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            self.close_connection = True
            raise ApiError(400, "incomplete request body")
        return data

    def _send_json(self, status: int, payload: Any, method: str, allow: str | None = None) -> int:
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            status = 500
            body = json.dumps({"error": f"response is not JSON-serialisable: {exc}"}).encode("utf-8")
        extra = (("Allow", allow),) if allow else ()
        return self._send(status, body, _JSON_CONTENT_TYPE, method, extra=extra)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str | None,
        method: str,
        extra: tuple[tuple[str, str], ...] = (),
        cache: str | None = None,
    ) -> int:
        """Write one complete response (CORS headers, exact Content-Length)."""
        self.send_response(status)
        for name, value in _CORS_HEADERS:
            self.send_header(name, value)
        if content_type:
            self.send_header("Content-Type", content_type)
        if status != 204:
            self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        if body and method != "HEAD" and status != 204:
            self.wfile.write(body)
        return status

    # -- logging -------------------------------------------------------------

    def _log_line(self, method: str, status: int, t0: float) -> None:
        if not self.server.quiet:
            ms = (time.perf_counter() - t0) * 1000.0
            sys.stderr.write(f"{method} {self.path} {status} {ms:.1f}ms\n")

    def _log_exception(self, message: str) -> None:
        if not self.server.quiet:
            sys.stderr.write(f"{message}\n{traceback.format_exc()}")

    def version_string(self) -> str:
        return self.server_version

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        """Silenced: :meth:`_log_line` logs each request with its duration."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base-class signature
        if not self.server.quiet:
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def _os_error_text(exc: OSError) -> str:
    if exc.filename is not None:
        return f"{exc.strerror or type(exc).__name__}: {exc.filename}"
    return str(exc) or type(exc).__name__


class RadixNetHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` carrying the service and static configuration."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: ModelService,
        frontend_dir: str | None = None,
        quiet: bool = False,
    ) -> None:
        self.service = service
        self.frontend_dir = frontend_dir
        self.quiet = bool(quiet)
        self.help_page = _help_page(frontend_dir).encode("utf-8")
        super().__init__(address, ApiHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"

    def frontend_root(self) -> str | None:
        """Real path of the built frontend, or ``None`` when ``index.html`` is missing.

        Evaluated per request, so building the frontend while the server
        runs makes it appear without a restart.
        """
        if not self.frontend_dir:
            return None
        root = os.path.realpath(self.frontend_dir)
        return root if os.path.isfile(os.path.join(root, "index.html")) else None


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_path: str | None = None,
    checkpoint_dir: str | None = None,
    frontend_dir: str | None = None,
    backend: str = "auto",
    device: str | None = None,
    seed: int = 0,
    quiet: bool = False,
    upload_dir: str | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
) -> tuple[RadixNetHTTPServer, ModelService]:
    """Build (and bind) the server; ``port=0`` picks a free port.

    ``model_path`` is loaded when it exists and is the default target of
    ``POST /api/save``; ``checkpoint_dir`` enables the checkpoint endpoints;
    ``upload_dir`` enables the upload endpoints (training files kept on the
    server); ``frontend_dir`` is the built React app; ``ollama_url`` /
    ``ollama_model`` are the defaults of the ``/api/ollama/*`` endpoints
    (``$OLLAMA_HOST`` / ``$RADIXNET_OLLAMA_MODEL`` when omitted).  ``quiet``
    silences the per-request log lines (stderr).
    """
    service = ModelService(
        model_path=model_path, checkpoint_dir=checkpoint_dir, backend=backend, device=device,
        seed=seed, quiet=quiet, upload_dir=upload_dir, ollama_url=ollama_url, ollama_model=ollama_model,
    )
    server = RadixNetHTTPServer((host, port), service, frontend_dir=frontend_dir, quiet=quiet)
    return server, service


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_path: str | None = None,
    checkpoint_dir: str | None = None,
    frontend_dir: str | None = None,
    backend: str = "auto",
    device: str | None = None,
    seed: int = 0,
    quiet: bool = False,
    upload_dir: str | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
) -> None:
    """Serve until ``KeyboardInterrupt``; a running job is stopped on the way out."""
    server, service = create_server(
        host, port, model_path=model_path, checkpoint_dir=checkpoint_dir, frontend_dir=frontend_dir,
        backend=backend, device=device, seed=seed, quiet=quiet, upload_dir=upload_dir,
        ollama_url=ollama_url, ollama_model=ollama_model,
    )
    if not quiet:
        sys.stderr.write(f"radixnet API listening on {server.url} (Ctrl-C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            sys.stderr.write("\ninterrupted; shutting down\n")
    finally:
        service.shutdown()
        server.server_close()
