"""HTTP JSON API and static file serving for RadixNet (standard library only).

Architecture
------------
* :class:`ModelService` owns one model (a :class:`~radixnet.model.RadixNet` or a
  :class:`~radixnet.countnet.CountRewardNet`, switchable at run time) behind an
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

import base64
import binascii
import contextlib
import dataclasses
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
from .archive import extract_texts, is_zip
from .backend import describe_backends
from .checkpoint import CheckpointManager
from .chatgpt import (
    DEFAULT_MODEL as CHATGPT_DEFAULT_MODEL,
    DEFAULT_URL as CHATGPT_DEFAULT_URL,
    ChatGPTClient,
    api_key_configured as chatgpt_key_configured,
)
from .chatgpt import normalise_url as normalise_chatgpt_url
from .codegen import (
    PHASES as CODEGEN_PHASES,
    CodeGenConfig,
    CodeGenTrainer,
    Problem,
    Sandbox,
    check_style,
    decide,
    default_teacher_model,
    parse_problem_file,
    parse_problems,
)
from .gan import EvolveConfig, Evolver
from .graph import END, START, RadixCyclicGraph
from .llm import PROVIDERS, LLMClient, LLMError, normalise_provider
from .beam import Prediction, path_probability
from .dialogue import DEFAULT_SPEAKERS
from .duo import FilterConfig, NegativeFilter
from .model import GraphModel, RadixNet, TrainConfig, load_model, model_class, model_kinds, new_model
from .negative import NegativeNet
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
from .blame import teach_reviews
from .schedule import ScheduleError
from .schedule import describe as describe_schedules
from .schedule import preview_points
from .search import PathResult
from .speech import DEFAULT_RATE as SPEECH_DEFAULT_RATE
from .speech import SpeechError
from .speech import decode_text as decode_speech_text
from .speech import describe as describe_speech
from .speech import teach as teach_speech
from .speech import transcribe as transcribe_speech
from .tutor import (
    DEFAULT_PLAN_LESSONS,
    DEFAULT_TUTOR_MODEL,
    ERROR_TYPES as TUTOR_ERROR_TYPES,
    LEVELS as TUTOR_LEVELS,
    MODES as TUTOR_MODES,
    TWONRL_PER as TUTOR_TWONRL_PER,
    Exercise,
    TutorConfig,
    TutorTrainer,
    default_tutor_model,
    plan_lessons,
    report_card,
)
from .vision import VisionError
from .vision import decode_text as decode_image_text
from .vision import describe as describe_vision
from .vision import encode_image

__all__ = [
    "ApiError",
    "Job",
    "ModelService",
    "ApiHandler",
    "RadixNetHTTPServer",
    "create_server",
    "run_server",
    "MAX_BODY_BYTES",
    "MAX_UPLOAD_BYTES",
    "DEFAULT_GRAPH_LIMIT",
]

MAX_BODY_BYTES = 64 * 1024 * 1024
"""Largest accepted JSON request body (a training corpus can be big)."""
MAX_UPLOAD_BYTES: int | None = None
"""Largest accepted ``POST /api/uploads`` body: ``None`` = no limit (a ZIP of a whole source tree is fine)."""
_FILTER_FIELDS = frozenset(f.name for f in dataclasses.fields(FilterConfig))
"""Request fields that configure the filter rather than the generation behind it."""

_UPLOAD_PATH = "/api/uploads"
_BINARY_ROUTES = {
    "/api/uploads": None,
    "/api/images/encode": "image",
    "/api/speech/transcribe": "speech",
    "/api/speech/teach": "speech",
}
"""POST routes whose bodies may be raw bytes or multipart (value: the default name of a raw body, None = required)."""

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
        chatgpt_url: str | None = None,
        chatgpt_model: str | None = None,
        kind: str | None = None,
    ) -> None:
        self.model_path = os.path.abspath(model_path) if model_path else None
        self.checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else None
        self.upload_dir = os.path.abspath(upload_dir) if upload_dir else None
        self._upload_lock = threading.Lock()
        self._archive_cache: dict[str, tuple[tuple[int, int], dict]] = {}  # path -> ((size, mtime_ns), summary)
        self.ollama_url = normalise_url(ollama_url or OLLAMA_DEFAULT_URL)
        self.ollama_model = (ollama_model or OLLAMA_DEFAULT_MODEL).strip() or OLLAMA_DEFAULT_MODEL
        self.chatgpt_url = normalise_chatgpt_url(chatgpt_url or CHATGPT_DEFAULT_URL)
        self.chatgpt_model = (chatgpt_model or CHATGPT_DEFAULT_MODEL).strip() or CHATGPT_DEFAULT_MODEL
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
        self._codegen_history: list[dict] = []
        self._tutor_history: list[dict] = []
        self._discriminator: GraphModel | None = None
        self._discriminator_seed: int | None = None
        self._parked: dict[str, GraphModel] = {}  # models of the other kinds, kept while another one is active
        self._path_kind = model_class(kind).kind  # the kind ``model_path`` belongs to
        if self.model_path and os.path.isfile(self.model_path):
            self.model: GraphModel = load_model(self.model_path, backend=backend, device=device)
            self._path_kind = self.model.kind
            self.loaded_from: str | None = self.model_path
        else:
            self.model = new_model(self._path_kind, seed=self.seed, backend=backend, device=device)
            self.loaded_from = None

    # -- locking -------------------------------------------------------------

    @contextlib.contextmanager
    def session(self) -> Iterator[GraphModel]:
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
    def mutating(self) -> Iterator[GraphModel]:
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
    def discriminator(self) -> GraphModel | None:
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

    # -- model kinds ---------------------------------------------------------

    @property
    def kind(self) -> str:
        """The active model's kind (``"radix"`` or ``"count"``)."""
        return self.model.kind

    def model_path_for(self, kind: str) -> str | None:
        """Default file of a model kind: ``model_path`` for the kind it belongs to, ``<stem>.<kind><ext>`` otherwise."""
        if not self.model_path:
            return None
        if kind == self._path_kind:
            return self.model_path
        root, ext = os.path.splitext(self.model_path)
        if ext == ".gz":
            root, inner = os.path.splitext(root)
            ext = inner + ext
        return f"{root}.{kind}{ext}"

    def describe_model(self) -> dict:
        """The active kind, every available kind and where each is saved by default."""
        kinds = model_kinds()
        return {
            "kind": self.kind,
            "label": type(self.model).label,
            "kinds": kinds,
            "model_path": self.model_path_for(self.kind),
            "paths": {k["kind"]: self.model_path_for(k["kind"]) for k in kinds},
            "in_memory": sorted({self.kind, *self._parked}),
            "weights": self.model.weight_config() if hasattr(self.model, "weight_config") else None,
        }

    def select_kind(self, kind: str) -> dict:
        """Make ``kind`` the active model: the one kept in memory, else its file, else a fresh one.

        The model that was active stays in memory (unsaved work included) and
        comes back when its kind is selected again; the evolve discriminator
        and history belong to a model and are dropped on a switch.
        """
        cls = model_class(kind)
        self._ensure_idle()
        with self.session():
            self._ensure_idle()
            origin = "active"
            if cls.kind != self.model.kind:
                current = self.model
                model = self._parked.pop(cls.kind, None)
                origin = "memory"
                if model is None:
                    path = self.model_path_for(cls.kind)
                    if path and os.path.isfile(path):
                        model = load_model(path, backend=self.backend_name, device=self.device)
                        origin = "file"
                        if model.kind != cls.kind:
                            raise ApiError(400, f"{path} holds a {model.kind} model, not a {cls.kind} one")
                    else:
                        model = cls(seed=self.seed, backend=self.backend_name, device=self.device)
                        origin = "new"
                self._parked[current.kind] = current
                self.model = model
                self._discriminator = None
                self._discriminator_seed = None
                self._evolve_history = []
                self._log(f"model kind {current.kind} -> {model.kind} ({origin})")
            result = self.describe_model()
            result["origin"] = origin
            result["stats"] = self.model.stats()
            return result

    # -- jobs (continued) ----------------------------------------------------

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
        strength: float | None = None,
        bad_weights: list[float] | None = None,
        good_weights: list[float] | None = None,
        **overrides: Any,
    ) -> dict:
        """Start a ``2nrl`` job (``overrides``: batch_size, auto_compress, clip, shuffle; ``strength``: count model).

        ``bad_weights`` / ``good_weights`` scale the two phases per text: how
        bad a failure is and how good a text is, rather than one rate for all.
        """
        TrainConfig(epochs=neg_epochs, lr=neg_lr, **overrides).validate()
        TrainConfig(epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, **overrides).validate()

        def work(job: Job) -> None:
            self.model.two_nrl(
                bad, good, neg_epochs=neg_epochs, pos_epochs=pos_epochs, neg_lr=neg_lr, pos_lr=pos_lr,
                progress=self._progress(job), stop_event=job.stop_event, strength=strength,
                bad_weights=bad_weights, good_weights=good_weights, **overrides,
            )

        return self._start_job("2nrl", work)

    def start_feedback(
        self,
        good: list[str],
        bad: list[str],
        neg_epochs: int = 2,
        pos_epochs: int = 3,
        neg_lr: float = 0.5,
        pos_lr: float = 0.1,
        strength: float | None = None,
        good_weights: list[float] | None = None,
        bad_weights: list[float] | None = None,
        **overrides: Any,
    ) -> dict:
        """Start a ``feedback`` job from rated texts (thumbs up = ``good``, thumbs down = ``bad``).

        Both sets: ``two_nrl``.  Only ``good``: ``reward``.  Only ``bad``:
        ``punish``.  For RadixNet that is 2NRL (bad, invert, good), a
        positive-phase pass, or a negative-phase pass followed by an
        inversion; the count / reward model penalises and rewards the paths
        by ``strength`` instead.  ``good_weights`` / ``bad_weights`` (one per
        text) turn the thumbs into ratings: each text is learned in
        proportion to how good or bad it was rated.
        """
        action = feedback_action(good, bad)
        if action is None:
            raise ApiError(400, "nothing to learn from: give 'good' (thumbs up) and/or 'bad' (thumbs down) texts")
        TrainConfig(epochs=neg_epochs, lr=neg_lr, **overrides).validate()
        TrainConfig(epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10, **overrides).validate()

        def work(job: Job) -> None:
            progress = self._progress(job)
            if action == "2nrl":
                self.model.two_nrl(
                    bad, good, neg_epochs=neg_epochs, pos_epochs=pos_epochs, neg_lr=neg_lr, pos_lr=pos_lr,
                    progress=progress, stop_event=job.stop_event, strength=strength,
                    bad_weights=bad_weights, good_weights=good_weights, **overrides,
                )
            elif action == "reward":
                self.model.reward(
                    good, epochs=pos_epochs, lr=pos_lr, strength=strength, weights=good_weights, progress=progress,
                    stop_event=job.stop_event, **overrides,
                )
            else:
                self.model.punish(
                    bad, epochs=neg_epochs, lr=neg_lr, strength=strength, weights=bad_weights, progress=progress,
                    stop_event=job.stop_event, **overrides,
                )

        return self._start_job("feedback", work)

    def start_evolve(self, corpus: list[str], generations: int | None, config: EvolveConfig, blame: bool = False) -> dict:
        """Start an ``evolve`` job (``generations`` ``None`` / ``0`` = until stopped).

        The discriminator survives across runs of the same model with the same
        ``config.seed``; its records are appended to :meth:`evolve_history`.
        With ``blame`` the discriminator also teaches the negative network:
        every fake it scores below the real texts is blamed, the real texts
        clear blame.
        """
        if not generations:
            generations = None
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        self._ensure_idle()
        with self.session() as model:
            self._ensure_idle()
            disc = self._discriminator if self._discriminator_seed == config.seed else None
            evolver = Evolver(
                model, corpus, discriminator=disc, config=config,
                negative=self.negative_model() if blame else None,
            )
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
            kind=self.kind,
            model_label=type(self.model).label,
            kinds=model_kinds(),
            job=job.to_dict() if job is not None else None,
            backends=self.backends,
            model_path=self.model_path_for(self.kind),
            checkpoint_dir=self.checkpoint_dir,
            upload_dir=self.upload_dir,
            ollama={"url": self.ollama_url, "model": self.ollama_model},
            chatgpt={"url": self.chatgpt_url, "model": self.chatgpt_model, "configured": chatgpt_key_configured()},
        )
        return stats

    def predict(self, prefix: str, **options: Any) -> dict:
        """The active model's prediction; the count model adds ``top`` / ``bottom`` (K continuations each)."""
        with self.session() as model:
            result = model.predict(prefix, **options)
        payload = {
            "prefix": prefix,
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
        }
        if isinstance(result, Prediction):
            payload.update(
                mode=result.mode, k=result.k, beam=result.beam,
                top=[_path_dict(r) for r in result.top], bottom=[_path_dict(r) for r in result.bottom],
            )
        return payload

    def generate(self, **options: Any) -> dict:
        """Whole texts from the prediction search (``beam``: the K most likely), sampling, or the cheapest path."""
        with self.session() as model:
            results = model.generate(**options)
        return {"samples": [_sample_dict(r) for r in results]}

    def converse(self, opening: str = "", turns: int = 6, partner: str | None = None, **options: Any) -> dict:
        """The active model converses with itself, or with the model of another kind kept in memory (``partner``)."""
        with self.session() as model:
            other: GraphModel | None = None
            if partner and partner.strip().lower() != model.kind:
                try:
                    kind = model_class(partner).kind
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                other = self._parked.get(kind)
                if other is None:
                    raise ApiError(
                        400, f"no {kind} model in memory to converse with; select that kind once to load it"
                    )
            try:
                spoken = model.converse(opening, turns, partner=other, **options)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
        speakers = list(options.get("speakers") or DEFAULT_SPEAKERS)
        return {
            "kind": model.kind,
            "partner": other.kind if other is not None else None,
            "speakers": speakers,
            "turns": [t.to_dict() for t in spoken],
            "count": len(spoken),
        }

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

    # -- the negative network ------------------------------------------------

    def negative_model(self) -> NegativeNet:
        """The one negative network of this service (the active model when that kind is selected, else the parked one).

        On first use it is loaded from its file (``model.negative.json``
        beside the model path) or created empty; it lives in the same
        ``_parked`` store as the other kinds, so selecting ``negative`` as the
        active model hands back this very object.
        """
        if isinstance(self.model, NegativeNet):
            return self.model
        model = self._parked.get(NegativeNet.kind)
        if model is None:
            path = self.model_path_for(NegativeNet.kind)
            if path and os.path.isfile(path):
                model = load_model(path, backend=self.backend_name, device=self.device)
                if not isinstance(model, NegativeNet):
                    raise ApiError(400, f"{path} holds a {model.kind} model, not a negative one")
                self._log(f"negative model loaded from {path}")
            else:
                model = NegativeNet(seed=self.seed, backend=self.backend_name, device=self.device)
            self._parked[NegativeNet.kind] = model
        return model

    def positive_model(self) -> GraphModel:
        """The model the negative network filters: the active one, or a parked / saved positive model."""
        if not isinstance(self.model, NegativeNet):
            return self.model
        for kind in (self._path_kind, RadixNet.kind, "count"):
            parked = self._parked.get(kind)
            if parked is not None and not isinstance(parked, NegativeNet):
                return parked
            path = self.model_path_for(kind)
            if path and os.path.isfile(path):
                model = load_model(path, backend=self.backend_name, device=self.device)
                if not isinstance(model, NegativeNet):
                    self._parked[model.kind] = model
                    return model
        raise ApiError(
            409, "the negative network is the active model and no positive model is in memory; "
                 "select the radix or count kind first (POST /api/model/select)"
        )

    def negative_status(self) -> dict:
        """Stats, the reason table and the journal of the negative network."""
        with self.session():
            model = self.negative_model()
            return {
                "path": self.model_path_for(NegativeNet.kind),
                "active": isinstance(self.model, NegativeNet),
                "stats": model.stats(),
                "reasons": model.reasons(),
                "journal": model.recent(20),
                "weights": model.weight_config(),
                "settings": {"threshold": model.threshold, "min_coverage": model.min_coverage},
            }

    def _negative_result(self, model: NegativeNet, records: list[dict], **extra: Any) -> dict:
        return {"records": records, "reasons": model.reasons(), "stats": model.stats(), **extra}

    def negative_blame(
        self, texts: list[str], reason: str, severity: float = 1.0, source: str = "api", note: str = "",
        epochs: int = 1,
    ) -> dict:
        """Teach the negative network a failure (synchronous: rated sets are small)."""
        if not texts:
            raise ApiError(400, "give the failed texts as 'texts' (a list) or 'text' (one per line)")
        with self.mutating():
            model = self.negative_model()
            try:
                records = model.blame(texts, reason=reason, severity=severity, source=source, note=note, epochs=epochs)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            return self._negative_result(model, records, texts=len(texts), reason=reason, severity=severity)

    def negative_clear(self, texts: list[str], weight: float = 1.0, epochs: int = 1) -> dict:
        """The tutor passed these texts: take blame off the fragments they share with known failures."""
        if not texts:
            raise ApiError(400, "give the passed texts as 'texts' (a list) or 'text' (one per line)")
        with self.mutating():
            model = self.negative_model()
            try:
                records = model.clear(texts, weight=weight, epochs=epochs)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            last = records[-1] if records else {}
            return self._negative_result(
                model, records, texts=len(texts), matched=last.get("matched", 0), unmatched=last.get("unmatched", 0)
            )

    def negative_judge(
        self, texts: list[str], threshold: float | None = None, min_coverage: float | None = None, spans: int = 5,
    ) -> dict:
        """Why these texts look like failures."""
        if not texts:
            raise ApiError(400, "give the texts to judge as 'texts' (a list) or 'text' (one per line)")
        with self.session():
            model = self.negative_model()
            try:
                verdicts = [model.judge(t, threshold=threshold, min_coverage=min_coverage, spans=spans) for t in texts]
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            return {"verdicts": verdicts, "stats": model.stats()}

    def negative_filter(
        self, texts: list[str] | None = None, count: int = 3, no_ratio: bool = False, **options: Any
    ) -> dict:
        """The pair at work: the positive model writes, the negative one vetoes (or the given texts are judged)."""
        fields = {k: v for k, v in options.items() if k in _FILTER_FIELDS and v is not None}
        if no_ratio:
            fields["ratio"] = None  # a None in ``options`` means "unset"; only this turns the ratio rule off
        config = FilterConfig(**fields)
        try:
            config.validate()
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc
        generate = {k: v for k, v in options.items() if k not in _FILTER_FIELDS}
        with (self.mutating() if config.learn else self.session()):
            negative = self.negative_model()
            pair = NegativeFilter(self.positive_model(), negative, config)
            try:
                if texts:
                    outcome = pair.filter(texts)
                    result = {
                        "texts": outcome["kept"], "kept": outcome["kept"], "verdicts": outcome["verdicts"],
                        "rejected": [v for v in outcome["verdicts"] if v["decision"] == "reject"],
                        "candidates": len(texts), "asked": len(texts), "rate": outcome["rate"],
                    }
                else:
                    result = pair.generate(count=count, **generate)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            result["pair"] = pair.describe()
            return result

    def negative_forget(self, reason: str | None = None, factor: float = 0.0) -> dict:
        """Drop (or fade) the blame behind one reason - the tutor can be wrong too."""
        with self.mutating():
            model = self.negative_model()
            try:
                result = model.forget(reason, factor=factor)
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            return self._negative_result(model, [], **result)

    def negative_settings(self, **options: Any) -> dict:
        """Change how strictly the negative network judges (``threshold``, ``min_coverage``) and its weight scales."""
        with self.mutating():
            model = self.negative_model()
            for name in ("threshold", "min_coverage"):
                value = options.get(name)
                if value is not None:
                    setattr(model, name, float(value))
            scales = {k: options.get(k) for k in ("share_scale", "blame_scale", "clear_scale")}
            if any(v is not None for v in scales.values()):
                try:
                    model.configure_weights(**{k: v for k, v in scales.items() if v is not None})
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
            return {
                "settings": {"threshold": model.threshold, "min_coverage": model.min_coverage},
                "weights": model.weight_config(),
                "stats": model.stats(),
            }

    def negative_reset(self, seed: int | None = None) -> dict:
        """Forget every failure: a fresh, empty negative network."""
        with self.mutating():
            model = NegativeNet(
                seed=self.seed if seed is None else int(seed), backend=self.backend_name, device=self.device
            )
            if isinstance(self.model, NegativeNet):
                self.model = model
            else:
                self._parked[NegativeNet.kind] = model
            return {"stats": model.stats(), "reasons": [], "journal": []}

    def negative_save(self, path: str | None = None) -> dict:
        """Write the negative network (default: its file beside the model path)."""
        target = path or self.model_path_for(NegativeNet.kind)
        if not target:
            raise ApiError(400, "no 'path' given and the server was started without a model path")
        target = os.path.abspath(target)
        with self.session():
            self.negative_model().save(target)
        return {"path": target, "bytes": os.path.getsize(target)}

    def negative_teach(self, reviews: list[dict], threshold: float = 6.0, source: str = "review") -> dict:
        """Hand a tutor's review (ratings, verdicts, critiques) to the negative network."""
        with self.mutating():
            model = self.negative_model()
            report = teach_reviews(model, reviews, threshold=threshold, source=source)
            return {
                "blamed": report["blamed"], "cleared": report["cleared"], "unmatched": report["unmatched"],
                "edges": report["edges"], "reasons": report["reasons"], "lessons": report["lessons"],
                "severity_mean": report["severity_mean"], "stats": model.stats(),
                "reason_table": model.reasons(),
            }

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | None = None) -> dict:
        """Write the model (default: ``model_path``); allowed while a job runs
        because it snapshots the model at an epoch boundary."""
        target = path or self.model_path_for(self.kind)
        if not target:
            raise ApiError(400, "no 'path' given and the server was started without a model path")
        target = os.path.abspath(target)
        negative: dict | None = None
        with self.session() as model:
            model.save(target)
            companion = self._parked.get(NegativeNet.kind)
            if path is None and companion is not None and companion.graph.total_blame:
                # the negative network is a second file beside the model; saving the work means saving both
                side = self.model_path_for(NegativeNet.kind)
                if side:
                    companion.save(side)
                    negative = {"path": side, "bytes": os.path.getsize(side)}
        return {"path": target, "bytes": os.path.getsize(target), "negative": negative}

    def load(self, path: str) -> dict:
        """Replace the model with the one at ``path`` (parsed outside the lock)."""
        self._ensure_idle()
        model = load_model(path, backend=self.backend_name, device=self.device)
        return self._replace_model(model)

    def reset(self, seed: int | None = None, kind: str | None = None, **options: Any) -> dict:
        """Replace the model with a fresh one (``seed`` defaults to the server seed; ``kind`` to the active kind).

        ``options`` (count model only): ``count_scale``, ``global_scale``,
        ``window_scale``, ``reward_scale``, ``window``.
        """
        self._ensure_idle()
        cls = model_class(kind or self.kind)
        extra = {k: v for k, v in options.items() if v is not None}
        if extra and cls.kind != "count":
            raise ApiError(400, f"weight options ({', '.join(sorted(extra))}) apply to the count model only")
        try:
            model = cls(seed=self.seed if seed is None else seed, backend=self.backend_name, device=self.device, **extra)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc
        return self._replace_model(model)

    def configure_weights(self, **options: Any) -> dict:
        """Change the count model's dual frequency function (400 for RadixNet); returns the config and stats."""
        with self.mutating() as model:
            if model.kind != "count":
                raise ApiError(400, "the weight function can be configured on the count model only (select it first)")
            try:
                config = model.configure_weights(**{k: v for k, v in options.items() if v is not None})
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            return {"weights": config, "stats": model.stats()}

    def _replace_model(self, model: GraphModel) -> dict:
        """Install ``model``; a model of another kind that was active is kept in memory (see :meth:`select_kind`)."""
        with self.mutating():
            if model.kind != self.model.kind:
                self._parked[self.model.kind] = self.model
            self._parked.pop(model.kind, None)
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
        """Every uploaded file with its size and non-blank line count (an archive counts the lines of its text entries)."""
        root = self._require_uploads()
        with self._upload_lock:
            names = sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n)) and not n.endswith(".part"))
            records = [self._record(root, name) for name in names]
        return {"uploads": records, "upload_dir": root}

    def _archive_info(self, path: str) -> dict:
        """What a ZIP upload holds - text entries with their line counts, skipped entries - cached per file version."""
        st = os.stat(path)
        key = (st.st_size, st.st_mtime_ns)
        cached = self._archive_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            extracted, skipped = extract_texts(data, os.path.basename(path))
            info = _archive_summary(extracted, skipped)
        except ValueError as exc:
            info = _archive_summary([], [], error=str(exc))
        self._archive_cache[path] = (key, info)
        return info

    def _record(self, root: str, name: str) -> dict:
        """The listing record of one upload; a ZIP archive reports its text entries instead of its own bytes as text."""
        path = os.path.join(root, name)
        if not _is_archive_file(path):
            return _upload_record(root, name)
        info = self._archive_info(path)
        st = os.stat(path)
        record = {
            "name": name,
            "bytes": st.st_size,
            "chars": info["chars"],
            "lines": info["lines"],
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="milliseconds"),
            "archive": True,
            "files": info["files"],
            "skipped": len(info["skipped"]),
        }
        if info["error"]:
            record["error"] = info["error"]
        return record

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

    def upload_bytes(self, name: str, data: bytes) -> dict:
        """A binary upload: a ZIP archive is kept as one upload, anything else is stored as UTF-8 text.

        Returns ``{"uploads": [records], "archives": [summary]}`` (``archives``
        only for an archive).
        """
        if is_zip(name, data):
            return self.store_archive(name, data)
        return {"uploads": [self.upload(name, _decode_text(data))]}

    def store_archive(self, name: str, data: bytes) -> dict:
        """Keep a ZIP archive as a single upload; its text entries are unpacked in memory whenever it is used.

        The archive is validated first: one over the entry or size limits, a
        corrupt one, or one without a single text entry is refused (400).
        Directories, metadata, nested archives, encrypted, binary and empty
        entries are ignored and listed with a reason in the summary.
        """
        root = self._require_uploads()
        archive_name = sanitize_upload_name(name)
        if not archive_name.lower().endswith(".zip"):
            archive_name += ".zip"
        try:
            extracted, skipped = extract_texts(data, archive_name)
        except ValueError as exc:
            raise ApiError(400, f"{archive_name}: {exc}") from exc
        if not extracted:
            reasons = "; ".join(f"{item.path}: {item.reason}" for item in skipped[:8])
            raise ApiError(400, f"{archive_name} holds no text files to train on" + (f" ({reasons})" if reasons else " (it is empty)"))
        path = os.path.join(root, archive_name)
        with self._upload_lock:
            replaced = os.path.isfile(path)
            part = path + ".part"
            with open(part, "wb") as fh:
                fh.write(data)
            os.replace(part, path)
            st = os.stat(path)
            self._archive_cache[path] = ((st.st_size, st.st_mtime_ns), _archive_summary(extracted, skipped))
            record = self._record(root, archive_name)
        record["replaced"] = replaced
        summary = {
            "name": archive_name,
            "bytes": len(data),
            "entries": len(extracted) + len(skipped),
            "extracted": len(extracted),
            "skipped": [item.to_dict() for item in skipped],
        }
        self._log(f"upload {archive_name} ({len(data)} bytes; {len(extracted)} text file(s) inside, {len(skipped)} skipped)")
        return {"uploads": [record], "archives": [summary]}

    def delete_upload(self, name: str) -> dict:
        path = self._upload_path(name)
        with self._upload_lock:
            if not os.path.isfile(path):
                raise ApiError(404, f"no upload named {os.path.basename(path)!r}")
            os.remove(path)
            self._archive_cache.pop(path, None)
        return {"deleted": os.path.basename(path)}

    def upload_entries(self, name: str) -> list[tuple[str, str]]:
        """``[(file name, text)]`` of an upload: one pair for a text file, one per text entry of a ZIP archive.

        Archives are unpacked in memory here, behind the scenes: the upload
        directory only ever holds the ``.zip`` itself.
        """
        path = self._upload_path(name)
        with self._upload_lock:
            if not os.path.isfile(path):
                raise ApiError(404, f"no upload named {os.path.basename(path)!r}")
            with open(path, "rb") as fh:
                data = fh.read()
        base = os.path.basename(path)
        if is_zip(base, data):
            try:
                extracted, _skipped = extract_texts(data, base)
            except ValueError as exc:
                raise ApiError(400, f"{base}: {exc}") from exc
            return [(entry.path, entry.text) for entry in extracted]
        return [(base, _decode_text(data))]

    def upload_texts(self, names: list[str], whole_file: bool = False) -> list[str]:
        """Training texts read from uploads: one per non-blank line, or each file (archive entry) as one text."""
        texts: list[str] = []
        for name in names:
            for _entry, blob in self.upload_entries(name):
                if whole_file:
                    if blob.strip():
                        texts.append(blob)
                else:
                    texts.extend(line for line in blob.splitlines() if line.strip())
        return texts

    # -- LLM providers (local Ollama, hosted ChatGPT) -------------------------

    def ollama_client(self, url: str | None = None, model: str | None = None, timeout: float | None = None) -> OllamaClient:
        """A client for the request's Ollama overrides, falling back to the server defaults."""
        try:
            return OllamaClient(url or self.ollama_url, model or self.ollama_model, timeout)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc

    def chatgpt_client(self, url: str | None = None, model: str | None = None, timeout: float | None = None) -> ChatGPTClient:
        """A client for the request's ChatGPT overrides; the API key is the server's own (never a request field)."""
        try:
            return ChatGPTClient(url or self.chatgpt_url, model or self.chatgpt_model, timeout)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc

    def llm_client(
        self, provider: str | None = None, url: str | None = None, model: str | None = None, timeout: float | None = None
    ) -> LLMClient:
        """A client for ``provider`` (``"ollama"`` | ``"chatgpt"``); 400 when ChatGPT has no key on this server."""
        try:
            provider = normalise_provider(provider)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc
        if provider == "ollama":
            return self.ollama_client(url, model, timeout)
        if not chatgpt_key_configured():
            raise ApiError(
                400,
                "ChatGPT is not configured on this server: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) in its "
                "environment and restart it, or use the 'ollama' provider",
            )
        return self.chatgpt_client(url, model, timeout)

    def sample_texts(
        self, count: int, prefix: str = "", max_length: int = 60, temperature: float = 1.0, seed: int | None = None
    ) -> list[str]:
        """``count`` stochastic texts from the model (held under the lock only while sampling)."""
        with self.session() as model:
            return sample_texts(model, count, prefix=prefix, max_length=max_length, temperature=temperature, seed=seed)

    # -- code generation (sandbox + Ollama judge + 2NRL) ---------------------

    def read_upload(self, name: str) -> str:
        """Content of one uploaded file (404 when missing); the text entries of an archive joined by blank lines."""
        return "\n\n".join(text for _entry, text in self.upload_entries(name))

    @contextlib.contextmanager
    def pause_lock(self) -> Iterator[None]:
        """Release the model lock around slow external work inside a job (sandbox runs, LLM calls).

        Mutating requests are still refused while the job runs (409), so only
        readers get in; the lock is re-acquired before the job touches the
        model again.
        """
        self._lock.release()
        try:
            yield
        finally:
            self._lock.acquire()

    def start_codegen(
        self,
        problems: list[Problem],
        config: CodeGenConfig,
        client: LLMClient,
        sandbox: Sandbox,
        judge_client: LLMClient | None = None,
        blame: bool = False,
    ) -> dict:
        """Start a ``codegen`` job: teacher / model phases over ``problems`` with 2NRL rewards.

        With ``blame`` the sandbox, the style checker and the judge also teach
        the negative network why each rejected program was rejected.
        """
        config.validate()
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        negative = self.negative_model() if blame else None

        def work(job: Job) -> None:
            trainer = CodeGenTrainer(
                self.model, client, sandbox, config, external=self.pause_lock, judge_client=judge_client,
                negative=negative,
            )
            trainer.run(
                problems, progress=self._progress(job, self._codegen_history), stop_event=job.stop_event,
                checkpoint_manager=manager,
            )

        return self._start_job("codegen", work)

    def codegen_history(self) -> dict:
        return {"history": list(self._codegen_history)}

    # -- English lessons (Ollama sets and marks the exercises) ---------------

    def start_tutor(
        self, config: TutorConfig, client: LLMClient, grader_client: LLMClient | None = None, blame: bool = False,
    ) -> dict:
        """Start a ``tutor`` job: rounds of prefix -> completion -> grade -> 2NRL.

        With ``blame`` every failed sentence also teaches the negative network
        why it failed: the mistake the teacher named is the reason, its mark
        the severity, and only the characters the correction changed are
        blamed.
        """
        config.validate()
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        negative = self.negative_model() if blame else None

        def work(job: Job) -> None:
            trainer = TutorTrainer(
                self.model, client, config, external=self.pause_lock, grader_client=grader_client,
                negative=negative,
            )
            trainer.run(
                progress=self._progress(job, self._tutor_history), stop_event=job.stop_event,
                checkpoint_manager=manager,
            )

        return self._start_job("tutor", work)

    def tutor_history(self) -> dict:
        return {"history": list(self._tutor_history)}

    def tutor_card(self) -> dict | None:
        """The report card at the end of the last tutor run, or ``None`` when nothing has been marked yet."""
        for record in reversed(self._tutor_history):
            if record.get("kind") == "report":
                return dict(record)
        return None

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
        "full_text": result.full_text,
        "cost": result.cost,
        "probability": path_probability(result),
        "path": list(result.labels),
        "node_ids": list(result.node_ids),
        "step_costs": list(result.step_costs),
        "reached_end": result.reached_end,
    }


def _path_dict(result: PathResult) -> dict:
    """One of the top / bottom continuations of a count-model prediction."""
    return {
        "continuation": result.text,
        "full_text": result.full_text,
        "cost": result.cost,
        "probability": path_probability(result),
        "step_costs": list(result.step_costs),
        "path": list(result.labels),
        "node_ids": list(result.node_ids),
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
    edge_reward = getattr(graph, "edge_reward", None)
    shares_of = getattr(graph, "shares", None)
    edges = []
    for p in ids:
        shares = {e: (all_, recent) for _c, e, all_, recent in shares_of(p)} if shares_of is not None else {}
        for c, e, cost in graph.child_costs(p):
            if c in chosen:
                edge = {
                    "source": p, "target": c, "weight": edge_w[e], "count": edge_count[e],
                    "prob": math.exp(-cost), "cost": cost,
                }
                if edge_reward is not None:
                    edge["reward"] = edge_reward[e]
                    edge["share"], edge["recent_share"] = shares.get(e, (0.0, 0.0))
                    edge["recent_count"] = graph.window_edge_count[e]
                edges.append(edge)
    edges.sort(key=lambda d: (d["source"], d["target"]))
    view = {
        "nodes": nodes, "edges": edges, "limit": limit,
        "total_nodes": graph.num_nodes(), "total_edges": graph.num_edges(),
    }
    if edge_reward is not None:
        view["total_traversals"] = graph.total_traversals
        view["window_traversals"] = graph.window_traversals
        view["window"] = graph.window
    return view


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

    def mapping(self, name: str, default: Any = _MISSING) -> Any:
        """A JSON object field, such as a report card handed back to the tutor."""
        value = self._lookup(name)
        if value is _MISSING:
            return self._default(name, default, "object")
        if not isinstance(value, dict):
            raise self._bad(name, "an object", value)
        return value

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

    def weights(self, name: str, count: int, scale: float = 1.0) -> list[float] | None:
        """An optional list of ``count`` non-negative numbers (per-text ratings), divided by ``scale``."""
        value = self._lookup(name)
        if value is _MISSING or value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            raise ApiError(400, f"'{name}' must be a list of numbers")
        if len(value) != count:
            raise ApiError(400, f"'{name}' has {len(value)} entries for {count} text(s)")
        numbers = [float(v) / scale for v in value]
        for number in numbers:
            if not (number >= 0.0) or number != number:  # NaN and negatives
                raise ApiError(400, f"'{name}' must hold numbers >= 0 (got {number * scale})")
        return numbers

    def names(self, name: str) -> list[str]:
        """An optional list of non-empty strings (upload names); absent -> ``[]``."""
        value = self._lookup(name)
        if value is _MISSING:
            return []
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise self._bad(name, "a list of non-empty strings", value)
        return list(value)

    def _upload_payload(self) -> str | bytes:
        """The content of one upload: ``content`` (text), ``content_base64`` (bytes) or internal ``data`` (bytes)."""
        data = self._lookup("data")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        encoded = self._lookup("content_base64")
        if encoded is not _MISSING:
            if not isinstance(encoded, str):
                raise self._bad("content_base64", "a base64 string", encoded)
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(400, f"'content_base64' is not valid base64: {exc}") from exc
        if self._lookup("content") is _MISSING:
            raise ApiError(400, "missing field 'content' (string) or 'content_base64' (base64 of a text file or ZIP archive)")
        return self.text("content")

    def upload_files(self) -> list[tuple[str, str | bytes]]:
        """``{name, content | content_base64}`` or ``{files: [...]}`` -> ``[(name, text or bytes), ...]``.

        Bytes (``content_base64``, multipart parts, raw bodies) may hold a ZIP
        archive, which the service unpacks; text is stored as it is.
        """
        files = self._lookup("files")
        if files is _MISSING:
            return [(self.text("name"), self._upload_payload())]
        if not isinstance(files, list) or not files:
            raise self._bad("files", "a non-empty list of {name, content | content_base64} objects", files)
        result: list[tuple[str, str | bytes]] = []
        for i, item in enumerate(files):
            if not isinstance(item, dict):
                raise ApiError(400, f"'files[{i}]' must be an object with 'name' and 'content' (or 'content_base64')")
            entry = Fields(item)
            result.append((entry.text("name"), entry._upload_payload()))
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


def _is_archive_file(path: str) -> bool:
    """Whether an upload on disk is a ZIP archive (by its magic bytes)."""
    try:
        with open(path, "rb") as fh:
            return is_zip(path, fh.read(4))
    except OSError:
        return False


def _archive_summary(extracted: list, skipped: list, error: str | None = None) -> dict:
    """Counts of a ZIP upload's text entries and its skipped entries (what the listing shows)."""
    entries = [
        {"path": e.path, "bytes": e.bytes, "lines": sum(1 for line in e.text.splitlines() if line.strip())}
        for e in extracted
    ]
    return {
        "files": len(entries),
        "entries": entries,
        "skipped": [item.to_dict() for item in skipped],
        "lines": sum(e["lines"] for e in entries),
        "chars": sum(len(e.text) for e in extracted),
        "error": error,
    }


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
        files.append({"name": filename, "data": payload})
    if not files:
        raise ApiError(400, "multipart body contains no file parts (use -F file=@corpus.txt)")
    return files


def _upload_body(body: bytes, content_type: str, query: dict[str, list[str]], default_name: str | None = None) -> dict:
    """Body of ``POST /api/uploads`` (and the image encoder) in any accepted form, normalised to the JSON form.

    * ``application/json``: ``{name, content | content_base64}`` or ``{files: [...]}``
    * ``multipart/form-data``: every part with a file name (bytes: text files or ZIP archives)
    * anything else (``text/plain``, ``--data-binary @file.zip``): the raw body, named by ``?name=``
    """
    kind = content_type.split(";", 1)[0].strip().lower()
    names = [n for n in query.get("name", []) if n.strip()]
    if kind == "multipart/form-data":
        return {"files": _multipart_files(body, content_type)}
    if kind == "application/json" or (kind == "" and not names and default_name is None):
        return parse_body(body)
    if not names:
        if default_name is None:
            raise ApiError(400, "raw uploads need a ?name=<file name> query parameter (or send JSON {name, content})")
        names = [default_name]
    return {"files": [{"name": names[0], "data": body}]}


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


def _ratings(f: Fields, side: str, texts: list[str]) -> list[float] | None:
    """Per-text weights of one side: ``<side>_weights`` (0..1 shares) or ``<side>_ratings`` (marks out of 10).

    A rating says *how* good or bad each text is, so the network learns a
    9-out-of-10 text nine tenths as hard as a perfect one instead of treating
    every thumbs up alike.
    """
    weights = f.weights(f"{side}_weights", len(texts))
    ratings = f.weights(f"{side}_ratings", len(texts), scale=10.0)
    if weights is not None and ratings is not None:
        raise ApiError(400, f"give '{side}_weights' or '{side}_ratings', not both")
    return weights if weights is not None else ratings


def feedback_action(good: list[str], bad: list[str]) -> str | None:
    """What rated texts lead to: ``"2nrl"`` (both), ``"reward"`` (good only), ``"punish"`` (bad only), ``None``."""
    if good and bad:
        return "2nrl"
    if good:
        return "reward"
    if bad:
        return "punish"
    return None


def _r_feedback(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    whole_file = f.flag("whole_file", False)
    good = f.texts_optional("good", "good_text")
    bad = f.texts_optional("bad", "bad_text")
    for names, target in ((f.names("good_files"), good), (f.names("bad_files"), bad)):
        if names:
            target.extend(svc.upload_texts(names, whole_file=whole_file))
    action = feedback_action(good, bad)
    if action is None:
        raise ApiError(
            400, "give 'good' (thumbs up) and/or 'bad' (thumbs down) texts: lists, good_text / bad_text (one per line) "
                 "or good_files / bad_files (upload names)"
        )
    overrides = _train_overrides(f)
    overrides.setdefault("batch_size", 4)  # rated sets are small
    good_weights = _ratings(f, "good", good)
    bad_weights = _ratings(f, "bad", bad)
    job = svc.start_feedback(
        good, bad,
        neg_epochs=f.integer("neg_epochs", 2, minimum=0),
        pos_epochs=f.integer("pos_epochs", 3, minimum=0),
        neg_lr=f.number("neg_lr", 0.5, minimum=0.0),
        pos_lr=f.number("pos_lr", 0.1, minimum=0.0),
        strength=f.number("strength", None, minimum=0.0),
        good_weights=good_weights,
        bad_weights=bad_weights,
        **overrides,
    )
    return 202, {
        "job": job, "action": action, "good": len(good), "bad": len(bad),
        "good_weights": good_weights, "bad_weights": bad_weights,
    }


def _train_config(f: Fields) -> TrainConfig:
    """The optional training settings of a request body, defaults from :class:`TrainConfig`."""
    return TrainConfig(
        epochs=f.integer("epochs", TrainConfig.epochs, minimum=0),
        lr=f.number("lr", TrainConfig.lr, minimum=0.0),
        act_lr=f.number("act_lr", TrainConfig.act_lr, minimum=0.0),
        lr_schedule=f.text("lr_schedule", "").strip() or None,
        act_lr_schedule=f.text("act_lr_schedule", "").strip() or None,
        reverse_schedule=f.flag("reverse_schedule", False),
        batch_size=f.integer("batch_size", TrainConfig.batch_size, minimum=1),
        clip=f.number("clip", TrainConfig.clip),
        auto_compress=f.flag("auto_compress", TrainConfig.auto_compress),
        shuffle=f.flag("shuffle", TrainConfig.shuffle),
        checkpoint_every=f.integer("checkpoint_every", TrainConfig.checkpoint_every, minimum=0),
    )


def _r_train(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    texts = _texts_and_files(svc, f, "texts", "text", "files", whole_file=f.flag("whole_file", False))
    return 202, {"job": svc.start_train(texts, _train_config(f))}


def _r_schedule(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, describe_schedules()


def _r_schedule_preview(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """The per-epoch rates a pair of schedule expressions gives (the graph the frontend draws)."""
    lr_schedule = f.text("lr_schedule", "").strip() or None
    act_lr_schedule = f.text("act_lr_schedule", "").strip() or None
    epochs = f.integer("epochs", TrainConfig.epochs, minimum=0)
    lr = f.number("lr", TrainConfig.lr, minimum=0.0)
    act_lr = f.number("act_lr", TrainConfig.act_lr, minimum=0.0)
    reverse = f.flag("reverse_schedule", False)
    try:
        points = preview_points(lr_schedule, act_lr_schedule, epochs, lr, act_lr, reverse=reverse)
    except ScheduleError as exc:
        raise ApiError(400, str(exc)) from exc
    return 200, {
        "lr_schedule": lr_schedule, "act_lr_schedule": act_lr_schedule, "epochs": epochs, "lr": lr, "act_lr": act_lr,
        "reverse_schedule": reverse, "points": points,
    }


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
        k=f.integer("k", 5, minimum=0),
        beam=f.integer("beam", None, minimum=1),
    )


def _r_generate(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.generate(
        count=f.integer("count", 1, minimum=0),
        max_length=f.integer("max_length", 60, minimum=0),
        mode=f.text("mode", "sample"),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        seed=f.integer("seed", None),
        prefix=f.text("prefix", ""),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        beam=f.integer("beam", None, minimum=1),
    )


def _r_converse(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    partner = f.text("partner", None)
    return 200, svc.converse(
        opening=f.text("opening", ""),
        turns=f.integer("turns", 6, minimum=0),
        partner=partner or None,
        mode=f.text("mode", "beam"),
        max_length=f.integer("max_length", 60, minimum=0),
        context=f.integer("context", 12, minimum=0),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        k=f.integer("k", 5, minimum=1),
        beam=f.integer("beam", None, minimum=1),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        seed=f.integer("seed", None),
        speakers=f.names("speakers") or list(DEFAULT_SPEAKERS),
        history=f.texts_optional("history", "history_text"),
        avoid_repeats=f.flag("avoid_repeats", True),
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
        strength=f.number("strength", None, minimum=0.0),
        bad_weights=_ratings(f, "bad", bad),
        good_weights=_ratings(f, "good", good),
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
        blatant_mode=f.text("blatant_mode", d.blatant_mode),
        blatant_margin=f.number("blatant_margin", d.blatant_margin, minimum=0.0),
        blatant_boost=f.number("blatant_boost", d.blatant_boost, minimum=1.0),
    )
    return 202, {"job": svc.start_evolve(corpus, generations, config, blame=f.flag("blame", False))}


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


def _weight_options(f: Fields) -> dict:
    return {
        "count_scale": f.number("count_scale", None),
        "global_scale": f.number("global_scale", None),
        "window_scale": f.number("window_scale", None),
        "reward_scale": f.number("reward_scale", None),
        "window": f.integer("window", None, minimum=1),
    }


def _r_reset(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.reset(f.integer("seed", None), kind=f.text("kind", None) or None, **_weight_options(f))


def _r_model_weights(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.configure_weights(**_weight_options(f))


def _r_model(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.describe_model()


def _r_model_select(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    kind = f.text("kind")
    if not kind:
        raise ApiError(400, "'kind' must not be empty")
    try:
        return 200, svc.select_kind(kind)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc


def _r_negative(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_status()


def _negative_texts(f: Fields, what: str) -> list[str]:
    texts = f.texts_optional("texts", "text")
    if not texts:
        raise ApiError(400, f"give the {what} as 'texts' (a list) or 'text' (one per line)")
    return texts


def _r_negative_blame(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_blame(
        _negative_texts(f, "failed texts"),
        reason=f.text("reason", "unspecified"),
        severity=f.number("severity", 1.0, minimum=0.0),
        source=f.text("source", "api"),
        note=f.text("note", ""),
        epochs=f.integer("epochs", 1, minimum=1),
    )


def _r_negative_clear(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_clear(
        _negative_texts(f, "passed texts"),
        weight=f.number("weight", 1.0, minimum=0.0),
        epochs=f.integer("epochs", 1, minimum=1),
    )


def _r_negative_judge(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_judge(
        _negative_texts(f, "texts to judge"),
        threshold=f.number("threshold", None, minimum=0.0),
        min_coverage=f.number("min_coverage", None, minimum=0.0),
        spans=f.integer("spans", 5, minimum=0),
    )


def _r_negative_filter(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_filter(
        texts=f.texts_optional("texts", "text"),
        count=f.integer("count", 3, minimum=0),
        threshold=f.number("threshold", None, minimum=0.0),
        min_coverage=f.number("min_coverage", None, minimum=0.0),
        ratio=f.number("ratio", 0.0),
        no_ratio=f.flag("no_ratio", False),
        peak=f.number("peak", None, minimum=0.0),
        over_sample=f.integer("over_sample", 3, minimum=1),
        strict=f.flag("strict", False),
        spans=f.integer("spans", 3, minimum=0),
        learn=f.flag("learn", False),
        reason=f.text("reason", "filtered"),
        mode=f.text("mode", "sample"),
        max_length=f.integer("max_length", 60, minimum=0),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        prefix=f.text("prefix", ""),
        seed=f.integer("seed", None),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        beam=f.integer("beam", None, minimum=1),
    )


def _r_negative_forget(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_forget(f.text("reason", None), factor=f.number("factor", 0.0, minimum=0.0))


def _r_negative_settings(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_settings(
        threshold=f.number("threshold", None, minimum=0.0),
        min_coverage=f.number("min_coverage", None, minimum=0.0),
        share_scale=f.number("share_scale", None),
        blame_scale=f.number("blame_scale", None),
        clear_scale=f.number("clear_scale", None),
    )


def _r_negative_reset(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_reset(f.integer("seed", None))


def _r_negative_save(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_save(f.text("path", None))


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


def _option(f: Fields, q: dict, name: str, default: Any) -> Any:
    """A request option from the JSON body when there is one, else from the query string (multipart / raw bodies)."""
    if f.present(name):
        return f._lookup(name)
    values = q.get(name)
    return values[0] if values else default


def _flag_option(f: Fields, q: dict, name: str, default: bool = False) -> bool:
    """A boolean option from the JSON body or the query string (``?train=true``)."""
    raw = _option(f, q, name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _audio_bytes(f: Fields, what: str) -> tuple[str, bytes]:
    """The one audio file of a request: multipart, a raw body or JSON ``content_base64``."""
    files = f.upload_files()
    name, payload = files[0]
    if not isinstance(payload, bytes):
        raise ApiError(400, f"send the {what} as bytes: multipart/form-data, a raw body, or JSON {{name, content_base64}}")
    return name, payload


def _r_images(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, describe_vision()


def _r_image_encode(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Image bytes (multipart, raw or JSON content_base64) -> the encoded text; optionally saved as an upload and trained on."""
    files = f.upload_files()
    name, payload = files[0]
    if not isinstance(payload, bytes):
        raise ApiError(400, "send the image as bytes: multipart/form-data, a raw body, or JSON {name, content_base64}")
    try:
        size = int(_option(f, q, "size", 0) or 0) or None
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "'size' must be an integer") from exc
    encoder = str(_option(f, q, "encoder", "auto") or "auto")
    train = _flag_option(f, q, "train")
    save_as = _option(f, q, "save_as", None)
    if train:
        svc._ensure_idle()
    try:
        result = encode_image(payload, size=size or 128, encoder=encoder)
    except VisionError as exc:
        raise ApiError(400, str(exc)) from exc
    result.update(name=name, upload=None, job=None)
    if save_as:
        result["upload"] = svc.upload(str(save_as), result["text"] + "\n")
    if train:
        config = TrainConfig(
            epochs=int(_option(f, q, "epochs", 3)), lr=float(_option(f, q, "lr", 0.5)),
            act_lr=float(_option(f, q, "act_lr", TrainConfig.act_lr)), batch_size=int(_option(f, q, "batch_size", 8)),
        )
        result["job"] = svc.start_train([result["text"]], config)
        return 202, result
    return 200, result


def _r_image_decode(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    text = f.text("text")
    encoder = f.text("encoder", None) or None
    try:
        result = decode_image_text(text, encoder=encoder)
    except VisionError as exc:
        raise ApiError(400, str(exc)) from exc
    png = result.pop("png")
    result["png_base64"] = base64.b64encode(png).decode("ascii")
    return 200, result


def _r_speech(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, describe_speech()


def _r_speech_transcribe(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Speech to text: audio bytes -> the words, with the backend that did it."""
    name, payload = _audio_bytes(f, "audio")
    try:
        result = transcribe_speech(
            payload,
            backend=str(_option(f, q, "backend", "auto") or "auto"),
            text=str(_option(f, q, "transcript", "") or ""),
            language=_option(f, q, "language", None) or None,
            model=_option(f, q, "asr_model", None) or None,
            url=_option(f, q, "asr_url", None) or None,
        )
    except SpeechError as exc:
        raise ApiError(400, str(exc)) from exc
    result["name"] = name
    return 200, result


def _r_speech_teach(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """One utterance -> the transcript and the waveform behind one unique token; optionally trained on."""
    name, payload = _audio_bytes(f, "recording")
    train = _flag_option(f, q, "train")
    save_as = _option(f, q, "save_as", None)
    try:
        rate = int(_option(f, q, "rate", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "'rate' must be an integer") from exc
    if train:
        svc._ensure_idle()
    try:
        result = teach_speech(
            payload,
            transcript=str(_option(f, q, "transcript", "") or ""),
            backend=str(_option(f, q, "backend", "auto") or "auto"),
            language=_option(f, q, "language", None) or None,
            asr_model=_option(f, q, "asr_model", None) or None,
            asr_url=_option(f, q, "asr_url", None) or None,
            rate=rate or SPEECH_DEFAULT_RATE,
            codec=str(_option(f, q, "codec", "auto") or "auto"),
            normalise=_flag_option(f, q, "normalise"),
            waveform=_flag_option(f, q, "waveform", True),
            pair=_flag_option(f, q, "pair"),
            token=_option(f, q, "token", None) or None,
            unique=_flag_option(f, q, "unique", True),
        )
    except SpeechError as exc:
        raise ApiError(400, str(exc)) from exc
    if train and not result["texts"]:
        raise ApiError(400, "nothing to train on: the audio produced neither a transcript nor a waveform")
    result.update(name=name, upload=None, job=None)
    if save_as:
        result["upload"] = svc.upload(str(save_as), "\n".join(result["texts"]) + "\n")
    if train:
        config = TrainConfig(
            epochs=int(_option(f, q, "epochs", 3)), lr=float(_option(f, q, "lr", 0.5)),
            act_lr=float(_option(f, q, "act_lr", TrainConfig.act_lr)), batch_size=int(_option(f, q, "batch_size", 8)),
        )
        result["job"] = svc.start_train(result["texts"], config)
        return 202, result
    return 200, result


def _r_speech_decode(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """An encoded - or predicted - waveform text back to a WAV file that can be played."""
    text = f.text("text")
    codec = f.text("codec", None) or None
    try:
        result = decode_speech_text(text, codec=codec)
    except SpeechError as exc:
        raise ApiError(400, str(exc)) from exc
    wav = result.pop("wav")
    result["wav_base64"] = base64.b64encode(wav).decode("ascii")
    return 200, result


def _r_upload(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    uploads: list[dict] = []
    archives: list[dict] = []
    for name, payload in f.upload_files():
        if isinstance(payload, bytes):
            result = svc.upload_bytes(name, payload)
            uploads.extend(result["uploads"])
            archives.extend(result.get("archives", []))
        else:
            uploads.append(svc.upload(name, payload))
    if not uploads:
        raise ApiError(400, "nothing was uploaded")
    body: dict[str, Any] = {"uploads": uploads}
    if archives:
        body["archives"] = archives
    return 201, body


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


def _r_chatgpt_models(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Is ChatGPT usable as a tutor on this server, and which models does its key have?"""
    url = (q.get("url") or [None])[0]
    client = svc.chatgpt_client(url or None)
    configured = chatgpt_key_configured()
    error = None if configured else "no API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) for the server process"
    models: list[dict] = []
    if configured:
        try:
            models = client.models()
        except LLMError as exc:
            error = str(exc)
    return 200, {
        "available": error is None,
        "configured": configured,
        "url": client.url,
        "model": client.model,
        "models": [{"name": m.get("name"), "owned_by": m.get("owned_by"), "created": m.get("created")} for m in models],
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
    result["negative"] = None
    if f.flag("blame", False):
        result["negative"] = svc.negative_teach(result["reviews"], threshold=threshold, source="review")
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


def _problems_from(svc: ModelService, f: Fields) -> list[Problem]:
    """``problems`` (strings / objects), ``problems_text`` (one per line) and ``problem_files`` (uploads)."""
    items: list[Any] = []
    raw = f._body.get("problems")
    if raw is not None:
        if not isinstance(raw, list):
            raise ApiError(400, "'problems' must be a list of strings or {prompt, tests, expected_output} objects")
        items.extend(raw)
    text = f.text("problems_text", None)
    if text:
        items.extend(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    for name in f.names("problem_files"):
        for entry, content in svc.upload_entries(name):  # a ZIP upload contributes every text entry it holds
            try:
                items.extend(p.to_dict() for p in parse_problem_file(content, os.path.splitext(entry)[1]))
            except ValueError as exc:
                where = f"upload {name!r}" if entry == name else f"upload {name!r} ({entry})"
                raise ApiError(400, f"{where}: {exc}") from exc
    if not items:
        raise ApiError(
            400, "missing 'problems' (list of strings / objects), 'problems_text' (one per line) or 'problem_files' (upload names)"
        )
    try:
        return parse_problems(items)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc


def _default_llm_model(svc: ModelService, provider: str) -> str:
    """The model this server tutors / judges with when a request names none.

    ChatGPT follows the server's own default (``--chatgpt-model`` /
    ``$RADIXNET_OPENAI_MODEL``); Ollama keeps codegen's default
    (``$RADIXNET_CODEGEN_MODEL``), which is deliberately not the model of the
    ``/api/ollama/*`` endpoints.
    """
    return svc.chatgpt_model if provider == "chatgpt" else default_teacher_model("ollama")


def _provider_field(f: Fields, name: str, alias: str | None, default: str) -> str:
    """``"ollama"`` | ``"chatgpt"`` from a request field (``"openai"`` and ``"gpt"`` are accepted too)."""
    raw = f.text(name, None) or (f.text(alias, None) if alias else None)
    try:
        return normalise_provider(raw) if raw else default
    except ValueError as exc:
        raise ApiError(400, f"'{name}' must be one of {', '.join(PROVIDERS)} (got {raw!r})") from exc


def _codegen_config(svc: ModelService, f: Fields) -> CodeGenConfig:
    d = CodeGenConfig()
    raw_phases = f._body.get("phases", f._body.get("phase"))
    if raw_phases is None:
        phases: tuple[str, ...] = d.phases
    elif isinstance(raw_phases, str):
        phases = CODEGEN_PHASES if raw_phases.strip().lower() == "both" else (raw_phases.strip().lower(),)
    elif isinstance(raw_phases, list) and all(isinstance(p, str) for p in raw_phases):
        phases = tuple(p.strip().lower() for p in raw_phases)
    else:
        raise ApiError(400, "'phases' must be 'both', 'teacher', 'model' or a list of those")
    teacher_provider = _provider_field(f, "teacher_provider", "provider", d.teacher_provider)
    judge_provider = _provider_field(f, "judge_provider", None, teacher_provider)
    teacher_model = f.text("teacher_model", None) or f.text("model", None) or _default_llm_model(svc, teacher_provider)
    judge_model = f.text("judge_model", None)
    if judge_model is None and judge_provider != teacher_provider:
        judge_model = _default_llm_model(svc, judge_provider)
    config = CodeGenConfig(
        teacher_provider=teacher_provider,
        teacher_model=teacher_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        phases=phases,
        rounds=f.integer("rounds", d.rounds, minimum=1),
        teacher_attempts=f.integer("teacher_attempts", d.teacher_attempts, minimum=1),
        model_attempts=f.integer("model_attempts", d.model_attempts, minimum=1),
        first_attempt_dijkstra=f.flag("first_attempt_dijkstra", d.first_attempt_dijkstra),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        max_length=f.integer("max_length", d.max_length, minimum=1),
        strictness=f.text("strictness", d.strictness).strip().lower(),
        use_judge=f.flag("judge", d.use_judge),
        fallback_teacher=f.flag("fallback_teacher", d.fallback_teacher),
        twonrl_per=f.text("twonrl_per", d.twonrl_per).strip().lower(),
        replay=f.flag("replay", d.replay),
        replay_limit=f.integer("replay_limit", d.replay_limit, minimum=0),
        teacher_prompt=f.text("teacher_prompt", None),
        model_prompt=f.text("model_prompt", d.model_prompt),
        neg_epochs=f.integer("neg_epochs", d.neg_epochs, minimum=0),
        pos_epochs=f.integer("pos_epochs", d.pos_epochs, minimum=0),
        neg_lr=f.number("neg_lr", d.neg_lr, minimum=0.0),
        pos_lr=f.number("pos_lr", d.pos_lr, minimum=0.0),
        batch_size=f.integer("batch_size", d.batch_size, minimum=1),
        checkpoint_every=f.integer("checkpoint_every", d.checkpoint_every, minimum=0),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


def _codegen_clients(svc: ModelService, f: Fields, config: CodeGenConfig) -> tuple[LLMClient, LLMClient]:
    """The tutor and judge clients of a codegen request (``url`` / ``judge_url`` override the server defaults)."""
    timeout = f.number("timeout", None, minimum=1.0)
    client = svc.llm_client(config.teacher_provider, f.text("url", None), config.teacher_model, timeout)
    if config.judge_provider == config.teacher_provider and not f.text("judge_url", None):
        return client, client
    judge = svc.llm_client(config.judge_provider, f.text("judge_url", None), config.resolved_judge_model, timeout)
    return client, judge


def _sandbox_from(f: Fields) -> Sandbox:
    return Sandbox(
        timeout=f.number("sandbox_timeout", 10.0, minimum=0.1),
        memory_mb=f.integer("memory_mb", 256, minimum=0),
        isolate_network=f.flag("network_isolation", True),
    )


def _r_codegen_start(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    problems = _problems_from(svc, f)
    config = _codegen_config(svc, f)
    client, judge = _codegen_clients(svc, f, config)
    sandbox = _sandbox_from(f)
    job = svc.start_codegen(problems, config, client, sandbox, judge_client=judge, blame=f.flag("blame", False))
    return 202, {
        "job": job, "problems": [p.id for p in problems], "config": config.to_dict(),
        "sandbox": {"timeout": sandbox.timeout, "memory_mb": sandbox.memory_mb, "network_isolated": sandbox.network_isolated},
    }


def _r_codegen_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.codegen_history()


def _r_codegen_solve(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    raw = f._body.get("problem")
    if raw is None:
        raise ApiError(400, "missing field 'problem' (a string or {prompt, tests, expected_output})")
    try:
        problem = Problem.from_any(raw, 1)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    source = f.text("source", "model").strip().lower()
    if source not in ("model", "teacher"):
        raise ApiError(400, f"'source' must be 'model' or 'teacher' (got {source!r})")
    count = f.integer("attempts", 1, minimum=1)
    config = _codegen_config(svc, f)
    config.teacher_attempts = count
    config.model_attempts = count
    config.fallback_teacher = False
    client, judge = _codegen_clients(svc, f, config)
    sandbox = _sandbox_from(f)
    try:
        if source == "model":
            with svc.session() as model:
                trainer = CodeGenTrainer(model, client, sandbox, config, judge_client=judge)
                codes = [trainer.generate_with_model(problem, i) for i in range(count)]
            attempts = []
            for i, code in enumerate(codes):  # sandbox and judge run without the model lock
                attempt = trainer.evaluate(problem, code, "model", i)
                attempts.append(attempt)
                if attempt.verdict.correct:
                    break
        else:
            trainer = CodeGenTrainer(None, client, sandbox, config, judge_client=judge)
            attempts = trainer.solve_with_teacher(problem, None, "teacher", 0)
    except LLMError as exc:
        raise ApiError(502, str(exc)) from exc
    return 200, {
        "problem": problem.to_dict(), "source": source, "attempts": [a.to_dict() for a in attempts],
        "correct": any(a.verdict.correct for a in attempts), "sandbox": {"network_isolated": sandbox.network_isolated},
    }


def _r_codegen_run(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    code = f.text("code")
    if not code.strip():
        raise ApiError(400, "'code' is empty")
    if not code.endswith("\n"):
        code += "\n"
    strictness = f.text("strictness", "strict").strip().lower()
    sandbox = _sandbox_from(f)
    run = sandbox.run(code, tests=f.text("tests", None), expected_output=f.text("expected_output", None), stdin=f.text("stdin", ""))
    style = check_style(code)
    try:
        verdict = decide(run, style, None, strictness)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return 200, {"run": run.to_dict(), "style": style.to_dict(), "verdict": verdict.to_dict()}


def _default_tutor_model(svc: ModelService, provider: str = "ollama") -> str:
    """The model this server teaches with when a request names none.

    Ollama: ``RADIXNET_TUTOR_MODEL`` when it is set, else the model the server
    was started with.  ChatGPT: the server's own default
    (``--chatgpt-model`` / ``$RADIXNET_OPENAI_MODEL``).
    """
    if provider == "chatgpt":
        return svc.chatgpt_model
    return os.environ.get("RADIXNET_TUTOR_MODEL", "").strip() or svc.ollama_model


def _tutor_clients(svc: ModelService, f: Fields, config: TutorConfig) -> tuple[LLMClient, LLMClient]:
    """The teacher and marker clients of a tutor request (``url`` / ``grader_url`` override the server defaults)."""
    timeout = f.number("timeout", None, minimum=1.0)
    client = svc.llm_client(config.tutor_provider, f.text("url", None), config.tutor_model, timeout)
    if config.grader_provider == config.tutor_provider and not f.text("grader_url", None):
        return client, client
    grader = svc.llm_client(config.grader_provider, f.text("grader_url", None), config.resolved_grader_model, timeout)
    return client, grader


def _tutor_config(f: Fields, svc: ModelService) -> TutorConfig:
    """The tutoring settings of a request body, defaults from :class:`~radixnet.tutor.TutorConfig`."""
    d = TutorConfig()
    tutor_provider = _provider_field(f, "tutor_provider", "provider", d.tutor_provider)
    grader_provider = _provider_field(f, "grader_provider", None, tutor_provider)
    grader_model = f.text("grader_model", None) or None
    if grader_model is None and grader_provider != tutor_provider:
        grader_model = _default_tutor_model(svc, grader_provider)
    config = TutorConfig(
        topic=f.text("topic", d.topic),
        rounds=f.integer("rounds", d.rounds, minimum=1),
        exercises=f.integer("exercises", d.exercises, minimum=1),
        attempts=f.integer("attempts", d.attempts, minimum=1),
        focus=f.text("focus", None) or None,
        level=f.text("level", d.level),
        words=f.text("words", d.words),
        tutor_provider=tutor_provider,
        tutor_model=f.text("tutor_model", None) or f.text("model", None) or _default_tutor_model(svc, tutor_provider),
        grader_provider=grader_provider,
        grader_model=grader_model,
        mode=f.text("mode", d.mode).strip().lower(),
        length=f.integer("length", d.length, minimum=0),
        max_length=f.integer("max_length", d.max_length, minimum=1),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        to_end=f.flag("to_end", d.to_end),
        beam=f.integer("beam", None, minimum=1),
        threshold=f.number("threshold", d.threshold, minimum=0.0),
        grammar_weight=f.number("grammar_weight", d.grammar_weight, minimum=0.0),
        batch=f.integer("batch", d.batch, minimum=1),
        adapt=f.flag("adapt", d.adapt),
        drills=f.integer("drills", d.drills, minimum=0),
        plan=f.integer("plan", d.plan, minimum=0),
        teach_answer=f.flag("teach_answer", d.teach_answer),
        learn=f.flag("learn", d.learn),
        twonrl_per=f.text("twonrl_per", d.twonrl_per).strip().lower(),
        diff_corrections=f.flag("diff_corrections", d.diff_corrections),
        keep_weight=f.number("keep_weight", d.keep_weight, minimum=0.0),
        min_weight=f.number("min_weight", d.min_weight, minimum=0.0),
        neg_epochs=f.integer("neg_epochs", d.neg_epochs, minimum=0),
        pos_epochs=f.integer("pos_epochs", d.pos_epochs, minimum=0),
        neg_lr=f.number("neg_lr", d.neg_lr, minimum=0.0),
        pos_lr=f.number("pos_lr", d.pos_lr, minimum=0.0),
        batch_size=f.integer("batch_size", d.batch_size, minimum=1),
        strength=f.number("strength", None, minimum=0.0),
        replay=f.flag("replay", d.replay),
        replay_limit=f.integer("replay_limit", d.replay_limit, minimum=0),
        checkpoint_every=f.integer("checkpoint_every", d.checkpoint_every, minimum=0),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


def _r_tutor(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """What the tutor can be asked for: the teachers on offer, the marking vocabulary and every default."""
    return 200, {
        "url": svc.ollama_url,
        "model": _default_tutor_model(svc),
        "env_model": DEFAULT_TUTOR_MODEL,
        "providers": {
            "ollama": {"url": svc.ollama_url, "model": _default_tutor_model(svc, "ollama"), "configured": True},
            "chatgpt": {
                "url": svc.chatgpt_url,
                "model": _default_tutor_model(svc, "chatgpt"),
                "configured": chatgpt_key_configured(),
            },
        },
        "error_types": list(TUTOR_ERROR_TYPES),
        "modes": list(TUTOR_MODES),
        "twonrl_per": list(TUTOR_TWONRL_PER),
        "levels": list(TUTOR_LEVELS),
        "plan_lessons": DEFAULT_PLAN_LESSONS,
        "defaults": TutorConfig().to_dict(),
    }


def _r_tutor_start(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    config = _tutor_config(f, svc)
    client, grader = _tutor_clients(svc, f, config)
    job = svc.start_tutor(config, client, grader_client=grader, blame=f.flag("blame", False))
    return 202, {"job": job, "config": config.to_dict(), "url": client.url}


def _r_tutor_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.tutor_history()


def _r_tutor_lesson(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """One round of lessons without training: exercises, completions, grades and the report card."""
    config = _tutor_config(f, svc)
    client, grader = _tutor_clients(svc, f, config)
    given = f.texts_optional("prefixes", "prefix")
    trainer = TutorTrainer(None, client, config, grader_client=grader)
    try:
        if given:
            exercises = [
                Exercise(id=f"e{i + 1}", prefix=prefix, focus=config.focus or "")
                for i, prefix in enumerate(given[: config.exercises])
            ]
        else:
            exercises = trainer.set_exercises(1)
        with svc.session() as model:  # the search runs under the lock, the LLM calls do not
            trainer.model = model
            lessons = [trainer.complete(ex, attempt) for ex in exercises for attempt in range(config.attempts)]
        trainer.grade(lessons)
    except LLMError as exc:
        raise ApiError(502, str(exc)) from exc
    return 200, {
        "source": "given" if given else config.tutor_provider, "model": client.model, "url": client.url,
        "config": config.to_dict(), "exercises": [e.to_dict() for e in exercises],
        "lessons": [lesson.to_dict() for lesson in lessons], "report": report_card(lessons),
    }


def _r_tutor_plan(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """The lessons to run next: the teacher turns a report card into a syllabus of its weakest points."""
    config = _tutor_config(f, svc)
    client, _grader = _tutor_clients(svc, f, config)
    card = f.mapping("report", None) or svc.tutor_card()
    if card is None:
        raise ApiError(400, "no report card yet: run some lessons first, or send one as 'report'")
    count = f.integer("count", config.plan or DEFAULT_PLAN_LESSONS, minimum=1)
    try:
        plan = plan_lessons(
            client, card, topic=config.topic, level=config.level, count=count, exercises=config.exercises,
            drills=config.drills, model=config.tutor_model,
        )
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    except LLMError as exc:
        raise ApiError(502, str(exc)) from exc
    return 200, {
        "plan": plan.to_dict(), "source": plan.source, "provider": config.tutor_provider,
        "model": client.model, "url": client.url, "report": card,
    }


_ENDPOINTS: tuple[tuple[str, str, RouteFn, str], ...] = (
    ("GET", "/api/health", _r_health, "liveness check and package version"),
    ("GET", "/api/status", _r_status, "model stats (with the active kind), current job, backend availability, paths"),
    ("GET", "/api/model", _r_model, "the active model kind, every kind (radix | count) and their default files"),
    ("POST", "/api/model/select", _r_model_select,
     "switch the active model kind: {kind: radix | count}; the previous model stays in memory"),
    ("POST", "/api/train", _r_train,
     "start a training job: {texts | text | files, epochs, lr, act_lr, lr_schedule, act_lr_schedule, reverse_schedule, "
     "batch_size, auto_compress}"),
    ("GET", "/api/schedule", _r_schedule, "what a learning-rate schedule expression may use: variables, functions, helpers, presets"),
    ("POST", "/api/schedule/preview", _r_schedule_preview,
     "the rate of every epoch for schedule expressions: {lr_schedule, act_lr_schedule, epochs, lr, act_lr, "
     "reverse_schedule} -> {points}"),
    ("GET", "/api/job", _r_job, "status of the current / last job"),
    ("POST", "/api/job/stop", _r_job_stop, "ask the running job to stop"),
    ("POST", "/api/predict", _r_predict,
     "continue a prefix: {prefix, length, mode: dijkstra | beam | sample, to_end, step_penalty, temperature, max_length "
     "(optional cap; default none), k, beam (beam mode: the top-K and bottom-K continuations)}"),
    ("POST", "/api/generate", _r_generate,
     "generate whole texts with the prediction search: {count, max_length, mode: beam (the K most likely) | sample | "
     "dijkstra, temperature, seed, prefix, step_penalty, beam}"),
    ("POST", "/api/converse", _r_converse,
     "the model converses with itself - each reply is the prediction search picking up the end of the previous "
     "line: {opening, turns, mode: beam | sample, max_length, context, temperature, k, beam, step_penalty, seed, "
     "speakers, history (utterances so far, to continue), partner (another kind in memory answers), avoid_repeats}"),
    ("POST", "/api/score", _r_score, "log-probability of a text: {text}"),
    ("POST", "/api/2nrl", _r_two_nrl,
     "start a 2NRL job: {bad | bad_text, good | good_text, neg_epochs, pos_epochs, neg_lr, pos_lr, "
     "bad_weights | bad_ratings, good_weights | good_ratings (per text: how bad / how good)}"),
    ("POST", "/api/feedback", _r_feedback,
     "learn from rated texts: {good (thumbs up), bad (thumbs down), good_ratings / bad_ratings (marks out of 10, "
     "or good_weights / bad_weights as 0..1 shares), ...} -> 2NRL when both, reward on good alone, punish "
     "(negative phase + invert) on bad alone; every text is learned in proportion to its rating"),
    ("POST", "/api/invert", _r_invert, "invert the network"),
    ("POST", "/api/compress", _r_compress, "merge unary chains"),
    ("POST", "/api/evolve/start", _r_evolve_start,
     "start the GAN-style loop: {corpus | corpus_text | corpus_files, generations, samples, ..., blatant_mode: none | "
     "fail_invert | activation | state, blatant_margin, blatant_boost, blame (the discriminator also teaches the "
     "negative network)}"),
    ("POST", "/api/evolve/stop", _r_evolve_stop, "stop the evolve job"),
    ("GET", "/api/evolve/history", _r_evolve_history, "generation records of all evolve runs"),
    ("POST", "/api/save", _r_save, "save the model: {path} (default: the server's model path)"),
    ("POST", "/api/load", _r_load, "load a model file: {path}"),
    ("POST", "/api/reset", _r_reset,
     "replace the model with a fresh one: {seed, kind, count model: count_scale, global_scale, window_scale, reward_scale, window}"),
    ("POST", "/api/model/weights", _r_model_weights,
     "count model: change the dual frequency function {count_scale, global_scale, window_scale, reward_scale, window} -> {weights, stats}"),
    ("GET", "/api/negative", _r_negative,
     "the negative network: stats, the reason table (what the tutor blamed), the journal of what it said and the "
     "filter settings"),
    ("POST", "/api/negative/blame", _r_negative_blame,
     "teach it a failure: {texts | text, reason, severity, source, note, epochs} - the only thing that adds "
     "structure to the negative network"),
    ("POST", "/api/negative/clear", _r_negative_clear,
     "the tutor passed these texts: {texts | text, weight, epochs} takes blame off the fragments they share with "
     "known failures (nothing is created)"),
    ("POST", "/api/negative/judge", _r_negative_judge,
     "why texts look like failures: {texts | text, threshold, min_coverage, spans} -> verdicts with risk, coverage, "
     "the reasons and the fragments to blame"),
    ("POST", "/api/negative/filter", _r_negative_filter,
     "the pair: the positive model writes, the negative one vetoes - {count, prefix, mode, max_length, temperature, "
     "over_sample, threshold, min_coverage, ratio, no_ratio, peak (blame on a single fragment), strict, learn} or "
     "{texts} to judge given texts"),
    ("POST", "/api/negative/forget", _r_negative_forget, "drop or fade the blame behind a reason: {reason, factor}"),
    ("POST", "/api/negative/settings", _r_negative_settings,
     "how strictly it judges: {threshold, min_coverage, share_scale, blame_scale, clear_scale}"),
    ("POST", "/api/negative/reset", _r_negative_reset, "forget every failure: {seed} -> a fresh negative network"),
    ("POST", "/api/negative/save", _r_negative_save, "save the negative network: {path} (default: beside the model path)"),
    ("GET", "/api/checkpoints", _r_checkpoints, "list checkpoints and the latest pointer"),
    ("POST", "/api/checkpoints/save", _r_checkpoint_save, "write a checkpoint: {tag}"),
    ("POST", "/api/checkpoints/restore", _r_checkpoint_restore, "restore a checkpoint: {name}"),
    ("GET", "/api/graph", _r_graph, "top nodes by visit count and the edges among them (?limit=150)"),
    ("GET", "/api/history", _r_history, "the model's training history"),
    ("GET", "/api/uploads", _r_uploads, "uploaded training files (name, bytes, lines)"),
    ("POST", "/api/uploads", _r_upload,
     "upload text files or ZIP archives (a ZIP is one upload; its text files are unpacked on the server when it is "
     "used): JSON {name, content | content_base64} | {files: [...]}, multipart/form-data, or a raw body with ?name="),
    ("POST", "/api/uploads/delete", _r_upload_delete, "delete an uploaded file: {name}"),
    ("GET", "/api/images", _r_images, "the image encoders: Pillow / torch / diffusers availability, the configured VAE, what auto picks"),
    ("POST", "/api/images/encode", _r_image_encode,
     "encode an image as text (the Stable Diffusion VAE run backwards, quantised, base64): multipart / raw / JSON "
     "{name, content_base64} + ?size=128&encoder=auto|sd|tiny&train=true&save_as=NAME -> {text, encoder, latent_shape, ...}"),
    ("POST", "/api/images/decode", _r_image_decode,
     "decode an encoded or predicted text back to an image: {text, encoder} -> {png_base64, width, height, repaired}"),
    ("GET", "/api/speech", _r_speech,
     "the speech backends: faster-whisper / whisper / a transcription server, ffmpeg, the microphone recorders, "
     "the waveform codecs and the unique token"),
    ("POST", "/api/speech/transcribe", _r_speech_transcribe,
     "speech to text: audio as multipart / a raw body / JSON {name, content_base64} (+ backend, language, "
     "asr_model, asr_url, transcript) -> {transcript, backend, model, language, seconds}"),
    ("POST", "/api/speech/teach", _r_speech_teach,
     "teach the model an utterance: the same audio forms plus transcript, rate, codec, normalise, waveform, pair, "
     "token, unique, train, save_as, epochs, lr, batch_size -> {token, transcript, asr, audio, texts, job, upload}"),
    ("POST", "/api/speech/decode", _r_speech_decode,
     "decode an encoded or predicted waveform text back to audio: {text, codec} -> {wav_base64, rate, seconds, repaired}"),
    ("GET", "/api/ollama/models", _r_ollama_models, "models installed in Ollama (?url= overrides the server default); never fails"),
    ("POST", "/api/ollama/corpus", _r_ollama_corpus,
     "training lines from a prompt: {prompt, lines, style: good|garbage, model, url, save_as, train, epochs, lr, batch_size}"),
    ("POST", "/api/ollama/review", _r_ollama_review,
     "adversarial LLM review of the model's samples or {texts}: {count, prefix, max_length, threshold, apply: none|2nrl, "
     "good, good_files, blame (teach the negative network what failed and why)}"),
    ("GET", "/api/chatgpt/models", _r_chatgpt_models,
     "is ChatGPT usable as a tutor here (server-side OPENAI_API_KEY) and which models the key has (?url=); never fails"),
    ("POST", "/api/codegen/start", _r_codegen_start,
     "start a codegen job: {problems | problems_text | problem_files, phases: both|teacher|model, rounds, "
     "teacher_provider: ollama|chatgpt, teacher_model, judge_provider, judge_model, blame (the sandbox and the judge "
     "also teach the negative network), ...}"),
    ("GET", "/api/codegen/history", _r_codegen_history, "attempt / problem / round records of all codegen runs"),
    ("POST", "/api/codegen/solve", _r_codegen_solve,
     "solve one problem without training: {problem, source: model|teacher, attempts, judge, ...} -> attempts with sandbox runs and verdicts"),
    ("POST", "/api/codegen/run", _r_codegen_run, "run a program in the sandbox: {code, tests, expected_output, sandbox_timeout, memory_mb}"),
    ("GET", "/api/tutor", _r_tutor,
     "the English tutor: the teachers on offer (ollama, chatgpt: url, model, configured), the error types a "
     "completion is marked with, the completion modes and every default setting"),
    ("POST", "/api/tutor/start", _r_tutor_start,
     "start a tutor job - the teacher writes the prefixes, the network completes them, the teacher marks the "
     "grammar and the 2NRL follows: {topic, rounds, exercises, attempts, focus, level, mode, threshold, "
     "grammar_weight, drills, adapt, twonrl_per: round|lesson, diff_corrections, keep_weight, min_weight, "
     "neg_epochs, pos_epochs, neg_lr, pos_lr, tutor_provider: ollama|chatgpt, tutor_model, grader_provider, "
     "grader_model, url, grader_url, blame (every failed sentence also teaches the negative network why it "
     "failed), ...}"),
    ("GET", "/api/tutor/history", _r_tutor_history, "lesson / round / report records of all tutor runs"),
    ("POST", "/api/tutor/lesson", _r_tutor_lesson,
     "one round of lessons without training: {topic, exercises, prefixes (skip the LLM and use these), attempts, "
     "threshold, ...} -> completions with grades (grammar, spelling, fluency, error, correction) and a report card"),
    ("POST", "/api/tutor/plan", _r_tutor_plan,
     "the lesson plan a report card implies: {report (default: the card at the end of the last run), count, topic, "
     "level, exercises, drills, tutor_provider, tutor_model, url} -> {plan: {summary, level, weak, targets, "
     "lessons: [{focus, targets, topic, why, exercises, drills, prefixes}]}, source}"),
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
            if lookup == "POST" and path in _BINARY_ROUTES:
                fields = Fields(_upload_body(body, self.headers.get("Content-Type", ""), params, _BINARY_ROUTES[path]))
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

    def _is_upload_request(self) -> bool:
        """``POST /api/uploads`` bodies (text files, ZIP archives) are not held to the JSON body limit."""
        return self.command == "POST" and urlsplit(self.path).path.rstrip("/") == _UPLOAD_PATH

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
        limit = MAX_UPLOAD_BYTES if self._is_upload_request() else MAX_BODY_BYTES
        if limit is not None and length > limit:
            self.close_connection = True
            raise ApiError(413, f"request body too large ({length} bytes; the limit is {limit})")
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
    chatgpt_url: str | None = None,
    chatgpt_model: str | None = None,
    kind: str | None = None,
) -> tuple[RadixNetHTTPServer, ModelService]:
    """Build (and bind) the server; ``port=0`` picks a free port.

    ``model_path`` is loaded when it exists and is the default target of
    ``POST /api/save``; ``checkpoint_dir`` enables the checkpoint endpoints;
    ``upload_dir`` enables the upload endpoints (training files kept on the
    server); ``frontend_dir`` is the built React app; ``ollama_url`` /
    ``ollama_model`` are the defaults of the ``/api/ollama/*`` endpoints
    (``$OLLAMA_HOST`` / ``$RADIXNET_OLLAMA_MODEL`` when omitted) and
    ``chatgpt_url`` / ``chatgpt_model`` those of the ChatGPT provider
    (``$OPENAI_BASE_URL`` / ``$RADIXNET_OPENAI_MODEL``; the key always comes
    from the server's ``$OPENAI_API_KEY``).  ``quiet`` silences the
    per-request log lines (stderr).
    """
    service = ModelService(
        model_path=model_path, checkpoint_dir=checkpoint_dir, backend=backend, device=device,
        seed=seed, quiet=quiet, upload_dir=upload_dir, ollama_url=ollama_url, ollama_model=ollama_model,
        chatgpt_url=chatgpt_url, chatgpt_model=chatgpt_model, kind=kind,
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
    chatgpt_url: str | None = None,
    chatgpt_model: str | None = None,
    kind: str | None = None,
) -> None:
    """Serve until ``KeyboardInterrupt``; a running job is stopped on the way out."""
    server, service = create_server(
        host, port, model_path=model_path, checkpoint_dir=checkpoint_dir, frontend_dir=frontend_dir,
        backend=backend, device=device, seed=seed, quiet=quiet, upload_dir=upload_dir,
        ollama_url=ollama_url, ollama_model=ollama_model, chatgpt_url=chatgpt_url, chatgpt_model=chatgpt_model,
        kind=kind,
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
