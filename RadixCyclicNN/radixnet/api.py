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
from .agent import (
    BLATANT_MODES as AGENT_BLATANT_MODES,
    MEDIATION as AGENT_MEDIATION,
    PHASES as AGENT_PHASES,
    AgentConfig,
    AgentTrainer,
    Task,
    parse_task_file,
    parse_tasks,
    write_criteria,
)
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
from .encoding import (
    BACK_LABEL, END_LABEL, START_LABEL, THINK_LABEL, WINDOW, WORDS, Decoder, Encoder, Encoding, parse_encoding,
    word_rows,
)
from .gan import EvolveConfig, Evolver
from .graph import END, START, RadixCyclicGraph
from .llm import PROVIDERS, LLMClient, LLMError, normalise_provider
from .beam import Prediction, path_probability
from .assistant import (
    ANTHROPIC, OPENAI, AnthropicStream, Ask, AskError, OpenAIStream, Reply, input_units as assistant_input_units,
    kind_of_id, model_id, parse_request, respond as assistant_respond, to_format,
)
from .dialogue import DEFAULT_SPEAKERS, EXPLORE, repeats as dialogue_repeats
from .thinking import THINK_DEPTH, THINK_LENGTH, THINK_QUESTIONS, think, think_on
from .duo import FilterConfig, NegativeFilter
from .model import (
    GraphModel,
    RadixNet,
    TrainConfig,
    load_model,
    model_class,
    model_kinds,
    new_model,
)
from .negative import NegativeNet
from .penalty import DEFAULT_TRAVERSAL, resolve_traversal
from .ollama import (
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
    DEFAULT_URL as OLLAMA_DEFAULT_URL,
    STYLES as OLLAMA_STYLES,
    OllamaClient,
    OllamaError,
    adversarial_correction,
    adversarial_review,
    corpus_from_prompt,
    normalise_url,
    sample_texts,
    think_value,
    thoughts_from_prompt,
)
from .blame import teach_corrections, teach_reviews
from .schedule import ScheduleError
from .schedule import describe as describe_schedules
from .schedule import preview_points
from .search import PathResult, check_sampling
from .training import check_plan
from .speech import DEFAULT_RATE as SPEECH_DEFAULT_RATE
from .speech import SpeechError
from .speech import decode_text as decode_speech_text
from .speech import describe as describe_speech
from .speech import teach as teach_speech
from .speech import transcribe as transcribe_speech
from .tools import ToolBox, WebClient, default_toolbox, parse_call
from .tutor import (
    DEFAULT_PLAN_LESSONS,
    DEFAULT_TUTOR_MODEL,
    ERROR_TYPES as TUTOR_ERROR_TYPES,
    LEVELS as TUTOR_LEVELS,
    MODES as TUTOR_MODES,
    TWONRL_PER as TUTOR_TWONRL_PER,
    UPGRADE_STEPS as TUTOR_UPGRADE_STEPS,
    WORDS_LADDER as TUTOR_WORDS_LADDER,
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
    ("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization, X-API-Key, anthropic-version, anthropic-beta"),
    ("Access-Control-Max-Age", "86400"),
)

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"
_HTML_CONTENT_TYPE = "text/html; charset=utf-8"
_NDJSON_CONTENT_TYPE = "application/x-ndjson; charset=utf-8"

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

    def __init__(self, status: int, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.param = param  # the request field to blame, when one is (named by the /v1 error shapes)


class StreamedResponse:
    """A route's answer that is written as it happens: one JSON object per line (``application/x-ndjson``).

    ``run(write)`` does the work and hands every event to ``write``; the handler
    sends the headers with the first one and chunks the rest out as they come,
    so a client reads the events while the model is still talking.  A request
    refused *before* the first event is an ordinary 4xx JSON error; a failure
    after it is the stream's last event, ``{"event": "error", "error": ...}``,
    because the status line has already gone.
    """

    __slots__ = ("run",)

    def __init__(self, run: Callable[[Callable[[Any], None]], None]) -> None:
        self.run = run


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
        tool_options: dict | None = None,
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
        self.tool_options: dict[str, Any] = {
            "offline": False, "allow_private": False, "search_url": None, "web_timeout": 20.0,
            "max_bytes": 2_000_000, "python_tool": False, "sandbox_timeout": 10.0,
            "browser": False, "page_timeout": 30.0,
            **(tool_options or {}),
        }
        self._browser: Any = None  # one headless Chrome for the whole server, started on first use
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
        self._agent_history: list[dict] = []
        self._tutor_history: list[dict] = []
        self._critic_history: list[dict] = []
        self._chat_history: list[dict] = []
        self._discriminator: GraphModel | None = None
        self._discriminator_seed: int | None = None
        self._parked: dict[str, GraphModel] = {}  # models of the other kinds, kept while another one is active
        self._born = int(time.time())  # what /v1/models reports as every model's ``created``
        self.guard_config = FilterConfig()  # how strictly the negative network guards the output paths
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
        """The active model's kind (``"radix"``, ``"count"`` or ``"resonant"``)."""
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
            "units": self.model.encoding.units_name,
            "kinds": kinds,
            "model_path": self.model_path_for(self.kind),
            "paths": {k["kind"]: self.model_path_for(k["kind"]) for k in kinds},
            "in_memory": sorted({self.kind, *self._parked}),
            "weights": self.model.weight_config() if hasattr(self.model, "weight_config") else None,
            "attention": self.model.attention_config(),
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
            units=self.model.encoding.units_name,
            # the replay buffer the model keeps (../SPEC-SearchAndTraining.md §4): {size, texts, seen} or null
            replay=self.model.replay_summary(),
            kinds=model_kinds(),
            job=job.to_dict() if job is not None else None,
            backends=self.backends,
            model_path=self.model_path_for(self.kind),
            checkpoint_dir=self.checkpoint_dir,
            upload_dir=self.upload_dir,
            ollama={"url": self.ollama_url, "model": self.ollama_model},
            chatgpt={"url": self.chatgpt_url, "model": self.chatgpt_model, "configured": chatgpt_key_configured()},
            tools={"names": sorted(self.toolbox().names()), **self.tool_options},
        )
        return stats

    def guard(self, model: GraphModel | None = None, provenance: bool | None = None) -> NegativeFilter | None:
        """The pair on the way out: the negative network guarding ``model``'s output, or ``None``.

        ``provenance`` overrides the server's own setting for this one answer
        (``POST /api/negative/settings {"provenance": false}`` sets it for
        every answer): off, the vetoes still apply but the report says how
        many, not which or why.

        Every answer this service hands out - :meth:`generate`,
        :meth:`predict`, :meth:`converse` - goes through this filter, so the
        two networks work in tandem without anyone asking for it: the positive
        model writes and the negative one, built from nothing but the tutor's
        failures, vetoes what it recognises (:mod:`radixnet.duo`).

        ``None`` - no guard, the output goes out as written - when there is
        nothing to guard *with*: the negative network is the active model (it
        cannot filter itself), there is none in memory and none saved beside
        the model, or the one there has never been taught a failure and would
        veto nothing.  An empty negative network is never *created* here - an
        answer is not the place to bring one into being.  Call it with the
        model lock held; the output paths do.
        """
        model = self.model if model is None else model
        if isinstance(model, NegativeNet):
            return None  # generating *from* the failures: there is no positive half to guard
        negative = self._parked.get(NegativeNet.kind)
        if negative is None:
            path = self.model_path_for(NegativeNet.kind)
            if not (path and os.path.isfile(path)):
                return None
            negative = self.negative_model()  # loads it from that file and parks it
        if negative is model:
            return None
        config = self.guard_config
        if provenance is not None and bool(provenance) != config.provenance:
            config = dataclasses.replace(config, provenance=bool(provenance))  # this answer's own choice
        pair = NegativeFilter(model, negative, config)
        return pair if pair.ready else None

    @staticmethod
    def _guard_report(pair: NegativeFilter, verdicts: list[dict], **extra: Any) -> dict:
        """What the guard did, for the caller to show: the vetoes, with the reason and the fragment behind each.

        Without provenance (``FilterConfig.provenance`` off) the report is the
        counts alone - how many candidates were judged and how many vetoed -
        and neither the vetoes nor the verdicts are listed.
        """
        rejected = [v for v in verdicts if v["decision"] == "reject"]
        if not pair.config.provenance:
            return {
                "on": True,
                "provenance": False,
                "judged": len(verdicts),
                "vetoed": len(rejected),
                "negative": pair.negative.stats(),
                "config": pair.describe()["config"],
                **extra,
            }
        return {
            "on": True,
            "vetoed": len(rejected),
            "rejected": rejected,
            "verdicts": verdicts,
            "negative": pair.negative.stats(),
            "config": pair.describe()["config"],
            **extra,
        }

    def predict(self, prefix: str, *, guard: bool = True, provenance: bool | None = None, **options: Any) -> dict:
        """The active model's prediction; the count model adds ``top`` / ``bottom`` (K continuations each).

        The negative network guards the answer (:meth:`guard`): the best
        continuation it does *not* veto is the one that comes back, and when
        it vetoes every one of them nothing does - ``continuation`` is empty,
        ``full_text`` is the prefix alone and ``guard`` says why.
        ``guard=False`` hands out what the positive model wrote, unfiltered.
        """
        with self.session() as model:
            result = model.predict(prefix, **options)
            pair = self.guard(model, provenance) if guard else None
            report = None
            if pair is not None:
                result, verdicts = pair.rank(prefix, result)  # the survivors, best first
                report = self._guard_report(pair, verdicts, candidates=len(verdicts),
                                            kept=len(verdicts) - sum(v["decision"] == "reject" for v in verdicts))
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
            "guard": report,
        }
        if isinstance(result, Prediction):
            payload.update(
                mode=result.mode, k=result.k, beam=result.beam, traversal=result.traversal,
                top=[_path_dict(r) for r in result.top], bottom=[_path_dict(r) for r in result.bottom],
            )
        return payload

    def generate(self, guard: bool = True, provenance: bool | None = None, **options: Any) -> dict:
        """Whole texts from the prediction search (``beam``: the K most likely), sampling, or the cheapest path.

        The negative network guards them (:meth:`guard`): the model is asked
        for ``over_sample`` times as many as were wanted and what the negative
        half recognises as failure never reaches the answer.  Fewer than
        ``count`` samples come back when it vetoed too much - that is
        information, and ``guard`` carries every veto with its reason.
        ``guard=False`` returns what the positive model wrote, unfiltered.
        """
        with self.session() as model:
            pair = self.guard(model, provenance) if guard else None
            if pair is None:
                return {"samples": [_sample_dict(r) for r in model.generate(**options)], "guard": None}
            count = options.pop("count", 1)
            outcome = pair.generate(count=count, **options)
        return {
            "samples": [_sample_dict(r) for r in outcome["results"]],
            "guard": self._guard_report(
                pair, outcome["verdicts"], candidates=outcome["candidates"], kept=len(outcome["kept"]),
                asked=outcome["asked"], rate=outcome["rate"],
            ),
        }

    def think(self, **options: Any) -> dict:
        """The active model thinks: one thought from the THINK sentinel (:func:`radixnet.thinking.think`).

        With ``learn`` (the default) the thought teaches the model where it
        stopped to think, so - like a conversation - a thought changes the
        model; the server keeps that in memory until something saves.
        """
        with self.session() as model:
            try:
                thought = think(model, **options)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            return {"kind": model.kind, **thought.to_dict()}

    def start_train_thoughts(self, thoughts: list[str], config: TrainConfig, questions: bool = True) -> dict:
        """Start a ``train`` job that teaches ``thoughts`` as thoughts (:func:`radixnet.thinking.think_on`)."""
        config.validate()
        if self.model.kind == "negative":
            raise ApiError(400, "the negative network judges; it does not think")

        def work(job: Job) -> None:
            think_on(
                self.model, thoughts, questions=questions, config=config, progress=self._progress(job),
                stop_event=job.stop_event,
            )

        return self._start_job("train", work)

    def converse(
        self, opening: str = "", turns: int = 6, partner: str | None = None, *, guard: bool = True,
        provenance: bool | None = None, **options: Any,
    ) -> dict:
        """The active model converses with itself, or with the model of another kind kept in memory (``partner``).

        The negative network guards every turn (:meth:`guard`): a reply it
        vetoes is left unsaid and the voice looks for another one, exactly as
        it does for a line it has already spoken.  Each turn counts its own
        vetoes (``vetoed``) and ``guard`` carries them with their reasons.
        """
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
            pair = self.guard(model, provenance) if guard else None
            report = None
            try:
                if pair is None:
                    spoken = model.converse(opening, turns, partner=other, **options)
                else:
                    outcome = pair.converse(opening, turns, partner=other, **options)
                    spoken = outcome["turns"]
                    # ``vetoed`` counts the distinct texts refused, ``refusals`` how often one was (a turn may be offered
                    # the same candidate again after its context was shortened)
                    report = self._guard_report(pair, outcome["verdicts"], refusals=outcome["vetoed"])
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
        speakers = list(options.get("speakers") or DEFAULT_SPEAKERS)
        return {
            "kind": model.kind,
            "partner": other.kind if other is not None else None,
            "speakers": speakers,
            "turns": [t.to_dict() for t in spoken],
            "count": len(spoken),
            # the duplicates the search could not avoid, ready to be punished (POST /api/feedback "bad")
            "repeats": dialogue_repeats(spoken),
            "guard": report,
        }

    # -- today's format: messages in, an assistant message out ----------------

    def voice_for(self, name: str) -> GraphModel:
        """The model a request's ``model`` names: the active one, or a kind kept in memory (:func:`kind_of_id`).

        ``""``, ``radixnet`` and the active kind's id answer with the active
        model; another kind answers with the model of that kind in memory -
        which is what ``partner`` does for ``/api/converse`` - and one that is
        not there is a 404 saying so.  Call it with the model lock held.
        """
        kind = kind_of_id(name)
        if not kind or kind == self.model.kind:
            return self.model
        if kind == "word":
            for candidate in (self.model, *self._parked.values()):
                if candidate.kind == "count" and candidate.encoding.unit == WORDS:
                    return candidate
        else:
            found = self._parked.get(kind)
            if found is not None:
                return found
        raise ApiError(
            404, f"the model {name!r} is not in memory; the models here are {', '.join(self.model_ids())} "
                 "(select a kind once to load it)", param="model",
        )

    def model_ids(self) -> list[str]:
        """The ids of every model in memory, the active one first."""
        return [model_id(m) for m in (self.model, *self._parked.values())]

    def list_models(self) -> dict:
        """``GET /v1/models``: the models in memory as an OpenAI model list, ``radixnet-<kind>`` each."""
        with self.session():
            models = [self.model, *self._parked.values()]
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_id(m), "object": "model", "created": self._born, "owned_by": "radixnet",
                        "kind": m.kind, "label": type(m).label, "encoding": str(m.encoding),
                        "units": m.encoding.units_name, "active": m is self.model,
                    }
                    for m in models
                ],
            }

    def count_input(self, ask: Ask) -> dict:
        """``POST /v1/messages/count_tokens``: the units of the model's encoding a request holds."""
        with self.session():
            voice = self.voice_for(ask.model)
            return {"input_tokens": assistant_input_units(voice.encoding, ask)}

    def respond(self, ask: Ask) -> Reply:
        """Answer a request in today's format (:mod:`radixnet.assistant`), the guard on the way out.

        The negative network vetoes candidates before they are spoken, as it
        does for every other answer here (:meth:`guard`), and every veto is a
        line of the thinking with the reason; ``ask.guard`` off hands out what
        the positive model wrote.  With ``ask.learn`` (the default) a rethink
        teaches the graph where it goes round, so a conversation changes the
        model, exactly as ``/api/converse`` does.
        """
        with self.session():
            voice = self.voice_for(ask.model)
            pair = self.guard(voice) if ask.guard else None
            try:
                return assistant_respond(voice, ask, pair=pair)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc

    def respond_stream(self, ask: Ask, dialect: str) -> "EventStream":
        """:meth:`respond` as server-sent events in ``dialect``'s shape, written as the search produces them.

        The frames go out from inside the search: every line of the thinking
        the moment the search takes that step, every chunk of the text as the
        walk is decoded.  The model lock is held for the whole stream.
        """
        def run(write: Callable[[str], None]) -> None:
            with self.session():
                voice = self.voice_for(ask.model)
                pair = self.guard(voice) if ask.guard else None
                renderer: list = []

                def on_event(event: dict) -> None:
                    if event["type"] == "start" and not renderer:
                        if dialect == OPENAI:
                            renderer.append(OpenAIStream(
                                event["id"], event["model"], event["created"], include_usage=ask.include_usage,
                                thinking=ask.thinking,
                            ))
                        else:
                            renderer.append(AnthropicStream(
                                event["id"], event["model"], event["created"], input_units=event["input_units"],
                                thinking=ask.thinking,
                            ))
                    for frame in renderer[0].frames(event):
                        write(frame)

                assistant_respond(voice, ask, pair=pair, on_event=on_event)

        return EventStream(run, OpenAIStream.error if dialect == OPENAI else AnthropicStream.error)

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

    def paths(self, limit: int = 50) -> dict:
        """The judged paths: what each step did in the context it was taken from."""
        if limit < 0:
            raise ApiError(400, f"'limit' must be >= 0 (got {limit})")
        with self.session() as model:
            if not hasattr(model, "paths"):
                raise ApiError(400, f"the {model.kind} model does not count paths")
            graph = model.graph
            rows = []
            for row in model.paths(limit=limit):
                prev = graph.text_of(graph.labels[row["prev"]]) if row["prev"] < len(graph.labels) else ""
                parent = graph.edge_parent[row["edge"]] if row["edge"] < len(graph.edge_parent) else -1
                child = next((c for c, e in graph.children[parent].items() if e == row["edge"]), -1) if parent >= 0 else -1
                rows.append({
                    **row,
                    "after": prev,
                    "parent": parent,
                    "parent_label": graph.text_of(graph.labels[parent]) if 0 <= parent < len(graph.labels) else "",
                    "child": child,
                    "child_label": graph.text_of(graph.labels[child]) if 0 <= child < len(graph.labels) else "",
                })
            return {
                "totals": graph.path_totals(), "paths": rows, "limit": limit,
                "path_scale": graph.weight_config()["path_scale"],
            }

    def words(self, limit: int = 50) -> dict:
        """A word encoding's alphabet: the words the graph has read and how many grams hold each.

        There is no vocabulary object to read - a gram is text - so the
        alphabet is whatever the graph's grams are made of, which is the one
        count that survives compression.
        """
        if limit < 0:
            raise ApiError(400, f"'limit' must be >= 0 (got {limit})")
        with self.session() as model:
            encoding = model.encoding
            if encoding.unit != WORDS:
                raise ApiError(
                    400,
                    f"this model counts in {encoding.units_name}, so it has no words to list; "
                    f"a word alphabet needs a word encoding (--encoding word:{encoding.n}:{encoding.stride})",
                )
            rows = word_rows(encoding, model.graph.trigram_index)
            return {
                "words": rows[:limit] if limit > 0 else rows, "limit": limit,
                "vocabulary": len(rows), "units": encoding.units_name,
                "encoding": str(encoding),
            }

    def node_ratios(self, limit: int = 20, node: str | None = None) -> dict:
        """Each node against the nodes around it: its traffic and its reward, shared out both ways."""
        if limit < 0:
            raise ApiError(400, f"'limit' must be >= 0 (got {limit})")
        with self.session() as model:
            if not hasattr(model, "node_ratios"):
                raise ApiError(400, f"the {model.kind} model does not count node ratios")
            graph = model.graph
            wanted = None
            if node:
                # a word model is addressed in words; ``symbols_of`` is the identity everywhere else
                key = graph.symbols_of(node)
                wanted = next((i for i, label in enumerate(graph.labels) if label == key and graph.alive[i]), None)
                if wanted is None:
                    found = graph.lookup(key) if len(key) == WINDOW else None
                    if found is None:
                        raise ApiError(404, f"no node labelled {node!r}: give a node label, or one of its trigrams")
                    wanted = found[0]
            return {
                "nodes": model.node_ratios(limit=limit, node=wanted), "limit": limit,
                "node": node, "total_nodes": graph.num_nodes(), "totals": graph.path_totals(),
            }

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

    def recall_quiz(
        self, texts: list[str], labels: list[str], modality: str, *, said: list[str] | None = None,
        blame: bool = False, threshold: float = 6.0, source: str = "", **options: Any
    ) -> dict:
        """Ask the model to write out texts it was taught, mark what comes back, and optionally blame the failures.

        The recall tutor needs no LLM: the right answer is the encoded
        utterance / image itself, so the mark is the agreement over the payload
        and the original is the correction (:mod:`radixnet.recall`).
        """
        from . import blame as blame_module
        from . import recall as recall_module

        if not texts:
            raise ApiError(400, f"nothing to ask about: send a {modality} or 'texts' (already-encoded)")
        with self.mutating() if blame else self.session():
            model = self.positive_model()
            try:
                lessons = recall_module.quiz(
                    model, texts, labels=labels, said=said or [], threshold=threshold, **options
                )
            except (TypeError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            doc = {
                "modality": modality,
                "lessons": [lesson.to_dict() for lesson in lessons],
                "report": recall_module.report_card(lessons),
                "negative": None,
            }
            if blame:
                negative = self.negative_model()
                report = blame_module.teach_recall(
                    negative, lessons, threshold=threshold, source=source or modality,
                )
                doc["negative"] = {
                    "taught": {k: report[k] for k in ("blamed", "cleared", "edges", "reasons", "severity_mean")},
                    "reasons": negative.reasons(),
                    "stats": negative.stats(),
                }
            return doc

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
                "settings": {
                    "threshold": model.threshold, "min_coverage": model.min_coverage,
                    "provenance": self.guard_config.provenance,
                },
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
                    result.pop("results")  # the walks behind the texts; JSON carries the texts
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
        """Change how strictly the negative network judges (``threshold``, ``min_coverage``), its weight scales,
        and whether the guard's vetoes carry their ``provenance`` (the rule, the reasons and the fragments behind
        each) or only their count."""
        with self.mutating():
            model = self.negative_model()
            for name in ("threshold", "min_coverage"):
                value = options.get(name)
                if value is not None:
                    setattr(model, name, float(value))
            if options.get("provenance") is not None:
                self.guard_config.provenance = bool(options["provenance"])
            scales = {k: options.get(k) for k in ("share_scale", "blame_scale", "clear_scale")}
            if any(v is not None for v in scales.values()):
                try:
                    model.configure_weights(**{k: v for k, v in scales.items() if v is not None})
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
            return {
                "settings": {
                    "threshold": model.threshold, "min_coverage": model.min_coverage,
                    "provenance": self.guard_config.provenance,
                },
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
                "edges": report["edges"], "reasons": report["reasons"], "lessons": report["faults"],
                "severity_mean": report["severity_mean"], "stats": model.stats(),
                "reason_table": model.reasons(),
            }

    def negative_teach_corrections(
        self, corrections: list[dict], severity: float = 1.0, source: str = "correction"
    ) -> dict:
        """Hand a copy editor's corrections to the negative network: only the characters it changed are blamed."""
        with self.mutating():
            model = self.negative_model()
            report = teach_corrections(model, corrections, severity=severity, source=source)
            return {
                "blamed": report["blamed"], "cleared": report["cleared"], "unmatched": report["unmatched"],
                "edges": report["edges"], "edits": report["edits"], "uncorrected": report["uncorrected"],
                "reasons": report["reasons"], "lessons": report["faults"],
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

    def reset(
        self, seed: int | None = None, kind: str | None = None, encoding: Encoding | None = None, **options: Any
    ) -> dict:
        """Replace the model with a fresh one (``seed`` defaults to the server seed; ``kind`` to the active kind).

        ``encoding`` is how the new model reads text - the unit, the n of the
        n-gram and the stride - and is fixed for its life; ``None`` is the
        character trigram.

        ``options`` are the score-function settings of the kind that has them:
        ``count_scale``, ``global_scale``, ``window_scale``, ``reward_scale``
        and ``window`` for the count model; ``buckets``, ``period``,
        ``kick_scale``, ``resonance_scale``, ``amp_scale``, ``reward_scale``
        and ``concentration`` for the resonant one.
        """
        self._ensure_idle()
        cls = model_class(kind or self.kind)
        extra = {k: v for k, v in options.items() if v is not None}
        if extra and not hasattr(cls, "weight_config"):
            raise ApiError(400, f"weight options ({', '.join(sorted(extra))}) do not apply to the {cls.kind} model")
        try:
            model = cls(
                seed=self.seed if seed is None else seed, backend=self.backend_name, device=self.device,
                encoding=encoding, **extra,
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc
        return self._replace_model(model)

    # -- the encoding: what a text becomes before the graph ever sees it -----

    def encoding(self) -> dict:
        """The encoding the active model reads in: the unit, the n of the n-gram, the stride, the sentinels.

        It is a *choice*, and ``configurable`` says so - but one made when a
        model is created and fixed for its life, because the graph's labels,
        its splits and merges and its saved file are all written in it.
        ``POST /api/reset`` is where it is chosen; a model trained at one
        encoding cannot be read at another.
        """
        enc = self.model.encoding
        return {
            "encoding": str(enc),
            "unit": enc.unit,
            "window": enc.n,
            "ngram": enc.n,
            "stride": enc.stride,
            "overlap": enc.overlap,
            "start_label": START_LABEL,
            "end_label": END_LABEL,
            "back_label": BACK_LABEL,
            "think_label": THINK_LABEL,
            "configurable": True,
            "note": (
                f"Text goes in as {enc.describe()}, and comes back out of the (possibly compressed) node "
                "labels along a path. The encoding is fixed for a model's life - every label is written in "
                'it - so it is chosen when a model is made: POST /api/reset with {"encoding": "word:2:1"}, '
                "or {unit, ngram, stride}."
            ),
        }

    def encoding_preview(self, text: str) -> dict:
        """One text through the encoder and back through both decoders.

        ``windows`` is what the model's :class:`~radixnet.encoding.Encoding`
        makes of the text and ``decoded`` what its decoder makes of those again
        - ``round_trip`` is whether the two agree, which they do for any text
        the encoding can represent (:meth:`Encoding.normalize`).  ``path`` is the same
        text through the *graph*: the nodes it walks and what
        :meth:`Decoder.decode_path` reads back off their labels, which is where
        the radix compression becomes visible - a node whose label is longer
        than the window is a merged chain.  ``path.known`` is false when the
        structure cannot walk the text as it stands, and ``path.reason`` says
        which of the reasons it is: windows never seen (they are listed in
        ``unknown_windows``), or a text every window of which *is* known that
        still cannot be walked from START to END - a missing edge, a step into
        the middle of a merged node, or a text that is only part of one the
        model was trained on, since the walk has to reach the end of a text.
        """
        if not isinstance(text, str):
            raise ApiError(400, "'text' must be a string")
        enc = self.model.encoding
        encoder, decoder = Encoder(encoding=enc), Decoder(encoding=enc)
        windows = encoder.encode(text)
        decoded = decoder.decode_trigrams(windows)
        info = {
            **self.encoding(),
            "text": text,
            "chars": enc.length(text),
            "windows": windows,
            "count": len(windows),
            "decoded": decoded,
            # what comes back is what the encoding can represent
            "round_trip": decoded == enc.normalize(text),
        }
        with self.session() as model:
            graph = model.graph
            index = graph.trigram_index
            unknown = [w for w in windows if w not in index]
            info["unknown_windows"] = unknown
            walked = graph.node_path(windows) if windows else None
            if walked is None:
                if not windows:
                    reason = f"the text is shorter than one window ({enc.n} {enc.unit}s)"
                elif unknown:
                    reason = (
                        f"{len(unknown)} of the {len(windows)} windows have never been seen: "
                        + ", ".join(json.dumps(w) for w in unknown[:5])  # the Go port quotes them the same way
                        + ("..." if len(unknown) > 5 else "")
                    )
                else:
                    reason = (
                        "every window is known, but the structure cannot walk the whole text from START to END "
                        "as it stands - a missing edge, a step into the middle of a merged node, or a text that "
                        "is only part of one it was trained on (the walk has to reach the end of a text)"
                    )
                info["path"] = {
                    "known": False, "reason": reason,
                    "labels": [], "node_ids": [], "decoded": "", "nodes": 0, "compressed": 0,
                }
            else:
                labels = [graph.labels[n] for n in walked]
                real = [graph.labels[n] for n in walked if n not in (START, END)]
                info["path"] = {
                    "known": True,
                    "reason": None,
                    "labels": labels,
                    "node_ids": list(walked),
                    "decoded": decoder.decode_path(real, 0, True, skip_sentinels=False),
                    "nodes": len(real),
                    "compressed": sum(1 for label in real if enc.length(label) > enc.n),
                }
            info["kind"] = model.kind
        return info

    def configure_weights(self, **options: Any) -> dict:
        """Change the active model's score function (400 for RadixNet); returns the config and stats.

        The count model takes the dual frequency function's scales, the
        resonant one its phase and resonance settings.
        """
        with self.mutating() as model:
            if not hasattr(model, "configure_weights"):
                raise ApiError(400, f"the {model.kind} model has no configurable weight function (select another kind first)")
            try:
                config = model.configure_weights(**{k: v for k, v in options.items() if v is not None})
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            return {"weights": config, "stats": model.stats()}

    def attention(self) -> dict:
        """The active model's attention band: where inside a gram its corrections land (:mod:`radixnet.attention`)."""
        with self.session() as model:
            return {"kind": model.kind, "attention": model.attention_config()}

    def configure_attention(self, on: bool | None = None, blur: float | None = None) -> dict:
        """Switch the active model's band on (at ``blur``) or off; 400 for a kind that is never corrected."""
        with self.mutating() as model:
            try:
                config = model.configure_attention(on=on, blur=blur)
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            return {"kind": model.kind, "attention": config, "stats": model.stats()}

    def attention_preview(self, wrong: str, right: str, blur: float | None = None) -> dict:
        """Where one correction would land, gram by gram, under the writer rule and a band; changes nothing."""
        with self.session() as model:
            try:
                return {"kind": model.kind, **model.attention_preview(wrong, right, blur=blur)}
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc

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

    # -- tool use (browsing + the Ollama-mediated agent loop) ----------------

    def toolbox(
        self,
        *,
        offline: bool | None = None,
        allow_private: bool | None = None,
        search_url: str | None = None,
        web_timeout: float | None = None,
        max_bytes: int | None = None,
        python_tool: bool | None = None,
        browser: bool | None = None,
    ) -> ToolBox:
        """The tools of one request: the server's defaults with the request's overrides applied."""
        options = dict(self.tool_options)
        for key, value in (
            ("offline", offline), ("allow_private", allow_private), ("search_url", search_url),
            ("web_timeout", web_timeout), ("max_bytes", max_bytes), ("python_tool", python_tool),
            ("browser", browser),
        ):
            if value is not None:
                options[key] = value
        try:
            web = None
            if not options["offline"]:
                web = WebClient(
                    timeout=options["web_timeout"], max_bytes=options["max_bytes"],
                    allow_private=options["allow_private"], search_url=options["search_url"] or None,
                )
            sandbox = Sandbox(timeout=options["sandbox_timeout"]) if options["python_tool"] else None
            drawn_by = self.browser(options["page_timeout"]) if options["browser"] and not options["offline"] else None
            return default_toolbox(web, sandbox=sandbox, upload_dir=self.upload_dir, offline=options["offline"],
                                   browser=drawn_by)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, str(exc)) from exc

    def browser(self, page_timeout: float = 30.0) -> Any:
        """The one headless Chrome of this server, started on first use and shared by every request.

        A browser takes a second or two to start, so one per request would be
        unusable; it is stopped with the server.
        """
        from .browser import BrowserClient, describe as describe_browser

        with self._upload_lock:  # any lock will do: this only guards the one-time construction
            if self._browser is None:
                found = describe_browser()
                if not found["available"]:
                    raise ApiError(400, f"the browser is not available: {found['error']}")
                self._browser = BrowserClient(page_timeout=page_timeout)
            return self._browser

    def describe_tools(self) -> dict:
        from .browser import describe as describe_browser

        box = self.toolbox()
        return {
            "tools": box.describe(), "names": box.names(), "count": len(box),
            "options": dict(self.tool_options), "upload_dir": self.upload_dir,
            "call_format": '<tool>name {"argument": "value"}</tool>',
            "browser": describe_browser(),
        }

    def call_tool(self, toolbox: ToolBox, name: str, arguments: dict) -> dict:
        """Run one tool outside any job (the model lock is not taken: no tool touches the model)."""
        return toolbox.call(name, arguments).to_dict()

    def start_agent(
        self, tasks: list[Task], config: AgentConfig, client: OllamaClient, toolbox: ToolBox, blame: bool = False,
    ) -> dict:
        """Start an ``agent`` job: criteria, tool calls, judging, teaching and 2NRL over ``tasks``.

        With ``blame`` the judge also teaches the negative network: every failed
        attempt is blamed for what went wrong in it, and the correct run of the
        same task is the correction the diff is taken against.
        """
        config.validate()
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        negative = self.negative_model() if blame else None

        def work(job: Job) -> None:
            trainer = AgentTrainer(self.model, client, toolbox, config, external=self.pause_lock, negative=negative)
            trainer.run(
                tasks, progress=self._progress(job, self._agent_history), stop_event=job.stop_event,
                checkpoint_manager=manager,
            )

        return self._start_job("agent", work)

    def start_explore(
        self, steps: int | None, config: AgentConfig, client: OllamaClient, toolbox: ToolBox,
        seeds: list[str] | None = None, blame: bool = False,
    ) -> dict:
        """Start an ``explore`` job: the network chooses every task itself (``steps`` ``None`` / 0 = until stopped).

        ``blame`` teaches the negative network from every failure it finds
        along the way, exactly as :meth:`start_agent` does.
        """
        config.validate()
        manager = self._require_checkpoints("checkpoint_every") if config.checkpoint_every else None
        negative = self.negative_model() if blame else None

        def work(job: Job) -> None:
            trainer = AgentTrainer(self.model, client, toolbox, config, external=self.pause_lock, negative=negative)
            trainer.frontier.extend(seeds or ())
            trainer.explore(
                steps=steps, progress=self._progress(job, self._agent_history), stop_event=job.stop_event,
                checkpoint_manager=manager,
            )

        return self._start_job("explore", work)

    def agent_history(self) -> dict:
        return {"history": list(self._agent_history)}

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

    def start_critic(self, config: Any, client: LLMClient) -> dict:
        """Start a ``critic`` job: rounds of write -> review -> blame, teaching the negative network on its own.

        The positive model writes the texts, the LLM marks them and the
        failures blame the negative network with the critique as the reason and
        the mark as the severity (:mod:`radixnet.critic`).  The positive model
        is only read from - nothing here trains, rewards or inverts it - so the
        loop can be left running beside whatever else is teaching it.
        """
        from .critic import Critic

        config.validate()
        model = self.positive_model()
        negative = self.negative_model()

        def work(job: Job) -> None:
            critic = Critic(model, negative, client, config, external=self.pause_lock)
            critic.run(progress=self._progress(job, self._critic_history), stop_event=job.stop_event)

        return self._start_job("critic", work)

    def critic_history(self) -> dict:
        return {"history": list(self._critic_history)}

    def start_chat(self, config: Any, client: LLMClient, judge_client: LLMClient | None = None) -> dict:
        """Start a ``chat`` job: the LLM converses with the model, marks every reply, and both networks learn.

        The partner keeps its lines short and easy to carry on from, the model
        replies by continuing them (:func:`radixnet.dialogue.reply`), and the
        judge marks each reply against the line it answered
        (:mod:`radixnet.chat`).  Unlike the critic this loop *does* train the
        positive model - the replies that passed are rewarded and the ones that
        failed punished - so it takes the model lock like any other training
        job; the negative network learns from the same verdicts when one is in
        memory.
        """
        from .chat import Chat

        config.validate()
        negative = self.negative_model() if config.blame or config.guard else None

        def work(job: Job) -> None:
            loop = Chat(self.model, client, config, negative=negative, external=self.pause_lock,
                        judge_client=judge_client)
            loop.run(progress=self._progress(job, self._chat_history), stop_event=job.stop_event)

        return self._start_job("chat", work)

    def chat_history(self) -> dict:
        return {"history": list(self._chat_history)}

    def chat_card(self) -> dict | None:
        """The report at the end of the last conversation run, or ``None`` when it has never run."""
        for record in reversed(self._chat_history):
            if record.get("kind") == "report":
                return dict(record)
        return None

    def critic_card(self) -> dict | None:
        """The report at the end of the last automatic run, or ``None`` when it has never run."""
        for record in reversed(self._critic_history):
            if record.get("kind") == "report":
                return dict(record)
        return None

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
        """Stop a running job and wait (bounded) for its thread to finish; close the browser if one was started."""
        browser, self._browser = self._browser, None
        if browser is not None:
            browser.close()
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
    count = graph.node_count
    real = [i for i in range(2, len(graph.labels)) if alive[i]]
    top = heapq.nsmallest(limit, real, key=lambda i: (-count(i), i)) if limit < len(real) else real
    ids = [START, END, *sorted(top)]
    chosen = set(ids)
    labels, z, a, b, h, k = graph.labels, graph.z, graph.a, graph.b, graph.h, graph.k
    nodes = [
        {
            # ``text_of`` is the identity for a character model and the words for a word model
            "id": i, "label": graph.text_of(labels[i]), "count": graph.count[i],
            "count_resets": graph.count_resets.get(i, 0),
            "activation": graph.activation_of(i),
            "z": z[i], "a": a[i], "b": b[i], "h": h[i], "k": k[i],
        }
        for i in ids
    ]
    edge_w = graph.edge_w
    edge_count = graph.edge_count
    edge_reward = getattr(graph, "edge_reward", None)
    shares_of = getattr(graph, "shares", None)
    window_counts = getattr(graph, "window_edge_count", None)   # the count model only
    advance = getattr(graph, "advance", None)                   # the resonant model only
    edges = []
    for p in ids:
        shares = {e: (first, second) for _c, e, first, second in shares_of(p)} if shares_of is not None else {}
        for c, e, cost in graph.child_costs(p):
            if c in chosen:
                edge = {
                    "source": p, "target": c, "weight": edge_w[e], "count": edge_count[e],
                    "count_resets": graph.edge_count_resets.get(e, 0),
                    "prob": math.exp(-cost), "cost": cost,
                }
                if edge_reward is not None:
                    edge["reward"] = edge_reward[e]
                if window_counts is not None:
                    edge["share"], edge["recent_share"] = shares.get(e, (0.0, 0.0))
                    edge["recent_count"] = window_counts[e]
                elif advance is not None:
                    edge["coherence"], edge["mu"] = shares.get(e, (0.0, 0.0))
                edges.append(edge)
    edges.sort(key=lambda d: (d["source"], d["target"]))
    view = {
        "nodes": nodes, "edges": edges, "limit": limit,
        "total_nodes": graph.num_nodes(), "total_edges": graph.num_edges(),
    }
    if edge_reward is not None:
        view["total_traversals"] = graph.total_traversals.value
        view["total_traversals_resets"] = graph.total_traversals.resets
    if window_counts is not None:
        view["window_traversals"] = graph.window_traversals
        view["window"] = graph.window
    if advance is not None:
        view["buckets"] = graph.buckets
        for node in nodes:
            node["advance"] = advance[node["id"]]
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

    def body(self) -> dict:
        """The body as it arrived, for a request read as one document (the ``/v1`` dialects)."""
        return self._body

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


class EventStream:
    """An answer streamed as server-sent events: ``run(write)`` writes every frame the moment it is produced.

    A route returns one in place of a JSON document.  The handler sends the
    headers, calls ``run`` with a function that writes one frame, and closes
    the connection when it returns; ``error(message)`` is the frame a failure
    after the headers went out is reported with, in the dialect's own shape.
    """

    content_type = "text/event-stream; charset=utf-8"

    def __init__(self, run: Callable[[Callable[[str], None]], None], error: Callable[[str], str]) -> None:
        self.run = run
        self.error = error


def _v1_dialect(path: str) -> str | None:
    """Which dialect a ``/v1`` path speaks (``None`` for the API's own routes)."""
    if not (path == "/v1" or path.startswith("/v1/")):
        return None
    return ANTHROPIC if path.startswith("/v1/messages") else OPENAI


def _shape_error(dialect: str | None, status: int, message: str, param: str | None = None) -> dict:
    """An error as the client expects it: the API's ``{"error": "..."}``, or the dialect's own envelope."""
    if dialect is None:
        return {"error": message}
    if dialect == OPENAI:
        kind = "server_error" if status >= 500 else "invalid_request_error"
        code = "model_not_found" if status == 404 and param == "model" else None
        return {"error": {"message": message, "type": kind, "param": param, "code": code}}
    if status >= 500:
        kind = "api_error"
    elif status == 404:
        kind = "not_found_error"
    elif status == 413:
        kind = "request_too_large"
    else:
        kind = "invalid_request_error"
    return {"type": "error", "error": {"type": kind, "message": message}}


def _v1_ask(f: Fields, dialect: str) -> Ask:
    try:
        return parse_request(f.body(), dialect)
    except AskError as exc:
        raise ApiError(400, exc.message, param=exc.param) from exc


def _r_v1_chat_completions(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    ask = _v1_ask(f, OPENAI)
    if ask.stream:
        with svc.session():
            svc.voice_for(ask.model)  # a model that is not here is a 404 before any frame goes out
        return 200, svc.respond_stream(ask, OPENAI)
    return 200, to_format(svc.respond(ask), OPENAI)


def _r_v1_messages(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    ask = _v1_ask(f, ANTHROPIC)
    if ask.stream:
        with svc.session():
            svc.voice_for(ask.model)
        return 200, svc.respond_stream(ask, ANTHROPIC)
    return 200, to_format(svc.respond(ask), ANTHROPIC)


def _r_v1_count_tokens(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.count_input(_v1_ask(f, ANTHROPIC))


def _r_v1_models(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.list_models()


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
    """The optional training settings of a request body, defaults from :class:`TrainConfig`.

    ``order``, ``curriculum``, ``replay``, ``replay_size``, ``patience``,
    ``min_delta`` and ``reverse`` are the training methods of
    ``../SPEC-SearchAndTraining.md``; a value out of range is a 400, before any
    job starts.
    """
    config = TrainConfig(
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
        order=f.text("order", TrainConfig.order).strip().lower() or TrainConfig.order,
        curriculum=f.number("curriculum", TrainConfig.curriculum),
        replay=f.number("replay", TrainConfig.replay, minimum=0.0),
        replay_size=f.integer("replay_size", None, minimum=0),
        patience=f.integer("patience", TrainConfig.patience, minimum=0),
        min_delta=f.number("min_delta", TrainConfig.min_delta, minimum=0.0),
        reverse=f.flag("reverse", TrainConfig.reverse),
    )
    try:
        check_plan(config.order, config.curriculum, config.replay, config.replay_size, config.patience,
                   config.min_delta)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


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


def _traversal_fields(f: Fields) -> dict:
    """``traversal`` and its two scales, as every search endpoint reads them (:mod:`radixnet.penalty`)."""
    return {
        "traversal": resolve_traversal(f.text("traversal", DEFAULT_TRAVERSAL)),
        "penalty_scale": f.number("penalty_scale", 1.0, minimum=0.0),
        "merit_scale": f.number("merit_scale", 1.0, minimum=0.0),
    }


def _search_fields(f: Fields) -> dict:
    """The sampling filters and the beam's diversity (``../SPEC-SearchAndTraining.md`` §1-2).

    Only the ones a request gives are passed on - each is off at its default,
    so a request that names none of them searches exactly as it always did.
    """
    out: dict[str, Any] = {}
    if f.present("top_k"):
        out["top_k"] = f.integer("top_k", 0, minimum=0)
    if f.present("top_p"):
        out["top_p"] = f.number("top_p", 1.0)
    if f.present("min_p"):
        out["min_p"] = f.number("min_p", 0.0)
    if f.present("diversity"):
        out["diversity"] = f.number("diversity", 0.0, minimum=0.0)
    try:
        check_sampling(out.get("top_k", 0), out.get("top_p", 1.0), out.get("min_p", 0.0))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return out


def _r_predict(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    prefix = f.text("prefix")
    return 200, svc.predict(
        prefix,
        guard=f.flag("guard", True),
        provenance=f.flag("provenance", None),
        length=f.integer("length", 20, minimum=0),
        mode=f.text("mode", "dijkstra"),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        to_end=f.flag("to_end", False),
        max_length=f.integer("max_length", None, minimum=0),
        k=f.integer("k", 5, minimum=0),
        beam=f.integer("beam", None, minimum=1),
        **_traversal_fields(f),
        **_search_fields(f),
    )


def _r_generate(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.generate(
        guard=f.flag("guard", True),
        provenance=f.flag("provenance", None),
        count=f.integer("count", 1, minimum=0),
        max_length=f.integer("max_length", 60, minimum=0),
        mode=f.text("mode", "sample"),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        seed=f.integer("seed", None),
        prefix=f.text("prefix", ""),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        beam=f.integer("beam", None, minimum=1),
        **_traversal_fields(f),
        **_search_fields(f),
    )


def _converse_fields(f: Fields) -> dict:
    """The one conversation ``POST /api/converse`` and ``POST /api/converse/stream`` both read from a body."""
    partner = f.text("partner", None)
    return dict(
        opening=f.text("opening", ""),
        turns=f.integer("turns", 6, minimum=0),
        partner=partner or None,
        guard=f.flag("guard", True),
        provenance=f.flag("provenance", None),
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
        avoid_word_repeats=f.flag("avoid_word_repeats", True),
        explore=f.integer("explore", EXPLORE, minimum=0),
        learn=f.flag("learn", True),
        think=f.flag("think", True),
        think_depth=f.integer("think_depth", THINK_DEPTH, minimum=0),
    )


def _r_think(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.think(
        about=f.text("about", ""),
        mode=f.text("mode", "beam"),
        k=f.integer("k", 5, minimum=1),
        beam=f.integer("beam", None, minimum=1),
        max_length=f.integer("max_length", THINK_LENGTH, minimum=0),
        temperature=f.number("temperature", 1.0, minimum=0.0),
        step_penalty=f.number("step_penalty", 0.0, minimum=0.0),
        seed=f.integer("seed", None),
        max_depth=f.integer("depth", THINK_DEPTH, minimum=0),
        max_questions=f.integer("questions", THINK_QUESTIONS, minimum=0),
        learn=f.flag("learn", True),
    )


def _r_converse(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.converse(**_converse_fields(f))


def _r_converse_stream(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """The same conversation, streamed as it happens (:class:`radixnet.dialogue.StreamFn`): every event is one
    JSON line, ``turn`` events are the answer and the rest is the window a backtrack may still rewrite, and
    the last line is ``{"event": "done", ...}`` carrying the document ``POST /api/converse`` answers with."""
    options = _converse_fields(f)

    def run(write: Callable[[Any], None]) -> None:
        document = svc.converse(stream=write, **options)
        write({"event": "done", **document})

    return 200, StreamedResponse(run)


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
    """Score-function settings of every kind that has one; the model rejects the ones it does not know."""
    return {
        # count model
        "count_scale": f.number("count_scale", None),
        "global_scale": f.number("global_scale", None),
        "window_scale": f.number("window_scale", None),
        "path_scale": f.number("path_scale", None),
        "window": f.integer("window", None, minimum=1),
        # resonant model
        "buckets": f.integer("buckets", None, minimum=1),
        "period": f.number("period", None),
        "kick_scale": f.number("kick_scale", None),
        "resonance_scale": f.number("resonance_scale", None),
        "amp_scale": f.number("amp_scale", None),
        "concentration": f.number("concentration", None),
        # both
        "reward_scale": f.number("reward_scale", None),
    }


def _encoding_option(f: Fields) -> Encoding | None:
    """``encoding`` as a spec, or the three dials on their own; ``None`` when nothing was asked for."""
    spec = f.text("encoding", None) or ""
    unit = f.text("unit", None) or ""
    n = f.integer("ngram", None)
    stride = f.integer("stride", None)
    if not (spec or unit or n is not None or stride is not None):
        return None
    try:
        enc = parse_encoding(spec)
        if unit:
            enc = Encoding(unit=unit, n=enc.n, stride=enc.stride)
        if n is not None:
            enc = Encoding(unit=enc.unit, n=n, stride=min(enc.stride if enc.sliding else n, n))
        if stride is not None:
            enc = Encoding(unit=enc.unit, n=enc.n, stride=stride)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return enc


def _r_reset(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.reset(
        f.integer("seed", None), kind=f.text("kind", None) or None, encoding=_encoding_option(f),
        **_weight_options(f),
    )


def _r_model_weights(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.configure_weights(**_weight_options(f))


def _r_attention(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.attention()


def _r_attention_set(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.configure_attention(on=f.flag("on", None), blur=f.number("blur", None))


def _r_attention_preview(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.attention_preview(f.text("wrong", "") or "", f.text("right", "") or "", blur=f.number("blur", None))


def _r_encoding(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.encoding()


def _r_encoding_preview(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.encoding_preview(f.text("text", "") or "")


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
        provenance=f.flag("provenance", True),
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
        provenance=f.flag("provenance", None),
    )


def _r_negative_reset(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.negative_reset(f.integer("seed", None))


def _critic_config(f: Fields) -> Any:
    """The automatic-teaching settings of a request body (:class:`~radixnet.critic.CriticConfig` defaults)."""
    from .critic import CriticConfig

    d = CriticConfig()
    config = CriticConfig(
        rounds=f.integer("rounds", d.rounds, minimum=0),
        count=f.integer("count", d.count, minimum=1),
        prefix=f.text("prefix", d.prefix),
        max_length=f.integer("max_length", d.max_length, minimum=0),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        threshold=f.number("threshold", d.threshold, minimum=0.0),
        context=f.text("context", d.context),
        provider=_provider_field(f, "provider", "reviewer_provider", d.provider),
        reviewer_model=f.text("reviewer_model", None) or f.text("model", None) or d.reviewer_model,
        clear_passes=f.flag("clear_passes", d.clear_passes),
        epochs=f.integer("epochs", d.epochs, minimum=0),
        seed=f.integer("seed", d.seed),
        correct=f.flag("correct", d.correct),
        severity=f.number("severity", d.severity, minimum=0.0),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


def _r_negative_auto(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Start the automatic loop: the model writes, the LLM reviews, the failures blame the negative network."""
    config = _critic_config(f)
    client = svc.llm_client(config.provider, f.text("url", None), config.reviewer_model or None,
                            f.number("timeout", None, minimum=1.0))
    job = svc.start_critic(config, client)
    return 202, {"job": job, "config": config.to_dict(), "url": client.url, "reviewer": client.model}


def _r_negative_auto_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.critic_history()


def _chat_config(f: Fields) -> Any:
    """The conversation settings of a request body (:class:`~radixnet.chat.ChatConfig` defaults)."""
    from .chat import ChatConfig

    d = ChatConfig()
    config = ChatConfig(
        conversations=f.integer("conversations", d.conversations, minimum=0),
        turns=f.integer("turns", d.turns, minimum=1),
        topic=f.text("topic", d.topic),
        opening=f.text("opening", d.opening),
        persona=f.text("persona", d.persona),
        context=f.integer("context", d.context, minimum=0),
        max_length=f.integer("max_length", d.max_length, minimum=1),
        mode=f.text("mode", d.mode).strip().lower(),
        k=f.integer("k", d.k, minimum=1),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        partner_temperature=f.number("partner_temperature", d.partner_temperature, minimum=0.0),
        threshold=f.number("threshold", d.threshold, minimum=0.0),
        provider=_provider_field(f, "provider", "partner_provider", d.provider),
        partner_model=f.text("partner_model", None) or f.text("model", None) or d.partner_model,
        judge_model=f.text("judge_model", None) or d.judge_model,
        guard=f.flag("guard", d.guard),
        blame=f.flag("blame", d.blame),
        clear_passes=f.flag("clear_passes", d.clear_passes),
        learn=f.flag("learn", d.learn),
        teach_partner=f.flag("teach_partner", d.teach_partner),
        avoid_repeats=f.flag("avoid_repeats", d.avoid_repeats),
        avoid_word_repeats=f.flag("avoid_word_repeats", d.avoid_word_repeats),
        explore=f.integer("explore", d.explore, minimum=0),
        neg_epochs=f.integer("neg_epochs", d.neg_epochs, minimum=0),
        pos_epochs=f.integer("pos_epochs", d.pos_epochs, minimum=0),
        neg_lr=f.number("neg_lr", d.neg_lr, minimum=0.0),
        pos_lr=f.number("pos_lr", d.pos_lr, minimum=0.0),
        batch_size=f.integer("batch_size", d.batch_size, minimum=1),
        strength=f.number("strength", d.strength, minimum=0.0),
        epochs=f.integer("epochs", d.epochs, minimum=0),
        seed=f.integer("seed", d.seed),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


def _r_chat_start(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Start a conversation job: the LLM talks to the model, marks every reply, and both networks learn."""
    config = _chat_config(f)
    timeout = f.number("timeout", None, minimum=1.0)
    client = svc.llm_client(config.provider, f.text("url", None), config.partner_model or None, timeout)
    judge = client
    judge_url = f.text("judge_url", None)
    if config.judge_model or judge_url:
        judge = svc.llm_client(config.provider, judge_url, config.judge_model or None, timeout)
    job = svc.start_chat(config, client, judge_client=judge)
    return 202, {
        "job": job, "config": config.to_dict(), "url": client.url, "partner": client.model, "judge": judge.model,
        "speakers": list(_chat_speakers()),
    }


def _chat_speakers() -> tuple[str, str]:
    from .chat import DEFAULT_SPEAKERS

    return DEFAULT_SPEAKERS


def _r_chat_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.chat_history()


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


def _r_paths(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    raw = q.get("limit", ["50"])[-1]
    try:
        limit = int(raw)
    except ValueError:
        raise ApiError(400, f"query parameter 'limit' must be an integer (got {raw!r})") from None
    return 200, svc.paths(limit)


def _r_nodes(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    raw = q.get("limit", ["20"])[-1]
    try:
        limit = int(raw)
    except ValueError:
        raise ApiError(400, f"query parameter 'limit' must be an integer (got {raw!r})") from None
    node = q.get("node", [None])[-1]
    return 200, svc.node_ratios(limit, node)


def _r_words(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    raw = q.get("limit", ["50"])[-1]
    try:
        limit = int(raw)
    except ValueError:
        raise ApiError(400, f"query parameter 'limit' must be an integer (got {raw!r})") from None
    return 200, svc.words(limit)


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


def _recall_options(f: Fields, q: dict, modality: str) -> dict:
    """The recall-tutor options a request may carry (see :mod:`radixnet.recall`)."""
    lead = _option(f, q, "lead", None)
    try:
        return {
            "lead": None if lead in (None, "") else int(lead),
            "length": int(_option(f, q, "length", 0) or 0),
            "attempts": max(1, int(_option(f, q, "attempts", 1) or 1)),
            "mode": str(_option(f, q, "mode", "beam") or "beam"),
            "temperature": float(_option(f, q, "temperature", 1.0) or 1.0),
            "listen_back": modality == "speech" and _flag_option(f, q, "listen_back"),
        }
    except (TypeError, ValueError) as exc:
        raise ApiError(400, f"bad recall option: {exc}") from exc


def _recall_threshold(f: Fields, q: dict) -> float:
    try:
        return float(_option(f, q, "threshold", 6.0) or 6.0)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "'threshold' must be a number out of 10") from exc


def _r_image_tutor(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Ask the network to draw back a picture it was shown, and blame what it got wrong."""
    texts = [t for t in f.texts_optional("texts", "text") if t.strip()]
    labels = [f"text {i + 1}" for i in range(len(texts))]
    if not texts:
        files = f.upload_files()
        name, payload = files[0]
        if not isinstance(payload, bytes):
            raise ApiError(400, "send the image as bytes: multipart/form-data, a raw body, or JSON {name, content_base64}")
        try:
            size = int(_option(f, q, "size", 0) or 0) or 128
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "'size' must be an integer") from exc
        try:
            texts = [encode_image(payload, size=size, encoder=str(_option(f, q, "encoder", "auto") or "auto"))["text"]]
        except VisionError as exc:
            raise ApiError(400, str(exc)) from exc
        labels = [name]
    return 200, svc.recall_quiz(
        texts, labels, "image", blame=_flag_option(f, q, "blame"), threshold=_recall_threshold(f, q),
        **_recall_options(f, q, "image"),
    )


def _r_speech_tutor(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """Ask the network to say back an utterance it was taught, and blame what it got wrong."""
    texts = [t for t in f.texts_optional("texts", "text") if t.strip()]
    labels = [f"text {i + 1}" for i in range(len(texts))]
    said = [""] * len(texts)
    if not texts:
        name, payload = _audio_bytes(f, "recording")
        try:
            rate = int(_option(f, q, "rate", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "'rate' must be an integer") from exc
        try:
            taught = teach_speech(
                payload,
                transcript=str(_option(f, q, "transcript", "") or ""),
                backend=str(_option(f, q, "backend", "auto") or "auto"),
                language=_option(f, q, "language", None) or None,
                asr_model=_option(f, q, "asr_model", None) or None,
                asr_url=_option(f, q, "asr_url", None) or None,
                rate=rate or SPEECH_DEFAULT_RATE,
                codec=str(_option(f, q, "codec", "auto") or "auto"),
                normalise=_flag_option(f, q, "normalise"),
                token=_option(f, q, "token", None) or None,
                unique=_flag_option(f, q, "unique", True),
            )
        except SpeechError as exc:
            raise ApiError(400, str(exc)) from exc
        waveform = next((t for t in taught["texts"] if "aud:" in t), "")
        if not waveform:
            raise ApiError(400, "nothing to remember: the recording produced no waveform")
        texts, labels, said = [waveform], [f"{name} {taught['token']}"], [taught["transcript"]]
    return 200, svc.recall_quiz(
        texts, labels, "speech", said=said, blame=_flag_option(f, q, "blame"),
        threshold=_recall_threshold(f, q), **_recall_options(f, q, "speech"),
    )


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


def _r_ollama_think(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """A thinking model thinks about a prompt; its thinking is returned and, with ``train``, taught as thoughts."""
    prompt = f.text("prompt")
    if not prompt.strip():
        raise ApiError(400, "'prompt' must not be empty")
    lines = f.integer("lines", 5, minimum=1)
    raw_think = f._lookup("think")
    try:
        level = think_value(True if raw_think is _MISSING else raw_think)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    temperature = f.number("temperature", 0.7, minimum=0.0)
    train = f.flag("train", False)
    with_answers = f.flag("with_answers", False)
    questions = f.flag("questions", True)
    client = _ollama_client(svc, f)
    if train:
        svc._ensure_idle()  # do not spend an LLM call on a request that cannot start a job
        if svc.model.kind == "negative":
            raise ApiError(400, "the negative network judges; it does not think")
    try:
        thoughts = thoughts_from_prompt(
            client, prompt, lines=lines, model=client.model, think=level, temperature=temperature,
        )
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    if not thoughts:
        raise ApiError(502, f"Ollama model {client.model!r} wrote no questions to think about")
    thinking = [t["thinking"] for t in thoughts if t["thinking"]]
    result: dict[str, Any] = {
        "prompt": prompt, "model": client.model, "url": client.url, "think": level, "count": len(thoughts),
        "thinking": len(thinking), "thoughts": thoughts, "upload": None, "job": None,
    }
    save_as = f.text("save_as", None)
    if save_as:
        result["upload"] = svc.upload(save_as, "\n".join(thinking) + ("\n" if thinking else ""))
    if not train:
        return 200, result
    if not thinking:
        raise ApiError(
            502, f"Ollama model {client.model!r} returned no thinking to train on: use a thinking model "
                 "(qwen3, deepseek-r1, gpt-oss, ...) on an Ollama that separates it",
        )
    config = _train_config(f)
    if with_answers:
        answers = [t["answer"] for t in thoughts if t["answer"]]
        config.validate()

        def work(job: Job) -> None:
            think_on(
                svc.model, thinking, questions=questions, config=config, progress=svc._progress(job),
                stop_event=job.stop_event,
            )
            if answers and not job.stop_event.is_set():
                svc.model.train(answers, config, progress=svc._progress(job), stop_event=job.stop_event)

        result["job"] = svc._start_job("train", work)
    else:
        result["job"] = svc.start_train_thoughts(thinking, config, questions=questions)
    return 202, result


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


def _r_ollama_correct(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """The copy editor: the model's samples (or given texts) written out correctly; the diff can blame."""
    given = f.texts_optional("texts", "text")
    client = _ollama_client(svc, f)
    severity = f.number("severity", 1.0, minimum=0.0)
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
        result = adversarial_correction(
            None, client, texts=samples, context=f.text("context", None), ollama_model=client.model,
        )
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    result["source"] = source
    result["url"] = client.url
    result["severity"] = severity
    result["negative"] = None
    if f.flag("blame", False):
        result["negative"] = svc.negative_teach_corrections(result["corrections"], severity=severity, source="correction")
    return 200, result


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


def _toolbox_from(svc: ModelService, f: Fields, q: dict | None = None) -> ToolBox:
    """The request's toolbox: the server's tool defaults with ``offline`` / ``allow_private`` / ... applied."""
    query = q or {}

    def flag(name: str) -> bool | None:
        if f.present(name):
            return bool(f.flag(name, False))
        raw = (query.get(name) or [None])[0]
        return None if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")

    return svc.toolbox(
        offline=flag("offline"), allow_private=flag("allow_private"), python_tool=flag("python_tool"),
        browser=flag("browser"),
        search_url=f.text("search_url", None), web_timeout=f.number("web_timeout", None, minimum=0.1),
        max_bytes=f.integer("max_bytes", None, minimum=1024),
    )


def _agent_config(f: Fields) -> AgentConfig:
    """An :class:`AgentConfig` from the request body (every field optional)."""
    d = AgentConfig()
    phase = f.text("phase", "model").strip().lower()
    if phase not in AGENT_PHASES + ("both",):
        raise ApiError(400, f"'phase' must be 'model', 'teacher' or 'both' (got {phase!r})")
    mediation = f.text("mediation", d.mediation).strip().lower()
    if mediation not in AGENT_MEDIATION:
        raise ApiError(400, f"'mediation' must be one of {', '.join(AGENT_MEDIATION)} (got {mediation!r})")
    blatant = f.text("blatant_mode", d.blatant_mode).strip().lower()
    if blatant not in AGENT_BLATANT_MODES:
        raise ApiError(400, f"'blatant_mode' must be one of {', '.join(AGENT_BLATANT_MODES)} (got {blatant!r})")
    twonrl_per = f.text("twonrl_per", d.twonrl_per).strip().lower()
    if twonrl_per not in ("task", "round"):
        raise ApiError(400, f"'twonrl_per' must be 'task' or 'round' (got {twonrl_per!r})")
    config = AgentConfig(
        agent_model=f.text("agent_model", None) or f.text("model", None) or d.agent_model,
        judge_model=f.text("judge_model", None),
        phases=AGENT_PHASES if phase == "both" else (phase,),
        rounds=f.integer("rounds", d.rounds, minimum=1),
        max_steps=f.integer("max_steps", d.max_steps, minimum=1),
        model_attempts=f.integer("model_attempts", d.model_attempts, minimum=1),
        teacher_attempts=f.integer("teacher_attempts", d.teacher_attempts, minimum=1),
        first_attempt_dijkstra=f.flag("first_attempt_dijkstra", d.first_attempt_dijkstra),
        temperature=f.number("temperature", d.temperature, minimum=0.0),
        max_length=f.integer("max_length", d.max_length, minimum=1),
        mediation=mediation,
        criteria_count=f.integer("criteria", d.criteria_count, minimum=1),
        strict=f.flag("strict", d.strict),
        use_judge=f.flag("judge", d.use_judge),
        teach_on_failure=f.flag("teach", d.teach_on_failure),
        observation_chars=f.integer("observation_chars", d.observation_chars, minimum=0),
        twonrl_per=twonrl_per,
        replay=f.flag("replay", d.replay),
        read_reward=f.flag("read_reward", d.read_reward),
        blatant_mode=blatant,
        blatant_margin=f.number("blatant_margin", d.blatant_margin, minimum=0.001),
        blatant_boost=f.number("blatant_boost", d.blatant_boost, minimum=1.0),
        avoid_blamed=f.flag("avoid_blamed", d.avoid_blamed),
        avoid_threshold=f.number("avoid_threshold", d.avoid_threshold, minimum=0.0),
        neg_epochs=f.integer("neg_epochs", d.neg_epochs, minimum=0),
        pos_epochs=f.integer("pos_epochs", d.pos_epochs, minimum=0),
        neg_lr=f.number("neg_lr", d.neg_lr, minimum=0.0),
        pos_lr=f.number("pos_lr", d.pos_lr, minimum=0.0),
        batch_size=f.integer("batch_size", d.batch_size, minimum=1),
        checkpoint_every=f.integer("checkpoint_every", 0, minimum=0),
        seed=f.integer("seed", 0),
    )
    try:
        config.validate()
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return config


def _tasks_from(svc: ModelService, f: Fields) -> list[Task]:
    """``tasks`` (strings / objects), ``tasks_text`` (one per line) and ``task_files`` (uploads)."""
    items: list[Any] = []
    raw = f._body.get("tasks")
    if raw is not None:
        if not isinstance(raw, list):
            raise ApiError(400, "'tasks' must be a list of strings or {prompt, criteria, answer, seeds} objects")
        items.extend(raw)
    text = f.text("tasks_text", "")
    if text.strip():
        items.extend(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    tasks: list[Task] = []
    try:
        if items:
            tasks = parse_tasks(items)
        for name in f.names("task_files"):
            tasks.extend(parse_task_file(svc.read_upload(name), os.path.splitext(name)[1]))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    if not tasks:
        raise ApiError(400, "no tasks: give 'tasks', 'tasks_text' or 'task_files'")
    for index, task in enumerate(tasks, 1):  # ids must stay unique once the sources are combined
        task.id = task.id if task.id not in {t.id for t in tasks[: index - 1]} else f"t{index}"
    return tasks


def _agent_client(svc: ModelService, f: Fields, config: AgentConfig) -> OllamaClient:
    return svc.ollama_client(f.text("url", None), config.agent_model, f.number("timeout", None, minimum=1.0))


def _r_tools(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.describe_tools()


def _r_tool_call(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    toolbox = _toolbox_from(svc, f, q)
    raw = f.text("call", "")
    if raw.strip():
        call = parse_call(raw if raw.lstrip().startswith("<tool>") else f"<tool>{raw}</tool>", toolbox)
        if call is None or not call.ok:
            raise ApiError(400, call.error if call is not None else f"no tool call in {raw[:120]!r}")
        name, arguments = call.name, call.arguments
    else:
        name = f.text("tool").strip()
        arguments = f._body.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ApiError(400, "'arguments' must be an object")
    if not toolbox.has(name):
        raise ApiError(404, f"unknown tool {name!r} (have: {', '.join(toolbox.names())})")
    return 200, svc.call_tool(toolbox, name, arguments)


def _r_agent_start(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    tasks = _tasks_from(svc, f)
    config = _agent_config(f)
    toolbox = _toolbox_from(svc, f)
    client = _agent_client(svc, f, config)
    blame = f.flag("blame", False)
    job = svc.start_agent(tasks, config, client, toolbox, blame=blame)
    return 202, {"job": job, "tasks": [t.to_dict() for t in tasks], "tools": toolbox.names(),
                 "config": config.to_dict(), "blame": blame}


def _r_agent_explore(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    config = _agent_config(f)
    toolbox = _toolbox_from(svc, f)
    client = _agent_client(svc, f, config)
    steps = f.integer("steps", 10, minimum=0)
    seeds = [s for s in f.texts_optional("seed_urls", "seed_url") if s.strip()]
    blame = f.flag("blame", False)
    job = svc.start_explore(steps or None, config, client, toolbox, seeds, blame=blame)
    return 202, {"job": job, "steps": steps or None, "seeds": seeds, "tools": toolbox.names(),
                 "config": config.to_dict(), "blame": blame}


def _r_agent_history(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    return 200, svc.agent_history()


def _r_agent_criteria(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """The acceptance criteria the LLM writes for a task, without attempting anything."""
    tasks = _tasks_from(svc, f)
    config = _agent_config(f)
    client = _agent_client(svc, f, config)
    out = []
    try:
        for task in tasks:
            out.append({"task": task.id, "prompt": task.prompt,
                        "criteria": write_criteria(client, task, model=config.agent_model, count=config.criteria_count)})
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    return 200, {"tasks": out, "model": config.agent_model, "url": client.url}


def _r_agent_solve(svc: ModelService, f: Fields, q: dict) -> tuple[int, Any]:
    """One task through the loop with no training: the network's attempt, or the teacher's demonstration."""
    tasks = _tasks_from(svc, f)
    if len(tasks) != 1:
        raise ApiError(400, f"/api/agent/solve takes exactly one task (got {len(tasks)})")
    task = tasks[0]
    source = f.text("source", "model").strip().lower()
    if source not in ("model", "teacher"):
        raise ApiError(400, f"'source' must be 'model' or 'teacher' (got {source!r})")
    config = _agent_config(f)
    config.teach_on_failure = False
    toolbox = _toolbox_from(svc, f)
    client = _agent_client(svc, f, config)
    try:
        with svc.session() as model:
            trainer = AgentTrainer(model, client, toolbox, config)
            criteria = trainer.criteria_for(task)
            if source == "model":
                attempt = trainer.solve_with_model(task, 0, None, "model")
            else:
                attempt = trainer.solve_with_teacher(task, criteria, 0)
            attempt.verdict = trainer.judge(task, criteria, attempt)
            gap = trainer.gap_of(attempt, criteria)
    except OllamaError as exc:
        raise ApiError(502, str(exc)) from exc
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return 200, {
        "task": task.to_dict(), "source": source, "criteria": criteria, "attempt": attempt.to_dict(),
        "transcript": attempt.text, "correct": attempt.verdict.correct, "gap": gap,
        "frontier": trainer.frontier[:20], "tools": toolbox.names(),
    }


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
        brief=f.text("brief", d.brief),
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
        variants=f.integer("variants", d.variants, minimum=0),
        variant_weight=f.number("variant_weight", d.variant_weight, minimum=0.0),
        plan=f.integer("plan", d.plan, minimum=0),
        batches=f.integer("batches", d.batches, minimum=0),
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
        "words_ladder": list(TUTOR_WORDS_LADDER),
        "upgrade_steps": list(TUTOR_UPGRADE_STEPS),
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
            client, card, topic=config.topic, level=config.level, words=config.words, threshold=config.threshold,
            count=count, exercises=config.exercises, drills=config.drills, model=config.tutor_model,
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
    ("POST", "/v1/chat/completions", _r_v1_chat_completions,
     "today's format, OpenAI's dialect: {model, messages, max_tokens, temperature, stop, n, stream, stream_options, "
     "tools, thinking, and the dialogue's dials (mode, context, k, explore, avoid_repeats, learn, guard, seed)} -> a "
     "chat.completion whose message carries the thinking as reasoning_content, or chat.completion.chunk events "
     "with stream: true; a token is one unit of the model's encoding"),
    ("POST", "/v1/messages", _r_v1_messages,
     "today's format, Anthropic's dialect: {model, system, messages, max_tokens, stop_sequences, stream, tools, "
     "thinking, and the same dials} -> a message of thinking, text and tool_use blocks, or the message_start / "
     "content_block_* / message_delta / message_stop events with stream: true"),
    ("POST", "/v1/messages/count_tokens", _r_v1_count_tokens,
     "{system, messages} -> {input_tokens}: the units of the model's encoding the request holds"),
    ("GET", "/v1/models", _r_v1_models, "the models in memory as an OpenAI model list: radixnet-<kind>, the active one first"),
    ("GET", "/api/status", _r_status,
     "model stats (with the active kind), current job, backend availability, paths, replay (the model's replay "
     "buffer: {size, texts, seen} or null)"),
    ("GET", "/api/model", _r_model, "the active model kind, every kind (radix | count) and their default files"),
    ("POST", "/api/model/select", _r_model_select,
     "switch the active model kind: {kind: radix | count}; the previous model stays in memory"),
    ("POST", "/api/train", _r_train,
     "start a training job: {texts | text | files, epochs, lr, act_lr, lr_schedule, act_lr_schedule, reverse_schedule, "
     "batch_size, clip, shuffle, auto_compress, checkpoint_every, order: corpus | shortest-first | longest-first | "
     "shuffle, curriculum (the share of the ordered texts the first epoch walks, growing to all; 1 = off), replay "
     "(the share of the run's texts each epoch rehearses from the model's replay buffer; 0 = off), replay_size (the "
     "buffer's capacity from now on; 0 drops it), patience, min_delta (stop after `patience` full epochs without "
     "the loss improving by `min_delta`; 0 = off), reverse (read every text backwards, in the model's units - its "
     "last character or word first - so the model learns what comes before; off by default)}"),
    ("GET", "/api/schedule", _r_schedule, "what a learning-rate schedule expression may use: variables, functions, helpers, presets"),
    ("POST", "/api/schedule/preview", _r_schedule_preview,
     "the rate of every epoch for schedule expressions: {lr_schedule, act_lr_schedule, epochs, lr, act_lr, "
     "reverse_schedule} -> {points}"),
    ("GET", "/api/job", _r_job, "status of the current / last job"),
    ("POST", "/api/job/stop", _r_job_stop, "ask the running job to stop"),
    ("POST", "/api/predict", _r_predict,
     "continue a prefix: {prefix, length, mode: dijkstra | beam | sample, to_end, step_penalty, temperature, max_length "
     "(optional cap; default none), k, beam (beam mode: the top-K and bottom-K continuations), traversal: reward "
     "(default) | punishment (the rewards leave the score and the punishments price every step, so the cheapest "
     "path is the least punished one), penalty_scale, merit_scale (0 = nothing but the punishments decides), "
     "top_k, top_p, min_p (sample mode: keep the k cheapest steps, the nucleus holding p of the mass, the steps at "
     "least min_p as likely as the best; off at 0 / 1 / 0), diversity (beam mode: the K continuations spread out - "
     "a path pays diversity times its overlap with one already picked; off at 0), guard "
     "(default on: the negative network vetoes the continuations it recognises as failures), provenance (false: "
     "the guard reports how many it vetoed, not which or why)}"),
    ("POST", "/api/generate", _r_generate,
     "generate whole texts with the prediction search: {count, max_length, mode: beam (the K most likely) | sample | "
     "dijkstra, temperature, seed, prefix, step_penalty, beam, traversal: reward (default) | punishment, "
     "penalty_scale, merit_scale, top_k, top_p, min_p (sample mode), diversity (beam mode), "
     "guard (default on: the model over-samples and the "
     "negative network vetoes what it recognises as failure), provenance (default: the server's setting; false "
     "reports how many candidates the guard vetoed, not which or why)}"),
    ("POST", "/api/converse", _r_converse,
     "the model converses with itself - each reply is the prediction search picking up the end of the previous "
     "line: {opening, turns, mode: beam | sample, max_length, context, temperature, k, beam, step_penalty, seed, "
     "speakers, history (utterances so far, to continue), partner (another kind in memory answers), avoid_repeats "
     "(what the conversation has heard), avoid_word_repeats (a reply repeating its own words), explore (times a "
     "reply that caught itself repeating may back up and look for another way on; 0 = not at all), learn "
     "(default on: what a rethink finds out is taught to the graph, so the model itself learns where it goes "
     "round - a conversation with this on changes the model), think (default on: a voice that caught itself "
     "repeating thinks about it before it backs up, and the thought rides on the turn's rethink), think_depth "
     "(how deep a thought may question itself), "
     "guard (default on: a reply the negative network vetoes is left unsaid), provenance (false: how many were "
     "vetoed, not which or why)} "
     "-> {..., turns, repeats: the duplicates spoken anyway, to punish}"),
    ("POST", "/api/converse/stream", _r_converse_stream,
     "the same conversation streamed as it happens: the same body, answered as application/x-ndjson - one JSON "
     "object per line, each with event, index and speaker. turn events (turn: the turn as /api/converse writes "
     "it) are the answer and are never taken back; between them is the window a backtrack may still rewrite: "
     "look (from: the context it continues, \"\" for a fresh text), draft (text, cost: what it was about to say), "
     "caught (kind, noticed, cut: what it keeps), backtrack (step, cut, wider), found (text, cost, explored) or "
     "stuck (explored); the last line is {event: done, ...} with the /api/converse document; a failure after the "
     "first line is {event: error, error}"),
    ("POST", "/api/think", _r_think,
     "the model thinks - one thought from the THINK sentinel, in the language of the thoughts it was taught "
     "(POST /api/ollama/think), questioning itself where it has learned to: {about (think at the node where this "
     "text ends), mode: beam | sample, k, beam, max_length, temperature, step_penalty, seed, depth (how deep it may "
     "question itself), questions (per thought), learn (default on: the node is taught to stop and think there - a "
     "thought changes the model)} -> {kind, trigger, at, about, text, stopped: end | length | nothing, then: end | back "
     "| think (what it triggered when it stopped), taught, handed_over, questions: [the same], cost, probability, "
     "labels, node_ids, step_costs}"),
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
     "replace the model with a fresh one: {seed, kind, encoding | unit + ngram + stride, and the kind's "
     "score-function settings - count: count_scale, global_scale, window_scale, reward_scale, window; resonant: "
     "buckets, period, kick_scale, resonance_scale, amp_scale, reward_scale, concentration}.  The encoding is how "
     "text becomes grams and is fixed for the model's life: unit char | word, ngram the n of the n-gram, stride the "
     "units between two grams (1 = sliding window, n = non-overlapping groups); \"encoding\" sets all three "
     "(char:3:1 the default, char:5:5 groups of five letters, word:2:1 word bigrams, word:3:1 word trigrams)"),
    ("GET", "/api/encoding", _r_encoding,
     "the text encoding the active model reads in: {encoding, unit, window (the n of the n-gram), ngram, stride, "
     "overlap, start_label, end_label, back_label, configurable: true (fixed for a model's life, chosen when one "
     "is made - see POST /api/reset), note}"),
    ("POST", "/api/encoding/preview", _r_encoding_preview,
     "one text through the encoder and back: {text} -> the same document plus {chars, windows, count, decoded, "
     "round_trip, kind, path: {known, reason, labels, node_ids, decoded, nodes, compressed}} - path is the text "
     "through the graph's own (possibly merged) node labels"),
    ("POST", "/api/model/weights", _r_model_weights,
     "change the active model's score function - count: {count_scale, global_scale, window_scale, reward_scale, "
     "path_scale, window}; resonant: {buckets, period, kick_scale, resonance_scale, amp_scale, reward_scale, "
     "concentration} -> {weights, stats}"),
    ("GET", "/api/model/attention", _r_attention,
     "the active model's attention band - where inside a gram a correction's blame and credit land: {kind, "
     "attention: {on, blur, weights (the band over one gram, 1 at the centre, 1 - blur at both ends; null while "
     "off), ngram, stride, unit, units, applies (does this kind learn from corrections), default_blur}}"),
    ("POST", "/api/model/attention", _r_attention_set,
     "switch the band: {on, blur} - blur (0..1) alone switches it on, on: true without a blur uses the one it had "
     "(else default_blur), on: false switches it off (each changed unit charged to the step that wrote it); "
     "400 for a kind that is never corrected -> {kind, attention, stats}"),
    ("POST", "/api/model/attention/preview", _r_attention_preview,
     "where one correction would land, gram by gram, and nothing changes: {wrong, right, blur (default the "
     "model's, else default_blur)} -> {kind, attention, blur, weights, changes, wrong, right}, each side {text, "
     "units, grams, spans, writer (the gram writes a changed unit: the rule with the band off), charges (its "
     "share under the band, at most 1), focus (it sees a change most sharply), end (the step into END answers "
     "for the position after the last unit)}"),
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
     "over_sample, threshold, min_coverage, ratio, no_ratio, peak (blame on a single fragment), strict, learn, "
     "provenance (false: verdicts carry the decision and the rule alone)} or {texts} to judge given texts"),
    ("POST", "/api/negative/forget", _r_negative_forget, "drop or fade the blame behind a reason: {reason, factor}"),
    ("POST", "/api/negative/settings", _r_negative_settings,
     "how strictly it judges: {threshold, min_coverage, share_scale, blame_scale, clear_scale, provenance (whether "
     "the guard's vetoes on every answer say why - the rule, the reasons, the blamed fragments - or only how many)}"),
    ("POST", "/api/negative/reset", _r_negative_reset, "forget every failure: {seed} -> a fresh negative network"),
    ("POST", "/api/chat/start", _r_chat_start,
     "start a chat job - an LLM converses with the model and marks every reply: {conversations (0 = until "
     "stopped), turns, topic, opening, persona, context, max_length, mode: beam|sample, k, temperature, "
     "partner_temperature, threshold, provider: ollama|chatgpt, partner_model, judge_model, url, judge_url, "
     "timeout, guard (the negative network vetoes a reply before it is spoken), blame, clear_passes, learn "
     "(2NRL on the marked replies), teach_partner (the partner's own lines join the positive phase), "
     "avoid_repeats, avoid_word_repeats, explore, neg_epochs, pos_epochs, neg_lr, pos_lr, batch_size, strength, "
     "epochs, seed}"),
    ("GET", "/api/chat/history", _r_chat_history,
     "exchange / conversation / report records of all chat runs"),
    ("POST", "/api/negative/auto", _r_negative_auto,
     "the Negative tab, automatic: start a job that has the model write texts, an LLM reviewer mark them and every "
     "failure blame the negative network - {rounds (0 = until stopped), count, prefix, max_length, temperature, "
     "threshold, context (what the texts are meant to be), provider: ollama|chatgpt, reviewer_model, url, timeout, "
     "clear_passes, epochs, seed, correct (letter-level corrections instead of marks: only the characters the editor "
     "changed are blamed), severity (blame per corrected text)}; the positive model is only read from"),
    ("GET", "/api/negative/auto/history", _r_negative_auto_history, "round / report records of all automatic runs"),
    ("POST", "/api/negative/save", _r_negative_save, "save the negative network: {path} (default: beside the model path)"),
    ("GET", "/api/checkpoints", _r_checkpoints, "list checkpoints and the latest pointer"),
    ("POST", "/api/checkpoints/save", _r_checkpoint_save, "write a checkpoint: {tag}"),
    ("POST", "/api/checkpoints/restore", _r_checkpoint_restore, "restore a checkpoint: {name}"),
    ("GET", "/api/graph", _r_graph, "top nodes by visit count and the edges among them (?limit=150)"),
    ("GET", "/api/paths", _r_paths,
     "count model: the judged paths (?limit=50) - what each step did in the context it was taken from: "
     "{totals, path_scale, paths: [{after, parent_label, child_label, seen, correct, incorrect, correct_ratio, "
     "seen_ratio, term}]}"),
    ("GET", "/api/nodes", _r_nodes,
     "count model: each node against the nodes around it (?limit=20, ?node=LABEL) - {nodes: [{node, label, visits, "
     "from: [{label, seen, seen_ratio, reward, reward_ratio, path_seen, path_ratio, correct, incorrect, "
     "correct_ratio}], to: [...], in_totals, out_totals}]}"),
    ("GET", "/api/words", _r_words,
     "word model: its alphabet (?limit=50) - {vocabulary, units, words: [{word, id, trigrams}]}, most read first"),
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
    ("POST", "/api/images/tutor", _r_image_tutor,
     "the recall tutor: ask the network to draw back an image it was shown and mark what comes back "
     "{image bytes | texts, size, encoder, lead, length, attempts, mode, temperature, threshold, blame} -> "
     "lessons with a mark out of 10 and the reason each failure failed"),
    ("POST", "/api/speech/tutor", _r_speech_tutor,
     "the recall tutor: ask the network to say back an utterance it was taught and mark what comes back "
     "{recording bytes | texts, transcript, rate, codec, lead, length, attempts, mode, temperature, "
     "threshold, listen_back, blame} -> lessons with a mark out of 10 and the reason each failure failed"),
    ("POST", "/api/speech/decode", _r_speech_decode,
     "decode an encoded or predicted waveform text back to audio: {text, codec} -> {wav_base64, rate, seconds, repaired}"),
    ("GET", "/api/ollama/models", _r_ollama_models, "models installed in Ollama (?url= overrides the server default); never fails"),
    ("POST", "/api/ollama/corpus", _r_ollama_corpus,
     "training lines from a prompt: {prompt, lines, style: good|garbage, model, url, save_as, train, epochs, lr, batch_size}"),
    ("POST", "/api/ollama/think", _r_ollama_think,
     "a thinking model thinks about a prompt and the network is taught its thinking as thoughts: {prompt, lines "
     "(questions to think about), think: true | false | low | medium | high, temperature, model, url, timeout, "
     "save_as, train (teach the thinking as thoughts that begin at the THINK sentinel, and where a thought "
     "questions itself as a place to stop and think), questions (default on), with_answers (also train the "
     "answers as texts), epochs, lr, batch_size} -> {prompt, model, url, think, count, thinking (how many "
     "thought), thoughts: [{question, thinking, answer}], upload, job}; 202 with the job when training"),
    ("POST", "/api/ollama/review", _r_ollama_review,
     "adversarial LLM review of the model's samples or {texts}: {count, prefix, max_length, threshold, apply: none|2nrl, "
     "good, good_files, blame (teach the negative network what failed and why)}"),
    ("POST", "/api/ollama/correct", _r_ollama_correct,
     "letter-level LLM correction of the model's samples or {texts}: {count, prefix, max_length, temperature, seed, "
     "context, model, url, timeout, blame (blame only the characters the editor changed in the negative network; "
     "the unchanged texts clear blame), severity} -> {corrections: [{text, correction, verdict, reason, note, "
     "changes}], corrected, unchanged, uncorrected, edits, change_rate}"),
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
    ("GET", "/api/tools", _r_tools,
     "the external tools the network can call by writing <tool>name {...}</tool>: names, arguments, JSON schemas"),
    ("POST", "/api/tools/call", _r_tool_call,
     "call one tool directly: {tool, arguments} or {call: 'name {\"arg\": \"value\"}'} + tool overrides "
     "{offline, allow_private, search_url, web_timeout, max_bytes, python_tool, browser} -> the ToolResult"),
    ("POST", "/api/agent/start", _r_agent_start,
     "start an agent job over tasks: {tasks | tasks_text | task_files, phase: model|teacher|both, rounds, max_steps, "
     "mediation: repair|always|never, criteria, judge, teach, blame (teach the negative network from the failures), "
     "blatant_mode, blatant_margin, blatant_boost, 2NRL options}"),
    ("POST", "/api/agent/explore", _r_agent_explore,
     "start an explore job - the network chooses every task itself: {steps (0 = until stopped), seed_urls, blame, "
     "...the agent options}"),
    ("GET", "/api/agent/history", _r_agent_history, "criteria / step / attempt / task records of all agent and explore runs"),
    ("POST", "/api/agent/criteria", _r_agent_criteria,
     "the acceptance criteria the LLM writes for {tasks}, without attempting anything"),
    ("POST", "/api/agent/solve", _r_agent_solve,
     "one task through the loop without training: {task, source: model|teacher, ...} -> the transcript, the verdict "
     "and how badly it failed"),
    ("POST", "/api/tutor/start", _r_tutor_start,
     "start a tutor job - the teacher writes the prefixes, the network completes them, the teacher marks the "
     "grammar and the 2NRL follows: {topic, rounds, exercises, attempts, focus, level, mode, threshold, "
     "grammar_weight, drills, adapt, twonrl_per: round|lesson, diff_corrections, keep_weight, min_weight, "
     "neg_epochs, pos_epochs, neg_lr, pos_lr, brief, plan, batches (auto run: each batch planned from the last, "
     "0 = until stopped), tutor_provider: ollama|chatgpt, tutor_model, grader_provider, "
     "grader_model, url, grader_url, blame (every failed sentence also teaches the negative network why it "
     "failed, and the teacher is asked why it is wrong and for 'variants' more sentences with the same "
     "mistake, blamed at 'variant_weight' of its severity), ...}"),
    ("GET", "/api/tutor/history", _r_tutor_history, "lesson / round / report records of all tutor runs"),
    ("POST", "/api/tutor/lesson", _r_tutor_lesson,
     "one round of lessons without training: {topic, exercises, prefixes (skip the LLM and use these), attempts, "
     "threshold, ...} -> completions with grades (grammar, spelling, fluency, error, correction) and a report card"),
    ("POST", "/api/tutor/plan", _r_tutor_plan,
     "the lesson plan a report card implies: {report (default: the card at the end of the last run), count, topic, "
     "level, words, threshold, exercises, drills, tutor_provider, tutor_model, url} -> {plan: {summary, prompt "
     "(the brief for the next batch: start a run with it as 'brief'), upgrade: {step, level, words, threshold, "
     "drills, note}, level, weak, targets, lessons: [{focus, targets, topic, why, exercises, drills, prefixes}]}, "
     "source}"),
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
        if path == "/v1" or path.startswith("/v1/"):
            return self._handle_api(method, path.rstrip("/") or "/v1", query, body)
        if method in ("GET", "HEAD"):
            return self._handle_static(method, path)
        return self._send_json(405, {"error": f"method {method} is not allowed for {path}"}, method, allow="GET, HEAD, OPTIONS")

    # -- API -----------------------------------------------------------------

    def _handle_api(self, method: str, path: str, query: str, body: bytes) -> int:
        dialect = _v1_dialect(path)  # a /v1 route answers its errors in its dialect's envelope
        route = _ROUTES.get(path)
        if route is None:
            return self._send_json(404, _shape_error(dialect, 404, f"unknown API endpoint {path}"), method)
        lookup = "GET" if method == "HEAD" else method
        fn = route.get(lookup)
        if fn is None:
            allow = ", ".join(sorted(route) + ["OPTIONS"])
            message = f"method {method} is not allowed for {path}; use {allow}"
            return self._send_json(405, _shape_error(dialect, 405, message), method, allow=allow)
        service = self.server.service
        param: str | None = None
        try:
            params = parse_qs(query, keep_blank_values=True)
            if lookup == "POST" and path in _BINARY_ROUTES:
                fields = Fields(_upload_body(body, self.headers.get("Content-Type", ""), params, _BINARY_ROUTES[path]))
            else:
                fields = Fields(parse_body(body) if lookup == "POST" else {})
            status, payload = fn(service, fields, params)
        except ApiError as exc:
            status, payload, param = exc.status, {"error": exc.message}, exc.param
        except FileNotFoundError as exc:
            status, payload = 404, {"error": _os_error_text(exc)}
        except (ValueError, TypeError) as exc:
            status, payload = 400, {"error": str(exc) or type(exc).__name__}
        except OSError as exc:
            status, payload = 400, {"error": _os_error_text(exc)}
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged here
            self._log_exception(f"{method} {path} failed")
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(payload, EventStream):
            return self._send_sse(status, payload, method)
        if isinstance(payload, StreamedResponse):
            return self._send_stream(payload, method, path)
        if dialect is not None and status >= 400 and isinstance(payload, dict) and isinstance(payload.get("error"), str):
            payload = _shape_error(dialect, status, payload["error"], param)
        return self._send_json(status, payload, method)

    def _send_sse(self, status: int, stream: EventStream, method: str) -> int:
        """Write a streamed answer: the headers, then every frame as it is produced, then close the connection.

        No ``Content-Length`` can be known in advance, so the response is
        delimited by closing the connection (``Connection: close``), which
        every client of these formats handles.  A failure after the headers
        went out is written as the dialect's error frame.
        """
        self.send_response(status)
        for name, value in _CORS_HEADERS:
            self.send_header(name, value)
        self.send_header("Content-Type", stream.content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        if method == "HEAD":
            return status

        def write(frame: str) -> None:
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        try:
            stream.run(write)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # the client went away; nothing to tell it
        except ApiError as exc:
            with contextlib.suppress(OSError):
                write(stream.error(exc.message))
        except Exception as exc:  # noqa: BLE001 - reported to the client in the stream, logged here
            self._log_exception("streaming failed")
            with contextlib.suppress(OSError):
                write(stream.error(f"{type(exc).__name__}: {exc}"))
        return status

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

    def _send_stream(self, streamed: StreamedResponse, method: str, path: str) -> int:
        """Write a :class:`StreamedResponse`: chunked ``application/x-ndjson``, one event per line, each flushed
        as it is written.  The headers wait for the first event, so a request refused before anything was
        streamed still gets its 4xx JSON; a failure after that is the stream's last event."""
        started = False

        def write(event: Any) -> None:
            nonlocal started
            try:
                line = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            except (TypeError, ValueError) as exc:
                line = json.dumps({"event": "error", "error": f"event is not JSON-serialisable: {exc}"}) + "\n"
            data = line.encode("utf-8")
            if not started:
                started = True
                self.send_response(200)
                for name, value in _CORS_HEADERS:
                    self.send_header(name, value)
                self.send_header("Content-Type", _NDJSON_CONTENT_TYPE)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        try:
            streamed.run(write)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True  # the client went away mid-conversation
            return 200
        except ApiError as exc:
            if not started:
                return self._send_json(exc.status, {"error": exc.message}, method)
            write({"event": "error", "error": exc.message})
        except (ValueError, TypeError) as exc:
            if not started:
                return self._send_json(400, {"error": str(exc) or type(exc).__name__}, method)
            write({"event": "error", "error": str(exc) or type(exc).__name__})
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged here
            self._log_exception(f"{method} {path} failed")
            if not started:
                return self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"}, method)
            write({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        if not started:
            return self._send(200, b"", _NDJSON_CONTENT_TYPE, method)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return 200

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
    tool_options: dict | None = None,
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
    from the server's ``$OPENAI_API_KEY``); ``tool_options`` are the defaults
    of the ``/api/tools`` and ``/api/agent/*`` endpoints (``offline``,
    ``allow_private``, ``search_url``, ``web_timeout``, ``max_bytes``,
    ``python_tool``, ``sandbox_timeout``).  ``quiet`` silences the per-request
    log lines (stderr).
    """
    service = ModelService(
        model_path=model_path, checkpoint_dir=checkpoint_dir, backend=backend, device=device,
        seed=seed, quiet=quiet, upload_dir=upload_dir, ollama_url=ollama_url, ollama_model=ollama_model,
        chatgpt_url=chatgpt_url, chatgpt_model=chatgpt_model, kind=kind, tool_options=tool_options,
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
    tool_options: dict | None = None,
) -> None:
    """Serve until ``KeyboardInterrupt``; a running job is stopped on the way out."""
    server, service = create_server(
        host, port, model_path=model_path, checkpoint_dir=checkpoint_dir, frontend_dir=frontend_dir,
        backend=backend, device=device, seed=seed, quiet=quiet, upload_dir=upload_dir,
        ollama_url=ollama_url, ollama_model=ollama_model, chatgpt_url=chatgpt_url, chatgpt_model=chatgpt_model,
        kind=kind, tool_options=tool_options,
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
