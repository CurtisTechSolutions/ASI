"""A small HTTP API, and the static files of the browser front end.

``python -m audioimage serve`` puts the whole tool behind a page: load a clip,
watch it become a plane, play it with the playhead running across the picture,
paint on the plane and hear what you painted.

The server is :mod:`http.server` from the standard library - the package still
has no dependencies - and every route does exactly what the matching command
does, so what the page shows is what the format is.

Routes
------
``GET  /api/info``        what this build supports, and the defaults the page builds its controls from
``POST /api/tone``        settings -> ``audio/wav``
``POST /api/encode``      a WAV body -> ``image/png``, the picture
``POST /api/decode``      a PNG body -> ``audio/wav``, the sound it holds
``POST /api/view``        a WAV or PNG body -> ``image/png``, the labelled chart
``POST /api/roundtrip``   a WAV body -> JSON: both artefacts and what survived

Binary routes answer with the file itself and put their report in the
``X-AudioImage-Meta`` header as base64 JSON, so the browser can hand the body
straight to an ``<audio>`` element or an ``<img>`` without unpacking anything.

The listening socket defaults to 127.0.0.1.  Serving this on a public interface
would hand anyone who can reach it a way to spend your CPU on Griffin-Lim, so
``--host`` has to be given deliberately.
"""

from __future__ import annotations

import base64
import json
import os
import posixpath
import socket
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .backend import BACKENDS, describe_backends
from .codec import PHASE_MODES, EncodeConfig, compare, decode, encode, image_to_plane, read_meta
from .colormap import COLORMAPS
from .dsp import WINDOWS
from .png import Image, PngError, SIGNATURE, read_png_bytes, write_png_bytes
from .render import ViewConfig, render_view, render_waveform
from .spectrogram import FREQ_SCALES, ORIGINS, SCALES, PlaneConfig
from .synth import KINDS, generate
from .wav import Audio, WavError, read_wav_bytes, write_wav_bytes

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "ServerError", "frontend_dir", "make_server", "run_server"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_BODY = 256 * 1024 * 1024
"""Refuse a request body larger than this (a quarter of a gigabyte of WAV)."""
MAX_SECONDS = 600.0
"""Refuse a clip longer than ten minutes: the phase rebuild would run for hours."""
META_HEADER = "X-AudioImage-Meta"

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".wav": "audio/wav",
    ".map": "application/json",
    ".woff2": "font/woff2",
}


class ServerError(Exception):
    """A bad request: turned into a JSON error with a status code."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


def frontend_dir(path: str | None = None) -> str:
    """Where the page lives: ``path``, else ``frontend/`` beside the package."""
    if path:
        return os.path.abspath(path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


# ---------------------------------------------------------------------------
# reading the query string
# ---------------------------------------------------------------------------


class Params:
    """The query string, with the type conversions every route needs."""

    __slots__ = ("raw",)

    def __init__(self, query: str) -> None:
        self.raw = parse_qs(query, keep_blank_values=True)

    def text(self, name: str, default: str = "") -> str:
        values = self.raw.get(name)
        return values[0] if values and values[0] != "" else default

    def choice(self, name: str, options: "tuple[str, ...]", default: str) -> str:
        value = self.text(name, default)
        if value not in options:
            raise ServerError(f"{name} must be one of {', '.join(options)}; got {value!r}")
        return value

    def number(self, name: str, default: float, low: float | None = None, high: float | None = None) -> float:
        value = self.text(name)
        if value == "":
            return default
        try:
            out = float(value)
        except ValueError:
            raise ServerError(f"{name} must be a number; got {value!r}") from None
        if out != out or out in (float("inf"), float("-inf")):
            raise ServerError(f"{name} must be a finite number; got {value!r}")
        if low is not None and out < low:
            raise ServerError(f"{name} must be at least {low}; got {out}")
        if high is not None and out > high:
            raise ServerError(f"{name} must be at most {high}; got {out}")
        return out

    def integer(self, name: str, default: int, low: int | None = None, high: int | None = None) -> int:
        return int(self.number(name, float(default), low, high))

    def flag(self, name: str, default: bool = False) -> bool:
        value = self.text(name, "1" if default else "0").lower()
        return value in ("1", "true", "yes", "on")

    def optional_int(self, name: str, low: int | None = None, high: int | None = None) -> int | None:
        return None if self.text(name) == "" else self.integer(name, 0, low, high)

    def optional_number(self, name: str, low: float | None = None, high: float | None = None) -> float | None:
        return None if self.text(name) == "" else self.number(name, 0.0, low, high)


def _encode_config(params: Params) -> EncodeConfig:
    """Build the encode settings from a query string, validating as it goes."""
    n_fft = params.integer("n_fft", 1024, 2, 65536)
    if n_fft & (n_fft - 1):
        raise ServerError(f"n_fft must be a power of two; got {n_fft}")
    try:
        return EncodeConfig(
            n_fft=n_fft,
            hop=params.integer("hop", max(1, n_fft // 4), 1, n_fft),
            window=params.choice("window", WINDOWS, "hann"),
            center=params.flag("center", True),
            colormap=params.choice("colormap", COLORMAPS, "gray"),
            depth=params.integer("depth", 16),
            phase="rgb" if params.flag("lossless") else params.choice("phase", PHASE_MODES, "none"),
            phase_floor=params.number("phase_floor", 60.0, 1.0, 400.0),
            lossless=params.flag("lossless"),
            lossless_bits=params.integer("lossless_bits", 16),
            plane=PlaneConfig(
                scale=params.choice("scale", SCALES, "db"),
                top_db=params.number("top_db", 80.0, 1.0, 400.0),
                ref=params.optional_number("ref", 1e-12),
                freq_scale=params.choice("freq_scale", FREQ_SCALES, "linear"),
                f_min=params.number("f_min", 0.0, 0.0),
                f_max=params.optional_number("f_max", 0.0),
                height=params.optional_int("height", 1, 8192),
                width=params.optional_int("width", 1, 16384),
                origin=params.choice("origin", ORIGINS, "lower"),
            ),
        )
    except ValueError as exc:  # the dataclasses validate the combinations
        raise ServerError(str(exc)) from None


def _read_audio(body: bytes) -> Audio:
    if not body:
        raise ServerError("no audio was sent")
    try:
        audio = read_wav_bytes(body, source="upload")
    except WavError as exc:
        raise ServerError(f"that is not a WAV this tool can read: {exc}") from None
    if not audio.samples:
        raise ServerError("that clip has no samples in it")
    if audio.duration > MAX_SECONDS:
        raise ServerError(f"that clip is {audio.duration:.0f}s; the limit is {MAX_SECONDS:.0f}s")
    return audio


def _read_image(body: bytes) -> Image:
    if not body:
        raise ServerError("no picture was sent")
    try:
        return read_png_bytes(body)
    except PngError as exc:
        raise ServerError(f"that is not a PNG this tool can read: {exc}") from None


# ---------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------


class Api:
    """The routes, kept apart from the HTTP plumbing so they can be tested directly."""

    def __init__(self, backend: str = "auto") -> None:
        self.backend = backend

    # -- GET ---------------------------------------------------------------

    def info(self, params: Params) -> dict[str, Any]:
        """Everything the page needs to build its controls."""
        defaults = EncodeConfig()
        return {
            "tool": "audioimage",
            "version": __version__,
            "backend": describe_backends(),
            "limits": {"max_body": MAX_BODY, "max_seconds": MAX_SECONDS},
            "options": {
                "windows": list(WINDOWS),
                "scales": list(SCALES),
                "freq_scales": list(FREQ_SCALES),
                "colormaps": list(COLORMAPS),
                "origins": list(ORIGINS),
                "phase_modes": list(PHASE_MODES),
                "backends": list(BACKENDS),
                "kinds": list(KINDS),
                "depths": [8, 16],
            },
            "defaults": {
                "n_fft": defaults.n_fft,
                "hop": defaults.hop,
                "window": defaults.window,
                "scale": defaults.plane.scale,
                "top_db": defaults.plane.top_db,
                "freq_scale": defaults.plane.freq_scale,
                "origin": defaults.plane.origin,
                "colormap": defaults.colormap,
                "depth": defaults.depth,
                "phase": defaults.phase,
                "phase_floor": defaults.phase_floor,
                "lossless": defaults.lossless,
                "iters": 64,
                "momentum": 0.99,
                "rate": 22050,
            },
        }

    # -- POST --------------------------------------------------------------

    def tone(self, params: Params, body: bytes) -> "tuple[bytes, str, dict[str, Any]]":
        """Settings -> a WAV."""
        notes = tuple(n for n in params.text("notes").replace(",", " ").split() if n)
        extra: dict[str, Any] = {
            "amplitude": params.number("amplitude", 0.6, 0.0, 1.0),
            "noise": params.choice("noise", ("white", "pink"), "white"),
            "method": params.choice("method", ("linear", "log"), "linear"),
            "seed": params.integer("seed", 0),
        }
        if notes:
            extra["notes"] = notes
        if params.text("f1"):
            extra["f1"] = params.number("f1", 4000.0, 1.0)
        try:
            audio = generate(
                params.choice("kind", KINDS, "melody"),
                duration=params.number("duration", 2.0, 0.01, MAX_SECONDS),
                rate=params.integer("rate", 22050, 1000, 192000),
                freq=params.number("freq", 440.0, 0.1),
                **extra,
            )
        except ValueError as exc:
            raise ServerError(str(exc)) from None
        return write_wav_bytes(audio), "audio/wav", audio.describe()

    def encode(self, params: Params, body: bytes) -> "tuple[bytes, str, dict[str, Any]]":
        """A WAV -> the picture of its spectrum."""
        audio = _read_audio(body)
        cfg = _encode_config(params)
        started = time.monotonic()
        image = encode(audio, cfg, backend=self.backend)
        blob = write_png_bytes(image)
        return (
            blob,
            "image/png",
            {
                "meta": read_meta(image),
                "image": image.describe(),
                "audio": audio.describe(),
                "seconds": round(time.monotonic() - started, 3),
                "bytes": len(blob),
            },
        )

    def decode(self, params: Params, body: bytes) -> "tuple[bytes, str, dict[str, Any]]":
        """A picture -> the sound it holds."""
        image = _read_image(body)
        cfg = _encode_config(params)
        started = time.monotonic()
        result = decode(
            image,
            fallback=cfg,
            sample_rate=params.optional_int("rate", 1000, 192000),
            iterations=params.integer("iters", 64, 0, 2000),
            momentum=params.number("momentum", 0.99, 0.0, 1.0),
            seed=params.integer("seed", 0),
            init=params.choice("init", ("random", "zeros"), "random"),
            backend=self.backend,
            level=params.choice("level", ("auto", "peak", "none"), "auto"),
        )
        blob = write_wav_bytes(result.audio, bits=params.integer("bits", 16))
        return (
            blob,
            "audio/wav",
            {
                **result.describe(),
                "seconds": round(time.monotonic() - started, 3),
                "meta": result.meta,
                "bytes": len(blob),
            },
        )

    def view(self, params: Params, body: bytes) -> "tuple[bytes, str, dict[str, Any]]":
        """A WAV or a picture -> the labelled chart of it."""
        cfg = _encode_config(params)
        if body[:8] == SIGNATURE:
            image = _read_image(body)
            meta = read_meta(image)
            plane_cfg = PlaneConfig.from_meta(meta) if meta else cfg.plane
            plane = image_to_plane(image, str(meta.get("colormap", "gray")))
            rate = params.optional_int("rate", 1000, 192000) or int(meta.get("sample_rate", 0)) or 22050
            hop = int(meta.get("hop", cfg.hop))
        else:
            audio = _read_audio(body)
            image = encode(audio, cfg, backend=self.backend)
            plane = image_to_plane(image, cfg.colormap)
            plane_cfg = PlaneConfig.from_meta(read_meta(image))
            rate, hop = audio.sample_rate, cfg.hop
        view = ViewConfig(
            colormap=params.choice("view_colormap", COLORMAPS, "magma"),
            title=params.text("title"),
            axes=params.flag("axes", True),
            legend=params.flag("legend", True),
            scale=params.integer("zoom", 1, 1, 8),
            width=params.optional_int("view_width", 1, 4096),
            height=params.optional_int("view_height", 1, 4096),
        )
        picture = render_view(plane, rate, hop, plane_cfg, view)
        return write_png_bytes(picture), "image/png", {"image": picture.describe()}

    def waveform(self, params: Params, body: bytes) -> "tuple[bytes, str, dict[str, Any]]":
        """A WAV -> a plot of its samples."""
        audio = _read_audio(body)
        picture = render_waveform(
            audio,
            params.integer("view_width", 900, 16, 4096),
            params.integer("view_height", 220, 16, 2048),
            ViewConfig(colormap="gray", title=params.text("title"), axes=params.flag("axes", True), legend=False),
        )
        return write_png_bytes(picture), "image/png", {"image": picture.describe()}

    def roundtrip(self, params: Params, body: bytes) -> dict[str, Any]:
        """A WAV -> the picture, the sound that comes back, and what survived."""
        audio = _read_audio(body)
        cfg = _encode_config(params)
        started = time.monotonic()
        image = encode(audio, cfg, backend=self.backend)
        encode_seconds = time.monotonic() - started
        png = write_png_bytes(image)

        started = time.monotonic()
        result = decode(
            image,
            iterations=params.integer("iters", 64, 0, 2000),
            momentum=params.number("momentum", 0.99, 0.0, 1.0),
            seed=params.integer("seed", 0),
            backend=self.backend,
            level=params.choice("level", ("auto", "peak", "none"), "auto"),
        )
        decode_seconds = time.monotonic() - started
        wav = write_wav_bytes(result.audio)
        return {
            "metrics": compare(audio, result.audio, cfg, backend=self.backend),
            "meta": read_meta(image),
            "encode_seconds": round(encode_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "iterations": result.iterations,
            "griffin_lim_error": round(result.error, 6),
            "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "audio": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"),
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def make_handler(api: Api, directory: str, quiet: bool = False) -> "type[BaseHTTPRequestHandler]":
    """Build the request handler bound to one API and one directory of files."""

    binary_routes: dict[str, Callable[[Params, bytes], "tuple[bytes, str, dict[str, Any]]"]] = {
        "/api/tone": api.tone,
        "/api/encode": api.encode,
        "/api/decode": api.decode,
        "/api/view": api.view,
        "/api/waveform": api.waveform,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = f"audioimage/{__version__}"
        protocol_version = "HTTP/1.1"

        # -- helpers -------------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:
            if not quiet:
                sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

        def _send(self, status: int, body: bytes, content_type: str, extra: "dict[str, str] | None" = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # the page is same-origin; these are the headers that keep a stray
            # script or a framing attempt from getting at it anyway
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=float).encode("utf-8")
            self._send(status, body, "application/json")

        def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
            self._json({"error": message}, status)

        def _body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ServerError("Content-Length is not a number") from None
            if length < 0:
                raise ServerError("Content-Length is negative")
            if length > MAX_BODY:
                raise ServerError(f"that is {length} bytes; the limit is {MAX_BODY}", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return self.rfile.read(length) if length else b""

        def _static(self, path: str) -> None:
            """Serve a file from the front-end directory, and nothing outside it."""
            clean = posixpath.normpath(unquote(path))
            if clean in ("/", "", "."):
                clean = "/index.html"
            parts = [p for p in clean.split("/") if p not in ("", ".", "..")]
            target = os.path.join(directory, *parts)
            root = os.path.realpath(directory)
            real = os.path.realpath(target)
            if real != root and not real.startswith(root + os.sep):
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            if not os.path.isfile(real):
                if not os.path.isdir(root):
                    self._error(
                        f"the front end is not where the server expected it ({directory})", HTTPStatus.NOT_FOUND
                    )
                else:
                    self._error("not found", HTTPStatus.NOT_FOUND)
                return
            try:
                with open(real, "rb") as fh:
                    blob = fh.read()
            except OSError as exc:
                self._error(f"cannot read that file: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            kind = _TYPES.get(os.path.splitext(real)[1].lower(), "application/octet-stream")
            self._send(HTTPStatus.OK, blob, kind)

        # -- verbs ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - the name http.server requires
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/info":
                    self._json(api.info(Params(parsed.query)))
                elif parsed.path.startswith("/api/"):
                    self._error(f"no such route: {parsed.path}", HTTPStatus.NOT_FOUND)
                else:
                    self._static(parsed.path)
            except ServerError as exc:
                self._error(str(exc), exc.status)
            except Exception as exc:  # pragma: no cover - a bug, not a bad request
                traceback.print_exc()
                self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                params = Params(parsed.query)
                body = self._body()
                if parsed.path == "/api/roundtrip":
                    self._json(api.roundtrip(params, body))
                    return
                route = binary_routes.get(parsed.path)
                if route is None:
                    self._error(f"no such route: {parsed.path}", HTTPStatus.NOT_FOUND)
                    return
                blob, kind, report = route(params, body)
                meta = base64.b64encode(json.dumps(report, default=float).encode("utf-8")).decode("ascii")
                self._send(HTTPStatus.OK, blob, kind, {META_HEADER: meta})
            except ServerError as exc:
                self._error(str(exc), exc.status)
            except (ValueError, TypeError, ArithmeticError, LookupError) as exc:
                self._error(f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # pragma: no cover - a bug, not a bad request
                traceback.print_exc()
                self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    directory: str | None = None,
    backend: str = "auto",
    quiet: bool = False,
) -> _Server:
    """Build the server without starting it (the tests bind to port 0)."""
    handler = make_handler(Api(backend), frontend_dir(directory), quiet)
    try:
        return _Server((host, port), handler)
    except OSError as exc:
        raise ServerError(f"cannot listen on {host}:{port}: {exc}") from None


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    directory: str | None = None,
    backend: str = "auto",
    quiet: bool = False,
    ready: "Callable[[_Server], None] | None" = None,
) -> None:
    """Serve until interrupted."""
    httpd = make_server(host, port, directory, backend, quiet)
    if ready is not None:
        ready(httpd)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()


def _unused() -> None:  # pragma: no cover
    _ = (socket, threading)
