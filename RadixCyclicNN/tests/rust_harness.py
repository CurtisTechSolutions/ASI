"""What every Rust parity suite shares: the binary, built once, and the two CLIs.

``tests/test_rust_parity.py`` holds the model itself to Python's; each area of
the port that came after it (the negative network's routes, the LLM clients,
the tools, images and speech, the teaching loops, ...) has its own
``tests/test_rust_parity_<area>.py`` built on this module, so the areas can be
tested - and ported - one at a time.  ``make rust-parity`` runs them all.

    from tests.rust_harness import build, rust, py, serve

    def setUpModule():
        build()          # raises SkipTest without a Rust toolchain
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUST_DIR = os.path.join(ROOT, "rust")
CORPUS = os.path.join(ROOT, "data", "sample_corpus.txt")
GARBAGE = os.path.join(ROOT, "data", "sample_garbage.txt")
CARGO = shutil.which("cargo")

_BINARY: str | None = None
_SKIP: str | None = None
_TMP: tempfile.TemporaryDirectory | None = None


def build() -> str:
    """The release binary, built the first time any suite asks; SkipTest without cargo."""
    global _BINARY, _SKIP
    if _BINARY is not None:
        return _BINARY
    if _SKIP is not None:
        raise unittest.SkipTest(_SKIP)
    if CARGO is None:
        _SKIP = "no Rust toolchain on PATH"
        raise unittest.SkipTest(_SKIP)
    proc = subprocess.run(
        [CARGO, "build", "--release", "--bin", "radixnet"],
        cwd=RUST_DIR, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        _SKIP = f"cargo build failed:\n{proc.stderr[-2000:]}"
        raise unittest.SkipTest(_SKIP)
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(RUST_DIR, "target"))
    binary = os.path.join(target, "release", "radixnet")
    if not os.path.exists(binary):
        _SKIP = f"no binary at {binary}"
        raise unittest.SkipTest(_SKIP)
    _BINARY = binary
    return binary


def tmpdir() -> str:
    """A scratch directory that lives as long as the test run."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory(prefix="radixnet-rust-")
    return _TMP.name


def _run(cmd: list[str], expect: int, env: dict | None, stdin: str | None, timeout: int) -> dict | None:
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env, input=stdin,
    )
    if proc.returncode != expect:
        raise AssertionError(
            f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}"
            f"\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    if proc.stdout.strip():
        return json.loads(proc.stdout)
    return {"error": proc.stderr.strip()} if expect else None


def rust(*args, model: str, expect: int = 0, env: dict | None = None, stdin: str | None = None,
         timeout: int = 600) -> dict | None:
    """``radixnet --json --model MODEL ARGS...`` from the Rust port, as one JSON document."""
    cmd = [build(), "--json", "--model", model, *[str(a) for a in args]]
    return _run(cmd, expect, env, stdin, timeout)


def py(*args, model: str, expect: int = 0, env: dict | None = None, stdin: str | None = None,
       timeout: int = 900) -> dict | None:
    """``python -m radixnet --json --model MODEL ARGS...``, as one JSON document."""
    cmd = [sys.executable, "-m", "radixnet", "--json", "--model", model, *[str(a) for a in args]]
    out = _run(cmd, expect, env, stdin, timeout)
    return out if out is not None else {"error": ""}


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def assert_close(test: unittest.TestCase, a, b, tol: float = 1e-9) -> None:
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        test.assertLessEqual(abs(x - y), tol, (x, y))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """``radixnet serve`` on a free port, with the client the API tests use."""

    def __init__(self, model: str, *extra: str, env: dict | None = None):
        from tests.test_api import Client

        self.port = free_port()
        self.model = model
        self.proc = subprocess.Popen(
            [build(), "--model", model, *extra, "serve", "--host", "127.0.0.1", "--port", str(self.port),
             "--frontend-dir", os.path.join(tmpdir(), "no-frontend")],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        self.client = Client(f"http://127.0.0.1:{self.port}")
        for _ in range(400):
            try:
                if self.client.get("/api/health")[0] == 200:
                    return
            except Exception:  # noqa: BLE001 - the server is still starting
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.05)
        err = self.close()
        raise AssertionError(f"the Rust server did not start:\n{err}")

    def get(self, path: str):
        return self.client.get(path)

    def post(self, path: str, body: dict | None = None):
        return self.client.post(path, body if body is not None else {})

    def wait_job(self, timeout: float = 120.0) -> dict:
        """Polls ``/api/job`` until the job is no longer running; returns it."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, job, _ = self.get("/api/job")
            if not job or job.get("state") != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("the job did not finish")

    def close(self) -> str:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return (self.proc.stderr.read() if self.proc.stderr else "")[-4000:]


def serve(test_case, model: str, *extra: str, env: dict | None = None) -> Server:
    """A server that is stopped when ``test_case`` (a TestCase or its class) is done."""
    server = Server(model, *extra, env=env)
    if isinstance(test_case, type):
        test_case.addClassCleanup(server.close)
    else:
        test_case.addCleanup(server.close)
    return server
