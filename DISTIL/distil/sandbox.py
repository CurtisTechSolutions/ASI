"""Running code the system wrote, without running it in the system.

This package generates Python and then executes it. That is the point -- a tool
that has not been run is a guess -- and it is also the sharpest edge in the
design, so the containment is stated plainly rather than implied:

**What this does.** A separate interpreter process, a working directory that is a
fresh temp tree, a wall-clock timeout with `kill()` on expiry, address-space and
CPU rlimits where the platform has them, no inherited proxy or API-key
environment, and an AST screen (`screen()`) that refuses obviously destructive
source *before* it runs.

**What this is not.** It is not a security boundary. A subprocess shares the
kernel, the filesystem and the network namespace with its parent. Determined
code escapes an AST denylist trivially -- `getattr(__builtins__, 'ev' + 'al')`
defeats it -- and the screen is there to catch an LLM that cheerfully wrote
`shutil.rmtree('/')` because the docstring said "clean up", which is the failure
that actually happens.

For a real boundary, run the whole package in a container or a VM with no
network and a read-only mount. That is a deployment decision, not a library one,
and pretending a `setrlimit` call substitutes for it would be the dishonest
version of this file.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

#: Calls and attributes the screen refuses outright.
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
FORBIDDEN_ATTRS = {("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
                   ("os", "rmdir"), ("os", "kill"), ("shutil", "rmtree"), ("shutil", "move"),
                   ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "Popen"),
                   ("subprocess", "check_output"), ("sys", "exit")}
FORBIDDEN_IMPORTS = {"socket", "http", "urllib", "requests", "ftplib", "smtplib",
                     "telnetlib", "ctypes", "multiprocessing", "webbrowser"}
#: Allowed despite living in a forbidden module: writing inside the sandbox cwd
#: is the whole point of a tool that produces a file.
ALLOWED_ATTRS = {("os", "getcwd"), ("os", "listdir"), ("os", "makedirs"), ("os", "path")}


@dataclass
class Screening:
    ok: bool
    findings: list[str]

    def __bool__(self) -> bool:
        return self.ok


def screen(source: str) -> Screening:
    """Static refusal pass. Parses, then walks.

    A syntax error is a finding rather than an exception: unparseable source is
    exactly what a generator produces on a bad day, and the caller wants it as a
    grade, not as a traceback.
    """
    findings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Screening(False, [f"syntax error line {exc.lineno}: {exc.msg}"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                findings.append(f"forbidden call: {fn.id}() at line {node.lineno}")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                pair = (fn.value.id, fn.attr)
                if pair in FORBIDDEN_ATTRS and pair not in ALLOWED_ATTRS:
                    findings.append(f"forbidden call: {pair[0]}.{pair[1]}() at line {node.lineno}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    findings.append(f"forbidden import: {name} at line {node.lineno}")
    return Screening(not findings, findings)


@dataclass
class Execution:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    screened_out: bool = False
    findings: list[str] = None

    @property
    def diagnostic(self) -> str:
        """The single most informative line of what went wrong.

        The last line of a traceback is the exception; everything above it is the
        path taken to reach it. For grading and for memory, the exception is the
        signal -- `GREN/DESIGN.md` 15 makes the same argument about compiler
        diagnostics, and the reasoning is identical: a structured error string is
        a class label, not prose.
        """
        if self.screened_out:
            return "; ".join(self.findings or ["screened out"])
        if self.timed_out:
            return f"timed out after {self.seconds:.1f}s"
        if self.ok:
            return "ok"
        tail = [l for l in (self.stderr or "").strip().splitlines() if l.strip()]
        return tail[-1] if tail else f"exit {self.returncode}"


def _limits(memory_mb: int, cpu_seconds: int):
    """rlimits in the child, where the platform supports them. `resource` is
    absent on Windows and some hardened builds, so this degrades to the timeout
    alone rather than refusing to run."""
    try:
        import resource
    except ImportError:
        return None

    def apply():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
        except (ValueError, OSError):
            pass            # a limit the platform will not set is not worth dying over
    return apply


def _child_env() -> dict:
    """A deliberately impoverished environment.

    Credentials are stripped so generated code cannot spend the operator's API
    budget or exfiltrate a key, and the proxy variables go with them so a tool
    that tries to reach the network fails immediately instead of hanging until
    the timeout.
    """
    keep = {"PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"          # determinism: same source, same behaviour
    env["PYTHONIOENCODING"] = "utf-8"
    env["DISTIL_SANDBOX"] = "1"          # so generated code can tell where it is
    return env


def run_source(source: str, argv: list[str] | None = None, timeout: float = 10.0,
               memory_mb: int = 512, apply_screen: bool = True,
               files: dict[str, str] | None = None,
               extra_env: dict | None = None) -> Execution:
    """Write `source` to a fresh temp tree and run it. Always cleans up."""
    scr = screen(source) if apply_screen else Screening(True, [])
    if not scr.ok:
        return Execution(False, -1, "", "; ".join(scr.findings), 0.0,
                         screened_out=True, findings=scr.findings)
    workdir = Path(tempfile.mkdtemp(prefix="distil-run-"))
    try:
        for name, body in (files or {}).items():
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        script = workdir / "_main.py"
        script.write_text(source)
        return run_file(script, argv, timeout, memory_mb, cwd=workdir, extra_env=extra_env)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_file(path: Path, argv: list[str] | None = None, timeout: float = 10.0,
             memory_mb: int = 512, cwd: Path | None = None,
             extra_env: dict | None = None) -> Execution:
    started = time.monotonic()
    env = _child_env()
    env.update(extra_env or {})
    proc = subprocess.Popen(
        [sys.executable, "-I", str(path), *(argv or [])],   # -I: no user site, no cwd on path
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(cwd or path.parent), env=env,
        preexec_fn=_limits(memory_mb, int(timeout) + 1) if os.name == "posix" else None)
    try:
        out, err = proc.communicate(timeout=timeout)
        elapsed = time.monotonic() - started
        return Execution(proc.returncode == 0, proc.returncode, out, err, elapsed)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        # An infinite loop is the common case and it has no diagnostic of its own,
        # so the timeout has to become one or the grade reads as a mystery.
        return Execution(False, -9, out, err, time.monotonic() - started, timed_out=True)
