"""Run the actual make serve recipe under /bin/bash with small stand-in servers."""
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"
STUB = '''\
import os, pathlib, socket, subprocess, sys, time
role, port = sys.argv[1:]
pathlib.Path(role + '.pid').write_text(str(os.getpid()))
if role == 'backend':
    server = socket.socket()
    server.bind(('127.0.0.1', int(port)))
    server.listen()
if role != 'child':
    child = subprocess.Popen([sys.executable, __file__, 'child', port])
    pathlib.Path(role + '-child.pid').write_text(str(child.pid))
pathlib.Path(role + '.ready').touch()
while not pathlib.Path(role + '.exit').exists():
    time.sleep(0.02)
sys.exit(int(pathlib.Path(role + '.exit').read_text()))
'''


@unittest.skipUnless(os.name == "posix", "serve uses POSIX process groups")
class TestServe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "stub.py").write_text(STUB)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        command = shlex.quote(sys.executable) + " stub.py"
        self.proc = subprocess.Popen([
            "make", "--no-print-directory", "-f", str(MAKEFILE), "-o", "build",
            "-o", "frontend/node_modules", "serve", "SHELL=/bin/bash",
            "HOST=127.0.0.1", f"PORT={self.port}",
            f"SERVE_BACKEND={command} backend {self.port}",
            f"SERVE_FRONTEND=exec {command} frontend {self.port}",
        ], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        self.addCleanup(self.stop)
        self.wait_for(lambda: (self.root / "frontend.ready").exists())

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        for path in self.root.glob("*.pid"):
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.communicate(timeout=5)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.03)
        self.fail("launcher did not reach the expected state")

    def assert_stopped(self):
        output, _ = self.proc.communicate(timeout=5)
        self.assertNotIn("invalid option", output)
        self.assertNotIn("run_pending_traps", output)
        for role in ("backend", "frontend", "backend-child", "frontend-child"):
            path = self.root / (role + ".pid")
            if not path.exists():
                continue
            pid = int(path.read_text())
            def gone():
                # Orphaned grandchildren may briefly be zombies awaiting init's reaper.
                result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
                return result.returncode != 0 or result.stdout.strip().startswith("Z")
            self.wait_for(gone)
        return output

    def test_frontend_failure_stops_backend_and_descendants(self):
        (self.root / "frontend.exit").write_text("7")
        output = self.assert_stopped()
        self.assertNotEqual(self.proc.returncode, 0)
        self.assertIn("one of the two stopped (exit 7)", output)

    def test_backend_failure_stops_frontend_and_descendants(self):
        (self.root / "backend.exit").write_text("9")
        output = self.assert_stopped()
        self.assertNotEqual(self.proc.returncode, 0)
        self.assertIn("one of the two stopped (exit 9)", output)

    def test_clean_exit_is_success(self):
        (self.root / "frontend.exit").write_text("0")
        self.assert_stopped()
        self.assertEqual(self.proc.returncode, 0)

    def test_ctrl_c_stops_both_groups(self):
        os.killpg(self.proc.pid, signal.SIGINT)
        self.assert_stopped()

    def test_sigterm_stops_both_groups(self):
        os.killpg(self.proc.pid, signal.SIGTERM)
        self.assert_stopped()
