"""Tests for radixnet.api - the HTTP JSON API and static file serving.

Every test talks to a real ``ThreadingHTTPServer`` bound to port 0 on
127.0.0.1 through ``urllib.request`` (proxies disabled).  Servers use the
python backend and ``quiet=True``.
"""

import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import __version__, api  # noqa: E402
from radixnet.model import RadixNet, TrainConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = _lines("sample_corpus.txt")
GARBAGE = _lines("sample_garbage.txt")
SLOW_CORPUS = CORPUS * 4  # enough transitions per epoch that a job outlives a few requests
SLOW_EPOCHS = 100_000  # never reached: the tests stop the job
STATS_KEYS = {
    "nodes", "edges", "trigrams", "compression_ratio", "inverted", "backend", "device", "epochs_total",
    "trained_chars", "trained_texts", "twonrl_runs", "history_len", "last_loss",
}
JOB_KEYS = {"id", "type", "state", "progress", "history", "error", "started_at", "finished_at", "stop_requested"}
PREDICT_KEYS = {
    "prefix", "kind", "continuation", "full_text", "cost", "probability", "step_costs", "path", "node_ids", "expanded",
    "reached_end",
}
NODE_KEYS = {"id", "label", "count", "activation", "z", "a", "b", "h", "k"}
EDGE_KEYS = {"source", "target", "weight", "count", "prob", "cost"}


class Client:
    """Tiny JSON client over ``urllib`` (never raises on HTTP errors)."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path, body=None, raw=None, headers=None):
        """Returns ``(status, payload, headers)``; JSON bodies are decoded."""
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with self.opener.open(req, timeout=60) as resp:
                return resp.status, self._decode(resp), resp.headers
        except urllib.error.HTTPError as exc:
            return exc.code, self._decode(exc), exc.headers

    @staticmethod
    def _decode(resp):
        payload = resp.read()
        if resp.headers.get("Content-Type", "").startswith("application/json") and payload:
            return json.loads(payload.decode("utf-8"))
        return payload

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body=None, **kwargs):
        return self.request("POST", path, body={} if body is None else body, **kwargs)


def start_server(add_cleanup, **kwargs):
    """Start a server on a free port in a thread; returns ``(client, server, service)``."""
    kwargs.setdefault("backend", "python")
    kwargs.setdefault("quiet", True)
    server, service = api.create_server("127.0.0.1", 0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        thread.join(5)
        server.server_close()
        service.shutdown(timeout=10)

    add_cleanup(stop)
    return Client(server.url), server, service


def wait_for_job(client, done=lambda job: job["state"] != "running", timeout=60.0):
    """Poll ``GET /api/job`` until ``done(job)`` (fails the test on timeout)."""
    deadline = time.monotonic() + timeout
    while True:
        status, job, _ = client.get("/api/job")
        assert status == 200, job
        if job is not None and done(job):
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job did not reach the expected state in {timeout}s: {job}")
        time.sleep(0.02)


def raw_request(server, payload):
    """Send raw bytes to the server and return everything it answers."""
    with socket.create_connection(server.server_address, timeout=5) as sock:
        sock.sendall(payload)
        chunks = []
        while True:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# protocol: CORS, OPTIONS, errors, fallback page
# ---------------------------------------------------------------------------


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup)

    def test_health(self):
        status, data, headers = self.client.get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data, {"ok": True, "version": __version__})
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Server"], f"radixnet/{__version__}")

    def test_cors_headers_on_every_response(self):
        for status, _, headers in (
            self.client.get("/api/health"),
            self.client.get("/api/nope"),
            self.client.get("/"),
            self.client.post("/api/predict", raw=b"nope"),
        ):
            self.assertIn(status, (200, 400, 404))
            self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
            self.assertIn("POST", headers["Access-Control-Allow-Methods"])
            self.assertIn("Content-Type", headers["Access-Control-Allow-Headers"])

    def test_options_preflight(self):
        for path in ("/api/train", "/api/nope", "/"):
            status, payload, headers = self.client.request("OPTIONS", path)
            self.assertEqual(status, 204, path)
            self.assertEqual(payload, b"")
            self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
            self.assertEqual(headers["Access-Control-Allow-Methods"], "GET, POST, OPTIONS")
            self.assertIn("Content-Type", headers["Access-Control-Allow-Headers"])
            self.assertIsNone(headers.get("Content-Length"))

    def test_unknown_api_path_is_404(self):
        status, data, _ = self.client.get("/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("/api/nope", data["error"])
        status, data, _ = self.client.post("/api/does/not/exist", {"x": 1})
        self.assertEqual(status, 404)
        self.assertIn("error", data)
        # a trailing slash is tolerated
        self.assertEqual(self.client.get("/api/health/")[0], 200)

    def test_wrong_method_is_405_with_allow(self):
        status, data, headers = self.client.get("/api/train")
        self.assertEqual(status, 405)
        self.assertIn("POST", data["error"])
        self.assertEqual(headers["Allow"], "POST, OPTIONS")
        status, data, headers = self.client.post("/api/status")
        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET, OPTIONS")
        status, data, headers = self.client.request("POST", "/", body={})
        self.assertEqual(status, 405)
        self.assertIn("GET", headers["Allow"])

    def test_bad_json_bodies_are_400(self):
        cases = [
            (b"{not json", "invalid JSON"),
            (b"[1, 2]", "must be an object"),
            (b'{"prefix": NaN}', "invalid JSON"),
            (b"\xff\xfe", "UTF-8"),
        ]
        for raw, fragment in cases:
            status, data, _ = self.client.post("/api/predict", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertIn(fragment, data["error"])

    def test_missing_and_mistyped_fields_are_400(self):
        cases = [
            ("/api/predict", {}, "prefix"),
            ("/api/predict", {"prefix": None}, "prefix"),
            ("/api/predict", {"prefix": 5}, "'prefix' must be a string"),
            ("/api/predict", {"prefix": "the", "length": "ten"}, "'length' must be an integer"),
            ("/api/predict", {"prefix": "the", "length": 2.5}, "'length' must be an integer"),
            ("/api/predict", {"prefix": "the", "length": -1}, "'length' must be >= 0"),
            ("/api/predict", {"prefix": "the", "to_end": "yes"}, "'to_end' must be a boolean"),
            ("/api/predict", {"prefix": "the", "step_penalty": True}, "'step_penalty' must be a finite number"),
            ("/api/predict", {"prefix": "the", "mode": "teleport"}, "mode"),
            ("/api/score", {}, "text"),
            ("/api/train", {}, "'texts'"),
            ("/api/train", {"texts": "not a list"}, "list of strings"),
            ("/api/train", {"texts": ["ok", 3]}, "list of strings"),
            ("/api/train", {"texts": []}, "no texts"),
            ("/api/train", {"text": "\n  \n"}, "no texts"),
            ("/api/train", {"texts": ["hello world"], "epochs": -1}, "'epochs' must be >= 0"),
            ("/api/train", {"texts": ["hello world"], "lr": -0.1}, "'lr' must be >= 0"),
            ("/api/train", {"texts": ["hello world"], "batch_size": 0}, "'batch_size' must be >= 1"),
            ("/api/train", {"texts": ["hello world"], "clip": 0}, "clip"),
            ("/api/2nrl", {"bad": ["asdf qwer"]}, "'good'"),
            ("/api/evolve/start", {"corpus": CORPUS[:3], "samples": 0}, "'samples' must be >= 1"),
            ("/api/evolve/start", {"corpus": CORPUS[:3], "generations": -2}, "'generations' must be >= 0"),
            ("/api/load", {}, "path"),
            ("/api/load", {"path": ""}, "path"),
            ("/api/checkpoints/restore", {}, "name"),
        ]
        for path, body, fragment in cases:
            status, data, _ = self.client.post(path, body)
            self.assertEqual(status, 400, (path, body, data))
            self.assertIn(fragment, data["error"], (path, body))
        status, data, _ = self.client.get("/api/graph?limit=abc")
        self.assertEqual(status, 400)
        self.assertIn("limit", data["error"])
        status, data, _ = self.client.get("/api/graph?limit=-3")
        self.assertEqual(status, 400)
        self.assertIn("limit", data["error"])

    def test_body_too_large_is_413(self):
        request = (
            b"POST /api/predict HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
            b"Content-Length: 999999999999\r\n\r\n"
        )
        response = raw_request(self.server, request)
        head, _, body = response.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 413"), response[:80])
        self.assertIn("too large", json.loads(body)["error"])

    def test_unsupported_method_gets_json_error_from_base_class(self):
        response = raw_request(self.server, b"BREW / HTTP/1.1\r\nHost: test\r\n\r\n")
        head, _, body = response.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 501"), response[:80])
        self.assertIn(b"Content-Type: application/json", head)
        self.assertIn("501", json.loads(body)["error"])

    def test_keep_alive_connection_stays_in_sync_after_errors(self):
        # Two requests on one connection: an error response must still consume the body.
        request = (
            b"POST /api/nope HTTP/1.1\r\nHost: test\r\nContent-Length: 7\r\n\r\n"
            b'{"a":1}'
            b"GET /api/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"
        )
        response = raw_request(self.server, request)
        self.assertEqual(response.count(b"HTTP/1.1 "), 2)
        self.assertIn(b"HTTP/1.1 404", response)
        self.assertIn(b"HTTP/1.1 200", response)
        self.assertIn(b'"ok": true', response)

    def test_head_request(self):
        status, payload, headers = self.client.request("HEAD", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"")
        self.assertGreater(int(headers["Content-Length"]), 10)
        status, payload, headers = self.client.request("HEAD", "/")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"")
        self.assertTrue(headers["Content-Type"].startswith("text/html"))

    def test_static_fallback_page_without_frontend(self):
        for path in ("/", "/index.html"):
            status, payload, headers = self.client.get(path)
            self.assertEqual(status, 200, path)
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
            self.assertEqual(int(headers["Content-Length"]), len(payload))
            page = payload.decode("utf-8")
            self.assertIn("npm run build", page)
            self.assertIn("npm install", page)
            self.assertIn('href="/api/health"', page)
            self.assertIn("/api/train", page)
            self.assertIn(__version__, page)
        status, data, _ = self.client.get("/assets/index-abc.js")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_job_endpoints_before_any_job(self):
        status, data, _ = self.client.get("/api/job")
        self.assertEqual(status, 200)
        self.assertIsNone(data)
        status, data, _ = self.client.post("/api/job/stop")
        self.assertEqual(status, 404)
        self.assertIn("no job", data["error"])
        status, data, _ = self.client.post("/api/evolve/stop")
        self.assertEqual(status, 404)
        self.assertIn("evolve", data["error"])
        status, data, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        self.assertIsNone(data["job"])

    def test_request_logging_line(self):
        server, service = api.create_server("127.0.0.1", 0, backend="python", quiet=False)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        try:
            with mock.patch("sys.stderr") as stderr:
                status, _, _ = Client(server.url).get("/api/health")
                self.assertEqual(status, 200)
                deadline = time.monotonic() + 5
                lines = ""
                while "/api/health" not in lines and time.monotonic() < deadline:
                    lines = "".join(call.args[0] for call in stderr.write.call_args_list)
                    time.sleep(0.01)  # the handler logs after the response has been sent
        finally:
            server.shutdown()
            thread.join(5)
            server.server_close()
            service.shutdown()
        self.assertRegex(lines, r"GET /api/health 200 \d+\.\dms")


# ---------------------------------------------------------------------------
# static frontend serving
# ---------------------------------------------------------------------------


class TestStaticFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.dist = os.path.join(cls.tmp.name, "dist")
        os.makedirs(os.path.join(cls.dist, "assets"))
        os.makedirs(os.path.join(cls.dist, "sub"))
        cls.index = "<!doctype html><title>app</title><script src=/assets/app.js></script>"
        with open(os.path.join(cls.dist, "index.html"), "w", encoding="utf-8") as fh:
            fh.write(cls.index)
        with open(os.path.join(cls.dist, "assets", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("console.log('hi');")
        with open(os.path.join(cls.dist, "assets", "style.css"), "w", encoding="utf-8") as fh:
            fh.write("body{margin:0}")
        with open(os.path.join(cls.dist, "sub", "index.html"), "w", encoding="utf-8") as fh:
            fh.write("<p>sub</p>")
        with open(os.path.join(cls.dist, "favicon.ico"), "wb") as fh:
            fh.write(b"\x00\x00\x01\x00")
        cls.secret = os.path.join(cls.tmp.name, "secret.txt")
        with open(cls.secret, "w", encoding="utf-8") as fh:
            fh.write("top secret")
        os.symlink(cls.secret, os.path.join(cls.dist, "link.txt"))
        os.symlink(cls.tmp.name, os.path.join(cls.dist, "updir"))
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, frontend_dir=cls.dist)

    def test_index_and_assets_with_mime_types(self):
        status, payload, headers = self.client.get("/")
        self.assertEqual(status, 200)
        self.assertEqual(payload.decode("utf-8"), self.index)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-cache")
        self.assertEqual(int(headers["Content-Length"]), len(self.index.encode("utf-8")))
        status, payload, headers = self.client.get("/assets/app.js")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"console.log('hi');")
        self.assertEqual(headers["Content-Type"], "application/javascript; charset=utf-8")
        self.assertIn("immutable", headers["Cache-Control"])
        status, payload, headers = self.client.get("/assets/style.css")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/css; charset=utf-8")
        status, payload, headers = self.client.get("/favicon.ico")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/x-icon")
        status, payload, headers = self.client.get("/sub/")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"<p>sub</p>")

    def test_spa_fallback_only_for_routes(self):
        for path in ("/predict", "/graph/", "/deep/client/route", "/index.html?x=1#train"):
            status, payload, headers = self.client.get(path)
            self.assertEqual(status, 200, path)
            self.assertEqual(payload.decode("utf-8"), self.index, path)
            self.assertTrue(headers["Content-Type"].startswith("text/html"))
        for path in ("/assets/missing.js", "/missing.png", "/assets/nested/missing.css"):
            status, data, _ = self.client.get(path)
            self.assertEqual(status, 404, path)
            self.assertIn("not found", data["error"])

    def test_path_traversal_is_rejected(self):
        attempts = (
            "/../secret.txt",
            "/../../../../etc/passwd",
            "/%2e%2e/secret.txt",
            "/assets/%2e%2e/%2e%2e/secret.txt",
            "/assets/..%2fsecret.txt",
            "/assets/../../secret.txt",
            "/..%5csecret.txt",
            "/link.txt",
            "/updir/secret.txt",
            "/updir/secret",  # dotless: must not fall back to index.html either
            "//" + self.secret.lstrip("/"),
        )
        for path in attempts:
            status, payload, _ = self.client.get(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"top secret", payload if isinstance(payload, bytes) else b"")
        root = os.path.realpath(self.dist)
        for path in ("/link.txt", "/updir/secret", "/../secret.txt", "/a\x00b", "/..\\secret.txt"):
            with self.assertRaises(PermissionError, msg=path):
                api._resolve_static(root, path)
        self.assertIsNone(api._resolve_static(root, "/assets/missing.js"))
        self.assertIsNone(api._resolve_static(root, "/route"))
        self.assertEqual(
            api._resolve_static(os.path.realpath(self.dist), "/./assets//app.js"),
            os.path.join(os.path.realpath(self.dist), "assets", "app.js"),
        )

    def test_api_still_served_next_to_frontend(self):
        status, data, _ = self.client.get("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.client.get("/api/nope")[0], 404)

    def test_frontend_appears_once_built(self):
        with tempfile.TemporaryDirectory() as empty:
            client, _, _ = start_server(self.addCleanup, frontend_dir=os.path.join(empty, "dist"))
            status, payload, _ = client.get("/")
            self.assertEqual(status, 200)
            self.assertIn(b"npm run build", payload)
            self.assertIn(b"dist", payload)
            self.assertEqual(client.get("/predict")[0], 404)
            os.makedirs(os.path.join(empty, "dist"))
            with open(os.path.join(empty, "dist", "index.html"), "w", encoding="utf-8") as fh:
                fh.write("<h1>built</h1>")
            status, payload, _ = client.get("/")
            self.assertEqual(status, 200)
            self.assertEqual(payload, b"<h1>built</h1>")
            self.assertEqual(client.get("/predict")[1], b"<h1>built</h1>")


# ---------------------------------------------------------------------------
# synchronous endpoints on a trained model
# ---------------------------------------------------------------------------


class TestEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup)
        cls.service.model.train(CORPUS, epochs=2, batch_size=8)

    def test_status(self):
        status, data, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(STATS_KEYS <= set(data), STATS_KEYS - set(data))
        self.assertEqual(data["backend"], "python")
        self.assertEqual(data["epochs_total"], 2)
        self.assertGreater(data["nodes"], 2)
        self.assertIsNone(data["job"])
        self.assertIsNone(data["model_path"])
        self.assertIsNone(data["checkpoint_dir"])
        backends = data["backends"]
        self.assertEqual(set(backends), {"python", "torch", "cuda", "mps", "default"})
        self.assertTrue(backends["python"])
        self.assertIn(backends["default"], ("python", "torch"))

    def test_predict(self):
        body = {"prefix": "the quick br", "length": 8, "mode": "dijkstra", "to_end": False, "step_penalty": 0.0}
        status, data, _ = self.client.post("/api/predict", body)
        self.assertEqual(status, 200)
        self.assertEqual(set(data), PREDICT_KEYS)
        self.assertEqual(data["prefix"], "the quick br")
        self.assertEqual(data["full_text"], data["prefix"] + data["continuation"])
        self.assertGreaterEqual(len(data["continuation"]), 1)
        self.assertIn(" br", data["path"][0])  # the located context node
        self.assertEqual(len(data["step_costs"]), len(data["path"]) - 1)
        self.assertEqual(len(data["node_ids"]), len(data["path"]))
        self.assertAlmostEqual(data["cost"], sum(data["step_costs"]), places=9)
        self.assertGreaterEqual(data["expanded"], 1)
        status, data, _ = self.client.post("/api/predict", {"prefix": "the quick br", "length": 6, "to_end": True})
        self.assertEqual(status, 200)
        self.assertTrue(data["reached_end"])
        self.assertEqual(data["path"][-1], "</s>")
        status, data, _ = self.client.post(
            "/api/predict", {"prefix": "the", "length": 15, "mode": "sample", "temperature": 0.5}
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["full_text"], "the" + data["continuation"])
        status, data, _ = self.client.post("/api/predict", {"prefix": "", "length": 5})
        self.assertEqual(status, 200)
        self.assertEqual(data["path"][0], "<s>")

    def test_generate(self):
        status, data, _ = self.client.post("/api/generate", {"count": 3, "max_length": 25, "temperature": 1.0})
        self.assertEqual(status, 200)
        self.assertEqual(len(data["samples"]), 3)
        for sample in data["samples"]:
            self.assertTrue({"text", "cost", "path"} <= set(sample))
            self.assertLessEqual(len(sample["text"]), 25)
            self.assertEqual(sample["path"][0], "<s>")
            self.assertEqual(len(sample["step_costs"]), len(sample["path"]) - 1)
        status, data, _ = self.client.post("/api/generate", {"count": 4, "mode": "dijkstra", "max_length": 40})
        self.assertEqual(status, 200)
        self.assertEqual(len(data["samples"]), 1)
        self.assertTrue(data["samples"][0]["reached_end"])
        status, data, _ = self.client.post("/api/generate", {"count": 2, "seed": 7})
        status2, data2, _ = self.client.post("/api/generate", {"count": 2, "seed": 7})
        self.assertEqual((status, status2), (200, 200))
        self.assertEqual(data, data2)
        status, data, _ = self.client.post("/api/generate", {"count": 0})
        self.assertEqual((status, data), (200, {"samples": []}))
        # beam: the K most likely complete texts from the prediction search
        status, data, _ = self.client.post("/api/generate", {"count": 3, "mode": "beam", "max_length": 40, "beam": 24})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["samples"]), 3)
        self.assertEqual(len({s["text"] for s in data["samples"]}), 3)
        costs = [s["cost"] for s in data["samples"]]
        self.assertEqual(costs, sorted(costs))
        for sample in data["samples"]:
            self.assertTrue({"text", "full_text", "cost", "probability", "path", "reached_end"} <= set(sample))
            self.assertEqual(sample["full_text"], sample["text"])
            self.assertTrue(sample["reached_end"] or len(sample["text"]) == 40, sample)
            self.assertAlmostEqual(sample["probability"], math.exp(-sample["cost"]), places=9)
        # every mode continues a prefix; the samples are whole texts
        for mode in ("beam", "sample", "dijkstra"):
            status, data, _ = self.client.post("/api/generate", {"count": 2, "mode": mode, "prefix": "the ", "max_length": 20})
            self.assertEqual(status, 200, data)
            self.assertEqual(len(data["samples"]), 1 if mode == "dijkstra" else 2)
            for sample in data["samples"]:
                self.assertTrue(sample["text"].startswith("the "), sample["text"])
                self.assertEqual(sample["full_text"], sample["text"])
        status, data, _ = self.client.post("/api/generate", {"mode": "nope"})
        self.assertEqual(status, 400)
        status, data, _ = self.client.post("/api/generate", {"mode": "beam", "beam": 0})
        self.assertEqual(status, 400)

    def test_converse(self):
        status, data, _ = self.client.post("/api/converse", {"opening": "the cat sat on the mat", "turns": 4})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["kind"], data["partner"], data["speakers"]), ("radix", None, ["A", "B"]))
        self.assertEqual(data["count"], len(data["turns"]))
        self.assertEqual(data["count"], 5)
        turns = data["turns"]
        self.assertEqual((turns[0]["speaker"], turns[0]["text"], turns[0]["given"]), ("A", "the cat sat on the mat", True))
        for i, turn in enumerate(turns):
            self.assertEqual(turn["index"], i)
            self.assertEqual(turn["speaker"], "AB"[i % 2])
            self.assertTrue({"text", "context", "reply", "cost", "probability", "fresh", "given", "repeat",
                             "candidates", "skipped", "labels", "node_ids", "step_costs", "reached_end"} <= set(turn))
            if i and not turn["fresh"]:
                words, wanted = turns[i - 1]["text"].split(), turn["context"].split()
                self.assertTrue(any(words[j:j + len(wanted)] == wanted for j in range(len(words))), (words, wanted))
                self.assertEqual(turn["text"], turn["context"] + turn["reply"])
        texts = [t["text"] for t in turns]
        self.assertEqual(len(set(texts)), len(texts))
        # continue: the history is picked up, indices and speakers carry on
        status, more, _ = self.client.post("/api/converse", {"turns": 2, "history": texts, "speakers": ["me", "you"]})
        self.assertEqual(status, 200, more)
        self.assertEqual([(t["index"], t["speaker"]) for t in more["turns"]], [(5, "you"), (6, "me")])
        self.assertEqual(more["speakers"], ["me", "you"])
        status, sampled, _ = self.client.post("/api/converse", {"turns": 3, "mode": "sample", "seed": 3, "max_length": 20})
        status2, sampled2, _ = self.client.post("/api/converse", {"turns": 3, "mode": "sample", "seed": 3, "max_length": 20})
        self.assertEqual((status, status2), (200, 200))
        self.assertEqual(sampled, sampled2)
        self.assertEqual(sampled["count"], 3)
        status, data, _ = self.client.post("/api/converse", {"turns": 0})
        self.assertEqual((status, data["count"], data["turns"]), (200, 0, []))
        # errors
        for body in ({"mode": "nope"}, {"k": 0}, {"turns": -1}, {"partner": "nope"}, {"partner": "count"},
                     {"speakers": ["", "B"]}, {"history": "not a list"}, {"opening": 5}):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/converse", body)
                self.assertEqual(status, 400, data)
                self.assertIn("error", data)
        status, data, _ = self.client.post("/api/converse", {"partner": "count"})
        self.assertIn("in memory", data["error"])

    def test_score(self):
        status, good, _ = self.client.post("/api/score", {"text": CORPUS[0]})
        self.assertEqual(status, 200)
        self.assertEqual(set(good), {"log_prob", "per_char", "chars", "transitions", "unknown_transitions"})
        self.assertEqual(good["chars"], len(CORPUS[0]))
        self.assertEqual(good["unknown_transitions"], 0)
        status, bad, _ = self.client.post("/api/score", {"text": "zqxj vwk plmn qzx"})
        self.assertEqual(status, 200)
        self.assertGreater(bad["unknown_transitions"], 0)
        self.assertGreater(good["per_char"], bad["per_char"])

    def test_graph(self):
        status, data, _ = self.client.get("/api/graph?limit=5")
        self.assertEqual(status, 200)
        ids = [n["id"] for n in data["nodes"]]
        self.assertEqual(ids[:2], [0, 1])
        self.assertEqual(len(ids), 7)
        self.assertEqual(len(set(ids)), 7)
        self.assertEqual(data["limit"], 5)
        self.assertEqual(data["total_nodes"], self.service.model.graph.num_nodes())
        graph = self.service.model.graph
        counts = sorted((graph.count[i] for i in graph.alive_nodes()[2:]), reverse=True)
        self.assertEqual(sorted((n["count"] for n in data["nodes"][2:]), reverse=True), counts[:5])
        for node in data["nodes"]:
            self.assertEqual(set(node), NODE_KEYS)
            self.assertAlmostEqual(node["activation"], graph.activation_of(node["id"]))
        self.assertEqual(data["nodes"][0]["label"], "<s>")
        self.assertEqual(data["nodes"][1]["label"], "</s>")
        chosen = set(ids)
        for edge in data["edges"]:
            self.assertEqual(set(edge), EDGE_KEYS)
            self.assertIn(edge["source"], chosen)
            self.assertIn(edge["target"], chosen)
            self.assertGreaterEqual(edge["cost"], 0.0)
            self.assertTrue(0.0 <= edge["prob"] <= 1.0)
            self.assertAlmostEqual(edge["prob"], math.exp(-edge["cost"]))
            self.assertIn(edge["target"], graph.children[edge["source"]])
        status, data, _ = self.client.get("/api/graph?limit=0")
        self.assertEqual(status, 200)
        self.assertEqual([n["id"] for n in data["nodes"]], [0, 1])
        self.assertEqual(data["edges"], [])
        status, data, _ = self.client.get("/api/graph")
        self.assertEqual(status, 200)
        self.assertLessEqual(len(data["nodes"]), api.DEFAULT_GRAPH_LIMIT + 2)
        status, data, _ = self.client.get("/api/graph?limit=100000")
        self.assertEqual(len(data["nodes"]), graph.num_nodes())
        self.assertEqual(len(data["edges"]), graph.num_edges())

    def test_history(self):
        status, data, _ = self.client.get("/api/history")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["history"]), 2)
        self.assertEqual([r["epoch"] for r in data["history"]], [1, 2])
        self.assertTrue({"loss", "perplexity", "nodes", "edges", "compression_ratio"} <= set(data["history"][0]))
        status, data, _ = self.client.get("/api/evolve/history")
        self.assertEqual((status, data), (200, {"history": []}))

    def test_invert_and_compress(self):
        before = self.client.get("/api/status")[1]["inverted"]
        status, data, _ = self.client.post("/api/invert")
        self.assertEqual(status, 200)
        self.assertTrue(STATS_KEYS <= set(data))
        self.assertEqual(data["inverted"], not before)
        status, data, _ = self.client.post("/api/invert")
        self.assertEqual(data["inverted"], before)
        status, data, _ = self.client.post("/api/compress")
        self.assertEqual(status, 200)
        self.assertTrue({"merges"} | STATS_KEYS <= set(data))
        self.assertEqual(data["merges"], 0)  # auto-compressed during training already


# ---------------------------------------------------------------------------
# background jobs
# ---------------------------------------------------------------------------


class TestJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt_dir = os.path.join(self.tmp.name, "ckpt")
        self.client, self.server, self.service = start_server(self.addCleanup, checkpoint_dir=self.ckpt_dir)

    def test_train_job_lifecycle(self):
        status, data, _ = self.client.post(
            "/api/train", {"texts": CORPUS, "epochs": 3, "lr": 0.5, "act_lr": 0.01, "batch_size": 8, "auto_compress": True}
        )
        self.assertEqual(status, 202)
        job = data["job"]
        self.assertEqual(set(job), JOB_KEYS)
        self.assertEqual((job["type"], job["state"], job["error"], job["finished_at"]), ("train", "running", None, None))
        self.assertFalse(job["stop_requested"])
        self.assertRegex(job["started_at"], r"^\d{4}-\d{2}-\d{2}T")
        done = wait_for_job(self.client)
        self.assertEqual(done["id"], job["id"])
        self.assertEqual(done["state"], "done")
        self.assertIsNone(done["error"])
        self.assertIsNotNone(done["finished_at"])
        self.assertEqual(len(done["history"]), 3)
        self.assertEqual(done["progress"], done["history"][-1])
        self.assertEqual([r["epoch"] for r in done["history"]], [1, 2, 3])
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["job"]["state"], "done")
        self.assertEqual(stats["epochs_total"], 3)
        self.assertEqual(stats["last_loss"], done["history"][-1]["loss"])
        self.assertEqual(len(self.client.get("/api/history")[1]["history"]), 3)
        status, pred, _ = self.client.post("/api/predict", {"prefix": "the quick brown", "length": 10})
        self.assertEqual(status, 200)
        self.assertGreater(len(pred["path"]), 0)
        # a second job is allowed once the first is finished
        status, data, _ = self.client.post("/api/train", {"text": "hello world\n\nhello there\n", "epochs": 1})
        self.assertEqual(status, 202)
        self.assertEqual(data["job"]["id"], "train-2")
        self.assertEqual(wait_for_job(self.client)["state"], "done")
        self.assertEqual(self.client.get("/api/status")[1]["trained_texts"], len(CORPUS) + 2)

    def test_mutations_refused_and_reads_served_while_training(self):
        status, data, _ = self.client.post("/api/train", {"texts": SLOW_CORPUS, "epochs": SLOW_EPOCHS, "batch_size": 4})
        self.assertEqual(status, 202)
        job_id = data["job"]["id"]
        refused = [
            ("/api/train", {"texts": ["hello world"]}),
            ("/api/2nrl", {"bad": ["asdf qwer"], "good": ["hello world"]}),
            ("/api/evolve/start", {"corpus": CORPUS[:3]}),
            ("/api/invert", {}),
            ("/api/compress", {}),
            ("/api/reset", {}),
            ("/api/load", {"path": os.path.join(self.tmp.name, "nope.json")}),
            ("/api/checkpoints/restore", {"name": "nope"}),
            ("/api/evolve/stop", {}),
        ]
        for path, body in refused:
            status, data, _ = self.client.post(path, body)
            self.assertEqual(status, 409, (path, data))
            self.assertIn(job_id, data["error"])
        # reads block at most one epoch and then succeed
        status, data, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(data["job"]["state"], "running")
        self.assertEqual(data["job"]["id"], job_id)
        status, data, _ = self.client.post("/api/predict", {"prefix": "the quick br", "length": 5})
        self.assertEqual(status, 200)
        self.assertEqual(set(data), PREDICT_KEYS)
        self.assertEqual(self.client.post("/api/generate", {"count": 1, "max_length": 10})[0], 200)
        self.assertEqual(self.client.post("/api/score", {"text": CORPUS[0]})[0], 200)
        self.assertEqual(self.client.get("/api/graph?limit=3")[0], 200)
        self.assertEqual(self.client.get("/api/history")[0], 200)
        self.assertEqual(self.client.get("/api/checkpoints")[0], 200)
        # snapshots are allowed: they read the model at an epoch boundary
        snapshot = os.path.join(self.tmp.name, "snapshot.json")
        status, data, _ = self.client.post("/api/save", {"path": snapshot})
        self.assertEqual(status, 200)
        self.assertTrue(os.path.isfile(snapshot))
        status, data, _ = self.client.post("/api/checkpoints/save", {"tag": "mid"})
        self.assertEqual(status, 200)
        self.assertEqual(data["tag"], "mid")
        status, running, _ = self.client.get("/api/job")
        self.assertEqual(running["state"], "running")
        status, data, _ = self.client.post("/api/job/stop")
        self.assertEqual(status, 200)
        self.assertTrue(data["stop_requested"])
        self.assertEqual(data["id"], job_id)
        stopped = wait_for_job(self.client)
        self.assertEqual(stopped["state"], "stopped")
        self.assertIsNone(stopped["error"])
        self.assertLess(len(stopped["history"]), SLOW_EPOCHS)
        self.assertGreaterEqual(len(stopped["history"]), 1)
        # stopping again is idempotent, and the model is mutable again
        self.assertEqual(self.client.post("/api/job/stop")[1]["state"], "stopped")
        self.assertEqual(self.client.post("/api/invert")[0], 200)

    def test_train_job_checkpoints_every_epoch(self):
        status, data, _ = self.client.post(
            "/api/train", {"texts": CORPUS[:10], "epochs": 2, "batch_size": 8, "checkpoint_every": 1}
        )
        self.assertEqual(status, 202)
        self.assertEqual(wait_for_job(self.client)["state"], "done")
        status, data, _ = self.client.get("/api/checkpoints")
        self.assertEqual(status, 200)
        self.assertEqual([(c["tag"], c["step"]) for c in data["checkpoints"]], [("epoch", 1), ("epoch", 2)])
        self.assertEqual(data["latest"]["step"], 2)
        client, _, _ = start_server(self.addCleanup)  # no checkpoint directory
        status, data, _ = client.post("/api/train", {"texts": CORPUS[:3], "epochs": 1, "checkpoint_every": 1})
        self.assertEqual(status, 400)
        self.assertIn("checkpoint directory", data["error"])

    def test_two_nrl_job(self):
        self.service.model.train(CORPUS[:20], epochs=1, batch_size=8)
        inverted_before = self.client.get("/api/status")[1]["inverted"]
        body = {
            "bad_text": "\n".join(GARBAGE[:5]) + "\n\n", "good": CORPUS[:5],
            "neg_epochs": 1, "pos_epochs": 2, "neg_lr": 0.5, "pos_lr": 0.1, "batch_size": 8,
        }
        status, data, _ = self.client.post("/api/2nrl", body)
        self.assertEqual(status, 202)
        self.assertEqual(data["job"]["type"], "2nrl")
        self.assertEqual(data["job"]["state"], "running")
        done = wait_for_job(self.client)
        self.assertEqual(done["state"], "done")
        self.assertIsNone(done["error"])
        phases = [r["phase"] for r in done["history"]]
        self.assertEqual(phases, ["negative", "positive", "positive"])
        self.assertEqual(done["progress"]["phase"], "positive")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["inverted"], not inverted_before)
        self.assertEqual(stats["twonrl_runs"], 1)
        self.assertEqual(stats["epochs_total"], 4)
        status, data, _ = self.client.post("/api/2nrl", {"bad": ["asdf qwer"], "good": ["hello world"], "neg_lr": -1})
        self.assertEqual(status, 400)
        self.assertIn("neg_lr", data["error"])
        status, data, _ = self.client.post("/api/2nrl", {"bad": ["asdf qwer"], "good": ["hello world"], "batch_size": 0})
        self.assertEqual(status, 400)
        self.assertIn("batch_size", data["error"])

    def test_evolve_job_runs_until_stopped(self):
        self.service.model.train(CORPUS[:20], epochs=1, batch_size=8)
        body = {
            "corpus": CORPUS[:10], "generations": None, "samples": 2, "real_per_generation": 2,
            "max_length": 12, "batch_size": 4, "temperature": 1.0, "checkpoint_every": 0,
        }
        status, data, _ = self.client.post("/api/evolve/start", body)
        self.assertEqual(status, 202)
        self.assertEqual(data["job"]["type"], "evolve")
        self.assertEqual(data["job"]["state"], "running")
        running = wait_for_job(self.client, done=lambda job: len(job["history"]) >= 1 or job["state"] != "running")
        self.assertEqual(running["state"], "running", running["error"])
        status, data, _ = self.client.post("/api/evolve/stop")
        self.assertEqual(status, 200)
        self.assertTrue(data["stop_requested"])
        stopped = wait_for_job(self.client)
        self.assertEqual(stopped["state"], "stopped")
        self.assertIsNone(stopped["error"])
        self.assertGreaterEqual(len(stopped["history"]), 1)
        record = stopped["history"][0]
        self.assertTrue({"generation", "fake_score_mean", "real_score_mean", "gap", "gen_loss", "sample"} <= set(record))
        self.assertEqual(record["generation"], 1)
        status, data, _ = self.client.get("/api/evolve/history")
        self.assertEqual(status, 200)
        self.assertEqual(data["history"], stopped["history"])
        discriminator = self.service.discriminator
        self.assertIsInstance(discriminator, RadixNet)
        # a bounded second run continues with the same discriminator and appends to the history
        status, data, _ = self.client.post("/api/evolve/start", dict(body, generations=1))
        self.assertEqual(status, 202)
        done = wait_for_job(self.client)
        self.assertEqual(done["state"], "done")
        self.assertEqual(len(done["history"]), 1)
        self.assertIs(self.service.discriminator, discriminator)
        history = self.client.get("/api/evolve/history")[1]["history"]
        self.assertEqual(len(history), len(stopped["history"]) + 1)
        # a different seed starts a fresh discriminator
        status, data, _ = self.client.post("/api/evolve/start", dict(body, generations=1, seed=9))
        self.assertEqual(status, 202)
        self.assertEqual(wait_for_job(self.client)["state"], "done")
        self.assertIsNot(self.service.discriminator, discriminator)
        status, data, _ = self.client.post("/api/evolve/start", {"corpus": ["ab", "c"]})
        self.assertEqual(status, 400)
        self.assertIn("corpus", data["error"])

    def test_evolve_stop_refuses_other_job_types(self):
        status, data, _ = self.client.post("/api/train", {"texts": SLOW_CORPUS, "epochs": SLOW_EPOCHS, "batch_size": 4})
        self.assertEqual(status, 202)
        status, data, _ = self.client.post("/api/evolve/stop")
        self.assertEqual(status, 409)
        self.assertIn("train", data["error"])
        self.client.post("/api/job/stop")
        self.assertEqual(wait_for_job(self.client)["state"], "stopped")
        status, data, _ = self.client.post("/api/evolve/stop")
        self.assertEqual(status, 404)


# ---------------------------------------------------------------------------
# persistence: save / load / reset / checkpoints
# ---------------------------------------------------------------------------


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = os.path.join(self.tmp.name, "model.json")
        self.ckpt_dir = os.path.join(self.tmp.name, "ckpt")
        self.client, self.server, self.service = start_server(
            self.addCleanup, model_path=self.model_path, checkpoint_dir=self.ckpt_dir
        )
        self.service.model.train(CORPUS[:15], epochs=1, batch_size=8)

    def test_save_load_round_trip(self):
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["model_path"], self.model_path)
        self.assertEqual(stats["checkpoint_dir"], self.ckpt_dir)
        status, data, _ = self.client.post("/api/save")
        self.assertEqual(status, 200)
        self.assertEqual(data["path"], self.model_path)
        self.assertEqual(data["bytes"], os.path.getsize(self.model_path))
        self.assertGreater(data["bytes"], 100)
        other = os.path.join(self.tmp.name, "other.json.gz")
        status, data, _ = self.client.post("/api/save", {"path": other})
        self.assertEqual(status, 200)
        self.assertEqual(data["path"], other)
        self.assertTrue(os.path.isfile(other))
        text_fields = ("continuation", "full_text", "cost", "path", "step_costs")
        prediction = self.client.post("/api/predict", {"prefix": "the quick br", "length": 8})[1]
        prediction = {k: prediction[k] for k in text_fields}  # node ids are compacted on save
        status, data, _ = self.client.post("/api/reset", {"seed": 3})
        self.assertEqual(status, 200)
        self.assertEqual((data["nodes"], data["edges"], data["epochs_total"]), (2, 0, 0))
        self.assertEqual(self.service.model.seed, 3)
        status, data, _ = self.client.post("/api/load", {"path": other})
        self.assertEqual(status, 200)
        self.assertEqual({k: data[k] for k in ("nodes", "edges", "trigrams", "epochs_total")},
                         {k: stats[k] for k in ("nodes", "edges", "trigrams", "epochs_total")})
        again = self.client.post("/api/predict", {"prefix": "the quick br", "length": 8})[1]
        self.assertEqual({k: again[k] for k in text_fields}, prediction)
        self.assertEqual(len(self.client.get("/api/history")[1]["history"]), 1)
        status, data, _ = self.client.post("/api/load", {"path": os.path.join(self.tmp.name, "missing.json")})
        self.assertEqual(status, 404)
        self.assertIn("missing.json", data["error"])
        bogus = os.path.join(self.tmp.name, "bogus.json")
        with open(bogus, "w", encoding="utf-8") as fh:
            fh.write('{"format": "something-else"}')
        status, data, _ = self.client.post("/api/load", {"path": bogus})
        self.assertEqual(status, 400)
        self.assertIn("radixnet", data["error"])
        status, data, _ = self.client.post("/api/reset")
        self.assertEqual(status, 200)
        self.assertEqual(self.service.model.seed, 0)

    def test_save_requires_a_path_without_model_path(self):
        client, _, _ = start_server(self.addCleanup)
        status, data, _ = client.post("/api/save")
        self.assertEqual(status, 400)
        self.assertIn("path", data["error"])
        target = os.path.join(self.tmp.name, "explicit.json")
        status, data, _ = client.post("/api/save", {"path": target})
        self.assertEqual(status, 200)
        self.assertEqual(data["path"], target)
        status, data, _ = client.post("/api/save", {"path": self.tmp.name})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_model_loaded_at_startup(self):
        self.client.post("/api/save")
        stats = self.client.get("/api/status")[1]
        client, _, service = start_server(self.addCleanup, model_path=self.model_path)
        self.assertEqual(service.loaded_from, self.model_path)
        loaded = client.get("/api/status")[1]
        self.assertEqual((loaded["nodes"], loaded["edges"], loaded["epochs_total"]), (stats["nodes"], stats["edges"], 1))
        self.assertEqual(self.service.loaded_from, None)

    def test_checkpoints(self):
        status, data, _ = self.client.get("/api/checkpoints")
        self.assertEqual((status, data), (200, {"checkpoints": [], "latest": None}))
        status, record, _ = self.client.post("/api/checkpoints/save", {"tag": "manual"})
        self.assertEqual(status, 200)
        self.assertEqual(set(record), {"name", "path", "step", "tag", "metrics", "saved_at", "bytes"})
        self.assertEqual((record["tag"], record["step"]), ("manual", 1))
        self.assertEqual(record["metrics"]["epoch"], 1)
        self.assertTrue(os.path.isfile(record["path"]))
        status, data, _ = self.client.post("/api/checkpoints/save", {})
        self.assertEqual(status, 200)
        self.assertEqual(data["tag"], "manual")  # default tag, same step -> overwrites
        status, data, _ = self.client.post("/api/checkpoints/save", {"tag": "before-reset"})
        self.assertEqual(status, 200)
        status, data, _ = self.client.get("/api/checkpoints")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(c["name"] for c in data["checkpoints"]), sorted([record["name"], data["latest"]["name"]]))
        self.assertEqual(data["latest"]["tag"], "before-reset")
        stats = self.client.get("/api/status")[1]
        self.client.post("/api/reset", {"seed": 1})
        self.assertEqual(self.client.get("/api/status")[1]["nodes"], 2)
        status, restored, _ = self.client.post("/api/checkpoints/restore", {"name": record["name"]})
        self.assertEqual(status, 200)
        self.assertEqual((restored["nodes"], restored["edges"], restored["epochs_total"]),
                         (stats["nodes"], stats["edges"], stats["epochs_total"]))
        status, data, _ = self.client.post("/api/checkpoints/restore", {"name": "ckpt-nope-000001"})
        self.assertEqual(status, 404)
        self.assertIn("nope", data["error"])
        status, data, _ = self.client.post("/api/checkpoints/save", {"tag": "bad tag!"})
        self.assertEqual(status, 400)
        self.assertIn("tag", data["error"])

    def test_checkpoint_endpoints_without_directory(self):
        client, _, _ = start_server(self.addCleanup)
        status, data, _ = client.get("/api/checkpoints")
        self.assertEqual((status, data), (200, {"checkpoints": [], "latest": None}))
        for path, body in (("/api/checkpoints/save", {"tag": "x"}), ("/api/checkpoints/restore", {"name": "x"})):
            status, data, _ = client.post(path, body)
            self.assertEqual(status, 400, path)
            self.assertIn("--checkpoint-dir", data["error"])


# ---------------------------------------------------------------------------
# the service and run_server without HTTP
# ---------------------------------------------------------------------------


class TestModelService(unittest.TestCase):
    def test_reader_handoff_while_a_job_runs(self):
        service = api.ModelService(backend="python", quiet=True)
        job = service.start_train(SLOW_CORPUS, TrainConfig(epochs=SLOW_EPOCHS, batch_size=4))
        self.assertEqual(job["state"], "running")
        t0 = time.monotonic()
        for _ in range(3):
            with service.session() as model:
                self.assertIsInstance(model.stats()["nodes"], int)
        self.assertLess(time.monotonic() - t0, 30.0)
        self.assertTrue(service.job.running)
        with self.assertRaises(api.ApiError) as ctx:
            service.invert()
        self.assertEqual(ctx.exception.status, 409)
        service.shutdown()
        self.assertEqual(service.job_status()["state"], "stopped")
        self.assertFalse(service.job.thread.is_alive())
        self.assertIn("inverted", service.invert())

    def test_failing_job_is_reported_not_raised(self):
        service = api.ModelService(backend="python", quiet=True)

        def boom(job):
            raise RuntimeError("boom")

        job = service._start_job("train", boom)
        self.assertEqual(job["state"], "running")
        service.job.thread.join(5)
        status = service.job_status()
        self.assertEqual(status["state"], "error")
        self.assertEqual(status["error"], "RuntimeError: boom")
        self.assertIsNotNone(status["finished_at"])
        # the service is usable again afterwards
        self.assertEqual(service.start_train(CORPUS[:3], TrainConfig(epochs=1))["type"], "train")
        service.job.thread.join(10)
        self.assertEqual(service.job_status()["state"], "done")

    def test_run_server_shuts_down_cleanly_on_keyboard_interrupt(self):
        captured = {}

        def fake_serve_forever(server, poll_interval=0.5):
            captured["server"] = server
            server.service.start_train(SLOW_CORPUS, TrainConfig(epochs=SLOW_EPOCHS, batch_size=4))
            raise KeyboardInterrupt

        with mock.patch.object(api.RadixNetHTTPServer, "serve_forever", fake_serve_forever):
            api.run_server("127.0.0.1", 0, backend="python", quiet=True)
        server = captured["server"]
        self.assertEqual(server.service.job_status()["state"], "stopped")
        self.assertFalse(server.service.job.thread.is_alive())
        self.assertEqual(server.socket.fileno(), -1)  # closed


if __name__ == "__main__":
    unittest.main()
