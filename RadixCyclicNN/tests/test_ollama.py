"""Tests for ``radixnet.ollama`` (client, corpus from a prompt, adversarial review), its API endpoints and CLI.

A fake Ollama server (standard library) stands in for the real one: it answers
``/api/tags``, ``/api/generate`` (numbered lines for corpus prompts, JSON
ratings for review prompts) and ``/api/chat``, records every request and can
be switched into failure modes.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet, ollama  # noqa: E402
from radixnet.ollama import (  # noqa: E402
    OllamaClient,
    OllamaError,
    _loads_lenient,
    _parse_reviews,
    adversarial_review,
    corpus_from_prompt,
    normalise_url,
    parse_lines,
    review_texts,
)

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import CORPUS, start_server  # noqa: F401
except ImportError:
    from tests.test_api import CORPUS, start_server  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"epochs": 3, "lr": 0.5, "batch_size": 4}


# ---------------------------------------------------------------------------
# fake Ollama
# ---------------------------------------------------------------------------


class _FakeHandler(BaseHTTPRequestHandler):
    server: "FakeOllama"

    def log_message(self, *args):  # silence
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [
                {"name": "fake:latest", "size": 123, "modified_at": "2026-01-01T00:00:00Z", "details": {"family": "fake"}},
                {"name": "other:7b", "size": 456, "modified_at": "2026-01-02T00:00:00Z", "details": {}},
            ]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(("POST", self.path, body))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
            return
        if self.server.raw_response is not None:
            self._raw(200, self.server.raw_response)
            return
        if self.path == "/api/generate":
            self._json(200, {"model": body.get("model"), "response": self._generate(body), "done": True})
        elif self.path == "/api/chat":
            self._json(200, {"model": body.get("model"), "message": {"role": "assistant", "content": "chat reply"}})
        else:
            self._json(404, {"error": "not found"})

    def _generate(self, body):
        prompt = body.get("prompt", "")
        system = body.get("system", "")
        if body.get("format") == "json":  # a review request
            if self.server.review_response is not None:
                return self.server.review_response
            reviews = []
            for line in prompt.splitlines():
                match = re.match(r"\[(\d+)\] (.*)", line)
                if match:
                    index, text = int(match.group(1)), match.group(2)
                    rating = 9 if "good" in text.lower() else 2
                    reviews.append({"index": index, "rating": rating, "verdict": "pass" if rating >= 6 else "fail",
                                    "critique": "fine" if rating >= 6 else "nonsense"})
            return json.dumps({"reviews": reviews})
        match = re.search(r"exactly (\d+) lines", system)
        count = int(match.group(1)) if match else 5
        style = "garbage" if "WRONG" in system else "good"
        topic = "unknown"
        match = re.search(r"instructions: (.*)", prompt)
        if match:
            topic = match.group(1).strip()
        lines = [f"{i + 1}. {style} line {i + 1} about {topic}" for i in range(count)]
        lines.insert(1, "")  # blank line to drop
        lines.append(f"{count + 1}. {style} line 1 about {topic}")  # duplicate (differs only by numbering)
        return "\n".join(lines)


class FakeOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _FakeHandler)
        self.requests = []
        self.fail_with = None
        self.raw_response = None
        self.review_response = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_fake(add_cleanup):
    server = FakeOllama()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    add_cleanup(server.server_close)
    add_cleanup(server.shutdown)
    return server


def closed_port_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def wait_job(client, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job, _ = client.get("/api/job")
        assert status == 200, (status, job)
        if job and job.get("state") != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish in time")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)

    def test_normalise_url(self):
        self.assertEqual(normalise_url("localhost:11434"), "http://localhost:11434")
        self.assertEqual(normalise_url("https://box:1/"), "https://box:1")
        self.assertEqual(normalise_url("  http://a  "), "http://a")
        with self.assertRaises(ValueError):
            normalise_url("")
        self.assertTrue(ollama.DEFAULT_URL.startswith("http"))
        self.assertTrue(ollama.DEFAULT_MODEL)

    def test_models_and_available(self):
        client = OllamaClient(self.fake.url, "fake:latest")
        self.assertTrue(client.available())
        self.assertEqual([m["name"] for m in client.models()], ["fake:latest", "other:7b"])
        down = OllamaClient(closed_port_url(), "fake:latest", timeout=2)
        self.assertFalse(down.available())
        with self.assertRaises(OllamaError) as ctx:
            down.models()
        self.assertIn("cannot reach Ollama", str(ctx.exception))

    def test_generate_and_chat(self):
        client = OllamaClient(self.fake.url, "fake:latest")
        text = client.generate("hello", system="exactly 2 lines", model="other:7b", options={"temperature": 0.1})
        self.assertIn("good line 1", text)
        method, path, body = self.fake.requests[-1]
        self.assertEqual((method, path, body["model"], body["stream"]), ("POST", "/api/generate", "other:7b", False))
        self.assertEqual(body["options"], {"temperature": 0.1})
        self.assertNotIn("format", body)
        client.generate("hello", json_mode=True)
        self.assertEqual(self.fake.requests[-1][2]["format"], "json")
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "chat reply")
        self.assertEqual(self.fake.requests[-1][1], "/api/chat")

    def test_errors_are_ollama_errors(self):
        client = OllamaClient(self.fake.url, "fake:latest")
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError) as ctx:
            client.generate("x")
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))
        self.fake.fail_with = None
        self.fake.raw_response = b"this is not json"
        with self.assertRaises(OllamaError) as ctx:
            client.generate("x")
        self.assertIn("invalid JSON", str(ctx.exception))


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


class ParsingTests(unittest.TestCase):
    def test_parse_lines(self):
        text = "1. The cat sat.\n\n- The dog ran.\n* \"Quoted line\"\n2) the cat sat.\n```\n(3) Third one\nThird one\n"
        self.assertEqual(parse_lines(text), ["The cat sat.", "The dog ran.", "Quoted line", "Third one"])
        self.assertEqual(parse_lines(text, limit=2), ["The cat sat.", "The dog ran."])
        self.assertEqual(parse_lines(""), [])

    def test_loads_lenient(self):
        self.assertEqual(_loads_lenient('{"a": 1}'), {"a": 1})
        self.assertEqual(_loads_lenient('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_loads_lenient('Sure! Here it is: {"a": [1, 2]} hope it helps'), {"a": [1, 2]})
        self.assertEqual(_loads_lenient("[1, 2]"), [1, 2])
        self.assertIsNone(_loads_lenient("no json here"))

    def test_parse_reviews(self):
        raw = json.dumps({"reviews": [
            {"index": 0, "rating": 8, "verdict": "pass", "critique": "ok"},
            {"index": 1, "rating": "3.5", "critique": ""},
            {"index": 2, "rating": 42},
            {"index": 7, "rating": 5},
            {"index": "x", "score": 4, "reason": "meh"},
            "junk",
            {"index": 3, "rating": "n/a"},
        ]})
        parsed = _parse_reviews(raw, 5)
        self.assertEqual(parsed[0], {"rating": 8.0, "critique": "ok"})
        self.assertEqual(parsed[1], {"rating": 3.5, "critique": "no critique given"})
        self.assertEqual(parsed[2]["rating"], 10.0)  # clamped
        self.assertNotIn(7, parsed)  # out of range
        self.assertEqual(parsed[4], {"rating": 4.0, "critique": "meh"})  # position used when the index is unusable
        self.assertNotIn(3, parsed)  # unusable rating
        self.assertEqual(_parse_reviews("garbage", 3), {})
        self.assertEqual(_parse_reviews('[{"rating": 7}]', 1), {0: {"rating": 7.0, "critique": "no critique given"}})
        self.assertEqual(_parse_reviews('{"rating": 2, "critique": "bad"}', 1), {0: {"rating": 2.0, "critique": "bad"}})


# ---------------------------------------------------------------------------
# corpus and review against the fake server
# ---------------------------------------------------------------------------


class CorpusAndReviewTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest")

    def test_corpus_from_prompt(self):
        lines = corpus_from_prompt(self.client, "cats and dogs", lines=4)
        self.assertEqual(lines, [f"good line {i} about cats and dogs" for i in range(1, 5)])
        body = self.fake.requests[-1][2]
        self.assertIn("exactly 4 lines", body["system"])
        self.assertNotIn("WRONG", body["system"])
        self.assertIn("cats and dogs", body["prompt"])

        garbage = corpus_from_prompt(self.client, "cats and dogs", lines=3, style="garbage", model="other:7b")
        self.assertEqual(garbage, [f"garbage line {i} about cats and dogs" for i in range(1, 4)])
        body = self.fake.requests[-1][2]
        self.assertIn("WRONG", body["system"])
        self.assertEqual(body["model"], "other:7b")

        with self.assertRaises(ValueError):
            corpus_from_prompt(self.client, "", lines=3)
        with self.assertRaises(ValueError):
            corpus_from_prompt(self.client, "x", lines=0)
        with self.assertRaises(ValueError):
            corpus_from_prompt(self.client, "x", style="weird")

    def test_review_texts(self):
        reviews = review_texts(self.client, ["a good sentence", "zzzz", "   ", "another good one"], threshold=6)
        self.assertEqual([r["index"] for r in reviews], [0, 1, 2, 3])
        self.assertEqual([r["verdict"] for r in reviews], ["pass", "fail", "fail", "pass"])
        self.assertEqual([r["rating"] for r in reviews], [9.0, 2.0, 0.0, 9.0])
        self.assertEqual(reviews[2]["critique"], "empty output")
        body = self.fake.requests[-1][2]
        self.assertEqual(body["format"], "json")
        self.assertIn("[0] a good sentence", body["prompt"])
        self.assertNotIn("[2]", body["prompt"])  # blank texts are not sent
        self.assertIn("rating of 6 or more", body["system"])

    def test_review_threshold_and_context(self):
        reviews = review_texts(self.client, ["a good sentence"], threshold=9.5, context="trained on nursery rhymes")
        self.assertEqual(reviews[0]["verdict"], "fail")  # 9 < 9.5: the verdict follows the rating, not the LLM
        self.assertIn("Context: trained on nursery rhymes", self.fake.requests[-1][2]["prompt"])

    def test_review_batches_and_unrated(self):
        texts = [f"good {i}" for i in range(45)]
        reviews = review_texts(self.client, texts, batch=20)
        self.assertEqual(len(reviews), 45)
        self.assertEqual(len([r for r in self.fake.requests if r[1] == "/api/generate"]), 3)
        self.assertTrue(all(r["verdict"] == "pass" for r in reviews))
        self.fake.review_response = "I cannot rate these."
        reviews = review_texts(self.client, ["good one", "bad one"])
        self.assertEqual([r["verdict"] for r in reviews], ["unrated", "unrated"])
        self.assertTrue(all(r["rating"] is None for r in reviews))
        self.assertEqual(reviews[0]["critique"], "no review returned")

    def test_adversarial_review_of_model_samples(self):
        model = RadixNet(seed=1, backend="python")
        model.train(["a good sentence about cats", "a good sentence about dogs"], **FAST)
        result = adversarial_review(model, self.client, count=3, max_length=40, seed=3)
        self.assertEqual((result["source"], result["model"], result["threshold"]), ("model", "fake:latest", 6.0))
        self.assertEqual(len(result["texts"]), 3)
        self.assertEqual(len(result["reviews"]), 3)
        self.assertEqual(sorted(result["good"] + result["bad"]), sorted(result["texts"]))
        self.assertIsNotNone(result["pass_rate"])
        with_prefix = adversarial_review(model, self.client, count=2, prefix="a good", max_length=30)
        self.assertTrue(all(t.startswith("a good") for t in with_prefix["texts"]))

    def test_adversarial_review_of_given_texts(self):
        result = adversarial_review(None, self.client, texts=["good one", "nonsense zzz", "good two"], threshold=6)
        self.assertEqual(result["source"], "given")
        self.assertEqual(result["good"], ["good one", "good two"])
        self.assertEqual(result["bad"], ["nonsense zzz"])
        self.assertAlmostEqual(result["mean_rating"], (9 + 2 + 9) / 3)
        self.assertAlmostEqual(result["pass_rate"], 2 / 3)
        with self.assertRaises(ValueError):
            adversarial_review(None, self.client)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-ollama-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(
            self.addCleanup, upload_dir=self.dir, ollama_url=self.fake.url, ollama_model="fake:latest",
        )

    def test_status_and_models(self):
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["ollama"], {"url": self.fake.url, "model": "fake:latest"})
        status, data, _ = self.client.get("/api/ollama/models")
        self.assertEqual(status, 200, data)
        self.assertTrue(data["available"])
        self.assertIsNone(data["error"])
        self.assertEqual((data["url"], data["model"]), (self.fake.url, "fake:latest"))
        self.assertEqual([m["name"] for m in data["models"]], ["fake:latest", "other:7b"])
        self.assertEqual(set(data["models"][0]), {"name", "size", "modified_at", "details"})

        down = closed_port_url()
        status, data, _ = self.client.get("/api/ollama/models?url=" + down)
        self.assertEqual(status, 200, data)
        self.assertFalse(data["available"])
        self.assertEqual(data["models"], [])
        self.assertIn("cannot reach Ollama", data["error"])
        self.assertEqual(data["url"], down)

    def test_corpus(self):
        status, data, _ = self.client.post("/api/ollama/corpus", {"prompt": "the sea", "lines": 3})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["texts"], [f"good line {i} about the sea" for i in range(1, 4)])
        self.assertEqual((data["style"], data["model"], data["lines"], data["upload"], data["job"]), ("good", "fake:latest", 3, None, None))

        body = {"prompt": "the sea", "lines": 2, "style": "garbage", "model": "other:7b", "save_as": "sea-garbage.txt"}
        status, data, _ = self.client.post("/api/ollama/corpus", body)
        self.assertEqual(status, 200, data)
        self.assertEqual(data["texts"], ["garbage line 1 about the sea", "garbage line 2 about the sea"])
        self.assertEqual(data["upload"]["name"], "sea-garbage.txt")
        self.assertEqual(data["upload"]["lines"], 2)
        status, uploads, _ = self.client.get("/api/uploads")
        self.assertEqual([u["name"] for u in uploads["uploads"]], ["sea-garbage.txt"])

        body = {"prompt": "the sea", "lines": 4, "train": True, "epochs": 2, "lr": 0.5, "batch_size": 4}
        status, data, _ = self.client.post("/api/ollama/corpus", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(data["job"]["type"], "train")
        self.assertEqual(wait_job(self.client)["state"], "done")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 4)

    def test_corpus_errors(self):
        status, data, _ = self.client.post("/api/ollama/corpus", {"lines": 3})
        self.assertEqual(status, 400, data)
        self.assertIn("prompt", data["error"])
        status, data, _ = self.client.post("/api/ollama/corpus", {"prompt": "x", "style": "weird"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/ollama/corpus", {"prompt": "x", "url": "   "})
        self.assertEqual(status, 400, data)
        self.fake.fail_with = 500
        status, data, _ = self.client.post("/api/ollama/corpus", {"prompt": "x"})
        self.assertEqual(status, 502, data)
        self.assertIn("HTTP 500", data["error"])
        self.fake.fail_with = None
        status, data, _ = self.client.post("/api/ollama/corpus", {"prompt": "x", "url": closed_port_url(), "timeout": 2})
        self.assertEqual(status, 502, data)
        self.assertIn("cannot reach Ollama", data["error"])

    def test_review_given_texts_and_apply_two_nrl(self):
        body = {"texts": ["a good one", "zzz nonsense", "a good two"], "threshold": 6}
        status, data, _ = self.client.post("/api/ollama/review", body)
        self.assertEqual(status, 200, data)
        self.assertEqual(data["source"], "given")
        self.assertEqual(data["good"], ["a good one", "a good two"])
        self.assertEqual(data["bad"], ["zzz nonsense"])
        self.assertEqual([r["verdict"] for r in data["reviews"]], ["pass", "fail", "pass"])
        self.assertEqual(set(data["reviews"][0]), {"index", "text", "rating", "verdict", "critique"})
        self.assertIsNone(data["job"])
        self.assertEqual(data["url"], self.fake.url)

        self.client.post("/api/uploads", {"name": "good.txt", "content": "\n".join(CORPUS[:3]) + "\n"})
        body.update(apply="2nrl", good=["one more good line"], good_files=["good.txt"], neg_epochs=1, pos_epochs=1, batch_size=4)
        status, data, _ = self.client.post("/api/ollama/review", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(data["job"]["type"], "2nrl")
        self.assertEqual(wait_job(self.client)["state"], "done")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["twonrl_runs"], 1)
        self.assertEqual(stats["trained_texts"], 1 + 2 + 1 + 3)  # bad + passed + given good + upload

    def test_review_model_samples(self):
        self.client.post("/api/train", {"texts": ["a good sentence about cats", "a good sentence about dogs"], **FAST})
        wait_job(self.client)
        status, data, _ = self.client.post("/api/ollama/review", {"count": 3, "max_length": 40})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["source"], "model")
        self.assertEqual(len(data["reviews"]), 3)
        status, data, _ = self.client.post("/api/ollama/review", {"count": 2, "prefix": "a good", "max_length": 30})
        self.assertEqual(status, 200, data)
        self.assertTrue(all(t.startswith("a good") for t in data["texts"]))

    def test_review_errors(self):
        status, data, _ = self.client.post("/api/ollama/review", {"texts": ["x"], "apply": "maybe"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/ollama/review", {"texts": ["all good", "still good"], "apply": "2nrl"})
        self.assertEqual(status, 400, data)
        self.assertIn("nothing failed", data["error"])
        status, data, _ = self.client.post("/api/ollama/review", {"texts": ["zzz", "qqq"], "apply": "2nrl"})
        self.assertEqual(status, 400, data)
        self.assertIn("no text passed", data["error"])
        status, data, _ = self.client.post("/api/ollama/review", {"count": 0})
        self.assertEqual(status, 400, data)
        self.fake.fail_with = 503
        status, data, _ = self.client.post("/api/ollama/review", {"texts": ["x"]})
        self.assertEqual(status, 502, data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-ollama-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")

    def run_cli(self, *args, expect=0):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        env = dict(os.environ, PYTHONWARNINGS="ignore", OLLAMA_HOST=self.fake.url, RADIXNET_OLLAMA_MODEL="fake:latest")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(proc.returncode, expect, proc.stderr[-2000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_models(self):
        doc = self.run_cli("ollama", "models")
        self.assertEqual((doc["url"], doc["model"]), (self.fake.url, "fake:latest"))
        self.assertEqual([m["name"] for m in doc["models"]], ["fake:latest", "other:7b"])
        doc = self.run_cli("ollama", "--url", closed_port_url(), "--timeout", "2", "models", expect=1)
        self.assertIsNone(doc)

    def test_corpus_out_and_train(self):
        out = os.path.join(self.dir, "corpus.txt")
        doc = self.run_cli("ollama", "--ollama-model", "other:7b", "corpus", "--prompt", "mountains", "--lines", "3",
                           "--style", "garbage", "--out", out)
        self.assertEqual(doc["texts"], [f"garbage line {i} about mountains" for i in range(1, 4)])
        self.assertEqual((doc["model"], doc["out"], doc["trained"]), ("other:7b", out, None))
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(fh.read().splitlines(), doc["texts"])

        doc = self.run_cli("ollama", "corpus", "--prompt", "mountains", "--lines", "4", "--train", "--epochs", "2")
        self.assertEqual(doc["trained"]["stats"]["trained_texts"], 4)
        self.assertEqual(len(doc["trained"]["epochs"]), 2)
        self.assertTrue(os.path.isfile(self.model))

    def test_review_and_two_nrl(self):
        doc = self.run_cli("ollama", "review", "--text", "a good line", "--text", "zzz", "--threshold", "6")
        self.assertEqual(doc["source"], "given")
        self.assertEqual((doc["good"], doc["bad"], doc["two_nrl"]), (["a good line"], ["zzz"], None))
        # sampling needs a trained model
        self.run_cli("ollama", "review", "--count", "2", expect=1)
        self.run_cli("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", "1",
                     "--lr", "0.5", "--batch-size", "4")
        doc = self.run_cli("ollama", "review", "--count", "2", "--max-length", "30")
        self.assertEqual(doc["source"], "model")
        self.assertEqual(len(doc["reviews"]), 2)
        good = os.path.join(self.dir, "good.txt")
        with open(good, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\n")
        doc = self.run_cli("ollama", "review", "--text", "zzz nonsense", "--text", "good text", "--2nrl", "--good", good,
                           "--neg-epochs", "1", "--pos-epochs", "1")
        self.assertEqual(doc["two_nrl"]["bad_texts"], 1)
        self.assertEqual(doc["two_nrl"]["good_texts"], 2)
        self.assertEqual(doc["two_nrl"]["stats"]["twonrl_runs"], 1)
        self.run_cli("ollama", "review", "--text", "good only", "--2nrl", expect=1)


if __name__ == "__main__":
    unittest.main()
