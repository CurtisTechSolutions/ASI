"""Cross-language parity of the Go port of the count / reward model (``go/``).

Skipped when no Go toolchain is available.  The Go CLI is built once into a
temporary directory; then both implementations train the same corpus with the
same settings and must produce the same structure, counts, rewards, sliding
window and RNG state, the same predictions, generated texts, scores and
conversation, and each side must load and continue the other's model file.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.model import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GO_DIR = os.path.join(ROOT, "go")
CORPUS = os.path.join(ROOT, "data", "sample_corpus.txt")
GARBAGE = os.path.join(ROOT, "data", "sample_garbage.txt")
GO = shutil.which("go")

TMP = None
BINARY = None


def setUpModule():
    global TMP, BINARY
    if GO is None:
        raise unittest.SkipTest("no Go toolchain on PATH")
    TMP = tempfile.TemporaryDirectory(prefix="radixnet-go-")
    BINARY = os.path.join(TMP.name, "radixnet-count")
    env = {**os.environ, "GOFLAGS": os.environ.get("GOFLAGS", "-mod=mod"), "GOCACHE": os.environ.get("GOCACHE", os.path.join(TMP.name, "gocache"))}
    proc = subprocess.run([GO, "build", "-o", BINARY, "./cmd/radixnet-count"], cwd=GO_DIR, capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        raise unittest.SkipTest(f"go build failed:\n{proc.stderr}")


def tearDownModule():
    if TMP is not None:
        TMP.cleanup()


def go(*args, model, expect=0):
    cmd = [BINARY, "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def py(*args, model):
    cmd = [sys.executable, "-m", "radixnet", "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def assert_close(test, a, b, tol=1e-9):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        test.assertLessEqual(abs(x - y), tol, (x, y))


class TestGoParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py_model = os.path.join(TMP.name, "py.count.json")
        cls.go_model = os.path.join(TMP.name, "go.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.py_model)
        cls.go_train = go("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, "--workers", 4, model=cls.go_model)
        cls.py_doc = load_json(cls.py_model)
        cls.go_doc = load_json(cls.go_model)

    def test_same_graph_after_training(self):
        p, g = self.py_doc["graph"], self.go_doc["graph"]
        self.assertEqual(p["nodes"]["labels"], g["nodes"]["labels"])
        self.assertEqual(p["nodes"]["count"], g["nodes"]["count"])
        self.assertEqual(p["nodes"]["a"], g["nodes"]["a"])
        self.assertEqual(p["nodes"]["k"], g["nodes"]["k"])
        self.assertEqual((p["edges"]["src"], p["edges"]["dst"]), (g["edges"]["src"], g["edges"]["dst"]))
        self.assertEqual(p["edges"]["count"], g["edges"]["count"])
        self.assertEqual(p["edges"]["reward"], g["edges"]["reward"])
        assert_close(self, p["edges"]["w"], g["edges"]["w"], 1e-12)
        self.assertEqual(p["rng_state"], g["rng_state"])
        self.assertEqual(p["inverted"], g["inverted"])
        pw, gw = p["weights"], g["weights"]
        self.assertEqual(pw["window_events"], gw["window_events"])
        for key in ("function", "count_scale", "global_scale", "window_scale", "reward_scale", "window", "smoothing", "kind", "total_traversals"):
            self.assertEqual(pw[key], gw[key], key)
        self.assertEqual(self.go_doc["format"], "radixnet-count")
        self.assertEqual(self.go_doc["kind"], "count")
        for key in ("epochs_total", "trained_texts", "trained_chars", "twonrl_runs", "feedback_passes"):
            self.assertEqual(self.py_doc["meta"][key], self.go_doc["meta"][key], key)
        self.assertEqual(len(self.py_doc["history"]), len(self.go_doc["history"]))
        for a, b in zip(self.py_doc["history"], self.go_doc["history"]):
            self.assertEqual((a["epoch"], a["nodes"], a["edges"], a["transitions"], a["merges"]), (b["epoch"], b["nodes"], b["edges"], b["transitions"], b["merges"]))
            self.assertLessEqual(abs(a["loss"] - b["loss"]), 1e-9)
        self.assertEqual(self.go_train["workers"], 4)

    def test_same_predictions_generation_scores_and_conversation(self):
        for prefix in ("the cat", "the", "on the ma", "zzq", ""):
            with self.subTest(prefix=prefix):
                a = py("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam", model=self.py_model)
                b = go("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam", model=self.go_model)
                self.assertEqual(a["continuation"], b["continuation"])
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                self.assertEqual([t["full_text"] for t in a["bottom"]], [t["full_text"] for t in b["bottom"]])
                assert_close(self, [t["cost"] for t in a["top"]], [t["cost"] for t in b["top"]])
                self.assertEqual(a["path"], b["path"])
        a = py("generate", "--mode", "beam", "--count", 4, "--max-length", 40, model=self.py_model)
        b = go("generate", "--mode", "beam", "--count", 4, "--max-length", 40, model=self.go_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        assert_close(self, [s["cost"] for s in a["samples"]], [s["cost"] for s in b["samples"]])
        a = py("generate", "--mode", "beam", "--count", 2, "--prefix", "the ", "--max-length", 20, model=self.py_model)
        b = go("generate", "--mode", "beam", "--count", 2, "--prefix", "the ", "--max-length", 20, model=self.go_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        a = py("score", "--data", GARBAGE, model=self.py_model)
        b = go("score", "--data", GARBAGE, model=self.go_model)
        self.assertEqual(a["count"], b["count"])
        assert_close(self, [r["log_prob"] for r in a["results"]], [r["log_prob"] for r in b["results"]])
        self.assertEqual([r["unknown_transitions"] for r in a["results"]], [r["unknown_transitions"] for r in b["results"]])
        self.assertLessEqual(abs(a["mean_per_char"] - b["mean_per_char"]), 1e-9)
        a = py("converse", "--opening", "the cat sat on the mat", "--turns", 6, model=self.py_model)
        b = go("converse", "--opening", "the cat sat on the mat", "--turns", 6, model=self.go_model)
        self.assertEqual(a["transcript"], b["transcript"])
        self.assertEqual([t["context"] for t in a["turns"]], [t["context"] for t in b["turns"]])
        self.assertEqual([t["fresh"] for t in a["turns"]], [t["fresh"] for t in b["turns"]])

    def test_each_side_loads_and_continues_the_other(self):
        # Python loads the Go file: same predictions as its own model
        loaded = load_model(self.go_model)
        self.assertIsInstance(loaded, CountRewardNet)
        own = load_model(self.py_model)
        for prefix in ("the cat", "the dog"):
            a = own.predict(prefix, length=8, k=3)
            b = loaded.predict(prefix, length=8, k=3)
            self.assertEqual([r.full_text for r in a.top], [r.full_text for r in b.top])
            self.assertLessEqual(abs(a.cost - b.cost), 1e-9)
        self.assertEqual(loaded.stats()["total_traversals"], own.stats()["total_traversals"])
        # Go loads the Python file
        info = go("info", model=self.py_model)
        self.assertEqual((info["stats"]["nodes"], info["stats"]["edges"], info["stats"]["epochs_total"]), (own.stats()["nodes"], own.stats()["edges"], 2))
        # both continue training (one more epoch) on the other's file and stay identical
        cross_py = os.path.join(TMP.name, "cross_py.count.json")
        cross_go = os.path.join(TMP.name, "cross_go.count.json")
        shutil.copy(self.go_model, cross_py)
        shutil.copy(self.py_model, cross_go)
        py("train", "--data", CORPUS, "--epochs", 1, model=cross_py)   # Python continues the Go model
        go("train", "--data", CORPUS, "--epochs", 1, "--workers", 3, model=cross_go)  # Go continues the Python model
        a, b = load_json(cross_py)["graph"], load_json(cross_go)["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])
        self.assertEqual(a["edges"]["count"], b["edges"]["count"])
        self.assertEqual(a["rng_state"], b["rng_state"])
        self.assertEqual(a["weights"]["window_events"], b["weights"]["window_events"])
        self.assertEqual(a["weights"]["total_traversals"], b["weights"]["total_traversals"])
        assert_close(self, a["edges"]["w"], b["edges"]["w"], 1e-12)

    def test_feedback_2nrl_and_invert_match(self):
        py_path = os.path.join(TMP.name, "fb_py.count.json")
        go_path = os.path.join(TMP.name, "fb_go.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.go_model, go_path)
        py("feedback", "--good-text", "the cat sat on the mat", "--bad-text", "the dog ate the bone", "--neg-epochs", 2, "--pos-epochs", 1, "--strength", 0.5, model=py_path)
        b = go("feedback", "--good-text", "the cat sat on the mat", "--bad-text", "the dog ate the bone", "--neg-epochs", 2, "--pos-epochs", 1, "--strength", 0.5, model=go_path)
        self.assertEqual(b["action"], "2nrl")
        a_doc, b_doc = load_json(py_path)["graph"], load_json(go_path)["graph"]
        self.assertEqual(a_doc["edges"]["reward"], b_doc["edges"]["reward"])
        self.assertEqual(a_doc["edges"]["count"], b_doc["edges"]["count"])
        assert_close(self, a_doc["edges"]["w"], b_doc["edges"]["w"], 1e-12)
        a_stats = py("info", model=py_path)["stats"]
        b_stats = go("info", model=go_path)["stats"]
        for key in ("twonrl_runs", "feedback_passes", "total_traversals", "window_traversals"):
            self.assertEqual(a_stats[key], b_stats[key], key)
        for key in ("rewards_total", "penalties_total", "edge_reward_positive", "edge_reward_negative"):
            self.assertLessEqual(abs(a_stats[key] - b_stats[key]), 1e-9, key)
        py("invert", model=py_path)
        go("invert", model=go_path)
        self.assertEqual(load_json(py_path)["graph"]["edges"]["reward"], load_json(go_path)["graph"]["edges"]["reward"])
        self.assertTrue(load_json(go_path)["graph"]["inverted"])
        py("weights", "--global-scale", 0.7, "--window", 500, model=py_path)
        b = go("weights", "--global-scale", 0.7, "--window", 500, model=go_path)
        self.assertEqual((b["weights"]["global_scale"], b["weights"]["window"]), (0.7, 500))
        a_doc, b_doc = load_json(py_path)["graph"], load_json(go_path)["graph"]
        assert_close(self, a_doc["edges"]["w"], b_doc["edges"]["w"], 1e-12)
        self.assertEqual(a_doc["weights"]["window_events"], b_doc["weights"]["window_events"])

    def test_go_specific_options(self):
        path = os.path.join(TMP.name, "paras.count.json")
        doc = go("--seed", 2, "train", "--data", CORPUS, "--epochs", 1, "--split", "paragraphs", "--workers", 2, model=path)
        self.assertEqual(doc["split"], "paragraphs")
        self.assertGreaterEqual(doc["texts"], 1)
        loaded = load_model(path)  # a paragraph-trained Go model loads in Python
        self.assertEqual(loaded.kind, "count")
        self.assertEqual(loaded.stats()["epochs_total"], 1)
        go("predict", "--prefix", "the", "--mode", "nope", model=path, expect=1)
        go("train", model=path, expect=1)  # --data is required


class TestGoServer(unittest.TestCase):
    """`radixnet-count serve` speaks the Python server's JSON contract for the count model: the frontend's
    requests and the shapes the Python API tests assert on must be served the same way."""

    @classmethod
    def setUpClass(cls):
        import socket
        import time
        from tests.test_api import Client

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-go-server-")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.model = os.path.join(cls.tmp.name, "model.count.json")
        cls.proc = subprocess.Popen(
            [BINARY, "--model", cls.model, "--seed", "1", "--workers", "2", "serve", "--port", str(cls.port),
             "--frontend-dir", os.path.join(ROOT, "frontend", "dist"), "--upload-dir", os.path.join(cls.tmp.name, "uploads"),
             "--checkpoint-dir", os.path.join(cls.tmp.name, "ckpt"), "--quiet"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        cls.client = Client(f"http://127.0.0.1:{cls.port}")
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                status, doc, _ = cls.client.get("/api/health")
                if status == 200 and doc.get("ok"):
                    break
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if cls.proc.poll() is not None:
                raise AssertionError("the Go server exited early:\n" + cls.proc.stderr.read())
            time.sleep(0.05)
        else:
            cls.proc.kill()
            raise AssertionError("the Go server did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.tmp.cleanup()

    def wait_job(self):
        import time

        deadline = time.time() + 30
        while time.time() < deadline:
            status, job, _ = self.client.get("/api/job")
            self.assertEqual(status, 200)
            if job and job["state"] != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    def test_contract_matches_the_python_api(self):
        from tests.test_api import CORPUS, EDGE_KEYS, JOB_KEYS, NODE_KEYS, PREDICT_KEYS, STATS_KEYS

        status, health, _ = self.client.get("/api/health")
        self.assertEqual((status, health["ok"], health["engine"]), (200, True, "go"))
        status, st, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(STATS_KEYS <= set(st), STATS_KEYS - set(st))
        self.assertEqual((st["kind"], st["engine"], st["job"]), ("count", "go", None))
        self.assertEqual([k["kind"] for k in st["kinds"]], ["count"])
        # the frontend's Train tab: a job, polled until done
        status, doc, _ = self.client.post("/api/train", {"texts": CORPUS, "epochs": 2, "lr": 0.5, "act_lr": 0.05, "batch_size": 4, "auto_compress": True})
        self.assertEqual(status, 202, doc)
        self.assertTrue(JOB_KEYS <= set(doc["job"]))
        job = self.wait_job()
        self.assertEqual((job["type"], job["state"], len(job["history"])), ("train", "done", 2))
        status, st, _ = self.client.get("/api/status")
        self.assertEqual((st["epochs_total"], st["trained_texts"]), (2, len(CORPUS)))
        # the Predict tab (beam with K / beam width, plus the Like button's feedback)
        status, p, _ = self.client.post("/api/predict", {"prefix": "the cat", "length": 8, "mode": "beam", "k": 3, "beam": 12})
        self.assertEqual(status, 200, p)
        self.assertTrue(PREDICT_KEYS <= set(p), PREDICT_KEYS - set(p))
        self.assertTrue({"top", "bottom", "k", "beam", "mode"} <= set(p))
        self.assertEqual(p["full_text"], "the cat" + p["continuation"])
        # the same model file, loaded in Python, predicts the same thing
        status, saved, _ = self.client.post("/api/save", {})
        self.assertEqual(status, 200, saved)
        own = load_model(saved["path"])
        mine = own.predict("the cat", length=8, mode="beam", k=3, beam=12)
        self.assertEqual([t["full_text"] for t in p["top"]], [r.full_text for r in mine.top])
        self.assertLessEqual(abs(p["cost"] - mine.cost), 1e-9)
        # Generate / Score / Converse tabs
        status, g, _ = self.client.post("/api/generate", {"count": 3, "mode": "beam", "max_length": 40})
        self.assertEqual((status, len(g["samples"])), (200, 3))
        self.assertTrue({"text", "full_text", "cost", "probability", "path", "reached_end"} <= set(g["samples"][0]))
        status, sc, _ = self.client.post("/api/score", {"text": CORPUS[0]})
        self.assertEqual(status, 200)
        self.assertEqual(set(sc), {"log_prob", "per_char", "chars", "transitions", "unknown_transitions"})
        status, c, _ = self.client.post("/api/converse", {"opening": CORPUS[0], "turns": 3})
        self.assertEqual((status, c["count"], c["kind"]), (200, 4, "count"))
        # thumbs on the Generate tab: a feedback job
        status, fb, _ = self.client.post("/api/feedback", {"good": [g["samples"][0]["text"]], "bad": [g["samples"][1]["text"]], "neg_epochs": 2, "pos_epochs": 3, "neg_lr": 0.5, "pos_lr": 0.1, "strength": 1})
        self.assertEqual((status, fb["action"]), (202, "2nrl"), fb)
        job = self.wait_job()
        self.assertEqual((job["type"], job["state"]), ("feedback", "done"))
        # Graph tab
        status, graph, _ = self.client.get("/api/graph?limit=5")
        self.assertEqual(status, 200)
        self.assertTrue(NODE_KEYS <= set(graph["nodes"][0]))
        self.assertTrue((EDGE_KEYS | {"reward", "share", "recent_share", "recent_count"}) <= set(graph["edges"][0]))
        # model selector: the one kind, other kinds refused
        status, m, _ = self.client.get("/api/model")
        self.assertEqual((status, m["kind"], m["in_memory"]), (200, "count", ["count"]))
        status, doc, _ = self.client.post("/api/model/select", {"kind": "radix"})
        self.assertEqual(status, 400)
        # Python-only tabs are told so; unknown endpoints and wrong methods behave like the Python server
        status, doc, _ = self.client.get("/api/evolve/history")
        self.assertEqual(status, 404)
        self.assertIn("Go server", doc["error"])
        status, doc, _ = self.client.get("/api/nope")
        self.assertEqual(status, 404)
        status, doc, headers = self.client.get("/api/predict")
        self.assertEqual(status, 405)
        self.assertIn("POST", headers.get("Allow", ""))
        # the prebuilt frontend is served with the SPA fallback
        status, body, headers = self.client.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        status, body, _ = self.client.request("GET", "/some/route")
        self.assertEqual(status, 200)

    def test_uploads_checkpoints_and_split(self):
        import zipfile
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a/one.txt", "the cat sat on the mat\nthe cat ran to the door\n")
            z.writestr("two.txt", "the dog sat on the log\n")
            z.writestr("img.png", b"\x89PNG\x00\x00")
        status, doc, _ = self.client.request(
            "POST", "/api/uploads?name=corpus.zip", raw=buf.getvalue(), headers={"Content-Type": "application/zip"},
        )
        self.assertEqual(status, 201, doc)
        record = doc["uploads"][0]
        self.assertEqual((record["name"], record["archive"], record["files"], record["lines"]), ("corpus.zip", True, 2, 3))
        status, doc, _ = self.client.post("/api/uploads", {"name": "notes.txt", "content": "para one line a\npara one line b\n\npara two\n"})
        self.assertEqual(status, 201, doc)
        status, listing, _ = self.client.get("/api/uploads")
        self.assertEqual([u["name"] for u in listing["uploads"]], ["corpus.zip", "notes.txt"])
        status, before, _ = self.client.get("/api/status")
        # goroutines over paragraphs: the pasted text and the files are cut into paragraphs on the server
        status, doc, _ = self.client.post("/api/train", {"text": "first para\ncontinues\n\nsecond para\n", "files": ["notes.txt"], "split": "paragraphs", "epochs": 1})
        self.assertEqual(status, 202, doc)
        self.wait_job()
        status, after, _ = self.client.get("/api/status")
        self.assertEqual(after["trained_texts"] - before["trained_texts"], 4)
        status, doc, _ = self.client.post("/api/train", {"files": ["corpus.zip"], "epochs": 1})
        self.assertEqual(status, 202, doc)
        self.wait_job()
        status, later, _ = self.client.get("/api/status")
        self.assertEqual(later["trained_texts"] - after["trained_texts"], 3)
        # checkpoints in the Python manager's layout
        status, ck, _ = self.client.post("/api/checkpoints/save", {"tag": "manual"})
        self.assertEqual(status, 200, ck)
        self.assertTrue(ck["name"].startswith("ckpt-manual-") and ck["name"].endswith(".json.gz"))
        status, listing, _ = self.client.get("/api/checkpoints")
        self.assertEqual(listing["latest"]["name"], ck["name"])
        from radixnet.checkpoint import CheckpointManager

        manager = CheckpointManager(os.path.join(self.tmp.name, "ckpt"))
        self.assertEqual([r["name"] for r in manager.list()], [ck["name"]])
        restored = manager.load(ck["name"])
        self.assertEqual(restored.kind, "count")
        self.assertEqual(restored.stats()["epochs_total"], later["epochs_total"])
        status, doc, _ = self.client.post("/api/uploads/delete", {"name": "notes.txt"})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
