"""Cross-language parity of the Go port of the count / reward model (``go/``).

Skipped when no Go toolchain is available.  The Go CLI is built once into a
temporary directory; then both implementations train the same corpus with the
same settings and must produce the same structure, counts, rewards, sliding
window and RNG state, the same predictions, generated texts, scores and
conversation, and each side must load and continue the other's model file.
The punishment traversal is part of that contract: both sides must walk the
same least-punished paths at the same costs.
"""

import array
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.counter import COUNTER_LIMIT  # noqa: E402
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


def go(*args, model, expect=0, env=None):
    # --exact: atomic counting, so the numbers are reproducible and comparable with Python
    # (the default is one goroutine per text with plain, racy increments)
    cmd = [BINARY, "--json", "--exact", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    if proc.stdout.strip():
        return json.loads(proc.stdout)
    return {"error": proc.stderr.strip()} if expect else None


def py(*args, model, expect=0, env=None):
    cmd = [sys.executable, "-m", "radixnet", "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {"error": proc.stderr.strip()}


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
        # duplicates: with nothing thought twice about, both sides skip the same candidates, speak the same
        # repeats and stop at the same turn
        a = py("converse", "--turns", 40, "--explore", 0, model=self.py_model)
        b = go("converse", "--turns", 40, "--explore", 0, model=self.go_model)
        self.assertEqual(a["transcript"], b["transcript"])
        self.assertEqual([t["repeat"] for t in a["turns"]], [t["repeat"] for t in b["turns"]])
        self.assertTrue(a["repeats"])
        self.assertEqual(a["repeats"], b["repeats"])
        # and both know the same stutters, with the setting on and off
        self.assertEqual([t["stutter"] for t in a["turns"]], [t["stutter"] for t in b["turns"]])
        a = py("converse", "--turns", 40, "--allow-word-repeats", "--explore", 0, model=self.py_model)
        b = go("converse", "--turns", 40, "--allow-word-repeats", "--explore", 0, model=self.go_model)
        self.assertEqual(a["transcript"], b["transcript"])
        self.assertEqual([t["stutter"] for t in a["turns"]], [t["stutter"] for t in b["turns"]])
        self.assertEqual([t["repeat"] for t in a["turns"]], [t["repeat"] for t in b["turns"]])
        # both catch themselves in the same places, keep the same words, explore the same number of paths -
        # and teach the same nodes to hand over in future (the CLI leaves the file alone without --save, so
        # both sides start each run from the same model)
        a = py("converse", "--turns", 40, model=self.py_model)
        b = go("converse", "--turns", 40, model=self.go_model)
        self.assertEqual(a["transcript"], b["transcript"])
        self.assertEqual([t["rethink"] for t in a["turns"]], [t["rethink"] for t in b["turns"]])
        self.assertTrue([t for t in a["turns"] if t["rethink"]], a["transcript"])
        self.assertTrue(a["taught"], a["transcript"])
        self.assertEqual(a["taught"], b["taught"])

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

    def test_a_wrapped_model_file_crosses_over_unchanged(self):
        """Counters wrap at COUNTER_LIMIT; both sides must read the other's reset counts and agree."""
        py_path = os.path.join(TMP.name, "wrap_py.count.json")
        go_path = os.path.join(TMP.name, "wrap_go.count.json")
        py("--kind", "count", "--seed", 4, "train", "--data", CORPUS, "--epochs", 1, model=py_path)

        model = load_model(py_path)  # wind every counter to one event short of its reset
        graph = model.graph
        step = COUNTER_LIMIT - 1
        graph.count = [c + step for c in graph.count]
        graph.edge_count = [c + step for c in graph.edge_count]
        graph.traversals += step
        graph.total_traversals += step
        with open(py_path, "w", encoding="utf-8") as fh:
            json.dump(model.to_dict(), fh)
        shutil.copy(py_path, go_path)

        py("train", "--data", CORPUS, "--epochs", 1, model=py_path)   # the epoch takes every counter round
        go("train", "--data", CORPUS, "--epochs", 1, model=go_path)
        a, b = load_json(py_path)["graph"], load_json(go_path)["graph"]
        self.assertTrue(a["nodes"]["count_resets"], "Python recorded no reset")
        self.assertTrue(b["nodes"]["count_resets"], "Go recorded no reset")
        for block, key in (("nodes", "count"), ("nodes", "count_resets"), ("edges", "count"), ("edges", "count_resets")):
            self.assertEqual(a[block][key], b[block][key], f"{block}.{key}")
        for key in ("traversals", "traversals_resets"):
            self.assertEqual(a[key], b[key], key)
        for key in ("total_traversals", "total_traversals_resets", "window_events"):
            self.assertEqual(a["weights"][key], b["weights"][key], key)
        self.assertEqual(a["weights"]["total_traversals_resets"], 1)
        assert_close(self, a["edges"]["w"], b["edges"]["w"], 1e-12)
        # and the wrapped counts still predict the same continuations on both sides
        for prefix in ("the cat", "the dog"):
            with self.subTest(prefix=prefix):
                x = py("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=py_path)
                y = go("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=go_path)
                self.assertEqual(x["continuation"], y["continuation"])
                self.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9)

    def test_judged_paths_price_the_same_step_differently(self):
        """Both sides count correct / incorrect per path, and both let the context move the search."""
        py_path = os.path.join(TMP.name, "paths_py.count.json")
        go_path = os.path.join(TMP.name, "paths_go.count.json")
        corpus = os.path.join(TMP.name, "paths_corpus.txt")
        with open(corpus, "w", encoding="utf-8") as fh:
            fh.write("a cat sat\nthe cat sat\na cat ran\n")
        for path, run in ((py_path, py), (go_path, go)):
            run("--seed", 1, *(("--kind", "count") if run is py else ()), "train", "--data", corpus, "--epochs", 3, model=path)
            run("feedback", "--good-text", "a cat sat", "--strength", 1, model=path)
            run("feedback", "--bad-text", "the cat sat", "--strength", 1, model=path)
        a_doc, b_doc = load_json(py_path)["graph"], load_json(go_path)["graph"]
        for column in ("prev", "edge", "seen", "correct", "incorrect"):
            self.assertEqual(a_doc["paths"][column], b_doc["paths"][column], column)
        self.assertGreater(len(a_doc["paths"]["prev"]), 0)
        self.assertEqual(a_doc["weights"]["path_scale"], b_doc["weights"]["path_scale"])
        # a judged path changes what the search does, identically on both sides
        for prefix in ("a cat", "the cat", "a "):
            with self.subTest(prefix=prefix):
                a = py("predict", "--prefix", prefix, "--length", 6, "--k", 3, model=py_path)
                b = go("predict", "--prefix", prefix, "--length", 6, "--k", 3, model=go_path)
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
        a_stats, b_stats = py("info", model=py_path)["stats"], go("info", model=go_path)["stats"]
        for key in ("path_contexts", "path_judged", "path_seen", "path_correct", "path_incorrect"):
            self.assertEqual(a_stats[key], b_stats[key], key)
        self.assertGreater(a_stats["path_correct"], 0)
        self.assertGreater(a_stats["path_incorrect"], 0)

    def test_a_node_is_read_the_same_way_from_both_sides(self):
        """The node ratios: both languages share a node's traffic and its reward out the same way."""
        py_path = os.path.join(TMP.name, "ratios_py.count.json")
        go_path = os.path.join(TMP.name, "ratios_go.count.json")
        corpus = os.path.join(TMP.name, "ratios_corpus.txt")
        with open(corpus, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\na cat ran to the park\nthe cat sat on the log\n")
        for path, run in ((py_path, py), (go_path, go)):
            run("--seed", 1, *(("--kind", "count") if run is py else ()), "train", "--data", corpus, "--epochs", 2, model=path)
            run("correct", "--wrong", "the cat ran to the mat", "--right", "the cat sat on the mat", model=path)
        a = py("nodes", "--limit", 0, model=py_path)["nodes"]
        b = go("nodes", "--limit", 0, model=go_path)["nodes"]
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):  # == over the parsed rows: Go writes 0 where Python writes 0.0
            self.assertEqual(x, y, f"node row {i}")
        branch = next((row for row in a if len(row["to"]) > 1 and row["out_totals"]["incorrect"]), None)
        self.assertIsNotNone(branch, "the correction should have split a branch's verdicts")
        self.assertAlmostEqual(sum(r["seen_ratio"] for r in branch["to"]), 1.0)
        self.assertAlmostEqual(sum(abs(r["reward_ratio"]) for r in branch["to"]), 1.0)
        self.assertTrue(any(r["reward_ratio"] > 0 for r in branch["to"]))
        self.assertTrue(any(r["reward_ratio"] < 0 for r in branch["to"]))
        # and asking for that one node by its label reads the same on both sides
        one_py = py("nodes", "--node", branch["label"], model=py_path)["nodes"]
        one_go = go("nodes", "--node", branch["label"], model=go_path)["nodes"]
        self.assertEqual(one_py, one_go)
        self.assertEqual([row["node"] for row in one_py], [branch["node"]])

    def test_the_punishment_traversal_matches(self):
        """The traversal option, both sides: the same least-punished paths and the same costs."""
        py_path = os.path.join(TMP.name, "trav_py.count.json")
        go_path = os.path.join(TMP.name, "trav_go.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.go_model, go_path)
        for run, path in ((py, py_path), (go, go_path)):
            run("feedback", "--good-text", "the cat sat on the mat", "--strength", 4, model=path)
            run("feedback", "--bad-text", "the dog ate the bone", "--strength", 4, model=path)
        self.assertEqual(
            load_json(py_path)["graph"]["edges"]["reward"], load_json(go_path)["graph"]["edges"]["reward"]
        )
        for scales in (("--merit-scale", 1), ("--merit-scale", 0), ("--penalty-scale", 3)):
            for prefix in ("the cat", "the dog", "the ", ""):
                with self.subTest(prefix=prefix, scales=scales):
                    common = ("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam",
                              "--traversal", "punishment", *scales)
                    a = py(*common, model=py_path)
                    b = go(*common, model=go_path)
                    self.assertEqual((a["traversal"], b["traversal"]), ("punishment", "punishment"))
                    self.assertEqual(a["full_text"], b["full_text"])
                    self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                    self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                    self.assertEqual([t["full_text"] for t in a["bottom"]], [t["full_text"] for t in b["bottom"]])
                    assert_close(self, [t["cost"] for t in a["top"]], [t["cost"] for t in b["top"]])
        # the two traversals really do differ on a model that has been both praised and corrected
        moved = []
        for prefix in ("the cat", "the dog", "the ", "on the ", ""):
            common = ("predict", "--prefix", prefix, "--length", 12, "--k", 3, "--mode", "beam")
            plain = py(*common, model=py_path)
            other = py(*common, "--traversal", "punishment", "--merit-scale", 0, model=py_path)
            moved.append(plain["full_text"] != other["full_text"])
        self.assertTrue(any(moved), "the punishment traversal walked exactly where the rewards led")
        common = ("generate", "--mode", "beam", "--count", 4, "--max-length", 40, "--traversal", "punishment")
        a = py(*common, model=py_path)
        b = go(*common, model=go_path)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        assert_close(self, [s["cost"] for s in a["samples"]], [s["cost"] for s in b["samples"]])
        # and an unknown name is refused on both sides
        py("predict", "--prefix", "the", "--traversal", "nope", model=py_path, expect=1)
        go("predict", "--prefix", "the", "--traversal", "nope", model=go_path, expect=1)

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

    def test_corrections_move_the_same_trigram_nodes(self):
        """The tutor's correction: both sides align the sentences the same way and move the same edges."""
        pairs = [
            ("the cat sit on the mat", "the cat sits on the mat"),
            ("the dogs run in the mat", "the dogs run in the park"),
            ("a apple a day", "an apple a day"),
            ("we was happy", "we were happy"),
            ("the cat sat", "the cat sat on the mat"),
            ("the mat on sat cat", "the cat sat on the mat"),
            ("i have ate the bone", "i have eaten the bone"),
            ("the cat sat on the mat", "the cat sat on the mat"),
        ]
        for wrong, right in pairs:  # the alignment itself, without a model
            with self.subTest(wrong=wrong):
                a = py("correct", "--wrong", wrong, "--right", right, "--dry-run", model=self.py_model)
                b = go("correct", "--wrong", wrong, "--right", right, "--dry-run", model=self.go_model)
                self.assertEqual(a["changes"], b["changes"])
        py_path = os.path.join(TMP.name, "corr_py.count.json")
        go_path = os.path.join(TMP.name, "corr_go.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.go_model, go_path)
        for wrong, right in pairs:
            a = py("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, "--strength", 0.5, model=py_path)
            b = go("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, "--strength", 0.5, model=go_path)
            for key in ("edits", "penalised", "rewarded", "kept", "wrong_chars", "right_chars", "changes"):
                self.assertEqual(a[key], b[key], f"{key} for {wrong!r} -> {right!r}")
            for key in ("penalty", "reward", "loss"):
                self.assertLessEqual(abs(a[key] - b[key]), 1e-9, key)
        a_doc, b_doc = load_json(py_path)["graph"], load_json(go_path)["graph"]
        self.assertEqual(a_doc["nodes"]["labels"], b_doc["nodes"]["labels"])
        self.assertEqual(a_doc["edges"]["reward"], b_doc["edges"]["reward"])
        self.assertEqual(a_doc["edges"]["count"], b_doc["edges"]["count"])
        self.assertEqual(a_doc["weights"]["window_events"], b_doc["weights"]["window_events"])
        assert_close(self, a_doc["edges"]["w"], b_doc["edges"]["w"], 1e-12)
        # the judged paths: both sides counted the same steps in the same contexts
        a_paths, b_paths = load_json(py_path)["graph"]["paths"], load_json(go_path)["graph"]["paths"]
        for column in ("prev", "edge", "seen", "correct", "incorrect"):
            self.assertEqual(a_paths[column], b_paths[column], column)
        self.assertGreater(len(a_paths["prev"]), 0)
        a_stats = py("info", model=py_path)["stats"]
        b_stats = go("info", model=go_path)["stats"]
        for key in ("feedback_passes", "total_traversals", "window_traversals", "trained_texts",
                    "path_contexts", "path_judged", "path_seen", "path_correct", "path_incorrect"):
            self.assertEqual(a_stats[key], b_stats[key], key)
        for key in ("rewards_total", "penalties_total", "edge_reward_positive", "edge_reward_negative"):
            self.assertLessEqual(abs(a_stats[key] - b_stats[key]), 1e-9, key)

    def test_zip_corpus_streams_through_identically(self):
        """Both sides train from the same ZIP archive: Python unpacks it, Go streams it in chunks."""
        import io
        import zipfile

        with open(CORPUS, encoding="utf-8") as fh:
            lines = [l for l in fh.read().splitlines() if l.strip()]
        third = len(lines) // 3
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("part1.txt", "\n".join(lines[:third]) + "\n")
            z.writestr("more/part2.txt", "\n".join(lines[third:2 * third]) + "\n")
            z.writestr("part3.txt", "\n".join(lines[2 * third:]) + "\n")
            z.writestr("cover.png", b"\x89PNG\x00\x00")
            z.writestr("__MACOSX/._part1.txt", b"meta")
        archive = os.path.join(TMP.name, "corpus.zip")
        with open(archive, "wb") as fh:
            fh.write(buf.getvalue())
        py_path = os.path.join(TMP.name, "zip_py.count.json")
        go_path = os.path.join(TMP.name, "zip_go.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", archive, "--epochs", 2, model=py_path)
        doc = go("--seed", 1, "train", "--data", archive, "--epochs", 2, "--chunk", 7, model=go_path)
        self.assertEqual((doc["texts"], doc["chunk"], doc["records"][0]["chunks"]), (len(lines), 7, (len(lines) + 6) // 7))
        a, b = load_json(py_path)["graph"], load_json(go_path)["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])
        self.assertEqual(a["edges"]["count"], b["edges"]["count"])
        self.assertEqual(a["rng_state"], b["rng_state"])
        self.assertEqual(a["weights"]["window_events"], b["weights"]["window_events"])
        assert_close(self, a["edges"]["w"], b["edges"]["w"], 1e-12)
        self.assertEqual(load_json(py_path)["meta"]["trained_texts"], load_json(go_path)["meta"]["trained_texts"])
        # the same archive uploaded to the Python server style path and streamed by Go's server is covered below

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


class TestGoTutorParity(unittest.TestCase):
    """One fake teacher, both tutors: the same prompts, the same marks, the same model afterwards."""

    def setUp(self):
        try:  # the fake Ollama of the tutor tests plays the English teacher for both sides
            from test_tutor import start_fake
        except ImportError:
            from tests.test_tutor import start_fake
        self.fake = start_fake(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_TUTOR_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path = os.path.join(TMP.name, "tutor_py.count.json")
        self.go_path = os.path.join(TMP.name, "tutor_go.count.json")
        for path in (self.py_path, self.go_path):
            if os.path.exists(path):
                os.remove(path)
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.py_path)
        go("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.go_path)

    def calls(self):
        """(system, prompt) of every Ollama call so far, so both sides' prompts can be compared."""
        return [(body.get("system", ""), body.get("prompt", ""))
                for _method, _path, body in self.fake.requests if body]

    def test_both_tutors_ask_the_same_and_learn_the_same(self):
        options = ("tutor", "--topic", "animals", "--rounds", 2, "--exercises", 2, "--attempts", 1,
                   "--mode", "beam", "--threshold", 9.5, "--drills", 2, "--neg-epochs", 1, "--pos-epochs", 1,
                   "--strength", 1.0)
        a = py(*options, model=self.py_path, env=self.env)
        py_calls = self.calls()
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)
        go_calls = self.calls()

        # the same conversation with the teacher: same system prompts, same exercises, same sentences to mark
        self.assertEqual(len(py_calls), len(go_calls))
        for i, (first, second) in enumerate(zip(py_calls, go_calls)):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")
        self.assertGreaterEqual(len(py_calls), 6)  # exercises + marking + drills, twice

        # the same marks
        self.assertEqual(a["report"]["lessons"], b["report"]["lessons"])
        self.assertEqual(a["report"]["passed"], b["report"]["passed"])
        self.assertEqual(a["report"]["errors"], b["report"]["errors"])
        self.assertEqual(a["report"]["weakest"], b["report"]["weakest"])
        self.assertLessEqual(abs(a["report"]["mean_score"] - b["report"]["mean_score"]), 1e-9)
        self.assertEqual([l["sentence"] for l in a["lessons"]], [l["sentence"] for l in b["lessons"]])
        self.assertEqual([l["grade"]["correction"] for l in a["lessons"]], [l["grade"]["correction"] for l in b["lessons"]])
        self.assertEqual(len(a["lessons"]), 4)  # two rounds of two exercises

        # and the same model: the grades reached the graph as the same rewards and penalties
        py_doc, go_doc = load_json(self.py_path)["graph"], load_json(self.go_path)["graph"]
        self.assertEqual(py_doc["edges"]["count"], go_doc["edges"]["count"])
        assert_close(self, py_doc["edges"]["reward"], go_doc["edges"]["reward"], 1e-9)
        assert_close(self, py_doc["edges"]["w"], go_doc["edges"]["w"], 1e-9)
        py_stats, go_stats = a["stats"], b["stats"]
        for key in ("twonrl_runs", "feedback_passes", "total_traversals", "nodes", "edges"):
            self.assertEqual(py_stats[key], go_stats[key], key)
        for key in ("rewards_total", "penalties_total", "edge_reward_positive", "edge_reward_negative"):
            self.assertLessEqual(abs(py_stats[key] - go_stats[key]), 1e-9, key)

    def test_both_tutors_widen_a_mistake_the_same_way(self):
        """--blame --variants: the same question to the teacher, and the same family in the negative network."""
        py_negative = self.py_path.replace(".count.json", ".count.negative.json")
        go_negative = self.go_path.replace(".count.json", ".count.negative.json")
        for path in (py_negative, go_negative):
            if os.path.exists(path):
                os.remove(path)
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--blame", "--variants", 2, "--variant-weight", 0.5,
                   "--neg-epochs", 1, "--pos-epochs", 1)
        a = py(*options, "--negative", py_negative, model=self.py_path, env=self.env)
        py_calls = self.calls()
        self.fake.requests.clear()
        b = go(*options, "--negative", go_negative, model=self.go_path, env=self.env)
        go_calls = self.calls()

        # the same conversation, the extra "why" call included
        self.assertEqual(len(py_calls), len(go_calls))
        for i, (first, second) in enumerate(zip(py_calls, go_calls)):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")
        self.assertEqual(sum(1 for system, _ in py_calls if "explaining a beginner's mistake" in system), 1)

        # the same explanation and the same family on every lesson
        self.assertEqual([l["why"] for l in a["lessons"]], [l["why"] for l in b["lessons"]])
        self.assertEqual([l["variants"] for l in a["lessons"]], [l["variants"] for l in b["lessons"]])
        self.assertTrue(all(len(l["variants"]) == 2 for l in a["lessons"]))
        rounds = [r for r in a["records"] if r["kind"] == "round"]
        self.assertEqual(rounds[0]["similar"], 2 * rounds[0]["lessons"])

        # and the same negative network afterwards
        p, g = load_json(py_negative)["graph"], load_json(go_negative)["graph"]
        self.assertEqual(p["nodes"]["labels"], g["nodes"]["labels"])
        self.assertEqual((p["edges"]["src"], p["edges"]["dst"]), (g["edges"]["src"], g["edges"]["dst"]))
        assert_close(self, p["edges"]["blame"], g["edges"]["blame"], 1e-12)
        self.assertEqual(p["edges"]["reasons"], g["edges"]["reasons"])
        for model, negative in ((self.py_path, py_negative), (self.go_path, go_negative)):
            runner = py if model == self.py_path else go
            verdict = runner("negative", "why", "--text", "the dogs sits on the mat 0", "--negative", negative,
                             model=model)["verdicts"][0]
            self.assertNotEqual(verdict["verdict"], "pass")  # a sentence neither network ever wrote

    def test_both_tutors_plan_the_same_next_lessons(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--dry-run", "--plan", 2)
        a = py(*options, model=self.py_path, env=self.env)
        py_calls = self.calls()
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)
        go_calls = self.calls()

        # the report card is handed back to the teacher with the same words on both sides
        self.assertEqual(len(py_calls), len(go_calls))
        for i, (first, second) in enumerate(zip(py_calls, go_calls)):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")
        plan_prompts = [prompt for system, prompt in py_calls if "planning the next lessons" in system]
        self.assertEqual(len(plan_prompts), 1)
        self.assertIn("Lessons marked: 2", plan_prompts[0])

        # and the same syllabus comes out, down to the brief the next batch would be taught to
        self.assertEqual(a["plan"]["lessons"], b["plan"]["lessons"])
        self.assertEqual(a["plan"]["targets"], b["plan"]["targets"])
        self.assertEqual(a["plan"]["weak"], b["plan"]["weak"])
        self.assertEqual((a["plan"]["level"], a["plan"]["source"]), (b["plan"]["level"], b["plan"]["source"]))
        self.assertEqual(a["plan"]["summary"], b["plan"]["summary"])
        self.assertEqual(a["plan"]["prompt"], b["plan"]["prompt"])
        self.assertEqual(a["plan"]["upgrade"], b["plan"]["upgrade"])
        self.assertEqual(a["plan"]["upgrade"]["step"], "hold")  # nothing passed at 9.5
        self.assertEqual(a["plan"]["lessons"][0]["targets"], "agreement")
        self.assertEqual([r["kind"] for r in a["records"]], [r["kind"] for r in b["records"]])

    def test_both_tutors_teach_to_the_same_brief(self):
        brief = "Drill plural nouns first. Stay at beginner. Keep the sentences about animals."
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam", "--dry-run",
                   "--brief", brief)
        py(*options, model=self.py_path, env=self.env)
        py_calls = self.calls()
        self.fake.requests.clear()
        go(*options, model=self.go_path, env=self.env)
        self.assertEqual(len(py_calls), len(self.calls()))
        for i, (first, second) in enumerate(zip(py_calls, self.calls())):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")
        written = [prompt for system, prompt in py_calls if "writing exercises" in system]
        self.assertEqual(len(written), 1)
        self.assertIn(f"The plan for this batch of lessons: {brief}", written[0])

    def test_both_tutors_auto_run_the_same_way(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--dry-run", "--batches", 2)
        a = py(*options, model=self.py_path, env=self.env)
        py_calls = self.calls()
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)

        # the same conversation: exercises, marking, the plan between the batches, then the second batch's exercises
        self.assertEqual(len(py_calls), len(self.calls()))
        for i, (first, second) in enumerate(zip(py_calls, self.calls())):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")
        self.assertEqual([r["kind"] for r in a["records"]], [r["kind"] for r in b["records"]])
        self.assertEqual([r["kind"] for r in a["records"]], ["round", "report", "plan", "batch", "round", "report"])
        started_py = [r for r in a["records"] if r["kind"] == "batch"]
        started_go = [r for r in b["records"] if r["kind"] == "batch"]
        self.assertEqual(started_py, started_go)  # the same brief, step, level, openings, pass mark and drills
        self.assertEqual(started_py[0]["batch"], 2)
        written = [prompt for system, prompt in py_calls if "writing exercises" in system]
        self.assertEqual(len(written), 2)
        self.assertIn(f"The plan for this batch of lessons: {started_py[0]['brief']}", written[1])

    def test_dry_runs_agree_and_change_nothing(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam", "--dry-run")
        before = load_json(self.py_path)["graph"]["edges"]["reward"]
        a = py(*options, model=self.py_path, env=self.env)
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)
        self.assertEqual([l["sentence"] for l in a["lessons"]], [l["sentence"] for l in b["lessons"]])
        self.assertEqual(a["report"]["passed"], b["report"]["passed"])
        self.assertIsNone(a["saved"])
        self.assertEqual(b["saved"], "")
        self.assertEqual(load_json(self.py_path)["graph"]["edges"]["reward"], before)
        self.assertEqual(load_json(self.go_path)["graph"]["edges"]["reward"], before)


class TestGoNegativeParity(unittest.TestCase):
    """The negative network in both languages: the same blame, the same corrections, the same verdicts, and
    each side reading the other's model file."""

    def setUp(self):
        self.py_model = os.path.join(TMP.name, f"neg_py_{self.id().rsplit('.', 1)[-1]}.count.json")
        self.go_model = os.path.join(TMP.name, f"neg_go_{self.id().rsplit('.', 1)[-1]}.count.json")
        self.py_negative = self.py_model.replace(".count.json", ".count.negative.json")
        self.go_negative = self.go_model.replace(".count.json", ".count.negative.json")
        for path in (self.py_negative, self.go_negative):
            if os.path.exists(path):
                os.remove(path)

    def teach(self, model):
        """The same three lessons on both sides: two whole failures and one correction."""
        lessons = [
            ("negative", "blame", "--text", "the the the the cat", "--reason", "repetition",
             "--severity", 1.5, "--source", "review", "--note", "it repeats the same word"),
            ("negative", "blame", "--text", "the cat cat cat sat", "--reason", "gibberish", "--source", "review"),
            ("negative", "clear", "--text", "the cat sat on the mat"),
        ]
        runner = py if model == self.py_model else go
        for args in lessons:
            runner(*args, model=model)

    def test_the_same_blame_and_the_same_verdicts(self):
        self.teach(self.py_model)
        self.teach(self.go_model)
        py_doc, go_doc = load_json(self.py_negative), load_json(self.go_negative)
        self.assertEqual(py_doc["format"], "radixnet-negative")
        self.assertEqual(go_doc["format"], "radixnet-negative")
        self.assertEqual(py_doc["kind"], go_doc["kind"])
        p, g = py_doc["graph"], go_doc["graph"]
        self.assertEqual(p["nodes"]["labels"], g["nodes"]["labels"])
        self.assertEqual((p["edges"]["src"], p["edges"]["dst"]), (g["edges"]["src"], g["edges"]["dst"]))
        assert_close(self, p["edges"]["blame"], g["edges"]["blame"], 1e-12)
        assert_close(self, p["edges"]["clear"], g["edges"]["clear"], 1e-12)
        self.assertEqual(p["edges"]["fails"], g["edges"]["fails"])
        self.assertEqual(p["edges"]["reasons"], g["edges"]["reasons"])
        assert_close(self, p["edges"]["w"], g["edges"]["w"], 1e-12)
        pw, gw = p["weights"], g["weights"]
        for key in ("function", "kind", "share_scale", "blame_scale", "clear_scale", "smoothing",
                    "total_blame", "total_fails", "total_clear"):
            self.assertEqual(pw[key], gw[key], key)
        self.assertEqual(pw["reasons"], gw["reasons"])
        self.assertEqual(py_doc["filter"], go_doc["filter"])
        self.assertEqual([e["reason"] for e in py_doc["log"]], [e["reason"] for e in go_doc["log"]])
        self.assertEqual([e["note"] for e in py_doc["log"]], [e["note"] for e in go_doc["log"]])
        # the same judgements, sentence for sentence
        for text in ("the the the the cat", "the cat cat cat sat", "the cat sat on the mat", "a wholly unseen line"):
            a = py("negative", "why", "--text", text, model=self.py_model)["verdicts"][0]
            b = go("negative", "why", "--text", text, model=self.go_model)["verdicts"][0]
            for key in ("verdict", "why", "transitions", "blamed", "known"):
                self.assertEqual(a[key], b[key], f"{text}: {key}")
            for key in ("risk", "peak", "coverage", "blame"):
                self.assertAlmostEqual(a[key], b[key], places=9, msg=f"{text}: {key}")
            self.assertEqual([r["reason"] for r in a["reasons"]], [r["reason"] for r in b["reasons"]], text)
            self.assertEqual([s["fragment"] for s in a["spans"]], [s["fragment"] for s in b["spans"]], text)

    def test_a_correction_blames_the_same_characters(self):
        py("negative", "blame", "--text", "the cat sat here", "--reason", "other", model=self.py_model)
        go("negative", "blame", "--text", "the cat sat here", "--reason", "other", model=self.go_model)
        py("--kind", "count", "train", "--data", CORPUS, "--epochs", 1, model=self.py_model)
        go("train", "--data", CORPUS, "--epochs", 1, model=self.go_model)
        for model in (self.py_model, self.go_model):
            runner = py if model == self.py_model else go
            runner("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sits on the mat",
                   "--blame", "--reason", "agreement", "--note", "the verb must agree", model=model)
        a = py("negative", "why", "--text", "the cat sit on the mat", model=self.py_model)["verdicts"][0]
        b = go("negative", "why", "--text", "the cat sit on the mat", model=self.go_model)["verdicts"][0]
        self.assertEqual([s["fragment"] for s in a["spans"]], [s["fragment"] for s in b["spans"]])
        self.assertEqual([s["reason"] for s in a["spans"]], [s["reason"] for s in b["spans"]])
        self.assertAlmostEqual(a["risk"], b["risk"], places=9)
        self.assertEqual(a["why"], b["why"])

    def test_each_side_reads_the_other_s_negative_model(self):
        self.teach(self.py_model)
        self.teach(self.go_model)
        text = "the the the the cat"
        # Go judges the model Python taught, and Python the one Go taught
        crossed_go = go("negative", "why", "--text", text, "--negative", self.py_negative, model=self.go_model)["verdicts"][0]
        crossed_py = py("negative", "why", "--text", text, "--negative", self.go_negative, model=self.py_model)["verdicts"][0]
        own = py("negative", "why", "--text", text, model=self.py_model)["verdicts"][0]
        for other in (crossed_go, crossed_py):
            self.assertEqual(other["verdict"], own["verdict"])
            self.assertAlmostEqual(other["risk"], own["risk"], places=9)
            self.assertEqual(other["why"], own["why"])

    def test_the_filter_agrees_on_what_to_veto(self):
        self.teach(self.py_model)
        self.teach(self.go_model)
        py("--kind", "count", "train", "--data", CORPUS, "--epochs", 2, model=self.py_model)
        go("train", "--data", CORPUS, "--epochs", 2, model=self.go_model)
        args = ("negative", "filter", "--text", "the the the the cat", "--text", "a rainy day in autumn")
        a = py(*args, model=self.py_model)
        b = go(*args, model=self.go_model)
        self.assertEqual([v["decision"] for v in a["verdicts"]], [v["decision"] for v in b["verdicts"]])
        self.assertEqual([v["rule"] for v in a["verdicts"]], [v["rule"] for v in b["verdicts"]])
        self.assertEqual(a["kept"], b["kept"])
        for x, y in zip(a["verdicts"], b["verdicts"]):
            self.assertAlmostEqual(x["ratio"], y["ratio"], places=9)
            self.assertAlmostEqual(x["peak"], y["peak"], places=9)
            self.assertEqual(x["why"], y["why"])
        # the peak rule fires on the same candidate in both
        peak = ("negative", "filter", "--text", "the the the the cat", "--no-ratio", "--threshold", 1e9, "--peak", 1)
        self.assertEqual(py(*peak, model=self.py_model)["verdicts"][0]["rule"],
                         go(*peak, model=self.go_model)["verdicts"][0]["rule"])


    def test_the_guard_vetoes_the_same_replies_on_both_sides(self):
        """The pair on the ordinary output paths: both CLIs filter what they write, and agree on what to drop."""
        self.teach(self.py_model)
        self.teach(self.go_model)
        py("--kind", "count", "train", "--data", CORPUS, "--epochs", 2, model=self.py_model)
        go("train", "--data", CORPUS, "--epochs", 2, model=self.go_model)
        a = py("generate", "--count", 3, "--mode", "beam", "--max-length", 40, model=self.py_model)
        b = go("generate", "-count", 3, "-mode", "beam", "-max-length", 40, model=self.go_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        self.assertEqual(a["guard"]["asked"], b["guard"]["asked"])
        self.assertEqual([v["text"] for v in a["guard"]["verdicts"]], [v["text"] for v in b["guard"]["verdicts"]])
        self.assertEqual([v["decision"] for v in a["guard"]["verdicts"]], [v["decision"] for v in b["guard"]["verdicts"]])
        self.assertEqual([v["why"] for v in a["guard"]["rejected"]], [v["why"] for v in b["guard"]["rejected"]])
        # and both hand out what the positive model wrote when asked not to filter
        a = py("generate", "--count", 3, "--mode", "beam", "--max-length", 40, "--no-guard", model=self.py_model)
        b = go("generate", "-count", 3, "-mode", "beam", "-max-length", 40, "-no-guard", model=self.go_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        self.assertIsNone(a["guard"])
        self.assertIsNone(b["guard"])

    def test_the_guard_ranks_the_same_continuations_and_replies(self):
        self.teach(self.py_model)
        self.teach(self.go_model)
        py("--kind", "count", "train", "--data", CORPUS, "--epochs", 2, model=self.py_model)
        go("train", "--data", CORPUS, "--epochs", 2, model=self.go_model)
        a = py("predict", "--prefix", "the ", "--mode", "beam", "--k", 4, model=self.py_model)
        b = go("predict", "-prefix", "the ", "-mode", "beam", "-k", 4, model=self.go_model)
        self.assertEqual(a["continuation"], b["continuation"])
        # the two CLIs name a continuation differently (text / continuation); full_text is the same in both
        self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
        self.assertEqual(a["guard"]["kept"], b["guard"]["kept"])
        a = py("converse", "--turns", 3, "--opening", "the the the the cat", model=self.py_model)
        b = go("converse", "-turns", 3, "-opening", "the the the the cat", model=self.go_model)
        self.assertEqual([t["text"] for t in a["turns"]], [t["text"] for t in b["turns"]])
        self.assertEqual([t["vetoed"] for t in a["turns"]], [t["vetoed"] for t in b["turns"]])
        self.assertEqual(a["guard"]["vetoed"], b["guard"]["vetoed"])


class TestGoServerNegative(unittest.TestCase):
    """The Go server's negative endpoints answer the JSON the frontend's Negative tab expects."""

    @classmethod
    def setUpClass(cls):
        import socket
        import time
        from tests.test_api import Client

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-go-negative-")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.model = os.path.join(cls.tmp.name, "model.count.json")
        cls.proc = subprocess.Popen(
            [BINARY, "--model", cls.model, "--exact", "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=GO_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        cls.client = Client(f"http://127.0.0.1:{port}")
        for _ in range(200):
            try:
                if cls.client.get("/api/health")[0] == 200:
                    break
            except Exception:  # noqa: BLE001 - the server is still starting
                pass
            time.sleep(0.05)
        else:
            cls.tearDownClass()
            raise AssertionError("the Go server did not start")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "proc", None) is not None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        cls.tmp.cleanup()

    def test_the_negative_endpoints(self):
        status, data, _ = self.client.post("/api/negative/blame", {"texts": ["the the the the cat"],
                                                                   "reason": "repetition", "note": "it repeats"})
        self.assertEqual(status, 200, data)
        self.assertEqual([r["reason"] for r in data["reasons"]], ["repetition"])
        status, data, _ = self.client.get("/api/negative")
        self.assertEqual(status, 200, data)
        self.assertEqual(set(data), {"path", "active", "stats", "reasons", "journal", "weights", "settings"})
        self.assertEqual(data["stats"]["kind"], "negative")
        self.assertEqual(data["journal"][0]["note"], "it repeats")
        status, data, _ = self.client.post("/api/negative/judge", {"text": "the the the the cat"})
        self.assertEqual(status, 200, data)
        verdict = data["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertLessEqual({"risk", "peak", "coverage", "blame", "reasons", "spans", "why"}, set(verdict))
        status, data, _ = self.client.post("/api/negative/filter", {"texts": ["the the the the cat", "an unseen line"]})
        self.assertEqual(status, 200, data)
        self.assertEqual([v["decision"] for v in data["verdicts"]], ["reject", "pass"])
        self.assertIsNone(data["verdicts"][1]["rule"])
        self.assertEqual(data["kept"], ["an unseen line"])
        status, data, _ = self.client.post("/api/negative/settings", {"threshold": 2.0})
        self.assertEqual(data["settings"]["threshold"], 2.0)
        status, data, _ = self.client.post("/api/negative/save")
        self.assertEqual(status, 200, data)
        self.assertTrue(os.path.isfile(data["path"]))
        from radixnet.model import load_model

        self.assertEqual(load_model(data["path"]).kind, "negative")


class TestGoCriticParity(unittest.TestCase):
    """One fake reviewer, both loops: the same prompts, the same marks, the same blame."""

    def setUp(self):
        try:  # the fake Ollama of the ollama tests plays the reviewer for both sides
            from test_ollama import start_fake
        except ImportError:
            from tests.test_ollama import start_fake
        self.fake = start_fake(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path = os.path.join(TMP.name, "critic_py.count.json")
        self.go_path = os.path.join(TMP.name, "critic_go.count.json")
        for path in (self.py_path, self.go_path):
            for name in (path, path.replace(".count.json", ".count.negative.json")):
                if os.path.exists(name):
                    os.remove(name)
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.py_path)
        go("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.go_path)

    def prompts(self):
        """The prompt of every call so far, so both sides' conversations can be compared."""
        return [body.get("prompt", "") for _method, _path, body in self.fake.requests if body]

    def test_both_loops_ask_the_same_and_blame_the_same(self):
        options = ("negative", "auto", "--rounds", 2, "--count", 4, "--max-length", 40, "--threshold", 6)
        a = py(*options, model=self.py_path, env=self.env)
        py_prompts = self.prompts()
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)
        go_prompts = self.prompts()

        # the same texts written, so the same conversation with the reviewer
        self.assertEqual(len(py_prompts), len(go_prompts))
        for i, (first, second) in enumerate(zip(py_prompts, go_prompts)):
            self.assertEqual(first, second, f"call {i} differs between the two loops")
        self.assertGreaterEqual(len(py_prompts), 2)

        # the same rounds, marked the same and blamed the same
        py_rounds = [r for r in a["records"] if r["kind"] == "round"]
        go_rounds = [r for r in b["records"] if r["kind"] == "round"]
        self.assertEqual(len(py_rounds), 2)
        self.assertEqual(len(py_rounds), len(go_rounds))
        for first, second in zip(py_rounds, go_rounds):
            for key in ("round", "texts", "passed", "failed", "blamed", "cleared", "edges", "reasons"):
                self.assertEqual(first[key], second[key], f"round {first['round']}: {key}")
            self.assertAlmostEqual(first["mean_rating"], second["mean_rating"], places=9)
            self.assertAlmostEqual(first["severity_mean"], second["severity_mean"], places=9)

        # the same report card
        for key in ("rounds", "reviewed", "blamed", "cleared", "edges", "reasons"):
            self.assertEqual(a["report"][key], b["report"][key], key)
        self.assertAlmostEqual(a["report"]["mean_rating"], b["report"]["mean_rating"], places=9)

        # and the same negative network on disk
        py_doc = load_json(self.py_path.replace(".count.json", ".count.negative.json"))
        go_doc = load_json(self.go_path.replace(".count.json", ".count.negative.json"))
        self.assertEqual(py_doc["format"], go_doc["format"])
        self.assertEqual(py_doc["graph"]["nodes"]["labels"], go_doc["graph"]["nodes"]["labels"])
        self.assertEqual(py_doc["graph"]["edges"]["blame"], go_doc["graph"]["edges"]["blame"])
        self.assertEqual(py_doc["graph"]["edges"]["fails"], go_doc["graph"]["edges"]["fails"])
        self.assertEqual(sorted(r["reason"] for r in py_doc["log"]),
                         sorted(r["reason"] for r in go_doc["log"]))
        self.assertEqual({r["source"] for r in py_doc["log"]}, {"critic"})
        self.assertEqual({r["source"] for r in go_doc["log"]}, {"critic"})

    def test_the_positive_model_comes_out_untouched_on_both_sides(self):
        before = (load_json(self.py_path), load_json(self.go_path))
        options = ("negative", "auto", "--rounds", 1, "--count", 3, "--max-length", 40)
        py(*options, model=self.py_path, env=self.env)
        self.fake.requests.clear()
        go(*options, model=self.go_path, env=self.env)
        self.assertEqual(before[0], load_json(self.py_path))
        self.assertEqual(before[1], load_json(self.go_path))

    def test_both_reviews_mark_the_same_given_texts(self):
        texts = ["the cat sat on the mat", "xxxx xxxx xxxx"]
        options = ("ollama", "review", "--threshold", 6, *sum((("--text", t) for t in texts), ()))
        a = py(*options, model=self.py_path, env=self.env)
        self.fake.requests.clear()
        b = go(*options, model=self.go_path, env=self.env)
        self.assertEqual([r["verdict"] for r in a["reviews"]], [r["verdict"] for r in b["reviews"]])
        self.assertEqual([r["rating"] for r in a["reviews"]], [r["rating"] for r in b["reviews"]])
        self.assertEqual(a["good"], b["good"])
        self.assertEqual(a["bad"], b["bad"])


class TestGoToolsParity(unittest.TestCase):
    """The tools both sides offer, and the text format the network calls them in."""

    def setUp(self):
        self.py_model = os.path.join(TMP.name, "tools_py.count.json")
        self.go_model = os.path.join(TMP.name, "tools_go.count.json")

    def test_the_same_tools_with_the_same_schemas(self):
        a = py("tools", "list", "--offline", model=self.py_model)
        b = go("tools", "list", "--offline", model=self.go_model)
        self.assertEqual(a["tools"], b["tools"])
        a = py("tools", "list", model=self.py_model)
        b = go("tools", "list", model=self.go_model)
        self.assertEqual([t["name"] for t in a["tools"]], [t["name"] for t in b["tools"]])
        self.assertEqual([t["signature"] for t in a["tools"]], [t["signature"] for t in b["tools"]])
        self.assertEqual(a["tools"], b["tools"])  # descriptions, parameters, defaults and all
        a = py("tools", "describe", "--tool", "web_fetch", model=self.py_model)
        b = go("tools", "describe", "--tool", "web_fetch", model=self.go_model)
        self.assertEqual(a["schema"], b["schema"])  # what the LLM is handed for tool calling

    def test_the_calculator_answers_the_same(self):
        for expression in ("2 * (3 + 4)", "sqrt(841)", "7 / 2", "7 // 2", "-7 % 3", "round(2.5)",
                           "round(3.14159, 2)", "min(3, 1, 2)", "sum([1, 2, 3])", "2 ^ 8", "1 < 2",
                           "abs(-3)", "int(3.9)", "float(3)", "log(100, 10)", "hypot(3, 4)", "0.1 + 0.2"):
            a = py("tools", "call", "--tool", "calculator", "--arg", f"expression={expression}", model=self.py_model)
            b = go("tools", "call", "--tool", "calculator", "--arg", f"expression={expression}", model=self.go_model)
            self.assertEqual((a["ok"], a["output"]), (b["ok"], b["output"]), expression)
        # both refuse the same expressions, and say so on stderr with exit 1
        for bad in ("10 / 0", "open('x')", "nosuchname"):
            args = ("tools", "call", "--tool", "calculator", "--arg", f"expression={bad}")
            self.assertIn("calculator", py(*args, model=self.py_model, expect=1)["error"], bad)
            self.assertIn("calculator", go(*args, model=self.go_model, expect=1)["error"], bad)

    def test_a_call_is_written_and_read_the_same(self):
        """The text format is the contract: a call one side writes is the call the other reads."""
        written = 'calculator {"expression": "6*7"}'
        a = py("tools", "call", "--call", written, model=self.py_model)
        b = go("tools", "call", "--call", written, model=self.go_model)
        self.assertEqual((a["tool"], a["arguments"], a["output"]), (b["tool"], b["arguments"], b["output"]))
        # the lenient reader: a bare value, key=value pairs and a truncated call all arrive the same
        for raw in ("calculator 5*5", "calculator expression=2+2", 'calculator {"expression": "9"'):
            a = py("tools", "call", "--call", raw, model=self.py_model)
            b = go("tools", "call", "--call", raw, model=self.go_model)
            self.assertEqual((a["arguments"], a["output"]), (b["arguments"], b["output"]), raw)


class TestGoCodeGenParity(unittest.TestCase):
    """One fake teacher, both code-generation trainers: the same prompts, the same attempts, the same solutions."""

    def setUp(self):
        try:  # the fake Ollama of the codegen tests plays teacher and judge for both sides
            from test_codegen import start_fake
        except ImportError:
            from tests.test_codegen import start_fake
        self.fake = start_fake(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_CODEGEN_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path = os.path.join(TMP.name, "codegen_py.count.json")
        self.go_path = os.path.join(TMP.name, "codegen_go.count.json")
        for path in (self.py_path, self.go_path):
            for name in (path, path.replace(".count.json", ".count.negative.json")):
                if os.path.exists(name):
                    os.remove(name)
        self.problems = os.path.join(TMP.name, "codegen_problems.txt")
        with open(self.problems, "w", encoding="utf-8") as fh:
            fh.write("Print the word hello.\n")
        # the teacher's first program does not run; its fix does
        self.script = ["```python\nprint(hello)\n```", '```python\nprint("hello")\n```']

    def prompts(self):
        """Every prompt so far, with the two things in them that cannot match masked out: the sandbox's own
        scratch directory (a fresh one per run, named in the tracebacks) and how long a program took."""
        out = []
        for body in self.fake.requests:
            if not body:
                continue
            prompt = re.sub(r"radixnet-sandbox-\w+", "radixnet-sandbox-X", body.get("prompt", ""))
            out.append(re.sub(r"exit code (-?\d+), [\d.]+s", r"exit code \1, Ts", prompt))
        return out

    def test_both_trainers_ask_the_same_and_solve_the_same(self):
        self.fake.solutions = list(self.script)
        options = ("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-attempts", 2,
                   "--sandbox-timeout", 20, "--blame")
        a = py(*options, model=self.py_path, env=self.env)
        py_prompts = self.prompts()
        self.fake.requests.clear()
        self.fake.solutions = list(self.script)
        b = go(*options, model=self.go_path, env=self.env)
        go_prompts = self.prompts()

        # the same conversation with the teacher and the judge, prompt for prompt
        self.assertEqual(len(py_prompts), len(go_prompts))
        for i, (first, second) in enumerate(zip(py_prompts, go_prompts)):
            self.assertEqual(first, second, f"call {i} differs between the two trainers")
        self.assertGreaterEqual(len(py_prompts), 3)  # write, judge the fix, judge again

        # the same solution, found the same way
        self.assertEqual(a["solutions"], b["solutions"])
        self.assertEqual(a["solutions"], {"p1": 'Print the word hello.\nprint("hello")\n'})
        self.assertEqual((a["solved"], b["solved"]), (1, 1))

        # and the rejected program blamed the same reason on both sides
        py_negative = load_json(self.py_path.replace(".count.json", ".count.negative.json"))
        go_negative = load_json(self.go_path.replace(".count.json", ".count.negative.json"))
        self.assertEqual([r["reason"] for r in a["negative"]["reasons"]],
                         [r["reason"] for r in b["negative"]["reasons"]])
        self.assertEqual(py_negative["graph"]["weights"]["reasons"], go_negative["graph"]["weights"]["reasons"])
        self.assertEqual(py_negative["graph"]["edges"]["reasons"], go_negative["graph"]["edges"]["reasons"])

    def test_both_sandboxes_report_the_same_failure(self):
        """The sandbox is the same Python bootstrap on both sides, so a crash reads the same."""
        self.fake.solutions = ["```python\nraise ValueError('boom')\n```"]
        options = ("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-attempts", 1,
                   "--sandbox-timeout", 20)
        a = py(*options, model=self.py_path, env=self.env)
        self.fake.solutions = ["```python\nraise ValueError('boom')\n```"]
        b = go(*options, model=self.go_path, env=self.env)
        self.assertEqual((a["solved"], b["solved"]), (0, 0))
        py_record = [r for r in a["records"] if r["kind"] == "problem"][0]
        go_record = [r for r in b["records"] if r["kind"] == "problem"][0]
        self.assertEqual(py_record["issues"], go_record["issues"])
        self.assertEqual(py_record["correct"], go_record["correct"])


class TestGoChatParity(unittest.TestCase):
    """One fake partner, both chat loops: the same conversation, the same marks, the same learning."""

    def setUp(self):
        try:  # the fake partner / judge of the chat tests answers both sides
            from test_chat import start_partner
        except ImportError:
            from tests.test_chat import start_partner
        self.fake = start_partner(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path = os.path.join(TMP.name, "chat_py.count.json")
        self.go_path = os.path.join(TMP.name, "chat_go.count.json")
        for path in (self.py_path, self.go_path):
            for name in (path, path.replace(".count.json", ".count.negative.json")):
                if os.path.exists(name):
                    os.remove(name)
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.py_path)
        go("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=self.go_path)

    def prompts(self):
        """Every prompt the partner and the judge were given, so both conversations can be compared."""
        return [body.get("prompt", "") for body in self.fake.requests if body]

    def restart(self):
        """A fresh fake for the second side: the same lines, in the same order, from the start."""
        self.fake.requests.clear()
        self.fake.lines = 0
        self.fake.marked = 0

    def test_both_loops_converse_the_same_and_learn_the_same(self):
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 2,
                   "--turns", 2, "--topic", "animals", "--neg-epochs", 1, "--pos-epochs", 1)
        a = py(*options, model=self.py_path, env=self.env)
        py_prompts = self.prompts()
        self.restart()
        b = go(*options, model=self.go_path, env=self.env)
        go_prompts = self.prompts()

        # the same conversation with the partner and the judge, prompt for prompt
        self.assertEqual(len(py_prompts), len(go_prompts))
        for i, (first, second) in enumerate(zip(py_prompts, go_prompts)):
            self.assertEqual(first, second, f"call {i} differs between the two loops")
        self.assertGreaterEqual(len(py_prompts), 4)  # open, carry on, open again, and the two judgements

        # the same conversations, marked the same and learned from the same
        py_held = [r for r in a["records"] if r["kind"] == "conversation"]
        go_held = [r for r in b["records"] if r["kind"] == "conversation"]
        self.assertEqual(len(py_held), 2)
        self.assertEqual(len(py_held), len(go_held))
        for first, second in zip(py_held, go_held):
            for key in ("conversation", "transcript", "exchanges", "passed", "failed", "stalled", "vetoed",
                        "blamed", "cleared", "edges", "reasons", "action", "bad", "good",
                        "overall_rating", "overall_critique"):
                self.assertEqual(first[key], second[key], f"conversation {first['conversation']}: {key}")
            self.assertEqual([(r["index"], r["said"], r["text"], r["rating"], r["verdict"]) for r in first["reviews"]],
                             [(r["index"], r["said"], r["text"], r["rating"], r["verdict"]) for r in second["reviews"]])
            self.assertAlmostEqual(first["mean_rating"], second["mean_rating"], places=9)

        # the same report card, and the same speakers in it
        for key in ("conversations", "exchanges", "passed", "failed", "vetoed", "stalled", "blamed",
                    "cleared", "edges", "reasons", "overall_rating"):
            self.assertEqual(a["report"][key], b["report"][key], key)
        self.assertAlmostEqual(a["report"]["mean_rating"], b["report"]["mean_rating"], places=9)
        self.assertEqual(a["speakers"], b["speakers"])
        self.assertEqual(a["speakers"], ["Partner", "Model"])

        # and the same two networks on disk: 2NRL trained the positive one, the marks taught the negative one
        py_model, go_model = load_json(self.py_path), load_json(self.go_path)
        self.assertEqual(py_model["graph"]["nodes"]["labels"], go_model["graph"]["nodes"]["labels"])
        self.assertEqual(py_model["graph"]["edges"]["count"], go_model["graph"]["edges"]["count"])
        assert_close(self, py_model["graph"]["edges"]["reward"], go_model["graph"]["edges"]["reward"], 1e-9)
        py_neg = load_json(self.py_path.replace(".count.json", ".count.negative.json"))
        go_neg = load_json(self.go_path.replace(".count.json", ".count.negative.json"))
        self.assertEqual(py_neg["graph"]["nodes"]["labels"], go_neg["graph"]["nodes"]["labels"])
        self.assertEqual(py_neg["graph"]["edges"]["blame"], go_neg["graph"]["edges"]["blame"])
        self.assertEqual({r["source"] for r in py_neg["log"]}, {"chat"})
        self.assertEqual({r["source"] for r in go_neg["log"]}, {"chat"})

    def test_a_given_opening_is_spoken_as_it_is_on_both_sides(self):
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 1,
                   "--turns", 1, "--opening", "tell me about the cat", "--no-learn", "--no-blame")
        a = py(*options, model=self.py_path, env=self.env)
        self.restart()
        b = go(*options, model=self.go_path, env=self.env)
        py_held = next(r for r in a["records"] if r["kind"] == "conversation")
        go_held = next(r for r in b["records"] if r["kind"] == "conversation")
        self.assertEqual(py_held["transcript"][0], {"speaker": "Partner", "text": "tell me about the cat"})
        self.assertEqual(py_held["transcript"], go_held["transcript"])
        self.assertIsNone(a["saved"])  # nothing was learned, so neither model was written back
        self.assertIsNone(b["saved"])

    def test_neither_side_learns_with_no_learn(self):
        before = (load_json(self.py_path), load_json(self.go_path))
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 1,
                   "--turns", 1, "--no-learn", "--no-blame")
        py(*options, model=self.py_path, env=self.env)
        self.restart()
        go(*options, model=self.go_path, env=self.env)
        self.assertEqual(before[0], load_json(self.py_path))
        self.assertEqual(before[1], load_json(self.go_path))


class TestGoMediaParity(unittest.TestCase):
    """The media text formats: both sides must encode the same recording and the same picture identically.

    This matters more than it looks.  The utterance token is a digest of the
    waveform text and the token is *in the text the model trains on*, so if the
    two sides encoded a recording differently the same file would teach two
    different things and the "one model, two languages" guarantee would quietly
    stop holding.
    """

    def setUp(self):
        self.wav = os.path.join(TMP.name, "probe.wav")
        self.png = os.path.join(TMP.name, "probe.png")
        self.model = os.path.join(TMP.name, "media.count.json")
        for path in (self.model, self.model.replace(".count.json", ".count.negative.json")):
            if os.path.exists(path):
                os.remove(path)
        from radixnet import speech
        rate, seconds = 8000, 0.05
        samples = array.array("f", [0.6 * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(rate * seconds))])
        with open(self.wav, "wb") as fh:
            fh.write(speech.wav_bytes(samples, rate))
        with open(self.png, "wb") as fh:
            fh.write(_gradient_png(64, 64))

    def test_both_sides_encode_the_same_utterance_identically(self):
        options = ("speech", "teach", self.wav, "--text", "the cat sat on the mat", "--rate", 8000)
        a = py(*options, model=self.model)
        b = go(*options, model=self.model)
        self.assertEqual(a["token"], b["token"], "the utterance token must match: it is in the trained text")
        self.assertEqual(a["texts"], b["texts"])
        self.assertEqual(a["audio"]["codec"], b["audio"]["codec"])
        self.assertEqual(a["audio"]["samples"], b["audio"]["samples"])
        self.assertEqual(a["audio"]["bytes"], b["audio"]["bytes"])

    def test_each_side_decodes_the_other_s_waveform(self):
        text = py("speech", "teach", self.wav, "--rate", 8000, model=self.model)["texts"][0]
        out_py = os.path.join(TMP.name, "back_py.wav")
        out_go = os.path.join(TMP.name, "back_go.wav")
        a = py("speech", "decode", "--text", text, "--out", out_py, model=self.model)
        b = go("speech", "decode", "--text", text, "--out", out_go, model=self.model)
        self.assertEqual((a["codec"], a["rate"], a["samples"]), (b["codec"], b["rate"], b["samples"]))
        self.assertEqual(load_bytes(out_py), load_bytes(out_go), "the decoded WAV must be identical")

    def test_both_sides_read_the_same_image_text(self):
        """The image *format* is shared: each side parses the other's text to the same bytes.

        The pixel reduction is not: Python's thumbnail encoder resamples with
        Pillow's Lanczos filter and Go's with a box filter, so the same source
        image gives two different (equally valid) texts.  Both sides read, train
        on, predict and decode either one - what differs is only which bytes a
        given picture turns into, and only for the ``tiny`` encoder.
        """
        from radixnet import vision

        encoded = go("image", "encode", self.png, "--size", 64, "--encoder", "tiny", model=self.model)
        self.assertEqual(encoded["encoder"], "tiny")
        self.assertEqual(encoded["latent_shape"], [3, 8, 8])
        # Python reads what Go wrote, byte for byte (parse_text needs no Pillow)
        name, width, height, payload, repaired = vision.parse_text(encoded["text"])
        self.assertEqual((name, width, height, repaired), ("tiny", 64, 64, False))
        self.assertEqual(len(payload), encoded["bytes"])
        # and Go reads what Python packs
        text = vision.pack_text("tiny", 64, 64, payload)
        self.assertEqual(text, encoded["text"])

    def test_both_recall_tutors_mark_the_same(self):
        py("--kind", "count", "--seed", 1, "speech", "teach", self.wav, "--text", "hello there",
           "--rate", 8000, "--train", "--epochs", 4, model=self.model)
        options = ("speech", "tutor", self.wav, "--text", "hello there", "--rate", 8000, "--length", 120)
        a = py(*options, model=self.model)
        b = go(*options, model=self.model)
        self.assertEqual(len(a["lessons"]), len(b["lessons"]))
        for first, second in zip(a["lessons"], b["lessons"]):
            self.assertEqual(first["grade"]["error"], second["grade"]["error"])
            self.assertEqual(first["grade"]["passed"], second["grade"]["passed"])
            self.assertAlmostEqual(first["grade"]["score"], second["grade"]["score"], places=9)
            self.assertEqual(first["grade"]["facts"]["expected_bytes"], second["grade"]["facts"]["expected_bytes"])
            self.assertAlmostEqual(first["grade"]["facts"]["agreement"], second["grade"]["facts"]["agreement"],
                                   places=9)
            self.assertEqual(first["grade"]["comment"], second["grade"]["comment"])
        self.assertEqual(a["report"]["passed"], b["report"]["passed"])
        self.assertAlmostEqual(a["report"]["mean_score"], b["report"]["mean_score"], places=9)


def _gradient_png(width, height):
    """A deterministic RGB gradient as a PNG, built by hand so the test needs no Pillow."""
    rows = b"".join(
        b"\x00" + b"".join(bytes(((x * 255) // width, (y * 255) // height, 128)) for x in range(width))
        for y in range(height)
    )

    def chunk(tag, body):
        payload = tag + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


def load_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


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
        # the Network settings tab: the encoder / decoder, identical on both sides
        status, enc, _ = self.client.get("/api/encoding")
        self.assertEqual(status, 200)
        from radixnet.api import ModelService  # the Python service answers the same document

        mine_enc = ModelService(saved["path"], backend="python")
        self.assertEqual(enc, mine_enc.encoding())
        for text in (CORPUS[0], CORPUS[0][:-4], "qqzzxx qqzz", "ab", ""):
            with self.subTest(text=text):
                status, prev, _ = self.client.post("/api/encoding/preview", {"text": text})
                self.assertEqual(status, 200, prev)
                theirs = mine_enc.encoding_preview(text)
                for key in ("windows", "count", "decoded", "round_trip", "chars", "unknown_windows"):
                    self.assertEqual(prev[key], theirs[key], key)
                self.assertEqual(prev["path"]["known"], theirs["path"]["known"])
                self.assertEqual(prev["path"]["labels"], theirs["path"]["labels"])
                self.assertEqual(prev["path"]["decoded"], theirs["path"]["decoded"])
                self.assertEqual(prev["path"]["nodes"], theirs["path"]["nodes"])
                self.assertEqual(prev["path"]["compressed"], theirs["path"]["compressed"])
                self.assertEqual(prev["path"]["reason"], theirs["path"]["reason"])
        # model selector: the one kind, other kinds refused
        status, m, _ = self.client.get("/api/model")
        self.assertEqual((status, m["kind"], m["in_memory"]), (200, "count", ["count"]))
        status, doc, _ = self.client.post("/api/model/select", {"kind": "radix"})
        self.assertEqual(status, 400)
        # what the Go server now serves too: evolve, the Ollama corpus / review, the automatic loop, images,
        # speech, code generation and tool use
        for path in ("/api/evolve/history", "/api/ollama/models", "/api/negative/auto/history",
                     "/api/images", "/api/speech", "/api/codegen/history", "/api/tools", "/api/agent/history",
                     "/api/chat/history"):
            status, doc, _ = self.client.get(path)
            self.assertEqual(status, 200, f"{path}: {doc}")
        # what is still Python-only is told so; unknown endpoints and wrong methods behave like the Python server
        status, doc, _ = self.client.post("/api/schedule/preview", {"lr_schedule": "lr"})
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
        # a large archive streams in as a raw body and streams through training in chunks
        big = io.BytesIO()
        with zipfile.ZipFile(big, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for part in range(3):
                z.writestr(f"vol{part}.txt", "".join(f"the cat number {i % 53} sat on mat {(i * 7) % 61}\n" for i in range(8000)))
        status, doc, _ = self.client.request(
            "POST", "/api/uploads?name=big.zip", raw=big.getvalue(), headers={"Content-Type": "application/zip"},
        )
        self.assertEqual(status, 201, doc)
        self.assertEqual((doc["uploads"][0]["files"], doc["uploads"][0]["lines"]), (3, 24000))
        status, doc, _ = self.client.post("/api/train", {"files": ["big.zip"], "epochs": 1, "chunk_size": 4096})
        self.assertEqual(status, 202, doc)
        job = self.wait_job()
        self.assertEqual((job["state"], job["history"][0]["chunks"]), ("done", 6))
        status, final, _ = self.client.get("/api/status")
        self.assertEqual(final["trained_texts"] - later["trained_texts"], 24000)
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
        self.assertEqual(restored.stats()["epochs_total"], final["epochs_total"])
        status, doc, _ = self.client.post("/api/uploads/delete", {"name": "notes.txt"})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()


class TestGoEncodingParity(unittest.TestCase):
    """The encoding dial is the same dial on both sides.

    Every encoding the two implementations offer - n-grams of any size, groups
    of letters, word grams - must build the same graph from the same corpus,
    write the same file, and read each other's.
    """

    ENCODINGS = ["char:3:1", "char:5:1", "char:4:4", "char:5:5", "char:6:3",
                 "word:1:1", "word:2:1", "word:3:1", "word:2:2"]

    def test_the_same_graph_from_the_same_corpus(self):
        for spec in self.ENCODINGS:
            with self.subTest(encoding=spec):
                tag = spec.replace(":", "-")
                py_model = os.path.join(TMP.name, f"enc-py-{tag}.json")
                go_model = os.path.join(TMP.name, f"enc-go-{tag}.json")
                py("--kind", "count", "--seed", 1, "--encoding", spec, "train",
                   "--data", CORPUS, "--epochs", 2, model=py_model)
                go("--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 2,
                   "--workers", 4, model=go_model)
                p, g = load_json(py_model)["graph"], load_json(go_model)["graph"]
                self.assertEqual(p.get("encoding"), g.get("encoding"), "the written encoding block")
                self.assertEqual(p["nodes"]["labels"], g["nodes"]["labels"])
                self.assertEqual(p["nodes"]["count"], g["nodes"]["count"])
                self.assertEqual((p["edges"]["src"], p["edges"]["dst"]), (g["edges"]["src"], g["edges"]["dst"]))
                self.assertEqual(p["edges"]["count"], g["edges"]["count"])
                assert_close(self, p["edges"]["w"], g["edges"]["w"], 1e-12)
                self.assertEqual(p["rng_state"], g["rng_state"])
                # only a non-default encoding is written at all
                self.assertEqual("encoding" in p, spec != "char:3:1", spec)

    def test_each_side_reads_the_other(self):
        spec = "word:2:1"
        py_model = os.path.join(TMP.name, "enc-cross-py.json")
        go_model = os.path.join(TMP.name, "enc-cross-go.json")
        py("--kind", "count", "--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 1,
           model=py_model)
        go("--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 1, model=go_model)
        # each side reports the other's file as the word bigram it is
        for reader, path in ((go, py_model), (py, go_model)):
            info = reader("info", model=path)
            self.assertEqual(info["stats"]["encoding"], spec)
            self.assertEqual((info["stats"]["ngram"], info["stats"]["stride"]), (2, 1))
        # and continues it: one more epoch on either side leaves the same graph
        py("--kind", "count", "train", "--data", CORPUS, "--epochs", 1, model=go_model)
        go("train", "--data", CORPUS, "--epochs", 1, model=py_model)
        a, b = load_json(py_model)["graph"], load_json(go_model)["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])
        self.assertEqual(a["nodes"]["count"], b["nodes"]["count"])
        self.assertEqual(a["encoding"], b["encoding"])

    def test_the_same_prediction_in_words(self):
        spec = "word:2:1"
        py_model = os.path.join(TMP.name, "enc-pred-py.json")
        go_model = os.path.join(TMP.name, "enc-pred-go.json")
        py("--kind", "count", "--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 2,
           model=py_model)
        go("--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 2, model=go_model)
        a = py("predict", "--prefix", "the cat", "--k", 3, "--length", 4, model=py_model)
        b = go("predict", "--prefix", "the cat", "--k", 3, "--length", 4, model=go_model)
        self.assertEqual(a["continuation"], b["continuation"])
        self.assertEqual(a["full_text"], b["full_text"])
        self.assertEqual([r["full_text"] for r in a["top"]], [r["full_text"] for r in b["top"]])
        assert_close(self, [r["cost"] for r in a["top"]], [r["cost"] for r in b["top"]])
        # the words come back whole, with the space the prefix needs
        self.assertTrue(a["full_text"].startswith("the cat "), a["full_text"])
        self.assertEqual(a["continuation"], " ".join(a["continuation"].split()))

    def test_both_refuse_the_same_nonsense(self):
        model = os.path.join(TMP.name, "enc-bad.json")
        for spec in ("rune:3", "char:2:3", "char:0"):
            with self.subTest(spec=spec):
                self.assertTrue(py("--encoding", spec, "info", model=model, expect=1)["error"])
                self.assertTrue(go("--encoding", spec, "info", model=model, expect=1)["error"])
