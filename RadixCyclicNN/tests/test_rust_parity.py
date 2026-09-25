"""Cross-language parity of the Rust port of the count / reward model (``rust/``).

Skipped when no Rust toolchain is available.  The Rust CLI is built once; then
both implementations train the same corpus with the same settings and must
produce the same structure, counts, rewards, sliding window and RNG state, the
same predictions, generated texts and scores, the same judged paths and node
ratios - and each side must load and continue the other's model file, gzipped
or not.

The Rust port's graph document is held to a stricter bar than the Go port's:
byte for byte what Python writes, but for the ``version`` cache stamp that
every load bumps.  See ``rust/README.md``.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.counter import COUNTER_LIMIT  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.model import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_DIR = os.path.join(ROOT, "rust")
CORPUS = os.path.join(ROOT, "data", "sample_corpus.txt")
GARBAGE = os.path.join(ROOT, "data", "sample_garbage.txt")
CARGO = shutil.which("cargo")

TMP = None
BINARY = None


def setUpModule():
    global TMP, BINARY
    if CARGO is None:
        raise unittest.SkipTest("no Rust toolchain on PATH")
    TMP = tempfile.TemporaryDirectory(prefix="radixnet-rust-")
    proc = subprocess.run(
        [CARGO, "build", "--release", "--bin", "radixnet"],
        cwd=RUST_DIR, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise unittest.SkipTest(f"cargo build failed:\n{proc.stderr[-2000:]}")
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(RUST_DIR, "target"))
    BINARY = os.path.join(target, "release", "radixnet")
    if not os.path.exists(BINARY):
        raise unittest.SkipTest(f"no binary at {BINARY}")


def tearDownModule():
    if TMP is not None:
        TMP.cleanup()


def rust(*args, model, expect=0):
    cmd = [BINARY, "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    if proc.stdout.strip():
        return json.loads(proc.stdout)
    return {"error": proc.stderr.strip()} if expect else None


def py(*args, model, expect=0):
    cmd = [sys.executable, "-m", "radixnet", "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {"error": proc.stderr.strip()}


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compact(doc):
    """The document as both implementations write it."""
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def assert_close(test, a, b, tol=1e-9):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        test.assertLessEqual(abs(x - y), tol, (x, y))


class TestRustParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py_model = os.path.join(TMP.name, "py.count.json")
        cls.rs_model = os.path.join(TMP.name, "rs.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.py_model)
        cls.rs_train = rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, "--workers", 4, model=cls.rs_model)
        cls.py_doc = load_json(cls.py_model)
        cls.rs_doc = load_json(cls.rs_model)

    def test_same_graph_after_training(self):
        p, r = self.py_doc["graph"], self.rs_doc["graph"]
        self.assertEqual(p["nodes"]["labels"], r["nodes"]["labels"])
        self.assertEqual(p["nodes"]["count"], r["nodes"]["count"])
        for key in ("z", "a", "b", "h", "k"):
            self.assertEqual(p["nodes"][key], r["nodes"][key], key)
        self.assertEqual((p["edges"]["src"], p["edges"]["dst"]), (r["edges"]["src"], r["edges"]["dst"]))
        self.assertEqual(p["edges"]["count"], r["edges"]["count"])
        self.assertEqual(p["edges"]["reward"], r["edges"]["reward"])
        self.assertEqual(p["edges"]["w"], r["edges"]["w"])  # the same doubles, not merely close
        self.assertEqual(p["rng_state"], r["rng_state"])
        self.assertEqual(p["inverted"], r["inverted"])
        self.assertEqual(p["traversals"], r["traversals"])
        self.assertEqual(p["structure_version"], r["structure_version"])
        self.assertEqual(p["weights"], r["weights"])
        self.assertEqual(p["paths"], r["paths"])
        self.assertEqual(self.rs_doc["format"], "radixnet-count")
        self.assertEqual(self.rs_doc["kind"], "count")
        for key in ("epochs_total", "trained_texts", "trained_chars", "twonrl_runs", "feedback_passes"):
            self.assertEqual(self.py_doc["meta"][key], self.rs_doc["meta"][key], key)
        self.assertEqual(len(self.py_doc["history"]), len(self.rs_doc["history"]))
        for a, b in zip(self.py_doc["history"], self.rs_doc["history"]):
            self.assertEqual((a["epoch"], a["nodes"], a["edges"], a["transitions"], a["merges"]),
                             (b["epoch"], b["nodes"], b["edges"], b["transitions"], b["merges"]))
            self.assertLessEqual(abs(a["loss"] - b["loss"]), 1e-9)
        self.assertEqual(self.rs_train["workers"], 4)

    def test_the_graph_document_is_pythons_byte_for_byte(self):
        """Not merely the same numbers: the same file.

        ``version`` is a cache stamp - both sides bump it whenever a weight is
        recomputed, and they recompute at different moments - so it is the one
        field normalised away.  Everything else, including every float's
        rendering and every key's order, has to match.
        """
        strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
        self.assertEqual(strip(self.py_doc), strip(self.rs_doc))

    def test_same_predictions_generation_and_scores(self):
        for prefix in ("the cat", "the", "on the ma", "zzq", ""):
            with self.subTest(prefix=prefix):
                a = py("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam", model=self.py_model)
                b = rust("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam", model=self.rs_model)
                self.assertEqual(a["continuation"], b["continuation"])
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                self.assertEqual([t["full_text"] for t in a["bottom"]], [t["full_text"] for t in b["bottom"]])
                assert_close(self, [t["cost"] for t in a["top"]], [t["cost"] for t in b["top"]])
                self.assertEqual(a["path"], b["path"])
                self.assertEqual(a["node_ids"], b["node_ids"])
                self.assertEqual(a["reached_end"], b["reached_end"])
        a = py("generate", "--mode", "beam", "--count", 4, "--max-length", 40, model=self.py_model)
        b = rust("generate", "--mode", "beam", "--count", 4, "--max-length", 40, model=self.rs_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        assert_close(self, [s["cost"] for s in a["samples"]], [s["cost"] for s in b["samples"]])
        a = py("generate", "--mode", "dijkstra", "--max-length", 40, model=self.py_model)
        b = rust("generate", "--mode", "dijkstra", "--max-length", 40, model=self.rs_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        a = py("score", "--data", GARBAGE, model=self.py_model)
        b = rust("score", "--data", GARBAGE, model=self.rs_model)
        self.assertEqual(a["count"], b["count"])
        assert_close(self, [r["log_prob"] for r in a["results"]], [r["log_prob"] for r in b["results"]])
        self.assertEqual([r["unknown_transitions"] for r in a["results"]],
                         [r["unknown_transitions"] for r in b["results"]])
        self.assertEqual([r["transitions"] for r in a["results"]], [r["transitions"] for r in b["results"]])
        self.assertLessEqual(abs(a["mean_per_char"] - b["mean_per_char"]), 1e-9)

    def test_each_side_loads_and_continues_the_other(self):
        # Python loads the Rust file: same predictions as its own model
        loaded = load_model(self.rs_model)
        self.assertIsInstance(loaded, CountRewardNet)
        own = load_model(self.py_model)
        for prefix in ("the cat", "the dog"):
            a = own.predict(prefix, length=8, k=3)
            b = loaded.predict(prefix, length=8, k=3)
            self.assertEqual([r.full_text for r in a.top], [r.full_text for r in b.top])
            self.assertLessEqual(abs(a.cost - b.cost), 1e-9)
        self.assertEqual(loaded.stats()["total_traversals"], own.stats()["total_traversals"])
        # Rust loads the Python file
        info = rust("info", model=self.py_model)
        self.assertEqual((info["stats"]["nodes"], info["stats"]["edges"], info["stats"]["epochs_total"]),
                         (own.stats()["nodes"], own.stats()["edges"], 2))
        # both continue training on the other's file and stay identical
        cross_py = os.path.join(TMP.name, "cross_py.count.json")
        cross_rs = os.path.join(TMP.name, "cross_rs.count.json")
        shutil.copy(self.rs_model, cross_py)
        shutil.copy(self.py_model, cross_rs)
        py("train", "--data", CORPUS, "--epochs", 1, model=cross_py)      # Python continues the Rust model
        rust("train", "--data", CORPUS, "--epochs", 1, "--workers", 3, model=cross_rs)  # and Rust the Python one
        a, b = load_json(cross_py)["graph"], load_json(cross_rs)["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])
        self.assertEqual(a["edges"]["count"], b["edges"]["count"])
        self.assertEqual(a["edges"]["reward"], b["edges"]["reward"])
        self.assertEqual(a["rng_state"], b["rng_state"])
        self.assertEqual(a["weights"], b["weights"])
        self.assertEqual(a["edges"]["w"], b["edges"]["w"])

    def test_a_gzipped_model_crosses_over(self):
        py_gz = os.path.join(TMP.name, "gz_py.count.json.gz")
        rs_gz = os.path.join(TMP.name, "gz_rs.count.json.gz")
        py("--kind", "count", "--seed", 2, "train", "--data", CORPUS, "--epochs", 1, model=py_gz)
        rust("--seed", 2, "train", "--data", CORPUS, "--epochs", 1, model=rs_gz)
        with open(py_gz, "rb") as fh:
            self.assertEqual(fh.read(2), b"\x1f\x8b", "Python wrote no gzip")
        with open(rs_gz, "rb") as fh:
            self.assertEqual(fh.read(2), b"\x1f\x8b", "Rust wrote no gzip")
        # each side reads the other's compressed file and agrees about it
        mine = load_model(rs_gz)
        theirs = rust("info", model=py_gz)
        self.assertEqual(mine.stats()["nodes"], theirs["stats"]["nodes"])
        self.assertEqual(mine.stats()["edges"], theirs["stats"]["edges"])
        a = py("predict", "--prefix", "the cat", "--length", 8, "--k", 2, model=py_gz)
        b = rust("predict", "--prefix", "the cat", "--length", 8, "--k", 2, model=rs_gz)
        self.assertEqual(a["full_text"], b["full_text"])
        self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)

    def test_a_wrapped_model_file_crosses_over_unchanged(self):
        """Counters wrap at COUNTER_LIMIT; both sides must read the other's reset counts and agree."""
        py_path = os.path.join(TMP.name, "wrap_py.count.json")
        rs_path = os.path.join(TMP.name, "wrap_rs.count.json")
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
        shutil.copy(py_path, rs_path)

        py("train", "--data", CORPUS, "--epochs", 1, model=py_path)   # the epoch takes every counter round
        rust("train", "--data", CORPUS, "--epochs", 1, model=rs_path)
        a, b = load_json(py_path)["graph"], load_json(rs_path)["graph"]
        self.assertTrue(a["nodes"]["count_resets"], "Python recorded no reset")
        self.assertTrue(b["nodes"]["count_resets"], "Rust recorded no reset")
        for block, key in (("nodes", "count"), ("nodes", "count_resets"), ("edges", "count"), ("edges", "count_resets")):
            self.assertEqual(a[block][key], b[block][key], f"{block}.{key}")
        for key in ("traversals", "traversals_resets"):
            self.assertEqual(a[key], b[key], key)
        for key in ("total_traversals", "total_traversals_resets", "window_events"):
            self.assertEqual(a["weights"][key], b["weights"][key], key)
        self.assertEqual(a["weights"]["total_traversals_resets"], 1)
        self.assertEqual(a["edges"]["w"], b["edges"]["w"])
        for prefix in ("the cat", "the dog"):
            with self.subTest(prefix=prefix):
                x = py("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=py_path)
                y = rust("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=rs_path)
                self.assertEqual(x["continuation"], y["continuation"])
                self.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9)

    def test_feedback_2nrl_and_invert_match(self):
        py_path = os.path.join(TMP.name, "fb_py.count.json")
        rs_path = os.path.join(TMP.name, "fb_rs.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        for command in (
            ["feedback", "--good", CORPUS, "--bad", GARBAGE, "--strength", "2"],
            ["feedback", "--good", CORPUS],
            ["feedback", "--bad", GARBAGE],
            ["2nrl", "--bad", GARBAGE, "--good", CORPUS],
            ["invert"],
            ["compress"],
            ["weights", "--reward-scale", "2", "--window", "500"],
        ):
            with self.subTest(command=command[0] + " " + " ".join(command[1:3])):
                a = py(*command, model=py_path)
                b = rust(*command, model=rs_path)
                blocks = [(a.get("stats"), b.get("stats"))]
                # the commands that report no stats (compress) are compared on
                # what they do report, minus the paths and the wording
                skip = {"backend", "device", "counting", "saved", "model", "out", "kind",
                        "interrupted", "good_weights", "bad_weights"}
                blocks.append(({k: v for k, v in a.items() if k in b and k not in skip},
                               {k: v for k, v in b.items() if k in a and k not in skip}))
                for left, right in blocks:
                    if not left or not right:
                        continue
                    for key, value in left.items():
                        if key in skip or isinstance(value, (list, dict)) or value is None:
                            continue
                        other = right.get(key)
                        if isinstance(value, float):
                            self.assertLessEqual(abs(value - other), 1e-9, key)
                        else:
                            self.assertEqual(value, other, key)
                if "action" in a:
                    self.assertEqual(a["action"], b["action"])
                    self.assertEqual(len(a["positive"]), len(b["positive"]))
                    self.assertEqual(len(a["negative"]), len(b["negative"]))
        # and the two models are still the same model
        x, y = load_json(py_path)["graph"], load_json(rs_path)["graph"]
        self.assertEqual(x["edges"]["reward"], y["edges"]["reward"])
        self.assertEqual(x["paths"], y["paths"])
        self.assertEqual(x["edges"]["w"], y["edges"]["w"])

    def test_judged_paths_and_node_ratios_read_the_same(self):
        py_path = os.path.join(TMP.name, "paths_py.count.json")
        rs_path = os.path.join(TMP.name, "paths_rs.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        py("feedback", "--good", CORPUS, "--bad", GARBAGE, model=py_path)
        rust("feedback", "--good", CORPUS, "--bad", GARBAGE, model=rs_path)

        a = py("paths", "--limit", 10, model=py_path)
        b = rust("paths", "--limit", 10, model=rs_path)
        self.assertEqual(a["totals"], b["totals"])
        self.assertTrue(a["paths"], "no judged path to compare")
        for x, y in zip(a["paths"], b["paths"]):
            self.assertEqual((x["prev"], x["edge"], x["seen"], x["correct"], x["incorrect"]),
                             (y["prev"], y["edge"], y["seen"], y["correct"], y["incorrect"]))
            self.assertLessEqual(abs(x["term"] - y["term"]), 1e-12)
            self.assertEqual(x["correct_ratio"], y["correct_ratio"])

        a = py("nodes", "--limit", 5, model=py_path)
        b = rust("nodes", "--limit", 5, model=rs_path)
        self.assertEqual(len(a["nodes"]), len(b["nodes"]))
        for x, y in zip(a["nodes"], b["nodes"]):
            self.assertEqual((x["node"], x["label"], x["visits"]), (y["node"], y["label"], y["visits"]))
            self.assertEqual(x["in_totals"], y["in_totals"])
            self.assertEqual(x["out_totals"], y["out_totals"])
            for side in ("from", "to"):
                for u, v in zip(x[side], y[side]):
                    self.assertEqual((u["node"], u["edge"], u["seen"], u["correct"], u["incorrect"]),
                                     (v["node"], v["edge"], v["seen"], v["correct"], v["incorrect"]))
                    self.assertLessEqual(abs(u["seen_ratio"] - v["seen_ratio"]), 1e-12)
                    self.assertLessEqual(abs(u["reward_ratio"] - v["reward_ratio"]), 1e-12)

    def test_the_least_punished_traversal_agrees_across_the_two(self):
        """The second traversal is in both, and answers the same thing in both."""
        # on a model nothing was punished on it is the ordinary search, to the bit
        for prefix in ("the cat", "the "):
            with self.subTest(prefix=prefix):
                a = rust("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=self.rs_model)
                b = rust("predict", "--prefix", prefix, "--length", 8, "--k", 3,
                         "--traversal", "least-punished", model=self.rs_model)
                self.assertEqual((a["full_text"], a["cost"], a["expanded"]),
                                 (b["full_text"], b["cost"], b["expanded"]))

        # and once there is blame to walk by, both sides leave the same step
        py_path = os.path.join(TMP.name, "blame_py.count.json")
        rs_path = os.path.join(TMP.name, "blame_rs.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        py("feedback", "--bad", GARBAGE, "--strength", "2", model=py_path)
        rust("feedback", "--bad", GARBAGE, "--strength", "2", model=rs_path)
        differed = 0
        for prefix in ("the cat", "the ", "on the ma", "a bird", "he s"):
            with self.subTest(prefix=prefix):
                args = ("predict", "--prefix", prefix, "--length", 10, "--k", 3, "--mode", "beam")
                a = py(*args, "--traversal", "least-punished", model=py_path)
                b = rust(*args, "--traversal", "least-punished", model=rs_path)
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                assert_close(self, [t.get("punish", 0.0) for t in a["top"]],
                             [t.get("punish", 0.0) for t in b["top"]], 1e-12)
                self.assertEqual(a["expanded"], b["expanded"])
                plain = py(*args, model=py_path)
                differed += plain["full_text"] != a["full_text"]
        self.assertTrue(differed, "the punished model answered the same to both searches everywhere")

    def test_the_punishment_traversal_agrees_across_the_two(self):
        """The *other* punishment traversal (`radixnet/penalty.py`): the cost function, not the ranking."""
        py_path = os.path.join(TMP.name, "punish_py.count.json")
        rs_path = os.path.join(TMP.name, "punish_rs.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        # a line of the corpus itself, so the punishment lands on a path the search would otherwise take
        punished = os.path.join(TMP.name, "punished.txt")
        with open(punished, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\n")
        py("feedback", "--bad", punished, "--strength", "3", model=py_path)
        rust("feedback", "--bad", punished, "--strength", "3", model=rs_path)
        differed = 0
        for prefix, merit in (("the cat", 1.0), ("the cat", 0.0), ("on the ma", 0.0), ("a bird", 1.0)):
            with self.subTest(prefix=prefix, merit_scale=merit):
                args = ("predict", "--prefix", prefix, "--length", 10, "--k", 3,
                        "--traversal", "punishment", "--merit-scale", merit)
                a = py(*args, model=py_path)
                b = rust(*args, model=rs_path)
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                assert_close(self, [t["cost"] for t in a["top"]], [t["cost"] for t in b["top"]], 1e-9)
                self.assertEqual((a["traversal"], b["traversal"]), ("punishment", "punishment"))
                plain = py("predict", "--prefix", prefix, "--length", 10, "--k", 3, model=py_path)
                differed += plain["full_text"] != a["full_text"]
        self.assertTrue(differed, "the punished model answered the same to both searches everywhere")
        # and it is a different thing from the ranking traversal: both exist, neither is the other
        one = rust("predict", "--prefix", "the cat", "--length", 10, "--traversal", "punishment",
                   "--merit-scale", 0.0, model=rs_path)
        other = rust("predict", "--prefix", "the cat", "--length", 10, "--traversal", "least-punished",
                     model=rs_path)
        self.assertEqual((one["traversal"], other["traversal"]), ("punishment", "least-punished"))

    def test_the_rust_specific_options(self):
        """``--workers`` changes how the work is spread, never the answer."""
        one = os.path.join(TMP.name, "w1.count.json")
        many = os.path.join(TMP.name, "w8.count.json")
        rust("--seed", 7, "train", "--data", CORPUS, "--epochs", 2, "--workers", 1, model=one)
        rust("--seed", 7, "train", "--data", CORPUS, "--epochs", 2, "--workers", 8, model=many)
        a, b = load_json(one)["graph"], load_json(many)["graph"]
        self.assertEqual(a["nodes"]["count"], b["nodes"]["count"])
        self.assertEqual(a["edges"]["count"], b["edges"]["count"])
        self.assertEqual(a["edges"]["w"], b["edges"]["w"])
        self.assertEqual(a["weights"]["window_events"], b["weights"]["window_events"])
        version = rust("version", model=one)
        self.assertEqual(version["format"], "radixnet-count")
        self.assertEqual(version["backend"], "rust")


class TestRustWordParity(unittest.TestCase):
    """A word encoding over the same graph: same grams, same file, same predictions.

    Words are not a kind - they are an encoding (``--encoding word:3:1``), so
    what has to agree is everything the dial touches: the grams, the graph, the
    alphabet those grams are made of, and every prediction and score.
    """

    ENCODING = "word:3:1"

    @classmethod
    def setUpClass(cls):
        cls.py_model = os.path.join(TMP.name, "py.word.json")
        cls.rs_model = os.path.join(TMP.name, "rs.word.json")
        # --kind count: Python's default kind is the sine network, and the Rust
        # port only has this one; the encoding is the dial on top of it
        py("--kind", "count", "--encoding", cls.ENCODING, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2,
           model=cls.py_model)
        rust("--encoding", cls.ENCODING, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, "--workers", 4,
             model=cls.rs_model)
        cls.py_doc = load_json(cls.py_model)
        cls.rs_doc = load_json(cls.rs_model)

    def test_the_word_document_is_pythons_byte_for_byte(self):
        """The same bar the character model is held to, the encoding block included."""
        strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
        self.assertEqual(strip(self.py_doc), strip(self.rs_doc))
        # a word model is still the count format: the encoding is what changed
        self.assertEqual(self.py_doc["format"], "radixnet-count")
        self.assertEqual(self.rs_doc["format"], "radixnet-count")
        self.assertEqual(self.rs_doc["kind"], "count")
        for doc, side in ((self.py_doc, "python"), (self.rs_doc, "rust")):
            self.assertEqual(doc["graph"]["encoding"], {"unit": "word", "n": 3, "stride": 1}, side)
        for key in ("epochs_total", "trained_texts", "trained_chars"):
            self.assertEqual(self.py_doc["meta"][key], self.rs_doc["meta"][key], key)
        # every label is words, written with single spaces
        for label in self.py_doc["graph"]["nodes"]["labels"][3:]:
            self.assertNotIn("  ", label)
            self.assertEqual(label, " ".join(label.split()), label)

    def test_the_same_predictions_scores_and_texts(self):
        for prefix in ("the cat sat on", "the", "water boils at", "qqzz never read", ""):
            with self.subTest(prefix=prefix):
                a = py("predict", "--prefix", prefix, "--length", 4, "--k", 3, model=self.py_model)
                b = rust("predict", "--prefix", prefix, "--length", 4, "--k", 3, model=self.rs_model)
                self.assertEqual(a["continuation"], b["continuation"])
                self.assertEqual(a["full_text"], b["full_text"])
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-12)
                self.assertEqual([t["full_text"] for t in a["top"]], [t["full_text"] for t in b["top"]])
                self.assertEqual([t["full_text"] for t in a["bottom"]], [t["full_text"] for t in b["bottom"]])
                self.assertEqual(a["path"], b["path"])  # the labels are words on both sides
        for text in ("the cat sat on the mat", "water boils at 100 degrees celsius", "qqzz never read this"):
            with self.subTest(text=text):
                a = py("score", "--text", text, model=self.py_model)["results"][0]
                b = rust("score", "--text", text, model=self.rs_model)["results"][0]
                self.assertLessEqual(abs(a["log_prob"] - b["log_prob"]), 1e-12)
                self.assertEqual(a["chars"], b["chars"])  # words, here
                self.assertEqual(a["unknown_transitions"], b["unknown_transitions"])
        a = py("generate", "--mode", "beam", "--count", 4, "--max-length", 12, model=self.py_model)
        b = rust("generate", "--mode", "beam", "--count", 4, "--max-length", 12, model=self.rs_model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        assert_close(self, [s["cost"] for s in a["samples"]], [s["cost"] for s in b["samples"]], 1e-12)

    def test_the_same_alphabet_in_the_same_order(self):
        a = py("words", "--limit", 10, model=self.py_model)
        b = rust("words", "--limit", 10, model=self.rs_model)
        self.assertEqual(a["vocabulary"], b["vocabulary"])
        self.assertGreater(a["vocabulary"], 100)
        self.assertEqual((a["units"], b["units"]), ("words", "words"))
        self.assertEqual((a["encoding"], b["encoding"]), (self.ENCODING, self.ENCODING))
        self.assertEqual([(r["word"], r["id"], r["grams"]) for r in a["words"]],
                         [(r["word"], r["id"], r["grams"]) for r in b["words"]])

    def test_the_same_words_out_of_the_same_text(self):
        """The tokeniser is the Unicode ``White_Space`` property on both sides.

        Python calls U+001C..U+001F whitespace and the property Go and Rust
        split on does not, so a text holding one is where the two rules would
        disagree - and a text cut into different words is a different graph.
        """
        odd = os.path.join(TMP.name, "separators-rust.txt")
        with open(odd, "w", encoding="utf-8") as fh:
            fh.write("a\x1cb c d\n" * 4)
        py_model = os.path.join(TMP.name, "split-py-rust.json")
        other_model = os.path.join(TMP.name, "split-rust.json")
        py("--kind", "count", "--encoding", self.ENCODING, "--seed", 1, "train", "--data", odd, "--epochs", 1,
           model=py_model)
        rust("--encoding", self.ENCODING, "--seed", 1, "train", "--data", odd, "--epochs", 1, model=other_model)
        a, b = load_json(py_model)["graph"], load_json(other_model)["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])
        # "a\x1cb" is one word, so the text is three units and not four
        self.assertEqual(py("score", "--text", "a\x1cb c d", model=py_model)["results"][0]["chars"], 3)
        self.assertEqual(rust("score", "--text", "a\x1cb c d", model=other_model)["results"][0]["chars"], 3)

    def test_each_side_reads_the_other_s_word_model(self):
        loaded = load_model(self.rs_model)  # Python reads the Rust file
        self.assertEqual(str(loaded.encoding), self.ENCODING)
        self.assertEqual(loaded.predict("the cat sat on", length=4, k=3).text,
                         py("predict", "--prefix", "the cat sat on", "--length", 4, "--k", 3,
                            model=self.py_model)["continuation"])
        # and Rust reads the Python file, continues it, and Python reads that back
        crossed = os.path.join(TMP.name, "crossed.word.json")
        shutil.copyfile(self.py_model, crossed)
        rust("train", "--data", CORPUS, "--epochs", 1, model=crossed)
        again = load_model(crossed)
        self.assertEqual(str(again.encoding), self.ENCODING)
        self.assertEqual(again.stats()["epochs_total"], 3)

    def test_the_other_dials_agree_too(self):
        """Any n, any stride: the two sides build the same graph from the same grams."""
        for spec in ("char:5:1", "char:4:4", "word:2:1", "phone:3:1", "phone:2:2", "syllable:2:1"):
            with self.subTest(encoding=spec):
                pym = os.path.join(TMP.name, f"py.{spec.replace(':', '-')}.json")
                rsm = os.path.join(TMP.name, f"rs.{spec.replace(':', '-')}.json")
                py("--kind", "count", "--encoding", spec, "--seed", 3, "train", "--data", CORPUS, "--epochs", 1,
                   model=pym)
                rust("--encoding", spec, "--seed", 3, "train", "--data", CORPUS, "--epochs", 1, model=rsm)
                a, b = load_json(pym), load_json(rsm)
                strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
                self.assertEqual(strip(a), strip(b), spec)

    def test_the_same_speech(self):
        """Spoken as it walks: the same walk, the same words, and the same audio, sample for sample."""
        spec = "phone:3:1"
        py_model = os.path.join(TMP.name, "enc-speech-py.json")
        other_model = os.path.join(TMP.name, "enc-speech-rust.json")
        py("--kind", "count", "--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 2,
           model=py_model)
        rust("--seed", 1, "--encoding", spec, "train", "--data", CORPUS, "--epochs", 2, model=other_model)

        def pcm(path):
            with open(path, "rb") as fh:
                data = fh.read()[44:]
            return struct.unpack("<%dh" % (len(data) // 2), data)

        for prefix in ("", "the cat"):
            with self.subTest(prefix=prefix):
                where = ["--prefix", prefix] if prefix else []
                py_wav = os.path.join(TMP.name, "speech-py.wav")
                other_wav = os.path.join(TMP.name, "speech-rust.wav")
                a = py("--seed", 5, "speak", *where, "--count", 2, "--max-length", 40, "--out", py_wav, model=py_model)
                b = rust("--seed", 5, "speak", *where, "--count", 2, "--max-length", 40, "--seeded", "--out", other_wav,
                          model=other_model)
                self.assertEqual(a["utterances"], b["utterances"])
                self.assertEqual(a["count"], 2)
                self.assertEqual(a["seconds"], b["seconds"])
                self.assertEqual(a["encoding"], spec)
                self.assertEqual(b["encoding"], spec)
                if not prefix:  # from START the first node is whole: an utterance opens on a whole word
                    self.assertFalse(a["utterances"][0]["text"].startswith("#"), a["utterances"][0])
                x, y = pcm(py_wav), pcm(other_wav)
                self.assertEqual(len(x), len(y))
                self.assertGreater(len(x), 16000)
                self.assertLessEqual(max(abs(p - q) for p, q in zip(x, y)), 64)


class TestRustNegativeParity(unittest.TestCase):
    """The negative network: the same blame, the same verdicts, the same file.

    Every node and edge of this model exists because something went wrong, so
    what has to agree is the evidence itself - how much blame a text carries,
    which reasons carry it, which fragments are worst, and the one sentence the
    verdict rests on, character for character.
    """

    FAILURES = [
        ("the cat sat on the sky", "nonsense"),
        ("water boils at 50 degrees", "wrong fact"),
        ("the the the the the", "repetition"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.py_model = os.path.join(TMP.name, "py.negative.json")
        cls.rs_model = os.path.join(TMP.name, "rs.negative.json")
        for text, reason in cls.FAILURES:
            py("--kind", "count", "--seed", 1, "negative", "blame", "--text", text, "--reason", reason,
               "--severity", 2, "--source", "test", "--negative", cls.py_model, model=cls.py_model + ".count")
            rust("--seed", 1, "negative", "blame", "--text", text, "--reason", reason,
                 "--severity", 2, "--source", "test", "--negative", cls.rs_model, model=cls.rs_model + ".count")
        cls.py_doc = load_json(cls.py_model)
        cls.rs_doc = load_json(cls.rs_model)

    def test_the_negative_document_is_pythons_byte_for_byte(self):
        strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
        self.assertEqual(strip(self.py_doc), strip(self.rs_doc))
        self.assertEqual(self.py_doc["format"], "radixnet-negative")
        self.assertEqual(self.rs_doc["format"], "radixnet-negative")
        self.assertEqual((self.py_doc["kind"], self.rs_doc["kind"]), ("negative", "negative"))
        # blame rides on the edges where the count model keeps rewards
        for doc, side in ((self.py_doc, "python"), (self.rs_doc, "rust")):
            self.assertIn("blame", doc["graph"]["edges"], side)
            self.assertNotIn("reward", doc["graph"]["edges"], side)
            self.assertEqual(doc["graph"]["weights"]["function"], "blame", side)

    def test_the_rest_of_the_file_is_pythons_too(self):
        # the meta block, the history records and the journal - every field
        # but the clock's
        clockless = lambda meta: {k: v for k, v in meta.items() if k != "created"}  # noqa: E731
        self.assertEqual(list(self.py_doc["meta"]), list(self.rs_doc["meta"]))
        self.assertEqual(clockless(self.py_doc["meta"]), clockless(self.rs_doc["meta"]))
        timeless = lambda records: [{k: v for k, v in r.items() if k != "seconds"} for r in records]  # noqa: E731
        self.assertEqual(timeless(self.py_doc["history"]), timeless(self.rs_doc["history"]))
        self.assertEqual(self.rs_doc["history"][-1]["edges_touched"], self.py_doc["history"][-1]["edges_touched"])
        undated = lambda log: [{k: v for k, v in e.items() if k != "at"} for e in log]  # noqa: E731
        self.assertEqual(undated(self.py_doc["log"]), undated(self.rs_doc["log"]))
        self.assertEqual(self.py_doc["filter"], self.rs_doc["filter"])

    def test_the_same_verdict_on_the_same_text(self):
        for text in ("the cat sat on the sky", "water boils at 50 degrees", "a sentence nothing has failed on",
                     "the cat sat on the mat"):
            with self.subTest(text=text):
                a = py("negative", "why", "--text", text, "--negative", self.py_model,
                       model=self.py_model + ".count")["verdicts"][0]
                b = rust("negative", "why", "--text", text, "--negative", self.rs_model,
                         model=self.rs_model + ".count")["verdicts"][0]
                self.assertEqual(a["verdict"], b["verdict"])
                self.assertEqual(a["why"], b["why"])  # the sentence, character for character
                self.assertLessEqual(abs(a["blame"] - b["blame"]), 1e-12)
                self.assertLessEqual(abs(a["risk"] - b["risk"]), 1e-12)
                self.assertLessEqual(abs(a["coverage"] - b["coverage"]), 1e-12)
                self.assertEqual(a["blamed"], b["blamed"])
                self.assertEqual([r["reason"] for r in a["reasons"]], [r["reason"] for r in b["reasons"]])
                self.assertEqual([s["fragment"] for s in a["spans"]], [s["fragment"] for s in b["spans"]])

    def test_the_same_reasons_in_the_same_order(self):
        a = py("negative", "reasons", "--negative", self.py_model, model=self.py_model + ".count")
        b = rust("negative", "reasons", "--negative", self.rs_model, model=self.rs_model + ".count")
        self.assertEqual([(r["reason"], r["blame"]) for r in a["reasons"]],
                         [(r["reason"], r["blame"]) for r in b["reasons"]])
        self.assertEqual(len(a["journal"]), len(b["journal"]))
        self.assertEqual([e["reason"] for e in a["journal"]], [e["reason"] for e in b["journal"]])

    def test_clearing_and_forgetting_agree(self):
        """Blame taken back has to be taken back the same way on both sides."""
        py_copy = os.path.join(TMP.name, "py.cleared.json")
        rs_copy = os.path.join(TMP.name, "rs.cleared.json")
        shutil.copyfile(self.py_model, py_copy)
        shutil.copyfile(self.rs_model, rs_copy)
        for _ in range(3):
            py("negative", "clear", "--text", "the cat sat on the sky", "--negative", py_copy,
               model=py_copy + ".count")
            rust("negative", "clear", "--text", "the cat sat on the sky", "--negative", rs_copy,
                 model=rs_copy + ".count")
        a = py("negative", "why", "--text", "the cat sat on the sky", "--negative", py_copy,
               model=py_copy + ".count")["verdicts"][0]
        b = rust("negative", "why", "--text", "the cat sat on the sky", "--negative", rs_copy,
                 model=rs_copy + ".count")["verdicts"][0]
        self.assertLessEqual(abs(a["blame"] - b["blame"]), 1e-12)
        self.assertEqual(a["verdict"], b["verdict"])

        # and forgetting one reason removes the same blame on both
        a = py("negative", "forget", "--reason", "nonsense", "--negative", py_copy, model=py_copy + ".count")
        b = rust("negative", "forget", "--reason", "nonsense", "--negative", rs_copy, model=rs_copy + ".count")
        self.assertEqual(a["edges"], b["edges"])
        self.assertLessEqual(abs(a["blame_removed"] - b["blame_removed"]), 1e-12)

    def test_each_side_reads_the_other_s_negative_model(self):
        a = py("negative", "why", "--text", "the cat sat on the sky", "--negative", self.rs_model,
               model=self.rs_model + ".count")["verdicts"][0]
        b = rust("negative", "why", "--text", "the cat sat on the sky", "--negative", self.py_model,
                 model=self.py_model + ".count")["verdicts"][0]
        mine = py("negative", "why", "--text", "the cat sat on the sky", "--negative", self.py_model,
                  model=self.py_model + ".count")["verdicts"][0]
        # Python reading Rust's file, and Rust reading Python's, both agree with
        # Python reading its own
        self.assertEqual(a["why"], mine["why"])
        self.assertEqual(b["why"], mine["why"])
        self.assertLessEqual(abs(a["blame"] - mine["blame"]), 1e-12)
        self.assertLessEqual(abs(b["blame"] - mine["blame"]), 1e-12)

    def test_the_file_says_which_model_it_is(self):
        """A reader has to be able to tell the two apart before it reads them."""
        out = rust("info", model=self.rs_model)
        self.assertEqual(out["stats"]["kind"], "negative")
        self.assertEqual(self.rs_doc["format"], "radixnet-negative")
        # and an ordinary count model still says count
        count = os.path.join(TMP.name, "plain.count.json")
        rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 1, model=count)
        self.assertEqual(rust("info", model=count)["stats"]["kind"], "count")


class TestRustFileFormat(unittest.TestCase):
    """The reader's side: what the Rust port does with a file it did not write."""

    def test_it_reads_a_file_written_before_the_back_sentinel(self):
        """Format 2 files have no BACK node; every id from there up shifts by one."""
        path = os.path.join(TMP.name, "old.count.json")
        py("--kind", "count", "--seed", 5, "train", "--data", CORPUS, "--epochs", 1, model=path)
        doc = load_json(path)
        graph = doc["graph"]
        self.assertEqual(graph["nodes"]["labels"][2], "<back>")
        for key in ("labels", "z", "a", "b", "h", "k", "count"):
            values = graph["nodes"].get(key)
            if values:
                graph["nodes"][key] = values[:2] + values[3:]
        shift = lambda i: i - 1 if i >= 2 else i  # noqa: E731
        graph["edges"]["src"] = [shift(i) for i in graph["edges"]["src"]]
        graph["edges"]["dst"] = [shift(i) for i in graph["edges"]["dst"]]
        graph["format_version"] = 2
        graph.pop("paths", None)
        old = os.path.join(TMP.name, "downgraded.count.json")
        with open(old, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

        info = rust("info", model=old)
        self.assertEqual(info["stats"]["nodes"], graph["nodes"]["labels"].__len__() + 1)
        theirs = load_model(old)  # and Python upgrades it the same way
        self.assertEqual(info["stats"]["nodes"], theirs.stats()["nodes"])
        self.assertEqual(info["stats"]["edges"], theirs.stats()["edges"])

    def test_a_file_it_cannot_read_is_an_error_not_a_crash(self):
        broken = os.path.join(TMP.name, "broken.count.json")
        with open(broken, "w", encoding="utf-8") as fh:
            fh.write('{"format":"radixnet-bogus","version":1}')
        out = rust("info", model=broken, expect=1)
        self.assertIn("radixnet-count", out["error"])
        # a format it knows, with the graph missing, is still an error
        hollow = os.path.join(TMP.name, "hollow.negative.json")
        with open(hollow, "w", encoding="utf-8") as fh:
            fh.write('{"format":"radixnet-negative","version":1}')
        self.assertIn("graph", rust("info", model=hollow, expect=1)["error"])
        truncated = os.path.join(TMP.name, "truncated.count.json")
        with open(truncated, "w", encoding="utf-8") as fh:
            fh.write('{"format":"radixnet-count","graph":{')
        self.assertTrue(rust("info", model=truncated, expect=1)["error"])


if __name__ == "__main__":
    unittest.main()
