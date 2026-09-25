"""The Rust port's conversation (``rust/src/dialogue.rs``) against ``radixnet/dialogue.py``.

The same model, the same opening, the same options: both implementations must
say the same lines, skip and veto the same candidates, catch themselves
repeating at the same words, back out of it the same way - and, with
``--save``, teach the graph the same lesson about where its walks go round.
"""

import os
import shutil
import unittest

import json
import subprocess
import sys

from tests.rust_harness import CORPUS, ROOT, build, load_json, py, rust, serve, tmpdir

KEYS = ("text", "speaker", "context", "reply", "fresh", "given", "repeat", "stutter", "candidates", "skipped",
        "vetoed", "rethink", "labels", "node_ids", "reached_end")


def _lines(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def rust_lines(*args, model):
    """``--json`` output as JSON Lines (a streamed command): one document per line."""
    return _lines([build(), "--json", "--model", model, *[str(a) for a in args]])


def py_lines(*args, model):
    return _lines([sys.executable, "-m", "radixnet", "--json", "--model", model, *[str(a) for a in args]])


def same_stream(test, a, b):
    """Two streamed conversations say the same things in the same order: event for event, the texts equal and
    the costs within tolerance."""
    test.assertEqual([e["event"] for e in a], [e["event"] for e in b])
    for x, y in zip(a, b):
        if x["event"] == "done":
            continue  # the documents differ in shape between the CLI and the route; the callers compare them
        test.assertEqual((x["index"], x["speaker"]), (y["index"], y["speaker"]), (x, y))
        if x["event"] == "turn":
            for key in KEYS:
                test.assertEqual(x["turn"][key], y["turn"][key], (key, x["turn"]["text"]))
            test.assertLessEqual(abs(x["turn"]["cost"] - y["turn"]["cost"]), 1e-9)
        elif x["event"] in ("draft", "found"):
            test.assertEqual({k: v for k, v in x.items() if k != "cost"}, {k: v for k, v in y.items() if k != "cost"})
            test.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9)
        else:
            test.assertEqual(x, y)


def setUpModule():
    build()


class TestRustConverseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(tmpdir(), "dialogue")
        os.makedirs(cls.root, exist_ok=True)
        cls.model = os.path.join(cls.root, "m.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)

    def copy(self, name, negative=False):
        model = os.path.join(self.root, f"{name}.count.json")
        shutil.copy(self.model, model)
        if negative:
            shutil.copy(self.negative, os.path.join(self.root, f"{name}.count.negative.json"))
        return model

    def assert_same_turns(self, a, b):
        self.assertEqual(len(a["turns"]), len(b["turns"]), (a["transcript"], b["transcript"]))
        for x, y in zip(a["turns"], b["turns"]):
            for key in KEYS:
                self.assertEqual(x[key], y[key], (key, x["text"], y["text"]))
            self.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9)
        self.assertEqual(a["transcript"], b["transcript"])
        self.assertEqual(a["repeats"], b["repeats"])
        self.assertEqual(a["taught"], b["taught"])

    def test_the_same_conversation(self):
        for extra in ([], ["--opening", "the cat sat"], ["--turns", 8, "--context", 20, "--k", 3],
                      ["--allow-repeats"], ["--allow-word-repeats", "--explore", 0],
                      ["--speakers", "Ann,Bob,Cy", "--no-learn"]):
            args = ["converse", "--turns", 6, *extra]
            a = py(*args, model=self.copy("cp"))
            b = rust(*args, model=self.copy("cr"))
            self.assert_same_turns(a, b)
            self.assertEqual(a["speakers"], b["speakers"])
            self.assertIsNone(b["guard"])

    def test_the_same_conversation_streamed(self):
        # the stream is a view of the same conversation: both sides look at the same contexts, catch themselves
        # on the same drafts, back up to the same cuts, find (or fail to find) the same ways on, and speak the
        # same turns - and the last line is the document the plain command answers with
        for extra in ([], ["--explore", 0], ["--allow-word-repeats"], ["--opening", "the cat sat", "--k", 3]):
            with self.subTest(extra=extra):
                a = py_lines("converse", "--turns", 12, "--stream", *extra, model=self.copy("lp"))
                b = rust_lines("converse", "--turns", 12, "--stream", *extra, model=self.copy("lr"))
                same_stream(self, a, b)
                self.assertEqual((a[-1]["event"], b[-1]["event"]), ("done", "done"))
                self.assertEqual(a[-1]["transcript"], b[-1]["transcript"])
                self.assertEqual(a[-1]["taught"], b[-1]["taught"])
                kinds = {e["event"] for e in a}
                self.assertIn("turn", kinds)
                if not extra:
                    self.assertTrue({"draft", "caught", "backtrack"} <= kinds, kinds)
                plain = rust("converse", "--turns", 12, *extra, model=self.copy("lq"))
                self.assertEqual({k: v for k, v in b[-1].items() if k != "event"}, plain)

    def test_sampled_conversations_agree_under_a_seed(self):
        args = ["--seed", 7, "converse", "--mode", "sample", "--turns", 5, "--temperature", 0.8]
        self.assert_same_turns(py(*args, model=self.copy("sp")), rust(*args, model=self.copy("sr")))

    def test_what_a_rethink_teaches_the_graph_is_the_same(self):
        args = ["converse", "--turns", 10, "--k", 2, "--save"]
        pm, rm = self.copy("tp"), self.copy("tr")
        a = py(*args, model=pm)
        b = rust(*args, model=rm)
        self.assert_same_turns(a, b)
        if not a["taught"]:
            self.skipTest("this corpus never made a voice back up")
        pg, rg = load_json(pm)["graph"], load_json(rm)["graph"]
        self.assertEqual(pg["nodes"]["labels"], rg["nodes"]["labels"])
        self.assertEqual(pg["edges"]["count"], rg["edges"]["count"])
        self.assertEqual(pg["edges"]["reward"], rg["edges"]["reward"])

    def test_a_guarded_conversation_vetoes_the_same_replies(self):
        self.negative = os.path.join(self.root, "m.count.negative.json")
        if not os.path.exists(self.negative):
            args = ["negative", "blame", "--reason", "nonsense", "--severity", 2]
            for turn in py("converse", "--turns", 4, "--no-learn", model=self.model)["turns"]:
                args += ["--text", turn["text"]]
            py(*args, model=self.model)
        args = ["converse", "--turns", 5, "--no-learn"]
        a = py(*args, model=self.copy("gp", negative=True))
        b = rust(*args, model=self.copy("gr", negative=True))
        self.assert_same_turns(a, b)
        self.assertIsNotNone(b["guard"])
        for key in ("vetoed", "refusals"):
            self.assertEqual(a["guard"][key], b["guard"][key], key)
        self.assertEqual([v["text"] for v in a["guard"]["verdicts"]], [v["text"] for v in b["guard"]["verdicts"]])


class TestRustConverseRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(tmpdir(), "dialogue-route")
        os.makedirs(root, exist_ok=True)
        cls.model = os.path.join(root, "model.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)
        cls.expected = py("converse", "--turns", 4, "--opening", "the cat", "--no-learn", model=cls.model)
        cls.server = serve(cls, cls.model)

    def test_the_route_answers_the_converse_tab(self):
        status, data, _ = self.server.post("/api/converse", {"opening": "the cat", "turns": 4, "learn": False})
        self.assertEqual(status, 200, data)
        self.assertEqual(set(data), {"kind", "partner", "speakers", "turns", "count", "repeats", "guard"})
        self.assertEqual([t["text"] for t in data["turns"]], [t["text"] for t in self.expected["turns"]])
        self.assertEqual(data["count"], len(data["turns"]))
        self.assertIsNone(data["guard"])
        status, data, _ = self.server.post("/api/converse", {"k": 0})
        self.assertEqual(status, 400, data)
        status, data, _ = self.server.post("/api/converse", {"partner": "radix"})
        self.assertEqual(status, 400, data)

    def test_the_stream_route_says_what_the_python_stream_says(self):
        # the route streams the same events the Python CLI streams for the same model, and ends with the
        # document the plain route answers with
        status, raw, headers = self.server.post("/api/converse/stream", {"opening": "the cat", "turns": 4, "learn": False})
        self.assertEqual(status, 200, raw)
        self.assertTrue(headers.get("Content-Type", "").startswith("application/x-ndjson"), headers)
        events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        expected = py_lines("converse", "--turns", 4, "--opening", "the cat", "--no-learn", "--stream", model=self.model)
        same_stream(self, events, expected)
        self.assertEqual(events[-1]["event"], "done")
        status, plain, _ = self.server.post("/api/converse", {"opening": "the cat", "turns": 4, "learn": False})
        self.assertEqual(status, 200, plain)
        self.assertEqual({k: v for k, v in events[-1].items() if k != "event"}, plain)
        self.assertEqual([e["turn"] for e in events if e["event"] == "turn"], plain["turns"])
        # refused before anything was streamed: an ordinary 400
        status, data, headers = self.server.post("/api/converse/stream", {"k": 0})
        self.assertEqual(status, 400, data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
