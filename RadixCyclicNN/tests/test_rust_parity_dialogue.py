"""The Rust port's conversation (``rust/src/dialogue.rs``) against ``radixnet/dialogue.py``.

The same model, the same opening, the same options: both implementations must
say the same lines, skip and veto the same candidates, catch themselves
repeating at the same words, back out of it the same way - and, with
``--save``, teach the graph the same lesson about where its walks go round.
"""

import os
import shutil
import unittest

from tests.rust_harness import CORPUS, build, load_json, py, rust, serve, tmpdir

KEYS = ("text", "speaker", "context", "reply", "fresh", "given", "repeat", "stutter", "candidates", "skipped",
        "vetoed", "rethink", "labels", "node_ids", "reached_end")


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


if __name__ == "__main__":
    unittest.main()
