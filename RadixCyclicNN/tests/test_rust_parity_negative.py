"""The Rust port's negative network at work: the filter pair, the guard, and the routes.

``tests/test_rust_parity.py`` holds the negative network's *file* and its
verdicts to Python's.  This suite holds what is built on them: the pair that
filters the positive model's output (``radixnet/duo.py``, ``rust/src/duo.rs``),
the guard every answer goes through, and the ``/api/negative`` routes the
frontend's Negative tab reads.
"""

import os
import shutil
import unittest

from tests.rust_harness import CORPUS, build, load_json, py, rust, serve, tmpdir

FAILURES = ["the the the the cat", "cat the sat mat on the the", "the sky sat on the cat"]


def setUpModule():
    build()


class TestRustFilterParity(unittest.TestCase):
    """One positive model, one negative network, both implementations: the same decisions."""

    @classmethod
    def setUpClass(cls):
        root = os.path.join(tmpdir(), "filter")
        os.makedirs(root, exist_ok=True)
        cls.model = os.path.join(root, "m.count.json")
        cls.negative = os.path.join(root, "m.count.negative.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)
        args = ["negative", "blame", "--reason", "nonsense", "--severity", 2]
        for text in FAILURES:
            args += ["--text", text]
        py(*args, model=cls.model)
        cls.root = root

    def copy(self, name):
        """A private copy of the pair, so a learning filter cannot change the other side's."""
        model = os.path.join(self.root, f"{name}.count.json")
        shutil.copy(self.model, model)
        shutil.copy(self.negative, os.path.join(self.root, f"{name}.count.negative.json"))
        return model

    def test_the_same_texts_get_the_same_decisions(self):
        texts = [*FAILURES, "the cat sat on the mat", "an unseen line of text"]
        args = ["negative", "filter"]
        for text in texts:
            args += ["--text", text]
        a = py(*args, model=self.copy("fp"))
        b = rust(*args, model=self.copy("fr"))
        self.assertEqual(a["kept"], b["kept"])
        self.assertEqual(a["texts"], b["texts"])
        self.assertEqual([v["decision"] for v in a["verdicts"]], [v["decision"] for v in b["verdicts"]])
        self.assertEqual([v["rule"] for v in a["verdicts"]], [v["rule"] for v in b["verdicts"]])
        self.assertEqual([v["why"] for v in a["verdicts"]], [v["why"] for v in b["verdicts"]])
        for x, y in zip(a["verdicts"], b["verdicts"]):
            for key in ("risk", "peak", "coverage", "blame", "ratio", "positive", "negative"):
                self.assertLessEqual(abs(x[key] - y[key]), 1e-9, (key, x["text"]))
            self.assertEqual(set(x), set(y))
        self.assertEqual(a["rate"], b["rate"])
        self.assertEqual(a["pair"]["config"], b["pair"]["config"])
        self.assertIn("reject", [v["decision"] for v in b["verdicts"]])

    def test_the_stricter_rules_agree(self):
        texts = ["the cat sat on the sky", "the the cat sat", "the cat sat on the mat"]
        for extra in (["--strict"], ["--peak", "0.5"], ["--no-ratio"], ["--threshold", "0.2", "--min-coverage", "0.1"]):
            args = ["negative", "filter", *extra]
            for text in texts:
                args += ["--text", text]
            a = py(*args, model=self.copy("sp"))
            b = rust(*args, model=self.copy("sr"))
            self.assertEqual([(v["decision"], v["rule"], v["why"]) for v in a["verdicts"]],
                             [(v["decision"], v["rule"], v["why"]) for v in b["verdicts"]], extra)

    def test_a_learning_filter_blames_what_it_rejects(self):
        args = ["negative", "filter", "--learn", "--text", FAILURES[0], "--text", "the the the mat"]
        pm, rm = self.copy("lp"), self.copy("lr")
        a = py(*args, model=pm)
        b = rust(*args, model=rm)
        self.assertEqual([v["decision"] for v in a["verdicts"]], [v["decision"] for v in b["verdicts"]])
        self.assertIn("saved", b)
        pn = load_json(os.path.join(self.root, "lp.count.negative.json"))
        rn = load_json(os.path.join(self.root, "lr.count.negative.json"))
        self.assertEqual(pn["graph"]["nodes"]["labels"], rn["graph"]["nodes"]["labels"])
        self.assertEqual(pn["graph"]["edges"]["blame"], rn["graph"]["edges"]["blame"])
        self.assertEqual(pn["log"][-1]["source"], rn["log"][-1]["source"])
        self.assertEqual(rn["log"][-1]["source"], "filter")

    def test_the_guard_reranks_a_prediction_the_same_way(self):
        args = ["predict", "--prefix", "the ", "--length", 12, "--k", 5]
        a = py(*args, model=self.model)
        b = rust(*args, model=self.model)
        self.assertIsNotNone(a["guard"])
        self.assertIsNotNone(b["guard"])
        self.assertEqual(a["continuation"], b["continuation"])
        self.assertEqual(a["guard"]["vetoed"], b["guard"]["vetoed"])
        self.assertEqual(a["guard"]["candidates"], b["guard"]["candidates"])
        self.assertEqual([v["text"] for v in a["guard"]["verdicts"]], [v["text"] for v in b["guard"]["verdicts"]])
        self.assertEqual(a["guard"]["config"], b["guard"]["config"])
        # --no-guard hands out what the positive model wrote
        self.assertIsNone(rust(*args, "--no-guard", model=self.model)["guard"])

    def test_the_guard_filters_generated_texts_the_same_way(self):
        args = ["generate", "--count", 3, "--mode", "beam", "--max-length", 30]
        a = py(*args, model=self.model)
        b = rust(*args, model=self.model)
        self.assertEqual([s["text"] for s in a["samples"]], [s["text"] for s in b["samples"]])
        for key in ("vetoed", "candidates", "asked", "kept", "rate"):
            self.assertEqual(a["guard"][key], b["guard"][key], key)

    def test_no_negative_network_no_guard(self):
        root = os.path.join(tmpdir(), "unguarded")
        os.makedirs(root, exist_ok=True)
        model = os.path.join(root, "u.count.json")
        shutil.copy(self.model, model)
        self.assertIsNone(rust("predict", "--prefix", "the ", model=model)["guard"])


class TestRustServerNegative(unittest.TestCase):
    """The Rust server's negative endpoints answer the JSON the frontend's Negative tab expects."""

    @classmethod
    def setUpClass(cls):
        root = os.path.join(tmpdir(), "server-negative")
        os.makedirs(root, exist_ok=True)
        cls.model = os.path.join(root, "model.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)
        cls.server = serve(cls, cls.model)

    def test_the_negative_endpoints(self):
        s = self.server
        status, data, _ = s.post("/api/negative/blame", {"texts": ["the the the the cat"],
                                                         "reason": "repetition", "note": "it repeats"})
        self.assertEqual(status, 200, data)
        self.assertEqual([r["reason"] for r in data["reasons"]], ["repetition"])
        self.assertEqual(data["records"][-1]["phase"], "negative")
        self.assertEqual(data["records"][-1]["matched"], 1)
        status, data, _ = s.get("/api/negative")
        self.assertEqual(status, 200, data)
        self.assertEqual(set(data), {"path", "active", "stats", "reasons", "journal", "weights", "settings"})
        self.assertEqual(data["stats"]["kind"], "negative")
        self.assertEqual(data["journal"][0]["note"], "it repeats")
        self.assertEqual(data["weights"]["function"], "blame")
        status, data, _ = s.post("/api/negative/judge", {"text": "the the the the cat"})
        self.assertEqual(status, 200, data)
        verdict = data["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertLessEqual({"risk", "peak", "coverage", "blame", "reasons", "spans", "why"}, set(verdict))
        status, data, _ = s.post("/api/negative/filter", {"texts": ["the the the the cat", "an unseen line"]})
        self.assertEqual(status, 200, data)
        self.assertEqual([v["decision"] for v in data["verdicts"]], ["reject", "pass"])
        self.assertIsNone(data["verdicts"][1]["rule"])
        self.assertEqual(data["kept"], ["an unseen line"])
        self.assertIn("pair", data)
        status, data, _ = s.post("/api/negative/clear", {"text": "the cat sat on the mat"})
        self.assertEqual(status, 200, data)
        self.assertIn("matched", data)
        status, data, _ = s.post("/api/negative/settings", {"threshold": 2.0})
        self.assertEqual(data["settings"]["threshold"], 2.0)
        status, data, _ = s.post("/api/negative/settings", {"threshold": 0.5})
        status, data, _ = s.post("/api/negative/save", {})
        self.assertEqual(status, 200, data)
        self.assertTrue(os.path.isfile(data["path"]))
        from radixnet.model import load_model

        self.assertEqual(load_model(data["path"]).kind, "negative")
        # the output paths are guarded now that the negative network knows a failure
        status, data, _ = s.post("/api/predict", {"prefix": "the ", "length": 10})
        self.assertEqual(status, 200, data)
        self.assertIsNotNone(data["guard"])
        self.assertTrue(data["guard"]["on"])
        status, data, _ = s.post("/api/predict", {"prefix": "the ", "length": 10, "guard": False})
        self.assertIsNone(data["guard"])
        status, data, _ = s.post("/api/generate", {"count": 2, "mode": "beam"})
        self.assertEqual(status, 200, data)
        self.assertIsNotNone(data["guard"])
        self.assertEqual(data["guard"]["asked"], 6)
        # forgetting and resetting
        status, data, _ = s.post("/api/negative/forget", {"reason": "repetition"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["reason"], "repetition")
        self.assertGreater(data["edges"], 0)
        status, data, _ = s.post("/api/negative/reset", {})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["reasons"], [])
        status, data, _ = s.get("/api/status")
        self.assertIn("POST /api/negative/filter", data["routes"])

    def test_bad_requests_say_why(self):
        s = self.server
        status, data, _ = s.post("/api/negative/blame", {})
        self.assertEqual(status, 400)
        self.assertIn("texts", data["error"])
        status, data, _ = s.post("/api/negative/filter", {"texts": ["x"], "min_coverage": 2})
        self.assertEqual(status, 400)
        self.assertIn("min_coverage", data["error"])


if __name__ == "__main__":
    unittest.main()
