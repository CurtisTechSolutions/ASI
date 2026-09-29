"""The negative network through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_negative.py).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.model import load_model  # noqa: E402
from radixnet.negative import NegativeNet

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the model's checkout: its sample data, and the directory `python -m radixnet` runs in
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")


FAILURES = [
    "the the the the cat",
    "the cat cat cat sat",
]


GOOD = [
    "the cat sat on the mat",
    "the dog sat on the log",
]


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-negative-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model_path = os.path.join(cls.tmp.name, "model.json")
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, model_path=cls.model_path)

    def setUp(self):
        status, data, _ = self.client.post("/api/negative/reset")
        self.assertEqual(status, 200, data)

    def blame(self, **body):
        body.setdefault("texts", FAILURES)
        body.setdefault("reason", "repetition")
        status, data, _ = self.client.post("/api/negative/blame", body)
        self.assertEqual(status, 200, data)
        return data

    def test_status_reasons_and_journal(self):
        self.blame(note="it repeats", source="review")
        status, data, _ = self.client.get("/api/negative")
        self.assertEqual(status, 200)
        self.assertEqual(data["path"], os.path.join(self.tmp.name, "model.negative.json"))
        self.assertEqual([r["reason"] for r in data["reasons"]], ["repetition"])
        self.assertEqual(data["journal"][0]["note"], "it repeats")
        self.assertEqual(data["stats"]["kind"], "negative")
        self.assertIn("threshold", data["settings"])

    def test_blame_judge_and_clear(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/judge", {"text": FAILURES[0]})
        self.assertEqual(status, 200, data)
        verdict = data["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertEqual(verdict["reasons"][0]["reason"], "repetition")
        status, data, _ = self.client.post("/api/negative/clear", {"texts": [GOOD[0]]})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["matched"] + data["unmatched"], 1)
        status, data, _ = self.client.post("/api/negative/judge", {"texts": []})
        self.assertEqual(status, 400, data)

    def test_filter_vetoes_a_known_failure(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/filter", {"texts": [FAILURES[0], "an unseen sentence"]})
        self.assertEqual(status, 200, data)
        self.assertEqual([v["decision"] for v in data["verdicts"]], ["reject", "pass"])
        self.assertEqual(data["kept"], ["an unseen sentence"])
        self.assertEqual(data["rejected"][0]["rule"], "blame")
        self.assertEqual(data["pair"]["negative"]["kind"], "negative")

    def test_filter_generates_through_the_pair(self):
        self.client.post("/api/train", {"texts": GOOD, "epochs": 2, "lr": 0.5, "batch_size": 4})
        from tests.test_api import wait_for_job

        wait_for_job(self.client)
        self.blame()
        status, data, _ = self.client.post("/api/negative/filter", {"count": 2, "max_length": 30, "over_sample": 2})
        self.assertEqual(status, 200, data)
        self.assertLessEqual(len(data["texts"]), 2)
        self.assertEqual(len(data["verdicts"]), data["candidates"])
        self.assertTrue(all("why" in v for v in data["verdicts"]))

    def test_filter_rules_can_be_tuned(self):
        self.blame()
        body = {"texts": [FAILURES[0]], "threshold": 1e9, "min_coverage": 0.0}
        status, data, _ = self.client.post("/api/negative/filter", body)
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["rule"], "ratio")  # blame cannot reject, the ratio still can
        status, data, _ = self.client.post("/api/negative/filter", {**body, "no_ratio": True})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["decision"], "suspect")  # both rules off: nothing rejects
        self.assertIsNone(data["verdicts"][0]["ratio_threshold"])
        status, data, _ = self.client.post("/api/negative/filter", {**body, "no_ratio": True, "strict": True})
        self.assertEqual(data["verdicts"][0]["rule"], "suspect")

    def test_settings_forget_and_save(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/settings", {"threshold": 2.5, "min_coverage": 0.1})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["settings"], {"threshold": 2.5, "min_coverage": 0.1, "provenance": True})
        status, data, _ = self.client.post("/api/negative/settings", {"clear_scale": 2.0})
        self.assertEqual(data["weights"]["clear_scale"], 2.0)
        status, data, _ = self.client.post("/api/negative/forget", {"reason": "repetition"})
        self.assertEqual(status, 200, data)
        self.assertGreater(data["blame_removed"], 0)
        status, data, _ = self.client.post("/api/negative/save")
        self.assertEqual(status, 200, data)
        self.assertTrue(os.path.isfile(data["path"]))
        self.assertIsInstance(load_model(data["path"]), NegativeNet)

    def test_save_writes_the_negative_model_too(self):
        self.blame()
        status, data, _ = self.client.post("/api/save")
        self.assertEqual(status, 200, data)
        self.assertIsNotNone(data["negative"])
        self.assertTrue(os.path.isfile(data["negative"]["path"]))

    def test_selecting_the_negative_kind_shares_the_model(self):
        self.blame()
        status, data, _ = self.client.post("/api/model/select", {"kind": "negative"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "negative")
        self.assertGreater(data["stats"]["failures_total"], 0)
        # the filter still works: the positive model comes from the parked kinds
        status, data, _ = self.client.post("/api/negative/filter", {"texts": [FAILURES[0]]})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["decision"], "reject")
        self.assertEqual(self.client.post("/api/model/select", {"kind": "radix"})[1]["kind"], "radix")

    def test_errors(self):
        status, data, _ = self.client.post("/api/negative/blame", {"texts": []})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/negative/forget", {"factor": 2.0})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/negative/filter", {"over_sample": 0})
        self.assertEqual(status, 400, data)


class TestCli(unittest.TestCase):
    """The ``negative`` command group end to end (subprocess, --json)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_cli import run_json

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-negative-cli-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model = os.path.join(cls.tmp.name, "model.json")
        cls.negative = os.path.join(cls.tmp.name, "model.negative.json")
        run_json("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2,
                 "--batch-size", 8, "--lr", 0.5, model=cls.model)

    def run_json(self, *args, **kwargs):
        from tests.test_cli import run_json

        kwargs.setdefault("model", self.model)
        return run_json(*args, **kwargs)

    def test_blame_why_reasons_filter_and_forget(self):
        doc = self.run_json("negative", "blame", "--text", FAILURES[0], "--reason", "repetition",
                            "--note", "it repeats", "--source", "review")
        self.assertEqual(doc["reason"], "repetition")
        self.assertTrue(os.path.isfile(self.negative))
        self.assertEqual(doc["reasons"][0]["reason"], "repetition")

        doc = self.run_json("negative", "why", "--text", FAILURES[0])
        verdict = doc["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertTrue(verdict["spans"])

        doc = self.run_json("negative", "reasons")
        self.assertEqual(doc["journal"][0]["note"], "it repeats")

        doc = self.run_json("negative", "filter", "--text", FAILURES[0], "--text", "a wholly unseen line")
        self.assertEqual([v["decision"] for v in doc["verdicts"]], ["reject", "pass"])
        self.assertEqual(doc["texts"], ["a wholly unseen line"])

        doc = self.run_json("negative", "filter", "--count", 2, "--max-length", 30)
        self.assertEqual(len(doc["verdicts"]), doc["candidates"])

        doc = self.run_json("negative", "clear", "--text", GOOD[0])
        self.assertEqual(doc["texts"], 1)

        doc = self.run_json("negative", "forget", "--reason", "repetition")
        self.assertGreater(doc["blame_removed"], 0)

    def test_correct_can_teach_both_networks(self):
        from tests.test_cli import run_json

        model = os.path.join(self.tmp.name, "pair.count.json")
        negative = os.path.join(self.tmp.name, "pair.count.negative.json")
        run_json("--kind", "count", "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"),
                 "--epochs", 1, model=model)
        doc = run_json("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sits on the mat",
                       "--blame", "--reason", "agreement", "--note", "the verb must agree", model=model)
        self.assertEqual(doc["penalised"], 1)  # the positive model moved one step
        self.assertEqual(doc["negative"]["blamed"]["blamed"], 1)  # and so did the negative one
        self.assertEqual(doc["negative"]["blamed"]["reason"], "agreement")
        self.assertTrue(os.path.isfile(negative))
        blamed = load_model(negative)
        self.assertEqual([span["fragment"] for span in blamed.judge("the cat sit on the mat")["spans"]], ["sit "])

    def test_kind_negative_trains_by_blaming(self):
        from tests.test_cli import run_json

        model = os.path.join(self.tmp.name, "kind.negative.json")
        doc = run_json("--kind", "negative", "train", "--data", os.path.join(ROOT, "data", "sample_garbage.txt"),
                       "--epochs", 1, model=model)
        self.assertEqual(doc["stats"]["kind"], "negative")
        self.assertGreater(doc["stats"]["failures_total"], 0)

    def test_missing_negative_model_is_an_error(self):
        from tests.test_cli import run_cli

        proc = run_cli("negative", "why", "--text", "anything at all", "--negative",
                       os.path.join(self.tmp.name, "absent.json"), model=self.model, expect=1)
        self.assertIn("negative model file not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
