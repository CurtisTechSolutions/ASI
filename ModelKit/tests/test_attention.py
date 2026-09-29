"""The attention band through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_attention.py).
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.attention import DEFAULT_BLUR, AttentionBand
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.model import load_model  # noqa: E402


TEXTS = [
    "the cat sat on the mat",
    "the cat ran to the door",
    "the dog sat on the log",
    "the dog ate the bone",
    "the bat sat on the mat",
]


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-attention-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )

    def select(self, kind):
        status, data, _ = self.client.post("/api/model/select", {"kind": kind})
        self.assertEqual(status, 200, data)

    def test_the_band_through_the_api(self):
        self.select("radix")
        status, data, _ = self.client.get("/api/model/attention")
        self.assertEqual((status, data["kind"], data["attention"]["applies"]), (200, "radix", False))
        status, data, _ = self.client.post("/api/model/attention", {"blur": 0.5})
        self.assertEqual(status, 400)
        self.assertIn("never corrected", data["error"])
        self.select("count")
        status, data, _ = self.client.get("/api/model/attention")
        self.assertEqual(data["attention"], {
            "on": False, "blur": None, "weights": None, "ngram": 3, "stride": 1, "unit": "char", "units": "chars",
            "applies": True, "default_blur": DEFAULT_BLUR,
        })
        status, data, _ = self.client.post("/api/model/attention", {"on": True})
        self.assertEqual((status, data["attention"]["blur"], data["stats"]["attention_blur"]), (200, 0.5, 0.5))
        status, data, _ = self.client.post("/api/model/attention", {"blur": 0.2})
        self.assertEqual(data["attention"]["weights"], [0.8, 1.0, 0.8])
        self.assertEqual(self.client.get("/api/status")[1]["attention_blur"], 0.2)
        self.assertEqual(self.client.get("/api/model")[1]["attention"]["blur"], 0.2)
        status, data, _ = self.client.post("/api/model/attention", {"blur": 3})
        self.assertEqual(status, 400)
        self.assertIn("[0, 1]", data["error"])
        status, data, _ = self.client.post("/api/model/attention", {"on": "yes"})
        self.assertEqual(status, 400)
        status, data, _ = self.client.post(
            "/api/model/attention/preview", {"wrong": "the cat sat", "right": "the bat sat", "blur": 1.0}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["blur"], data["weights"]), (1.0, [0.0, 1.0, 0.0]))
        self.assertEqual(data["wrong"]["charges"][2:5], [0.0, 1.0, 0.0])
        self.assertEqual(data["changes"], [{"op": "replace", "wrong": "c", "right": "b"}])
        self.assertEqual(self.client.get("/api/model/attention")[1]["attention"]["blur"], 0.2)  # a preview changes nothing
        status, data, _ = self.client.post("/api/model/attention", {"on": False, "blur": 0.9})
        self.assertEqual((status, data["attention"]["on"], data["attention"]["blur"]), (200, False, None))


class TestCli(unittest.TestCase):
    def test_show_switch_and_preview(self):
        from modelkit.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.count.json")
            model = CountRewardNet(seed=1)
            model.train(TEXTS, epochs=1)
            model.save(path)

            def run(*argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--model", path, "--json", *argv])
                return code, json.loads(out.getvalue()) if out.getvalue().strip() else None

            code, doc = run("attention")
            self.assertEqual((code, doc["attention"]["on"], doc["saved"]), (0, False, None))
            code, doc = run("attention", "--blur", "0.25", "--dry-run")
            self.assertEqual((code, doc["attention"]["blur"], doc["saved"]), (0, 0.25, None))
            self.assertIsNone(load_model(path).graph.attention.blur)  # a dry run saves nothing
            code, doc = run("attention", "--on", "--wrong", "the cat sat", "--right", "the bat sat")
            self.assertEqual((code, doc["attention"]["blur"]), (0, DEFAULT_BLUR))
            self.assertEqual(doc["preview"]["wrong"]["charges"][2:5], [0.25, 0.5, 0.25])
            self.assertEqual(load_model(path).graph.attention, AttentionBand(DEFAULT_BLUR))
            code, doc = run("attention", "--off", "--blur", "0.5")
            self.assertNotEqual(code, 0)
            code, doc = run("attention", "--off")
            self.assertEqual((code, doc["attention"]["on"]), (0, False))
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("attention", json.load(fh)["graph"])


if __name__ == "__main__":
    unittest.main()
