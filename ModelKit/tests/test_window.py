"""The dynamic window through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_window.py).
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

from radixnet.model import RadixNet, load_model  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.window import DynamicWindow  # noqa: E402


LONG = "the quick brown fox jumps over the lazy dog while the cat sat on the mat and the bird sang in the tree all afternoon"


TEXTS = [
    LONG,
    "the quick brown fox jumps over the lazy dog while the cat ran to the door",
    "a bird sang in the tree all afternoon",
]


def trained(cls, encoding=None, **options):
    """A model of ``cls`` taught the texts (blamed, for the negative network): long unary chains, merged."""
    model = cls(seed=1, encoding=encoding)
    if cls is NegativeNet:
        model.blame(TEXTS, reason="other")
    else:
        model.train(TEXTS, epochs=1, **options)
    return model


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-window-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )

    def test_the_window_through_the_api(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post("/api/model/select", {"kind": "count"})
        self.assertEqual(status, 200, data)
        status, data, _ = self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.assertEqual(status, 202, data)
        wait_for_job(self.client)
        status, data, _ = self.client.get("/api/model/window")
        self.assertEqual((status, data["kind"], data["window"]["on"], data["window"]["longer"]), (200, "count", False, None))
        self.assertGreater(data["window"]["longest"], 16)
        status, data, _ = self.client.post("/api/model/window/step", {})
        self.assertEqual(status, 400)
        self.assertIn("off", data["error"])
        status, data, _ = self.client.post("/api/model/window", {"on": True})
        self.assertEqual((status, data["window"]["sizes"], data["window"]["size"], data["stats"]["dynamic_window"]), (200, [32, 16, 8, 4], 32, 32))
        self.assertEqual(self.client.get("/api/status")[1]["dynamic_window"], 32)
        self.assertEqual(self.client.get("/api/model")[1]["dynamic_window"]["size"], 32)
        status, data, _ = self.client.post("/api/model/window", {"size": 16})
        self.assertEqual((status, data["window"]["size"], data["window"]["next"]), (200, 16, 8))
        self.assertGreater(data["window"]["longer"], 0)
        status, data, _ = self.client.post("/api/model/window/step", {"steps": 2})
        self.assertEqual((status, data["step"]["sizes"], data["window"]["size"], data["stats"]["dynamic_window"]), (200, [16, 8], 4, 4))
        self.assertEqual(data["step"]["nodes_after"], data["stats"]["nodes"])
        self.assertGreater(data["step"]["splits"], 0)
        status, data, _ = self.client.post("/api/model/window", {"top": 16, "floor": 8, "auto": False})
        self.assertEqual((status, data["window"]["sizes"], data["window"]["size"], data["window"]["auto"]), (200, [16, 8], 8, False))
        for body in ({"top": 12}, {"size": 3}, {"floor": 64}, {"on": "yes"}, {"steps": 0}):
            with self.subTest(body=body):
                path = "/api/model/window/step" if "steps" in body else "/api/model/window"
                status, data, _ = self.client.post(path, body)
                self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/model/window", {"on": False})
        self.assertEqual((status, data["window"]["on"], data["stats"]["dynamic_window"]), (200, False, None))
        self.assertIsNone(self.client.get("/api/status")[1]["dynamic_window"])


class TestCli(unittest.TestCase):
    def test_show_set_step_and_off(self):
        from modelkit.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            trained(RadixNet).save(path)

            def run(*argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--model", path, "--json", *argv])
                return code, json.loads(out.getvalue()) if out.getvalue().strip() else None

            code, doc = run("window")
            self.assertEqual((code, doc["window"]["on"], doc["saved"], doc["step"]), (0, False, None, None))
            code, doc = run("window", "--on", "--dry-run")
            self.assertEqual((code, doc["window"]["size"], doc["saved"]), (0, 32, None))
            self.assertFalse(load_model(path).graph.dynamic_window.on)  # a dry run saves nothing
            code, doc = run("window", "--top", "16", "--floor", "8", "--manual")
            self.assertEqual((code, doc["window"]["sizes"], doc["window"]["auto"], doc["changed"]), (0, [16, 8], False, {"top": 16, "floor": 8, "size": 16, "auto": False}))
            self.assertEqual(load_model(path).graph.dynamic_window, DynamicWindow(16, 8, 16, False))
            code, doc = run("window", "--step")
            self.assertEqual((code, doc["step"]["sizes"], doc["window"]["size"]), (0, [16], 8))
            self.assertGreater(doc["step"]["splits"], 0)
            self.assertEqual(load_model(path).graph.dynamic_window.size, 8)
            code, doc = run("window", "--step", "2", "--out", os.path.join(tmp, "stepped.json"))
            self.assertEqual((code, doc["step"]["sizes"], doc["saved"]["path"]), (0, [8, 16], os.path.join(tmp, "stepped.json")))
            self.assertEqual(load_model(path).graph.dynamic_window.size, 8)  # --out left the model alone
            code, doc = run("info")
            self.assertEqual(doc["stats"]["dynamic_window"], 8)
            for argv in (("window", "--off", "--top", "16"), ("window", "--off", "--step"), ("window", "--auto", "--manual"),
                         ("window", "--top", "20"), ("window", "--size", "4")):
                with self.subTest(argv=argv):
                    code, doc = run(*argv)
                    self.assertNotEqual(code, 0)
            code, doc = run("window", "--off")
            self.assertEqual((code, doc["window"]["on"], doc["changed"]), (0, False, {"on": False}))
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("dynamic_window", json.load(fh)["graph"])
            code, doc = run("window", "--step")
            self.assertNotEqual(code, 0)  # off: nothing to step


if __name__ == "__main__":
    unittest.main()
