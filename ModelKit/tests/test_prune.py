"""Auto prune through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_prune.py).
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

from radixnet.encoding import Encoder  # noqa: E402
from radixnet.model import RadixNet, load_model  # noqa: E402
from radixnet.prune import AutoPrune  # noqa: E402


TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown cat sat on the mat",
    "a bird sang in the tree all afternoon",
]
UNWALKED = "the lazy cat sang"


def trained(path):
    """A sine model taught the texts, with one more text registered but never walked, saved at ``path``."""
    model = RadixNet(seed=1)
    model.train(TEXTS, epochs=1)
    model.graph.observe_sequence(Encoder().encode(UNWALKED), count=False)
    model.save(path)
    return model


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-prune-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )

    def test_auto_prune_through_the_api(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post("/api/model/select", {"kind": "count"})
        self.assertEqual(status, 200, data)
        status, data, _ = self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.assertEqual(status, 202, data)
        wait_for_job(self.client)
        status, data, _ = self.client.get("/api/model/prune")
        self.assertEqual((status, data["kind"], data["prune"]["on"], data["prune"]["candidates"], data["prune"]["rule"]),
                         (200, "count", False, None, None))
        self.assertEqual((data["prune"]["default_min_count"], data["prune"]["default_min_share"], data["prune"]["default_every"]), (1, 0.0, 1))
        self.assertGreater(data["prune"]["edges"], 0)
        # the stats and the model description are every port's, and only Python prunes: the setting has its own route
        self.assertNotIn("auto_prune", self.client.get("/api/status")[1])
        self.assertNotIn("auto_prune", self.client.get("/api/model")[1])
        status, data, _ = self.client.post("/api/model/prune", {"on": True})
        self.assertEqual((status, data["prune"]["on"], data["prune"]["min_count"], data["prune"]["every"], data["prune"]["auto"]),
                         (200, True, 1, 1, True))
        self.assertNotIn("auto_prune", data["stats"])
        self.assertEqual(data["prune"]["candidates"], 0)  # every edge was walked
        self.assertEqual(self.client.get("/api/model/prune")[1]["prune"]["on"], True)
        status, data, _ = self.client.post("/api/model/prune", {"min_share": 0.3, "every": 2, "auto": False})
        self.assertEqual((status, data["prune"]["min_count"], data["prune"]["min_share"], data["prune"]["every"], data["prune"]["auto"]),
                         (200, 1, 0.3, 2, False))
        self.assertGreater(data["prune"]["candidates"], 0)  # the rare continuations of the busy nodes
        self.assertIn("under 0.3", data["prune"]["rule"])
        edges = data["prune"]["edges"]
        status, data, _ = self.client.post("/api/model/prune/now", {})
        self.assertEqual(status, 200, data)
        self.assertGreater(data["pruned"]["edges"], 0)
        self.assertEqual(data["pruned"]["edges_before"], edges)
        self.assertEqual(data["pruned"]["edges_after"], data["stats"]["edges"])
        self.assertEqual(data["pruned"]["nodes_after"], data["stats"]["nodes"])
        self.assertEqual(data["prune"]["candidates"], 0)
        status, data, _ = self.client.post("/api/model/prune/now", {"min_count": 1000})  # this prune alone
        self.assertEqual((status, data["prune"]["min_count"]), (200, 1))
        self.assertGreater(data["pruned"]["edges"], 0)
        for body in ({"min_count": -1}, {"min_share": 1.0}, {"every": 0}, {"on": "yes"}, {"min_share": "a lot"}):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/model/prune", body)
                self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/model/prune/now", {"min_share": 2})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/model/prune", {"on": False})
        self.assertEqual((status, data["prune"]["on"]), (200, False))
        self.assertFalse(self.client.get("/api/model/prune")[1]["prune"]["on"])


class TestCli(unittest.TestCase):
    def test_show_set_prune_and_off(self):
        from modelkit.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            trained(path)

            def run(*argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--model", path, "--json", *argv])
                return code, json.loads(out.getvalue()) if out.getvalue().strip() else None

            code, doc = run("prune")
            self.assertEqual((code, doc["prune"]["on"], doc["prune"]["candidates"], doc["saved"], doc["pruned"]), (0, False, None, None, None))
            code, doc = run("prune", "--on", "--dry-run")
            self.assertEqual((code, doc["prune"]["on"], doc["saved"]), (0, True, None))
            self.assertGreater(doc["prune"]["candidates"], 0)  # the text registered but never walked
            self.assertFalse(load_model(path).graph.auto_prune.on)  # a dry run saves nothing
            code, doc = run("prune", "--min-count", "2", "--every", "3", "--manual")
            self.assertEqual((code, doc["prune"]["min_count"], doc["prune"]["every"], doc["prune"]["auto"], doc["changed"]),
                             (0, 2, 3, False, {"min_count": 2, "min_share": 0.0, "every": 3, "auto": False}))
            self.assertEqual(load_model(path).graph.auto_prune, AutoPrune(True, 2, 0.0, 3, False))
            code, doc = run("prune", "--now", "--out", os.path.join(tmp, "pruned.json"))
            self.assertEqual((code, doc["saved"]["path"]), (0, os.path.join(tmp, "pruned.json")))
            self.assertGreater(doc["pruned"]["edges"], 0)
            self.assertEqual(doc["pruned"]["edges_after"], doc["prune"]["edges"])
            self.assertEqual(load_model(path).stats()["edges"], doc["pruned"]["edges_before"])  # --out left the model alone
            code, doc = run("prune", "--now")
            self.assertEqual((code, doc["pruned"]["edges_before"], doc["saved"]["path"]), (0, doc["pruned"]["edges_before"], path))
            self.assertEqual(load_model(path).stats()["edges"], doc["pruned"]["edges_after"])
            code, doc = run("prune")
            self.assertEqual((doc["prune"]["min_count"], doc["prune"]["every"], doc["prune"]["auto"]), (2, 3, False))
            for argv in (("prune", "--off", "--min-count", "2"), ("prune", "--auto", "--manual"), ("prune", "--min-count", "-1"),
                         ("prune", "--min-share", "1"), ("prune", "--every", "0")):
                with self.subTest(argv=argv):
                    code, doc = run(*argv)
                    self.assertNotEqual(code, 0)
            code, doc = run("prune", "--off")
            self.assertEqual((code, doc["prune"]["on"], doc["changed"]), (0, False, {"on": False}))
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("auto_prune", json.load(fh)["graph"])
            code, doc = run("prune", "--now")  # off: a prune by hand still works, at the defaults
            self.assertEqual((code, doc["pruned"]["edges"], doc["prune"]["on"]), (0, 0, False))


if __name__ == "__main__":
    unittest.main()
