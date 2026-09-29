"""The resonant (phase) model through the kit: checkpoints, evolve, conversation, the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_resonance.py).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from modelkit.dialogue import converse  # noqa: E402

from radixnet.model import load_model  # noqa: E402
from radixnet.resonance import ResonantNet


TEXTS = [
    "the sun rises in the east",
    "the sun sets in the west",
    "the moon rises at night",
    "the earth orbits the sun",
]


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model_path = os.path.join(cls.tmp.name, "model.json")
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, model_path=cls.model_path)

    def wait(self):
        from tests.test_api import wait_for_job

        return wait_for_job(self.client)

    def setUp(self):
        status, data, _ = self.client.post("/api/model/select", {"kind": "resonant"})
        self.assertEqual(status, 200, data)

    def test_select_train_and_predict(self):
        self.assertEqual(self.client.get("/api/status")[1]["kind"], "resonant")
        self.assertEqual(self.client.post("/api/train", {"texts": TEXTS, "epochs": 2})[0], 202)
        self.wait()
        status, data, _ = self.client.post("/api/predict", {"prefix": "the ", "length": 10, "mode": "beam", "k": 2})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "resonant")
        self.assertTrue(data["full_text"].startswith("the "))
        self.assertLessEqual(len(data["top"]), 2)

    def test_status_reports_the_phase_and_the_layer(self):
        self.client.post("/api/train", {"texts": TEXTS + ["lol lol lol lol"], "epochs": 2})
        self.wait()
        status = self.client.get("/api/status")[1]
        self.assertEqual(status["kind"], "resonant")
        self.assertIn("coherence_mean", status)
        self.assertIn("signatures", status["meta"])

    def test_graph_view_reports_coherence_and_the_advance(self):
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.wait()
        graph = self.client.get("/api/graph?limit=5")[1]
        self.assertIn("buckets", graph)
        self.assertNotIn("window", graph)
        self.assertTrue(all("advance" in n for n in graph["nodes"]))
        self.assertTrue(all({"coherence", "mu", "reward"} <= set(e) for e in graph["edges"]))

    def test_weight_endpoint(self):
        status, data, _ = self.client.post("/api/model/weights", {"kick_scale": 1.0, "resonance_scale": 2.0})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["weights"]["kick_scale"], data["weights"]["resonance_scale"]), (1.0, 2.0))
        status, data, _ = self.client.post("/api/model/weights", {"window": 5})  # a count-model option
        self.assertEqual(status, 400)
        self.assertIn("window", data["error"])

    def test_reset_with_phase_options(self):
        status, data, _ = self.client.post("/api/reset", {"kind": "resonant", "buckets": 16, "kick_scale": 0.5})
        self.assertEqual(status, 200, data)
        self.assertEqual(self.client.get("/api/model")[1]["weights"]["buckets"], 16)

    def test_switching_kinds_keeps_the_model_in_memory(self):
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.wait()
        nodes = self.client.get("/api/status")[1]["nodes"]
        self.assertEqual(self.client.post("/api/model/select", {"kind": "radix"})[1]["kind"], "radix")
        status, data, _ = self.client.post("/api/model/select", {"kind": "resonant"})
        self.assertEqual((status, data["origin"]), (200, "memory"))
        self.assertEqual(self.client.get("/api/status")[1]["nodes"], nodes)


class TestCli(unittest.TestCase):
    def test_train_predict_info_and_weights(self):
        from tests.test_cli import run_cli, run_json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            data = os.path.join(tmp, "corpus.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(TEXTS))
            run_cli("--kind", "resonant", "train", "--data", data, "--epochs", "2", model=path)
            info = run_json("info", model=path)
            self.assertEqual(info["stats"]["kind"], "resonant")
            self.assertEqual(info["stats"]["epochs_total"], 2)
            predicted = run_json("predict", "--prefix", "the ", "--length", "8", "--mode", "beam", model=path)
            self.assertTrue(predicted["full_text"].startswith("the "))
            weights = run_json("weights", "--kick-scale", "1.0", "--buckets", "16", model=path)
            self.assertEqual((weights["weights"]["kick_scale"], weights["weights"]["buckets"]), (1.0, 16))
            run_cli("weights", "--window", "5", model=path, expect=1)  # a count-model option

    def test_the_default_model_file_follows_the_kind(self):
        from modelkit.cli import DEFAULT_RESONANT_MODEL, build_parser

        args = build_parser().parse_args(["--kind", "resonant", "info"])
        self.assertEqual(args.kind, "resonant")
        self.assertEqual(DEFAULT_RESONANT_MODEL, "model.resonant.json")


class TestEvolveAndDialogue(unittest.TestCase):
    def test_evolver_runs_a_generation(self):
        from modelkit.gan import EvolveConfig, Evolver

        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=2)
        evolver = Evolver(model, TEXTS, config=EvolveConfig(samples=3, real_per_generation=2, max_length=30, seed=1))
        record = evolver.run_generation()
        for field in ("generation", "fake_score_mean", "real_score_mean", "gap", "nodes", "edges"):
            self.assertIn(field, record)

    def test_converse(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=2)
        turns = converse(model, "the sun", turns=2)
        self.assertEqual(len(turns), 3)


class TestPersistence(unittest.TestCase):

    def test_checkpoints_keep_the_kind(self):
        from modelkit.checkpoint import CheckpointManager

        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep=2)
            record = manager.save(model, 1, "epoch", {"loss": 0.5})
            back = load_model(record["path"])
        self.assertIsInstance(back, ResonantNet)
        self.assertEqual(back.kind, "resonant")


if __name__ == "__main__":
    unittest.main()
