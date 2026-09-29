"""The count / reward model through the kit: checkpoints, evolve, the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_countnet.py).
"""

import math
import os
import sys
import tempfile
import unittest
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.model import load_model  # noqa: E402

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the model's checkout: its sample data, and the directory `python -m radixnet` runs in
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")


TEXTS = [
    "the cat sat on the mat",
    "the cat ran to the door",
    "the dog sat on the log",
    "the dog ate the bone",
    "the cat sat on the mat",
]


def trained(epochs=2, seed=1):
    model = CountRewardNet(seed=seed)
    model.train(TEXTS, epochs=epochs)
    return model


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-count-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model_path = os.path.join(cls.tmp.name, "model.json")
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, model_path=cls.model_path)

    def wait(self):
        from tests.test_api import wait_for_job

        return wait_for_job(self.client)

    def select(self, kind):
        status, data, _ = self.client.post("/api/model/select", {"kind": kind})
        self.assertEqual(status, 200, data)
        return data

    def test_model_endpoints_and_switching(self):
        status, data, _ = self.client.get("/api/model")
        self.assertEqual(status, 200)
        self.assertEqual(data["kind"], "radix")
        self.assertEqual([k["kind"] for k in data["kinds"]], ["radix", "count", "negative", "resonant"])
        self.assertEqual(data["paths"]["count"], os.path.join(self.tmp.name, "model.count.json"))
        data = self.select("count")
        self.assertEqual((data["kind"], data["origin"]), ("count", "new"))
        self.assertEqual(self.client.get("/api/status")[1]["kind"], "count")
        self.assertEqual(self.client.get("/api/status")[1]["model_path"], data["paths"]["count"])
        self.assertEqual(self.select("count")["origin"], "active")
        self.assertEqual(self.select("radix")["origin"], "memory")
        self.assertEqual(self.select("count")["origin"], "memory")
        status, data, _ = self.client.post("/api/model/select", {"kind": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("unknown model kind", data["error"])
        status, data, _ = self.client.post("/api/model/select", {})
        self.assertEqual(status, 400)

    def test_train_predict_feedback_on_the_count_model(self):
        self.select("count")
        status, data, _ = self.client.post("/api/train", {"texts": TEXTS, "epochs": 2})
        self.assertEqual(status, 202, data)
        job = self.wait()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(job["history"]), 2)
        self.assertIsNone(job["history"][0].get("lr"))
        status, p, _ = self.client.post("/api/predict", {"prefix": "the cat", "length": 8, "k": 2, "mode": "beam"})
        self.assertEqual(status, 200, p)
        self.assertEqual((p["kind"], p["k"], p["mode"]), ("count", 2, "beam"))
        self.assertEqual(len(p["top"]), 2)
        self.assertEqual(p["continuation"], p["top"][0]["continuation"])
        self.assertEqual(set(p["top"][0]), {"continuation", "full_text", "cost", "probability", "step_costs", "path", "node_ids", "reached_end"})
        self.assertAlmostEqual(p["probability"], math.exp(-p["cost"]))
        status, p, _ = self.client.post("/api/predict", {"prefix": "the cat", "length": 8, "mode": "sample"})
        self.assertEqual((status, p["mode"], p["bottom"]), (200, "sample", []))
        status, data, _ = self.client.post("/api/feedback", {"good": [TEXTS[0]], "bad": ["zzz qqq"], "strength": 2.0})
        self.assertEqual(status, 202, data)
        job = self.wait()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual([(r["phase"], r["reward"]) for r in job["history"]][:3], [("negative", -2.0), ("negative", -2.0), ("positive", 2.0)])
        self.assertFalse(self.client.get("/api/status")[1]["inverted"])
        status, data, _ = self.client.post("/api/feedback", {"bad": ["zzz qqq"], "neg_epochs": 1})
        self.assertEqual(status, 202)
        self.assertEqual(self.wait()["state"], "done")
        self.assertFalse(self.client.get("/api/status")[1]["inverted"])  # punishing never inverts the count model
        status, graph, _ = self.client.get("/api/graph?limit=10")
        self.assertTrue(all("reward" in e for e in graph["edges"]))
        # the judged paths reached the graph and the endpoint reports them
        status, paths, _ = self.client.get("/api/paths?limit=5")
        self.assertEqual(status, 200, paths)
        self.assertGreater(paths["totals"]["contexts"], 0)
        self.assertGreater(paths["totals"]["correct"], 0)
        self.assertGreater(paths["totals"]["incorrect"], 0)
        self.assertEqual(paths["path_scale"], 1.0)
        self.assertLessEqual(len(paths["paths"]), 5)
        row = paths["paths"][0]
        self.assertEqual(
            set(row) >= {"after", "parent_label", "child_label", "seen", "correct", "incorrect", "correct_ratio",
                         "seen_ratio", "term"},
            True,
        )
        status, st, _ = self.client.get("/api/status")
        self.assertEqual(st["path_contexts"], paths["totals"]["contexts"])
        self.assertEqual(st["path_correct"], paths["totals"]["correct"])
        status, data, _ = self.client.get("/api/paths?limit=nope")
        self.assertEqual(status, 400)
        # and the same judgements from the node's point of view
        status, nodes, _ = self.client.get("/api/nodes?limit=4")
        self.assertEqual(status, 200, nodes)
        self.assertTrue(0 < len(nodes["nodes"]) <= 4)
        picked = next((n for n in nodes["nodes"] if len(n["to"]) > 1), None)
        self.assertIsNotNone(picked, "no node in the top 4 branches")
        self.assertAlmostEqual(sum(r["seen_ratio"] for r in picked["to"]), 1.0)
        self.assertTrue(set(picked["to"][0]) >= {"node", "label", "edge", "seen", "seen_ratio", "reward",
                                                 "reward_ratio", "path_seen", "path_ratio", "correct", "incorrect",
                                                 "correct_ratio"})
        status, one, _ = self.client.get(f"/api/nodes?node={quote(picked['label'])}")
        self.assertEqual((status, len(one["nodes"])), (200, 1))
        self.assertEqual(one["nodes"][0]["node"], picked["node"])
        self.assertEqual(self.client.get("/api/nodes?node=no+such+label")[0], 404)
        self.assertEqual(self.client.get("/api/nodes?limit=nope")[0], 400)
        status, data, _ = self.client.post("/api/save", {})
        self.assertEqual(status, 200)
        self.assertTrue(data["path"].endswith("model.count.json"))
        self.assertIsInstance(load_model(data["path"]), CountRewardNet)

    def test_weight_function_endpoint(self):
        self.select("count")
        status, data, _ = self.client.get("/api/model")
        self.assertEqual(status, 200)
        self.assertEqual(data["weights"]["function"], "dual-frequency")
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.wait()
        status, data, _ = self.client.post("/api/model/weights", {"window_scale": 2.0, "window": 20, "path_scale": 0.5})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["weights"]["window_scale"], data["weights"]["window"]), (2.0, 20))
        self.assertEqual(data["weights"]["path_scale"], 0.5)
        self.assertEqual(data["stats"]["window"], 20)
        self.assertLessEqual(data["stats"]["window_traversals"], 20)
        self.assertGreater(data["stats"]["total_traversals"], 20)
        status, graph, _ = self.client.get("/api/graph?limit=10")
        self.assertIn("total_traversals", graph)
        self.assertTrue(all({"share", "recent_share", "recent_count"} <= set(e) for e in graph["edges"]))
        status, data, _ = self.client.post("/api/model/weights", {"window": 0})
        self.assertEqual(status, 400)
        status, data, _ = self.client.post("/api/reset", {"kind": "count", "window": 5, "global_scale": 0.5})
        self.assertEqual((status, data["window"], data["global_scale"]), (200, 5, 0.5))
        self.select("radix")
        status, data, _ = self.client.post("/api/model/weights", {"window": 10})
        self.assertEqual(status, 400)
        self.assertIn("weight function", data["error"])
        status, data, _ = self.client.post("/api/reset", {"kind": "radix", "window": 10})
        self.assertEqual(status, 400)

    def test_partner_conversation_between_the_kinds_in_memory(self):
        self.select("count")
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 2})
        self.wait()
        self.select("radix")
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 2, "lr": 1.0, "batch_size": 1})
        self.wait()
        # "learn": false keeps the models exactly as trained - this is about who speaks, not about rethinks
        status, data, _ = self.client.post(
            "/api/converse",
            {"opening": TEXTS[0], "turns": 4, "partner": "count", "speakers": ["radix", "count"], "learn": False},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["kind"], data["partner"]), ("radix", "count"))
        self.assertEqual(data["count"], 5)
        self.assertEqual([t["speaker"] for t in data["turns"]], ["radix", "count", "radix", "count", "radix"])
        # the same kind as a partner means talking to itself
        status, same, _ = self.client.post(
            "/api/converse", {"opening": TEXTS[0], "turns": 2, "partner": "radix", "learn": False},
        )
        self.assertEqual((status, same["partner"]), (200, None))
        self.select("count")
        status, data, _ = self.client.post("/api/converse", {"turns": 3, "partner": "radix", "learn": False})
        self.assertEqual((status, data["kind"], data["partner"], data["count"]), (200, "count", "radix", 3))

    def test_radix_beam_prediction_and_load_switches_kind(self):
        self.select("count")
        self.client.post("/api/train", {"texts": TEXTS[:2], "epochs": 1})
        self.wait()
        status, data, _ = self.client.post("/api/save", {"path": os.path.join(self.tmp.name, "saved.count.json")})
        self.assertEqual(status, 200)
        self.select("radix")
        self.client.post("/api/train", {"texts": TEXTS[:2], "epochs": 1})
        self.wait()
        # the radix model runs the same beam search: top / bottom K in one prediction
        status, p, _ = self.client.post("/api/predict", {"prefix": "the", "length": 3, "k": 3, "beam": 8, "mode": "beam"})
        self.assertEqual(status, 200, p)
        self.assertEqual(p["kind"], "radix")
        self.assertEqual((p["mode"], p["k"], p["beam"]), ("beam", 3, 8))
        self.assertTrue(1 <= len(p["top"]) <= 3, p)
        self.assertLessEqual(len(p["bottom"]), 3)  # no path appears on both sides: a tiny graph leaves it empty
        self.assertFalse({t["full_text"] for t in p["top"]} & {b["full_text"] for b in p["bottom"]})
        self.assertEqual(p["continuation"], p["top"][0]["continuation"])
        # dijkstra stays the exact single-path search without top / bottom
        status, p, _ = self.client.post("/api/predict", {"prefix": "the", "length": 3, "k": 3, "mode": "dijkstra"})
        self.assertEqual(status, 200, p)
        self.assertNotIn("top", p)
        status, data, _ = self.client.post("/api/load", {"path": os.path.join(self.tmp.name, "saved.count.json")})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "count")
        self.assertEqual(self.client.get("/api/model")[1]["in_memory"], ["count", "radix"])
        status, data, _ = self.client.post("/api/reset", {"kind": "radix", "seed": 3})
        self.assertEqual((status, data["kind"]), (200, "radix"))
        status, data, _ = self.client.post("/api/reset", {"kind": "nope"})
        self.assertEqual(status, 400)


class TestCli(unittest.TestCase):
    def test_kind_option_end_to_end(self):
        from tests.test_cli import run_cli, run_json

        corpus = os.path.join(ROOT, "data", "sample_corpus.txt")
        with tempfile.TemporaryDirectory(prefix="radixnet-count-cli-") as tmp:
            model = os.path.join(tmp, "m.json")
            doc = run_json("--kind", "count", "train", "--data", corpus, "--epochs", 2, model=model)
            self.assertEqual(doc["stats"]["kind"], "count")
            self.assertEqual(doc["model"]["kind"], "new")
            self.assertIsInstance(load_model(model), CountRewardNet)
            doc = run_json("predict", "--prefix", "the quick", "--length", 8, "--k", 2, model=model)
            self.assertEqual((doc["kind"], doc["k"]), ("count", 2))
            self.assertEqual(len(doc["top"]), 2)
            self.assertIn("probability", doc["top"][0])
            proc = run_cli("predict", "--prefix", "the quick", "--length", 8, "--k", 2, model=model, json_mode=False)
            self.assertIn("top 2 continuation(s)", proc.stdout)
            self.assertIn("bottom", proc.stdout)
            doc = run_json("feedback", "--good-text", "the quick brown fox jumps over the lazy dog", "--bad-text", "zzz qqq",
                           "--strength", 2, model=model)
            self.assertEqual(doc["action"], "2nrl")
            self.assertFalse(doc["inverted"])
            self.assertEqual(doc["negative"][0]["reward"], -2.0)
            doc = run_json("info", model=model)
            self.assertEqual(doc["stats"]["kind"], "count")
            self.assertGreater(doc["stats"]["rewards_total"], 0)
            # correct: the alignment alone, then the same correction taught to the model
            doc = run_json("correct", "--wrong", "the quick brown fox jump", "--right", "the quick brown fox jumps",
                           "--dry-run", model=model)
            self.assertEqual(doc["changes"], [{"op": "insert", "wrong": "", "right": "s"}])
            self.assertTrue(doc["dry_run"])
            penalties = load_model(model).stats()["penalties_total"]
            doc = run_json("correct", "--wrong", "the quick brown fox jump", "--right", "the quick brown fox jumps",
                           "--keep", 0.0, model=model)
            self.assertEqual((doc["penalised"], doc["rewarded"], doc["kept"]), (1, 1, 0))
            self.assertEqual(doc["changes"], [{"op": "insert", "wrong": "", "right": "s"}])
            self.assertGreater(load_model(model).stats()["penalties_total"], penalties)
            proc = run_cli("correct", "--wrong", "a cat", "--right", "a cat", "--dry-run", model=model, json_mode=False)
            self.assertIn("nothing to teach", proc.stdout)
            # paths: the judged steps, and the counters the correction left behind
            doc = run_json("paths", "--limit", 5, model=model)
            self.assertGreater(doc["totals"]["contexts"], 0)
            self.assertGreater(doc["totals"]["incorrect"], 0)
            self.assertLessEqual(len(doc["paths"]), 5)
            self.assertEqual(set(doc["paths"][0]) >= {"prev", "edge", "seen", "correct", "incorrect", "term"}, True)
            proc = run_cli("paths", "--limit", 3, model=model, json_mode=False)
            self.assertIn("correct %", proc.stdout)
            doc = run_json("weights", "--path-scale", 0.0, model=model)
            self.assertEqual(doc["weights"]["path_scale"], 0.0)
            doc = run_json("weights", model=model)
            self.assertEqual(doc["weights"]["function"], "dual-frequency")
            self.assertEqual(doc["changed"], {})
            doc = run_json("weights", "--window", 50, "--window-scale", 0.5, model=model)
            self.assertEqual((doc["weights"]["window"], doc["weights"]["window_scale"]), (50, 0.5))
            self.assertIsNotNone(doc["saved"])
            self.assertEqual(load_model(model).weight_config()["window"], 50)
            # --kind on an existing file is only a note; the file's kind wins
            proc = run_cli("--kind", "radix", "info", model=model, json_mode=False)
            self.assertIn("kind", proc.stdout)
            self.assertIn("count", proc.stdout)
            self.assertIn("--kind radix applies to new models only", proc.stderr)


class TestPersistenceAndKinds(unittest.TestCase):

    def test_checkpoints_keep_the_kind(self):
        from modelkit.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep=2)
            model = trained()
            manager.save(model, 1, "epoch")
            restored = manager.load_latest()
            self.assertIsInstance(restored, CountRewardNet)
            self.assertEqual(restored.stats(), model.stats())

    def test_evolver_uses_the_generator_kind(self):
        from modelkit.gan import EvolveConfig, Evolver

        model = trained()
        evolver = Evolver(model, TEXTS, config=EvolveConfig(samples=3, neg_epochs=1, pos_epochs=1, disc_neg_epochs=1, disc_pos_epochs=1))
        self.assertIsInstance(evolver.discriminator, CountRewardNet)
        record = evolver.run_generation()
        self.assertEqual(record["generation"], 1)


if __name__ == "__main__":
    unittest.main()
