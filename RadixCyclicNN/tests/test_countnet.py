"""Tests for radixnet.countnet (the count / reward model) and radixnet.beam (top-K / bottom-K search).

Also covers the model-kind plumbing: ``load_model`` detecting the kind, the
API's ``/api/model`` endpoints, kind-aware prediction / feedback and the
CLI's ``--kind`` option.
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.beam import Prediction, beam_predict, default_beam, path_probability  # noqa: E402
from radixnet.countnet import COUNT_MODEL_FORMAT, CountRewardGraph, CountRewardNet  # noqa: E402
from radixnet.graph import END, START  # noqa: E402
from radixnet.model import RadixNet, load_model, model_class, model_from_dict, model_kinds, new_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = [
    "the cat sat on the mat",
    "the cat ran to the door",
    "the dog sat on the log",
    "the dog ate the bone",
    "the cat sat on the mat",
]


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


def trained(epochs=2, seed=1):
    model = CountRewardNet(seed=seed)
    model.train(TEXTS, epochs=epochs)
    return model


class TestWeightFunction(unittest.TestCase):
    def test_weight_is_log_count_plus_reward(self):
        g = CountRewardGraph(seed=0, count_scale=1.0, reward_scale=2.0)
        self.assertAlmostEqual(g.weight(0, 0.0), 0.0)
        self.assertAlmostEqual(g.weight(3, 0.0), math.log(4))
        self.assertAlmostEqual(g.weight(3, -1.5), math.log(4) - 3.0)
        self.assertAlmostEqual(g.weight(-5, 1.0), 2.0)  # a negative count counts as 0

    def test_activations_are_one_so_scores_are_weights(self):
        model = trained()
        g = model.graph
        for node in g.alive_nodes():
            self.assertAlmostEqual(g.activation_of(node), 1.0)
        for p in g.alive_nodes():
            for c, score in g.child_scores(p):
                self.assertAlmostEqual(score, g.edge_w[g.children[p][c]])
        for p in g.alive_nodes():
            for c, e in g.children[p].items():
                self.assertAlmostEqual(g.edge_w[e], g.weight(g.edge_count[e], g.edge_reward[e]))

    def test_probabilities_follow_counts(self):
        model = CountRewardNet(seed=0)
        model.train(["abx", "abx", "abx", "aby"], epochs=1)
        g = model.graph
        node, _ = g.lookup("abx")
        parent = next(iter(g.parents[node]))
        probs = dict(g.child_probs(parent))
        x = probs[node]
        y = probs[g.lookup("aby")[0]]
        # (1 + 3) : (1 + 1) = 2 : 1
        self.assertAlmostEqual(x / y, 2.0)
        model.train(["abx", "abx", "abx", "aby"], epochs=1)  # counts double: (1 + 6) : (1 + 2)
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[node] / probs[g.lookup("aby")[0]], 7 / 3)

    def test_reward_and_penalty_move_odds_by_e(self):
        model = CountRewardNet(seed=0)
        model.train(["abx", "aby"], epochs=1)
        g = model.graph
        x = g.lookup("abx")[0]
        y = g.lookup("aby")[0]
        parent = next(iter(g.parents[x]))
        self.assertAlmostEqual(dict(g.child_probs(parent))[x], 0.5)
        model.reward(["abx"], strength=1.0)
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[x] / probs[y], math.e * (1 + 2) / (1 + 1))  # reward also counts a traversal
        model.punish(["abx"], strength=2.0)
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[x] / probs[y], math.exp(-1.0) * 3 / 2)
        self.assertEqual(g.edge_count[g.children[parent][x]], 2)  # penalties do not count traversals

    def test_invert_flips_rewards_only(self):
        model = trained()
        model.reward([TEXTS[0]], strength=1.5)
        g = model.graph
        counts = list(g.edge_count)
        rewards = list(g.edge_reward)
        model.invert()
        self.assertTrue(g.inverted)
        self.assertEqual(g.edge_count, counts)
        self.assertEqual(g.edge_reward, [-r for r in rewards])
        model.invert()
        self.assertFalse(g.inverted)
        self.assertEqual(g.edge_reward, rewards)

    def test_invariants_and_compression(self):
        model = trained(epochs=3)
        model.graph.check_invariants(TEXTS, compressed=True)
        self.assertGreater(model.graph.compression_ratio(), 1.0)
        self.assertEqual(len(model.graph.edge_reward), len(model.graph.edge_w))


class TestTraining(unittest.TestCase):
    def test_records_and_meta(self):
        model = CountRewardNet(seed=3)
        records = model.train(TEXTS + ["hi"], epochs=3, lr=0.9, act_lr=0.9, batch_size=1)  # rates are ignored
        self.assertEqual([r["epoch"] for r in records], [1, 2, 3])
        self.assertEqual(model.history, records)
        for r in records:
            self.assertEqual(r["skipped_short"], 1)
            self.assertTrue(r["traversed"])
            self.assertEqual(r["reward"], 0.0)
            self.assertAlmostEqual(r["perplexity"], math.exp(r["loss"]), places=12)
        self.assertEqual(model.meta["trained_texts"], 5)
        self.assertEqual(model.meta["epochs_total"], 3)
        # every pass walks the same transitions and the loss never rises: counts only sharpen the data
        self.assertEqual(len({r["transitions"] for r in records}), 1)
        self.assertLessEqual(records[-1]["loss"], records[0]["loss"] + 1e-12)
        stats = model.stats()
        self.assertEqual(stats["kind"], "count")
        self.assertEqual((stats["backend"], stats["device"]), ("python", "cpu"))
        for key in ("rewards_total", "penalties_total", "feedback_passes", "edge_reward_positive", "edge_reward_negative"):
            self.assertIn(key, stats)

    def test_two_nrl_penalises_then_rewards_without_inverting(self):
        model = trained()
        # a bad text the model has seen: its path exists, so the penalty lowers a real score (an unseen bad text
        # gets its path created with a negative reward, which is still far likelier than an unknown trigram)
        bad, good = ["the dog sat on the log"], ["the dog ate the bone"]
        before_bad = model.score(bad[0])["log_prob"]
        model.punish(bad, epochs=2, strength=1.5)
        self.assertLess(model.score(bad[0])["log_prob"], before_bad)
        mid_good = model.score(good[0])["log_prob"]  # shares "the dog " with the bad text, so it dropped a little too
        model.reward(good, epochs=1, strength=1.5)
        self.assertGreater(model.score(good[0])["log_prob"], mid_good)
        result = model.two_nrl(bad, good, neg_epochs=2, pos_epochs=1, strength=1.5)
        self.assertEqual([r["phase"] for r in result["negative"]], ["negative", "negative"])
        self.assertEqual([r["phase"] for r in result["positive"]], ["positive"])
        self.assertEqual([r["reward"] for r in result["negative"] + result["positive"]], [-1.5, -1.5, 1.5])
        self.assertFalse(result["inverted"])
        self.assertEqual(model.meta["twonrl_runs"], 1)
        self.assertEqual(model.meta["feedback_passes"], 6)
        with self.assertRaises(TypeError):
            model.two_nrl(bad, good, lr=0.1)

    def test_stop_event_skips_the_positive_phase(self):
        import threading

        model = trained()
        stop = threading.Event()
        stop.set()
        result = model.two_nrl(["zzz qqq"], ["the cat sat on the mat"], neg_epochs=3, pos_epochs=3)
        self.assertEqual(len(result["negative"]), 3)
        self.assertEqual(len(result["positive"]), 3)
        result = model.two_nrl(["zzz qqq"], ["the cat sat on the mat"], neg_epochs=3, pos_epochs=3, stop_event=stop)
        self.assertEqual(len(result["negative"]), 1)
        self.assertEqual(result["positive"], [])

    def test_generate_score_and_locate_work(self):
        model = trained()
        samples = model.generate(count=3, max_length=30, seed=7)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(isinstance(s.text, str) for s in samples))
        best = model.generate(mode="dijkstra", max_length=40)[0]
        self.assertTrue(best.reached_end)
        self.assertGreater(model.score("the cat sat on the mat")["log_prob"], model.score("the cat sat on the log")["log_prob"] - 50)
        self.assertEqual(model.score("qqq")["unknown_transitions"], 2)
        self.assertEqual(model.locate("")[0], START)


class TestPrediction(unittest.TestCase):
    def test_top_and_bottom_k(self):
        model = trained()
        p = model.predict("the cat", length=8, k=3)
        self.assertIsInstance(p, Prediction)
        self.assertEqual((p.k, p.beam, p.mode), (3, default_beam(3), "beam"))
        self.assertEqual(len(p.top), 3)
        self.assertLessEqual(len(p.bottom), 3)
        self.assertGreater(len(p.bottom), 0)
        costs = [r.cost for r in p.top]
        self.assertEqual(costs, sorted(costs))
        worst = [r.cost for r in p.bottom]
        self.assertEqual(worst, sorted(worst, reverse=True))
        self.assertGreaterEqual(min(worst), max(costs) - 1e-9)
        # the prediction itself is the best path
        self.assertEqual((p.text, p.cost, p.node_ids), (p.top[0].text, p.top[0].cost, p.top[0].node_ids))
        self.assertEqual(p.full_text, "the cat" + p.text)
        self.assertTrue(all(r.full_text == "the cat" + r.text for r in p.top + p.bottom))
        texts = [tuple(r.node_ids) for r in p.top + p.bottom]
        self.assertEqual(len(texts), len(set(texts)))  # no path on both sides
        for r in p.top:
            self.assertTrue(r.reached_end or len(r.text) >= 8)
        d = p.to_dict()
        self.assertEqual(set(d) - set(p.top[0].to_dict()), {"top", "bottom", "k", "beam", "mode"})
        self.assertEqual(len(d["top"]), 3)

    def test_dijkstra_alias_sample_mode_and_edge_cases(self):
        model = trained()
        alias = model.predict("the cat", length=8, k=2, mode="dijkstra")
        self.assertEqual(alias.mode, "beam")
        self.assertEqual(alias.text, model.predict("the cat", length=8, k=2).text)
        walk = model.predict("the cat", length=6, mode="sample")
        self.assertEqual((walk.mode, len(walk.top), walk.bottom), ("sample", 1, []))
        self.assertTrue(walk.full_text.startswith("the cat"))
        self.assertEqual(model.predict("the cat", length=0, k=2).text, "")
        self.assertEqual(model.predict("the cat", length=5, k=0).top, [])
        capped = model.predict("the cat", length=3, k=2, max_length=4)
        self.assertTrue(all(len(r.text) <= 4 for r in capped.top + capped.bottom))
        unknown = model.predict("qqq", length=5, k=2)
        self.assertTrue(unknown.full_text.startswith("qqq"))
        to_end = model.predict("the", length=1, k=2, to_end=True)
        self.assertTrue(all(r.reached_end for r in to_end.top))
        self.assertEqual(model.predict("the", length=0, k=2, to_end=True).text, "")  # length 0 emits nothing, as RadixNet
        with self.assertRaises(ValueError):
            model.predict("the", mode="nope")
        with self.assertRaises(ValueError):
            model.predict("the", k=-1)
        with self.assertRaises(ValueError):
            model.predict("the", beam=0)
        with self.assertRaises(TypeError):
            model.predict(5)

    def test_beam_matches_dijkstra_on_a_radix_model(self):
        model = RadixNet(seed=2, backend="python")
        model.train(_lines("sample_corpus.txt")[:20], epochs=2, batch_size=8, lr=0.3)
        node, offset = model.locate("the ")
        top, bottom, expanded = beam_predict(model.graph, node, offset, min_chars=6, k=4, beam=64)
        self.assertGreater(expanded, 0)
        best = model.predict("the ", length=6)
        self.assertAlmostEqual(top[0].cost, best.cost, places=9)
        self.assertEqual(top[0].text, best.text)
        self.assertEqual(len(top), 4)
        self.assertTrue(all(0.0 < path_probability(r) <= 1.0 for r in top + bottom))
        self.assertEqual(beam_predict(model.graph, node, offset, min_chars=6, k=0), ([], [], 0))

    def test_bottom_paths_stay_comparable_with_to_end(self):
        model = trained()
        node, offset = model.locate("the ")
        top, bottom, _ = beam_predict(model.graph, node, offset, min_chars=0, k=2, to_end=True)
        longest_top = max(len(r.text) for r in top)
        self.assertTrue(all(len(r.text) <= 2 * longest_top + 40 for r in bottom))


class TestPersistenceAndKinds(unittest.TestCase):
    def test_round_trip_and_kind_detection(self):
        model = trained()
        model.reward([TEXTS[0]], strength=2.0)
        model.punish(["zzz qqq"], strength=0.5)
        d = model.to_dict()
        self.assertEqual((d["format"], d["kind"]), (COUNT_MODEL_FORMAT, "count"))
        self.assertIn("reward", d["graph"]["edges"])
        again = model_from_dict(d)
        self.assertIsInstance(again, CountRewardNet)
        self.assertEqual(again.stats(), model.stats())

        def rewards_by_labels(graph):  # to_dict compacts tombstoned edges away, so edge ids differ after a reload
            return {
                (graph.labels[p], graph.labels[c]): (graph.edge_count[e], graph.edge_reward[e])
                for p in graph.alive_nodes() for c, e in graph.children[p].items()
            }

        self.assertEqual(rewards_by_labels(again.graph), rewards_by_labels(model.graph))
        first, second = again.predict("the cat", length=8, k=3), model.predict("the cat", length=8, k=3)
        self.assertEqual([(r.text, round(r.cost, 9)) for r in first.top], [(r.text, round(r.cost, 9)) for r in second.top])
        self.assertEqual([(r.text, round(r.cost, 9)) for r in first.bottom], [(r.text, round(r.cost, 9)) for r in second.bottom])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            model.save(path)
            loaded = load_model(path)
            self.assertIsInstance(loaded, CountRewardNet)
            self.assertEqual(loaded.history, model.history)
            radix = RadixNet(seed=0, backend="python")
            radix.save(os.path.join(tmp, "r.json"))
            self.assertIsInstance(load_model(os.path.join(tmp, "r.json")), RadixNet)
            with self.assertRaises(ValueError):
                CountRewardNet.load(os.path.join(tmp, "r.json"))
        with self.assertRaises(ValueError):
            model_from_dict({"format": "nope"})

    def test_kind_registry(self):
        self.assertEqual([k["kind"] for k in model_kinds()], ["radix", "count"])
        self.assertIs(model_class("count"), CountRewardNet)
        self.assertIs(model_class(None), RadixNet)
        self.assertIs(model_class(" Radix "), RadixNet)
        with self.assertRaises(ValueError):
            model_class("nope")
        self.assertIsInstance(new_model("count", seed=4), CountRewardNet)
        self.assertEqual(new_model("count", seed=4).seed, 4)

    def test_checkpoints_keep_the_kind(self):
        from radixnet.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep=2)
            model = trained()
            manager.save(model, 1, "epoch")
            restored = manager.load_latest()
            self.assertIsInstance(restored, CountRewardNet)
            self.assertEqual(restored.stats(), model.stats())

    def test_radix_reward_and_punish(self):
        model = RadixNet(seed=1, backend="python")
        model.train(TEXTS, epochs=1, batch_size=4)
        records = model.reward([TEXTS[0]], epochs=2, lr=0.2, batch_size=4, strength=3.0)  # strength is ignored
        self.assertEqual([r["phase"] for r in records], ["positive", "positive"])
        self.assertEqual(records[0]["act_lr"], 0.2 / 10)
        self.assertFalse(model.graph.inverted)
        records = model.punish(["zzz qqq"], epochs=1, lr=0.5, batch_size=4)
        self.assertEqual([r["phase"] for r in records], ["negative"])
        self.assertTrue(model.graph.inverted)

    def test_evolver_uses_the_generator_kind(self):
        from radixnet.gan import EvolveConfig, Evolver

        model = trained()
        evolver = Evolver(model, TEXTS, config=EvolveConfig(samples=3, neg_epochs=1, pos_epochs=1, disc_neg_epochs=1, disc_pos_epochs=1))
        self.assertIsInstance(evolver.discriminator, CountRewardNet)
        record = evolver.run_generation()
        self.assertEqual(record["generation"], 1)


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
        self.assertEqual([k["kind"] for k in data["kinds"]], ["radix", "count"])
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
        status, data, _ = self.client.post("/api/save", {})
        self.assertEqual(status, 200)
        self.assertTrue(data["path"].endswith("model.count.json"))
        self.assertIsInstance(load_model(data["path"]), CountRewardNet)

    def test_radix_ignores_beam_options_and_load_switches_kind(self):
        self.select("count")
        self.client.post("/api/train", {"texts": TEXTS[:2], "epochs": 1})
        self.wait()
        status, data, _ = self.client.post("/api/save", {"path": os.path.join(self.tmp.name, "saved.count.json")})
        self.assertEqual(status, 200)
        self.select("radix")
        status, p, _ = self.client.post("/api/predict", {"prefix": "the", "length": 3, "k": 3, "beam": 8, "mode": "beam"})
        self.assertEqual(status, 200, p)
        self.assertEqual(p["kind"], "radix")
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
            # --kind on an existing file is only a note; the file's kind wins
            proc = run_cli("--kind", "radix", "info", model=model, json_mode=False)
            self.assertIn("kind", proc.stdout)
            self.assertIn("count", proc.stdout)
            self.assertIn("--kind radix applies to new models only", proc.stderr)


if __name__ == "__main__":
    unittest.main()
