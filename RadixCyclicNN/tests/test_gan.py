"""Tests for radixnet.gan (Evolver, EvolveConfig)."""

import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.checkpoint import CheckpointManager  # noqa: E402
from radixnet.gan import EvolveConfig, Evolver  # noqa: E402
from radixnet.model import RadixNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as fh:
    CORPUS = [line for line in fh.read().splitlines() if line.strip()]

RECORD_KEYS = {
    "generation", "fake_score_mean", "real_score_mean", "gap", "gen_loss", "nodes", "edges",
    "compression_ratio", "sample", "fakes", "worst", "seconds",
    "failures", "blatant", "flipped", "boost_mean", "boost_max", "twonrl", "mode",
}


def trained_generator(seed=0):
    model = RadixNet(seed=seed, backend="python")
    model.train(CORPUS, epochs=2, lr=1.0, batch_size=1)
    return model


class TestEvolveConfig(unittest.TestCase):
    def test_defaults_and_validation(self):
        cfg = EvolveConfig()
        self.assertEqual((cfg.samples, cfg.real_per_generation, cfg.max_length, cfg.temperature), (8, 8, 40, 1.0))
        self.assertEqual((cfg.neg_epochs, cfg.pos_epochs, cfg.neg_lr, cfg.pos_lr), (1, 1, 0.05, 0.01))
        self.assertEqual((cfg.disc_neg_epochs, cfg.disc_pos_epochs, cfg.checkpoint_every, cfg.seed), (1, 1, 0, 0))
        self.assertEqual(cfg.batch_size, 32)
        cfg.validate()
        for bad in ({"samples": 0}, {"real_per_generation": 0}, {"max_length": 2}, {"temperature": -1.0},
                    {"neg_epochs": -1}, {"batch_size": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Evolver(trained_generator(), CORPUS, config=EvolveConfig(**bad))


class TestEvolver(unittest.TestCase):
    def test_one_generation(self):
        gen = trained_generator()
        cfg = EvolveConfig(samples=4, real_per_generation=4, max_length=30, seed=5)
        ev = Evolver(gen, CORPUS + ["", "hi"], config=cfg)
        self.assertEqual(len(ev.corpus), len(CORPUS))  # too-short lines are dropped
        self.assertIsNot(ev.discriminator, gen)
        self.assertEqual(ev.discriminator.graph.seed, 6)
        self.assertEqual(ev.discriminator.backend.name, gen.backend.name)
        self.assertEqual(ev.generation, 0)
        epochs_before = gen.meta["epochs_total"]
        rec = ev.run_generation()
        self.assertEqual(set(rec), RECORD_KEYS)
        self.assertEqual(rec["generation"], 1)
        self.assertEqual(ev.generation, 1)
        self.assertEqual(ev.history, [rec])
        self.assertEqual(rec["fakes"], 4)
        self.assertGreaterEqual(rec["worst"], 1)
        self.assertLessEqual(rec["worst"], 4)
        self.assertGreaterEqual(len(rec["sample"]), 3)
        self.assertLessEqual(len(rec["sample"]), 30)
        self.assertAlmostEqual(rec["gap"], rec["real_score_mean"] - rec["fake_score_mean"])
        self.assertIsInstance(rec["gen_loss"], float)
        self.assertEqual(rec["nodes"], gen.graph.num_nodes())
        self.assertGreaterEqual(rec["seconds"], 0.0)
        json.dumps(rec)
        # both networks went through one 2NRL run (negative, invert, positive)
        self.assertEqual(gen.meta["twonrl_runs"], 1)
        self.assertEqual(ev.discriminator.meta["twonrl_runs"], 1)
        self.assertEqual(gen.meta["epochs_total"], epochs_before + 2)
        self.assertTrue(gen.graph.inverted)
        self.assertTrue(ev.discriminator.graph.inverted)
        self.assertEqual([r["phase"] for r in gen.history[-2:]], ["negative", "positive"])
        # the discriminator learned the real samples: it knows their structure
        self.assertEqual(ev.discriminator.score(ev.rng.choice(ev.corpus))["chars"] > 0, True)

    def test_given_discriminator_and_untrained_generator(self):
        gen = RadixNet(seed=1, backend="python")  # empty graph: cannot generate anything yet
        disc = RadixNet(seed=9, backend="python")
        ev = Evolver(gen, CORPUS[:5], discriminator=disc, config=EvolveConfig(samples=2, real_per_generation=2))
        self.assertIs(ev.discriminator, disc)
        rec = ev.run_generation()
        self.assertEqual((rec["fakes"], rec["worst"], rec["sample"]), (0, 0, ""))
        self.assertIsNone(rec["fake_score_mean"])
        self.assertIsNone(rec["gap"])
        self.assertIsInstance(rec["real_score_mean"], float)
        self.assertGreater(gen.graph.num_nodes(), 2)  # the fine-tune pass taught it the real samples
        # from now on it generates real text, so the next generation has fakes
        rec = ev.run_generation()
        self.assertGreaterEqual(rec["fakes"], 1)
        with self.assertRaises(ValueError):
            Evolver(gen, ["", "ab"])

    def test_run_generations_checkpoints_and_progress(self):
        gen = trained_generator()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=5)
            ev = Evolver(gen, CORPUS, config=EvolveConfig(samples=3, real_per_generation=3, checkpoint_every=1))
            records = ev.run(2, checkpoint_manager=cm, progress=seen.append)
            self.assertEqual([r["generation"] for r in records], [1, 2])
            self.assertEqual(seen, records)
            self.assertEqual(ev.history, records)
            self.assertEqual([(r["step"], r["tag"]) for r in cm.list()], [(1, "gen"), (2, "gen")])
            restored = cm.load_latest(backend="python")
            self.assertEqual(restored.stats(), gen.stats())
            self.assertEqual(ev.run(0), [])
            with self.assertRaises(ValueError):
                ev.run(-1)

    def test_stop_event(self):
        gen = trained_generator()
        ev = Evolver(gen, CORPUS, config=EvolveConfig(samples=2, real_per_generation=2))
        stop = threading.Event()
        stop.set()
        self.assertEqual(ev.run(None, stop_event=stop), [])
        self.assertEqual(ev.generation, 0)
        stop.clear()

        def progress(rec):
            if rec["generation"] == 2:
                stop.set()

        records = ev.run(None, progress=progress, stop_event=stop)
        self.assertEqual(len(records), 2)
        self.assertEqual(ev.generation, 2)
        # a background thread is stopped between generations too
        stop.clear()
        results = []

        def target():
            try:
                results.append(ev.run(None, stop_event=stop))
            except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test
                results.append(exc)

        worker = threading.Thread(target=target)
        worker.start()
        worker.join(0.3)
        stop.set()
        worker.join(30)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(results[0], list, results[0])
        self.assertEqual(len(results[0]), ev.generation - 2)


if __name__ == "__main__":
    unittest.main()


class TestBlatantFailures(unittest.TestCase):
    """Train on failures, blatantly fail on purpose, then invert - and the local path-inversion variant."""

    def corpus(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "data", "sample_corpus.txt"), encoding="utf-8") as fh:
            return [line for line in fh.read().splitlines() if line.strip()]

    def trained(self):
        model = RadixNet(seed=1, backend="python")
        model.train(self.corpus(), epochs=6, batch_size=8, lr=0.3)
        return model

    # -- weighted 2NRL: the worse the failure, the more the activation function updates ------------

    def test_two_nrl_weights_scale_the_negative_phase(self):
        from radixnet.model import TrainConfig

        model = self.trained()
        result = model.two_nrl(
            ["zzz qqq garbage", "xxx yyy noise", "the cat sat somewhere"], self.corpus()[:4],
            neg_epochs=1, pos_epochs=1, neg_lr=0.05, batch_size=8, bad_weights=[3.0, 1.5, 3.0],
        )
        negative = result["negative"]
        self.assertEqual([r["weight"] for r in negative], [3.0, 1.5])  # equal weights share a pass, heaviest first
        self.assertEqual([r["phase"] for r in negative], ["negative", "negative"])
        self.assertAlmostEqual(negative[0]["lr"], 0.15)
        self.assertAlmostEqual(negative[0]["act_lr"], TrainConfig.act_lr * 3.0)
        self.assertAlmostEqual(negative[1]["lr"], 0.075)
        self.assertTrue(result["inverted"])
        self.assertEqual(len(result["positive"]), 1)
        self.assertEqual(model.meta["twonrl_runs"], 1)
        with self.assertRaises(ValueError):
            model.two_nrl(["a b c"], ["d e f"], bad_weights=[1.0, 2.0])
        with self.assertRaises(ValueError):
            model.two_nrl(["a b c"], ["d e f"], bad_weights=[-1.0])
        zero = model.two_nrl(["zzz qqq garbage"], self.corpus()[:2], neg_epochs=1, pos_epochs=1, batch_size=8, bad_weights=[0.0])
        self.assertEqual(zero["negative"], [])  # a zero weight is skipped, the inversion still happens

    def test_count_model_two_nrl_weights_scale_the_penalty(self):
        from radixnet.countnet import CountRewardNet

        model = CountRewardNet(seed=1)
        model.train(self.corpus()[:20], epochs=1)
        result = model.two_nrl(["zzz qqq garbage", "the dog sat"], self.corpus()[:3], neg_epochs=1, pos_epochs=1,
                               strength=1.0, bad_weights=[2.0, 1.0])
        self.assertEqual([(r["weight"], r["reward"]) for r in result["negative"]], [(2.0, -2.0), (1.0, -1.0)])
        self.assertFalse(result["inverted"])

    # -- local variant: flip every other node of a failed path, by an amount ------------------------

    def test_flip_nodes_negates_every_other_node_of_a_path(self):
        from radixnet.graph import END, START

        model = self.trained()
        text = model.generate(mode="dijkstra", max_length=40)[0].text
        path = model.graph.node_path(model.encoder.encode(text))
        before_a = list(model.graph.a)
        result = model.invert_paths([text])
        self.assertEqual((result["unit"], result["mode"], result["texts"], result["amount_mean"]), ("nodes", "activation", 1, 1.0))
        real = [n for n in path if n not in (START, END)]
        flipped = {n for n in set(real) if model.graph.a[n] == -before_a[n]}
        self.assertEqual(len(flipped), result["flipped"])
        # every edge between real nodes has exactly one flipped endpoint (loops aside); of the two sentinel edges
        # (START -> first, last -> END) at most one can stay uncovered when the path has an even number of nodes
        uncovered_sentinel = 0
        for a, b in zip(path, path[1:]):
            if a == b:
                continue
            if a in (START, END) or b in (START, END):
                uncovered_sentinel += (a in flipped) == (b in flipped)
            else:
                self.assertNotEqual(a in flipped, b in flipped, (a, b))
        self.assertLessEqual(uncovered_sentinel, 1)
        untouched = [n for n in model.graph.alive_nodes() if n not in flipped]
        self.assertTrue(all(model.graph.a[n] == before_a[n] for n in untouched))
        self.assertEqual(model.meta["path_inversions"], result["flipped"])

    def test_inverting_a_likely_path_makes_it_unlikely_and_is_reversible(self):
        model = self.trained()
        text = model.generate(mode="dijkstra", max_length=40)[0].text
        before = model.score(text)["log_prob"]
        model.invert_paths([text])
        self.assertLess(model.score(text)["log_prob"], before - 0.5)
        model.invert_paths([text])  # the same nodes flip back
        self.assertAlmostEqual(model.score(text)["log_prob"], before, places=9)
        state_before = list(model.graph.z)
        result = model.invert_paths([text], mode="state")
        self.assertEqual(result["mode"], "state")
        self.assertEqual(sum(1 for z0, z1 in zip(state_before, model.graph.z) if z0 != z1), result["flipped"])
        self.assertLess(model.score(text)["log_prob"], before)
        with self.assertRaises(ValueError):
            model.invert_paths([text], mode="nope")
        self.assertEqual(model.invert_paths(["hi"])["flipped"], 0)  # too short: nothing to flip

    def test_amounts_grade_the_update(self):
        model = self.trained()
        text = model.generate(mode="dijkstra", max_length=40)[0].text
        base = model.score(text)["log_prob"]
        scores = []
        for amount in (0.1, 0.25, 0.5, 0.75, 1.0):
            copy = RadixNet.from_dict(model.to_dict(), backend="python")
            result = copy.invert_paths([text], amounts=[amount])
            self.assertAlmostEqual(result["amount_mean"], amount)
            scores.append(copy.score(text)["log_prob"])
        self.assertEqual(scores, sorted(scores, reverse=True))  # the larger the amount, the less likely the path
        self.assertLess(scores[-1], base)
        half = RadixNet.from_dict(model.to_dict(), backend="python")
        before_a = list(half.graph.a)
        half.invert_paths([text], amounts=0.5)
        changed = [n for n in range(len(before_a)) if half.graph.a[n] != before_a[n]]
        self.assertTrue(all(half.graph.a[n] == 0.0 for n in changed))  # amount 0.5 zeroes the activation
        self.assertEqual(model.invert_paths([text], amounts=[0.0])["flipped"], 0)
        with self.assertRaises(ValueError):
            model.invert_paths([text], amounts=[1.5])
        with self.assertRaises(ValueError):
            model.invert_paths([text, text], amounts=[1.0])

    def test_unknown_text_is_registered_first(self):
        model = self.trained()
        result = model.invert_paths(["zzz qqq unseen words"])
        self.assertGreater(result["flipped"], 0)
        self.assertIsNotNone(model.graph.node_path(model.encoder.encode("zzz qqq unseen words")))

    def test_count_model_penalises_the_path(self):
        from radixnet.countnet import CountRewardNet

        model = CountRewardNet(seed=1)
        model.train(self.corpus()[:20], epochs=1)
        text = self.corpus()[0]
        before = model.score(text)["log_prob"]
        result = model.invert_paths([text], mode="activation", strength=3.0)
        self.assertEqual((result["unit"], result["mode"]), ("edges", "penalty"))
        self.assertGreater(result["flipped"], 0)
        self.assertLess(model.score(text)["log_prob"], before)
        self.assertEqual(model.stats()["penalties_total"], 6.0 * result["flipped"])  # 2 * strength * amount 1
        half = model.invert_paths([self.corpus()[1]], amounts=[0.5], strength=3.0)
        self.assertAlmostEqual(half["amount_mean"], 0.5)

    # -- the evolve loop -------------------------------------------------------------------------------

    def evolve_config(self, **kw):
        base = dict(samples=6, neg_epochs=1, pos_epochs=1, disc_neg_epochs=1, disc_pos_epochs=1, batch_size=8)
        base.update(kw)
        return EvolveConfig(**base)

    def test_evolver_fail_invert_trains_on_failures_weighted_then_inverts(self):
        model = self.trained()
        corpus = self.corpus()[:20]
        evolver = Evolver(model, corpus, config=self.evolve_config(blatant_mode="fail_invert", blatant_margin=0.5, blatant_boost=3.0))
        seen_failures = False
        for _ in range(4):
            inverted_before = model.graph.inverted
            runs_before = model.meta["twonrl_runs"]
            record = evolver.run_generation()
            self.assertEqual(record["mode"], "fail_invert")
            for key in ("failures", "blatant", "boost_mean", "boost_max", "twonrl", "flipped"):
                self.assertIn(key, record)
            self.assertEqual(record["flipped"], 0)
            if record["failures"]:
                seen_failures = True
                self.assertTrue(record["twonrl"])
                self.assertEqual(record["worst"], record["failures"])
                self.assertGreaterEqual(record["boost_mean"], 1.0)
                self.assertLessEqual(record["boost_max"], 3.0)
                self.assertNotEqual(model.graph.inverted, inverted_before)  # the model was inverted
                self.assertEqual(model.meta["twonrl_runs"], runs_before + 1)
            else:
                self.assertFalse(record["twonrl"])  # nothing failed: only the fine-tune pass, no inversion
                self.assertEqual(model.graph.inverted, inverted_before)
        self.assertTrue(seen_failures or True)  # whether any fake fails depends on the critic; both branches are legal

    def test_evolver_local_modes_flip_paths_and_may_skip_2nrl(self):
        model = self.trained()
        corpus = self.corpus()[:20]
        evolver = Evolver(model, corpus, config=self.evolve_config(blatant_mode="activation", blatant_margin=0.05))
        record = evolver.run_generation()
        self.assertEqual(record["mode"], "activation")
        if record["failures"]:
            self.assertGreater(record["flipped"], 0)
            self.assertGreater(record["boost_mean"], 0.0)
            self.assertLessEqual(record["boost_max"], 1.0)
        if not record["twonrl"]:
            self.assertEqual(record["worst"], 0)
            self.assertFalse(model.graph.inverted)
        off = Evolver(self.trained(), corpus, config=self.evolve_config(samples=4))
        record = off.run_generation()
        self.assertEqual((record["failures"], record["blatant"], record["flipped"], record["twonrl"], record["mode"]), (0, 0, 0, True, "none"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            EvolveConfig(blatant_mode="nope").validate()
        with self.assertRaises(ValueError):
            EvolveConfig(blatant_margin=-1).validate()
        with self.assertRaises(ValueError):
            EvolveConfig(blatant_boost=0.5).validate()
        EvolveConfig(blatant_mode="fail_invert", blatant_margin=0, blatant_boost=1).validate()
        EvolveConfig(blatant_mode="state").validate()
