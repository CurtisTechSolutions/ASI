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
