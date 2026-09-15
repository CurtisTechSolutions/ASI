"""Tests for audioimage.qlearn - the per-hertz Q-learning network."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage import EncodeConfig, PlaneConfig, encode  # noqa: E402
from audioimage.codec import image_to_plane  # noqa: E402
from audioimage.qlearn import QConfig, QLearnError, QNet, plane_strips, strips_plane  # noqa: E402
from audioimage.synth import melody  # noqa: E402

F_MAX = 120


def strips_of(seconds=0.25):
    cfg = EncodeConfig(
        n_fft=512, hop=128,
        plane=PlaneConfig(freq_scale="linear", f_min=0, f_max=F_MAX, height=F_MAX),
    )
    clip = melody(("C3", "E3", "G3"), seconds, 11025)
    return plane_strips(image_to_plane(encode(clip, cfg)))


class TestConfig(unittest.TestCase):
    def test_one_node_per_hertz(self):
        """The whole arrangement rests on this: 200 Hz of range is 200 nodes."""
        self.assertEqual(QConfig(f_min=0, f_max=200).nodes, 200)
        self.assertEqual(QConfig(f_min=100, f_max=300).nodes, 200)
        self.assertEqual(QNet(QConfig(f_min=0, f_max=200)).nodes, 200)

    def test_action_space(self):
        delta = QConfig(f_max=50, action_mode="delta", span=3)
        self.assertEqual(delta.actions, 7)
        self.assertEqual(delta.hold, 3)
        level = QConfig(f_max=50, action_mode="level", levels=16)
        self.assertEqual(level.actions, 16)

    def test_table_size(self):
        cfg = QConfig(f_max=100, levels=16, context=4, context_buckets=4, span=3)
        self.assertEqual(cfg.states, 64)
        self.assertEqual(cfg.table_size, 100 * 64 * 7)

    def test_context_off(self):
        self.assertEqual(QConfig(f_max=10, context=0).buckets, 1)
        self.assertEqual(QConfig(f_max=10, context=0, levels=8).states, 8)

    def test_rejections(self):
        for kwargs in (
            {"f_min": 100, "f_max": 100},
            {"f_max": 10, "levels": 1},
            {"f_max": 10, "action_mode": "guess"},
            {"f_max": 10, "span": 0},
            {"f_max": 10, "context": -1},
            {"f_max": 10, "learning_rate": 0.0},
            {"f_max": 10, "discount": 1.0},
            {"f_max": 10, "epsilon": 2.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(QLearnError):
                    QConfig(**kwargs)


class TestStrips(unittest.TestCase):
    def test_a_strip_is_a_column(self):
        plane = [[1, 2, 3], [4, 5, 6]]           # 2 rows (top first), 3 columns
        strips = plane_strips(plane)
        self.assertEqual(len(strips), 3)         # one strip per column
        self.assertEqual(len(strips[0]), 2)      # one pixel per node
        # row 0 is the top of the picture = the highest frequency, so it is last
        self.assertEqual(strips[0], [4, 1])

    def test_round_trip(self):
        plane = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        self.assertEqual(strips_plane(plane_strips(plane)), plane)

    def test_empty(self):
        self.assertEqual(plane_strips([]), [])
        self.assertEqual(strips_plane([]), [])

    def test_strip_height_matches_nodes(self):
        strips = strips_of()
        self.assertEqual(len(strips[0]), F_MAX)
        self.assertEqual(len(strips[0]), QConfig(f_min=0, f_max=F_MAX).nodes)


class TestNet(unittest.TestCase):
    def setUp(self):
        self.strips = strips_of()
        self.net = QNet(QConfig(f_min=0, f_max=F_MAX, levels=8, context=4))

    def test_quantising(self):
        for level in range(8):
            self.assertEqual(self.net.level_of(self.net.intensity_of(level)), level)
        self.assertEqual(self.net.level_of(0.0), 0)
        self.assertEqual(self.net.level_of(1.0), 7)
        self.assertEqual(self.net.level_of(-5.0), 0)

    def test_a_strip_must_be_the_right_height(self):
        with self.assertRaises(QLearnError):
            self.net.predict([0.5] * (F_MAX + 1))

    def test_warm_start_holds(self):
        """Untrained, every node should say "same as now" - not "go silent"."""
        strip = self.strips[10]
        predicted = self.net.predict(strip)
        for got, want in zip(predicted, strip):
            self.assertEqual(self.net.level_of(got), self.net.level_of(want))

    def test_warm_start_equals_the_persistence_baseline(self):
        untrained = self.net.evaluate(self.strips)
        baseline_exact = 0.0
        top = self.net.config.levels - 1
        baseline_error = 0.0
        for a, b in zip(self.strips, self.strips[1:]):
            now, nxt = self.net._levels(a), self.net._levels(b)
            baseline_exact += sum(1 for i in range(len(now)) if now[i] == nxt[i]) / len(now)
            baseline_error += sum(abs(now[i] - nxt[i]) for i in range(len(now))) / (len(now) * top)
        pairs = len(self.strips) - 1
        self.assertAlmostEqual(untrained["exact"], baseline_exact / pairs, places=9)
        self.assertAlmostEqual(untrained["error"], baseline_error / pairs, places=9)

    def test_training_runs_and_reports(self):
        history = self.net.train(self.strips, epochs=3)
        self.assertEqual(len(history), 3)
        for entry in history:
            for key in ("reward", "exact", "error", "epsilon", "epoch"):
                self.assertIn(key, entry)
        self.assertEqual(self.net.epochs, 3)
        self.assertGreater(self.net.trained_strips, 0)

    def test_exploration_decays(self):
        first = self.net.epsilon
        self.net.train(self.strips, epochs=3)
        self.assertLess(self.net.epsilon, first)
        self.assertGreaterEqual(self.net.epsilon, self.net.config.epsilon_min)

    def test_training_fills_the_table(self):
        before = self.net.visited()
        self.net.train(self.strips, epochs=2)
        self.assertGreater(self.net.visited(), before)

    def test_reward_improves_over_epochs(self):
        """It does learn - it just does not learn enough to beat holding still."""
        history = self.net.train(self.strips, epochs=6)
        self.assertGreater(history[-1]["reward"], history[0]["reward"])

    def test_deterministic(self):
        a = QNet(QConfig(f_min=0, f_max=F_MAX, levels=8, seed=7))
        b = QNet(QConfig(f_min=0, f_max=F_MAX, levels=8, seed=7))
        a.train(self.strips, epochs=2)
        b.train(self.strips, epochs=2)
        self.assertEqual(a.evaluate(self.strips), b.evaluate(self.strips))

    def test_training_needs_two_strips(self):
        with self.assertRaises(QLearnError):
            self.net.train(self.strips[:1], epochs=1)
        with self.assertRaises(QLearnError):
            self.net.train(self.strips, epochs=0)

    def test_stop_callback(self):
        calls = []
        self.net.train(self.strips, epochs=5, stop=lambda: len(calls) > 0 or calls.append(1))
        self.assertLess(self.net.epochs, 5)

    def test_predict_shape_and_range(self):
        out = self.net.predict(self.strips[0])
        self.assertEqual(len(out), F_MAX)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in out))

    def test_delta_actions_stay_in_range(self):
        """A node at the top or bottom must not walk off the end."""
        for strip in ([0.0] * F_MAX, [1.0] * F_MAX):
            out = self.net.predict(strip)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in out))

    def test_generate(self):
        self.net.train(self.strips, epochs=2)
        out = self.net.generate(self.strips[:4], steps=10)
        self.assertEqual(len(out), 14)
        self.assertTrue(all(len(s) == F_MAX for s in out))
        without = self.net.generate(self.strips[:4], steps=10, keep_seed=False)
        self.assertEqual(len(without), 10)

    def test_generate_rejections(self):
        with self.assertRaises(QLearnError):
            self.net.generate([], steps=4)
        with self.assertRaises(QLearnError):
            self.net.generate(self.strips[:2], steps=0)

    def test_temperature_changes_the_output(self):
        self.net.train(self.strips, epochs=2)
        cold = self.net.generate(self.strips[:3], steps=8, temperature=0.0)
        warm = self.net.generate(self.strips[:3], steps=8, temperature=0.5)
        self.assertNotEqual(cold, warm)

    def test_absolute_action_mode_works_too(self):
        net = QNet(QConfig(f_min=0, f_max=F_MAX, levels=8, action_mode="level"))
        net.train(self.strips, epochs=2)
        self.assertEqual(len(net.predict(self.strips[0])), F_MAX)

    def test_generated_plane_becomes_a_picture(self):
        """The output has to be a plane the codec will take."""
        from audioimage.codec import plane_to_image

        self.net.train(self.strips, epochs=1)
        plane = strips_plane(self.net.generate(self.strips[:4], steps=12))
        image = plane_to_image(plane, "gray", 16)
        self.assertEqual(image.height, F_MAX)
        self.assertEqual(image.width, 16)

    def test_describe(self):
        d = self.net.describe()
        self.assertEqual(d["nodes"], F_MAX)
        self.assertEqual(d["hz_per_node"], 1)
        self.assertEqual(d["range"], [0, F_MAX])
        self.assertIn(d["backend"], ("python", "numpy"))


class TestFiles(unittest.TestCase):
    def test_save_and_load(self):
        strips = strips_of()
        net = QNet(QConfig(f_min=0, f_max=F_MAX, levels=8))
        net.train(strips, epochs=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deep", "model.json.gz")
            net.save(path)
            self.assertTrue(os.path.exists(path))
            back = QNet.load(path)
            self.assertEqual(back.nodes, net.nodes)
            self.assertEqual(back.epochs, net.epochs)
            self.assertEqual(back.evaluate(strips), net.evaluate(strips))
            self.assertEqual(back.predict(strips[0]), net.predict(strips[0]))

    def test_rubbish_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json.gz")
            with open(path, "wb") as fh:
                fh.write(b"not a model")
            with self.assertRaises(QLearnError):
                QNet.load(path)

    def test_wrong_kind(self):
        import gzip
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json.gz")
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump({"kind": "something-else"}, fh)
            with self.assertRaises(QLearnError):
                QNet.load(path)


if __name__ == "__main__":
    unittest.main()
