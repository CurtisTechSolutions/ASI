"""The benches run and say what the design says they measure."""

import unittest

from radixpair import bench

CORPUS = ["the cat sat on the mat", "the cat sat on the log", "the dog ate the bone", "a cat and a dog",
          "the bird sang on the wire", "a dog and a cat sat", "the mat was on the floor", "the log was on the fire"]


class TestBench(unittest.TestCase):
    def test_throughput(self):
        r = bench.throughput("chars", 3, units=5000)
        self.assertEqual(r["count"]["increments"], sum(min(3, t) for t in range(1, 5003)))
        self.assertGreater(r["count"]["per_second"], 0)
        self.assertGreater(r["credit"]["per_second"], 0)

    def test_compare_reports_the_three_backoffs(self):
        r = bench.compare(CORPUS, "chars", 3, holdout=0.25, seed=1)
        self.assertEqual(set(r["bits"]), {"all", "deepest", "none"})
        self.assertGreater(r["bits"]["none"]["held"], 0)
        self.assertEqual(r["train_texts"] + r["held_texts"], len(CORPUS))
        s = bench.compare(CORPUS, "chars", 3, holdout=0.25, seed=1, smoothing=0.0)
        self.assertEqual(s["smoothing"], 0.0)

    def test_feedback_is_the_authors_case(self):
        r = bench.feedback("chars", 4)
        self.assertEqual(r["smoothing"], 0.0)
        self.assertEqual(r["mat rewarded 5, punished 1"]["reward"]["continues"], "mat")
        self.assertEqual(r["mat rewarded 5, punished 1"]["punishment"]["continues"], "log")
        self.assertTrue(r["unbuyable"])

    def test_rungs_reach_a_sibling_only_at_every_level(self):
        r = bench.rungs(CORPUS, "chars", 3, samples=6, seed=2)
        self.assertGreaterEqual(r["samples"], 1)
        self.assertGreater(r["sibling_lift_bits"]["all"], 0.0)
        self.assertAlmostEqual(r["sibling_lift_bits"]["final"], 0.0)
        two = bench.rungs(CORPUS, "chars", 2, samples=4, seed=2)      # one unit of context: the root is shared
        self.assertGreater(two["sibling_lift_bits"]["all"], 0.0)
        self.assertAlmostEqual(two["sibling_lift_bits"]["final"], 0.0)
        with self.assertRaises(ValueError):
            bench.rungs(CORPUS, "chars", 1)

    def test_split(self):
        train, held = bench.split(CORPUS, 0.25, seed=0)
        self.assertEqual(len(train) + len(held), len(CORPUS))
        self.assertEqual(bench.split(["one"], 0.5), (["one"], ["one"]))


if __name__ == "__main__":
    unittest.main()
