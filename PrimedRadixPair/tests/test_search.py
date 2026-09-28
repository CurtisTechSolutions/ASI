"""Walks over the pair: the cheapest path, the shift, the fall, the greedy and sampled walks."""

import math
import random
import unittest

from radixpair.codec import CharsCodec
from radixpair.pair import RadixPair, Settings, TRAVERSALS
from radixpair.search import MODES, dijkstra, greedy


def unfloored(pair, ctx, j):
    f = pair.settings.floor
    n = pair.R_out
    return (pair.fold(ctx)[j] - f / n) / (1 - f)


class TestDijkstra(unittest.TestCase):
    def setUp(self):
        self.codec = CharsCodec()
        self.pair = RadixPair(self.codec, 4)
        for _ in range(3):
            self.pair.count.observe_text("the cat sat on the mat")
        self.pair.count.observe_text("a dog ran")

    def test_returns_the_trained_continuation(self):
        ctx = [self.codec.start] + self.codec.encode("the cat sat o")
        r = dijkstra(self.pair, ctx, 3)
        self.assertEqual(r.text, "n t")
        self.assertEqual(len(r.units), 3)
        self.assertFalse(r.fallback)
        self.assertEqual(r.mode, "dijkstra")
        self.assertEqual(len(r.symbols), 3)

    def test_the_cost_decomposes_and_bounds_the_fold(self):
        ctx = [self.codec.start] + self.codec.encode("the cat")
        r = dijkstra(self.pair, ctx, 5)
        self.assertAlmostEqual(r.cost, sum(r.step_costs), places=9)
        walk = list(ctx)
        fold_cost = 0.0
        for x in r.units:
            fold_cost += -math.log(unfloored(self.pair, walk[-self.pair.D:], self.pair.index[x]))
            walk.append(x)
        self.assertGreaterEqual(r.cost + 1e-9, fold_cost)

    def test_the_shift_is_free_and_forced(self):
        pair = RadixPair(CharsCodec("abcd"), 3)
        for _ in range(3):
            pair.count.observe_text("abcabcabc")
        ctx = pair.codec.encode("ab")                  # a full context: the first emission lands at level L
        r = dijkstra(pair, ctx, 3)
        kinds = [k for k, _ in r.hops]
        self.assertEqual(kinds[0], "emit")
        self.assertIn("shift", kinds)
        self.assertEqual(r.text, "cab")

    def test_the_fallback_on_the_expansion_cap_never_raises(self):
        ctx = [self.codec.start] + self.codec.encode("the")
        r = dijkstra(self.pair, ctx, 6, max_expansions=1)
        self.assertTrue(r.fallback)
        self.assertLessEqual(len(r.units), 6)

    def test_to_end(self):
        pair = RadixPair(CharsCodec("abcd"), 3)
        for _ in range(5):
            pair.count.observe_text("ab")
        r = dijkstra(pair, [pair.codec.start], 0, to_end=True)
        self.assertTrue(r.reached_end)
        self.assertEqual(r.text, "")          # the cheapest complete text is one step long
        g = greedy(pair, [pair.codec.start], 10, to_end=True)
        self.assertTrue(g.reached_end)
        self.assertEqual(g.text, "ab")

    def test_both_traversals_and_an_empty_model(self):
        for traversal in TRAVERSALS:
            r = dijkstra(self.pair, [self.codec.start], 3, traversal=traversal)
            self.assertEqual(r.traversal, traversal)
            self.assertEqual(len(r.units), 3)
        empty = RadixPair(CharsCodec("abcd"), 3)
        r = dijkstra(empty, [], 3)
        self.assertEqual(len(r.units), 3)
        self.assertTrue(math.isfinite(r.cost))
        with self.assertRaises(ValueError):
            dijkstra(self.pair, [], 3, traversal="sideways")


class TestGreedy(unittest.TestCase):
    def setUp(self):
        self.codec = CharsCodec()
        self.pair = RadixPair(self.codec, 4)
        for _ in range(3):
            self.pair.count.observe_text("the cat sat on the mat")

    def test_greedy_is_the_argmax_of_the_fold(self):
        ctx = [self.codec.start] + self.codec.encode("the cat")
        r = greedy(self.pair, ctx, 4)
        walk = list(ctx)
        for x in r.units:
            P = self.pair.fold(walk[-self.pair.D:])
            best = max(range(len(P)), key=lambda k: (P[k], -self.pair.emits[k]))
            self.assertEqual(x, self.pair.emits[best])
            walk.append(x)
        self.assertEqual(r.mode, "greedy")
        self.assertAlmostEqual(r.cost, sum(r.step_costs))

    def test_sampling_is_seeded_and_terminates(self):
        ctx = [self.codec.start]
        a = greedy(self.pair, ctx, 12, rng=random.Random(5), temperature=1.0)
        b = greedy(self.pair, ctx, 12, rng=random.Random(5), temperature=1.0)
        self.assertEqual(a.units, b.units)
        self.assertEqual(a.mode, "sample")
        self.assertLessEqual(len(a.units), 12)
        c = greedy(self.pair, ctx, 0, rng=random.Random(1), temperature=1.0, to_end=True)
        self.assertTrue(c.reached_end or len(c.units) > 0)
        with self.assertRaises(ValueError):
            greedy(self.pair, ctx, 3, temperature=-1.0)

    def test_modes(self):
        self.assertEqual(MODES, ("dijkstra", "greedy", "sample"))


if __name__ == "__main__":
    unittest.main()
