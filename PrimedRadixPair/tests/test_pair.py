"""The rungs: the fold is FilterBankRadix's, a reward lifts an unread step, the punishment traversal is unbuyable."""

import random
import unittest

from radixpair.codec import CharsCodec
from radixpair.pair import BACKOFFS, RadixPair, Settings, TRAVERSALS, softmax

ALPHA, FLOOR = 2.0, 0.02


def filterbank_fold(counts, emits, ctx, x):
    """FilterBankRadix/DESIGN.md section 5.3: shortest level first, p = own * p_here + (1 - own) * p."""
    p = 1.0 / len(emits)
    for k in range(len(ctx), -1, -1):
        s = tuple(ctx[k:])
        tot = sum(counts.get(s + (y,), 0) for y in emits)
        own = tot / (tot + ALPHA)
        here = counts.get(s + (x,), 0) / tot if tot else 0.0
        p = own * here + (1 - own) * p
    return (1 - FLOOR) * p + FLOOR / len(emits)


def naive_counts(codec, texts, L):
    counts = {}
    for text in texts:
        ids = codec.padded(text)
        for t in range(1, len(ids) + 1):
            for l in range(1, min(L, t) + 1):
                key = tuple(ids[t - l:t])
                counts[key] = counts.get(key, 0) + 1
    return counts


class TestTheFold(unittest.TestCase):
    def test_the_rungs_are_the_fold(self):
        rng = random.Random(7)
        codec = CharsCodec("abcd")
        for trial in range(20):
            L = rng.choice((2, 3, 4))
            pair = RadixPair(codec, L, Settings(smoothing=0.0))
            texts = ["".join(rng.choice("abcd") for _ in range(rng.randrange(1, 25))) for _ in range(rng.randrange(1, 12))]
            for t in texts:
                pair.count.observe_text(t)
            counts = naive_counts(codec, texts, L)
            for _ in range(10):
                ctx = [rng.choice(codec.padded("abcd")) for _ in range(rng.randrange(0, L))]
                P = pair.fold(ctx)
                self.assertAlmostEqual(sum(P), 1.0, places=9)
                for j, x in enumerate(pair.emits):
                    self.assertAlmostEqual(P[j], filterbank_fold(counts, pair.emits, ctx, x), places=9)

    def test_every_answer_is_a_distribution(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 3)
        for t in ("abcab", "dad", "cccc"):
            pair.count.observe_text(t)
        pair.reward.credit(codec.padded("abd"), 2.0)
        pair.reward.credit(codec.padded("dad"), -1.0)
        for traversal in TRAVERSALS:
            for backoff in BACKOFFS:
                for ctx in ([], codec.encode("a"), codec.encode("ab"), codec.encode("zz")):
                    P = pair.fold(ctx, traversal, backoff)
                    self.assertAlmostEqual(sum(P), 1.0, places=9)
                    self.assertTrue(all(p > 0 for p in P))

    def test_deepest_and_none(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 3)
        for t in ("abcab", "abd"):
            pair.count.observe_text(t)
        ctx = codec.encode("ab")
        i, l = pair.context_node(ctx)
        q = pair.q(i, l)
        own = pair.own(i, l)
        n = pair.R_out
        deepest = pair.fold(ctx, backoff="deepest")
        none = pair.fold(ctx, backoff="none")
        for j in range(n):
            self.assertAlmostEqual(deepest[j], (1 - FLOOR) * (own * q[j] + (1 - own) / n) + FLOOR / n)
            self.assertAlmostEqual(none[j], (1 - FLOOR) * q[j] + FLOOR / n)
        unread = pair.fold(codec.encode("dd"), backoff="none")
        self.assertAlmostEqual(unread[0], 1.0 / n)


class TestTheRung(unittest.TestCase):
    def test_a_reward_lifts_an_unread_step_with_smoothing_and_cannot_without(self):
        codec = CharsCodec("abcd")
        b = codec.encode("b")[0]
        for smoothing, lifts in ((0.5, True), (0.0, False)):
            pair = RadixPair(codec, 3, Settings(smoothing=smoothing))
            pair.count.observe_text("aaaa")
            ctx = codec.encode("aa")
            before = pair.fold(ctx)[pair.index[b]]
            pair.reward.credit(codec.padded("aab"), 3.0)
            after = pair.fold(ctx)[pair.index[b]]
            if lifts:
                self.assertGreater(after, before * 3)
            else:
                self.assertAlmostEqual(after, before)

    def test_the_punishment_traversal_ignores_rewards_and_is_unbuyable(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 3)
        for t in ("abab", "abac", "abad"):
            pair.count.observe_text(t)
        ctx = codec.encode("ab")
        before = pair.fold(ctx, traversal="punishment")
        pair.reward.credit(codec.padded("abab"), 50.0)
        self.assertEqual(pair.fold(ctx, traversal="punishment"), before)
        self.assertNotEqual(pair.fold(ctx, traversal="reward"), before)
        pair.reward.credit(codec.padded("abab"), -1.0)
        punished = pair.fold(ctx, traversal="punishment")
        a = pair.index[codec.encode("a")[0]]
        self.assertLess(punished[a], before[a])          # "ab" -> a was a step of the punished text
        pair.reward.credit(codec.padded("abab"), 500.0)
        self.assertEqual(pair.fold(ctx, traversal="punishment"), punished)

    def test_rungs_at_every_level_reach_a_sibling_context_and_final_does_not(self):
        codec = CharsCodec("abcd")
        b = codec.encode("b")[0]
        lifts = {}
        for rungs in ("all", "final"):
            pair = RadixPair(codec, 3, Settings(rungs=rungs))
            for t in ("aab", "cab", "cac", "aac"):
                pair.count.observe_text(t)
            sibling = codec.encode("ca")         # shares the step's last unit 'a', not its context 'aa'
            j = pair.index[b]
            before = pair.fold(sibling)[j]
            pair.reward.credit(codec.padded("aab"), 2.0, rungs)
            lifts[rungs] = pair.fold(sibling)[j] / before
        self.assertGreater(lifts["all"], 1.05)
        self.assertAlmostEqual(lifts["final"], 1.0)

    def test_the_cache_follows_both_versions(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 2)
        pair.count.observe_text("ab")
        ctx = codec.encode("a")
        first = pair.fold(ctx)
        pair.count.observe_text("ac")
        self.assertNotEqual(pair.fold(ctx), first)
        second = pair.fold(ctx)
        pair.reward.credit(codec.padded("ab"), 1.0)
        self.assertNotEqual(pair.fold(ctx), second)

    def test_scores_read_rewards_at_the_deepest_level_only_under_final(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 3, Settings(rungs="final"))
        pair.count.observe_text("aab")
        pair.reward.credit(codec.padded("aab"), 2.0, "all")     # written everywhere on purpose
        root_scores = pair.scores(0, 0)
        merit_only = [s for s in root_scores]
        pair2 = RadixPair(codec, 3, Settings(rungs="all"))
        pair2.count.observe_text("aab")
        self.assertEqual(merit_only, pair2.scores(0, 0))         # final: the root reads no reward


class TestSettingsAndScore(unittest.TestCase):
    def test_settings(self):
        s = Settings.from_dict({"smoothing": 0.0, "backoff": "none"})
        self.assertEqual(s.backoff, "none")
        with self.assertRaises(ValueError):
            Settings.from_dict({"rungs": "some"})
        with self.assertRaises(ValueError):
            Settings.from_dict({"floor": 1.5})
        with self.assertRaises(ValueError):
            Settings.from_dict({"nonsense": 1})
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 2, Settings(rungs="final"))
        with self.assertRaises(ValueError):
            pair.configure(rungs="all")
        pair.configure(smoothing=0.0, backoff="deepest")
        self.assertEqual(pair.settings.backoff, "deepest")
        self.assertEqual(pair.count.smoothing, 0.0)

    def test_softmax(self):
        self.assertEqual(softmax([float("-inf")] * 3), [1 / 3] * 3)
        p = softmax([0.0, float("-inf")])
        self.assertEqual(p, [1.0, 0.0])

    def test_score(self):
        codec = CharsCodec("abcd")
        pair = RadixPair(codec, 3)
        pair.count.observe_text("abcab")
        pair.reward.credit(codec.padded("abcab"), 1.0)
        pair.reward.credit(codec.padded("abd"), -2.0)
        s = pair.score("abcab")
        self.assertEqual(s.units, 5)
        self.assertEqual(len(s.per_unit), 6)
        self.assertGreater(s.mean_reward, 0)
        t = pair.score("abd")
        self.assertEqual(t.worst_penalty, 2.0)
        self.assertGreater(t.bits, s.bits)
        with self.assertRaises(ValueError):
            pair.fold([], traversal="sideways")


if __name__ == "__main__":
    unittest.main()
