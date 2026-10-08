"""The count tree: counts are substring counts, written by reading alone."""

import unittest

from radixpair.codec import CharsCodec
from radixpair.count import CountTree
from radixpair.reward import RewardTree


def naive_counts(codec, texts, L):
    counts = {}
    for text in texts:
        ids = codec.padded(text)
        for t in range(1, len(ids) + 1):
            for l in range(1, min(L, t) + 1):
                key = tuple(ids[t - l:t])
                counts[key] = counts.get(key, 0) + 1
    return counts


class TestCountTree(unittest.TestCase):
    def setUp(self):
        self.codec = CharsCodec("abcd")
        self.texts = ["abcab", "bbc", "a", "dadab", "cccc"]
        self.tree = CountTree(self.codec, 3)
        for t in self.texts:
            self.tree.observe_text(t)

    def test_counts_are_substring_counts(self):
        naive = naive_counts(self.codec, self.texts, 3)
        for i in range(self.tree.N):
            self.assertEqual(self.tree.cnt[i], naive.get(self.tree.address.seq(i), 0), self.tree.address.seq(i))

    def test_the_context_count_is_the_node_count_away_from_the_marks(self):
        a = self.tree.address
        for i in range(a.bases[3]):
            l = a.level(i)
            seq = a.seq(i)
            if 1 <= l and seq[0] != self.codec.start and seq[-1] != self.codec.end:
                self.assertEqual(self.tree.ctx(i, l), self.tree.cnt[i], seq)

    def test_a_mark_inside_a_sequence_stays_at_zero(self):
        a = self.tree.address
        s, e = self.codec.start, self.codec.end
        for seq in ((3, s), (3, s, 4), (e, 3), (3, e, 4), (s, s)):
            self.assertEqual(self.tree.cnt[a.of(seq)], 0, seq)

    def test_observe_is_additive_and_bumps_the_version_once(self):
        tree = CountTree(self.codec, 3)
        v = tree.version
        n = tree.observe_text("abab")
        self.assertEqual(tree.version, v + 1)
        before = list(tree.cnt)
        tree.observe_text("abab")
        self.assertEqual([2 * c for c in before], list(tree.cnt))
        self.assertEqual(n, sum(min(3, t) for t in range(1, 7)))
        self.assertEqual(tree.texts, 2)
        self.assertEqual(tree.units, 8)

    def test_nothing_but_observe_writes_the_counts(self):
        reward = RewardTree(self.codec, 3, self.tree.address)
        before = list(self.tree.cnt)
        reward.credit(self.codec.padded("abcab"), 2.0)
        reward.credit(self.codec.padded("abcab"), -1.0)
        self.assertEqual(before, list(self.tree.cnt))

    def test_share_own_and_smoothing(self):
        a = self.tree.address
        i, l = a.of(self.codec.encode("ab")), 2
        kids = self.tree.children(i, l)
        self.assertEqual(sum(kids), self.tree.ctx(i, l))
        self.assertAlmostEqual(sum(self.tree.share(i, l)), 1.0)
        self.assertAlmostEqual(self.tree.own(i, l), self.tree.ctx(i, l) / (self.tree.ctx(i, l) + 2.0))
        self.tree.smoothing = 0.0
        unread = a.of(self.codec.encode("dd"))
        self.assertEqual(self.tree.share(unread, 2), [0.0] * len(self.codec.emits()))
        self.assertEqual(self.tree.own(unread, 2), 0.0)

    def test_the_ceiling_is_refused_with_its_numbers(self):
        with self.assertRaises(ValueError) as cm:
            CountTree(CharsCodec(), 4, node_ceiling=1000)
        self.assertIn("1,222,981", str(cm.exception))
        self.assertIn("1,000", str(cm.exception))
        CountTree(CharsCodec(), 4, node_ceiling=None)

    def test_unk_is_tallied(self):
        tree = CountTree(CharsCodec("ab"), 2)
        tree.observe_text("abz")
        self.assertEqual(tree.unk, 1)
        self.assertEqual(tree.units, 3)


if __name__ == "__main__":
    unittest.main()
