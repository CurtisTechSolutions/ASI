"""The reward tree: written by outcomes alone; a punishment is a negative reward; the two sums kept apart."""

import unittest

from radixpair.codec import CharsCodec
from radixpair.count import CountTree
from radixpair.reward import RewardTree


class TestRewardTree(unittest.TestCase):
    def setUp(self):
        self.codec = CharsCodec("abcd")
        self.tree = RewardTree(self.codec, 3)
        self.ids = self.codec.padded("abcab")

    def test_credit_writes_the_ids_observe_counts(self):
        count = CountTree(self.codec, 3, self.tree.address)
        count.observe(self.ids)
        n = self.tree.credit(self.ids, 1.5)
        self.assertEqual(n, sum(min(3, t) for t in range(1, len(self.ids) + 1)))
        for i in range(self.tree.N):
            self.assertAlmostEqual(self.tree.plus[i], 1.5 * count.cnt[i])
            self.assertEqual(self.tree.minus[i], 0.0)

    def test_final_credits_the_deepest_level_only(self):
        n = self.tree.credit(self.ids, 1.0, rungs="final")
        self.assertEqual(n, len(self.ids) - 2)
        a = self.tree.address
        for i, p, m in self.tree.nonzero():
            self.assertEqual(a.level(i), 3)
        with self.assertRaises(ValueError):
            self.tree.credit(self.ids, 1.0, rungs="some")

    def test_a_punishment_is_a_negative_reward_and_the_sums_stay_apart(self):
        self.tree.credit(self.ids, 2.0)
        self.tree.credit(self.ids, -2.0)
        i = self.tree.address.of(self.codec.encode("abc"))
        self.assertEqual(self.tree.reward(i), 0.0)
        self.assertEqual(self.tree.penalty(i), 2.0)
        self.assertEqual(self.tree.credit(self.ids, 0.0), 0)
        self.assertEqual(self.tree.judged, 2)
        self.assertGreater(self.tree.rewards_total, 0)
        self.assertEqual(self.tree.rewards_total, self.tree.penalties_total)

    def test_invert_twice_is_identity(self):
        self.tree.credit(self.ids, 3.0)
        self.tree.credit(self.codec.padded("dd"), -1.0)
        plus, minus = list(self.tree.plus), list(self.tree.minus)
        self.tree.invert()
        self.assertEqual(list(self.tree.plus), minus)
        self.assertEqual(list(self.tree.minus), plus)
        self.tree.invert()
        self.assertEqual((list(self.tree.plus), list(self.tree.minus)), (plus, minus))

    def test_rewards_and_penalties_over_a_block(self):
        self.tree.credit(self.ids, 1.0)
        i = self.tree.address.of(self.codec.encode("ab"))
        rewards = self.tree.rewards(i, 2)
        j = self.codec.emits().index(self.codec.encode("c")[0])
        self.assertEqual(rewards[j], 1.0)
        self.assertEqual(sum(1 for r in rewards if r), 2)     # "ab" -> c, and the final "ab" -> </s>
        self.assertEqual(rewards[self.codec.emits().index(self.codec.end)], 1.0)
        self.assertEqual(self.tree.penalties(i, 2), [0.0] * len(rewards))

    def test_skip_credits_the_outcome_and_not_the_prefix(self):
        ids = self.codec.padded("abcd")
        n = self.tree.credit(ids, 1.0, skip=3)            # <s> a b are context; c d </s> are credited
        self.assertEqual(n, 3 * 3)
        a = self.tree.address
        self.assertEqual(self.tree.plus[a.of(self.codec.encode("ab"))], 0.0)
        self.assertEqual(self.tree.plus[a.of(self.codec.encode("abc"))], 1.0)
        self.assertEqual(self.tree.plus[a.of(self.codec.encode("c"))], 1.0)
        m = self.tree.credit(ids, 1.0, rungs="final", skip=3)
        self.assertEqual(m, 3)

    def test_version_and_nonzero(self):
        v = self.tree.version
        self.tree.credit(self.ids, 1.0)
        self.assertEqual(self.tree.version, v + 1)
        self.assertTrue(all(p >= 0 and m >= 0 for _, p, m in self.tree.nonzero()))
        self.assertEqual(self.tree.memory_bytes(), 16 * self.tree.N)


if __name__ == "__main__":
    unittest.main()
