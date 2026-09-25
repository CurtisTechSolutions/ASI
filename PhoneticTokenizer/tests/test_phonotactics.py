"""Tests for phonetok.phonotactics."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.lexicon import Lexicon  # noqa: E402
from phonetok.phones import strip_stress  # noqa: E402
from phonetok.phonotactics import Phonotactics  # noqa: E402


class TestHabits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon.core()
        cls.p = Phonotactics.from_lexicon(cls.lex)

    def test_fit(self):
        self.assertEqual(self.p.words, sum(1 for _ in self.lex.items()))
        self.assertGreater(len(self.p.bigrams), 500)
        self.assertGreater(self.p.onsets_initial[("S", "T")], 0)
        self.assertGreater(self.p.codas_final[("T",)], 0)
        self.assertEqual(self.p.syllable_counts[1] + self.p.syllable_counts[2] > 0, True)

    def test_affinity(self):
        p = self.p
        self.assertGreater(p.affinity("S", "T"), 1.0)
        self.assertGreater(p.affinity("K", "W"), 1.0)
        self.assertLess(p.affinity("T", "L"), -1.0)
        self.assertLess(p.affinity("#", "NG"), -3.0)
        self.assertGreater(p.affinity("NG", "#"), 0.0)
        self.assertEqual(p.affinity("S", "T"), p.affinity("S", "T"))
        self.assertAlmostEqual(p.affinity("AE1", "T"), p.affinity("AE", "T"))  # stress ignored by default

    def test_probabilities_sum_to_one(self):
        p = self.p
        total = sum(p.prob("S", b) for b in p.unigrams)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertGreater(p.prob("S", "T"), p.prob("S", "L"))
        self.assertEqual(p.next("#", 3)[0][0], "S")
        self.assertEqual(p.next("QQ"), [])

    def test_score_prefers_english(self):
        p = self.p
        self.assertGreater(p.score("S T R EH1 NG TH".split()), p.score("NG S T R EH1 L K".split()))
        self.assertGreater(p.score("K AE1 T".split()), p.score("K T AE1".split()))
        self.assertEqual(p.score([]), p.log_prob("#", "#"))
        i, lp = p.weakest("NG S T R EH1 L K".split())
        self.assertEqual(i, 0)  # the # -> NG step

    def test_well_formed(self):
        p = self.p
        self.assertTrue(p.well_formed("S T R EH1 NG TH S".split()))
        self.assertTrue(p.well_formed("K AH0 M P Y UW1 T ER0".split()))
        self.assertFalse(p.well_formed("NG AE1 T".split()))
        self.assertFalse(p.well_formed("T L AE1 P".split()))
        self.assertFalse(p.well_formed("S T R".split()))
        self.assertFalse(p.well_formed([]))
        self.assertEqual(p.violations("T L AE1 P".split()), ["T L cannot start a syllable"])
        self.assertEqual(p.violations([]), ["empty"])
        self.assertEqual(p.violations("K AE1 T".split()), [])


class TestBuilding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon.core()
        cls.p = Phonotactics.from_lexicon(cls.lex)

    def test_build_is_seeded_and_well_formed(self):
        p = self.p
        a = [p.build(random.Random(i)) for i in range(20)]
        b = [p.build(random.Random(i)) for i in range(20)]
        self.assertEqual(a, b)
        for phones in a:
            self.assertTrue(p.well_formed(phones), phones)
            self.assertEqual(sum(x.endswith("1") for x in phones), 1, phones)
        self.assertGreater(len(set(map(tuple, a))), 10)

    def test_build_avoids_the_lexicon(self):
        p = self.p
        known = {tuple(strip_stress(ph)) for _, ph in self.lex.items()}
        for i in range(30):
            phones = p.build(random.Random(i), avoid=(ph for _, ph in self.lex.items()))
            self.assertNotIn(tuple(strip_stress(phones)), known)

    def test_build_syllable_count(self):
        from phonetok.syllables import syllable_count

        for n in (1, 2, 3):
            phones = self.p.build(random.Random(5), syllables=n)
            self.assertEqual(syllable_count(phones), n)

    def test_unfitted_refuses(self):
        with self.assertRaises(ValueError):
            Phonotactics().build(random.Random(1))

    def test_blend(self):
        p = self.p
        smog, _, _ = p.blend("S M OW1 K".split(), "F AA1 G".split())
        self.assertEqual(smog, ["S", "M", "AA1", "G"])
        brunch, _, _ = p.blend("B R EH1 K F AH0 S T".split(), "L AH1 N CH".split())
        self.assertEqual(brunch, ["B", "R", "AH1", "N", "CH"])
        with self.assertRaises(ValueError):
            p.blend(["S"], ["K", "AE1", "T"])

    def test_round_trip_through_dict(self):
        p = self.p
        q = Phonotactics.from_dict(p.to_dict())
        self.assertEqual(q.words, p.words)
        self.assertEqual(q.bigrams, p.bigrams)
        self.assertEqual(q.unigrams, p.unigrams)
        self.assertAlmostEqual(q.affinity("S", "T"), p.affinity("S", "T"))
        self.assertEqual(q.build(random.Random(3)), p.build(random.Random(3)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
