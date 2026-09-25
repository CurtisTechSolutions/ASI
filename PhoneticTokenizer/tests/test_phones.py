"""Tests for phonetok.phones: the inventory, the ids, the features, the IPA."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok import phones as P  # noqa: E402


class TestInventory(unittest.TestCase):
    def test_thirty_nine_phonemes(self):
        self.assertEqual(len(P.PHONEMES), 39)
        self.assertEqual(len(P.VOWELS), 15)
        self.assertEqual(len(P.CONSONANTS), 24)
        self.assertEqual(set(P.VOWELS) | set(P.CONSONANTS), set(P.PHONEMES))

    def test_the_alphabet_is_fixed(self):
        """92 ids: four specials, the boundary, three pauses, then cmudict.symbols in its own order."""
        self.assertEqual(len(P.SYMBOLS), 92)
        self.assertEqual(P.SYMBOLS[:8], ("<pad>", "<unk>", "<s>", "</s>", "#", ",", ".", "?"))
        self.assertEqual(P.SYMBOLS[8:12], ("AA", "AA0", "AA1", "AA2"))
        self.assertEqual(P.SYMBOLS[-1], "ZH")
        self.assertEqual(P.SYMBOL_ID["K"], P.SYMBOLS.index("K"))
        self.assertEqual((P.PAD_ID, P.UNK_ID, P.BOS_ID, P.EOS_ID, P.BOUNDARY_ID), (0, 1, 2, 3, 4))
        self.assertEqual(len(set(P.SYMBOLS)), 92)

    def test_the_token_grammar(self):
        for token in ("K", "AH0", "AH1", "AH2", "AH", "ZH", "NG"):
            self.assertTrue(P.is_phone(token), token)
        for token in ("k", "AH3", "A", "I", "#", "", "K.AE1", "X"):
            self.assertFalse(P.is_phone(token), token)
        self.assertEqual(P.base("AH0"), "AH")
        self.assertEqual(P.base("K"), "K")
        self.assertEqual(P.stress_of("AH2"), 2)
        self.assertIsNone(P.stress_of("K"))
        self.assertTrue(P.is_vowel("IY1") and not P.is_vowel("Y"))
        self.assertEqual(P.strip_stress(["K", "AE1", "T"]), ["K", "AE", "T"])
        self.assertEqual(P.with_stress("AE", 1), "AE1")
        self.assertEqual(P.with_stress("K", 1), "K")
        with self.assertRaises(ValueError):
            P.check_phones(["K", "AE1", "X"])


class TestFeatures(unittest.TestCase):
    def test_every_phone_has_a_vector(self):
        for p in P.PHONEMES:
            vec = P.features(p)
            self.assertEqual(len(vec), P.FEATURE_DIM)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in vec), p)
        self.assertEqual(len(P.FEATURE_NAMES), P.FEATURE_DIM)

    def test_voicing_is_one_step(self):
        """P and B differ in voice (and the sonority that follows from it), nothing else."""
        diff = [i for i, (a, b) in enumerate(zip(P.features("P"), P.features("B"))) if a != b]
        self.assertEqual([P.FEATURE_NAMES[i] for i in diff], ["voice", "sonority"])

    def test_like_sounds_are_close(self):
        self.assertLess(P.distance("P", "B"), P.distance("P", "IY1"))
        self.assertLess(P.distance("IY1", "IH1"), P.distance("IY1", "AA1"))
        self.assertLess(P.distance("M", "N"), P.distance("M", "S"))
        self.assertEqual(P.distance("K", "K"), 0.0)
        self.assertEqual(P.similarity("K", "K"), 1.0)

    def test_stress_is_a_feature(self):
        i = P.FEATURE_NAMES.index("stress")
        self.assertEqual([P.features(v)[i] for v in ("AH0", "AH2", "AH1")], [0.0, 0.5, 1.0])

    def test_sonority_rises_to_the_vowel(self):
        self.assertLess(P.sonority("T"), P.sonority("S"))
        self.assertLess(P.sonority("S"), P.sonority("N"))
        self.assertLess(P.sonority("N"), P.sonority("L"))
        self.assertLess(P.sonority("L"), P.sonority("R"))
        self.assertLess(P.sonority("R"), P.sonority("W"))
        self.assertLess(P.sonority("W"), P.sonority("IY1"))
        self.assertLess(P.sonority("IY1"), P.sonority("AA1"))

    def test_an_unknown_phone_is_refused(self):
        with self.assertRaises(ValueError):
            P.features("X")


class TestIPA(unittest.TestCase):
    def test_to_ipa(self):
        self.assertEqual(P.to_ipa(["K", "AE1", "T"]), "kˈæt")
        self.assertEqual(P.to_ipa(["K", "AE1", "T"], stress=False), "kæt")
        self.assertEqual(P.to_ipa(["DH", "AH0"]), "ðə")
        self.assertEqual(P.to_ipa(["B", "ER0"]), "bɚ")
        self.assertEqual(P.to_ipa(["S", "T", "R", "EH1", "NG", "TH", "S"], stress=False), "stɹɛŋθs")

    def test_from_ipa_round_trips(self):
        for phones in (["K", "AE1", "T"], ["DH", "AH0"], ["S", "T", "R", "EH1", "NG", "TH", "S"], ["CH", "IY1", "Z"],
                       ["JH", "AH1", "JH"], ["OW1", "L", "D"], ["AY1"], ["AW1", "T"]):
            self.assertEqual(P.from_ipa(P.to_ipa(phones)), phones, phones)

    def test_from_ipa_gives_an_unmarked_vowel_no_stress(self):
        self.assertEqual(P.from_ipa("kæt"), ["K", "AE0", "T"])
        with self.assertRaises(ValueError):
            P.from_ipa("k@t")


class TestSuffixes(unittest.TestCase):
    def test_plural(self):
        self.assertEqual(P.plural_suffix(["K", "AE1", "T"]), ["S"])
        self.assertEqual(P.plural_suffix(["D", "AO1", "G"]), ["Z"])
        self.assertEqual(P.plural_suffix(["B", "AH1", "S"]), ["IH0", "Z"])
        self.assertEqual(P.plural_suffix(["D", "EY1"]), ["Z"])

    def test_past(self):
        self.assertEqual(P.past_suffix(["W", "AO1", "K"]), ["T"])
        self.assertEqual(P.past_suffix(["L", "AH1", "V"]), ["D"])
        self.assertEqual(P.past_suffix(["W", "EY1", "T"]), ["IH0", "D"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
