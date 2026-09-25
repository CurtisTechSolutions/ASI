"""Tests for phonetok.syllables."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.syllables import (  # noqa: E402
    ONSETS, Syllable, alliterates, is_legal_coda, is_legal_onset, rhymes, stressed_syllable, syllabify,
    syllable_count,
)


def breaks(phones):
    return " | ".join(s.text() for s in syllabify(phones.split()))


class TestOnsets(unittest.TestCase):
    def test_the_list(self):
        self.assertEqual(len(ONSETS), 77)
        self.assertTrue(is_legal_onset(()))
        self.assertTrue(is_legal_onset(("S", "T", "R")))
        self.assertTrue(is_legal_onset(("SH", "W")))
        self.assertFalse(is_legal_onset(("NG",)))
        self.assertFalse(is_legal_onset(("T", "L")))
        self.assertFalse(is_legal_onset(("S", "T", "R", "Y")))

    def test_codas(self):
        for coda in ("", "K S TH S", "L M", "N T S", "R L D", "M P", "S P", "N JH", "K T", "L F TH S"):
            self.assertTrue(is_legal_coda(coda.split()), coda)
        for coda in ("T L", "HH", "K W", "M B L", "S T R L K"):
            self.assertFalse(is_legal_coda(coda.split()), coda)


class TestSyllabify(unittest.TestCase):
    def test_maximal_onset(self):
        self.assertEqual(breaks("S T R EH1 NG TH S"), "S.T.R.EH1.NG.TH.S")
        self.assertEqual(breaks("AE1 S T R OW0"), "AE1.S | T.R.OW0")
        self.assertEqual(breaks("EH1 K S T R AH0"), "EH1.K | S.T.R.AH0")
        self.assertEqual(breaks("K AH0 M P Y UW1 T ER0"), "K.AH0.M | P.Y.UW1 | T.ER0")
        self.assertEqual(breaks("P EY1 P ER0"), "P.EY1 | P.ER0")
        self.assertEqual(breaks("AH0 B AW1 T"), "AH0 | B.AW1.T")

    def test_a_stressed_lax_vowel_closes(self):
        self.assertEqual(breaks("B AH1 T ER0"), "B.AH1.T | ER0")
        self.assertEqual(breaks("HH AE1 P IY0"), "HH.AE1.P | IY0")
        self.assertEqual(breaks("B AH0 N AE1 N AH0"), "B.AH0 | N.AE1.N | AH0")
        self.assertEqual(breaks("S IH1 NG IH0 NG"), "S.IH1.NG | IH0.NG")
        self.assertEqual(" | ".join(s.text() for s in syllabify("B AH1 T ER0".split(), close_lax=False)), "B.AH1 | T.ER0")

    def test_no_vowel(self):
        self.assertEqual(breaks("HH M"), "HH.M")
        self.assertEqual(syllabify([]), [])
        self.assertEqual(syllable_count([]), 0)
        self.assertEqual(syllable_count(["HH", "M"]), 1)
        self.assertEqual(syllable_count("K AH0 M P Y UW1 T ER0".split()), 3)

    def test_the_syllable(self):
        s = syllabify("S T R EH1 NG TH S".split())[0]
        self.assertEqual((s.onset, s.nucleus, s.coda), (("S", "T", "R"), "EH1", ("NG", "TH", "S")))
        self.assertEqual(s.rime, ("EH1", "NG", "TH", "S"))
        self.assertEqual(s.stress, 1)
        self.assertFalse(s.open)
        self.assertEqual(str(s), "S.T.R.EH1.NG.TH.S")
        self.assertEqual(Syllable.parse("S.T.R.EH1.NG.TH.S"), s)
        self.assertEqual(stressed_syllable(syllabify("K AH0 M P Y UW1 T ER0".split())), 1)
        self.assertIsNone(stressed_syllable([]))


class TestRhyme(unittest.TestCase):
    def test_rhymes(self):
        self.assertTrue(rhymes("K AE1 T".split(), "HH AE1 T".split()))
        self.assertTrue(rhymes("N EY1 SH AH0 N".split(), "S T EY1 SH AH0 N".split()))
        self.assertFalse(rhymes("K AE1 T".split(), "K AH1 T".split()))
        self.assertFalse(rhymes("K AE1 T".split(), "K AE1 P".split()))
        self.assertFalse(rhymes([], "K AE1 T".split()))

    def test_alliterates(self):
        self.assertTrue(alliterates("S T R EH1 NG TH".split(), "S T R AO1 NG".split()))
        self.assertTrue(alliterates("B IH1 G".split(), "B AE1 D".split()))
        self.assertFalse(alliterates("B IH1 G".split(), "P IH1 G".split()))
        self.assertFalse(alliterates("AE1 T".split(), "AE1 D".split()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
