"""Tests for phonetok.g2p: the transcriber and the respelling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.g2p import Transcriber, respell  # noqa: E402
from phonetok.lexicon import Lexicon  # noqa: E402


class TestTranscriber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Transcriber(Lexicon.core())

    def how(self, word):
        """Where a fresh transcriber gets the word from (a second reading would say ``memory``)."""
        return Transcriber(Lexicon.core()).explain(word)[1]

    def test_the_lexicon_first(self):
        self.assertEqual(self.t.word("the"), ["DH", "AH0"])
        self.assertEqual(self.t.word("The"), ["DH", "AH0"])
        self.assertEqual(self.t.word("cat"), ["K", "AE1", "T"])
        self.assertEqual(self.t.word('"cat"'), ["K", "AE1", "T"])
        self.assertEqual(self.how("cat"), "lexicon")
        self.assertEqual(self.t.word(""), [])
        self.assertEqual(self.t.word("..."), [])

    def test_morphology(self):
        t = self.t
        self.assertEqual(t.word("cats"), ["K", "AE1", "T", "S"])
        self.assertEqual(t.word("dogs"), ["D", "AO1", "G", "Z"])
        self.assertEqual(t.word("boxes"), ["B", "AA1", "K", "S", "IH0", "Z"])
        self.assertEqual(t.word("cities"), ["S", "IH1", "T", "IY0", "Z"])
        self.assertEqual(t.word("bigger"), ["B", "IH1", "G", "ER0"])
        self.assertEqual(t.word("happiest"), ["HH", "AE1", "P", "IY0", "IH0", "S", "T"])
        self.assertEqual(t.word("quickly"), ["K", "W", "IH1", "K", "L", "IY0"])
        self.assertEqual(t.word("unhappy"), ["AH0", "N", "HH", "AE1", "P", "IY0"])
        self.assertEqual(t.word("toothbrush"), ["T", "UW1", "TH", "B", "R", "AH2", "SH"])
        self.assertEqual(t.word("keyboard"), ["K", "IY1", "B", "AO2", "R", "D"])
        self.assertEqual(self.how("cats"), "morphology")
        self.assertEqual(self.how("toothbrush"), "morphology")

    def test_contractions(self):
        t = self.t
        self.assertEqual(t.word("cat's"), ["K", "AE1", "T", "S"])
        self.assertEqual(t.word("dog's"), ["D", "AO1", "G", "Z"])
        self.assertEqual(t.word("dogs'"), ["D", "AO1", "G", "Z"])
        self.assertEqual(t.word("fox's"), ["F", "AA1", "K", "S", "IH0", "Z"])
        self.assertEqual(t.word("it'll"), ["IH1", "T", "AH0", "L"])
        self.assertEqual(t.word("would've"), ["W", "UH1", "D", "AH0", "V"])
        self.assertEqual(t.word("they'd"), ["DH", "EY1", "D"])
        self.assertEqual(t.word("I'm"), ["AY1", "M"])

    def test_numbers_and_joins(self):
        t = self.t
        self.assertEqual(t.word("42"), ["F", "AO1", "R", "T", "IY0", "T", "UW1"])
        self.assertEqual(t.word("3.5"), ["TH", "R", "IY1", "P", "OY1", "N", "T", "F", "AY1", "V"])
        self.assertEqual(t.word("mp3"), ["EH1", "M", "P", "IY1", "TH", "R", "IY1"])
        self.assertEqual(t.word("well-known"), t.word("well") + t.word("known"))
        self.assertEqual(t.word("u.s."), ["Y", "UW1", "EH1", "S"])
        self.assertEqual(self.how("42"), "number")
        self.assertEqual(self.how("well-known"), "joined")

    def test_letters(self):
        t = self.t
        self.assertEqual(t.word("FBI"), ["EH1", "F", "B", "IY1", "AY1"])
        self.assertEqual(t.word("xq"), ["EH1", "K", "S", "K", "Y", "UW1"])
        self.assertEqual(t.word("b"), ["B", "IY1"])
        self.assertEqual(t.word("a"), ["AH0"])  # the article, as the lexicon lists it first
        self.assertEqual(self.how("FBI"), "letters")
        self.assertEqual(self.how("nasa"), "rules")  # lower case: a word, not an acronym

    def test_rules_last_and_remembered(self):
        t = Transcriber(Lexicon.core())
        phones, how = t.explain("zebra")
        self.assertEqual(how, "rules")
        self.assertEqual(phones, ["Z", "EH1", "B", "R", "AH0"])
        self.assertEqual(t.explain("zebra")[1], "memory")
        self.assertEqual(t.memory["zebra"], tuple(phones))
        self.assertEqual(t.counts["zebra"], 2)
        no_rules = Transcriber(Lexicon.core(), rules=False)
        self.assertEqual(no_rules.explain("zebra"), ([], "none"))
        self.assertEqual(no_rules.explain("cat")[1], "lexicon")

    def test_spell(self):
        t = Transcriber(Lexicon.core())
        self.assertEqual(t.spell(["K", "AE1", "T"]), "cat")
        self.assertEqual(t.spell(["T", "UW1"]), "to")  # the shortest homophone, then the alphabetical
        t.word("two")
        t.word("two")
        self.assertEqual(t.spell(["T", "UW1"]), "two")  # the one read most
        self.assertEqual(t.spell(["T", "UW0"]), "two")  # stress ignored when nothing matches exactly
        t.word("zebra")
        self.assertEqual(t.spell(["Z", "EH1", "B", "R", "AH0"]), "zebra")  # remembered
        self.assertEqual(t.spell(["Z", "IH1", "B", "R", "AH0"]), "zibra")  # respelled
        self.assertEqual(t.spell([]), "")


class TestRespell(unittest.TestCase):
    def test_examples(self):
        cases = {
            "K AE1 T": "cat", "M EY1 K": "make", "S T R EH1 NG TH": "strength", "HH AE1 P IY0": "happy",
            "B AH1 T ER0": "butter", "T AY1 M": "time", "G OW1": "go", "S IY1": "see", "P EY1 P ER0": "paper",
            "K Y UW1 T": "cute", "B UH1 K": "book", "S K AY1": "sky", "N AY1 T": "nite", "AH0 B AW1 T": "abowt",
            "JH AH1 M P": "jump", "K IY1 P": "keep", "B AE1 K": "back", "M Y UW1 Z IH0 K": "muzik",
            "F L AW1 ER0": "flower", "DH AH0": "tha", "": "",
        }
        for phones, spelling in cases.items():
            self.assertEqual(respell(phones.split()), spelling, phones)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
