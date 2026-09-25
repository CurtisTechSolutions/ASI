"""The phonetic units of the encoding dial: ``phone`` and ``syllable`` (``../PhoneticTokenizer``).

:mod:`tests.test_encoding` and :mod:`tests.test_encodings_end_to_end` drive the
two units through the encoder and the whole stack beside every other encoding;
this module pins what is theirs alone - the text read as sounds, a label cut
into the units it was made of, a prefix matched by its sounds, a prediction
spelled back into words - and is skipped when the tokenizer is not importable.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer"))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import (  # noqa: E402
    CHARS, PHONES, SYLLABLES, WORDS, Encoding, parse_encoding, phonetic_tokenizer, word_rows,
)

try:
    import phonetok  # noqa: F401
except ImportError:  # pragma: no cover
    phonetok = None

TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "a quick brown fox jumps over the lazy dog",
]


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestTheUnits(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(parse_encoding("phone"), Encoding(unit=PHONES))
        self.assertEqual(parse_encoding("phones:2:2"), Encoding(unit=PHONES, n=2, stride=2))
        self.assertEqual(parse_encoding("sound:4"), Encoding(unit=PHONES, n=4))
        self.assertEqual(parse_encoding("syllable:2:1"), Encoding(unit=SYLLABLES, n=2))
        self.assertEqual(parse_encoding("syllables"), Encoding(unit=SYLLABLES))
        self.assertEqual(str(Encoding(unit=PHONES)), "phone:3:1")
        self.assertEqual(Encoding(unit=PHONES).describe(), "3-phone grams, stride 1 (sliding)")
        self.assertEqual(Encoding(unit=SYLLABLES, n=2).describe(), "2-syllable grams, stride 1 (sliding)")
        self.assertEqual((Encoding(unit=PHONES).units_name, Encoding(unit=SYLLABLES).units_name), ("phones", "syllables"))
        self.assertTrue(Encoding(unit=PHONES).phonetic and Encoding(unit=SYLLABLES).phonetic)
        self.assertFalse(Encoding().phonetic or Encoding(unit=WORDS).phonetic)
        self.assertEqual(Encoding.from_dict({"unit": "phone", "n": 3, "stride": 1}), Encoding(unit=PHONES))
        with self.assertRaises(ValueError):
            parse_encoding("rune:3")

    def test_text_becomes_sounds(self):
        enc = Encoding(unit=PHONES)
        self.assertEqual(enc.units("The cat sat."), ["DH", "AH0", "#", "K", "AE1", "T", "#", "S", "AE1", "T", "."])
        self.assertEqual(enc.length("the cat"), 6)
        self.assertEqual(enc.encode("the cat"), ["DH AH0 #", "AH0 # K", "# K AE1", "K AE1 T"])
        self.assertEqual(enc.normalize("the cat"), "DH AH0 # K AE1 T")
        self.assertEqual(enc.decode_grams(enc.encode("the cat sat")), enc.normalize("the cat sat"))
        syl = Encoding(unit=SYLLABLES, n=2)
        self.assertEqual(syl.units("butter cup"), ["B.AH1.T", "ER0", "#", "K.AH1.P"])
        self.assertEqual(syl.encode("butter cup"), ["B.AH1.T ER0", "ER0 #", "# K.AH1.P"])

    def test_a_label_cuts_into_the_units_it_was_made_of(self):
        """Any run of units - a gram, a compressed label, one starting or ending on a # or a pause - reads back as itself."""
        enc = Encoding(unit=PHONES)
        for text in TEXTS + ["hello, world. why?", "the K AE1 T sat"]:
            units = enc.units(text)
            for lo in range(len(units)):
                for hi in range(lo + 1, min(len(units), lo + 5) + 1):
                    run = " ".join(units[lo:hi])
                    self.assertEqual(enc.units(run), units[lo:hi], run)
                    self.assertEqual(enc.length(run), hi - lo, run)

    def test_a_prefix_is_matched_by_its_sounds(self):
        enc = Encoding(unit=PHONES)
        label = "DH AH0 # K AE1 T # S AE1 T"
        self.assertTrue(enc.has_unit_prefix(label, "the cat"))
        self.assertTrue(enc.has_unit_prefix(label, "DH AH0 # K AE1 T"))
        self.assertFalse(enc.has_unit_prefix(label, "the ca"))
        self.assertFalse(enc.has_unit_prefix(label, "the cats"))
        self.assertTrue(enc.has_unit_prefix(label, ""))
        self.assertEqual(enc.join("the cat", "# S AE1 T"), "DH AH0 # K AE1 T # S AE1 T")
        self.assertEqual(enc.join("", "K AE1 T"), "K AE1 T")

    def test_spelling_back(self):
        enc = Encoding(unit=PHONES)
        self.assertEqual(enc.spell("DH AH0 # K AE1 T # S AE1 T ."), "the cat sat.")
        self.assertEqual(enc.spell("Z IH1 B R AH0"), "zibra")  # sounds no known word has are respelled
        self.assertEqual(Encoding().spell("the cat"), "the cat")  # nothing to do for letters
        self.assertEqual(Encoding(unit=SYLLABLES).spell("B.AH1.T ER0 # K.AH1.P"), "butter cup")

    def test_the_alphabet(self):
        enc = Encoding(unit=PHONES)
        net = CountRewardNet(seed=1, encoding=enc)
        net.train(TEXTS, epochs=1)
        rows = word_rows(enc, net.graph.trigram_index)
        self.assertEqual(rows[0]["word"], "#")  # the boundary is the commonest unit
        self.assertTrue(all(len(r["word"]) <= 3 for r in rows))
        self.assertIn("AE1", {r["word"] for r in rows})

    def test_a_model_over_sounds_is_read_back_as_words(self):
        enc = Encoding(unit=PHONES)
        net = CountRewardNet(seed=1, encoding=enc)
        net.train(TEXTS, epochs=2)
        net.graph.check_invariants(TEXTS)
        pred = net.predict("the cat sat", length=4, k=2)
        self.assertTrue(pred.full_text.startswith("DH AH0 # K AE1 T # S AE1 T"), pred.full_text)
        self.assertTrue(enc.spell(pred.full_text).startswith("the cat sat"), enc.spell(pred.full_text))
        score = net.score("the cat sat on the mat")
        self.assertEqual(score["unknown_transitions"], 0)
        self.assertEqual(net.stats()["units"], "phones")

    def test_the_tokenizer_is_shared_and_portable(self):
        tok = phonetic_tokenizer(PHONES)
        self.assertIs(tok, phonetic_tokenizer(PHONES))
        self.assertIn("the core lexicon", tok.lexicon.describe())
        self.assertIsNot(tok, phonetic_tokenizer(SYLLABLES))


class TestWithoutTheTokenizer(unittest.TestCase):
    def test_a_phonetic_unit_is_refused_with_a_hint(self):
        import radixnet.encoding as encoding_module

        saved_cache = dict(encoding_module._PHONETIC)
        saved_module = sys.modules.get("phonetok")
        encoding_module._PHONETIC.clear()
        sys.modules["phonetok"] = None  # makes `import phonetok` raise ImportError
        try:
            with self.assertRaises(ValueError) as caught:
                Encoding(unit=PHONES)
            self.assertIn("PhoneticTokenizer", str(caught.exception))
            Encoding()  # letters and words need nothing
            Encoding(unit=WORDS)
        finally:
            if saved_module is None:
                del sys.modules["phonetok"]
            else:
                sys.modules["phonetok"] = saved_module
            encoding_module._PHONETIC.clear()
            encoding_module._PHONETIC.update(saved_cache)
        self.assertEqual(Encoding().unit, CHARS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
