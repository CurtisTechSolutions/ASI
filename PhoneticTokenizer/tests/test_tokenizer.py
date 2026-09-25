"""Tests for phonetok.tokenizer: the four levels, the text form, the ids, the way back."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.lexicon import Lexicon  # noqa: E402
from phonetok.phones import FEATURE_DIM, SYMBOLS  # noqa: E402
from phonetok.tokenizer import (  # noqa: E402
    CONSTITUENT, FIXED, LEVELS, PHONEME, SYLLABLE, WORD, PhoneticTokenizer, Token, Vocab, parse_token,
)

TEXT = "The cat sat on the mat."


def tok(level=PHONEME, **kw):
    return PhoneticTokenizer(level=level, lexicon=Lexicon.core(), **kw)


class TestTokenGrammar(unittest.TestCase):
    def test_parse_token(self):
        self.assertEqual(parse_token("K"), ("phone", ("K",)))
        self.assertEqual(parse_token("AE1"), ("nucleus", ("AE1",)))
        self.assertEqual(parse_token("#"), ("boundary", ()))
        self.assertEqual(parse_token(","), ("pause", ()))
        self.assertEqual(parse_token("<s>"), ("special", ()))
        self.assertEqual(parse_token("S.T.R-"), ("onset", ("S", "T", "R")))
        self.assertEqual(parse_token("-NG.TH.S"), ("coda", ("NG", "TH", "S")))
        self.assertEqual(parse_token("K.AE1.T"), ("syllable", ("K", "AE1", "T")))
        self.assertEqual(parse_token("B.AH1.T.ER0"), ("word", ("B", "AH1", "T", "ER0")))
        for bad in ("", "k", "cat", "K.", ".K", "AE1-", "-AE1", "S.T.R-x", "K..T", "X", "A", "I", "the", "-"):
            self.assertIsNone(parse_token(bad), bad)


class TestLevels(unittest.TestCase):
    def test_the_four_text_forms(self):
        self.assertEqual(tok(PHONEME).text(TEXT), "DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T .")
        self.assertEqual(tok(CONSTITUENT).text(TEXT), "DH- AH0 # K- AE1 -T # S- AE1 -T # AA1 -N # DH- AH0 # M- AE1 -T .")
        self.assertEqual(tok(SYLLABLE).text(TEXT), "DH.AH0 # K.AE1.T # S.AE1.T # AA1.N # DH.AH0 # M.AE1.T .")
        self.assertEqual(tok(WORD).text(TEXT), "DH.AH0 # K.AE1.T # S.AE1.T # AA1.N # DH.AH0 # M.AE1.T .")
        self.assertEqual(tok(SYLLABLE).tokens("butter strengths"), ["B.AH1.T", "ER0", "#", "S.T.R.EH1.NG.K.TH.S"])
        self.assertEqual(tok(CONSTITUENT).tokens("butter"), ["B-", "AH1", "-T", "ER0"])
        self.assertEqual(tok(WORD).tokens("butter"), ["B.AH1.T.ER0"])
        with self.assertRaises(ValueError):
            PhoneticTokenizer(level="letters")

    def test_idempotent_and_mixed(self):
        for level in LEVELS:
            t = tok(level)
            form = t.text(TEXT)
            self.assertEqual(t.text(form), form, level)
        self.assertEqual(tok().text("the K AE1 T sat"), "DH AH0 # K AE1 T # S AE1 T")
        self.assertEqual(tok().text("DH.AH0 # K.AE1.T"), "DH AH0 # K AE1 T")
        self.assertEqual(tok(SYLLABLE).text("DH AH0 # K AE1 T S"), "DH.AH0 # K.AE1.T.S")
        self.assertEqual(tok(CONSTITUENT).text("K.AE1.T"), "K- AE1 -T")

    def test_punctuation_becomes_pauses(self):
        t = tok()
        self.assertEqual(t.text("Hello, world!"), "HH AH0 L OW1 , W ER1 L D .")
        self.assertEqual(t.text("Why? Because."), "W AY1 ? B IH0 K AO1 Z .")
        self.assertEqual(t.text('"The cat" (sat); the mat: yes...'), "DH AH0 # K AE1 T # S AE1 T , DH AH0 # M AE1 T , Y EH1 S .")
        self.assertEqual(t.text("one - two -- three"), "W AH1 N , T UW1 , TH R IY1")
        self.assertEqual(t.text("wait... what?!"), "W EY1 T . W AH1 T ?")
        self.assertEqual(t.text(", , ."), ", , .")  # pause tokens stay; punctuation would not
        self.assertEqual(t.text("the cat, , sat"), "DH AH0 # K AE1 T , , S AE1 T")  # a pause token is kept as given
        for run in ("# K AE1 T", "DH AH0 #", "AH0 # K", ", DH AH0", "K AE1 T ,", "# #", ","):
            self.assertEqual(t.text(run), run, run)  # any run of tokens reads back as itself
        self.assertEqual(t.text(""), "")
        self.assertEqual(tok(pauses=False).text("Hello, world!"), "HH AH0 L OW1 # W ER1 L D")
        self.assertEqual(tok(boundaries=False).text("the cat, sat"), "DH AH0 K AE1 T , S AE1 T")
        self.assertEqual(tok(stress=False).text("the cat"), "DH AH # K AE T")

    def test_pieces(self):
        t = tok()
        self.assertEqual(t.pieces('"Hello," she said. K AE1 T # ok? ,'),
                         [("word", "Hello"), ("punct", ","), ("word", "she"), ("word", "said"), ("punct", "."),
                          ("sounds", "K"), ("sounds", "AE1"), ("sounds", "T"), ("boundary", "#"), ("word", "ok"),
                          ("punct", "?"), ("pause", ",")])

    def test_tokens_know_their_word(self):
        toks = tok().tokenize("the cat, sat")
        self.assertEqual([t.word for t in toks], [0, 0, -1, 1, 1, 1, -1, 2, 2, 2])
        self.assertEqual([t.kind for t in toks][:4], ["phone", "nucleus", "boundary", "phone"])
        self.assertEqual(toks[6], Token(",", "pause"))
        self.assertEqual(str(toks[0]), "DH")


class TestIds(unittest.TestCase):
    def test_phoneme_ids_are_the_fixed_alphabet(self):
        t = tok()
        self.assertEqual(t.vocab_size, 92)
        self.assertEqual(list(t.vocab.tokens), list(SYMBOLS))
        ids = t.encode("the cat")
        self.assertEqual(ids, [SYMBOLS.index(x) for x in ("DH", "AH0", "#", "K", "AE1", "T")])
        self.assertEqual(t.vocab_size, 92)  # nothing grows
        self.assertEqual(t.vocab.id("K.AE1.T"), 1)  # <unk>: not a phoneme token

    def test_other_levels_grow(self):
        t = tok(SYLLABLE)
        self.assertEqual(t.vocab_size, len(FIXED))
        ids = t.encode("the cat sat")
        self.assertEqual(ids, [8, 4, 9, 4, 10])
        self.assertEqual(t.vocab.tokens[8:], ["DH.AH0", "K.AE1.T", "S.AE1.T"])
        self.assertEqual(t.encode("the dog", grow=False), [8, 4, 1])
        self.assertEqual(t.encode("the dog"), [8, 4, 11])
        frozen = Vocab(FIXED, frozen=True)
        self.assertEqual(frozen.id("X", grow=True), 1)
        self.assertEqual(frozen.token(99), "<unk>")

    def test_decode(self):
        for level in LEVELS:
            t = tok(level)
            self.assertEqual(t.decode(t.encode(TEXT)), "the cat sat on the mat.", level)
            self.assertEqual(t.decode(t.tokens("Hello, world! Why?")), "hello, world. why?", level)
            self.assertEqual(t.decode(t.tokenize("zebra")), "zebra", level)
        t = tok()
        self.assertEqual(t.decode(["K", "AE1", "T", "S"]), "cats")  # the plural is read through the stem's sounds
        self.assertEqual(t.decode(["Z", "IH1", "B", "R", "AH0"]), "zibra")  # respelled
        self.assertEqual(t.decode([]), "")
        self.assertEqual(t.decode(["<s>", "K", "AE1", "T", "</s>"]), "cat")
        self.assertEqual(t.words_of(["K", "AE1", "T", "#", "S", "AE1", "T"]), [["K", "AE1", "T"], ["S", "AE1", "T"]])
        self.assertEqual(t.words_of(t.encode("the cat")), [["DH", "AH0"], ["K", "AE1", "T"]])
        with self.assertRaises(ValueError):
            t.parse(["K", "cat"])
        parsed = t.parse(["K", "AE1", "T", "#", "S", "AE1", "T"])
        self.assertEqual([p.word for p in parsed], [0, 0, 0, -1, 1, 1, 1])


class TestSound(unittest.TestCase):
    def test_helpers(self):
        t = tok()
        self.assertEqual(t.ipa("The cat sat."), "ðə ˈkæt ˈsæt")
        self.assertEqual(t.ipa("computer"), "kəmˈpjutɚ")
        self.assertEqual([s.text() for s in t.syllables("computer")], ["K.AH0.M", "P.Y.UW1", "T.ER0"])
        self.assertEqual(t.pronounce("cat"), ["K", "AE1", "T"])
        self.assertTrue(t.rhymes("cat", "hat"))
        self.assertFalse(t.rhymes("cat", "dog"))
        self.assertTrue(t.alliterates("big", "bad"))
        self.assertGreater(t.affinity("S", "T"), 0)
        self.assertGreater(t.score("strength"), t.score("ngstrelk"))
        self.assertTrue(t.well_formed("strength"))
        self.assertEqual(t.blend("smoke", "fog"), "smog")
        self.assertEqual(t.blend("breakfast", "lunch"), "brunch")
        coined = [t.coin(i) for i in range(5)]
        self.assertEqual(coined, [t.coin(i) for i in range(5)])
        self.assertTrue(all(c and c.isalpha() for c in coined), coined)

    def test_features(self):
        t = tok()
        self.assertEqual(len(t.features("K")), FEATURE_DIM)
        self.assertEqual(t.features("#"), [0.0] * FEATURE_DIM)
        self.assertEqual(t.features(4), [0.0] * FEATURE_DIM)
        self.assertEqual(t.features("S.T-")[1], 1.0)  # both consonantal
        self.assertAlmostEqual(t.features("K.AE1.T")[0], 1 / 3)  # one syllabic phone of three

    def test_explain(self):
        rows = tok().explain("The zebra, 42!")
        self.assertEqual([r["word"] for r in rows], ["The", "zebra", "42"])
        self.assertEqual([r["how"] for r in rows], ["lexicon", "rules", "number"])
        self.assertEqual(rows[1]["syllables"], ["Z.EH1.B", "R.AH0"])
        self.assertEqual(rows[0]["ipa"], "ðə")
        self.assertIsInstance(rows[1]["score"], float)


class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = tok(SYLLABLE)
            t.encode("the zebra sat")
            path = os.path.join(tmp, "tok.json")
            t.save(path)
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["format"], "phonetok")
            self.assertEqual(doc["level"], "syllable")
            self.assertIn("zebra", doc["memory"])
            back = PhoneticTokenizer.load(path, lexicon=Lexicon.core())
            self.assertEqual(back.vocab.tokens, t.vocab.tokens)
            self.assertEqual(back.encode("the zebra sat", grow=False), t.encode("the zebra sat", grow=False))
            self.assertEqual(back.decode(back.encode("zebra")), "zebra")
            with self.assertRaises(ValueError):
                tok(PHONEME).load_dict(doc)
            with self.assertRaises(ValueError):
                tok(SYLLABLE).load_dict({"format": "other"})
            self.assertIn("syllable tokens", repr(back))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
