"""Tests for radixnet.encoding."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.encoding import (  # noqa: E402
    CHARS, END_LABEL, START_LABEL, WINDOW, WORDS, Decoder, Encoder, Encoding, parse_encoding,
)

EVERY_ENCODING = [
    Encoding(), Encoding(n=1), Encoding(n=2), Encoding(n=5), Encoding(n=4, stride=4),
    Encoding(n=5, stride=5), Encoding(n=6, stride=3), Encoding(unit=WORDS, n=1),
    Encoding(unit=WORDS, n=2), Encoding(unit=WORDS, n=3), Encoding(unit=WORDS, n=2, stride=2),
]
"""The encodings the tests drive end to end."""


class TestEncoder(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(Encoder().encode("hello"), ["hel", "ell", "llo"])

    def test_short_text(self):
        enc = Encoder()
        self.assertEqual(enc.encode(""), [])
        self.assertEqual(enc.encode("a"), [])
        self.assertEqual(enc.encode("ab"), [])
        self.assertEqual(enc.encode("abc"), ["abc"])

    def test_window_constant(self):
        self.assertEqual(WINDOW, 3)
        self.assertEqual(Encoder().window, 3)

    def test_unicode(self):
        text = "héllo wörld 😀!"
        grams = Encoder().encode(text)
        self.assertEqual(len(grams), len(text) - 2)
        self.assertTrue(all(len(g) == 3 for g in grams))
        self.assertEqual(grams[-1], text[-3:])
        self.assertEqual(Decoder().decode_trigrams(grams), text)

    def test_custom_window(self):
        self.assertEqual(Encoder(2).encode("abcd"), ["ab", "bc", "cd"])
        self.assertEqual(Encoder(4).encode("abc"), [])
        with self.assertRaises(ValueError):
            Encoder(0)


class TestDecoder(unittest.TestCase):
    def test_decode_trigrams(self):
        dec = Decoder()
        self.assertEqual(dec.decode_trigrams([]), "")
        self.assertEqual(dec.decode_trigrams(["abc"]), "abc")
        self.assertEqual(dec.decode_trigrams(["hel", "ell", "llo"]), "hello")

    def test_round_trip_random(self):
        rng = random.Random(1234)
        enc, dec = Encoder(), Decoder()
        alphabet = "ab cdé😀\n"
        for _ in range(500):
            n = rng.randint(3, 40)
            text = "".join(rng.choice(alphabet) for _ in range(n))
            self.assertEqual(dec.decode_trigrams(enc.encode(text)), text)

    def test_round_trip_custom_window(self):
        enc, dec = Encoder(5), Decoder(5)
        for text in ["abcde", "abcdefgh", "the quick brown fox"]:
            self.assertEqual(dec.decode_trigrams(enc.encode(text)), text)

    def test_decode_path_include_context(self):
        dec = Decoder()
        labels = ["the c", " cat", "at sat"]  # consecutive labels overlap by two chars
        # first label in full, then label[2:] for every following label
        self.assertEqual(dec.decode_path(labels), "the cat sat")
        self.assertEqual(dec.decode_path(labels, start_offset=1), "he cat sat")
        self.assertEqual(dec.decode_path(labels, start_offset=2, include_context=True), "e cat sat")

    def test_decode_path_continuation_only(self):
        dec = Decoder()
        labels = ["the c", " cat", "at sat"]
        # matched trigram "the" at offset 0 -> remainder " c", then continuation
        self.assertEqual(dec.decode_path(labels, 0, include_context=False), " cat sat")
        # matched trigram "he " at offset 1 -> remainder "c"
        self.assertEqual(dec.decode_path(labels, 1, include_context=False), "cat sat")
        # matched the last trigram "e c" -> no remainder from the first node
        self.assertEqual(dec.decode_path(labels, 2, include_context=False), "at sat")

    def test_decode_path_skips_sentinels(self):
        dec = Decoder()
        labels = [START_LABEL, "abc", "bcd", END_LABEL]
        self.assertEqual(dec.decode_path(labels), "abcd")
        self.assertEqual(dec.decode_path(labels, include_context=False), "d")
        self.assertEqual(dec.decode_path([START_LABEL, END_LABEL]), "")
        self.assertEqual(dec.decode_path([]), "")

    def test_decode_path_matches_trigram_decode(self):
        rng = random.Random(7)
        enc, dec = Encoder(), Decoder()
        for _ in range(200):
            text = "".join(rng.choice("abc ") for _ in range(rng.randint(3, 30)))
            grams = enc.encode(text)
            # uncompressed labels are exactly the trigrams
            self.assertEqual(dec.decode_path(grams), text)
            self.assertEqual(dec.decode_path(grams, include_context=False), text[3:])

    def test_decode_path_can_keep_sentinel_lookalikes(self):
        dec = Decoder()
        # a real node may be labelled "<s>": the text "x<s>y" contains that trigram
        labels = ["x<s", "<s>", "s>y"]
        self.assertEqual(dec.decode_path(labels), "x<sy")  # spec default: skipped by label
        self.assertEqual(dec.decode_path(labels, skip_sentinels=False), "x<s>y")
        self.assertEqual(dec.decode_path(labels, 0, include_context=False, skip_sentinels=False), ">y")
        self.assertEqual(dec.decode_path([START_LABEL, END_LABEL], skip_sentinels=False), "<s>s>")
        self.assertEqual(dec.decode_path([], skip_sentinels=False), "")

    def test_negative_offset_rejected(self):
        with self.assertRaises(ValueError):
            Decoder().decode_path(["abc"], start_offset=-1)


if __name__ == "__main__":
    unittest.main()


class TestEncodingDial(unittest.TestCase):
    """The three dials: what a unit is, how many make a gram, how far apart they start."""

    def test_defaults_and_validation(self):
        self.assertEqual(Encoding(), Encoding(unit=CHARS, n=WINDOW, stride=1))
        self.assertTrue(Encoding().is_default())
        self.assertEqual(Encoding().overlap, WINDOW - 1)
        self.assertEqual(Encoding(n=4, stride=4).overlap, 0)
        self.assertFalse(Encoding(n=4, stride=4).sliding)
        for bad in (dict(unit="rune"), dict(n=0), dict(stride=0), dict(n=2, stride=3)):
            with self.assertRaises(ValueError):
                Encoding(**bad)

    def test_parse(self):
        cases = {
            "": Encoding(), "trigram": Encoding(), "char:4": Encoding(n=4),
            "chars:4:4": Encoding(n=4, stride=4), "char:5:groups": Encoding(n=5, stride=5),
            "letters:7:2": Encoding(n=7, stride=2), "word": Encoding(unit=WORDS, n=1),
            "word:2": Encoding(unit=WORDS, n=2), "word-trigram": Encoding(unit=WORDS, n=3),
            "WORD:2:2": Encoding(unit=WORDS, n=2, stride=2),
        }
        for spec, want in cases.items():
            self.assertEqual(parse_encoding(spec), want, spec)
            self.assertEqual(parse_encoding(str(want)), want, spec)  # str() round trips
        for bad in ("rune:3", "char:x", "char:3:y", "char:2:3", "char:0", "a:b:c:d"):
            with self.assertRaises(ValueError, msg=bad):
                parse_encoding(bad)

    def test_encode(self):
        cases = [
            (Encoding(), "hello", ["hel", "ell", "llo"]),
            (Encoding(n=1), "abc", ["a", "b", "c"]),
            (Encoding(n=4, stride=4), "abcdefghij", ["abcd", "efgh"]),
            (Encoding(n=5, stride=5), "abcdefghij", ["abcde", "fghij"]),
            (Encoding(n=4, stride=2), "abcdef", ["abcd", "cdef"]),
            (Encoding(), "hi", []),
            (Encoding(unit=WORDS, n=2), "the cat sat down", ["the cat", "cat sat", "sat down"]),
            (Encoding(unit=WORDS, n=3), "the cat sat down", ["the cat sat", "cat sat down"]),
            (Encoding(unit=WORDS, n=2, stride=2), "the cat sat down here", ["the cat", "sat down"]),
            (Encoding(unit=WORDS, n=2), "  the   cat  ", ["the cat"]),
            (Encoding(unit=WORDS, n=2), "alone", []),
        ]
        for enc, text, want in cases:
            self.assertEqual(enc.encode(text), want, f"{enc}.encode({text!r})")

    def test_normalize_is_what_comes_back(self):
        cases = [
            (Encoding(), "hello", "hello"),
            (Encoding(n=4, stride=4), "abcdefghij", "abcdefgh"),
            (Encoding(n=4, stride=4), "abc", ""),
            (Encoding(unit=WORDS, n=2), "  the  cat   sat ", "the cat sat"),
            (Encoding(unit=WORDS, n=2, stride=2), "the cat sat down here", "the cat sat down"),
        ]
        for enc, text, want in cases:
            self.assertEqual(enc.normalize(text), want, f"{enc}.normalize({text!r})")
            grams = enc.encode(text)
            if grams:
                self.assertEqual(enc.decode_grams(grams), want, f"{enc} decode_grams")

    def test_units_and_join(self):
        for enc in EVERY_ENCODING:
            text = "the quick brown fox jumps"
            view = enc.units(text)
            self.assertEqual(len(view), enc.length(text), str(enc))
            self.assertEqual(enc.piece(text, 0), text if enc.unit == CHARS else " ".join(text.split()))
            self.assertEqual(enc.piece(text, 2, 2), "")
            whole = " ".join(text.split()) if enc.unit == WORDS else text
            self.assertEqual(enc.join(*[enc.piece(text, i, i + 1) for i in range(len(view))]), whole)

    def test_unit_prefix_stops_at_a_word_boundary(self):
        word = Encoding(unit=WORDS, n=2)
        self.assertFalse(word.has_unit_prefix("the cat", "the ca"))
        self.assertTrue(word.has_unit_prefix("the cat", "the"))
        self.assertTrue(word.has_unit_prefix("the cat", "the cat"))
        self.assertTrue(Encoding().has_unit_prefix("the cat", "the ca"))

    def test_decode_path(self):
        word = Encoding(unit=WORDS, n=2)
        self.assertEqual(word.decode_path(["the cat", "cat sat", "sat down"]), "the cat sat down")
        # the continuation only: the matched gram is dropped
        self.assertEqual(word.decode_path(["the cat", "cat sat"], 0, include_context=False), "sat")
        groups = Encoding(n=4, stride=4)
        self.assertEqual(groups.decode_path(["abcd", "efgh"]), "abcdefgh")

    def test_encoder_and_decoder_carry_an_encoding(self):
        self.assertEqual(Encoder(5).encode("abcdefg"), ["abcde", "bcdef", "cdefg"])
        word = Encoding(unit=WORDS, n=2)
        self.assertEqual(Encoder(encoding=word).encode("the cat sat"), ["the cat", "cat sat"])
        self.assertEqual(Decoder(encoding=word).decode_trigrams(["the cat", "cat sat"]), "the cat sat")
        self.assertEqual(Encoder().window, WINDOW)
