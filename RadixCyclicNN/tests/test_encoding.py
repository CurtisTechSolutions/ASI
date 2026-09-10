"""Tests for radixnet.encoding."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.encoding import END_LABEL, START_LABEL, WINDOW, Decoder, Encoder  # noqa: E402


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
