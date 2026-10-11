"""The base64 unit: anything - a text, an image, a video - as the n-grams of its base64 symbols.

:mod:`tests.test_encoding` covers the encodings of text; this holds the base64
unit to what it claims: the codec (:func:`base64_text`, :func:`base64_prefix`,
:func:`base64_bytes`), how a file becomes texts (:func:`base64_file_texts`),
what a prediction decodes into (:func:`decoded_prediction`), and every model
kind trained on the symbols of real bytes, saved and read back.
"""

import base64
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.encoding import (  # noqa: E402
    BASE64, BASE64_ALPHABET, CHARACTER_UNITS, CHARS, UNIT_KINDS, Encoding, base64_bytes, base64_file_texts,
    base64_prefix, base64_text, decoded_prediction, is_utf8, parse_encoding,
)
from radixnet.graph import FIRST  # noqa: E402
from radixnet.model import load_model, new_model  # noqa: E402

B64 = Encoding(unit=BASE64)

TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the log",
    "a quick brown fox jumps over the lazy dog",
]

# a tiny binary blob that is not UTF-8: what an image or a recording looks like to the unit
BLOB = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1])


class TestTheDial(unittest.TestCase):
    def test_parse_and_name(self):
        for spec in ("base64", "b64", "BASE64:3:1", " base64 "):
            with self.subTest(spec=spec):
                self.assertEqual(parse_encoding(spec), Encoding(unit=BASE64, n=3, stride=1))
        enc = parse_encoding("base64:4:4")
        self.assertEqual((enc.unit, enc.n, enc.stride), (BASE64, 4, 4))
        self.assertEqual(str(enc), "base64:4:4")
        self.assertEqual(enc.describe(), "4-symbol grams, stride 4 (groups)")
        self.assertEqual(enc.units_name, "symbols")
        self.assertTrue(enc.is_base64)
        self.assertFalse(enc.is_default())
        self.assertFalse(Encoding().is_base64)
        self.assertIn(BASE64, UNIT_KINDS)
        self.assertEqual(CHARACTER_UNITS, (CHARS, BASE64))
        with self.assertRaises(ValueError) as caught:
            parse_encoding("base65")
        self.assertIn("base64", str(caught.exception))

    def test_the_file_block(self):
        enc = parse_encoding("base64:5:2")
        self.assertEqual(enc.to_dict(), {"unit": "base64", "n": 5, "stride": 2})
        self.assertEqual(Encoding.from_dict(enc.to_dict()), enc)

    def test_a_symbol_is_one_unit(self):
        """The symbols slice like characters: one code point, one unit."""
        text = "dGhlIGNhdA"
        self.assertEqual(B64.units(text), text)
        self.assertEqual(B64.length(text), 10)
        self.assertEqual(B64.join("dGh", "", "lIG"), "dGhlIG")
        self.assertTrue(B64.has_unit_prefix("dGhlIG", "dGh"))
        self.assertFalse(B64.has_unit_prefix("dGhlIG", "dGx"))
        self.assertEqual(B64.reverse("abc+/"), "/+cba")
        self.assertEqual(B64.encode("dGhlIA"), ["dGh", "Ghl", "hlI", "lIA"])


class TestTheCodec(unittest.TestCase):
    def test_bytes_become_symbols_without_padding(self):
        self.assertEqual(base64_text(b""), "")
        self.assertEqual(base64_text(b"a"), "YQ")
        self.assertEqual(base64_text(b"ab"), "YWI")
        self.assertEqual(base64_text(b"abc"), "YWJj")
        self.assertEqual(base64_text(b"the cat"), "dGhlIGNhdA")
        self.assertEqual(base64_text(bytes(range(256))), base64.b64encode(bytes(range(256))).decode().rstrip("="))
        self.assertTrue(set(base64_text(BLOB)) <= set(BASE64_ALPHABET))
        self.assertEqual(len(BASE64_ALPHABET), 64)

    def test_round_trip_of_any_bytes(self):
        rng = random.Random(7)
        for size in list(range(0, 40)) + [255, 1000, 4097]:
            data = bytes(rng.randrange(256) for _ in range(size))
            with self.subTest(size=size):
                symbols = base64_text(data)
                self.assertEqual(len(symbols), (size * 8 + 5) // 6)
                self.assertEqual(base64_bytes(symbols), data)

    def test_the_decoder_takes_whatever_was_written(self):
        """A predicted text always decodes: padding, whitespace and stray characters are skipped, a dangling
        symbol is dropped, and the bits a last partial group has over are ignored."""
        self.assertEqual(base64_bytes(""), b"")
        self.assertEqual(base64_bytes("YQ=="), b"a")
        self.assertEqual(base64_bytes("YW\nJj \t"), b"abc")
        self.assertEqual(base64_bytes("Y-W_J.j"), b"abc")  # the URL-safe symbols are not this alphabet's
        self.assertEqual(base64_bytes("YWJjZ"), b"abc")  # one symbol cannot make a byte
        self.assertEqual(base64_bytes("Y"), b"")
        self.assertEqual(base64_bytes("YR"), b"a")  # R's low four bits are left over
        self.assertEqual(base64_bytes("YWJ"), b"ab")

    def test_a_prefix_is_the_symbols_its_bytes_fix(self):
        self.assertEqual(base64_prefix(b""), "")
        self.assertEqual(base64_prefix(b"t"), "d")
        self.assertEqual(base64_prefix(b"th"), "dG")
        self.assertEqual(base64_prefix(b"the"), "dGhl")
        self.assertEqual(base64_prefix(b"the d"), "dGhlIG")
        self.assertEqual(base64_prefix(b"the cat"), "dGhlIGNhd")
        rng = random.Random(3)
        for size in range(0, 30):
            data = bytes(rng.randrange(256) for _ in range(size))
            with self.subTest(size=size):
                prefix = base64_prefix(data)
                self.assertEqual(len(prefix), size * 8 // 6)
                for tail in (b"", b"\x00", b"\xff", b"xyz"):  # every continuation of the bytes shares it
                    self.assertTrue(base64_text(data + tail).startswith(prefix))

    def test_is_utf8(self):
        self.assertTrue(is_utf8(b"the cat \xe2\x82\xac"))
        self.assertTrue(is_utf8(b""))
        self.assertFalse(is_utf8(BLOB))
        self.assertFalse(is_utf8(b"\xed\xa0\x80"))  # a surrogate is not UTF-8


class TestFilesBecomeTexts(unittest.TestCase):
    def test_a_text_file_is_one_text_per_line(self):
        data = b"one\r\n  \ntwo\n\n\tthree \n"
        self.assertEqual(base64_file_texts(data), [base64_text(b"one\r"), base64_text(b"two"), base64_text(b"\tthree ")])
        self.assertEqual([base64_bytes(t) for t in base64_file_texts(data)], [b"one\r", b"two", b"\tthree "])

    def test_anything_else_is_one_text_of_every_byte(self):
        self.assertEqual(base64_file_texts(BLOB), [base64_text(BLOB)])
        self.assertEqual(base64_bytes(base64_file_texts(BLOB)[0]), BLOB)

    def test_whole_file_keeps_every_byte(self):
        data = b"one\ntwo\n\n"
        self.assertEqual(base64_file_texts(data, whole_file=True), [base64_text(data)])
        self.assertEqual(base64_bytes(base64_file_texts(data, whole_file=True)[0]), data)

    def test_nothing_comes_from_nothing(self):
        self.assertEqual(base64_file_texts(b""), [])
        self.assertEqual(base64_file_texts(b"", whole_file=True), [])
        self.assertEqual(base64_file_texts(b" \n\t\n"), [])
        self.assertEqual(base64_file_texts(b" \n\t\n", whole_file=True), [base64_text(b" \n\t\n")])


class TestContent(unittest.TestCase):
    def test_what_comes_in(self):
        self.assertEqual(B64.text_input("the cat"), "dGhlIGNhdA")
        self.assertEqual(B64.prefix_input("the cat"), "dGhlIGNhd")
        self.assertEqual(B64.content_text(BLOB), base64_text(BLOB))
        # every other unit takes text as it is
        plain = Encoding()
        self.assertEqual(plain.text_input("the cat"), "the cat")
        self.assertEqual(plain.prefix_input("the cat"), "the cat")
        self.assertEqual(plain.content_text("café".encode()), "café")

    def test_what_goes_out(self):
        self.assertEqual(B64.content_bytes(base64_text(BLOB)), BLOB)
        self.assertEqual(Encoding().content_bytes("café"), "café".encode())

    def test_a_prediction_decodes(self):
        out = decoded_prediction(B64, "dGhlIGNhdCBzYXQ", b"the cat")
        self.assertEqual(out, {"decoded": "the cat sat", "decoded_continuation": " sat", "decoded_bytes": 11})
        # no prefix: all of it was written
        self.assertEqual(decoded_prediction(B64, "dGhlIGNhdA")["decoded_continuation"], "the cat")
        # every other encoding adds nothing
        self.assertEqual(decoded_prediction(Encoding(), "the cat sat", b"the cat"), {})

    def test_the_last_byte_of_a_prefix_is_the_walks_to_finish(self):
        """Asked ``the d``, the symbols fix 4 bytes and half of the fifth; a walk that finishes it as ``l`` said
        ``the log``, and the continuation is everything after the bytes the two share."""
        prefix = base64_prefix(b"the d")
        self.assertEqual(prefix, "dGhlIG")
        out = decoded_prediction(B64, prefix + "xvZw", b"the d")
        self.assertEqual((out["decoded"], out["decoded_continuation"]), ("the log", "log"))
        out = decoded_prediction(B64, base64_text(b"the dog"), b"the d")
        self.assertEqual((out["decoded"], out["decoded_continuation"]), ("the dog", "og"))

    def test_bytes_that_are_not_text_read_as_replacement_characters(self):
        """One U+FFFD per maximal invalid subpart (the Unicode Standard's practice, and what Rust and the
        Go port's decoder write too)."""
        cases = [
            (b"ok\xe2\x82A\xff", "ok�A�"),          # a truncated sequence is one subpart
            (b"\xed\xa0\x80", "���"),           # a surrogate: three
            (b"\xf0\x90\x80", "�"),                       # a truncated four-byte sequence at the end
            (b"\xc0\xaf", "��"),                     # an overlong form: two
            (b"\xf4\x90\x80\x80", "����"),  # past U+10FFFF: four
            (b"\xe2\x82\xac", "€"),
        ]
        for data, text in cases:
            with self.subTest(data=data):
                self.assertEqual(decoded_prediction(B64, base64_text(data))["decoded"], text)


class TestEveryKindLearnsTheSymbols(unittest.TestCase):
    TRAIN = [base64_text(t.encode()) for t in TEXTS] + [base64_text(BLOB)]

    def test_train_predict_decode_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("count", "radix", "resonant", "negative"):
                with self.subTest(kind=kind):
                    model = new_model(kind, seed=1, encoding=B64)
                    model.train(self.TRAIN, epochs=2)
                    self.assertEqual(model.stats()["encoding"], "base64:3:1")
                    self.assertEqual(model.stats()["units"], "symbols")
                    alphabet = set(BASE64_ALPHABET)
                    for label in model.graph.labels[FIRST:]:
                        if label:
                            self.assertTrue(set(label) <= alphabet, f"{label!r} is not base64 symbols")
                    path = os.path.join(tmp, f"{kind}.json")
                    model.save(path)
                    loaded = load_model(path)
                    self.assertEqual(loaded.encoding, B64)
                    self.assertEqual(set(loaded.graph.trigram_index), set(model.graph.trigram_index))
                    if kind == "negative":
                        continue
                    result = loaded.predict(B64.prefix_input("the cat"), length=12)
                    decoded = decoded_prediction(B64, result.full_text, b"the cat")
                    self.assertTrue(decoded["decoded"].startswith("the ca"), decoded)

    def test_a_count_model_says_the_sentence_back(self):
        model = new_model("count", seed=1, encoding=B64)
        model.train(self.TRAIN, epochs=3)
        result = model.predict(B64.prefix_input("the cat sat on the"), length=8, to_end=True)
        self.assertTrue(result.reached_end)
        decoded = decoded_prediction(B64, result.full_text, b"the cat sat on the")
        self.assertIn(decoded["decoded"], ("the cat sat on the mat", "the cat sat on the floor"))

    def test_a_binary_file_comes_back_byte_for_byte(self):
        """From START a model of one file walks that file: the decoded walk is the file again."""
        model = new_model("count", seed=1, encoding=B64)
        model.train([base64_text(BLOB)], epochs=2)
        result = model.predict("", length=1, to_end=True)
        self.assertEqual(base64_bytes(result.full_text), BLOB)

    def test_a_score_reads_the_bytes(self):
        model = new_model("count", seed=1, encoding=B64)
        model.train(self.TRAIN, epochs=2)
        known = model.score(B64.text_input(TEXTS[0]))
        noise = model.score(B64.text_input("qzx wqzj vbn qzx wqzj vbn"))
        self.assertEqual(known["unknown_transitions"], 0)
        self.assertGreater(noise["unknown_transitions"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
