"""The token unit of the encoding dial: the traditional LLM tokenizer (``radixnet/bpe.py``).

:mod:`tests.test_encoding` and :mod:`tests.test_encodings_end_to_end` drive the unit through the
encoder and the whole stack beside every other encoding; this module pins what is the tokenizer's
own - the byte alphabet, GPT-2's pre-tokenizer, the merges, ids and the exact round trip, learning,
the merges file - and the text form a graph is built over: a label cuts into the tokens it was made
of, a prefix typed as text is looked for as its tokens, and a prediction is spelled back into text.
``tests/tokens_fixture.json`` is what the Go and Rust ports are held to.
"""

import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from radixnet import bpe  # noqa: E402
from radixnet.bpe import (  # noqa: E402
    BYTE_CHARS, ENDOFTEXT, GLUE, BPETokenizer, byte_text, char_class, pretokenize, text_bytes,
)
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import CHARS, TOKENS, WORDS, Encoding, parse_encoding, word_rows  # noqa: E402

import make_tokens_fixture  # noqa: E402

TEXTS = [
    "the cat sat on the mat.",
    "the cat sat on the floor!",
    "The dog sat on the mat, and the cat walked away.",
    "a quick brown fox jumps over the lazy dog",
]

CORPUS = os.path.join(os.path.dirname(HERE), "data", "sample_corpus.txt")


def sample_corpus() -> list[str]:
    with open(CORPUS, encoding="utf-8") as fh:
        return [line for line in fh.read().split("\n") if line.strip()]


def random_text(rng: random.Random, length: int) -> str:
    """A text that is hard on a tokenizer: letters, digits, punctuation, whitespace of every kind, other scripts."""
    alphabet = ("abcdefghijklmnopqrstuvwxyz ABCXYZ 0123456789 .,;:!?'\"()-_/ \t\n\n  'sx'll 're "
                "é ü ñ ß ж λ 日本 “” — … 😀 👍🏽   　 ‍ \x00 \x1f \x7f")
    return "".join(rng.choice(alphabet) for _ in range(length))


class TestTheByteAlphabet(unittest.TestCase):
    def test_gpt2s_map(self):
        self.assertEqual(len(set(BYTE_CHARS)), 256)
        self.assertEqual(BYTE_CHARS[0x20], "Ġ")
        self.assertEqual(BYTE_CHARS[0x0A], "Ċ")
        self.assertEqual(BYTE_CHARS[0x09], "ĉ")
        self.assertEqual(BYTE_CHARS[ord("a")], "a")
        self.assertEqual(BYTE_CHARS[0xE9], "é")  # printable Latin-1 is itself
        self.assertEqual(BYTE_CHARS[0xAD], "Ń")  # the soft hyphen is not printable, and moves
        for ch in BYTE_CHARS:  # no byte is written as whitespace, and none as the glue mark
            self.assertFalse(ch.isspace(), repr(ch))
            self.assertNotEqual(ch, GLUE)

    def test_bytes_round_trip(self):
        data = bytes(range(256))
        self.assertEqual(text_bytes(byte_text(data)), data)
        self.assertEqual(byte_text(b" cat"), "Ġcat")
        self.assertIsNone(text_bytes("cat" + GLUE))


class TestThePreTokenizer(unittest.TestCase):
    def test_gpt2s_pattern(self):
        cases = {
            "Hello, world!": ["Hello", ",", " world", "!"],
            " it's": [" it", "'s"],
            "I'LL": ["I", "'LL"],  # the contractions in either case
            " 's": [" '", "s"],
            "a  b": ["a", " ", " b"],  # a run of spaces leaves its last to the word after it
            "a   ": ["a", "   "],  # at the end it keeps them all
            "\n\nfoo": ["\n", "\n", "foo"],  # a newline is not a space: the word after it takes none
            " \t\tx": [" \t", "\t", "x"],
            "x86_64 3.14": ["x", "86", "_", "64", " 3", ".", "14"],
            "“hi” — ok": ["“", "hi", "”", " —", " ok"],
            "😀👍🏽 go": ["😀👍🏽", " go"],
            "日本語。": ["日本語", "。"],
        }
        for text, want in cases.items():
            self.assertEqual(pretokenize(text), want, repr(text))

    def test_the_pieces_join_back(self):
        rng = random.Random(3)
        for _ in range(300):
            text = random_text(rng, rng.randint(0, 60))
            self.assertEqual("".join(pretokenize(text)), text, repr(text))

    def test_the_classes(self):
        self.assertEqual([char_class(c) for c in " \t　 "], ["S"] * 4)
        self.assertEqual([char_class(c) for c in "0579"], ["D"] * 4)
        self.assertEqual([char_class(c) for c in ".,'\x00\x1c“—€→😀"], ["P"] * 10)
        self.assertEqual([char_class(c) for c in "aZéж日٣"], ["L"] * 6)  # a digit of another script is a letter


class TestTheTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.bundled()
        cls.small = BPETokenizer.learn(sample_corpus(), vocab_size=320)

    def test_the_bundled_vocabulary(self):
        tok = self.tok
        self.assertEqual(tok.vocab_size, 4096)
        self.assertEqual(tok.token_id(ENDOFTEXT), 4095)
        self.assertEqual(tok.id_token(4095), ENDOFTEXT)
        self.assertEqual(tok.id_token(ord("a")), "a")  # ids 0-255 are the bytes
        self.assertIn("34a64a1", tok.note)
        self.assertEqual(tok.tokens("the cat sat"), ["Ġthe", "Ġcat", "Ġsat"])
        self.assertEqual(tok.describe()["kind"], "byte-level BPE")

    def test_every_text_round_trips_exactly(self):
        rng = random.Random(7)
        texts = TEXTS + ["", " ", "\n", "  x  ", "a<|endoftext|>b"] + [random_text(rng, rng.randint(0, 80)) for _ in range(200)]
        for tok in (self.tok, self.small, BPETokenizer()):
            for text in texts:
                ids = tok.encode(text)
                self.assertEqual(tok.decode(ids), text, repr(text))
                self.assertTrue(all(0 <= i < tok.vocab_size - len(tok.special) for i in ids))
                self.assertEqual(tok.spell(tok.text(text)), text, repr(text))

    def test_bytes_are_never_unknown(self):
        tok = BPETokenizer()  # no merges at all: every byte is a token of its own
        self.assertEqual(tok.encode("hé"), [0x20, ord("h"), 0xC3, 0xA9])
        self.assertEqual(tok.vocab_size, 257)

    def test_the_special_token(self):
        tok = self.tok
        plain = tok.encode("a<|endoftext|>b")
        special = tok.encode("a<|endoftext|>b", allowed_special=True)
        self.assertNotIn(tok.token_id(ENDOFTEXT), plain)  # read as the characters it is, unless asked
        self.assertIn(tok.token_id(ENDOFTEXT), special)
        self.assertEqual(tok.decode(special), "a<|endoftext|>b")
        self.assertEqual(tok.decode([tok.token_id(ENDOFTEXT)]), ENDOFTEXT)

    def test_a_broken_character_decodes_as_replacement(self):
        tok = self.tok
        self.assertEqual(tok.decode([0x20, 0xE2, 0x82]), "�")  # one per maximal broken piece
        self.assertEqual(tok.decode([0x20, 0xED, 0xA0, 0x80]), "�" * 3)
        with self.assertRaises(ValueError):
            tok.decode([tok.vocab_size])


class TestTheTextForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.bundled()
        cls.small = BPETokenizer.learn(sample_corpus(), vocab_size=320)

    def test_rendering(self):
        render = BPETokenizer.render
        self.assertEqual(render("Ġcat"), "cat")  # a space and printable ASCII: bare
        self.assertEqual(render("ing"), GLUE + "ing")  # no space before it: glued
        self.assertEqual(render("Ġ"), GLUE + "Ġ")  # a space alone
        self.assertEqual(render("ĠĠ"), GLUE + "ĠĠ")
        self.assertEqual(render("Ċ"), GLUE + "Ċ")
        self.assertEqual(render("ĠcafÃ©"), GLUE + "ĠcafÃ©")  # not ASCII after the space: glued, in the byte alphabet

    def test_plain_words_are_their_own_text_form(self):
        self.assertEqual(self.tok.text("the cat sat on the mat"), "the cat sat on the mat")
        self.assertEqual(self.tok.text("The walking cat."), "The walk ⁀ing cat ⁀.")

    def test_any_run_of_units_reads_back_as_itself(self):
        """A gram, a compressed label, one that starts inside a word or on a space: each reads back as the tokens it was."""
        rng = random.Random(11)
        texts = TEXTS + [random_text(rng, rng.randint(1, 50)) for _ in range(120)]
        for tok in (self.tok, self.small):
            for text in texts:
                units = tok.units(text)
                self.assertEqual(tok.units(" ".join(units)), units, repr(text))
                for lo in range(len(units)):
                    for hi in range(lo + 1, min(len(units), lo + 6) + 1):
                        run = " ".join(units[lo:hi])
                        self.assertEqual(tok.units(run), units[lo:hi], f"{text!r}: {run!r}")

    def test_text_and_units_agree_where_both_apply(self):
        """A text of single-spaced pieces that are each bare tokens reads the same as text and as units."""
        tok = self.tok
        words = [w for w in (tok.render(t) for t in tok.vocab) if not w.startswith(GLUE) and tok.read(w)]
        self.assertGreater(len(words), 1000)
        rng = random.Random(5)
        for _ in range(300):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 8)))
            self.assertEqual(tok.units(text), [tok.render(t) for t in tok.tokens(text)], text)

    def test_what_is_not_the_text_form_is_read_as_text(self):
        tok = self.tok
        self.assertEqual(tok.units("cat  sat"), ["cat", GLUE + "Ġ", "sat"])  # two spaces are text
        self.assertEqual(tok.units(" cat"), [GLUE + "Ġ", "cat"])  # so is a leading space
        self.assertEqual(tok.spell(" ".join(tok.units("cat  sat"))), "cat  sat")
        self.assertEqual(tok.units(GLUE + "Ġcat"), ["cat"])  # a glued piece of a bare token is read, and written bare
        self.assertIsNone(tok.read(GLUE + "nosuchtoken"))
        self.assertIsNone(tok.read(""))

    def test_spelling(self):
        tok = self.tok
        self.assertEqual(tok.spell("The walk ⁀ing cat ⁀."), "The walking cat.")
        self.assertEqual(tok.spell("⁀ing cat"), "ing cat")  # a run from inside a word spells from inside it
        self.assertEqual(tok.spell("⁀Ċ ⁀Ċ the"), "\n\n the")
        self.assertEqual(tok.bytes_of("the cat"), b" the cat")


class TestLearning(unittest.TestCase):
    def test_the_same_corpus_learns_the_same_merges(self):
        corpus = sample_corpus()
        a = BPETokenizer.learn(corpus, vocab_size=400)
        b = BPETokenizer.learn(list(reversed(corpus)), vocab_size=400)
        self.assertEqual(a.merges, b.merges)  # the order of the texts does not matter, ties go to the bytes
        self.assertEqual(a.vocab_size, 400)
        self.assertEqual(a.merges[0], ("Ġ", "t"))  # the commonest pair of English: a space and a t

    def test_the_most_frequent_pair_first(self):
        tok = BPETokenizer.learn(["ab ab ab cd"], vocab_size=260, min_frequency=1)
        self.assertEqual(tok.merges[:2], [("Ġ", "a"), ("Ġa", "b")])
        self.assertEqual(tok.tokens("ab ab"), ["Ġab", "Ġab"])

    def test_it_stops_when_nothing_repeats(self):
        tok = BPETokenizer.learn(["abc"], vocab_size=4096)
        self.assertEqual(tok.merges, [])  # every pair occurs once: under the minimum frequency
        self.assertEqual(tok.vocab_size, 257)
        with self.assertRaises(ValueError):
            BPETokenizer.learn(["abc"], vocab_size=100)


class TestTheMergesFile(unittest.TestCase):
    def test_round_trip(self):
        tok = BPETokenizer.learn(sample_corpus(), vocab_size=300, note="two\nlines")
        text = tok.dumps()
        self.assertTrue(text.startswith("#version: 0.2 radixnet-bpe\n#note: two\n#note: lines\n"))
        back = BPETokenizer.loads(text)
        self.assertEqual(back.merges, tok.merges)
        self.assertEqual(back.vocab, tok.vocab)
        self.assertEqual(back.note, "two\nlines")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "merges.txt")
            tok.save(path)
            self.assertEqual(BPETokenizer.load(path).merges, tok.merges)

    def test_gpt2s_own_format_reads(self):
        tok = BPETokenizer.loads("#version: 0.2\nĠ t\nĠt h\nĠth e\n")
        self.assertEqual(tok.tokens("the"), ["Ġthe"])
        self.assertEqual(tok.vocab[256:], ["Ġt", "Ġth", "Ġthe"])

    def test_bad_files_are_refused(self):
        for text in ("a", "a b c", "a  b", "Ġth e\n", "a b\na b\n", "a b\nx" + GLUE + " y\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                BPETokenizer.loads(text)

    def test_the_environment_names_another_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "merges.txt")
            BPETokenizer.learn(sample_corpus(), vocab_size=300).save(path)
            old = os.environ.get(bpe.ENV)
            os.environ[bpe.ENV] = path
            bpe.reset_default_tokenizer()
            try:
                self.assertEqual(bpe.default_tokenizer().vocab_size, 300)
                os.environ[bpe.ENV] = os.path.join(tmp, "missing.txt")
                bpe.reset_default_tokenizer()
                with self.assertRaises(ValueError):
                    Encoding(unit=TOKENS)  # refused when made, not mid-run
            finally:
                if old is None:
                    os.environ.pop(bpe.ENV, None)
                else:
                    os.environ[bpe.ENV] = old
                bpe.reset_default_tokenizer()
        self.assertEqual(bpe.default_tokenizer().vocab_size, 4096)


class TestTheUnit(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(parse_encoding("token"), Encoding(unit=TOKENS))
        self.assertEqual(parse_encoding("bpe:2:2"), Encoding(unit=TOKENS, n=2, stride=2))
        self.assertEqual(parse_encoding("subwords:4"), Encoding(unit=TOKENS, n=4))
        self.assertEqual(str(Encoding(unit=TOKENS)), "token:3:1")
        self.assertEqual(Encoding(unit=TOKENS).describe(), "3-token grams, stride 1 (sliding)")
        self.assertEqual(Encoding(unit=TOKENS).units_name, "tokens")
        self.assertTrue(Encoding(unit=TOKENS).spells)
        self.assertFalse(Encoding(unit=TOKENS).phonetic)
        self.assertFalse(Encoding().spells or Encoding(unit=WORDS).spells)
        self.assertEqual(Encoding.from_dict({"unit": "token", "n": 3, "stride": 1}), Encoding(unit=TOKENS))

    def test_the_encoder_and_decoder(self):
        enc = Encoding(unit=TOKENS)
        self.assertEqual(enc.units("The walking cat."), ["The", "walk", "⁀ing", "cat", "⁀."])
        self.assertEqual(enc.length("The walking cat."), 5)
        self.assertEqual(enc.encode("The walking cat."), ["The walk ⁀ing", "walk ⁀ing cat", "⁀ing cat ⁀."])
        self.assertEqual(enc.decode_grams(enc.encode("The walking cat.")), "The walk ⁀ing cat ⁀.")
        self.assertEqual(enc.spell(enc.normalize("The walking cat.")), "The walking cat.")
        path = ["The walk ⁀ing", "walk ⁀ing cat", "⁀ing cat ⁀."]
        self.assertEqual(enc.decode_path(path), "The walk ⁀ing cat ⁀.")
        self.assertEqual(enc.decode_path(path, include_context=False), "cat ⁀.")  # past the matched gram
        self.assertEqual(Encoding(unit=TOKENS, n=2, stride=2).encode("The walking cat."), ["The walk", "⁀ing cat"])

    def test_a_prefix_is_matched_by_its_tokens(self):
        enc = Encoding(unit=TOKENS)
        label = "the cat sat ⁀."
        self.assertTrue(enc.has_unit_prefix(label, "the cat"))
        self.assertTrue(enc.has_unit_prefix(label, "the cat sat."))  # text: its tokens are the label's
        self.assertFalse(enc.has_unit_prefix(label, "the ca"))  # "ca" is not the token "cat"
        self.assertEqual(enc.join("the cat", "sat."), "the cat sat ⁀.")

    def test_reading_backwards(self):
        enc = Encoding(unit=TOKENS)
        backwards = enc.reverse("The walking cat.")
        self.assertEqual(backwards, "⁀. cat ⁀ing walk The")
        self.assertEqual(enc.reverse(backwards), "The walk ⁀ing cat ⁀.")

    def test_a_model_over_tokens(self):
        """Trained, compressed and asked: the answer is tokens, and spells the text it continues."""
        net = CountRewardNet(seed=1, encoding=Encoding(unit=TOKENS))
        net.train(TEXTS, epochs=2)
        for label in net.graph.labels[4:]:  # every label reads back as the units it holds
            self.assertEqual(" ".join(net.encoding.units(label)), label)
        result = net.predict("the cat sat", length=4)
        self.assertTrue(result.full_text.startswith("the cat sat "), result.full_text)
        spelled = net.encoding.spell(result.full_text)
        self.assertTrue(spelled.startswith("the cat sat "), spelled)
        self.assertIn(spelled, [net.encoding.spell(net.encoding.normalize(t))[:len(spelled)] for t in TEXTS])
        rows = word_rows(net.encoding, net.graph.trigram_index)
        self.assertEqual(rows[0]["word"], "the")
        self.assertIn("⁀.", [r["word"] for r in rows])

    def test_the_file_carries_the_unit(self):
        net = CountRewardNet(seed=1, encoding=Encoding(unit=TOKENS, n=2))
        net.train(TEXTS[:2], epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            net.save(path)
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["graph"]["encoding"], {"unit": "token", "n": 2, "stride": 1})
            back = CountRewardNet.load(path)
            self.assertEqual(back.encoding, net.encoding)
            self.assertEqual(back.predict("the cat").full_text, net.predict("the cat").full_text)
        self.assertNotEqual(Encoding(unit=TOKENS).unit, CHARS)


class TestTheFixture(unittest.TestCase):
    def test_the_fixture_the_ports_read_is_up_to_date(self):
        with open(make_tokens_fixture.OUT, encoding="utf-8") as fh:
            on_disk = fh.read()
        self.assertEqual(on_disk, make_tokens_fixture.render(make_tokens_fixture.build()),
                         "tests/tokens_fixture.json is stale: python3 tests/make_tokens_fixture.py")

    def test_the_fixture_holds_its_own_promises(self):
        with open(make_tokens_fixture.OUT, encoding="utf-8") as fh:
            doc = json.load(fh)
        for side in ("bundled", "small"):
            for case in doc[side]["cases"]:
                self.assertEqual(case["units"], case["reread"], case["text"])
                self.assertEqual(case["spelled"], case["text"], case["text"])
                self.assertEqual("".join(case["pretokens"]), " " + case["text"])


if __name__ == "__main__":
    unittest.main()
