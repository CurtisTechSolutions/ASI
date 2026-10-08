"""The encoder/decoder: every preset round-trips, agrees with its library, and says what its ids are."""

import json
import os
import tempfile
import unittest

from radixpair.codec import (CODECS, BytesCodec, CharsCodec, Gpt2Codec, PhonesCodec, SyllablesCodec, codec_from_dict,
                             make_codec)
from radixpair.gpt2 import PRE_TOKENIZER, Gpt2BPE, bytes_to_unicode

from radixpair.codec import phonetok_module

try:
    phonetok_module()
    HAVE_PHONETOK = True
except ValueError:
    HAVE_PHONETOK = False

GPT2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpt2")


def synthetic_gpt2(directory: str) -> tuple[str, str]:
    """A GPT-2-shaped vocabulary: the 256 byte tokens and a few merges, in the release's two files."""
    b2u = bytes_to_unicode()
    encoder = {b2u[b]: b for b in range(256)}
    merges = [("t", "h"), ("th", "e"), ("Ġ", "c"), ("Ġc", "a"), ("Ġca", "t"), ("Ġ", "s"), ("Ġs", "a"), ("Ġsa", "t"),
              ("Ġ", "t"), ("Ġt", "he")]
    for a, b in merges:
        encoder[a + b] = len(encoder)
    enc_path = os.path.join(directory, "encoder.json")
    bpe_path = os.path.join(directory, "vocab.bpe")
    with open(enc_path, "w", encoding="utf-8") as fh:
        json.dump(encoder, fh)
    with open(bpe_path, "w", encoding="utf-8") as fh:
        fh.write("#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in merges))
    return enc_path, bpe_path


class TestChars(unittest.TestCase):
    def test_the_alphabet(self):
        c = CharsCodec()
        self.assertEqual(c.R, 33)
        self.assertEqual((c.start, c.end, c.unk), (0, 1, 2))
        self.assertEqual(len(c.emits()), 32)
        self.assertNotIn(c.start, c.emits())
        self.assertIn(c.end, c.emits())

    def test_normalisation_and_unk(self):
        c = CharsCodec()
        self.assertEqual(c.decode(c.encode("The  Cat’s HAT")), "the cat's hat")
        self.assertEqual(c.decode(c.encode("ok? 42")), "ok? ??")
        self.assertEqual(c.encode("the ")[-1], 3)          # a trailing space is kept: id 3 is the space
        self.assertEqual(c.padded("a"), [0, 4, 1])
        self.assertAlmostEqual(c.unk_share(c.encode("ab?")), 1 / 3)

    def test_round_trip_through_the_file(self):
        c = CharsCodec()
        d = codec_from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(d.encode("hello, world."), c.encode("hello, world."))
        self.assertEqual(d.R, 33)
        with self.assertRaises(ValueError):
            CharsCodec("aa")


class TestBytes(unittest.TestCase):
    def test_bytes(self):
        b = BytesCodec()
        self.assertEqual(b.R, 259)
        self.assertIsNone(b.unk)
        self.assertEqual(len(b.emits()), 258)
        self.assertEqual(b.decode(b.encode("héllo — 世界")), "héllo — 世界")
        self.assertEqual(b.encode("A"), [3 + 65])
        self.assertEqual(codec_from_dict(b.to_dict()).R, 259)


@unittest.skipUnless(HAVE_PHONETOK, "the phonetic tokenizer is not importable")
class TestPhones(unittest.TestCase):
    def test_the_fixed_alphabet(self):
        p = PhonesCodec()
        self.assertEqual(p.R, 92)
        self.assertEqual((p.start, p.end, p.unk), (2, 3, 1))
        self.assertEqual(p.dead, frozenset({0}))
        self.assertEqual(len(p.emits()), 90)
        from phonetok import Lexicon, PhoneticTokenizer
        tok = PhoneticTokenizer(level="phoneme", lexicon=Lexicon.portable())
        text = "The cat sat on the mat."
        ids = p.encode(text)
        self.assertEqual(ids, tok.encode(text, grow=False))
        self.assertEqual(p.decode(ids), "DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T .")   # phones out
        self.assertEqual(p.spell(ids), "the cat sat on the mat.")                                   # the English they spell
        self.assertEqual(p.encode(p.decode(ids)), ids)                # the text form reads back unchanged
        self.assertEqual(p.encode("DH AH0 # K AE1 T"), p.encode("the cat"))   # phones in work too
        q = codec_from_dict(p.to_dict())
        self.assertEqual(q.encode(text), p.encode(text))
        self.assertEqual(PhonesCodec(stress=False).R, 92)


@unittest.skipUnless(HAVE_PHONETOK, "the phonetic tokenizer is not importable")
class TestSyllables(unittest.TestCase):
    def test_the_lexicon_vocabulary_is_closed_sorted_and_frozen(self):
        s = SyllablesCodec()
        self.assertEqual(s.syllables, sorted(set(s.syllables)))
        self.assertEqual(s.R, len(s.syllables) + 8)
        self.assertGreater(len(s.syllables), 1000)
        self.assertEqual((s.start, s.end, s.unk), (2, 3, 1))
        ids = s.encode("The cat sat on the mat.")
        self.assertNotIn(s.unk, ids)
        self.assertEqual(s.decode(ids), "DH.AH0 # K.AE1.T # S.AE1.T # AA1.N # DH.AH0 # M.AE1.T .")
        self.assertEqual(s.spell(ids), "the cat sat on the mat.")
        unknown = s.encode("The zyxqua mat.")
        self.assertIn(s.unk, unknown)                      # a syllable outside the vocabulary is <unk>
        self.assertEqual(s.R, SyllablesCodec().R)          # deterministic

    def test_agrees_with_the_tokenizer_token_for_token(self):
        s = SyllablesCodec()
        text = "Butter and strengths."
        tokens = s.tok.tokens(text)
        self.assertEqual([s.symbol(i) for i in s.encode(text)],
                         [t if t in s.tok.vocab else "<unk>" for t in tokens])

    def test_the_corpus_vocabulary_and_the_cap(self):
        s = SyllablesCodec(vocabulary="corpus", texts=["the cat sat on the mat", "a dog ran"])
        self.assertLess(len(s.syllables), 20)
        self.assertNotIn(s.unk, s.encode("the cat sat"))
        t = SyllablesCodec(vocabulary="corpus", texts=["the cat sat on the mat", "a dog ran"], top=3)
        self.assertEqual(len(t.syllables), 3)
        self.assertEqual(t.R, 11)
        with self.assertRaises(ValueError):
            SyllablesCodec(vocabulary="corpus")
        with self.assertRaises(ValueError):
            SyllablesCodec(vocabulary="words")

    def test_round_trip_through_the_file(self):
        s = SyllablesCodec(vocabulary="corpus", texts=["the cat sat on the mat"])
        d = codec_from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(d.syllables, s.syllables)
        self.assertEqual(d.encode("the cat sat"), s.encode("the cat sat"))
        self.assertEqual(d.R, s.R)


class TestGpt2(unittest.TestCase):
    def test_the_byte_table_is_a_bijection(self):
        table = bytes_to_unicode()
        self.assertEqual(len(table), 256)
        self.assertEqual(len(set(table.values())), 256)

    def test_the_pre_tokenizer(self):
        self.assertEqual(PRE_TOKENIZER.findall("Hello world's 123!!"), ["Hello", " world", "'s", " 123", "!!"])
        self.assertEqual(PRE_TOKENIZER.findall("a  b"), ["a", " ", " b"])

    def test_a_synthetic_vocabulary_merges_and_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            enc, bpe = synthetic_gpt2(d)
            g = Gpt2Codec(files=d)
            self.assertEqual(g.V, 266)
            self.assertEqual((g.start, g.end, g.unk, g.R), (266, 267, 268, 269))
            ids = g.encode("the cat sat")
            self.assertEqual([g.symbol(i) for i in ids], ["the", " cat", " sat"])
            self.assertEqual(g.decode(ids), "the cat sat")
            text = "the cat, été 世界!"
            self.assertEqual(g.decode(g.encode(text)), text)
            h = Gpt2Codec(files=(enc, bpe))
            self.assertEqual(h.encode("the cat"), g.encode("the cat"))
            with self.assertRaises(ValueError):
                Gpt2Codec(files=os.path.join(d, "nowhere"))

    def test_the_cap_renumbers_and_reads_the_rest_as_unk(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic_gpt2(d)
            g = Gpt2Codec(files=d, top=2, texts=["the cat cat cat sat"])
            self.assertEqual(g.R, 5)
            self.assertEqual((g.start, g.end, g.unk), (0, 1, 2))
            self.assertEqual(g.kept, [g.bpe.encoder["Ġcat"], g.bpe.encoder["the"]])
            self.assertEqual(g.encode("the cat sat"), [4, 3, 2])
            self.assertEqual(g.decode(g.encode("the cat sat")), "the cat")
            e = codec_from_dict(json.loads(json.dumps(g.to_dict())))
            self.assertEqual(e.kept, g.kept)
            self.assertEqual(e.encode("the cat sat"), g.encode("the cat sat"))
            with self.assertRaises(ValueError):
                Gpt2Codec(files=d, top=2)

    @unittest.skipUnless(os.path.isfile(os.path.join(GPT2_DIR, "encoder.json")) or os.path.isfile(os.path.join(GPT2_DIR, "vocab.json")),
                         "GPT-2's vocabulary files are not in data/gpt2")
    def test_the_real_files(self):
        g = Gpt2Codec(files=GPT2_DIR)
        self.assertEqual(g.V, 50257)
        self.assertEqual(g.R, 50260)
        text = "The quick brown fox jumps over the lazy dog. Hello world!"
        self.assertEqual(g.decode(g.encode(text)), text)
        try:
            import tiktoken  # type: ignore
        except ImportError:
            self.skipTest("tiktoken is not installed for the parity check")
        enc = tiktoken.get_encoding("gpt2")
        self.assertEqual(g.encode(text), enc.encode(text))


class TestRegistry(unittest.TestCase):
    def test_names(self):
        self.assertEqual(set(CODECS), {"chars", "bytes", "bpe", "phones", "syllables", "gpt2", "external"})
        from radixpair.codec import DEFAULT_CODEC, DEFAULT_L
        self.assertEqual(DEFAULT_CODEC, "phones")
        self.assertEqual(DEFAULT_L["phones"], 3)
        self.assertEqual(CharsCodec().spell(CharsCodec().encode("hi")), "hi")
        with self.assertRaises(ValueError):
            make_codec("words")
        with self.assertRaises(ValueError):
            codec_from_dict({"name": "words"})

    def test_external_needs_its_package(self):
        try:
            import tiktoken  # type: ignore  # noqa: F401
        except ImportError:
            with self.assertRaises(ValueError):
                make_codec("external", spec="tiktoken:gpt2")
        with self.assertRaises(ValueError):
            make_codec("external", spec="nonsense")


if __name__ == "__main__":
    unittest.main()
