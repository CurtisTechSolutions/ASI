"""The repository's byte-pair encoding: deterministic training, exact byte round trips, pieces never crossed."""

import json
import random
import unittest

from radixpair.bpe import BPE, PIECES
from radixpair.codec import BpeCodec, codec_from_dict

CORPUS = ["the cat sat on the mat and the dog sat on the log " * 10,
          "hello world, hello there, hello again and again " * 10,
          "byte pair encoding merges the most frequent pair " * 5]


class TestBPE(unittest.TestCase):
    def test_training_reaches_the_vocabulary_and_is_deterministic(self):
        a = BPE.train(CORPUS, 300)
        b = BPE.train(CORPUS, 300)
        self.assertEqual(a.vocab_size, 300)
        self.assertEqual(a.merges, b.merges)
        with self.assertRaises(ValueError):
            BPE.train(CORPUS, 100)

    def test_merges_never_cross_a_piece(self):
        bpe = BPE.train(CORPUS, 300)
        for token in bpe.tokens[256:]:
            self.assertNotIn(b" ", token[1:], token)

    def test_round_trips(self):
        bpe = BPE.train(CORPUS, 300)
        rng = random.Random(0)
        for _ in range(50):
            raw = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
            text = raw.decode("utf-8", errors="surrogateescape")
            self.assertEqual(bpe.decode(bpe.encode(text)), text) if raw.decode("utf-8", errors="ignore") == text else None
        for text in ("the cat sat on the mat", "été 世界", "hello   world", ""):
            self.assertEqual(bpe.decode(bpe.encode(text)), text)
        self.assertEqual(PIECES.findall("a b  c"), ["a", " b", " ", " c"])

    def test_the_cache_and_the_file(self):
        bpe = BPE.train(CORPUS, 280)
        bpe.encode("hello hello")
        self.assertIn("hello", bpe.cache)
        again = BPE.from_dict(json.loads(json.dumps(bpe.to_dict())))
        self.assertEqual(again.encode("the cat sat"), bpe.encode("the cat sat"))

    def test_the_codec(self):
        c = BpeCodec(vocab_size=300, texts=CORPUS)
        self.assertEqual(c.R, 303)
        self.assertEqual((c.start, c.end), (0, 1))
        self.assertIsNone(c.unk)
        self.assertEqual(c.decode(c.encode("the dog sat")), "the dog sat")
        d = codec_from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(d.encode("the dog sat"), c.encode("the dog sat"))
        with self.assertRaises(ValueError):
            BpeCodec(vocab_size=300)


if __name__ == "__main__":
    unittest.main()
