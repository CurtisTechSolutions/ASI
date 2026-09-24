"""Every encoding, end to end: the graph, the four model kinds, and the file.

:mod:`tests.test_encoding` covers the encoder and decoder on their own; this
drives the whole stack with each of them - trains, walks, compresses, predicts,
scores, saves and reloads - so that "n-grams of any size, groups of letters and
words" is a claim about the model and not only about the encoder.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import CHARS, PHONES, SYLLABLES, WORDS, Encoding  # noqa: E402
from radixnet.graph import RadixCyclicGraph  # noqa: E402
from radixnet.model import load_model, new_model  # noqa: E402

EVERY_ENCODING = [
    Encoding(),                                  # the trigram: the default
    Encoding(n=1), Encoding(n=2), Encoding(n=5),  # any n, sliding
    Encoding(n=4, stride=4), Encoding(n=5, stride=5),  # groups of letters
    Encoding(n=6, stride=3),                     # groups that half-overlap
    Encoding(unit=WORDS, n=1), Encoding(unit=WORDS, n=2), Encoding(unit=WORDS, n=3),
    Encoding(unit=WORDS, n=2, stride=2),         # word groups
]
try:  # the sounds, when the phonetic tokenizer (../PhoneticTokenizer) is importable
    EVERY_ENCODING += [Encoding(unit=PHONES), Encoding(unit=PHONES, n=2, stride=2), Encoding(unit=SYLLABLES, n=2)]
except ValueError:  # pragma: no cover - the unit says what it needs
    pass

TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "a quick brown fox jumps over the lazy dog",
]


class TestGraphUnderEveryEncoding(unittest.TestCase):
    def test_round_trip_and_invariants(self):
        """What goes in comes back out - and the index, the overlaps and the
        gram alignment hold for every encoding, compressed or not."""
        for enc in EVERY_ENCODING:
            with self.subTest(encoding=str(enc)):
                g = RadixCyclicGraph(seed=1, encoding=enc)
                for text in TEXTS:
                    grams = enc.encode(text)
                    if grams:
                        g.observe_sequence(grams)
                g.check_invariants(TEXTS)
                g.compress()
                g.check_invariants(TEXTS, compressed=True)
                for text in TEXTS:
                    grams = enc.encode(text)
                    if not grams:
                        continue
                    path = g.node_path(grams)
                    self.assertIsNotNone(path, f"{text!r} does not walk through the graph")
                    decoded = enc.decode_path(
                        [g.labels[i] for i in path[1:-1]], 0, True, skip_sentinels=False
                    )
                    self.assertEqual(decoded, enc.normalize(text))

    def test_the_index_holds_whole_grams(self):
        for enc in EVERY_ENCODING:
            with self.subTest(encoding=str(enc)):
                g = RadixCyclicGraph(seed=1, encoding=enc)
                for text in TEXTS:
                    if enc.encode(text):
                        g.observe_sequence(enc.encode(text))
                for gram in g.trigram_index:
                    self.assertEqual(enc.length(gram), enc.n, f"{gram!r} is not {enc.n} {enc.unit}s")


class TestModelsUnderEveryEncoding(unittest.TestCase):
    def test_train_predict_and_score(self):
        for enc in EVERY_ENCODING:
            with self.subTest(encoding=str(enc)):
                net = CountRewardNet(seed=1, encoding=enc)
                net.train(TEXTS, epochs=2)
                net.graph.check_invariants(TEXTS)
                self.assertGreater(net.graph.num_trigrams(), 0)
                stats = net.stats()
                self.assertEqual(stats["encoding"], str(enc))
                self.assertEqual((stats["ngram"], stats["stride"]), (enc.n, enc.stride))
                pred = net.predict("the cat sat", length=3, k=2)
                self.assertTrue(pred.top, "no continuation")
                known = net.score(TEXTS[0])
                noise = net.score("qzx wqzj vbn qzx wqzj vbn")
                self.assertLessEqual(known["unknown_transitions"], noise["unknown_transitions"])

    def test_every_kind_takes_one(self):
        enc = Encoding(unit=WORDS, n=2)
        for kind in ("radix", "count", "negative", "resonant"):
            with self.subTest(kind=kind):
                model = new_model(kind, seed=1, encoding=enc)
                self.assertEqual(model.encoding, enc)
                self.assertEqual(model.graph.encoding, enc)
                self.assertEqual(model.stats()["encoding"], "word:2:1")

    def test_a_word_model_predicts_whole_words(self):
        net = CountRewardNet(seed=1, encoding=Encoding(unit=WORDS, n=2))
        net.train(TEXTS, epochs=3)
        corpus_words = set(" ".join(TEXTS).split())
        pred = net.predict("the cat sat", length=3, k=3)
        self.assertTrue(pred.top)
        for result in pred.top:
            for word in result.text.split():
                self.assertIn(word, corpus_words, f"predicted {word!r}, not a word of the corpus")
        for label in net.graph.labels[3:]:
            if label:
                self.assertEqual(label, " ".join(label.split()), "a label is not a clean word sequence")

    def test_a_group_encoding_has_no_overlap(self):
        enc = Encoding(n=4, stride=4)
        net = CountRewardNet(seed=1, encoding=enc)
        net.train(["abcdefghijkl", "abcdmnopijkl"], epochs=2)
        for gram in net.graph.trigram_index:
            self.assertEqual(len(gram), 4)
        for label in net.graph.labels[3:]:
            if label:
                self.assertEqual(len(label) % 4, 0, f"{label!r} is not whole groups")


class TestTheFileCarriesIt(unittest.TestCase):
    def test_save_and_load(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for enc in EVERY_ENCODING:
                with self.subTest(encoding=str(enc)):
                    net = CountRewardNet(seed=1, encoding=enc)
                    net.train(TEXTS, epochs=1)
                    path = os.path.join(tmp, str(enc).replace(":", "-") + ".json")
                    net.save(path)
                    with open(path, encoding="utf-8") as fh:
                        raw = fh.read()
                    # only a non-default encoding is written, so an ordinary file is unchanged
                    self.assertEqual('"encoding"' in raw, not enc.is_default(), str(enc))
                    loaded = load_model(path)
                    self.assertEqual(loaded.encoding, enc)
                    # a saved graph is compacted (dead nodes dropped), so the alive
                    # labels and the whole gram index are what must survive
                    alive = [lab for lab, ok in zip(net.graph.labels, net.graph.alive) if ok]
                    self.assertEqual(loaded.graph.labels, alive)
                    self.assertEqual(set(loaded.graph.trigram_index), set(net.graph.trigram_index))
                    loaded.graph.check_invariants(TEXTS)

    def test_a_default_file_says_nothing(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(g.encoding.encode("hello there"))
        self.assertNotIn("encoding", json.loads(json.dumps(g.to_dict())))


class TestCorrectionsFollowTheEncoding(unittest.TestCase):
    def test_a_word_correction_marks_whole_words(self):
        enc = Encoding(unit=WORDS, n=2)
        net = CountRewardNet(seed=1, encoding=enc)
        net.train(["the cat sat on the mat", "the cat ran on the mat"], epochs=2)
        out = net.correct("the cat sat on the mat", "the cat slept on the mat", strength=1.0)
        self.assertGreater(out["edits"], 0)
        self.assertEqual((out["wrong_chars"], out["right_chars"]), (1, 1))  # one word changed


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
