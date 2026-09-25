"""The voice: a walk is heard as it goes, and the END sentinel closes each utterance (``radixnet/voice.py``)."""

import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer"))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import PHONES, WORDS, Encoding  # noqa: E402
from radixnet.graph import END  # noqa: E402
from radixnet.search import sample_walk  # noqa: E402

try:
    import phonetok  # noqa: F401
    from radixnet.voice import Speaker, speak_walks  # noqa: E402
except ImportError:  # pragma: no cover
    phonetok = None

TEXTS = ["the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "a bird in the hand"]


def trained(encoding):
    net = CountRewardNet(seed=1, encoding=encoding)
    net.train(TEXTS, epochs=2)
    return net


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestTheWalkIsHeard(unittest.TestCase):
    def test_the_walk_reports_every_step_and_its_sentinel(self):
        net = trained(Encoding(unit=PHONES))
        steps = []
        walk = sample_walk(net.graph, 0, 0, max_chars=None, rng=__import__("random").Random(1), on_step=steps.append)
        self.assertEqual(steps, walk.node_ids[1:])
        self.assertEqual(steps[-1], END)  # the final sentinel is the last step reported
        self.assertTrue(walk.reached_end)

    def test_a_model_of_sounds_speaks_as_it_walks(self):
        net = trained(Encoding(unit=PHONES))
        chunks = list(speak_walks(net, prefix="the cat", count=2, max_length=30, seed=1))
        self.assertGreater(len(chunks), 2)  # at least a word per utterance, and the ends
        pcm = b"".join(chunks)
        self.assertGreater(len(pcm) / 2 / 16000, 1.0)
        peak = max(abs(x) for x in struct.unpack(f"<{len(pcm) // 2}h", pcm))
        self.assertGreater(peak, 5000)
        self.assertLess(peak, 32767)
        # the same seed says the same thing
        self.assertEqual(b"".join(speak_walks(net, prefix="the cat", count=2, max_length=30, seed=1)), pcm)
        said = []
        list(speak_walks(net, prefix="the cat", count=1, max_length=30, seed=1,
                         on_utterance=lambda i, text, spelled: said.append((text, spelled))))
        self.assertEqual(len(said), 1)
        self.assertTrue(said[0][0].startswith("DH AH0 # K AE1 T"), said[0])
        self.assertTrue(said[0][1].startswith("the cat"), said[0])

    def test_what_is_spoken_is_what_the_walk_says(self):
        """The utterance is the walk's text: from START (the first node whole) and from a prefix."""
        for enc in (Encoding(unit=PHONES), Encoding(), Encoding(unit=WORDS, n=2)):
            net = trained(enc)
            for prefix in ("", "the cat"):
                for seed in (1, 2, 3):
                    with self.subTest(encoding=str(enc), prefix=prefix, seed=seed):
                        said = []
                        list(speak_walks(net, prefix=prefix, count=2, max_length=40, seed=seed,
                                         on_utterance=lambda i, text, spelled: said.append(text)))
                        walks = net.generate(max_length=40, mode="sample", count=2, seed=seed, prefix=prefix)
                        self.assertEqual(len(said), 2)
                        for spoken, walk in zip(said, walks):
                            if prefix:
                                self.assertTrue(spoken.startswith(walk.text), (spoken, walk.text))
                            else:
                                self.assertEqual(enc.truncate(spoken, 40), walk.text)

    def test_words_and_letters_are_read_through_the_tokenizer(self):
        for enc in (Encoding(), Encoding(unit=WORDS, n=2)):
            with self.subTest(encoding=str(enc)):
                net = trained(enc)
                pcm = b"".join(speak_walks(net, prefix="the cat", count=1, max_length=20, seed=1))
                self.assertGreater(len(pcm) / 2 / 16000, 0.5)

    def test_the_speaker(self):
        speaker = Speaker(Encoding(unit=PHONES))
        self.assertEqual(speaker.feed("DH AH0"), b"")   # a word in progress
        self.assertNotEqual(speaker.feed("# K AE1 T"), b"")  # the boundary commits "the"
        closing = speaker.end()                          # the sentinel commits "cat"
        self.assertGreater(len(closing), 1000)
        self.assertEqual(speaker.tokens[-1], "</s>")
        letters = Speaker(Encoding())
        self.assertEqual(letters.feed("the c"), b"")     # "the" is spoken only once its word ends
        letters.feed("at.")
        self.assertIn("K", letters.tokens)
        self.assertEqual(letters.tokens[0], "DH")
        words = Speaker(Encoding(unit=WORDS, n=2))
        words.feed("the cat")
        self.assertEqual(words.tokens[:3], ["DH", "AH0", "#"])

    def test_the_command_writes_a_wav(self):
        from radixnet.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            wav = os.path.join(tmp, "s.wav")
            corpus = os.path.join(tmp, "c.txt")
            with open(corpus, "w", encoding="utf-8") as f:
                f.write("\n".join(TEXTS) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                main(["--model", model, "--kind", "count", "--encoding", "phone:3:1", "--seed", "1", "train", "--data", corpus, "--epochs", "1"])
                code = main(["--model", model, "--seed", "1", "speak", "--prefix", "the cat", "--count", "2", "--out", wav])
            self.assertEqual(code, 0)
            self.assertIn("utterances", out.getvalue())
            with wave.open(wav) as w:
                self.assertGreater(w.getnframes(), 16000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
