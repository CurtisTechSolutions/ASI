"""The voice: a walk is heard as it goes, and the END sentinel closes each utterance (``modelkit/voice.py``)."""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer"))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import PHONES, WORDS, Encoding  # noqa: E402
from radixnet.graph import END  # noqa: E402
from radixnet.search import sample_walk  # noqa: E402

try:
    import phonetok  # noqa: F401
    from modelkit.voice import Speaker, speak_walks  # noqa: E402
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
        from modelkit.cli import main

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


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestTheOutputDecoder(unittest.TestCase):
    """``say``: any text a model produced is heard after the fact - through the same voice, closed by the same sentinel."""

    def test_say_speaks_any_output_in_the_models_units(self):
        from modelkit.voice import Spoken, say, speak_texts, spelled

        for enc in (Encoding(unit=PHONES), Encoding(), Encoding(unit=WORDS, n=2)):
            with self.subTest(encoding=str(enc)):
                texts = ["the cat sat", "DH AH0 # D AO1 G"]
                spoken = say(enc, texts)
                self.assertIsInstance(spoken, Spoken)
                self.assertEqual((spoken.decoder, spoken.rate, spoken.encoding), ("voice", 16000, str(enc)))
                self.assertEqual([u.text for u in spoken.utterances], texts)
                self.assertEqual(spoken.samples, sum(u.samples for u in spoken.utterances))
                self.assertGreater(spoken.seconds, 1.0)
                self.assertTrue(all(u.tokens[-1] == "</s>" for u in spoken.utterances))
                self.assertEqual(spoken.utterances[0].tokens[:4], ["DH", "AH0", "#", "K"])
                wav = spoken.wav()
                self.assertTrue(wav.startswith(b"RIFF"))
                self.assertEqual(len(wav), 44 + len(spoken.pcm))
                # the streaming form makes the same bytes and reports the same utterances
                said = []
                self.assertEqual(b"".join(speak_texts(enc, texts, on_utterance=lambda i, u: said.append(u))), spoken.pcm)
                self.assertEqual(said, spoken.utterances)
                # one text is one utterance: the Speaker fed it whole and closed by the sentinel
                speaker = Speaker(enc)
                self.assertEqual(say(enc, "the cat sat").pcm, speaker.feed("the cat sat") + speaker.end())
                doc = spoken.to_dict()
                self.assertEqual(set(doc), {"rate", "samples", "seconds", "encoding", "decoder", "vocoder", "count",
                                            "utterances"})
                self.assertEqual((doc["count"], doc["seconds"]), (2, spoken.samples / 16000))
                self.assertEqual(set(doc["utterances"][0]), {"text", "spelled", "tokens", "samples", "seconds"})
        # what a text spells: a text of sounds through the tokenizer, anything else as it is
        self.assertEqual(spelled(Encoding(unit=PHONES), "DH AH0 # K AE1 T"), "the cat")
        self.assertEqual(spelled(Encoding(unit=PHONES), "the cat"), "the cat")
        self.assertEqual(spelled(Encoding(), "DH AH0 # K AE1 T"), "DH AH0 # K AE1 T")
        self.assertEqual(say(Encoding(unit=PHONES), ["DH AH0 # D AO1 G"]).utterances[0].spelled, "the dog")

    def test_the_decoder_says_what_the_walk_was_heard_saying(self):
        """Hearing a walk as it walks and saying its text afterwards are the same audio, byte for byte."""
        from modelkit.voice import say

        for enc in (Encoding(unit=PHONES), Encoding(), Encoding(unit=WORDS, n=2)):
            with self.subTest(encoding=str(enc)):
                net = trained(enc)
                said = []
                live = b"".join(speak_walks(net, prefix="the cat", count=2, max_length=30, seed=1,
                                            on_utterance=lambda i, text, spelled: said.append(text)))
                self.assertEqual(len(said), 2)
                self.assertEqual(say(enc, said).pcm, live)

    def test_the_say_command_and_speak_on_the_outputs(self):
        from modelkit.cli import main

        def run(argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(argv)
            return code, out.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            corpus = os.path.join(tmp, "c.txt")
            with open(corpus, "w", encoding="utf-8") as f:
                f.write("\n".join(TEXTS) + "\n")
            # no model file: the encoding says how a text is read
            wav = os.path.join(tmp, "said.wav")
            code, out = run(["--model", model, "--encoding", "phone:3:1", "--json", "say", "the cat sat", "DH AH0 # D AO1 G",
                             "--out", wav])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual((doc["count"], doc["decoder"], doc["encoding"], doc["rate"], doc["sink"], doc["polish"]),
                             (2, "voice", "phone:3:1", 16000, wav, 0))
            self.assertEqual([u["spelled"] for u in doc["utterances"]], ["the cat sat", "the dog"])
            with wave.open(wav) as w:
                self.assertEqual(w.getnframes(), sum(u["samples"] for u in doc["utterances"]))
            # --data FILE: one utterance per line
            code, out = run(["--model", model, "--json", "say", "--data", corpus, "--out", wav])
            self.assertEqual(code, 0, out)
            self.assertEqual(json.loads(out)["count"], len(TEXTS))
            # nothing to say
            self.assertNotEqual(run(["--model", model, "say", "--out", wav])[0], 0)
            # --speak FILE on predict and generate: the outputs, heard
            code, out = run(["--model", model, "--kind", "count", "--encoding", "phone:3:1", "--seed", "1", "train",
                             "--data", corpus, "--epochs", "1"])
            self.assertEqual(code, 0, out)
            heard = os.path.join(tmp, "heard.wav")
            code, out = run(["--model", model, "--json", "predict", "--prefix", "the cat", "--length", "4", "--speak", heard])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual((doc["speech"]["count"], doc["speech"]["out"], doc["speech"]["decoder"]), (1, heard, "voice"))
            self.assertEqual(doc["speech"]["utterances"][0]["text"], doc["full_text"])
            with wave.open(heard) as w:
                self.assertEqual(w.getnframes(), doc["speech"]["samples"])
            code, out = run(["--model", model, "--seed", "1", "--json", "generate", "--count", "2", "--max-length", "20",
                             "--speak", heard])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual([u["text"] for u in doc["speech"]["utterances"]], [s["text"] for s in doc["samples"]])
            # a prediction without --speak carries no speech
            code, out = run(["--model", model, "--json", "predict", "--prefix", "the cat", "--length", "4"])
            self.assertNotIn("speech", json.loads(out))
            # speech decode: what is not a waveform is spoken
            code, out = run(["--model", model, "--json", "speech", "decode", "--text", "the cat sat", "--out", heard])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual((doc["decoder"], doc["count"], doc["out"]), ("voice", 1, heard))
            self.assertNotEqual(run(["--model", model, "speech", "decode", "--text", "  ", "--out", heard])[0], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
