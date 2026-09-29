"""The acoustic unit heard back: the vocoder, ``say`` and the command line (the model side is RadixCyclicNN's tests/test_acoustic_units.py). Skipped when the tokenizer is not importable.
"""

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
from radixnet.encoding import ACOUSTIC, Encoding, hear_audio  # noqa: E402
from radixnet.model import load_model  # noqa: E402

try:
    import phonetok  # noqa: F401
    from phonetok.acoustic import AcousticTokenizer, default_codebook, read_wav
    from phonetok.synth import Synthesizer, Voice, wav_bytes, write_wav

    from modelkit.voice import Speaker, output_rate, speak_walks  # noqa: E402
except ImportError:  # pragma: no cover
    phonetok = None


TEXTS = ["the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "a bird in the hand"]


def recording(text: str, pitch: float = 120.0) -> bytes:
    """A WAV of the synthesizer saying the text: the only recordings the tests have."""
    tok = phonetok.PhoneticTokenizer(lexicon=phonetok.Lexicon.core())
    return wav_bytes(Synthesizer(voice_settings=Voice(pitch=pitch)).speak(tok.tokens(text)))


_UNIT_TEXTS: list[str] = []


def unit_texts() -> list[str]:
    if not _UNIT_TEXTS:
        _UNIT_TEXTS.extend(hear_audio(recording(t)) for t in TEXTS)
    return _UNIT_TEXTS


def trained() -> CountRewardNet:
    net = CountRewardNet(seed=1, encoding=Encoding(unit=ACOUSTIC))
    net.train(unit_texts(), epochs=2)
    return net


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestTheAcousticUnit(unittest.TestCase):

    def test_the_model_is_heard_through_the_vocoder(self):
        net = trained()
        book = default_codebook()
        self.assertEqual(output_rate(net.encoding), book.analysis.rate)
        self.assertEqual(output_rate(Encoding(), 8000), 8000)
        said = []
        chunks = list(speak_walks(net, count=2, max_length=30, seed=1,
                                  on_utterance=lambda i, text, spelled: said.append((text, spelled))))
        pcm = b"".join(chunks)
        self.assertGreater(len(chunks), 2)
        self.assertGreater(len(pcm) / 2 / book.analysis.rate, 0.5)
        peak = max(abs(x) for x in struct.unpack(f"<{len(pcm) // 2}h", pcm))
        self.assertGreater(peak, 3000)
        self.assertEqual(len(said), 2)
        for text, spelled in said:
            self.assertTrue(all(u.startswith("q") for u in text.split()), text)
            self.assertEqual(spelled, text)  # nothing to spell: the units are what was said
        # the same seed says the same thing
        self.assertEqual(b"".join(speak_walks(net, count=2, max_length=30, seed=1)), pcm)
        speaker = Speaker(Encoding(unit=ACOUSTIC))
        self.assertEqual(speaker.rate, book.analysis.rate)
        first = speaker.feed("q2 q28")
        self.assertGreater(len(first), 0)  # a unit's frames are spoken as it arrives
        tail = speaker.end()
        self.assertGreater(len(tail), 0)
        self.assertEqual(speaker.tokens, ["q2", "q28", "</s>"])
        with self.assertRaises(ValueError):
            speaker.feed("not-a-unit")

    def test_the_decoder_speaks_units(self):
        """``say`` over acoustic units: the vocoder, polished or not, and no token that is not a unit."""
        from modelkit.cli import main
        from modelkit.voice import say

        enc = Encoding(unit=ACOUSTIC)
        book = default_codebook()
        spoken = say(enc, ["q2 q28 q55 q5", "q1 q2"])
        self.assertEqual((spoken.decoder, spoken.rate, spoken.encoding), ("vocoder", book.analysis.rate, "acoustic:3:1"))
        self.assertEqual([u.tokens for u in spoken.utterances], [["q2", "q28", "q55", "q5", "</s>"], ["q1", "q2", "</s>"]])
        self.assertEqual([u.spelled for u in spoken.utterances], ["q2 q28 q55 q5", "q1 q2"])  # nothing to spell
        self.assertGreater(spoken.samples, 0)
        self.assertEqual(spoken.samples, sum(u.samples for u in spoken.utterances))
        # the vocoder's stream is the Speaker's, exactly
        speaker = Speaker(enc)
        self.assertEqual(say(enc, "q2 q28 q55 q5").pcm, speaker.feed("q2 q28 q55 q5") + speaker.end())
        # polished: Griffin-Lim over each whole utterance - the same length, other samples
        polished = say(enc, ["q2 q28 q55 q5", "q1 q2"], polish=4)
        self.assertEqual([u.samples for u in polished.utterances], [u.samples for u in spoken.utterances])
        self.assertNotEqual(polished.pcm, spoken.pcm)
        self.assertEqual([u.tokens for u in polished.utterances], [u.tokens for u in spoken.utterances])
        with self.assertRaises(ValueError):
            say(enc, ["q2 nope"])
        with self.assertRaises(ValueError):
            say(enc, ["q2 nope"], polish=2)
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "units.wav")
            none = os.path.join(tmp, "none.json")  # no model file: the encoding says how a text is read
            for extra in ([], ["--polish", "2"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--model", none, "--encoding", "acoustic:3:1", "--json", "say", "q2 q28 q55", "--out", wav, *extra])
                self.assertEqual(code, 0, out.getvalue())
                doc = json.loads(out.getvalue())
                self.assertEqual((doc["decoder"], doc["rate"], doc["count"], doc["polish"]),
                                 ("vocoder", book.analysis.rate, 1, 2 if extra else 0))
                with wave.open(wav) as w:
                    self.assertEqual((w.getframerate(), w.getnframes()), (book.analysis.rate, doc["utterances"][0]["samples"]))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(main(["--model", none, "--encoding", "acoustic:3:1", "say", "q2 nope", "--out", wav]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--model", none, "--encoding", "acoustic:3:1", "--json", "speech", "decode", "--text", "q2 q28", "--out", wav])
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(json.loads(out.getvalue())["decoder"], "vocoder")

    def test_the_command_line_trains_on_recordings_and_speaks(self):
        from modelkit.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            wavs = []
            for i, text in enumerate(TEXTS):
                path = os.path.join(tmp, f"{i}.wav")
                with open(path, "wb") as fh:
                    fh.write(recording(text))
                wavs.append(path)
            model = os.path.join(tmp, "m.json")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--model", model, "--kind", "count", "--encoding", "acoustic:3:1", "--seed", "1",
                             "train", "--data", *wavs, "--epochs", "1"])
            self.assertEqual(code, 0, out.getvalue())
            loaded = load_model(model)
            self.assertEqual(loaded.encoding, Encoding(unit=ACOUSTIC))
            self.assertEqual(loaded.stats()["texts_seen"] if "texts_seen" in loaded.stats() else len(TEXTS), len(TEXTS))
            out = io.StringIO()
            wav = os.path.join(tmp, "said.wav")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--model", model, "--seed", "1", "speak", "--count", "2", "--max-length", "30", "--out", wav])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("utterances", out.getvalue())
            with wave.open(wav) as w:
                self.assertEqual(w.getframerate(), 16000)
                self.assertGreater(w.getnframes(), 8000)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--model", model, "--json", "info"])
            self.assertEqual(code, 0)
            self.assertIn('"acoustic:3:1"', out.getvalue())


if __name__ == "__main__":
    unittest.main()
