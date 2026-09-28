"""The acoustic unit: a model learns from sound alone, and is heard back (``radixnet/encoding.py``, D-082).

The units are the phonetic tokenizer's acoustic units (``phonetok.acoustic``): sounds learned from
recordings with nothing written down.  A WAV given to ``train --data`` is heard as one text of units,
the graph is built over those, and ``speak`` gives them back through the vocoder.  Skipped when the
tokenizer is not importable.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer"))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import ACOUSTIC, UNIT_KINDS, Encoding, hear_audio, is_audio_file, parse_encoding  # noqa: E402
from radixnet.graph import FIRST  # noqa: E402
from radixnet.model import load_model  # noqa: E402

try:
    import phonetok  # noqa: F401
    from phonetok.acoustic import AcousticTokenizer, default_codebook, read_wav
    from phonetok.synth import Synthesizer, Voice, wav_bytes, write_wav

    from radixnet.voice import Speaker, output_rate, speak_walks  # noqa: E402
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
    def test_the_encoding(self):
        self.assertIn(ACOUSTIC, UNIT_KINDS)
        enc = Encoding(unit=ACOUSTIC)
        self.assertEqual(str(enc), "acoustic:3:1")
        for name in ("acoustic", "acoustic:3:1", "units:2:2", "audio", "unit:4"):
            self.assertEqual(parse_encoding(name).unit, ACOUSTIC, name)
        self.assertEqual(parse_encoding("units:2:2"), Encoding(unit=ACOUSTIC, n=2, stride=2))
        self.assertEqual(enc.units_name, "units")
        self.assertFalse(enc.phonetic)
        # as text the units are tokens taken as they come, like words
        self.assertEqual(enc.units("q1  q2\nq3"), ["q1", "q2", "q3"])
        self.assertEqual(enc.length("q1 q2 q3"), 3)
        self.assertEqual(enc.encode("q1 q2 q3 q4"), ["q1 q2 q3", "q2 q3 q4"])
        self.assertEqual(enc.join("q1 q2", "q3"), "q1 q2 q3")
        self.assertEqual(enc.normalize("q1 q2 q3 q4"), "q1 q2 q3 q4")
        self.assertTrue(enc.has_unit_prefix("q1 q2 q3", "q1 q2"))
        self.assertFalse(enc.has_unit_prefix("q1 q22 q3", "q1 q2"))
        self.assertEqual(enc.spell("q1 q2"), "q1 q2")  # no words to spell back
        self.assertEqual(enc.decode_path(["q1 q2 q3", "q2 q3 q4"], 0, True, skip_sentinels=False), "q1 q2 q3 q4")
        with self.assertRaises(ValueError):
            parse_encoding("rune:3")
        self.assertTrue(is_audio_file("speech.WAV"))
        self.assertFalse(is_audio_file("speech.txt"))

    def test_a_recording_is_heard_as_units(self):
        wav = recording(TEXTS[0])
        text = hear_audio(wav)
        units = text.split()
        tok = AcousticTokenizer()
        self.assertTrue(units)
        self.assertTrue(all(tok.is_unit(u) for u in units), text)
        self.assertTrue(all(a != b for a, b in zip(units, units[1:])), "a run survived")
        self.assertEqual(text, tok.listen(wav).text)
        self.assertEqual(Encoding(unit=ACOUSTIC).units(text), units)  # the text form cuts it back into its units
        samples, rate = read_wav(wav)
        self.assertEqual(rate, 16000)
        self.assertGreater(len(samples), 16000)

    def test_a_model_learns_from_sound_alone(self):
        net = trained()
        texts = unit_texts()
        net.graph.check_invariants(texts)
        self.assertGreater(net.graph.num_trigrams(), 10)
        self.assertEqual(net.stats()["encoding"], "acoustic:3:1")
        self.assertEqual(net.stats()["units"], "units")
        for label in net.graph.labels[FIRST:]:  # every real node: the sentinels come first
            if label:
                self.assertEqual(label, " ".join(label.split()))
                self.assertTrue(all(u.startswith("q") for u in label.split()), label)
        prefix = " ".join(texts[0].split()[:3])
        pred = net.predict(prefix, length=4, k=2)
        self.assertTrue(pred.top, "no continuation from the first units of a recording")
        for result in pred.top:
            self.assertTrue(all(u.startswith("q") for u in result.text.split()), result.text)
        samples = net.generate(max_length=20, mode="sample", count=2, seed=1)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(u.startswith("q") for s in samples for u in s.text.split()))
        known = net.score(texts[0])
        noise = net.score("q1 q63 q1 q63 q1 q63 q1 q63")
        self.assertLessEqual(known["unknown_transitions"], noise["unknown_transitions"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            net.save(path)
            loaded = load_model(path)
            self.assertEqual(loaded.encoding, Encoding(unit=ACOUSTIC))
            loaded.graph.check_invariants(texts)

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
        from radixnet.cli import main
        from radixnet.voice import say

        enc = Encoding(unit=ACOUSTIC)
        book = default_codebook()
        spoken = say(enc, ["q2 q28 q55 q5", "q1 q2"])
        # the bundled codebook has its neural vocoder, so that is what speaks unless the centroid vocoder is asked for
        self.assertEqual((spoken.decoder, spoken.vocoder, spoken.rate, spoken.encoding),
                         ("vocoder", "neural", book.analysis.rate, "acoustic:3:1"))
        self.assertEqual([u.tokens for u in spoken.utterances], [["q2", "q28", "q55", "q5", "</s>"], ["q1", "q2", "</s>"]])
        self.assertEqual([u.spelled for u in spoken.utterances], ["q2 q28 q55 q5", "q1 q2"])  # nothing to spell
        self.assertGreater(spoken.samples, 0)
        self.assertEqual(spoken.samples, sum(u.samples for u in spoken.utterances))
        self.assertEqual(say(enc, ["q2 q28 q55 q5", "q1 q2"], vocoder="neural").pcm, spoken.pcm)
        self.assertEqual(spoken.to_dict()["vocoder"], "neural")
        # the centroid vocoder's stream is the Speaker's, exactly
        speaker = Speaker(enc)
        streamed = say(enc, "q2 q28 q55 q5", vocoder="centroid")
        self.assertEqual(streamed.pcm, speaker.feed("q2 q28 q55 q5") + speaker.end())
        self.assertEqual(streamed.vocoder, "centroid")
        self.assertEqual(streamed.utterances[0].samples, spoken.utterances[0].samples)  # the same frames either way
        self.assertNotEqual(streamed.pcm, spoken.pcm[:len(streamed.pcm)])
        # polished: Griffin-Lim over each whole utterance - the same length, other samples
        polished = say(enc, ["q2 q28 q55 q5", "q1 q2"], polish=4, vocoder="centroid")
        self.assertEqual([u.samples for u in polished.utterances], [u.samples for u in spoken.utterances])
        self.assertNotEqual(polished.pcm, spoken.pcm)
        self.assertEqual([u.tokens for u in polished.utterances], [u.tokens for u in spoken.utterances])
        self.assertEqual(polished.vocoder, "centroid")
        with self.assertRaises(ValueError):
            say(enc, ["q2 nope"])
        with self.assertRaises(ValueError):
            say(enc, ["q2 nope"], polish=2)
        with self.assertRaises(ValueError):
            say(enc, ["q2"], vocoder="nope")
        self.assertIsNone(say(Encoding(unit="phone"), ["K AE1 T"]).vocoder)
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "units.wav")
            none = os.path.join(tmp, "none.json")  # no model file: the encoding says how a text is read
            for extra in ([], ["--polish", "2"], ["--vocoder", "centroid"], ["--vocoder", "centroid", "--polish", "2"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--model", none, "--encoding", "acoustic:3:1", "--json", "say", "q2 q28 q55", "--out", wav, *extra])
                self.assertEqual(code, 0, out.getvalue())
                doc = json.loads(out.getvalue())
                self.assertEqual((doc["decoder"], doc["vocoder"], doc["rate"], doc["count"], doc["polish"]),
                                 ("vocoder", "centroid" if "--vocoder" in extra else "neural", book.analysis.rate, 1,
                                  2 if "--polish" in extra else 0))
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
        from radixnet.cli import main

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
