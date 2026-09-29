"""The acoustic unit: a model learns from sound alone, and is heard back (``radixnet/encoding.py``, D-082).

The units are the phonetic tokenizer's acoustic units (``phonetok.acoustic``): sounds learned from
recordings with nothing written down.  A WAV given to ``train --data`` is heard as one text of units,
the graph is built over those, and ``speak`` gives them back through the vocoder.  Skipped when the
tokenizer is not importable.
"""

import os
import sys
import tempfile
import unittest

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
