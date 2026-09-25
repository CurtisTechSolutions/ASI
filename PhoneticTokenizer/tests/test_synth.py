"""Tests for phonetok.synth: the formant synthesizer, its stream and its files."""

import io
import math
import os
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.lexicon import Lexicon  # noqa: E402
from phonetok.phones import PHONEMES  # noqa: E402
from phonetok.synth import (  # noqa: E402
    RATE, Synthesizer, Voice, duration, load_voice, voice, wav_bytes, wav_header, write_wav,
)
from phonetok.tokenizer import PhoneticTokenizer  # noqa: E402


def samples(pcm):
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)


class TestTheVoiceTable(unittest.TestCase):
    def test_every_phoneme_has_a_row(self):
        table = voice()
        self.assertEqual(set(table), set(PHONEMES))
        for phone, sound in table.items():
            self.assertGreater(sound.ms, 0, phone)
            self.assertTrue(all(f > 0 for f in sound.f), phone)
            self.assertTrue(all(0 < b < 1000 for b in sound.bw), phone)
            self.assertTrue(0 <= sound.amp <= 1 and 0 <= sound.noise_amp <= 1, phone)
        self.assertIsNotNone(table["AY"].f2)
        self.assertIsNone(table["AA"].f2)
        self.assertEqual(load_voice(), table)


class TestSpeaking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = PhoneticTokenizer(lexicon=Lexicon.core())

    def test_a_sentence_is_audio(self):
        pcm = Synthesizer().speak(self.tok.tokens("The cat sat on the mat."))
        self.assertGreater(duration(pcm), 1.0)
        self.assertLess(duration(pcm), 3.0)
        s = samples(pcm)
        peak = max(abs(x) for x in s)
        self.assertGreater(peak, 8000)   # audible
        self.assertLess(peak, 32767)     # not clipped
        rms = math.sqrt(sum(x * x for x in s) / len(s))
        self.assertGreater(rms, 300)
        self.assertEqual(s[0], 0)        # a lead-in of silence
        self.assertEqual(s[-1], 0)       # and a close

    def test_deterministic(self):
        tokens = self.tok.tokens("Strengths, butter? Aah!")
        self.assertEqual(Synthesizer().speak(tokens), Synthesizer().speak(tokens))

    def test_the_stream_is_the_whole(self):
        tokens = self.tok.tokens("The quick brown fox jumps over the lazy dog.")
        whole = Synthesizer().speak(tokens)
        chunks = list(Synthesizer().stream(tokens))
        self.assertGreater(len(chunks), 5)  # a chunk per word, roughly
        self.assertEqual(b"".join(chunks), whole)
        # fed one token at a time, a word comes out when the token after it arrives
        synth = Synthesizer()
        got = [synth.feed(t) for t in tokens]
        self.assertEqual(got[0], b"")  # DH: nothing yet
        self.assertEqual(got[1], b"")  # AH0: still the same word
        self.assertNotEqual(got[2], b"")  # #: "the" is committed
        self.assertEqual(b"".join(got) + synth.end(), whole)

    def test_the_final_sentinel_ends_the_utterance(self):
        synth = Synthesizer()
        for t in self.tok.tokens("hello world"):
            synth.feed(t)
        closing = synth.feed("</s>")
        self.assertGreater(len(closing), 0)          # "world" and the closing silence
        self.assertEqual(synth.feed("</s>"), b"")    # nothing pending: nothing more
        self.assertEqual(synth.end(), b"")
        # a full stop and a question end differently: the last syllable falls, or rises
        stop = Synthesizer().speak(self.tok.tokens("the cat sat."))
        question = Synthesizer().speak(self.tok.tokens("the cat sat?"))
        self.assertNotEqual(stop, question)
        self.assertEqual(len(stop), len(question))  # the same pause length, a different tune

    def test_levels_and_specials(self):
        s = Synthesizer()
        self.assertEqual(s.feed("<s>"), b"")
        self.assertEqual(s.feed("not a token"), b"")
        syl = Synthesizer().speak(PhoneticTokenizer(level="syllable", lexicon=Lexicon.core()).tokens("butter cup"))
        pho = Synthesizer().speak(self.tok.tokens("butter cup"))
        self.assertEqual(syl, pho)  # a syllable is its phones
        bare = Synthesizer().speak(["K", "AE", "T"])  # a vowel without stress still speaks
        self.assertGreater(duration(bare), 0.1)
        self.assertEqual(Synthesizer().speak([]), b"")

    def test_voice_settings(self):
        tokens = self.tok.tokens("the cat sat")
        quick = Synthesizer(voice_settings=Voice(tempo=2.0)).speak(tokens)
        slow = Synthesizer(voice_settings=Voice(tempo=0.5)).speak(tokens)
        self.assertLess(len(quick), len(slow))
        quiet = Synthesizer(voice_settings=Voice(gain=0.1)).speak(tokens)
        self.assertLess(max(abs(x) for x in samples(quiet)), max(abs(x) for x in samples(Synthesizer().speak(tokens))))
        low = Synthesizer(rate=8000).speak(tokens)
        self.assertAlmostEqual(duration(low, 8000), duration(Synthesizer().speak(tokens)), places=1)

    def test_files(self):
        pcm = Synthesizer().speak(self.tok.tokens("hi"))
        data = wav_bytes(pcm)
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()), (1, 2, RATE, len(pcm) // 2))
        header = wav_header(RATE)
        self.assertEqual(len(header), 44)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(wav_header(RATE, len(pcm) // 2) + pcm, data)  # the sized header is the file's
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hi.wav")
            write_wav(path, pcm)
            with wave.open(path) as w:
                self.assertEqual(w.readframes(w.getnframes()), pcm)


class TestTheCommand(unittest.TestCase):
    def test_say_writes_a_wav(self):
        import contextlib

        from phonetok.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["say", "--core", "--out", path, "The cat sat."])
            self.assertEqual(code, 0)
            self.assertIn("s of speech written to", out.getvalue())
            with wave.open(path) as w:
                self.assertGreater(w.getnframes(), RATE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
