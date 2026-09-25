"""The acoustic units: audio to units with nothing written down, and back (``phonetok/acoustic.py``)."""

import cmath
import contextlib
import io
import math
import os
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok import PhoneticTokenizer, Lexicon  # noqa: E402
from phonetok.acoustic import (  # noqa: E402
    DEFAULT_ANALYSIS, DEFAULT_CODEBOOK, AcousticTokenizer, Analysis, Codebook, Vocoder, band_centres, collapse_runs,
    default_codebook, dist2, fft, frames_of, griffin_lim, hann, kmeans, learn, mel, mel_filters, mel_to_hz, pcm_samples,
    read_wav, resample, run_lengths, synthesize, unit_frames, voicing_of,
)
from phonetok.synth import Synthesizer, Voice, wav_bytes  # noqa: E402

TEXTS = ["the cat sat on the mat", "the quick brown fox jumps over the lazy dog", "practice makes perfect"]


def spoken(text: str, pitch: float = 120.0) -> list[float]:
    tok = PhoneticTokenizer(lexicon=Lexicon.core())
    return pcm_samples(Synthesizer(voice_settings=Voice(pitch=pitch)).speak(tok.tokens(text)))


_CLIPS: list[list[float]] = []


def clips() -> list[list[float]]:
    if not _CLIPS:
        _CLIPS.extend(spoken(t) for t in TEXTS)
    return _CLIPS


_BOOK: list[Codebook] = []


def small_book() -> Codebook:
    """A codebook of eight units learned from the three clips (once per run)."""
    if not _BOOK:
        _BOOK.append(learn(clips(), k=8, seed=3, iterations=30, note="test"))
    return _BOOK[0]


class TestAudioFiles(unittest.TestCase):
    def test_a_wav_round_trip(self):
        pcm = struct.pack("<6h", 0, 16384, -16384, 32767, -32768, 1)
        samples, rate = read_wav(wav_bytes(pcm, 16000))
        self.assertEqual(rate, 16000)
        self.assertEqual([round(s, 4) for s in samples], [0.0, 0.5, -0.5, 1.0, -1.0, 0.0])

    def test_every_format(self):
        def wav(tag, channels, rate, bits, body, extensible=False):
            if extensible:
                fmt = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * channels * bits // 8, channels * bits // 8,
                                  bits, 22, bits, 0) + struct.pack("<H", tag) + b"\0" * 14
            else:
                fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
            chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"LIST" + struct.pack("<I", 3) + b"abc\0"
            chunks += b"data" + struct.pack("<I", len(body)) + body
            return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks

        cases = [
            (wav(1, 1, 8000, 8, bytes([128, 255, 0])), 8000, [0.0, 0.992, -1.0]),
            (wav(1, 2, 44100, 16, struct.pack("<4h", 16384, -16384, 8192, 8192)), 44100, [0.0, 0.25]),
            (wav(1, 1, 16000, 24, b"\x00\x00\x40\x00\x00\xc0"), 16000, [0.5, -0.5]),
            (wav(1, 1, 16000, 32, struct.pack("<2i", 1 << 30, -(1 << 30))), 16000, [0.5, -0.5]),
            (wav(3, 1, 16000, 32, struct.pack("<2f", 0.25, -0.75)), 16000, [0.25, -0.75]),
            (wav(3, 1, 16000, 64, struct.pack("<2d", 0.125, -0.5)), 16000, [0.125, -0.5]),
            (wav(1, 1, 22050, 16, struct.pack("<2h", 16384, -16384), extensible=True), 22050, [0.5, -0.5]),
        ]
        for data, rate, want in cases:
            with self.subTest(rate=rate, want=want):
                samples, got_rate = read_wav(data)
                self.assertEqual(got_rate, rate)
                self.assertEqual([round(s, 3) for s in samples], want)
        with self.assertRaises(ValueError):
            read_wav(b"RIFF\0\0\0\0WAVEjunk")
        with self.assertRaises(ValueError):
            read_wav(b"not a wav at all")

    def test_resampling_keeps_the_tone(self):
        src = [math.sin(2 * math.pi * 440 * i / 8000) for i in range(8000)]
        out = resample(src, 8000, 16000)
        self.assertEqual(len(out), 16000)
        crossings = sum(1 for a, b in zip(out[1000:15000], out[1001:15001]) if (a < 0) != (b < 0))
        self.assertAlmostEqual(crossings / (14000 / 16000), 880, delta=6)  # two crossings per cycle
        self.assertGreater(max(out), 0.9)
        self.assertEqual(resample(src, 8000, 8000), src)
        self.assertEqual(len(resample(src, 8000, 6000)), 6000)


class TestTheAnalysis(unittest.TestCase):
    def test_the_fft_is_the_dft(self):
        rng = random.Random(1)
        xs = [rng.uniform(-1, 1) for _ in range(256)]
        re, im = list(xs), [0.0] * 256
        fft(re, im)
        for k in (0, 1, 5, 100, 128, 200, 255):
            want = sum(xs[t] * cmath.exp(-2j * math.pi * k * t / 256) for t in range(256))
            self.assertAlmostEqual(abs(want - complex(re[k], im[k])), 0.0, places=9)
        fft(re, im, inverse=True)
        self.assertAlmostEqual(max(abs(a - b) for a, b in zip(re, xs)), 0.0, places=12)
        self.assertAlmostEqual(sum(hann(400)) / 400, 0.5, places=12)

    def test_the_mel_filters_cover_the_band(self):
        a = DEFAULT_ANALYSIS
        filters = mel_filters(a)
        self.assertEqual(len(filters), a.bands)
        self.assertAlmostEqual(mel_to_hz(mel(1234.5)), 1234.5, places=6)
        covered = [0.0] * (a.fft // 2 + 1)
        for first, weights in filters:
            self.assertTrue(weights)
            self.assertTrue(all(0.0 < w <= 1.0 for w in weights))
            for i, w in enumerate(weights):
                covered[first + i] += w
        lo = math.ceil(a.fmin * a.fft / a.rate) + 1
        hi = math.floor(a.fmax * a.fft / a.rate) - 1
        self.assertTrue(all(covered[k] > 0.0 for k in range(lo + 1, hi)), "a bin no filter touches")
        centres = band_centres(a)
        self.assertEqual(centres, sorted(centres))
        self.assertLess(centres[0], 100)
        self.assertGreater(centres[-1], 7000)
        with self.assertRaises(ValueError):
            Analysis(fft=500).validate()
        with self.assertRaises(ValueError):
            Analysis(fmax=9000).validate()

    def test_frames(self):
        a = DEFAULT_ANALYSIS
        samples = clips()[0]
        frames = frames_of(samples, a)
        self.assertEqual(len(frames), (len(samples) - a.frame) // a.hop + 1)
        self.assertTrue(all(len(f) == a.bands for f in frames))
        for b in range(a.bands):  # normalised: every band's mean is zero
            self.assertAlmostEqual(sum(f[b] for f in frames) / len(frames), 0.0, places=9)
        raw = frames_of(samples, a, normalize=False)
        self.assertEqual(len(raw), len(frames))
        self.assertGreater(max(max(f) for f in raw), min(min(f) for f in raw) + 5)
        self.assertEqual(len(frames_of([0.0] * 10, a)), 1)  # short audio: one frame, padded
        self.assertEqual(len(frames_of([], a)), 1)
        silence = frames_of([0.0] * 4000, a, normalize=False)
        self.assertTrue(all(abs(v - math.log(1e-10)) < 1e-9 for f in silence for v in f))
        self.assertAlmostEqual(dist2([1.0, 2.0], [0.0, 0.0]), 5.0)

    def test_a_voiced_frame_and_an_unvoiced_one(self):
        a = DEFAULT_ANALYSIS
        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        synth = Synthesizer(voice_settings=Voice())
        vowel = frames_of(pcm_samples(synth.speak(["AE1", "AE1", "AE1"])), a, normalize=False)
        hiss = frames_of(pcm_samples(synth.speak(["S", "S", "S"])), a, normalize=False)
        self.assertEqual(voicing_of(vowel[len(vowel) // 2], a), 1.0)
        self.assertEqual(voicing_of(hiss[len(hiss) // 2], a), 0.0)
        self.assertTrue(tok.tokens("a"))


class TestLearning(unittest.TestCase):
    def test_kmeans_finds_separated_blobs(self):
        rng = random.Random(5)
        points = []
        for centre in ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)):
            points += [[centre[0] + rng.gauss(0, 0.5), centre[1] + rng.gauss(0, 0.5)] for _ in range(40)]
        found = kmeans(points, 3, random.Random(1), iterations=50)
        self.assertTrue(found.converged)
        self.assertLess(found.inertia, 120 * 0.5 * 0.5 * 2 * 2)
        centres = sorted(tuple(round(v) for v in c) for c in found.centroids)
        self.assertEqual(centres, [(0, 0), (0, 10), (10, 0)])
        again = kmeans(points, 3, random.Random(1), iterations=50)
        self.assertEqual(again.centroids, found.centroids)  # the same seed, the same codebook
        self.assertEqual(sorted(set(found.assignments)), [0, 1, 2])
        with self.assertRaises(ValueError):
            kmeans(points[:2], 3, random.Random(1))

    def test_learning_a_codebook_from_speech(self):
        book = small_book()
        a = book.analysis
        self.assertEqual((book.k, len(book.mean)), (8, a.bands))
        self.assertEqual(sum(book.counts), sum(len(frames_of(c, a)) for c in clips()))
        self.assertTrue(all(c > 0 for c in book.counts), "an empty unit")
        self.assertTrue(all(r >= 1.0 for r in book.runs))
        self.assertGreater(book.inertia, 0.0)
        self.assertEqual(book.names(), [f"q{i}" for i in range(8)])
        # the file carries everything, to the last digit
        text = book.dumps()
        back = Codebook.loads(text)
        self.assertEqual(back.centroids, book.centroids)
        self.assertEqual((back.runs, back.counts, back.mean, back.seed), (book.runs, book.counts, book.mean, 3))
        self.assertEqual(back.analysis, a)
        self.assertEqual(back.note, "test")
        self.assertEqual(back.dumps(), text)
        self.assertEqual(learn(clips(), k=8, seed=3, iterations=30).centroids, book.centroids)
        with self.assertRaises(ValueError):
            learn([], k=4)

    def test_unit_names(self):
        book = small_book()
        self.assertEqual(book.index("q0"), 0)
        self.assertEqual(book.index("q7"), 7)
        for bad in ("q8", "q07", "q", "7", "Q1", "q1.5", "DH", "#"):
            self.assertIsNone(book.index(bad), bad)
        self.assertEqual(collapse_runs([1, 1, 2, 2, 2, 1, 3, 3]), [1, 2, 1, 3])
        self.assertEqual(run_lengths([1, 1, 2, 2, 2, 1], 3), ([0, 3, 3], [0, 2, 1]))
        self.assertEqual(book.hold(0), max(1, int(book.runs[0] + 0.5)))


class TestHearingAndSpeakingBack(unittest.TestCase):
    def test_hearing(self):
        tok = AcousticTokenizer(small_book())
        heard = tok.listen(clips()[0])
        self.assertEqual(heard.frames, len(frames_of(clips()[0])))
        self.assertEqual(len(heard.codes), heard.frames)
        self.assertLess(len(heard.units), heard.frames)  # runs collapsed
        self.assertTrue(all(tok.is_unit(u) for u in heard.units))
        self.assertTrue(all(a != b for a, b in zip(heard.units, heard.units[1:])), "a run survived")
        self.assertAlmostEqual(heard.seconds, len(clips()[0]) / 16000)
        # the same samples through a WAV file (16-bit, exactly what the synthesizer wrote) are the same units
        wav = wav_bytes(struct.pack(f"<{len(clips()[0])}h", *[int(round(s * 32768)) for s in clips()[0]]))
        self.assertEqual(tok.hear(wav), heard.units)
        every = AcousticTokenizer(small_book(), collapse=False).hear(clips()[0])
        self.assertEqual(len(every), heard.frames)
        # a different rate is resampled: the same sounds, near enough
        slow = resample(clips()[0], 16000, 8000)
        self.assertGreater(len(set(tok.hear(slow, rate=8000)) & set(heard.units)), 3)

    def test_the_text_form(self):
        tok = AcousticTokenizer(small_book())
        self.assertEqual(tok.text("q1 q1 q2   q3\nq3"), "q1 q2 q3")
        self.assertEqual(tok.text(["q1", "q2"]), "q1 q2")
        self.assertEqual(tok.text(tok.text("q7 q0")), "q7 q0")
        self.assertEqual(tok.units_of(""), [])
        with self.assertRaises(ValueError):
            tok.text("q1 hello")
        with self.assertRaises(ValueError):
            tok.text("q9")

    def test_the_vocoder_gives_the_sounds_back(self):
        book = small_book()
        tok = AcousticTokenizer(book)
        heard = tok.listen(clips()[0])
        # frame for frame, so the timeline is the original's: the sounds heard again are mostly the same units
        voc = tok.vocoder()
        frames = [[c + m for c, m in zip(book.centroids[code], book.mean)] for code in heard.codes]
        pcm = b"".join(voc.feed_frame(f) for f in frames) + voc.end()
        self.assertEqual(len(pcm) // 2, (heard.frames - 1) * book.analysis.hop + book.analysis.frame)
        again = tok.listen(pcm_samples(pcm))
        agree = sum(1 for x, y in zip(heard.codes, again.codes) if x == y) / heard.frames
        self.assertGreater(agree, 0.6, agree)
        peak = max(abs(v) for v in struct.unpack(f"<{len(pcm) // 2}h", pcm))
        self.assertGreater(peak, 3000)
        self.assertLess(peak, 32767)
        # polished, the frames agree with each other better
        polished = griffin_lim(frames, book.analysis, iterations=4)
        self.assertEqual(len(polished), len(pcm) // 2)
        again = tok.listen(polished)
        self.assertGreater(sum(1 for x, y in zip(heard.codes, again.codes) if x == y) / heard.frames, 0.7)

    def test_units_spoken_back(self):
        book = small_book()
        tok = AcousticTokenizer(book)
        units = tok.hear(clips()[2])
        pcm = tok.synthesize(units)
        held = sum(book.hold(book.index(u)) for u in units)
        self.assertEqual(len(pcm) // 2, (held - 1) * book.analysis.hop + book.analysis.frame)
        self.assertEqual(len(unit_frames(units, book)), held)
        # streaming is the same audio, chunk by chunk
        chunks = list(tok.stream(units))
        self.assertGreater(len(chunks), len(units) // 2)
        self.assertEqual(b"".join(chunks), pcm)
        self.assertEqual(synthesize(units, book), pcm)
        # louder, higher, polished: still the right length
        self.assertEqual(len(tok.synthesize(units, gain=0.5, pitch=200.0)), len(pcm))
        self.assertEqual(len(tok.synthesize(units, polish=2)), len(pcm))
        self.assertEqual(tok.synthesize([]), b"")
        with self.assertRaises(ValueError):
            tok.synthesize(["q1", "nope"])
        with self.assertRaises(ValueError):
            Vocoder(book, pitch=0.0)
        v = Vocoder(book)
        self.assertEqual(v.end(), b"")

    def test_the_bundled_codebook(self):
        self.assertTrue(os.path.exists(DEFAULT_CODEBOOK), "no bundled codebook (tests/make_codebook.py writes it)")
        book = default_codebook()
        self.assertIs(book, default_codebook())
        self.assertEqual(book.k, 64)
        self.assertEqual(book.analysis, DEFAULT_ANALYSIS)
        self.assertIn("synthesizer", book.note)
        self.assertTrue(all(c > 0 for c in book.counts))
        tok = AcousticTokenizer()
        heard = tok.listen(clips()[1])
        self.assertGreater(len(set(heard.units)), 10)  # a sentence uses a good part of the inventory
        self.assertLess(len(heard.units), heard.frames)
        self.assertTrue(all(tok.is_unit(u) for u in heard.units))
        # the same text in another voice is heard as mostly the same units
        other = AcousticTokenizer().hear(spoken(TEXTS[1], pitch=175.0))
        shared = len(set(other) & set(heard.units)) / len(set(heard.units))
        self.assertGreater(shared, 0.5, shared)


class TestTheCommands(unittest.TestCase):
    def run_cli(self, *argv):
        from phonetok.cli import main

        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_hear_learn_replay_codebook(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            wavs = []
            for i, samples in enumerate(clips()):
                path = os.path.join(tmp, f"c{i}.wav")
                with open(path, "wb") as fh:
                    fh.write(wav_bytes(struct.pack(f"<{len(samples)}h", *[int(s * 32767) for s in samples])))
                wavs.append(path)
            book = os.path.join(tmp, "book.tsv")
            code, out, err = self.run_cli("learn", *wavs, "--units", "6", "--seed", "2", "--iterations", "20",
                                          "--out", book, "--note", "three clips")
            self.assertEqual(code, 0, err)
            self.assertIn("6 units", out)
            learned = Codebook.load(book)
            self.assertEqual(learned.k, 6)
            self.assertEqual(learned.note, "three clips")
            code, out, err = self.run_cli("hear", wavs[0], "--codebook", book)
            self.assertEqual(code, 0, err)
            units = out.split()
            self.assertTrue(units)
            self.assertTrue(all(learned.index(u) is not None for u in units))
            code, out, err = self.run_cli("hear", wavs[0], wavs[1], "--codebook", book, "--json")
            doc = json.loads(out)
            self.assertEqual([d["path"] for d in doc], wavs[:2])
            self.assertEqual(doc[0]["units"], units)
            self.assertGreater(doc[0]["frames"], len(units))
            code, out, err = self.run_cli("hear", wavs[0], "--codebook", book, "--frames")
            self.assertEqual(len(out.split()), doc[0]["frames"])
            wav = os.path.join(tmp, "back.wav")
            code, out, err = self.run_cli("replay", *units, "--codebook", book, "--out", wav, "--polish", "1")
            self.assertEqual(code, 0, err)
            self.assertIn("written to", out)
            with open(wav, "rb") as fh:
                samples, rate = read_wav(fh.read())
            self.assertEqual(rate, 16000)
            self.assertGreater(len(samples), 16000)
            code, out, err = self.run_cli("codebook", book)
            self.assertEqual(code, 0, err)
            self.assertIn("6 units", out)
            self.assertIn("three clips", out)
            code, out, err = self.run_cli("codebook")  # the bundled one
            self.assertEqual(code, 0, err)
            self.assertIn("64 units", out)
            code, out, err = self.run_cli("replay", "q1", "zz", "--codebook", book, "--out", wav)
            self.assertEqual(code, 1)
            self.assertIn("not a unit", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
