"""Tests for audioimage.synth."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage import dsp  # noqa: E402
from audioimage.synth import (  # noqa: E402
    KINDS,
    adsr,
    chirp,
    chord,
    fade,
    generate,
    harmonics,
    melody,
    mix,
    noise,
    note_to_hz,
    sine,
)
from audioimage.wav import Audio  # noqa: E402

RATE = 16000


def dominant_hz(audio, n_fft=2048):
    frames = dsp.stft(audio.samples, n_fft, n_fft // 4)
    middle = dsp.magnitudes(frames)[len(frames) // 2]
    return max(range(len(middle)), key=lambda i: middle[i]) * audio.sample_rate / n_fft


class TestNotes(unittest.TestCase):
    def test_known_notes(self):
        self.assertAlmostEqual(note_to_hz("A4"), 440.0)
        self.assertAlmostEqual(note_to_hz("A5"), 880.0)
        self.assertAlmostEqual(note_to_hz("A3"), 220.0)
        self.assertAlmostEqual(note_to_hz("C4"), 261.6256, places=3)

    def test_accidentals(self):
        self.assertAlmostEqual(note_to_hz("A#4"), note_to_hz("Bb4"))
        self.assertAlmostEqual(note_to_hz("C#5"), 554.365, places=2)

    def test_case_and_default_octave(self):
        self.assertAlmostEqual(note_to_hz("a4"), 440.0)
        self.assertAlmostEqual(note_to_hz("A"), 440.0)

    def test_a_semitone_is_a_twelfth_of_an_octave(self):
        self.assertAlmostEqual(note_to_hz("A#4") / note_to_hz("A4"), 2 ** (1 / 12))

    def test_rejections(self):
        for bad in ("", "H4", "zz"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    note_to_hz(bad)


class TestGenerators(unittest.TestCase):
    def test_sine_is_at_its_frequency(self):
        self.assertAlmostEqual(dominant_hz(sine(1000.0, 0.5, RATE)), 1000.0, delta=RATE / 2048)

    def test_lengths(self):
        self.assertEqual(len(sine(440, 0.25, RATE).samples), 4000)
        self.assertEqual(len(noise(0.25, RATE).samples), 4000)

    def test_amplitude(self):
        self.assertAlmostEqual(sine(440, 0.1, RATE, amplitude=0.3).peak, 0.3, places=3)

    def test_chirp_rises(self):
        """A sweep must be low at the start and high at the end."""
        audio = chirp(200, 4000, 1.0, RATE)
        half = len(audio.samples) // 2
        early = Audio(audio.samples[: half // 2], RATE)
        late = Audio(audio.samples[half + half // 2 :], RATE)
        self.assertLess(dominant_hz(early), dominant_hz(late))

    def test_log_chirp(self):
        audio = chirp(200, 4000, 0.5, RATE, method="log")
        self.assertEqual(len(audio.samples), 8000)
        with self.assertRaises(ValueError):
            chirp(0, 4000, 0.5, RATE, method="log")
        with self.assertRaises(ValueError):
            chirp(200, 4000, 0.5, RATE, method="sideways")

    def test_noise_kinds(self):
        white = noise(0.2, RATE, kind="white")
        pink = noise(0.2, RATE, kind="pink")
        self.assertEqual(len(white.samples), len(pink.samples))
        # pink noise puts more of its energy low down than white noise does
        def tilt(audio):
            mags = dsp.magnitudes(dsp.stft(audio.samples, 512, 128))
            low = sum(sum(row[1:20]) for row in mags)
            high = sum(sum(row[100:]) for row in mags)
            return low / max(high, 1e-12)

        self.assertGreater(tilt(pink), tilt(white))
        with self.assertRaises(ValueError):
            noise(0.1, RATE, kind="brown")

    def test_noise_is_seeded(self):
        self.assertEqual(noise(0.05, RATE, seed=3).samples, noise(0.05, RATE, seed=3).samples)
        self.assertNotEqual(noise(0.05, RATE, seed=3).samples, noise(0.05, RATE, seed=4).samples)

    def test_chord_contains_every_note(self):
        audio = chord(("C4", "E4", "G4"), 0.5, RATE)
        mags = dsp.magnitudes(dsp.stft(audio.samples, 4096, 1024))
        middle = mags[len(mags) // 2]
        for name in ("C4", "E4", "G4"):
            index = int(round(note_to_hz(name) * 4096 / RATE))
            neighbourhood = max(middle[index - 2 : index + 3])
            self.assertGreater(neighbourhood, max(middle) * 0.2)

    def test_chord_needs_notes(self):
        with self.assertRaises(ValueError):
            chord((), 0.5, RATE)

    def test_harmonics_stack(self):
        audio = harmonics(200.0, 0.5, RATE, count=4)
        mags = dsp.magnitudes(dsp.stft(audio.samples, 4096, 1024))
        middle = mags[len(mags) // 2]
        for h in range(1, 5):
            index = int(round(200.0 * h * 4096 / RATE))
            self.assertGreater(max(middle[index - 2 : index + 3]), max(middle) * 0.05)

    def test_melody_length_and_shape(self):
        audio = melody(("C4", "E4"), 0.25, RATE)
        self.assertEqual(len(audio.samples), 8000)
        self.assertLess(abs(audio.samples[0]), 1e-6)  # the envelope starts silent

    def test_melody_needs_notes(self):
        with self.assertRaises(ValueError):
            melody((), 0.25, RATE)

    def test_numeric_notes_are_allowed(self):
        self.assertAlmostEqual(dominant_hz(chord((1000.0,), 0.5, RATE)), 1000.0, delta=RATE / 2048)

    def test_adsr(self):
        env = adsr(1000, RATE)
        self.assertAlmostEqual(env[0], 0.0)
        self.assertLess(env[-1], 0.1)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in env))

    def test_fade(self):
        audio = fade(sine(440, 0.2, RATE), 0.01)
        self.assertAlmostEqual(audio.samples[0], 0.0, places=9)
        self.assertLess(abs(audio.samples[-1]), 0.05)
        # a fade longer than the clip must not blow up
        self.assertEqual(len(fade(sine(440, 0.01, RATE), 10.0).samples), 160)

    def test_mix(self):
        combined = mix(sine(440, 0.2, RATE), sine(880, 0.1, RATE))
        self.assertEqual(len(combined.samples), 3200)
        self.assertAlmostEqual(combined.peak, 0.99, places=6)
        with self.assertRaises(ValueError):
            mix(sine(440, 0.1, 8000), sine(440, 0.1, 16000))
        with self.assertRaises(ValueError):
            mix()

    def test_bad_durations(self):
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    sine(440, bad, RATE)


class TestGenerate(unittest.TestCase):
    def test_every_kind(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                audio = generate(kind, duration=0.3, rate=RATE)
                self.assertGreater(len(audio.samples), 0)
                self.assertLessEqual(audio.peak, 1.0)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            generate("didgeridoo")

    def test_notes_are_passed_through(self):
        audio = generate("chord", duration=0.4, rate=RATE, notes=("A4",))
        self.assertAlmostEqual(dominant_hz(audio), 440.0, delta=RATE / 2048)


if __name__ == "__main__":
    unittest.main()
