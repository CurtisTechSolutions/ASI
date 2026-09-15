"""Tests for audioimage.dsp - the hand-written transforms."""

import cmath
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage import dsp  # noqa: E402


def naive_dft(values):
    """The definition of the DFT, to check the fast one against."""
    n = len(values)
    return [sum(values[m] * cmath.exp(-2j * math.pi * k * m / n) for m in range(n)) for k in range(n)]


def noise(n, seed=0):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


class TestWindows(unittest.TestCase):
    def test_shapes(self):
        for name in dsp.WINDOWS:
            w = dsp.window_values(name, 8)
            self.assertEqual(len(w), 8)
            # blackman's first sample is 0.42 - 0.5 + 0.08, which floating point
            # makes a hair below zero rather than exactly zero
            self.assertTrue(all(-1e-12 <= v <= 1.0 + 1e-12 for v in w), f"{name}: {w}")

    def test_periodic_not_symmetric(self):
        # a periodic Hann starts at 0 and does NOT come back to 0 at the end,
        # which is what makes overlapping frames sum flat
        w = dsp.window_values("hann", 8)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertGreater(w[-1], 0.0)

    def test_circle_window(self):
        """A half circle: nothing at the ends, full weight held across the middle."""
        w = dsp.window_values("circle", 32)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertAlmostEqual(max(w), 1.0)
        self.assertGreater(w[16], 0.99)
        # it stays near full weight much longer than a cosine does
        hann = dsp.window_values("hann", 32)
        self.assertGreater(sum(w), sum(hann))
        self.assertEqual(len(w), 32)

    def test_circle_window_reconstructs(self):
        x = noise(2000, 23)
        back = dsp.istft(dsp.stft(x, 256, 64, "circle"), 256, 64, "circle", length=len(x))
        self.assertLess(max(abs(a - b) for a, b in zip(x, back)), 1e-9)

    def test_rect_is_all_ones(self):
        self.assertEqual(list(dsp.window_values("rect", 4)), [1.0] * 4)

    def test_unknown_window(self):
        with self.assertRaises(ValueError):
            dsp.window_values("triangle", 8)
        with self.assertRaises(ValueError):
            dsp.window_values("hann", 0)

    def test_cola_is_flat(self):
        # hann at a quarter hop overlaps to a constant: perfect reconstruction
        sums = dsp.cola_sum("hann", 256, 64)
        self.assertLess(max(sums) - min(sums), 1e-12)


class TestTransforms(unittest.TestCase):
    def test_fft_matches_definition(self):
        x = noise(64)
        for got, want in zip(dsp.fft(x), naive_dft(x)):
            self.assertAlmostEqual(got.real, want.real, places=9)
            self.assertAlmostEqual(got.imag, want.imag, places=9)

    def test_fft_inverse(self):
        x = noise(128, 3)
        back = dsp.ifft(dsp.fft(x))
        self.assertLess(max(abs(a - b.real) for a, b in zip(x, back)), 1e-12)

    def test_rfft_matches_fft(self):
        x = noise(64, 5)
        full = naive_dft(x)
        half = dsp.rfft(x)
        self.assertEqual(len(half), 33)
        for got, want in zip(half, full[:33]):
            self.assertAlmostEqual(abs(got - want), 0.0, places=9)

    def test_irfft_inverts_rfft(self):
        x = noise(256, 7)
        self.assertLess(max(abs(a - b) for a, b in zip(x, dsp.irfft(dsp.rfft(x), 256))), 1e-12)

    def test_irfft_ignores_impossible_phase(self):
        """DC and Nyquist cannot be complex for a real signal, so they are realified.

        A spectrum that has been through a picture carries magnitudes with made-up
        phase, including on those two bins.  The output must stay real and must
        match what a spectrum with those parts already removed would give.
        """
        spec = dsp.rfft(noise(64, 11))
        dirty = list(spec)
        dirty[0] = complex(dirty[0].real, 3.5)
        dirty[-1] = complex(dirty[-1].real, -2.25)
        clean = list(spec)
        clean[0] = complex(clean[0].real, 0.0)
        clean[-1] = complex(clean[-1].real, 0.0)
        self.assertLess(max(abs(a - b) for a, b in zip(dsp.irfft(dirty, 64), dsp.irfft(clean, 64))), 1e-12)

    def test_sizes_must_be_powers_of_two(self):
        with self.assertRaises(ValueError):
            dsp.fft(noise(30))
        with self.assertRaises(ValueError):
            dsp.rfft(noise(30))
        with self.assertRaises(ValueError):
            dsp.irfft([0j] * 16, 30)

    def test_empty(self):
        self.assertEqual(dsp.fft([]), [])
        self.assertEqual(dsp.rfft([]), [])
        self.assertEqual(dsp.irfft([]), [])

    def test_known_signal(self):
        # a sine at exactly bin 4 puts all its energy in bin 4
        n = 64
        x = [math.sin(2 * math.pi * 4 * t / n) for t in range(n)]
        mags = [abs(v) for v in dsp.rfft(x)]
        self.assertEqual(max(range(len(mags)), key=lambda i: mags[i]), 4)
        self.assertAlmostEqual(mags[4], n / 2, places=6)


class TestStft(unittest.TestCase):
    def test_perfect_reconstruction(self):
        x = noise(2000, 13)
        for window in ("hann", "hamming", "blackman"):
            for hop in (64, 128):
                with self.subTest(window=window, hop=hop):
                    frames = dsp.stft(x, 256, hop, window)
                    back = dsp.istft(frames, 256, hop, window, length=len(x))
                    self.assertEqual(len(back), len(x))
                    self.assertLess(max(abs(a - b) for a, b in zip(x, back)), 1e-9)

    def test_shape(self):
        frames = dsp.stft(noise(1000), 256, 64)
        self.assertEqual(len(frames), dsp.frame_count(1000, 256, 64))
        self.assertTrue(all(len(f) == 129 for f in frames))

    def test_every_sample_is_covered(self):
        # the last frames must reach past the end of the signal, or the tail is lost
        for length in (10, 255, 256, 257, 1000):
            with self.subTest(length=length):
                x = noise(length, length)
                back = dsp.istft(dsp.stft(x, 256, 64), 256, 64, length=length)
                self.assertLess(max(abs(a - b) for a, b in zip(x, back)), 1e-9)

    def test_no_center(self):
        x = noise(1024, 17)
        frames = dsp.stft(x, 256, 64, center=False)
        back = dsp.istft(frames, 256, 64, center=False, length=len(x))
        # the very edges have no overlap without centring; the middle is exact
        self.assertLess(max(abs(a - b) for a, b in zip(x[256:-256], back[256:-256])), 1e-9)

    def test_magnitudes(self):
        frames = dsp.stft(noise(512), 128, 32)
        mags = dsp.magnitudes(frames)
        self.assertEqual(len(mags), len(frames))
        self.assertTrue(all(v >= 0 for row in mags for v in row))

    def test_istft_of_nothing(self):
        self.assertEqual(dsp.istft([], 256, 64), [])

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            dsp.stft(noise(64), 100, 25)  # n_fft not a power of two
        with self.assertRaises(ValueError):
            dsp.stft(noise(64), 128, 0)  # hop of zero


class TestGriffinLim(unittest.TestCase):
    def _mags(self, n=2048, n_fft=256, hop=64):
        rate = 8000
        x = [0.6 * math.sin(2 * math.pi * 440 * t / rate) for t in range(n)]
        return x, dsp.magnitudes(dsp.stft(x, n_fft, hop))

    def test_converges(self):
        x, mags = self._mags()
        errors = []
        dsp.griffin_lim(mags, 256, 64, iterations=12, length=len(x), progress=lambda i, t, e: errors.append(e))
        self.assertEqual(len(errors), 12)
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.2)

    def test_more_passes_help(self):
        x, mags = self._mags()
        few = []
        many = []
        dsp.griffin_lim(mags, 256, 64, iterations=4, length=len(x), progress=lambda i, t, e: few.append(e))
        dsp.griffin_lim(mags, 256, 64, iterations=24, length=len(x), progress=lambda i, t, e: many.append(e))
        self.assertLess(many[-1], few[-1])

    def test_momentum_helps(self):
        """The over-relaxation must be damped: a raw 0.99 step diverges."""
        x, mags = self._mags()
        plain, fast = [], []
        dsp.griffin_lim(mags, 256, 64, iterations=16, momentum=0.0, length=len(x), progress=lambda i, t, e: plain.append(e))
        dsp.griffin_lim(mags, 256, 64, iterations=16, momentum=0.99, length=len(x), progress=lambda i, t, e: fast.append(e))
        self.assertLess(fast[-1], plain[-1])

    def test_recovers_a_sine(self):
        x, mags = self._mags()
        out = dsp.griffin_lim(mags, 256, 64, iterations=40, length=len(x))
        self.assertEqual(len(out), len(x))
        # the phase is not the original's, but the spectrum is
        got = dsp.magnitudes(dsp.stft(out, 256, 64))
        num = sum((a - b) ** 2 for ra, rb in zip(mags, got) for a, b in zip(ra, rb)) ** 0.5
        den = sum(a * a for ra in mags for a in ra) ** 0.5
        self.assertLess(num / den, 0.05)

    def test_deterministic(self):
        x, mags = self._mags(1024)
        a = dsp.griffin_lim(mags, 256, 64, iterations=5, seed=7, length=len(x))
        b = dsp.griffin_lim(mags, 256, 64, iterations=5, seed=7, length=len(x))
        self.assertEqual(a, b)

    def test_zero_iterations_is_the_raw_inverse(self):
        x, mags = self._mags(512)
        out = dsp.griffin_lim(mags, 256, 64, iterations=0, init="zeros", length=len(x))
        self.assertEqual(len(out), len(x))

    def test_bad_parameters(self):
        _, mags = self._mags(512)
        with self.assertRaises(ValueError):
            dsp.griffin_lim(mags, 256, 64, iterations=-1)
        with self.assertRaises(ValueError):
            dsp.griffin_lim(mags, 256, 64, momentum=2.0)
        with self.assertRaises(ValueError):
            dsp.griffin_lim(mags, 256, 64, init="guess")
        with self.assertRaises(ValueError):
            dsp.griffin_lim(mags, 512, 64)  # bins do not match n_fft

    def test_empty(self):
        self.assertEqual(dsp.griffin_lim([], 256, 64), [])


class TestResample(unittest.TestCase):
    def test_lengths(self):
        x = noise(1000, 19)
        self.assertEqual(len(dsp.resample(x, 8000, 16000)), 2000)
        self.assertEqual(len(dsp.resample(x, 8000, 4000)), 500)
        self.assertEqual(len(dsp.resample(x, 8000, 8000)), 1000)

    def test_preserves_a_constant(self):
        flat = [0.5] * 100
        self.assertTrue(all(abs(v - 0.5) < 1e-9 for v in dsp.resample(flat, 8000, 12000)))

    def test_bad_rates(self):
        with self.assertRaises(ValueError):
            dsp.resample([0.0], 0, 8000)


if __name__ == "__main__":
    unittest.main()
