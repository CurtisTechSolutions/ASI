"""Tests for audioimage.spectrogram - the plane mapping."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage import dsp  # noqa: E402
from audioimage.spectrogram import (  # noqa: E402
    FREQ_SCALES,
    SCALES,
    PlaneConfig,
    Spectrogram,
    from_plane,
    hz_to_mel,
    mel_to_hz,
    row_frequencies,
    to_plane,
)

RATE = 16000


def spectrogram(freq=440.0, seconds=0.5, n_fft=512, hop=128):
    n = int(RATE * seconds)
    sig = [0.6 * math.sin(2 * math.pi * freq * t / RATE) for t in range(n)]
    return Spectrogram(dsp.magnitudes(dsp.stft(sig, n_fft, hop)), RATE, n_fft, hop)


def relative_error(a, b):
    num = sum((x - y) ** 2 for ra, rb in zip(a.mags, b.mags) for x, y in zip(ra, rb)) ** 0.5
    den = sum(x * x for row in a.mags for x in row) ** 0.5
    return num / den if den else 0.0


class TestSpectrogram(unittest.TestCase):
    def test_properties(self):
        spec = spectrogram()
        self.assertEqual(spec.bins, 257)
        self.assertEqual(spec.nyquist, 8000)
        self.assertGreater(spec.peak, 0)
        self.assertAlmostEqual(spec.bin_hz(spec.hz_bin(1000.0)), 1000.0, places=6)
        self.assertAlmostEqual(spec.frame_time(2), 2 * 128 / RATE)
        self.assertIn("frames", spec.describe())

    def test_empty(self):
        spec = Spectrogram([], RATE, 512, 128)
        self.assertEqual(spec.frames, 0)
        self.assertEqual(spec.bins, 257)
        self.assertEqual(spec.peak, 0.0)


class TestMel(unittest.TestCase):
    def test_round_trip(self):
        for hz in (0.0, 100.0, 440.0, 1000.0, 8000.0):
            self.assertAlmostEqual(mel_to_hz(hz_to_mel(hz)), hz, places=6)

    def test_monotonic(self):
        self.assertLess(hz_to_mel(100), hz_to_mel(1000))
        self.assertEqual(hz_to_mel(0.0), 0.0)


class TestPlaneConfig(unittest.TestCase):
    def test_resolution(self):
        spec = spectrogram()
        cfg = PlaneConfig().resolved(spec)
        self.assertEqual(cfg.height, spec.bins)
        self.assertEqual(cfg.width, spec.frames)
        self.assertEqual(cfg.f_max, spec.nyquist)
        self.assertAlmostEqual(cfg.ref, spec.peak)

    def test_silent_clip_gets_a_usable_ref(self):
        silent = Spectrogram([[0.0] * 257 for _ in range(4)], RATE, 512, 128)
        self.assertEqual(PlaneConfig().resolved(silent).ref, 1.0)

    def test_meta_round_trip(self):
        cfg = PlaneConfig(scale="sqrt", freq_scale="mel", top_db=60.0, height=128).resolved(spectrogram())
        self.assertEqual(PlaneConfig.from_meta(cfg.to_meta()), cfg)

    def test_meta_ignores_unrelated_keys(self):
        cfg = PlaneConfig.from_meta({"scale": "linear", "sample_rate": 22050, "tool": "x"})
        self.assertEqual(cfg.scale, "linear")

    def test_rejections(self):
        for kwargs in (
            {"scale": "loud"},
            {"freq_scale": "bark"},
            {"origin": "middle"},
            {"top_db": 0.0},
            {"height": 0},
            {"width": -1},
            {"ref": 0.0},
            {"f_min": -1.0},
            {"f_min": 100.0, "f_max": 50.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    PlaneConfig(**kwargs)


class TestRowFrequencies(unittest.TestCase):
    def test_linear(self):
        rows = row_frequencies(PlaneConfig(height=5, f_max=8000.0), 8000.0)
        self.assertEqual(rows, [0.0, 2000.0, 4000.0, 6000.0, 8000.0])

    def test_all_scales_are_increasing(self):
        for scale in FREQ_SCALES:
            rows = row_frequencies(PlaneConfig(freq_scale=scale, height=32, f_max=8000.0), 8000.0)
            with self.subTest(scale=scale):
                self.assertEqual(len(rows), 32)
                self.assertTrue(all(b > a for a, b in zip(rows, rows[1:])))

    def test_circle_axis_blends_the_ends_and_expands_the_middle(self):
        rows = row_frequencies(PlaneConfig(freq_scale="circle", height=65, f_max=8000.0), 8000.0)
        gaps = [b - a for a, b in zip(rows, rows[1:])]
        self.assertTrue(all(g > 0 for g in gaps))            # still a frequency axis
        self.assertGreater(gaps[0], gaps[32] * 3)            # the bottom is blended
        self.assertGreater(gaps[-1], gaps[32] * 3)           # so is the top
        self.assertAlmostEqual(rows[0], 0.0)
        self.assertAlmostEqual(rows[-1], 8000.0, places=6)

    def test_circle_bulge_controls_how_hard(self):
        def ratio(bulge):
            rows = row_frequencies(PlaneConfig(freq_scale="circle", freq_bulge=bulge, height=65, f_max=8000.0), 8000.0)
            gaps = [b - a for a, b in zip(rows, rows[1:])]
            return gaps[0] / gaps[32]

        self.assertAlmostEqual(ratio(0.0), 1.0, places=6)     # 0 is a plain linear axis
        self.assertGreater(ratio(0.7), ratio(0.3))

    def test_circle_bulge_is_checked(self):
        with self.assertRaises(ValueError):
            PlaneConfig(freq_bulge=1.5)

    def test_circle_warp_is_monotonic(self):
        from audioimage.spectrogram import circle_warp

        values = [circle_warp(i / 40) for i in range(41)]
        self.assertAlmostEqual(values[0], 0.0, places=9)
        self.assertAlmostEqual(values[-1], 1.0, places=9)
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))

    def test_single_row(self):
        self.assertEqual(len(row_frequencies(PlaneConfig(height=1), 8000.0)), 1)


class TestPlane(unittest.TestCase):
    def test_shape_and_range(self):
        spec = spectrogram()
        plane = to_plane(spec, PlaneConfig())
        self.assertEqual(len(plane), spec.bins)
        self.assertEqual(len(plane[0]), spec.frames)
        self.assertTrue(all(0.0 <= v <= 1.0 for row in plane for v in row))

    def test_low_frequencies_sit_at_the_bottom(self):
        """origin='lower' is how a spectrogram is normally drawn."""
        spec = spectrogram(freq=440.0)
        plane = to_plane(spec, PlaneConfig(origin="lower"))
        loudest = max(range(len(plane)), key=lambda r: plane[r][3])
        self.assertGreater(loudest, len(plane) * 0.8)  # near the bottom
        flipped = to_plane(spec, PlaneConfig(origin="upper"))
        loudest_up = max(range(len(flipped)), key=lambda r: flipped[r][3])
        self.assertLess(loudest_up, len(flipped) * 0.2)  # near the top

    def test_the_loud_row_is_the_right_frequency(self):
        spec = spectrogram(freq=1000.0)
        plane = to_plane(spec, PlaneConfig())
        row = max(range(len(plane)), key=lambda r: plane[r][3])
        bin_index = len(plane) - 1 - row  # origin='lower' flips it
        self.assertAlmostEqual(spec.bin_hz(bin_index), 1000.0, delta=spec.bin_hz(1))

    def test_linear_axis_is_exactly_invertible(self):
        """On a linear frequency axis nothing is resampled, so nothing is lost.

        ``linear`` and ``sqrt`` come back to floating-point accuracy.  ``db``
        cannot: it deliberately throws away everything more than ``top_db``
        under the peak, which for a pure tone is most of the leakage floor.
        That discarded floor is the whole of the error below.
        """
        spec = spectrogram()
        for scale, tolerance in (("linear", 1e-12), ("sqrt", 1e-12), ("db", 1e-3)):
            cfg = PlaneConfig(scale=scale).resolved(spec)
            back = from_plane(to_plane(spec, cfg), cfg, RATE, 512, 128, frames=spec.frames)
            with self.subTest(scale=scale):
                self.assertLess(relative_error(spec, back), tolerance)
        self.assertEqual(set(SCALES), {"linear", "sqrt", "db"})  # a new scale needs a tolerance here

    def test_db_keeps_everything_inside_its_window(self):
        """Whatever is within top_db of the peak survives the round trip closely."""
        spec = spectrogram()
        cfg = PlaneConfig(scale="db", top_db=80.0).resolved(spec)
        back = from_plane(to_plane(spec, cfg), cfg, RATE, 512, 128, frames=spec.frames)
        floor = spec.peak * 10 ** (-80.0 / 20.0)
        for want_row, got_row in zip(spec.mags, back.mags):
            for want, got in zip(want_row, got_row):
                if want > floor * 10:  # comfortably inside the window
                    self.assertLess(abs(got - want) / want, 1e-3)

    def test_resampled_axes_are_close(self):
        spec = spectrogram()
        for freq_scale in ("log", "mel"):
            cfg = PlaneConfig(freq_scale=freq_scale).resolved(spec)
            back = from_plane(to_plane(spec, cfg), cfg, RATE, 512, 128, frames=spec.frames)
            with self.subTest(freq_scale=freq_scale):
                self.assertLess(relative_error(spec, back), 0.35)

    def test_explicit_size(self):
        spec = spectrogram()
        cfg = PlaneConfig(height=64, width=40).resolved(spec)
        plane = to_plane(spec, cfg)
        self.assertEqual(len(plane), 64)
        self.assertEqual(len(plane[0]), 40)
        back = from_plane(plane, cfg, RATE, 512, 128, frames=spec.frames)
        self.assertEqual(back.frames, spec.frames)
        self.assertEqual(back.bins, spec.bins)

    def test_silence_is_white(self):
        silent = Spectrogram([[0.0] * 257 for _ in range(4)], RATE, 512, 128)
        plane = to_plane(silent, PlaneConfig())
        self.assertTrue(all(v == 0.0 for row in plane for v in row))

    def test_the_floor_decodes_to_silence(self):
        """Intensity 0 must be no sound at all, not a tone 80 dB down."""
        cfg = PlaneConfig(scale="db").resolved(spectrogram())
        back = from_plane([[0.0] * 4 for _ in range(257)], cfg, RATE, 512, 128, frames=4)
        self.assertTrue(all(v == 0.0 for row in back.mags for v in row))

    def test_peak_is_black(self):
        spec = spectrogram()
        cfg = PlaneConfig().resolved(spec)
        self.assertAlmostEqual(max(max(r) for r in to_plane(spec, cfg)), 1.0, places=9)

    def test_empty_plane(self):
        spec = Spectrogram([], RATE, 512, 128)
        plane = to_plane(spec, PlaneConfig(height=8, width=3))
        self.assertEqual(len(plane), 8)
        self.assertEqual(from_plane([], PlaneConfig(), RATE, 512, 128).frames, 0)


if __name__ == "__main__":
    unittest.main()
