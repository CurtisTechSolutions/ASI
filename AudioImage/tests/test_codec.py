"""Tests for audioimage.codec - the encode/decode round trip."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.codec import (  # noqa: E402
    META_KEY,
    EncodeConfig,
    compare,
    decode,
    decode_file,
    encode,
    encode_file,
    image_to_plane,
    plane_to_image,
    read_meta,
)
from audioimage.png import Image, read_png, write_png  # noqa: E402
from audioimage.spectrogram import PlaneConfig  # noqa: E402
from audioimage.wav import Audio, write_wav  # noqa: E402

RATE = 16000


def tone(freq=440.0, seconds=0.4, rate=RATE, harmonics=(1.0, 0.5, 0.25)):
    n = int(rate * seconds)
    return Audio(
        [sum(g * math.sin(2 * math.pi * freq * (h + 1) * t / rate) for h, g in enumerate(harmonics)) * 0.4 for t in range(n)],
        rate,
        "test",
    )


SMALL = EncodeConfig(n_fft=256, hop=64)


class TestEncodeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = EncodeConfig()
        self.assertEqual(cfg.n_fft, 1024)
        self.assertEqual(cfg.bins, 513)
        self.assertEqual(cfg.depth, 16)

    def test_rejections(self):
        for kwargs in ({"n_fft": 1000}, {"n_fft": 0}, {"hop": 0}, {"hop": 4096}, {"depth": 12}, {"colormap": "nope"}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    EncodeConfig(**kwargs)


class TestPixels(unittest.TestCase):
    def test_black_is_one(self):
        image = plane_to_image([[1.0, 0.0]], "gray", 16)
        self.assertEqual(image.get(0, 0), 0)
        self.assertEqual(image.get(1, 0), 65535)

    def test_round_trip(self):
        plane = [[i / 20 for i in range(21)] for _ in range(3)]
        back = image_to_plane(plane_to_image(plane, "gray", 16), "gray")
        for want_row, got_row in zip(plane, back):
            for want, got in zip(want_row, got_row):
                self.assertAlmostEqual(got, want, places=4)

    def test_colour_maps_round_trip(self):
        plane = [[i / 20 for i in range(21)]]
        for name in ("fire", "ice", "viridis", "magma"):
            with self.subTest(name=name):
                back = image_to_plane(plane_to_image(plane, name, 8), name)
                for want, got in zip(plane[0], back[0]):
                    self.assertAlmostEqual(got, want, delta=0.02)

    def test_colour_maps_force_8_bit_rgb(self):
        image = plane_to_image([[0.5]], "magma", 16)
        self.assertEqual((image.mode, image.depth), ("RGB", 8))

    def test_empty_plane(self):
        with self.assertRaises(Exception):
            plane_to_image([], "gray", 8)


class TestRoundTrip(unittest.TestCase):
    def test_metadata_is_written(self):
        image = encode(tone(), SMALL)
        meta = read_meta(image)
        self.assertEqual(meta["v"], 1)
        self.assertEqual(meta["sample_rate"], RATE)
        self.assertEqual(meta["n_fft"], 256)
        self.assertEqual(meta["hop"], 64)
        self.assertEqual(meta["samples"], len(tone().samples))
        self.assertIn("ref", meta)
        self.assertIn("rms", meta)
        self.assertTrue(meta["tool"].startswith("audioimage"))

    def test_metadata_is_stable(self):
        """The same clip and settings must produce byte-identical files."""
        from audioimage.png import write_png_bytes

        self.assertEqual(write_png_bytes(encode(tone(), SMALL)), write_png_bytes(encode(tone(), SMALL)))

    def test_picture_shape(self):
        image = encode(tone(), SMALL)
        self.assertEqual(image.height, SMALL.bins)
        self.assertEqual(image.mode, "L")
        self.assertEqual(image.depth, 16)

    def test_spectrum_survives(self):
        audio = tone()
        result = decode(encode(audio, SMALL), iterations=60)
        metrics = compare(audio, result.audio, SMALL)
        self.assertLess(metrics["spectral_convergence"], 0.15)
        self.assertLess(metrics["log_spectral_distance"], 3.0)

    def test_more_passes_are_better(self):
        audio = tone()
        image = encode(audio, SMALL)
        few = compare(audio, decode(image, iterations=4).audio, SMALL)["spectral_convergence"]
        many = compare(audio, decode(image, iterations=60).audio, SMALL)["spectral_convergence"]
        self.assertLess(many, few)

    def test_length_is_restored(self):
        audio = tone(seconds=0.37)
        result = decode(encode(audio, SMALL), iterations=2)
        self.assertEqual(len(result.audio.samples), len(audio.samples))
        self.assertEqual(result.audio.sample_rate, RATE)

    def test_level_is_restored(self):
        audio = tone()
        result = decode(encode(audio, SMALL), iterations=8, level="auto")
        self.assertAlmostEqual(result.audio.rms, audio.rms, places=6)

    def test_level_modes(self):
        image = encode(tone(), SMALL)
        self.assertAlmostEqual(decode(image, iterations=4, level="peak").audio.peak, 0.99, places=6)
        raw = decode(image, iterations=4, level="none").audio
        self.assertNotAlmostEqual(raw.peak, 0.99, places=6)
        with self.assertRaises(ValueError):
            decode(image, level="loud")

    def test_deterministic(self):
        image = encode(tone(), SMALL)
        a = decode(image, iterations=6, seed=5).audio.samples
        b = decode(image, iterations=6, seed=5).audio.samples
        self.assertEqual(a, b)

    def test_every_colormap_round_trips(self):
        audio = tone()
        for name in ("gray", "gray-inv", "fire", "viridis"):
            with self.subTest(name=name):
                cfg = EncodeConfig(n_fft=256, hop=64, colormap=name)
                metrics = compare(audio, decode(encode(audio, cfg), iterations=40).audio, cfg)
                self.assertLess(metrics["spectral_convergence"], 0.45)

    def test_scales_round_trip(self):
        audio = tone()
        for scale in ("db", "linear", "sqrt"):
            with self.subTest(scale=scale):
                cfg = EncodeConfig(n_fft=256, hop=64, plane=PlaneConfig(scale=scale))
                metrics = compare(audio, decode(encode(audio, cfg), iterations=40).audio, cfg)
                self.assertLess(metrics["spectral_convergence"], 0.3)

    def test_resized_plane_round_trips(self):
        audio = tone()
        cfg = EncodeConfig(n_fft=256, hop=64, plane=PlaneConfig(height=64, width=48))
        image = encode(audio, cfg)
        self.assertEqual((image.width, image.height), (48, 64))
        result = decode(image, iterations=30)
        self.assertEqual(len(result.audio.samples), len(audio.samples))

    def test_8_bit_is_worse_than_16(self):
        audio = tone()
        from audioimage.spectrogram import Spectrogram, from_plane, to_plane
        from audioimage import dsp

        spec = Spectrogram(dsp.magnitudes(dsp.stft(audio.samples, 256, 64)), RATE, 256, 64)
        cfg = PlaneConfig().resolved(spec)
        errors = {}
        for depth in (8, 16):
            back = from_plane(image_to_plane(plane_to_image(to_plane(spec, cfg), "gray", depth)), cfg, RATE, 256, 64, frames=spec.frames)
            num = sum((x - y) ** 2 for ra, rb in zip(spec.mags, back.mags) for x, y in zip(ra, rb)) ** 0.5
            den = sum(x * x for r in spec.mags for x in r) ** 0.5
            errors[depth] = num / den
        self.assertLess(errors[16], errors[8])
        self.assertLess(errors[16], 1e-3)

    def test_empty_clip_is_refused(self):
        with self.assertRaises(ValueError):
            encode(Audio([], RATE))


class TestWithoutMetadata(unittest.TestCase):
    def test_a_foreign_picture_still_decodes(self):
        """A picture drawn by hand has no settings; defaults must carry it."""
        image = Image(40, 129, "L", 8)
        image.fill(255)
        for x in range(10, 30):
            image.set(x, 64, 0)  # one dark horizontal line: a steady tone
        result = decode(image, sample_rate=8000, iterations=12)
        self.assertFalse(result.had_meta)
        self.assertEqual(result.audio.sample_rate, 8000)
        self.assertGreater(len(result.audio.samples), 0)
        self.assertGreater(result.audio.peak, 0.0)

    def test_n_fft_is_inferred_from_height(self):
        image = Image(20, 257, "L", 8)
        image.fill(255)
        image.set(10, 128, 0)
        result = decode(image, sample_rate=8000, iterations=1)
        self.assertEqual(result.spectrogram.n_fft, 512)

    def test_a_drawn_line_becomes_the_right_pitch(self):
        height, n_fft, rate = 257, 512, 8000
        image = Image(60, height, "L", 8)
        image.fill(255)
        row_from_bottom = 32  # bin 32 -> 32 * 8000 / 512 = 500 Hz
        for x in range(60):
            image.set(x, height - 1 - row_from_bottom, 0)
        audio = decode(image, sample_rate=rate, iterations=30).audio
        from audioimage import dsp

        mags = dsp.magnitudes(dsp.stft(audio.samples, n_fft, 128))
        middle = mags[len(mags) // 2]
        self.assertEqual(max(range(len(middle)), key=lambda i: middle[i]), row_from_bottom)

    def test_damaged_metadata_is_ignored(self):
        image = encode(tone(), SMALL)
        image.text[META_KEY] = "{not json"
        self.assertEqual(read_meta(image), {})
        image.text[META_KEY] = json.dumps([1, 2, 3])  # valid json, wrong shape
        self.assertEqual(read_meta(image), {})

    def test_no_metadata_at_all(self):
        self.assertEqual(read_meta(Image(2, 2)), {})


class TestCompare(unittest.TestCase):
    def test_identical_clips_score_perfectly(self):
        audio = tone()
        metrics = compare(audio, audio, SMALL)
        self.assertAlmostEqual(metrics["spectral_convergence"], 0.0, places=9)
        self.assertAlmostEqual(metrics["log_spectral_distance"], 0.0, places=9)

    def test_level_does_not_matter(self):
        """A quieter copy of the same sound is the same sound."""
        audio = tone()
        quiet = Audio([s * 0.1 for s in audio.samples], RATE)
        self.assertAlmostEqual(compare(audio, quiet, SMALL)["spectral_convergence"], 0.0, places=6)

    def test_empty(self):
        self.assertEqual(compare(Audio([], RATE), Audio([], RATE))["samples"], 0)


class TestFiles(unittest.TestCase):
    def test_encode_and_decode_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "in.wav")
            png = os.path.join(tmp, "out.png")
            out = os.path.join(tmp, "back.wav")
            write_wav(wav, tone())
            image, audio = encode_file(wav, png, SMALL)
            self.assertTrue(os.path.exists(png))
            self.assertEqual(image.height, SMALL.bins)
            result = decode_file(png, out, iterations=4)
            self.assertTrue(os.path.exists(out))
            self.assertEqual(len(result.audio.samples), len(audio.samples))

    def test_picture_survives_a_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "x.png")
            write_png(png, encode(tone(), SMALL))
            self.assertEqual(read_meta(read_png(png))["n_fft"], 256)


if __name__ == "__main__":
    unittest.main()
