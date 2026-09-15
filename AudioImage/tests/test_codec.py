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
    META_VERSION,
    PHASE_MODES,
    pack_phase,
    unpack_phase,
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
        self.assertEqual(meta["v"], META_VERSION)
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


class TestStoredPhase(unittest.TestCase):
    """phase="rgb" puts the angle in the blue channel, so nothing has to be guessed."""

    def setUp(self):
        self.audio = tone()
        self.cfg = EncodeConfig(n_fft=256, hop=64, phase="rgb")

    def test_modes(self):
        self.assertEqual(PHASE_MODES, ("none", "rgb"))
        self.assertEqual(EncodeConfig().phase, "none")

    def test_the_picture_is_rgb_and_still_grey(self):
        """R and G carry the level, so the picture reads as a spectrogram."""
        image = encode(self.audio, self.cfg)
        self.assertEqual(image.mode, "RGB")
        for x in range(0, image.width, 7):
            for y in range(0, image.height, 11):
                r, g, _ = image.get_rgb(x, y)
                self.assertEqual(r, g)

    def test_silence_stays_white(self):
        """Where nothing is sounding the blue channel repeats the grey."""
        image = encode(Audio([0.0] * 4000, RATE), EncodeConfig(n_fft=256, hop=64, phase="rgb"))
        for x in range(0, image.width, 5):
            r, g, b = image.get_rgb(x, 20)
            self.assertEqual((r, g, b), (image.maxval,) * 3)

    def test_metadata_records_it(self):
        meta = read_meta(encode(self.audio, self.cfg))
        self.assertEqual(meta["phase"], "rgb")
        self.assertEqual(meta["v"], META_VERSION)
        self.assertIn("phase_floor", meta)

    def test_decoding_needs_no_passes(self):
        result = decode(encode(self.audio, self.cfg), iterations=64)
        self.assertTrue(result.stored_phase)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.audio.meta["phase"], "stored")

    def test_the_waveform_comes_back(self):
        """The real prize: the samples match, not merely the spectrum."""
        result = decode(encode(self.audio, self.cfg))
        metrics = compare(self.audio, result.audio, self.cfg)
        self.assertGreater(metrics["waveform_snr"], 30.0)
        self.assertLess(metrics["spectral_convergence"], 0.02)

    def test_it_beats_rebuilding_the_phase(self):
        stored = compare(self.audio, decode(encode(self.audio, self.cfg)).audio, self.cfg)
        rebuilt_cfg = EncodeConfig(n_fft=256, hop=64)
        rebuilt = compare(self.audio, decode(encode(self.audio, rebuilt_cfg), iterations=64).audio, rebuilt_cfg)
        self.assertLess(stored["spectral_convergence"], rebuilt["spectral_convergence"])
        self.assertGreater(stored["waveform_snr"], rebuilt["waveform_snr"] + 20.0)

    def test_16_bit_is_better_than_8(self):
        scores = {}
        for depth in (8, 16):
            cfg = EncodeConfig(n_fft=256, hop=64, phase="rgb", depth=depth)
            scores[depth] = compare(self.audio, decode(encode(self.audio, cfg)).audio, cfg)["waveform_snr"]
        self.assertGreater(scores[16], scores[8])

    def test_a_deeper_floor_keeps_more_phase(self):
        sizes = {}
        from audioimage.png import write_png_bytes

        for floor in (30.0, 90.0):
            cfg = EncodeConfig(n_fft=256, hop=64, phase="rgb", phase_floor=floor)
            sizes[floor] = len(write_png_bytes(encode(self.audio, cfg)))
        self.assertLess(sizes[30.0], sizes[90.0])

    def test_refuses_layouts_it_cannot_invert(self):
        """An angle cannot be interpolated, so a warped or resized plane is refused."""
        for kwargs, why in (
            ({"colormap": "magma"}, "a colour ramp has no room for the phase"),
            ({"plane": PlaneConfig(freq_scale="mel")}, "a warped axis resamples"),
            ({"plane": PlaneConfig(freq_scale="log")}, "a warped axis resamples"),
            ({"plane": PlaneConfig(f_min=100.0)}, "a cropped axis resamples"),
            ({"plane": PlaneConfig(height=64)}, "a resized plane interpolates"),
        ):
            with self.subTest(why=why):
                with self.assertRaises(ValueError):
                    EncodeConfig(n_fft=256, hop=64, phase="rgb", **kwargs)

    def test_refuses_a_resized_width(self):
        cfg = EncodeConfig(n_fft=256, hop=64, phase="rgb", plane=PlaneConfig(width=20))
        with self.assertRaises(ValueError):
            encode(self.audio, cfg)

    def test_bad_settings(self):
        with self.assertRaises(ValueError):
            EncodeConfig(phase="maybe")
        with self.assertRaises(ValueError):
            EncodeConfig(phase="rgb", phase_floor=0.0)

    def test_a_version_1_picture_still_decodes(self):
        """Pictures written before phase existed carry none, and must still work."""
        image = encode(self.audio, EncodeConfig(n_fft=256, hop=64))
        result = decode(image, iterations=4)
        self.assertFalse(result.stored_phase)
        self.assertEqual(result.audio.meta["phase"], "rebuilt")

    def test_packing_directly(self):
        plane = [[0.0, 0.5], [1.0, 0.25]]
        phases = [[0.0, 1.5], [-3.0, 3.0]]
        image = pack_phase(plane, phases, depth=16, floor=0.0)
        got_plane, got_phase = unpack_phase(image)
        for a, b in zip(plane, got_plane):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=4)
        for a, b in zip(phases, got_phase):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=3)

    def test_packing_rejects_mismatched_shapes(self):
        from audioimage.png import PngError

        with self.assertRaises(PngError):
            pack_phase([[0.0, 0.5]], [[0.0]], 8, 0.0)
        with self.assertRaises(PngError):
            pack_phase([], [], 8, 0.0)

    def test_unpack_refuses_a_grey_picture(self):
        from audioimage.png import PngError

        with self.assertRaises(PngError):
            unpack_phase(encode(self.audio, EncodeConfig(n_fft=256, hop=64)))

    def test_angles_wrap_rather_than_clamp(self):
        """+pi and -pi are the same angle; the last bucket must meet the first."""
        image = pack_phase([[0.9, 0.9]], [[math.pi - 1e-9, -math.pi]], depth=8, floor=0.0)
        _, phases = unpack_phase(image)
        self.assertAlmostEqual(abs(phases[0][0] - phases[0][1]) % (2 * math.pi), 0.0, places=2)


class TestLossless(unittest.TestCase):
    """The correction channel: what comes out is what went in, sample for sample."""

    def setUp(self):
        self.audio = tone()
        self.cfg = EncodeConfig(n_fft=256, hop=64, phase="rgb", lossless=True)

    def _wav(self, audio):
        from audioimage.wav import write_wav_bytes

        return write_wav_bytes(audio)

    def test_the_file_comes_back_byte_for_byte(self):
        result = decode(encode(self.audio, self.cfg))
        self.assertTrue(result.lossless)
        self.assertEqual(self._wav(result.audio), self._wav(self.audio))

    def test_every_sample_is_exact(self):
        from audioimage.wav import read_wav_bytes

        result = decode(encode(self.audio, self.cfg))
        src = read_wav_bytes(self._wav(self.audio)).samples
        got = read_wav_bytes(self._wav(result.audio)).samples
        self.assertEqual(len(src), len(got))
        self.assertEqual(src, got)

    def test_it_holds_for_awkward_material(self):
        """Noise is the hardest case for everything else here; it must be exact too."""
        from audioimage.synth import chirp, noise

        for name, clip in (("noise", noise(0.3, RATE)), ("sweep", chirp(50, 6000, 0.3, RATE))):
            with self.subTest(name=name):
                result = decode(encode(clip, self.cfg))
                self.assertTrue(result.lossless)
                self.assertEqual(self._wav(result.audio), self._wav(clip))

    def test_silence_is_exact(self):
        from audioimage.wav import Audio

        quiet = Audio([0.0] * 3000, RATE)
        result = decode(encode(quiet, self.cfg))
        self.assertEqual(self._wav(result.audio), self._wav(quiet))

    def test_the_picture_gains_a_channel(self):
        image = encode(self.audio, self.cfg)
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(read_meta(image)["lossless"], True)

    def test_the_picture_is_still_grey_and_readable(self):
        image = encode(self.audio, self.cfg)
        for x in range(0, image.width, 5):
            for y in range(0, image.height, 9):
                r, g, _ = image.get_rgb(x, y)
                self.assertEqual(r, g)

    def test_it_needs_a_phase_to_lean_on(self):
        with self.assertRaises(ValueError):
            encode(self.audio, EncodeConfig(n_fft=256, hop=64, lossless=True))

    def test_a_picture_without_the_channel_is_not_claimed_lossless(self):
        result = decode(encode(self.audio, EncodeConfig(n_fft=256, hop=64, phase="rgb")))
        self.assertFalse(result.lossless)

    def test_the_level_is_left_alone(self):
        """Rescaling exact samples would stop them being exact."""
        result = decode(encode(self.audio, self.cfg), level="peak")
        self.assertEqual(self._wav(result.audio), self._wav(self.audio))

    def test_bad_bit_depth(self):
        with self.assertRaises(ValueError):
            EncodeConfig(lossless_bits=12)

    def test_it_refuses_rather_than_truncating(self):
        """Too few pixels to hold the correction has to be an error, not a silent loss."""
        from audioimage.codec import _pack_residual
        from audioimage.png import Image

        tiny = Image(2, 2, "RGBA", 16)
        with self.assertRaises(ValueError) as ctx:
            _pack_residual(tiny, [0] * 100)
        self.assertIn("pixels", str(ctx.exception))

    def test_a_correction_that_does_not_fit_is_refused(self):
        from audioimage.codec import _pack_residual
        from audioimage.png import Image

        image = Image(4, 4, "RGBA", 8)
        with self.assertRaises(ValueError):
            _pack_residual(image, [9999])
