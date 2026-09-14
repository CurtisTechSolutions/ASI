"""Tests for audioimage.wav."""

import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.wav import Audio, WavError, read_wav, read_wav_bytes, write_wav, write_wav_bytes  # noqa: E402


def tone(n=800, rate=8000, freq=440.0, amp=0.5):
    return Audio([amp * math.sin(2 * math.pi * freq * t / rate) for t in range(n)], rate)


class TestAudio(unittest.TestCase):
    def test_properties(self):
        a = Audio([0.0, 0.5, -1.0, 0.25], 8000)
        self.assertEqual(len(a), 4)
        self.assertAlmostEqual(a.duration, 4 / 8000)
        self.assertEqual(a.peak, 1.0)
        self.assertAlmostEqual(a.rms, math.sqrt((0 + 0.25 + 1 + 0.0625) / 4))

    def test_empty(self):
        a = Audio([], 8000)
        self.assertEqual(a.peak, 0.0)
        self.assertEqual(a.rms, 0.0)
        self.assertEqual(a.duration, 0.0)

    def test_bad_rate(self):
        with self.assertRaises(WavError):
            Audio([0.0], 0)

    def test_normalized(self):
        a = Audio([0.1, -0.2], 8000).normalized(0.8)
        self.assertAlmostEqual(a.peak, 0.8)
        # silence has no peak to scale to, and must not divide by zero
        self.assertEqual(Audio([0.0, 0.0], 8000).normalized().samples, [0.0, 0.0])

    def test_clipped(self):
        self.assertEqual(Audio([2.0, -3.0, 0.5], 8000).clipped().samples, [1.0, -1.0, 0.5])

    def test_describe(self):
        d = tone().describe()
        for key in ("sample_rate", "samples", "duration", "peak", "rms"):
            self.assertIn(key, d)


class TestRoundTrip(unittest.TestCase):
    def test_every_width(self):
        a = tone()
        for bits, floating, tolerance in ((8, False, 8e-3), (16, False, 3e-5), (24, False, 2e-7), (32, False, 1e-9), (32, True, 1e-7)):
            with self.subTest(bits=bits, floating=floating):
                back = read_wav_bytes(write_wav_bytes(a, bits=bits, floating=floating))
                self.assertEqual(back.sample_rate, a.sample_rate)
                self.assertEqual(len(back.samples), len(a.samples))
                self.assertLess(max(abs(x - y) for x, y in zip(a.samples, back.samples)), tolerance)

    def test_metadata(self):
        back = read_wav_bytes(write_wav_bytes(tone(), bits=24))
        self.assertEqual(back.meta["bits"], 24)
        self.assertEqual(back.meta["channels"], 1)
        self.assertEqual(back.meta["format"], "pcm")

    def test_clipping_does_not_wrap(self):
        # +1.5 must saturate, not wrap around to a large negative sample
        back = read_wav_bytes(write_wav_bytes(Audio([1.5, -1.5], 8000), bits=16))
        self.assertGreater(back.samples[0], 0.9)
        self.assertLess(back.samples[1], -0.9)

    def test_odd_length_is_padded(self):
        blob = write_wav_bytes(Audio([0.5], 8000), bits=8)  # 1 byte of data needs a pad byte
        self.assertEqual(len(blob) % 2, 0)
        self.assertEqual(len(read_wav_bytes(blob).samples), 1)

    def test_file_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "x.wav")
            write_wav(path, tone())  # the directory does not exist yet
            self.assertTrue(os.path.exists(path))
            self.assertEqual(len(read_wav(path).samples), 800)


class TestReading(unittest.TestCase):
    def _wav(self, fmt, channels, rate, bits, payload):
        fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
        body += b"data" + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", len(body)) + body

    def test_stereo_is_mixed_down(self):
        payload = struct.pack("<4h", 1000, 3000, -1000, -3000)  # two frames, two channels
        audio = read_wav_bytes(self._wav(1, 2, 8000, 16, payload))
        self.assertEqual(len(audio.samples), 2)
        self.assertAlmostEqual(audio.samples[0], 2000 / 32768.0)
        self.assertTrue(audio.meta["mixed_down"])

    def test_stereo_kept_interleaved(self):
        payload = struct.pack("<4h", 1000, 3000, -1000, -3000)
        audio = read_wav_bytes(self._wav(1, 2, 8000, 16, payload), mono=False)
        self.assertEqual(len(audio.samples), 4)
        self.assertFalse(audio.meta["mixed_down"])

    def test_extensible_format(self):
        # WAVE_FORMAT_EXTENSIBLE carries the real format code in its GUID
        # 16 bytes of standard fmt, then cbSize, wValidBitsPerSample and
        # dwChannelMask, which puts the sub-format GUID at offset 24
        fmt_chunk = (
            struct.pack("<HHIIHH", 0xFFFE, 1, 8000, 16000, 2, 16)
            + struct.pack("<HHI", 22, 16, 0)
            + struct.pack("<H", 1)
            + b"\x00" * 14
        )
        payload = struct.pack("<2h", 16384, -16384)
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
        body += b"data" + struct.pack("<I", len(payload)) + payload
        audio = read_wav_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        self.assertAlmostEqual(audio.samples[0], 0.5, places=4)

    def test_unknown_chunks_are_skipped(self):
        blob = write_wav_bytes(tone(8))
        extra = b"LIST" + struct.pack("<I", 4) + b"INFO"
        spliced = blob[:12] + extra + blob[12:]
        spliced = b"RIFF" + struct.pack("<I", len(spliced) - 8) + spliced[8:]
        self.assertEqual(len(read_wav_bytes(spliced).samples), 8)

    def test_rejections(self):
        for blob, reason in (
            (b"nope", "bad magic"),
            (b"RIFF" + struct.pack("<I", 4) + b"WAVE", "no fmt"),
            (self._wav(9999, 1, 8000, 16, b"\x00\x00"), "unknown format"),
            (self._wav(1, 1, 8000, 12, b"\x00\x00"), "odd width"),
            (self._wav(1, 0, 8000, 16, b"\x00\x00"), "no channels"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaises(WavError):
                    read_wav_bytes(blob)

    def test_float64_is_read(self):
        payload = struct.pack("<2d", 0.25, -0.75)
        audio = read_wav_bytes(self._wav(3, 1, 8000, 64, payload))
        self.assertAlmostEqual(audio.samples[0], 0.25)
        self.assertAlmostEqual(audio.samples[1], -0.75)


if __name__ == "__main__":
    unittest.main()
