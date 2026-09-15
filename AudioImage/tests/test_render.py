"""Tests for audioimage.font and audioimage.render."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.font import CHAR_WIDTH, FIRST_CHAR, LAST_CHAR, glyph, text_width  # noqa: E402
from audioimage.render import (  # noqa: E402
    Canvas,
    ViewConfig,
    format_hz,
    format_time,
    nice_ticks,
    render_view,
    render_waveform,
)
from audioimage.spectrogram import PlaneConfig  # noqa: E402
from audioimage.synth import melody  # noqa: E402


class TestFont(unittest.TestCase):
    def test_every_printable_character_has_a_glyph(self):
        for code in range(FIRST_CHAR, LAST_CHAR + 1):
            bits = glyph(chr(code))
            self.assertEqual(len(bits), CHAR_WIDTH)
            self.assertTrue(all(0 <= v < 128 for v in bits), chr(code))

    def test_space_is_blank_and_letters_are_not(self):
        self.assertEqual(glyph(" "), (0, 0, 0, 0, 0))
        for char in "AZaz09":
            self.assertGreater(sum(glyph(char)), 0, char)

    def test_glyphs_are_distinct(self):
        seen = {}
        for code in range(FIRST_CHAR + 1, LAST_CHAR + 1):  # space is legitimately blank
            char = chr(code)
            bits = glyph(char)
            self.assertNotIn(bits, seen, f"{char!r} draws the same as {seen.get(bits)!r}")
            seen[bits] = char

    def test_unknown_characters_are_blank(self):
        self.assertEqual(glyph("é"), (0, 0, 0, 0, 0))
        self.assertEqual(glyph(""), (0, 0, 0, 0, 0))

    def test_width(self):
        self.assertEqual(text_width(""), 0)
        self.assertEqual(text_width("A"), CHAR_WIDTH)
        self.assertEqual(text_width("AB"), CHAR_WIDTH * 2 + 1)
        self.assertEqual(text_width("AB", 2), (CHAR_WIDTH * 2 + 1) * 2)


class TestCanvas(unittest.TestCase):
    def test_background(self):
        canvas = Canvas(4, 3, (10, 20, 30))
        self.assertEqual(canvas.image.get_rgb(0, 0), (10, 20, 30))
        self.assertEqual((canvas.width, canvas.height), (4, 3))

    def test_drawing_outside_is_ignored(self):
        canvas = Canvas(4, 4)
        canvas.pixel(-5, -5, (0, 0, 0))
        canvas.pixel(99, 99, (0, 0, 0))
        canvas.hline(-2, 1, 10, (0, 0, 0))  # must not raise

    def test_text_marks_pixels(self):
        canvas = Canvas(40, 12)
        used = canvas.text(1, 1, "Hi", (0, 0, 0))
        self.assertEqual(used, text_width("Hi"))
        ink = sum(1 for y in range(12) for x in range(40) if canvas.image.get_rgb(x, y) == (0, 0, 0))
        self.assertGreater(ink, 5)

    def test_text_scale(self):
        canvas = Canvas(60, 30)
        canvas.text(0, 0, "A", (0, 0, 0), scale=3)
        ink = sum(1 for y in range(30) for x in range(60) if canvas.image.get_rgb(x, y) == (0, 0, 0))
        self.assertGreaterEqual(ink, 9)


class TestLabels(unittest.TestCase):
    def test_nice_ticks(self):
        self.assertEqual(nice_ticks(0.0, 10.0, 6), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        ticks = nice_ticks(0.0, 8000.0, 5)
        self.assertTrue(all(b > a for a, b in zip(ticks, ticks[1:])))
        self.assertEqual(nice_ticks(5.0, 5.0), [5.0])

    def test_format_hz(self):
        self.assertEqual(format_hz(0.0), "0")
        self.assertEqual(format_hz(440.0), "440")
        self.assertEqual(format_hz(8000.0), "8k")
        self.assertEqual(format_hz(1500.0), "1.5k")

    def test_format_time(self):
        self.assertEqual(format_time(0.0), "0s")
        self.assertEqual(format_time(1.5), "1.5s")
        self.assertEqual(format_time(0.25), "0.25s")
        self.assertEqual(format_time(12.0), "12s")


class TestViews(unittest.TestCase):
    def setUp(self):
        self.plane = [[(r + c) % 17 / 16 for c in range(40)] for r in range(33)]
        self.cfg = PlaneConfig(height=33, width=40, f_max=8000.0)

    def test_size_grows_by_the_margins(self):
        image = render_view(self.plane, 16000, 128, self.cfg)
        self.assertGreater(image.width, 40)
        self.assertGreater(image.height, 33)
        self.assertEqual(image.mode, "RGB")

    def test_bare_view_is_almost_the_plane(self):
        view = ViewConfig(axes=False, legend=False)
        image = render_view(self.plane, 16000, 128, self.cfg, view)
        self.assertEqual(image.width, 42)  # one pixel of margin each side
        self.assertEqual(image.height, 35)

    def test_zoom(self):
        view = ViewConfig(axes=False, legend=False, scale=3)
        image = render_view(self.plane, 16000, 128, self.cfg, view)
        self.assertEqual(image.width, 40 * 3 + 2)

    def test_resize(self):
        view = ViewConfig(axes=False, legend=False, width=80, height=20)
        image = render_view(self.plane, 16000, 128, self.cfg, view)
        self.assertEqual((image.width, image.height), (82, 22))

    def test_every_colormap_and_scale(self):
        for name in ("gray", "fire", "viridis", "magma", "ice"):
            for freq_scale in ("linear", "log", "mel"):
                with self.subTest(name=name, freq_scale=freq_scale):
                    cfg = PlaneConfig(height=33, width=40, f_max=8000.0, freq_scale=freq_scale)
                    image = render_view(self.plane, 16000, 128, cfg, ViewConfig(colormap=name, title="t"))
                    self.assertGreater(image.width, 40)

    def test_labels_do_not_overlap(self):
        """A log axis bunches its round numbers; the renderer must drop the extras."""
        cfg = PlaneConfig(height=200, width=40, f_max=8000.0, freq_scale="log")
        plane = [[0.5] * 40 for _ in range(200)]
        image = render_view(plane, 16000, 128, cfg)  # must not raise or overprint
        self.assertGreater(image.height, 200)

    def test_empty_is_refused(self):
        with self.assertRaises(ValueError):
            render_view([], 16000, 128, self.cfg)

    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            ViewConfig(scale=0)
        with self.assertRaises(ValueError):
            ViewConfig(colormap="nope")


class TestWaveform(unittest.TestCase):
    def test_size(self):
        audio = melody(("C4", "E4"), 0.1, 8000)
        image = render_waveform(audio, 200, 80)
        self.assertGreater(image.width, 200)
        self.assertGreater(image.height, 80)

    def test_silence_draws_a_line(self):
        from audioimage.wav import Audio

        image = render_waveform(Audio([0.0] * 800, 8000), 100, 40, ViewConfig(axes=False))
        self.assertEqual(image.mode, "RGB")

    def test_empty_clip(self):
        from audioimage.wav import Audio

        render_waveform(Audio([], 8000), 100, 40)  # must not divide by zero

    def test_too_small(self):
        from audioimage.wav import Audio

        with self.assertRaises(ValueError):
            render_waveform(Audio([0.0], 8000), 4, 4)


if __name__ == "__main__":
    unittest.main()
