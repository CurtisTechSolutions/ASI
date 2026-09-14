"""Tests for audioimage.png - the hand-written PNG codec."""

import os
import random
import struct
import sys
import unittest
import warnings
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.png import SIGNATURE, Image, PngError, read_png, read_png_bytes, write_png, write_png_bytes  # noqa: E402

try:
    from PIL import Image as PILImage  # a cross-check when it happens to be installed
except Exception:  # pragma: no cover - Pillow is not a dependency
    PILImage = None


def random_image(width=37, height=23, mode="L", depth=8, seed=0):
    rng = random.Random(seed)
    im = Image(width, height, mode, depth)
    top = im.maxval
    for y in range(height):
        for x in range(width):
            im.set_rgb(x, y, (rng.randrange(top + 1), rng.randrange(256), rng.randrange(256)))
    return im


class TestImage(unittest.TestCase):
    def test_defaults(self):
        im = Image(4, 3)
        self.assertEqual((im.width, im.height, im.mode, im.depth), (4, 3, "L", 8))
        self.assertEqual(im.channels, 1)
        self.assertEqual(im.maxval, 255)
        self.assertEqual(len(im.data), 12)

    def test_rgb(self):
        im = Image(2, 2, "RGB")
        im.set_rgb(1, 1, (10, 20, 30))
        self.assertEqual(im.get_rgb(1, 1), (10, 20, 30))
        self.assertEqual(len(im.data), 12)

    def test_grey_reads_as_a_triple(self):
        im = Image(2, 2, "L", 16)
        im.set(0, 0, 4000)
        self.assertEqual(im.get_rgb(0, 0), (4000, 4000, 4000))
        self.assertEqual(im.maxval, 65535)

    def test_fill_and_row(self):
        im = Image(3, 2)
        im.fill(7)
        self.assertEqual(list(im.row(1)), [7, 7, 7])

    def test_rejections(self):
        for args, why in (
            ((0, 5), "no width"),
            ((5, 0), "no height"),
            ((5, 5, "CMYK"), "unknown mode"),
            ((5, 5, "L", 4), "odd depth"),
        ):
            with self.subTest(why=why):
                with self.assertRaises(PngError):
                    Image(*args)

    def test_wrong_buffer_size(self):
        from array import array

        with self.assertRaises(PngError):
            Image(2, 2, "L", 8, array("H", [1, 2, 3]))


class TestRoundTrip(unittest.TestCase):
    def test_every_mode(self):
        for mode, depth in (("L", 8), ("L", 16), ("RGB", 8), ("RGB", 16)):
            with self.subTest(mode=mode, depth=depth):
                im = random_image(mode=mode, depth=depth)
                back = read_png_bytes(write_png_bytes(im))
                self.assertEqual((back.width, back.height, back.mode, back.depth), (im.width, im.height, mode, depth))
                self.assertEqual(list(back.data), list(im.data))

    def test_16_bit_colour(self):
        """What lets one picture hold a full-precision magnitude and a phase."""
        im = random_image(24, 18, "RGB", 16, seed=3)
        back = read_png_bytes(write_png_bytes(im))
        self.assertEqual((back.mode, back.depth), ("RGB", 16))
        self.assertEqual(list(back.data), list(im.data))
        self.assertGreater(max(im.data), 255)  # genuinely using the extra byte

    def test_text_chunks_survive(self):
        im = random_image()
        im.text = {"audioimage": '{"v":1,"sample_rate":22050}', "Software": "audioimage"}
        self.assertEqual(read_png_bytes(write_png_bytes(im)).text, im.text)

    def test_signature(self):
        self.assertTrue(write_png_bytes(random_image()).startswith(SIGNATURE))

    def test_single_pixel(self):
        im = Image(1, 1, "L", 16)
        im.set(0, 0, 65535)
        self.assertEqual(read_png_bytes(write_png_bytes(im)).get(0, 0), 65535)

    def test_adaptive_filtering_compresses(self):
        """A smooth picture must come out far smaller than its raw bytes."""
        im = Image(200, 200, "L", 16)
        for y in range(200):
            for x in range(200):
                im.set(x, y, int((x + y) / 398 * 65535))
        self.assertLess(len(write_png_bytes(im)), 200 * 200 * 2 // 10)

    def test_file_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deep", "x.png")
            im = random_image(mode="L", depth=16)
            write_png(path, im)
            self.assertEqual(list(read_png(path).data), list(im.data))

    def test_values_are_clamped(self):
        im = Image(2, 1, "L", 8)
        im.data[0] = 300  # out of range for 8 bits
        self.assertEqual(read_png_bytes(write_png_bytes(im)).get(0, 0), 255)


class TestReading(unittest.TestCase):
    def test_rejections(self):
        with self.assertRaises(PngError):
            read_png_bytes(b"not a png at all")
        with self.assertRaises(PngError):
            read_png_bytes(SIGNATURE)  # no IHDR

    def test_interlaced_is_refused(self):
        def chunk(tag, payload):
            return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))

        blob = SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 1)) + chunk(b"IEND", b"")
        with self.assertRaises(PngError) as ctx:
            read_png_bytes(blob)
        self.assertIn("interlaced", str(ctx.exception))

    def test_truncated_data(self):
        blob = bytearray(write_png_bytes(random_image()))
        with self.assertRaises(PngError):
            read_png_bytes(bytes(blob[:60]))

    @unittest.skipIf(PILImage is None, "Pillow is not installed")
    def test_pillow_reads_what_we_write(self):
        import io

        for mode, depth in (("L", 8), ("L", 16), ("RGB", 8)):
            with self.subTest(mode=mode, depth=depth):
                im = random_image(64, 40, mode, depth, seed=depth)
                pil = PILImage.open(io.BytesIO(write_png_bytes(im)))
                self.assertEqual(pil.size, (64, 40))
                # no convert() here: converting a 16-bit image to "L" would
                # squash the samples to 8 bits before they could be compared
                with warnings.catch_warnings():  # getdata is deprecated in new Pillow
                    warnings.simplefilter("ignore", DeprecationWarning)
                    pixels = list(pil.getdata())
                flat = [v for px in pixels for v in (px if isinstance(px, tuple) else (px,))]
                self.assertEqual(flat, list(im.data))

    @unittest.skipIf(PILImage is None, "Pillow is not installed")
    def test_we_read_what_pillow_writes(self):
        import io

        for pil_mode, mode, depth in (("L", "L", 8), ("I;16", "L", 16), ("RGB", "RGB", 8), ("P", "RGB", 8)):
            with self.subTest(pil_mode=pil_mode):
                src = PILImage.new("RGB", (20, 10))
                for x in range(20):
                    for y in range(10):
                        src.putpixel((x, y), (x * 11 % 256, y * 23 % 256, (x + y) * 7 % 256))
                if pil_mode != "RGB":
                    src = src.convert(pil_mode)
                buf = io.BytesIO()
                src.save(buf, "PNG")
                mine = read_png_bytes(buf.getvalue())
                self.assertEqual((mine.width, mine.height), (20, 10))
                self.assertEqual(mine.mode, mode)
                for x in range(20):
                    for y in range(10):
                        want = src.convert("RGB").getpixel((x, y)) if mine.mode == "RGB" else (src.getpixel((x, y)),) * 3
                        self.assertEqual(mine.get_rgb(x, y), want)


if __name__ == "__main__":
    unittest.main()
