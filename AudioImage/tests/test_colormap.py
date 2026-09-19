"""Tests for audioimage.colormap."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.colormap import COLORMAPS, GRAY, NearestColor, get_colormap  # noqa: E402


class TestGray(unittest.TestCase):
    def test_the_rule(self):
        """Black is 1.0 and white is 0.0 - the whole format rests on this."""
        self.assertEqual(GRAY.to_value(1.0, 255), 0)
        self.assertEqual(GRAY.to_value(0.0, 255), 255)
        self.assertEqual(GRAY.to_value(1.0, 65535), 0)
        self.assertEqual(GRAY.to_value(0.0, 65535), 65535)

    def test_inverse_is_exact_at_16_bits(self):
        for i in range(0, 65536, 61):
            intensity = i / 65535
            self.assertAlmostEqual(GRAY.from_value(GRAY.to_value(intensity, 65535), 65535), intensity, places=9)

    def test_clamping(self):
        self.assertEqual(GRAY.to_value(5.0, 255), 0)
        self.assertEqual(GRAY.to_value(-5.0, 255), 255)
        self.assertEqual(GRAY.from_value(999, 255), 0.0)

    def test_inverted_ramp(self):
        inv = get_colormap("gray-inv")
        self.assertEqual(inv.to_value(1.0, 255), 255)
        self.assertEqual(inv.to_value(0.0, 255), 0)


class TestColorMaps(unittest.TestCase):
    def test_all_named_maps_exist(self):
        for name in COLORMAPS:
            cmap = get_colormap(name)
            self.assertEqual(len(cmap.lut), 256)
            self.assertIn(cmap.mode, ("L", "RGB"))

    def test_unknown(self):
        with self.assertRaises(ValueError):
            get_colormap("rainbow-unicorn")

    def test_lut_round_trip(self):
        """Every colour a map can produce must map back to (almost) the same value."""
        for name in COLORMAPS:
            cmap = get_colormap(name)
            worst = max(abs(cmap.from_rgb(cmap.to_rgb(i / 255)) - i / 255) for i in range(256))
            with self.subTest(name=name):
                self.assertLessEqual(worst, 1 / 255 + 1e-9)

    def test_cached(self):
        self.assertIs(get_colormap("fire"), get_colormap("fire"))

    def test_swatch(self):
        self.assertEqual(len(get_colormap("magma").swatch(5)), 5)

    def test_ends(self):
        for name in ("gray", "fire", "ice"):
            cmap = get_colormap(name)
            self.assertEqual(cmap.to_rgb(0.0), (255, 255, 255))
            self.assertEqual(cmap.to_rgb(1.0), (0, 0, 0))


class TestNearestColor(unittest.TestCase):
    def test_exact_colours(self):
        """A colour from the table comes back as an entry holding that exact colour.

        Not necessarily the same index: a ramp can land on the same 8-bit
        colour twice, and the lowest of those indices always wins so that
        inversion is stable.
        """
        cmap = get_colormap("viridis")
        finder = NearestColor(cmap.lut)
        for i in (0, 37, 128, 255):
            self.assertEqual(cmap.lut[finder.index(cmap.lut[i])], cmap.lut[i])
            self.assertLessEqual(finder.index(cmap.lut[i]), i)

    def test_backends_agree_exactly(self):
        """A picture must decode the same whether or not numpy is installed."""
        rng = random.Random(4)
        colors = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(2000)]
        for name in COLORMAPS:
            with self.subTest(name=name):
                fast = NearestColor(get_colormap(name).lut, use_numpy=True)
                slow = NearestColor(get_colormap(name).lut, use_numpy=False)
                self.assertEqual(fast.index_many(colors), [slow.index(c) for c in colors])

    def test_ties_go_to_the_lowest_index(self):
        # two entries exactly as far from (128, 0, 0): the first one wins
        finder = NearestColor([(0, 0, 0), (100, 0, 0), (156, 0, 0)], use_numpy=False)
        self.assertEqual(finder.index((128, 0, 0)), 1)
        fast = NearestColor([(0, 0, 0), (100, 0, 0), (156, 0, 0)], use_numpy=True)
        self.assertEqual(fast.index_many([(128, 0, 0)]), [1])

    def test_nearest_is_actually_nearest(self):
        cmap = get_colormap("magma")
        finder = NearestColor(cmap.lut, use_numpy=False)
        rng = random.Random(11)
        for _ in range(50):
            c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            got = finder.index(c)
            best = min(range(256), key=lambda i: sum((c[k] - cmap.lut[i][k]) ** 2 for k in range(3)))
            d_got = sum((c[k] - cmap.lut[got][k]) ** 2 for k in range(3))
            d_best = sum((c[k] - cmap.lut[best][k]) ** 2 for k in range(3))
            self.assertEqual(d_got, d_best)

    def test_index_many_matches_index(self):
        cmap = get_colormap("ice")
        rng = random.Random(5)
        colors = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(200)]
        finder = NearestColor(cmap.lut)
        self.assertEqual(finder.index_many(colors), [finder.index(c) for c in colors])

    def test_from_rgb_many(self):
        cmap = get_colormap("fire")
        values = [i / 255 for i in range(0, 256, 17)]
        colors = [cmap.to_rgb(v) for v in values]
        for got, want in zip(cmap.from_rgb_many(colors), values):
            self.assertAlmostEqual(got, want, places=2)


if __name__ == "__main__":
    unittest.main()
