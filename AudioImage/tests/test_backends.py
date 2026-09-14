"""The numpy backend must agree with the hand-written one, value for value."""

import contextlib
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage import backend, dsp  # noqa: E402

HAVE_NUMPY = backend.numpy_available()
skip_without_numpy = unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")


@contextlib.contextmanager
def _override(value):
    """Run with $AUDIOIMAGE_BACKEND set, and put it back afterwards."""
    before = os.environ.get(backend.ENV_VAR)
    os.environ[backend.ENV_VAR] = value
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(backend.ENV_VAR, None)
        else:
            os.environ[backend.ENV_VAR] = before


@contextlib.contextmanager
def _no_override():
    """Run with $AUDIOIMAGE_BACKEND unset, whatever the caller's shell had."""
    before = os.environ.pop(backend.ENV_VAR, None)
    try:
        yield
    finally:
        if before is not None:
            os.environ[backend.ENV_VAR] = before


def noise(n, seed=0):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


class TestSelection(unittest.TestCase):
    def test_resolve(self):
        self.assertEqual(backend.resolve("python"), "python")
        self.assertIn(backend.resolve("auto"), ("python", "numpy"))

    def test_python_always_works(self):
        self.assertIs(backend.get_backend("python"), dsp)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            backend.resolve("cuda")
        with self.assertRaises(ValueError):
            backend.get_backend("cuda")

    def test_describe(self):
        info = backend.describe_backends()
        self.assertIn(info["selected"], ("python", "numpy"))
        self.assertEqual(info["backends"], list(backend.BACKENDS))

    def test_auto_prefers_numpy_when_present(self):
        with _no_override():
            self.assertEqual(backend.resolve("auto"), "numpy" if HAVE_NUMPY else "python")

    def test_environment_override(self):
        """$AUDIOIMAGE_BACKEND pins what "auto" resolves to."""
        for value, want in (("python", "python"), ("numpy", "numpy" if HAVE_NUMPY else "numpy")):
            with _override(value):
                self.assertEqual(backend.resolve("auto"), want)

    def test_a_meaningless_override_is_ignored(self):
        with _override("banana"):
            self.assertIn(backend.resolve("auto"), ("python", "numpy"))

    def test_override_is_reported(self):
        with _override("python"):
            self.assertEqual(backend.describe_backends()["override"], "python")
        with _no_override():
            self.assertNotIn("override", backend.describe_backends())


@skip_without_numpy
class TestParity(unittest.TestCase):
    def setUp(self):
        self.np = backend.get_backend("numpy")
        self.py = dsp

    def _close(self, a, b, tol=1e-9, label=""):
        flat_a = [v for row in a for v in row] if a and isinstance(a[0], list) else list(a)
        flat_b = [v for row in b for v in row] if b and isinstance(b[0], list) else list(b)
        self.assertEqual(len(flat_a), len(flat_b), label)
        worst = max((abs(x - y) for x, y in zip(flat_a, flat_b)), default=0.0)
        self.assertLess(worst, tol, f"{label}: worst difference {worst}")

    def test_rfft(self):
        x = noise(256, 1)
        self._close(self.py.rfft(x), self.np.rfft(x), 1e-10, "rfft")

    def test_irfft(self):
        spec = self.py.rfft(noise(256, 2))
        self._close(self.py.irfft(spec, 256), self.np.irfft(spec, 256), 1e-10, "irfft")

    def test_fft(self):
        x = noise(128, 3)
        self._close(self.py.fft(x), self.np.fft(x), 1e-10, "fft")
        self._close(self.py.ifft(x), self.np.ifft(x), 1e-10, "ifft")

    def test_stft(self):
        x = noise(4096, 4)
        self._close(self.py.stft(x, 256, 64), self.np.stft(x, 256, 64), 1e-10, "stft")

    def test_istft(self):
        x = noise(4096, 5)
        frames = self.py.stft(x, 256, 64)
        self._close(
            self.py.istft(frames, 256, 64, length=len(x)),
            self.np.istft(frames, 256, 64, length=len(x)),
            1e-10,
            "istft",
        )

    def test_istft_of_an_impossible_spectrum(self):
        """Random phase puts an imaginary part on DC and Nyquist; both must drop it."""
        import cmath

        rng = random.Random(9)
        frames = self.py.stft(noise(2048, 6), 256, 64)
        dirty = [[cmath.rect(abs(v), rng.uniform(-math.pi, math.pi)) for v in row] for row in frames]
        self._close(self.py.istft(dirty, 256, 64), self.np.istft(dirty, 256, 64), 1e-10, "istft(dirty)")

    def test_magnitudes(self):
        frames = self.py.stft(noise(1024, 7), 128, 32)
        self._close(self.py.magnitudes(frames), self.np.magnitudes(frames), 1e-12, "magnitudes")

    def test_griffin_lim(self):
        x = noise(2048, 8)
        mags = self.py.magnitudes(self.py.stft(x, 256, 64))
        for init in ("zeros", "random"):
            for momentum in (0.0, 0.99):
                with self.subTest(init=init, momentum=momentum):
                    a = self.py.griffin_lim(mags, 256, 64, iterations=5, momentum=momentum, init=init, seed=3, length=len(x))
                    b = self.np.griffin_lim(mags, 256, 64, iterations=5, momentum=momentum, init=init, seed=3, length=len(x))
                    self._close(a, b, 1e-9, f"griffin_lim({init}, {momentum})")

    def test_plain_python_types(self):
        """numpy scalars must not leak out - they break json and confuse callers."""
        x = noise(512, 10)
        frames = self.np.stft(x, 128, 32)
        self.assertIs(type(frames[0][0]), complex)
        self.assertIs(type(self.np.magnitudes(frames)[0][0]), float)
        self.assertIs(type(self.np.istft(frames, 128, 32)[0]), float)
        self.assertIs(type(self.np.griffin_lim(self.np.magnitudes(frames), 128, 32, iterations=1)[0]), float)

    def test_shared_helpers_are_the_same_objects(self):
        self.assertIs(self.np.window_values, self.py.window_values)
        self.assertIs(self.np.frame_count, self.py.frame_count)


if __name__ == "__main__":
    unittest.main()
