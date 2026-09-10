"""Tests for radixnet.activation."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.activation import (  # noqa: E402
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_K,
    SineActivation,
    edge_signal,
    sine_activation,
    sine_derivative,
    sine_partials,
)

XS = [-12.0, -7.5, -3.0, -1.0, -0.25, 0.0, 0.3, 1.0, 2.5, 4.0, 9.0, 15.0]
PARAM_SETS = [
    (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K),
    (1.0, 1.0 / 3.0, 0.0, 0.0),
    (-0.7, 0.9, 1.3, -0.4),
    (2.0, 0.05, -2.0, 0.75),
]


class TestDefaults(unittest.TestCase):
    def test_default_equals_minus_sin_x_over_3(self):
        for x in XS:
            self.assertAlmostEqual(sine_activation(x), -1.0 * math.sin(x / 3.0), places=15)
            self.assertAlmostEqual(SineActivation()(x), -1.0 * math.sin(x / 3.0), places=15)

    def test_default_constants(self):
        self.assertEqual((DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K), (-1.0, 1.0 / 3.0, 0.0, 0.0))

    def test_partials_shape_and_consistency(self):
        for params in PARAM_SETS:
            for x in XS:
                p = sine_partials(x, *params)
                self.assertEqual(len(p), 6)
                self.assertAlmostEqual(p[0], sine_activation(x, *params), places=15)
                self.assertAlmostEqual(p[1], sine_derivative(x, *params), places=15)
                self.assertEqual(p[5], 1.0)


class TestNumericDerivatives(unittest.TestCase):
    """Central finite differences for df/dx, df/da, df/db, df/dh, df/dk."""

    EPS = 1e-6

    def _check(self, index, perturb):
        for params in PARAM_SETS:
            for x in XS:
                analytic = sine_partials(x, *params)[index]
                plus = sine_activation(*perturb(x, params, self.EPS))
                minus = sine_activation(*perturb(x, params, -self.EPS))
                numeric = (plus - minus) / (2 * self.EPS)
                self.assertAlmostEqual(analytic, numeric, delta=1e-6, msg=f"index {index} x={x} params={params}")

    def test_dfdx(self):
        self._check(1, lambda x, p, e: (x + e, *p))

    def test_dfda(self):
        self._check(2, lambda x, p, e: (x, p[0] + e, p[1], p[2], p[3]))

    def test_dfdb(self):
        self._check(3, lambda x, p, e: (x, p[0], p[1] + e, p[2], p[3]))

    def test_dfdh(self):
        self._check(4, lambda x, p, e: (x, p[0], p[1], p[2] + e, p[3]))

    def test_dfdk(self):
        self._check(5, lambda x, p, e: (x, p[0], p[1], p[2], p[3] + e))


class TestSineActivationObject(unittest.TestCase):
    def test_methods_match_functions(self):
        act = SineActivation(-0.7, 0.9, 1.3, -0.4)
        for x in XS:
            self.assertEqual(act(x), sine_activation(x, -0.7, 0.9, 1.3, -0.4))
            self.assertEqual(act.derivative(x), sine_derivative(x, -0.7, 0.9, 1.3, -0.4))
            self.assertEqual(act.partials(x), sine_partials(x, -0.7, 0.9, 1.3, -0.4))

    def test_inverted_flips_amplitude(self):
        act = SineActivation()
        inv = act.inverted()
        self.assertEqual(inv.a, -act.a)
        self.assertEqual((inv.b, inv.h, inv.k), (act.b, act.h, act.k))
        for x in XS:
            self.assertAlmostEqual(inv(x), math.sin(x / 3.0), places=15)
        self.assertEqual(inv.inverted(), act)

    def test_dict_round_trip(self):
        act = SineActivation(2.0, 0.05, -2.0, 0.75)
        d = act.to_dict()
        self.assertEqual(d, {"a": 2.0, "b": 0.05, "h": -2.0, "k": 0.75})
        self.assertEqual(SineActivation.from_dict(d), act)
        self.assertEqual(SineActivation.from_dict({}), SineActivation())

    def test_slots(self):
        act = SineActivation()
        with self.assertRaises(AttributeError):
            act.extra = 1  # type: ignore[attr-defined]


class TestEdgeSignal(unittest.TestCase):
    def test_product(self):
        self.assertEqual(edge_signal(2.0, -0.5, 0.25), -0.25)
        self.assertEqual(edge_signal(1.0, 0.0, 3.0), 0.0)
        # activation of child * activation of parent, scaled by the weight
        fp, fc = sine_activation(1.0), sine_activation(-2.0)
        self.assertAlmostEqual(edge_signal(0.7, fp, fc), 0.7 * fc * fp, places=15)


if __name__ == "__main__":
    unittest.main()
