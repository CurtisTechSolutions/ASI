//! The parametric sine activation of the radix model.
//!
//! Every node of a radix graph owns an activation `f(x) = a * sin(b * (x - h)) + k`;
//! with the default parameters it is exactly the author's `-sin(x / 3)`.  All
//! four parameters are learnable per node, which is what "update the activation
//! function instead of fighting the vanishing gradient" means in this design.
//!
//! A port of `radixnet/activation.py`.  The formulas are written in Python's
//! operation order on purpose - `a * sin(b * (z - h)) + k`, never a fused or
//! reassociated form - because the edge scores built from them are compared
//! with Python's to the bit.

/// The default amplitude: negative, so a fresh node reads `-sin(x / 3)`.
pub const DEFAULT_A: f64 = -1.0;
/// The default frequency.
pub const DEFAULT_B: f64 = 1.0 / 3.0;
/// The default phase shift.
pub const DEFAULT_H: f64 = 0.0;
/// The default offset.
pub const DEFAULT_K: f64 = 0.0;

/// `a * sin(b * (x - h)) + k`.
#[inline]
pub fn sine(x: f64, a: f64, b: f64, h: f64, k: f64) -> f64 {
    a * (b * (x - h)).sin() + k
}

/// `df/dx = a * b * cos(b * (x - h))`.
pub fn sine_derivative(x: f64, a: f64, b: f64, h: f64) -> f64 {
    a * b * (b * (x - h)).cos()
}

/// `(f, df/dx, df/da, df/db, df/dh)` at `x` (`df/dk` is 1).
///
/// With `u = b * (x - h)`: `f = a sin u + k`, `df/dx = a b cos u`,
/// `df/da = sin u`, `df/db = a (x - h) cos u`, `df/dh = -a b cos u` - the
/// tuple the backend caches per node and batch.
#[inline]
pub fn partials(x: f64, a: f64, b: f64, h: f64, k: f64) -> [f64; 5] {
    let d = x - h;
    let u = b * d;
    let su = u.sin();
    let cu = u.cos();
    let abc = a * b * cu;
    [a * su + k, abc, su, a * d * cu, -abc]
}

/// The signal edge `p -> c` carries: its weight times both activations.
#[inline]
pub fn edge_signal(w: f64, fp: f64, fc: f64) -> f64 {
    w * fp * fc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_minus_sin_of_a_third() {
        for x in [-4.0f64, -1.5, 0.0, 0.7, 3.0] {
            // `b = 1/3` multiplies where `x / 3` divides, so the last bit may differ
            assert!((sine(x, DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K) + (x / 3.0).sin()).abs() < 1e-15);
        }
    }

    #[test]
    fn the_partials_match_finite_differences() {
        let (x, a, b, h, k) = (0.8, -1.3, 0.4, 0.2, 0.1);
        let p = partials(x, a, b, h, k);
        assert_eq!(p[0], sine(x, a, b, h, k));
        let eps = 1e-6;
        let fd = |f: &dyn Fn(f64) -> f64| (f(eps) - f(-eps)) / (2.0 * eps);
        assert!((p[1] - fd(&|d| sine(x + d, a, b, h, k))).abs() < 1e-7);
        assert!((p[2] - fd(&|d| sine(x, a + d, b, h, k))).abs() < 1e-7);
        assert!((p[3] - fd(&|d| sine(x, a, b + d, h, k))).abs() < 1e-7);
        assert!((p[4] - fd(&|d| sine(x, a, b, h + d, k))).abs() < 1e-7);
        assert_eq!(p[1], sine_derivative(x, a, b, h));
    }

    #[test]
    fn negating_amplitude_and_offset_negates_the_unit() {
        let (x, a, b, h, k) = (1.1, -0.9, 0.3, 0.05, 0.2);
        assert_eq!(sine(x, -a, b, h, -k), -sine(x, a, b, h, k));
        assert_eq!(edge_signal(2.0, 0.5, -0.25), -0.25);
    }
}
