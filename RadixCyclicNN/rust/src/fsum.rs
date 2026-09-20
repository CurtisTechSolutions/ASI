//! An exact float sum (Shewchuk's algorithm, as Python's `math.fsum` uses it),
//! so a path cost comes out the same here as in Python and in Go.

/// The correctly rounded sum of `xs`, without the special handling of
/// infinities the CPython implementation adds.
pub fn fsum(xs: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::with_capacity(8);
    for &value in xs {
        let mut x = value;
        let mut i = 0usize;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::fsum;

    #[test]
    fn the_sum_is_the_one_that_rounds_right() {
        let xs = [1e100, 1.0, -1e100, 1.0];
        assert_eq!(fsum(&xs), 2.0);
        assert_eq!(fsum(&[]), 0.0);
        // a naive sum of ten tenths is 0.9999999999999999; this one is not
        let naive: f64 = [0.1f64; 10].iter().sum();
        assert_ne!(naive, 1.0);
        assert_eq!(fsum(&[0.1; 10]), 1.0);
    }
}
