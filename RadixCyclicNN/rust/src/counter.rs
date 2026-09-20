//! Cyclic counters - the odometer every growing integer in the model runs on,
//! identical to `radixnet/counter.py` and to the Go port's `counter.go`.
//!
//! A number that only ever counts up - traversals, visit counts, epochs,
//! trained characters - eventually leaves the range of whatever holds it, and
//! long before `i64` it leaves the 53-bit mantissa of the JSON number that
//! carries it through a model file.  So every counter is a two-digit odometer
//! in base [`COUNTER_LIMIT`]:
//!
//! ```text
//! total = resets * COUNTER_LIMIT + value        with 0 <= value < COUNTER_LIMIT
//! ```
//!
//! Wrapping is never done per increment: the hot loops add to a raw `i64` and a
//! carry sweep at a safe point moves whatever crossed the limit into the resets.

use std::collections::HashMap;

/// Where every counter wraps back to 0, counting the wrap as a reset: `10^15`,
/// exactly representable as an `f64` and four orders of magnitude below `i64`.
pub const COUNTER_LIMIT: i64 = 1_000_000_000_000_000;

/// Normalises a raw `(value, resets)` pair into `0 <= value < COUNTER_LIMIT`.
pub fn carry(value: i64, resets: i64) -> (i64, i64) {
    let mut turns = value / COUNTER_LIMIT;
    let mut value = value - turns * COUNTER_LIMIT;
    if value < 0 {
        value += COUNTER_LIMIT;
        turns -= 1;
    }
    let mut resets = (resets + turns) % COUNTER_LIMIT;
    if resets < 0 {
        resets += COUNTER_LIMIT;
    }
    (value, resets)
}

/// One scalar odometer: the reading plus how often it wrapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Counter {
    pub value: i64,
    pub resets: i64,
}

/// A version stamp no real reading can equal, so a cache marked with it is
/// always stale.
pub const INVALID_STAMP: Counter = Counter { value: -1, resets: 0 };

impl Counter {
    pub fn new(value: i64, resets: i64) -> Counter {
        let (value, resets) = carry(value, resets);
        Counter { value, resets }
    }

    /// The exact number of events counted, as a float.
    pub fn float(self) -> f64 {
        self.resets as f64 * COUNTER_LIMIT as f64 + self.value as f64
    }

    /// The reading `n` events later, wrapping as often as it takes.
    pub fn bumped(self, n: i64) -> Counter {
        Counter::new(self.value + n, self.resets)
    }

    /// Advances the counter in place.
    pub fn add(&mut self, n: i64) {
        *self = self.bumped(n);
    }

    /// Orders two readings by the number of events they stand for.
    pub fn less(self, other: Counter) -> bool {
        if self.resets != other.resets {
            return self.resets < other.resets;
        }
        self.value < other.value
    }
}

/// Wraps every entry of a parallel counter slice in place; returns how many
/// wrapped.  `resets` holds the reset counts of the ids that ever wrapped
/// (absent = 0), so the common case costs one comparison per entry.
pub fn carry_series(values: &mut [i64], resets: &mut HashMap<usize, i64>) -> usize {
    let mut wrapped = 0;
    for (i, v) in values.iter_mut().enumerate() {
        if *v >= COUNTER_LIMIT || *v < 0 {
            let (nv, nr) = carry(*v, resets.get(&i).copied().unwrap_or(0));
            *v = nv;
            if nr != 0 {
                resets.insert(i, nr);
            } else {
                resets.remove(&i);
            }
            wrapped += 1;
        }
    }
    wrapped
}

/// The exact count of an id in a parallel counter slice, as the float the
/// weight function sums.
pub fn counter_total(value: i64, resets: &HashMap<usize, i64>, id: usize) -> f64 {
    match resets.get(&id) {
        Some(&r) if r != 0 => r as f64 * COUNTER_LIMIT as f64 + value as f64,
        _ => value as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_odometer_goes_round() {
        let c = Counter::new(COUNTER_LIMIT - 1, 0).bumped(2);
        assert_eq!((c.value, c.resets), (1, 1));
        assert_eq!(c.float(), COUNTER_LIMIT as f64 + 1.0);
        assert!(Counter::new(5, 0).less(Counter::new(1, 1)));
    }

    #[test]
    fn a_sweep_only_touches_what_crossed_the_limit() {
        let mut values = vec![1, COUNTER_LIMIT + 3, 7];
        let mut resets = HashMap::new();
        assert_eq!(carry_series(&mut values, &mut resets), 1);
        assert_eq!(values, vec![1, 3, 7]);
        assert_eq!(resets.get(&1), Some(&1));
        assert_eq!(counter_total(3, &resets, 1), COUNTER_LIMIT as f64 + 3.0);
    }
}
