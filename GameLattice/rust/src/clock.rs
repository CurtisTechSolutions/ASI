//! One clock for every lattice (`DESIGN.md` §5.3, §12.3): every lattice ticks on every step of any lattice, so
//! an idle sense's traces fade; wall-clock idleness enters as explicit ticks.

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedClock {
    pub ticks: i64,
    /// Ticks per second of wall-clock idleness, when the body is given one (0: idleness is not time).
    pub ticks_per_second: f64,
}

impl SharedClock {
    /// A lattice stepped `k` times: the others let `k` pass.  Returns the ticks the others must take.
    pub fn stepped(&mut self, k: i64) -> i64 {
        self.ticks += k.max(0);
        k.max(0)
    }

    pub fn idle(&mut self, seconds: f64) -> i64 {
        let k = (seconds.max(0.0) * self.ticks_per_second).floor() as i64;
        self.ticks += k;
        k
    }
}
