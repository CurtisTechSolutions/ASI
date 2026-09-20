//! A Mersenne Twister with the exact semantics of CPython's `random.Random`:
//! the same seeding (`init_by_array` over the 32-bit chunks of `|seed|`), the
//! same 53-bit doubles and the same state layout, so a walk sampled here draws
//! the same numbers the Python and Go implementations draw.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A Python-compatible pseudo random generator.
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// Seeds a generator like `random.Random(seed)`.
    pub fn new(seed: i64) -> Mt19937 {
        let mut rng = Mt19937 { mt: [0; N], index: N };
        rng.seed(seed);
        rng
    }

    /// Reseeds like `random.seed(seed)` for an integer seed: the key is
    /// `|seed|` split into 32-bit words, least significant first.
    pub fn seed(&mut self, seed: i64) {
        let n = seed.unsigned_abs();
        let mut key = Vec::new();
        let mut u = n;
        loop {
            key.push((u & 0xffff_ffff) as u32);
            u >>= 32;
            if u == 0 {
                break;
            }
        }
        self.init_by_array(&key);
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.index = N;
    }

    /// The next 32-bit output (`genrand_uint32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            let twist = |a: u32, b: u32| -> u32 {
                let y = (a & UPPER_MASK) | (b & LOWER_MASK);
                (y >> 1) ^ (MATRIX_A * (y & 1))
            };
            for kk in 0..N - M {
                self.mt[kk] = self.mt[kk + M] ^ twist(self.mt[kk], self.mt[kk + 1]);
            }
            for kk in N - M..N - 1 {
                self.mt[kk] = self.mt[kk + M - N] ^ twist(self.mt[kk], self.mt[kk + 1]);
            }
            self.mt[N - 1] = self.mt[M - 1] ^ twist(self.mt[N - 1], self.mt[0]);
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// The next double in `[0, 1)`, exactly like `random.random()`.
    pub fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `a + (b - a) * random()`, like `random.uniform`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::Mt19937;

    #[test]
    fn the_stream_is_pythons() {
        // random.Random(0).random() three times, and random.Random(1).random()
        let mut rng = Mt19937::new(0);
        assert!((rng.next_f64() - 0.8444218515250481).abs() < 1e-15);
        assert!((rng.next_f64() - 0.7579544029403025).abs() < 1e-15);
        assert!((rng.next_f64() - 0.420571580830845).abs() < 1e-15);
        let mut rng = Mt19937::new(1);
        assert!((rng.next_f64() - 0.13436424411240122).abs() < 1e-15);
    }
}
