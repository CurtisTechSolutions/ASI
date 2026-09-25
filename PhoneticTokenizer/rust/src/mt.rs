//! A Mersenne Twister with the semantics of CPython's `random.Random` for what
//! the phonotactics draw: the same seeding, the same 53-bit doubles and the
//! same `randrange(n)`, so a word coined with a seed in one port is the word
//! coined with that seed in the others.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER: u32 = 0x80000000;
const LOWER: u32 = 0x7fffffff;

/// What the phonotactics need of a random source.
pub trait Rng {
    fn float64(&mut self) -> f64;
    fn rand_below(&mut self, n: usize) -> usize;
}

/// A Python-compatible pseudo random generator.
pub struct Mt {
    mt: [u32; N],
    index: usize,
}

impl Mt {
    /// Seeds like `random.Random(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Mt {
        let mut r = Mt {
            mt: [0; N],
            index: N,
        };
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1812433253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = if key.len() > N { key.len() } else { N };
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525)))
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
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x80000000;
        self.index = N;
    }

    /// The next 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            let mut kk = 0;
            while kk < N - M {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ (MATRIX_A * (y & 1));
                kk += 1;
            }
            while kk < N - 1 {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ (MATRIX_A * (y & 1));
                kk += 1;
            }
            let y = (self.mt[N - 1] & UPPER) | (self.mt[0] & LOWER);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (MATRIX_A * (y & 1));
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }
}

impl Rng for Mt {
    /// `random()`: a double in [0, 1) from 53 bits, as CPython makes it.
    fn float64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `randrange(n)`: rejection sampling over `getrandbits(n.bit_length())`.
    fn rand_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "randrange of a non-positive n");
        let k = usize::BITS - n.leading_zeros();
        loop {
            let v = self.getrandbits(k) as usize;
            if v < n {
                return v;
            }
        }
    }
}
