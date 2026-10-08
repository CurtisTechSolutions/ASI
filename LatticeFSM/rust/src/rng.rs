//! The random generator: xoshiro256** seeded through splitmix64, written out so the crate has no
//! dependencies and a seed means the same draws on every platform.  (It is not Python's Mersenne Twister:
//! the two ports agree on everything deterministic and draw their own samples; `DESIGN.md` §9.)

#[derive(Clone, Debug, PartialEq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A float in `[0, 1)` with 53 bits of randomness.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// An integer in `[low, high]`.
    pub fn randint(&mut self, low: usize, high: usize) -> usize {
        low + (self.random() * (high - low + 1) as f64) as usize
    }

    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.randint(0, items.len() - 1)]
    }

    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    pub fn from_state(s: [u64; 4]) -> Rng {
        Rng { s }
    }
}
