//! The hash the trigram index and the edge tables use.
//!
//! The standard library's default hasher is SipHash-1-3, chosen to be
//! unpredictable to an attacker who can pick the keys.  Nothing here is
//! attacker-chosen - the keys are trigrams and node ids out of a corpus - and
//! Go's map hashes them with a far cheaper function, so this port keeps the
//! comparison honest by hashing them cheaply too: the multiply-rotate-xor of
//! `rustc`'s own `FxHasher`, written out (it is 20 lines, and a dependency for
//! 20 lines would be the only one in the project).

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A fast, non-cryptographic hasher for small keys.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while rest.len() >= 8 {
            self.add(u64::from_ne_bytes(rest[..8].try_into().unwrap()));
            rest = &rest[8..];
        }
        if rest.len() >= 4 {
            self.add(u32::from_ne_bytes(rest[..4].try_into().unwrap()) as u64);
            rest = &rest[4..];
        }
        for &b in rest {
            self.add(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64)
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64)
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n)
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64)
    }
    #[inline]
    fn write_i64(&mut self, n: i64) {
        self.add(n as u64)
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// The `BuildHasher` behind every map in the model.
pub type FxBuild = BuildHasherDefault<FxHasher>;

/// A `HashMap` that hashes with [`FxHasher`].
pub type Map<K, V> = std::collections::HashMap<K, V, FxBuild>;
/// A `HashSet` that hashes with [`FxHasher`].
pub type Set<K> = std::collections::HashSet<K, FxBuild>;

/// An empty [`Map`].
pub fn map<K, V>() -> Map<K, V> {
    Map::default()
}

/// An empty [`Set`].
pub fn set<K>() -> Set<K> {
    Set::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_land_where_they_were_put() {
        let mut m: Map<u64, &str> = map();
        for i in 0..1000u64 {
            m.insert(i, "x");
        }
        assert_eq!(m.len(), 1000);
        assert_eq!(m.get(&999), Some(&"x"));
        assert_eq!(m.get(&1000), None);
    }
}
