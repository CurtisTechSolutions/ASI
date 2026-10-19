//! Fast tables for the counting models: a hash function that costs a multiply a word, and a context key that lives
//! inline instead of on the heap.
//!
//! Training a chain or an N-gram model is mostly looking contexts up and adding to what they hold - tens of millions
//! of times for a few megabytes of text.  The standard `HashMap` hashes with SipHash, built to resist an attacker
//! choosing the keys, and a `Vec<u32>` key is a heap allocation for every new context and a pointer to follow on every
//! comparison.  [`FastMap`] hashes with the Fx function (the one rustc uses for its own tables: rotate, xor, multiply),
//! and [`Ctx`] holds up to [`MAX_CTX`] ids in place.  A `Ctx` hashes, compares and orders exactly as the `[u32]` slice
//! it holds, so a table keyed by `Ctx` is still looked up by a slice (`map.get(&ids[a..b])`), and anything sorted by
//! its keys comes out in the same order as before.  Nothing the models compute depends on a table's iteration order.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::ops::Deref;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// The Fx hash: for every word, rotate, xor it in, multiply.  Not for keys an attacker chooses - a corpus crafted to
/// collide could slow a table down, never change what it holds.
#[derive(Clone, Copy, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().expect("eight bytes")));
        }
        let rest = chunks.remainder();
        if rest.len() >= 4 {
            self.add(u32::from_le_bytes(rest[..4].try_into().expect("four bytes")) as u64);
            for &b in &rest[4..] {
                self.add(b as u64);
            }
        } else {
            for &b in rest {
                self.add(b as u64);
            }
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
/// A `HashMap` hashed with [`FxHasher`]; make one with `FastMap::default()`.
pub type FastMap<K, V> = HashMap<K, V, FxBuild>;
pub type FastSet<T> = HashSet<T, FxBuild>;

/// The longest context a [`Ctx`] holds: a chain of 16 nodes reads 15 inputs (`chain::MAX_NODES - 1`), an N-gram model of
/// order 12 eleven.
pub const MAX_CTX: usize = 15;

/// A context - the last few ids - held in place: no allocation to make one, no pointer to follow to compare two.
#[derive(Clone, Copy)]
pub struct Ctx {
    len: u8,
    ids: [u32; MAX_CTX],
}

impl Ctx {
    /// `ids` as a key (at most [`MAX_CTX`] of them).
    #[inline]
    pub fn new(ids: &[u32]) -> Ctx {
        assert!(
            ids.len() <= MAX_CTX,
            "a context of {} ids (at most {MAX_CTX})",
            ids.len()
        );
        let mut out = [0u32; MAX_CTX];
        out[..ids.len()].copy_from_slice(ids);
        Ctx {
            len: ids.len() as u8,
            ids: out,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.ids[..self.len as usize]
    }
}

impl Deref for Ctx {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &[u32] {
        self.as_slice()
    }
}

impl Borrow<[u32]> for Ctx {
    #[inline]
    fn borrow(&self) -> &[u32] {
        self.as_slice()
    }
}

impl From<&[u32]> for Ctx {
    fn from(ids: &[u32]) -> Ctx {
        Ctx::new(ids)
    }
}

impl From<Vec<u32>> for Ctx {
    fn from(ids: Vec<u32>) -> Ctx {
        Ctx::new(&ids)
    }
}

/// Hashed exactly as the slice, so a lookup by `&[u32]` finds it.
impl Hash for Ctx {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl PartialEq for Ctx {
    #[inline]
    fn eq(&self, other: &Ctx) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Ctx {}

/// Ordered as the slice: shorter-is-less only where one is a prefix of the other, as `Vec<u32>` was.
impl Ord for Ctx {
    fn cmp(&self, other: &Ctx) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl PartialOrd for Ctx {
    fn partial_cmp(&self, other: &Ctx) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Ctx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn a_ctx_is_its_slice() {
        let a = Ctx::new(&[3, 1, 4]);
        let b = Ctx::new(&[3, 1, 4, 1]);
        assert_eq!(a.as_slice(), &[3, 1, 4]);
        assert!(a < b && Ctx::new(&[3, 2]) > b && Ctx::new(&[]) < a);
        let build = FxBuild::default();
        assert_eq!(build.hash_one(a), build.hash_one(&[3u32, 1, 4][..]));
        let mut m: FastMap<Ctx, u32> = FastMap::default();
        m.insert(a, 7);
        assert_eq!(m.get(&[3u32, 1, 4][..]), Some(&7));
        assert_eq!(m.get(&[3u32, 1][..]), None);
    }

    #[test]
    #[should_panic]
    fn too_long_a_context_is_refused() {
        Ctx::new(&[0; MAX_CTX + 1]);
    }
}
