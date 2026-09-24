//! BLAKE2b (RFC 7693), written out - what the resonant model hashes a gram
//! with, and what names an utterance's `<speech:digest>` token.
//!
//! The phase model gives every gram a phase of its own, and it must be the
//! same phase in every process and in every implementation: Python's `hash()`
//! is salted per process, so `radixnet/resonance.py` takes it from
//! `hashlib.blake2b(gram, digest_size=8)` instead.  A model file trained in
//! Python and continued here has to read its grams at the same phases, so this
//! is the same function, unkeyed and with a variable digest length, and not an
//! approximation of it.  No dependencies, like the rest of the crate.

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline]
fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], counter: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&block[i * 8..i * 8 + 8]);
        *word = u64::from_le_bytes(bytes);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter as u64;
    v[13] ^= (counter >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for round in 0..12 {
        let s = &SIGMA[round % 10];
        mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// The unkeyed BLAKE2b digest of `data`, `outlen` bytes long (1..=64), as
/// `hashlib.blake2b(data, digest_size=outlen).digest()` gives it.
pub fn blake2b(data: &[u8], outlen: usize) -> Vec<u8> {
    let outlen = outlen.clamp(1, 64);
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ outlen as u64;
    let mut counter: u128 = 0;
    let mut block = [0u8; 128];
    let mut rest = data;
    // every block but the last is compressed as it fills; the last one - full
    // or not, empty input included - carries the final flag
    while rest.len() > 128 {
        block.copy_from_slice(&rest[..128]);
        counter += 128;
        compress(&mut h, &block, counter, false);
        rest = &rest[128..];
    }
    block = [0u8; 128];
    block[..rest.len()].copy_from_slice(rest);
    counter += rest.len() as u128;
    compress(&mut h, &block, counter, true);
    let mut out = Vec::with_capacity(64);
    for word in h {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.truncate(outlen);
    out
}

/// [`blake2b`] as lower-case hex: `hashlib.blake2b(data, digest_size=outlen).hexdigest()`.
pub fn hexdigest(data: &[u8], outlen: usize) -> String {
    blake2b(data, outlen).iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_matches_hashlib_across_block_edges() {
        assert_eq!(
            hexdigest(b"", 64),
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
             d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
        );
        // the utterance token's digest: hashlib.blake2b(b"example", digest_size=4).hexdigest()
        assert_eq!(hexdigest(b"example", 4), "0da00195");
        // longer than one block, exactly one block, one byte past it, exactly two
        let long: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        assert_eq!(hexdigest(&long, 4), "dd84ce7c");
        assert_eq!(hexdigest(&long[..128], 8), "c2d13df1b6617e82");
        assert_eq!(hexdigest(&long[..129], 8), "f667e47a6d5350a6");
        assert_eq!(hexdigest(&long[..256], 16), "c2472c0ac37a8dbdb25f05ada0d82643");
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn the_rfc_vector() {
        assert_eq!(
            hex(&blake2b(b"abc", 64)),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
    }

    #[test]
    fn the_short_digests_python_takes() {
        // python3 -c "import hashlib; print(hashlib.blake2b(b'...', digest_size=8).hexdigest())"
        assert_eq!(hex(&blake2b(b"", 8)), "e4a6a0577479b2b4");
        assert_eq!(hex(&blake2b(b"abc", 8)), "d8bb14d833d59559");
        assert_eq!(hex(&blake2b(b"the", 8)), "5edaab6c90973a2e");
        assert_eq!(hex(&blake2b("é t".as_bytes(), 8)), "c19e082f9b736b33");
        assert_eq!(hex(&blake2b(&[b'x'; 200], 8)), "56304c0e1b76a18d");
    }
}
