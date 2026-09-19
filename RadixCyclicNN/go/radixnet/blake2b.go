package radixnet

import "encoding/binary"

// BLAKE2b, enough of it for the utterance token.
//
// The token a spoken text hangs off is a short digest of the waveform
// (speech.go), and it is *in the text the model trains on*, so both languages
// have to produce the same one for the same recording or the two models diverge
// on the same input.  Python gets it from hashlib; Go's standard library does
// not have BLAKE2b and this module takes no dependencies, so here it is - RFC
// 7693, unkeyed, for the short digests the token needs.
//
// This is the same reasoning that put MT19937 and Shewchuk summation in this
// package by hand: parity is a property of the file format, not an optimisation.

var blake2bIV = [8]uint64{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
}

var blake2bSigma = [12][16]byte{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
}

func rotr64(x uint64, n uint) uint64 { return x>>n | x<<(64-n) }

// blake2bCompress mixes one 128-byte block into h; t is the byte counter and
// last marks the final block.
func blake2bCompress(h *[8]uint64, block []byte, t uint64, last bool) {
	var m [16]uint64
	for i := 0; i < 16; i++ {
		m[i] = binary.LittleEndian.Uint64(block[i*8 : i*8+8])
	}
	var v [16]uint64
	copy(v[:8], h[:])
	copy(v[8:], blake2bIV[:])
	v[12] ^= t
	// t is 128-bit in the spec; a token's input never reaches 2^64 bytes, so the
	// high half (v[13]) stays zero.
	if last {
		v[14] = ^v[14]
	}
	mix := func(a, b, c, d int, x, y uint64) {
		v[a] += v[b] + x
		v[d] = rotr64(v[d]^v[a], 32)
		v[c] += v[d]
		v[b] = rotr64(v[b]^v[c], 24)
		v[a] += v[b] + y
		v[d] = rotr64(v[d]^v[a], 16)
		v[c] += v[d]
		v[b] = rotr64(v[b]^v[c], 63)
	}
	for round := 0; round < 12; round++ {
		s := &blake2bSigma[round]
		mix(0, 4, 8, 12, m[s[0]], m[s[1]])
		mix(1, 5, 9, 13, m[s[2]], m[s[3]])
		mix(2, 6, 10, 14, m[s[4]], m[s[5]])
		mix(3, 7, 11, 15, m[s[6]], m[s[7]])
		mix(0, 5, 10, 15, m[s[8]], m[s[9]])
		mix(1, 6, 11, 12, m[s[10]], m[s[11]])
		mix(2, 7, 8, 13, m[s[12]], m[s[13]])
		mix(3, 4, 9, 14, m[s[14]], m[s[15]])
	}
	for i := 0; i < 8; i++ {
		h[i] ^= v[i] ^ v[i+8]
	}
}

// Blake2b is the unkeyed BLAKE2b digest of data, size bytes long (1..64) -
// hashlib.blake2b(data, digest_size=size).digest().
func Blake2b(data []byte, size int) []byte {
	if size < 1 {
		size = 1
	}
	if size > 64 {
		size = 64
	}
	h := blake2bIV
	h[0] ^= 0x01010000 ^ uint64(size) // no key, fanout 1, depth 1

	var counter uint64
	at := 0
	// every block but the last goes through as a full 128 bytes
	for len(data)-at > 128 {
		counter += 128
		blake2bCompress(&h, data[at:at+128], counter, false)
		at += 128
	}
	last := make([]byte, 128)
	tail := copy(last, data[at:])
	counter += uint64(tail)
	blake2bCompress(&h, last, counter, true)

	out := make([]byte, 64)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint64(out[i*8:], h[i])
	}
	return out[:size]
}
