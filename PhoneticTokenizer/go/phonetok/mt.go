package phonetok

// A Mersenne Twister with the semantics of CPython's random.Random for what
// the phonotactics draw: the same seeding (init_by_array over the 32-bit chunks
// of |seed|), the same 53-bit doubles (random()) and the same randrange(n)
// (rejection sampling over getrandbits), so that a word coined with a seed in
// one port is the word coined with that seed in the others.

const (
	mtN         = 624
	mtM         = 397
	mtMatrixA   = 0x9908b0df
	mtUpperMask = 0x80000000
	mtLowerMask = 0x7fffffff
)

// Rng is what the phonotactics need of a random source.
type Rng interface {
	Float64() float64
	RandBelow(n int) int
}

// MT is a Python-compatible pseudo random generator.
type MT struct {
	mt    [mtN]uint32
	index int
}

// NewMT seeds a generator like random.Random(seed) for a non-negative integer seed.
func NewMT(seed uint32) *MT {
	r := &MT{}
	r.initByArray([]uint32{seed})
	return r
}

// NewMT64 seeds like random.Random(seed) for a 64-bit non-negative seed.
func NewMT64(seed uint64) *MT {
	r := &MT{}
	key := []uint32{uint32(seed & 0xffffffff)}
	if hi := uint32(seed >> 32); hi != 0 {
		key = append(key, hi)
	}
	r.initByArray(key)
	return r
}

func (r *MT) initGenrand(s uint32) {
	r.mt[0] = s
	for i := 1; i < mtN; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.index = mtN
}

func (r *MT) initByArray(key []uint32) {
	r.initGenrand(19650218)
	i, j := 1, 0
	k := mtN
	if len(key) > k {
		k = len(key)
	}
	for ; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= mtN {
			r.mt[0] = r.mt[mtN-1]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k = mtN - 1; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1566083941)) - uint32(i)
		i++
		if i >= mtN {
			r.mt[0] = r.mt[mtN-1]
			i = 1
		}
	}
	r.mt[0] = 0x80000000
	r.index = mtN
}

// Uint32 is the next 32-bit output.
func (r *MT) Uint32() uint32 {
	if r.index >= mtN {
		var kk int
		for kk = 0; kk < mtN-mtM; kk++ {
			y := (r.mt[kk] & mtUpperMask) | (r.mt[kk+1] & mtLowerMask)
			r.mt[kk] = r.mt[kk+mtM] ^ (y >> 1) ^ (mtMatrixA * (y & 1))
		}
		for ; kk < mtN-1; kk++ {
			y := (r.mt[kk] & mtUpperMask) | (r.mt[kk+1] & mtLowerMask)
			r.mt[kk] = r.mt[kk+mtM-mtN] ^ (y >> 1) ^ (mtMatrixA * (y & 1))
		}
		y := (r.mt[mtN-1] & mtUpperMask) | (r.mt[0] & mtLowerMask)
		r.mt[mtN-1] = r.mt[mtM-1] ^ (y >> 1) ^ (mtMatrixA * (y & 1))
		r.index = 0
	}
	y := r.mt[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// Float64 is random(): a double in [0, 1) from 53 bits, as CPython makes it.
func (r *MT) Float64() float64 {
	a := r.Uint32() >> 5
	b := r.Uint32() >> 6
	return (float64(a)*67108864.0 + float64(b)) * (1.0 / 9007199254740992.0)
}

// getrandbits is CPython's for k in 1..32.
func (r *MT) getrandbits(k int) uint32 {
	return r.Uint32() >> (32 - k)
}

// RandBelow is randrange(n) for 0 < n <= 2^32 - 1: rejection sampling over getrandbits(n.bit_length()).
func (r *MT) RandBelow(n int) int {
	if n <= 0 {
		panic("randrange of a non-positive n")
	}
	k := 0
	for v := n; v > 0; v >>= 1 {
		k++
	}
	for {
		v := int(r.getrandbits(k))
		if v < n {
			return v
		}
	}
}
