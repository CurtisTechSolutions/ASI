package radixnet

// A Mersenne Twister with the exact semantics of CPython's random.Random:
// the same seeding (init_by_array over the 32-bit chunks of |seed|), the same
// 53-bit doubles and the same state layout, so an rng_state written by either
// implementation continues the other's sequence.

import "fmt"

const (
	mtN         = 624
	mtM         = 397
	mtMatrixA   = 0x9908b0df
	mtUpperMask = 0x80000000
	mtLowerMask = 0x7fffffff
)

// MT19937 is a Python-compatible pseudo random generator.
type MT19937 struct {
	mt    [mtN]uint32
	index int
}

// NewMT19937 seeds a generator like random.Random(seed).
func NewMT19937(seed int64) *MT19937 {
	r := &MT19937{}
	r.Seed(seed)
	return r
}

// Seed reseeds like random.seed(seed) for an integer seed: the key is |seed|
// split into 32-bit words, least significant first (a zero seed gives [0]).
func (r *MT19937) Seed(seed int64) {
	n := seed
	if n < 0 {
		n = -n
	}
	key := []uint32{}
	u := uint64(n)
	for {
		key = append(key, uint32(u&0xffffffff))
		u >>= 32
		if u == 0 {
			break
		}
	}
	r.initByArray(key)
}

func (r *MT19937) initGenrand(s uint32) {
	r.mt[0] = s
	for i := 1; i < mtN; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.index = mtN
}

func (r *MT19937) initByArray(key []uint32) {
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

// Uint32 returns the next 32-bit output (genrand_uint32).
func (r *MT19937) Uint32() uint32 {
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

// Float64 returns the next double in [0, 1) exactly like random.random().
func (r *MT19937) Float64() float64 {
	a := r.Uint32() >> 5
	b := r.Uint32() >> 6
	return (float64(a)*67108864.0 + float64(b)) * (1.0 / 9007199254740992.0)
}

// Uniform returns a + (b - a) * random(), like random.uniform.
func (r *MT19937) Uniform(a, b float64) float64 { return a + (b-a)*r.Float64() }

// State returns the generator state in Python's getstate() layout:
// [3, [624 words..., index], null].
func (r *MT19937) State() []any {
	words := make([]any, mtN+1)
	for i, w := range r.mt {
		words[i] = uint64(w)
	}
	words[mtN] = r.index
	return []any{3, words, nil}
}

// SetState restores a state produced by State() or by Python's getstate().
func (r *MT19937) SetState(state []any) error {
	if len(state) < 2 {
		return fmt.Errorf("rng_state must have at least 2 entries")
	}
	words, ok := state[1].([]any)
	if !ok || len(words) != mtN+1 {
		return fmt.Errorf("rng_state must hold %d words plus the index", mtN)
	}
	var mt [mtN]uint32
	for i := 0; i < mtN; i++ {
		v, err := anyToUint64(words[i])
		if err != nil || v > 0xffffffff {
			return fmt.Errorf("rng_state word %d is not a 32-bit integer", i)
		}
		mt[i] = uint32(v)
	}
	idx, err := anyToUint64(words[mtN])
	if err != nil || idx > mtN {
		return fmt.Errorf("rng_state index out of range")
	}
	r.mt = mt
	r.index = int(idx)
	return nil
}

func anyToUint64(v any) (uint64, error) {
	switch x := v.(type) {
	case float64:
		if x < 0 || x != float64(uint64(x)) {
			return 0, fmt.Errorf("not an unsigned integer: %v", x)
		}
		return uint64(x), nil
	case uint64:
		return x, nil
	case int:
		if x < 0 {
			return 0, fmt.Errorf("negative")
		}
		return uint64(x), nil
	case int64:
		if x < 0 {
			return 0, fmt.Errorf("negative")
		}
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	}
	return 0, fmt.Errorf("unexpected type %T", v)
}
