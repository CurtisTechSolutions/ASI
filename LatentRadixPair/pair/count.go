package pair

import "github.com/CurtisTechSolutions/ASI/LatentRadixPair/tokenizer"

// CountTree counts, under every prefix of a position's code, the outcome that followed. It is written only
// by reading.
type CountTree struct {
	Addr             *Address
	Cnt              []int64 // per cell
	Ctx              []int64 // per node: how often it was a context
	Texts, Units     int
	Version          int
	Alpha, Smoothing float64
	chain            []int
}

// NewCountTree allocates the cells.
func NewCountTree(addr *Address, alpha, smoothing float64) *CountTree {
	return &CountTree{Addr: addr, Cnt: make([]int64, addr.Cells()), Ctx: make([]int64, addr.N), Alpha: alpha, Smoothing: smoothing}
}

// Observe reads one text: outcomes[i] followed the context whose code is codes.At(i); every prefix of that
// code counts it.
func (t *CountTree) Observe(codes tokenizer.Codes, outcomes []int) {
	for i, x := range outcomes {
		t.chain = t.Addr.Chain(codes.At(i), t.chain)
		for _, node := range t.chain {
			t.Cnt[t.Addr.Cell(node, x)]++
			t.Ctx[node]++
		}
	}
	t.Texts++
	t.Units += len(outcomes)
	t.Version++
}

// Children are a node's outcome counts, into out.
func (t *CountTree) Children(node int, out []int64) []int64 {
	start := t.Addr.Cell(node, 0)
	return append(out[:0], t.Cnt[start:start+t.Addr.Out]...)
}

// Share is (count + s) / (ctx + s * Out) per outcome, into out; all zero when nothing was read and s == 0.
func (t *CountTree) Share(node int, out []float64) []float64 {
	start := t.Addr.Cell(node, 0)
	out = out[:0]
	denom := float64(t.Ctx[node]) + t.Smoothing*float64(t.Addr.Out)
	for _, c := range t.Cnt[start : start+t.Addr.Out] {
		if denom <= 0 {
			out = append(out, 0)
		} else {
			out = append(out, (float64(c)+t.Smoothing)/denom)
		}
	}
	return out
}

// Own is ctx / (ctx + alpha): the share of the answer a context keeps before falling back to its parent.
func (t *CountTree) Own(node int) float64 {
	c := float64(t.Ctx[node])
	if c <= 0 {
		return 0
	}
	return c / (c + t.Alpha)
}

// Coverage is how much of a text's contexts this tree has read: over its positions and code levels, the
// share of nodes with a context count, each level weighted by its depth (a full code seen before is more
// evidence than a coarse one).
func (t *CountTree) Coverage(codes tokenizer.Codes) float64 {
	var seen, of float64
	for i := 0; i < codes.Len(); i++ {
		t.chain = t.Addr.Chain(codes.At(i), t.chain)
		for l := 1; l < len(t.chain); l++ {
			w := float64(l)
			of += w
			if t.Ctx[t.chain[l]] > 0 {
				seen += w
			}
		}
	}
	if of == 0 {
		return 0
	}
	return seen / of
}

// Nonzero calls fn for every non-zero cell, in cell order.
func (t *CountTree) Nonzero(fn func(cell int, c int64)) {
	for i, c := range t.Cnt {
		if c != 0 {
			fn(i, c)
		}
	}
}
