// Package pair is the primed radix pair over the tokenizer's codes: a count tree and a reward tree that
// share one address space, whose nodes are every prefix of a context code (the empty prefix is the root),
// each holding the outcomes that followed contexts with that code: the 256 bytes and the end mark.
//
// Priming is brute force: every code prefix has a node before anything is read, so a context's node is a
// sum, never a search, and the parent of a node is the same context described one symbol more coarsely. That
// parent chain is what the fold backs off along.
package pair

import (
	"fmt"
	"sort"
)

// End is the outcome that closes a text; Out is the outcome alphabet: the bytes and End.
const (
	End = 256
	Out = 257
)

// Address numbers the context nodes.
type Address struct {
	Radices []int // radix of each code symbol
	Out     int   // outcomes per node
	Bases   []int // Bases[l] is the first node of level l; Bases[D+1] is N
	N       int   // context nodes
}

// NewAddress lays out the levels and refuses an address space above cellCeiling cells.
func NewAddress(radices []int, out int, cellCeiling int) (*Address, error) {
	if len(radices) == 0 {
		return nil, fmt.Errorf("a code needs at least one symbol")
	}
	a := &Address{Radices: append([]int(nil), radices...), Out: out, Bases: make([]int, len(radices)+2)}
	width := 1
	for l, r := range radices {
		if r < 1 {
			return nil, fmt.Errorf("symbol %d has radix %d", l, r)
		}
		a.Bases[l+1] = a.Bases[l] + width
		width *= r
		if width > cellCeiling {
			return nil, fmt.Errorf("level %d alone needs %d nodes, above the ceiling", l+1, width)
		}
	}
	a.N = a.Bases[len(radices)] + width
	a.Bases[len(radices)+1] = a.N
	if cells := a.N * out; cells > cellCeiling {
		return nil, fmt.Errorf("%d context nodes x %d outcomes = %d cells, above the ceiling of %d", a.N, out, cells, cellCeiling)
	}
	return a, nil
}

// D is the code depth.
func (a *Address) D() int { return len(a.Radices) }

// Cells is the number of (node, outcome) cells each tree holds.
func (a *Address) Cells() int { return a.N * a.Out }

// Of is the node of the first l symbols of code.
func (a *Address) Of(code []int32, l int) int {
	v := 0
	for j := 0; j < l; j++ {
		v = v*a.Radices[j] + int(code[j])
	}
	return a.Bases[l] + v
}

// Chain fills out with the node of every prefix of code: the root first, the full code last (D+1 nodes).
func (a *Address) Chain(code []int32, out []int) []int {
	out = out[:0]
	v := 0
	out = append(out, 0)
	for j, r := range a.Radices {
		v = v*r + int(code[j])
		out = append(out, a.Bases[j+1]+v)
	}
	return out
}

// Level is how many symbols a node's prefix has.
func (a *Address) Level(node int) int { return sort.SearchInts(a.Bases, node+1) - 1 }

// Code is the symbols of a node's prefix.
func (a *Address) Code(node int) []int32 {
	l := a.Level(node)
	v := node - a.Bases[l]
	out := make([]int32, l)
	for j := l - 1; j >= 0; j-- {
		out[j] = int32(v % a.Radices[j])
		v /= a.Radices[j]
	}
	return out
}

// Parent is the node one symbol coarser; the root is its own parent.
func (a *Address) Parent(node int) int {
	l := a.Level(node)
	if l == 0 {
		return 0
	}
	return a.Bases[l-1] + (node-a.Bases[l])/a.Radices[l-1]
}

// Cell is the index of an outcome under a node.
func (a *Address) Cell(node, x int) int { return node*a.Out + x }

// Valid reports whether a code's symbols are within their radices.
func (a *Address) Valid(code []int32) bool {
	if len(code) != len(a.Radices) {
		return false
	}
	for j, c := range code {
		if c < 0 || int(c) >= a.Radices[j] {
			return false
		}
	}
	return true
}
