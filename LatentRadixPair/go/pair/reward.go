package pair

import (
	"fmt"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/tokenizer"
)

// RewardTree holds, cell for cell with the count tree, what outcomes earned and what they were punished
// for, kept apart: a punishment is never bought back by a reward. It is written only by judged outcomes.
type RewardTree struct {
	Addr                         *Address
	Plus, Minus                  []float64
	Judged                       int
	RewardsTotal, PenaltiesTotal float64
	Version                      int
	chain                        []int
}

// NewRewardTree allocates the cells.
func NewRewardTree(addr *Address) *RewardTree {
	return &RewardTree{Addr: addr, Plus: make([]float64, addr.Cells()), Minus: make([]float64, addr.Cells())}
}

// Credit adds amount (negative: a punishment) to the cells of a judged text's steps from position skip on
// (the earlier positions are context, not outcome), at every level (rungs "all") or the full code only
// ("final"). It returns the cells credited.
func (r *RewardTree) Credit(codes tokenizer.Codes, outcomes []int, amount float64, rungs string, skip int) (int, error) {
	if rungs != "all" && rungs != "final" {
		return 0, fmt.Errorf("rungs must be all or final, got %q", rungs)
	}
	if amount == 0 {
		return 0, nil
	}
	target, a := r.Plus, amount
	if amount < 0 {
		target, a = r.Minus, -amount
	}
	if skip < 0 {
		skip = 0
	}
	n := 0
	for i := skip; i < len(outcomes); i++ {
		x := outcomes[i]
		if x < 0 || x >= r.Addr.Out {
			return n, fmt.Errorf("outcome %d outside 0..%d", x, r.Addr.Out-1)
		}
		r.chain = r.Addr.Chain(codes.At(i), r.chain)
		if rungs == "all" {
			for _, node := range r.chain {
				target[r.Addr.Cell(node, x)] += a
			}
			n += len(r.chain)
		} else {
			target[r.Addr.Cell(r.chain[len(r.chain)-1], x)] += a
			n++
		}
	}
	if amount > 0 {
		r.RewardsTotal += a * float64(n)
	} else {
		r.PenaltiesTotal += a * float64(n)
	}
	r.Judged++
	r.Version++
	return n, nil
}

// Invert swaps rewards and penalties; an involution.
func (r *RewardTree) Invert() {
	r.Plus, r.Minus = r.Minus, r.Plus
	r.RewardsTotal, r.PenaltiesTotal = r.PenaltiesTotal, r.RewardsTotal
	r.Version++
}

// Reward is the net of one cell; Penalty its penalty side.
func (r *RewardTree) Reward(cell int) float64  { return r.Plus[cell] - r.Minus[cell] }
func (r *RewardTree) Penalty(cell int) float64 { return r.Minus[cell] }

// Rewards are a node's net rewards per outcome, into out; Penalties its penalties.
func (r *RewardTree) Rewards(node int, out []float64) []float64 {
	start := r.Addr.Cell(node, 0)
	out = out[:0]
	for i := start; i < start+r.Addr.Out; i++ {
		out = append(out, r.Plus[i]-r.Minus[i])
	}
	return out
}

func (r *RewardTree) Penalties(node int, out []float64) []float64 {
	start := r.Addr.Cell(node, 0)
	return append(out[:0], r.Minus[start:start+r.Addr.Out]...)
}

// Nonzero calls fn for every cell with a reward or a penalty, in cell order.
func (r *RewardTree) Nonzero(fn func(cell int, plus, minus float64)) {
	for i := range r.Plus {
		if r.Plus[i] != 0 || r.Minus[i] != 0 {
			fn(i, r.Plus[i], r.Minus[i])
		}
	}
}
