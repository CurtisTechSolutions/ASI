package radixnet

import (
	"fmt"
	"sort"
	"sync"
)

// Prediction is the best path plus the top-K and bottom-K paths of one search.
type Prediction struct {
	PathResult
	Top    []*PathResult `json:"top"`
	Bottom []*PathResult `json:"bottom"`
	K      int           `json:"k"`
	Beam   int           `json:"beam"`
	Mode   string        `json:"mode"`
	// Traversal names the search that produced this, when it was not the
	// ordinary one by cost.
	Traversal string `json:"traversal,omitempty"`
}

// DefaultBeam is the beam width used when none is given.
func DefaultBeam(k int) int {
	if k < 1 {
		k = 1
	}
	if 4*k > 16 {
		return 4 * k
	}
	return 16
}

type beamState struct {
	cost   float64
	chars  int
	node   int
	entry  int
	punish float64 // the worst step on the way here (ByLeastPunished)
}

type beamEntry struct {
	node, parent int
	step         float64
}

type finished struct {
	cost   float64
	ids    []int
	step   []float64
	punish float64
}

func lessState(a, b beamState, traversal Traversal) bool {
	if traversal == ByLeastPunished && a.punish != b.punish {
		return a.punish < b.punish // the walk that has the least against it leads
	}
	if a.cost != b.cost {
		return a.cost < b.cost
	}
	if a.chars != b.chars {
		return a.chars < b.chars
	}
	if a.node != b.node {
		return a.node < b.node
	}
	return a.entry < b.entry
}

// runBeam is one beam: the k cheapest (or, with worst, dearest) complete paths.
// Under ByLeastPunished "cheapest" reads as "least punished, and cheapest among
// those": a path is ranked by its worst step first and by its summed cost only
// where two paths carry the same worst step.
func runBeam(g *Graph, startNode, startChars, minChars, cap, k, width int, stepPenalty float64, toEnd bool, maxSteps, maxExpansions int, worst bool, traversal Traversal) ([]finished, int) {
	entries := []beamEntry{{startNode, -1, 0}}
	sign := 1.0
	if worst {
		sign = -1.0
	}
	complete := func(node, chars int) bool {
		if node == End {
			return true
		}
		if cap >= 0 && chars >= cap {
			return true
		}
		return !toEnd && chars >= minChars
	}
	// A kept path is two numbers, higher is better in this beam's own direction:
	// what the walk was punished for and what it cost.  Under ByReward the first
	// is 0 on every path, so the whole comparison is the cost order the beam has
	// always used.
	type doneKey struct {
		first, second float64
		entry         int
	}
	keyOf := func(cost, punish float64, entry int) doneKey {
		first := 0.0
		if traversal == ByLeastPunished {
			first = -punish * sign
		}
		return doneKey{first, -cost * sign, entry}
	}
	betterThan := func(a, b doneKey) bool {
		if a.first != b.first {
			return a.first > b.first
		}
		return a.second > b.second
	}
	done := make([]doneKey, 0, k)
	worstIndex := func() int { // the kept path that ranks last: the first to be replaced
		best := 0
		for i := 1; i < len(done); i++ {
			a, b := done[i], done[best]
			if a.first != b.first {
				if a.first < b.first {
					best = i
				}
				continue
			}
			if a.second != b.second {
				if a.second < b.second {
					best = i
				}
				continue
			}
			if a.entry < b.entry {
				best = i
			}
		}
		return best
	}
	offer := func(cost, punish float64, entry int) {
		if k == 0 {
			return
		}
		key := keyOf(cost, punish, entry)
		if len(done) < k {
			done = append(done, key)
			return
		}
		w := worstIndex()
		if betterThan(key, done[w]) {
			done[w] = key
		}
	}
	frontier := []beamState{}
	start := beamState{0, startChars, startNode, 0, 0}
	if complete(startNode, startChars) {
		offer(0, 0, 0)
	} else {
		frontier = append(frontier, start)
	}
	fallback := frontier
	expanded, steps := 0, 0
	for len(frontier) > 0 && steps < maxSteps && expanded < maxExpansions {
		steps++
		candidates := make([]beamState, 0, len(frontier)*2)
		for _, st := range frontier {
			expanded++
			// where the walk came from: a judged path prices the next step by it
			prev := -1
			if parent := entries[st.entry].parent; parent >= 0 {
				prev = entries[parent].node
			} else if st.node == Start {
				prev = Start
			}
			children := Onward(g.ChildCostsFrom(st.node, prev))
			if traversal == ByLeastPunished {
				children = LeastPunished(children)
			}
			for _, cc := range children {
				nchars := st.chars
				if cc.Child != End {
					nchars += g.labelLen[cc.Child] - Overlap
				}
				step := cc.Cost + stepPenalty
				ncost := st.cost + step
				npunish := st.punish
				if cc.Punish > npunish {
					npunish = cc.Punish // a path is as punished as its worst step
				}
				childEntry := len(entries)
				entries = append(entries, beamEntry{cc.Child, st.entry, step})
				if complete(cc.Child, nchars) {
					offer(ncost, npunish, childEntry)
				} else {
					candidates = append(candidates, beamState{ncost, nchars, cc.Child, childEntry, npunish})
				}
			}
			if expanded >= maxExpansions {
				break
			}
		}
		if len(candidates) > 0 {
			fallback = candidates
		}
		if worst {
			sort.Slice(candidates, func(i, j int) bool { return lessState(candidates[j], candidates[i], traversal) })
		} else {
			sort.Slice(candidates, func(i, j int) bool { return lessState(candidates[i], candidates[j], traversal) })
		}
		if len(candidates) > width {
			candidates = candidates[:width]
		}
		frontier = candidates
		// neither number shrinks along a path - a cost is a sum of costs and a
		// punishment the worst step so far - so once k paths finished and the best
		// partial one already ranks below the k-th of them, the best side is settled
		if !worst && k > 0 && len(done) == k && len(frontier) > 0 {
			w := worstIndex()
			head := keyOf(frontier[0].cost, frontier[0].punish, frontier[0].entry)
			if !betterThan(head, done[w]) {
				break
			}
		}
	}
	pathOf := func(entry int) ([]int, []float64) {
		ids := []int{}
		stepsOut := []float64{}
		for entry >= 0 {
			en := entries[entry]
			ids = append(ids, en.node)
			if en.parent >= 0 {
				stepsOut = append(stepsOut, en.step)
			}
			entry = en.parent
		}
		for i, j := 0, len(ids)-1; i < j; i, j = i+1, j-1 {
			ids[i], ids[j] = ids[j], ids[i]
		}
		for i, j := 0, len(stepsOut)-1; i < j; i, j = i+1, j-1 {
			stepsOut[i], stepsOut[j] = stepsOut[j], stepsOut[i]
		}
		return ids, stepsOut
	}
	type pick struct {
		cost   float64
		punish float64
		entry  int
	}
	var picks []pick
	if len(done) > 0 {
		for _, d := range done {
			punish := -d.first * sign
			if punish == 0 {
				punish = 0 // a negative zero reads as a punishment that is not there
			}
			picks = append(picks, pick{-d.second * sign, punish, d.entry})
		}
		sort.Slice(picks, func(i, j int) bool {
			if traversal == ByLeastPunished && picks[i].punish != picks[j].punish {
				return picks[i].punish*sign < picks[j].punish*sign
			}
			a, b := picks[i].cost*sign, picks[j].cost*sign
			if a != b {
				return a < b
			}
			return picks[i].entry < picks[j].entry
		})
	} else if len(fallback) > 0 {
		// nothing completed within the limits: the surviving partial paths, most characters first
		fb := append([]beamState(nil), fallback...)
		sort.Slice(fb, func(i, j int) bool {
			if fb[i].chars != fb[j].chars {
				return fb[i].chars > fb[j].chars
			}
			if traversal == ByLeastPunished && fb[i].punish != fb[j].punish {
				return fb[i].punish*sign < fb[j].punish*sign
			}
			a, b := fb[i].cost*sign, fb[j].cost*sign
			if a != b {
				return a < b
			}
			return fb[i].entry < fb[j].entry
		})
		if len(fb) > k {
			fb = fb[:k]
		}
		for _, s := range fb {
			picks = append(picks, pick{s.cost, s.punish, s.entry})
		}
	}
	out := make([]finished, 0, len(picks))
	for _, p := range picks {
		ids, st := pathOf(p.entry)
		out = append(out, finished{p.cost, ids, st, p.punish})
	}
	return out, expanded
}

// BeamOptions configure BeamPredict.
type BeamOptions struct {
	K             int
	Beam          int // 0 = DefaultBeam(K)
	MaxChars      int // -1 = no cap
	StepPenalty   float64
	ToEnd         bool
	MaxSteps      int // 0 = default (500 with ToEnd, else MinChars + 50)
	MaxExpansions int // 0 = 200000
	// Traversal is how the two beams rank a path: ByReward (the default) by
	// cost, ByLeastPunished by the blame on its worst step first.
	Traversal Traversal
}

// BeamPredict returns the k cheapest complete paths (rising cost) and the k
// most expensive ones not among them (falling cost), plus the expansions.
// The two beams run concurrently when the bottom side's cap is known up front.
func (g *Graph) BeamPredict(startNode, startOffset, minChars int, opts BeamOptions) ([]*PathResult, []*PathResult, int, error) {
	if opts.K < 0 {
		return nil, nil, 0, fmt.Errorf("k must be >= 0, got %d", opts.K)
	}
	if opts.StepPenalty < 0 {
		return nil, nil, 0, fmt.Errorf("step_penalty must be >= 0")
	}
	width := opts.Beam
	if width == 0 {
		width = DefaultBeam(opts.K)
	}
	if width < 1 {
		return nil, nil, 0, fmt.Errorf("beam must be >= 1, got %d", opts.Beam)
	}
	maxChars := opts.MaxChars
	if maxChars >= 0 && maxChars < minChars {
		maxChars = minChars
	}
	maxSteps := opts.MaxSteps
	if maxSteps == 0 {
		if opts.ToEnd {
			maxSteps = 500
		} else {
			maxSteps = minChars + 50
		}
	}
	maxExp := opts.MaxExpansions
	if maxExp == 0 {
		maxExp = 200_000
	}
	g.Prepare()
	startChars, err := startEmission(g, startNode, startOffset)
	if err != nil {
		return nil, nil, 0, err
	}
	if opts.K == 0 {
		return []*PathResult{}, []*PathResult{}, 0, nil
	}
	var best, worstPaths []finished
	var expanded, expandedWorst int
	bottomCap := maxChars
	if (bottomCap < 0 && opts.ToEnd) || g.Workers == 1 {
		// the bottom cap depends on the best side, or one worker was asked for:
		// run the two beams in turn
		best, expanded = runBeam(g, startNode, startChars, minChars, maxChars, opts.K, width, opts.StepPenalty, opts.ToEnd, maxSteps, maxExp, false, opts.Traversal)
		if bottomCap < 0 && opts.ToEnd {
			longest, emitted := 0, 0
			for _, f := range best {
				sum := 0
				for _, n := range f.ids[1:] {
					if l := g.labelLen[n]; l > longest {
						longest = l
					}
					if n != End {
						sum += g.labelLen[n] - Overlap
					}
				}
				if sum > emitted {
					emitted = sum
				}
			}
			bottomCap = 2*emitted + longest + 8
			if minChars > bottomCap {
				bottomCap = minChars
			}
			if bottomCap < 16 {
				bottomCap = 16
			}
		}
		worstPaths, expandedWorst = runBeam(g, startNode, startChars, minChars, bottomCap, opts.K, width, opts.StepPenalty, opts.ToEnd, maxSteps, maxExp, true, opts.Traversal)
	} else {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			best, expanded = runBeam(g, startNode, startChars, minChars, maxChars, opts.K, width, opts.StepPenalty, opts.ToEnd, maxSteps, maxExp, false, opts.Traversal)
		}()
		go func() {
			defer wg.Done()
			worstPaths, expandedWorst = runBeam(g, startNode, startChars, minChars, bottomCap, opts.K, width, opts.StepPenalty, opts.ToEnd, maxSteps, maxExp, true, opts.Traversal)
		}()
		wg.Wait()
	}
	expanded += expandedWorst
	seen := make(map[string]bool, len(best))
	top := make([]*PathResult, 0, len(best))
	for _, f := range best {
		seen[fmt.Sprint(f.ids)] = true
		result := buildResult(g, f.ids, f.step, startOffset, maxChars, expanded, nil)
		result.Punish = f.punish
		top = append(top, result)
	}
	bottom := make([]*PathResult, 0, len(worstPaths))
	for _, f := range worstPaths {
		if !seen[fmt.Sprint(f.ids)] {
			result := buildResult(g, f.ids, f.step, startOffset, bottomCap, expanded, nil)
			result.Punish = f.punish
			bottom = append(bottom, result)
		}
	}
	return top, bottom, expanded, nil
}
