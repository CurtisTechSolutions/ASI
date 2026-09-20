package radixnet

import (
	"fmt"
	"math"
	"sync/atomic"
)

// Smoothing is the additive smoothing of both frequency ratios.
const Smoothing = 0.5

// EdgeWeight is the dual frequency function for one edge:
//
//	R_all    = (count + s) / (parentTotal + s * degree)
//	R_recent = (windowCount + s) / (windowTotal + s * degree)
//	weight   = countScale * log(1 + count) + globalScale * log(R_all)
//	         + windowScale * log(R_recent) + rewardScale * reward
func (g *Graph) EdgeWeight(count, reward, parentTotal float64, degree int, windowCount, windowTotal float64) float64 {
	s := Smoothing
	count = math.Max(0, count)
	windowCount = math.Max(0, windowCount)
	if degree < 1 {
		degree = 1
	}
	d := float64(degree)
	rAll := (count + s) / (math.Max(0, parentTotal) + s*d)
	rRecent := (windowCount + s) / (math.Max(0, windowTotal) + s*d)
	return g.CountScale*math.Log1p(count) + g.GlobalScale*math.Log(rAll) + g.WindowScale*math.Log(rRecent) + g.RewardScale*reward
}

// WindowTraversals is the number of traversals currently inside the sliding window.
func (g *Graph) WindowTraversals() int { return len(g.window) - g.windowHead }

// RecordTraversals counts traversals of edges in order: all time (already
// counted by the walk), in the sliding window and in the global total, and
// marks the parents whose rows changed.
func (g *Graph) RecordTraversals(edges []int) int {
	limit := g.WindowSize
	for _, e := range edges {
		g.window = append(g.window, e)
		g.WindowEdgeCount[e]++
		g.dirty[g.EdgeParent[e]] = struct{}{}
		if len(g.window)-g.windowHead > limit {
			old := g.window[g.windowHead]
			g.windowHead++
			if old < len(g.WindowEdgeCount) && g.WindowEdgeCount[old] > 0 {
				g.WindowEdgeCount[old]--
				g.dirty[g.EdgeParent[old]] = struct{}{}
			}
		}
	}
	if g.windowHead > 4096 && g.windowHead > len(g.window)/2 {
		// slide the live part down inside the same array: a counting pass over a
		// huge corpus compacts millions of times and must not allocate here
		n := copy(g.window, g.window[g.windowHead:])
		g.window = g.window[:n]
		g.windowHead = 0
	}
	g.TotalTraversals.Add(int64(len(edges)))
	return len(edges)
}

// Configure changes the scales / window size and recomputes every weight (on a
// negative graph: the blame function's scales).
func (g *Graph) Configure(opts map[string]float64) error {
	if g.Neg != nil {
		return g.ConfigureNegative(opts)
	}
	for name, value := range opts {
		switch name {
		case "window":
			size := int(value)
			if size < 1 {
				return fmt.Errorf("window must be >= 1, got %v", value)
			}
			g.WindowSize = size
			for len(g.window)-g.windowHead > size {
				old := g.window[g.windowHead]
				g.windowHead++
				if old < len(g.WindowEdgeCount) && g.WindowEdgeCount[old] > 0 {
					g.WindowEdgeCount[old]--
				}
			}
		case "count_scale", "global_scale", "window_scale", "reward_scale", "path_scale":
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("%s must be a finite number, got %v", name, value)
			}
			switch name {
			case "count_scale":
				g.CountScale = value
			case "global_scale":
				g.GlobalScale = value
			case "window_scale":
				g.WindowScale = value
			case "reward_scale":
				g.RewardScale = value
			case "path_scale":
				g.PathScale = value
				g.ctxVersion = invalidStamp // every context is priced again
			}
		default:
			return fmt.Errorf("unknown weight option %q", name)
		}
	}
	g.dirtyAll = true
	g.flushWeights()
	return nil
}

// WeightConfig describes the weight function (the "weights" block of a model file).
type WeightConfig struct {
	Function    string  `json:"function"`
	CountScale  float64 `json:"count_scale"`
	GlobalScale float64 `json:"global_scale"`
	WindowScale float64 `json:"window_scale"`
	RewardScale float64 `json:"reward_scale"`
	PathScale   float64 `json:"path_scale"`
	Window      int     `json:"window"`
	Smoothing   float64 `json:"smoothing"`
}

// WeightConfig returns the current weight function settings.
func (g *Graph) WeightConfig() WeightConfig {
	return WeightConfig{
		Function: "dual-frequency", CountScale: g.CountScale, GlobalScale: g.GlobalScale,
		WindowScale: g.WindowScale, RewardScale: g.RewardScale, PathScale: g.PathScale,
		Window: g.WindowSize, Smoothing: Smoothing,
	}
}

// Share is one edge's ratio of its node's traversals, all time and in the window.
type Share struct {
	Child, Edge int
	All, Recent float64
}

// Shares lists the shares of p's edges.
func (g *Graph) Shares(p int) []Share {
	adj := &g.children[p]
	counts := make([]float64, len(adj.order))
	var total float64
	var recent int64
	for i := range adj.order {
		e := adj.edges[i]
		counts[i] = counterTotal(atomic.LoadInt64(&g.EdgeCount[e]), g.EdgeCountResets, e)
		total += counts[i]
		recent += g.WindowEdgeCount[e]
	}
	out := make([]Share, 0, len(adj.order))
	for i, c := range adj.order {
		e := adj.edges[i]
		s := Share{Child: c, Edge: e}
		if total > 0 {
			s.All = counts[i] / total
		}
		if recent > 0 {
			s.Recent = float64(g.WindowEdgeCount[e]) / float64(recent)
		}
		out = append(out, s)
	}
	return out
}

// recomputeRow writes the weight function to every edge leaving p: the dual
// frequency function of the count model, or the blame function of a negative
// graph.
func (g *Graph) recomputeRow(p int) {
	if g.Neg != nil {
		g.recomputeNegativeRow(p)
		return
	}
	adj := &g.children[p]
	if adj.size() == 0 || !g.Alive[p] {
		return
	}
	counts := make([]float64, len(adj.order))
	var total float64
	var recent int64
	for i := range adj.order {
		e := adj.edges[i]
		counts[i] = g.edgeTraversalsF(e)
		total += counts[i] // an explicit left-to-right sum, as in the Python implementation
		recent += g.WindowEdgeCount[e]
	}
	degree := adj.size()
	for i := range adj.order {
		e := adj.edges[i]
		g.EdgeW[e] = g.EdgeWeight(counts[i], g.EdgeReward[e], total, degree, float64(g.WindowEdgeCount[e]), float64(recent))
	}
}

// RecomputeWeights writes the weight function to every alive edge, in
// parallel over the nodes for large graphs.
func (g *Graph) RecomputeWeights() {
	n := len(g.Labels)
	workers := g.Workers
	if n < 4096 || workers == 1 {
		for p := 0; p < n; p++ {
			g.recomputeRow(p)
		}
	} else {
		parallelRanges(n, workers, func(lo, hi int) {
			for p := lo; p < hi; p++ {
				g.recomputeRow(p)
			}
		})
	}
	for k := range g.dirty {
		delete(g.dirty, k)
	}
	g.dirtyAll = false
	g.weightsStructure = g.StructureVersion
	g.Version.Add(1)
}

// flushWeights brings the weights up to date: every row after a structural
// change or a global change, otherwise only the rows of the touched parents.
func (g *Graph) flushWeights() {
	if g.dirtyAll || g.weightsStructure != g.StructureVersion {
		g.RecomputeWeights()
		return
	}
	if len(g.dirty) == 0 {
		return
	}
	if len(g.dirty) > len(g.Labels)/4 {
		g.RecomputeWeights()
		return
	}
	for p := range g.dirty {
		if p < len(g.Labels) {
			g.recomputeRow(p)
		}
		delete(g.dirty, p)
	}
	g.Version.Add(1)
}

// WeightsStale reports whether Prepare would change anything.
func (g *Graph) WeightsStale() bool {
	return g.dirtyAll || g.weightsStructure != g.StructureVersion || len(g.dirty) > 0
}

// AddReward adds amount (negative = penalty) to the reward of every listed
// alive edge; returns how many were touched.
func (g *Graph) AddReward(edges []int, amount float64) int {
	touched := 0
	for _, e := range edges {
		if e >= 0 && e < len(g.EdgeReward) && g.EdgeAlive[e] {
			g.EdgeReward[e] += amount
			g.dirty[g.EdgeParent[e]] = struct{}{}
			touched++
		}
	}
	return touched
}

// TotalReward is (sum of positive rewards, sum of negative rewards) over alive edges.
func (g *Graph) TotalReward() (pos, neg float64) {
	for e, ok := range g.EdgeAlive {
		if ok {
			r := g.EdgeReward[e]
			if r > 0 {
				pos += r
			} else if r < 0 {
				neg += r
			}
		}
	}
	return pos, neg
}

// Invert flips the sign of every reward - or, on a negative graph, swaps
// blame and clearing (see invertNegative).
func (g *Graph) Invert() {
	if g.Neg != nil {
		g.invertNegative()
		return
	}
	for e, ok := range g.EdgeAlive {
		if ok {
			g.EdgeReward[e] = -g.EdgeReward[e]
		}
	}
	g.Inverted = !g.Inverted
	g.dirtyAll = true
	g.flushWeights()
}

// -- costs -------------------------------------------------------------------------

// ChildCost is one out-edge of a node with its -log softmax probability and
// the punishment the model carries against that step (see EdgePunishment).
type ChildCost struct {
	Child, Edge int
	Cost        float64
	Punish      float64
}

// EdgePunishment is what the model has been taught *against* one edge: the
// penalty side of its reward, on the same scale the reward function uses.
//
// It is deliberately one-sided.  A rewarded edge is not *less* punished than an
// edge nothing was ever said about - it is exactly as unpunished, which is what
// lets the least-punished traversal (search.go) rank walks by what went wrong on
// them instead of by what went well.
func (g *Graph) EdgePunishment(e int) float64 {
	if e < 0 || e >= len(g.EdgeReward) {
		return 0
	}
	return g.RewardScale * math.Max(0, -g.EdgeReward[e])
}

// StepPunishment is the punishment of one step taken from prev: the edge's own,
// plus PathScale * log(1 + incorrect) for the walks that came from prev and
// were judged wrong here.  prev < 0 is a step with no judged context.
//
// The second term counts the failures alone - not the failures against the
// successes, the way the cost function's path term weighs them.  That is the
// point: a step that was wrong here once is a step that was wrong here, and no
// amount of being right afterwards makes it a step nothing is held against.
// Blame cannot be bought off, which is what the traversal is for.
func (g *Graph) StepPunishment(prev, e int) float64 {
	punish := g.EdgePunishment(e)
	if prev >= 0 && g.PathScale != 0 && len(g.paths) > 0 {
		punish += g.PathScale * math.Log1p(float64(g.PathIncorrect(prev, e)))
	}
	return punish
}

// ensureCosts recomputes the per-edge costs when weights or structure changed
// (parallel over the nodes for large graphs).
func (g *Graph) ensureCosts() {
	if g.costsVersion == g.Version && len(g.edgeCost) == len(g.EdgeW) {
		return
	}
	if len(g.edgeCost) != len(g.EdgeW) {
		g.edgeCost = make([]float64, len(g.EdgeW))
		g.edgePunish = make([]float64, len(g.EdgeW))
	}
	n := len(g.Labels)
	row := func(p int) {
		adj := &g.children[p]
		if adj.size() == 0 || !g.Alive[p] {
			return
		}
		m := math.Inf(-1)
		for i := range adj.order {
			if w := g.EdgeW[adj.edges[i]]; w > m {
				m = w
			}
		}
		sum := 0.0
		for i := range adj.order {
			sum += math.Exp(g.EdgeW[adj.edges[i]] - m)
		}
		lse := m + math.Log(sum)
		for i := range adj.order {
			e := adj.edges[i]
			g.edgeCost[e] = lse - g.EdgeW[e]
			g.edgePunish[e] = g.EdgePunishment(e)
		}
	}
	if n < 4096 || g.Workers == 1 {
		for p := 0; p < n; p++ {
			row(p)
		}
	} else {
		parallelRanges(n, g.Workers, func(lo, hi int) {
			for p := lo; p < hi; p++ {
				row(p)
			}
		})
	}
	g.costsVersion = g.Version
}

// Prepare brings weights and edge costs up to date.  Call it before reading
// costs from several goroutines; afterwards ChildCosts, Trace and the
// searches are safe to run concurrently as long as nothing mutates the graph.
func (g *Graph) Prepare() {
	g.flushWeights()
	g.ensureCosts()
	g.ensureContextCosts()
}

// ChildCostsFrom lists p's out-edges as a walk that arrived from prev sees
// them.  Without a judged context the costs are the edge costs; where a path
// *has* been judged, its context adds PathScale * log((correct + s) /
// (incorrect + s)) to that edge's weight before the softmax - so the same edge
// is cheap for the walk that was right here and dear for the one that was
// wrong, which is the whole point of counting paths instead of edges.
func (g *Graph) ChildCostsFrom(p, prev int) []ChildCost {
	if prev < 0 || g.PathScale == 0 || len(g.paths) == 0 {
		return g.ChildCosts(p)
	}
	if g.costsVersion != g.Version || g.WeightsStale() {
		g.Prepare()
	}
	if costs, ok := g.ctxCache[PathKey{prev, p}]; ok {
		return costs
	}
	return g.ChildCosts(p)
}

// ensureContextCosts prices every judged context, once, while the graph is
// still single-threaded: the searches then only read the table (the two beams
// run side by side).
func (g *Graph) ensureContextCosts() {
	if g.ctxVersion == g.Version {
		return
	}
	g.ctxVersion = g.Version
	g.ctxCache = nil
	if len(g.paths) == 0 || g.PathScale == 0 {
		return
	}
	pairs := map[PathKey]bool{} // (the node that called, the node whose children it prices)
	for key := range g.paths {
		if key.Edge >= 0 && key.Edge < len(g.EdgeParent) {
			pairs[PathKey{key.Prev, g.EdgeParent[key.Edge]}] = true
		}
	}
	cache := make(map[PathKey][]ChildCost, len(pairs))
	for pair := range pairs {
		p := pair.Edge // the parent node, as stored above
		if p < 0 || p >= len(g.children) || !g.Alive[p] {
			continue
		}
		adj := &g.children[p]
		if adj.size() == 0 {
			continue
		}
		weights := make([]float64, len(adj.order))
		punish := make([]float64, len(adj.order))
		m := math.Inf(-1)
		for i := range adj.order {
			e := adj.edges[i]
			weights[i] = g.EdgeW[e] + g.PathScale*g.PathTerm(pair.Prev, e)
			punish[i] = g.edgePunish[e] + g.PathScale*math.Log1p(float64(g.PathIncorrect(pair.Prev, e)))
			if weights[i] > m {
				m = weights[i]
			}
		}
		terms := make([]float64, len(weights))
		for i, w := range weights {
			terms[i] = math.Exp(w - m)
		}
		lse := m + math.Log(fsum(terms))
		costs := make([]ChildCost, len(adj.order))
		for i, c := range adj.order {
			costs[i] = ChildCost{c, adj.edges[i], lse - weights[i], punish[i]}
		}
		cache[pair] = costs
	}
	g.ctxCache = cache
}

// ChildCosts lists p's out-edges with their costs (-log softmax + nothing
// else); prepares the graph when it is stale.
func (g *Graph) ChildCosts(p int) []ChildCost {
	if g.costsVersion != g.Version || g.WeightsStale() {
		g.Prepare()
	}
	adj := &g.children[p]
	out := make([]ChildCost, len(adj.order))
	for i, c := range adj.order {
		e := adj.edges[i]
		out[i] = ChildCost{c, e, g.edgeCost[e], g.edgePunish[e]}
	}
	return out
}

// EdgeCost is the cost of one edge after Prepare.
func (g *Graph) EdgeCost(e int) float64 { return g.edgeCost[e] }
