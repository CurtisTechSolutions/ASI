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
		g.window = append(g.window[:0:0], g.window[g.windowHead:]...)
		g.windowHead = 0
	}
	g.TotalTraversals += int64(len(edges))
	return len(edges)
}

// Configure changes the scales / window size and recomputes every weight.
func (g *Graph) Configure(opts map[string]float64) error {
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
		case "count_scale", "global_scale", "window_scale", "reward_scale":
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
	Window      int     `json:"window"`
	Smoothing   float64 `json:"smoothing"`
}

// WeightConfig returns the current weight function settings.
func (g *Graph) WeightConfig() WeightConfig {
	return WeightConfig{
		Function: "dual-frequency", CountScale: g.CountScale, GlobalScale: g.GlobalScale,
		WindowScale: g.WindowScale, RewardScale: g.RewardScale, Window: g.WindowSize, Smoothing: Smoothing,
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
	var total, recent int64
	for _, c := range adj.order {
		e := adj.edge[c]
		total += atomic.LoadInt64(&g.EdgeCount[e])
		recent += g.WindowEdgeCount[e]
	}
	out := make([]Share, 0, len(adj.order))
	for _, c := range adj.order {
		e := adj.edge[c]
		s := Share{Child: c, Edge: e}
		if total > 0 {
			s.All = float64(atomic.LoadInt64(&g.EdgeCount[e])) / float64(total)
		}
		if recent > 0 {
			s.Recent = float64(g.WindowEdgeCount[e]) / float64(recent)
		}
		out = append(out, s)
	}
	return out
}

// recomputeRow writes the dual frequency weight to every edge leaving p.
func (g *Graph) recomputeRow(p int) {
	adj := &g.children[p]
	if adj.size() == 0 || !g.Alive[p] {
		return
	}
	var total, recent int64
	for _, c := range adj.order {
		e := adj.edge[c]
		total += g.EdgeCount[e]
		recent += g.WindowEdgeCount[e]
	}
	degree := adj.size()
	for _, c := range adj.order {
		e := adj.edge[c]
		g.EdgeW[e] = g.EdgeWeight(float64(g.EdgeCount[e]), g.EdgeReward[e], float64(total), degree, float64(g.WindowEdgeCount[e]), float64(recent))
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
	g.Version++
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
	g.Version++
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

// Invert flips the sign of every reward.
func (g *Graph) Invert() {
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

// ChildCost is one out-edge of a node with its -log softmax probability.
type ChildCost struct {
	Child, Edge int
	Cost        float64
}

// ensureCosts recomputes the per-edge costs when weights or structure changed
// (parallel over the nodes for large graphs).
func (g *Graph) ensureCosts() {
	if g.costsVersion == g.Version && len(g.edgeCost) == len(g.EdgeW) {
		return
	}
	if len(g.edgeCost) != len(g.EdgeW) {
		g.edgeCost = make([]float64, len(g.EdgeW))
	}
	n := len(g.Labels)
	row := func(p int) {
		adj := &g.children[p]
		if adj.size() == 0 || !g.Alive[p] {
			return
		}
		m := math.Inf(-1)
		for _, c := range adj.order {
			if w := g.EdgeW[adj.edge[c]]; w > m {
				m = w
			}
		}
		sum := 0.0
		for _, c := range adj.order {
			sum += math.Exp(g.EdgeW[adj.edge[c]] - m)
		}
		lse := m + math.Log(sum)
		for _, c := range adj.order {
			e := adj.edge[c]
			g.edgeCost[e] = lse - g.EdgeW[e]
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
		e := adj.edge[c]
		out[i] = ChildCost{c, e, g.edgeCost[e]}
	}
	return out
}

// EdgeCost is the cost of one edge after Prepare.
func (g *Graph) EdgeCost(e int) float64 { return g.edgeCost[e] }
