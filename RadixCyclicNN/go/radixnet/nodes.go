package radixnet

import (
	"math"
	"sort"
)

// A node against the nodes around it.
//
// An edge's counters say what that step did; they do not say what it did
// *here*, among the other ways out of the same node.  These are the ratios
// that answer that: for one node, a row per previous node and a row per next
// node, each carrying its share of the traffic on that side, its share of the
// reward on that side, and what the judged paths (paths.go) made of it.
//
// The denominators are the side's own, not the node's visits: a node is
// entered without an in-edge whenever a text starts on it, so Visits can be
// larger than everything In adds up to.

// NeighbourStats is one neighbour of a node, and how much of the node went that way.
type NeighbourStats struct {
	Node       int    `json:"node"`
	Label      string `json:"label"`
	Edge       int    `json:"edge"`
	Seen       int64  `json:"seen"`
	SeenResets int64  `json:"seen_resets"`
	// SeenRatio is this edge's share of the traversals on its side of the node.
	SeenRatio float64 `json:"seen_ratio"`
	Reward    float64 `json:"reward"`
	// RewardRatio is this edge's share of the reward *magnitude* on its side,
	// signed - so a penalty reads as a negative share of the pressure on the
	// node, and the two sides compare without the signs cancelling out.
	RewardRatio float64 `json:"reward_ratio"`
	// PathSeen is how much of the edge's traffic a judged context has been
	// watching, PathRatio that as a share of the edge's traversals.
	PathSeen     int64    `json:"path_seen"`
	PathRatio    *float64 `json:"path_ratio"`
	Correct      int64    `json:"correct"`
	Incorrect    int64    `json:"incorrect"`
	CorrectRatio *float64 `json:"correct_ratio"`
}

// SideTotals is what a whole in- or out-side of a node did.
type SideTotals struct {
	Edges        int      `json:"edges"`
	Seen         int64    `json:"seen"`
	Reward       float64  `json:"reward"`
	PathSeen     int64    `json:"path_seen"`
	Correct      int64    `json:"correct"`
	Incorrect    int64    `json:"incorrect"`
	CorrectRatio *float64 `json:"correct_ratio"`
}

// NodeStats is one node with both of its sides.
type NodeStats struct {
	Node        int              `json:"node"`
	Label       string           `json:"label"`
	Visits      int64            `json:"visits"`
	VisitResets int64            `json:"visit_resets"`
	From        []NeighbourStats `json:"from"`
	To          []NeighbourStats `json:"to"`
	InTotals    SideTotals       `json:"in_totals"`
	OutTotals   SideTotals       `json:"out_totals"`
}

// EdgePaths is (seen, correct, incorrect) of one edge, summed over every caller that reached it.
func (g *Graph) EdgePaths(edge int) (seen, correct, incorrect int64) {
	for prev := range g.pathsByEdge[edge] {
		if row, ok := g.paths[PathKey{prev, edge}]; ok {
			seen += row.Seen
			correct += row.Correct
			incorrect += row.Incorrect
		}
	}
	return seen, correct, incorrect
}

// sideRows is one row per neighbour: its share of the side's traffic and of the side's reward.
func (g *Graph) sideRows(pairs []Transition) []NeighbourStats {
	// by neighbour id, so both languages add the shares up in the same order
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].P != pairs[j].P {
			return pairs[i].P < pairs[j].P
		}
		return pairs[i].E < pairs[j].E
	})
	traversals := make(map[int]float64, len(pairs))
	total, mass := 0.0, 0.0
	for _, t := range pairs {
		traversals[t.E] = g.edgeTraversalsF(t.E)
	}
	for _, t := range pairs {
		total += traversals[t.E]
	}
	for _, t := range pairs {
		mass += math.Abs(g.EdgeReward[t.E])
	}
	rows := make([]NeighbourStats, 0, len(pairs))
	for _, t := range pairs {
		pathSeen, correct, incorrect := g.EdgePaths(t.E)
		reward := g.EdgeReward[t.E]
		row := NeighbourStats{
			Node: t.P, Label: g.TextOf(g.Label(t.P)), Edge: t.E,
			Seen: g.EdgeCount[t.E], SeenResets: g.EdgeCountResets[t.E],
			Reward: reward, PathSeen: pathSeen, Correct: correct, Incorrect: incorrect,
		}
		if total != 0 {
			row.SeenRatio = traversals[t.E] / total
		}
		if mass != 0 {
			row.RewardRatio = reward / mass
		}
		if walked := traversals[t.E]; walked != 0 {
			ratio := float64(pathSeen) / walked
			row.PathRatio = &ratio
		}
		if judged := correct + incorrect; judged > 0 {
			ratio := float64(correct) / float64(judged)
			row.CorrectRatio = &ratio
		}
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Seen != b.Seen {
			return a.Seen > b.Seen
		}
		if a.Reward != b.Reward {
			return a.Reward > b.Reward
		}
		return a.Node < b.Node
	})
	return rows
}

func sideTotals(rows []NeighbourStats) SideTotals {
	out := SideTotals{Edges: len(rows)}
	for _, row := range rows {
		out.Seen += row.Seen
		out.Reward += row.Reward
		out.PathSeen += row.PathSeen
		out.Correct += row.Correct
		out.Incorrect += row.Incorrect
	}
	if judged := out.Correct + out.Incorrect; judged > 0 {
		ratio := float64(out.Correct) / float64(judged)
		out.CorrectRatio = &ratio
	}
	return out
}

// NodeRatios describes one node against its neighbours, or nil when it is not a live node.
func (g *Graph) NodeRatios(node int) *NodeStats {
	if node < 0 || node >= len(g.Alive) || !g.Alive[node] {
		return nil
	}
	in := make([]Transition, 0, g.parents[node].size())
	for i, p := range g.parents[node].order {
		in = append(in, Transition{p, g.parents[node].edges[i]})
	}
	from, to := g.sideRows(in), g.sideRows(g.Children(node))
	return &NodeStats{
		Node: node, Label: g.TextOf(g.Label(node)),
		Visits: g.Count[node], VisitResets: g.CountResets[node],
		From: from, To: to,
		InTotals: sideTotals(from), OutTotals: sideTotals(to),
	}
}

// NodeRatioRows is NodeRatios of the most visited nodes (node >= 0: only that
// one); limit 0 returns all of them.
func (g *Graph) NodeRatioRows(limit, node int) []NodeStats {
	if node >= 0 {
		if one := g.NodeRatios(node); one != nil {
			return []NodeStats{*one}
		}
		return []NodeStats{}
	}
	order := make([]int, 0, g.NumNodes())
	for i, alive := range g.Alive {
		if alive {
			order = append(order, i)
		}
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if ca, cb := counterTotal(g.Count[a], g.CountResets, a), counterTotal(g.Count[b], g.CountResets, b); ca != cb {
			return ca > cb
		}
		return a < b
	})
	if limit > 0 && len(order) > limit {
		order = order[:limit]
	}
	out := make([]NodeStats, 0, len(order))
	for _, i := range order {
		if one := g.NodeRatios(i); one != nil {
			out = append(out, *one)
		}
	}
	return out
}
