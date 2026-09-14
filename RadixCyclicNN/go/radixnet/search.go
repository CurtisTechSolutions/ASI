package radixnet

import (
	"fmt"
	"math"
)

// PathResult is the outcome of a prediction / generation walk: Text is the
// emitted continuation, Labels / NodeIDs the path (sentinels included), Cost
// the summed edge cost, Expanded the search effort, ReachedEnd whether the
// path ends at END and FullText prefix + text.
type PathResult struct {
	Text       string    `json:"text"`
	Labels     []string  `json:"labels"`
	NodeIDs    []int     `json:"node_ids"`
	Cost       float64   `json:"cost"`
	StepCosts  []float64 `json:"step_costs"`
	Expanded   int       `json:"expanded"`
	ReachedEnd bool      `json:"reached_end"`
	FullText   string    `json:"full_text"`
}

// Probability is exp(-cost) (0 for an infinite cost).
func (r *PathResult) Probability() float64 {
	if math.IsInf(r.Cost, 0) || math.IsNaN(r.Cost) {
		return 0
	}
	return math.Exp(-r.Cost)
}

// startEmission is the number of characters the start node emits (its
// remainder after the matched trigram); START and END emit nothing.
func startEmission(g *Graph, startNode, startOffset int) (int, error) {
	if startNode < 0 || startNode >= len(g.Labels) || !g.Alive[startNode] {
		return 0, fmt.Errorf("start node %d is not alive", startNode)
	}
	if startNode == Start || startNode == End {
		return 0, nil
	}
	remainder := g.labelLen[startNode] - (startOffset + Window)
	if startOffset < 0 || remainder < 0 {
		return 0, fmt.Errorf("start_offset %d out of range for label %q", startOffset, g.Labels[startNode])
	}
	return remainder, nil
}

// buildResult decodes a node path into a PathResult.  includeContext nil
// defaults to "true from START, false otherwise"; maxChars < 0 means no cap.
func buildResult(g *Graph, nodeIDs []int, stepCosts []float64, startOffset, maxChars, expanded int, includeContext *bool) *PathResult {
	labels := make([]string, len(nodeIDs))
	for i, n := range nodeIDs {
		labels[i] = g.Labels[n]
	}
	startNode := nodeIDs[0]
	ctx := startNode == Start
	if includeContext != nil {
		ctx = *includeContext
	}
	offset := startOffset
	if startNode == Start || startNode == End {
		offset = 0
	}
	real := make([]string, 0, len(labels))
	for i, n := range nodeIDs {
		if n != Start && n != End {
			real = append(real, labels[i])
		}
	}
	text := DecodePath(real, offset, ctx)
	if maxChars >= 0 {
		text = truncateRunes(text, maxChars)
	}
	if stepCosts == nil {
		stepCosts = []float64{}
	}
	return &PathResult{
		Text: text, Labels: labels, NodeIDs: nodeIDs, Cost: fsum(stepCosts), StepCosts: stepCosts,
		Expanded: expanded, ReachedEnd: nodeIDs[len(nodeIDs)-1] == End,
	}
}

// SampleWalk is a stochastic walk sampling each child from softmax(-cost /
// temperature); temperature 0 is greedy.  maxChars < 0 means no limit; rng nil
// uses the graph's own generator.
func (g *Graph) SampleWalk(startNode, startOffset, maxChars int, temperature float64, rng *MT19937, includeContext *bool) (*PathResult, error) {
	if temperature < 0 {
		return nil, fmt.Errorf("temperature must be >= 0")
	}
	if rng == nil {
		rng = g.rng
	}
	g.Prepare()
	chars, err := startEmission(g, startNode, startOffset)
	if err != nil {
		return nil, err
	}
	node := startNode
	nodeIDs := []int{node}
	stepCosts := []float64{}
	steps := 0
	for {
		if node == End || (maxChars >= 0 && chars >= maxChars) {
			break
		}
		costs := g.ChildCosts(node)
		if len(costs) == 0 {
			break
		}
		var pick ChildCost
		if temperature == 0 || len(costs) == 1 {
			pick = costs[0]
			for _, item := range costs[1:] {
				if item.Cost < pick.Cost {
					pick = item
				}
			}
		} else {
			invT := 1.0 / temperature
			lowest := costs[0].Cost
			for _, item := range costs[1:] {
				if item.Cost < lowest {
					lowest = item.Cost
				}
			}
			weights := make([]float64, len(costs))
			for i, item := range costs {
				weights[i] = math.Exp(-(item.Cost - lowest) * invT)
			}
			r := rng.Float64() * fsum(weights)
			pick = costs[len(costs)-1]
			acc := 0.0
			for i, item := range costs {
				acc += weights[i]
				if r < acc {
					pick = item
					break
				}
			}
		}
		stepCosts = append(stepCosts, pick.Cost)
		nodeIDs = append(nodeIDs, pick.Child)
		if pick.Child != End {
			chars += g.labelLen[pick.Child] - Overlap
		}
		node = pick.Child
		steps++
	}
	return buildResult(g, nodeIDs, stepCosts, startOffset, maxChars, steps, includeContext), nil
}
