package radixnet

import (
	"fmt"
	"math"
	"strings"
)

// Traversal is how a walk chooses its way through the graph.
//
// The model has always searched by what went *right*: an edge's weight is a
// function of how often the corpus traversed it and of the reward it collected,
// and the cheapest path is the likeliest one.  ByLeastPunished searches by what
// went *wrong* instead - it ranks a step by the blame on it and lets the
// ordinary cost decide only between steps nothing is held against.  See
// SPEC-LeastPunished.md.
type Traversal int

const (
	// ByReward is the original search: -log softmax of the dual frequency weight.
	ByReward Traversal = iota
	// ByLeastPunished ranks a step by its punishment first (EdgePunishment plus
	// what the walks through its context got wrong) and by cost only to break
	// ties, and ranks a whole path by its *worst* step.
	ByLeastPunished
)

// String is the traversal's name as the CLI and the API spell it.
func (t Traversal) String() string {
	if t == ByLeastPunished {
		return "least-punished"
	}
	return "reward"
}

// ParseTraversal reads a traversal name as a *ranking*: "", "reward" or
// "rewards" for the original search and "least-punished" (also
// "least_punished", "blame") for the one that follows the blame.
//
// "punishment" is the other punishment traversal (penalty.go), which prices a
// step rather than ordering the walks: the beams rank it exactly as they rank a
// rewarded one, so it reads as ByReward here and is never an alias of
// ByLeastPunished - the two are different currencies.
func ParseTraversal(name string) (Traversal, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "reward", "rewards", "cost", TraversalPunishment, "penalty":
		return ByReward, nil
	case "least-punished", "least_punished", "leastpunished", "blame":
		return ByLeastPunished, nil
	}
	return ByReward, fmt.Errorf("unknown traversal %q; expected 'reward', 'punishment' or 'least-punished'", name)
}

// PunishTolerance is how close two punishments have to be to count as equal:
// the same penalty applied in a different order can land a bit or two apart,
// and a walk is not "more punished" for that.
const PunishTolerance = 1e-12

// LeastPunished keeps the children the model has the least against - the whole
// of the change, in one function.  Where nothing at this node was ever punished
// (every step of an untutored graph) every child ties at zero and the slice
// comes back untouched, so the walk costs exactly what it always did; where
// something *was* punished, the steps that carry more blame than the cleanest
// one here are not options any more, however well rewarded they are.
func LeastPunished(costs []ChildCost) []ChildCost {
	if len(costs) < 2 {
		return costs
	}
	low := costs[0].Punish
	same := true
	for _, it := range costs[1:] {
		if it.Punish != costs[0].Punish {
			same = false
		}
		if it.Punish < low {
			low = it.Punish
		}
	}
	if same {
		return costs
	}
	keep := low + PunishTolerance
	out := make([]ChildCost, 0, len(costs))
	for _, it := range costs {
		if it.Punish <= keep {
			out = append(out, it)
		}
	}
	return out
}

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
	// Punish is the walk's worst step under the least-punished traversal
	// (0 everywhere nothing was ever punished).
	Punish float64 `json:"punish,omitempty"`
}

// Probability is exp(-cost) (0 for an infinite cost).
// Onward is the children a walk may actually take, given what the model has
// learned about going round.
//
// Back is not a continuation - it emits nothing and no text passes through it -
// so it never appears in a path.  But it *competes* with the real children for
// probability, and when it is the cheapest of them the model's most likely next
// step at this node is to stop rather than carry on: the walk hands over, which
// here means the branch offers nothing and the search goes on with its others
// (Graph.ObserveBack).
func Onward(costs []ChildCost) []ChildCost {
	back, hasBack := math.Inf(1), false
	for _, it := range costs {
		if it.Child == Back && (!hasBack || it.Cost < back) {
			back, hasBack = it.Cost, true
		}
	}
	if !hasBack {
		return costs
	}
	onward := make([]ChildCost, 0, len(costs))
	handOver := true
	for _, it := range costs {
		if it.Child == Back {
			continue
		}
		onward = append(onward, it)
		if it.Cost < back {
			handOver = false
		}
	}
	if handOver {
		return nil
	}
	return onward
}

func (r *PathResult) Probability() float64 {
	if math.IsInf(r.Cost, 0) || math.IsNaN(r.Cost) {
		return 0
	}
	return math.Exp(-r.Cost)
}

// startEmission is the number of units the start node emits (its remainder
// after the matched gram); START and END emit nothing.
func startEmission(g *Graph, startNode, startOffset int) (int, error) {
	if startNode < 0 || startNode >= len(g.Labels) || !g.Alive[startNode] {
		return 0, fmt.Errorf("start node %d is not alive", startNode)
	}
	if startNode == Start || startNode == End {
		return 0, nil
	}
	remainder := g.labelLen[startNode] - (startOffset + g.Enc.N)
	if startOffset < 0 || remainder < 0 {
		return 0, fmt.Errorf("start_offset %d out of range for label %q", startOffset, g.Labels[startNode])
	}
	return remainder, nil
}

// buildResult decodes a node path into a PathResult.  includeContext nil
// defaults to "true from START, false otherwise"; maxChars < 0 means no cap
// (and counts the encoding's units, so words under a word encoding).
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
	text := g.DecodePath(real, offset, ctx)
	if maxChars >= 0 {
		text = g.Enc.Truncate(text, maxChars)
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
// temperature); temperature 0 is greedy.  maxChars < 0 means no limit (in the
// encoding's units); rng nil uses the graph's own generator; costs nil reads
// the graph's own cost function, and the punishment traversal hands in its own
// (see penalty.go).
func (g *Graph) SampleWalk(startNode, startOffset, maxChars int, temperature float64, rng *MT19937, includeContext *bool, costs CostFn) (*PathResult, error) {
	return g.SampleWalkBy(startNode, startOffset, maxChars, temperature, rng, includeContext, costs, ByReward)
}

// SampleWalkBy is SampleWalk under a chosen traversal.  ByLeastPunished keeps
// only the least punished children at every node and then samples among them
// exactly as before: punishment decides *what* may be walked, the cost decides
// which of those it is.  It reads the blame off the graph itself, so a costs
// handed in with it is ignored - the two are different currencies and are not
// composed (../../SPEC-LeastPunished.md, penalty.go).
func (g *Graph) SampleWalkBy(startNode, startOffset, maxChars int, temperature float64, rng *MT19937, includeContext *bool, costs CostFn, traversal Traversal) (*PathResult, error) {
	if traversal == ByLeastPunished {
		costs = nil
	}
	if temperature < 0 {
		return nil, fmt.Errorf("temperature must be >= 0")
	}
	if rng == nil {
		rng = g.rng
	}
	g.Prepare()
	childCosts := g.costsOrDefault(costs)
	chars, err := startEmission(g, startNode, startOffset)
	if err != nil {
		return nil, err
	}
	node := startNode
	nodeIDs := []int{node}
	stepCosts := []float64{}
	punish := 0.0
	steps := 0
	cameFrom := -1
	if startNode == Start {
		cameFrom = Start // every walk from the sentinel starts in the same context
	}
	for {
		if node == End || (maxChars >= 0 && chars >= maxChars) {
			break
		}
		// a node the model expects to go round offers nothing
		options := Onward(childCosts(node, cameFrom))
		if traversal == ByLeastPunished {
			options = LeastPunished(options)
		}
		if len(options) == 0 {
			break
		}
		var pick ChildCost
		if temperature == 0 || len(options) == 1 {
			pick = options[0]
			for _, item := range options[1:] {
				if item.Cost < pick.Cost {
					pick = item
				}
			}
		} else {
			invT := 1.0 / temperature
			lowest := options[0].Cost
			for _, item := range options[1:] {
				if item.Cost < lowest {
					lowest = item.Cost
				}
			}
			weights := make([]float64, len(options))
			for i, item := range options {
				weights[i] = math.Exp(-(item.Cost - lowest) * invT)
			}
			r := rng.Float64() * fsum(weights)
			pick = options[len(options)-1]
			acc := 0.0
			for i, item := range options {
				acc += weights[i]
				if r < acc {
					pick = item
					break
				}
			}
		}
		stepCosts = append(stepCosts, pick.Cost)
		if pick.Punish > punish {
			punish = pick.Punish // a walk is as punished as its worst step
		}
		nodeIDs = append(nodeIDs, pick.Child)
		if pick.Child != End {
			chars += g.labelLen[pick.Child] - g.Enc.Overlap()
		}
		cameFrom = node
		node = pick.Child
		steps++
	}
	walk := buildResult(g, nodeIDs, stepCosts, startOffset, maxChars, steps, includeContext)
	walk.Punish = punish
	return walk, nil
}
