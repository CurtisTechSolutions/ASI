package radixnet

import (
	"fmt"
	"math"
	"strings"
)

// The punishment traversal: a walk priced by what the network was punished
// for, not by what it was rewarded for.
//
// Every search here - the two beams and the sampler - reads the graph through
// one funnel: []ChildCost for a node, with cost = -log P(child | parent).
// Which cost function fills that list is the *traversal*, and there are two of
// them:
//
//	"reward"      what the model believes.  The edge weight carries
//	              RewardScale * reward, so a path the tutor rewarded is cheap
//	              and the search follows the rewards.  The default, and what
//	              every release before this option did.
//	"punishment"  what the model was punished for.  The rewards leave the score
//	              altogether and only the penalties price the step, so the
//	              cheapest path is the one that accumulated the *least
//	              punishment*.
//
// The traversal rests on one split, ChildEvidence: merit is what speaks *for* a
// step with every reward taken out of it - frequency and structure, what the
// corpus did rather than what a judge said about it - and penalty >= 0 is what
// speaks against it.  The step's score is meritScale*merit -
// penaltyScale*penalty, and the cost the usual -log softmax over the parent's
// children, so costs stay non-negative, exp(-cost) is still a path's
// probability and the numbers stay comparable with the reward traversal's.
// meritScale = 0 is the pure form: nothing but the punishment decides.
//
// The count / reward model splits EdgeReward in half - the negative half is the
// punishment, the positive half is what this traversal drops - and a judged
// path context splits the same way.  The negative network is nothing but
// punishment, so its currency is net blame against the cleared text that ran
// through the same edge.  This mirrors radixnet/penalty.py exactly.

// The traversals a search can run.  The first two are cost functions and live
// here; the third is a different kind of thing - a ranking, which orders a walk
// by the punishment on its worst step before its cost, and which only the count
// model can run because only it keeps the judged paths that ranking reads
// (search.go, ../../SPEC-LeastPunished.md).  It is named here so one flag
// offers all three; the cost functions below simply do not apply to it.
const (
	TraversalReward        = "reward"
	TraversalPunishment    = "punishment"
	TraversalLeastPunished = "least-punished"
	// DefaultTraversal is what every search runs unless told otherwise.
	DefaultTraversal = TraversalReward
)

// Traversals lists the traversal names, in the order the CLI and the API offer them.
var Traversals = []string{TraversalReward, TraversalPunishment, TraversalLeastPunished}

// ResolveTraversal normalises a traversal name ("" = the default) and rejects anything else.
func ResolveTraversal(name string) (string, error) {
	got := strings.ToLower(strings.TrimSpace(name))
	if got == "" {
		return DefaultTraversal, nil
	}
	for _, known := range Traversals {
		if got == known {
			return got, nil
		}
	}
	return "", fmt.Errorf("unknown traversal %q; expected 'reward', 'punishment' or 'least-punished'", name)
}

// CostFn is what a search reads the graph through: (parent, prev) -> the
// parent's out-edges with their costs.  Graph.ChildCostsFrom is the default
// one; the punishment traversal hands in PenaltyCosts.Costs instead.
type CostFn func(p, prev int) []ChildCost

// EdgeEvidence splits one out-edge's evidence in two: what speaks for the step
// (rewards removed) and what speaks against it.
type EdgeEvidence struct {
	Child, Edge    int
	Merit, Penalty float64
}

// ChildEvidence is the merit / penalty split of p's out-edges, as a walk that
// arrived from prev sees them (prev < 0: no context).
//
// On the count / reward model the weight is "frequency terms + RewardScale *
// reward", so the two halves come apart exactly: the merit is the weight with
// the whole reward subtracted back out and the penalty is RewardScale *
// max(0, -reward).  A judged context adds the part of its term that says the
// step was right here to the merit and the part that says it was wrong here to
// the penalty.  On the negative network nothing was ever rewarded: the penalty
// is log(1 + net blame) and the merit log(1 + cleared text), both on a log
// scale so a thousandfold blame does not make one child infinitely dearer than
// the rest.
func (g *Graph) ChildEvidence(p, prev int) []EdgeEvidence {
	adj := &g.children[p]
	out := make([]EdgeEvidence, len(adj.order))
	if g.Neg != nil {
		for i, c := range adj.order {
			e := adj.edges[i]
			out[i] = EdgeEvidence{c, e, math.Log1p(math.Max(0, g.Neg.Clear[e])), math.Log1p(g.Evidence(e))}
		}
		return out
	}
	context := prev >= 0 && g.PathScale != 0 && len(g.paths) > 0
	for i, c := range adj.order {
		e := adj.edges[i]
		reward := g.EdgeReward[e]
		merit := g.EdgeW[e] - g.RewardScale*reward
		penalty := g.RewardScale * math.Max(0, -reward)
		if context {
			if term := g.PathScale * g.PathTerm(prev, e); term > 0 {
				merit += term
			} else {
				penalty += -term
			}
		}
		out[i] = EdgeEvidence{c, e, merit, penalty}
	}
	return out
}

// PenaltyCosts is the punishment traversal's cost function, priced once for the
// whole graph so the two beams can read it side by side.
//
// Build it with Graph.PenaltyCosts while the graph is still single-threaded;
// afterwards Costs only reads the tables, exactly as ChildCostsFrom does.
type PenaltyCosts struct {
	g        *Graph
	edgeCost []float64
	ctx      map[PathKey][]ChildCost
}

// PenaltyCosts prices every edge by the punishment it carries.  penaltyScale
// weighs the punishments, meritScale what the corpus did (0 = nothing but the
// punishments decides); both must be >= 0.
func (g *Graph) PenaltyCosts(penaltyScale, meritScale float64) (*PenaltyCosts, error) {
	if penaltyScale < 0 {
		return nil, fmt.Errorf("penalty_scale must be >= 0, got %v", penaltyScale)
	}
	if meritScale < 0 {
		return nil, fmt.Errorf("merit_scale must be >= 0, got %v", meritScale)
	}
	g.Prepare()
	pc := &PenaltyCosts{g: g, edgeCost: make([]float64, len(g.EdgeW))}
	score := func(ev EdgeEvidence) float64 { return meritScale*ev.Merit - penaltyScale*ev.Penalty }
	for p := range g.Labels {
		if !g.Alive[p] || g.children[p].size() == 0 {
			continue
		}
		for _, cc := range softmaxCosts(g.ChildEvidence(p, -1), score) {
			pc.edgeCost[cc.Edge] = cc.Cost
		}
	}
	// the judged contexts, priced once each, exactly as ensureContextCosts does for the reward traversal
	if g.PathScale != 0 && len(g.paths) > 0 {
		pairs := map[PathKey]bool{}
		for key := range g.paths {
			if key.Edge >= 0 && key.Edge < len(g.EdgeParent) {
				pairs[PathKey{key.Prev, g.EdgeParent[key.Edge]}] = true
			}
		}
		pc.ctx = make(map[PathKey][]ChildCost, len(pairs))
		for pair := range pairs {
			p := pair.Edge // the parent node, as stored above
			if p < 0 || p >= len(g.children) || !g.Alive[p] || g.children[p].size() == 0 {
				continue
			}
			pc.ctx[pair] = softmaxCosts(g.ChildEvidence(p, pair.Prev), score)
		}
	}
	return pc, nil
}

// softmaxCosts turns scored out-edges into -log softmax costs, numerically stable.
func softmaxCosts(evidence []EdgeEvidence, score func(EdgeEvidence) float64) []ChildCost {
	if len(evidence) == 0 {
		return nil
	}
	scores := make([]float64, len(evidence))
	m := math.Inf(-1)
	for i, ev := range evidence {
		scores[i] = score(ev)
		if scores[i] > m {
			m = scores[i]
		}
	}
	terms := make([]float64, len(scores))
	for i, s := range scores {
		terms[i] = math.Exp(s - m)
	}
	lse := m + math.Log(fsum(terms))
	out := make([]ChildCost, len(evidence))
	for i, ev := range evidence {
		// Punish stays zero here: it is the least-punished traversal's currency
		// (search.go), and this traversal has priced the blame into Cost instead
		out[i] = ChildCost{Child: ev.Child, Edge: ev.Edge, Cost: lse - scores[i]}
	}
	return out
}

// Costs lists p's out-edges as a walk that arrived from prev sees them under
// the punishment traversal.  Safe to call from several goroutines.
func (pc *PenaltyCosts) Costs(p, prev int) []ChildCost {
	if prev >= 0 && pc.ctx != nil {
		if costs, ok := pc.ctx[PathKey{prev, p}]; ok {
			return costs
		}
	}
	adj := &pc.g.children[p]
	out := make([]ChildCost, len(adj.order))
	for i, c := range adj.order {
		e := adj.edges[i]
		out[i] = ChildCost{Child: c, Edge: e, Cost: pc.edgeCost[e]}
	}
	return out
}

// TraversalCosts is the cost function a search should read the graph through,
// or nil for the model's own - which is what the reward traversal returns, so
// the searches then call ChildCostsFrom and nothing about them changes.
func (g *Graph) TraversalCosts(traversal string, penaltyScale, meritScale float64) (CostFn, error) {
	name, err := ResolveTraversal(traversal)
	if err != nil {
		return nil, err
	}
	if name == TraversalReward || name == TraversalLeastPunished {
		// the least-punished traversal reads the blame itself, not through a cost function
		return nil, nil
	}
	pc, err := g.PenaltyCosts(penaltyScale, meritScale)
	if err != nil {
		return nil, err
	}
	return pc.Costs, nil
}

// costsOrDefault is what every search calls: the given cost function, or the graph's own.
func (g *Graph) costsOrDefault(costs CostFn) CostFn {
	if costs == nil {
		return g.ChildCostsFrom
	}
	return costs
}
