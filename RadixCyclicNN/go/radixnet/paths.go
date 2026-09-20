package radixnet

import (
	"math"
	"sort"
)

// Paths: what a *walk* did, not what an edge did.
//
// An edge is the right move in one sentence and the wrong one in another, so
// counting rewards per edge blurs the two together.  A path context is the
// step in the company it kept - the node that called the edge's parent, and
// the edge it then took - and it carries three numbers: how often the step was
// seen at all, how often it was part of an output judged correct, and how
// often it was part of one judged wrong.  Those counters then price the step:
// the same edge is cheap for the walk that was right here and dear for the one
// that was wrong.
//
// A context is born when a path is judged and is kept up to date by every
// later traversal; an unjudged pass never creates one, so training a corpus
// cannot fill the table with the second-order counts of a whole language.

// PathKey is one step in its context: the node before the edge's parent, and the edge.
type PathKey struct {
	Prev int
	Edge int
}

// PathRow counts what a context did.
type PathRow struct {
	Seen      int64
	Correct   int64
	Incorrect int64
}

// PathStats is one context with its ratios, as the API and the CLI report it.
type PathStats struct {
	Prev         int      `json:"prev"`
	Edge         int      `json:"edge"`
	Seen         int64    `json:"seen"`
	Correct      int64    `json:"correct"`
	Incorrect    int64    `json:"incorrect"`
	CorrectRatio *float64 `json:"correct_ratio"`
	SeenRatio    *float64 `json:"seen_ratio"`
	Term         float64  `json:"term"`
}

// PathTotals is how much of the graph has been judged as paths rather than as edges.
type PathTotals struct {
	Contexts  int   `json:"contexts"`
	Judged    int   `json:"judged"`
	Seen      int64 `json:"seen"`
	Correct   int64 `json:"correct"`
	Incorrect int64 `json:"incorrect"`
}

// RecordPath counts one walk of a whole text, step by step, in its own
// context.  transitions are the (parent, edge) steps of a single text in
// order; the context of step k is the parent of step k-1 (START for the first
// step, which has none).  outcome says what the walk was judged to be -
// PathCorrect, PathIncorrect or PathUnjudged - and create (default: whenever
// the walk was judged) decides whether contexts never seen before are added.
func (g *Graph) RecordPath(transitions []Transition, outcome PathOutcome, create bool) int {
	if len(transitions) == 0 || (!create && len(g.paths) == 0) {
		return 0
	}
	touched := 0
	prev := Start
	for i, t := range transitions {
		if i > 0 {
			prev = transitions[i-1].P
		}
		if t.E < 0 || t.E >= len(g.EdgeAlive) || !g.EdgeAlive[t.E] {
			continue
		}
		row := g.pathRow(prev, t.E, create)
		if row == nil {
			continue
		}
		row.Seen++
		switch outcome {
		case PathCorrect:
			row.Correct++
		case PathIncorrect:
			row.Incorrect++
		}
		touched++
	}
	if touched > 0 && outcome != PathUnjudged {
		g.Version.Add(1) // the contexts that moved make their node's costs stale
	}
	return touched
}

// PathOutcome is what a walk was judged to be.
type PathOutcome int

// The three verdicts a walk can carry.
const (
	PathUnjudged PathOutcome = iota
	PathCorrect
	PathIncorrect
)

// MarkSteps judges single steps - what a diff blames or teaches - rather than a whole walk.
func (g *Graph) MarkSteps(steps []PathKey, correct bool) int {
	marked := 0
	for _, step := range steps {
		if step.Edge < 0 || step.Edge >= len(g.EdgeAlive) || !g.EdgeAlive[step.Edge] {
			continue
		}
		row := g.pathRow(step.Prev, step.Edge, true)
		row.Seen++
		if correct {
			row.Correct++
		} else {
			row.Incorrect++
		}
		marked++
	}
	if marked > 0 {
		g.Version.Add(1)
	}
	return marked
}

// pathRow is one context's counters, created on demand (and indexed both ways so a split can find them).
func (g *Graph) pathRow(prev, edge int, create bool) *PathRow {
	key := PathKey{prev, edge}
	if row, ok := g.paths[key]; ok {
		return row
	}
	if !create {
		return nil
	}
	if g.paths == nil {
		g.paths = map[PathKey]*PathRow{}
		g.pathsByEdge = map[int]map[int]bool{}
		g.pathsByPrev = map[int]map[int]bool{}
	}
	row := &PathRow{}
	g.paths[key] = row
	addToIndex(g.pathsByEdge, edge, prev)
	addToIndex(g.pathsByPrev, prev, edge)
	g.pathParents = nil
	return row
}

func (g *Graph) dropPath(prev, edge int) *PathRow {
	key := PathKey{prev, edge}
	row, ok := g.paths[key]
	if !ok {
		return nil
	}
	delete(g.paths, key)
	dropFromIndex(g.pathsByEdge, edge, prev)
	dropFromIndex(g.pathsByPrev, prev, edge)
	g.pathParents = nil
	return row
}

// addPath adds one context's counters into another (a split or a merge moved the step).
func (g *Graph) addPath(prev, edge int, row *PathRow) {
	into := g.pathRow(prev, edge, true)
	into.Seen += row.Seen
	into.Correct += row.Correct
	into.Incorrect += row.Incorrect
}

func addToIndex(index map[int]map[int]bool, key, value int) {
	holder, ok := index[key]
	if !ok {
		holder = map[int]bool{}
		index[key] = holder
	}
	holder[value] = true
}

func dropFromIndex(index map[int]map[int]bool, key, value int) {
	holder, ok := index[key]
	if !ok {
		return
	}
	delete(holder, value)
	if len(holder) == 0 {
		delete(index, key)
	}
}

// PathTerm is log((correct + s) / (incorrect + s)) of a context: zero until a
// path is judged, and symmetric.
func (g *Graph) PathTerm(prev, edge int) float64 {
	row, ok := g.paths[PathKey{prev, edge}]
	if !ok || (row.Correct == 0 && row.Incorrect == 0) {
		return 0
	}
	return math.Log((float64(row.Correct) + Smoothing) / (float64(row.Incorrect) + Smoothing))
}

// PathIncorrect is how often a walk that came from prev was judged wrong here:
// the one number the least-punished traversal steers by (search.go), counted
// against nothing.
func (g *Graph) PathIncorrect(prev, edge int) int64 {
	if row, ok := g.paths[PathKey{prev, edge}]; ok {
		return row.Incorrect
	}
	return 0
}

// PathStatsOf describes one context, or nil when it has never been walked.
func (g *Graph) PathStatsOf(prev, edge int) *PathStats {
	row, ok := g.paths[PathKey{prev, edge}]
	if !ok {
		return nil
	}
	out := &PathStats{Prev: prev, Edge: edge, Seen: row.Seen, Correct: row.Correct, Incorrect: row.Incorrect,
		Term: g.PathTerm(prev, edge)}
	if judged := row.Correct + row.Incorrect; judged > 0 {
		ratio := float64(row.Correct) / float64(judged)
		out.CorrectRatio = &ratio
	}
	if edge >= 0 && edge < len(g.EdgeCount) && g.EdgeCount[edge] > 0 {
		ratio := float64(row.Seen) / float64(g.EdgeCount[edge])
		out.SeenRatio = &ratio
	}
	return out
}

// PathTotals counts the whole table.
func (g *Graph) PathTotals() PathTotals {
	out := PathTotals{Contexts: len(g.paths)}
	for _, row := range g.paths {
		out.Seen += row.Seen
		out.Correct += row.Correct
		out.Incorrect += row.Incorrect
		if row.Correct != 0 || row.Incorrect != 0 {
			out.Judged++
		}
	}
	return out
}

// PathContexts lists every context (node >= 0: only the steps leaving that
// node), most judged first; limit 0 returns all of them.
func (g *Graph) PathContexts(limit, node int) []PathStats {
	out := []PathStats{}
	for key := range g.paths {
		if node >= 0 && (key.Edge >= len(g.EdgeParent) || g.EdgeParent[key.Edge] != node) {
			continue
		}
		if stats := g.PathStatsOf(key.Prev, key.Edge); stats != nil {
			out = append(out, *stats)
		}
	}
	sortPathStats(out)
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func sortPathStats(rows []PathStats) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		aj, bj := a.Correct+a.Incorrect, b.Correct+b.Incorrect
		if aj != bj {
			return aj > bj
		}
		if a.Seen != b.Seen {
			return a.Seen > b.Seen
		}
		if a.Prev != b.Prev {
			return a.Prev < b.Prev
		}
		return a.Edge < b.Edge
	})
}

// NodesWithPaths are the nodes whose out-edges carry a context: everywhere
// else a step costs exactly what its edge costs.
func (g *Graph) NodesWithPaths() map[int]bool {
	if g.pathParents == nil {
		parents := map[int]bool{}
		for edge := range g.pathsByEdge {
			if edge >= 0 && edge < len(g.EdgeParent) {
				parents[g.EdgeParent[edge]] = true
			}
		}
		g.pathParents = parents
	}
	return g.pathParents
}

// splitPaths follows the contexts through a split: q -> P -> c becomes q -> A
// -> B -> c, so a context (q, e) of a moved edge becomes (A, e) and the new
// edge A -> B inherits (q, A->B), the step q now calls.
func (g *Graph) splitPaths(a, b int, moved []int, bridge int) {
	if len(g.paths) == 0 || len(moved) == 0 {
		return
	}
	for _, edge := range moved {
		for prev := range copyKeys(g.pathsByEdge[edge]) {
			row := g.dropPath(prev, edge)
			if row == nil {
				continue
			}
			g.addPath(a, edge, row)
			if bridge >= 0 {
				g.addPath(prev, bridge, row)
			}
		}
	}
	g.ctxVersion = invalidStamp
}

// mergePaths follows the contexts through a merge: the chain was unary, so
// what it knew about was never a choice.  The edge p -> c dies with its
// contexts, and so do the contexts of c's out-edges ("having come to c from
// p"); contexts that arrive through c are re-keyed to p.
func (g *Graph) mergePaths(p, child, dying int, moved []int) {
	if len(g.paths) == 0 {
		return
	}
	for prev := range copyKeys(g.pathsByEdge[dying]) {
		g.dropPath(prev, dying)
	}
	for _, edge := range moved {
		g.dropPath(p, edge)
	}
	for edge := range copyKeys(g.pathsByPrev[child]) {
		if row := g.dropPath(child, edge); row != nil {
			g.addPath(p, edge, row)
		}
	}
	g.ctxVersion = invalidStamp
}

func copyKeys(set map[int]bool) map[int]bool {
	out := make(map[int]bool, len(set))
	for k := range set {
		out[k] = true
	}
	return out
}

// Label is a node's label ("" when the node is not alive).
func (g *Graph) Label(node int) string {
	if node < 0 || node >= len(g.Labels) {
		return ""
	}
	return g.Labels[node]
}

// NumNodeIDs is how many node ids exist (alive or tombstoned).
func (g *Graph) NumNodeIDs() int { return len(g.Labels) }

// ParentOfEdge is the node an edge leaves, or -1.
func (g *Graph) ParentOfEdge(edge int) int {
	if edge < 0 || edge >= len(g.EdgeParent) {
		return -1
	}
	return g.EdgeParent[edge]
}
