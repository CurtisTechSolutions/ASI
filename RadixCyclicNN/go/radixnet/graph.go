package radixnet

import (
	"fmt"
	"sync"
)

// Node ids of the two sentinels.
const (
	Start = 0
	End   = 1
)

// New edge weights of the sine network are drawn from [WLow, WHigh]; the count
// model discards the value but consumes the same random number, so the RNG
// state stays in step with the Python implementation.
const (
	WLow  = 0.5
	WHigh = 1.5
)

// Default sine parameters written to model files (activations are the
// constant 1 in the count model: a = 0, k = 1).
const (
	defaultB = 1.0 / 3.0
	defaultH = 0.0
)

const (
	graphFormat        = "radixnet-graph"
	graphFormatVersion = 1
)

type loc struct{ node, off int }

// adjacency holds a node's edges keyed by the node at the other end - child
// -> edge id for the out-edges, parent -> edge id for the in-edges - in
// insertion order (Python dict order, which decides the summation order of
// the softmax and therefore keeps the numbers identical to the Python model).
// Two parallel slices searched linearly cover the small degrees of almost
// every node in a few dozen bytes; a node past adjacencyIndexAt entries (the
// sentinels, hubs like " th") gets a map index too.  Go maps cost a few
// hundred bytes each, which for millions of nodes was most of the graph.
type adjacency struct {
	order []int       // node ids in insertion order
	edges []int       // edge ids, parallel to order
	idx   map[int]int // node id -> position, only past adjacencyIndexAt entries
}

// adjacencyIndexAt is the degree above which an adjacency keeps a map index.
const adjacencyIndexAt = 16

// find returns the position of node c, or -1.
func (a *adjacency) find(c int) int {
	if a.idx != nil {
		if i, ok := a.idx[c]; ok {
			return i
		}
		return -1
	}
	for i, x := range a.order {
		if x == c {
			return i
		}
	}
	return -1
}

func (a *adjacency) get(c int) (int, bool) {
	if i := a.find(c); i >= 0 {
		return a.edges[i], true
	}
	return 0, false
}

// set records edge e to / from node c, appending c when it is new.
func (a *adjacency) set(c, e int) {
	if i := a.find(c); i >= 0 {
		a.edges[i] = e
		return
	}
	a.order = append(a.order, c)
	a.edges = append(a.edges, e)
	if a.idx != nil {
		a.idx[c] = len(a.order) - 1
	} else if len(a.order) > adjacencyIndexAt {
		a.idx = make(map[int]int, 2*len(a.order))
		for i, x := range a.order {
			a.idx[x] = i
		}
	}
}

// unset drops node c; the last entry takes its slot, so the order of the
// rest changes - fine for the parents, which have no order to keep (the
// children are only ever cleared wholesale).
func (a *adjacency) unset(c int) bool {
	i := a.find(c)
	if i < 0 {
		return false
	}
	last := len(a.order) - 1
	if i != last {
		a.order[i], a.edges[i] = a.order[last], a.edges[last]
		if a.idx != nil {
			a.idx[a.order[i]] = i
		}
	}
	a.order = a.order[:last]
	a.edges = a.edges[:last]
	if a.idx != nil {
		delete(a.idx, c)
	}
	return true
}

func (a *adjacency) clear() {
	a.order = a.order[:0]
	a.edges = a.edges[:0]
	a.idx = nil
}

func (a *adjacency) size() int { return len(a.order) }

// Transition is one traversed edge: the parent it leaves and the edge id.
type Transition struct{ P, E int }

// Graph is the self-compressing cyclic trigram graph of the count / reward
// model.  Storage is flat slices indexed by node id / edge id; removed nodes
// and edges are tombstoned (ids are never reused) and compacted by ToDoc.
type Graph struct {
	Seed int64
	rng  *MT19937

	Labels   []string
	labelLen []int // characters per label
	Count    []int64
	Alive    []bool
	children []adjacency
	parents  []adjacency // in-edges: parent -> edge id (any order)

	EdgeW      []float64
	EdgeCount  []int64
	EdgeAlive  []bool
	EdgeParent []int

	// Neg is the negative network's evidence (nil on a count / reward graph).
	Neg *NegativeData

	// the count / reward numbers
	EdgeReward      []float64
	WindowEdgeCount []int64
	window          []int
	windowHead      int
	WindowSize      int
	TotalTraversals int64
	CountScale      float64
	RewardScale     float64
	GlobalScale     float64
	WindowScale     float64

	index    map[string]loc
	Inverted bool

	Version          int
	StructureVersion int
	nAliveNodes      int
	nAliveEdges      int

	// lazy weights and costs
	dirty            map[int]struct{}
	dirtyAll         bool
	weightsStructure int
	edgeCost         []float64
	costsVersion     int

	// mu guards structural changes: splits / merges / new nodes and edges take
	// the write lock, concurrent traces the read lock.
	mu      sync.RWMutex
	Workers int
}

// GraphOptions are the dual frequency function's scales and the window size.
type GraphOptions struct {
	CountScale  float64
	RewardScale float64
	GlobalScale float64
	WindowScale float64
	Window      int
}

// DefaultGraphOptions mirror the Python defaults (geometric mean of the two shares).
func DefaultGraphOptions() GraphOptions {
	return GraphOptions{CountScale: 0, RewardScale: 1, GlobalScale: 0.5, WindowScale: 0.5, Window: 10_000}
}

// NewGraph creates an empty graph with the two sentinels.
func NewGraph(seed int64, opts GraphOptions) (*Graph, error) {
	if opts.Window < 1 {
		return nil, fmt.Errorf("window must be >= 1, got %d", opts.Window)
	}
	g := &Graph{
		Seed:             seed,
		rng:              NewMT19937(seed),
		index:            make(map[string]loc),
		dirty:            make(map[int]struct{}),
		weightsStructure: -1,
		costsVersion:     -1,
		WindowSize:       opts.Window,
		CountScale:       opts.CountScale,
		RewardScale:      opts.RewardScale,
		GlobalScale:      opts.GlobalScale,
		WindowScale:      opts.WindowScale,
		Workers:          Workers,
	}
	g.newNode(StartLabel, 0)
	g.newNode(EndLabel, 0)
	return g, nil
}

// -- construction --------------------------------------------------------------

func (g *Graph) newNode(label string, count int64) int {
	nid := len(g.Labels)
	g.Labels = append(g.Labels, label)
	g.labelLen = append(g.labelLen, runeLen(label))
	g.Count = append(g.Count, count)
	g.Alive = append(g.Alive, true)
	g.children = append(g.children, adjacency{})
	g.parents = append(g.parents, adjacency{})
	g.nAliveNodes++
	g.Version++
	g.StructureVersion++
	return nid
}

func (g *Graph) newEdge(p, c int, count int64) int {
	// the sine network draws a weight here; the count model recomputes the
	// weight but consumes the same random number
	g.rng.Uniform(WLow, WHigh)
	e := len(g.EdgeW)
	g.EdgeW = append(g.EdgeW, 0.0)
	g.EdgeCount = append(g.EdgeCount, count)
	g.EdgeAlive = append(g.EdgeAlive, true)
	g.EdgeParent = append(g.EdgeParent, p)
	g.EdgeReward = append(g.EdgeReward, 0.0)
	g.WindowEdgeCount = append(g.WindowEdgeCount, 0)
	if g.Neg != nil {
		g.Neg.appendEdge()
	}
	g.children[p].set(c, e)
	g.parents[c].set(p, e)
	g.nAliveEdges++
	g.Version++
	g.StructureVersion++
	g.dirty[p] = struct{}{}
	return e
}

func (g *Graph) createTrigramNode(trigram string) int {
	if runeLen(trigram) != Window {
		panic(fmt.Sprintf("expected a %d-character trigram, got %q", Window, trigram))
	}
	nid := g.newNode(trigram, 0)
	g.index[trigram] = loc{nid, 0}
	return nid
}

// -- sizes and lookup ------------------------------------------------------------

// NumNodes is the number of alive nodes including the sentinels.
func (g *Graph) NumNodes() int { return g.nAliveNodes }

// NumEdges is the number of alive edges.
func (g *Graph) NumEdges() int { return g.nAliveEdges }

// NumTrigrams is the number of distinct trigrams stored.
func (g *Graph) NumTrigrams() int { return len(g.index) }

// CompressionRatio is trigrams per real node.
func (g *Graph) CompressionRatio() float64 {
	real := g.nAliveNodes - 2
	if real < 1 {
		real = 1
	}
	return float64(len(g.index)) / float64(real)
}

// AliveNodes lists alive node ids in increasing order.
func (g *Graph) AliveNodes() []int {
	out := make([]int, 0, g.nAliveNodes)
	for i, ok := range g.Alive {
		if ok {
			out = append(out, i)
		}
	}
	return out
}

// Lookup finds the node and offset holding a trigram.
func (g *Graph) Lookup(trigram string) (node, off int, ok bool) {
	l, found := g.index[trigram]
	return l.node, l.off, found
}

// LabelLen is the character length of a node's label.
func (g *Graph) LabelLen(node int) int { return g.labelLen[node] }

// Children lists (child, edge) pairs of p in insertion order.
func (g *Graph) Children(p int) []Transition {
	adj := &g.children[p]
	out := make([]Transition, len(adj.order))
	for i, c := range adj.order {
		out[i] = Transition{c, adj.edges[i]}
	}
	return out
}

// Edge returns the id of edge p -> c.
func (g *Graph) Edge(p, c int) (int, bool) { return g.children[p].get(c) }

// Degree is the number of out-edges of p.
func (g *Graph) Degree(p int) int { return g.children[p].size() }

// Parents lists the parent ids of c (any order).
func (g *Graph) Parents(c int) []int {
	return append([]int(nil), g.parents[c].order...)
}

// -- structural operations ----------------------------------------------------------

// Split cuts a node between its trigrams i-1 and i: A keeps the id with
// label[:i+2], B is a new node with label[i:] that inherits A's out-edges
// (edge ids kept) and count; A gets the single new edge A -> B.
func (g *Graph) Split(node, i int) (int, int, error) {
	if node == Start || node == End {
		return 0, 0, fmt.Errorf("cannot split a sentinel node")
	}
	if node < 0 || node >= len(g.Labels) || !g.Alive[node] {
		return 0, 0, fmt.Errorf("node %d is not alive", node)
	}
	label := []rune(g.Labels[node])
	length := len(label)
	if i < 1 || i > length-Window {
		return 0, 0, fmt.Errorf("split index %d out of range 1..%d for label %q", i, length-Window, string(label))
	}
	a := node
	b := g.newNode(string(label[i:]), g.Count[a])
	chA := &g.children[a]
	chB := &g.children[b]
	for i, c := range chA.order {
		e := chA.edges[i]
		chB.set(c, e)
		pc := &g.parents[c]
		pc.unset(a)
		pc.set(b, e)
		g.EdgeParent[e] = b
	}
	chA.clear()
	g.newEdge(a, b, g.Count[a])
	for j := i; j < length-Overlap; j++ {
		g.index[string(label[j:j+Window])] = loc{b, j - i}
	}
	g.Labels[a] = string(label[:i+Overlap])
	g.labelLen[a] = i + Overlap
	g.dirtyAll = true
	return a, b, nil
}

// MergeChild merges p's single child c into p when the chain is unary
// (p has exactly one child, c exactly one parent, no sentinels, p != c).
func (g *Graph) MergeChild(p int) bool {
	if p == Start || p == End || p < 0 || p >= len(g.Labels) || !g.Alive[p] {
		return false
	}
	ch := &g.children[p]
	if ch.size() != 1 {
		return false
	}
	c := ch.order[0]
	if c == p || c == Start || c == End {
		return false
	}
	pc := &g.parents[c]
	if pc.size() != 1 {
		return false
	}
	if g.blocksMerge(ch.edges[0]) {
		return false // a blamed transition stays an edge, so the negative network can still name it
	}
	lp := []rune(g.Labels[p])
	lc := []rune(g.Labels[c])
	shift := len(lp) - Overlap
	e := ch.edges[0]
	ch.clear()
	pc.clear()
	g.EdgeAlive[e] = false
	g.nAliveEdges--
	// activations are the constant 1: no rescale of the moved edges is needed
	cc := &g.children[c]
	for i, target := range cc.order {
		e2 := cc.edges[i]
		if target == c {
			target = p
		}
		pt := &g.parents[target]
		pt.unset(c)
		pt.set(p, e2)
		ch.set(target, e2)
		g.EdgeParent[e2] = p
	}
	cc.clear()
	for j := 0; j < len(lc)-Overlap; j++ {
		g.index[string(lc[j:j+Window])] = loc{p, shift + j}
	}
	g.Labels[p] = string(lp) + string(lc[Overlap:])
	g.labelLen[p] = len(lp) + len(lc) - Overlap
	g.Labels[c] = ""
	g.labelLen[c] = 0
	if g.Count[c] > g.Count[p] {
		g.Count[p] = g.Count[c]
	}
	g.Alive[c] = false
	g.nAliveNodes--
	g.Version++
	g.StructureVersion++
	g.dirtyAll = true
	return true
}

// Compress merges every unary chain until none remains; returns the merge count.
func (g *Graph) Compress() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	merges := 0
	for {
		done := 0
		for p := 2; p < len(g.Labels); p++ {
			if g.Alive[p] {
				for g.MergeChild(p) {
					done++
				}
			}
		}
		if done == 0 {
			return merges
		}
		merges += done
	}
}

// ObserveSequence registers a training sequence START -> t0 -> ... -> tn -> END,
// splitting nodes so every transition runs from the last trigram of one node
// to the first of another over an edge created on demand.  Returns the
// (parent, edge) transitions; count also bumps the visit counters.  Takes the
// structure write lock.
func (g *Graph) ObserveSequence(trigrams []string, count bool) ([]Transition, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.observeLocked(trigrams, count)
}

func (g *Graph) observeLocked(trigrams []string, count bool) ([]Transition, error) {
	if len(trigrams) == 0 {
		return nil, nil
	}
	didSplit := false
	transitions := make([]Transition, 0, len(trigrams)+1)

	x := trigrams[0]
	l, ok := g.index[x]
	var px, ox int
	if !ok {
		px, ox = g.createTrigramNode(x), 0
	} else {
		px, ox = l.node, l.off
	}
	if ox != 0 {
		_, b, err := g.Split(px, ox)
		if err != nil {
			return nil, err
		}
		px, ox = b, 0
		didSplit = true
	}
	e, ok := g.children[Start].get(px)
	if !ok {
		e = g.newEdge(Start, px, 0)
	}
	transitions = append(transitions, Transition{Start, e})
	if count {
		g.Count[Start]++
		g.Count[px]++
		g.EdgeCount[e]++
	}
	for idx := 1; idx < len(trigrams); idx++ {
		y := trigrams[idx]
		l, ok := g.index[y]
		var py, oy int
		if !ok {
			py, oy = g.createTrigramNode(y), 0
		} else {
			py, oy = l.node, l.off
		}
		if py == px && oy == ox+1 {
			ox = oy
			x = y
			continue
		}
		if runeSlice(x, 1, -1) != runeSlice(y, 0, Overlap) {
			return nil, fmt.Errorf("trigrams %q -> %q do not overlap", x, y)
		}
		if ox+Window < g.labelLen[px] {
			if _, _, err := g.Split(px, ox+1); err != nil {
				return nil, err
			}
			didSplit = true
			ly := g.index[y]
			py, oy = ly.node, ly.off
		}
		if oy != 0 {
			_, b, err := g.Split(py, oy)
			if err != nil {
				return nil, err
			}
			py, oy = b, 0
			didSplit = true
			lx := g.index[x]
			px, ox = lx.node, lx.off
		}
		e, ok := g.children[px].get(py)
		if !ok {
			e = g.newEdge(px, py, 0)
		}
		transitions = append(transitions, Transition{px, e})
		if count {
			g.Count[py]++
			g.EdgeCount[e]++
		}
		px, ox, x = py, 0, y
	}
	if ox+Window < g.labelLen[px] {
		if _, _, err := g.Split(px, ox+1); err != nil {
			return nil, err
		}
		didSplit = true
	}
	e, ok = g.children[px].get(End)
	if !ok {
		e = g.newEdge(px, End, 0)
	}
	transitions = append(transitions, Transition{px, e})
	if count {
		g.Count[End]++
		g.EdgeCount[e]++
	}
	if didSplit {
		traced, _, ok := g.trace(trigrams)
		if !ok {
			return nil, fmt.Errorf("internal error: observed sequence is not walkable")
		}
		transitions = traced
	}
	return transitions, nil
}

// Trace walks a sequence through the structure without modifying it (read
// lock): the transitions and the node path START ... END, or ok=false when a
// trigram is unknown, an edge is missing or a split would be needed.
func (g *Graph) Trace(trigrams []string) ([]Transition, []int, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.trace(trigrams)
}

// TraceUnlocked is Trace without the read lock, for read-only phases in which
// no goroutine changes the structure (the counting passes of an epoch).
func (g *Graph) TraceUnlocked(trigrams []string) ([]Transition, []int, bool) {
	return g.trace(trigrams)
}

func (g *Graph) trace(trigrams []string) ([]Transition, []int, bool) {
	if len(trigrams) == 0 {
		return nil, nil, false
	}
	l, ok := g.index[trigrams[0]]
	if !ok || l.off != 0 {
		return nil, nil, false
	}
	px, ox := l.node, l.off
	e, ok := g.children[Start].get(px)
	if !ok {
		return nil, nil, false
	}
	transitions := make([]Transition, 0, len(trigrams)+1)
	transitions = append(transitions, Transition{Start, e})
	path := make([]int, 0, len(trigrams)+2)
	path = append(path, Start, px)
	for idx := 1; idx < len(trigrams); idx++ {
		l, ok := g.index[trigrams[idx]]
		if !ok {
			return nil, nil, false
		}
		py, oy := l.node, l.off
		if py == px && oy == ox+1 {
			ox = oy
			continue
		}
		if oy != 0 || ox+Window != g.labelLen[px] {
			return nil, nil, false
		}
		e, ok := g.children[px].get(py)
		if !ok {
			return nil, nil, false
		}
		transitions = append(transitions, Transition{px, e})
		path = append(path, py)
		px, ox = py, 0
	}
	if ox+Window != g.labelLen[px] {
		return nil, nil, false
	}
	e, ok = g.children[px].get(End)
	if !ok {
		return nil, nil, false
	}
	transitions = append(transitions, Transition{px, e})
	path = append(path, End)
	return transitions, path, true
}

// NodePath is the node path of a sequence (START ... END) or ok=false.
func (g *Graph) NodePath(trigrams []string) ([]int, bool) {
	_, path, ok := g.Trace(trigrams)
	return path, ok
}

// CheckInvariants verifies the structural invariants (and, with texts, that
// every text walks through the graph and decodes back to itself); compressed
// additionally requires that no unary chain remains.
func (g *Graph) CheckInvariants(texts []string, compressed bool) error {
	n := len(g.Labels)
	if !(len(g.Count) == n && len(g.Alive) == n && len(g.children) == n && len(g.parents) == n && len(g.labelLen) == n) {
		return fmt.Errorf("node arrays have inconsistent lengths")
	}
	m := len(g.EdgeW)
	if !(len(g.EdgeCount) == m && len(g.EdgeAlive) == m && len(g.EdgeReward) == m && len(g.WindowEdgeCount) == m && len(g.EdgeParent) == m) {
		return fmt.Errorf("edge arrays have inconsistent lengths")
	}
	if n < 2 || !g.Alive[Start] || !g.Alive[End] || g.Labels[Start] != StartLabel || g.Labels[End] != EndLabel {
		return fmt.Errorf("sentinels missing or changed")
	}
	if g.parents[Start].size() != 0 || g.children[End].size() != 0 {
		return fmt.Errorf("START has parents or END has children")
	}
	aliveNodes := 0
	seen := make(map[int]bool)
	for p := 0; p < n; p++ {
		if !g.Alive[p] {
			if g.children[p].size() != 0 || g.parents[p].size() != 0 {
				return fmt.Errorf("dead node %d still has edges", p)
			}
			continue
		}
		aliveNodes++
		if p > End && g.labelLen[p] < Window {
			return fmt.Errorf("node %d label %q shorter than %d", p, g.Labels[p], Window)
		}
		if g.labelLen[p] != runeLen(g.Labels[p]) {
			return fmt.Errorf("node %d has a stale label length", p)
		}
		for i, c := range g.children[p].order {
			e := g.children[p].edges[i]
			if e < 0 || e >= m || !g.EdgeAlive[e] || seen[e] {
				return fmt.Errorf("edge %d on %d->%d is invalid, dead or listed twice", e, p, c)
			}
			seen[e] = true
			if !g.Alive[c] || c == Start || p == End {
				return fmt.Errorf("edge %d->%d touches a dead node or a sentinel illegally", p, c)
			}
			if got, ok := g.parents[c].get(p); !ok || got != e {
				return fmt.Errorf("edge %d->%d missing from parents", p, c)
			}
			if g.EdgeParent[e] != p {
				return fmt.Errorf("edge %d records parent %d, listed under %d", e, g.EdgeParent[e], p)
			}
			if p > End && c > End {
				lp := []rune(g.Labels[p])
				if string(lp[len(lp)-Overlap:]) != runeSlice(g.Labels[c], 0, Overlap) {
					return fmt.Errorf("edge %d->%d violates the window overlap", p, c)
				}
			}
		}
		for i, q := range g.parents[p].order {
			e := g.parents[p].edges[i]
			if got, ok := g.children[q].get(p); !ok || got != e {
				return fmt.Errorf("parents[%d] lists %d but children[%d] does not", p, q, q)
			}
		}
	}
	if aliveNodes != g.nAliveNodes {
		return fmt.Errorf("alive node counter is stale")
	}
	aliveEdges := 0
	for _, ok := range g.EdgeAlive {
		if ok {
			aliveEdges++
		}
	}
	if len(seen) != aliveEdges || aliveEdges != g.nAliveEdges {
		return fmt.Errorf("alive edge bookkeeping is stale")
	}
	expected := 0
	for p := 2; p < n; p++ {
		if !g.Alive[p] {
			continue
		}
		label := []rune(g.Labels[p])
		for o := 0; o < len(label)-Overlap; o++ {
			t := string(label[o : o+Window])
			if l, ok := g.index[t]; !ok || l.node != p || l.off != o {
				return fmt.Errorf("trigram %q of node %d@%d indexed as %v", t, p, o, l)
			}
			expected++
		}
	}
	if len(g.index) != expected {
		return fmt.Errorf("trigram index has %d entries, expected %d", len(g.index), expected)
	}
	if compressed {
		for p := 2; p < n; p++ {
			if g.Alive[p] && g.children[p].size() == 1 {
				c := g.children[p].order[0]
				if !(c == p || c <= End || g.parents[c].size() != 1) {
					return fmt.Errorf("unary chain %d->%d survived compress", p, c)
				}
			}
		}
	}
	for _, text := range texts {
		grams := Encode(text)
		if grams == nil {
			continue
		}
		path, ok := g.NodePath(grams)
		if !ok {
			return fmt.Errorf("text %q is not walkable through the graph", text)
		}
		labels := make([]string, 0, len(path))
		for _, id := range path[1 : len(path)-1] {
			labels = append(labels, g.Labels[id])
		}
		if decoded := DecodePath(labels, 0, true); decoded != text {
			return fmt.Errorf("round trip of %q gave %q", text, decoded)
		}
	}
	return nil
}
