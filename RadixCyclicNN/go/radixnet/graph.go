package radixnet

import (
	"fmt"
	"strings"
	"sync"
)

// Node ids of the sentinels.  Start and End are where a text begins and ends -
// observed in the corpus, like everything else.  Back is where the graph has
// *learned* that a walk goes round: nothing in a corpus says so, so its edges
// are taught by the voices that caught themselves repeating (Backtrack).  An
// edge p -> Back competes for probability with p's real children, so the more
// often walks through p had to be backed out of, the likelier the search is to
// hand over there instead of carrying on.  Think is where the graph has learned
// to stop and *think*: its in-edges are taught by experience too - an event at
// p made the model think there (thinking.go) - and, like Back's, compete with
// the real children for probability; unlike Back it also has out-edges, which
// are how thoughts begin, because a thought is trained from Think the way a
// text is trained from Start (IsOrigin).  First is the first node id that is
// not a sentinel.
const (
	Start = 0
	End   = 1
	Back  = 2
	Think = 3
	First = 4
)

// IsOrigin reports whether node is a sentinel a sequence may begin at: Start
// for a text, Think for a thought.
func IsOrigin(node int) bool { return node == Start || node == Think }

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

// BackZ is the Back sentinel's state: fixed at the far edge of the range the
// sine model draws from instead of drawn, so adding the sentinel moved no
// random stream and its activation is firmly non-zero (see the Python graph).
const BackZ = 4.5

// ThinkZ is the Think sentinel's state: fixed at the *other* edge of that range,
// as firmly non-zero as Back's and drawn from no stream either.
const ThinkZ = -4.5

const (
	graphFormat = "radixnet-graph"
	// 2 added the counter reset fields (version 1 files load with no resets); 3 the Back sentinel and 4 the
	// Think sentinel (older files gain an unvisited one on load, and their node ids shift up by one)
	graphFormatVersion = 4
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

	// Every visit count is a cyclic counter (counter.go): the slices above hold
	// the odometer readings, these maps the reset counts of the ids that ever
	// wrapped (absent = 0), and CarryCounters does the wrapping - never the
	// counting loops, which stay plain or atomic increments.
	CountResets     map[int]int64
	EdgeCountResets map[int]int64
	// Traversals counts every increment made to those counters and so bounds
	// each of them: while it has not wrapped, none of them can have.
	Traversals Counter

	// Neg is the negative network's evidence (nil on a count / reward graph).
	Neg *NegativeData

	// the count / reward numbers
	EdgeReward      []float64
	WindowEdgeCount []int64
	window          []int
	windowHead      int
	WindowSize      int
	TotalTraversals Counter
	CountScale      float64
	RewardScale     float64
	GlobalScale     float64
	WindowScale     float64

	index    map[string]loc
	Inverted bool

	// Enc is how this graph turns text into grams and its labels back into
	// text: the unit (characters or words), the n of the n-gram and the
	// stride between consecutive grams.  It is fixed when the graph is
	// created - every label, every index key and every offset is measured in
	// its units - and travels with the model file.
	Enc Encoding

	Version          Counter
	StructureVersion Counter
	nAliveNodes      int
	nAliveEdges      int

	// what the walks did: (the node before an edge's parent, the edge) -> seen / correct / incorrect
	paths       map[PathKey]*PathRow
	pathsByEdge map[int]map[int]bool // edge -> the nodes that called it
	pathsByPrev map[int]map[int]bool // node -> the edges it called
	pathParents map[int]bool         // nodes whose costs depend on where the walk came from (lazy)
	ctxCache    map[PathKey][]ChildCost
	ctxVersion  Counter
	PathScale   float64

	// lazy weights and costs
	dirty            map[int]struct{}
	dirtyAll         bool
	weightsStructure Counter
	edgeCost         []float64
	edgePunish       []float64 // the penalty side of every edge's reward (EdgePunishment)
	costsVersion     Counter

	// mu guards structural changes: splits / merges / new nodes and edges take
	// the write lock, concurrent traces the read lock.
	mu      sync.RWMutex
	Workers int
}

// GraphOptions are the dual frequency function's scales, the size of the
// sliding count window and the encoding the graph is built in.
type GraphOptions struct {
	CountScale  float64
	RewardScale float64
	GlobalScale float64
	WindowScale float64
	PathScale   float64
	// Window is the size of the sliding *count* window (the recent share of
	// the weight function), not the n of the n-gram - that is Encoding.N.
	Window int
	// Encoding is how text becomes grams; the zero value is the character
	// trigram of stride 1 the model was born with.
	Encoding Encoding
}

// DefaultGraphOptions mirror the Python defaults (geometric mean of the two shares).
func DefaultGraphOptions() GraphOptions {
	return GraphOptions{CountScale: 0, RewardScale: 1, GlobalScale: 0.5, WindowScale: 0.5, PathScale: 1,
		Window: 10_000, Encoding: DefaultEncoding()}
}

// NewGraph creates an empty graph with the two sentinels.
func NewGraph(seed int64, opts GraphOptions) (*Graph, error) {
	if opts.Window < 1 {
		return nil, fmt.Errorf("window must be >= 1, got %d", opts.Window)
	}
	enc := opts.Encoding.WithDefaults()
	if err := enc.Validate(); err != nil {
		return nil, err
	}
	g := &Graph{
		Seed:             seed,
		rng:              NewMT19937(seed),
		index:            make(map[string]loc),
		dirty:            make(map[int]struct{}),
		CountResets:      map[int]int64{},
		EdgeCountResets:  map[int]int64{},
		weightsStructure: invalidStamp,
		costsVersion:     invalidStamp,
		Enc:              enc,
		WindowSize:       opts.Window,
		CountScale:       opts.CountScale,
		RewardScale:      opts.RewardScale,
		GlobalScale:      opts.GlobalScale,
		WindowScale:      opts.WindowScale,
		PathScale:        opts.PathScale,
		paths:            map[PathKey]*PathRow{},
		pathsByEdge:      map[int]map[int]bool{},
		pathsByPrev:      map[int]map[int]bool{},
		ctxVersion:       invalidStamp,
		Workers:          Workers,
	}
	g.newNode(StartLabel, 0, 0)
	g.newNode(EndLabel, 0, 0)
	g.newNode(BackLabel, 0, 0)
	g.newNode(ThinkLabel, 0, 0)
	return g, nil
}

// -- counters ------------------------------------------------------------------

// invalidStamp is a version stamp no real reading can equal, so a cache marked
// with it is always stale.
var invalidStamp = Counter{Value: -1}

// setResets stores one reset count, keeping the map sparse (absent = never wrapped).
func setResets(resets map[int]int64, key int, value int64) {
	if value != 0 {
		resets[key] = value
	} else {
		delete(resets, key)
	}
}

// NodeCount is how often node i was visited, as an odometer reading.
func (g *Graph) NodeCount(i int) Counter { return Counter{g.Count[i], g.CountResets[i]} }

// EdgeTraversals is how often edge e was traversed, as an odometer reading.
func (g *Graph) EdgeTraversals(e int) Counter { return Counter{g.EdgeCount[e], g.EdgeCountResets[e]} }

// edgeTraversalsF is the exact traversal count of an edge, as the float the
// weight function sums (the Python implementation computes it the same way).
func (g *Graph) edgeTraversalsF(e int) float64 {
	return counterTotal(g.EdgeCount[e], g.EdgeCountResets, e)
}

// CarryCounters sets every counter that reached CounterLimit back to 0,
// counting the reset; returns how many wrapped.
//
// This is the only place the visit counters wrap, so the counting loops stay
// plain (or atomic) increments and several goroutines can count into a raw
// int64 at once.  Call it at a safe point - the end of an epoch, before a save
// - while nothing else is touching the graph.  While the graph has not seen
// CounterLimit increments no counter can have reached the limit (Traversals
// counts them all and so bounds every single one), so the sweep is skipped
// after one comparison; force runs it anyway.
func (g *Graph) CarryCounters(force bool) int {
	wrapped := 0
	if force || g.Traversals.Resets != 0 {
		wrapped = CarrySeries(g.Count, g.CountResets) + CarrySeries(g.EdgeCount, g.EdgeCountResets)
	}
	if n := g.Neg; n != nil && (force || n.TotalFails.Resets != 0) {
		// the negative network's fail counts, bounded by TotalFails the same way
		wrapped += CarrySeries(n.Fails, n.FailsResets) + CarrySeries(n.ReasonFails, n.ReasonFailsResets)
	}
	return wrapped
}

// -- construction --------------------------------------------------------------

func (g *Graph) newNode(label string, count, countResets int64) int {
	nid := len(g.Labels)
	g.Labels = append(g.Labels, label)
	g.labelLen = append(g.labelLen, g.Enc.Len(label))
	g.Count = append(g.Count, count)
	if countResets != 0 {
		g.CountResets[nid] = countResets
	}
	g.Alive = append(g.Alive, true)
	g.children = append(g.children, adjacency{})
	g.parents = append(g.parents, adjacency{})
	g.nAliveNodes++
	g.Version.Add(1)
	g.StructureVersion.Add(1)
	return nid
}

func (g *Graph) newEdge(p, c int, count, countResets int64) int {
	// the sine network draws a weight here; the count model recomputes the
	// weight but consumes the same random number
	g.rng.Uniform(WLow, WHigh)
	e := len(g.EdgeW)
	g.EdgeW = append(g.EdgeW, 0.0)
	g.EdgeCount = append(g.EdgeCount, count)
	if countResets != 0 {
		g.EdgeCountResets[e] = countResets
	}
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
	g.Version.Add(1)
	g.StructureVersion.Add(1)
	g.dirty[p] = struct{}{}
	return e
}

func (g *Graph) createGramNode(gram string) int {
	if g.Enc.Len(gram) != g.Enc.N {
		panic(fmt.Sprintf("expected a gram of %d %s, got %q", g.Enc.N, g.Enc.Unit, gram))
	}
	// a gram cut out of a text is a slice of it, and this one is about to
	// outlive the text: copy it, or the whole line stays in memory behind it
	gram = strings.Clone(gram)
	nid := g.newNode(gram, 0, 0)
	g.index[gram] = loc{nid, 0}
	return nid
}

// Nothing in a corpus says where a walk loops, so this is the one thing the
// graph learns from *experience* rather than from observation - a voice that
// caught itself repeating and had to back up (Backtrack).  Three things are
// taught at once, and all three are ordinary learned quantities:
//
//   - p -> Back is created on first use and bumped like any observed
//     transition.  It competes with p's real children for probability, so every
//     hand-over raises the model's own estimate that walks through p go round -
//     and once that estimate beats the real children, the search hands over
//     there by itself, wherever it is walking.
//   - went - the child the walk was about to loop through - gets amount dearer.
//   - instead - the child the voice took after backing up - gets amount cheaper.
//
// The first is where it goes round; the other two are what to do instead.
// Pass -1 for a child that is not known.  Returns the Back edge id.
// NudgeEdge moves the weight of p -> c so the transition gets amount likelier
// (negative: dearer).  The count model derives its weights from counts and
// rewards, so this is where the sine model's direct nudge would go; here it
// only reports whether such an edge exists.
func (g *Graph) NudgeEdge(p, c int, amount float64) bool {
	if p < First || p >= len(g.children) {
		return false
	}
	_, ok := g.children[p].get(c)
	return ok && amount != 0
}

// ObserveBack teaches what a voice learned by backing out of a repeat at p.
func (g *Graph) ObserveBack(p int, went, instead int, amount float64) (int, error) {
	if p < First || p >= len(g.Labels) || !g.Alive[p] {
		return 0, fmt.Errorf("node %d is not a real node to go back from", p)
	}
	if amount < 0 {
		return 0, fmt.Errorf("amount must be >= 0, got %v", amount)
	}
	e, ok := g.children[p].get(Back)
	if !ok {
		e = g.newEdge(p, Back, 0, 0)
	}
	g.Count[Back]++
	g.EdgeCount[e]++
	g.Traversals.Add(1)
	g.Version.Add(1)
	// a weight here is a function of the counts and the rewards, so going round is taught by the Back edge's
	// count and what to do instead by a penalty on the step it looped through and a reward on the step it took
	// after backing up - the same rewards 2NRL moves (the sine model nudges the weights themselves instead)
	g.RecordTraversals([]int{e})
	g.AddReward([]int{e}, amount) // the hand-over itself, learned the way this model learns everything
	for _, pair := range [2]struct {
		child int
		sign  float64
	}{{went, -1}, {instead, 1}} {
		if pair.child < First || pair.child == Back {
			continue
		}
		if edge, ok := g.children[p].get(pair.child); ok {
			g.AddReward([]int{edge}, pair.sign*amount)
		}
	}
	g.RecomputeWeights()
	return e, nil
}

// BackCost is what the model thinks a walk arriving at p costs to go round, or
// ok = false when it has no idea (a node that was never backed out of).  When it
// is the cheapest of p's children the model's most likely next step there is to
// stop, which is what the search acts on.
func (g *Graph) BackCost(p int) (float64, bool) {
	for _, it := range g.ChildCosts(p) {
		if it.Child == Back {
			return it.Cost, true
		}
	}
	return 0, false
}

// ObserveThink teaches that something at p made the model stop and think.
//
// The twin of ObserveBack, learned the same way - from experience, never from a
// corpus: an event at p (a voice catching itself repeating, a question asked
// about the text that ends here, a thought questioning itself) called for a
// thought, and the model remembers where (Think in thinking.go).  p -> Think is
// created on first use and bumped like any observed transition; it competes
// with p's real children for probability, so the oftener walks through p had to
// stop and think, the likelier a thought passing through p is to question itself
// there.  Nothing is taught about what to do instead - that is the thought's
// business, and what it hands over to when it stops (Back, or nothing).
// Returns the Think edge id.
func (g *Graph) ObserveThink(p int, amount float64) (int, error) {
	if p < First || p >= len(g.Labels) || !g.Alive[p] {
		return 0, fmt.Errorf("node %d is not a real node to think at", p)
	}
	if amount < 0 {
		return 0, fmt.Errorf("amount must be >= 0, got %v", amount)
	}
	e, ok := g.children[p].get(Think)
	if !ok {
		e = g.newEdge(p, Think, 0, 0)
	}
	g.Count[Think]++
	g.EdgeCount[e]++
	g.Traversals.Add(1)
	g.Version.Add(1)
	// as with Back: the count model learns the transition from its count and a reward on the edge
	g.RecordTraversals([]int{e})
	g.AddReward([]int{e}, amount)
	g.RecomputeWeights()
	return e, nil
}

// ThinkCost is what the model thinks it costs to stop and think at p, or
// ok = false when it never had to: the cost of p's Think edge, to compare with
// the costs of its real children.  When it is the cheapest of them the model's
// most likely next step there is to question what it is doing, which is what a
// thought passing through p acts on.
func (g *Graph) ThinkCost(p int) (float64, bool) {
	for _, it := range g.ChildCosts(p) {
		if it.Child == Think {
			return it.Cost, true
		}
	}
	return 0, false
}

// ThinksAt reports whether the model has learned to stop and think at p: its
// Think edge is the cheapest way on (Back is not a way on and does not count).
func (g *Graph) ThinksAt(p int) bool {
	costs := g.ChildCosts(p)
	thinking, ok := 0.0, false
	for _, it := range costs {
		if it.Child == Think {
			thinking, ok = it.Cost, true
			break
		}
	}
	if !ok {
		return false
	}
	for _, it := range costs {
		if it.Child != Think && it.Child != Back && it.Cost < thinking {
			return false
		}
	}
	return true
}

// -- sizes and lookup ------------------------------------------------------------

// NumNodes is the number of alive nodes including the sentinels.
func (g *Graph) NumNodes() int { return g.nAliveNodes }

// NumEdges is the number of alive edges.
func (g *Graph) NumEdges() int { return g.nAliveEdges }

// NumTrigrams is the number of distinct grams stored (trigrams under the
// default encoding, which is where the name comes from).
func (g *Graph) NumTrigrams() int { return len(g.index) }

// NumGrams is the number of distinct grams stored, whatever the encoding.
func (g *Graph) NumGrams() int { return len(g.index) }

// CompressionRatio is grams per real node.
func (g *Graph) CompressionRatio() float64 {
	real := g.nAliveNodes - First
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

// Lookup finds the node and the unit offset holding a gram.
func (g *Graph) Lookup(gram string) (node, off int, ok bool) {
	l, found := g.index[gram]
	return l.node, l.off, found
}

// Encode cuts a text into this graph's grams (Graph.Enc).
func (g *Graph) Encode(text string) []string { return g.Enc.Encode(text) }

// DecodePath turns node labels back into text under this graph's encoding.
func (g *Graph) DecodePath(labels []string, startOffset int, includeContext bool) string {
	return g.Enc.DecodePath(labels, startOffset, includeContext)
}

// LabelLen is the length of a node's label in the encoding's units.
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

// Split cuts a node between the gram starting at unit i and the one before
// it: A keeps the id and the label up to the end of the earlier gram, B is a
// new node with label[i:] that inherits A's out-edges (edge ids kept) and
// count; A gets the single new edge A -> B.  i is a unit offset and must be
// the start of one of the node's grams - a positive multiple of the stride.
func (g *Graph) Split(node, i int) (int, int, error) {
	if node < First {
		return 0, 0, fmt.Errorf("cannot split a sentinel node")
	}
	if node < 0 || node >= len(g.Labels) || !g.Alive[node] {
		return 0, 0, fmt.Errorf("node %d is not alive", node)
	}
	enc := g.Enc
	label := enc.Units(g.Labels[node])
	length := label.Len()
	stride, overlap := enc.Stride, enc.Overlap()
	if i < stride || i > length-enc.N || i%stride != 0 {
		return 0, 0, fmt.Errorf("split index %d out of range %d..%d (a multiple of the stride %d) for label %q",
			i, stride, length-enc.N, stride, label.Text())
	}
	a := node
	aResets := g.CountResets[a]
	b := g.newNode(label.Slice(i, -1), g.Count[a], aResets)
	chA := &g.children[a]
	chB := &g.children[b]
	moved := append([]int(nil), chA.edges...)
	for i, c := range chA.order {
		e := chA.edges[i]
		chB.set(c, e)
		pc := &g.parents[c]
		pc.unset(a)
		pc.set(b, e)
		g.EdgeParent[e] = b
	}
	chA.clear()
	g.newEdge(a, b, g.Count[a], aResets)
	for j := i; j <= length-enc.N; j += stride {
		g.index[label.Slice(j, j+enc.N)] = loc{b, j - i}
	}
	g.Labels[a] = label.Slice(0, i+overlap)
	g.labelLen[a] = i + overlap
	g.dirtyAll = true
	bridge := -1
	if e, ok := g.children[a].get(b); ok {
		bridge = e
	}
	g.splitPaths(a, b, moved, bridge) // q -> P -> c is now q -> A -> B -> c
	g.pathParents = nil
	return a, b, nil
}

// MergeChild merges p's single child c into p when the chain is unary
// (p has exactly one child, c exactly one parent, no sentinels, p != c).
func (g *Graph) MergeChild(p int) bool {
	if p < First || p >= len(g.Labels) || !g.Alive[p] {
		return false
	}
	ch := &g.children[p]
	if ch.size() != 1 {
		return false
	}
	c := ch.order[0]
	if c == p || c < First {
		return false
	}
	pc := &g.parents[c]
	if pc.size() != 1 {
		return false
	}
	if g.blocksMerge(ch.edges[0]) {
		return false // a blamed transition stays an edge, so the negative network can still name it
	}
	enc := g.Enc
	lp := enc.Units(g.Labels[p])
	lc := enc.Units(g.Labels[c])
	overlap := enc.Overlap()
	shift := lp.Len() - overlap
	e := ch.edges[0]
	movedOut := append([]int(nil), g.children[c].edges...)
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
	for j := 0; j <= lc.Len()-enc.N; j += enc.Stride {
		g.index[lc.Slice(j, j+enc.N)] = loc{p, shift + j}
	}
	g.Labels[p] = enc.Join(lp.Text(), lc.Slice(overlap, -1))
	g.labelLen[p] = lp.Len() + lc.Len() - overlap
	g.Labels[c] = ""
	g.labelLen[c] = 0
	if g.NodeCount(p).Less(g.NodeCount(c)) {
		g.Count[p] = g.Count[c]
		setResets(g.CountResets, p, g.CountResets[c])
	}
	g.Alive[c] = false
	g.nAliveNodes--
	g.Version.Add(1)
	g.StructureVersion.Add(1)
	g.dirtyAll = true
	g.mergePaths(p, c, e, movedOut) // the chain was unary: what its contexts knew was never a choice
	g.pathParents = nil
	return true
}

// Compress merges every unary chain until none remains; returns the merge count.
func (g *Graph) Compress() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	merges := 0
	for {
		done := 0
		for p := First; p < len(g.Labels); p++ {
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
// splitting nodes so every transition runs from the last gram of one node to
// the first of another over an edge created on demand.  Returns the
// (parent, edge) transitions; count also bumps the visit counters.  Takes the
// structure write lock.
func (g *Graph) ObserveSequence(grams []string, count bool) ([]Transition, error) {
	return g.ObserveFrom(Start, grams, count)
}

// ObserveFrom is ObserveSequence from the sentinel origin: Start for a text,
// Think for a *thought* - the same structure, the same counting, the same edge
// into End, only the first edge leaves the other sentinel (IsOrigin).
func (g *Graph) ObserveFrom(origin int, grams []string, count bool) ([]Transition, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.observeFrom(origin, grams, count)
}

func (g *Graph) observeLocked(trigrams []string, count bool) ([]Transition, error) {
	return g.observeFrom(Start, trigrams, count)
}

func (g *Graph) observeFrom(origin int, trigrams []string, count bool) ([]Transition, error) {
	if len(trigrams) == 0 {
		return nil, nil
	}
	if !IsOrigin(origin) {
		return nil, fmt.Errorf("a sequence begins at Start or Think, not at node %d", origin)
	}
	enc := g.Enc
	stride, overlap := enc.Stride, enc.Overlap()
	didSplit := false
	transitions := make([]Transition, 0, len(trigrams)+1)

	x := trigrams[0]
	l, ok := g.index[x]
	var px, ox int
	if !ok {
		px, ox = g.createGramNode(x), 0
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
	e, ok := g.children[origin].get(px)
	if !ok {
		e = g.newEdge(origin, px, 0, 0)
	}
	transitions = append(transitions, Transition{origin, e})
	if count {
		g.Count[origin]++
		g.Count[px]++
		g.EdgeCount[e]++
	}
	for idx := 1; idx < len(trigrams); idx++ {
		y := trigrams[idx]
		l, ok := g.index[y]
		var py, oy int
		if !ok {
			py, oy = g.createGramNode(y), 0
		} else {
			py, oy = l.node, l.off
		}
		if py == px && oy == ox+stride {
			ox = oy
			x = y
			continue
		}
		if enc.Slice(x, stride, -1) != enc.Slice(y, 0, overlap) {
			return nil, fmt.Errorf("grams %q -> %q do not overlap", x, y)
		}
		if ox+enc.N < g.labelLen[px] {
			if _, _, err := g.Split(px, ox+stride); err != nil {
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
			e = g.newEdge(px, py, 0, 0)
		}
		transitions = append(transitions, Transition{px, e})
		if count {
			g.Count[py]++
			g.EdgeCount[e]++
		}
		px, ox, x = py, 0, y
	}
	if ox+enc.N < g.labelLen[px] {
		if _, _, err := g.Split(px, ox+stride); err != nil {
			return nil, err
		}
		didSplit = true
	}
	e, ok = g.children[px].get(End)
	if !ok {
		e = g.newEdge(px, End, 0, 0)
	}
	transitions = append(transitions, Transition{px, e})
	if count {
		g.Count[End]++
		g.EdgeCount[e]++
		// every transition bumped one node counter and one edge counter, plus the origin's:
		// the total bounds each of them and so decides when CarryCounters has work
		g.Traversals.Add(int64(2*len(transitions) + 1))
	}
	if didSplit {
		traced, _, ok := g.traceFrom(origin, trigrams)
		if !ok {
			return nil, fmt.Errorf("internal error: observed sequence is not walkable")
		}
		transitions = traced
	}
	return transitions, nil
}

// Trace walks a sequence through the structure without modifying it (read
// lock): the transitions and the node path START ... END, or ok=false when a
// gram is unknown, an edge is missing or a split would be needed.
func (g *Graph) Trace(trigrams []string) ([]Transition, []int, bool) {
	return g.TraceFrom(Start, trigrams)
}

// TraceFrom is Trace from the sentinel origin (Start for a text, Think for a
// thought): the path begins there.
func (g *Graph) TraceFrom(origin int, trigrams []string) ([]Transition, []int, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.traceFrom(origin, trigrams)
}

// TraceUnlocked is Trace without the read lock, for read-only phases in which
// no goroutine changes the structure (the counting passes of an epoch).
func (g *Graph) TraceUnlocked(trigrams []string) ([]Transition, []int, bool) {
	return g.traceFrom(Start, trigrams)
}

func (g *Graph) trace(trigrams []string) ([]Transition, []int, bool) {
	return g.traceFrom(Start, trigrams)
}

func (g *Graph) traceFrom(origin int, trigrams []string) ([]Transition, []int, bool) {
	if len(trigrams) == 0 || !IsOrigin(origin) {
		return nil, nil, false
	}
	stride := g.Enc.Stride
	l, ok := g.index[trigrams[0]]
	if !ok || l.off != 0 {
		return nil, nil, false
	}
	px, ox := l.node, l.off
	e, ok := g.children[origin].get(px)
	if !ok {
		return nil, nil, false
	}
	transitions := make([]Transition, 0, len(trigrams)+1)
	transitions = append(transitions, Transition{origin, e})
	path := make([]int, 0, len(trigrams)+2)
	path = append(path, origin, px)
	for idx := 1; idx < len(trigrams); idx++ {
		l, ok := g.index[trigrams[idx]]
		if !ok {
			return nil, nil, false
		}
		py, oy := l.node, l.off
		if py == px && oy == ox+stride {
			ox = oy
			continue
		}
		if oy != 0 || ox+g.Enc.N != g.labelLen[px] {
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
	if ox+g.Enc.N != g.labelLen[px] {
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

// NodePathFrom is NodePath from the sentinel origin (a thought's path begins at Think).
func (g *Graph) NodePathFrom(origin int, trigrams []string) ([]int, bool) {
	_, path, ok := g.TraceFrom(origin, trigrams)
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
	if n < First || !g.Alive[Start] || !g.Alive[End] || !g.Alive[Back] || !g.Alive[Think] ||
		g.Labels[Start] != StartLabel || g.Labels[End] != EndLabel || g.Labels[Back] != BackLabel ||
		g.Labels[Think] != ThinkLabel {
		return fmt.Errorf("sentinels missing or changed")
	}
	if g.parents[Start].size() != 0 || g.children[End].size() != 0 {
		return fmt.Errorf("START has parents or END has children")
	}
	if g.children[Back].size() != 0 {
		return fmt.Errorf("BACK must not have children")
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
		if p >= First && g.labelLen[p] < g.Enc.N {
			return fmt.Errorf("node %d label %q shorter than %d", p, g.Labels[p], g.Enc.N)
		}
		if g.labelLen[p] != g.Enc.Len(g.Labels[p]) {
			return fmt.Errorf("node %d has a stale label length", p)
		}
		for i, c := range g.children[p].order {
			e := g.children[p].edges[i]
			if e < 0 || e >= m || !g.EdgeAlive[e] || seen[e] {
				return fmt.Errorf("edge %d on %d->%d is invalid, dead or listed twice", e, p, c)
			}
			seen[e] = true
			if !g.Alive[c] || c == Start || p == End || p == Back {
				return fmt.Errorf("edge %d->%d touches a dead node or a sentinel illegally", p, c)
			}
			if got, ok := g.parents[c].get(p); !ok || got != e {
				return fmt.Errorf("edge %d->%d missing from parents", p, c)
			}
			if g.EdgeParent[e] != p {
				return fmt.Errorf("edge %d records parent %d, listed under %d", e, g.EdgeParent[e], p)
			}
			if p >= First && c >= First && g.Enc.Overlap() > 0 {
				lp := g.Enc.Units(g.Labels[p])
				if lp.Slice(lp.Len()-g.Enc.Overlap(), -1) != g.Enc.Slice(g.Labels[c], 0, g.Enc.Overlap()) {
					return fmt.Errorf("edge %d->%d violates the gram overlap", p, c)
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
	for p := First; p < n; p++ {
		if !g.Alive[p] {
			continue
		}
		label := g.Enc.Units(g.Labels[p])
		for o := 0; o <= label.Len()-g.Enc.N; o += g.Enc.Stride {
			t := label.Slice(o, o+g.Enc.N)
			if l, ok := g.index[t]; !ok || l.node != p || l.off != o {
				return fmt.Errorf("gram %q of node %d@%d indexed as %v", t, p, o, l)
			}
			expected++
		}
		if rest := (label.Len() - g.Enc.N) % g.Enc.Stride; rest != 0 {
			return fmt.Errorf("node %d label %q holds %d units, %d past its last whole gram",
				p, g.Labels[p], label.Len(), rest)
		}
	}
	if len(g.index) != expected {
		return fmt.Errorf("gram index has %d entries, expected %d", len(g.index), expected)
	}
	if compressed {
		for p := First; p < n; p++ {
			if g.Alive[p] && g.children[p].size() == 1 {
				c := g.children[p].order[0]
				if !(c == p || c < First || g.parents[c].size() != 1) {
					return fmt.Errorf("unary chain %d->%d survived compress", p, c)
				}
			}
		}
	}
	for _, text := range texts {
		grams := g.Encode(text)
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
		// what comes back is what the encoding can represent: the original
		// text under a sliding character window, its words under a word
		// encoding, and everything but the ragged tail under a grouping one
		if want := g.Enc.Normalize(text); g.DecodePath(labels, 0, true) != want {
			return fmt.Errorf("round trip of %q gave %q", text, g.DecodePath(labels, 0, true))
		}
	}
	return nil
}
