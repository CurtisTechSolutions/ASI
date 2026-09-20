package radixnet

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ModelFormat is the file format shared with the Python count model.
const (
	ModelFormat        = "radixnet-count"
	ModelFormatVersion = 1
	// WordFormat is the word model's own format, so that every reader written
	// before it existed refuses the file by the check it already makes
	// (../../SPEC-WordNGrams.md).
	WordFormat = "radixnet-word"
)

type nodesDoc struct {
	Labels []string  `json:"labels"`
	Z      []float64 `json:"z"`
	A      []float64 `json:"a"`
	B      []float64 `json:"b"`
	H      []float64 `json:"h"`
	K      []float64 `json:"k"`
	Count  []int64   `json:"count"`
	// how often each node's counter wrapped; written only once something has
	// (see counter.go), so an ordinary file carries no reset arrays at all
	CountResets []int64 `json:"count_resets,omitempty"`
}

type edgesDoc struct {
	Src    []int     `json:"src"`
	Dst    []int     `json:"dst"`
	W      []float64 `json:"w"`
	Count  []int64   `json:"count"`
	Reward []float64 `json:"reward"`
	// how often each edge's counter wrapped; written only once something has
	// (see counter.go), so an ordinary file carries no reset arrays at all
	CountResets []int64 `json:"count_resets,omitempty"`

	// the negative network's evidence (omitted on a count / reward graph)
	Blame       []float64      `json:"blame,omitempty"`
	Fails       []int64        `json:"fails,omitempty"`
	FailsResets []int64        `json:"fails_resets,omitempty"`
	Clear       []float64      `json:"clear,omitempty"`
	Reasons     [][][2]float64 `json:"reasons,omitempty"`
}

// reasonRegistryDoc is the negative graph's reason registry.
type reasonRegistryDoc struct {
	Labels []string  `json:"labels"`
	Blame  []float64 `json:"blame"`
	Fails  []int64   `json:"fails"`
	// how often each reason's fail counter wrapped; written only once one has
	FailsResets []int64 `json:"fails_resets,omitempty"`
}

// negativeWeightsDoc is the "weights" block of a negative graph, written
// exactly as the Python implementation writes it.
type negativeWeightsDoc struct {
	NegativeWeightConfig
	Kind             string            `json:"kind"`
	TotalBlame       float64           `json:"total_blame"`
	TotalFails       int64             `json:"total_fails"`
	TotalFailsResets int64             `json:"total_fails_resets"`
	TotalClear       float64           `json:"total_clear"`
	Reasons          reasonRegistryDoc `json:"reasons"`
}

// pathsDoc is the judged paths: one row per (the node that called the step, the edge it took).
type pathsDoc struct {
	Prev      []int   `json:"prev"`
	Edge      []int   `json:"edge"`
	Seen      []int64 `json:"seen"`
	Correct   []int64 `json:"correct"`
	Incorrect []int64 `json:"incorrect"`
}

type weightsDoc struct {
	WeightConfig
	Kind                  string `json:"kind"`
	TotalTraversals       int64  `json:"total_traversals"`
	TotalTraversalsResets int64  `json:"total_traversals_resets"`
	WindowEvents          []int  `json:"window_events"`

	// the negative form, read from the same block
	ShareScale       float64            `json:"share_scale"`
	BlameScale       float64            `json:"blame_scale"`
	ClearScale       float64            `json:"clear_scale"`
	TotalBlame       float64            `json:"total_blame"`
	TotalFails       int64              `json:"total_fails"`
	TotalFailsResets int64              `json:"total_fails_resets"`
	TotalClear       float64            `json:"total_clear"`
	Reasons          *reasonRegistryDoc `json:"reasons"`

	// neg, when set, is what MarshalJSON writes instead of the count form
	neg *negativeWeightsDoc

	present map[string]bool // which keys the file carried (defaults depend on it, like Python's dict.get)
}

// negative reports whether this block describes a negative graph.
func (w *weightsDoc) negative() bool {
	return w != nil && (w.Kind == "negative" || w.Function == "blame" || w.has("share_scale"))
}

// MarshalJSON writes the count form, or the negative one when it is set.
func (w *weightsDoc) MarshalJSON() ([]byte, error) {
	if w.neg != nil {
		return json.Marshal(w.neg)
	}
	type countForm struct {
		WeightConfig
		Kind                  string `json:"kind"`
		TotalTraversals       int64  `json:"total_traversals"`
		TotalTraversalsResets int64  `json:"total_traversals_resets"`
		WindowEvents          []int  `json:"window_events"`
	}
	return json.Marshal(countForm{WeightConfig: w.WeightConfig, Kind: w.Kind, TotalTraversals: w.TotalTraversals,
		TotalTraversalsResets: w.TotalTraversalsResets, WindowEvents: w.WindowEvents})
}

// UnmarshalJSON also records which keys were present.
func (w *weightsDoc) UnmarshalJSON(b []byte) error {
	type alias weightsDoc
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*w = weightsDoc(a)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err == nil {
		w.present = make(map[string]bool, len(raw))
		for k := range raw {
			w.present[k] = true
		}
	}
	return nil
}

func (w *weightsDoc) has(key string) bool { return w != nil && w.present[key] }

// countWeightKeys are the keys MarshalJSON writes for the count form.
func countWeightKeys() map[string]bool {
	return map[string]bool{
		"function": true, "count_scale": true, "global_scale": true, "window_scale": true, "reward_scale": true,
		"path_scale": true, "window": true, "smoothing": true, "kind": true, "total_traversals": true,
		"total_traversals_resets": true, "window_events": true,
	}
}

// GraphDoc is the JSON layout of a graph (the "graph" block of a model file).
type GraphDoc struct {
	Format                 string      `json:"format"`
	FormatVersion          int         `json:"format_version"`
	Seed                   int64       `json:"seed"`
	Inverted               bool        `json:"inverted"`
	Version                int64       `json:"version"`
	VersionResets          int64       `json:"version_resets"`
	StructureVersion       int64       `json:"structure_version"`
	StructureVersionResets int64       `json:"structure_version_resets"`
	Traversals             int64       `json:"traversals"`
	TraversalsResets       int64       `json:"traversals_resets"`
	Nodes                  nodesDoc    `json:"nodes"`
	Edges                  edgesDoc    `json:"edges"`
	RngState               []any       `json:"rng_state"`
	Weights                *weightsDoc `json:"weights,omitempty"`
	Paths                  *pathsDoc   `json:"paths,omitempty"`

	// the word model's alphabet: what the symbols are, and the words in id order
	Units      string   `json:"units,omitempty"`
	Vocabulary []string `json:"vocabulary,omitempty"`
}

// ToDoc snapshots the graph with dead nodes and edges compacted away (node
// ids remapped: START, END, then alive nodes in id order), exactly like the
// Python implementation.
func (g *Graph) ToDoc() *GraphDoc {
	g.Prepare()
	g.CarryCounters(false) // a saved file always holds a wrapped reading
	remap := make(map[int]int, g.nAliveNodes)
	order := make([]int, 0, g.nAliveNodes)
	for old, ok := range g.Alive {
		if ok {
			remap[old] = len(order)
			order = append(order, old)
		}
	}
	doc := &GraphDoc{Format: graphFormat, FormatVersion: graphFormatVersion, Seed: g.Seed, Inverted: g.Inverted,
		Version: g.Version.Value, VersionResets: g.Version.Resets,
		StructureVersion: g.StructureVersion.Value, StructureVersionResets: g.StructureVersion.Resets,
		Traversals: g.Traversals.Value, TraversalsResets: g.Traversals.Resets}
	n := len(order)
	doc.Nodes = nodesDoc{Labels: make([]string, n), Z: make([]float64, n), A: make([]float64, n), B: make([]float64, n),
		H: make([]float64, n), K: make([]float64, n), Count: make([]int64, n)}
	nodeResets := make([]int64, n)
	anyNodeResets := false
	for i, old := range order {
		doc.Nodes.Labels[i] = g.Labels[old]
		doc.Nodes.B[i] = defaultB
		doc.Nodes.H[i] = defaultH
		doc.Nodes.K[i] = 1.0
		doc.Nodes.Count[i] = g.Count[old]
		if r := g.CountResets[old]; r != 0 {
			nodeResets[i], anyNodeResets = r, true
		}
	}
	if anyNodeResets { // the reset counts ride along only once something has actually wrapped
		doc.Nodes.CountResets = nodeResets
	}
	edgeIndex := make(map[int]int, g.nAliveEdges)
	doc.Edges = edgesDoc{Src: []int{}, Dst: []int{}, W: []float64{}, Count: []int64{}, Reward: []float64{}}
	negative := g.Neg
	if negative != nil {
		doc.Edges.Blame, doc.Edges.Fails = []float64{}, []int64{}
		doc.Edges.Clear, doc.Edges.Reasons = []float64{}, [][][2]float64{}
	}
	for _, old := range order {
		adj := &g.children[old]
		for i, c := range adj.order {
			e := adj.edges[i]
			edgeIndex[e] = len(doc.Edges.Src)
			doc.Edges.Src = append(doc.Edges.Src, remap[old])
			doc.Edges.Dst = append(doc.Edges.Dst, remap[c])
			doc.Edges.W = append(doc.Edges.W, g.EdgeW[e])
			doc.Edges.Count = append(doc.Edges.Count, g.EdgeCount[e])
			doc.Edges.CountResets = append(doc.Edges.CountResets, g.EdgeCountResets[e])
			doc.Edges.Reward = append(doc.Edges.Reward, g.EdgeReward[e])
			if negative != nil {
				doc.Edges.Blame = append(doc.Edges.Blame, negative.Blame[e])
				doc.Edges.Fails = append(doc.Edges.Fails, negative.Fails[e])
				doc.Edges.FailsResets = append(doc.Edges.FailsResets, negative.FailsResets[e])
				doc.Edges.Clear = append(doc.Edges.Clear, negative.Clear[e])
				doc.Edges.Reasons = append(doc.Edges.Reasons, reasonPairs(negative.Reasons[e]))
			}
		}
	}
	doc.RngState = g.rng.State()
	if negative != nil {
		if !anyNonZero(doc.Edges.FailsResets) {
			doc.Edges.FailsResets = nil
		}
		reasonResets := make([]int64, len(negative.ReasonNames))
		for id := range negative.ReasonNames {
			reasonResets[id] = negative.ReasonFailsResets[id]
		}
		if !anyNonZero(reasonResets) {
			reasonResets = nil
		}
		doc.Weights = &weightsDoc{neg: &negativeWeightsDoc{
			NegativeWeightConfig: g.NegativeWeightConfig(), Kind: "negative", TotalBlame: negative.TotalBlame,
			TotalFails: negative.TotalFails.Value, TotalFailsResets: negative.TotalFails.Resets,
			TotalClear: negative.TotalClear,
			Reasons: reasonRegistryDoc{
				Labels:      append([]string{}, negative.ReasonNames...),
				Blame:       append([]float64{}, negative.ReasonBlame...),
				Fails:       append([]int64{}, negative.ReasonFails...),
				FailsResets: reasonResets,
			},
		}}
		return doc
	}
	events := make([]int, 0, g.WindowTraversals())
	for _, e := range g.window[g.windowHead:] {
		if ni, ok := edgeIndex[e]; ok {
			events = append(events, ni)
		}
	}
	if !anyNonZero(doc.Edges.CountResets) {
		doc.Edges.CountResets = nil
	}
	doc.Weights = &weightsDoc{WeightConfig: g.WeightConfig(), Kind: "count-reward",
		TotalTraversals: g.TotalTraversals.Value, TotalTraversalsResets: g.TotalTraversals.Resets, WindowEvents: events,
		// which keys this block carries, exactly as UnmarshalJSON records them for a file: a
		// document handed straight back to GraphFromDoc (a test, a copy) then reads like the
		// file it would have been, instead of like one written before the dual frequency function
		present: countWeightKeys()}
	rows := make([][3]int64, 0, len(g.paths))
	keys := make([][2]int, 0, len(g.paths))
	for key, row := range g.paths {
		ni, ok := edgeIndex[key.Edge]
		pi, known := remap[key.Prev]
		if !ok || !known {
			continue
		}
		keys = append(keys, [2]int{pi, ni})
		rows = append(rows, [3]int64{row.Seen, row.Correct, row.Incorrect})
	}
	order2 := make([]int, len(keys))
	for i := range order2 {
		order2[i] = i
	}
	// the two implementations keep their tables in different orders; the file has one
	sort.Slice(order2, func(i, j int) bool {
		a, b := keys[order2[i]], keys[order2[j]]
		if a[0] != b[0] {
			return a[0] < b[0]
		}
		return a[1] < b[1]
	})
	paths := &pathsDoc{Prev: []int{}, Edge: []int{}, Seen: []int64{}, Correct: []int64{}, Incorrect: []int64{}}
	for _, at := range order2 {
		paths.Prev = append(paths.Prev, keys[at][0])
		paths.Edge = append(paths.Edge, keys[at][1])
		paths.Seen = append(paths.Seen, rows[at][0])
		paths.Correct = append(paths.Correct, rows[at][1])
		paths.Incorrect = append(paths.Incorrect, rows[at][2])
	}
	doc.Paths = paths
	if g.IsWords() {
		doc.Units = WordUnits
		doc.Vocabulary = g.Vocab.List()
	}
	return doc
}

// reasonPairs renders one edge's reasons as Python writes them: [[id, blame], ...] by id.
func reasonPairs(entries []ReasonBlame) [][2]float64 {
	out := make([][2]float64, 0, len(entries))
	for _, entry := range entries {
		out = append(out, [2]float64{float64(entry.ID), entry.Blame})
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

// GraphFromDoc rebuilds a graph from its document.
// withBack gives a graph document the Back sentinel: format 3 as it is, anything
// older upgraded.  Files written before Back existed have Start and End and then
// their real nodes, so the sentinel is inserted at Back and every node id from
// there up shifts by one.  It arrives unvisited and with no edges: a model that
// has never caught itself repeating has nothing to say about where it goes round.
func withBack(d *GraphDoc) {
	if d.FormatVersion >= 3 {
		return
	}
	labels := d.Nodes.Labels
	if len(labels) < Back || (len(labels) > Back && labels[Back] == BackLabel) {
		return
	}
	insertStr := func(v []string, at int, x string) []string {
		return append(v[:at:at], append([]string{x}, v[at:]...)...)
	}
	insertF := func(v []float64, at int, x float64) []float64 {
		if v == nil {
			return nil
		}
		return append(v[:at:at], append([]float64{x}, v[at:]...)...)
	}
	insertI := func(v []int64, at int, x int64) []int64 {
		if v == nil {
			return nil
		}
		return append(v[:at:at], append([]int64{x}, v[at:]...)...)
	}
	// the sentinel takes Start's activation parameters, whatever kind of model wrote the file, and its own
	// fixed state
	at := func(v []float64, fallback float64) float64 {
		if len(v) > Start {
			return v[Start]
		}
		return fallback
	}
	d.Nodes.Labels = insertStr(labels, Back, BackLabel)
	d.Nodes.Z = insertF(d.Nodes.Z, Back, BackZ)
	d.Nodes.A = insertF(d.Nodes.A, Back, at(d.Nodes.A, 0))
	d.Nodes.B = insertF(d.Nodes.B, Back, at(d.Nodes.B, defaultB))
	d.Nodes.H = insertF(d.Nodes.H, Back, at(d.Nodes.H, defaultH))
	d.Nodes.K = insertF(d.Nodes.K, Back, at(d.Nodes.K, 1))
	d.Nodes.Count = insertI(d.Nodes.Count, Back, 0)
	d.Nodes.CountResets = insertI(d.Nodes.CountResets, Back, 0)
	shift := func(i int) int {
		if i >= Back {
			return i + 1
		}
		return i
	}
	for i := range d.Edges.Src {
		d.Edges.Src[i] = shift(d.Edges.Src[i])
	}
	for i := range d.Edges.Dst {
		d.Edges.Dst[i] = shift(d.Edges.Dst[i])
	}
	d.FormatVersion = graphFormatVersion
}

func GraphFromDoc(d *GraphDoc) (*Graph, error) {
	if d.Format != graphFormat {
		return nil, fmt.Errorf("not a %s document", graphFormat)
	}
	withBack(d)
	labels := d.Nodes.Labels
	n := len(labels)
	if n < First || labels[Start] != StartLabel || labels[End] != EndLabel || labels[Back] != BackLabel {
		return nil, fmt.Errorf("graph document is missing the START/END/BACK sentinels")
	}
	for _, arr := range [][]float64{d.Nodes.Z, d.Nodes.A, d.Nodes.B, d.Nodes.H, d.Nodes.K} {
		if arr != nil && len(arr) != n {
			return nil, fmt.Errorf("node arrays have inconsistent lengths")
		}
	}
	if len(d.Nodes.Count) != n {
		return nil, fmt.Errorf("node arrays have inconsistent lengths")
	}
	// files written before the dual frequency function carry no global_scale: they get the old
	// log(1 + count) weight (count_scale 1, no frequency terms); every present key wins over a default
	w := d.Weights
	if w.negative() {
		return negativeGraphFromDoc(d)
	}
	legacy := !w.has("global_scale")
	opts := DefaultGraphOptions()
	if legacy {
		opts.CountScale, opts.GlobalScale, opts.WindowScale = 1.0, 0.0, 0.0
	}
	if w.has("count_scale") {
		opts.CountScale = w.CountScale
	}
	if w.has("reward_scale") {
		opts.RewardScale = w.RewardScale
	}
	if w.has("global_scale") {
		opts.GlobalScale = w.GlobalScale
	}
	if w.has("window_scale") {
		opts.WindowScale = w.WindowScale
	}
	if w.has("path_scale") {
		opts.PathScale = w.PathScale
	}
	if w.has("window") {
		opts.Window = w.Window
		if opts.Window < 1 {
			opts.Window = 1
		}
	}
	if d.Units != "" && d.Units != WordUnits {
		return nil, fmt.Errorf("unknown graph units %q", d.Units)
	}
	opts.Words = d.Units == WordUnits || d.Vocabulary != nil
	g, err := NewGraph(d.Seed, opts)
	if err != nil {
		return nil, err
	}
	if opts.Words {
		vocab, err := VocabularyFrom(d.Vocabulary)
		if err != nil {
			return nil, err
		}
		g.Vocab = vocab
	}
	g.Inverted = d.Inverted
	g.Labels = make([]string, n)
	copy(g.Labels, labels)
	g.labelLen = make([]int, n)
	g.Count = make([]int64, n)
	copy(g.Count, d.Nodes.Count)
	if len(d.Nodes.CountResets) != 0 && len(d.Nodes.CountResets) != n {
		return nil, fmt.Errorf("node arrays have inconsistent lengths")
	}
	g.CountResets = resetsMap(d.Nodes.CountResets)
	g.Alive = make([]bool, n)
	g.children = make([]adjacency, n)
	g.parents = make([]adjacency, n)
	g.index = make(map[string]loc, n*2)
	for nid := 0; nid < n; nid++ {
		g.Alive[nid] = true
		g.labelLen[nid] = runeLen(labels[nid])
		if nid < First {
			continue
		}
		label := []rune(labels[nid])
		if len(label) < Window {
			return nil, fmt.Errorf("node %d label %q is shorter than %d", nid, labels[nid], Window)
		}
		for o := 0; o < len(label)-Overlap; o++ {
			t := string(label[o : o+Window])
			if _, dup := g.index[t]; dup {
				return nil, fmt.Errorf("trigram %q appears in two nodes", t)
			}
			g.index[t] = loc{nid, o}
		}
	}
	g.nAliveNodes = n
	m := len(d.Edges.Src)
	if !(len(d.Edges.Dst) == m && len(d.Edges.W) == m && len(d.Edges.Count) == m) {
		return nil, fmt.Errorf("edge arrays have inconsistent lengths")
	}
	if d.Edges.Reward != nil && len(d.Edges.Reward) != m {
		return nil, fmt.Errorf("edge reward array has an inconsistent length")
	}
	g.EdgeW = make([]float64, m)
	copy(g.EdgeW, d.Edges.W)
	g.EdgeCount = make([]int64, m)
	copy(g.EdgeCount, d.Edges.Count)
	if len(d.Edges.CountResets) != 0 && len(d.Edges.CountResets) != m {
		return nil, fmt.Errorf("edge arrays have inconsistent lengths")
	}
	g.EdgeCountResets = resetsMap(d.Edges.CountResets)
	g.EdgeAlive = make([]bool, m)
	g.EdgeParent = make([]int, m)
	g.EdgeReward = make([]float64, m)
	if d.Edges.Reward != nil {
		copy(g.EdgeReward, d.Edges.Reward)
	}
	g.WindowEdgeCount = make([]int64, m)
	for e := 0; e < m; e++ {
		p, c := d.Edges.Src[e], d.Edges.Dst[e]
		if p < 0 || p >= n || c < 0 || c >= n {
			return nil, fmt.Errorf("invalid edge %d -> %d", p, c)
		}
		if _, dup := g.children[p].get(c); dup {
			return nil, fmt.Errorf("duplicate edge %d -> %d", p, c)
		}
		g.children[p].set(c, e)
		g.parents[c].set(p, e)
		g.EdgeAlive[e] = true
		g.EdgeParent[e] = p
	}
	g.nAliveEdges = m
	if d.RngState != nil {
		if err := g.rng.SetState(d.RngState); err != nil {
			return nil, err
		}
	}
	g.Version = NewCounter(d.Version, d.VersionResets)
	g.StructureVersion = NewCounter(d.StructureVersion, d.StructureVersionResets)
	if d.FormatVersion >= 2 {
		g.Traversals = NewCounter(d.Traversals, d.TraversalsResets)
	} else { // a format 1 file counted into plain integers: their sum bounds every one of them
		var seen int64
		for _, c := range g.Count {
			seen += c
		}
		for _, c := range g.EdgeCount {
			seen += c
		}
		g.Traversals = NewCounter(seen, 0)
	}
	if p := d.Paths; p != nil {
		for i := range p.Prev {
			if i >= len(p.Edge) || p.Edge[i] < 0 || p.Edge[i] >= m || p.Prev[i] < 0 || p.Prev[i] >= n {
				continue
			}
			row := g.pathRow(p.Prev[i], p.Edge[i], true)
			row.Seen = at64(p.Seen, i)
			row.Correct = at64(p.Correct, i)
			row.Incorrect = at64(p.Incorrect, i)
		}
	}
	if d.Weights != nil {
		g.TotalTraversals = NewCounter(d.Weights.TotalTraversals, d.Weights.TotalTraversalsResets)
		for _, e := range d.Weights.WindowEvents {
			if e >= 0 && e < m {
				g.window = append(g.window, e)
				g.WindowEdgeCount[e]++
			}
		}
	}
	g.dirtyAll = true
	g.weightsStructure = invalidStamp
	g.CarryCounters(true) // normalise whatever the file carried, however it was written
	if err := g.checkVocabulary(); err != nil {
		return nil, err
	}
	return g, nil
}

// negativeGraphFromDoc rebuilds a negative graph: the shared structure first,
// then the evidence arrays and the reason registry.
func negativeGraphFromDoc(d *GraphDoc) (*Graph, error) {
	w := d.Weights
	opts := DefaultNegativeOptions()
	if w.has("share_scale") {
		opts.ShareScale = w.ShareScale
	}
	if w.has("blame_scale") {
		opts.BlameScale = w.BlameScale
	}
	if w.has("clear_scale") {
		opts.ClearScale = w.ClearScale
	}
	// the structure, counts and rng come from the shared reader; a copy of the
	// document without the negative marker keeps it on the count path
	plain := *d
	plainWeights := *w
	plainWeights.Kind, plainWeights.Function = "count-reward", "dual-frequency"
	delete(plainWeights.present, "share_scale")
	plain.Weights = &plainWeights
	g, err := GraphFromDoc(&plain)
	if err != nil {
		return nil, err
	}
	n := newNegativeData(opts)
	m := len(g.EdgeW)
	edges := d.Edges
	for name, length := range map[string]int{"blame": len(edges.Blame), "clear": len(edges.Clear), "fails": len(edges.Fails)} {
		if length != 0 && length != m {
			return nil, fmt.Errorf("edge %s array has an inconsistent length", name)
		}
	}
	if len(edges.Reasons) != 0 && len(edges.Reasons) != m {
		return nil, fmt.Errorf("edge reason array has an inconsistent length")
	}
	n.Blame = make([]float64, m)
	n.Fails = make([]int64, m)
	n.Clear = make([]float64, m)
	n.Reasons = make([][]ReasonBlame, m)
	copy(n.Blame, edges.Blame)
	copy(n.Fails, edges.Fails)
	copy(n.Clear, edges.Clear)
	if len(edges.FailsResets) != 0 && len(edges.FailsResets) != m {
		return nil, fmt.Errorf("edge fails_resets array has an inconsistent length")
	}
	n.FailsResets = resetsMap(edges.FailsResets)
	if registry := w.Reasons; registry != nil {
		for _, label := range registry.Labels {
			n.ReasonID(label)
		}
		for id := range n.ReasonNames {
			if id < len(registry.Blame) {
				n.ReasonBlame[id] = registry.Blame[id]
			}
			if id < len(registry.Fails) {
				n.ReasonFails[id] = registry.Fails[id]
			}
			if id < len(registry.FailsResets) {
				setResets(n.ReasonFailsResets, id, registry.FailsResets[id])
			}
		}
	}
	for e, entries := range edges.Reasons {
		for _, pair := range entries {
			id := int(pair[0])
			if id >= 0 && id < len(n.ReasonNames) {
				n.Reasons[e] = append(n.Reasons[e], ReasonBlame{ID: id, Blame: pair[1]})
			}
		}
	}
	n.TotalBlame, n.TotalClear = w.TotalBlame, w.TotalClear
	n.TotalFails = NewCounter(w.TotalFails, w.TotalFailsResets)
	g.Neg = n
	g.TotalTraversals = Counter{}
	g.dirtyAll = true
	g.weightsStructure = invalidStamp
	return g, nil
}

// ModelDoc is the JSON layout of a model file.
type ModelDoc struct {
	Format  string           `json:"format"`
	Version int              `json:"version"`
	SavedAt string           `json:"saved_at"`
	Kind    string           `json:"kind"`
	Meta    map[string]any   `json:"meta"`
	History []map[string]any `json:"history"`

	// the negative network's journal and filter settings (omitted on a count model)
	Log    []LogEntry `json:"log,omitempty"`
	Filter *filterDoc `json:"filter,omitempty"`

	Graph *GraphDoc `json:"graph"`
}

// filterDoc is how strictly a negative network judges.
type filterDoc struct {
	Threshold   float64 `json:"threshold"`
	MinCoverage float64 `json:"min_coverage"`
}

// utcNow is an ISO-8601 UTC timestamp like Python's datetime.isoformat(timespec="seconds").
func utcNow() string { return time.Now().UTC().Format("2006-01-02T15:04:05-07:00") }

// ToDoc snapshots the model.
func (m *Model) ToDoc() *ModelDoc {
	history := make([]map[string]any, len(m.History))
	for i, r := range m.History {
		history[i] = copyMap(r)
	}
	doc := &ModelDoc{Format: ModelFormat, Version: ModelFormatVersion, SavedAt: utcNow(), Kind: m.Kind(),
		Meta: copyMap(m.Meta), History: history, Graph: m.G.ToDoc()}
	if m.IsWords() {
		doc.Format = WordFormat
	}
	if m.IsNegative() {
		doc.Format = NegativeFormat
		doc.Log = append([]LogEntry{}, m.Neg.Log...)
		doc.Filter = &filterDoc{Threshold: m.Neg.Threshold, MinCoverage: m.Neg.MinCoverage}
	}
	return doc
}

// FromDoc rebuilds a model from its document: the count / reward model, or the
// negative network (the format decides, as in Python's load_model).
func FromDoc(d *ModelDoc) (*Model, error) {
	if d.Format != ModelFormat && d.Format != NegativeFormat && d.Format != WordFormat {
		return nil, fmt.Errorf("not a %s, %s or %s model document", ModelFormat, WordFormat, NegativeFormat)
	}
	if d.Version > ModelFormatVersion {
		return nil, fmt.Errorf("unsupported %s model version %d", d.Format, d.Version)
	}
	if d.Graph == nil {
		return nil, fmt.Errorf("model document has no graph")
	}
	g, err := GraphFromDoc(d.Graph)
	if err != nil {
		return nil, err
	}
	m := newModelWithGraph(g)
	m.History = make([]map[string]any, len(d.History))
	for i, r := range d.History {
		m.History[i] = copyMap(r)
	}
	if g.IsNegative() {
		m.makeNegative()
		if len(d.Log) > MaxLogEntries {
			d.Log = d.Log[len(d.Log)-MaxLogEntries:]
		}
		m.Neg.Log = append([]LogEntry{}, d.Log...)
		if d.Filter != nil {
			m.Neg.Threshold, m.Neg.MinCoverage = d.Filter.Threshold, d.Filter.MinCoverage
		}
	}
	for k, v := range d.Meta {
		m.Meta[k] = v
	}
	m.carryMeta() // a file may carry a counter that was never wrapped
	if d.Format == NegativeFormat && !g.IsNegative() {
		return nil, fmt.Errorf("%s document without a negative graph", NegativeFormat)
	}
	if (d.Format == WordFormat) != g.IsWords() {
		if d.Format == WordFormat {
			return nil, fmt.Errorf("%s document without a word graph", WordFormat)
		}
		return nil, fmt.Errorf("a word graph belongs to a %s document, not %s", WordFormat, d.Format)
	}
	return m, nil
}

// MarshalJSON renders a document compactly without HTML escaping (like json.dumps).
func marshalCompact(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := encodeCompact(&buf, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// encodeCompact streams a document as compact JSON without HTML escaping and
// without the encoder's trailing newline, so a model of any size is written
// without ever holding its serialised form in memory.
func encodeCompact(w io.Writer, v any) error {
	tw := &trimNewline{w: w}
	enc := json.NewEncoder(tw)
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

// trimNewline passes everything through but holds back trailing newlines,
// writing them only when more data follows (json.Encoder ends with one).
type trimNewline struct {
	w    io.Writer
	held int // newlines seen at the end of the stream so far
}

func (t *trimNewline) Write(p []byte) (int, error) {
	cut := len(p)
	for cut > 0 && p[cut-1] == '\n' {
		cut--
	}
	if cut > 0 {
		for ; t.held > 0; t.held-- { // they were not trailing after all
			if _, err := t.w.Write([]byte{'\n'}); err != nil {
				return 0, err
			}
		}
		if _, err := t.w.Write(p[:cut]); err != nil {
			return 0, err
		}
	}
	t.held += len(p) - cut
	return len(p), nil
}

// Save writes the model as JSON (gzip when path ends with .gz) atomically.
func (m *Model) Save(path string) error {
	return writeDocAtomic(path, m.ToDoc(), strings.HasSuffix(path, ".gz"))
}

// writeDocAtomic streams a document into a temporary file and renames it over
// path.  Nothing bigger than the write buffer is held in memory.
func writeDocAtomic(path string, doc any, useGzip bool) error {
	return writeAtomicWith(path, func(w io.Writer) error { return encodeCompact(w, doc) }, useGzip)
}

func writeBytesAtomic(path string, data []byte, useGzip bool) error {
	return writeAtomicWith(path, func(w io.Writer) error { _, err := w.Write(data); return err }, useGzip)
}

func writeAtomicWith(path string, write func(io.Writer) error, useGzip bool) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*.part")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpName) }
	failed := func(err error) error {
		tmp.Close()
		cleanup()
		return err
	}
	buf := bufio.NewWriterSize(tmp, 1<<20)
	var out io.Writer = buf
	var zw *gzip.Writer
	if useGzip {
		zw = gzip.NewWriter(buf)
		out = zw
	}
	if err := write(out); err != nil {
		return failed(err)
	}
	if zw != nil {
		if err := zw.Close(); err != nil {
			return failed(err)
		}
	}
	if err := buf.Flush(); err != nil {
		return failed(err)
	}
	if err := tmp.Sync(); err != nil {
		return failed(err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		cleanup()
		return err
	}
	return nil
}

// ReadJSONFile decodes a JSON document straight from the file (gunzipping it
// when it carries the gzip magic), so a huge model is never held twice.
func ReadJSONFile(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	magic, err := br.Peek(2)
	var r io.Reader = br
	if err == nil && magic[0] == 0x1f && magic[1] == 0x8b {
		zr, zerr := gzip.NewReader(br)
		if zerr != nil {
			return zerr
		}
		defer zr.Close()
		r = zr
	}
	return json.NewDecoder(r).Decode(v)
}

// Load reads a model written by Save (or by the Python implementation).
func Load(path string) (*Model, error) {
	var doc ModelDoc
	if err := ReadJSONFile(path, &doc); err != nil {
		return nil, err
	}
	return FromDoc(&doc)
}

func copyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// at64 reads one counter of a paths block, tolerating a short column.
func at64(values []int64, i int) int64 {
	if i < len(values) {
		return values[i]
	}
	return 0
}
