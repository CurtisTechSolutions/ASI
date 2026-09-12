package radixnet

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ModelFormat is the file format shared with the Python count model.
const (
	ModelFormat        = "radixnet-count"
	ModelFormatVersion = 1
)

type nodesDoc struct {
	Labels []string  `json:"labels"`
	Z      []float64 `json:"z"`
	A      []float64 `json:"a"`
	B      []float64 `json:"b"`
	H      []float64 `json:"h"`
	K      []float64 `json:"k"`
	Count  []int64   `json:"count"`
}

type edgesDoc struct {
	Src    []int     `json:"src"`
	Dst    []int     `json:"dst"`
	W      []float64 `json:"w"`
	Count  []int64   `json:"count"`
	Reward []float64 `json:"reward"`
}

type weightsDoc struct {
	WeightConfig
	Kind            string `json:"kind"`
	TotalTraversals int64  `json:"total_traversals"`
	WindowEvents    []int  `json:"window_events"`

	present map[string]bool // which keys the file carried (defaults depend on it, like Python's dict.get)
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

// GraphDoc is the JSON layout of a graph (the "graph" block of a model file).
type GraphDoc struct {
	Format           string      `json:"format"`
	FormatVersion    int         `json:"format_version"`
	Seed             int64       `json:"seed"`
	Inverted         bool        `json:"inverted"`
	Version          int         `json:"version"`
	StructureVersion int         `json:"structure_version"`
	Nodes            nodesDoc    `json:"nodes"`
	Edges            edgesDoc    `json:"edges"`
	RngState         []any       `json:"rng_state"`
	Weights          *weightsDoc `json:"weights,omitempty"`
}

// ToDoc snapshots the graph with dead nodes and edges compacted away (node
// ids remapped: START, END, then alive nodes in id order), exactly like the
// Python implementation.
func (g *Graph) ToDoc() *GraphDoc {
	g.Prepare()
	remap := make(map[int]int, g.nAliveNodes)
	order := make([]int, 0, g.nAliveNodes)
	for old, ok := range g.Alive {
		if ok {
			remap[old] = len(order)
			order = append(order, old)
		}
	}
	doc := &GraphDoc{Format: graphFormat, FormatVersion: graphFormatVersion, Seed: g.Seed, Inverted: g.Inverted,
		Version: g.Version, StructureVersion: g.StructureVersion}
	n := len(order)
	doc.Nodes = nodesDoc{Labels: make([]string, n), Z: make([]float64, n), A: make([]float64, n), B: make([]float64, n),
		H: make([]float64, n), K: make([]float64, n), Count: make([]int64, n)}
	for i, old := range order {
		doc.Nodes.Labels[i] = g.Labels[old]
		doc.Nodes.B[i] = defaultB
		doc.Nodes.H[i] = defaultH
		doc.Nodes.K[i] = 1.0
		doc.Nodes.Count[i] = g.Count[old]
	}
	edgeIndex := make(map[int]int, g.nAliveEdges)
	doc.Edges = edgesDoc{Src: []int{}, Dst: []int{}, W: []float64{}, Count: []int64{}, Reward: []float64{}}
	for _, old := range order {
		adj := &g.children[old]
		for _, c := range adj.order {
			e := adj.edge[c]
			edgeIndex[e] = len(doc.Edges.Src)
			doc.Edges.Src = append(doc.Edges.Src, remap[old])
			doc.Edges.Dst = append(doc.Edges.Dst, remap[c])
			doc.Edges.W = append(doc.Edges.W, g.EdgeW[e])
			doc.Edges.Count = append(doc.Edges.Count, g.EdgeCount[e])
			doc.Edges.Reward = append(doc.Edges.Reward, g.EdgeReward[e])
		}
	}
	doc.RngState = g.rng.State()
	events := make([]int, 0, g.WindowTraversals())
	for _, e := range g.window[g.windowHead:] {
		if ni, ok := edgeIndex[e]; ok {
			events = append(events, ni)
		}
	}
	doc.Weights = &weightsDoc{WeightConfig: g.WeightConfig(), Kind: "count-reward", TotalTraversals: g.TotalTraversals, WindowEvents: events}
	return doc
}

// GraphFromDoc rebuilds a graph from its document.
func GraphFromDoc(d *GraphDoc) (*Graph, error) {
	if d.Format != graphFormat {
		return nil, fmt.Errorf("not a %s document", graphFormat)
	}
	labels := d.Nodes.Labels
	n := len(labels)
	if n < 2 || labels[Start] != StartLabel || labels[End] != EndLabel {
		return nil, fmt.Errorf("graph document is missing the START/END sentinels")
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
	if w.has("window") {
		opts.Window = w.Window
		if opts.Window < 1 {
			opts.Window = 1
		}
	}
	g, err := NewGraph(d.Seed, opts)
	if err != nil {
		return nil, err
	}
	g.Inverted = d.Inverted
	g.Labels = make([]string, n)
	copy(g.Labels, labels)
	g.labelLen = make([]int, n)
	g.Count = make([]int64, n)
	copy(g.Count, d.Nodes.Count)
	g.Alive = make([]bool, n)
	g.children = make([]adjacency, n)
	g.parents = make([]map[int]int, n)
	g.index = make(map[string]loc, n*2)
	for nid := 0; nid < n; nid++ {
		g.Alive[nid] = true
		g.labelLen[nid] = runeLen(labels[nid])
		if nid < 2 {
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
		if g.parents[c] == nil {
			g.parents[c] = make(map[int]int, 2)
		}
		g.parents[c][p] = e
		g.EdgeAlive[e] = true
		g.EdgeParent[e] = p
	}
	g.nAliveEdges = m
	if d.RngState != nil {
		if err := g.rng.SetState(d.RngState); err != nil {
			return nil, err
		}
	}
	g.Version = d.Version
	g.StructureVersion = d.StructureVersion
	if d.Weights != nil {
		g.TotalTraversals = d.Weights.TotalTraversals
		for _, e := range d.Weights.WindowEvents {
			if e >= 0 && e < m {
				g.window = append(g.window, e)
				g.WindowEdgeCount[e]++
			}
		}
	}
	g.dirtyAll = true
	g.weightsStructure = -1
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
	Graph   *GraphDoc        `json:"graph"`
}

// utcNow is an ISO-8601 UTC timestamp like Python's datetime.isoformat(timespec="seconds").
func utcNow() string { return time.Now().UTC().Format("2006-01-02T15:04:05-07:00") }

// ToDoc snapshots the model.
func (m *Model) ToDoc() *ModelDoc {
	history := make([]map[string]any, len(m.History))
	for i, r := range m.History {
		history[i] = copyMap(r)
	}
	return &ModelDoc{Format: ModelFormat, Version: ModelFormatVersion, SavedAt: utcNow(), Kind: "count",
		Meta: copyMap(m.Meta), History: history, Graph: m.G.ToDoc()}
}

// FromDoc rebuilds a model from its document.
func FromDoc(d *ModelDoc) (*Model, error) {
	if d.Format != ModelFormat {
		return nil, fmt.Errorf("not a %s model document", ModelFormat)
	}
	if d.Version > ModelFormatVersion {
		return nil, fmt.Errorf("unsupported %s model version %d", ModelFormat, d.Version)
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
	for k, v := range d.Meta {
		m.Meta[k] = v
	}
	return m, nil
}

// MarshalJSON renders a document compactly without HTML escaping (like json.dumps).
func marshalCompact(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// Save writes the model as JSON (gzip when path ends with .gz) atomically.
func (m *Model) Save(path string) error {
	payload, err := marshalCompact(m.ToDoc())
	if err != nil {
		return err
	}
	return writeBytesAtomic(path, payload, strings.HasSuffix(path, ".gz"))
}

func writeBytesAtomic(path string, data []byte, useGzip bool) error {
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
	if useGzip {
		zw := gzip.NewWriter(tmp)
		if _, err := zw.Write(data); err != nil {
			tmp.Close()
			cleanup()
			return err
		}
		if err := zw.Close(); err != nil {
			tmp.Close()
			cleanup()
			return err
		}
	} else if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		cleanup()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		cleanup()
		return err
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

// ReadJSONFile loads a JSON document, gunzipping it when it carries the gzip magic.
func ReadJSONFile(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(raw) >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
		zr, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			return err
		}
		raw, err = io.ReadAll(zr)
		if err != nil {
			return err
		}
	}
	return json.Unmarshal(raw, v)
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
