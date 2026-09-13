package radixnet

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// NegativeFormat is the model file format of the negative network, shared with
// the Python implementation (radixnet/negative.py).
const NegativeFormat = "radixnet-negative"

// UnspecifiedReason is recorded when a failure arrives without one (the tutor always gives one).
const UnspecifiedReason = "unspecified"

// MaxEdgeReasons is how many reasons an edge keeps; the smallest is dropped
// when a ninth appears (the graph-level totals keep every one of them).
const MaxEdgeReasons = 8

// MaxLogEntries is how many failures the journal keeps (the newest win).
const MaxLogEntries = 200

const maxLogTextChars = 160

// NegativeOptions are the scales of the blame weight function.
type NegativeOptions struct {
	ShareScale float64
	BlameScale float64
	ClearScale float64
}

// DefaultNegativeOptions mirror the Python defaults: the edge's share of the
// failure mass leaving its parent, with one cleared traversal cancelling one
// ordinary failure.
func DefaultNegativeOptions() NegativeOptions {
	return NegativeOptions{ShareScale: 1, BlameScale: 0, ClearScale: 1}
}

// ReasonBlame is one reason's contribution to an edge's blame.
type ReasonBlame struct {
	ID    int
	Blame float64
}

// NegativeData is everything a negative graph keeps that a count graph does
// not: the per-edge evidence and the registry of the tutor's reasons.  It is
// nil on a count / reward graph, so the count model pays nothing for it.
type NegativeData struct {
	// per edge, parallel to Graph.EdgeW
	Blame   []float64
	Fails   []int64
	Clear   []float64
	Reasons [][]ReasonBlame

	// the reason registry
	ReasonNames []string
	ReasonBlame []float64
	ReasonFails []int64
	reasonIDs   map[string]int

	TotalBlame float64
	TotalFails int64
	TotalClear float64

	ShareScale float64
	BlameScale float64
	ClearScale float64
}

func newNegativeData(o NegativeOptions) *NegativeData {
	return &NegativeData{
		reasonIDs: map[string]int{}, ShareScale: o.ShareScale, BlameScale: o.BlameScale, ClearScale: o.ClearScale,
	}
}

// appendEdge grows the per-edge arrays for one new edge.
func (n *NegativeData) appendEdge() {
	n.Blame = append(n.Blame, 0)
	n.Fails = append(n.Fails, 0)
	n.Clear = append(n.Clear, 0)
	n.Reasons = append(n.Reasons, nil)
}

// CleanReason normalises a reason tag: one trimmed lower-case line, never empty.
func CleanReason(reason string) string {
	text := strings.ToLower(strings.Join(strings.Fields(reason), " "))
	if text == "" {
		return UnspecifiedReason
	}
	return truncateRunes(text, 60)
}

// ReasonID is the id of a reason tag, registering it on first sight.
func (n *NegativeData) ReasonID(reason string) int {
	label := CleanReason(reason)
	if id, ok := n.reasonIDs[label]; ok {
		return id
	}
	id := len(n.ReasonNames)
	n.reasonIDs[label] = id
	n.ReasonNames = append(n.ReasonNames, label)
	n.ReasonBlame = append(n.ReasonBlame, 0)
	n.ReasonFails = append(n.ReasonFails, 0)
	return id
}

// ReasonLabel is the tag of a reason id ("unspecified" for an unknown id).
func (n *NegativeData) ReasonLabel(id int) string {
	if id >= 0 && id < len(n.ReasonNames) {
		return n.ReasonNames[id]
	}
	return UnspecifiedReason
}

// lookupReason returns the id of an already registered tag.
func (n *NegativeData) lookupReason(reason string) (int, bool) {
	id, ok := n.reasonIDs[CleanReason(reason)]
	return id, ok
}

// NewNegativeGraph creates an empty negative graph: the same self-compressing
// cyclic graph, with the blame arrays and the reason registry attached.
func NewNegativeGraph(seed int64, o NegativeOptions) (*Graph, error) {
	g, err := NewGraph(seed, DefaultGraphOptions())
	if err != nil {
		return nil, err
	}
	g.Neg = newNegativeData(o)
	return g, nil
}

// IsNegative reports whether this graph keeps blame instead of counts and rewards.
func (g *Graph) IsNegative() bool { return g != nil && g.Neg != nil }

// Evidence is the net evidence against an edge: max(0, blame - clear_scale * clear).
//
// Blame and clearing cancel: an edge the tutor's passes cross as often as its
// failures do is not evidence of anything, however busy it is - which is what
// keeps a common fragment out of the verdict.
func (g *Graph) Evidence(e int) float64 {
	n := g.Neg
	if n == nil || e < 0 || e >= len(n.Blame) || !g.EdgeAlive[e] {
		return 0
	}
	return math.Max(0, n.Blame[e]-n.ClearScale*n.Clear[e])
}

// NegativeEdgeWeight is the blame weight of one edge from its net evidence:
//
//	R_bad  = (net + s) / (net leaving the parent + s * degree)
//	weight = share_scale * log(R_bad) + blame_scale * log(1 + net)
func (g *Graph) NegativeEdgeWeight(evidence, parentEvidence float64, degree int) float64 {
	n := g.Neg
	evidence = math.Max(0, evidence)
	if degree < 1 {
		degree = 1
	}
	d := float64(degree)
	rBad := (evidence + Smoothing) / (math.Max(0, parentEvidence) + Smoothing*d)
	return n.ShareScale*math.Log(rBad) + n.BlameScale*math.Log1p(evidence)
}

// recomputeNegativeRow writes the blame weight to every edge leaving p.
func (g *Graph) recomputeNegativeRow(p int) {
	adj := &g.children[p]
	if adj.size() == 0 || !g.Alive[p] {
		return
	}
	degree := adj.size()
	total := 0.0
	net := make([]float64, len(adj.order))
	for i := range adj.order {
		net[i] = g.Evidence(adj.edges[i])
		total += net[i]
	}
	for i := range adj.order {
		g.EdgeW[adj.edges[i]] = g.NegativeEdgeWeight(net[i], total, degree)
	}
}

// RecordFailure blames every listed alive edge by severity for reason;
// returns how many were blamed.
func (g *Graph) RecordFailure(edges []int, severity float64, reason string) int {
	n := g.Neg
	if n == nil {
		return 0
	}
	amount := math.Abs(severity)
	if math.IsNaN(amount) || math.IsInf(amount, 0) {
		return 0
	}
	id := n.ReasonID(reason)
	touched := 0
	for _, e := range edges {
		if e < 0 || e >= len(n.Blame) || !g.EdgeAlive[e] {
			continue
		}
		n.Blame[e] += amount
		n.Fails[e]++
		if amount > 0 {
			n.Reasons[e] = addReasonBlame(n.Reasons[e], id, amount)
		}
		g.dirty[g.EdgeParent[e]] = struct{}{}
		touched++
	}
	if touched > 0 {
		n.TotalBlame += amount * float64(touched)
		n.TotalFails += int64(touched)
		n.ReasonBlame[id] += amount * float64(touched)
		n.ReasonFails[id]++
		g.Version++
	}
	return touched
}

// addReasonBlame adds amount to one reason of an edge, dropping the weakest
// entry when the list is full (the graph totals keep it).
func addReasonBlame(list []ReasonBlame, id int, amount float64) []ReasonBlame {
	for i := range list {
		if list[i].ID == id {
			list[i].Blame += amount
			return list
		}
	}
	list = append(list, ReasonBlame{ID: id, Blame: amount})
	if len(list) > MaxEdgeReasons {
		// the weakest of the *others* goes: a reason just added carries the least
		// blame of all by construction and would otherwise evict itself
		weakest := -1
		for i := range list[:len(list)-1] {
			if weakest < 0 || list[i].Blame < list[weakest].Blame {
				weakest = i
			}
		}
		list = append(list[:weakest], list[weakest+1:]...)
	}
	return list
}

// RecordClear credits every listed alive edge with weight of cleared text (the
// tutor passed it); returns how many were credited.
func (g *Graph) RecordClear(edges []int, weight float64) int {
	n := g.Neg
	if n == nil {
		return 0
	}
	amount := math.Abs(weight)
	if math.IsNaN(amount) || math.IsInf(amount, 0) {
		return 0
	}
	touched := 0
	for _, e := range edges {
		if e < 0 || e >= len(n.Clear) || !g.EdgeAlive[e] {
			continue
		}
		n.Clear[e] += amount
		g.dirty[g.EdgeParent[e]] = struct{}{}
		touched++
	}
	if touched > 0 {
		n.TotalClear += amount * float64(touched)
		g.Version++
	}
	return touched
}

// blocksMerge reports whether a unary chain's edge carries evidence and must
// therefore survive compression: a step *inside* a node has no edge to carry
// blame, so a correction that blamed a single transition would be folded away
// and forgotten.  Everything the tutor never ruled on still merges.
func (g *Graph) blocksMerge(e int) bool {
	n := g.Neg
	if n == nil || e < 0 || e >= len(n.Blame) {
		return false
	}
	return n.Blame[e] > 0 || n.Clear[e] > 0
}

// ForgetResult reports what a Forget call removed.
type ForgetResult struct {
	Reason       string  `json:"reason"`
	Edges        int     `json:"edges"`
	BlameRemoved float64 `json:"blame_removed"`
}

// Forget scales blame down (factor 0 = forget it entirely, 0.5 = halve it),
// for one reason or for all of them - the tutor can be wrong too.
func (g *Graph) Forget(reason string, factor float64) (*ForgetResult, error) {
	n := g.Neg
	if n == nil {
		return nil, fmt.Errorf("not a negative graph")
	}
	if !(factor >= 0 && factor < 1) {
		return nil, fmt.Errorf("factor must lie in [0, 1), got %v", factor)
	}
	all := reason == ""
	out := &ForgetResult{Reason: "*"}
	id := -1
	if !all {
		out.Reason = CleanReason(reason)
		known, ok := n.lookupReason(reason)
		if !ok {
			return out, nil
		}
		id = known
	}
	for e, alive := range g.EdgeAlive {
		if !alive {
			continue
		}
		if all {
			drop := n.Blame[e] * (1 - factor)
			if drop == 0 {
				continue
			}
			n.Blame[e] -= drop
			if factor == 0 {
				n.Reasons[e] = nil
			} else {
				for i := range n.Reasons[e] {
					n.Reasons[e][i].Blame *= factor
				}
			}
			out.BlameRemoved += drop
			out.Edges++
			g.dirty[g.EdgeParent[e]] = struct{}{}
			continue
		}
		for i, entry := range n.Reasons[e] {
			if entry.ID != id {
				continue
			}
			drop := entry.Blame * (1 - factor)
			n.Blame[e] = math.Max(0, n.Blame[e]-drop)
			if factor == 0 {
				n.Reasons[e] = append(n.Reasons[e][:i], n.Reasons[e][i+1:]...)
			} else {
				n.Reasons[e][i].Blame *= factor
			}
			out.BlameRemoved += drop
			out.Edges++
			g.dirty[g.EdgeParent[e]] = struct{}{}
			break
		}
	}
	if all {
		for i := range n.ReasonBlame {
			n.ReasonBlame[i] *= factor
		}
		n.TotalBlame *= factor
	} else {
		n.TotalBlame = math.Max(0, n.TotalBlame-n.ReasonBlame[id]*(1-factor))
		n.ReasonBlame[id] *= factor
	}
	if out.Edges > 0 {
		g.Version++
		g.flushWeights()
	}
	return out, nil
}

// invertNegative swaps blame and clearing: what the tutor rejected becomes
// what it accepted, and back.  The inverse of a network of failures is a
// network of the text that passed.
func (g *Graph) invertNegative() {
	n := g.Neg
	n.Blame, n.Clear = n.Clear, n.Blame
	n.TotalBlame, n.TotalClear = n.TotalClear, n.TotalBlame
	g.Inverted = !g.Inverted
	g.dirtyAll = true
	g.flushWeights()
}

// ConfigureNegative changes the blame weight function's scales and recomputes every weight.
func (g *Graph) ConfigureNegative(opts map[string]float64) error {
	n := g.Neg
	if n == nil {
		return fmt.Errorf("not a negative graph")
	}
	for name, value := range opts {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s must be a finite number, got %v", name, value)
		}
		switch name {
		case "share_scale":
			n.ShareScale = value
		case "blame_scale":
			n.BlameScale = value
		case "clear_scale":
			n.ClearScale = value
		default:
			return fmt.Errorf("unknown weight option %q", name)
		}
	}
	g.dirtyAll = true
	g.flushWeights()
	return nil
}

// NegativeWeightConfig describes the blame weight function (the "weights" block of a negative model file).
type NegativeWeightConfig struct {
	Function   string  `json:"function"`
	ShareScale float64 `json:"share_scale"`
	BlameScale float64 `json:"blame_scale"`
	ClearScale float64 `json:"clear_scale"`
	Smoothing  float64 `json:"smoothing"`
}

// NegativeWeightConfig returns the current blame weight settings.
func (g *Graph) NegativeWeightConfig() NegativeWeightConfig {
	n := g.Neg
	return NegativeWeightConfig{
		Function: "blame", ShareScale: n.ShareScale, BlameScale: n.BlameScale, ClearScale: n.ClearScale,
		Smoothing: Smoothing,
	}
}

// ReasonRow is one row of the reason table.
type ReasonRow struct {
	Reason string  `json:"reason"`
	Blame  float64 `json:"blame"`
	Fails  int64   `json:"fails"`
	Edges  int     `json:"edges"`
	Share  float64 `json:"share"`
}

// ReasonTable is everything the tutor has blamed, heaviest blame first.
func (g *Graph) ReasonTable() []ReasonRow {
	n := g.Neg
	if n == nil {
		return nil
	}
	edges := make([]int, len(n.ReasonNames))
	for e, alive := range g.EdgeAlive {
		if !alive {
			continue
		}
		for _, entry := range n.Reasons[e] {
			if entry.ID >= 0 && entry.ID < len(edges) {
				edges[entry.ID]++
			}
		}
	}
	total := 0.0
	for _, blame := range n.ReasonBlame {
		total += blame
	}
	if total == 0 {
		total = 1
	}
	rows := make([]ReasonRow, 0, len(n.ReasonNames))
	for id, label := range n.ReasonNames {
		rows = append(rows, ReasonRow{
			Reason: label, Blame: n.ReasonBlame[id], Fails: n.ReasonFails[id], Edges: edges[id],
			Share: n.ReasonBlame[id] / total,
		})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Blame != rows[j].Blame {
			return rows[i].Blame > rows[j].Blame
		}
		return rows[i].Reason < rows[j].Reason
	})
	return rows
}

// EdgeReasons lists one edge's reasons, heaviest first.
func (g *Graph) EdgeReasons(e int) []ReasonRow {
	n := g.Neg
	if n == nil || e < 0 || e >= len(n.Reasons) {
		return nil
	}
	rows := make([]ReasonRow, 0, len(n.Reasons[e]))
	for _, entry := range n.Reasons[e] {
		rows = append(rows, ReasonRow{Reason: n.ReasonLabel(entry.ID), Blame: entry.Blame})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Blame != rows[j].Blame {
			return rows[i].Blame > rows[j].Blame
		}
		return rows[i].Reason < rows[j].Reason
	})
	return rows
}

// ---------------------------------------------------------------------------
// the model: blaming, clearing, corrections and the verdict
// ---------------------------------------------------------------------------

// LogEntry is one line of the negative network's journal: what was blamed, for
// what reason, and in whose words.
type LogEntry struct {
	At       string  `json:"at"`
	Text     string  `json:"text"`
	Reason   string  `json:"reason"`
	Severity float64 `json:"severity"`
	Source   string  `json:"source"`
	Note     string  `json:"note"`
}

// Negative is the model-level state of a negative network: the journal and how
// strictly it judges (nil on a count model).
type Negative struct {
	Log         []LogEntry
	Threshold   float64
	MinCoverage float64
}

// DefaultThreshold rejects a text that carries, on average, a whole failure's
// worth of net evidence per transition.
const DefaultThreshold = 1.0

// DefaultMinCoverage is the share of a text's transitions that must be known
// failures before any rule may reject it.
const DefaultMinCoverage = 0.5

// NewNegativeModel creates an empty negative network: the same machinery as
// the count model, but every node and edge it ever grows is there because
// something went wrong.
func NewNegativeModel(seed int64, o NegativeOptions) (*Model, error) {
	g, err := NewNegativeGraph(seed, o)
	if err != nil {
		return nil, err
	}
	m := newModelWithGraph(g)
	m.makeNegative()
	return m, nil
}

// makeNegative attaches the model-level negative state and its extra counters.
func (m *Model) makeNegative() {
	m.Neg = &Negative{Threshold: DefaultThreshold, MinCoverage: DefaultMinCoverage}
	for key, value := range map[string]any{
		"failures_total": int64(0), "blame_total": 0.0, "cleared_total": int64(0),
		"judgements": int64(0), "rejected": int64(0),
	} {
		if _, seen := m.Meta[key]; !seen {
			m.Meta[key] = value
		}
	}
	if _, seen := m.Meta["sources"]; !seen {
		m.Meta["sources"] = map[string]any{}
	}
}

// IsNegative reports whether this model is the negative network.
func (m *Model) IsNegative() bool { return m != nil && m.Neg != nil && m.G.IsNegative() }

// requireNegative guards the operations that only a negative network has.
func (m *Model) requireNegative(what string) error {
	if !m.IsNegative() {
		return fmt.Errorf("%s needs the negative network (this is the %s model)", what, m.Kind())
	}
	return nil
}

// sources counts the tutors a failure came from.
func (m *Model) addSource(source string) {
	if source == "" {
		return
	}
	table, _ := m.Meta["sources"].(map[string]any)
	if table == nil {
		table = map[string]any{}
		m.Meta["sources"] = table
	}
	table[source] = toFloat(table[source]) + 1
}

// note appends one failure to the journal (the oldest entries drop out).
func (m *Model) note(text, reason string, severity float64, source, comment string) {
	entry := LogEntry{
		At: utcNow(), Text: truncateRunes(text, maxLogTextChars), Reason: reason, Severity: severity,
		Source: source, Note: truncateRunes(strings.Join(strings.Fields(comment), " "), 300),
	}
	m.Neg.Log = append(m.Neg.Log, entry)
	if len(m.Neg.Log) > MaxLogEntries {
		m.Neg.Log = append([]LogEntry{}, m.Neg.Log[len(m.Neg.Log)-MaxLogEntries:]...)
	}
}

// Recent returns the newest journal entries, newest first.
func (m *Model) Recent(limit int) []LogEntry {
	if m.Neg == nil || limit <= 0 {
		return []LogEntry{}
	}
	log := m.Neg.Log
	if limit > len(log) {
		limit = len(log)
	}
	out := make([]LogEntry, 0, limit)
	for i := len(log) - 1; i >= len(log)-limit; i-- {
		out = append(out, log[i])
	}
	return out
}

// Reasons is everything the tutor has blamed, heaviest first.
func (m *Model) Reasons() []ReasonRow {
	if !m.IsNegative() {
		return nil
	}
	return m.G.ReasonTable()
}

// BlameOptions describe one lesson: why a text failed, how badly, and who says so.
type BlameOptions struct {
	Reason   string
	Severity float64 // 0 = one ordinary failure
	Source   string
	Note     string
	Epochs   int // 0 = one pass
	// NoCompress leaves the structure uncompressed after the pass.
	NoCompress bool
	Progress   func(map[string]any)
	Stop       func() bool
}

func (o BlameOptions) severity() float64 {
	if o.Severity == 0 {
		return 1
	}
	return math.Abs(o.Severity)
}

func (o BlameOptions) epochs() int {
	if o.Epochs <= 0 {
		return 1
	}
	return o.Epochs
}

// register registers encoded failures structurally and returns their
// transitions.  Two passes: a text registered later can split a node an
// earlier one pointed at, so the transitions are re-derived once the structure
// has settled (the second pass never splits again).
func (m *Model) register(grams [][]string) ([][]Transition, error) {
	g := m.G
	for _, gram := range grams {
		if len(gram) == 0 {
			continue
		}
		if _, err := g.ObserveSequence(gram, false); err != nil {
			return nil, err
		}
	}
	out := make([][]Transition, len(grams))
	for i, gram := range grams {
		if len(gram) == 0 {
			continue
		}
		transitions, err := g.ObserveSequence(gram, false)
		if err != nil {
			return nil, err
		}
		out[i] = transitions
	}
	return out, nil
}

// sharedEdges are the transitions a text shares with the failure structure
// (nothing is created).  Cleared text is ordinary, correct text: it will not
// walk a graph of failures end to end, so every crossing it does share counts
// and the rest is simply skipped.
func (m *Model) sharedEdges(text string) []int {
	out := []int{}
	for _, c := range m.Crossings(text) {
		if c.Edge >= 0 {
			out = append(out, c.Edge)
		}
	}
	return out
}

// Blame teaches the negative network a failure: it registers every text's path
// and blames its edges by Severity for Reason.  This is the only operation
// that adds structure to the negative network.
func (m *Model) Blame(texts []string, o BlameOptions) ([]map[string]any, error) {
	return m.blamePass(texts, o, true, 1)
}

// Clear credits the edges the texts share with known failures: the tutor
// passed them, so blame and clearing cancel there.  Nothing is created.
func (m *Model) Clear(texts []string, weight float64, epochs int) ([]map[string]any, error) {
	if weight == 0 {
		weight = 1
	}
	return m.blamePass(texts, BlameOptions{Epochs: epochs}, false, math.Abs(weight))
}

// blamePass is one routine for both halves: blame (which registers and creates
// structure) and clearing (which only credits what is already there).
func (m *Model) blamePass(texts []string, o BlameOptions, blame bool, weight float64) ([]map[string]any, error) {
	if err := m.requireNegative("blaming"); err != nil {
		return nil, err
	}
	kept, skippedShort := cleanTexts(texts)
	g := m.G
	amount := weight
	reason := CleanReason(o.Reason)
	if blame {
		amount = o.severity()
	}
	grams := m.encodeAll(kept)
	if blame {
		m.metaAddInt("trained_texts", int64(len(kept)))
		chars := int64(0)
		for _, text := range kept {
			chars += int64(runeLen(text))
			m.note(text, reason, amount, o.Source, o.Note)
		}
		m.metaAddInt("trained_chars", chars)
		for range kept {
			m.addSource(o.Source)
		}
		if _, err := m.register(grams); err != nil { // settle the structure once
			return nil, err
		}
	}
	pendingMerges := 0
	if blame && !o.NoCompress {
		pendingMerges = g.Compress()
	}
	records := []map[string]any{}
	for epoch := 0; epoch < o.epochs(); epoch++ {
		started := time.Now()
		var perText [][]Transition
		if blame {
			walked, err := m.register(grams)
			if err != nil {
				return nil, err
			}
			perText = walked
		}
		touched, matched, transitions := 0, 0, 0
		if blame {
			for _, walk := range perText {
				if len(walk) == 0 {
					continue
				}
				matched++
				transitions += len(walk)
				touched += g.RecordFailure(edgesOf(walk), amount, reason)
			}
		} else {
			for _, text := range kept {
				edges := m.sharedEdges(text)
				if len(edges) == 0 {
					continue
				}
				matched++
				transitions += len(edges)
				touched += g.RecordClear(edges, amount)
			}
		}
		if blame {
			m.metaAddInt("failures_total", int64(matched))
			m.metaAddFloat("blame_total", amount*float64(touched))
		} else {
			m.metaAddInt("cleared_total", int64(matched))
		}
		g.Prepare()
		loss := 0.0
		if blame {
			flat := []Transition{}
			for _, walk := range perText {
				flat = append(flat, walk...)
			}
			loss = meanCost(g, flat)
		} else {
			loss = m.meanCostOfTexts(kept)
		}
		merges := pendingMerges
		if blame && !o.NoCompress {
			merges += g.Compress()
		}
		pendingMerges = 0
		m.metaAddInt("epochs_total", 1)
		record := map[string]any{
			"epoch": m.metaInt("epochs_total"), "loss": loss,
			"perplexity":        math.Exp(math.Min(loss, maxLogPerplexity)),
			"nodes":             g.NumNodes(),
			"edges":             g.NumEdges(),
			"trigrams":          g.NumTrigrams(),
			"compression_ratio": g.CompressionRatio(),
			"merges":            merges,
			"transitions":       transitions,
			"seconds":           time.Since(started).Seconds(),
			"skipped_short":     skippedShort,
			"phase":             phaseOf(blame),
			"texts":             len(kept),
			"matched":           matched,
			"unmatched":         len(kept) - matched,
			"edges_touched":     touched,
			"reason":            reasonOrNil(blame, reason),
			"severity":          severityOrNil(blame, amount),
			"source":            sourceOrNil(o.Source),
		}
		m.History = append(m.History, record)
		records = append(records, record)
		if o.Progress != nil {
			o.Progress(record)
		}
		if o.Stop != nil && o.Stop() {
			break
		}
	}
	return records, nil
}

func phaseOf(blame bool) string {
	if blame {
		return "negative"
	}
	return "clear"
}

func reasonOrNil(blame bool, reason string) any {
	if blame {
		return reason
	}
	return nil
}

func severityOrNil(blame bool, amount float64) any {
	if blame {
		return amount
	}
	return nil
}

func sourceOrNil(source string) any {
	if source == "" {
		return nil
	}
	return source
}

// meanCostOfTexts is the mean -log P of the transitions the texts share with
// the structure (the loss of a clearing pass).
func (m *Model) meanCostOfTexts(texts []string) float64 {
	total, n := 0.0, 0
	for _, text := range texts {
		for _, e := range m.sharedEdges(text) {
			total += m.G.EdgeCost(e)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return total / float64(n)
}

// NegativeCorrection reports what one taught correction blamed and cleared.
type NegativeCorrection struct {
	Changes    []Edit  `json:"changes"`
	Edits      int     `json:"edits"`
	Blamed     int     `json:"blamed"`
	Cleared    int     `json:"cleared"`
	Reason     string  `json:"reason"`
	Severity   float64 `json:"severity"`
	Phase      string  `json:"phase"`
	WrongChars int     `json:"wrong_chars"`
	RightChars int     `json:"right_chars"`
}

// BlameCorrection learns one correction: it blames only the characters the
// tutor changed, and clears what it kept.
//
// wrong is the sentence the network wrote, right the sentence the teacher
// wrote instead.  The two are aligned character by character (Edits) and
//
//   - only the steps of wrong that wrote a character the teacher struck out or
//     replaced are blamed - the words both sentences agree on are not the
//     mistake, so they carry no verdict;
//   - the correction clears blame wherever the failure structure already knows
//     it (nothing is created: a correction is correct English, and this is a
//     network of failures), which also lifts the blame off the words the wrong
//     sentence shares with it;
//   - an edge the correction walks too is never blamed - the network wrote the
//     right characters by another route.
func (m *Model) BlameCorrection(wrong, right string, o BlameOptions) (*NegativeCorrection, error) {
	if err := m.requireNegative("teaching a correction"); err != nil {
		return nil, err
	}
	reason := CleanReason(o.Reason)
	amount := o.severity()
	wrongSpans, rightSpans := ChangedSpans(wrong, right)
	out := &NegativeCorrection{
		Changes: DiffSummary(wrong, right, 8), Edits: len(DiffSummary(wrong, right, 0)), Reason: reason,
		Severity: amount, Phase: "correction",
	}
	for _, s := range wrongSpans {
		out.WrongChars += s.Hi - s.Lo
	}
	for _, s := range rightSpans {
		out.RightChars += s.Hi - s.Lo
	}
	if runeLen(wrong) < Window {
		return out, nil
	}
	grams := Encode(wrong)
	if _, err := m.register([][]string{grams}); err != nil { // the failure joins the structure; the correction never does
		return nil, err
	}
	cleared := map[int]bool{}
	clearedEdges := []int{}
	if runeLen(right) >= Window {
		for _, e := range m.sharedEdges(right) {
			if !cleared[e] {
				cleared[e] = true
				clearedEdges = append(clearedEdges, e)
			}
		}
	}
	blamed := []int{}
	for _, e := range m.stepsOver(grams, runeLen(wrong), wrongSpans) {
		if !cleared[e] {
			blamed = append(blamed, e)
		}
	}
	if len(blamed) > 0 && amount > 0 {
		out.Blamed = m.G.RecordFailure(blamed, amount, reason)
		m.metaAddInt("failures_total", 1)
		m.metaAddFloat("blame_total", amount*float64(out.Blamed))
		m.metaAddInt("trained_texts", 1)
		m.metaAddInt("trained_chars", int64(runeLen(wrong)))
		m.addSource(o.Source)
		m.note(wrong, reason, amount, o.Source, o.Note)
	}
	if len(clearedEdges) > 0 {
		out.Cleared = m.G.RecordClear(clearedEdges, 1)
		m.metaAddInt("cleared_total", 1)
	}
	m.G.Prepare()
	return out, nil
}

// Forget drops (or fades) the blame of one reason, or of everything - the
// tutor can be wrong too.
func (m *Model) Forget(reason string, factor float64) (*ForgetResult, error) {
	if err := m.requireNegative("forgetting a reason"); err != nil {
		return nil, err
	}
	return m.G.Forget(reason, factor)
}

// ---------------------------------------------------------------------------
// the verdict: what a text shares with the failures, and why
// ---------------------------------------------------------------------------

// Crossing is one transition of a text through the failure structure.  Edge is
// -1 where the negative network has never been (an unknown trigram, a missing
// edge, or a junction it cannot represent), which is the common case for text
// that never failed.
type Crossing struct {
	Index    int         `json:"index"`
	Start    int         `json:"start"`
	End      int         `json:"end"`
	Fragment string      `json:"fragment"`
	Parent   int         `json:"parent"`
	Edge     int         `json:"edge"`
	Blame    float64     `json:"blame"`
	Fails    int64       `json:"fails"`
	Clear    float64     `json:"clear"`
	Evidence float64     `json:"evidence"`
	Reason   string      `json:"reason"`
	Reasons  []ReasonRow `json:"reasons"`
}

// Crossings walks a text through the failure structure, in order.
func (m *Model) Crossings(text string) []Crossing {
	g := m.G
	if !m.IsNegative() {
		return nil
	}
	grams := Encode(text)
	if len(grams) == 0 {
		return []Crossing{}
	}
	g.Prepare()
	length := runeLen(text)
	node, offset, lost := Start, 0, false
	crossing := func(i, edge int) Crossing {
		start := i - 1
		if start < 0 {
			start = 0
		}
		end := i + Window
		if end > length {
			end = length
		}
		entry := Crossing{Index: i, Start: start, End: end, Fragment: runeSlice(text, start, end), Parent: node,
			Edge: edge, Reasons: []ReasonRow{}}
		if edge >= 0 {
			entry.Blame = g.Neg.Blame[edge]
			entry.Fails = g.Neg.Fails[edge]
			entry.Clear = g.Neg.Clear[edge]
			entry.Evidence = g.Evidence(edge)
			entry.Reasons = g.EdgeReasons(edge)
			if len(entry.Reasons) > 0 {
				entry.Reason = entry.Reasons[0].Reason
			}
		}
		return entry
	}
	edgeFrom := func(n int) int {
		if lost || (node != Start && offset+Window != g.LabelLen(node)) {
			return -1
		}
		if e, ok := g.Edge(node, n); ok {
			return e
		}
		return -1
	}
	out := make([]Crossing, 0, len(grams)+1)
	for i, gram := range grams {
		at, o, known := g.Lookup(gram)
		if !known {
			out = append(out, crossing(i, -1))
			lost = true
			continue
		}
		if !lost && at == node && o == offset+1 {
			offset = o // a deterministic step inside a compressed node: no edge, nothing to blame
			continue
		}
		edge := -1
		if o == 0 {
			edge = edgeFrom(at)
		}
		out = append(out, crossing(i, edge))
		node, offset, lost = at, o, false
	}
	out = append(out, crossing(len(grams), edgeFrom(End)))
	return out
}

// BlamedSpan is one blamed fragment of a judged text.
type BlamedSpan struct {
	Start    int     `json:"start"`
	End      int     `json:"end"`
	Fragment string  `json:"fragment"`
	Blame    float64 `json:"blame"`
	Fails    int64   `json:"fails"`
	Clear    float64 `json:"clear"`
	Reason   string  `json:"reason"`
}

// Verdict is why a text looks like a failure.
type Verdict struct {
	Text        string       `json:"text"`
	Chars       int          `json:"chars"`
	Transitions int          `json:"transitions"`
	Known       int          `json:"known"`
	Blamed      int          `json:"blamed"`
	Coverage    float64      `json:"coverage"`
	Blame       float64      `json:"blame"`
	Risk        float64      `json:"risk"`
	Peak        float64      `json:"peak"`
	PerChar     float64      `json:"per_char"`
	Threshold   float64      `json:"threshold"`
	MinCoverage float64      `json:"min_coverage"`
	Verdict     string       `json:"verdict"`
	Reasons     []ReasonRow  `json:"reasons"`
	Spans       []BlamedSpan `json:"spans"`
	Why         string       `json:"why"`
}

// JudgeOptions override the model's own thresholds for one judgement.
type JudgeOptions struct {
	Threshold   *float64
	MinCoverage *float64
	Spans       int // 0 = 5
}

// DefaultJudgeOptions report the five worst fragments.
func DefaultJudgeOptions() JudgeOptions { return JudgeOptions{Spans: 5} }

// Judge is the filter: how much of a text is built out of known failure, which
// reasons that blame carries and which fragments carry it.
//
//   - Blame is the summed net evidence of the transitions the text shares with
//     known failures, and Risk that sum over all of the text's transitions, so
//     repeating a failure blamed once scores about 1 (more where the same
//     fragment failed several times over) and sharing a third of one's
//     transitions with it scores about 0.33; compression moves both numbers
//     together, so the ratio does not depend on it.
//   - Coverage is the share of transitions that are known failures, and Peak
//     the most evidence any single transition carries (a correction blames one
//     transition, so a fragment the teacher keeps rewriting shows up there long
//     before Risk moves).
//   - Verdict is "reject" (enough coverage and Risk at or above Threshold),
//     "suspect" (known failures, but below it) or "pass".
//
// A text the network has never seen fail scores 0 and passes: this is a
// filter, not a censor.
func (m *Model) Judge(text string, o JudgeOptions) *Verdict {
	if !m.IsNegative() {
		return nil
	}
	limit, floor := m.Neg.Threshold, m.Neg.MinCoverage
	if o.Threshold != nil {
		limit = *o.Threshold
	}
	if o.MinCoverage != nil {
		floor = *o.MinCoverage
	}
	spans := o.Spans
	if spans == 0 {
		spans = 5
	}
	crossings := m.Crossings(text)
	chars := runeLen(text)
	blamed := make([]Crossing, 0, len(crossings))
	known := 0
	blame := 0.0
	for _, c := range crossings {
		if c.Edge >= 0 {
			known++
		}
		if c.Evidence > 0 {
			blamed = append(blamed, c)
			blame += c.Evidence
		}
	}
	out := &Verdict{
		Text: text, Chars: chars, Transitions: len(crossings), Known: known, Blamed: len(blamed),
		Threshold: limit, MinCoverage: floor, Reasons: []ReasonRow{}, Spans: []BlamedSpan{},
	}
	if len(crossings) > 0 {
		out.Coverage = float64(len(blamed)) / float64(len(crossings))
		out.Risk = blame / float64(len(crossings))
	}
	out.Blame = blame
	if chars > 0 {
		out.PerChar = blame / float64(chars)
	}
	byReason := map[string]float64{}
	order := []string{}
	for _, c := range blamed {
		share := 0.0
		if c.Blame > 0 {
			share = c.Evidence / c.Blame
		}
		if len(c.Reasons) == 0 {
			if _, seen := byReason[UnspecifiedReason]; !seen {
				order = append(order, UnspecifiedReason)
			}
			byReason[UnspecifiedReason] += c.Evidence
			continue
		}
		for _, row := range c.Reasons {
			if _, seen := byReason[row.Reason]; !seen {
				order = append(order, row.Reason)
			}
			byReason[row.Reason] += row.Blame * share
		}
	}
	total := 0.0
	for _, value := range byReason {
		total += value
	}
	if total == 0 {
		total = 1
	}
	for _, name := range order {
		out.Reasons = append(out.Reasons, ReasonRow{Reason: name, Blame: byReason[name], Share: byReason[name] / total})
	}
	sort.SliceStable(out.Reasons, func(i, j int) bool {
		if out.Reasons[i].Blame != out.Reasons[j].Blame {
			return out.Reasons[i].Blame > out.Reasons[j].Blame
		}
		return out.Reasons[i].Reason < out.Reasons[j].Reason
	})
	ranked := append([]Crossing{}, blamed...)
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Evidence > ranked[j].Evidence })
	if len(ranked) > 0 {
		out.Peak = ranked[0].Evidence
	}
	for i, c := range ranked {
		if i >= spans {
			break
		}
		out.Spans = append(out.Spans, BlamedSpan{Start: c.Start, End: c.End, Fragment: c.Fragment, Blame: c.Evidence,
			Fails: c.Fails, Clear: c.Clear, Reason: c.Reason})
	}
	switch {
	case out.Coverage >= floor && len(blamed) > 0 && out.Risk >= limit:
		out.Verdict = "reject"
	case len(blamed) > 0:
		out.Verdict = "suspect"
	default:
		out.Verdict = "pass"
	}
	out.Why = whyVerdict(out.Verdict, len(blamed), len(crossings), out.Risk, out.Reasons, out.Spans)
	m.metaAddInt("judgements", 1)
	if out.Verdict == "reject" {
		m.metaAddInt("rejected", 1)
	}
	return out
}

// whyVerdict is the one sentence a verdict rests on.
func whyVerdict(verdict string, blamed, total int, risk float64, reasons []ReasonRow, spans []BlamedSpan) string {
	if blamed == 0 {
		return "nothing here has failed before"
	}
	out := fmt.Sprintf("%d of %d transitions are known failures (risk %.2f)", blamed, total, risk)
	if len(reasons) > 0 {
		out += ", mostly " + pythonRepr(reasons[0].Reason)
	}
	if len(spans) > 0 {
		out += ", worst at " + pythonRepr(spans[0].Fragment)
	}
	switch verdict {
	case "reject":
		out += "; rejected"
	case "suspect":
		out += "; below the threshold, kept"
	}
	return out
}

// pythonRepr quotes a string the way Python's repr() does, so the sentence a
// verdict carries is character for character the one the Python
// implementation writes (the parity tests compare them).
func pythonRepr(text string) string {
	quote := byte('\'')
	if strings.ContainsRune(text, '\'') && !strings.ContainsRune(text, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range text {
		switch {
		case r == rune(quote) || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString("\\n")
		case r == '\r':
			b.WriteString("\\r")
		case r == '\t':
			b.WriteString("\\t")
		case r < 0x20 || r == 0x7f:
			b.WriteString(fmt.Sprintf("\\x%02x", r))
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// negativeStats are the statistics of a negative network (Model.Stats routes here).
func (m *Model) negativeStats(lastLoss any) map[string]any {
	g := m.G
	n := g.Neg
	top := g.ReasonTable()
	if len(top) > 5 {
		top = top[:5]
	}
	sources := map[string]any{}
	if table, ok := m.Meta["sources"].(map[string]any); ok {
		for key, value := range table {
			sources[key] = value
		}
	}
	return map[string]any{
		"kind":              "negative",
		"nodes":             g.NumNodes(),
		"edges":             g.NumEdges(),
		"trigrams":          g.NumTrigrams(),
		"compression_ratio": g.CompressionRatio(),
		"inverted":          g.Inverted,
		"backend":           "go",
		"device":            m.deviceLabel(),
		"counting":          m.Counting(),
		"epochs_total":      m.metaInt("epochs_total"),
		"trained_chars":     m.metaInt("trained_chars"),
		"trained_texts":     m.metaInt("trained_texts"),
		"twonrl_runs":       m.metaInt("twonrl_runs"),
		"history_len":       len(m.History),
		"last_loss":         lastLoss,
		"failures_total":    m.metaInt("failures_total"),
		"blame_total":       toFloat(m.Meta["blame_total"]),
		"cleared_total":     m.metaInt("cleared_total"),
		"judgements":        m.metaInt("judgements"),
		"rejected":          m.metaInt("rejected"),
		"sources":           sources,
		"edge_blame_total":  n.TotalBlame,
		"edge_clear_total":  n.TotalClear,
		"reason_count":      len(n.ReasonNames),
		"top_reasons":       top,
		"threshold":         m.Neg.Threshold,
		"min_coverage":      m.Neg.MinCoverage,
		"share_scale":       n.ShareScale,
		"blame_scale":       n.BlameScale,
		"clear_scale":       n.ClearScale,
		"log_entries":       len(m.Neg.Log),
	}
}
