package radixnet

import (
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"
)

// UnknownProb is the probability charged for a transition the structure does not know.
const UnknownProb = 1e-6

var logUnknown = math.Log(UnknownProb)

const maxLogPerplexity = 700.0

// Model is the count / reward model: a Graph plus training history and metadata.
type Model struct {
	G       *Graph
	History []map[string]any
	Meta    map[string]any
	Workers int
}

// NewModel creates an untrained model.
func NewModel(seed int64, opts GraphOptions) (*Model, error) {
	g, err := NewGraph(seed, opts)
	if err != nil {
		return nil, err
	}
	return newModelWithGraph(g), nil
}

func newModelWithGraph(g *Graph) *Model {
	m := &Model{G: g, Workers: Workers}
	m.Meta = map[string]any{
		"created": utcNow(), "seed": g.Seed, "epochs_total": 0, "trained_chars": 0, "trained_texts": 0,
		"twonrl_runs": 0, "rewards_total": 0.0, "penalties_total": 0.0, "feedback_passes": 0,
	}
	return m
}

// Kind is the model kind shared with the Python implementation.
func (m *Model) Kind() string { return "count" }

func (m *Model) workers() int {
	if m.Workers > 0 {
		return m.Workers
	}
	return Workers
}

// -- meta helpers -----------------------------------------------------------------------

func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	}
	return 0
}

func (m *Model) metaInt(key string) int64 { return int64(toFloat(m.Meta[key])) }

func (m *Model) metaAddInt(key string, delta int64) { m.Meta[key] = m.metaInt(key) + delta }

func (m *Model) metaAddFloat(key string, delta float64) { m.Meta[key] = toFloat(m.Meta[key]) + delta }

// -- text helpers ---------------------------------------------------------------------------

// cleanTexts drops texts shorter than a trigram; returns (usable, skipped).
func cleanTexts(texts []string) ([]string, int) {
	kept := make([]string, 0, len(texts))
	skipped := 0
	for _, t := range texts {
		if runeLen(t) < Window {
			skipped++
		} else {
			kept = append(kept, t)
		}
	}
	return kept, skipped
}

// encodeAll encodes texts on the worker goroutines.
func (m *Model) encodeAll(texts []string) [][]string {
	grams := make([][]string, len(texts))
	parallelFor(len(texts), m.workers(), func(i int) { grams[i] = Encode(texts[i]) })
	return grams
}

// register makes every text walkable: the walkable ones are detected in
// parallel (read-only), the others are observed structurally in corpus order
// (the one sequential phase: splits reshape a shared radix structure).
func (m *Model) register(grams [][]string) error {
	g := m.G
	needs := make([]bool, len(grams))
	parallelFor(len(grams), m.workers(), func(i int) {
		if len(grams[i]) == 0 {
			return
		}
		_, _, ok := g.Trace(grams[i])
		needs[i] = !ok
	})
	g.mu.Lock()
	defer g.mu.Unlock()
	for i, need := range needs {
		if need {
			if _, err := g.observeLocked(grams[i], false); err != nil {
				return err
			}
		}
	}
	return nil
}

// traceAll derives every text's transitions and node path from the frozen
// structure in parallel; a text that cannot be walked is registered and traced
// again (sequentially, under the lock).
func (m *Model) traceAll(grams [][]string) ([][]Transition, [][]int, error) {
	g := m.G
	transitions := make([][]Transition, len(grams))
	paths := make([][]int, len(grams))
	failed := int32(0)
	parallelFor(len(grams), m.workers(), func(i int) {
		if len(grams[i]) == 0 {
			return
		}
		tr, path, ok := g.TraceUnlocked(grams[i])
		if ok {
			transitions[i], paths[i] = tr, path
		} else {
			atomic.StoreInt32(&failed, 1)
		}
	})
	if failed != 0 {
		if err := m.register(grams); err != nil {
			return nil, nil, err
		}
		for i := range grams {
			if len(grams[i]) == 0 {
				continue
			}
			tr, path, ok := g.Trace(grams[i])
			if !ok {
				return nil, nil, fmt.Errorf("internal error: text %d is not walkable after registration", i)
			}
			transitions[i], paths[i] = tr, path
		}
	}
	return transitions, paths, nil
}

// -- training ------------------------------------------------------------------------------------

// TrainOptions configure the passes over the texts.
type TrainOptions struct {
	Epochs       int
	AutoCompress bool
	Phase        string
	Progress     func(record map[string]any)
	Stop         func() bool
}

// DefaultTrainOptions mirror the Python defaults (5 epochs, compression after every epoch).
func DefaultTrainOptions() TrainOptions { return TrainOptions{Epochs: 5, AutoCompress: true} }

// Train counts one traversal of every text's path per epoch.
func (m *Model) Train(texts []string, opts TrainOptions) ([]map[string]any, error) {
	return m.passes(texts, opts, true, 0.0)
}

// Reward (thumbs up): epochs passes that traverse and reward (+strength) every path.
func (m *Model) Reward(texts []string, epochs int, strength float64) ([]map[string]any, error) {
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	opts.Phase = "positive"
	return m.passes(texts, opts, true, math.Abs(strength))
}

// Punish (thumbs down): epochs passes that penalise (-strength) every path; no traversal is counted.
func (m *Model) Punish(texts []string, epochs int, strength float64) ([]map[string]any, error) {
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	opts.Phase = "negative"
	return m.passes(texts, opts, false, -math.Abs(strength))
}

// TwoNRLResult is the outcome of TwoNRL.
type TwoNRLResult struct {
	Negative []map[string]any `json:"negative"`
	Positive []map[string]any `json:"positive"`
	Inverted bool             `json:"inverted"`
}

// TwoNRL penalises the bad texts (negEpochs passes), then counts and rewards
// the good ones (posEpochs passes); nothing is inverted.
func (m *Model) TwoNRL(bad, good []string, negEpochs, posEpochs int, strength float64) (*TwoNRLResult, error) {
	negative, err := m.Punish(bad, negEpochs, strength)
	if err != nil {
		return nil, err
	}
	positive, err := m.Reward(good, posEpochs, strength)
	if err != nil {
		return nil, err
	}
	m.metaAddInt("twonrl_runs", 1)
	return &TwoNRLResult{Negative: negative, Positive: positive, Inverted: m.G.Inverted}, nil
}

func (m *Model) passes(texts []string, opts TrainOptions, count bool, reward float64) ([]map[string]any, error) {
	if opts.Epochs < 0 {
		return nil, fmt.Errorf("epochs must be >= 0, got %d", opts.Epochs)
	}
	texts, skippedShort := cleanTexts(texts)
	g := m.G
	records := []map[string]any{}
	grams := m.encodeAll(texts)
	if count {
		m.metaAddInt("trained_texts", int64(len(texts)))
		chars := int64(0)
		for _, t := range texts {
			chars += int64(runeLen(t))
		}
		m.metaAddInt("trained_chars", chars)
	}
	// build the structure first (no counting) and compress it, so every pass -
	// the first included - walks the same transitions
	if err := m.register(grams); err != nil {
		return nil, err
	}
	pendingMerges := 0
	if opts.AutoCompress {
		pendingMerges = g.Compress()
	}
	for epoch := 0; epoch < opts.Epochs; epoch++ {
		t0 := time.Now()
		perText, paths, err := m.traceAll(grams)
		if err != nil {
			return nil, err
		}
		total := 0
		for _, tr := range perText {
			total += len(tr)
		}
		edges := make([]int, 0, total)
		for _, tr := range perText {
			for _, t := range tr {
				edges = append(edges, t.E)
			}
		}
		if count {
			// lock-free counting: atomic increments from the worker goroutines
			parallelFor(len(perText), m.workers(), func(i int) {
				for _, t := range perText[i] {
					atomic.AddInt64(&g.EdgeCount[t.E], 1)
				}
				for _, n := range paths[i] {
					atomic.AddInt64(&g.Count[n], 1)
				}
			})
			// the sliding window follows the corpus order
			g.RecordTraversals(edges)
		}
		if reward != 0 {
			g.AddReward(edges, reward)
			m.metaAddInt("feedback_passes", 1)
			if reward > 0 {
				m.metaAddFloat("rewards_total", reward*float64(len(edges)))
			} else {
				m.metaAddFloat("penalties_total", -reward*float64(len(edges)))
			}
		}
		g.Prepare()
		loss := m.meanCost(edges)
		merges := pendingMerges
		pendingMerges = 0
		if opts.AutoCompress {
			merges += g.Compress()
		}
		m.metaAddInt("epochs_total", 1)
		record := map[string]any{
			"epoch":             m.metaInt("epochs_total"),
			"loss":              loss,
			"perplexity":        math.Exp(math.Min(loss, maxLogPerplexity)),
			"nodes":             g.NumNodes(),
			"edges":             g.NumEdges(),
			"trigrams":          g.NumTrigrams(),
			"compression_ratio": g.CompressionRatio(),
			"merges":            merges,
			"transitions":       total,
			"seconds":           time.Since(t0).Seconds(),
			"skipped_short":     skippedShort,
			"traversed":         count,
			"reward":            reward,
		}
		if opts.Phase != "" {
			record["phase"] = opts.Phase
		}
		m.History = append(m.History, record)
		records = append(records, record)
		if opts.Progress != nil {
			opts.Progress(record)
		}
		if opts.Stop != nil && opts.Stop() {
			break
		}
	}
	return records, nil
}

// meanCost is the mean -log P of the traversed edges under the current weights (parallel reduction).
func (m *Model) meanCost(edges []int) float64 {
	if len(edges) == 0 {
		return 0
	}
	g := m.G
	workers := m.workers()
	if len(edges) < 4096 || workers < 2 {
		total := 0.0
		for _, e := range edges {
			total += g.edgeCost[e]
		}
		return total / float64(len(edges))
	}
	chunk := (len(edges) + workers - 1) / workers
	partial := make([]float64, workers)
	parallelFor(workers, workers, func(w int) {
		lo := w * chunk
		hi := lo + chunk
		if hi > len(edges) {
			hi = len(edges)
		}
		s := 0.0
		for _, e := range edges[lo:hi] {
			s += g.edgeCost[e]
		}
		partial[w] = s
	})
	total := 0.0
	for _, s := range partial {
		total += s
	}
	return total / float64(len(edges))
}

// Invert flips the sign of every reward.
func (m *Model) Invert() { m.G.Invert() }

// ConfigureWeights changes the weight function's scales / window.
func (m *Model) ConfigureWeights(opts map[string]float64) error { return m.G.Configure(opts) }

// pathsOf returns the node paths of texts, registering a text structurally when it cannot be walked yet.
func (m *Model) pathsOf(texts []string) ([][]int, error) {
	g := m.G
	paths := make([][]int, 0, len(texts))
	for _, text := range texts {
		grams := Encode(text)
		if grams == nil {
			continue
		}
		path, ok := g.NodePath(grams)
		if !ok {
			if _, err := g.ObserveSequence(grams, false); err != nil {
				return nil, err
			}
			path, ok = g.NodePath(grams)
		}
		if ok && len(path) > 0 {
			paths = append(paths, path)
		}
	}
	return paths, nil
}

// InvertPathsResult reports what InvertPaths did.
type InvertPathsResult struct {
	Texts      int     `json:"texts"`
	Flipped    int     `json:"flipped"`
	Unit       string  `json:"unit"`
	Mode       string  `json:"mode"`
	AmountMean float64 `json:"amount_mean"`
}

// InvertPaths (failures): every edge of a text's path loses strength * 2 *
// amount reward; amounts nil means 1 for every text.
func (m *Model) InvertPaths(texts []string, amounts []float64, strength float64) (*InvertPathsResult, error) {
	texts, _ = cleanTexts(texts)
	if amounts == nil {
		amounts = make([]float64, len(texts))
		for i := range amounts {
			amounts[i] = 1
		}
	}
	if len(amounts) != len(texts) {
		return nil, fmt.Errorf("amounts has %d entries for %d texts", len(amounts), len(texts))
	}
	for _, v := range amounts {
		if v < 0 || v > 1 || math.IsNaN(v) {
			return nil, fmt.Errorf("amounts must lie in [0, 1], got %v", v)
		}
	}
	paths, err := m.pathsOf(texts)
	if err != nil {
		return nil, err
	}
	g := m.G
	penalties := make(map[int]float64)
	order := []int{}
	for i, path := range paths {
		if i >= len(amounts) {
			break
		}
		penalty := math.Abs(strength) * 2.0 * amounts[i]
		if penalty <= 0 {
			continue
		}
		for j := 0; j+1 < len(path); j++ {
			if e, ok := g.Edge(path[j], path[j+1]); ok {
				if _, seen := penalties[e]; !seen {
					order = append(order, e)
				}
				if penalty > penalties[e] {
					penalties[e] = penalty
				}
			}
		}
	}
	touched := 0
	for _, e := range order {
		touched += g.AddReward([]int{e}, -penalties[e])
		m.metaAddFloat("penalties_total", penalties[e])
	}
	if touched > 0 {
		m.metaAddInt("feedback_passes", 1)
	}
	applied, sum := 0, 0.0
	for _, v := range amounts {
		if v > 0 {
			applied++
			sum += v
		}
	}
	mean := 0.0
	if applied > 0 {
		mean = sum / float64(applied)
	}
	return &InvertPathsResult{Texts: len(texts), Flipped: touched, Unit: "edges", Mode: "penalty", AmountMean: mean}, nil
}

// -- prediction ------------------------------------------------------------------------------------

// locate finds where prefix ends in the graph: (node, offset, matched characters of the located trigram).
func (m *Model) locate(prefix string) (node, offset, matched int) {
	g := m.G
	runes := []rune(prefix)
	n := len(runes)
	if n == 0 {
		return Start, 0, 0
	}
	if n >= Window {
		if nd, off, ok := g.Lookup(string(runes[n-Window:])); ok {
			return nd, off, Window
		}
		for _, k := range []int{Window - 1, 1} {
			if nd, off, ok := m.bestTrigram(string(runes[n-k:])); ok {
				return nd, off, k
			}
		}
		return Start, 0, 0
	}
	if nd, ok := m.bestNodeWithPrefix(prefix); ok {
		return nd, 0, n
	}
	return Start, 0, 0
}

// bestTrigram is the most visited (node, offset) holding a trigram that starts with key.
func (m *Model) bestTrigram(key string) (int, int, bool) {
	g := m.G
	found := false
	var best loc
	var bestRank [3]int64
	for t, l := range g.index {
		if len(t) >= len(key) && t[:len(key)] == key {
			rank := [3]int64{-g.Count[l.node], int64(l.node), int64(l.off)}
			if !found || rank[0] < bestRank[0] || (rank[0] == bestRank[0] && (rank[1] < bestRank[1] || (rank[1] == bestRank[1] && rank[2] < bestRank[2]))) {
				best, bestRank, found = l, rank, true
			}
		}
	}
	return best.node, best.off, found
}

// bestNodeWithPrefix is the most visited real node whose label starts with prefix.
func (m *Model) bestNodeWithPrefix(prefix string) (int, bool) {
	g := m.G
	best, bestCount := -1, int64(-1)
	for node := 2; node < len(g.Labels); node++ {
		if g.Alive[node] && len(g.Labels[node]) >= len(prefix) && g.Labels[node][:len(prefix)] == prefix {
			if c := g.Count[node]; c > bestCount {
				best, bestCount = node, c
			}
		}
	}
	return best, best >= 0
}

// prefixStart is (node, offset, lead): where the prefix ends and the unmatched
// remainder of the located trigram, which every predicted path starts with.
func (m *Model) prefixStart(prefix string) (int, int, string) {
	node, offset, matched := m.locate(prefix)
	lead := ""
	if node != Start && matched < Window {
		lead = runeSlice(m.G.Labels[node], offset+matched, offset+Window)
	}
	return node, offset, lead
}

// PredictOptions configure Predict; MaxLength < 0 means no cap, Beam 0 the default width.
type PredictOptions struct {
	Length      int
	Mode        string
	K           int
	Beam        int
	StepPenalty float64
	Temperature float64
	ToEnd       bool
	MaxLength   int
}

// DefaultPredictOptions mirror the Python defaults.
func DefaultPredictOptions() PredictOptions {
	return PredictOptions{Length: 20, Mode: "beam", K: 5, Temperature: 1.0, MaxLength: -1}
}

func checkPredictArgs(o PredictOptions) error {
	if o.Length < 0 {
		return fmt.Errorf("length must be >= 0, got %d", o.Length)
	}
	if o.K < 0 {
		return fmt.Errorf("k must be >= 0, got %d", o.K)
	}
	if o.Beam < 0 {
		return fmt.Errorf("beam must be >= 1, got %d", o.Beam)
	}
	if o.StepPenalty < 0 {
		return fmt.Errorf("step_penalty must be >= 0")
	}
	if o.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	return nil
}

// Predict continues prefix: the K most likely and the K least likely
// continuations in one beam search ("dijkstra" is an alias of "beam"), or one
// stochastic walk ("sample").  Text is the continuation, FullText prefix + text.
func (m *Model) Predict(prefix string, o PredictOptions) (*Prediction, error) {
	if err := checkPredictArgs(o); err != nil {
		return nil, err
	}
	mode := o.Mode
	if mode == "" || mode == "dijkstra" {
		mode = "beam"
	}
	if mode != "beam" && mode != "sample" {
		return nil, fmt.Errorf("unknown mode %q; expected 'beam', 'dijkstra' or 'sample'", o.Mode)
	}
	return m.search(prefix, o.Length, mode, o.K, o.Beam, o.StepPenalty, o.Temperature, o.ToEnd, o.MaxLength, nil)
}

// search is the prediction engine shared by Predict, Generate and Converse.
func (m *Model) search(prefix string, length int, mode string, k, beam int, stepPenalty, temperature float64, toEnd bool, maxLength int, rng *MT19937) (*Prediction, error) {
	g := m.G
	node, offset, lead := m.prefixStart(prefix)
	leadLen := runeLen(lead)
	want := length - leadLen
	if want < 0 {
		want = 0
	}
	var top, bottom []*PathResult
	var expanded, width int
	cap := -1
	var err error
	if mode == "beam" {
		maxChars := -1
		if maxLength < 0 && length == 0 {
			cap, maxChars = 0, 0
		} else if maxLength >= 0 {
			cap = length
			if maxLength > cap {
				cap = maxLength
			}
			maxChars = cap - leadLen
			if want > maxChars {
				maxChars = want
			}
		}
		top, bottom, expanded, err = g.BeamPredict(node, offset, want, BeamOptions{K: k, Beam: beam, MaxChars: maxChars, StepPenalty: stepPenalty, ToEnd: toEnd})
		if err != nil {
			return nil, err
		}
		width = beam
		if width == 0 {
			width = DefaultBeam(k)
		}
	} else {
		cap = length
		if maxLength >= 0 {
			cap = maxLength
		}
		maxChars := cap - leadLen
		if maxChars < 0 {
			maxChars = 0
		}
		walk, err := g.SampleWalk(node, offset, maxChars, temperature, rng, nil)
		if err != nil {
			return nil, err
		}
		top, bottom, expanded, width = []*PathResult{walk}, []*PathResult{}, walk.Expanded, 0
	}
	fix := func(r *PathResult) {
		if lead != "" {
			r.Text = lead + r.Text
			if cap >= 0 {
				r.Text = truncateRunes(r.Text, cap)
			}
		}
		r.FullText = prefix + r.Text
	}
	for _, r := range top {
		fix(r)
	}
	for _, r := range bottom {
		fix(r)
	}
	var best *PathResult
	if len(top) > 0 {
		best = top[0]
	} else {
		text := lead
		if cap >= 0 {
			text = truncateRunes(lead, cap)
		}
		best = &PathResult{Text: text, Labels: []string{g.Labels[node]}, NodeIDs: []int{node}, StepCosts: []float64{}}
		best.FullText = prefix + best.Text
	}
	pred := &Prediction{PathResult: *best, Top: top, Bottom: bottom, K: k, Beam: width, Mode: mode}
	pred.Expanded = expanded
	pred.Labels = append([]string(nil), best.Labels...)
	pred.NodeIDs = append([]int(nil), best.NodeIDs...)
	pred.StepCosts = append([]float64(nil), best.StepCosts...)
	return pred, nil
}

// GenerateOptions configure Generate.
type GenerateOptions struct {
	MaxLength   int
	Mode        string
	Temperature float64
	Count       int
	Seed        *int64
	Prefix      string
	StepPenalty float64
	Beam        int
}

// DefaultGenerateOptions mirror the Python defaults.
func DefaultGenerateOptions() GenerateOptions {
	return GenerateOptions{MaxLength: 60, Mode: "sample", Temperature: 1.0, Count: 1}
}

// Generate whole texts with the prediction search, from START or continuing a
// prefix: "beam" (the count most likely distinct complete texts), "sample"
// (count stochastic walks) or "dijkstra" (the single cheapest complete text).
// Every result's Text is the whole text, prefix included.
func (m *Model) Generate(o GenerateOptions) ([]*PathResult, error) {
	if o.MaxLength < 0 {
		return nil, fmt.Errorf("max_length must be >= 0, got %d", o.MaxLength)
	}
	if o.Count < 0 {
		return nil, fmt.Errorf("count must be >= 0, got %d", o.Count)
	}
	mode := o.Mode
	if mode == "" {
		mode = "sample"
	}
	if mode != "beam" && mode != "dijkstra" && mode != "sample" {
		return nil, fmt.Errorf("unknown mode %q; expected 'beam', 'dijkstra' or 'sample'", o.Mode)
	}
	whole := func(r *PathResult) *PathResult {
		r.FullText = o.Prefix + r.Text
		r.Text = r.FullText
		return r
	}
	if o.Count == 0 {
		return []*PathResult{}, nil
	}
	if mode == "sample" {
		var rng *MT19937
		if o.Seed != nil {
			rng = NewMT19937(*o.Seed)
		}
		results := make([]*PathResult, 0, o.Count)
		for i := 0; i < o.Count; i++ {
			walk, err := m.search(o.Prefix, o.MaxLength, "sample", 0, 0, 0.0, o.Temperature, false, o.MaxLength, rng)
			if err != nil {
				return nil, err
			}
			results = append(results, whole(&walk.PathResult))
		}
		return results, nil
	}
	k := o.Count
	if mode == "dijkstra" {
		k = 1
	}
	found, err := m.search(o.Prefix, 0, "beam", k, o.Beam, o.StepPenalty, 1.0, true, o.MaxLength, nil)
	if err != nil {
		return nil, err
	}
	results := make([]*PathResult, 0, len(found.Top))
	for _, r := range found.Top {
		results = append(results, whole(r))
	}
	if mode == "dijkstra" && len(results) > 1 {
		results = results[:1]
	}
	return results, nil
}

// -- scoring ------------------------------------------------------------------------------------------

// Score is the log-probability of a text under the model.
type Score struct {
	LogProb            float64 `json:"log_prob"`
	PerChar            float64 `json:"per_char"`
	Chars              int     `json:"chars"`
	Transitions        int     `json:"transitions"`
	UnknownTransitions int     `json:"unknown_transitions"`
}

// edgeLogProb is log P(c | p) for the edge taken from p's trigram at offset,
// or ok=false when p is not positioned at its last trigram or the edge is missing.
func (m *Model) edgeLogProb(p, offset, c int) (float64, bool) {
	g := m.G
	if p != Start && offset+Window != g.labelLen[p] {
		return 0, false
	}
	e, ok := g.Edge(p, c)
	if !ok {
		return 0, false
	}
	return -g.EdgeCost(e), true
}

// Score walks the text START -> ... -> END; unknown trigrams, missing edges and
// transitions that would need a split cost log(UnknownProb).
func (m *Model) Score(text string) Score {
	g := m.G
	g.Prepare()
	grams := Encode(text)
	chars := runeLen(text)
	if grams == nil {
		return Score{Chars: chars}
	}
	logProb := 0.0
	transitions, unknown := 0, 0
	node, offset := Start, 0
	lost := false
	for _, t := range grams {
		l, ok := g.index[t]
		if !ok {
			transitions++
			unknown++
			logProb += logUnknown
			lost = true
			continue
		}
		if !lost && l.node == node && l.off == offset+1 {
			offset = l.off
			continue
		}
		transitions++
		lp, known := 0.0, false
		if !lost && l.off == 0 {
			lp, known = m.edgeLogProb(node, offset, l.node)
		}
		if !known {
			unknown++
			logProb += logUnknown
		} else {
			logProb += lp
		}
		node, offset, lost = l.node, l.off, false
	}
	transitions++
	lp, known := 0.0, false
	if !lost {
		lp, known = m.edgeLogProb(node, offset, End)
	}
	if !known {
		unknown++
		logProb += logUnknown
	} else {
		logProb += lp
	}
	denom := chars
	if denom < 1 {
		denom = 1
	}
	return Score{LogProb: logProb, PerChar: logProb / float64(denom), Chars: chars, Transitions: transitions, UnknownTransitions: unknown}
}

// ScoreAll scores many texts on the worker goroutines.
func (m *Model) ScoreAll(texts []string) []Score {
	m.G.Prepare()
	out := make([]Score, len(texts))
	parallelFor(len(texts), m.workers(), func(i int) { out[i] = m.Score(texts[i]) })
	return out
}

// -- statistics ----------------------------------------------------------------------------------------

// Stats mirrors the Python model's stats() dictionary.
func (m *Model) Stats() map[string]any {
	g := m.G
	pos, neg := g.TotalReward()
	var lastLoss any
	if len(m.History) > 0 {
		lastLoss = m.History[len(m.History)-1]["loss"]
	}
	return map[string]any{
		"kind":                 "count",
		"nodes":                g.NumNodes(),
		"edges":                g.NumEdges(),
		"trigrams":             g.NumTrigrams(),
		"compression_ratio":    g.CompressionRatio(),
		"inverted":             g.Inverted,
		"backend":              "go",
		"device":               fmt.Sprintf("cpu x%d", m.workers()),
		"epochs_total":         m.metaInt("epochs_total"),
		"trained_chars":        m.metaInt("trained_chars"),
		"trained_texts":        m.metaInt("trained_texts"),
		"twonrl_runs":          m.metaInt("twonrl_runs"),
		"history_len":          len(m.History),
		"last_loss":            lastLoss,
		"rewards_total":        toFloat(m.Meta["rewards_total"]),
		"penalties_total":      toFloat(m.Meta["penalties_total"]),
		"feedback_passes":      m.metaInt("feedback_passes"),
		"edge_reward_positive": pos,
		"edge_reward_negative": neg,
		"count_scale":          g.CountScale,
		"reward_scale":         g.RewardScale,
		"global_scale":         g.GlobalScale,
		"window_scale":         g.WindowScale,
		"window":               g.WindowSize,
		"total_traversals":     g.TotalTraversals,
		"window_traversals":    g.WindowTraversals(),
	}
}

// SortedKeys is a small helper for deterministic output of maps.
func SortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
