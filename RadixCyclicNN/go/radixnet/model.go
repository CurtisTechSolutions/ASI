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
//
// Training spawns one goroutine per text (Workers 0 = no cap).  The counters
// are bumped with plain increments from all those goroutines at once: a data
// race by design - two goroutines hitting the same edge in the same instant
// can lose an update - accepted for speed.  Exact switches the increments to
// atomics, which never lose an update and make the result identical to the
// sequential run and to the Python implementation (the parity tests use it).
type Model struct {
	G       *Graph
	History []map[string]any
	Meta    map[string]any
	Workers int
	Exact   bool

	// Neg is the model-level state of the negative network - the journal and
	// how strictly it judges - and is nil on a count / reward model.
	Neg *Negative
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
func (m *Model) Kind() string {
	if m.G != nil && m.G.IsNegative() {
		return "negative"
	}
	return "count"
}

// workers is the goroutine cap: Model.Workers, else the package default (0 = unbounded).
func (m *Model) workers() int {
	if m.Workers > 0 {
		return m.Workers
	}
	return Workers
}

// Counting reports how the counters are bumped: "racy" (plain increments, lost updates possible) or "exact" (atomic).
func (m *Model) Counting() string {
	if m.Exact {
		return "exact"
	}
	return "racy"
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

// -- training ------------------------------------------------------------------------------------

// DefaultChunkSize is how many texts a pass takes from its source at a time:
// the corpus never has to fit in memory, only one chunk (and the graph).
const DefaultChunkSize = 8192

// TrainOptions configure the passes over the texts.
type TrainOptions struct {
	Epochs       int
	AutoCompress bool
	Phase        string
	Progress     func(record map[string]any)
	Stop         func() bool
	// ChunkSize is the number of texts streamed from the source per chunk (0 = DefaultChunkSize).
	ChunkSize int
	// ParallelParts streams every part of the source (archive entries, files) on its own goroutine at
	// once instead of in corpus order.
	ParallelParts bool
	// Inflight bounds the chunks being read, processed or waiting for their turn at once (0 = two per
	// CPU): the memory ceiling of a pass is the graph plus Inflight chunks, whatever the corpus size.
	Inflight int
}

// DefaultTrainOptions mirror the Python defaults (5 epochs, compression after every epoch).
func DefaultTrainOptions() TrainOptions { return TrainOptions{Epochs: 5, AutoCompress: true} }

// Train counts one traversal of every text's path per epoch - or, on the
// negative network, blames every text: training it *is* blaming (see Blame).
func (m *Model) Train(texts []string, opts TrainOptions) ([]map[string]any, error) {
	if m.IsNegative() {
		return m.Blame(texts, blameFromTrain(opts))
	}
	return m.passesSource(SliceSource(texts), opts, true, 0.0)
}

// TrainSource is Train over a streaming source (a massive ZIP archive, a file,
// several of them): the source is re-read for every pass, chunk by chunk.  The
// negative network blames what the source holds instead, which needs the texts
// in memory - a corpus of failures is small by construction.
func (m *Model) TrainSource(src TextSource, opts TrainOptions) ([]map[string]any, error) {
	if m.IsNegative() {
		texts, err := CollectTexts(src)
		if err != nil {
			return nil, err
		}
		return m.Blame(texts, blameFromTrain(opts))
	}
	return m.passesSource(src, opts, true, 0.0)
}

// blameFromTrain carries a training call's settings over to a blame pass.
func blameFromTrain(opts TrainOptions) BlameOptions {
	return BlameOptions{Epochs: opts.Epochs, NoCompress: !opts.AutoCompress, Progress: opts.Progress, Stop: opts.Stop}
}

// Reward (thumbs up): epochs passes that traverse and reward (+strength) every
// path - or, on the negative network, clearing: it never learns *from* correct
// text, it only lets go of blame.
func (m *Model) Reward(texts []string, epochs int, strength float64) ([]map[string]any, error) {
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	opts.Phase = "positive"
	if m.IsNegative() {
		return m.RewardWith(texts, opts, strength)
	}
	return m.passes(texts, opts, true, math.Abs(strength))
}

// RewardWith is Reward with explicit pass options (progress / stop hooks, phase name).
// On the negative network it clears blame instead (nothing is created).
func (m *Model) RewardWith(texts []string, opts TrainOptions, strength float64) ([]map[string]any, error) {
	if m.IsNegative() {
		weight := math.Abs(strength)
		if weight == 0 {
			weight = 1
		}
		return m.Clear(texts, weight, opts.Epochs)
	}
	if opts.Phase == "" {
		opts.Phase = "positive"
	}
	return m.passes(texts, opts, true, math.Abs(strength))
}

// Punish (thumbs down): epochs passes that penalise (-strength) every path; no traversal is counted.
func (m *Model) Punish(texts []string, epochs int, strength float64) ([]map[string]any, error) {
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	opts.Phase = "negative"
	if m.IsNegative() {
		return m.PunishWith(texts, opts, strength)
	}
	return m.passes(texts, opts, false, -math.Abs(strength))
}

// PunishWith is Punish with explicit pass options (progress / stop hooks, phase name).
// On the negative network it blames the texts (thumbs down: reason "thumbs-down").
func (m *Model) PunishWith(texts []string, opts TrainOptions, strength float64) ([]map[string]any, error) {
	if m.IsNegative() {
		o := blameFromTrain(opts)
		o.Reason, o.Severity, o.Source = "thumbs-down", math.Abs(strength), "feedback"
		return m.Blame(texts, o)
	}
	if opts.Phase == "" {
		opts.Phase = "negative"
	}
	return m.passes(texts, opts, false, -math.Abs(strength))
}

// MetaInt reads an integer metadata entry (0 when absent).
func (m *Model) MetaInt(key string) int64 { return m.metaInt(key) }

// WeightGroup is a set of texts that share a reward / penalty weight.
type WeightGroup struct {
	Weight float64
	Texts  []string
}

// WeightGroups groups texts of equal (3-decimal) weight, heaviest first;
// weights of 0 are dropped.  A rating per text - how good or how bad it is -
// becomes one pass per distinct weight, scaled by it.
func WeightGroups(texts []string, weights []float64, name string) ([]WeightGroup, error) {
	if len(weights) != len(texts) {
		return nil, fmt.Errorf("%s has %d entries for %d text(s)", name, len(weights), len(texts))
	}
	order := []float64{}
	grouped := map[float64][]string{}
	for i, text := range texts {
		weight := weights[i]
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
			return nil, fmt.Errorf("%s must be finite and >= 0, got %g", name, weight)
		}
		if weight == 0 {
			continue
		}
		key := math.Round(weight*1000) / 1000
		if _, seen := grouped[key]; !seen {
			order = append(order, key)
		}
		grouped[key] = append(grouped[key], text)
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(order)))
	groups := make([]WeightGroup, 0, len(order))
	for _, weight := range order {
		groups = append(groups, WeightGroup{Weight: weight, Texts: grouped[weight]})
	}
	return groups, nil
}

// RewardWeighted rewards every text in proportion to its weight (a rating out
// of 1 rather than a single like): one pass per group of equally rated texts,
// each rewarded by weight * strength.  A nil weights slice rewards everything
// alike (RewardWith).
func (m *Model) RewardWeighted(texts []string, weights []float64, opts TrainOptions, strength float64) ([]map[string]any, error) {
	if weights == nil {
		return m.RewardWith(texts, opts, strength)
	}
	return m.weightedPasses(texts, weights, opts, true, math.Abs(strength), "positive")
}

// PunishWeighted penalises every text in proportion to its weight: the worse a
// failure, the larger the penalty.
func (m *Model) PunishWeighted(texts []string, weights []float64, opts TrainOptions, strength float64) ([]map[string]any, error) {
	if weights == nil {
		return m.PunishWith(texts, opts, -math.Abs(strength))
	}
	return m.weightedPasses(texts, weights, opts, false, -math.Abs(strength), "negative")
}

func (m *Model) weightedPasses(
	texts []string, weights []float64, opts TrainOptions, count bool, reward float64, phase string,
) ([]map[string]any, error) {
	name := "good_weights"
	if !count {
		name = "bad_weights"
	}
	groups, err := WeightGroups(texts, weights, name)
	if err != nil {
		return nil, err
	}
	if opts.Phase == "" {
		opts.Phase = phase
	}
	progress := opts.Progress
	records := []map[string]any{}
	for _, group := range groups {
		if opts.Stop != nil && opts.Stop() {
			break
		}
		pass := opts
		pass.Progress = nil // the records are tagged with the weight before they are reported
		grouped, err := m.passes(group.Texts, pass, count, reward*group.Weight)
		if err != nil {
			return records, err
		}
		for _, record := range grouped {
			record["weight"] = group.Weight
			if progress != nil {
				progress(record)
			}
		}
		records = append(records, grouped...)
	}
	return records, nil
}

// TwoNRLOptions are the knobs of TwoNRLWeighted.
type TwoNRLOptions struct {
	NegEpochs int
	PosEpochs int
	Strength  float64
	Progress  func(record map[string]any)
	Stop      func() bool
}

// TwoNRLWeighted is TwoNRL with a rating per text: badWeights scale the
// penalties (the worse a failure, the harder it is pushed away) and
// goodWeights the rewards (the better a text, the more of it is kept).  A nil
// slice weighs that side alike.
func (m *Model) TwoNRLWeighted(
	bad []string, badWeights []float64, good []string, goodWeights []float64, o TwoNRLOptions,
) (*TwoNRLResult, error) {
	strength := o.Strength
	if strength <= 0 {
		strength = 1.0
	}
	negOpts := TrainOptions{Epochs: o.NegEpochs, AutoCompress: true, Phase: "negative", Progress: o.Progress, Stop: o.Stop}
	negative, err := m.PunishWeighted(bad, badWeights, negOpts, strength)
	if err != nil {
		return nil, err
	}
	var positive []map[string]any
	if o.Stop == nil || !o.Stop() {
		posOpts := TrainOptions{Epochs: o.PosEpochs, AutoCompress: true, Phase: "positive", Progress: o.Progress, Stop: o.Stop}
		positive, err = m.RewardWeighted(good, goodWeights, posOpts, strength)
		if err != nil {
			return nil, err
		}
	}
	m.metaAddInt("twonrl_runs", 1)
	return &TwoNRLResult{Negative: negative, Positive: positive, Inverted: m.G.Inverted}, nil
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
	return m.passesSource(SliceSource(texts), opts, count, reward)
}

// passesSource runs opts.Epochs passes over a streaming source with uncapped
// fan-out: the reader never waits - every chunk of ChunkSize texts is
// processed on its own goroutine and every text of a chunk on its own (with
// ParallelParts every part of the source streams on its own goroutine too).
// The only steps that wait are the two that are ordered by nature and run in
// corpus order on a sequencer: the structure pass observes the novel texts (a
// single writer: Go aborts on concurrent map writes) and the counting passes
// apply the sliding window and the rewards.  The counters themselves are
// bumped by the text goroutines, racily unless Exact.  Memory is the graph
// plus at most Inflight chunks: the reader waits when that many are in
// flight, which is what keeps a corpus of any size from piling up.
func (m *Model) passesSource(src TextSource, opts TrainOptions, count bool, reward float64) ([]map[string]any, error) {
	if opts.Epochs < 0 {
		return nil, fmt.Errorf("epochs must be >= 0, got %d", opts.Epochs)
	}
	g := m.G
	records := []map[string]any{}
	chunkSize := opts.ChunkSize
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	parts, err := openParts(src)
	if err != nil {
		return nil, err
	}
	defer parts.Close()
	workers := m.workers()

	// build the structure first (no counting) and compress it, so every pass -
	// the first included - walks the same transitions
	var stats streamStats
	seq := newSequencer(parts.Len())
	err = runParts(parts, chunkSize, workers, opts.Inflight, opts.ParallelParts, &stats, seq, func(part, idx int, chunk []string) error {
		grams := m.encodeAll(chunk)
		novel := make([]bool, len(chunk))
		parallelFor(len(chunk), workers, func(i int) {
			_, _, ok := g.Trace(grams[i])
			novel[i] = !ok
		})
		seq.submit(part, idx, func() {
			g.mu.Lock()
			defer g.mu.Unlock()
			for i, n := range novel {
				if n {
					_, _ = g.observeLocked(grams[i], false)
				}
			}
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	seq.wait()
	if count {
		m.metaAddInt("trained_texts", stats.texts)
		m.metaAddInt("trained_chars", stats.chars)
	}
	pendingMerges := 0
	if opts.AutoCompress {
		pendingMerges = g.Compress()
	}

	for epoch := 0; epoch < opts.Epochs; epoch++ {
		t0 := time.Now()
		traversed := make([]int64, len(g.EdgeW))
		extra := map[int]int64{} // traversals of edges created by a late split (sequencer only)
		var total int64
		var chunks int64
		var passStats streamStats
		seq := newSequencer(parts.Len())
		err := runParts(parts, chunkSize, workers, opts.Inflight, opts.ParallelParts, &passStats, seq, func(part, idx int, chunk []string) error {
			atomic.AddInt64(&chunks, 1)
			grams := m.encodeAll(chunk)
			perText := make([][]Transition, len(chunk))
			paths := make([][]int, len(chunk))
			failed := make([]bool, len(chunk))
			var anyFailed atomic.Bool
			bump := func(i int) {
				tr, path, ok := g.Trace(grams[i])
				if !ok {
					failed[i] = true
					anyFailed.Store(true)
					return
				}
				perText[i], paths[i] = tr, path
				if m.Exact {
					for _, t := range tr {
						atomic.AddInt64(&traversed[t.E], 1)
					}
					if count {
						for _, t := range tr {
							atomic.AddInt64(&g.EdgeCount[t.E], 1)
						}
						for _, n := range path {
							atomic.AddInt64(&g.Count[n], 1)
						}
					}
					return
				}
				// racy by design: plain increments from one goroutine per text
				for _, t := range tr {
					traversed[t.E]++
				}
				if count {
					for _, t := range tr {
						g.EdgeCount[t.E]++
					}
					for _, n := range path {
						g.Count[n]++
					}
				}
			}
			parallelFor(len(chunk), workers, bump)
			seq.submit(part, idx, func() {
				if anyFailed.Load() {
					// a text that needs a split after all: observe it (single writer), trace and count it here
					for i := range chunk {
						if !failed[i] {
							continue
						}
						g.mu.Lock()
						_, _ = g.observeLocked(grams[i], false)
						g.mu.Unlock()
						tr, path, ok := g.Trace(grams[i])
						if !ok {
							continue
						}
						perText[i], paths[i] = tr, path
						for _, t := range tr {
							if t.E < len(traversed) {
								atomic.AddInt64(&traversed[t.E], 1)
							} else {
								extra[t.E]++
							}
							if count {
								atomic.AddInt64(&g.EdgeCount[t.E], 1)
							}
						}
						if count {
							for _, n := range path {
								atomic.AddInt64(&g.Count[n], 1)
							}
						}
					}
				}
				size := 0
				for _, tr := range perText {
					size += len(tr)
				}
				edges := make([]int, 0, size) // exactly once: a pass over a huge corpus builds millions of these
				for _, tr := range perText {
					for _, t := range tr {
						edges = append(edges, t.E)
					}
				}
				atomic.AddInt64(&total, int64(len(edges)))
				if count {
					g.RecordTraversals(edges) // the sliding window follows the corpus order
				}
				if reward != 0 {
					g.AddReward(edges, reward)
				}
			})
			return nil
		})
		if err != nil {
			return nil, err
		}
		seq.wait()
		if reward != 0 {
			m.metaAddInt("feedback_passes", 1)
			if reward > 0 {
				m.metaAddFloat("rewards_total", reward*float64(total))
			} else {
				m.metaAddFloat("penalties_total", -reward*float64(total))
			}
		}
		g.Prepare()
		loss := m.weightedCost(traversed, extra, total)
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
			"chunks":            int(chunks),
			"parts":             parts.Len(),
			"seconds":           time.Since(t0).Seconds(),
			"skipped_short":     int(stats.skippedShort),
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

// weightedCost is the mean cost of the pass's traversals: every edge's cost
// times how often the pass traversed it (a parallel reduction over the edges).
func (m *Model) weightedCost(traversed []int64, extra map[int]int64, total int64) float64 {
	if total == 0 {
		return 0
	}
	g := m.G
	n := len(traversed)
	tail := 0.0
	for e, c := range extra {
		tail += float64(c) * g.edgeCost[e]
	}
	if n < 4096 || m.workers() == 1 {
		sum := 0.0
		for e, c := range traversed {
			if c != 0 {
				sum += float64(c) * g.edgeCost[e]
			}
		}
		return (sum + tail) / float64(total)
	}
	chunk := 4096
	blocks := (n + chunk - 1) / chunk
	partial := make([]float64, blocks)
	parallelFor(blocks, m.workers(), func(b int) {
		lo, hi := b*chunk, (b+1)*chunk
		if hi > n {
			hi = n
		}
		s := 0.0
		for e := lo; e < hi; e++ {
			if c := traversed[e]; c != 0 {
				s += float64(c) * g.edgeCost[e]
			}
		}
		partial[b] = s
	})
	sum := tail
	for _, s := range partial {
		sum += s
	}
	return sum / float64(total)
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
	if m.IsNegative() {
		return m.negativeStats(lastLoss)
	}
	return map[string]any{
		"kind":                 "count",
		"nodes":                g.NumNodes(),
		"edges":                g.NumEdges(),
		"trigrams":             g.NumTrigrams(),
		"compression_ratio":    g.CompressionRatio(),
		"inverted":             g.Inverted,
		"backend":              "go",
		"device":               m.deviceLabel(),
		"counting":             m.Counting(),
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

// deviceLabel describes the goroutine pool for stats: "cpu" with the cap or "cpu (one goroutine per text)".
func (m *Model) deviceLabel() string {
	if w := m.workers(); w > 0 {
		return fmt.Sprintf("cpu x%d", w)
	}
	return "cpu (one goroutine per text)"
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
