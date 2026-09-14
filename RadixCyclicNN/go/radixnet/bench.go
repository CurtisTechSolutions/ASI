package radixnet

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// Benchmarks: how fast the count / reward model counts and predicts.
//
// The corpus is synthetic and structured the same way the Python benchmark's
// is - verbatim lines, lines spliced at a word boundary, lines with a word or
// two swapped - so the graph it builds has the same shape: unary chains to
// compress, shared prefixes to branch on and realistic fan-out.  The exact
// sentences are not held to Python's, because a benchmark measures throughput
// rather than numbers; what is comparable is the corpus size and the work done.

// Benchmark defaults, matching the Python module.
const (
	BenchDefaultChars  = 20000
	BenchDefaultEpochs = 2
	BenchPredictLength = 20
	benchMinPrefix     = 3
	benchMaxPrefix     = 12
	benchMinPredicts   = 100
	benchMaxPredicts   = 20000
)

var benchFallbackLines = []string{
	"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill",
	"the cat chased the mouse", "the quick brown fox jumps over the lazy dog",
	"rain fell on the quiet town", "she walked to the river at dawn",
	"the old man read a book by the fire", "children played in the summer field",
	"a train passed through the empty station",
}

// BenchLines are the non-blank lines of at least three characters of the sample
// corpus, falling back to a built-in list when the file cannot be read.
func BenchLines(path string) []string {
	lines := []string{}
	if data, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if trimmed := strings.TrimSpace(line); len([]rune(trimmed)) >= 3 {
				lines = append(lines, trimmed)
			}
		}
	}
	if len(lines) == 0 {
		return append([]string{}, benchFallbackLines...)
	}
	return lines
}

// SyntheticCorpus builds deterministic training texts totalling at least chars
// characters out of base, by the three recipes described above.
func SyntheticCorpus(chars int, seed int64, base []string) ([]string, error) {
	if chars < 0 {
		return nil, fmt.Errorf("chars must be >= 0, got %d", chars)
	}
	lines := []string{}
	for _, line := range base {
		if len([]rune(line)) >= 3 {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("need at least one base line of >= 3 characters")
	}
	words := map[string]bool{}
	for _, line := range lines {
		for _, word := range strings.Fields(line) {
			words[word] = true
		}
	}
	vocab := make([]string, 0, len(words))
	for word := range words {
		vocab = append(vocab, word)
	}
	sort.Strings(vocab)

	rng := rand.New(rand.NewSource(seed))
	texts := []string{}
	total := 0
	for total < chars {
		roll := rng.Float64()
		var text string
		switch {
		case roll < 0.4 || len(vocab) < 2:
			text = lines[rng.Intn(len(lines))]
		case roll < 0.7:
			left := strings.Fields(lines[rng.Intn(len(lines))])
			right := strings.Fields(lines[rng.Intn(len(lines))])
			text = strings.Join(append(append([]string{}, left[:1+rng.Intn(len(left))]...),
				right[rng.Intn(len(right)):]...), " ")
		default:
			parts := strings.Fields(lines[rng.Intn(len(lines))])
			swapped := append([]string{}, parts...)
			for i := 0; i <= rng.Intn(2); i++ {
				swapped[rng.Intn(len(swapped))] = vocab[rng.Intn(len(vocab))]
			}
			text = strings.Join(swapped, " ")
		}
		if len([]rune(text)) < 3 {
			text = lines[rng.Intn(len(lines))]
		}
		texts = append(texts, text)
		total += len([]rune(text))
	}
	return texts, nil
}

// BenchPrefixes cuts count prefixes (3-12 characters) out of the corpus, so
// every prediction starts at a node the trained graph knows.
func BenchPrefixes(texts []string, count int, seed int64) ([]string, error) {
	if count < 0 {
		return nil, fmt.Errorf("count must be >= 0, got %d", count)
	}
	if count > 0 && len(texts) == 0 {
		return nil, fmt.Errorf("cannot cut prefixes out of an empty corpus")
	}
	rng := rand.New(rand.NewSource(seed + 1))
	out := make([]string, 0, count)
	for i := 0; i < count; i++ {
		runes := []rune(texts[rng.Intn(len(texts))])
		high := benchMaxPrefix
		if len(runes) < high {
			high = len(runes)
		}
		length := benchMinPrefix
		if high > benchMinPrefix {
			length += rng.Intn(high - benchMinPrefix + 1)
		}
		start := 0
		if len(runes) > length {
			start = rng.Intn(len(runes) - length + 1)
		}
		out = append(out, string(runes[start:start+length]))
	}
	return out, nil
}

// BenchOptions steer one benchmark run.
type BenchOptions struct {
	Chars       int
	Epochs      int
	Seed        int64
	Predictions int    // 0: chars/2 clamped to [100, 20000]
	CorpusPath  string // the sample corpus to build sentences out of
	Workers     int
	Exact       bool
}

// RunBenchmark trains on a synthetic corpus, then predicts, and reports the
// rates - the same keys the Python benchmark reports.
func RunBenchmark(o BenchOptions) (map[string]any, error) {
	if o.Chars < 1 {
		return nil, fmt.Errorf("chars must be >= 1, got %d", o.Chars)
	}
	if o.Epochs < 1 {
		return nil, fmt.Errorf("epochs must be >= 1, got %d", o.Epochs)
	}
	predictions := o.Predictions
	if predictions == 0 {
		predictions = o.Chars / 2
		if predictions < benchMinPredicts {
			predictions = benchMinPredicts
		}
		if predictions > benchMaxPredicts {
			predictions = benchMaxPredicts
		}
	}
	if predictions < 1 {
		return nil, fmt.Errorf("predictions must be >= 1, got %d", predictions)
	}
	texts, err := SyntheticCorpus(o.Chars, o.Seed, BenchLines(o.CorpusPath))
	if err != nil {
		return nil, err
	}
	totalChars := 0
	for _, text := range texts {
		totalChars += len([]rune(text))
	}

	model, err := NewModel(o.Seed, DefaultGraphOptions())
	if err != nil {
		return nil, err
	}
	model.Workers, model.Exact = o.Workers, o.Exact
	started := time.Now()
	// AutoCompress on, like the Python benchmark: compression is part of the
	// work a training pass does, so timing it without would flatter the result.
	records, err := model.Train(texts, TrainOptions{Epochs: o.Epochs, AutoCompress: true})
	if err != nil {
		return nil, err
	}
	trainSeconds := time.Since(started).Seconds()

	prefixes, err := BenchPrefixes(texts, predictions, o.Seed)
	if err != nil {
		return nil, err
	}
	model.G.Prepare()
	expansions := 0
	var sample *Prediction
	started = time.Now()
	for i, prefix := range prefixes {
		result, err := model.Predict(prefix, PredictOptions{Length: BenchPredictLength, Mode: "beam", K: 1})
		if err != nil {
			return nil, err
		}
		expansions += result.Expanded
		if i == 0 {
			sample = result
		}
	}
	predictSeconds := time.Since(started).Seconds()

	transitions := int64(0)
	for _, record := range records {
		transitions += int64(intOf(record["transitions"]))
	}
	g := model.G
	out := map[string]any{
		"seed": o.Seed, "chars": totalChars, "texts": len(texts), "epochs": o.Epochs,
		"train_seconds": trainSeconds, "transitions": transitions,
		"transitions_per_sec": rate(float64(transitions), trainSeconds),
		"chars_per_sec":       rate(float64(totalChars*o.Epochs), trainSeconds),
		"predict_count":       predictions, "predict_seconds": predictSeconds,
		"predictions_per_sec":         rate(float64(predictions), predictSeconds),
		"dijkstra_expansions":         expansions,
		"dijkstra_expansions_per_sec": rate(float64(expansions), predictSeconds),
		"nodes":                       g.NumNodes(),
		"edges":                       g.NumEdges(),
		"trigrams":                    g.NumTrigrams(),
		"compression_ratio":           g.CompressionRatio(),
		"workers":                     model.Workers,
		"counting":                    map[bool]string{true: "exact", false: "racy"}[model.Exact],
	}
	if sample != nil {
		out["sample_prediction"] = map[string]any{
			"prefix": prefixes[0], "continuation": sample.Text, "cost": sample.Cost,
			"reached_end": sample.ReachedEnd,
		}
	}
	return out, nil
}

// rate is count/seconds, or 0 when no time passed at all.
func rate(count, seconds float64) float64 {
	if seconds <= 0 {
		return 0
	}
	return count / seconds
}
