package radixnet

import (
	"strings"
	"testing"
)

func TestBenchLinesFallBackWhenTheFileIsGone(t *testing.T) {
	lines := BenchLines("/definitely/not/a/corpus.txt")
	if len(lines) == 0 {
		t.Fatal("the fallback list must not be empty")
	}
	for _, line := range lines {
		if len([]rune(line)) < 3 {
			t.Errorf("every base line is at least three characters: %q", line)
		}
	}
}

func TestSyntheticCorpusIsDeterministicAndLongEnough(t *testing.T) {
	base := []string{"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill"}
	first, err := SyntheticCorpus(500, 7, base)
	if err != nil {
		t.Fatalf("SyntheticCorpus: %v", err)
	}
	second, err := SyntheticCorpus(500, 7, base)
	if err != nil {
		t.Fatalf("SyntheticCorpus: %v", err)
	}
	if strings.Join(first, "|") != strings.Join(second, "|") {
		t.Error("the same (chars, seed, lines) must give the same corpus")
	}
	total := 0
	for _, text := range first {
		if len([]rune(text)) < 3 {
			t.Errorf("every text is at least three characters: %q", text)
		}
		total += len([]rune(text))
	}
	if total < 500 {
		t.Errorf("the corpus must reach the asked-for size: %d", total)
	}
	if other, _ := SyntheticCorpus(500, 8, base); strings.Join(other, "|") == strings.Join(first, "|") {
		t.Error("a different seed must give a different corpus")
	}
}

func TestSyntheticCorpusEdges(t *testing.T) {
	base := []string{"the cat sat on the mat"}
	if texts, err := SyntheticCorpus(0, 1, base); err != nil || len(texts) != 0 {
		t.Errorf("zero characters give no texts: %v %v", texts, err)
	}
	if _, err := SyntheticCorpus(-1, 1, base); err == nil {
		t.Error("a negative size must be refused")
	}
	if _, err := SyntheticCorpus(10, 1, []string{"ab"}); err == nil {
		t.Error("base lines shorter than a trigram must be refused")
	}
}

func TestBenchPrefixesComeOutOfTheCorpus(t *testing.T) {
	texts := []string{"the cat sat on the mat", "the dog sat on the log"}
	prefixes, err := BenchPrefixes(texts, 20, 3)
	if err != nil {
		t.Fatalf("BenchPrefixes: %v", err)
	}
	if len(prefixes) != 20 {
		t.Fatalf("expected 20 prefixes, got %d", len(prefixes))
	}
	for _, prefix := range prefixes {
		if n := len([]rune(prefix)); n < benchMinPrefix || n > benchMaxPrefix {
			t.Errorf("prefix %q is %d characters", prefix, n)
		}
		found := false
		for _, text := range texts {
			if strings.Contains(text, prefix) {
				found = true
			}
		}
		if !found {
			t.Errorf("every prefix must occur in the corpus: %q", prefix)
		}
	}
	if _, err := BenchPrefixes(nil, 1, 1); err == nil {
		t.Error("cutting prefixes out of nothing must be refused")
	}
}

func TestRunBenchmarkReportsTheRates(t *testing.T) {
	result, err := RunBenchmark(BenchOptions{Chars: 2000, Epochs: 2, Seed: 1, Predictions: 20, Exact: true})
	if err != nil {
		t.Fatalf("RunBenchmark: %v", err)
	}
	for _, key := range []string{
		"chars", "texts", "epochs", "train_seconds", "transitions", "transitions_per_sec", "chars_per_sec",
		"predict_count", "predict_seconds", "predictions_per_sec", "dijkstra_expansions",
		"dijkstra_expansions_per_sec", "nodes", "edges", "trigrams", "compression_ratio", "sample_prediction",
	} {
		if _, ok := result[key]; !ok {
			t.Errorf("the report is missing %q", key)
		}
	}
	if result["predict_count"] != 20 {
		t.Errorf("predict_count: %v", result["predict_count"])
	}
	if transitions, _ := result["transitions"].(int64); transitions <= 0 {
		t.Errorf("transitions: %v", result["transitions"])
	}
	if ratio, _ := result["compression_ratio"].(float64); ratio < 1 {
		t.Errorf("the benchmark must compress as it trains: ratio %v", ratio)
	}
	sample, _ := result["sample_prediction"].(map[string]any)
	if sample == nil || sample["prefix"] == "" {
		t.Errorf("sample_prediction: %v", result["sample_prediction"])
	}
}

func TestRunBenchmarkRefusesNonsense(t *testing.T) {
	for name, o := range map[string]BenchOptions{
		"chars":       {Chars: 0, Epochs: 1},
		"epochs":      {Chars: 100, Epochs: 0},
		"predictions": {Chars: 100, Epochs: 1, Predictions: -1},
	} {
		if _, err := RunBenchmark(o); err == nil {
			t.Errorf("%s: expected the options to be refused", name)
		}
	}
}

func TestRateIsZeroWhenNoTimePassed(t *testing.T) {
	if rate(10, 0) != 0 {
		t.Error("a zero elapsed time must not divide")
	}
	if rate(10, 2) != 5 {
		t.Error("rate is count / seconds")
	}
}
