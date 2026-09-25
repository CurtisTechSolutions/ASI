package radixnet

// The word view of a word encoding: the alphabet the graph has read, and what
// every number of the model is counted in.
//
// A word encoding has no vocabulary object - a gram is text - so the alphabet
// is derived from the gram index, which is the one count compression cannot
// change.

import (
	"strings"
	"testing"
)

func wordModel(t *testing.T, n int, texts ...string) *Model {
	t.Helper()
	opts := DefaultGraphOptions()
	opts.Encoding = Encoding{Unit: Words, N: n, Stride: 1}
	m, err := NewModel(0, opts)
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	m.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if len(texts) > 0 {
		if _, err := m.Train(texts, TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
			t.Fatalf("train: %v", err)
		}
	}
	return m
}

func TestUnitsNameFollowsTheEncoding(t *testing.T) {
	if got := DefaultEncoding().UnitsName(); got != "chars" {
		t.Fatalf("the default encoding counts in %q", got)
	}
	if got := (Encoding{Unit: Words, N: 3, Stride: 1}).UnitsName(); got != "words" {
		t.Fatalf("a word encoding counts in %q", got)
	}
	// and a model says so in its stats, because a per-word number read as
	// per-character is read wrong
	m := wordModel(t, 3, "the cat sat on the mat")
	if got := m.Stats()["units"]; got != "words" {
		t.Fatalf("stats units = %v", got)
	}
	plain, _ := NewModel(0, DefaultGraphOptions())
	if got := plain.Stats()["units"]; got != "chars" {
		t.Fatalf("a character model's stats units = %v", got)
	}
}

func TestVocabularyIsWhatTheGramsAreMadeOf(t *testing.T) {
	enc := Encoding{Unit: Words, N: 3, Stride: 1}
	counts := enc.Vocabulary([]string{"the cat sat", "cat sat on"})
	want := map[string]int{"the": 1, "cat": 2, "sat": 2, "on": 1}
	if len(counts) != len(want) {
		t.Fatalf("counted %v, want %v", counts, want)
	}
	for word, n := range want {
		if counts[word] != n {
			t.Fatalf("%q counted %d, want %d (%v)", word, counts[word], n, counts)
		}
	}
	// a character encoding counts characters, which is what its units are
	if got := DefaultEncoding().Vocabulary([]string{"the"}); len(got) != 3 {
		t.Fatalf("a character encoding's units are characters: %v", got)
	}
}

func TestTopWordsIsTheAlphabetMostReadFirst(t *testing.T) {
	m := wordModel(t, 3,
		"the cat sat on the mat", "the cat sat on the log", "the dog sat on the mat")
	rows := m.TopWords(4)
	if len(rows) != 4 {
		t.Fatalf("asked for 4 words, got %d", len(rows))
	}
	// the id is the rank, because the graph has no vocabulary to number from
	for i, row := range rows {
		if row.ID != i {
			t.Fatalf("row %d carries id %d", i, row.ID)
		}
	}
	for i := 1; i < len(rows); i++ {
		if rows[i].Grams > rows[i-1].Grams {
			t.Fatalf("rows are not most read first: %+v", rows)
		}
	}
	// Trigrams is Grams under the name a word-model client knew it by
	for _, row := range rows {
		if row.Trigrams != row.Grams {
			t.Fatalf("%+v: the two counts disagree", row)
		}
	}
	// limit 0 is the whole alphabet, and every word of the corpus is in it
	all := m.TopWords(0)
	seen := map[string]bool{}
	for _, row := range all {
		seen[row.Word] = true
	}
	for _, word := range strings.Fields("the cat sat on mat log dog") {
		if !seen[word] {
			t.Fatalf("%q was read but is not in the alphabet: %+v", word, all)
		}
	}
	// a character model has no alphabet of words
	plain, _ := NewModel(0, DefaultGraphOptions())
	if got := plain.TopWords(5); got != nil {
		t.Fatalf("a character model listed words: %+v", got)
	}
}

func TestAWordModelCountsAndPredictsInWords(t *testing.T) {
	m := wordModel(t, 3,
		"the cat sat on the mat", "the cat sat on the log", "the dog sat on the mat")
	o := DefaultPredictOptions()
	o.Length = 2
	found, err := m.Predict("the cat sat on", o)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	// two words on, not two characters
	if text := found.FullText; text != "the cat sat on the mat" && text != "the cat sat on the log" {
		t.Fatalf("predicted %q", text)
	}
	score := m.Score("the cat sat on the mat")
	if score.Chars != 6 {
		t.Fatalf("a six-word text scored over %d units", score.Chars)
	}
}
