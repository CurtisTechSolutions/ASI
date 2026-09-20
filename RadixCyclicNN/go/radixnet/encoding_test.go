package radixnet

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// theEncodings are the ones the tests exercise end to end: the default, a
// bigger n, groups of four and five letters, and word bigrams / trigrams.
var theEncodings = []Encoding{
	{Chars, 3, 1},
	{Chars, 1, 1},
	{Chars, 2, 1},
	{Chars, 5, 1},
	{Chars, 4, 4},
	{Chars, 5, 5},
	{Chars, 6, 3},
	{Words, 1, 1},
	{Words, 2, 1},
	{Words, 3, 1},
	{Words, 2, 2},
}

func TestEncodingDefaultsAndValidation(t *testing.T) {
	if got := (Encoding{}).WithDefaults(); got != DefaultEncoding() {
		t.Fatalf("the zero encoding must be the default one, got %v", got)
	}
	if !DefaultEncoding().IsDefault() || DefaultEncoding().Overlap() != Overlap {
		t.Fatalf("the default encoding is not the trigram: %v", DefaultEncoding())
	}
	for _, bad := range []Encoding{{"rune", 3, 1}, {Chars, 0, 1}, {Chars, 3, 0}, {Chars, 2, 3}} {
		if err := bad.Validate(); err == nil {
			t.Errorf("%v must not validate", bad)
		}
	}
	for _, enc := range theEncodings {
		if err := enc.Validate(); err != nil {
			t.Errorf("%v: %v", enc, err)
		}
	}
}

func TestParseEncoding(t *testing.T) {
	cases := map[string]Encoding{
		"":              {Chars, 3, 1},
		"trigram":       {Chars, 3, 1},
		"char:4":        {Chars, 4, 1},
		"chars:4:4":     {Chars, 4, 4},
		"char:5:groups": {Chars, 5, 5},
		"letters:7:2":   {Chars, 7, 2},
		"word":          {Words, 1, 1},
		"word:2":        {Words, 2, 1},
		"word-trigram":  {Words, 3, 1},
		"WORD:2:2":      {Words, 2, 2},
	}
	for spec, want := range cases {
		got, err := ParseEncoding(spec)
		if err != nil {
			t.Fatalf("%q: %v", spec, err)
		}
		if got != want {
			t.Errorf("%q: got %v, want %v", spec, got, want)
		}
		// every encoding prints as a spec that parses back to itself
		back, err := ParseEncoding(got.String())
		if err != nil || back != got {
			t.Errorf("%q: %v does not round trip through %q (%v)", spec, got, got.String(), err)
		}
	}
	for _, bad := range []string{"rune:3", "char:x", "char:3:y", "char:2:3", "char:0", "a:b:c:d"} {
		if _, err := ParseEncoding(bad); err == nil {
			t.Errorf("%q must not parse", bad)
		}
	}
}

func TestEncodeUnderEveryEncoding(t *testing.T) {
	cases := []struct {
		enc  Encoding
		text string
		want []string
	}{
		{Encoding{Chars, 3, 1}, "hello", []string{"hel", "ell", "llo"}},
		{Encoding{Chars, 1, 1}, "abc", []string{"a", "b", "c"}},
		{Encoding{Chars, 4, 4}, "abcdefghij", []string{"abcd", "efgh"}},
		{Encoding{Chars, 5, 5}, "abcdefghij", []string{"abcde", "fghij"}},
		{Encoding{Chars, 4, 2}, "abcdef", []string{"abcd", "cdef"}},
		{Encoding{Chars, 3, 1}, "hi", nil},
		{Encoding{Words, 2, 1}, "the cat sat down", []string{"the cat", "cat sat", "sat down"}},
		{Encoding{Words, 3, 1}, "the cat sat down", []string{"the cat sat", "cat sat down"}},
		{Encoding{Words, 2, 2}, "the cat sat down here", []string{"the cat", "sat down"}},
		{Encoding{Words, 2, 1}, "  the   cat  ", []string{"the cat"}},
		{Encoding{Words, 2, 1}, "alone", nil},
	}
	for _, c := range cases {
		if got := c.enc.Encode(c.text); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%v.Encode(%q) = %q, want %q", c.enc, c.text, got, c.want)
		}
	}
}

func TestEncodeMatchesTheDefaultFunction(t *testing.T) {
	for _, text := range []string{"", "ab", "hello there", "ünïcode text"} {
		if got, want := DefaultEncoding().Encode(text), Encode(text); !reflect.DeepEqual(got, want) {
			t.Errorf("%q: %q != %q", text, got, want)
		}
	}
}

func TestNormalizeIsWhatComesBack(t *testing.T) {
	cases := []struct {
		enc        Encoding
		text, want string
	}{
		{Encoding{Chars, 3, 1}, "hello", "hello"},
		{Encoding{Chars, 4, 4}, "abcdefghij", "abcdefgh"},
		{Encoding{Chars, 4, 4}, "abc", ""},
		{Encoding{Words, 2, 1}, "  the  cat   sat ", "the cat sat"},
		{Encoding{Words, 2, 2}, "the cat sat down here", "the cat sat down"},
	}
	for _, c := range cases {
		if got := c.enc.Normalize(c.text); got != c.want {
			t.Errorf("%v.Normalize(%q) = %q, want %q", c.enc, c.text, got, c.want)
		}
		// what Normalize keeps is exactly what the grams decode back to
		if grams := c.enc.Encode(c.text); grams != nil {
			if got := c.enc.DecodeGrams(grams); got != c.want {
				t.Errorf("%v: grams of %q decode to %q, want %q", c.enc, c.text, got, c.want)
			}
		}
	}
}

func TestUnitsSliceAndJoin(t *testing.T) {
	for _, enc := range theEncodings {
		u := enc.Units("the quick brown fox jumps")
		if u.Len() != enc.Len(u.Text()) {
			t.Fatalf("%v: Units.Len %d != Len %d", enc, u.Len(), enc.Len(u.Text()))
		}
		if got := u.Slice(0, u.Len()); got != u.Text() {
			t.Errorf("%v: the whole slice is %q, want %q", enc, got, u.Text())
		}
		if got := u.Slice(2, 2); got != "" {
			t.Errorf("%v: an empty range is %q", enc, got)
		}
		parts := make([]string, 0, u.Len())
		for i := 0; i < u.Len(); i++ {
			parts = append(parts, u.At(i))
		}
		if got := enc.Join(parts...); got != u.Text() {
			t.Errorf("%v: joining the units gives %q, want %q", enc, got, u.Text())
		}
	}
}

func TestHasUnitPrefixStopsAtAWordBoundary(t *testing.T) {
	word := Encoding{Words, 2, 1}
	if word.HasUnitPrefix("the cat", "the ca") {
		t.Error("a word encoding must not match half a word")
	}
	if !word.HasUnitPrefix("the cat", "the") || !word.HasUnitPrefix("the cat", "the cat") {
		t.Error("a word encoding must match whole words")
	}
	if !DefaultEncoding().HasUnitPrefix("the cat", "the ca") {
		t.Error("a character encoding matches any prefix")
	}
}

// -- the graph under every encoding -------------------------------------------------

func encodingCorpus(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "data", "sample_corpus.txt"))
	if err != nil {
		t.Skipf("sample corpus not found: %v", err)
	}
	texts := SplitTexts(string(raw), "lines", 0)
	if len(texts) > 40 {
		texts = texts[:40]
	}
	return texts
}

func modelWith(t *testing.T, enc Encoding, texts []string, epochs int) *Model {
	t.Helper()
	opts := DefaultGraphOptions()
	opts.Encoding = enc
	m, err := NewModel(1, opts)
	if err != nil {
		t.Fatalf("%v: %v", enc, err)
	}
	m.Exact = true
	topts := DefaultTrainOptions()
	topts.Epochs = epochs
	if _, err := m.Train(texts, topts); err != nil {
		t.Fatalf("%v: %v", enc, err)
	}
	return m
}

// Every encoding has to hold the invariants the trigram does: the index, the
// overlaps, and a round trip of every text it was trained on.
func TestGraphRoundTripsUnderEveryEncoding(t *testing.T) {
	texts := encodingCorpus(t)
	for _, enc := range theEncodings {
		enc := enc
		t.Run(enc.String(), func(t *testing.T) {
			m := modelWith(t, enc, texts, 2)
			if err := m.G.CheckInvariants(texts, false); err != nil {
				t.Fatalf("invariants before compression: %v", err)
			}
			m.G.Compress()
			if err := m.G.CheckInvariants(texts, true); err != nil {
				t.Fatalf("invariants after compression: %v", err)
			}
			for _, text := range texts {
				want := enc.Normalize(text)
				if want == "" {
					continue
				}
				grams := enc.Encode(text)
				path, ok := m.G.NodePath(grams)
				if !ok {
					t.Fatalf("%q does not walk through the graph", text)
				}
				labels := make([]string, 0, len(path))
				for _, id := range path[1 : len(path)-1] {
					labels = append(labels, m.G.Label(id))
				}
				if got := m.G.DecodePath(labels, 0, true); got != want {
					t.Fatalf("round trip of %q gave %q", want, got)
				}
			}
		})
	}
}

func TestTrainPredictAndScoreUnderEveryEncoding(t *testing.T) {
	texts := encodingCorpus(t)
	for _, enc := range theEncodings {
		enc := enc
		t.Run(enc.String(), func(t *testing.T) {
			m := modelWith(t, enc, texts, 2)
			if m.G.NumGrams() == 0 || m.G.NumNodes() <= First {
				t.Fatalf("nothing was learned: %d grams, %d nodes", m.G.NumGrams(), m.G.NumNodes())
			}
			if got := m.Stats()["encoding"]; got != enc.String() {
				t.Errorf("stats report the encoding as %v, want %v", got, enc.String())
			}
			// a prefix of the corpus predicts something, and scoring a trained
			// text beats scoring noise
			prefix := enc.Normalize(texts[0])
			if u := enc.Units(prefix); u.Len() > enc.N {
				prefix = u.Slice(0, u.Len()-1)
			}
			out, err := m.Predict(prefix, PredictOptions{Length: 10, Mode: "beam", K: 3})
			if err != nil {
				t.Fatalf("predict: %v", err)
			}
			if len(out.Top) == 0 {
				t.Fatalf("no continuation for %q", prefix)
			}
			known := m.Score(texts[0])
			noise := m.Score("qzx wqzj vbn qzx wqzj vbn")
			if known.UnknownTransitions > noise.UnknownTransitions {
				t.Errorf("a trained text is stranger than noise: %+v vs %+v", known, noise)
			}
			if _, err := m.Generate(GenerateOptions{MaxLength: 20, Mode: "beam", Count: 1}); err != nil {
				t.Fatalf("generate: %v", err)
			}
		})
	}
}

// A word encoding predicts whole words, and a grouping one whole groups.
func TestWordEncodingWorksInWords(t *testing.T) {
	texts := []string{
		"the cat sat on the mat",
		"the cat sat on the floor",
		"the dog sat on the mat",
	}
	m := modelWith(t, Encoding{Words, 2, 1}, texts, 3)
	out, err := m.Predict("the cat sat", PredictOptions{Length: 3, Mode: "beam", K: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Top) == 0 {
		t.Fatal("no continuation")
	}
	for _, r := range out.Top {
		for _, word := range strings.Fields(r.Text) {
			if !strings.Contains(strings.Join(texts, " "), word) {
				t.Errorf("predicted %q, which is not a word of the corpus (%q)", word, r.Text)
			}
		}
		// every label of the walk is a whole number of words
		for _, label := range r.Labels {
			if label == StartLabel || label == EndLabel || label == BackLabel {
				continue
			}
			if strings.TrimSpace(label) != label || strings.Contains(label, "  ") {
				t.Errorf("label %q is not a clean word sequence", label)
			}
		}
	}
	if err := m.G.CheckInvariants(texts, false); err != nil {
		t.Fatal(err)
	}
}

func TestGroupEncodingHasNoOverlap(t *testing.T) {
	enc := Encoding{Chars, 4, 4}
	if enc.Overlap() != 0 || enc.Sliding() {
		t.Fatalf("groups of four must not overlap: %+v", enc)
	}
	texts := []string{"abcdefghijkl", "abcdmnopijkl"}
	m := modelWith(t, enc, texts, 2)
	for gram := range m.G.index {
		if enc.Len(gram) != 4 {
			t.Errorf("gram %q is not four characters", gram)
		}
	}
	if err := m.G.CheckInvariants(texts, false); err != nil {
		t.Fatal(err)
	}
	m.G.Compress()
	if err := m.G.CheckInvariants(texts, true); err != nil {
		t.Fatal(err)
	}
}

// The encoding travels with the model file, and only when it is not the default.
func TestEncodingSurvivesASaveAndIsWrittenOnlyWhenItHasTo(t *testing.T) {
	texts := encodingCorpus(t)
	dir := t.TempDir()
	for _, enc := range theEncodings {
		enc := enc
		t.Run(enc.String(), func(t *testing.T) {
			m := modelWith(t, enc, texts, 1)
			path := filepath.Join(dir, strings.ReplaceAll(enc.String(), ":", "-")+".json")
			if err := m.Save(path); err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Contains(string(raw), `"encoding"`); got != !enc.IsDefault() {
				t.Errorf("the file %s an encoding block, for %v", map[bool]string{true: "carries", false: "does not carry"}[got], enc)
			}
			loaded, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if loaded.Encoding() != enc {
				t.Fatalf("loaded as %v, saved as %v", loaded.Encoding(), enc)
			}
			if err := loaded.G.CheckInvariants(texts, false); err != nil {
				t.Fatalf("a reloaded graph broke: %v", err)
			}
			if loaded.G.NumGrams() != m.G.NumGrams() || loaded.G.NumNodes() != m.G.NumNodes() {
				t.Errorf("the reloaded graph is not the saved one")
			}
		})
	}
}

// The diff marks units, so a word correction blames whole words.
func TestDiffFollowsTheEncoding(t *testing.T) {
	word := Encoding{Words, 2, 1}
	spans, _ := word.ChangedSpans("the cat sat down", "the dog sat down")
	if len(spans) != 1 || spans[0] != (Span{1, 2}) {
		t.Fatalf("a word diff must mark word 1, got %v", spans)
	}
	edits := word.DiffSummary("the cat sat down", "the dog sat down", 0)
	if len(edits) != 1 || edits[0].Wrong != "cat" || edits[0].Right != "dog" {
		t.Fatalf("got %+v", edits)
	}
	// the character diff is untouched: same text, character spans
	charSpans, _ := ChangedSpans("the cat sat down", "the dog sat down")
	if len(charSpans) != 1 || charSpans[0].Lo != 4 {
		t.Fatalf("the character diff changed: %v", charSpans)
	}
}

func TestCorrectMovesWordStepsUnderAWordEncoding(t *testing.T) {
	texts := []string{"the cat sat on the mat", "the cat ran on the mat"}
	m := modelWith(t, Encoding{Words, 2, 1}, texts, 2)
	out, err := m.Correct("the cat sat on the mat", "the cat slept on the mat", CorrectOptions{Strength: 1, Weight: 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Edits == 0 || out.Penalised == 0 {
		t.Fatalf("a word correction moved nothing: %+v", out)
	}
	if out.WrongChars != 1 || out.RightChars != 1 {
		t.Errorf("one word changed, got wrong=%d right=%d", out.WrongChars, out.RightChars)
	}
}
