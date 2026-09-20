package radixnet

import (
	"strings"
	"testing"
)

// The word n-gram model over the same graph (../../SPEC-WordNGrams.md §10):
// the alphabet, the encoder, the graph, compression, prediction and the file.
// The Python side of the parity is ../../tests/test_go_parity.py.

func wordModel(t *testing.T, texts []string, epochs int) *Model {
	t.Helper()
	opts := DefaultGraphOptions()
	opts.Words = true
	m, err := NewModel(0, opts)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	m.Exact = true
	if _, err := m.Train(texts, TrainOptions{Epochs: epochs, AutoCompress: true}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	return m
}

var wordTexts = []string{
	"the cat sat on the mat",
	"the cat sat on the rug",
	"the dog sat on the mat",
	"the dog ate the bone in the garden",
	"a bird sang in the garden",
}

func TestWordSymbolRoundTrip(t *testing.T) {
	for _, id := range []int{0, 1, 2, 1000, 55039, 55040, 55041, MaxWords - 1} {
		symbol, err := WordSymbol(id)
		if err != nil {
			t.Fatalf("WordSymbol(%d): %v", id, err)
		}
		back, err := SymbolWord(symbol)
		if err != nil || back != id {
			t.Fatalf("SymbolWord(WordSymbol(%d)) = %d, %v", id, back, err)
		}
	}
	if _, err := WordSymbol(MaxWords); err == nil {
		t.Error("the cap must be refused")
	}
	if MaxWords != 1_111_808 {
		t.Errorf("MaxWords = %d", MaxWords)
	}
}

func TestWordSymbolsSkipTheSurrogates(t *testing.T) {
	below, _ := WordSymbol(SurrogateLo - WordBase - 1)
	above, _ := WordSymbol(SurrogateLo - WordBase)
	if below != SurrogateLo-1 || above != SurrogateHi+1 {
		t.Fatalf("around the surrogate block: U+%04X then U+%04X", below, above)
	}
	for code := rune(SurrogateLo); code <= SurrogateHi; code++ {
		if _, err := SymbolWord(code); err == nil {
			t.Fatalf("U+%04X carries no word", code)
		}
	}
	last, _ := WordSymbol(MaxWords - 1)
	if last != 0x10FFFF {
		t.Errorf("the last word symbol is U+%04X", last)
	}
	for _, ch := range []rune{'a', ' ', '<', 0x00FF} {
		if _, err := SymbolWord(ch); err == nil {
			t.Errorf("%q is not a word symbol", ch)
		}
	}
}

func TestSplitWordsIsTheWholeTokeniser(t *testing.T) {
	got := SplitWords("The mat. the mat")
	want := []string{"The", "mat.", "the", "mat"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("SplitWords = %v", got)
	}
	// the rule is the Unicode White_Space property, which U+001C..U+001F are not part of
	if out := SplitWords("a\x1cb"); len(out) != 1 || out[0] != "a\x1cb" {
		t.Errorf("SplitWords(a\\x1cb) = %v", out)
	}
	if out := SplitWords("a\tb\nc\r\nd\x0be\x0cf g\u0085h"); len(out) != 8 {
		t.Errorf("SplitWords over every whitespace = %v", out)
	}
	if out := SplitWords("   "); len(out) != 0 {
		t.Errorf("SplitWords(spaces) = %v", out)
	}
}

func TestVocabularyOrderAndUnknown(t *testing.T) {
	v := NewVocabulary()
	if v.Len() != 1 || v.Words[0] != UnknownWord || v.ID("never read") != UnknownID {
		t.Fatalf("a fresh vocabulary is %v", v.Words)
	}
	symbols := v.Encode("the cat sat on the mat", true)
	if v.Len() != 6 {
		t.Errorf("vocabulary = %v", v.Words)
	}
	if got := v.Decode(symbols); got != "the cat sat on the mat" {
		t.Errorf("round trip = %q", got)
	}
	if got := v.Decode(v.Encode("the qux sat", false)); got != "the <unk> sat" {
		t.Errorf("an unread word is %q", got)
	}
	if v.Len() != 6 {
		t.Error("encoding without grow must not grow the vocabulary")
	}
	back, err := VocabularyFrom(v.List())
	if err != nil || strings.Join(back.List(), "|") != strings.Join(v.List(), "|") {
		t.Fatalf("VocabularyFrom: %v / %v", back, err)
	}
	if _, err := VocabularyFrom([]string{"the", "cat"}); err == nil {
		t.Error("a vocabulary starts with the unknown word")
	}
	if _, err := VocabularyFrom([]string{UnknownWord, "the", "the"}); err == nil {
		t.Error("a vocabulary holds no word twice")
	}
}

func TestWordRoundTripNormalisesWhitespace(t *testing.T) {
	m := wordModel(t, wordTexts, 1)
	if got := m.Normalise("a  b\n c"); got != "a b c" {
		t.Errorf("Normalise = %q", got)
	}
	if got := m.Words(m.Symbols("the  cat\tsat", false)); got != "the cat sat" {
		t.Errorf("round trip = %q", got)
	}
}

func TestWordGraphCompressesPhrases(t *testing.T) {
	m := wordModel(t, []string{"the cat sat on the mat", "a dog sat on the mat"}, 1)
	found := map[string]bool{}
	for node := 0; node < m.G.NumNodeIDs(); node++ {
		if m.G.Alive[node] {
			found[m.G.TextOf(m.G.Label(node))] = true
		}
	}
	if !found["sat on the mat"] || !found["the cat sat on"] {
		t.Errorf("a repeated phrase is one node; labels are %v", found)
	}
	if err := m.G.CheckInvariants(nil, true); err != nil {
		t.Errorf("invariants: %v", err)
	}
}

func TestWordPredictionAndScore(t *testing.T) {
	m := wordModel(t, wordTexts, 3)
	if m.Kind() != "word" || m.Units() != WordUnits {
		t.Fatalf("kind %q units %q", m.Kind(), m.Units())
	}
	p, err := m.Predict("the cat sat on", PredictOptions{Length: 2, K: 3, Mode: "beam"})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if p.Text != "the mat" && p.Text != "the rug" {
		t.Errorf("continuation = %q", p.Text)
	}
	if p.FullText != "the cat sat on "+p.Text {
		t.Errorf("full text = %q", p.FullText)
	}
	for _, label := range p.Labels {
		if strings.ContainsRune(label, WordBase) {
			t.Errorf("a reported label is words, got %q", label)
		}
	}
	known := m.Score("the cat sat on the mat")
	unknown := m.Score("the cat sat on the flurb")
	if known.Chars != 6 {
		t.Errorf("Chars counts words, got %d", known.Chars)
	}
	if known.UnknownTransitions != 0 || unknown.UnknownTransitions == 0 {
		t.Errorf("an unread word is an unknown transition: %+v / %+v", known, unknown)
	}
	if unknown.LogProb >= known.LogProb {
		t.Errorf("an unread word costs something: %v vs %v", unknown.LogProb, known.LogProb)
	}
	// two different unread words are the same symbol, so they score identically
	if m.Score("the qux sat") != m.Score("the quux sat") {
		t.Error("two unread words are one symbol")
	}
	gen, err := m.Generate(GenerateOptions{Count: 2, Mode: "beam", MaxLength: 12})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for _, r := range gen {
		if r.Text != r.FullText || strings.Contains(r.Text, "  ") || r.Text != m.Normalise(r.Text) {
			t.Errorf("a generated text is whole and spaced, got %q", r.Text)
		}
	}
}

func TestWordCorrectionIsWordByWord(t *testing.T) {
	m := wordModel(t, wordTexts, 3)
	out, err := m.Correct("the dog sat on the mat", "the dog sat on the rug", CorrectOptions{Strength: 1, Weight: 1, Reward: 1})
	if err != nil {
		t.Fatalf("Correct: %v", err)
	}
	if out.WrongChars != 1 || out.RightChars != 1 {
		t.Errorf("one word differs: %+v", out)
	}
	if len(out.Changes) != 1 || out.Changes[0].Wrong != "mat" || out.Changes[0].Right != "rug" {
		t.Errorf("the change is reported in words: %+v", out.Changes)
	}
}

func TestWordModelFileRoundTrip(t *testing.T) {
	m := wordModel(t, wordTexts, 2)
	if _, err := m.Reward([]string{"the cat sat on the mat"}, 1, 1.5); err != nil {
		t.Fatalf("Reward: %v", err)
	}
	doc := m.ToDoc()
	if doc.Format != WordFormat || doc.Kind != "word" {
		t.Fatalf("format %q kind %q", doc.Format, doc.Kind)
	}
	if doc.Graph.Units != WordUnits || strings.Join(doc.Graph.Vocabulary, "|") != strings.Join(m.Vocabulary().List(), "|") {
		t.Fatalf("the file carries the alphabet: %q %v", doc.Graph.Units, doc.Graph.Vocabulary)
	}
	again, err := FromDoc(doc)
	if err != nil {
		t.Fatalf("FromDoc: %v", err)
	}
	if !again.IsWords() || strings.Join(again.Vocabulary().List(), "|") != strings.Join(m.Vocabulary().List(), "|") {
		t.Fatal("the vocabulary comes back in order")
	}
	before, _ := m.Predict("the cat sat on", PredictOptions{Length: 2, K: 3, Mode: "beam"})
	after, _ := again.Predict("the cat sat on", PredictOptions{Length: 2, K: 3, Mode: "beam"})
	if before.Text != after.Text || before.Cost != after.Cost {
		t.Errorf("a reloaded model predicts the same: %q %v vs %q %v", before.Text, before.Cost, after.Text, after.Cost)
	}
	if m.Score("the dog sat on the mat") != again.Score("the dog sat on the mat") {
		t.Error("a reloaded model scores the same")
	}
}

func TestAWordFileIsNotACountFile(t *testing.T) {
	word := wordModel(t, wordTexts, 1).ToDoc()
	word.Format = ModelFormat // a count reader would read word symbols as text
	if _, err := FromDoc(word); err == nil {
		t.Error("a count document with a word graph must be refused")
	}
	count, err := NewModel(0, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := count.Train(wordTexts, TrainOptions{Epochs: 1, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	doc := count.ToDoc()
	doc.Format = WordFormat
	if _, err := FromDoc(doc); err == nil {
		t.Error("a word document without a word graph must be refused")
	}
}

func TestWordStatsNameTheUnits(t *testing.T) {
	m := wordModel(t, wordTexts, 1)
	stats := m.Stats()
	if stats["units"] != WordUnits || stats["kind"] != "word" {
		t.Errorf("stats = %v / %v", stats["kind"], stats["units"])
	}
	if stats["vocabulary"] != m.Vocabulary().Len() {
		t.Errorf("vocabulary = %v", stats["vocabulary"])
	}
	// trained_chars counts the model's symbols, which are words here
	if got := m.MetaInt("trained_chars"); got != 32 {
		t.Errorf("trained words = %d", got)
	}
	rows := m.TopWords(3)
	if len(rows) != 3 || rows[0].Word != "the" {
		t.Errorf("top words = %+v", rows)
	}
}
