package phonetok

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDataFilesMatchThePythonPackage(t *testing.T) {
	for _, name := range []string{"core.dict", "rules.lts"} {
		theirs, err := os.ReadFile(filepath.Join("..", "..", "phonetok", "data", name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		ours, err := os.ReadFile(filepath.Join("data", name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if string(theirs) != string(ours) {
			t.Errorf("%s differs from the Python package's copy: run `make sync-data`", name)
		}
	}
}

func TestNumbers(t *testing.T) {
	cases := map[string]string{
		"42": "forty two", "3.14": "three point one four", "1,000": "one thousand", "21st": "twenty first",
		"50%": "fifty percent", "$5": "five dollars", "$1": "one dollar", "$2.50": "two dollars fifty cents",
		"-7": "minus seven", "1000000": "one million", "2024": "two thousand twenty four", "0": "zero",
		"100th": "one hundredth", "12th": "twelfth", "mp3": "", "abc": "", "3.1st": "",
	}
	for token, want := range cases {
		if got := NumberWords(token); got != want {
			t.Errorf("NumberWords(%q) = %q, want %q", token, got, want)
		}
	}
	if got := strings.Join(SplitAlphanumeric("b2b"), "|"); got != "b|2|b" {
		t.Errorf("SplitAlphanumeric: %q", got)
	}
}

func TestRulesAndStress(t *testing.T) {
	if RuleCount() < 250 {
		t.Errorf("only %d rules", RuleCount())
	}
	cases := map[string]string{
		"cat": "K AE1 T", "strength": "S T R EH1 NG TH", "nation": "N EY1 SH AH0 N", "tuna": "T UW1 N AH0",
		"rebuild": "R IH0 B IH1 L D", "specific": "S P AH0 S IH1 F IH0 K", "walked": "W AO1 K T",
		"boxes": "B AA1 K S IH0 Z", "physics": "F IH1 Z IH0 K S",
	}
	for word, want := range cases {
		if got := strings.Join(LetterToSound(word), " "); got != want {
			t.Errorf("LetterToSound(%q) = %q, want %q", word, got, want)
		}
	}
}

func TestSyllables(t *testing.T) {
	if OnsetCount() != 77 {
		t.Errorf("%d onsets, want 77", OnsetCount())
	}
	cases := map[string]string{
		"S T R EH1 NG TH S": "S.T.R.EH1.NG.TH.S", "AE1 S T R OW0": "AE1.S | T.R.OW0", "B AH1 T ER0": "B.AH1.T | ER0",
		"B AH0 N AE1 N AH0": "B.AH0 | N.AE1.N | AH0", "K AH0 M P Y UW1 T ER0": "K.AH0.M | P.Y.UW1 | T.ER0", "HH M": "HH.M",
	}
	for phones, want := range cases {
		var parts []string
		for _, s := range Syllabify(strings.Fields(phones), true) {
			parts = append(parts, s.Text())
		}
		if got := strings.Join(parts, " | "); got != want {
			t.Errorf("Syllabify(%q) = %q, want %q", phones, got, want)
		}
	}
	if !Rhymes(strings.Fields("K AE1 T"), strings.Fields("HH AE1 T")) || Rhymes(strings.Fields("K AE1 T"), strings.Fields("K AH1 T")) {
		t.Error("rhymes")
	}
	if !IsLegalCoda(strings.Fields("K S TH S")) || IsLegalCoda(strings.Fields("T L")) {
		t.Error("codas")
	}
}

func TestMersenneTwisterIsPythons(t *testing.T) {
	// random.Random(1).random() and random.Random(0).random(), as CPython prints them
	if got := NewMT(1).Float64(); got != 0.13436424411240122 {
		t.Errorf("seed 1: %v", got)
	}
	if got := NewMT(0).Float64(); got != 0.8444218515250481 {
		t.Errorf("seed 0: %v", got)
	}
	r := NewMT(3)
	// random.Random(3).randrange(7) draws getrandbits(3) until < 7
	if got := r.RandBelow(7); got < 0 || got >= 7 {
		t.Errorf("randrange: %d", got)
	}
}

func TestLexiconAndTranscriber(t *testing.T) {
	lex := CoreLexicon()
	if lex.Len() < 1500 || strings.Join(lex.Lookup("the"), " ") != "DH AH0" || lex.Lookup("qzxv") != nil {
		t.Errorf("the core lexicon: %s", lex.Describe())
	}
	if got := strings.Join(lex.Spellings([]string{"T", "UW1"}, true), " "); got != "to too two" {
		t.Errorf("homophones: %q", got)
	}
	tr := NewTranscriber(lex)
	cases := map[string]string{
		"cats": "K AE1 T S morphology", "boxes": "B AA1 K S IH0 Z morphology", "cities": "S IH1 T IY0 Z morphology",
		"toothbrush": "T UW1 TH B R AH2 SH morphology", "42": "F AO1 R T IY0 T UW1 number",
		"well-known": "W EH1 L N OW1 N joined", "FBI": "EH1 F B IY1 AY1 letters", "zebra": "Z EH1 B R AH0 rules",
		"cat's": "K AE1 T S morphology", "it'll": "IH1 T AH0 L lexicon", "I'm": "AY1 M lexicon", "would've": "W UH1 D AH0 V morphology",
	}
	for word, want := range cases {
		phones, how := NewTranscriber(lex).Explain(word)
		if got := strings.Join(phones, " ") + " " + how; got != want {
			t.Errorf("Explain(%q) = %q, want %q", word, got, want)
		}
	}
	if _, how := tr.Explain("zebra"); how != "rules" {
		t.Errorf("first reading: %s", how)
	}
	if _, how := tr.Explain("zebra"); how != "memory" {
		t.Errorf("second reading: %s", how)
	}
	if got := tr.Spell([]string{"Z", "IH1", "B", "R", "AH0"}); got != "zibra" {
		t.Errorf("respelled: %q", got)
	}
}

func TestSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	tok, _ := NewTokenizer(SyllableLvl, CoreLexicon())
	tok.Encode("the zebra sat", true)
	path := filepath.Join(dir, "tok.json")
	if err := tok.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := LoadTokenizer(path, CoreLexicon())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(back.Vocab.Tokens, " ") != strings.Join(tok.Vocab.Tokens, " ") {
		t.Error("the vocabulary did not survive")
	}
	if got := back.DecodeIDs(back.Encode("zebra", false)); got != "zebra" {
		t.Errorf("decode after load: %q", got)
	}
	other, _ := NewTokenizer(Phoneme, CoreLexicon())
	if err := other.LoadDocument(tok.ToDocument()); err == nil {
		t.Error("a syllable file loaded into a phoneme tokenizer")
	}
}
