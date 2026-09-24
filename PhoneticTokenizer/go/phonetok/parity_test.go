package phonetok

// The cross-port contract: ../../tests/parity.json is what the Python package
// says about a battery of texts, and this port must say the same.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type parityRow struct {
	Text    string   `json:"text"`
	Tokens  []string `json:"tokens"`
	Kinds   []string `json:"kinds"`
	Words   []int    `json:"words"`
	IDs     []int    `json:"ids"`
	Decoded string   `json:"decoded"`
}

type parityLevel struct {
	Rows  []parityRow `json:"rows"`
	Vocab []string    `json:"vocab"`
	Bare  []string    `json:"bare"`
}

type parityExplain struct {
	Word      string   `json:"word"`
	Phones    []string `json:"phones"`
	How       string   `json:"how"`
	Syllables []string `json:"syllables"`
	IPA       string   `json:"ipa"`
}

type parityBlend struct {
	Phones   []string `json:"phones"`
	Kept     int      `json:"kept"`
	Dropped  int      `json:"dropped"`
	Spelling string   `json:"spelling"`
}

type parityDoc struct {
	Symbols          []string               `json:"symbols"`
	Levels           map[string]parityLevel `json:"levels"`
	Explain          [][]parityExplain      `json:"explain"`
	Syllables        map[string][]string    `json:"syllables"`
	Rules            map[string][]string    `json:"rules"`
	Respell          map[string]string      `json:"respell"`
	Features         map[string][]float64   `json:"features"`
	IPA              map[string]string      `json:"ipa"`
	Affinity         map[string][]float64   `json:"affinity"`
	Score            map[string]float64     `json:"score"`
	Next             map[string][][2]any    `json:"next"`
	Coin             map[string][]string    `json:"coin"`
	Blend            map[string]parityBlend `json:"blend"`
	TranscriberSpell map[string]string      `json:"transcriber_spell"`
}

func loadParity(t *testing.T) parityDoc {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "tests", "parity.json"))
	if err != nil {
		t.Fatalf("the parity fixture: %v", err)
	}
	var doc parityDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("the parity fixture: %v", err)
	}
	return doc
}

func joined(items []string) string { return strings.Join(items, " ") }

func TestParityAlphabet(t *testing.T) {
	doc := loadParity(t)
	if joined(doc.Symbols) != joined(Symbols) {
		t.Fatalf("the alphabet differs:\n%v\n%v", doc.Symbols, Symbols)
	}
}

func TestParityLevels(t *testing.T) {
	doc := loadParity(t)
	texts := []string{}
	for _, row := range doc.Levels[Phoneme].Rows {
		texts = append(texts, row.Text)
	}
	for _, level := range Levels {
		want := doc.Levels[level]
		tok, err := NewTokenizer(level, CoreLexicon())
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range want.Rows {
			toks := tok.Tokenize(row.Text)
			texts, kinds, words := []string{}, []string{}, []int{}
			for _, tk := range toks {
				texts = append(texts, tk.Text)
				kinds = append(kinds, tk.Kind)
				words = append(words, tk.Word)
			}
			if joined(texts) != joined(row.Tokens) {
				t.Errorf("%s %d: tokens\n got %q\nwant %q", level, i, joined(texts), joined(row.Tokens))
			}
			if joined(kinds) != joined(row.Kinds) {
				t.Errorf("%s %d: kinds %v want %v", level, i, kinds, row.Kinds)
			}
			if !equalInts(words, row.Words) {
				t.Errorf("%s %d: words %v want %v", level, i, words, row.Words)
			}
			ids := tok.Encode(row.Text, true)
			if !equalInts(ids, row.IDs) {
				t.Errorf("%s %d: ids %v want %v", level, i, ids, row.IDs)
			}
			if got := tok.Decode(tok.Tokens(row.Text)); got != row.Decoded {
				t.Errorf("%s %d: decoded %q want %q", level, i, got, row.Decoded)
			}
			// idempotent
			form := tok.Text(row.Text)
			if tok.Text(form) != form {
				t.Errorf("%s %d: not idempotent: %q -> %q", level, i, form, tok.Text(form))
			}
		}
		if joined(tok.Vocab.Tokens) != joined(want.Vocab) {
			t.Errorf("%s: the vocabulary differs after reading the battery", level)
		}
		bare, err := NewTokenizer(level, CoreLexicon())
		if err != nil {
			t.Fatal(err)
		}
		bare.Stress, bare.Boundaries, bare.PausesOn = false, false, false
		for i, text := range texts {
			if got := bare.Text(text); got != want.Bare[i] {
				t.Errorf("%s bare %d: %q want %q", level, i, got, want.Bare[i])
			}
		}
	}
}

func TestParityWords(t *testing.T) {
	doc := loadParity(t)
	tok, _ := NewTokenizer(Phoneme, CoreLexicon())
	for i, row := range doc.Levels[Phoneme].Rows {
		got := tok.Explain(row.Text)
		want := doc.Explain[i]
		if len(got) != len(want) {
			t.Errorf("explain %d: %d rows want %d", i, len(got), len(want))
			continue
		}
		for k := range want {
			g, w := got[k], want[k]
			if g.Word != w.Word || joined(g.Phones) != joined(w.Phones) || g.How != w.How ||
				joined(g.Syllables) != joined(w.Syllables) || g.IPA != w.IPA {
				t.Errorf("explain %d/%d: got %+v want %+v", i, k, g, w)
			}
		}
		if got := tok.IPA(row.Text); got != doc.IPA[row.Text] {
			t.Errorf("ipa %q: %q want %q", row.Text, got, doc.IPA[row.Text])
		}
	}
	for word, want := range doc.Syllables {
		var got []string
		for _, s := range Syllabify(tok.Pronounce(word), true) {
			got = append(got, s.Text())
		}
		if joined(got) != joined(want) {
			t.Errorf("syllables %s: %v want %v", word, got, want)
		}
	}
	for word, want := range doc.Rules {
		if got := LetterToSound(word); joined(got) != joined(want) {
			t.Errorf("rules %s: %v want %v", word, got, want)
		}
	}
	for phones, want := range doc.Respell {
		if got := Respell(strings.Fields(phones)); got != want {
			t.Errorf("respell %s: %q want %q", phones, got, want)
		}
	}
	for phone, want := range doc.Features {
		got, err := Features(phone)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-12 {
				t.Errorf("features %s[%d]: %v want %v", phone, i, got[i], want[i])
			}
		}
	}
	tr := NewTranscriber(CoreLexicon())
	for _, w := range []string{"two", "two", "right", "zebra"} {
		tr.Word(w)
	}
	for phones, want := range doc.TranscriberSpell {
		if got := tr.Spell(strings.Fields(phones)); got != want {
			t.Errorf("spell %s: %q want %q", phones, got, want)
		}
	}
}

func TestParityPhonotactics(t *testing.T) {
	doc := loadParity(t)
	lex := CoreLexicon()
	p := PhonotacticsFromLexicon(lex, false)
	for pair, want := range doc.Affinity {
		ab := strings.Fields(pair)
		if got := p.Affinity(ab[0], ab[1]); math.Abs(got-want[0]) > 1e-9 {
			t.Errorf("affinity %s: %v want %v", pair, got, want[0])
		}
		if got := p.Prob(ab[0], ab[1]); math.Abs(got-want[1]) > 1e-12 {
			t.Errorf("prob %s: %v want %v", pair, got, want[1])
		}
	}
	for phones, want := range doc.Score {
		if got := p.Score(strings.Fields(phones)); math.Abs(got-want) > 1e-9 {
			t.Errorf("score %s: %v want %v", phones, got, want)
		}
	}
	for a, want := range doc.Next {
		got := p.Next(a, 5)
		if len(got) != len(want) {
			t.Errorf("next %s: %v want %v", a, got, want)
			continue
		}
		for i := range want {
			if got[i].Phone != want[i][0].(string) || math.Abs(got[i].P-want[i][1].(float64)) > 1e-12 {
				t.Errorf("next %s[%d]: %v want %v", a, i, got[i], want[i])
			}
		}
	}
	tok, _ := NewTokenizer(Phoneme, lex)
	for key, want := range doc.Coin {
		var got []string
		if strings.HasPrefix(key, "seed 5 syllables ") {
			n := int(key[len(key)-1] - '0')
			got = p.Build(NewMT(5), n, 1, 3, nil, 50)
		} else {
			seed := 0
			for _, ch := range key {
				seed = seed*10 + int(ch-'0')
			}
			got = p.Build(NewMT(uint32(seed)), 0, 1, 3, lex.Items(), 50)
		}
		if joined(got) != joined(want) {
			t.Errorf("coin %s: %v want %v", key, got, want)
		}
	}
	for pair, want := range doc.Blend {
		ab := strings.Fields(pair)
		phones, kept, dropped, err := p.Blend(tok.Pronounce(ab[0]), tok.Pronounce(ab[1]))
		if err != nil {
			t.Fatal(err)
		}
		if joined(phones) != joined(want.Phones) || kept != want.Kept || dropped != want.Dropped {
			t.Errorf("blend %s: %v %d %d want %v %d %d", pair, phones, kept, dropped, want.Phones, want.Kept, want.Dropped)
		}
		if got, _ := tok.Blend(ab[0], ab[1]); got != want.Spelling {
			t.Errorf("blend %s: %q want %q", pair, got, want.Spelling)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
