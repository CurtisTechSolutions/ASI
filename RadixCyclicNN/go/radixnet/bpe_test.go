package radixnet

// The cross-port contract of the tokenizer: ../../tests/tokens_fixture.json is
// what radixnet/bpe.py makes of a battery of hard texts, under the bundled
// merges and under a small vocabulary learned from the sample corpus, and this
// port must make the same - pre-token for pre-token, token for token.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type tokensCase struct {
	Text       string   `json:"text"`
	Pretokens  []string `json:"pretokens"`
	Tokens     []string `json:"tokens"`
	IDs        []int    `json:"ids"`
	SpecialIDs []int    `json:"special_ids"`
	Units      []string `json:"units"`
	Reread     []string `json:"reread"`
	Spelled    string   `json:"spelled"`
}

type tokensRead struct {
	Text    string   `json:"text"`
	Units   []string `json:"units"`
	Spelled string   `json:"spelled"`
}

type tokensBroken struct {
	IDs  []int  `json:"ids"`
	Text string `json:"text"`
}

type tokensSide struct {
	MergesText string         `json:"merges_text"`
	VocabSize  int            `json:"vocab_size"`
	Merges     int            `json:"merges"`
	Cases      []tokensCase   `json:"cases"`
	Reads      []tokensRead   `json:"reads"`
	Broken     []tokensBroken `json:"broken"`
}

type tokensFixture struct {
	Glue    string          `json:"glue"`
	Classes [][2]any        `json:"classes"`
	Bundled tokensSide      `json:"bundled"`
	Small   tokensSide      `json:"small"`
	Raw     json.RawMessage `json:"-"`
}

func loadTokensFixture(t *testing.T) tokensFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "tests", "tokens_fixture.json"))
	if err != nil {
		t.Fatalf("the fixture: %v", err)
	}
	var f tokensFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("the fixture: %v", err)
	}
	return f
}

func sameStrings(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func sameInts(a, b []int) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func checkTokensSide(t *testing.T, name string, tok *BPETokenizer, side tokensSide) {
	t.Helper()
	if tok.VocabSize() != side.VocabSize || len(tok.Merges) != side.Merges {
		t.Fatalf("%s: vocabulary %d / merges %d, want %d / %d", name, tok.VocabSize(), len(tok.Merges), side.VocabSize, side.Merges)
	}
	for _, c := range side.Cases {
		if got := Pretokenize(" " + c.Text); !sameStrings(got, c.Pretokens) {
			t.Errorf("%s %q: pretokens %q, want %q", name, c.Text, got, c.Pretokens)
		}
		if got := tok.Tokens(c.Text); !sameStrings(got, c.Tokens) {
			t.Errorf("%s %q: tokens %q, want %q", name, c.Text, got, c.Tokens)
		}
		if got := tok.Encode(c.Text, false); !sameInts(got, c.IDs) {
			t.Errorf("%s %q: ids %v, want %v", name, c.Text, got, c.IDs)
		}
		if got := tok.Encode(c.Text, true); !sameInts(got, c.SpecialIDs) {
			t.Errorf("%s %q: ids with the special token %v, want %v", name, c.Text, got, c.SpecialIDs)
		}
		if got := tok.Units(c.Text); !sameStrings(got, c.Units) {
			t.Errorf("%s %q: units %q, want %q", name, c.Text, got, c.Units)
		}
		form := strings.Join(c.Units, " ")
		if got := tok.Units(form); !sameStrings(got, c.Reread) {
			t.Errorf("%s %q: reread %q, want %q", name, form, got, c.Reread)
		}
		if got := tok.Spell(form); got != c.Spelled {
			t.Errorf("%s %q: spelled %q, want %q", name, form, got, c.Spelled)
		}
		if got, err := tok.Decode(c.IDs); err != nil || got != c.Text {
			t.Errorf("%s: decode %v = %q (%v), want %q", name, c.IDs, got, err, c.Text)
		}
	}
	for _, r := range side.Reads {
		if got := tok.Units(r.Text); !sameStrings(got, r.Units) {
			t.Errorf("%s read %q: units %q, want %q", name, r.Text, got, r.Units)
		}
		if got := tok.Spell(r.Text); got != r.Spelled {
			t.Errorf("%s read %q: spelled %q, want %q", name, r.Text, got, r.Spelled)
		}
	}
	for _, b := range side.Broken {
		if got, err := tok.Decode(b.IDs); err != nil || got != b.Text {
			t.Errorf("%s: decode %v = %q (%v), want %q", name, b.IDs, got, err, b.Text)
		}
	}
}

func TestTokenizerFixture(t *testing.T) {
	f := loadTokensFixture(t)
	if f.Glue != BPEGlue {
		t.Fatalf("the glue is %q, want %q", BPEGlue, f.Glue)
	}
	for _, pair := range f.Classes {
		cp := rune(pair[0].(float64))
		if got := string(charClass(cp)); got != pair[1].(string) {
			t.Errorf("class of U+%04X: %s, want %s", cp, got, pair[1])
		}
	}
	bundled, err := BundledTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	checkTokensSide(t, "bundled", bundled, f.Bundled)
	small, err := ParseMerges(f.Small.MergesText)
	if err != nil {
		t.Fatal(err)
	}
	checkTokensSide(t, "small", small, f.Small)
}

func TestTheEmbeddedMergesAreTheBundledOnes(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "radixnet", "data", "merges.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != bundledMerges {
		t.Fatal("go/radixnet/data/merges.txt differs from radixnet/data/merges.txt: make tokenizer-merges")
	}
}

func TestTheTokenUnit(t *testing.T) {
	enc, err := ParseEncoding("bpe:3:1")
	if err != nil {
		t.Fatal(err)
	}
	if enc.String() != "token:3:1" || enc.UnitsName() != "tokens" || enc.Describe() != "3-token grams, stride 1 (sliding)" {
		t.Fatalf("%s / %s / %s", enc, enc.UnitsName(), enc.Describe())
	}
	if !enc.Unit.Spells() || enc.Unit.Phonetic() || enc.Unit.Tokens() {
		t.Fatal("the token unit is read and spelled, and is neither sounds nor tokens taken as they come")
	}
	if got := enc.Encode("The walking cat."); !sameStrings(got, []string{"The walk ⁀ing", "walk ⁀ing cat", "⁀ing cat ⁀."}) {
		t.Fatalf("grams %q", got)
	}
	if got := enc.Len("The walking cat."); got != 5 {
		t.Fatalf("length %d", got)
	}
	if got := enc.Normalize("The walking cat."); got != "The walk ⁀ing cat ⁀." {
		t.Fatalf("normalized %q", got)
	}
	if got := enc.Spell("The walk ⁀ing cat ⁀."); got != "The walking cat." {
		t.Fatalf("spelled %q", got)
	}
	path := []string{"The walk ⁀ing", "walk ⁀ing cat", "⁀ing cat ⁀."}
	if got := enc.DecodePath(path, 0, false); got != "cat ⁀." {
		t.Fatalf("decoded %q", got)
	}
	if !enc.HasUnitPrefix("the cat sat ⁀.", "the cat sat.") || enc.HasUnitPrefix("the cat sat ⁀.", "the ca") {
		t.Fatal("a prefix is matched by its tokens")
	}
	if got := enc.Join("the cat", "sat."); got != "the cat sat ⁀." {
		t.Fatalf("joined %q", got)
	}
	if got := enc.Reverse("The walking cat."); got != "⁀. cat ⁀ing walk The" {
		t.Fatalf("reversed %q", got)
	}
	if got := Spelled(enc, "the cat sat ⁀."); got != "the cat sat." {
		t.Fatalf("Spelled %q", got)
	}
	if _, err := ParseEncoding("tokenz:3"); err == nil {
		t.Fatal("an unknown unit was accepted")
	}
}

func TestDecodeUTF8KeepsWhatIsUnfinished(t *testing.T) {
	text, rest := decodeUTF8([]byte("a\xe2\x82"), false)
	if text != "a" || string(rest) != "\xe2\x82" {
		t.Fatalf("%q %q", text, rest)
	}
	text, rest = decodeUTF8([]byte("\xe2\x82\xac"), false)
	if text != "€" || rest != nil {
		t.Fatalf("%q %q", text, rest)
	}
	if text, _ = decodeUTF8([]byte("a\xe2\x82"), true); text != "a�" {
		t.Fatalf("%q", text)
	}
}

func TestATokenModelsLabelsReadBackAsTheirTokens(t *testing.T) {
	enc := Encoding{BPETokens, 3, 1}
	texts := []string{"the cat sat on the mat.", "the cat sat on the floor!", "a quick brown fox jumps over the lazy dog"}
	m := modelWith(t, enc, texts, 2)
	m.G.Compress()
	for _, id := range m.G.AliveNodes() {
		if id < First {
			continue
		}
		label := m.G.Label(id)
		if got := enc.Units(label).Text(); got != label {
			t.Fatalf("label %q reads back as %q", label, got)
		}
	}
	out, err := m.Predict("the cat sat", PredictOptions{Length: 4, Mode: "beam", K: 3})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.FullText, "the cat sat on the ") || !strings.HasPrefix(enc.Spell(out.FullText), "the cat sat on the ") {
		t.Fatalf("predicted %q (%q)", out.FullText, enc.Spell(out.FullText))
	}
}
