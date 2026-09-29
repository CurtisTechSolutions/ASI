package tokenizer

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func tinyConfig() Config {
	return Config{Window: 6, Embed: 4, EncHidden: 12, DecHidden: 12, OutEmbed: 6, Levels: [][]int{{3, 3}, {4}, {2, 2}},
		Noise: 0.1, Recency: 0.7, Predict: 0.5, StartShare: 0.1}
}

var corpus = [][]byte{
	[]byte("the cat sat on the mat. the cat sat on the log. the dog ate the bone. a cat and a dog."),
	[]byte("the bird sang on the wire. a dog and a cat sat. the mat was on the floor. the log was on the fire."),
	{0, 1, 2, 3, 255, 254, 253, 128, 127, 10, 13, 9, 0, 0, 0, 200},
}

// The whole network's analytic gradient matches finite differences (without the rounding).
func TestGradCheck(t *testing.T) {
	tok, err := New(tinyConfig(), 5)
	if err != nil {
		t.Fatal(err)
	}
	worst, err := tok.GradCheck(corpus, 3, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if worst > 2e-2 {
		t.Fatalf("worst relative gradient error %.4f", worst)
	}
}

// Codes are in range, the same context always gets the same code, and every position of a text is coded,
// bytes of any value included.
func TestEncode(t *testing.T) {
	tok, _ := New(tinyConfig(), 1)
	for _, text := range corpus {
		codes := tok.EncodeAll(text)
		if codes.Len() != len(text)+1 || codes.D != 3 {
			t.Fatalf("codes: %d positions of %d symbols, want %d of 3", codes.Len(), codes.D, len(text)+1)
		}
		for i := 0; i < codes.Len(); i++ {
			c := codes.At(i)
			for s, v := range c {
				if v < 0 || int(v) >= tok.Radix(s) {
					t.Fatalf("position %d symbol %d = %d, radix %d", i, s, v, tok.Radix(s))
				}
			}
			single := tok.Encode(text[:i])
			if !bytes.Equal(int32sBytes(single), int32sBytes(c)) {
				t.Fatalf("position %d: EncodeAll %v, Encode %v", i, c, single)
			}
		}
	}
	if r := tok.Radices(); r[0] != 9 || r[1] != 4 || r[2] != 4 {
		t.Fatalf("radices %v", r)
	}
	if b := tok.Bits(); b < 7.16 || b > 7.18 {
		t.Fatalf("bits %v", b)
	}
}

func int32sBytes(v []int32) []byte {
	out := make([]byte, 0, 4*len(v))
	for _, x := range v {
		out = append(out, byte(x), byte(x>>8), byte(x>>16), byte(x>>24))
	}
	return out
}

// Training lowers the loss and raises exact reconstruction; the log is kept; decoding gives a window of
// symbols from any number of known symbols.
func TestTrainAndDecode(t *testing.T) {
	tok, _ := New(tinyConfig(), 2)
	o := DefaultOptions()
	o.Steps, o.Batch, o.EvalEvery, o.EvalWindows = 150, 32, 50, 64
	stats, err := tok.Train(corpus, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 3 || tok.Steps != 150 || len(tok.Stats) != 3 {
		t.Fatalf("stats %d, steps %d, kept %d", len(stats), tok.Steps, len(tok.Stats))
	}
	first, last := stats[0], stats[len(stats)-1]
	if last.Loss >= first.Loss || last.Loss > 7.5 {
		t.Errorf("loss did not fall: %.3f -> %.3f bits", first.Loss, last.Loss)
	}
	if len(last.Next) != 3 || last.Next[2] >= stats[0].Next[2] {
		t.Errorf("next-byte bits did not fall: %v -> %v", stats[0].Next, last.Next)
	}
	if len(last.Acc) != 3 || last.Acc[2] < last.Acc[0] || len(last.Tail) != 3 {
		t.Errorf("accuracy by depth %v (tail %v): more symbols should not reconstruct worse", last.Acc, last.Tail)
	}
	w := tok.Weights()
	if len(w) != 6 || w[5] <= w[0] || w[5] < 1 {
		t.Errorf("recency weights %v should rise toward the newest byte", w)
	}
	code := tok.Encode([]byte("the cat sat on the "))
	for known := 1; known <= 3; known++ {
		syms, err := tok.DecodeSymbols(code, known)
		if err != nil || len(syms) != 6 {
			t.Fatalf("decode with %d known: %v %v", known, syms, err)
		}
		text, _ := tok.Decode(code, known)
		if len([]rune(text)) != 6 {
			t.Fatalf("rendered %q", text)
		}
	}
	if _, err := tok.DecodeSymbols([]int32{99, 0, 0}, 0); err == nil {
		t.Error("expected an out-of-range symbol to be refused")
	}
	if !strings.Contains(Render([]int{PAD, 'a'}), "·a") {
		t.Error("PAD should render as a middle dot")
	}
}

// A saved tokenizer loads bit for bit.
func TestSaveLoad(t *testing.T) {
	tok, _ := New(tinyConfig(), 9)
	o := DefaultOptions()
	o.Steps, o.Batch, o.EvalEvery, o.EvalWindows = 20, 16, 10, 32
	if _, err := tok.Train(corpus, o); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tok.json.gz")
	if err := tok.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if back.Steps != 20 || back.Corpus != tok.Corpus || len(back.Stats) != 2 || back.Window != 6 {
		t.Fatalf("bookkeeping lost: %+v", back.Info())
	}
	for _, text := range corpus {
		a, b := tok.EncodeAll(text), back.EncodeAll(text)
		if !bytes.Equal(int32sBytes(a.Data), int32sBytes(b.Data)) {
			t.Fatalf("codes differ after reload")
		}
	}
	for i, p := range tok.Params() {
		q := back.Params()[i]
		for k := range p.W.Data {
			if p.W.Data[k] != q.W.Data[k] {
				t.Fatalf("weight %s[%d] changed", p.Name, k)
			}
		}
	}
}

// Levels parse both spellings and round-trip.
func TestParseLevels(t *testing.T) {
	for _, s := range []string{"4,4:4,4:4,4", "16:16:16", "8:5,5"} {
		l, err := ParseLevels(s)
		if err != nil || LevelsString(l) != s {
			t.Errorf("%q -> %v %v -> %q", s, l, err, LevelsString(l))
		}
	}
	if _, err := ParseLevels("4:1"); err == nil {
		t.Error("a level count of 1 should be refused")
	}
	if err := (Config{}).Validate(); err == nil {
		t.Error("the empty config should be refused")
	}
}
