package radixnet

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var bandTexts = []string{
	"the cat sat on the mat",
	"the cat ran to the door",
	"the dog sat on the log",
	"the dog ate the bone",
	"the bat sat on the mat",
}

// flatModel is a count model trained without compression: every node is one
// gram, so every step is one gram's charge.
func flatModel(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	if _, err := m.Train(bandTexts, TrainOptions{Epochs: 1, AutoCompress: false}); err != nil {
		t.Fatal(err)
	}
	return m
}

func blur(v float64) *float64 { return &v }
func yes(v bool) *bool        { return &v }

// childLabel is the label of the node an edge enters.
func childLabel(g *Graph, edge int) string {
	for p := range g.children {
		adj := &g.children[p]
		for i, c := range adj.order {
			if adj.edges[i] == edge {
				return g.Labels[c]
			}
		}
	}
	return ""
}

func TestBandPeaksAtTheCentre(t *testing.T) {
	if got := BandWeights(3, 0.5); !reflect.DeepEqual(got, []float64{0.5, 1, 0.5}) {
		t.Fatalf("n=3: %v", got)
	}
	if got := BandWeights(5, 0.5); !reflect.DeepEqual(got, []float64{0.5, 0.75, 1, 0.75, 0.5}) {
		t.Fatalf("n=5: %v", got)
	}
	if got := BandWeights(2, 0.5); !reflect.DeepEqual(got, []float64{0.5, 0.5}) {
		t.Fatalf("a gram of two is all ends: %v", got)
	}
	if got := BandWeights(1, 1); !reflect.DeepEqual(got, []float64{1}) {
		t.Fatalf("a gram of one is all centre: %v", got)
	}
	for _, bad := range []float64{-0.1, 1.5, math.NaN(), math.Inf(1)} {
		if CheckBlur(bad) == nil {
			t.Fatalf("blur %v must be refused", bad)
		}
	}
}

// Each changed unit hands out one charge, the gram with it at its centre taking the most.
func TestSpreadChargesTheCentreMost(t *testing.T) {
	enc := DefaultEncoding()
	wrong, right := "the cat sat", "the bat sat"
	spans, _ := enc.ChangedSpans(wrong, right)
	grams := enc.Encode(wrong)
	out := SpreadCharges(3, 1, len(grams), enc.Len(wrong), spans, BandWeights(3, 0.5))
	want := []float64{0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0}
	if !reflect.DeepEqual(out.Shares, want) {
		t.Fatalf("shares %v, want %v", out.Shares, want)
	}
	for g, f := range out.Focus {
		if f != (g == 3) {
			t.Fatalf("the focus is %q alone: %v", grams[3], out.Focus)
		}
	}
	writes, _ := WriterMarks(3, 1, len(grams), enc.Len(wrong), spans)
	for g, w := range writes {
		if w != (g == 2) {
			t.Fatalf("the writer is %q alone: %v", grams[2], writes)
		}
	}
}

// The three invariants the rule rests on, over every encoding and blur.
func TestSpreadInvariants(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog again and again"
	for _, spec := range []string{"char:3:1", "char:5:1", "char:4:4", "char:6:3", "char:2:1", "word:1:1", "word:2:1", "word:3:1"} {
		enc, err := ParseEncoding(spec)
		if err != nil {
			t.Fatal(err)
		}
		grams := enc.Encode(text)
		length := enc.Len(text)
		covered := 0
		if length >= enc.N {
			covered = (length-enc.N)/enc.Stride*enc.Stride + enc.N
		}
		for _, b := range []float64{0, 0.3, 0.5, 1} {
			weights := BandWeights(enc.N, b)
			for unit := 0; unit < covered; unit++ { // one marked unit, one charge
				out := SpreadCharges(enc.N, enc.Stride, len(grams), length, []Span{{unit, unit + 1}}, weights)
				total := 0.0
				for _, s := range out.Shares {
					total += s
				}
				if math.Abs(total-1) > 1e-12 {
					t.Fatalf("%s blur %g unit %d: the shares add up to %v", spec, b, unit, total)
				}
			}
			// a whole-text judgement leaves every gram at a full charge
			out := SpreadCharges(enc.N, enc.Stride, len(grams), length, []Span{{0, length}}, weights)
			for g, s := range out.Shares {
				if s < 1-1e-12 || !out.Focus[g] {
					t.Fatalf("%s blur %g: gram %d took %v of a whole-text judgement", spec, b, g, s)
				}
			}
		}
	}
	// a unit only one gram sees is charged to it in full
	out := SpreadCharges(4, 4, 3, 12, []Span{{4, 5}}, BandWeights(4, 1))
	if !reflect.DeepEqual(out.Shares, []float64{0, 1, 0}) {
		t.Fatalf("groups: %v", out.Shares)
	}
	if out := SpreadCharges(3, 1, 3, 5, []Span{{0, 1}}, BandWeights(3, 1)); out.Shares[0] != 1 {
		t.Fatalf("the first unit, seen only at an edge the band gives nothing, is split evenly: %v", out.Shares)
	}
	if out := SpreadCharges(3, 1, 9, 11, []Span{{11, 11}}, BandWeights(3, 0.5)); !out.End {
		t.Fatal("the position after the last unit belongs to the step into END")
	}
}

// With the band on the charge follows the centre; the verdict goes there too.
func TestCorrectWithTheBand(t *testing.T) {
	m := flatModel(t)
	if _, err := m.ConfigureAttention(nil, blur(0.5)); err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), m.G.EdgeReward...)
	out, err := m.Correct("the cat sat on the mat", "the bat sat on the mat", CorrectOptions{Strength: 1, Weight: 2, Reward: 1})
	if err != nil {
		t.Fatal(err)
	}
	penalties, rewards := map[string]float64{}, map[string]float64{}
	for e, value := range m.G.EdgeReward {
		was := 0.0
		if e < len(before) {
			was = before[e]
		}
		switch delta := value - was; {
		case delta < 0:
			penalties[childLabel(m.G, e)] = delta
		case delta > 0:
			rewards[childLabel(m.G, e)] = delta
		}
	}
	if want := map[string]float64{"e c": -0.5, " ca": -1, "cat": -0.5}; !reflect.DeepEqual(penalties, want) {
		t.Fatalf("penalties %v, want %v", penalties, want)
	}
	if want := map[string]float64{"e b": 0.25, " ba": 0.5, "bat": 0.25}; !reflect.DeepEqual(rewards, want) {
		t.Fatalf("rewards %v, want %v", rewards, want)
	}
	if out.Penalised != 3 || out.Penalty != 2 || out.MarkedIncorrect != 1 || out.MarkedCorrect != 1 {
		t.Fatalf("one changed letter, one charge, one verdict each way: %+v", out)
	}
}

// keep tops a partly charged step up: charge + (1 - charge) * keep.
func TestCorrectKeepWithTheBand(t *testing.T) {
	m := flatModel(t)
	if _, err := m.ConfigureAttention(nil, blur(0.5)); err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), m.G.EdgeReward...)
	if _, err := m.Correct("the cat sat on the mat", "the bat sat on the mat", CorrectOptions{Strength: 1, Weight: 1, Reward: 1, Keep: 0.5}); err != nil {
		t.Fatal(err)
	}
	rewards := map[string]float64{}
	for e, value := range m.G.EdgeReward {
		if e < len(before) && value-before[e] > 0 {
			rewards[childLabel(m.G, e)] = value - before[e]
		}
	}
	if rewards["e b"] != 0.625 || rewards[" ba"] != 0.75 || rewards["the"] != 0.5 {
		t.Fatalf("rewards %v", rewards)
	}
}

// Switched on and back off, a model corrects exactly as one that never had a band.
func TestCorrectOffTheBandIsUnchanged(t *testing.T) {
	docs := []string{}
	for _, toggle := range []bool{false, true} {
		m := trained(t, 2, 1)
		if toggle {
			if _, err := m.ConfigureAttention(nil, blur(0.7)); err != nil {
				t.Fatal(err)
			}
			if _, err := m.ConfigureAttention(yes(false), nil); err != nil {
				t.Fatal(err)
			}
		}
		for _, pair := range [][2]string{{"the cat sit on the mat", "the cat sits on the mat"}, {"a dog ate the bone", "the dog ate the bone"}} {
			if _, err := m.Correct(pair[0], pair[1], CorrectOptions{Strength: 1, Weight: 1, Reward: 1, Keep: 0.25}); err != nil {
				t.Fatal(err)
			}
		}
		raw, err := json.Marshal(m.G.ToDoc())
		if err != nil {
			t.Fatal(err)
		}
		docs = append(docs, string(raw))
	}
	if docs[0] != docs[1] {
		t.Fatal("a band switched off must leave corrections exactly as they were")
	}
}

// The negative network is blamed by share: one changed letter, one charge at the severity.
func TestBlameCorrectionWithTheBand(t *testing.T) {
	m, err := NewNegativeModel(1, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.ConfigureAttention(nil, blur(0.5)); err != nil {
		t.Fatal(err)
	}
	out, err := m.BlameCorrection("the cat sat on the mat", "the bat sat on the mat", BlameOptions{Reason: "spelling", Severity: 2})
	if err != nil {
		t.Fatal(err)
	}
	total, most := 0.0, 0.0
	blamed := 0
	for _, b := range m.G.Neg.Blame {
		if b > 0 {
			total += b
			most = math.Max(most, b)
			blamed++
		}
	}
	if out.Blamed != blamed || total != 2 || most != 1 || m.G.Neg.TotalBlame != 2 {
		t.Fatalf("blamed %d (%+v), total %v, most %v, graph total %v", blamed, out, total, most, m.G.Neg.TotalBlame)
	}
}

// The band travels with the file, beside the encoding, and only while it is on.
func TestAttentionTravelsWithTheFile(t *testing.T) {
	enc, _ := ParseEncoding("word:3:1")
	opts := DefaultGraphOptions()
	opts.Encoding = enc
	m, err := NewModel(1, opts)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "model.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if _, has := doc["graph"].(map[string]any)["attention"]; has {
		t.Fatal("a band that is off writes nothing")
	}
	cfg, err := m.ConfigureAttention(yes(true), nil)
	if err != nil || !cfg.On || *cfg.Blur != DefaultBlur || !reflect.DeepEqual(cfg.Weights, []float64{0.5, 1, 0.5}) {
		t.Fatalf("switched on at the default blur: %+v %v", cfg, err)
	}
	if _, err := m.ConfigureAttention(nil, blur(0.3)); err != nil {
		t.Fatal(err)
	}
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if back.G.Attention != (AttentionBand{On: true, Blur: 0.3}) || back.Stats()["attention_blur"] != 0.3 {
		t.Fatalf("read back as %+v", back.G.Attention)
	}
	if _, err := m.ConfigureAttention(nil, blur(1.2)); err == nil {
		t.Fatal("a blur past 1 must be refused")
	}
	if cfg, _ := m.ConfigureAttention(yes(false), blur(0.9)); cfg.On || cfg.Blur != nil || cfg.Weights != nil {
		t.Fatalf("off whatever the blur says: %+v", cfg)
	}
}

// A preview shows both rules and changes nothing.
func TestAttentionPreview(t *testing.T) {
	m := flatModel(t)
	before, _ := json.Marshal(m.G.ToDoc())
	p, err := m.AttentionPreview("the cat sat", "the bat sat", nil)
	if err != nil {
		t.Fatal(err)
	}
	if p.Blur != DefaultBlur || !reflect.DeepEqual(p.Wrong.Charges[2:5], []float64{0.25, 0.5, 0.25}) {
		t.Fatalf("preview %+v", p.Wrong)
	}
	if p, _ := m.AttentionPreview("the cat sat", "the bat sat", blur(1)); !reflect.DeepEqual(p.Wrong.Charges[2:5], []float64{0, 1, 0}) {
		t.Fatalf("blur 1: %v", p.Wrong.Charges)
	}
	after, _ := json.Marshal(m.G.ToDoc())
	if string(before) != string(after) {
		t.Fatal("a preview must not change the model")
	}
}
