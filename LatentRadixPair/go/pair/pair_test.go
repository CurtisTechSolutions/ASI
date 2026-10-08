package pair

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/tokenizer"
)

func tinyTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	cfg := tokenizer.Config{Window: 8, Embed: 8, EncHidden: 48, DecHidden: 48, OutEmbed: 8,
		Levels: [][]int{{4, 4}, {4, 4}, {4}}, Noise: 0.02, Recency: 0.6, Predict: 1, StartShare: 0.1}
	tok, err := tokenizer.New(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	o := tokenizer.DefaultOptions()
	o.Steps, o.Batch, o.EvalEvery, o.EvalWindows, o.LR = 400, 64, 200, 64, 5e-3
	var corpus [][]byte
	for _, x := range texts {
		corpus = append(corpus, []byte(x))
	}
	stats, err := tok.Train(corpus, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tiny tokenizer: %.3f bits, accuracy %v, tail %v", stats[len(stats)-1].Loss, stats[len(stats)-1].Acc, stats[len(stats)-1].Tail)
	return tok
}

var texts = []string{"the cat sat on the mat", "the cat sat on the log", "the dog ate the bone", "a cat and a dog",
	"the bird sang on the wire", "a dog and a cat sat", "the mat was on the floor", "the log was on the fire"}

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

// Every code prefix has one node, ids are dense, levels and parents agree, and the ceiling is enforced.
func TestAddressOracle(t *testing.T) {
	for _, radices := range [][]int{{2}, {3, 4}, {6, 4, 4}, {2, 3, 5, 2}} {
		a, err := NewAddress(radices, Out, 1<<30)
		if err != nil {
			t.Fatal(err)
		}
		seen := make([]bool, a.N)
		var walk func(code []int32)
		walk = func(code []int32) {
			l := len(code)
			node := a.Of(code, l)
			if node < 0 || node >= a.N || seen[node] {
				t.Fatalf("radices %v: code %v -> bad or duplicate node %d", radices, code, node)
			}
			seen[node] = true
			if a.Level(node) != l {
				t.Fatalf("radices %v: node %d level %d, want %d", radices, node, a.Level(node), l)
			}
			back := a.Code(node)
			for j := range code {
				if back[j] != code[j] {
					t.Fatalf("radices %v: node %d -> %v, want %v", radices, node, back, code)
				}
			}
			if l > 0 && a.Parent(node) != a.Of(code, l-1) {
				t.Fatalf("radices %v: parent of %v is %d, want %d", radices, code, a.Parent(node), a.Of(code, l-1))
			}
			if l < len(radices) {
				for x := 0; x < radices[l]; x++ {
					walk(append(append([]int32(nil), code...), int32(x)))
				}
			} else {
				chain := a.Chain(code, nil)
				if len(chain) != l+1 || chain[0] != 0 || chain[l] != node {
					t.Fatalf("chain of %v: %v", code, chain)
				}
			}
		}
		walk(nil)
		for i, s := range seen {
			if !s {
				t.Fatalf("radices %v: node %d never reached", radices, i)
			}
		}
		if a.Parent(0) != 0 {
			t.Fatal("the root's parent is the root")
		}
	}
	if _, err := NewAddress([]int{16, 16, 16}, Out, 1000); err == nil {
		t.Error("expected the ceiling to refuse 4369 x 257 cells")
	}
}

// Reading a text counts its outcomes under every prefix of each position's code; the fold is the hand
// formula; unread contexts fall back to their parents and finally to the uniform.
func TestObserveAndFold(t *testing.T) {
	tok := tinyTokenizer(t)
	m, err := Prime(tok, DefaultSettings(), 1)
	if err != nil {
		t.Fatal(err)
	}
	m.Train(texts)
	if m.Pair.Count.Texts != 8 || m.Pair.Count.Units != 8+len(strings.Join(texts, "")) {
		t.Fatalf("read %d texts, %d units", m.Pair.Count.Texts, m.Pair.Count.Units)
	}
	if m.Pair.Count.Ctx[0] != int64(m.Pair.Count.Units) {
		t.Fatalf("root ctx %d, want %d", m.Pair.Count.Ctx[0], m.Pair.Count.Units)
	}
	// hand fold of "the ca" under backoff all
	code := m.Code("the ca")
	chain := m.Pair.Addr.Chain(code, nil)
	s := m.Pair.Settings
	P := make([]float64, Out)
	for j := range P {
		P[j] = 1.0 / Out
	}
	for _, node := range chain {
		ctx := float64(m.Pair.Count.Ctx[node])
		if ctx == 0 {
			continue
		}
		own := ctx / (ctx + s.Alpha)
		q := make([]float64, Out)
		start := m.Pair.Addr.Cell(node, 0)
		sum := 0.0
		for x := 0; x < Out; x++ {
			q[x] = (float64(m.Pair.Count.Cnt[start+x]) + s.Smoothing) / (ctx + s.Smoothing*Out)
			sum += q[x]
		}
		for x := range P {
			P[x] = own*q[x]/sum + (1-own)*P[x]
		}
	}
	for x := range P {
		P[x] = (1-s.Floor)*P[x] + s.Floor/Out
	}
	got, err := m.Fold("the ca", "reward", "all")
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for x := range got {
		total += got[x]
		if math.Abs(got[x]-P[x]) > 1e-12 {
			t.Fatalf("fold[%d] = %v, hand %v", x, got[x], P[x])
		}
	}
	if math.Abs(total-1) > 1e-9 {
		t.Fatalf("fold sums to %v", total)
	}
	if got['t'] < 0.5 {
		t.Errorf("after 'the ca' the model should expect 't', got %.3f", got['t'])
	}
	// an unread text still gets a proper distribution under every backoff
	for _, backoff := range []string{"all", "deepest", "none"} {
		P, err := m.Fold("zq xk vj", "punishment", backoff)
		if err != nil {
			t.Fatal(err)
		}
		sum := 0.0
		for _, v := range P {
			sum += v
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("backoff %s: sum %v", backoff, sum)
		}
	}
	if _, err := m.Fold("x", "sideways", ""); err == nil {
		t.Error("expected an unknown traversal to be refused")
	}
	if c := m.Coverage("the cat sat on the mat"); c <= m.Coverage("zq xk vj qq") {
		t.Errorf("coverage of read text %v should beat unread", c)
	}
}

// Rewards lift a step and punishments sink it, the prefix is context not outcome, punishments are not bought
// back, and rungs "final" leaves the coarser levels alone.
func TestFeedback(t *testing.T) {
	tok := tinyTokenizer(t)
	m, _ := Prime(tok, DefaultSettings(), 1)
	m.Train(texts)
	before, _ := m.Fold("a cat sat on the ", "reward", "")
	if _, err := m.Reward([]string{"mat"}, 5, nil, true, "a cat sat on the ", 2); err != nil {
		t.Fatal(err)
	}
	after, _ := m.Fold("a cat sat on the ", "reward", "")
	if after['m'] <= before['m'] {
		t.Errorf("reward should lift 'm': %.4f -> %.4f", before['m'], after['m'])
	}
	// the prefix's own steps were not credited: 'c' and 's' occur only in the prefix
	for _, b := range []byte{'c', 's'} {
		if v := m.Pair.Reward.Plus[m.Pair.Addr.Cell(0, int(b))]; v != 0 {
			t.Errorf("prefix step %q credited: %v", b, v)
		}
	}
	// the outcome's end mark was credited at the root
	if m.Pair.Reward.Plus[m.Pair.Addr.Cell(0, End)] <= 0 {
		t.Error("the end of the outcome should be credited")
	}
	pen, _ := m.Fold("a cat sat on the ", "punishment", "")
	if _, err := m.Punish([]string{"mat"}, 20, nil, "a cat sat on the ", 2); err != nil {
		t.Fatal(err)
	}
	pen2, _ := m.Fold("a cat sat on the ", "punishment", "")
	if pen2['m'] >= pen['m'] {
		t.Errorf("punishment should sink 'm' in the punishment traversal: %.4f -> %.4f", pen['m'], pen2['m'])
	}
	if _, err := m.Reward([]string{"mat"}, 100, nil, false, "a cat sat on the ", 2); err != nil {
		t.Fatal(err)
	}
	pen3, _ := m.Fold("a cat sat on the ", "punishment", "")
	if pen3['m'] > pen2['m']+1e-12 {
		t.Errorf("a reward must not buy a punishment back: %.6f -> %.6f", pen2['m'], pen3['m'])
	}
	if m.Pair.Reward.Judged != 3 || m.Pair.Reward.PenaltiesTotal <= 0 {
		t.Errorf("judged %d, penalties %v", m.Pair.Reward.Judged, m.Pair.Reward.PenaltiesTotal)
	}
	// rungs final: only the full code's cells
	s := DefaultSettings()
	s.Rungs = "final"
	m2, _ := Prime(tok, s, 1)
	m2.Train(texts)
	if _, err := m2.Reward([]string{"log"}, 1, nil, false, "the cat sat on the ", 0); err != nil {
		t.Fatal(err)
	}
	for node := 0; node < m2.Pair.Addr.Bases[m2.Pair.Addr.D()]; node++ {
		start := m2.Pair.Addr.Cell(node, 0)
		for _, v := range m2.Pair.Reward.Plus[start : start+Out] {
			if v != 0 {
				t.Fatalf("rungs final credited node %d at level %d", node, m2.Pair.Addr.Level(node))
			}
		}
	}
	// two_nrl and feedback marks
	if _, err := m.TwoNRL([]string{"the cat sat on the fire"}, []string{"the cat sat on the mat"}, 1, "", 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Feedback([]string{"good one", "bad one", "ignored"}, []float64{0.5, -0.25, 0}, "prefix ", 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Feedback([]string{"x"}, []float64{1, 2}, "", 0); err == nil {
		t.Error("expected mismatched marks to be refused")
	}
	if len(m.History) != 6 {
		t.Errorf("history has %d records", len(m.History))
	}
}

// Greedy walks are deterministic, respect length, stop at the end mark, and price every step; sampling at a
// low temperature agrees with greedy; scoring prices read text below unread text.
func TestWalkAndScore(t *testing.T) {
	tok := tinyTokenizer(t)
	m, _ := Prime(tok, DefaultSettings(), 1)
	for i := 0; i < 5; i++ {
		m.Train(texts)
	}
	a, err := m.Predict("the cat sat on the ", 6, "greedy", "reward", 0, false, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := m.Predict("the cat sat on the ", 6, "greedy", "reward", 0, false, "", 0)
	if a.Text != b.Text || len(a.Units) > 6 || len(a.StepCosts) != len(a.Units) || a.Peak <= 0 {
		t.Fatalf("greedy: %v vs %v", a, b)
	}
	if !strings.HasPrefix(a.FullText, "the cat sat on the ") {
		t.Errorf("full text %q", a.FullText)
	}
	if a.Text != "mat" && a.Text != "log" && !strings.HasPrefix(a.Text, "mat") && !strings.HasPrefix(a.Text, "log") {
		t.Logf("greedy continuation %q (a tiny tokenizer; not asserted)", a.Text)
	}
	for i := 0; i < 10; i++ {
		m.Train([]string{"the cat sat on the mat"})
	}
	e, _ := m.Predict("the cat sat on the mat", 10, "greedy", "reward", 0, true, "", 50)
	if !e.ReachedEnd || e.Units[len(e.Units)-1] != End || e.Text != "" {
		t.Errorf("to_end walk after a text read 15 times should end at once: %v", e)
	}
	c, _ := m.Predict("the cat sat on the ma", 1, "sample", "reward", 0.01, false, "", 0)
	if c.Mode != "sample" || c.Text != "t" {
		t.Errorf("cold sample after 'the ma' should give 't', got %q", c.Text)
	}
	if _, err := m.Predict("x", 1, "dijkstra", "reward", 0, false, "", 0); err == nil {
		t.Error("expected an unknown mode to be refused")
	}
	read, _ := m.Score("the cat sat on the mat", "reward", "")
	unread, _ := m.Score("zq xk vj qq", "reward", "")
	if read.Bits >= unread.Bits || read.Units != 22 || len(read.PerUnit) != 23 || read.PerUnit[22].Unit != "</s>" {
		t.Errorf("score read %.3f bits (%d units) vs unread %.3f", read.Bits, read.Units, unread.Bits)
	}
	if Symbol('\n') != "\\n" || Symbol(0) != "\\x00" || Symbol('a') != "a" {
		t.Error("symbol rendering")
	}
}

// A verdict is worth strength / outcomes per rung: 1/2 in a two-outcome game, 1/69 for English by default,
// the raw amount at outcomes 1; every judging call takes the unit and records it.
func TestOutcomeUnits(t *testing.T) {
	tok := tinyTokenizer(t)
	m, _ := Prime(tok, DefaultSettings(), 1)
	if m.Pair.Settings.Outcomes != EnglishPhones || EnglishPhones != 69 {
		t.Fatalf("default outcomes %d", m.Pair.Settings.Outcomes)
	}
	cell := func(prefix string, x byte) int {
		code := m.Code(prefix)
		return m.Pair.Addr.Cell(m.Pair.Addr.Of(code, m.Pair.Addr.D()), int(x))
	}
	r, err := m.Reward([]string{"x"}, 1, nil, false, "game ", 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Pair.Reward.Plus[cell("game ", 'x')]; math.Abs(got-0.5) > 1e-12 {
		t.Errorf("two-outcome verdict credited %v, want 0.5", got)
	}
	if r["outcomes"] != 2 || math.Abs(r["amount"].(float64)-0.5) > 1e-12 {
		t.Errorf("record %v", r)
	}
	m.Reward([]string{"y"}, 1, nil, false, "english ", 0)
	if got := m.Pair.Reward.Plus[cell("english ", 'y')]; math.Abs(got-1.0/69) > 1e-12 {
		t.Errorf("English verdict credited %v, want 1/69", got)
	}
	m.Punish([]string{"z"}, 3, nil, "raw ", 1)
	if got := m.Pair.Reward.Minus[cell("raw ", 'z')]; math.Abs(got-3) > 1e-12 {
		t.Errorf("raw punishment charged %v, want 3", got)
	}
	r, _ = m.TwoNRL([]string{"b"}, []string{"g"}, 1, "p ", 4)
	if r["outcomes"] != 4 || math.Abs(r["amount"].(float64)-0.25) > 1e-12 {
		t.Errorf("two_nrl record %v", r)
	}
	if got := m.Pair.Reward.Minus[cell("p ", 'b')]; math.Abs(got-0.25) > 1e-12 {
		t.Errorf("two_nrl charged %v, want 0.25", got)
	}
	r, _ = m.Feedback([]string{"f"}, []float64{0.5}, "q ", 10)
	if r["outcomes"] != 10 {
		t.Errorf("feedback record %v", r)
	}
	if got := m.Pair.Reward.Plus[cell("q ", 'f')]; math.Abs(got-0.05) > 1e-12 {
		t.Errorf("feedback credited %v, want 0.05", got)
	}
	s := DefaultSettings()
	s.Outcomes = 0
	if _, err := Prime(tok, s, 1); err == nil {
		t.Error("outcomes 0 should be refused")
	}
}

// A saved model reloads with the same counts, rewards, history and random state.
func TestSaveLoad(t *testing.T) {
	tok := tinyTokenizer(t)
	m, _ := Prime(tok, DefaultSettings(), 3)
	m.Train(texts)
	m.Reward([]string{"mat"}, 2, nil, true, "a cat sat on the ", 0)
	m.Punish([]string{"fire"}, 1, nil, "a cat sat on the ", 0)
	s1, _ := m.Predict("the ", 4, "sample", "reward", 1, false, "", 0)
	path := filepath.Join(t.TempDir(), "model.json.gz")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if back.Pair.Count.Texts != 9 || back.Pair.Reward.Judged != 2 || len(back.History) != 3 || back.Created != m.Created {
		t.Fatalf("bookkeeping lost: %+v", back.Info())
	}
	for i := range m.Pair.Count.Cnt {
		if m.Pair.Count.Cnt[i] != back.Pair.Count.Cnt[i] {
			t.Fatalf("count cell %d differs", i)
		}
	}
	for i := range m.Pair.Count.Ctx {
		if m.Pair.Count.Ctx[i] != back.Pair.Count.Ctx[i] {
			t.Fatalf("ctx %d differs", i)
		}
	}
	for i := range m.Pair.Reward.Plus {
		if m.Pair.Reward.Plus[i] != back.Pair.Reward.Plus[i] || m.Pair.Reward.Minus[i] != back.Pair.Reward.Minus[i] {
			t.Fatalf("reward cell %d differs", i)
		}
	}
	s2, _ := m.Predict("the ", 4, "sample", "reward", 1, false, "", 0)
	s3, _ := back.Predict("the ", 4, "sample", "reward", 1, false, "", 0)
	if s2.Text != s3.Text {
		t.Errorf("random state not restored: %q vs %q (first draw %q)", s2.Text, s3.Text, s1.Text)
	}
	a, _ := m.Fold("the cat sat on the ", "reward", "")
	b, _ := back.Fold("the cat sat on the ", "reward", "")
	for x := range a {
		if a[x] != b[x] {
			t.Fatalf("fold differs at %d", x)
		}
	}
	if !equalInts(m.Info()["radices"].([]int), back.Info()["radices"].([]int)) {
		t.Error("radices differ")
	}
}
