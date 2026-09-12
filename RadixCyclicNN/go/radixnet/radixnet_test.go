package radixnet

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
)

func corpus(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "data", "sample_corpus.txt"))
	if err != nil {
		t.Skipf("sample corpus not found: %v", err)
	}
	return SplitTexts(string(raw), "lines", 0)
}

// trained builds a deterministic model: Exact counting (atomics) so the tests can compare numbers;
// workers 0 means one goroutine per text.
func trained(t *testing.T, epochs int, workers int) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Workers = workers
	m.G.Workers = workers
	m.Exact = true
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	if _, err := m.Train(corpus(t), opts); err != nil {
		t.Fatal(err)
	}
	return m
}

// -- Mersenne Twister: bit-for-bit Python's random.Random ------------------------------

func TestMT19937MatchesPython(t *testing.T) {
	cases := []struct {
		seed  int64
		mt0_3 [3]uint32
		mt623 uint32
		vals  [3]float64
		unif  float64
	}{
		{0, [3]uint32{2147483648, 766982754, 497961170}, 2902720905, [3]float64{0.8444218515250481, 0.7579544029403025, 0.420571580830845}, 1.3444218515250481},
		{1, [3]uint32{2147483648, 163610392, 3414454684}, 1781494222, [3]float64{0.13436424411240122, 0.8474337369372327, 0.763774618976614}, 0.6343642441124012},
		{12345, [3]uint32{2147483648, 2105189241, 1699489545}, 238504783, [3]float64{0.41661987254534116, 0.010169169457068361, 0.8252065092537432}, 0.9166198725453412},
		{1099511627781, [3]uint32{2147483648, 550267540, 1323789645}, 2187275131, [3]float64{0.5043802970418443, 0.2686044399723282, 0.9257865475671585}, 1.0043802970418443},
		{-7, [3]uint32{2147483648, 2261207994, 869397079}, 2448866966, [3]float64{0.32383276483316237, 0.15084917392450192, 0.6509344730398537}, 0.8238327648331624},
	}
	for _, c := range cases {
		r := NewMT19937(c.seed)
		if r.mt[0] != c.mt0_3[0] || r.mt[1] != c.mt0_3[1] || r.mt[2] != c.mt0_3[2] || r.mt[623] != c.mt623 || r.index != mtN {
			t.Fatalf("seed %d: state differs from Python", c.seed)
		}
		for i, want := range c.vals {
			if got := r.Float64(); got != want {
				t.Fatalf("seed %d draw %d: %v != %v", c.seed, i, got, want)
			}
		}
		if r.index != 6 {
			t.Fatalf("seed %d: index %d after 3 doubles", c.seed, r.index)
		}
		if got := NewMT19937(c.seed).Uniform(0.5, 1.5); got != c.unif {
			t.Fatalf("seed %d uniform: %v != %v", c.seed, got, c.unif)
		}
		state := r.State()
		r2 := NewMT19937(0)
		if err := r2.SetState(state); err != nil {
			t.Fatal(err)
		}
		if r2.Float64() != r.Float64() {
			t.Fatalf("seed %d: restored state diverges", c.seed)
		}
	}
}

func TestFsumIsExact(t *testing.T) {
	xs := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1e16, 1.0, -1e16}
	if got := fsum(xs); got != 2.0 {
		t.Fatalf("fsum = %v, want 2.0", got)
	}
	if fsum(nil) != 0 || fsum([]float64{1.5}) != 1.5 {
		t.Fatal("trivial sums")
	}
}

// -- encoding and text splitting ----------------------------------------------------------

func TestEncodeAndDecode(t *testing.T) {
	if got := Encode("hello"); !reflect.DeepEqual(got, []string{"hel", "ell", "llo"}) {
		t.Fatalf("Encode = %v", got)
	}
	if Encode("hi") != nil {
		t.Fatal("short text must give nil")
	}
	if got := Encode("héllo"); got[1] != "éll" {
		t.Fatalf("code points, not bytes: %v", got)
	}
	if got := DecodePath([]string{"hello", "lo wo", "world"}, 0, true); got != "hello world" {
		t.Fatalf("DecodePath = %q", got)
	}
	if got := DecodePath([]string{"hello"}, 0, false); got != "lo" {
		t.Fatalf("continuation only: %q", got)
	}
	if got := runeSlice("héllo", 1, 3); got != "él" {
		t.Fatalf("runeSlice = %q", got)
	}
	if got := truncateRunes("héllo", 2); got != "hé" {
		t.Fatalf("truncateRunes = %q", got)
	}
}

func TestSplitTexts(t *testing.T) {
	content := "first line\nsecond line\n\nthird para line one\nline two\n\n\nlast\n"
	if got := SplitTexts(content, "lines", 0); len(got) != 5 {
		t.Fatalf("lines: %v", got)
	}
	paras := SplitTexts(content, "paragraphs", 0)
	if !reflect.DeepEqual(paras, []string{"first line second line", "third para line one line two", "last"}) {
		t.Fatalf("paragraphs: %v", paras)
	}
	pages := SplitTexts("a\nb\nc\nd\ne", "pages", 2)
	if !reflect.DeepEqual(pages, []string{"a b", "c d", "e"}) {
		t.Fatalf("pages by line count: %v", pages)
	}
	ff := SplitTexts("page one\nstill one\fpage two", "pages", 0)
	if !reflect.DeepEqual(ff, []string{"page one still one", "page two"}) {
		t.Fatalf("pages by form feed: %v", ff)
	}
	if got := SplitTexts("x\ny\n", "file", 0); !reflect.DeepEqual(got, []string{"x\ny"}) {
		t.Fatalf("file: %v", got)
	}
}

// -- structure --------------------------------------------------------------------------------

func TestGraphInvariantsAndRoundTrip(t *testing.T) {
	texts := corpus(t)
	m := trained(t, 3, 4)
	if err := m.G.CheckInvariants(texts, true); err != nil {
		t.Fatal(err)
	}
	if m.G.CompressionRatio() <= 1 {
		t.Fatalf("compression ratio %v", m.G.CompressionRatio())
	}
	g := m.G
	if len(g.EdgeReward) != len(g.EdgeW) || len(g.WindowEdgeCount) != len(g.EdgeW) || len(g.EdgeParent) != len(g.EdgeW) {
		t.Fatal("edge arrays out of step")
	}
	// every stored weight is the dual frequency function of the tracked numbers
	g.Prepare()
	for _, p := range g.AliveNodes() {
		ch := g.Children(p)
		var total, recent int64
		for _, c := range ch {
			total += g.EdgeCount[c.E]
			recent += g.WindowEdgeCount[c.E]
		}
		for _, c := range ch {
			want := g.EdgeWeight(float64(g.EdgeCount[c.E]), g.EdgeReward[c.E], float64(total), len(ch), float64(g.WindowEdgeCount[c.E]), float64(recent))
			if math.Abs(want-g.EdgeW[c.E]) > 1e-12 {
				t.Fatalf("edge %d weight %v, want %v", c.E, g.EdgeW[c.E], want)
			}
		}
	}
}

func TestSplitAndMerge(t *testing.T) {
	g, _ := NewGraph(0, DefaultGraphOptions())
	if _, err := g.ObserveSequence(Encode("hello world"), true); err != nil {
		t.Fatal(err)
	}
	g.Compress()
	if g.NumNodes() != 3 { // START, END and one compressed node
		t.Fatalf("expected one compressed node, got %d nodes", g.NumNodes())
	}
	if _, err := g.ObserveSequence(Encode("hello there"), true); err != nil {
		t.Fatal(err)
	}
	if err := g.CheckInvariants([]string{"hello world", "hello there"}, false); err != nil {
		t.Fatal(err)
	}
	node, off, ok := g.Lookup("lo ") // the branch point: "hello " is followed by "o w" and "o t"
	if !ok {
		t.Fatal("lo  missing")
	}
	if off+Window != g.LabelLen(node) || g.Degree(node) != 2 {
		t.Fatalf("the branch point must end a node with two children: node %q degree %d", g.Labels[node], g.Degree(node))
	}
	g.Compress()
	if err := g.CheckInvariants([]string{"hello world", "hello there"}, true); err != nil {
		t.Fatal(err)
	}
	if _, _, err := g.Split(Start, 1); err == nil {
		t.Fatal("splitting a sentinel must fail")
	}
}

// -- weights: lazy rows agree with a full recompute; counting is exact under goroutines ----

func TestLazyWeightsMatchFullRecompute(t *testing.T) {
	m := trained(t, 2, 4)
	g := m.G
	// feedback touches a few rows only
	if _, err := m.Punish([]string{"the cat sat on the mat"}, 1, 1.0); err != nil {
		t.Fatal(err)
	}
	g.Prepare()
	lazy := append([]float64(nil), g.EdgeW...)
	g.RecomputeWeights()
	for e := range lazy {
		if g.EdgeAlive[e] && lazy[e] != g.EdgeW[e] {
			t.Fatalf("edge %d: lazy %v != full %v", e, lazy[e], g.EdgeW[e])
		}
	}
	// rewards land on the path's edges and lower its probability
	before := m.Score("the cat sat on the mat")
	m.Punish([]string{"the cat sat on the mat"}, 3, 1.0)
	after := m.Score("the cat sat on the mat")
	if !(after.LogProb < before.LogProb) {
		t.Fatalf("penalty must lower the log-prob: %v -> %v", before.LogProb, after.LogProb)
	}
	pos, neg := g.TotalReward()
	if pos != 0 || neg >= 0 {
		t.Fatalf("rewards %v / %v", pos, neg)
	}
	m.Invert()
	pos, neg = g.TotalReward()
	if pos <= 0 || neg != 0 || !g.Inverted {
		t.Fatalf("invert must flip rewards: %v / %v", pos, neg)
	}
}

func TestWorkersDoNotChangeTheModel(t *testing.T) {
	one := trained(t, 3, 1)
	many := trained(t, 3, 0) // no cap: one goroutine per text
	a, b := one.ToDoc().Graph, many.ToDoc().Graph
	if !reflect.DeepEqual(a.Nodes.Labels, b.Nodes.Labels) || !reflect.DeepEqual(a.Nodes.Count, b.Nodes.Count) {
		t.Fatal("node structure or counts differ between 1 and 8 workers")
	}
	if !reflect.DeepEqual(a.Edges.Src, b.Edges.Src) || !reflect.DeepEqual(a.Edges.Count, b.Edges.Count) || !reflect.DeepEqual(a.Edges.W, b.Edges.W) {
		t.Fatal("edges differ between 1 and 8 workers")
	}
	if !reflect.DeepEqual(a.Weights.WindowEvents, b.Weights.WindowEvents) || !reflect.DeepEqual(a.RngState, b.RngState) {
		t.Fatal("window or RNG state differ between 1 and 8 workers")
	}
	if one.History[2]["loss"] != many.History[2]["loss"] {
		lossA, lossB := one.History[2]["loss"].(float64), many.History[2]["loss"].(float64)
		if math.Abs(lossA-lossB) > 1e-12 {
			t.Fatalf("loss differs: %v vs %v", lossA, lossB)
		}
	}
}

// -- prediction, generation, scoring ----------------------------------------------------------

func TestPredictBeamTopAndBottom(t *testing.T) {
	m := trained(t, 2, 4)
	opts := DefaultPredictOptions()
	opts.Length, opts.K = 8, 3
	p, err := m.Predict("the cat", opts)
	if err != nil {
		t.Fatal(err)
	}
	if p.Mode != "beam" || p.K != 3 || p.Beam != 16 {
		t.Fatalf("prediction header %+v", p)
	}
	if len(p.Top) < 1 || len(p.Top) > 3 || len(p.Bottom) > 3 {
		t.Fatalf("top %d bottom %d", len(p.Top), len(p.Bottom))
	}
	if p.Text != p.Top[0].Text || p.FullText != "the cat"+p.Text {
		t.Fatalf("best path %q / %q", p.Text, p.FullText)
	}
	for i := 1; i < len(p.Top); i++ {
		if p.Top[i].Cost < p.Top[i-1].Cost {
			t.Fatal("top must rise in cost")
		}
	}
	for i := 1; i < len(p.Bottom); i++ {
		if p.Bottom[i].Cost > p.Bottom[i-1].Cost {
			t.Fatal("bottom must fall in cost")
		}
	}
	seen := map[string]bool{}
	for _, r := range p.Top {
		seen[r.FullText] = true
	}
	for _, r := range p.Bottom {
		if seen[r.FullText] {
			t.Fatal("a path appears on both sides")
		}
		if runeLen(r.Text) < 8 && !r.ReachedEnd {
			t.Fatalf("bottom path too short: %q", r.Text)
		}
	}
	// costs are -log softmax: probabilities of a node's children sum to one
	g := m.G
	node, _, _ := g.Lookup("the")
	sum := 0.0
	for _, c := range g.ChildCosts(node) {
		sum += math.Exp(-c.Cost)
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("child probabilities sum to %v", sum)
	}
	// a sample walk and greedy sampling stay inside the structure
	opts.Mode, opts.Temperature = "sample", 0
	s, err := m.Predict("the cat", opts)
	if err != nil || s.Mode != "sample" || len(s.Top) != 1 {
		t.Fatalf("sample: %v %+v", err, s)
	}
	if _, err := m.Predict("the", PredictOptions{Mode: "nope", Temperature: 1}); err == nil {
		t.Fatal("unknown mode must fail")
	}
	if _, err := m.Predict("the", PredictOptions{Length: -1, Temperature: 1}); err == nil {
		t.Fatal("negative length must fail")
	}
}

func TestGenerateAndScore(t *testing.T) {
	m := trained(t, 2, 4)
	opts := DefaultGenerateOptions()
	opts.Mode, opts.Count, opts.MaxLength = "beam", 4, 40
	texts, err := m.Generate(opts)
	if err != nil || len(texts) != 4 {
		t.Fatalf("generate: %v %d", err, len(texts))
	}
	distinct := map[string]bool{}
	for i, r := range texts {
		if r.Text != r.FullText || r.NodeIDs[0] != Start || (!r.ReachedEnd && runeLen(r.Text) != 40) {
			t.Fatalf("sample %d: %+v", i, r)
		}
		if i > 0 && r.Cost < texts[i-1].Cost {
			t.Fatal("beam texts must rise in cost")
		}
		distinct[r.Text] = true
		if sc := m.Score(r.Text); sc.UnknownTransitions != 0 {
			t.Fatalf("generated text %q has unknown transitions", r.Text)
		}
	}
	if len(distinct) != 4 {
		t.Fatal("beam texts must be distinct")
	}
	opts.Mode, opts.Count = "dijkstra", 3
	one, _ := m.Generate(opts)
	if len(one) != 1 || one[0].Text != texts[0].Text {
		t.Fatalf("dijkstra = the most likely beam text: %v", one)
	}
	seed := int64(7)
	opts.Mode, opts.Count, opts.Seed, opts.Prefix = "sample", 3, &seed, "the "
	a, _ := m.Generate(opts)
	b, _ := m.Generate(opts)
	for i := range a {
		if a[i].Text != b[i].Text || !strings.HasPrefix(a[i].Text, "the ") {
			t.Fatalf("seeded sampling with a prefix: %q vs %q", a[i].Text, b[i].Text)
		}
	}
	good := m.Score("the cat sat on the mat")
	bad := m.Score("zqxj vwk plmn qzx")
	if good.UnknownTransitions != 0 || bad.UnknownTransitions == 0 || good.PerChar <= bad.PerChar {
		t.Fatalf("scores: %+v / %+v", good, bad)
	}
	if got := m.Score("hi"); got.Chars != 2 || got.Transitions != 0 {
		t.Fatalf("short text: %+v", got)
	}
	all := m.ScoreAll([]string{"the cat sat on the mat", "zqxj vwk plmn qzx"})
	if all[0] != good || all[1] != bad {
		t.Fatal("ScoreAll must equal Score")
	}
}

func TestConverse(t *testing.T) {
	m := trained(t, 2, 4)
	opts := DefaultConverseOptions()
	opts.Turns = 6
	turns, err := m.Converse("the cat sat on the mat", opts)
	if err != nil || len(turns) != 7 {
		t.Fatalf("converse: %v %d", err, len(turns))
	}
	if !turns[0].Given || turns[0].Speaker != "A" || turns[0].Text != "the cat sat on the mat" {
		t.Fatalf("opening: %+v", turns[0])
	}
	seen := map[string]bool{}
	for i, tr := range turns {
		if tr.Index != i || tr.Speaker != DefaultSpeakers[i%2] {
			t.Fatalf("turn %d: %+v", i, tr)
		}
		key := Normalize(tr.Text)
		if seen[key] {
			t.Fatalf("repeated utterance %q", tr.Text)
		}
		seen[key] = true
		if i > 0 && !tr.Fresh {
			words := strings.Fields(turns[i-1].Text)
			wanted := strings.Fields(tr.Context)
			found := false
			for j := 0; j+len(wanted) <= len(words); j++ {
				if reflect.DeepEqual(words[j:j+len(wanted)], wanted) {
					found = true
				}
			}
			if !found || tr.Text != tr.Context+tr.Reply {
				t.Fatalf("turn %d did not pick up the previous line: %+v", i, tr)
			}
		}
	}
	again, _ := m.Converse("the cat sat on the mat", opts)
	if Transcript(again) != Transcript(turns) {
		t.Fatal("beam conversations must be deterministic")
	}
	if got := TailContext("the cat sat on the mat", 12); got != "on the mat" {
		t.Fatalf("TailContext = %q", got)
	}
	empty, _ := NewModel(0, DefaultGraphOptions())
	if turns, _ := empty.Converse("", opts); len(turns) != 0 {
		t.Fatal("an untrained model has nothing to say")
	}
}

// The default mode: one goroutine per text bumping shared counters with plain increments.  A collision
// may lose an update, so the counts are compared with a tolerance; the structure, the window and the
// weights' consistency with the counts are exact.  The race detector would (rightly) flag the plain
// increments, so this test only runs without it.
func TestRacyCountingStaysUsable(t *testing.T) {
	if raceEnabled {
		t.Skip("racy counting is a data race by design; skipped under the race detector")
	}
	exact := trained(t, 3, 0)
	racy, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	racy.Workers, racy.G.Workers, racy.Exact = 0, 0, false
	if racy.Counting() != "racy" || exact.Counting() != "exact" {
		t.Fatal("counting labels")
	}
	texts := corpus(t)
	opts := DefaultTrainOptions()
	opts.Epochs = 3
	if _, err := racy.Train(texts, opts); err != nil {
		t.Fatal(err)
	}
	if err := racy.G.CheckInvariants(texts, true); err != nil {
		t.Fatal(err)
	}
	a, b := exact.ToDoc().Graph, racy.ToDoc().Graph
	if !reflect.DeepEqual(a.Nodes.Labels, b.Nodes.Labels) || !reflect.DeepEqual(a.Edges.Src, b.Edges.Src) {
		t.Fatal("the structure must not depend on the counting mode")
	}
	if !reflect.DeepEqual(a.Weights.WindowEvents, b.Weights.WindowEvents) || a.Weights.TotalTraversals != b.Weights.TotalTraversals {
		t.Fatal("the sliding window is applied in order and stays exact")
	}
	var exactSum, racySum int64
	for i := range a.Edges.Count {
		exactSum += a.Edges.Count[i]
		racySum += b.Edges.Count[i]
		if b.Edges.Count[i] > a.Edges.Count[i] {
			t.Fatalf("a racy count can only lose updates, never gain them (edge %d: %d > %d)", i, b.Edges.Count[i], a.Edges.Count[i])
		}
	}
	if racySum < exactSum*9/10 {
		t.Fatalf("too many lost updates: %d of %d traversals counted", racySum, exactSum)
	}
	// predictions still work and the weights are the function of whatever was counted
	racy.G.Prepare()
	p, err := racy.Predict("the cat", DefaultPredictOptions())
	if err != nil || p.Text == "" {
		t.Fatalf("predict on the racy model: %v %+v", err, p)
	}
}

// -- persistence ----------------------------------------------------------------------------------

func TestSaveLoadRoundTrip(t *testing.T) {
	m := trained(t, 2, 4)
	m.Punish([]string{"the dog ate the bone"}, 1, 0.5)
	dir := t.TempDir()
	for _, name := range []string{"model.count.json", "model.count.json.gz"} {
		path := filepath.Join(dir, name)
		if err := m.Save(path); err != nil {
			t.Fatal(err)
		}
		loaded, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		a, b := m.ToDoc().Graph, loaded.ToDoc().Graph
		a.Version, b.Version = 0, 0 // the reload recomputes the weights once, which bumps the version
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("%s: graph differs after reload", name)
		}
		if loaded.Stats()["epochs_total"] != m.Stats()["epochs_total"] || len(loaded.History) != len(m.History) {
			t.Fatalf("%s: meta / history differ", name)
		}
		p1, _ := m.Predict("the cat", DefaultPredictOptions())
		p2, _ := loaded.Predict("the cat", DefaultPredictOptions())
		if p1.FullText != p2.FullText || p1.Cost != p2.Cost {
			t.Fatalf("%s: predictions differ after reload", name)
		}
		// the RNG continues identically
		w1, _ := m.G.SampleWalk(Start, 0, 30, 1.0, nil, nil)
		w2, _ := loaded.G.SampleWalk(Start, 0, 30, 1.0, nil, nil)
		if w1.Text != w2.Text {
			t.Fatalf("%s: RNG state not restored", name)
		}
		if err := loaded.G.CheckInvariants(corpus(t), true); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := Load(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("missing file must fail")
	}
	bad := filepath.Join(dir, "bad.json")
	os.WriteFile(bad, []byte(`{"format":"radixnet","version":1}`), 0o644)
	if _, err := Load(bad); err == nil || !strings.Contains(err.Error(), "radixnet-count") {
		t.Fatalf("wrong format must be reported: %v", err)
	}
}

func TestLegacyWeightsBlockDefaults(t *testing.T) {
	m := trained(t, 1, 2)
	doc := m.ToDoc()
	// a file written before the dual frequency function: no global_scale key at all
	doc.Graph.Weights = nil
	raw, err := marshalCompact(doc)
	if err != nil {
		t.Fatal(err)
	}
	var back ModelDoc
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	legacy, err := FromDoc(&back)
	if err != nil {
		t.Fatal(err)
	}
	g := legacy.G
	if g.CountScale != 1 || g.GlobalScale != 0 || g.WindowScale != 0 || g.RewardScale != 1 || g.WindowSize != 10_000 {
		t.Fatalf("legacy defaults: %+v", g.WeightConfig())
	}
	if g.WindowTraversals() != 0 {
		t.Fatal("a legacy file has no window events")
	}
	// present keys win over the defaults
	var partial weightsDoc
	if err := json.Unmarshal([]byte(`{"count_scale":0.25,"window":7}`), &partial); err != nil {
		t.Fatal(err)
	}
	back.Graph.Weights = &partial
	legacy, err = FromDoc(&back)
	if err != nil {
		t.Fatal(err)
	}
	if legacy.G.CountScale != 0.25 || legacy.G.WindowSize != 7 || legacy.G.GlobalScale != 0 {
		t.Fatalf("partial legacy block: %+v", legacy.G.WeightConfig())
	}
}

// A soft memory limit is the difference between a collection and an OOM kill.
func TestParseSizeAndMemoryLimit(t *testing.T) {
	cases := map[string]int64{"": 0, "1024": 1024, "2k": 2 << 10, "3MiB": 3 << 20, "1.5g": 1536 << 20, "off": -1, "none": -1}
	for text, want := range cases {
		got, err := ParseSize(text)
		if err != nil || got != want {
			t.Fatalf("ParseSize(%q) = %d, %v; want %d", text, got, err, want)
		}
	}
	if _, err := ParseSize("later"); err == nil {
		t.Fatal("ParseSize must reject a non-size")
	}
	if n, source := AvailableMemory(); n <= 0 || source == "" {
		t.Fatalf("AvailableMemory() = %d, %q", n, source)
	}
	before := debug.SetMemoryLimit(-1)
	defer debug.SetMemoryLimit(before)
	if got := ApplyMemoryLimit(512<<20, DefaultMemoryFraction); got.Bytes != 512<<20 || got.Source != "flag" {
		t.Fatalf("explicit limit: %+v", got)
	}
	if got := debug.SetMemoryLimit(-1); got != 512<<20 {
		t.Fatalf("limit not applied: %d", got)
	}
	if got := ApplyMemoryLimit(-1, DefaultMemoryFraction); got.Bytes != 0 || got.String() != "no memory limit" {
		t.Fatalf("off: %+v", got)
	}
	auto := ApplyMemoryLimit(0, DefaultMemoryFraction)
	if auto.Bytes <= 0 || auto.Source == "" || !strings.Contains(auto.String(), "soft memory limit") {
		t.Fatalf("auto limit: %+v", auto)
	}
	if total, _ := AvailableMemory(); auto.Bytes > total {
		t.Fatalf("auto limit %d above the available %d", auto.Bytes, total)
	}
}

// Saving streams: a model file is never held in memory, and what it holds is
// byte for byte what the buffered encoder wrote.
func TestSaveStreamsTheSameBytes(t *testing.T) {
	m := trained(t, 2, 0)
	dir := t.TempDir()
	plain := filepath.Join(dir, "m.json")
	if err := m.Save(plain); err != nil {
		t.Fatal(err)
	}
	want, err := marshalCompact(m.ToDoc())
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(plain)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("streamed file differs from the buffered encoding (%d vs %d bytes)", len(got), len(want))
	}
	if n := len(got); n == 0 || got[n-1] == '\n' {
		t.Fatal("the file must not end with the encoder's newline")
	}
	zipped := filepath.Join(dir, "m.json.gz")
	if err := m.Save(zipped); err != nil {
		t.Fatal(err)
	}
	back, err := Load(zipped)
	if err != nil {
		t.Fatal(err)
	}
	if back.G.NumEdges() != m.G.NumEdges() || back.G.TotalTraversals != m.G.TotalTraversals {
		t.Fatal("the gzipped round trip lost counts")
	}
}

// The adjacency lists are two parallel slices, with a map index only for the
// high-degree nodes: the lookups must agree either side of that threshold.
func TestAdjacencyKeepsOrderAndFindsEveryEntry(t *testing.T) {
	var a adjacency
	n := 4 * adjacencyIndexAt
	for i := 0; i < n; i++ {
		a.set(i*3, i*7)
	}
	if a.idx == nil {
		t.Fatal("a high-degree adjacency must keep a map index")
	}
	if a.size() != n {
		t.Fatalf("size %d, want %d", a.size(), n)
	}
	for i := 0; i < n; i++ {
		if a.order[i] != i*3 || a.edges[i] != i*7 {
			t.Fatalf("entry %d is %d -> %d", i, a.order[i], a.edges[i])
		}
		if e, ok := a.get(i * 3); !ok || e != i*7 {
			t.Fatalf("get(%d) = %d, %v", i*3, e, ok)
		}
	}
	if _, ok := a.get(1); ok {
		t.Fatal("get must not invent an entry")
	}
	a.set(9, 99) // overwriting keeps the position
	if e, _ := a.get(9); e != 99 || a.size() != n {
		t.Fatalf("overwrite: %d entries, 9 -> %d", a.size(), e)
	}
	a.set(9, 3*7) // back to what the loop wrote
	if !a.unset(0) || a.unset(0) {
		t.Fatal("unset must remove exactly once")
	}
	if a.size() != n-1 {
		t.Fatalf("size after unset: %d", a.size())
	}
	for i := 1; i < n; i++ {
		if e, ok := a.get(i * 3); !ok || e != i*7 {
			t.Fatalf("lost %d after unset: %d, %v", i*3, e, ok)
		}
	}
	a.clear()
	if a.size() != 0 || a.idx != nil {
		t.Fatal("clear must empty the adjacency")
	}
	var small adjacency
	small.set(5, 1)
	small.set(6, 2)
	if small.idx != nil {
		t.Fatal("a small adjacency must not allocate a map")
	}
	if e, ok := small.get(6); !ok || e != 2 {
		t.Fatalf("small get: %d, %v", e, ok)
	}
	if _, ok := small.get(7); ok {
		t.Fatal("small get must not invent an entry")
	}
}
