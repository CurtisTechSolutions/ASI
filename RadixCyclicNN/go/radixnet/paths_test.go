package radixnet

import "testing"

// judged trains a tiny corpus and marks one sentence right, one wrong.
func judged(t *testing.T, corpus []string, good, bad string) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	if _, err := m.Train(corpus, TrainOptions{Epochs: 3, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Reward([]string{good}, 1, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Punish([]string{bad}, 1, 1); err != nil {
		t.Fatal(err)
	}
	return m
}

// walk is [(prev, parent, edge)] of a text: what called each step.
func walk(t *testing.T, m *Model, text string) [][3]int {
	t.Helper()
	transitions, _, ok := m.G.Trace(Encode(text))
	if !ok {
		t.Fatalf("%q does not walk", text)
	}
	out := [][3]int{}
	prev := Start
	for i, tr := range transitions {
		if i > 0 {
			prev = transitions[i-1].P
		}
		out = append(out, [3]int{prev, tr.P, tr.E})
	}
	return out
}

func TestTrainingCountsNoPathsAndAJudgementStartsTheTable(t *testing.T) {
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	corpus := []string{"the cat sat on the mat", "the dog ate the bone"}
	if _, err := m.Train(corpus, TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	if got := m.G.PathTotals(); got.Contexts != 0 { // a corpus is not a judgement
		t.Fatalf("training should count no paths: %+v", got)
	}
	if _, err := m.Reward([]string{"the cat sat on the mat"}, 1, 1); err != nil {
		t.Fatal(err)
	}
	first := m.G.PathTotals()
	if first.Contexts == 0 || first.Correct != first.Seen || first.Incorrect != 0 {
		t.Fatalf("a rewarded walk is a correct path: %+v", first)
	}
	if _, err := m.Punish([]string{"the dog ate the bone"}, 1, 1); err != nil {
		t.Fatal(err)
	}
	second := m.G.PathTotals()
	if second.Incorrect == 0 {
		t.Fatalf("a punished walk is an incorrect path: %+v", second)
	}
	// a later training pass keeps the counters up to date without inventing contexts
	if _, err := m.Train([]string{"the cat sat on the mat"}, TrainOptions{Epochs: 1, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	third := m.G.PathTotals()
	if third.Contexts != second.Contexts || third.Seen <= second.Seen ||
		third.Correct != second.Correct || third.Incorrect != second.Incorrect {
		t.Fatalf("an unjudged pass only counts what it sees: %+v -> %+v", second, third)
	}
}

// The point of counting paths: the same edge is the right move after one word
// and the wrong one after another, which no per-edge counter can tell apart.
func TestTheSameEdgeIsRightAfterOneWordAndWrongAfterAnother(t *testing.T) {
	good, bad := "a cat sat", "the cat sat"
	m := judged(t, []string{good, bad, "a cat ran"}, good, bad)
	g := m.G
	from := map[[2]int]int{}
	for _, step := range walk(t, m, good) {
		from[[2]int{step[1], step[2]}] = step[0]
	}
	edge, goodPrev, badPrev, parent := -1, -1, -1, -1
	for _, step := range walk(t, m, bad) {
		key := [2]int{step[1], step[2]}
		if was, ok := from[key]; ok && was != step[0] && g.Degree(step[1]) > 1 {
			edge, goodPrev, badPrev, parent = step[2], was, step[0], step[1]
			break
		}
	}
	if edge < 0 {
		t.Fatal("the two sentences should share a step, reached from different nodes, with a choice")
	}
	if g.PathTerm(goodPrev, edge) <= 0 || g.PathTerm(badPrev, edge) >= 0 {
		t.Fatalf("the step should be right from %d and wrong from %d: %v / %v",
			goodPrev, badPrev, g.PathTerm(goodPrev, edge), g.PathTerm(badPrev, edge))
	}
	costOf := func(costs []ChildCost) float64 {
		for _, cc := range costs {
			if cc.Edge == edge {
				return cc.Cost
			}
		}
		t.Fatalf("edge %d is not a child of %d", edge, parent)
		return 0
	}
	plain := costOf(g.ChildCosts(parent))
	if costOf(g.ChildCostsFrom(parent, goodPrev)) >= plain {
		t.Fatal("the step should be cheaper for the walk that was right here")
	}
	if costOf(g.ChildCostsFrom(parent, badPrev)) <= plain {
		t.Fatal("the step should be dearer for the walk that was wrong here")
	}
	stats := g.PathStatsOf(goodPrev, edge)
	if stats == nil || stats.Correct != 1 || stats.Incorrect != 0 || *stats.CorrectRatio != 1 || *stats.SeenRatio <= 0 {
		t.Fatalf("the ratios of a judged context: %+v", stats)
	}
	if g.PathStatsOf(goodPrev, 10000) != nil {
		t.Fatal("an unknown context has no stats")
	}
}

func TestPathCountersSurviveARoundTrip(t *testing.T) {
	good, bad := "a cat sat", "the cat sat"
	m := judged(t, []string{good, bad, "a cat ran"}, good, bad)
	before := m.G.PathTotals()
	again, err := FromDoc(m.ToDoc())
	if err != nil {
		t.Fatal(err)
	}
	if got := again.G.PathTotals(); got != before {
		t.Fatalf("the counters should survive a round trip: %+v -> %+v", before, got)
	}
	if again.G.WeightConfig().PathScale != 1 {
		t.Fatalf("path_scale should survive: %v", again.G.WeightConfig().PathScale)
	}
	for _, prefix := range []string{"a cat", "the cat", "a "} {
		a, err := m.Predict(prefix, PredictOptions{Length: 6, Mode: "beam", K: 3})
		if err != nil {
			t.Fatal(err)
		}
		b, err := again.Predict(prefix, PredictOptions{Length: 6, Mode: "beam", K: 3})
		if err != nil {
			t.Fatal(err)
		}
		if a.FullText != b.FullText {
			t.Fatalf("%q: %q != %q", prefix, a.FullText, b.FullText)
		}
	}
}

func TestCompressionKeepsTheContextsThatAreStillAChoice(t *testing.T) {
	good, bad := "a cat sat", "the cat sat"
	m := judged(t, []string{good, bad, "a cat ran"}, good, bad)
	before := m.G.PathTotals()
	m.G.Compress()
	after := m.G.PathTotals()
	if after.Contexts > before.Contexts || after.Correct > before.Correct {
		t.Fatalf("a merge can only drop forced steps: %+v -> %+v", before, after)
	}
	if err := m.G.CheckInvariants(nil, false); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{good, bad} {
		if _, ok := m.G.NodePath(Encode(text)); !ok {
			t.Fatalf("%q no longer walks after compression", text)
		}
	}
}

func TestPathScaleCanBeTurnedOff(t *testing.T) {
	good, bad := "a cat sat", "the cat sat"
	m := judged(t, []string{good, bad, "a cat ran"}, good, bad)
	g := m.G
	prev, parent := -1, -1
	for key := range g.paths {
		if p := g.EdgeParent[key.Edge]; g.Degree(p) > 1 {
			prev, parent = key.Prev, p
			break
		}
	}
	if parent < 0 {
		t.Fatal("expected a judged context at a node with a choice")
	}
	withPaths := append([]ChildCost(nil), g.ChildCostsFrom(parent, prev)...)
	if err := g.Configure(map[string]float64{"path_scale": 0}); err != nil {
		t.Fatal(err)
	}
	if g.WeightConfig().PathScale != 0 {
		t.Fatalf("path_scale: %v", g.WeightConfig().PathScale)
	}
	plain := g.ChildCosts(parent)
	off := g.ChildCostsFrom(parent, prev)
	for i := range off {
		if off[i].Cost != plain[i].Cost {
			t.Fatalf("with the scale off a context costs what its edge costs: %v vs %v", off, plain)
		}
		if withPaths[i].Cost == off[i].Cost && withPaths[i].Edge == off[i].Edge && i == 0 {
			t.Fatal("the scale should have changed something while it was on")
		}
	}
}
