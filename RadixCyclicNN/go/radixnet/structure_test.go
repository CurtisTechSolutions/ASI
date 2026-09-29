package radixnet

import (
	"math"
	"reflect"
	"testing"
)

// Splits and merges must not change what the count model hangs on its edges -
// the recent shares, the verdicts and the prices - and the bottom beam of the
// least-punished traversal finds the most punished paths (D-092).

const (
	structT1  = "abcdefghij klm" // shares "abcdefghij " with structT2, then parts
	structT2  = "abcdefghij xyz"
	structT3r = "mnop tuv" // shares nothing with the other two
	structT3w = "mnop tuw"
)

func exactModel(t *testing.T, opts GraphOptions) *Model {
	t.Helper()
	m, err := NewModel(1, opts)
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	return m
}

func mustTrain(t *testing.T, m *Model, texts []string, epochs int, autoCompress bool) {
	t.Helper()
	if _, err := m.Train(texts, TrainOptions{Epochs: epochs, AutoCompress: autoCompress}); err != nil {
		t.Fatal(err)
	}
}

func nodeLabelled(t *testing.T, g *Graph, label string) int {
	t.Helper()
	for _, n := range g.AliveNodes() {
		if n >= First && g.Labels[n] == label {
			return n
		}
	}
	t.Fatalf("no node %q", label)
	return -1
}

// forkAt trains one text fifty times and a text that parts from it mid-node
// once: the fork, and (probability, traversals, windowed traversals) of each
// way on by the first three units of the child's label.
func forkAt(t *testing.T, opts GraphOptions, autoCompress bool) (*Graph, int, map[string][3]float64) {
	t.Helper()
	m := exactModel(t, opts)
	mustTrain(t, m, []string{"abcdefghij klmnop"}, 50, autoCompress)
	mustTrain(t, m, []string{"abcdefghij qrstuv"}, 1, autoCompress)
	g := m.G
	fork := -1
	for _, p := range g.AliveNodes() {
		if p >= First && g.Degree(p) == 2 {
			if fork >= 0 {
				t.Fatalf("two forks: %d and %d", fork, p)
			}
			fork = p
		}
	}
	if fork < 0 {
		t.Fatal("no fork")
	}
	ways := map[string][3]float64{}
	for _, cc := range g.ChildCosts(fork) {
		ways[g.Labels[cc.Child][:3]] = [3]float64{math.Exp(-cc.Cost), float64(g.EdgeCount[cc.Edge]), float64(g.WindowEdgeCount[cc.Edge])}
	}
	return g, fork, ways
}

func TestASplitHandsTheBridgeItsWindowHistory(t *testing.T) {
	g, _, compressed := forkAt(t, DefaultGraphOptions(), true)
	_, _, plain := forkAt(t, DefaultGraphOptions(), false)
	// the bridge carries the fifty windowed traversals of the node it came out of, as the chain's edge
	// does, so the compressed graph and the chain price the new branch alike
	for _, key := range []string{"j k", "j q"} {
		if compressed[key][1] != plain[key][1] || compressed[key][2] != plain[key][2] {
			t.Fatalf("%q: (traversals, windowed) %v compressed, %v on the chain", key, compressed[key], plain[key])
		}
		if math.Abs(compressed[key][0]-plain[key][0]) > 1e-12 {
			t.Fatalf("%q: probability %v compressed, %v on the chain", key, compressed[key][0], plain[key][0])
		}
	}
	if compressed["j k"][1] != 50 || compressed["j k"][2] != 50 {
		t.Fatalf("the bridge: %v", compressed["j k"])
	}
	if compressed["j q"][0] > 0.05 {
		t.Fatalf("one traversal in fifty-one is not a fifth of the way on: %v", compressed["j q"][0])
	}
	if err := g.CheckInvariants(nil, false); err != nil {
		t.Fatal(err)
	}
}

func TestTheRewrittenWindowKeepsItsSize(t *testing.T) {
	opts := DefaultGraphOptions()
	opts.Window = 60
	g, fork, ways := forkAt(t, opts, true)
	if g.WindowTraversals() != 60 {
		t.Fatalf("the window holds %d events, not its size", g.WindowTraversals())
	}
	var sum int64
	for _, c := range g.WindowEdgeCount {
		sum += c
	}
	if sum != 60 {
		t.Fatalf("every event in the window is counted once: %d", sum)
	}
	if ways["j k"][2] == 0 {
		t.Fatal("the bridge holds part of the window")
	}
	if ways["j q"][0] > 0.06 {
		t.Fatalf("the new branch: %v", ways["j q"])
	}
	if len(g.Children(fork)) != 2 {
		t.Fatal("the fork")
	}
}

func judgedPair(t *testing.T) *Model {
	t.Helper()
	m := exactModel(t, DefaultGraphOptions())
	mustTrain(t, m, []string{structT1, structT1, structT1, structT1, structT1, structT3r}, 1, true)
	for i := 0; i < 3; i++ { // the model wrote xyz, the teacher wrote klm: blame the xyz step
		if _, err := m.Correct(structT2, structT1, CorrectOptions{Strength: 1, Weight: 1, Reward: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := m.Correct(structT3w, structT3r, CorrectOptions{Strength: 0.3, Weight: 1, Reward: 1}); err != nil {
		t.Fatal(err) // a light blame on the tuw step
	}
	return m
}

// halved trains texts and has the dynamic window halve their shared prefix and hold the halves apart.
func halved(t *testing.T, texts ...string) *Model {
	t.Helper()
	m := exactModel(t, DefaultGraphOptions())
	mustTrain(t, m, texts, 2, true)
	on, off, eight := true, false, 8
	if _, err := m.ConfigureWindow(&on, &eight, &eight, &eight, &off); err != nil {
		t.Fatal(err)
	}
	step, err := m.WindowStep(1, true)
	if err != nil {
		t.Fatal(err)
	}
	if step.Splits != 1 {
		t.Fatalf("one split: %+v", step)
	}
	return m
}

func windowOff(t *testing.T, m *Model) {
	t.Helper()
	off := false
	if _, err := m.ConfigureWindow(&off, nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
}

type walkRow struct {
	text         string
	punish, cost float64
}

func ranking(t *testing.T, m *Model, traversal string, k int) (top, bottom []walkRow) {
	t.Helper()
	o := DefaultPredictOptions()
	o.Length, o.ToEnd, o.MaxLength, o.K, o.Traversal = 0, true, 40, k, traversal
	p, err := m.Predict("", o)
	if err != nil {
		t.Fatal(err)
	}
	rows := func(results []*PathResult) []walkRow {
		out := []walkRow{}
		for _, r := range results {
			out = append(out, walkRow{r.Text, math.Round(r.Punish*1e6) / 1e6, math.Round(r.Cost*1e9) / 1e9})
		}
		return out
	}
	return rows(p.Top), rows(p.Bottom)
}

func rankings(t *testing.T, m *Model) map[string][]walkRow {
	t.Helper()
	out := map[string][]walkRow{}
	for _, trav := range []string{"reward", "least-punished"} {
		out[trav], _ = ranking(t, m, trav, 4)
	}
	return out
}

func TestASplitHandsTheBridgeNoVerdict(t *testing.T) {
	m := judgedPair(t)
	g := m.G
	before := rankings(t, m)
	if before["least-punished"][0].text != structT1 || before["least-punished"][1].text != structT3r {
		t.Fatalf("least punished: %v", before["least-punished"])
	}
	totals := g.PathTotals()
	if n, err := g.SplitWindow(8); err != nil || n != 1 {
		t.Fatalf("one split of the shared prefix: %d %v", n, err)
	}
	a := nodeLabelled(t, g, "abcdefg")
	bridge := g.Children(a)[0].E
	if seen, correct, incorrect := g.EdgePaths(bridge); seen != 0 || correct != 0 || incorrect != 0 {
		t.Fatalf("a forced step carries no verdict: %d %d %d", seen, correct, incorrect)
	}
	if got := g.PathTotals(); got != totals {
		t.Fatalf("the split moved the verdicts, it did not add any: %+v -> %+v", totals, got)
	}
	// the same walks, the same blame, the same prices - a split is invisible to both traversals
	if after := rankings(t, m); !reflect.DeepEqual(after, before) {
		t.Fatalf("a split changed a ranking: %v -> %v", before, after)
	}
	if err := g.CheckInvariants(nil, false); err != nil {
		t.Fatal(err)
	}
}

func TestAMergeKeepsTheVerdictsAndThePrices(t *testing.T) {
	m := halved(t, structT1, structT2)
	g := m.G
	if _, err := m.Correct(structT2, structT1, CorrectOptions{Strength: 1, Weight: 1, Reward: 1}); err != nil {
		t.Fatal(err) // verdicts on the fork, which the first half now calls
	}
	if _, err := m.Punish([]string{structT1}, 1, 0.5); err != nil {
		t.Fatal(err)
	}
	a := nodeLabelled(t, g, "abcdefg")
	b := g.Children(a)[0].P
	rows := func(prev, parent int) map[int]PathRow {
		out := map[int]PathRow{}
		for key, row := range g.paths {
			if key.Prev == prev && g.EdgeParent[key.Edge] == parent {
				out[key.Edge] = *row
			}
		}
		return out
	}
	kept := rows(a, b)
	if len(kept) != 2 {
		t.Fatalf("both ways out of the fork were judged: %v", kept)
	}
	before := rankings(t, m)
	windowOff(t, m)
	if n := g.Compress(); n != 1 {
		t.Fatalf("one merge: %d", n)
	}
	// the fork's contexts are keyed by the merged node's caller now, counters intact
	if got := rows(Start, a); !reflect.DeepEqual(got, kept) {
		t.Fatalf("the verdicts: %v -> %v", kept, got)
	}
	if after := rankings(t, m); !reflect.DeepEqual(after, before) {
		t.Fatalf("a merge changed what the model predicts: %v -> %v", before, after)
	}
	if err := g.CheckInvariants(nil, true); err != nil {
		t.Fatal(err)
	}
}

func TestAMergeCarriesWhatTheForcedStepAloneWasTaught(t *testing.T) {
	m := halved(t, structT1)
	g := m.G
	a := g.Children(Start)[0].P // the first half of the only text, held apart from the second
	bridge := g.Children(a)[0].E
	into := g.Children(Start)[0].E
	g.AddReward([]int{bridge}, -1) // blame the forced step alone
	if top, _ := ranking(t, m, "least-punished", 1); top[0].punish != 1 {
		t.Fatalf("the blame: %v", top)
	}
	pos, neg := g.TotalReward()
	windowOff(t, m)
	if n := g.Compress(); n != 1 {
		t.Fatalf("one merge: %d", n)
	}
	if g.EdgeReward[into] != -1 {
		t.Fatalf("the blame moved onto the step into the merged node: %v", g.EdgeReward[into])
	}
	if p, n := g.TotalReward(); p != pos || n != neg {
		t.Fatalf("the ledger: %v %v -> %v %v", pos, neg, p, n)
	}
	if top, _ := ranking(t, m, "least-punished", 1); top[0].punish != 1 {
		t.Fatalf("the blame after the merge: %v", top)
	}

	// a pass that penalised every edge of the path moves nothing: the step into the node explains it
	m = halved(t, structT1)
	g = m.G
	if _, err := m.Punish([]string{structT1}, 1, 0.5); err != nil {
		t.Fatal(err)
	}
	into = g.Children(Start)[0].E
	prices, _ := ranking(t, m, "reward", 1)
	windowOff(t, m)
	if n := g.Compress(); n != 1 {
		t.Fatalf("one merge: %d", n)
	}
	if g.EdgeReward[into] != -0.5 {
		t.Fatalf("the step into the merged node keeps its own penalty: %v", g.EdgeReward[into])
	}
	if after, _ := ranking(t, m, "reward", 1); !reflect.DeepEqual(after, prices) {
		t.Fatalf("the prices: %v -> %v", prices, after)
	}
}

func TestTheBottomBeamFindsTheMostPunishedPaths(t *testing.T) {
	m := judgedPair(t)
	top, bottom := ranking(t, m, "least-punished", 1)
	if top[0].text != structT1 || len(bottom) != 1 || bottom[0].text != structT2 {
		t.Fatalf("the walk the corrections blamed three times over: %v / %v", top, bottom)
	}
	if bottom[0].punish < 4 {
		t.Fatalf("its blame: %v", bottom[0])
	}
	if _, byCost := ranking(t, m, "reward", 1); len(byCost) != 1 || byCost[0].text != structT2 {
		t.Fatalf("by cost it is the dearest walk too: %v", byCost)
	}
	_, two := ranking(t, m, "least-punished", 2)
	if len(two) != 2 || two[0].text != structT2 || two[1].text != structT3w {
		t.Fatalf("most punished first: %v", two)
	}
}
