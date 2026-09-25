package radixnet

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The dynamic window (window.go, ../../SPEC-DynamicWindow.md): the ladder and
// what it refuses, a step halving every node longer than the window into two
// halves joined by a bridge that carries the node's count, compression that
// stops at the window, the cycle back up regrowing what stayed unary, the
// automatic step at the end of every epoch, and the file block.

var windowTexts = []string{
	"the quick brown fox jumps over the lazy dog while the cat sat on the mat and the bird sang in the tree all afternoon",
	"the quick brown fox jumps over the lazy dog while the cat ran to the door",
	"a bird sang in the tree all afternoon",
}

func windowModel(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	if _, err := m.Train(windowTexts, TrainOptions{Epochs: 1, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	return m
}

func longestLabel(g *Graph) int {
	_, longest := g.LongerThan(0)
	return longest
}

func ints(v ...int) []int { return v }

func TestLadderHalvesFromTheTopToTheFloor(t *testing.T) {
	if got := Ladder(32, 4); !reflect.DeepEqual(got, ints(32, 16, 8, 4)) {
		t.Fatalf("ladder: %v", got)
	}
	if got := Ladder(8, 8); !reflect.DeepEqual(got, ints(8)) {
		t.Fatalf("one rung: %v", got)
	}
	for _, bad := range [][3]int{{20, 4, 0}, {32, 3, 0}, {4, 32, 0}, {32, 4, 12}, {32, 4, 64}, {32, 8, 4}} {
		if err := CheckLadder(bad[0], bad[1], bad[2]); err == nil {
			t.Fatalf("%v should be refused", bad)
		}
	}
	if err := CheckLadder(32, 4, 8); err != nil {
		t.Fatal(err)
	}
	w, err := NewDynamicWindow(32, 4, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	seen := []int{}
	for i := 0; i < 6; i++ {
		seen = append(seen, w.Size)
		w = w.Advanced()
	}
	if !reflect.DeepEqual(seen, ints(32, 16, 8, 4, 32, 16)) {
		t.Fatalf("the ladder goes round: %v", seen)
	}
	off := DynamicWindow{}
	if off.On || off.Next() != 0 || off.String() != "off" || len(off.Sizes()) != 0 || off.Advanced() != off {
		t.Fatalf("off: %+v", off)
	}
}

func TestANodeLongerThanTheWindowIsHalvedAtItsMiddleGram(t *testing.T) {
	g, _ := NewGraph(1, DefaultGraphOptions())
	if _, err := g.ObserveSequence(g.Encode("abcdefghij"), true); err != nil {
		t.Fatal(err)
	}
	g.Compress()
	labels := func() []string {
		out := []string{}
		for n := First; n < len(g.Labels); n++ {
			if g.Alive[n] {
				out = append(out, g.Labels[n])
			}
		}
		return out
	}
	if !reflect.DeepEqual(labels(), []string{"abcdefghij"}) {
		t.Fatalf("one merged node: %v", labels())
	}
	if n, _ := g.SplitWindow(8); n != 1 || !reflect.DeepEqual(labels(), []string{"abcdef", "efghij"}) {
		t.Fatalf("halved at 8: %d %v", n, labels())
	}
	if n, _ := g.SplitWindow(4); n != 2 {
		t.Fatalf("halved at 4: %d %v", n, labels())
	}
	want := map[string]bool{"abcd": true, "cdef": true, "efgh": true, "ghij": true}
	for _, l := range labels() {
		if !want[l] {
			t.Fatalf("at 4: %v", labels())
		}
	}
	if n, _ := g.SplitWindow(4); n != 0 {
		t.Fatal("nothing is longer than the window now")
	}
	if n, _ := g.SplitWindow(3); n != 4 {
		t.Fatalf("every node halves into two grams: %d", n)
	}
	if n, _ := g.SplitWindow(1); n != 0 {
		t.Fatal("a node of one gram cannot be halved")
	}
	if _, err := g.SplitWindow(0); err == nil {
		t.Fatal("size 0 must be refused")
	}
	if err := g.CheckInvariants([]string{"abcdefghij"}, false); err != nil {
		t.Fatal(err)
	}
}

func TestTheHalvesCarryTheCountAndTheBridgeIsHeavyByIt(t *testing.T) {
	g, _ := NewGraph(1, DefaultGraphOptions())
	for i := 0; i < 3; i++ {
		if _, err := g.ObserveSequence(g.Encode("abcdefghij"), true); err != nil {
			t.Fatal(err)
		}
	}
	g.Compress()
	node := First
	for !g.Alive[node] {
		node++
	}
	count := g.Count[node]
	if count != 3 {
		t.Fatalf("count %d", count)
	}
	if n, _ := g.SplitWindow(8); n != 1 {
		t.Fatal("one split")
	}
	a, b := node, len(g.Labels)-1
	if g.Count[b] != count || g.Count[a] != count {
		t.Fatalf("both halves carry the count: %d %d", g.Count[a], g.Count[b])
	}
	e, ok := g.Edge(a, b)
	if !ok || g.EdgeCount[e] != count {
		t.Fatalf("the bridge carries every traversal of the node: %v %d", ok, g.EdgeCount[e])
	}
	g.Prepare()
	costs := g.ChildCosts(a)
	if len(costs) != 1 || costs[0].Child != b || costs[0].Cost != 0 {
		t.Fatalf("the only way on: %v", costs)
	}
	// another child attaching to the first half starts with one traversal against the bridge's three
	if _, err := g.ObserveSequence(g.Encode("abcdefxyz"), true); err != nil {
		t.Fatal(err)
	}
	g.Prepare()
	best, bestCost := -1, 1e9
	for _, c := range g.ChildCosts(a) {
		if c.Cost < bestCost {
			best, bestCost = c.Child, c.Cost
		}
	}
	if best != b {
		t.Fatalf("the bridge stays the likeliest way on: %v", g.ChildCosts(a))
	}
	if err := g.CheckInvariants([]string{"abcdefghij", "abcdefxyz"}, false); err != nil {
		t.Fatal(err)
	}
}

func TestCompressionStopsAtTheWindow(t *testing.T) {
	m := windowModel(t)
	g := m.G
	if longestLabel(g) <= 16 {
		t.Fatalf("the corpus should merge into a node longer than 16: %d", longestLabel(g))
	}
	if _, err := m.ConfigureWindow(nil, intp(16), intp(4), intp(16), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := g.SplitWindow(16); err != nil {
		t.Fatal(err)
	}
	if longestLabel(g) > 16 {
		t.Fatalf("longest %d", longestLabel(g))
	}
	if merges := g.Compress(); merges != 0 {
		t.Fatalf("the window holds the halves apart: %d merges", merges)
	}
	if err := g.CheckInvariants(windowTexts, true); err != nil {
		t.Fatal(err)
	}
	if _, err := m.ConfigureWindow(yes(false), nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if merges := g.Compress(); merges == 0 || longestLabel(g) <= 16 {
		t.Fatalf("off: the unary halves merge back (%d merges, longest %d)", merges, longestLabel(g))
	}
	if err := g.CheckInvariants(windowTexts, true); err != nil {
		t.Fatal(err)
	}
}

func TestAStepMergesHalvesAndMovesAndTheTopRegrows(t *testing.T) {
	m := windowModel(t)
	g := m.G
	if _, err := m.WindowStep(1, true); err == nil {
		t.Fatal("off: nothing to step")
	}
	if _, err := m.ConfigureWindow(yes(true), nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	nodes := g.NumNodes()
	done, err := m.WindowStep(4, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(done.Sizes, ints(32, 16, 8, 4)) || done.To != 32 || done.Splits == 0 || longestLabel(g) > 4 {
		t.Fatalf("four steps: %+v longest %d", done, longestLabel(g))
	}
	if done.NodesAfter-done.NodesBefore != done.Splits || g.NumNodes() <= nodes {
		t.Fatalf("every split is one more node: %+v", done)
	}
	again, err := m.WindowStep(1, true)
	if err != nil {
		t.Fatal(err)
	}
	if again.Merges == 0 || again.Splits != 0 || g.NumNodes() != nodes {
		t.Fatalf("at the top, what stayed unary merges back: %+v nodes %d of %d", again, g.NumNodes(), nodes)
	}
	if _, err := m.WindowStep(0, true); err == nil {
		t.Fatal("steps must be >= 1")
	}
	for _, text := range windowTexts {
		if s := m.Score(text); s.UnknownTransitions != 0 {
			t.Fatalf("%q: %d unknown transitions after the steps", text, s.UnknownTransitions)
		}
	}
}

func TestTheSettings(t *testing.T) {
	m, _ := NewModel(1, DefaultGraphOptions())
	cfg := m.WindowConfig()
	if cfg.On || cfg.Size != nil || len(cfg.Sizes) != 0 || cfg.Longer != nil || cfg.Heavy != nil || cfg.Floor != 4 || !cfg.Auto {
		t.Fatalf("off: %+v", cfg)
	}
	if cfg, _ = m.ConfigureWindow(nil, nil, nil, nil, nil); cfg.On {
		t.Fatal("nothing asked, nothing changed")
	}
	cfg, err := m.ConfigureWindow(yes(true), nil, nil, nil, nil)
	if err != nil || *cfg.Top != 32 || cfg.Floor != 4 || *cfg.Size != 32 || !cfg.Auto || *cfg.Next != 16 || !reflect.DeepEqual(cfg.Sizes, ints(32, 16, 8, 4)) {
		t.Fatalf("on: %+v %v", cfg, err)
	}
	if cfg, _ = m.ConfigureWindow(nil, nil, nil, intp(8), yes(false)); *cfg.Size != 8 || cfg.Auto {
		t.Fatalf("size 8, by hand: %+v", cfg)
	}
	if cfg, _ = m.ConfigureWindow(nil, intp(16), nil, nil, nil); *cfg.Top != 16 || *cfg.Size != 8 || cfg.Auto {
		t.Fatalf("a new top keeps the size on the ladder: %+v", cfg)
	}
	if cfg, _ = m.ConfigureWindow(nil, nil, intp(16), nil, nil); cfg.Floor != 16 || *cfg.Size != 16 || !reflect.DeepEqual(cfg.Sizes, ints(16)) {
		t.Fatalf("a new floor lifts the size: %+v", cfg)
	}
	for _, bad := range []struct{ top, floor, size *int }{{intp(20), nil, nil}, {nil, intp(3), nil}, {nil, intp(128), nil}, {nil, nil, intp(12)}, {nil, nil, intp(1)}} {
		if _, err := m.ConfigureWindow(nil, bad.top, bad.floor, bad.size, nil); err == nil {
			t.Fatalf("%v should be refused", bad)
		}
	}
	if cfg, _ = m.ConfigureWindow(yes(false), nil, nil, intp(8), nil); cfg.On || cfg.Size != nil {
		t.Fatalf("off: %+v", cfg)
	}
	if m.Stats()["dynamic_window"] != nil {
		t.Fatal("stats: null while off")
	}
}

func TestItStepsByItselfAtTheEndOfEveryEpoch(t *testing.T) {
	m := windowModel(t)
	if _, err := m.ConfigureWindow(yes(true), nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	records, err := m.Train(windowTexts, TrainOptions{Epochs: 5, AutoCompress: true})
	if err != nil {
		t.Fatal(err)
	}
	windows, splits := []int{}, []bool{}
	for _, r := range records {
		windows = append(windows, r["window"].(int))
		splits = append(splits, r["splits"].(int) > 0)
	}
	if !reflect.DeepEqual(windows, ints(32, 16, 8, 4, 32)) || !reflect.DeepEqual(splits, []bool{false, true, true, true, false}) {
		t.Fatalf("windows %v splits %v", windows, splits)
	}
	if records[4]["merges"].(int) == 0 {
		t.Fatal("back at the top, the epoch's compression regrew the chains")
	}
	if m.G.DynamicWindow.Size != 16 || m.Stats()["dynamic_window"] != 16 {
		t.Fatalf("the window stands at 16: %+v", m.G.DynamicWindow)
	}
	for _, text := range windowTexts {
		if s := m.Score(text); s.UnknownTransitions != 0 {
			t.Fatalf("%q: %d unknown transitions", text, s.UnknownTransitions)
		}
	}
	if _, err := m.ConfigureWindow(nil, nil, nil, nil, yes(false)); err != nil {
		t.Fatal(err)
	}
	more, _ := m.Train(windowTexts, TrainOptions{Epochs: 2, AutoCompress: true})
	if _, has := more[0]["window"]; has || m.G.DynamicWindow.Size != 16 {
		t.Fatal("by hand: the ladder does not move")
	}
}

func TestItTravelsWithTheFileAndOffIsTheOldFile(t *testing.T) {
	m := windowModel(t)
	dir := t.TempDir()
	before, _ := json.Marshal(m.G.ToDoc())
	if _, err := m.ConfigureWindow(nil, intp(16), intp(4), intp(8), yes(false)); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "model.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	block := doc["graph"].(map[string]any)["dynamic_window"]
	if !reflect.DeepEqual(block, map[string]any{"top": 16.0, "floor": 4.0, "size": 8.0, "auto": false}) {
		t.Fatalf("block: %v", block)
	}
	again, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if again.G.DynamicWindow != (DynamicWindow{On: true, Top: 16, Floor: 4, Size: 8, Auto: false}) || again.Stats()["dynamic_window"] != 8 {
		t.Fatalf("loaded: %+v", again.G.DynamicWindow)
	}
	if _, err := m.ConfigureWindow(yes(false), nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(m.G.ToDoc())
	if string(before) != string(after) {
		t.Fatal("off is the old document to the byte")
	}
	bad := filepath.Join(dir, "bad.json")
	os.WriteFile(bad, []byte(`{"format":"radixnet-count","version":1,"graph":{"format":"radixnet-graph","format_version":4,"dynamic_window":{"top":12},"seed":1,"nodes":{"labels":["<s>","</s>","<back>","<think>"],"z":[0,0,0,0],"a":[0,0,0,0],"b":[0,0,0,0],"h":[0,0,0,0],"k":[1,1,1,1],"count":[0,0,0,0]},"edges":{"src":[],"dst":[],"w":[],"count":[]}}}`), 0o644)
	if _, err := Load(bad); err == nil {
		t.Fatal("a top that is not a power of two is refused on load")
	}
}

func intp(v int) *int { return &v }
