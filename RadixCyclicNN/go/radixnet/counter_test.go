package radixnet

import (
	"encoding/json"
	"testing"
)

// nearTheLimit winds every counter of a graph to one event before its reset.
func nearTheLimit(g *Graph) {
	step := CounterLimit - 1
	for i := range g.Count {
		g.Count[i] += step
	}
	for e := range g.EdgeCount {
		g.EdgeCount[e] += step
	}
	g.Traversals.Add(step)
	g.TotalTraversals.Add(step)
}

func TestCarryMovesWholeTurnsIntoTheResets(t *testing.T) {
	cases := []struct{ value, resets, wantValue, wantResets int64 }{
		{0, 0, 0, 0},
		{CounterLimit - 1, 0, CounterLimit - 1, 0},
		{CounterLimit, 0, 0, 1},
		{CounterLimit + 7, 2, 7, 3},
		{3*CounterLimit + 5, 0, 5, 3},
		{CounterLimit, CounterLimit - 1, 0, 0}, // the resets wrap too: the pair itself cycles
	}
	for _, c := range cases {
		v, r := Carry(c.value, c.resets)
		if v != c.wantValue || r != c.wantResets {
			t.Fatalf("Carry(%d, %d) = (%d, %d), want (%d, %d)", c.value, c.resets, v, r, c.wantValue, c.wantResets)
		}
	}
}

func TestCounterWrapsAndKeepsTheExactTotal(t *testing.T) {
	c := NewCounter(CounterLimit-2, 0)
	c.Add(1)
	if c.Value != CounterLimit-1 || c.Resets != 0 {
		t.Fatalf("one short of the limit: %+v", c)
	}
	c.Add(1)
	if c.Value != 0 || c.Resets != 1 {
		t.Fatalf("at the limit the counter must go back to 0 and count the reset: %+v", c)
	}
	c.Add(3)
	if c.Float() != float64(CounterLimit)+3 {
		t.Fatalf("total after the reset = %v, want %v", c.Float(), float64(CounterLimit)+3)
	}
	if !NewCounter(CounterLimit-1, 0).Less(c) || c.Less(c) {
		t.Fatal("a wrapped reading must order above the one before it")
	}
	// both digits stay inside float64's exact range, which is why the limit is 10^15
	for _, digit := range []int64{CounterLimit - 1, CounterLimit / 3, 1} {
		if int64(float64(digit)) != digit {
			t.Fatalf("%d does not survive the float64 a model file carries it in", digit)
		}
	}
}

func TestCarrySeriesWrapsASliceAndStaysSparse(t *testing.T) {
	values := []int64{1, CounterLimit + 4, 2, 2 * CounterLimit}
	resets := map[int]int64{}
	if n := CarrySeries(values, resets); n != 2 {
		t.Fatalf("wrapped %d counters, want 2", n)
	}
	want := []int64{1, 4, 2, 0}
	for i, v := range values {
		if v != want[i] {
			t.Fatalf("values = %v, want %v", values, want)
		}
	}
	if len(resets) != 2 || resets[1] != 1 || resets[3] != 2 {
		t.Fatalf("only the ids that wrapped may be stored, got %v", resets)
	}
	if n := CarrySeries(values, resets); n != 0 {
		t.Fatalf("a second sweep wrapped %d counters, want 0", n)
	}
}

func TestVisitCountersWrapWithoutLosingTheirHistory(t *testing.T) {
	m := trained(t, 2, 2)
	g := m.G
	nearTheLimit(g)
	before := make([]float64, len(g.EdgeCount))
	for e := range g.EdgeCount {
		before[e] = g.edgeTraversalsF(e)
	}
	opts := DefaultTrainOptions()
	opts.Epochs = 1
	if _, err := m.Train(corpus(t)[:20], opts); err != nil { // an epoch carries at its end
		t.Fatal(err)
	}
	for i, c := range g.Count {
		if c < 0 || c >= CounterLimit {
			t.Fatalf("node %d did not wrap: %d", i, c)
		}
	}
	for e, c := range g.EdgeCount {
		if c < 0 || c >= CounterLimit {
			t.Fatalf("edge %d did not wrap: %d", e, c)
		}
		if g.edgeTraversalsF(e) < before[e] {
			t.Fatalf("edge %d lost its history: %v < %v", e, g.edgeTraversalsF(e), before[e])
		}
	}
	if len(g.CountResets) == 0 || len(g.EdgeCountResets) == 0 {
		t.Fatal("no reset was recorded")
	}
	if g.TotalTraversals.Resets != 1 {
		t.Fatalf("the traversal total went round %d times, want 1", g.TotalTraversals.Resets)
	}
	if err := g.CheckInvariants(nil, false); err != nil {
		t.Fatal(err)
	}
}

func TestTheWeightsAreWhatTheyWouldBeWithoutWrapping(t *testing.T) {
	m := trained(t, 2, 2)
	g := m.G
	nearTheLimit(g)
	opts := DefaultTrainOptions()
	opts.Epochs = 1
	if _, err := m.Train(corpus(t)[:20], opts); err != nil {
		t.Fatal(err)
	}
	g.Prepare()
	wrapped := append([]float64(nil), g.EdgeW...)

	// the same totals as one plain integer each: the weights must not budge
	for e := range g.EdgeCount {
		if r := g.EdgeCountResets[e]; r != 0 {
			g.EdgeCount[e] += r * CounterLimit
		}
	}
	g.EdgeCountResets = map[int]int64{}
	g.RecomputeWeights()
	for e, w := range g.EdgeW {
		if w != wrapped[e] {
			t.Fatalf("edge %d weight changed across the reset: %v != %v", e, w, wrapped[e])
		}
	}
}

func TestWrappedGraphRoundTripsThroughItsDocument(t *testing.T) {
	m := trained(t, 2, 2)
	nearTheLimit(m.G)
	opts := DefaultTrainOptions()
	opts.Epochs = 1
	if _, err := m.Train(corpus(t)[:20], opts); err != nil {
		t.Fatal(err)
	}
	doc := m.G.ToDoc()
	if doc.FormatVersion != 2 {
		t.Fatalf("graph format version %d, want 2", doc.FormatVersion)
	}
	if doc.Nodes.CountResets == nil || doc.Edges.CountResets == nil {
		t.Fatal("a wrapped graph must write its reset arrays")
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	var back GraphDoc
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	g2, err := GraphFromDoc(&back)
	if err != nil {
		t.Fatal(err)
	}
	again := g2.ToDoc()
	for i, c := range doc.Nodes.Count {
		if again.Nodes.Count[i] != c || again.Nodes.CountResets[i] != doc.Nodes.CountResets[i] {
			t.Fatalf("node %d: (%d, %d) != (%d, %d)", i, again.Nodes.Count[i], again.Nodes.CountResets[i], c, doc.Nodes.CountResets[i])
		}
	}
	for e, c := range doc.Edges.Count {
		if again.Edges.Count[e] != c || again.Edges.CountResets[e] != doc.Edges.CountResets[e] {
			t.Fatalf("edge %d: (%d, %d) != (%d, %d)", e, again.Edges.Count[e], again.Edges.CountResets[e], c, doc.Edges.CountResets[e])
		}
		if again.Edges.W[e] != doc.Edges.W[e] {
			t.Fatalf("edge %d weight changed on reload: %v != %v", e, again.Edges.W[e], doc.Edges.W[e])
		}
	}
	if g2.TotalTraversals != m.G.TotalTraversals || g2.Traversals != m.G.Traversals {
		t.Fatalf("traversal totals changed on reload: %+v / %+v", g2.TotalTraversals, g2.Traversals)
	}
}

func TestAnUnwrappedGraphWritesNoResetArrays(t *testing.T) {
	doc := trained(t, 1, 2).G.ToDoc()
	if doc.Nodes.CountResets != nil || doc.Edges.CountResets != nil {
		t.Fatal("a graph that never wrapped must not write reset arrays")
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"version_resets", "structure_version_resets", "traversals", "traversals_resets"} {
		if _, ok := fields[key]; !ok {
			t.Fatalf("the graph document is missing %q", key)
		}
	}
}

func TestAResetArrayOfTheWrongLengthIsRejected(t *testing.T) {
	for _, block := range []string{"nodes", "edges"} {
		doc := trained(t, 1, 2).G.ToDoc()
		if block == "nodes" {
			doc.Nodes.CountResets = []int64{1, 2}
		} else {
			doc.Edges.CountResets = []int64{1, 2}
		}
		if _, err := GraphFromDoc(doc); err == nil {
			t.Fatalf("%s: a reset array of the wrong length must be rejected", block)
		}
	}
}

func TestAFormat1FileLoadsWithItsCountsWrapped(t *testing.T) {
	doc := trained(t, 1, 2).G.ToDoc()
	doc.FormatVersion = 1
	doc.Traversals, doc.TraversalsResets = 0, 0 // format 1 carried no traversal total
	wanted := make([]int64, len(doc.Edges.Count))
	for e := range doc.Edges.Count {
		doc.Edges.Count[e] += CounterLimit // plain integers, over the limit
		wanted[e] = doc.Edges.Count[e]
	}
	g, err := GraphFromDoc(doc)
	if err != nil {
		t.Fatal(err)
	}
	for e, c := range g.EdgeCount {
		if c < 0 || c >= CounterLimit {
			t.Fatalf("edge %d did not wrap on load: %d", e, c)
		}
		if got := g.edgeTraversalsF(e); got != float64(wanted[e]) {
			t.Fatalf("edge %d: loaded %v events, want %v", e, got, float64(wanted[e]))
		}
	}
	if g.Traversals.Resets == 0 {
		t.Fatal("the traversal total must bound the loaded counts")
	}
}

func TestTheCarrySweepIsSkippedUntilACounterCanHaveWrapped(t *testing.T) {
	g, err := NewGraph(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.ObserveSequence(Encode("hello world"), true); err != nil {
		t.Fatal(err)
	}
	g.Count[2] = CounterLimit + 3 // nothing has counted that far, so nothing is swept
	if n := g.CarryCounters(false); n != 0 || g.Count[2] != CounterLimit+3 {
		t.Fatalf("the guard let a sweep through: %d wrapped, count %d", n, g.Count[2])
	}
	if n := g.CarryCounters(true); n != 1 || g.Count[2] != 3 || g.CountResets[2] != 1 {
		t.Fatalf("forced sweep: %d wrapped, count %d, resets %d", n, g.Count[2], g.CountResets[2])
	}
}

func TestVersionStampsStayExactAcrossAReset(t *testing.T) {
	g, err := NewGraph(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	g.Version = NewCounter(CounterLimit-1, 0)
	stamp := g.Version
	g.Version.Add(1)
	if g.Version.Value != 0 || g.Version.Resets != 1 {
		t.Fatalf("the stamp did not wrap: %+v", g.Version)
	}
	if g.Version == stamp || g.Version == (Counter{}) {
		t.Fatal("a wrapped stamp must not look like an earlier one, or a cache goes stale unnoticed")
	}
}

func TestLifetimeCountersWrapIntoTheirResetField(t *testing.T) {
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	if got := m.MetaAddInt("epochs_total", CounterLimit-1); got != CounterLimit-1 {
		t.Fatalf("epochs_total = %d", got)
	}
	if m.MetaCounter("epochs_total").Resets != 0 {
		t.Fatal("nothing has wrapped yet")
	}
	if got := m.MetaAddInt("epochs_total", 2); got != 1 {
		t.Fatalf("epochs_total after the reset = %d, want 1", got)
	}
	c := m.MetaCounter("epochs_total")
	if c.Resets != 1 || c.Float() != float64(CounterLimit)+1 {
		t.Fatalf("epochs_total = %+v", c)
	}
	stats := m.Stats()
	if stats["epochs_total"] != int64(1) || stats["epochs_total_resets"] != int64(1) {
		t.Fatalf("stats report %v / %v", stats["epochs_total"], stats["epochs_total_resets"])
	}
}
