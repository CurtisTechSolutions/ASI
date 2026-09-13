package radixnet

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var negativeFailures = []string{"the the the the cat", "the cat cat cat sat"}
var negativeGood = []string{"the cat sat on the mat", "the dog sat on the log"}

func taught(t *testing.T) *Model {
	t.Helper()
	m, err := NewNegativeModel(1, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("new negative model: %v", err)
	}
	if _, err := m.Blame(negativeFailures, BlameOptions{Reason: "repetition", Source: "review", Note: "it repeats itself"}); err != nil {
		t.Fatalf("blame: %v", err)
	}
	return m
}

func TestNegativeWeightFunction(t *testing.T) {
	g, err := NewNegativeGraph(0, NegativeOptions{ShareScale: 1, BlameScale: 0.5, ClearScale: 1})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Log((3+Smoothing)/(4+2*Smoothing)) + 0.5*math.Log1p(3)
	if got := g.NegativeEdgeWeight(3, 4, 2); math.Abs(got-want) > 1e-12 {
		t.Fatalf("edge weight %v, want %v", got, want)
	}
	if g.NegativeWeightConfig().Function != "blame" {
		t.Fatalf("the weight function is called blame: %+v", g.NegativeWeightConfig())
	}
}

func TestEvidenceIsBlameMinusClearing(t *testing.T) {
	m := taught(t)
	g := m.G
	edge := -1
	for e, alive := range g.EdgeAlive {
		if alive && g.Neg.Blame[e] > 0 {
			edge = e
			break
		}
	}
	if edge < 0 {
		t.Fatal("nothing was blamed")
	}
	blame := g.Neg.Blame[edge]
	if got := g.Evidence(edge); math.Abs(got-blame) > 1e-12 {
		t.Fatalf("evidence %v, want the blame %v", got, blame)
	}
	g.RecordClear([]int{edge}, blame/2)
	if got := g.Evidence(edge); math.Abs(got-blame/2) > 1e-12 {
		t.Fatalf("clearing cancels blame: %v", got)
	}
	g.RecordClear([]int{edge}, blame) // cleared more often than blamed: no evidence left, never negative
	if got := g.Evidence(edge); got != 0 {
		t.Fatalf("evidence never goes negative: %v", got)
	}
}

func TestBlameRecordsTheReasonAndTheJournal(t *testing.T) {
	m, _ := NewNegativeModel(1, DefaultNegativeOptions())
	if m.G.NumNodes() != 2 {
		t.Fatalf("a fresh negative network holds the two sentinels only: %d", m.G.NumNodes())
	}
	records, err := m.Blame([]string{"the the the the cat"}, BlameOptions{
		Reason: "Repetition ", Severity: 2, Source: "review", Note: "  it repeats\n the same word ",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 {
		t.Fatalf("one pass, one record: %d", len(records))
	}
	record := records[0]
	if record["phase"] != "negative" || record["reason"] != "repetition" || toFloat(record["severity"]) != 2 {
		t.Fatalf("the record carries the lesson: %+v", record)
	}
	if toFloat(record["edges_touched"]) == 0 || m.G.NumNodes() <= 2 {
		t.Fatalf("blaming builds structure: %+v", record)
	}
	reasons := m.Reasons()
	if len(reasons) != 1 || reasons[0].Reason != "repetition" {
		t.Fatalf("the reason is normalised and registered: %+v", reasons)
	}
	entry := m.Recent(1)
	if len(entry) != 1 || entry[0].Note != "it repeats the same word" || entry[0].Source != "review" {
		t.Fatalf("the journal keeps the tutor's words: %+v", entry)
	}
	if m.Kind() != "negative" {
		t.Fatalf("kind %q", m.Kind())
	}
}

func TestClearNeverCreatesStructure(t *testing.T) {
	m := taught(t)
	nodes, edges := m.G.NumNodes(), m.G.NumEdges()
	records, err := m.Clear([]string{"a sentence that never failed"}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if m.G.NumNodes() != nodes || m.G.NumEdges() != edges {
		t.Fatalf("clearing creates nothing: %d/%d -> %d/%d", nodes, edges, m.G.NumNodes(), m.G.NumEdges())
	}
	last := records[len(records)-1]
	if toFloat(last["matched"]) != 0 || toFloat(last["unmatched"]) != 1 {
		t.Fatalf("nothing matched: %+v", last)
	}
	before := m.Judge(negativeGood[0], DefaultJudgeOptions()).Blame
	if _, err := m.Clear([]string{negativeGood[0]}, 1, 1); err != nil {
		t.Fatal(err)
	}
	if after := m.Judge(negativeGood[0], DefaultJudgeOptions()).Blame; !(after < before) {
		t.Fatalf("clearing lifts blame off shared fragments: %v -> %v", before, after)
	}
}

func TestBlamedTransitionsSurviveCompression(t *testing.T) {
	m, _ := NewNegativeModel(3, DefaultNegativeOptions())
	if _, err := m.BlameCorrection("the cat sit on the mat", "the cat sits on the mat", BlameOptions{Reason: "agreement"}); err != nil {
		t.Fatal(err)
	}
	before := m.Judge("the cat sit on the mat", DefaultJudgeOptions())
	m.G.Compress()
	after := m.Judge("the cat sit on the mat", DefaultJudgeOptions())
	if len(after.Spans) == 0 || after.Spans[0].Fragment != before.Spans[0].Fragment {
		t.Fatalf("compression must not fold the blamed step away: %+v -> %+v", before.Spans, after.Spans)
	}
	if math.Abs(after.Blame-before.Blame) > 1e-9 {
		t.Fatalf("the blame survives compression: %v -> %v", before.Blame, after.Blame)
	}
}

func TestCorrectionBlamesOnlyWhatChanged(t *testing.T) {
	m, _ := NewNegativeModel(1, DefaultNegativeOptions())
	out, err := m.BlameCorrection("the cat sit on the mat", "the cat sits on the mat", BlameOptions{
		Reason: "agreement", Severity: 1.5, Source: "tutor", Note: "the verb must agree with the subject",
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Edits != 1 || out.Blamed != 1 {
		t.Fatalf("one edit, one blamed step: %+v", out)
	}
	if out.Cleared < 1 {
		t.Fatalf("the correction clears what it kept: %+v", out)
	}
	verdict := m.Judge("the cat sit on the mat", DefaultJudgeOptions())
	if len(verdict.Spans) != 1 || verdict.Spans[0].Fragment != "sit " {
		t.Fatalf("only the changed fragment is blamed: %+v", verdict.Spans)
	}
	if verdict.Reasons[0].Reason != "agreement" {
		t.Fatalf("the tutor's mistake is the reason: %+v", verdict.Reasons)
	}
	if m.Judge("the cat sits on the mat", DefaultJudgeOptions()).Verdict != "pass" {
		t.Fatal("the correction itself passes")
	}
	// an unchanged sentence blames nothing
	m2, _ := NewNegativeModel(2, DefaultNegativeOptions())
	same, err := m2.BlameCorrection("the cat sat on the mat", "the cat sat on the mat", BlameOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if same.Edits != 0 || same.Blamed != 0 || m2.G.Neg.TotalBlame != 0 {
		t.Fatalf("nothing changed, nothing blamed: %+v", same)
	}
}

func TestJudge(t *testing.T) {
	m := taught(t)
	verdict := m.Judge(negativeFailures[0], DefaultJudgeOptions())
	if verdict.Verdict != "reject" || verdict.Coverage != 1 || verdict.Risk < verdict.Threshold {
		t.Fatalf("a known failure is rejected: %+v", verdict)
	}
	if len(verdict.Reasons) == 0 || verdict.Reasons[0].Reason != "repetition" {
		t.Fatalf("with its reason: %+v", verdict.Reasons)
	}
	if len(verdict.Spans) == 0 {
		t.Fatal("and the fragments carrying it")
	}
	share := 0.0
	for _, row := range verdict.Reasons {
		share += row.Share
	}
	if math.Abs(share-1) > 1e-9 {
		t.Fatalf("the reason shares add up to 1: %v", share)
	}
	unseen := m.Judge("something entirely different", DefaultJudgeOptions())
	if unseen.Verdict != "pass" || unseen.Blame != 0 || unseen.Why != "nothing here has failed before" {
		t.Fatalf("unseen text passes: %+v", unseen)
	}
	zero, high := 0.0, 1e9
	if m.Judge(negativeGood[0], JudgeOptions{Threshold: &zero, MinCoverage: &zero}).Verdict != "reject" {
		t.Fatal("a zero threshold rejects anything with blame on it")
	}
	if m.Judge(negativeFailures[0], JudgeOptions{Threshold: &high}).Verdict != "suspect" {
		t.Fatal("an unreachable threshold leaves a known failure merely suspect")
	}
	if m.metaInt("judgements") == 0 {
		t.Fatal("judgements are counted")
	}
}

func TestRiskIsPerTransition(t *testing.T) {
	m, _ := NewNegativeModel(9, DefaultNegativeOptions())
	if _, err := m.Blame([]string{negativeGood[0]}, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	verdict := m.Judge(negativeGood[0], DefaultJudgeOptions())
	if math.Abs(verdict.Risk-1) > 1e-9 {
		t.Fatalf("a text blamed once scores 1: %v", verdict.Risk)
	}
	if _, err := m.Blame([]string{negativeGood[0]}, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	if again := m.Judge(negativeGood[0], DefaultJudgeOptions()); math.Abs(again.Risk-2) > 1e-9 {
		t.Fatalf("blamed twice scores 2: %v", again.Risk)
	}
}

func TestCrossingsAndPrediction(t *testing.T) {
	m := taught(t)
	crossings := m.Crossings(negativeFailures[0])
	for _, c := range crossings {
		if c.Edge < 0 {
			t.Fatalf("every crossing of a known failure is an edge: %+v", c)
		}
		if c.End < c.Start {
			t.Fatalf("a span runs forwards: %+v", c)
		}
	}
	if len(m.Crossings("")) != 0 {
		t.Fatal("an empty text has no crossings")
	}
	for _, c := range m.Crossings("zzz qqq vvv") {
		if c.Edge >= 0 {
			t.Fatalf("unknown text crosses nothing: %+v", c)
		}
	}
	found, err := m.Predict("the ", PredictOptions{Length: 8, K: 2, Mode: "beam"})
	if err != nil {
		t.Fatal(err)
	}
	if len(found.Top) == 0 || found.Top[0].Text == "" {
		t.Fatalf("the negative network predicts the ways to fail: %+v", found.Top)
	}
}

func TestForgetAndInvert(t *testing.T) {
	m, _ := NewNegativeModel(6, DefaultNegativeOptions())
	m.Blame([]string{negativeFailures[0]}, BlameOptions{Reason: "repetition"})
	m.Blame([]string{negativeFailures[1]}, BlameOptions{Reason: "gibberish"})
	total := m.G.Neg.TotalBlame
	result, err := m.Forget("gibberish", 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.BlameRemoved <= 0 || m.G.Neg.TotalBlame >= total {
		t.Fatalf("forgetting removes blame: %+v", result)
	}
	for _, row := range m.Reasons() {
		if row.Reason == "gibberish" && row.Blame != 0 {
			t.Fatalf("the reason is forgotten: %+v", row)
		}
	}
	if missing, _ := m.Forget("never-seen", 0); missing.Edges != 0 {
		t.Fatalf("an unknown reason changes nothing: %+v", missing)
	}
	if _, err := m.Forget("", 1); err == nil {
		t.Fatal("a factor of 1 forgets nothing and is refused")
	}
	// inverting swaps blame and clearing
	m.Clear([]string{negativeGood[0]}, 1, 1)
	blame, clear := m.G.Neg.TotalBlame, m.G.Neg.TotalClear
	m.Invert()
	if m.G.Neg.TotalBlame != clear || m.G.Neg.TotalClear != blame || !m.G.Inverted {
		t.Fatalf("invert swaps blame and clearing: %v/%v", m.G.Neg.TotalBlame, m.G.Neg.TotalClear)
	}
	m.Invert()
	if m.G.Neg.TotalBlame != blame || m.G.Inverted {
		t.Fatal("two inversions are the identity")
	}
}

func TestEdgeReasonsAreCapped(t *testing.T) {
	m, _ := NewNegativeModel(7, DefaultNegativeOptions())
	for i := 0; i < MaxEdgeReasons+3; i++ {
		if _, err := m.Blame([]string{negativeFailures[0]}, BlameOptions{Reason: "reason-" + string(rune('a'+i))}); err != nil {
			t.Fatal(err)
		}
	}
	for e, alive := range m.G.EdgeAlive {
		if alive && len(m.G.Neg.Reasons[e]) > MaxEdgeReasons {
			t.Fatalf("an edge keeps at most %d reasons: %d", MaxEdgeReasons, len(m.G.Neg.Reasons[e]))
		}
	}
	if len(m.Reasons()) != MaxEdgeReasons+3 {
		t.Fatalf("the totals keep every reason: %d", len(m.Reasons()))
	}
}

func TestJournalIsBounded(t *testing.T) {
	m, _ := NewNegativeModel(8, DefaultNegativeOptions())
	for i := 0; i < MaxLogEntries+5; i++ {
		m.Blame([]string{"failure number " + string(rune('a'+i%26)) + " here"}, BlameOptions{Reason: "other"})
	}
	if len(m.Neg.Log) != MaxLogEntries {
		t.Fatalf("the journal keeps %d entries: %d", MaxLogEntries, len(m.Neg.Log))
	}
	if len(m.Recent(0)) != 0 {
		t.Fatal("a limit of zero returns nothing")
	}
}

func TestNegativePersistence(t *testing.T) {
	m := taught(t)
	m.Clear([]string{negativeGood[0]}, 1, 1)
	m.Neg.Threshold, m.Neg.MinCoverage = 1.5, 0.75
	dir := t.TempDir()
	path := filepath.Join(dir, "model.negative.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !back.IsNegative() || back.Kind() != "negative" {
		t.Fatalf("the format decides the kind: %q", back.Kind())
	}
	if back.Neg.Threshold != 1.5 || back.Neg.MinCoverage != 0.75 {
		t.Fatalf("the filter settings survive: %+v", back.Neg)
	}
	if len(back.Recent(5)) != len(m.Recent(5)) {
		t.Fatal("the journal survives")
	}
	if math.Abs(back.G.Neg.TotalBlame-m.G.Neg.TotalBlame) > 1e-12 || math.Abs(back.G.Neg.TotalClear-m.G.Neg.TotalClear) > 1e-12 {
		t.Fatal("the totals survive")
	}
	before := m.Judge(negativeFailures[0], DefaultJudgeOptions())
	after := back.Judge(negativeFailures[0], DefaultJudgeOptions())
	if before.Risk != after.Risk || before.Why != after.Why {
		t.Fatalf("the same verdict after a round trip: %+v vs %+v", before, after)
	}
	if want, got := len(m.Reasons()), len(back.Reasons()); want != got {
		t.Fatalf("the reason registry survives: %d vs %d", want, got)
	}
	// a count model is still a count model
	count, _ := NewModel(0, DefaultGraphOptions())
	countPath := filepath.Join(dir, "model.count.json")
	if err := count.Save(countPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(countPath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.IsNegative() || loaded.Kind() != "count" {
		t.Fatalf("the count model is untouched: %q", loaded.Kind())
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"format":"radixnet-negative"`, `"function":"blame"`, `"kind":"negative"`, `"blame":[`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("the file speaks the Python format (%s missing)", want)
		}
	}
}

func TestNegativeGuards(t *testing.T) {
	count, _ := NewModel(0, DefaultGraphOptions())
	if count.IsNegative() {
		t.Fatal("a count model is not a negative network")
	}
	if _, err := count.Blame([]string{"anything at all"}, BlameOptions{}); err == nil {
		t.Fatal("blaming needs the negative network")
	}
	if _, err := count.BlameCorrection("a", "b", BlameOptions{}); err == nil {
		t.Fatal("so does a correction")
	}
	if count.Judge("anything", DefaultJudgeOptions()) != nil {
		t.Fatal("and a verdict")
	}
	if _, err := count.Forget("x", 0); err == nil {
		t.Fatal("and forgetting")
	}
}

func TestTrainingANegativeModelIsBlaming(t *testing.T) {
	m, _ := NewNegativeModel(11, DefaultNegativeOptions())
	records, err := m.Train([]string{"the the the the cat"}, TrainOptions{Epochs: 1, AutoCompress: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0]["phase"] != "negative" {
		t.Fatalf("training the negative network blames: %+v", records)
	}
	if m.G.Neg.TotalBlame == 0 || m.G.TotalTraversals != 0 {
		t.Fatalf("blame, not traversals: blame=%v traversals=%d", m.G.Neg.TotalBlame, m.G.TotalTraversals)
	}
	if _, err := m.TrainSource(SliceSource([]string{"the cat cat cat sat"}), TrainOptions{Epochs: 1}); err != nil {
		t.Fatal(err)
	}
	if m.Judge("the cat cat cat sat", DefaultJudgeOptions()).Verdict != "reject" {
		t.Fatal("a source is blamed too")
	}
	// thumbs down blames, thumbs up clears
	if _, err := m.Punish([]string{"a brand new failure here"}, 1, 1); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, row := range m.Reasons() {
		if row.Reason == "thumbs-down" {
			found = true
		}
	}
	if !found {
		t.Fatalf("punishing blames under its own reason: %+v", m.Reasons())
	}
	before := m.G.Neg.TotalBlame
	if _, err := m.Reward([]string{"the the the the cat"}, 1, 1); err != nil {
		t.Fatal(err)
	}
	if m.G.Neg.TotalBlame != before || m.G.Neg.TotalClear == 0 {
		t.Fatalf("rewarding clears, it never blames: %v / %v", m.G.Neg.TotalBlame, m.G.Neg.TotalClear)
	}
}
