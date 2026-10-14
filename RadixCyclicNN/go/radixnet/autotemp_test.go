package radixnet

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// Auto-temperature (autotemp.go, ../../SPEC-SearchAndTraining.md §10): the
// ladder's rungs and their draws, what counts as an answer, the record a climb
// leaves, and the two searches the model has - a greedy walk that goes round
// and is mended by a warmer one, a deterministic search that falls back to
// walks, a generation whose silent samples are drawn again - and, off, a search
// that is exactly the one it always was, draw for draw.  The cases of
// tests/test_autotemp.py.

// loopCorpus is a strong cycle - "ab " comes round again far more often than it
// ends - so a greedy walk goes round for ever.
var loopCorpus = []string{loopLine, loopLine, loopLine, "ab end"}

var loopLine = strings.Repeat("ab ", 20) + "end"

// endsCorpus is a text that ends as often as it goes on, so a walk from its end
// says nothing half the time.
var endsCorpus = []string{"the cat sat on the mat", "the cat sat on the mat and slept"}

func trainedOn(t *testing.T, texts []string, epochs int) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	opts := DefaultTrainOptions()
	opts.Epochs = epochs
	if _, err := m.Train(texts, opts); err != nil {
		t.Fatal(err)
	}
	return m
}

func at(t float64) *float64 { return &t }

func TestTheLadderDoublesUpToTheCeiling(t *testing.T) {
	if AutoFloor != 0.5 || AutoDraws != 3 {
		t.Fatalf("the floor and the draws: %v, %v", AutoFloor, AutoDraws)
	}
	for _, c := range []struct {
		start, ceiling float64
		want           []float64
	}{
		{1, 2, []float64{1, 2}},
		{1, 3, []float64{1, 2, 3}},
		{1, 1.5, []float64{1, 1.5}},
		{0.7, 2, []float64{0.7, 1.4, 2}},
		{0.25, 1, []float64{0.25, 0.5, 1}},
		{0, 2, []float64{0, 0.5, 1, 2}}, // greedy climbs to the floor first
		{0, 0.3, []float64{0, 0.3}},
		{3, 2, []float64{3}}, // it never cools
		{2, 2, []float64{2}},
	} {
		if got := TemperatureLadder(c.start, c.ceiling); !reflect.DeepEqual(got, c.want) {
			t.Errorf("TemperatureLadder(%v, %v) = %v, want %v", c.start, c.ceiling, got, c.want)
		}
	}
	if got := TemperatureLadder(1, 0); len(got) != 0 {
		t.Errorf("off is no ladder: %v", got)
	}
	for _, ok := range []float64{0, 2.5} {
		if err := CheckAutoTemperature(ok); err != nil {
			t.Errorf("%v is a ceiling: %v", ok, err)
		}
	}
	for _, bad := range []float64{-1, math.NaN(), math.Inf(1)} {
		if CheckAutoTemperature(bad) == nil {
			t.Errorf("%v was accepted", bad)
		}
	}
}

func TestAnAnswerWroteSomethingAndReachedTheEndWhenAsked(t *testing.T) {
	if !Answered(&PathResult{Text: " sat"}, false) || Answered(&PathResult{Text: "", ReachedEnd: true}, false) ||
		Answered(&PathResult{Text: "  \n", ReachedEnd: true}, false) {
		t.Fatal("an answer is something written")
	}
	if !Answered(&PathResult{Text: " sat", ReachedEnd: true}, true) || Answered(&PathResult{Text: " sat"}, true) {
		t.Fatal("and, when the end was asked for, it reached it")
	}
}

// searches is a search that answers only at the temperatures in answersAt (as
// sampled walks), and its log: "as asked" or the rung of every walk.
func searches(answersAt map[float64]bool) (func(*float64) (*Prediction, error), *[]string) {
	log := []string{}
	return func(rung *float64) (*Prediction, error) {
		if rung == nil {
			log = append(log, "as asked")
		} else {
			log = append(log, "sample "+formatRung(*rung))
		}
		text := ""
		if rung != nil && answersAt[*rung] {
			text = "said"
		}
		return &Prediction{PathResult: PathResult{Text: text, ReachedEnd: true}}, nil
	}, &log
}

func formatRung(t float64) string {
	switch t {
	case 0:
		return "0"
	case 0.5:
		return "0.5"
	case 1:
		return "1"
	case 2:
		return "2"
	}
	return "?"
}

func pickAnswer(p *Prediction) (*Prediction, bool) {
	answer := FirstAnswer(p, false)
	return answer, answer != nil
}

func TestTheClimb(t *testing.T) {
	// an answer as asked is the whole of it
	calls := 0
	result, record, err := Climb(func(rung *float64) (*Prediction, error) {
		calls++
		return &Prediction{PathResult: PathResult{Text: "said", ReachedEnd: true}}, nil
	}, pickAnswer, 1, 2, true)
	if err != nil || result.Text != "said" || calls != 1 {
		t.Fatalf("as asked: %v %+v %d", err, result, calls)
	}
	if want := (AutoTemperature{Ceiling: 2, Found: true, Temperature: at(1), Tries: 1}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("the record of an answer as asked: %+v", record)
	}
	// every rung is three draws, and the first answer ends it
	search, log := searches(map[float64]bool{2: true})
	result, record, _ = Climb(search, pickAnswer, 1, 2, false)
	if want := []string{"as asked", "sample 1", "sample 1", "sample 1", "sample 2"}; !reflect.DeepEqual(*log, want) {
		t.Fatalf("the walks: %v", *log)
	}
	if result.Text != "said" || !record.Climbed || !record.Found || *record.Temperature != 2 || record.Tries != 5 {
		t.Fatalf("found at the ceiling: %+v %+v", result, record)
	}
	// a greedy rung is one walk
	search, log = searches(nil)
	_, record, _ = Climb(search, pickAnswer, 0, 1, true)
	if want := []string{"as asked", "sample 0", "sample 0.5", "sample 0.5", "sample 0.5", "sample 1", "sample 1",
		"sample 1"}; !reflect.DeepEqual(*log, want) {
		t.Fatalf("the greedy walks: %v", *log)
	}
	if record.Found || record.Temperature != nil || record.Tries != 8 {
		t.Fatalf("nothing found: %+v", record)
	}
	// nothing found gives back the search as asked
	first := &Prediction{PathResult: PathResult{Labels: []string{"as asked"}}}
	calls = 0
	result, record, _ = Climb(func(rung *float64) (*Prediction, error) {
		calls++
		if calls == 1 {
			return first, nil
		}
		return &Prediction{}, nil
	}, pickAnswer, 1, 2, false)
	if result != first {
		t.Fatalf("the search as asked comes back: %+v", result)
	}
	if want := (AutoTemperature{Ceiling: 2, Climbed: true, Tries: 7}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("the record of nothing found: %+v", record)
	}
	// a caller's own test - a guard's veto - decides as well as the model's
	texts := []string{"vetoed", "vetoed", "clean"}
	calls = 0
	result, record, _ = Climb(func(rung *float64) (*Prediction, error) {
		calls++
		return &Prediction{PathResult: PathResult{Text: texts[calls-1]}}, nil
	}, func(p *Prediction) (*Prediction, bool) {
		return p, Answered(&p.PathResult, false) && p.Text != "vetoed"
	}, 1, 2, true)
	if result.Text != "clean" || record.Tries != 3 || *record.Temperature != 1 {
		t.Fatalf("the caller's veto: %+v %+v", result, record)
	}
}

func TestTheFirstAnsweringContinuationIsPromoted(t *testing.T) {
	cut := &PathResult{Text: " ab ab", Labels: []string{"a"}, NodeIDs: []int{3}, Cost: 1, Expanded: 7}
	ends := &PathResult{Text: " end", Labels: []string{"b"}, NodeIDs: []int{4, End}, Cost: 2, StepCosts: []float64{2},
		ReachedEnd: true, FullText: "ab end", Punish: 0.5, Expanded: 9}
	p := &Prediction{PathResult: PathResult{Text: " ab ab", Expanded: 40}, Top: []*PathResult{cut, ends},
		Bottom: []*PathResult{cut}, K: 2, Beam: 16, Mode: "beam", Traversal: "reward"}
	got := FirstAnswer(p, true)
	if got == nil || got == p {
		t.Fatalf("the continuation that ends is the answer: %+v", got)
	}
	if got.Text != " end" || !got.ReachedEnd || got.Cost != 2 || got.FullText != "ab end" || got.Punish != 0.5 ||
		!reflect.DeepEqual(got.NodeIDs, ends.NodeIDs) || !reflect.DeepEqual(got.StepCosts, ends.StepCosts) {
		t.Fatalf("its path and cost replace the prediction's: %+v", got.PathResult)
	}
	if got.Expanded != 40 || len(got.Top) != 2 || len(got.Bottom) != 1 || got.K != 2 || got.Beam != 16 ||
		got.Mode != "beam" || got.Traversal != "reward" {
		t.Fatalf("the search's own fields stay: %+v", got)
	}
	if p.Text != " ab ab" || p.Expanded != 40 {
		t.Fatalf("the prediction asked is not touched: %+v", p.PathResult)
	}
	if FirstAnswer(p, false) != p {
		t.Fatal("a prediction that answers is its own answer")
	}
	if FirstAnswer(&Prediction{Top: []*PathResult{cut}}, true) != nil {
		t.Fatal("nothing that ends is no answer")
	}
}

func TestOffIsTheSearchItAlwaysWas(t *testing.T) {
	m := trainedOn(t, loopCorpus, 2)
	o := DefaultPredictOptions()
	o.Mode, o.Length = "sample", 12
	before := m.G.rng.State()
	plain, err := m.Predict("ab", o)
	if err != nil {
		t.Fatal(err)
	}
	after := m.G.rng.State()
	if err := m.G.rng.SetState(before); err != nil {
		t.Fatal(err)
	}
	result, record, err := AutoPredict(m, "ab", o, 0)
	if err != nil || record != nil {
		t.Fatalf("off: %v %+v", err, record)
	}
	if result.Text != plain.Text || !reflect.DeepEqual(result.NodeIDs, plain.NodeIDs) {
		t.Fatalf("the same walk: %q / %q", result.Text, plain.Text)
	}
	if !reflect.DeepEqual(m.G.rng.State(), after) {
		t.Fatal("the same draws, no more")
	}
	if _, _, err := AutoPredict(m, "ab", o, -1); err == nil {
		t.Fatal("a negative ceiling is refused")
	}
}

func TestAGreedyWalkThatGoesRoundIsMendedByAWarmerOne(t *testing.T) {
	m := trainedOn(t, loopCorpus, 2)
	o := DefaultPredictOptions()
	o.Mode, o.Temperature, o.Length, o.MaxLength, o.ToEnd = "sample", 0, 20, 30, true
	cold, err := m.Predict("ab", o)
	if err != nil {
		t.Fatal(err)
	}
	if cold.ReachedEnd {
		t.Fatalf("greedy goes round and never ends: %q", cold.Text)
	}
	result, record, err := AutoPredict(m, "ab", o, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.ReachedEnd || strings.TrimSpace(result.Text) == "" {
		t.Fatalf("a warmer walk ends: %+v", result.PathResult)
	}
	if !record.Climbed || !record.Found || record.Temperature == nil || *record.Temperature <= 0 || record.Tries <= 1 {
		t.Fatalf("the record: %+v", record)
	}
}

func TestADeterministicSearchThatFindsNothingFallsBackToWalks(t *testing.T) {
	// the beam reads no temperature: when it finds nothing, the ladder is where sampling starts - and the walks
	// are held to the same cap, so an end no path reaches inside it is not reached by walking either
	m := trainedOn(t, loopCorpus, 2)
	o := DefaultPredictOptions()
	o.Mode, o.Length, o.MaxLength, o.ToEnd = "beam", 2, 3, true // " end" needs 4
	beam, err := m.Predict("ab", o)
	if err != nil {
		t.Fatal(err)
	}
	if FirstAnswer(beam, true) != nil {
		t.Fatalf("no path ends inside the cap: %+v", beam.PathResult)
	}
	result, record, err := AutoPredict(m, "ab", o, 2)
	if err != nil {
		t.Fatal(err)
	}
	if want := (AutoTemperature{Ceiling: 2, Climbed: true, Tries: 1 + 2*AutoDraws}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("the record: %+v", record)
	}
	if result.Text != beam.Text || !reflect.DeepEqual(result.NodeIDs, beam.NodeIDs) {
		t.Fatalf("the search as asked comes back: %+v", result.PathResult)
	}
}

func TestAnAnswerLowerInTheBeamIsTheBeamsOwn(t *testing.T) {
	// with room to end, the beam's cheapest path still goes round to the cap - but its top K hold paths that
	// end, and the first of them is the answer: found as asked, nothing walked
	m := trainedOn(t, loopCorpus, 2)
	o := DefaultPredictOptions()
	o.Mode, o.Length, o.MaxLength, o.ToEnd = "beam", 2, 30, true
	beam, err := m.Predict("ab", o)
	if err != nil {
		t.Fatal(err)
	}
	if beam.ReachedEnd {
		t.Fatal("the cheapest path is the cycle, cut at the cap")
	}
	var ending *PathResult
	for _, p := range beam.Top {
		if p.ReachedEnd {
			ending = p
			break
		}
	}
	if ending == nil {
		t.Fatal("a path in the top K ends")
	}
	result, record, err := AutoPredict(m, "ab", o, 2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != ending.Text || !reflect.DeepEqual(result.NodeIDs, ending.NodeIDs) || !result.ReachedEnd {
		t.Fatalf("the first that ends: %+v", result.PathResult)
	}
	if len(result.Top) != len(beam.Top) || result.Top[0].Text != beam.Top[0].Text {
		t.Fatal("the ranking is the search's")
	}
	if want := (AutoTemperature{Ceiling: 2, Found: true, Tries: 1}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("found as asked, by a search that reads no temperature: %+v", record)
	}
}

func TestNoWayOnIsNoAnswerAtAnyTemperature(t *testing.T) {
	m := trainedOn(t, []string{"the cat sat"}, 1)
	o := DefaultPredictOptions()
	o.Mode, o.Length = "sample", 5
	result, record, err := AutoPredict(m, "the cat sat", o, 2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "" {
		t.Fatalf("nothing follows: %q", result.Text)
	}
	if want := (AutoTemperature{Ceiling: 2, Climbed: true, Tries: 1 + 2*AutoDraws}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("the record: %+v", record)
	}
}

func texts(results []*PathResult) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i] = r.Text
	}
	return out
}

func TestSamplesThatSaidNothingAreDrawnAgain(t *testing.T) {
	m := trainedOn(t, endsCorpus, 2)
	prefix := "the cat sat on the mat"
	seed := int64(4)
	o := DefaultGenerateOptions()
	o.Count, o.Mode, o.Prefix, o.Seed = 6, "sample", prefix, &seed
	plain, err := m.Generate(o)
	if err != nil {
		t.Fatal(err)
	}
	// off: the generation it always was
	off, record, err := AutoGenerate(m, o, 0)
	if err != nil || record != nil || !reflect.DeepEqual(texts(off), texts(plain)) {
		t.Fatalf("off: %v %+v %v", err, record, texts(off))
	}
	said := []string{}
	for _, r := range plain {
		if Wrote(m, r, prefix) {
			said = append(said, r.Text)
		}
	}
	if len(said) == len(plain) {
		t.Fatal("some walks end where the prefix does")
	}
	results, record, err := AutoGenerate(m, o, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 6 || !record.Climbed || !record.Found {
		t.Fatalf("six that said something: %v %+v", texts(results), record)
	}
	for _, r := range results {
		if !strings.HasPrefix(r.Text, prefix) || len(r.Text) <= len(prefix) {
			t.Fatalf("every one says something beyond the prefix: %q", r.Text)
		}
	}
	// the samples that said something come first, in their order
	if !reflect.DeepEqual(texts(results[:len(said)]), said) {
		t.Fatalf("the first generation's first: %v / %v", texts(results), said)
	}
	// and the whole climb is reproducible from the seed
	again, _, err := AutoGenerate(m, o, 2)
	if err != nil || !reflect.DeepEqual(texts(again), texts(results)) {
		t.Fatalf("reproducible: %v / %v", texts(again), texts(results))
	}
}

func TestTooFewAtTheCeilingIsWhatThereIs(t *testing.T) {
	m := trainedOn(t, []string{"the cat sat"}, 1)
	o := DefaultGenerateOptions()
	o.Count, o.Mode, o.Prefix = 3, "sample", "the cat sat"
	results, record, err := AutoGenerate(m, o, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 || results == nil || record.Found || record.Ceiling != 1 || record.Temperature != nil {
		t.Fatalf("nothing to say: %v %+v", texts(results), record)
	}
	if record.Tries != 1+AutoDraws { // the ladder (1, 1): one rung
		t.Fatalf("tries: %d", record.Tries)
	}
}
