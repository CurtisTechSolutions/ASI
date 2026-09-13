package radixnet

import (
	"math"
	"testing"
)

var duoCorpus = []string{
	"the cat sat on the mat",
	"the dog sat on the log",
	"the cat ran to the door",
	"the dog ate the bone",
}

const duoFailure = "the the the the the"

func duoPair(t *testing.T, config FilterConfig) *Filter {
	t.Helper()
	positive, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	positive.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if _, err := positive.Train(duoCorpus, TrainOptions{Epochs: 3, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	negative, err := NewNegativeModel(2, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := negative.Blame([]string{duoFailure}, BlameOptions{Reason: "repetition", Source: "review",
		Note: "it repeats the same word"}); err != nil {
		t.Fatal(err)
	}
	pair, err := NewFilter(positive, negative, config)
	if err != nil {
		t.Fatal(err)
	}
	return pair
}

func TestFilterConfigValidation(t *testing.T) {
	if err := DefaultFilterConfig().Validate(); err != nil {
		t.Fatal(err)
	}
	bad := DefaultFilterConfig()
	over := 1.5
	bad.MinCoverage = &over
	if err := bad.Validate(); err == nil {
		t.Fatal("a coverage above 1 is refused")
	}
	nan := math.NaN()
	bad = DefaultFilterConfig()
	bad.Peak = &nan
	if err := bad.Validate(); err == nil {
		t.Fatal("NaN is refused")
	}
	negative, _ := NewNegativeModel(1, DefaultNegativeOptions())
	if _, err := NewFilter(negative, negative, DefaultFilterConfig()); err == nil {
		t.Fatal("the two halves must differ")
	}
	count, _ := NewModel(1, DefaultGraphOptions())
	if _, err := NewFilter(count, count, DefaultFilterConfig()); err == nil {
		t.Fatal("the negative half must be a negative network")
	}
}

func TestFilterJudge(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	verdict := pair.Judge(duoFailure)
	if verdict.Decision != "reject" || ruleName(verdict.Rule) != "blame" {
		t.Fatalf("a known failure is rejected by blame: %+v", verdict)
	}
	if len(verdict.Reasons) == 0 || verdict.Reasons[0].Reason != "repetition" {
		t.Fatalf("with its reason: %+v", verdict.Reasons)
	}
	if verdict.Ratio <= 0 {
		t.Fatalf("it reads more like failure than like the corpus: %v", verdict.Ratio)
	}
	corpus := pair.Judge(duoCorpus[0])
	if corpus.Decision == "reject" || corpus.Ratio >= 0 {
		t.Fatalf("corpus text passes: %+v", corpus)
	}
	unseen := pair.Judge("a wholly unrelated sentence")
	if unseen.Decision != "pass" || ruleName(unseen.Rule) != "" || unseen.Blame != 0 {
		t.Fatalf("unseen text passes: %+v", unseen)
	}
}

func TestFilterRules(t *testing.T) {
	high, zero := 1e9, 0.0
	// the ratio rule can reject on its own
	config := DefaultFilterConfig()
	config.Threshold, config.MinCoverage = &high, &zero
	ratio := 0.0
	config.Ratio = &ratio
	verdict := duoPair(t, config).Judge(duoFailure)
	if verdict.Decision != "reject" || ruleName(verdict.Rule) != "ratio" {
		t.Fatalf("the ratio rule rejects on its own: %+v", verdict)
	}
	// ... and can be turned off
	config.Ratio = nil
	if off := duoPair(t, config).Judge(duoFailure); off.Decision != "suspect" {
		t.Fatalf("with both rules off nothing rejects: %+v", off)
	}
	// the coverage gate protects unknown text
	gate := DefaultFilterConfig()
	one := 1.0
	gate.Threshold, gate.MinCoverage = &zero, &one
	if unknown := duoPair(t, gate).Judge("nothing here ever failed"); unknown.Decision != "pass" {
		t.Fatalf("the gate protects unknown text: %+v", unknown)
	}
	// strict drops the suspects
	strict := DefaultFilterConfig()
	strict.Threshold, strict.Ratio, strict.Strict = &high, nil, true
	if suspect := duoPair(t, strict).Judge(duoCorpus[0]); suspect.Coverage > 0 && ruleName(suspect.Rule) != "suspect" {
		t.Fatalf("strict drops a suspect: %+v", suspect)
	}
}

func TestPeakRuleCatchesOneCorrectedFragment(t *testing.T) {
	config := DefaultFilterConfig()
	config.Ratio = nil
	pair := duoPair(t, config)
	// a negative network that knows nothing but this one correction: the wrong
	// word is the only blamed fragment in the sentence
	fresh, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	pair.Negative = fresh
	if _, err := pair.Negative.BlameCorrection("the cat sit on the mat", "the cat sits on the mat",
		BlameOptions{Reason: "agreement", Severity: 1.5}); err != nil {
		t.Fatal(err)
	}
	lenient := pair.Judge("the cat sit on the mat")
	if lenient.Decision != "suspect" {
		t.Fatalf("one wrong fragment in a long sentence is only suspect: %+v", lenient)
	}
	if math.Abs(lenient.Peak-1.5) > 1e-9 {
		t.Fatalf("the peak is the evidence on one transition: %v", lenient.Peak)
	}
	peak := 1.0
	pair.Config.Peak = &peak
	rejected := pair.Judge("the cat sit on the mat")
	if rejected.Decision != "reject" || ruleName(rejected.Rule) != "peak" {
		t.Fatalf("the peak rule vetoes it: %+v", rejected)
	}
	if pair.Judge("nothing like it at all").Decision != "pass" {
		t.Fatal("and leaves unknown text alone")
	}
}

func TestFilterBatchAndLearning(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	outcome, err := pair.Filter([]string{duoFailure, duoCorpus[0], "a wholly unrelated sentence"})
	if err != nil {
		t.Fatal(err)
	}
	if len(outcome.Rejected) != 1 || outcome.Rejected[0].Text != duoFailure {
		t.Fatalf("one veto: %+v", outcome.Rejected)
	}
	if len(outcome.Kept) != 2 || outcome.Rate == nil || math.Abs(*outcome.Rate-2.0/3.0) > 1e-9 {
		t.Fatalf("two kept: %+v", outcome)
	}
	before := pair.Negative.G.Neg.TotalBlame
	if _, err := pair.Filter([]string{duoFailure}); err != nil {
		t.Fatal(err)
	}
	if pair.Negative.G.Neg.TotalBlame != before {
		t.Fatal("learning is off by default: the tutor supplies the negatives")
	}
	pair.Config.Learn, pair.Config.Reason = true, "filtered"
	if _, err := pair.Filter([]string{duoFailure}); err != nil {
		t.Fatal(err)
	}
	if pair.Negative.G.Neg.TotalBlame <= before {
		t.Fatal("with learn on, a reject is blamed")
	}
	found := false
	for _, row := range pair.Negative.Reasons() {
		if row.Reason == "filtered" {
			found = true
		}
	}
	if !found {
		t.Fatalf("under its own reason: %+v", pair.Negative.Reasons())
	}
	if entry := pair.Negative.Recent(1); len(entry) != 1 || entry[0].Source != "filter" {
		t.Fatalf("and its own source: %+v", entry)
	}
}

func TestFilterGenerateAndPredict(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	o := DefaultGenerateOptions()
	o.MaxLength, o.Mode = 30, "sample"
	seed := int64(5)
	o.Seed = &seed
	out, err := pair.Generate(2, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Texts) > 2 {
		t.Fatalf("at most the wanted count comes back: %+v", out.Texts)
	}
	if out.Asked != 6 {
		t.Fatalf("three candidates per wanted text: %d", out.Asked)
	}
	if len(out.Verdicts) != out.Candidates {
		t.Fatalf("one verdict per candidate: %d vs %d", len(out.Verdicts), out.Candidates)
	}
	risk := map[string]float64{}
	for _, v := range out.Verdicts {
		risk[v.Text] = v.Risk
	}
	for i := 1; i < len(out.Texts); i++ { // the cleanest come back first
		if risk[out.Texts[i-1]] > risk[out.Texts[i]] {
			t.Fatalf("the survivors are ranked: %+v", out.Texts)
		}
	}
	if empty, err := pair.Generate(0, o); err != nil || len(empty.Texts) != 0 {
		t.Fatalf("a count of zero writes nothing: %+v %v", empty, err)
	}
	if _, err := pair.Generate(-1, o); err == nil {
		t.Fatal("a negative count is refused")
	}
	prediction, err := pair.Predict("the ", PredictOptions{Length: 10, K: 3, Mode: "beam"})
	if err != nil {
		t.Fatal(err)
	}
	if prediction.Prefix != "the " || len(prediction.Verdicts) != prediction.Candidates {
		t.Fatalf("every candidate is judged: %+v", prediction)
	}
	for _, v := range prediction.Verdicts {
		if len(v.Text) < 4 || v.Text[:4] != "the " {
			t.Fatalf("the candidates continue the prefix: %q", v.Text)
		}
	}
	described := pair.Describe()
	if described["positive"].(map[string]any)["kind"] != "count" || described["negative"].(map[string]any)["kind"] != "negative" {
		t.Fatalf("describe names both halves: %+v", described)
	}
}

// ruleName is the rule a verdict fired, or "" when none did.
func ruleName(rule *string) string {
	if rule == nil {
		return ""
	}
	return *rule
}
