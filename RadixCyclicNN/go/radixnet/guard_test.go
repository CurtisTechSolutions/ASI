package radixnet

import "testing"

// The guard is the pair on the ordinary output path: Ready says whether it is
// worth putting in the way, Rank takes the vetoed continuations out of a
// finished prediction, and Converse leaves a vetoed reply unsaid.

func TestFilterReadyOnlyWithBlame(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	if !pair.Ready() {
		t.Fatal("a network with blame on it is ready to guard")
	}
	if _, err := pair.Negative.Forget("repetition", 0); err != nil {
		t.Fatal(err)
	}
	if pair.Ready() {
		t.Fatal("forgetting the blame leaves nothing to guard with")
	}
	empty, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	untaught, err := NewFilter(pair.Positive, empty, DefaultFilterConfig())
	if err != nil {
		t.Fatal(err)
	}
	if untaught.Ready() {
		t.Fatal("an untaught negative network vetoes nothing")
	}
}

func TestGenerateCarriesTheWalksBehindTheTexts(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	seed := int64(5)
	outcome, err := pair.Generate(2, GenerateOptions{MaxLength: 30, Mode: "sample", Temperature: 1, Seed: &seed})
	if err != nil {
		t.Fatal(err)
	}
	if len(outcome.Results) != len(outcome.Texts) {
		t.Fatalf("one walk per text: %d walks, %d texts", len(outcome.Results), len(outcome.Texts))
	}
	for i, result := range outcome.Results {
		if result.Text != outcome.Texts[i] {
			t.Fatalf("walk %d is not the text it belongs to: %q vs %q", i, result.Text, outcome.Texts[i])
		}
	}
	empty, err := pair.Generate(0, GenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(empty.Results) != 0 {
		t.Fatalf("nothing wanted, nothing walked: %+v", empty.Results)
	}
}

func TestRankDropsTheVetoedContinuations(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	found, err := pair.Positive.Predict("the ", PredictOptions{Length: 10, Mode: "beam", K: 4})
	if err != nil {
		t.Fatal(err)
	}
	ranked, verdicts := pair.Rank("the ", found)
	if len(verdicts) != len(found.Top) {
		t.Fatalf("one verdict per continuation: %d verdicts, %d offered", len(verdicts), len(found.Top))
	}
	kept := 0
	for _, verdict := range verdicts {
		if verdict.Decision != "reject" {
			kept++
		}
	}
	if len(ranked.Top) != kept {
		t.Fatalf("the survivors stay: %d of %d", len(ranked.Top), kept)
	}
	if len(ranked.Top) > 0 && ranked.Text != ranked.Top[0].Text {
		t.Fatalf("the best survivor is the prediction: %q vs %q", ranked.Text, ranked.Top[0].Text)
	}
	if ranked.Expanded != found.Expanded {
		t.Fatalf("the search that ran is still reported: %d vs %d", ranked.Expanded, found.Expanded)
	}
	if ranked.K != found.K || ranked.Beam != found.Beam || ranked.Mode != found.Mode {
		t.Fatalf("the search's own settings survive: %+v", ranked)
	}
}

func TestRankLeavesNothingWhenEverythingIsVetoed(t *testing.T) {
	config := DefaultFilterConfig()
	zero := 0.0
	config.Threshold, config.MinCoverage = &zero, &zero
	pair := duoPair(t, config)
	if _, err := pair.Negative.Blame(duoCorpus, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	found, err := pair.Positive.Predict("the ", PredictOptions{Length: 10, Mode: "beam", K: 4})
	if err != nil {
		t.Fatal(err)
	}
	ranked, verdicts := pair.Rank("the ", found)
	for _, verdict := range verdicts {
		if verdict.Decision != "reject" {
			t.Fatalf("everything the model can write is known-bad: %+v", verdict)
		}
	}
	if ranked.Text != "" || ranked.FullText != "the " || len(ranked.Top) != 0 {
		t.Fatalf("nothing survives, so the prediction is the prefix: %q %q %+v", ranked.Text, ranked.FullText, ranked.Top)
	}
}

func TestGuardedConverseLeavesTheVetoedRepliesUnsaid(t *testing.T) {
	pair := duoPair(t, DefaultFilterConfig())
	o := DefaultConverseOptions()
	o.Turns = 4
	outcome, err := pair.Converse(duoCorpus[0], o)
	if err != nil {
		t.Fatal(err)
	}
	if len(outcome.Turns) == 0 {
		t.Fatal("the conversation still happens")
	}
	if outcome.Turns[0].Text != duoCorpus[0] {
		t.Fatalf("the opening is spoken as given: %q", outcome.Turns[0].Text)
	}
	refused := map[string]bool{}
	for _, verdict := range outcome.Rejected {
		refused[verdict.Text] = true
	}
	for _, turn := range outcome.Turns[1:] {
		if refused[turn.Text] {
			t.Fatalf("a vetoed reply was spoken anyway: %q", turn.Text)
		}
	}
	vetoed := 0
	for _, turn := range outcome.Turns {
		vetoed += turn.Vetoed
	}
	if vetoed != outcome.Vetoed {
		t.Fatalf("the turns count their own vetoes: %d vs %d", vetoed, outcome.Vetoed)
	}
}

func TestGuardedConverseStopsWhenNothingMayBeSaid(t *testing.T) {
	config := DefaultFilterConfig()
	zero := 0.0
	config.Threshold, config.MinCoverage = &zero, &zero
	pair := duoPair(t, config)
	if _, err := pair.Negative.Blame(duoCorpus, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	o := DefaultConverseOptions()
	o.Turns = 3
	outcome, err := pair.Converse("", o)
	if err != nil {
		t.Fatal(err)
	}
	if len(outcome.Turns) != 0 {
		t.Fatalf("nothing may be spoken, so nothing is: %+v", outcome.Turns)
	}
	if len(outcome.Rejected) == 0 {
		t.Fatal("and the refusals are reported")
	}
}

func TestGuardedConverseCanBlameWhatItRefused(t *testing.T) {
	config := DefaultFilterConfig()
	zero := 0.0
	config.Threshold, config.MinCoverage, config.Learn, config.Reason = &zero, &zero, true, "vetoed"
	pair := duoPair(t, config)
	if _, err := pair.Negative.Blame(duoCorpus, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	before := pair.Negative.G.Neg.TotalBlame
	o := DefaultConverseOptions()
	o.Turns = 2
	if _, err := pair.Converse("", o); err != nil {
		t.Fatal(err)
	}
	if pair.Negative.G.Neg.TotalBlame <= before {
		t.Fatalf("learn blames what the veto stopped: %v -> %v", before, pair.Negative.G.Neg.TotalBlame)
	}
	found := false
	for _, row := range pair.Negative.Reasons() {
		if row.Reason == "vetoed" {
			found = true
		}
	}
	if !found {
		t.Fatalf("under its own reason: %+v", pair.Negative.Reasons())
	}
}
