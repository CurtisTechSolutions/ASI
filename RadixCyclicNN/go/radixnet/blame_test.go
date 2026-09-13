package radixnet

import (
	"math"
	"testing"
)

func gradedLesson(sentence, answer, errorType, correction, comment string, score *float64, passed bool) *Lesson {
	return &Lesson{
		Exercise: Exercise{ID: "r1e1", Prefix: "the cat", Answer: answer},
		Sentence: sentence,
		Grade: Grade{Score: score, Passed: passed, Error: errorType, Correction: correction, Comment: comment,
			GradedBy: "ollama"},
	}
}

func tutorLessons() []*Lesson {
	return []*Lesson{
		gradedLesson("the cat sit on the mat", "the cat sleeps on the mat", "agreement", "the cat sits on the mat",
			"The verb must agree with the subject.", floatPtr(4), false),
		gradedLesson("the dogs run in the park", "", "none", "", "Good.", floatPtr(9), true),
		gradedLesson("she go to the shop yesterday", "she went to the shop", "tense", "",
			"The past tense of 'go' is 'went'.", floatPtr(2), false),
	}
}

func TestClassify(t *testing.T) {
	cases := map[string]string{
		"The text repeats the same word over and over.": "repetition",
		"Pure gibberish with garbled characters":        "gibberish",
		"It is cut off mid-sentence":                    "truncated",
		"Broken grammar and wrong word order":           "grammar",
		"Several misspellings":                          "spelling",
		"It contradicts itself":                         "contradiction",
		"The claim is factually wrong":                  "false",
		"Completely incoherent":                         "incoherent",
		"Irrelevant to the prompt":                      "off-topic",
		"empty output":                                  "empty",
		"Something nobody has a word for":               DefaultReason,
	}
	for critique, want := range cases {
		if got := Classify(critique, ClassifyOptions{}); got != want {
			t.Fatalf("%q -> %q, want %q", critique, got, want)
		}
	}
	if got := Classify("", ClassifyOptions{Verdict: "unrated"}); got != "unrated" {
		t.Fatalf("an unrated failure says so: %q", got)
	}
	zero := 0.0
	if got := Classify("", ClassifyOptions{Rating: &zero}); got != "gibberish" {
		t.Fatalf("a zero mark with no words: %q", got)
	}
	if got := Classify("nothing matching", ClassifyOptions{Default: "mine"}); got != "mine" {
		t.Fatalf("the default wins: %q", got)
	}
}

func TestSeverityFromRating(t *testing.T) {
	if got := SeverityFromRating(floatPtr(0), 6); math.Abs(got-2) > 1e-9 {
		t.Fatalf("a hopeless answer weighs 2: %v", got)
	}
	if got := SeverityFromRating(floatPtr(6), 6); math.Abs(got-0.25) > 1e-9 {
		t.Fatalf("a bare pass weighs the floor: %v", got)
	}
	if got := SeverityFromRating(floatPtr(3), 6); math.Abs(got-(0.25+2.0)/2) > 1e-9 {
		t.Fatalf("halfway: %v", got)
	}
	if got := SeverityFromRating(nil, 6); got != 1 {
		t.Fatalf("an unrated failure weighs one ordinary failure: %v", got)
	}
	if got := SeverityFromRating(floatPtr(99), 6); math.Abs(got-0.25) > 1e-9 {
		t.Fatalf("clamped: %v", got)
	}
	if got := SeverityFromRating(floatPtr(1), 0); math.Abs(got-0.25) > 1e-9 {
		t.Fatalf("a zero threshold never divides by zero: %v", got)
	}
}

func TestFaultsFromLessons(t *testing.T) {
	faults, passed := FaultsFromLessons(tutorLessons(), 6, "tutor")
	if len(faults) != 2 || faults[0].Reason != "agreement" || faults[1].Reason != "tense" {
		t.Fatalf("the teacher's mistakes are the reasons: %+v", faults)
	}
	if faults[0].Correction != "the cat sits on the mat" {
		t.Fatalf("the correction rides along: %+v", faults[0])
	}
	if faults[1].Correction != "" {
		t.Fatalf("a lesson without a correction carries none: %+v", faults[1])
	}
	if !(faults[1].Severity > faults[0].Severity) {
		t.Fatalf("2/10 is worse than 4/10: %v vs %v", faults[1].Severity, faults[0].Severity)
	}
	if faults[0].Source != "tutor" || faults[0].Note == "" {
		t.Fatalf("the tutor's words are kept: %+v", faults[0])
	}
	want := map[string]bool{"the dogs run in the park": true, "the cat sits on the mat": true,
		"the cat sleeps on the mat": true, "she went to the shop": true}
	for _, text := range passed {
		if !want[text] {
			t.Fatalf("unexpected cleared text %q", text)
		}
		delete(want, text)
	}
	if len(want) != 0 {
		t.Fatalf("the passes, corrections and model answers all clear blame; missing %+v", want)
	}
	// an unrecognised mistake falls back to the critique
	fallback, _ := FaultsFromLessons([]*Lesson{gradedLesson("the the the cat", "", "none", "",
		"It repeats the same word over and over.", floatPtr(1), false)}, 6, "tutor")
	if len(fallback) != 1 || fallback[0].Reason != "repetition" {
		t.Fatalf("the critique decides when the teacher named no mistake: %+v", fallback)
	}
}

func TestTeachLessons(t *testing.T) {
	negative, err := NewNegativeModel(7, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	report, err := TeachLessons(negative, tutorLessons(), 6, true, "tutor", TeachOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Blamed != 2 || report.Reasons["agreement"] != 1 || report.Reasons["tense"] != 1 {
		t.Fatalf("both failures are blamed: %+v", report)
	}
	if report.Passed == 0 {
		t.Fatalf("the passes clear blame: %+v", report)
	}
	verdict := negative.Judge("the cat sit on the mat", DefaultJudgeOptions())
	if len(verdict.Reasons) == 0 || verdict.Reasons[0].Reason != "agreement" {
		t.Fatalf("the corrected sentence is blamed for what the teacher named: %+v", verdict.Reasons)
	}
	if len(verdict.Spans) != 1 || verdict.Spans[0].Fragment != "sit " {
		t.Fatalf("and only where it changed: %+v", verdict.Spans)
	}
	if negative.Judge("the cat sits on the mat", DefaultJudgeOptions()).Verdict != "pass" {
		t.Fatal("the correction itself passes")
	}
	sources, _ := negative.Stats()["sources"].(map[string]any)
	if toFloat(sources["tutor"]) != 2 {
		t.Fatalf("the tutor is recorded as the source: %+v", sources)
	}
	// a stopped run teaches nothing
	stopped, err := TeachLessons(negative, tutorLessons(), 6, true, "tutor",
		TeachOptions{Stop: func() bool { return true }})
	if err != nil || stopped.Blamed != 0 {
		t.Fatalf("the stop event is honoured: %+v %v", stopped, err)
	}
	// a count model cannot be taught
	count, _ := NewModel(1, DefaultGraphOptions())
	if _, err := TeachLessons(count, tutorLessons(), 6, true, "tutor", TeachOptions{}); err == nil {
		t.Fatal("teaching needs the negative network")
	}
}

func TestTutorTrainerTeachesTheNegativeNetwork(t *testing.T) {
	negative, _ := NewNegativeModel(8, DefaultNegativeOptions())
	model, _ := NewModel(1, DefaultGraphOptions())
	client, err := NewOllamaClient("http://127.0.0.1:1", "fake", 0)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTutorTrainer(model, client, DefaultTutorConfig())
	if err != nil {
		t.Fatal(err)
	}
	if report, err := trainer.TeachNegative(tutorLessons()); err != nil || report != nil {
		t.Fatalf("without a negative network there is nothing to teach: %+v %v", report, err)
	}
	trainer.Negative = negative
	report, err := trainer.TeachNegative(tutorLessons())
	if err != nil {
		t.Fatal(err)
	}
	if report.Blamed != 2 || report.Reasons["agreement"] != 1 {
		t.Fatalf("the round's failures reach the negative network: %+v", report)
	}
	if negative.Judge("the cat sit on the mat", DefaultJudgeOptions()).Reasons[0].Reason != "agreement" {
		t.Fatal("with the mistake the teacher named")
	}
}
