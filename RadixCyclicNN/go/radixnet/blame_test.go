package radixnet

import (
	"math"
	"strings"
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

func TestAWidenedMistakeBlamesItsWholeFamily(t *testing.T) {
	lesson := tutorLessons()[0]
	lesson.Why = "A singular subject takes a singular verb; the student drops the -s."
	lesson.Variants = []TutorCorrection{
		{Wrong: "the dog sleep in the sun", Right: "the dog sleeps in the sun", Weight: 0.5},
		{Wrong: "she walk to the shop", Right: "she walks to the shop", Weight: 0.5},
		{Wrong: "the cat sit on the mat"}, // a repeat of the student's own sentence: dropped
		{Wrong: ""},
	}
	faults, passed := FaultsFromLessons([]*Lesson{lesson}, 6, "tutor")
	if len(faults) != 3 {
		t.Fatalf("the mistake and its two usable variants: %+v", faults)
	}
	for _, fault := range faults {
		if fault.Reason != "agreement" {
			t.Fatalf("the same mistake keeps the same reason: %+v", fault)
		}
	}
	if faults[1].Source != "tutor:similar" || faults[2].Source != "tutor:similar" {
		t.Fatalf("a variant says where it came from: %+v", faults)
	}
	if !closeTo(faults[1].Severity, faults[0].Severity*0.5) {
		t.Fatalf("the student never wrote it, so it weighs less: %g vs %g", faults[1].Severity, faults[0].Severity)
	}
	if faults[1].Correction != "the dog sleeps in the sun" || !strings.Contains(faults[1].Note, "drops the -s") {
		t.Fatalf("a variant is diffed against its own correct form, noted with the explanation: %+v", faults[1])
	}
	found := false
	for _, text := range passed {
		if text == "she walks to the shop" {
			found = true
		}
	}
	if !found {
		t.Fatalf("what the teacher wrote is clean text: %v", passed)
	}

	negative, err := NewNegativeModel(11, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	report, err := TeachLessons(negative, []*Lesson{lesson}, 6, true, "tutor", TeachOptions{})
	if err != nil || report.Blamed != 3 {
		t.Fatalf("the whole family is blamed: %+v %v", report, err)
	}
	// a sentence the network never wrote is now known to be wrong in the same way
	if verdict := negative.Judge("the dog sleep in the sun", DefaultJudgeOptions()); verdict.Verdict == "pass" {
		t.Fatalf("the family should be recognised: %+v", verdict)
	}
	if negative.Judge("the dog sleeps in the sun", DefaultJudgeOptions()).Verdict != "pass" {
		t.Fatal("its correct form is clean")
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

// -- the copy editor's reasons and faults ------------------------------------

func TestReasonFromChanges(t *testing.T) {
	cases := []struct {
		name string
		want string
		edit []Change
	}{
		{"nothing", "none", nil},
		{"equal run", "none", []Change{{Op: "equal", Wrong: "a", Right: "a"}}},
		{"punctuation", "punctuation", []Change{{Op: "delete", Wrong: "?", Right: ""}}},
		{"spacing", "spacing", []Change{{Op: "insert", Wrong: "", Right: " "}}},
		{"case", "capitalisation", []Change{{Op: "replace", Wrong: "the", Right: "The"}}},
		{"comma carried", "punctuation", []Change{{Op: "replace", Wrong: "Hi how", Right: "Hi, how"}}},
		{"spelling", "spelling", []Change{{Op: "replace", Wrong: "howe", Right: "how"}}},
		{"a word", "grammar", []Change{{Op: "insert", Wrong: "", Right: " the"}}},
		{"widest wins", "grammar", []Change{{Op: "delete", Wrong: "?", Right: ""}, {Op: "replace", Wrong: "cat sat", Right: "cats"}}},
		{"spelling over punctuation", "spelling", []Change{{Op: "delete", Wrong: "?", Right: ""}, {Op: "replace", Wrong: "e", Right: ""}}},
	}
	for _, c := range cases {
		if got := ReasonFromChanges(c.edit); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestCorrectionReason(t *testing.T) {
	spelling := []Change{{Op: "replace", Wrong: "howe", Right: "how"}}
	cases := []struct {
		reason, note, want string
		changes            []Change
	}{
		{"spelling", "", "spelling", nil},                             // the editor's word wins
		{" Typo. ", "", "spelling", nil},                              // aliases, however punctuated
		{"capitalization", "", "capitalisation", nil},                 // and however spelt
		{"word order", "", "word-order", nil},                         // spaces become hyphens
		{"none", "it is cut off mid-sentence", "fragment", spelling},  // "none" never tags a changed text: the note is read
		{"", "pure gibberish", "nonsense", spelling},                  // the critique vocabulary maps onto the editor's
		{"", "the sentence contradicts itself", "spelling", spelling}, // a reason the editor has no word for: the diff
		{"", "", "spelling", spelling},                                // no words at all: the diff
		{"", "", "grammar", nil},                                      // and no diff either: grammar, never none
		{"vibes", "", "spelling", spelling},                           // an unknown word: the diff
	}
	for _, c := range cases {
		if got := CorrectionReason(c.reason, c.note, c.changes); got != c.want {
			t.Errorf("(%q, %q): got %q, want %q", c.reason, c.note, got, c.want)
		}
	}
}

func TestFaultsFromCorrectionsAndTeach(t *testing.T) {
	fixed, same, unread := "Hi, how are you?", "the cat sat on the mat", "the dog"
	entries := []CorrectionEntry{
		{Text: "Hi howe are you??", Correction: &fixed, Verdict: "corrected", Reason: "spelling", Note: "spelling fixed",
			Changes: []Change{{Op: "replace", Wrong: "howe", Right: "how"}}},
		{Text: same, Correction: &same, Verdict: "unchanged", Reason: "none", Note: "nothing"},
		{Text: unread, Correction: nil, Verdict: "uncorrected", Note: "no correction returned"},
		{Text: "", Correction: &fixed, Verdict: "corrected"},                          // nothing to blame
		{Text: same, Correction: &same, Verdict: "corrected"},                         // the verdict lies: no change, no fault
		{Text: "cat", Correction: &fixed, Verdict: "corrected", Reason: "none"},       // "none" on a changed text: reason from the note / diff
		{Text: "as it was", Correction: strPtr("as it was"), Verdict: "", Reason: ""}, // no verdict, no change: unchanged
	}
	faults, unchanged := FaultsFromCorrections(entries, -1.5, "correction")
	if len(faults) != 2 || len(unchanged) != 2 {
		t.Fatalf("two faults, two unchanged: %+v %v", faults, unchanged)
	}
	if f := faults[0]; f.Correction != fixed || f.Reason != "spelling" || f.Severity != 1.5 || f.Note != "spelling fixed" || f.Source != "correction" {
		t.Fatalf("the fault carries the correction, the editor's word and |severity|: %+v", f)
	}
	if faults[1].Reason != "grammar" {
		t.Fatalf("\"none\" on a changed text is replaced: %+v", faults[1])
	}
	if unchanged[0] != same || unchanged[1] != "as it was" {
		t.Fatalf("the unchanged texts: %v", unchanged)
	}

	negative, err := NewNegativeModel(1, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	report, err := TeachCorrections(negative, entries[:3], 1.5, true, "", TeachOptions{})
	if err != nil {
		t.Fatalf("TeachCorrections: %v", err)
	}
	if report.Blamed != 1 || report.Passed != 1 || report.Uncorrected != 1 || report.Source != "correction" || report.Severity != 1.5 {
		t.Fatalf("the report: %+v", report)
	}
	if report.Edits == 0 || report.Edges == 0 || report.Reasons["spelling"] != 1 || report.SeverityMean != 1.5 {
		t.Fatalf("the diff was blamed under the editor's word: %+v", report)
	}
	if len(report.Records) == 0 || report.Records[0]["phase"] != "correction" || report.Records[0]["edits"] != report.Edits {
		t.Fatalf("the correction's outcome is on record: %+v", report.Records)
	}
	if verdict := negative.Judge(fixed, JudgeOptions{}); verdict.Verdict != "pass" {
		t.Fatalf("the correction itself is clean: %+v", verdict)
	}
	verdict := negative.Judge("Hi howe are you??", JudgeOptions{})
	if len(verdict.Reasons) == 0 || verdict.Reasons[0].Reason != "spelling" || verdict.Blamed == 0 {
		t.Fatalf("the mistake is known, and why: %+v", verdict)
	}
	// passes do not clear when asked not to; the count of unchanged texts is still reported
	fresh, _ := NewNegativeModel(1, DefaultNegativeOptions())
	report, err = TeachCorrections(fresh, entries[:3], 0.5, false, "editor", TeachOptions{})
	if err != nil || report.Cleared != 0 || report.Passed != 1 || report.Source != "editor" {
		t.Fatalf("clear off: %v %+v", err, report)
	}
}

func strPtr(s string) *string { return &s }
