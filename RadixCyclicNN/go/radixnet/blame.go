package radixnet

import (
	"math"
	"strings"
)

// Where the negative network's data comes from: the tutor's verdicts, turned into blame.
//
// Nothing in the negative network is invented.  Every failure it knows was
// handed to it by something outside the network that looked at an output and
// said it was wrong, and why:
//
//   - the English tutor (tutor.go): the LLM sets an exercise, the network
//     completes it and the LLM marks the sentence - a mark out of 10, the
//     single worst mistake named out of ErrorTypes (agreement, tense, article,
//     ...), one sentence of teaching and the correction, the same sentence
//     written out in correct English.  The mistake is the reason, the mark is
//     the severity, and the diff against the correction says which characters
//     were wrong (FaultsFromLessons, Model.BlameCorrection);
//   - a person blaming a text by hand (the CLI's `negative blame`, the API).
//
// A Fault is a text, a reason (the tutor's own error type, or one out of
// Reasons picked from its words by Classify), a severity (how badly it failed)
// and a note (the tutor's sentence, kept verbatim for the journal).  Teach
// hands faults to the negative network; the texts the tutor passed are cleared
// in the same call, so a fragment that shows up in good and bad output alike
// stops carrying the verdict on its own.

// DefaultReason is recorded when nothing in the tutor's words is recognised.
const DefaultReason = "other"

// Reasons are the reason tags of reviewed text (the order Classify tries them in).
var Reasons = []string{
	"empty", "gibberish", "repetition", "truncated", "grammar", "spelling", "contradiction", "false",
	"incoherent", "off-topic", DefaultReason,
}

type reasonPattern struct {
	reason  string
	needles []string
}

var reasonPatterns = []reasonPattern{
	{"empty", []string{"empty output", "no output", "produced nothing", "blank"}},
	{"gibberish", []string{"gibberish", "nonsense word", "word salad", "garbled", "random character", "not words", "noise"}},
	{"repetition", []string{"repeat", "repetit", "duplicat", "over and over", "loops"}},
	{"truncated", []string{"truncat", "cut off", "cut short", "incomplete", "unfinished", "mid-sentence", "mid sentence"}},
	{"grammar", []string{"grammar", "grammatic", "syntax", "word order", "punctuation", "agreement", "tense", "malformed sentence"}},
	{"spelling", []string{"spelling", "misspell", "typo"}},
	{"contradiction", []string{"contradict", "inconsisten", "conflicts with"}},
	{"false", []string{"false", "factual", "inaccurate", "untrue", "wrong fact", "not true", "misleading"}},
	{"incoherent", []string{"incoheren", "meaningless", "make no sense", "does not make sense", "doesn't make sense",
		"nonsensical", "confusing", "unintelligible"}},
	{"off-topic", []string{"off-topic", "off topic", "irrelevant", "unrelated", "does not answer", "ignores the prompt"}},
}

// ClassifyOptions are what else is known about a critique.
type ClassifyOptions struct {
	Verdict string
	Rating  *float64
	Default string
}

// Classify picks the reason tag behind a tutor's critique; its own words
// decide, and Default (else "other") covers what the vocabulary does not know.
//
// "unrated" comes back when the tutor failed a text without a critique the
// vocabulary recognises - it still failed, and the network records that nobody
// said why.
func Classify(critique string, o ClassifyOptions) string {
	text := strings.ToLower(strings.Join(strings.Fields(critique), " "))
	for _, pattern := range reasonPatterns {
		for _, needle := range pattern.needles {
			if strings.Contains(text, needle) {
				return pattern.reason
			}
		}
	}
	if strings.ToLower(strings.TrimSpace(o.Verdict)) == "unrated" {
		return "unrated"
	}
	if o.Rating != nil && *o.Rating <= 0 && text == "" {
		return "gibberish"
	}
	if o.Default != "" {
		return o.Default
	}
	return DefaultReason
}

// SeverityFromRating maps a mark out of 10 onto how heavily the failure is
// blamed: 2.0 at rating 0, 0.25 at the pass threshold.  An unrated failure
// (rating nil) is blamed like one ordinary failure.
func SeverityFromRating(rating *float64, threshold float64) float64 {
	const floor, ceiling = 0.25, 2.0
	if rating == nil {
		return 1
	}
	limit := threshold
	if limit <= 0 {
		limit = 1
	}
	share := math.Max(0, math.Min(1, (limit-*rating)/limit))
	return floor + (ceiling-floor)*share
}

// Fault is one lesson for the negative network.
type Fault struct {
	Text       string  `json:"text"`
	Reason     string  `json:"reason"`
	Severity   float64 `json:"severity"`
	Note       string  `json:"note"`
	Source     string  `json:"source"`
	Correction string  `json:"correction,omitempty"`
}

// FaultsFromLessons turns a round of English lessons into faults and the texts
// that clear blame.
//
// The English tutor is the richest source of negatives there is: the mistake
// it named (Grade.Error, one of ErrorTypes) is the reason, its mark is the
// severity, its sentence of teaching is the note and its correction rides
// along in the fault, so Teach can blame only the characters the teacher
// actually changed.  The sentences that passed, the corrections themselves and
// the teacher's own model answers all come back as cleared text.
func FaultsFromLessons(lessons []*Lesson, threshold float64, source string) ([]Fault, []string) {
	faults := []Fault{}
	passed := []string{}
	seen := map[string]bool{}
	add := func(text string) {
		text = strings.Join(strings.Fields(text), " ")
		if text == "" || seen[text] {
			return
		}
		seen[text] = true
		passed = append(passed, text)
	}
	for _, lesson := range lessons {
		if lesson == nil {
			continue
		}
		sentence := strings.Join(strings.Fields(lesson.Sentence), " ")
		correction := strings.Join(strings.Fields(lesson.Grade.Correction), " ")
		answer := strings.Join(strings.Fields(lesson.Exercise.Answer), " ")
		switch {
		case sentence != "" && lesson.Grade.Passed:
			add(sentence)
		case sentence != "":
			reason := strings.ToLower(strings.TrimSpace(lesson.Grade.Error))
			if reason == "" || reason == "none" {
				reason = Classify(lesson.Grade.Comment, ClassifyOptions{Verdict: lesson.Grade.GradedBy})
			}
			fault := Fault{
				Text: sentence, Reason: reason, Severity: SeverityFromRating(lesson.Grade.Score, threshold),
				Note: lesson.Grade.Comment, Source: source,
			}
			if correction != "" && correction != sentence {
				fault.Correction = correction
			}
			faults = append(faults, fault)
		}
		add(correction)
		add(answer)
	}
	return faults, passed
}

// TeachReport is what one round of blaming taught the negative network.
type TeachReport struct {
	Blamed       int              `json:"blamed"`
	Cleared      int              `json:"cleared"`
	Unmatched    int              `json:"unmatched"`
	Edges        int              `json:"edges"`
	Reasons      map[string]int   `json:"reasons"`
	SeverityMean float64          `json:"severity_mean"`
	Passed       int              `json:"passed"`
	Threshold    float64          `json:"threshold"`
	Source       string           `json:"source"`
	Faults       []Fault          `json:"faults"`
	Records      []map[string]any `json:"records"`
}

// TeachOptions steer one Teach call.
type TeachOptions struct {
	Epochs      int
	ClearEpochs int
	Progress    func(map[string]any)
	Stop        func() bool
}

// Teach hands the tutor's faults to the negative network; the passed texts
// clear blame afterwards.  A fault that carries a correction is taught from
// its diff, so only the characters the teacher changed are blamed.
func Teach(negative *Model, faults []Fault, passed []string, o TeachOptions) (*TeachReport, error) {
	report := &TeachReport{Reasons: map[string]int{}, Faults: faults, Records: []map[string]any{}}
	if negative == nil {
		return report, nil
	}
	if err := negative.requireNegative("teaching"); err != nil {
		return nil, err
	}
	severities := 0.0
	for _, fault := range faults {
		if o.Stop != nil && o.Stop() {
			break
		}
		if strings.TrimSpace(fault.Text) == "" {
			continue
		}
		reason := fault.Reason
		if reason == "" {
			reason = DefaultReason
		}
		severity := fault.Severity
		if severity == 0 {
			severity = 1
		}
		source := fault.Source
		if source == "" {
			source = "tutor"
		}
		opts := BlameOptions{Reason: reason, Severity: severity, Source: source, Note: fault.Note,
			Epochs: o.Epochs, Progress: o.Progress, Stop: o.Stop}
		if fault.Correction != "" {
			// the tutor wrote the sentence out correctly: blame only the characters it changed
			out, err := negative.BlameCorrection(fault.Text, fault.Correction, opts)
			if err != nil {
				return nil, err
			}
			report.Edges += out.Blamed
		} else {
			records, err := negative.Blame([]string{fault.Text}, opts)
			if err != nil {
				return nil, err
			}
			if len(records) > 0 {
				report.Edges += int(toFloat(records[len(records)-1]["edges_touched"]))
				report.Records = append(report.Records, records...)
			}
		}
		report.Reasons[reason]++
		report.Blamed++
		severities += severity
	}
	if report.Blamed > 0 {
		report.SeverityMean = severities / float64(report.Blamed)
	}
	texts := []string{}
	for _, text := range passed {
		if strings.TrimSpace(text) != "" {
			texts = append(texts, text)
		}
	}
	report.Passed = len(texts)
	if len(texts) > 0 && !(o.Stop != nil && o.Stop()) {
		records, err := negative.Clear(texts, 1, o.ClearEpochs)
		if err != nil {
			return nil, err
		}
		if len(records) > 0 {
			last := records[len(records)-1]
			report.Cleared = int(toFloat(last["matched"]))
			report.Unmatched = int(toFloat(last["unmatched"]))
			report.Records = append(report.Records, records...)
		}
	}
	return report, nil
}

// TeachLessons feeds a round of English lessons into the negative network.
func TeachLessons(negative *Model, lessons []*Lesson, threshold float64, clearPasses bool, source string, o TeachOptions) (*TeachReport, error) {
	if source == "" {
		source = "tutor"
	}
	if threshold <= 0 {
		threshold = 6
	}
	faults, passed := FaultsFromLessons(lessons, threshold, source)
	if !clearPasses {
		passed = nil
	}
	report, err := Teach(negative, faults, passed, o)
	if err != nil {
		return nil, err
	}
	report.Source, report.Threshold = source, threshold
	return report, nil
}
