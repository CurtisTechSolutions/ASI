package radixnet

import (
	"math"
	"strings"
	"unicode"
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
//   - the adversarial reviewer (review.go): an LLM rates the network's own
//     texts 0-10, passes or fails each one and writes a one-sentence critique;
//   - the copy editor (CorrectTexts): the same LLM writes each of the
//     network's texts out correctly, changing as little as it can, and the diff
//     between the two is the lesson - "Hi howe are you??" against "Hi, how are
//     you?" blames the e and the second ?, not the sentence
//     (FaultsFromCorrections, Model.BlameCorrection); a text it handed back
//     unchanged clears blame;
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

// CorrectionReasons are the reason tags of a corrected text: what the copy
// editor's smallest change put right ("none": nothing).
var CorrectionReasons = []string{
	"spelling", "punctuation", "capitalisation", "spacing", "agreement", "tense", "article", "preposition",
	"plural", "pronoun", "word-order", "vocabulary", "repetition", "fragment", "nonsense", "grammar", "none",
}

// CorrectionSeverity is how heavily one correction is blamed: one ordinary
// failure per corrected text, placed only on its changed characters.
const CorrectionSeverity = 1.0

// correctionAliases are the editor's other words for the vocabulary's reasons.
var correctionAliases = map[string]string{
	"typo": "spelling", "misspelling": "spelling", "misspelt": "spelling", "spell": "spelling",
	"capitalization": "capitalisation", "case": "capitalisation", "casing": "capitalisation",
	"capital": "capitalisation", "uppercase": "capitalisation", "lowercase": "capitalisation",
	"whitespace": "spacing", "space": "spacing", "spaces": "spacing",
	"subject-verb": "agreement", "verb-agreement": "agreement", "conjugation": "agreement",
	"articles": "article", "prepositions": "preposition", "plurals": "plural", "number": "plural",
	"pronouns": "pronoun", "order": "word-order", "word order": "word-order", "syntax": "grammar",
	"wording": "vocabulary", "word-choice": "vocabulary", "word choice": "vocabulary", "word": "vocabulary",
	"repeat": "repetition", "repeated": "repetition", "duplicate": "repetition", "duplication": "repetition",
	"incomplete": "fragment", "truncated": "fragment", "unfinished": "fragment",
	"gibberish": "nonsense", "meaningless": "nonsense", "garbled": "nonsense",
	"ok": "none", "correct": "none", "nothing": "none", "unchanged": "none", "no change": "none",
}

// isCorrectionReason reports whether word is one of CorrectionReasons.
func isCorrectionReason(word string) bool {
	for _, reason := range CorrectionReasons {
		if reason == word {
			return true
		}
	}
	return false
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

// punctuation is what a change that only moves punctuation is made of.
const punctuation = ".,;:!?'\"-()[]{}\u2018\u2019\u201c\u201d\u2013\u2014/&"

func isPunctuation(r rune) bool { return strings.ContainsRune(punctuation, r) }

// ReasonFromChanges is the reason a diff speaks for itself, from what its
// edits touched ("none" when nothing changed).
//
// Every edit is read for what it moved and the widest kind wins: a whole word
// inserted or struck out is grammar, a change that only turns a word into
// another word is spelling, and one that only touches punctuation, letter case
// or spaces is that.
func ReasonFromChanges(changes []Change) string {
	kinds := map[string]bool{}
	for _, edit := range changes {
		if edit.Op == "equal" || edit.Wrong == edit.Right {
			continue
		}
		kinds[changeKind(edit.Wrong, edit.Right)] = true
	}
	for _, kind := range []string{"grammar", "spelling", "punctuation", "spacing", "capitalisation"} {
		if kinds[kind] {
			return kind
		}
	}
	return "none"
}

// changeKind is what one edit moved: punctuation, spaces, letter case, the
// letters of a word, or whole words.
func changeKind(wrong, right string) string {
	moved := wrong + right
	if moved != "" && strings.IndexFunc(moved, func(r rune) bool { return !isPunctuation(r) }) < 0 {
		return "punctuation"
	}
	if moved != "" && strings.TrimSpace(moved) == "" {
		return "spacing"
	}
	if strings.ToLower(wrong) == strings.ToLower(right) {
		return "capitalisation"
	}
	without := func(s string) string {
		return strings.Map(func(r rune) rune {
			if isPunctuation(r) || unicode.IsSpace(r) {
				return -1
			}
			return r
		}, s)
	}
	if without(wrong) == without(right) {
		// only punctuation and spaces moved, however the letters were carried along
		if strings.IndexFunc(moved, isPunctuation) >= 0 {
			return "punctuation"
		}
		return "spacing"
	}
	if strings.IndexFunc(wrong, unicode.IsSpace) >= 0 || strings.IndexFunc(right, unicode.IsSpace) >= 0 {
		return "grammar" // a space moved with the letters: a word was added, dropped or reordered
	}
	return "spelling" // letters changed inside one word
}

// CorrectionReason is the reason tag behind a copy editor's correction, out of
// CorrectionReasons.
//
// The editor's own word wins when the vocabulary knows it (aliases such as
// typo or capitalization are accepted); failing that its note is read the way
// a critique is (Classify), and failing that the diff decides
// (ReasonFromChanges).  "none" is only ever what the diff says about an
// unchanged text, so a changed text is never tagged with it.
func CorrectionReason(reason, note string, changes []Change) string {
	word := strings.Trim(strings.ToLower(collapse(reason)), ".:;,\"'")
	if alias, ok := correctionAliases[word]; ok {
		word = alias
	}
	if word != "none" && isCorrectionReason(word) {
		return word
	}
	if hyphenated := strings.ReplaceAll(word, " ", "-"); word != "none" && isCorrectionReason(hyphenated) {
		return hyphenated
	}
	// the note is read as a critique; "other" (nothing recognised) is not a correction reason and falls through
	spoken := Classify(note, ClassifyOptions{Default: DefaultReason})
	if mapped, ok := map[string]string{"gibberish": "nonsense", "truncated": "fragment", "incoherent": "nonsense"}[spoken]; ok {
		spoken = mapped
	}
	if isCorrectionReason(spoken) {
		return spoken
	}
	if fromDiff := ReasonFromChanges(changes); fromDiff != "none" {
		return fromDiff
	}
	return "grammar"
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
			why := strings.Join(strings.Fields(lesson.Why), " ")
			for _, variant := range lesson.Variants {
				wrong := strings.Join(strings.Fields(variant.Wrong), " ")
				right := strings.Join(strings.Fields(variant.Right), " ")
				if wrong == "" || wrong == sentence {
					continue
				}
				weight := variant.Weight
				if weight < 0 {
					weight = 0
				}
				note := why
				if note == "" {
					note = lesson.Grade.Comment
				}
				similar := Fault{
					Text: wrong, Reason: reason, Severity: fault.Severity * weight, Note: note,
					Source: source + ":similar",
				}
				if right != "" && right != wrong {
					similar.Correction = right
					add(right)
				}
				faults = append(faults, similar)
			}
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
	// Severity is the blame per corrected text (TeachCorrections); Edits the
	// characters the editor changed over all corrections, in units of the
	// network's encoding; Uncorrected the texts it said nothing usable about.
	Severity    float64 `json:"severity"`
	Edits       int     `json:"edits"`
	Uncorrected int     `json:"uncorrected"`
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
			report.Edits += out.Edits
			report.Records = append(report.Records, correctionRecord(out))
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

// correctionRecord is a taught correction as the report's records carry it
// (the Python outcome dict, so both journals read the same).
func correctionRecord(out *NegativeCorrection) map[string]any {
	changes := out.Changes
	if changes == nil {
		changes = []Edit{}
	}
	return map[string]any{
		"changes": changes, "edits": out.Edits, "blamed": out.Blamed, "cleared": out.Cleared, "reason": out.Reason,
		"severity": out.Severity, "phase": out.Phase, "wrong_chars": out.WrongChars, "right_chars": out.RightChars,
	}
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

// FaultsFromReviews turns an adversarial review into faults and the texts that
// clear blame.
//
// Everything the reviewer did not pass becomes a fault whose reason comes from
// its critique and whose severity comes from its rating; the texts it passed
// come back separately so they can take blame off what they share.
func FaultsFromReviews(reviews []Review, threshold float64, source string) ([]Fault, []string) {
	faults := []Fault{}
	passed := []string{}
	for _, review := range reviews {
		if review.Text == "" {
			continue
		}
		if strings.ToLower(strings.TrimSpace(review.Verdict)) == "pass" {
			passed = append(passed, review.Text)
			continue
		}
		faults = append(faults, Fault{
			Text:     review.Text,
			Reason:   Classify(review.Critique, ClassifyOptions{Verdict: review.Verdict, Rating: review.Rating}),
			Severity: SeverityFromRating(review.Rating, threshold),
			Note:     review.Critique,
			Source:   source,
		})
	}
	return faults, passed
}

// TeachReviews feeds an adversarial review straight into the negative network.
func TeachReviews(negative *Model, reviews []Review, threshold float64, clearPasses bool, source string,
	o TeachOptions) (*TeachReport, error) {
	if source == "" {
		source = "review"
	}
	if threshold <= 0 {
		threshold = 6
	}
	faults, passed := FaultsFromReviews(reviews, threshold, source)
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

// FaultsFromCorrections turns a copy editor's corrections into faults and the
// texts that clear blame.
//
// Every text the editor changed becomes a fault that carries its correction,
// so Teach routes it through Model.BlameCorrection and only the characters the
// editor struck out or replaced are blamed; its reason is the editor's word for
// the mistake (CorrectionReason) and its note the editor's sentence.  The
// texts it handed back unchanged come back separately to clear blame.  A text
// it gave no usable answer for ("uncorrected") is neither: nobody said
// anything about it.
func FaultsFromCorrections(corrections []CorrectionEntry, severity float64, source string) ([]Fault, []string) {
	faults := []Fault{}
	unchanged := []string{}
	amount := math.Abs(severity)
	for _, entry := range corrections {
		if entry.Text == "" || entry.Correction == nil {
			continue
		}
		correction := *entry.Correction
		verdict := strings.ToLower(strings.TrimSpace(entry.Verdict))
		if verdict == "unchanged" || (verdict == "" && correction == entry.Text) {
			unchanged = append(unchanged, entry.Text)
			continue
		}
		if correction == entry.Text {
			continue
		}
		reason := strings.ToLower(strings.TrimSpace(entry.Reason))
		if reason == "none" || !isCorrectionReason(reason) {
			reason = CorrectionReason(reason, entry.Note, entry.Changes)
		}
		faults = append(faults, Fault{
			Text: entry.Text, Reason: reason, Severity: amount, Note: entry.Note, Source: source,
			Correction: correction,
		})
	}
	return faults, unchanged
}

// TeachCorrections feeds a copy editor's corrections straight into the
// negative network (see FaultsFromCorrections).
//
// The report is Teach's plus Source, Severity, Faults, Passed (the unchanged
// texts), Edits (the characters the editor changed over all faults, in units
// of the network's encoding) and Uncorrected.
func TeachCorrections(negative *Model, corrections []CorrectionEntry, severity float64, clearPasses bool,
	source string, o TeachOptions) (*TeachReport, error) {
	if source == "" {
		source = "correction"
	}
	faults, unchanged := FaultsFromCorrections(corrections, severity, source)
	passed := unchanged
	if !clearPasses {
		passed = nil
	}
	report, err := Teach(negative, faults, passed, o)
	if err != nil {
		return nil, err
	}
	report.Source, report.Severity, report.Passed = source, math.Abs(severity), len(unchanged)
	for _, entry := range corrections {
		if entry.Verdict == "uncorrected" {
			report.Uncorrected++
		}
	}
	return report, nil
}

// CodeReasons are the reason tags of a reviewed *program*.
var CodeReasons = []string{"timeout", "crash", "wrong-output", "task-not-done", "style", "naming", DefaultReason}

// CodeSeverity is how heavily each code failure is blamed (1 = one ordinary failure).
var CodeSeverity = map[string]float64{
	"timeout": 1.5, "crash": 1.5, "wrong-output": 1.25, "task-not-done": 1.0,
	"style": 0.5, "naming": 0.5, DefaultReason: 1.0,
}

// CodeReason is why a code attempt was rejected, from the sandbox, the style
// report and the judge, in the order those matter.
func CodeReason(attempt *Attempt) string {
	if attempt == nil {
		return DefaultReason
	}
	run, style, verdict := attempt.Run, attempt.Style, attempt.Verdict
	switch {
	case run != nil && run.TimedOut:
		return "timeout"
	case run != nil && !run.OK:
		return "crash"
	case run != nil && run.ExpectedOK != nil && !*run.ExpectedOK:
		return "wrong-output"
	case verdict.Task != nil && !*verdict.Task:
		return "task-not-done"
	case !verdict.Naming || !style.NamingOK:
		return "naming"
	case !verdict.PEP8 || !style.PEP8OK:
		return "style"
	}
	words := verdict.Critique
	if strings.TrimSpace(words) == "" {
		words = strings.Join(firstN(verdict.Issues, 3), "; ")
	}
	return Classify(words, ClassifyOptions{})
}

// FaultsFromAttempts turns a problem's attempts into faults and the texts that
// clear blame: a rejected program is blamed for what the sandbox, the style
// checker or the judge found, and the accepted ones clear.
func FaultsFromAttempts(attempts []*Attempt, source string) ([]Fault, []string) {
	faults := []Fault{}
	correct := []string{}
	for _, attempt := range attempts {
		if attempt == nil || attempt.Text == "" {
			continue
		}
		if attempt.Verdict.Correct {
			correct = append(correct, attempt.Text)
			continue
		}
		reason := CodeReason(attempt)
		severity, ok := CodeSeverity[reason]
		if !ok {
			severity = 1
		}
		faults = append(faults, Fault{
			Text: attempt.Text, Reason: reason, Severity: severity,
			Note: attempt.Feedback(), Source: source,
		})
	}
	return faults, correct
}

// TeachAttempts feeds the code-generation teacher's rejected attempts into the
// negative network.
func TeachAttempts(negative *Model, attempts []*Attempt, clearPasses bool, source string,
	o TeachOptions) (*TeachReport, error) {
	if source == "" {
		source = "codegen"
	}
	faults, correct := FaultsFromAttempts(attempts, source)
	if !clearPasses {
		correct = nil
	}
	report, err := Teach(negative, faults, correct, o)
	if err != nil {
		return nil, err
	}
	report.Source = source
	return report, nil
}
