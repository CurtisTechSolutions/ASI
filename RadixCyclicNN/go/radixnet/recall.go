package radixnet

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// What the network remembers of what it was shown: the recall tutor for speech
// and images.
//
// The English tutor needs an LLM because nobody knows in advance what the right
// sentence is.  Speech and images need none, because the right answer is on
// file: an utterance or a picture was encoded into text (speech.go, vision.go)
// and trained on, so asking the network to write that text again is an exercise
// whose correction already exists.
//
// One round is the same three steps the English tutor takes: the opening of a
// text it was taught, the network's completion, and a marking that runs what
// came back through the codec and compares it with the original.

// RecallModalities are the two things a network is taught that it can be asked
// to remember exactly.
var RecallModalities = []string{"speech", "image"}

// DefaultRecallLead is the payload characters the exercise gives away.  A spoken
// text carries a token of its own, so the exercise can be the token alone; an
// image header is the same for every picture of that size, so without a lead
// there is nothing to say which one is wanted.
var DefaultRecallLead = map[string]int{"speech": 0, "image": 16}

// RecallTolerance is the byte distance at which two payloads stop agreeing at
// all (one eighth of the range).
//
// Agreement is per byte, max(0, 1 - |written - true| / tolerance), averaged over
// the payload that was asked for - so a byte the network never wrote counts as
// total disagreement and truncation needs no separate penalty.  An exact recall
// scores 1, one off by a quantisation step or two ~0.95, and random bytes ~0.12:
// the metric has to be generous about the last bits of a quantised sample and
// unforgiving about noise.
const RecallTolerance = 32.0

// Length ratios outside which a completion is short or long rather than merely
// wrong, and the agreement below which a readable, right-length recall is still
// a distortion (an image: a drift).
const (
	RecallTruncated = 0.9
	RecallOverrun   = 1.1
	RecallAgreement = 0.9
)

const (
	recallFlatPeak   = 0.02 // below this peak a decoded waveform is silence
	recallFlatSpread = 2.0  // below this byte spread an image payload has no picture in it
	recallRailed     = 0.5  // fraction of railed samples above which it is clipping / noise
)

// ModalityOf reads "speech" or "image" out of a text's own header.
func ModalityOf(text string) (string, error) {
	if strings.Contains(text, "aud:") {
		return "speech", nil
	}
	if strings.Contains(text, "img:") {
		return "image", nil
	}
	return "", fmt.Errorf("not an encoded utterance or image: expected an 'aud:' or 'img:' header")
}

// payloadAt is the index where the base64 payload starts (just past the header).
func payloadAt(text string) (int, error) {
	for _, header := range []string{"aud:", "img:"} {
		at := strings.Index(text, header)
		if at < 0 {
			continue
		}
		colons := 0
		for i := at; i < len(text); i++ {
			if text[i] == ':' {
				colons++
				if colons == 3 {
					return i + 1, nil
				}
			}
		}
		return len(text), nil
	}
	return 0, fmt.Errorf("not an encoded utterance or image: expected an 'aud:' or 'img:' header")
}

// RecallCue is the opening the network is given: everything up to the payload,
// plus lead characters of it.  A negative lead means the modality's default.
func RecallCue(text string, lead int) (string, error) {
	start, err := payloadAt(text)
	if err != nil {
		return "", err
	}
	if lead < 0 {
		modality, err := ModalityOf(text)
		if err != nil {
			return "", err
		}
		lead = DefaultRecallLead[modality]
	}
	end := start + lead
	if end > len(text) {
		end = len(text)
	}
	return text[:end], nil
}

// RecallExercise is one text the network was taught, and the opening it is asked
// to continue.
type RecallExercise struct {
	ID        string `json:"id"`
	Modality  string `json:"modality"`
	Reference string `json:"-"` // the whole true text
	Cue       string `json:"cue"`
	Label     string `json:"label"`
	// Index is which of the given texts it came from: texts that are not encoded
	// are skipped, so this is not its position in the list.
	Index int `json:"index"`
}

// Answer is the continuation that would be perfect.
func (e *RecallExercise) Answer() string { return e.Reference[len(e.Cue):] }

// RecallExercises turns taught texts into exercises; texts that are not encoded
// are skipped.
func RecallExercises(texts []string, lead int, labels []string) []RecallExercise {
	out := []RecallExercise{}
	for i, text := range texts {
		body := strings.TrimSpace(text)
		if body == "" {
			continue
		}
		modality, err := ModalityOf(body)
		if err != nil {
			continue
		}
		cue, err := RecallCue(body, lead)
		if err != nil {
			continue
		}
		label := ""
		if i < len(labels) {
			label = labels[i]
		}
		out = append(out, RecallExercise{
			ID: fmt.Sprintf("r%d", i+1), Modality: modality, Reference: body, Cue: cue, Label: label, Index: i,
		})
	}
	return out
}

// -- the marking --------------------------------------------------------------

// recallBytes is (payload, repaired, error) of an encoded text.
func recallBytes(text, modality string) ([]byte, bool, string) {
	var payload []byte
	var repaired bool
	var err error
	if modality == "speech" {
		_, _, _, payload, repaired, err = ParseSpeechText(text)
	} else {
		_, _, _, payload, repaired, err = ParseImageText(text)
	}
	if err != nil {
		return nil, false, err.Error()
	}
	return payload, repaired, ""
}

// recallAgreement is the mean per-byte agreement over truth; bytes never
// written count as 0.
func recallAgreement(written, truth []byte) float64 {
	if len(truth) == 0 {
		if len(written) == 0 {
			return 1
		}
		return 0
	}
	total := 0.0
	for i, expected := range truth {
		if i >= len(written) {
			break // the rest was never written: no agreement to add
		}
		total += math.Max(0, 1-math.Abs(float64(written[i])-float64(expected))/RecallTolerance)
	}
	return total / float64(len(truth))
}

// recallSpread is the standard deviation of the payload bytes.
func recallSpread(payload []byte) float64 {
	if len(payload) == 0 {
		return 0
	}
	mean := 0.0
	for _, b := range payload {
		mean += float64(b)
	}
	mean /= float64(len(payload))
	sum := 0.0
	for _, b := range payload {
		sum += (float64(b) - mean) * (float64(b) - mean)
	}
	return math.Sqrt(sum / float64(len(payload)))
}

// recallRailedFraction is the share of bytes sitting against either end.
func recallRailedFraction(payload []byte) float64 {
	if len(payload) == 0 {
		return 0
	}
	n := 0
	for _, b := range payload {
		if b <= 2 || b >= 253 {
			n++
		}
	}
	return float64(n) / float64(len(payload))
}

// recallPeak is the peak amplitude of a decoded waveform payload.
func recallPeak(payload []byte, codec string) (float64, error) {
	if len(payload) == 0 {
		return 0, nil
	}
	_, _, decode, err := GetSpeechCodec(codec)
	if err != nil {
		return 0, err
	}
	peak := 0.0
	for _, value := range decode(payload) {
		peak = math.Max(peak, math.Abs(float64(value)))
	}
	return peak, nil
}

// codecOf is the codec a waveform text declares ("auto" when it declares none
// the reader understands).
func codecOf(text string) string {
	codec, _, _, _, _, err := ParseSpeechText(text)
	if err != nil {
		return "auto"
	}
	return codec
}

// recallSilent: does this waveform payload decode to (near) nothing?
func recallSilent(payload []byte, codec string) bool {
	peak, err := recallPeak(payload, codec)
	if err != nil { // a codec the reader does not know: fall back to the raw byte spread
		return recallSpread(payload) < recallFlatSpread
	}
	return peak < recallFlatPeak
}

// RecallFacts is what a check reports: facts, not a verdict.
type RecallFacts struct {
	Modality         string  `json:"modality"`
	Readable         bool    `json:"readable"`
	Repaired         bool    `json:"repaired"`
	Error            string  `json:"error"`
	WrittenBytes     int     `json:"written_bytes"`
	ExpectedBytes    int     `json:"expected_bytes"`
	LengthRatio      float64 `json:"length_ratio"`
	Agreement        float64 `json:"agreement"`
	Flat             bool    `json:"flat"`
	ReferenceFlat    bool    `json:"reference_flat"`
	Extreme          bool    `json:"extreme"`
	ReferenceExtreme bool    `json:"reference_extreme"`
	Said             string  `json:"said"`
	Heard            string  `json:"heard"`
	Match            *bool   `json:"match"`
}

// recallCommon is the facts both modalities share.
func recallCommon(payload, truth []byte, repaired bool, failure, modality string) RecallFacts {
	readable := failure == ""
	ratio := 1.0
	switch {
	case len(truth) > 0:
		ratio = float64(len(payload)) / float64(len(truth))
	case len(payload) > 0:
		ratio = 2
	}
	facts := RecallFacts{
		Modality: modality, Readable: readable, Repaired: repaired, Error: failure,
		WrittenBytes: len(payload), ExpectedBytes: len(truth), LengthRatio: ratio,
		ReferenceExtreme: recallRailedFraction(truth) > recallRailed,
	}
	if readable {
		facts.Agreement = recallAgreement(payload, truth)
		facts.Extreme = recallRailedFraction(payload) > recallRailed
	}
	return facts
}

// CheckSpeechRecall reports the facts about a recalled waveform, against the
// waveform it should have been.  said / heard are the words that were spoken and
// the round-trip transcript; when both are given the completion is also marked
// on whether it still says the same thing.
func CheckSpeechRecall(written, reference, said, heard string) RecallFacts {
	truth, _, _ := recallBytes(reference, "speech")
	payload, repaired, failure := recallBytes(written, "speech")
	facts := recallCommon(payload, truth, repaired, failure, "speech")
	if facts.Readable {
		facts.Flat = recallSilent(payload, codecOf(written))
	}
	facts.ReferenceFlat = len(truth) > 0 && recallSilent(truth, codecOf(reference))
	facts.Said = strings.Join(strings.Fields(said), " ")
	facts.Heard = strings.Join(strings.Fields(heard), " ")
	if facts.Said != "" && facts.Heard != "" {
		match := strings.EqualFold(facts.Said, facts.Heard)
		facts.Match = &match
	}
	return facts
}

// CheckImageRecall reports the facts about a recalled image text.
func CheckImageRecall(written, reference string) RecallFacts {
	truth, _, _ := recallBytes(reference, "image")
	payload, repaired, failure := recallBytes(written, "image")
	facts := recallCommon(payload, truth, repaired, failure, "image")
	if facts.Readable {
		facts.Flat = recallSpread(payload) < recallFlatSpread
	}
	facts.ReferenceFlat = len(truth) > 0 && recallSpread(truth) < recallFlatSpread
	return facts
}

// RecallMark is the mark out of 10 a set of facts earns.
//
// The agreement over the payload is the mark, thinned by whatever the network
// wrote past the end - agreement only asks whether the true bytes came back, so
// without that a completion that rambles after a perfect recall would score full
// marks.  Repairing the base64 costs a point, and a completion that is silent,
// railed or says the wrong words is capped however well its bytes happen to line
// up: those are failures of a different kind from being slightly off.
func RecallMark(facts RecallFacts) float64 {
	if !facts.Readable {
		return 0
	}
	score := 10 * facts.Agreement
	if facts.WrittenBytes > facts.ExpectedBytes && facts.ExpectedBytes > 0 {
		score *= float64(facts.ExpectedBytes) / float64(facts.WrittenBytes) // the excess is waste
	}
	if facts.Repaired {
		score--
	}
	if facts.Flat && !facts.ReferenceFlat {
		score = math.Min(score, 2)
	}
	if facts.Extreme && !facts.ReferenceExtreme {
		score = math.Min(score, 3)
	}
	if facts.Match != nil && !*facts.Match {
		score = math.Min(score, 4)
	}
	return math.Max(0, math.Min(10, score))
}

// RecallReason names the single worst thing wrong with a recalled waveform or
// image.  The order is the order the faults matter in: a completion that cannot
// be read at all is not also judged on its length, and one that stopped early is
// not blamed for the base64 it never got to.  "none" comes back when nothing is
// wrong with it.
func RecallReason(facts RecallFacts) string {
	speech := facts.Modality != "image"
	if !facts.Readable {
		return "unreadable"
	}
	if facts.LengthRatio < RecallTruncated {
		return "truncated"
	}
	if facts.LengthRatio > RecallOverrun {
		return "overrun"
	}
	if facts.Repaired {
		return "garbled"
	}
	if facts.Flat && !facts.ReferenceFlat {
		if speech {
			return "silence"
		}
		return "blank"
	}
	if facts.Extreme && !facts.ReferenceExtreme {
		if speech {
			return "clipping"
		}
		return "noise"
	}
	if facts.Match != nil && !*facts.Match {
		return "mishearing"
	}
	if facts.Agreement < RecallAgreement {
		if speech {
			return "distortion"
		}
		return "drift"
	}
	return "none"
}

// SpeechRecallReasons and ImageRecallReasons are the two vocabularies.
var (
	SpeechRecallReasons = []string{"unreadable", "truncated", "overrun", "garbled", "silence", "clipping",
		"mishearing", "distortion"}
	ImageRecallReasons = []string{"unreadable", "truncated", "overrun", "garbled", "blank", "noise", "drift"}
)

// recallComment is one sentence of teaching, in the plain words the journal keeps.
func recallComment(facts RecallFacts, reason string, score float64) string {
	thing := "waveform"
	if facts.Modality == "image" {
		thing = "image"
	}
	switch reason {
	case "unreadable":
		failure := facts.Error
		if failure == "" {
			failure = "the header is gone"
		}
		return fmt.Sprintf("the completion is not an encoded %s: %s", thing, failure)
	case "truncated":
		return fmt.Sprintf("the %s stops after %d of %d bytes (%.0f%% of it)", thing,
			facts.WrittenBytes, facts.ExpectedBytes, facts.LengthRatio*100)
	case "overrun":
		return fmt.Sprintf("the %s runs on to %d bytes where %d were wanted", thing,
			facts.WrittenBytes, facts.ExpectedBytes)
	case "garbled":
		return fmt.Sprintf("the base64 had to be repaired before the %s could be read", thing)
	case "silence":
		return "it decodes to silence: nothing was said back"
	case "blank":
		return "it decodes to a flat image: nothing was drawn"
	case "clipping":
		return "the waveform is railed against the limits rather than shaped"
	case "noise":
		return "the image is all extremes: noise rather than a picture"
	case "mishearing":
		return fmt.Sprintf("it says %s where %s was said", pythonRepr(facts.Heard), pythonRepr(facts.Said))
	case "distortion", "drift":
		return fmt.Sprintf("the %s is the right shape but wrong: %.0f%% agreement with the original",
			thing, facts.Agreement*100)
	}
	return fmt.Sprintf("recalled at %.1f/10", score)
}

// RecallGrade is the marking of one recalled text: the shape the English tutor's
// grade has.
type RecallGrade struct {
	Score      float64     `json:"score"`
	Passed     bool        `json:"passed"`
	Error      string      `json:"error"`
	Correction string      `json:"correction"`
	Comment    string      `json:"comment"`
	GradedBy   string      `json:"graded_by"`
	Facts      RecallFacts `json:"facts"`
}

// GradeRecall turns facts into a grade: the mark, the reason, the correction and
// one sentence of teaching.  A pass carries no reason, no correction and no
// comment - exactly like the English tutor's grade.
func GradeRecall(facts RecallFacts, correction string, threshold float64) RecallGrade {
	score := RecallMark(facts)
	passed := score >= threshold
	grade := RecallGrade{Score: score, Passed: passed, Error: "none", GradedBy: "recall", Facts: facts}
	if !passed {
		grade.Error = RecallReason(facts)
		grade.Correction = correction
		grade.Comment = recallComment(facts, grade.Error, score)
	}
	return grade
}

// RecallLesson is one exercise, one completion by the network, one grade.
type RecallLesson struct {
	Exercise RecallExercise `json:"exercise"`
	Attempt  int            `json:"attempt"`
	Mode     string         `json:"mode"`
	Text     string         `json:"text"` // cue + continuation: what was marked
	Chars    int            `json:"continuation_chars"`
	Cost     float64        `json:"cost"`
	Seconds  float64        `json:"seconds"`
	Grade    RecallGrade    `json:"grade"`
}

// RecallOptions steer one quiz.
type RecallOptions struct {
	// Lead is the payload characters given away; -1 means the modality's default.
	Lead        int
	Labels      []string
	Attempts    int
	Mode        string
	Temperature float64
	// Length caps how many payload characters are asked for (0: all of it) and
	// trims the reference to match, so a capped quiz is a fair one.
	Length    int
	Threshold float64
	// Said are the words each utterance actually says, by input position.
	Said     []string
	Progress func(RecallLesson)
	Stop     func() bool
}

// DefaultRecallOptions mirror the Python defaults.
func DefaultRecallOptions() RecallOptions {
	return RecallOptions{Lead: -1, Attempts: 1, Mode: "beam", Temperature: 1, Threshold: 6}
}

// Quiz asks the network to write out texts it was taught, and marks what comes
// back.
func Quiz(model *Model, texts []string, o RecallOptions) ([]RecallLesson, error) {
	if model == nil {
		return nil, fmt.Errorf("a model to ask is required")
	}
	if o.Attempts < 1 {
		o.Attempts = 1
	}
	if o.Mode == "" {
		o.Mode = "beam"
	}
	items := RecallExercises(texts, o.Lead, o.Labels)
	lessons := []RecallLesson{}
	for _, item := range items {
		reference := item.Reference
		wanted := len(reference) - len(item.Cue)
		if o.Length > 0 && o.Length < wanted {
			wanted = o.Length
			reference = item.Cue + item.Answer()[:wanted] // a capped quiz is marked against what it asked for
		}
		budget := int(float64(wanted)*RecallOverrun) + 8
		if budget < 1 {
			budget = 1
		}
		for attempt := 0; attempt < o.Attempts; attempt++ {
			if o.Stop != nil && o.Stop() {
				return lessons, nil
			}
			mode := o.Mode
			if attempt > 0 {
				mode = "sample"
			}
			started := time.Now()
			result, err := model.Predict(item.Cue, PredictOptions{
				Length: budget, Mode: mode, K: 1, Temperature: o.Temperature, MaxLength: budget,
			})
			if err != nil {
				return lessons, err
			}
			written := item.Cue + result.Text
			var facts RecallFacts
			if item.Modality == "speech" {
				said := ""
				if item.Index < len(o.Said) {
					said = o.Said[item.Index]
				}
				facts = CheckSpeechRecall(written, reference, said, "")
			} else {
				facts = CheckImageRecall(written, reference)
			}
			lesson := RecallLesson{
				Exercise: item, Attempt: attempt, Mode: mode, Text: written, Chars: len(result.Text),
				Cost: result.Cost, Seconds: time.Since(started).Seconds(),
				Grade: GradeRecall(facts, reference, o.Threshold),
			}
			lessons = append(lessons, lesson)
			if o.Progress != nil {
				o.Progress(lesson)
			}
			if lesson.Grade.Passed {
				break // it remembered: no need to ask again
			}
		}
	}
	return lessons, nil
}

// RecallReportCard is what a quiz came to.
func RecallReportCard(lessons []RecallLesson) map[string]any {
	reasons := map[string]int{}
	passed, scores, agreements := 0, 0.0, 0.0
	kinds := map[string]bool{}
	for _, lesson := range lessons {
		kinds[lesson.Exercise.Modality] = true
		scores += lesson.Grade.Score
		agreements += lesson.Grade.Facts.Agreement
		if lesson.Grade.Passed {
			passed++
		} else if lesson.Grade.Error != "none" {
			reasons[lesson.Grade.Error]++
		}
	}
	modality := ""
	if len(kinds) == 1 {
		for name := range kinds {
			modality = name
		}
	} else if len(kinds) > 1 {
		modality = "mixed"
	}
	card := map[string]any{
		"lessons": len(lessons), "passed": passed, "reasons": reasons, "modality": modality,
		"mean_score": nil, "mean_agreement": 0.0,
	}
	if len(lessons) > 0 {
		mean := scores / float64(len(lessons))
		card["mean_score"] = &mean
		card["mean_agreement"] = agreements / float64(len(lessons))
	}
	return card
}

// FaultsFromRecall turns a round of recall lessons into faults and the texts
// that clear blame.
//
// The recall tutor needs no LLM: it asked the network to write out an utterance
// or a picture it had been taught, so the correct text is on file and rides
// along in the fault as the correction.  Only the characters the network got
// wrong are therefore blamed, and a payload it remembered exactly clears blame.
func FaultsFromRecall(lessons []RecallLesson, threshold float64, source string) ([]Fault, []string) {
	faults := []Fault{}
	passed := []string{}
	seen := map[string]bool{}
	add := func(text string) {
		if text == "" || seen[text] {
			return
		}
		seen[text] = true
		passed = append(passed, text)
	}
	for _, lesson := range lessons {
		if lesson.Text == "" {
			continue
		}
		if lesson.Grade.Passed {
			add(lesson.Text)
			continue
		}
		reason := strings.ToLower(strings.TrimSpace(lesson.Grade.Error))
		if reason == "" || reason == "none" {
			reason = RecallReason(lesson.Grade.Facts)
		}
		if reason == "none" {
			reason = DefaultReason
		}
		score := lesson.Grade.Score
		fault := Fault{
			Text: lesson.Text, Reason: reason, Severity: SeverityFromRating(&score, threshold),
			Note: lesson.Grade.Comment, Source: source,
		}
		if lesson.Grade.Correction != "" && lesson.Grade.Correction != lesson.Text {
			fault.Correction = lesson.Grade.Correction
		}
		faults = append(faults, fault)
		add(lesson.Grade.Correction) // the original is correct by construction
	}
	return faults, passed
}

// TeachRecall feeds a round of the speech / image recall tutor into the negative
// network.
func TeachRecall(negative *Model, lessons []RecallLesson, threshold float64, clearPasses bool, source string,
	o TeachOptions) (*TeachReport, error) {
	if source == "" {
		source = "recall"
	}
	if threshold <= 0 {
		threshold = 6
	}
	faults, passed := FaultsFromRecall(lessons, threshold, source)
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
