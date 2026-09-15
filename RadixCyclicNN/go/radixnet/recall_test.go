package radixnet

import (
	"strings"
	"testing"
)

const recallRate = 8000

// recallUtterance is one short recording as the texts it trains on.
func recallUtterance(t *testing.T, freq float64) (string, string) {
	t.Helper()
	wav := WAVBytes(tone(0.05, recallRate, freq, 0.6), recallRate, 1)
	o := DefaultTeachSpeechOptions()
	o.Transcript, o.Rate = "hello there", recallRate
	taught, err := TeachSpeech(wav, o)
	if err != nil {
		t.Fatalf("TeachSpeech: %v", err)
	}
	for _, text := range taught.Texts {
		if strings.Contains(text, "aud:") {
			return text, taught.Token
		}
	}
	t.Fatal("no waveform text")
	return "", ""
}

// recallPicture is an image text with a repeatable payload (no decoding needed).
func recallPicture() (string, int) {
	n := 3 * 8 * 8
	payload := make([]byte, n)
	for i := range payload {
		payload[i] = byte((i*7 + 11) % 256)
	}
	return PackImageText("tiny", 64, 64, payload), n
}

func TestModalityOf(t *testing.T) {
	text, _ := recallUtterance(t, 220)
	if got, err := ModalityOf(text); err != nil || got != "speech" {
		t.Errorf("speech: %q %v", got, err)
	}
	picture, _ := recallPicture()
	if got, err := ModalityOf(picture); err != nil || got != "image" {
		t.Errorf("image: %q %v", got, err)
	}
	if _, err := ModalityOf("the cat sat on the mat"); err == nil {
		t.Error("prose is neither")
	}
}

func TestRecallCue(t *testing.T) {
	text, token := recallUtterance(t, 220)
	cue, err := RecallCue(text, -1)
	if err != nil {
		t.Fatalf("RecallCue: %v", err)
	}
	if cue != token+" aud:mu:8000x1:" {
		t.Fatalf("a spoken cue is the token and the header: %q", cue)
	}
	if !strings.HasPrefix(text, cue) || strings.Contains(cue, "=") {
		t.Error("the cue gives nothing of the payload away")
	}
	picture, _ := recallPicture()
	cue, _ = RecallCue(picture, -1)
	if len(cue) != len("img:tiny:64x64:")+DefaultRecallLead["image"] {
		t.Errorf("an image cue carries a lead: %q", cue)
	}
	bare, _ := RecallCue(picture, 0)
	if bare != "img:tiny:64x64:" {
		t.Errorf("lead 0: %q", bare)
	}
	short := PackImageText("tiny", 64, 64, []byte{1, 2})
	if got, _ := RecallCue(short, 999); got != short {
		t.Error("the cue never runs past a short text")
	}
}

func TestRecallExercisesSkipWhatIsNotEncoded(t *testing.T) {
	text, _ := recallUtterance(t, 220)
	picture, _ := recallPicture()
	items := RecallExercises([]string{text, "the cat sat on the mat", "", picture},
		-1, []string{"a", "b", "c", "d"})
	if len(items) != 2 {
		t.Fatalf("expected two exercises, got %d", len(items))
	}
	if items[0].Modality != "speech" || items[1].Modality != "image" {
		t.Errorf("modalities: %q %q", items[0].Modality, items[1].Modality)
	}
	// the index is where it came from, not where it ended up
	if items[0].Index != 0 || items[1].Index != 3 {
		t.Errorf("indexes: %d %d", items[0].Index, items[1].Index)
	}
	if items[0].Label != "a" || items[1].Label != "d" {
		t.Errorf("labels: %q %q", items[0].Label, items[1].Label)
	}
	if items[0].Cue+items[0].Answer() != text {
		t.Error("cue + answer is the reference")
	}
}

func TestRecallMarkingSpeech(t *testing.T) {
	text, token := recallUtterance(t, 220)
	_, _, _, truth, _, err := ParseSpeechText(text)
	if err != nil {
		t.Fatalf("ParseSpeechText: %v", err)
	}

	perfect := CheckSpeechRecall(text, text, "", "")
	if RecallMark(perfect) != 10 || RecallReason(perfect) != "none" {
		t.Errorf("a perfect recall: %.2f %q", RecallMark(perfect), RecallReason(perfect))
	}

	prose := CheckSpeechRecall(token+" the cat sat on the mat", text, "", "")
	if prose.Readable || RecallMark(prose) != 0 || RecallReason(prose) != "unreadable" {
		t.Errorf("prose: %+v", prose)
	}

	half := CheckSpeechRecall(text[:len(text)/2], text, "", "")
	if RecallReason(half) != "truncated" || RecallMark(half) >= 6 {
		t.Errorf("half: %q %.2f", RecallReason(half), RecallMark(half))
	}

	long := CheckSpeechRecall(token+" "+PackSpeechText("mu", recallRate, 1, append(append([]byte{}, truth...), truth...)),
		text, "", "")
	if RecallReason(long) != "overrun" {
		t.Errorf("overrun: %q", RecallReason(long))
	}
	if mark := RecallMark(long); mark < 4.9 || mark > 5.1 {
		t.Errorf("perfect bytes, half of them waste: %.2f", mark)
	}

	quiet := MuLawEncode(make([]float32, len(truth)))
	silence := CheckSpeechRecall(token+" "+PackSpeechText("mu", recallRate, 1, quiet), text, "", "")
	if !silence.Flat || silence.ReferenceFlat || RecallReason(silence) != "silence" {
		t.Errorf("silence: %+v", silence)
	}
	if RecallMark(silence) > 2 {
		t.Errorf("silence is capped: %.2f", RecallMark(silence))
	}

	railed := make([]byte, len(truth))
	for i := range railed {
		if (i/8)%2 == 0 {
			railed[i] = 255
		}
	}
	clipping := CheckSpeechRecall(token+" "+PackSpeechText("mu", recallRate, 1, railed), text, "", "")
	if !clipping.Extreme || RecallReason(clipping) != "clipping" {
		t.Errorf("clipping: %+v", clipping)
	}

	wrong := make([]byte, len(truth))
	for i, b := range truth {
		wrong[i] = byte((int(b) + 90) % 256)
	}
	distorted := CheckSpeechRecall(token+" "+PackSpeechText("mu", recallRate, 1, wrong), text, "", "")
	if RecallReason(distorted) != "distortion" {
		t.Errorf("distortion: %q (agreement %.3f)", RecallReason(distorted), distorted.Agreement)
	}
}

func TestRecallMishearing(t *testing.T) {
	text, _ := recallUtterance(t, 220)
	heard := CheckSpeechRecall(text, text, "hello there", "yellow bear")
	if heard.Match == nil || *heard.Match || RecallReason(heard) != "mishearing" {
		t.Errorf("mishearing: %+v", heard)
	}
	if RecallMark(heard) > 4 {
		t.Errorf("a mishearing is capped: %.2f", RecallMark(heard))
	}
	same := CheckSpeechRecall(text, text, "Hello there", "hello  there")
	if same.Match == nil || !*same.Match || RecallReason(same) != "none" {
		t.Errorf("the same words: %+v", same)
	}
	if CheckSpeechRecall(text, text, "", "").Match != nil {
		t.Error("nothing is claimed about the words without a transcript")
	}
}

func TestRecallSilenceIsNotAFaultWhenTheRecordingWasSilent(t *testing.T) {
	quiet := PackSpeechText("mu", recallRate, 1, MuLawEncode(make([]float32, 200)))
	facts := CheckSpeechRecall(quiet, quiet, "", "")
	if !facts.Flat || !facts.ReferenceFlat || RecallReason(facts) != "none" {
		t.Errorf("a silent recording recalled silently is right: %+v", facts)
	}
}

func TestRecallMarkingImages(t *testing.T) {
	text, size := recallPicture()
	_, _, _, truth, _, _ := ParseImageText(text)

	if facts := CheckImageRecall(text, text); RecallMark(facts) != 10 || RecallReason(facts) != "none" {
		t.Errorf("perfect: %.2f %q", RecallMark(facts), RecallReason(facts))
	}
	if facts := CheckImageRecall("the cat sat on the mat", text); RecallReason(facts) != "unreadable" {
		t.Errorf("prose: %q", RecallReason(facts))
	}
	flat := make([]byte, size)
	for i := range flat {
		flat[i] = 128
	}
	if facts := CheckImageRecall(PackImageText("tiny", 64, 64, flat), text); RecallReason(facts) != "blank" {
		t.Errorf("blank, not silence: %q", RecallReason(facts))
	}
	noisy := make([]byte, size)
	for i := range noisy {
		if i%2 == 0 {
			noisy[i] = 255
		}
	}
	if facts := CheckImageRecall(PackImageText("tiny", 64, 64, noisy), text); RecallReason(facts) != "noise" {
		t.Errorf("noise, not clipping: %q", RecallReason(facts))
	}
	wrong := make([]byte, size)
	for i, b := range truth {
		wrong[i] = byte((int(b) + 90) % 256)
	}
	if facts := CheckImageRecall(PackImageText("tiny", 64, 64, wrong), text); RecallReason(facts) != "drift" {
		t.Errorf("drift, not distortion: %q", RecallReason(facts))
	}
	if CheckImageRecall(text, text).Match != nil {
		t.Error("an image never claims to have heard anything")
	}
}

func TestRecallAgreementCurve(t *testing.T) {
	truth := make([]byte, 64)
	for i := range truth {
		truth[i] = byte(i)
	}
	if recallAgreement(truth, truth) != 1 {
		t.Error("an exact payload agrees completely")
	}
	if got := recallAgreement(truth[:32], truth); got < 0.49 || got > 0.51 {
		t.Errorf("bytes never written count as total disagreement: %g", got)
	}
	near := make([]byte, 64)
	for i, b := range truth {
		near[i] = b + 1
	}
	if got := recallAgreement(near, truth); got < 0.95 {
		t.Errorf("a quantisation step barely costs anything: %g", got)
	}
	if got := recallAgreement([]byte{200}, []byte{200 - byte(RecallTolerance)}); got != 0 {
		t.Errorf("a byte past the tolerance agrees not at all: %g", got)
	}
	if recallAgreement(nil, nil) != 1 || recallAgreement([]byte("abc"), nil) != 0 {
		t.Error("an empty truth is matched only by an empty answer")
	}
}

func TestGradeRecall(t *testing.T) {
	text, _ := recallPicture()
	pass := GradeRecall(CheckImageRecall(text, text), text, 6)
	if !pass.Passed || pass.Error != "none" || pass.Correction != "" || pass.Comment != "" {
		t.Errorf("a pass carries no reason, correction or comment: %+v", pass)
	}
	fail := GradeRecall(CheckImageRecall("img:tiny:64x64:AAAA", text), text, 6)
	if fail.Passed || fail.Correction != text || fail.Error != "truncated" {
		t.Errorf("a failure carries the original as its correction: %+v", fail)
	}
	if !strings.Contains(fail.Comment, "stops after") {
		t.Errorf("comment: %q", fail.Comment)
	}
}

func TestQuizMarksWhatTheNetworkWritesBack(t *testing.T) {
	text, _ := recallUtterance(t, 220)
	model, _ := NewModel(7, DefaultGraphOptions())
	model.Exact = true
	if _, err := model.Train([]string{text}, TrainOptions{Epochs: 3, AutoCompress: true}); err != nil {
		t.Fatalf("train: %v", err)
	}
	o := DefaultRecallOptions()
	o.Length = 120
	lessons, err := Quiz(model, []string{text}, o)
	if err != nil {
		t.Fatalf("Quiz: %v", err)
	}
	if len(lessons) != 1 {
		t.Fatalf("expected one lesson, got %d", len(lessons))
	}
	lesson := lessons[0]
	if lesson.Exercise.Modality != "speech" || !strings.HasPrefix(lesson.Text, lesson.Exercise.Cue) {
		t.Errorf("lesson: %+v", lesson.Exercise)
	}
	// 120 base64 characters were asked for, so 90 bytes are expected
	if lesson.Grade.Facts.ExpectedBytes != 90 {
		t.Errorf("a capped quiz is marked against what it asked for: %d", lesson.Grade.Facts.ExpectedBytes)
	}
}

func TestQuizStopsAtTheFirstPass(t *testing.T) {
	text, _ := recallPicture()
	model, _ := NewModel(7, DefaultGraphOptions())
	model.Exact = true
	if _, err := model.Train([]string{text}, TrainOptions{Epochs: 6, AutoCompress: true}); err != nil {
		t.Fatalf("train: %v", err)
	}
	o := DefaultRecallOptions()
	o.Attempts = 3
	lessons, err := Quiz(model, []string{text}, o)
	if err != nil {
		t.Fatalf("Quiz: %v", err)
	}
	if len(lessons) > 3 {
		t.Errorf("attempts are capped: %d", len(lessons))
	}
	for i, lesson := range lessons {
		if lesson.Grade.Passed && i != len(lessons)-1 {
			t.Error("a pass must end the attempts")
		}
	}
}

func TestQuizStopEventAndProgress(t *testing.T) {
	text, _ := recallPicture()
	model, _ := NewModel(7, DefaultGraphOptions())
	o := DefaultRecallOptions()
	o.Stop = func() bool { return true }
	lessons, err := Quiz(model, []string{text}, o)
	if err != nil || len(lessons) != 0 {
		t.Errorf("a stopped quiz runs nothing: %v %v", lessons, err)
	}
	seen := 0
	o = DefaultRecallOptions()
	o.Length, o.Progress = 40, func(RecallLesson) { seen++ }
	if _, err := Quiz(model, []string{text}, o); err != nil {
		t.Fatalf("Quiz: %v", err)
	}
	if seen != 1 {
		t.Errorf("progress saw %d lessons", seen)
	}
	if _, err := Quiz(nil, []string{text}, DefaultRecallOptions()); err == nil {
		t.Error("a quiz without a model must be refused")
	}
}

func TestRecallReportCard(t *testing.T) {
	empty := RecallReportCard(nil)
	if empty["lessons"] != 0 || empty["modality"] != "" || empty["mean_score"] != nil {
		t.Errorf("empty: %v", empty)
	}
	text, _ := recallPicture()
	lessons := []RecallLesson{
		{Exercise: RecallExercise{Modality: "image"}, Grade: GradeRecall(CheckImageRecall(text, text), text, 6)},
		{Exercise: RecallExercise{Modality: "image"}, Grade: GradeRecall(CheckImageRecall("img:tiny:64x64:AAAA", text), text, 6)},
	}
	card := RecallReportCard(lessons)
	if card["lessons"] != 2 || card["passed"] != 1 || card["modality"] != "image" {
		t.Fatalf("card: %v", card)
	}
	if card["reasons"].(map[string]int)["truncated"] != 1 {
		t.Errorf("reasons: %v", card["reasons"])
	}
}

func TestFaultsAndTeachFromRecall(t *testing.T) {
	text, _ := recallPicture()
	_, _, _, truth, _, _ := ParseImageText(text)
	half := len(truth) / 2
	wrong := append(append([]byte{}, truth[:half]...), make([]byte, 0, half)...)
	for _, b := range truth[half:] {
		wrong = append(wrong, byte((int(b)+90)%256))
	}
	drifted := PackImageText("tiny", 64, 64, wrong)
	lessons := []RecallLesson{{
		Exercise: RecallExercise{ID: "r1", Modality: "image", Reference: text, Label: "a.png"},
		Text:     drifted,
		Grade:    GradeRecall(CheckImageRecall(drifted, text), text, 6),
	}}
	faults, passed := FaultsFromRecall(lessons, 6, "recall")
	if len(faults) != 1 || faults[0].Reason != "drift" || faults[0].Correction != text {
		t.Fatalf("faults: %+v", faults)
	}
	if len(passed) != 1 || passed[0] != text {
		t.Errorf("the original is correct by construction: %v", passed)
	}
	negative, _ := NewNegativeModel(7, DefaultNegativeOptions())
	report, err := TeachRecall(negative, lessons, 6, true, "vision", TeachOptions{})
	if err != nil {
		t.Fatalf("TeachRecall: %v", err)
	}
	if report.Blamed != 1 || report.Source != "vision" || report.Edges == 0 {
		t.Fatalf("report: %+v", report)
	}
	if negative.Judge(drifted, JudgeOptions{}).Blame <= 0 {
		t.Error("the drifted text must carry blame")
	}
	// the opening both texts share is not the mistake, so it carries none
	if negative.Judge(text[:len("img:tiny:64x64:")+40], JudgeOptions{}).Blame != 0 {
		t.Error("what it remembered must carry no blame")
	}
}

func TestTeachRecallPassesOnlyClear(t *testing.T) {
	text, _ := recallPicture()
	lessons := []RecallLesson{{
		Exercise: RecallExercise{Modality: "image", Reference: text},
		Text:     text,
		Grade:    GradeRecall(CheckImageRecall(text, text), text, 6),
	}}
	faults, passed := FaultsFromRecall(lessons, 6, "recall")
	if len(faults) != 0 || len(passed) != 1 {
		t.Errorf("a pass only clears: %v %v", faults, passed)
	}
}
