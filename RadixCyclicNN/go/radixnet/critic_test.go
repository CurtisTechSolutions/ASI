package radixnet

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
)

// scriptedReviewer marks from a script and records every prompt it was given.
type scriptedReviewer struct {
	ratingOf func(text string) float64
	critique string
	prompts  []string
}

func (s *scriptedReviewer) Provider() string  { return ProviderOllama }
func (s *scriptedReviewer) BaseURL() string   { return "http://scripted" }
func (s *scriptedReviewer) ModelName() string { return "scripted" }
func (s *scriptedReviewer) Available() bool   { return true }

func (s *scriptedReviewer) Models() ([]map[string]any, error) { return nil, nil }

func (s *scriptedReviewer) Generate(prompt string, o LLMOptions) (string, error) {
	s.prompts = append(s.prompts, prompt)
	type entry struct {
		Index    int     `json:"index"`
		Rating   float64 `json:"rating"`
		Critique string  `json:"critique"`
	}
	out := []entry{}
	for _, line := range strings.Split(prompt, "\n") {
		if !strings.HasPrefix(line, "[") {
			continue
		}
		close := strings.Index(line, "]")
		if close < 0 {
			continue
		}
		index, err := strconv.Atoi(line[1:close])
		if err != nil {
			continue
		}
		text := strings.TrimPrefix(line[close+1:], " ")
		out = append(out, entry{Index: index, Rating: s.ratingOf(text), Critique: s.critique})
	}
	body, _ := json.Marshal(map[string]any{"reviews": out})
	return string(body), nil
}

func failEverything() *scriptedReviewer {
	return &scriptedReviewer{ratingOf: func(string) float64 { return 2 }, critique: "it repeats the same word over and over"}
}

func criticModels(t *testing.T) (*Model, *Model) {
	t.Helper()
	positive, err := NewModel(5, DefaultGraphOptions())
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	positive.Exact = true
	if _, err := positive.Train([]string{
		"good sentences about the cat", "good sentences about the dog", "xxxx xxxx xxxx xxxx",
	}, TrainOptions{Epochs: 4}); err != nil {
		t.Fatalf("train: %v", err)
	}
	negative, err := NewNegativeModel(5, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	return positive, negative
}

func newTestCritic(t *testing.T, reviewer LLMClient, config CriticConfig) *Critic {
	t.Helper()
	positive, negative := criticModels(t)
	loop, err := NewCritic(positive, negative, reviewer, config)
	if err != nil {
		t.Fatalf("NewCritic: %v", err)
	}
	return loop
}

func smallConfig(rounds int) CriticConfig {
	config := DefaultCriticConfig()
	config.Rounds, config.Count, config.MaxLength = rounds, 4, 24
	seed := int64(1)
	config.Seed = &seed
	return config
}

func TestCriticConfigDefaultsAndValidation(t *testing.T) {
	config := DefaultCriticConfig()
	if config.Provider != ProviderOllama || config.Rounds != 3 || !config.ClearPasses {
		t.Fatalf("unexpected defaults: %+v", config)
	}
	if err := config.Validate(); err != nil {
		t.Fatalf("the defaults must validate: %v", err)
	}
	for name, broken := range map[string]CriticConfig{
		"rounds":      {Rounds: -1, Count: 1, Threshold: 6, Provider: "ollama"},
		"count":       {Count: 0, Threshold: 6, Provider: "ollama"},
		"max_length":  {Count: 1, MaxLength: -1, Threshold: 6, Provider: "ollama"},
		"temperature": {Count: 1, Temperature: -1, Threshold: 6, Provider: "ollama"},
		"threshold":   {Count: 1, Threshold: 11, Provider: "ollama"},
		"epochs":      {Count: 1, Threshold: 6, Epochs: -1, Provider: "ollama"},
		"provider":    {Count: 1, Threshold: 6, Provider: "gemini"},
	} {
		if err := broken.Validate(); err == nil {
			t.Errorf("%s: expected the configuration to be refused", name)
		}
	}
}

func TestCriticNeedsBothModels(t *testing.T) {
	positive, negative := criticModels(t)
	if _, err := NewCritic(nil, negative, failEverything(), smallConfig(1)); err == nil {
		t.Error("a loop without a positive model must be refused")
	}
	if _, err := NewCritic(positive, nil, failEverything(), smallConfig(1)); err == nil {
		t.Error("a loop without a negative network must be refused")
	}
}

func TestCriticRoundWritesReviewsAndBlames(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(1))
	record, err := loop.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if record["kind"] != "round" || record["round"] != 1 {
		t.Fatalf("unexpected record: %v", record)
	}
	if record["texts"] != 4 || record["failed"] != 4 || record["blamed"] != 4 {
		t.Fatalf("the scripted reviewer fails everything: %v", record)
	}
	if intOf(record["edges"]) == 0 {
		t.Error("blaming four texts must touch edges")
	}
	if got := record["reasons"].(map[string]int)["repetition"]; got != 4 {
		t.Errorf("the critique picks the reason: got %v", record["reasons"])
	}
	if loop.Negative.G.Neg.TotalBlame <= 0 {
		t.Error("the negative network learned nothing")
	}
}

func TestCriticBlameIsSourcedToTheLoop(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(1))
	if _, err := loop.RunRound(); err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	entries := loop.Negative.Recent(1)
	if len(entries) != 1 || entries[0].Source != "critic" {
		t.Fatalf("the journal must name the loop: %v", entries)
	}
	if !strings.Contains(entries[0].Note, "repeats") {
		t.Errorf("the critique is kept verbatim: %q", entries[0].Note)
	}
	sources, _ := loop.Negative.Stats()["sources"].(map[string]any)
	if got := intOf(sources["critic"]); got != 4 {
		t.Errorf("sources: got %v", loop.Negative.Stats()["sources"])
	}
}

func TestCriticTheMarkSetsTheSeverity(t *testing.T) {
	hard := newTestCritic(t, &scriptedReviewer{ratingOf: func(string) float64 { return 0 }, critique: "gibberish"}, smallConfig(1))
	soft := newTestCritic(t, &scriptedReviewer{ratingOf: func(string) float64 { return 5 }, critique: "gibberish"}, smallConfig(1))
	first, err := hard.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	second, err := soft.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if first["severity_mean"].(float64) <= second["severity_mean"].(float64) {
		t.Errorf("a worse mark must be blamed harder: %v vs %v", first["severity_mean"], second["severity_mean"])
	}
}

func TestCriticWhatItPassesIsNotBlamed(t *testing.T) {
	loop := newTestCritic(t, &scriptedReviewer{ratingOf: func(string) float64 { return 9 }, critique: "reads well"}, smallConfig(1))
	record, err := loop.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if record["failed"] != 0 || record["blamed"] != 0 {
		t.Fatalf("nothing failed, so nothing may be blamed: %v", record)
	}
	if loop.Negative.G.Neg.TotalBlame != 0 {
		t.Error("the negative network must stay empty")
	}
}

func TestCriticRunReturnsRoundsAndOneReport(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(2))
	records, err := loop.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	kinds := []string{}
	for _, record := range records {
		kinds = append(kinds, record["kind"].(string))
	}
	if strings.Join(kinds, ",") != "round,round,report" {
		t.Fatalf("unexpected records: %v", kinds)
	}
	card := records[len(records)-1]
	if card["rounds"] != 2 || card["reviewed"] != 8 || card["blamed"] != 8 {
		t.Fatalf("report card: %v", card)
	}
	if card["stats"] == nil {
		t.Error("the card carries the negative network's stats")
	}
}

func TestCriticProgressSeesEveryRecord(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(2))
	seen := []string{}
	loop.Progress = func(record map[string]any) { seen = append(seen, record["kind"].(string)) }
	if _, err := loop.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Join(seen, ",") != "round,round,report" {
		t.Fatalf("progress saw %v", seen)
	}
}

func TestCriticStopsBetweenRoundsNotInsideOne(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(0)) // forever
	rounds := 0
	loop.Progress = func(record map[string]any) {
		if record["kind"] == "round" {
			rounds++
		}
	}
	loop.Stop = func() bool { return rounds >= 1 }
	records, err := loop.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rounds != 1 {
		t.Fatalf("the loop must finish the round it is in and stop: ran %d", rounds)
	}
	if records[len(records)-1]["kind"] != "report" {
		t.Error("a stopped run still ends with its report")
	}
}

func TestCriticStoppedBeforeItStartsStillReports(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(5))
	loop.Stop = func() bool { return true }
	records, err := loop.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(records) != 1 || records[0]["kind"] != "report" || records[0]["rounds"] != 0 {
		t.Fatalf("unexpected records: %v", records)
	}
}

func TestCriticRoundsZeroMeansUntilStopped(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(0))
	rounds := 0
	loop.Progress = func(record map[string]any) {
		if record["kind"] == "round" {
			rounds++
		}
	}
	loop.Stop = func() bool { return rounds >= 3 }
	if _, err := loop.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rounds != 3 {
		t.Fatalf("expected three rounds, got %d", rounds)
	}
}

func TestCriticSeedAdvancesSoRoundsDiffer(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(3))
	first := *loop.seed()
	loop.RoundNo = 2
	if second := *loop.seed(); second == first {
		t.Error("the seed must advance with the round")
	}
	loop.Config.Seed = nil
	if loop.seed() != nil {
		t.Error("an unseeded loop leaves the sampling alone")
	}
}

func TestCriticLeavesThePositiveModelAlone(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(2))
	before := fmt.Sprintf("%d/%d/%v", loop.Model.G.NumNodes(), loop.Model.G.NumEdges(), loop.Model.Meta["epochs_total"])
	if _, err := loop.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	after := fmt.Sprintf("%d/%d/%v", loop.Model.G.NumNodes(), loop.Model.G.NumEdges(), loop.Model.Meta["epochs_total"])
	if before != after {
		t.Errorf("the loop only reads the positive model: %s -> %s", before, after)
	}
}

func TestCriticTellsTheReviewerTheContext(t *testing.T) {
	reviewer := failEverything()
	config := smallConfig(1)
	config.Context = "plain English about cats"
	loop := newTestCritic(t, reviewer, config)
	if _, err := loop.RunRound(); err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if len(reviewer.prompts) == 0 || !strings.Contains(reviewer.prompts[0], "plain English about cats") {
		t.Errorf("the context must reach the reviewer: %v", reviewer.prompts)
	}
}

func TestCriticExternalWrapsTheSlowCall(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(1))
	order := []string{}
	loop.External = func(fn func() error) error {
		order = append(order, "in")
		err := fn()
		order = append(order, "out")
		return err
	}
	if _, err := loop.RunRound(); err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if strings.Join(order, ",") != "in,out" {
		t.Errorf("External must wrap the review: %v", order)
	}
}

func TestCriticReportCardEmptyAndTrend(t *testing.T) {
	card := CriticReportCard(nil)
	if card["rounds"] != 0 || card["mean_rating"] != (*float64)(nil) || card["trend"] != nil || card["stats"] != nil {
		t.Fatalf("an empty run: %v", card)
	}
	low, high := 3.0, 7.0
	rounds := []map[string]any{
		{"kind": "round", "texts": 2, "mean_rating": &low, "blamed": 2, "cleared": 0, "edges": 4,
			"reasons": map[string]int{"a": 2}, "stats": map[string]any{}},
		{"kind": "round", "texts": 2, "mean_rating": &high, "blamed": 0, "cleared": 2, "edges": 0,
			"reasons": map[string]int{}, "stats": map[string]any{"x": 1}},
	}
	card = CriticReportCard(rounds)
	if card["trend"].(float64) != 4 {
		t.Errorf("trend: %v", card["trend"])
	}
	if *card["mean_rating"].(*float64) != 5 || card["reviewed"] != 4 {
		t.Errorf("card: %v", card)
	}
	if card["reasons"].(map[string]int)["a"] != 2 {
		t.Errorf("reasons: %v", card["reasons"])
	}
}

func TestCriticReportCardOneRoundHasNoTrend(t *testing.T) {
	rating := 5.0
	card := CriticReportCard([]map[string]any{{"kind": "round", "mean_rating": &rating}})
	if card["trend"] != nil {
		t.Errorf("one round cannot have a trend: %v", card["trend"])
	}
}

// -- the copy editor ------------------------------------------------------------

// scriptedEditor copy-edits from a script - howe -> how, ?? -> ?, a comma after
// an opening Hi - and records every prompt and option it was given.
type scriptedEditor struct {
	edit    func(text string) (string, string) // the correction and the editor's word for it
	prompts []string
	options []LLMOptions
}

// fakeCopyEdit is the fake editor's whole craft (the same as the Python tests').
func fakeCopyEdit(text string) (string, string) {
	corrected, reason := text, "none"
	if strings.Contains(corrected, "howe") {
		corrected, reason = strings.ReplaceAll(corrected, "howe", "how"), "spelling"
	}
	for strings.Contains(corrected, "??") {
		corrected = strings.ReplaceAll(corrected, "??", "?")
		if reason == "none" {
			reason = "punctuation"
		}
	}
	if strings.HasPrefix(corrected, "Hi ") && !strings.HasPrefix(corrected, "Hi, ") {
		corrected = "Hi, " + corrected[3:]
		if reason == "none" {
			reason = "punctuation"
		}
	}
	return corrected, reason
}

func (s *scriptedEditor) Provider() string  { return ProviderOllama }
func (s *scriptedEditor) BaseURL() string   { return "http://scripted" }
func (s *scriptedEditor) ModelName() string { return "editor" }
func (s *scriptedEditor) Available() bool   { return true }

func (s *scriptedEditor) Models() ([]map[string]any, error) { return nil, nil }

func (s *scriptedEditor) Generate(prompt string, o LLMOptions) (string, error) {
	s.prompts = append(s.prompts, prompt)
	s.options = append(s.options, o)
	type entry struct {
		Index      int    `json:"index"`
		Correction string `json:"correction"`
		Reason     string `json:"reason"`
		Note       string `json:"note"`
	}
	out := []entry{}
	for _, line := range strings.Split(prompt, "\n") {
		if !strings.HasPrefix(line, "[") {
			continue
		}
		close := strings.Index(line, "]")
		if close < 0 {
			continue
		}
		index, err := strconv.Atoi(line[1:close])
		if err != nil {
			continue
		}
		edit := s.edit
		if edit == nil {
			edit = fakeCopyEdit
		}
		corrected, reason := edit(strings.TrimPrefix(line[close+1:], " "))
		note := "nothing"
		if reason != "none" {
			note = reason + " fixed"
		}
		out = append(out, entry{index, corrected, reason, note})
	}
	body, _ := json.Marshal(map[string]any{"corrections": out})
	return string(body), nil
}

func TestParseCorrectionsReadsEveryShape(t *testing.T) {
	parsed := parseCorrections(`{"corrections": [{"index": 1, "correction": "\"the cat\" ", "reason": " Spelling ",
		"note": "one  word\nwas wrong"}, {"index": 0, "corrected": "hello", "error": "typo", "comment": "a slip"},
		{"index": 7, "correction": "out of range"}, {"index": 1, "correction": "second answer for one"}]`, 3)
	if len(parsed) != 2 {
		t.Fatalf("two entries in range, the first answer per index: %+v", parsed)
	}
	if got := parsed[1]; got.correction != "the cat" || got.reason != "spelling" || got.note != "one word was wrong" {
		t.Fatalf("quotes and spaces are trimmed, the reason lowered, the note collapsed: %+v", got)
	}
	if got := parsed[0]; got.correction != "hello" || got.reason != "typo" || got.note != "a slip" {
		t.Fatalf("corrected / error / comment stand in: %+v", got)
	}
	if got := parseCorrections(`["one", "two"]`, 2); got[0].correction != "one" || got[1].correction != "two" {
		t.Fatalf("a bare list of corrected lines is read by position: %+v", got)
	}
	if got := parseCorrections("```json\n{\"correction\": \"just one\", \"reason\": \"none\"}\n```", 1); got[0].correction != "just one" {
		t.Fatalf("one object for one text, fences and all: %+v", got)
	}
	if got := parseCorrections(`{"results": [{"text": "by text"}]}`, 1); got[0].correction != "by text" {
		t.Fatalf("results / text: %+v", got)
	}
	for _, raw := range []string{"no json here", `{"corrections": "not a list"}`, `[{"index": 0, "correction": 5}]`, ""} {
		if got := parseCorrections(raw, 2); len(got) != 0 {
			t.Fatalf("%q: nothing understood, got %+v", raw, got)
		}
	}
}

func TestCorrectionEntryVerdictsReasonsAndNotes(t *testing.T) {
	same := "the cat sat"
	if entry := correctionEntry(0, same, &same, "none", ""); entry.Verdict != "unchanged" || entry.Reason != "none" ||
		entry.Note != "nothing" || entry.Edits != 0 || entry.Correction == nil || len(entry.Changes) != 0 {
		t.Fatalf("an unchanged text: %+v", entry)
	}
	if entry := correctionEntry(0, same, &same, "", "fine as it is"); entry.Note != "fine as it is" {
		t.Fatalf("the editor's note is kept: %+v", entry)
	}
	fixed := "the cat sat."
	entry := correctionEntry(2, same, &fixed, "", "")
	if entry.Verdict != "corrected" || entry.Note != "corrected" || entry.Index != 2 || entry.Edits != 1 {
		t.Fatalf("a corrected text: %+v", entry)
	}
	if entry.Reason != "punctuation" {
		t.Fatalf("no word from the editor: the diff decides: %+v", entry)
	}
	if change := entry.Changes[0]; change.Op != "insert" || change.Right != "." || change.At != [2]int{11, 11} || change.To != [2]int{11, 12} {
		t.Fatalf("the change carries its spans: %+v", change)
	}
	if entry.WrongChars != 0 || entry.RightChars != 1 {
		t.Fatalf("the characters on either side: %+v", entry)
	}
	if entry := correctionEntry(1, same, nil, "", ""); entry.Verdict != "uncorrected" || entry.Note != "no correction returned" ||
		entry.Correction != nil || entry.Reason != "" || len(entry.Changes) != 0 {
		t.Fatalf("no usable answer: %+v", entry)
	}
	if entry := correctionEntry(1, "  ", nil, "", "empty output"); entry.Note != "empty output" {
		t.Fatalf("a blank text says so: %+v", entry)
	}
}

func TestCorrectTextsAsksAndDiffs(t *testing.T) {
	editor := &scriptedEditor{}
	texts := []string{"Hi howe are you??", "the cat sat on the mat", "howe??", "   "}
	entries, err := CorrectTexts(editor, texts, "short greetings", "", DefaultCorrectionBatch)
	if err != nil {
		t.Fatalf("CorrectTexts: %v", err)
	}
	if len(editor.prompts) != 1 {
		t.Fatalf("one call for one batch: %d", len(editor.prompts))
	}
	want := "Context: short greetings\n\nCorrect these 3 texts:\n[0] Hi howe are you??\n[1] the cat sat on the mat\n" +
		"[2] howe??\n\nReturn the JSON now."
	if editor.prompts[0] != want {
		t.Fatalf("the prompt, byte for byte:\n%q\nwant\n%q", editor.prompts[0], want)
	}
	if o := editor.options[0]; !o.JSON || o.Temperature != 0 || !strings.Contains(o.System, "spelling, punctuation") ||
		strings.Contains(o.System, ", none") || !strings.Contains(o.System, "meticulous copy editor") {
		t.Fatalf("JSON mode, temperature 0 and the reason list without 'none': %+v", o)
	}
	verdicts := []string{}
	for _, entry := range entries {
		verdicts = append(verdicts, entry.Verdict)
	}
	if strings.Join(verdicts, ",") != "corrected,unchanged,corrected,uncorrected" {
		t.Fatalf("verdicts: %v", verdicts)
	}
	first := entries[0]
	if *first.Correction != "Hi, how are you?" || first.Reason != "spelling" || first.Note != "spelling fixed" {
		t.Fatalf("the editor's word and note: %+v", first)
	}
	if first.Edits != len(first.Changes) || first.Edits == 0 {
		t.Fatalf("edits count the changes: %+v", first)
	}
	wrong, right := 0, 0
	for _, change := range first.Changes {
		if change.Op == "equal" {
			t.Fatalf("equal runs are dropped: %+v", change)
		}
		wrong += change.At[1] - change.At[0]
		right += change.To[1] - change.To[0]
	}
	if first.WrongChars != wrong || first.RightChars != right || wrong == 0 {
		t.Fatalf("the characters on either side add up: %+v", first)
	}
	if blank := entries[3]; blank.Correction != nil || blank.Note != "empty output" || blank.Index != 3 {
		t.Fatalf("a blank text is not sent and comes back uncorrected: %+v", blank)
	}
	summary := SummariseCorrections("given", "editor", texts, entries)
	if len(summary.Corrected) != 2 || len(summary.Unchanged) != 1 || len(summary.Uncorrected) != 1 {
		t.Fatalf("the split: %+v", summary)
	}
	if summary.ChangeRate == nil || math.Abs(*summary.ChangeRate-2.0/3.0) > 1e-9 {
		t.Fatalf("two of the three answered texts were changed: %v", summary.ChangeRate)
	}
	if summary.Edits != entries[0].Edits+entries[2].Edits || summary.WrongChars != entries[0].WrongChars+entries[2].WrongChars {
		t.Fatalf("the totals: %+v", summary)
	}
	if empty := SummariseCorrections("given", "editor", nil, nil); empty.ChangeRate != nil || len(empty.Texts) != 0 {
		t.Fatalf("nothing answered: no rate: %+v", empty)
	}
	if _, err := CorrectTexts(editor, texts, "", "", 0); err == nil {
		t.Fatal("a batch of zero is refused")
	}
}

func TestCorrectTextsBatchesAndSkipsBlankBatches(t *testing.T) {
	editor := &scriptedEditor{}
	entries, err := CorrectTexts(editor, []string{"a howe", "b howe", " ", "c howe"}, "", "", 2)
	if err != nil {
		t.Fatalf("CorrectTexts: %v", err)
	}
	if len(editor.prompts) != 2 || !strings.HasPrefix(editor.prompts[1], "Correct these 1 texts:\n[1] c howe") {
		t.Fatalf("two batches, the second asking about one text with its index within the batch: %q", editor.prompts)
	}
	if entries[3].Index != 3 || entries[3].Verdict != "corrected" || entries[2].Verdict != "uncorrected" {
		t.Fatalf("indices are global: %+v", entries)
	}
	if _, err := CorrectTexts(editor, []string{" ", ""}, "", "", 20); err != nil || len(editor.prompts) != 2 {
		t.Fatalf("nothing to ask about: no call: %v %d", err, len(editor.prompts))
	}
}

func TestAdversarialCorrectionSamplesOrTakesTexts(t *testing.T) {
	positive, _ := criticModels(t)
	editor := &scriptedEditor{}
	seed := int64(3)
	result, err := AdversarialCorrection(positive, editor, AdversarialCorrectionOptions{Count: 3, MaxLength: 24, Seed: &seed})
	if err != nil {
		t.Fatalf("AdversarialCorrection: %v", err)
	}
	if result.Source != "model" || len(result.Texts) != 3 || result.Model != "editor" {
		t.Fatalf("sampled from the model: %+v", result)
	}
	result, err = AdversarialCorrection(nil, editor, AdversarialCorrectionOptions{Texts: []string{"howe??"}})
	if err != nil || result.Source != "given" || len(result.Corrected) != 1 {
		t.Fatalf("given texts: %v %+v", err, result)
	}
	if _, err := AdversarialCorrection(nil, editor, AdversarialCorrectionOptions{}); err == nil {
		t.Fatal("neither a model nor texts is refused")
	}
}

func correctingConfig(rounds int, severity float64) CriticConfig {
	config := smallConfig(rounds)
	config.Correct, config.Severity = true, severity
	return config
}

func TestCriticCorrectingRoundBlamesOnlyTheDiff(t *testing.T) {
	// an editor that strikes the last character of everything: every text is corrected, by a deletion
	strike := &scriptedEditor{edit: func(text string) (string, string) { return text[:len(text)-1], "fragment" }}
	loop := newTestCritic(t, strike, correctingConfig(1, 2))
	record, err := loop.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if record["mode"] != "correct" || record["severity"] != 2.0 || record["round"] != 1 {
		t.Fatalf("a correcting round says so: %+v", record)
	}
	for _, key := range []string{"corrections", "change_rate", "corrected", "unchanged", "uncorrected", "edits",
		"wrong_chars", "blamed", "cleared", "unmatched", "edges", "reasons", "severity_mean", "stats", "seconds"} {
		if _, ok := record[key]; !ok {
			t.Fatalf("missing %q in %+v", key, record)
		}
	}
	for _, key := range []string{"threshold", "reviews", "mean_rating", "pass_rate", "passed", "failed"} {
		if _, ok := record[key]; ok {
			t.Fatalf("no marks in a correcting round: %q in %+v", key, record)
		}
	}
	if record["corrected"] != 4 || record["blamed"] != 4 || record["unchanged"] != 0 || record["edits"] != 4 {
		t.Fatalf("every text corrected once and blamed: %+v", record)
	}
	if rate := record["change_rate"].(*float64); rate == nil || *rate != 1 {
		t.Fatalf("change rate 1: %v", rate)
	}
	if record["severity_mean"] != 2.0 || record["reasons"].(map[string]int)["fragment"] != 4 {
		t.Fatalf("the editor's word at the configured severity: %+v", record)
	}
	if loop.Negative.G.Neg.TotalBlame <= 0 {
		t.Fatal("the negative network learnt something")
	}
	for _, entry := range loop.Negative.Recent(4) {
		if entry.Source != "critic" || entry.Reason != "fragment" || entry.Severity != 2 {
			t.Fatalf("sourced to the loop: %+v", entry)
		}
	}
	if !strings.HasPrefix(strike.prompts[0], "Correct these 4 texts:") || !strike.options[0].JSON {
		t.Fatalf("the editor was asked, in JSON mode: %q", strike.prompts[0])
	}
}

func TestCriticCorrectingRoundClearsWhatTheEditorLeftAlone(t *testing.T) {
	identity := &scriptedEditor{edit: func(text string) (string, string) { return text, "none" }}
	loop := newTestCritic(t, identity, correctingConfig(1, 1))
	record, err := loop.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if record["corrected"] != 0 || record["blamed"] != 0 || record["unchanged"] != 4 || record["edits"] != 0 {
		t.Fatalf("nothing changed, so nothing blamed: %+v", record)
	}
	if rate := record["change_rate"].(*float64); rate == nil || *rate != 0 {
		t.Fatalf("change rate 0: %v", rate)
	}
	if loop.Negative.G.Neg.TotalBlame != 0 {
		t.Error("the negative network stays empty")
	}
}

func TestCriticReviewRoundSaysItsMode(t *testing.T) {
	loop := newTestCritic(t, failEverything(), smallConfig(1))
	record, err := loop.RunRound()
	if err != nil {
		t.Fatalf("RunRound: %v", err)
	}
	if record["mode"] != "review" {
		t.Fatalf("a review round says so: %+v", record)
	}
}

func TestCriticReportCardForCorrectingRounds(t *testing.T) {
	strike := &scriptedEditor{edit: func(text string) (string, string) { return text[:len(text)-1], "fragment" }}
	loop := newTestCritic(t, strike, correctingConfig(2, 1))
	records, err := loop.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	card := records[len(records)-1]
	if card["kind"] != "report" || card["rounds"] != 2 || card["reviewed"] != 8 || card["corrected"] != 8 {
		t.Fatalf("the card totals the editor's rounds: %+v", card)
	}
	if card["unchanged"] != 0 || card["uncorrected"] != 0 || card["edits"] != 8 || card["blamed"] != 8 {
		t.Fatalf("and their counts: %+v", card)
	}
	if rate := card["change_rate"].(*float64); rate == nil || *rate != 1 {
		t.Fatalf("the mean change rate: %v", rate)
	}
	if trend, ok := card["change_trend"].(float64); !ok || trend != 0 {
		t.Fatalf("the change trend over two rounds: %v", card["change_trend"])
	}
	if card["mean_rating"] != (*float64)(nil) || card["trend"] != nil {
		t.Fatalf("no marks were given: %+v", card)
	}
	// a review run has none of the editor's keys
	review := CriticReportCard([]map[string]any{{"kind": "round", "mode": "review", "texts": 2, "reasons": map[string]int{}}})
	for _, key := range []string{"corrected", "unchanged", "uncorrected", "edits", "change_rate", "change_trend"} {
		if _, ok := review[key]; ok {
			t.Fatalf("%q on a review card: %+v", key, review)
		}
	}
	// one correcting round has a rate but no trend
	one := CriticReportCard(records[:1])
	if one["change_trend"] != nil || one["change_rate"].(*float64) == nil {
		t.Fatalf("one round: %+v", one)
	}
}

func TestCriticConfigRefusesANegativeSeverity(t *testing.T) {
	config := DefaultCriticConfig()
	if config.Severity != CorrectionSeverity || config.Correct {
		t.Fatalf("the defaults: %+v", config)
	}
	config.Severity = -1
	if err := config.Validate(); err == nil {
		t.Fatal("a negative severity must be refused")
	}
}
