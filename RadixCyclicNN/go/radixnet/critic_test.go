package radixnet

import (
	"encoding/json"
	"fmt"
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
