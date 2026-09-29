package server

import (
	"encoding/json"
	"strings"
	"testing"
)

var evolveCorpus = []string{
	"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill",
	"the cat chased the mouse", "rain fell on the quiet town",
}

// evolveEnv is a service trained on the corpus, ready to evolve.
func evolveEnv(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, false)
	if status, doc := e.post("/api/train", map[string]any{"texts": evolveCorpus, "epochs": 3}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitForJob()
	return e
}

func TestEvolveStartRunsGenerations(t *testing.T) {
	e := evolveEnv(t)
	status, doc := e.post("/api/evolve/start", map[string]any{
		"corpus": evolveCorpus, "generations": 2, "samples": 4, "max_length": 24,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	job, _ := doc["job"].(map[string]any)
	if job == nil || job["type"] != "evolve" {
		t.Fatalf("expected an evolve job: %+v", doc)
	}
	if corpus, _ := doc["corpus"].(float64); int(corpus) != len(evolveCorpus) {
		t.Errorf("corpus: %v", doc["corpus"])
	}
	finished := e.waitForJob()
	if finished["state"] != "done" {
		t.Fatalf("job: %+v", finished)
	}
	status, history := e.get("/api/evolve/history")
	if status != 200 {
		t.Fatalf("history: %d", status)
	}
	records, _ := history["history"].([]any)
	if len(records) != 2 {
		t.Fatalf("expected two generations, got %d", len(records))
	}
	first, _ := records[0].(map[string]any)
	for _, key := range []string{"generation", "fake_score_mean", "real_score_mean", "gap", "fakes", "twonrl", "mode"} {
		if _, ok := first[key]; !ok {
			t.Errorf("the record is missing %q", key)
		}
	}
}

func TestEvolveCanBlameTheNegativeNetwork(t *testing.T) {
	e := evolveEnv(t)
	status, doc := e.post("/api/evolve/start", map[string]any{
		"corpus": evolveCorpus, "generations": 2, "samples": 4, "max_length": 24,
		"blatant_mode": "fail_invert", "blame": true,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	e.waitForJob()
	status, tab := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d", status)
	}
	stats, _ := tab["stats"].(map[string]any)
	if failures, _ := stats["failures_total"].(float64); failures > 0 {
		sources, _ := stats["sources"].(map[string]any)
		if sources["evolve"] == nil {
			t.Errorf("the blame must be sourced to the loop: %v", sources)
		}
	}
}

func TestEvolveStopAsksTheJobToStop(t *testing.T) {
	e := evolveEnv(t)
	status, doc := e.post("/api/evolve/start", map[string]any{
		"corpus": evolveCorpus, "generations": 0, "samples": 4, "max_length": 24, // forever
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	status, stopped := e.post("/api/evolve/stop", map[string]any{})
	if status != 200 {
		t.Fatalf("stop: %d %+v", status, stopped)
	}
	finished := e.waitForJob()
	if finished["state"] != "stopped" && finished["state"] != "done" {
		t.Fatalf("job: %+v", finished)
	}
}

func TestEvolveRefusesABadConfig(t *testing.T) {
	e := evolveEnv(t)
	for name, body := range map[string]map[string]any{
		"samples":     {"corpus": evolveCorpus, "samples": 0},
		"mode":        {"corpus": evolveCorpus, "blatant_mode": "sideways"},
		"generations": {"corpus": evolveCorpus, "generations": -1},
		"boost":       {"corpus": evolveCorpus, "blatant_boost": 0.5},
	} {
		status, doc := e.post("/api/evolve/start", body)
		if status != 400 {
			t.Errorf("%s: expected 400, got %d %+v", name, status, doc)
		}
	}
	status, _ := e.post("/api/evolve/start", map[string]any{})
	if status != 400 {
		t.Errorf("a missing corpus must be a 400, got %d", status)
	}
}

func TestEvolveHistoryIsEmptyBeforeItRuns(t *testing.T) {
	e := evolveEnv(t)
	status, doc := e.get("/api/evolve/history")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	if records, _ := doc["history"].([]any); len(records) != 0 {
		t.Errorf("expected no records: %v", records)
	}
}

func TestEvolveRoutesAreDocumented(t *testing.T) {
	e := evolveEnv(t)
	status, doc := e.get("/api")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	raw, _ := json.Marshal(doc)
	for _, path := range []string{"/api/evolve/start", "/api/evolve/stop", "/api/evolve/history"} {
		if !strings.Contains(string(raw), path) {
			t.Errorf("%s is not in the endpoint index", path)
		}
	}
}
