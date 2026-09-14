package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var criticCorpus = []string{
	"good sentences about the cat", "good sentences about the dog", "xxxx xxxx xxxx xxxx",
}

var reviewLine = regexp.MustCompile(`^\[(\d+)\] (.*)$`)

// fakeReviewer answers the review and corpus prompts: anything with "good" in
// it passes, everything else fails for repeating itself.
type fakeReviewer struct {
	server  *httptest.Server
	prompts []string
}

func newFakeReviewer(t *testing.T) *fakeReviewer {
	t.Helper()
	fake := &fakeReviewer{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "fake:latest"}}})
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		prompt, _ := body["prompt"].(string)
		system, _ := body["system"].(string)
		fake.prompts = append(fake.prompts, prompt)
		json.NewEncoder(w).Encode(map[string]any{"response": fake.answer(system, prompt)})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeReviewer) answer(system, prompt string) string {
	if strings.Contains(system, "adversarial reviewer") {
		type entry struct {
			Index    int     `json:"index"`
			Rating   float64 `json:"rating"`
			Critique string  `json:"critique"`
		}
		out := []entry{}
		for _, line := range strings.Split(prompt, "\n") {
			match := reviewLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			index, _ := strconv.Atoi(match[1])
			if strings.Contains(match[2], "good") {
				out = append(out, entry{index, 9, "reads well"})
			} else {
				out = append(out, entry{index, 2, "it repeats the same word over and over"})
			}
		}
		raw, _ := json.Marshal(map[string]any{"reviews": out})
		return string(raw)
	}
	// a corpus request: plain lines
	return "one line about it\ntwo lines about it\nthree lines about it"
}

// criticEnv is a service whose LLM defaults point at the fake reviewer, trained
// on a corpus so it has something to write.
func criticEnv(t *testing.T) (*env, *fakeReviewer) {
	t.Helper()
	fake := newFakeReviewer(t)
	dir := t.TempDir()
	svc, err := NewService(Options{
		ModelPath: filepath.Join(dir, "model.count.json"), Seed: 1, Exact: true, Quiet: true,
		OllamaURL: fake.server.URL, OllamaModel: "fake:latest",
	})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(NewHandler(svc, "", true, nil))
	t.Cleanup(ts.Close)
	e := &env{t: t, ts: ts, svc: svc, dir: dir}
	if status, doc := e.post("/api/train", map[string]any{"texts": criticCorpus, "epochs": 4}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitForJob()
	return e, fake
}

// waitForJob polls until no job is running.
func (e *env) waitForJob() map[string]any {
	e.t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		status, doc := e.get("/api/job")
		if status == 200 && doc != nil {
			if state, _ := doc["state"].(string); state != "" && state != "running" {
				return doc
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	e.t.Fatal("the job did not finish in time")
	return nil
}

func TestNegativeAutoRunsRoundsAndTeaches(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/negative/auto", map[string]any{"rounds": 2, "count": 4, "max_length": 24})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	job, _ := doc["job"].(map[string]any)
	if job == nil || job["type"] != "critic" {
		t.Fatalf("expected a critic job: %+v", doc)
	}
	if doc["reviewer"] != "fake:latest" {
		t.Errorf("reviewer: %v", doc["reviewer"])
	}
	finished := e.waitForJob()
	if finished["state"] != "done" {
		t.Fatalf("job: %+v", finished)
	}
	status, history := e.get("/api/negative/auto/history")
	if status != 200 {
		t.Fatalf("history: %d", status)
	}
	records, _ := history["history"].([]any)
	kinds := []string{}
	for _, record := range records {
		row, _ := record.(map[string]any)
		kinds = append(kinds, row["kind"].(string))
	}
	if strings.Join(kinds, ",") != "round,round,report" {
		t.Fatalf("records: %v", kinds)
	}
	// the tab's own endpoint sees what the loop taught, with nobody typing anything
	status, negative := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d", status)
	}
	stats, _ := negative["stats"].(map[string]any)
	if failures, _ := stats["failures_total"].(float64); failures <= 0 {
		t.Fatalf("the loop taught nothing: %+v", stats)
	}
	sources, _ := stats["sources"].(map[string]any)
	if sources["critic"] == nil {
		t.Errorf("the blame must be sourced to the loop: %v", sources)
	}
}

func TestNegativeAutoRefusesABadConfig(t *testing.T) {
	e, _ := criticEnv(t)
	for name, body := range map[string]map[string]any{
		"count":     {"count": 0},
		"rounds":    {"rounds": -1},
		"provider":  {"provider": "gemini"},
		"threshold": {"threshold": 11},
	} {
		status, doc := e.post("/api/negative/auto", body)
		if status != 400 {
			t.Errorf("%s: expected 400, got %d %+v", name, status, doc)
		}
	}
}

func TestNegativeAutoHistoryIsEmptyBeforeItRuns(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.get("/api/negative/auto/history")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	records, _ := doc["history"].([]any)
	if len(records) != 0 {
		t.Errorf("expected no records, got %v", records)
	}
}

func TestOllamaModels(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.get("/api/ollama/models")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	if doc["available"] != true {
		t.Fatalf("expected the fake to answer: %+v", doc)
	}
	models, _ := doc["models"].([]any)
	if len(models) != 1 {
		t.Fatalf("models: %+v", doc["models"])
	}
}

func TestOllamaModelsNeverFails(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.get("/api/ollama/models?url=http://127.0.0.1:1")
	if status != 200 {
		t.Fatalf("a dead endpoint must still answer 200: %d %+v", status, doc)
	}
	if doc["available"] != false || doc["error"] == nil {
		t.Errorf("expected available=false with an error: %+v", doc)
	}
}

func TestOllamaCorpus(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/corpus", map[string]any{"prompt": "cats", "lines": 3})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	texts, _ := doc["texts"].([]any)
	if len(texts) != 3 {
		t.Fatalf("texts: %+v", doc["texts"])
	}
	if doc["style"] != "good" || doc["job"] != nil {
		t.Errorf("unexpected: %+v", doc)
	}
}

func TestOllamaCorpusCanTrain(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/corpus", map[string]any{"prompt": "cats", "lines": 3, "train": true, "epochs": 1})
	if status != 202 {
		t.Fatalf("status %d %+v", status, doc)
	}
	if doc["job"] == nil {
		t.Fatal("expected a training job")
	}
	e.waitForJob()
}

func TestOllamaCorpusRefusesABadStyle(t *testing.T) {
	e, _ := criticEnv(t)
	status, _ := e.post("/api/ollama/corpus", map[string]any{"prompt": "cats", "style": "sideways"})
	if status != 400 {
		t.Errorf("expected 400, got %d", status)
	}
	status, _ = e.post("/api/ollama/corpus", map[string]any{})
	if status != 400 {
		t.Errorf("a missing prompt must be a 400, got %d", status)
	}
}

func TestOllamaReviewMarksGivenTexts(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/review", map[string]any{
		"texts": []string{"good sentences about the cat", "xxxx xxxx xxxx"},
	})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	if doc["source"] != "given" {
		t.Errorf("source: %v", doc["source"])
	}
	good, _ := doc["good"].([]any)
	bad, _ := doc["bad"].([]any)
	if len(good) != 1 || len(bad) != 1 {
		t.Fatalf("expected one of each: good=%v bad=%v", good, bad)
	}
	if doc["negative"] != nil {
		t.Error("without blame nothing is taught")
	}
}

func TestOllamaReviewSamplesFromTheModel(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/review", map[string]any{"count": 3, "max_length": 24})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	if doc["source"] != "model" {
		t.Errorf("source: %v", doc["source"])
	}
	texts, _ := doc["texts"].([]any)
	if len(texts) != 3 {
		t.Errorf("texts: %v", doc["texts"])
	}
}

func TestOllamaReviewCanBlame(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/review", map[string]any{
		"texts": []string{"xxxx xxxx xxxx"}, "blame": true,
	})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	negative, _ := doc["negative"].(map[string]any)
	if negative == nil {
		t.Fatal("expected the negative network to be taught")
	}
	taught, _ := negative["taught"].(map[string]any)
	if blamed, _ := taught["blamed"].(float64); blamed != 1 {
		t.Errorf("blamed: %v", taught["blamed"])
	}
	status, tab := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d", status)
	}
	stats, _ := tab["stats"].(map[string]any)
	if failures, _ := stats["failures_total"].(float64); failures != 1 {
		t.Errorf("the tab must see it: %v", stats["failures_total"])
	}
}

func TestTheNewRoutesAreDocumented(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.get("/api")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	raw, _ := json.Marshal(doc)
	for _, path := range []string{
		"/api/negative/auto", "/api/negative/auto/history", "/api/ollama/models",
		"/api/ollama/corpus", "/api/ollama/review",
	} {
		if !strings.Contains(string(raw), path) {
			t.Errorf("%s is not in the endpoint index", path)
		}
	}
	_ = os.Getenv
}
