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

// fakeReviewer answers the review, correction and corpus prompts: anything
// with "good" in it passes, everything else fails for repeating itself; as a
// copy editor it fixes howe, doubled question marks and an opening Hi without
// its comma (the same craft as the Python tests' fake).
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

// fakeCopyEdit is the fake editor's whole craft; a text with nothing to fix
// comes back unchanged with the reason "none".
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

func (f *fakeReviewer) answer(system, prompt string) string {
	if strings.Contains(prompt, "Correct these") { // a copy-editing request
		type entry struct {
			Index      int    `json:"index"`
			Correction string `json:"correction"`
			Reason     string `json:"reason"`
			Note       string `json:"note"`
		}
		out := []entry{}
		for _, line := range strings.Split(prompt, "\n") {
			match := reviewLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			index, _ := strconv.Atoi(match[1])
			corrected, reason := fakeCopyEdit(match[2])
			note := "nothing"
			if reason != "none" {
				note = reason + " fixed"
			}
			out = append(out, entry{index, corrected, reason, note})
		}
		raw, _ := json.Marshal(map[string]any{"corrections": out})
		return string(raw)
	}
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
		"/api/ollama/corpus", "/api/ollama/review", "/api/ollama/correct",
	} {
		if !strings.Contains(string(raw), path) {
			t.Errorf("%s is not in the endpoint index", path)
		}
	}
	_ = os.Getenv
}

func TestOllamaCorrectGivenTexts(t *testing.T) {
	e, fake := criticEnv(t)
	status, doc := e.post("/api/ollama/correct", map[string]any{
		"texts": []string{"Hi howe are you??", "good sentences about the cat", "   "}, "context": "greetings",
	})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	hasKeys(t, doc, "source", "model", "texts", "corrections", "corrected", "unchanged", "uncorrected", "edits",
		"wrong_chars", "right_chars", "change_rate", "url", "severity", "negative")
	if doc["source"] != "given" || doc["negative"] != nil || toFloat(doc["severity"]) != 1 {
		t.Fatalf("given texts, nothing taught, the default severity: %+v", doc)
	}
	corrections := doc["corrections"].([]any)
	verdicts := []string{}
	for _, row := range corrections {
		verdicts = append(verdicts, row.(map[string]any)["verdict"].(string))
	}
	if strings.Join(verdicts, ",") != "corrected,unchanged,uncorrected" {
		t.Fatalf("verdicts: %v", verdicts)
	}
	first := corrections[0].(map[string]any)
	if first["correction"] != "Hi, how are you?" || first["reason"] != "spelling" || first["note"] != "spelling fixed" {
		t.Fatalf("the editor's answer: %+v", first)
	}
	hasKeys(t, first, "index", "text", "correction", "verdict", "reason", "note", "changes", "edits", "wrong_chars", "right_chars")
	change := first["changes"].([]any)[0].(map[string]any)
	hasKeys(t, change, "op", "wrong", "right", "at", "to")
	if len(doc["corrected"].([]any)) != 1 || len(doc["unchanged"].([]any)) != 1 || len(doc["uncorrected"].([]any)) != 1 {
		t.Fatalf("the split: %+v", doc)
	}
	if toFloat(doc["change_rate"]) != 0.5 {
		t.Fatalf("one of the two answered texts was changed: %v", doc["change_rate"])
	}
	want := "Context: greetings\n\nCorrect these 2 texts:\n[0] Hi howe are you??\n[1] good sentences about the cat\n\nReturn the JSON now."
	if len(fake.prompts) != 1 || fake.prompts[0] != want {
		t.Fatalf("the editor's prompt: %q", fake.prompts)
	}
	if status, _ := e.post("/api/ollama/correct", map[string]any{"text": "x", "severity": -1}); status != 400 {
		t.Fatalf("a negative severity is a 400: %d", status)
	}
}

func TestOllamaCorrectSamplesFromTheModel(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/correct", map[string]any{"count": 3, "max_length": 24})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	if doc["source"] != "model" || len(doc["texts"].([]any)) != 3 || len(doc["corrections"].([]any)) != 3 {
		t.Fatalf("three samples corrected: %+v", doc)
	}
}

func TestOllamaCorrectCanBlame(t *testing.T) {
	e, _ := criticEnv(t)
	status, doc := e.post("/api/ollama/correct", map[string]any{
		"texts": []string{"Hi howe are you??", "good sentences about the cat"}, "blame": true, "severity": 1.5,
	})
	if status != 200 {
		t.Fatalf("status %d %+v", status, doc)
	}
	negative, _ := doc["negative"].(map[string]any)
	if negative == nil {
		t.Fatal("expected the negative network to be taught")
	}
	hasKeys(t, negative, "blamed", "cleared", "unmatched", "edges", "edits", "uncorrected", "reasons", "lessons",
		"severity_mean", "stats", "reason_table")
	if toFloat(negative["blamed"]) != 1 || toFloat(negative["severity_mean"]) != 1.5 || toFloat(negative["edits"]) == 0 {
		t.Fatalf("one text blamed at the asked severity, by its diff: %+v", negative)
	}
	if lessons := negative["lessons"].([]any); len(lessons) != 1 || lessons[0].(map[string]any)["correction"] != "Hi, how are you?" {
		t.Fatalf("the lesson carries the correction: %+v", negative["lessons"])
	}
	if table := negative["reason_table"].([]any); len(table) != 1 || table[0].(map[string]any)["reason"] != "spelling" {
		t.Fatalf("under the editor's word: %+v", negative["reason_table"])
	}
	status, tab := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d", status)
	}
	if failures := toFloat(tab["stats"].(map[string]any)["failures_total"]); failures != 1 {
		t.Errorf("the tab must see it: %v", failures)
	}
	status, judged := e.post("/api/negative/judge", map[string]any{"texts": []string{"Hi, how are you?", "Hi howe are you??"}})
	if status != 200 {
		t.Fatalf("judge: %d", status)
	}
	verdicts := judged["verdicts"].([]any)
	if verdicts[0].(map[string]any)["verdict"] != "pass" {
		t.Fatalf("the correction is clean: %+v", verdicts[0])
	}
	if reasons := verdicts[1].(map[string]any)["reasons"].([]any); len(reasons) == 0 || reasons[0].(map[string]any)["reason"] != "spelling" {
		t.Fatalf("the mistake is known: %+v", verdicts[1])
	}
}

func TestNegativeAutoCanCorrect(t *testing.T) {
	e, fake := criticEnv(t)
	status, doc := e.post("/api/negative/auto", map[string]any{"rounds": 1, "count": 3, "max_length": 24, "correct": true, "severity": 2})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	config := doc["config"].(map[string]any)
	if config["correct"] != true || toFloat(config["severity"]) != 2 {
		t.Fatalf("the config echoes the editor's settings: %+v", config)
	}
	e.waitForJob()
	status, history := e.get("/api/negative/auto/history")
	if status != 200 {
		t.Fatalf("history: %d", status)
	}
	records := history["history"].([]any)
	if len(records) != 2 {
		t.Fatalf("one round and one report: %+v", records)
	}
	round := records[0].(map[string]any)
	if round["mode"] != "correct" || toFloat(round["severity"]) != 2 || toFloat(round["texts"]) != 3 {
		t.Fatalf("a correcting round: %+v", round)
	}
	hasKeys(t, round, "corrections", "change_rate", "corrected", "unchanged", "uncorrected", "edits", "wrong_chars")
	if toFloat(round["corrected"])+toFloat(round["unchanged"])+toFloat(round["uncorrected"]) != 3 {
		t.Fatalf("every text has a verdict: %+v", round)
	}
	report := records[1].(map[string]any)
	hasKeys(t, report, "corrected", "unchanged", "uncorrected", "edits", "change_rate", "change_trend")
	if len(fake.prompts) != 1 || !strings.HasPrefix(fake.prompts[0], "Correct these 3 texts:") {
		t.Fatalf("the editor was asked once: %q", fake.prompts)
	}
	if status, _ := e.post("/api/negative/auto", map[string]any{"correct": true, "severity": -1}); status != 400 {
		t.Fatalf("a negative severity is a 400: %d", status)
	}
}
