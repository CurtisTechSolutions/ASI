package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The chat job: an LLM converses with the model and marks every reply.

var chatCorpus = []string{
	"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill",
}

var chatPartnerLines = []string{"tell me about the cat", "and what about the dog", "now tell me about a bird"}

var chatExchange = regexp.MustCompile(`^\[(\d+)\] Partner: (.*)$`)

// fakePartner plays the partner and the judge, from the same prompts the real
// ones would see.
type fakePartner struct {
	server *httptest.Server
	lines  int
	marked int
}

func newFakePartner(t *testing.T) *fakePartner {
	t.Helper()
	fake := &fakePartner{}
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
		json.NewEncoder(w).Encode(map[string]any{"response": fake.answer(system, prompt)})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakePartner) answer(system, prompt string) string {
	switch {
	case strings.Contains(system, "marking a conversation"):
		type entry struct {
			Index    int     `json:"index"`
			Rating   float64 `json:"rating"`
			Critique string  `json:"critique"`
		}
		out := []entry{}
		for _, line := range strings.Split(prompt, "\n") {
			match := chatExchange.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			index, _ := strconv.Atoi(match[1])
			out = append(out, entry{index, 9, "it follows on"})
		}
		f.marked++
		raw, _ := json.Marshal(map[string]any{
			"reviews": out, "overall": map[string]any{"rating": 7, "critique": "it kept up, mostly"},
		})
		return string(raw)
	case strings.Contains(system, "conversation with a very small"):
		f.lines++
		return chatPartnerLines[(f.lines-1)%len(chatPartnerLines)]
	}
	return "unexpected prompt"
}

// chatEnv is a service whose LLM defaults point at the fake partner, trained so
// the model has something to reply with.
func chatEnv(t *testing.T) (*env, *fakePartner) {
	t.Helper()
	fake := newFakePartner(t)
	dir := t.TempDir()
	svc, err := NewService(Options{
		ModelPath: filepath.Join(dir, "model.count.json"), Seed: 5, Exact: true, Quiet: true,
		OllamaURL: fake.server.URL, OllamaModel: "fake:latest",
	})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(NewHandler(svc, "", true, nil))
	t.Cleanup(ts.Close)
	e := &env{t: t, ts: ts, svc: svc, dir: dir}
	if status, doc := e.post("/api/train", map[string]any{"texts": chatCorpus, "epochs": 4}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitForJob()
	return e, fake
}

func TestChatStartAndHistory(t *testing.T) {
	e, fake := chatEnv(t)
	status, doc := e.post("/api/chat/start", map[string]any{
		"conversations": 1, "turns": 2, "model": "fake:latest", "url": fake.server.URL,
		"topic": "animals", "learn": false,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	job, _ := doc["job"].(map[string]any)
	if job == nil || job["type"] != "chat" {
		t.Fatalf("expected a chat job: %+v", doc)
	}
	config, _ := doc["config"].(map[string]any)
	if config["topic"] != "animals" {
		t.Errorf("config: %+v", config)
	}
	if doc["partner"] != "fake:latest" || doc["judge"] != "fake:latest" {
		t.Errorf("partner / judge: %v, %v", doc["partner"], doc["judge"])
	}
	speakers, _ := doc["speakers"].([]any)
	if len(speakers) != 2 || speakers[0] != "Partner" || speakers[1] != "Model" {
		t.Errorf("speakers: %v", speakers)
	}
	if finished := e.waitForJob(); finished["state"] != "done" {
		t.Fatalf("job: %+v", finished)
	}
	status, history := e.get("/api/chat/history")
	if status != 200 {
		t.Fatalf("history: %d", status)
	}
	records, _ := history["history"].([]any)
	kinds := map[string]int{}
	var held map[string]any
	for _, record := range records {
		row, _ := record.(map[string]any)
		kind, _ := row["kind"].(string)
		kinds[kind]++
		if kind == "conversation" && held == nil {
			held = row
		}
	}
	if kinds["conversation"] != 1 || kinds["report"] != 1 || kinds["exchange"] == 0 {
		t.Fatalf("records: %v", kinds)
	}
	transcript, _ := held["transcript"].([]any)
	if len(transcript) == 0 {
		t.Fatalf("no transcript: %+v", held)
	}
	first, _ := transcript[0].(map[string]any)
	if first["speaker"] != "Partner" {
		t.Errorf("the partner did not open: %+v", first)
	}
	if fake.marked != 1 { // one judgement per conversation
		t.Errorf("marked %d time(s)", fake.marked)
	}
	// the failures and passes are sourced to this loop
	status, negative := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d", status)
	}
	if negative["stats"] == nil {
		t.Errorf("no negative network was consulted: %+v", negative)
	}
}

func TestChatLearnsAndTeachesTheNegativeNetwork(t *testing.T) {
	e, fake := chatEnv(t)
	status, doc := e.post("/api/chat/start", map[string]any{
		"conversations": 1, "turns": 2, "url": fake.server.URL, "threshold": 10, // nothing can pass
		"neg_epochs": 1, "pos_epochs": 1,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	if finished := e.waitForJob(); finished["state"] != "done" {
		t.Fatalf("job: %+v", finished)
	}
	_, negative := e.get("/api/negative")
	stats, _ := negative["stats"].(map[string]any)
	if failures, _ := stats["failures_total"].(float64); failures <= 0 {
		t.Fatalf("the loop taught nothing: %+v", stats)
	}
	sources, _ := stats["sources"].(map[string]any)
	if sources["chat"] == nil {
		t.Errorf("the blame must be sourced to the loop: %v", sources)
	}
	_, history := e.get("/api/chat/history")
	records, _ := history["history"].([]any)
	for _, record := range records {
		row, _ := record.(map[string]any)
		if row["kind"] != "conversation" {
			continue
		}
		if row["action"] == nil {
			t.Errorf("it marked but did not learn: %+v", row)
		}
	}
}

func TestChatRefusesABadConfig(t *testing.T) {
	e, _ := chatEnv(t)
	for name, body := range map[string]map[string]any{
		"turns":     {"turns": 0},
		"provider":  {"provider": "gemini"},
		"threshold": {"threshold": 11},
		"mode":      {"mode": "dijkstra"},
		"k":         {"k": 0},
	} {
		status, doc := e.post("/api/chat/start", body)
		if status != 400 {
			t.Errorf("%s: expected 400, got %d %+v", name, status, doc)
		}
	}
}

func TestChatHistoryIsEmptyBeforeItRuns(t *testing.T) {
	e, _ := chatEnv(t)
	status, doc := e.get("/api/chat/history")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	records, _ := doc["history"].([]any)
	if len(records) != 0 {
		t.Errorf("expected no records, got %v", records)
	}
}
