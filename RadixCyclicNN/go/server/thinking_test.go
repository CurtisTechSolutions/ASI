package server

import (
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// thinkingEnv is criticEnv with an upload directory, so a thinking request may save what it heard.
func thinkingEnv(t *testing.T) (*env, *fakeReviewer) {
	t.Helper()
	fake := newFakeReviewer(t)
	dir := t.TempDir()
	svc, err := NewService(Options{
		ModelPath: filepath.Join(dir, "model.count.json"), Seed: 1, Exact: true, Quiet: true,
		UploadDir: filepath.Join(dir, "uploads"), OllamaURL: fake.server.URL, OllamaModel: "fake:latest",
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

var thinkCorpus = []string{"ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat", "the cat sat on the mat"}

func thinkEnv(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, true)
	if status, doc := e.post("/api/train", map[string]any{"texts": thinkCorpus, "epochs": 3}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitJob()
	return e
}

func TestTheThinkRouteThinks(t *testing.T) {
	e := thinkEnv(t)
	status, doc := e.post("/api/think", map[string]any{"about": "the cat"})
	if status != 200 {
		t.Fatalf("think: %d %+v", status, doc)
	}
	hasKeys(t, doc, "kind", "trigger", "at", "about", "text", "depth", "stopped", "then", "taught", "handed_over",
		"cost", "probability", "expanded", "questioned", "questions", "labels", "node_ids", "step_costs")
	at := int(doc["at"].(float64))
	if doc["kind"] != "count" || doc["trigger"] != radixnet.TriggerAsked || at < radixnet.First {
		t.Fatalf("thought %+v", doc)
	}
	if int(doc["taught"].(float64)) != at || doc["then"] != radixnet.ThenEnd || doc["stopped"] != radixnet.StoppedNothing {
		t.Fatalf("thought %+v", doc)
	}
	if questions, ok := doc["questions"].([]any); !ok || len(questions) != 0 {
		t.Fatalf("questions %v", doc["questions"])
	}
	// learning off: nothing taught
	status, doc = e.post("/api/think", map[string]any{"about": "the cat", "learn": false})
	if status != 200 || int(doc["taught"].(float64)) != -1 {
		t.Fatalf("think without learning: %d %+v", status, doc)
	}
	for _, bad := range []map[string]any{{"mode": "walk"}, {"depth": -1}, {"k": 0}, {"about": 3}} {
		if status, doc := e.post("/api/think", bad); status != 400 {
			t.Fatalf("%v: %d %+v", bad, status, doc)
		}
	}
	status, info := e.get("/api/encoding")
	if status != 200 || info["think_label"] != radixnet.ThinkLabel {
		t.Fatalf("encoding: %d %+v", status, info)
	}
}

func TestAConversationThinksAndSaysSo(t *testing.T) {
	e := thinkEnv(t)
	status, doc := e.post("/api/converse", map[string]any{"turns": 6, "guard": false})
	if status != 200 {
		t.Fatalf("converse: %d %+v", status, doc)
	}
	thoughts := 0
	for _, raw := range doc["turns"].([]any) {
		turn := raw.(map[string]any)
		rethink, ok := turn["rethink"].(map[string]any)
		if !ok {
			continue
		}
		if _, present := rethink["thought"]; !present {
			t.Fatalf("rethink without a thought key: %v", rethink)
		}
		thought, ok := rethink["thought"].(map[string]any)
		if !ok {
			continue
		}
		thoughts++
		if thought["then"] != radixnet.ThenBack || !radixnet.BacksUp(thought["trigger"].(string)) {
			t.Fatalf("thought %v", thought)
		}
	}
	if thoughts == 0 {
		t.Fatalf("no turn thought: %v", doc["turns"])
	}
	status, doc = e.post("/api/converse", map[string]any{"turns": 4, "guard": false, "think": false})
	if status != 200 {
		t.Fatalf("converse: %d %+v", status, doc)
	}
	for _, raw := range doc["turns"].([]any) {
		if rethink, ok := raw.(map[string]any)["rethink"].(map[string]any); ok && rethink["thought"] != nil {
			t.Fatalf("a conversation with thinking off thought: %v", rethink)
		}
	}
	if _, doc := e.get("/api"); !strings.Contains(doc["endpoints"].([]any)[0].(map[string]any)["doc"].(string), "") {
		t.Fatal("no endpoint index")
	}
}

func TestOllamaThinkTeachesThoughts(t *testing.T) {
	e, fake := thinkingEnv(t)
	status, doc := e.post("/api/ollama/think", map[string]any{"prompt": "the sea", "lines": 2})
	if status != 200 {
		t.Fatalf("think: %d %+v", status, doc)
	}
	if doc["count"] != 2.0 || doc["thinking"] != 2.0 || doc["think"] != true || doc["job"] != nil || doc["upload"] != nil {
		t.Fatalf("doc %+v", doc)
	}
	thoughts := doc["thoughts"].([]any)
	second := thoughts[1].(map[string]any)
	if second["question"] != "question 2 about the sea?" || !strings.HasPrefix(second["thinking"].(string), "Let me think about") {
		t.Fatalf("thought %v", second)
	}
	// the questions were asked without thinking, each answer with it
	asked := fake.bodies[len(fake.bodies)-3:]
	if _, ok := asked[0]["think"]; ok {
		t.Fatalf("the questions request asked for thinking: %v", asked[0])
	}
	if asked[1]["think"] != true || asked[2]["think"] != true {
		t.Fatalf("the answers were not asked to think: %v %v", asked[1], asked[2])
	}

	body := map[string]any{"prompt": "the sea", "lines": 3, "train": true, "think": "high", "save_as": "sea-thoughts.txt", "epochs": 3}
	status, doc = e.post("/api/ollama/think", body)
	if status != 202 {
		t.Fatalf("think and train: %d %+v", status, doc)
	}
	if doc["think"] != "high" || doc["job"].(map[string]any)["type"] != "train" {
		t.Fatalf("doc %+v", doc)
	}
	upload := doc["upload"].(map[string]any)
	if upload["name"] != "sea-thoughts.txt" || upload["lines"] != 3.0 {
		t.Fatalf("upload %v", upload)
	}
	if job := e.waitForJob(); job["state"] != "done" {
		t.Fatalf("job %v", job)
	}
	g := e.svc.model.G
	if len(g.Children(radixnet.Think)) == 0 { // it has thoughts now
		t.Fatal("Think has no children")
	}
	if len(g.Parents(radixnet.Think)) == 0 { // and knows where a thought questions itself
		t.Fatal("Think has no parents")
	}
	status, thought := e.post("/api/think", map[string]any{"learn": false})
	if status != 200 || thought["text"] == "" {
		t.Fatalf("think: %d %+v", status, thought)
	}
	// the thoughts are thoughts, not texts: nothing from Start begins with them
	status, samples := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "guard": false})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, samples)
	}
	for _, raw := range samples["samples"].([]any) {
		if text := raw.(map[string]any)["text"].(string); strings.HasPrefix(strings.ToLower(text), "let me think") {
			t.Fatalf("a text begins like a thought: %q", text)
		}
	}
	before := e.svc.model.Stats()["trained_texts"]
	body = map[string]any{"prompt": "the sea", "lines": 1, "train": true, "with_answers": true, "epochs": 3}
	if status, doc = e.post("/api/ollama/think", body); status != 202 {
		t.Fatalf("think with answers: %d %+v", status, doc)
	}
	if job := e.waitForJob(); job["state"] != "done" {
		t.Fatalf("job %v", job)
	}
	// 1 thought + its 2 questions + 1 answer
	if after := e.svc.model.Stats()["trained_texts"]; after.(int64)-before.(int64) != 4 {
		t.Fatalf("trained texts went from %v to %v", before, after)
	}
}

func TestOllamaThinkErrors(t *testing.T) {
	e, fake := criticEnv(t)
	if status, doc := e.post("/api/ollama/think", map[string]any{"lines": 3}); status != 400 {
		t.Fatalf("no prompt: %d %+v", status, doc)
	}
	if status, doc := e.post("/api/ollama/think", map[string]any{"prompt": "x", "think": "loud"}); status != 400 {
		t.Fatalf("bad think: %d %+v", status, doc)
	}
	fake.thinking = "none"
	status, doc := e.post("/api/ollama/think", map[string]any{"prompt": "x", "lines": 1})
	if status != 200 || doc["thinking"] != 0.0 {
		t.Fatalf("a model that does not think: %d %+v", status, doc)
	}
	status, doc = e.post("/api/ollama/think", map[string]any{"prompt": "x", "lines": 1, "train": true})
	if status != 502 || !strings.Contains(doc["error"].(string), "no thinking") {
		t.Fatalf("training on no thinking: %d %+v", status, doc)
	}
	fake.thinking = "inline"
	status, doc = e.post("/api/ollama/think", map[string]any{"prompt": "x", "lines": 1})
	if status != 200 || doc["thinking"] != 1.0 {
		t.Fatalf("inline thinking: %d %+v", status, doc)
	}
	entry := doc["thoughts"].([]any)[0].(map[string]any)
	if !strings.HasPrefix(entry["thinking"].(string), "Let me think about") || !strings.HasPrefix(entry["answer"].(string), "The answer to") {
		t.Fatalf("entry %v", entry)
	}
}
