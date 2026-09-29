package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The tools and the agent over HTTP.  A fake site stands in for the internet
// and a fake Ollama plays the four LLM roles, so every call the agent makes
// really happens - against 127.0.0.1, which is what allow_private is for.

const agentHome = "<html><head><title>Cats</title></head><body><p>A cat has four legs.</p>" +
	"<a href='/more'>more about cats</a></body></html>"

func agentSite(t *testing.T) *httptest.Server {
	t.Helper()
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(agentHome))
	}))
	t.Cleanup(site.Close)
	return site
}

var agentCriterion = regexp.MustCompile(`(?m)^\d+\. `)

// agentLLM answers as criteria writer, mediator, judge and teacher.
func agentLLM(t *testing.T, site string) *httptest.Server {
	t.Helper()
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "fake:latest"}}})
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		call := []any{map[string]any{"function": map[string]any{
			"name": "web_fetch", "arguments": map[string]any{"url": site + "/cats"},
		}}}
		if strings.HasSuffix(r.URL.Path, "/api/chat") {
			messages, _ := body["messages"].([]any)
			system, replies := "", 0
			for _, item := range messages {
				message, _ := item.(map[string]any)
				if message["role"] == "system" {
					system, _ = message["content"].(string)
				}
				if message["role"] == "tool" {
					replies++
				}
			}
			switch {
			case strings.Contains(system, "tool mediator"):
				json.NewEncoder(w).Encode(map[string]any{"message": map[string]any{
					"role": "assistant", "content": "", "tool_calls": call}})
			case strings.Contains(system, "solve a task with the tools") && replies == 0:
				json.NewEncoder(w).Encode(map[string]any{"message": map[string]any{
					"role": "assistant", "content": "", "tool_calls": call}})
			default:
				json.NewEncoder(w).Encode(map[string]any{"message": map[string]any{
					"role": "assistant", "content": "A cat has four legs."}})
			}
			return
		}
		system, _ := body["system"].(string)
		prompt, _ := body["prompt"].(string)
		switch {
		case strings.Contains(system, "judge of an attempt"):
			parts := strings.Split(prompt, "Answer:")
			good := strings.Contains(strings.ToLower(parts[len(parts)-1]), "four")
			count := len(agentCriterion.FindAllString(prompt, -1))
			if count == 0 {
				count = 1
			}
			met := make([]bool, count)
			for i := range met {
				met[i] = good
			}
			score := 2
			if good {
				score = 9
			}
			raw, _ := json.Marshal(map[string]any{"met": met, "score": score, "correct": good, "critique": "fine"})
			fmt.Fprintf(w, `{"response": %q}`, string(raw))
		case strings.Contains(system, "acceptance criteria"):
			raw, _ := json.Marshal(map[string]any{"criteria": []string{"The answer names four"}})
			fmt.Fprintf(w, `{"response": %q}`, string(raw))
		case strings.Contains(system, "choosing what to look into next"):
			raw, _ := json.Marshal(map[string]any{"question": "How many legs does a cat have?", "seed": site + "/cats"})
			fmt.Fprintf(w, `{"response": %q}`, string(raw))
		default:
			fmt.Fprint(w, `{"response": "unexpected prompt"}`)
		}
	}))
	t.Cleanup(fake.Close)
	return fake
}

// agentEnv is a service whose LLM defaults point at the fake roles and whose
// web tools may reach the fake site.
func agentEnv(t *testing.T) *env {
	t.Helper()
	site := agentSite(t)
	fake := agentLLM(t, site.URL)
	dir := t.TempDir()
	svc, err := NewService(Options{
		ModelPath: filepath.Join(dir, "model.count.json"), Seed: 1, Exact: true, Quiet: true,
		OllamaURL: fake.URL, OllamaModel: "fake:latest", AllowPrivate: true, SearchURL: site.URL + "/search?q={query}",
		UploadDir: filepath.Join(dir, "uploads"),
	})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(NewHandler(svc, "", true, nil))
	t.Cleanup(ts.Close)
	return &env{t: t, ts: ts, svc: svc, dir: dir}
}

func TestToolsAreDescribedAndCalled(t *testing.T) {
	e := agentEnv(t)
	status, doc := e.get("/api/tools")
	if status != 200 {
		t.Fatalf("tools: %d %+v", status, doc)
	}
	names := []string{}
	for _, name := range doc["names"].([]any) {
		names = append(names, name.(string))
	}
	if len(names) < 4 || names[0] != "web_search" {
		t.Fatalf("the built-in set: %+v", names)
	}
	if doc["call_format"] != `<tool>name {"argument": "value"}</tool>` {
		t.Fatalf("how the network calls one: %+v", doc["call_format"])
	}
	first := doc["tools"].([]any)[0].(map[string]any)
	if first["signature"] != "web_search(query: string, [limit: integer])" || first["network"] != true {
		t.Fatalf("the signature: %+v", first)
	}
	// by name and arguments
	status, doc = e.post("/api/tools/call", map[string]any{"tool": "calculator", "arguments": map[string]any{"expression": "6*7"}})
	if status != 200 || doc["ok"] != true || doc["output"] != "42" {
		t.Fatalf("calculator: %d %+v", status, doc)
	}
	// or as the network would write it
	status, doc = e.post("/api/tools/call", map[string]any{"call": `calculator {"expression": "sqrt(841)"}`})
	if status != 200 || doc["output"] != "29.0" {
		t.Fatalf("a written call: %d %+v", status, doc)
	}
	// a failing call is an observation, not an error
	status, doc = e.post("/api/tools/call", map[string]any{"tool": "calculator", "arguments": map[string]any{"expression": "1/0"}})
	if status != 200 || doc["ok"] != false || doc["error"] != "division by zero" {
		t.Fatalf("a failure: %d %+v", status, doc)
	}
	if status, doc = e.post("/api/tools/call", map[string]any{"tool": "nosuchtool"}); status != 404 {
		t.Fatalf("an unknown tool is a 404: %d %+v", status, doc)
	}
	// offline drops the web tools
	status, doc = e.post("/api/tools/call", map[string]any{"tool": "web_fetch", "offline": true,
		"arguments": map[string]any{"url": "http://example.com"}})
	if status != 404 {
		t.Fatalf("offline: %d %+v", status, doc)
	}
}

func TestAgentCriteriaAndSolve(t *testing.T) {
	e := agentEnv(t)
	status, doc := e.post("/api/agent/criteria", map[string]any{"tasks": []any{"How many legs does a cat have?"}})
	if status != 200 {
		t.Fatalf("criteria: %d %+v", status, doc)
	}
	first := doc["tasks"].([]any)[0].(map[string]any)
	if criteria := first["criteria"].([]any); len(criteria) != 1 || criteria[0] != "The answer names four" {
		t.Fatalf("the criteria: %+v", first)
	}
	// the teacher demonstrates with the real tools, and the judge marks it
	status, doc = e.post("/api/agent/solve", map[string]any{
		"task": "How many legs does a cat have?", "source": "teacher", "max_steps": 2,
	})
	if status != 200 {
		t.Fatalf("solve: %d %+v", status, doc)
	}
	if doc["correct"] != true {
		t.Fatalf("the teacher solves it: %+v", doc)
	}
	attempt := doc["attempt"].(map[string]any)
	steps := attempt["steps"].([]any)
	if len(steps) != 1 || steps[0].(map[string]any)["tool"] != "web_fetch" {
		t.Fatalf("it really called the tool: %+v", steps)
	}
	if !strings.Contains(doc["transcript"].(string), "<tool>web_fetch") {
		t.Fatalf("the transcript is what the network would learn: %q", doc["transcript"])
	}
	if status, doc = e.post("/api/agent/solve", map[string]any{"tasks": []any{"a", "b"}}); status != 400 {
		t.Fatalf("solve takes one task: %d %+v", status, doc)
	}
}

func TestAgentRunsAJobAndKeepsItsHistory(t *testing.T) {
	e := agentEnv(t)
	status, doc := e.get("/api/agent/history")
	if status != 200 || len(doc["history"].([]any)) != 0 {
		t.Fatalf("nothing has run yet: %d %+v", status, doc)
	}
	status, doc = e.post("/api/agent/start", map[string]any{
		"tasks_text": "How many legs does a cat have?\n# a comment is not a task\n",
		"max_steps":  1, "model_attempts": 1, "neg_epochs": 1, "pos_epochs": 1, "max_length": 60,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	if tasks := doc["tasks"].([]any); len(tasks) != 1 {
		t.Fatalf("one task, the comment skipped: %+v", tasks)
	}
	job := e.waitForJob()
	if job["state"] != "done" {
		t.Fatalf("the job finishes: %+v", job)
	}
	status, doc = e.get("/api/agent/history")
	if status != 200 {
		t.Fatalf("history: %d %+v", status, doc)
	}
	kinds := map[string]int{}
	for _, row := range doc["history"].([]any) {
		kinds[row.(map[string]any)["kind"].(string)]++
	}
	if kinds["criteria"] != 1 || kinds["task"] != 1 || kinds["round"] != 1 {
		t.Fatalf("criteria, one task, one round: %+v", kinds)
	}
	if kinds["attempt"] < 2 { // the network's attempt and the teacher's demonstration
		t.Fatalf("both attempts recorded: %+v", kinds)
	}
	if status, doc = e.post("/api/agent/start", map[string]any{}); status != 400 {
		t.Fatalf("no tasks is a 400: %d %+v", status, doc)
	}
}

func TestExploreChoosesItsOwnTasks(t *testing.T) {
	e := agentEnv(t)
	status, doc := e.post("/api/agent/explore", map[string]any{
		"steps": 1, "max_steps": 1, "model_attempts": 1, "neg_epochs": 1, "pos_epochs": 1, "max_length": 60,
	})
	if status != 202 {
		t.Fatalf("explore: %d %+v", status, doc)
	}
	job := e.waitForJob()
	if job["state"] != "done" {
		t.Fatalf("the job finishes: %+v", job)
	}
	status, doc = e.get("/api/agent/history")
	if status != 200 {
		t.Fatalf("history: %d %+v", status, doc)
	}
	proposals := 0
	for _, row := range doc["history"].([]any) {
		record := row.(map[string]any)
		if record["kind"] == "proposal" {
			proposals++
			if !strings.Contains(record["prompt"].(string), "cat") {
				t.Fatalf("the LLM turned the emission into a question: %+v", record)
			}
		}
	}
	if proposals != 1 {
		t.Fatalf("one proposal per step: %d", proposals)
	}
}
