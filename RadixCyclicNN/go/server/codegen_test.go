package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// fakeCoder answers the codegen prompts: the first program it is asked for
// does not run, the fix does, and the judge passes anything that ran.
type fakeCoder struct {
	server  *httptest.Server
	prompts []string
}

func newFakeCoder(t *testing.T) *fakeCoder {
	t.Helper()
	fake := &fakeCoder{}
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

func (f *fakeCoder) answer(system, prompt string) string {
	if strings.Contains(system, "strict reviewer") {
		raw, _ := json.Marshal(map[string]any{
			"runs": true, "task": true, "pep8": true, "naming": true, "correct": true,
			"critique": "it does what was asked",
		})
		return string(raw)
	}
	if strings.Contains(prompt, "not acceptable yet") { // the fix prompt
		return "```python\nprint(\"hello\")\n```"
	}
	return "```python\nprint(hello)\n```" // NameError: the sandbox rejects it
}

// codegenEnv is a service whose LLM defaults point at the fake teacher.
func codegenEnv(t *testing.T) (*env, *fakeCoder) {
	t.Helper()
	fake := newFakeCoder(t)
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
	return &env{t: t, ts: ts, svc: svc, dir: dir}, fake
}

func TestCodeGenRunsAProgramInTheSandbox(t *testing.T) {
	e, _ := codegenEnv(t)
	status, doc := e.post("/api/codegen/run", map[string]any{"code": "print(\"hello\")\n", "expected_output": "hello"})
	if status != 200 {
		t.Fatalf("run: %d %+v", status, doc)
	}
	run := doc["run"].(map[string]any)
	if run["ok"] != true || strings.TrimSpace(run["stdout"].(string)) != "hello" {
		t.Fatalf("it runs and prints: %+v", run)
	}
	if run["expected_ok"] != true {
		t.Fatalf("and matches the expected output: %+v", run)
	}
	if doc["verdict"].(map[string]any)["correct"] != true {
		t.Fatalf("so the verdict is correct: %+v", doc["verdict"])
	}
	status, doc = e.post("/api/codegen/run", map[string]any{"code": "raise ValueError('boom')\n"})
	if status != 200 {
		t.Fatalf("crash: %d %+v", status, doc)
	}
	run = doc["run"].(map[string]any)
	if run["ok"] != false || !strings.Contains(run["error"].(string), "ValueError") {
		t.Fatalf("a crash is reported, not raised: %+v", run)
	}
	if strings.Contains(run["stderr"].(string), "runpy") {
		t.Fatalf("the harness frames are cleaned out of the traceback: %+v", run["stderr"])
	}
	if status, doc = e.post("/api/codegen/run", map[string]any{"code": "  \n"}); status != 400 {
		t.Fatalf("an empty program is a 400: %d %+v", status, doc)
	}
}

func TestCodeGenSolvesOneProblemWithTheTeacher(t *testing.T) {
	e, _ := codegenEnv(t)
	status, doc := e.post("/api/codegen/solve", map[string]any{
		"problem": "Print the word hello.", "source": "teacher", "attempts": 2, "sandbox_timeout": 20,
	})
	if status != 200 {
		t.Fatalf("solve: %d %+v", status, doc)
	}
	if doc["correct"] != true {
		t.Fatalf("the teacher's fix solves it: %+v", doc)
	}
	attempts := doc["attempts"].([]any)
	if len(attempts) != 2 {
		t.Fatalf("one failure, then the fix: %+v", attempts)
	}
	first := attempts[0].(map[string]any)
	if first["verdict"].(map[string]any)["correct"] != false {
		t.Fatalf("the first attempt does not run: %+v", first)
	}
	if status, doc = e.post("/api/codegen/solve", map[string]any{"problem": "x", "source": "nobody"}); status != 400 {
		t.Fatalf("an unknown source is a 400: %d %+v", status, doc)
	}
	if status, doc = e.post("/api/codegen/solve", map[string]any{}); status != 400 {
		t.Fatalf("no problem is a 400: %d %+v", status, doc)
	}
}

func TestCodeGenStartsAJobAndKeepsItsHistory(t *testing.T) {
	e, _ := codegenEnv(t)
	status, doc := e.get("/api/codegen/history")
	if status != 200 || len(doc["history"].([]any)) != 0 {
		t.Fatalf("nothing has run yet: %d %+v", status, doc)
	}
	status, doc = e.post("/api/codegen/start", map[string]any{
		"problems_text": "Print the word hello.\n# a comment is not a problem\n",
		"phases":        "teacher", "teacher_attempts": 2, "sandbox_timeout": 20, "blame": true,
	})
	if status != 202 {
		t.Fatalf("start: %d %+v", status, doc)
	}
	if ids := doc["problems"].([]any); len(ids) != 1 || ids[0] != "p1" {
		t.Fatalf("one problem, the comment skipped: %+v", doc["problems"])
	}
	job := e.waitForJob()
	if job["state"] != "done" {
		t.Fatalf("the job finishes: %+v", job)
	}
	status, doc = e.get("/api/codegen/history")
	if status != 200 {
		t.Fatalf("history: %d %+v", status, doc)
	}
	kinds := map[string]int{}
	for _, row := range doc["history"].([]any) {
		kinds[row.(map[string]any)["kind"].(string)]++
	}
	if kinds["attempt"] != 2 || kinds["problem"] != 1 || kinds["round"] != 1 {
		t.Fatalf("two attempts, one problem, one round: %+v", kinds)
	}
	// the rejected program taught the negative network why it was rejected
	status, doc = e.get("/api/negative")
	if status != 200 {
		t.Fatalf("negative: %d %+v", status, doc)
	}
	if toFloat(doc["stats"].(map[string]any)["edge_blame_total"]) <= 0 {
		t.Fatalf("blame reached the negative network: %+v", doc["stats"])
	}
	if status, doc = e.post("/api/codegen/start", map[string]any{}); status != 400 {
		t.Fatalf("no problems is a 400: %d %+v", status, doc)
	}
}
