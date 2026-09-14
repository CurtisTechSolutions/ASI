package radixnet

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// A fake Ollama plays all four LLM roles: it writes criteria, mediates a
// garbled emission into a real call, judges the answer and demonstrates the
// task with the real tools.  The web tools point at the fake site of
// tools_test.go, so every call the agent makes really happens.

type fakeAgentLLM struct {
	server      *httptest.Server
	site        string
	nativeTools bool
	teacherOnce bool
	answer      string
	mediations  int
	proposals   int
	criteriaRaw string
	judgeRaw    string
	mediatorRaw string
}

func newFakeAgentLLM(t *testing.T, site string) *fakeAgentLLM {
	t.Helper()
	fake := &fakeAgentLLM{site: site, nativeTools: true, answer: "A cat has four legs."}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "fake:latest"}}})
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if strings.HasSuffix(r.URL.Path, "/api/chat") {
			json.NewEncoder(w).Encode(map[string]any{"message": fake.chat(body)})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"response": fake.generate(body)})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeAgentLLM) catsURL() string { return f.site + "/cats" }

func (f *fakeAgentLLM) chat(body map[string]any) map[string]any {
	messages, _ := body["messages"].([]any)
	system := ""
	tools := 0
	if list, ok := body["tools"].([]any); ok {
		tools = len(list)
	}
	toolReplies := 0
	for _, item := range messages {
		message, _ := item.(map[string]any)
		if message["role"] == "system" {
			system, _ = message["content"].(string)
		}
		if message["role"] == "tool" {
			toolReplies++
		}
	}
	call := []any{map[string]any{"function": map[string]any{
		"name": "web_fetch", "arguments": map[string]any{"url": f.catsURL()},
	}}}
	if strings.Contains(system, "tool mediator") {
		f.mediations++
		if f.mediatorRaw != "" {
			return map[string]any{"role": "assistant", "content": f.mediatorRaw}
		}
		if f.nativeTools && tools > 0 {
			return map[string]any{"role": "assistant", "content": "", "tool_calls": call}
		}
		// a model without tool calling: the JSON is asked for in the content
		raw, _ := json.Marshal(map[string]any{"tool": "web_fetch", "arguments": map[string]any{"url": f.catsURL()}})
		return map[string]any{"role": "assistant", "content": string(raw)}
	}
	if strings.Contains(system, "solve a task with the tools") {
		if toolReplies == 0 && !f.teacherOnce {
			return map[string]any{"role": "assistant", "content": "", "tool_calls": call}
		}
		return map[string]any{"role": "assistant", "content": f.answer}
	}
	return map[string]any{"role": "assistant", "content": f.answer}
}

var numberedCriterion = regexp.MustCompile(`(?m)^\d+\. `)

func (f *fakeAgentLLM) generate(body map[string]any) string {
	system, _ := body["system"].(string)
	prompt, _ := body["prompt"].(string)
	if strings.Contains(system, "judge of an attempt") { // matched first: the judge's prompt also says "acceptance criteria"
		if f.judgeRaw != "" {
			return f.judgeRaw
		}
		parts := strings.Split(prompt, "Answer:")
		answer := strings.ToLower(parts[len(parts)-1])
		good := strings.Contains(answer, "four")
		count := len(numberedCriterion.FindAllString(prompt, -1))
		if count == 0 {
			count = 2
		}
		met := make([]bool, count)
		for i := range met {
			met[i] = good
		}
		score, critique := 1, "it never says how many"
		if good {
			score, critique = 9, "fine"
		}
		raw, _ := json.Marshal(map[string]any{"met": met, "score": score, "correct": good, "critique": critique})
		return string(raw)
	}
	if strings.Contains(system, "acceptance criteria") {
		if f.criteriaRaw != "" {
			return f.criteriaRaw
		}
		raw, _ := json.Marshal(map[string]any{"criteria": []string{
			"The answer says how many legs a cat has", "The answer names four",
		}})
		return string(raw)
	}
	if strings.Contains(system, "choosing what to look into next") {
		f.proposals++
		raw, _ := json.Marshal(map[string]any{
			"question": fmt.Sprintf("Question %d about cats?", f.proposals), "seed": f.catsURL(),
		})
		return string(raw)
	}
	return "unexpected prompt"
}

// agentCase is a fake site, a fake Ollama, an untrained network and the real
// toolbox.
type agentCase struct {
	site   *httptest.Server
	fake   *fakeAgentLLM
	client LLMClient
	box    *ToolBox
	model  *Model
}

func newAgentCase(t *testing.T) *agentCase {
	t.Helper()
	site := fakeSite(t)
	fake := newFakeAgentLLM(t, site.URL)
	client, err := NewOllamaClient(fake.server.URL, "fake:latest", 20*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	box, err := DefaultToolBox(ToolOptions{AllowPrivate: true, SearchURL: site.URL + "/search?q={query}"})
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	model.Exact = true
	return &agentCase{site: site, fake: fake, client: client, box: box, model: model}
}

func (c *agentCase) trainer(t *testing.T, change func(*AgentConfig)) *AgentTrainer {
	t.Helper()
	config := DefaultAgentConfig()
	config.AgentModel, config.MaxSteps, config.ModelAttempts = "fake:latest", 2, 1
	config.NegEpochs, config.PosEpochs, config.Candidates = 1, 1, 3
	config.MaxLength = 80
	if change != nil {
		change(&config)
	}
	trainer, err := NewAgentTrainer(c.model, c.client, c.box, config)
	if err != nil {
		t.Fatal(err)
	}
	return trainer
}

func catTask() Task { return Task{ID: "t1", Prompt: "How many legs does a cat have?"} }

// -- tasks ---------------------------------------------------------------------

func TestParseTasksStringsAndObjects(t *testing.T) {
	tasks, err := ParseTasks([]any{
		"first question",
		map[string]any{"prompt": "second", "criteria": []any{"it answers"}, "answer": "yes",
			"seeds": []any{"http://example.com"}},
		map[string]any{"question": "third", "id": "own", "urls": "http://example.org"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if tasks[0].ID != "t1" || tasks[0].Prompt != "first question" {
		t.Fatalf("a bare string: %+v", tasks[0])
	}
	if tasks[1].Criteria[0] != "it answers" || tasks[1].Answer != "yes" || tasks[1].Seeds[0] != "http://example.com" {
		t.Fatalf("an object: %+v", tasks[1])
	}
	if tasks[2].ID != "own" || tasks[2].Prompt != "third" || tasks[2].Seeds[0] != "http://example.org" {
		t.Fatalf("alternative keys: %+v", tasks[2])
	}
}

func TestParseTasksRejectsRubbish(t *testing.T) {
	for _, items := range [][]any{
		{""}, {42}, {map[string]any{}}, {map[string]any{"prompt": "x", "criteria": 1}},
		{map[string]any{"prompt": "x", "answer": 3}},
		{map[string]any{"prompt": "a", "id": "same"}, map[string]any{"prompt": "b", "id": "same"}},
	} {
		if tasks, err := ParseTasks(items); err == nil {
			t.Fatalf("%+v was accepted as %+v", items, tasks)
		}
	}
}

func TestParseTaskFileFormats(t *testing.T) {
	text, err := ParseTaskFile("first\n# a comment\n\nsecond\n", ".txt")
	if err != nil || len(text) != 2 || text[1].Prompt != "second" {
		t.Fatalf("plain text: %+v %v", text, err)
	}
	lines, err := ParseTaskFile("{\"prompt\": \"a\"}\n{\"prompt\": \"b\"}\n", ".jsonl")
	if err != nil || len(lines) != 2 {
		t.Fatalf("jsonl: %+v %v", lines, err)
	}
	object, err := ParseTaskFile("{\"tasks\": [\"a\", \"b\"]}", ".json")
	if err != nil || len(object) != 2 {
		t.Fatalf("json object: %+v %v", object, err)
	}
	if _, err := ParseTaskFile("", ".txt"); err == nil {
		t.Fatal("an empty file is refused")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "tasks.txt")
	os.WriteFile(path, []byte("only one\n"), 0o644)
	loaded, err := LoadTasks(path)
	if err != nil || len(loaded) != 1 {
		t.Fatalf("from disk: %+v %v", loaded, err)
	}
}

func TestAgentConfigValidation(t *testing.T) {
	config := DefaultAgentConfig()
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*AgentConfig){
		func(c *AgentConfig) { c.Phases = nil },
		func(c *AgentConfig) { c.Phases = []string{"nobody"} },
		func(c *AgentConfig) { c.Mediation = "sometimes" },
		func(c *AgentConfig) { c.Rounds = 0 },
		func(c *AgentConfig) { c.MaxSteps = 0 },
		func(c *AgentConfig) { c.Candidates = 0 },
		func(c *AgentConfig) { c.CriteriaCount = 99 },
		func(c *AgentConfig) { c.TwoNRLPer = "epoch" },
		func(c *AgentConfig) { c.BlatantMode = "shout" },
		func(c *AgentConfig) { c.BlatantBoost = 0.5 },
		func(c *AgentConfig) { c.PassScore = 0 },
	} {
		bad := DefaultAgentConfig()
		change(&bad)
		if err := bad.Validate(); err == nil {
			t.Fatalf("%+v was accepted", bad)
		}
	}
}

// -- the four roles --------------------------------------------------------------

func TestWriteCriteriaBeforeAnythingIsAttempted(t *testing.T) {
	c := newAgentCase(t)
	criteria, err := WriteCriteria(c.client, catTask(), "fake:latest", 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(criteria) != 2 || criteria[0] != "The answer says how many legs a cat has" {
		t.Fatalf("the criteria: %+v", criteria)
	}
	// a task that carries its own keeps them, and the LLM is never asked
	own := catTask()
	own.Criteria = []string{"mine"}
	criteria, err = WriteCriteria(c.client, own, "fake:latest", 4)
	if err != nil || len(criteria) != 1 || criteria[0] != "mine" {
		t.Fatalf("its own: %+v %v", criteria, err)
	}
	// plain lines are accepted, and an unusable answer still yields a criterion
	c.fake.criteriaRaw = "the answer is specific\nthe answer names a number"
	criteria, _ = WriteCriteria(c.client, catTask(), "fake:latest", 4)
	if len(criteria) != 2 {
		t.Fatalf("plain lines: %+v", criteria)
	}
	c.fake.criteriaRaw = "   "
	criteria, _ = WriteCriteria(c.client, catTask(), "fake:latest", 4)
	if len(criteria) != 1 || !strings.Contains(criteria[0], "How many legs") {
		t.Fatalf("the fallback names the task: %+v", criteria)
	}
}

func TestMediateCallRepairsAnEmission(t *testing.T) {
	c := newAgentCase(t)
	call, err := MediateCall(c.client, "!!! garbled !!!", c.box, catTask(), "", "fake:latest", "")
	if err != nil {
		t.Fatal(err)
	}
	if call.Name != "web_fetch" || !strings.HasSuffix(call.Arguments["url"].(string), "/cats") {
		t.Fatalf("the native tool call: %+v", call)
	}
	// a model without tool calling answers with JSON in the content
	c.fake.nativeTools = false
	call, err = MediateCall(c.client, "!!!", c.box, catTask(), "", "fake:latest", "")
	if err != nil || call.Name != "web_fetch" {
		t.Fatalf("the JSON fallback: %+v %v", call, err)
	}
	// an unusable answer falls back to a network tool with the task as its argument
	c.fake.mediatorRaw = "I cannot help with that."
	call, err = MediateCall(c.client, "!!!", c.box, catTask(), "", "fake:latest", "")
	if err != nil || call.Name != "web_search" || call.Arguments["query"] != catTask().Prompt {
		t.Fatalf("the fallback call: %+v %v", call, err)
	}
}

func TestJudgeAttemptMarksEachCriterion(t *testing.T) {
	c := newAgentCase(t)
	criteria := []string{"The answer says how many legs a cat has", "The answer names four"}
	judged, err := JudgeAttempt(c.client, catTask(), criteria, "TASK: ...", "A cat has four legs.", "fake:latest")
	if err != nil {
		t.Fatal(err)
	}
	if len(judged.Met) != 2 || !judged.Met[0] || judged.Correct == nil || !*judged.Correct {
		t.Fatalf("a good answer: %+v", judged)
	}
	judged, _ = JudgeAttempt(c.client, catTask(), criteria, "TASK: ...", "I do not know.", "fake:latest")
	if judged.Correct == nil || *judged.Correct {
		t.Fatalf("a bad answer: %+v", judged)
	}
	// marks written as words are read; a list that cannot be read whole is dropped whole
	c.fake.judgeRaw = `{"met": ["yes", "no"], "score": 5, "correct": false}`
	judged, _ = JudgeAttempt(c.client, catTask(), criteria, "", "something", "fake:latest")
	if len(judged.Met) != 2 || !judged.Met[0] || judged.Met[1] {
		t.Fatalf("words: %+v", judged)
	}
	c.fake.judgeRaw = `{"met": ["yes", "perhaps"], "score": 5}`
	judged, _ = JudgeAttempt(c.client, catTask(), criteria, "", "something", "fake:latest")
	if len(judged.Met) != 0 {
		t.Fatalf("all or nothing: %+v", judged)
	}
	c.fake.judgeRaw = "not JSON at all"
	judged, _ = JudgeAttempt(c.client, catTask(), criteria, "", "something", "fake:latest")
	if judged.Critique != "the judge returned nothing usable" {
		t.Fatalf("an unusable answer is reported: %+v", judged)
	}
}

func TestDecideAgent(t *testing.T) {
	criteria := []string{"a", "b"}
	yes, no := true, false
	verdict := decideAgent(JudgedAttempt{Met: []bool{true, true}, Correct: &yes}, criteria, "", true)
	if verdict.Correct || verdict.JudgedBy != "answer" {
		t.Fatalf("no answer is never correct: %+v", verdict)
	}
	verdict = decideAgent(JudgedAttempt{Met: []bool{true, false}, Correct: &yes}, criteria, "something", true)
	if verdict.Correct || len(verdict.Issues) != 1 || verdict.Issues[0] != "b" {
		t.Fatalf("strict needs every criterion: %+v", verdict)
	}
	verdict = decideAgent(JudgedAttempt{Met: []bool{true, false}, Correct: &yes}, criteria, "something", false)
	if !verdict.Correct {
		t.Fatalf("lenient takes the judge's word: %+v", verdict)
	}
	verdict = decideAgent(JudgedAttempt{Met: []bool{true, true}}, criteria, "something", true)
	if !verdict.Correct || verdict.JudgedBy != "ollama" {
		t.Fatalf("no verdict falls back to the marks: %+v", verdict)
	}
	verdict = decideAgent(JudgedAttempt{Correct: &no}, criteria, "something", true)
	if verdict.Correct {
		t.Fatalf("a refusal stands: %+v", verdict)
	}
}

func TestTeachTaskReallyCallsTheTools(t *testing.T) {
	c := newAgentCase(t)
	steps, answer, err := TeachTask(c.client, catTask(), c.box, "fake:latest", 3, nil, "", nil, 600)
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 1 || steps[0].Call.Name != "web_fetch" || !steps[0].Result.OK {
		t.Fatalf("the teacher's call really ran: %+v", steps)
	}
	if !strings.Contains(steps[0].Result.Output, "four legs") {
		t.Fatalf("and read the page: %+v", steps[0].Result.Output)
	}
	if answer != "A cat has four legs." {
		t.Fatalf("the answer: %q", answer)
	}
	c.fake.teacherOnce = true
	steps, answer, err = TeachTask(c.client, catTask(), c.box, "fake:latest", 3, nil, "", nil, 600)
	if err != nil || len(steps) != 0 || answer == "" {
		t.Fatalf("an answer without any call: %+v %q %v", steps, answer, err)
	}
}

func TestProposeTaskTurnsAnEmissionIntoAQuestion(t *testing.T) {
	c := newAgentCase(t)
	task, err := ProposeTask(c.client, "cats cats", "fake:latest", []string{"http://example.org/a"}, nil, 3)
	if err != nil {
		t.Fatal(err)
	}
	if task.ID != "x3" || !strings.HasPrefix(task.Prompt, "Question 1 about cats") {
		t.Fatalf("the question: %+v", task)
	}
	if len(task.Seeds) != 1 || !strings.HasSuffix(task.Seeds[0], "/cats") {
		t.Fatalf("the seed: %+v", task.Seeds)
	}
}

// -- the trainer -----------------------------------------------------------------

func TestAgentRunsATaskAndLearnsFromIt(t *testing.T) {
	c := newAgentCase(t)
	trainer := c.trainer(t, nil)
	records, err := trainer.Run([]Task{catTask()})
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[string]int{}
	for _, record := range trainer.History {
		kinds[record["kind"].(string)]++
	}
	if kinds["criteria"] != 1 || kinds["task"] != 1 || kinds["round"] != 1 {
		t.Fatalf("one task, one round: %+v", kinds)
	}
	task := records[0]
	if task["kind"] != "task" || task["taught"] != true {
		t.Fatalf("the untrained network fails, so the teacher steps in: %+v", task)
	}
	if task["correct"] != true || task["solved_by"] != "teacher" {
		t.Fatalf("the teacher solves it: %+v", task)
	}
	if task["action"] != "2nrl" {
		t.Fatalf("a failure and a success: 2NRL: %+v", task)
	}
	if len(trainer.Solutions()) != 1 {
		t.Fatalf("the correct transcript is kept: %+v", trainer.Solutions())
	}
	if trainer.Model.Meta["twonrl_runs"] == nil {
		t.Fatal("the network was taught")
	}
}

func TestAgentMediationIsCountedAsSomeoneElsesCall(t *testing.T) {
	c := newAgentCase(t)
	trainer := c.trainer(t, func(config *AgentConfig) {
		config.Mediation = "always"
		config.TeachOnFailure = false
		config.MaxSteps = 1
	})
	attempt, err := trainer.SolveWithModel(catTask(), 0, "model", 1)
	if err != nil {
		t.Fatal(err)
	}
	if attempt.Calls() != 1 || attempt.OwnCalls() != 0 {
		t.Fatalf("the mediator wrote it: %+v", attempt.Steps[0])
	}
	if attempt.Steps[0].Source != "mediator" || !attempt.Steps[0].Repaired() {
		t.Fatalf("and says so: %+v", attempt.Steps[0])
	}
	if autonomy := attempt.Autonomy(); autonomy == nil || *autonomy != 0 {
		t.Fatalf("autonomy is 0: %v", autonomy)
	}
	if !strings.Contains(attempt.Text, "<tool>web_fetch") {
		t.Fatalf("the repaired call is what the network learns: %q", attempt.Text)
	}
}

func TestAgentGapOf(t *testing.T) {
	c := newAgentCase(t)
	trainer := c.trainer(t, nil)
	criteria := []string{"a", "b"}
	score := 9.0
	correct := &AgentAttempt{Answer: "four", Verdict: AgentVerdict{Correct: true, Score: &score}}
	if gap := trainer.GapOf(correct, criteria); gap != 0 {
		t.Fatalf("a correct attempt has no gap: %v", gap)
	}
	silent := &AgentAttempt{Verdict: AgentVerdict{}}
	if gap := trainer.GapOf(silent, criteria); gap != 1 {
		t.Fatalf("no answer is the whole gap: %v", gap)
	}
	low := 3.0
	poor := &AgentAttempt{Answer: "something", Verdict: AgentVerdict{Score: &low, Met: []bool{true, false}}}
	if gap := trainer.GapOf(poor, criteria); gap < 0.5 || gap > 0.51 {
		t.Fatalf("the worse of the two measures: %v", gap)
	}
	high := 5.9
	near := &AgentAttempt{Answer: "something", Verdict: AgentVerdict{Score: &high, Met: []bool{true, true}}}
	if gap := trainer.GapOf(near, criteria); gap != 0.1 {
		t.Fatalf("a failure that is only just a failure still trains: %v", gap)
	}
}

func TestAgentExploresOnItsOwn(t *testing.T) {
	c := newAgentCase(t)
	trainer := c.trainer(t, func(config *AgentConfig) { config.MaxSteps = 1 })
	records, err := trainer.Explore(2)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("two steps: %+v", len(records))
	}
	proposals := 0
	for _, record := range trainer.History {
		if record["kind"] == "proposal" {
			proposals++
			if !strings.HasPrefix(record["prompt"].(string), "Question ") {
				t.Fatalf("the LLM turned the emission into a question: %+v", record)
			}
		}
	}
	if proposals != 2 {
		t.Fatalf("one proposal per step: %d", proposals)
	}
	if records[0]["kind"] != "explore" || records[0]["step"] != 1 {
		t.Fatalf("the record: %+v", records[0])
	}
	if len(trainer.Visited) == 0 {
		t.Fatal("the pages it read are remembered")
	}
}

func TestAgentStopsBetweenTasks(t *testing.T) {
	c := newAgentCase(t)
	trainer := c.trainer(t, nil)
	trainer.Stop = func() bool { return true }
	records, err := trainer.Run([]Task{catTask()})
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		if record["kind"] == "task" {
			t.Fatalf("a stop before the first task solves none of them: %+v", record)
		}
	}
}
