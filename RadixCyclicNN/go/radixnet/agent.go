package radixnet

// Tool use: the network browses and solves on its own, an LLM sets the bar and
// teaches.
//
//	task -> acceptance criteria (LLM) -> the network works with tools -> judge (LLM)
//	     -> teach when it failed (LLM) -> 2NRL (punish the failures, reward the correct run)
//
// The network only ever emits characters, so a tool call is text it writes
// (<tool>web_fetch {"url": "..."}</tool>, see tools.go) and the answer comes
// back as text it reads.  Around that, the LLM plays four roles and never the
// one of solving the task for the network unless it has to: criteria,
// mediator, judge and teacher.
//
// The Go twin of the Python radixnet.agent: the same prompts, the same
// records and the same 2NRL.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DefaultAgentModel is the model that writes the criteria, mediates, judges
// and teaches (RADIXNET_AGENT_MODEL overrides).
func DefaultAgentModel() string {
	if named := strings.TrimSpace(os.Getenv("RADIXNET_AGENT_MODEL")); named != "" {
		return named
	}
	return DefaultTutorModel()
}

// AgentPhases and Mediation are what a configuration may ask for.
var (
	AgentPhases = []string{"teacher", "model"}
	Mediation   = []string{"repair", "always", "never"}
)

const (
	maxCriteria        = 8
	maxTranscriptChars = 6000
	// minEmission is what the cheapest path is asked for: enough for a whole
	// <tool>name {...}</tool> line.
	minEmission = 48
)

// -- tasks --------------------------------------------------------------------

// Task is something to find out: a question, optionally with criteria and a
// known answer already written down.
type Task struct {
	ID       string   `json:"id"`
	Prompt   string   `json:"prompt"`
	Criteria []string `json:"criteria"`
	Answer   string   `json:"answer"`
	Seeds    []string `json:"seeds"`
}

// ToDict is the task as decoded JSON, the shape ParseTasks reads back.
func (t Task) ToDict() map[string]any {
	criteria := t.Criteria
	if criteria == nil {
		criteria = []string{}
	}
	seeds := t.Seeds
	if seeds == nil {
		seeds = []string{}
	}
	var answer any
	if t.Answer != "" {
		answer = t.Answer
	}
	return map[string]any{"id": t.ID, "prompt": t.Prompt, "criteria": criteria, "answer": answer, "seeds": seeds}
}

// taskFrom reads one task out of a decoded JSON value.
func taskFrom(item any, index int) (Task, error) {
	switch value := item.(type) {
	case string:
		if strings.TrimSpace(value) == "" {
			return Task{}, fmt.Errorf("task %d: empty prompt", index)
		}
		return Task{ID: fmt.Sprintf("t%d", index), Prompt: strings.TrimSpace(value)}, nil
	case map[string]any:
		prompt := ""
		for _, key := range []string{"prompt", "task", "question", "goal"} {
			if text, ok := value[key].(string); ok && strings.TrimSpace(text) != "" {
				prompt = strings.TrimSpace(text)
				break
			}
		}
		if prompt == "" {
			return Task{}, fmt.Errorf("task %d: missing 'prompt'", index)
		}
		criteria, err := stringList(value["criteria"], true)
		if err != nil {
			return Task{}, fmt.Errorf("task %d: 'criteria' must be a list of strings", index)
		}
		seeds, err := stringList(firstPresent(value, "seeds", "urls"), false)
		if err != nil {
			return Task{}, fmt.Errorf("task %d: 'seeds' must be a list of URLs", index)
		}
		answer := ""
		if raw := firstPresent(value, "answer", "expected"); raw != nil {
			text, ok := raw.(string)
			if !ok {
				return Task{}, fmt.Errorf("task %d: 'answer' must be a string", index)
			}
			answer = text
		}
		id := fmt.Sprintf("t%d", index)
		if raw, ok := value["id"]; ok && raw != nil {
			if text := strings.TrimSpace(pythonString(raw)); text != "" {
				id = text
			}
		}
		return Task{ID: id, Prompt: prompt, Criteria: criteria, Answer: answer, Seeds: seeds}, nil
	}
	return Task{}, fmt.Errorf("task %d: must be a string or an object, got %s", index, jsonTypeName(item))
}

// stringList reads a list of strings, a newline-separated string (when lines
// is set) or a single string.
func stringList(raw any, lines bool) ([]string, error) {
	out := []string{}
	switch value := raw.(type) {
	case nil:
		return out, nil
	case string:
		if lines {
			for _, line := range strings.Split(value, "\n") {
				if trimmed := strings.TrimSpace(line); trimmed != "" {
					out = append(out, trimmed)
				}
			}
			return out, nil
		}
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			out = append(out, trimmed)
		}
		return out, nil
	case []any:
		for _, item := range value {
			text, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("not a string")
			}
			if trimmed := strings.TrimSpace(text); trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out, nil
	case []string:
		for _, text := range value {
			if trimmed := strings.TrimSpace(text); trimmed != "" {
				out = append(out, trimmed)
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("not a list of strings")
}

func jsonTypeName(value any) string {
	switch value.(type) {
	case nil:
		return "NoneType"
	case bool:
		return "bool"
	case float64:
		return "float"
	case string:
		return "str"
	case []any:
		return "list"
	case map[string]any:
		return "dict"
	}
	return fmt.Sprintf("%T", value)
}

// ParseTasks reads tasks from decoded JSON values (strings or objects); ids
// default to t1, t2, ...
func ParseTasks(items []any) ([]Task, error) {
	tasks := make([]Task, 0, len(items))
	seen := map[string]bool{}
	for i, item := range items {
		task, err := taskFrom(item, i+1)
		if err != nil {
			return nil, err
		}
		if seen[task.ID] {
			return nil, fmt.Errorf("duplicate task id %s", pythonRepr(task.ID))
		}
		seen[task.ID] = true
		tasks = append(tasks, task)
	}
	return tasks, nil
}

// ParseTaskFile reads a task file: .txt one question per line (# comments),
// .jsonl objects, .json a list or {"tasks": [...]}.
func ParseTaskFile(text, ext string) ([]Task, error) {
	items := []any{}
	switch strings.ToLower(ext) {
	case ".jsonl":
		for _, line := range strings.Split(text, "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			var value any
			if err := json.Unmarshal([]byte(line), &value); err != nil {
				return nil, fmt.Errorf("%v", err)
			}
			items = append(items, value)
		}
	case ".json":
		var data any
		if err := json.Unmarshal([]byte(text), &data); err != nil {
			return nil, fmt.Errorf("%v", err)
		}
		switch value := data.(type) {
		case []any:
			items = value
		case map[string]any:
			list, ok := firstPresent(value, "tasks", "problems").([]any)
			if !ok {
				return nil, fmt.Errorf("a JSON task file must hold a list or an object with a 'tasks' list")
			}
			items = list
		default:
			return nil, fmt.Errorf("a JSON task file must hold a list or an object with a 'tasks' list")
		}
	default:
		for _, line := range strings.Split(text, "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed != "" && !strings.HasPrefix(trimmed, "#") {
				items = append(items, trimmed)
			}
		}
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no tasks in the file")
	}
	return ParseTasks(items)
}

// LoadTasks reads a task file from disk.
func LoadTasks(path string) ([]Task, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseTaskFile(string(raw), filepath.Ext(path))
}

// -- one step, one attempt, one verdict ----------------------------------------

// Step is one tool call inside an attempt, and where the call came from.
type Step struct {
	Index    int
	Call     *ToolCall
	Result   *ToolResult
	Source   string // "model" | "mediator" | "teacher"
	Emission string
}

// Repaired reports whether the mediator wrote this call.
func (s *Step) Repaired() bool { return s.Source == "mediator" }

// ToDict is the step as the API and the CLI report it.
func (s *Step) ToDict() map[string]any {
	return map[string]any{
		"index": s.Index, "tool": s.Call.Name, "arguments": s.Call.Arguments, "source": s.Source,
		"ok": s.Result.OK, "output": clipString(s.Result.Output, 500), "error": nilString(s.Result.Error),
		"seconds": round4(s.Result.Seconds), "emission": clipString(s.Emission, 200),
	}
}

// AgentVerdict is what the judge decided about a finished attempt.
type AgentVerdict struct {
	Correct  bool     `json:"correct"`
	Score    *float64 `json:"score"`
	Met      []bool   `json:"met"`
	Critique string   `json:"critique"`
	Issues   []string `json:"issues"`
	JudgedBy string   `json:"judged_by"` // "ollama" | "answer" | "none"
}

// ToDict is the verdict as the API reports it.
func (v AgentVerdict) ToDict() map[string]any {
	met := v.Met
	if met == nil {
		met = []bool{}
	}
	issues := v.Issues
	if issues == nil {
		issues = []string{}
	}
	return map[string]any{
		"correct": v.Correct, "score": v.Score, "met": met, "critique": v.Critique,
		"issues": issues, "judged_by": v.JudgedBy,
	}
}

// Failure is a failed transcript and how badly it failed (Gap in [0, 1],
// 1 = it got nothing right).
type Failure struct {
	Text   string
	Gap    float64
	Task   string
	Source string
}

// AgentAttempt is one run at a task: the steps taken, the answer given, the
// transcript learned and the verdict.
type AgentAttempt struct {
	Index   int
	Source  string // "model" | "teacher"
	Steps   []*Step
	Answer  string
	Text    string
	Verdict AgentVerdict
	Seconds float64
}

// Calls is how many tools the attempt called.
func (a *AgentAttempt) Calls() int { return len(a.Steps) }

// OwnCalls are the calls the network wrote by itself (the mediator repaired
// the rest).
func (a *AgentAttempt) OwnCalls() int {
	own := 0
	for _, step := range a.Steps {
		if step.Source == "model" {
			own++
		}
	}
	return own
}

// Autonomy is the share of calls the network wrote itself (nil with no calls).
func (a *AgentAttempt) Autonomy() *float64 {
	if len(a.Steps) == 0 {
		return nil
	}
	value := float64(a.OwnCalls()) / float64(len(a.Steps))
	return &value
}

// Feedback is why the attempt was rejected, in the words the teacher sees.
func (a *AgentAttempt) Feedback() string {
	parts := []string{}
	if a.Answer == "" {
		parts = append(parts, "It never gave an answer.")
	}
	failed := []string{}
	for _, step := range a.Steps {
		if !step.Result.OK {
			failed = append(failed, step.Call.Name+": "+step.Result.Error)
		}
	}
	if len(failed) > 0 {
		parts = append(parts, "Failed tool calls: "+strings.Join(firstN(failed, 4), "; "))
	}
	if len(a.Verdict.Issues) > 0 {
		parts = append(parts, "Unmet criteria: "+strings.Join(firstN(a.Verdict.Issues, 6), "; "))
	}
	if a.Verdict.Critique != "" {
		parts = append(parts, "Judge: "+a.Verdict.Critique)
	}
	if len(parts) == 0 {
		return "It was judged incorrect."
	}
	return strings.Join(parts, "\n")
}

// ToDict is the attempt as the API and the CLI report it.
func (a *AgentAttempt) ToDict() map[string]any {
	steps := make([]map[string]any, 0, len(a.Steps))
	for _, step := range a.Steps {
		steps = append(steps, step.ToDict())
	}
	return map[string]any{
		"index": a.Index, "source": a.Source, "answer": nilString(a.Answer), "calls": a.Calls(),
		"own_calls": a.OwnCalls(), "autonomy": a.Autonomy(), "steps": steps, "verdict": a.Verdict.ToDict(),
		"correct": a.Verdict.Correct, "text_chars": runeLen(a.Text), "seconds": a.Seconds,
	}
}

// nilString is a string as JSON null when it is empty, the way Python reports
// an optional field.
func nilString(text string) any {
	if text == "" {
		return nil
	}
	return text
}

func round4(value float64) float64 { return math.Round(value*10000) / 10000 }

// clipTranscript is the tail of a long transcript (what matters for the next
// step).
func clipTranscript(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return "... " + string(runes[len(runes)-limit:])
}

// -- the four LLM roles --------------------------------------------------------

const criteriaSystem = "You write acceptance criteria for a task that will be solved by browsing the web with " +
	"tools. Write at most %d short, checkable statements that any correct answer must satisfy: what the answer has " +
	"to contain, how specific it has to be, and what would make it wrong. Judge only the answer, never the wording " +
	"or the effort. Do not solve the task and do not state the answer. Reply with JSON only, of the form " +
	"{\"criteria\": [\"<statement>\", ...]}."

// WriteCriteria asks for the statements a correct answer must satisfy, before
// anything is attempted.
//
// A task that already carries criteria keeps them; an LLM that answers with
// nothing usable falls back to one criterion naming the task itself, so the
// judge always has something objective to mark against.
func WriteCriteria(client LLMClient, task Task, model string, count int) ([]string, error) {
	if len(task.Criteria) > 0 {
		return append([]string{}, task.Criteria...), nil
	}
	count = clampInt(count, 1, maxCriteria)
	raw, err := client.Generate("Task: "+task.Prompt+"\n\nWrite the criteria now.", LLMOptions{
		System: fmt.Sprintf(criteriaSystem, count), Model: model, JSON: true, Temperature: 0.2,
	})
	if err != nil {
		return nil, err
	}
	data := loadsLenient(raw)
	var items []any
	switch value := data.(type) {
	case map[string]any:
		if list, ok := value["criteria"].([]any); ok {
			items = list
		} else {
			keys := make([]string, 0, len(value))
			for key := range value {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				items = append(items, value[key])
			}
		}
	case []any:
		items = value
	}
	criteria := []string{}
	for _, item := range items {
		text := ""
		switch value := item.(type) {
		case string:
			text = value
		case map[string]any:
			text = firstString(value, "criterion", "text")
		}
		text = collapse(text)
		if text != "" && !contains(criteria, text) {
			criteria = append(criteria, text)
		}
	}
	if len(criteria) == 0 {
		criteria = ParseLines(raw, count)
	}
	if len(criteria) == 0 {
		criteria = []string{"The answer states, correctly and specifically, what the task asked for: " + task.Prompt}
	}
	return firstN(criteria, count), nil
}

const mediatorSystem = "You are the tool mediator for a very small character-level neural network that is learning " +
	"to use tools. It writes its intentions as raw text, which is often garbled, truncated or empty. Your job is to " +
	"turn what it wrote into exactly ONE valid tool call that makes progress on the task — never to solve the task " +
	"yourself. Read its text charitably: if it reaches for a search, search for what it seems to want; if it names " +
	"a URL, open that URL; if it is unreadable, choose the call that the task and the transcript so far make " +
	"obvious. Never repeat a call that is already in the transcript. Reply with JSON only, of the form " +
	"{\"tool\": \"<name>\", \"arguments\": {...}}."

const teacherAgentSystem = "You solve a task with the tools you are given, so that a very small neural network can " +
	"learn from your example. Work the way it must: call one tool at a time, read the result, and call another if " +
	"you need to. Use the tools for anything factual — never answer from memory alone. When you know the answer, " +
	"stop calling tools and reply with the answer itself, one short sentence, nothing else."

const judgeAgentSystem = "You are the judge of an attempt at a task by a very small neural network that browses " +
	"with tools. You are given the acceptance criteria written before the attempt, the transcript of the tool calls " +
	"and their results, and the answer. Mark each criterion strictly: met only if the answer really satisfies it — " +
	"an answer that is empty, vague, off-topic, or contradicted by the transcript meets nothing. Never reward " +
	"effort, only the answer. Reply with JSON only, of the form {\"met\": [true, false, ...] with one entry per " +
	"criterion in order, \"score\": <0-10>, \"correct\": true or false, \"critique\": \"<one sentence naming the " +
	"worst flaw, or 'none'>\"}."

const proposeSystem = "A very small neural network that browses the web is choosing what to look into next, and it " +
	"writes its intentions as raw, often garbled text. Turn what it wrote into ONE concrete question that can be " +
	"settled by browsing and answered in a sentence. Follow whatever it seems to be reaching for; when it is " +
	"unreadable, ask something that follows on from what it has already looked at, not a repeat of it. Prefer a " +
	"question about one of the unvisited pages when they fit. Reply with JSON only, of the form " +
	"{\"question\": \"<the question>\", \"seed\": \"<a URL to start from, or empty>\"}."

// chatClient is a provider that can hold a conversation; toolCallingClient one
// that can be handed tool schemas (Ollama).  The agent duck-types on both, as
// the Python one does.
type chatClient interface {
	Chat(messages []map[string]any, o LLMOptions) (string, error)
}

type toolCallingClient interface {
	ChatMessage(messages []map[string]any, o LLMOptions, tools []map[string]any) (map[string]any, error)
}

func toolsBlock(box *ToolBox) string { return "The tools are:\n" + box.Catalogue() }

// chatOnce is one assistant message, using tool calling where the provider has
// it and JSON mode where it does not.
func chatOnce(client LLMClient, messages []map[string]any, tools []map[string]any, o LLMOptions) (map[string]any, error) {
	if caller, ok := client.(toolCallingClient); ok {
		message, err := caller.ChatMessage(messages, o, tools)
		if err == nil {
			return message, nil
		}
		if len(tools) == 0 {
			return nil, err
		}
	}
	talker, ok := client.(chatClient)
	if !ok {
		return nil, llmErrorf("%s cannot hold a conversation", client.Provider())
	}
	json := o
	json.JSON = true
	content, err := talker.Chat(messages, json)
	if err != nil {
		return nil, err
	}
	return map[string]any{"role": "assistant", "content": content}, nil
}

// callFromMessage reads one call out of Ollama's native tool_calls, or out of
// a JSON answer in the content.
func callFromMessage(message map[string]any, box *ToolBox) *ToolCall {
	if calls, ok := message["tool_calls"].([]any); ok {
		for _, entry := range calls {
			object, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			function, ok := object["function"].(map[string]any)
			if !ok {
				continue
			}
			name := strings.TrimSpace(pythonString(function["name"]))
			arguments := function["arguments"]
			if text, ok := arguments.(string); ok {
				arguments = loadsLenient(text)
			}
			if name != "" && box.Has(name) {
				object, _ := arguments.(map[string]any)
				if object == nil {
					object = map[string]any{}
				}
				return &ToolCall{Name: name, Arguments: object}
			}
		}
	}
	content, _ := message["content"].(string)
	if data, ok := loadsLenient(content).(map[string]any); ok {
		name := strings.TrimSpace(pythonString(firstPresent(data, "tool", "name", "function")))
		arguments := firstPresent(data, "arguments", "args", "parameters")
		if text, ok := arguments.(string); ok {
			arguments = loadsLenient(text)
		}
		if name != "" && name != "None" && box.Has(name) {
			object, _ := arguments.(map[string]any)
			if object == nil {
				object = map[string]any{}
			}
			return &ToolCall{Name: name, Arguments: object}
		}
	}
	return nil
}

// MediateCall turns the network's unusable emission into one valid call - the
// provider's tool-calling API, then JSON, then a guess.
//
// It fails only when the LLM cannot be reached; an answer that names no known
// tool falls back to the first network tool with the task as its argument, so
// the loop always makes a move.
func MediateCall(client LLMClient, emission string, box *ToolBox, task Task, transcript, model, hint string) (*ToolCall, error) {
	written := strings.TrimSpace(emission)
	if written == "" {
		written = "(nothing)"
	} else {
		written = clipString(written, 500)
	}
	seen := clipTranscript(transcript, 2000)
	if seen == "" {
		seen = "(nothing yet)"
	}
	user := "Task: " + task.Prompt + "\n\n" + toolsBlock(box) + "\n\nTranscript so far:\n" + seen +
		"\n\nWhat the network just wrote:\n" + written + "\n\n"
	if hint != "" {
		user += hint + "\n\n"
	}
	user += "Give the one tool call to make now."
	messages := []map[string]any{
		{"role": "system", "content": mediatorSystem},
		{"role": "user", "content": user},
	}
	options := LLMOptions{Model: model, Temperature: 0.2}
	var call *ToolCall
	if caller, ok := client.(toolCallingClient); ok {
		if message, err := caller.ChatMessage(messages, options, box.Schemas()); err == nil {
			call = callFromMessage(message, box)
		}
	}
	if call == nil { // a model without tool calling: ask for the JSON directly
		talker, ok := client.(chatClient)
		if !ok {
			return nil, llmErrorf("%s cannot hold a conversation", client.Provider())
		}
		json := options
		json.JSON = true
		raw, err := talker.Chat(messages, json)
		if err != nil {
			return nil, err
		}
		call = callFromMessage(map[string]any{"content": raw}, box)
	}
	if call == nil {
		var fallback *Tool
		for _, tool := range box.Tools() {
			if tool.Network {
				fallback = tool
				break
			}
		}
		if fallback == nil && box.Len() > 0 {
			fallback = box.Tools()[0]
		}
		if fallback == nil {
			return &ToolCall{Arguments: map[string]any{}, Error: "no tools are installed"}, nil
		}
		arguments := map[string]any{}
		for _, param := range fallback.Params {
			if param.Required {
				arguments[param.Name] = task.Prompt
				break
			}
		}
		call = &ToolCall{Name: fallback.Name, Arguments: arguments}
	}
	tool, err := box.Get(call.Name)
	if err != nil {
		call.Error = err.Error()
		return call, nil
	}
	values, err := tool.Coerce(call.Arguments)
	if err != nil {
		call.Error = err.Error() // reported on the call, executed as a failure
		return call, nil
	}
	call.Arguments = values
	return call, nil
}

var (
	trueWords  = []string{"true", "yes", "y", "pass", "passed", "met", "ok", "correct", "1"}
	falseWords = []string{"false", "no", "n", "fail", "failed", "unmet", "not_met", "missing", "incorrect", "0"}
)

// criterionMark reads one criterion's mark: a bool, a number, or a word the LLM used
// for it (ok false = unreadable).
func criterionMark(value any) (bool, bool) {
	switch typed := value.(type) {
	case bool:
		return typed, true
	case float64:
		return typed != 0, true
	case map[string]any:
		for _, key := range []string{"met", "ok", "pass", "passed", "result", "verdict"} {
			if inner, ok := typed[key]; ok {
				return criterionMark(inner)
			}
		}
		return false, false
	case string:
		word := strings.ToLower(strings.Trim(strings.TrimSpace(typed), "."))
		if contains(trueWords, word) {
			return true, true
		}
		if contains(falseWords, word) {
			return false, true
		}
	}
	return false, false
}

// marksOf is the per-criterion marks of a judge's answer, or nothing when any
// of them cannot be read.
//
// All or nothing on purpose: a partly readable list would silently misalign
// with the criteria it is supposed to mark.
func marksOf(value any) []bool {
	list, ok := value.([]any)
	if !ok || len(list) == 0 {
		return nil
	}
	marks := make([]bool, 0, len(list))
	for _, item := range list {
		mark, ok := criterionMark(item)
		if !ok {
			return nil
		}
		marks = append(marks, mark)
	}
	return marks
}

// JudgedAttempt is what the judge said about a finished attempt.
type JudgedAttempt struct {
	Met      []bool
	Score    *float64
	Correct  *bool
	Critique string
}

// JudgeAttempt marks a finished attempt against the criteria written before it
// started.
func JudgeAttempt(client LLMClient, task Task, criteria []string, transcript, answer, model string) (JudgedAttempt, error) {
	lines := make([]string, 0, len(criteria))
	for i, criterion := range criteria {
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, criterion))
	}
	numbered := strings.Join(lines, "\n")
	if numbered == "" {
		numbered = "1. The answer is correct."
	}
	given := answer
	if given == "" {
		given = "(no answer was given)"
	}
	user := "Task: " + task.Prompt + "\n\nAcceptance criteria:\n" + numbered + "\n\nTranscript:\n" +
		clipTranscript(transcript, maxTranscriptChars) + "\n\nAnswer: " + given + "\n\n"
	if task.Answer != "" {
		user += "The known correct answer is: " + task.Answer + "\n\n"
	}
	count := len(criteria)
	if count == 0 {
		count = 1
	}
	user += fmt.Sprintf("Return the JSON with exactly %d entries in \"met\".", count)
	raw, err := client.Generate(user, LLMOptions{System: judgeAgentSystem, Model: model, JSON: true, Temperature: 0.1})
	if err != nil {
		return JudgedAttempt{}, err
	}
	data, ok := loadsLenient(raw).(map[string]any)
	if !ok {
		return JudgedAttempt{Critique: "the judge returned nothing usable"}, nil
	}
	out := JudgedAttempt{Met: marksOf(firstPresent(data, "met", "criteria_met", "results"))}
	if number, err := toNumber(firstPresent(data, "score", "rating")); err == nil {
		value := math.Max(0, math.Min(10, number))
		out.Score = &value
	}
	switch value := firstPresent(data, "correct", "verdict").(type) {
	case bool:
		out.Correct = &value
	case string:
		word := strings.ToLower(strings.TrimSpace(value))
		decided := word == "true" || word == "pass" || word == "yes" || word == "correct"
		out.Correct = &decided
	}
	out.Critique = collapse(pythonString(firstPresent(data, "critique", "reason")))
	if out.Critique == "None" {
		out.Critique = ""
	}
	return out, nil
}

// decideAgent turns the judge's answer into a verdict; no answer is never
// correct, and strict needs every criterion met.
func decideAgent(judged JudgedAttempt, criteria []string, answer string, strict bool) AgentVerdict {
	met := judged.Met
	issues := []string{}
	for i, criterion := range criteria {
		if i < len(met) && !met[i] {
			issues = append(issues, criterion)
		}
	}
	if answer == "" {
		critique := judged.Critique
		if critique == "" {
			critique = "no answer was given"
		}
		unmet := issues
		if len(unmet) == 0 {
			unmet = append([]string{}, criteria...)
		}
		return AgentVerdict{Correct: false, Score: judged.Score, Met: met, Critique: critique,
			Issues: unmet, JudgedBy: "answer"}
	}
	allMet := len(met) > 0 && len(met) >= len(criteria)
	for _, ok := range met {
		if !ok {
			allMet = false
		}
	}
	correct := false
	if judged.Correct != nil {
		correct = *judged.Correct
	} else if len(met) > 0 {
		correct = allMet
	} else {
		correct = judged.Score != nil && *judged.Score >= 6.0
	}
	if strict && len(met) > 0 {
		correct = correct && allMet
	}
	judgedBy := "none"
	if len(judged.Met) > 0 || judged.Score != nil {
		judgedBy = "ollama"
	}
	return AgentVerdict{Correct: correct, Score: judged.Score, Met: met, Critique: judged.Critique,
		Issues: issues, JudgedBy: judgedBy}
}

// TeachTask has the LLM solve the task with the real tools; it returns the
// steps it took and the answer it reached.
//
// Every call is executed for real, so the demonstration the network learns
// from is a transcript of things that actually happened - not an invented one.
func TeachTask(
	client LLMClient, task Task, box *ToolBox, model string, maxSteps int, criteria []string,
	feedback string, external func(func() error) error, observationChars int,
) ([]*Step, string, error) {
	run := external
	if run == nil {
		run = func(fn func() error) error { return fn() }
	}
	listed := ""
	for _, criterion := range criteria {
		listed += "- " + criterion + "\n"
	}
	user := "Task: " + task.Prompt + "\n\n" + toolsBlock(box) + "\n"
	if listed != "" {
		user += "\nA correct answer must satisfy:\n" + listed
	}
	if feedback != "" {
		user += "\nAn earlier attempt failed:\n" + feedback + "\n"
	}
	if len(task.Seeds) > 0 {
		user += "\nStart from: " + strings.Join(task.Seeds, ", ") + "\n"
	}
	user += "\nMake the first tool call now."
	messages := []map[string]any{
		{"role": "system", "content": teacherAgentSystem},
		{"role": "user", "content": user},
	}
	schemas := box.Schemas()
	steps := []*Step{}
	if maxSteps < 1 {
		maxSteps = 1
	}
	for index := 0; index < maxSteps; index++ {
		var message map[string]any
		if err := run(func() error {
			out, err := chatOnce(client, messages, schemas, LLMOptions{Model: model, Temperature: 0.3})
			message = out
			return err
		}); err != nil {
			return steps, "", err
		}
		call := callFromMessage(message, box)
		content := strings.TrimSpace(pythonString(message["content"]))
		if content == "None" {
			content = ""
		}
		if call == nil {
			if content != "" {
				return steps, collapse(content), nil
			}
			if len(steps) > 0 {
				break
			}
			messages = append(messages, map[string]any{"role": "user", "content": "Make a tool call now, or give the answer."})
			continue
		}
		var result *ToolResult
		if err := run(func() error { result = box.Run(call); return nil }); err != nil {
			return steps, "", err
		}
		steps = append(steps, &Step{Index: index, Call: call, Result: result, Source: "teacher"})
		if _, ok := message["role"]; ok {
			messages = append(messages, message)
		} else {
			messages = append(messages, map[string]any{"role": "assistant", "content": content})
		}
		observation := result.Output
		if !result.OK {
			observation = "ERROR: " + result.Error
		}
		limit := observationChars * 2
		if limit < 100 {
			limit = 100
		}
		messages = append(messages, map[string]any{"role": "tool", "content": clipString(observation, limit)})
	}
	answer := ""
	if err := run(func() error {
		talker, ok := client.(chatClient)
		if !ok {
			return llmErrorf("%s cannot hold a conversation", client.Provider())
		}
		final := append(append([]map[string]any{}, messages...),
			map[string]any{"role": "user", "content": "Give the answer now, one short sentence, nothing else."})
		out, err := talker.Chat(final, LLMOptions{Model: model, Temperature: 0.2})
		answer = collapse(out)
		return err
	}); err != nil {
		return steps, "", err
	}
	return steps, answer, nil
}

// ProposeTask turns the network's own emission into one concrete, answerable
// question (exploration picks its own tasks).
func ProposeTask(client LLMClient, emission, model string, frontier, visited []string, index int) (Task, error) {
	unvisited := firstN(frontier, 12)
	seen := visited
	if len(seen) > 8 {
		seen = seen[len(seen)-8:]
	}
	listed := func(title string, urls []string) string {
		if len(urls) == 0 {
			return ""
		}
		out := title + ":\n"
		for _, url := range urls {
			out += "- " + url + "\n"
		}
		return out + "\n"
	}
	written := strings.TrimSpace(emission)
	if written == "" {
		written = "(nothing)"
	} else {
		written = clipString(written, 500)
	}
	user := "What the network wrote:\n" + written + "\n\n" +
		listed("Unvisited pages it has come across", unvisited) +
		listed("Pages it has already read", seen) + "Give the question now."
	raw, err := client.Generate(user, LLMOptions{System: proposeSystem, Model: model, JSON: true, Temperature: 0.8})
	if err != nil {
		return Task{}, err
	}
	question, seed := "", ""
	if data, ok := loadsLenient(raw).(map[string]any); ok {
		question = collapse(pythonString(firstPresent(data, "question", "task", "prompt")))
		if question == "None" {
			question = ""
		}
		seed = strings.TrimSpace(pythonString(firstPresent(data, "seed", "url")))
		if seed == "None" {
			seed = ""
		}
	}
	if question == "" {
		if lines := ParseLines(raw, 1); len(lines) > 0 {
			question = lines[0]
		}
	}
	if question == "" {
		if len(unvisited) > 0 {
			question = "What can be found out about " + unvisited[0] + "?"
		} else {
			question = "What is on the front page of a well-known encyclopaedia?"
		}
	}
	task := Task{ID: fmt.Sprintf("x%d", index), Prompt: question}
	if strings.HasPrefix(seed, "http") {
		task.Seeds = []string{seed}
	}
	return task, nil
}

// -- configuration -------------------------------------------------------------

// AgentConfig is how the loop runs: who attempts, how far, how strictly it is
// judged, and how hard it is taught.
type AgentConfig struct {
	AgentModel       string   `json:"agent_model"`
	JudgeModel       string   `json:"judge_model"`
	Provider         string   `json:"provider"`
	Phases           []string `json:"phases"`
	Rounds           int      `json:"rounds"`
	MaxSteps         int      `json:"max_steps"`      // tool calls per attempt
	ModelAttempts    int      `json:"model_attempts"` // attempts by the network per task
	TeacherAttempts  int      `json:"teacher_attempts"`
	FirstDijkstra    bool     `json:"first_attempt_dijkstra"` // the first attempt is the beam search, not a sample
	Candidates       int      `json:"candidates"`             // continuations offered per step before the mediator steps in
	Temperature      float64  `json:"temperature"`
	MaxLength        int      `json:"max_length"` // characters the network writes per step
	Mediation        string   `json:"mediation"`  // "repair" | "always" | "never"
	CriteriaCount    int      `json:"criteria_count"`
	Strict           bool     `json:"strict"` // every criterion must be met
	UseJudge         bool     `json:"use_judge"`
	TeachOnFailure   bool     `json:"teach_on_failure"`
	ObservationChars int      `json:"observation_chars"`
	TwoNRLPer        string   `json:"twonrl_per"` // "task" | "round"
	Replay           bool     `json:"replay"`
	ReplayLimit      int      `json:"replay_limit"`
	ReadReward       bool     `json:"read_reward"`  // also fine-tune on the text of the pages that were read
	BlatantMode      string   `json:"blatant_mode"` // "none" | "fail_invert" | "activation" | "state"
	BlatantMargin    float64  `json:"blatant_margin"`
	BlatantBoost     float64  `json:"blatant_boost"`
	PassScore        float64  `json:"pass_score"` // the judge's 0-10 score a correct answer is expected to reach
	NegEpochs        int      `json:"neg_epochs"`
	PosEpochs        int      `json:"pos_epochs"`
	Strength         float64  `json:"strength"`
	Seed             int64    `json:"seed"`
}

// DefaultAgentConfig mirrors the Python defaults.
func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		AgentModel: DefaultAgentModel(), Provider: ProviderOllama, Phases: []string{"model"}, Rounds: 1,
		MaxSteps: 6, ModelAttempts: 2, TeacherAttempts: 1, FirstDijkstra: true, Candidates: 5, Temperature: 1,
		MaxLength: 200, Mediation: "repair", CriteriaCount: 4, Strict: true, UseJudge: true,
		TeachOnFailure: true, ObservationChars: 600, TwoNRLPer: "task", Replay: true, ReplayLimit: 64,
		BlatantMode: "fail_invert", BlatantMargin: 0.5, BlatantBoost: 4, PassScore: 6, NegEpochs: 2,
		PosEpochs: 3, Strength: 1,
	}
}

// Validate resolves the provider and checks the settings.
func (c *AgentConfig) Validate() error {
	if len(c.Phases) == 0 {
		return fmt.Errorf("phases must be a non-empty subset of %s", strings.Join(AgentPhases, ", "))
	}
	for _, phase := range c.Phases {
		if !contains(AgentPhases, phase) {
			return fmt.Errorf("phases must be a non-empty subset of %s", strings.Join(AgentPhases, ", "))
		}
	}
	provider, err := NormaliseProvider(c.Provider)
	if err != nil {
		return fmt.Errorf("provider must be one of %s", strings.Join(Providers(), ", "))
	}
	c.Provider = provider
	if strings.TrimSpace(c.AgentModel) == "" {
		c.AgentModel = DefaultProviderModel(c.Provider)
	}
	if !contains(Mediation, c.Mediation) {
		return fmt.Errorf("mediation must be one of %s (got %s)", strings.Join(Mediation, ", "), pythonRepr(c.Mediation))
	}
	if c.Rounds < 1 {
		return fmt.Errorf("rounds must be >= 1")
	}
	if c.MaxSteps < 1 {
		return fmt.Errorf("max_steps must be >= 1")
	}
	if c.ModelAttempts < 1 || c.TeacherAttempts < 1 {
		return fmt.Errorf("model_attempts and teacher_attempts must be >= 1")
	}
	if c.MaxLength < 1 {
		return fmt.Errorf("max_length must be >= 1")
	}
	if c.Candidates < 1 {
		return fmt.Errorf("candidates must be >= 1")
	}
	if c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.CriteriaCount < 1 || c.CriteriaCount > maxCriteria {
		return fmt.Errorf("criteria_count must be between 1 and %d", maxCriteria)
	}
	if c.TwoNRLPer != "task" && c.TwoNRLPer != "round" {
		return fmt.Errorf("twonrl_per must be 'task' or 'round'")
	}
	if !contains(BlatantModes, c.BlatantMode) {
		return fmt.Errorf("blatant_mode must be one of %s (got %s)", strings.Join(BlatantModes, ", "), pythonRepr(c.BlatantMode))
	}
	if c.BlatantMargin <= 0 {
		return fmt.Errorf("blatant_margin must be > 0")
	}
	if c.BlatantBoost < 1 {
		return fmt.Errorf("blatant_boost must be >= 1")
	}
	if c.PassScore <= 0 || c.PassScore > 10 {
		return fmt.Errorf("pass_score must be in (0, 10]")
	}
	if c.ObservationChars < 0 || c.ReplayLimit < 0 {
		return fmt.Errorf("observation_chars and replay_limit must be >= 0")
	}
	if c.NegEpochs < 0 || c.PosEpochs < 0 || c.Strength < 0 {
		return fmt.Errorf("epochs and strength must be >= 0")
	}
	return nil
}

// judgeModelName is the model the judge answers with.
func (c *AgentConfig) judgeModelName() string {
	if strings.TrimSpace(c.JudgeModel) != "" {
		return c.JudgeModel
	}
	return c.AgentModel
}

// -- the trainer ---------------------------------------------------------------

// AgentTrainer runs tasks (or self-chosen exploration) through tools and
// teaches the network with 2NRL.
type AgentTrainer struct {
	Model   *Model
	Client  LLMClient
	ToolBox *ToolBox
	Config  AgentConfig
	// External wraps the slow outside work (tool calls, LLM calls) so a server
	// can release its model lock around it.
	External func(func() error) error
	Progress func(map[string]any)
	Stop     func() bool

	History  []map[string]any
	Frontier []string
	Visited  []string

	replay   []string
	solved   map[string]string
	criteria map[string][]string
	pages    []string
	rng      *MT19937
}

// NewAgentTrainer validates the configuration and builds the loop.
func NewAgentTrainer(model *Model, client LLMClient, box *ToolBox, config AgentConfig) (*AgentTrainer, error) {
	if model == nil {
		return nil, fmt.Errorf("a model to teach is required")
	}
	if box == nil || box.Len() == 0 {
		return nil, fmt.Errorf("the toolbox is empty: there is nothing for the network to call")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &AgentTrainer{
		Model: model, Client: client, ToolBox: box, Config: config,
		solved: map[string]string{}, criteria: map[string][]string{}, rng: NewMT19937(config.Seed),
	}, nil
}

func (t *AgentTrainer) external(fn func() error) error {
	if t.External == nil {
		return fn()
	}
	return t.External(fn)
}

func (t *AgentTrainer) stopped() bool { return t.Stop != nil && t.Stop() }

func (t *AgentTrainer) emit(record map[string]any) {
	t.History = append(t.History, record)
	if t.Progress != nil {
		t.Progress(record)
	}
}

func (t *AgentTrainer) modelName(judge bool) string {
	if judge {
		return t.Config.judgeModelName()
	}
	return t.Config.AgentModel
}

// Solutions are the correct transcripts found so far, by task id.
func (t *AgentTrainer) Solutions() map[string]string {
	out := make(map[string]string, len(t.solved))
	for id, text := range t.solved {
		out[id] = text
	}
	return out
}

// noteURLs keeps the pages a tool turned up as the frontier of the exploration.
func (t *AgentTrainer) noteURLs(result *ToolResult) {
	found := []string{}
	for _, key := range []string{"results", "links"} {
		items, _ := result.Meta[key].([]map[string]any)
		if items == nil {
			if list, ok := result.Meta[key].([]any); ok {
				for _, item := range list {
					if object, ok := item.(map[string]any); ok {
						items = append(items, object)
					}
				}
			}
		}
		for _, item := range items {
			if url, ok := item["url"].(string); ok && strings.HasPrefix(url, "http") {
				found = append(found, url)
			}
		}
	}
	if url, ok := result.Meta["url"].(string); ok && strings.HasPrefix(url, "http") && !contains(t.Visited, url) {
		t.Visited = append(t.Visited, url)
	}
	for _, candidate := range found {
		if !contains(t.Visited, candidate) && !contains(t.Frontier, candidate) {
			t.Frontier = append(t.Frontier, candidate)
		}
	}
	if len(t.Frontier) > 200 {
		t.Frontier = t.Frontier[len(t.Frontier)-200:]
	}
	if t.Config.ReadReward && result.OK && result.Meta["chars"] != nil {
		if page := strings.TrimSpace(result.Output); runeLen(page) >= 3 {
			t.pages = append(t.pages, page)
			if len(t.pages) > 32 {
				t.pages = t.pages[len(t.pages)-32:]
			}
		}
	}
}

// CriteriaFor is the acceptance criteria of a task, written once by the LLM
// and remembered.
func (t *AgentTrainer) CriteriaFor(task Task) ([]string, error) {
	if known, ok := t.criteria[task.ID]; ok {
		return known, nil
	}
	var criteria []string
	switch {
	case len(task.Criteria) > 0:
		criteria = append([]string{}, task.Criteria...)
	case !t.Config.UseJudge:
		criteria = []string{"The answer states what the task asked for: " + task.Prompt}
	default:
		if err := t.external(func() error {
			out, err := WriteCriteria(t.Client, task, t.modelName(false), t.Config.CriteriaCount)
			criteria = out
			return err
		}); err != nil {
			return nil, err
		}
	}
	t.criteria[task.ID] = criteria
	return criteria, nil
}

// emissions are what the network offers to write next, most likely first.
//
// The first attempt is deterministic: the beam search returns its Candidates
// most likely continuations, and the caller takes the first one that is a
// usable call.  A single cheapest path is no good here: Dijkstra accepts
// reaching END as a goal, so it happily answers with a couple of characters
// that can never be a whole call.  Later attempts sample instead, which is
// where the variety comes from.
func (t *AgentTrainer) emissions(transcript string, attemptIndex int) ([]string, error) {
	cfg := t.Config
	prefix := clipTranscript(transcript, 2000)
	wanted := minEmission
	if cfg.MaxLength < wanted {
		wanted = cfg.MaxLength
	}
	texts := []string{}
	if attemptIndex == 0 && cfg.FirstDijkstra {
		prediction, err := t.Model.Predict(prefix, PredictOptions{
			Length: wanted, Mode: "beam", K: cfg.Candidates, MaxLength: cfg.MaxLength,
		})
		if err != nil {
			return nil, err
		}
		if len(prediction.Top) > 0 {
			for _, result := range prediction.Top {
				texts = append(texts, result.Text)
			}
		} else {
			texts = append(texts, prediction.Text)
		}
	} else {
		for i := 0; i < cfg.Candidates; i++ {
			prediction, err := t.Model.Predict(prefix, PredictOptions{
				Length: 1, Mode: "sample", Temperature: cfg.Temperature, MaxLength: cfg.MaxLength,
			})
			if err != nil {
				return nil, err
			}
			texts = append(texts, prediction.Text)
		}
	}
	offered := []string{}
	seen := map[string]bool{}
	for _, text := range texts {
		if seen[text] || strings.TrimSpace(text) == "" {
			continue
		}
		seen[text] = true
		offered = append(offered, text)
	}
	// a candidate that closed its tag wrote a whole call; an unterminated one only started one
	sort.SliceStable(offered, func(i, j int) bool {
		return whole(offered[i]) && !whole(offered[j])
	})
	return offered, nil
}

// whole reports whether an emission closed the tag it opened.
func whole(text string) bool {
	return strings.Contains(text, CallClose) || strings.Contains(text, AnswerClose)
}

// nextCall is one step's move: what the network wrote, repaired if it has to be.
//
// Every candidate it offers is tried in turn; the first usable call (or an
// answer it gives instead) is the network's own move.  Only when none of them
// can be used does the mediator step in, on the best candidate.
func (t *AgentTrainer) nextCall(task Task, transcript string, attemptIndex int) (*ToolCall, string, string, string, error) {
	cfg := t.Config
	candidates := []string{}
	if cfg.Mediation != "always" {
		out, err := t.emissions(transcript, attemptIndex)
		if err != nil {
			return nil, "", "", "", err
		}
		candidates = out
	}
	var broken *ToolCall
	for _, emission := range candidates {
		answerAt := strings.Index(emission, AnswerOpen)
		callAt := strings.Index(emission, CallOpen)
		if answerAt >= 0 && (callAt < 0 || answerAt < callAt) {
			if answer := FindAnswer(emission); answer != "" {
				return nil, "model", emission, answer, nil
			}
		}
		call := ParseCall(emission, t.ToolBox)
		if call != nil && call.OK() {
			return call, "model", emission, "", nil
		}
		if broken == nil && call != nil {
			broken = call
		}
	}
	emission := ""
	if len(candidates) > 0 {
		emission = candidates[0]
	}
	if cfg.Mediation == "never" {
		// a broken call is executed as a failure and learned from; nothing at all ends the attempt
		return broken, "model", emission, "", nil
	}
	hint := ""
	if broken != nil && broken.Error != "" {
		hint = "Its call could not be read: " + broken.Error
	}
	var mediated *ToolCall
	if err := t.external(func() error {
		out, err := MediateCall(t.Client, emission, t.ToolBox, task, transcript, t.modelName(false), hint)
		mediated = out
		return err
	}); err != nil {
		return nil, "", "", "", err
	}
	return mediated, "mediator", emission, "", nil
}

// SolveWithModel is one attempt by the network: it writes calls, the toolbox
// runs them, it answers.
func (t *AgentTrainer) SolveWithModel(task Task, index int, phase string, round int) (*AgentAttempt, error) {
	started := time.Now()
	cfg := t.Config
	transcript := TaskHeader(task.Prompt)
	if len(task.Seeds) > 0 {
		transcript += "START: " + strings.Join(task.Seeds, ", ") + "\n"
	}
	steps := []*Step{}
	answer := ""
	for stepIndex := 0; stepIndex < cfg.MaxSteps; stepIndex++ {
		if t.stopped() {
			break
		}
		call, source, emission, said, err := t.nextCall(task, transcript, index)
		if err != nil {
			return nil, err
		}
		if call == nil {
			answer = said
			break
		}
		var result *ToolResult
		if err := t.external(func() error { result = t.ToolBox.Run(call); return nil }); err != nil {
			return nil, err
		}
		t.noteURLs(result)
		step := &Step{Index: stepIndex, Call: call, Result: result, Source: source, Emission: emission}
		steps = append(steps, step)
		transcript += CallText(call.Name, call.Arguments) + FormatObservation(result, cfg.ObservationChars)
		t.emitStep(phase, task, index, step)
	}
	if answer == "" && !t.stopped() {
		// out of steps: whatever it writes now is its answer, tagged or not - an untrained network
		// answers with noise, and that noise is the failure the judge rejects and 2NRL trains on
		offered, err := t.emissions(transcript, index)
		if err != nil {
			return nil, err
		}
		for _, emission := range offered {
			if found := FindAnswer(emission); found != "" {
				answer = found
				break
			}
		}
		if answer == "" && len(offered) > 0 {
			answer = clipString(collapse(offered[0]), 200)
		}
	}
	text := transcript
	if answer != "" {
		text += AnswerText(answer)
	}
	return &AgentAttempt{Index: index, Source: "model", Steps: steps, Answer: answer, Text: text,
		Seconds: time.Since(started).Seconds()}, nil
}

// SolveWithTeacher is the LLM's demonstration: the same tools, really called,
// and the answer it reaches.
func (t *AgentTrainer) SolveWithTeacher(task Task, criteria []string, index int, feedback string) (*AgentAttempt, error) {
	started := time.Now()
	cfg := t.Config
	steps, answer, err := TeachTask(t.Client, task, t.ToolBox, t.modelName(false), cfg.MaxSteps, criteria,
		feedback, t.External, cfg.ObservationChars)
	if err != nil {
		return nil, err
	}
	transcriptSteps := make([]TranscriptStep, 0, len(steps))
	for _, step := range steps {
		t.noteURLs(step.Result)
		transcriptSteps = append(transcriptSteps, TranscriptStep{
			Name: step.Call.Name, Arguments: step.Call.Arguments, Result: step.Result,
		})
	}
	var given *string
	if answer != "" {
		given = &answer
	}
	text := TranscriptText(task.Prompt, transcriptSteps, given, cfg.ObservationChars)
	return &AgentAttempt{Index: index, Source: "teacher", Steps: steps, Answer: answer, Text: text,
		Seconds: time.Since(started).Seconds()}, nil
}

// Judge marks an attempt against the criteria written before it started.
func (t *AgentTrainer) Judge(task Task, criteria []string, attempt *AgentAttempt) (AgentVerdict, error) {
	if attempt.Answer == "" {
		return AgentVerdict{Correct: false, Critique: "no answer was given", Issues: append([]string{}, criteria...),
			JudgedBy: "answer"}, nil
	}
	if !t.Config.UseJudge {
		correct := task.Answer != "" &&
			strings.Contains(strings.ToLower(attempt.Answer), strings.ToLower(strings.TrimSpace(task.Answer)))
		score := 0.0
		if correct {
			score = 10
		}
		verdict := AgentVerdict{Correct: correct, Score: &score, JudgedBy: "none"}
		if task.Answer != "" {
			verdict.JudgedBy = "answer"
		}
		if !correct {
			verdict.Critique = "the known answer is not in it"
			verdict.Issues = append([]string{}, criteria...)
		}
		return verdict, nil
	}
	var judged JudgedAttempt
	if err := t.external(func() error {
		out, err := JudgeAttempt(t.Client, task, criteria, attempt.Text, attempt.Answer, t.modelName(true))
		judged = out
		return err
	}); err != nil {
		return AgentVerdict{}, err
	}
	return decideAgent(judged, criteria, attempt.Answer, t.Config.Strict), nil
}

// GapOf is how badly an attempt failed, in [0, 1]: 1 = it answered nothing or
// satisfied nothing.
//
// The judge gives a 0-10 score and marks each criterion, so the gap is the
// worse of "how far below PassScore it scored" and "what share of the criteria
// it missed".  Every failure keeps a floor of 0.1 - a failure that is only
// just a failure still trains.
func (t *AgentTrainer) GapOf(attempt *AgentAttempt, criteria []string) float64 {
	if attempt.Verdict.Correct {
		return 0
	}
	if attempt.Answer == "" {
		return 1
	}
	gaps := []float64{}
	if attempt.Verdict.Score != nil {
		gaps = append(gaps, math.Max(0, (t.Config.PassScore-*attempt.Verdict.Score)/t.Config.PassScore))
	}
	if met := attempt.Verdict.Met; len(met) > 0 {
		missed := 0
		for _, ok := range met {
			if !ok {
				missed++
			}
		}
		gaps = append(gaps, float64(missed)/float64(len(met)))
	} else if len(criteria) > 0 {
		gaps = append(gaps, 1)
	}
	worst := 1.0
	if len(gaps) > 0 {
		worst = gaps[0]
		for _, gap := range gaps[1:] {
			if gap > worst {
				worst = gap
			}
		}
	}
	return math.Min(1, math.Max(0.1, worst))
}

// -- learning: failures first, then invert ---------------------------------------

// Learn trains on the failures - the worse, the harder - then inverts, then
// fine-tunes on what was right.
//
// This is the GAN's blatant-failure handling applied to tool use: with
// BlatantMode "fail_invert" every failed transcript is a negative pass whose
// strength is multiplied by min(BlatantBoost, 1 + gap / BlatantMargin), so the
// network is pushed to reproduce its worst attempts on purpose before Invert
// turns that into avoidance.  The local modes (activation / state) instead
// negate the nodes on a failed path, and blatant failures then leave the 2NRL
// garbage set.  Nothing failed -> the correct run is rewarded and nothing is
// inverted.
func (t *AgentTrainer) Learn(failures []Failure, good []string) map[string]any {
	cfg := t.Config
	goodAll := uniqueStrings(good)
	if cfg.ReadReward {
		for _, page := range t.pages {
			if !contains(goodAll, page) {
				goodAll = append(goodAll, page)
			}
		}
	}
	if cfg.Replay {
		extra := []string{}
		for _, text := range t.replay {
			if !contains(goodAll, text) {
				extra = append(extra, text)
			}
		}
		if cfg.ReplayLimit > 0 && len(extra) > cfg.ReplayLimit {
			extra = extra[len(extra)-cfg.ReplayLimit:]
		}
		if cfg.ReplayLimit > 0 {
			goodAll = append(goodAll, extra...)
		}
	}
	worst := map[string]float64{}
	order := []string{}
	for _, failure := range failures { // a transcript that failed twice keeps its worst gap
		if failure.Text == "" || contains(goodAll, failure.Text) {
			continue
		}
		if _, seen := worst[failure.Text]; !seen {
			order = append(order, failure.Text)
		}
		if failure.Gap > worst[failure.Text] {
			worst[failure.Text] = failure.Gap
		}
	}
	bad := order
	gaps := make([]float64, 0, len(bad))
	blatant := 0
	for _, text := range bad {
		gaps = append(gaps, worst[text])
		if worst[text] > cfg.BlatantMargin {
			blatant++
		}
	}
	result := map[string]any{
		"bad": len(bad), "good": len(goodAll), "action": nil, "neg_loss": nil, "pos_loss": nil,
		"failures": len(bad), "blatant": blatant, "mode": cfg.BlatantMode, "flipped": 0,
		"boost_mean": nil, "boost_max": nil,
	}
	if len(bad) == 0 && len(goodAll) == 0 {
		return result
	}
	var weights []float64
	switch {
	case len(bad) > 0 && cfg.BlatantMode == "fail_invert":
		for _, gap := range gaps {
			weights = append(weights, math.Min(cfg.BlatantBoost, 1+gap/cfg.BlatantMargin))
		}
	case len(bad) > 0 && (cfg.BlatantMode == "activation" || cfg.BlatantMode == "state"):
		// the local variant: move every other node of a failed path toward its negation, the worse the more
		amounts := make([]float64, 0, len(gaps))
		for _, gap := range gaps {
			amounts = append(amounts, math.Min(1, gap/(2*cfg.BlatantMargin)))
		}
		if outcome, err := t.Model.InvertPaths(bad, amounts, cfg.Strength); err == nil {
			result["flipped"] = outcome.Flipped
			result["boost_mean"] = outcome.AmountMean
			result["boost_max"] = maxOf(amounts)
		}
		keptText, keptGaps := []string{}, []float64{}
		for i, text := range bad {
			if gaps[i] <= cfg.BlatantMargin {
				keptText, keptGaps = append(keptText, text), append(keptGaps, gaps[i])
			}
		}
		bad, gaps = keptText, keptGaps
	}
	if len(weights) > 0 {
		total := 0.0
		for _, weight := range weights {
			total += weight
		}
		result["boost_mean"] = total / float64(len(weights))
		result["boost_max"] = maxOf(weights)
	}
	switch {
	case len(bad) > 0 && len(goodAll) > 0:
		outcome, err := t.Model.TwoNRLWeighted(bad, weights, goodAll, nil, TwoNRLOptions{
			NegEpochs: cfg.NegEpochs, PosEpochs: cfg.PosEpochs, Strength: cfg.Strength, Stop: t.Stop,
		})
		if err == nil {
			result["action"] = "2nrl"
			result["neg_loss"], result["pos_loss"] = lastLoss(outcome.Negative), lastLoss(outcome.Positive)
		}
	case len(goodAll) > 0:
		records, err := t.Model.Reward(goodAll, cfg.PosEpochs, cfg.Strength)
		if err == nil {
			result["action"], result["pos_loss"] = "reward", lastLoss(records)
		}
	case len(bad) > 0:
		records, err := t.Model.PunishWeighted(bad, weights,
			TrainOptions{Epochs: cfg.NegEpochs, AutoCompress: true, Phase: "negative", Stop: t.Stop}, cfg.Strength)
		if err == nil {
			if !t.stopped() {
				t.Model.Invert()
			}
			result["action"], result["neg_loss"] = "punish", lastLoss(records)
		}
	}
	for _, text := range good {
		if !contains(t.replay, text) {
			t.replay = append(t.replay, text)
		}
	}
	if cfg.ReplayLimit > 0 && len(t.replay) > cfg.ReplayLimit {
		t.replay = t.replay[len(t.replay)-cfg.ReplayLimit:]
	}
	t.pages = nil
	return result
}

func maxOf(values []float64) float64 {
	best := 0.0
	for i, value := range values {
		if i == 0 || value > best {
			best = value
		}
	}
	return best
}

// -- one task -------------------------------------------------------------------

func (t *AgentTrainer) emitStep(phase string, task Task, attempt int, step *Step) {
	t.emit(map[string]any{
		"kind": "step", "phase": phase, "task": task.ID, "attempt": attempt + 1, "step": step.Index + 1,
		"tool": step.Call.Name, "arguments": step.Call.Arguments, "source": step.Source,
		"ok": step.Result.OK, "error": nilString(step.Result.Error), "output": clipString(step.Result.Output, 300),
		"seconds": math.Round(step.Result.Seconds*1000) / 1000,
	})
}

func (t *AgentTrainer) emitAttempt(phase string, round int, task Task, attempt *AgentAttempt) {
	met := attempt.Verdict.Met
	if met == nil {
		met = []bool{}
	}
	t.emit(map[string]any{
		"kind": "attempt", "phase": phase, "round": round, "task": task.ID, "attempt": attempt.Index + 1,
		"source": attempt.Source, "calls": attempt.Calls(), "own_calls": attempt.OwnCalls(),
		"autonomy": attempt.Autonomy(), "answer": nilString(attempt.Answer), "correct": attempt.Verdict.Correct,
		"score": attempt.Verdict.Score, "met": met, "critique": attempt.Verdict.Critique,
		"seconds": attempt.Seconds,
	})
}

// RunTask is the criteria, the attempts, the judging and (on failure) the
// teacher's demonstration for one task.
func (t *AgentTrainer) RunTask(task Task, phase string, round int) (map[string]any, []Failure, []string, error) {
	started := time.Now()
	cfg := t.Config
	criteria, err := t.CriteriaFor(task)
	if err != nil {
		return nil, nil, nil, err
	}
	t.emit(map[string]any{
		"kind": "criteria", "phase": phase, "round": round, "task": task.ID,
		"prompt": clipString(task.Prompt, 300), "criteria": criteria,
	})
	attempts := []*AgentAttempt{}
	solvedByModel := false
	if phase == "model" {
		for index := 0; index < cfg.ModelAttempts; index++ {
			if t.stopped() {
				break
			}
			attempt, err := t.SolveWithModel(task, index, phase, round)
			if err != nil {
				return nil, nil, nil, err
			}
			verdict, err := t.Judge(task, criteria, attempt)
			if err != nil {
				return nil, nil, nil, err
			}
			attempt.Verdict = verdict
			attempts = append(attempts, attempt)
			t.emitAttempt(phase, round, task, attempt)
			if verdict.Correct {
				solvedByModel = true
				break
			}
		}
	}
	anyCorrect := false
	for _, attempt := range attempts {
		anyCorrect = anyCorrect || attempt.Verdict.Correct
	}
	taught := false
	if (phase == "teacher" || (cfg.TeachOnFailure && !anyCorrect)) && !t.stopped() {
		feedback := ""
		if len(attempts) > 0 {
			feedback = attempts[len(attempts)-1].Feedback()
		}
		for try := 0; try < cfg.TeacherAttempts; try++ {
			attempt, err := t.SolveWithTeacher(task, criteria, len(attempts), feedback)
			if err != nil {
				return nil, nil, nil, err
			}
			verdict, err := t.Judge(task, criteria, attempt)
			if err != nil {
				return nil, nil, nil, err
			}
			attempt.Verdict = verdict
			attempts = append(attempts, attempt)
			taught = true
			t.emitAttempt(phase, round, task, attempt)
			if verdict.Correct || t.stopped() {
				break
			}
			feedback = attempt.Feedback()
		}
	}
	var correct *AgentAttempt
	good := []string{}
	failures := []Failure{}
	for _, attempt := range attempts {
		if attempt.Verdict.Correct {
			if correct == nil {
				correct = attempt
			}
			good = append(good, attempt.Text)
		} else {
			failures = append(failures, Failure{Text: attempt.Text, Gap: t.GapOf(attempt, criteria),
				Task: task.ID, Source: attempt.Source})
		}
	}
	if correct != nil {
		t.solved[task.ID] = correct.Text
	}
	own, calls := 0, 0
	for _, attempt := range attempts {
		if attempt.Source == "model" {
			own += attempt.OwnCalls()
			calls += attempt.Calls()
		}
	}
	shown := correct
	if shown == nil && len(attempts) > 0 {
		shown = attempts[len(attempts)-1]
	}
	record := map[string]any{
		"kind": "task", "phase": phase, "round": round, "task": task.ID, "prompt": clipString(task.Prompt, 200),
		"criteria": len(criteria), "attempts": len(attempts), "correct": correct != nil,
		"solved_by": nil, "model_solved": solvedByModel, "taught": taught, "calls": calls, "own_calls": own,
		"autonomy": nil, "answer": nil, "score": nil, "issues": []string{}, "bad": len(failures),
		"good": len(good), "gap_max": nil, "seconds": time.Since(started).Seconds(),
	}
	if correct != nil {
		record["solved_by"] = correct.Source
	}
	if calls > 0 {
		record["autonomy"] = float64(own) / float64(calls)
	}
	if shown != nil {
		record["answer"] = nilString(shown.Answer)
		record["score"] = shown.Verdict.Score
		record["issues"] = firstN(shown.Verdict.Issues, 4)
	}
	if len(failures) > 0 {
		worst := failures[0].Gap
		for _, failure := range failures[1:] {
			if failure.Gap > worst {
				worst = failure.Gap
			}
		}
		record["gap_max"] = worst
	}
	return record, failures, good, nil
}

// -- driving ---------------------------------------------------------------------

// Run works every round and phase over the tasks; it returns the task / round
// records (the step and attempt records go to Progress).
func (t *AgentTrainer) Run(tasks []Task) ([]map[string]any, error) {
	if len(tasks) == 0 {
		return nil, fmt.Errorf("no tasks to solve")
	}
	cfg := t.Config
	records := []map[string]any{}
	for round := 1; round <= cfg.Rounds; round++ {
		for _, phase := range cfg.Phases {
			started := time.Now()
			roundBad := []Failure{}
			roundGood := []string{}
			solved, modelSolved, done := 0, 0, 0
			for _, task := range tasks {
				if t.stopped() {
					break
				}
				record, bad, good, err := t.RunTask(task, phase, round)
				if err != nil {
					return records, err
				}
				if cfg.TwoNRLPer == "task" {
					for key, value := range t.Learn(bad, good) {
						record[key] = value
					}
				} else {
					roundBad, roundGood = append(roundBad, bad...), append(roundGood, good...)
				}
				done++
				if record["correct"] == true {
					solved++
				}
				if record["model_solved"] == true {
					modelSolved++
				}
				t.emit(record)
				records = append(records, record)
			}
			summary := map[string]any{
				"kind": "round", "phase": phase, "round": round, "tasks": done, "solved": solved,
				"model_solved": modelSolved, "seconds": time.Since(started).Seconds(),
			}
			if cfg.TwoNRLPer == "round" && (len(roundBad) > 0 || len(roundGood) > 0) && !t.stopped() {
				for key, value := range t.Learn(roundBad, roundGood) {
					summary[key] = value
				}
			}
			t.emit(summary)
			records = append(records, summary)
			if t.stopped() {
				return records, nil
			}
		}
	}
	return records, nil
}

// Propose lets the network choose what to look into next; the LLM turns that
// into an answerable question.
func (t *AgentTrainer) Propose(index int) (Task, string, error) {
	seed := strings.TrimSpace(TaskHeader("")) // "TASK:" - what every transcript it has learned starts with
	temperature := math.Max(0.8, t.Config.Temperature)
	prediction, err := t.Model.Predict(seed, PredictOptions{
		Length: 1, Mode: "sample", Temperature: temperature, MaxLength: t.Config.MaxLength,
	})
	if err != nil {
		return Task{}, "", err
	}
	emission := prediction.Text
	var task Task
	if err := t.external(func() error {
		out, err := ProposeTask(t.Client, emission, t.modelName(false), t.Frontier, t.Visited, index)
		task = out
		return err
	}); err != nil {
		return Task{}, "", err
	}
	if len(task.Seeds) > 0 {
		kept := []string{}
		for _, url := range t.Frontier {
			if !contains(task.Seeds, url) {
				kept = append(kept, url)
			}
		}
		t.Frontier = kept
	}
	return task, emission, nil
}

// Explore browses on the network's own initiative: it picks each task, the
// loop judges and teaches it.  steps 0 runs until Stop says otherwise.
func (t *AgentTrainer) Explore(steps int) ([]map[string]any, error) {
	records := []map[string]any{}
	index := 0
	for !t.stopped() && (steps <= 0 || index < steps) {
		index++
		started := time.Now()
		task, emission, err := t.Propose(index)
		if err != nil {
			return records, err
		}
		seeds := task.Seeds
		if seeds == nil {
			seeds = []string{}
		}
		t.emit(map[string]any{
			"kind": "proposal", "step": index, "task": task.ID, "prompt": task.Prompt,
			"emission": clipString(collapse(emission), 200), "seeds": seeds,
			"frontier": len(t.Frontier), "visited": len(t.Visited),
		})
		if t.stopped() {
			break
		}
		record, bad, good, err := t.RunTask(task, "model", index)
		if err != nil {
			return records, err
		}
		for key, value := range t.Learn(bad, good) {
			record[key] = value
		}
		record["kind"], record["step"] = "explore", index
		record["frontier"], record["visited"] = len(t.Frontier), len(t.Visited)
		record["seconds"] = time.Since(started).Seconds()
		t.emit(record)
		records = append(records, record)
	}
	return records, nil
}
