package radixnet

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func testSandbox(t *testing.T) *Sandbox {
	t.Helper()
	box, err := NewSandbox("", 10*time.Second, 256, 0, true)
	if err != nil {
		t.Fatalf("NewSandbox: %v", err)
	}
	return box
}

func TestParseProblems(t *testing.T) {
	problems, err := ParseProblems([]any{
		"print hello",
		map[string]any{"prompt": "add two numbers", "tests": "assert add(1, 2) == 3", "id": "add"},
		map[string]any{"task": "print the date", "expected": "today"},
	})
	if err != nil {
		t.Fatalf("ParseProblems: %v", err)
	}
	if len(problems) != 3 {
		t.Fatalf("got %d problems", len(problems))
	}
	if problems[0].ID != "p1" || problems[0].Prompt != "print hello" {
		t.Errorf("first: %+v", problems[0])
	}
	if problems[1].ID != "add" || problems[1].Tests == nil {
		t.Errorf("second: %+v", problems[1])
	}
	if problems[2].ExpectedOutput == nil || *problems[2].ExpectedOutput != "today" {
		t.Errorf("third: %+v", problems[2])
	}
}

func TestParseProblemsRefusesRubbish(t *testing.T) {
	if _, err := ParseProblems(nil); err == nil {
		t.Error("no problems must be refused")
	}
	for name, items := range map[string][]any{
		"empty":        {"   "},
		"no prompt":    {map[string]any{"id": "x"}},
		"bad tests":    {map[string]any{"prompt": "x", "tests": 5}},
		"bad expected": {map[string]any{"prompt": "x", "expected_output": 5}},
		"wrong type":   {42},
	} {
		if _, err := ParseProblems(items); err == nil {
			t.Errorf("%s: expected the problem to be refused", name)
		}
	}
}

func TestParseProblemsDeduplicatesIds(t *testing.T) {
	problems, err := ParseProblems([]any{
		map[string]any{"prompt": "a", "id": "same"},
		map[string]any{"prompt": "b", "id": "same"},
		map[string]any{"prompt": "c", "id": "same"},
	})
	if err != nil {
		t.Fatalf("ParseProblems: %v", err)
	}
	if problems[0].ID != "same" || problems[1].ID != "same-2" || problems[2].ID != "same-3" {
		t.Errorf("ids: %q %q %q", problems[0].ID, problems[1].ID, problems[2].ID)
	}
}

func TestParseProblemFileFormats(t *testing.T) {
	lines, err := ParseProblemFile("# a comment\nprint hello\n\nadd numbers\n", ".txt")
	if err != nil || len(lines) != 2 {
		t.Fatalf("plain lines: %v %v", lines, err)
	}
	jsonl, err := ParseProblemFile("{\"prompt\": \"a\"}\n{\"prompt\": \"b\"}\n", ".jsonl")
	if err != nil || len(jsonl) != 2 {
		t.Fatalf("jsonl: %v %v", jsonl, err)
	}
	wrapped, err := ParseProblemFile(`{"problems": [{"prompt": "a"}]}`, ".json")
	if err != nil || len(wrapped) != 1 {
		t.Fatalf("json object: %v %v", wrapped, err)
	}
	list, err := ParseProblemFile(`["a", "b", "c"]`, ".json")
	if err != nil || len(list) != 3 {
		t.Fatalf("json list: %v %v", list, err)
	}
	if _, err := ParseProblemFile(`{"nope": 1}`, ".json"); err == nil {
		t.Error("a JSON file with no problems must be refused")
	}
	if _, err := ParseProblemFile("not json", ".jsonl"); err == nil {
		t.Error("a bad JSONL line must be refused")
	}
}

func TestSandboxRunsAProgram(t *testing.T) {
	box := testSandbox(t)
	run, err := box.Run("print('hello')\n", nil, nil, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !run.OK || strings.TrimSpace(run.Stdout) != "hello" {
		t.Fatalf("run: %+v", run)
	}
	if run.ExitCode == nil || *run.ExitCode != 0 {
		t.Errorf("exit code: %v", run.ExitCode)
	}
}

func TestSandboxReportsACrash(t *testing.T) {
	box := testSandbox(t)
	run, err := box.Run("raise ValueError('boom')\n", nil, nil, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if run.OK || run.Error == "" {
		t.Fatalf("a crash must be reported: %+v", run)
	}
	if !strings.Contains(run.Error, "boom") {
		t.Errorf("the error must name the failure: %q", run.Error)
	}
	// the bootstrap's own frames are not the program's problem
	if strings.Contains(run.Stderr, "runpy") {
		t.Errorf("the traceback still has harness frames: %q", run.Stderr)
	}
}

func TestSandboxTimesOut(t *testing.T) {
	box, err := NewSandbox("", 1500*time.Millisecond, 256, 0, true)
	if err != nil {
		t.Fatalf("NewSandbox: %v", err)
	}
	run, err := box.Run("while True:\n    pass\n", nil, nil, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !run.TimedOut || run.OK {
		t.Fatalf("an endless program must time out: %+v", run)
	}
	if !strings.Contains(run.Stderr, "TimeoutError") {
		t.Errorf("stderr: %q", run.Stderr)
	}
}

func TestSandboxChecksTheExpectedOutput(t *testing.T) {
	box := testSandbox(t)
	want := "hello"
	run, _ := box.Run("print('hello')\n", nil, &want, "")
	if run.ExpectedOK == nil || !*run.ExpectedOK {
		t.Errorf("matching output: %+v", run.ExpectedOK)
	}
	run, _ = box.Run("print('goodbye')\n", nil, &want, "")
	if run.ExpectedOK == nil || *run.ExpectedOK {
		t.Errorf("differing output: %+v", run.ExpectedOK)
	}
}

func TestSandboxAppendsTheTests(t *testing.T) {
	box := testSandbox(t)
	tests := "assert add(1, 2) == 3\n"
	good := "def add(a, b):\n    return a + b\n"
	run, _ := box.Run(good, &tests, nil, "")
	if !run.OK {
		t.Errorf("passing tests: %+v", run)
	}
	bad := "def add(a, b):\n    return a * b\n"
	run, _ = box.Run(bad, &tests, nil, "")
	if run.OK {
		t.Error("failing tests must fail the run")
	}
}

func TestSandboxIsolatesTheEnvironment(t *testing.T) {
	box := testSandbox(t)
	run, err := box.Run("import os\nprint(sorted(os.environ))\n", nil, nil, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// only the five variables the sandbox sets, whatever the parent's environment holds
	for _, name := range []string{"PATH", "HOME", "TMPDIR", "LANG", "PYTHONIOENCODING"} {
		if !strings.Contains(run.Stdout, name) {
			t.Errorf("the sandbox must set %s: %q", name, run.Stdout)
		}
	}
	if strings.Contains(run.Stdout, "RADIXNET") {
		t.Errorf("the parent's environment must not leak: %q", run.Stdout)
	}
}

func TestSandboxLimitsMemory(t *testing.T) {
	box, err := NewSandbox("", 20*time.Second, 64, 0, true)
	if err != nil {
		t.Fatalf("NewSandbox: %v", err)
	}
	run, err := box.Run("x = bytearray(512 * 1024 * 1024)\nprint(len(x))\n", nil, nil, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if run.OK {
		t.Error("half a gigabyte must not fit in a 64 MB limit")
	}
}

func TestCheckStyleOnCleanCode(t *testing.T) {
	clean := "def add_two(first, second):\n    return first + second\n"
	report := CheckStyle(clean, "")
	if !report.OK || !report.SyntaxOK || !report.PEP8OK || !report.NamingOK {
		t.Fatalf("clean code: %+v", report)
	}
}

func TestCheckStyleFindsFormatting(t *testing.T) {
	report := CheckStyle("def f():\n  return 1   \n", "")
	joined := strings.Join(report.Issues, " | ")
	if report.PEP8OK {
		t.Fatalf("issues: %v", report.Issues)
	}
	for _, code := range []string{"E111", "W291"} {
		if !strings.Contains(joined, code) {
			t.Errorf("expected %s in %q", code, joined)
		}
	}
}

func TestCheckStyleFindsNaming(t *testing.T) {
	report := CheckStyle("def BadName(BadArg):\n    return BadArg\n", "")
	joined := strings.Join(report.Issues, " | ")
	if report.NamingOK {
		t.Fatalf("issues: %v", report.Issues)
	}
	if !strings.Contains(joined, "N802") || !strings.Contains(joined, "N803") {
		t.Errorf("expected N802 and N803 in %q", joined)
	}
}

func TestCheckStyleReportsASyntaxError(t *testing.T) {
	report := CheckStyle("def broken(\n", "")
	if report.SyntaxOK || report.OK {
		t.Fatalf("a syntax error: %+v", report)
	}
	if len(report.Issues) != 1 || !strings.Contains(report.Issues[0], "syntax error") {
		t.Errorf("issues: %v", report.Issues)
	}
}

func TestCheckStyleSaysWhenTheNamingPassWasSkipped(t *testing.T) {
	report := CheckStyle("def f():\n    return 1\n", "/definitely/not/a/python")
	joined := strings.Join(report.Issues, " | ")
	if !strings.Contains(joined, "naming check was skipped") {
		t.Errorf("an unchecked name must not pass silently: %v", report.Issues)
	}
}

func TestExtractCode(t *testing.T) {
	if got := ExtractCode("chatter\n```python\nprint(1)\n```\nmore"); got != "print(1)\n" {
		t.Errorf("fenced: %q", got)
	}
	long := "```py\na = 1\n```\ntext\n```\nb = 2\nc = 3\nd = 4\n```"
	if got := ExtractCode(long); !strings.Contains(got, "b = 2") {
		t.Errorf("the longest block wins: %q", got)
	}
	if got := ExtractCode("print(1)"); got != "print(1)\n" {
		t.Errorf("unfenced: %q", got)
	}
	if got := ExtractCode(""); got != "" {
		t.Errorf("empty: %q", got)
	}
}

func TestSolutionTextAndModelPrefix(t *testing.T) {
	problem := Problem{ID: "p1", Prompt: "  add two numbers  "}
	if got := SolutionText(problem, "  def add(): pass  "); got != "add two numbers\ndef add(): pass\n" {
		t.Errorf("solution text: %q", got)
	}
	prefix, err := ModelPrefix("# {problem}\n", problem)
	if err != nil || prefix != "# add two numbers\n" {
		t.Errorf("prefix: %q %v", prefix, err)
	}
	if _, err := ModelPrefix("no placeholder", problem); err == nil {
		t.Error("a template without {problem} must be refused")
	}
}

func TestDecideCombinesTheSignals(t *testing.T) {
	clean := StyleReport{OK: true, SyntaxOK: true, PEP8OK: true, NamingOK: true}
	ran := &RunResult{OK: true}
	no, yes := false, true

	verdict, err := Decide(ran, clean, &JudgeOpinion{Task: true, PEP8: true, Naming: true}, "strict", "ollama")
	if err != nil || !verdict.Correct || verdict.JudgedBy != "ollama" {
		t.Fatalf("all green: %+v %v", verdict, err)
	}
	verdict, _ = Decide(ran, clean, &JudgeOpinion{Task: false}, "strict", "ollama")
	if verdict.Correct || verdict.Task == nil || *verdict.Task {
		t.Errorf("the judge said no: %+v", verdict)
	}
	crashed := &RunResult{OK: false, Error: "boom"}
	verdict, _ = Decide(crashed, clean, nil, "strict", "ollama")
	if verdict.Correct || verdict.JudgedBy != "sandbox" || len(verdict.Issues) == 0 {
		t.Errorf("a crash is the sandbox's verdict: %+v", verdict)
	}
	wrong := &RunResult{OK: true, ExpectedOK: &no}
	verdict, _ = Decide(wrong, clean, nil, "strict", "ollama")
	if verdict.Correct || verdict.JudgedBy != "sandbox" {
		t.Errorf("wrong output: %+v", verdict)
	}
	passed := &RunResult{OK: true, ExpectedOK: &yes}
	verdict, _ = Decide(passed, clean, nil, "strict", "ollama")
	if !verdict.Correct || verdict.JudgedBy != "tests" {
		t.Errorf("the tests judged it: %+v", verdict)
	}
	verdict, _ = Decide(ran, clean, nil, "strict", "ollama")
	if verdict.JudgedBy != "none" || verdict.Task != nil {
		t.Errorf("nothing to judge with: %+v", verdict)
	}
	// style only matters in strict mode
	messy := StyleReport{SyntaxOK: true, PEP8OK: false, NamingOK: true, Issues: []string{"E501 ..."}}
	verdict, _ = Decide(ran, messy, nil, "strict", "ollama")
	if verdict.Correct {
		t.Error("strict mode must reject a style failure")
	}
	verdict, _ = Decide(ran, messy, nil, "lenient", "ollama")
	if !verdict.Correct {
		t.Error("lenient mode must accept it")
	}
	if _, err := Decide(ran, clean, nil, "sideways", "ollama"); err == nil {
		t.Error("an unknown strictness must be refused")
	}
}

func TestAttemptFeedback(t *testing.T) {
	no := false
	attempt := &Attempt{
		Code:  "print(1)\n",
		Run:   &RunResult{OK: false, Error: "ValueError: boom", Stderr: "Traceback...\nValueError: boom"},
		Style: StyleReport{Issues: []string{"E501 line 1: line too long"}},
		Verdict: CodeVerdict{Task: &no, JudgedBy: "ollama", Critique: "it does not add anything",
			Issues: []string{"wrong result"}},
	}
	feedback := attempt.Feedback()
	for _, want := range []string{"It did not run", "boom", "Style checker", "Reviewer", "Issues"} {
		if !strings.Contains(feedback, want) {
			t.Errorf("feedback is missing %q: %s", want, feedback)
		}
	}
	fine := &Attempt{Run: &RunResult{OK: true}, Verdict: CodeVerdict{}}
	if fine.Feedback() != "It was judged incorrect." {
		t.Errorf("nothing to say: %q", fine.Feedback())
	}
}

func TestCodeReasonAndFaults(t *testing.T) {
	no := false
	cases := map[string]*Attempt{
		"timeout":       {Run: &RunResult{TimedOut: true}, Text: "a"},
		"crash":         {Run: &RunResult{OK: false}, Text: "b"},
		"wrong-output":  {Run: &RunResult{OK: true, ExpectedOK: &no}, Text: "c"},
		"task-not-done": {Run: &RunResult{OK: true}, Verdict: CodeVerdict{Task: &no, PEP8: true, Naming: true}, Style: StyleReport{PEP8OK: true, NamingOK: true}, Text: "d"},
		"naming":        {Run: &RunResult{OK: true}, Verdict: CodeVerdict{PEP8: true}, Style: StyleReport{PEP8OK: true}, Text: "e"},
		"style":         {Run: &RunResult{OK: true}, Verdict: CodeVerdict{Naming: true}, Style: StyleReport{NamingOK: true}, Text: "f"},
	}
	for want, attempt := range cases {
		if got := CodeReason(attempt); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	}
	faults, correct := FaultsFromAttempts([]*Attempt{
		{Run: &RunResult{TimedOut: true}, Text: "slow"},
		{Run: &RunResult{OK: true}, Verdict: CodeVerdict{Correct: true}, Text: "good"},
	}, "codegen")
	if len(faults) != 1 || faults[0].Reason != "timeout" || faults[0].Severity != 1.5 {
		t.Fatalf("faults: %+v", faults)
	}
	if len(correct) != 1 || correct[0] != "good" {
		t.Errorf("correct: %v", correct)
	}
}

func TestTeachAttempts(t *testing.T) {
	negative, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	report, err := TeachAttempts(negative, []*Attempt{
		{Run: &RunResult{TimedOut: true}, Text: "print a loop forever"},
	}, true, "codegen", TeachOptions{})
	if err != nil {
		t.Fatalf("TeachAttempts: %v", err)
	}
	if report.Blamed != 1 || report.Source != "codegen" {
		t.Fatalf("report: %+v", report)
	}
	if negative.G.Neg.TotalBlame <= 0 {
		t.Error("the failure must reach the network")
	}
}

func TestCodeGenConfigValidation(t *testing.T) {
	config := DefaultCodeGenConfig()
	if err := config.Validate(); err != nil {
		t.Fatalf("the defaults must validate: %v", err)
	}
	if config.TeacherModel == "" {
		t.Error("validate must fill in the teacher's model")
	}
	if config.JudgeProvider != config.TeacherProvider {
		t.Error("the judge defaults to the teacher's provider")
	}
	for name, change := range map[string]func(*CodeGenConfig){
		"provider":   func(c *CodeGenConfig) { c.TeacherProvider = "gemini" },
		"phases":     func(c *CodeGenConfig) { c.Phases = []string{"sideways"} },
		"no phases":  func(c *CodeGenConfig) { c.Phases = nil },
		"rounds":     func(c *CodeGenConfig) { c.Rounds = 0 },
		"attempts":   func(c *CodeGenConfig) { c.ModelAttempts = 0 },
		"strictness": func(c *CodeGenConfig) { c.Strictness = "sideways" },
		"twonrl":     func(c *CodeGenConfig) { c.TwoNRLPer = "sideways" },
		"prompt":     func(c *CodeGenConfig) { c.ModelPrompt = "no placeholder" },
		"timeout":    func(c *CodeGenConfig) { c.SandboxTimeout = 0 },
		"memory":     func(c *CodeGenConfig) { c.MemoryMB = -1 },
	} {
		broken := DefaultCodeGenConfig()
		change(&broken)
		if err := broken.Validate(); err == nil {
			t.Errorf("%s: expected the configuration to be refused", name)
		}
	}
}

func TestCodeGenConfigRoundTripsThroughJSON(t *testing.T) {
	config := DefaultCodeGenConfig()
	_ = config.Validate()
	raw, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back CodeGenConfig
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Rounds != config.Rounds || back.Strictness != config.Strictness {
		t.Errorf("round trip: %+v", back)
	}
}

// scriptedTeacher answers with the next program on its list, and judges
// everything it is shown as correct.
type scriptedTeacher struct {
	programs []string
	prompts  []string
	judge    map[string]any
}

func (s *scriptedTeacher) Provider() string                  { return ProviderOllama }
func (s *scriptedTeacher) BaseURL() string                   { return "http://scripted" }
func (s *scriptedTeacher) ModelName() string                 { return "scripted" }
func (s *scriptedTeacher) Available() bool                   { return true }
func (s *scriptedTeacher) Models() ([]map[string]any, error) { return nil, nil }

func (s *scriptedTeacher) Generate(prompt string, o LLMOptions) (string, error) {
	s.prompts = append(s.prompts, prompt)
	if o.JSON { // the judge
		verdict := s.judge
		if verdict == nil {
			verdict = map[string]any{"runs": true, "task": true, "pep8": true, "naming": true, "correct": true,
				"critique": "it does what was asked"}
		}
		body, _ := json.Marshal(verdict)
		return string(body), nil
	}
	if len(s.programs) == 0 {
		return "```python\nprint(\"hello\")\n```", nil
	}
	next := s.programs[0]
	if len(s.programs) > 1 {
		s.programs = s.programs[1:]
	}
	return next, nil
}

func codegenTrainer(t *testing.T, teacher *scriptedTeacher, config CodeGenConfig) *CodeGenTrainer {
	t.Helper()
	model, err := NewModel(7, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	model.Exact = true
	trainer, err := NewCodeGenTrainer(model, teacher, config)
	if err != nil {
		t.Fatal(err)
	}
	return trainer
}

func TestTrainerRunsTheTeacherPhase(t *testing.T) {
	config := DefaultCodeGenConfig()
	config.Phases = []string{"teacher"}
	config.TeacherAttempts = 2
	config.SandboxTimeout = 20
	teacher := &scriptedTeacher{programs: []string{
		"```python\nprint(hello)\n```",     // NameError: the sandbox rejects it
		"```python\nprint(\"hello\")\n```", // the fix runs
	}}
	trainer := codegenTrainer(t, teacher, config)
	problems, err := ParseProblems([]any{"Print the word hello."})
	if err != nil {
		t.Fatal(err)
	}
	records, err := trainer.Run(problems)
	if err != nil {
		t.Fatal(err)
	}
	attempts, problemRecords, rounds := 0, 0, 0
	for _, record := range trainer.History { // the attempt records are progress, not results
		if record["kind"] == "attempt" {
			attempts++
		}
	}
	for _, record := range records {
		switch record["kind"] {
		case "problem":
			problemRecords++
			if record["correct"] != true {
				t.Fatalf("the teacher's second attempt solves it: %+v", record)
			}
		case "round":
			rounds++
		}
	}
	if attempts != 2 || problemRecords != 1 || rounds != 1 {
		t.Fatalf("one problem, two attempts, one round: %d/%d/%d", attempts, problemRecords, rounds)
	}
	solutions := trainer.Solutions()
	if solutions["p1"] != "Print the word hello.\nprint(\"hello\")\n" {
		t.Fatalf("the correct program is kept: %q", solutions["p1"])
	}
	if trainer.Model.Meta["twonrl_runs"] == nil {
		t.Fatal("2NRL ran on the problem")
	}
}

func TestTrainerStopsBetweenProblems(t *testing.T) {
	config := DefaultCodeGenConfig()
	config.Phases = []string{"teacher"}
	config.TeacherAttempts = 1
	config.SandboxTimeout = 20
	trainer := codegenTrainer(t, &scriptedTeacher{}, config)
	trainer.Stop = func() bool { return true }
	problems, err := ParseProblems([]any{"Print the word hello.", "Print the word world."})
	if err != nil {
		t.Fatal(err)
	}
	records, err := trainer.Run(problems)
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		if record["kind"] == "problem" {
			t.Fatalf("a stop before the first problem solves none of them: %+v", record)
		}
	}
}

func TestTrainerBlamesTheRejectedPrograms(t *testing.T) {
	config := DefaultCodeGenConfig()
	config.Phases = []string{"teacher"}
	config.TeacherAttempts = 1
	config.SandboxTimeout = 20
	teacher := &scriptedTeacher{programs: []string{"```python\nprint(hello)\n```"}} // never runs
	trainer := codegenTrainer(t, teacher, config)
	negative, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	trainer.Negative = negative
	problems, err := ParseProblems([]any{"Print the word hello."})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.Run(problems); err != nil {
		t.Fatal(err)
	}
	if negative.G.Neg.TotalBlame <= 0 {
		t.Fatal("the crash blames the negative network")
	}
	reasons := negative.Reasons()
	if len(reasons) == 0 || reasons[0].Reason != "crash" {
		t.Fatalf("under the reason the sandbox found: %+v", reasons)
	}
}
