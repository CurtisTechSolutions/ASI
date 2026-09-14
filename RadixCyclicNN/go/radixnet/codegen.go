package radixnet

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Semi-supervised code generation: an LLM teacher writes programs, a sandbox
// runs them, a style checker and an LLM judge mark them, and 2NRL teaches the
// network from what was accepted and what was rejected.
//
// Two things about this port are worth knowing.
//
// The sandbox runs the *same* Python bootstrap the Python implementation does -
// `python3 -I -B -c <bootstrap> mem cpu script`, with the same `unshare -rn`
// prefix, the same scrubbed environment and the same scratch directory - so the
// limits are set by the same code in the same place and the isolation is
// identical rather than reimplemented.  Codegen needs a Python interpreter
// whichever language drives it, because the programs it writes are Python.
//
// The style check's *formatting* rules are pure text and are done here.  Its
// naming rules and the blank-line rule need Python's `ast` to know where the
// definitions are, so they go through that same interpreter: writing a Python
// parser in Go to match `ast.walk` would disagree with Python on real programs,
// which is worse than not having it.

// CodeGenPhases are the two ways a program is produced.
var CodeGenPhases = []string{"teacher", "model"}

// Strictness levels of the verdict.
var Strictness = []string{"strict", "lenient"}

// DefaultCodeGenModel is the teacher this build asks for by default.
func DefaultCodeGenModel(provider string) string {
	if provider == ProviderChatGPT {
		return ""
	}
	if named := strings.TrimSpace(os.Getenv("RADIXNET_CODEGEN_MODEL")); named != "" {
		return named
	}
	return "gemma4"
}

const (
	maxOutputChars  = 4000
	maxLineLength   = 79
	sandboxFileSize = 16 << 20
)

// Problem is a task to solve: a prompt, optional test code appended to the
// program, and optional exact stdout.
type Problem struct {
	ID             string  `json:"id"`
	Prompt         string  `json:"prompt"`
	Tests          *string `json:"tests"`
	ExpectedOutput *string `json:"expected_output"`
}

// ToDict is the problem as decoded JSON, the shape ParseProblems reads back.
func (p Problem) ToDict() map[string]any {
	return map[string]any{"id": p.ID, "prompt": p.Prompt, "tests": p.Tests, "expected_output": p.ExpectedOutput}
}

// ParseProblems reads problems from decoded JSON values (strings or objects);
// duplicate ids get a numeric suffix.
func ParseProblems(items []any) ([]Problem, error) {
	problems := []Problem{}
	seen := map[string]bool{}
	for i, item := range items {
		problem, err := problemFrom(item, i+1)
		if err != nil {
			return nil, err
		}
		id, n := problem.ID, 2
		for seen[id] {
			id = fmt.Sprintf("%s-%d", problem.ID, n)
			n++
		}
		problem.ID = id
		seen[id] = true
		problems = append(problems, problem)
	}
	if len(problems) == 0 {
		return nil, fmt.Errorf("no problems given")
	}
	return problems, nil
}

func problemFrom(item any, index int) (Problem, error) {
	switch value := item.(type) {
	case string:
		if strings.TrimSpace(value) == "" {
			return Problem{}, fmt.Errorf("problem %d: empty prompt", index)
		}
		return Problem{ID: fmt.Sprintf("p%d", index), Prompt: strings.TrimSpace(value)}, nil
	case map[string]any:
		prompt := ""
		for _, key := range []string{"prompt", "problem", "task", "question"} {
			if text, ok := value[key].(string); ok && strings.TrimSpace(text) != "" {
				prompt = strings.TrimSpace(text)
				break
			}
		}
		if prompt == "" {
			return Problem{}, fmt.Errorf("problem %d: missing 'prompt'", index)
		}
		problem := Problem{ID: fmt.Sprintf("p%d", index), Prompt: prompt}
		if raw, ok := value["tests"]; ok && raw != nil {
			tests, isStr := raw.(string)
			if !isStr {
				return Problem{}, fmt.Errorf("problem %d: 'tests' must be a string of Python code", index)
			}
			problem.Tests = &tests
		}
		expected, ok := value["expected_output"]
		if !ok {
			expected, ok = value["expected"]
		}
		if ok && expected != nil {
			text, isStr := expected.(string)
			if !isStr {
				return Problem{}, fmt.Errorf("problem %d: 'expected_output' must be a string", index)
			}
			problem.ExpectedOutput = &text
		}
		if raw, ok := value["id"]; ok && raw != nil {
			if id := strings.TrimSpace(fmt.Sprint(raw)); id != "" {
				problem.ID = id
			}
		}
		return problem, nil
	}
	return Problem{}, fmt.Errorf("problem %d: expected a string or an object", index)
}

// ParseProblemFile reads a problem file: .jsonl (one object per line), .json (a
// list or {"problems": [...]}), anything else one prompt per non-blank line
// with # for comments.
func ParseProblemFile(text, ext string) ([]Problem, error) {
	items := []any{}
	switch strings.ToLower(ext) {
	case ".jsonl":
		for _, line := range strings.Split(text, "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			var item any
			if err := json.Unmarshal([]byte(line), &item); err != nil {
				return nil, fmt.Errorf("not valid JSON on one line: %v", err)
			}
			items = append(items, item)
		}
	case ".json":
		var data any
		if err := json.Unmarshal([]byte(text), &data); err != nil {
			return nil, fmt.Errorf("not valid JSON: %v", err)
		}
		switch value := data.(type) {
		case map[string]any:
			list, ok := value["problems"].([]any)
			if !ok {
				return nil, fmt.Errorf(`a JSON problem file must hold a list or {"problems": [...]}`)
			}
			items = list
		case []any:
			items = value
		default:
			return nil, fmt.Errorf(`a JSON problem file must hold a list or {"problems": [...]}`)
		}
	default:
		for _, line := range strings.Split(text, "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "#") {
				continue
			}
			items = append(items, trimmed)
		}
	}
	return ParseProblems(items)
}

// LoadProblems reads a problem file from disk.
func LoadProblems(path string) ([]Problem, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseProblemFile(string(data), filepath.Ext(path))
}

// -- the sandbox --------------------------------------------------------------

// RunResult is what happened when a program ran.
type RunResult struct {
	OK              bool    `json:"ok"`
	ExitCode        *int    `json:"exit_code"`
	Stdout          string  `json:"stdout"`
	Stderr          string  `json:"stderr"`
	Error           string  `json:"error"`
	TimedOut        bool    `json:"timed_out"`
	Seconds         float64 `json:"seconds"`
	ExpectedOK      *bool   `json:"expected_ok"`
	NetworkIsolated bool    `json:"network_isolated"`
}

// sandboxBootstrap is the Python the child runs: it sets its own resource limits
// and then runs the program.  It is character for character the Python
// implementation's, so both languages sandbox identically.
const sandboxBootstrap = `import runpy, sys
try:
    import resource
except ImportError:
    resource = None
mem, cpu, script = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if resource is not None:
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    if cpu:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.argv = [script]
runpy.run_path(script, run_name="__main__")`

// Sandbox runs a generated program in a scratch directory with resource limits.
type Sandbox struct {
	Python     string
	Timeout    time.Duration
	MemoryMB   int
	CPUSeconds int
	unshare    []string
}

// NewSandbox probes for network isolation and returns the sandbox.
func NewSandbox(python string, timeout time.Duration, memoryMB, cpuSeconds int, isolateNetwork bool) (*Sandbox, error) {
	if timeout <= 0 {
		return nil, fmt.Errorf("timeout must be > 0")
	}
	if memoryMB < 0 {
		return nil, fmt.Errorf("memory_mb must be >= 0 (0 = unlimited)")
	}
	if strings.TrimSpace(python) == "" {
		python = pythonBinary()
	}
	if cpuSeconds <= 0 {
		cpuSeconds = int(timeout.Seconds()) + 1
	}
	box := &Sandbox{Python: python, Timeout: timeout, MemoryMB: memoryMB, CPUSeconds: cpuSeconds}
	if isolateNetwork {
		box.unshare = probeUnshare()
	}
	return box, nil
}

// NetworkIsolated reports whether programs run without a network.
func (s *Sandbox) NetworkIsolated() bool { return s.unshare != nil }

// pythonBinary is $RADIXNET_PYTHON, else the first python on the PATH.
func pythonBinary() string {
	if named := strings.TrimSpace(os.Getenv("RADIXNET_PYTHON")); named != "" {
		return named
	}
	for _, name := range []string{"python3", "python"} {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return "python3"
}

// probeUnshare returns the prefix that puts the child in its own network
// namespace, or nil when an unprivileged user cannot make one here.
func probeUnshare() []string {
	if _, err := exec.LookPath("unshare"); err != nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := exec.CommandContext(ctx, "unshare", "-rn", "true").Run(); err != nil {
		return nil
	}
	return []string{"unshare", "-rn"}
}

// Run runs code (with tests appended) and reports what happened; a failing
// program is a result, not an error.
func (s *Sandbox) Run(code string, tests *string, expected *string, stdin string) (*RunResult, error) {
	source := code
	if !strings.HasSuffix(source, "\n") {
		source += "\n"
	}
	if tests != nil && strings.TrimSpace(*tests) != "" {
		source += "\n\n# --- tests ---\n" + strings.Trim(*tests, "\n") + "\n"
	}
	workdir, err := os.MkdirTemp("", "radixnet-sandbox-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(workdir)
	script := filepath.Join(workdir, "solution.py")
	if err := os.WriteFile(script, []byte(source), 0o600); err != nil {
		return nil, err
	}
	args := append(append([]string{}, s.unshare...), s.Python, "-I", "-B", "-c", sandboxBootstrap,
		strconv.Itoa(s.MemoryMB<<20), strconv.Itoa(s.CPUSeconds), script)
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = workdir
	cmd.Env = []string{
		"PATH=" + orDefault(os.Getenv("PATH"), "/usr/bin:/bin"),
		"HOME=" + workdir, "TMPDIR=" + workdir, "LANG=C.UTF-8", "PYTHONIOENCODING=utf-8",
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdin, cmd.Stdout, cmd.Stderr = strings.NewReader(stdin), &stdout, &stderr
	started := time.Now()
	runErr := cmd.Run()
	seconds := time.Since(started).Seconds()

	timedOut := ctx.Err() == context.DeadlineExceeded
	var exitCode *int
	if !timedOut {
		code := cmd.ProcessState.ExitCode()
		exitCode = &code
	}
	errText := stderr.String()
	if timedOut {
		errText = strings.TrimSpace(errText +
			fmt.Sprintf("\nTimeoutError: the program did not finish within %g seconds", s.Timeout.Seconds()))
	}
	errText = cleanTraceback(errText)
	ok := !timedOut && runErr == nil
	failure := ""
	if !ok {
		for _, line := range strings.Split(errText, "\n") {
			if strings.TrimSpace(line) != "" {
				failure = strings.TrimSpace(line)
			}
		}
		if failure == "" {
			failure = fmt.Sprintf("exit code %v", exitCodeText(exitCode))
		}
		if exitCode != nil && *exitCode == -1 && !timedOut {
			failure = "killed (memory or CPU limit exceeded): " + failure
		}
	}
	var expectedOK *bool
	if expected != nil {
		matched := ok && strings.TrimSpace(stdout.String()) == strings.TrimSpace(*expected)
		expectedOK = &matched
	}
	return &RunResult{
		OK: ok, ExitCode: exitCode, Stdout: truncateOutput(stdout.String()), Stderr: truncateOutput(errText),
		Error: failure, TimedOut: timedOut, Seconds: seconds, ExpectedOK: expectedOK,
		NetworkIsolated: s.NetworkIsolated(),
	}, nil
}

func exitCodeText(code *int) string {
	if code == nil {
		return "none"
	}
	return strconv.Itoa(*code)
}

func orDefault(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func truncateOutput(text string) string {
	if len(text) <= maxOutputChars {
		return text
	}
	return text[:maxOutputChars] + fmt.Sprintf("\n... (%d more characters)", len(text)-maxOutputChars)
}

// cleanTraceback drops the bootstrap's own frames from a traceback, so what is
// reported is the program's failure rather than the harness's.
func cleanTraceback(stderr string) string {
	lines := strings.Split(stderr, "\n")
	out := make([]string, 0, len(lines))
	skip := false
	for _, line := range lines {
		if strings.HasPrefix(line, `  File "<string>"`) || strings.Contains(line, "runpy.py") ||
			strings.Contains(line, "<frozen runpy>") {
			skip = true
			continue
		}
		if skip && strings.HasPrefix(line, "    ") {
			continue
		}
		skip = false
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

// -- the objective style check ------------------------------------------------

// StyleReport is what the style checker found.
type StyleReport struct {
	OK       bool     `json:"ok"`
	SyntaxOK bool     `json:"syntax_ok"`
	PEP8OK   bool     `json:"pep8_ok"`
	NamingOK bool     `json:"naming_ok"`
	Issues   []string `json:"issues"`
}

// styleHelper is the Python half of the style check: the rules that need to know
// where the definitions are.  It reports a syntax error, the E302 blank-line
// rule and the naming rules, as JSON.  The formatting rules are pure text and
// are done in Go; this exists because matching `ast.walk` without Python would
// disagree with Python on real programs.
const styleHelper = `import ast, json, re, sys

SNAKE = re.compile(r"^_*[a-z][a-z0-9_]*$|^_+$")
CAPWORDS = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
CONSTANT = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
DUNDER = re.compile(r"^__[a-z0-9_]+__$")

code = sys.stdin.read()
try:
    tree = ast.parse(code)
except SyntaxError as exc:
    print(json.dumps({"syntax_error": f"syntax error: {exc.msg} (line {exc.lineno})"}))
    raise SystemExit(0)
lines = code.split("\n")
formatting = []
body = tree.body
for previous, node in zip(body, body[1:]):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = getattr(previous, "end_lineno", previous.lineno)
        blank = [not lines[i - 1].strip() for i in range(end + 1, first) if 1 <= i - 1 < len(lines)]
        comment_only = all(lines[i - 1].lstrip().startswith("#") or not lines[i - 1].strip()
                           for i in range(end + 1, first))
        if comment_only and sum(blank) < 2:
            formatting.append(f"E302 line {first}: expected 2 blank lines before a top-level definition")
naming = []
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not (SNAKE.match(node.name) or DUNDER.match(node.name)):
            naming.append(f"N802 line {node.lineno}: function name '{node.name}' should be snake_case")
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
            if not SNAKE.match(arg.arg):
                naming.append(f"N803 line {node.lineno}: argument '{arg.arg}' should be snake_case")
    elif isinstance(node, ast.ClassDef):
        if not CAPWORDS.match(node.name):
            naming.append(f"N801 line {node.lineno}: class name '{node.name}' should use CapWords")
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        name = node.id
        if not (SNAKE.match(name) or CONSTANT.match(name) or CAPWORDS.match(name)):
            naming.append(f"N806 line {node.lineno}: variable '{name}' should be snake_case "
                          "(or UPPER_CASE for a constant)")
seen = set()
naming = [n for n in naming if not (n in seen or seen.add(n))]
print(json.dumps({"formatting": formatting, "naming": naming}))`

// CheckStyle checks PEP 8 formatting and naming without external tools.
//
// python is the interpreter the AST half runs in (the sandbox's); when it
// cannot be run the formatting rules still apply and the report says the naming
// pass was skipped rather than claiming the names are fine.
func CheckStyle(code, python string) StyleReport {
	if strings.TrimSpace(python) == "" {
		python = pythonBinary()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "-I", "-B", "-c", styleHelper)
	cmd.Stdin = strings.NewReader(code)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	astErr := cmd.Run()

	var parsed struct {
		SyntaxError string   `json:"syntax_error"`
		Formatting  []string `json:"formatting"`
		Naming      []string `json:"naming"`
	}
	if astErr == nil {
		_ = json.Unmarshal(stdout.Bytes(), &parsed)
	}
	if parsed.SyntaxError != "" {
		return StyleReport{Issues: []string{parsed.SyntaxError}}
	}
	formatting := checkFormatting(code)
	formatting = append(formatting, parsed.Formatting...)
	naming := parsed.Naming
	if astErr != nil {
		// say so rather than pass silently: an unchecked name is not a good name
		formatting = append(formatting,
			"W000 the naming check was skipped: no usable Python interpreter for the AST pass")
	}
	return StyleReport{
		OK: len(formatting) == 0 && len(naming) == 0, SyntaxOK: true,
		PEP8OK: len(formatting) == 0, NamingOK: len(naming) == 0,
		Issues: append(append([]string{}, formatting...), naming...),
	}
}

// checkFormatting is the part of PEP 8 that is pure text: indentation, line
// length, whitespace and the final newline.
func checkFormatting(code string) []string {
	issues := []string{}
	lines := strings.Split(code, "\n")
	if code != "" && !strings.HasSuffix(code, "\n") {
		issues = append(issues, "W292 no newline at end of file")
	}
	for i, line := range lines {
		number := i + 1
		stripped := strings.TrimLeft(line, " \t")
		if stripped == "" {
			if line != "" {
				issues = append(issues, fmt.Sprintf("W293 line %d: whitespace on a blank line", number))
			}
			continue
		}
		indent := line[:len(line)-len(stripped)]
		if strings.Contains(indent, "\t") {
			issues = append(issues, fmt.Sprintf("W191 line %d: indentation contains tabs", number))
		} else if len(indent)%4 != 0 {
			issues = append(issues, fmt.Sprintf("E111 line %d: indentation is not a multiple of four", number))
		}
		if len(line) > maxLineLength {
			issues = append(issues, fmt.Sprintf("E501 line %d: line too long (%d > %d)", number, len(line), maxLineLength))
		}
		if strings.TrimRight(line, " \t") != line {
			issues = append(issues, fmt.Sprintf("W291 line %d: trailing whitespace", number))
		}
	}
	return issues
}

// -- texts: code blocks, training texts, model prefixes -----------------------

var codeFence = regexp.MustCompile("(?si)```(?:python3?|py)?[ \t]*\r?\n(.*?)```")

// ExtractCode is the Python program inside an LLM answer: the longest fenced
// block, else the whole text.
func ExtractCode(text string) string {
	longest := ""
	for _, match := range codeFence.FindAllStringSubmatch(text, -1) {
		if strings.TrimSpace(match[1]) != "" && len(match[1]) > len(longest) {
			longest = match[1]
		}
	}
	if longest != "" {
		return strings.Trim(longest, "\n") + "\n"
	}
	stripped := strings.TrimSpace(text)
	if strings.HasPrefix(stripped, "```") {
		stripped = strings.Trim(stripped, "`")
		if strings.HasPrefix(strings.ToLower(stripped), "python") {
			stripped = stripped[6:]
		}
		stripped = strings.TrimSpace(stripped)
	}
	if stripped == "" {
		return ""
	}
	return stripped + "\n"
}

// SolutionText is what the network trains on: the question, a newline, the answer.
func SolutionText(problem Problem, code string) string {
	return strings.TrimSpace(problem.Prompt) + "\n" + strings.TrimSpace(code) + "\n"
}

// ModelPrefix is the prefix the network continues into code ({problem} is the prompt).
func ModelPrefix(template string, problem Problem) (string, error) {
	if !strings.Contains(template, "{problem}") {
		return "", fmt.Errorf("model_prompt must contain {problem}")
	}
	return strings.ReplaceAll(template, "{problem}", strings.TrimSpace(problem.Prompt)), nil
}

// -- the teacher and the judge ------------------------------------------------

const teacherSystem = "You are an expert Python programmer writing small, self-contained programs. Answer with exactly one " +
	"```python code block and nothing else: no explanations before or after it. The program must run with " +
	"`python3 solution.py` on a plain Python 3 installation (standard library only), must not read input unless " +
	"the task says so, must finish on its own, and must print its result when the task asks for output. Follow " +
	"PEP 8: four-space indentation, lines of at most 79 characters, snake_case function and variable names, " +
	"CapWords class names, UPPER_CASE constants, two blank lines before top-level definitions, a newline at the " +
	"end of the file."

const judgeSystem = "You are a strict reviewer of small Python programs written for a stated task. Decide whether the program " +
	"genuinely accomplishes the task (not merely runs), whether its formatting follows PEP 8, and whether its " +
	"naming follows PEP 8 (snake_case functions and variables, CapWords classes, UPPER_CASE constants, " +
	"descriptive names). Be adversarial: look for wrong results, missing requirements, unhandled cases and " +
	"sloppy names. Reply with JSON only, exactly of the form {\"task_accomplished\": true or false, " +
	"\"pep8\": true or false, \"naming\": true or false, \"score\": <0-10>, \"issues\": [\"...\"], " +
	"\"critique\": \"one sentence\"}."

// problemBlock is the task as the teacher and the judge are shown it.
func problemBlock(problem Problem) string {
	text := "Task:\n" + strings.TrimSpace(problem.Prompt) + "\n"
	if problem.ExpectedOutput != nil {
		text += "\nThe program's standard output must be exactly:\n" + strings.TrimSpace(*problem.ExpectedOutput) + "\n"
	}
	if problem.Tests != nil && *problem.Tests != "" {
		text += "\nThis test code is appended to the program and must pass:\n" + strings.TrimSpace(*problem.Tests) + "\n"
	}
	return text
}

// TeacherGenerate asks the teacher for a first program.
func TeacherGenerate(client LLMClient, problem Problem, extra, model string) (string, error) {
	user := problemBlock(problem)
	if strings.TrimSpace(extra) != "" {
		user += "\nAdditional instructions:\n" + strings.TrimSpace(extra) + "\n"
	}
	user += "\nWrite the program now."
	raw, err := client.Generate(user, LLMOptions{System: teacherSystem, Model: model, Temperature: 0.3})
	if err != nil {
		return "", err
	}
	return ExtractCode(raw), nil
}

// TeacherFix asks the teacher to correct a program that failed or was judged wrong.
func TeacherFix(client LLMClient, problem Problem, attempt *Attempt, extra, model string) (string, error) {
	user := problemBlock(problem)
	user += "\nThis program is not acceptable yet:\n```python\n" + strings.TrimRight(attempt.Code, "\n") +
		"\n```\n\nWhat went wrong:\n" + attempt.Feedback() + "\n"
	if strings.TrimSpace(extra) != "" {
		user += "\nAdditional instructions:\n" + strings.TrimSpace(extra) + "\n"
	}
	user += "\nReturn the complete corrected program as one ```python block and nothing else."
	raw, err := client.Generate(user, LLMOptions{System: teacherSystem, Model: model, Temperature: 0.3})
	if err != nil {
		return "", err
	}
	return ExtractCode(raw), nil
}

// JudgeOpinion is what the LLM judge said.
type JudgeOpinion struct {
	Task     bool     `json:"task"`
	PEP8     bool     `json:"pep8"`
	Naming   bool     `json:"naming"`
	Score    *float64 `json:"score"`
	Issues   []string `json:"issues"`
	Critique string   `json:"critique"`
}

// asBool reads the many ways an LLM writes a boolean.
func asBool(value any, def bool) bool {
	switch v := value.(type) {
	case bool:
		return v
	case float64:
		return v != 0
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "true", "yes", "y", "1", "pass", "passed", "ok":
			return true
		case "false", "no", "n", "0", "fail", "failed":
			return false
		}
	}
	return def
}

// JudgeWithLLM asks the judge whether the program accomplishes the task.
func JudgeWithLLM(client LLMClient, problem Problem, code string, run *RunResult, style StyleReport,
	model string) (*JudgeOpinion, error) {
	user := problemBlock(problem)
	user += fmt.Sprintf("\nProgram:\n```python\n%s\n```\n\nExecution: exit code %s, %.2fs\nstdout:\n%s\nstderr:\n%s\n",
		strings.TrimRight(code, "\n"), exitCodeText(run.ExitCode), run.Seconds,
		orDefault(clipString(run.Stdout, 1500), "(empty)"), orDefault(clipString(run.Stderr, 800), "(empty)"))
	if run.ExpectedOK != nil {
		if *run.ExpectedOK {
			user += "\nThe stdout matches the expected output.\n"
		} else {
			user += "\nThe stdout does NOT match the expected output.\n"
		}
	}
	if problem.Tests != nil && *problem.Tests != "" {
		if run.OK {
			user += "\nThe appended tests passed.\n"
		} else {
			user += "\nThe appended tests FAILED.\n"
		}
	}
	summary := "no issues"
	if !style.OK {
		summary = strings.Join(firstN(style.Issues, 8), "; ")
	}
	user += "\nAutomated style check: " + summary + "\n\nReturn the JSON now."

	raw, err := client.Generate(user, LLMOptions{System: judgeSystem, Model: model, JSON: true, Temperature: 0.1})
	if err != nil {
		return nil, err
	}
	data, ok := loadsLenient(raw).(map[string]any)
	if !ok {
		return nil, llmErrorf("the judge did not answer with a JSON object")
	}
	task := data["task_accomplished"]
	if task == nil {
		task = data["correct"]
	}
	if task == nil {
		task = data["task"]
	}
	opinion := &JudgeOpinion{
		Task:   asBool(task, false),
		PEP8:   asBool(firstPresent(data, "pep8", "formatting"), true),
		Naming: asBool(data["naming"], true),
		Issues: []string{},
	}
	if score, ok := numberOf(data["score"]); ok {
		bounded := math.Max(0, math.Min(10, score))
		opinion.Score = &bounded
	}
	switch issues := data["issues"].(type) {
	case string:
		if strings.TrimSpace(issues) != "" {
			opinion.Issues = []string{strings.TrimSpace(issues)}
		}
	case []any:
		for _, item := range issues {
			if text := strings.TrimSpace(fmt.Sprint(item)); text != "" {
				opinion.Issues = append(opinion.Issues, text)
			}
		}
	}
	opinion.Issues = firstN(opinion.Issues, 10)
	for _, key := range []string{"critique", "reason"} {
		if text, ok := data[key].(string); ok && strings.TrimSpace(text) != "" {
			opinion.Critique = strings.TrimSpace(text)
			break
		}
	}
	return opinion, nil
}

func firstPresent(data map[string]any, keys ...string) any {
	for _, key := range keys {
		if value, ok := data[key]; ok {
			return value
		}
	}
	return nil
}

func firstN[T any](values []T, n int) []T {
	if len(values) <= n {
		return values
	}
	return values[:n]
}

func clipString(text string, n int) string {
	if len(text) <= n {
		return text
	}
	return text[:n]
}

// -- the verdict --------------------------------------------------------------

// CodeVerdict combines the sandbox result, the style report and the LLM's
// opinion into one answer (the negative network's Verdict is a different thing).
type CodeVerdict struct {
	Correct  bool     `json:"correct"`
	Runs     bool     `json:"runs"`
	Task     *bool    `json:"task"`
	PEP8     bool     `json:"pep8"`
	Naming   bool     `json:"naming"`
	Score    *float64 `json:"score"`
	Issues   []string `json:"issues"`
	Critique string   `json:"critique"`
	JudgedBy string   `json:"judged_by"` // sandbox | tests | ollama | chatgpt | none
}

// Decide is the one verdict the three signals come to.
func Decide(run *RunResult, style StyleReport, llm *JudgeOpinion, strictness, judgedBy string) (CodeVerdict, error) {
	if !contains(Strictness, strictness) {
		return CodeVerdict{}, fmt.Errorf("strictness must be one of %s", strings.Join(Strictness, ", "))
	}
	issues := []string{}
	if !run.OK {
		issues = append(issues, orDefault(run.Error, "the program did not run"))
	}
	if run.ExpectedOK != nil && !*run.ExpectedOK {
		issues = append(issues, "stdout differs from the expected output")
	}
	var task *bool
	switch {
	case !run.OK || (run.ExpectedOK != nil && !*run.ExpectedOK):
		no := false
		task, judgedBy = &no, "sandbox"
	case llm != nil:
		value := llm.Task
		task = &value
	case run.ExpectedOK != nil && *run.ExpectedOK:
		yes := true
		task, judgedBy = &yes, "tests"
	default:
		judgedBy = "none" // nothing to judge with: running counts
	}
	pep8, naming := style.PEP8OK, style.NamingOK
	if llm != nil {
		pep8, naming = pep8 && llm.PEP8, naming && llm.Naming
	}
	issues = append(issues, firstN(style.Issues, 6)...)
	verdict := CodeVerdict{Runs: run.OK, Task: task, PEP8: pep8, Naming: naming, JudgedBy: judgedBy}
	if llm != nil {
		issues = append(issues, firstN(llm.Issues, 6)...)
		verdict.Score, verdict.Critique = llm.Score, llm.Critique
	}
	verdict.Issues = issues
	strict := strictness == "strict"
	verdict.Correct = run.OK &&
		!(run.ExpectedOK != nil && !*run.ExpectedOK) &&
		!(task != nil && !*task) &&
		(!strict || (pep8 && naming && style.OK))
	return verdict, nil
}

// Attempt is one program, run, marked and judged.
type Attempt struct {
	Index   int         `json:"index"`
	Source  string      `json:"source"` // the teacher's provider, or "model"
	Code    string      `json:"code"`
	Text    string      `json:"-"`
	Run     *RunResult  `json:"run"`
	Style   StyleReport `json:"style"`
	Verdict CodeVerdict `json:"verdict"`
	Seconds float64     `json:"seconds"`
}

// Feedback is the human-readable reason a program was rejected, which is what
// the teacher's fix prompt is given.
func (a *Attempt) Feedback() string {
	parts := []string{}
	if !a.Run.OK {
		parts = append(parts, "It did not run: "+a.Run.Error)
		if a.Run.Stderr != "" {
			parts = append(parts, "stderr:\n"+lastN(a.Run.Stderr, 1200))
		}
	}
	if a.Run.ExpectedOK != nil && !*a.Run.ExpectedOK {
		parts = append(parts, "Its output differed from the expected output. Actual stdout:\n"+
			orDefault(lastN(a.Run.Stdout, 800), "(empty)"))
	}
	if len(a.Style.Issues) > 0 {
		parts = append(parts, "Style checker: "+strings.Join(firstN(a.Style.Issues, 8), "; "))
	}
	switch {
	case contains(Providers(), a.Verdict.JudgedBy) && a.Verdict.Task != nil && !*a.Verdict.Task:
		parts = append(parts, "Reviewer: the task is not accomplished. "+a.Verdict.Critique)
	case a.Verdict.Critique != "":
		parts = append(parts, "Reviewer: "+a.Verdict.Critique)
	}
	if len(a.Verdict.Issues) > 0 {
		parts = append(parts, clipString("Issues: "+strings.Join(uniqueStrings(a.Verdict.Issues), "; "), 1500))
	}
	if len(parts) == 0 {
		return "It was judged incorrect."
	}
	return strings.Join(parts, "\n")
}

// ToDict is the attempt as the records carry it.
func (a *Attempt) ToDict() map[string]any {
	return map[string]any{
		"index": a.Index, "source": a.Source, "code": a.Code, "text_chars": len(a.Text),
		"run": a.Run, "style": a.Style, "verdict": a.Verdict, "correct": a.Verdict.Correct,
		"seconds": a.Seconds,
	}
}

func lastN(text string, n int) string {
	if len(text) <= n {
		return text
	}
	return text[len(text)-n:]
}

func uniqueStrings(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	return out
}

// -- the trainer --------------------------------------------------------------

// CodeGenConfig is the loop's settings.
type CodeGenConfig struct {
	TeacherProvider string   `json:"teacher_provider"`
	TeacherModel    string   `json:"teacher_model"`
	JudgeProvider   string   `json:"judge_provider"`
	JudgeModel      string   `json:"judge_model"`
	Phases          []string `json:"phases"`
	Rounds          int      `json:"rounds"`
	TeacherAttempts int      `json:"teacher_attempts"`
	ModelAttempts   int      `json:"model_attempts"`
	FirstDijkstra   bool     `json:"first_attempt_dijkstra"`
	Temperature     float64  `json:"temperature"`
	MaxLength       int      `json:"max_length"`
	Strictness      string   `json:"strictness"`
	UseJudge        bool     `json:"use_judge"`
	FallbackTeacher bool     `json:"fallback_teacher"`
	TwoNRLPer       string   `json:"twonrl_per"`
	Replay          bool     `json:"replay"`
	ReplayLimit     int      `json:"replay_limit"`
	TeacherPrompt   string   `json:"teacher_prompt"`
	ModelPrompt     string   `json:"model_prompt"`
	NegEpochs       int      `json:"neg_epochs"`
	PosEpochs       int      `json:"pos_epochs"`
	Strength        float64  `json:"strength"`
	// Sandbox settings.
	SandboxTimeout float64 `json:"sandbox_timeout"`
	MemoryMB       int     `json:"memory_mb"`
	IsolateNetwork bool    `json:"network_isolation"`
	Python         string  `json:"python"`
}

// DefaultCodeGenConfig mirrors the Python defaults.
func DefaultCodeGenConfig() CodeGenConfig {
	return CodeGenConfig{
		TeacherProvider: ProviderOllama, Phases: append([]string{}, CodeGenPhases...), Rounds: 1,
		TeacherAttempts: 3, ModelAttempts: 4, FirstDijkstra: true, Temperature: 1, MaxLength: 800,
		Strictness: "strict", UseJudge: true, FallbackTeacher: true, TwoNRLPer: "problem", Replay: true,
		ReplayLimit: 64, ModelPrompt: "{problem}\n", NegEpochs: 2, PosEpochs: 3, Strength: 1,
		SandboxTimeout: 10, MemoryMB: 256, IsolateNetwork: true,
	}
}

// Validate resolves the providers and checks the settings.
func (c *CodeGenConfig) Validate() error {
	provider, err := NormaliseProvider(c.TeacherProvider)
	if err != nil {
		return fmt.Errorf("teacher_provider must be one of %s", strings.Join(Providers(), ", "))
	}
	c.TeacherProvider = provider
	if strings.TrimSpace(c.JudgeProvider) == "" {
		c.JudgeProvider = c.TeacherProvider
	} else if c.JudgeProvider, err = NormaliseProvider(c.JudgeProvider); err != nil {
		return fmt.Errorf("judge_provider must be one of %s", strings.Join(Providers(), ", "))
	}
	if strings.TrimSpace(c.TeacherModel) == "" {
		c.TeacherModel = DefaultCodeGenModel(c.TeacherProvider)
	}
	if len(c.Phases) == 0 {
		return fmt.Errorf("phases must be a non-empty subset of %s", strings.Join(CodeGenPhases, ", "))
	}
	for _, phase := range c.Phases {
		if !contains(CodeGenPhases, phase) {
			return fmt.Errorf("phases must be a non-empty subset of %s", strings.Join(CodeGenPhases, ", "))
		}
	}
	if c.Rounds < 1 {
		return fmt.Errorf("rounds must be >= 1")
	}
	if c.TeacherAttempts < 1 || c.ModelAttempts < 1 {
		return fmt.Errorf("teacher_attempts and model_attempts must be >= 1")
	}
	if c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.MaxLength < 1 {
		return fmt.Errorf("max_length must be >= 1")
	}
	if !contains(Strictness, c.Strictness) {
		return fmt.Errorf("strictness must be one of %s", strings.Join(Strictness, ", "))
	}
	if c.TwoNRLPer != "problem" && c.TwoNRLPer != "round" {
		return fmt.Errorf("twonrl_per must be 'problem' or 'round'")
	}
	if c.ReplayLimit < 0 || c.NegEpochs < 0 || c.PosEpochs < 0 {
		return fmt.Errorf("replay_limit and epochs must be >= 0")
	}
	if c.SandboxTimeout <= 0 {
		return fmt.Errorf("sandbox_timeout must be > 0")
	}
	if c.MemoryMB < 0 {
		return fmt.Errorf("memory_mb must be >= 0 (0 = unlimited)")
	}
	if _, err := ModelPrefix(c.ModelPrompt, Problem{ID: "check", Prompt: "x"}); err != nil {
		return err
	}
	return nil
}

// ResolvedJudgeModel is the model the judge answers with.
func (c *CodeGenConfig) ResolvedJudgeModel() string {
	if strings.TrimSpace(c.JudgeModel) != "" {
		return c.JudgeModel
	}
	if c.JudgeProvider == c.TeacherProvider {
		return c.TeacherModel
	}
	return DefaultCodeGenModel(c.JudgeProvider)
}

// CodeGenTrainer runs the teacher / model phases over problems and applies 2NRL
// to the network.
type CodeGenTrainer struct {
	Model       *Model
	Client      LLMClient
	JudgeClient LLMClient
	Config      CodeGenConfig
	Sandbox     *Sandbox
	// Negative is an optional negative network: the sandbox and the judge are
	// its tutor, so every rejected program is blamed for what they found.
	Negative *Model
	// External wraps the slow work (sandbox runs, LLM calls) so a server can
	// release its model lock around it.
	External func(func() error) error
	Progress func(map[string]any)
	Stop     func() bool

	History []map[string]any
	replay  []string
	solved  map[string]string
}

// NewCodeGenTrainer validates the configuration and builds the loop.
func NewCodeGenTrainer(model *Model, client LLMClient, config CodeGenConfig) (*CodeGenTrainer, error) {
	if model == nil {
		return nil, fmt.Errorf("a model to teach is required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	box, err := NewSandbox(config.Python, time.Duration(config.SandboxTimeout*float64(time.Second)),
		config.MemoryMB, 0, config.IsolateNetwork)
	if err != nil {
		return nil, err
	}
	return &CodeGenTrainer{
		Model: model, Client: client, JudgeClient: client, Config: config, Sandbox: box,
		solved: map[string]string{},
	}, nil
}

func (t *CodeGenTrainer) external(fn func() error) error {
	if t.External == nil {
		return fn()
	}
	return t.External(fn)
}

func (t *CodeGenTrainer) stopped() bool { return t.Stop != nil && t.Stop() }

// Evaluate runs a program, checks its style and (when it runs) asks the judge.
func (t *CodeGenTrainer) Evaluate(problem Problem, code, source string, index int) (*Attempt, error) {
	started := time.Now()
	if !strings.HasSuffix(code, "\n") {
		code += "\n"
	}
	var run *RunResult
	if strings.TrimSpace(code) == "" {
		run = &RunResult{Error: "empty program", NetworkIsolated: t.Sandbox.NetworkIsolated()}
	} else {
		if err := t.external(func() error {
			out, err := t.Sandbox.Run(code, problem.Tests, problem.ExpectedOutput, "")
			run = out
			return err
		}); err != nil {
			return nil, err
		}
	}
	style := CheckStyle(code, t.Sandbox.Python)
	var llm *JudgeOpinion
	if run.OK && t.Config.UseJudge {
		if err := t.external(func() error {
			out, err := JudgeWithLLM(t.JudgeClient, problem, code, run, style, t.Config.ResolvedJudgeModel())
			llm = out
			return err
		}); err != nil {
			return nil, err
		}
	}
	verdict, err := Decide(run, style, llm, t.Config.Strictness, t.Config.JudgeProvider)
	if err != nil {
		return nil, err
	}
	return &Attempt{
		Index: index, Source: source, Code: code, Text: SolutionText(problem, code), Run: run,
		Style: style, Verdict: verdict, Seconds: time.Since(started).Seconds(),
	}, nil
}

// SolveWithTeacher asks the teacher for a program and lets it fix its own.
func (t *CodeGenTrainer) SolveWithTeacher(problem Problem, phase string, round, startIndex int) ([]*Attempt, error) {
	cfg := t.Config
	var code string
	if err := t.external(func() error {
		out, err := TeacherGenerate(t.Client, problem, cfg.TeacherPrompt, cfg.TeacherModel)
		code = out
		return err
	}); err != nil {
		return nil, err
	}
	attempts := []*Attempt{}
	for i := 0; i < cfg.TeacherAttempts; i++ {
		attempt, err := t.Evaluate(problem, code, cfg.TeacherProvider, startIndex+i)
		if err != nil {
			return attempts, err
		}
		attempts = append(attempts, attempt)
		t.emitAttempt(phase, round, problem, attempt)
		if attempt.Verdict.Correct || i == cfg.TeacherAttempts-1 || t.stopped() {
			break
		}
		if err := t.external(func() error {
			out, err := TeacherFix(t.Client, problem, attempt, cfg.TeacherPrompt, cfg.TeacherModel)
			code = out
			return err
		}); err != nil {
			return attempts, err
		}
	}
	return attempts, nil
}

// GenerateWithModel is the network's index-th program for a problem.
func (t *CodeGenTrainer) GenerateWithModel(problem Problem, index int) (string, error) {
	cfg := t.Config
	prefix, err := ModelPrefix(cfg.ModelPrompt, problem)
	if err != nil {
		return "", err
	}
	mode := "sample"
	if index == 0 && cfg.FirstDijkstra {
		mode = "beam"
	}
	result, err := t.Model.Predict(prefix, PredictOptions{
		Length: 1, Mode: mode, K: 1, Temperature: cfg.Temperature, MaxLength: cfg.MaxLength,
		ToEnd: mode == "beam",
	})
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// SolveWithModel lets the network try, and falls back to the teacher.
func (t *CodeGenTrainer) SolveWithModel(problem Problem, phase string, round int) ([]*Attempt, error) {
	cfg := t.Config
	attempts := []*Attempt{}
	for i := 0; i < cfg.ModelAttempts; i++ {
		code, err := t.GenerateWithModel(problem, i)
		if err != nil {
			return attempts, err
		}
		attempt, err := t.Evaluate(problem, code, "model", i)
		if err != nil {
			return attempts, err
		}
		attempts = append(attempts, attempt)
		t.emitAttempt(phase, round, problem, attempt)
		if attempt.Verdict.Correct || t.stopped() {
			break
		}
	}
	solved := false
	for _, attempt := range attempts {
		solved = solved || attempt.Verdict.Correct
	}
	if !solved && cfg.FallbackTeacher && !t.stopped() {
		more, err := t.SolveWithTeacher(problem, phase, round, len(attempts))
		attempts = append(attempts, more...)
		if err != nil {
			return attempts, err
		}
	}
	return attempts, nil
}

// Learn applies 2NRL to the wrong and correct solution texts; replay keeps
// earlier successes in the positive phase.
func (t *CodeGenTrainer) Learn(bad, good []string) map[string]any {
	cfg := t.Config
	goodAll := uniqueStrings(good)
	if cfg.Replay {
		known := map[string]bool{}
		for _, text := range goodAll {
			known[text] = true
		}
		extra := []string{}
		for _, text := range t.replay {
			if !known[text] {
				extra = append(extra, text)
			}
		}
		if cfg.ReplayLimit > 0 && len(extra) > cfg.ReplayLimit {
			extra = extra[len(extra)-cfg.ReplayLimit:]
		}
		goodAll = append(goodAll, extra...)
	}
	result := map[string]any{"bad": len(bad), "good": len(goodAll), "action": nil, "neg_loss": nil, "pos_loss": nil}
	switch {
	case len(bad) == 0 && len(goodAll) == 0:
		return result
	case len(bad) > 0 && len(goodAll) > 0:
		outcome, err := t.Model.TwoNRL(bad, goodAll, cfg.NegEpochs, cfg.PosEpochs, cfg.Strength)
		if err == nil {
			result["action"] = "2nrl"
			result["neg_loss"], result["pos_loss"] = lastLoss(outcome.Negative), lastLoss(outcome.Positive)
		}
	case len(goodAll) > 0:
		records, err := t.Model.Reward(goodAll, cfg.PosEpochs, cfg.Strength)
		if err == nil {
			result["action"], result["pos_loss"] = "reward", lastLoss(records)
		}
	default:
		records, err := t.Model.Punish(bad, cfg.NegEpochs, cfg.Strength)
		if err == nil {
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
	return result
}

// Solutions are the correct programs found so far, by problem id: the text
// the model is taught (the prompt and the program that answered it).
func (t *CodeGenTrainer) Solutions() map[string]string {
	out := make(map[string]string, len(t.solved))
	for id, text := range t.solved {
		out[id] = text
	}
	return out
}

func (t *CodeGenTrainer) emit(record map[string]any) {
	t.History = append(t.History, record)
	if t.Progress != nil {
		t.Progress(record)
	}
}

func (t *CodeGenTrainer) emitAttempt(phase string, round int, problem Problem, attempt *Attempt) {
	t.emit(map[string]any{
		"kind": "attempt", "phase": phase, "round": round, "problem": problem.ID, "attempt": attempt.Index + 1,
		"source": attempt.Source, "runs": attempt.Verdict.Runs, "correct": attempt.Verdict.Correct,
		"score": attempt.Verdict.Score, "judged_by": attempt.Verdict.JudgedBy, "error": attempt.Run.Error,
		"issues": firstN(attempt.Verdict.Issues, 4), "code": clipString(attempt.Code, 2000),
		"stdout": clipString(attempt.Run.Stdout, 500), "seconds": attempt.Seconds,
	})
}

// RunProblem solves one problem and reports what it learned from it.
func (t *CodeGenTrainer) RunProblem(problem Problem, phase string, round int) (map[string]any, []string, []string, error) {
	started := time.Now()
	var attempts []*Attempt
	var err error
	if phase == "teacher" {
		attempts, err = t.SolveWithTeacher(problem, phase, round, 0)
	} else {
		attempts, err = t.SolveWithModel(problem, phase, round)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	if len(attempts) == 0 {
		return nil, nil, nil, fmt.Errorf("no attempt was produced for %q", problem.ID)
	}
	var correct *Attempt
	good, bad := []string{}, []string{}
	modelSolved := false
	for _, attempt := range attempts {
		if attempt.Verdict.Correct {
			if correct == nil {
				correct = attempt
			}
			good = append(good, attempt.Text)
			modelSolved = modelSolved || attempt.Source == "model"
		} else {
			bad = append(bad, attempt.Text)
		}
	}
	shown := attempts[len(attempts)-1]
	var solvedBy any
	if correct != nil {
		t.solved[problem.ID] = correct.Text
		shown, solvedBy = correct, correct.Source
	}
	record := map[string]any{
		"kind": "problem", "phase": phase, "round": round, "problem": problem.ID,
		"prompt": clipString(problem.Prompt, 200), "attempts": len(attempts), "correct": correct != nil,
		"solved_by": solvedBy, "model_solved": modelSolved, "bad": len(bad), "good": len(good),
		"score": shown.Verdict.Score, "code": clipString(shown.Code, 2000),
		"issues": firstN(shown.Verdict.Issues, 4), "seconds": time.Since(started).Seconds(),
	}
	if taught := t.teachNegative(attempts); taught != nil {
		record["negative_blamed"] = taught.Blamed
		record["negative_reasons"] = taught.Reasons
	}
	return record, bad, good, nil
}

// teachNegative hands the rejected attempts to the negative network: the sandbox
// and the judge are its tutor.
func (t *CodeGenTrainer) teachNegative(attempts []*Attempt) *TeachReport {
	if t.Negative == nil || len(attempts) == 0 {
		return nil
	}
	report, err := TeachAttempts(t.Negative, attempts, true, "codegen", TeachOptions{Stop: t.Stop})
	if err != nil {
		return nil
	}
	return report
}

// Run drives all rounds and phases over the problems.
func (t *CodeGenTrainer) Run(problems []Problem) ([]map[string]any, error) {
	if len(problems) == 0 {
		return nil, fmt.Errorf("no problems to solve")
	}
	cfg := t.Config
	records := []map[string]any{}
	for round := 1; round <= cfg.Rounds; round++ {
		for _, phase := range cfg.Phases {
			started := time.Now()
			roundBad, roundGood := []string{}, []string{}
			solved, modelSolved, done := 0, 0, 0
			for _, problem := range problems {
				if t.stopped() {
					break
				}
				record, bad, good, err := t.RunProblem(problem, phase, round)
				if err != nil {
					return records, err
				}
				if cfg.TwoNRLPer == "problem" {
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
				"kind": "round", "phase": phase, "round": round, "problems": done, "solved": solved,
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
