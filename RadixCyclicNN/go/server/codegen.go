package server

// Teaching the model to write programs over HTTP: POST /api/codegen/start runs
// the teacher / model phases over a list of problems and rewards the network
// with 2NRL, /api/codegen/solve answers one problem now, /api/codegen/run
// executes a program in the sandbox, and /api/codegen/history is the record.
// The Go twin of the Python server's codegen endpoints, answering the same
// JSON.

import (
	"path/filepath"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// StartCodeGen starts a `codegen` job: rounds of the teacher and the model
// solving problems, every attempt run in the sandbox and judged, and 2NRL on
// what came out right and wrong.
//
// With blame, the rejected programs also teach the negative network *why* they
// were rejected (the sandbox, the style check and the judge are its tutor).
func (s *Service) StartCodeGen(
	problems []radixnet.Problem, config radixnet.CodeGenConfig, client, judge radixnet.LLMClient, blame bool,
) (map[string]any, error) {
	if len(problems) == 0 {
		return nil, badRequest("no problems to solve")
	}
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	var negative *radixnet.Model
	if blame {
		found, err := s.negativeModel()
		if err != nil {
			return nil, err
		}
		negative = found
	}
	return s.startJob("codegen", func(job *Job, progress func(map[string]any), stop func() bool) error {
		trainer, err := radixnet.NewCodeGenTrainer(s.model, client, config)
		if err != nil {
			return err
		}
		trainer.JudgeClient = judge
		trainer.Negative = negative
		trainer.Progress = func(record map[string]any) {
			s.codegenHistory = append(s.codegenHistory, record)
			progress(record)
		}
		trainer.Stop = stop
		// The sandbox runs and the teacher thinks without the model lock, so readers stay served.
		trainer.External = func(fn func() error) error {
			s.mu.Unlock()
			defer s.mu.Lock()
			return fn()
		}
		_, err = trainer.Run(problems)
		return err
	})
}

// CodeGenHistory is the attempt / problem / round records of every run.
func (s *Service) CodeGenHistory() map[string]any {
	history := s.codegenHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("POST", "/api/codegen/start", rCodeGenStart)
	doc("POST", "/api/codegen/start", "start a codegen job: {problems | problems_text | problem_files, phases: "+
		"both|teacher|model, rounds, teacher_attempts, model_attempts, teacher_provider, judge_provider, "+
		"teacher_model, judge_model, url, judge_url, timeout, temperature, max_length, strictness: strict|lenient, "+
		"judge, fallback_teacher, twonrl_per: problem|round, replay, replay_limit, neg_epochs, pos_epochs, "+
		"strength, sandbox_timeout, memory_mb, network_isolation, blame (teach the negative network what was "+
		"rejected and why)}")
	route("GET", "/api/codegen/history", rCodeGenHistory)
	doc("GET", "/api/codegen/history", "attempt / problem / round records of all codegen runs")
	route("POST", "/api/codegen/solve", rCodeGenSolve)
	doc("POST", "/api/codegen/solve", "solve one problem now, without training: {problem, source: model|teacher, "+
		"attempts, + the codegen settings}")
	route("POST", "/api/codegen/run", rCodeGenRun)
	doc("POST", "/api/codegen/run", "run a program in the sandbox: {code, tests, expected_output, stdin, "+
		"strictness, sandbox_timeout, memory_mb, network_isolation}")
}

// problemsFrom reads the problems of a request: `problems` (strings or
// objects), `problems_text` (one per line) and `problem_files` (uploads).
func problemsFrom(rq *request) ([]radixnet.Problem, error) {
	items := []any{}
	if raw, ok := rq.f.lookup("problems"); ok && raw != nil {
		list, ok := raw.([]any)
		if !ok {
			return nil, badRequest("'problems' must be a list of strings or {prompt, tests, expected_output} objects")
		}
		items = append(items, list...)
	}
	text, err := rq.f.optText("problems_text", "")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "#") {
			items = append(items, trimmed)
		}
	}
	names, err := rq.f.names("problem_files")
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		// a ZIP upload was unpacked into one upload per entry when it arrived, so each
		// name here is one file and its own extension says how to read it
		contents, err := rq.svc.UploadTexts([]string{name}, "file", 0, true)
		if err != nil {
			return nil, err
		}
		for _, content := range contents {
			parsed, err := radixnet.ParseProblemFile(content, filepath.Ext(name))
			if err != nil {
				return nil, badRequest("upload %q: %v", name, err)
			}
			for _, problem := range parsed {
				items = append(items, problem.ToDict())
			}
		}
	}
	if len(items) == 0 {
		return nil, badRequest("missing 'problems' (list of strings / objects), 'problems_text' (one per line) " +
			"or 'problem_files' (upload names)")
	}
	problems, err := radixnet.ParseProblems(items)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return problems, nil
}

// codegenConfigFrom reads the codegen settings of a request body.
func codegenConfigFrom(rq *request) (radixnet.CodeGenConfig, error) {
	cfg := radixnet.DefaultCodeGenConfig()
	zeroI, oneI := 0, 1
	zero := 0.0
	f := rq.f
	if raw, ok := f.lookup("phases"); !ok || raw == nil {
		if raw, ok = f.lookup("phase"); ok && raw != nil {
			phases, err := phasesOf(raw)
			if err != nil {
				return cfg, err
			}
			cfg.Phases = phases
		}
	} else {
		phases, err := phasesOf(raw)
		if err != nil {
			return cfg, err
		}
		cfg.Phases = phases
	}
	var err error
	if cfg.TeacherProvider, err = providerField(rq, "teacher_provider", "provider", cfg.TeacherProvider); err != nil {
		return cfg, err
	}
	if cfg.JudgeProvider, err = providerField(rq, "judge_provider", "", cfg.TeacherProvider); err != nil {
		return cfg, err
	}
	if cfg.TeacherModel, err = f.optText("teacher_model", ""); err != nil {
		return cfg, err
	}
	if cfg.TeacherModel == "" {
		if cfg.TeacherModel, err = f.optText("model", ""); err != nil {
			return cfg, err
		}
	}
	if cfg.JudgeModel, err = f.optText("judge_model", ""); err != nil {
		return cfg, err
	}
	if cfg.Rounds, _, err = f.integer("rounds", cfg.Rounds, &oneI); err != nil {
		return cfg, err
	}
	if cfg.TeacherAttempts, _, err = f.integer("teacher_attempts", cfg.TeacherAttempts, &oneI); err != nil {
		return cfg, err
	}
	if cfg.ModelAttempts, _, err = f.integer("model_attempts", cfg.ModelAttempts, &oneI); err != nil {
		return cfg, err
	}
	if cfg.FirstDijkstra, err = f.flag("first_attempt_dijkstra", cfg.FirstDijkstra); err != nil {
		return cfg, err
	}
	if cfg.Temperature, _, err = f.number("temperature", cfg.Temperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.MaxLength, _, err = f.integer("max_length", cfg.MaxLength, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Strictness, err = f.optText("strictness", cfg.Strictness); err != nil {
		return cfg, err
	}
	cfg.Strictness = strings.ToLower(strings.TrimSpace(cfg.Strictness))
	if cfg.UseJudge, err = f.flag("judge", cfg.UseJudge); err != nil {
		return cfg, err
	}
	if cfg.FallbackTeacher, err = f.flag("fallback_teacher", cfg.FallbackTeacher); err != nil {
		return cfg, err
	}
	if cfg.TwoNRLPer, err = f.optText("twonrl_per", cfg.TwoNRLPer); err != nil {
		return cfg, err
	}
	cfg.TwoNRLPer = strings.ToLower(strings.TrimSpace(cfg.TwoNRLPer))
	if cfg.Replay, err = f.flag("replay", cfg.Replay); err != nil {
		return cfg, err
	}
	if cfg.ReplayLimit, _, err = f.integer("replay_limit", cfg.ReplayLimit, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.TeacherPrompt, err = f.optText("teacher_prompt", ""); err != nil {
		return cfg, err
	}
	if cfg.ModelPrompt, err = f.optText("model_prompt", cfg.ModelPrompt); err != nil {
		return cfg, err
	}
	if cfg.NegEpochs, _, err = f.integer("neg_epochs", cfg.NegEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.PosEpochs, _, err = f.integer("pos_epochs", cfg.PosEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Strength, _, err = f.number("strength", cfg.Strength, &zero); err != nil {
		return cfg, err
	}
	if cfg.SandboxTimeout, _, err = f.number("sandbox_timeout", cfg.SandboxTimeout, floatp(0.1)); err != nil {
		return cfg, err
	}
	if cfg.MemoryMB, _, err = f.integer("memory_mb", cfg.MemoryMB, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.IsolateNetwork, err = f.flag("network_isolation", cfg.IsolateNetwork); err != nil {
		return cfg, err
	}
	if err := cfg.Validate(); err != nil {
		return cfg, badRequest("%v", err)
	}
	return cfg, nil
}

// phasesOf reads "both", one phase name or a list of them.
func phasesOf(raw any) ([]string, error) {
	switch value := raw.(type) {
	case string:
		name := strings.ToLower(strings.TrimSpace(value))
		if name == "both" {
			return append([]string{}, radixnet.CodeGenPhases...), nil
		}
		return []string{name}, nil
	case []any:
		out := []string{}
		for _, item := range value {
			name, ok := item.(string)
			if !ok {
				return nil, badRequest("'phases' must be 'both', 'teacher', 'model' or a list of those")
			}
			out = append(out, strings.ToLower(strings.TrimSpace(name)))
		}
		return out, nil
	}
	return nil, badRequest("'phases' must be 'both', 'teacher', 'model' or a list of those")
}

// providerField reads a provider name from one of two field names.
func providerField(rq *request, name, alias, fallback string) (string, error) {
	raw, err := rq.f.optText(name, "")
	if err != nil {
		return "", err
	}
	if raw == "" && alias != "" {
		if raw, err = rq.f.optText(alias, ""); err != nil {
			return "", err
		}
	}
	if raw == "" {
		return fallback, nil
	}
	provider, err := radixnet.NormaliseProvider(raw)
	if err != nil {
		return "", badRequest("'%s' must be one of %s (got %q)", name, strings.Join(radixnet.Providers(), ", "), raw)
	}
	return provider, nil
}

// codegenClients are the teacher and judge clients of a request (url /
// judge_url override the server's defaults).
func codegenClients(rq *request, cfg radixnet.CodeGenConfig) (radixnet.LLMClient, radixnet.LLMClient, error) {
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return nil, nil, err
	}
	url, err := rq.f.optText("url", "")
	if err != nil {
		return nil, nil, err
	}
	client, err := rq.svc.LLMClient(cfg.TeacherProvider, url, cfg.TeacherModel, timeout)
	if err != nil {
		return nil, nil, err
	}
	judgeURL, err := rq.f.optText("judge_url", "")
	if err != nil {
		return nil, nil, err
	}
	if cfg.JudgeProvider == cfg.TeacherProvider && judgeURL == "" {
		return client, client, nil
	}
	judge, err := rq.svc.LLMClient(cfg.JudgeProvider, judgeURL, cfg.ResolvedJudgeModel(), timeout)
	if err != nil {
		return nil, nil, err
	}
	return client, judge, nil
}

func rCodeGenStart(rq *request) (int, any, error) {
	problems, err := problemsFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	config, err := codegenConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, judge, err := codegenClients(rq, config)
	if err != nil {
		return 0, nil, err
	}
	blame, err := rq.f.flag("blame", false)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartCodeGen(problems, config, client, judge, blame)
	if err != nil {
		return 0, nil, err
	}
	ids := make([]string, 0, len(problems))
	for _, problem := range problems {
		ids = append(ids, problem.ID)
	}
	box, err := radixnet.NewSandbox(config.Python, 0, config.MemoryMB, 0, config.IsolateNetwork)
	isolated := config.IsolateNetwork && err == nil && box.NetworkIsolated()
	return 202, map[string]any{
		"job": job, "problems": ids, "config": config,
		"sandbox": map[string]any{
			"timeout": config.SandboxTimeout, "memory_mb": config.MemoryMB, "network_isolated": isolated,
		},
	}, nil
}

func rCodeGenHistory(rq *request) (int, any, error) {
	return 200, rq.svc.CodeGenHistory(), nil
}

func rCodeGenSolve(rq *request) (int, any, error) {
	raw, ok := rq.f.lookup("problem")
	if !ok || raw == nil {
		return 0, nil, badRequest("missing field 'problem' (a string or {prompt, tests, expected_output})")
	}
	problems, err := radixnet.ParseProblems([]any{raw})
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	problem := problems[0]
	source, err := rq.f.optText("source", "model")
	if err != nil {
		return 0, nil, err
	}
	source = strings.ToLower(strings.TrimSpace(source))
	if source != "model" && source != "teacher" {
		return 0, nil, badRequest("'source' must be 'model' or 'teacher' (got %q)", source)
	}
	one := 1
	count, _, err := rq.f.integer("attempts", 1, &one)
	if err != nil {
		return 0, nil, err
	}
	config, err := codegenConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	config.TeacherAttempts, config.ModelAttempts, config.FallbackTeacher = count, count, false
	client, judge, err := codegenClients(rq, config)
	if err != nil {
		return 0, nil, err
	}
	attempts, err := rq.svc.CodeGenSolve(problem, source, count, config, client, judge)
	if err != nil {
		return 0, nil, err
	}
	correct := false
	rows := make([]map[string]any, 0, len(attempts))
	for _, attempt := range attempts {
		rows = append(rows, attempt.ToDict())
		if attempt.Verdict.Correct {
			correct = true
		}
	}
	isolated := false
	if len(attempts) > 0 && attempts[0].Run != nil {
		isolated = attempts[0].Run.NetworkIsolated
	}
	return 200, map[string]any{
		"problem": problem.ToDict(), "source": source, "attempts": rows, "correct": correct,
		"sandbox": map[string]any{"network_isolated": isolated},
	}, nil
}

// CodeGenSolve answers one problem now, without training: the model writes
// under the read lock, the sandbox and the judge run without it.
func (s *Service) CodeGenSolve(
	problem radixnet.Problem, source string, count int, config radixnet.CodeGenConfig, client, judge radixnet.LLMClient,
) ([]*radixnet.Attempt, error) {
	trainer, err := radixnet.NewCodeGenTrainer(s.model, client, config)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	trainer.JudgeClient = judge
	if source == "teacher" {
		attempts, err := trainer.SolveWithTeacher(problem, "teacher", 0, 0)
		if err != nil {
			return nil, badGateway(err)
		}
		return attempts, nil
	}
	codes := make([]string, 0, count)
	_, err = s.read(func(m *radixnet.Model) (any, error) {
		trainer.Model = m
		for i := 0; i < count; i++ {
			code, err := trainer.GenerateWithModel(problem, i)
			if err != nil {
				return nil, err
			}
			codes = append(codes, code)
		}
		return nil, nil
	})
	if err != nil {
		return nil, badRequest("%v", err)
	}
	attempts := []*radixnet.Attempt{}
	for i, code := range codes { // the sandbox and the judge run without the model lock
		attempt, err := trainer.Evaluate(problem, code, "model", i)
		if err != nil {
			return nil, badGateway(err)
		}
		attempts = append(attempts, attempt)
		if attempt.Verdict.Correct {
			break
		}
	}
	return attempts, nil
}

func rCodeGenRun(rq *request) (int, any, error) {
	code, err := rq.f.text("code", nil)
	if err != nil {
		return 0, nil, err
	}
	if strings.TrimSpace(code) == "" {
		return 0, nil, badRequest("'code' is empty")
	}
	if !strings.HasSuffix(code, "\n") {
		code += "\n"
	}
	strictness, err := rq.f.optText("strictness", "strict")
	if err != nil {
		return 0, nil, err
	}
	tests, err := rq.f.optText("tests", "")
	if err != nil {
		return 0, nil, err
	}
	expected, err := rq.f.optText("expected_output", "")
	if err != nil {
		return 0, nil, err
	}
	stdin, err := rq.f.optText("stdin", "")
	if err != nil {
		return 0, nil, err
	}
	timeout, _, err := rq.f.number("sandbox_timeout", 10, floatp(0.1))
	if err != nil {
		return 0, nil, err
	}
	zero := 0
	memory, _, err := rq.f.integer("memory_mb", 256, &zero)
	if err != nil {
		return 0, nil, err
	}
	isolate, err := rq.f.flag("network_isolation", true)
	if err != nil {
		return 0, nil, err
	}
	box, err := radixnet.NewSandbox("", time.Duration(timeout*float64(time.Second)), memory, 0, isolate)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	run, err := box.Run(code, optional(tests), optional(expected), stdin)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	style := radixnet.CheckStyle(code, "")
	verdict, err := radixnet.Decide(run, style, nil, strings.ToLower(strings.TrimSpace(strictness)), "")
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	return 200, map[string]any{"run": run, "style": style, "verdict": verdict}, nil
}

// optional is a pointer to text when it was given, nil when it was not.
func optional(text string) *string {
	if text == "" {
		return nil
	}
	return &text
}
