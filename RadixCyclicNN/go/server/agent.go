package server

// Tool use over HTTP: GET /api/tools says what the network may call,
// POST /api/tools/call runs one, /api/agent/start works a list of tasks and
// /api/agent/explore lets the network choose them itself.  The Go twin of the
// Python server's tools and agent endpoints, answering the same JSON.

import (
	"path/filepath"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// toolDefaults are the server's own tool settings, which a request may narrow.
type toolDefaults struct {
	Offline       bool
	AllowPrivate  bool
	SearchURL     string
	WebTimeout    float64
	MaxBytes      int
	PythonTool    bool
	SandboxTime   float64
	NetworkIsolat bool
}

// defaultToolDefaults mirror the Python server's.
func defaultToolDefaults() toolDefaults {
	return toolDefaults{WebTimeout: 20, MaxBytes: 2_000_000, SandboxTime: 10, NetworkIsolat: true}
}

// ToolBox builds the tools of one request: the server's defaults with the
// request's overrides applied.
func (s *Service) ToolBox(o radixnet.ToolOptions) (*radixnet.ToolBox, error) {
	box, err := radixnet.DefaultToolBox(o)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return box, nil
}

// DescribeTools is what the network may call, and how.
func (s *Service) DescribeTools() (map[string]any, error) {
	box, err := s.ToolBox(s.toolOptions())
	if err != nil {
		return nil, err
	}
	uploads := ""
	if s.uploads != nil {
		uploads = s.uploads.Dir
	}
	return map[string]any{
		"tools": box.Describe(), "names": box.Names(), "count": box.Len(),
		"options": map[string]any{
			"offline": s.tools.Offline, "allow_private": s.tools.AllowPrivate, "search_url": nilText(s.tools.SearchURL),
			"web_timeout": s.tools.WebTimeout, "max_bytes": s.tools.MaxBytes, "python_tool": s.tools.PythonTool,
			"sandbox_timeout": s.tools.SandboxTime,
		},
		"upload_dir":  nilText(uploads),
		"call_format": `<tool>name {"argument": "value"}</tool>`,
	}, nil
}

func nilText(text string) any {
	if text == "" {
		return nil
	}
	return text
}

// toolOptions are the server's defaults as the library takes them.
func (s *Service) toolOptions() radixnet.ToolOptions {
	o := radixnet.ToolOptions{
		Offline: s.tools.Offline, AllowPrivate: s.tools.AllowPrivate, SearchURL: s.tools.SearchURL,
		Timeout: s.tools.WebTimeout, MaxBytes: s.tools.MaxBytes,
	}
	if s.uploads != nil {
		o.UploadDir = s.uploads.Dir
	}
	if s.tools.PythonTool {
		if box, err := radixnet.NewSandbox("", time.Duration(s.tools.SandboxTime*float64(time.Second)), 256, 0,
			s.tools.NetworkIsolat); err == nil {
			o.Sandbox = box
		}
	}
	return o
}

// StartAgent starts an `agent` job: criteria, tool calls, judging, teaching
// and 2NRL over the tasks.
func (s *Service) StartAgent(
	tasks []radixnet.Task, config radixnet.AgentConfig, client radixnet.LLMClient, box *radixnet.ToolBox,
) (map[string]any, error) {
	if len(tasks) == 0 {
		return nil, badRequest("no tasks to solve")
	}
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	return s.startJob("agent", func(job *Job, progress func(map[string]any), stop func() bool) error {
		trainer, err := s.agentTrainer(config, client, box, progress, stop, nil)
		if err != nil {
			return err
		}
		_, err = trainer.Run(tasks)
		return err
	})
}

// StartExplore starts an `explore` job: the network chooses every task itself
// (steps 0 = until stopped).
func (s *Service) StartExplore(
	steps int, config radixnet.AgentConfig, client radixnet.LLMClient, box *radixnet.ToolBox, seeds []string,
) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	return s.startJob("explore", func(job *Job, progress func(map[string]any), stop func() bool) error {
		trainer, err := s.agentTrainer(config, client, box, progress, stop, seeds)
		if err != nil {
			return err
		}
		_, err = trainer.Explore(steps)
		return err
	})
}

// agentTrainer builds the loop a job runs, with the model lock released around
// the slow outside work.
func (s *Service) agentTrainer(
	config radixnet.AgentConfig, client radixnet.LLMClient, box *radixnet.ToolBox,
	progress func(map[string]any), stop func() bool, seeds []string,
) (*radixnet.AgentTrainer, error) {
	trainer, err := radixnet.NewAgentTrainer(s.model, client, box, config)
	if err != nil {
		return nil, err
	}
	trainer.Frontier = append(trainer.Frontier, seeds...)
	trainer.Progress = func(record map[string]any) {
		s.agentHistory = append(s.agentHistory, record)
		progress(record)
	}
	trainer.Stop = stop
	// the tools browse and the LLM thinks without the model lock, so readers stay served
	trainer.External = func(fn func() error) error {
		s.mu.Unlock()
		defer s.mu.Lock()
		return fn()
	}
	return trainer, nil
}

// AgentHistory is the criteria / step / attempt / task records of every agent
// and explore run.
func (s *Service) AgentHistory() map[string]any {
	history := s.agentHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("GET", "/api/tools", rTools)
	doc("GET", "/api/tools", "the external tools the network can call by writing <tool>name {...}</tool>: names, "+
		"arguments, JSON schemas")
	route("POST", "/api/tools/call", rToolCall)
	doc("POST", "/api/tools/call", "call one tool directly: {tool, arguments} or {call: 'name {\"arg\": \"value\"}'} "+
		"+ tool overrides {offline, allow_private, search_url, web_timeout, max_bytes, python_tool} -> the result")
	route("POST", "/api/agent/start", rAgentStart)
	doc("POST", "/api/agent/start", "start an agent job over tasks: {tasks | tasks_text | task_files, phase: "+
		"model|teacher|both, rounds, max_steps, mediation: repair|always|never, criteria, judge, teach, "+
		"blatant_mode, blatant_margin, blatant_boost, 2NRL options}")
	route("POST", "/api/agent/explore", rAgentExplore)
	doc("POST", "/api/agent/explore", "start an explore job - the network chooses every task itself: "+
		"{steps (0 = until stopped), seed_urls, ...the agent options}")
	route("GET", "/api/agent/history", rAgentHistory)
	doc("GET", "/api/agent/history", "criteria / step / attempt / task records of all agent and explore runs")
	route("POST", "/api/agent/criteria", rAgentCriteria)
	doc("POST", "/api/agent/criteria", "the acceptance criteria the LLM writes for {tasks}, without attempting anything")
	route("POST", "/api/agent/solve", rAgentSolve)
	doc("POST", "/api/agent/solve", "one task through the loop without training: {task, source: model|teacher, ...} "+
		"-> the transcript, the verdict and how badly it failed")
}

// toolBoxFrom is the toolbox of one request: the server's defaults with the
// request's overrides applied.
func toolBoxFrom(rq *request) (*radixnet.ToolBox, error) {
	o := rq.svc.toolOptions()
	flag := func(name string, current bool) (bool, error) {
		if rq.f.present(name) {
			return rq.f.flag(name, current)
		}
		if raw, ok := rq.queryValue(name); ok {
			value := strings.ToLower(strings.TrimSpace(raw))
			return value == "1" || value == "true" || value == "yes" || value == "on", nil
		}
		return current, nil
	}
	var err error
	if o.Offline, err = flag("offline", o.Offline); err != nil {
		return nil, err
	}
	if o.AllowPrivate, err = flag("allow_private", o.AllowPrivate); err != nil {
		return nil, err
	}
	python, err := flag("python_tool", o.Sandbox != nil)
	if err != nil {
		return nil, err
	}
	if !python {
		o.Sandbox = nil
	} else if o.Sandbox == nil {
		if box, err := radixnet.NewSandbox("", time.Duration(rq.svc.tools.SandboxTime*float64(time.Second)), 256, 0,
			rq.svc.tools.NetworkIsolat); err == nil {
			o.Sandbox = box
		}
	}
	if url, err := rq.f.optText("search_url", o.SearchURL); err == nil && url != "" {
		o.SearchURL = url
	}
	if timeout, ok, err := rq.f.number("web_timeout", o.Timeout, floatp(0.1)); err != nil {
		return nil, err
	} else if ok {
		o.Timeout = timeout
	}
	minBytes := 1024
	if bytes, ok, err := rq.f.integer("max_bytes", o.MaxBytes, &minBytes); err != nil {
		return nil, err
	} else if ok {
		o.MaxBytes = bytes
	}
	return rq.svc.ToolBox(o)
}

func rTools(rq *request) (int, any, error) {
	out, err := rq.svc.DescribeTools()
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rToolCall(rq *request) (int, any, error) {
	box, err := toolBoxFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	raw, err := rq.f.optText("call", "")
	if err != nil {
		return 0, nil, err
	}
	name := ""
	arguments := map[string]any{}
	if strings.TrimSpace(raw) != "" {
		text := raw
		if !strings.HasPrefix(strings.TrimLeft(text, " \t"), radixnet.CallOpen) {
			text = radixnet.CallOpen + text + radixnet.CallClose
		}
		call := radixnet.ParseCall(text, box)
		if call == nil {
			return 0, nil, badRequest("no tool call in %s", radixnet.PythonRepr(clipText(raw, 120)))
		}
		if !call.OK() {
			return 0, nil, badRequest("%s", call.Error)
		}
		name, arguments = call.Name, call.Arguments
	} else {
		if name, err = rq.f.text("tool", nil); err != nil {
			return 0, nil, err
		}
		name = strings.TrimSpace(name)
		if given, ok := rq.f.lookup("arguments"); ok && given != nil {
			object, ok := given.(map[string]any)
			if !ok {
				return 0, nil, badRequest("'arguments' must be an object")
			}
			arguments = object
		}
	}
	if !box.Has(name) {
		return 0, nil, &apiError{404, "unknown tool " + radixnet.PythonRepr(name) + " (have: " + strings.Join(box.Names(), ", ") + ")"}
	}
	// no tool touches the model, so this runs outside the lock
	return 200, box.Call(name, arguments), nil
}

// tasksFrom reads the tasks of a request: `tasks` (strings or objects),
// `tasks_text` (one per line) and `task_files` (uploads).
func tasksFrom(rq *request) ([]radixnet.Task, error) {
	items := []any{}
	if raw, ok := rq.f.lookup("tasks"); ok && raw != nil {
		list, ok := raw.([]any)
		if !ok {
			return nil, badRequest("'tasks' must be a list of strings or {prompt, criteria, answer, seeds} objects")
		}
		items = append(items, list...)
	}
	if raw, ok := rq.f.lookup("task"); ok && raw != nil {
		items = append(items, raw)
	}
	text, err := rq.f.optText("tasks_text", "")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "#") {
			items = append(items, trimmed)
		}
	}
	names, err := rq.f.names("task_files")
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		contents, err := rq.svc.UploadTexts([]string{name}, "file", 0, true)
		if err != nil {
			return nil, err
		}
		for _, content := range contents {
			parsed, err := radixnet.ParseTaskFile(content, filepath.Ext(name))
			if err != nil {
				return nil, badRequest("upload %s: %v", radixnet.PythonRepr(name), err)
			}
			for _, task := range parsed {
				items = append(items, task.ToDict())
			}
		}
	}
	if len(items) == 0 {
		return nil, badRequest("missing 'tasks' (list of strings / objects), 'tasks_text' (one per line) " +
			"or 'task_files' (upload names)")
	}
	tasks, err := radixnet.ParseTasks(items)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return tasks, nil
}

// agentConfigFrom reads the agent settings of a request body.
func agentConfigFrom(rq *request) (radixnet.AgentConfig, error) {
	cfg := radixnet.DefaultAgentConfig()
	zeroI, oneI := 0, 1
	zero := 0.0
	f := rq.f
	phase, err := f.optText("phase", "model")
	if err != nil {
		return cfg, err
	}
	phase = strings.ToLower(strings.TrimSpace(phase))
	switch phase {
	case "both":
		cfg.Phases = append([]string{}, radixnet.AgentPhases...)
	case "model", "teacher":
		cfg.Phases = []string{phase}
	default:
		return cfg, badRequest("'phase' must be 'model', 'teacher' or 'both' (got %s)", radixnet.PythonRepr(phase))
	}
	if cfg.Provider, err = providerField(rq, "provider", "", cfg.Provider); err != nil {
		return cfg, err
	}
	if cfg.AgentModel, err = f.optText("agent_model", ""); err != nil {
		return cfg, err
	}
	if cfg.AgentModel == "" {
		if cfg.AgentModel, err = f.optText("model", ""); err != nil {
			return cfg, err
		}
	}
	if cfg.AgentModel == "" {
		cfg.AgentModel = radixnet.DefaultAgentModel()
	}
	if cfg.JudgeModel, err = f.optText("judge_model", ""); err != nil {
		return cfg, err
	}
	if cfg.Rounds, _, err = f.integer("rounds", cfg.Rounds, &oneI); err != nil {
		return cfg, err
	}
	if cfg.MaxSteps, _, err = f.integer("max_steps", cfg.MaxSteps, &oneI); err != nil {
		return cfg, err
	}
	if cfg.ModelAttempts, _, err = f.integer("model_attempts", cfg.ModelAttempts, &oneI); err != nil {
		return cfg, err
	}
	if cfg.TeacherAttempts, _, err = f.integer("teacher_attempts", cfg.TeacherAttempts, &oneI); err != nil {
		return cfg, err
	}
	if cfg.FirstDijkstra, err = f.flag("first_attempt_dijkstra", cfg.FirstDijkstra); err != nil {
		return cfg, err
	}
	if cfg.Candidates, _, err = f.integer("candidates", cfg.Candidates, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Temperature, _, err = f.number("temperature", cfg.Temperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.MaxLength, _, err = f.integer("max_length", cfg.MaxLength, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Mediation, err = f.optText("mediation", cfg.Mediation); err != nil {
		return cfg, err
	}
	cfg.Mediation = strings.ToLower(strings.TrimSpace(cfg.Mediation))
	if !containsText(radixnet.Mediation, cfg.Mediation) {
		return cfg, badRequest("'mediation' must be one of %s (got %s)",
			strings.Join(radixnet.Mediation, ", "), radixnet.PythonRepr(cfg.Mediation))
	}
	if cfg.CriteriaCount, _, err = f.integer("criteria", cfg.CriteriaCount, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Strict, err = f.flag("strict", cfg.Strict); err != nil {
		return cfg, err
	}
	if cfg.UseJudge, err = f.flag("judge", cfg.UseJudge); err != nil {
		return cfg, err
	}
	if cfg.TeachOnFailure, err = f.flag("teach", cfg.TeachOnFailure); err != nil {
		return cfg, err
	}
	if cfg.ObservationChars, _, err = f.integer("observation_chars", cfg.ObservationChars, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.TwoNRLPer, err = f.optText("twonrl_per", cfg.TwoNRLPer); err != nil {
		return cfg, err
	}
	cfg.TwoNRLPer = strings.ToLower(strings.TrimSpace(cfg.TwoNRLPer))
	if cfg.TwoNRLPer != "task" && cfg.TwoNRLPer != "round" {
		return cfg, badRequest("'twonrl_per' must be 'task' or 'round' (got %s)", radixnet.PythonRepr(cfg.TwoNRLPer))
	}
	if cfg.Replay, err = f.flag("replay", cfg.Replay); err != nil {
		return cfg, err
	}
	if cfg.ReplayLimit, _, err = f.integer("replay_limit", cfg.ReplayLimit, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.ReadReward, err = f.flag("read_reward", cfg.ReadReward); err != nil {
		return cfg, err
	}
	if cfg.BlatantMode, err = f.optText("blatant_mode", cfg.BlatantMode); err != nil {
		return cfg, err
	}
	cfg.BlatantMode = strings.ToLower(strings.TrimSpace(cfg.BlatantMode))
	if !containsText(radixnet.BlatantModes, cfg.BlatantMode) {
		return cfg, badRequest("'blatant_mode' must be one of %s (got %s)",
			strings.Join(radixnet.BlatantModes, ", "), radixnet.PythonRepr(cfg.BlatantMode))
	}
	if cfg.BlatantMargin, _, err = f.number("blatant_margin", cfg.BlatantMargin, floatp(0.001)); err != nil {
		return cfg, err
	}
	if cfg.BlatantBoost, _, err = f.number("blatant_boost", cfg.BlatantBoost, floatp(1)); err != nil {
		return cfg, err
	}
	if cfg.PassScore, _, err = f.number("pass_score", cfg.PassScore, floatp(0.001)); err != nil {
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
	if seed, ok, err := f.integer("seed", 0, nil); err != nil {
		return cfg, err
	} else if ok {
		cfg.Seed = int64(seed)
	}
	if err := cfg.Validate(); err != nil {
		return cfg, badRequest("%v", err)
	}
	return cfg, nil
}

func containsText(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

// agentClient is the LLM of an agent request (url / timeout override the
// server's defaults).
func agentClient(rq *request, cfg radixnet.AgentConfig) (radixnet.LLMClient, error) {
	url, err := rq.f.optText("url", "")
	if err != nil {
		return nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return nil, err
	}
	return rq.svc.LLMClient(cfg.Provider, url, cfg.AgentModel, timeout)
}

func rAgentStart(rq *request) (int, any, error) {
	tasks, err := tasksFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	config, err := agentConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	box, err := toolBoxFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := agentClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartAgent(tasks, config, client, box)
	if err != nil {
		return 0, nil, err
	}
	rows := make([]map[string]any, 0, len(tasks))
	for _, task := range tasks {
		rows = append(rows, task.ToDict())
	}
	return 202, map[string]any{"job": job, "tasks": rows, "tools": box.Names(), "config": config}, nil
}

func rAgentExplore(rq *request) (int, any, error) {
	config, err := agentConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	box, err := toolBoxFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := agentClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	zero := 0
	steps, _, err := rq.f.integer("steps", 10, &zero)
	if err != nil {
		return 0, nil, err
	}
	seeds, err := rq.f.textsOptional("seed_urls", "seed_url")
	if err != nil {
		return 0, nil, err
	}
	kept := []string{}
	for _, seed := range seeds {
		if strings.TrimSpace(seed) != "" {
			kept = append(kept, seed)
		}
	}
	job, err := rq.svc.StartExplore(steps, config, client, box, kept)
	if err != nil {
		return 0, nil, err
	}
	var reported any
	if steps > 0 {
		reported = steps
	}
	return 202, map[string]any{"job": job, "steps": reported, "seeds": kept, "tools": box.Names(),
		"config": config}, nil
}

func rAgentHistory(rq *request) (int, any, error) {
	return 200, rq.svc.AgentHistory(), nil
}

func rAgentCriteria(rq *request) (int, any, error) {
	tasks, err := tasksFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	config, err := agentConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := agentClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	out := []map[string]any{}
	for _, task := range tasks {
		criteria, err := radixnet.WriteCriteria(client, task, config.AgentModel, config.CriteriaCount)
		if err != nil {
			return 0, nil, badGateway(err)
		}
		out = append(out, map[string]any{"task": task.ID, "prompt": task.Prompt, "criteria": criteria})
	}
	return 200, map[string]any{"tasks": out, "model": client.ModelName(), "url": client.BaseURL()}, nil
}

func rAgentSolve(rq *request) (int, any, error) {
	tasks, err := tasksFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	if len(tasks) != 1 {
		return 0, nil, badRequest("/api/agent/solve takes exactly one task (got %d)", len(tasks))
	}
	task := tasks[0]
	source, err := rq.f.optText("source", "model")
	if err != nil {
		return 0, nil, err
	}
	source = strings.ToLower(strings.TrimSpace(source))
	if source != "model" && source != "teacher" {
		return 0, nil, badRequest("'source' must be 'model' or 'teacher' (got %s)", radixnet.PythonRepr(source))
	}
	config, err := agentConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	config.TeachOnFailure = false
	box, err := toolBoxFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := agentClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.AgentSolve(task, source, config, client, box)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

// AgentSolve works one task through the loop without training: the network
// writes under the read lock, the tools and the LLM run without it.
func (s *Service) AgentSolve(
	task radixnet.Task, source string, config radixnet.AgentConfig, client radixnet.LLMClient, box *radixnet.ToolBox,
) (map[string]any, error) {
	trainer, err := radixnet.NewAgentTrainer(s.model, client, box, config)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	criteria, err := trainer.CriteriaFor(task)
	if err != nil {
		return nil, badGateway(err)
	}
	var attempt *radixnet.AgentAttempt
	if source == "model" {
		attempt, err = trainer.SolveWithModel(task, 0, "model", 1)
	} else {
		attempt, err = trainer.SolveWithTeacher(task, criteria, 0, "")
	}
	if err != nil {
		return nil, badGateway(err)
	}
	verdict, err := trainer.Judge(task, criteria, attempt)
	if err != nil {
		return nil, badGateway(err)
	}
	attempt.Verdict = verdict
	frontier := trainer.Frontier
	if len(frontier) > 20 {
		frontier = frontier[:20]
	}
	return map[string]any{
		"task": task.ToDict(), "source": source, "criteria": criteria, "attempt": attempt.ToDict(),
		"transcript": attempt.Text, "correct": verdict.Correct, "gap": trainer.GapOf(attempt, criteria),
		"frontier": frontier, "tools": box.Names(),
	}, nil
}

// clipText is the first n characters of a text, for an error message.
func clipText(text string, n int) string {
	runes := []rune(text)
	if len(runes) <= n {
		return text
	}
	return string(runes[:n])
}
