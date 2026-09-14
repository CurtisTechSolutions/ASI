package main

// `radixnet-count tools` and `radixnet-count agent` / `explore`: the external
// tools the network can call by writing <tool>name {...}</tool>, and the loop
// that teaches it to use them - the LLM writes the acceptance criteria,
// mediates what the network could not write itself, judges the answer,
// demonstrates when it failed, and 2NRL follows.  The Go twin of the Python
// CLI's tools / agent / explore commands.

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// toolFlags are the tool settings of one run (shared by tools, agent, explore
// and serve).
var toolFlags = struct {
	Offline        bool
	AllowPrivate   bool
	SearchURL      string
	Timeout        float64
	MaxBytes       int
	PythonTool     bool
	SandboxTimeout float64
	NoIsolation    bool
	UploadDir      string
}{Timeout: 20, MaxBytes: 2_000_000, SandboxTimeout: 10}

// addToolFlags registers the tool options on a command that offers tools.
func addToolFlags(fs *flag.FlagSet) {
	fs.BoolVar(&toolFlags.Offline, "offline", toolFlags.Offline, "no web tools at all (the calculator and the local tools only)")
	fs.BoolVar(&toolFlags.AllowPrivate, "allow-private", toolFlags.AllowPrivate,
		"allow loopback / private addresses (off: they are refused, which is the SSRF guard)")
	fs.StringVar(&toolFlags.SearchURL, "search-url", toolFlags.SearchURL, "search endpoint ({query} is substituted)")
	fs.Float64Var(&toolFlags.Timeout, "web-timeout", toolFlags.Timeout, "seconds per web request")
	fs.IntVar(&toolFlags.MaxBytes, "max-bytes", toolFlags.MaxBytes, "cap on a fetched page")
	fs.BoolVar(&toolFlags.PythonTool, "python-tool", toolFlags.PythonTool, "also offer the sandboxed `python` tool")
	fs.Float64Var(&toolFlags.SandboxTimeout, "sandbox-timeout", toolFlags.SandboxTimeout, "seconds a sandboxed program may run")
	fs.BoolVar(&toolFlags.NoIsolation, "no-network-isolation", toolFlags.NoIsolation,
		"do not run sandboxed programs in a separate network namespace")
}

// addUploadFlag registers --upload-dir on a command that has no directory of
// its own (serve already has one, and registering it twice would panic).
func addUploadFlag(fs *flag.FlagSet) {
	fs.StringVar(&toolFlags.UploadDir, "upload-dir", toolFlags.UploadDir, "offer `read_file` over this directory")
}

// openToolBox builds the tools of one run.
func openToolBox() *radixnet.ToolBox {
	o := radixnet.ToolOptions{
		Offline: toolFlags.Offline, AllowPrivate: toolFlags.AllowPrivate, SearchURL: toolFlags.SearchURL,
		Timeout: toolFlags.Timeout, MaxBytes: toolFlags.MaxBytes, UploadDir: toolFlags.UploadDir,
	}
	if toolFlags.PythonTool {
		box, err := radixnet.NewSandbox("", time.Duration(toolFlags.SandboxTimeout*float64(time.Second)), 256, 0,
			!toolFlags.NoIsolation)
		if err != nil {
			fail("%v", err)
		}
		o.Sandbox = box
	}
	box, err := radixnet.DefaultToolBox(o)
	if err != nil {
		fail("%v", err)
	}
	return box
}

func cmdTools(args []string) {
	action := ""
	rest := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		action, rest = args[0], args[1:]
	}
	switch action {
	case "", "list":
		cmdToolsList(rest)
	case "describe":
		cmdToolsDescribe(rest)
	case "call":
		cmdToolsCall(rest)
	default:
		fail("unknown tools action %q (have: list, describe, call)", action)
	}
}

func cmdToolsList(args []string) {
	fs := subFlagSet("tools list")
	addToolFlags(fs)
	addUploadFlag(fs)
	_ = fs.Parse(args)
	box := openToolBox()
	if jsonMode {
		emit(map[string]any{"tools": box.Describe(), "offline": toolFlags.Offline})
		return
	}
	say("%d tool(s); the network calls one by writing %s", box.Len(), `<tool>name {"argument": "value"}</tool>`)
	say("")
	for _, tool := range box.Tools() {
		network := ""
		if tool.Network {
			network = "  [network]"
		}
		say("%-56s %s%s", tool.Signature(), tool.Description, network)
	}
}

func cmdToolsDescribe(args []string) {
	fs := subFlagSet("tools describe")
	name := fs.String("tool", "", "the tool to describe")
	addToolFlags(fs)
	addUploadFlag(fs)
	_ = fs.Parse(args)
	box := openToolBox()
	tool, err := box.Get(strings.TrimSpace(*name))
	if err != nil {
		fail("%v", err)
	}
	if jsonMode {
		emit(map[string]any{"tool": tool.ToDict(), "schema": tool.Schema()})
		return
	}
	say("%s", tool.Signature())
	say("%s", tool.Description)
	say("")
	for _, param := range tool.Params {
		required := "optional"
		if param.Required {
			required = "required"
		}
		fallback := ""
		if param.Default != nil {
			fallback = fmt.Sprintf(" (default %v)", param.Default)
		}
		say("  %-14s %-8s %s  %s%s", param.Name, param.Type, required, param.Description, fallback)
	}
	raw, _ := json.MarshalIndent(tool.Schema(), "", "  ")
	say("")
	say("%s", raw)
}

func cmdToolsCall(args []string) {
	fs := subFlagSet("tools call")
	name := fs.String("tool", "", "the tool to call")
	call := fs.String("call", "", `a whole call, e.g. 'web_fetch {"url": "..."}'`)
	var pairs multiFlag
	fs.Var(&pairs, "arg", "an argument as key=value (repeatable)")
	addToolFlags(fs)
	addUploadFlag(fs)
	_ = fs.Parse(args)
	box := openToolBox()
	toolName := strings.TrimSpace(*name)
	arguments := map[string]any{}
	if strings.TrimSpace(*call) != "" {
		text := *call
		if !strings.HasPrefix(strings.TrimLeft(text, " \t"), radixnet.CallOpen) {
			text = radixnet.CallOpen + text + radixnet.CallClose
		}
		parsed := radixnet.ParseCall(text, box)
		if parsed == nil {
			fail("no tool call in %q", *call)
		}
		if !parsed.OK() {
			fail("%s", parsed.Error)
		}
		toolName, arguments = parsed.Name, parsed.Arguments
	} else {
		for _, pair := range pairs {
			key, value, found := strings.Cut(pair, "=")
			if !found {
				fail("--arg takes key=value (got %q)", pair)
			}
			arguments[strings.TrimSpace(key)] = value
		}
	}
	if toolName == "" {
		fail("give --tool NAME (with --arg k=v) or --call 'name {\"arg\": \"value\"}'")
	}
	result := box.Call(toolName, arguments)
	say("%s -> %s in %.3fs", strings.TrimSpace(radixnet.CallText(result.Tool, result.Arguments)),
		map[bool]string{true: "ok", false: "failed"}[result.OK], result.Seconds)
	if !result.OK {
		// a failed call is an observation to the loop, but an error to a shell
		fail("%s: %s", result.Tool, result.Error)
	}
	if jsonMode {
		emit(result)
		return
	}
	fmt.Println(result.Output)
}

// addAgentFlags registers the agent settings shared by `agent` and `explore`.
func addAgentFlags(fs *flag.FlagSet, cfg *radixnet.AgentConfig) (*string, *string, *float64, *bool, *bool, *bool, *bool) {
	phase := fs.String("phase", "model", "model: the network attempts; teacher: the LLM demonstrates; both")
	fs.StringVar(&cfg.Provider, "provider", cfg.Provider, "who plays the four LLM roles: ollama | chatgpt")
	fs.StringVar(&cfg.AgentModel, "agent-model", "", "the model that writes criteria, mediates, judges and teaches")
	fs.StringVar(&cfg.JudgeModel, "judge-model", "", "a different model for judging (default: the agent model)")
	url := fs.String("url", "", "the provider's base URL (default: its own)")
	timeout := fs.Float64("timeout", 0, "seconds to wait for one LLM answer")
	fs.IntVar(&cfg.Rounds, "rounds", cfg.Rounds, "passes over the task list")
	fs.IntVar(&cfg.MaxSteps, "max-steps", cfg.MaxSteps, "tool calls per attempt")
	fs.IntVar(&cfg.ModelAttempts, "model-attempts", cfg.ModelAttempts, "attempts by the network per task")
	fs.IntVar(&cfg.TeacherAttempts, "teacher-attempts", cfg.TeacherAttempts, "demonstrations the teacher may try")
	sampleFirst := fs.Bool("sample-first", false, "sample the first attempt too (default: the beam search)")
	fs.IntVar(&cfg.Candidates, "candidates", cfg.Candidates, "continuations offered per step before the mediator steps in")
	fs.Float64Var(&cfg.Temperature, "temperature", cfg.Temperature, "sampling temperature of the network's attempts")
	fs.IntVar(&cfg.MaxLength, "max-length", cfg.MaxLength, "characters the network writes per step")
	fs.StringVar(&cfg.Mediation, "mediation", cfg.Mediation, "repair (only unusable emissions) | always | never")
	fs.IntVar(&cfg.CriteriaCount, "criteria", cfg.CriteriaCount, "acceptance criteria the LLM writes per task")
	lenient := fs.Bool("lenient", false, "do not require every criterion to be met")
	noJudge := fs.Bool("no-judge", false, "no LLM judge: only a task's own known answer decides")
	noTeach := fs.Bool("no-teach", false, "do not let the teacher demonstrate after a failure")
	fs.IntVar(&cfg.ObservationChars, "observation-chars", cfg.ObservationChars, "characters of a tool's output kept in the transcript")
	fs.StringVar(&cfg.TwoNRLPer, "twonrl-per", cfg.TwoNRLPer, "apply 2NRL after every task, or once per round")
	fs.IntVar(&cfg.ReplayLimit, "replay-limit", cfg.ReplayLimit, "correct transcripts kept for replay (0 = all)")
	fs.BoolVar(&cfg.ReadReward, "read-reward", cfg.ReadReward, "also fine-tune on the text of the pages that were read")
	fs.StringVar(&cfg.BlatantMode, "blatant-mode", cfg.BlatantMode, "none | fail_invert | activation | state")
	fs.Float64Var(&cfg.BlatantMargin, "blatant-margin", cfg.BlatantMargin, "the gap at which a failure counts as blatant")
	fs.Float64Var(&cfg.BlatantBoost, "blatant-boost", cfg.BlatantBoost, "the largest multiplier a failure can earn")
	fs.Float64Var(&cfg.PassScore, "pass-score", cfg.PassScore, "the judge's 0-10 score a correct answer is expected to reach")
	fs.IntVar(&cfg.NegEpochs, "neg-epochs", cfg.NegEpochs, "epochs of the negative (punish) phase")
	fs.IntVar(&cfg.PosEpochs, "pos-epochs", cfg.PosEpochs, "epochs of the positive (reward) phase")
	fs.Float64Var(&cfg.Strength, "strength", cfg.Strength, "how hard 2NRL pushes")
	addToolFlags(fs)
	addUploadFlag(fs)
	return phase, url, timeout, sampleFirst, lenient, noJudge, noTeach
}

// agentSetup finishes a configuration from the flags and builds the loop.
func agentSetup(cfg radixnet.AgentConfig, phase, url string, timeout float64,
	sampleFirst, lenient, noJudge, noTeach, noReplay bool) (*radixnet.AgentTrainer, *radixnet.ToolBox, radixnet.LLMClient) {
	switch strings.ToLower(strings.TrimSpace(phase)) {
	case "both":
		cfg.Phases = append([]string{}, radixnet.AgentPhases...)
	case "model", "teacher":
		cfg.Phases = []string{strings.ToLower(strings.TrimSpace(phase))}
	default:
		fail("--phase must be model, teacher or both (got %q)", phase)
	}
	cfg.FirstDijkstra, cfg.Strict = !sampleFirst, !lenient
	cfg.UseJudge, cfg.TeachOnFailure, cfg.Replay = !noJudge, !noTeach, !noReplay
	cfg.Seed = seedFlag
	if err := cfg.Validate(); err != nil {
		fail("%v", err)
	}
	if cfg.Provider == radixnet.ProviderChatGPT && !radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT play the four roles, " +
			"or use -provider ollama")
	}
	client, err := radixnet.NewLLMClient(cfg.Provider, url, cfg.AgentModel, time.Duration(timeout*float64(time.Second)))
	if err != nil {
		fail("%v", err)
	}
	box := openToolBox()
	model := openModel(false)
	trainer, err := radixnet.NewAgentTrainer(model, client, box, cfg)
	if err != nil {
		fail("%v", err)
	}
	trainer.Stop = interruptible()
	if !jsonMode {
		trainer.Progress = sayAgentRecord
	}
	return trainer, box, client
}

func cmdAgent(args []string) {
	cfg := radixnet.DefaultAgentConfig()
	fs := subFlagSet("agent")
	tasksPath := fs.String("tasks", "", "one question per line, or .json / .jsonl objects {id, prompt, criteria, answer, seeds}")
	report := fs.String("report", "", "write a JSON report (tasks, config, records, solutions)")
	phase, url, timeout, sampleFirst, lenient, noJudge, noTeach := addAgentFlags(fs, &cfg)
	replayOff := fs.Bool("no-replay", false, "do not add earlier correct transcripts to every positive phase")
	_ = fs.Parse(permute(fs, args))
	if strings.TrimSpace(*tasksPath) == "" {
		fail("-tasks FILE is required (one question per line, or .json / .jsonl objects)")
	}
	tasks, err := radixnet.LoadTasks(*tasksPath)
	if err != nil {
		fail("cannot load tasks from %s: %v", *tasksPath, err)
	}
	trainer, box, client := agentSetup(cfg, *phase, *url, *timeout, *sampleFirst, *lenient, *noJudge, *noTeach, *replayOff)
	say("model:  %s", modelPath)
	say("tasks:  %d from %s", len(tasks), *tasksPath)
	say("phases: %s, %d round(s)", strings.Join(trainer.Config.Phases, " -> "), trainer.Config.Rounds)
	say("llm:    %s: %s at %s", trainer.Config.Provider, client.ModelName(), client.BaseURL())
	say("tools:  %s", strings.Join(box.Names(), ", "))
	say("")
	records, err := trainer.Run(tasks)
	if err != nil {
		fail("%v", err)
	}
	agentFinish(trainer, records, len(tasks), "task", *report)
}

func cmdExplore(args []string) {
	cfg := radixnet.DefaultAgentConfig()
	fs := subFlagSet("explore")
	steps := fs.Int("steps", 10, "tasks to work through (0: until interrupted)")
	var seeds multiFlag
	fs.Var(&seeds, "seed-url", "a page to start the exploration from (repeatable)")
	report := fs.String("report", "", "write a JSON report (config, records, solutions)")
	phase, url, timeout, sampleFirst, lenient, noJudge, noTeach := addAgentFlags(fs, &cfg)
	replayOff := fs.Bool("no-replay", false, "do not add earlier correct transcripts to every positive phase")
	_ = fs.Parse(permute(fs, args))
	trainer, box, client := agentSetup(cfg, *phase, *url, *timeout, *sampleFirst, *lenient, *noJudge, *noTeach, *replayOff)
	trainer.Frontier = append(trainer.Frontier, seeds...)
	stepsText := fmt.Sprintf("%d", *steps)
	if *steps == 0 {
		stepsText = "until interrupted"
	}
	say("model:  %s", modelPath)
	say("steps:  %s (the network chooses every task itself)", stepsText)
	say("llm:    %s: %s at %s", trainer.Config.Provider, client.ModelName(), client.BaseURL())
	say("tools:  %s", strings.Join(box.Names(), ", "))
	if len(seeds) > 0 {
		say("seeds:  %s", strings.Join(seeds, ", "))
	}
	say("")
	records, err := trainer.Explore(*steps)
	if err != nil {
		fail("%v", err)
	}
	agentFinish(trainer, records, len(records), "task", *report)
}

// agentFinish saves the model and reports what the run came to.
func agentFinish(trainer *radixnet.AgentTrainer, records []map[string]any, wanted int, unit, report string) {
	saved := saveModel(trainer.Model)
	solved, byModel, runs := 0, 0, 0
	for _, record := range records {
		if record["kind"] != "task" && record["kind"] != "explore" {
			continue
		}
		runs++
		if record["correct"] == true {
			solved++
		}
		if record["model_solved"] == true {
			byModel++
		}
	}
	solutions := trainer.Solutions()
	say("")
	say("%d/%d %s runs solved (%d by the network); %d %s(s) have a correct transcript",
		solved, runs, unit, byModel, len(solutions), unit)
	say("model: %s", saved)
	doc := map[string]any{
		"config": trainer.Config, "records": records, "solutions": solutions, "solved": solved,
		"model_solved": byModel, "runs": runs, "saved": saved,
		"frontier": trainer.Frontier, "visited": trainer.Visited,
	}
	if strings.TrimSpace(report) != "" {
		raw, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			fail("%v", err)
		}
		if err := os.WriteFile(report, append(raw, '\n'), 0o644); err != nil {
			fail("cannot write %s: %v", report, err)
		}
		say("report: %s", report)
	}
	if jsonMode {
		emit(doc)
	}
}

// sayAgentRecord prints one criteria / step / attempt / task record as it lands.
func sayAgentRecord(record map[string]any) {
	switch record["kind"] {
	case "criteria":
		fmt.Printf("%-8s %v: %v\n", record["phase"], record["task"], record["prompt"])
		if list, ok := record["criteria"].([]string); ok {
			for i, criterion := range list {
				fmt.Printf("    %d. %s\n", i+1, criterion)
			}
		}
	case "proposal":
		fmt.Printf("step %v: %v\n", record["step"], record["prompt"])
	case "step":
		mark := "ok"
		if record["ok"] != true {
			mark = "failed"
		}
		fmt.Printf("    step %v %v (%v): %s\n", record["step"], record["tool"], record["source"], mark)
	case "attempt":
		mark := "rejected"
		if record["correct"] == true {
			mark = "correct"
		}
		fmt.Printf("  attempt %v (%v): %s, %v call(s)\n", record["attempt"], record["source"], mark, record["calls"])
	case "task", "explore":
		solved := "unsolved"
		if record["correct"] == true {
			solved = "solved"
		}
		fmt.Printf("%-8s %v: %s by %v\n", record["phase"], record["task"], solved, record["solved_by"])
	case "round":
		fmt.Printf("round %v (%v): %v/%v solved, %v by the network\n", record["round"], record["phase"],
			record["solved"], record["tasks"], record["model_solved"])
	}
}
