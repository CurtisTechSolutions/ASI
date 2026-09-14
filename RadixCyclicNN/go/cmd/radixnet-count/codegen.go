package main

// `radixnet-count codegen`: the model learns to write Python programs.  The
// teacher (a local Ollama model, or ChatGPT) writes each solution, the sandbox
// runs it, the judge confirms it and 2NRL rewards what came out right and
// punishes what did not; in the model phase the network writes the programs
// itself.  The Go twin of the Python CLI's `codegen`, reading the same problem
// files and writing the same JSON report.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

func cmdCodeGen(args []string) {
	cfg := radixnet.DefaultCodeGenConfig()
	fs := subFlagSet("codegen")
	problemsPath := fs.String("problems", "", "one prompt per line, or .json / .jsonl objects {id, prompt, tests, expected_output}")
	phase := fs.String("phase", "both", "teacher: the tutor writes the solutions; model: the network writes them; both")
	rounds := fs.Int("rounds", cfg.Rounds, "passes over the problem list (each pass runs the chosen phases)")
	teacherProvider := fs.String("teacher-provider", cfg.TeacherProvider, "who tutors: ollama | chatgpt")
	provider := fs.String("provider", "", "alias of -teacher-provider")
	teacherModel := fs.String("teacher-model", "", "model that writes, fixes and judges (default: the provider's)")
	judgeProvider := fs.String("judge-provider", "", "judge with the other provider (default: the tutor's)")
	judgeModel := fs.String("judge-model", "", "a different model for judging (default: the teacher model)")
	url := fs.String("url", "", "the tutor's base URL (default: the provider's)")
	judgeURL := fs.String("judge-url", "", "base URL of the judge's provider (default: the same as -url)")
	timeout := fs.Float64("timeout", 0, "seconds to wait for one LLM answer")
	teacherAttempts := fs.Int("teacher-attempts", cfg.TeacherAttempts, "programs the teacher may try per problem (first + fixes)")
	modelAttempts := fs.Int("model-attempts", cfg.ModelAttempts, "programs the network may try before the teacher steps in")
	sampleFirst := fs.Bool("sample-first", false, "sample the network's first attempt too (default: the cheapest path)")
	temperature := fs.Float64("temperature", cfg.Temperature, "sampling temperature of the network's attempts")
	maxLength := fs.Int("max-length", cfg.MaxLength, "characters the network may generate per attempt")
	strictness := fs.String("strictness", cfg.Strictness, "strict: runs + task + PEP 8 + naming; lenient: runs + task")
	noJudge := fs.Bool("no-judge", false, "no LLM judge: the sandbox, the expected output and the tests decide")
	noFallback := fs.Bool("no-fallback-teacher", false, "in the model phase, never ask the teacher for the correct answer")
	twonrlPer := fs.String("twonrl-per", cfg.TwoNRLPer, "apply 2NRL after every problem, or once per round")
	noReplay := fs.Bool("no-replay", false, "do not add earlier correct solutions to every positive phase")
	replayLimit := fs.Int("replay-limit", cfg.ReplayLimit, "correct solutions kept for replay (0 = all)")
	teacherPrompt := fs.String("teacher-prompt", "", "extra instructions for the teacher")
	modelPrompt := fs.String("model-prompt", cfg.ModelPrompt, "prefix template the network continues into code; {problem} is the prompt")
	sandboxTimeout := fs.Float64("sandbox-timeout", cfg.SandboxTimeout, "seconds a program may run")
	memoryMB := fs.Int("memory-mb", cfg.MemoryMB, "memory limit of a program in MB (0 = unlimited)")
	noIsolation := fs.Bool("no-network-isolation", false, "do not run programs in a separate network namespace")
	python := fs.String("python", "", "the interpreter that runs the programs (default: python3)")
	negEpochs := fs.Int("neg-epochs", cfg.NegEpochs, "epochs of the negative (punish) phase")
	posEpochs := fs.Int("pos-epochs", cfg.PosEpochs, "epochs of the positive (reward) phase")
	strength := fs.Float64("strength", cfg.Strength, "how hard 2NRL pushes (the count model's learning rate)")
	report := fs.String("report", "", "write a JSON report (problems, config, records, solutions)")
	blame := fs.Bool("blame", false, "teach the negative network why the rejected programs were rejected")
	addNegativeFlag(fs)
	_ = fs.Parse(permute(fs, args))

	if strings.TrimSpace(*problemsPath) == "" {
		fail("-problems FILE is required (one prompt per line, or .json / .jsonl objects)")
	}
	problems, err := radixnet.LoadProblems(*problemsPath)
	if err != nil {
		fail("cannot load problems from %s: %v", *problemsPath, err)
	}
	if *provider != "" {
		*teacherProvider = *provider
	}
	cfg.TeacherProvider, cfg.TeacherModel = *teacherProvider, *teacherModel
	cfg.JudgeProvider, cfg.JudgeModel = *judgeProvider, *judgeModel
	cfg.Phases = phasesFor(*phase)
	cfg.Rounds, cfg.TeacherAttempts, cfg.ModelAttempts = *rounds, *teacherAttempts, *modelAttempts
	cfg.FirstDijkstra, cfg.Temperature, cfg.MaxLength = !*sampleFirst, *temperature, *maxLength
	cfg.Strictness, cfg.UseJudge, cfg.FallbackTeacher = strings.ToLower(*strictness), !*noJudge, !*noFallback
	cfg.TwoNRLPer, cfg.Replay, cfg.ReplayLimit = strings.ToLower(*twonrlPer), !*noReplay, *replayLimit
	cfg.TeacherPrompt, cfg.ModelPrompt = *teacherPrompt, *modelPrompt
	cfg.NegEpochs, cfg.PosEpochs, cfg.Strength = *negEpochs, *posEpochs, *strength
	cfg.SandboxTimeout, cfg.MemoryMB, cfg.IsolateNetwork = *sandboxTimeout, *memoryMB, !*noIsolation
	cfg.Python = *python
	if err := cfg.Validate(); err != nil {
		fail("%v", err)
	}
	needsKey := cfg.TeacherProvider == radixnet.ProviderChatGPT ||
		(cfg.UseJudge && cfg.JudgeProvider == radixnet.ProviderChatGPT)
	if needsKey && !radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT tutor, " +
			"or use -teacher-provider ollama")
	}
	wait := time.Duration(*timeout * float64(time.Second))
	client, err := radixnet.NewLLMClient(cfg.TeacherProvider, *url, cfg.TeacherModel, wait)
	if err != nil {
		fail("%v", err)
	}
	judge := client
	if cfg.JudgeProvider != cfg.TeacherProvider || *judgeURL != "" {
		if judge, err = radixnet.NewLLMClient(cfg.JudgeProvider, *judgeURL, cfg.ResolvedJudgeModel(), wait); err != nil {
			fail("%v", err)
		}
	}
	model := openModel(false)
	trainer, err := radixnet.NewCodeGenTrainer(model, client, cfg)
	if err != nil {
		fail("%v", err)
	}
	trainer.JudgeClient = judge
	var negative *radixnet.Model
	if *blame {
		negative = openNegative(false)
		trainer.Negative = negative
	}

	say("model:    %s", modelPath)
	say("problems: %d from %s", len(problems), *problemsPath)
	say("phases:   %s, %d round(s)", strings.Join(cfg.Phases, " -> "), cfg.Rounds)
	say("teacher:  %s: %s at %s", cfg.TeacherProvider, client.ModelName(), client.BaseURL())
	if cfg.UseJudge {
		say("judge:    %s: %s at %s", cfg.JudgeProvider, judge.ModelName(), judge.BaseURL())
	} else {
		say("judge:    off (the sandbox, the expected output and the tests decide)")
	}
	isolated := "NOT isolated"
	if trainer.Sandbox.NetworkIsolated() {
		isolated = "isolated"
	}
	say("sandbox:  timeout=%gs memory=%dMB network=%s", cfg.SandboxTimeout, cfg.MemoryMB, isolated)
	say("2NRL:     per %s, negative %d epoch(s), positive %d epoch(s), strength %g, replay %s, %s",
		cfg.TwoNRLPer, cfg.NegEpochs, cfg.PosEpochs, cfg.Strength,
		map[bool]string{true: "on", false: "off"}[cfg.Replay], cfg.Strictness)
	if negative != nil {
		say("negative: %s", negativeFile())
	}
	say("")

	trainer.Stop = interruptible()
	if !jsonMode {
		trainer.Progress = func(record map[string]any) { sayCodeGenRecord(record) }
	}
	records, err := trainer.Run(problems)
	if err != nil {
		fail("%v", err)
	}
	saved := saveModel(model)
	negativePath := ""
	if negative != nil {
		negativePath = saveNegative(negative)
	}
	solved, byModel, runs := 0, 0, 0
	for _, record := range records {
		if record["kind"] != "problem" {
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
	say("%d/%d problem runs solved (%d by the model); %d of %d problems have a correct solution",
		solved, runs, byModel, len(solutions), len(problems))
	say("model: %s", saved)
	if negativePath != "" {
		say("negative model: %s", negativePath)
		reasonTable(negative, 10)
	}
	doc := map[string]any{
		"problems": len(problems), "config": cfg, "records": records, "solutions": solutions,
		"solved": solved, "model_solved": byModel, "runs": runs, "saved": saved,
		"teacher":  map[string]any{"provider": cfg.TeacherProvider, "model": client.ModelName(), "url": client.BaseURL()},
		"negative": nil,
	}
	if negativePath != "" {
		doc["negative"] = map[string]any{
			"path": negativePath, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	if strings.TrimSpace(*report) != "" {
		raw, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			fail("%v", err)
		}
		if err := os.WriteFile(*report, append(raw, '\n'), 0o644); err != nil {
			fail("cannot write %s: %v", *report, err)
		}
		say("report: %s", *report)
	}
	if jsonMode {
		emit(doc)
	}
}

// phasesFor turns -phase into the phases to run.
func phasesFor(name string) []string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "both":
		return append([]string{}, radixnet.CodeGenPhases...)
	default:
		return []string{strings.ToLower(strings.TrimSpace(name))}
	}
}

// sayCodeGenRecord prints one attempt / problem / round record as it lands.
func sayCodeGenRecord(record map[string]any) {
	switch record["kind"] {
	case "attempt":
		mark := "rejected"
		if record["correct"] == true {
			mark = "correct"
		}
		fmt.Printf("  %-7s attempt %v (%v): %s%s\n", record["phase"], record["attempt"], record["source"], mark,
			reasonText(record["error"]))
	case "problem":
		solved := "unsolved"
		if record["correct"] == true {
			solved = "solved"
		}
		fmt.Printf("%-7s %-14s %s after %v attempt(s)\n", record["phase"], record["problem"], solved, record["attempts"])
	case "round":
		fmt.Printf("round %v (%v): %v/%v solved, %v by the model, %vs\n", record["round"], record["phase"],
			record["solved"], record["problems"], record["model_solved"], record["seconds"])
	}
}

func reasonText(value any) string {
	if text, ok := value.(string); ok && text != "" {
		return " (" + text + ")"
	}
	return ""
}
