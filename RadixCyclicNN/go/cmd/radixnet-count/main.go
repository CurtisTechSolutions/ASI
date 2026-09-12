// Command radixnet-count is the Go CLI of the count / reward model: train,
// predict, generate, score, feedback / 2NRL, invert, weights, info and
// converse, with model files interchangeable with the Python implementation.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/server"
)

const version = "0.1.0"

// bom is the UTF-8 byte order mark some editors put in front of a text file.
var bom = string([]byte{0xEF, 0xBB, 0xBF})

var (
	modelPath = "model.count.json"
	jsonMode  bool
	seedFlag  int64
	workers   = runtime.NumCPU()
	outPath   string
)

func addGlobalFlags(fs *flag.FlagSet) {
	fs.StringVar(&modelPath, "model", modelPath, "model file to load / save (gzip when the name ends with .gz)")
	fs.BoolVar(&jsonMode, "json", jsonMode, "print one JSON document instead of human-readable text")
	fs.Int64Var(&seedFlag, "seed", seedFlag, "RNG seed for a new model and for sampling")
	fs.IntVar(&workers, "workers", workers, "goroutines fanned out over texts and nodes")
	fs.StringVar(&outPath, "out", outPath, "where to save the model (default: --model)")
}

func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	addGlobalFlags(fs)
	return fs
}

// subFlagSet is a command's flag set; the global options are accepted here too.
func subFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	addGlobalFlags(fs)
	return fs
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "radixnet-count: error: "+format+"\n", args...)
	os.Exit(1)
}

func emit(doc any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		fail("%v", err)
	}
}

func say(format string, args ...any) {
	if !jsonMode {
		fmt.Printf(format+"\n", args...)
	}
}

func readTexts(paths []string, unit string, pageLines int) []string {
	var texts []string
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			fail("%v", err)
		}
		texts = append(texts, radixnet.SplitTexts(strings.TrimPrefix(string(raw), bom), unit, pageLines)...)
	}
	if len(texts) == 0 {
		fail("no texts found in %s", strings.Join(paths, ", "))
	}
	return texts
}

func openModel(required bool) *radixnet.Model {
	radixnet.Workers = workers
	if _, err := os.Stat(modelPath); err == nil {
		m, err := radixnet.Load(modelPath)
		if err != nil {
			fail("%s: %v", modelPath, err)
		}
		m.Workers = workers
		return m
	}
	if required {
		fail("model file not found: %s (train one first with `radixnet-count train --data FILE`)", modelPath)
	}
	m, err := radixnet.NewModel(seedFlag, radixnet.DefaultGraphOptions())
	if err != nil {
		fail("%v", err)
	}
	m.Workers = workers
	return m
}

func saveModel(m *radixnet.Model) string {
	path := outPath
	if path == "" {
		path = modelPath
	}
	if err := m.Save(path); err != nil {
		fail("cannot save %s: %v", path, err)
	}
	return path
}

func usage() {
	fmt.Fprintf(os.Stderr, `radixnet-count %s - the count / reward model in Go

usage: radixnet-count [global options] <command> [options]

commands:
  train      count traversals of the texts of --data files (lines, paragraphs or pages)
  predict    top-K / bottom-K continuations of --prefix (beam) or a stochastic walk
  generate   whole texts from the prediction search (beam / sample / dijkstra)
  score      log-probability of --text or every text of --data
  feedback   thumbs up (--good / --good-text) and thumbs down (--bad / --bad-text)
  2nrl       penalise --bad texts, then count + reward --good texts
  invert     flip the sign of every reward
  weights    show or change the dual frequency weight function
  info       statistics and the training history tail
  converse   the model talks to itself
  serve      HTTP API (+ the prebuilt frontend) speaking the Python server's JSON contract
  version    print the version

global options (before or after the command): --model PATH --json --seed N --workers N --out PATH
`, version)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	// global flags may come before the command
	fs := newFlagSet("radixnet-count")
	fs.Usage = usage
	args := os.Args[1:]
	cmdIndex := 0
	for cmdIndex < len(args) && strings.HasPrefix(args[cmdIndex], "-") {
		// consume "--flag value" or "--flag=value"
		if strings.Contains(args[cmdIndex], "=") || args[cmdIndex] == "--json" || args[cmdIndex] == "-json" {
			cmdIndex++
		} else {
			cmdIndex += 2
		}
	}
	if cmdIndex > len(args) {
		cmdIndex = len(args)
	}
	if err := fs.Parse(args[:cmdIndex]); err != nil {
		os.Exit(2)
	}
	if cmdIndex >= len(args) {
		usage()
		os.Exit(2)
	}
	command, rest := args[cmdIndex], args[cmdIndex+1:]
	switch command {
	case "train":
		cmdTrain(rest)
	case "predict":
		cmdPredict(rest)
	case "generate":
		cmdGenerate(rest)
	case "score":
		cmdScore(rest)
	case "feedback":
		cmdFeedback(rest)
	case "2nrl":
		cmdTwoNRL(rest)
	case "invert":
		cmdInvert(rest)
	case "weights":
		cmdWeights(rest)
	case "info":
		cmdInfo(rest)
	case "converse":
		cmdConverse(rest)
	case "serve":
		cmdServe(rest)
	case "version":
		if jsonMode {
			emit(map[string]any{"version": version, "go": runtime.Version()})
		} else {
			fmt.Println(version)
		}
	case "help", "-h", "--help":
		usage()
	default:
		fail("unknown command %q", command)
	}
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func cmdTrain(args []string) {
	fs := subFlagSet("train")
	var data multiFlag
	fs.Var(&data, "data", "text file (repeatable)")
	unit := fs.String("split", "lines", "how a file is cut into texts, each handled by its own goroutine: lines | paragraphs | pages | file")
	pageLines := fs.Int("page-lines", 50, "lines per page when a file has no form feeds (--split pages)")
	epochs := fs.Int("epochs", 5, "passes over the texts")
	noCompress := fs.Bool("no-compress", false, "do not merge unary chains after every epoch")
	window := fs.Int("window", 0, "sliding window size for a NEW model (default 10000)")
	globalScale := fs.Float64("global-scale", -1, "weight of log(all-time share) for a NEW model (default 0.5)")
	windowScale := fs.Float64("window-scale", -1, "weight of log(recent share) for a NEW model (default 0.5)")
	rewardScale := fs.Float64("reward-scale", -1, "weight of the reward for a NEW model (default 1)")
	countScale := fs.Float64("count-scale", -1, "weight of log(1 + traversals) for a NEW model (default 0)")
	_ = fs.Parse(args)
	if len(data) == 0 {
		fail("train needs --data FILE")
	}
	texts := readTexts(data, *unit, *pageLines)
	radixnet.Workers = workers
	var m *radixnet.Model
	if _, err := os.Stat(modelPath); err == nil {
		m = openModel(true)
	} else {
		opts := radixnet.DefaultGraphOptions()
		if *window > 0 {
			opts.Window = *window
		}
		if *globalScale >= 0 {
			opts.GlobalScale = *globalScale
		}
		if *windowScale >= 0 {
			opts.WindowScale = *windowScale
		}
		if *rewardScale >= 0 {
			opts.RewardScale = *rewardScale
		}
		if *countScale >= 0 {
			opts.CountScale = *countScale
		}
		var err error
		m, err = radixnet.NewModel(seedFlag, opts)
		if err != nil {
			fail("%v", err)
		}
		m.Workers = workers
	}
	say("training on %d texts (%s) with %d goroutines, %d epoch(s)", len(texts), *unit, workers, *epochs)
	say("%5s %9s %10s %7s %7s %8s %6s %6s %11s %8s", "epoch", "loss", "ppl", "nodes", "edges", "trigrams", "ratio", "merges", "transitions", "seconds")
	opts := radixnet.DefaultTrainOptions()
	opts.Epochs = *epochs
	opts.AutoCompress = !*noCompress
	opts.Progress = func(r map[string]any) {
		say("%5v %9.4f %10.3f %7v %7v %8v %6.2f %6v %11v %8.3f", r["epoch"], r["loss"], r["perplexity"], r["nodes"], r["edges"], r["trigrams"], r["compression_ratio"], r["merges"], r["transitions"], r["seconds"])
	}
	t0 := time.Now()
	records, err := m.Train(texts, opts)
	if err != nil {
		fail("%v", err)
	}
	path := saveModel(m)
	say("saved %s (%d texts, %.2fs)", path, len(texts), time.Since(t0).Seconds())
	if jsonMode {
		emit(map[string]any{"records": records, "texts": len(texts), "split": *unit, "workers": workers, "saved": path, "stats": m.Stats()})
	}
}

func predictDoc(prefix string, p *radixnet.Prediction) map[string]any {
	return map[string]any{
		"prefix": prefix, "kind": "count", "continuation": p.Text, "full_text": p.FullText, "cost": p.Cost,
		"probability": p.Probability(), "step_costs": p.StepCosts, "path": p.Labels, "node_ids": p.NodeIDs,
		"expanded": p.Expanded, "reached_end": p.ReachedEnd, "mode": p.Mode, "k": p.K, "beam": p.Beam,
		"top": pathDicts(p.Top), "bottom": pathDicts(p.Bottom),
	}
}

func pathDicts(paths []*radixnet.PathResult) []map[string]any {
	out := make([]map[string]any, 0, len(paths))
	for _, r := range paths {
		out = append(out, map[string]any{"continuation": r.Text, "full_text": r.FullText, "cost": r.Cost, "probability": r.Probability(),
			"step_costs": r.StepCosts, "path": r.Labels, "node_ids": r.NodeIDs, "reached_end": r.ReachedEnd})
	}
	return out
}

func quote(s string) string { return fmt.Sprintf("%q", s) }

func cmdPredict(args []string) {
	fs := subFlagSet("predict")
	prefix := fs.String("prefix", "", "text to continue")
	length := fs.Int("length", 20, "minimum characters to emit")
	maxLength := fs.Int("max-length", -1, "cap on the continuation (-1 = none)")
	mode := fs.String("mode", "beam", "beam | sample (dijkstra = beam)")
	k := fs.Int("k", 5, "top K and bottom K continuations")
	beam := fs.Int("beam", 0, "beam width (0 = max(4k, 16))")
	toEnd := fs.Bool("to-end", false, "run to the end of a text")
	stepPenalty := fs.Float64("step-penalty", 0, "extra cost per edge")
	temperature := fs.Float64("temperature", 1.0, "sample: softmax temperature (0 = greedy)")
	_ = fs.Parse(args)
	m := openModel(true)
	opts := radixnet.PredictOptions{Length: *length, Mode: *mode, K: *k, Beam: *beam, StepPenalty: *stepPenalty, Temperature: *temperature, ToEnd: *toEnd, MaxLength: *maxLength}
	p, err := m.Predict(*prefix, opts)
	if err != nil {
		fail("%v", err)
	}
	if jsonMode {
		emit(predictDoc(*prefix, p))
		return
	}
	fmt.Printf("prefix       %s\ncontinuation %s\nfull text    %s\ncost %.4f  p %.4g  path %s\n", quote(*prefix), quote(p.Text), quote(p.FullText), p.Cost, p.Probability(), strings.Join(p.Labels, " -> "))
	if len(p.Top) > 0 {
		fmt.Println("top:")
		for i, r := range p.Top {
			fmt.Printf("  %2d %8.4f %10.4g %s\n", i+1, r.Cost, r.Probability(), quote(r.FullText))
		}
	}
	if len(p.Bottom) > 0 {
		fmt.Println("bottom:")
		for i, r := range p.Bottom {
			fmt.Printf("  %2d %8.4f %10.4g %s\n", i+1, r.Cost, r.Probability(), quote(r.FullText))
		}
	}
}

func cmdGenerate(args []string) {
	fs := subFlagSet("generate")
	count := fs.Int("count", 1, "texts to generate (beam: the K most likely)")
	maxLength := fs.Int("max-length", 60, "maximum characters per text")
	mode := fs.String("mode", "sample", "beam | sample | dijkstra")
	prefix := fs.String("prefix", "", "start every text with this")
	temperature := fs.Float64("temperature", 1.0, "sample: softmax temperature")
	stepPenalty := fs.Float64("step-penalty", 0, "beam / dijkstra: extra cost per edge")
	beam := fs.Int("beam", 0, "beam width (0 = default)")
	seeded := fs.Bool("seeded", false, "sample with a private RNG seeded by --seed (reproducible)")
	_ = fs.Parse(args)
	m := openModel(true)
	opts := radixnet.GenerateOptions{MaxLength: *maxLength, Mode: *mode, Temperature: *temperature, Count: *count, Prefix: *prefix, StepPenalty: *stepPenalty, Beam: *beam}
	if *seeded {
		s := seedFlag
		opts.Seed = &s
	}
	results, err := m.Generate(opts)
	if err != nil {
		fail("%v", err)
	}
	if jsonMode {
		samples := make([]map[string]any, 0, len(results))
		for _, r := range results {
			samples = append(samples, map[string]any{"text": r.Text, "full_text": r.FullText, "cost": r.Cost, "probability": r.Probability(),
				"labels": r.Labels, "node_ids": r.NodeIDs, "step_costs": r.StepCosts, "expanded": r.Expanded, "reached_end": r.ReachedEnd})
		}
		emit(map[string]any{"samples": samples, "count": len(results), "mode": *mode, "prefix": *prefix, "max_length": *maxLength, "temperature": *temperature})
		return
	}
	fmt.Printf("%3s %9s %10s %4s  %s\n", "#", "cost", "prob", "end", "text")
	for i, r := range results {
		end := "no"
		if r.ReachedEnd {
			end = "yes"
		}
		fmt.Printf("%3d %9.4f %10.4g %4s  %s\n", i+1, r.Cost, r.Probability(), end, quote(r.Text))
	}
}

func cmdScore(args []string) {
	fs := subFlagSet("score")
	text := fs.String("text", "", "a single text")
	var data multiFlag
	fs.Var(&data, "data", "file with texts (repeatable)")
	unit := fs.String("split", "lines", "lines | paragraphs | pages | file")
	pageLines := fs.Int("page-lines", 50, "lines per page for --split pages")
	_ = fs.Parse(args)
	var texts []string
	if *text != "" {
		texts = []string{*text}
	} else if len(data) > 0 {
		texts = readTexts(data, *unit, *pageLines)
	} else {
		fail("score needs --text or --data")
	}
	m := openModel(true)
	scores := m.ScoreAll(texts)
	results := make([]map[string]any, len(scores))
	meanLog, meanPer := 0.0, 0.0
	for i, s := range scores {
		results[i] = map[string]any{"text": texts[i], "log_prob": s.LogProb, "per_char": s.PerChar, "chars": s.Chars, "transitions": s.Transitions, "unknown_transitions": s.UnknownTransitions}
		meanLog += s.LogProb
		meanPer += s.PerChar
	}
	meanLog /= float64(len(scores))
	meanPer /= float64(len(scores))
	if jsonMode {
		emit(map[string]any{"results": results, "count": len(results), "mean_log_prob": meanLog, "mean_per_char": meanPer})
		return
	}
	fmt.Printf("%10s %9s %5s %11s %7s  %s\n", "log_prob", "per_char", "chars", "transitions", "unknown", "text")
	for i, s := range scores {
		fmt.Printf("%10.4f %9.4f %5d %11d %7d  %s\n", s.LogProb, s.PerChar, s.Chars, s.Transitions, s.UnknownTransitions, quote(texts[i]))
	}
	if len(scores) > 1 {
		fmt.Printf("\n%d texts: mean log_prob %.4f, mean per_char %.4f\n", len(scores), meanLog, meanPer)
	}
}

func ratedTexts(fs *flag.FlagSet, args []string) (good, bad []string, epochsNeg, epochsPos *int, strength *float64) {
	var goodFiles, badFiles multiFlag
	fs.Var(&goodFiles, "good", "file of thumbs-up texts (repeatable)")
	fs.Var(&badFiles, "bad", "file of thumbs-down texts (repeatable)")
	goodText := fs.String("good-text", "", "one thumbs-up text")
	badText := fs.String("bad-text", "", "one thumbs-down text")
	unit := fs.String("split", "lines", "lines | paragraphs | pages | file")
	epochsNeg = fs.Int("neg-epochs", 2, "negative passes (penalties)")
	epochsPos = fs.Int("pos-epochs", 3, "positive passes (traversal + reward)")
	strength = fs.Float64("strength", 1.0, "reward / penalty per rated path")
	_ = fs.Parse(args)
	if len(goodFiles) > 0 {
		good = append(good, readTexts(goodFiles, *unit, 50)...)
	}
	if *goodText != "" {
		good = append(good, *goodText)
	}
	if len(badFiles) > 0 {
		bad = append(bad, readTexts(badFiles, *unit, 50)...)
	}
	if *badText != "" {
		bad = append(bad, *badText)
	}
	return good, bad, epochsNeg, epochsPos, strength
}

func cmdFeedback(args []string) {
	fs := subFlagSet("feedback")
	good, bad, negEpochs, posEpochs, strength := ratedTexts(fs, args)
	if len(good) == 0 && len(bad) == 0 {
		fail("feedback needs --good / --good-text and/or --bad / --bad-text")
	}
	m := openModel(true)
	action := "reward"
	var negative, positive []map[string]any
	var err error
	switch {
	case len(good) > 0 && len(bad) > 0:
		action = "2nrl"
		var res *radixnet.TwoNRLResult
		res, err = m.TwoNRL(bad, good, *negEpochs, *posEpochs, *strength)
		if err == nil {
			negative, positive = res.Negative, res.Positive
		}
	case len(good) > 0:
		positive, err = m.Reward(good, *posEpochs, *strength)
	default:
		action = "punish"
		negative, err = m.Punish(bad, *negEpochs, *strength)
	}
	if err != nil {
		fail("%v", err)
	}
	path := saveModel(m)
	say("%s: %d thumbs up, %d thumbs down; saved %s", action, len(good), len(bad), path)
	if jsonMode {
		emit(map[string]any{"action": action, "good": len(good), "bad": len(bad), "negative": negative, "positive": positive, "saved": path, "stats": m.Stats()})
	}
}

func cmdTwoNRL(args []string) {
	fs := subFlagSet("2nrl")
	good, bad, negEpochs, posEpochs, strength := ratedTexts(fs, args)
	if len(good) == 0 || len(bad) == 0 {
		fail("2nrl needs --bad FILE and --good FILE")
	}
	m := openModel(true)
	res, err := m.TwoNRL(bad, good, *negEpochs, *posEpochs, *strength)
	if err != nil {
		fail("%v", err)
	}
	path := saveModel(m)
	say("2NRL: %d bad texts penalised (%d passes), %d good texts counted + rewarded (%d passes); saved %s", len(bad), *negEpochs, len(good), *posEpochs, path)
	if jsonMode {
		emit(map[string]any{"negative": res.Negative, "positive": res.Positive, "inverted": res.Inverted, "saved": path, "stats": m.Stats()})
	}
}

func cmdInvert(args []string) {
	fs := subFlagSet("invert")
	_ = fs.Parse(args)
	m := openModel(true)
	m.Invert()
	path := saveModel(m)
	say("inverted (every reward flipped); saved %s", path)
	if jsonMode {
		emit(map[string]any{"saved": path, "stats": m.Stats()})
	}
}

func cmdWeights(args []string) {
	fs := subFlagSet("weights")
	globalScale := fs.Float64("global-scale", -1, "weight of log(all-time share)")
	windowScale := fs.Float64("window-scale", -1, "weight of log(recent share)")
	rewardScale := fs.Float64("reward-scale", -1, "weight of the reward")
	countScale := fs.Float64("count-scale", -1, "weight of log(1 + traversals)")
	window := fs.Int("window", 0, "sliding window size")
	_ = fs.Parse(args)
	m := openModel(true)
	changes := map[string]float64{}
	if *globalScale >= 0 {
		changes["global_scale"] = *globalScale
	}
	if *windowScale >= 0 {
		changes["window_scale"] = *windowScale
	}
	if *rewardScale >= 0 {
		changes["reward_scale"] = *rewardScale
	}
	if *countScale >= 0 {
		changes["count_scale"] = *countScale
	}
	if *window > 0 {
		changes["window"] = float64(*window)
	}
	saved := ""
	if len(changes) > 0 {
		if err := m.ConfigureWeights(changes); err != nil {
			fail("%v", err)
		}
		saved = saveModel(m)
	}
	cfg := m.G.WeightConfig()
	if jsonMode {
		emit(map[string]any{"weights": cfg, "changed": len(changes) > 0, "saved": saved, "stats": m.Stats()})
		return
	}
	fmt.Printf("function      %s\ncount_scale   %g\nglobal_scale  %g\nwindow_scale  %g\nreward_scale  %g\nwindow        %d\nsmoothing     %g\ntraversals    %d (window %d)\n",
		cfg.Function, cfg.CountScale, cfg.GlobalScale, cfg.WindowScale, cfg.RewardScale, cfg.Window, cfg.Smoothing, m.G.TotalTraversals, m.G.WindowTraversals())
	if saved != "" {
		fmt.Printf("saved %s\n", saved)
	}
}

func cmdInfo(args []string) {
	fs := subFlagSet("info")
	tail := fs.Int("tail", 5, "history records to show")
	_ = fs.Parse(args)
	m := openModel(true)
	stats := m.Stats()
	hist := m.History
	if *tail < len(hist) {
		hist = hist[len(hist)-*tail:]
	}
	if jsonMode {
		emit(map[string]any{"model": modelPath, "stats": stats, "history": hist, "meta": m.Meta})
		return
	}
	fmt.Printf("model %s\n", modelPath)
	for _, k := range radixnet.SortedKeys(stats) {
		fmt.Printf("%-22s %v\n", k, stats[k])
	}
	if len(hist) > 0 {
		fmt.Println("history:")
		for _, r := range hist {
			fmt.Printf("  epoch %v loss %.4f nodes %v edges %v\n", r["epoch"], toF(r["loss"]), r["nodes"], r["edges"])
		}
	}
}

func toF(v any) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return 0
}

func cmdConverse(args []string) {
	fs := subFlagSet("converse")
	opening := fs.String("opening", "", "the first line, spoken as given")
	turns := fs.Int("turns", 6, "turns to generate")
	mode := fs.String("mode", "beam", "beam | sample")
	maxLength := fs.Int("max-length", 60, "characters a reply may add to its context")
	context := fs.Int("context", 12, "characters of the previous line a reply picks up")
	k := fs.Int("k", 5, "candidates per turn")
	beam := fs.Int("beam", 0, "beam width (0 = default)")
	temperature := fs.Float64("temperature", 1.0, "sample: softmax temperature")
	stepPenalty := fs.Float64("step-penalty", 0, "extra cost per edge")
	speakers := fs.String("speakers", "A,B", "names of the voices")
	partner := fs.String("partner", "", "a second model file that speaks the second voice")
	allowRepeats := fs.Bool("allow-repeats", false, "do not skip continuations already heard")
	seeded := fs.Bool("seeded", false, "sample with a private RNG seeded by --seed")
	_ = fs.Parse(args)
	m := openModel(true)
	opts := radixnet.DefaultConverseOptions()
	opts.Turns, opts.Mode, opts.MaxLength, opts.Context, opts.K, opts.Beam = *turns, *mode, *maxLength, *context, *k, *beam
	opts.Temperature, opts.StepPenalty, opts.AvoidRepeats = *temperature, *stepPenalty, !*allowRepeats
	names := []string{}
	for _, s := range strings.Split(*speakers, ",") {
		if t := strings.TrimSpace(s); t != "" {
			names = append(names, t)
		}
	}
	if len(names) > 0 {
		opts.Speakers = names
	}
	if *seeded {
		s := seedFlag
		opts.Seed = &s
	}
	partnerKind := ""
	if *partner != "" {
		p, err := radixnet.Load(*partner)
		if err != nil {
			fail("partner model: %v", err)
		}
		p.Workers = workers
		opts.Partner = p
		partnerKind = "count"
	}
	turnsOut, err := m.Converse(*opening, opts)
	if err != nil {
		fail("%v", err)
	}
	if jsonMode {
		emit(map[string]any{"turns": turnsOut, "count": len(turnsOut), "speakers": opts.Speakers, "mode": *mode, "opening": *opening,
			"kind": "count", "partner_kind": partnerKind, "transcript": radixnet.Transcript(turnsOut)})
		return
	}
	for _, t := range turnsOut {
		fmt.Printf("%s: %s\n", t.Speaker, t.Text)
		detail := fmt.Sprintf("    cost %.4f  p %.4g", t.Cost, t.Probability)
		if t.Context != "" {
			detail += "  picked up " + quote(t.Context)
		}
		var flags []string
		if t.Given {
			flags = append(flags, "given")
		}
		if t.Fresh && !t.Given {
			flags = append(flags, "new topic")
		}
		if t.Repeat {
			flags = append(flags, "repeat")
		}
		if len(flags) > 0 {
			detail += "  [" + strings.Join(flags, ", ") + "]"
		}
		fmt.Println(detail)
	}
	if len(turnsOut) == 0 {
		fmt.Println("(nothing to say: train the model first)")
	}
}

var _ = filepath.Base

func cmdServe(args []string) {
	fs := subFlagSet("serve")
	host := fs.String("host", "127.0.0.1", "interface to bind")
	port := fs.Int("port", 8001, "TCP port")
	frontendDir := fs.String("frontend-dir", "frontend/dist", "built frontend to serve at / (empty = API only)")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory (enables /api/checkpoints)")
	uploadDir := fs.String("upload-dir", "uploads", "directory of uploaded training files (empty = uploads disabled)")
	keep := fs.Int("keep", 5, "checkpoints to keep")
	quiet := fs.Bool("quiet", false, "do not log requests")
	_ = fs.Parse(args)
	logf := func(line string) { fmt.Fprintln(os.Stderr, line) }
	svc, err := server.NewService(server.Options{
		ModelPath: modelPath, Seed: seedFlag, Workers: workers, UploadDir: *uploadDir, CheckpointDir: *checkpointDir,
		Keep: *keep, Quiet: *quiet, Log: logf,
	})
	if err != nil {
		fail("%v", err)
	}
	dir := *frontendDir
	if dir != "" {
		if st, statErr := os.Stat(filepath.Join(dir, "index.html")); statErr != nil || st.IsDir() {
			logf(fmt.Sprintf("note: no frontend build at %s (serving the API and a help page only)", dir))
			dir = ""
		} else if abs, absErr := filepath.Abs(dir); absErr == nil {
			dir = abs
		}
	}
	addr := fmt.Sprintf("%s:%d", *host, *port)
	origin, _ := os.Stat(modelPath)
	source := "a fresh model"
	if origin != nil {
		source = modelPath
	}
	logf(fmt.Sprintf("radixnet-count %s serving %s on http://%s (workers %d, frontend %s)", version, source, addr, workers, map[bool]string{true: dir, false: "none"}[dir != ""]))
	httpServer := &http.Server{Addr: addr, Handler: server.NewHandler(svc, dir, *quiet, logf), ReadHeaderTimeout: 30 * time.Second}
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fail("%v", err)
	}
}
