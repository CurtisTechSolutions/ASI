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
	"runtime/pprof"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/server"
)

const version = "0.1.0"

var (
	modelPath  = "model.count.json"
	jsonMode   bool
	seedFlag   int64
	workers    = 0
	exact      bool
	outPath    string
	memProfile string
	memLimit   = ""
	memory     radixnet.MemoryLimit
)

func addGlobalFlags(fs *flag.FlagSet) {
	fs.StringVar(&modelPath, "model", modelPath, "model file to load / save (gzip when the name ends with .gz)")
	fs.BoolVar(&jsonMode, "json", jsonMode, "print one JSON document instead of human-readable text")
	fs.Int64Var(&seedFlag, "seed", seedFlag, "RNG seed for a new model and for sampling")
	fs.IntVar(&workers, "workers", workers, "cap on the goroutines fanned out over texts and nodes (0 = none: one goroutine per text)")
	fs.BoolVar(&exact, "exact", exact, "count with atomic increments (no lost updates, reproducible); default: plain racy increments")
	fs.StringVar(&memProfile, "memprofile", memProfile, "write a heap profile to this file when the command finishes")
	fs.StringVar(&memLimit, "memlimit", memLimit, "soft memory limit, e.g. 2GiB (default: 80% of the container / machine memory; \"off\" to let the heap grow freely)")
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

// sourcesFor picks a streaming source per path: a ZIP archive (by its magic
// bytes) streams entry by entry, a text file line by line; nothing is loaded whole.
func sourcesFor(paths []string, unit string, pageLines int) radixnet.TextSource {
	var sources radixnet.MultiSource
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			fail("%v", err)
		}
		sources = append(sources, radixnet.SourceForFile(p, unit, pageLines))
	}
	return sources
}

// readTexts collects the texts of files (for the commands that need a list: score, feedback, 2nrl).
func readTexts(paths []string, unit string, pageLines int) []string {
	texts, err := radixnet.CollectTexts(sourcesFor(paths, unit, pageLines))
	if err != nil {
		fail("%v", err)
	}
	if len(texts) == 0 {
		fail("no texts found in %s", strings.Join(paths, ", "))
	}
	return texts
}

func configure(m *radixnet.Model) *radixnet.Model {
	m.Workers = workers
	m.G.Workers = workers
	m.Exact = exact
	return m
}

func openModel(required bool) *radixnet.Model {
	radixnet.Workers = workers
	if _, err := os.Stat(modelPath); err == nil {
		m, err := radixnet.Load(modelPath)
		if err != nil {
			fail("%s: %v", modelPath, err)
		}
		return configure(m)
	}
	if required {
		fail("model file not found: %s (train one first with `radixnet-count train --data FILE`)", modelPath)
	}
	m, err := radixnet.NewModel(seedFlag, radixnet.DefaultGraphOptions())
	if err != nil {
		fail("%v", err)
	}
	return configure(m)
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
  train      count traversals of the texts of --data files (text or ZIP, any size; lines, paragraphs or pages)
  predict    top-K / bottom-K continuations of --prefix (beam) or a stochastic walk
  generate   whole texts from the prediction search (beam / sample / dijkstra)
  score      log-probability of --text or every text of --data
  feedback   thumbs up (--good / --good-text) and thumbs down (--bad / --bad-text)
  2nrl       penalise --bad texts, then count + reward --good texts
  correct    teach one correction: only the trigram nodes --wrong and --right disagree on move
  invert     flip the sign of every reward
  weights    show or change the dual frequency weight function
  info       statistics and the training history tail
  converse   the model talks to itself
  tutor      English lessons: Ollama writes the prefix, the model completes it, Ollama marks it
  serve      HTTP API (+ the prebuilt frontend) speaking the Python server's JSON contract
  version    print the version

global options (before or after the command): --model PATH --json --seed N --workers N --exact --out PATH
                                             --memlimit SIZE (soft heap limit, default 80%% of the machine / container) --memprofile PATH
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
	boolFlags := map[string]bool{"json": true, "exact": true}
	for cmdIndex < len(args) && strings.HasPrefix(args[cmdIndex], "-") {
		// consume "--flag value", "--flag=value" or a boolean "--flag"
		name := strings.TrimLeft(args[cmdIndex], "-")
		if strings.Contains(args[cmdIndex], "=") || boolFlags[name] {
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
	want, err := radixnet.ParseSize(memLimit)
	if err != nil {
		fail("%v", err)
	}
	// the collector otherwise lets the heap grow to twice the live graph, which on a
	// container with a hard limit is an OOM kill rather than a garbage collection
	memory = radixnet.ApplyMemoryLimit(want, radixnet.DefaultMemoryFraction)
	if memProfile != "" {
		defer writeHeapProfile()
	}
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
	case "correct":
		cmdCorrect(rest)
	case "invert":
		cmdInvert(rest)
	case "weights":
		cmdWeights(rest)
	case "info":
		cmdInfo(rest)
	case "converse":
		cmdConverse(rest)
	case "tutor":
		cmdTutor(rest)
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
	chunk := fs.Int("chunk", radixnet.DefaultChunkSize, "texts streamed from the files per chunk (memory: the chunks in flight + the graph, whatever the corpus size)")
	parallelParts := fs.Bool("parallel-parts", false, "stream every archive entry / file at once on its own goroutine")
	inflight := fs.Int("inflight", 0, "chunks in flight at once (read, processed or waiting for their turn); 0 = two per CPU. Memory = the graph + inflight chunks")
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
	source := sourcesFor(data, *unit, *pageLines)
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
		configure(m)
	}
	pool := "one goroutine per text"
	if workers > 0 {
		pool = fmt.Sprintf("%d goroutines", workers)
	}
	before := m.MetaInt("trained_texts")
	say("training from %s (%s, chunks of %d texts): %s, %s counting, %d epoch(s), %s", strings.Join(data, ", "), *unit, *chunk, pool, m.Counting(), *epochs, memory)
	say("%5s %9s %10s %7s %7s %8s %6s %6s %11s %6s %8s", "epoch", "loss", "ppl", "nodes", "edges", "trigrams", "ratio", "merges", "transitions", "chunks", "seconds")
	opts := radixnet.DefaultTrainOptions()
	opts.Epochs = *epochs
	opts.AutoCompress = !*noCompress
	opts.ChunkSize = *chunk
	opts.ParallelParts = *parallelParts
	opts.Inflight = *inflight
	opts.Progress = func(r map[string]any) {
		say("%5v %9.4f %10.3f %7v %7v %8v %6.2f %6v %11v %6v %8.3f", r["epoch"], r["loss"], r["perplexity"], r["nodes"], r["edges"], r["trigrams"], r["compression_ratio"], r["merges"], r["transitions"], r["chunks"], r["seconds"])
	}
	t0 := time.Now()
	records, err := m.TrainSource(source, opts)
	if err != nil {
		fail("%v", err)
	}
	texts := int(m.MetaInt("trained_texts") - before)
	if texts == 0 && len(records) == 0 {
		fail("no texts found in %s", strings.Join(data, ", "))
	}
	writeHeapProfile() // with the trained model still live (a no-op without --memprofile)
	path := saveModel(m)
	say("saved %s (%d texts, %.2fs)", path, texts, time.Since(t0).Seconds())
	if jsonMode {
		emit(map[string]any{"records": records, "texts": texts, "split": *unit, "chunk": *chunk, "workers": workers, "counting": m.Counting(), "saved": path, "stats": m.Stats()})
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

func cmdCorrect(args []string) {
	fs := subFlagSet("correct")
	wrong := fs.String("wrong", "", "what the network wrote")
	right := fs.String("right", "", "what it should have written")
	strength := fs.Float64("strength", 1.0, "magnitude of one unit of feedback")
	weight := fs.Float64("weight", 1.0, "how bad the attempt was: the penalty is strength x weight")
	reward := fs.Float64("reward", 1.0, "what the correction is worth")
	keep := fs.Float64("keep", 0.25, "what the unchanged part of the correction still earns")
	noCount := fs.Bool("no-count", false, "do not traverse the correction (it is counted by default)")
	dryRun := fs.Bool("dry-run", false, "show the alignment without touching the model")
	_ = fs.Parse(args)
	if strings.TrimSpace(*wrong) == "" && strings.TrimSpace(*right) == "" {
		fail("correct needs --wrong (what the network wrote) and --right (what it should say)")
	}
	changes := radixnet.DiffSummary(*wrong, *right, 0)
	if *dryRun {
		sayChanges(*wrong, *right, changes)
		if jsonMode {
			emit(map[string]any{"wrong": *wrong, "right": *right, "changes": changes, "dry_run": true})
		}
		return
	}
	m := openModel(true)
	moved, err := m.Correct(*wrong, *right, radixnet.CorrectOptions{
		Strength: *strength, Weight: *weight, Reward: *reward, Keep: *keep, NoCount: *noCount,
	})
	if err != nil {
		fail("%v", err)
	}
	sayChanges(*wrong, *right, changes)
	say("moved: %d step(s) penalised, %d taught, %d kept at %g", moved.Penalised, moved.Rewarded, moved.Kept, *keep)
	path := saveModel(m)
	say("saved %s", path)
	if jsonMode {
		emit(map[string]any{
			"wrong": *wrong, "right": *right, "changes": changes, "edits": moved.Edits,
			"penalised": moved.Penalised, "rewarded": moved.Rewarded, "kept": moved.Kept,
			"penalty": moved.Penalty, "reward": moved.Reward, "loss": moved.Loss,
			"wrong_chars": moved.WrongChars, "right_chars": moved.RightChars,
			"saved": path, "stats": m.Stats(),
		})
	}
}

// sayChanges prints what the teacher changed, span by span.
func sayChanges(wrong, right string, changes []radixnet.Edit) {
	say("wrong  %s", wrong)
	say("right  %s", right)
	if len(changes) == 0 {
		say("the two sentences are the same: nothing to teach")
		return
	}
	say("%-8s %-24s %s", "change", "the network wrote", "the teacher wrote")
	for _, change := range changes {
		say("%-8s %-24s %s", change.Op, dashIfEmpty(change.Wrong), dashIfEmpty(change.Right))
	}
}

func dashIfEmpty(text string) string {
	if text == "" {
		return "-"
	}
	return text
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
		configure(p)
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

// cmdTutor runs the automated English lessons: Ollama writes sentence
// openings, the model completes them with the prediction search, Ollama marks
// the grammar and the grades drive the rewards and penalties.
func cmdTutor(args []string) {
	fs := subFlagSet("tutor")
	cfg := radixnet.DefaultTutorConfig()
	topic := fs.String("topic", cfg.Topic, "what the sentences are about")
	rounds := fs.Int("rounds", cfg.Rounds, "lesson rounds")
	exercises := fs.Int("exercises", cfg.Exercises, "sentence openings per round")
	attempts := fs.Int("attempts", cfg.Attempts, "completions per exercise (the first in --mode, the rest sampled)")
	focus := fs.String("focus", "", "pin every exercise to one point of grammar, e.g. 'past tense'")
	level := fs.String("level", cfg.Level, "how hard the exercises are")
	words := fs.String("words", cfg.Words, "how many words a prefix has")
	brief := fs.String("brief", "", "what this batch is being taught to: the prompt the last report card led to (the plan's brief)")
	url := fs.String("url", "", "the teacher's base URL (default: $OLLAMA_HOST, or $OPENAI_BASE_URL for chatgpt)")
	tutorProvider := fs.String("tutor-provider", radixnet.DefaultProvider,
		"who teaches: ollama (a local model) or chatgpt (OpenAI; needs $OPENAI_API_KEY)")
	tutorModel := fs.String("tutor-model", "", "model that sets and marks the exercises (default: the provider's own)")
	graderProvider := fs.String("grader-provider", "", "mark with the other provider (default: the teacher's)")
	graderModel := fs.String("grader-model", "", "a different model for the marking")
	graderURL := fs.String("grader-url", "", "base URL of the marker's provider (default: the same as --url)")
	timeout := fs.Float64("timeout", 0, "seconds to wait for one Ollama answer (default: 120)")
	mode := fs.String("mode", cfg.Mode, "how the model completes a prefix: beam | dijkstra | sample")
	length := fs.Int("length", cfg.Length, "characters the completion should reach")
	maxLength := fs.Int("max-length", cfg.MaxLength, "cap on the completion")
	temperature := fs.Float64("temperature", cfg.Temperature, "sampling temperature")
	noToEnd := fs.Bool("no-to-end", false, "stop at --length instead of finishing the sentence")
	threshold := fs.Float64("threshold", cfg.Threshold, "mark out of 10 a sentence must reach to pass")
	grammarWeight := fs.Float64("grammar-weight", cfg.GrammarWeight, "share of the mark that is grammar")
	batch := fs.Int("batch", cfg.Batch, "sentences marked in one Ollama call")
	noAdapt := fs.Bool("no-adapt", false, "do not drill the previous round's weakest points")
	drills := fs.Int("drills", cfg.Drills, "extra correct example sentences per round")
	plan := fs.Int("plan", cfg.Plan, "hand the report card at the end back to the teacher and print the next N lessons it plans (0 = off)")
	noTeachAnswer := fs.Bool("no-teach-answer", false, "a failed lesson learns only the correction")
	dryRun := fs.Bool("dry-run", false, "set and mark the exercises but train nothing and save nothing")
	noDiff := fs.Bool("no-diff-corrections", false, "learn a correction as two whole sentences instead of from its diff")
	keepWeight := fs.Float64("keep-weight", cfg.KeepWeight, "what the unchanged part of a correction still earns (0 = the fix alone)")
	twonrlPer := fs.String("twonrl-per", cfg.TwoNRLPer, "learn once per round, or after every lesson")
	minWeight := fs.Float64("min-weight", cfg.MinWeight, "penalty weight of a near miss (a hopeless answer weighs 1)")
	negEpochs := fs.Int("neg-epochs", cfg.NegEpochs, "negative passes (penalties)")
	posEpochs := fs.Int("pos-epochs", cfg.PosEpochs, "positive passes (traversal + reward)")
	strength := fs.Float64("strength", cfg.Strength, "reward / penalty per path, scaled by the mark")
	noReplay := fs.Bool("no-replay", false, "do not keep teaching earlier corrections")
	_ = fs.Parse(args)

	cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.Attempts = *topic, *rounds, *exercises, *attempts
	cfg.Focus, cfg.Level, cfg.Words, cfg.Brief = *focus, *level, *words, *brief
	cfg.TutorProvider, cfg.GraderProvider = *tutorProvider, *graderProvider
	cfg.TutorModel, cfg.GraderModel = *tutorModel, *graderModel
	cfg.Mode, cfg.Length, cfg.MaxLength, cfg.Temperature = *mode, *length, *maxLength, *temperature
	cfg.ToEnd = !*noToEnd
	cfg.Threshold, cfg.GrammarWeight, cfg.Batch = *threshold, *grammarWeight, *batch
	cfg.Adapt, cfg.Drills, cfg.TeachAnswer, cfg.Learn = !*noAdapt, *drills, !*noTeachAnswer, !*dryRun
	cfg.Plan = *plan
	cfg.TwoNRLPer, cfg.MinWeight = *twonrlPer, *minWeight
	cfg.DiffCorrections, cfg.KeepWeight = !*noDiff, *keepWeight
	cfg.NegEpochs, cfg.PosEpochs, cfg.Strength, cfg.Replay = *negEpochs, *posEpochs, *strength, !*noReplay
	if err := cfg.Validate(); err != nil { // also resolves the providers and the models they imply
		fail("%v", err)
	}
	if (cfg.TutorProvider == radixnet.ProviderChatGPT || cfg.GraderProvider == radixnet.ProviderChatGPT) &&
		!radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT teach, " +
			"or use -tutor-provider ollama")
	}
	timeoutDuration := time.Duration(*timeout * float64(time.Second))
	client, err := radixnet.NewLLMClient(cfg.TutorProvider, *url, cfg.TutorModel, timeoutDuration)
	if err != nil {
		fail("%v", err)
	}
	grader := client
	if cfg.GraderProvider != cfg.TutorProvider || strings.TrimSpace(*graderURL) != "" {
		if grader, err = radixnet.NewLLMClient(cfg.GraderProvider, *graderURL, cfg.ResolvedGraderModel(), timeoutDuration); err != nil {
			fail("%v", err)
		}
	}
	m := openModel(true)
	trainer, err := radixnet.NewTutorTrainer(m, client, cfg)
	if err != nil {
		fail("%v", err)
	}
	trainer.GraderClient = grader
	say("tutor: %s, %d round(s) x %d exercise(s), teacher %s: %s at %s, marked by %s: %s, pass at %g/10 (grammar %g)",
		cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.TutorProvider, cfg.TutorModel, client.BaseURL(),
		cfg.GraderProvider, cfg.ResolvedGraderModel(), cfg.Threshold, cfg.GrammarWeight)
	if cfg.DiffCorrections {
		say("corrections: from the diff with what the network wrote, only what changed moves (the rest keeps %g)", cfg.KeepWeight)
	} else {
		say("corrections: as whole sentences (--no-diff-corrections)")
	}
	if !jsonMode {
		trainer.Progress = func(record map[string]any) { sayLesson(record) }
	}
	records, err := trainer.Run()
	if err != nil {
		fail("%v", err)
	}
	card := radixnet.ReportCard(trainer.Lessons)
	saved := ""
	if !*dryRun {
		saved = saveModel(m)
	}
	say("report card: %v/%v passed, mean %s (grammar %s); mistakes: %s",
		card["passed"], card["lessons"], fmtMark(card["mean_score"]), fmtMark(card["mean_grammar"]), mistakes(card))
	planned := lastRecord(records, "plan")
	if planned != nil {
		sayPlan(planned)
	}
	if saved != "" {
		say("saved %s", saved)
	}
	if jsonMode {
		emit(map[string]any{
			"config": cfg, "records": records, "lessons": trainer.Lessons, "report": card,
			"plan": planned, "saved": saved, "stats": m.Stats(),
		})
	}
}

// lastRecord is the last record of one kind, or nil.
func lastRecord(records []map[string]any, kind string) map[string]any {
	for i := len(records) - 1; i >= 0; i-- {
		if name, _ := records[i]["kind"].(string); name == kind {
			return records[i]
		}
	}
	return nil
}

// sayPlan prints the lessons the teacher planned from the report card, the step
// up the marks earned and the brief that teaches the next batch.
func sayPlan(record map[string]any) {
	lessons, _ := record["lessons"].([]map[string]any)
	upgrade, _ := record["upgrade"].(map[string]any)
	say("lesson plan (%v): %v", record["source"], record["summary"])
	for i, lesson := range lessons {
		say("  %d. %v  [fixes %v, topic %v, %v exercise(s), %v drill(s)]  %v",
			i+1, lesson["focus"], lesson["targets"], lesson["topic"], lesson["exercises"], lesson["drills"],
			lesson["why"])
	}
	say("step up (%v): %v", upgrade["step"], upgrade["note"])
	say("brief: %v", record["prompt"])
	if len(lessons) > 0 {
		threshold, _ := upgrade["threshold"].(float64)
		prompt, _ := record["prompt"].(string)
		say("teach the next batch: radixnet-count tutor --topic %q --level %v --words %q --threshold %g "+
			"--exercises %v --drills %v --brief %q",
			lessons[0]["topic"], upgrade["level"], upgrade["words"], threshold, lessons[0]["exercises"],
			upgrade["drills"], prompt)
	}
}

// sayLesson prints one tutor record: a marked sentence, a round summary or the report card.
func sayLesson(record map[string]any) {
	switch record["kind"] {
	case "lesson":
		mark := "fail"
		if passed, _ := record["passed"].(bool); passed {
			mark = "pass"
		}
		say("  %v %s %s/10  %q", record["exercise"], mark, fmtMark(record["score"]), record["sentence"])
		if mark == "fail" {
			if correction, _ := record["correction"].(string); correction != "" {
				say("    correct: %q  (%v)", correction, record["error"])
			}
			if comment, _ := record["comment"].(string); comment != "" {
				say("    teacher: %s", comment)
			}
		}
	case "round":
		action := "nothing to learn"
		if learned, ok := record["action"].(string); ok && learned != "" {
			action = learned
		}
		weak := "nothing"
		if names, ok := record["weakest"].([]string); ok && len(names) > 0 {
			weak = strings.Join(names, ", ")
		}
		say("round %v: %v/%v passed, mean %s (grammar %s), weakest: %s -> %s (bad=%v, good=%v)",
			record["round"], record["passed"], record["lessons"], fmtMark(record["mean_score"]),
			fmtMark(record["mean_grammar"]), weak, action, record["bad"], record["good"])
	case "plan":
		lessons, _ := record["lessons"].([]map[string]any)
		targets, _ := record["targets"].([]string)
		drilling := "nothing in particular"
		if len(targets) > 0 {
			drilling = strings.Join(targets, ", ")
		}
		say("lesson plan (%v): %d lesson(s) at %v level, drilling %s",
			record["source"], len(lessons), record["level"], drilling)
	case "note":
		say("note: %v", record["message"])
	}
}

func fmtMark(value any) string {
	number, ok := value.(float64)
	if !ok {
		if pointer, isPointer := value.(*float64); isPointer && pointer != nil {
			number, ok = *pointer, true
		}
	}
	if !ok {
		return "-"
	}
	return fmt.Sprintf("%.2f", number)
}

func mistakes(card map[string]any) string {
	errors, _ := card["errors"].(map[string]int)
	if len(errors) == 0 {
		return "none"
	}
	parts := []string{}
	for _, name := range card["weakest"].([]string) {
		parts = append(parts, fmt.Sprintf("%s x%d", name, errors[name]))
	}
	return strings.Join(parts, ", ")
}

func cmdServe(args []string) {
	fs := subFlagSet("serve")
	host := fs.String("host", "127.0.0.1", "interface to bind")
	port := fs.Int("port", 8001, "TCP port")
	frontendDir := fs.String("frontend-dir", "frontend/dist", "built frontend to serve at / (empty = API only)")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory (enables /api/checkpoints)")
	uploadDir := fs.String("upload-dir", "uploads", "directory of uploaded training files (empty = uploads disabled)")
	keep := fs.Int("keep", 5, "checkpoints to keep")
	quiet := fs.Bool("quiet", false, "do not log requests")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL for /api/tutor (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	chatgptURL := fs.String("chatgpt-url", "", "OpenAI base URL for a ChatGPT teacher (default: $OPENAI_BASE_URL or https://api.openai.com/v1)")
	chatgptModel := fs.String("chatgpt-model", "", "default ChatGPT model (default: $RADIXNET_OPENAI_MODEL); the key is the server's own $OPENAI_API_KEY")
	ollamaModel := fs.String("ollama-model", "", "default teacher model for /api/tutor (default: $RADIXNET_TUTOR_MODEL or $RADIXNET_OLLAMA_MODEL)")
	_ = fs.Parse(args)
	logf := func(line string) { fmt.Fprintln(os.Stderr, line) }
	svc, err := server.NewService(server.Options{
		ModelPath: modelPath, Seed: seedFlag, Workers: workers, Exact: exact, UploadDir: *uploadDir, CheckpointDir: *checkpointDir,
		Keep: *keep, Quiet: *quiet, Log: logf, OllamaURL: *ollamaURL, OllamaModel: *ollamaModel,
		ChatGPTURL: *chatgptURL, ChatGPTModel: *chatgptModel,
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
	pool := "one goroutine per text"
	if workers > 0 {
		pool = fmt.Sprintf("%d goroutines", workers)
	}
	logf(fmt.Sprintf("radixnet-count %s serving %s on http://%s (%s, %s counting, %s, frontend %s)", version, source, addr, pool, map[bool]string{true: "exact", false: "racy"}[exact], memory, map[bool]string{true: dir, false: "none"}[dir != ""]))
	httpServer := &http.Server{Addr: addr, Handler: server.NewHandler(svc, dir, *quiet, logf), ReadHeaderTimeout: 30 * time.Second}
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fail("%v", err)
	}
}

// writeHeapProfile dumps the heap (after a GC) to --memprofile.
func writeHeapProfile() {
	if memProfile == "" {
		return
	}
	f, err := os.Create(memProfile)
	memProfile = "" // once

	if err != nil {
		fmt.Fprintf(os.Stderr, "memprofile: %v\n", err)
		return
	}
	defer f.Close()
	runtime.GC()
	if err := pprof.WriteHeapProfile(f); err != nil {
		fmt.Fprintf(os.Stderr, "memprofile: %v\n", err)
	}
}
