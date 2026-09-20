package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/server"
)

// The commands the Go CLI was missing beside the model itself: folding the
// graph up by hand, the checkpoint directory, and the benchmark.

// cmdCompress merges the unary chains of the graph (radix-tree path
// compression) and saves.  Training compresses as it goes; this is the manual
// pass, for a graph built another way or loaded from elsewhere.
func cmdCompress(args []string) {
	fs := flag.NewFlagSet("compress", flag.ExitOnError)
	out := fs.String("out", "", "where to save the model (default: --model)")
	_ = fs.Parse(args)

	m := openModel(true)
	before := fmt.Sprintf("%d nodes / %d edges", m.G.NumNodes(), m.G.NumEdges())
	merged := m.G.Compress()
	target := *out
	if target == "" {
		target = modelFile()
	}
	if err := m.Save(target); err != nil {
		fail("cannot save %s: %v", target, err)
	}
	say("before      %s", before)
	say("merged      %d chain(s)", merged)
	say("after       %d nodes / %d edges", m.G.NumNodes(), m.G.NumEdges())
	say("compression %.4f", m.G.CompressionRatio())
	say("saved       %s", target)
	if jsonMode {
		emit(map[string]any{
			"merged": merged, "nodes": m.G.NumNodes(), "edges": m.G.NumEdges(),
			"trigrams": m.G.NumTrigrams(), "compression_ratio": m.G.CompressionRatio(), "saved": target,
		})
	}
}

// cmdCheckpoints lists a checkpoint directory, or restores one into a model
// file.  The layout is the Python CheckpointManager's, so both sides can share
// a directory.
func cmdCheckpoints(args []string) {
	fs := flag.NewFlagSet("checkpoints", flag.ExitOnError)
	dir := fs.String("dir", "checkpoints", "checkpoint directory")
	restore := fs.String("restore", "", "checkpoint to restore: a name, its bare stem, a path, or `latest`")
	out := fs.String("out", "", "where to write the restored model (default: --model)")
	keep := fs.Int("keep", 5, "checkpoints the directory keeps (only used when saving)")
	_ = fs.Parse(args)

	manager := server.NewCheckpoints(*dir, *keep)
	if strings.TrimSpace(*restore) == "" {
		entries := manager.List()
		latest := ""
		if record := manager.Latest(); record != nil {
			latest, _ = record["name"].(string)
		}
		say("directory  %s", *dir)
		say("latest     %s", orDash(latest))
		say("")
		if len(entries) == 0 {
			say("no checkpoints yet")
		} else {
			say("%-34s %10s %8s  %s", "name", "step", "tag", "saved at")
			for _, entry := range entries {
				say("%-34v %10v %8v  %v", entry["name"], entry["step"], entry["tag"], entry["saved_at"])
			}
		}
		if jsonMode {
			emit(map[string]any{"dir": *dir, "latest": latest, "checkpoints": entries})
		}
		return
	}
	name := *restore
	if strings.EqualFold(strings.TrimSpace(name), "latest") {
		record := manager.Latest()
		if record == nil {
			fail("no latest checkpoint in %s", *dir)
		}
		name, _ = record["name"].(string)
	}
	path, err := manager.Resolve(name)
	if err != nil {
		fail("%v", err)
	}
	m, err := radixnet.Load(path)
	if err != nil {
		fail("%s: %v", path, err)
	}
	target := *out
	if target == "" {
		target = modelFile()
	}
	if err := m.Save(target); err != nil {
		fail("cannot save %s: %v", target, err)
	}
	say("restored %s -> %s (%s, %d nodes / %d edges)", filepath.Base(path), target, m.Kind(),
		m.G.NumNodes(), m.G.NumEdges())
	if jsonMode {
		emit(map[string]any{"restored": filepath.Base(path), "path": path, "out": target, "stats": m.Stats()})
	}
}

func orDash(text string) string {
	if strings.TrimSpace(text) == "" {
		return "-"
	}
	return text
}

// cmdBench measures how fast this build counts and predicts.
func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	chars := fs.Int("chars", radixnet.BenchDefaultChars, "characters of synthetic training text")
	epochs := fs.Int("epochs", radixnet.BenchDefaultEpochs, "training epochs to time")
	predictions := fs.Int("predictions", 0, "predictions to time (0: chars/2, clamped to [100, 20000])")
	corpus := fs.String("data", filepath.Join("data", "sample_corpus.txt"), "sample corpus the sentences are built from")
	traversal := fs.String("traversal", "reward", "the search to time: reward | least-punished")
	textsPath := fs.String("texts", "", "train on this file's lines instead of a synthetic corpus")
	prefixPath := fs.String("prefixes", "", "predict these lines instead of prefixes cut out of the corpus")
	punishEvery := fs.Int("punish-every", 0, "punish every Nth text before predicting, so there is blame to walk by")
	_ = fs.Parse(args)

	texts, err := benchFileLines(*textsPath)
	if err != nil {
		fail("%v", err)
	}
	prefixes, err := benchFileLines(*prefixPath)
	if err != nil {
		fail("%v", err)
	}
	if len(texts) > 0 {
		say("benchmark: %d text(s) from %s, %d epoch(s), %s traversal, %s counting, %s",
			len(texts), *textsPath, *epochs, *traversal, map[bool]string{true: "exact", false: "racy"}[exact], workerText())
	} else {
		say("benchmark: %d chars, %d epoch(s), %s traversal, %s counting, %s",
			*chars, *epochs, *traversal, map[bool]string{true: "exact", false: "racy"}[exact], workerText())
	}
	result, err := radixnet.RunBenchmark(radixnet.BenchOptions{
		Chars: *chars, Epochs: *epochs, Seed: seedFlag, Predictions: *predictions,
		CorpusPath: *corpus, Workers: workers, Exact: exact,
		Texts: texts, Prefixes: prefixes, Traversal: *traversal, PunishEvery: *punishEvery,
	})
	if err != nil {
		fail("%v", err)
	}
	say("")
	keys := make([]string, 0, len(result))
	for key := range result {
		if key != "sample_prediction" {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		say("%-28s %v", key, fmtValue(result[key]))
	}
	if sample, ok := result["sample_prediction"].(map[string]any); ok {
		say("")
		say("%-28s %q -> %q", "sample_prediction", sample["prefix"], sample["continuation"])
	}
	if jsonMode {
		emit(result)
	}
}

// benchFileLines reads one text per line (blank lines dropped); "" reads nothing.
func benchFileLines(path string) ([]string, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", path, err)
	}
	lines := []string{}
	for _, line := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("%s holds no usable line", path)
	}
	return lines, nil
}

// fmtValue renders a benchmark number readably: rates to whole numbers, ratios
// and seconds to four places.
func fmtValue(value any) string {
	number, ok := value.(float64)
	if !ok {
		return fmt.Sprintf("%v", value)
	}
	if number >= 1000 {
		return fmt.Sprintf("%.0f", number)
	}
	return fmt.Sprintf("%.4f", number)
}

func workerText() string {
	if workers == 0 {
		return "one goroutine per text"
	}
	return fmt.Sprintf("%d worker(s)", workers)
}
