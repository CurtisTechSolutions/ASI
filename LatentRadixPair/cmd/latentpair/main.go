// Command latentpair trains the context tokenizer, primes the radix pair over its codes, reads and judges
// texts, and predicts.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/pair"
	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/tokenizer"
)

const usage = `latentpair: a trained context tokenizer under a primed radix pair.

  latentpair tokenizer train --out TOK [options] FILES...   train the tokenizer on files
  latentpair tokenizer info --tokenizer TOK                 describe a tokenizer
  latentpair tokenizer encode --tokenizer TOK --text T      the code of a context (--all: every position)
  latentpair tokenizer decode --tokenizer TOK --text T      what its code decodes to, one symbol at a time
  latentpair tokenizer classes --tokenizer TOK FILES...     what each first symbol stands for in files
  latentpair prime --tokenizer TOK --out MODEL [settings]   prime the two trees for a tokenizer
  latentpair train --model MODEL [--out MODEL] FILES...     read files into the count tree
  latentpair reward --model MODEL --text T [--prefix P]     credit an outcome (also read)
  latentpair punish --model MODEL --text T [--prefix P]     charge an outcome
  latentpair judge --model MODEL --good T --bad T           two-sided judgement
  latentpair predict --model MODEL --prefix P [--length N]  continue a prefix
  latentpair fold --model MODEL --prefix P                  the next-byte distribution and the code's levels
  latentpair score --model MODEL --text T                   bits per byte and the reward readings of a text
  latentpair info --model MODEL                             describe a model
  latentpair demo [--steps N] FILES...                      the whole loop on some files
  latentpair bench --tokenizer TOK FILES...                 held-out bits per byte against raw byte contexts

Text arguments take "-" to read standard input. Model and tokenizer paths ending in .gz are gzipped.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "tokenizer":
		err = tokenizerCmd(os.Args[2:])
	case "prime":
		err = primeCmd(os.Args[2:])
	case "train":
		err = trainCmd(os.Args[2:])
	case "reward", "punish":
		err = judgeOneCmd(os.Args[1], os.Args[2:])
	case "judge":
		err = judgeCmd(os.Args[2:])
	case "predict":
		err = predictCmd(os.Args[2:])
	case "fold":
		err = foldCmd(os.Args[2:])
	case "score":
		err = scoreCmd(os.Args[2:])
	case "info":
		err = infoCmd(os.Args[2:])
	case "demo":
		err = demoCmd(os.Args[2:])
	case "bench":
		err = benchCmd(os.Args[2:])
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// readText resolves a --text style argument ("-" is standard input).
func readText(v string) (string, error) {
	if v == "-" {
		b, err := io.ReadAll(bufio.NewReader(os.Stdin))
		return string(b), err
	}
	return v, nil
}

// readFiles reads every path as one text.
func readFiles(paths []string) ([][]byte, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no files given")
	}
	var out [][]byte
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

func printJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(v)
}

func settingsFlags(fs *flag.FlagSet, s *pair.Settings) {
	fs.Float64Var(&s.Alpha, "alpha", s.Alpha, "own = ctx / (ctx + alpha)")
	fs.Float64Var(&s.Floor, "floor", s.Floor, "share of the uniform in every fold")
	fs.Float64Var(&s.Smoothing, "smoothing", s.Smoothing, "pseudo-count on every outcome")
	fs.Float64Var(&s.ShareScale, "share-scale", s.ShareScale, "weight of log share")
	fs.Float64Var(&s.RewardScale, "reward-scale", s.RewardScale, "weight of net reward (reward traversal)")
	fs.Float64Var(&s.MeritScale, "merit-scale", s.MeritScale, "weight of merit (punishment traversal)")
	fs.Float64Var(&s.PenaltyScale, "penalty-scale", s.PenaltyScale, "weight of penalties (punishment traversal)")
	fs.Float64Var(&s.Strength, "strength", s.Strength, "default judgement strength")
	fs.IntVar(&s.Outcomes, "outcomes", s.Outcomes, "potential outcomes of the judged space; a verdict is worth strength/outcomes")
	fs.StringVar(&s.Rungs, "rungs", s.Rungs, "levels a judgement credits: all | final")
	fs.StringVar(&s.Backoff, "backoff", s.Backoff, "all | deepest | none")
	fs.IntVar(&s.CellCeiling, "cell-ceiling", s.CellCeiling, "largest address space allowed")
}

func tokenizerFlags(fs *flag.FlagSet, c *tokenizer.Config, levels *string) {
	fs.IntVar(&c.Window, "window", c.Window, "bytes of context the encoder reads")
	fs.IntVar(&c.Embed, "embed", c.Embed, "byte embedding width")
	fs.IntVar(&c.EncHidden, "enc-hidden", c.EncHidden, "encoder hidden width")
	fs.IntVar(&c.DecHidden, "dec-hidden", c.DecHidden, "decoder hidden width")
	fs.IntVar(&c.OutEmbed, "out-embed", c.OutEmbed, "per-position output width")
	fs.StringVar(levels, "levels", tokenizer.LevelsString(c.Levels), "quantisation levels, symbols by ':' and dimensions by ','")
	fs.Float64Var(&c.Noise, "noise", c.Noise, "highest byte-corruption rate of a training input")
	fs.Float64Var(&c.Recency, "recency", c.Recency, "reconstruction weight per byte of age")
	fs.Float64Var(&c.Predict, "predict", c.Predict, "weight of predicting the byte after the window")
	fs.Float64Var(&c.StartShare, "start-share", c.StartShare, "share of training windows from text starts")
}

func trainFlags(fs *flag.FlagSet, o *tokenizer.Options) {
	fs.IntVar(&o.Steps, "steps", o.Steps, "training steps")
	fs.IntVar(&o.Batch, "batch", o.Batch, "windows per step")
	fs.Float64Var(&o.LR, "lr", o.LR, "peak learning rate")
	fs.Int64Var(&o.Seed, "seed", o.Seed, "random seed")
	fs.IntVar(&o.EvalEvery, "eval-every", o.EvalEvery, "steps between log lines")
	fs.IntVar(&o.EvalWindows, "eval-windows", o.EvalWindows, "windows the log evaluates")
}

func logStat(s tokenizer.Stat) {
	fmt.Printf("step %6d  recon %.3f bits  next %s  train %.3f  exact %s  tail %s  lr %.1e  %.0fs\n",
		s.Step, s.Loss, fmtList(s.Next), s.Train, fmtList(s.Acc), fmtList(s.Tail), s.LR, s.Seconds)
}

func fmtList(v []float64) string {
	var parts []string
	for _, x := range v {
		parts = append(parts, fmt.Sprintf("%.2f", x))
	}
	return strings.Join(parts, "/")
}

func tokenizerCmd(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("tokenizer needs a subcommand: train, info, encode, decode, classes")
	}
	switch args[0] {
	case "train":
		fs := flag.NewFlagSet("tokenizer train", flag.ExitOnError)
		cfg := tokenizer.DefaultConfig()
		o := tokenizer.DefaultOptions()
		var levels, out string
		tokenizerFlags(fs, &cfg, &levels)
		trainFlags(fs, &o)
		fs.StringVar(&out, "out", "tokenizer.json.gz", "where to save the tokenizer")
		fs.Parse(args[1:])
		l, err := tokenizer.ParseLevels(levels)
		if err != nil {
			return err
		}
		cfg.Levels = l
		texts, err := readFiles(fs.Args())
		if err != nil {
			return err
		}
		tok, err := tokenizer.New(cfg, o.Seed)
		if err != nil {
			return err
		}
		total := 0
		for _, t := range texts {
			total += len(t)
		}
		fmt.Printf("tokenizer: window %d, code %s (%.1f bits), %d parameters; corpus %d files, %d bytes\n",
			cfg.Window, levels, cfg.Bits(), tok.ParamCount(), len(texts), total)
		o.Log = logStat
		if _, err := tok.Train(texts, o); err != nil {
			return err
		}
		if err := tok.Save(out); err != nil {
			return err
		}
		fmt.Printf("saved %s\n", out)
		return nil
	case "info":
		fs := flag.NewFlagSet("tokenizer info", flag.ExitOnError)
		path := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
		fs.Parse(args[1:])
		tok, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		printJSON(tok.Info())
		return nil
	case "encode":
		fs := flag.NewFlagSet("tokenizer encode", flag.ExitOnError)
		path := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
		text := fs.String("text", "", "the context")
		all := fs.Bool("all", false, "the code before every byte, not just the last context")
		fs.Parse(args[1:])
		tok, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		t, err := readText(*text)
		if err != nil {
			return err
		}
		if !*all {
			fmt.Println(fmtCode(tok.Encode([]byte(t))))
			return nil
		}
		codes := tok.EncodeAll([]byte(t))
		for i := 0; i < codes.Len(); i++ {
			next := "</s>"
			if i < len(t) {
				next = pair.Symbol(int(t[i]))
			}
			fmt.Printf("%4d  %-12s next %s\n", i, fmtCode(codes.At(i)), next)
		}
		return nil
	case "decode":
		fs := flag.NewFlagSet("tokenizer decode", flag.ExitOnError)
		path := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
		text := fs.String("text", "", "the context")
		fs.Parse(args[1:])
		tok, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		t, err := readText(*text)
		if err != nil {
			return err
		}
		return showDecode(tok, []byte(t))
	case "classes":
		fs := flag.NewFlagSet("tokenizer classes", flag.ExitOnError)
		path := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
		fs.Parse(args[1:])
		tok, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		texts, err := readFiles(fs.Args())
		if err != nil {
			return err
		}
		return showClasses(tok, texts)
	}
	return fmt.Errorf("unknown tokenizer subcommand %q", args[0])
}

func fmtCode(code []int32) string {
	var parts []string
	for _, c := range code {
		parts = append(parts, fmt.Sprint(c))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

// showDecode prints a context's code and what each prefix of the code decodes to.
func showDecode(tok *tokenizer.Tokenizer, ctx []byte) error {
	code := tok.Encode(ctx)
	shown := ctx
	if len(shown) > tok.Window {
		shown = shown[len(shown)-tok.Window:]
	}
	fmt.Printf("context %q -> code %s\n", string(shown), fmtCode(code))
	for k := 1; k <= tok.Depth(); k++ {
		d, err := tok.Decode(code, k)
		if err != nil {
			return err
		}
		fmt.Printf("  %d symbol(s) %-12s -> %q\n", k, fmtCode(code[:k]), d)
	}
	return nil
}

// showClasses tallies the first symbol over the positions of texts and prints each class's prototype.
func showClasses(tok *tokenizer.Tokenizer, texts [][]byte) error {
	radix := tok.Radix(0)
	counts := make([]int, radix)
	examples := make([][]byte, radix)
	total := 0
	for _, text := range texts {
		codes := tok.EncodeAll(text)
		for i := 0; i < codes.Len(); i++ {
			c := int(codes.At(i)[0])
			counts[c]++
			total++
			if examples[c] == nil || (i%97 == 0 && i >= tok.Window) {
				lo := i - tok.Window
				if lo < 0 {
					lo = 0
				}
				examples[c] = text[lo:i]
			}
		}
	}
	fmt.Printf("first symbol over %d positions (radix %d):\n", total, radix)
	for c := 0; c < radix; c++ {
		code := make([]int32, tok.Depth())
		code[0] = int32(c)
		proto, _ := tok.Decode(code, 1)
		fmt.Printf("  %3d  %6.2f%%  prototype %q  example %q\n", c, 100*float64(counts[c])/float64(max(total, 1)), proto, string(examples[c]))
	}
	return nil
}

func primeCmd(args []string) error {
	fs := flag.NewFlagSet("prime", flag.ExitOnError)
	s := pair.DefaultSettings()
	settingsFlags(fs, &s)
	tokPath := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
	out := fs.String("out", "model.json.gz", "where to save the model")
	seed := fs.Int64("seed", 1, "random seed for sampling")
	fs.Parse(args)
	tok, err := tokenizer.Load(*tokPath)
	if err != nil {
		return err
	}
	m, err := pair.Prime(tok, s, *seed)
	if err != nil {
		return err
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	fmt.Printf("primed %d context nodes x %d outcomes = %d cells (%.1f MB) for codes %v; saved %s\n",
		m.Pair.Addr.N, pair.Out, m.Pair.Addr.Cells(), float64(m.Pair.MemoryBytes())/1e6, m.Pair.Addr.Radices, *out)
	return nil
}

func loadModel(fs *flag.FlagSet, args []string) (*pair.Model, string, error) {
	path := fs.String("model", "model.json.gz", "model file")
	out := fs.String("out", "", "where to save the changed model (default: over the model file)")
	fs.Parse(args)
	m, err := pair.Load(*path)
	if err != nil {
		return nil, "", err
	}
	if *out == "" {
		*out = *path
	}
	return m, *out, nil
}

func trainCmd(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	text := fs.String("text", "", "a text to read instead of files")
	m, out, err := loadModel(fs, args)
	if err != nil {
		return err
	}
	var texts [][]byte
	if *text != "" {
		t, err := readText(*text)
		if err != nil {
			return err
		}
		texts = [][]byte{[]byte(t)}
	} else if texts, err = readFiles(fs.Args()); err != nil {
		return err
	}
	started := time.Now()
	r := m.TrainBytes(texts)
	fmt.Printf("read %v texts, %v units in %.2fs; saving %s\n", r["texts"], r["units"], time.Since(started).Seconds(), out)
	return m.Save(out)
}

func judgeOneCmd(kind string, args []string) error {
	fs := flag.NewFlagSet(kind, flag.ExitOnError)
	text := fs.String("text", "", "the outcome")
	prefix := fs.String("prefix", "", "context the outcome followed (not judged itself)")
	strength := fs.Float64("strength", 0, "strength (0: the model's default)")
	outcomes := fs.Int("outcomes", 0, "potential outcomes of the space this verdict comes from (0: the model's setting)")
	noRead := fs.Bool("no-read", false, "reward without counting the text")
	m, out, err := loadModel(fs, args)
	if err != nil {
		return err
	}
	t, err := readText(*text)
	if err != nil {
		return err
	}
	var r pair.Record
	if kind == "reward" {
		r, err = m.Reward([]string{t}, *strength, nil, !*noRead, *prefix, *outcomes)
	} else {
		r, err = m.Punish([]string{t}, *strength, nil, *prefix, *outcomes)
	}
	if err != nil {
		return err
	}
	printJSON(r)
	return m.Save(out)
}

func judgeCmd(args []string) error {
	fs := flag.NewFlagSet("judge", flag.ExitOnError)
	good := fs.String("good", "", "the correct outcome")
	bad := fs.String("bad", "", "the wrong outcome")
	prefix := fs.String("prefix", "", "context both followed")
	strength := fs.Float64("strength", 0, "strength (0: the model's default)")
	outcomes := fs.Int("outcomes", 0, "potential outcomes of the space this verdict comes from (0: the model's setting)")
	m, out, err := loadModel(fs, args)
	if err != nil {
		return err
	}
	r, err := m.TwoNRL([]string{*bad}, []string{*good}, *strength, *prefix, *outcomes)
	if err != nil {
		return err
	}
	printJSON(r)
	return m.Save(out)
}

func predictCmd(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ExitOnError)
	prefix := fs.String("prefix", "", "the context to continue")
	length := fs.Int("length", 32, "bytes to generate")
	mode := fs.String("mode", "greedy", "greedy | sample")
	traversal := fs.String("traversal", "reward", "reward | punishment")
	backoff := fs.String("backoff", "", "all | deepest | none (default: the model's)")
	temperature := fs.Float64("temperature", 1, "sampling temperature")
	toEnd := fs.Bool("to-end", false, "walk until the end mark")
	asJSON := fs.Bool("json", false, "print the whole result")
	path := fs.String("model", "model.json.gz", "model file")
	fs.Parse(args)
	m, err := pair.Load(*path)
	if err != nil {
		return err
	}
	p, err := readText(*prefix)
	if err != nil {
		return err
	}
	r, err := m.Predict(p, *length, *mode, *traversal, *temperature, *toEnd, *backoff, 4096)
	if err != nil {
		return err
	}
	if *asJSON {
		printJSON(r)
		return nil
	}
	fmt.Printf("%s|%s\n", p, r.Text)
	fmt.Printf("(%d units, cost %.3f nats, %.2f bits/unit, reached end: %v, first-step peak %.3f)\n",
		len(r.Units), r.Cost, r.Cost/math.Ln2/float64(max(len(r.Units), 1)), r.ReachedEnd, r.Peak)
	if *mode == "sample" {
		return m.Save(*path)
	}
	return nil
}

func foldCmd(args []string) error {
	fs := flag.NewFlagSet("fold", flag.ExitOnError)
	prefix := fs.String("prefix", "", "the context")
	traversal := fs.String("traversal", "reward", "reward | punishment")
	backoff := fs.String("backoff", "", "all | deepest | none (default: the model's)")
	top := fs.Int("top", 8, "outcomes to list")
	path := fs.String("model", "model.json.gz", "model file")
	fs.Parse(args)
	m, err := pair.Load(*path)
	if err != nil {
		return err
	}
	p, err := readText(*prefix)
	if err != nil {
		return err
	}
	return showFold(m, p, *traversal, *backoff, *top)
}

// showFold prints a context's code with each level's reading and the top outcomes of the fold.
func showFold(m *pair.Model, prefix, traversal, backoff string, top int) error {
	code := m.Code(prefix)
	chain := m.Pair.Addr.Chain(code, nil)
	fmt.Printf("context %q -> code %s\n", prefix, fmtCode(code))
	for l, node := range chain {
		d := "(root: every context)"
		if l > 0 {
			d, _ = m.Tok.Decode(code, l)
			d = fmt.Sprintf("%q", d)
		}
		fmt.Printf("  level %d node %-8d seen %-8d own %.3f  decodes to %s\n", l, node, m.Pair.Count.Ctx[node], m.Pair.Count.Own(node), d)
	}
	P, err := m.Pair.Fold(code, traversal, backoff)
	if err != nil {
		return err
	}
	idx := make([]int, len(P))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return P[idx[a]] > P[idx[b]] })
	fmt.Printf("  next (%s):", traversal)
	for _, x := range idx[:min(top, len(idx))] {
		fmt.Printf("  %s %.3f", pair.Symbol(x), P[x])
	}
	fmt.Println()
	return nil
}

func scoreCmd(args []string) error {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	text := fs.String("text", "", "the text to price")
	traversal := fs.String("traversal", "reward", "reward | punishment")
	backoff := fs.String("backoff", "", "all | deepest | none (default: the model's)")
	asJSON := fs.Bool("json", false, "print every unit")
	path := fs.String("model", "model.json.gz", "model file")
	fs.Parse(args)
	m, err := pair.Load(*path)
	if err != nil {
		return err
	}
	t, err := readText(*text)
	if err != nil {
		return err
	}
	s, err := m.Score(t, *traversal, *backoff)
	if err != nil {
		return err
	}
	if *asJSON {
		printJSON(s)
		return nil
	}
	fmt.Printf("%.3f bits/unit over %d units, mean reward %.3f, worst penalty %.3f\n", s.Bits, s.Units, s.MeanReward, s.WorstPenalty)
	return nil
}

func infoCmd(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	path := fs.String("model", "model.json.gz", "model file")
	fs.Parse(args)
	m, err := pair.Load(*path)
	if err != nil {
		return err
	}
	printJSON(m.Info())
	return nil
}

// demoCmd runs the whole loop on some files: train a tokenizer briefly, prime, read, show what the codes
// stand for, predict, then reward and punish and show the change.
func demoCmd(args []string) error {
	fs := flag.NewFlagSet("demo", flag.ExitOnError)
	steps := fs.Int("steps", 400, "tokenizer training steps")
	levels := fs.String("levels", "4,4:4,4:4,4", "quantisation levels")
	seed := fs.Int64("seed", 1, "random seed")
	out := fs.String("out", "", "where to save the demo model (default: not saved)")
	fs.Parse(args)
	texts, err := readFiles(fs.Args())
	if err != nil {
		return err
	}
	cfg := tokenizer.DefaultConfig()
	if cfg.Levels, err = tokenizer.ParseLevels(*levels); err != nil {
		return err
	}
	tok, err := tokenizer.New(cfg, *seed)
	if err != nil {
		return err
	}
	total := 0
	for _, t := range texts {
		total += len(t)
	}
	fmt.Printf("== tokenizer: %d parameters, window %d, code %s (%.1f bits); %d files, %d bytes, %d steps\n",
		tok.ParamCount(), cfg.Window, *levels, cfg.Bits(), len(texts), total, *steps)
	o := tokenizer.DefaultOptions()
	o.Steps, o.Seed, o.EvalEvery, o.Log = *steps, *seed, max(*steps/4, 1), logStat
	if _, err := tok.Train(texts, o); err != nil {
		return err
	}
	m, err := pair.Prime(tok, pair.DefaultSettings(), *seed)
	if err != nil {
		return err
	}
	fmt.Printf("\n== primed %d context nodes x %d outcomes (%.1f MB); reading the files\n", m.Pair.Addr.N, pair.Out, float64(m.Pair.MemoryBytes())/1e6)
	started := time.Now()
	r := m.TrainBytes(texts)
	fmt.Printf("read %v units in %.2fs\n", r["units"], time.Since(started).Seconds())
	sample := texts[0]
	if len(sample) > 400 {
		sample = sample[:400]
	}
	end := len(sample) - 1
	for end > 0 && sample[end] != ' ' {
		end--
	}
	ctx := string(sample[:end+1])
	if len(ctx) > 60 {
		ctx = ctx[len(ctx)-60:]
	}
	fmt.Printf("\n== what a code stands for\n")
	if err := showDecode(tok, []byte(ctx)); err != nil {
		return err
	}
	fmt.Printf("\n== the fold after that context\n")
	if err := showFold(m, ctx, "reward", "", 6); err != nil {
		return err
	}
	fmt.Printf("\n== greedy continuation\n")
	p, err := m.Predict(ctx, 40, "greedy", "reward", 0, false, "", 0)
	if err != nil {
		return err
	}
	fmt.Printf("%q -> %q (%.2f bits/unit)\n", ctx, p.Text, p.Cost/math.Ln2/float64(max(len(p.Units), 1)))
	fmt.Printf("\n== verdicts in outcome units\n")
	before, _ := m.Fold(ctx, "reward", "")
	first := p.Text
	if len(first) > 4 {
		first = first[:4]
	}
	alt := "zzz"
	if _, err := m.Reward([]string{alt}, 1, nil, true, ctx, 0); err != nil {
		return err
	}
	english, _ := m.Fold(ctx, "reward", "")
	if _, err := m.Reward([]string{alt}, 1, nil, false, ctx, 2); err != nil {
		return err
	}
	game, _ := m.Fold(ctx, "reward", "")
	fmt.Printf("rewarded %q after the context at strength 1: P(%q) %.4f; as an English verdict (1/%d per rung) %.4f; as a two-outcome verdict (1/2 per rung) %.4f\n",
		alt, pair.Symbol(int(alt[0])), before[alt[0]], m.Pair.Settings.Outcomes, english[alt[0]], game[alt[0]])
	if first != "" {
		if _, err := m.Punish([]string{first}, 1, nil, ctx, 2); err != nil {
			return err
		}
		pen, _ := m.Fold(ctx, "punishment", "")
		rew, _ := m.Fold(ctx, "reward", "")
		fmt.Printf("punished %q after the context as a two-outcome verdict: P(%q) reward traversal %.4f, punishment traversal %.4f\n",
			first, pair.Symbol(int(first[0])), rew[first[0]], pen[first[0]])
	}
	p2, _ := m.Predict(ctx, 12, "greedy", "punishment", 0, false, "", 0)
	fmt.Printf("punishment traversal now continues %q -> %q\n", ctx, p2.Text)
	if *out != "" {
		if err := m.Save(*out); err != nil {
			return err
		}
		fmt.Printf("\nsaved %s\n", *out)
	}
	return nil
}
