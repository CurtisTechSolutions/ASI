package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// negativePath is --negative, else the negative model beside --model
// (model.count.json -> model.count.negative.json).
var negativePath = ""

func negativeFile() string {
	if strings.TrimSpace(negativePath) != "" {
		return negativePath
	}
	ext := filepath.Ext(modelFile())
	root := strings.TrimSuffix(modelFile(), ext)
	if ext == ".gz" {
		inner := filepath.Ext(root)
		root = strings.TrimSuffix(root, inner)
		ext = inner + ext
	}
	if strings.HasSuffix(root, ".negative") {
		return modelPath
	}
	return root + ".negative" + ext
}

// addNegativeFlag registers --negative on a command's flag set.
func addNegativeFlag(fs *flag.FlagSet) {
	fs.StringVar(&negativePath, "negative", negativePath,
		"negative model file (default: model.negative.json, i.e. beside --model)")
}

// openNegative loads the negative network from negativeFile(), else creates an empty one.
func openNegative(required bool) *radixnet.Model {
	path := negativeFile()
	if _, err := os.Stat(path); err == nil {
		m, err := radixnet.Load(path)
		if err != nil {
			fail("%s: %v", path, err)
		}
		if !m.IsNegative() {
			fail("%s holds a %s model, not a negative one", path, m.Kind())
		}
		return configure(checkEncoding(m, path))
	}
	if required {
		fail("negative model file not found: %s (teach it first with `radixnet-count negative blame --text '...' --reason gibberish`)", path)
	}
	negOpts := radixnet.DefaultNegativeOptions()
	negOpts.Encoding, _ = encodingFlags()
	m, err := radixnet.NewNegativeModel(seedFlag, negOpts)
	if err != nil {
		fail("%v", err)
	}
	return configure(checkEncoding(m, path))
}

// noGuard turns the guard off for one command: print what the positive model
// wrote, whatever the negative network says.
var noGuard = false

// guardConfig is how strictly the guard filters (the flags below set it).
var guardConfig = radixnet.DefaultFilterConfig()

// noProvenance makes the guard veto without saying why: the report counts the
// candidates it stopped instead of listing them with the rule, the reasons and
// the fragments behind each.
var noProvenance = false

// addGuardFlags registers the guard on a command that writes something: the
// negative network filters its output by default.
func addGuardFlags(fs *flag.FlagSet) {
	addNegativeFlag(fs)
	fs.BoolVar(&noGuard, "no-guard", noGuard,
		"do not filter: print what the positive model wrote, whatever the negative network says")
	fs.Func("threshold", "veto at this risk (blame per transition); default: the negative model's own", func(v string) error {
		f, err := strconv.ParseFloat(v, 64)
		guardConfig.Threshold = &f
		return err
	})
	fs.Func("min-coverage", "share of a text that must be known failure before any rule may veto it", func(v string) error {
		f, err := strconv.ParseFloat(v, 64)
		guardConfig.MinCoverage = &f
		return err
	})
	fs.IntVar(&guardConfig.OverSample, "over-sample", guardConfig.OverSample,
		"generate: candidates drawn per wanted text, so the guard has something to choose from")
	fs.BoolVar(&noProvenance, "no-provenance", noProvenance,
		"veto without saying why: report how many candidates the guard stopped, not which nor the rule, "+
			"the reasons and the fragments behind each")
}

// openGuard is the negative network guarding positive's output, or nil when
// there is nothing to guard with: --no-guard was given, there is no negative
// model file beside the model, or the one there has never been taught a
// failure and so would veto nothing.  The pair on every answer this program
// prints (radixnet.Filter): the positive model writes, the negative one
// vetoes what it recognises as a failure the tutor has already corrected.
func openGuard(positive *radixnet.Model) *radixnet.Filter {
	if noGuard || positive.IsNegative() {
		return nil
	}
	path := negativeFile()
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	negative := openNegative(true)
	guardConfig.Provenance = !noProvenance
	pair, err := radixnet.NewFilter(positive, negative, guardConfig)
	if err != nil {
		fail("%v", err)
	}
	if !pair.Ready() {
		return nil
	}
	note("guard: %s (%.4f blame over %d reasons)", path, negative.G.Neg.TotalBlame, len(negative.G.Neg.ReasonNames))
	return pair
}

// printVetoes reports the guard's work: what it let through, what it stopped
// and - with provenance - why.
func printVetoes(verdicts []*radixnet.FilterVerdict, what string) {
	rejected := []*radixnet.FilterVerdict{}
	for _, verdict := range verdicts {
		if verdict.Decision == "reject" {
			rejected = append(rejected, verdict)
		}
	}
	fmt.Printf("\nguard: %d of %d %s passed the negative network\n", len(verdicts)-len(rejected), len(verdicts), what)
	if len(rejected) == 0 || !guardConfig.Provenance {
		return
	}
	fmt.Printf("%-8s %8s %8s %8s  %-18s %s\n", "rule", "risk", "peak", "ratio", "reason", "vetoed")
	for _, verdict := range rejected {
		rule := "-"
		if verdict.Rule != nil {
			rule = *verdict.Rule
		}
		reason := "-"
		if len(verdict.Reasons) > 0 {
			reason = verdict.Reasons[0].Reason
		}
		fmt.Printf("%-8s %8.4f %8.4f %8.4f  %-18s %s\n", rule, verdict.Risk, verdict.Peak, verdict.Ratio, reason,
			quote(clip(verdict.Text, 46)))
	}
}

// guardDoc is the guard's report for --json: every veto, with the reason and
// the fragment behind it.  With -no-provenance it is the counts alone: how
// many were judged and how many vetoed, neither listed.
func guardDoc(pair *radixnet.Filter, verdicts []*radixnet.FilterVerdict, extra map[string]any) map[string]any {
	rejected := []*radixnet.FilterVerdict{}
	for _, verdict := range verdicts {
		if verdict.Decision == "reject" {
			rejected = append(rejected, verdict)
		}
	}
	var out map[string]any
	if pair.Config.Provenance {
		out = map[string]any{
			"on": true, "vetoed": len(rejected), "rejected": rejected, "verdicts": verdicts,
			"negative": pair.Negative.Stats(), "config": pair.Describe()["config"],
		}
	} else {
		out = map[string]any{
			"on": true, "provenance": false, "judged": len(verdicts), "vetoed": len(rejected),
			"negative": pair.Negative.Stats(), "config": pair.Describe()["config"],
		}
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

func saveNegative(m *radixnet.Model) string {
	path := negativeFile()
	if err := m.Save(path); err != nil {
		fail("cannot save %s: %v", path, err)
	}
	return path
}

// negativeTexts collects --text (repeatable) and --data files.
func negativeTexts(texts multiFlag, data multiFlag, unit string, pageLines int, what string) []string {
	out := []string{}
	for _, text := range texts {
		if strings.TrimSpace(text) != "" {
			out = append(out, text)
		}
	}
	if len(data) > 0 {
		out = append(out, readTexts(data, unit, pageLines)...)
	}
	if len(out) == 0 {
		fail("no texts to %s: give --text TEXT (repeatable) or --data FILE", what)
	}
	return out
}

func negativeUsage() {
	fmt.Fprint(os.Stderr, `radixnet-count negative <action> [options]

A copy of the network that keeps only its negative portions: every node and edge in it exists
because something went wrong there, and every edge remembers why - the reasons the tutor gave,
with the blame each one carries.

actions:
  blame     learn a failure: --text / --data, --reason TAG, --severity N, --source NAME, --note TEXT
  clear     the tutor passed these texts: take blame off the fragments they share
  why       why a text looks like a failure: risk, coverage, the reasons and the blamed fragments
  filter    the pair: the positive model writes, the negative one vetoes (or judge --text / --data)
  reasons   what the tutor has blamed, and the journal of what it said
  forget    drop or fade the blame behind --reason (the tutor can be wrong too)
  auto      teach it automatically: the model writes, an LLM reviews, the failures are blamed

The negative model lives beside --model unless --negative says otherwise.
`)
}

func cmdNegative(args []string) {
	if len(args) == 0 {
		negativeUsage()
		os.Exit(2)
	}
	action, rest := args[0], args[1:]
	switch action {
	case "blame":
		cmdNegativeBlame(rest)
	case "clear":
		cmdNegativeClear(rest)
	case "why":
		cmdNegativeWhy(rest)
	case "filter":
		cmdNegativeFilter(rest)
	case "reasons":
		cmdNegativeReasons(rest)
	case "forget":
		cmdNegativeForget(rest)
	case "auto":
		cmdNegativeAuto(rest)
	case "help", "-h", "--help":
		negativeUsage()
	default:
		fail("unknown negative action %q (blame, clear, why, filter, reasons, forget, auto)", action)
	}
}

func reasonTable(m *radixnet.Model, limit int) {
	rows := m.Reasons()
	if len(rows) == 0 {
		say("nothing has been blamed yet")
		return
	}
	say("%-16s %10s %7s %7s %7s", "reason", "blame", "fails", "edges", "share")
	for i, row := range rows {
		if limit > 0 && i >= limit {
			break
		}
		say("%-16s %10.4f %7d %7d %7.4f", row.Reason, row.Blame, row.Fails, row.Edges, row.Share)
	}
}

func cmdNegativeBlame(args []string) {
	fs := subFlagSet("negative blame")
	addNegativeFlag(fs)
	var texts, data multiFlag
	fs.Var(&texts, "text", "a failed text (repeatable)")
	fs.Var(&data, "data", "failed texts, one per line (repeatable; text or ZIP)")
	reason := fs.String("reason", radixnet.UnspecifiedReason, "why it failed (the tutor's verdict)")
	severity := fs.Float64("severity", 1.0, "how heavily to blame it (1 = one ordinary failure)")
	source := fs.String("source", "cli", "who says so (tutor, review, evolve, frontend, cli)")
	note := fs.String("note", "", "the tutor's own words, kept in the journal")
	epochs := fs.Int("epochs", 1, "blame passes over the texts")
	unit := fs.String("split", "lines", "how --data files are cut: lines | paragraphs | pages | file")
	pageLines := fs.Int("page-lines", 40, "lines per page when --split pages")
	_ = fs.Parse(args)
	failed := negativeTexts(texts, data, *unit, *pageLines, "blame")
	m := openNegative(false)
	records, err := m.Blame(failed, radixnet.BlameOptions{
		Reason: *reason, Severity: *severity, Source: *source, Note: *note, Epochs: *epochs,
	})
	if err != nil {
		fail("%v", err)
	}
	path := saveNegative(m)
	say("blamed %d text(s) for %s (severity %g)", len(failed), radixnet.CleanReason(*reason), *severity)
	if len(records) > 0 {
		last := records[len(records)-1]
		say("%v edge(s) know about it now; %v nodes / %v edges", last["edges_touched"], last["nodes"], last["edges"])
	}
	say("saved %s", path)
	reasonTable(m, 10)
	if jsonMode {
		emit(map[string]any{
			"texts": len(failed), "reason": radixnet.CleanReason(*reason), "severity": *severity,
			"source": *source, "records": records, "saved": path, "reasons": m.Reasons(), "stats": m.Stats(),
		})
	}
}

func cmdNegativeClear(args []string) {
	fs := subFlagSet("negative clear")
	addNegativeFlag(fs)
	var texts, data multiFlag
	fs.Var(&texts, "text", "a passed text (repeatable)")
	fs.Var(&data, "data", "passed texts, one per line (repeatable)")
	weight := fs.Float64("weight", 1.0, "how much blame one pass cancels")
	epochs := fs.Int("epochs", 1, "clearing passes over the texts")
	unit := fs.String("split", "lines", "how --data files are cut: lines | paragraphs | pages | file")
	pageLines := fs.Int("page-lines", 40, "lines per page when --split pages")
	_ = fs.Parse(args)
	passed := negativeTexts(texts, data, *unit, *pageLines, "clear")
	m := openNegative(true)
	records, err := m.Clear(passed, *weight, *epochs)
	if err != nil {
		fail("%v", err)
	}
	path := saveNegative(m)
	matched, unmatched, touched := 0, 0, 0
	if len(records) > 0 {
		last := records[len(records)-1]
		matched, unmatched = int(toF(last["matched"])), int(toF(last["unmatched"]))
		touched = int(toF(last["edges_touched"]))
	}
	say("matched %d of %d text(s) (%d edges cleared, %d shared nothing with a failure)", matched, len(passed), touched, unmatched)
	say("saved %s", path)
	if jsonMode {
		emit(map[string]any{"texts": len(passed), "weight": *weight, "matched": matched, "unmatched": unmatched,
			"records": records, "saved": path, "stats": m.Stats()})
	}
}

// sayVerdict prints one judgement: the sentence, the reasons and the fragments to blame.
func sayVerdict(text string, v *radixnet.Verdict) {
	say("text       %q", text)
	say("verdict    %s", v.Verdict)
	say("risk       %.4f", v.Risk)
	say("peak       %.4f", v.Peak)
	say("coverage   %.4f", v.Coverage)
	say("blame      %.4f", v.Blame)
	say("why        %s", v.Why)
	if len(v.Reasons) > 0 {
		say("%-16s %10s %7s", "reason", "blame", "share")
		for _, row := range v.Reasons {
			say("%-16s %10.4f %7.4f", row.Reason, row.Blame, row.Share)
		}
	}
	if len(v.Spans) > 0 {
		say("%-10s %-14s %10s %7s %s", "at", "fragment", "blame", "fails", "reason")
		for _, span := range v.Spans {
			say("%-10s %-14q %10.4f %7d %s", fmt.Sprintf("%d..%d", span.Start, span.End), span.Fragment,
				span.Blame, span.Fails, span.Reason)
		}
	}
}

func cmdNegativeWhy(args []string) {
	fs := subFlagSet("negative why")
	addNegativeFlag(fs)
	var texts, data multiFlag
	fs.Var(&texts, "text", "a text to judge (repeatable)")
	fs.Var(&data, "data", "texts to judge, one per line")
	threshold := fs.Float64("threshold", -1, "reject at this blame per transition (default: the model's)")
	minCoverage := fs.Float64("min-coverage", -1, "known-failing share needed before rejecting")
	spans := fs.Int("spans", 5, "blamed fragments to show")
	unit := fs.String("split", "lines", "how --data files are cut")
	pageLines := fs.Int("page-lines", 40, "lines per page when --split pages")
	_ = fs.Parse(args)
	judged := negativeTexts(texts, data, *unit, *pageLines, "judge")
	m := openNegative(true)
	o := radixnet.JudgeOptions{Spans: *spans}
	if *threshold >= 0 {
		o.Threshold = threshold
	}
	if *minCoverage >= 0 {
		o.MinCoverage = minCoverage
	}
	verdicts := []*radixnet.Verdict{}
	for i, text := range judged {
		verdict := m.Judge(text, o)
		verdicts = append(verdicts, verdict)
		if i > 0 {
			say("")
		}
		sayVerdict(text, verdict)
	}
	if jsonMode {
		emit(map[string]any{"verdicts": verdicts, "stats": m.Stats()})
	}
}

func cmdNegativeFilter(args []string) {
	fs := subFlagSet("negative filter")
	addNegativeFlag(fs)
	count := fs.Int("count", 3, "texts wanted out of the filter")
	prefix := fs.String("prefix", "", "continue this prefix")
	maxLength := fs.Int("max-length", 60, "characters per candidate")
	mode := fs.String("mode", "sample", "how the positive model writes: sample | beam | dijkstra")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature")
	stepPenalty := fs.Float64("step-penalty", 0.0, "extra cost per edge")
	overSample := fs.Int("over-sample", 3, "candidates drawn per wanted text")
	threshold := fs.Float64("threshold", -1, "reject at this blame per transition (default: the model's)")
	minCoverage := fs.Float64("min-coverage", -1, "known-failing share needed before rejecting")
	ratio := fs.Float64("ratio", 0, "reject when the candidate reads this much more like failure than like the training data")
	noRatio := fs.Bool("no-ratio", false, "judge by blame alone (turn the likelihood ratio off)")
	peak := fs.Float64("peak", -1, "reject a candidate carrying this much blame on a single fragment (off by default)")
	strict := fs.Bool("strict", false, "also drop candidates the negative network only finds suspect")
	spans := fs.Int("spans", 3, "blamed fragments per verdict")
	learn := fs.Bool("learn", false, "blame what the filter rejects (off: the tutor supplies the negatives)")
	terse := fs.Bool("no-provenance", false, "report the decisions alone: no risk, no reasons, no blamed fragments behind a veto")
	var texts, data multiFlag
	fs.Var(&texts, "text", "judge this text instead of generating (repeatable)")
	fs.Var(&data, "data", "judge the texts of a file instead of generating")
	unit := fs.String("split", "lines", "how --data files are cut")
	pageLines := fs.Int("page-lines", 40, "lines per page when --split pages")
	_ = fs.Parse(args)

	negative := openNegative(true)
	positive := openModel(true)
	config := radixnet.DefaultFilterConfig()
	config.OverSample, config.Strict, config.Spans, config.Learn = *overSample, *strict, *spans, *learn
	config.Provenance = !*terse
	if *threshold >= 0 {
		config.Threshold = threshold
	}
	if *minCoverage >= 0 {
		config.MinCoverage = minCoverage
	}
	if *noRatio {
		config.Ratio = nil
	} else {
		config.Ratio = ratio
	}
	if *peak >= 0 {
		config.Peak = peak
	}
	pair, err := radixnet.NewFilter(positive, negative, config)
	if err != nil {
		fail("%v", err)
	}
	given := []string{}
	for _, text := range texts {
		if strings.TrimSpace(text) != "" {
			given = append(given, text)
		}
	}
	if len(data) > 0 {
		given = append(given, readTexts(data, *unit, *pageLines)...)
	}
	var outcome *radixnet.FilterOutcome
	if len(given) > 0 {
		outcome, err = pair.Filter(given)
	} else {
		o := radixnet.DefaultGenerateOptions()
		o.MaxLength, o.Mode, o.Temperature, o.Prefix, o.StepPenalty = *maxLength, *mode, *temperature, *prefix, *stepPenalty
		if seedFlag != 0 {
			seed := seedFlag
			o.Seed = &seed
		}
		outcome, err = pair.Generate(*count, o)
	}
	if err != nil {
		fail("%v", err)
	}
	if config.Provenance {
		say("%-9s %-6s %8s %8s %9s %-14s %s", "decision", "rule", "risk", "peak", "ratio", "reason", "text")
	} else { // the decisions alone: what was kept and what was vetoed, not why
		say("%-9s %-6s %s", "decision", "rule", "text")
	}
	for _, v := range outcome.Verdicts {
		reason := "-"
		if len(v.Reasons) > 0 {
			reason = v.Reasons[0].Reason
		}
		rule := "-"
		if v.Rule != nil {
			rule = *v.Rule
		}
		if !config.Provenance {
			say("%-9s %-6s %q", v.Decision, rule, v.Text)
			continue
		}
		say("%-9s %-6s %8.4f %8.4f %9.4f %-14s %q", v.Decision, rule, v.Risk, v.Peak, v.Ratio, reason, v.Text)
	}
	say("")
	summary := fmt.Sprintf("%d candidates: %d passed the filter, %d vetoed", len(outcome.Verdicts), len(outcome.Kept), len(outcome.Rejected))
	if outcome.Rate != nil {
		summary += fmt.Sprintf(" (acceptance %.4f)", *outcome.Rate)
	}
	if len(outcome.Texts) < len(outcome.Kept) {
		summary += fmt.Sprintf("; returning the %d cleanest", len(outcome.Texts))
	}
	say("%s", summary)
	for _, text := range outcome.Texts {
		say("  %q", text)
	}
	for _, v := range outcome.Rejected {
		if v.Why == "" {
			say("  vetoed: %q", v.Text)
			continue
		}
		say("  vetoed: %q - %s", v.Text, v.Why)
	}
	doc := map[string]any{
		"texts": outcome.Texts, "kept": outcome.Kept, "rejected": outcome.Rejected, "verdicts": outcome.Verdicts,
		"candidates": outcome.Candidates, "asked": outcome.Asked, "rate": outcome.Rate, "pair": pair.Describe(),
	}
	if config.Learn {
		doc["saved"] = saveNegative(negative)
		say("saved %s", doc["saved"])
	}
	if jsonMode {
		emit(doc)
	}
}

func cmdNegativeReasons(args []string) {
	fs := subFlagSet("negative reasons")
	addNegativeFlag(fs)
	limit := fs.Int("limit", 20, "reasons to list")
	logLimit := fs.Int("log", 10, "journal entries to show")
	_ = fs.Parse(args)
	m := openNegative(true)
	stats := m.Stats()
	say("negative model  %s", negativeFile())
	say("failures        %v", stats["failures_total"])
	say("blame           %v", stats["edge_blame_total"])
	say("cleared         %v", stats["cleared_total"])
	say("nodes / edges   %v / %v", stats["nodes"], stats["edges"])
	say("judgements      %v (%v rejected)", stats["judgements"], stats["rejected"])
	say("filter          risk >= %v, coverage >= %v", stats["threshold"], stats["min_coverage"])
	say("")
	reasonTable(m, *limit)
	journal := m.Recent(*logLimit)
	if len(journal) > 0 {
		say("")
		say("%-26s %-16s %8s %-10s %-30s %s", "when", "reason", "severity", "source", "text", "the tutor said")
		for _, entry := range journal {
			say("%-26s %-16s %8.4f %-10s %-30q %s", entry.At, entry.Reason, entry.Severity, entry.Source,
				entry.Text, entry.Note)
		}
	}
	if jsonMode {
		emit(map[string]any{"reasons": m.Reasons(), "journal": journal, "stats": stats, "path": negativeFile()})
	}
}

func cmdNegativeForget(args []string) {
	fs := subFlagSet("negative forget")
	addNegativeFlag(fs)
	reason := fs.String("reason", "", "the reason to forget (default: every reason)")
	factor := fs.Float64("factor", 0, "share of the blame to keep (0 = forget it, 0.5 = halve it)")
	_ = fs.Parse(args)
	m := openNegative(true)
	result, err := m.Forget(*reason, *factor)
	if err != nil {
		fail("%v", err)
	}
	path := saveNegative(m)
	say("forgot %s: %.4f blame off %d edge(s)", result.Reason, result.BlameRemoved, result.Edges)
	say("saved %s", path)
	reasonTable(m, 10)
	if jsonMode {
		emit(map[string]any{"reason": result.Reason, "edges": result.Edges, "blame_removed": result.BlameRemoved,
			"saved": path, "reasons": m.Reasons(), "stats": m.Stats()})
	}
}

// cmdNegativeAuto is the Negative tab without anyone typing a failure into it:
// the model writes, an LLM reviews, the failures are blamed - round after round.
// With -correct the LLM is a copy editor instead of a critic: it writes each
// text out correctly changing as little as it can, and only the characters it
// changed are blamed.
func cmdNegativeAuto(args []string) {
	cfg := radixnet.DefaultCriticConfig()
	fs := flag.NewFlagSet("negative auto", flag.ExitOnError)
	rounds := fs.Int("rounds", cfg.Rounds, "rounds to run (0: until interrupted)")
	count := fs.Int("count", cfg.Count, "texts the model writes per round")
	prefix := fs.String("prefix", "", "continue this prefix instead of writing from scratch")
	maxLength := fs.Int("max-length", cfg.MaxLength, "characters per text")
	temperature := fs.Float64("temperature", cfg.Temperature, "sampling temperature of the writing")
	threshold := fs.Float64("threshold", cfg.Threshold, "pass mark out of 10: below it the text is blamed")
	context := fs.String("context", "", "what the reviewer is told the texts are meant to be (its yardstick)")
	provider := fs.String("provider", cfg.Provider, "who reviews: ollama | chatgpt")
	reviewerModel := fs.String("reviewer-model", "", "the reviewer's model (default: the provider's)")
	url := fs.String("url", "", "the reviewer's base URL (default: the provider's)")
	timeout := fs.Float64("timeout", 0, "per-request timeout in seconds")
	epochs := fs.Int("epochs", cfg.Epochs, "blame epochs per round")
	noClear := fs.Bool("no-clear", false, "do not let the texts it passed take blame off what they share")
	correct := fs.Bool("correct", false, "ask for letter-level corrections instead of marks: the LLM writes each text "+
		"out correctly changing as little as it can, and only the characters it changed are blamed (the texts it "+
		"handed back unchanged clear blame); no pass mark applies")
	severity := fs.Float64("severity", cfg.Severity, "blame per corrected text with -correct (1 = one ordinary failure)")
	addNegativeFlag(fs)
	_ = fs.Parse(args)

	cfg.Rounds, cfg.Count, cfg.Prefix, cfg.MaxLength = *rounds, *count, *prefix, *maxLength
	cfg.Temperature, cfg.Threshold, cfg.Context = *temperature, *threshold, *context
	cfg.Provider, cfg.ReviewerModel, cfg.Epochs = *provider, *reviewerModel, *epochs
	cfg.ClearPasses, cfg.Correct, cfg.Severity = !*noClear, *correct, *severity
	seed := seedFlag
	cfg.Seed = &seed
	if err := cfg.Validate(); err != nil {
		fail("%v", err)
	}
	if cfg.Provider == radixnet.ProviderChatGPT && !radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT review, " +
			"or use -provider ollama")
	}
	client, err := radixnet.NewLLMClient(cfg.Provider, *url, cfg.ReviewerModel, time.Duration(*timeout*float64(time.Second)))
	if err != nil {
		fail("%v", err)
	}
	model := openModel(true)
	negative := openNegative(false)
	loop, err := radixnet.NewCritic(model, negative, client, cfg)
	if err != nil {
		fail("%v", err)
	}
	roundsText := fmt.Sprintf("%d", cfg.Rounds)
	if cfg.Rounds == 0 {
		roundsText = "until interrupted"
	}
	say("negative model: %s", negativeFile())
	role := "reviewer"
	if cfg.Correct {
		role = "editor"
	}
	say("%s: %s: %s at %s", role, cfg.Provider, client.ModelName(), client.BaseURL())
	lesson := fmt.Sprintf("pass at %g/10", cfg.Threshold)
	if cfg.Correct {
		lesson = fmt.Sprintf("letter-level corrections, %g blame per corrected text", cfg.Severity)
	}
	say("%s round(s) x %d text(s) of %d chars, %s%s", roundsText, cfg.Count, cfg.MaxLength, lesson,
		map[bool]string{true: "", false: " (passes do not clear)"}[cfg.ClearPasses])
	if strings.TrimSpace(cfg.Context) != "" {
		say("context: %s", cfg.Context)
	}
	say("")
	stop := interruptible()
	loop.Stop = stop
	if !jsonMode {
		loop.Progress = func(record map[string]any) {
			if record["kind"] != "round" {
				return
			}
			if record["mode"] == "correct" {
				say("round %v: %v/%v corrected (%v change(s)), blamed %v over %v edge(s), cleared %v%s",
					record["round"], record["corrected"], record["texts"], record["edits"], record["blamed"],
					record["edges"], record["cleared"], reasonSummary(record["reasons"]))
				return
			}
			say("round %v: %v/%v failed, blamed %v over %v edge(s), cleared %v%s",
				record["round"], record["failed"], record["texts"], record["blamed"], record["edges"],
				record["cleared"], reasonSummary(record["reasons"]))
		}
	}
	records, err := loop.Run()
	if err != nil {
		fail("%v", err)
	}
	saved := saveNegative(negative)
	card := map[string]any{}
	if len(records) > 0 {
		if last := records[len(records)-1]; last["kind"] == "report" {
			card = last
		}
	}
	say("")
	if cfg.Correct {
		say("%v round(s): corrected %v of %v (%v change(s)), blamed %v, cleared %v; change rate %s%s",
			card["rounds"], card["corrected"], card["reviewed"], card["edits"], card["blamed"], card["cleared"],
			fmtRate(ratePtr(card["change_rate"])), trendSummary(card["change_trend"]))
	} else {
		say("%v round(s): reviewed %v, blamed %v, cleared %v; mean mark %s/10%s",
			card["rounds"], card["reviewed"], card["blamed"], card["cleared"], fmtMark(card["mean_rating"]),
			trendSummary(card["trend"]))
	}
	say("negative model: %s", saved)
	reasonTable(negative, 10)
	if jsonMode {
		emit(map[string]any{
			"config": cfg, "url": client.BaseURL(), "reviewer": client.ModelName(),
			"records": records, "report": card, "negative": map[string]any{
				"path": saved, "reasons": negative.Reasons(), "stats": negative.Stats(),
			},
		})
	}
}

// ratePtr reads a report card's rate however it was stored.
func ratePtr(value any) *float64 {
	switch v := value.(type) {
	case *float64:
		return v
	case float64:
		return &v
	}
	return nil
}

// reasonSummary renders a round's reason histogram as "; repetition x3".
func reasonSummary(value any) string {
	counts, ok := value.(map[string]int)
	if !ok || len(counts) == 0 {
		return ""
	}
	parts := []string{}
	for _, reason := range radixnet.ReasonOrder(counts) {
		parts = append(parts, fmt.Sprintf("%s x%d", reason, counts[reason]))
	}
	return "; " + strings.Join(parts, ", ")
}

// trendSummary renders the report card's trend as ", trend +1.25".
func trendSummary(value any) string {
	trend, ok := value.(float64)
	if !ok {
		return ""
	}
	return fmt.Sprintf(", trend %+.2f", trend)
}
