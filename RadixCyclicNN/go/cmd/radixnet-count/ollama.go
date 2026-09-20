package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// The LLM command groups: a training corpus written to order, the adversarial
// review that feeds the negative network, and ChatGPT's own two actions.

func ollamaUsage() {
	fmt.Fprint(os.Stderr, `usage: radixnet-count [global options] ollama <action> [options]

Two ways of hooking the network into a local large language model: a corpus written to
order, and an adversarial review of what the network itself writes.

actions:
  models   list the models the endpoint offers (never fails: it answers "is it there?")
  corpus   ask for --lines lines about --prompt (--style good | garbage), optionally training on them
  review   let the LLM mark --count samples (or --text / --data), optionally blaming the failures

--url and --ollama-model override $OLLAMA_HOST and $RADIXNET_OLLAMA_MODEL.
`)
}

func cmdOllama(args []string) {
	if len(args) == 0 {
		ollamaUsage()
		os.Exit(2)
	}
	action, rest := args[0], args[1:]
	switch action {
	case "models":
		cmdOllamaModels(rest)
	case "corpus":
		cmdOllamaCorpus(rest)
	case "review":
		cmdOllamaReview(rest)
	case "help", "-h", "--help":
		ollamaUsage()
	default:
		fail("unknown ollama action %q (models, corpus, review)", action)
	}
}

// llmFlags registers the provider flags an LLM command shares.
type llmFlags struct {
	url     *string
	model   *string
	timeout *float64
}

func addLLMFlags(fs *flag.FlagSet, modelFlag string) llmFlags {
	return llmFlags{
		url:     fs.String("url", "", "the endpoint's base URL (default: the provider's)"),
		model:   fs.String(modelFlag, "", "the model to answer with (default: the provider's)"),
		timeout: fs.Float64("timeout", 0, "per-request timeout in seconds"),
	}
}

func (f llmFlags) client(provider string) radixnet.LLMClient {
	client, err := radixnet.NewLLMClient(provider, *f.url, *f.model, time.Duration(*f.timeout*float64(time.Second)))
	if err != nil {
		fail("%v", err)
	}
	return client
}

func cmdOllamaModels(args []string) {
	fs := flag.NewFlagSet("ollama models", flag.ExitOnError)
	flags := addLLMFlags(fs, "ollama-model")
	_ = fs.Parse(args)
	client := flags.client(radixnet.ProviderOllama)
	models, err := client.Models()
	doc := map[string]any{
		"available": err == nil, "url": client.BaseURL(), "model": client.ModelName(),
		"models": models, "error": nil,
	}
	say("ollama         %s", client.BaseURL())
	say("default model  %s", client.ModelName())
	if err != nil {
		doc["error"] = err.Error()
		doc["models"] = []map[string]any{}
		say("error          %v", err)
	} else {
		say("")
		say("%-32s %12s", "model", "size")
		for _, model := range models {
			name, _ := model["name"].(string)
			say("%-32s %12s", name, humanSize(model["size"]))
		}
	}
	if jsonMode {
		emit(doc)
	}
}

// clip shortens a text for a table cell, marking that it was cut.
func clip(text string, width int) string {
	text = strings.Join(strings.Fields(text), " ")
	if len([]rune(text)) <= width {
		return text
	}
	return string([]rune(text)[:width-1]) + "…"
}

// humanSize renders a model's byte count the way `ollama list` does.
func humanSize(value any) string {
	bytes, ok := value.(float64)
	if !ok || bytes <= 0 {
		return "-"
	}
	units := []string{"B", "KB", "MB", "GB", "TB"}
	at := 0
	for bytes >= 1024 && at < len(units)-1 {
		bytes /= 1024
		at++
	}
	return fmt.Sprintf("%.1f %s", bytes, units[at])
}

func cmdOllamaCorpus(args []string) {
	fs := flag.NewFlagSet("ollama corpus", flag.ExitOnError)
	flags := addLLMFlags(fs, "ollama-model")
	prompt := fs.String("prompt", "", "what the lines should be about (required)")
	lines := fs.Int("lines", 20, "lines to ask for")
	style := fs.String("style", "good", "good (correct text) | garbage (deliberately wrong text)")
	out := fs.String("out", "", "write the lines to this file")
	train := fs.Bool("train", false, "train the model on the lines, then save it")
	epochs := fs.Int("epochs", 3, "training epochs with -train")
	modelOut := fs.String("model-out", "", "where to save the model with -train (default: --model)")
	_ = fs.Parse(args)

	if strings.TrimSpace(*prompt) == "" {
		fail("-prompt is required: say what the lines should be about")
	}
	client := flags.client(radixnet.ProviderOllama)
	say("ollama   %s: %s", client.BaseURL(), client.ModelName())
	say("prompt   %s", *prompt)
	say("style    %s", *style)
	texts, err := radixnet.CorpusFromPrompt(client, *prompt, *lines, *style, client.ModelName())
	if err != nil {
		fail("%v", err)
	}
	say("lines    %d", len(texts))
	say("")
	for _, text := range texts {
		say("  %s", text)
	}
	doc := map[string]any{
		"prompt": *prompt, "style": *style, "lines": len(texts), "texts": texts,
		"model": client.ModelName(), "url": client.BaseURL(), "out": nil, "trained": nil,
	}
	if *out != "" {
		if err := os.WriteFile(*out, []byte(strings.Join(texts, "\n")+"\n"), 0o644); err != nil {
			fail("cannot write %s: %v", *out, err)
		}
		doc["out"] = *out
		say("")
		say("written to %s", *out)
	}
	if *train {
		if len(texts) == 0 {
			fail("the model answered with no usable lines, so there is nothing to train on")
		}
		m := openModel(false)
		records, err := m.Train(texts, radixnet.TrainOptions{Epochs: *epochs})
		if err != nil {
			fail("%v", err)
		}
		target := *modelOut
		if target == "" {
			target = modelFile()
		}
		if err := m.Save(target); err != nil {
			fail("cannot save %s: %v", target, err)
		}
		doc["trained"] = map[string]any{"epochs": len(records), "texts": len(texts), "saved": target}
		say("")
		say("trained %d epoch(s) on %d line(s); saved to %s", len(records), len(texts), target)
	}
	if jsonMode {
		emit(doc)
	}
}

func cmdOllamaReview(args []string) {
	var texts, data multiFlag
	fs := flag.NewFlagSet("ollama review", flag.ExitOnError)
	flags := addLLMFlags(fs, "ollama-model")
	fs.Var(&texts, "text", "review this text instead of sampling (repeatable)")
	fs.Var(&data, "data", "review the texts of FILE (one per line) instead of sampling")
	count := fs.Int("count", 8, "samples to draw from the model")
	prefix := fs.String("prefix", "", "continue this prefix instead of generating from scratch")
	maxLength := fs.Int("max-length", 60, "characters per sample")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature")
	threshold := fs.Float64("threshold", 6.0, "ratings at or above this pass")
	context := fs.String("context", "", "what the texts are meant to be (the reviewer's yardstick)")
	blame := fs.Bool("blame", false, "teach the negative network what failed and why")
	addNegativeFlag(fs)
	_ = fs.Parse(args)

	client := flags.client(radixnet.ProviderOllama)
	o := radixnet.AdversarialReviewOptions{
		Count: *count, Prefix: *prefix, MaxLength: *maxLength, Temperature: *temperature,
		Threshold: *threshold, Context: *context, Model: client.ModelName(),
	}
	var model *radixnet.Model
	given := []string{}
	for _, text := range texts {
		if strings.TrimSpace(text) != "" {
			given = append(given, text)
		}
	}
	if len(data) > 0 {
		given = append(given, readTexts(data, "lines", 0)...)
	}
	if len(given) > 0 {
		o.Texts = given
	} else {
		model = openModel(true)
		seed := seedFlag
		o.Seed = &seed
	}
	say("ollama    %s: %s", client.BaseURL(), client.ModelName())
	if len(given) > 0 {
		say("reviewing %d given text(s), pass at %g/10", len(given), *threshold)
	} else {
		say("reviewing %d sample(s) of %d chars, pass at %g/10", *count, *maxLength, *threshold)
	}
	say("")
	result, err := radixnet.AdversarialReview(model, client, o)
	if err != nil {
		fail("%v", err)
	}
	say("%-7s %6s %s", "verdict", "rating", "text / critique")
	for _, review := range result.Reviews {
		rating := "-"
		if review.Rating != nil {
			rating = fmt.Sprintf("%.1f", *review.Rating)
		}
		say("%-7s %6s %s", review.Verdict, rating, clip(review.Text, 48))
		say("%-7s %6s   %s", "", "", clip(review.Critique, 64))
	}
	say("")
	say("%d passed, %d failed; mean mark %s/10", len(result.Good), len(result.Bad), fmtMark(result.MeanRating))
	doc := map[string]any{
		"source": result.Source, "model": result.Model, "threshold": result.Threshold,
		"texts": result.Texts, "reviews": result.Reviews, "mean_rating": result.MeanRating,
		"pass_rate": result.PassRate, "good": result.Good, "bad": result.Bad, "negative": nil,
	}
	if *blame {
		negative := openNegative(false)
		report, err := radixnet.TeachReviews(negative, result.Reviews, result.Threshold, true, "review",
			radixnet.TeachOptions{})
		if err != nil {
			fail("%v", err)
		}
		saved := saveNegative(negative)
		say("")
		say("blamed %d failure(s) over %d edge(s); cleared %d", report.Blamed, report.Edges, report.Cleared)
		say("negative model: %s", saved)
		reasonTable(negative, 10)
		doc["negative"] = map[string]any{
			"path": saved, "taught": report, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	if jsonMode {
		emit(doc)
	}
}

// -- ChatGPT ------------------------------------------------------------------

func chatgptUsage() {
	fmt.Fprint(os.Stderr, `usage: radixnet-count [global options] chatgpt <action> [options]

ChatGPT (or any OpenAI-compatible server) as the teacher, judge or reviewer.  Needs
$OPENAI_API_KEY (or $OPENAI_API_KEY_FILE); $OPENAI_BASE_URL points at another server.

actions:
  models   which models the key may use (never fails)
  ask      one completion of --prompt (--system, --temperature, --json-answer)
`)
}

func cmdChatGPT(args []string) {
	if len(args) == 0 {
		chatgptUsage()
		os.Exit(2)
	}
	action, rest := args[0], args[1:]
	switch action {
	case "models":
		cmdChatGPTModels(rest)
	case "ask":
		cmdChatGPTAsk(rest)
	case "help", "-h", "--help":
		chatgptUsage()
	default:
		fail("unknown chatgpt action %q (models, ask)", action)
	}
}

func cmdChatGPTModels(args []string) {
	fs := flag.NewFlagSet("chatgpt models", flag.ExitOnError)
	flags := addLLMFlags(fs, "chatgpt-model")
	_ = fs.Parse(args)
	configured := radixnet.ChatGPTConfigured()
	doc := map[string]any{"configured": configured, "available": false, "models": []any{}, "error": nil}
	if !configured {
		doc["error"] = "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE)"
		say("configured  no (set OPENAI_API_KEY or OPENAI_API_KEY_FILE)")
		if jsonMode {
			emit(doc)
		}
		return
	}
	client := flags.client(radixnet.ProviderChatGPT)
	doc["url"], doc["model"] = client.BaseURL(), client.ModelName()
	models, err := client.Models()
	say("chatgpt        %s", client.BaseURL())
	say("default model  %s", client.ModelName())
	if err != nil {
		doc["error"] = err.Error()
		say("error          %v", err)
	} else {
		doc["available"], doc["models"] = true, models
		say("")
		for _, model := range models {
			name, _ := model["name"].(string)
			say("  %s", name)
		}
	}
	if jsonMode {
		emit(doc)
	}
}

func cmdChatGPTAsk(args []string) {
	fs := flag.NewFlagSet("chatgpt ask", flag.ExitOnError)
	flags := addLLMFlags(fs, "chatgpt-model")
	prompt := fs.String("prompt", "", "what to ask (required)")
	system := fs.String("system", "", "a system instruction")
	temperature := fs.Float64("temperature", 0.7, "sampling temperature")
	jsonAnswer := fs.Bool("json-answer", false, "ask for a JSON answer")
	_ = fs.Parse(args)

	if strings.TrimSpace(*prompt) == "" {
		fail("-prompt is required")
	}
	if !radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE)")
	}
	client := flags.client(radixnet.ProviderChatGPT)
	answer, err := client.Generate(*prompt, radixnet.LLMOptions{
		System: *system, JSON: *jsonAnswer, Temperature: *temperature,
	})
	if err != nil {
		fail("%v", err)
	}
	say("%s", answer)
	if jsonMode {
		emit(map[string]any{
			"prompt": *prompt, "system": *system, "answer": answer,
			"model": client.ModelName(), "url": client.BaseURL(),
		})
	}
}
