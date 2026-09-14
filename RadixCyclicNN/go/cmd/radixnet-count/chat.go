package main

// `radixnet-count chat`: an LLM converses with the model and marks every
// reply.  The Go twin of the Python CLI's chat command - the other side of the
// line is a real language model, the network replies by continuing what it
// said (the same search `converse` uses), and the LLM then marks each reply out
// of 10 against the line it answered, and the conversation as a whole.

import (
	"fmt"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

func cmdChat(args []string) {
	cfg := radixnet.DefaultChatConfig()
	fs := subFlagSet("chat")
	conversations := fs.Int("conversations", cfg.Conversations, "conversations to hold (0: until Ctrl-C)")
	turns := fs.Int("turns", cfg.Turns, "replies the model gives per conversation")
	topic := fs.String("topic", "", "what to talk about (default: the partner chooses)")
	opening := fs.String("opening", "", "the first line, spoken as given (default: the partner opens)")
	persona := fs.String("persona", "", "who the partner is being (\"a curious child\", \"a vet\")")
	context := fs.Int("context", cfg.Context, "characters of the previous line a reply picks up")
	maxLength := fs.Int("max-length", cfg.MaxLength, "characters a reply may add to its context")
	mode := fs.String("mode", cfg.Mode, "how a reply is found: beam | sample")
	k := fs.Int("k", cfg.K, "candidates considered per reply")
	temperature := fs.Float64("temperature", cfg.Temperature, "sampling temperature of the replies")
	partnerTemperature := fs.Float64("partner-temperature", cfg.PartnerTemperature,
		"sampling temperature of the partner's lines")
	threshold := fs.Float64("threshold", cfg.Threshold, "pass mark out of 10: below it a reply is a failure")
	provider := fs.String("provider", cfg.Provider, "who converses and marks: ollama | chatgpt")
	partnerModel := fs.String("partner-model", "", "the partner's model (default: the provider's)")
	judgeModel := fs.String("judge-model", "", "a different model for marking (default: the partner's)")
	url := fs.String("url", "", "the partner's base URL (default: the provider's)")
	judgeURL := fs.String("judge-url", "", "base URL of the judge (default: -url)")
	timeout := fs.Float64("timeout", 0, "per-request timeout in seconds")
	allowRepeats := fs.Bool("allow-repeats", false, "let the model say something already heard")
	noGuard := fs.Bool("no-guard", false, "do not let the negative network veto a reply before it is spoken")
	noBlame := fs.Bool("no-blame", false, "do not blame the failed replies")
	noClear := fs.Bool("no-clear", false, "do not let the passed replies clear blame")
	noLearn := fs.Bool("no-learn", false, "mark the conversation but do not train on it")
	noTeachPartner := fs.Bool("no-teach-partner", false, "keep the partner's own lines out of the positive phase")
	epochs := fs.Int("epochs", cfg.Epochs, "blame epochs per conversation")
	negEpochs := fs.Int("neg-epochs", cfg.NegEpochs, "epochs of the negative (punish) phase")
	posEpochs := fs.Int("pos-epochs", cfg.PosEpochs, "epochs of the positive (reward) phase")
	strength := fs.Float64("strength", cfg.Strength, "the magnitude of a penalty or reward")
	addNegativeFlag(fs)
	_ = fs.Parse(args)

	cfg.Conversations, cfg.Turns, cfg.Topic, cfg.Opening, cfg.Persona = *conversations, *turns, *topic, *opening, *persona
	cfg.Context, cfg.MaxLength, cfg.K = *context, *maxLength, *k
	cfg.Mode = strings.ToLower(strings.TrimSpace(*mode))
	cfg.Temperature, cfg.PartnerTemperature, cfg.Threshold = *temperature, *partnerTemperature, *threshold
	cfg.Provider, cfg.PartnerModel, cfg.JudgeModel = *provider, *partnerModel, *judgeModel
	cfg.Guard, cfg.Blame, cfg.ClearPasses = !*noGuard, !*noBlame, !*noClear
	cfg.Learn, cfg.TeachPartner, cfg.AvoidRepeats = !*noLearn, !*noTeachPartner, !*allowRepeats
	cfg.Epochs, cfg.NegEpochs, cfg.PosEpochs, cfg.Strength = *epochs, *negEpochs, *posEpochs, *strength
	seed := seedFlag
	cfg.Seed = &seed
	if err := cfg.Validate(); err != nil {
		fail("%v", err)
	}
	if cfg.Provider == radixnet.ProviderChatGPT && !radixnet.ChatGPTConfigured() {
		fail("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT converse, " +
			"or use -provider ollama")
	}
	wait := time.Duration(*timeout * float64(time.Second))
	client, err := radixnet.NewLLMClient(cfg.Provider, *url, cfg.PartnerModel, wait)
	if err != nil {
		fail("%v", err)
	}
	judge := client
	if cfg.JudgeModel != "" || *judgeURL != "" {
		at := *judgeURL
		if at == "" {
			at = *url
		}
		if judge, err = radixnet.NewLLMClient(cfg.Provider, at, cfg.JudgeModel, wait); err != nil {
			fail("%v", err)
		}
	}
	model := openModel(true)
	var negative *radixnet.Model
	if cfg.Blame || cfg.Guard {
		negative = openNegative(false)
	}
	loop, err := radixnet.NewChat(model, negative, client, cfg)
	if err != nil {
		fail("%v", err)
	}
	loop.JudgeClient = judge

	conversationsText := fmt.Sprintf("%d", cfg.Conversations)
	if cfg.Conversations == 0 {
		conversationsText = "until interrupted"
	}
	topicText := cfg.Topic
	if strings.TrimSpace(topicText) == "" {
		topicText = "the partner chooses"
	}
	guardText := "off"
	if cfg.Guard && negative != nil {
		guardText = "on"
	}
	learnText := "nothing (-no-learn)"
	if cfg.Learn {
		learnText = "2NRL on the marked replies"
		if cfg.TeachPartner {
			learnText += " + the partner's lines"
		}
	}
	say("partner: %s: %s at %s", cfg.Provider, client.ModelName(), client.BaseURL())
	say("judge: %s at %s", judge.ModelName(), judge.BaseURL())
	say("%s conversation(s) x %d repl(ies) of %d chars, picking up %d, pass at %g/10",
		conversationsText, cfg.Turns, cfg.MaxLength, cfg.Context, cfg.Threshold)
	say("topic: %s", topicText)
	say("guard: %s; learns: %s", guardText, learnText)
	if negative != nil {
		say("negative model: %s", negativeFile())
	}
	say("")

	stop := interruptible()
	loop.Stop = stop
	progress := func(record map[string]any) {}
	if !jsonMode {
		progress = sayChatRecord
	}
	records, err := loop.Run(progress)
	if err != nil {
		fail("%v", err)
	}
	card := map[string]any{}
	if len(records) > 0 {
		if last := records[len(records)-1]; last["kind"] == "report" {
			card = last
		}
	}
	say("")
	say("%v conversation(s), %v repl(ies): %v passed, %v failed; mean mark %s/10%s",
		card["conversations"], card["exchanges"], card["passed"], card["failed"],
		fmtMark(card["mean_rating"]), trendSummary(card["trend"]))
	saved := ""
	if cfg.Learn {
		if stop() {
			say("stopped after %v conversation(s); saving", card["conversations"])
		}
		saved = saveModel(model)
		say("model: %s", saved)
	} else {
		say("nothing was learned (-no-learn): the model is untouched")
	}
	doc := map[string]any{
		"config": cfg, "url": client.BaseURL(), "partner": client.ModelName(), "judge": judge.ModelName(),
		"records": records, "report": card, "interrupted": stop(), "saved": nil,
		"speakers": radixnet.ChatSpeakers[:], "negative": nil,
	}
	if saved != "" {
		doc["saved"] = saved
	}
	if negative != nil {
		path := saveNegative(negative)
		say("negative model: %s", path)
		reasonTable(negative, 10)
		doc["negative"] = map[string]any{
			"path": path, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	if jsonMode {
		emit(doc)
	}
}

// sayChatRecord prints the conversation as it happens: every exchange as it is
// spoken, then how the judge marked the conversation.
func sayChatRecord(record map[string]any) {
	switch record["kind"] {
	case "exchange":
		say("%s: %v", radixnet.ChatSpeakers[0], record["said"])
		say("%s: %v", radixnet.ChatSpeakers[1], record["reply"])
		detail := "    (a fresh line)"
		if context, _ := record["context"].(string); context != "" {
			detail = fmt.Sprintf("    picked up %s", radixnet.PythonRepr(context))
		}
		if vetoed := intOfAny(record["vetoed"]); vetoed > 0 {
			detail += fmt.Sprintf("  [%d vetoed]", vetoed)
		}
		say("%s", detail)
	case "conversation":
		ended := ""
		if stalled, _ := record["stalled"].(string); stalled != "" {
			ended = fmt.Sprintf(" (%s)", stalled)
		}
		say("conversation %v: %v/%v replies failed, mean mark %s/10%s",
			record["conversation"], record["failed"], record["exchanges"],
			fmtMark(record["mean_rating"]), ended)
		say("")
	}
}

// intOfAny reads a count out of a record field however it was stored.
func intOfAny(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}
