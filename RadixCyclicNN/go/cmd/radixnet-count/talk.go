package main

// `radixnet-count talk`: talk to the model in today's format - messages in, a
// reply out, the thinking first and then the text, streamed as they arrive.
// The Go twin of the Python CLI's talk command: one reply per --message (or
// per line typed at the prompt), the conversation carried on from one to the
// next; --request FILE sends a request body written in the dialect's own
// shape; --json prints the dialect's own document, what the server's /v1
// route would return.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// talkBody is a request in the dialect's shape from the talk flags: the conversation so far, then what was just said.
type talkFlags struct {
	system      string
	maxTokens   int
	temperature float64
	stops       []string
	n           int
	tools       []string
	noThinking  bool
	mode        string
	context     int
	k           int
	beam        int
	stepPenalty float64
	explore     int
	allowRep    bool
	allowWord   bool
	noLearn     bool
	seeded      bool
}

func (f *talkFlags) body(dialect string, history []map[string]any, said string) map[string]any {
	messages := append([]map[string]any{}, history...)
	messages = append(messages, map[string]any{"role": "user", "content": said})
	body := map[string]any{
		"max_tokens": f.maxTokens, "temperature": f.temperature, "mode": f.mode, "context": f.context, "k": f.k,
		"step_penalty": f.stepPenalty, "explore": f.explore, "avoid_repeats": !f.allowRep,
		"avoid_word_repeats": !f.allowWord, "learn": !f.noLearn, "guard": !noGuard,
	}
	if f.beam > 0 {
		body["beam"] = f.beam
	}
	if f.seeded {
		body["seed"] = seedFlag
	}
	if dialect == radixnet.FormatAnthropic {
		if f.system != "" {
			body["system"] = f.system
		}
		if len(f.stops) > 0 {
			body["stop_sequences"] = f.stops
		}
		kind := "enabled"
		if f.noThinking {
			kind = "disabled"
		}
		body["thinking"] = map[string]any{"type": kind}
		if len(f.tools) > 0 {
			tools := []map[string]any{}
			for _, name := range f.tools {
				tools = append(tools, map[string]any{"name": name, "input_schema": map[string]any{"type": "object"}})
			}
			body["tools"] = tools
		}
	} else {
		if f.system != "" {
			messages = append([]map[string]any{{"role": "system", "content": f.system}}, messages...)
		}
		if len(f.stops) > 0 {
			body["stop"] = f.stops
		}
		if f.n > 1 {
			body["n"] = f.n
		}
		body["thinking"] = !f.noThinking
		if len(f.tools) > 0 {
			tools := []map[string]any{}
			for _, name := range f.tools {
				tools = append(tools, map[string]any{"type": "function",
					"function": map[string]any{"name": name, "parameters": map[string]any{"type": "object"}}})
			}
			body["tools"] = tools
		}
	}
	body["messages"] = messages
	return body
}

// spokenText is the reply's text, to carry the conversation on with.
func spokenText(doc map[string]any, dialect string) string {
	if dialect == radixnet.FormatAnthropic {
		var out strings.Builder
		if blocks, ok := doc["content"].([]map[string]any); ok {
			for _, b := range blocks {
				if b["type"] == "text" {
					text, _ := b["text"].(string)
					out.WriteString(text)
				}
			}
		}
		return out.String()
	}
	choices, _ := doc["choices"].([]map[string]any)
	if len(choices) == 0 {
		return ""
	}
	message, _ := choices[0]["message"].(map[string]any)
	text, _ := message["content"].(string)
	return text
}

func cmdTalk(args []string) {
	fs := subFlagSet("talk")
	var messages, stops, tools multiFlag
	f := talkFlags{}
	fs.Var(&messages, "message", "what to say (repeat for several turns of one conversation)")
	fs.StringVar(&f.system, "system", "", "a system prompt (accepted, and not read: the network cannot follow instructions)")
	request := fs.String("request", "", "a JSON request body in the --format shape, sent as it is")
	format := fs.String("format", "openai", "the dialect of the answer (and of --request): openai | anthropic")
	fs.IntVar(&f.maxTokens, "max-tokens", radixnet.DefaultMaxTokens, "units (characters, or words on a word model) the reply may add")
	fs.Float64Var(&f.temperature, "temperature", 1.0, "sample: softmax temperature")
	fs.Var(&stops, "stop", "a stop sequence: the reply is cut before it (repeatable)")
	fs.IntVar(&f.n, "n", 1, "openai: how many alternative replies, each unheard by the last")
	fs.Var(&tools, "tool", "offer a tool by name: a <tool> call the reply writes to it comes back as a tool call (repeatable)")
	fs.BoolVar(&f.noThinking, "no-thinking", false, "leave the thinking out of the answer")
	fs.StringVar(&f.mode, "mode", "beam", "how a reply is found: beam | sample")
	fs.IntVar(&f.context, "context", 12, "characters of the last message a reply picks up")
	fs.IntVar(&f.k, "k", 5, "candidates considered per reply")
	fs.IntVar(&f.beam, "beam", 0, "beam width (0 = default)")
	fs.Float64Var(&f.stepPenalty, "step-penalty", 0, "beam: extra cost per edge")
	fs.BoolVar(&f.allowRep, "allow-repeats", false, "do not skip replies the conversation already heard")
	fs.BoolVar(&f.allowWord, "allow-word-repeats", false, "do not skip a reply that repeats its own words")
	fs.IntVar(&f.explore, "explore", radixnet.Explore, "times a reply that caught itself repeating may back up and look for another way on")
	fs.BoolVar(&f.noLearn, "no-learn", false, "do not teach the graph where it goes round")
	fs.BoolVar(&f.seeded, "seeded", false, "sample with a private RNG seeded by --seed")
	saveLearned := fs.Bool("save", false, "write what it learned back to the model file")
	addGuardFlags(fs)
	_ = fs.Parse(args)
	f.stops, f.tools = stops, tools
	dialect := strings.ToLower(strings.TrimSpace(*format))
	if dialect != radixnet.FormatOpenAI && dialect != radixnet.FormatAnthropic {
		fail("--format must be openai or anthropic, got %q", *format)
	}
	m := openModel(true)
	pair := openGuard(m)
	units := m.Encoding().UnitsName()
	taught := map[int]bool{}
	exchanges := []map[string]any{}

	one := func(body map[string]any) map[string]any {
		// through JSON and back, so the request has exactly the shapes the server reads
		raw, err := json.Marshal(body)
		if err != nil {
			fail("bad request: %v", err)
		}
		var normalised map[string]any
		if err := json.Unmarshal(raw, &normalised); err != nil {
			fail("bad request: %v", err)
		}
		ask, err := radixnet.ParseRequest(normalised, dialect)
		if err != nil {
			fail("bad request: %v", err)
		}
		writing := false
		show := func(event map[string]any) {
			if jsonMode {
				return
			}
			kind, _ := event["type"].(string)
			switch kind {
			case "start":
				if index, _ := event["index"].(int); index > 0 {
					fmt.Printf("--- choice %d ---\n", index+1)
				}
				writing = false
			case "thinking":
				text, _ := event["text"].(string)
				fmt.Printf("  · %s\n", strings.TrimLeft(text, "\n"))
			case "text":
				if !writing {
					fmt.Print("model: ")
					writing = true
				}
				text, _ := event["text"].(string)
				fmt.Print(text)
				os.Stdout.Sync()
			case "tool_use":
				if writing {
					fmt.Println()
					writing = false
				}
				input, _ := event["input"].(map[string]any)
				rendered, _ := marshalArguments(input)
				fmt.Printf("tool call: %s %s\n", event["name"], rendered)
			case "done":
				if writing {
					fmt.Println()
				} else if event["turn"] == nil {
					fmt.Println("model: (nothing to say)")
				}
				out, _ := event["output_units"].(int)
				thought, _ := event["thinking_units"].(int)
				fmt.Printf("    [%s; %d %s said, %d thought]\n", event["stop_reason"], out-thought, units, thought)
			}
		}
		var guard *radixnet.Filter
		if ask.Guard {
			guard = pair
		}
		reply, err := radixnet.Respond(m, guard, ask, "", show)
		if err != nil {
			fail("%v", err)
		}
		for _, choice := range reply.Choices {
			if choice.Turn != nil && choice.Turn.Rethink != nil && choice.Turn.Rethink.Taught >= 0 {
				taught[choice.Turn.Rethink.Taught] = true
			}
		}
		doc := radixnet.ToFormat(reply, dialect)
		exchanges = append(exchanges, doc)
		return doc
	}

	history := []map[string]any{}
	var last map[string]any
	switch {
	case *request != "":
		raw, err := os.ReadFile(*request)
		if err != nil {
			fail("cannot read the request %s: %v", *request, err)
		}
		var body any
		if err := json.Unmarshal(raw, &body); err != nil {
			fail("cannot read the request %s: %v", *request, err)
		}
		obj, ok := body.(map[string]any)
		if !ok {
			fail("%s must hold a JSON object: a request in the %s shape", *request, dialect)
		}
		last = one(obj)
	case len(messages) > 0:
		for _, said := range messages {
			say("you: %s", said)
			last = one(f.body(dialect, history, said))
			history = append(history, map[string]any{"role": "user", "content": said},
				map[string]any{"role": "assistant", "content": spokenText(last, dialect)})
		}
	default:
		info, _ := os.Stdin.Stat()
		interactive := info != nil && (info.Mode()&os.ModeCharDevice) != 0 && !jsonMode
		if interactive {
			fmt.Printf("talking to %s (%s format); an empty line or Ctrl-D ends it\n", modelFile(), dialect)
		}
		reader := bufio.NewReader(os.Stdin)
		for {
			if interactive {
				fmt.Print("you> ")
			}
			line, err := reader.ReadString('\n')
			if line == "" && err != nil {
				break
			}
			said := strings.TrimRight(line, "\r\n")
			if strings.TrimSpace(said) == "" {
				break
			}
			if !interactive {
				say("you: %s", said)
			}
			last = one(f.body(dialect, history, said))
			history = append(history, map[string]any{"role": "user", "content": said},
				map[string]any{"role": "assistant", "content": spokenText(last, dialect)})
			if err != nil {
				break
			}
		}
		if last == nil {
			fail("nothing was said: give --message TEXT, --request FILE, or type a line")
		}
	}
	var saved any
	if len(taught) > 0 && *saveLearned {
		path := saveModel(m)
		saved = path
		say("saved %s (it learned to hand over at %d node(s))", path, len(taught))
	} else if len(taught) > 0 {
		say("it learned to hand over at %d node(s); --save writes that into the model", len(taught))
	}
	if !jsonMode {
		return
	}
	if len(exchanges) == 1 {
		if saved != nil {
			last["saved"] = saved
		}
		emit(last)
		return
	}
	nodes := []int{}
	for node := range taught {
		nodes = append(nodes, node)
	}
	sort.Ints(nodes)
	doc := map[string]any{"format": dialect, "exchanges": exchanges, "taught": nodes}
	if saved != nil {
		doc["saved"] = saved
	}
	emit(doc)
}

// marshalArguments renders a call's arguments as Python's json.dumps(sort_keys=True) does.
func marshalArguments(input map[string]any) (string, error) {
	keys := make([]string, 0, len(input))
	for k := range input {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		value, err := json.Marshal(input[k])
		if err != nil {
			return "", err
		}
		name, _ := json.Marshal(k)
		parts = append(parts, string(name)+": "+string(value))
	}
	return "{" + strings.Join(parts, ", ") + "}", nil
}
