package radixnet

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The model in conversation with an LLM that marks it: one conversation end to
// end (the partner opens, the model replies, the judge marks every reply and
// the conversation as a whole), what reaches the negative network and the
// positive one, the guard vetoing a reply before it is spoken, stalling, the
// report card and stopping between conversations.

var chatCorpus = []string{
	"the cat sat on the mat",
	"the dog sat on the log",
	"a bird flew over the hill",
}

// lines that end on words the corpus knows, so the model has something to continue
var partnerLines = []string{"tell me about the cat", "and what about the dog", "now tell me about a bird", "say more"}

var exchangeLine = regexp.MustCompile(`^\[(\d+)\] Partner: (.*)$`)

// fakePartner plays both sides of the LLM: the partner writing lines and the
// judge marking the replies, from the same prompts the real ones would see.
type fakePartner struct {
	prompts  []string
	lines    int
	marked   int
	passes   func(reply string) bool
	lineFor  func(i int) string
	answerAs string
}

func newFakePartner() *fakePartner {
	return &fakePartner{
		passes:  func(reply string) bool { return strings.Contains(reply, "sat") || strings.Contains(reply, "flew") },
		lineFor: func(i int) string { return partnerLines[i%len(partnerLines)] },
	}
}

func (f *fakePartner) Provider() string                  { return ProviderOllama }
func (f *fakePartner) BaseURL() string                   { return "http://fake" }
func (f *fakePartner) ModelName() string                 { return "fake:latest" }
func (f *fakePartner) Available() bool                   { return true }
func (f *fakePartner) Models() ([]map[string]any, error) { return nil, nil }

func (f *fakePartner) Generate(prompt string, o LLMOptions) (string, error) {
	f.prompts = append(f.prompts, prompt)
	switch {
	case strings.Contains(o.System, "marking a conversation"): // the judge
		if f.answerAs != "" {
			return f.answerAs, nil
		}
		type entry struct {
			Index    int     `json:"index"`
			Rating   float64 `json:"rating"`
			Critique string  `json:"critique"`
		}
		reviews := []entry{}
		lines := strings.Split(prompt, "\n")
		for i, line := range lines {
			match := exchangeLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			reply := ""
			if i+1 < len(lines) {
				if _, after, found := strings.Cut(lines[i+1], "Model:"); found {
					reply = strings.TrimSpace(after)
				}
			}
			index, _ := strconv.Atoi(match[1])
			rating, critique := 2.0, "it repeats the same words"
			if f.passes(reply) {
				rating, critique = 9.0, "it follows on"
			}
			reviews = append(reviews, entry{Index: index, Rating: rating, Critique: critique})
		}
		f.marked++
		body, _ := json.Marshal(map[string]any{
			"reviews": reviews,
			"overall": map[string]any{"rating": 7, "critique": "it kept up, mostly"},
		})
		return string(body), nil
	case strings.Contains(o.System, "conversation with a very small"): // the partner
		f.lines++
		return f.lineFor(f.lines - 1), nil
	}
	return "unexpected prompt", nil
}

func chatModel(t *testing.T) *Model {
	t.Helper()
	model, err := NewModel(5, DefaultGraphOptions())
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	model.Exact = true
	if _, err := model.Train(chatCorpus, TrainOptions{Epochs: 4}); err != nil {
		t.Fatalf("train: %v", err)
	}
	return model
}

// newTestChat is one loop over a trained model, not learning unless asked.
func newTestChat(t *testing.T, negative *Model, change func(*ChatConfig)) (*Chat, *fakePartner) {
	t.Helper()
	cfg := DefaultChatConfig()
	cfg.Turns, cfg.Learn = 2, false
	if change != nil {
		change(&cfg)
	}
	fake := newFakePartner()
	loop, err := NewChat(chatModel(t), negative, fake, cfg)
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	return loop, fake
}

func TestChatConfigDefaultsAreValid(t *testing.T) {
	cfg := DefaultChatConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("defaults: %v", err)
	}
	if ChatSpeakers != [2]string{"Partner", "Model"} {
		t.Fatalf("speakers = %v", ChatSpeakers)
	}
}

func TestChatConfigValidation(t *testing.T) {
	for name, change := range map[string]func(*ChatConfig){
		"conversations": func(c *ChatConfig) { c.Conversations = -1 },
		"turns":         func(c *ChatConfig) { c.Turns = 0 },
		"max_length":    func(c *ChatConfig) { c.MaxLength = 0 },
		"mode":          func(c *ChatConfig) { c.Mode = "dijkstra" },
		"k":             func(c *ChatConfig) { c.K = 0 },
		"temperature":   func(c *ChatConfig) { c.Temperature = -1 },
		"threshold":     func(c *ChatConfig) { c.Threshold = 11 },
		"neg_epochs":    func(c *ChatConfig) { c.NegEpochs = -1 },
		"strength":      func(c *ChatConfig) { c.Strength = -1 },
		"provider":      func(c *ChatConfig) { c.Provider = "nobody" },
		"context":       func(c *ChatConfig) { c.Context = -1 },
	} {
		cfg := DefaultChatConfig()
		change(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
}

func TestChatThePartnerOpensAndTheModelReplies(t *testing.T) {
	loop, _ := newTestChat(t, nil, nil)
	held, err := loop.Converse(nil)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	if len(held.Transcript) < 2 {
		t.Fatalf("transcript = %v", held.Transcript)
	}
	if held.Transcript[0].Speaker != "Partner" || held.Transcript[1].Speaker != "Model" {
		t.Fatalf("speakers = %q, %q", held.Transcript[0].Speaker, held.Transcript[1].Speaker)
	}
	if len(held.Exchanges) != 2 {
		t.Fatalf("exchanges = %d, want 2", len(held.Exchanges))
	}
	if held.Transcript[0].Text != partnerLines[0] {
		t.Fatalf("opening = %q", held.Transcript[0].Text)
	}
	for _, exchange := range held.Exchanges {
		if strings.TrimSpace(exchange.Said) == "" || strings.TrimSpace(exchange.Reply) == "" {
			t.Fatalf("empty exchange %+v", exchange)
		}
	}
	if held.Stalled != "" {
		t.Fatalf("stalled = %q", held.Stalled)
	}
}

func TestChatAGivenOpeningIsSpokenAsItIs(t *testing.T) {
	loop, fake := newTestChat(t, nil, func(c *ChatConfig) {
		c.Opening, c.Turns = "the cat sat on the mat", 1
	})
	held, err := loop.Converse(nil)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	if held.Transcript[0] != (Line{Speaker: "Partner", Text: "the cat sat on the mat"}) {
		t.Fatalf("opening = %+v", held.Transcript[0])
	}
	if fake.lines != 0 {
		t.Fatalf("the partner was asked to open %d time(s)", fake.lines)
	}
}

func TestChatTheModelRepliesByContinuingTheLine(t *testing.T) {
	loop, _ := newTestChat(t, nil, func(c *ChatConfig) { c.Opening, c.Turns = "tell me about the cat", 1 })
	held, err := loop.Converse(nil)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	turn := held.Turns[0]
	if turn.Context == "" {
		t.Fatalf("the reply picked up nothing: %+v", turn)
	}
	if !strings.HasPrefix(turn.Text, turn.Context) {
		t.Fatalf("%q does not continue %q", turn.Text, turn.Context)
	}
	if !strings.Contains("tell me about the cat", turn.Context) {
		t.Fatalf("context %q is not the tail of the line", turn.Context)
	}
}

func TestChatAConversationIsMarkedReplyByReply(t *testing.T) {
	loop, fake := newTestChat(t, nil, nil)
	record, err := loop.RunConversation(nil)
	if err != nil {
		t.Fatalf("RunConversation: %v", err)
	}
	if record["kind"] != "conversation" {
		t.Fatalf("kind = %v", record["kind"])
	}
	reviews := record["reviews"].([]Review)
	if len(reviews) != record["exchanges"].(int) {
		t.Fatalf("%d review(s) for %v exchange(s)", len(reviews), record["exchanges"])
	}
	if rating, _ := record["overall_rating"].(*float64); rating == nil || *rating != 7 {
		t.Fatalf("overall_rating = %v", record["overall_rating"])
	}
	if record["overall_critique"] != "it kept up, mostly" {
		t.Fatalf("overall_critique = %v", record["overall_critique"])
	}
	if record["passed"].(int)+record["failed"].(int) != record["exchanges"].(int) {
		t.Fatalf("passed + failed != exchanges: %v", record)
	}
	if mean, _ := record["mean_rating"].(*float64); mean == nil {
		t.Fatalf("mean_rating is nil")
	}
	for _, review := range reviews { // every reply was marked: none came back unrated
		if review.Verdict != "pass" && review.Verdict != "fail" {
			t.Fatalf("verdict = %q", review.Verdict)
		}
		if review.Rating == nil {
			t.Fatalf("review %d is unrated", review.Index)
		}
		if review.Said == "" { // and each carries the line it was answering
			t.Fatalf("review %d has no line to answer", review.Index)
		}
	}
	if fake.marked != 1 { // one judgement per conversation, not one per line
		t.Fatalf("marked %d time(s)", fake.marked)
	}
}

func TestChatAStalledConversationSaysSo(t *testing.T) {
	loop, _ := newTestChat(t, nil, func(c *ChatConfig) {
		c.Opening, c.Turns = "nothing here matches the corpus at all", 2
	})
	empty, err := NewModel(1, DefaultGraphOptions()) // untrained: it has nothing to say at all
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	loop.Model = empty
	held, err := loop.Converse(nil)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	if len(held.Exchanges) != 0 {
		t.Fatalf("exchanges = %v", held.Exchanges)
	}
	if held.Stalled == "" {
		t.Fatalf("a stalled conversation said nothing about it")
	}
}

func TestChatTheFailuresBlameTheNegativeNetwork(t *testing.T) {
	negative, err := NewNegativeModel(2, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	loop, fake := newTestChat(t, negative, func(c *ChatConfig) { c.Turns = 3 })
	fake.passes = func(string) bool { return false } // every reply is a failure
	record, err := loop.RunConversation(nil)
	if err != nil {
		t.Fatalf("RunConversation: %v", err)
	}
	if record["blamed"].(int) == 0 {
		t.Fatalf("nothing was blamed: %v", record)
	}
	if negative.G.Neg.TotalBlame <= 0 {
		t.Fatalf("total blame = %g", negative.G.Neg.TotalBlame)
	}
	if reasons, _ := record["reasons"].(map[string]int); reasons["repetition"] == 0 {
		t.Fatalf("reasons = %v", reasons) // "it repeats the same words"
	}
	recent := negative.Recent(1)
	if len(recent) != 1 || recent[0].Source != "chat" {
		t.Fatalf("recent = %v", recent)
	}
}

func TestChatThePassesClearBlame(t *testing.T) {
	negative, err := NewNegativeModel(2, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	if _, err := negative.Blame([]string{"the cat sat on the mat"}, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatalf("Blame: %v", err)
	}
	before := negative.G.Neg.TotalClear
	loop, fake := newTestChat(t, negative, nil)
	fake.passes = func(string) bool { return true }
	if _, err := loop.RunConversation(nil); err != nil {
		t.Fatalf("RunConversation: %v", err)
	}
	if negative.G.Neg.TotalClear <= before {
		t.Fatalf("total clear %g did not grow past %g", negative.G.Neg.TotalClear, before)
	}
}

func TestChatTheGuardVetoesAReplyBeforeItIsSpoken(t *testing.T) {
	negative, err := NewNegativeModel(2, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	// everything it can say is known-bad
	if _, err := negative.Blame(chatCorpus, BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatalf("Blame: %v", err)
	}
	loop, _ := newTestChat(t, negative, func(c *ChatConfig) { c.Guard = true })
	pair := loop.Pair()
	if pair == nil {
		t.Fatalf("the guard stood aside with a blamed negative network")
	}
	zero := 0.0
	pair.Config.Threshold, pair.Config.MinCoverage = &zero, &zero
	held, err := loop.Converse(nil)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	if held.Vetoed == 0 {
		t.Fatalf("nothing was vetoed: %+v", held)
	}
}

func TestChatTheGuardStandsAsideWithoutANegativeNetwork(t *testing.T) {
	loop, _ := newTestChat(t, nil, nil)
	if loop.Pair() != nil {
		t.Fatalf("a guard appeared out of nothing")
	}
	empty, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	loop.Negative = empty // nothing has ever been blamed, so it would veto nothing
	if loop.Pair() != nil {
		t.Fatalf("an empty negative network guarded anyway")
	}
}

func TestChatLearningTrainsThePositiveModel(t *testing.T) {
	loop, fake := newTestChat(t, nil, func(c *ChatConfig) {
		c.Learn, c.NegEpochs, c.PosEpochs = true, 1, 1
	})
	fake.passes = func(reply string) bool { return strings.Contains(reply, "sat") }
	before := loop.Model.Meta["twonrl_runs"]
	record, err := loop.RunConversation(nil)
	if err != nil {
		t.Fatalf("RunConversation: %v", err)
	}
	action, _ := record["action"].(string)
	switch action {
	case "2nrl":
		if intOf(loop.Model.Meta["twonrl_runs"]) <= intOf(before) {
			t.Fatalf("twonrl_runs did not move: %v", loop.Model.Meta["twonrl_runs"])
		}
	case "reward", "punish":
	default:
		t.Fatalf("action = %q", action)
	}
	if record["good"].(int) == 0 { // the partner's lines are correct data
		t.Fatalf("nothing was rewarded: %v", record)
	}
}

func TestChatThePartnersLinesCanBeKeptOutOfThePositivePhase(t *testing.T) {
	loop, fake := newTestChat(t, nil, func(c *ChatConfig) {
		c.Learn, c.Turns, c.TeachPartner, c.NegEpochs, c.PosEpochs = true, 1, false, 1, 1
	})
	fake.passes = func(string) bool { return false }
	record, err := loop.RunConversation(nil)
	if err != nil {
		t.Fatalf("RunConversation: %v", err)
	}
	if record["good"].(int) != 0 {
		t.Fatalf("good = %v", record["good"])
	}
	if record["action"] != "punish" {
		t.Fatalf("action = %v", record["action"])
	}
}

func TestChatRunHoldsEveryConversationAndReports(t *testing.T) {
	loop, _ := newTestChat(t, nil, func(c *ChatConfig) { c.Conversations = 2 })
	seen := []map[string]any{}
	records, err := loop.Run(func(record map[string]any) { seen = append(seen, record) })
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	held := []map[string]any{}
	for _, record := range records {
		if record["kind"] == "conversation" {
			held = append(held, record)
		}
	}
	card := records[len(records)-1]
	if len(held) != 2 {
		t.Fatalf("%d conversation(s)", len(held))
	}
	if card["kind"] != "report" || card["conversations"] != 2 {
		t.Fatalf("card = %v", card)
	}
	total := 0
	for i, record := range held {
		total += record["exchanges"].(int)
		if record["conversation"] != i+1 {
			t.Fatalf("conversation numbering: %v", record["conversation"])
		}
	}
	if card["exchanges"] != total {
		t.Fatalf("card exchanges = %v, want %d", card["exchanges"], total)
	}
	if mean, _ := card["mean_rating"].(*float64); mean == nil {
		t.Fatalf("mean_rating is nil")
	}
	if rating, _ := card["overall_rating"].(*float64); rating == nil || *rating != 7 {
		t.Fatalf("overall_rating = %v", card["overall_rating"])
	}
	streamed := false
	for _, record := range seen {
		if record["kind"] == "exchange" { // the turns stream as they happen
			streamed = true
		}
	}
	if !streamed {
		t.Fatalf("no exchange reached the progress hook")
	}
}

func TestChatStoppingBetweenConversations(t *testing.T) {
	loop, _ := newTestChat(t, nil, func(c *ChatConfig) { c.Conversations = 2 })
	loop.Stop = func() bool { return true }
	records, err := loop.Run(nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(records) != 1 || records[0]["kind"] != "report" {
		t.Fatalf("records = %v", records)
	}
	if records[0]["conversations"] != 0 {
		t.Fatalf("conversations = %v", records[0]["conversations"])
	}
}

func TestChatReportCardOfNothing(t *testing.T) {
	card := ChatReportCard(nil)
	if card["conversations"] != 0 || card["mean_rating"] != (*float64)(nil) || card["trend"] != nil {
		t.Fatalf("card = %v", card)
	}
}

func TestChatLineAsksForOneShortLine(t *testing.T) {
	fake := newFakePartner()
	line, err := ChatLine(fake, nil, ChatLineOptions{Topic: "animals", Persona: "a curious child"})
	if err != nil {
		t.Fatalf("ChatLine: %v", err)
	}
	if line != partnerLines[0] {
		t.Fatalf("line = %q", line)
	}
	if !strings.Contains(fake.prompts[0], "Open the conversation about animals") {
		t.Fatalf("prompt = %q", fake.prompts[0])
	}
	// and the conversation so far is quoted back speaker by speaker
	if _, err := ChatLine(fake, []Line{{Speaker: "Partner", Text: "hello"}, {Speaker: "Model", Text: "hi"}},
		ChatLineOptions{}); err != nil {
		t.Fatalf("ChatLine: %v", err)
	}
	if !strings.Contains(fake.prompts[1], "Partner: hello\nModel: hi") {
		t.Fatalf("prompt = %q", fake.prompts[1])
	}
}

func TestChatFirstLineTakesOneLineOutOfAnAnswer(t *testing.T) {
	for raw, want := range map[string]string{
		"":                             "",
		"  \n\nthe cat sat":            "the cat sat",
		"Model: the cat sat":           "the cat sat",
		"\"the cat sat\"":              "the cat sat",
		"the cat sat\nand the dog too": "the cat sat",
		"Partner:   the  cat   sat  ":  "the cat sat",
	} {
		if got := firstLine(raw); got != want {
			t.Fatalf("firstLine(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestChatReviewConversationMarksEveryReply(t *testing.T) {
	fake := newFakePartner()
	result, err := ReviewConversation(fake, []Exchange{
		{Said: "tell me about the cat", Reply: "the cat sat on the mat"},
		{Said: "and the dog", Reply: "xxxx"},
		{Said: "say more", Reply: "   "},
	}, "animals", "", 6)
	if err != nil {
		t.Fatalf("ReviewConversation: %v", err)
	}
	if len(result.Reviews) != 3 {
		t.Fatalf("%d review(s)", len(result.Reviews))
	}
	if result.Reviews[0].Verdict != "pass" || result.Reviews[1].Verdict != "fail" {
		t.Fatalf("verdicts = %q, %q", result.Reviews[0].Verdict, result.Reviews[1].Verdict)
	}
	if result.Reviews[2].Critique != "it said nothing" { // an empty reply is a failure, unasked
		t.Fatalf("empty reply: %+v", result.Reviews[2])
	}
	if result.Reviews[0].Said != "tell me about the cat" {
		t.Fatalf("said = %q", result.Reviews[0].Said)
	}
	if result.Model != "fake:latest" { // the judge falls back to the client's own model
		t.Fatalf("model = %q", result.Model)
	}
	if result.Overall == nil || result.Overall.Rating == nil || *result.Overall.Rating != 7 {
		t.Fatalf("overall = %+v", result.Overall)
	}
	if !strings.Contains(fake.prompts[0], "Topic: animals") {
		t.Fatalf("prompt = %q", fake.prompts[0])
	}
}

func TestChatReviewConversationWithoutAnOverall(t *testing.T) {
	fake := newFakePartner()
	fake.answerAs = `{"reviews": [{"index": 0, "rating": 8, "critique": "fine"}]}`
	result, err := ReviewConversation(fake, []Exchange{{Said: "hello", Reply: "hello there"}}, "", "judge:1", 6)
	if err != nil {
		t.Fatalf("ReviewConversation: %v", err)
	}
	if result.Overall != nil {
		t.Fatalf("overall = %+v", result.Overall)
	}
	if result.Model != "judge:1" {
		t.Fatalf("model = %q", result.Model)
	}
}
