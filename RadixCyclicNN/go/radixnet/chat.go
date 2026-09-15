package radixnet

import (
	"fmt"
	"strings"
	"time"
)

// The model in conversation with an LLM, and the LLM marking the conversation.
//
// Every other teacher in this project talks *at* the network: the English
// tutor writes a prefix and marks the completion (tutor.go), the critic
// reviews texts the model wrote alone (critic.go), the agent judges a
// transcript of tool calls (agent.go).  The one thing a language model is
// actually for - holding up its end of a conversation - was only ever tested
// against *itself* (dialogue.go, two voices of the same network), where
// nothing can tell it that its answer did not follow on.
//
// This is the Go twin of the Python `radixnet/chat.py`.  One conversation is:
//
//  1. the partner - a local Ollama model by default, ChatGPT when the provider
//     says so - says a line, told to keep it short, plain and easy to carry on
//     from, because that is what a character-level model can reply to
//     (ChatLine);
//  2. the model replies to it the way it replies to anything: the tail of that
//     line is located in the graph and continued (Model.Reply), so a reply is a
//     real walk of the network and not a prompt trick.  With a negative network
//     in hand the pair vetoes a reply before it is spoken (duo.go);
//  3. they take turns for Turns exchanges;
//  4. the judge then marks every reply the model gave out of 10 against the
//     line it was replying to, and the conversation as a whole
//     (ReviewConversation).
//
// What the marks buy: the failures blame the negative network and the passes
// clear it (TeachReviews), and - unlike the critic - this loop trains the
// positive model too: the replies that passed are rewarded, the ones that
// failed are punished, and the partner's own lines join the positive phase
// (TeachPartner) because they are exactly what a good reply in this
// conversation would have looked like.
//
// That last point is the whole idea.  A model that only ever learns from a
// corpus has no way of finding out that what it said did not answer the
// question; here something answers back, and says so.

// ChatConfig is how the conversation is held, marked and learned from.
type ChatConfig struct {
	// Conversations to hold; 0 means keep going until something stops it.
	Conversations int `json:"conversations"`
	// Turns is the replies the model gives per conversation (the partner speaks between them).
	Turns int `json:"turns"`
	// Topic is what to talk about; empty lets the partner choose.
	Topic string `json:"topic"`
	// Opening is the first line, given; empty lets the partner open.
	Opening string `json:"opening"`
	// Persona is who the partner is being ("a curious child", "a vet", ...).
	Persona string `json:"persona"`
	// Context is the characters of the previous line a reply picks up (a word boundary is respected).
	Context int `json:"context"`
	// MaxLength is the characters the model may add to a context in one reply.
	MaxLength int `json:"max_length"`
	// Mode is how a reply is found: "beam" (the most likely unheard one) or "sample".
	Mode string `json:"mode"`
	// K is the candidates considered per reply.
	K int `json:"k"`
	// Temperature is the sampling temperature of the model's replies.
	Temperature float64 `json:"temperature"`
	// PartnerTemperature is the sampling temperature of the partner's lines.
	PartnerTemperature float64 `json:"partner_temperature"`
	// Threshold is the pass mark out of 10: below it a reply is a failure.
	Threshold float64 `json:"threshold"`
	Provider  string  `json:"provider"`
	// PartnerModel is the partner's model name; empty means the client's own default.
	PartnerModel string `json:"partner_model"`
	// JudgeModel is a different model for marking; empty means the partner's.
	JudgeModel string `json:"judge_model"`
	// Guard lets the negative network veto a reply before it is spoken (duo.go).
	Guard bool `json:"guard"`
	// Blame blames the failed replies (and clears with the passed ones).
	Blame       bool `json:"blame"`
	ClearPasses bool `json:"clear_passes"`
	// Learn trains the positive model on the marked replies (2NRL), and lets its rethinks teach the graph
	// where it goes round (TeachBack).
	Learn bool `json:"learn"`
	// TeachPartner puts the partner's own lines in the positive phase: they are what a good reply looked like.
	TeachPartner bool `json:"teach_partner"`
	// AvoidRepeats keeps the model from saying something the conversation has already heard.
	AvoidRepeats bool `json:"avoid_repeats"`
	// AvoidWordRepeats keeps one reply from repeating its own words (a stutter: "say morning morning").
	AvoidWordRepeats bool `json:"avoid_word_repeats"`
	// Explore is how many times a reply that caught itself repeating may back up and look for another way on.
	Explore   int     `json:"explore"`
	NegEpochs int     `json:"neg_epochs"`
	PosEpochs int     `json:"pos_epochs"`
	Strength  float64 `json:"strength"`
	// Epochs of blaming per conversation.
	Epochs int `json:"epochs"`
	// Seed of the first conversation's sampling; later ones advance it, so they differ.
	Seed *int64 `json:"seed"`
}

// DefaultChatConfig mirrors the Python defaults.
func DefaultChatConfig() ChatConfig {
	return ChatConfig{
		Conversations: 1, Turns: 4, Context: 12, MaxLength: 60, Mode: "beam", K: 5, Temperature: 1,
		PartnerTemperature: 0.8, Threshold: 6, Provider: ProviderOllama, Guard: true, Blame: true,
		ClearPasses: true, Learn: true, TeachPartner: true, AvoidRepeats: true, AvoidWordRepeats: true,
		Explore: Explore, NegEpochs: 2, PosEpochs: 3,
		Strength: 1, Epochs: 1,
	}
}

// Validate checks the configuration.
func (c *ChatConfig) Validate() error {
	if c.Conversations < 0 {
		return fmt.Errorf("conversations must be >= 0 (0 = until stopped)")
	}
	if c.Turns < 1 {
		return fmt.Errorf("turns must be >= 1")
	}
	if c.Context < 0 {
		return fmt.Errorf("context must be >= 0")
	}
	if c.MaxLength < 1 {
		return fmt.Errorf("max_length must be >= 1")
	}
	if c.Mode != "beam" && c.Mode != "sample" {
		return fmt.Errorf("mode must be 'beam' or 'sample'")
	}
	if c.K < 1 {
		return fmt.Errorf("k must be >= 1")
	}
	if c.Temperature < 0 || c.PartnerTemperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.Threshold < 0 || c.Threshold > 10 {
		return fmt.Errorf("threshold must lie in [0, 10]")
	}
	if c.NegEpochs < 0 || c.PosEpochs < 0 || c.Epochs < 0 {
		return fmt.Errorf("epochs must be >= 0")
	}
	if c.Strength < 0 {
		return fmt.Errorf("strength must be >= 0")
	}
	provider, err := NormaliseProvider(c.Provider)
	if err != nil {
		return fmt.Errorf("provider must be one of: %s", strings.Join(Providers(), ", "))
	}
	c.Provider = provider
	return nil
}

// Chat is the conversation loop: the partner talks, the model replies, the
// judge marks, both networks learn.
//
// Model is the network doing the talking, Client any LLMClient (Ollama or
// ChatGPT) playing the partner, JudgeClient the one that marks (Client when it
// is nil), and Negative an optional negative network that learns what went
// wrong - and, with Config.Guard, vetoes a bad reply before it is ever spoken.
// External wraps the slow LLM calls so a server can release its model lock
// around them; the model's own replies are walks of the graph and are made
// with the lock held.
type Chat struct {
	Model       *Model
	Negative    *Model
	Client      LLMClient
	JudgeClient LLMClient
	Config      ChatConfig
	// External wraps a call that must happen without the model lock held.
	External func(func() error) error
	// Stop is polled before every reply and every conversation; a true answer ends the run cleanly.
	Stop func() bool

	History []map[string]any
	Number  int

	pair   *Filter
	paired bool
}

// NewChat validates the configuration and returns the loop.
func NewChat(model, negative *Model, client LLMClient, config ChatConfig) (*Chat, error) {
	if model == nil {
		return nil, fmt.Errorf("a model to converse with is required")
	}
	if client == nil {
		return nil, fmt.Errorf("an LLM client to converse with is required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Chat{Model: model, Negative: negative, Client: client, Config: config}, nil
}

// judge is the client that marks: the judge's own, or the partner's.
func (c *Chat) judge() LLMClient {
	if c.JudgeClient != nil {
		return c.JudgeClient
	}
	return c.Client
}

// external runs fn, through External when one is set.
func (c *Chat) external(fn func() error) error {
	if c.External == nil {
		return fn()
	}
	return c.External(fn)
}

func (c *Chat) stopped() bool { return c.Stop != nil && c.Stop() }

// -- the pair -----------------------------------------------------------------

// Pair is the negative network guarding this conversation, or nil when there
// is nothing to guard with.
func (c *Chat) Pair() *Filter {
	if !c.Config.Guard || c.Negative == nil {
		return nil
	}
	if !c.paired {
		c.paired = true
		if pair, err := NewFilter(c.Model, c.Negative, DefaultFilterConfig()); err == nil {
			c.pair = pair
		}
	}
	if c.pair == nil || !c.pair.Ready() {
		return nil
	}
	return c.pair
}

// chatVeto is the pair's verdict as a conversational veto, counting the
// distinct replies it stopped.
//
// A candidate can be offered again after the context is shortened, so the
// judgements are remembered rather than repeated, and Refused counts texts
// rather than refusals.
type chatVeto struct {
	pair *Filter
	seen map[string]bool
}

func (v *chatVeto) refuse(text string) bool {
	if decided, ok := v.seen[text]; ok {
		return decided
	}
	rejected := v.pair.Judge(text).Decision == "reject"
	v.seen[text] = rejected
	return rejected
}

// Refused is how many distinct texts the guard would not let through.
func (v *chatVeto) Refused() int {
	count := 0
	for _, rejected := range v.seen {
		if rejected {
			count++
		}
	}
	return count
}

// veto is what the model may not say, or nil when nothing is guarding.
func (c *Chat) veto() *chatVeto {
	pair := c.Pair()
	if pair == nil {
		return nil
	}
	return &chatVeto{pair: pair, seen: map[string]bool{}}
}

// -- one conversation ---------------------------------------------------------

// ChatHeld is one conversation as it was held: every line, the (said to it, its
// reply) pairs the judge marks, and why it ended early when it did.
type ChatHeld struct {
	Transcript []Line     `json:"transcript"`
	Exchanges  []Exchange `json:"exchanges"`
	Turns      []*Turn    `json:"turns"`
	// Repeats are the replies the model could only repeat (Repeats): punished
	// with the failures, whatever the judge made of them.
	Repeats []string `json:"repeats"`
	// Stalled is why it ended early - the model had nothing left to say, or
	// the partner went quiet - and empty when it ran its course.
	Stalled string `json:"stalled"`
	Vetoed  int    `json:"vetoed"`
}

// partnerLine is the LLM's next line; only the partner's thinking happens
// outside the model lock.
func (c *Chat) partnerLine(transcript []Line) (string, error) {
	cfg := c.Config
	line := ""
	err := c.external(func() error {
		out, err := ChatLine(c.Client, transcript, ChatLineOptions{
			Topic: cfg.Topic, Persona: cfg.Persona, Model: cfg.PartnerModel,
			Temperature: cfg.PartnerTemperature,
		})
		line = out
		return err
	})
	return line, err
}

// Converse holds one conversation.
func (c *Chat) Converse(progress func(map[string]any)) (*ChatHeld, error) {
	cfg := c.Config
	var rng *MT19937
	if cfg.Mode == "sample" && cfg.Seed != nil {
		rng = NewMT19937(*cfg.Seed + int64(c.Number))
	}
	veto := c.veto()
	var vetoFn func(string) bool
	if veto != nil {
		vetoFn = veto.refuse
	}
	held := &ChatHeld{Transcript: []Line{}, Exchanges: []Exchange{}, Turns: []*Turn{}, Repeats: []string{}}
	heard := NewHeard(nil) // what has been said, on both sides: the model does not repeat any of it

	line := strings.TrimSpace(cfg.Opening)
	if line == "" {
		opened, err := c.partnerLine(held.Transcript)
		if err != nil {
			return nil, err
		}
		line = opened
	}
	if line == "" {
		held.Stalled = "the partner said nothing"
		return held, nil
	}
	held.Transcript = append(held.Transcript, Line{Speaker: ChatSpeakers[0], Text: line})
	heard.Remember(line, "")
	for index := 0; index < cfg.Turns; index++ {
		if c.stopped() {
			held.Stalled = "stopped"
			break
		}
		turn, err := c.Model.Reply(line, ReplyOptions{
			Heard: heard, Index: len(held.Transcript), Speaker: ChatSpeakers[1], Mode: cfg.Mode,
			MaxLength: cfg.MaxLength, Context: cfg.Context, Temperature: cfg.Temperature, K: cfg.K,
			RNG: rng, AvoidRepeats: cfg.AvoidRepeats, AvoidWordRepeats: cfg.AvoidWordRepeats,
			Explore: cfg.Explore, Learn: cfg.Learn, Veto: vetoFn,
		})
		if err != nil {
			return nil, err
		}
		if turn == nil {
			// the guard can silence it outright, and that is worth saying
			held.Stalled = "the model had nothing to say"
			if veto != nil && veto.Refused() > 0 {
				held.Stalled = "the guard vetoed everything it could say"
			}
			break
		}
		held.Turns = append(held.Turns, turn)
		held.Transcript = append(held.Transcript, Line{Speaker: ChatSpeakers[1], Text: turn.Text})
		held.Exchanges = append(held.Exchanges, Exchange{Said: line, Reply: turn.Text})
		reply := ""
		if turn.Context != "" {
			reply = turn.Reply
		}
		heard.Remember(turn.Text, reply)
		if progress != nil {
			progress(map[string]any{
				"kind": "exchange", "conversation": c.Number, "exchange": index + 1,
				"said": line, "reply": turn.Text, "context": turn.Context, "fresh": turn.Fresh,
				"repeat": turn.Repeat, "stutter": turn.Stutter, "rethink": turn.Rethink, "vetoed": turn.Vetoed,
				"cost":        turn.Cost,
				"probability": turn.Probability,
			})
		}
		if index == cfg.Turns-1 {
			break // the last word is the model's: no line after it to reply to
		}
		next, err := c.partnerLine(held.Transcript)
		if err != nil {
			return nil, err
		}
		if next == "" {
			held.Stalled = "the partner went quiet"
			break
		}
		line = next
		held.Transcript = append(held.Transcript, Line{Speaker: ChatSpeakers[0], Text: line})
		heard.Remember(line, "")
	}
	held.Repeats = Repeats(held.Turns)
	if veto != nil {
		held.Vetoed = veto.Refused()
	}
	return held, nil
}

// RunConversation holds one conversation, marks it and learns from it,
// returning its record.
func (c *Chat) RunConversation(progress func(map[string]any)) (map[string]any, error) {
	cfg := c.Config
	c.Number++
	started := time.Now()
	held, err := c.Converse(progress)
	if err != nil {
		return nil, err
	}
	judgeModel := cfg.JudgeModel
	if judgeModel == "" {
		judgeModel = cfg.PartnerModel
	}
	review := &ReviewResult{
		Source: "chat", Model: judgeModel, Threshold: cfg.Threshold,
		Texts: []string{}, Reviews: []Review{}, Good: []string{}, Bad: []string{},
	}
	if len(held.Exchanges) > 0 {
		// the judge's thinking, also outside the lock
		if err := c.external(func() error {
			out, err := ReviewConversation(c.judge(), held.Exchanges, cfg.Topic, judgeModel, cfg.Threshold)
			if err != nil {
				return err
			}
			review = out
			return nil
		}); err != nil {
			return nil, err
		}
	}
	taught := &TeachReport{Reasons: map[string]int{}}
	if c.Negative != nil && cfg.Blame && len(review.Reviews) > 0 {
		taught, err = TeachReviews(c.Negative, review.Reviews, cfg.Threshold, cfg.ClearPasses, "chat",
			TeachOptions{Epochs: cfg.Epochs, Stop: c.Stop})
		if err != nil {
			return nil, err
		}
	}
	// the learning keys are always present, whether or not it learned, so a record has one shape
	learned := map[string]any{"action": nil, "bad": 0, "good": 0, "neg_loss": nil, "pos_loss": nil}
	if cfg.Learn {
		learned = c.Learn(review, held)
	}
	var overallRating *float64
	overallCritique := ""
	if review.Overall != nil {
		overallRating, overallCritique = review.Overall.Rating, review.Overall.Critique
	}
	// nil rather than "" when nobody named a model and there was nothing to mark,
	// so the record reads the same on both sides
	var judge any
	if review.Model != "" {
		judge = review.Model
	}
	lines := make([]map[string]any, 0, len(held.Transcript))
	for _, line := range held.Transcript {
		lines = append(lines, map[string]any{"speaker": line.Speaker, "text": line.Text})
	}
	record := map[string]any{
		"kind":             "conversation",
		"conversation":     c.Number,
		"topic":            cfg.Topic,
		"judge":            judge,
		"threshold":        cfg.Threshold,
		"transcript":       lines,
		"exchanges":        len(held.Exchanges),
		"reviews":          review.Reviews,
		"mean_rating":      review.MeanRating,
		"pass_rate":        review.PassRate,
		"passed":           len(review.Good),
		"failed":           len(review.Bad),
		"overall_rating":   overallRating,
		"overall_critique": overallCritique,
		"stalled":          held.Stalled,
		"vetoed":           held.Vetoed,
		"repeats":          len(held.Repeats), // replies the model could only repeat, punished with the failures
		"blamed":           taught.Blamed,
		"cleared":          taught.Cleared,
		"edges":            taught.Edges,
		"reasons":          taught.Reasons,
		"seconds":          time.Since(started).Seconds(),
	}
	for key, value := range learned {
		record[key] = value
	}
	c.History = append(c.History, record)
	return record, nil
}

// -- learning -----------------------------------------------------------------

// Learn applies 2NRL to the marked replies: what failed is punished, what
// passed (and the partner's own lines) rewarded.  A reply the model could only
// repeat is punished whatever the judge made of it: it said nothing new, and
// rewarding it would only make the duplicate likelier next time.
func (c *Chat) Learn(review *ReviewResult, held *ChatHeld) map[string]any {
	cfg := c.Config
	bad := []string{}
	for _, text := range review.Bad {
		if strings.TrimSpace(text) != "" {
			bad = append(bad, text)
		}
	}
	good := []string{}
	for _, text := range review.Good {
		if strings.TrimSpace(text) != "" {
			good = append(good, text)
		}
	}
	if cfg.TeachPartner {
		// the partner's own lines are what a good reply in this conversation would have looked like
		for _, line := range held.Transcript {
			if line.Speaker == ChatSpeakers[0] && strings.TrimSpace(line.Text) != "" {
				good = append(good, line.Text)
			}
		}
	}
	saidTwice := []string{}
	repeated := map[string]bool{}
	for _, text := range held.Repeats {
		if strings.TrimSpace(text) != "" {
			saidTwice = append(saidTwice, text)
			repeated[text] = true
		}
	}
	kept := []string{}
	for _, text := range uniqueStrings(good) {
		if !repeated[text] { // a duplicate is never rewarded, however well it was marked
			kept = append(kept, text)
		}
	}
	good = kept
	known := map[string]bool{}
	for _, text := range good {
		known[text] = true
	}
	kept = []string{}
	for _, text := range uniqueStrings(append(bad, saidTwice...)) {
		if !known[text] {
			kept = append(kept, text)
		}
	}
	bad = kept
	result := map[string]any{"action": nil, "bad": len(bad), "good": len(good), "repeats": len(saidTwice),
		"neg_loss": nil, "pos_loss": nil}
	switch {
	case len(bad) == 0 && len(good) == 0:
		return result
	case len(bad) > 0 && len(good) > 0:
		outcome, err := c.Model.TwoNRLWeighted(bad, nil, good, nil, TwoNRLOptions{
			NegEpochs: cfg.NegEpochs, PosEpochs: cfg.PosEpochs, Strength: cfg.Strength, Stop: c.Stop,
		})
		if err == nil {
			result["action"] = "2nrl"
			result["neg_loss"], result["pos_loss"] = lastLoss(outcome.Negative), lastLoss(outcome.Positive)
		}
	case len(good) > 0:
		records, err := c.Model.Reward(good, cfg.PosEpochs, cfg.Strength)
		if err == nil {
			result["action"], result["pos_loss"] = "reward", lastLoss(records)
		}
	default:
		records, err := c.Model.Punish(bad, cfg.NegEpochs, cfg.Strength)
		if err == nil {
			result["action"], result["neg_loss"] = "punish", lastLoss(records)
		}
	}
	return result
}

// -- the loop -----------------------------------------------------------------

// Run holds Config.Conversations conversations, or keeps going until Stop says
// otherwise when it is 0.  The stop is checked before every conversation, so a
// loop left running ends cleanly after the one it is in.  The records are the
// conversation records, followed by one report record.
func (c *Chat) Run(progress func(map[string]any)) ([]map[string]any, error) {
	limit := c.Config.Conversations
	records := []map[string]any{}
	for limit == 0 || len(records) < limit {
		if c.stopped() {
			break
		}
		record, err := c.RunConversation(progress)
		if err != nil {
			return records, err
		}
		records = append(records, record)
		if progress != nil {
			progress(record)
		}
	}
	card := ChatReportCard(records)
	records = append(records, card)
	if progress != nil {
		progress(card)
	}
	return records, nil
}

// ChatReportCard is what a run of conversations came to: how well the model
// held them up, and what it was taught.
//
// MeanRating averages the per-reply marks over the conversations that produced
// one, OverallRating the judge's verdict on each conversation as a whole, and
// Trend is the last conversation's mean minus the first's - positive when the
// model is answering better than it did at the start.
func ChatReportCard(records []map[string]any) map[string]any {
	held := make([]map[string]any, 0, len(records))
	for _, record := range records {
		if record["kind"] == "conversation" {
			held = append(held, record)
		}
	}
	reasons := map[string]int{}
	exchanges, passed, failed, vetoed, stalled, blamed, cleared, edges := 0, 0, 0, 0, 0, 0, 0, 0
	ratings, rates, overall := []float64{}, []float64{}, []float64{}
	for _, record := range held {
		exchanges += intOf(record["exchanges"])
		passed += intOf(record["passed"])
		failed += intOf(record["failed"])
		vetoed += intOf(record["vetoed"])
		blamed += intOf(record["blamed"])
		cleared += intOf(record["cleared"])
		edges += intOf(record["edges"])
		if text, ok := record["stalled"].(string); ok && text != "" {
			stalled++
		}
		if counts, ok := record["reasons"].(map[string]int); ok {
			for reason, count := range counts {
				reasons[reason] += count
			}
		}
		if value, ok := record["mean_rating"].(*float64); ok && value != nil {
			ratings = append(ratings, *value)
		}
		if value, ok := record["pass_rate"].(*float64); ok && value != nil {
			rates = append(rates, *value)
		}
		if value, ok := record["overall_rating"].(*float64); ok && value != nil {
			overall = append(overall, *value)
		}
	}
	card := map[string]any{
		"kind": "report", "conversations": len(held), "exchanges": exchanges, "passed": passed,
		"failed": failed, "vetoed": vetoed, "stalled": stalled, "blamed": blamed, "cleared": cleared,
		"edges": edges, "reasons": reasons, "mean_rating": meanOrNil(ratings), "pass_rate": meanOrNil(rates),
		"overall_rating": meanOrNil(overall), "trend": nil,
	}
	if len(ratings) > 1 {
		card["trend"] = ratings[len(ratings)-1] - ratings[0]
	}
	return card
}
