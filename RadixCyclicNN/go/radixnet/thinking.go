package radixnet

// Thinking - the fourth sentinel at work: what makes the model think, what it
// thinks, and what happens when it stops.
//
// The graph's Think sentinel faces both ways.  Its in-edges are where the model
// has learned to stop and think: an edge p -> Think is taught by experience,
// the way Back's are, whenever an *event* at p called for a thought.  Its
// out-edges are how thoughts begin: a thought is a text whose walk starts at
// Think instead of Start, observed from an LLM's thinking the way texts are
// observed from a corpus (Model.ThinkOn, fed by `radixnet-count ollama think` -
// ThoughtsFromPrompt).
//
// Model.Think is one thought.  It is triggered by an event - a voice that
// caught itself repeating (Model.Backtrack), a question asked of the model
// (`radixnet-count think -about ...`), a thought questioning itself - and it
// does four things, in order:
//
//  1. it teaches where it had to think: ObserveThink(p) on the node the event
//     happened at (unless the thought is one the model asked itself);
//  2. it thinks: the prediction search run from Think to the end of a text, so
//     the thought is in the language of the thoughts it was taught (nothing,
//     when it was taught none - a model cannot think in words it has not learned);
//  3. it may question itself: along the thought's own path, at a node where the
//     model has learned to think (Graph.ThinksAt), a nested thought is
//     triggered - one that must say something the chain of thoughts above it
//     has not - up to MaxDepth deep and MaxQuestions per thought;
//  4. when it stops, it triggers the sentinel the event calls for: a thought
//     that was thinking its way out of a repeat hands over to Back (ObserveBack
//     on the node, with the step it was about to loop through and the one it
//     took instead - the rethink's lesson, now taught *after* thinking rather
//     than instead of it); a thought the model asked itself returns to the
//     thought that asked ("think"); a thought that was merely asked for ends.
//
// Every thought is a Thought record - the trigger, the node, the text, its
// questions, how it stopped and what it triggered - and rides on the turn's
// Rethink as Thought when a conversation thought.  See the Python
// implementation's radixnet.thinking.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The triggers of a thought: what happened.
const (
	// TriggerAsked is a thought somebody asked for (`radixnet-count think`, POST /api/think).
	TriggerAsked = "asked"
	// TriggerQuestioned is a thought a thought asked itself.
	TriggerQuestioned = "questioned"
	// TriggerStutter and TriggerRepeat are a thought a voice had on catching itself repeating (Rethink.Kind).
	TriggerStutter = "stutter"
	TriggerRepeat  = "repeat"
)

// What a thought triggers when it stops: the Back sentinel, the end of it, or the thought it questioned.
const (
	ThenBack  = "back"
	ThenEnd   = "end"
	ThenThink = "think"
)

// How a thought stopped: it reached End, it ran out of length, or it had nothing (new) to think.
const (
	StoppedEnd     = "end"
	StoppedLength  = "length"
	StoppedNothing = "nothing"
)

const (
	// ThinkLength is the units a thought may run to by default.
	ThinkLength = 60
	// ThinkDepth is how deep a thought may question itself by default: a thought, and a question about it,
	// and one about that.
	ThinkDepth = 2
	// ThinkQuestions is the questions one thought may ask itself by default - one, so a thought cannot spend
	// itself questioning.
	ThinkQuestions = 1
)

// BacksUp reports whether a thought with this trigger hands over to Back when it
// stops: it was thinking its way out of a repeat.
func BacksUp(trigger string) bool { return trigger == TriggerStutter || trigger == TriggerRepeat }

// Thought is one thought: what triggered it, what it thought, what it questioned
// and what it triggered when it stopped.
type Thought struct {
	Trigger string `json:"trigger"`
	// At is the node it thought at (-1: nowhere in particular).
	At int `json:"at"`
	// About is the text it was thinking about, when there was one.
	About string `json:"about"`
	// Text is the thought itself ("" when it had nothing to think with).
	Text string `json:"text"`
	// Depth is 0 for a thought, 1 for a question it asked itself, 2 for a question about that.
	Depth   int    `json:"depth"`
	Stopped string `json:"stopped"`
	// Then is the sentinel it triggered when it stopped.
	Then string `json:"then"`
	// Taught is the node taught to think here (p -> Think), or -1.
	Taught int `json:"taught"`
	// HandedOver is the node taught to hand over when it stopped (p -> Back), or -1.
	HandedOver  int     `json:"handed_over"`
	Cost        float64 `json:"cost"`
	Probability float64 `json:"probability"`
	// Expanded is the search states the thought weighed.
	Expanded  int        `json:"expanded"`
	Questions []*Thought `json:"questions"`
	Labels    []string   `json:"labels"`
	NodeIDs   []int      `json:"node_ids"`
	StepCosts []float64  `json:"step_costs"`
}

func newThought(trigger string, at int, about string, depth int) *Thought {
	return &Thought{
		Trigger: trigger, At: at, About: about, Depth: depth, Stopped: StoppedNothing, Then: ThenEnd,
		Taught: -1, HandedOver: -1, Probability: 1,
		Questions: []*Thought{}, Labels: []string{}, NodeIDs: []int{}, StepCosts: []float64{},
	}
}

// Questioned is how many times the thought questioned itself.
func (t *Thought) Questioned() int { return len(t.Questions) }

// MarshalJSON writes the Python record: every field, "questioned" beside the questions, and empty lists
// rather than nulls.
func (t *Thought) MarshalJSON() ([]byte, error) {
	type plain Thought
	c := *t
	if c.Questions == nil {
		c.Questions = []*Thought{}
	}
	if c.Labels == nil {
		c.Labels = []string{}
	}
	if c.NodeIDs == nil {
		c.NodeIDs = []int{}
	}
	if c.StepCosts == nil {
		c.StepCosts = []float64{}
	}
	p := plain(c)
	return json.Marshal(struct {
		*plain
		Questioned int `json:"questioned"`
	}{&p, len(c.Questions)})
}

// ToDict is the thought as the API and the CLI emit it (the JSON record as a map).
func (t *Thought) ToDict() map[string]any {
	questions := make([]map[string]any, 0, len(t.Questions))
	for _, q := range t.Questions {
		questions = append(questions, q.ToDict())
	}
	labels, nodeIDs, stepCosts := t.Labels, t.NodeIDs, t.StepCosts
	if labels == nil {
		labels = []string{}
	}
	if nodeIDs == nil {
		nodeIDs = []int{}
	}
	if stepCosts == nil {
		stepCosts = []float64{}
	}
	return map[string]any{
		"trigger": t.Trigger, "at": t.At, "about": t.About, "text": t.Text, "depth": t.Depth,
		"stopped": t.Stopped, "then": t.Then, "taught": t.Taught, "handed_over": t.HandedOver,
		"cost": t.Cost, "probability": t.Probability, "expanded": t.Expanded,
		"questioned": len(t.Questions), "questions": questions,
		"labels": labels, "node_ids": nodeIDs, "step_costs": stepCosts,
	}
}

// Question is a sentence of a thought that ends in a question mark, and the byte offset it starts at.
type Question struct {
	Start int
	Text  string
}

const (
	thoughtTerminators = ".!?"
	thoughtSpaces      = " \t\n\r"
)

// QuestionsIn is the sentences of text that end in a question mark, each with
// the index it starts at.
//
// A sentence runs from the first character after the previous sentence's
// terminators (".", "!", "?") to the end of its own; a run of terminators
// containing "?" makes it a question.  "Hmm. Is that right? Yes." holds one, at
// index 5.  This is how a thought is found to have questioned itself, in the
// thinking an LLM wrote as much as in the model's own (Model.ThinkOn).
func QuestionsIn(text string) []Question {
	out := []Question{}
	n := len(text)
	i := 0
	for i < n {
		for i < n && strings.IndexByte(thoughtSpaces, text[i]) >= 0 {
			i++
		}
		if i >= n {
			break
		}
		start := i
		for i < n && strings.IndexByte(thoughtTerminators, text[i]) < 0 {
			i++
		}
		j := i
		for j < n && strings.IndexByte(thoughtTerminators, text[j]) >= 0 {
			j++
		}
		if i > start && strings.IndexByte(text[i:j], '?') >= 0 {
			out = append(out, Question{Start: start, Text: strings.TrimSpace(text[start:j])})
		}
		i = j
	}
	return out
}

// Place is the node the walk of prefix ends at, cut so that it ends there, or
// -1 when the graph cannot place it.
//
// The last gram of prefix is located; when it sits inside a compressed node the
// node is split after it, so that an edge taught from the node (ObserveThink)
// fires exactly where the prefix ends and not at the end of whatever the node
// went on to say.  A prefix the graph does not know whole - a guessed partial
// gram, a text it never saw - has no node to return, and -1 says so.
func (m *Model) Place(prefix string) int {
	g := m.G
	node, offset, lead := m.prefixStart(prefix)
	if node < First || lead != "" || !g.Alive[node] {
		return -1
	}
	if offset+g.Enc.N < g.labelLen[node] {
		if _, _, err := g.Split(node, offset+g.Enc.Stride); err != nil { // the gram becomes the last of its node
			return -1
		}
	}
	return node
}

func normalizeThought(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

// ThinkOptions configure one thought (Model.Think); DefaultThinkOptions are the Python defaults.
type ThinkOptions struct {
	// About is the text to think about: with no At, the thought is at the node where it ends (Place).
	About string
	// At is the node the event happened at (-1: nowhere in particular).
	At int
	// Trigger says what happened: TriggerAsked, TriggerStutter / TriggerRepeat, TriggerQuestioned, or a
	// caller's own word for its event.  It decides what the thought triggers when it stops.
	Trigger string
	// Went and Instead are what a thought backing out of a repeat teaches Back: the step it was about to
	// loop through and the one it took instead (-1: not known), as ObserveBack takes them.
	Went, Instead int
	// The prediction search's settings, run from Think: "beam" thinks the most likely thought, "sample"
	// draws one.
	Mode        string
	K           int
	Beam        int
	MaxLength   int
	StepPenalty float64
	Temperature float64
	RNG         *MT19937
	Seed        *int64
	// Depth is how deep this thought is (0: a thought, 1: a question it asked itself, ...); MaxDepth and
	// MaxQuestions bound how it questions itself (0 turns it off).
	Depth        int
	MaxDepth     int
	MaxQuestions int
	// Learn writes what the thought learned into the model: where it stopped to think, where it handed over.
	Learn bool
	// Amount is how much each lesson moves the model.
	Amount float64
}

// DefaultThinkOptions mirror the Python defaults.
func DefaultThinkOptions() ThinkOptions {
	return ThinkOptions{
		At: -1, Trigger: TriggerAsked, Went: -1, Instead: -1, Mode: "beam", K: 5, MaxLength: ThinkLength,
		Temperature: 1, MaxDepth: ThinkDepth, MaxQuestions: ThinkQuestions, Learn: true, Amount: 1,
	}
}

// Think is one thought, triggered by o.Trigger at node o.At (or at the end of o.About).
//
// With o.Learn the node is taught to think (ObserveThink) unless the thought is
// a question the model asked itself - it already knows to think there, that is
// why it asked.  Learn off thinks without writing anything into the model.
func (m *Model) Think(o ThinkOptions) (*Thought, error) {
	mode := strings.ToLower(o.Mode)
	if mode == "" || mode == "dijkstra" {
		mode = "beam"
	}
	if mode != "beam" && mode != "sample" {
		return nil, fmt.Errorf("unknown mode %q; expected 'beam' or 'sample'", o.Mode)
	}
	if o.MaxLength < 0 {
		return nil, fmt.Errorf("max_length must be >= 0, got %d", o.MaxLength)
	}
	if o.K < 1 {
		return nil, fmt.Errorf("k must be >= 1, got %d", o.K)
	}
	if o.Beam < 0 {
		return nil, fmt.Errorf("beam must be >= 1, got %d", o.Beam)
	}
	if o.MaxDepth < 0 || o.MaxQuestions < 0 {
		return nil, fmt.Errorf("max_depth and max_questions must be >= 0")
	}
	if o.Temperature < 0 {
		return nil, fmt.Errorf("temperature must be >= 0")
	}
	if o.Amount < 0 {
		return nil, fmt.Errorf("amount must be >= 0, got %v", o.Amount)
	}
	if o.RNG == nil && o.Seed != nil {
		o.RNG = NewMT19937(*o.Seed)
	}
	o.Mode = mode
	return m.thinkAt(o, map[string]bool{})
}

// thinkAt is Think inside the chain of thoughts already thought (normalised texts): a question that only
// repeats the thought above it is no question.
func (m *Model) thinkAt(o ThinkOptions, thought map[string]bool) (*Thought, error) {
	g := m.G
	at := o.At
	if at < 0 && strings.TrimSpace(o.About) != "" {
		at = m.Place(o.About)
	}
	if at >= 0 && (at < First || at >= len(g.Labels) || !g.Alive[at]) {
		return nil, fmt.Errorf("node %d is not a real node to think at", at)
	}
	record := newThought(o.Trigger, at, o.About, o.Depth)

	// 1. the event teaches where to think - from experience, as Back is taught
	if o.Learn && at >= First && o.Trigger != TriggerQuestioned {
		if _, err := g.ObserveThink(at, o.Amount); err != nil {
			return nil, err
		}
		record.Taught = at
	}

	// 2. the thought: the search from Think, in the language of the thoughts it was taught
	candidates, expanded, err := m.thoughtCandidates(o)
	if err != nil {
		return nil, err
	}
	record.Expanded = expanded
	var chosen *PathResult
	for _, cand := range candidates {
		if strings.TrimSpace(cand.Text) == "" {
			continue
		}
		if thought[normalizeThought(cand.Text)] {
			continue // a question that only repeats the thought above it is no question
		}
		chosen = cand
		break
	}
	if chosen != nil {
		record.Text = chosen.Text
		record.Cost = chosen.Cost
		record.Probability = chosen.Probability()
		record.Labels = append([]string{}, chosen.Labels...)
		record.NodeIDs = append([]int{}, chosen.NodeIDs...)
		record.StepCosts = append([]float64{}, chosen.StepCosts...)
		record.Stopped = StoppedLength
		if chosen.ReachedEnd {
			record.Stopped = StoppedEnd
		}

		// 3. questioning itself: where, along its own path, the model has learned to think
		if o.Depth < o.MaxDepth && o.MaxQuestions > 0 {
			soFar := make(map[string]bool, len(thought)+2)
			for k := range thought {
				soFar[k] = true
			}
			soFar[normalizeThought(record.Text)] = true
			for _, node := range record.NodeIDs {
				if node < First || !g.ThinksAt(node) {
					continue
				}
				q := o
				q.About, q.At, q.Trigger, q.Went, q.Instead, q.Depth = "", node, TriggerQuestioned, -1, -1, o.Depth+1
				question, err := m.thinkAt(q, soFar)
				if err != nil {
					return nil, err
				}
				record.Questions = append(record.Questions, question)
				soFar[normalizeThought(question.Text)] = true
				if len(record.Questions) >= o.MaxQuestions {
					break
				}
			}
		}
	}

	// 4. when thinking stops, it triggers the sentinel the event calls for
	switch {
	case o.Trigger == TriggerQuestioned:
		record.Then = ThenThink
	case BacksUp(o.Trigger) && at >= First:
		record.Then = ThenBack
		if o.Learn {
			// ObserveBack skips a step that is not one of the node's own children, and Back itself
			if _, err := g.ObserveBack(at, o.Went, o.Instead, o.Amount); err != nil {
				return nil, err
			}
			record.HandedOver = at
		}
	default:
		record.Then = ThenEnd
	}
	return record, nil
}

// thoughtCandidates are up to K thoughts from Think, most likely first (beam), or K walks (sample), and the
// states weighed.
func (m *Model) thoughtCandidates(o ThinkOptions) ([]*PathResult, int, error) {
	walk := rewardWalk
	walk.Origin = Think
	if o.Mode == "sample" {
		draws := o.K
		if draws < 1 {
			draws = 1
		}
		drawn := make([]*PathResult, 0, draws)
		expanded := 0
		for i := 0; i < draws; i++ {
			pred, err := m.search("", o.MaxLength, "sample", 0, 0, 0.0, o.Temperature, false, o.MaxLength, o.RNG, walk)
			if err != nil {
				return nil, 0, err
			}
			expanded += pred.Expanded
			one := pred.PathResult
			drawn = append(drawn, &one)
		}
		return drawn, expanded, nil
	}
	found, err := m.search("", 0, "beam", o.K, o.Beam, o.StepPenalty, o.Temperature, true, o.MaxLength, nil, walk)
	if err != nil {
		return nil, 0, err
	}
	return found.Top, found.Expanded, nil
}

// Learned is what Model.ThinkOn taught: the thoughts, the questions they asked
// themselves, the nodes taught to stop and think, and the epoch records.
type Learned struct {
	Thoughts  int              `json:"thoughts"`
	Questions int              `json:"questions"`
	Taught    int              `json:"taught"`
	Epochs    []map[string]any `json:"epochs"`
}

// ThinkOn teaches the model thoughts: texts whose walk begins at Think.
//
// Every thought is trained the way this model trains texts - the same
// structure, the same counting, the same compression - from the other sentinel,
// so the model learns how *thoughts* begin and go on without a word of them
// leaking into what it says from Start (Train with Origin Think; opts are its
// epochs and the rest).
//
// With questions, a thought that questions itself teaches that too: every
// sentence ending in "?" (QuestionsIn) marks the node before it as a place the
// model stops to think (ObserveThink by amount, on the node cut to end exactly
// there - Place) and is trained as a thought of its own, so a thought passing
// through that node later may question itself with it (Think).
func (m *Model) ThinkOn(thoughts []string, opts TrainOptions, questions bool, amount float64) (*Learned, error) {
	if m.IsNegative() {
		return nil, fmt.Errorf("the negative network judges; it does not think")
	}
	cleaned := make([]string, 0, len(thoughts))
	for _, t := range thoughts {
		if oneLine := strings.Join(strings.Fields(t), " "); oneLine != "" {
			cleaned = append(cleaned, oneLine)
		}
	}
	opts.Origin = Think
	records, err := m.Train(cleaned, opts)
	if err != nil {
		return nil, err
	}
	if records == nil {
		records = []map[string]any{}
	}
	learned := &Learned{Thoughts: len(cleaned), Epochs: records}
	if !questions || len(cleaned) == 0 || (opts.Stop != nil && opts.Stop()) {
		return learned, nil
	}
	g := m.G
	asked := []string{}
	for _, t := range cleaned {
		for _, q := range QuestionsIn(t) {
			if node := m.Place(t[:q.Start]); node >= First {
				if _, err := g.ObserveThink(node, amount); err != nil {
					return nil, err
				}
				learned.Taught++
			}
			asked = append(asked, q.Text)
		}
	}
	// a question is a thought of its own: the model may begin a thought with it
	enc := m.Encoding()
	kept := make([]string, 0, len(asked))
	for _, q := range asked {
		if enc.Len(q) >= enc.N {
			kept = append(kept, q)
		}
	}
	learned.Questions = len(kept)
	if len(kept) > 0 {
		more, err := m.Train(kept, opts)
		if err != nil {
			return nil, err
		}
		learned.Epochs = append(learned.Epochs, more...)
	}
	return learned, nil
}

// Summarize is one line saying what a thought did, for transcripts: `thought
// "..."; questioned itself once; then backed up`.
func Summarize(t *Thought) string {
	var said string
	switch {
	case t.Text != "":
		said = "thought " + quoteThought(t.Text)
	case t.Stopped == StoppedNothing && t.Depth > 0:
		said = "had nothing new to think"
	default:
		said = "had nothing to think with yet"
	}
	parts := []string{said}
	if n := len(t.Questions); n > 0 {
		times := "once"
		if n != 1 {
			times = fmt.Sprintf("%d times", n)
		}
		parts = append(parts, "questioned itself "+times)
	}
	switch t.Then {
	case ThenBack:
		parts = append(parts, "then backed up")
	case ThenThink:
		parts = append(parts, "then went back to the thought")
	case ThenEnd:
		parts = append(parts, "then went on")
	default:
		parts = append(parts, "then "+t.Then)
	}
	return strings.Join(parts, "; ")
}

func quoteThought(text string) string {
	return `"` + strings.ReplaceAll(text, `"`, `\"`) + `"`
}

// ThoughtsOf is every thought a conversation had, in order (the turns' rethinks' thoughts).
func ThoughtsOf(turns []*Turn) []*Thought {
	out := []*Thought{}
	for _, turn := range turns {
		if turn != nil && turn.Rethink != nil && turn.Rethink.Thought != nil {
			out = append(out, turn.Rethink.Thought)
		}
	}
	return out
}
