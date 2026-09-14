package radixnet

import (
	"fmt"
	"strings"
)

// DefaultSpeakers are the two voices of a conversation.
var DefaultSpeakers = []string{"A", "B"}

// Turn is one utterance of a conversation.
type Turn struct {
	Index       int       `json:"index"`
	Speaker     string    `json:"speaker"`
	Text        string    `json:"text"`
	Context     string    `json:"context"`
	Reply       string    `json:"reply"`
	Cost        float64   `json:"cost"`
	Probability float64   `json:"probability"`
	ReachedEnd  bool      `json:"reached_end"`
	Fresh       bool      `json:"fresh"`
	Given       bool      `json:"given"`
	Repeat      bool      `json:"repeat"`
	Candidates  int       `json:"candidates"`
	Skipped     int       `json:"skipped"`
	Vetoed      int       `json:"vetoed"`
	Labels      []string  `json:"labels"`
	NodeIDs     []int     `json:"node_ids"`
	StepCosts   []float64 `json:"step_costs"`
}

// TailContext is the last chars characters of text, cut forward to a word
// boundary when a word straddles the cut.
func TailContext(text string, chars int) string {
	text = strings.TrimRight(text, " \t\r\n")
	if chars <= 0 || text == "" {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= chars {
		return strings.TrimLeft(text, " \t")
	}
	tail := string(runes[len(runes)-chars:])
	if cut := strings.Index(tail, " "); cut >= 0 && strings.TrimSpace(tail[cut+1:]) != "" {
		tail = tail[cut+1:]
	}
	return strings.TrimLeft(tail, " \t")
}

// Normalize is the key two utterances are compared by.
func Normalize(text string) string { return strings.ToLower(strings.Join(strings.Fields(text), " ")) }

// Transcript renders "speaker: text" lines.
func Transcript(turns []*Turn) string {
	lines := make([]string, len(turns))
	for i, t := range turns {
		lines[i] = t.Speaker + ": " + t.Text
	}
	return strings.Join(lines, "\n")
}

// ConverseOptions configure Converse.
type ConverseOptions struct {
	Turns        int
	Mode         string
	MaxLength    int
	Context      int
	Temperature  float64
	K            int
	Beam         int
	StepPenalty  float64
	Seed         *int64
	Speakers     []string
	History      []string
	Partner      *Model
	AvoidRepeats bool
	// Veto is what a voice may not say: true for a candidate the speaker must
	// not speak.  The conversation knows nothing about why - Filter.Converse
	// passes its own judgement in (the negative network guarding the positive
	// one), and a candidate it refuses is skipped exactly like one that had
	// been said before, except that it may not even be the fallback.
	Veto func(string) bool
}

// DefaultConverseOptions mirror the Python defaults.
func DefaultConverseOptions() ConverseOptions {
	return ConverseOptions{Turns: 6, Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1.0, K: 5, Speakers: DefaultSpeakers, AvoidRepeats: true}
}

func shorter(context string) string {
	context = strings.TrimRight(context, " \t")
	if i := strings.LastIndex(context, " "); i >= 0 {
		return strings.TrimRight(context[:i], " \t")
	}
	return ""
}

func (m *Model) usable(context string) bool {
	node, _, lead := m.prefixStart(context)
	return node != Start && lead == ""
}

func (m *Model) candidates(context, mode string, k, beam, maxLength int, stepPenalty, temperature float64, rng *MT19937) ([]*PathResult, error) {
	if mode == "sample" {
		walk, err := m.search(context, 0, "sample", 0, 0, stepPenalty, temperature, false, maxLength, rng)
		if err != nil {
			return nil, err
		}
		return []*PathResult{&walk.PathResult}, nil
	}
	found, err := m.search(context, 0, "beam", k, beam, stepPenalty, 1.0, true, maxLength, nil)
	if err != nil {
		return nil, err
	}
	return found.Top, nil
}

// pick is the first candidate that adds something, is not vetoed and (when
// asked) is neither an utterance heard before nor an echo of the previous
// line; the best repeat is the fallback.  A vetoed candidate is never the
// fallback - that is the whole point of the veto.
func pick(cands []*PathResult, said map[string]bool, previous string, avoidRepeats bool, veto func(string) bool) (*PathResult, int, bool, int) {
	skipped, vetoed := 0, 0
	var fallback *PathResult
	for _, c := range cands {
		if strings.TrimSpace(c.Text) == "" {
			skipped++
			continue
		}
		if veto != nil && veto(c.FullText) {
			vetoed++
			skipped++
			continue
		}
		key := Normalize(c.FullText)
		if avoidRepeats && (said[key] || (previous != "" && strings.Contains(previous, key))) {
			if fallback == nil {
				fallback = c
			}
			skipped++
			continue
		}
		return c, skipped, false, vetoed
	}
	return fallback, skipped, fallback != nil, vetoed
}

// Converse lets the model talk to itself (or to opts.Partner) for opts.Turns
// new turns: every reply is the prediction search picking up the tail of the
// previous line.  See the Python implementation's radixnet.dialogue.
func (m *Model) Converse(opening string, opts ConverseOptions) ([]*Turn, error) {
	mode := opts.Mode
	if mode == "" || mode == "dijkstra" {
		mode = "beam"
	}
	if mode != "beam" && mode != "sample" {
		return nil, fmt.Errorf("unknown mode %q; expected 'beam' or 'sample'", opts.Mode)
	}
	if opts.Turns < 0 || opts.MaxLength < 0 || opts.Context < 0 || opts.K < 1 || opts.Beam < 0 || opts.Temperature < 0 || opts.StepPenalty < 0 {
		return nil, fmt.Errorf("invalid converse options")
	}
	speakers := opts.Speakers
	if len(speakers) == 0 {
		speakers = DefaultSpeakers
	}
	for _, s := range speakers {
		if strings.TrimSpace(s) == "" {
			return nil, fmt.Errorf("speakers must be non-empty names")
		}
	}
	var rng *MT19937
	if opts.Seed != nil {
		rng = NewMT19937(*opts.Seed)
	}
	voices := [2]*Model{m, m}
	if opts.Partner != nil {
		voices[1] = opts.Partner
	}
	saidList := append([]string(nil), opts.History...)
	result := []*Turn{}
	index := len(saidList)
	if strings.TrimSpace(opening) != "" {
		sc := m.Score(opening)
		cost := -sc.LogProb
		result = append(result, &Turn{Index: index, Speaker: speakers[index%len(speakers)], Text: opening, Reply: opening,
			Cost: cost, Probability: (&PathResult{Cost: cost}).Probability(), ReachedEnd: true, Fresh: true, Given: true,
			Candidates: 1, Labels: []string{}, NodeIDs: []int{}, StepCosts: []float64{}})
		saidList = append(saidList, opening)
		index++
	}
	said := make(map[string]bool)
	for _, t := range saidList {
		if strings.TrimSpace(t) != "" {
			said[Normalize(t)] = true
		}
	}
	for turn := 0; turn < opts.Turns; turn++ {
		voice := voices[index%2]
		previous := ""
		if len(saidList) > 0 {
			previous = saidList[len(saidList)-1]
		}
		spoken, err := voice.Reply(previous, ReplyOptions{
			Said: said, Index: index, Speaker: speakers[index%len(speakers)], Mode: mode,
			MaxLength: opts.MaxLength, Context: opts.Context, Temperature: opts.Temperature, K: opts.K,
			Beam: opts.Beam, StepPenalty: opts.StepPenalty, RNG: rng, AvoidRepeats: opts.AvoidRepeats,
			Veto: opts.Veto,
		})
		if err != nil {
			return nil, err
		}
		if spoken == nil {
			break
		}
		result = append(result, spoken)
		saidList = append(saidList, spoken.Text)
		said[Normalize(spoken.Text)] = true
		index++
	}
	return result, nil
}

// ReplyOptions are how one voice finds what to say next.
type ReplyOptions struct {
	// Said is what the conversation has already heard, so a reply does not repeat it.
	Said         map[string]bool
	Index        int
	Speaker      string
	Mode         string
	MaxLength    int
	Context      int
	Temperature  float64
	K            int
	Beam         int
	StepPenalty  float64
	RNG          *MT19937
	AvoidRepeats bool
	// Veto is what the speaker may not say (see ConverseOptions.Veto).
	Veto func(string) bool
}

// DefaultReplyOptions mirror the Python defaults.
func DefaultReplyOptions() ReplyOptions {
	return ReplyOptions{Speaker: "B", Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1, K: 5,
		AvoidRepeats: true}
}

// Reply is what this model says next after previous - one turn, or nil when it
// has nothing to say.
//
// This is the whole of a conversational turn, and Converse is a loop over it:
// the tail of previous is located in the graph and continued, the context
// loses a word at a time while nothing follows it, and a voice with nothing
// left to add changes the subject with a fresh text from START.
//
// It is exported because the other voice need not be a model at all: the chat
// loop (chat.go) has an LLM speak every other line and calls this for the
// model's own.
func (m *Model) Reply(previous string, o ReplyOptions) (*Turn, error) {
	mode := o.Mode
	if mode == "" || mode == "dijkstra" {
		mode = "beam"
	}
	if mode != "beam" && mode != "sample" {
		return nil, fmt.Errorf("unknown mode %q; expected 'beam' or 'sample'", o.Mode)
	}
	if o.MaxLength < 0 || o.Context < 0 || o.K < 1 || o.Beam < 0 || o.Temperature < 0 || o.StepPenalty < 0 {
		return nil, fmt.Errorf("invalid reply options")
	}
	said := o.Said
	if said == nil {
		said = map[string]bool{}
	}
	speaker := o.Speaker
	if speaker == "" {
		speaker = "B"
	}
	previousKey := Normalize(previous)
	ctx := TailContext(previous, o.Context)
	var spoken *PathResult
	offered, skipped, vetoed := 0, 0, 0
	repeat := false
	draws := 1
	if mode == "sample" {
		draws = o.K
	}
	for ctx != "" {
		if m.usable(ctx) {
			for d := 0; d < draws; d++ {
				cands, err := m.candidates(ctx, mode, o.K, o.Beam, o.MaxLength, o.StepPenalty, o.Temperature, o.RNG)
				if err != nil {
					return nil, err
				}
				offered += len(cands)
				var dropped, refused int
				spoken, dropped, repeat, refused = pick(cands, said, previousKey, o.AvoidRepeats, o.Veto)
				skipped += dropped
				vetoed += refused
				if spoken != nil && !repeat {
					break
				}
			}
			if spoken != nil && !repeat {
				break
			}
		}
		ctx = shorter(ctx)
	}
	if spoken == nil || repeat {
		// nothing (new) follows the previous line: change the subject with a fresh text
		var freshPick *PathResult
		freshRepeat := false
		for d := 0; d < draws; d++ {
			cands, err := m.candidates("", mode, o.K, o.Beam, o.MaxLength, o.StepPenalty, o.Temperature, o.RNG)
			if err != nil {
				return nil, err
			}
			offered += len(cands)
			var dropped, refused int
			freshPick, dropped, freshRepeat, refused = pick(cands, said, previousKey, o.AvoidRepeats, o.Veto)
			skipped += dropped
			vetoed += refused
			if freshPick != nil && !freshRepeat {
				break
			}
		}
		if freshPick != nil && (spoken == nil || !freshRepeat) {
			spoken, repeat, ctx = freshPick, freshRepeat, ""
		}
	}
	if spoken == nil {
		return nil, nil
	}
	text := spoken.Text
	if ctx != "" {
		text = spoken.FullText
	}
	return &Turn{Index: o.Index, Speaker: speaker, Text: text, Context: ctx, Reply: spoken.Text,
		Cost: spoken.Cost, Probability: spoken.Probability(), ReachedEnd: spoken.ReachedEnd, Fresh: ctx == "",
		Repeat: repeat, Candidates: offered, Skipped: skipped, Vetoed: vetoed,
		Labels: append([]string(nil), spoken.Labels...), NodeIDs: append([]int(nil), spoken.NodeIDs...),
		StepCosts: append([]float64(nil), spoken.StepCosts...)}, nil
}
