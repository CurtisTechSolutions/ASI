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

// Repeats are the duplicates a conversation could not avoid: the utterances of
// the turns flagged Repeat, each once.  They are the texts to punish - the
// thumbs down of a 2NRL negative phase - so the model stops offering them.
func Repeats(turns []*Turn) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, t := range turns {
		key := Normalize(t.Text)
		if t.Repeat && key != "" && !seen[key] {
			seen[key] = true
			out = append(out, t.Text)
		}
	}
	return out
}

// Heard is what a conversation has already heard, and what therefore counts as
// a duplicate.  An utterance duplicates the conversation when it was said
// before, when its reply adds what some earlier reply already added (the same
// continuation reached from another context), or when it merely echoes a line
// already spoken (the whole utterance sits inside one of them).  A longer
// utterance that happens to contain an earlier one is not a duplicate: it says
// more than was heard.
type Heard struct {
	keys  []string // the utterances heard, in order
	said  map[string]bool
	added map[string]bool // the words the replies added
}

// NewHeard remembers texts as the utterances a conversation opens with.
func NewHeard(texts []string) *Heard {
	h := &Heard{said: map[string]bool{}, added: map[string]bool{}}
	for _, text := range texts {
		h.Remember(text, "")
	}
	return h
}

// Remember takes an utterance (and the words its reply added) into the conversation.
func (h *Heard) Remember(text, reply string) {
	if key := Normalize(text); key != "" && !h.said[key] {
		h.said[key] = true
		h.keys = append(h.keys, key)
	}
	if added := Normalize(reply); added != "" {
		h.added[added] = true
	}
}

// Duplicate reports whether speaking text (a reply adding reply) would repeat the conversation.
func (h *Heard) Duplicate(text, reply string) bool {
	key := Normalize(text)
	if key == "" {
		return false
	}
	if h.said[key] {
		return true
	}
	if added := Normalize(reply); added != "" && h.added[added] {
		return true
	}
	for _, heard := range h.keys {
		if strings.Contains(heard, key) {
			return true
		}
	}
	return false
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

// pick returns the first candidate that adds something and (when asked) does
// not duplicate what the conversation has heard.  When they all do, the best
// duplicate is the fallback - the cheapest one that was never said word for
// word, else the cheapest of all - and speaking it flags the turn a repeat.
func pick(cands []*PathResult, heard *Heard, avoidRepeats bool) (*PathResult, int, bool) {
	skipped := 0
	var fallback *PathResult
	fallbackWordForWord := true
	for _, c := range cands {
		if strings.TrimSpace(c.Text) == "" {
			skipped++
			continue
		}
		if avoidRepeats && heard.Duplicate(c.FullText, c.Text) {
			wordForWord := heard.said[Normalize(c.FullText)]
			if fallback == nil || (fallbackWordForWord && !wordForWord) {
				fallback, fallbackWordForWord = c, wordForWord
			}
			skipped++
			continue
		}
		return c, skipped, false
	}
	return fallback, skipped, fallback != nil
}

// Converse lets the model talk to itself (or to opts.Partner) for opts.Turns
// new turns: every reply is the prediction search picking up the tail of the
// previous line.  When every candidate duplicates the conversation the best one
// is spoken and flagged a repeat (Repeats collects those, the utterances to
// punish); a duplicate already repeated ends the conversation instead of going
// round in circles.  See the Python implementation's radixnet.dialogue.
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
	repeated := map[string]bool{} // the duplicates already spoken: saying one of them again would only loop
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
	heard := NewHeard(saidList)
	for turn := 0; turn < opts.Turns; turn++ {
		voice := voices[index%2]
		speaker := speakers[index%len(speakers)]
		previous := ""
		if len(saidList) > 0 {
			previous = saidList[len(saidList)-1]
		}
		ctx := TailContext(previous, opts.Context)
		var spoken *PathResult
		offered, skipped := 0, 0
		repeat := false
		draws := 1
		if mode == "sample" {
			draws = opts.K
		}
		for ctx != "" {
			if voice.usable(ctx) {
				for d := 0; d < draws; d++ {
					cands, err := voice.candidates(ctx, mode, opts.K, opts.Beam, opts.MaxLength, opts.StepPenalty, opts.Temperature, rng)
					if err != nil {
						return nil, err
					}
					offered += len(cands)
					var dropped int
					spoken, dropped, repeat = pick(cands, heard, opts.AvoidRepeats)
					skipped += dropped
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
			var freshPick *PathResult
			freshRepeat := false
			for d := 0; d < draws; d++ {
				cands, err := voice.candidates("", mode, opts.K, opts.Beam, opts.MaxLength, opts.StepPenalty, opts.Temperature, rng)
				if err != nil {
					return nil, err
				}
				offered += len(cands)
				var dropped int
				freshPick, dropped, freshRepeat = pick(cands, heard, opts.AvoidRepeats)
				skipped += dropped
				if freshPick != nil && !freshRepeat {
					break
				}
			}
			if freshPick != nil && (spoken == nil || !freshRepeat) {
				spoken, repeat, ctx = freshPick, freshRepeat, ""
			}
		}
		if spoken == nil {
			break
		}
		text := spoken.Text
		if ctx != "" {
			text = spoken.FullText
		}
		if repeat && repeated[Normalize(text)] {
			break // the voice can only say a duplicate it has already repeated: the conversation is over
		}
		result = append(result, &Turn{Index: index, Speaker: speaker, Text: text, Context: ctx, Reply: spoken.Text,
			Cost: spoken.Cost, Probability: spoken.Probability(), ReachedEnd: spoken.ReachedEnd, Fresh: ctx == "", Repeat: repeat,
			Candidates: offered, Skipped: skipped, Labels: append([]string(nil), spoken.Labels...),
			NodeIDs: append([]int(nil), spoken.NodeIDs...), StepCosts: append([]float64(nil), spoken.StepCosts...)})
		saidList = append(saidList, text)
		reply := ""
		if ctx != "" {
			reply = spoken.Text
		}
		heard.Remember(text, reply)
		if repeat {
			repeated[Normalize(text)] = true
		}
		index++
	}
	return result, nil
}
