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

func pick(cands []*PathResult, said map[string]bool, previous string, avoidRepeats bool) (*PathResult, int, bool) {
	skipped := 0
	var fallback *PathResult
	for _, c := range cands {
		if strings.TrimSpace(c.Text) == "" {
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
		return c, skipped, false
	}
	return fallback, skipped, fallback != nil
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
		speaker := speakers[index%len(speakers)]
		previous := ""
		if len(saidList) > 0 {
			previous = saidList[len(saidList)-1]
		}
		previousKey := Normalize(previous)
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
					spoken, dropped, repeat = pick(cands, said, previousKey, opts.AvoidRepeats)
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
				freshPick, dropped, freshRepeat = pick(cands, said, previousKey, opts.AvoidRepeats)
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
		result = append(result, &Turn{Index: index, Speaker: speaker, Text: text, Context: ctx, Reply: spoken.Text,
			Cost: spoken.Cost, Probability: spoken.Probability(), ReachedEnd: spoken.ReachedEnd, Fresh: ctx == "", Repeat: repeat,
			Candidates: offered, Skipped: skipped, Labels: append([]string(nil), spoken.Labels...),
			NodeIDs: append([]int(nil), spoken.NodeIDs...), StepCosts: append([]float64(nil), spoken.StepCosts...)})
		saidList = append(saidList, text)
		said[Normalize(text)] = true
		index++
	}
	return result, nil
}
