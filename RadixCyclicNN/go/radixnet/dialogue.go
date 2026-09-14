package radixnet

import (
	"fmt"
	"strings"
)

// DefaultSpeakers are the two voices of a conversation.
var DefaultSpeakers = []string{"A", "B"}

// Explore is how many times a voice may back up out of a repeat by default
// (0 turns the exploring off).
const Explore = 3

// Rethink is a voice catching itself repeating, and what it did about it.
//
// The metacognition of a turn: Noticed is the run of words it caught itself
// saying twice, Cut what it kept of that attempt (everything said before the
// walk went round), Steps how many times it backed up, Explored the paths it
// weighed from there and Found whether one of them said something new.  A turn
// that never had to think twice has no record at all.
type Rethink struct {
	Noticed  string `json:"noticed"`
	Cut      string `json:"cut"`
	Steps    int    `json:"steps"`
	Explored int    `json:"explored"`
	Found    bool   `json:"found"`
}

// Turn is one utterance of a conversation.
type Turn struct {
	Index       int     `json:"index"`
	Speaker     string  `json:"speaker"`
	Text        string  `json:"text"`
	Context     string  `json:"context"`
	Reply       string  `json:"reply"`
	Cost        float64 `json:"cost"`
	Probability float64 `json:"probability"`
	ReachedEnd  bool    `json:"reached_end"`
	Fresh       bool    `json:"fresh"`
	Given       bool    `json:"given"`
	Repeat      bool    `json:"repeat"`
	Stutter     bool    `json:"stutter"`
	// Rethink is what the voice noticed about a repeat of its own, and how it backed out of it.
	Rethink    *Rethink  `json:"rethink"`
	Candidates int       `json:"candidates"`
	Skipped    int       `json:"skipped"`
	Vetoed     int       `json:"vetoed"`
	Labels     []string  `json:"labels"`
	NodeIDs    []int     `json:"node_ids"`
	StepCosts  []float64 `json:"step_costs"`
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

// LongestStutter is how long a run may be for Stutter to call its immediate
// repetition a stutter.
const LongestStutter = 4

// Stutter is the words an utterance says twice in a row, or "" when it says
// each thing once.  A stutter is a run of one to longest words repeated
// immediately after itself - "the the west", "say morning morning", "the cat
// the cat sat" - the shape a cyclic graph falls into when it walks a loop
// instead of going somewhere.  Words that come back later in the line are not
// a stutter: "where there is a will there is a way" says its words again, and
// says something with them.
func Stutter(text string, longest int) string {
	run, _ := caught(text, longest)
	return run
}

// StutterAt is where a Stutter starts saying itself again (a byte index into
// text), or -1.  This is where the walk went round: everything before it was
// said once, and a voice backing out of the loop keeps exactly that much
// (Backtrack).
func StutterAt(text string, longest int) int {
	_, at := caught(text, longest)
	return at
}

// caught is the run said twice in a row and where text starts saying it again.
func caught(text string, longest int) (string, int) {
	words, at := []string{}, []int{}
	for i := 0; i < len(text); {
		if text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' {
			i++
			continue
		}
		start := i
		for i < len(text) && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r' {
			i++
		}
		words = append(words, strings.ToLower(text[start:i]))
		at = append(at, start)
	}
	for i := range words {
		limit := (len(words) - i) / 2
		if limit > longest {
			limit = longest
		}
		for run := 1; run <= limit; run++ {
			same := true
			for j := 0; j < run; j++ {
				if words[i+j] != words[i+run+j] {
					same = false
					break
				}
			}
			if same {
				return strings.Join(words[i:i+run], " "), at[i+run]
			}
		}
	}
	return "", -1
}

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
	// AvoidWordRepeats keeps a reply from repeating its own words (a Stutter).
	AvoidWordRepeats bool
	// Explore is how many times a voice that caught itself repeating may back up and look for another way on.
	Explore int
	// Veto is what a voice may not say: true for a candidate the speaker must
	// not speak.  The conversation knows nothing about why - Filter.Converse
	// passes its own judgement in (the negative network guarding the positive
	// one), and a candidate it refuses is skipped exactly like one that had
	// been said before, except that it may not even be the fallback.
	Veto func(string) bool
}

// DefaultConverseOptions mirror the Python defaults.
func DefaultConverseOptions() ConverseOptions {
	return ConverseOptions{Turns: 6, Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1.0, K: 5,
		Speakers: DefaultSpeakers, AvoidRepeats: true, AvoidWordRepeats: true, Explore: Explore}
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

// offer is up to count continuations of context: the count most likely (beam),
// or that many walks (sample).
func (m *Model) offer(context, mode string, count, beam, maxLength int, stepPenalty, temperature float64, rng *MT19937) ([]*PathResult, error) {
	if mode != "sample" {
		return m.candidates(context, mode, count, beam, maxLength, stepPenalty, temperature, rng)
	}
	drawn := []*PathResult{}
	for i := 0; i < count; i++ {
		one, err := m.candidates(context, mode, 1, beam, maxLength, stepPenalty, temperature, rng)
		if err != nil {
			return nil, err
		}
		drawn = append(drawn, one...)
	}
	return drawn, nil
}

// BacktrackOptions configure Backtrack.
type BacktrackOptions struct {
	// Keep is what the voice may not rewrite - the context it picked up from the other voice.
	Keep         string
	Heard        *Heard
	Explore      int
	Mode         string
	K            int
	Beam         int
	MaxLength    int
	StepPenalty  float64
	Temperature  float64
	RNG          *MT19937
	AvoidRepeats bool
	Veto         func(string) bool
}

// Backtrack has a voice that caught itself repeating go back to where the loop
// started and look for another way on.
//
// text is the utterance it was about to say and StutterAt where it began
// saying itself again: everything before that was said once, so it is kept and
// the search runs again from there (a longer prefix than the turn started
// with, which forces the walk to leave the loop at exactly the point it went
// round - asking the same question again from the context would only rank the
// same answers).  Nothing found, or everything found repeats too?  Then it
// backs up one word further and looks wider, o.Explore times over.
//
// o.Keep is what it may not rewrite, so a voice rethinks what it said, never
// what it heard.  The candidate continues the kept words (its FullText is the
// whole utterance, its Cost the path it explored from where it backed up), and
// the Rethink says what it noticed and did, found or not.
func (m *Model) Backtrack(text string, o BacktrackOptions) (*PathResult, *Rethink, error) {
	heard := o.Heard
	if heard == nil {
		heard = NewHeard(nil)
	}
	noticed, at := caught(text, LongestStutter)
	record := &Rethink{Noticed: noticed}
	if noticed == "" || o.Explore <= 0 {
		return nil, record, nil
	}
	cut := text[:at]
	if len(cut) < len(o.Keep) { // the other voice's words: not this one's to rethink
		return nil, record, nil
	}
	for step := 0; step < o.Explore; step++ {
		if strings.TrimSpace(cut) == "" { // nothing of its own left to keep
			break
		}
		record.Cut, record.Steps = cut, step+1
		wider := o.K * (step + 2) // the further back it goes, the wider it looks
		cands, err := m.offer(cut, o.Mode, wider, o.Beam, o.MaxLength, o.StepPenalty, o.Temperature, o.RNG)
		if err != nil {
			return nil, record, err
		}
		for _, cand := range cands {
			record.Explored++
			if strings.TrimSpace(cand.Text) == "" || (o.Veto != nil && o.Veto(cand.FullText)) {
				continue
			}
			if Stutter(cand.FullText, LongestStutter) != "" ||
				(o.AvoidRepeats && heard.Duplicate(cand.FullText, cand.Text)) {
				continue
			}
			record.Found = true
			return cand, record, nil
		}
		shorter := shorter(cut)
		if len(shorter) < len(o.Keep) || shorter == cut {
			break
		}
		if shorter != "" && !strings.HasSuffix(shorter, " ") {
			shorter += " "
		}
		cut = shorter
	}
	return nil, record, nil
}

// pick returns the first candidate that adds something, is not vetoed and
// repeats nothing - neither what the conversation has heard (avoidRepeats) nor
// its own words (avoidWordRepeats: a Stutter).  When they all repeat
// something, the best of them is the fallback - the cheapest one that was
// never said word for word, else the cheapest of all - and speaking it flags
// the turn a repeat.  A vetoed candidate is never the fallback - that is the
// whole point of the veto.
// picked is what one look through the candidates turned up.
type picked struct {
	spoken  *PathResult // what to say, or the best repeat when everything repeated
	skipped int
	repeat  bool // spoken repeats something: nothing else was left
	vetoed  int
	looped  *PathResult // the best candidate rejected only for repeating its own words
}

func pick(cands []*PathResult, heard *Heard, avoidRepeats bool, veto func(string) bool, avoidWordRepeats bool) picked {
	out := picked{}
	fallbackWordForWord := true
	for _, c := range cands {
		if strings.TrimSpace(c.Text) == "" {
			out.skipped++
			continue
		}
		if veto != nil && veto(c.FullText) {
			out.vetoed++
			out.skipped++
			continue
		}
		heardBefore := avoidRepeats && heard.Duplicate(c.FullText, c.Text)
		wentRound := avoidWordRepeats && Stutter(c.FullText, LongestStutter) != ""
		if heardBefore || wentRound {
			if wentRound && !heardBefore && out.looped == nil {
				out.looped = c
			}
			wordForWord := heard.said[Normalize(c.FullText)]
			if out.spoken == nil || (fallbackWordForWord && !wordForWord) {
				out.spoken, fallbackWordForWord = c, wordForWord
			}
			out.skipped++
			continue
		}
		out.spoken = c
		return out
	}
	out.repeat = out.spoken != nil
	return out
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
		previous := ""
		if len(saidList) > 0 {
			previous = saidList[len(saidList)-1]
		}
		spoken, err := voice.Reply(previous, ReplyOptions{
			Heard: heard, Index: index, Speaker: speakers[index%len(speakers)], Mode: mode,
			MaxLength: opts.MaxLength, Context: opts.Context, Temperature: opts.Temperature, K: opts.K,
			Beam: opts.Beam, StepPenalty: opts.StepPenalty, RNG: rng, AvoidRepeats: opts.AvoidRepeats,
			AvoidWordRepeats: opts.AvoidWordRepeats, Explore: opts.Explore, Veto: opts.Veto,
		})
		if err != nil {
			return nil, err
		}
		if spoken == nil {
			break
		}
		if spoken.Repeat && repeated[Normalize(spoken.Text)] {
			break // the voice can only say a duplicate it has already repeated: the conversation is over
		}
		result = append(result, spoken)
		saidList = append(saidList, spoken.Text)
		reply := ""
		if spoken.Context != "" {
			reply = spoken.Reply
		}
		heard.Remember(spoken.Text, reply)
		if spoken.Repeat {
			repeated[Normalize(spoken.Text)] = true
		}
		index++
	}
	return result, nil
}

// ReplyOptions are how one voice finds what to say next.
type ReplyOptions struct {
	// Heard is what the conversation has already heard, so a reply does not duplicate it; remember a turn in
	// it before asking for the next one, or the same reply comes back.
	Heard        *Heard
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
	// AvoidWordRepeats keeps a reply from repeating its own words (a Stutter).
	AvoidWordRepeats bool
	// Explore is how many times a reply that caught itself repeating may back up (see ConverseOptions.Explore).
	Explore int
	// Veto is what the speaker may not say (see ConverseOptions.Veto).
	Veto func(string) bool
}

// DefaultReplyOptions mirror the Python defaults.
func DefaultReplyOptions() ReplyOptions {
	return ReplyOptions{Speaker: "B", Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1, K: 5,
		AvoidRepeats: true, AvoidWordRepeats: true, Explore: Explore}
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
	heard := o.Heard
	if heard == nil {
		heard = NewHeard([]string{previous})
	}
	speaker := o.Speaker
	if speaker == "" {
		speaker = "B"
	}
	ctx := TailContext(previous, o.Context)
	var spoken *PathResult
	var rethought *Rethink
	offered, skipped, vetoed := 0, 0, 0
	repeat := false
	draws := 1
	if mode == "sample" {
		draws = o.K
	}
	// A candidate rejected only for repeating its own words is worth backing out of: keep what it said before
	// the walk went round and look for another way on, once per turn.
	thinkAgain := func(p picked, keep string) (*PathResult, bool, error) {
		if p.looped == nil || o.Explore <= 0 || rethought != nil {
			return p.spoken, p.repeat, nil
		}
		found, record, err := m.Backtrack(p.looped.FullText, BacktrackOptions{
			Keep: keep, Heard: heard, Explore: o.Explore, Mode: mode, K: o.K, Beam: o.Beam,
			MaxLength: o.MaxLength, StepPenalty: o.StepPenalty, Temperature: o.Temperature, RNG: o.RNG,
			AvoidRepeats: o.AvoidRepeats, Veto: o.Veto,
		})
		if err != nil {
			return nil, false, err
		}
		rethought = record
		offered += record.Explored
		if found != nil {
			return found, false, nil
		}
		return p.spoken, p.repeat, nil
	}
	for ctx != "" {
		if m.usable(ctx) {
			for d := 0; d < draws; d++ {
				cands, err := m.candidates(ctx, mode, o.K, o.Beam, o.MaxLength, o.StepPenalty, o.Temperature, o.RNG)
				if err != nil {
					return nil, err
				}
				offered += len(cands)
				p := pick(cands, heard, o.AvoidRepeats, o.Veto, o.AvoidWordRepeats)
				skipped += p.skipped
				vetoed += p.vetoed
				if spoken, repeat, err = thinkAgain(p, ctx); err != nil {
					return nil, err
				}
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
			p := pick(cands, heard, o.AvoidRepeats, o.Veto, o.AvoidWordRepeats)
			skipped += p.skipped
			vetoed += p.vetoed
			if freshPick, freshRepeat, err = thinkAgain(p, ""); err != nil {
				return nil, err
			}
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
	// FullText is the whole utterance whatever it was continued from - the context, or the words a rethink kept
	text := spoken.FullText
	return &Turn{Index: o.Index, Speaker: speaker, Text: text, Context: ctx, Reply: text[len(ctx):],
		Cost: spoken.Cost, Probability: spoken.Probability(), ReachedEnd: spoken.ReachedEnd, Fresh: ctx == "",
		Repeat: repeat, Stutter: Stutter(text, LongestStutter) != "", Rethink: rethought,
		Candidates: offered, Skipped: skipped, Vetoed: vetoed,
		Labels: append([]string(nil), spoken.Labels...), NodeIDs: append([]int(nil), spoken.NodeIDs...),
		StepCosts: append([]float64(nil), spoken.StepCosts...)}, nil
}
