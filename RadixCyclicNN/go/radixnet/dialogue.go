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

// A Stream is where a conversation streams what it is doing, one event at a
// time, as it happens (Python's radixnet.dialogue.StreamFn).
//
// Every event carries "event" (its kind), "index" and "speaker" (whose turn
// it is).  The committed layer is "turn" ("turn": the *Turn) - a turn is
// spoken once and never taken back, so the turn events are the answer.  The
// window between two turns is what a backtrack may still rewrite: "look"
// ("from": the context it continues, "" for a fresh text from START), "draft"
// ("text", "cost": what it was about to say before it caught itself),
// "caught" ("kind", "noticed", "cut": what it keeps, "" when it cannot back
// up), "backtrack" ("step", "cut", "wider"), "found" ("text", "cost",
// "explored") or "stuck" ("explored": nothing new from any cut).  Streaming
// changes nothing about what is said.
type Stream func(map[string]any)

// StreamEvents is every kind of event a streamed conversation emits, in the
// order a turn goes through them.
var StreamEvents = []string{"look", "draft", "caught", "backtrack", "found", "stuck", "turn"}

// tagged is stream with the turn's index and speaker written into every event.
func tagged(stream Stream, index int, speaker string) Stream {
	if stream == nil {
		return nil
	}
	return func(event map[string]any) {
		event["index"], event["speaker"] = index, speaker
		stream(event)
	}
}

// spoken is the committed layer of the stream: a turn that has been spoken,
// and will not be taken back.
func spoken(stream Stream, t *Turn) {
	if stream != nil {
		stream(map[string]any{"event": "turn", "index": t.Index, "speaker": t.Speaker, "turn": t})
	}
}

// Rethink is a voice catching itself repeating, and what it did about it.
//
// The metacognition of a turn.  Kind is what it caught: "stutter" - its own
// words, twice in a row - or "repeat", something the conversation had already
// heard.  Noticed is the words themselves, Cut what it kept of that attempt
// (everything up to where it would have started saying them again), Steps how
// many times it backed up, Explored the paths it weighed from there and Found
// whether one of them said something new.  A turn that never had to think
// twice has no record at all.
type Rethink struct {
	Kind     string `json:"kind"`
	Noticed  string `json:"noticed"`
	Cut      string `json:"cut"`
	Steps    int    `json:"steps"`
	Explored int    `json:"explored"`
	Found    bool   `json:"found"`
	// Taught is the node it taught to hand over here in future (TeachBack), or -1 when it taught nothing.
	Taught int `json:"taught"`
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

// lastWordAt is where the last word of text starts (a byte index), or -1 when
// it has fewer than two: where a voice repeating a whole utterance has to
// differ, since everything before it can be said again.
func lastWordAt(text string) int {
	start, count := -1, 0
	for i := 0; i < len(text); {
		if text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' {
			i++
			continue
		}
		start, count = i, count+1
		for i < len(text) && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r' {
			i++
		}
	}
	if count < 2 {
		return -1
	}
	return start
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

// Match is what speaking text (a reply adding reply) would repeat, or "" when
// it says something new: the utterance it was going to say again, the words an
// earlier reply already added, or the line it would only echo - what a voice
// thinking twice about it has caught itself doing.
func (h *Heard) Match(text, reply string) string {
	key := Normalize(text)
	if key == "" {
		return ""
	}
	if h.said[key] {
		return key
	}
	if added := Normalize(reply); added != "" && h.added[added] {
		return added
	}
	for _, heard := range h.keys {
		if strings.Contains(heard, key) {
			return heard
		}
	}
	return ""
}

// Duplicate reports whether speaking text (a reply adding reply) would repeat the conversation.
func (h *Heard) Duplicate(text, reply string) bool { return h.Match(text, reply) != "" }

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
	// Learn teaches the model what each rethink found out (TeachBack), so the graph itself learns where it
	// goes round.  A conversation with this on changes the model.
	Learn bool
	// Veto is what a voice may not say: true for a candidate the speaker must
	// not speak.  The conversation knows nothing about why - Filter.Converse
	// passes its own judgement in (the negative network guarding the positive
	// one), and a candidate it refuses is skipped exactly like one that had
	// been said before, except that it may not even be the fallback.
	Veto func(string) bool
	// Stream is where the conversation is streamed as it happens: a "turn"
	// event for every turn spoken, the opening included, and between them what
	// each voice does before it commits (see Stream).  The turns returned are
	// exactly the ones streamed.
	Stream Stream
}

// DefaultConverseOptions mirror the Python defaults.
func DefaultConverseOptions() ConverseOptions {
	return ConverseOptions{Turns: 6, Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1.0, K: 5,
		Speakers: DefaultSpeakers, AvoidRepeats: true, AvoidWordRepeats: true, Explore: Explore, Learn: true}
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
		walk, err := m.search(context, 0, "sample", 0, 0, stepPenalty, temperature, false, maxLength, rng, rewardWalk)
		if err != nil {
			return nil, err
		}
		return []*PathResult{&walk.PathResult}, nil
	}
	found, err := m.search(context, 0, "beam", k, beam, stepPenalty, 1.0, true, maxLength, rngNone, rewardWalk)
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

// TeachBack teaches the graph what a rethink just found out, and returns the
// node it was taught at (-1: none).
//
// A voice that has to back up has learned something no corpus could tell it:
// *this* is where its walks go round.  The node it backed up to gets an edge
// into Back (Graph.ObserveBack), the step it was about to loop through gets
// dearer, and the step it took instead - when it found one - gets cheaper.
// Nothing else in the model is touched, and the whole of it is one bump of three
// ordinary edges.
//
// From then on the search itself hands over at that node (Onward), wherever it
// is walking: the trait is the model's, not the conversation's.
func (m *Model) TeachBack(text string, at int, found *PathResult, amount float64) int {
	node, _, lead := m.prefixStart(text[:at])
	if node < First || lead != "" || !m.G.Alive[node] {
		return -1 // nothing of its own to mark: the repeat started where the graph could not place it
	}
	word := text[at:]
	if i := strings.IndexByte(word, ' '); i >= 0 {
		word = word[:i]
	}
	went, _, wentLead := m.prefixStart(text[:at+len(word)])
	if wentLead != "" || went == node {
		went = -1
	} else if _, ok := m.G.children[node].get(went); !ok {
		went = -1
	}
	instead := -1
	if found != nil && len(found.NodeIDs) > 1 {
		if _, ok := m.G.children[node].get(found.NodeIDs[1]); ok {
			instead = found.NodeIDs[1]
		}
	}
	if _, err := m.G.ObserveBack(node, went, instead, amount); err != nil {
		return -1
	}
	return node
}

// BacktrackOptions configure Backtrack.
type BacktrackOptions struct {
	// Keep is what the voice may not rewrite - the context it picked up from the other voice.
	Keep string
	// Added is what the reply would add to the context (for the repeat that is a reply adding heard words).
	Added            string
	Heard            *Heard
	Explore          int
	Mode             string
	K                int
	Beam             int
	MaxLength        int
	StepPenalty      float64
	Temperature      float64
	RNG              *MT19937
	AvoidRepeats     bool
	AvoidWordRepeats bool
	// Learn teaches the model what the rethink found out (TeachBack), so the graph itself learns where it
	// goes round.  A conversation with this on changes the model.
	Learn bool
	Veto  func(string) bool
	// Stream watches the backing up happen: "caught" the moment it notices, "backtrack" for every step
	// back, then "found" or "stuck" (see Stream).  It changes nothing about what is found.
	Stream Stream
}

// Backtrack has a voice that caught itself repeating go back to where it would
// have started saying it again, and look for another way on.
//
// text is the utterance it was about to say (o.Added the words its reply would
// add to the context), and what it caught itself doing decides where it backs
// up to: a stutter - its own words twice in a row - is cut at StutterAt, where
// the walk went round, since everything before that was said once; a repeat of
// something the conversation has heard (Heard.Match) is cut at lastWordAt,
// since the whole line is a retread and it keeps as much of it as it can and
// differs at the end.
//
// The search then runs again from the cut - a longer prefix than the turn
// started with, which forces the walk to leave the line at exactly that point;
// asking the same question again from the context would only rank the same
// answers.  Nothing found, or everything found repeats too?  Then it backs up
// one word further and looks wider, o.Explore times over.
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
	kind, noticed, at := "", "", -1
	if o.AvoidWordRepeats {
		if noticed, at = caught(text, LongestStutter); noticed != "" {
			kind = "stutter"
		}
	}
	if noticed == "" && o.AvoidRepeats {
		if noticed = heard.Match(text, o.Added); noticed != "" {
			kind, at = "repeat", lastWordAt(text)
		}
	}
	record := &Rethink{Kind: kind, Noticed: noticed, Taught: -1}
	if noticed != "" && o.Stream != nil {
		// what it will keep - "" when there is no backing up from here: nowhere to cut, the exploring off, the
		// repeat inside the words it picked up, or nothing of its own before it
		kept := ""
		if at >= 0 && o.Explore > 0 && len(text[:at]) >= len(o.Keep) && strings.TrimSpace(text[:at]) != "" {
			kept = text[:at]
		}
		o.Stream(map[string]any{"event": "caught", "kind": kind, "noticed": noticed, "cut": kept})
	}
	if noticed == "" || at < 0 || o.Explore <= 0 {
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
		if o.Stream != nil {
			o.Stream(map[string]any{"event": "backtrack", "step": step + 1, "cut": cut, "wider": wider})
		}
		cands, err := m.offer(cut, o.Mode, wider, o.Beam, o.MaxLength, o.StepPenalty, o.Temperature, o.RNG)
		if err != nil {
			return nil, record, err
		}
		for _, cand := range cands {
			record.Explored++
			if strings.TrimSpace(cand.Text) == "" || (o.Veto != nil && o.Veto(cand.FullText)) {
				continue
			}
			if (o.AvoidWordRepeats && Stutter(cand.FullText, LongestStutter) != "") ||
				(o.AvoidRepeats && heard.Duplicate(cand.FullText, cand.Text)) {
				continue
			}
			record.Found = true
			if o.Learn {
				record.Taught = m.TeachBack(text, at, cand, 1.0)
			}
			if o.Stream != nil {
				o.Stream(map[string]any{"event": "found", "text": cand.FullText, "cost": cand.Cost, "explored": record.Explored})
			}
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
	if o.Learn {
		record.Taught = m.TeachBack(text, at, nil, 1.0) // it goes round here even if it found no way out
	}
	if o.Stream != nil && record.Steps > 0 {
		o.Stream(map[string]any{"event": "stuck", "explored": record.Explored})
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
	caught  *PathResult // the best candidate rejected for repeating: the one worth backing out of
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
			if out.caught == nil {
				out.caught = c
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
		spoken(opts.Stream, result[len(result)-1])
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
		said, err := voice.Reply(previous, ReplyOptions{
			Heard: heard, Index: index, Speaker: speakers[index%len(speakers)], Mode: mode,
			MaxLength: opts.MaxLength, Context: opts.Context, Temperature: opts.Temperature, K: opts.K,
			Beam: opts.Beam, StepPenalty: opts.StepPenalty, RNG: rng, AvoidRepeats: opts.AvoidRepeats,
			AvoidWordRepeats: opts.AvoidWordRepeats, Explore: opts.Explore, Learn: opts.Learn, Veto: opts.Veto,
			Stream: opts.Stream,
		})
		if err != nil {
			return nil, err
		}
		if said == nil {
			break
		}
		if said.Repeat && repeated[Normalize(said.Text)] {
			break // the voice can only say a duplicate it has already repeated: the conversation is over
		}
		result = append(result, said)
		spoken(opts.Stream, said)
		saidList = append(saidList, said.Text)
		reply := ""
		if said.Context != "" {
			reply = said.Reply
		}
		heard.Remember(said.Text, reply)
		if said.Repeat {
			repeated[Normalize(said.Text)] = true
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
	// Learn teaches the model what each rethink found out (see ConverseOptions.Learn).
	Learn bool
	// Veto is what the speaker may not say (see ConverseOptions.Veto).
	Veto func(string) bool
	// Stream watches the turn being found (see Stream): "look" for every context it continues (and "" for
	// a fresh text), then - when a candidate is caught repeating - "draft" and what Backtrack does about
	// it.  The turn itself is not this function's event: it is the caller's to speak (Converse streams it
	// as "turn"), since a reply may still be refused for repeating a duplicate already repeated.
	Stream Stream
}

// DefaultReplyOptions mirror the Python defaults.
func DefaultReplyOptions() ReplyOptions {
	return ReplyOptions{Speaker: "B", Mode: "beam", MaxLength: 60, Context: 12, Temperature: 1, K: 5,
		AvoidRepeats: true, AvoidWordRepeats: true, Explore: Explore, Learn: true}
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
	watch := tagged(o.Stream, o.Index, speaker)
	// A candidate rejected for repeating - its own words, or the conversation's - is worth backing out of:
	// keep what it said up to the repetition and look for another way on, once per turn.
	thinkAgain := func(p picked, keep string) (*PathResult, bool, error) {
		if p.caught == nil || o.Explore <= 0 || rethought != nil {
			return p.spoken, p.repeat, nil
		}
		if watch != nil {
			watch(map[string]any{"event": "draft", "text": p.caught.FullText, "cost": p.caught.Cost})
		}
		found, record, err := m.Backtrack(p.caught.FullText, BacktrackOptions{
			Keep: keep, Added: p.caught.Text, Heard: heard, Explore: o.Explore, Mode: mode, K: o.K, Beam: o.Beam,
			MaxLength: o.MaxLength, StepPenalty: o.StepPenalty, Temperature: o.Temperature, RNG: o.RNG,
			AvoidRepeats: o.AvoidRepeats, AvoidWordRepeats: o.AvoidWordRepeats, Learn: o.Learn, Veto: o.Veto,
			Stream: watch,
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
			if watch != nil {
				watch(map[string]any{"event": "look", "from": ctx})
			}
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
		if watch != nil {
			watch(map[string]any{"event": "look", "from": ""})
		}
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
