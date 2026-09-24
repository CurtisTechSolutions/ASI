package radixnet

// How a training run walks its texts: the order, the curriculum, replay and
// early stopping (../../SPEC-SearchAndTraining.md §3-6).
//
// The Go twin of radixnet/training.py, kept number for number: the shuffle and
// the replay buffer order their texts by SplitMix64 keys of the model's seed,
// so no random number is drawn and the model's own generator stays where it
// was; every rounding and every tie is the one the spec names.

import (
	"fmt"
	"math"
	"sort"
)

// Orders are the orders a run can walk its texts in; "corpus" is the one it
// always had.
var Orders = []string{"corpus", "shortest-first", "longest-first", "shuffle"}

const (
	shuffleSalt uint64 = 0x53485546464C45 // "SHUFFLE"
	replaySalt  uint64 = 0x5245504C4159   // "REPLAY"
)

// SplitMix64 is the SplitMix64 finaliser (wrapping arithmetic).
func SplitMix64(x uint64) uint64 {
	z := x + 0x9E3779B97F4A7C15
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// ShuffleKey is where text i goes in the shuffled order of epoch number
// epoch (smallest first).
func ShuffleKey(seed int64, epoch int64, i int) uint64 {
	return SplitMix64(SplitMix64(SplitMix64(uint64(seed)^shuffleSalt)^uint64(epoch)) ^ uint64(i))
}

// ReplayKey is the priority of the g-th text ever offered to a replay buffer
// (the smallest are kept).
func ReplayKey(seed int64, g int64) uint64 {
	return SplitMix64(SplitMix64(uint64(seed)^replaySalt) ^ uint64(g))
}

// Plan are the settings of how a training run walks its texts, each off in
// its zero value: Order "" is "corpus", Curriculum 0 is 1, ReplaySize nil
// leaves the model's buffer alone.
type Plan struct {
	Order      string
	Curriculum float64
	Replay     float64
	ReplaySize *int
	Patience   int
	MinDelta   float64
}

func (p Plan) order() string {
	if p.Order == "" {
		return "corpus"
	}
	return p.Order
}

func (p Plan) curriculum() float64 {
	if p.Curriculum == 0 {
		return 1
	}
	return p.Curriculum
}

// NeedsTexts is whether the run needs the whole list before the first epoch -
// a non-corpus order, a curriculum, replay, or a buffer to offer the texts to
// - and so reads a streaming source into memory.
func (p Plan) NeedsTexts() bool {
	return p.order() != "corpus" || p.curriculum() < 1 || p.Replay > 0 || p.ReplaySize != nil
}

// Active is whether anything about the run differs from the plain pass.
func (p Plan) Active() bool { return p.NeedsTexts() || p.Patience > 0 }

// Check is the error for a setting out of range (the spec's §7).
func (p Plan) Check() error {
	known := false
	for _, o := range Orders {
		if p.order() == o {
			known = true
		}
	}
	if !known {
		return fmt.Errorf("unknown order %q; expected one of: corpus, shortest-first, longest-first, shuffle", p.Order)
	}
	if c := p.curriculum(); !(c > 0 && c <= 1) {
		return fmt.Errorf("curriculum must lie in (0, 1], got %v", p.Curriculum)
	}
	if !(p.Replay >= 0) || math.IsInf(p.Replay, 0) {
		return fmt.Errorf("replay must be a finite number >= 0, got %v", p.Replay)
	}
	if p.ReplaySize != nil && *p.ReplaySize < 0 {
		return fmt.Errorf("replay_size must be >= 0, got %d", *p.ReplaySize)
	}
	if p.Patience < 0 {
		return fmt.Errorf("patience must be >= 0, got %d", p.Patience)
	}
	if !(p.MinDelta >= 0) || math.IsInf(p.MinDelta, 0) {
		return fmt.Errorf("min_delta must be a finite number >= 0, got %v", p.MinDelta)
	}
	return nil
}

// Paced is how many of the list epoch j (from 0) of epochs walks: the first
// curriculum of it, growing to all of it by the last epoch.
func Paced(n int, curriculum float64, j, epochs int) int {
	if n <= 0 {
		return 0
	}
	if curriculum >= 1 || epochs <= 1 {
		return n
	}
	frac := curriculum + ((1-curriculum)*float64(j))/float64(epochs-1)
	m := int(math.Ceil(frac * float64(n)))
	if m < 1 {
		m = 1
	}
	if m > n {
		m = n
	}
	return m
}

// ReplayCount is how many buffered texts an epoch rehearses: replay of the
// run's n texts, rounded half up, at most the pool.
func ReplayCount(replay float64, n, pool int) int {
	if replay <= 0 || pool <= 0 {
		return 0
	}
	r := math.Floor(replay*float64(n) + 0.5)
	if r >= float64(pool) {
		return pool // compared as a float: a huge share must not overflow the conversion
	}
	return int(r)
}

type replayItem struct {
	key  uint64
	g    int64
	text string
}

func replayLess(a, b replayItem) bool {
	if a.key != b.key {
		return a.key < b.key
	}
	return a.g < b.g
}

// ReplayBuffer is a bounded, uniform sample of every text a model was ever
// trained on: the g-th text ever offered gets the priority ReplayKey(seed, g)
// and the buffer keeps the Size entries with the smallest (priority, g) -
// bottom-k sampling, with no random number drawn.
type ReplayBuffer struct {
	Seed  int64
	Size  int
	Seen  int64
	items []replayItem
}

// NewReplayBuffer is a buffer holding entries (g, text), priorities
// recomputed from seed and only the Size smallest kept.
func NewReplayBuffer(size int, seed int64, seen int64, index []int64, texts []string) *ReplayBuffer {
	b := &ReplayBuffer{Seed: seed, Size: size, Seen: seen}
	for i, g := range index {
		b.items = append(b.items, replayItem{ReplayKey(seed, g), g, texts[i]})
	}
	sort.Slice(b.items, func(i, j int) bool { return replayLess(b.items[i], b.items[j]) })
	if len(b.items) > size {
		b.items = b.items[:size]
	}
	return b
}

// Len is how many texts the buffer holds.
func (b *ReplayBuffer) Len() int { return len(b.items) }

// Texts are the kept texts in priority order - the order an epoch rehearses
// them in.
func (b *ReplayBuffer) Texts() []string {
	out := make([]string, len(b.items))
	for i, it := range b.items {
		out[i] = it.text
	}
	return out
}

// Offer offers texts in order; each is kept while it is among the Size
// smallest priorities.
func (b *ReplayBuffer) Offer(texts []string) {
	for _, text := range texts {
		g := b.Seen
		b.Seen++
		if b.Size == 0 {
			continue
		}
		item := replayItem{ReplayKey(b.Seed, g), g, text}
		if len(b.items) >= b.Size {
			if !replayLess(item, b.items[len(b.items)-1]) {
				continue
			}
			b.items = b.items[:len(b.items)-1]
		}
		at := sort.Search(len(b.items), func(i int) bool { return replayLess(item, b.items[i]) })
		b.items = append(b.items, replayItem{})
		copy(b.items[at+1:], b.items[at:])
		b.items[at] = item
	}
}

// Resize sets a new capacity; a smaller one drops the entries with the
// largest priorities.
func (b *ReplayBuffer) Resize(size int) {
	b.Size = size
	if len(b.items) > size {
		b.items = b.items[:size]
	}
}

// ReplayDoc is the model file's "replay" block: {size, seen, index, texts},
// entries in priority order.
type ReplayDoc struct {
	Size  int      `json:"size"`
	Seen  int64    `json:"seen"`
	Index []int64  `json:"index"`
	Texts []string `json:"texts"`
}

// Doc is the buffer as the file carries it.
func (b *ReplayBuffer) Doc() *ReplayDoc {
	d := &ReplayDoc{Size: b.Size, Seen: b.Seen, Index: make([]int64, len(b.items)), Texts: make([]string, len(b.items))}
	for i, it := range b.items {
		d.Index[i], d.Texts[i] = it.g, it.text
	}
	return d
}

// ReplayFromDoc reads a "replay" block; the priorities are recomputed from
// seed, not trusted.
func ReplayFromDoc(d *ReplayDoc, seed int64) (*ReplayBuffer, error) {
	if len(d.Index) != len(d.Texts) {
		return nil, fmt.Errorf("replay block has %d indices for %d texts", len(d.Index), len(d.Texts))
	}
	return NewReplayBuffer(d.Size, seed, d.Seen, d.Index, d.Texts), nil
}

// EarlyStop stops a run that has stopped improving: Patience epochs without a
// loss MinDelta below the best.
type EarlyStop struct {
	Patience int
	MinDelta float64
	best     float64
	stale    int
}

// NewEarlyStop is a fresh stopper.
func NewEarlyStop(patience int, minDelta float64) *EarlyStop {
	return &EarlyStop{Patience: patience, MinDelta: minDelta, best: math.Inf(1)}
}

// Update takes one epoch's loss; true when the run should stop after it.
func (e *EarlyStop) Update(loss float64) bool {
	if loss < e.best-e.MinDelta {
		e.best = loss
		e.stale = 0
	} else {
		e.stale++
	}
	return e.Patience > 0 && e.stale >= e.Patience
}

// TrainingPlan is which texts each epoch of one Train call walks, what it
// rehearses, and when it stops.  Texts are the run's usable texts (the ones
// too short for a gram already dropped); the buffer is read as the run
// begins, and Finish hands back the one the model keeps afterwards.
type TrainingPlan struct {
	Texts    []string
	Skipped  int
	seed     int64
	epochs   int
	order    string
	c        float64
	size     *int
	buffer   *ReplayBuffer
	pool     []string
	rehearse int
	stopper  *EarlyStop
	base     []int
}

// NewTrainingPlan builds the plan of one run over texts (lengths in the
// encoding's units) for a model of seed with buffer.
func NewTrainingPlan(texts []string, lengths []int, seed int64, epochs int, p Plan, buffer *ReplayBuffer) (*TrainingPlan, error) {
	if err := p.Check(); err != nil {
		return nil, err
	}
	t := &TrainingPlan{Texts: texts, seed: seed, epochs: epochs, order: p.order(), c: p.curriculum(),
		size: p.ReplaySize, buffer: buffer, stopper: NewEarlyStop(p.Patience, p.MinDelta)}
	if buffer != nil {
		t.pool = buffer.Texts()
	}
	t.rehearse = ReplayCount(p.Replay, len(texts), len(t.pool))
	t.base = make([]int, len(texts))
	for i := range t.base {
		t.base[i] = i
	}
	switch t.order {
	case "shortest-first":
		sort.SliceStable(t.base, func(a, b int) bool { return lengths[t.base[a]] < lengths[t.base[b]] })
	case "longest-first":
		sort.SliceStable(t.base, func(a, b int) bool { return lengths[t.base[a]] > lengths[t.base[b]] })
	}
	return t, nil
}

// Plain is whether every epoch walks every text in corpus order and
// rehearses nothing - the pass it always was.
func (t *TrainingPlan) Plain() bool { return t.order == "corpus" && t.c >= 1 && t.rehearse == 0 }

// Replayed are the buffered texts the structure pass observes after the
// run's own: the whole pool, when any is rehearsed.
func (t *TrainingPlan) Replayed() []string {
	if t.rehearse == 0 {
		return nil
	}
	return append([]string(nil), t.pool...)
}

// Walked is how many of the run's texts epoch j walks.
func (t *TrainingPlan) Walked(j int) int { return Paced(len(t.Texts), t.c, j, t.epochs) }

// Epoch is the texts epoch j (from 0), whose record will carry epoch number,
// walks: the chosen texts of the run in order, then the rehearsed ones.
func (t *TrainingPlan) Epoch(j int, number int64) []string {
	order := t.base
	if t.order == "shuffle" {
		keys := make([]uint64, len(t.Texts))
		for i := range keys {
			keys[i] = ShuffleKey(t.seed, number, i)
		}
		order = make([]int, len(t.Texts))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			ka, kb := keys[order[a]], keys[order[b]]
			if ka != kb {
				return ka < kb
			}
			return order[a] < order[b]
		})
	}
	walked := t.Walked(j)
	out := make([]string, 0, walked+t.rehearse)
	for _, i := range order[:walked] {
		out = append(out, t.Texts[i])
	}
	for i := 0; i < t.rehearse; i++ {
		out = append(out, t.pool[(j*t.rehearse+i)%len(t.pool)])
	}
	return out
}

// Stop is whether the run stops after epoch j, given its loss; an epoch
// inside the curriculum does not count.
func (t *TrainingPlan) Stop(j int, loss float64) bool {
	if t.Walked(j) < len(t.Texts) {
		return false
	}
	return t.stopper.Update(loss)
}

// Finish is the replay buffer the model keeps after this run: resized if
// asked, with the run's texts offered.
func (t *TrainingPlan) Finish() *ReplayBuffer {
	buffer := t.buffer
	if t.size != nil {
		if *t.size == 0 {
			return nil
		}
		if buffer == nil {
			buffer = NewReplayBuffer(*t.size, t.seed, 0, nil, nil)
		} else {
			buffer.Resize(*t.size)
		}
	}
	if buffer != nil {
		buffer.Offer(t.Texts)
	}
	return buffer
}
