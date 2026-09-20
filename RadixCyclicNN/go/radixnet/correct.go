package radixnet

import "math"

// CorrectOptions weight the two halves of a correction.
type CorrectOptions struct {
	// Strength is the magnitude of one unit of feedback (0 = the default 1).
	Strength float64
	// Weight is how bad the attempt was: the penalty is Strength * Weight.
	Weight float64
	// Reward is what the correction is worth: the fix gains Strength * Reward.
	Reward float64
	// Keep is what the unchanged part of the correction still earns, as a
	// fraction of Reward.  It is 0 by default: a whole path is rewarded when
	// the *output* was correct (Model.Reward), not when it had to be corrected.
	Keep float64
	// NoCount leaves the correction's traversals uncounted (they are counted by default:
	// a corrected sentence is correct English whatever changed).
	NoCount bool
}

// DefaultCorrectOptions: one unit of feedback either way, a quarter of it for the words that did not change.
func DefaultCorrectOptions() CorrectOptions {
	return CorrectOptions{Strength: 1, Weight: 1, Reward: 1, Keep: 0}
}

// Correction reports what one taught correction moved.
type Correction struct {
	Edits           int     `json:"edits"`
	Changes         []Edit  `json:"changes"`
	Penalised       int     `json:"penalised"`
	Rewarded        int     `json:"rewarded"`
	Kept            int     `json:"kept"`
	Penalty         float64 `json:"penalty"`
	Reward          float64 `json:"reward"`
	Loss            float64 `json:"loss"`
	WrongChars      int     `json:"wrong_chars"`
	RightChars      int     `json:"right_chars"`
	MarkedCorrect   int     `json:"marked_correct"`
	MarkedIncorrect int     `json:"marked_incorrect"`
}

// Correct teaches one correction: it moves only the trigram nodes the two
// sentences disagree on.
//
// wrong is what the network wrote, right what the teacher wrote instead.  The
// two are aligned character by character (Edits) and every step of either path
// is charged with the characters it adds, so:
//
//   - the steps of wrong that wrote a character the teacher struck out or
//     replaced lose Strength * Weight of reward - and only those: the words
//     both sentences agree on keep what they earned;
//   - the steps of right that wrote what the teacher put there instead gain
//     Strength * Reward; the rest of the correction earns Keep times as much,
//     and Keep is 0 by default - a whole path is rewarded when the output was
//     correct, not when it had to be corrected;
//   - the correction is traversed once, as a training pass does, unless NoCount.
//
// An edge both sentences walk over a changed span - the network wrote the
// right characters by another route - is rewarded, never penalised.
func (m *Model) Correct(wrong, right string, o CorrectOptions) (*Correction, error) {
	// the encoding aligns the two: character by character by default, word by
	// word under a word encoding (Encoding.Edits), so there is nothing to
	// translate on either side of it
	return m.correct(wrong, right, o)
}

// correct is Correct over the graph's own symbols, whatever they stand for.
func (m *Model) correct(wrong, right string, o CorrectOptions) (*Correction, error) {
	if o.Strength <= 0 {
		o.Strength = 1
	}
	base := math.Abs(o.Strength)
	enc := m.Encoding()
	wrongSpans, rightSpans := enc.ChangedSpans(wrong, right)
	out := &Correction{Changes: enc.DiffSummary(wrong, right, 8), Edits: len(enc.DiffSummary(wrong, right, 0))}
	for _, s := range wrongSpans {
		out.WrongChars += s.Hi - s.Lo
	}
	for _, s := range rightSpans {
		out.RightChars += s.Hi - s.Lo
	}
	g := m.G
	wrongGrams, rightGrams := enc.Encode(wrong), enc.Encode(right)
	if wrongGrams == nil && rightGrams == nil {
		return out, nil
	}
	// both sentences join the structure before either is measured: observing one can split a node the
	// other's path runs through, and the split moves the very edge a penalty was meant for
	for _, grams := range [][]string{wrongGrams, rightGrams} {
		if grams == nil {
			continue
		}
		if _, ok := g.NodePath(grams); !ok {
			if _, err := g.ObserveSequence(grams, false); err != nil {
				return nil, err
			}
		}
	}
	penalties := map[int]float64{}
	order := []int{}
	blamed := []PathKey{}
	if wrongGrams != nil && len(wrongSpans) > 0 && base*o.Weight > 0 {
		blamed = m.stepsOver(wrongGrams, enc.Len(wrong), wrongSpans)
		for _, step := range blamed {
			if _, seen := penalties[step.Edge]; !seen {
				order = append(order, step.Edge)
			}
			penalties[step.Edge] = -base * o.Weight
		}
	}
	rewards := map[int]float64{}
	rewardOrder := []int{}
	fixed := map[int]bool{}
	taught := []PathKey{}
	if rightGrams != nil {
		transitions, err := g.ObserveSequence(rightGrams, !o.NoCount)
		if err != nil {
			return nil, err
		}
		if !o.NoCount {
			g.RecordTraversals(edgesOf(transitions))
			g.RecordPath(transitions, PathUnjudged, false) // the correction's own traffic
			m.metaAddInt("trained_texts", 1)
			m.metaAddInt("trained_chars", int64(enc.Len(right)))
		}
		if len(rightSpans) > 0 {
			taught = m.stepsOver(rightGrams, enc.Len(right), rightSpans)
			for _, step := range taught {
				fixed[step.Edge] = true
			}
		}
		for _, t := range transitions {
			share := o.Keep
			if fixed[t.E] {
				share = 1
			}
			amount := base * o.Reward * share
			if amount <= 0 {
				continue
			}
			if _, seen := rewards[t.E]; !seen {
				rewardOrder = append(rewardOrder, t.E)
			}
			rewards[t.E] = amount
		}
		g.Prepare()
		out.Loss = meanCost(g, transitions)
	}
	for _, e := range order {
		if _, both := rewards[e]; both { // the teacher wrote it too: it is not the mistake
			continue
		}
		g.AddReward([]int{e}, penalties[e])
		out.Penalised++
		out.Penalty += -penalties[e]
	}
	for _, e := range rewardOrder {
		g.AddReward([]int{e}, rewards[e])
		if fixed[e] {
			out.Rewarded++
		} else {
			out.Kept++
		}
		out.Reward += rewards[e]
	}
	// the counters follow the reward: what was blamed is a wrong path here, what was taught a right one
	stillBlamed := blamed[:0:0]
	for _, step := range blamed {
		if _, both := rewards[step.Edge]; !both {
			stillBlamed = append(stillBlamed, step)
		}
	}
	out.MarkedIncorrect = g.MarkSteps(stillBlamed, false)
	out.MarkedCorrect = g.MarkSteps(taught, true)
	if out.Penalised > 0 || out.Rewarded > 0 || out.Kept > 0 {
		m.metaAddInt("feedback_passes", 1)
		m.metaAddFloat("rewards_total", out.Reward)
		m.metaAddFloat("penalties_total", out.Penalty)
		g.Prepare()
	}
	return out, nil
}

// stepsOver returns the edges of a traced text whose step wrote a unit inside
// one of the spans.
//
// Every step is charged with the units it adds to the text: the first with the
// whole of its node's label, a later one with everything past the units it
// overlaps its parent by, and the step into END with the position just past
// the last unit - where a sentence that stopped too early went wrong.
func (m *Model) stepsOver(grams []string, length int, spans []Span) []PathKey {
	g := m.G
	overlap := g.Enc.Overlap()
	path, ok := g.NodePath(grams)
	if !ok || len(path) < 2 {
		return nil
	}
	out := []PathKey{}
	position := 0 // trigram index of the node being entered
	for index := 1; index < len(path); index++ {
		node := path[index]
		prev := Start // who called the step: START begins every walk
		if index >= 2 {
			prev = path[index-2]
		}
		e, has := g.Edge(path[index-1], node)
		if node == End {
			if has && spansTouch(length, length+1, spans) {
				out = append(out, PathKey{prev, e})
			}
			break
		}
		size := g.LabelLen(node)
		lo := position + overlap
		if index == 1 {
			lo = 0
		}
		if has && spansTouch(lo, position+size, spans) {
			out = append(out, PathKey{prev, e})
		}
		position += size - overlap
	}
	return out
}

// meanCost is the mean -log P of a path's transitions under the current weights.
func meanCost(g *Graph, transitions []Transition) float64 {
	if len(transitions) == 0 {
		return 0
	}
	total := 0.0
	for _, t := range transitions {
		total += g.EdgeCost(t.E)
	}
	return total / float64(len(transitions))
}

func edgesOf(transitions []Transition) []int {
	out := make([]int, len(transitions))
	for i, t := range transitions {
		out[i] = t.E
	}
	return out
}
