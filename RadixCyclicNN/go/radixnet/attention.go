package radixnet

import (
	"fmt"
	"math"
	"sort"
)

// The attention band: where inside an n-gram a correction's blame and credit
// land (radixnet/attention.py, ../../SPEC-AttentionBand.md).
//
// A reader's eye fixes on one point of a line and sees it sharply, and the
// letters either side of it blur with the distance.  A gram is this model's
// fixation, and the band is how sharply each of its N positions is seen: 1 at
// the centre, falling off in a straight line to 1 - Blur at the first and the
// last unit.
//
// Off (the zero value, and every model before it existed), each unit a
// correction changes is charged in full to the step that wrote it - the step
// whose gram ends on it.  On, each changed unit hands out exactly one charge,
// shared among the grams that see it in proportion to how sharply each sees
// it, so the gram with the change at its centre takes the most; a step is
// charged what its grams collected, never more than one full charge, and the
// judged-path verdict goes to the step that saw the change most sharply.  A
// judgement of a whole text marks every unit alike, and no band changes it.

// DefaultBlur is what switching the band on means when no blur is given: the
// ends of a gram seen half as sharply as its centre.
const DefaultBlur = 0.5

// AttentionBand is a model's band: off, or on with a Blur in [0, 1].  Unlike
// the encoding it changes nothing the graph holds, so it can be switched at
// any time; it travels with the model file while it is on.
type AttentionBand struct {
	On   bool
	Blur float64
}

// CheckBlur is the one rule for a blur: a finite number in [0, 1].
func CheckBlur(blur float64) error {
	if math.IsNaN(blur) || math.IsInf(blur, 0) || blur < 0 || blur > 1 {
		return fmt.Errorf("blur must lie in [0, 1], got %v", blur)
	}
	return nil
}

// BandWeights is how sharply each of a gram's n positions is seen: 1 at the
// centre, 1 - blur at both ends, linear between.  A gram of one is all centre
// and a gram of two all ends, so both are flat.  Each weight is
// 1 - blur*(d/c) with c = (n-1)/2 and d the distance from it; the conversion
// rounds the product on its own, so no fused multiply-add can make these
// bits differ from Python's and Rust's.
func BandWeights(n int, blur float64) []float64 {
	if n <= 1 {
		return []float64{1}
	}
	centre := float64(n-1) / 2
	out := make([]float64, n)
	for j := range out {
		d := math.Abs(float64(j) - centre)
		out[j] = 1 - float64(blur*(d/centre))
	}
	return out
}

// Weights is the band over one gram of n units, or nil while it is off.
func (b AttentionBand) Weights(n int) []float64 {
	if !b.On {
		return nil
	}
	return BandWeights(n, b.Blur)
}

// BlurOrNil is the blur a report carries: the number, or nil (JSON null) while the band is off.
func (b AttentionBand) BlurOrNil() any {
	if !b.On {
		return nil
	}
	return b.Blur
}

// String is "off" or "blur 0.5".
func (b AttentionBand) String() string {
	if !b.On {
		return "off"
	}
	return fmt.Sprintf("blur %g", b.Blur)
}

// Describe is the human form, with the band over a gram of n units.
func (b AttentionBand) Describe(n int) string {
	if !b.On {
		return "off: each changed unit is charged to the step that wrote it"
	}
	text := fmt.Sprintf("on, blur %g: the centre of each gram is charged most (", b.Blur)
	for j, w := range BandWeights(n, b.Blur) {
		if j > 0 {
			text += " "
		}
		text += fmt.Sprintf("%g", w)
	}
	return text + ")"
}

// JudgedUnits are the units a correction marks, ascending, within [0, length]:
// every unit of every span, and for an empty span - an insertion point - the
// unit it stands in front of.  length itself is the position after the last
// unit, which only the step into END answers for.
func JudgedUnits(spans []Span, length int) []int {
	marked := map[int]bool{}
	for _, s := range spans {
		top := s.Hi
		if top <= s.Lo {
			top = s.Lo + 1
		}
		if top > length+1 {
			top = length + 1
		}
		lo := s.Lo
		if lo < 0 {
			lo = 0
		}
		for u := lo; u < top; u++ {
			marked[u] = true
		}
	}
	out := make([]int, 0, len(marked))
	for u := range marked {
		out = append(out, u)
	}
	sort.Ints(out)
	return out
}

// Spread is what SpreadCharges hands each gram of a text.
type Spread struct {
	// Shares is, per gram, the charge it collected before any cap.
	Shares []float64
	// Focus is, per gram, whether it sees some marked unit most sharply of all
	// the grams that see that unit.
	Focus []bool
	// End is whether the position after the last unit is marked: the step
	// into END answers for it, in full.
	End bool
}

// SpreadCharges shares every marked unit out over the grams that see it, in
// proportion to the band: gram g covers [g*stride, g*stride+n) and sees unit u
// at position u - g*stride.  Each marked unit hands out exactly one charge -
// weight/total to each gram that sees it, or an even split when every one of
// them sees it at a weight of 0.  The order is part of the definition - units
// ascending, the grams that see one ascending, every sum left to right - so
// the three implementations add the same floats in the same order.
func SpreadCharges(n, stride, grams, length int, spans []Span, weights []float64) Spread {
	out := Spread{Shares: make([]float64, grams), Focus: make([]bool, grams)}
	for _, unit := range JudgedUnits(spans, length) {
		if unit >= length {
			out.End = true
			continue
		}
		lo := 0
		if unit >= n {
			lo = (unit-n)/stride + 1
		}
		hi := unit / stride
		if hi > grams-1 {
			hi = grams - 1
		}
		if lo > hi {
			continue // no gram covers it: the tail a grouping encoding drops
		}
		total, best := 0.0, 0.0
		for g := lo; g <= hi; g++ {
			w := weights[unit-g*stride]
			total += w
			if w > best {
				best = w
			}
		}
		viewers := float64(hi - lo + 1)
		for g := lo; g <= hi; g++ {
			w := weights[unit-g*stride]
			if total > 0 {
				out.Shares[g] += w / total
			} else {
				out.Shares[g] += 1 / viewers
			}
			if w == best {
				out.Focus[g] = true
			}
		}
	}
	return out
}

// WriterMarks is the rule with the band off, gram by gram: does gram g write a
// marked unit (the first gram writes all of its units, every later one the
// stride units past the overlap), and is the end marked.
func WriterMarks(n, stride, grams, length int, spans []Span) ([]bool, bool) {
	marked := JudgedUnits(spans, length)
	overlap := n - stride
	out := make([]bool, grams)
	for g := 0; g < grams; g++ {
		lo, hi := g*stride+overlap, g*stride+n
		if g == 0 {
			lo = 0
		}
		for _, u := range marked {
			if lo <= u && u < hi {
				out[g] = true
				break
			}
		}
	}
	end := false
	for _, u := range marked {
		if u == length {
			end = true
		}
	}
	return out, end
}

// ChargedStep is one step of a traced text a correction charges: who called
// it and the edge, the charge (at most 1) and whether it is the focus - the
// step that sees a changed unit most sharply, and so the one the judged-path
// verdict goes to.
type ChargedStep struct {
	PathKey
	Charge float64
	Focus  bool
}

// chargedSteps are the steps of a traced text a correction charges, in path
// order.  Off, they are stepsOver's, each charged 1 and each the focus; on,
// every changed unit is shared out over the grams that see it and a step is
// charged what the grams of its node collected, capped at 1.  length is the
// text's length in units.
func (m *Model) chargedSteps(grams []string, length int, spans []Span) []ChargedStep {
	g := m.G
	if !g.Attention.On {
		steps := m.stepsOver(grams, length, spans)
		out := make([]ChargedStep, len(steps))
		for i, step := range steps {
			out[i] = ChargedStep{PathKey: step, Charge: 1, Focus: true}
		}
		return out
	}
	path, ok := g.NodePath(grams)
	if !ok || len(path) < 2 {
		return nil
	}
	n, stride := g.Enc.N, g.Enc.Stride
	shared := SpreadCharges(n, stride, len(grams), length, spans, g.Attention.Weights(n))
	out := []ChargedStep{}
	gram := 0 // the text's first gram inside the node being entered
	for index := 1; index < len(path); index++ {
		node := path[index]
		prev := Start
		if index >= 2 {
			prev = path[index-2]
		}
		e, has := g.Edge(path[index-1], node)
		if node == End {
			if has && shared.End {
				out = append(out, ChargedStep{PathKey: PathKey{prev, e}, Charge: 1, Focus: true})
			}
			break
		}
		held := (g.LabelLen(node)-n)/stride + 1 // the grams a node of this length holds
		charge, focus := 0.0, false
		for w := gram; w < gram+held && w < len(grams); w++ {
			charge += shared.Shares[w]
			focus = focus || shared.Focus[w]
		}
		gram += held
		if has && charge > 0 {
			out = append(out, ChargedStep{PathKey: PathKey{prev, e}, Charge: math.Min(1, charge), Focus: focus})
		}
	}
	return out
}

// AttentionConfig is the band as the API, the CLI and the frontend show it.
type AttentionConfig struct {
	On          bool      `json:"on"`
	Blur        *float64  `json:"blur"`
	Weights     []float64 `json:"weights"`
	Ngram       int       `json:"ngram"`
	Stride      int       `json:"stride"`
	Unit        string    `json:"unit"`
	Units       string    `json:"units"`
	Applies     bool      `json:"applies"`
	DefaultBlur float64   `json:"default_blur"`
}

// AttentionConfig describes the model's band.  Both kinds the Go port runs -
// the count model and the negative network - learn from corrections, so the
// band always applies.
func (m *Model) AttentionConfig() AttentionConfig {
	band := m.G.Attention
	enc := m.Encoding()
	out := AttentionConfig{
		On: band.On, Weights: band.Weights(enc.N), Ngram: enc.N, Stride: enc.Stride, Unit: string(enc.Unit),
		Units: enc.UnitsName(), Applies: true, DefaultBlur: DefaultBlur,
	}
	if band.On {
		blur := band.Blur
		out.Blur = &blur
	}
	return out
}

// ConfigureAttention switches the band on (at blur, else the blur it had, else
// DefaultBlur) or off: a blur alone switches it on, on=false switches it off
// whatever blur says, and nil for both changes nothing.
func (m *Model) ConfigureAttention(on *bool, blur *float64) (AttentionConfig, error) {
	current := m.G.Attention
	switch {
	case on != nil && !*on:
		m.G.Attention = AttentionBand{}
	case (on != nil && *on) || blur != nil:
		value := DefaultBlur
		if blur != nil {
			value = *blur
		} else if current.On {
			value = current.Blur
		}
		if err := CheckBlur(value); err != nil {
			return m.AttentionConfig(), err
		}
		m.G.Attention = AttentionBand{On: true, Blur: value}
	}
	return m.AttentionConfig(), nil
}

// AttentionSide is one side of a correction, gram by gram: what the writer
// rule charges and what the band would.
type AttentionSide struct {
	Text    string    `json:"text"`
	Units   int       `json:"units"`
	Grams   []string  `json:"grams"`
	Spans   [][2]int  `json:"spans"`
	Writer  []bool    `json:"writer"`
	Charges []float64 `json:"charges"`
	Focus   []bool    `json:"focus"`
	End     bool      `json:"end"`
}

// AttentionPreview is where one correction would land, under both rules.
type AttentionPreview struct {
	Attention AttentionConfig `json:"attention"`
	Blur      float64         `json:"blur"`
	Weights   []float64       `json:"weights"`
	Changes   []Edit          `json:"changes"`
	Wrong     AttentionSide   `json:"wrong"`
	Right     AttentionSide   `json:"right"`
}

func attentionSide(enc Encoding, weights []float64, text string, spans []Span) AttentionSide {
	grams := enc.Encode(text)
	if grams == nil {
		grams = []string{}
	}
	length := enc.Len(text)
	writes, _ := WriterMarks(enc.N, enc.Stride, len(grams), length, spans)
	shared := SpreadCharges(enc.N, enc.Stride, len(grams), length, spans, weights)
	charges := make([]float64, len(grams))
	for g, share := range shared.Shares {
		charges[g] = math.Min(1, share)
	}
	pairs := make([][2]int, len(spans))
	for i, s := range spans {
		pairs[i] = [2]int{s.Lo, s.Hi}
	}
	return AttentionSide{
		Text: text, Units: length, Grams: grams, Spans: pairs, Writer: writes, Charges: charges,
		Focus: shared.Focus, End: shared.End,
	}
}

// AttentionPreview shows where one correction would land, gram by gram, under
// the writer rule and under a band: blur when given, else the model's own,
// else the default.  It needs only the encoding and changes nothing.
func (m *Model) AttentionPreview(wrong, right string, blur *float64) (*AttentionPreview, error) {
	band := m.G.Attention
	value := DefaultBlur
	if blur != nil {
		value = *blur
	} else if band.On {
		value = band.Blur
	}
	if err := CheckBlur(value); err != nil {
		return nil, err
	}
	enc := m.Encoding()
	weights := BandWeights(enc.N, value)
	wrongSpans, rightSpans := enc.ChangedSpans(wrong, right)
	changes := enc.DiffSummary(wrong, right, 0)
	if changes == nil {
		changes = []Edit{}
	}
	return &AttentionPreview{
		Attention: m.AttentionConfig(), Blur: value, Weights: weights, Changes: changes,
		Wrong: attentionSide(enc, weights, wrong, wrongSpans),
		Right: attentionSide(enc, weights, right, rightSpans),
	}, nil
}
