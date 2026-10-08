package pair

import (
	"fmt"
	"math"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/tokenizer"
)

// Settings are the numbers a prediction reads.
type Settings struct {
	Alpha        float64 `json:"alpha"`         // own = ctx / (ctx + alpha)
	Floor        float64 `json:"floor"`         // share of the uniform mixed into every fold
	Smoothing    float64 `json:"smoothing"`     // Jeffreys count added to every outcome
	ShareScale   float64 `json:"share_scale"`   // weight of log share in a step's merit
	RewardScale  float64 `json:"reward_scale"`  // weight of net reward in the reward traversal
	MeritScale   float64 `json:"merit_scale"`   // weight of merit in the punishment traversal
	PenaltyScale float64 `json:"penalty_scale"` // weight of penalties in the punishment traversal
	Strength     float64 `json:"strength"`      // default strength of a judgement
	Outcomes     int     `json:"outcomes"`      // potential outcomes of the judged space: a verdict is worth strength / outcomes
	Rungs        string  `json:"rungs"`         // all | final: which levels a judgement credits
	Backoff      string  `json:"backoff"`       // all | deepest | none
	CellCeiling  int     `json:"cell_ceiling"`  // largest address space allowed
}

// EnglishPhones is the outcome count of English as the repository's phonetic tokenizer spells it: 24
// consonants and 15 vowels at three stress levels. Without stress there are 39 phonemes; with the word
// boundary and the three pauses the tokenizer emits 88 symbols.
const EnglishPhones = 69

// DefaultSettings are the reference's numbers, except that the smoothing is 0: a pseudo-count on each of
// 257 outcomes would swamp a context read a few times, and the fold's backoff already covers what a context
// has not seen. Verdicts are in outcome units, English's by default: a verdict at strength 1 credits
// 1/69 per rung, and a two-outcome game's would credit 1/2. The ceiling allows 16 million cells (384 MB of
// trees).
func DefaultSettings() Settings {
	return Settings{Alpha: 2, Floor: 0.02, Smoothing: 0, ShareScale: 1, RewardScale: 1, MeritScale: 1, PenaltyScale: 1,
		Strength: 1, Outcomes: EnglishPhones, Rungs: "all", Backoff: "all", CellCeiling: 16_777_216}
}

// Validate checks the ranges.
func (s Settings) Validate() error {
	if s.Alpha < 0 || s.Floor < 0 || s.Floor >= 1 || s.Smoothing < 0 {
		return fmt.Errorf("alpha >= 0, 0 <= floor < 1 and smoothing >= 0")
	}
	if s.Rungs != "all" && s.Rungs != "final" {
		return fmt.Errorf("rungs must be all or final, got %q", s.Rungs)
	}
	if s.Backoff != "all" && s.Backoff != "deepest" && s.Backoff != "none" {
		return fmt.Errorf("backoff must be all, deepest or none, got %q", s.Backoff)
	}
	if s.CellCeiling < 1 {
		return fmt.Errorf("cell_ceiling must be positive")
	}
	if s.Outcomes < 1 {
		return fmt.Errorf("outcomes must be at least 1, got %d", s.Outcomes)
	}
	return nil
}

// Score is a text scored: the model's belief, and the reward tree's readings of its steps.
type Score struct {
	Bits         float64     `json:"bits"`
	MeanReward   float64     `json:"mean_reward"`
	WorstPenalty float64     `json:"worst_penalty"`
	Units        int         `json:"units"`
	PerUnit      []UnitScore `json:"per_unit"`
}

// UnitScore is one step of a scored text.
type UnitScore struct {
	Unit    string  `json:"unit"`
	Bits    float64 `json:"bits"`
	Reward  float64 `json:"reward"`
	Penalty float64 `json:"penalty"`
}

// Pair is the two trees over one address space.
type Pair struct {
	Addr     *Address
	Count    *CountTree
	Reward   *RewardTree
	Settings Settings
	shareBuf []float64
	rewBuf   []float64
	scoreBuf []float64
	chainBuf []int
}

// NewPair primes both trees for codes of the given radices.
func NewPair(radices []int, s Settings) (*Pair, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	addr, err := NewAddress(radices, Out, s.CellCeiling)
	if err != nil {
		return nil, err
	}
	return &Pair{Addr: addr, Count: NewCountTree(addr, s.Alpha, s.Smoothing), Reward: NewRewardTree(addr), Settings: s}, nil
}

// Configure changes the settings of a primed pair (the rungs are read at judgement time, so they may change
// too, but earlier judgements keep the levels they credited).
func (p *Pair) Configure(s Settings) error {
	if err := s.Validate(); err != nil {
		return err
	}
	p.Settings = s
	p.Count.Alpha, p.Count.Smoothing = s.Alpha, s.Smoothing
	return nil
}

// Scores are the step scores of a node's outcomes under a traversal.
func (p *Pair) Scores(node int, traversal string, out []float64) ([]float64, error) {
	if traversal != "reward" && traversal != "punishment" {
		return nil, fmt.Errorf("traversal must be reward or punishment, got %q", traversal)
	}
	s := p.Settings
	share := p.Count.Share(node, p.shareBuf)
	p.shareBuf = share
	out = out[:0]
	readRewards := s.Rungs == "all" || p.Addr.Level(node) == p.Addr.D()
	if traversal == "reward" {
		var rew []float64
		if readRewards && s.RewardScale != 0 {
			rew = p.Reward.Rewards(node, p.rewBuf)
			p.rewBuf = rew
		}
		for j, sh := range share {
			m := math.Inf(-1)
			if sh > 0 {
				m = s.ShareScale * math.Log(sh)
			}
			if rew != nil {
				m += s.RewardScale * rew[j]
			}
			out = append(out, m)
		}
		return out, nil
	}
	var pen []float64
	if readRewards && s.PenaltyScale != 0 {
		pen = p.Reward.Penalties(node, p.rewBuf)
		p.rewBuf = pen
	}
	for j, sh := range share {
		m := math.Inf(-1)
		if sh > 0 {
			m = s.MeritScale * s.ShareScale * math.Log(sh)
		}
		if pen != nil {
			m -= s.PenaltyScale * pen[j]
		}
		out = append(out, m)
	}
	return out, nil
}

// Softmax is a distribution from scores in place; -inf scores get 0; all -inf gives the uniform.
func Softmax(scores []float64) []float64 {
	top := math.Inf(-1)
	for _, s := range scores {
		if s > top {
			top = s
		}
	}
	if math.IsInf(top, -1) {
		for i := range scores {
			scores[i] = 1 / float64(len(scores))
		}
		return scores
	}
	sum := 0.0
	for i, s := range scores {
		if math.IsInf(s, -1) {
			scores[i] = 0
		} else {
			scores[i] = math.Exp(s - top)
		}
		sum += scores[i]
	}
	for i := range scores {
		scores[i] /= sum
	}
	return scores
}

// Q is a node's own next-outcome distribution under a traversal (softmax of its step scores).
func (p *Pair) Q(node int, traversal string) ([]float64, error) {
	scores, err := p.Scores(node, traversal, p.scoreBuf)
	if err != nil {
		return nil, err
	}
	p.scoreBuf = scores
	return Softmax(scores), nil
}

// Fold is the next-outcome distribution of a context: its code's nodes from the root down, each level
// keeping own = ctx / (ctx + alpha) of the answer and passing the rest to its parent's fold, floor included.
func (p *Pair) Fold(code []int32, traversal, backoff string) ([]float64, error) {
	s := p.Settings
	if backoff == "" {
		backoff = s.Backoff
	}
	if backoff != "all" && backoff != "deepest" && backoff != "none" {
		return nil, fmt.Errorf("backoff must be all, deepest or none, got %q", backoff)
	}
	if !p.Addr.Valid(code) {
		return nil, fmt.Errorf("code %v does not fit radices %v", code, p.Addr.Radices)
	}
	n := p.Addr.Out
	uniform := 1 / float64(n)
	p.chainBuf = p.Addr.Chain(code, p.chainBuf)
	chain := p.chainBuf
	P := make([]float64, n)
	switch backoff {
	case "none":
		deepest := chain[len(chain)-1]
		if p.Count.Ctx[deepest] > 0 {
			q, err := p.Q(deepest, traversal)
			if err != nil {
				return nil, err
			}
			copy(P, q)
		} else {
			for j := range P {
				P[j] = uniform
			}
		}
	case "deepest":
		deepest := chain[len(chain)-1]
		o := p.Count.Own(deepest)
		q, err := p.Q(deepest, traversal)
		if err != nil {
			return nil, err
		}
		for j := range P {
			P[j] = o*q[j] + (1-o)*uniform
		}
	default:
		for j := range P {
			P[j] = uniform
		}
		for _, node := range chain {
			o := p.Count.Own(node)
			if o == 0 {
				continue
			}
			q, err := p.Q(node, traversal)
			if err != nil {
				return nil, err
			}
			for j := range P {
				P[j] = o*q[j] + (1-o)*P[j]
			}
		}
	}
	f := s.Floor
	for j := range P {
		P[j] = (1-f)*P[j] + f*uniform
	}
	return P, nil
}

// Symbol names an outcome for people: printable bytes as themselves, the rest escaped, End as </s>.
func Symbol(x int) string {
	switch {
	case x == End:
		return "</s>"
	case x == '\n':
		return "\\n"
	case x == '\t':
		return "\\t"
	case x >= 32 && x < 127:
		return string(rune(x))
	default:
		return fmt.Sprintf("\\x%02x", x)
	}
}

// ScoreText prices a text whose positions have the given codes: bits per unit under the model's belief, and
// the reward tree's readings of every step at the full code.
func (p *Pair) ScoreText(codes tokenizer.Codes, outcomes []int, traversal, backoff string) (Score, error) {
	var out Score
	var bitsSum, rewSum, worst float64
	for i, x := range outcomes {
		P, err := p.Fold(codes.At(i), traversal, backoff)
		if err != nil {
			return out, err
		}
		bits := -math.Log2(P[x])
		cell := p.Addr.Cell(p.Addr.Of(codes.At(i), p.Addr.D()), x)
		r, pen := p.Reward.Reward(cell), p.Reward.Penalty(cell)
		bitsSum += bits
		rewSum += r
		if pen > worst {
			worst = pen
		}
		out.PerUnit = append(out.PerUnit, UnitScore{Unit: Symbol(x), Bits: bits, Reward: r, Penalty: pen})
	}
	n := float64(len(outcomes))
	if n < 1 {
		n = 1
	}
	out.Bits, out.MeanReward, out.WorstPenalty, out.Units = bitsSum/n, rewSum/n, worst, len(outcomes)-1
	if out.Units < 0 {
		out.Units = 0
	}
	return out, nil
}

// MemoryBytes is what the two trees hold.
func (p *Pair) MemoryBytes() int { return 24*p.Addr.Cells() + 8*p.Addr.N }
