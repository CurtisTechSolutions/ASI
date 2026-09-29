package tokenizer

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/nn"
)

// TailBytes is how many of a window's newest bytes the tail accuracy checks.
const TailBytes = 4

// Options steer training.
type Options struct {
	Steps       int
	Batch       int
	LR          float64
	Seed        int64
	EvalEvery   int
	EvalWindows int
	Clip        float64
	Quantize    bool // false trains without the rounding (the gradient-check path)
	Log         func(Stat)
}

// DefaultOptions: 2000 steps of 256 windows.
func DefaultOptions() Options {
	return Options{Steps: 2000, Batch: 256, LR: 2e-3, Seed: 1, EvalEvery: 100, EvalWindows: 512, Clip: 5, Quantize: true}
}

// Stat is one line of the training log.
type Stat struct {
	Step    int       `json:"step"`
	Loss    float64   `json:"loss_bits"` // recency-weighted reconstruction bits per byte of the evaluation windows, all symbols visible
	Next    []float64 `json:"next_bits"` // bits of the byte after the window, from the first 1..Depth symbols
	Acc     []float64 `json:"accuracy"`  // exact-byte reconstruction from the first 1..Depth symbols
	Tail    []float64 `json:"tail"`      // windows whose last TailBytes bytes were all exact, from 1..Depth symbols
	Train   float64   `json:"train_bits"`
	LR      float64   `json:"lr"`
	Seconds float64   `json:"seconds"`
}

// sampler draws training windows from texts, each text weighted by its length.
type sampler struct {
	texts  [][]byte
	cum    []int
	total  int
	W      int
	rng    *rand.Rand
	starts float64
}

func newSampler(texts [][]byte, W int, starts float64, rng *rand.Rand) *sampler {
	s := &sampler{W: W, rng: rng, starts: starts}
	for _, t := range texts {
		if len(t) == 0 {
			continue
		}
		s.texts = append(s.texts, t)
		s.total += len(t) + 1
		s.cum = append(s.cum, s.total)
	}
	return s
}

// draw picks a text and a position 0..len(text) in it.
func (s *sampler) draw() ([]byte, int) {
	r := s.rng.Intn(s.total)
	k := 0
	for s.cum[k] <= r {
		k++
	}
	text := s.texts[k]
	if s.rng.Float64() < s.starts {
		lim := s.W
		if lim > len(text)+1 {
			lim = len(text) + 1
		}
		return text, s.rng.Intn(lim)
	}
	return text, s.rng.Intn(len(text) + 1)
}

// fill draws n windows into a: clean targets, corrupted inputs at a per-row noise level, random depths.
func (s *sampler) fill(a *act, n int, noise float64, D int, corrupt bool) {
	W := s.W
	a.n = n
	if cap(a.windows) < n*W {
		a.windows = make([]int32, n*W)
		a.targets = make([]int, n*W)
		a.depth = make([]int, n)
		a.next = make([]int, n)
	}
	a.windows, a.targets, a.depth, a.next = a.windows[:n*W], a.targets[:n*W], a.depth[:n], a.next[:n]
	for i := 0; i < n; i++ {
		text, pos := s.draw()
		win := a.windows[i*W : (i+1)*W]
		window(text, pos, W, win)
		if pos < len(text) {
			a.next[i] = int(text[pos])
		} else {
			a.next[i] = PAD
		}
		for p, v := range win {
			a.targets[i*W+p] = int(v)
		}
		if corrupt && noise > 0 {
			level := s.rng.Float64() * noise
			for p, v := range win {
				if v != PAD && s.rng.Float64() < level {
					win[p] = int32(s.rng.Intn(256))
				}
			}
		}
		a.depth[i] = 1 + s.rng.Intn(D)
	}
}

// Train fits the tokenizer to texts and returns the log.
func (t *Tokenizer) Train(texts [][]byte, o Options) ([]Stat, error) {
	if o.Steps <= 0 || o.Batch <= 0 || o.LR <= 0 {
		return nil, fmt.Errorf("steps, batch and lr must be positive")
	}
	total := 0
	for _, x := range texts {
		total += len(x)
	}
	if total == 0 {
		return nil, fmt.Errorf("no training text")
	}
	D := t.Depth()
	rng := rand.New(rand.NewSource(o.Seed))
	train := newSampler(texts, t.Window, t.StartShare, rng)
	eval := newSampler(texts, t.Window, t.StartShare, rand.New(rand.NewSource(o.Seed+7919)))
	evalN := o.EvalWindows
	if evalN <= 0 {
		evalN = 256
	}
	evalAct := newAct()
	eval.fill(evalAct, evalN, 0, D, false)
	evalTargets := append([]int(nil), evalAct.targets...)
	tail := TailBytes
	if tail > t.Window {
		tail = t.Window
	}
	evaluate := func() (float64, []float64, []float64, []float64) {
		acc, tails, nexts := make([]float64, D), make([]float64, D), make([]float64, D)
		loss := 0.0
		for k := 1; k <= D; k++ {
			for i := range evalAct.depth {
				evalAct.depth[i] = k
			}
			copy(evalAct.targets, evalTargets)
			l, hits, next := t.forwardLoss(evalAct, o.Quantize, false)
			acc[k-1] = float64(hits) / float64(evalN*t.Window)
			tails[k-1] = float64(t.tailHits(evalAct, tail)) / float64(evalN)
			nexts[k-1] = next / math.Ln2
			if k == D {
				loss = (l - t.Predict*next) / math.Ln2
			}
		}
		return loss, acc, tails, nexts
	}
	params := t.Params()
	adam := nn.NewAdam(params)
	a := newAct()
	warm := o.Steps / 20
	if warm < 10 {
		warm = 10
	}
	var stats []Stat
	started := time.Now()
	runLoss := 0.0
	for step := 1; step <= o.Steps; step++ {
		lr := o.LR
		if step <= warm {
			lr = o.LR * float64(step) / float64(warm)
		} else {
			progress := float64(step-warm) / float64(o.Steps-warm+1)
			lr = o.LR * (0.1 + 0.9*0.5*(1+math.Cos(math.Pi*progress)))
		}
		train.fill(a, o.Batch, t.Noise, D, true)
		loss, _, _ := t.forwardLoss(a, o.Quantize, true)
		t.backward(a)
		nn.Clip(params, o.Clip)
		adam.Step(lr)
		t.Steps++
		if runLoss == 0 {
			runLoss = loss
		} else {
			runLoss = 0.95*runLoss + 0.05*loss
		}
		if (o.EvalEvery > 0 && step%o.EvalEvery == 0) || step == o.Steps {
			evalLoss, acc, tails, nexts := evaluate()
			s := Stat{Step: t.Steps, Loss: evalLoss, Next: nexts, Acc: acc, Tail: tails, Train: runLoss / math.Ln2, LR: lr, Seconds: time.Since(started).Seconds()}
			stats = append(stats, s)
			if o.Log != nil {
				o.Log(s)
			}
		}
	}
	t.Corpus += total
	t.Stats = append(t.Stats, stats...)
	return stats, nil
}

// GradCheck compares the analytic gradient against central finite differences on a tiny batch without the
// rounding (the rounding is piecewise constant). It returns the largest relative error found.
func (t *Tokenizer) GradCheck(texts [][]byte, seed int64, eps float64) (float64, error) {
	rng := rand.New(rand.NewSource(seed))
	s := newSampler(texts, t.Window, t.StartShare, rng)
	a := newAct()
	s.fill(a, 4, 0, t.Depth(), false)
	targets := append([]int(nil), a.targets...)
	loss := func() float64 {
		copy(a.targets, targets)
		l, _, _ := t.forwardLoss(a, false, false)
		return l
	}
	copy(a.targets, targets)
	nn.ZeroGrads(t.Params())
	t.forwardLoss(a, false, true)
	t.backward(a)
	worst := 0.0
	for _, p := range t.Params() {
		stride := len(p.W.Data)/12 + 1
		for i := 0; i < len(p.W.Data); i += stride {
			old := p.W.Data[i]
			p.W.Data[i] = old + float32(eps)
			lp := loss()
			p.W.Data[i] = old - float32(eps)
			lm := loss()
			p.W.Data[i] = old
			num := (lp - lm) / (2 * eps)
			ana := float64(p.G.Data[i])
			denom := math.Abs(num) + math.Abs(ana) + 1e-2
			if rel := math.Abs(num-ana) / denom; rel > worst {
				worst = rel
			}
		}
	}
	return worst, nil
}
