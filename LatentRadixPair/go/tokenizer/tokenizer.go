package tokenizer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/nn"
)

// Tokenizer is the trained encoder and decoder.
type Tokenizer struct {
	Config
	Seed   int64
	Steps  int    // training steps taken
	Corpus int    // bytes of training text
	Stats  []Stat // the training log

	Emb                                             *nn.Param // Symbols x Embed
	Enc1, Enc2, EncOut, Dec1, Dec2, DecOut, Unembed *nn.Dense
	NextHead                                        *nn.Dense // the byte after the window, from the decoder's state
}

// Codes are the codes of consecutive positions, Depth symbols each.
type Codes struct {
	D    int
	Data []int32
}

// Len is the number of positions.
func (c Codes) Len() int {
	if c.D == 0 {
		return 0
	}
	return len(c.Data) / c.D
}

// At is the code of position i.
func (c Codes) At(i int) []int32 { return c.Data[i*c.D : (i+1)*c.D] }

// New initialises a tokenizer from a seed.
func New(cfg Config, seed int64) (*Tokenizer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(seed))
	t := &Tokenizer{Config: cfg, Seed: seed}
	emb := nn.NewMat(Symbols, cfg.Embed)
	for i := range emb.Data {
		emb.Data[i] = float32(rng.NormFloat64() * 0.5)
	}
	t.Emb = &nn.Param{Name: "emb", W: emb, G: nn.NewMat(Symbols, cfg.Embed)}
	t.Enc1 = nn.NewDense("enc1", cfg.Window*cfg.Embed, cfg.EncHidden, rng, 1)
	t.Enc2 = nn.NewDense("enc2", cfg.EncHidden, cfg.EncHidden, rng, 1)
	t.EncOut = nn.NewDense("enc_out", cfg.EncHidden, cfg.Dims(), rng, 1)
	t.Dec1 = nn.NewDense("dec1", cfg.Dims()+cfg.Depth(), cfg.DecHidden, rng, 1)
	t.Dec2 = nn.NewDense("dec2", cfg.DecHidden, cfg.DecHidden, rng, 1)
	t.DecOut = nn.NewDense("dec_out", cfg.DecHidden, cfg.Window*cfg.OutEmbed, rng, 1)
	t.Unembed = nn.NewDense("unembed", cfg.OutEmbed, Symbols, rng, 1)
	t.NextHead = nn.NewDense("next", cfg.DecHidden, Symbols, rng, 1)
	return t, nil
}

// Params are every trainable matrix.
func (t *Tokenizer) Params() []*nn.Param {
	out := []*nn.Param{t.Emb}
	for _, d := range []*nn.Dense{t.Enc1, t.Enc2, t.EncOut, t.Dec1, t.Dec2, t.DecOut, t.Unembed, t.NextHead} {
		out = append(out, d.Params()...)
	}
	return out
}

// ParamCount is the number of trainable numbers.
func (t *Tokenizer) ParamCount() int {
	n := 0
	for _, p := range t.Params() {
		n += len(p.W.Data)
	}
	return n
}

// act holds one batch's activations and gradients, reused across steps.
type act struct {
	n                                              int
	windows                                        []int32 // n x Window input symbols
	targets                                        []int   // n*Window clean symbols
	next                                           []int   // n bytes after the windows (PAD: the text ended)
	depth                                          []int   // visible symbols per row
	x, h1, a1, h2, a2, z, th, q, u, g1, b1, g2, b2 *nn.Mat
	o, logits, nextLogits                          *nn.Mat
	weights                                        []float32 // n*Window reconstruction weights
	codes                                          []int32   // n x Depth
	dlogits, dnext, dnb2, do, db2, dg2, db1, dg1   *nn.Mat
	du, dz                                         *nn.Mat
	da2, dh2, da1, dh1, dx                         *nn.Mat
}

func newAct() *act {
	a := &act{}
	for _, m := range []**nn.Mat{&a.x, &a.h1, &a.a1, &a.h2, &a.a2, &a.z, &a.th, &a.q, &a.u, &a.g1, &a.b1, &a.g2, &a.b2,
		&a.o, &a.logits, &a.nextLogits, &a.dlogits, &a.dnext, &a.dnb2, &a.do, &a.db2, &a.dg2, &a.db1, &a.dg1, &a.du, &a.dz,
		&a.da2, &a.dh2, &a.da1, &a.dh1, &a.dx} {
		*m = nn.NewMat(0, 0)
	}
	return a
}

// embed gathers the windows' embeddings into x [n x Window*Embed].
func (t *Tokenizer) embed(a *act) {
	W, E := t.Window, t.Embed
	a.x.Resize(a.n, W*E)
	emb := t.Emb.W.Data
	for i := 0; i < a.n; i++ {
		row := a.x.Row(i)
		win := a.windows[i*W : (i+1)*W]
		for p, s := range win {
			copy(row[p*E:(p+1)*E], emb[int(s)*E:(int(s)+1)*E])
		}
	}
}

// encode runs the encoder on a.x and quantises: a.z, a.th (tanh), a.q (quantised, or tanh when quantize is
// false) and a.codes.
func (t *Tokenizer) encode(a *act, quantize bool) {
	t.Enc1.Forward(a.x, a.h1)
	nn.SiLU(a.h1, a.a1)
	t.Enc2.Forward(a.a1, a.h2)
	nn.SiLU(a.h2, a.a2)
	t.EncOut.Forward(a.a2, a.z)
	D, dims := t.Depth(), t.Dims()
	a.th.Resize(a.n, dims)
	a.q.Resize(a.n, dims)
	if cap(a.codes) < a.n*D {
		a.codes = make([]int32, a.n*D)
	}
	a.codes = a.codes[:a.n*D]
	for i := 0; i < a.n; i++ {
		z, th, q := a.z.Row(i), a.th.Row(i), a.q.Row(i)
		j := 0
		for s, levels := range t.Levels {
			id, mul := 0, 1
			for _, l := range levels {
				v := float32(math.Tanh(float64(z[j])))
				th[j] = v
				r := int(math.Floor(float64((v+1)/2*float32(l-1)) + 0.5))
				if r < 0 {
					r = 0
				} else if r > l-1 {
					r = l - 1
				}
				if quantize {
					q[j] = float32(r)/float32(l-1)*2 - 1
				} else {
					q[j] = v
				}
				id += r * mul
				mul *= l
				j++
			}
			a.codes[i*D+s] = int32(id)
		}
	}
}

// decoderInput builds a.u from a.q and a.depth: a symbol's dimensions when it is visible, else zeros, and a
// visibility flag per symbol.
func (t *Tokenizer) decoderInput(a *act) {
	D, dims := t.Depth(), t.Dims()
	a.u.Resize(a.n, dims+D)
	for i := 0; i < a.n; i++ {
		u, q := a.u.Row(i), a.q.Row(i)
		j := 0
		for s, levels := range t.Levels {
			if s < a.depth[i] {
				copy(u[j:j+len(levels)], q[j:j+len(levels)])
				u[dims+s] = 1
			}
			j += len(levels)
		}
	}
}

// decode runs the decoder on a.u into a.logits [n*Window x Symbols] (still logits).
func (t *Tokenizer) decode(a *act) {
	t.Dec1.Forward(a.u, a.g1)
	nn.SiLU(a.g1, a.b1)
	t.Dec2.Forward(a.b1, a.g2)
	nn.SiLU(a.g2, a.b2)
	t.DecOut.Forward(a.b2, a.o)
	view := &nn.Mat{Rows: a.n * t.Window, Cols: t.OutEmbed, Data: a.o.Data}
	t.Unembed.Forward(view, a.logits)
	t.NextHead.Forward(a.b2, a.nextLogits)
}

// backward runs the whole backward pass from a.dlogits, accumulating parameter gradients.
func (t *Tokenizer) backward(a *act) {
	W, Eo := t.Window, t.OutEmbed
	a.do.Resize(a.n, W*Eo)
	oView := &nn.Mat{Rows: a.n * W, Cols: Eo, Data: a.o.Data}
	doView := &nn.Mat{Rows: a.n * W, Cols: Eo, Data: a.do.Data}
	t.Unembed.Backward(oView, a.dlogits, doView)
	t.DecOut.Backward(a.b2, a.do, a.db2)
	if t.Predict > 0 {
		t.NextHead.Backward(a.b2, a.dnext, a.dnb2)
		for i, v := range a.dnb2.Data {
			a.db2.Data[i] += v
		}
	}
	nn.SiLUBackward(a.g2, a.db2, a.dg2)
	t.Dec2.Backward(a.b1, a.dg2, a.db1)
	nn.SiLUBackward(a.g1, a.db1, a.dg1)
	t.Dec1.Backward(a.u, a.dg1, a.du)
	D, dims := t.Depth(), t.Dims()
	a.dz.Resize(a.n, dims)
	for i := 0; i < a.n; i++ {
		du, dz, th := a.du.Row(i), a.dz.Row(i), a.th.Row(i)
		j := 0
		for s, levels := range t.Levels {
			for range levels {
				if s < a.depth[i] {
					dz[j] = du[j] * (1 - th[j]*th[j]) // straight through the rounding, then through tanh
				}
				j++
			}
		}
	}
	_ = D
	t.EncOut.Backward(a.a2, a.dz, a.da2)
	nn.SiLUBackward(a.h2, a.da2, a.dh2)
	t.Enc2.Backward(a.a1, a.dh2, a.da1)
	nn.SiLUBackward(a.h1, a.da1, a.dh1)
	t.Enc1.Backward(a.x, a.dh1, a.dx)
	E := t.Embed
	g := t.Emb.G.Data
	for i := 0; i < a.n; i++ {
		dx := a.dx.Row(i)
		win := a.windows[i*W : (i+1)*W]
		for p, s := range win {
			gs := g[int(s)*E : (int(s)+1)*E]
			for k, v := range dx[p*E : (p+1)*E] {
				gs[k] += v
			}
		}
	}
}

// forwardLoss runs a forward pass on windows (n x Window) against clean targets with the given visible
// depths and returns the loss (nats: the recency-weighted reconstruction plus Predict times the next byte's),
// the exact-byte hits of the reconstruction and the next byte's loss alone; with grad it also fills the
// gradients.
func (t *Tokenizer) forwardLoss(a *act, quantize, grad bool) (loss float64, hits int, next float64) {
	t.embed(a)
	t.encode(a, quantize)
	t.decoderInput(a)
	t.decode(a)
	var dl, dn *nn.Mat
	if grad {
		dl, dn = a.dlogits, a.dnext
	}
	if len(a.weights) != a.n*t.Window {
		w := t.Weights()
		a.weights = make([]float32, a.n*t.Window)
		for i := 0; i < a.n; i++ {
			copy(a.weights[i*t.Window:(i+1)*t.Window], w)
		}
	}
	loss, hits = nn.SoftmaxCrossEntropy(a.logits, a.targets, a.weights, dl)
	next, _ = nn.SoftmaxCrossEntropy(a.nextLogits, a.next, nil, dn)
	if grad && t.Predict > 0 && t.Predict != 1 {
		for i := range a.dnext.Data {
			a.dnext.Data[i] *= float32(t.Predict)
		}
	}
	return loss + t.Predict*next, hits, next
}

// tailHits counts, after forwardLoss, the rows whose last `tail` bytes were all reconstructed exactly.
func (t *Tokenizer) tailHits(a *act, tail int) int {
	W := t.Window
	hits := 0
	for i := 0; i < a.n; i++ {
		ok := true
		for p := W - tail; p < W && ok; p++ {
			row := a.logits.Row(i*W + p)
			arg := 0
			for j, v := range row {
				if v > row[arg] {
					arg = j
				}
			}
			ok = arg == a.targets[i*W+p]
		}
		if ok {
			hits++
		}
	}
	return hits
}

// window is the Window input symbols before position pos of text (PAD before its start).
func window(text []byte, pos, W int, out []int32) {
	for p := 0; p < W; p++ {
		i := pos - W + p
		if i < 0 {
			out[p] = PAD
		} else {
			out[p] = int32(text[i])
		}
	}
}

// EncodeAll is the code of every position 0..len(text): the context before each byte, and the context after
// the last one.
func (t *Tokenizer) EncodeAll(text []byte) Codes {
	const chunk = 4096
	n := len(text) + 1
	W, D := t.Window, t.Depth()
	out := Codes{D: D, Data: make([]int32, n*D)}
	a := newAct()
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		a.n = hi - lo
		if cap(a.windows) < a.n*W {
			a.windows = make([]int32, a.n*W)
		}
		a.windows = a.windows[:a.n*W]
		for i := lo; i < hi; i++ {
			window(text, i, W, a.windows[(i-lo)*W:(i-lo+1)*W])
		}
		t.embed(a)
		t.encode(a, true)
		copy(out.Data[lo*D:hi*D], a.codes)
	}
	return out
}

// Encode is the code of the context that ends a text (its last Window bytes).
func (t *Tokenizer) Encode(context []byte) []int32 {
	a := newAct()
	a.n = 1
	a.windows = make([]int32, t.Window)
	window(context, len(context), t.Window, a.windows)
	t.embed(a)
	t.encode(a, true)
	return append([]int32(nil), a.codes...)
}

// Latent is the unquantised latent (tanh-bounded) of a context, for inspection.
func (t *Tokenizer) Latent(context []byte) []float32 {
	a := newAct()
	a.n = 1
	a.windows = make([]int32, t.Window)
	window(context, len(context), t.Window, a.windows)
	t.embed(a)
	t.encode(a, false)
	return append([]float32(nil), a.th.Row(0)...)
}

// quantised is the grid value of symbol values: the decoder's view of a code.
func (t *Tokenizer) quantised(code []int32, q []float32) {
	j := 0
	for s, levels := range t.Levels {
		id := int(code[s])
		for _, l := range levels {
			r := id % l
			id /= l
			q[j] = float32(r)/float32(l-1)*2 - 1
			j++
		}
	}
}

// DecodeSymbols reconstructs the window a code stands for from its first known symbols (known <= Depth;
// 0 or more than Depth means all of them): Window symbols, bytes or PAD.
func (t *Tokenizer) DecodeSymbols(code []int32, known int) ([]int, error) {
	D := t.Depth()
	if len(code) != D {
		return nil, fmt.Errorf("a code has %d symbols, got %d", D, len(code))
	}
	for s, c := range code {
		if c < 0 || int(c) >= t.Radix(s) {
			return nil, fmt.Errorf("symbol %d is %d, outside 0..%d", s, c, t.Radix(s)-1)
		}
	}
	if known <= 0 || known > D {
		known = D
	}
	a := newAct()
	a.n = 1
	a.q.Resize(1, t.Dims())
	t.quantised(code, a.q.Row(0))
	a.depth = []int{known}
	t.decoderInput(a)
	t.decode(a)
	out := make([]int, t.Window)
	for p := 0; p < t.Window; p++ {
		row := a.logits.Row(p)
		arg := 0
		for j, v := range row {
			if v > row[arg] {
				arg = j
			}
		}
		out[p] = arg
	}
	return out, nil
}

// Decode renders DecodeSymbols as text, PAD as a middle dot.
func (t *Tokenizer) Decode(code []int32, known int) (string, error) {
	syms, err := t.DecodeSymbols(code, known)
	if err != nil {
		return "", err
	}
	return Render(syms), nil
}

// Render shows window symbols as text, PAD as a middle dot.
func Render(syms []int) string {
	out := make([]byte, 0, len(syms))
	for _, s := range syms {
		if s == PAD {
			out = append(out, "·"...)
		} else {
			out = append(out, byte(s))
		}
	}
	return string(out)
}

// Info describes the tokenizer.
func (t *Tokenizer) Info() map[string]any {
	last := Stat{}
	if len(t.Stats) > 0 {
		last = t.Stats[len(t.Stats)-1]
	}
	return map[string]any{
		"window": t.Window, "depth": t.Depth(), "radices": t.Radices(), "levels": LevelsString(t.Levels),
		"code_bits": t.Bits(), "params": t.ParamCount(), "steps": t.Steps, "corpus_bytes": t.Corpus, "seed": t.Seed,
		"loss_bits": last.Loss, "next_bits": last.Next, "accuracy": last.Acc, "tail_accuracy": last.Tail,
		"recency": t.Recency, "predict": t.Predict,
	}
}

func log2(x float64) float64 { return math.Log2(x) }
