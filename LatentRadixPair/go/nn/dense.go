package nn

import (
	"math"
	"math/rand"
)

// Param is a weight matrix with its gradient.
type Param struct {
	Name string
	W, G *Mat
}

// Dense is y = x · W + b with W [in x out].
type Dense struct {
	In, Out int
	W, B    *Param
}

// NewDense initialises W ~ N(0, 2/in) (Kaiming, for the SiLU that follows most layers) and b = 0.
func NewDense(name string, in, out int, rng *rand.Rand, scale float64) *Dense {
	w := NewMat(in, out)
	std := scale * math.Sqrt(2/float64(in))
	for i := range w.Data {
		w.Data[i] = float32(rng.NormFloat64() * std)
	}
	return &Dense{In: in, Out: out,
		W: &Param{Name: name + ".w", W: w, G: NewMat(in, out)},
		B: &Param{Name: name + ".b", W: NewMat(1, out), G: NewMat(1, out)}}
}

// Forward sets y = x · W + b.
func (d *Dense) Forward(x, y *Mat) {
	MatMul(x, d.W.W, y)
	b := d.B.W.Data
	for i := 0; i < y.Rows; i++ {
		row := y.Row(i)
		for j := range row {
			row[j] += b[j]
		}
	}
}

// Backward accumulates dW += xᵀ · dy and db += Σ dy, and sets dx = dy · Wᵀ when dx is not nil.
func (d *Dense) Backward(x, dy, dx *Mat) {
	MatMulAtBAdd(x, dy, d.W.G)
	g := d.B.G.Data
	for i := 0; i < dy.Rows; i++ {
		row := dy.Row(i)
		for j := range row {
			g[j] += row[j]
		}
	}
	if dx != nil {
		MatMulABt(dy, d.W.W, dx)
	}
}

// Params are the layer's parameters.
func (d *Dense) Params() []*Param { return []*Param{d.W, d.B} }

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// SiLU sets y = x · σ(x).
func SiLU(x, y *Mat) {
	y.Resize(x.Rows, x.Cols)
	parallel(x.Rows, func(lo, hi int) {
		for i := lo * x.Cols; i < hi*x.Cols; i++ {
			v := x.Data[i]
			y.Data[i] = v * sigmoid(v)
		}
	})
}

// SiLUBackward sets dx = dy · SiLU'(x) with SiLU'(x) = σ(x)(1 + x(1 - σ(x))).
func SiLUBackward(x, dy, dx *Mat) {
	dx.Resize(x.Rows, x.Cols)
	parallel(x.Rows, func(lo, hi int) {
		for i := lo * x.Cols; i < hi*x.Cols; i++ {
			v := x.Data[i]
			s := sigmoid(v)
			dx.Data[i] = dy.Data[i] * s * (1 + v*(1-s))
		}
	})
}

// SoftmaxCrossEntropy turns logits [n x c] into probabilities in place and returns the weighted mean
// negative log probability of the targets (nats; weights nil means uniform) and the number of argmax hits;
// dlogits gets the gradient of that mean.
func SoftmaxCrossEntropy(logits *Mat, targets []int, weights []float32, dlogits *Mat) (loss float64, hits int) {
	n, c := logits.Rows, logits.Cols
	if dlogits != nil {
		dlogits.Resize(n, c)
	}
	var total float64
	if weights == nil {
		total = float64(n)
	} else {
		for _, w := range weights[:n] {
			total += float64(w)
		}
	}
	for i := 0; i < n; i++ {
		inv := float32(1 / total)
		if weights != nil {
			inv *= weights[i]
		}
		row := logits.Row(i)
		top, arg := row[0], 0
		for j, v := range row {
			if v > top {
				top, arg = v, j
			}
		}
		var sum float64
		for j, v := range row {
			e := math.Exp(float64(v - top))
			row[j] = float32(e)
			sum += e
		}
		t := targets[i]
		if arg == t {
			hits++
		}
		for j := range row {
			row[j] /= float32(sum)
		}
		p := float64(row[t])
		if p < 1e-30 {
			p = 1e-30
		}
		loss -= float64(inv) * math.Log(p)
		if dlogits != nil {
			d := dlogits.Row(i)
			for j, v := range row {
				d[j] = v * inv
			}
			d[t] -= inv
		}
	}
	return loss, hits
}
