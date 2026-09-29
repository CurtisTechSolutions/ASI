package nn

import "math"

// Adam is the optimiser (Kingma & Ba) over a parameter list, with bias correction.
type Adam struct {
	Beta1, Beta2, Eps float64
	Params            []*Param
	m, v              []*Mat
	t                 int
}

// NewAdam prepares the moment estimates.
func NewAdam(params []*Param) *Adam {
	a := &Adam{Beta1: 0.9, Beta2: 0.99, Eps: 1e-8, Params: params}
	for _, p := range params {
		a.m = append(a.m, NewMat(p.W.Rows, p.W.Cols))
		a.v = append(a.v, NewMat(p.W.Rows, p.W.Cols))
	}
	return a
}

// Step applies the accumulated gradients at learning rate lr and clears them.
func (a *Adam) Step(lr float64) {
	a.t++
	c1 := 1 - math.Pow(a.Beta1, float64(a.t))
	c2 := 1 - math.Pow(a.Beta2, float64(a.t))
	b1, b2 := float32(a.Beta1), float32(a.Beta2)
	for k, p := range a.Params {
		w, g, m, v := p.W.Data, p.G.Data, a.m[k].Data, a.v[k].Data
		for i := range w {
			gi := g[i]
			m[i] = b1*m[i] + (1-b1)*gi
			v[i] = b2*v[i] + (1-b2)*gi*gi
			mh := float64(m[i]) / c1
			vh := float64(v[i]) / c2
			w[i] -= float32(lr * mh / (math.Sqrt(vh) + a.Eps))
			g[i] = 0
		}
	}
}

// ZeroGrads clears every gradient.
func ZeroGrads(params []*Param) {
	for _, p := range params {
		p.G.Zero()
	}
}

// Clip scales the gradients down to a global norm of at most limit and returns the norm before clipping.
func Clip(params []*Param, limit float64) float64 {
	var sum float64
	for _, p := range params {
		for _, g := range p.G.Data {
			sum += float64(g) * float64(g)
		}
	}
	norm := math.Sqrt(sum)
	if limit > 0 && norm > limit {
		s := float32(limit / norm)
		for _, p := range params {
			for i := range p.G.Data {
				p.G.Data[i] *= s
			}
		}
	}
	return norm
}
