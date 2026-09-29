package nn

import (
	"math"
	"math/rand"
	"testing"
)

func fill(m *Mat, rng *rand.Rand) {
	for i := range m.Data {
		m.Data[i] = float32(rng.NormFloat64())
	}
}

// The products agree with the plain triple loop, for shapes that take the parallel path and ones that don't.
func TestProducts(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, s := range [][3]int{{3, 5, 7}, {130, 17, 9}, {70, 200, 33}} {
		n, k, m := s[0], s[1], s[2]
		A, B := NewMat(n, k), NewMat(k, m)
		fill(A, rng)
		fill(B, rng)
		C := NewMat(0, 0)
		MatMul(A, B, C)
		Bt := NewMat(m, k)
		for i := 0; i < k; i++ {
			for j := 0; j < m; j++ {
				Bt.Data[j*k+i] = B.Data[i*m+j]
			}
		}
		C2 := NewMat(0, 0)
		MatMulABt(A, Bt, C2)
		At := NewMat(k, n)
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				At.Data[j*n+i] = A.Data[i*k+j]
			}
		}
		C3 := NewMat(k, m)
		Bn := NewMat(n, m)
		fill(Bn, rng)
		MatMulAtBAdd(A, Bn, C3)
		for i := 0; i < n; i++ {
			for j := 0; j < m; j++ {
				var want float64
				for p := 0; p < k; p++ {
					want += float64(A.Data[i*k+p]) * float64(B.Data[p*m+j])
				}
				if math.Abs(float64(C.Data[i*m+j])-want) > 1e-3 || math.Abs(float64(C2.Data[i*m+j])-want) > 1e-3 {
					t.Fatalf("%v: C[%d,%d] = %v / %v, want %v", s, i, j, C.Data[i*m+j], C2.Data[i*m+j], want)
				}
			}
		}
		for i := 0; i < k; i++ {
			for j := 0; j < m; j++ {
				var want float64
				for r := 0; r < n; r++ {
					want += float64(A.Data[r*k+i]) * float64(Bn.Data[r*m+j])
				}
				if math.Abs(float64(C3.Data[i*m+j])-want) > 1e-3 {
					t.Fatalf("%v: AtB[%d,%d] = %v, want %v", s, i, j, C3.Data[i*m+j], want)
				}
			}
		}
	}
}

// A two-layer network's analytic gradients match central finite differences.
func TestDenseGradients(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	l1 := NewDense("l1", 5, 7, rng, 1)
	l2 := NewDense("l2", 7, 4, rng, 1)
	x := NewMat(6, 5)
	fill(x, rng)
	targets := []int{0, 1, 2, 3, 1, 2}
	weights := []float32{1, 0.5, 2, 1, 0.25, 1}
	loss := func() float64 {
		h, a, y := NewMat(0, 0), NewMat(0, 0), NewMat(0, 0)
		l1.Forward(x, h)
		SiLU(h, a)
		l2.Forward(a, y)
		v, _ := SoftmaxCrossEntropy(y, targets, weights, nil)
		return v
	}
	h, a, y, dy, da, dh, dx := NewMat(0, 0), NewMat(0, 0), NewMat(0, 0), NewMat(0, 0), NewMat(0, 0), NewMat(0, 0), NewMat(0, 0)
	l1.Forward(x, h)
	SiLU(h, a)
	l2.Forward(a, y)
	SoftmaxCrossEntropy(y, targets, weights, dy)
	l2.Backward(a, dy, da)
	SiLUBackward(h, da, dh)
	l1.Backward(x, dh, dx)
	params := append(l1.Params(), l2.Params()...)
	checked := 0
	for _, p := range params {
		for i := range p.W.Data {
			if i%3 != 0 {
				continue
			}
			old := p.W.Data[i]
			const eps = 1e-2
			p.W.Data[i] = old + eps
			lp := loss()
			p.W.Data[i] = old - eps
			lm := loss()
			p.W.Data[i] = old
			num := (lp - lm) / (2 * eps)
			ana := float64(p.G.Data[i])
			if math.Abs(num-ana) > 1e-2*(math.Abs(num)+math.Abs(ana))+2e-3 {
				t.Errorf("%s[%d]: analytic %.6f numeric %.6f", p.Name, i, ana, num)
			}
			checked++
		}
	}
	if checked < 20 {
		t.Fatalf("checked only %d entries", checked)
	}
}

// Adam drives a quadratic to its minimum.
func TestAdam(t *testing.T) {
	p := &Param{Name: "p", W: NewMat(1, 3), G: NewMat(1, 3)}
	p.W.Data[0], p.W.Data[1], p.W.Data[2] = 3, -2, 0.5
	a := NewAdam([]*Param{p})
	for step := 0; step < 2000; step++ {
		for i, w := range p.W.Data {
			p.G.Data[i] = 2 * (w - float32(i))
		}
		a.Step(0.01)
	}
	for i, w := range p.W.Data {
		if math.Abs(float64(w)-float64(i)) > 1e-2 {
			t.Errorf("param %d = %v, want %d", i, w, i)
		}
	}
}
