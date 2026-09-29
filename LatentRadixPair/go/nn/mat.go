// Package nn is the small numeric core the tokenizer trains with: row-major float32 matrices, the three
// matrix products a dense layer's forward and backward passes need, the SiLU activation, a softmax
// cross-entropy and Adam. Every product splits its rows across goroutines.
package nn

import (
	"runtime"
	"sync"
)

// Mat is a dense row-major float32 matrix.
type Mat struct {
	Rows, Cols int
	Data       []float32
}

// NewMat is a zero matrix.
func NewMat(rows, cols int) *Mat {
	return &Mat{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
}

// Row is row i as a slice into the matrix.
func (m *Mat) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Zero clears the matrix.
func (m *Mat) Zero() {
	for i := range m.Data {
		m.Data[i] = 0
	}
}

// Resize makes the matrix rows x cols, reusing its storage when it fits, and zeroes it.
func (m *Mat) Resize(rows, cols int) *Mat {
	if m == nil {
		return NewMat(rows, cols)
	}
	n := rows * cols
	if cap(m.Data) < n {
		m.Data = make([]float32, n)
	}
	m.Data = m.Data[:n]
	m.Rows, m.Cols = rows, cols
	m.Zero()
	return m
}

// Workers is how many goroutines a product uses.
var Workers = runtime.NumCPU()

// parallel runs fn over [0, n) split into contiguous chunks, one per worker.
func parallel(n int, fn func(lo, hi int)) {
	w := Workers
	if w < 1 {
		w = 1
	}
	if n < 64 || w == 1 {
		fn(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// MatMul sets C = A · B for A [n x k] and B [k x m].
func MatMul(A, B, C *Mat) {
	if A.Cols != B.Rows {
		panic("nn: MatMul shape")
	}
	n, k, m := A.Rows, A.Cols, B.Cols
	C.Resize(n, m)
	parallel(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ci := C.Data[i*m : (i+1)*m]
			ai := A.Data[i*k : (i+1)*k]
			for p, a := range ai {
				if a == 0 {
					continue
				}
				bp := B.Data[p*m : (p+1)*m]
				bp = bp[:len(ci)]
				for j := range ci {
					ci[j] += a * bp[j]
				}
			}
		}
	})
}

// MatMulABt sets C = A · Bᵀ for A [n x k] and B [m x k]: the input gradient dx = dy · Wᵀ of a dense layer.
func MatMulABt(A, B, C *Mat) {
	if A.Cols != B.Cols {
		panic("nn: MatMulABt shape")
	}
	n, k, m := A.Rows, A.Cols, B.Rows
	C.Resize(n, m)
	parallel(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ai := A.Data[i*k : (i+1)*k]
			ci := C.Data[i*m : (i+1)*m]
			for j := range ci {
				bj := B.Data[j*k : (j+1)*k]
				bj = bj[:len(ai)]
				var s0, s1, s2, s3 float32
				p := 0
				for ; p+4 <= len(ai); p += 4 {
					s0 += ai[p] * bj[p]
					s1 += ai[p+1] * bj[p+1]
					s2 += ai[p+2] * bj[p+2]
					s3 += ai[p+3] * bj[p+3]
				}
				for ; p < len(ai); p++ {
					s0 += ai[p] * bj[p]
				}
				ci[j] = s0 + s1 + s2 + s3
			}
		}
	})
}

// MatMulAtBAdd adds Aᵀ · B to C for A [n x k] and B [n x m], C [k x m]: the weight gradient dW += xᵀ · dy.
func MatMulAtBAdd(A, B, C *Mat) {
	if A.Rows != B.Rows || C.Rows != A.Cols || C.Cols != B.Cols {
		panic("nn: MatMulAtBAdd shape")
	}
	n, k, m := A.Rows, A.Cols, B.Cols
	parallel(k, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ci := C.Data[i*m : (i+1)*m]
			for r := 0; r < n; r++ {
				a := A.Data[r*k+i]
				if a == 0 {
					continue
				}
				br := B.Data[r*m : (r+1)*m]
				br = br[:len(ci)]
				for j := range ci {
					ci[j] += a * br[j]
				}
			}
		}
	})
}
