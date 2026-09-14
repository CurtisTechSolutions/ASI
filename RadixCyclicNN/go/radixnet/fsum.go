package radixnet

import "math"

// fsum is the correctly rounded sum of xs (Shewchuk's algorithm as used by
// Python's math.fsum, without the special handling of infinities), so path
// costs agree with the Python implementation to the last bit.
func fsum(xs []float64) float64 {
	partials := make([]float64, 0, 8)
	for _, x := range xs {
		i := 0
		for _, y := range partials {
			if math.Abs(x) < math.Abs(y) {
				x, y = y, x
			}
			hi := x + y
			lo := y - (hi - x)
			if lo != 0 {
				partials[i] = lo
				i++
			}
			x = hi
		}
		partials = append(partials[:i], x)
	}
	n := len(partials)
	if n == 0 {
		return 0
	}
	n--
	hi := partials[n]
	lo := 0.0
	for n > 0 {
		x := hi
		n--
		y := partials[n]
		hi = x + y
		yr := hi - x
		lo = y - yr
		if lo != 0 {
			break
		}
	}
	if n > 0 && ((lo < 0 && partials[n-1] < 0) || (lo > 0 && partials[n-1] > 0)) {
		y := lo * 2
		x := hi + y
		yr := x - hi
		if y == yr {
			hi = x
		}
	}
	return hi
}
