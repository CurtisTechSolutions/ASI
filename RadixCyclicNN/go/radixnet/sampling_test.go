package radixnet

import (
	"math"
	"reflect"
	"testing"
)

// kept is what the filter leaves of options with these costs, as child ids
// (child = position, the order the node offered them in).
func kept(costs []float64, temperature float64, f SamplingFilter) []int {
	options := make([]ChildCost, len(costs))
	for i, c := range costs {
		options[i] = ChildCost{Child: i, Edge: 100 + i, Cost: c}
	}
	out := []int{}
	for _, o := range f.Filter(options, temperature) {
		out = append(out, o.Child)
	}
	return out
}

// The cases of tests/test_search_training.py, number for number.
func TestSamplingFilterKeepsWhatPythonKeeps(t *testing.T) {
	off := SamplingFilter{TopP: 1}
	cases := []struct {
		costs []float64
		temp  float64
		f     SamplingFilter
		want  []int
	}{
		{[]float64{0.5, 0.1, 2.0}, 1, off, []int{0, 1, 2}},
		{[]float64{0.5, 0.1, 2.0}, 0, SamplingFilter{TopK: 1, TopP: 1}, []int{0, 1, 2}},
		{[]float64{0.5, 0.1, 2.0, 0.3}, 1, SamplingFilter{TopK: 2, TopP: 1}, []int{1, 3}},
		{[]float64{0.2, 0.2, 0.2}, 1, SamplingFilter{TopK: 2, TopP: 1}, []int{0, 1}},
		{[]float64{0.1, 0.5, 2.0}, 1, SamplingFilter{TopP: 1, MinP: 0.5}, []int{0, 1}},
		{[]float64{0.1, 0.5, 2.0}, 1, SamplingFilter{TopP: 1, MinP: 0.9}, []int{0}},
		{[]float64{0.1, 0.5, 2.0}, 10, SamplingFilter{TopP: 1, MinP: 0.5}, []int{0, 1, 2}},
		{[]float64{0, 0, 0, 0}, 1, SamplingFilter{TopP: 0.5}, []int{0, 1}},
		{[]float64{0, 0, 0, 0}, 1, SamplingFilter{TopP: 0.51}, []int{0, 1, 2}},
		{[]float64{0, 0.1, 0.2, 5, 6}, 1, SamplingFilter{TopK: 4, TopP: 0.4, MinP: 0.5}, []int{0, 1}},
		{[]float64{3, 0.7, 1}, 1, SamplingFilter{TopK: 1, TopP: 1}, []int{1}},
		{[]float64{3, 0.7, 1}, 1, SamplingFilter{TopP: 1e-9}, []int{1}},
		{[]float64{3, 0.7, 1}, 1, SamplingFilter{TopP: 1, MinP: 0.999}, []int{1}},
	}
	for i, c := range cases {
		if got := kept(c.costs, c.temp, c.f); !reflect.DeepEqual(got, c.want) {
			t.Errorf("case %d: kept %v, want %v", i, got, c.want)
		}
	}
	for _, bad := range []SamplingFilter{{TopK: -1, TopP: 1}, {TopP: 1.5}, {TopP: math.NaN()}, {TopP: 1, MinP: 1}, {TopP: 1, MinP: -0.1}} {
		if bad.Check() == nil {
			t.Errorf("%+v was accepted", bad)
		}
	}
}

func TestDiversePickTradesCostForNovelty(t *testing.T) {
	if PathOverlap([]int{0, 1, 2, 3}, []int{0, 1, 2, 4}) != 2.0/3.0 || PathOverlap([]int{0, 1, 2}, []int{0, 1, 2, 3, 4}) != 1 ||
		PathOverlap([]int{0, 5, 2}, []int{0, 1, 2}) != 0 || PathOverlap([]int{0}, []int{0, 1}) != 0 {
		t.Fatal("path overlap")
	}
	paths := []DiverseCandidate{
		{0, 1.0, []int{0, 1, 2, 3}},
		{0, 1.1, []int{0, 1, 2, 4}},
		{0, 1.5, []int{0, 7, 8, 9}},
	}
	for _, c := range []struct {
		k         int
		diversity float64
		want      []int
	}{{3, 0, []int{0, 1, 2}}, {2, 1, []int{0, 2}}, {3, 1, []int{0, 2, 1}}} {
		if got := DiversePick(paths, c.k, c.diversity); !reflect.DeepEqual(got, c.want) {
			t.Errorf("DiversePick(k %d, diversity %v) = %v, want %v", c.k, c.diversity, got, c.want)
		}
	}
	punished := []DiverseCandidate{{0, 1, []int{0, 1}}, {1, 0.5, []int{0, 2}}, {0, 9, []int{0, 1, 3}}}
	if got := DiversePick(punished, 3, 100); !reflect.DeepEqual(got, []int{0, 2, 1}) {
		t.Errorf("punishment first: %v", got)
	}
}
