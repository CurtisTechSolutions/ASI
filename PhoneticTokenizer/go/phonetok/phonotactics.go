package phonetok

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Phonotactics (the port of phonotactics.py): which sounds go well together,
// and how to build a word out of them.

// Phonotactics are sound-to-sound habits counted from pronunciations, plus the law of the syllable.
type Phonotactics struct {
	Stress         bool
	Smoothing      float64
	Bigrams        map[[2]string]int
	Unigrams       map[string]int
	followers      map[string]map[string]int
	followerOrder  map[string][]string
	OnsetsInitial  map[string]int // clusters joined by spaces
	OnsetsMedial   map[string]int
	Nuclei         map[string]int
	CodasMedial    map[string]int
	CodasFinal     map[string]int
	SyllableCounts map[int]int
	Words          int
}

// NewPhonotactics is an empty table.
func NewPhonotactics(stress bool) *Phonotactics {
	return &Phonotactics{
		Stress: stress, Smoothing: 0.5, Bigrams: map[[2]string]int{}, Unigrams: map[string]int{},
		followers: map[string]map[string]int{}, followerOrder: map[string][]string{},
		OnsetsInitial: map[string]int{}, OnsetsMedial: map[string]int{}, Nuclei: map[string]int{},
		CodasMedial: map[string]int{}, CodasFinal: map[string]int{}, SyllableCounts: map[int]int{},
	}
}

func (p *Phonotactics) key(phone string) string {
	if p.Stress {
		return phone
	}
	return Base(phone)
}

// Fit counts the transitions and the syllable parts of every pronunciation.
func (p *Phonotactics) Fit(pronunciations [][]string) *Phonotactics {
	for _, phones := range pronunciations {
		if len(phones) == 0 {
			continue
		}
		keys := make([]string, 0, len(phones)+2)
		keys = append(keys, Boundary)
		for _, ph := range phones {
			keys = append(keys, p.key(ph))
		}
		keys = append(keys, Boundary)
		for i := 0; i+1 < len(keys); i++ {
			a, b := keys[i], keys[i+1]
			p.Bigrams[[2]string{a, b}]++
			p.Unigrams[a]++
			if p.followers[a] == nil {
				p.followers[a] = map[string]int{}
			}
			if _, seen := p.followers[a][b]; !seen {
				p.followerOrder[a] = append(p.followerOrder[a], b)
			}
			p.followers[a][b]++
		}
		p.Unigrams[Boundary]++
		syllables := Syllabify(phones, true)
		p.SyllableCounts[len(syllables)]++
		for i, s := range syllables {
			if i == 0 {
				p.OnsetsInitial[strings.Join(s.Onset, " ")]++
			} else {
				p.OnsetsMedial[strings.Join(s.Onset, " ")]++
			}
			nucleus := s.Nucleus
			if !p.Stress && IsVowel(nucleus) {
				nucleus = WithStress(nucleus, 0)
			}
			p.Nuclei[nucleus]++
			if i == len(syllables)-1 {
				p.CodasFinal[strings.Join(s.Coda, " ")]++
			} else {
				p.CodasMedial[strings.Join(s.Coda, " ")]++
			}
		}
		p.Words++
	}
	return p
}

// PhonotacticsFromLexicon fits on every pronunciation of a lexicon.
func PhonotacticsFromLexicon(lex *Lexicon, stress bool) *Phonotactics {
	items := lex.Items()
	prons := make([][]string, len(items))
	for i, e := range items {
		prons[i] = e.Phones
	}
	return NewPhonotactics(stress).Fit(prons)
}

// AlphabetSize is how many distinct first symbols were seen (at least 1).
func (p *Phonotactics) AlphabetSize() int {
	if len(p.Unigrams) == 0 {
		return 1
	}
	return len(p.Unigrams)
}

// Prob is P(b | a), add-k smoothed over the alphabet seen.
func (p *Phonotactics) Prob(a, b string) float64 {
	a, b = p.key(a), p.key(b)
	k := p.Smoothing
	return (float64(p.Bigrams[[2]string{a, b}]) + k) / (float64(p.Unigrams[a]) + k*float64(p.AlphabetSize()))
}

// LogProb is ln P(b | a).
func (p *Phonotactics) LogProb(a, b string) float64 { return math.Log(p.Prob(a, b)) }

// Affinity is how well b follows a, as pointwise mutual information in bits (0 = chance).
func (p *Phonotactics) Affinity(a, b string) float64 {
	a, b = p.key(a), p.key(b)
	total := 0
	for _, c := range p.Unigrams {
		total += c
	}
	if total == 0 {
		total = 1
	}
	k := p.Smoothing
	A := float64(p.AlphabetSize())
	pab := (float64(p.Bigrams[[2]string{a, b}]) + k) / (float64(total) + k*A*A)
	pa := (float64(p.Unigrams[a]) + k) / (float64(total) + k*A)
	pb := (float64(p.Unigrams[b]) + k) / (float64(total) + k*A)
	return math.Log2(pab / (pa * pb))
}

// Transition is one step of a word with its log-probability.
type Transition struct {
	From, To string
	LogProb  float64
}

// Transitions are every transition of a word, the boundaries included.
func (p *Phonotactics) Transitions(phones []string) []Transition {
	keys := make([]string, 0, len(phones)+2)
	keys = append(keys, Boundary)
	for _, ph := range phones {
		keys = append(keys, p.key(ph))
	}
	keys = append(keys, Boundary)
	out := make([]Transition, 0, len(keys)-1)
	for i := 0; i+1 < len(keys); i++ {
		out = append(out, Transition{keys[i], keys[i+1], p.LogProb(keys[i], keys[i+1])})
	}
	return out
}

// Score is the mean log-probability per transition.
func (p *Phonotactics) Score(phones []string) float64 {
	steps := p.Transitions(phones)
	if len(steps) == 0 {
		return 0
	}
	sum := 0.0
	for _, s := range steps {
		sum += s.LogProb
	}
	return sum / float64(len(steps))
}

// Weakest is the index and log-probability of the least likely step.
func (p *Phonotactics) Weakest(phones []string) (int, float64) {
	steps := p.Transitions(phones)
	best := 0
	for i, s := range steps {
		if s.LogProb < steps[best].LogProb {
			best = i
		}
	}
	return best, steps[best].LogProb
}

// Follower is one likely next sound with its probability.
type Follower struct {
	Phone string
	P     float64
}

// Next is the k likeliest sounds after phone ("#" for the start of a word).
func (p *Phonotactics) Next(phone string, k int) []Follower {
	a := p.key(phone)
	table := p.followers[a]
	if len(table) == 0 {
		return nil
	}
	order := append([]string(nil), p.followerOrder[a]...)
	sort.SliceStable(order, func(i, j int) bool { return table[order[i]] > table[order[j]] })
	total := 0
	for _, c := range table {
		total += c
	}
	if k > len(order) {
		k = len(order)
	}
	out := make([]Follower, 0, k)
	for _, b := range order[:k] {
		out = append(out, Follower{b, float64(table[b]) / float64(total)})
	}
	return out
}

// WellFormed reports whether an English speaker could say this: every onset legal, every coda
// falling, a vowel somewhere.
func WellFormed(phones []string) bool {
	if len(phones) == 0 {
		return false
	}
	hasVowel := false
	for _, p := range phones {
		if IsVowel(p) {
			hasVowel = true
		}
	}
	if !hasVowel {
		return false
	}
	for _, s := range Syllabify(phones, true) {
		if !IsLegalOnset(s.Onset) || !IsLegalCoda(s.Coda) {
			return false
		}
	}
	return true
}

// Violations say what is wrong with a sequence; none when nothing is.
func Violations(phones []string) []string {
	if len(phones) == 0 {
		return []string{"empty"}
	}
	var out []string
	hasVowel := false
	for _, p := range phones {
		if IsVowel(p) {
			hasVowel = true
		}
	}
	if !hasVowel {
		out = append(out, "no vowel")
	}
	for _, s := range Syllabify(phones, true) {
		if !IsLegalOnset(s.Onset) {
			out = append(out, strings.Join(s.Onset, " ")+" cannot start a syllable")
		}
		if !IsLegalCoda(s.Coda) {
			out = append(out, strings.Join(s.Coda, " ")+" cannot end a syllable")
		}
	}
	return out
}

// -- building words --------------------------------------------------------------

// weighted is a table of parts with float weights, drawn in key order like Python's sorted().
type weightedItem struct {
	key    string // the part, phones joined by spaces (a tuple in Python)
	weight float64
}

func lessCluster(a, b string) bool {
	// Python compares tuples of strings element by element, a shorter prefix first
	pa, pb := clusterParts(a), clusterParts(b)
	for i := 0; i < len(pa) && i < len(pb); i++ {
		if pa[i] != pb[i] {
			return pa[i] < pb[i]
		}
	}
	return len(pa) < len(pb)
}

func clusterParts(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, " ")
}

func sample(items []weightedItem, rng Rng) (string, bool) {
	if len(items) == 0 {
		return "", false
	}
	total := 0.0
	for _, it := range items {
		total += it.weight
	}
	r := rng.Float64() * total
	acc := 0.0
	for _, it := range items {
		acc += it.weight
		if r < acc {
			return it.key, true
		}
	}
	return items[len(items)-1].key, true
}

func (p *Phonotactics) draw(table map[string]int, rng Rng, prev string) ([]string, bool) {
	if len(table) == 0 {
		return nil, true
	}
	keys := make([]string, 0, len(table))
	for k := range table {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return lessCluster(keys[i], keys[j]) })
	items := make([]weightedItem, 0, len(keys))
	A := float64(p.AlphabetSize())
	for _, k := range keys {
		count := float64(table[k])
		parts := clusterParts(k)
		weight := count
		if len(parts) > 0 {
			weight = count * (p.Prob(prev, parts[0]) * A)
		}
		items = append(items, weightedItem{k, weight})
	}
	chosen, ok := sample(items, rng)
	if !ok {
		return nil, false
	}
	return clusterParts(chosen), true
}

func (p *Phonotactics) drawNucleus(rng Rng, prev string, stress int) string {
	keys := make([]string, 0, len(p.Nuclei))
	for k := range p.Nuclei {
		if IsVowel(k) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	items := make([]weightedItem, 0, len(keys))
	A := float64(p.AlphabetSize())
	for _, k := range keys {
		items = append(items, weightedItem{k, float64(p.Nuclei[k]) * p.Prob(prev, k) * A})
	}
	chosen, ok := sample(items, rng)
	if !ok {
		chosen = "AH0"
	}
	return WithStress(chosen, stress)
}

func (p *Phonotactics) sampleSyllables(rng Rng, lo, hi int) int {
	var ns []int
	for n := range p.SyllableCounts {
		if lo <= n && n <= hi {
			ns = append(ns, n)
		}
	}
	sort.Ints(ns)
	items := make([]weightedItem, 0, len(ns))
	for _, n := range ns {
		items = append(items, weightedItem{fmt.Sprint(n), float64(p.SyllableCounts[n])})
	}
	chosen, ok := sample(items, rng)
	if !ok {
		return lo
	}
	n := 0
	fmt.Sscan(chosen, &n)
	return n
}

// Build coins a pronounceable word from the habits (see phonotactics.py's build); syllables 0 means
// "as many as words usually have, between min and max"; avoid lists pronunciations not to return.
func (p *Phonotactics) Build(rng Rng, syllables, minSyllables, maxSyllables int, avoid []Entry, tries int) []string {
	if p.Words == 0 {
		panic("fit the phonotactics on some pronunciations first")
	}
	forbidden := map[string]bool{}
	for _, e := range avoid {
		forbidden[strings.Join(StripStress(e.Phones), " ")] = true
	}
	for try := 0; try < tries; try++ {
		n := syllables
		if n == 0 {
			n = p.sampleSyllables(rng, minSyllables, maxSyllables)
		}
		stressed := 0
		if n != 1 {
			if rng.Float64() < 0.6 {
				stressed = 0
			} else {
				stressed = rng.RandBelow(n)
			}
		}
		var phones []string
		ok := true
		for i := 0; i < n; i++ {
			prev := Boundary
			if len(phones) > 0 {
				prev = phones[len(phones)-1]
			}
			table := p.OnsetsMedial
			if i == 0 {
				table = p.OnsetsInitial
			}
			onset, good := p.draw(table, rng, prev)
			if !good {
				ok = false
				break
			}
			phones = append(phones, onset...)
			prev = Boundary
			if len(phones) > 0 {
				prev = phones[len(phones)-1]
			}
			stress := 0
			if i == stressed {
				stress = 1
			}
			nucleus := p.drawNucleus(rng, prev, stress)
			phones = append(phones, nucleus)
			table = p.CodasMedial
			if i == n-1 {
				table = p.CodasFinal
			}
			coda, good := p.draw(table, rng, nucleus)
			if !good {
				ok = false
				break
			}
			phones = append(phones, coda...)
		}
		if !ok || !WellFormed(phones) {
			continue
		}
		if forbidden[strings.Join(StripStress(phones), " ")] {
			continue
		}
		return phones
	}
	return nil
}

// Blend is the best portmanteau of two words: the phones, the syllables kept of a and dropped of b.
func (p *Phonotactics) Blend(a, b []string) ([]string, int, int, error) {
	hasVowel := func(ph []string) bool {
		for _, x := range ph {
			if IsVowel(x) {
				return true
			}
		}
		return false
	}
	if !hasVowel(a) || !hasVowel(b) {
		return nil, 0, 0, fmt.Errorf("both words need a vowel to be blended")
	}
	sa, sb := Syllabify(a, true), Syllabify(b, true)
	type candidate struct {
		phones []string
		i, j   int
	}
	var candidates []candidate
	flat := func(s []Syllable) []string {
		var out []string
		for _, syl := range s {
			out = append(out, syl.Phones()...)
		}
		return out
	}
	for i := 1; i <= len(sa); i++ {
		for j := 0; j < len(sb); j++ {
			head := flat(sa[:i])
			tail := flat(sb[j:])
			candidates = append(candidates, candidate{append(append([]string(nil), head...), tail...), i, j})
			headOpen := append(flat(sa[:i-1]), sa[i-1].Onset...)
			rest := append(append([]string(nil), sb[j].Rime()...), flat(sb[j+1:])...)
			candidates = append(candidates, candidate{append(append([]string(nil), headOpen...), rest...), i, j})
		}
	}
	bestTotal := 0.0
	var best *candidate
	longest := float64(len(a))
	if float64(len(b)) > longest {
		longest = float64(len(b))
	}
	for idx := range candidates {
		c := candidates[idx]
		if !WellFormed(c.phones) || len(c.phones) < 2 {
			continue
		}
		joint := p.Score(c.phones)
		balance := -math.Abs(float64(c.i)/float64(len(sa))-0.5) - math.Abs(float64(len(sb)-c.j)/float64(len(sb))-0.5)
		length := float64(len(c.phones)) / longest
		total := joint + 0.5*balance + 0.2*length
		if best == nil || total > bestTotal {
			best, bestTotal = &candidates[idx], total
		}
	}
	if best == nil {
		return append(append([]string(nil), a...), b...), len(sa), 0, nil
	}
	phones := append([]string(nil), best.phones...)
	primaries := 0
	for _, ph := range phones {
		if strings.HasSuffix(ph, "1") {
			primaries++
		}
	}
	if primaries > 1 {
		seen := false
		for k, ph := range phones {
			if strings.HasSuffix(ph, "1") {
				if seen {
					phones[k] = WithStress(ph, 2)
				}
				seen = true
			}
		}
	}
	if primaries == 0 {
		for k, ph := range phones {
			if IsVowel(ph) {
				phones[k] = WithStress(ph, 1)
				break
			}
		}
	}
	return phones, best.i, best.j, nil
}

// String is the one-line description.
func (p *Phonotactics) String() string {
	return fmt.Sprintf("Phonotactics(words=%d, pairs=%d, stress=%t)", p.Words, len(p.Bigrams), p.Stress)
}
