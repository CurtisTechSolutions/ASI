package phonetok

import "strings"

// Syllables (the port of syllables.py): maximal onset over the closed list of
// English onsets, a stressed lax vowel closing its syllable, rhyme and alliteration.

var onsetList = func() map[string]bool {
	m := map[string]bool{"": true}
	for _, c := range Consonants {
		if c != "NG" {
			m[c] = true
		}
	}
	for _, s := range strings.Split(`P L|P R|P Y|B L|B R|B Y|T R|T W|T Y|D R|D W|D Y|K L|K R|K W|K Y|G L|G R|G W|G Y|`+
		`F L|F R|F Y|V Y|TH R|TH W|TH Y|SH R|SH M|SH N|SH L|SH W|S P|S T|S K|S M|S N|S L|S W|S F|M Y|N Y|L Y|HH Y|`+
		`S P L|S P R|S P Y|S T R|S T Y|S K L|S K R|S K W|S K Y`, "|") {
		m[s] = true
	}
	return m
}()

// OnsetCount is how many onsets the list holds, the empty one included (77).
func OnsetCount() int { return len(onsetList) }

// LaxVowels are the vowels that do not end a stressed syllable.
var LaxVowels = toSet([]string{"AE", "AH", "EH", "IH", "UH"})

var appendix = toSet([]string{"S", "Z", "T", "D", "TH"})

// IsLegalOnset reports whether this consonant cluster may start a syllable.  The empty cluster may.
func IsLegalOnset(cluster []string) bool { return onsetList[strings.Join(cluster, " ")] }

// IsLegalCoda reports whether this cluster may close a syllable: its sonority falls away from the vowel,
// a run of coronal obstruents at the very end allowed after anything.
func IsLegalCoda(cluster []string) bool {
	if len(cluster) == 0 {
		return true
	}
	for _, c := range cluster {
		if c == "HH" || c == "W" || c == "Y" {
			return false
		}
	}
	if len(cluster) > 4 {
		return false
	}
	core := append([]string(nil), cluster...)
	for len(core) > 1 && appendix[core[len(core)-1]] {
		core = core[:len(core)-1]
	}
	for i := 1; i < len(core); i++ {
		if Sonority(core[i-1]) < Sonority(core[i]) {
			return false
		}
	}
	return true
}

// Syllable is one syllable: onset + nucleus + coda.
type Syllable struct {
	Onset   []string
	Nucleus string
	Coda    []string
}

// Stress is 0, 1 or 2 (0 for a vowel without a digit and for a consonant nucleus).
func (s Syllable) Stress() int {
	if st := StressOf(s.Nucleus); st >= 0 {
		return st
	}
	return 0
}

// Phones are the syllable's phones in order.
func (s Syllable) Phones() []string {
	out := append([]string(nil), s.Onset...)
	out = append(out, s.Nucleus)
	return append(out, s.Coda...)
}

// Rime is the nucleus and the coda.
func (s Syllable) Rime() []string { return append([]string{s.Nucleus}, s.Coda...) }

// Open reports whether the syllable has no coda.
func (s Syllable) Open() bool { return len(s.Coda) == 0 }

// Text is the syllable as one token: its phones joined by dots.
func (s Syllable) Text() string { return strings.Join(s.Phones(), ".") }

func splitCluster(cluster []string) int {
	for keep := 0; keep <= len(cluster); keep++ {
		if IsLegalOnset(cluster[keep:]) {
			return keep
		}
	}
	return len(cluster)
}

// Syllabify breaks a word's phones into syllables by maximal onset; closeLax applies the
// stressed-lax-vowel rule.  An empty input gives no syllables.
func Syllabify(phones []string, closeLax bool) []Syllable {
	if len(phones) == 0 {
		return nil
	}
	var vowels []int
	for i, p := range phones {
		if IsVowel(p) {
			vowels = append(vowels, i)
		}
	}
	if len(vowels) == 0 {
		peak := 0
		for i := range phones {
			if Sonority(phones[i]) > Sonority(phones[peak]) {
				peak = i
			}
		}
		return []Syllable{{append([]string(nil), phones[:peak]...), phones[peak], append([]string(nil), phones[peak+1:]...)}}
	}
	var out []Syllable
	onsetStart := 0
	for k, vi := range vowels {
		onset := append([]string(nil), phones[onsetStart:vi]...)
		var coda []string
		if k+1 < len(vowels) {
			cluster := phones[vi+1 : vowels[k+1]]
			keep := splitCluster(cluster)
			if closeLax && keep == 0 && len(cluster) > 0 && LaxVowels[Base(phones[vi])] && IsStressed(phones[vi]) &&
				IsLegalOnset(cluster[1:]) {
				keep = 1
			}
			coda = append([]string(nil), cluster[:keep]...)
			onsetStart = vi + 1 + keep
		} else {
			coda = append([]string(nil), phones[vi+1:]...)
		}
		out = append(out, Syllable{onset, phones[vi], coda})
	}
	return out
}

// SyllableCount is one per vowel, or one for a word with none.
func SyllableCount(phones []string) int {
	n := 0
	for _, p := range phones {
		if IsVowel(p) {
			n++
		}
	}
	if n == 0 && len(phones) > 0 {
		return 1
	}
	return n
}

// StressedSyllable is the index of the syllable carrying primary stress (the first secondary one,
// else the last), or -1 for none.
func StressedSyllable(syllables []Syllable) int {
	if len(syllables) == 0 {
		return -1
	}
	for i, s := range syllables {
		if s.Stress() == 1 {
			return i
		}
	}
	for i, s := range syllables {
		if s.Stress() == 2 {
			return i
		}
	}
	return len(syllables) - 1
}

// Rhymes reports whether two words share their sounds from the last stressed vowel on.
func Rhymes(a, b []string) bool {
	sa, sb := Syllabify(a, true), Syllabify(b, true)
	ia, ib := StressedSyllable(sa), StressedSyllable(sb)
	if ia < 0 || ib < 0 {
		return false
	}
	tail := func(s []Syllable, i int) []string {
		var out []string
		for _, syl := range s[i:] {
			out = append(out, syl.Phones()...)
		}
		return StripStress(out[len(s[i].Onset):])
	}
	return strings.Join(tail(sa, ia), " ") == strings.Join(tail(sb, ib), " ")
}

// Alliterates reports whether two words start with the same consonant sound(s).
func Alliterates(a, b []string) bool {
	sa, sb := Syllabify(a, true), Syllabify(b, true)
	if len(sa) == 0 || len(sb) == 0 || len(sa[0].Onset) == 0 {
		return false
	}
	return strings.Join(sa[0].Onset, " ") == strings.Join(sb[0].Onset, " ")
}
