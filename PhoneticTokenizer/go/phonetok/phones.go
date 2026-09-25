// Package phonetok is a phonetic tokenizer: text as the sounds it is made of.
//
// It is the Go port of the Python package of the same name (../../phonetok),
// module for module, and reads the same two data files - the core lexicon and
// the letter-to-sound rules - so that the three ports (Python, Go, Rust) turn
// the same text into the same sounds, token for token.
//
// This file is the sound inventory: the 39 phonemes of the CMU Pronouncing
// Dictionary's ARPAbet, the stress digit a vowel carries, the fixed alphabet
// of ids, the articulatory features, the sonority scale and the IPA.
package phonetok

import (
	"fmt"
	"math"
	"strings"
)

// Vowels are the 15 vowel phonemes, in ARPAbet order.
var Vowels = []string{"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

// Consonants are the 24 consonant phonemes, in ARPAbet order.
var Consonants = []string{"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"}

// The token grammar's fixed symbols.
const (
	Boundary      = "#"
	PauseShort    = ","
	PauseFull     = "."
	PauseQuestion = "?"
	PAD           = "<pad>"
	UNK           = "<unk>"
	BOS           = "<s>"
	EOS           = "</s>"
)

// Pauses are what punctuation becomes; Specials the four every tokenizer has.
var (
	Pauses   = []string{PauseShort, PauseFull, PauseQuestion}
	Specials = []string{PAD, UNK, BOS, EOS}
)

// Ids of the head of the alphabet.
const (
	PadID      = 0
	UnkID      = 1
	BosID      = 2
	EosID      = 3
	BoundaryID = 4
)

// Symbols is the fixed alphabet in id order: the specials, the boundary, the
// pauses, then every ARPAbet symbol as cmudict.symbols lists them (a bare vowel,
// its three stresses, then the consonants): 92 symbols.
var Symbols = func() []string {
	out := append([]string{}, Specials...)
	out = append(out, Boundary)
	out = append(out, Pauses...)
	for _, v := range Vowels {
		out = append(out, v, v+"0", v+"1", v+"2")
	}
	out = append(out, Consonants...)
	return out
}()

// SymbolID is symbol -> id.
var SymbolID = func() map[string]int {
	m := make(map[string]int, len(Symbols))
	for i, s := range Symbols {
		m[s] = i
	}
	return m
}()

var (
	vowelSet     = toSet(Vowels)
	consonantSet = toSet(Consonants)
)

func toSet(items []string) map[string]bool {
	m := make(map[string]bool, len(items))
	for _, s := range items {
		m[s] = true
	}
	return m
}

// Base is the phoneme without its stress digit: "AH0" -> "AH", "K" -> "K".
func Base(phone string) string {
	if n := len(phone); n > 0 && phone[n-1] >= '0' && phone[n-1] <= '2' {
		return phone[:n-1]
	}
	return phone
}

// StressOf is the stress digit of a vowel, or -1 for a consonant or a bare vowel.
func StressOf(phone string) int {
	if n := len(phone); n > 0 && phone[n-1] >= '0' && phone[n-1] <= '2' {
		return int(phone[n-1] - '0')
	}
	return -1
}

// IsVowel reports whether this is a vowel, with or without its stress digit.
func IsVowel(phone string) bool { return vowelSet[Base(phone)] }

// IsConsonant reports whether this is a consonant.
func IsConsonant(phone string) bool { return consonantSet[phone] }

// IsPhone reports whether token is an ARPAbet symbol: a consonant, or a vowel with or without a stress digit.
func IsPhone(token string) bool { return vowelSet[Base(token)] || consonantSet[token] }

// IsStressed is primary or secondary stress.
func IsStressed(phone string) bool { s := StressOf(phone); return s == 1 || s == 2 }

// StripStress is the same phones with every stress digit removed.
func StripStress(phones []string) []string {
	out := make([]string, len(phones))
	for i, p := range phones {
		out[i] = Base(p)
	}
	return out
}

// WithStress is a vowel with its stress set; a consonant is left alone.
func WithStress(phone string, stress int) string {
	b := Base(phone)
	if vowelSet[b] {
		return fmt.Sprintf("%s%d", b, stress)
	}
	return phone
}

// CheckPhones returns an error at the first token that is not a phone.
func CheckPhones(phones []string) error {
	for _, p := range phones {
		if !IsPhone(p) {
			return fmt.Errorf("%q is not an ARPAbet phone", p)
		}
	}
	return nil
}

// -- classes and articulation ----------------------------------------------------

var (
	Stops            = toSet([]string{"B", "D", "G", "K", "P", "T"})
	Affricates       = toSet([]string{"CH", "JH"})
	Fricatives       = toSet([]string{"DH", "F", "S", "SH", "TH", "V", "Z", "ZH"})
	Aspirates        = toSet([]string{"HH"})
	Nasals           = toSet([]string{"M", "N", "NG"})
	Liquids          = toSet([]string{"L", "R"})
	Glides           = toSet([]string{"W", "Y"})
	Sibilants        = toSet([]string{"S", "Z", "SH", "ZH", "CH", "JH"})
	VoicedConsonants = toSet([]string{"B", "D", "DH", "G", "JH", "L", "M", "N", "NG", "R", "V", "W", "Y", "Z", "ZH"})
	labial           = toSet([]string{"B", "P", "M", "F", "V", "W"})
	coronal          = toSet([]string{"T", "D", "S", "Z", "N", "L", "R", "TH", "DH", "SH", "ZH", "CH", "JH"})
	dorsal           = toSet([]string{"K", "G", "NG", "W", "Y"})
	glottal          = toSet([]string{"HH"})
	strident         = toSet([]string{"S", "Z", "SH", "ZH", "CH", "JH", "F", "V"})
	Diphthongs       = toSet([]string{"AW", "AY", "EY", "OW", "OY"})
	RhoticVowels     = toSet([]string{"ER"})
	continuant       = func() map[string]bool {
		m := map[string]bool{}
		for _, set := range []map[string]bool{Fricatives, Aspirates, Liquids, Glides} {
			for k := range set {
				m[k] = true
			}
		}
		return m
	}()
)

// IsSonorant reports whether a consonant is a nasal, a liquid or a glide.
func IsSonorant(c string) bool { return Nasals[c] || Liquids[c] || Glides[c] }

// Sonority is where a phone sits on the scale, 0 (a voiceless stop) to 10 (a low vowel); the stress digit is ignored.
func Sonority(phone string) int {
	b := Base(phone)
	switch {
	case Stops[b]:
		if VoicedConsonants[b] {
			return 1
		}
		return 0
	case Affricates[b]:
		if VoicedConsonants[b] {
			return 2
		}
		return 1
	case Fricatives[b]:
		if VoicedConsonants[b] {
			return 3
		}
		return 2
	case b == "HH":
		return 2
	case Nasals[b]:
		return 4
	case b == "L":
		return 5
	case b == "R":
		return 6
	case Glides[b]:
		return 7
	}
	switch b {
	case "IY", "IH", "UW", "UH", "ER":
		return 8
	case "EY", "EH", "OW", "AO", "AH", "OY":
		return 9
	case "AE", "AA", "AY", "AW":
		return 10
	}
	panic("not a phone: " + phone)
}

// MaxSonority is the top of the scale.
const MaxSonority = 10

type vowelSpace struct {
	height, back, round, tense float64
	glide                      string
}

var vowelSpaces = map[string]vowelSpace{
	"IY": {1.0, 0.0, 0.0, 1.0, ""}, "IH": {0.8, 0.2, 0.0, 0.0, ""}, "EY": {0.6, 0.0, 0.0, 1.0, "Y"},
	"EH": {0.5, 0.2, 0.0, 0.0, ""}, "AE": {0.2, 0.2, 0.0, 0.0, ""}, "AA": {0.0, 0.9, 0.0, 1.0, ""},
	"AO": {0.3, 1.0, 1.0, 1.0, ""}, "OW": {0.6, 1.0, 1.0, 1.0, "W"}, "UH": {0.8, 0.9, 1.0, 0.0, ""},
	"UW": {1.0, 1.0, 1.0, 1.0, ""}, "AH": {0.4, 0.6, 0.0, 0.0, ""}, "ER": {0.5, 0.5, 0.0, 1.0, ""},
	"AY": {0.1, 0.5, 0.0, 1.0, "Y"}, "AW": {0.1, 0.5, 0.0, 1.0, "W"}, "OY": {0.3, 1.0, 1.0, 1.0, "Y"},
}

// FeatureNames are the 22 dimensions of Features, in order.
var FeatureNames = []string{
	"syllabic", "consonantal", "sonorant", "voice", "nasal", "continuant", "strident", "lateral", "rhotic",
	"labial", "coronal", "dorsal", "glottal",
	"high", "low", "back", "round", "tense", "glide_y", "glide_w",
	"stress", "sonority",
}

// FeatureDim is the length of a feature vector.
const FeatureDim = 22

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Features is the articulatory feature vector of a phone (FeatureNames), stress included.
func Features(phone string) ([]float64, error) {
	b := Base(phone)
	if !vowelSet[b] && !consonantSet[b] {
		return nil, fmt.Errorf("%q is not an ARPAbet phone", phone)
	}
	stress := 0.0
	switch StressOf(phone) {
	case 1:
		stress = 1.0
	case 2:
		stress = 0.5
	}
	son := float64(Sonority(b)) / MaxSonority
	if vowelSet[b] {
		v := vowelSpaces[b]
		return []float64{
			1, 0, 1, 1, 0, 1, 0, 0, b2f(RhoticVowels[b]),
			0, 0, 0, 0,
			v.height, 1 - v.height, v.back, v.round, v.tense, b2f(v.glide == "Y"), b2f(v.glide == "W"),
			stress, son,
		}, nil
	}
	return []float64{
		0,
		b2f(!(Glides[b] || Aspirates[b])),
		b2f(IsSonorant(b)),
		b2f(VoicedConsonants[b]),
		b2f(Nasals[b]),
		b2f(continuant[b]),
		b2f(strident[b]),
		b2f(b == "L"),
		b2f(b == "R"),
		b2f(labial[b]),
		b2f(coronal[b]),
		b2f(dorsal[b]),
		b2f(glottal[b]),
		b2f(b == "Y" || b == "W"),
		0,
		b2f(b == "W" || b == "K" || b == "G" || b == "NG"),
		b2f(b == "W"),
		0, 0, 0,
		0, son,
	}, nil
}

// Distance is how differently two phones are made: the Euclidean distance of their feature vectors.
func Distance(a, b string) float64 {
	fa, ea := Features(a)
	fb, eb := Features(b)
	if ea != nil || eb != nil {
		panic("not a phone")
	}
	sum := 0.0
	for i := range fa {
		d := fa[i] - fb[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Similarity is 1 for the same phone, falling towards 0.
func Similarity(a, b string) float64 { return 1 / (1 + Distance(a, b)) }

// -- the IPA ---------------------------------------------------------------------

// IPA is ARPAbet -> IPA, one entry per phoneme.
var IPA = map[string]string{
	"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
	"IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
	"B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
	"M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
	"W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

var ipaReduced = map[string]string{"AH": "ə", "ER": "ɚ"}

// ToIPA is phones as one IPA string, a stress mark before a stressed vowel (when stress is true).
func ToIPA(phones []string, stress bool) string {
	var b strings.Builder
	for _, p := range phones {
		base := Base(p)
		s := StressOf(p)
		switch {
		case vowelSet[base]:
			if stress && s == 1 {
				b.WriteString("ˈ")
			} else if stress && s == 2 {
				b.WriteString("ˌ")
			}
			if r, ok := ipaReduced[base]; ok && s == 0 {
				b.WriteString(r)
			} else {
				b.WriteString(IPA[base])
			}
		case consonantSet[base]:
			b.WriteString(IPA[base])
		default:
			b.WriteString(p)
		}
	}
	return b.String()
}

// -- the possessive / plural / past suffixes ---------------------------------------

// PluralSuffix is the -s / -'s ending after these phones: IH0 Z after a sibilant, Z after voice, S otherwise.
func PluralSuffix(phones []string) []string {
	if len(phones) == 0 {
		return []string{"Z"}
	}
	last := Base(phones[len(phones)-1])
	switch {
	case Sibilants[last]:
		return []string{"IH0", "Z"}
	case vowelSet[last] || VoicedConsonants[last]:
		return []string{"Z"}
	}
	return []string{"S"}
}

// PastSuffix is the -ed ending: IH0 D after T or D, D after voice, T otherwise.
func PastSuffix(phones []string) []string {
	if len(phones) == 0 {
		return []string{"D"}
	}
	last := Base(phones[len(phones)-1])
	switch {
	case last == "T" || last == "D":
		return []string{"IH0", "D"}
	case vowelSet[last] || VoicedConsonants[last]:
		return []string{"D"}
	}
	return []string{"T"}
}
