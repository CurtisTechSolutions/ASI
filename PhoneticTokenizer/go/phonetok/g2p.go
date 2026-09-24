package phonetok

import (
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// Grapheme to phoneme (the port of g2p.py): the transcriber, its memory, and the respelling.

// Suffix and prefix tables with fixed sounds.
var suffixTable = []struct {
	suffix string
	sounds []string
}{
	{"ness", []string{"N", "AH0", "S"}}, {"ment", []string{"M", "AH0", "N", "T"}}, {"less", []string{"L", "AH0", "S"}},
	{"ful", []string{"F", "AH0", "L"}}, {"able", []string{"AH0", "B", "AH0", "L"}}, {"ish", []string{"IH0", "SH"}},
	{"ly", []string{"L", "IY0"}}, {"er", []string{"ER0"}}, {"est", []string{"IH0", "S", "T"}}, {"ing", []string{"IH0", "NG"}},
	{"y", []string{"IY0"}},
}

var prefixTable = []struct {
	prefix string
	sounds []string
}{
	{"under", []string{"AH2", "N", "D", "ER0"}}, {"inter", []string{"IH2", "N", "T", "ER0"}}, {"super", []string{"S", "UW2", "P", "ER0"}},
	{"multi", []string{"M", "AH2", "L", "T", "IY0"}}, {"semi", []string{"S", "EH2", "M", "IY0"}}, {"over", []string{"OW2", "V", "ER0"}},
	{"anti", []string{"AE2", "N", "T", "IY0"}}, {"auto", []string{"AO2", "T", "OW0"}}, {"micro", []string{"M", "AY2", "K", "R", "OW0"}},
	{"non", []string{"N", "AA2", "N"}}, {"out", []string{"AW2", "T"}}, {"pre", []string{"P", "R", "IY2"}}, {"dis", []string{"D", "IH0", "S"}},
	{"mis", []string{"M", "IH0", "S"}}, {"sub", []string{"S", "AH0", "B"}}, {"un", []string{"AH0", "N"}}, {"re", []string{"R", "IY0"}},
	{"de", []string{"D", "IY0"}}, {"co", []string{"K", "OW0"}},
}

var contractionTable = []struct {
	ending string
	sounds []string
}{
	{"n't", []string{"N", "T"}}, {"'ll", []string{"L"}}, {"'re", []string{"R"}}, {"'ve", []string{"V"}}, {"'d", []string{"D"}}, {"'m", []string{"M"}},
}

var (
	joinersRe = regexp.MustCompile(`[-/_.]+`)
	notLetter = regexp.MustCompile(`[^a-z']`)
)

// Transcriber turns words into sounds through the lexicon, the morphology and the rules, and sounds back into words.
type Transcriber struct {
	Lexicon  *Lexicon
	UseRules bool
	Remember bool
	// Memory holds the words the lexicon did not, with the sounds they were given, in first-read order.
	Memory      map[string][]string
	memoryOrder []string
	// Counts is how often each spelled word has been transcribed: the tie-breaker among homophones.
	Counts map[string]int
}

// NewTranscriber makes a transcriber over a lexicon (the portable one when nil).
func NewTranscriber(lexicon *Lexicon) *Transcriber {
	if lexicon == nil {
		lexicon, _ = PortableLexicon()
	}
	return &Transcriber{Lexicon: lexicon, UseRules: true, Remember: true, Memory: map[string][]string{}, Counts: map[string]int{}}
}

func trimQuotes(w string) string {
	return strings.Trim(w, "'\"`‘’“”")
}

// Word is the sounds of one spelled word (no spaces); nil if it has none.
func (t *Transcriber) Word(word string) []string {
	phones, _ := t.Explain(word)
	return phones
}

// Explain is the sounds of a word and where they came from: lexicon, memory, number, joined,
// morphology, letters, rules or none.
func (t *Transcriber) Explain(word string) ([]string, string) {
	raw := word
	w := strings.ToLower(trimQuotes(word))
	if w == "" {
		return nil, "none"
	}
	t.Counts[w]++
	if found := t.Lexicon.Lookup(w); found != nil {
		return append([]string(nil), found...), "lexicon"
	}
	if m, ok := t.Memory[w]; ok {
		return append([]string(nil), m...), "memory"
	}
	phones, how := t.derive(w, raw)
	if len(phones) > 0 && t.Remember && how != "none" {
		if _, ok := t.Memory[w]; !ok {
			t.memoryOrder = append(t.memoryOrder, w)
		}
		t.Memory[w] = append([]string(nil), phones...)
	}
	return phones, how
}

func hasDigit(s string) bool {
	return strings.ContainsAny(s, "0123456789")
}

func isUpperWord(s string) bool {
	has := false
	for _, r := range s {
		if unicode.IsLower(r) {
			return false
		}
		if unicode.IsUpper(r) {
			has = true
		}
	}
	return has
}

func (t *Transcriber) derive(w, raw string) ([]string, string) {
	if hasDigit(w) {
		if words := NumberWords(w); words != "" {
			return t.join(strings.Fields(words)), "number"
		}
		return t.join(SplitAlphanumeric(w)), "number"
	}
	if joinersRe.MatchString(w) {
		var parts []string
		for _, p := range joinersRe.Split(w, -1) {
			if p != "" {
				parts = append(parts, p)
			}
		}
		if len(parts) > 1 {
			return t.join(parts), "joined"
		}
		if len(parts) == 0 {
			return nil, "none"
		}
		w = parts[0]
	}
	if phones := t.contraction(w); phones != nil {
		return phones, "morphology"
	}
	if phones := t.morphology(w); phones != nil {
		return phones, "morphology"
	}
	if !strings.ContainsAny(w, "abcdefghijklmnopqrstuvwxyz") {
		return nil, "none"
	}
	rawRunes := []rune(raw)
	if !strings.ContainsAny(w, "aeiouy") || (isUpperWord(raw) && len(rawRunes) >= 2 && len(rawRunes) <= 5) {
		return t.letters(w), "letters"
	}
	if !t.UseRules {
		return nil, "none"
	}
	return LetterToSound(notLetter.ReplaceAllString(w, "")), "rules"
}

func (t *Transcriber) join(parts []string) []string {
	var out []string
	for _, p := range parts {
		phones, _ := t.Explain(p)
		out = append(out, phones...)
	}
	return out
}

func (t *Transcriber) letters(w string) []string {
	var out []string
	for _, r := range w {
		if !unicode.IsLetter(r) {
			continue
		}
		found := t.Lexicon.Pronunciations(string(r))
		var named []string
		for _, p := range found {
			for _, x := range p {
				if StressOf(x) == 1 {
					named = p
					break
				}
			}
			if named != nil {
				break
			}
		}
		switch {
		case named != nil:
			out = append(out, named...)
		case len(found) > 0:
			out = append(out, found[0]...)
		default:
			out = append(out, LetterToSound(string(r))...)
		}
	}
	return out
}

func (t *Transcriber) stem(stem string) []string {
	if found := t.Lexicon.Lookup(stem); found != nil {
		return append([]string(nil), found...)
	}
	if m, ok := t.Memory[stem]; ok {
		return append([]string(nil), m...)
	}
	return nil
}

func (t *Transcriber) contraction(w string) []string {
	if !strings.Contains(w, "'") {
		return nil
	}
	for _, ending := range []string{"'s", "s'"} {
		if strings.HasSuffix(w, ending) && len(w) > 2 {
			base := w[:len(w)-2]
			if ending == "s'" {
				base += "s"
			}
			if stemPhones := t.stem(base); stemPhones != nil {
				if ending == "s'" {
					return stemPhones
				}
				return append(stemPhones, PluralSuffix(stemPhones)...)
			}
		}
	}
	for _, c := range contractionTable {
		if strings.HasSuffix(w, c.ending) && len(w) > len(c.ending) {
			stemPhones := t.stem(w[:len(w)-len(c.ending)])
			if stemPhones == nil {
				continue
			}
			last := ""
			if len(stemPhones) > 0 {
				last = Base(stemPhones[len(stemPhones)-1])
			}
			switch {
			case c.ending == "n't" && last == "N":
				return append(stemPhones, "T")
			case last != "" && vowelSet[last] && c.ending != "n't":
				return append(stemPhones, c.sounds...)
			case c.ending == "'d" && (last == "T" || last == "D"):
				return append(stemPhones, "AH0", "D")
			case c.ending == "'d" && last != "" && !vowelSet[last]:
				return append(stemPhones, "D")
			}
			return append(append(stemPhones, "AH0"), c.sounds...)
		}
	}
	return nil
}

func (t *Transcriber) stems(stem string, suffixStartsWithVowel bool) []string {
	candidates := []string{stem}
	if suffixStartsWithVowel {
		candidates = append(candidates, stem+"e")
		if len(stem) >= 3 && stem[len(stem)-1] == stem[len(stem)-2] && strings.IndexByte("bdfgklmnprstz", stem[len(stem)-1]) >= 0 {
			candidates = append(candidates, stem[:len(stem)-1])
		}
		if strings.HasSuffix(stem, "i") {
			candidates = append(candidates, stem[:len(stem)-1]+"y")
		}
	}
	return candidates
}

func (t *Transcriber) morphology(w string) []string {
	if len(w) < 4 {
		return nil
	}
	if strings.HasSuffix(w, "ies") && len(w) > 4 {
		if s := t.stem(w[:len(w)-3] + "y"); s != nil {
			return append(s, PluralSuffix(s)...)
		}
	}
	if strings.HasSuffix(w, "es") && len(w) > 3 {
		if s := t.stem(w[:len(w)-2]); s != nil {
			if len(s) > 0 && Sibilants[Base(s[len(s)-1])] {
				return append(s, "IH0", "Z")
			}
			return append(s, PluralSuffix(s)...)
		}
	}
	if strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") {
		if s := t.stem(w[:len(w)-1]); s != nil {
			return append(s, PluralSuffix(s)...)
		}
	}
	if strings.HasSuffix(w, "ed") && len(w) > 4 {
		for _, candidate := range t.stems(w[:len(w)-2], true) {
			if s := t.stem(candidate); s != nil {
				return append(s, PastSuffix(s)...)
			}
		}
	}
	for _, sx := range suffixTable {
		if strings.HasSuffix(w, sx.suffix) && len(w)-len(sx.suffix) >= 3 {
			for _, candidate := range t.stems(w[:len(w)-len(sx.suffix)], strings.IndexByte("aeiouy", sx.suffix[0]) >= 0) {
				if s := t.stem(candidate); s != nil {
					if sx.suffix == "ly" && len(s) > 0 && s[len(s)-1] == "L" {
						return append(s, sx.sounds[1:]...) // real-ly, full-y: one L, as the dictionary says
					}
					return append(s, sx.sounds...)
				}
			}
		}
	}
	for _, px := range prefixTable {
		if strings.HasPrefix(w, px.prefix) && len(w)-len(px.prefix) >= 3 {
			if s := t.stem(w[len(px.prefix):]); s != nil {
				return append(append([]string(nil), px.sounds...), s...)
			}
		}
	}
	// a compound of two known words: the cut nearest the middle wins
	var best []string
	bestBalance := 0.0
	half := float64(len(w)) / 2
	for cut := 3; cut < len(w)-2; cut++ {
		left, right := t.stem(w[:cut]), t.stem(w[cut:])
		if left == nil || right == nil {
			continue
		}
		demoted := make([]string, len(right))
		for i, p := range right {
			if StressOf(p) == 1 {
				demoted[i] = WithStress(p, 2)
			} else {
				demoted[i] = p
			}
		}
		balance := float64(cut) - half
		if balance < 0 {
			balance = -balance
		}
		if best == nil || balance < bestBalance {
			best, bestBalance = append(left, demoted...), balance
		}
	}
	return best
}

// Spell is a word that sounds like phones: a remembered one, a lexicon one, or a respelling.
func (t *Transcriber) Spell(phones []string) string {
	if len(phones) == 0 {
		return ""
	}
	key := strings.Join(phones, " ")
	var remembered []string
	for _, w := range t.memoryOrder {
		if strings.Join(t.Memory[w], " ") == key {
			remembered = append(remembered, w)
		}
	}
	candidates := remembered
	if len(candidates) == 0 {
		candidates = t.Lexicon.Spellings(phones, true)
	}
	if len(candidates) == 0 {
		loose := strings.Join(StripStress(phones), " ")
		for _, w := range t.memoryOrder {
			if strings.Join(StripStress(t.Memory[w]), " ") == loose {
				remembered = append(remembered, w)
			}
		}
		candidates = remembered
		if len(candidates) == 0 {
			candidates = t.Lexicon.Spellings(phones, false)
		}
	}
	if len(candidates) == 0 {
		return Respell(phones)
	}
	best := candidates[0]
	for _, w := range candidates[1:] {
		if spellLess(w, best, t.Counts) {
			best = w
		}
	}
	return best
}

// spellLess orders homophones: the one read most, then the shorter, then the alphabetical.
func spellLess(a, b string, counts map[string]int) bool {
	if counts[a] != counts[b] {
		return counts[a] > counts[b]
	}
	if len([]rune(a)) != len([]rune(b)) {
		return len([]rune(a)) < len([]rune(b))
	}
	return a < b
}

// -- respelling -------------------------------------------------------------------

var consonantSpelling = map[string]string{
	"B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
	"M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
	"W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}
var vowelSpelling = map[string]string{
	"AA": "o", "AE": "a", "AH": "u", "AO": "aw", "AW": "ow", "AY": "i", "EH": "e", "ER": "er", "EY": "ay",
	"IH": "i", "IY": "ee", "OW": "o", "OY": "oy", "UH": "oo", "UW": "oo",
}
var tenseMagic = map[string]string{"EY": "a", "AY": "i", "OW": "o", "UW": "u"}
var frontVowels = toSet([]string{"EH", "IY", "IH", "EY", "AY"})

const magicOK = "bdfgklmnprstvz"

// Respell is an invented spelling that reads back as phones: K AE1 T -> cat, M EY1 K -> make.
func Respell(phones []string) string {
	n := len(phones)
	out := make([]string, 0, n)
	for i, p := range phones {
		b := Base(p)
		var nxt, nxtB, after string
		if i+1 < n {
			nxt = phones[i+1]
			nxtB = Base(nxt)
		}
		if i+2 < n {
			after = Base(phones[i+2])
		}
		_, nxtIsVowel := vowelSpelling[nxtB]
		_, afterIsVowel := vowelSpelling[after]
		if consonantSet[b] {
			var prev string
			if i > 0 {
				prev = Base(phones[i-1])
			}
			switch {
			case b == "K":
				soft := (nxtIsVowel && !frontVowels[nxtB]) || nxtB == "L" || nxtB == "R" || nxtB == "Y"
				if soft {
					out = append(out, "c")
				} else {
					out = append(out, "k")
				}
				if nxt == "" && i > 0 && (prev == "AE" || prev == "EH" || prev == "IH" || prev == "AH") && StressOf(phones[i-1]) != 0 {
					out[len(out)-1] = "ck"
				}
			case b == "Y" && nxtB == "UW":
				out = append(out, "")
			default:
				out = append(out, consonantSpelling[b])
			}
			if i > 0 && (prev == "AE" || prev == "EH" || prev == "IH" || prev == "AH" || prev == "UH") && IsStressed(phones[i-1]) &&
				nxtIsVowel && strings.Contains("B D G L M N P R S T Z", b) && len(b) == 1 {
				out[len(out)-1] = out[len(out)-1] + out[len(out)-1]
			}
			continue
		}
		var spelling string
		magic := consonantSet[nxtB] && after == "" && i+2 >= n && strings.IndexByte(magicOK, consonantSpelling[nxtB][0]) >= 0 && len(consonantSpelling[nxtB]) == 1
		switch {
		case b == "UW" && i > 0 && Base(phones[i-1]) == "Y":
			if magic {
				spelling = "u\x00"
			} else {
				spelling = "u"
			}
		case b == "IY" && nxt == "" && StressOf(p) == 0 && i > 0:
			spelling = "y"
		case b == "AY" && nxt == "":
			if i > 0 {
				spelling = "y"
			} else {
				spelling = "i"
			}
		case b == "EY" && nxt == "":
			spelling = "ay"
		case b == "OW" && nxt == "":
			spelling = "o"
		case tenseMagic[b] != "" && magic:
			spelling = tenseMagic[b] + "\x00"
		case tenseMagic[b] != "" && consonantSet[nxtB] && afterIsVowel && b != "IY":
			spelling = tenseMagic[b]
		case b == "AH" && StressOf(p) == 0:
			if nxt == "" || i == 0 {
				spelling = "a"
			} else {
				spelling = "u"
			}
		default:
			spelling = vowelSpelling[b]
		}
		out = append(out, spelling)
	}
	text := strings.Join(out, "")
	if i := strings.IndexByte(text, 0); i >= 0 {
		rest := text[i+1:]
		if len(rest) >= 1 {
			text = text[:i] + rest[:1] + "e" + rest[1:]
		} else {
			text = text[:i] + rest
		}
		text = strings.ReplaceAll(text, "\x00", "")
	}
	return text
}

// sortedKeys is a helper for stable listings.
func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
