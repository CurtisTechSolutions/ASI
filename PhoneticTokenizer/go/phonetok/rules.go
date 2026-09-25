package phonetok

import (
	_ "embed"
	"fmt"
	"strings"
)

// Letter-to-sound rules (the port of rules.py): the same rule table as the
// Python package, read from the same data file, and the same context engine.

//go:embed data/rules.lts
var rulesText string

// Rule is one rewrite rule, LEFT[MATCH]RIGHT=PHONES.
type Rule struct {
	Left, Match, Right string
	Phones             []string
	Source             string
}

// ParseRule reads one rule line.
func ParseRule(text string) (Rule, error) {
	open := strings.IndexByte(text, '[')
	if open < 0 {
		return Rule{}, fmt.Errorf("not a rule: %q", text)
	}
	closeIdx := strings.IndexByte(text[open:], ']')
	if closeIdx < 0 {
		return Rule{}, fmt.Errorf("not a rule: %q", text)
	}
	closeIdx += open
	eq := strings.IndexByte(text[closeIdx:], '=')
	if eq < 0 {
		return Rule{}, fmt.Errorf("not a rule: %q", text)
	}
	eq += closeIdx
	match := text[open+1 : closeIdx]
	if match == "" {
		return Rule{}, fmt.Errorf("not a rule: %q", text)
	}
	return Rule{
		Left: text[:open], Match: match, Right: text[closeIdx+1 : eq],
		Phones: strings.Fields(text[eq+1:]), Source: text,
	}, nil
}

// Rules are the rules grouped by the first letter of their match, in the order they are tried.
var Rules = func() map[byte][]Rule {
	table := map[byte][]Rule{}
	for _, line := range strings.Split(rulesText, "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, ";") {
			continue
		}
		r, err := ParseRule(line)
		if err != nil {
			panic(err)
		}
		table[r.Match[0]] = append(table[r.Match[0]], r)
	}
	return table
}()

// RuleCount is how many rules there are.
func RuleCount() int {
	n := 0
	for _, rs := range Rules {
		n += len(rs)
	}
	return n
}

const (
	letterVowels = "AEIOUY"
	letterVoiced = "BDGJLMNRVWZ"
	letterFront  = "EIY"
	sibilantOne  = "SCGZXJ"
	uPlainOne    = "TSRDLZNJ"
)

var (
	sibilantTwo = []string{"CH", "SH"}
	uPlainTwo   = []string{"TH", "CH", "SH"}
	suffixes    = []string{"ING", "ELY", "ER", "ES", "ED", "E"}
)

func isVowelLetter(ch byte) bool     { return strings.IndexByte(letterVowels, ch) >= 0 }
func isConsonantLetter(ch byte) bool { return ch >= 'A' && ch <= 'Z' && !isVowelLetter(ch) }

func startsWithAny(word string, pos int, options []string) (int, bool) {
	for _, o := range options {
		if strings.HasPrefix(word[pos:], o) {
			return len(o), true
		}
	}
	return 0, false
}

func endsWithAny(word string, pos int, options []string) (int, bool) {
	for _, o := range options {
		if pos >= len(o) && word[pos-len(o):pos] == o {
			return len(o), true
		}
	}
	return 0, false
}

func matchRight(word string, pos int, pattern string, k int) bool {
	if k == len(pattern) {
		return true
	}
	sym := pattern[k]
	switch sym {
	case '#':
		j := pos
		for j < len(word) && isVowelLetter(word[j]) {
			j++
			if matchRight(word, j, pattern, k+1) {
				return true
			}
		}
		return false
	case ':':
		j := pos
		for {
			if matchRight(word, j, pattern, k+1) {
				return true
			}
			if j < len(word) && isConsonantLetter(word[j]) {
				j++
			} else {
				return false
			}
		}
	}
	if pos >= len(word) {
		return false
	}
	ch := word[pos]
	switch sym {
	case '^':
		return isConsonantLetter(ch) && matchRight(word, pos+1, pattern, k+1)
	case '.':
		return strings.IndexByte(letterVoiced, ch) >= 0 && matchRight(word, pos+1, pattern, k+1)
	case '+':
		return strings.IndexByte(letterFront, ch) >= 0 && matchRight(word, pos+1, pattern, k+1)
	case '&':
		if n, ok := startsWithAny(word, pos, sibilantTwo); ok && matchRight(word, pos+n, pattern, k+1) {
			return true
		}
		return strings.IndexByte(sibilantOne, ch) >= 0 && matchRight(word, pos+1, pattern, k+1)
	case '@':
		if n, ok := startsWithAny(word, pos, uPlainTwo); ok && matchRight(word, pos+n, pattern, k+1) {
			return true
		}
		return strings.IndexByte(uPlainOne, ch) >= 0 && matchRight(word, pos+1, pattern, k+1)
	case '%':
		for _, suffix := range suffixes {
			if strings.HasPrefix(word[pos:], suffix) {
				rest := word[pos+len(suffix):]
				if (rest == " " || rest == "S ") && matchRight(word, len(word), pattern, k+1) {
					return true
				}
			}
		}
		return false
	}
	return ch == sym && matchRight(word, pos+1, pattern, k+1)
}

func matchLeft(word string, pos int, pattern string, k int) bool {
	if k == 0 {
		return true
	}
	sym := pattern[k-1]
	switch sym {
	case '#':
		j := pos
		for j > 0 && isVowelLetter(word[j-1]) {
			j--
			if matchLeft(word, j, pattern, k-1) {
				return true
			}
		}
		return false
	case ':':
		j := pos
		for {
			if matchLeft(word, j, pattern, k-1) {
				return true
			}
			if j > 0 && isConsonantLetter(word[j-1]) {
				j--
			} else {
				return false
			}
		}
	}
	if pos <= 0 {
		return false
	}
	ch := word[pos-1]
	switch sym {
	case '^':
		return isConsonantLetter(ch) && matchLeft(word, pos-1, pattern, k-1)
	case '.':
		return strings.IndexByte(letterVoiced, ch) >= 0 && matchLeft(word, pos-1, pattern, k-1)
	case '+':
		return strings.IndexByte(letterFront, ch) >= 0 && matchLeft(word, pos-1, pattern, k-1)
	case '&':
		if n, ok := endsWithAny(word, pos, sibilantTwo); ok && matchLeft(word, pos-n, pattern, k-1) {
			return true
		}
		return strings.IndexByte(sibilantOne, ch) >= 0 && matchLeft(word, pos-1, pattern, k-1)
	case '@':
		if n, ok := endsWithAny(word, pos, uPlainTwo); ok && matchLeft(word, pos-n, pattern, k-1) {
			return true
		}
		return strings.IndexByte(uPlainOne, ch) >= 0 && matchLeft(word, pos-1, pattern, k-1)
	}
	return ch == sym && matchLeft(word, pos-1, pattern, k-1)
}

// ApplyRules sounds out a word with the rules alone: phones without stress (schwas excepted).
// Letters the rules do not cover are skipped.  A trace, if not nil, collects the rule that fired at each step.
func ApplyRules(word string, trace *[]Rule) []string {
	text := " " + strings.ToUpper(word) + " "
	var out []string
	pos := 1
	end := len(text) - 1
	for pos < end {
		ch := text[pos]
		rules := Rules[ch]
		if len(rules) == 0 {
			pos++
			continue
		}
		fired := false
		for _, rule := range rules {
			if !strings.HasPrefix(text[pos:], rule.Match) {
				continue
			}
			if !matchRight(text, pos+len(rule.Match), rule.Right, 0) {
				continue
			}
			if !matchLeft(text, pos, rule.Left, len(rule.Left)) {
				continue
			}
			out = append(out, rule.Phones...)
			if trace != nil {
				*trace = append(*trace, rule)
			}
			pos += len(rule.Match)
			fired = true
			break
		}
		if !fired {
			pos++
		}
	}
	return out
}

// -- stress ------------------------------------------------------------------------

var prefixes = []string{
	"a", "ab", "ac", "ad", "af", "ag", "al", "ap", "ar", "as", "at", "be", "com", "con", "cor", "de", "dis", "em",
	"en", "ex", "for", "im", "in", "inter", "ir", "mis", "ob", "per", "pre", "pro", "re", "sub", "sup", "sur",
	"trans", "un",
}

var penultSuffixes = []string{"tion", "sion", "cian", "ic", "ics", "ical", "ity", "ities", "ify", "ial", "ual", "eous",
	"ious", "ian", "ience", "ient", "ish", "ive", "ia", "ium", "ular", "itis", "ology", "ogy",
	"ographer", "ography", "ometer", "osis", "atic", "ency", "ancy", "ent", "ant", "ator", "ators"}

var penultSuffixVowels = map[string]int{
	"tion": 1, "sion": 1, "cian": 1, "ic": 1, "ics": 1, "ical": 2, "ity": 2, "ities": 2, "ify": 2, "ial": 1,
	"ual": 1, "eous": 1, "ious": 1, "ian": 1, "ience": 1, "ient": 1, "ish": 1, "ive": 1, "ia": 1, "ium": 1,
	"ular": 2, "itis": 1, "ology": 2, "ogy": 2, "ographer": 2, "ography": 2, "ometer": 2, "osis": 1, "atic": 1,
	"ency": 2, "ancy": 2, "ent": 1, "ant": 1, "ator": 3, "ators": 3,
}

var finalStressSuffixes = []string{"ee", "eer", "ese", "ette", "esque", "ique", "oon", "ain", "een", "ine", "aire"}

var tense = toSet([]string{"EY", "IY", "AY", "OW", "UW", "AW", "OY", "AO"})
var reduced = map[string]string{"AE": "AH", "EH": "AH", "AA": "AH", "UH": "AH", "AO": "AH"}

// AssignStress gives the rules' output for word its stress digits (see rules.py's assign_stress).
func AssignStress(word string, phones []string) []string {
	var vowels []int
	for i, p := range phones {
		if IsVowel(p) {
			vowels = append(vowels, i)
		}
	}
	out := append([]string(nil), phones...)
	if len(vowels) == 0 {
		return out
	}
	var free []int
	for _, i := range vowels {
		if StressOf(phones[i]) < 0 {
			free = append(free, i)
		}
	}
	if len(free) == 0 {
		hasPrimary := false
		for _, i := range vowels {
			if StressOf(phones[i]) == 1 {
				hasPrimary = true
			}
		}
		if !hasPrimary {
			i := vowels[0]
			out[i] = Base(out[i]) + "1"
		}
		return out
	}
	primary := free[0]
	if len(free) > 1 {
		primary = pickPrimary(strings.ToLower(word), phones, vowels, free)
	}
	primarySyllable := 0
	for k, v := range vowels {
		if v == primary {
			primarySyllable = k
		}
	}
	for _, i := range free {
		b := Base(phones[i])
		switch {
		case i == primary:
			out[i] = b + "1"
		case i == vowels[0] && primarySyllable >= 2:
			out[i] = b + "2"
		case i > primary && tense[b]:
			out[i] = b + "2"
		default:
			if r, ok := reduced[b]; ok {
				b = r
			}
			out[i] = b + "0"
		}
	}
	return out
}

func contains(items []int, x int) bool {
	for _, i := range items {
		if i == x {
			return true
		}
	}
	return false
}

func pickPrimary(word string, phones []string, vowels, free []int) int {
	nSyl := len(vowels)
	for _, suffix := range finalStressSuffixes {
		if strings.HasSuffix(word, suffix) && nSyl >= 2 {
			return free[len(free)-1]
		}
	}
	for _, suffix := range penultSuffixes {
		if strings.HasSuffix(word, suffix) {
			back := penultSuffixVowels[suffix]
			idx := len(vowels) - 1 - back
			if idx >= 0 && idx < len(vowels) {
				candidate := vowels[idx]
				if contains(free, candidate) {
					return candidate
				}
				nearer := -1
				for _, i := range free {
					if i <= candidate {
						nearer = i
					}
				}
				if nearer >= 0 {
					return nearer
				}
			}
			break
		}
	}
	if nSyl >= 2 {
		// the prefixes, longest first (Python sorts them by length, descending, keeping order among equals)
		sorted := append([]string(nil), prefixes...)
		stableSortByLenDesc(sorted)
		for _, prefix := range sorted {
			if strings.HasPrefix(word, prefix) && len(word) > len(prefix)+2 {
				var later []int
				for _, i := range free {
					if i > vowels[0] {
						later = append(later, i)
					}
				}
				if len(later) > 0 && nSyl <= 3 {
					return later[0]
				}
				break
			}
		}
	}
	if nSyl >= 4 {
		candidate := vowels[len(vowels)-3]
		if contains(free, candidate) {
			return candidate
		}
	}
	return free[0]
}

func stableSortByLenDesc(items []string) {
	// insertion sort: stable, and the list is short
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && len(items[j]) > len(items[j-1]); j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}

// LetterToSound is phones with stress for a spelled word, from the rules alone.
func LetterToSound(word string) []string {
	return AssignStress(word, ApplyRules(word, nil))
}
