package phonetok

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// The pronouncing lexicon (the port of lexicon.py): CMU-format entries, layered
// from the bundled core and the file named by PHONETOK_LEXICON.

//go:embed data/core.dict
var coreDict string

// EnvLexicon names the environment variable that points at a full dictionary file.
const EnvLexicon = "PHONETOK_LEXICON"

// Entry is one (word, pronunciation) pair.
type Entry struct {
	Word   string
	Phones []string
}

// ParseEntry reads one line of a CMU-format file; ok is false for a comment or a blank.
func ParseEntry(line string) (word string, phones []string, ok bool) {
	if strings.HasPrefix(line, ";;;") {
		return "", nil, false
	}
	if i := strings.IndexByte(line, '#'); i >= 0 {
		line = line[:i]
	}
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return "", nil, false
	}
	word = strings.ToLower(parts[0])
	if strings.HasSuffix(word, ")") {
		if i := strings.LastIndexByte(word, '('); i >= 0 {
			word = word[:i]
		}
	}
	phones = make([]string, len(parts)-1)
	for i, p := range parts[1:] {
		phones[i] = strings.ToUpper(p)
	}
	return word, phones, true
}

// ReadEntries parses every entry of a CMU-format stream.
func ReadEntries(r io.Reader) ([]Entry, error) {
	var out []Entry
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if w, p, ok := ParseEntry(sc.Text()); ok {
			out = append(out, Entry{w, p})
		}
	}
	return out, sc.Err()
}

// CoreEntries are the bundled core lexicon's entries.
func CoreEntries() []Entry {
	entries, _ := ReadEntries(strings.NewReader(coreDict))
	return entries
}

// Lexicon is a word -> pronunciations map and its inverse.  Every method
// lower-cases the word it is given.
type Lexicon struct {
	entries map[string][][]string
	order   []string // words in the order first added, so that listings are stable
	reverse map[string][]string
	loose   map[string][]string
	Sources []string
}

// NewLexicon is an empty lexicon.
func NewLexicon() *Lexicon {
	return &Lexicon{entries: map[string][][]string{}}
}

// CoreLexicon is the bundled core lexicon.
func CoreLexicon() *Lexicon {
	lex := NewLexicon()
	lex.Extend(CoreEntries())
	lex.Sources = append(lex.Sources, "the core lexicon")
	return lex
}

// LoadLexicon reads a CMU-format file.
func LoadLexicon(path string) (*Lexicon, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	entries, err := ReadEntries(f)
	if err != nil {
		return nil, err
	}
	lex := NewLexicon()
	lex.Extend(entries)
	lex.Sources = append(lex.Sources, "file "+path)
	return lex, nil
}

// PortableLexicon is the lexicon every port builds alike: the core, plus the
// PHONETOK_LEXICON file if it is set (the Python package's Lexicon.portable()).
func PortableLexicon() (*Lexicon, error) {
	lex := CoreLexicon()
	if path := os.Getenv(EnvLexicon); path != "" {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		entries, err := ReadEntries(f)
		if err != nil {
			return nil, err
		}
		lex.Extend(entries)
		lex.Sources = append(lex.Sources, "file "+path)
	}
	return lex, nil
}

func samePhones(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Extend adds entries; a pronunciation the word already has is not added twice.  Returns how many were new.
func (l *Lexicon) Extend(entries []Entry) int {
	added := 0
	for _, e := range entries {
		word := strings.ToLower(e.Word)
		phones := append([]string(nil), e.Phones...)
		known, ok := l.entries[word]
		if !ok {
			l.entries[word] = [][]string{phones}
			l.order = append(l.order, word)
			added++
			continue
		}
		dup := false
		for _, k := range known {
			if samePhones(k, phones) {
				dup = true
				break
			}
		}
		if !dup {
			l.entries[word] = append(known, phones)
			added++
		}
	}
	if added > 0 {
		l.reverse, l.loose = nil, nil
	}
	return added
}

// Add adds a pronunciation; first makes it the one Lookup returns.
func (l *Lexicon) Add(word string, phones []string, first bool) error {
	if err := CheckPhones(phones); err != nil {
		return err
	}
	word = strings.ToLower(word)
	phones = append([]string(nil), phones...)
	known := l.entries[word]
	if known == nil {
		l.order = append(l.order, word)
	}
	kept := known[:0:0]
	for _, k := range known {
		if !samePhones(k, phones) {
			kept = append(kept, k)
		}
	}
	if first {
		kept = append([][]string{phones}, kept...)
	} else {
		kept = append(kept, phones)
	}
	l.entries[word] = kept
	l.reverse, l.loose = nil, nil
	return nil
}

// Len is how many words the lexicon holds.
func (l *Lexicon) Len() int { return len(l.entries) }

// Has reports whether a word is in the lexicon.
func (l *Lexicon) Has(word string) bool {
	_, ok := l.entries[strings.ToLower(word)]
	return ok
}

// Lookup is the first pronunciation of a word, or nil.
func (l *Lexicon) Lookup(word string) []string {
	variants := l.entries[strings.ToLower(word)]
	if len(variants) == 0 {
		return nil
	}
	return variants[0]
}

// Pronunciations is every pronunciation of a word; nil if it is unknown.
func (l *Lexicon) Pronunciations(word string) [][]string {
	return l.entries[strings.ToLower(word)]
}

// Words are the words in the order they were first added.
func (l *Lexicon) Words() []string { return l.order }

// Items is every (word, pronunciation) pair, alternatives included, in first-added order.
func (l *Lexicon) Items() []Entry {
	var out []Entry
	for _, w := range l.order {
		for _, p := range l.entries[w] {
			out = append(out, Entry{w, p})
		}
	}
	return out
}

func (l *Lexicon) buildReverse() {
	l.reverse, l.loose = map[string][]string{}, map[string][]string{}
	for _, w := range l.order {
		for _, p := range l.entries[w] {
			key := strings.Join(p, " ")
			l.reverse[key] = append(l.reverse[key], w)
			loose := strings.Join(StripStress(p), " ")
			l.loose[loose] = append(l.loose[loose], w)
		}
	}
}

// Spellings are the words pronounced exactly like phones (homophones), in the order they were added;
// with stress false the stress digits are ignored on both sides.
func (l *Lexicon) Spellings(phones []string, stress bool) []string {
	if l.reverse == nil {
		l.buildReverse()
	}
	if stress {
		return append([]string(nil), l.reverse[strings.Join(phones, " ")]...)
	}
	return append([]string(nil), l.loose[strings.Join(StripStress(phones), " ")]...)
}

// Describe is "N words from <sources>".
func (l *Lexicon) Describe() string {
	src := "nowhere"
	if len(l.Sources) > 0 {
		src = strings.Join(l.Sources, ", ")
	}
	return fmt.Sprintf("%d words from %s", len(l.entries), src)
}

// SortedWords are the words in alphabetical order (for listings).
func (l *Lexicon) SortedWords() []string {
	out := append([]string(nil), l.order...)
	sort.Strings(out)
	return out
}
