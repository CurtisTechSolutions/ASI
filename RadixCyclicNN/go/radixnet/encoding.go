// Package radixnet is the Go port of the count / reward model of RadixCyclicNN:
// a self-compressing cyclic n-gram graph whose edge weights are a dual
// frequency function of traversal counts (all time and inside a sliding
// window) plus rewards, with beam-search prediction (top-K and bottom-K
// continuations), generation, scoring, 2NRL feedback and a self-conversation.
//
// Model files are interchangeable with the Python implementation
// (format "radixnet-count"): both sides read and write the same JSON layout,
// including the Mersenne Twister state, so a model trained here continues in
// Python and vice versa with identical numbers - in every encoding, not only
// the default one.  A graph built with anything but the character trigram of
// stride 1 writes an "encoding" block, and both implementations read it.
//
// Concurrency: training fans goroutines out over the texts (lines, paragraphs
// or pages of a file) for encoding, tracing and counting (lock-free atomic
// increments), recomputes weights and edge costs in parallel over the nodes,
// and runs the two beams of a prediction side by side.  Only structural
// changes (splits and merges of nodes) take a write lock.
package radixnet

import (
	"encoding/base64"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Window is the default n-gram size: the trigram the model was born with.  It
// is the default of Encoding.N, not a property of every graph - a graph
// encodes with its own Encoding (Graph.Enc), whose n can be any number.
const Window = 3

// Overlap is how many units two consecutive labels share under the default
// encoding (a sliding window of stride 1).  A graph's own overlap is
// Graph.Enc.Overlap().
const Overlap = Window - 1

// StartLabel and EndLabel are the labels of the two sentinel nodes.
const (
	StartLabel = "<s>"
	EndLabel   = "</s>"
	// BackLabel is the third sentinel: where the graph has learned a walk goes round (Back).
	BackLabel = "<back>"
)

// -- what a unit is --------------------------------------------------------------

// UnitKind is what one position of a text is: a character or a word.  It is
// the atom everything downstream counts in - the n of the n-gram, the stride,
// a node's label length, the length of a prediction, the spans of a diff.
type UnitKind string

const (
	// Chars: one unit is one character (one code point).  "hello" is 5 units.
	Chars UnitKind = "char"
	// Words: one unit is one whitespace-delimited word.  "the cat sat" is 3
	// units, and the text is normalised to single spaces on the way in - a
	// word encoding keeps the words, not the layout.
	Words UnitKind = "word"
)

// Valid reports whether k is a unit kind this package knows.
func (k UnitKind) Valid() bool { return k == Chars || k == Words }

// -- the encoding ------------------------------------------------------------------

// Encoding is how a text becomes the grams the graph is built from, and how
// the labels of a walk become text again.  Three dials:
//
//   - Unit  - what one position is: a character (Chars) or a word (Words).
//   - N     - how many units one gram holds: the n of the n-gram.  Any n >= 1.
//   - Stride - how many units apart consecutive grams start.  1 is the sliding
//     window the model was born with ("hello" -> hel, ell, llo); N is
//     non-overlapping groups ("hello" with N=4 -> hell), and anything in
//     between overlaps by N - Stride.
//
// So the trigram model is {Chars, 3, 1}, groups of five letters are
// {Chars, 5, 5}, and word bigrams are {Words, 2, 1}.
//
// Two consecutive grams of a text share Overlap() units, which is what lets
// the graph chain them: node A ends with the units node B starts with.  With
// Stride == N nothing is shared, and the graph becomes a plain chain of
// groups - still a graph, just one whose nodes only ever meet at their ends.
type Encoding struct {
	Unit   UnitKind `json:"unit"`
	N      int      `json:"n"`
	Stride int      `json:"stride"`
}

// DefaultEncoding is the character trigram with stride 1: what every model
// used before the encoding became a choice, and the only encoding the Python
// implementation reads.
func DefaultEncoding() Encoding { return Encoding{Unit: Chars, N: Window, Stride: 1} }

// WithDefaults fills the zero value in: an empty unit is Chars, n 0 is Window,
// stride 0 is 1 (a sliding window).  Every entry point runs its encoding
// through this, so the zero Encoding is the default one.
func (e Encoding) WithDefaults() Encoding {
	if e.Unit == "" {
		e.Unit = Chars
	}
	if e.N == 0 {
		e.N = Window
	}
	if e.Stride == 0 {
		e.Stride = 1
	}
	return e
}

// Validate reports what is wrong with an encoding, if anything.
func (e Encoding) Validate() error {
	if !e.Unit.Valid() {
		return fmt.Errorf("unit must be %q or %q, got %q", Chars, Words, e.Unit)
	}
	if e.N < 1 {
		return fmt.Errorf("n must be >= 1, got %d", e.N)
	}
	if e.Stride < 1 {
		return fmt.Errorf("stride must be >= 1, got %d", e.Stride)
	}
	if e.Stride > e.N {
		return fmt.Errorf("stride %d must be <= n %d: a larger stride would skip units", e.Stride, e.N)
	}
	return nil
}

// Overlap is how many units two consecutive grams share: N - Stride, so
// N - 1 for a sliding window and 0 for groups.
func (e Encoding) Overlap() int { return e.N - e.Stride }

// IsDefault reports whether this is the character trigram of stride 1 - the
// encoding a model file leaves unwritten, and the one Python also speaks.
func (e Encoding) IsDefault() bool { return e == DefaultEncoding() }

// Sliding reports whether consecutive grams overlap at all.
func (e Encoding) Sliding() bool { return e.Stride < e.N }

// String is the compact spec ParseEncoding reads back: "char:3:1".
func (e Encoding) String() string { return fmt.Sprintf("%s:%d:%d", e.Unit, e.N, e.Stride) }

// Describe is the human form: "character trigrams, stride 1 (sliding)".
func (e Encoding) Describe() string {
	unit := "character"
	if e.Unit == Words {
		unit = "word"
	}
	kind := "sliding"
	if !e.Sliding() {
		kind = "groups"
	}
	return fmt.Sprintf("%d-%s grams, stride %d (%s)", e.N, unit, e.Stride, kind)
}

// ParseEncoding reads a spec: "unit[:n[:stride]]", where unit is char / chars /
// character / word / words, n defaults to Window and stride to 1 (a sliding
// window); "unit:n:n" and the shorthand "unit:n:groups" are the
// non-overlapping form.  A few names for the common ones are accepted too:
// "trigram", "bigram", "word-bigram", "word-trigram".
func ParseEncoding(spec string) (Encoding, error) {
	s := strings.ToLower(strings.TrimSpace(spec))
	if s == "" {
		return DefaultEncoding(), nil
	}
	switch s {
	case "default", "trigram", "trigrams":
		return DefaultEncoding(), nil
	case "bigram", "bigrams":
		return Encoding{Chars, 2, 1}, nil
	case "word", "words", "unigram-words", "word-unigram":
		return Encoding{Words, 1, 1}, nil
	case "word-bigram", "word-bigrams", "bigram-words":
		return Encoding{Words, 2, 1}, nil
	case "word-trigram", "word-trigrams", "trigram-words":
		return Encoding{Words, 3, 1}, nil
	}
	parts := strings.Split(s, ":")
	if len(parts) > 3 {
		return Encoding{}, fmt.Errorf("encoding %q: expected unit[:n[:stride]]", spec)
	}
	e := DefaultEncoding()
	switch parts[0] {
	case "char", "chars", "character", "characters", "letter", "letters":
		e.Unit = Chars
	case "word", "words":
		e.Unit = Words
	default:
		return Encoding{}, fmt.Errorf("encoding %q: unit must be char or word, got %q", spec, parts[0])
	}
	if len(parts) > 1 && parts[1] != "" {
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			return Encoding{}, fmt.Errorf("encoding %q: n must be a number, got %q", spec, parts[1])
		}
		e.N = n
	}
	e.Stride = 1
	if len(parts) > 2 && parts[2] != "" {
		switch parts[2] {
		case "groups", "group", "blocks", "block":
			e.Stride = e.N
		case "sliding", "slide":
			e.Stride = 1
		default:
			st, err := strconv.Atoi(parts[2])
			if err != nil {
				return Encoding{}, fmt.Errorf("encoding %q: stride must be a number, got %q", spec, parts[2])
			}
			e.Stride = st
		}
	}
	if err := e.Validate(); err != nil {
		return Encoding{}, fmt.Errorf("encoding %q: %w", spec, err)
	}
	return e, nil
}

// -- units -------------------------------------------------------------------------

// Units is a text cut into units, indexed so that slicing by unit is O(1) -
// which is what the graph does all day (every split, every merge, every
// re-index of a label walks its grams).  starts holds the byte offset of every
// unit plus the end of the text.
type Units struct {
	text   string
	starts []int
	sep    int
}

// Len is the number of units.
func (u Units) Len() int { return len(u.starts) - 1 }

// Text is the whole (normalised) text.
func (u Units) Text() string { return u.text }

// Slice is units [from, to) as text; to < 0 means "to the end".  Out-of-range
// bounds are clamped, and an empty range is "".
func (u Units) Slice(from, to int) string {
	n := u.Len()
	if to < 0 || to > n {
		to = n
	}
	if from < 0 {
		from = 0
	}
	if from >= to {
		return ""
	}
	end := u.starts[to]
	if to < n {
		end -= u.sep // the separator before unit `to` belongs to neither side
	}
	return u.text[u.starts[from]:end]
}

// At is one unit.
func (u Units) At(i int) string { return u.Slice(i, i+1) }

// Units cuts text into the units of this encoding, normalising it on the way:
// a word encoding collapses every run of whitespace to a single space, a
// character encoding leaves the text exactly as it is.
func (e Encoding) Units(text string) Units {
	if e.Unit == Words {
		return wordUnits(text)
	}
	return charUnits(text)
}

func charUnits(text string) Units {
	starts := make([]int, 0, len(text)+1)
	for i := range text {
		starts = append(starts, i)
	}
	starts = append(starts, len(text))
	return Units{text: text, starts: starts, sep: 0}
}

func wordUnits(text string) Units {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return Units{text: "", starts: []int{0}, sep: 1}
	}
	var b strings.Builder
	size := len(fields) - 1
	for _, f := range fields {
		size += len(f)
	}
	b.Grow(size)
	starts := make([]int, 0, len(fields)+1)
	for i, f := range fields {
		if i > 0 {
			b.WriteByte(' ')
		}
		starts = append(starts, b.Len())
		b.WriteString(f)
	}
	starts = append(starts, b.Len())
	return Units{text: b.String(), starts: starts, sep: 1}
}

// Len is the number of units in a text.
func (e Encoding) Len(text string) int {
	if e.Unit == Words {
		return len(strings.Fields(text))
	}
	return utf8.RuneCountInString(text)
}

// Slice is units [from, to) of a text; to < 0 means "to the end".
func (e Encoding) Slice(text string, from, to int) string {
	if e.Unit == Chars {
		return runeSlice(text, from, to)
	}
	return e.Units(text).Slice(from, to)
}

// Join glues unit-aligned pieces back together: nothing between characters, a
// single space between words.  Empty pieces are dropped, so a label that
// contributes no unit adds no separator.
func (e Encoding) Join(parts ...string) string {
	if e.Unit != Words {
		return strings.Join(parts, "")
	}
	kept := parts[:0:0]
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, " ")
}

// Truncate is the first n units of a text; n < 0 leaves it alone.
func (e Encoding) Truncate(text string, n int) string {
	if n < 0 {
		return text
	}
	if e.Unit == Chars {
		return truncateRunes(text, n)
	}
	return e.Slice(text, 0, n)
}

// HasUnitPrefix reports whether text starts with prefix on a unit boundary:
// plain string prefix for characters, whole words for words ("the ca" is not a
// word prefix of "the cat sat", "the cat" is).
func (e Encoding) HasUnitPrefix(text, prefix string) bool {
	if prefix == "" {
		return true
	}
	if e.Unit != Words {
		return strings.HasPrefix(text, prefix)
	}
	return text == prefix || strings.HasPrefix(text, prefix+" ")
}

// -- encoder -----------------------------------------------------------------------

// Encode cuts text into its grams: N units each, Stride units apart.  nil when
// the text holds fewer than N units.
//
//	{Chars, 3, 1} "hello"       -> ["hel", "ell", "llo"]
//	{Chars, 4, 4} "hello there" -> ["hell", "o th", "ere"] -> ["hell", "o th"]
//	{Words, 2, 1} "the cat sat" -> ["the cat", "cat sat"]
//
// The tail that does not fill a whole gram is dropped, exactly as the last two
// characters of a text are dropped by the trigram encoding: Normalize says
// what is left.
func (e Encoding) Encode(text string) []string {
	e = e.WithDefaults()
	if e.Unit == Chars { // the hot path of a training pass: no unit index needed
		runes := []rune(text)
		n := len(runes)
		if n < e.N {
			return nil
		}
		grams := make([]string, (n-e.N)/e.Stride+1)
		for i := range grams {
			at := i * e.Stride
			grams[i] = string(runes[at : at+e.N])
		}
		return grams
	}
	return e.EncodeUnits(e.Units(text))
}

// EncodeUnits is Encode over an already-cut text, for callers that need both.
func (e Encoding) EncodeUnits(u Units) []string {
	n := u.Len()
	if n < e.N {
		return nil
	}
	count := (n-e.N)/e.Stride + 1
	grams := make([]string, count)
	for i := 0; i < count; i++ {
		at := i * e.Stride
		grams[i] = u.Slice(at, at+e.N)
	}
	return grams
}

// Covered is how many units of a text its grams actually cover: everything for
// a sliding window, everything but the ragged tail for groups.  0 when the
// text is too short to encode.
func (e Encoding) Covered(n int) int {
	if n < e.N {
		return 0
	}
	return (n-e.N)/e.Stride*e.Stride + e.N
}

// Normalize is the text this encoding can represent: the units its grams cover,
// written the way the decoder writes them.  For the default encoding that is
// the text itself; a word encoding loses the original spacing, and a grouping
// encoding loses the tail that does not fill a group.  It is what a round trip
// through the graph returns, so it is what a round trip is checked against.
func (e Encoding) Normalize(text string) string {
	e = e.WithDefaults()
	u := e.Units(text)
	covered := e.Covered(u.Len())
	if covered == 0 {
		return ""
	}
	return u.Slice(0, covered)
}

// -- decoder -----------------------------------------------------------------------

// DecodePath decodes node labels in path order into text.  Every label after
// the first contributes its part beyond the overlap; the first contributes
// label[startOffset:] when includeContext is true, else label[startOffset+N:]
// (the deterministic remainder of a compressed node after the matched gram).
// Offsets are in units, and sentinel labels are never skipped here: callers
// strip START / END by id.
func (e Encoding) DecodePath(labels []string, startOffset int, includeContext bool) string {
	e = e.WithDefaults()
	firstCut := startOffset
	if !includeContext {
		firstCut += e.N
	}
	if e.Unit == Chars {
		out := make([]byte, 0, 64)
		for i, label := range labels {
			cut := e.Overlap()
			if i == 0 {
				cut = firstCut
			}
			out = append(out, runeSlice(label, cut, -1)...)
		}
		return string(out)
	}
	parts := make([]string, 0, len(labels))
	for i, label := range labels {
		cut := e.Overlap()
		if i == 0 {
			cut = firstCut
		}
		parts = append(parts, e.Slice(label, cut, -1))
	}
	return e.Join(parts...)
}

// DecodeGrams decodes raw grams in order - the inverse of Encode, up to the
// tail Encode dropped.
func (e Encoding) DecodeGrams(grams []string) string {
	e = e.WithDefaults()
	parts := make([]string, 0, len(grams))
	for i, gram := range grams {
		if i == 0 {
			parts = append(parts, gram)
			continue
		}
		parts = append(parts, e.Slice(gram, e.Overlap(), -1))
	}
	return e.Join(parts...)
}

// -- the default encoding, for callers that have no graph to ask -------------------

// Encode is DefaultEncoding().Encode: the overlapping character trigrams of a
// text.  Code that belongs to a graph encodes with that graph's Encoding
// (Graph.Encode) instead.
func Encode(text string) []string { return DefaultEncoding().Encode(text) }

// DecodePath is DefaultEncoding().DecodePath.
func DecodePath(labels []string, startOffset int, includeContext bool) string {
	return DefaultEncoding().DecodePath(labels, startOffset, includeContext)
}

// -- rune helpers ------------------------------------------------------------------

// runeLen is the character (code point) length of a string, the unit the
// Python implementation measures text in.
func runeLen(s string) int { return utf8.RuneCountInString(s) }

// runeSlice returns s[from:to] in characters (code points); to < 0 means "to the end".
func runeSlice(s string, from, to int) string {
	if from <= 0 && to < 0 {
		return s
	}
	if from < 0 {
		from = 0
	}
	i := 0
	start, end := -1, len(s)
	for bi := range s {
		if i == from {
			start = bi
		}
		if to >= 0 && i == to {
			end = bi
			break
		}
		i++
	}
	if start < 0 {
		if from == i { // from == len(s) in characters
			return ""
		}
		return ""
	}
	if end < start {
		return ""
	}
	return s[start:end]
}

// DecodeTrigrams is the inverse of Encode under the default encoding: the
// first gram in full, then the part of each following one past the overlap.
// A graph decodes with its own encoding (Encoding.DecodeGrams).
func DecodeTrigrams(grams []string) string { return DefaultEncoding().DecodeGrams(grams) }

// truncateRunes returns the first n characters of s.
func truncateRunes(s string, n int) string {
	if n < 0 {
		return s
	}
	i := 0
	for bi := range s {
		if i == n {
			return s[:bi]
		}
		i++
	}
	return s
}

// b64Junk is everything outside the base64 alphabet: a predicted payload is
// routinely interrupted by whitespace or a stray character.
var b64Junk = regexp.MustCompile(`[^A-Za-z0-9+/=]`)

// RepairBase64 decodes the base64 tail of a media text, repairing it first;
// returns (payload, repaired, error).
//
// The media encoders (vision.go, speech.go) pack their payload as base64 into a
// text the network trains on and *predicts*, so what comes back may be cut off,
// padded with junk or interrupted by whitespace.  Characters outside the
// alphabet are dropped, a single dangling character (which can never decode)
// goes with them, the padding is completed, and `repaired` says whether any of
// that changed the text.  The payload comes back as it decodes - callers pad or
// truncate it to the length their format needs.
func RepairBase64(body string) ([]byte, bool, error) {
	clean := strings.TrimRight(b64Junk.ReplaceAllString(body, ""), "=")
	if len(clean)%4 == 1 { // a single dangling character can never decode
		clean = clean[:len(clean)-1]
	}
	padded := clean + strings.Repeat("=", (4-len(clean)%4)%4)
	repaired := padded != strings.TrimSpace(body) // a clean text comes back unchanged, padding included
	payload, err := base64.StdEncoding.DecodeString(padded)
	if err != nil {
		return nil, repaired, fmt.Errorf("the base64 part cannot be decoded: %w", err)
	}
	return payload, repaired, nil
}
