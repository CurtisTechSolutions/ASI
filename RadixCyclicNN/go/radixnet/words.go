package radixnet

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// The word alphabet (see ../../SPEC-WordNGrams.md).
//
// Not one of the graph's structural rules mentions a character: a label is a
// sequence of symbols, an edge exists where two labels overlap by Overlap
// symbols, the index maps a Window-symbol key to (node, offset).  A word
// n-gram model is therefore this model over an alphabet whose symbols are
// words, and a symbol is one code point:
//
//	id 0          -> U+0100          the unknown word
//	id i (i >= 1) -> U+0100 + i      skipping the surrogate block D800..DFFF
//
// Starting past Latin-1 keeps every label out of the range text, sentinels and
// punctuation live in; skipping the surrogates keeps every label a string Go,
// Python and Rust can all hold and write (Go replaces a lone surrogate with
// U+FFFD, which would silently corrupt a model).
const (
	// WordBase is the code point of word 0.
	WordBase = 0x0100
	// SurrogateLo and SurrogateHi bound the block no word may land in.
	SurrogateLo = 0xD800
	SurrogateHi = 0xDFFF
	surrogates  = SurrogateHi - SurrogateLo + 1
	// MaxWords is 1 111 808: every code point above Latin-1 that is not a surrogate.
	MaxWords = 0x110000 - WordBase - surrogates
	// UnknownWord is word 0: a word the model has never read maps to it at
	// prediction and scoring time, and two different unread words are the same symbol.
	UnknownWord = "<unk>"
	// UnknownID is that word's id.
	UnknownID = 0
	// WordUnits names the alphabet in a model file and in every report.
	WordUnits = "words"
	// CharUnits is what every other kind counts in.
	CharUnits = "chars"
)

// WordSymbol is the code point that carries a word id.
func WordSymbol(id int) (rune, error) {
	if id < 0 || id >= MaxWords {
		return 0, fmt.Errorf("word id must lie in [0, %d), got %d", MaxWords, id)
	}
	code := rune(WordBase + id)
	if code >= SurrogateLo {
		code += surrogates
	}
	return code, nil
}

// SymbolWord is the inverse of WordSymbol; an error for a code point that carries no word.
func SymbolWord(symbol rune) (int, error) {
	if symbol < WordBase || (symbol >= SurrogateLo && symbol <= SurrogateHi) {
		return 0, fmt.Errorf("U+%04X is not a word symbol", symbol)
	}
	id := int(symbol) - WordBase
	if symbol > SurrogateHi {
		id -= surrogates
	}
	return id, nil
}

// SplitWords is the whole tokeniser: maximal runs of code points the Unicode
// White_Space property does not cover.  Punctuation stays attached to the word
// it touches and case is kept ("mat." and "mat" are two words, "The" and "the"
// are two words), because every refinement of that rule is a step towards a
// vocabulary that has to be designed, versioned and defended.
func SplitWords(text string) []string { return strings.FieldsFunc(text, unicode.IsSpace) }

// Vocabulary is a word model's alphabet: a two-way map between words and single
// code points.  Id 0 is always UnknownWord; every other word is in it because
// training read it, in the order it was first read.  Nothing is frozen, pruned
// or learned - there is no tokeniser here and no training run before the
// training run.
type Vocabulary struct {
	Words []string
	ids   map[string]int
}

// NewVocabulary is an empty vocabulary: the unknown word and nothing else.
func NewVocabulary() *Vocabulary {
	return &Vocabulary{Words: []string{UnknownWord}, ids: map[string]int{UnknownWord: UnknownID}}
}

// VocabularyFrom rebuilds a vocabulary from a file's list, which is in id order.
func VocabularyFrom(words []string) (*Vocabulary, error) {
	v := NewVocabulary()
	if len(words) == 0 {
		return v, nil
	}
	if words[0] != UnknownWord {
		return nil, fmt.Errorf("a vocabulary starts with %q, got %q", UnknownWord, words[0])
	}
	for _, word := range words[1:] {
		if _, seen := v.ids[word]; seen {
			return nil, fmt.Errorf("the vocabulary holds %q twice", word)
		}
		if _, err := v.Add(word); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// Len is how many words the vocabulary holds, the unknown word included.
func (v *Vocabulary) Len() int { return len(v.Words) }

// ID is the id of a word, or UnknownID when it has never been read.
func (v *Vocabulary) ID(word string) int {
	if id, ok := v.ids[word]; ok {
		return id
	}
	return UnknownID
}

// Add returns the id of a word, giving it the next one if it is new.
func (v *Vocabulary) Add(word string) (int, error) {
	if id, ok := v.ids[word]; ok {
		return id, nil
	}
	if len(v.Words) >= MaxWords {
		return 0, fmt.Errorf("a word model holds at most %d words", MaxWords)
	}
	id := len(v.Words)
	v.Words = append(v.Words, word)
	v.ids[word] = id
	return id, nil
}

// Encode is text as the graph's symbols, one code point per word.  grow gives
// an unread word the next id (training); without it an unread word is
// UnknownWord, whose windows are not in the index, so the search and the score
// charge it what they charge any unknown transition.
func (v *Vocabulary) Encode(text string, grow bool) string {
	words := SplitWords(text)
	if len(words) == 0 {
		return ""
	}
	var b strings.Builder
	b.Grow(len(words) * 3)
	for _, word := range words {
		id := v.ID(word)
		if grow {
			id, _ = v.Add(word) // the cap is 1 111 808 words; past it everything is <unk>
		}
		symbol, err := WordSymbol(id)
		if err != nil {
			symbol = WordBase // the unknown word
		}
		b.WriteRune(symbol)
	}
	return b.String()
}

// Decode is the words a symbol string stands for, joined by single spaces.  A
// symbol this vocabulary has no word for decodes to UnknownWord: a file is
// checked when it is loaded (see checkVocabulary), so past that point this
// cannot happen.
func (v *Vocabulary) Decode(symbols string) string {
	if symbols == "" {
		return ""
	}
	out := make([]string, 0, 8)
	for _, symbol := range symbols {
		id, err := SymbolWord(symbol)
		if err != nil || id >= len(v.Words) {
			out = append(out, UnknownWord)
			continue
		}
		out = append(out, v.Words[id])
	}
	return strings.Join(out, " ")
}

// List is the vocabulary in id order, as the model file carries it.
func (v *Vocabulary) List() []string { return append([]string(nil), v.Words...) }

// -- the graph's side of the alphabet ---------------------------------------------------------------

// IsWords reports whether this graph's symbols are words rather than characters.
func (g *Graph) IsWords() bool { return g != nil && g.Vocab != nil }

// Units is what the graph counts in: "words" on a word graph, "chars" everywhere else.
func (g *Graph) Units() string {
	if g.IsWords() {
		return WordUnits
	}
	return CharUnits
}

// TextOf is a label as text, for anything that reports one: the identity on a
// character graph, and the words the code points stand for on a word graph.  A
// sentinel label is not made of words and comes back as it is.
func (g *Graph) TextOf(label string) string {
	if !g.IsWords() || label == StartLabel || label == EndLabel || label == BackLabel {
		return label
	}
	return g.Vocab.Decode(label)
}

// SymbolsOf is text as this graph's symbols, the inverse of TextOf, for
// anything that looks a label up.  An unread word comes back as the unknown
// symbol: the vocabulary only grows while training.
func (g *Graph) SymbolsOf(text string) string {
	if !g.IsWords() {
		return text
	}
	return g.Vocab.Encode(text, false)
}

// checkVocabulary makes sure every symbol every label carries is a word this
// graph's vocabulary holds - the check that turns a corrupt file into an error
// instead of a decoding surprise much later.
func (g *Graph) checkVocabulary() error {
	if !g.IsWords() {
		return nil
	}
	size := g.Vocab.Len()
	for node := First; node < len(g.Labels); node++ {
		if !g.Alive[node] {
			continue
		}
		for _, symbol := range g.Labels[node] {
			id, err := SymbolWord(symbol)
			if err != nil {
				return fmt.Errorf("node %d label holds %v", node, err)
			}
			if id >= size {
				return fmt.Errorf("node %d holds word %d, past the end of a vocabulary of %d", node, id, size)
			}
		}
	}
	return nil
}

// -- the model's side -------------------------------------------------------------------------------

// IsWords reports whether this model's symbols are words.
func (m *Model) IsWords() bool { return m != nil && m.G.IsWords() }

// Vocabulary is the word model's alphabet, or nil on any other kind.
func (m *Model) Vocabulary() *Vocabulary {
	if !m.IsWords() {
		return nil
	}
	return m.G.Vocab
}

// Units is what the model counts in: lengths, caps and per-symbol scores are in
// these ("words" on a word model, "chars" everywhere else).
func (m *Model) Units() string { return m.G.Units() }

// Symbols is text as the graph's symbols; grow gives an unread word the next id.
func (m *Model) Symbols(text string, grow bool) string {
	if !m.IsWords() {
		return text
	}
	return m.G.Vocab.Encode(text, grow)
}

// Words is the text a symbol string stands for (the identity off a word model).
func (m *Model) Words(symbols string) string {
	if !m.IsWords() {
		return symbols
	}
	return m.G.Vocab.Decode(symbols)
}

// Normalise is text as a word model can represent it: its words joined by
// single spaces.  A word model's round trip costs the original whitespace and
// nothing else, and this is what it costs.
func (m *Model) Normalise(text string) string {
	if !m.IsWords() {
		return text
	}
	return strings.Join(SplitWords(text), " ")
}

// decodeResult puts one walk's texts and labels back into words.
func (m *Model) decodeResult(r *PathResult) *PathResult {
	if r == nil || !m.IsWords() {
		return r
	}
	r.Text = m.Words(r.Text)
	r.FullText = m.Words(r.FullText)
	for i, label := range r.Labels {
		r.Labels[i] = m.G.TextOf(label)
	}
	return r
}

// decodePrediction puts a whole prediction - the best walk and both lists - back into words.
func (m *Model) decodePrediction(p *Prediction) *Prediction {
	if p == nil || !m.IsWords() {
		return p
	}
	for _, r := range p.Top {
		m.decodeResult(r)
	}
	for _, r := range p.Bottom {
		m.decodeResult(r)
	}
	m.decodeResult(&p.PathResult) // the prediction's own labels are a copy of the best walk's, not the same slice
	return p
}

// WordRow is one word of the alphabet with how much of the graph holds it.
type WordRow struct {
	Word     string `json:"word"`
	ID       int    `json:"id"`
	Trigrams int    `json:"trigrams"`
}

// TopWords is the most read words: Trigrams is how many of the graph's windows
// the word appears in - what the graph actually knows about it, and the one
// count that survives compression.  Ties keep id order; limit 0 is all of them.
func (m *Model) TopWords(limit int) []WordRow {
	if !m.IsWords() {
		return nil
	}
	g := m.G
	g.Prepare()
	vocab := g.Vocab
	counts := make([]int, vocab.Len())
	for trigram := range g.index {
		for _, symbol := range trigram {
			if id, err := SymbolWord(symbol); err == nil && id < len(counts) {
				counts[id]++
			}
		}
	}
	rows := make([]WordRow, 0, len(counts))
	for id, word := range vocab.Words {
		if id == UnknownID && counts[id] == 0 {
			continue // <unk> only shows up where the graph really holds it
		}
		rows = append(rows, WordRow{Word: word, ID: id, Trigrams: counts[id]})
	}
	sortWordRows(rows)
	if limit > 0 && limit < len(rows) {
		rows = rows[:limit]
	}
	return rows
}

func sortWordRows(rows []WordRow) {
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Trigrams != rows[j].Trigrams {
			return rows[i].Trigrams > rows[j].Trigrams
		}
		return rows[i].ID < rows[j].ID
	})
}

// wordParts maps every text of a source to its words as the reader reads it:
// one reader, in corpus order, so the ids a training run hands out do not
// depend on how the chunks raced (see passesSource).
type wordParts struct {
	Parts
	vocab *Vocabulary
}

func (w wordParts) Each(part int, fn func(string) error) error {
	return w.Parts.Each(part, func(text string) error { return fn(w.vocab.Encode(text, true)) })
}
