package phonetok

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// The phonetic tokenizer (the port of tokenizer.py): text in, the sounds it is
// made of out, at one of four levels; and back.

// The levels.
const (
	Phoneme     = "phoneme"
	Constituent = "constituent"
	SyllableLvl = "syllable"
	WordLvl     = "word"
)

// Levels are the four, in order.
var Levels = []string{Phoneme, Constituent, SyllableLvl, WordLvl}

// Fixed are the tokens every level's vocabulary starts with, in id order.
var Fixed = func() []string {
	out := append([]string{}, Specials...)
	out = append(out, Boundary)
	return append(out, Pauses...)
}()

const (
	trailingMarks = ".,;:!?…\"')]}»’”"
	leadingMarks  = "(\"'[{«‘“…"
)

var pauseOf = map[rune]string{'.': PauseFull, '!': PauseFull, '…': PauseFull, '?': PauseQuestion, ',': PauseShort,
	';': PauseShort, ':': PauseShort, ')': PauseShort, ']': PauseShort, '}': PauseShort}
var pauseStrength = map[string]int{PauseShort: 1, PauseFull: 2, PauseQuestion: 3}
var dashes = map[string]bool{"-": true, "--": true, "—": true, "–": true}
var clusterRe = regexp.MustCompile(`^(?:[A-Z]{1,2}[012]?)(?:\.[A-Z]{1,2}[012]?)*$`)

// Token is one token: its text, what kind of thing it is, the phones it stands for, and its word.
type Token struct {
	Text   string   `json:"text"`
	Kind   string   `json:"kind"`
	Phones []string `json:"phones,omitempty"`
	Word   int      `json:"word"`
}

// ParseToken says what a token string is: (kind, phones, true), or ok false if it is not a phonetic token.
func ParseToken(text string) (kind string, phones []string, ok bool) {
	if text == "" {
		return "", nil, false
	}
	for _, s := range Specials {
		if text == s {
			return "special", nil, true
		}
	}
	if text == Boundary {
		return "boundary", nil, true
	}
	for _, p := range Pauses {
		if text == p {
			return "pause", nil, true
		}
	}
	if IsPhone(text) {
		if IsVowel(text) {
			return "nucleus", []string{text}, true
		}
		return "phone", []string{text}, true
	}
	onset := strings.HasSuffix(text, "-") && !strings.HasPrefix(text, "-")
	coda := strings.HasPrefix(text, "-") && !strings.HasSuffix(text, "-")
	body := text
	if onset {
		body = text[:len(text)-1]
	} else if coda {
		body = text[1:]
	}
	if !clusterRe.MatchString(body) {
		return "", nil, false
	}
	phones = strings.Split(body, ".")
	vowels := 0
	for _, p := range phones {
		if !IsPhone(p) {
			return "", nil, false
		}
		if IsVowel(p) {
			vowels++
		}
	}
	if onset || coda {
		if vowels > 0 {
			return "", nil, false
		}
		if onset {
			return "onset", phones, true
		}
		return "coda", phones, true
	}
	if len(phones) == 1 {
		if vowels == 1 {
			return "nucleus", phones, true
		}
		return "phone", phones, true
	}
	if vowels <= 1 {
		return "syllable", phones, true
	}
	return "word", phones, true
}

// Vocab is token strings <-> ids: the fixed head, then whatever is read, in order.
type Vocab struct {
	Tokens []string
	ids    map[string]int
	Frozen bool
}

// NewVocab makes a vocabulary from its first tokens.
func NewVocab(tokens []string, frozen bool) *Vocab {
	v := &Vocab{ids: map[string]int{}, Frozen: frozen}
	for _, t := range tokens {
		v.Add(t)
	}
	return v
}

// Len is how many tokens have ids.
func (v *Vocab) Len() int { return len(v.Tokens) }

// Has reports whether a token has an id.
func (v *Vocab) Has(token string) bool { _, ok := v.ids[token]; return ok }

// Add gives a token the next id if it has none.
func (v *Vocab) Add(token string) int {
	if i, ok := v.ids[token]; ok {
		return i
	}
	v.ids[token] = len(v.Tokens)
	v.Tokens = append(v.Tokens, token)
	return len(v.Tokens) - 1
}

// ID is the id of a token; a new token gets the next id if grow and the vocabulary is not frozen, else <unk>.
func (v *Vocab) ID(token string, grow bool) int {
	if i, ok := v.ids[token]; ok {
		return i
	}
	if grow && !v.Frozen {
		return v.Add(token)
	}
	return UnkID
}

// Token is the token of an id, or <unk>.
func (v *Vocab) Token(i int) string {
	if i >= 0 && i < len(v.Tokens) {
		return v.Tokens[i]
	}
	return UNK
}

// Tokenizer is text as sounds at one of four levels.
type Tokenizer struct {
	Level        string
	Stress       bool
	Boundaries   bool
	PausesOn     bool
	Transcriber  *Transcriber
	Vocab        *Vocab
	phonotactics *Phonotactics
}

// NewTokenizer makes a tokenizer at a level over a lexicon (the portable one when nil).
func NewTokenizer(level string, lexicon *Lexicon) (*Tokenizer, error) {
	known := false
	for _, l := range Levels {
		if l == level {
			known = true
		}
	}
	if !known {
		return nil, fmt.Errorf("level must be one of %v, got %q", Levels, level)
	}
	t := &Tokenizer{Level: level, Stress: true, Boundaries: true, PausesOn: true, Transcriber: NewTranscriber(lexicon)}
	if level == Phoneme {
		t.Vocab = NewVocab(Symbols, true)
	} else {
		t.Vocab = NewVocab(Fixed, false)
	}
	return t, nil
}

// Lexicon is the transcriber's lexicon.
func (t *Tokenizer) Lexicon() *Lexicon { return t.Transcriber.Lexicon }

// Phonotactics are the sound habits, fitted on the lexicon the first time they are needed.
func (t *Tokenizer) Phonotactics() *Phonotactics {
	if t.phonotactics == nil {
		t.phonotactics = PhonotacticsFromLexicon(t.Lexicon(), false)
	}
	return t.phonotactics
}

// Describe is the one-line description.
func (t *Tokenizer) Describe() string {
	stress := "kept"
	if !t.Stress {
		stress = "dropped"
	}
	return fmt.Sprintf("%s tokens, stress %s, %d ids, lexicon of %s", t.Level, stress, t.Vocab.Len(), t.Lexicon().Describe())
}

// Piece is one piece of a text: a word to transcribe, a sound token, a boundary
// or a pause given as a token, or punctuation to make a pause of.
type Piece struct {
	Kind string // "word", "sounds", "boundary", "pause", "punct"
	Text string
}

// Pieces cuts a text into pieces (see tokenizer.py's pieces()).
func (t *Tokenizer) Pieces(text string) []Piece {
	var out []Piece
	for _, raw := range strings.Fields(text) {
		if kind, _, ok := ParseToken(raw); ok {
			switch kind {
			case "boundary":
				out = append(out, Piece{"boundary", raw})
			case "pause":
				out = append(out, Piece{"pause", raw})
			case "special":
			default:
				out = append(out, Piece{"sounds", raw})
			}
			continue
		}
		if dashes[raw] {
			out = append(out, Piece{"punct", PauseShort})
			continue
		}
		word := strings.TrimLeft(raw, leadingMarks)
		pause := ""
		for {
			runes := []rune(word)
			if len(runes) == 0 || !strings.ContainsRune(trailingMarks, runes[len(runes)-1]) {
				break
			}
			if mark, ok := pauseOf[runes[len(runes)-1]]; ok && (pause == "" || pauseStrength[mark] > pauseStrength[pause]) {
				pause = mark
			}
			word = string(runes[:len(runes)-1])
		}
		if word != "" {
			out = append(out, Piece{"word", word})
		}
		if pause != "" {
			out = append(out, Piece{"punct", pause})
		}
	}
	return out
}

func (t *Tokenizer) stressed(phones []string) []string {
	if t.Stress {
		return phones
	}
	return StripStress(phones)
}

// Transcribe is the phones of each word of the text, punctuation dropped.
func (t *Tokenizer) Transcribe(text string) [][]string {
	var words [][]string
	var current []string
	for _, piece := range t.Pieces(text) {
		switch piece.Kind {
		case "word":
			if len(current) > 0 {
				words = append(words, current)
			}
			current = nil
			if phones := t.Transcriber.Word(piece.Text); len(phones) > 0 {
				words = append(words, t.stressed(phones))
			}
		case "sounds":
			_, phones, _ := ParseToken(piece.Text)
			current = append(current, phones...)
		default:
			if len(current) > 0 {
				words = append(words, current)
			}
			current = nil
		}
	}
	if len(current) > 0 {
		words = append(words, current)
	}
	return words
}

// Tokenize is the tokens of a text at this level, boundaries and pauses included.
func (t *Tokenizer) Tokenize(text string) []Token {
	var out []Token
	wordIndex := 0
	var current []string
	last := ""
	flush := func() {
		if len(current) > 0 {
			if last == "word" && t.Boundaries {
				out = append(out, Token{Text: Boundary, Kind: "boundary", Word: -1})
			}
			out = append(out, t.wordTokens(t.stressed(current), wordIndex)...)
			wordIndex++
			last = "word"
		}
		current = nil
	}
	for _, piece := range t.Pieces(text) {
		switch piece.Kind {
		case "word":
			flush()
			if phones := t.Transcriber.Word(piece.Text); len(phones) > 0 {
				current = phones
				flush()
			}
		case "sounds":
			_, phones, _ := ParseToken(piece.Text)
			current = append(current, phones...)
		case "boundary": // a boundary given as a token stays, wherever it is
			flush()
			out = append(out, Token{Text: Boundary, Kind: "boundary", Word: -1})
			last = "boundary"
		case "pause": // so does a pause given as a token
			flush()
			out = append(out, Token{Text: piece.Text, Kind: "pause", Word: -1})
			last = "pause"
		default: // punctuation: one pause, after something, never two in a row
			flush()
			if t.PausesOn && last != "pause" && len(out) > 0 {
				out = append(out, Token{Text: piece.Text, Kind: "pause", Word: -1})
				last = "pause"
			}
		}
	}
	flush()
	return out
}

func (t *Tokenizer) wordTokens(phones []string, word int) []Token {
	switch t.Level {
	case Phoneme:
		out := make([]Token, len(phones))
		for i, p := range phones {
			kind := "phone"
			if IsVowel(p) {
				kind = "nucleus"
			}
			out[i] = Token{p, kind, []string{p}, word}
		}
		return out
	case WordLvl:
		return []Token{{strings.Join(phones, "."), "word", append([]string(nil), phones...), word}}
	}
	var out []Token
	for _, s := range Syllabify(phones, true) {
		if t.Level == SyllableLvl {
			out = append(out, Token{s.Text(), "syllable", s.Phones(), word})
			continue
		}
		if len(s.Onset) > 0 {
			out = append(out, Token{strings.Join(s.Onset, ".") + "-", "onset", s.Onset, word})
		}
		out = append(out, Token{s.Nucleus, "nucleus", []string{s.Nucleus}, word})
		if len(s.Coda) > 0 {
			out = append(out, Token{"-" + strings.Join(s.Coda, "."), "coda", s.Coda, word})
		}
	}
	return out
}

// Tokens are the token strings.
func (t *Tokenizer) Tokens(text string) []string {
	toks := t.Tokenize(text)
	out := make([]string, len(toks))
	for i, tk := range toks {
		out[i] = tk.Text
	}
	return out
}

// Text is the text form: the tokens joined by single spaces.  Text(Text(x)) == Text(x).
func (t *Tokenizer) Text(text string) string { return strings.Join(t.Tokens(text), " ") }

// Encode is the ids of the tokens; an unseen token grows the vocabulary (grow) or is <unk>.
func (t *Tokenizer) Encode(text string, grow bool) []int {
	toks := t.Tokens(text)
	out := make([]int, len(toks))
	for i, tk := range toks {
		out[i] = t.Vocab.ID(tk, grow)
	}
	return out
}

// Parse turns token strings back into Tokens; an error for one that is not phonetic.
func (t *Tokenizer) Parse(tokens []string) ([]Token, error) {
	var out []Token
	word := 0
	for _, text := range tokens {
		kind, phones, ok := ParseToken(text)
		if !ok {
			return nil, fmt.Errorf("%q is not a phonetic token", text)
		}
		switch kind {
		case "boundary", "pause":
			if len(out) > 0 && out[len(out)-1].Kind != "boundary" && out[len(out)-1].Kind != "pause" {
				word++
			}
			out = append(out, Token{text, kind, nil, -1})
		case "special":
			out = append(out, Token{text, kind, nil, -1})
		default:
			out = append(out, Token{text, kind, phones, word})
		}
	}
	return out, nil
}

// WordsOf groups token strings back into the phones of each word.
func (t *Tokenizer) WordsOf(tokens []string) [][]string {
	var words [][]string
	var current []string
	for _, tk := range tokens {
		kind, phones, ok := ParseToken(tk)
		if !ok {
			continue
		}
		switch kind {
		case "boundary", "pause", "special":
			if len(current) > 0 {
				words = append(words, current)
			}
			current = nil
		default:
			current = append(current, phones...)
		}
	}
	if len(current) > 0 {
		words = append(words, current)
	}
	return words
}

// WordsOfIDs is WordsOf over ids.
func (t *Tokenizer) WordsOfIDs(ids []int) [][]string {
	return t.WordsOf(t.tokensOfIDs(ids))
}

func (t *Tokenizer) tokensOfIDs(ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = t.Vocab.Token(id)
	}
	return out
}

// Decode is the sounds spelled back as words, with the pauses as punctuation.
func (t *Tokenizer) Decode(tokens []string) string {
	var out []string
	var current []string
	for _, tk := range tokens {
		kind, phones, ok := ParseToken(tk)
		if !ok {
			continue
		}
		switch kind {
		case "boundary", "pause", "special":
			if len(current) > 0 {
				out = append(out, t.Transcriber.Spell(current))
				current = nil
			}
			if kind == "pause" && len(out) > 0 {
				out[len(out)-1] += tk
			}
		default:
			current = append(current, phones...)
		}
	}
	if len(current) > 0 {
		out = append(out, t.Transcriber.Spell(current))
	}
	return strings.Join(out, " ")
}

// DecodeIDs is Decode over ids.
func (t *Tokenizer) DecodeIDs(ids []int) string { return t.Decode(t.tokensOfIDs(ids)) }

// FeaturesOf is the articulatory features of a token: a phone's own, the mean over a cluster's or a
// syllable's phones, zeros for a boundary, a pause or a special.
func (t *Tokenizer) FeaturesOf(token string) []float64 {
	out := make([]float64, FeatureDim)
	_, phones, ok := ParseToken(token)
	if !ok || len(phones) == 0 {
		return out
	}
	for _, p := range phones {
		f, _ := Features(p)
		for i := range out {
			out[i] += f[i]
		}
	}
	for i := range out {
		out[i] /= float64(len(phones))
	}
	return out
}

// Pronounce is the phones of one word.
func (t *Tokenizer) Pronounce(word string) []string { return t.stressed(t.Transcriber.Word(word)) }

// IPA is the text in the IPA, a stress mark before each stressed syllable.
func (t *Tokenizer) IPA(text string) string {
	var words []string
	for _, phones := range t.Transcribe(text) {
		var b strings.Builder
		for _, s := range Syllabify(phones, true) {
			switch s.Stress() {
			case 1:
				b.WriteString("ˈ")
			case 2:
				b.WriteString("ˌ")
			}
			b.WriteString(ToIPA(s.Phones(), false))
		}
		words = append(words, b.String())
	}
	return strings.Join(words, " ")
}

// Syllables are the syllables of one word.
func (t *Tokenizer) Syllables(word string) []Syllable { return Syllabify(t.Pronounce(word), true) }

// RhymesWith reports whether two words rhyme.
func (t *Tokenizer) RhymesWith(a, b string) bool { return Rhymes(t.Pronounce(a), t.Pronounce(b)) }

// AlliteratesWith reports whether two words alliterate.
func (t *Tokenizer) AlliteratesWith(a, b string) bool {
	return Alliterates(t.Pronounce(a), t.Pronounce(b))
}

// Affinity is how well sound b follows sound a (bits of mutual information; 0 is chance).
func (t *Tokenizer) Affinity(a, b string) float64 { return t.Phonotactics().Affinity(a, b) }

// Score is how English a word sounds: the mean log-probability of its sound transitions.
func (t *Tokenizer) Score(word string) float64 { return t.Phonotactics().Score(t.Pronounce(word)) }

// WellFormed reports whether an English speaker could say the word.
func (t *Tokenizer) WellFormed(word string) bool { return WellFormed(t.Pronounce(word)) }

// Coin is a new, pronounceable word (spelled as it sounds), built from the sound habits with a seeded RNG.
func (t *Tokenizer) Coin(seed uint32, syllables int) string {
	rng := NewMT(seed)
	phones := t.Phonotactics().Build(rng, syllables, 1, 3, t.Lexicon().Items(), 50)
	if len(phones) == 0 {
		return ""
	}
	return t.Transcriber.Spell(phones)
}

// Blend is a portmanteau of two words at the joint where their sounds meet best.
func (t *Tokenizer) Blend(a, b string) (string, error) {
	phones, _, _, err := t.Phonotactics().Blend(t.Pronounce(a), t.Pronounce(b))
	if err != nil {
		return "", err
	}
	return t.Transcriber.Spell(phones), nil
}

// Explanation is one row of Explain.
type Explanation struct {
	Word      string   `json:"word"`
	Phones    []string `json:"phones"`
	How       string   `json:"how"`
	Syllables []string `json:"syllables"`
	IPA       string   `json:"ipa"`
	Score     *float64 `json:"score"`
}

// Explain is one row per word: the spelling, its phones, where they came from, the syllables and the IPA.
func (t *Tokenizer) Explain(text string) []Explanation {
	var rows []Explanation
	for _, piece := range t.Pieces(text) {
		if piece.Kind != "word" {
			continue
		}
		phones, how := t.Transcriber.Explain(piece.Text)
		phones = t.stressed(phones)
		var syl []string
		for _, s := range Syllabify(phones, true) {
			syl = append(syl, s.Text())
		}
		row := Explanation{Word: piece.Text, Phones: phones, How: how, Syllables: syl, IPA: t.IPA(piece.Text)}
		if len(phones) > 0 {
			score := round3(t.Phonotactics().Score(phones))
			row.Score = &score
		}
		rows = append(rows, row)
	}
	return rows
}

func round3(x float64) float64 {
	return float64(int64(x*1000+copysign(0.5, x))) / 1000
}

func copysign(mag, sign float64) float64 {
	if sign < 0 {
		return -mag
	}
	return mag
}

// -- persistence -----------------------------------------------------------------

// Document is what a tokenizer file holds: everything learned by reading.
type Document struct {
	Format     string              `json:"format"`
	Version    int                 `json:"version"`
	Level      string              `json:"level"`
	Stress     bool                `json:"stress"`
	Boundaries bool                `json:"boundaries"`
	Pauses     bool                `json:"pauses"`
	Vocab      []string            `json:"vocab"`
	Memory     map[string][]string `json:"memory"`
	Counts     map[string]int      `json:"counts"`
}

// ToDocument is the tokenizer's state.
func (t *Tokenizer) ToDocument() Document {
	memory := map[string][]string{}
	for w, p := range t.Transcriber.Memory {
		memory[w] = append([]string(nil), p...)
	}
	return Document{
		Format: "phonetok", Version: 1, Level: t.Level, Stress: t.Stress, Boundaries: t.Boundaries, Pauses: t.PausesOn,
		Vocab: append([]string(nil), t.Vocab.Tokens...), Memory: memory, Counts: t.Transcriber.Counts,
	}
}

// LoadDocument reads a state into this tokenizer.
func (t *Tokenizer) LoadDocument(d Document) error {
	if d.Format != "phonetok" {
		return fmt.Errorf("not a phonetok tokenizer document")
	}
	if d.Level != "" && d.Level != t.Level {
		return fmt.Errorf("the file holds a %s tokenizer, this one is %s", d.Level, t.Level)
	}
	t.Stress, t.Boundaries, t.PausesOn = d.Stress, d.Boundaries, d.Pauses
	if t.Level != Phoneme {
		tokens := d.Vocab
		if len(tokens) == 0 {
			tokens = Fixed
		}
		t.Vocab = NewVocab(tokens, false)
	}
	for _, w := range sortedKeysOf(d.Memory) {
		if _, ok := t.Transcriber.Memory[w]; !ok {
			t.Transcriber.memoryOrder = append(t.Transcriber.memoryOrder, w)
		}
		t.Transcriber.Memory[w] = append([]string(nil), d.Memory[w]...)
	}
	for w, c := range d.Counts {
		t.Transcriber.Counts[w] += c
	}
	return nil
}

func sortedKeysOf(m map[string][]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sortStrings(keys)
	return keys
}

// Save writes the tokenizer's state as JSON.
func (t *Tokenizer) Save(path string) error {
	data, err := json.MarshalIndent(t.ToDocument(), "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadTokenizer reads a tokenizer file over a lexicon (the portable one when nil).
func LoadTokenizer(path string, lexicon *Lexicon) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d Document
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, err
	}
	level := d.Level
	if level == "" {
		level = Phoneme
	}
	t, err := NewTokenizer(level, lexicon)
	if err != nil {
		return nil, err
	}
	return t, t.LoadDocument(d)
}
