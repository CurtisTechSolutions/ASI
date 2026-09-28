package radixnet

// The traditional LLM tokenizer: byte-level byte-pair encoding, the way GPT-2
// and its descendants read text - the Go twin of radixnet/bpe.py, token for
// token (tests/tokens_fixture.json holds the two to it).
//
// A text is cut into pre-tokens by GPT-2's pattern (a word with the space
// before it, a run of digits, a run of punctuation, a run of whitespace, an
// English contraction), each pre-token's UTF-8 bytes are written in GPT-2's
// byte alphabet (Ġ for the space, Ċ for the newline) and merged pairwise by a
// ranked list of merges - the pair with the best rank anywhere in the
// pre-token is merged everywhere it occurs, left to right - until no pair of
// the list is left.  Every byte is a token of its own, so nothing is
// unreadable.  Ids 0-255 are the bytes, the merged tokens follow in rank
// order, and <|endoftext|> closes the vocabulary.  A space is put before
// every text, and Decode takes it away again.
//
// The text form is what a graph over tokens is built from: a token that is a
// space and printable ASCII is written bare ("Ġcat" is cat), any other glued
// on with ⁀ ("ing" is ⁀ing), units joined by single spaces.  It is
// idempotent - any run of its units reads back as itself - which is what lets
// a label cut into the tokens it was made of (../../SPEC-Tokens.md).

import (
	_ "embed"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

// BPEGlue is the mark of a glued token in the text form: ⁀ (U+2040 CHARACTER
// TIE), outside the byte alphabet, so no token holds it.
const BPEGlue = "⁀"

// EndOfText is the special token that closes the vocabulary.
const EndOfText = "<|endoftext|>"

// TokenizerEnv names a merges file to read instead of the bundled one.
const TokenizerEnv = "RADIXNET_TOKENIZER"

// bundledMerges is radixnet/data/merges.txt, copied beside this package so it
// can be embedded (make tokenizer-merges keeps the copy in step).
//
//go:embed data/merges.txt
var bundledMerges string

// -- the byte alphabet -----------------------------------------------------------------

var (
	byteChars [256]rune
	charBytes = map[rune]byte{}
)

func init() {
	moved := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			byteChars[b] = rune(b)
		} else {
			byteChars[b] = rune(256 + moved)
			moved++
		}
		charBytes[byteChars[b]] = byte(b)
	}
}

// ByteText is bytes written in the byte alphabet: " cat" -> "Ġcat".
func ByteText(data []byte) string {
	var b strings.Builder
	b.Grow(len(data) * 2)
	for _, c := range data {
		b.WriteRune(byteChars[c])
	}
	return b.String()
}

// TextBytes is the bytes a string of the byte alphabet stands for; ok is false
// when a character is not in it.
func TextBytes(text string) ([]byte, bool) {
	out := make([]byte, 0, len(text))
	for _, r := range text {
		c, found := charBytes[r]
		if !found {
			return nil, false
		}
		out = append(out, c)
	}
	return out, true
}

// -- the pre-tokenizer -------------------------------------------------------------------

// The four classes of the pre-tokenizer.
const (
	classSpace  = 'S'
	classDigit  = 'D'
	classPunct  = 'P'
	classLetter = 'L'
)

// punctRanges is where the punctuation class lies: every ASCII character that
// is not a letter, a digit or whitespace, and a fixed list of Unicode blocks -
// radixnet/bpe.py's _PUNCT_RANGES, range for range.
var punctRanges = [...][2]rune{
	{0x0000, 0x002F}, {0x003A, 0x0040}, {0x005B, 0x0060}, {0x007B, 0x00BF}, {0x00D7, 0x00D7},
	{0x00F7, 0x00F7}, {0x2000, 0x206F}, {0x20A0, 0x20CF}, {0x2190, 0x2BFF}, {0x2E00, 0x2E7F},
	{0x3001, 0x3003}, {0x3008, 0x3011}, {0x3014, 0x301F}, {0xFE00, 0xFE0F}, {0xFF01, 0xFF0F},
	{0xFF1A, 0xFF20}, {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65}, {0x1F000, 0x1FAFF}, {0xE0020, 0xE007F},
}

// isWhiteSpace is the Unicode White_Space property, written out: the 25 code
// points the three ports agree are space.
func isWhiteSpace(r rune) bool {
	switch r {
	case '\t', '\n', '\v', '\f', '\r', ' ', 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200A
}

// charClass is the class of one character: space, digit, punctuation or letter
// (everything else, every script's letters included).
func charClass(r rune) byte {
	if isWhiteSpace(r) {
		return classSpace
	}
	if r >= '0' && r <= '9' {
		return classDigit
	}
	for _, span := range punctRanges {
		if r < span[0] {
			break
		}
		if r <= span[1] {
			return classPunct
		}
	}
	return classLetter
}

func asciiLowerRune(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + 32
	}
	return r
}

// contraction is how many characters of 's 't 're 've 'm 'll 'd (either case)
// start at rs[i], or 0.
func contraction(rs []rune, i int) int {
	if rs[i] != '\'' || i+1 >= len(rs) {
		return 0
	}
	switch asciiLowerRune(rs[i+1]) {
	case 's', 't', 'm', 'd':
		return 2
	}
	if i+2 < len(rs) {
		a, b := asciiLowerRune(rs[i+1]), asciiLowerRune(rs[i+2])
		if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
			return 3
		}
	}
	return 0
}

// Pretokenize cuts a text into GPT-2's pre-tokens:
// 's|'t|'re|'ve|'m|'ll|'d| ?L+| ?D+| ?P+|\s+(?!\S)|\s+ over the classes above.
// The pieces join back into the text exactly.
func Pretokenize(text string) []string {
	rs := []rune(text)
	n := len(rs)
	var out []string
	for i := 0; i < n; {
		if k := contraction(rs, i); k > 0 {
			out = append(out, string(rs[i:i+k]))
			i += k
			continue
		}
		j := i
		if rs[i] == ' ' && i+1 < n && charClass(rs[i+1]) != classSpace {
			j = i + 1
		}
		if c := charClass(rs[j]); c != classSpace {
			k := j + 1
			for k < n && charClass(rs[k]) == c {
				k++
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		k := i + 1
		for k < n && charClass(rs[k]) == classSpace {
			k++
		}
		if k < n && k-i >= 2 { // leave the last whitespace character to what follows
			k--
		}
		out = append(out, string(rs[i:k]))
		i = k
	}
	return out
}

// -- the tokenizer ------------------------------------------------------------------------

type bpePair [2]string

// BPETokenizer is a byte-level BPE tokenizer: ranked merges over the byte
// alphabet, ids, and the text form.  Safe for concurrent use.
type BPETokenizer struct {
	Vocab     []string    // id -> token, in the byte alphabet: the 256 bytes, then the merged tokens
	Merges    [][2]string // by rank
	Special   []string
	Note      string
	ids       map[string]int
	ranks     map[bpePair]int
	specialAt map[string]int

	mu    sync.Mutex
	words map[string][]string // the tokens of a pre-token
	bare  map[string]bool     // whether a bare piece reads as a token
}

const bpeCacheLimit = 1 << 16

// NewBPETokenizer is a tokenizer over merges (in rank order) and special tokens.
func NewBPETokenizer(merges [][2]string, special []string, note string) (*BPETokenizer, error) {
	t := &BPETokenizer{
		Vocab: make([]string, 256), ids: map[string]int{}, ranks: map[bpePair]int{}, specialAt: map[string]int{},
		Note: note, words: map[string][]string{}, bare: map[string]bool{},
	}
	for b := 0; b < 256; b++ {
		t.Vocab[b] = string(byteChars[b])
		t.ids[t.Vocab[b]] = b
	}
	for _, m := range merges {
		if err := t.addMerge(m[0], m[1]); err != nil {
			return nil, err
		}
	}
	for _, s := range special {
		if _, dup := t.specialAt[s]; dup {
			return nil, fmt.Errorf("the special token %q is listed twice", s)
		}
		t.specialAt[s] = len(t.Special)
		t.Special = append(t.Special, s)
	}
	return t, nil
}

func (t *BPETokenizer) addMerge(left, right string) error {
	p := bpePair{left, right}
	if _, dup := t.ranks[p]; dup {
		return fmt.Errorf("merge %d (%s %s) is listed twice", len(t.Merges)+1, left, right)
	}
	for _, side := range p {
		if _, ok := t.ids[side]; !ok {
			return fmt.Errorf("merge %d (%s %s): %q is not a token yet - a merge joins two tokens the bytes and "+
				"the merges before it made", len(t.Merges)+1, left, right, side)
		}
	}
	t.ranks[p] = len(t.Merges)
	t.Merges = append(t.Merges, [2]string{left, right})
	if _, ok := t.ids[left+right]; !ok {
		t.ids[left+right] = len(t.Vocab)
		t.Vocab = append(t.Vocab, left+right)
	}
	return nil
}

// ParseMerges reads a merges file: GPT-2's merges.txt - lines of "left right",
// '#' starting a comment line and "#note:" lines kept as the note.
func ParseMerges(text string) (*BPETokenizer, error) {
	var merges [][2]string
	var notes []string
	for number, raw := range strings.Split(text, "\n") {
		line := strings.TrimRight(raw, "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "#note:") {
				notes = append(notes, strings.TrimSpace(line[len("#note:"):]))
			}
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("line %d: a merge is two tokens separated by one space, got %q", number+1, line)
		}
		for _, side := range parts {
			if _, ok := TextBytes(side); !ok {
				return nil, fmt.Errorf("line %d: %q is not written in the byte alphabet", number+1, side)
			}
		}
		merges = append(merges, [2]string{parts[0], parts[1]})
	}
	t, err := NewBPETokenizer(merges, []string{EndOfText}, strings.Join(notes, "\n"))
	if err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	return t, nil
}

// LoadMerges reads a merges file from disk.
func LoadMerges(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseMerges(string(data))
}

// BundledTokenizer is the tokenizer over the merges that ship with the package.
func BundledTokenizer() (*BPETokenizer, error) { return ParseMerges(bundledMerges) }

var (
	bpeOnce sync.Once
	bpeTok  *BPETokenizer
	bpeErr  error
)

// DefaultTokenizer is the tokenizer the token unit reads through: the merges
// file RADIXNET_TOKENIZER names, else the bundled merges - read once and kept.
// A model's tokens mean nothing without the merges that made them: train and
// predict with the same file.
func DefaultTokenizer() (*BPETokenizer, error) {
	bpeOnce.Do(func() {
		path := os.Getenv(TokenizerEnv)
		where := "the bundled merges"
		if path != "" {
			where = TokenizerEnv + "=" + path
			bpeTok, bpeErr = LoadMerges(path)
		} else {
			bpeTok, bpeErr = BundledTokenizer()
		}
		if bpeErr != nil {
			bpeErr = fmt.Errorf("the token unit cannot read its merges (%s): %w", where, bpeErr)
		}
	})
	return bpeTok, bpeErr
}

// VocabSize is every id: the bytes, the merged tokens and the special tokens.
func (t *BPETokenizer) VocabSize() int { return len(t.Vocab) + len(t.Special) }

// TokenID is the id of a token in the byte alphabet, or of a special token.
func (t *BPETokenizer) TokenID(token string) (int, bool) {
	if id, ok := t.ids[token]; ok {
		return id, true
	}
	if at, ok := t.specialAt[token]; ok {
		return len(t.Vocab) + at, true
	}
	return 0, false
}

// IDToken is the token an id stands for (a special token as itself).
func (t *BPETokenizer) IDToken(id int) (string, error) {
	if id >= 0 && id < len(t.Vocab) {
		return t.Vocab[id], nil
	}
	if at := id - len(t.Vocab); at >= 0 && at < len(t.Special) {
		return t.Special[at], nil
	}
	return "", fmt.Errorf("id %d is outside a vocabulary of %d", id, t.VocabSize())
}

// mergeWord is GPT-2's bpe over one pre-token's byte characters: the pair with
// the lowest rank is merged everywhere it occurs, left to right and without
// overlap, again and again.
func (t *BPETokenizer) mergeWord(parts []string) []string {
	for len(parts) > 1 {
		best, bestRank := -1, 0
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.ranks[bpePair{parts[i], parts[i+1]}]; ok && (best < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		first, second := parts[best], parts[best+1]
		merged := make([]string, 0, len(parts))
		for i := 0; i < len(parts); {
			if i+1 < len(parts) && parts[i] == first && parts[i+1] == second {
				merged = append(merged, first+second)
				i += 2
			} else {
				merged = append(merged, parts[i])
				i++
			}
		}
		parts = merged
	}
	return parts
}

func (t *BPETokenizer) word(pretoken string) []string {
	t.mu.Lock()
	found, ok := t.words[pretoken]
	t.mu.Unlock()
	if ok {
		return found
	}
	data := []byte(pretoken)
	parts := make([]string, len(data))
	for i, c := range data {
		parts[i] = string(byteChars[c])
	}
	found = t.mergeWord(parts)
	t.mu.Lock()
	if len(t.words) >= bpeCacheLimit {
		t.words = map[string][]string{}
	}
	t.words[pretoken] = found
	t.mu.Unlock()
	return found
}

func (t *BPETokenizer) tokensOf(text string) []string {
	var out []string
	for _, p := range Pretokenize(text) {
		out = append(out, t.word(p)...)
	}
	return out
}

// Tokens is the tokens of a text in the byte alphabet, read with a space before
// it: "the cat." -> [Ġthe Ġcat .].  A special token in the text is its characters.
func (t *BPETokenizer) Tokens(text string) []string { return t.tokensOf(" " + text) }

// Encode is the ids of a text; allowedSpecial turns <|endoftext|> in it into its id.
func (t *BPETokenizer) Encode(text string, allowedSpecial bool) []int {
	framed := " " + text
	var ids []int
	add := func(part string) {
		for _, tok := range t.tokensOf(part) {
			ids = append(ids, t.ids[tok])
		}
	}
	if !allowedSpecial || len(t.Special) == 0 {
		add(framed)
		return ids
	}
	specials := append([]string(nil), t.Special...)
	sort.SliceStable(specials, func(a, b int) bool { return len(specials[a]) > len(specials[b]) })
	at := 0
	for at <= len(framed) {
		next, which := -1, ""
		for _, s := range specials { // the leftmost special, the longest where two start together
			if k := strings.Index(framed[at:], s); k >= 0 && (next < 0 || at+k < next) {
				next, which = at+k, s
			}
		}
		if next < 0 {
			break
		}
		add(framed[at:next])
		ids = append(ids, len(t.Vocab)+t.specialAt[which])
		at = next + len(which)
	}
	add(framed[at:])
	return ids
}

// TokenBytes is the bytes a token stands for; a special token is its own text.
func (t *BPETokenizer) TokenBytes(token string) ([]byte, error) {
	if _, ok := t.specialAt[token]; ok {
		return []byte(token), nil
	}
	data, ok := TextBytes(token)
	if !ok {
		return nil, fmt.Errorf("%q is not a token of the byte alphabet", token)
	}
	return data, nil
}

// DecodeBytes is the bytes of a run of ids, the leading space the encoder put
// there taken away.
func (t *BPETokenizer) DecodeBytes(ids []int) ([]byte, error) {
	var out []byte
	for _, id := range ids {
		tok, err := t.IDToken(id)
		if err != nil {
			return nil, err
		}
		data, err := t.TokenBytes(tok)
		if err != nil {
			return nil, err
		}
		out = append(out, data...)
	}
	if len(out) > 0 && out[0] == ' ' {
		out = out[1:]
	}
	return out, nil
}

// Decode is the text of a run of ids: the inverse of Encode, exactly.  A run
// cut inside a character decodes what is broken as U+FFFD, one for each
// maximal broken piece - Python's rule.
func (t *BPETokenizer) Decode(ids []int) (string, error) {
	data, err := t.DecodeBytes(ids)
	if err != nil {
		return "", err
	}
	text, _ := decodeUTF8(data, true)
	return text, nil
}

// decodeUTF8 reads bytes as UTF-8, a broken sequence as one U+FFFD for each
// maximal broken piece (the Unicode practice Python and Rust follow).  Unless
// final, a valid but unfinished sequence at the end is handed back unread.
func decodeUTF8(data []byte, final bool) (string, []byte) {
	var b strings.Builder
	b.Grow(len(data))
	for i := 0; i < len(data); {
		c := data[i]
		if c < 0x80 {
			b.WriteByte(c)
			i++
			continue
		}
		need, lo, hi := 0, byte(0x80), byte(0xBF)
		switch {
		case c >= 0xC2 && c <= 0xDF:
			need = 1
		case c == 0xE0:
			need, lo = 2, 0xA0
		case c >= 0xE1 && c <= 0xEC, c == 0xEE, c == 0xEF:
			need = 2
		case c == 0xED:
			need, hi = 2, 0x9F
		case c == 0xF0:
			need, lo = 3, 0x90
		case c >= 0xF1 && c <= 0xF3:
			need = 3
		case c == 0xF4:
			need, hi = 3, 0x8F
		default: // a continuation byte where none belongs, or a byte UTF-8 never uses
			b.WriteRune(utf8.RuneError)
			i++
			continue
		}
		k := 1
		for k <= need {
			if i+k >= len(data) {
				if !final {
					return b.String(), data[i:]
				}
				break
			}
			d := data[i+k]
			if k == 1 && (d < lo || d > hi) || k > 1 && (d < 0x80 || d > 0xBF) {
				break
			}
			k++
		}
		if k <= need { // broken after k-1 good continuation bytes: one replacement for all of them
			b.WriteRune(utf8.RuneError)
			i += k
			continue
		}
		b.Write(data[i : i+need+1])
		i += need + 1
	}
	return b.String(), nil
}

// -- the text form ----------------------------------------------------------------------

func printableASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '!' || s[i] > '~' {
			return false
		}
	}
	return true
}

// Render is a token as the text form writes it: "Ġcat" -> cat, "ing" -> ⁀ing.
func (t *BPETokenizer) Render(token string) string {
	if rest, ok := strings.CutPrefix(token, "Ġ"); ok && rest != "" && printableASCII(rest) {
		return rest
	}
	return BPEGlue + token
}

// Read is the token a piece of the text form stands for.  A glued piece is the
// token after the mark, if the vocabulary has it; a bare piece is printable
// ASCII and stands for the token of a space and itself - but only when that is
// what the tokenizer makes of it as text, so that the two readings agree.
func (t *BPETokenizer) Read(piece string) (string, bool) {
	if piece == "" {
		return "", false
	}
	if rest, ok := strings.CutPrefix(piece, BPEGlue); ok {
		_, known := t.ids[rest]
		return rest, known
	}
	t.mu.Lock()
	ok, seen := t.bare[piece]
	t.mu.Unlock()
	if !seen {
		ok = false
		if printableASCII(piece) {
			tokens := t.tokensOf(" " + piece)
			ok = len(tokens) == 1 && tokens[0] == "Ġ"+piece
		}
		t.mu.Lock()
		if len(t.bare) >= bpeCacheLimit {
			t.bare = map[string]bool{}
		}
		t.bare[piece] = ok
		t.mu.Unlock()
	}
	if !ok {
		return "", false
	}
	return "Ġ" + piece, true
}

// Units is a text as the units of the text form: "The cat sat." -> [The cat
// sat ⁀.].  Text that already is the text form - single spaces between pieces
// that each Read - is read piece by piece; anything else is read as text.
func (t *BPETokenizer) Units(text string) []string {
	if text == "" {
		return nil
	}
	pieces := strings.Split(text, " ")
	units := make([]string, 0, len(pieces))
	for _, p := range pieces {
		tok, ok := t.Read(p)
		if !ok {
			units = units[:0]
			for _, tok := range t.Tokens(text) {
				units = append(units, t.Render(tok))
			}
			return units
		}
		units = append(units, t.Render(tok))
	}
	return units
}

// Text is the text form of a text: its units joined by single spaces.  Idempotent.
func (t *BPETokenizer) Text(text string) string { return strings.Join(t.Units(text), " ") }

func unitToken(unit string) string {
	if rest, ok := strings.CutPrefix(unit, BPEGlue); ok {
		return rest
	}
	return "Ġ" + unit
}

// TokenIDs is the ids of a text given as text or in the text form.
func (t *BPETokenizer) TokenIDs(text string) []int {
	units := t.Units(text)
	ids := make([]int, len(units))
	for i, u := range units {
		ids[i] = t.ids[unitToken(u)]
	}
	return ids
}

// BytesOf is the bytes of a text's units, the leading space kept: what a walk
// said, byte by byte.
func (t *BPETokenizer) BytesOf(text string) []byte {
	var out []byte
	for _, u := range t.Units(text) {
		data, _ := TextBytes(unitToken(u))
		out = append(out, data...)
	}
	return out
}

// Spell is what a text of the text form says: "The cat sat ⁀." -> "The cat sat.".
func (t *BPETokenizer) Spell(text string) string {
	s, err := t.Decode(t.TokenIDs(text))
	if err != nil {
		return text
	}
	return s
}
