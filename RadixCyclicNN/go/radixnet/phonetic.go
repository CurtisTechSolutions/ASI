package radixnet

// The phonetic units of the encoding read text through the phonetic tokenizer
// (the sibling PhoneticTokenizer module, the Go port of the phonetok package).
// One tokenizer per unit, made on first use and kept: what it remembers of the
// words it sounded out is what lets a prediction be spelled back.  The lexicon
// is the *portable* one - the bundled core plus the file PHONETOK_LEXICON names
// - the lexicon the Python and Rust ports read too, so that a model whose
// symbols are sounds means the same sounds in every port.

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
)

// -- the acoustic units ---------------------------------------------------------------

var (
	acousticOnce sync.Once
	acousticTok  *phonetok.AcousticTokenizer
	acousticErr  error
)

// acousticTokenizer is the acoustic units' tokenizer over its codebook - the
// bundled one, or the file PHONETOK_CODEBOOK names - made once; a model's units
// mean nothing without the codebook that made them, so the two travel together.
func acousticTokenizer() (*phonetok.AcousticTokenizer, error) {
	acousticOnce.Do(func() {
		var book *phonetok.Codebook
		if path := os.Getenv("PHONETOK_CODEBOOK"); path != "" {
			if book, acousticErr = phonetok.LoadCodebook(path); acousticErr != nil {
				acousticErr = fmt.Errorf("the acoustic unit's codebook (PHONETOK_CODEBOOK): %w", acousticErr)
				return
			}
		}
		acousticTok, acousticErr = phonetok.NewAcousticTokenizer(book, true)
		if acousticErr != nil {
			acousticErr = fmt.Errorf("the acoustic unit needs its codebook: %w", acousticErr)
		}
	})
	return acousticTok, acousticErr
}

// HearAudio is a WAV file's bytes as a text of acoustic units - "q2 q28 q55 ...",
// runs collapsed.  One recording is one utterance, and so one text.
func HearAudio(data []byte) (string, error) {
	tok, err := acousticTokenizer()
	if err != nil {
		return "", err
	}
	heard, err := tok.ListenWAV(data)
	if err != nil {
		return "", err
	}
	return heard.Text(), nil
}

// IsAudioFile reports whether a file is a WAV, by its name: something to hear
// rather than read.
func IsAudioFile(path string) bool { return strings.HasSuffix(strings.ToLower(path), ".wav") }

// OutputRate is the sample rate a model is heard at: rate for the synthesizer,
// the codebook's for acoustic units.
func OutputRate(enc Encoding, rate int) (int, error) {
	if enc.Unit != Acoustic {
		return rate, nil
	}
	tok, err := acousticTokenizer()
	if err != nil {
		return 0, err
	}
	return tok.Book.Analysis.Rate, nil
}

type phoneticBridge struct {
	mu  sync.Mutex // the tokenizer remembers what it reads; training reads from many goroutines
	tok *phonetok.Tokenizer
	err error
}

var (
	phoneticSet = map[UnitKind]*phoneticBridge{}
	phoneticMu  sync.Mutex
)

// phoneticTokenizer is the tokenizer a phonetic unit reads through, or the
// reason it could not be made.
func phoneticTokenizer(unit UnitKind) (*phoneticBridge, error) {
	phoneticMu.Lock()
	b, ok := phoneticSet[unit]
	if !ok {
		b = &phoneticBridge{}
		phoneticSet[unit] = b
		lex, err := phonetok.PortableLexicon()
		if err != nil {
			b.err = fmt.Errorf("the %s unit needs the phonetic lexicon: %w", unit, err)
		} else {
			level := phonetok.Phoneme
			if unit == Syllables {
				level = phonetok.SyllableLvl
			}
			b.tok, b.err = phonetok.NewTokenizer(level, lex)
		}
	}
	phoneticMu.Unlock()
	return b, b.err
}

// phoneticText is the text as the sounds it is made of, joined by single
// spaces: the text form of the tokenizer, which is idempotent.
func phoneticText(unit UnitKind, text string) string {
	b, err := phoneticTokenizer(unit)
	if err != nil {
		panic(err) // Validate refused the encoding before any text could reach here
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.tok.Text(text)
}

// Spell is the words a phonetic text spells ("DH AH0 # K AE1 T" -> "the cat"):
// a text given in words is read as the sounds it makes first, so it spells
// itself back, and so does a text that mixes the two.  A character or word
// encoding returns the text as it is.
func (e Encoding) Spell(text string) string {
	if !e.Unit.Phonetic() {
		return text
	}
	b, err := phoneticTokenizer(e.Unit)
	if err != nil {
		return text
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.tok.Decode(strings.Fields(b.tok.Text(text)))
}

// SpellTail is what tail - the last units of whole - spells, as the part of
// Spell(whole) it wrote.  A continuation spelled on its own loses the space
// before its first word, and cannot finish a word the text before it began, so
// the whole text is spelled and what the text before the tail spells is cut
// off its front: SpellTail("DH AH0 # K AE1 T # S AE1 T", "# S AE1 T") is
// " sat", and the two pieces join back into Spell(whole).  When the tail
// finished a word the head began, the cut falls back to where that word
// starts.  A tail that does not end whole is spelled on its own.  A character
// or word encoding returns the tail as it is.  Python's Encoding.spell_tail,
// character for character.
func (e Encoding) SpellTail(whole, tail string) string {
	if !e.Unit.Phonetic() {
		return tail
	}
	if !strings.HasSuffix(whole, tail) {
		return e.Spell(tail)
	}
	spelled := e.Spell(whole)
	head := e.Spell(whole[:len(whole)-len(tail)])
	if !strings.HasPrefix(spelled, head) { // the tail finished a word the head began: cut where that word starts
		common := commonRunePrefix(spelled, head)
		head = spelled[:strings.LastIndexByte(common, ' ')+1]
	}
	return spelled[len(head):]
}

// commonRunePrefix is the longest run of code points a and b start with.
func commonRunePrefix(a, b string) string {
	n := 0
	for n < len(a) && n < len(b) {
		ra, size := utf8.DecodeRuneInString(a[n:])
		rb, _ := utf8.DecodeRuneInString(b[n:])
		if ra != rb {
			break
		}
		n += size
	}
	return a[:n]
}

// -- what a model of sounds said, in words ---------------------------------------------

// SpelledPrediction is the words a prediction of a model of sounds spells:
// {"spelled", "spelled_continuation"} - the whole text read back as English,
// and the part of it the continuation wrote (SpellTail), so the prefix's words
// are "spelled" without it.  Every other encoding is its own spelling and gets
// an empty map: the fields are there only when they say something.  Python's
// spelled_prediction.
func SpelledPrediction(enc Encoding, fullText, continuation string) map[string]any {
	if !enc.Unit.Phonetic() {
		return map[string]any{}
	}
	return map[string]any{"spelled": enc.Spell(fullText), "spelled_continuation": enc.SpellTail(fullText, continuation)}
}

// ReaderText is what a reader - a teacher, a reviewer, a partner - is shown
// of a text a model wrote: a model of sounds is shown the English its sounds
// spell, so an LLM marks, corrects and answers words rather than
// DH.AH0 # K.AE1.T; every other text is shown as it is.  What the reader
// writes back stays in words: a model of sounds reads it as the sounds it
// makes.  Python's reader_text.
func ReaderText(enc Encoding, text string) string {
	if !enc.Unit.Phonetic() {
		return text
	}
	return enc.Spell(text)
}

// SpelledCompletion is a completion of prefix by a model of sounds, as a
// reader is shown it: the prefix and the continuation in words, the sentence
// being the two joined.  When the sounds kept the prefix's words, the prefix
// is the one given - capitals, punctuation and trailing space as they were -
// and the continuation is the words after it (a pause that attaches to the
// prefix's last word stays on it); when the continuation finished a word the
// prefix began, both are cut from the words of the whole (SpellTail).  Any
// other encoding is its own spelling.  Python's spelled_completion.
func SpelledCompletion(enc Encoding, prefix, fullText, continuation string) (string, string) {
	if !enc.Unit.Phonetic() {
		return prefix, continuation
	}
	tail := enc.SpellTail(fullText, continuation)
	whole := enc.Spell(fullText)
	head := ""
	if strings.HasSuffix(whole, tail) {
		head = whole[:len(whole)-len(tail)]
	}
	if !sameWords(head, prefix) {
		return head, tail
	}
	if first, _ := utf8.DecodeRuneInString(tail); tail == "" || unicode.IsSpace(first) {
		return prefix, strings.TrimLeftFunc(tail, unicode.IsSpace) // a word of its own after the prefix
	}
	return strings.TrimRightFunc(prefix, unicode.IsSpace), tail
}

// sameWords reports whether two texts hold the same words, capitals aside.
func sameWords(a, b string) bool {
	x, y := strings.Fields(a), strings.Fields(b)
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if strings.ToLower(x[i]) != strings.ToLower(y[i]) {
			return false
		}
	}
	return true
}

// SpelledThought is a thought's record (Thought.ToDict) with "spelled" beside
// its text, and its questions' too; any other encoding, the record as it is.
func SpelledThought(enc Encoding, t *Thought) map[string]any {
	doc := t.ToDict()
	if !enc.Unit.Phonetic() {
		return doc
	}
	doc["spelled"] = enc.Spell(t.Text)
	questions := make([]map[string]any, 0, len(t.Questions))
	for _, q := range t.Questions {
		questions = append(questions, SpelledThought(enc, q))
	}
	doc["questions"] = questions
	return doc
}

// SpelledTurnRecord is a turn of a model of sounds as it is written out: the
// turn, the words it spells ("spelled"), the part of them the search added
// after the context it picked up ("spelled_reply"), and its rethink with the
// thought spelled.
type SpelledTurnRecord struct {
	*Turn
	Spelled      string          `json:"spelled"`
	SpelledReply string          `json:"spelled_reply"`
	Rethink      *spelledRethink `json:"rethink"` // shadows the turn's own
}

type spelledRethink struct {
	*Rethink
	Thought map[string]any `json:"thought"` // shadows the rethink's own
}

// SpelledTurn is a turn as it is written out: with the words it spells beside
// its sounds for a model of sounds (SpelledTurnRecord), the turn itself for any
// other encoding.  Python's spelled_turn.
func SpelledTurn(enc Encoding, t *Turn) any {
	if !enc.Unit.Phonetic() {
		return t
	}
	out := &SpelledTurnRecord{Turn: t, Spelled: enc.Spell(t.Text), SpelledReply: enc.SpellTail(t.Text, t.Reply)}
	if t.Rethink != nil {
		out.Rethink = &spelledRethink{Rethink: t.Rethink}
		if t.Rethink.Thought != nil {
			out.Rethink.Thought = SpelledThought(enc, t.Rethink.Thought)
		}
	}
	return out
}

// SpelledTurns is every turn as SpelledTurn writes it out.
func SpelledTurns(enc Encoding, turns []*Turn) []any {
	out := make([]any, 0, len(turns))
	for _, t := range turns {
		out = append(out, SpelledTurn(enc, t))
	}
	return out
}
