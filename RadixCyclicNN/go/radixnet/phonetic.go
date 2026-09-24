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
	"strings"
	"sync"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
)

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
// a character or word encoding returns the text as it is.
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
	return b.tok.Decode(strings.Fields(text))
}
