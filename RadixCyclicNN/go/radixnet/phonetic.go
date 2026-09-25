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
