package radixnet

// Speech from a walk: the model is heard as it traverses its graph (the port of
// radixnet/voice.py).  A model whose symbols are sounds emits phones as it
// walks; a model of words or letters emits text.  Speaker takes either, one
// step at a time, turns it into the tokens of the phonetic tokenizer and feeds
// the formant synthesizer, which hands back 16-bit PCM as soon as a word can be
// committed.  The utterance is closed by the final sentinel: when the walk
// steps onto End, Speaker.End flushes what is pending with the closing
// intonation.  One walk, one utterance.

import (
	"strings"
	"unicode"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
)

// SpeakOptions configure Model.SpeakWalks.
type SpeakOptions struct {
	Prefix      string
	Count       int
	MaxLength   int // units per walk at most; < 0 is no cap
	Temperature float64
	Seed        *int64 // a private RNG; nil uses the model's own
	Rate        int
	Pitch       float64
	Tempo       float64
	Gain        float64
}

// Speaker turns what a model emits into speech, step by step.
type Speaker struct {
	enc         Encoding
	synth       *phonetok.Synthesizer
	letters     string
	spokenWords int
	// Tokens is every token that reached the synthesizer, for the record.
	Tokens []string
}

// NewSpeaker makes a speaker for a model's encoding.
func NewSpeaker(enc Encoding, rate int, pitch, tempo, gain float64) (*Speaker, error) {
	if _, err := phoneticTokenizer(Phones); err != nil { // the words of a word or letter model are read through it
		return nil, err
	}
	voice := phonetok.DefaultVoice()
	voice.Pitch, voice.Tempo, voice.Gain = pitch, tempo, gain
	return &Speaker{enc: enc, synth: phonetok.NewSynthesizer(rate, voice)}, nil
}

// Feed takes one emitted piece (the units a step added) and returns the PCM that is ready.
func (s *Speaker) Feed(piece string) []byte {
	if s.enc.Unit.Phonetic() {
		return s.feedTokens(strings.Fields(phoneticText(s.enc.Unit, piece)))
	}
	if s.enc.Unit == Words {
		return s.feedWords(strings.Fields(piece))
	}
	// letters: a word ends at whitespace; punctuation ends it too and becomes a pause
	var out []byte
	for _, r := range piece {
		if unicode.IsSpace(r) {
			out = append(out, s.flushLetters()...)
			continue
		}
		s.letters += string(r)
		if strings.ContainsRune(".,;:!?", r) {
			out = append(out, s.flushLetters()...)
		}
	}
	return out
}

func (s *Speaker) flushLetters() []byte {
	if s.letters == "" {
		return nil
	}
	word := s.letters
	s.letters = ""
	return s.feedWords([]string{word})
}

func (s *Speaker) feedWords(words []string) []byte {
	var out []byte
	b, err := phoneticTokenizer(Phones)
	if err != nil {
		return nil
	}
	for _, w := range words {
		b.mu.Lock()
		tokens := b.tok.Tokens(w)
		b.mu.Unlock()
		if len(tokens) == 0 {
			continue
		}
		if s.spokenWords > 0 && tokens[0] != "," && tokens[0] != "." && tokens[0] != "?" {
			out = append(out, s.feedTokens([]string{"#"})...)
		}
		s.spokenWords++
		out = append(out, s.feedTokens(tokens)...)
	}
	return out
}

func (s *Speaker) feedTokens(tokens []string) []byte {
	var out []byte
	for _, t := range tokens {
		s.Tokens = append(s.Tokens, t)
		out = append(out, s.synth.Feed(t)...)
	}
	return out
}

// End is the final sentinel: the utterance is closed, and what was pending is spoken.
func (s *Speaker) End() []byte {
	out := s.flushLetters()
	s.Tokens = append(s.Tokens, "</s>")
	out = append(out, s.synth.End()...)
	s.spokenWords = 0
	return out
}

// SpeakWalks walks the model Count times from Prefix and speaks each walk as it
// goes: emit gets every chunk of PCM the moment it is made, said (if not nil)
// is told what each walk said once it has ended - the text in the model's
// units and the words it spells.
func (m *Model) SpeakWalks(o SpeakOptions, emit func(chunk []byte), said func(i int, text, spelled string)) error {
	enc := m.Encoding()
	g := m.G
	var rng *MT19937
	if o.Seed != nil {
		rng = NewMT19937(*o.Seed)
	}
	for i := 0; i < o.Count; i++ {
		speaker, err := NewSpeaker(enc, o.Rate, o.Pitch, o.Tempo, o.Gain)
		if err != nil {
			return err
		}
		var parts []string
		say := func(piece string) {
			if piece == "" {
				return
			}
			parts = append(parts, piece)
			if chunk := speaker.Feed(piece); len(chunk) > 0 {
				emit(chunk)
			}
		}
		if o.Prefix != "" {
			say(o.Prefix)
		}
		node, offset, lead := m.prefixStart(o.Prefix)
		if lead != "" {
			say(lead)
		}
		if node >= First { // a compressed node's label runs on past the located gram: the walk's first emission
			say(enc.Slice(g.Labels[node], offset+enc.N, -1))
		}
		// from START nothing precedes the first node stepped onto, so it is said whole; every
		// node after it - and every node after a located prefix - adds what lies past the
		// overlap (the rule DecodePath decodes a path by)
		wholeFirst := node < First
		walk, err := g.SampleWalkListening(node, offset, o.MaxLength, o.Temperature, rng, nil, nil, ByReward, SamplingFilter{},
			func(c int) {
				if c >= First {
					cut := enc.Overlap()
					if wholeFirst {
						cut = 0
					}
					wholeFirst = false
					say(enc.Slice(g.Labels[c], cut, -1))
				} else if c == End {
					if chunk := speaker.End(); len(chunk) > 0 {
						emit(chunk)
					}
				}
			})
		if err != nil {
			return err
		}
		if !walk.ReachedEnd { // cut off by the length: the sentinel is ours to send
			if chunk := speaker.End(); len(chunk) > 0 {
				emit(chunk)
			}
		}
		if said != nil {
			text := enc.Join(parts...)
			said(i, text, enc.Spell(text))
		}
	}
	return nil
}
