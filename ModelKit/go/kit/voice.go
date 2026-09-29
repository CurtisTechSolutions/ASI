package kit

// Speech from a walk: the model is heard as it traverses its graph (the port of
// modelkit/voice.py).  A model whose symbols are sounds emits phones as it
// walks; a model of words or letters emits text.  Speaker takes either, one
// step at a time, turns it into the tokens of the phonetic tokenizer and feeds
// the formant synthesizer, which hands back 16-bit PCM as soon as a word can be
// committed.  The utterance is closed by the final sentinel: when the walk
// steps onto End, Speaker.End flushes what is pending with the closing
// intonation.  One walk, one utterance.

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// SpeakOptions configure SpeakWalks.
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
	enc         radixnet.Encoding
	synth       *phonetok.Synthesizer
	vocoder     *phonetok.Vocoder // acoustic units are spoken through their codebook's vocoder instead
	letters     string
	spokenWords int
	// Rate is the sample rate of the PCM: the synthesizer's, or the codebook's for acoustic units.
	Rate int
	// Tokens is every token that reached the voice, for the record.
	Tokens []string
}

// NewSpeaker makes a speaker for a model's encoding.
func NewSpeaker(enc radixnet.Encoding, rate int, pitch, tempo, gain float64) (*Speaker, error) {
	if enc.Unit == radixnet.Acoustic {
		tok, err := radixnet.AcousticTokenizer()
		if err != nil {
			return nil, err
		}
		// the vocoder's gain is a multiplier on the level the units were learned at, so the
		// voice's default of half scale is the codebook's own level
		voc, err := phonetok.NewVocoder(tok.Book, gain*2.0, pitch)
		if err != nil {
			return nil, err
		}
		return &Speaker{enc: enc, vocoder: voc, Rate: tok.Book.Analysis.Rate}, nil
	}
	if _, err := radixnet.PhoneticTokens(radixnet.Phones, ""); err != nil { // the words of a word or letter model are read through it
		return nil, err
	}
	voice := phonetok.DefaultVoice()
	voice.Pitch, voice.Tempo, voice.Gain = pitch, tempo, gain
	return &Speaker{enc: enc, synth: phonetok.NewSynthesizer(rate, voice), Rate: rate}, nil
}

// Feed takes one emitted piece (the units a step added) and returns the PCM that is ready.
// A token that is not a unit of the codebook (which a model over its units never emits) is skipped.
func (s *Speaker) Feed(piece string) []byte {
	if s.vocoder != nil {
		var out []byte
		for _, unit := range strings.Fields(piece) {
			s.Tokens = append(s.Tokens, unit)
			if chunk, err := s.vocoder.Feed(unit); err == nil {
				out = append(out, chunk...)
			}
		}
		return out
	}
	if s.enc.Unit.Phonetic() {
		return s.feedTokens(strings.Fields(radixnet.PhoneticText(s.enc.Unit, piece)))
	}
	if s.enc.Unit == radixnet.Words {
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
	for _, w := range words {
		tokens, err := radixnet.PhoneticTokens(radixnet.Phones, w)
		if err != nil {
			return nil
		}
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
	if s.vocoder != nil {
		s.Tokens = append(s.Tokens, "</s>")
		return s.vocoder.End()
	}
	out := s.flushLetters() // a letter model's last word, so the sentinel is recorded after it
	s.Tokens = append(s.Tokens, "</s>")
	out = append(out, s.synth.End()...)
	s.spokenWords = 0
	return out
}

// SpeakWalks walks the model Count times from Prefix and speaks each walk as it
// goes: emit gets every chunk of PCM the moment it is made, said (if not nil)
// is told what each walk said once it has ended - the text in the model's
// units and the words it spells.
func SpeakWalks(m *radixnet.Model, o SpeakOptions, emit func(chunk []byte), said func(i int, text, spelled string)) error {
	enc := m.Encoding()
	g := m.G
	var rng *radixnet.MT19937
	if o.Seed != nil {
		rng = radixnet.NewMT19937(*o.Seed)
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
		node, offset, lead := m.PrefixStart(o.Prefix)
		if lead != "" {
			say(lead)
		}
		if node >= radixnet.First { // a compressed node's label runs on past the located gram: the walk's first emission
			say(enc.Slice(g.Labels[node], offset+enc.N, -1))
		}
		// from START nothing precedes the first node stepped onto, so it is said whole; every
		// node after it - and every node after a located prefix - adds what lies past the
		// overlap (the rule DecodePath decodes a path by)
		wholeFirst := node < radixnet.First
		walk, err := g.SampleWalkListening(node, offset, o.MaxLength, o.Temperature, rng, nil, nil, radixnet.ByReward, radixnet.SamplingFilter{},
			func(c int) {
				if c >= radixnet.First {
					cut := enc.Overlap()
					if wholeFirst {
						cut = 0
					}
					wholeFirst = false
					say(enc.Slice(g.Labels[c], cut, -1))
				} else if c == radixnet.End {
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

// -- the output decoder --------------------------------------------------------------

// SayOptions shape the voice the output decoder speaks with: the one SpeakWalks
// walks with, and for acoustic units how many Griffin-Lim iterations polish
// each whole utterance (0: the streaming vocoder's output as it is).
type SayOptions struct {
	Rate   int
	Pitch  float64
	Tempo  float64
	Gain   float64
	Polish int
}

// DefaultSayOptions is the default voice.
func DefaultSayOptions() SayOptions {
	return SayOptions{Rate: phonetok.Rate, Pitch: 120, Tempo: 1, Gain: 0.5}
}

// Utterance is one text of a model spoken: what it said, what that spells, and
// how much audio it made.
type Utterance struct {
	Text    string   // the text as the model wrote it, in its units
	Spelled string   // the words a text of sounds spells; the text itself for every other unit
	Tokens  []string // every token that reached the voice, the closing sentinel last
	Samples int
	Seconds float64
}

// Dict is the utterance as the CLI prints it and the API returns it.
func (u Utterance) Dict() map[string]any {
	tokens := u.Tokens
	if tokens == nil {
		tokens = []string{}
	}
	return map[string]any{"text": u.Text, "spelled": u.Spelled, "tokens": tokens, "samples": u.Samples, "seconds": u.Seconds}
}

// Spoken is the speech of one or more texts: 16-bit mono PCM, and what each
// utterance was.
type Spoken struct {
	PCM        []byte
	Rate       int
	Encoding   string // the encoding the texts were read in
	Decoder    string // "voice": the formant synthesizer; "vocoder": the acoustic codebook's vocoder
	Utterances []Utterance
}

// Samples is how many PCM samples were made.
func (s *Spoken) Samples() int { return len(s.PCM) / 2 }

// Seconds is how long the speech lasts.
func (s *Spoken) Seconds() float64 {
	if s.Rate == 0 {
		return 0
	}
	return float64(s.Samples()) / float64(s.Rate)
}

// WAV is the speech as a WAV file.
func (s *Spoken) WAV() []byte { return phonetok.WavBytes(s.PCM, s.Rate) }

// Dict is the record without the audio: what the CLI prints and the API
// returns beside wav_base64.
func (s *Spoken) Dict() map[string]any {
	utterances := make([]map[string]any, 0, len(s.Utterances))
	for _, u := range s.Utterances {
		utterances = append(utterances, u.Dict())
	}
	return map[string]any{
		"rate": s.Rate, "samples": s.Samples(), "seconds": s.Seconds(), "encoding": s.Encoding,
		"decoder": s.Decoder, "count": len(s.Utterances), "utterances": utterances,
	}
}

// DecoderName is what speaks a model's texts: the codebook's vocoder for
// acoustic units, the formant voice for the rest.
func DecoderName(enc radixnet.Encoding) string {
	if enc.Unit == radixnet.Acoustic {
		return "vocoder"
	}
	return "voice"
}

// Spelled is the words a text spells: through the tokenizer for a model of
// sounds (so "the cat" given to a phone model spells "the cat" too), the text
// itself for every other unit.
func Spelled(enc radixnet.Encoding, text string) string {
	if !enc.Unit.Phonetic() {
		return text
	}
	return enc.Spell(radixnet.PhoneticText(enc.Unit, text))
}

// checkUnits refuses a text of acoustic units holding a token that is not a
// unit of the codebook, as the Python vocoder does (the Speaker skips such a
// token, which a model over its units never emits; a text given from outside
// can hold anything).
func checkUnits(enc radixnet.Encoding, text string) error {
	if enc.Unit != radixnet.Acoustic {
		return nil
	}
	tok, err := radixnet.AcousticTokenizer()
	if err != nil {
		return err
	}
	for _, u := range strings.Fields(text) {
		if !tok.IsUnit(u) {
			return fmt.Errorf("not a unit of this codebook: %q", u)
		}
	}
	return nil
}

// SpeakTexts is the output decoder in its streaming form: every text - a
// prediction, a sample, a turn - is fed whole to a Speaker for the encoding
// and closed by the END sentinel, exactly as a walk is, so a text of sounds is
// spoken directly, a text of acoustic units through the codebook's vocoder,
// and a text of words or letters is read through the tokenizer word by word.
// emit gets every chunk of PCM as it is made; said (if not nil) each
// utterance once it has been spoken.
func SpeakTexts(enc radixnet.Encoding, texts []string, o SayOptions, emit func(chunk []byte), said func(i int, u Utterance)) error {
	for i, text := range texts {
		if err := checkUnits(enc, text); err != nil {
			return err
		}
		speaker, err := NewSpeaker(enc, o.Rate, o.Pitch, o.Tempo, o.Gain)
		if err != nil {
			return err
		}
		made := 0
		for _, chunk := range [][]byte{speaker.Feed(text), speaker.End()} {
			if len(chunk) > 0 {
				made += len(chunk)
				emit(chunk)
			}
		}
		if said != nil {
			said(i, Utterance{
				Text: text, Spelled: Spelled(enc, text), Tokens: append([]string{}, speaker.Tokens...),
				Samples: made / 2, Seconds: float64(made) / 2 / float64(speaker.Rate),
			})
		}
	}
	return nil
}

// Say is the output decoder: texts in a model's units become speech, one
// utterance each, through the same voice SpeakWalks walks with and closed by
// the same rule, the END sentinel.  o.Polish applies to acoustic units only:
// that many Griffin-Lim iterations over each whole utterance once it is
// known, which the streaming vocoder cannot do.
func Say(enc radixnet.Encoding, texts []string, o SayOptions) (*Spoken, error) {
	rate, err := radixnet.OutputRate(enc, o.Rate)
	if err != nil {
		return nil, err
	}
	spoken := &Spoken{Rate: rate, Encoding: enc.String(), Decoder: DecoderName(enc), Utterances: []Utterance{}}
	if spoken.Decoder == "vocoder" && o.Polish > 0 {
		// the whole utterance is known, so it can be polished: the vocoder's gain convention
		// is the Speaker's (the voice's half scale is the codebook's own level)
		tok, err := radixnet.AcousticTokenizer()
		if err != nil {
			return nil, err
		}
		for _, text := range texts {
			if err := checkUnits(enc, text); err != nil {
				return nil, err
			}
			units := strings.Fields(text)
			pcm, err := phonetok.Synthesize(units, tok.Book, o.Polish, o.Gain*2.0, o.Pitch)
			if err != nil {
				return nil, err
			}
			spoken.PCM = append(spoken.PCM, pcm...)
			spoken.Utterances = append(spoken.Utterances, Utterance{
				Text: text, Spelled: text, Tokens: append(append([]string{}, units...), "</s>"),
				Samples: len(pcm) / 2, Seconds: float64(len(pcm)) / 2 / float64(rate),
			})
		}
		return spoken, nil
	}
	err = SpeakTexts(enc, texts, o,
		func(chunk []byte) { spoken.PCM = append(spoken.PCM, chunk...) },
		func(_ int, u Utterance) { spoken.Utterances = append(spoken.Utterances, u) })
	if err != nil {
		return nil, err
	}
	return spoken, nil
}
