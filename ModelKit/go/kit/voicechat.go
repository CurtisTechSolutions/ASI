package kit

// Talking with the model by voice: every utterance is heard, learned, answered
// and spoken (D-089; Python's modelkit/voicechat.py).
//
// The Voice tab listens all the time.  Each thing the person says reaches here
// as one turn: the recording, and what the browser heard in it.  A turn is
//
//  1. heard - TeachSpeech makes the utterance into the texts the model learns,
//     the transcript and the waveform behind one unique token (D-035); a model
//     of acoustic units hears the recording as its units instead (D-082), the
//     one text made of sound it can learn;
//  2. trained - the model is trained on those texts at once, so the reply
//     already knows them;
//  3. answered - the model replies as it does in every conversation here
//     (Reply: the end of the line picked up and continued), or Ollama
//     answers for it (answer = "ollama"), or Ollama steps in only when the
//     model has nothing to say to the person ("auto": no reply, or a fresh
//     text that picked up none of the line);
//  4. spoken - the reply goes through the output decoder (D-088) and its audio
//     comes back in chunks the browser plays as they arrive;
//  5. taught - a reply Ollama wrote is taught to the model, so that next time
//     it may answer by itself.
//
// VoiceTurn does all of it and reports every step to write as it happens (the
// JSON Lines of POST /api/voice/turn/stream).  The model's own parts -
// training and replying - come in as VoiceParts, so the service takes its
// locks around exactly those and Ollama is asked outside them.

import (
	"encoding/base64"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// VoiceSpeakers are who speaks: the person, and the model (or Ollama on its behalf).
var VoiceSpeakers = [2]string{"You", "Model"}

// VoiceAnswers are who answers: the model with Ollama stepping in when it has
// nothing to say, the model alone, Ollama alone, nobody.
var VoiceAnswers = []string{"auto", "model", "ollama", "none"}

// VoiceChunkSamples are the samples per audio event: half a second at 16 kHz,
// so playback starts long before the reply has ended.
const VoiceChunkSamples = 8000

// VoiceOptions are one turn's settings: how it is heard, learned, answered and
// spoken (Python's VoiceOptions).
type VoiceOptions struct {
	// Teach is how the utterance is heard: the transcript (the Go build
	// transcribes nothing itself, so it must come with the audio), the
	// waveform text's sample rate and codec, whether the sound is learned,
	// the token's uniqueness.
	Teach TeachSpeechOptions
	// Language is kept for the record: nothing here transcribes.
	Language  string
	Train     bool
	Epochs    int
	LR        float64
	BatchSize int
	Answer    string
	// LearnReply teaches the model a reply Ollama wrote for it.
	LearnReply  bool
	Persona     string
	Topic       string
	Mode        string
	MaxLength   int
	Context     int
	K           int
	Beam        int // 0: the default width
	StepPenalty float64
	Temperature float64
	Seed        *int64
	// AvoidRepeats and AvoidWordRepeats are the reply's (ReplyOptions).
	AvoidRepeats     bool
	AvoidWordRepeats bool
	Explore          int
	// Learn teaches the graph what a rethink finds out, as in every conversation here.
	Learn             bool
	OllamaTemperature float64
	Speak             bool
	VoiceRate         int
	Pitch             float64
	Tempo             float64
	Gain              float64
	Polish            int
	Chunk             int
}

// DefaultVoiceOptions mirror the Python defaults.
func DefaultVoiceOptions() VoiceOptions {
	return VoiceOptions{
		Teach: DefaultTeachSpeechOptions(), Train: true, Epochs: 2, LR: 0.5, BatchSize: 8, Answer: "auto",
		LearnReply: true, Mode: "beam", MaxLength: 60, Context: 12, K: 5, Temperature: 1, AvoidRepeats: true,
		AvoidWordRepeats: true, Explore: Explore, Learn: true, OllamaTemperature: 0.8, Speak: true,
		VoiceRate: phonetok.Rate, Pitch: 120, Tempo: 1, Gain: 0.5, Chunk: VoiceChunkSamples,
	}
}

// Validate is Python's VoiceOptions.validate, word for word.
func (o VoiceOptions) Validate() error {
	if !slices.Contains(VoiceAnswers, o.Answer) {
		return fmt.Errorf("'answer' must be one of %s, got %s", strings.Join(VoiceAnswers, ", "), radixnet.PythonRepr(o.Answer))
	}
	if o.Mode != "beam" && o.Mode != "sample" {
		return fmt.Errorf("'mode' must be beam or sample, got %s", radixnet.PythonRepr(o.Mode))
	}
	if o.Epochs < 0 || o.BatchSize < 1 || o.LR < 0 {
		return fmt.Errorf("'epochs' and 'lr' must be at least 0 and 'batch_size' at least 1")
	}
	if o.Pitch <= 0 || o.Tempo <= 0 || o.Gain < 0 || o.VoiceRate < 1 {
		return fmt.Errorf("'pitch' and 'tempo' must be above 0, 'gain' at least 0 and 'voice_rate' at least 1")
	}
	if o.Chunk < 1 || o.Polish < 0 || o.MaxLength < 0 || o.Context < 0 || o.K < 1 {
		return fmt.Errorf("'chunk' and 'k' must be at least 1; 'polish', 'max_length' and 'context' at least 0")
	}
	return nil
}

// sayOptions is the voice the reply is spoken with.
func (o VoiceOptions) sayOptions() SayOptions {
	return SayOptions{Rate: o.VoiceRate, Pitch: o.Pitch, Tempo: o.Tempo, Gain: o.Gain, Polish: o.Polish}
}

// VoiceOutcome is what one turn came to: the document the API answers with,
// and the reply's audio.
type VoiceOutcome struct {
	Document map[string]any
	PCM      []byte // the reply as 16-bit PCM (empty when nothing was spoken)
	Rate     int
}

// WAV is the reply as a WAV.
func (v *VoiceOutcome) WAV() []byte { return phonetok.WavBytes(v.PCM, v.Rate) }

// VoiceParts are the model's own parts of a turn: Learn trains it on texts and
// says what it did ({"texts", "epochs", ...}); ModelReply is its own answer to
// a line, given what the conversation has heard (nil when it has nothing to
// say).  Either may be nil: the turn then skips that step.
type VoiceParts struct {
	Learn      func(texts []string) (map[string]any, error)
	ModelReply func(line string, heard *Heard) (*Turn, error)
}

func squashWords(text string) string { return strings.Join(strings.Fields(text), " ") }

// TidyHistory is a conversation so far as (speaker, text) pairs: blank texts
// dropped, the rest squashed, a nameless speaker the person.
func TidyHistory(pairs []Line) []Line {
	out := make([]Line, 0, len(pairs))
	for _, pair := range pairs {
		text := squashWords(pair.Text)
		if text == "" {
			continue
		}
		speaker := strings.TrimSpace(pair.Speaker)
		if speaker == "" {
			speaker = VoiceSpeakers[0]
		}
		out = append(out, Line{Speaker: speaker, Text: text})
	}
	return out
}

// HistoryPairs is a conversation so far as (speaker, text) pairs, from
// [speaker, text] pairs or {"speaker", "text"} records (Python's history_pairs).
func HistoryPairs(history []any) ([]Line, error) {
	pairs := make([]Line, 0, len(history))
	for i, item := range history {
		var speaker, text any = "", ""
		switch v := item.(type) {
		case map[string]any:
			if s, ok := v["speaker"]; ok {
				speaker = s
			}
			if t, ok := v["text"]; ok {
				text = t
			}
		case []any:
			if len(v) != 2 {
				return nil, fmt.Errorf("history[%d] must be [speaker, text] or {speaker, text}", i)
			}
			speaker, text = v[0], v[1]
		default:
			return nil, fmt.Errorf("history[%d] must be [speaker, text] or {speaker, text}", i)
		}
		s, okSpeaker := speaker.(string)
		t, okText := text.(string)
		if !okSpeaker || !okText {
			return nil, fmt.Errorf("history[%d]: the speaker and the text must be strings", i)
		}
		pairs = append(pairs, Line{Speaker: s, Text: t})
	}
	return TidyHistory(pairs), nil
}

// Hearing is what an utterance was heard as: the transcript, the texts to
// learn, the token and the audio (Python's hear).
type Hearing struct {
	Transcript string
	Token      *string // the token the texts share; a model of acoustic units has none
	Texts      []string
	Audio      *EncodedAudio // the waveform text's record, nil without one
	Units      *string       // a model of acoustic units' units ("" when nothing was heard); nil for every other model
	ASR        map[string]any
}

// fields are the record's, as the heard event and the document list them:
// what is missing is a plain nil, so a caller in this process can test for it.
func (h *Hearing) fields() map[string]any {
	var token, units, audio any
	if h.Token != nil {
		token = *h.Token
	}
	if h.Units != nil {
		units = *h.Units
	}
	if h.Audio != nil {
		audio = h.Audio
	}
	return map[string]any{
		"transcript": h.Transcript, "token": token, "texts": h.Texts, "audio": audio, "units": units, "asr": h.ASR,
	}
}

// orNil is a record that may be missing, as a plain nil when it is.
func orNil(doc map[string]any) any {
	if doc == nil {
		return nil
	}
	return doc
}

// voiceASR is what transcribed an utterance: the caller, or nobody - the Go
// build transcribes nothing itself.
func voiceASR(transcript, language string) map[string]any {
	var backend, lang, problem any
	if language != "" {
		lang = language
	}
	if strings.TrimSpace(transcript) != "" {
		backend = "given"
	} else {
		problem = "no transcript was given (transcription is Python-only: send the words with the audio)"
	}
	return map[string]any{"backend": backend, "model": nil, "language": lang, "seconds": 0.0, "error": problem}
}

// HearUtterance is what an utterance was heard as.
//
// A model of acoustic units learns the recording as its units (HearAudio) -
// the transcript is kept for the conversation, but nothing made of words is in
// its alphabet.
func HearUtterance(data []byte, enc radixnet.Encoding, o VoiceOptions) (*Hearing, error) {
	if enc.Unit == radixnet.Acoustic {
		words := squashWords(o.Teach.Transcript)
		units := ""
		if len(data) > 0 {
			heard, err := radixnet.HearAudio(data)
			if err != nil {
				return nil, err
			}
			units = heard
		}
		var backend, lang any
		if words != "" {
			backend = "given"
		}
		if o.Language != "" {
			lang = o.Language
		}
		texts := []string{}
		if units != "" {
			texts = []string{units}
		}
		return &Hearing{
			Transcript: words, Texts: texts, Units: &units,
			ASR: map[string]any{"backend": backend, "model": nil, "language": lang, "seconds": 0.0, "error": nil},
		}, nil
	}
	taught, err := TeachSpeech(data, o.Teach)
	if err != nil {
		return nil, err
	}
	token := taught.Token
	texts := taught.Texts
	if texts == nil {
		texts = []string{}
	}
	return &Hearing{
		Transcript: taught.Transcript, Token: &token, Texts: texts, Audio: taught.Audio,
		ASR: voiceASR(taught.Transcript, o.Language),
	}, nil
}

// SpeechEncoding is the encoding a reply is spoken in: the model's own, unless
// it is words a model of acoustic units cannot say.
func SpeechEncoding(enc radixnet.Encoding, by string) radixnet.Encoding {
	if enc.Unit == radixnet.Acoustic && by != "model" {
		return radixnet.DefaultEncoding()
	}
	return enc
}

// AudioEvents are the reply's audio as audio events of chunk samples: 16-bit
// PCM, base64, played as they arrive.
func AudioEvents(pcm []byte, rate, chunk int) []map[string]any {
	step := max(chunk, 1) * 2
	events := make([]map[string]any, 0, len(pcm)/step+1)
	for i := 0; i < len(pcm); i += step {
		piece := pcm[i:min(i+step, len(pcm))]
		events = append(events, map[string]any{
			"event": "audio", "rate": rate, "samples": len(piece) / 2,
			"pcm_base64": base64.StdEncoding.EncodeToString(piece),
		})
	}
	return events
}

// withEvent is {"event": name, ...doc}.
func withEvent(name string, doc map[string]any) map[string]any {
	event := maps.Clone(doc)
	if event == nil {
		event = map[string]any{}
	}
	event["event"] = name
	return event
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// VoiceTurn is one turn of talking with the model: heard, trained, answered,
// spoken, taught - each an event to write (which may be nil).
//
// data is the recording (a WAV; nil for a typed line, with the transcript in
// the options the words), history the conversation so far, parts the model's
// parts, ollama the client when Ollama may answer (a nil interface when it may
// not).  Fails for a turn that cannot be heard (unreadable audio, nothing
// said, a token that is no unit) - a bad request, on the server.
func VoiceTurn(data []byte, enc radixnet.Encoding, o VoiceOptions, history []Line, parts VoiceParts, ollama LLMClient, write func(map[string]any)) (*VoiceOutcome, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}
	emit := func(event map[string]any) {
		if write != nil {
			write(event)
		}
	}
	started := time.Now()
	// 1. heard
	heard, err := HearUtterance(data, enc, o)
	if err != nil {
		return nil, err
	}
	line := heard.Transcript
	if line == "" && heard.Units != nil {
		line = *heard.Units
	}
	emit(withEvent("heard", heard.fields()))
	// 2. trained
	var trained map[string]any
	if o.Train && len(heard.Texts) > 0 && parts.Learn != nil {
		began := time.Now()
		doc, err := parts.Learn(heard.Texts)
		if err != nil {
			return nil, err
		}
		trained = maps.Clone(doc)
		if trained == nil {
			trained = map[string]any{}
		}
		if _, ok := trained["texts"]; !ok {
			trained["texts"] = len(heard.Texts)
		}
		trained["seconds"] = round3(time.Since(began).Seconds())
		emit(withEvent("trained", trained))
	}
	// 3. answered
	said := make([]string, 0, len(history)+1)
	for _, pair := range history {
		said = append(said, pair.Text)
	}
	if line != "" {
		said = append(said, line)
	}
	heardSoFar := NewHeard(said)
	by, replyText := "none", ""
	var record any
	var ollamaError any
	var modelTurn *Turn
	if line != "" && (o.Answer == "model" || o.Answer == "auto") && parts.ModelReply != nil {
		turn, err := parts.ModelReply(line, heardSoFar)
		if err != nil {
			return nil, err
		}
		modelTurn = turn
	}
	if modelTurn != nil {
		by, replyText, record = "model", modelTurn.Text, modelTurn
	}
	// Ollama answers for the model, or steps in when it could not answer the person: nothing, or a
	// fresh text that picked up none of the line
	wantsOllama := o.Answer == "ollama" || (o.Answer == "auto" && (modelTurn == nil || modelTurn.Fresh))
	if wantsOllama && heard.Transcript != "" {
		if ollama == nil {
			ollamaError = "no Ollama to ask"
		} else {
			// Ollama reads the model's lines as words: a model of sounds is heard in the words it spells
			transcript := make([]Line, 0, len(history)+1)
			for _, line := range history {
				if line.Speaker == VoiceSpeakers[1] {
					line.Text = Spelled(enc, line.Text)
				}
				transcript = append(transcript, line)
			}
			transcript = append(transcript, Line{Speaker: VoiceSpeakers[0], Text: heard.Transcript})
			text, err := ReplyLine(ollama, transcript, ReplyLineOptions{
				Persona: o.Persona, Topic: o.Topic, Speakers: VoiceSpeakers, Temperature: o.OllamaTemperature,
			})
			if err != nil {
				ollamaError = err.Error()
			} else if text != "" {
				by, replyText = "ollama", text
			}
		}
	}
	voice := SpeechEncoding(enc, by)
	spelledText := ""
	if replyText != "" {
		spelledText = Spelled(voice, replyText)
	}
	reply := map[string]any{
		"by": by, "text": replyText, "spelled": spelledText, "turn": record, "ollama_error": ollamaError,
	}
	emit(withEvent("reply", reply))
	// 4. spoken
	var spokenDoc map[string]any
	var pcm []byte
	rate := o.VoiceRate
	if o.Speak && replyText != "" {
		spoken, err := Say(voice, []string{replyText}, o.sayOptions())
		if err != nil {
			return nil, err
		}
		pcm, rate = spoken.PCM, spoken.Rate
		for _, event := range AudioEvents(pcm, rate, o.Chunk) {
			emit(event)
		}
		spokenDoc = spoken.Dict()
		emit(withEvent("spoken", spokenDoc))
	}
	// 5. taught
	var taught map[string]any
	if by == "ollama" && o.LearnReply && replyText != "" && parts.Learn != nil && enc.Unit != radixnet.Acoustic {
		doc, err := parts.Learn([]string{replyText})
		if err != nil {
			return nil, err
		}
		taught = maps.Clone(doc)
		if taught == nil {
			taught = map[string]any{}
		}
		if _, ok := taught["texts"]; !ok {
			taught["texts"] = 1
		}
		emit(withEvent("taught", taught))
	}
	spokenPairs := make([][]string, 0, len(history)+2)
	for _, pair := range history {
		spokenPairs = append(spokenPairs, []string{pair.Speaker, pair.Text})
	}
	if line != "" {
		spokenPairs = append(spokenPairs, []string{VoiceSpeakers[0], line})
	}
	if replyText != "" {
		spokenPairs = append(spokenPairs, []string{VoiceSpeakers[1], replyText})
	}
	document := heard.fields()
	maps.Copy(document, map[string]any{
		"line": line, "trained": orNil(trained), "reply": reply, "by": by, "spoken": orNil(spokenDoc),
		"taught": orNil(taught), "history": spokenPairs, "encoding": enc.String(), "answer": o.Answer,
		"seconds": round3(time.Since(started).Seconds()),
	})
	return &VoiceOutcome{Document: document, PCM: pcm, Rate: rate}, nil
}

// DescribeVoice is what the Voice tab has to work with: the speakers, the
// answer modes, the voice and the transcription (Python's describe).
func DescribeVoice(enc radixnet.Encoding, ollama *OllamaClient) map[string]any {
	speech := DescribeSpeech()
	var client any
	if ollama != nil {
		client = map[string]any{"url": ollama.URL, "model": ollama.Model}
	}
	return map[string]any{
		"speakers": VoiceSpeakers[:], "answers": VoiceAnswers, "encoding": enc.String(),
		"decoder": DecoderName(enc), "acoustic": enc.Unit == radixnet.Acoustic, "default_rate": DefaultSpeechRate,
		"chunk": VoiceChunkSamples,
		"transcription": map[string]any{
			"backends": speech["backends"], "auto": speech["auto"], "faster_whisper": speech["faster_whisper"],
			"whisper": speech["whisper"],
		},
		"ollama": client,
	}
}
