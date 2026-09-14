package radixnet

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Teaching the model by talking to it: the waveform as text, and the token both
// halves of one utterance hang off.
//
// One utterance becomes two texts that start with the same unique token:
//
//	<speech:9f2a1c7d> the cat sat on the mat
//	<speech:9f2a1c7d> aud:mu:8000x1:gOXv7NgqFBAYTePu7dwvFRAXPuHu...
//
// so in the cyclic graph the words and the sound leave the same node.  The
// waveform text is the recording itself - mixed to mono, resampled and
// quantised to one mu-law byte per sample - base64-encoded, so the trigram
// network trains on it like any other text.
//
// What is here is everything that is code: the WAV reader, both codecs, the
// resampling, the token and the text format.  Transcription is not: the local
// Whisper backends are Python packages, and `given` (the caller's own
// transcript, which is what the browser's dictation and --text supply) needs no
// backend at all, so it is the one this side offers.

const speechHeader = "aud"

// SpeechToken is the marker every spoken text starts with.
const SpeechToken = "<speech>"

// Waveform codecs.
var SpeechCodecs = []string{"auto", "mu", "pcm8"}

// DefaultSpeechCodec and DefaultSpeechRate are what an unspecified encode uses.
const (
	DefaultSpeechCodec = "mu"
	DefaultSpeechRate  = 8000
	MinSpeechRate      = 1000
	MaxSpeechRate      = 48000
	// MuLaw is the compression parameter of the mu-law codec (256 levels).
	MuLaw          = 255.0
	speechTokenHex = 8
)

var speechTextRe = regexp.MustCompile(`aud:([a-z0-9_]+):(\d+)x(\d+):`)

// SpeechError is unreadable audio, an unknown codec or a text that is not an
// encoded waveform.
type SpeechError struct{ Message string }

func (e *SpeechError) Error() string { return e.Message }

func speechErrorf(format string, args ...any) error {
	return &SpeechError{Message: fmt.Sprintf(format, args...)}
}

// PackSpeechText is aud:<codec>:<rate>x<channels>:<base64 of one byte per sample>.
func PackSpeechText(codec string, rate, channels int, payload []byte) string {
	return fmt.Sprintf("%s:%s:%dx%d:%s", speechHeader, codec, rate, channels,
		base64.StdEncoding.EncodeToString(payload))
}

// ParseSpeechText reads (codec, rate, channels, payload, repaired) out of an
// encoded-waveform text.
//
// The header is looked for *anywhere* in the text, because a spoken text
// carries its token first and a paired one its transcript last; the payload
// therefore runs from the header to the next whitespace.
func ParseSpeechText(text string) (string, int, int, []byte, bool, error) {
	match := speechTextRe.FindStringSubmatchIndex(text)
	if match == nil {
		return "", 0, 0, nil, false, speechErrorf(
			"not an encoded waveform: expected 'aud:<codec>:<rate>x<channels>:<base64>'")
	}
	codec := text[match[2]:match[3]]
	rate, _ := strconv.Atoi(text[match[4]:match[5]])
	channels, _ := strconv.Atoi(text[match[6]:match[7]])
	body := ""
	if rest := text[match[1]:]; strings.TrimSpace(rest) != "" {
		body = strings.Fields(rest)[0]
	}
	payload, repaired, err := RepairBase64(body)
	if err != nil {
		return "", 0, 0, nil, false, speechErrorf("%v", err)
	}
	if rate <= 0 || channels <= 0 {
		return "", 0, 0, nil, false, speechErrorf("invalid waveform header %dx%d", rate, channels)
	}
	return codec, rate, channels, payload, repaired, nil
}

// UtteranceToken is the token both texts of one utterance start with: the plain
// marker with a short digest of the waveform folded in, so every utterance has
// its own and the transcript and the sound share it.
func UtteranceToken(payload string, unique bool) string {
	if !unique {
		return SpeechToken
	}
	digest := Blake2b([]byte(payload), (speechTokenHex+1)/2)
	return SpeechToken[:len(SpeechToken)-1] + ":" + fmt.Sprintf("%x", digest)[:speechTokenHex] + ">"
}

// CheckSpeechToken validates a token: <speech>, <speech:hex> or any non-empty
// label without whitespace.
func CheckSpeechToken(token string) (string, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return "", speechErrorf("the token must not be empty")
	}
	if strings.ContainsAny(token, " \t\n\r\f\v") {
		return "", speechErrorf("the token must not contain whitespace: %q", token)
	}
	return token, nil
}

// SpeechTexts are the texts one utterance trains on, each starting with token:
// the transcript, the waveform, and with pair a third of the waveform followed
// by its transcript (so the search can run from the sound into the words).
func SpeechTexts(token, transcript, audioText string, pair bool) ([]string, error) {
	token, err := CheckSpeechToken(token)
	if err != nil {
		return nil, err
	}
	transcript = strings.Join(strings.Fields(transcript), " ")
	audioText = strings.TrimSpace(audioText)
	texts := []string{}
	if transcript != "" {
		texts = append(texts, token+" "+transcript)
	}
	if audioText != "" {
		texts = append(texts, token+" "+audioText)
		if pair && transcript != "" {
			texts = append(texts, token+" "+audioText+" "+transcript)
		}
	}
	return texts, nil
}

// -- the codecs ---------------------------------------------------------------

// MuLawEncode maps samples in [-1, 1] onto one mu-law byte each: the classic
// 8-bit speech quantisation, whose relative error stays at ~2% of the amplitude
// from full scale down to a whisper where linear 8-bit is already at 38%.
func MuLawEncode(samples []float32) []byte {
	scale := math.Log1p(MuLaw)
	out := make([]byte, len(samples))
	for i, sample := range samples {
		value := math.Max(-1, math.Min(1, float64(sample)))
		magnitude := math.Log1p(MuLaw*math.Abs(value)) / scale
		companded := magnitude
		if value < 0 {
			companded = -magnitude
		}
		level := int(math.Round((companded + 1) * 127.5))
		out[i] = byte(math.Max(0, math.Min(255, float64(level))))
	}
	return out
}

// MuLawDecode is the inverse of MuLawEncode.
func MuLawDecode(payload []byte) []float32 {
	out := make([]float32, len(payload))
	for i, b := range payload {
		companded := float64(b)/127.5 - 1
		magnitude := (math.Pow(1+MuLaw, math.Abs(companded)) - 1) / MuLaw
		if companded < 0 {
			magnitude = -magnitude
		}
		out[i] = float32(magnitude)
	}
	return out
}

// PCM8Encode maps samples in [-1, 1] onto one signed byte each.
func PCM8Encode(samples []float32) []byte {
	out := make([]byte, len(samples))
	for i, sample := range samples {
		value := math.Max(-1, math.Min(1, float64(sample)))
		level := int(math.Round(value * 127))
		out[i] = byte(int8(math.Max(-127, math.Min(127, float64(level)))))
	}
	return out
}

// PCM8Decode is the inverse; -128 (which only a predicted text holds) clamps to -1.
func PCM8Decode(payload []byte) []float32 {
	out := make([]float32, len(payload))
	for i, b := range payload {
		value := float64(int8(b)) / 127
		out[i] = float32(math.Max(-1, value))
	}
	return out
}

// GetSpeechCodec returns (name, encode, decode); "auto" resolves to mu-law.
func GetSpeechCodec(name string) (string, func([]float32) []byte, func([]byte) []float32, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "" || key == "auto" {
		key = DefaultSpeechCodec
	}
	switch key {
	case "mu":
		return key, MuLawEncode, MuLawDecode, nil
	case "pcm8":
		return key, PCM8Encode, PCM8Decode, nil
	}
	return "", nil, nil, speechErrorf("unknown waveform codec %q; expected one of: %s",
		name, strings.Join(SpeechCodecs, ", "))
}

// CheckSpeechRate validates a sample rate.
func CheckSpeechRate(rate int) (int, error) {
	if rate < MinSpeechRate || rate > MaxSpeechRate {
		return 0, speechErrorf("the sample rate must be between %d and %d Hz, got %d",
			MinSpeechRate, MaxSpeechRate, rate)
	}
	return rate, nil
}

// -- audio --------------------------------------------------------------------

// Audio is decoded audio: interleaved samples in [-1, 1], the rate and the
// channel count.
//
// The samples are float32, not float64, because Python's are: it carries them
// in an array("f"), so every value is rounded to single precision as it is
// stored.  The waveform text has to come out byte-identical on both sides - the
// utterance token is a digest of it, and the text is what the model trains on -
// so the arithmetic is done in float64 and rounded to float32 at exactly the
// points Python rounds.
type Audio struct {
	Samples  []float32
	Rate     int
	Channels int
}

// Seconds is the length of the recording.
func (a *Audio) Seconds() float64 {
	if a.Rate <= 0 || a.Channels <= 0 {
		return 0
	}
	return float64(len(a.Samples)) / float64(a.Rate*a.Channels)
}

// WAVE format tags the reader understands.
const (
	wavePCM         = 0x0001
	waveFloat       = 0x0003
	waveALaw        = 0x0006
	waveMuLaw       = 0x0007
	waveExtensible  = 0xFFFE
	riffHeaderBytes = 12
)

// ParseWAV reads a RIFF/WAVE file: PCM 8 (unsigned) / 16 / 24 / 32, IEEE float
// 32 / 64, A-law and mu-law, WAVE_FORMAT_EXTENSIBLE resolved through its
// SubFormat tag, chunks walked in order with the odd-length padding, and a
// streamed data chunk of declared size 0 read to the end.
func ParseWAV(data []byte) (*Audio, error) {
	if len(data) < riffHeaderBytes || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, speechErrorf("not a RIFF/WAVE file")
	}
	format, channels, rate, bits := 0, 0, 0, 0
	var body []byte
	at := riffHeaderBytes
	for at+8 <= len(data) {
		id := string(data[at : at+4])
		size := int(binary.LittleEndian.Uint32(data[at+4 : at+8]))
		start := at + 8
		if size == 0 && id == "data" {
			size = len(data) - start // a streamed file declares nothing
		}
		if start+size > len(data) {
			size = len(data) - start
		}
		if size < 0 {
			break
		}
		chunk := data[start : start+size]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return nil, speechErrorf("the fmt chunk is too short")
			}
			format = int(binary.LittleEndian.Uint16(chunk[0:2]))
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			rate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bits = int(binary.LittleEndian.Uint16(chunk[14:16]))
			if format == waveExtensible && len(chunk) >= 26 {
				format = int(binary.LittleEndian.Uint16(chunk[24:26])) // the SubFormat's first two bytes
			}
		case "data":
			body = chunk
		}
		at = start + size
		if size%2 == 1 {
			at++ // chunks are padded to an even length
		}
	}
	if channels < 1 || rate < 1 || body == nil {
		return nil, speechErrorf("the WAV file has no usable fmt / data chunks")
	}
	samples, err := samplesFromPCM(body, format, bits)
	if err != nil {
		return nil, err
	}
	return &Audio{Samples: samples, Rate: rate, Channels: channels}, nil
}

// samplesFromPCM turns a data chunk into floats in [-1, 1].
func samplesFromPCM(data []byte, format, bits int) ([]float32, error) {
	switch {
	case format == waveMuLaw:
		return MuLawDecode(data), nil
	case format == waveALaw:
		out := make([]float32, len(data))
		for i, b := range data {
			out[i] = float32(aLawToLinear(b))
		}
		return out, nil
	case format == waveFloat && bits == 32:
		out := make([]float32, len(data)/4)
		for i := range out {
			value := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : i*4+4])))
			out[i] = float32(math.Max(-1, math.Min(1, value)))
		}
		return out, nil
	case format == waveFloat && bits == 64:
		out := make([]float32, len(data)/8)
		for i := range out {
			value := math.Float64frombits(binary.LittleEndian.Uint64(data[i*8 : i*8+8]))
			out[i] = float32(math.Max(-1, math.Min(1, value)))
		}
		return out, nil
	case format == wavePCM && bits == 8: // unsigned
		out := make([]float32, len(data))
		for i, b := range data {
			out[i] = float32((float64(b) - 128) / 128)
		}
		return out, nil
	case format == wavePCM && bits == 16:
		out := make([]float32, len(data)/2)
		for i := range out {
			out[i] = float32(float64(int16(binary.LittleEndian.Uint16(data[i*2:i*2+2]))) / 32768)
		}
		return out, nil
	case format == wavePCM && bits == 24:
		out := make([]float32, len(data)/3)
		for i := range out {
			raw := int32(data[i*3]) | int32(data[i*3+1])<<8 | int32(data[i*3+2])<<16
			if raw&0x800000 != 0 {
				raw |= ^0xFFFFFF // sign-extend
			}
			out[i] = float32(float64(raw) / 8388608)
		}
		return out, nil
	case format == wavePCM && bits == 32:
		out := make([]float32, len(data)/4)
		for i := range out {
			out[i] = float32(float64(int32(binary.LittleEndian.Uint32(data[i*4:i*4+4]))) / 2147483648)
		}
		return out, nil
	}
	return nil, speechErrorf("unsupported WAV sample format (tag %d, %d bits)", format, bits)
}

// aLawToLinear is the G.711 A-law expansion.
func aLawToLinear(b byte) float64 {
	b ^= 0x55
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1
		b &= 0x7F
	}
	exponent := int(b>>4) & 0x07
	mantissa := int(b & 0x0F)
	value := 0
	if exponent == 0 {
		value = (mantissa << 4) + 8
	} else {
		value = ((mantissa << 4) + 0x108) << (exponent - 1)
	}
	return sign * float64(value) / 32768
}

// ToMono averages interleaved channels down to one.
func ToMono(samples []float32, channels int) []float32 {
	if channels <= 1 {
		return samples
	}
	frames := len(samples) / channels
	out := make([]float32, frames)
	for i := 0; i < frames; i++ {
		sum := 0.0
		for c := 0; c < channels; c++ {
			sum += float64(samples[i*channels+c])
		}
		out[i] = float32(sum / float64(channels))
	}
	return out
}

// Resample changes the sample rate by linear interpolation (the next step
// quantises to a byte anyway).
func Resample(samples []float32, src, dst int) []float32 {
	if src == dst || len(samples) == 0 || src <= 0 || dst <= 0 {
		return samples
	}
	count := int(math.Round(float64(len(samples)) * float64(dst) / float64(src)))
	if count < 1 {
		count = 1
	}
	step := float64(src) / float64(dst) // the rate ratio, not a stretch to the last sample
	last := len(samples) - 1
	out := make([]float32, count)
	for i := range out {
		position := float64(i) * step
		low := int(position)
		if low >= last {
			out[i] = samples[last]
			continue
		}
		fraction := position - float64(low)
		out[i] = float32(float64(samples[low])*(1-fraction) + float64(samples[low+1])*fraction)
	}
	return out
}

// NormaliseSamples scales a quiet recording up so its peak reaches headroom.
func NormaliseSamples(samples []float32, headroom float64) []float32 {
	peak := 0.0
	for _, value := range samples {
		peak = math.Max(peak, math.Abs(float64(value)))
	}
	if peak <= 1e-9 { // silence is left alone
		return samples
	}
	gain := headroom / peak
	out := make([]float32, len(samples))
	for i, value := range samples {
		out[i] = float32(float64(value) * gain)
	}
	return out
}

// WAVBytes writes samples in [-1, 1] as a 16-bit PCM WAV file.
func WAVBytes(samples []float32, rate, channels int) []byte {
	body := make([]byte, len(samples)*2)
	for i, sample := range samples {
		level := int(math.Round(math.Max(-32768, math.Min(32767, float64(sample)*32767))))
		binary.LittleEndian.PutUint16(body[i*2:], uint16(int16(level)))
	}
	out := make([]byte, 0, 44+len(body))
	out = append(out, "RIFF"...)
	out = binary.LittleEndian.AppendUint32(out, uint32(36+len(body)))
	out = append(out, "WAVE"...)
	out = append(out, "fmt "...)
	out = binary.LittleEndian.AppendUint32(out, 16)
	out = binary.LittleEndian.AppendUint16(out, wavePCM)
	out = binary.LittleEndian.AppendUint16(out, uint16(channels))
	out = binary.LittleEndian.AppendUint32(out, uint32(rate))
	out = binary.LittleEndian.AppendUint32(out, uint32(rate*channels*2))
	out = binary.LittleEndian.AppendUint16(out, uint16(channels*2))
	out = binary.LittleEndian.AppendUint16(out, 16)
	out = append(out, "data"...)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(body)))
	return append(out, body...)
}

// -- encode / decode ----------------------------------------------------------

// EncodedAudio is what EncodeAudio produces.
type EncodedAudio struct {
	Text       string  `json:"text"`
	Codec      string  `json:"codec"`
	Rate       int     `json:"rate"`
	Channels   int     `json:"channels"`
	Samples    int     `json:"samples"`
	Seconds    float64 `json:"seconds"`
	Bytes      int     `json:"bytes"`
	Chars      int     `json:"chars"`
	Normalised bool    `json:"normalised"`
	Source     string  `json:"source"`
}

// EncodeAudio turns WAV bytes into the waveform text: mixed to mono, resampled,
// optionally peak-normalised, quantised to one byte per sample and base64'd.
func EncodeAudio(data []byte, rate int, codec string, normalise bool) (*EncodedAudio, error) {
	if rate == 0 {
		rate = DefaultSpeechRate
	}
	rate, err := CheckSpeechRate(rate)
	if err != nil {
		return nil, err
	}
	name, encode, _, err := GetSpeechCodec(codec)
	if err != nil {
		return nil, err
	}
	audio, err := ParseWAV(data)
	if err != nil {
		return nil, err
	}
	samples := Resample(ToMono(audio.Samples, audio.Channels), audio.Rate, rate)
	if normalise {
		samples = NormaliseSamples(samples, 0.99)
	}
	payload := encode(samples)
	text := PackSpeechText(name, rate, 1, payload)
	return &EncodedAudio{
		Text: text, Codec: name, Rate: rate, Channels: 1, Samples: len(samples),
		Seconds: float64(len(samples)) / float64(rate), Bytes: len(payload), Chars: len(text),
		Normalised: normalise, Source: fmt.Sprintf("%d Hz, %d channel(s)", audio.Rate, audio.Channels),
	}, nil
}

// DecodedAudio is what DecodeSpeechText produces.
type DecodedAudio struct {
	WAV      []byte  `json:"-"`
	Codec    string  `json:"codec"`
	Rate     int     `json:"rate"`
	Channels int     `json:"channels"`
	Samples  int     `json:"samples"`
	Seconds  float64 `json:"seconds"`
	Bytes    int     `json:"bytes"`
	Repaired bool    `json:"repaired"`
}

// DecodeSpeechText turns an encoded - or predicted - waveform text back into a
// 16-bit PCM WAV, so what the graph says can be listened to.
func DecodeSpeechText(text, codec string) (*DecodedAudio, error) {
	headerCodec, rate, channels, payload, repaired, err := ParseSpeechText(text)
	if err != nil {
		return nil, err
	}
	name := codec
	if strings.TrimSpace(name) == "" {
		name = headerCodec
	}
	name, _, decode, err := GetSpeechCodec(name)
	if err != nil {
		return nil, err
	}
	if _, err := CheckSpeechRate(rate); err != nil {
		return nil, err
	}
	samples := decode(payload)
	return &DecodedAudio{
		WAV: WAVBytes(samples, rate, channels), Codec: name, Rate: rate, Channels: channels,
		Samples: len(samples), Seconds: float64(len(samples)) / float64(rate*channels),
		Bytes: len(payload), Repaired: repaired,
	}, nil
}

// TaughtSpeech is what TeachSpeech produces: the texts one utterance trains on.
type TaughtSpeech struct {
	Token      string        `json:"token"`
	Transcript string        `json:"transcript"`
	Audio      *EncodedAudio `json:"audio"`
	Texts      []string      `json:"texts"`
	Chars      int           `json:"chars"`
	Pair       bool          `json:"pair"`
}

// TeachSpeechOptions steer one utterance.
type TeachSpeechOptions struct {
	Transcript string
	Rate       int
	Codec      string
	Normalise  bool
	Waveform   bool
	Pair       bool
	Token      string
	Unique     bool
}

// DefaultTeachSpeechOptions mirror the Python defaults.
func DefaultTeachSpeechOptions() TeachSpeechOptions {
	return TeachSpeechOptions{Rate: DefaultSpeechRate, Codec: "auto", Waveform: true, Unique: true}
}

// TeachSpeech turns one utterance into the texts the network learns, both
// behind the same unique token.
//
// Unlike the Python side this does not transcribe: the local Whisper backends
// are Python packages, so the transcript must be given (which is what the
// browser's dictation, --text and the API's `transcript` supply).
func TeachSpeech(data []byte, o TeachSpeechOptions) (*TaughtSpeech, error) {
	words := strings.Join(strings.Fields(o.Transcript), " ")
	if len(data) == 0 && words == "" {
		return nil, speechErrorf("nothing to learn: pass audio, a transcript, or both")
	}
	var audio *EncodedAudio
	if len(data) > 0 && o.Waveform {
		encoded, err := EncodeAudio(data, o.Rate, o.Codec, o.Normalise)
		if err != nil {
			return nil, err
		}
		audio = encoded
	}
	seed := words
	audioText := ""
	if audio != nil {
		seed, audioText = audio.Text, audio.Text
	}
	token := o.Token
	if strings.TrimSpace(token) == "" {
		token = UtteranceToken(seed, o.Unique)
	}
	token, err := CheckSpeechToken(token)
	if err != nil {
		return nil, err
	}
	texts, err := SpeechTexts(token, words, audioText, o.Pair)
	if err != nil {
		return nil, err
	}
	chars := 0
	for _, text := range texts {
		chars += len([]rune(text))
	}
	return &TaughtSpeech{
		Token: token, Transcript: words, Audio: audio, Texts: texts, Chars: chars,
		Pair: o.Pair && words != "" && audio != nil,
	}, nil
}

// DescribeSpeech reports what this side can do with audio.
func DescribeSpeech() map[string]any {
	return map[string]any{
		"engine": "go", "codecs": SpeechCodecs, "default_codec": DefaultSpeechCodec,
		"default_rate": DefaultSpeechRate, "token": SpeechToken,
		"token_example": UtteranceToken("example", true),
		// the keys the Speech tab reads, answered truthfully for this build: the
		// words come with the audio, and nothing here shells out to ffmpeg
		"faster_whisper": false, "whisper": false, "whisper_model": "", "server_url": nil,
		"auto": "given", "given_always": true, "ffmpeg": false, "recorders": []string{},
		"backends": []string{"given"},
		"backends_note": "transcription backends are Python-only (faster-whisper / openai-whisper are Python " +
			"packages): send the words with the audio, which is what the browser's dictation does",
		"text_format": speechHeader + ":<codec>:<rate>x<channels>:<base64 of one byte per sample>",
		"formats":     "WAV (PCM 8/16/24/32, IEEE float 32/64, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE)",
	}
}
