package radixnet

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func tone(seconds float64, rate int, freq, amplitude float64) []float32 {
	n := int(float64(rate) * seconds)
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(amplitude * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return out
}

func TestSpeechTextPackAndParse(t *testing.T) {
	payload := []byte{0, 64, 128, 192, 255}
	text := PackSpeechText("mu", 8000, 1, payload)
	if !strings.HasPrefix(text, "aud:mu:8000x1:") {
		t.Fatalf("text: %q", text)
	}
	codec, rate, channels, back, repaired, err := ParseSpeechText(text)
	if err != nil {
		t.Fatalf("ParseSpeechText: %v", err)
	}
	if codec != "mu" || rate != 8000 || channels != 1 || repaired {
		t.Errorf("parsed %q %d %d repaired=%v", codec, rate, channels, repaired)
	}
	if !bytes.Equal(back, payload) {
		t.Errorf("payload: %v", back)
	}
}

func TestSpeechHeaderIsFoundBehindATokenAndBeforeATranscript(t *testing.T) {
	body := PackSpeechText("mu", 8000, 1, []byte{1, 2, 3})
	for _, text := range []string{
		"<speech:9f2a1c7d> " + body,
		"<speech:9f2a1c7d> " + body + " the cat sat on the mat",
	} {
		codec, _, _, payload, _, err := ParseSpeechText(text)
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		if codec != "mu" || len(payload) != 3 {
			t.Errorf("%q: codec %q, %d bytes", text, codec, len(payload))
		}
	}
}

func TestNotAWaveformText(t *testing.T) {
	if _, _, _, _, _, err := ParseSpeechText("the cat sat on the mat"); err == nil {
		t.Error("prose is not an encoded waveform")
	}
	if _, _, _, _, _, err := ParseSpeechText("aud:mu:0x1:AAAA"); err == nil {
		t.Error("a zero rate must be refused")
	}
}

func TestUtteranceTokenIsUniqueAndStable(t *testing.T) {
	first := UtteranceToken("aud:mu:8000x1:AAAA", true)
	if first != UtteranceToken("aud:mu:8000x1:AAAA", true) {
		t.Error("the same waveform must always give the same token")
	}
	if first == UtteranceToken("aud:mu:8000x1:AAAB", true) {
		t.Error("a different waveform must give a different token")
	}
	if !strings.HasPrefix(first, "<speech:") || !strings.HasSuffix(first, ">") {
		t.Errorf("token: %q", first)
	}
	if UtteranceToken("anything", false) != SpeechToken {
		t.Error("a shared token is the plain marker")
	}
	// no three-character window of the token can collide with the graph's sentinels
	if strings.Contains(first, "<s>") {
		t.Errorf("the token must not contain a sentinel: %q", first)
	}
}

func TestCheckSpeechToken(t *testing.T) {
	if _, err := CheckSpeechToken("  "); err == nil {
		t.Error("an empty token must be refused")
	}
	if _, err := CheckSpeechToken("two words"); err == nil {
		t.Error("whitespace in a token must be refused")
	}
	if got, err := CheckSpeechToken(" <speech:abc> "); err != nil || got != "<speech:abc>" {
		t.Errorf("a token is trimmed: %q %v", got, err)
	}
}

func TestSpeechTexts(t *testing.T) {
	texts, err := SpeechTexts("<speech:a>", "the cat  sat", "aud:mu:8000x1:AAAA", false)
	if err != nil {
		t.Fatalf("SpeechTexts: %v", err)
	}
	if len(texts) != 2 || texts[0] != "<speech:a> the cat sat" {
		t.Fatalf("texts: %v", texts)
	}
	paired, _ := SpeechTexts("<speech:a>", "the cat sat", "aud:mu:8000x1:AAAA", true)
	if len(paired) != 3 || !strings.HasSuffix(paired[2], "the cat sat") {
		t.Fatalf("paired: %v", paired)
	}
	only, _ := SpeechTexts("<speech:a>", "", "aud:mu:8000x1:AAAA", true)
	if len(only) != 1 {
		t.Errorf("a waveform with no transcript is one text: %v", only)
	}
}

func TestCodecsRoundTripBytes(t *testing.T) {
	for _, name := range []string{"mu", "pcm8"} {
		_, encode, decode, err := GetSpeechCodec(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		samples := tone(0.01, 8000, 220, 0.6)
		back := decode(encode(samples))
		if len(back) != len(samples) {
			t.Fatalf("%s: %d samples became %d", name, len(samples), len(back))
		}
		// re-encoding a decoded byte must give the same byte back (a stable code)
		once := encode(samples)
		if twice := encode(decode(once)); !bytes.Equal(once, twice) {
			t.Errorf("%s: the codec is not stable", name)
		}
	}
}

func TestMuLawBeatsLinearOnQuietAudio(t *testing.T) {
	quiet := tone(0.01, 8000, 220, 0.02) // a whisper
	muErr, pcmErr := 0.0, 0.0
	mu := MuLawDecode(MuLawEncode(quiet))
	pcm := PCM8Decode(PCM8Encode(quiet))
	for i, want := range quiet {
		muErr += math.Abs(float64(mu[i] - want))
		pcmErr += math.Abs(float64(pcm[i] - want))
	}
	if muErr >= pcmErr {
		t.Errorf("mu-law must be kinder to quiet audio: mu %g, pcm8 %g", muErr, pcmErr)
	}
}

func TestGetSpeechCodecRefusesTheUnknown(t *testing.T) {
	if name, _, _, err := GetSpeechCodec("auto"); err != nil || name != "mu" {
		t.Errorf("auto resolves to mu-law: %q %v", name, err)
	}
	if _, _, _, err := GetSpeechCodec("flac"); err == nil {
		t.Error("an unknown codec must be refused")
	}
}

func TestWAVRoundTrip(t *testing.T) {
	samples := tone(0.05, 16000, 440, 0.5)
	audio, err := ParseWAV(WAVBytes(samples, 16000, 1))
	if err != nil {
		t.Fatalf("ParseWAV: %v", err)
	}
	if audio.Rate != 16000 || audio.Channels != 1 || len(audio.Samples) != len(samples) {
		t.Fatalf("audio: %d Hz, %d ch, %d samples", audio.Rate, audio.Channels, len(audio.Samples))
	}
	for i, want := range samples {
		if math.Abs(float64(audio.Samples[i]-want)) > 1e-3 {
			t.Fatalf("sample %d: %g != %g", i, audio.Samples[i], want)
		}
	}
	if math.Abs(audio.Seconds()-0.05) > 1e-6 {
		t.Errorf("seconds: %g", audio.Seconds())
	}
}

func TestParseWAVRefusesRubbish(t *testing.T) {
	if _, err := ParseWAV([]byte("not a wav file at all")); err == nil {
		t.Error("non-RIFF bytes must be refused")
	}
	if _, err := ParseWAV([]byte("RIFF____WAVE")); err == nil {
		t.Error("a WAVE with no fmt / data must be refused")
	}
}

func TestToMonoAndResample(t *testing.T) {
	stereo := []float32{1, -1, 0.5, -0.5}
	mono := ToMono(stereo, 2)
	if len(mono) != 2 || mono[0] != 0 || mono[1] != 0 {
		t.Errorf("mono: %v", mono)
	}
	if got := ToMono(stereo, 1); len(got) != 4 {
		t.Error("one channel passes through")
	}
	up := Resample([]float32{0, 1}, 1, 2)
	if len(up) != 4 {
		t.Errorf("resampled to %d samples", len(up))
	}
	if got := Resample([]float32{0, 1}, 8000, 8000); len(got) != 2 {
		t.Error("the same rate passes through")
	}
}

func TestNormaliseSamples(t *testing.T) {
	quiet := []float32{0.1, -0.05}
	loud := NormaliseSamples(quiet, 0.99)
	if math.Abs(float64(loud[0])-0.99) > 1e-6 {
		t.Errorf("the peak must reach the headroom: %v", loud)
	}
	// normalising is a scale to the headroom in both directions, as Python's is:
	// only silence is left alone
	full := NormaliseSamples([]float32{1, -1}, 0.99)
	if math.Abs(float64(full[0])-0.99) > 1e-6 {
		t.Errorf("a full-scale recording is scaled down to the headroom: %v", full)
	}
	silence := []float32{0, 0}
	if got := NormaliseSamples(silence, 0.99); got[0] != 0 {
		t.Errorf("silence is left alone: %v", got)
	}
}

func TestEncodeAudioAndDecodeBack(t *testing.T) {
	wav := WAVBytes(tone(0.05, 16000, 440, 0.5), 16000, 1)
	encoded, err := EncodeAudio(wav, 8000, "auto", false)
	if err != nil {
		t.Fatalf("EncodeAudio: %v", err)
	}
	if encoded.Codec != "mu" || encoded.Rate != 8000 || encoded.Channels != 1 {
		t.Fatalf("encoded: %+v", encoded)
	}
	if encoded.Samples != 400 { // 0.05s at 8 kHz
		t.Errorf("samples: %d", encoded.Samples)
	}
	decoded, err := DecodeSpeechText(encoded.Text, "")
	if err != nil {
		t.Fatalf("DecodeSpeechText: %v", err)
	}
	if decoded.Samples != encoded.Samples || decoded.Repaired {
		t.Fatalf("decoded: %+v", decoded)
	}
	back, err := ParseWAV(decoded.WAV)
	if err != nil {
		t.Fatalf("the decoded WAV must be readable: %v", err)
	}
	if len(back.Samples) != encoded.Samples {
		t.Errorf("round trip: %d samples", len(back.Samples))
	}
}

func TestEncodeAudioRefusesBadRates(t *testing.T) {
	wav := WAVBytes(tone(0.01, 8000, 220, 0.5), 8000, 1)
	if _, err := EncodeAudio(wav, 10, "auto", false); err == nil {
		t.Error("a rate below the floor must be refused")
	}
	if _, err := EncodeAudio(wav, 8000, "flac", false); err == nil {
		t.Error("an unknown codec must be refused")
	}
}

func TestTeachSpeechBothTextsShareTheToken(t *testing.T) {
	wav := WAVBytes(tone(0.02, 8000, 220, 0.5), 8000, 1)
	o := DefaultTeachSpeechOptions()
	o.Transcript = "the cat sat on the mat"
	taught, err := TeachSpeech(wav, o)
	if err != nil {
		t.Fatalf("TeachSpeech: %v", err)
	}
	if len(taught.Texts) != 2 {
		t.Fatalf("texts: %v", len(taught.Texts))
	}
	for _, text := range taught.Texts {
		if !strings.HasPrefix(text, taught.Token+" ") {
			t.Errorf("every text starts with the token: %q", text[:40])
		}
	}
	if taught.Audio == nil || taught.Transcript != "the cat sat on the mat" {
		t.Errorf("taught: %+v", taught)
	}
}

func TestTeachSpeechTranscriptOnlyAndWaveformOnly(t *testing.T) {
	o := DefaultTeachSpeechOptions()
	o.Transcript = "the cat sat"
	taught, err := TeachSpeech(nil, o)
	if err != nil {
		t.Fatalf("TeachSpeech: %v", err)
	}
	if len(taught.Texts) != 1 || taught.Audio != nil {
		t.Fatalf("a transcript alone is one text: %+v", taught)
	}
	wav := WAVBytes(tone(0.02, 8000, 220, 0.5), 8000, 1)
	o = DefaultTeachSpeechOptions()
	silent, err := TeachSpeech(wav, o)
	if err != nil {
		t.Fatalf("TeachSpeech: %v", err)
	}
	if len(silent.Texts) != 1 || silent.Audio == nil {
		t.Fatalf("a waveform alone is one text: %+v", silent)
	}
	if _, err := TeachSpeech(nil, DefaultTeachSpeechOptions()); err == nil {
		t.Error("nothing at all must be refused")
	}
}

func TestTeachSpeechPairAndSharedToken(t *testing.T) {
	wav := WAVBytes(tone(0.02, 8000, 220, 0.5), 8000, 1)
	o := DefaultTeachSpeechOptions()
	o.Transcript, o.Pair = "the cat sat", true
	taught, _ := TeachSpeech(wav, o)
	if len(taught.Texts) != 3 || !taught.Pair {
		t.Fatalf("pair: %+v", taught.Texts)
	}
	o.Unique = false
	shared, _ := TeachSpeech(wav, o)
	if shared.Token != SpeechToken {
		t.Errorf("a shared token is the plain marker: %q", shared.Token)
	}
}

func TestDescribeSpeech(t *testing.T) {
	info := DescribeSpeech()
	if info["default_codec"] != "mu" || info["token"] != SpeechToken {
		t.Errorf("describe: %v", info)
	}
}
