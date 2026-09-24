package phonetok

// A voice for the sounds (the port of synth.py): a source-filter formant
// synthesizer that speaks phones as they arrive, over the same voice table.

import (
	_ "embed"
	"encoding/binary"
	"math"
	"strconv"
	"strings"
)

//go:embed data/voice.tsv
var voiceText string

// Rate is the default sample rate.
const Rate = 16000

// Sound is one row of the voice table: what a phone is made of.
type Sound struct {
	Phone    string
	Kind     string
	F        [3]float64
	BW       [3]float64
	Ms       float64
	Voiced   bool
	NoiseF   float64
	NoiseBW  float64
	NoiseAmp float64
	Amp      float64
	F2       *[3]float64 // a diphthong's second target
}

// VoiceTable is phone -> Sound, read once.
var VoiceTable = func() map[string]Sound {
	table := map[string]Sound{}
	for _, line := range strings.Split(voiceText, "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Fields(line)
		nums := make([]float64, 0, 15)
		for _, c := range cols[2:] {
			v, _ := strconv.ParseFloat(c, 64)
			nums = append(nums, v)
		}
		s := Sound{
			Phone: cols[0], Kind: cols[1], F: [3]float64{nums[0], nums[1], nums[2]}, BW: [3]float64{nums[3], nums[4], nums[5]},
			Ms: nums[6], Voiced: nums[7] >= 1, NoiseF: nums[8], NoiseBW: nums[9], NoiseAmp: nums[10], Amp: nums[11],
		}
		if s.Kind == "diph" {
			s.F2 = &[3]float64{nums[12], nums[13], nums[14]}
		}
		table[s.Phone] = s
	}
	return table
}()

// VoiceSettings is how the voice is set: its pitch, its pace, its loudness, the pauses it takes.
type VoiceSettings struct {
	Pitch      float64
	Tempo      float64
	Gain       float64
	ShortPause float64
	FullPause  float64
	Lead       float64
}

// DefaultVoice is the voice as synth.py sets it.
func DefaultVoice() VoiceSettings {
	return VoiceSettings{Pitch: 120, Tempo: 1, Gain: 0.5, ShortPause: 0.15, FullPause: 0.40, Lead: 0.05}
}

type noiseSource struct{ state uint32 }

func newNoise() *noiseSource { return &noiseSource{state: 0x9E3779B9} }

func (n *noiseSource) next() float64 {
	x := n.state
	x ^= x << 13
	x ^= x >> 17
	x ^= x << 5
	n.state = x
	return float64(x)/2147483648.0 - 1.0
}

func resonator(f, bw, rate float64) (a, b, c float64) {
	c = -math.Exp(-2.0 * math.Pi * bw / rate)
	b = 2.0 * math.Exp(-math.Pi*bw/rate) * math.Cos(2.0*math.Pi*f/rate)
	a = 1.0 - b - c
	return
}

type segment struct {
	fFrom, fTo, bw            [3]float64
	seconds                   float64
	voiced                    bool
	amp                       float64
	noiseF, noiseBW, noiseAmp float64
	pitchFrom, pitchTo        float64
	aspiration                bool
}

// Synthesizer speaks phones as they arrive: feed it tokens, take the PCM; End closes the utterance.
type Synthesizer struct {
	Rate         int
	Voice        VoiceSettings
	noise        *noiseSource
	pending      []string
	lastF        [3]float64
	y            [4][2]float64
	phase        float64
	srcPrev      float64
	started      bool
	elapsed      float64
	words        int
	TotalSamples int
}

// NewSynthesizer makes a synthesizer at a rate with a voice.
func NewSynthesizer(rate int, voice VoiceSettings) *Synthesizer {
	return &Synthesizer{Rate: rate, Voice: voice, noise: newNoise(), lastF: [3]float64{500, 1500, 2500}, phase: 1.0}
}

// Feed takes one token; the PCM that can be committed comes back (often nothing).
func (s *Synthesizer) Feed(token string) []byte {
	if token == EOS {
		return s.End()
	}
	if token == BOS {
		return nil
	}
	kind, phones, ok := ParseToken(token)
	if !ok {
		return nil
	}
	switch kind {
	case "boundary":
		return s.flush("")
	case "pause":
		pause := s.Voice.FullPause
		if token == PauseShort {
			pause = s.Voice.ShortPause
		}
		return append(s.flush(token), s.silence(pause)...)
	case "special":
		return nil
	}
	s.pending = append(s.pending, phones...)
	return nil
}

// End is the final sentinel: what is pending is spoken with the closing intonation, then the utterance ends.
func (s *Synthesizer) End() []byte {
	out := s.flush(PauseFull)
	if s.started {
		out = append(out, s.silence(0.12)...)
	}
	s.started, s.elapsed, s.words = false, 0, 0
	s.lastF = [3]float64{500, 1500, 2500}
	return out
}

// Speak is the whole of a token stream, ended.
func (s *Synthesizer) Speak(tokens []string) []byte {
	var out []byte
	for _, t := range tokens {
		out = append(out, s.Feed(t)...)
	}
	return append(out, s.End()...)
}

func (s *Synthesizer) flush(final string) []byte {
	phones := s.pending
	s.pending = nil
	if len(phones) == 0 {
		return nil
	}
	var out []byte
	if !s.started {
		out = append(out, s.silence(s.Voice.Lead)...)
		s.started = true
	}
	for _, seg := range s.wordSegments(phones, final) {
		out = append(out, s.render(seg)...)
	}
	s.words++
	return out
}

func (s *Synthesizer) pitchAt(seconds float64, stress int, final string, lastSyllable bool) float64 {
	f0 := s.Voice.Pitch * (1.0 - 0.12*math.Min(seconds/3.0, 1.0))
	switch stress {
	case 1:
		f0 *= 1.18
	case 2:
		f0 *= 1.08
	}
	switch {
	case lastSyllable && final == PauseQuestion:
		f0 *= 1.35
	case lastSyllable && final == PauseFull:
		f0 *= 0.82
	case lastSyllable && final == PauseShort:
		f0 *= 1.06
	}
	return f0
}

func (s *Synthesizer) wordSegments(phones []string, final string) []segment {
	lastVowel := len(phones) - 1
	for i := len(phones) - 1; i >= 0; i-- {
		if IsVowel(phones[i]) {
			lastVowel = i
			break
		}
	}
	var segs []segment
	t := s.elapsed
	tempo := s.Voice.Tempo
	for i, phone := range phones {
		b := Base(phone)
		sound, ok := VoiceTable[b]
		if !ok {
			continue
		}
		stress := StressOf(phone)
		lastSyllable := i >= lastVowel
		seconds := sound.Ms / 1000.0 / tempo
		if IsVowel(b) {
			if stress == 1 {
				seconds *= 1.25
			} else if stress == 0 {
				seconds *= 0.75
			}
		}
		if lastSyllable && final != "" {
			if IsVowel(b) {
				seconds *= 1.35
			} else {
				seconds *= 1.15
			}
		}
		amp := sound.Amp
		if IsVowel(b) && stress == 0 {
			amp *= 0.7
		}
		pitchFrom := s.pitchAt(t, stress, final, lastSyllable)
		pitchTo := s.pitchAt(t+seconds, stress, final, lastSyllable)
		switch sound.Kind {
		case "stop", "affr":
			closure := seconds * 0.4
			if sound.Kind == "stop" {
				closure = seconds * 0.55
			}
			segs = append(segs, segment{s.lastF, sound.F, sound.BW, closure, sound.Voiced, sound.Amp * 0.4, 0, 0, 0, pitchFrom, pitchFrom, false})
			burst := seconds - closure
			if sound.Kind == "stop" {
				burst = 0.012
			}
			segs = append(segs, segment{sound.F, sound.F, sound.BW, burst, false, 0, sound.NoiseF, sound.NoiseBW, sound.NoiseAmp, pitchFrom, pitchFrom, false})
			if !sound.Voiced && sound.Kind == "stop" {
				segs = append(segs, segment{sound.F, sound.F, sound.BW, 0.035, false, 0, 0, 0, 0.18, pitchFrom, pitchFrom, true})
			}
			s.lastF = sound.F
		case "asp":
			segs = append(segs, segment{s.lastF, sound.F, sound.BW, seconds, false, 0, 0, 0, sound.NoiseAmp, pitchFrom, pitchTo, true})
			s.lastF = sound.F
		default:
			if sound.Kind == "diph" && sound.F2 != nil {
				half := seconds / 2.0
				mid := (pitchFrom + pitchTo) / 2.0
				segs = append(segs, segment{s.lastF, sound.F, sound.BW, half, true, amp, 0, 0, 0, pitchFrom, mid, false})
				segs = append(segs, segment{sound.F, *sound.F2, sound.BW, half, true, amp, 0, 0, 0, mid, pitchTo, false})
				s.lastF = *sound.F2
			} else {
				segs = append(segs, segment{s.lastF, sound.F, sound.BW, seconds, sound.Voiced, amp, sound.NoiseF, sound.NoiseBW, sound.NoiseAmp, pitchFrom, pitchTo, false})
				s.lastF = sound.F
			}
		}
		t += seconds
	}
	s.elapsed = t
	return segs
}

func (s *Synthesizer) silence(seconds float64) []byte {
	n := int(math.Round(seconds * float64(s.Rate)))
	s.TotalSamples += n
	s.elapsed += seconds
	return make([]byte, 2*n)
}

func (s *Synthesizer) render(seg segment) []byte {
	rate := float64(s.Rate)
	total := int(math.Round(seg.seconds * rate))
	if total < 1 {
		total = 1
	}
	frame := int(rate * 5.0 / 1000.0)
	if frame < 1 {
		frame = 1
	}
	out := make([]byte, 0, 2*total)
	gain := s.Voice.Gain * 32767.0
	nyquist := rate * 0.45
	done := 0
	for done < total {
		n := frame
		if total-done < n {
			n = total - done
		}
		mid := (float64(done) + float64(n)/2.0) / float64(total)
		g := math.Min(mid/0.4, 1.0)
		var coefs [3][3]float64
		for k := 0; k < 3; k++ {
			f := seg.fFrom[k] + (seg.fTo[k]-seg.fFrom[k])*g
			a, b, c := resonator(math.Min(f, nyquist), seg.bw[k], rate)
			coefs[k] = [3]float64{a, b, c}
		}
		hasNoise := seg.noiseAmp > 0 && !seg.aspiration
		var na, nb, nc float64
		if hasNoise {
			na, nb, nc = resonator(math.Min(seg.noiseF, nyquist), math.Max(seg.noiseBW, 50.0), rate)
		}
		pitch := seg.pitchFrom + (seg.pitchTo-seg.pitchFrom)*mid
		period := rate / math.Max(pitch, 40.0)
		for i := 0; i < n; i++ {
			src := 0.0
			if seg.voiced && seg.amp > 0 {
				s.phase += 1.0 / period
				if s.phase >= 1.0 {
					s.phase -= 1.0
				}
				ph := s.phase
				var flow float64
				switch {
				case ph < 0.4:
					flow = 0.5 * (1.0 - math.Cos(math.Pi*ph/0.4))
				case ph < 0.6:
					flow = math.Cos(math.Pi * (ph - 0.4) / 0.4)
				default:
					flow = 0.0
				}
				src = (flow - s.srcPrev) * 4.0 * seg.amp
				s.srcPrev = flow
			}
			if seg.aspiration && seg.noiseAmp > 0 {
				src += s.noise.next() * seg.noiseAmp * 0.6
			}
			x := src
			for k := 0; k < 3; k++ {
				m := &s.y[k]
				v := coefs[k][0]*x + coefs[k][1]*m[0] + coefs[k][2]*m[1]
				m[1] = m[0]
				m[0] = v
				x = v
			}
			if hasNoise {
				m := &s.y[3]
				v := na*s.noise.next()*seg.noiseAmp + nb*m[0] + nc*m[1]
				m[1] = m[0]
				m[0] = v
				x += v * 3.0
			}
			sample := x * gain * 0.09
			if sample > 32767.0 {
				sample = 32767.0
			} else if sample < -32767.0 {
				sample = -32767.0
			}
			out = binary.LittleEndian.AppendUint16(out, uint16(int16(sample)))
		}
		done += n
	}
	s.TotalSamples += total
	return out
}

// WavHeader is the 44-byte header of a 16-bit mono WAV; samples < 0 gives the streaming form.
func WavHeader(rate int, samples int) []byte {
	size := uint32(0xFFFFFFFF)
	riff := uint32(0xFFFFFFFF)
	if samples >= 0 {
		size = uint32(samples * 2)
		riff = 36 + size
	}
	h := make([]byte, 0, 44)
	h = append(h, "RIFF"...)
	h = binary.LittleEndian.AppendUint32(h, riff)
	h = append(h, "WAVEfmt "...)
	h = binary.LittleEndian.AppendUint32(h, 16)
	h = binary.LittleEndian.AppendUint16(h, 1)
	h = binary.LittleEndian.AppendUint16(h, 1)
	h = binary.LittleEndian.AppendUint32(h, uint32(rate))
	h = binary.LittleEndian.AppendUint32(h, uint32(rate*2))
	h = binary.LittleEndian.AppendUint16(h, 2)
	h = binary.LittleEndian.AppendUint16(h, 16)
	h = append(h, "data"...)
	h = binary.LittleEndian.AppendUint32(h, size)
	return h
}

// WavBytes is a WAV file holding pcm.
func WavBytes(pcm []byte, rate int) []byte {
	return append(WavHeader(rate, len(pcm)/2), pcm...)
}
