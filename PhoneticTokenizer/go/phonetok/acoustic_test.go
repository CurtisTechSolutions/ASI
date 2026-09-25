package phonetok

// The acoustic units: audio to units with nothing written down, and back (acoustic.go).

import (
	"encoding/binary"
	"math"
	"math/cmplx"
	"math/rand"
	"strings"
	"testing"
)

func spokenSamples(t *testing.T, tokens string, pitch float64) []float64 {
	t.Helper()
	voice := DefaultVoice()
	voice.Pitch = pitch
	return PCMSamples(NewSynthesizer(Rate, voice).Speak(strings.Fields(tokens)))
}

var acousticClips = []string{
	"DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T",
	"DH AH0 # K W IH1 K # B R AW1 N # F AA1 K S # JH AH1 M P S # OW1 V ER0 # DH AH0 # L EY1 Z IY0 # D AO1 G",
	"P R AE1 K T AH0 S # M EY1 K S # P ER1 F IH0 K T",
}

func clipSamples(t *testing.T) [][]float64 {
	t.Helper()
	out := make([][]float64, len(acousticClips))
	for i, c := range acousticClips {
		out[i] = spokenSamples(t, c, 120)
	}
	return out
}

func smallBook(t *testing.T) *Codebook {
	t.Helper()
	book, err := Learn(clipSamples(t), 8, 3, 30, DefaultAnalysis(), "test")
	if err != nil {
		t.Fatal(err)
	}
	return book
}

func TestReadWAVEveryFormat(t *testing.T) {
	wav := func(tag, channels, rate, bits int, body []byte) []byte {
		fmtChunk := make([]byte, 16)
		binary.LittleEndian.PutUint16(fmtChunk[0:], uint16(tag))
		binary.LittleEndian.PutUint16(fmtChunk[2:], uint16(channels))
		binary.LittleEndian.PutUint32(fmtChunk[4:], uint32(rate))
		binary.LittleEndian.PutUint32(fmtChunk[8:], uint32(rate*channels*bits/8))
		binary.LittleEndian.PutUint16(fmtChunk[12:], uint16(channels*bits/8))
		binary.LittleEndian.PutUint16(fmtChunk[14:], uint16(bits))
		size := func(n int) []byte { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, uint32(n)); return b }
		var chunks []byte
		chunks = append(chunks, []byte("fmt ")...)
		chunks = append(chunks, size(16)...)
		chunks = append(chunks, fmtChunk...)
		chunks = append(chunks, []byte("LIST")...)
		chunks = append(chunks, size(3)...)
		chunks = append(chunks, 'a', 'b', 'c', 0)
		chunks = append(chunks, []byte("data")...)
		chunks = append(chunks, size(len(body))...)
		chunks = append(chunks, body...)
		out := []byte("RIFF")
		out = append(out, size(4+len(chunks))...)
		out = append(out, []byte("WAVE")...)
		return append(out, chunks...)
	}
	i16 := func(vs ...int16) []byte {
		b := make([]byte, 2*len(vs))
		for i, v := range vs {
			binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
		}
		return b
	}
	f32 := func(vs ...float32) []byte {
		b := make([]byte, 4*len(vs))
		for i, v := range vs {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return b
	}
	cases := []struct {
		data []byte
		rate int
		want []float64
	}{
		{wav(1, 1, 8000, 8, []byte{128, 255, 0}), 8000, []float64{0, 0.992, -1}},
		{wav(1, 2, 44100, 16, i16(16384, -16384, 8192, 8192)), 44100, []float64{0, 0.25}},
		{wav(1, 1, 16000, 24, []byte{0, 0, 0x40, 0, 0, 0xc0}), 16000, []float64{0.5, -0.5}},
		{wav(3, 1, 16000, 32, f32(0.25, -0.75)), 16000, []float64{0.25, -0.75}},
	}
	for _, c := range cases {
		samples, rate, err := ReadWAV(c.data)
		if err != nil {
			t.Fatal(err)
		}
		if rate != c.rate || len(samples) != len(c.want) {
			t.Fatalf("rate %d, %d samples", rate, len(samples))
		}
		for i, w := range c.want {
			if math.Abs(samples[i]-w) > 1e-3 {
				t.Errorf("sample %d: %g, want %g", i, samples[i], w)
			}
		}
	}
	if _, _, err := ReadWAV([]byte("not a wav at all")); err == nil {
		t.Error("junk was read as a WAV")
	}
	pcm := i16(0, 16384, -16384)
	back, rate, err := ReadWAV(WavBytes(pcm, 16000))
	if err != nil || rate != 16000 || back[1] != 0.5 || back[2] != -0.5 {
		t.Fatalf("the synthesizer's own WAV: %v %d %v", err, rate, back)
	}
	if got := PCMSamples(pcm); got[1] != 0.5 {
		t.Errorf("PCMSamples: %v", got)
	}
}

func TestResampleKeepsTheTone(t *testing.T) {
	src := make([]float64, 8000)
	for i := range src {
		src[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 8000)
	}
	out, err := Resample(src, 8000, 16000)
	if err != nil || len(out) != 16000 {
		t.Fatalf("%v, %d samples", err, len(out))
	}
	crossings := 0
	for i := 1001; i < 15001; i++ {
		if (out[i-1] < 0) != (out[i] < 0) {
			crossings++
		}
	}
	if per := float64(crossings) / (14000.0 / 16000.0); math.Abs(per-880) > 6 {
		t.Errorf("%g crossings per second, want 880", per)
	}
}

func TestFFTIsTheDFT(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	xs := make([]float64, 256)
	for i := range xs {
		xs[i] = rng.Float64()*2 - 1
	}
	re := append([]float64(nil), xs...)
	im := make([]float64, 256)
	FFT(re, im, false)
	for _, k := range []int{0, 1, 5, 100, 128, 200, 255} {
		var want complex128
		for n, x := range xs {
			want += complex(x, 0) * cmplx.Exp(complex(0, -2*math.Pi*float64(k)*float64(n)/256))
		}
		if cmplx.Abs(want-complex(re[k], im[k])) > 1e-9 {
			t.Errorf("bin %d: %v, want %v", k, complex(re[k], im[k]), want)
		}
	}
	FFT(re, im, true)
	for i, x := range xs {
		if math.Abs(re[i]-x) > 1e-12 {
			t.Fatalf("inverse: sample %d is %g, want %g", i, re[i], x)
		}
	}
}

func TestFramesAndFilters(t *testing.T) {
	a := DefaultAnalysis()
	filters := melFilters(a)
	if len(filters) != a.Bands {
		t.Fatalf("%d filters", len(filters))
	}
	covered := make([]float64, a.FFT/2+1)
	for _, f := range filters {
		if len(f.weights) == 0 {
			t.Fatal("an empty filter")
		}
		for i, w := range f.weights {
			covered[f.first+i] += w
		}
	}
	for k := 3; k < 254; k++ {
		if covered[k] <= 0 {
			t.Errorf("bin %d is under no filter", k)
		}
	}
	samples := spokenSamples(t, acousticClips[0], 120)
	frames, err := FramesOf(samples, a)
	if err != nil {
		t.Fatal(err)
	}
	if want := (len(samples)-a.Frame)/a.Hop + 1; len(frames) != want {
		t.Fatalf("%d frames, want %d", len(frames), want)
	}
	for b := 0; b < a.Bands; b++ {
		mean := 0.0
		for _, f := range frames {
			mean += f[b]
		}
		if math.Abs(mean/float64(len(frames))) > 1e-9 {
			t.Errorf("band %d is not normalised: mean %g", b, mean)
		}
	}
	one, _ := FramesOf([]float64{0, 0, 0}, a)
	if len(one) != 1 {
		t.Errorf("short audio gave %d frames", len(one))
	}
	if err := (Analysis{Rate: 16000, Frame: 400, Hop: 160, FFT: 500, Bands: 40, Fmin: 20, Fmax: 8000}).Validate(); err == nil {
		t.Error("an fft of 500 was accepted")
	}
	vowel, _ := Frames(spokenSamples(t, "AE1 AE1 AE1", 120), a, false)
	hiss, _ := Frames(spokenSamples(t, "S S S", 120), a, false)
	if v := VoicingOf(vowel[len(vowel)/2], a); v != 1 {
		t.Errorf("a vowel's voicing is %g", v)
	}
	if v := VoicingOf(hiss[len(hiss)/2], a); v != 0 {
		t.Errorf("a hiss's voicing is %g", v)
	}
}

func TestKMeansFindsSeparatedBlobs(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	var points [][]float64
	for _, c := range [][2]float64{{0, 0}, {10, 0}, {0, 10}} {
		for i := 0; i < 40; i++ {
			points = append(points, []float64{c[0] + rng.NormFloat64()*0.5, c[1] + rng.NormFloat64()*0.5})
		}
	}
	found, err := KMeans(points, 3, NewMT(1), 50)
	if err != nil || !found.Converged {
		t.Fatalf("%v converged=%v", err, found.Converged)
	}
	seen := map[[2]int]bool{}
	for _, c := range found.Centroids {
		seen[[2]int{int(math.Round(c[0])), int(math.Round(c[1]))}] = true
	}
	if !seen[[2]int{0, 0}] || !seen[[2]int{10, 0}] || !seen[[2]int{0, 10}] {
		t.Errorf("centroids %v", found.Centroids)
	}
	again, _ := KMeans(points, 3, NewMT(1), 50)
	for c := range found.Centroids {
		if found.Centroids[c][0] != again.Centroids[c][0] {
			t.Fatal("the same seed learned something else")
		}
	}
	if _, err := KMeans(points[:2], 3, NewMT(1), 5); err == nil {
		t.Error("two points made three units")
	}
}

func TestLearningACodebookFromSpeech(t *testing.T) {
	book := smallBook(t)
	if book.K() != 8 || len(book.Mean) != 40 {
		t.Fatalf("k %d, mean %d", book.K(), len(book.Mean))
	}
	for i, c := range book.Counts {
		if c == 0 || book.Runs[i] < 1 {
			t.Errorf("unit %d: %d frames, run %g", i, c, book.Runs[i])
		}
	}
	back, err := ParseCodebook(book.Dumps())
	if err != nil {
		t.Fatal(err)
	}
	for i := range book.Centroids {
		for b := range book.Centroids[i] {
			if back.Centroids[i][b] != book.Centroids[i][b] {
				t.Fatal("the file lost a digit")
			}
		}
	}
	if back.Note != "test" || back.Seed != 3 || back.Analysis != book.Analysis || back.Dumps() != book.Dumps() {
		t.Error("the file does not carry everything")
	}
	if book.Index("q0") != 0 || book.Index("q7") != 7 {
		t.Error("unit names")
	}
	for _, bad := range []string{"q8", "q07", "q", "7", "Q1", "DH", "#"} {
		if book.Index(bad) >= 0 {
			t.Errorf("%q is a unit", bad)
		}
	}
	if got := CollapseRuns([]int{1, 1, 2, 2, 2, 1, 3, 3}); len(got) != 4 || got[3] != 3 {
		t.Errorf("collapse: %v", got)
	}
	if _, err := Learn(nil, 4, 1, 5, DefaultAnalysis(), ""); err == nil {
		t.Error("nothing was learned from")
	}
}

func TestHearingAndSpeakingBack(t *testing.T) {
	book := smallBook(t)
	tok, err := NewAcousticTokenizer(book, true)
	if err != nil {
		t.Fatal(err)
	}
	clips := clipSamples(t)
	heard, err := tok.Listen(clips[0], 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(heard.Codes) != heard.Frames || len(heard.Units) >= heard.Frames || len(heard.Units) == 0 {
		t.Fatalf("%d units, %d codes, %d frames", len(heard.Units), len(heard.Codes), heard.Frames)
	}
	for i := 1; i < len(heard.Units); i++ {
		if heard.Units[i] == heard.Units[i-1] {
			t.Fatal("a run survived")
		}
	}
	pcm := make([]byte, 2*len(clips[0]))
	for i, s := range clips[0] {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(math.Round(s*32768))))
	}
	viaWav, err := tok.ListenWAV(WavBytes(pcm, 16000))
	if err != nil || strings.Join(viaWav.Units, " ") != heard.Text() {
		t.Errorf("through a WAV: %v %q", err, viaWav.Text())
	}
	if text, err := tok.Text("q1 q1 q2   q3\nq3"); err != nil || text != "q1 q2 q3" {
		t.Errorf("text form: %q %v", text, err)
	}
	if _, err := tok.Text("q1 hello"); err == nil {
		t.Error("junk passed as units")
	}
	// frame for frame, the sounds heard again are mostly the same units
	voc, _ := tok.Vocoder(1.0, Pitch)
	var out []byte
	for _, code := range heard.Codes {
		frame := make([]float64, 40)
		for b := range frame {
			frame[b] = book.Centroids[code][b] + book.Mean[b]
		}
		out = append(out, voc.FeedFrame(frame)...)
	}
	out = append(out, voc.End()...)
	if want := (heard.Frames-1)*160 + 400; len(out)/2 != want {
		t.Fatalf("%d samples, want %d", len(out)/2, want)
	}
	again, _ := tok.Listen(PCMSamples(out), 0)
	agree := 0
	for i := range heard.Codes {
		if i < len(again.Codes) && again.Codes[i] == heard.Codes[i] {
			agree++
		}
	}
	if share := float64(agree) / float64(heard.Frames); share < 0.6 {
		t.Errorf("only %.2f of the frames came back as the same unit", share)
	}
	peak := 0
	for i := 0; i+1 < len(out); i += 2 {
		if v := int(int16(binary.LittleEndian.Uint16(out[i:]))); v > peak {
			peak = v
		}
	}
	if peak < 3000 || peak >= 32767 {
		t.Errorf("peak %d", peak)
	}
	// the units spoken back: streaming is the same audio as one go
	units := heard.Units
	whole, err := tok.Synthesize(units, 0, 1.0, Pitch)
	if err != nil {
		t.Fatal(err)
	}
	voc, _ = tok.Vocoder(1.0, Pitch)
	var streamed []byte
	for _, u := range units {
		chunk, _ := voc.Feed(u)
		streamed = append(streamed, chunk...)
	}
	streamed = append(streamed, voc.End()...)
	if string(streamed) != string(whole) {
		t.Error("streaming and one go differ")
	}
	held := 0
	for _, u := range units {
		held += book.Hold(book.Index(u))
	}
	if len(whole)/2 != (held-1)*160+400 {
		t.Errorf("%d samples for %d held frames", len(whole)/2, held)
	}
	polished, err := tok.Synthesize(units, 2, 0.5, 200)
	if err != nil || len(polished) != len(whole) {
		t.Errorf("polished: %v, %d bytes", err, len(polished))
	}
	if _, err := tok.Synthesize([]string{"q1", "nope"}, 0, 1, Pitch); err == nil {
		t.Error("a non-unit was spoken")
	}
	if _, err := NewVocoder(book, 1, 0); err == nil {
		t.Error("a pitch of 0 was accepted")
	}
}

func TestTheBundledCodebook(t *testing.T) {
	book, err := DefaultCodebook()
	if err != nil {
		t.Fatal(err)
	}
	if book.K() != 64 || book.Analysis != DefaultAnalysis() || !strings.Contains(book.Note, "synthesizer") {
		t.Fatalf("k %d, %+v, %q", book.K(), book.Analysis, book.Note)
	}
	tok, _ := NewAcousticTokenizer(nil, true)
	heard, err := tok.Listen(spokenSamples(t, acousticClips[1], 120), 0)
	if err != nil {
		t.Fatal(err)
	}
	distinct := map[string]bool{}
	for _, u := range heard.Units {
		distinct[u] = true
		if !tok.IsUnit(u) {
			t.Fatalf("%q is not a unit", u)
		}
	}
	if len(distinct) <= 10 || len(heard.Units) >= heard.Frames {
		t.Errorf("%d distinct of %d units over %d frames", len(distinct), len(heard.Units), heard.Frames)
	}
	other, _ := tok.Hear(spokenSamples(t, acousticClips[1], 175), 0)
	shared := 0
	seen := map[string]bool{}
	for _, u := range other {
		seen[u] = true
	}
	for u := range distinct {
		if seen[u] {
			shared++
		}
	}
	if share := float64(shared) / float64(len(distinct)); share < 0.5 {
		t.Errorf("another voice shares only %.2f of the units", share)
	}
}
