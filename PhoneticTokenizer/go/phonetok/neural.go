package phonetok

// The neural vocoder for the acoustic units (phonetok/neural.py, ported): the units heard back through a
// filter learned from recordings, over the vocoder's own excitation.
//
// A pulse train at the voice's pitch and the vocoder's deterministic noise run on under the utterance, and a
// small network - an embedding of the unit codes, residual blocks of dilated convolutions at the frame rate,
// and a head - writes for every frame and every bin of the transform how loud each of the two should be
// there.  The frames are overlap-added exactly as the vocoder's own are, so this port and the others hear
// the same samples from the same weights.  Inference only: training is the Python module's (numpy), and it
// writes the weights file this reads (<codebook stem>.vocoder.json, tensors as base64 little-endian float32).

import (
	_ "embed"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

//go:embed data/acoustic.vocoder.json
var vocoderText string

// VocoderSuffix is what a codebook's vocoder file ends in: codebook.tsv -> codebook.vocoder.json.
const VocoderSuffix = ".vocoder.json"

// NeuralSlope is the slope of the leaky rectifier below zero.
const NeuralSlope = 0.1

// NeuralClip is the range a predicted log gain is held to.
var NeuralClip = [2]float64{-24.0, 8.0}

// NeuralHeadroom is what Synthesize scales the waveform by at gain 1: three decibels of headroom for the
// peaks the dispersed excitation still has.
const NeuralHeadroom = 0.7

// VocoderSpec is the architecture: what the file says.
type VocoderSpec struct {
	Units     int
	Bins      int
	Channels  int
	Kernel    int
	Dilations []int
}

// Validate checks the architecture.
func (s VocoderSpec) Validate() error {
	if s.Units < 1 || s.Bins < 2 || s.Channels < 1 {
		return fmt.Errorf("impossible vocoder: %+v", s)
	}
	if s.Kernel < 1 || s.Kernel%2 == 0 {
		return fmt.Errorf("the kernel must be odd, got %d", s.Kernel)
	}
	if len(s.Dilations) == 0 {
		return fmt.Errorf("the dilations must be at least 1, got none")
	}
	for _, d := range s.Dilations {
		if d < 1 {
			return fmt.Errorf("the dilations must be at least 1, got %v", s.Dilations)
		}
	}
	return nil
}

// Context is the frames on either side that can reach a frame's output: the receptive field's radius.
func (s VocoderSpec) Context() int {
	total := 0
	for _, d := range s.Dilations {
		total += (s.Kernel - 1) / 2 * d
	}
	return total
}

// Parameters counts the weights.
func (s VocoderSpec) Parameters() int {
	c, k := s.Channels, s.Kernel
	return s.Units*c + len(s.Dilations)*(c*c*k+c+c*c+c) + 2*s.Bins*c + 2*s.Bins + s.Units*2*s.Bins
}

type vocoderBlock struct {
	taps []float64 // the convolution's weights per tap: [tap][out][in]
	b1   []float64
	w2   []float64 // [out][in]
	b2   []float64
}

// UnitVocoder is codes to samples through the learned filter over the vocoder's excitation.
type UnitVocoder struct {
	Analysis Analysis
	Spec     VocoderSpec
	embed    []float64 // [units][channels]
	blocks   []vocoderBlock
	headW    []float64 // [2 * bins][channels]: the pulse train's log gains, then the noise's
	headB    []float64
	template []float64 // [units][2 * bins]: every unit's own log gains, added to the head's output
	Note     string
	Checksum float64
	Trained  any
}

type vocoderTensor struct {
	Shape []int  `json:"shape"`
	F32   string `json:"f32"`
}

type vocoderBlockFile struct {
	W1 vocoderTensor `json:"w1"`
	B1 vocoderTensor `json:"b1"`
	W2 vocoderTensor `json:"w2"`
	B2 vocoderTensor `json:"b2"`
}

type vocoderFile struct {
	Phonetok string  `json:"phonetok"`
	Version  *int    `json:"version"`
	Model    string  `json:"model"`
	Note     string  `json:"note"`
	Units    int     `json:"units"`
	Checksum float64 `json:"checksum"`
	Analysis struct {
		Rate  int `json:"rate"`
		Frame int `json:"frame"`
		Hop   int `json:"hop"`
		FFT   int `json:"fft"`
	} `json:"analysis"`
	Channels  int   `json:"channels"`
	Kernel    int   `json:"kernel"`
	Dilations []int `json:"dilations"`
	Trained   any   `json:"trained"`
	Weights   struct {
		Embed  vocoderTensor      `json:"embed"`
		Blocks []vocoderBlockFile `json:"blocks"`
		Head   struct {
			W vocoderTensor `json:"w"`
			B vocoderTensor `json:"b"`
		} `json:"head"`
		Template vocoderTensor `json:"template"`
	} `json:"weights"`
}

func (t vocoderTensor) values(shape ...int) ([]float64, error) {
	if len(t.Shape) != len(shape) {
		return nil, fmt.Errorf("a tensor of shape %v where %v was expected", t.Shape, shape)
	}
	count := 1
	for i, s := range shape {
		if t.Shape[i] != s {
			return nil, fmt.Errorf("a tensor of shape %v where %v was expected", t.Shape, shape)
		}
		count *= s
	}
	data, err := base64.StdEncoding.DecodeString(t.F32)
	if err != nil {
		return nil, fmt.Errorf("a tensor that is not base64: %w", err)
	}
	if len(data) != 4*count {
		return nil, fmt.Errorf("a tensor of %d values packed as %d bytes", count, len(data))
	}
	out := make([]float64, count)
	for i := range out {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:])))
	}
	return out, nil
}

// ParseVocoder reads a vocoder file's text.
func ParseVocoder(text string) (*UnitVocoder, error) {
	var doc vocoderFile
	if err := json.Unmarshal([]byte(text), &doc); err != nil {
		return nil, fmt.Errorf("not a phonetok acoustic vocoder file: %w", err)
	}
	if doc.Phonetok != "acoustic vocoder" {
		return nil, fmt.Errorf("not a phonetok acoustic vocoder file")
	}
	model := doc.Model
	if model == "" {
		model = "source-filter"
	}
	version := 1
	if doc.Version != nil {
		version = *doc.Version
	}
	if model != "source-filter" || version != 1 {
		return nil, fmt.Errorf("a vocoder of a kind this version does not know: %s v%d", model, version)
	}
	analysis := DefaultAnalysis()
	analysis.Rate, analysis.Frame, analysis.Hop, analysis.FFT = doc.Analysis.Rate, doc.Analysis.Frame, doc.Analysis.Hop, doc.Analysis.FFT
	spec := VocoderSpec{Units: doc.Units, Bins: doc.Analysis.FFT/2 + 1, Channels: doc.Channels, Kernel: doc.Kernel,
		Dilations: append([]int(nil), doc.Dilations...)}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	c, k := spec.Channels, spec.Kernel
	embed, err := doc.Weights.Embed.values(spec.Units, c)
	if err != nil {
		return nil, err
	}
	if len(doc.Weights.Blocks) != len(spec.Dilations) {
		return nil, fmt.Errorf("%d blocks of weights for %d dilations", len(doc.Weights.Blocks), len(spec.Dilations))
	}
	blocks := make([]vocoderBlock, 0, len(doc.Weights.Blocks))
	for _, b := range doc.Weights.Blocks {
		w1, err := b.W1.values(c, c, k) // [out][in][tap]
		if err != nil {
			return nil, err
		}
		taps := make([]float64, k*c*c)
		for o := 0; o < c; o++ {
			for i := 0; i < c; i++ {
				for j := 0; j < k; j++ {
					taps[(j*c+o)*c+i] = w1[(o*c+i)*k+j]
				}
			}
		}
		b1, err := b.B1.values(c)
		if err != nil {
			return nil, err
		}
		w2, err := b.W2.values(c, c)
		if err != nil {
			return nil, err
		}
		b2, err := b.B2.values(c)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, vocoderBlock{taps: taps, b1: b1, w2: w2, b2: b2})
	}
	headW, err := doc.Weights.Head.W.values(2*spec.Bins, c)
	if err != nil {
		return nil, err
	}
	headB, err := doc.Weights.Head.B.values(2 * spec.Bins)
	if err != nil {
		return nil, err
	}
	template, err := doc.Weights.Template.values(spec.Units, 2*spec.Bins)
	if err != nil {
		return nil, err
	}
	return &UnitVocoder{Analysis: analysis, Spec: spec, embed: embed, blocks: blocks, headW: headW, headB: headB,
		template: template, Note: doc.Note, Checksum: doc.Checksum, Trained: doc.Trained}, nil
}

// LoadVocoder reads a vocoder file.
func LoadVocoder(path string) (*UnitVocoder, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	v, err := ParseVocoder(string(data))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return v, nil
}

// DefaultVocoder is the vocoder trained for the bundled codebook, read afresh.
func DefaultVocoder() (*UnitVocoder, error) {
	return ParseVocoder(vocoderText)
}

// CodebookChecksum is a cheap fingerprint of a codebook, kept in its vocoder so the two are not mixed up.
func CodebookChecksum(book *Codebook) float64 {
	total := 0.0
	for _, c := range book.Centroids {
		for _, v := range c {
			total += v
		}
	}
	for _, v := range book.Mean {
		total += v
	}
	return math.Round((total+float64(book.K()))*1e6) / 1e6
}

// Matches says whether this vocoder was trained for book: the same units, analysis and fingerprint.
func (v *UnitVocoder) Matches(book *Codebook) bool {
	a, b := v.Analysis, book.Analysis
	return v.Spec.Units == book.K() && a.Rate == b.Rate && a.Frame == b.Frame && a.Hop == b.Hop && a.FFT == b.FFT &&
		math.Abs(v.Checksum-CodebookChecksum(book)) < 1e-3
}

// Describe is what the vocoder is, as the CLI prints it.
func (v *UnitVocoder) Describe() map[string]any {
	s, a := v.Spec, v.Analysis
	return map[string]any{
		"units": s.Units, "channels": s.Channels, "kernel": s.Kernel, "dilations": append([]int{}, s.Dilations...),
		"bins": s.Bins, "parameters": s.Parameters(), "context_frames": s.Context(),
		"context_seconds": float64(s.Context()*a.Hop) / float64(a.Rate),
		"analysis":        map[string]any{"rate": a.Rate, "frame": a.Frame, "hop": a.Hop, "fft": a.FFT},
		"checksum":        v.Checksum, "note": v.Note, "trained": v.Trained,
	}
}

// Filters is the log gains of every frame: the pulse train's per bin, and the noise's (each [frames][bins]).
func (v *UnitVocoder) Filters(codes []int) (lp, ln [][]float64, err error) {
	s := v.Spec
	c, k := s.Channels, s.Kernel
	centre := (k - 1) / 2
	count := len(codes)
	for _, code := range codes {
		if code < 0 || code >= s.Units {
			return nil, nil, fmt.Errorf("code %d is not one of this vocoder's %d units", code, s.Units)
		}
	}
	x := make([]float64, 0, count*c)
	for _, code := range codes {
		x = append(x, v.embed[code*c:(code+1)*c]...)
	}
	for bi, block := range v.blocks {
		d := s.Dilations[bi]
		pre := make([]float64, count*c)
		for t := 0; t < count; t++ {
			copy(pre[t*c:(t+1)*c], block.b1)
		}
		for j := 0; j < k; j++ {
			shift := (j - centre) * d
			tap := block.taps[j*c*c : (j+1)*c*c]
			for t := 0; t < count; t++ {
				src := t + shift
				if src < 0 || src >= count {
					continue
				}
				xs := x[src*c : (src+1)*c]
				acc := pre[t*c : (t+1)*c]
				for o := 0; o < c; o++ {
					row := tap[o*c : (o+1)*c]
					total := 0.0
					for i := 0; i < c; i++ {
						total += row[i] * xs[i]
					}
					acc[o] += total
				}
			}
		}
		h := make([]float64, c)
		for t := 0; t < count; t++ {
			for i, val := range pre[t*c : (t+1)*c] {
				if val > 0 {
					h[i] = val
				} else {
					h[i] = NeuralSlope * val
				}
			}
			xt := x[t*c : (t+1)*c]
			for o := 0; o < c; o++ {
				row := block.w2[o*c : (o+1)*c]
				total := block.b2[o]
				for i := 0; i < c; i++ {
					total += row[i] * h[i]
				}
				xt[o] += total
			}
		}
	}
	bins := s.Bins
	lo, hi := NeuralClip[0], NeuralClip[1]
	lp = make([][]float64, count)
	ln = make([][]float64, count)
	for t := 0; t < count; t++ {
		xt := x[t*c : (t+1)*c]
		template := v.template[codes[t]*2*bins : (codes[t]+1)*2*bins]
		z := make([]float64, 2*bins)
		for o := range z {
			row := v.headW[o*c : (o+1)*c]
			total := template[o] + v.headB[o]
			for i := 0; i < c; i++ {
				total += row[i] * xt[i]
			}
			z[o] = math.Min(hi, math.Max(lo, total))
		}
		lp[t] = z[:bins]
		ln[t] = z[bins:]
	}
	return lp, ln, nil
}

// Spectra is the full (Hermitian) spectrum of every frame, as the inverse transform takes it.
func (v *UnitVocoder) Spectra(codes []int, pitch float64) ([]frameSpectrum, error) {
	lp, ln, err := v.Filters(codes)
	if err != nil {
		return nil, err
	}
	pulses, noises, err := ExcitationSpectra(len(codes), v.Analysis, pitch)
	if err != nil {
		return nil, err
	}
	n := v.Analysis.FFT
	half := n / 2
	out := make([]frameSpectrum, 0, len(codes))
	for t := range codes {
		re := make([]float64, n)
		im := make([]float64, n)
		for k := 0; k <= half; k++ {
			gp := math.Exp(lp[t][k])
			gn := math.Exp(ln[t][k])
			re[k] = pulses[t].re[k]*gp + noises[t].re[k]*gn
			im[k] = pulses[t].im[k]*gp + noises[t].im[k]*gn
		}
		im[0] = 0
		im[half] = 0
		for k := 1; k < half; k++ {
			re[n-k] = re[k]
			im[n-k] = -im[k]
		}
		out = append(out, frameSpectrum{re: re, im: im})
	}
	return out, nil
}

// Samples is the waveform of a run of codes, in [-1, 1].
func (v *UnitVocoder) Samples(codes []int, pitch float64) ([]float64, error) {
	spectra, err := v.Spectra(codes, pitch)
	if err != nil {
		return nil, err
	}
	return istft(spectra, v.Analysis), nil
}

// Synthesize is 16-bit PCM at the analysis' rate: the waveform at gain times NeuralHeadroom.
func (v *UnitVocoder) Synthesize(codes []int, gain, pitch float64) ([]byte, error) {
	samples, err := v.Samples(codes, pitch)
	if err != nil {
		return nil, err
	}
	return packPCM(samples, gain*NeuralHeadroom), nil
}

// CodesOf is the code of every frame a run of units stands for: each unit held for its typical run.
func CodesOf(units []string, book *Codebook) ([]int, error) {
	var codes []int
	for _, unit := range units {
		index := book.Index(unit)
		if index < 0 {
			return nil, fmt.Errorf("not a unit of this codebook: %q", unit)
		}
		for i := 0; i < book.Hold(index); i++ {
			codes = append(codes, index)
		}
	}
	return codes, nil
}

// VocoderPathFor is where a codebook's vocoder lives: codebook.tsv -> codebook.vocoder.json.
func VocoderPathFor(codebookPath string) string {
	if strings.EqualFold(filepath.Ext(codebookPath), ".tsv") {
		return strings.TrimSuffix(codebookPath, filepath.Ext(codebookPath)) + VocoderSuffix
	}
	return codebookPath + VocoderSuffix
}

// FindVocoder is the vocoder that belongs to a codebook, when there is one: path names the file outright;
// else $PHONETOK_VOCODER; else the file beside the codebook's own (source), or the bundled vocoder for a codebook
// with no source, when it matches.  A file that claims to belong to the codebook but does not match it is an
// error; none at all is nil.
func FindVocoder(book *Codebook, source, path string) (*UnitVocoder, error) {
	named := path
	if named == "" {
		named = os.Getenv("PHONETOK_VOCODER")
	}
	check := func(v *UnitVocoder, name string) (*UnitVocoder, error) {
		if !v.Matches(book) {
			return nil, fmt.Errorf("%s was not trained for this codebook", name)
		}
		return v, nil
	}
	if named != "" {
		if info, err := os.Stat(named); err != nil || info.IsDir() {
			return nil, fmt.Errorf("vocoder file not found: %s", named)
		}
		v, err := LoadVocoder(named)
		if err != nil {
			return nil, err
		}
		return check(v, named)
	}
	if source != "" {
		beside := VocoderPathFor(source)
		if info, err := os.Stat(beside); err != nil || info.IsDir() {
			return nil, nil
		}
		v, err := LoadVocoder(beside)
		if err != nil {
			return nil, err
		}
		return check(v, beside)
	}
	// a codebook that came from nowhere may be the bundled one: the bundled vocoder is tried, and merely not
	// the codebook's when it does not match
	v, err := DefaultVocoder()
	if err != nil || !v.Matches(book) {
		return nil, nil
	}
	return v, nil
}

// -- the excitation: the source every port makes the same --------------------------------------

// Chirp is how many samples each pulse is spread over: a Hann-windowed linear chirp of unit energy sweeping
// from zero to the Nyquist frequency, so the waveform peaks where the recordings' do instead of three times
// higher (an impulse's harmonics are all in phase at the pulse).
const Chirp = 64

// ChirpKernel is the chirp every pulse becomes, of unit energy.
func ChirpKernel() []float64 {
	window := Hann(Chirp)
	kernel := make([]float64, Chirp)
	norm := 0.0
	for t := range kernel {
		kernel[t] = math.Cos(math.Pi*float64(t*t)/(2.0*Chirp)) * window[t]
		norm += kernel[t] * kernel[t]
	}
	norm = math.Sqrt(norm)
	for t := range kernel {
		kernel[t] /= norm
	}
	return kernel
}

// Disperse is the pulse train with every pulse spread into the chirp (pulses taken in order, so every port
// sums alike).
func Disperse(pulses []float64) []float64 {
	kernel := ChirpKernel()
	out := make([]float64, len(pulses))
	for p, v := range pulses {
		if v == 0 {
			continue
		}
		for t, k := range kernel {
			if p+t >= len(out) {
				break
			}
			out[p+t] += v * k
		}
	}
	return out
}

// PulseTrain is unit impulses at pitch from phase zero: exactly the vocoder's pulse train (see Disperse).
func PulseTrain(length int, pitch float64, rate int) ([]float64, error) {
	if pitch <= 0 {
		return nil, fmt.Errorf("pitch must be positive, got %g", pitch)
	}
	step := pitch / float64(rate)
	phase := 0.0
	out := make([]float64, length)
	for n := range out {
		phase += step
		if phase >= 1 {
			phase -= 1
			out[n] = 1
		}
	}
	return out, nil
}

// NoiseTrain is the first length values of the vocoder's noise (xorshift32 from its fixed seed), in [-1, 1).
func NoiseTrain(length int) []float64 {
	noise := newNoise()
	out := make([]float64, length)
	for i := range out {
		out[i] = noise.next()
	}
	return out
}

// ExcitationLength is the samples count frames cover.
func ExcitationLength(count int, a Analysis) int {
	if count <= 0 {
		return 0
	}
	return (count-1)*a.Hop + a.Frame
}

// ExcitationSpectra is the spectra of the dispersed pulse train's frames and of the noise's, count of each
// (full transforms; the first fft/2 + 1 bins are the half the network's gains apply to).
func ExcitationSpectra(count int, a Analysis, pitch float64) (pulses, noises []frameSpectrum, err error) {
	length := ExcitationLength(count, a)
	train, err := PulseTrain(length, pitch, a.Rate)
	if err != nil {
		return nil, nil, err
	}
	return stft(Disperse(train), a, count), stft(NoiseTrain(length), a, count), nil
}
