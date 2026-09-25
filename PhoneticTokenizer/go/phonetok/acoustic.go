package phonetok

// Acoustic units (the port of acoustic.py): speech as the sounds a codebook
// learned from it, with nothing written down.  Audio at 16 kHz is cut into
// 25 ms frames every 10 ms, each frame is 40 log-mel energies with the
// utterance's mean taken out, every frame goes to its nearest codebook entry
// and a run of one entry is one unit - a token like q17.  The codebook is
// learned by k-means from any recordings (Learn), it is a data file that
// carries its own analysis settings (the bundled one is data/acoustic.tsv),
// and the Vocoder turns units back into a waveform frame by frame.
//
// Every operation here is the Python module's, in the same order, so a unit
// is the same unit in every port.

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

//go:embed data/acoustic.tsv
var acousticText string

// AcousticRate is the sample rate the analysis runs at.
const AcousticRate = 16000

// UnitPrefix starts every unit token: q (for quantised) and the codebook index.
const UnitPrefix = "q"

// LogFloor is added to every band's energy before the log.
const LogFloor = 1e-10

// Pitch is the vocoder's pulse train, in Hz.
const Pitch = 120.0

const twoPi = 2.0 * math.Pi

// Analysis is how audio becomes frames: the settings a codebook carries.
type Analysis struct {
	Rate        int
	Frame       int
	Hop         int
	FFT         int
	Bands       int
	Fmin        float64
	Fmax        float64
	Preemphasis float64
	Normalize   bool
}

// DefaultAnalysis is 25 ms frames every 10 ms at 16 kHz, 40 bands between 20 Hz and 8 kHz, normalised.
func DefaultAnalysis() Analysis {
	return Analysis{Rate: AcousticRate, Frame: 400, Hop: 160, FFT: 512, Bands: 40, Fmin: 20.0, Fmax: 8000.0,
		Preemphasis: 0.97, Normalize: true}
}

// Validate refuses settings no analysis can run with.
func (a Analysis) Validate() error {
	if a.Rate < 1 || a.Frame < 2 || a.Hop < 1 || a.Bands < 1 {
		return fmt.Errorf("impossible analysis: %+v", a)
	}
	if a.FFT < a.Frame || a.FFT&(a.FFT-1) != 0 {
		return fmt.Errorf("fft must be a power of two of at least the frame (%d), got %d", a.Frame, a.FFT)
	}
	if !(0.0 <= a.Fmin && a.Fmin < a.Fmax && a.Fmax <= float64(a.Rate)/2.0) {
		return fmt.Errorf("the bands must lie in 0 .. %g Hz, got %g .. %g", float64(a.Rate)/2.0, a.Fmin, a.Fmax)
	}
	return nil
}

// -- audio files --------------------------------------------------------------------------

// ReadWAV is the samples of a WAV file in [-1, 1], mono (channels averaged), and its rate.
// PCM of 8, 16, 24 and 32 bits and IEEE float of 32 and 64 bits are read.
func ReadWAV(data []byte) ([]float64, int, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("not a WAV file (no RIFF/WAVE header)")
	}
	var tag, channels, rate, bits int
	haveFmt := false
	var body []byte
	pos := 12
	for pos+8 <= len(data) {
		chunk := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		end := pos + 8 + size
		if end > len(data) {
			end = len(data)
		}
		payload := data[pos+8 : end]
		if chunk == "fmt " && len(payload) >= 16 {
			tag = int(binary.LittleEndian.Uint16(payload[0:2]))
			channels = int(binary.LittleEndian.Uint16(payload[2:4]))
			rate = int(binary.LittleEndian.Uint32(payload[4:8]))
			bits = int(binary.LittleEndian.Uint16(payload[14:16]))
			if tag == 0xFFFE && len(payload) >= 26 {
				tag = int(binary.LittleEndian.Uint16(payload[24:26]))
			}
			haveFmt = true
		} else if chunk == "data" {
			body = payload
		}
		pos += 8 + size + (size & 1)
	}
	if !haveFmt || body == nil {
		return nil, 0, errors.New("not a WAV file (no fmt or data chunk)")
	}
	if channels < 1 || rate < 1 {
		return nil, 0, fmt.Errorf("WAV with %d channels at %d Hz", channels, rate)
	}
	var values []float64
	switch {
	case tag == 1 && bits == 8:
		values = make([]float64, len(body))
		for i, b := range body {
			values[i] = (float64(b) - 128) / 128.0
		}
	case tag == 1 && bits == 16:
		n := len(body) / 2
		values = make([]float64, n)
		for i := 0; i < n; i++ {
			values[i] = float64(int16(binary.LittleEndian.Uint16(body[2*i:]))) / 32768.0
		}
	case tag == 1 && bits == 24:
		n := len(body) / 3
		values = make([]float64, n)
		for i := 0; i < n; i++ {
			v := int(body[3*i]) | int(body[3*i+1])<<8 | int(body[3*i+2])<<16
			if v&0x800000 != 0 {
				v -= 0x1000000
			}
			values[i] = float64(v) / 8388608.0
		}
	case tag == 1 && bits == 32:
		n := len(body) / 4
		values = make([]float64, n)
		for i := 0; i < n; i++ {
			values[i] = float64(int32(binary.LittleEndian.Uint32(body[4*i:]))) / 2147483648.0
		}
	case tag == 3 && bits == 32:
		n := len(body) / 4
		values = make([]float64, n)
		for i := 0; i < n; i++ {
			values[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:])))
		}
	case tag == 3 && bits == 64:
		n := len(body) / 8
		values = make([]float64, n)
		for i := 0; i < n; i++ {
			values[i] = math.Float64frombits(binary.LittleEndian.Uint64(body[8*i:]))
		}
	case tag == 1:
		return nil, 0, fmt.Errorf("%d-bit PCM WAV is not supported", bits)
	case tag == 3:
		return nil, 0, fmt.Errorf("%d-bit float WAV is not supported", bits)
	default:
		return nil, 0, fmt.Errorf("WAV format tag %d is not supported (PCM and IEEE float are)", tag)
	}
	if channels > 1 {
		frames := len(values) / channels
		mono := make([]float64, frames)
		for i := 0; i < frames; i++ {
			s := 0.0
			for c := 0; c < channels; c++ {
				s += values[i*channels+c]
			}
			mono[i] = s / float64(channels)
		}
		values = mono
	}
	return values, rate, nil
}

// LoadWAV reads a WAV file.
func LoadWAV(path string) ([]float64, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	return ReadWAV(data)
}

// PCMSamples is 16-bit mono PCM (what the synthesizer makes) as samples in [-1, 1].
func PCMSamples(pcm []byte) []float64 {
	n := len(pcm) / 2
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = float64(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768.0
	}
	return out
}

// Resample brings samples at src Hz to dst Hz by a windowed-sinc interpolation.
func Resample(samples []float64, src, dst int) ([]float64, error) {
	if src == dst {
		return append([]float64(nil), samples...), nil
	}
	if src < 1 || dst < 1 {
		return nil, fmt.Errorf("cannot resample %d Hz to %d Hz", src, dst)
	}
	const taps = 16
	n := len(samples)
	count := n * dst / src
	ratio := float64(src) / float64(dst)
	cutoff := math.Min(1.0, float64(dst)/float64(src)) * 0.95
	out := make([]float64, 0, count)
	for j := 0; j < count; j++ {
		center := float64(j) * ratio
		base := int(math.Floor(center))
		acc := 0.0
		norm := 0.0
		for i := base - taps + 1; i <= base+taps; i++ {
			if i < 0 || i >= n {
				continue
			}
			x := float64(i) - center
			var w float64
			if x == 0.0 {
				w = 1.0
			} else {
				w = math.Sin(math.Pi*cutoff*x) / (math.Pi * cutoff * x) * (0.5 + 0.5*math.Cos(math.Pi*x/taps))
			}
			acc += samples[i] * w
			norm += w
		}
		if norm != 0.0 {
			out = append(out, acc/norm)
		} else {
			out = append(out, 0.0)
		}
	}
	return out, nil
}

// -- the analysis -------------------------------------------------------------------------

// Hann is the periodic Hann window of n samples.
func Hann(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(twoPi*float64(i)/float64(n))
	}
	return w
}

// FFT transforms re + i im in place (a length that is a power of two); inverse
// transforms the other way and divides by the length.  Radix-2, the twiddle
// advanced by multiplication, exactly as the Python module does it.
func FFT(re, im []float64, inverse bool) {
	n := len(re)
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j |= bit
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for length := 2; length <= n; length <<= 1 {
		ang := sign * twoPi / float64(length)
		wr := math.Cos(ang)
		wi := math.Sin(ang)
		half := length >> 1
		for start := 0; start < n; start += length {
			cr := 1.0
			ci := 0.0
			for k := start; k < start+half; k++ {
				m := k + half
				tr := re[m]*cr - im[m]*ci
				ti := re[m]*ci + im[m]*cr
				re[m] = re[k] - tr
				im[m] = im[k] - ti
				re[k] = re[k] + tr
				im[k] = im[k] + ti
				ncr := cr*wr - ci*wi
				nci := cr*wi + ci*wr
				cr = ncr
				ci = nci
			}
		}
	}
	if inverse {
		fn := float64(n)
		for i := 0; i < n; i++ {
			re[i] = re[i] / fn
			im[i] = im[i] / fn
		}
	}
}

// Mel is the HTK mel scale.
func Mel(hz float64) float64 { return 2595.0 * math.Log10(1.0+hz/700.0) }

// MelToHz is its inverse.
func MelToHz(m float64) float64 { return 700.0 * (math.Pow(10.0, m/2595.0) - 1.0) }

type melFilter struct {
	first   int
	weights []float64
}

var (
	filtersMu    sync.Mutex
	filtersCache = map[Analysis][]melFilter{}
)

// melFilters is the triangular filters: per band, the first bin it touches and its weight on each bin from there.
func melFilters(a Analysis) []melFilter {
	filtersMu.Lock()
	defer filtersMu.Unlock()
	if cached, ok := filtersCache[a]; ok {
		return cached
	}
	lo, hi := Mel(a.Fmin), Mel(a.Fmax)
	edges := make([]float64, a.Bands+2)
	for i := range edges {
		edges[i] = MelToHz(lo + (hi-lo)*float64(i)/float64(a.Bands+1))
	}
	half := a.FFT / 2
	filters := make([]melFilter, 0, a.Bands)
	for b := 0; b < a.Bands; b++ {
		left, center, right := edges[b], edges[b+1], edges[b+2]
		first := -1
		var weights []float64
		for k := 0; k <= half; k++ {
			f := float64(k*a.Rate) / float64(a.FFT)
			var w float64
			switch {
			case f <= left || f >= right:
				w = 0.0
			case f <= center:
				w = (f - left) / (center - left)
			default:
				w = (right - f) / (right - center)
			}
			if w > 0.0 {
				if first < 0 {
					first = k
				}
				weights = append(weights, w)
			} else if first >= 0 {
				break
			}
		}
		if first < 0 {
			first = int(center*float64(a.FFT)/float64(a.Rate) + 0.5)
			weights = []float64{1.0}
		}
		filters = append(filters, melFilter{first, weights})
	}
	filtersCache[a] = filters
	return filters
}

// Preemphasize is y[n] = x[n] - p x[n-1].
func Preemphasize(samples []float64, p float64) []float64 {
	out := append([]float64(nil), samples...)
	for i := len(out) - 1; i > 0; i-- {
		out[i] = out[i] - p*out[i-1]
	}
	return out
}

// Frames is the log-mel frames of samples (already at a.Rate); at least one frame, zero-padded if need be.
// normalize subtracts every band's mean over the utterance (the learner reads the raw frames once first).
func Frames(samples []float64, a Analysis, normalize bool) ([][]float64, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	var y []float64
	if a.Preemphasis != 0 {
		y = Preemphasize(samples, a.Preemphasis)
	} else {
		y = append([]float64(nil), samples...)
	}
	for len(y) < a.Frame {
		y = append(y, 0.0)
	}
	count := (len(y)-a.Frame)/a.Hop + 1
	window := Hann(a.Frame)
	filters := melFilters(a)
	half := a.FFT / 2
	out := make([][]float64, 0, count)
	re := make([]float64, a.FFT)
	im := make([]float64, a.FFT)
	power := make([]float64, half+1)
	for t := 0; t < count; t++ {
		start := t * a.Hop
		for i := 0; i < a.Frame; i++ {
			re[i] = y[start+i] * window[i]
		}
		for i := a.Frame; i < a.FFT; i++ {
			re[i] = 0.0
		}
		for i := range im {
			im[i] = 0.0
		}
		FFT(re, im, false)
		for k := 0; k <= half; k++ {
			power[k] = re[k]*re[k] + im[k]*im[k]
		}
		frame := make([]float64, 0, a.Bands)
		for _, f := range filters {
			e := 0.0
			for i, w := range f.weights {
				e += w * power[f.first+i]
			}
			frame = append(frame, math.Log(e+LogFloor))
		}
		out = append(out, frame)
	}
	if normalize && len(out) > 0 {
		means := BandMeans(out)
		for _, frame := range out {
			for b := 0; b < a.Bands; b++ {
				frame[b] = frame[b] - means[b]
			}
		}
	}
	return out, nil
}

// FramesOf is Frames with the analysis' own normalisation setting.
func FramesOf(samples []float64, a Analysis) ([][]float64, error) {
	return Frames(samples, a, a.Normalize)
}

// BandMeans is the mean of every dimension over the frames.
func BandMeans(frames [][]float64) []float64 {
	if len(frames) == 0 {
		return nil
	}
	dims := len(frames[0])
	sums := make([]float64, dims)
	for _, frame := range frames {
		for b := 0; b < dims; b++ {
			sums[b] += frame[b]
		}
	}
	for b := range sums {
		sums[b] = sums[b] / float64(len(frames))
	}
	return sums
}

// Dist2 is the squared Euclidean distance, summed in order.
func Dist2(x, y []float64) float64 {
	d := 0.0
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		diff := x[i] - y[i]
		d += diff * diff
	}
	return d
}

// -- the codebook -------------------------------------------------------------------------

// Codebook is what was learned: the centroids and what the training frames said about them.
type Codebook struct {
	Analysis  Analysis
	Centroids [][]float64
	// Runs is the mean length in frames of a run of each unit in the training audio.
	Runs []float64
	// Counts is the frames each unit claimed in the training audio.
	Counts []int
	// Mean is the mean log-mel of the training frames before normalisation: decoding adds it back.
	Mean    []float64
	Seed    int64
	Note    string
	Inertia float64
}

// K is the number of units.
func (c *Codebook) K() int { return len(c.Centroids) }

// Validate refuses a codebook whose tables disagree.
func (c *Codebook) Validate() error {
	if err := c.Analysis.Validate(); err != nil {
		return err
	}
	if len(c.Centroids) == 0 {
		return errors.New("an empty codebook")
	}
	for _, cen := range c.Centroids {
		if len(cen) != c.Analysis.Bands {
			return fmt.Errorf("a centroid of %d values in a codebook of %d bands", len(cen), c.Analysis.Bands)
		}
	}
	if len(c.Runs) != c.K() || len(c.Counts) != c.K() || len(c.Mean) != c.Analysis.Bands {
		return errors.New("a codebook whose tables disagree about its size")
	}
	return nil
}

// Name is the token of a unit.
func (c *Codebook) Name(index int) string { return UnitPrefix + strconv.Itoa(index) }

// Names is every unit's token, in order.
func (c *Codebook) Names() []string {
	out := make([]string, c.K())
	for i := range out {
		out[i] = c.Name(i)
	}
	return out
}

// Index is the codebook index of a unit token, or -1 when the token is not one of this codebook's units.
func (c *Codebook) Index(token string) int {
	if len(token) < 2 || !strings.HasPrefix(token, UnitPrefix) {
		return -1
	}
	for _, r := range token[1:] {
		if r < '0' || r > '9' {
			return -1
		}
	}
	i, err := strconv.Atoi(token[1:])
	if err != nil || i >= c.K() || token != c.Name(i) {
		return -1
	}
	return i
}

// Nearest is the index of the centroid nearest the frame (the first of equals).
func (c *Codebook) Nearest(frame []float64) int {
	best := 0
	bestD := Dist2(frame, c.Centroids[0])
	for i := 1; i < c.K(); i++ {
		d := Dist2(frame, c.Centroids[i])
		if d < bestD {
			best, bestD = i, d
		}
	}
	return best
}

// Codes is every frame's nearest unit.
func (c *Codebook) Codes(frames [][]float64) []int {
	out := make([]int, len(frames))
	for i, f := range frames {
		out[i] = c.Nearest(f)
	}
	return out
}

// Units is the unit tokens of the frames, runs collapsed when asked.
func (c *Codebook) Units(frames [][]float64, collapse bool) []string {
	codes := c.Codes(frames)
	if collapse {
		codes = CollapseRuns(codes)
	}
	out := make([]string, len(codes))
	for i, code := range codes {
		out[i] = c.Name(code)
	}
	return out
}

// Hold is the frames a unit is held for when decoded: its typical run, at least one.
func (c *Codebook) Hold(index int) int {
	h := int(c.Runs[index] + 0.5)
	if h < 1 {
		return 1
	}
	return h
}

func formatFloat(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

// Dumps is the codebook as its TSV file.
func (c *Codebook) Dumps() string {
	a := c.Analysis
	var sb strings.Builder
	fmt.Fprintf(&sb, "# phonetok acoustic codebook: %d units learned by k-means over log-mel frames\n", c.K())
	if c.Note != "" {
		for _, line := range strings.Split(c.Note, "\n") {
			fmt.Fprintf(&sb, "# %s\n", line)
		}
	}
	normalize := 0
	if a.Normalize {
		normalize = 1
	}
	fmt.Fprintf(&sb, "rate\t%d\nframe\t%d\nhop\t%d\nfft\t%d\nbands\t%d\nfmin\t%s\nfmax\t%s\npreemphasis\t%s\n"+
		"normalize\t%d\nseed\t%d\ninertia\t%s\nunits\t%d\n", a.Rate, a.Frame, a.Hop, a.FFT, a.Bands,
		formatFloat(a.Fmin), formatFloat(a.Fmax), formatFloat(a.Preemphasis), normalize, c.Seed,
		formatFloat(c.Inertia), c.K())
	sb.WriteString("mean")
	for _, v := range c.Mean {
		sb.WriteString("\t" + formatFloat(v))
	}
	sb.WriteString("\n")
	for i, cen := range c.Centroids {
		fmt.Fprintf(&sb, "%s\t%d\t%s", c.Name(i), c.Counts[i], formatFloat(c.Runs[i]))
		for _, v := range cen {
			sb.WriteString("\t" + formatFloat(v))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// Dump writes the codebook file.
func (c *Codebook) Dump(path string) error { return os.WriteFile(path, []byte(c.Dumps()), 0o644) }

// ParseCodebook reads a codebook file's text.
func ParseCodebook(text string) (*Codebook, error) {
	settings := map[string]string{}
	var note []string
	var mean []float64
	book := &Codebook{}
	rows := 0
	floats := func(parts []string) ([]float64, error) {
		out := make([]float64, len(parts))
		for i, p := range parts {
			v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil {
				return nil, fmt.Errorf("unreadable number %q in the codebook", p)
			}
			out[i] = v
		}
		return out, nil
	}
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimRight(raw, "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if !strings.HasPrefix(line, "# phonetok acoustic codebook") {
				note = append(note, strings.TrimPrefix(strings.TrimPrefix(line, "#"), " "))
			}
			continue
		}
		parts := strings.Split(line, "\t")
		key := parts[0]
		switch {
		case key == "mean":
			v, err := floats(parts[1:])
			if err != nil {
				return nil, err
			}
			mean = v
		case strings.HasPrefix(key, UnitPrefix) && len(key) > 1 && isDigits(key[1:]):
			if len(parts) < 4 {
				return nil, fmt.Errorf("codebook row %q is incomplete", key)
			}
			if key != UnitPrefix+strconv.Itoa(rows) {
				return nil, fmt.Errorf("codebook rows out of order: %q where %q was expected", key, UnitPrefix+strconv.Itoa(rows))
			}
			count, err := strconv.Atoi(parts[1])
			if err != nil {
				return nil, fmt.Errorf("unreadable count in codebook row %q", key)
			}
			run, err := strconv.ParseFloat(parts[2], 64)
			if err != nil {
				return nil, fmt.Errorf("unreadable run in codebook row %q", key)
			}
			cen, err := floats(parts[3:])
			if err != nil {
				return nil, err
			}
			book.Centroids = append(book.Centroids, cen)
			book.Counts = append(book.Counts, count)
			book.Runs = append(book.Runs, run)
			rows++
		case len(parts) == 2:
			settings[key] = parts[1]
		default:
			return nil, fmt.Errorf("unreadable codebook line: %q", line)
		}
	}
	a := DefaultAnalysis()
	var err error
	intSetting := func(key string, into *int) {
		if err != nil {
			return
		}
		if v, ok := settings[key]; ok {
			*into, err = strconv.Atoi(v)
		}
	}
	floatSetting := func(key string, into *float64) {
		if err != nil {
			return
		}
		if v, ok := settings[key]; ok {
			*into, err = strconv.ParseFloat(v, 64)
		}
	}
	intSetting("rate", &a.Rate)
	intSetting("frame", &a.Frame)
	intSetting("hop", &a.Hop)
	intSetting("fft", &a.FFT)
	intSetting("bands", &a.Bands)
	floatSetting("fmin", &a.Fmin)
	floatSetting("fmax", &a.Fmax)
	floatSetting("preemphasis", &a.Preemphasis)
	if v, ok := settings["normalize"]; ok {
		a.Normalize = v != "0" && v != "false" && v != "no"
	}
	if v, ok := settings["seed"]; ok && err == nil {
		book.Seed, err = strconv.ParseInt(v, 10, 64)
	}
	if v, ok := settings["inertia"]; ok && err == nil {
		book.Inertia, err = strconv.ParseFloat(v, 64)
	}
	if err != nil {
		return nil, fmt.Errorf("unreadable codebook settings: %v", err)
	}
	book.Analysis = a
	book.Mean = mean
	book.Note = strings.Join(note, "\n")
	if err := book.Validate(); err != nil {
		return nil, err
	}
	return book, nil
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// LoadCodebook reads a codebook file.
func LoadCodebook(path string) (*Codebook, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseCodebook(string(data))
}

var (
	defaultOnce sync.Once
	defaultBook *Codebook
	defaultErr  error
)

// DefaultCodebook is the bundled codebook, read once.
func DefaultCodebook() (*Codebook, error) {
	defaultOnce.Do(func() { defaultBook, defaultErr = ParseCodebook(acousticText) })
	return defaultBook, defaultErr
}

// CollapseRuns folds every run of one code into one.
func CollapseRuns(codes []int) []int {
	var out []int
	for _, c := range codes {
		if len(out) == 0 || out[len(out)-1] != c {
			out = append(out, c)
		}
	}
	return out
}

// RunLengths is, per code, the frames it claimed and the runs it made.
func RunLengths(codes []int, k int) (frames, runs []int) {
	frames = make([]int, k)
	runs = make([]int, k)
	prev := -1
	for _, c := range codes {
		frames[c]++
		if c != prev {
			runs[c]++
		}
		prev = c
	}
	return frames, runs
}

// -- learning -----------------------------------------------------------------------------

// Learned is what KMeans found.
type Learned struct {
	Centroids   [][]float64
	Assignments []int
	Inertia     float64
	Iterations  int
	Converged   bool
}

// KMeans is k-means++ seeding, then Lloyd's iterations until nothing moves (or iterations are up).
// Every draw comes from rng (CPython's Mersenne Twister), so the same seed learns the same codebook in every port.
func KMeans(points [][]float64, k int, rng Rng, iterations int) (Learned, error) {
	n := len(points)
	if k < 1 {
		return Learned{}, errors.New("k must be at least 1")
	}
	if n < k {
		return Learned{}, fmt.Errorf("%d frames cannot make %d units", n, k)
	}
	first := rng.RandBelow(n)
	centroids := [][]float64{append([]float64(nil), points[first]...)}
	d2 := make([]float64, n)
	for i, p := range points {
		d2[i] = Dist2(p, centroids[0])
	}
	for len(centroids) < k {
		total := 0.0
		for _, d := range d2 {
			total += d
		}
		chosen := n - 1
		if total > 0.0 {
			r := rng.Float64() * total
			acc := 0.0
			for i, d := range d2 {
				acc += d
				if acc >= r {
					chosen = i
					break
				}
			}
		} else {
			chosen = rng.RandBelow(n)
		}
		centre := append([]float64(nil), points[chosen]...)
		centroids = append(centroids, centre)
		for i, p := range points {
			if d := Dist2(p, centre); d < d2[i] {
				d2[i] = d
			}
		}
	}
	dims := 0
	if n > 0 {
		dims = len(points[0])
	}
	assignments := make([]int, n)
	for i := range assignments {
		assignments[i] = -1
	}
	inertia := 0.0
	converged := false
	done := 0
	for it := 0; it < iterations; it++ {
		done = it + 1
		changed := 0
		inertia = 0.0
		for i, p := range points {
			best := 0
			bestD := Dist2(p, centroids[0])
			for c := 1; c < k; c++ {
				if d := Dist2(p, centroids[c]); d < bestD {
					best, bestD = c, d
				}
			}
			if best != assignments[i] {
				changed++
				assignments[i] = best
			}
			d2[i] = bestD
			inertia += bestD
		}
		if changed == 0 {
			converged = true
			break
		}
		sums := make([][]float64, k)
		for c := range sums {
			sums[c] = make([]float64, dims)
		}
		counts := make([]int, k)
		for i, p := range points {
			c := assignments[i]
			s := sums[c]
			for b := 0; b < dims; b++ {
				s[b] += p[b]
			}
			counts[c]++
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				cen := make([]float64, dims)
				for b := 0; b < dims; b++ {
					cen[b] = sums[c][b] / float64(counts[c])
				}
				centroids[c] = cen
			} else {
				far := 0
				farD := -1.0
				for i := 0; i < n; i++ {
					if d2[i] > farD {
						far, farD = i, d2[i]
					}
				}
				centroids[c] = append([]float64(nil), points[far]...)
				assignments[far] = c
				d2[far] = 0.0
			}
		}
	}
	return Learned{Centroids: centroids, Assignments: assignments, Inertia: inertia, Iterations: done, Converged: converged}, nil
}

// Learn is a codebook of k units learned from recordings (each samples at a.Rate).  Nothing but the audio is
// looked at: the frames are analysed raw, their mean over all the recordings is kept for decoding, every
// recording is normalised on its own, and k-means clusters the lot.
func Learn(recordings [][]float64, k int, seed int64, iterations int, a Analysis, note string) (*Codebook, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	var utterances [][][]float64
	sums := make([]float64, a.Bands)
	total := 0
	for _, samples := range recordings {
		frames, err := Frames(samples, a, false)
		if err != nil {
			return nil, err
		}
		for _, f := range frames {
			for b := 0; b < a.Bands; b++ {
				sums[b] += f[b]
			}
		}
		total += len(frames)
		if a.Normalize {
			means := BandMeans(frames)
			for _, f := range frames {
				for b := 0; b < a.Bands; b++ {
					f[b] = f[b] - means[b]
				}
			}
		}
		utterances = append(utterances, frames)
	}
	if total == 0 {
		return nil, errors.New("no audio to learn from")
	}
	mean := make([]float64, a.Bands)
	for b := range mean {
		mean[b] = sums[b] / float64(total)
	}
	var points [][]float64
	for _, frames := range utterances {
		points = append(points, frames...)
	}
	found, err := KMeans(points, k, NewMT64(uint64(seed)), iterations)
	if err != nil {
		return nil, err
	}
	counts := make([]int, k)
	runs := make([]int, k)
	pos := 0
	for _, frames := range utterances {
		codes := found.Assignments[pos : pos+len(frames)]
		pos += len(frames)
		fr, ru := RunLengths(codes, k)
		for c := 0; c < k; c++ {
			counts[c] += fr[c]
			runs[c] += ru[c]
		}
	}
	meanRuns := make([]float64, k)
	for c := 0; c < k; c++ {
		if runs[c] > 0 {
			meanRuns[c] = float64(counts[c]) / float64(runs[c])
		} else {
			meanRuns[c] = 1.0
		}
	}
	return &Codebook{Analysis: a, Centroids: found.Centroids, Runs: meanRuns, Counts: counts, Mean: mean, Seed: seed,
		Note: note, Inertia: found.Inertia}, nil
}

// -- decoding -----------------------------------------------------------------------------

var (
	areasMu    sync.Mutex
	areasCache = map[Analysis][]float64{}
)

// BandAreas is, per band, the summed weight of its filter: the width in bins its energy was summed over.
func BandAreas(a Analysis) []float64 {
	areasMu.Lock()
	defer areasMu.Unlock()
	if cached, ok := areasCache[a]; ok {
		return cached
	}
	filters := melFilters(a)
	areas := make([]float64, len(filters))
	for b, f := range filters {
		total := 0.0
		for _, w := range f.weights {
			total += w
		}
		areas[b] = total
	}
	areasCache[a] = areas
	return areas
}

// MagnitudesOf is the linear magnitude spectrum (fft/2 + 1 bins) a log-mel frame stands for: a band's energy
// was summed over its bins, so it is spread back as that energy per unit of filter weight through the same
// triangles; where two filters overlap their weights sum to one, so a bin under both takes their mean.
func MagnitudesOf(logmel []float64, a Analysis) []float64 {
	half := a.FFT / 2
	power := make([]float64, half+1)
	areas := BandAreas(a)
	for b, f := range melFilters(a) {
		e := math.Exp(logmel[b]) / areas[b]
		for i, w := range f.weights {
			power[f.first+i] += w * e
		}
	}
	out := make([]float64, half+1)
	for k := 0; k <= half; k++ {
		out[k] = math.Sqrt(power[k])
	}
	return out
}

// BandCentres is the centre frequency of every band, in Hz.
func BandCentres(a Analysis) []float64 {
	lo, hi := Mel(a.Fmin), Mel(a.Fmax)
	out := make([]float64, a.Bands)
	for b := range out {
		out[b] = MelToHz(lo + (hi-lo)*float64(b+1)/float64(a.Bands+1))
	}
	return out
}

var (
	voicingMu    sync.Mutex
	voicingCache = map[Analysis][2][]int{}
)

// VoicingOf is how voiced a frame sounds, 0 to 1, read off its tilt: the low bands (under 1 kHz) over the
// high (over 3 kHz).  Fully voiced from a tilt of 4 nats, fully unvoiced from -2.
func VoicingOf(logmel []float64, a Analysis) float64 {
	voicingMu.Lock()
	bands, ok := voicingCache[a]
	if !ok {
		for b, c := range BandCentres(a) {
			if c < 1000.0 {
				bands[0] = append(bands[0], b)
			}
			if c > 3000.0 {
				bands[1] = append(bands[1], b)
			}
		}
		voicingCache[a] = bands
	}
	voicingMu.Unlock()
	low, high := bands[0], bands[1]
	if len(low) == 0 || len(high) == 0 {
		return 1.0
	}
	sl, sh := 0.0, 0.0
	for _, b := range low {
		sl += logmel[b]
	}
	for _, b := range high {
		sh += logmel[b]
	}
	tilt := sl/float64(len(low)) - sh/float64(len(high))
	v := (tilt + 2.0) / 6.0
	if v < 0.0 {
		return 0.0
	}
	if v > 1.0 {
		return 1.0
	}
	return v
}

// excitation is what the vocoder takes its phases from: a pulse train at the voice's pitch, and noise, both
// running on continuously so consecutive frames agree about the phase of every bin.
type excitation struct {
	a      Analysis
	step   float64
	phase  float64
	noise  *noiseSource
	pulses []float64
	noises []float64
	offset int
	window []float64
}

func newExcitation(a Analysis, pitch float64) (*excitation, error) {
	if pitch <= 0.0 {
		return nil, fmt.Errorf("pitch must be positive, got %g", pitch)
	}
	return &excitation{a: a, step: pitch / float64(a.Rate), noise: newNoise(), window: Hann(a.Frame)}, nil
}

func (e *excitation) ensure(upto int) {
	for e.offset+len(e.pulses) < upto {
		e.phase += e.step
		if e.phase >= 1.0 {
			e.phase -= 1.0
			e.pulses = append(e.pulses, 1.0)
		} else {
			e.pulses = append(e.pulses, 0.0)
		}
		e.noises = append(e.noises, e.noise.next())
	}
}

// phases is the unit phasors (cos, sin per bin, fft/2 + 1 of them) of the excitation frame at start.
func (e *excitation) phases(start int, voicing float64) (cos, sin []float64) {
	a := e.a
	e.ensure(start + a.Frame)
	if start-e.offset > 4*a.Frame {
		drop := start - e.offset - a.Frame
		e.pulses = e.pulses[drop:]
		e.noises = e.noises[drop:]
		e.offset += drop
	}
	noiseGain := 0.25*(1.0-voicing) + 0.02
	base := start - e.offset
	re := make([]float64, a.FFT)
	im := make([]float64, a.FFT)
	for i := 0; i < a.Frame; i++ {
		re[i] = (voicing*e.pulses[base+i] + noiseGain*e.noises[base+i]) * e.window[i]
	}
	FFT(re, im, false)
	half := a.FFT / 2
	cos = make([]float64, half+1)
	sin = make([]float64, half+1)
	for k := 0; k <= half; k++ {
		cos[k] = 1.0
		if m := math.Sqrt(re[k]*re[k] + im[k]*im[k]); m > 0.0 {
			cos[k] = re[k] / m
			sin[k] = im[k] / m
		}
	}
	return cos, sin
}

// spectrum is the full (Hermitian) spectrum of a frame from its magnitudes and unit phasors.
func spectrum(mags, cos, sin []float64, n int) (re, im []float64) {
	half := n / 2
	re = make([]float64, n)
	im = make([]float64, n)
	for k := 0; k <= half; k++ {
		re[k] = mags[k] * cos[k]
		im[k] = mags[k] * sin[k]
	}
	im[0] = 0.0
	im[half] = 0.0
	for k := 1; k < half; k++ {
		re[n-k] = re[k]
		im[n-k] = -im[k]
	}
	return re, im
}

func packPCM(samples []float64, gain float64) []byte {
	out := make([]byte, 0, 2*len(samples))
	scale := gain * 32767.0
	for _, x := range samples {
		s := x * scale
		if s > 32767.0 {
			s = 32767.0
		} else if s < -32767.0 {
			s = -32767.0
		}
		v := int16(s)
		out = append(out, byte(v), byte(v>>8))
	}
	return out
}

// Vocoder turns units into a waveform as they come: every unit is its centroid held for its typical run, every
// frame is spread back over the linear spectrum with the excitation's phases, and the samples no later frame
// can touch are handed back at once.  End flushes the tail.
type Vocoder struct {
	Book         *Codebook
	Gain         float64
	Pitch        float64
	TotalSamples int
	a            Analysis
	window       []float64
	excitation   *excitation
	acc          []float64
	norm         []float64
	frames       int
	emitted      int
}

// NewVocoder is a vocoder over a codebook (nil: the bundled one).
func NewVocoder(book *Codebook, gain, pitch float64) (*Vocoder, error) {
	if book == nil {
		var err error
		if book, err = DefaultCodebook(); err != nil {
			return nil, err
		}
	}
	if err := book.Validate(); err != nil {
		return nil, err
	}
	exc, err := newExcitation(book.Analysis, pitch)
	if err != nil {
		return nil, err
	}
	return &Vocoder{Book: book, Gain: gain, Pitch: pitch, a: book.Analysis, window: Hann(book.Analysis.Frame),
		excitation: exc}, nil
}

// Feed takes one unit; the PCM that no later unit can change comes back.
func (v *Vocoder) Feed(unit string) ([]byte, error) {
	index := v.Book.Index(unit)
	if index < 0 {
		return nil, fmt.Errorf("not a unit of this codebook: %q", unit)
	}
	frame := make([]float64, v.a.Bands)
	for b := range frame {
		frame[b] = v.Book.Centroids[index][b] + v.Book.Mean[b]
	}
	var out []byte
	for i := 0; i < v.Book.Hold(index); i++ {
		out = append(out, v.FeedFrame(frame)...)
	}
	return out, nil
}

// FeedFrame takes one absolute log-mel frame (the codebook's mean already added); the PCM that is ready comes back.
func (v *Vocoder) FeedFrame(logmel []float64) []byte {
	a := v.a
	start := v.frames * a.Hop
	mags := MagnitudesOf(logmel, a)
	cos, sin := v.excitation.phases(start, VoicingOf(logmel, a))
	re, im := spectrum(mags, cos, sin, a.FFT)
	FFT(re, im, true)
	end := start + a.Frame
	for len(v.acc) < end {
		v.acc = append(v.acc, 0.0)
		v.norm = append(v.norm, 0.0)
	}
	for i := 0; i < a.Frame; i++ {
		w := v.window[i]
		v.acc[start+i] += re[i] * w
		v.norm[start+i] += w * w
	}
	v.frames++
	ready := v.frames * a.Hop
	if ready > len(v.acc) {
		ready = len(v.acc)
	}
	return v.emit(ready)
}

func (v *Vocoder) emit(upto int) []byte {
	if upto <= v.emitted {
		return nil
	}
	chunk := make([]float64, 0, upto-v.emitted)
	for i := v.emitted; i < upto; i++ {
		norm := v.norm[i]
		if norm > 1e-3 {
			chunk = append(chunk, v.acc[i]/norm)
		} else {
			chunk = append(chunk, v.acc[i]/1e-3)
		}
	}
	v.emitted = upto
	v.TotalSamples += len(chunk)
	return packPCM(chunk, v.Gain)
}

// End is the tail: what the last frames left pending.  The vocoder is ready for the next utterance.
func (v *Vocoder) End() []byte {
	out := v.emit(len(v.acc))
	v.excitation, _ = newExcitation(v.a, v.Pitch)
	v.acc = nil
	v.norm = nil
	v.frames = 0
	v.emitted = 0
	return out
}

type frameSpectrum struct{ re, im []float64 }

func stft(samples []float64, a Analysis, count int) []frameSpectrum {
	window := Hann(a.Frame)
	out := make([]frameSpectrum, 0, count)
	for t := 0; t < count; t++ {
		start := t * a.Hop
		re := make([]float64, a.FFT)
		im := make([]float64, a.FFT)
		for i := 0; i < a.Frame; i++ {
			s := 0.0
			if start+i < len(samples) {
				s = samples[start+i]
			}
			re[i] = s * window[i]
		}
		FFT(re, im, false)
		out = append(out, frameSpectrum{re, im})
	}
	return out
}

func istft(spectra []frameSpectrum, a Analysis) []float64 {
	window := Hann(a.Frame)
	length := 0
	if len(spectra) > 0 {
		length = (len(spectra)-1)*a.Hop + a.Frame
	}
	acc := make([]float64, length)
	norm := make([]float64, length)
	for t, s := range spectra {
		re := append([]float64(nil), s.re...)
		im := append([]float64(nil), s.im...)
		FFT(re, im, true)
		start := t * a.Hop
		for j := 0; j < a.Frame; j++ {
			w := window[j]
			acc[start+j] += re[j] * w
			norm[start+j] += w * w
		}
	}
	out := make([]float64, length)
	for i := range out {
		if norm[i] > 1e-3 {
			out[i] = acc[i] / norm[i]
		} else {
			out[i] = acc[i] / 1e-3
		}
	}
	return out
}

// GriffinLim is samples for a run of absolute log-mel frames, polished: the first pass is exactly the
// Vocoder's, and each iteration transforms the samples back, keeps the phases that came out and restores
// the magnitudes.
func GriffinLim(frames [][]float64, a Analysis, iterations int, pitch float64) ([]float64, error) {
	n := a.FFT
	half := n / 2
	count := len(frames)
	if count == 0 {
		return nil, nil
	}
	magnitudes := make([][]float64, count)
	for t, f := range frames {
		magnitudes[t] = MagnitudesOf(f, a)
	}
	exc, err := newExcitation(a, pitch)
	if err != nil {
		return nil, err
	}
	spectra := make([]frameSpectrum, 0, count)
	for t, f := range frames {
		cos, sin := exc.phases(t*a.Hop, VoicingOf(f, a))
		re, im := spectrum(magnitudes[t], cos, sin, n)
		spectra = append(spectra, frameSpectrum{re, im})
	}
	for it := 0; it < iterations; it++ {
		samples := istft(spectra, a)
		fresh := stft(samples, a, count)
		for t := range fresh {
			re, im := fresh[t].re, fresh[t].im
			mags := magnitudes[t]
			for k := 0; k <= half; k++ {
				m := math.Sqrt(re[k]*re[k] + im[k]*im[k])
				if m > 0.0 {
					re[k] = re[k] / m * mags[k]
					im[k] = im[k] / m * mags[k]
				} else {
					re[k] = mags[k]
					im[k] = 0.0
				}
			}
			im[0] = 0.0
			im[half] = 0.0
			for k := 1; k < half; k++ {
				re[n-k] = re[k]
				im[n-k] = -im[k]
			}
		}
		spectra = fresh
	}
	return istft(spectra, a), nil
}

// UnitFrames is the absolute log-mel frames a run of units stands for: each centroid plus the mean, held for its run.
func UnitFrames(units []string, book *Codebook) ([][]float64, error) {
	var frames [][]float64
	for _, unit := range units {
		index := book.Index(unit)
		if index < 0 {
			return nil, fmt.Errorf("not a unit of this codebook: %q", unit)
		}
		frame := make([]float64, book.Analysis.Bands)
		for b := range frame {
			frame[b] = book.Centroids[index][b] + book.Mean[b]
		}
		for i := 0; i < book.Hold(index); i++ {
			frames = append(frames, frame)
		}
	}
	return frames, nil
}

// Synthesize is 16-bit PCM at the codebook's rate for a run of units; polish is the number of Griffin-Lim
// iterations over the whole utterance (0 is exactly what the streaming Vocoder makes).
func Synthesize(units []string, book *Codebook, polish int, gain, pitch float64) ([]byte, error) {
	if polish <= 0 {
		voc, err := NewVocoder(book, gain, pitch)
		if err != nil {
			return nil, err
		}
		var out []byte
		for _, u := range units {
			chunk, err := voc.Feed(u)
			if err != nil {
				return nil, err
			}
			out = append(out, chunk...)
		}
		return append(out, voc.End()...), nil
	}
	frames, err := UnitFrames(units, book)
	if err != nil {
		return nil, err
	}
	samples, err := GriffinLim(frames, book.Analysis, polish, pitch)
	if err != nil {
		return nil, err
	}
	return packPCM(samples, gain), nil
}

// -- the facade ---------------------------------------------------------------------------

// Heard is what an utterance was heard as.
type Heard struct {
	Units   []string
	Codes   []int
	Seconds float64
	Frames  int
}

// Text is the units as one line.
func (h Heard) Text() string { return strings.Join(h.Units, " ") }

// AcousticTokenizer is audio to units and back, over one codebook.
type AcousticTokenizer struct {
	Book     *Codebook
	Collapse bool
}

// NewAcousticTokenizer is a tokenizer over book (nil: the bundled codebook); collapse folds runs into one token.
func NewAcousticTokenizer(book *Codebook, collapse bool) (*AcousticTokenizer, error) {
	if book == nil {
		var err error
		if book, err = DefaultCodebook(); err != nil {
			return nil, err
		}
	}
	if err := book.Validate(); err != nil {
		return nil, err
	}
	return &AcousticTokenizer{Book: book, Collapse: collapse}, nil
}

// Samples is a WAV file's bytes as samples at the codebook's rate.
func (t *AcousticTokenizer) Samples(wav []byte) ([]float64, error) {
	values, rate, err := ReadWAV(wav)
	if err != nil {
		return nil, err
	}
	return Resample(values, rate, t.Book.Analysis.Rate)
}

// Listen is everything an utterance (samples at rate) was heard as.
func (t *AcousticTokenizer) Listen(samples []float64, rate int) (Heard, error) {
	if rate == 0 {
		rate = t.Book.Analysis.Rate
	}
	samples, err := Resample(samples, rate, t.Book.Analysis.Rate)
	if err != nil {
		return Heard{}, err
	}
	frames, err := FramesOf(samples, t.Book.Analysis)
	if err != nil {
		return Heard{}, err
	}
	codes := t.Book.Codes(frames)
	kept := codes
	if t.Collapse {
		kept = CollapseRuns(codes)
	}
	units := make([]string, len(kept))
	for i, c := range kept {
		units[i] = t.Book.Name(c)
	}
	return Heard{Units: units, Codes: codes, Seconds: float64(len(samples)) / float64(t.Book.Analysis.Rate),
		Frames: len(frames)}, nil
}

// ListenWAV is Listen over a WAV file's bytes.
func (t *AcousticTokenizer) ListenWAV(wav []byte) (Heard, error) {
	values, rate, err := ReadWAV(wav)
	if err != nil {
		return Heard{}, err
	}
	return t.Listen(values, rate)
}

// Hear is the units of an utterance.
func (t *AcousticTokenizer) Hear(samples []float64, rate int) ([]string, error) {
	h, err := t.Listen(samples, rate)
	return h.Units, err
}

// IsUnit says whether a token is one of the codebook's units.
func (t *AcousticTokenizer) IsUnit(token string) bool { return t.Book.Index(token) >= 0 }

// UnitsOf is the unit tokens of a text of units, checked against the codebook.
func (t *AcousticTokenizer) UnitsOf(text string) ([]string, error) {
	var units []string
	for _, token := range strings.Fields(text) {
		if t.Book.Index(token) < 0 {
			return nil, fmt.Errorf("not a unit of this codebook: %q", token)
		}
		if t.Collapse && len(units) > 0 && units[len(units)-1] == token {
			continue
		}
		units = append(units, token)
	}
	return units, nil
}

// Text is the text form of a text of units: one line, space-separated, runs collapsed; idempotent.
func (t *AcousticTokenizer) Text(text string) (string, error) {
	units, err := t.UnitsOf(text)
	if err != nil {
		return "", err
	}
	return strings.Join(units, " "), nil
}

// Synthesize is the units spoken back, as 16-bit PCM at the codebook's rate.
func (t *AcousticTokenizer) Synthesize(units []string, polish int, gain, pitch float64) ([]byte, error) {
	return Synthesize(units, t.Book, polish, gain, pitch)
}

// Vocoder is a streaming vocoder over the codebook.
func (t *AcousticTokenizer) Vocoder(gain, pitch float64) (*Vocoder, error) {
	return NewVocoder(t.Book, gain, pitch)
}
