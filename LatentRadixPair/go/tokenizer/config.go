// Package tokenizer is the trained context tokenizer: a feed-forward autoencoder that compresses the last
// Window bytes of context into a short code of small integers (the symbols the primed tree is addressed by)
// and decodes a code back into the bytes it stood for.
//
// The design borrows from image generation. Latent diffusion models first compress an image with an
// autoencoder and model only the latent; here the network compresses the context and the primed radix pair
// models only the codes. The bottleneck is finite scalar quantisation (round each bounded latent dimension to
// a few levels; the code is the grid cell), which never suffers codebook collapse. Training masks the finer
// symbols at random, the way masked-token and absorbing-state diffusion models corrupt their inputs, so the
// first symbol carries the coarsest description of the context and each further symbol refines it: exactly
// the ancestor chain the tree backs off along. The encoder's input is corrupted with a byte-noise level drawn
// afresh per example, the way a diffusion step's noise level is, so nearby contexts share codes.
package tokenizer

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// PAD is the input symbol for positions before a text starts; Symbols is the input and output alphabet:
// the 256 bytes and PAD.
const (
	PAD     = 256
	Symbols = 257
)

// Config is the tokenizer's shape.
type Config struct {
	Window     int     `json:"window"`      // bytes of context the encoder reads
	Embed      int     `json:"embed"`       // width of a byte's embedding
	EncHidden  int     `json:"enc_hidden"`  // encoder hidden width (two layers)
	DecHidden  int     `json:"dec_hidden"`  // decoder hidden width (two layers)
	OutEmbed   int     `json:"out_embed"`   // per-position width before the shared unembedding
	Levels     [][]int `json:"levels"`      // quantisation levels of each symbol's dimensions
	Noise      float64 `json:"noise"`       // highest byte-corruption rate of a training input
	Recency    float64 `json:"recency"`     // weight of a byte's reconstruction, per byte of age (1: uniform)
	Predict    float64 `json:"predict"`     // weight of predicting the byte after the window
	StartShare float64 `json:"start_share"` // share of training windows drawn from the first Window bytes
}

// DefaultConfig: 16 bytes of context into three symbols of radix 16 (two dimensions of four levels each),
// the newest byte's reconstruction weighing 1 and each older byte 0.6 of the next, and predicting the next
// byte weighing four times the reconstruction (the measured best of the shapes tried; DESIGN.md section 10).
func DefaultConfig() Config {
	return Config{Window: 16, Embed: 16, EncHidden: 256, DecHidden: 256, OutEmbed: 32,
		Levels: [][]int{{4, 4}, {4, 4}, {4, 4}}, Noise: 0.1, Recency: 0.6, Predict: 4, StartShare: 0.05}
}

// Weights are the reconstruction weights of the window's positions, oldest first, mean 1.
func (c Config) Weights() []float32 {
	w := make([]float32, c.Window)
	sum := 0.0
	for p := range w {
		age := c.Window - 1 - p
		v := math.Pow(c.Recency, float64(age))
		w[p] = float32(v)
		sum += v
	}
	for p := range w {
		w[p] *= float32(float64(c.Window) / sum)
	}
	return w
}

// Depth is the number of code symbols.
func (c Config) Depth() int { return len(c.Levels) }

// Dims is the number of latent dimensions over all symbols.
func (c Config) Dims() int {
	n := 0
	for _, l := range c.Levels {
		n += len(l)
	}
	return n
}

// Radix is how many values symbol s takes.
func (c Config) Radix(s int) int {
	r := 1
	for _, l := range c.Levels[s] {
		r *= l
	}
	return r
}

// Radices are the radices of all symbols.
func (c Config) Radices() []int {
	out := make([]int, c.Depth())
	for s := range out {
		out[s] = c.Radix(s)
	}
	return out
}

// Bits is the information a full code carries.
func (c Config) Bits() float64 {
	b := 0.0
	for s := range c.Levels {
		b += log2(float64(c.Radix(s)))
	}
	return b
}

// Validate checks the shape.
func (c Config) Validate() error {
	if c.Window < 1 || c.Embed < 1 || c.EncHidden < 1 || c.DecHidden < 1 || c.OutEmbed < 1 {
		return fmt.Errorf("window, embed, enc_hidden, dec_hidden and out_embed must be positive")
	}
	if len(c.Levels) == 0 {
		return fmt.Errorf("at least one code symbol is needed")
	}
	for s, dims := range c.Levels {
		if len(dims) == 0 {
			return fmt.Errorf("symbol %d has no dimensions", s)
		}
		for _, l := range dims {
			if l < 2 {
				return fmt.Errorf("symbol %d: every dimension needs at least 2 levels", s)
			}
		}
	}
	if c.Noise < 0 || c.Noise >= 1 || c.StartShare < 0 || c.StartShare > 1 {
		return fmt.Errorf("0 <= noise < 1 and 0 <= start_share <= 1")
	}
	if c.Recency <= 0 || c.Recency > 1 {
		return fmt.Errorf("0 < recency <= 1")
	}
	if c.Predict < 0 {
		return fmt.Errorf("predict must be >= 0")
	}
	return nil
}

// ParseLevels reads "4,4:4,4:4,4" (symbols separated by colons, dimensions by commas) or "16:16:16"
// (one dimension per symbol).
func ParseLevels(text string) ([][]int, error) {
	var out [][]int
	for _, sym := range strings.Split(strings.TrimSpace(text), ":") {
		var dims []int
		for _, d := range strings.Split(sym, ",") {
			v, err := strconv.Atoi(strings.TrimSpace(d))
			if err != nil || v < 2 {
				return nil, fmt.Errorf("levels: %q is not a level count of at least 2", d)
			}
			dims = append(dims, v)
		}
		out = append(out, dims)
	}
	return out, nil
}

// LevelsString is the inverse of ParseLevels.
func LevelsString(levels [][]int) string {
	var syms []string
	for _, dims := range levels {
		var ds []string
		for _, d := range dims {
			ds = append(ds, strconv.Itoa(d))
		}
		syms = append(syms, strings.Join(ds, ","))
	}
	return strings.Join(syms, ":")
}
