package tokenizer

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/nn"
)

// Format names the file format.
const Format = "latent-tokenizer"

type weightJSON struct {
	Rows int    `json:"rows"`
	Cols int    `json:"cols"`
	Data string `json:"data"` // little-endian float32, base64
}

type fileJSON struct {
	Format  string                `json:"format"`
	Version int                   `json:"version"`
	Config  Config                `json:"config"`
	Seed    int64                 `json:"seed"`
	Steps   int                   `json:"steps"`
	Corpus  int                   `json:"corpus_bytes"`
	Stats   []Stat                `json:"stats"`
	Weights map[string]weightJSON `json:"weights"`
}

func encodeMat(m *nn.Mat) weightJSON {
	buf := make([]byte, 4*len(m.Data))
	for i, v := range m.Data {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	return weightJSON{Rows: m.Rows, Cols: m.Cols, Data: base64.StdEncoding.EncodeToString(buf)}
}

func decodeMat(w weightJSON, into *nn.Mat) error {
	buf, err := base64.StdEncoding.DecodeString(w.Data)
	if err != nil {
		return err
	}
	if w.Rows != into.Rows || w.Cols != into.Cols || len(buf) != 4*len(into.Data) {
		return fmt.Errorf("weight is %dx%d (%d bytes), want %dx%d", w.Rows, w.Cols, len(buf), into.Rows, into.Cols)
	}
	for i := range into.Data {
		into.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return nil
}

// ToJSON serialises the tokenizer.
func (t *Tokenizer) ToJSON() ([]byte, error) {
	f := fileJSON{Format: Format, Version: 1, Config: t.Config, Seed: t.Seed, Steps: t.Steps, Corpus: t.Corpus,
		Stats: t.Stats, Weights: map[string]weightJSON{}}
	for _, p := range t.Params() {
		f.Weights[p.Name] = encodeMat(p.W)
	}
	return json.Marshal(f)
}

// FromJSON reads a tokenizer.
func FromJSON(data []byte) (*Tokenizer, error) {
	var f fileJSON
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	if f.Format != Format {
		return nil, fmt.Errorf("not a %s file (format %q)", Format, f.Format)
	}
	t, err := New(f.Config, f.Seed)
	if err != nil {
		return nil, err
	}
	t.Steps, t.Corpus, t.Stats = f.Steps, f.Corpus, f.Stats
	for _, p := range t.Params() {
		w, ok := f.Weights[p.Name]
		if !ok {
			return nil, fmt.Errorf("weight %s missing", p.Name)
		}
		if err := decodeMat(w, p.W); err != nil {
			return nil, fmt.Errorf("weight %s: %v", p.Name, err)
		}
	}
	return t, nil
}

// Save writes the tokenizer (gzip when the path ends in .gz), through a temporary file.
func (t *Tokenizer) Save(path string) error {
	data, err := t.ToJSON()
	if err != nil {
		return err
	}
	return WriteFile(path, data)
}

// Load reads a tokenizer saved by Save.
func Load(path string) (*Tokenizer, error) {
	data, err := ReadFile(path)
	if err != nil {
		return nil, err
	}
	return FromJSON(data)
}

// WriteFile writes data to path, gzipped when the name ends in .gz, atomically.
func WriteFile(path string, data []byte) error {
	if strings.HasSuffix(path, ".gz") {
		var buf bytes.Buffer
		zw := gzip.NewWriter(&buf)
		if _, err := zw.Write(data); err != nil {
			return err
		}
		if err := zw.Close(); err != nil {
			return err
		}
		data = buf.Bytes()
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadFile reads a file written by WriteFile, sniffing gzip.
func ReadFile(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		zr, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		return io.ReadAll(zr)
	}
	return data, nil
}
