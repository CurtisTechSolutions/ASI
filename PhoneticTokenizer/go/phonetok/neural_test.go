package phonetok

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestPulseTrainAndNoise(t *testing.T) {
	pulses, err := PulseTrain(16000, 100.0, 16000)
	if err != nil {
		t.Fatal(err)
	}
	count, first := 0, -1
	for i, v := range pulses {
		if v == 1.0 {
			count++
			if first < 0 {
				first = i
			}
		}
	}
	// the phase accumulates 0.00625 a sample, which is not exact in binary: the first pulse lands on sample 160
	// and ninety-nine fit in a second, in every port alike
	if count != 99 || first != 160 {
		t.Fatalf("pulses %d first %d", count, first)
	}
	if _, err := PulseTrain(10, 0, 16000); err == nil {
		t.Fatal("a pitch of zero")
	}
	noise := NoiseTrain(5)
	longer := NoiseTrain(8)
	for i, v := range noise {
		if v != longer[i] || v < -1 || v >= 1 {
			t.Fatalf("noise %v vs %v", noise, longer)
		}
	}
	kernel := ChirpKernel()
	energy := 0.0
	for _, v := range kernel {
		energy += v * v
	}
	if len(kernel) != Chirp || math.Abs(energy-1) > 1e-12 {
		t.Fatalf("kernel of %d samples, energy %g", len(kernel), energy)
	}
	train, _ := PulseTrain(1600, 100.0, 16000)
	dispersed := Disperse(train)
	energy = 0
	for _, v := range dispersed {
		energy += v * v
	}
	if math.Abs(energy-9) > 1e-9 || dispersed[100] != 0 || dispersed[160] != 0 || dispersed[161] == 0 {
		t.Fatalf("dispersed energy %g", energy)
	}
}

func TestTensors(t *testing.T) {
	got, err := vocoderTensor{Shape: []int{2}, F32: "AACAPwAAAEA="}.values(2)
	if err != nil || got[0] != 1.0 || got[1] != 2.0 {
		t.Fatalf("%v %v", got, err)
	}
	if _, err := (vocoderTensor{Shape: []int{2}, F32: "AACAPwAAAEA="}).values(3); err == nil {
		t.Fatal("a wrong shape")
	}
	if _, err := ParseVocoder(`{"phonetok": "codebook"}`); err == nil {
		t.Fatal("not a vocoder")
	}
}

func TestTheBundledVocoderSpeaksItsCodebook(t *testing.T) {
	book, err := DefaultCodebook()
	if err != nil {
		t.Fatal(err)
	}
	vocoder, err := DefaultVocoder()
	if err != nil {
		t.Fatal(err)
	}
	if !vocoder.Matches(book) || vocoder.Spec.Units != book.K() {
		t.Fatal("the bundled vocoder is not the bundled codebook's")
	}
	units := []string{"q1", "q2", "q3"}
	codes, err := CodesOf(units, book)
	if err != nil {
		t.Fatal(err)
	}
	held := book.Hold(1) + book.Hold(2) + book.Hold(3)
	if len(codes) != held {
		t.Fatalf("%d codes for %d frames", len(codes), held)
	}
	pcm, err := vocoder.Synthesize(codes, 1.0, 120.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(pcm)/2 != ExcitationLength(held, book.Analysis) {
		t.Fatalf("%d samples", len(pcm)/2)
	}
	if _, _, err := vocoder.Filters([]int{book.K()}); err == nil {
		t.Fatal("a code past the units")
	}
	if _, err := CodesOf([]string{"nope"}, book); err == nil {
		t.Fatal("not a unit")
	}
	found, err := FindVocoder(book, "", "")
	if err != nil || found == nil || found.Checksum != vocoder.Checksum {
		t.Fatalf("find: %v %v", found, err)
	}
	if VocoderPathFor("mine.tsv") != "mine.vocoder.json" || VocoderPathFor("mine") != "mine.vocoder.json" {
		t.Fatal(VocoderPathFor("mine.tsv"))
	}
	tok, err := NewAcousticTokenizer(nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if name, err := tok.VocoderName(nil); err != nil || name != "neural" {
		t.Fatalf("%s %v", name, err)
	}
	again, err := tok.Synthesize(units, 0, 1.0, 120.0)
	if err != nil || string(again) != string(pcm) {
		t.Fatalf("the tokenizer speaks otherwise: %v", err)
	}
	centroid := false
	other, err := tok.SynthesizeWith(units, 0, 1.0, 120.0, &centroid)
	if err != nil || string(other) == string(pcm) {
		t.Fatalf("the centroid vocoder speaks the same: %v", err)
	}
	// a codebook file without a vocoder beside it has none; a vocoder named that is not there is an error
	dir := t.TempDir()
	path := filepath.Join(dir, "book.tsv")
	if err := os.WriteFile(path, []byte(book.Dumps()), 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadAcousticTokenizer(path, true)
	if err != nil {
		t.Fatal(err)
	}
	if name, err := loaded.VocoderName(nil); err != nil || name != "centroid" {
		t.Fatalf("%s %v", name, err)
	}
	insist := true
	if _, err := loaded.VocoderName(&insist); err == nil {
		t.Fatal("insisting on a vocoder there is not")
	}
	loaded.VocoderPath = filepath.Join(dir, "missing.json")
	if _, err := FindVocoder(book, path, loaded.VocoderPath); err == nil {
		t.Fatal("a missing vocoder file")
	}
}
