package radixnet

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
)

// The acoustic unit (D-082): a recording is heard as a text of learned units, a
// model learns from such texts alone, and is heard back through the vocoder.
func TestAcousticUnit(t *testing.T) {
	enc, err := ParseEncoding("acoustic:3:1")
	if err != nil || enc.Unit != Acoustic || enc.N != 3 {
		t.Fatalf("%v %+v", err, enc)
	}
	if e, _ := ParseEncoding("units:2:2"); e != (Encoding{Acoustic, 2, 2}) {
		t.Errorf("units:2:2 parsed as %+v", e)
	}
	if !Acoustic.Valid() || Acoustic.Phonetic() || !Acoustic.Tokens() || Acoustic.Word() != "unit" {
		t.Error("the kind's properties")
	}
	if got := enc.Units("q1  q2\nq3"); got.Len() != 3 {
		t.Errorf("units: %d", got.Len())
	}
	if grams := enc.Encode("q1 q2 q3 q4"); strings.Join(grams, "|") != "q1 q2 q3|q2 q3 q4" {
		t.Errorf("grams %v", grams)
	}
	if enc.Join("q1 q2", "q3") != "q1 q2 q3" || !enc.HasUnitPrefix("q1 q2 q3", "q1 q2") || enc.HasUnitPrefix("q1 q22 q3", "q1 q2") {
		t.Error("join / prefix")
	}
	if enc.Spell("q1 q2") != "q1 q2" {
		t.Error("spell")
	}
	if _, err := ParseEncoding("rune:3"); err == nil {
		t.Error("rune accepted")
	}
	// a recording is heard as units
	var texts []string
	dir := t.TempDir()
	var paths []string
	for i, tokens := range []string{
		"DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T",
		"DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # F L AO1 R",
		"DH AH0 # D AO1 G # S AE1 T # AA1 N # DH AH0 # M AE1 T",
	} {
		wav := phonetok.WavBytes(phonetok.NewSynthesizer(phonetok.Rate, phonetok.DefaultVoice()).Speak(strings.Fields(tokens)), phonetok.Rate)
		text, err := HearAudio(wav)
		if err != nil {
			t.Fatal(err)
		}
		units := strings.Fields(text)
		if len(units) < 10 {
			t.Fatalf("heard %q", text)
		}
		for j, u := range units {
			if !strings.HasPrefix(u, "q") || (j > 0 && units[j-1] == u) {
				t.Fatalf("unit %q at %d in %q", u, j, text)
			}
		}
		path := filepath.Join(dir, "clip"+string(rune('0'+i))+".wav")
		if err := os.WriteFile(path, wav, 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
		texts = append(texts, text)
	}
	if !IsAudioFile("x.WAV") || IsAudioFile("x.txt") {
		t.Error("IsAudioFile")
	}
	got, err := CollectTexts(SourceForFile(paths[0], "lines", 0))
	if err != nil || len(got) != 1 || got[0] != texts[0] {
		t.Fatalf("the audio source: %v %v", err, got)
	}
	// a model learns from sound alone
	opts := DefaultGraphOptions()
	opts.Encoding = enc
	m, err := NewModel(1, opts)
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	train := DefaultTrainOptions()
	train.Epochs = 2
	if _, err := m.Train(texts, train); err != nil {
		t.Fatal(err)
	}
	if m.G.NumTrigrams() < 10 {
		t.Fatalf("%d grams", m.G.NumTrigrams())
	}
	for _, label := range m.G.Labels[First:] {
		for _, u := range strings.Fields(label) {
			if !strings.HasPrefix(u, "q") {
				t.Fatalf("label %q", label)
			}
		}
	}
	// and is heard back
	seed := int64(1)
	var said []string
	pcm := 0
	err = m.SpeakWalks(SpeakOptions{Count: 2, MaxLength: 30, Temperature: 1, Seed: &seed, Rate: 8000, Pitch: 120, Tempo: 1, Gain: 0.5},
		func(chunk []byte) { pcm += len(chunk) },
		func(i int, text, spelled string) {
			said = append(said, text)
			if spelled != text {
				t.Errorf("spelled %q for %q", spelled, text)
			}
		})
	if err != nil || len(said) != 2 || pcm < 16000 {
		t.Fatalf("%v: %d utterances, %d bytes", err, len(said), pcm)
	}
	for _, u := range strings.Fields(said[0]) {
		if !strings.HasPrefix(u, "q") {
			t.Fatalf("said %q", said[0])
		}
	}
	sp, err := NewSpeaker(enc, 8000, 120, 1, 0.5)
	if err != nil || sp.Rate != 16000 {
		t.Fatalf("%v rate %d", err, sp.Rate)
	}
	if len(sp.Feed("q2 q28")) == 0 || len(sp.End()) == 0 || strings.Join(sp.Tokens, " ") != "q2 q28 </s>" {
		t.Errorf("the speaker: %v", sp.Tokens)
	}
	if r, _ := OutputRate(enc, 8000); r != 16000 {
		t.Errorf("output rate %d", r)
	}
	if r, _ := OutputRate(DefaultEncoding(), 8000); r != 8000 {
		t.Errorf("output rate %d", r)
	}
}
