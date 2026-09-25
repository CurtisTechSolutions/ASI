package radixnet

import (
	"strings"
	"testing"
)

var spokenTexts = []string{
	"the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "a bird in the hand",
}

func spokenModel(t *testing.T, spec string) *Model {
	t.Helper()
	enc, err := ParseEncoding(spec)
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGraphOptions()
	opts.Encoding = enc
	m, err := NewModel(1, opts)
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	train := DefaultTrainOptions()
	train.Epochs = 2
	if _, err := m.Train(spokenTexts, train); err != nil {
		t.Fatal(err)
	}
	return m
}

// What is spoken is what the walk says: from START (the first node whole) and
// from a prefix, the utterance is the text a seeded sample of the same walk
// generates.
func TestSpeakWalksSaysWhatTheWalkSays(t *testing.T) {
	for _, spec := range []string{"phone:3:1", "char:3:1", "word:2:1"} {
		m := spokenModel(t, spec)
		enc := m.Encoding()
		for _, prefix := range []string{"", "the cat"} {
			for seed := int64(1); seed <= 3; seed++ {
				s := seed
				var said []string
				pcm := 0
				err := m.SpeakWalks(SpeakOptions{Prefix: prefix, Count: 2, MaxLength: 40, Temperature: 1, Seed: &s,
					Rate: 16000, Pitch: 120, Tempo: 1, Gain: 0.5},
					func(chunk []byte) { pcm += len(chunk) },
					func(i int, text, spelled string) { said = append(said, text) })
				if err != nil {
					t.Fatalf("%s %q seed %d: %v", spec, prefix, seed, err)
				}
				s = seed
				walks, err := m.Generate(GenerateOptions{MaxLength: 40, Mode: "sample", Temperature: 1, Count: 2, Seed: &s,
					Prefix: prefix, Traversal: "reward", PenaltyScale: 1, MeritScale: 1, TopP: 1})
				if err != nil {
					t.Fatal(err)
				}
				if len(said) != 2 || len(walks) != 2 {
					t.Fatalf("%s %q seed %d: %d utterances, %d walks", spec, prefix, seed, len(said), len(walks))
				}
				if pcm == 0 {
					t.Fatalf("%s %q seed %d: nothing spoken", spec, prefix, seed)
				}
				for i, spoken := range said {
					want := walks[i].Text
					if prefix != "" {
						if !strings.HasPrefix(spoken, want) {
							t.Errorf("%s %q seed %d: spoken %q does not start with the walk's %q", spec, prefix, seed, spoken, want)
						}
					} else if got := enc.Truncate(spoken, 40); got != want {
						t.Errorf("%s seed %d: spoken %q, the walk says %q", spec, seed, got, want)
					}
				}
			}
		}
	}
}
