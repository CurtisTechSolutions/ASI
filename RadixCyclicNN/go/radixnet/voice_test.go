package radixnet

import (
	"bytes"
	"reflect"
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

// The output decoder: hearing a walk as it walks and saying its text afterwards
// are the same audio, byte for byte; a text of acoustic units is spoken through
// the vocoder, polished or not, and refused when it holds a token that is not a
// unit.
func TestSayIsTheWalkHeardAfterTheFact(t *testing.T) {
	for _, spec := range []string{"phone:3:1", "char:3:1", "word:2:1"} {
		m := spokenModel(t, spec)
		s := int64(1)
		var said []string
		var live []byte
		err := m.SpeakWalks(SpeakOptions{Prefix: "the cat", Count: 2, MaxLength: 30, Temperature: 1, Seed: &s,
			Rate: 16000, Pitch: 120, Tempo: 1, Gain: 0.5},
			func(chunk []byte) { live = append(live, chunk...) },
			func(i int, text, spelled string) { said = append(said, text) })
		if err != nil {
			t.Fatal(err)
		}
		spoken, err := Say(m.Encoding(), said, DefaultSayOptions())
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(spoken.PCM, live) {
			t.Errorf("%s: the decoder says %d bytes, the walk was heard as %d", spec, len(spoken.PCM), len(live))
		}
		if len(spoken.Utterances) != 2 || spoken.Decoder != "voice" || spoken.Rate != 16000 || spoken.Encoding != spec {
			t.Errorf("%s: %d utterances, decoder %q, rate %d, encoding %q", spec, len(spoken.Utterances), spoken.Decoder, spoken.Rate, spoken.Encoding)
		}
		total := 0
		for _, u := range spoken.Utterances {
			total += u.Samples
			if u.Tokens[len(u.Tokens)-1] != "</s>" {
				t.Errorf("%s: the utterance %q was not closed by the sentinel: %v", spec, u.Text, u.Tokens)
			}
			if u.Seconds != float64(u.Samples)/16000 {
				t.Errorf("%s: %d samples are %g s, not %g", spec, u.Samples, float64(u.Samples)/16000, u.Seconds)
			}
		}
		if total != spoken.Samples() || spoken.Seconds() != float64(total)/16000 {
			t.Errorf("%s: %d samples in the utterances, %d spoken (%g s)", spec, total, spoken.Samples(), spoken.Seconds())
		}
		if wav := spoken.WAV(); len(wav) != 44+len(spoken.PCM) || string(wav[:4]) != "RIFF" {
			t.Errorf("%s: not a WAV of the speech: %d bytes", spec, len(wav))
		}
		if doc := spoken.Dict(); doc["count"] != 2 || doc["decoder"] != "voice" {
			t.Errorf("%s: the record says %v", spec, doc)
		}
	}
	// what a text spells: a text of sounds through the tokenizer, anything else as it is
	phones, _ := ParseEncoding("phone:3:1")
	if got := Spelled(phones, "DH AH0 # K AE1 T"); got != "the cat" {
		t.Errorf("the sounds spell %q", got)
	}
	if got := Spelled(phones, "the cat"); got != "the cat" {
		t.Errorf("the words spell %q", got)
	}
	if got := Spelled(DefaultEncoding(), "DH AH0"); got != "DH AH0" {
		t.Errorf("letters spell %q", got)
	}
	// acoustic units: the codebook's vocoder, at its rate
	acoustic, _ := ParseEncoding("acoustic:3:1")
	units := []string{"q2 q28 q55 q5", "q1 q2"}
	spoken, err := Say(acoustic, units, DefaultSayOptions())
	if err != nil {
		t.Fatal(err)
	}
	if spoken.Decoder != "vocoder" || spoken.Rate != 16000 || len(spoken.Utterances) != 2 || spoken.Samples() == 0 {
		t.Fatalf("units: decoder %q, rate %d, %d utterances, %d samples", spoken.Decoder, spoken.Rate, len(spoken.Utterances), spoken.Samples())
	}
	if !reflect.DeepEqual(spoken.Utterances[1].Tokens, []string{"q1", "q2", "</s>"}) || spoken.Utterances[1].Spelled != "q1 q2" {
		t.Errorf("units: %v spells %q", spoken.Utterances[1].Tokens, spoken.Utterances[1].Spelled)
	}
	o := DefaultSayOptions()
	o.Polish = 4
	polished, err := Say(acoustic, units, o)
	if err != nil {
		t.Fatal(err)
	}
	if polished.Samples() != spoken.Samples() || bytes.Equal(polished.PCM, spoken.PCM) {
		t.Errorf("polished: %d samples against %d, same audio %v", polished.Samples(), spoken.Samples(), bytes.Equal(polished.PCM, spoken.PCM))
	}
	for _, opts := range []SayOptions{DefaultSayOptions(), o} {
		if _, err := Say(acoustic, []string{"q2 nope"}, opts); err == nil || !strings.Contains(err.Error(), "not a unit") {
			t.Errorf("a token that is not a unit was spoken (polish %d): %v", opts.Polish, err)
		}
	}
}
