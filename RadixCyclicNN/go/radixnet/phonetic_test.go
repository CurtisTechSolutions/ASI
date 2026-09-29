package radixnet

import (
	"reflect"
	"testing"
)

func mustEncoding(t *testing.T, spec string) Encoding {
	t.Helper()
	enc, err := ParseEncoding(spec)
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

// A text given in words is read as its sounds first, so it spells itself back,
// and so does a text that mixes the two.
func TestSpellReadsWordsAsTheirSounds(t *testing.T) {
	phones, syllables := mustEncoding(t, "phone:3:1"), mustEncoding(t, "syllable:2:1")
	for _, c := range []struct {
		enc        Encoding
		text, want string
	}{
		{phones, "DH AH0 # K AE1 T # S AE1 T .", "the cat sat."},
		{phones, "the cat sat", "the cat sat"},
		{phones, "the K AE1 T sat", "the cat sat"},
		{syllables, "B.AH1.T ER0 # K.AH1.P", "butter cup"},
		{syllables, "the K.AE1.T sat", "the cat sat"},
		{mustEncoding(t, "char:3:1"), "DH AH0", "DH AH0"},
	} {
		if got := c.enc.Spell(c.text); got != c.want {
			t.Errorf("%s Spell(%q) = %q, want %q", c.enc, c.text, got, c.want)
		}
	}
}

// A continuation is spelled as the part of the whole's words it wrote, with the
// space before it; one that finishes a word the prefix began is cut where that
// word starts.  Python's test_a_continuation_is_spelled_as_the_part_it_wrote.
func TestSpellTailIsThePartTheContinuationWrote(t *testing.T) {
	for _, spec := range []string{"phone:3:1", "syllable:2:1"} {
		enc := mustEncoding(t, spec)
		whole := enc.Join("the cat sat on the mat. hello")
		for _, prefix := range []string{"", "the", "the cat", "the cat sat on the mat"} {
			head := enc.Join(prefix)
			tail := whole[len(head):]
			if got := enc.Spell(head) + enc.SpellTail(whole, tail); got != "the cat sat on the mat. hello" {
				t.Errorf("%s after %q: %q", spec, prefix, got)
			}
		}
		if got := enc.SpellTail(whole, ""); got != "" {
			t.Errorf("%s: an empty tail spells %q", spec, got)
		}
	}
	enc := mustEncoding(t, "phone:3:1")
	enc.Units("the ca") // the tokenizer reads "ca" as K AH1, and remembers it
	if got := enc.SpellTail("DH AH0 # K AH1 T # S AE1 T", "T # S AE1 T"); got != "cut sat" {
		t.Errorf("a word the prefix began: %q", got)
	}
	if got := enc.SpellTail("DH AH0 # K AE1 T", "S AE1 T"); got != "sat" {
		t.Errorf("a tail that does not end the text: %q", got)
	}
	if got := mustEncoding(t, "char:3:1").SpellTail("the cat sat", " sat"); got != " sat" {
		t.Errorf("letters: %q", got)
	}
}

// The records a model of sounds answers with carry the words they spell, keyed
// as Python keys them; any other encoding's records are left as they are.
func TestSpelledRecords(t *testing.T) {
	enc := mustEncoding(t, "phone:3:1")
	whole := "DH AH0 # K AE1 T # S AE1 T"
	if got := SpelledPrediction(enc, whole, "# S AE1 T"); !reflect.DeepEqual(got,
		map[string]any{"spelled": "the cat sat", "spelled_continuation": " sat"}) {
		t.Errorf("prediction: %v", got)
	}
	if got := SpelledPrediction(mustEncoding(t, "char:3:1"), "the cat", "cat"); len(got) != 0 {
		t.Errorf("letters: %v", got)
	}
}
