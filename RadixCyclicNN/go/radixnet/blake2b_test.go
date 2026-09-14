package radixnet

import (
	"encoding/hex"
	"strings"
	"testing"
)

// The digests Python's hashlib produces for the same inputs: the utterance
// token is in the text the model trains on, so the two languages must agree on
// it exactly or a recording teaches two different things.
func TestBlake2bMatchesPython(t *testing.T) {
	for _, row := range []struct {
		input  string
		size   int
		digest string
	}{
		{"example", 4, "0da00195"},
		{"the cat sat on the mat", 4, "fa566b3b"},
		{"", 4, "1271cf25"},
		{strings.Repeat("x", 300), 4, "b12499e1"}, // more than one block
	} {
		got := hex.EncodeToString(Blake2b([]byte(row.input), row.size))
		if got != row.digest {
			t.Errorf("blake2b(%q, %d) = %s, want %s", clipForTest(row.input), row.size, got, row.digest)
		}
	}
}

// The published BLAKE2b-512 vectors, so a mistake in the compression function
// shows up here rather than only as a token that happens to differ.
func TestBlake2bFullDigestVectors(t *testing.T) {
	for input, want := range map[string]string{
		"abc": "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1" +
			"7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
		"": "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419" +
			"d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
	} {
		if got := hex.EncodeToString(Blake2b([]byte(input), 64)); got != want {
			t.Errorf("blake2b(%q, 64) = %s, want %s", input, got, want)
		}
	}
}

func TestBlake2bSizeIsClamped(t *testing.T) {
	if len(Blake2b([]byte("x"), 0)) != 1 {
		t.Error("a size below 1 clamps to 1")
	}
	if len(Blake2b([]byte("x"), 100)) != 64 {
		t.Error("a size above 64 clamps to 64")
	}
}

func TestBlake2bBlockBoundaries(t *testing.T) {
	// 127, 128 and 129 bytes exercise the "last block" bookkeeping
	for _, n := range []int{0, 1, 127, 128, 129, 255, 256} {
		if got := Blake2b([]byte(strings.Repeat("a", n)), 8); len(got) != 8 {
			t.Errorf("%d bytes: digest length %d", n, len(got))
		}
	}
	// exactly one block must not be treated as two
	if a, b := Blake2b([]byte(strings.Repeat("a", 128)), 8), Blake2b([]byte(strings.Repeat("a", 129)), 8); string(a) == string(b) {
		t.Error("128 and 129 bytes must hash differently")
	}
}

func clipForTest(text string) string {
	if len(text) <= 20 {
		return text
	}
	return text[:20] + "…"
}
