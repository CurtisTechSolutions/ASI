package kit

import (
	"encoding/json"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// mustEncoding is the model's own test helper (phonetic_test.go).
func mustEncoding(t *testing.T, spec string) radixnet.Encoding {
	t.Helper()
	enc, err := radixnet.ParseEncoding(spec)
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

// The turns and the thoughts a model of sounds writes out carry the words they
// spell, keyed as Python keys them; any other encoding's are left as they are.
func TestSpelledTurnsAndThoughts(t *testing.T) {
	enc := mustEncoding(t, "phone:3:1")
	whole := "DH AH0 # K AE1 T # S AE1 T"
	thought := &Thought{Text: "DH AH0 # K AE1 T", Questions: []*Thought{{Text: "S AE1 T ."}}}
	doc := SpelledThought(enc, thought)
	if doc["spelled"] != "the cat" || doc["questions"].([]map[string]any)[0]["spelled"] != "sat." {
		t.Errorf("thought: %v", doc)
	}
	turn := &Turn{Index: 1, Text: whole, Context: "the cat", Reply: "# S AE1 T",
		Rethink: &Rethink{Kind: "repeat", Taught: -1, Thought: &Thought{Text: "K AE1 T"}}}
	data, err := json.Marshal(SpelledTurn(enc, turn))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	rethink := got["rethink"].(map[string]any)
	if got["spelled"] != "the cat sat" || got["spelled_reply"] != " sat" || got["text"] != whole ||
		rethink["kind"] != "repeat" || rethink["thought"].(map[string]any)["spelled"] != "cat" {
		t.Errorf("turn: %s", data)
	}
	// a turn without a rethink writes a null one, as the plain turn does
	data, _ = json.Marshal(SpelledTurn(enc, &Turn{Text: whole, Reply: whole}))
	if err := json.Unmarshal(data, &got); err != nil || got["rethink"] != nil || got["spelled_reply"] != "the cat sat" {
		t.Errorf("no rethink: %s", data)
	}
	if SpelledTurn(mustEncoding(t, "char:3:1"), turn) != any(turn) {
		t.Error("a turn of letters is written as it is")
	}
}
