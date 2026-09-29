package kit

// The turns and the thoughts of a model of sounds as they are written out, with
// the words their sounds spell beside them.  The spelling itself is the
// model's (radixnet.Encoding.Spell and SpellTail), and so is the spelling of a
// prediction (radixnet.SpelledPrediction).

import "github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"

// SpelledThought is a thought's record (Thought.ToDict) with "spelled" beside
// its text, and its questions' too; any other encoding, the record as it is.
func SpelledThought(enc radixnet.Encoding, t *Thought) map[string]any {
	doc := t.ToDict()
	if !enc.Unit.Phonetic() {
		return doc
	}
	doc["spelled"] = enc.Spell(t.Text)
	questions := make([]map[string]any, 0, len(t.Questions))
	for _, q := range t.Questions {
		questions = append(questions, SpelledThought(enc, q))
	}
	doc["questions"] = questions
	return doc
}

// SpelledTurnRecord is a turn of a model of sounds as it is written out: the
// turn, the words it spells ("spelled"), the part of them the search added
// after the context it picked up ("spelled_reply"), and its rethink with the
// thought spelled.
type SpelledTurnRecord struct {
	*Turn
	Spelled      string          `json:"spelled"`
	SpelledReply string          `json:"spelled_reply"`
	Rethink      *spelledRethink `json:"rethink"` // shadows the turn's own
}

type spelledRethink struct {
	*Rethink
	Thought map[string]any `json:"thought"` // shadows the rethink's own
}

// SpelledTurn is a turn as it is written out: with the words it spells beside
// its sounds for a model of sounds (SpelledTurnRecord), the turn itself for any
// other encoding.  Python's spelled_turn.
func SpelledTurn(enc radixnet.Encoding, t *Turn) any {
	if !enc.Unit.Phonetic() {
		return t
	}
	out := &SpelledTurnRecord{Turn: t, Spelled: enc.Spell(t.Text), SpelledReply: enc.SpellTail(t.Text, t.Reply)}
	if t.Rethink != nil {
		out.Rethink = &spelledRethink{Rethink: t.Rethink}
		if t.Rethink.Thought != nil {
			out.Rethink.Thought = SpelledThought(enc, t.Rethink.Thought)
		}
	}
	return out
}

// SpelledTurns is every turn as SpelledTurn writes it out.
func SpelledTurns(enc radixnet.Encoding, turns []*Turn) []any {
	out := make([]any, 0, len(turns))
	for _, t := range turns {
		out = append(out, SpelledTurn(enc, t))
	}
	return out
}
