package kit

import "unicode/utf8"

// Two small helpers the model keeps for itself (radixnet's encoding.go and
// model.go), copied here rather than exported, so that the model's API stays
// the model.

// runeLen is the character (code point) length of a string, the unit the
// Python implementation measures text in.
func runeLen(s string) int { return utf8.RuneCountInString(s) }

// toFloat is a number read from a decoded JSON value or a record: float64, int
// or int64, and 0 for anything else.
func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	}
	return 0
}
