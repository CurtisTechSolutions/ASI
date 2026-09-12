// Package radixnet is the Go port of the count / reward model of RadixCyclicNN:
// a self-compressing cyclic trigram graph whose edge weights are a dual
// frequency function of traversal counts (all time and inside a sliding
// window) plus rewards, with beam-search prediction (top-K and bottom-K
// continuations), generation, scoring, 2NRL feedback and a self-conversation.
//
// Model files are interchangeable with the Python implementation
// (format "radixnet-count"): both sides read and write the same JSON layout,
// including the Mersenne Twister state, so a model trained here continues in
// Python and vice versa with identical numbers.
//
// Concurrency: training fans goroutines out over the texts (lines, paragraphs
// or pages of a file) for encoding, tracing and counting (lock-free atomic
// increments), recomputes weights and edge costs in parallel over the nodes,
// and runs the two beams of a prediction side by side.  Only structural
// changes (splits and merges of nodes) take a write lock.
package radixnet

import "unicode/utf8"

// Window is the sliding-window length of the encoding (trigrams).
const Window = 3

// Overlap is the number of characters two consecutive labels share.
const Overlap = Window - 1

// StartLabel and EndLabel are the labels of the two sentinel nodes.
const (
	StartLabel = "<s>"
	EndLabel   = "</s>"
)

// runeLen is the character (code point) length of a string, the unit the
// Python implementation measures text in.
func runeLen(s string) int { return utf8.RuneCountInString(s) }

// runeSlice returns s[from:to] in characters (code points); to < 0 means "to the end".
func runeSlice(s string, from, to int) string {
	if from <= 0 && to < 0 {
		return s
	}
	i := 0
	start, end := -1, len(s)
	for bi := range s {
		if i == from {
			start = bi
		}
		if to >= 0 && i == to {
			end = bi
			break
		}
		i++
	}
	if start < 0 {
		if from == i { // from == len(s) in characters
			return ""
		}
		return ""
	}
	if end < start {
		return ""
	}
	return s[start:end]
}

// Encode returns the overlapping windows of text (stride 1); nil when the text
// is shorter than Window characters: "hello" -> ["hel", "ell", "llo"].
func Encode(text string) []string {
	runes := []rune(text)
	n := len(runes) - Window + 1
	if n <= 0 {
		return nil
	}
	grams := make([]string, n)
	for i := 0; i < n; i++ {
		grams[i] = string(runes[i : i+Window])
	}
	return grams
}

// DecodePath decodes node labels in path order into text.  Every label after
// the first contributes its part beyond the overlap; the first contributes
// label[startOffset:] when includeContext is true, else label[startOffset+Window:]
// (the deterministic remainder of a compressed node after the matched window).
// Sentinel labels are never skipped here: callers strip START / END by id.
func DecodePath(labels []string, startOffset int, includeContext bool) string {
	firstCut := startOffset
	if !includeContext {
		firstCut += Window
	}
	out := make([]byte, 0, 64)
	for i, label := range labels {
		if i == 0 {
			out = append(out, runeSlice(label, firstCut, -1)...)
		} else {
			out = append(out, runeSlice(label, Overlap, -1)...)
		}
	}
	return string(out)
}

// truncateRunes returns the first n characters of s.
func truncateRunes(s string, n int) string {
	if n < 0 {
		return s
	}
	i := 0
	for bi := range s {
		if i == n {
			return s[:bi]
		}
		i++
	}
	return s
}
