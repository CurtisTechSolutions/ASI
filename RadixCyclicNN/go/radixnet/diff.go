package radixnet

// Character diff between what the network wrote and what the teacher
// corrected.  The tutor's marking used to be a verdict on a whole sentence:
// the failed attempt was garbage, the correction gospel, and every edge of
// either path moved by the same amount.  Most of a corrected sentence is
// however word for word what the network wrote - the teacher changes a tense,
// an article, a plural - and punishing the parts that were already right is a
// tax on the trigrams that earned their place.
//
// Edits aligns the two sentences so the model can move only the trigram nodes
// the mistake ran through (Model.Correct).  The alignment is a plain
// longest-common-subsequence diff over characters after the shared prefix and
// suffix have been trimmed: deterministic, dependency-free and cheap at
// sentence length, and written the same way in Python (radixnet/diff.py) so
// both implementations mark the same characters.

// MaxDiffCells caps the alignment: above this many cells (|a| x |b| after
// trimming) the middle is marked changed as a whole instead of aligned.
const MaxDiffCells = 4_000_000

// MinEqualRun: changes any closer than a trigram are one change - no trigram
// fits in the gap, so the same nodes are to blame either way.
const MinEqualRun = Window

// Edit is one step of the alignment: Op is "equal", "replace", "delete" or
// "insert".  The spans are half-open rune ranges, wrong[A0:A1] against
// right[B0:B1]; "delete" is text the network wrote and the teacher struck
// out, "insert" text the teacher added.
type Edit struct {
	Op    string `json:"op"`
	A0    int    `json:"-"`
	A1    int    `json:"-"`
	B0    int    `json:"-"`
	B1    int    `json:"-"`
	Wrong string `json:"wrong"`
	Right string `json:"right"`
}

// Span is a half-open range of rune positions.
type Span struct{ Lo, Hi int }

// Edits aligns wrong against right rune by rune; equal runs included, in order.
func Edits(wrong, right string) []Edit {
	a, b := []rune(wrong), []rune(right)
	n, m := len(a), len(b)
	head := 0
	for head < n && head < m && a[head] == b[head] {
		head++
	}
	tail := 0
	for tail < n-head && tail < m-head && a[n-1-tail] == b[m-1-tail] {
		tail++
	}
	midA, midB := a[head:n-tail], b[head:m-tail]
	out := []Edit{}
	if head > 0 {
		out = append(out, newEdit("equal", 0, head, 0, head, a, b))
	}
	switch {
	case len(midA) == 0 && len(midB) == 0:
	case len(midA) == 0:
		out = append(out, newEdit("insert", head, head, head, m-tail, a, b))
	case len(midB) == 0:
		out = append(out, newEdit("delete", head, n-tail, head, head, a, b))
	case len(midA)*len(midB) > MaxDiffCells: // too big to align: one change covering the middle
		out = append(out, newEdit("replace", head, n-tail, head, m-tail, a, b))
	default:
		out = append(out, alignRunes(midA, midB, head, head, a, b)...)
	}
	if tail > 0 {
		out = append(out, newEdit("equal", n-tail, n, m-tail, m, a, b))
	}
	return coalesceEdits(mergeEdits(out, a, b), a, b)
}

// ChangedSpans is where the two sentences disagree: the ranges of wrong, then
// of right.  An insertion is an empty span on the side that lacks the text,
// kept at the position where it belongs: the model blames the step that
// walked past it.
func ChangedSpans(wrong, right string) ([]Span, []Span) {
	left, other := []Span{}, []Span{}
	for _, e := range Edits(wrong, right) {
		if e.Op == "equal" {
			continue
		}
		left = append(left, Span{e.A0, e.A1})
		other = append(other, Span{e.B0, e.B1})
	}
	return left, other
}

// DiffSummary is what changed, for a lesson record; limit 0 keeps every change.
func DiffSummary(wrong, right string, limit int) []Edit {
	out := []Edit{}
	for _, e := range Edits(wrong, right) {
		if e.Op != "equal" {
			out = append(out, e)
		}
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func newEdit(op string, a0, a1, b0, b1 int, a, b []rune) Edit {
	return Edit{Op: op, A0: a0, A1: a1, B0: b0, B1: b1, Wrong: string(a[a0:a1]), Right: string(b[b0:b1])}
}

// alignRunes walks the longest common subsequence of two trimmed middles into edits.
func alignRunes(a, b []rune, offA, offB int, full, fullB []rune) []Edit {
	n, m := len(a), len(b)
	// lcs[i][j] = length of the longest common subsequence of a[i:] and b[j:]
	lcs := make([][]int, n+1)
	flat := make([]int, (n+1)*(m+1))
	for i := range lcs {
		lcs[i] = flat[i*(m+1) : (i+1)*(m+1)]
	}
	for i := n - 1; i >= 0; i-- {
		row, next := lcs[i], lcs[i+1]
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				row[j] = next[j+1] + 1
			} else if next[j] >= row[j+1] {
				row[j] = next[j]
			} else {
				row[j] = row[j+1]
			}
		}
	}
	out := []Edit{}
	i, j := 0, 0
	for i < n || j < m {
		if i < n && j < m && a[i] == b[j] {
			startI, startJ := i, j
			for i < n && j < m && a[i] == b[j] {
				i, j = i+1, j+1
			}
			out = append(out, newEdit("equal", offA+startI, offA+i, offB+startJ, offB+j, full, fullB))
			continue
		}
		startI, startJ := i, j
		// the same tie-break as the Python implementation: a deletion first when both are equally good
		for (i < n || j < m) && !(i < n && j < m && a[i] == b[j]) {
			if i < n && (j >= m || lcs[i+1][j] >= lcs[i][j+1]) {
				i++
			} else {
				j++
			}
		}
		op := "insert"
		switch {
		case i > startI && j > startJ:
			op = "replace"
		case i > startI:
			op = "delete"
		}
		out = append(out, newEdit(op, offA+startI, offA+i, offB+startJ, offB+j, full, fullB))
	}
	return out
}

// mergeEdits joins neighbouring edits of the same kind (the trimming can split a run in two).
func mergeEdits(items []Edit, a, b []rune) []Edit {
	out := []Edit{}
	for _, edit := range items {
		if len(out) > 0 && joinsEdit(out[len(out)-1], edit) {
			last := out[len(out)-1]
			op := last.Op
			if op != edit.Op {
				op = "replace"
			}
			out[len(out)-1] = newEdit(op, last.A0, edit.A1, last.B0, edit.B1, a, b)
			continue
		}
		out = append(out, edit)
	}
	return out
}

// coalesceEdits swallows equal runs shorter than a trigram between two changes
// ("mat" -> "park", not "m" -> "p" and "t" -> "rk").
func coalesceEdits(items []Edit, a, b []rune) []Edit {
	out := []Edit{}
	for _, edit := range items {
		n := len(out)
		if n >= 2 && edit.Op != "equal" && out[n-1].Op == "equal" && out[n-2].Op != "equal" &&
			out[n-1].A1-out[n-1].A0 < MinEqualRun {
			before := out[n-2]
			op := before.Op
			if edit.A1 > before.A0 && edit.B1 > before.B0 {
				op = "replace"
			}
			out = append(out[:n-2], newEdit(op, before.A0, edit.A1, before.B0, edit.B1, a, b))
			continue
		}
		out = append(out, edit)
	}
	return out
}

func joinsEdit(last, edit Edit) bool {
	if last.A1 != edit.A0 || last.B1 != edit.B0 {
		return false
	}
	if last.Op == edit.Op {
		return true
	}
	return last.Op != "equal" && edit.Op != "equal" // delete + insert = replace
}

// spansTouch: does the half-open range [lo, hi) meet any changed span?  An
// empty span is an insertion point: the characters belong on the other side,
// so the step that walked straight past the position is the one at fault.
func spansTouch(lo, hi int, spans []Span) bool {
	for _, s := range spans {
		end := s.Hi
		if end == s.Lo {
			end = s.Lo + 1
		}
		if s.Lo < hi && lo < end {
			return true
		}
	}
	return false
}
