package radixnet

import (
	"fmt"
	"testing"
)

func diffPairs() [][2]string {
	return [][2]string{
		{"the cat sit", "the cat sits"},
		{"he go to school", "he goes to school"},
		{"a apple a day", "an apple a day"},
		{"the cats is hungry", "the cats are hungry"},
		{"i have ate", "i have eaten"},
		{"", "hello there"},
		{"nothing to fix", "nothing to fix"},
		{"she walk to the shop yesterday", "she walked to the shop yesterday"},
		{"we was happy", "we were happy"},
		{"the mat on sat cat", "the cat sat on the mat"},
	}
}

func TestEditsAlignsSentences(t *testing.T) {
	for _, pair := range diffPairs() {
		wrong, right := pair[0], pair[1]
		edits := Edits(wrong, right)
		a, b := "", ""
		for _, e := range edits {
			if e.Op == "equal" && e.Wrong != e.Right {
				t.Fatalf("%q -> %q: an equal run must hold the same text: %+v", wrong, right, e)
			}
			a += e.Wrong
			b += e.Right
		}
		if a != wrong || b != right { // the edits rebuild both sides exactly
			t.Fatalf("%q -> %q rebuilt as %q / %q", wrong, right, a, b)
		}
		for i := 1; i < len(edits); i++ {
			if edits[i-1].A1 != edits[i].A0 || edits[i-1].B1 != edits[i].B0 {
				t.Fatalf("%q -> %q: the spans must run end to end: %+v", wrong, right, edits)
			}
			if edits[i-1].Op == edits[i].Op {
				t.Fatalf("%q -> %q: neighbouring edits of one kind must be merged: %+v", wrong, right, edits)
			}
		}
	}
	if changes := DiffSummary("nothing to fix", "nothing to fix", 0); len(changes) != 0 {
		t.Fatalf("identical sentences have no changes: %v", changes)
	}
	changes := DiffSummary("the cats is hungry", "the cats are hungry", 0)
	if len(changes) != 1 || changes[0].Op != "replace" || changes[0].Wrong != "is" || changes[0].Right != "are" {
		t.Fatalf("one word replaced: %+v", changes)
	}
	wrongSpans, rightSpans := ChangedSpans("the cat sit", "the cat sits")
	if len(wrongSpans) != 1 || wrongSpans[0] != (Span{11, 11}) || rightSpans[0] != (Span{11, 12}) {
		t.Fatalf("an insertion is an empty span where it belongs: %v %v", wrongSpans, rightSpans)
	}
}

// The point of the whole exercise: only the trigram nodes the two sentences
// disagree on move, and the words they share keep what they earned.
func TestCorrectMovesOnlyTheDifference(t *testing.T) {
	m := trained(t, 2, 1)
	if _, err := m.Train([]string{"the cat sits on the mat", "the dogs run in the park"}, TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	wrong, right := "the cat sit on the mat", "the cat sits on the mat"
	before := append([]float64(nil), m.G.EdgeReward...)
	out, err := m.Correct(wrong, right, CorrectOptions{Strength: 1, Weight: 1, Reward: 1, Keep: 0})
	if err != nil {
		t.Fatal(err)
	}
	if out.Edits != 1 || out.Penalised != 1 || out.Rewarded != 1 || out.Kept != 0 {
		t.Fatalf("one insertion, one step blamed, one step taught: %+v", out)
	}
	moved := map[int]float64{}
	for e := range m.G.EdgeReward {
		was := 0.0
		if e < len(before) {
			was = before[e]
		}
		if delta := m.G.EdgeReward[e] - was; delta != 0 {
			moved[e] = delta
		}
	}
	if len(moved) != 2 {
		t.Fatalf("keep=0 moves exactly the two steps that differ, moved %v", moved)
	}
	penalised, rewarded := 0, 0
	for _, delta := range moved {
		if delta < 0 {
			penalised++
		} else {
			rewarded++
		}
	}
	if penalised != 1 || rewarded != 1 {
		t.Fatalf("one penalty and one reward, got %v", moved)
	}
	// the blamed step is the one that wrote " " where the teacher wrote "s"
	blamed := -1
	for e, delta := range moved {
		if delta < 0 {
			blamed = e
		}
	}
	if want := m.stepsOver(Encode(wrong), runeLen(wrong), []Span{{11, 11}}); len(want) != 1 || want[0] != blamed {
		t.Fatalf("the penalty landed on edge %d, the diff blames %v", blamed, want)
	}
}

// keep spreads a smaller reward over the rest of the correction; count is what a training pass does.
func TestCorrectKeepsAndCounts(t *testing.T) {
	m := trained(t, 2, 1)
	texts := m.MetaInt("trained_texts")
	out, err := m.Correct("the cat sit on the mat", "the cat sits on the mat", DefaultCorrectOptions())
	if err != nil {
		t.Fatal(err)
	}
	if out.Kept == 0 {
		t.Fatalf("the unchanged words of the correction earn Keep: %+v", out)
	}
	if out.Reward <= 0 || out.Penalty <= 0 {
		t.Fatalf("both halves should have moved: %+v", out)
	}
	if got := m.MetaInt("trained_texts") - texts; got != 1 {
		t.Fatalf("the correction is traversed once, trained_texts moved by %d", got)
	}
	if m.MetaInt("feedback_passes") == 0 {
		t.Fatal("a correction is a feedback pass")
	}
	// with nothing to correct there is nothing to blame
	out, err = m.Correct("the cat sits on the mat", "the cat sits on the mat", DefaultCorrectOptions())
	if err != nil {
		t.Fatal(err)
	}
	if out.Edits != 0 || out.Penalised != 0 || out.Rewarded != 0 {
		t.Fatalf("identical sentences change nothing: %+v", out)
	}
}

// A sentence that stopped too early is blamed on the step into END.
func TestCorrectBlamesAnEarlyEnd(t *testing.T) {
	m := trained(t, 2, 1)
	if _, err := m.Train([]string{"the cat sat on the mat"}, TrainOptions{Epochs: 1, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	wrong, right := "the cat sat", "the cat sat on the mat"
	out, err := m.Correct(wrong, right, CorrectOptions{Strength: 1, Weight: 1, Reward: 1, Keep: 0})
	if err != nil {
		t.Fatal(err)
	}
	if out.Penalised != 1 {
		t.Fatalf("the step that ended the sentence is the mistake: %+v", out)
	}
	grams := Encode(wrong)
	path, ok := m.G.NodePath(grams)
	if !ok {
		t.Fatal("the short sentence should walk")
	}
	last, _ := m.G.Edge(path[len(path)-2], End)
	blamed := m.stepsOver(grams, runeLen(wrong), []Span{{runeLen(wrong), runeLen(wrong)}})
	if len(blamed) != 1 || blamed[0] != last {
		t.Fatalf("the blame should fall on the edge into END (%d), got %v", last, blamed)
	}
}

func TestCorrectHandlesShortAndEmptyText(t *testing.T) {
	m := trained(t, 1, 1)
	for _, pair := range [][2]string{{"", ""}, {"ab", "ab"}, {"", "the cat sits"}, {"the cat sits", ""}} {
		if _, err := m.Correct(pair[0], pair[1], DefaultCorrectOptions()); err != nil {
			t.Fatalf("%q -> %q: %v", pair[0], pair[1], err)
		}
	}
	if err := m.G.CheckInvariants(nil, false); err != nil {
		t.Fatal(err)
	}
}

// The graph stays walkable and consistent after a run of corrections.
func TestCorrectKeepsTheGraphSound(t *testing.T) {
	m := trained(t, 2, 1)
	for i := 0; i < 12; i++ {
		wrong := fmt.Sprintf("the cat sit on the mat number %d", i)
		right := fmt.Sprintf("the cat sits on the mat number %d", i)
		if _, err := m.Correct(wrong, right, DefaultCorrectOptions()); err != nil {
			t.Fatal(err)
		}
		if err := m.G.CheckInvariants(nil, false); err != nil {
			t.Fatalf("round %d: %v", i, err)
		}
	}
}
