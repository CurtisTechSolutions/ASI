package radixnet

import (
	"math"
	"testing"
)

// trainedPair is a model taught the two sentences the traversal tests turn on.
func trainedPair(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(0, DefaultGraphOptions())
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	m.Workers, m.Exact = 1, true
	m.G.Workers = 1
	texts := []string{"the cat sat on the mat", "the cat sat on the log"}
	if _, err := m.Train(texts, TrainOptions{Epochs: 3, AutoCompress: true}); err != nil {
		t.Fatalf("train: %v", err)
	}
	return m
}

func TestParseTraversal(t *testing.T) {
	for name, want := range map[string]Traversal{
		"": ByReward, "reward": ByReward, "Rewards": ByReward,
		"least-punished": ByLeastPunished, "LEAST_PUNISHED": ByLeastPunished, "blame": ByLeastPunished,
	} {
		got, err := ParseTraversal(name)
		if err != nil || got != want {
			t.Fatalf("ParseTraversal(%q) = %v, %v; want %v", name, got, err, want)
		}
	}
	if _, err := ParseTraversal("sideways"); err == nil {
		t.Fatal("an unknown traversal should be an error")
	}
	if got := ByLeastPunished.String(); got != "least-punished" {
		t.Fatalf("String() = %q", got)
	}
}

// With nothing punished every step ties at zero blame, and the new traversal
// has to be the old one - the same paths, the same costs, the same effort.
func TestLeastPunishedIsTheOldSearchUntilSomethingIsPunished(t *testing.T) {
	m := trainedPair(t)
	for _, prefix := range []string{"the ", "the cat", "sat on"} {
		a, err := m.Predict(prefix, PredictOptions{Length: 8, K: 3, Mode: "beam"})
		if err != nil {
			t.Fatalf("predict: %v", err)
		}
		b, err := m.Predict(prefix, PredictOptions{Length: 8, K: 3, Mode: "beam", Traversal: "least-punished"})
		if err != nil {
			t.Fatalf("predict: %v", err)
		}
		if a.FullText != b.FullText || a.Cost != b.Cost || a.Expanded != b.Expanded {
			t.Fatalf("prefix %q: %q/%v/%d vs %q/%v/%d", prefix, a.FullText, a.Cost, a.Expanded, b.FullText, b.Cost, b.Expanded)
		}
	}
}

// The whole point: a step that was judged wrong once is not walked again, even
// when it was rewarded five times over and is by far the cheapest way on.
func TestLeastPunishedLeavesAJudgedStep(t *testing.T) {
	m := trainedPair(t)
	good := []string{"the cat sat on the mat"}
	if _, err := m.Reward(good, 1, 5.0); err != nil {
		t.Fatalf("reward: %v", err)
	}
	if _, err := m.Punish(good, 1, 1.0); err != nil {
		t.Fatalf("punish: %v", err)
	}
	const prefix = "the cat sat on the "
	byReward, err := m.Predict(prefix, PredictOptions{Length: 6, K: 3, Mode: "beam"})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	byBlame, err := m.Predict(prefix, PredictOptions{Length: 6, K: 3, Mode: "beam", Traversal: "least-punished"})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if byReward.FullText != "the cat sat on the mat" {
		t.Fatalf("the reward search should take the rewarded step, got %q", byReward.FullText)
	}
	if byBlame.FullText != "the cat sat on the log" {
		t.Fatalf("the least-punished search should leave the blamed step, got %q", byBlame.FullText)
	}
	if byBlame.Cost <= byReward.Cost {
		t.Fatalf("the unpunished walk should be the dearer one: %v vs %v", byBlame.Cost, byReward.Cost)
	}
	if byBlame.Punish != 0 {
		t.Fatalf("the walk it took carries blame: %v", byBlame.Punish)
	}
	if byBlame.Traversal != "least-punished" {
		t.Fatalf("the prediction does not say which search wrote it: %q", byBlame.Traversal)
	}
	// the blamed walk is still on offer, ranked behind everything unpunished
	found := false
	for _, r := range byBlame.Top {
		if r.FullText == "the cat sat on the mat" && r.Punish > 0 {
			found = true
		}
	}
	if !found {
		t.Fatal("the blamed walk should still be offered, marked with its blame")
	}
}

// A reward is not an eraser: the edge's own penalty is netted off by one, but
// what a judged walk did is counted on its own and stays counted.
func TestPunishmentIsNotBoughtOff(t *testing.T) {
	m := trainedPair(t)
	good := []string{"the cat sat on the mat"}
	if _, err := m.Punish(good, 1, 1.0); err != nil {
		t.Fatalf("punish: %v", err)
	}
	if _, err := m.Reward(good, 1, 50.0); err != nil {
		t.Fatalf("reward: %v", err)
	}
	m.G.Prepare()
	judged, punished := 0, 0
	for _, row := range m.G.PathContexts(0, -1) {
		if row.Incorrect == 0 {
			continue
		}
		judged++
		if m.G.StepPunishment(row.Prev, row.Edge) > 0 {
			punished++
		}
		if m.G.EdgePunishment(row.Edge) != 0 {
			t.Fatalf("the edge's own reward is positive now, so its penalty should be 0")
		}
	}
	if judged == 0 {
		t.Fatal("the punished pass recorded no context")
	}
	if punished != judged {
		t.Fatalf("%d of %d judged steps lost their blame to a reward", judged-punished, judged)
	}
}

// LeastPunished keeps the cleanest children and nothing else; an untouched node
// comes back untouched.
func TestLeastPunishedFilter(t *testing.T) {
	costs := []ChildCost{{Child: 4, Edge: 0, Cost: 1}, {Child: 5, Edge: 1, Cost: 2}}
	if got := LeastPunished(costs); len(got) != 2 {
		t.Fatalf("nothing is punished, so nothing should be dropped: %v", got)
	}
	costs[0].Punish = 0.5
	got := LeastPunished(costs)
	if len(got) != 1 || got[0].Child != 5 {
		t.Fatalf("the punished child should be gone: %v", got)
	}
	costs[1].Punish = 0.5 + PunishTolerance/2 // the same penalty, a bit or two apart
	if got := LeastPunished(costs); len(got) != 2 {
		t.Fatalf("punishments within the tolerance are the same punishment: %v", got)
	}
	costs[1].Punish = math.Inf(1)
	if got := LeastPunished(costs); len(got) != 1 || got[0].Child != 4 {
		t.Fatalf("an infinitely punished child should be gone: %v", got)
	}
}
