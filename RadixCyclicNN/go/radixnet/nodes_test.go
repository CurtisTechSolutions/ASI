package radixnet

import (
	"math"
	"testing"
)

// branching trains a corpus with a real choice in it and corrects one answer,
// so one arm of the branch is rewarded and the other punished.
func branching(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	corpus := []string{"the cat sat on the mat", "a cat ran to the park", "the cat sat on the log"}
	if _, err := m.Train(corpus, TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Correct("the cat ran to the mat", "the cat sat on the mat", DefaultCorrectOptions()); err != nil {
		t.Fatal(err)
	}
	return m
}

// branchNode is the node the two arms leave from.
func branchNode(t *testing.T, m *Model) int {
	t.Helper()
	for node := 0; node < m.G.NumNodeIDs(); node++ {
		if m.G.Label(node) == "at " {
			if len(m.G.Children(node)) < 2 {
				t.Fatalf("node %q does not branch", "at ")
			}
			return node
		}
	}
	t.Fatalf("no node labelled %q", "at ")
	return -1
}

func sideOf(rows []NeighbourStats, label string) *NeighbourStats {
	for i := range rows {
		if rows[i].Label == label {
			return &rows[i]
		}
	}
	return nil
}

func TestEachSideSharesOutTheTrafficItCarried(t *testing.T) {
	m := branching(t)
	row := m.G.NodeRatios(branchNode(t, m))
	if row == nil {
		t.Fatal("the branch node has no ratios")
	}
	for _, side := range []struct {
		name   string
		rows   []NeighbourStats
		totals SideTotals
	}{{"from", row.From, row.InTotals}, {"to", row.To, row.OutTotals}} {
		if len(side.rows) == 0 {
			t.Fatalf("%s: no neighbours", side.name)
		}
		shares, seen := 0.0, int64(0)
		for i, r := range side.rows {
			shares += r.SeenRatio
			seen += r.Seen
			if i > 0 && side.rows[i-1].Seen < r.Seen {
				t.Errorf("%s: rows are not most walked first: %d before %d", side.name, side.rows[i-1].Seen, r.Seen)
			}
		}
		if math.Abs(shares-1) > 1e-9 {
			t.Errorf("%s: the shares add up to %g, want 1", side.name, shares)
		}
		if seen != side.totals.Seen || len(side.rows) != side.totals.Edges {
			t.Errorf("%s: totals say %d over %d edges, the rows say %d over %d",
				side.name, side.totals.Seen, side.totals.Edges, seen, len(side.rows))
		}
	}
}

func TestTheRewardShareIsSignedAndAddsUpToTheSide(t *testing.T) {
	m := branching(t)
	row := m.G.NodeRatios(branchNode(t, m))
	rewarded, punished, mass := 0, 0, 0.0
	for _, r := range row.To {
		mass += math.Abs(r.RewardRatio)
		switch {
		case r.Reward > 0:
			rewarded++
			if r.RewardRatio <= 0 {
				t.Errorf("%q earned %g but its share is %g", r.Label, r.Reward, r.RewardRatio)
			}
		case r.Reward < 0:
			punished++
			if r.RewardRatio >= 0 {
				t.Errorf("%q was penalised %g but its share is %g", r.Label, r.Reward, r.RewardRatio)
			}
		}
	}
	if rewarded == 0 || punished == 0 {
		t.Fatalf("the correction should have moved one arm each way, got %d rewarded / %d punished", rewarded, punished)
	}
	if math.Abs(mass-1) > 1e-9 {
		t.Errorf("the reward shares have magnitude %g, want 1", mass)
	}
}

func TestTheJudgedPathsLandOnTheWayTheyWereWalked(t *testing.T) {
	m := branching(t)
	row := m.G.NodeRatios(branchNode(t, m))
	right, wrong := sideOf(row.To, "t sat"), sideOf(row.To, "t ran ")
	if right == nil || wrong == nil {
		t.Fatalf("the branch lost an arm: %+v", row.To)
	}
	if right.Correct != 1 || right.Incorrect != 0 {
		t.Errorf("the rewarded arm counted %d correct / %d incorrect, want 1 / 0", right.Correct, right.Incorrect)
	}
	if wrong.Correct != 0 || wrong.Incorrect != 1 {
		t.Errorf("the punished arm counted %d correct / %d incorrect, want 0 / 1", wrong.Correct, wrong.Incorrect)
	}
	if wrong.PathRatio == nil || *wrong.PathRatio != float64(wrong.PathSeen)/float64(wrong.Seen) {
		t.Errorf("path_ratio is %v, want %d/%d", wrong.PathRatio, wrong.PathSeen, wrong.Seen)
	}
	if row.OutTotals.Correct != 2 || row.OutTotals.Incorrect != 1 {
		t.Errorf("the out side counted %d / %d, want 2 / 1", row.OutTotals.Correct, row.OutTotals.Incorrect)
	}
	if row.OutTotals.CorrectRatio == nil || math.Abs(*row.OutTotals.CorrectRatio-2.0/3.0) > 1e-9 {
		t.Errorf("the out side's correct ratio is %v, want 2/3", row.OutTotals.CorrectRatio)
	}
}

func TestAnUnjudgedGraphHasTheSharesButNoVerdicts(t *testing.T) {
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	if _, err := m.Train([]string{"the cat sat on the mat", "a cat ran to the park"},
		TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
		t.Fatal(err)
	}
	rows := m.G.NodeRatioRows(0, -1)
	if len(rows) == 0 {
		t.Fatal("no nodes")
	}
	for _, row := range rows {
		for _, side := range [][]NeighbourStats{row.From, row.To} {
			for _, r := range side {
				if r.Correct != 0 || r.Incorrect != 0 || r.PathSeen != 0 {
					t.Errorf("%q carries a verdict nothing judged: %+v", r.Label, r)
				}
				if r.CorrectRatio != nil || r.RewardRatio != 0 {
					t.Errorf("%q has a ratio nothing earned: %+v", r.Label, r)
				}
			}
		}
		if row.InTotals.CorrectRatio != nil {
			t.Errorf("%q has an in-side correct ratio with nothing judged", row.Label)
		}
	}
}

func TestTheNodeTableIsMostVisitedFirst(t *testing.T) {
	m := branching(t)
	rows := m.G.NodeRatioRows(3, -1)
	if len(rows) != 3 {
		t.Fatalf("asked for 3 nodes, got %d", len(rows))
	}
	for i := 1; i < len(rows); i++ {
		if rows[i-1].Visits < rows[i].Visits {
			t.Errorf("row %d (%d visits) comes before row %d (%d visits)", i-1, rows[i-1].Visits, i, rows[i].Visits)
		}
	}
	node := branchNode(t, m)
	one := m.G.NodeRatioRows(0, node)
	if len(one) != 1 || one[0].Node != node {
		t.Errorf("asking for one node gave %+v", one)
	}
	if m.G.NodeRatios(m.G.NumNodeIDs()+5) != nil {
		t.Error("a node that does not exist has ratios")
	}
}
