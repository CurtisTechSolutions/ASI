package radixnet

import (
	"math"
	"strings"
	"testing"
)

// The punishment traversal: the same searches, priced by what the network was
// punished for instead of by what it was rewarded for.  The Python side's
// tests/test_penalty.py is the same contract.

var traversalTexts = []string{
	"the cat sat on the mat",
	"the cat ate the rat",
	"the cat ran up the hill",
}

const (
	praised  = "the cat sat on the mat"
	punished = "the cat ate the rat"
)

// traversalModel builds a model with one branch praised, one punished and one left alone.
func traversalModel(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	opts := DefaultTrainOptions()
	opts.Epochs = 4
	if _, err := m.Train(traversalTexts, opts); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Reward([]string{praised}, 1, 4.0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Punish([]string{punished}, 1, 4.0); err != nil {
		t.Fatal(err)
	}
	m.G.Prepare()
	return m
}

// forkNode is the node where the texts part company (the one with the most children).
func forkNode(g *Graph) int {
	best, most := -1, 0
	for p := range g.Labels {
		if g.Alive[p] && g.children[p].size() > most {
			best, most = p, g.children[p].size()
		}
	}
	return best
}

func costsByChild(costs []ChildCost) map[int]float64 {
	out := make(map[int]float64, len(costs))
	for _, cc := range costs {
		out[cc.Child] = cc.Cost
	}
	return out
}

func TestResolveTraversal(t *testing.T) {
	for _, given := range []string{"", "reward", " REWARD "} {
		if got, err := ResolveTraversal(given); err != nil || got != TraversalReward {
			t.Fatalf("ResolveTraversal(%q) = %q, %v", given, got, err)
		}
	}
	if got, _ := ResolveTraversal("Punishment"); got != TraversalPunishment {
		t.Fatalf("punishment not recognised: %q", got)
	}
	if _, err := ResolveTraversal("penalty"); err == nil {
		t.Fatal("an unknown traversal must be refused")
	}
}

func TestChildEvidenceSplitsTheReward(t *testing.T) {
	g := traversalModel(t).G
	node := forkNode(g)
	positive, negative := false, false
	for _, ev := range g.ChildEvidence(node, -1) {
		reward := g.EdgeReward[ev.Edge]
		if want := g.EdgeW[ev.Edge] - g.RewardScale*reward; math.Abs(ev.Merit-want) > 1e-12 {
			t.Fatalf("merit %v, want the weight with the reward taken out (%v)", ev.Merit, want)
		}
		if want := g.RewardScale * math.Max(0, -reward); math.Abs(ev.Penalty-want) > 1e-12 {
			t.Fatalf("penalty %v, want %v", ev.Penalty, want)
		}
		if ev.Penalty < 0 {
			t.Fatalf("a penalty is never negative, got %v", ev.Penalty)
		}
		positive = positive || reward > 0
		negative = negative || reward < 0
	}
	if !positive || !negative {
		t.Fatal("the fixture must have both a praised and a punished edge at the fork")
	}
}

func TestPunishmentCostsDropTheRewards(t *testing.T) {
	m := traversalModel(t)
	g := m.G
	node := forkNode(g)
	pc, err := g.PenaltyCosts(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	reward := costsByChild(g.ChildCosts(node))
	punish := costsByChild(pc.Costs(node, -1))
	var good, plain, bad = -1, -1, -1
	for i, c := range g.children[node].order {
		switch e := g.children[node].edges[i]; {
		case g.EdgeReward[e] > 0:
			good = c
		case g.EdgeReward[e] < 0:
			bad = c
		default:
			plain = c
		}
	}
	if good < 0 || plain < 0 || bad < 0 {
		t.Skip("the fork has no praised / neutral / punished trio to compare")
	}
	bought := reward[plain] - reward[good]
	if bought <= 3 {
		t.Fatalf("the reward traversal should buy the praised child a large discount, got %v", bought)
	}
	if left := punish[plain] - punish[good]; left >= bought-3 {
		t.Fatalf("the punishment traversal still honours the reward: %v of %v left", left, bought)
	}
	if punish[bad] <= punish[plain]+1 {
		t.Fatalf("the punished child must be the dear one: %v vs %v", punish[bad], punish[plain])
	}
}

func TestPurePunishmentIsIndifferentBetweenUnpunishedChildren(t *testing.T) {
	g := traversalModel(t).G
	node := forkNode(g)
	pc, err := g.PenaltyCosts(1, 0) // merit scale 0: nothing but the punishments decides
	if err != nil {
		t.Fatal(err)
	}
	var seen []float64
	for i, c := range g.children[node].order {
		if g.EdgeReward[g.children[node].edges[i]] >= 0 {
			seen = append(seen, costsByChild(pc.Costs(node, -1))[c])
		}
	}
	if len(seen) < 2 {
		t.Skip("nothing to compare")
	}
	for _, cost := range seen[1:] {
		if math.Abs(cost-seen[0]) > 1e-12 {
			t.Fatalf("unpunished children priced differently: %v", seen)
		}
	}
}

func TestPunishmentCostsAreStillADistribution(t *testing.T) {
	g := traversalModel(t).G
	pc, err := g.PenaltyCosts(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	for p := range g.Labels {
		if !g.Alive[p] || g.children[p].size() == 0 {
			continue
		}
		total := 0.0
		for _, cc := range pc.Costs(p, -1) {
			if cc.Cost < -1e-12 {
				t.Fatalf("node %d: a cost must be >= 0, got %v", p, cc.Cost)
			}
			total += math.Exp(-cc.Cost)
		}
		if math.Abs(total-1) > 1e-9 {
			t.Fatalf("node %d: the probabilities sum to %v", p, total)
		}
	}
}

func TestNegativeScalesAreRefused(t *testing.T) {
	g := traversalModel(t).G
	if _, err := g.PenaltyCosts(-1, 1); err == nil {
		t.Fatal("a negative penalty_scale must be refused")
	}
	if _, err := g.PenaltyCosts(1, -1); err == nil {
		t.Fatal("a negative merit_scale must be refused")
	}
}

func TestTheRewardTraversalCostsNothingExtra(t *testing.T) {
	g := traversalModel(t).G
	fn, err := g.TraversalCosts("reward", 1, 1)
	if err != nil || fn != nil {
		t.Fatalf("the reward traversal must read the graph's own costs, got %v, %v", fn, err)
	}
	if fn, err = g.TraversalCosts("punishment", 1, 1); err != nil || fn == nil {
		t.Fatalf("the punishment traversal must bring its own costs: %v", err)
	}
	if _, err = g.TraversalCosts("nonsense", 1, 1); err == nil {
		t.Fatal("an unknown traversal must be refused")
	}
}

func TestPredictAndGenerateTakeTheTraversal(t *testing.T) {
	m := traversalModel(t)
	o := DefaultPredictOptions()
	o.Length, o.K = 16, 3
	byReward, err := m.Predict("the cat", o)
	if err != nil {
		t.Fatal(err)
	}
	o.Traversal, o.MeritScale = TraversalPunishment, 0
	byPunishment, err := m.Predict("the cat", o)
	if err != nil {
		t.Fatal(err)
	}
	if byReward.Traversal != TraversalReward || byPunishment.Traversal != TraversalPunishment {
		t.Fatalf("the prediction must say which traversal ran: %q / %q", byReward.Traversal, byPunishment.Traversal)
	}
	if byReward.Text == byPunishment.Text {
		t.Fatalf("both traversals wrote %q; the punished branch should have moved", byReward.Text)
	}
	if strings.Contains(byPunishment.Text, "ate the rat") {
		t.Fatalf("the punishment traversal walked into the corrected branch: %q", byPunishment.Text)
	}
	for _, mode := range []string{"beam", "sample"} {
		o.Mode = mode
		if _, err := m.Predict("the cat", o); err != nil {
			t.Fatalf("mode %s: %v", mode, err)
		}
	}
	for _, mode := range []string{"beam", "sample", "dijkstra"} {
		g := DefaultGenerateOptions()
		g.Mode, g.Count, g.MaxLength = mode, 2, 40
		g.Traversal = TraversalPunishment
		if out, err := m.Generate(g); err != nil || len(out) == 0 {
			t.Fatalf("generate %s: %v (%d texts)", mode, err, len(out))
		}
	}
}

func TestAnUnknownTraversalIsRefusedByPredict(t *testing.T) {
	m := traversalModel(t)
	o := DefaultPredictOptions()
	o.Traversal = "least-blame"
	if _, err := m.Predict("the cat", o); err == nil {
		t.Fatal("predict must refuse an unknown traversal")
	}
}

func TestTheNegativeNetworkWalksItsLeastBlamedWay(t *testing.T) {
	m, err := NewNegativeModel(1, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Blame([]string{punished}, BlameOptions{Reason: "nonsense", Severity: 8, Epochs: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Blame([]string{praised}, BlameOptions{Reason: "nonsense", Severity: 1, Epochs: 1}); err != nil {
		t.Fatal(err)
	}
	g := m.G
	g.Prepare()
	for p := range g.Labels {
		if !g.Alive[p] || g.children[p].size() == 0 {
			continue
		}
		for _, ev := range g.ChildEvidence(p, -1) {
			if want := math.Log1p(g.Evidence(ev.Edge)); math.Abs(ev.Penalty-want) > 1e-12 {
				t.Fatalf("the negative network's penalty is its net blame: %v, want %v", ev.Penalty, want)
			}
			if ev.Merit < 0 {
				t.Fatalf("cleared text is never negative merit, got %v", ev.Merit)
			}
		}
	}
	o := DefaultPredictOptions()
	o.Length, o.K = 14, 2
	likeliest, err := m.Predict("the cat", o)
	if err != nil {
		t.Fatal(err)
	}
	o.Traversal = TraversalPunishment
	leastBlamed, err := m.Predict("the cat", o)
	if err != nil {
		t.Fatal(err)
	}
	if likeliest.Text == leastBlamed.Text {
		t.Fatalf("both traversals wrote %q through the failure structure", likeliest.Text)
	}
}
