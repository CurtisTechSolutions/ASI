package radixnet

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var thinkTexts = []string{"ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat", "the cat sat on the mat"}

var thinkThoughts = []string{
	"okay, the cat sat. is that right? yes it did.",
	"the sky is blue because of light",
	"hmm. what did the cat do? it sat on the mat.",
}

// thinker is a small deterministic model with something to talk about and no thoughts yet.
func thinker(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(3, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	opts := DefaultTrainOptions()
	opts.Epochs = 3
	if _, err := m.Train(thinkTexts, opts); err != nil {
		t.Fatal(err)
	}
	return m
}

func quietThought(t *testing.T, m *Model, o ThinkOptions) *Thought {
	t.Helper()
	thought, err := m.Think(o)
	if err != nil {
		t.Fatal(err)
	}
	return thought
}

func TestQuestionsInFindsTheSentencesEndingInAQuestionMark(t *testing.T) {
	cases := []struct {
		text string
		want []Question
	}{
		{"Hmm. Is that right? Yes. Why not?", []Question{{5, "Is that right?"}, {25, "Why not?"}}},
		{"no questions here.", []Question{}},
		{"  really?!  ", []Question{{2, "really?!"}}},
		{"a? b. c?", []Question{{0, "a?"}, {6, "c?"}}},
		{"", []Question{}},
		{"???", []Question{}}, // a terminator run with no sentence before it
		{"hmm, what did the cat do? it sat.", []Question{{0, "hmm, what did the cat do?"}}},
	}
	for _, c := range cases {
		if got := QuestionsIn(c.text); !reflect.DeepEqual(got, c.want) {
			t.Errorf("QuestionsIn(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestPlaceCutsTheNodeSoThePrefixEndsThere(t *testing.T) {
	m := thinker(t)
	g := m.G
	g.Compress()
	node := m.Place("the cat ")
	if node < First {
		t.Fatalf("place gave %d", node)
	}
	// the prefix's last gram is now the node's last: an edge taught from it fires exactly here
	_, offset, _ := m.locate("the cat ")
	if offset+g.Enc.N != g.labelLen[node] {
		t.Fatalf("the gram at offset %d is not the last of %q", offset, g.Labels[node])
	}
	if err := g.CheckInvariants(thinkTexts, false); err != nil {
		t.Fatal(err)
	}
	if m.Place("zzz qqq") != -1 { // a text the graph never saw
		t.Fatal("an unknown text was placed")
	}
	if m.Place("") != -1 {
		t.Fatal("an empty prefix was placed")
	}
}

func TestAModelTaughtNoThoughtsHasNothingToThinkWith(t *testing.T) {
	m := thinker(t)
	thought := quietThought(t, m, DefaultThinkOptions())
	if thought.Trigger != TriggerAsked || thought.At != -1 || thought.Text != "" || thought.Stopped != StoppedNothing {
		t.Fatalf("thought %+v", thought)
	}
	if thought.Then != ThenEnd || thought.Taught != -1 || thought.HandedOver != -1 || thought.Questioned() != 0 {
		t.Fatalf("thought %+v", thought)
	}
	if len(m.G.Children(Think)) != 0 {
		t.Fatal("Think grew children")
	}
	if !strings.Contains(Summarize(thought), "nothing to think with") {
		t.Fatalf("summary %q", Summarize(thought))
	}
	// the JSON record is the Python one: every key, lists rather than nulls
	raw, err := json.Marshal(thought)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"trigger", "at", "about", "text", "depth", "stopped", "then", "taught", "handed_over",
		"cost", "probability", "expanded", "questioned", "questions", "labels", "node_ids", "step_costs"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("key %q missing from %s", key, raw)
		}
	}
	for _, key := range []string{"questions", "labels", "node_ids", "step_costs"} {
		if _, ok := doc[key].([]any); !ok {
			t.Errorf("%q is %v, want a list", key, doc[key])
		}
	}
	if !reflect.DeepEqual(doc, jsonRoundTrip(t, thought.ToDict())) {
		t.Fatalf("ToDict %v differs from the JSON record %v", thought.ToDict(), doc)
	}
}

func jsonRoundTrip(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

func TestAskingAboutATextTeachesTheModelToThinkThere(t *testing.T) {
	m := thinker(t)
	g := m.G
	o := DefaultThinkOptions()
	o.About = "the cat"
	thought := quietThought(t, m, o)
	if thought.At < First || thought.Taught != thought.At {
		t.Fatalf("at %d taught %d", thought.At, thought.Taught)
	}
	if _, ok := g.Edge(thought.At, Think); !ok {
		t.Fatal("no edge into Think")
	}
	if g.Count[Think] != 1 {
		t.Fatalf("Think count %d", g.Count[Think])
	}
	if thought.Then != ThenEnd || thought.HandedOver != -1 { // a thought asked for ends; nothing is backed out of
		t.Fatalf("then %q handed over %d", thought.Then, thought.HandedOver)
	}
	if len(g.Parents(Back)) != 0 {
		t.Fatal("Back was taught")
	}
	doc := thought.ToDict()
	if doc["trigger"] != TriggerAsked || doc["at"] != thought.At || doc["about"] != "the cat" {
		t.Fatalf("doc %v", doc)
	}
	if err := g.CheckInvariants(thinkTexts, false); err != nil {
		t.Fatal(err)
	}
}

func TestLearningOffThinksWithoutWritingAnything(t *testing.T) {
	m := thinker(t)
	untouched := thinker(t)
	o := DefaultThinkOptions()
	o.About, o.Learn = "the cat", false
	thought := quietThought(t, m, o)
	if thought.At < First || thought.Taught != -1 {
		t.Fatalf("at %d taught %d", thought.At, thought.Taught)
	}
	if len(m.G.Children(Think)) != 0 || len(m.G.Parents(Think)) != 0 {
		t.Fatal("Think grew edges")
	}
	// Place may split a node to say where the prefix ends, which is structure, not learning: compressed
	// again, the graph is the one it was
	m.G.Compress()
	untouched.G.Compress()
	if m.G.NumEdges() != untouched.G.NumEdges() || m.G.NumNodes() != untouched.G.NumNodes() {
		t.Fatalf("%d nodes / %d edges, want %d / %d", m.G.NumNodes(), m.G.NumEdges(), untouched.G.NumNodes(), untouched.G.NumEdges())
	}
}

func TestThinkOnTeachesThoughtsAndWhereTheyQuestionThemselves(t *testing.T) {
	m := thinker(t)
	g := m.G
	opts := DefaultTrainOptions()
	opts.Epochs = 3
	learned, err := m.ThinkOn(thinkThoughts, opts, true, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	// "is that right?" and "what did the cat do?"
	if learned.Thoughts != 3 || learned.Questions != 2 || learned.Taught != 2 || len(learned.Epochs) == 0 {
		t.Fatalf("learned %+v", learned)
	}
	if len(g.Children(Think)) == 0 || g.Count[Think] < 2 {
		t.Fatalf("Think has %d children and count %d", len(g.Children(Think)), g.Count[Think])
	}
	// the thoughts begin at Think: Start saw none of them (a thought opening like a text - "the sky is blue"
	// after "the cat sat" - shares the text's nodes, one graph with two origins)
	startCount := g.Count[Start]
	for _, thought := range thinkThoughts {
		grams := g.Encode(thought)
		if _, ok := g.NodePathFrom(Think, grams); !ok {
			t.Fatalf("%q is not walkable from Think", thought)
		}
		if _, ok := g.NodePath(grams); ok && !strings.HasPrefix(thought, "the ") {
			t.Fatalf("%q is walkable from Start", thought)
		}
	}
	if g.Count[Start] != startCount {
		t.Fatal("Start was counted for a thought")
	}
	if err := g.CheckInvariants(thinkTexts, false); err != nil {
		t.Fatal(err)
	}
	// now it has thoughts to think with
	o := DefaultThinkOptions()
	o.Learn = false
	thought := quietThought(t, m, o)
	if thought.Text == "" || len(thought.NodeIDs) == 0 || thought.NodeIDs[0] != Think {
		t.Fatalf("thought %+v", thought)
	}
	for _, n := range thought.NodeIDs {
		if n == Start {
			t.Fatal("a thought passed through Start")
		}
	}
	if thought.Stopped != StoppedEnd && thought.Stopped != StoppedLength {
		t.Fatalf("stopped %q", thought.Stopped)
	}
	if !(thought.Probability > 0) {
		t.Fatalf("probability %v", thought.Probability)
	}
	s := DefaultThinkOptions()
	seed := int64(1)
	s.Mode, s.Seed, s.Learn = "sample", &seed, false
	if sampled := quietThought(t, m, s); sampled.Text == "" {
		t.Fatal("a sampled thought said nothing")
	}
}

func TestAThoughtQuestionsItselfWhereTheModelLearnedToThink(t *testing.T) {
	m := thinker(t)
	g := m.G
	opts := DefaultTrainOptions()
	opts.Epochs = 3
	if _, err := m.ThinkOn([]string{"the sky is blue because of light", "why is that so?"}, opts, false, 1.0); err != nil {
		t.Fatal(err)
	}
	// teach it, by experience, to stop and think on the path of whatever it thinks first: a node that thinks a
	// lot makes the thought through it dearer, so the best thought may change - then teach the new one's path
	// too; within a few rounds the thought it thinks passes a node it has learned to think at
	quiet := DefaultThinkOptions()
	quiet.Learn = false
	thought := quietThought(t, m, quiet)
	node := -1
	for round := 0; round < 4 && node < 0; round++ {
		first := -1
		for _, n := range thought.NodeIDs {
			if n >= First {
				first = n
				break
			}
		}
		if first < First {
			t.Fatalf("thought %+v has no real node", thought)
		}
		for tries := 0; !g.ThinksAt(first); tries++ {
			if tries > 500 {
				t.Fatalf("node %d never learned to think", first)
			}
			if _, err := g.ObserveThink(first, 1.0); err != nil {
				t.Fatal(err)
			}
		}
		thought = quietThought(t, m, quiet)
		for _, n := range thought.NodeIDs {
			if n >= First && g.ThinksAt(n) {
				node = n
				break
			}
		}
	}
	if node < First {
		t.Fatal("no thought passed a node it learned to think at")
	}
	if thought.Questioned() != 1 {
		t.Fatalf("questioned %d times: %+v", thought.Questioned(), thought)
	}
	q := thought.Questions[0]
	if q.Trigger != TriggerQuestioned || q.At != node || q.Depth != 1 || q.Then != ThenThink {
		t.Fatalf("question %+v", q)
	}
	if q.Taught != -1 { // it already knew to think here: that is why it asked
		t.Fatalf("the question taught %d", q.Taught)
	}
	// a question must say something the thought above it did not
	if normalizeThought(q.Text) == normalizeThought(thought.Text) {
		t.Fatalf("the question %q only repeats the thought", q.Text)
	}
	if !strings.Contains(Summarize(thought), "questioned itself once") {
		t.Fatalf("summary %q", Summarize(thought))
	}
	doc := thought.ToDict()
	if doc["questioned"] != 1 || doc["questions"].([]map[string]any)[0]["trigger"] != TriggerQuestioned {
		t.Fatalf("doc %v", doc)
	}
	// the depth is bounded, and so is the number of questions
	shallow := quiet
	shallow.MaxDepth = 0
	if quietThought(t, m, shallow).Questioned() != 0 {
		t.Fatal("a thought of depth 0 questioned itself")
	}
	none := quiet
	none.MaxQuestions = 0
	if quietThought(t, m, none).Questioned() != 0 {
		t.Fatal("a thought with no questions questioned itself")
	}
}

func repeatNode(t *testing.T, g *Graph) int {
	t.Helper()
	if node, _, ok := g.Lookup("ha "); ok {
		return node
	}
	for _, n := range g.AliveNodes() {
		if n >= First {
			return n
		}
	}
	t.Fatal("no real node")
	return -1
}

func TestAThoughtOutOfARepeatHandsOverToBackWhenItStops(t *testing.T) {
	m := thinker(t)
	g := m.G
	node := repeatNode(t, g)
	o := DefaultThinkOptions()
	o.At, o.Trigger, o.Went = node, TriggerStutter, g.Children(node)[0].P
	thought := quietThought(t, m, o)
	if thought.Then != ThenBack || thought.Taught != node || thought.HandedOver != node {
		t.Fatalf("thought %+v", thought)
	}
	if _, ok := g.Edge(node, Think); !ok {
		t.Fatal("no edge into Think")
	}
	if _, ok := g.Edge(node, Back); !ok {
		t.Fatal("no edge into Back")
	}
	if _, ok := g.BackCost(node); !ok {
		t.Fatal("no back cost")
	}
	if !strings.Contains(Summarize(thought), "then backed up") {
		t.Fatalf("summary %q", Summarize(thought))
	}
	// read-only: the same thought, nothing written
	fresh := thinker(t)
	q := DefaultThinkOptions()
	q.At, q.Trigger, q.Learn = repeatNode(t, fresh.G), TriggerStutter, false
	quiet := quietThought(t, fresh, q)
	if quiet.Then != ThenBack || quiet.Taught != -1 || quiet.HandedOver != -1 {
		t.Fatalf("thought %+v", quiet)
	}
	if len(fresh.G.Parents(Back)) != 0 || len(fresh.G.Parents(Think)) != 0 {
		t.Fatal("a read-only thought wrote into the graph")
	}
}

func TestThinkValidation(t *testing.T) {
	m := thinker(t)
	bad := []func(*ThinkOptions){
		func(o *ThinkOptions) { o.Mode = "walk" },
		func(o *ThinkOptions) { o.At = Start },
		func(o *ThinkOptions) { o.K = 0 },
		func(o *ThinkOptions) { o.MaxDepth = -1 },
		func(o *ThinkOptions) { o.MaxLength = -1 },
		func(o *ThinkOptions) { o.Temperature = -1 },
	}
	for i, tweak := range bad {
		o := DefaultThinkOptions()
		tweak(&o)
		if _, err := m.Think(o); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	negative, err := NewNegativeModel(1, DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := negative.ThinkOn(thinkThoughts, DefaultTrainOptions(), true, 1.0); err == nil {
		t.Fatal("the negative network was taught thoughts")
	}
	opts := DefaultTrainOptions()
	opts.Origin = Think
	if _, err := negative.Train(thinkThoughts, opts); err == nil {
		t.Fatal("the negative network was trained from Think")
	}
	if _, err := m.G.ObserveFrom(Back, m.G.Encode("the cat"), true); err == nil {
		t.Fatal("a sequence was observed from Back")
	}
}

func TestTheModelFileKeepsItsThoughts(t *testing.T) {
	m := thinker(t)
	opts := DefaultTrainOptions()
	opts.Epochs = 3
	if _, err := m.ThinkOn(thinkThoughts, opts, true, 1.0); err != nil {
		t.Fatal(err)
	}
	quiet := DefaultThinkOptions()
	quiet.Learn = false
	before := quietThought(t, m, quiet)
	path := filepath.Join(t.TempDir(), "model.count.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.G.CheckInvariants(thinkTexts, false); err != nil {
		t.Fatal(err)
	}
	doc := loaded.G.ToDoc()
	if doc.FormatVersion != 4 || doc.Nodes.Labels[Think] != ThinkLabel {
		t.Fatalf("format %d, labels %v", doc.FormatVersion, doc.Nodes.Labels[:First])
	}
	after := quietThought(t, loaded, quiet)
	if after.Text != before.Text || !reflect.DeepEqual(after.Labels, before.Labels) {
		t.Fatalf("thought %q / %v after loading, %q / %v before", after.Text, after.Labels, before.Text, before.Labels)
	}
	if d := after.Cost - before.Cost; d > 1e-9 || d < -1e-9 {
		t.Fatalf("cost %v after loading, %v before", after.Cost, before.Cost)
	}
}

func TestOlderFilesGainTheThinkSentinelOnLoad(t *testing.T) {
	m := thinker(t)
	doc := m.G.ToDoc()
	// a format 3 file: no Think node, every id from Think up one lower
	shift := func(i int) int {
		if i > Think {
			return i - 1
		}
		return i
	}
	old := *doc
	old.FormatVersion = 3
	old.Nodes.Labels = append(append([]string{}, doc.Nodes.Labels[:Think]...), doc.Nodes.Labels[First:]...)
	old.Nodes.Z = append(append([]float64{}, doc.Nodes.Z[:Think]...), doc.Nodes.Z[First:]...)
	old.Nodes.A = append(append([]float64{}, doc.Nodes.A[:Think]...), doc.Nodes.A[First:]...)
	old.Nodes.B = append(append([]float64{}, doc.Nodes.B[:Think]...), doc.Nodes.B[First:]...)
	old.Nodes.H = append(append([]float64{}, doc.Nodes.H[:Think]...), doc.Nodes.H[First:]...)
	old.Nodes.K = append(append([]float64{}, doc.Nodes.K[:Think]...), doc.Nodes.K[First:]...)
	old.Nodes.Count = append(append([]int64{}, doc.Nodes.Count[:Think]...), doc.Nodes.Count[First:]...)
	old.Edges.Src = append([]int{}, doc.Edges.Src...)
	old.Edges.Dst = append([]int{}, doc.Edges.Dst...)
	for i := range old.Edges.Src {
		old.Edges.Src[i], old.Edges.Dst[i] = shift(old.Edges.Src[i]), shift(old.Edges.Dst[i])
	}
	g, err := GraphFromDoc(&old)
	if err != nil {
		t.Fatal(err)
	}
	if g.Labels[Think] != ThinkLabel || g.Count[Think] != 0 || len(g.Children(Think)) != 0 || len(g.Parents(Think)) != 0 {
		t.Fatalf("Think after the upgrade: %q count %d", g.Labels[Think], g.Count[Think])
	}
	if !reflect.DeepEqual(g.Labels, m.G.ToDoc().Nodes.Labels) {
		t.Fatal("the upgraded labels differ")
	}
	if err := g.CheckInvariants(thinkTexts, false); err != nil {
		t.Fatal(err)
	}
	if g.ToDoc().FormatVersion != 4 {
		t.Fatal("the upgraded graph does not write format 4")
	}
}

func TestOnwardDropsThinkButOnlyBackHandsOver(t *testing.T) {
	costs := []ChildCost{{Child: 10, Cost: 1}, {Child: Think, Cost: 0.5}, {Child: 11, Cost: 2}}
	if got := Onward(costs); len(got) != 2 || got[0].Child != 10 || got[1].Child != 11 {
		t.Fatalf("onward %v", got)
	}
	// Think cheaper than everything still hands nothing over; Back cheaper than everything does
	withBack := append(costs, ChildCost{Child: Back, Cost: 0.25})
	if got := Onward(withBack); got != nil {
		t.Fatalf("onward %v, want a hand-over", got)
	}
	dearBack := append(costs, ChildCost{Child: Back, Cost: 1.5})
	if got := Onward(dearBack); len(got) != 2 {
		t.Fatalf("onward %v", got)
	}
	if got := Onward([]ChildCost{{Child: 10, Cost: 1}}); len(got) != 1 {
		t.Fatalf("onward %v", got)
	}
}

func TestAConversationThinksBeforeBackingUp(t *testing.T) {
	m := thinker(t)
	opts := DefaultConverseOptions()
	opts.Turns = 6
	turns, err := m.Converse("", opts)
	if err != nil {
		t.Fatal(err)
	}
	thoughts := ThoughtsOf(turns)
	if len(thoughts) == 0 {
		t.Fatalf("no turn thought: %v", Transcript(turns))
	}
	for _, thought := range thoughts {
		if thought.Then != ThenBack || !BacksUp(thought.Trigger) || thought.Taught < First || thought.HandedOver != thought.Taught {
			t.Fatalf("thought %+v", thought)
		}
	}
	raw, _ := json.Marshal(turns)
	if !strings.Contains(string(raw), `"thought":{`) {
		t.Fatal("the turn JSON carries no thought")
	}
	quiet := thinker(t)
	opts.Think = false
	turns, err = quiet.Converse("", opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(ThoughtsOf(turns)) != 0 {
		t.Fatal("a conversation with thinking off thought")
	}
	if len(quiet.G.Parents(Think)) != 0 {
		t.Fatal("Think was taught with thinking off")
	}
}
