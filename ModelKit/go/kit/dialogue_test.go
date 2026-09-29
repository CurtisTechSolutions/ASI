package kit

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// corpus and trained are the model's own test helpers (radixnet_test.go): the
// sample corpus of RadixCyclicNN, and a model trained on it.

func corpus(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "RadixCyclicNN", "data", "sample_corpus.txt"))
	if err != nil {
		t.Skipf("sample corpus not found: %v", err)
	}
	return radixnet.SplitTexts(string(raw), "lines", 0)
}

// trained builds a deterministic model: Exact counting (atomics) so the tests can compare numbers;
// workers 0 means one goroutine per text.
func trained(t *testing.T, epochs int, workers int) *radixnet.Model {
	t.Helper()
	m, err := radixnet.NewModel(1, radixnet.DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Workers = workers
	m.G.Workers = workers
	m.Exact = true
	opts := radixnet.DefaultTrainOptions()
	opts.Epochs = epochs
	if _, err := m.Train(corpus(t), opts); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestConverse(t *testing.T) {
	m := trained(t, 2, 4)
	opts := DefaultConverseOptions()
	opts.Turns, opts.Learn = 6, false
	turns, err := Converse(m, "the cat sat on the mat", opts)
	if err != nil || len(turns) != 7 {
		t.Fatalf("converse: %v %d", err, len(turns))
	}
	if !turns[0].Given || turns[0].Speaker != "A" || turns[0].Text != "the cat sat on the mat" {
		t.Fatalf("opening: %+v", turns[0])
	}
	seen := map[string]bool{}
	for i, tr := range turns {
		if tr.Index != i || tr.Speaker != DefaultSpeakers[i%2] {
			t.Fatalf("turn %d: %+v", i, tr)
		}
		key := Normalize(tr.Text)
		if seen[key] {
			t.Fatalf("repeated utterance %q", tr.Text)
		}
		seen[key] = true
		if i > 0 && !tr.Fresh {
			words := strings.Fields(turns[i-1].Text)
			wanted := strings.Fields(tr.Context)
			found := false
			for j := 0; j+len(wanted) <= len(words); j++ {
				if reflect.DeepEqual(words[j:j+len(wanted)], wanted) {
					found = true
				}
			}
			if !found || tr.Text != tr.Context+tr.Reply {
				t.Fatalf("turn %d did not pick up the previous line: %+v", i, tr)
			}
		}
	}
	// deterministic on a model in the same state - with Learn on, the first conversation taught it something
	plain := opts
	plain.Learn = false
	first, _ := Converse(m, "the cat sat on the mat", plain)
	again, _ := Converse(m, "the cat sat on the mat", plain)
	if Transcript(again) != Transcript(first) {
		t.Fatal("beam conversations must be deterministic")
	}
	if got := TailContext("the cat sat on the mat", 12); got != "on the mat" {
		t.Fatalf("TailContext = %q", got)
	}
	empty, _ := radixnet.NewModel(0, radixnet.DefaultGraphOptions())
	if turns, _ := Converse(empty, "", opts); len(turns) != 0 {
		t.Fatal("an untrained model has nothing to say")
	}
}

// A run of words repeated immediately after itself - and only that.
func TestStutter(t *testing.T) {
	for line, want := range map[string]string{
		"the the west":                                "the",
		"say morning morning":                         "morning",
		"the cat the cat sat":                         "the cat",
		"The  The":                                    "the", // whitespace and case aside
		"the cat sat on the mat":                      "",
		"where there is a will there is a way":        "",
		"a bird in the hand is worth two in the bush": "",
		"blowers blower":                              "",
		"park":                                        "",
		"":                                            "",
	} {
		if got := Stutter(line, LongestStutter); got != want {
			t.Fatalf("Stutter(%q) = %q; want %q", line, got, want)
		}
	}
	long := "the cat sat on the mat the cat sat on the mat"
	if got := Stutter(long, LongestStutter); got != "" { // six words twice over: past the default
		t.Fatalf("Stutter(%q) = %q", long, got)
	}
	if got := Stutter(long, 6); got != "the cat sat on the mat" {
		t.Fatalf("Stutter(%q, 6) = %q", long, got)
	}
	if got := Stutter("the the west", 0); got != "" {
		t.Fatalf("Stutter with no run allowed = %q", got)
	}
}

// A voice that can only repeat its own words: punished, or allowed to say them on request.
func TestConverseWordRepeats(t *testing.T) {
	m, err := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	m.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if _, err := m.Train([]string{"ha ha ha ha ha"}, radixnet.TrainOptions{Epochs: 3}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	opts := DefaultConverseOptions()
	opts.Turns = 4
	turns, err := Converse(m, "", opts)
	if err != nil || len(turns) == 0 {
		t.Fatalf("converse: %v %d", err, len(turns))
	}
	stutters := 0
	for _, tr := range turns {
		if !tr.Repeat { // nothing it could say repeated nothing
			t.Fatalf("turn %q was not flagged a repeat", tr.Text)
		}
		if tr.Stutter != (Stutter(tr.Text, LongestStutter) != "") {
			t.Fatalf("turn %q: stutter = %v", tr.Text, tr.Stutter)
		}
		if tr.Stutter {
			stutters++
		}
	}
	if stutters == 0 || len(Repeats(turns)) != len(turns) {
		t.Fatalf("%d stutters, %d punished of %d turns", stutters, len(Repeats(turns)), len(turns))
	}
	// allowed instead: spoken freely, flagged for what they are, and punished for nothing
	opts.AvoidWordRepeats = false
	loose, err := Converse(m, "", opts)
	if err != nil || len(loose) != opts.Turns {
		t.Fatalf("converse: %v %d", err, len(loose))
	}
	for _, tr := range loose {
		if !tr.Stutter || tr.Repeat {
			t.Fatalf("turn %q: stutter = %v, repeat = %v", tr.Text, tr.Stutter, tr.Repeat)
		}
	}
	if got := Repeats(loose); len(got) != 0 {
		t.Fatalf("nothing should be punished: %q", got)
	}
}

// A conversation streamed as it happens: the turns are the answer, the rest is the window a backtrack may still
// rewrite - and streaming changes nothing about what is said.
func TestConverseStream(t *testing.T) {
	ways, _ := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
	ways.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if _, err := ways.Train([]string{"ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat"},
		radixnet.TrainOptions{Epochs: 3}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	stuck, _ := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
	stuck.Exact = true
	if _, err := stuck.Train([]string{"ha ha ha ha ha"}, radixnet.TrainOptions{Epochs: 3}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	streamed := func(m *radixnet.Model, opening string, opts ConverseOptions) ([]*Turn, []map[string]any) {
		events := []map[string]any{}
		opts.Stream = func(event map[string]any) { events = append(events, event) }
		turns, err := Converse(m, opening, opts)
		if err != nil {
			t.Fatalf("converse: %v", err)
		}
		return turns, events
	}
	kinds := map[string]bool{}
	for _, kind := range StreamEvents {
		kinds[kind] = true
	}

	// the turns streamed are the turns returned, and the window between two turns belongs to the one that follows
	m := trained(t, 2, 4)
	opts := DefaultConverseOptions()
	opts.Turns, opts.Learn = 6, false
	turns, events := streamed(m, "the cat sat on the mat", opts)
	if len(turns) != 7 {
		t.Fatalf("%d turns", len(turns))
	}
	spoken := []*Turn{}
	looks := map[int]string{}
	owner := -1
	for i := len(events) - 1; i >= 0; i-- {
		event := events[i]
		if !kinds[event["event"].(string)] {
			t.Fatalf("unknown event %v", event)
		}
		if event["event"] == "turn" {
			owner = event["index"].(int)
		} else if event["index"].(int) != owner {
			t.Fatalf("event %v belongs to turn %d", event, owner)
		}
	}
	for _, event := range events {
		switch event["event"] {
		case "turn":
			spoken = append(spoken, event["turn"].(*Turn))
		case "look":
			looks[event["index"].(int)] = event["from"].(string)
		}
	}
	if !reflect.DeepEqual(spoken, turns) {
		t.Fatalf("the turns streamed are not the turns returned: %v vs %v", spoken, turns)
	}
	for _, tr := range turns[1:] {
		// a turn's context is where its last look started from ("" when it changed the subject)
		if from, ok := looks[tr.Index]; !ok || from != tr.Context {
			t.Fatalf("turn %d picked up %q but last looked from %q", tr.Index, tr.Context, from)
		}
	}

	// the window shows the backing up: what it was about to say, what it caught, each step back, the way on
	opts = DefaultConverseOptions()
	opts.Turns, opts.Learn = 6, false
	turns, events = streamed(ways, "", opts)
	rethought := 0
	found := 0
	for _, tr := range turns {
		if tr.Rethink == nil {
			continue
		}
		rethought++
		window := []map[string]any{}
		for _, event := range events {
			if event["index"].(int) == tr.Index && event["event"] != "turn" {
				window = append(window, event)
			}
		}
		var draft, caught map[string]any
		steps := []map[string]any{}
		for i, event := range window {
			switch event["event"] {
			case "draft":
				draft = event
				if i+1 >= len(window) || window[i+1]["event"] != "caught" {
					t.Fatalf("a draft is followed by what it caught: %v", window)
				}
			case "caught":
				caught = event
			case "backtrack":
				steps = append(steps, event)
			case "found":
				found++
				if event["text"] != tr.Text || event["explored"] != tr.Rethink.Explored {
					t.Fatalf("found %v for turn %+v", event, tr)
				}
			case "stuck":
				if event["explored"] != tr.Rethink.Explored || tr.Rethink.Found {
					t.Fatalf("stuck %v for turn %+v", event, tr)
				}
			}
		}
		if draft == nil || caught == nil {
			t.Fatalf("no draft / caught in the window of turn %d: %v", tr.Index, window)
		}
		if caught["kind"] != tr.Rethink.Kind || caught["noticed"] != tr.Rethink.Noticed {
			t.Fatalf("caught %v vs %+v", caught, tr.Rethink)
		}
		if len(steps) != tr.Rethink.Steps {
			t.Fatalf("%d backtrack events for %d steps", len(steps), tr.Rethink.Steps)
		}
		if len(steps) > 0 {
			if caught["cut"] != steps[0]["cut"] || !strings.HasPrefix(draft["text"].(string), steps[0]["cut"].(string)) {
				t.Fatalf("the cut is where the draft is kept to: %v %v %v", caught, steps[0], draft)
			}
			if steps[len(steps)-1]["cut"] != tr.Rethink.Cut {
				t.Fatalf("the last cut is the record's: %v vs %q", steps[len(steps)-1], tr.Rethink.Cut)
			}
			for i, step := range steps {
				if step["step"] != i+1 || step["wider"].(int) < 5 {
					t.Fatalf("step %d: %v", i+1, step)
				}
			}
		} else if caught["cut"] != "" {
			t.Fatalf("no step back, but a cut: %v", caught)
		}
	}
	if rethought == 0 || found == 0 {
		t.Fatalf("nothing thought twice about (%d rethinks, %d found): %v", rethought, found, Transcript(turns))
	}
	_, events = streamed(stuck, "", opts)
	seen := map[string]int{}
	for _, event := range events {
		seen[event["event"].(string)]++
	}
	if seen["stuck"] == 0 || seen["found"] != 0 {
		t.Fatalf("a voice with nowhere to go gets stuck: %v", seen)
	}
	opts.Explore = 0
	_, events = streamed(ways, "", opts)
	for _, event := range events {
		if event["event"] != "look" && event["event"] != "turn" {
			t.Fatalf("nothing to catch, but %v", event)
		}
	}

	// streaming changes nothing: the same turns, and the same graph afterwards
	silent, watched := trained(t, 2, 4), trained(t, 2, 4)
	opts = DefaultConverseOptions()
	opts.Turns = 10
	plain, err := Converse(silent, "the cat sat on the mat", opts)
	if err != nil {
		t.Fatalf("converse: %v", err)
	}
	turns, events = streamed(watched, "the cat sat on the mat", opts)
	if !reflect.DeepEqual(turns, plain) {
		t.Fatalf("streaming changed the conversation:\n%s\nvs\n%s", Transcript(turns), Transcript(plain))
	}
	if silent.G.NumEdges() != watched.G.NumEdges() {
		t.Fatalf("streaming changed what was learned: %d vs %d edges", silent.G.NumEdges(), watched.G.NumEdges())
	}
	count := 0
	for _, event := range events {
		if event["event"] == "turn" {
			count++
		}
	}
	if count != len(plain) {
		t.Fatalf("%d turn events for %d turns", count, len(plain))
	}

	// and Backtrack streams on its own, without a turn to belong to
	events = nil
	bo := BacktrackOptions{Explore: Explore, Mode: "beam", K: 3, MaxLength: 60, AvoidRepeats: true, AvoidWordRepeats: true,
		Stream: func(event map[string]any) { events = append(events, event) }}
	way, record, err := Backtrack(ways, "ha ha ha", bo)
	if err != nil || way == nil {
		t.Fatalf("backtrack: %v %+v", err, record)
	}
	if len(events) != 3 || events[0]["event"] != "caught" || events[1]["event"] != "backtrack" || events[2]["event"] != "found" {
		t.Fatalf("events: %v", events)
	}
	if events[0]["cut"] != "ha " || events[0]["noticed"] != "ha" || events[1]["wider"] != 6 || events[2]["text"] != way.FullText {
		t.Fatalf("events: %v", events)
	}
	events = nil
	bo.Keep = "ha ha "
	if _, _, err := Backtrack(ways, "ha ha ha", bo); err != nil || len(events) != 1 || events[0]["cut"] != "" {
		t.Fatalf("the words it picked up: %v %v", err, events)
	}
	events = nil
	bo.Keep = ""
	if _, _, err := Backtrack(ways, "the cat sat on the mat", bo); err != nil || len(events) != 0 {
		t.Fatalf("nothing caught: %v %v", err, events)
	}
}

// Catching itself repeating, backing up to where the walk went round, and looking for another way on.
func TestBacktrack(t *testing.T) {
	// "ha ha ..." loops; the other two lines leave the loop after "ha "
	ways, err := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	ways.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if _, err := ways.Train([]string{"ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat"},
		radixnet.TrainOptions{Epochs: 3}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	stuck, _ := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
	stuck.Exact = true
	if _, err := stuck.Train([]string{"ha ha ha ha ha"}, radixnet.TrainOptions{Epochs: 3}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	opts := BacktrackOptions{Explore: Explore, Mode: "beam", K: 3, MaxLength: 60,
		AvoidRepeats: true, AvoidWordRepeats: true}

	found, record, err := Backtrack(ways, "ha ha ha", opts)
	if err != nil || found == nil || !record.Found {
		t.Fatalf("backtrack: %v %+v", err, record)
	}
	if record.Kind != "stutter" || record.Noticed != "ha" || record.Cut != "ha " || record.Steps != 1 ||
		record.Explored == 0 {
		t.Fatalf("record = %+v", record)
	}
	if !strings.HasPrefix(found.FullText, "ha ") || Stutter(found.FullText, LongestStutter) != "" {
		t.Fatalf("the way on keeps what was said once and does not go round again: %q", found.FullText)
	}

	// a voice with nowhere else to go says so - and it did look
	found, record, _ = Backtrack(stuck, "ha ha ha", opts)
	if found != nil || record.Found || record.Noticed != "ha" || record.Explored == 0 {
		t.Fatalf("stuck: %v %+v", found, record)
	}

	// the words it picked up are not its own to rethink
	keep := opts
	keep.Keep = "ha ha "
	found, record, _ = Backtrack(ways, "ha ha ha", keep)
	if found != nil || record.Steps != 0 || record.Explored != 0 {
		t.Fatalf("keep: %v %+v", found, record)
	}

	// a whole utterance it has said before is a retread end to end: it keeps all it can and differs at the end
	said := opts
	said.Heard = NewHeard([]string{"ha and then the cat sat"})
	found, record, _ = Backtrack(ways, "ha and then the cat sat", said)
	if record.Kind != "repeat" || record.Noticed != "ha and then the cat sat" || record.Steps == 0 {
		t.Fatalf("a heard line: %+v", record)
	}
	if !strings.HasPrefix("ha and then the cat ", record.Cut) {
		t.Fatalf("cut %q is not that last word, or further back", record.Cut)
	}
	if found != nil && !strings.HasPrefix(found.FullText, record.Cut) {
		t.Fatalf("a way on keeps what it cut: %q from %q", found.FullText, record.Cut)
	}

	// one word is nothing to back up from, and nothing is caught with the settings off
	one := opts
	one.Heard = NewHeard([]string{"ha"})
	if found, record, _ = Backtrack(ways, "ha", one); found != nil || record.Kind != "repeat" || record.Steps != 0 {
		t.Fatalf("one word: %v %+v", found, record)
	}
	off := said
	off.AvoidRepeats = false
	if found, record, _ = Backtrack(ways, "ha and then the cat sat", off); found != nil || record.Kind != "" {
		t.Fatalf("repeats allowed: %v %+v", found, record)
	}
	noWords := opts
	noWords.AvoidWordRepeats = false
	if found, record, _ = Backtrack(ways, "ha ha ha", noWords); found != nil || record.Kind != "" {
		t.Fatalf("word repeats allowed: %v %+v", found, record)
	}

	// nothing to rethink, and nothing to explore
	if found, record, _ = Backtrack(ways, "the cat sat on the mat", opts); found != nil || record.Noticed != "" {
		t.Fatalf("clean: %v %+v", found, record)
	}
	noExplore := opts
	noExplore.Explore = 0
	if found, record, _ = Backtrack(ways, "ha ha ha", noExplore); found != nil || record.Steps != 0 {
		t.Fatalf("off: %v %+v", found, record)
	}

	// and a conversation backs out of the loop it walks into
	cfg := DefaultConverseOptions()
	cfg.Turns = 4
	turns, err := Converse(ways, "", cfg)
	if err != nil {
		t.Fatalf("converse: %v", err)
	}
	thought := 0
	for _, turn := range turns {
		if turn.Rethink == nil {
			continue
		}
		thought++
		if turn.Rethink.Noticed == "" || turn.Text != turn.Context+turn.Reply {
			t.Fatalf("turn %+v", turn)
		}
		if turn.Rethink.Found && (!strings.HasPrefix(turn.Text, turn.Rethink.Cut) || turn.Repeat || turn.Stutter) {
			t.Fatalf("a way out keeps what was said once: %+v", turn)
		}
	}
	if thought == 0 {
		t.Fatal("nothing was ever noticed")
	}
	cfg.Explore = 0
	plain, _ := Converse(ways, "", cfg)
	for _, turn := range plain {
		if turn.Rethink != nil {
			t.Fatalf("nothing is noticed with the exploring off: %+v", turn)
		}
	}
}

// The rethink is not only a way out of this turn: what it finds out is taught to the graph.
func TestLearnsWhereItGoesRound(t *testing.T) {
	build := func() *radixnet.Model {
		m, _ := radixnet.NewModel(3, radixnet.DefaultGraphOptions())
		m.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
		if _, err := m.Train([]string{"ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat"},
			radixnet.TrainOptions{Epochs: 3}); err != nil {
			t.Fatalf("Train: %v", err)
		}
		return m
	}
	opts := BacktrackOptions{Explore: Explore, Mode: "beam", K: 3, MaxLength: 60,
		AvoidRepeats: true, AvoidWordRepeats: true, Learn: true}

	m := build()
	if len(m.G.Parents(radixnet.Back)) != 0 {
		t.Fatal("a fresh model has no idea where it goes round")
	}
	_, record, err := Backtrack(m, "ha ha ha", opts)
	if err != nil || record.Taught < radixnet.First {
		t.Fatalf("backing out teaches the node: %v %+v", err, record)
	}
	if _, ok := m.G.Edge(record.Taught, radixnet.Back); !ok {
		t.Fatalf("node %d was not taught to hand over", record.Taught)
	}
	if _, ok := m.G.BackCost(record.Taught); !ok {
		t.Fatalf("node %d has no back cost", record.Taught)
	}

	// enough hand-overs and the search refuses by itself
	node := record.Taught
	for i := 0; i < 6; i++ {
		if _, err := m.G.ObserveBack(node, -1, -1, 1); err != nil {
			t.Fatalf("ObserveBack: %v", err)
		}
	}
	if len(radixnet.Onward(m.G.ChildCosts(node))) != 0 {
		t.Fatalf("the model's most likely next step at %d should be to stop", node)
	}

	// a conversation leaves the model knowing more than it did, and with the learning off it does not
	m = build()
	cfg := DefaultConverseOptions()
	cfg.Turns = 6
	if _, err := Converse(m, "", cfg); err != nil {
		t.Fatalf("converse: %v", err)
	}
	if len(m.G.Parents(radixnet.Back)) == 0 {
		t.Fatal("a conversation taught it nothing")
	}
	quiet := build()
	cfg.Learn = false
	turns, _ := Converse(quiet, "", cfg)
	if len(quiet.G.Parents(radixnet.Back)) != 0 {
		t.Fatal("nothing is taught with the learning off")
	}
	for _, turn := range turns {
		if turn.Rethink != nil && turn.Rethink.Taught != -1 {
			t.Fatalf("turn taught %d with the learning off", turn.Rethink.Taught)
		}
	}

	// a repeat the graph cannot place teaches nothing
	blank := build()
	if got := TeachBack(blank, "zzz zzz", 4, nil, 1); got != -1 {
		t.Fatalf("TeachBack on an unknown prefix = %d", got)
	}
}

// What counts as a duplicate, and what a conversation does with the ones it cannot avoid.
func TestHeardAndRepeats(t *testing.T) {
	heard := NewHeard([]string{"the cat sat on the mat"})
	if !heard.Duplicate("The   Cat Sat On The Mat", "") { // whitespace and case aside
		t.Fatal("an utterance said before is a duplicate")
	}
	if heard.Duplicate("the dog barked", "") {
		t.Fatal("an unheard utterance is not a duplicate")
	}
	if !heard.Duplicate("on the mat", "") || heard.Duplicate("on the mat outside", "") {
		t.Fatal("an echo adds nothing; a longer utterance says more than was heard")
	}
	heard.Remember("the cat sat", " sat")
	if !heard.Duplicate("the dog sat", " sat") || heard.Duplicate("the dog sat", " sat down") {
		t.Fatal("the same words added after another context are a duplicate")
	}
	if heard.Duplicate("", "") || NewHeard(nil).Duplicate("anything", "") {
		t.Fatal("nothing is a duplicate of an empty conversation")
	}

	turns := []*Turn{{Text: "the cat sat"}, {Text: " west", Repeat: true}, {Text: "West", Repeat: true}, {Text: "  ", Repeat: true}}
	if got := Repeats(turns); !reflect.DeepEqual(got, []string{" west"}) {
		t.Fatalf("Repeats = %q; the flagged utterances, each once", got)
	}
	if got := Repeats(nil); len(got) != 0 {
		t.Fatalf("Repeats(nil) = %q", got)
	}

	// a long conversation on a small corpus runs out of new things to say: it speaks a duplicate once,
	// flags it, and stops rather than saying it again (with the exploring off - a voice that backs out of
	// its repeats keeps finding new things to say, which is TestBacktrack's business)
	m := trained(t, 2, 4)
	opts := DefaultConverseOptions()
	opts.Turns, opts.Explore = 40, 0
	long, err := Converse(m, "", opts)
	if err != nil {
		t.Fatalf("converse: %v", err)
	}
	said := map[string]bool{}
	for _, tr := range long {
		key := Normalize(tr.Text)
		if said[key] && !tr.Repeat { // only a flagged duplicate may say something twice
			t.Fatalf("utterance %q spoken twice", tr.Text)
		}
		said[key] = true
	}
	punished := Repeats(long)
	if len(punished) == 0 || len(long) >= opts.Turns {
		t.Fatalf("%d turns, %d to punish: the conversation should run out of new things to say", len(long), len(punished))
	}
	// and no duplicate is ever spoken twice: the conversation ends instead of going round in circles
	for i, tr := range long {
		if !tr.Repeat {
			continue
		}
		for _, later := range long[i+1:] {
			if Normalize(later.Text) == Normalize(tr.Text) {
				t.Fatalf("duplicate %q spoken again at turn %d", tr.Text, later.Index)
			}
		}
	}
}
