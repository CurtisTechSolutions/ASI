package radixnet

import (
	"encoding/json"
	"reflect"
	"sort"
	"testing"
)

// The keys are pinned: the Python and Rust ports assert the same numbers.
func TestTrainingKeysArePinned(t *testing.T) {
	cases := []struct {
		got, want uint64
	}{
		{SplitMix64(0), 0xE220A8397B1DCDAF},
		{ShuffleKey(7, 1, 0), 2915934175157713820},
		{ShuffleKey(-3, 2, 5), 14722852399570101988},
		{ReplayKey(7, 0), 5594249859331236493},
		{ReplayKey(-1, 9), 6800963518696358292},
	}
	for i, c := range cases {
		if c.got != c.want {
			t.Errorf("key %d: got %d, want %d", i, c.got, c.want)
		}
	}
}

func TestPacedAndReplayCount(t *testing.T) {
	var got []int
	for j := 0; j < 4; j++ {
		got = append(got, Paced(10, 0.3, j, 4))
	}
	if !reflect.DeepEqual(got, []int{3, 6, 8, 10}) {
		t.Fatalf("paced: %v", got)
	}
	if Paced(10, 0.01, 0, 4) != 1 || Paced(10, 0.5, 0, 1) != 10 || Paced(0, 0.5, 0, 3) != 0 {
		t.Fatal("paced edge cases")
	}
	for _, c := range []struct {
		replay     float64
		n, pool, r int
	}{{0.25, 10, 100, 3}, {0.24, 10, 100, 2}, {5, 10, 7, 7}, {1e300, 10, 7, 7}, {0, 10, 7, 0}, {1, 10, 0, 0}} {
		if got := ReplayCount(c.replay, c.n, c.pool); got != c.r {
			t.Errorf("ReplayCount(%v, %d, %d) = %d, want %d", c.replay, c.n, c.pool, got, c.r)
		}
	}
}

func TestReplayBufferIsBottomKWhateverTheOrder(t *testing.T) {
	texts := make([]string, 50)
	for i := range texts {
		texts[i] = "text " + string(rune('a'+i%26)) + string(rune('0'+i/26))
	}
	one := NewReplayBuffer(8, 7, 0, nil, nil)
	one.Offer(texts)
	two := NewReplayBuffer(8, 7, 0, nil, nil)
	for i := 0; i < 50; i += 3 {
		end := i + 3
		if end > 50 {
			end = 50
		}
		two.Offer(texts[i:end])
	}
	if !reflect.DeepEqual(one.Doc(), two.Doc()) {
		t.Fatal("the order of the offers changed the buffer")
	}
	want := make([]int64, 50)
	for g := range want {
		want[g] = int64(g)
	}
	sort.Slice(want, func(a, b int) bool {
		ka, kb := ReplayKey(7, want[a]), ReplayKey(7, want[b])
		if ka != kb {
			return ka < kb
		}
		return want[a] < want[b]
	})
	if !reflect.DeepEqual(one.Doc().Index, want[:8]) || one.Seen != 50 {
		t.Fatalf("buffer %v (seen %d), want %v", one.Doc().Index, one.Seen, want[:8])
	}
	back, err := ReplayFromDoc(one.Doc(), 7)
	if err != nil || !reflect.DeepEqual(back.Doc(), one.Doc()) {
		t.Fatalf("round trip: %v", err)
	}
	if _, err := ReplayFromDoc(&ReplayDoc{Size: 3, Index: []int64{0, 1}, Texts: []string{"a"}}, 7); err == nil {
		t.Fatal("a block with more indices than texts was read")
	}
	if _, err := ReplayFromDoc(&ReplayDoc{Size: -1, Index: []int64{0}, Texts: []string{"a"}}, 7); err == nil {
		t.Fatal("a negative size was read")
	}
	var bare ReplayDoc // a block without size or seen holds as many as it lists, as Python reads it
	if err := json.Unmarshal([]byte(`{"index": [4, 2], "texts": ["x", "y"]}`), &bare); err != nil || bare.Size != 2 || bare.Seen != 2 {
		t.Fatalf("defaults: %+v %v", bare, err)
	}
}

func TestEarlyStopAndThePlan(t *testing.T) {
	stop := NewEarlyStop(2, 0.1)
	if stop.Update(5) || stop.Update(4.95) || !stop.Update(4.99) {
		t.Fatal("patience 2, min_delta 0.1")
	}
	plan, err := NewTrainingPlan([]string{"ccc", "a", "bb", "dddd"}, []int{3, 1, 2, 4}, 1, 3,
		Plan{Order: "shortest-first", Curriculum: 0.5}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for j, want := range [][]string{{"a", "bb"}, {"a", "bb", "ccc"}, {"a", "bb", "ccc", "dddd"}} {
		if got := plan.Epoch(j, int64(j+1)); !reflect.DeepEqual(got, want) {
			t.Errorf("epoch %d: %v, want %v", j, got, want)
		}
	}
	for _, bad := range []Plan{{Order: "random"}, {Curriculum: 1.5}, {Replay: -1}, {Patience: -1}, {MinDelta: -1}} {
		if bad.Check() == nil {
			t.Errorf("%+v was accepted", bad)
		}
	}
	negative := -1
	if (Plan{ReplaySize: &negative}).Check() == nil {
		t.Error("a negative replay size was accepted")
	}
}

func TestPlannedTrainingKeepsABufferAndStops(t *testing.T) {
	texts := corpus(t)
	m, err := NewModel(2, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	size := 6
	opts := DefaultTrainOptions()
	opts.Epochs = 6
	opts.Plan = Plan{ReplaySize: &size, Patience: 1, MinDelta: 100}
	records, err := m.Train(texts, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[1]["early_stop"] != true || records[0]["early_stop"] != nil {
		t.Fatalf("early stop: %d records, %v", len(records), records[len(records)-1]["early_stop"])
	}
	if m.Replay == nil || m.Replay.Len() != 6 || m.Replay.Seen != int64(len(texts)) {
		t.Fatalf("buffer: %+v", m.Replay)
	}
	before := m.Replay.Doc()
	// a feedback pass - a thumbs up, or any pass stamped with a phase - leaves the buffer alone
	if _, err := m.Reward(texts[:3], 1, 1); err != nil {
		t.Fatal(err)
	}
	stamped := DefaultTrainOptions()
	stamped.Epochs = 1
	stamped.Phase = "negative"
	if _, err := m.Train(texts[3:6], stamped); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.Replay.Doc(), before) {
		t.Fatal("a feedback pass changed the replay buffer")
	}
	doc := m.ToDoc()
	if doc.Replay == nil || !reflect.DeepEqual(doc.Replay, before) {
		t.Fatal("the buffer is not in the document")
	}
}
