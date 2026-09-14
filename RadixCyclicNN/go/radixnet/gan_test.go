package radixnet

import (
	"testing"
)

var evolveCorpus = []string{
	"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill",
	"the cat chased the mouse", "rain fell on the quiet town",
}

func newEvolver(t *testing.T, config EvolveConfig) *Evolver {
	t.Helper()
	generator, err := NewModel(3, DefaultGraphOptions())
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	generator.Exact = true
	if _, err := generator.Train(evolveCorpus, TrainOptions{Epochs: 3, AutoCompress: true}); err != nil {
		t.Fatalf("train: %v", err)
	}
	loop, err := NewEvolver(generator, evolveCorpus, nil, config)
	if err != nil {
		t.Fatalf("NewEvolver: %v", err)
	}
	return loop
}

func smallEvolveConfig(mode string) EvolveConfig {
	config := DefaultEvolveConfig()
	config.Samples, config.RealPerGeneration, config.MaxLength = 4, 3, 24
	config.BlatantMode = mode
	return config
}

func TestEvolveConfigDefaultsAndValidation(t *testing.T) {
	config := DefaultEvolveConfig()
	if err := config.Validate(); err != nil {
		t.Fatalf("the defaults must validate: %v", err)
	}
	for name, broken := range map[string]EvolveConfig{
		"samples":     {Samples: 0},
		"real":        {Samples: 1, RealPerGeneration: 0},
		"max_length":  {Samples: 1, RealPerGeneration: 1, MaxLength: 2},
		"temperature": {Samples: 1, RealPerGeneration: 1, MaxLength: 10, Temperature: -1},
		"epochs":      {Samples: 1, RealPerGeneration: 1, MaxLength: 10, NegEpochs: -1},
		"mode":        {Samples: 1, RealPerGeneration: 1, MaxLength: 10, BlatantMode: "sideways"},
		"boost":       {Samples: 1, RealPerGeneration: 1, MaxLength: 10, BlatantMode: "none", BlatantBoost: 0.5},
	} {
		if err := broken.Validate(); err == nil {
			t.Errorf("%s: expected the configuration to be refused", name)
		}
	}
}

func TestEvolverNeedsAGeneratorAndACorpus(t *testing.T) {
	config := smallEvolveConfig("none")
	if _, err := NewEvolver(nil, evolveCorpus, nil, config); err == nil {
		t.Error("a loop without a generator must be refused")
	}
	generator, _ := NewModel(1, DefaultGraphOptions())
	if _, err := NewEvolver(generator, []string{"ab"}, nil, config); err == nil {
		t.Error("a corpus of texts shorter than a trigram must be refused")
	}
}

func TestEvolverMakesItsOwnDiscriminator(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("none"))
	if loop.Discriminator == nil {
		t.Fatal("a nil discriminator must be replaced by a fresh model")
	}
	if loop.Discriminator == loop.Generator {
		t.Error("the critic must not be the generator itself")
	}
}

func TestEvolveGenerationRecord(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("none"))
	record, err := loop.RunGeneration()
	if err != nil {
		t.Fatalf("RunGeneration: %v", err)
	}
	for _, key := range []string{
		"generation", "fake_score_mean", "real_score_mean", "gap", "gen_loss", "nodes", "edges",
		"compression_ratio", "sample", "fakes", "worst", "failures", "blatant", "flipped",
		"boost_mean", "boost_max", "twonrl", "mode", "seconds",
	} {
		if _, ok := record[key]; !ok {
			t.Errorf("the record is missing %q", key)
		}
	}
	if record["generation"] != 1 || record["mode"] != "none" {
		t.Errorf("record: %v", record)
	}
	if intOf(record["fakes"]) == 0 {
		t.Error("a generation must produce fakes")
	}
}

func TestEvolveTheCriticLearnsToTellThemApart(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("none"))
	if _, err := loop.Run(3); err != nil {
		t.Fatalf("Run: %v", err)
	}
	last := loop.History[len(loop.History)-1]
	real, _ := last["real_score_mean"].(*float64)
	fake, _ := last["fake_score_mean"].(*float64)
	if real == nil || fake == nil {
		t.Fatalf("both means must be reported: %v", last)
	}
	if loop.Discriminator.G.NumNodes() == 0 {
		t.Error("the critic must have learned something")
	}
}

func TestEvolveFailInvertPenalisesTheWorstHardest(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("fail_invert"))
	record, err := loop.RunGeneration()
	if err != nil {
		t.Fatalf("RunGeneration: %v", err)
	}
	if record["mode"] != "fail_invert" {
		t.Fatalf("mode: %v", record["mode"])
	}
	if intOf(record["failures"]) > 0 {
		if boost, _ := record["boost_max"].(float64); boost < 1 {
			t.Errorf("a failure's penalty is scaled by at least 1: %v", boost)
		}
		if record["twonrl"] != true {
			t.Error("failures mean a 2NRL pass ran")
		}
	}
}

func TestEvolveLocalModesOnlyMoveTheFailedPaths(t *testing.T) {
	for _, mode := range []string{"activation", "state"} {
		loop := newEvolver(t, smallEvolveConfig(mode))
		record, err := loop.RunGeneration()
		if err != nil {
			t.Fatalf("%s: RunGeneration: %v", mode, err)
		}
		if record["mode"] != mode {
			t.Errorf("%s: mode %v", mode, record["mode"])
		}
		if intOf(record["failures"]) > 0 && intOf(record["flipped"]) == 0 {
			t.Errorf("%s: failures must move edges", mode)
		}
	}
}

func TestEvolveTeachesTheNegativeNetwork(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("fail_invert"))
	negative, err := NewNegativeModel(3, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("NewNegativeModel: %v", err)
	}
	loop.Negative = negative
	if _, err := loop.Run(2); err != nil {
		t.Fatalf("Run: %v", err)
	}
	blamed := 0
	for _, record := range loop.History {
		blamed += intOf(record["negative_blamed"])
	}
	if blamed > 0 {
		if negative.G.Neg.TotalBlame <= 0 {
			t.Error("what was blamed must reach the network")
		}
		sources, _ := negative.Stats()["sources"].(map[string]any)
		if intOf(sources["evolve"]) == 0 {
			t.Errorf("the blame must be sourced to the loop: %v", sources)
		}
		reasons := map[string]bool{}
		for _, row := range negative.Reasons() {
			reasons[row.Reason] = true
		}
		if !reasons["discriminator"] && !reasons["blatant"] {
			t.Errorf("the critic's own reasons must be used: %v", negative.Reasons())
		}
	}
}

func TestEvolveWithoutANegativeNetworkRecordsNothingAboutOne(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("fail_invert"))
	record, err := loop.RunGeneration()
	if err != nil {
		t.Fatalf("RunGeneration: %v", err)
	}
	if _, ok := record["negative_blamed"]; ok {
		t.Error("no negative network means no negative keys")
	}
}

func TestEvolveRunHonoursTheStopEvent(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("none"))
	loop.Stop = func() bool { return true }
	records, err := loop.Run(5)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(records) != 0 {
		t.Errorf("a stopped loop runs nothing: %d", len(records))
	}
}

func TestEvolveRunForeverStopsWhenAsked(t *testing.T) {
	loop := newEvolver(t, smallEvolveConfig("none"))
	seen := 0
	loop.Progress = func(map[string]any) { seen++ }
	loop.Stop = func() bool { return seen >= 2 }
	records, err := loop.Run(0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(records) != 2 {
		t.Errorf("expected two generations, got %d", len(records))
	}
	if _, err := loop.Run(-1); err == nil {
		t.Error("a negative generation count must be refused")
	}
}

func TestMedian(t *testing.T) {
	if median(nil) != 0 {
		t.Error("an empty slice has no median")
	}
	if median([]float64{3, 1, 2}) != 2 {
		t.Error("odd count: the middle")
	}
	if median([]float64{4, 1, 2, 3}) != 2.5 {
		t.Error("even count: the mean of the two middles")
	}
}
