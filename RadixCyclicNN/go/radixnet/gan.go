package radixnet

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// The self-upgrade loop: the model is the generator, a second network is the
// discriminator, and each generation the critic learns real-from-fake while the
// generator learns from what the critic rejected.
//
// On the count / reward model there is no activation to flip, so a "failure" is
// pushed away by penalising every edge of its path in proportion to how badly
// the critic scored it (InvertPaths); everything else is the Python loop:
//
//  1. the generator samples Samples fakes;
//  2. the discriminator runs 2NRL over (fakes, a sample of the real corpus) and
//     then scores both sides - per character, so long and short texts compare;
//  3. the fakes it scored below the real texts are the failures, and how far
//     below is how bad each one is (the gap g, in nats per character);
//  4. BlatantMode decides what the generator does with them.
//
// With a negative network attached the discriminator is also its tutor: every
// failure is blamed by how far below it landed and the real texts clear blame.

// BlatantModes are how failed fakes drive the generator's update.
var BlatantModes = []string{"none", "fail_invert", "activation", "state"}

// EvolveConfig is the loop's hyper-parameters.
type EvolveConfig struct {
	Samples           int     `json:"samples"`
	RealPerGeneration int     `json:"real_per_generation"`
	MaxLength         int     `json:"max_length"`
	Temperature       float64 `json:"temperature"`
	NegEpochs         int     `json:"neg_epochs"`
	PosEpochs         int     `json:"pos_epochs"`
	DiscNegEpochs     int     `json:"disc_neg_epochs"`
	DiscPosEpochs     int     `json:"disc_pos_epochs"`
	Strength          float64 `json:"strength"`
	Seed              int64   `json:"seed"`
	// BlatantMode: "none" (the worst half is ordinary 2NRL garbage), "fail_invert"
	// (every failure is penalised in proportion to how bad it is), or
	// "activation" / "state" (the local variant: only the failed paths move).
	BlatantMode string `json:"blatant_mode"`
	// BlatantMargin is the per-character gap at which a failure counts as blatant.
	BlatantMargin float64 `json:"blatant_margin"`
	// BlatantBoost caps the multiplier a failure's penalty is scaled by.
	BlatantBoost float64 `json:"blatant_boost"`
}

// DefaultEvolveConfig mirrors the Python defaults.
func DefaultEvolveConfig() EvolveConfig {
	return EvolveConfig{
		Samples: 8, RealPerGeneration: 8, MaxLength: 40, Temperature: 1, NegEpochs: 1, PosEpochs: 1,
		DiscNegEpochs: 1, DiscPosEpochs: 1, Strength: 1, BlatantMode: "none", BlatantMargin: 1, BlatantBoost: 4,
	}
}

// Validate checks the configuration.
func (c *EvolveConfig) Validate() error {
	if c.Samples < 1 {
		return fmt.Errorf("samples must be >= 1")
	}
	if c.RealPerGeneration < 1 {
		return fmt.Errorf("real_per_generation must be >= 1")
	}
	if c.MaxLength < Window {
		return fmt.Errorf("max_length must be >= %d", Window)
	}
	if c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.NegEpochs < 0 || c.PosEpochs < 0 || c.DiscNegEpochs < 0 || c.DiscPosEpochs < 0 {
		return fmt.Errorf("epochs must be >= 0")
	}
	if c.Strength < 0 {
		return fmt.Errorf("strength must be >= 0")
	}
	if !contains(BlatantModes, c.BlatantMode) {
		return fmt.Errorf("blatant_mode must be one of: none, fail_invert, activation, state")
	}
	if c.BlatantMargin < 0 {
		return fmt.Errorf("blatant_margin must be >= 0")
	}
	if c.BlatantBoost < 1 {
		return fmt.Errorf("blatant_boost must be >= 1")
	}
	return nil
}

// Evolver runs the loop.
type Evolver struct {
	Generator     *Model
	Discriminator *Model
	Corpus        []string
	Config        EvolveConfig
	// Negative is an optional negative network: here the discriminator is the
	// critic, so every fake it scores below the real texts is blamed.
	Negative *Model
	Progress func(map[string]any)
	Stop     func() bool

	History    []map[string]any
	Generation int
	rng        *rand.Rand
}

// NewEvolver validates the configuration and builds the loop; a nil
// discriminator gets a fresh model of the same kind, seeded beside the
// generator's.
func NewEvolver(generator *Model, corpus []string, discriminator *Model, config EvolveConfig) (*Evolver, error) {
	if generator == nil {
		return nil, fmt.Errorf("a generator is required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	kept := []string{}
	for _, text := range corpus {
		if len([]rune(text)) >= Window {
			kept = append(kept, text)
		}
	}
	if len(kept) == 0 {
		return nil, fmt.Errorf("corpus needs at least one text of %d+ characters", Window)
	}
	if discriminator == nil {
		fresh, err := NewModel(config.Seed+1, DefaultGraphOptions())
		if err != nil {
			return nil, err
		}
		fresh.Workers, fresh.Exact = generator.Workers, generator.Exact
		discriminator = fresh
	}
	return &Evolver{
		Generator: generator, Discriminator: discriminator, Corpus: kept, Config: config,
		rng: rand.New(rand.NewSource(config.Seed)),
	}, nil
}

// fakes samples this generation's fakes, falling back to the cheapest path when
// every sample came out shorter than a trigram.
func (e *Evolver) fakes() ([]string, error) {
	cfg := e.Config
	results, err := e.Generator.Generate(GenerateOptions{
		MaxLength: cfg.MaxLength, Mode: "sample", Temperature: cfg.Temperature, Count: cfg.Samples,
	})
	if err != nil {
		return nil, err
	}
	out := []string{}
	for _, result := range results {
		if len([]rune(result.Text)) >= Window {
			out = append(out, result.Text)
		}
	}
	if len(out) == 0 {
		best, err := e.Generator.Generate(GenerateOptions{MaxLength: cfg.MaxLength, Mode: "beam", Count: 1})
		if err != nil {
			return nil, err
		}
		if len(best) > 0 && len([]rune(best[0].Text)) >= Window {
			out = append(out, best[0].Text)
		}
	}
	return out, nil
}

// realSample draws this generation's real texts.
func (e *Evolver) realSample() []string {
	want := e.Config.RealPerGeneration
	if want > len(e.Corpus) {
		want = len(e.Corpus)
	}
	order := e.rng.Perm(len(e.Corpus))[:want]
	out := make([]string, 0, want)
	for _, at := range order {
		out = append(out, e.Corpus[at])
	}
	return out
}

// RunGeneration runs one generation and returns (and records) its summary.
func (e *Evolver) RunGeneration() (map[string]any, error) {
	cfg := e.Config
	started := time.Now()
	fakes, err := e.fakes()
	if err != nil {
		return nil, err
	}
	real := e.realSample()

	// the critic learns real from fake, then marks both
	if _, err := e.Discriminator.TwoNRL(fakes, real, cfg.DiscNegEpochs, cfg.DiscPosEpochs, cfg.Strength); err != nil {
		return nil, err
	}
	fakeScores := make([]float64, len(fakes))
	for i, fake := range fakes {
		fakeScores[i] = e.Discriminator.Score(fake).PerChar
	}
	realScores := make([]float64, len(real))
	for i, text := range real {
		realScores[i] = e.Discriminator.Score(text).PerChar
	}
	fakeMean, realMean := meanOrNil(fakeScores), meanOrNil(realScores)

	worst := []string{}
	if len(fakes) > 0 {
		cut := median(fakeScores)
		for i, fake := range fakes {
			if fakeScores[i] <= cut {
				worst = append(worst, fake)
			}
		}
		if len(worst) == 0 {
			worst = []string{fakes[0]}
		}
	}

	// the failures: fakes the critic scored below the real texts, each with its gap
	failures, gaps, blatant := []string{}, []float64{}, []string{}
	if cfg.BlatantMode != "none" && realMean != nil && cfg.BlatantMargin > 0 {
		for i, fake := range fakes {
			gap := *realMean - fakeScores[i]
			if gap <= 0 {
				continue
			}
			failures = append(failures, fake)
			gaps = append(gaps, gap)
			if gap > cfg.BlatantMargin {
				blatant = append(blatant, fake)
			}
		}
	}

	var positive []map[string]any
	twonrl, flipped := false, 0
	boostMean, boostMax := 0.0, 0.0
	switch {
	case cfg.BlatantMode == "fail_invert" && len(failures) > 0:
		// blatantly fail on purpose: the worse the fake, the harder it is pushed away
		weights := make([]float64, len(gaps))
		for i, gap := range gaps {
			weights[i] = math.Min(cfg.BlatantBoost, 1+gap/cfg.BlatantMargin)
			boostMax = math.Max(boostMax, weights[i])
		}
		result, err := e.Generator.TwoNRLWeighted(failures, weights, real, nil, TwoNRLOptions{
			NegEpochs: cfg.NegEpochs, PosEpochs: cfg.PosEpochs, Strength: cfg.Strength,
		})
		if err != nil {
			return nil, err
		}
		positive, worst, twonrl = result.Positive, failures, true
		if mean := meanOrNil(weights); mean != nil {
			boostMean = *mean
		}
	case cfg.BlatantMode == "fail_invert":
		worst = nil // nothing scored below the real texts: no failure to train on
		fallthrough
	default:
		if (cfg.BlatantMode == "activation" || cfg.BlatantMode == "state") && len(failures) > 0 {
			// the local variant: only the failed paths move, the worse the more
			amounts := make([]float64, len(gaps))
			for i, gap := range gaps {
				amounts[i] = math.Min(1, gap/(2*cfg.BlatantMargin))
				boostMax = math.Max(boostMax, amounts[i])
			}
			outcome, err := e.Generator.InvertPaths(failures, amounts, cfg.Strength)
			if err != nil {
				return nil, err
			}
			flipped, boostMean = outcome.Flipped, outcome.AmountMean
			isBlatant := map[string]bool{}
			for _, text := range blatant {
				isBlatant[text] = true
			}
			kept := []string{}
			for _, text := range worst {
				if !isBlatant[text] {
					kept = append(kept, text)
				}
			}
			worst = kept
		}
		if len(worst) > 0 {
			result, err := e.Generator.TwoNRL(worst, real, cfg.NegEpochs, cfg.PosEpochs, cfg.Strength)
			if err != nil {
				return nil, err
			}
			positive, twonrl = result.Positive, true
		} else {
			// nothing left for the negative pass: only the fine-tune pass, no inversion
			records, err := e.Generator.Reward(real, cfg.PosEpochs, cfg.Strength)
			if err != nil {
				return nil, err
			}
			positive = records
		}
	}

	taught := e.teachNegative(fakes, fakeScores, realMean, real)
	e.Generation++
	g := e.Generator.G
	sample := ""
	if len(fakes) > 0 {
		sample = fakes[0]
	}
	var gap *float64
	if fakeMean != nil && realMean != nil {
		value := *realMean - *fakeMean
		gap = &value
	}
	record := map[string]any{
		"generation": e.Generation, "fake_score_mean": fakeMean, "real_score_mean": realMean, "gap": gap,
		"gen_loss": lastLoss(positive), "nodes": g.NumNodes(), "edges": g.NumEdges(),
		"compression_ratio": g.CompressionRatio(), "sample": sample, "fakes": len(fakes),
		"worst": len(worst), "failures": len(failures), "blatant": len(blatant), "flipped": flipped,
		"boost_mean": boostMean, "boost_max": boostMax, "twonrl": twonrl, "mode": cfg.BlatantMode,
		"seconds": time.Since(started).Seconds(),
	}
	if taught != nil {
		record["negative_blamed"] = taught.Blamed
		record["negative_edges"] = taught.Edges
		record["negative_reasons"] = taught.Reasons
	}
	e.History = append(e.History, record)
	return record, nil
}

// teachNegative hands this generation's failures to the negative network: the
// discriminator is its tutor.  Every fake the critic scored below the real texts
// is blamed by how far below it landed, and the real texts clear blame.
func (e *Evolver) teachNegative(fakes []string, scores []float64, realMean *float64, real []string) *TeachReport {
	if e.Negative == nil || realMean == nil {
		return nil
	}
	cfg := e.Config
	margin := cfg.BlatantMargin
	if margin <= 0 {
		margin = 1
	}
	faults := []Fault{}
	for i, fake := range fakes {
		gap := *realMean - scores[i]
		if gap <= 0 {
			continue
		}
		reason := "discriminator"
		if gap > cfg.BlatantMargin {
			reason = "blatant"
		}
		faults = append(faults, Fault{
			Text: fake, Reason: reason, Severity: math.Max(0.25, math.Min(2, gap/margin)),
			Note:   fmt.Sprintf("the discriminator scored it %.3f per char below the real texts", gap),
			Source: "evolve",
		})
	}
	if len(faults) == 0 {
		return &TeachReport{Reasons: map[string]int{}, Faults: faults, Records: []map[string]any{}}
	}
	report, err := Teach(e.Negative, faults, real, TeachOptions{Stop: e.Stop})
	if err != nil {
		return nil
	}
	return report
}

// Run runs generations generations, or until Stop says so when it is 0.
func (e *Evolver) Run(generations int) ([]map[string]any, error) {
	if generations < 0 {
		return nil, fmt.Errorf("generations must be >= 0 (0 = until stopped)")
	}
	records := []map[string]any{}
	for generations == 0 || len(records) < generations {
		if e.Stop != nil && e.Stop() {
			break
		}
		record, err := e.RunGeneration()
		if err != nil {
			return records, err
		}
		records = append(records, record)
		if e.Progress != nil {
			e.Progress(record)
		}
	}
	return records, nil
}

// median of a slice (the lower of the two middles for an even count, as
// Python's statistics.median does for the cut this loop needs).
func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64{}, values...)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[mid]
	}
	return (sorted[mid-1] + sorted[mid]) / 2
}
