package radixnet

import (
	"fmt"
	"time"
)

// The negative network feeding itself: an LLM reviewer on a loop.
//
// Every other tutor hands the negative network its failures as a side effect of
// doing something else - the English tutor marks a sentence, the recall tutor
// finds a misremembered waveform.  This is the loop that needs nothing else
// running, so nobody has to type a failure in by hand:
//
//  1. the positive model writes Count texts of its own (a stochastic walk,
//     optionally continuing a prefix);
//  2. an LLM reviewer marks each one out of 10, passes or fails it against the
//     threshold and writes a one-sentence critique (AdversarialReview).  Any
//     LLMClient will do, so a local Ollama model reviews by default and ChatGPT
//     reviews when the provider says so;
//  3. the failures blame the negative network, with the critique picking the
//     reason and the mark setting the severity, and the passes clear blame off
//     the fragments they share with known failures (TeachReviews).
//
// Then it does it again.  Nothing is invented: every failure still arrives from
// something outside the network that looked at an output and said it was wrong,
// and why - the only change is that nobody has to sit there doing it.
//
// With Correct the reviewer is a copy editor instead: it writes each text out
// correctly with as few characters changed as it can (CorrectTexts), and the
// diff between the two is the lesson - only the characters it changed are
// blamed, at Severity per corrected text, and a text it handed back unchanged
// clears blame (TeachCorrections).  There is no mark and no pass mark in that
// mode; the change rate (the share of texts it had to change) takes the place
// of the mean rating in the round records and the report card.
//
// The loop touches the negative network only.  It never trains, rewards or
// inverts the positive model, so it can be left running beside anything else
// that is teaching it.

// CriticConfig is how the reviewer is run, and what it is told.
type CriticConfig struct {
	// Rounds to run; 0 means keep going until something stops it.
	Rounds int `json:"rounds"`
	// Count is the texts the model writes per round, for the reviewer to mark.
	Count int `json:"count"`
	// Prefix continues this instead of writing from scratch.
	Prefix      string  `json:"prefix"`
	MaxLength   int     `json:"max_length"`
	Temperature float64 `json:"temperature"`
	// Threshold is the pass mark out of 10: below it the text is a failure.
	Threshold float64 `json:"threshold"`
	// Context is what the reviewer is told the texts are meant to be (its yardstick).
	Context  string `json:"context"`
	Provider string `json:"provider"`
	// ReviewerModel is the reviewer's model name; empty means the client's own default.
	ReviewerModel string `json:"reviewer_model"`
	// ClearPasses lets the texts the reviewer passed take blame off what they share.
	ClearPasses bool `json:"clear_passes"`
	// Epochs of blaming per round.
	Epochs int `json:"epochs"`
	// Seed of the first round's sampling; later rounds advance it, so rounds differ.
	Seed *int64 `json:"seed"`
	// Correct asks for letter-level corrections instead of marks: only the
	// characters the editor changed are blamed.
	Correct bool `json:"correct"`
	// Severity is how heavily one corrected text is blamed (Correct mode; a
	// marked failure's severity comes from its mark).
	Severity float64 `json:"severity"`
}

// DefaultCriticConfig mirrors the Python defaults.
func DefaultCriticConfig() CriticConfig {
	return CriticConfig{
		Rounds: 3, Count: 8, MaxLength: 60, Temperature: 1, Threshold: 6,
		Provider: ProviderOllama, ClearPasses: true, Epochs: 1, Severity: CorrectionSeverity,
	}
}

// Validate checks the configuration.
func (c *CriticConfig) Validate() error {
	if c.Rounds < 0 {
		return fmt.Errorf("rounds must be >= 0 (0 = until stopped)")
	}
	if c.Count < 1 {
		return fmt.Errorf("count must be >= 1")
	}
	if c.MaxLength < 0 {
		return fmt.Errorf("max_length must be >= 0")
	}
	if c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.Threshold < 0 || c.Threshold > 10 {
		return fmt.Errorf("threshold must lie in [0, 10]")
	}
	if c.Epochs < 0 {
		return fmt.Errorf("epochs must be >= 0")
	}
	if c.Severity < 0 {
		return fmt.Errorf("severity must be >= 0")
	}
	if _, err := NormaliseProvider(c.Provider); err != nil {
		return fmt.Errorf("provider must be one of: ollama, chatgpt")
	}
	return nil
}

// Critic is the reviewer on a loop, keeping the negative network fed.
type Critic struct {
	Model    *Model // the positive model whose output is reviewed (only read from)
	Negative *Model // the negative network that learns from the verdicts
	Client   LLMClient
	Config   CriticConfig
	// External wraps the slow LLM call (the server releases its model lock around it).
	External func(func() error) error
	// Progress receives every round and report record as it happens.
	Progress func(map[string]any)
	// Stop is polled between rounds; a true answer ends the run cleanly.
	Stop func() bool

	History []map[string]any
	RoundNo int
}

// NewCritic validates the configuration and returns the loop.
func NewCritic(model, negative *Model, client LLMClient, config CriticConfig) (*Critic, error) {
	if model == nil {
		return nil, fmt.Errorf("a positive model to review is required")
	}
	if negative == nil {
		return nil, fmt.Errorf("a negative network to teach is required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Critic{Model: model, Negative: negative, Client: client, Config: config}, nil
}

// external runs fn, through External when one is set.
func (c *Critic) external(fn func() error) error {
	if c.External == nil {
		return fn()
	}
	return c.External(fn)
}

// seed is this round's sampling seed: the configured one advanced by the round,
// so a seeded run is reproducible and its rounds still differ.
func (c *Critic) seed() *int64 {
	if c.Config.Seed == nil {
		return nil
	}
	next := *c.Config.Seed + int64(c.RoundNo)
	return &next
}

// RunRound writes, reviews (or corrects) and blames once, returning the
// round's record.
func (c *Critic) RunRound() (map[string]any, error) {
	cfg := c.Config
	c.RoundNo++
	started := time.Now()
	// The model writes first, under whatever lock the caller holds: sampling
	// walks the graph, so it must not happen while another request may mutate it.
	samples, err := SampleTexts(c.Model, cfg.Count, cfg.Prefix, cfg.MaxLength, cfg.Temperature, c.seed())
	if err != nil {
		return nil, err
	}
	if cfg.Correct {
		return c.correctRound(samples, started)
	}
	// Only the reviewer's thinking happens outside the lock - it is a network
	// call that touches nothing of ours.
	var reviews []Review
	if err := c.external(func() error {
		out, err := ReviewTexts(c.Client, samples, cfg.Context, cfg.ReviewerModel, cfg.Threshold, DefaultReviewBatch)
		reviews = out
		return err
	}); err != nil {
		return nil, err
	}
	reviewer := cfg.ReviewerModel
	if reviewer == "" {
		reviewer = c.Client.ModelName()
	}
	review := SummariseReviews("model", reviewer, cfg.Threshold, samples, reviews)
	taught, err := TeachReviews(c.Negative, review.Reviews, cfg.Threshold, cfg.ClearPasses, "critic",
		TeachOptions{Epochs: cfg.Epochs, Stop: c.Stop})
	if err != nil {
		return nil, err
	}
	passed, failed := len(review.Good), len(review.Bad)
	record := map[string]any{
		"kind":          "round",
		"round":         c.RoundNo,
		"reviewer":      review.Model,
		"mode":          "review",
		"threshold":     cfg.Threshold,
		"texts":         len(review.Texts),
		"reviews":       review.Reviews,
		"mean_rating":   review.MeanRating,
		"pass_rate":     review.PassRate,
		"passed":        passed,
		"failed":        failed,
		"blamed":        taught.Blamed,
		"cleared":       taught.Cleared,
		"unmatched":     taught.Unmatched,
		"edges":         taught.Edges,
		"reasons":       taught.Reasons,
		"severity_mean": taught.SeverityMean,
		"stats":         c.Negative.Stats(),
		"seconds":       time.Since(started).Seconds(),
	}
	c.History = append(c.History, record)
	return record, nil
}

// correctRound is the editor's round: every text is written out correctly and
// only the diff is blamed.
func (c *Critic) correctRound(samples []string, started time.Time) (map[string]any, error) {
	cfg := c.Config
	var corrections []CorrectionEntry
	if err := c.external(func() error {
		out, err := CorrectTexts(c.Client, samples, cfg.Context, cfg.ReviewerModel, DefaultCorrectionBatch)
		corrections = out
		return err
	}); err != nil {
		return nil, err
	}
	editor := cfg.ReviewerModel
	if editor == "" {
		editor = c.Client.ModelName()
	}
	result := SummariseCorrections("model", editor, samples, corrections)
	taught, err := TeachCorrections(c.Negative, result.Corrections, cfg.Severity, cfg.ClearPasses, "critic",
		TeachOptions{Epochs: cfg.Epochs, Stop: c.Stop})
	if err != nil {
		return nil, err
	}
	record := map[string]any{
		"kind":          "round",
		"round":         c.RoundNo,
		"reviewer":      result.Model,
		"mode":          "correct",
		"severity":      cfg.Severity,
		"texts":         len(result.Texts),
		"corrections":   result.Corrections,
		"change_rate":   result.ChangeRate,
		"corrected":     len(result.Corrected),
		"unchanged":     len(result.Unchanged),
		"uncorrected":   len(result.Uncorrected),
		"edits":         result.Edits,
		"wrong_chars":   result.WrongChars,
		"blamed":        taught.Blamed,
		"cleared":       taught.Cleared,
		"unmatched":     taught.Unmatched,
		"edges":         taught.Edges,
		"reasons":       taught.Reasons,
		"severity_mean": taught.SeverityMean,
		"stats":         c.Negative.Stats(),
		"seconds":       time.Since(started).Seconds(),
	}
	c.History = append(c.History, record)
	return record, nil
}

// Run loops until Config.Rounds is reached, or forever when it is 0, checking
// Stop between rounds so stopping finishes the round it is in rather than
// abandoning a half-taught one.  The records are the round records followed by
// one report record summarising them.
func (c *Critic) Run() ([]map[string]any, error) {
	limit := c.Config.Rounds
	records := []map[string]any{}
	for limit == 0 || len(records) < limit {
		if c.Stop != nil && c.Stop() {
			break
		}
		record, err := c.RunRound()
		if err != nil {
			return records, err
		}
		records = append(records, record)
		if c.Progress != nil {
			c.Progress(record)
		}
	}
	card := CriticReportCard(records)
	records = append(records, card)
	if c.Progress != nil {
		c.Progress(card)
	}
	return records, nil
}

// CriticReportCard is what a run of rounds came to: how much was reviewed,
// blamed and cleared, and why.  Trend is the last round's mean mark minus the
// first's - positive when the reviewer is marking the output better than it did
// at the start.  Rounds run by the copy editor (Correct) add corrected,
// unchanged, uncorrected, edits, change_rate and change_trend (the last
// round's change rate minus the first's - negative when the editor has less to
// put right than it had at the start).
func CriticReportCard(records []map[string]any) map[string]any {
	rounds := make([]map[string]any, 0, len(records))
	for _, record := range records {
		if record["kind"] == "round" {
			rounds = append(rounds, record)
		}
	}
	reasons := map[string]int{}
	reviewed, blamed, cleared, edges := 0, 0, 0, 0
	ratings := []float64{}
	rates := []float64{}
	changes := []float64{}
	corrected, unchanged, uncorrected, edits, correcting := 0, 0, 0, 0, false
	for _, record := range rounds {
		reviewed += intOf(record["texts"])
		blamed += intOf(record["blamed"])
		cleared += intOf(record["cleared"])
		edges += intOf(record["edges"])
		if record["mode"] == "correct" {
			correcting = true
			corrected += intOf(record["corrected"])
			unchanged += intOf(record["unchanged"])
			uncorrected += intOf(record["uncorrected"])
			edits += intOf(record["edits"])
		}
		if value, ok := record["change_rate"].(*float64); ok && value != nil {
			changes = append(changes, *value)
		}
		if counts, ok := record["reasons"].(map[string]int); ok {
			for reason, count := range counts {
				reasons[reason] += count
			}
		}
		if value, ok := record["mean_rating"].(*float64); ok && value != nil {
			ratings = append(ratings, *value)
		}
		if value, ok := record["pass_rate"].(*float64); ok && value != nil {
			rates = append(rates, *value)
		}
	}
	card := map[string]any{
		"kind": "report", "rounds": len(rounds), "reviewed": reviewed, "blamed": blamed,
		"cleared": cleared, "edges": edges, "reasons": reasons,
		"mean_rating": meanOrNil(ratings), "pass_rate": meanOrNil(rates), "trend": nil, "stats": nil,
	}
	if len(ratings) > 1 {
		card["trend"] = ratings[len(ratings)-1] - ratings[0]
	}
	if len(rounds) > 0 {
		card["stats"] = rounds[len(rounds)-1]["stats"]
	}
	if correcting {
		// the editor's rounds: how often it had to change something, and how much
		card["corrected"], card["unchanged"], card["uncorrected"], card["edits"] = corrected, unchanged, uncorrected, edits
		card["change_rate"], card["change_trend"] = meanOrNil(changes), nil
		if len(changes) > 1 {
			card["change_trend"] = changes[len(changes)-1] - changes[0]
		}
	}
	return card
}

// intOf reads an int out of a record field however it was stored.
func intOf(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}

// meanOrNil is the mean of the values, or nil when there are none (so the JSON
// says "nothing was marked" rather than "marked zero").
func meanOrNil(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	mean := sum / float64(len(values))
	return &mean
}
