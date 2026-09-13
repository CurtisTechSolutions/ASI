package radixnet

import (
	"fmt"
	"math"
	"sort"
)

// The two networks as one output path: the positive model writes, the negative
// one vetoes.
//
// This is the GAN at output time rather than at training time.  The finished
// pair works together on every answer:
//
//  1. the positive model (the count / reward model) proposes candidates - it is
//     the generator, and the only one of the two that can write;
//  2. the negative model judges each candidate - it is the discriminator, and
//     the only one of the two that knows what going wrong looks like;
//  3. what survives is returned, what does not comes back with the reason it
//     was dropped, the fragment to blame and who said so.
//
// Three signals decide, and any one of them is enough to reject:
//
//   - blame: Risk, the net evidence per transition, against Threshold (behind
//     the coverage gate, so text the tutor has never failed is never vetoed on
//     a hunch);
//   - peak (off by default): the evidence on a single transition, for vetoing a
//     candidate that carries one fragment the tutor has already corrected even
//     though the rest of it is fine;
//   - ratio: log P_negative(text) - log P_positive(text) per character, the
//     classic discriminator logit of two generative models.

// FilterConfig is how strictly the negative network filters the positive one's
// output.  The pointers are "unset" (the negative model's own thresholds) and,
// for Ratio and Peak, "off".
type FilterConfig struct {
	Threshold   *float64
	MinCoverage *float64
	// Ratio is the log-odds (negative minus positive, per character) at or above
	// which a candidate is rejected; nil turns the rule off.
	Ratio *float64
	// Peak is the evidence on a single transition at or above which a candidate
	// is rejected; nil turns the rule off.
	Peak *float64
	// OverSample is how many candidates are drawn per wanted output (0 = 3).
	OverSample int
	// Strict also drops candidates the negative network only finds suspect.
	Strict bool
	// Spans is how many blamed fragments a verdict reports (0 = 3).
	Spans int
	// Learn blames what this filter rejects (off: the tutor supplies the
	// negatives, the filter only applies them).
	Learn bool
	// Reason is recorded when Learn is on.
	Reason string
}

// DefaultFilterConfig: the negative model's own thresholds, the ratio rule at
// 0, no peak rule, three candidates per wanted text.
func DefaultFilterConfig() FilterConfig {
	ratio := 0.0
	return FilterConfig{Ratio: &ratio, OverSample: 3, Spans: 3, Reason: "filtered"}
}

// Validate rejects settings the filter cannot run with.
func (c FilterConfig) Validate() error {
	for name, value := range map[string]*float64{"threshold": c.Threshold, "min_coverage": c.MinCoverage,
		"ratio": c.Ratio, "peak": c.Peak} {
		if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return fmt.Errorf("%s must be a finite number or unset", name)
		}
	}
	if c.MinCoverage != nil && (*c.MinCoverage < 0 || *c.MinCoverage > 1) {
		return fmt.Errorf("min_coverage must lie in [0, 1], got %v", *c.MinCoverage)
	}
	if c.OverSample < 0 {
		return fmt.Errorf("over_sample must be >= 1, got %d", c.OverSample)
	}
	if c.Spans < 0 {
		return fmt.Errorf("spans must be >= 0, got %d", c.Spans)
	}
	return nil
}

func (c FilterConfig) overSample() int {
	if c.OverSample <= 0 {
		return 3
	}
	return c.OverSample
}

func (c FilterConfig) spans() int {
	if c.Spans <= 0 {
		return 3
	}
	return c.Spans
}

// Filter is the positive model's output filtered by the negative one.
type Filter struct {
	Positive *Model
	Negative *Model
	Config   FilterConfig
}

// NewFilter pairs a positive model with a negative one.
func NewFilter(positive, negative *Model, config FilterConfig) (*Filter, error) {
	if negative == nil || !negative.IsNegative() {
		return nil, fmt.Errorf("the negative model must be a negative network")
	}
	if positive == nil {
		return nil, fmt.Errorf("the positive model is missing")
	}
	if positive == negative {
		return nil, fmt.Errorf("the positive and negative models must be two different networks")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Filter{Positive: positive, Negative: negative, Config: config}, nil
}

// FilterVerdict is the pair's verdict on one text.
type FilterVerdict struct {
	Text           string       `json:"text"`
	Decision       string       `json:"decision"`
	Rule           *string      `json:"rule"`
	Risk           float64      `json:"risk"`
	Peak           float64      `json:"peak"`
	Coverage       float64      `json:"coverage"`
	Blame          float64      `json:"blame"`
	Threshold      float64      `json:"threshold"`
	MinCoverage    float64      `json:"min_coverage"`
	PeakThreshold  *float64     `json:"peak_threshold"`
	Ratio          float64      `json:"ratio"`
	RatioThreshold *float64     `json:"ratio_threshold"`
	Positive       float64      `json:"positive"`
	Negative       float64      `json:"negative"`
	Reasons        []ReasonRow  `json:"reasons"`
	Spans          []BlamedSpan `json:"spans"`
	Why            string       `json:"why"`
}

// Judge is the pair's verdict on one text: the negative network's blame, the
// likelihood ratio, and the decision.
func (f *Filter) Judge(text string) *FilterVerdict {
	cfg := f.Config
	verdict := f.Negative.Judge(text, JudgeOptions{Threshold: cfg.Threshold, MinCoverage: cfg.MinCoverage, Spans: cfg.spans()})
	positive := f.Positive.Score(text)
	negative := f.Negative.Score(text)
	ratio := negative.PerChar - positive.PerChar
	gate := verdict.Coverage >= verdict.MinCoverage && verdict.Blamed > 0
	rule := ""
	switch {
	case verdict.Verdict == "reject":
		rule = "blame"
	case cfg.Peak != nil && verdict.Peak >= *cfg.Peak && verdict.Blamed > 0:
		rule = "peak" // a single fragment the tutor has corrected is enough, whatever the rest of the text is
	case gate && cfg.Ratio != nil && ratio >= *cfg.Ratio:
		rule = "ratio"
	}
	decision := verdict.Verdict
	if rule != "" {
		decision = "reject"
	}
	if decision == "suspect" && cfg.Strict {
		decision, rule = "reject", "suspect"
	}
	out := &FilterVerdict{
		Text: text, Decision: decision, Risk: verdict.Risk, Peak: verdict.Peak,
		Coverage: verdict.Coverage, Blame: verdict.Blame, Threshold: verdict.Threshold,
		MinCoverage: verdict.MinCoverage, PeakThreshold: cfg.Peak, Ratio: ratio, RatioThreshold: cfg.Ratio,
		Positive: positive.PerChar, Negative: negative.PerChar, Reasons: verdict.Reasons, Spans: verdict.Spans,
	}
	if rule != "" {
		out.Rule = &rule // null when nothing rejected, as the Python filter reports it
	}
	out.Why = filterWhy(decision, rule, verdict, ratio)
	return out
}

// filterWhy is the one sentence a decision rests on.
func filterWhy(decision, rule string, verdict *Verdict, ratio float64) string {
	if decision == "pass" {
		return verdict.Why
	}
	switch rule {
	case "peak":
		where, reason := "", ""
		if len(verdict.Spans) > 0 {
			where = " at " + pythonRepr(verdict.Spans[0].Fragment)
			if verdict.Spans[0].Reason != "" {
				reason = fmt.Sprintf(" (%s)", verdict.Spans[0].Reason)
			}
		}
		return fmt.Sprintf("it carries %.2f of blame on one fragment%s%s; rejected", verdict.Peak, where, reason)
	case "ratio":
		reason := ""
		if len(verdict.Reasons) > 0 {
			reason = ", mostly " + pythonRepr(verdict.Reasons[0].Reason)
		}
		return fmt.Sprintf("it reads %.2f nats/char more like known failure than like the training data%s; rejected", ratio, reason)
	case "suspect":
		return replaceSuffix(verdict.Why, "; below the threshold, kept", "; rejected (strict)")
	}
	return verdict.Why
}

func replaceSuffix(text, suffix, replacement string) string {
	if len(text) >= len(suffix) && text[len(text)-len(suffix):] == suffix {
		return text[:len(text)-len(suffix)] + replacement
	}
	return text
}

// FilterOutcome is what a batch of candidates came to.
type FilterOutcome struct {
	Texts      []string         `json:"texts"`
	Kept       []string         `json:"kept"`
	Rejected   []*FilterVerdict `json:"rejected"`
	Verdicts   []*FilterVerdict `json:"verdicts"`
	Candidates int              `json:"candidates"`
	Asked      int              `json:"asked"`
	Rate       *float64         `json:"rate"`
}

// Filter judges every text; with Config.Learn the rejected ones are blamed as
// new failures.
func (f *Filter) Filter(texts []string) (*FilterOutcome, error) {
	out := &FilterOutcome{Texts: []string{}, Kept: []string{}, Rejected: []*FilterVerdict{},
		Verdicts: []*FilterVerdict{}, Candidates: len(texts), Asked: len(texts)}
	for _, text := range texts {
		verdict := f.Judge(text)
		out.Verdicts = append(out.Verdicts, verdict)
		if verdict.Decision == "reject" {
			out.Rejected = append(out.Rejected, verdict)
		} else {
			out.Kept = append(out.Kept, text)
		}
	}
	out.Texts = out.Kept
	if len(texts) > 0 {
		rate := float64(len(out.Kept)) / float64(len(texts))
		out.Rate = &rate
	}
	if len(out.Rejected) > 0 && f.Config.Learn {
		blamed := []string{}
		for _, verdict := range out.Rejected {
			if runeLen(verdict.Text) >= Window {
				blamed = append(blamed, verdict.Text)
			}
		}
		reason := f.Config.Reason
		if reason == "" {
			reason = "filtered"
		}
		if _, err := f.Negative.Blame(blamed, BlameOptions{Reason: reason, Source: "filter",
			Note: "rejected by the filter"}); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Generate writes through the pair: the positive model over-samples, the
// negative one vetoes, and the Count survivors with the least blame come back
// (fewer when the filter vetoed too much - that is information, not an error).
func (f *Filter) Generate(count int, o GenerateOptions) (*FilterOutcome, error) {
	if count < 0 {
		return nil, fmt.Errorf("count must be >= 0, got %d", count)
	}
	out := &FilterOutcome{Texts: []string{}, Kept: []string{}, Rejected: []*FilterVerdict{}, Verdicts: []*FilterVerdict{}}
	if count == 0 {
		return out, nil
	}
	asked := count * f.Config.overSample()
	o.Count = asked
	results, err := f.Positive.Generate(o)
	if err != nil {
		return nil, err
	}
	candidates := []string{}
	seen := map[string]bool{}
	for _, result := range results {
		if result.Text == "" || seen[result.Text] {
			continue
		}
		seen[result.Text] = true
		candidates = append(candidates, result.Text)
	}
	outcome, err := f.Filter(candidates)
	if err != nil {
		return nil, err
	}
	outcome.Asked = asked
	keepers := []*FilterVerdict{}
	for _, verdict := range outcome.Verdicts {
		if verdict.Decision != "reject" {
			keepers = append(keepers, verdict)
		}
	}
	sort.SliceStable(keepers, func(i, j int) bool {
		if keepers[i].Risk != keepers[j].Risk {
			return keepers[i].Risk < keepers[j].Risk
		}
		return keepers[i].Ratio < keepers[j].Ratio
	})
	texts := []string{}
	for i, verdict := range keepers {
		if i >= count {
			break
		}
		texts = append(texts, verdict.Text)
	}
	outcome.Texts = texts
	return outcome, nil
}

// FilterPrediction is a filtered continuation of a prefix.
type FilterPrediction struct {
	Prefix     string           `json:"prefix"`
	Text       *string          `json:"text"`
	Kept       []string         `json:"kept"`
	Rejected   []*FilterVerdict `json:"rejected"`
	Verdicts   []*FilterVerdict `json:"verdicts"`
	Candidates int              `json:"candidates"`
	Warning    *string          `json:"warning"`
}

// Predict continues a prefix through the pair: the positive model's top-K
// continuations, minus the vetoed ones, plus what the negative network expects
// to go wrong from here.
func (f *Filter) Predict(prefix string, o PredictOptions) (*FilterPrediction, error) {
	found, err := f.Positive.Predict(prefix, o)
	if err != nil {
		return nil, err
	}
	candidates := []string{}
	seen := map[string]bool{}
	tops := found.Top
	if len(tops) == 0 {
		best := found.PathResult
		tops = []*PathResult{&best}
	}
	for _, result := range tops {
		if result == nil || result.Text == "" || seen[result.Text] {
			continue
		}
		seen[result.Text] = true
		candidates = append(candidates, prefix+result.Text)
	}
	outcome, err := f.Filter(candidates)
	if err != nil {
		return nil, err
	}
	out := &FilterPrediction{Prefix: prefix, Kept: outcome.Kept, Rejected: outcome.Rejected,
		Verdicts: outcome.Verdicts, Candidates: len(candidates)}
	if len(outcome.Kept) > 0 {
		best := outcome.Kept[0]
		out.Text = &best
	}
	warnOpts := o
	warnOpts.K = 1
	if warning, err := f.Negative.Predict(prefix, warnOpts); err == nil && warning.Text != "" {
		text := prefix + warning.Text
		out.Warning = &text
	}
	return out, nil
}

// Describe reports what the pair is made of, for a status line.
func (f *Filter) Describe() map[string]any {
	cfg := f.Config
	threshold, minCoverage := f.Negative.Neg.Threshold, f.Negative.Neg.MinCoverage
	if cfg.Threshold != nil {
		threshold = *cfg.Threshold
	}
	if cfg.MinCoverage != nil {
		minCoverage = *cfg.MinCoverage
	}
	return map[string]any{
		"positive": f.Positive.Stats(),
		"negative": f.Negative.Stats(),
		"config": map[string]any{
			"threshold": threshold, "min_coverage": minCoverage, "ratio": cfg.Ratio, "peak": cfg.Peak,
			"over_sample": cfg.overSample(), "strict": cfg.Strict, "learn": cfg.Learn,
		},
	}
}
