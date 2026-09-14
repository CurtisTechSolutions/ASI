package server

// The negative network over HTTP: GET /api/negative describes it, the
// /api/negative/* endpoints teach it, judge with it and run the pair, and
// /api/tutor/start can hand it every sentence the teacher marked down.  The Go
// twin of the Python server's negative endpoints, answering the same JSON.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// negativePath is the negative network's file: model.negative.json beside the
// server's model path (empty when the server runs without one).
func (s *Service) negativePath() string {
	if s.modelPath == "" {
		return ""
	}
	ext := filepath.Ext(s.modelPath)
	root := strings.TrimSuffix(s.modelPath, ext)
	if ext == ".gz" {
		inner := filepath.Ext(root)
		root = strings.TrimSuffix(root, inner)
		ext = inner + ext
	}
	if strings.HasSuffix(root, ".negative") {
		return s.modelPath
	}
	abs, err := filepath.Abs(root + ".negative" + ext)
	if err != nil {
		return root + ".negative" + ext
	}
	return abs
}

// negativeModel is the one negative network of this service, loaded from its
// file or created empty on first use.
func (s *Service) negativeModel() (*radixnet.Model, error) {
	if s.negative != nil {
		return s.negative, nil
	}
	path := s.negativePath()
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			m, err := radixnet.Load(path)
			if err != nil {
				return nil, badRequest("%s: %v", path, err)
			}
			if !m.IsNegative() {
				return nil, badRequest("%s holds a %s model, not a negative one", path, m.Kind())
			}
			m.Workers, m.G.Workers, m.Exact = s.workers, s.workers, s.exact
			s.negative = m
			s.logf(fmt.Sprintf("negative model loaded from %s", path))
			return m, nil
		}
	}
	m, err := radixnet.NewNegativeModel(s.seed, radixnet.DefaultNegativeOptions())
	if err != nil {
		return nil, err
	}
	m.Workers, m.G.Workers, m.Exact = s.workers, s.workers, s.exact
	s.negative = m
	return m, nil
}

// negativeRead runs fn under the read lock with the negative network.
func (s *Service) negativeRead(fn func(m *radixnet.Model) (any, error)) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	m, err := s.negativeModel()
	if err != nil {
		return nil, err
	}
	return fn(m)
}

// negativeMutate runs fn under the write lock, refusing while a job runs.
func (s *Service) negativeMutate(fn func(m *radixnet.Model) (any, error)) (any, error) {
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	m, err := s.negativeModel()
	if err != nil {
		return nil, err
	}
	return fn(m)
}

// NegativeStatus is the stats, reason table and journal of the negative network.
func (s *Service) NegativeStatus() (map[string]any, error) {
	out, err := s.negativeRead(func(m *radixnet.Model) (any, error) {
		return map[string]any{
			"path":     s.negativePath(),
			"active":   false, // the Go server always runs the count model; the negative network is its companion
			"stats":    m.Stats(),
			"reasons":  m.Reasons(),
			"journal":  m.Recent(20),
			"weights":  m.G.NegativeWeightConfig(),
			"settings": map[string]any{"threshold": m.Neg.Threshold, "min_coverage": m.Neg.MinCoverage},
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

func negativeResult(m *radixnet.Model, records []map[string]any, extra map[string]any) map[string]any {
	if records == nil {
		records = []map[string]any{}
	}
	out := map[string]any{"records": records, "reasons": m.Reasons(), "stats": m.Stats()}
	for key, value := range extra {
		out[key] = value
	}
	return out
}

// NegativeBlame teaches the negative network a failure (synchronous: rated sets are small).
func (s *Service) NegativeBlame(texts []string, o radixnet.BlameOptions) (map[string]any, error) {
	if len(texts) == 0 {
		return nil, badRequest("give the failed texts as 'texts' (a list) or 'text' (one per line)")
	}
	out, err := s.negativeMutate(func(m *radixnet.Model) (any, error) {
		records, err := m.Blame(texts, o)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return negativeResult(m, records, map[string]any{
			"texts": len(texts), "reason": radixnet.CleanReason(o.Reason), "severity": o.Severity,
		}), nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeClear takes blame off the fragments the passed texts share with known failures.
func (s *Service) NegativeClear(texts []string, weight float64, epochs int) (map[string]any, error) {
	if len(texts) == 0 {
		return nil, badRequest("give the passed texts as 'texts' (a list) or 'text' (one per line)")
	}
	out, err := s.negativeMutate(func(m *radixnet.Model) (any, error) {
		records, err := m.Clear(texts, weight, epochs)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		matched, unmatched := 0, 0
		if len(records) > 0 {
			last := records[len(records)-1]
			matched, unmatched = int(toFloat(last["matched"])), int(toFloat(last["unmatched"]))
		}
		return negativeResult(m, records, map[string]any{
			"texts": len(texts), "matched": matched, "unmatched": unmatched,
		}), nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeJudge is why these texts look like failures.
func (s *Service) NegativeJudge(texts []string, o radixnet.JudgeOptions) (map[string]any, error) {
	if len(texts) == 0 {
		return nil, badRequest("give the texts to judge as 'texts' (a list) or 'text' (one per line)")
	}
	out, err := s.negativeRead(func(m *radixnet.Model) (any, error) {
		verdicts := make([]*radixnet.Verdict, 0, len(texts))
		for _, text := range texts {
			verdicts = append(verdicts, m.Judge(text, o))
		}
		return map[string]any{"verdicts": verdicts, "stats": m.Stats()}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeFilter runs the pair: the positive model writes, the negative one
// vetoes (or the given texts are judged).
func (s *Service) NegativeFilter(texts []string, count int, config radixnet.FilterConfig, o radixnet.GenerateOptions) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	run := func(m *radixnet.Model) (any, error) {
		pair, err := radixnet.NewFilter(s.model, m, config)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		var outcome *radixnet.FilterOutcome
		if len(texts) > 0 {
			outcome, err = pair.Filter(texts)
		} else {
			outcome, err = pair.Generate(count, o)
		}
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return map[string]any{
			"texts": outcome.Texts, "kept": outcome.Kept, "rejected": outcome.Rejected,
			"verdicts": outcome.Verdicts, "candidates": outcome.Candidates, "asked": outcome.Asked,
			"rate": outcome.Rate, "pair": pair.Describe(),
		}, nil
	}
	var out any
	var err error
	if config.Learn {
		out, err = s.negativeMutate(run)
	} else {
		out, err = s.negativeRead(run)
	}
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeForget drops (or fades) the blame behind one reason.
func (s *Service) NegativeForget(reason string, factor float64) (map[string]any, error) {
	out, err := s.negativeMutate(func(m *radixnet.Model) (any, error) {
		result, err := m.Forget(reason, factor)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return negativeResult(m, nil, map[string]any{
			"reason": result.Reason, "edges": result.Edges, "blame_removed": result.BlameRemoved,
		}), nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeSettings changes how strictly the negative network judges and the scales of its weight function.
func (s *Service) NegativeSettings(threshold, minCoverage *float64, scales map[string]float64) (map[string]any, error) {
	out, err := s.negativeMutate(func(m *radixnet.Model) (any, error) {
		if threshold != nil {
			m.Neg.Threshold = *threshold
		}
		if minCoverage != nil {
			m.Neg.MinCoverage = *minCoverage
		}
		if len(scales) > 0 {
			if err := m.G.ConfigureNegative(scales); err != nil {
				return nil, badRequest("%v", err)
			}
		}
		return map[string]any{
			"settings": map[string]any{"threshold": m.Neg.Threshold, "min_coverage": m.Neg.MinCoverage},
			"weights":  m.G.NegativeWeightConfig(),
			"stats":    m.Stats(),
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NegativeReset forgets every failure: a fresh, empty negative network.
func (s *Service) NegativeReset(seed *int64) (map[string]any, error) {
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	use := s.seed
	if seed != nil {
		use = *seed
	}
	m, err := radixnet.NewNegativeModel(use, radixnet.DefaultNegativeOptions())
	if err != nil {
		return nil, err
	}
	m.Workers, m.G.Workers, m.Exact = s.workers, s.workers, s.exact
	s.negative = m
	return map[string]any{"stats": m.Stats(), "reasons": m.Reasons(), "journal": m.Recent(0)}, nil
}

// NegativeSave writes the negative network (default: beside the model path).
func (s *Service) NegativeSave(path string) (map[string]any, error) {
	target := path
	if target == "" {
		target = s.negativePath()
	}
	if target == "" {
		return nil, badRequest("no 'path' given and the server was started without a model path")
	}
	abs, err := filepath.Abs(target)
	if err != nil {
		return nil, err
	}
	if _, err := s.negativeRead(func(m *radixnet.Model) (any, error) { return nil, m.Save(abs) }); err != nil {
		return nil, err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return nil, err
	}
	return map[string]any{"path": abs, "bytes": st.Size()}, nil
}

// toFloat reads a number out of a record whatever numeric type it holds.
func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	}
	return 0
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("GET", "/api/negative", rNegative)
	doc("GET", "/api/negative", "the negative network: stats, the reason table (what the tutor blamed), the journal of what it said and the filter settings")
	route("POST", "/api/negative/blame", rNegativeBlame)
	doc("POST", "/api/negative/blame", "teach it a failure: {texts | text, reason, severity, source, note, epochs} - the only call that adds structure to the negative network")
	route("POST", "/api/negative/clear", rNegativeClear)
	doc("POST", "/api/negative/clear", "the tutor passed these texts: {texts | text, weight, epochs} takes blame off the fragments they share with known failures (nothing is created)")
	route("POST", "/api/negative/judge", rNegativeJudge)
	doc("POST", "/api/negative/judge", "why texts look like failures: {texts | text, threshold, min_coverage, spans} -> verdicts with risk, peak, coverage, the reasons and the fragments to blame")
	route("POST", "/api/negative/filter", rNegativeFilter)
	doc("POST", "/api/negative/filter", "the pair: the positive model writes, the negative one vetoes - {count, prefix, mode, max_length, temperature, over_sample, threshold, min_coverage, ratio, no_ratio, peak, strict, learn} or {texts} to judge given texts")
	route("POST", "/api/negative/forget", rNegativeForget)
	doc("POST", "/api/negative/forget", "drop or fade the blame behind a reason: {reason, factor}")
	route("POST", "/api/negative/settings", rNegativeSettings)
	doc("POST", "/api/negative/settings", "how strictly it judges: {threshold, min_coverage, share_scale, blame_scale, clear_scale}")
	route("POST", "/api/negative/reset", rNegativeReset)
	doc("POST", "/api/negative/reset", "forget every failure: {seed} -> a fresh negative network")
	route("POST", "/api/negative/save", rNegativeSave)
	doc("POST", "/api/negative/save", "save the negative network: {path} (default: beside the model path)")
}

func rNegative(rq *request) (int, any, error) {
	out, err := rq.svc.NegativeStatus()
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

// negativeTexts reads {texts | text} out of a request body.
func negativeTexts(rq *request, what string) ([]string, error) {
	texts, err := rq.f.textsOptional("texts", "text")
	if err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, badRequest("give the %s as 'texts' (a list) or 'text' (one per line)", what)
	}
	return texts, nil
}

func rNegativeBlame(rq *request) (int, any, error) {
	texts, err := negativeTexts(rq, "failed texts")
	if err != nil {
		return 0, nil, err
	}
	reason, err := rq.f.optText("reason", radixnet.UnspecifiedReason)
	if err != nil {
		return 0, nil, err
	}
	zero := 0.0
	severity, _, err := rq.f.number("severity", 1, &zero)
	if err != nil {
		return 0, nil, err
	}
	source, err := rq.f.optText("source", "api")
	if err != nil {
		return 0, nil, err
	}
	note, err := rq.f.optText("note", "")
	if err != nil {
		return 0, nil, err
	}
	one := 1
	epochs, _, err := rq.f.integer("epochs", 1, &one)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.NegativeBlame(texts, radixnet.BlameOptions{
		Reason: reason, Severity: severity, Source: source, Note: note, Epochs: epochs,
	})
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeClear(rq *request) (int, any, error) {
	texts, err := negativeTexts(rq, "passed texts")
	if err != nil {
		return 0, nil, err
	}
	zero := 0.0
	weight, _, err := rq.f.number("weight", 1, &zero)
	if err != nil {
		return 0, nil, err
	}
	one := 1
	epochs, _, err := rq.f.integer("epochs", 1, &one)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.NegativeClear(texts, weight, epochs)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

// optionalNumber reads a field that may be absent (nil = unset).
func optionalNumber(rq *request, name string) (*float64, error) {
	if !rq.f.present(name) {
		return nil, nil
	}
	zero := 0.0
	value, _, err := rq.f.number(name, 0, &zero)
	if err != nil {
		return nil, err
	}
	return &value, nil
}

func rNegativeJudge(rq *request) (int, any, error) {
	texts, err := negativeTexts(rq, "texts to judge")
	if err != nil {
		return 0, nil, err
	}
	threshold, err := optionalNumber(rq, "threshold")
	if err != nil {
		return 0, nil, err
	}
	minCoverage, err := optionalNumber(rq, "min_coverage")
	if err != nil {
		return 0, nil, err
	}
	zero := 0
	spans, _, err := rq.f.integer("spans", 5, &zero)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.NegativeJudge(texts, radixnet.JudgeOptions{Threshold: threshold, MinCoverage: minCoverage, Spans: spans})
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeFilter(rq *request) (int, any, error) {
	texts, err := rq.f.textsOptional("texts", "text")
	if err != nil {
		return 0, nil, err
	}
	zero, one := 0, 1
	zeroF := 0.0
	count, _, err := rq.f.integer("count", 3, &zero)
	if err != nil {
		return 0, nil, err
	}
	config := radixnet.DefaultFilterConfig()
	if config.Threshold, err = optionalNumber(rq, "threshold"); err != nil {
		return 0, nil, err
	}
	if config.MinCoverage, err = optionalNumber(rq, "min_coverage"); err != nil {
		return 0, nil, err
	}
	if config.Peak, err = optionalNumber(rq, "peak"); err != nil {
		return 0, nil, err
	}
	ratio, _, err := rq.f.number("ratio", 0, nil)
	if err != nil {
		return 0, nil, err
	}
	noRatio, err := rq.f.flag("no_ratio", false)
	if err != nil {
		return 0, nil, err
	}
	if noRatio {
		config.Ratio = nil
	} else {
		config.Ratio = &ratio
	}
	if config.OverSample, _, err = rq.f.integer("over_sample", 3, &one); err != nil {
		return 0, nil, err
	}
	if config.Spans, _, err = rq.f.integer("spans", 3, &zero); err != nil {
		return 0, nil, err
	}
	if config.Strict, err = rq.f.flag("strict", false); err != nil {
		return 0, nil, err
	}
	if config.Learn, err = rq.f.flag("learn", false); err != nil {
		return 0, nil, err
	}
	if config.Reason, err = rq.f.optText("reason", "filtered"); err != nil {
		return 0, nil, err
	}
	o := radixnet.DefaultGenerateOptions()
	if o.MaxLength, _, err = rq.f.integer("max_length", 60, &zero); err != nil {
		return 0, nil, err
	}
	if o.Mode, err = rq.f.optText("mode", "sample"); err != nil {
		return 0, nil, err
	}
	if o.Temperature, _, err = rq.f.number("temperature", 1, &zeroF); err != nil {
		return 0, nil, err
	}
	if o.Prefix, err = rq.f.optText("prefix", ""); err != nil {
		return 0, nil, err
	}
	if o.StepPenalty, _, err = rq.f.number("step_penalty", 0, &zeroF); err != nil {
		return 0, nil, err
	}
	if seed, ok, err := rq.f.integer("seed", 0, nil); err != nil {
		return 0, nil, err
	} else if ok {
		value := int64(seed)
		o.Seed = &value
	}
	out, err := rq.svc.NegativeFilter(texts, count, config, o)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeForget(rq *request) (int, any, error) {
	reason, err := rq.f.optText("reason", "")
	if err != nil {
		return 0, nil, err
	}
	zero := 0.0
	factor, _, err := rq.f.number("factor", 0, &zero)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.NegativeForget(reason, factor)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeSettings(rq *request) (int, any, error) {
	threshold, err := optionalNumber(rq, "threshold")
	if err != nil {
		return 0, nil, err
	}
	minCoverage, err := optionalNumber(rq, "min_coverage")
	if err != nil {
		return 0, nil, err
	}
	scales := map[string]float64{}
	for _, name := range []string{"share_scale", "blame_scale", "clear_scale"} {
		if !rq.f.present(name) {
			continue
		}
		value, _, err := rq.f.number(name, 0, nil)
		if err != nil {
			return 0, nil, err
		}
		scales[name] = value
	}
	out, err := rq.svc.NegativeSettings(threshold, minCoverage, scales)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeReset(rq *request) (int, any, error) {
	var seed *int64
	if value, ok, err := rq.f.integer("seed", 0, nil); err != nil {
		return 0, nil, err
	} else if ok {
		use := int64(value)
		seed = &use
	}
	out, err := rq.svc.NegativeReset(seed)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rNegativeSave(rq *request) (int, any, error) {
	path, err := rq.f.optText("path", "")
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.NegativeSave(path)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}
