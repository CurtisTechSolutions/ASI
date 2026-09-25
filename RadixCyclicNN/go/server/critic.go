package server

import (
	"fmt"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// badGateway reports an upstream LLM failure: the request was fine, the
// provider was not.
func badGateway(err error) error { return &apiError{502, err.Error()} }

// The Negative tab teaching itself, and the Ollama endpoints the loop is built
// out of: a prompt-driven corpus and the adversarial review.

// StartCritic starts a `critic` job: rounds of write -> review -> blame, so the
// negative network is taught without anyone typing a failure in by hand.
//
// The positive model writes the texts, the LLM marks them and the failures
// blame the negative network with the critique as the reason and the mark as
// the severity.  The positive model is only read from - nothing here trains,
// rewards or inverts it - so the loop can be left running beside whatever else
// is teaching it.
func (s *Service) StartCritic(config radixnet.CriticConfig, client radixnet.LLMClient) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	negative, err := s.negativeModel()
	if err != nil {
		return nil, err
	}
	return s.startJob("critic", func(job *Job, progress func(map[string]any), stop func() bool) error {
		loop, err := radixnet.NewCritic(s.model, negative, client, config)
		if err != nil {
			return err
		}
		loop.Progress = func(record map[string]any) {
			s.criticHistory = append(s.criticHistory, record)
			progress(record)
		}
		loop.Stop = stop
		// The reviewer thinks without the model lock, so readers stay served.
		loop.External = func(fn func() error) error {
			s.mu.Unlock()
			defer s.mu.Lock()
			return fn()
		}
		_, err = loop.Run()
		return err
	})
}

// CriticHistory is the round / report records of every automatic run.
func (s *Service) CriticHistory() map[string]any {
	history := s.criticHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("POST", "/api/negative/auto", rNegativeAuto)
	doc("POST", "/api/negative/auto", "the Negative tab, automatic: start a job that has the model write texts, an "+
		"LLM reviewer mark them and every failure blame the negative network - {rounds (0 = until stopped), count, "+
		"prefix, max_length, temperature, threshold, context (what the texts are meant to be), provider: "+
		"ollama|chatgpt, reviewer_model, url, timeout, clear_passes, epochs, seed}; the positive model is only read from")
	route("GET", "/api/negative/auto/history", rNegativeAutoHistory)
	doc("GET", "/api/negative/auto/history", "round / report records of all automatic runs")
	route("GET", "/api/ollama/models", rOllamaModels)
	doc("GET", "/api/ollama/models", "is Ollama reachable here and which models it has (?url=); never fails")
	route("POST", "/api/ollama/corpus", rOllamaCorpus)
	doc("POST", "/api/ollama/corpus", "a training corpus written to order: {prompt, lines, style: good|garbage, "+
		"url, model, timeout, train (start a training job on the lines), epochs, save_as}")
	route("POST", "/api/ollama/review", rOllamaReview)
	doc("POST", "/api/ollama/review", "adversarial LLM review of the model's samples or {texts}: {count, prefix, "+
		"max_length, temperature, threshold, context, url, model, timeout, seed, blame (teach the negative network "+
		"what failed and why)}")
	route("POST", "/api/ollama/think", rOllamaThink)
	doc("POST", "/api/ollama/think", "a thinking model thinks about a prompt and the network is taught its thinking "+
		"as thoughts: {prompt, lines (questions to think about), think: true | false | low | medium | high, "+
		"temperature, model, url, timeout, save_as, train (teach the thinking as thoughts that begin at the THINK "+
		"sentinel, and where a thought questions itself), with_answers (train the answers as texts too), questions, "+
		"epochs} -> {prompt, model, url, think, count, thinking, thoughts: [{question, thinking, answer}], upload, job}")
}

// criticConfigFrom reads the automatic-teaching settings of a request body.
func criticConfigFrom(rq *request) (radixnet.CriticConfig, error) {
	cfg := radixnet.DefaultCriticConfig()
	zeroI, oneI := 0, 1
	zero := 0.0
	var err error
	if cfg.Rounds, _, err = rq.f.integer("rounds", cfg.Rounds, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Count, _, err = rq.f.integer("count", cfg.Count, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Prefix, err = rq.f.optText("prefix", ""); err != nil {
		return cfg, err
	}
	if cfg.MaxLength, _, err = rq.f.integer("max_length", cfg.MaxLength, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Temperature, _, err = rq.f.number("temperature", cfg.Temperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.Threshold, _, err = rq.f.number("threshold", cfg.Threshold, &zero); err != nil {
		return cfg, err
	}
	if cfg.Context, err = rq.f.optText("context", ""); err != nil {
		return cfg, err
	}
	provider, err := rq.f.optText("provider", cfg.Provider)
	if err != nil {
		return cfg, err
	}
	if cfg.Provider, err = radixnet.NormaliseProvider(provider); err != nil {
		return cfg, badRequest("'provider' must be one of ollama, chatgpt (got %q)", provider)
	}
	if cfg.ReviewerModel, err = rq.f.optText("reviewer_model", ""); err != nil {
		return cfg, err
	}
	if cfg.ReviewerModel == "" {
		if cfg.ReviewerModel, err = rq.f.optText("model", ""); err != nil {
			return cfg, err
		}
	}
	if cfg.ClearPasses, err = rq.f.flag("clear_passes", cfg.ClearPasses); err != nil {
		return cfg, err
	}
	if cfg.Epochs, _, err = rq.f.integer("epochs", cfg.Epochs, &zeroI); err != nil {
		return cfg, err
	}
	if seed, ok, err := rq.f.integer("seed", 0, nil); err != nil {
		return cfg, err
	} else if ok {
		value := int64(seed)
		cfg.Seed = &value
	}
	if err := cfg.Validate(); err != nil {
		return cfg, badRequest("%v", err)
	}
	return cfg, nil
}

func rNegativeAuto(rq *request) (int, any, error) {
	config, err := criticConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	url, err := rq.f.optText("url", "")
	if err != nil {
		return 0, nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return 0, nil, err
	}
	client, err := rq.svc.LLMClient(config.Provider, url, config.ReviewerModel, timeout)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartCritic(config, client)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{
		"job": job, "config": config, "url": client.BaseURL(), "reviewer": client.ModelName(),
	}, nil
}

func rNegativeAutoHistory(rq *request) (int, any, error) {
	return 200, rq.svc.CriticHistory(), nil
}

// -- Ollama -----------------------------------------------------------------

func rOllamaModels(rq *request) (int, any, error) {
	url, _ := rq.queryValue("url")
	client, err := rq.svc.OllamaClient(url, "", 0)
	if err != nil {
		return 0, nil, err
	}
	models, listErr := client.Models()
	out := map[string]any{
		"available": listErr == nil, "url": client.BaseURL(), "model": client.ModelName(),
		"models": []map[string]any{}, "error": nil,
	}
	if listErr != nil {
		out["error"] = listErr.Error()
		return 200, out, nil // never fails: "is it there?" is the answer
	}
	rows := make([]map[string]any, 0, len(models))
	for _, model := range models {
		rows = append(rows, map[string]any{
			"name": model["name"], "size": model["size"], "modified_at": model["modified_at"],
			"details": model["details"],
		})
	}
	out["models"] = rows
	return 200, out, nil
}

// ollamaClient builds the reviewer client of an Ollama request (url / model /
// timeout fall back to the server's own defaults).
func ollamaClientOf(rq *request) (*radixnet.OllamaClient, error) {
	url, err := rq.f.optText("url", "")
	if err != nil {
		return nil, err
	}
	model, err := rq.f.optText("model", "")
	if err != nil {
		return nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return nil, err
	}
	return rq.svc.OllamaClient(url, model, timeout)
}

func rOllamaCorpus(rq *request) (int, any, error) {
	prompt, err := rq.f.text("prompt", nil)
	if err != nil {
		return 0, nil, err
	}
	one := 1
	lines, _, err := rq.f.integer("lines", 20, &one)
	if err != nil {
		return 0, nil, err
	}
	style, err := rq.f.optText("style", "good")
	if err != nil {
		return 0, nil, err
	}
	client, err := ollamaClientOf(rq)
	if err != nil {
		return 0, nil, err
	}
	train, err := rq.f.flag("train", false)
	if err != nil {
		return 0, nil, err
	}
	saveAs, err := rq.f.optText("save_as", "")
	if err != nil {
		return 0, nil, err
	}
	// The request's own fields are checked here, so a bad style is a 400 and only
	// a failure of the provider itself becomes a 502.
	switch strings.ToLower(strings.TrimSpace(style)) {
	case "", "good", "garbage":
	default:
		return 0, nil, badRequest("'style' must be one of %s (got %q)", strings.Join(radixnet.CorpusStyles, ", "), style)
	}
	if strings.TrimSpace(prompt) == "" {
		return 0, nil, badRequest("'prompt' must not be empty")
	}
	texts, err := radixnet.CorpusFromPrompt(client, prompt, lines, style, client.ModelName())
	if err != nil {
		return 0, nil, badGateway(err)
	}
	out := map[string]any{
		"prompt": prompt, "style": style, "lines": len(texts), "texts": texts,
		"model": client.ModelName(), "url": client.BaseURL(), "upload": nil, "job": nil,
	}
	if saveAs != "" {
		upload, err := rq.svc.uploads.StoreText(saveAs, strings.Join(texts, "\n")+"\n")
		if err != nil {
			return 0, nil, err
		}
		out["upload"] = upload
	}
	if !train {
		return 200, out, nil
	}
	if len(texts) == 0 {
		return 0, nil, badRequest("the model answered with no usable lines, so there is nothing to train on")
	}
	zeroI := 0
	epochs, _, err := rq.f.integer("epochs", 3, &zeroI)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTrain(texts, epochs, true)
	if err != nil {
		return 0, nil, err
	}
	out["job"] = job
	return 202, out, nil
}

// rOllamaThink is POST /api/ollama/think: a thinking model thinks about a
// prompt; its thinking is returned and, with train, taught as thoughts.
func rOllamaThink(rq *request) (int, any, error) {
	f := rq.f
	prompt, err := f.text("prompt", nil)
	if err != nil {
		return 0, nil, err
	}
	if strings.TrimSpace(prompt) == "" {
		return 0, nil, badRequest("'prompt' must not be empty")
	}
	lines, _, err := f.integer("lines", 5, intp(1))
	if err != nil {
		return 0, nil, err
	}
	// missing - or null, which the API reads as missing - asks the model to think; "default" leaves it to the model
	var level any = true
	if raw, present := f.lookup("think"); present && raw != nil {
		if level, err = radixnet.ThinkValue(raw); err != nil {
			return 0, nil, badRequest("%v", err)
		}
	}
	temperature, _, err := f.number("temperature", 0.7, floatp(0))
	if err != nil {
		return 0, nil, err
	}
	train, err := f.flag("train", false)
	if err != nil {
		return 0, nil, err
	}
	withAnswers, err := f.flag("with_answers", false)
	if err != nil {
		return 0, nil, err
	}
	questions, err := f.flag("questions", true)
	if err != nil {
		return 0, nil, err
	}
	saveAs, err := f.optText("save_as", "")
	if err != nil {
		return 0, nil, err
	}
	epochs, _, err := f.integer("epochs", 3, intp(0))
	if err != nil {
		return 0, nil, err
	}
	client, err := ollamaClientOf(rq)
	if err != nil {
		return 0, nil, err
	}
	if train {
		// do not spend an LLM call on a request that cannot start a job
		if err := rq.svc.ensureIdle(); err != nil {
			return 0, nil, err
		}
		if rq.svc.model.IsNegative() {
			return 0, nil, badRequest("the negative network judges; it does not think")
		}
	}
	thoughts, err := radixnet.ThoughtsFromPrompt(client, prompt, lines, client.ModelName(), level, temperature)
	if err != nil {
		return 0, nil, badGateway(err)
	}
	if len(thoughts) == 0 {
		return 0, nil, badGateway(fmt.Errorf("Ollama model %q wrote no questions to think about", client.ModelName()))
	}
	thinking, answers := []string{}, []string{}
	for _, t := range thoughts {
		if t.Thinking != "" {
			thinking = append(thinking, t.Thinking)
		}
		if t.Answer != "" {
			answers = append(answers, t.Answer)
		}
	}
	out := map[string]any{
		"prompt": prompt, "model": client.ModelName(), "url": client.BaseURL(), "think": level,
		"count": len(thoughts), "thinking": len(thinking), "thoughts": thoughts, "upload": nil, "job": nil,
	}
	if saveAs != "" {
		text := strings.Join(thinking, "\n")
		if len(thinking) > 0 {
			text += "\n"
		}
		upload, err := rq.svc.uploads.StoreText(saveAs, text)
		if err != nil {
			return 0, nil, err
		}
		out["upload"] = upload
	}
	if !train {
		return 200, out, nil
	}
	if len(thinking) == 0 {
		return 0, nil, badGateway(fmt.Errorf("Ollama model %q returned no thinking to train on: use a thinking model "+
			"(qwen3, deepseek-r1, gpt-oss, ...) on an Ollama that separates it", client.ModelName()))
	}
	if !withAnswers {
		answers = nil
	}
	job, err := rq.svc.StartTrainThoughts(thinking, answers, epochs, questions)
	if err != nil {
		return 0, nil, err
	}
	out["job"] = job
	return 202, out, nil
}

func rOllamaReview(rq *request) (int, any, error) {
	given, err := rq.f.textsOptional("texts", "text")
	if err != nil {
		return 0, nil, err
	}
	client, err := ollamaClientOf(rq)
	if err != nil {
		return 0, nil, err
	}
	zero := 0.0
	threshold, _, err := rq.f.number("threshold", 6, &zero)
	if err != nil {
		return 0, nil, err
	}
	o := radixnet.AdversarialReviewOptions{Threshold: threshold, Model: client.ModelName()}
	one := 1
	if o.Count, _, err = rq.f.integer("count", 8, &one); err != nil {
		return 0, nil, err
	}
	if o.Prefix, err = rq.f.optText("prefix", ""); err != nil {
		return 0, nil, err
	}
	zeroI := 0
	if o.MaxLength, _, err = rq.f.integer("max_length", 60, &zeroI); err != nil {
		return 0, nil, err
	}
	if o.Temperature, _, err = rq.f.number("temperature", 1, &zero); err != nil {
		return 0, nil, err
	}
	if o.Context, err = rq.f.optText("context", ""); err != nil {
		return 0, nil, err
	}
	if seed, ok, err := rq.f.integer("seed", 0, nil); err != nil {
		return 0, nil, err
	} else if ok {
		value := int64(seed)
		o.Seed = &value
	}
	if len(given) > 0 {
		o.Texts = given
	}
	blame, err := rq.f.flag("blame", false)
	if err != nil {
		return 0, nil, err
	}
	result, err := rq.svc.Review(client, o, blame)
	if err != nil {
		return 0, nil, err
	}
	return 200, result, nil
}

// Review runs one adversarial review, optionally teaching the negative network
// what failed and why.
func (s *Service) Review(client radixnet.LLMClient, o radixnet.AdversarialReviewOptions, blame bool) (map[string]any, error) {
	var negative *radixnet.Model
	if blame {
		found, err := s.negativeModel()
		if err != nil {
			return nil, err
		}
		negative = found
	}
	// The model writes under the lock (sampling walks the graph), the reviewer
	// thinks without it (a network call that touches nothing of ours), and the
	// blaming takes the lock again.
	samples, source := o.Texts, "given"
	if samples == nil {
		s.mu.Lock()
		drawn, err := radixnet.SampleTexts(s.model, o.Count, o.Prefix, o.MaxLength, o.Temperature, o.Seed)
		s.mu.Unlock()
		if err != nil {
			return nil, badRequest("%v", err)
		}
		samples, source = drawn, "model"
	}
	reviewer := o.Model
	if reviewer == "" {
		reviewer = client.ModelName()
	}
	reviews, err := radixnet.ReviewTexts(client, samples, o.Context, o.Model, o.Threshold, radixnet.DefaultReviewBatch)
	if err != nil {
		return nil, badGateway(err)
	}
	result := radixnet.SummariseReviews(source, reviewer, o.Threshold, samples, reviews)
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]any{
		"source": result.Source, "model": result.Model, "threshold": result.Threshold, "texts": result.Texts,
		"reviews": result.Reviews, "mean_rating": result.MeanRating, "pass_rate": result.PassRate,
		"good": result.Good, "bad": result.Bad, "negative": nil,
	}
	if negative != nil {
		report, err := radixnet.TeachReviews(negative, result.Reviews, result.Threshold, true, "review",
			radixnet.TeachOptions{})
		if err != nil {
			return nil, err
		}
		out["negative"] = map[string]any{
			"taught": report, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	return out, nil
}
