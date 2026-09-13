package server

// The English tutor over HTTP: GET /api/tutor describes it, POST
// /api/tutor/start runs rounds of prefix -> completion -> grade -> 2NRL as a
// job, GET /api/tutor/history returns their records, POST /api/tutor/lesson
// marks one round without training anything and POST /api/tutor/plan turns a
// report card into the syllabus of the lessons that follow.  The Go twin of
// the Python server's /api/tutor endpoints, answering the same JSON.

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// OllamaClient builds a client for the request's overrides, falling back to the server defaults.
func (s *Service) OllamaClient(url, model string, timeout float64) (*radixnet.OllamaClient, error) {
	if strings.TrimSpace(url) == "" {
		url = s.ollamaURL
	}
	if strings.TrimSpace(model) == "" {
		model = s.ollamaModel
	}
	client, err := radixnet.NewOllamaClient(url, model, time.Duration(timeout*float64(time.Second)))
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return client, nil
}

// ChatGPTClient builds a client for the request's overrides; the API key is the
// server's own (OPENAI_API_KEY) and is never a request field.
func (s *Service) ChatGPTClient(url, model string, timeout float64) (*radixnet.ChatGPTClient, error) {
	if strings.TrimSpace(url) == "" {
		url = s.chatgptURL
	}
	if strings.TrimSpace(model) == "" {
		model = s.chatgptModel
	}
	client, err := radixnet.NewChatGPTClient(url, model, time.Duration(timeout*float64(time.Second)))
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return client, nil
}

// LLMClient builds a client for one provider; a chatgpt teacher without a key on
// this server is refused before anything is sent.
func (s *Service) LLMClient(provider, url, model string, timeout float64) (radixnet.LLMClient, error) {
	name, err := radixnet.NormaliseProvider(provider)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	if name == radixnet.ProviderOllama {
		return s.OllamaClient(url, model, timeout)
	}
	if !radixnet.ChatGPTConfigured() {
		return nil, badRequest("ChatGPT is not configured on this server: set OPENAI_API_KEY (or " +
			"OPENAI_API_KEY_FILE) in its environment and restart it, or use the 'ollama' provider")
	}
	return s.ChatGPTClient(url, model, timeout)
}

// DefaultTutorModel is the teacher this server uses when a request names none.
func (s *Service) DefaultTutorModel() string { return s.ollamaModel }

// DefaultTutorModelFor is the model this server teaches with on one provider.
func (s *Service) DefaultTutorModelFor(provider string) string {
	if provider == radixnet.ProviderChatGPT {
		return s.chatgptModel
	}
	return s.ollamaModel
}

// StartTutor starts a tutor job: rounds of exercise, completion, grade and 2NRL.
func (s *Service) StartTutor(config radixnet.TutorConfig, client, grader radixnet.LLMClient) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	return s.startJob("tutor", func(job *Job, progress func(map[string]any), stop func() bool) error {
		trainer, err := radixnet.NewTutorTrainer(s.model, client, config)
		if err != nil {
			return err
		}
		if grader != nil {
			trainer.GraderClient = grader
		}
		trainer.Progress = func(record map[string]any) {
			s.tutorHistory = append(s.tutorHistory, record)
			progress(record)
		}
		trainer.Stop = stop
		// The teacher thinks without the model lock, so readers stay served.
		trainer.External = func(fn func() error) error {
			s.mu.Unlock()
			defer s.mu.Lock()
			return fn()
		}
		_, err = trainer.Run()
		return err
	})
}

// TutorHistory is the lesson / round / report records of every tutor run.
func (s *Service) TutorHistory() map[string]any {
	history := s.tutorHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// TutorCard is the report card at the end of the last tutor run, or nil when
// nothing has been marked yet.  A running job holds the write lock between
// records, so the scan waits for it like any other reader.
func (s *Service) TutorCard() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := len(s.tutorHistory) - 1; i >= 0; i-- {
		if kind, _ := s.tutorHistory[i]["kind"].(string); kind == "report" {
			return s.tutorHistory[i]
		}
	}
	return nil
}

// TutorPlan asks the teacher for the lessons to run next, given a report card:
// one point of grammar each, aimed at the mistakes the marking found.
func (s *Service) TutorPlan(
	config radixnet.TutorConfig, client radixnet.LLMClient, card map[string]any, count int,
) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	plan, err := radixnet.PlanLessons(client, card, radixnet.PlanRequest{
		Topic: config.Topic, Level: config.Level, Words: config.Words, Threshold: config.Threshold, Count: count,
		Exercises: config.Exercises, Drills: config.Drills, Model: config.TutorModel,
	})
	if err != nil {
		if radixnet.IsLLMError(err) {
			return nil, &apiError{502, err.Error()}
		}
		return nil, badRequest("%v", err)
	}
	return map[string]any{
		"plan": plan.Map(), "source": plan.Source, "provider": radixnet.ProviderOf(client),
		"model": client.ModelName(), "url": client.BaseURL(), "report": card,
	}, nil
}

// TutorLesson runs one round of lessons without training: the exercises (or
// the given prefixes), the completions and their grades.
func (s *Service) TutorLesson(
	config radixnet.TutorConfig, client, grader radixnet.LLMClient, prefixes []string,
) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	trainer, err := radixnet.NewTutorTrainer(s.model, client, config)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	if grader != nil {
		trainer.GraderClient = grader
	}
	source := radixnet.ProviderOf(client)
	var exercises []radixnet.Exercise
	if len(prefixes) > 0 {
		source = "given"
		if len(prefixes) > config.Exercises {
			prefixes = prefixes[:config.Exercises]
		}
		for i, prefix := range prefixes {
			exercises = append(exercises, radixnet.Exercise{
				ID: fmt.Sprintf("e%d", i+1), Prefix: strings.TrimSpace(prefix), Focus: config.Focus,
			})
		}
	} else if exercises, err = trainer.SetExercises(1); err != nil {
		return nil, err
	}
	lessons := []*radixnet.Lesson{}
	completed, err := s.read(func(m *radixnet.Model) (any, error) { // the search runs under the read lock
		out := []*radixnet.Lesson{}
		for _, exercise := range exercises {
			for attempt := 0; attempt < config.Attempts; attempt++ {
				lesson, err := trainer.Complete(exercise, attempt)
				if err != nil {
					return nil, err
				}
				out = append(out, lesson)
			}
		}
		return out, nil
	})
	if err != nil {
		return nil, err
	}
	lessons = completed.([]*radixnet.Lesson)
	if err := trainer.GradeLessons(lessons); err != nil { // the LLM call is outside the lock
		return nil, err
	}
	return map[string]any{
		"source": source, "model": client.ModelName(), "url": client.BaseURL(), "config": config,
		"exercises": exercises, "lessons": lessons, "report": radixnet.ReportCard(lessons),
	}, nil
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("GET", "/api/tutor", rTutor)
	doc("GET", "/api/tutor", "the English tutor: the teachers on offer (ollama, chatgpt: url, model, configured), the error types a completion is marked with, the completion modes and every default setting")
	route("POST", "/api/tutor/start", rTutorStart)
	doc("POST", "/api/tutor/start", "start a tutor job - the teacher writes the prefixes, the network completes them, the teacher marks the grammar and the 2NRL follows: {topic, rounds, exercises, attempts, focus, level, mode, threshold, grammar_weight, drills, adapt, twonrl_per: round|lesson, diff_corrections, keep_weight, min_weight, neg_epochs, pos_epochs, strength, tutor_provider: ollama|chatgpt, tutor_model, grader_provider, grader_model, url, grader_url, ...}")
	route("GET", "/api/tutor/history", rTutorHistory)
	doc("GET", "/api/tutor/history", "lesson / round / report records of all tutor runs")
	route("GET", "/api/chatgpt/models", rChatGPTModels)
	doc("GET", "/api/chatgpt/models", "is ChatGPT usable as a teacher here (server-side OPENAI_API_KEY) and which models the key has (?url=); never fails")
	route("POST", "/api/tutor/lesson", rTutorLesson)
	doc("POST", "/api/tutor/lesson", "one round of lessons without training: {topic, exercises, prefixes (skip the LLM and use these), attempts, threshold, ...} -> completions with grades (grammar, spelling, fluency, error, correction) and a report card")
	route("POST", "/api/tutor/plan", rTutorPlan)
	doc("POST", "/api/tutor/plan", "the lesson plan a report card implies: {report (default: the card at the end of the last run), count, topic, level, words, threshold, exercises, drills, tutor_model, url} -> {plan: {summary, prompt (the brief for the next batch: start a run with it as 'brief'), upgrade: {step, level, words, threshold, drills, note}, level, weak, targets, lessons: [{focus, targets, topic, why, exercises, drills, prefixes}]}, source}")
}

// rChatGPTModels reports whether ChatGPT can teach on this server and which
// models its key may use; like the Python endpoint it answers 200 either way.
func rChatGPTModels(rq *request) (int, any, error) {
	url, _ := rq.queryValue("url")
	client, err := rq.svc.ChatGPTClient(url, "", 0)
	if err != nil {
		return 0, nil, err
	}
	configured := radixnet.ChatGPTConfigured()
	message := ""
	models := []map[string]any{}
	if !configured {
		message = "no API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) for the server process"
	} else if found, err := client.Models(); err != nil {
		message = err.Error()
	} else {
		for _, model := range found {
			models = append(models, map[string]any{
				"name": model["name"], "owned_by": model["owned_by"], "created": model["created"],
			})
		}
	}
	var failure any
	if message != "" {
		failure = message
	}
	return 200, map[string]any{
		"available": message == "", "configured": configured, "url": client.BaseURL(), "model": client.ModelName(),
		"models": models, "error": failure,
	}, nil
}

func rTutor(rq *request) (int, any, error) {
	return 200, map[string]any{
		"url":       rq.svc.ollamaURL,
		"model":     rq.svc.DefaultTutorModel(),
		"env_model": radixnet.DefaultTutorModel(),
		"providers": map[string]any{
			radixnet.ProviderOllama: map[string]any{
				"url": rq.svc.ollamaURL, "model": rq.svc.DefaultTutorModelFor(radixnet.ProviderOllama),
				"configured": true,
			},
			radixnet.ProviderChatGPT: map[string]any{
				"url": rq.svc.chatgptURL, "model": rq.svc.DefaultTutorModelFor(radixnet.ProviderChatGPT),
				"configured": radixnet.ChatGPTConfigured(),
			},
		},
		"error_types":   radixnet.ErrorTypes,
		"modes":         radixnet.TutorModes,
		"twonrl_per":    radixnet.TwoNRLPer,
		"levels":        radixnet.Levels,
		"words_ladder":  radixnet.WordsLadder,
		"upgrade_steps": radixnet.UpgradeSteps,
		"plan_lessons":  radixnet.DefaultPlanLessons,
		"defaults":      radixnet.DefaultTutorConfig(),
	}, nil
}

// tutorConfigFrom reads the tutoring settings of a request body.
func tutorConfigFrom(rq *request) (radixnet.TutorConfig, error) {
	f := rq.f
	d := radixnet.DefaultTutorConfig()
	c := d
	c.TutorModel = rq.svc.DefaultTutorModel()
	var err error
	text := func(name, def string, into *string) {
		if err != nil {
			return
		}
		var value string
		if value, err = f.optText(name, def); err == nil {
			*into = value
		}
	}
	integer := func(name string, def, minimum int, into *int) {
		if err != nil {
			return
		}
		var value int
		if value, _, err = f.integer(name, def, intp(minimum)); err == nil {
			*into = value
		}
	}
	number := func(name string, def, minimum float64, into *float64) {
		if err != nil {
			return
		}
		var value float64
		if value, _, err = f.number(name, def, floatp(minimum)); err == nil {
			*into = value
		}
	}
	flag := func(name string, def bool, into *bool) {
		if err != nil {
			return
		}
		var value bool
		if value, err = f.flag(name, def); err == nil {
			*into = value
		}
	}
	text("topic", d.Topic, &c.Topic)
	integer("rounds", d.Rounds, 1, &c.Rounds)
	integer("exercises", d.Exercises, 1, &c.Exercises)
	integer("attempts", d.Attempts, 1, &c.Attempts)
	text("focus", "", &c.Focus)
	text("level", d.Level, &c.Level)
	text("words", d.Words, &c.Words)
	text("brief", d.Brief, &c.Brief)
	provider := func(name, alias, def string, into *string) {
		if err != nil {
			return
		}
		var value string
		if value, err = f.optText(name, ""); err != nil {
			return
		}
		if strings.TrimSpace(value) == "" && alias != "" {
			if value, err = f.optText(alias, ""); err != nil {
				return
			}
		}
		if strings.TrimSpace(value) == "" {
			*into = def
			return
		}
		var normalised string
		if normalised, err = radixnet.NormaliseProvider(value); err != nil {
			err = badRequest("'%s' must be one of %s (got %q)", name, strings.Join(radixnet.Providers(), ", "), value)
			return
		}
		*into = normalised
	}
	provider("tutor_provider", "provider", d.TutorProvider, &c.TutorProvider)
	provider("grader_provider", "", c.TutorProvider, &c.GraderProvider)
	if err == nil {
		c.TutorModel = rq.svc.DefaultTutorModelFor(c.TutorProvider)
	}
	text("tutor_model", c.TutorModel, &c.TutorModel)
	text("model", c.TutorModel, &c.TutorModel)
	text("grader_model", "", &c.GraderModel)
	if err == nil && strings.TrimSpace(c.GraderModel) == "" && c.GraderProvider != c.TutorProvider {
		c.GraderModel = rq.svc.DefaultTutorModelFor(c.GraderProvider)
	}
	text("mode", d.Mode, &c.Mode)
	integer("length", d.Length, 0, &c.Length)
	integer("max_length", d.MaxLength, 1, &c.MaxLength)
	number("temperature", d.Temperature, 0, &c.Temperature)
	flag("to_end", d.ToEnd, &c.ToEnd)
	integer("beam", 0, 0, &c.Beam)
	integer("k", d.K, 0, &c.K)
	number("threshold", d.Threshold, 0, &c.Threshold)
	number("grammar_weight", d.GrammarWeight, 0, &c.GrammarWeight)
	integer("batch", d.Batch, 1, &c.Batch)
	flag("adapt", d.Adapt, &c.Adapt)
	integer("drills", d.Drills, 0, &c.Drills)
	integer("plan", d.Plan, 0, &c.Plan)
	flag("teach_answer", d.TeachAnswer, &c.TeachAnswer)
	flag("learn", d.Learn, &c.Learn)
	text("twonrl_per", d.TwoNRLPer, &c.TwoNRLPer)
	flag("diff_corrections", d.DiffCorrections, &c.DiffCorrections)
	number("keep_weight", d.KeepWeight, 0, &c.KeepWeight)
	number("min_weight", d.MinWeight, 0, &c.MinWeight)
	integer("neg_epochs", d.NegEpochs, 0, &c.NegEpochs)
	integer("pos_epochs", d.PosEpochs, 0, &c.PosEpochs)
	number("strength", d.Strength, 0, &c.Strength)
	flag("replay", d.Replay, &c.Replay)
	integer("replay_limit", d.ReplayLimit, 0, &c.ReplayLimit)
	if err != nil {
		return c, err
	}
	// the learning rates of the sine network are accepted and ignored, as elsewhere on this server
	for _, name := range []string{"neg_lr", "pos_lr"} {
		if _, _, err := rq.f.number(name, 0, floatp(0)); err != nil {
			return c, err
		}
	}
	c.Mode = strings.ToLower(strings.TrimSpace(c.Mode))
	c.TwoNRLPer = strings.ToLower(strings.TrimSpace(c.TwoNRLPer))
	if err := c.Validate(); err != nil {
		return c, badRequest("%v", err)
	}
	return c, nil
}

// tutorClients builds the teacher and marker of a tutor request (url /
// grader_url / timeout overrides); they are the same client unless the
// providers or the URLs differ.
func tutorClients(rq *request, config radixnet.TutorConfig) (radixnet.LLMClient, radixnet.LLMClient, error) {
	url, err := rq.f.optText("url", "")
	if err != nil {
		return nil, nil, err
	}
	graderURL, err := rq.f.optText("grader_url", "")
	if err != nil {
		return nil, nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return nil, nil, err
	}
	client, err := rq.svc.LLMClient(config.TutorProvider, url, config.TutorModel, timeout)
	if err != nil {
		return nil, nil, err
	}
	if config.GraderProvider == config.TutorProvider && strings.TrimSpace(graderURL) == "" {
		return client, client, nil
	}
	grader, err := rq.svc.LLMClient(config.GraderProvider, graderURL, config.ResolvedGraderModel(), timeout)
	if err != nil {
		return nil, nil, err
	}
	return client, grader, nil
}

func rTutorStart(rq *request) (int, any, error) {
	config, err := tutorConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, grader, err := tutorClients(rq, config)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTutor(config, client, grader)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{"job": job, "config": config, "url": client.BaseURL()}, nil
}

func rTutorHistory(rq *request) (int, any, error) {
	return 200, rq.svc.TutorHistory(), nil
}

func rTutorLesson(rq *request) (int, any, error) {
	config, err := tutorConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, grader, err := tutorClients(rq, config)
	if err != nil {
		return 0, nil, err
	}
	prefixes, err := rq.f.textsOptional("prefixes", "prefix")
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.TutorLesson(config, client, grader, prefixes)
	if err != nil {
		if radixnet.IsLLMError(err) {
			return 0, nil, &apiError{502, err.Error()}
		}
		return 0, nil, err
	}
	return 200, out, nil
}

func rTutorPlan(rq *request) (int, any, error) {
	config, err := tutorConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, _, err := tutorClients(rq, config)
	if err != nil {
		return 0, nil, err
	}
	card, err := rq.f.mapping("report")
	if err != nil {
		return 0, nil, err
	}
	if card == nil {
		if card = rq.svc.TutorCard(); card == nil {
			return 0, nil, badRequest("no report card yet: run some lessons first, or send one as 'report'")
		}
	}
	fallback := config.Plan
	if fallback < 1 {
		fallback = radixnet.DefaultPlanLessons
	}
	count, _, err := rq.f.integer("count", fallback, intp(1))
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.TutorPlan(config, client, card, count)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

// -- ratings on the rated-text endpoints -----------------------------------

// ratingsOf reads the per-text weights of one side: <side>_weights (0..1
// shares) or <side>_ratings (marks out of 10).  A rating says how good or bad
// each text is, so the network learns a 9-out-of-10 text nine tenths as hard
// as a perfect one instead of treating every thumbs up alike.
func ratingsOf(f fields, side string, texts []string) ([]float64, error) {
	weights, err := weightList(f, side+"_weights", len(texts), 1.0)
	if err != nil {
		return nil, err
	}
	ratings, err := weightList(f, side+"_ratings", len(texts), 10.0)
	if err != nil {
		return nil, err
	}
	if weights != nil && ratings != nil {
		return nil, badRequest("give '%s_weights' or '%s_ratings', not both", side, side)
	}
	if weights != nil {
		return weights, nil
	}
	return ratings, nil
}

func weightList(f fields, name string, count int, scale float64) ([]float64, error) {
	value, ok := f.lookup(name)
	if !ok || value == nil {
		return nil, nil
	}
	items, isList := value.([]any)
	if !isList {
		return nil, badRequest("'%s' must be a list of numbers (got %s %v)", name, jsonType(value), value)
	}
	if len(items) != count {
		return nil, badRequest("'%s' has %d entries for %d text(s)", name, len(items), count)
	}
	out := make([]float64, 0, len(items))
	for _, item := range items {
		number, isNumber := item.(float64)
		if !isNumber {
			return nil, badRequest("'%s' must be a list of numbers (got %s %v)", name, jsonType(item), item)
		}
		if math.IsNaN(number) || math.IsInf(number, 0) || number < 0 {
			return nil, badRequest("'%s' must hold numbers >= 0 (got %v)", name, number)
		}
		out = append(out, number/scale)
	}
	return out, nil
}
