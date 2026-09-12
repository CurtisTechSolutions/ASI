package server

// The English tutor over HTTP: GET /api/tutor describes it, POST
// /api/tutor/start runs rounds of prefix -> completion -> grade -> 2NRL as a
// job, GET /api/tutor/history returns their records and POST /api/tutor/lesson
// marks one round without training anything.  The Go twin of the Python
// server's /api/tutor endpoints, answering the same JSON.

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

// DefaultTutorModel is the teacher this server uses when a request names none.
func (s *Service) DefaultTutorModel() string { return s.ollamaModel }

// StartTutor starts a tutor job: rounds of exercise, completion, grade and 2NRL.
func (s *Service) StartTutor(config radixnet.TutorConfig, client *radixnet.OllamaClient) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	return s.startJob("tutor", func(job *Job, progress func(map[string]any), stop func() bool) error {
		trainer, err := radixnet.NewTutorTrainer(s.model, client, config)
		if err != nil {
			return err
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

// TutorLesson runs one round of lessons without training: the exercises (or
// the given prefixes), the completions and their grades.
func (s *Service) TutorLesson(config radixnet.TutorConfig, client *radixnet.OllamaClient, prefixes []string) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	trainer, err := radixnet.NewTutorTrainer(s.model, client, config)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	source := "ollama"
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
		"source": source, "model": client.Model, "url": client.URL, "config": config,
		"exercises": exercises, "lessons": lessons, "report": radixnet.ReportCard(lessons),
	}, nil
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("GET", "/api/tutor", rTutor)
	doc("GET", "/api/tutor", "the English tutor: the default teacher model, the error types a completion is marked with, the completion modes and every default setting")
	route("POST", "/api/tutor/start", rTutorStart)
	doc("POST", "/api/tutor/start", "start a tutor job - Ollama writes the prefixes, the network completes them, Ollama marks the grammar and the 2NRL follows: {topic, rounds, exercises, attempts, focus, level, mode, threshold, grammar_weight, drills, adapt, twonrl_per: round|lesson, min_weight, neg_epochs, pos_epochs, strength, tutor_model, url, ...}")
	route("GET", "/api/tutor/history", rTutorHistory)
	doc("GET", "/api/tutor/history", "lesson / round / report records of all tutor runs")
	route("POST", "/api/tutor/lesson", rTutorLesson)
	doc("POST", "/api/tutor/lesson", "one round of lessons without training: {topic, exercises, prefixes (skip the LLM and use these), attempts, threshold, ...} -> completions with grades (grammar, spelling, fluency, error, correction) and a report card")
}

func rTutor(rq *request) (int, any, error) {
	return 200, map[string]any{
		"url":         rq.svc.ollamaURL,
		"model":       rq.svc.DefaultTutorModel(),
		"env_model":   radixnet.DefaultTutorModel(),
		"error_types": radixnet.ErrorTypes,
		"modes":       radixnet.TutorModes,
		"twonrl_per":  radixnet.TwoNRLPer,
		"defaults":    radixnet.DefaultTutorConfig(),
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
	text("tutor_model", c.TutorModel, &c.TutorModel)
	text("model", c.TutorModel, &c.TutorModel)
	text("grader_model", "", &c.GraderModel)
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
	flag("teach_answer", d.TeachAnswer, &c.TeachAnswer)
	flag("learn", d.Learn, &c.Learn)
	text("twonrl_per", d.TwoNRLPer, &c.TwoNRLPer)
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

// tutorClient builds the client of a tutor request (url / timeout overrides).
func tutorClient(rq *request, config radixnet.TutorConfig) (*radixnet.OllamaClient, error) {
	url, err := rq.f.optText("url", "")
	if err != nil {
		return nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return nil, err
	}
	return rq.svc.OllamaClient(url, config.TutorModel, timeout)
}

func rTutorStart(rq *request) (int, any, error) {
	config, err := tutorConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := tutorClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTutor(config, client)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{"job": job, "config": config, "url": client.URL}, nil
}

func rTutorHistory(rq *request) (int, any, error) {
	return 200, rq.svc.TutorHistory(), nil
}

func rTutorLesson(rq *request) (int, any, error) {
	config, err := tutorConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	client, err := tutorClient(rq, config)
	if err != nil {
		return 0, nil, err
	}
	prefixes, err := rq.f.textsOptional("prefixes", "prefix")
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.TutorLesson(config, client, prefixes)
	if err != nil {
		if radixnet.IsOllamaError(err) {
			return 0, nil, &apiError{502, err.Error()}
		}
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
