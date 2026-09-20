package server

import (
	"path/filepath"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// The self-upgrade loop as a job: the model generates, a discriminator judges,
// and 2NRL follows.  The discriminator lives beside the model and is kept
// between runs, as the Python server's does.

// discriminatorPath is discriminator.json beside the model path.
func (s *Service) discriminatorPath() string {
	if s.modelPath == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(s.modelPath), "discriminator.json")
}

// discriminatorModel is the critic of this service (nil until first used, then
// loaded from its file or created fresh alongside the generator).
func (s *Service) discriminatorModel(seed int64) (*radixnet.Model, error) {
	if s.discriminator != nil {
		return s.discriminator, nil
	}
	path := s.discriminatorPath()
	if path != "" {
		if m, err := radixnet.Load(path); err == nil {
			s.discriminator = m
			s.logf("discriminator loaded from " + path)
			return m, nil
		}
	}
	// the discriminator reads the generator's texts: same encoding, or it
	// would be judging grams the generator never writes
	opts := radixnet.DefaultGraphOptions()
	opts.Encoding = s.model.Encoding()
	fresh, err := radixnet.NewModel(seed+1, opts)
	if err != nil {
		return nil, err
	}
	fresh.Workers, fresh.Exact = s.workers, s.exact
	s.discriminator = fresh
	return fresh, nil
}

// StartEvolve starts an `evolve` job over the given corpus.
func (s *Service) StartEvolve(corpus []string, config radixnet.EvolveConfig, generations int, blame bool) (map[string]any, error) {
	if err := config.Validate(s.model.Encoding()); err != nil {
		return nil, badRequest("%v", err)
	}
	if generations < 0 {
		return nil, badRequest("generations must be >= 0 (0 = until stopped)")
	}
	discriminator, err := s.discriminatorModel(config.Seed)
	if err != nil {
		return nil, err
	}
	var negative *radixnet.Model
	if blame {
		found, err := s.negativeModel()
		if err != nil {
			return nil, err
		}
		negative = found
	}
	return s.startJob("evolve", func(job *Job, progress func(map[string]any), stop func() bool) error {
		loop, err := radixnet.NewEvolver(s.model, corpus, discriminator, config)
		if err != nil {
			return err
		}
		loop.Negative = negative
		loop.Stop = stop
		loop.Progress = func(record map[string]any) {
			s.evolveHistory = append(s.evolveHistory, record)
			progress(record)
		}
		_, err = loop.Run(generations)
		return err
	})
}

// EvolveHistory is the generation records of every evolve run.
func (s *Service) EvolveHistory() map[string]any {
	history := s.evolveHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("POST", "/api/evolve/start", rEvolveStart)
	doc("POST", "/api/evolve/start", "start the self-upgrade loop: {corpus | corpus_text | corpus_files, "+
		"generations (0 = until stopped), samples, real_per_generation, max_length, temperature, neg_epochs, "+
		"pos_epochs, disc_neg_epochs, disc_pos_epochs, strength, blatant_mode: none|fail_invert|activation|state, "+
		"blatant_margin, blatant_boost, blame (the discriminator also teaches the negative network)}")
	route("POST", "/api/evolve/stop", rEvolveStop)
	doc("POST", "/api/evolve/stop", "ask the running evolve job to stop after the current generation")
	route("GET", "/api/evolve/history", rEvolveHistory)
	doc("GET", "/api/evolve/history", "generation records of all evolve runs")
}

func evolveConfigFrom(rq *request) (radixnet.EvolveConfig, error) {
	cfg := radixnet.DefaultEvolveConfig()
	cfg.Seed = rq.svc.seed
	zeroI, oneI := 0, 1
	zero, one := 0.0, 1.0
	var err error
	if cfg.Samples, _, err = rq.f.integer("samples", cfg.Samples, &oneI); err != nil {
		return cfg, err
	}
	if cfg.RealPerGeneration, _, err = rq.f.integer("real_per_generation", cfg.RealPerGeneration, &oneI); err != nil {
		return cfg, err
	}
	if cfg.MaxLength, _, err = rq.f.integer("max_length", cfg.MaxLength, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Temperature, _, err = rq.f.number("temperature", cfg.Temperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.NegEpochs, _, err = rq.f.integer("neg_epochs", cfg.NegEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.PosEpochs, _, err = rq.f.integer("pos_epochs", cfg.PosEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.DiscNegEpochs, _, err = rq.f.integer("disc_neg_epochs", cfg.DiscNegEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.DiscPosEpochs, _, err = rq.f.integer("disc_pos_epochs", cfg.DiscPosEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Strength, _, err = rq.f.number("strength", cfg.Strength, &zero); err != nil {
		return cfg, err
	}
	if cfg.BlatantMode, err = rq.f.optText("blatant_mode", cfg.BlatantMode); err != nil {
		return cfg, err
	}
	if cfg.BlatantMargin, _, err = rq.f.number("blatant_margin", cfg.BlatantMargin, &zero); err != nil {
		return cfg, err
	}
	if cfg.BlatantBoost, _, err = rq.f.number("blatant_boost", cfg.BlatantBoost, &one); err != nil {
		return cfg, err
	}
	if seed, ok, err := rq.f.integer("seed", 0, nil); err != nil {
		return cfg, err
	} else if ok {
		cfg.Seed = int64(seed)
	}
	if err := cfg.Validate(rq.svc.Model().Encoding()); err != nil {
		return cfg, badRequest("%v", err)
	}
	return cfg, nil
}

func rEvolveStart(rq *request) (int, any, error) {
	unit, pageLines, err := splitOptions(rq.f)
	if err != nil {
		return 0, nil, err
	}
	wholeFile, err := rq.f.flag("whole_file", false)
	if err != nil {
		return 0, nil, err
	}
	corpus, err := textsAndFiles(rq, "corpus", "corpus_text", "corpus_files", unit, pageLines, wholeFile)
	if err != nil {
		return 0, nil, err
	}
	config, err := evolveConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	zeroI := 0
	generations, _, err := rq.f.integer("generations", 0, &zeroI)
	if err != nil {
		return 0, nil, err
	}
	blame, err := rq.f.flag("blame", false)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartEvolve(corpus, config, generations, blame)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{"job": job, "config": config, "generations": generations, "corpus": len(corpus)}, nil
}

func rEvolveStop(rq *request) (int, any, error) {
	job, err := rq.svc.StopJob()
	if err != nil {
		return 0, nil, err
	}
	return 200, job, nil
}

func rEvolveHistory(rq *request) (int, any, error) {
	return 200, rq.svc.EvolveHistory(), nil
}
