package server

import (
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// An LLM on the other side of the line: the partner talks to the model, the
// model replies by walking its graph, and the judge marks every reply against
// the line it answered.  Unlike the critic this loop *does* train the positive
// model, so it takes the model lock like any other training job; the negative
// network learns from the same verdicts when one is in memory.

// StartChat starts a `chat` job.
func (s *Service) StartChat(
	config radixnet.ChatConfig, client radixnet.LLMClient, judge radixnet.LLMClient,
) (map[string]any, error) {
	if err := config.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
	var negative *radixnet.Model
	if config.Blame || config.Guard {
		found, err := s.negativeModel()
		if err != nil {
			return nil, err
		}
		negative = found
	}
	return s.startJob("chat", func(job *Job, progress func(map[string]any), stop func() bool) error {
		loop, err := radixnet.NewChat(s.model, negative, client, config)
		if err != nil {
			return err
		}
		loop.JudgeClient = judge
		loop.Stop = stop
		// the partner and the judge think without the model lock, so readers stay served
		loop.External = func(fn func() error) error {
			s.mu.Unlock()
			defer s.mu.Lock()
			return fn()
		}
		_, err = loop.Run(func(record map[string]any) {
			s.chatHistory = append(s.chatHistory, record)
			progress(record)
		})
		return err
	})
}

// ChatHistory is the exchange / conversation / report records of every chat run.
func (s *Service) ChatHistory() map[string]any {
	history := s.chatHistory
	if history == nil {
		history = []map[string]any{}
	}
	return map[string]any{"history": history}
}

// ChatCard is the report at the end of the last conversation run, or nil when
// it has never run.
func (s *Service) ChatCard() map[string]any {
	for i := len(s.chatHistory) - 1; i >= 0; i-- {
		if kind, _ := s.chatHistory[i]["kind"].(string); kind == "report" {
			return s.chatHistory[i]
		}
	}
	return nil
}

// -- HTTP -------------------------------------------------------------------

func init() {
	route("POST", "/api/chat/start", rChatStart)
	doc("POST", "/api/chat/start", "start a chat job - an LLM converses with the model and marks every reply: "+
		"{conversations (0 = until stopped), turns, topic, opening, persona, context, max_length, mode: "+
		"beam|sample, k, temperature, partner_temperature, threshold, provider: ollama|chatgpt, partner_model, "+
		"judge_model, url, judge_url, timeout, guard (the negative network vetoes a reply before it is spoken), "+
		"blame, clear_passes, learn (2NRL on the marked replies), teach_partner (the partner's own lines join the "+
		"positive phase), avoid_repeats, neg_epochs, pos_epochs, strength, epochs, seed}")
	route("GET", "/api/chat/history", rChatHistory)
	doc("GET", "/api/chat/history", "exchange / conversation / report records of all chat runs")
}

// chatConfigFrom reads the conversation settings of a request body.
func chatConfigFrom(rq *request) (radixnet.ChatConfig, error) {
	cfg := radixnet.DefaultChatConfig()
	zeroI, oneI := 0, 1
	zero := 0.0
	var err error
	if cfg.Conversations, _, err = rq.f.integer("conversations", cfg.Conversations, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Turns, _, err = rq.f.integer("turns", cfg.Turns, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Topic, err = rq.f.optText("topic", cfg.Topic); err != nil {
		return cfg, err
	}
	if cfg.Opening, err = rq.f.optText("opening", cfg.Opening); err != nil {
		return cfg, err
	}
	if cfg.Persona, err = rq.f.optText("persona", cfg.Persona); err != nil {
		return cfg, err
	}
	if cfg.Context, _, err = rq.f.integer("context", cfg.Context, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.MaxLength, _, err = rq.f.integer("max_length", cfg.MaxLength, &oneI); err != nil {
		return cfg, err
	}
	mode, err := rq.f.optText("mode", cfg.Mode)
	if err != nil {
		return cfg, err
	}
	cfg.Mode = strings.ToLower(strings.TrimSpace(mode))
	if cfg.K, _, err = rq.f.integer("k", cfg.K, &oneI); err != nil {
		return cfg, err
	}
	if cfg.Temperature, _, err = rq.f.number("temperature", cfg.Temperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.PartnerTemperature, _, err = rq.f.number("partner_temperature", cfg.PartnerTemperature, &zero); err != nil {
		return cfg, err
	}
	if cfg.Threshold, _, err = rq.f.number("threshold", cfg.Threshold, &zero); err != nil {
		return cfg, err
	}
	provider, err := rq.f.optText("provider", "")
	if err != nil {
		return cfg, err
	}
	if provider == "" {
		if provider, err = rq.f.optText("partner_provider", ""); err != nil {
			return cfg, err
		}
	}
	if provider != "" {
		if cfg.Provider, err = radixnet.NormaliseProvider(provider); err != nil {
			return cfg, badRequest("'provider' must be one of ollama, chatgpt (got %q)", provider)
		}
	}
	if cfg.PartnerModel, err = rq.f.optText("partner_model", ""); err != nil {
		return cfg, err
	}
	if cfg.PartnerModel == "" {
		if cfg.PartnerModel, err = rq.f.optText("model", ""); err != nil {
			return cfg, err
		}
	}
	if cfg.JudgeModel, err = rq.f.optText("judge_model", ""); err != nil {
		return cfg, err
	}
	for _, field := range []struct {
		name string
		into *bool
	}{
		{"guard", &cfg.Guard}, {"blame", &cfg.Blame}, {"clear_passes", &cfg.ClearPasses},
		{"learn", &cfg.Learn}, {"teach_partner", &cfg.TeachPartner}, {"avoid_repeats", &cfg.AvoidRepeats},
	} {
		if *field.into, err = rq.f.flag(field.name, *field.into); err != nil {
			return cfg, err
		}
	}
	if cfg.NegEpochs, _, err = rq.f.integer("neg_epochs", cfg.NegEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.PosEpochs, _, err = rq.f.integer("pos_epochs", cfg.PosEpochs, &zeroI); err != nil {
		return cfg, err
	}
	if cfg.Strength, _, err = rq.f.number("strength", cfg.Strength, &zero); err != nil {
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

func rChatStart(rq *request) (int, any, error) {
	config, err := chatConfigFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	url, err := rq.f.optText("url", "")
	if err != nil {
		return 0, nil, err
	}
	judgeURL, err := rq.f.optText("judge_url", "")
	if err != nil {
		return 0, nil, err
	}
	timeout, _, err := rq.f.number("timeout", 0, floatp(1))
	if err != nil {
		return 0, nil, err
	}
	client, err := rq.svc.LLMClient(config.Provider, url, config.PartnerModel, timeout)
	if err != nil {
		return 0, nil, err
	}
	judge := client
	if config.JudgeModel != "" || judgeURL != "" {
		if judgeURL == "" {
			judgeURL = url
		}
		if judge, err = rq.svc.LLMClient(config.Provider, judgeURL, config.JudgeModel, timeout); err != nil {
			return 0, nil, err
		}
	}
	job, err := rq.svc.StartChat(config, client, judge)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{
		"job": job, "config": config, "url": client.BaseURL(), "partner": client.ModelName(),
		"judge": judge.ModelName(), "speakers": radixnet.ChatSpeakers[:],
	}, nil
}

func rChatHistory(rq *request) (int, any, error) {
	return 200, rq.svc.ChatHistory(), nil
}
