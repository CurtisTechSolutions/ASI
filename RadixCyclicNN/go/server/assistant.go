package server

// The model in today's format, served: POST /v1/chat/completions (OpenAI's
// Chat Completions), POST /v1/messages (Anthropic's Messages), their token
// count and the model list - the same routes the Python server answers, with
// the same documents, streamed as server-sent events from inside the search
// when the request says stream: true (radixnet/assistant.go).

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

func init() {
	route("POST", "/v1/chat/completions", rChatCompletions)
	doc("POST", "/v1/chat/completions", "today's format, OpenAI's dialect: {model, messages, max_tokens, temperature, stop, n, stream, stream_options, tools, thinking, and the dialogue's dials (mode, context, k, explore, avoid_repeats, learn, guard, seed)} -> a chat.completion whose message carries the thinking as reasoning_content, or chat.completion.chunk events with stream: true; a token is one unit of the model's encoding")
	route("POST", "/v1/messages", rMessages)
	doc("POST", "/v1/messages", "today's format, Anthropic's dialect: {model, system, messages, max_tokens, stop_sequences, stream, tools, thinking, and the same dials} -> a message of thinking, text and tool_use blocks, or the message_start / content_block_* / message_delta / message_stop events with stream: true")
	route("POST", "/v1/messages/count_tokens", rCountTokens)
	doc("POST", "/v1/messages/count_tokens", "{system, messages} -> {input_tokens}: the units of the model's encoding the request holds")
	route("GET", "/v1/models", rV1Models)
	doc("GET", "/v1/models", "the models in memory as an OpenAI model list: radixnet-<kind>, the active one first")
}

// v1Error is an error a /v1 route answers in its dialect's own envelope, naming the field to blame.
type v1Error struct {
	status  int
	message string
	param   string
	dialect string
}

func (e *v1Error) Error() string { return e.message }

// eventStream is an answer written as server-sent events: run writes every
// frame the moment it is produced through the function it is handed, and
// errorFrame is the frame a failure after the headers went out is reported with.
type eventStream struct {
	run        func(write func(string) error) error
	errorFrame func(string) string
}

// writeStream sends a streamed answer: the headers, then every frame as it is
// produced (flushed one by one), then the end of the body.
func (h *Handler) writeEvents(w http.ResponseWriter, es *eventStream) int {
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)
	write := func(frame string) error {
		if _, err := io.WriteString(w, frame); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}
	if err := es.run(write); err != nil {
		// the headers are out: the failure goes down the stream in the dialect's own frame
		_ = write(es.errorFrame(err.Error()))
	}
	return 200
}

// voiceFor is the model a request's model names: the running one for "",
// "radixnet", "count" or "word" on a word model - this server runs the count
// model only - and a 404 for anything else.
func (s *Service) voiceFor(name string) (*radixnet.Model, error) {
	kind := radixnet.KindOfID(name)
	if kind == "" || kind == "count" || (kind == "word" && s.units() == "words") {
		return s.model, nil
	}
	return nil, &apiError{404, "the model " + radixnet.PythonRepr(name) + " is not in memory; the models here are " +
		strings.Join(s.modelIDs(), ", ") + " (this server runs the count / reward model only)"}
}

// modelIDs are the ids of every model in memory, the running one first.
func (s *Service) modelIDs() []string {
	ids := []string{radixnet.ModelID(s.model)}
	if s.negative != nil {
		ids = append(ids, radixnet.ModelID(s.negative))
	}
	return ids
}

// ListModels is GET /v1/models: the models in memory as an OpenAI model list.
func (s *Service) ListModels() map[string]any {
	describe := func(m *radixnet.Model, active bool) map[string]any {
		label := ModelLabel
		if m.IsNegative() {
			label = "Negative network"
		}
		return map[string]any{"id": radixnet.ModelID(m), "object": "model", "created": s.started.Unix(), "owned_by": "radixnet",
			"kind": m.Kind(), "label": label, "encoding": m.Encoding().String(), "units": m.Encoding().UnitsName(), "active": active}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	data := []map[string]any{describe(s.model, true)}
	if s.negative != nil {
		data = append(data, describe(s.negative, false))
	}
	return map[string]any{"object": "list", "data": data}
}

// Respond answers a request in today's format, the guard on the way out, every
// event to emit as it happens (nil: none).  The model lock is held for the
// whole reply - the stream writes from inside the search.
func (s *Service) Respond(ask *radixnet.Ask, emit func(map[string]any)) (*radixnet.AssistantReply, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		voice, err := s.voiceFor(ask.Model)
		if err != nil {
			return nil, err
		}
		var pair *radixnet.Filter
		if ask.Guard {
			pair = s.guard(nil)
		}
		reply, err := radixnet.Respond(voice, pair, ask, "", emit)
		if err != nil {
			var bad *radixnet.AskError
			if errors.As(err, &bad) {
				return nil, &apiError{400, bad.Message}
			}
			return nil, badRequest("%v", err)
		}
		return reply, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(*radixnet.AssistantReply), nil
}

// CountInput is POST /v1/messages/count_tokens: the units of the model's encoding a request holds.
func (s *Service) CountInput(ask *radixnet.Ask) (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		voice, err := s.voiceFor(ask.Model)
		if err != nil {
			return nil, err
		}
		return map[string]any{"input_tokens": radixnet.InputUnits(voice.Encoding(), ask)}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// v1Ask reads the request body in the dialect's shape; a bad one is a v1Error naming the field.
func v1Ask(rq *request, dialect string) (*radixnet.Ask, error) {
	body := map[string]any{}
	if len(strings.TrimSpace(string(rq.body))) > 0 {
		if err := json.Unmarshal(rq.body, &body); err != nil {
			return nil, &v1Error{400, "request body is not valid JSON", "", dialect}
		}
	}
	ask, err := radixnet.ParseRequest(body, dialect)
	if err != nil {
		var bad *radixnet.AskError
		if errors.As(err, &bad) {
			return nil, &v1Error{400, bad.Message, bad.Param, dialect}
		}
		return nil, &v1Error{400, err.Error(), "", dialect}
	}
	return ask, nil
}

// inDialect turns any error into the dialect's envelope.
func inDialect(err error, dialect string) error {
	var already *v1Error
	if errors.As(err, &already) {
		return err
	}
	var ae *apiError
	if errors.As(err, &ae) {
		param := ""
		if ae.status == 404 && strings.HasPrefix(ae.message, "the model ") {
			param = "model"
		}
		return &v1Error{ae.status, ae.message, param, dialect}
	}
	return &v1Error{400, err.Error(), "", dialect}
}

// talkRoute answers one of the two dialects: the whole document, or the stream.
func talkRoute(rq *request, dialect string) (int, any, error) {
	ask, err := v1Ask(rq, dialect)
	if err != nil {
		return 0, nil, err
	}
	if !ask.Stream {
		reply, err := rq.svc.Respond(ask, nil)
		if err != nil {
			return 0, nil, inDialect(err, dialect)
		}
		return 200, radixnet.ToFormat(reply, dialect), nil
	}
	// a model that is not here is a 404 before any frame goes out
	rq.svc.mu.RLock()
	_, err = rq.svc.voiceFor(ask.Model)
	rq.svc.mu.RUnlock()
	if err != nil {
		return 0, nil, inDialect(err, dialect)
	}
	svc := rq.svc
	errorFrame := radixnet.OpenAIErrorFrame
	if dialect == radixnet.FormatAnthropic {
		errorFrame = radixnet.AnthropicErrorFrame
	}
	return 200, &eventStream{errorFrame: errorFrame, run: func(write func(string) error) error {
		var openai *radixnet.OpenAIStream
		var anthropic *radixnet.AnthropicStream
		var failed error
		emit := func(event map[string]any) {
			if failed != nil {
				return
			}
			if event["type"] == "start" && openai == nil && anthropic == nil {
				token, _ := event["id"].(string)
				model, _ := event["model"].(string)
				created, _ := event["created"].(int64)
				if dialect == radixnet.FormatOpenAI {
					openai = &radixnet.OpenAIStream{Token: token, Model: model, Created: created, IncludeUsage: ask.IncludeUsage, Thinking: ask.Thinking}
				} else {
					inputUnits, _ := event["input_units"].(int)
					anthropic = radixnet.NewAnthropicStream(token, model, created, inputUnits, ask.Thinking)
				}
			}
			var frames []string
			if openai != nil {
				frames = openai.Frames(event)
			} else if anthropic != nil {
				frames = anthropic.Frames(event)
			}
			for _, frame := range frames {
				if err := write(frame); err != nil {
					failed = err
					return
				}
			}
		}
		if _, err := svc.Respond(ask, emit); err != nil {
			return err
		}
		return failed
	}}, nil
}

func rChatCompletions(rq *request) (int, any, error) { return talkRoute(rq, radixnet.FormatOpenAI) }

func rMessages(rq *request) (int, any, error) { return talkRoute(rq, radixnet.FormatAnthropic) }

func rCountTokens(rq *request) (int, any, error) {
	ask, err := v1Ask(rq, radixnet.FormatAnthropic)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.CountInput(ask)
	if err != nil {
		return 0, nil, inDialect(err, radixnet.FormatAnthropic)
	}
	return 200, out, nil
}

func rV1Models(rq *request) (int, any, error) { return 200, rq.svc.ListModels(), nil }
