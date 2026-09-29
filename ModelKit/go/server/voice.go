package server

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/CurtisTechSolutions/ASI/ModelKit/go/kit"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Talking with the model by voice (the Voice tab, D-089): every utterance is
// heard, learned, answered and spoken (radixnet/voicechat.go).

func init() {
	route("GET", "/api/voice", rVoice)
	doc("GET", "/api/voice", "talking with the model by voice (the Voice tab): the speakers, the answer modes "+
		"(auto | model | ollama | none), the model's encoding and decoder, the transcription backends and the "+
		"Ollama it would ask")
	route("POST", "/api/voice/turn", rVoiceTurn)
	doc("POST", "/api/voice/turn", "one turn of talking with the model by voice: {recording as JSON {name, "+
		"content_base64} and / or transcript, history: [[speaker, text], ...], answer: auto (the model, Ollama "+
		"when it has nothing to say) | model | ollama | none, train (default on), epochs, learn_reply (teach the "+
		"model a reply Ollama wrote), persona, topic, url, ollama_model, timeout, mode, max_length, context, k, "+
		"beam, temperature, seed, explore, learn, guard, speak, voice_rate, pitch, tempo, gain, polish, rate, "+
		"codec, pair, unique, normalise, waveform} -> {transcript, line, token, texts, audio, units, asr, "+
		"trained, reply: {by, text, spelled, turn, ollama_error}, by, spoken, taught, history, encoding, rate, "+
		"wav_base64}: the utterance is heard and learned (the transcript and the waveform behind one token, or "+
		"a model of acoustic units' units), answered, spoken through the model's voice, and the reply taught "+
		"when Ollama wrote it")
	route("POST", "/api/voice/turn/stream", rVoiceTurnStream)
	doc("POST", "/api/voice/turn/stream", "the same turn as it happens: application/x-ndjson, one event per "+
		"line - heard, trained, reply, audio (pcm_base64: 16-bit PCM chunks at rate, to play as they arrive), "+
		"spoken, taught - then {event: done, ...} with the /api/voice/turn document (without wav_base64: the "+
		"audio was the events)")
}

// voiceRequest is what one voice turn is made of: the recording (if any), the
// options, the history, the Ollama to ask, and how the guard stands (Python's
// _voice_request).
type voiceRequest struct {
	data       []byte
	spoken     bool // a recording came with the request
	options    kit.VoiceOptions
	history    []kit.Line
	ollama     *kit.OllamaClient
	guard      bool
	provenance *bool
}

// voiceOptionsFrom reads one voice turn's settings (Python's _voice_options).
func voiceOptionsFrom(rq *request) (kit.VoiceOptions, error) {
	f := rq.f
	o := kit.DefaultVoiceOptions()
	var err error
	if o.Teach, err = speechOptionsFrom(rq); err != nil {
		return o, err
	}
	o.Teach.Token = ""
	if o.Language, err = f.optText("language", ""); err != nil {
		return o, err
	}
	if o.Train, err = f.flag("train", true); err != nil {
		return o, err
	}
	if o.Epochs, _, err = f.integer("epochs", 2, nil); err != nil {
		return o, err
	}
	if o.LR, _, err = f.number("lr", 0.5, nil); err != nil {
		return o, err
	}
	if o.BatchSize, _, err = f.integer("batch_size", 8, nil); err != nil {
		return o, err
	}
	answer, err := f.optText("answer", "auto")
	if err != nil {
		return o, err
	}
	if o.Answer = strings.ToLower(strings.TrimSpace(answer)); o.Answer == "" {
		o.Answer = "auto"
	}
	if o.LearnReply, err = f.flag("learn_reply", true); err != nil {
		return o, err
	}
	if o.Persona, err = f.optText("persona", ""); err != nil {
		return o, err
	}
	if o.Topic, err = f.optText("topic", ""); err != nil {
		return o, err
	}
	if o.Mode, err = f.optText("mode", "beam"); err != nil {
		return o, err
	}
	if o.Mode == "" {
		o.Mode = "beam"
	}
	if o.MaxLength, _, err = f.integer("max_length", 60, nil); err != nil {
		return o, err
	}
	if o.Context, _, err = f.integer("context", 12, nil); err != nil {
		return o, err
	}
	if o.K, _, err = f.integer("k", 5, nil); err != nil {
		return o, err
	}
	if o.Beam, _, err = f.integer("beam", 0, nil); err != nil {
		return o, err
	}
	if o.StepPenalty, _, err = f.number("step_penalty", 0, nil); err != nil {
		return o, err
	}
	if o.Temperature, _, err = f.number("temperature", 1, nil); err != nil {
		return o, err
	}
	if seed, given, err := f.integer("seed", 0, nil); err != nil {
		return o, err
	} else if given {
		s := int64(seed)
		o.Seed = &s
	}
	if o.AvoidRepeats, err = f.flag("avoid_repeats", true); err != nil {
		return o, err
	}
	if o.AvoidWordRepeats, err = f.flag("avoid_word_repeats", true); err != nil {
		return o, err
	}
	if o.Explore, _, err = f.integer("explore", kit.Explore, nil); err != nil {
		return o, err
	}
	if o.Learn, err = f.flag("learn", true); err != nil {
		return o, err
	}
	if o.OllamaTemperature, _, err = f.number("ollama_temperature", 0.8, nil); err != nil {
		return o, err
	}
	if o.Speak, err = f.flag("speak", true); err != nil {
		return o, err
	}
	if o.VoiceRate, _, err = f.integer("voice_rate", o.VoiceRate, nil); err != nil {
		return o, err
	}
	if o.Pitch, _, err = f.number("pitch", 120, nil); err != nil {
		return o, err
	}
	if o.Tempo, _, err = f.number("tempo", 1, nil); err != nil {
		return o, err
	}
	if o.Gain, _, err = f.number("gain", 0.5, nil); err != nil {
		return o, err
	}
	if o.Polish, _, err = f.integer("polish", 0, nil); err != nil {
		return o, err
	}
	if err := o.Validate(); err != nil {
		return o, badRequest("%v", err)
	}
	return o, nil
}

// voiceRequestFrom reads what one voice turn is made of.
func voiceRequestFrom(rq *request) (*voiceRequest, error) {
	req := &voiceRequest{}
	if rq.f.present("files") || rq.f.present("name") || rq.f.present("content") || rq.f.present("content_base64") {
		_, data, err := mediaBytes(rq, "recording")
		if err != nil {
			return nil, err
		}
		req.data, req.spoken = data, true
	}
	o, err := voiceOptionsFrom(rq)
	if err != nil {
		return nil, err
	}
	req.options = o
	if !req.spoken && strings.TrimSpace(o.Teach.Transcript) == "" {
		return nil, badRequest("nothing was said: send a recording (JSON {name, content_base64}), a transcript, or both")
	}
	var items []any
	switch v, _ := rq.f.lookup("history"); h := v.(type) {
	case nil:
	case []any:
		items = h
	case string:
		if strings.TrimSpace(h) != "" {
			var parsed any
			if err := json.Unmarshal([]byte(h), &parsed); err != nil {
				return nil, badRequest("'history' must be a JSON list of [speaker, text] pairs")
			}
			list, ok := parsed.([]any)
			if !ok {
				return nil, badRequest("'history' must be a list of [speaker, text] pairs")
			}
			items = list
		}
	default:
		return nil, badRequest("'history' must be a list of [speaker, text] pairs")
	}
	if req.history, err = kit.HistoryPairs(items); err != nil {
		return nil, badRequest("%v", err)
	}
	if o.Answer == "auto" || o.Answer == "ollama" {
		url, err := rq.f.optText("url", "")
		if err != nil {
			return nil, err
		}
		model, err := rq.f.optText("ollama_model", "")
		if err != nil {
			return nil, err
		}
		timeout, _, err := rq.f.number("timeout", 0, nil)
		if err != nil {
			return nil, err
		}
		if req.ollama, err = rq.svc.OllamaClient(url, model, timeout); err != nil {
			return nil, err
		}
	}
	if o.Train || (o.LearnReply && (o.Answer == "auto" || o.Answer == "ollama")) {
		if err := rq.svc.ensureIdle(); err != nil {
			return nil, err
		}
	}
	if req.guard, err = rq.f.flag("guard", true); err != nil {
		return nil, err
	}
	if req.provenance, err = optionalFlag(rq.f, "provenance"); err != nil {
		return nil, err
	}
	return req, nil
}

// VoiceTurn is one turn of talking with the model by voice
// (kit.VoiceTurn): heard, trained, answered, spoken, taught.
//
// Training and the model's own reply hold the model's lock for exactly as long
// as they run (a job running elsewhere refuses them, 409); Ollama is asked
// outside it, as the chat loop does, so a slow answer never holds the server.
// The guard vetoes the model's reply as it does every other answer (guard).
func (s *Service) VoiceTurn(req *voiceRequest, write func(map[string]any)) (*kit.VoiceOutcome, error) {
	o := req.options
	found, _ := s.read(func(m *radixnet.Model) (any, error) { return m.Encoding(), nil })
	enc := found.(radixnet.Encoding)
	var rng *radixnet.MT19937
	if o.Seed != nil {
		rng = radixnet.NewMT19937(*o.Seed)
	}
	index := 1 + len(req.history)
	parts := kit.VoiceParts{
		Learn: func(texts []string) (map[string]any, error) {
			if _, err := s.mutate(func(m *radixnet.Model) (any, error) {
				return m.Train(texts, radixnet.TrainOptions{Epochs: o.Epochs, AutoCompress: true})
			}); err != nil {
				return nil, err
			}
			return map[string]any{"texts": len(texts), "epochs": o.Epochs}, nil
		},
		ModelReply: func(line string, heard *kit.Heard) (*kit.Turn, error) {
			out, err := s.read(func(m *radixnet.Model) (any, error) {
				ro := kit.DefaultReplyOptions()
				ro.Heard, ro.Index, ro.Speaker = heard, index, kit.VoiceSpeakers[1]
				ro.Mode, ro.MaxLength, ro.Context, ro.Temperature = o.Mode, o.MaxLength, o.Context, o.Temperature
				ro.K, ro.Beam, ro.StepPenalty, ro.RNG = o.K, o.Beam, o.StepPenalty, rng
				ro.AvoidRepeats, ro.AvoidWordRepeats = o.AvoidRepeats, o.AvoidWordRepeats
				ro.Explore, ro.Learn = o.Explore, o.Learn
				if req.guard {
					if pair := s.guard(req.provenance); pair != nil {
						seen := map[string]bool{} // a candidate offered again after a shorter context is judged once
						ro.Veto = func(text string) bool {
							if refused, known := seen[text]; known {
								return refused
							}
							refused := pair.Judge(text).Decision == "reject"
							seen[text] = refused
							return refused
						}
					}
				}
				return kit.Reply(m, line, ro)
			})
			if err != nil {
				return nil, err
			}
			turn, _ := out.(*kit.Turn)
			return turn, nil
		},
	}
	var client kit.LLMClient
	if req.ollama != nil {
		client = req.ollama
	}
	outcome, err := kit.VoiceTurn(req.data, enc, o, req.history, parts, client, write)
	if err != nil {
		if _, ok := err.(*apiError); ok {
			return nil, err
		}
		return nil, badRequest("%v", err) // unreadable audio, nothing said, a bad history, a token that is no unit
	}
	return outcome, nil
}

// rVoice is GET /api/voice: what the Voice tab has to work with.
func rVoice(rq *request) (int, any, error) {
	found, _ := rq.svc.read(func(m *radixnet.Model) (any, error) { return m.Encoding(), nil })
	client, _ := rq.svc.OllamaClient("", "", 0)
	return 200, kit.DescribeVoice(found.(radixnet.Encoding), client), nil
}

// rVoiceTurn is POST /api/voice/turn: one turn of talking with the model by
// voice, answered whole: the document and the reply's WAV.
func rVoiceTurn(rq *request) (int, any, error) {
	req, err := voiceRequestFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	outcome, err := rq.svc.VoiceTurn(req, nil)
	if err != nil {
		return 0, nil, err
	}
	doc := outcome.Document
	doc["rate"] = outcome.Rate
	doc["wav_base64"] = nil
	if len(outcome.PCM) > 0 {
		doc["wav_base64"] = base64.StdEncoding.EncodeToString(outcome.WAV())
	}
	return 200, doc, nil
}

// rVoiceTurnStream is POST /api/voice/turn/stream: the same turn as it
// happens, every step one JSON line, the reply's audio in chunks played as
// they arrive, then done with the whole document (without wav_base64: the
// audio was the events).
func rVoiceTurnStream(rq *request) (int, any, error) {
	req, err := voiceRequestFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	return 200, &streamResponse{run: func(write func(any) error) error {
		// a client that goes away mid-turn does not stop the turn, which finishes as it would have; its
		// events are simply not written any more
		var gone error
		outcome, err := rq.svc.VoiceTurn(req, func(event map[string]any) {
			if gone == nil {
				gone = write(event)
			}
		})
		if err != nil {
			return err
		}
		if gone != nil {
			return nil
		}
		doc := outcome.Document
		doc["event"] = "done"
		doc["rate"] = outcome.Rate
		return write(doc)
	}}, nil
}
