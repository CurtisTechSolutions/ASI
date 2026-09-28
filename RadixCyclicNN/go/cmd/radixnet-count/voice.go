package main

import (
	"bufio"
	"flag"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// radixnet-count speech talk: one turn of talking with the model by voice.

// historyLines is a conversation file: one line per utterance, `Speaker: text`
// (a bare line is the person's) - Python's _history_lines.
func historyLines(text string) []radixnet.Line {
	pairs := []radixnet.Line{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		speaker, said, found := strings.Cut(line, ":")
		if found && speaker != "" && !strings.Contains(strings.TrimSpace(speaker), " ") {
			pairs = append(pairs, radixnet.Line{Speaker: strings.TrimSpace(speaker), Text: strings.TrimSpace(said)})
		} else {
			pairs = append(pairs, radixnet.Line{Speaker: radixnet.VoiceSpeakers[0], Text: line})
		}
	}
	return pairs
}

// deliverSpeech is where the reply's speech goes - a player (--play), stdout
// (--raw) or a WAV (--out) - and the sink's name.
func deliverSpeech(pcm []byte, rate int, out string, play, raw bool) string {
	switch {
	case play:
		player := phonetok.FindPlayer()
		if player == nil {
			fail("no player found (aplay, paplay, ffplay, play or afplay); use --out FILE or --raw")
		}
		cmd := exec.Command(player[0], player[1:]...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			fail("%v", err)
		}
		if err := cmd.Start(); err != nil {
			fail("%v", err)
		}
		stdin.Write(phonetok.WavHeader(rate, -1))
		stdin.Write(pcm)
		stdin.Close()
		cmd.Wait()
		return player[0]
	case raw:
		w := bufio.NewWriter(os.Stdout)
		w.Write(pcm)
		w.Flush()
		return "stdout"
	default:
		if err := os.WriteFile(out, phonetok.WavBytes(pcm, rate), 0o644); err != nil {
			fail("cannot write %s: %v", out, err)
		}
		return out
	}
}

// cmdSpeechTalk is `speech talk [FILE] [-text ...]`: one turn of talking with
// the model by voice - a recording is heard, learned, answered and spoken
// back, the same turn the Voice tab makes for every utterance (POST
// /api/voice/turn, radixnet/voicechat.go).  Recording from the microphone is
// Python-only, as transcribing is: give the WAV, and the words with it.
func cmdSpeechTalk(args []string) {
	fs := flag.NewFlagSet("speech talk", flag.ExitOnError)
	file := fs.String("file", "", "the recording (WAV)")
	o := addSpeechEncodeFlags(fs)
	noWaveform := fs.Bool("no-waveform", false, "learn the transcript only, not the sound")
	pair := fs.Bool("pair", false, "also learn the waveform followed by its transcript")
	shared := fs.Bool("shared-token", false, "use the plain <speech> for every utterance")
	history := fs.String("history", "", "the conversation so far, one `Speaker: text` line per utterance")
	noTrain := fs.Bool("no-train", false, "do not learn what was said")
	epochs := fs.Int("epochs", 2, "epochs over the utterance's texts")
	modelOut := fs.String("model-out", "", "save the model here instead of --model")
	answer := fs.String("answer", "auto", "who answers: auto (the model, Ollama when it has nothing to say) | model | ollama | none")
	llm := addLLMFlags(fs, "ollama-model")
	persona := fs.String("persona", "", "who Ollama speaks as, when it answers")
	topic := fs.String("topic", "", "what the conversation is about, for Ollama")
	noLearnReply := fs.Bool("no-learn-reply", false, "do not teach the model a reply Ollama wrote")
	mode := fs.String("mode", "beam", "how the model's reply is found: beam | sample")
	maxLength := fs.Int("max-length", 60, "units a reply may add to its context")
	context := fs.Int("context", 12, "characters of the line the reply picks up")
	k := fs.Int("k", 5, "candidates considered (beam: the K most likely)")
	beam := fs.Int("beam", 0, "beam: beam width (0 = max(4 * k, 16))")
	temperature := fs.Float64("temperature", 1, "sample: softmax temperature")
	explore := fs.Int("explore", radixnet.Explore, "times a reply that caught itself repeating may back up and look for another way on")
	noLearn := fs.Bool("no-learn", false, "do not teach the graph where it goes round")
	seeded := fs.Bool("seeded", false, "reply with a private RNG seeded by --seed")
	addGuardFlags(fs)
	noSpeak := fs.Bool("no-speak", false, "answer in text only")
	out := fs.String("out", "reply.wav", "the WAV the reply is written to")
	play := fs.Bool("play", false, "play the reply (aplay, paplay, ffplay, play or afplay)")
	raw := fs.Bool("raw", false, "write the reply as 16-bit PCM to stdout")
	voiceRate := fs.Int("voice-rate", phonetok.Rate, "the voice's sample rate")
	pitch := fs.Float64("pitch", 120, "the voice's base pitch in Hz")
	tempo := fs.Float64("tempo", 1, "the voice's pace")
	gain := fs.Float64("gain", 0.5, "peak level as a share of full scale")
	polish := fs.Int("polish", 0, "acoustic units through the centroid vocoder: Griffin-Lim iterations over the whole reply")
	vocoderFlag := fs.String("vocoder", "auto", "acoustic units: the codebook's neural vocoder when it has one (auto), that or nothing (neural), or the codebook's own (centroid)")
	_ = fs.Parse(permute(fs, args))

	path := *file
	if path == "" && fs.NArg() > 0 {
		path = fs.Arg(0)
	}
	var data []byte
	source := "typed"
	if path != "" {
		read, err := os.ReadFile(path)
		if err != nil {
			fail("cannot read %s: %v", path, err)
		}
		data, source = read, path
	}
	if path == "" && strings.TrimSpace(o.Transcript) == "" {
		fail("nothing was said: give a recording FILE, -text, or both")
	}
	var soFar []radixnet.Line
	if *history != "" {
		body, err := os.ReadFile(*history)
		if err != nil {
			fail("cannot read %s: %v", *history, err)
		}
		soFar = radixnet.TidyHistory(historyLines(string(body)))
	}
	options := radixnet.DefaultVoiceOptions()
	options.Teach = *o
	options.Teach.Token = ""
	options.Teach.Waveform, options.Teach.Pair, options.Teach.Unique = !*noWaveform, *pair, !*shared
	options.Train, options.Epochs = !*noTrain, *epochs
	options.Answer, options.LearnReply = strings.ToLower(strings.TrimSpace(*answer)), !*noLearnReply
	options.Persona, options.Topic = *persona, *topic
	options.Mode, options.MaxLength, options.Context, options.K, options.Beam = *mode, *maxLength, *context, *k, *beam
	options.Temperature, options.Explore, options.Learn = *temperature, *explore, !*noLearn
	options.Speak, options.VoiceRate = !*noSpeak, *voiceRate
	options.Pitch, options.Tempo, options.Gain, options.Polish = *pitch, *tempo, *gain, *polish
	options.Vocoder = *vocoderFlag
	if *seeded {
		seed := seedFlag
		options.Seed = &seed
	}
	if err := options.Validate(); err != nil {
		fail("%v", err)
	}
	var client radixnet.LLMClient
	if options.Answer == "auto" || options.Answer == "ollama" {
		ollama, err := radixnet.NewOllamaClient(*llm.url, *llm.model, time.Duration(*llm.timeout*float64(time.Second)))
		if err != nil {
			fail("%v", err)
		}
		client = ollama
	}
	m := openModel(false)
	guard := openGuard(m)
	var rng *radixnet.MT19937
	if options.Seed != nil {
		rng = radixnet.NewMT19937(*options.Seed)
	}
	learned := 0
	parts := radixnet.VoiceParts{
		Learn: func(texts []string) (map[string]any, error) {
			records, err := m.Train(texts, radixnet.TrainOptions{Epochs: options.Epochs, AutoCompress: true})
			if err != nil {
				return nil, err
			}
			learned += len(texts)
			var loss any
			if len(records) > 0 {
				loss = records[len(records)-1]["loss"]
			}
			return map[string]any{"texts": len(texts), "epochs": len(records), "loss": loss}, nil
		},
		ModelReply: func(line string, heard *radixnet.Heard) (*radixnet.Turn, error) {
			ro := radixnet.DefaultReplyOptions()
			ro.Heard, ro.Index, ro.Speaker = heard, 1+len(soFar), radixnet.VoiceSpeakers[1]
			ro.Mode, ro.MaxLength, ro.Context, ro.Temperature = options.Mode, options.MaxLength, options.Context, options.Temperature
			ro.K, ro.Beam, ro.RNG, ro.Explore, ro.Learn = options.K, options.Beam, rng, options.Explore, options.Learn
			if guard != nil {
				seen := map[string]bool{}
				ro.Veto = func(text string) bool {
					if refused, known := seen[text]; known {
						return refused
					}
					refused := guard.Judge(text).Decision == "reject"
					seen[text] = refused
					return refused
				}
			}
			return m.Reply(line, ro)
		},
	}
	outcome, err := radixnet.VoiceTurn(data, m.Encoding(), options, soFar, parts, client, nil)
	if err != nil {
		fail("%v", err)
	}
	doc := outcome.Document
	doc["source"] = source
	doc["saved"] = nil
	if learned > 0 {
		target := *modelOut
		if target == "" {
			target = modelFile()
		}
		if err := m.Save(target); err != nil {
			fail("cannot save %s: %v", target, err)
		}
		doc["saved"] = map[string]any{"path": target}
	}
	doc["sink"] = nil
	if len(outcome.PCM) > 0 {
		doc["sink"] = deliverSpeech(outcome.PCM, outcome.Rate, *out, *play, *raw)
	}
	reply, _ := doc["reply"].(map[string]any)
	replyText, _ := reply["text"].(string)
	spelled, _ := reply["spelled"].(string)
	if !*raw {
		say("model           %s", modelFile())
		say("source          %s", source)
		transcript, _ := doc["transcript"].(string)
		say("heard           %s", orDash(quoteText(transcript)))
		if trained, ok := doc["trained"].(map[string]any); ok && trained != nil {
			say("learned         %v text(s)", trained["texts"])
		} else {
			say("learned         -")
		}
		say("answered by     %v", doc["by"])
		said := spelled
		if said == "" {
			said = replyText
		}
		say("reply           %s", orDash(quoteText(said)))
		if replyText != "" && replyText != spelled {
			say("in units        %s", quoteText(clip(replyText, 80)))
		}
		if spoken, ok := doc["spoken"].(map[string]any); ok && spoken != nil {
			say("speech          %.2f s at %v Hz -> %v", spoken["seconds"], spoken["rate"], doc["sink"])
		} else {
			say("speech          -")
		}
		if doc["taught"] != nil {
			say("taught the reply yes")
		} else {
			say("taught the reply no")
		}
		if saved, ok := doc["saved"].(map[string]any); ok {
			say("saved           %v", saved["path"])
		} else {
			say("saved           -")
		}
	}
	if problem, _ := reply["ollama_error"].(string); problem != "" {
		note("note: Ollama could not answer (%s)", problem)
	}
	if jsonMode {
		emit(doc)
	}
}

// quoteText is text in double quotes with whitespace escaped (JSON string
// syntax), as the Python CLI quotes what a voice says; "" for nothing.
func quoteText(text string) string {
	if text == "" {
		return ""
	}
	return quote(text)
}
