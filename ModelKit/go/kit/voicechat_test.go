package kit

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Talking with the model by voice (voicechat.go, D-089): one turn on the
// package alone, the model's parts as closures and Ollama faked.

func voiceClip(seconds float64) []byte { return WAVBytes(tone(seconds, 16000, 440, 0.5), 16000, 1) }

func voiceTurn(text string, fresh bool) *Turn {
	context := strings.Fields(text)[0]
	if fresh {
		context = ""
	}
	return &Turn{Index: 1, Speaker: VoiceSpeakers[1], Text: text, Context: context, Reply: text, Cost: 1,
		Probability: 0.5, ReachedEnd: true, Fresh: fresh}
}

func voiceOptions(transcript, answer string) VoiceOptions {
	o := DefaultVoiceOptions()
	o.Teach.Transcript = transcript
	o.Answer = answer
	return o
}

// fakeVoiceOllama answers every /api/generate with one line and remembers the bodies.
type fakeVoiceOllama struct {
	server *httptest.Server
	mu     sync.Mutex
	seen   []map[string]any
}

func newFakeVoiceOllama(t *testing.T) *fakeVoiceOllama {
	t.Helper()
	fake := &fakeVoiceOllama{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path != "/api/generate" {
			w.WriteHeader(404)
			w.Write([]byte("{}"))
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		fake.mu.Lock()
		fake.seen = append(fake.seen, body)
		fake.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"response": "Hi there, friend."})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeVoiceOllama) asked() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.seen)
}

func (f *fakeVoiceOllama) last() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.seen[len(f.seen)-1]
}

func (f *fakeVoiceOllama) client(t *testing.T) *OllamaClient {
	t.Helper()
	client, err := NewOllamaClient(f.server.URL, "fake:latest", 0)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func eventKinds(events []map[string]any) []string {
	kinds := make([]string, 0, len(events))
	for _, e := range events {
		kind, _ := e["event"].(string)
		kinds = append(kinds, kind)
	}
	return kinds
}

func TestVoiceTurnHeardTrainedAnsweredAndSpoken(t *testing.T) {
	events := []map[string]any{}
	learned := [][]string{}
	var askedLine string
	var askedHeard *Heard
	parts := VoiceParts{
		Learn: func(texts []string) (map[string]any, error) {
			learned = append(learned, append([]string{}, texts...))
			return map[string]any{"texts": len(texts), "epochs": 1}, nil
		},
		ModelReply: func(line string, heard *Heard) (*Turn, error) {
			askedLine, askedHeard = line, heard
			return voiceTurn("sat on the mat", false), nil
		},
	}
	o := voiceOptions("the cat sat", "model")
	o.Epochs = 1
	history := []Line{{"You", "hello"}, {"Model", "hi there"}}
	out, err := VoiceTurn(voiceClip(0.2), radixnet.DefaultEncoding(), o, history, parts, nil, func(e map[string]any) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	doc := out.Document
	kinds := eventKinds(events)
	if strings.Join(kinds[:3], ",") != "heard,trained,reply" || kinds[len(kinds)-1] != "spoken" || len(kinds) < 5 {
		t.Fatalf("events: %v", kinds)
	}
	for _, kind := range kinds[3 : len(kinds)-1] {
		if kind != "audio" {
			t.Fatalf("events: %v", kinds)
		}
	}
	// heard: the transcript and the waveform behind one token, both learned before the reply
	texts := doc["texts"].([]string)
	if doc["transcript"] != "the cat sat" || !strings.HasPrefix(doc["token"].(string), "<speech:") || len(texts) != 2 {
		t.Fatalf("heard: %v", doc)
	}
	if !strings.HasSuffix(texts[0], " the cat sat") || !strings.Contains(texts[1], "aud:mu:") {
		t.Fatalf("texts: %v", texts)
	}
	if len(learned) != 1 || strings.Join(learned[0], "|") != strings.Join(texts, "|") {
		t.Fatalf("learned: %v", learned)
	}
	if doc["trained"].(map[string]any)["texts"] != 2 || doc["audio"].(*EncodedAudio).Codec != "mu" {
		t.Fatalf("trained / audio: %v %v", doc["trained"], doc["audio"])
	}
	// answered: the line, with the conversation so far heard
	if askedLine != "the cat sat" || askedHeard == nil || !askedHeard.Duplicate("hi there", "") || !askedHeard.Duplicate("the cat sat", "") {
		t.Fatalf("asked: %q %v", askedLine, askedHeard)
	}
	reply := doc["reply"].(map[string]any)
	if doc["by"] != "model" || reply["text"] != "sat on the mat" || reply["spelled"] != "sat on the mat" || reply["ollama_error"] != nil {
		t.Fatalf("reply: %v", reply)
	}
	if reply["turn"].(*Turn).Text != "sat on the mat" {
		t.Fatalf("turn: %v", reply["turn"])
	}
	spokenPairs, _ := json.Marshal(doc["history"])
	if string(spokenPairs) != `[["You","hello"],["Model","hi there"],["You","the cat sat"],["Model","sat on the mat"]]` {
		t.Fatalf("history: %s", spokenPairs)
	}
	// spoken: the audio events are the reply's speech, chunk by chunk, exactly what the decoder says
	pcm := []byte{}
	samples := 0
	for _, e := range events {
		if e["event"] != "audio" {
			continue
		}
		piece, err := base64.StdEncoding.DecodeString(e["pcm_base64"].(string))
		if err != nil {
			t.Fatal(err)
		}
		if e["samples"].(int) > VoiceChunkSamples || e["rate"] != 16000 {
			t.Fatalf("audio event: %v", e)
		}
		samples += e["samples"].(int)
		pcm = append(pcm, piece...)
	}
	said, _ := Say(radixnet.DefaultEncoding(), []string{"sat on the mat"}, DefaultSayOptions())
	if string(pcm) != string(out.PCM) || string(pcm) != string(said.PCM) {
		t.Fatalf("the audio events are not the reply's speech")
	}
	spoken := doc["spoken"].(map[string]any)
	if spoken["samples"] != samples || spoken["decoder"] != "voice" || out.Rate != 16000 {
		t.Fatalf("spoken: %v (rate %d)", spoken, out.Rate)
	}
	if wav := out.WAV(); !strings.HasPrefix(string(wav), "RIFF") || len(wav) != 44+len(pcm) {
		t.Fatalf("wav: %d bytes", len(wav))
	}
	if doc["taught"] != nil || doc["encoding"] != "char:3:1" {
		t.Fatalf("taught / encoding: %v %v", doc["taught"], doc["encoding"])
	}
}

func TestVoiceTurnTypedTextNothingToSayAndNobodyAsked(t *testing.T) {
	events := []map[string]any{}
	learned, replied := 0, 0
	parts := VoiceParts{
		Learn: func(texts []string) (map[string]any, error) {
			learned++
			return map[string]any{"texts": len(texts)}, nil
		},
		ModelReply: func(string, *Heard) (*Turn, error) {
			replied++
			return nil, nil
		},
	}
	out, err := VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("hello there", "auto"), nil, parts, nil, func(e map[string]any) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	doc := out.Document
	if len(doc["texts"].([]string)) != 1 || doc["audio"] != nil { // the transcript alone, behind its token
		t.Fatalf("typed: %v", doc)
	}
	reply := doc["reply"].(map[string]any)
	if doc["by"] != "none" || reply["text"] != "" || doc["spoken"] != nil || len(out.PCM) != 0 {
		t.Fatalf("nothing to say: %v", doc)
	}
	if strings.Join(eventKinds(events), ",") != "heard,trained,reply" {
		t.Fatalf("events: %v", eventKinds(events))
	}
	if history, _ := json.Marshal(doc["history"]); string(history) != `[["You","hello there"]]` {
		t.Fatalf("history: %s", history)
	}
	if learned != 1 || replied != 1 {
		t.Fatalf("calls: %d %d", learned, replied)
	}
	// nobody asked: learned, not answered
	out, err = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("hello", "none"), nil, parts, nil, nil)
	if err != nil || out.Document["by"] != "none" || replied != 1 {
		t.Fatalf("none: %v %v %d", err, out, replied)
	}
	// not learned when told not to
	o := voiceOptions("hello", "auto")
	o.Train = false
	out, err = VoiceTurn(nil, radixnet.DefaultEncoding(), o, nil, parts, nil, nil)
	if err != nil || out.Document["trained"] != nil || learned != 2 {
		t.Fatalf("no train: %v %v %d", err, out, learned)
	}
	// nothing said at all, a nobody, a history that is no pair
	if _, err := VoiceTurn(nil, radixnet.DefaultEncoding(), DefaultVoiceOptions(), nil, VoiceParts{}, nil, nil); err == nil {
		t.Fatal("nothing said must fail")
	}
	if _, err := VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("hi", "nobody"), nil, VoiceParts{}, nil, nil); err == nil {
		t.Fatal("a nobody must fail")
	}
	if _, err := HistoryPairs([]any{"not a pair"}); err == nil {
		t.Fatal("a history that is no pair must fail")
	}
}

func TestVoiceTurnOllamaAnswersForTheModelAndTheModelIsTaught(t *testing.T) {
	fake := newFakeVoiceOllama(t)
	client := fake.client(t)
	learned := [][]string{}
	learn := func(texts []string) (map[string]any, error) {
		learned = append(learned, append([]string{}, texts...))
		return map[string]any{"texts": len(texts)}, nil
	}
	never := func(string, *Heard) (*Turn, error) { return voiceTurn("never", false), nil }
	events := []map[string]any{}
	o := voiceOptions("how are you", "ollama")
	o.Persona = "a cat"
	out, err := VoiceTurn(nil, radixnet.DefaultEncoding(), o, []Line{{"You", "hello"}, {"Model", "hi"}}, VoiceParts{learn, never}, client, func(e map[string]any) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	doc := out.Document
	reply := doc["reply"].(map[string]any)
	text, _ := reply["text"].(string)
	if doc["by"] != "ollama" || text == "" || reply["spelled"] != text || reply["turn"] != nil {
		t.Fatalf("ollama reply: %v", doc)
	}
	// what was said, then what Ollama said
	if len(learned) != 2 || strings.Join(learned[0], "|") != strings.Join(doc["texts"].([]string), "|") || strings.Join(learned[1], "|") != text {
		t.Fatalf("learned: %v", learned)
	}
	if doc["taught"].(map[string]any)["texts"] != 1 {
		t.Fatalf("taught: %v", doc["taught"])
	}
	told := []string{}
	for _, kind := range eventKinds(events) {
		if kind != "audio" {
			told = append(told, kind)
		}
	}
	if strings.Join(told, ",") != "heard,trained,reply,spoken,taught" || len(out.PCM) < 1000 {
		t.Fatalf("events: %v (%d bytes)", told, len(out.PCM))
	}
	body := fake.last()
	system, _ := body["system"].(string)
	prompt, _ := body["prompt"].(string)
	if body["model"] != "fake:latest" || !strings.Contains(system, "voice of a very small language model") || !strings.Contains(system, "You are a cat") {
		t.Fatalf("system: %v %q", body["model"], system)
	}
	if !strings.Contains(prompt, "You: how are you") || !strings.Contains(prompt, "Model: hi") {
		t.Fatalf("prompt: %q", prompt)
	}
	// auto: the model first; Ollama when it has nothing to say, or only a fresh text that picked up none of the line
	before := fake.asked()
	answers := func(string, *Heard) (*Turn, error) { return voiceTurn("the cat sat", false), nil }
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "auto"), nil, VoiceParts{learn, answers}, client, nil)
	if out.Document["by"] != "model" || fake.asked() != before {
		t.Fatalf("auto with an answer: %v %d", out.Document["by"], fake.asked())
	}
	silent := func(string, *Heard) (*Turn, error) { return nil, nil }
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "auto"), nil, VoiceParts{learn, silent}, client, nil)
	if out.Document["by"] != "ollama" || fake.asked() != before+1 {
		t.Fatalf("auto with nothing: %v %d", out.Document["by"], fake.asked())
	}
	fresh := func(string, *Heard) (*Turn, error) { return voiceTurn("a bird", true), nil }
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "auto"), nil, VoiceParts{learn, fresh}, client, nil)
	if out.Document["by"] != "ollama" || fake.asked() != before+2 {
		t.Fatalf("auto with a fresh text: %v %d", out.Document["by"], fake.asked())
	}
	// not taught when told not to
	o = voiceOptions("the cat", "ollama")
	o.LearnReply = false
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), o, nil, VoiceParts{Learn: learn}, client, nil)
	if out.Document["by"] != "ollama" || out.Document["taught"] != nil {
		t.Fatalf("learn_reply off: %v", out.Document)
	}
}

func TestVoiceTurnAnUnreachableOllamaIsRecordedNotRaised(t *testing.T) {
	down, err := NewOllamaClient("http://127.0.0.1:1", "fake:latest", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	out, err := VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "ollama"), nil, VoiceParts{}, down, nil)
	if err != nil {
		t.Fatal(err)
	}
	reply := out.Document["reply"].(map[string]any)
	problem, _ := reply["ollama_error"].(string)
	if out.Document["by"] != "none" || reply["text"] != "" || !strings.Contains(problem, "cannot reach Ollama") {
		t.Fatalf("down: %v", out.Document)
	}
	// in auto the model's fresh text stands when Ollama is down, and the error is on record
	fresh := func(string, *Heard) (*Turn, error) { return voiceTurn("a bird", true), nil }
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "auto"), nil, VoiceParts{ModelReply: fresh}, down, nil)
	reply = out.Document["reply"].(map[string]any)
	problem, _ = reply["ollama_error"].(string)
	if out.Document["by"] != "model" || reply["text"] != "a bird" || !strings.Contains(problem, "cannot reach Ollama") {
		t.Fatalf("auto, down: %v", out.Document)
	}
	out, _ = VoiceTurn(nil, radixnet.DefaultEncoding(), voiceOptions("the cat", "auto"), nil, VoiceParts{}, nil, nil)
	if out.Document["reply"].(map[string]any)["ollama_error"] != "no Ollama to ask" {
		t.Fatalf("no ollama: %v", out.Document)
	}
}

func TestVoiceTurnAModelOfSoundsAndOneOfAcousticUnits(t *testing.T) {
	// sounds: the model's reply is phones, spoken as they are and spelled for the record
	phones, _ := radixnet.ParseEncoding("phone:3:1")
	dog := func(string, *Heard) (*Turn, error) { return voiceTurn("DH AH0 # D AO1 G", false), nil }
	out, err := VoiceTurn(nil, phones, voiceOptions("the cat", "model"), nil, VoiceParts{ModelReply: dog}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.Document["reply"].(map[string]any)["spelled"] != "the dog" || out.Document["spoken"].(map[string]any)["decoder"] != "voice" {
		t.Fatalf("phones: %v", out.Document)
	}
	said, _ := Say(phones, []string{"DH AH0 # D AO1 G"}, DefaultSayOptions())
	if string(out.PCM) != string(said.PCM) {
		t.Fatal("the phones were not spoken as the decoder speaks them")
	}
	// acoustic units: the recording is heard as units, the one text of sound the model can learn
	acoustic, _ := radixnet.ParseEncoding("acoustic:3:1")
	heard, err := HearUtterance(voiceClip(0.3), acoustic, voiceOptions("the cat", "auto"))
	if err != nil {
		t.Fatal(err)
	}
	if heard.Transcript != "the cat" || heard.Token != nil || len(heard.Texts) != 1 || heard.Units == nil || *heard.Units != heard.Texts[0] {
		t.Fatalf("heard: %+v", heard)
	}
	for _, unit := range strings.Fields(heard.Texts[0]) {
		if !strings.HasPrefix(unit, "q") {
			t.Fatalf("units: %q", heard.Texts[0])
		}
	}
	learned := [][]string{}
	learn := func(texts []string) (map[string]any, error) {
		learned = append(learned, append([]string{}, texts...))
		return map[string]any{"texts": len(texts)}, nil
	}
	units := func(string, *Heard) (*Turn, error) { return voiceTurn("q2 q28 q55 q5", false), nil }
	out, err = VoiceTurn(voiceClip(0.3), acoustic, voiceOptions("the cat", "model"), nil, VoiceParts{learn, units}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(learned) != 1 || strings.Join(learned[0], "|") != strings.Join(out.Document["texts"].([]string), "|") {
		t.Fatalf("learned: %v", learned)
	}
	if out.Document["by"] != "model" || out.Document["spoken"].(map[string]any)["decoder"] != "vocoder" || out.Rate != 16000 {
		t.Fatalf("acoustic reply: %v (rate %d)", out.Document, out.Rate)
	}
	// a reply Ollama wrote is words: spoken through the voice, and not taught to a model of units
	fake := newFakeVoiceOllama(t)
	out, err = VoiceTurn(nil, acoustic, voiceOptions("the cat", "ollama"), nil, VoiceParts{Learn: learn}, fake.client(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.Document["by"] != "ollama" || out.Document["spoken"].(map[string]any)["decoder"] != "voice" || out.Document["taught"] != nil {
		t.Fatalf("ollama for an acoustic model: %v", out.Document)
	}
	if len(out.Document["texts"].([]string)) != 0 || len(learned) != 1 { // nothing of sound to learn from a typed line
		t.Fatalf("typed for an acoustic model: %v %v", out.Document["texts"], learned)
	}
	// without a recording an acoustic model learns nothing, and without a transcript nobody can answer
	out, _ = VoiceTurn(nil, acoustic, voiceOptions("hi", "none"), nil, VoiceParts{}, nil, nil)
	if len(out.Document["texts"].([]string)) != 0 || out.Document["trained"] != nil {
		t.Fatalf("acoustic, typed, none: %v", out.Document)
	}
}

func TestVoiceSmallParts(t *testing.T) {
	pairs, err := HistoryPairs([]any{
		[]any{"A", " x  y "}, map[string]any{"speaker": "", "text": "z"}, []any{"B", "  "},
	})
	if err != nil || len(pairs) != 2 || pairs[0] != (Line{"A", "x y"}) || pairs[1] != (Line{"You", "z"}) {
		t.Fatalf("pairs: %v %v", pairs, err)
	}
	for _, bad := range [][]any{{[]any{"x"}}, {[]any{"A", 1}}, {1}} {
		if _, err := HistoryPairs(bad); err == nil {
			t.Fatalf("%v must be refused", bad)
		}
	}
	acoustic, _ := radixnet.ParseEncoding("acoustic:3:1")
	phones, _ := radixnet.ParseEncoding("phone:3:1")
	if SpeechEncoding(acoustic, "ollama") != radixnet.DefaultEncoding() || SpeechEncoding(acoustic, "model") != acoustic || SpeechEncoding(phones, "ollama") != phones {
		t.Fatal("the speech encoding")
	}
	chunks := AudioEvents(make([]byte, 2*20000), 16000, 8000)
	if len(chunks) != 3 || chunks[0]["samples"] != 8000 || chunks[1]["samples"] != 8000 || chunks[2]["samples"] != 4000 {
		t.Fatalf("chunks: %v", chunks)
	}
	if piece, _ := base64.StdEncoding.DecodeString(chunks[0]["pcm_base64"].(string)); len(piece) != 16000 {
		t.Fatalf("chunk bytes: %d", len(piece))
	}
	for _, bad := range []func(o *VoiceOptions){
		func(o *VoiceOptions) { o.Answer = "x" }, func(o *VoiceOptions) { o.Mode = "dijkstra" },
		func(o *VoiceOptions) { o.Epochs = -1 }, func(o *VoiceOptions) { o.Pitch = 0 },
		func(o *VoiceOptions) { o.K = 0 }, func(o *VoiceOptions) { o.Chunk = 0 },
	} {
		o := DefaultVoiceOptions()
		bad(&o)
		if o.Validate() == nil {
			t.Fatalf("%+v must be refused", o)
		}
	}
	if err := DefaultVoiceOptions().Validate(); err != nil {
		t.Fatal(err)
	}
	o := DefaultVoiceOptions()
	o.Answer = "x"
	if o.Validate().Error() != "'answer' must be one of auto, model, ollama, none, got 'x'" {
		t.Fatalf("message: %v", o.Validate())
	}
	client, _ := NewOllamaClient("http://127.0.0.1:1", "m", 0)
	info := DescribeVoice(phones, client)
	speakers, _ := json.Marshal(info["speakers"])
	if string(speakers) != `["You","Model"]` || info["decoder"] != "voice" || info["acoustic"] != false {
		t.Fatalf("describe: %v", info)
	}
	if ollama := info["ollama"].(map[string]any); ollama["url"] != "http://127.0.0.1:1" || ollama["model"] != "m" {
		t.Fatalf("ollama: %v", info["ollama"])
	}
	if _, ok := info["transcription"].(map[string]any)["backends"]; !ok || DescribeVoice(acoustic, nil)["ollama"] != nil {
		t.Fatalf("transcription / no ollama: %v", info)
	}
}
