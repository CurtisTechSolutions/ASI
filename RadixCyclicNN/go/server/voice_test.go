package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// The Voice tab's routes (voice.go, D-089): GET /api/voice, POST /api/voice/turn whole and streamed.

func TestVoiceRoutesHearLearnAnswerAndSpeak(t *testing.T) {
	e := newEnv(t, false)
	if status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2}); status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	e.waitJob()
	status, info := e.get("/api/voice")
	if status != 200 || info["encoding"] != "char:3:1" || info["decoder"] != "voice" || info["acoustic"] != false {
		t.Fatalf("voice: %d %v", status, info)
	}
	speakers, _ := json.Marshal(info["speakers"])
	answers, _ := json.Marshal(info["answers"])
	if string(speakers) != `["You","Model"]` || string(answers) != `["auto","model","ollama","none"]` {
		t.Fatalf("speakers / answers: %s %s", speakers, answers)
	}
	if url, _ := info["ollama"].(map[string]any)["url"].(string); !strings.HasPrefix(url, "http") {
		t.Fatalf("ollama: %v", info["ollama"])
	}
	// a turn whole: the recording and its words are learned, the model answers, the reply is a WAV
	_, st := e.get("/api/status")
	before := st["trained_texts"].(float64)
	status, doc := e.post("/api/voice/turn", map[string]any{
		"name": "u.wav", "content_base64": b64(testWAV()), "transcript": "the cat sat on the mat",
		"answer": "model", "epochs": 1, "seed": 1,
	})
	if status != 200 {
		t.Fatalf("turn: %d %v", status, doc)
	}
	if doc["transcript"] != "the cat sat on the mat" || doc["by"] != "model" || doc["encoding"] != "char:3:1" || doc["rate"] != 16000.0 {
		t.Fatalf("turn: %v", doc)
	}
	trained := doc["trained"].(map[string]any)
	if trained["texts"] != 2.0 || trained["epochs"] != 1.0 {
		t.Fatalf("trained: %v", trained)
	}
	if _, st := e.get("/api/status"); st["trained_texts"] != before+2 {
		t.Fatalf("trained_texts: %v -> %v", before, st["trained_texts"])
	}
	reply := doc["reply"].(map[string]any)
	text, _ := reply["text"].(string)
	if text == "" || reply["turn"].(map[string]any)["speaker"] != "Model" {
		t.Fatalf("reply: %v", reply)
	}
	wav, err := base64.StdEncoding.DecodeString(doc["wav_base64"].(string))
	if err != nil || !bytes.HasPrefix(wav, []byte("RIFF")) || float64(len(wav)) != 44+2*doc["spoken"].(map[string]any)["samples"].(float64) {
		t.Fatalf("wav: %v (%d bytes)", err, len(wav))
	}
	history := doc["history"].([]any)
	last := history[len(history)-1].([]any)
	if history[len(history)-2].([]any)[1] != "the cat sat on the mat" || last[0] != "Model" || last[1] != text {
		t.Fatalf("history: %v", history)
	}
	// a recording alone, with no words heard in it: the sound is learned, nobody can answer
	status, doc = e.post("/api/voice/turn", map[string]any{"name": "u.wav", "content_base64": b64(testWAV()), "answer": "model"})
	texts, _ := doc["texts"].([]any)
	if status != 200 || doc["transcript"] != "" || doc["by"] != "none" || len(texts) != 1 || !strings.Contains(texts[0].(string), "aud:mu:") {
		t.Fatalf("a recording alone: %d %v", status, doc)
	}
	if doc["reply"].(map[string]any)["text"] != "" || doc["spoken"] != nil || doc["wav_base64"] != nil {
		t.Fatalf("a recording alone: %v", doc)
	}
	// typed, unheard of a recording, not spoken, nobody answering
	status, doc = e.post("/api/voice/turn", map[string]any{"transcript": "a bird in the hand", "answer": "none", "speak": false})
	texts, _ = doc["texts"].([]any)
	if status != 200 || doc["by"] != "none" || doc["wav_base64"] != nil || doc["spoken"] != nil || len(texts) != 1 {
		t.Fatalf("typed: %d %v", status, doc)
	}
}

func TestVoiceTurnStreamsAndRefuses(t *testing.T) {
	e := newEnv(t, false)
	if status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2}); status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	e.waitJob()
	raw, _ := json.Marshal(map[string]any{
		"transcript": "the cat sat", "answer": "model", "history": [][]string{{"You", "hello"}, {"Model", "hi there"}},
		"epochs": 1,
	})
	resp, err := http.Post(e.ts.URL+"/api/voice/turn/stream", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/x-ndjson") {
		t.Fatalf("stream: %d %v", resp.StatusCode, resp.Header)
	}
	body, _ := io.ReadAll(resp.Body)
	events := []map[string]any{}
	for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("not a JSON line: %q (%v)", line, err)
		}
		events = append(events, event)
	}
	kinds := []string{}
	pcm := 0
	for _, event := range events {
		kind, _ := event["event"].(string)
		kinds = append(kinds, kind)
		if kind == "audio" {
			piece, _ := base64.StdEncoding.DecodeString(event["pcm_base64"].(string))
			pcm += len(piece)
		}
	}
	joined := strings.Join(kinds, ",")
	if !strings.HasPrefix(joined, "heard,trained,reply,") || !strings.HasSuffix(joined, "spoken,done") || !strings.Contains(joined, "audio") {
		t.Fatalf("events: %v", kinds)
	}
	done := events[len(events)-1]
	if _, there := done["wav_base64"]; there {
		t.Fatalf("done carries the WAV: %v", done)
	}
	history := done["history"].([]any)
	if history[0].([]any)[1] != "hello" || history[1].([]any)[1] != "hi there" {
		t.Fatalf("history: %v", history)
	}
	if float64(pcm/2) != done["spoken"].(map[string]any)["samples"].(float64) {
		t.Fatalf("the audio events are not the spoken samples: %d", pcm/2)
	}
	// a request refused before anything was streamed is an ordinary 400
	status, doc := e.post("/api/voice/turn/stream", map[string]any{"answer": "model"})
	if status != 400 || !strings.Contains(doc["error"].(string), "nothing was said") {
		t.Fatalf("refused stream: %d %v", status, doc)
	}
	for _, body := range []map[string]any{
		{}, {"transcript": "hi", "answer": "nobody"}, {"transcript": "hi", "epochs": "many"},
		{"transcript": "hi", "history": "not a list"}, {"transcript": "hi", "pitch": 0},
		{"name": "u.wav", "content_base64": "bm90IGF1ZGlv", "answer": "model"},
	} {
		if status, doc := e.post("/api/voice/turn", body); status != 400 {
			t.Fatalf("%v: %d %v", body, status, doc)
		}
	}
	// Ollama that cannot be reached: the turn still happens, the error is on record
	status, doc = e.post("/api/voice/turn", map[string]any{
		"transcript": "the cat sat", "answer": "ollama", "url": "http://127.0.0.1:1", "timeout": 1, "speak": false,
	})
	problem, _ := doc["reply"].(map[string]any)["ollama_error"].(string)
	if status != 200 || doc["by"] != "none" || !strings.Contains(problem, "cannot reach Ollama") || doc["trained"].(map[string]any)["texts"] != 1.0 {
		t.Fatalf("ollama down: %d %v", status, doc)
	}
}
