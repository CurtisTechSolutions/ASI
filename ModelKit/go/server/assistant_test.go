package server

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// The /v1 routes: today's format in both dialects, streamed and not, with the
// dialects' own error envelopes (assistant.go).

func talkEnv(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, false)
	status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2})
	if status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	e.waitJob()
	return e
}

// frames reads a server-sent event stream into (event name, decoded data) pairs.
func frames(t *testing.T, body io.Reader) []map[string]any {
	t.Helper()
	out := []map[string]any{}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	event, data := "", []string{}
	flush := func() {
		if len(data) == 0 {
			return
		}
		raw := strings.Join(data, "\n")
		frame := map[string]any{"event": event, "raw": raw}
		var parsed any
		if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
			frame["data"] = parsed
		}
		out = append(out, frame)
		event, data = "", nil
	}
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event: "):
			event = line[7:]
		case strings.HasPrefix(line, "data: "):
			data = append(data, line[6:])
		}
	}
	flush()
	return out
}

func TestTodaysFormatRoutes(t *testing.T) {
	e := talkEnv(t)
	status, models := e.get("/v1/models")
	if status != 200 || models["object"] != "list" {
		t.Fatalf("models: %d %v", status, models)
	}
	data := models["data"].([]any)
	first := data[0].(map[string]any)
	if first["id"] != "radixnet-count" || first["object"] != "model" || first["active"] != true || first["units"] != "chars" {
		t.Fatalf("the model list: %v", first)
	}

	body := map[string]any{"model": "radixnet-count", "messages": []map[string]any{{"role": "user", "content": "tell me about the cat"}},
		"max_tokens": 30, "learn": false}
	status, doc := e.post("/v1/chat/completions", body)
	if status != 200 || doc["object"] != "chat.completion" || doc["model"] != "radixnet-count" {
		t.Fatalf("chat completions: %d %v", status, doc)
	}
	choice := doc["choices"].([]any)[0].(map[string]any)
	message := choice["message"].(map[string]any)
	said, _ := message["content"].(string)
	thought, _ := message["reasoning_content"].(string)
	if said == "" || !strings.HasPrefix(thought, `answering "tell me about the cat"`) {
		t.Fatalf("the message: %v", message)
	}
	if usage := doc["usage"].(map[string]any); usage["prompt_tokens"].(float64) != float64(len("tell me about the cat")) {
		t.Fatalf("usage: %v", usage)
	}
	for _, name := range []string{"", "radixnet", "count", "RadixNet-Count"} {
		body["model"] = name
		status, again := e.post("/v1/chat/completions", body)
		if status != 200 || again["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"] != said {
			t.Fatalf("the active model answers to %q: %d %v", name, status, again)
		}
	}
	// the Messages dialect
	status, msg := e.post("/v1/messages", map[string]any{"system": "be brief", "messages": []map[string]any{{"role": "user", "content": "the dog sat"}},
		"max_tokens": 30, "learn": false})
	if status != 200 || msg["type"] != "message" || msg["role"] != "assistant" {
		t.Fatalf("messages: %d %v", status, msg)
	}
	blocks := msg["content"].([]any)
	if len(blocks) != 2 || blocks[0].(map[string]any)["type"] != "thinking" || blocks[1].(map[string]any)["type"] != "text" {
		t.Fatalf("the blocks: %v", blocks)
	}
	if !strings.Contains(blocks[0].(map[string]any)["thinking"].(string), "a system prompt was given") {
		t.Fatalf("the system prompt is acknowledged: %v", blocks[0])
	}
	status, counted := e.post("/v1/messages/count_tokens", map[string]any{"system": "be brief", "messages": []map[string]any{{"role": "user", "content": "the dog sat"}}})
	if status != 200 || counted["input_tokens"].(float64) != float64(len("be brief")+len("the dog sat")) {
		t.Fatalf("count_tokens: %d %v", status, counted)
	}
	// the errors take the dialect's shape
	status, bad := e.post("/v1/chat/completions", map[string]any{"messages": "no"})
	if status != 400 || bad["error"].(map[string]any)["param"] != "messages" || bad["error"].(map[string]any)["type"] != "invalid_request_error" {
		t.Fatalf("a bad request: %d %v", status, bad)
	}
	status, bad = e.post("/v1/chat/completions", map[string]any{"model": "radixnet-radix", "messages": []map[string]any{{"role": "user", "content": "x"}}})
	if status != 404 || bad["error"].(map[string]any)["code"] != "model_not_found" {
		t.Fatalf("a model that is not here: %d %v", status, bad)
	}
	status, bad = e.post("/v1/messages", map[string]any{"model": "radix", "messages": []map[string]any{{"role": "user", "content": "x"}}})
	if status != 404 || bad["type"] != "error" || bad["error"].(map[string]any)["type"] != "not_found_error" {
		t.Fatalf("a model that is not here, Messages: %d %v", status, bad)
	}
	status, bad = e.get("/v1/nothing")
	if status != 404 || bad["error"].(map[string]any)["type"] != "invalid_request_error" {
		t.Fatalf("an unknown route: %d %v", status, bad)
	}
	status, bad = e.get("/v1/chat/completions")
	if status != 405 || bad["error"].(map[string]any)["type"] != "invalid_request_error" {
		t.Fatalf("a wrong method: %d %v", status, bad)
	}
}

func TestTodaysFormatStreams(t *testing.T) {
	e := talkEnv(t)
	body := map[string]any{"messages": []map[string]any{{"role": "user", "content": "the cat sat"}}, "max_tokens": 30, "learn": false}
	status, whole := e.post("/v1/chat/completions", body)
	if status != 200 {
		t.Fatalf("whole: %d %v", status, whole)
	}
	said := whole["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"].(string)

	post := func(path string, body map[string]any) *http.Response {
		raw, _ := json.Marshal(body)
		resp, err := http.Post(e.ts.URL+path, "application/json", strings.NewReader(string(raw)))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	resp := post("/v1/chat/completions", body)
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("the stream: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	got := frames(t, resp.Body)
	if got[len(got)-1]["raw"] != "[DONE]" {
		t.Fatalf("the stream ends in [DONE]: %v", got[len(got)-1])
	}
	streamed := ""
	for _, f := range got[:len(got)-1] {
		chunk, _ := f["data"].(map[string]any)
		if chunk["object"] != "chat.completion.chunk" {
			t.Fatalf("a chunk: %v", f)
		}
		for _, c := range chunk["choices"].([]any) {
			if text, ok := c.(map[string]any)["delta"].(map[string]any)["content"].(string); ok {
				streamed += text
			}
		}
	}
	if streamed != said {
		t.Fatalf("the chunks join into the text: %q vs %q", streamed, said)
	}
	if usage := got[len(got)-2]["data"].(map[string]any)["usage"].(map[string]any); usage["prompt_tokens"].(float64) != 11 {
		t.Fatalf("the usage chunk: %v", usage)
	}
	// the Messages dialect streams typed blocks
	delete(body, "stream_options")
	resp = post("/v1/messages", body)
	defer resp.Body.Close()
	got = frames(t, resp.Body)
	names := []string{}
	streamed = ""
	for _, f := range got {
		names = append(names, f["event"].(string))
		if f["event"] == "content_block_delta" {
			delta := f["data"].(map[string]any)["delta"].(map[string]any)
			if delta["type"] == "text_delta" {
				streamed += delta["text"].(string)
			}
		}
	}
	if names[0] != "message_start" || names[len(names)-1] != "message_stop" || names[len(names)-2] != "message_delta" {
		t.Fatalf("the events: %v", names)
	}
	if streamed != said {
		t.Fatalf("the text deltas join into the text: %q vs %q", streamed, said)
	}
	// a model that is not here is a 404 document, not a stream
	body["model"] = "nope"
	resp = post("/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != 404 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		t.Fatalf("a missing model under stream: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
}
