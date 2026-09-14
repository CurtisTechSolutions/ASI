package radixnet

// ChatGPT as the teacher: the client itself, and a round of lessons taught
// through it.  A fake OpenAI server (httptest) answers /v1/models and
// /v1/chat/completions, records the Authorization header it was called with,
// plays the same English teacher as the fake Ollama of tutor_test.go, and can
// reject optional request fields the way a picky model does.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type fakeOpenAI struct {
	server   *httptest.Server
	bodies   []map[string]any
	auth     []string
	reject   []string // fields a request must not carry
	failWith int
	answer   string // replaces the generated answer when set
}

func newFakeOpenAI(t *testing.T) *fakeOpenAI {
	t.Helper()
	fake := &fakeOpenAI{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.auth = append(fake.auth, r.Header.Get("Authorization"))
		if fake.failWith != 0 {
			writeOpenAIError(w, fake.failWith, "boom", "", "")
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			writeJSONBody(w, map[string]any{"object": "list", "data": []map[string]any{
				{"id": "fake-gpt", "owned_by": "openai", "created": 1735689600},
				{"id": "another-gpt", "owned_by": "system", "created": 1735689601},
			}})
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			writeOpenAIError(w, 404, "unknown path", "", "")
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		fake.bodies = append(fake.bodies, body)
		for _, field := range fake.reject {
			if _, present := body[field]; present {
				writeOpenAIError(w, 400, fmt.Sprintf("Unsupported parameter: %q is not supported with this model.", field),
					"unsupported_parameter", field)
				return
			}
		}
		writeJSONBody(w, map[string]any{"choices": []map[string]any{
			{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": fake.reply(body)}},
		}})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func writeOpenAIError(w http.ResponseWriter, status int, message, code, param string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": message, "type": "invalid_request_error", "code": code, "param": param,
	}})
}

// reply plays the English teacher, by the same rule as the fake Ollama.
func (f *fakeOpenAI) reply(body map[string]any) string {
	if f.answer != "" {
		return f.answer
	}
	system, user := "", ""
	messages, _ := body["messages"].([]any)
	for _, item := range messages {
		message, _ := item.(map[string]any)
		content, _ := message["content"].(string)
		if message["role"] == "system" {
			system += content
		} else {
			user += content
		}
	}
	switch {
	case strings.Contains(system, "writing exercises"):
		count := numberIn(system, `exactly (\d+) entries`, 2)
		pool := []map[string]any{
			{"prefix": "the cat sat on", "focus": "prepositions of place", "answer": "the cat sat on the mat"},
			{"prefix": "the dogs run", "focus": "subject-verb agreement", "answer": "the dogs run in the park"},
		}
		exercises := []map[string]any{}
		for i := 0; i < count; i++ {
			exercises = append(exercises, pool[i%len(pool)])
		}
		raw, _ := json.Marshal(map[string]any{"exercises": exercises})
		return string(raw)
	case strings.Contains(system, "marking sentence completions"):
		grades := []map[string]any{}
		for _, line := range strings.Split(user, "\n") {
			match := gradeLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			grade := markSentence(match[2], strings.Split(match[3], "   (drilling:")[0])
			grade["index"] = numberIn(match[1], `(\d+)`, 0)
			grades = append(grades, grade)
		}
		raw, _ := json.Marshal(map[string]any{"grades": grades})
		return string(raw)
	case strings.Contains(system, "model sentences"):
		count := numberIn(system, `exactly (\d+) lines`, 3)
		lines := make([]string, count)
		for i := range lines {
			lines[i] = fmt.Sprintf("%d. the cat sat on the mat number %d", i+1, i)
		}
		return strings.Join(lines, "\n")
	}
	return "answer to: " + strings.TrimSpace(user)
}

func (f *fakeOpenAI) client(t *testing.T) *ChatGPTClient {
	t.Helper()
	client, err := NewChatGPTClient(f.server.URL, "fake-gpt", 0)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

// -- the client --------------------------------------------------------------

func TestNormaliseChatGPTURL(t *testing.T) {
	cases := map[string]string{
		"api.openai.com":              "https://api.openai.com/v1",
		"https://api.openai.com/v1/":  "https://api.openai.com/v1",
		"  http://localhost:1234/v1 ": "http://localhost:1234/v1",
		"http://127.0.0.1:8080":       "http://127.0.0.1:8080/v1",
	}
	for raw, want := range cases {
		got, err := NormaliseChatGPTURL(raw)
		if err != nil || got != want {
			t.Fatalf("NormaliseChatGPTURL(%q) = %q, %v; want %q", raw, got, err, want)
		}
	}
	for _, bad := range []string{"", "   ", "ftp://host/v1", "https://"} {
		if _, err := NormaliseChatGPTURL(bad); err == nil {
			t.Fatalf("%q must be refused", bad)
		}
	}
	if _, err := NormaliseChatGPTURL("http://gateway.example.com/v1"); err == nil ||
		!strings.Contains(err.Error(), "unencrypted") {
		t.Fatalf("a plain-http remote host must be refused, got %v", err)
	}
	t.Setenv("RADIXNET_OPENAI_ALLOW_INSECURE", "1")
	if _, err := NormaliseChatGPTURL("http://gateway.example.com/v1"); err != nil {
		t.Fatalf("the escape hatch must allow it: %v", err)
	}
}

func TestChatGPTClientBasics(t *testing.T) {
	fake := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	client := fake.client(t)
	if client.Provider() != ProviderChatGPT || client.ModelName() != "fake-gpt" {
		t.Fatalf("provider / model wrong: %s %s", client.Provider(), client.ModelName())
	}
	if !client.Available() {
		t.Fatal("the fake server must answer")
	}
	models, err := client.Models()
	if err != nil || len(models) != 2 || models[0]["name"] != "fake-gpt" {
		t.Fatalf("Models() = %v, %v", models, err)
	}
	if fake.auth[0] != "Bearer sk-test-key" {
		t.Fatalf("the key must travel in the Authorization header, got %q", fake.auth[0])
	}
	if strings.Contains(client.String(), "sk-test-key") {
		t.Fatal("the key must never be printed")
	}
	answer, err := client.Generate("hi there", LLMOptions{System: "be brief", Temperature: 0.3})
	if err != nil || answer != "answer to: hi there" {
		t.Fatalf("Generate = %q, %v", answer, err)
	}
	body := fake.bodies[len(fake.bodies)-1]
	if body["model"] != "fake-gpt" || body["temperature"] != 0.3 {
		t.Fatalf("request body wrong: %v", body)
	}
}

func TestChatGPTWithoutAKeySendsNothing(t *testing.T) {
	fake := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY_FILE", "")
	client := fake.client(t)
	if client.Configured() || ChatGPTConfigured() {
		t.Fatal("no key must mean not configured")
	}
	if client.Available() {
		t.Fatal("no key must mean not available")
	}
	_, err := client.Generate("hi", LLMOptions{})
	if err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("the error must name the key, got %v", err)
	}
	if len(fake.bodies) != 0 || len(fake.auth) != 0 {
		t.Fatal("nothing may be sent without a key")
	}
}

func TestChatGPTDropsRejectedFields(t *testing.T) {
	fake := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	fake.reject = []string{"temperature", "response_format"}
	client := fake.client(t)
	answer, err := client.Generate("hi", LLMOptions{Temperature: 0.3, JSON: true})
	if err != nil || answer != "answer to: hi" {
		t.Fatalf("Generate = %q, %v", answer, err)
	}
	if len(fake.bodies) != 3 {
		t.Fatalf("expected two retries, got %d requests", len(fake.bodies))
	}
	last := fake.bodies[len(fake.bodies)-1]
	if _, present := last["temperature"]; present {
		t.Fatalf("temperature should have been dropped: %v", last)
	}
	if _, present := last["response_format"]; present {
		t.Fatalf("response_format should have been dropped: %v", last)
	}
}

func TestChatGPTErrorsCarryTheReason(t *testing.T) {
	fake := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	client := fake.client(t)
	fake.failWith = 401
	_, err := client.Generate("hi", LLMOptions{})
	if !IsChatGPTError(err) || !IsLLMError(err) || !strings.Contains(err.Error(), "HTTP 401") {
		t.Fatalf("expected a ChatGPT 401, got %v", err)
	}
	if !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("a 401 should hint at the key: %v", err)
	}
}

func TestProviderHelpers(t *testing.T) {
	for raw, want := range map[string]string{
		"": ProviderOllama, "ollama": ProviderOllama, "OpenAI": ProviderChatGPT, "gpt": ProviderChatGPT,
	} {
		got, err := NormaliseProvider(raw)
		if err != nil || got != want {
			t.Fatalf("NormaliseProvider(%q) = %q, %v; want %q", raw, got, err, want)
		}
	}
	if _, err := NormaliseProvider("bard"); err == nil {
		t.Fatal("an unknown provider must be refused")
	}
	client, err := NewLLMClient("chatgpt", "https://api.openai.com", "", 0)
	if err != nil || ProviderOf(client) != ProviderChatGPT {
		t.Fatalf("NewLLMClient = %v, %v", client, err)
	}
	if DefaultProviderModel("chatgpt") != DefaultChatGPTModel() ||
		DefaultProviderModel("ollama") != DefaultTutorModel() {
		t.Fatal("each provider must keep its own default model")
	}
}

// -- the lessons -------------------------------------------------------------

func TestTutorTaughtByChatGPT(t *testing.T) {
	fake := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	model := tutorModel(t)
	cfg := tutorConfig()
	cfg.TutorProvider, cfg.TutorModel = ProviderChatGPT, "fake-gpt"
	trainer, err := NewTutorTrainer(model, fake.client(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	records, err := trainer.Run()
	if err != nil {
		t.Fatal(err)
	}
	if len(trainer.Lessons) != 2 {
		t.Fatalf("expected 2 lessons, got %d", len(trainer.Lessons))
	}
	for _, lesson := range trainer.Lessons {
		if lesson.Grade.GradedBy != ProviderChatGPT && lesson.Grade.GradedBy != "empty" {
			t.Fatalf("a lesson should record who marked it: %q", lesson.Grade.GradedBy)
		}
	}
	if records[len(records)-1]["kind"] != "report" {
		t.Fatalf("the run should end with a report card: %v", records[len(records)-1])
	}
	wrote, marked := false, false
	for _, body := range fake.bodies {
		messages, _ := body["messages"].([]any)
		first, _ := messages[0].(map[string]any)
		system, _ := first["content"].(string)
		wrote = wrote || strings.Contains(system, "writing exercises")
		marked = marked || strings.Contains(system, "marking sentence completions")
	}
	if !wrote || !marked {
		t.Fatalf("ChatGPT should have set and marked the exercises (wrote=%v marked=%v)", wrote, marked)
	}
}

func TestTutorTaughtByChatGPTMarkedLocally(t *testing.T) {
	openai := newFakeOpenAI(t)
	ollama := newFakeTeacher(t)
	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	cfg := tutorConfig()
	cfg.TutorProvider, cfg.TutorModel = ProviderChatGPT, "fake-gpt"
	cfg.GraderProvider, cfg.GraderModel = ProviderOllama, "fake:latest"
	trainer, err := NewTutorTrainer(tutorModel(t), openai.client(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainer.GraderClient = ollama.client(t)
	if _, err := trainer.Run(); err != nil {
		t.Fatal(err)
	}
	for _, lesson := range trainer.Lessons {
		if lesson.Grade.GradedBy != ProviderOllama && lesson.Grade.GradedBy != "empty" {
			t.Fatalf("the local model marked it, got %q", lesson.Grade.GradedBy)
		}
	}
	if len(ollama.prompts["grades"]) == 0 {
		t.Fatal("the marking should have gone to the local model")
	}
	if len(ollama.prompts["exercises"]) != 0 {
		t.Fatal("the exercises should have come from ChatGPT")
	}
}

func TestTutorConfigResolvesProviders(t *testing.T) {
	cfg := DefaultTutorConfig()
	cfg.TutorProvider, cfg.TutorModel = "openai", ""
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	if cfg.TutorProvider != ProviderChatGPT || cfg.GraderProvider != ProviderChatGPT {
		t.Fatalf("providers not resolved: %+v", cfg)
	}
	if cfg.TutorModel != DefaultChatGPTModel() || cfg.ResolvedGraderModel() != DefaultChatGPTModel() {
		t.Fatalf("models not resolved: %+v", cfg)
	}
	mixed := DefaultTutorConfig()
	mixed.TutorProvider, mixed.TutorModel, mixed.GraderProvider = ProviderChatGPT, "gpt-x", ProviderOllama
	if err := mixed.Validate(); err != nil {
		t.Fatal(err)
	}
	if mixed.ResolvedGraderModel() != DefaultTutorModel() {
		t.Fatalf("a local marker keeps its own default: %q", mixed.ResolvedGraderModel())
	}
	bad := DefaultTutorConfig()
	bad.TutorProvider = "bard"
	if err := bad.Validate(); err == nil {
		t.Fatal("an unknown provider must be refused")
	}
}
