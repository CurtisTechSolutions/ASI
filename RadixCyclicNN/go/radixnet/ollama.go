package radixnet

// Ollama: a minimal client for a local large language model (https://ollama.com).
//
// Only what the tutor needs, over the standard library: the installed models
// (GET /api/tags) and one non-streaming completion (POST /api/generate), plus
// the two helpers that turn an LLM answer back into data - ParseLines for
// "one item per line" answers and loadsLenient for JSON wrapped in prose or
// code fences.
//
// The endpoint follows Ollama's own convention: OLLAMA_HOST (host:port
// without a scheme is accepted) else http://127.0.0.1:11434; the default
// model comes from RADIXNET_OLLAMA_MODEL, else llama3.2.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// DefaultOllamaTimeout is how long one answer may take (local models can be slow).
const DefaultOllamaTimeout = 120 * time.Second

// DefaultOllamaModelName is used when neither the caller nor the environment names a model.
const DefaultOllamaModelName = "llama3.2"

// OllamaError is an Ollama that cannot be reached, answers an error, or returns something unusable.
type OllamaError struct{ Message string }

func (e *OllamaError) Error() string { return e.Message }

func ollamaErrorf(format string, args ...any) error {
	return &OllamaError{Message: fmt.Sprintf(format, args...)}
}

// IsOllamaError reports whether err came from talking to Ollama.
func IsOllamaError(err error) bool {
	var target *OllamaError
	return errors.As(err, &target)
}

// NormaliseOllamaURL turns "host:port" or a full URL into "http://host:port" without a trailing slash.
func NormaliseOllamaURL(url string) (string, error) {
	text := strings.TrimRight(strings.TrimSpace(url), "/")
	if text == "" {
		return "", fmt.Errorf("the Ollama URL is empty")
	}
	if !strings.Contains(text, "://") {
		text = "http://" + text
	}
	return text, nil
}

// DefaultOllamaURL is $OLLAMA_HOST, else the local default.
func DefaultOllamaURL() string {
	url, err := NormaliseOllamaURL(strings.TrimSpace(os.Getenv("OLLAMA_HOST")))
	if err != nil {
		return "http://127.0.0.1:11434"
	}
	return url
}

// DefaultOllamaModel is $RADIXNET_OLLAMA_MODEL, else llama3.2.
func DefaultOllamaModel() string {
	if name := strings.TrimSpace(os.Getenv("RADIXNET_OLLAMA_MODEL")); name != "" {
		return name
	}
	return DefaultOllamaModelName
}

// OllamaClient talks to one Ollama server.  It implements LLMClient, so the
// tutor cannot tell it from a ChatGPTClient.
type OllamaClient struct {
	URL     string
	Model   string
	Timeout time.Duration
	client  *http.Client
}

// NewOllamaClient normalises the URL and fills in the defaults for an empty url / model / timeout.
func NewOllamaClient(url, model string, timeout time.Duration) (*OllamaClient, error) {
	if strings.TrimSpace(url) == "" {
		url = DefaultOllamaURL()
	}
	normalised, err := NormaliseOllamaURL(url)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(model) == "" {
		model = DefaultOllamaModel()
	}
	if timeout <= 0 {
		timeout = DefaultOllamaTimeout
	}
	return &OllamaClient{URL: normalised, Model: strings.TrimSpace(model), Timeout: timeout, client: &http.Client{}}, nil
}

// Provider is "ollama".
func (c *OllamaClient) Provider() string { return ProviderOllama }

// BaseURL is the endpoint it talks to.
func (c *OllamaClient) BaseURL() string { return c.URL }

// ModelName is the model it answers with by default.
func (c *OllamaClient) ModelName() string { return c.Model }

func (c *OllamaClient) request(method, path string, body any, timeout time.Duration) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, ollamaErrorf("cannot encode the request for %s: %v", path, err)
		}
		reader = bytes.NewReader(raw)
	}
	if timeout <= 0 {
		timeout = c.Timeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, c.URL+path, reader)
	if err != nil {
		return nil, ollamaErrorf("cannot reach Ollama at %s: %v", c.URL, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, ollamaErrorf("cannot reach Ollama at %s: %v", c.URL, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, ollamaErrorf("cannot read Ollama's answer to %s: %v", path, err)
	}
	if resp.StatusCode >= 400 {
		detail := strings.TrimSpace(string(raw))
		if len(detail) > 500 {
			detail = detail[:500]
		}
		var doc map[string]any
		if json.Unmarshal(raw, &doc) == nil {
			if message, ok := doc["error"].(string); ok && message != "" {
				detail = message
			}
		}
		return nil, ollamaErrorf("Ollama %s %s failed with HTTP %d: %s", method, path, resp.StatusCode, detail)
	}
	return raw, nil
}

// Models lists the installed models (GET /api/tags).
func (c *OllamaClient) Models() ([]map[string]any, error) {
	raw, err := c.request("GET", "/api/tags", nil, 0)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, ollamaErrorf("Ollama returned invalid JSON for /api/tags: %v", err)
	}
	if doc.Models == nil {
		return nil, ollamaErrorf("unexpected /api/tags response (no 'models' list)")
	}
	return doc.Models, nil
}

// Available reports whether the server answers at all.
func (c *OllamaClient) Available() bool {
	_, err := c.Models()
	return err == nil
}

// Generate runs one non-streaming completion (POST /api/generate): the answer alone.
func (c *OllamaClient) Generate(prompt string, o LLMOptions) (string, error) {
	got, err := c.Complete(prompt, o)
	if err != nil {
		return "", err
	}
	return got.Response, nil
}

// Completion is one answer with the model's thinking beside it ("" for a model that does not think).
type Completion struct {
	Response string `json:"response"`
	Thinking string `json:"thinking"`
}

// Complete is one completion with the model's thinking beside its answer.
//
// o.Think asks a thinking model for its reasoning (Ollama's think field: true,
// false or a level - "low", "medium", "high"; nil leaves the choice to the
// model).  The reasoning comes back as Ollama's thinking field when the server
// separates it, or is cut out of the answer when the model wrote it inline
// between <think> tags (SplitThinking); it is "" for a model that does not think.
func (c *OllamaClient) Complete(prompt string, o LLMOptions) (*Completion, error) {
	model := strings.TrimSpace(o.Model)
	if model == "" {
		model = c.Model
	}
	body := map[string]any{"model": model, "prompt": prompt, "stream": false}
	if o.System != "" {
		body["system"] = o.System
	}
	if o.JSON {
		body["format"] = "json"
	}
	if o.Temperature > 0 {
		body["options"] = map[string]any{"temperature": o.Temperature}
	}
	if o.Think != nil {
		body["think"] = o.Think
	}
	raw, err := c.request("POST", "/api/generate", body, o.Timeout)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Response *string `json:"response"`
		Thinking *string `json:"thinking"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, ollamaErrorf("Ollama returned invalid JSON for /api/generate: %v", err)
	}
	if doc.Response == nil {
		return nil, ollamaErrorf("unexpected /api/generate response (no 'response' text)")
	}
	text, thinking := *doc.Response, ""
	if doc.Thinking != nil {
		thinking = *doc.Thinking
	}
	if strings.TrimSpace(thinking) == "" {
		thinking, text = SplitThinking(text)
	}
	return &Completion{Response: text, Thinking: strings.TrimSpace(thinking)}, nil
}

// ThinkLevels are the reasoning levels a thinking model may be asked for, beside plain on / off.
var ThinkLevels = []string{"low", "medium", "high"}

// ThinkValue is Ollama's think field for a request: nil (not sent), a bool, or
// one of ThinkLevels.  A string is read leniently - "true" / "on" / "yes" and
// "false" / "off" / "no" are the two bools, a level is a level - so the flag can
// come from a command line or a JSON body as it is.
func ThinkValue(think any) (any, error) {
	switch v := think.(type) {
	case nil:
		return nil, nil
	case bool:
		return v, nil
	case float64: // a JSON number
		return ThinkValue(strconv.FormatFloat(v, 'f', -1, 64))
	case int:
		return ThinkValue(strconv.Itoa(v))
	}
	text := strings.ToLower(strings.TrimSpace(fmt.Sprint(think)))
	switch text {
	case "", "none", "default":
		return nil, nil
	case "true", "on", "yes", "1":
		return true, nil
	case "false", "off", "no", "0":
		return false, nil
	}
	for _, level := range ThinkLevels {
		if text == level {
			return level, nil
		}
	}
	return nil, fmt.Errorf("think must be true, false or one of %s (got %q)", strings.Join(ThinkLevels, ", "), fmt.Sprint(think))
}

var thinkOpen = regexp.MustCompile(`(?i)<(think|thinking|reasoning)>`)

// SplitThinking is (thinking, answer) of an answer that wrote its reasoning
// inline between <think> tags.  The first tagged block is the thinking and
// everything else the answer; a block left open is thinking to the end.  A text
// without tags is all answer.
func SplitThinking(text string) (string, string) {
	match := thinkOpen.FindStringSubmatchIndex(text)
	if match == nil {
		return "", strings.TrimSpace(text)
	}
	tag := text[match[2]:match[3]]
	rest := text[match[1]:]
	close := regexp.MustCompile(`(?i)</` + regexp.QuoteMeta(tag) + `>`).FindStringIndex(rest)
	if close == nil {
		return strings.TrimSpace(rest), strings.TrimSpace(text[:match[0]])
	}
	thinking := strings.TrimSpace(rest[:close[0]])
	answer := strings.TrimSpace(text[:match[0]] + rest[close[1]:])
	return thinking, answer
}

const questionsSystem = "You write questions for a small language model to think about. Answer with exactly %d lines and nothing " +
	"else: one short, concrete question per line, plain text, no numbering, no bullets, no quotes, no blank " +
	"lines, no headings and no commentary. Every question must be about the topic requested and answerable " +
	"in a sentence or two."

const thinkSystem = "Think the question through before you answer, step by step, in short plain sentences - say what you " +
	"know, what you are unsure of and ask yourself whether you are right - and then answer in one short " +
	"sentence."

// QuestionsFromPrompt asks the LLM for lines short questions about prompt - what the network will be taught to think about.
func QuestionsFromPrompt(client LLMClient, prompt string, lines int, model string) ([]string, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, fmt.Errorf("prompt must not be empty")
	}
	if lines < 1 {
		return nil, fmt.Errorf("lines must be >= 1")
	}
	user := fmt.Sprintf("Topic / instructions: %s\n\nWrite the %d questions now.", strings.TrimSpace(prompt), lines)
	raw, err := client.Generate(user, LLMOptions{System: fmt.Sprintf(questionsSystem, lines), Model: model, Temperature: 0.9})
	if err != nil {
		return nil, err
	}
	return ParseLines(raw, lines), nil
}

// Thinking is one question the LLM thought about: the question, the thinking
// behind its answer (one line; "" for a model that does not think) and the answer.
type Thinking struct {
	Question string `json:"question"`
	Thinking string `json:"thinking"`
	Answer   string `json:"answer"`
}

// ThoughtsFromPrompt is the LLM's thinking about prompt: lines questions about
// it, and the thinking behind each answer.
//
// Two calls a question: QuestionsFromPrompt writes the questions and Complete
// (think) answers each one with its reasoning.  The thinking and the answer are
// each collapsed to one line; Thinking is "" for a model that does not think,
// which the caller reports rather than trains on.  The thinking is what
// Model.ThinkOn teaches the network as its own thoughts.
func ThoughtsFromPrompt(client *OllamaClient, prompt string, lines int, model string, think any, temperature float64) ([]Thinking, error) {
	questions, err := QuestionsFromPrompt(client, prompt, lines, model)
	if err != nil {
		return nil, err
	}
	out := make([]Thinking, 0, len(questions))
	for _, question := range questions {
		got, err := client.Complete(question, LLMOptions{System: thinkSystem, Model: model, Think: think, Temperature: temperature})
		if err != nil {
			return nil, err
		}
		out = append(out, Thinking{
			Question: question,
			Thinking: strings.Join(strings.Fields(got.Thinking), " "),
			Answer:   strings.Join(strings.Fields(got.Response), " "),
		})
	}
	return out, nil
}

var linePrefix = regexp.MustCompile(`^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*`)

// Chat is one chat turn: the assistant's text.  messages are
// {"role", "content"} maps, as Ollama's /api/chat takes them.
func (c *OllamaClient) Chat(messages []map[string]any, o LLMOptions) (string, error) {
	message, err := c.ChatMessage(messages, o, nil)
	if err != nil {
		return "", err
	}
	content, _ := message["content"].(string)
	return content, nil
}

// ChatMessage is the whole assistant message of one chat turn.
//
// tools are JSON-schema function definitions (Ollama's own tools format, see
// ToolBox.Schemas); a model that supports tool calling answers with
// {"tool_calls": [...]} beside (or instead of) "content".  The message comes
// back as it arrived, with "content" guaranteed to be a string.
func (c *OllamaClient) ChatMessage(messages []map[string]any, o LLMOptions, tools []map[string]any) (map[string]any, error) {
	model := strings.TrimSpace(o.Model)
	if model == "" {
		model = c.Model
	}
	body := map[string]any{"model": model, "messages": messages, "stream": false}
	if o.JSON {
		body["format"] = "json"
	}
	if o.Temperature > 0 {
		body["options"] = map[string]any{"temperature": o.Temperature}
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if o.Think != nil {
		body["think"] = o.Think
	}
	raw, err := c.request("POST", "/api/chat", body, o.Timeout)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Message map[string]any `json:"message"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, ollamaErrorf("Ollama returned invalid JSON for /api/chat: %v", err)
	}
	if doc.Message == nil {
		return nil, ollamaErrorf("unexpected /api/chat response (no message)")
	}
	if _, ok := doc.Message["content"].(string); !ok {
		doc.Message["content"] = ""
	}
	return doc.Message, nil
}

// ParseLines turns an LLM answer into clean lines: numbering, bullets and
// quotes stripped, blank lines, code fences and duplicates dropped.  A limit
// of 0 or less keeps every line.
func ParseLines(text string, limit int) []string {
	lines := []string{}
	seen := map[string]bool{}
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "```") {
			continue
		}
		line = strings.TrimSpace(linePrefix.ReplaceAllString(line, ""))
		line = strings.TrimSpace(strings.Trim(line, "\"'`“”"))
		if line == "" {
			continue
		}
		key := strings.ToLower(line)
		if seen[key] {
			continue
		}
		seen[key] = true
		lines = append(lines, line)
		if limit > 0 && len(lines) >= limit {
			break
		}
	}
	return lines
}

// loadsLenient decodes JSON from an LLM answer, tolerating code fences and prose around the object.
func loadsLenient(raw string) any {
	text := strings.TrimSpace(raw)
	if strings.HasPrefix(text, "```") {
		text = strings.Trim(text, "`")
		if strings.HasPrefix(strings.ToLower(text), "json") {
			text = text[4:]
		}
		text = strings.TrimSpace(text)
	}
	var value any
	if json.Unmarshal([]byte(text), &value) == nil {
		return value
	}
	for _, pair := range [][2]string{{"{", "}"}, {"[", "]"}} {
		start, end := strings.Index(text, pair[0]), strings.LastIndex(text, pair[1])
		if start >= 0 && start < end {
			if json.Unmarshal([]byte(text[start:end+1]), &value) == nil {
				return value
			}
		}
	}
	return nil
}
