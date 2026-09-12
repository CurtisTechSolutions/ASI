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

// OllamaClient talks to one Ollama server.
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

// OllamaGenerateOptions are the knobs of one completion.
type OllamaGenerateOptions struct {
	System      string        // the system prompt
	Model       string        // override the client's model
	JSON        bool          // ask for a JSON answer (format: json)
	Temperature float64       // 0 = Ollama's own default
	Timeout     time.Duration // 0 = the client's timeout
}

// Generate runs one non-streaming completion (POST /api/generate).
func (c *OllamaClient) Generate(prompt string, o OllamaGenerateOptions) (string, error) {
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
	raw, err := c.request("POST", "/api/generate", body, o.Timeout)
	if err != nil {
		return "", err
	}
	var doc struct {
		Response *string `json:"response"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return "", ollamaErrorf("Ollama returned invalid JSON for /api/generate: %v", err)
	}
	if doc.Response == nil {
		return "", ollamaErrorf("unexpected /api/generate response (no 'response' text)")
	}
	return *doc.Response, nil
}

var linePrefix = regexp.MustCompile(`^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*`)

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
