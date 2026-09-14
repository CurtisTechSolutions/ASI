package radixnet

// ChatGPT: the hosted alternative to a local model, over OpenAI's API
// (https://platform.openai.com).  The same three calls the tutor needs - the
// models a key may use (GET /models), a health check and one non-streaming
// completion (POST /chat/completions) - so a ChatGPTClient stands in for an
// OllamaClient wherever an LLMClient is taken.
//
// Configuration comes from the environment, the way the OpenAI tools expect
// it: OPENAI_API_KEY (or OPENAI_API_KEY_FILE, a file holding it, for Docker
// secrets), OPENAI_BASE_URL (any OpenAI-compatible server works), and
// RADIXNET_OPENAI_MODEL for the default model.  The key is read per request
// and never stored in a config, a record or a log line; unlike the local
// provider every prompt is sent to OpenAI and billed.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

// DefaultChatGPTTimeout is how long one answer may take.
const DefaultChatGPTTimeout = 120 * time.Second

// DefaultChatGPTModelName is used when neither the caller nor the environment names a model.
const DefaultChatGPTModelName = "gpt-4o-mini"

// OpenAIURL is the hosted API; $OPENAI_BASE_URL overrides it.
const OpenAIURL = "https://api.openai.com/v1"

// ChatGPTError is an OpenAI API that cannot be reached, refuses the request, or answers something unusable.
type ChatGPTError struct {
	Message string
	Status  int
	Code    string
	Param   string
}

func (e *ChatGPTError) Error() string { return e.Message }

func chatgptErrorf(format string, args ...any) error {
	return &ChatGPTError{Message: fmt.Sprintf(format, args...)}
}

// IsChatGPTError reports whether err came from talking to the OpenAI API.
func IsChatGPTError(err error) bool {
	var target *ChatGPTError
	return errors.As(err, &target)
}

var localHosts = map[string]bool{
	"localhost": true, "127.0.0.1": true, "::1": true, "0.0.0.0": true, "host.docker.internal": true,
}

func isLocalHost(host string) bool {
	host = strings.ToLower(host)
	return localHosts[host] || strings.HasSuffix(host, ".local")
}

func envFlag(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// NormaliseChatGPTURL turns "host" or a full URL into a base URL without a trailing
// slash ("/v1" added when there is no path).  A key must not travel unencrypted to a
// remote host, so plain http:// is refused there unless RADIXNET_OPENAI_ALLOW_INSECURE is set.
func NormaliseChatGPTURL(raw string) (string, error) {
	text := strings.TrimSpace(raw)
	if text == "" {
		return "", fmt.Errorf("the OpenAI base URL is empty")
	}
	if !strings.Contains(text, "://") {
		text = "https://" + text
	}
	parsed, err := url.Parse(text)
	if err != nil {
		return "", fmt.Errorf("the OpenAI base URL is not a URL (got %q): %v", raw, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("the OpenAI base URL must be http:// or https:// (got %q)", raw)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("the OpenAI base URL has no host (got %q)", raw)
	}
	if parsed.Scheme == "http" && !isLocalHost(parsed.Hostname()) && !envFlag("RADIXNET_OPENAI_ALLOW_INSECURE") {
		return "", fmt.Errorf("refusing to send the API key unencrypted to %q: use https://, a local server, "+
			"or set RADIXNET_OPENAI_ALLOW_INSECURE=1", parsed.Hostname())
	}
	path := strings.TrimRight(parsed.Path, "/")
	if path == "" {
		path = "/v1"
	}
	return parsed.Scheme + "://" + parsed.Host + path, nil
}

// DefaultChatGPTURL is $OPENAI_BASE_URL (or $OPENAI_API_BASE), else the hosted API.
func DefaultChatGPTURL() string {
	configured := strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	if configured == "" {
		configured = strings.TrimSpace(os.Getenv("OPENAI_API_BASE"))
	}
	if configured == "" {
		return OpenAIURL
	}
	if normalised, err := NormaliseChatGPTURL(configured); err == nil {
		return normalised
	}
	return OpenAIURL
}

// DefaultChatGPTModel is $RADIXNET_OPENAI_MODEL, else gpt-4o-mini.
func DefaultChatGPTModel() string {
	if name := strings.TrimSpace(os.Getenv("RADIXNET_OPENAI_MODEL")); name != "" {
		return name
	}
	return DefaultChatGPTModelName
}

// ChatGPTAPIKey is $OPENAI_API_KEY, else the content of the file named by $OPENAI_API_KEY_FILE.
func ChatGPTAPIKey() string {
	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" {
		return key
	}
	if path := strings.TrimSpace(os.Getenv("OPENAI_API_KEY_FILE")); path != "" {
		if raw, err := os.ReadFile(path); err == nil {
			return strings.TrimSpace(string(raw))
		}
	}
	return ""
}

// ChatGPTConfigured reports whether a key is available (its value is never returned or logged).
func ChatGPTConfigured() bool { return ChatGPTAPIKey() != "" }

// ChatGPTClient talks to one OpenAI-compatible endpoint.
type ChatGPTClient struct {
	URL     string
	Model   string
	Timeout time.Duration
	apiKey  string // empty: read from the environment per request
	client  *http.Client
}

// NewChatGPTClient normalises the URL and fills in the defaults for an empty url / model / timeout.
func NewChatGPTClient(rawURL, model string, timeout time.Duration) (*ChatGPTClient, error) {
	if strings.TrimSpace(rawURL) == "" {
		rawURL = DefaultChatGPTURL()
	}
	normalised, err := NormaliseChatGPTURL(rawURL)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(model) == "" {
		model = DefaultChatGPTModel()
	}
	if timeout <= 0 {
		timeout = DefaultChatGPTTimeout
	}
	return &ChatGPTClient{URL: normalised, Model: strings.TrimSpace(model), Timeout: timeout, client: &http.Client{}}, nil
}

// Provider is "chatgpt".
func (c *ChatGPTClient) Provider() string { return ProviderChatGPT }

// BaseURL is the endpoint it talks to.
func (c *ChatGPTClient) BaseURL() string { return c.URL }

// ModelName is the model it answers with by default.
func (c *ChatGPTClient) ModelName() string { return c.Model }

// Configured reports whether a key is available for this client.
func (c *ChatGPTClient) Configured() bool { return c.key() != "" }

// String never shows the key.
func (c *ChatGPTClient) String() string { return fmt.Sprintf("ChatGPTClient(%s, %s)", c.URL, c.Model) }

func (c *ChatGPTClient) key() string {
	if strings.TrimSpace(c.apiKey) != "" {
		return strings.TrimSpace(c.apiKey)
	}
	return ChatGPTAPIKey()
}

func (c *ChatGPTClient) request(method, path string, body any, timeout time.Duration) ([]byte, error) {
	key := c.key()
	if key == "" {
		return nil, chatgptErrorf("no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) in the " +
			"environment of the process that talks to ChatGPT")
	}
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, chatgptErrorf("cannot encode the request for %s: %v", path, err)
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
		return nil, chatgptErrorf("cannot reach the OpenAI API at %s: %v", c.URL, err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if org := strings.TrimSpace(os.Getenv("OPENAI_ORG_ID")); org != "" {
		req.Header.Set("OpenAI-Organization", org)
	}
	if project := strings.TrimSpace(os.Getenv("OPENAI_PROJECT_ID")); project != "" {
		req.Header.Set("OpenAI-Project", project)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, chatgptErrorf("cannot reach the OpenAI API at %s: %v", c.URL, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, chatgptErrorf("cannot read the OpenAI answer to %s: %v", path, err)
	}
	if resp.StatusCode >= 400 {
		return nil, c.httpError(method, path, resp.StatusCode, raw)
	}
	return raw, nil
}

func (c *ChatGPTClient) httpError(method, path string, status int, raw []byte) error {
	detail := strings.TrimSpace(string(raw))
	if len(detail) > 1000 {
		detail = detail[:1000]
	}
	code, param := "", ""
	var doc struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
			Param   string `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &doc) == nil && doc.Error.Message != "" {
		detail, code, param = doc.Error.Message, doc.Error.Code, doc.Error.Param
	}
	hint := ""
	switch status {
	case 401:
		hint = " (check OPENAI_API_KEY)"
	case 403:
		hint = " (the key may not be allowed to use this model)"
	case 404:
		hint = fmt.Sprintf(" (does the key have access to model %q?)", c.Model)
	case 429:
		hint = " (rate limit or exhausted quota)"
	}
	return &ChatGPTError{
		Message: fmt.Sprintf("OpenAI %s %s failed with HTTP %d: %s%s", method, path, status, detail, hint),
		Status:  status, Code: code, Param: param,
	}
}

// Models lists the models the key may use (GET /models), as {"name", "id", "owned_by", "created"}.
func (c *ChatGPTClient) Models() ([]map[string]any, error) {
	raw, err := c.request("GET", "/models", nil, 0)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, chatgptErrorf("the OpenAI API returned invalid JSON for /models: %v", err)
	}
	if doc.Data == nil {
		return nil, chatgptErrorf("unexpected /models response (no 'data' list)")
	}
	models := make([]map[string]any, 0, len(doc.Data))
	for _, item := range doc.Data {
		id, _ := item["id"].(string)
		if id == "" {
			continue
		}
		models = append(models, map[string]any{
			"name": id, "id": id, "owned_by": item["owned_by"], "created": item["created"],
		})
	}
	return models, nil
}

// Available reports whether a key is configured and the API answers.
func (c *ChatGPTClient) Available() bool {
	_, err := c.Models()
	return err == nil
}

// optionalFields are the body fields worth dropping and retrying when a model rejects them.
var optionalFields = []string{"temperature", "response_format"}

var (
	rejectedPattern = regexp.MustCompile(`(?i)unsupported|unrecognized|not supported|does not support|unknown parameter`)
	quotedName      = regexp.MustCompile(`['"` + "`" + `]([A-Za-z_][A-Za-z0-9_]*)['"` + "`" + `]`)
	argumentName    = regexp.MustCompile(`(?i)argument(?: supplied)?:\s*([A-Za-z_][A-Za-z0-9_]*)`)
)

// rejectedField names the optional body field an error blames, when dropping it is worth one retry.
func rejectedField(err error, body map[string]any) string {
	var failure *ChatGPTError
	if !errors.As(err, &failure) || failure.Status != 400 || !rejectedPattern.MatchString(failure.Message) {
		return ""
	}
	names := []string{}
	if failure.Param != "" {
		names = append(names, failure.Param)
	}
	for _, match := range argumentName.FindAllStringSubmatch(failure.Message, -1) {
		names = append(names, match[1])
	}
	for _, match := range quotedName.FindAllStringSubmatch(failure.Message, -1) {
		names = append(names, match[1])
	}
	for _, name := range names {
		if _, present := body[name]; !present {
			continue
		}
		for _, field := range optionalFields {
			if name == field {
				return name
			}
		}
	}
	return ""
}

// Generate runs one non-streaming completion (POST /chat/completions).  A model
// that rejects an optional field (the reasoning models refuse temperature, older
// ones response_format) is asked again without it instead of failing.
func (c *ChatGPTClient) Generate(prompt string, o LLMOptions) (string, error) {
	model := strings.TrimSpace(o.Model)
	if model == "" {
		model = c.Model
	}
	messages := []map[string]any{}
	if o.System != "" {
		messages = append(messages, map[string]any{"role": "system", "content": o.System})
	}
	messages = append(messages, map[string]any{"role": "user", "content": prompt})
	body := map[string]any{"model": model, "messages": messages}
	if o.Temperature > 0 {
		body["temperature"] = o.Temperature
	}
	if o.JSON {
		body["response_format"] = map[string]any{"type": "json_object"}
	}
	for {
		raw, err := c.request("POST", "/chat/completions", body, o.Timeout)
		if err != nil {
			if field := rejectedField(err, body); field != "" {
				delete(body, field) // this model does not take that field: ask again without it
				continue
			}
			return "", err
		}
		return chatCompletionText(raw)
	}
}

// chatCompletionText is the assistant text of a chat-completions response.
func chatCompletionText(raw []byte) (string, error) {
	var doc struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content any    `json:"content"`
				Refusal string `json:"refusal"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return "", chatgptErrorf("the OpenAI API returned invalid JSON for /chat/completions: %v", err)
	}
	if len(doc.Choices) == 0 {
		return "", chatgptErrorf("unexpected chat completion (no 'choices')")
	}
	choice := doc.Choices[0]
	switch content := choice.Message.Content.(type) {
	case string:
		if strings.TrimSpace(content) != "" {
			return content, nil
		}
	case []any: // content parts instead of one string
		var text strings.Builder
		for _, part := range content {
			if item, ok := part.(map[string]any); ok {
				if value, ok := item["text"].(string); ok {
					text.WriteString(value)
				}
			}
		}
		if strings.TrimSpace(text.String()) != "" {
			return text.String(), nil
		}
	}
	if strings.TrimSpace(choice.Message.Refusal) != "" {
		return "", chatgptErrorf("the model refused to answer: %s", strings.TrimSpace(choice.Message.Refusal))
	}
	return "", chatgptErrorf("the model returned an empty answer (finish_reason=%q)", choice.FinishReason)
}
