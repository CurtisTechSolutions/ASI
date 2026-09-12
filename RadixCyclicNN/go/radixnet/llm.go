package radixnet

// Provider-independent plumbing for the large language models the tutor talks to.
//
// The lessons need three things of a provider: a list of models, a health
// check and one completion.  Two providers offer them - a local model served
// by Ollama (ollama.go) and OpenAI's hosted ChatGPT (chatgpt.go) - so
// everything that asks an LLM a question takes an LLMClient and never looks at
// which one it got.  LLMError is the failure both raise, so IsLLMError means
// "the LLM did not work" whichever provider is in use.

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// Provider names the tutor accepts.
const (
	ProviderOllama  = "ollama"
	ProviderChatGPT = "chatgpt"
	DefaultProvider = ProviderOllama
)

// Providers lists every provider that can teach or mark.
func Providers() []string { return []string{ProviderOllama, ProviderChatGPT} }

// NormaliseProvider maps "", "openai" and "gpt" onto the provider names; anything else is an error.
func NormaliseProvider(name string) (string, error) {
	text := strings.ToLower(strings.TrimSpace(name))
	switch text {
	case "":
		return DefaultProvider, nil
	case ProviderOllama, "local":
		return ProviderOllama, nil
	case ProviderChatGPT, "openai", "gpt", "chat-gpt", "chat_gpt":
		return ProviderChatGPT, nil
	}
	return "", fmt.Errorf("provider must be one of %s (got %q)", strings.Join(Providers(), ", "), name)
}

// LLMOptions are the knobs of one LLM completion, whichever provider answers it
// (Model.Generate has its own GenerateOptions for the network's own texts).
type LLMOptions struct {
	System      string        // the system prompt
	Model       string        // override the client's model
	JSON        bool          // ask for a JSON answer
	Temperature float64       // 0 = the provider's own default
	Timeout     time.Duration // 0 = the client's timeout
}

// LLMClient is what the tutor needs of a provider.
type LLMClient interface {
	Provider() string                                     // "ollama" | "chatgpt"
	BaseURL() string                                      // the endpoint it talks to
	ModelName() string                                    // the model it answers with by default
	Models() ([]map[string]any, error)                    // the models it offers
	Available() bool                                      // does it answer right now (never panics)
	Generate(prompt string, o LLMOptions) (string, error) // one completion
}

// LLMError is a provider that returned something unusable; the transport failures
// keep their own types (OllamaError, ChatGPTError), and IsLLMError matches all three.
type LLMError struct{ Message string }

func (e *LLMError) Error() string { return e.Message }

func llmErrorf(format string, args ...any) error {
	return &LLMError{Message: fmt.Sprintf(format, args...)}
}

// IsLLMError reports whether err came from talking to an LLM provider.
func IsLLMError(err error) bool {
	var generic *LLMError
	return errors.As(err, &generic) || IsOllamaError(err) || IsChatGPTError(err)
}

// DefaultProviderModel is the model a provider teaches with when none is named.
func DefaultProviderModel(provider string) string {
	name, err := NormaliseProvider(provider)
	if err != nil || name == ProviderOllama {
		return DefaultTutorModel()
	}
	return DefaultChatGPTModel()
}

// NewLLMClient builds a client for a provider; empty url / model / timeout take that provider's defaults.
func NewLLMClient(provider, url, model string, timeout time.Duration) (LLMClient, error) {
	name, err := NormaliseProvider(provider)
	if err != nil {
		return nil, err
	}
	if name == ProviderChatGPT {
		return NewChatGPTClient(url, model, timeout)
	}
	return NewOllamaClient(url, model, timeout)
}

// ProviderOf is the provider a client belongs to (the default for a nil one).
func ProviderOf(client LLMClient) string {
	if client == nil {
		return DefaultProvider
	}
	return client.Provider()
}
