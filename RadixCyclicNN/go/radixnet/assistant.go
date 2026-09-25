package radixnet

// The model in today's format: messages in, an assistant message out -
// thinking, text, tool calls, streamed.  The Go twin of Python's
// radixnet/assistant.py, and held to it by TestGoAssistantParity.
//
// Every language model is talked to the same way now: a list of {role, content}
// messages goes in, an assistant message comes back, and while it is being
// written the client sees it arrive - the model's thinking first, then the
// answer, chunk by chunk, over server-sent events.  Two dialects cover every
// client there is: OpenAI's Chat Completions (POST /v1/chat/completions,
// reasoning_content, data: [DONE]) and Anthropic's Messages (POST /v1/messages,
// typed content blocks, one event per block).
//
// Nothing here is a prompt trick.  A reply is what Model.Reply says next after
// the last line of the conversation - exactly as in the Converse and Chat
// tabs - and the format is a rendering of what that search already produces:
// the thinking is the search's own trace, line by line as it happens (never
// prose the model did not produce); the text streams one chunk per node of the
// walk; a <tool> line the model writes comes back as a tool call when the
// request offered that tool; the stop reason says how the walk ended; usage
// counts units of the model's encoding - a character, or a word.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"
)

// The two dialects.
const (
	FormatOpenAI    = "openai"
	FormatAnthropic = "anthropic"
)

// How a reply ended, in the Messages vocabulary; FinishReason is the same in Chat Completions'.
const (
	StopEndTurn      = "end_turn"
	StopMaxTokens    = "max_tokens"
	StopStopSequence = "stop_sequence"
	StopToolUse      = "tool_use"
	StopRefusal      = "refusal"
)

// ModelIDPrefix: a model's id is its kind behind it (radixnet-count).
const ModelIDPrefix = "radixnet-"

// DefaultMaxTokens is what a reply may add when the request sets no max_tokens: the dialogue's max_length.
const DefaultMaxTokens = 60

// MaxChoices is how many alternative replies (n) one request may ask for.
const MaxChoices = 8

// FinishReason is a stop reason in Chat Completions' words.
func FinishReason(stop string) string {
	switch stop {
	case StopMaxTokens:
		return "length"
	case StopToolUse:
		return "tool_calls"
	case StopRefusal:
		return "content_filter"
	}
	return "stop"
}

// AskError is a request that cannot be answered as it stands; Param names the field when one is to blame.
type AskError struct {
	Message string
	Param   string
}

func (e *AskError) Error() string { return e.Message }

func askErrorf(param, format string, args ...any) *AskError {
	return &AskError{Message: fmt.Sprintf(format, args...), Param: param}
}

// AskMessage is one line of the conversation as the model reads it: an
// assistant's tool calls and a tool's results are rendered into the agent's
// transcript format, which is the shape the network was trained on.
type AskMessage struct {
	Role string
	Text string
}

// OfferedTool is a tool the request offered: a written call to it comes back as a tool call.
type OfferedTool struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// Ask is a request in either dialect, normalised: what to answer, and how the search should look for it.
type Ask struct {
	Messages     []AskMessage
	System       string
	Model        string
	MaxTokens    int
	Temperature  float64
	Stop         []string
	N            int
	Stream       bool
	IncludeUsage bool
	Tools        []OfferedTool
	Thinking     bool
	// the dialogue's own dials, named as POST /api/converse names them
	Mode             string
	Context          int
	K                int
	Beam             int
	StepPenalty      float64
	Explore          int
	AvoidRepeats     bool
	AvoidWordRepeats bool
	Learn            bool
	Guard            bool
	Seed             *int64
}

// DefaultAsk mirrors the Python defaults.
func DefaultAsk() Ask {
	return Ask{MaxTokens: DefaultMaxTokens, Temperature: 1, N: 1, Thinking: true, Mode: "beam", Context: 12, K: 5,
		Explore: Explore, AvoidRepeats: true, AvoidWordRepeats: true, Learn: true, Guard: true}
}

// Validate is why the request cannot be answered, if it cannot.
func (a *Ask) Validate() error {
	if len(a.Messages) == 0 {
		return askErrorf("messages", "messages must hold at least one user or assistant message")
	}
	for i, m := range a.Messages {
		if m.Role != "user" && m.Role != "assistant" && m.Role != "tool" {
			return askErrorf("messages", "messages[%d].role must be user, assistant or tool (got %s)", i, PythonRepr(m.Role))
		}
	}
	if a.MaxTokens < 1 {
		return askErrorf("max_tokens", "max_tokens must be >= 1")
	}
	if a.Temperature < 0 {
		return askErrorf("temperature", "temperature must be >= 0")
	}
	if a.N < 1 || a.N > MaxChoices {
		return askErrorf("n", "n must be between 1 and %d", MaxChoices)
	}
	if a.Mode != "beam" && a.Mode != "sample" {
		return askErrorf("mode", "mode must be beam or sample")
	}
	if a.Context < 0 {
		return askErrorf("context", "context must be >= 0")
	}
	if a.K < 1 {
		return askErrorf("k", "k must be >= 1")
	}
	if a.Beam < 0 {
		return askErrorf("beam", "beam must be >= 1")
	}
	if a.StepPenalty < 0 {
		return askErrorf("step_penalty", "step_penalty must be >= 0")
	}
	if a.Explore < 0 {
		return askErrorf("explore", "explore must be >= 0")
	}
	for i, seq := range a.Stop {
		if seq == "" {
			return askErrorf("stop", "stop[%d] must be a non-empty string", i)
		}
	}
	seen := map[string]bool{}
	for i, tool := range a.Tools {
		if !wholeRe.MatchString(tool.Name) {
			return askErrorf("tools", "tools[%d].name must be an identifier (got %s)", i, PythonRepr(tool.Name))
		}
		if seen[tool.Name] {
			return askErrorf("tools", "tools[%d].name %s is offered twice", i, PythonRepr(tool.Name))
		}
		seen[tool.Name] = true
	}
	return nil
}

// Previous is the line the reply continues: the last message.
func (a *Ask) Previous() string {
	if len(a.Messages) == 0 {
		return ""
	}
	return a.Messages[len(a.Messages)-1].Text
}

// Prefill is whether the last message is the assistant's own: the reply then
// continues it and returns only what it adds.
func (a *Ask) Prefill() bool {
	return len(a.Messages) > 0 && a.Messages[len(a.Messages)-1].Role == "assistant"
}

func askTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64, int, int64:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	}
	return "object"
}

// fieldInt reads an integer field (absent or null: the default); *def == nil means none.
func fieldInt(body map[string]any, name string, def *int, minimum *int) (*int, error) {
	value, ok := body[name]
	if !ok || value == nil {
		return def, nil
	}
	var n int
	switch v := value.(type) {
	case float64:
		if v != math.Trunc(v) || math.IsInf(v, 0) {
			return nil, askErrorf(name, "%s must be an integer (got number)", name)
		}
		n = int(v)
	case int:
		n = v
	case int64:
		n = int(v)
	default:
		return nil, askErrorf(name, "%s must be an integer (got %s)", name, askTypeName(value))
	}
	if minimum != nil && n < *minimum {
		return nil, askErrorf(name, "%s must be >= %d (got %d)", name, *minimum, n)
	}
	return &n, nil
}

func fieldNum(body map[string]any, name string, def float64, minimum *float64) (float64, error) {
	value, ok := body[name]
	if !ok || value == nil {
		return def, nil
	}
	var n float64
	switch v := value.(type) {
	case float64:
		n = v
	case int:
		n = float64(v)
	case int64:
		n = float64(v)
	default:
		return 0, askErrorf(name, "%s must be a number (got %s)", name, askTypeName(value))
	}
	if math.IsNaN(n) || math.IsInf(n, 0) {
		return 0, askErrorf(name, "%s must be a number (got number)", name)
	}
	if minimum != nil && n < *minimum {
		return 0, askErrorf(name, "%s must be >= %s (got %s)", name, pyNumber(*minimum), pyNumber(n))
	}
	return n, nil
}

// pyNumber is a number as Python prints it in a message: -1, 0.5.
func pyNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return fmt.Sprintf("%d", int64(v))
	}
	return pythonFloat(v)
}

func fieldBool(body map[string]any, name string, def bool) (bool, error) {
	value, ok := body[name]
	if !ok || value == nil {
		return def, nil
	}
	b, ok := value.(bool)
	if !ok {
		return false, askErrorf(name, "%s must be a boolean (got %s)", name, askTypeName(value))
	}
	return b, nil
}

func fieldStr(body map[string]any, name, def string) (string, error) {
	value, ok := body[name]
	if !ok || value == nil {
		return def, nil
	}
	s, ok := value.(string)
	if !ok {
		return "", askErrorf(name, "%s must be a string (got %s)", name, askTypeName(value))
	}
	return s, nil
}

// dials reads the dialogue's own dials from either dialect's body.
func dials(body map[string]any, ask *Ask) error {
	mode, err := fieldStr(body, "mode", "beam")
	if err != nil {
		return err
	}
	mode = strings.ToLower(mode)
	if mode == "" || mode == "dijkstra" {
		mode = "beam"
	}
	ask.Mode = mode
	zero, one := 0, 1
	if v, err := fieldInt(body, "context", &ask.Context, &zero); err != nil {
		return err
	} else {
		ask.Context = *v
	}
	if v, err := fieldInt(body, "k", &ask.K, &one); err != nil {
		return err
	} else {
		ask.K = *v
	}
	if v, err := fieldInt(body, "beam", nil, &one); err != nil {
		return err
	} else if v != nil {
		ask.Beam = *v
	}
	if ask.StepPenalty, err = fieldNum(body, "step_penalty", 0, floatp(0)); err != nil {
		return err
	}
	if v, err := fieldInt(body, "explore", &ask.Explore, &zero); err != nil {
		return err
	} else {
		ask.Explore = *v
	}
	if ask.AvoidRepeats, err = fieldBool(body, "avoid_repeats", true); err != nil {
		return err
	}
	if ask.AvoidWordRepeats, err = fieldBool(body, "avoid_word_repeats", true); err != nil {
		return err
	}
	if ask.Learn, err = fieldBool(body, "learn", true); err != nil {
		return err
	}
	if ask.Guard, err = fieldBool(body, "guard", true); err != nil {
		return err
	}
	seed, err := fieldInt(body, "seed", nil, nil)
	if err != nil {
		return err
	}
	if seed != nil {
		s := int64(*seed)
		ask.Seed = &s
	}
	return nil
}

func floatp(v float64) *float64 { return &v }

// textParts is the text of a message's content: a string, or a list of typed parts joined by newlines.
func textParts(content any, where, dialect string) (string, error) {
	switch c := content.(type) {
	case nil:
		return "", nil
	case string:
		return c, nil
	case []any:
		parts := []string{}
		for i, part := range c {
			if s, ok := part.(string); ok {
				parts = append(parts, s)
				continue
			}
			obj, ok := part.(map[string]any)
			if !ok {
				return "", askErrorf("messages", "%s.content[%d] must be an object with a type", where, i)
			}
			kind, _ := obj["type"].(string)
			switch {
			case kind == "text" || kind == "refusal":
				key := "text"
				if kind == "refusal" {
					key = "refusal"
				}
				text, ok := obj[key].(string)
				if !ok {
					return "", askErrorf("messages", "%s.content[%d].text must be a string", where, i)
				}
				parts = append(parts, text)
			case kind == "thinking" || kind == "redacted_thinking":
				continue // what the model thought is not what it said
			case kind == "tool_use" && dialect == FormatAnthropic:
				name, _ := obj["name"].(string)
				if name == "" {
					return "", askErrorf("messages", "%s.content[%d].name must be the tool's name", where, i)
				}
				var input map[string]any
				switch in := obj["input"].(type) {
				case nil:
					input = map[string]any{}
				case map[string]any:
					input = in
				default:
					return "", askErrorf("messages", "%s.content[%d].input must be an object", where, i)
				}
				parts = append(parts, strings.TrimRight(CallText(name, input), "\n"))
			case kind == "tool_result" && dialect == FormatAnthropic:
				inner, err := textParts(obj["content"], fmt.Sprintf("%s.content[%d]", where, i), dialect)
				if err != nil {
					return "", err
				}
				parts = append(parts, strings.TrimRight(ResultText(inner, DefaultObservationChars), "\n"))
			default:
				what := askTypeName(obj["type"])
				if s, ok := obj["type"].(string); ok {
					what = PythonRepr(s)
				}
				return "", askErrorf("messages", "%s.content[%d]: content of type %s is not supported; this model reads text", where, i, what)
			}
		}
		return strings.Join(parts, "\n"), nil
	}
	return "", askErrorf("messages", "%s.content must be a string or a list of content parts", where)
}

// toolCallsText renders a Chat Completions assistant message's tool_calls as the <tool> lines the model writes.
func toolCallsText(calls any, where string) (string, error) {
	if calls == nil {
		return "", nil
	}
	items, ok := calls.([]any)
	if !ok {
		return "", askErrorf("messages", "%s.tool_calls must be a list", where)
	}
	lines := []string{}
	for i, item := range items {
		call, _ := item.(map[string]any)
		function, _ := call["function"].(map[string]any)
		name, _ := function["name"].(string)
		if function == nil || name == "" {
			return "", askErrorf("messages", "%s.tool_calls[%d] must be a function call with a name", where, i)
		}
		arguments := map[string]any{}
		switch raw := function["arguments"].(type) {
		case string:
			if strings.TrimSpace(raw) != "" {
				var parsed any
				if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
					return "", askErrorf("messages", "%s.tool_calls[%d].function.arguments is not JSON: %v", where, i, err)
				}
				obj, ok := parsed.(map[string]any)
				if !ok {
					return "", askErrorf("messages", "%s.tool_calls[%d].function.arguments must encode an object", where, i)
				}
				arguments = obj
			}
		case map[string]any:
			arguments = raw
		}
		lines = append(lines, strings.TrimRight(CallText(name, arguments), "\n"))
	}
	return strings.Join(lines, "\n"), nil
}

// offeredTools reads the tools offered, in one shape.
func offeredTools(items any, dialect string) ([]OfferedTool, error) {
	if items == nil {
		return nil, nil
	}
	list, ok := items.([]any)
	if !ok {
		return nil, askErrorf("tools", "tools must be a list")
	}
	out := []OfferedTool{}
	for i, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, askErrorf("tools", "tools[%d] must be an object", i)
		}
		spec := obj
		var schema any
		if dialect == FormatOpenAI {
			if kind, present := obj["type"]; present && kind != "function" {
				return nil, askErrorf("tools", "tools[%d].type must be \"function\"", i)
			}
			if function, present := obj["function"]; present {
				spec, ok = function.(map[string]any)
				if !ok {
					return nil, askErrorf("tools", "tools[%d].function must be an object", i)
				}
			}
			schema = spec["parameters"]
		} else {
			schema = obj["input_schema"]
		}
		name, _ := spec["name"].(string)
		if name == "" {
			return nil, askErrorf("tools", "tools[%d] has no name", i)
		}
		parameters := map[string]any{}
		switch s := schema.(type) {
		case nil:
		case map[string]any:
			parameters = s
		default:
			return nil, askErrorf("tools", "tools[%d]: the parameters must be a JSON schema object", i)
		}
		description := ""
		switch d := spec["description"].(type) {
		case string:
			description = d
		case nil:
		default:
			description = fmt.Sprint(d)
		}
		out = append(out, OfferedTool{Name: name, Description: description, Parameters: parameters})
	}
	return out, nil
}

// stopSequences reads the stop sequences: a list of non-empty strings (Chat Completions also takes one string).
func stopSequences(value any, name string, oneString bool) ([]string, error) {
	var items []string
	switch v := value.(type) {
	case nil:
		return nil, nil
	case string:
		if !oneString {
			return nil, askErrorf(name, "%s must be a list of strings", name)
		}
		items = []string{v}
	case []any:
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				what := "a string or a list of strings"
				if !oneString {
					what = "a list of strings"
				}
				return nil, askErrorf(name, "%s must be %s", name, what)
			}
			items = append(items, s)
		}
	default:
		what := "a string or a list of strings"
		if !oneString {
			what = "a list of strings"
		}
		return nil, askErrorf(name, "%s must be %s", name, what)
	}
	for _, s := range items {
		if s == "" {
			return nil, askErrorf(name, "%s must not hold an empty sequence", name)
		}
	}
	return items, nil
}

// askMessages reads the conversation and the system prompt out of a messages list.
func askMessages(items any, dialect string) ([]AskMessage, string, error) {
	list, ok := items.([]any)
	if !ok {
		return nil, "", askErrorf("messages", "messages must be a list of {role, content} objects")
	}
	system := []string{}
	out := []AskMessage{}
	for i, item := range list {
		where := fmt.Sprintf("messages[%d]", i)
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, "", askErrorf("messages", "%s must be an object with a role and content", where)
		}
		role, ok := obj["role"].(string)
		if !ok {
			return nil, "", askErrorf("messages", "%s.role must be a string", where)
		}
		role = strings.ToLower(role)
		switch role {
		case "developer":
			role = "system"
		case "function":
			role = "tool"
		}
		if dialect == FormatAnthropic && role != "user" && role != "assistant" {
			return nil, "", askErrorf("messages", "%s.role must be user or assistant (got %s)", where, PythonRepr(role))
		}
		if role != "system" && role != "user" && role != "assistant" && role != "tool" {
			return nil, "", askErrorf("messages", "%s.role must be one of system, user, assistant, tool (got %s)", where, PythonRepr(role))
		}
		text, err := textParts(obj["content"], where, dialect)
		if err != nil {
			return nil, "", err
		}
		if role == "assistant" && dialect == FormatOpenAI {
			calls, err := toolCallsText(obj["tool_calls"], where)
			if err != nil {
				return nil, "", err
			}
			parts := []string{}
			for _, p := range []string{text, calls} {
				if p != "" {
					parts = append(parts, p)
				}
			}
			text = strings.Join(parts, "\n")
		}
		if role == "tool" && dialect == FormatOpenAI {
			text = strings.TrimRight(ResultText(text, DefaultObservationChars), "\n")
		}
		if role == "system" {
			if strings.TrimSpace(text) != "" {
				system = append(system, text)
			}
			continue
		}
		if strings.TrimSpace(text) == "" {
			continue // an empty line says nothing and is not heard
		}
		out = append(out, AskMessage{Role: role, Text: text})
	}
	return out, strings.Join(system, "\n"), nil
}

// ParseOpenAI reads a Chat Completions request (Python's parse_openai).
func ParseOpenAI(body map[string]any) (*Ask, error) {
	if body == nil {
		return nil, askErrorf("", "the request body must be a JSON object")
	}
	ask := DefaultAsk()
	var err error
	if ask.Model, err = fieldStr(body, "model", ""); err != nil {
		return nil, err
	}
	items, present := body["messages"]
	if !present {
		return nil, askErrorf("messages", "messages is required")
	}
	if ask.Messages, ask.System, err = askMessages(items, FormatOpenAI); err != nil {
		return nil, err
	}
	one := 1
	limit, err := fieldInt(body, "max_completion_tokens", nil, &one)
	if err != nil {
		return nil, err
	}
	if limit == nil {
		if limit, err = fieldInt(body, "max_tokens", nil, &one); err != nil {
			return nil, err
		}
	}
	if limit != nil {
		ask.MaxTokens = *limit
	}
	if ask.Temperature, err = fieldNum(body, "temperature", 1, floatp(0)); err != nil {
		return nil, err
	}
	if ask.Stop, err = stopSequences(body["stop"], "stop", true); err != nil {
		return nil, err
	}
	n, err := fieldInt(body, "n", &one, &one)
	if err != nil {
		return nil, err
	}
	ask.N = *n
	if ask.Stream, err = fieldBool(body, "stream", false); err != nil {
		return nil, err
	}
	switch options := body["stream_options"].(type) {
	case nil:
	case map[string]any:
		if ask.IncludeUsage, err = fieldBool(options, "include_usage", false); err != nil {
			return nil, err
		}
	default:
		return nil, askErrorf("stream_options", "stream_options must be an object")
	}
	if ask.Tools, err = offeredTools(body["tools"], FormatOpenAI); err != nil {
		return nil, err
	}
	if choice, _ := body["tool_choice"].(string); choice == "none" {
		ask.Tools = nil
	}
	thinking, err := fieldBool(body, "thinking", true)
	if err != nil {
		return nil, err
	}
	effort, err := fieldStr(body, "reasoning_effort", "")
	if err != nil {
		return nil, err
	}
	ask.Thinking = thinking && effort != "none"
	if err := dials(body, &ask); err != nil {
		return nil, err
	}
	if err := ask.Validate(); err != nil {
		return nil, err
	}
	return &ask, nil
}

// ParseAnthropic reads a Messages request (Python's parse_anthropic).
func ParseAnthropic(body map[string]any) (*Ask, error) {
	if body == nil {
		return nil, askErrorf("", "the request body must be a JSON object")
	}
	ask := DefaultAsk()
	var err error
	if ask.Model, err = fieldStr(body, "model", ""); err != nil {
		return nil, err
	}
	items, present := body["messages"]
	if !present {
		return nil, askErrorf("messages", "messages is required")
	}
	if ask.Messages, _, err = askMessages(items, FormatAnthropic); err != nil {
		return nil, err
	}
	if ask.System, err = textParts(body["system"], "system", FormatAnthropic); err != nil {
		return nil, err
	}
	one := 1
	limit, err := fieldInt(body, "max_tokens", nil, &one)
	if err != nil {
		return nil, err
	}
	if limit != nil {
		ask.MaxTokens = *limit
	}
	if ask.Temperature, err = fieldNum(body, "temperature", 1, floatp(0)); err != nil {
		return nil, err
	}
	if ask.Stop, err = stopSequences(body["stop_sequences"], "stop_sequences", false); err != nil {
		return nil, err
	}
	if ask.Stream, err = fieldBool(body, "stream", false); err != nil {
		return nil, err
	}
	if ask.Tools, err = offeredTools(body["tools"], FormatAnthropic); err != nil {
		return nil, err
	}
	if choice, ok := body["tool_choice"].(map[string]any); ok && choice["type"] == "none" {
		ask.Tools = nil
	}
	switch thinking := body["thinking"].(type) {
	case nil:
	case bool:
		ask.Thinking = thinking
	case map[string]any:
		kind, _ := thinking["type"].(string)
		if kind != "enabled" && kind != "disabled" && kind != "adaptive" {
			return nil, askErrorf("thinking", "thinking must be {\"type\": \"enabled\"} or {\"type\": \"disabled\"}")
		}
		ask.Thinking = kind != "disabled"
	default:
		return nil, askErrorf("thinking", "thinking must be {\"type\": \"enabled\"} or {\"type\": \"disabled\"}")
	}
	if err := dials(body, &ask); err != nil {
		return nil, err
	}
	if err := ask.Validate(); err != nil {
		return nil, err
	}
	return &ask, nil
}

// ParseRequest is ParseOpenAI or ParseAnthropic, by dialect.
func ParseRequest(body map[string]any, dialect string) (*Ask, error) {
	switch dialect {
	case FormatOpenAI:
		return ParseOpenAI(body)
	case FormatAnthropic:
		return ParseAnthropic(body)
	}
	return nil, askErrorf("format", "format must be one of openai, anthropic (got %s)", PythonRepr(dialect))
}

// ModelID is radixnet-<kind>: what a model is called in /v1/models and in every answer.
func ModelID(m *Model) string { return ModelIDPrefix + m.Kind() }

// KindOfID is the kind a requested model id names ("": whichever model is active).
func KindOfID(name string) string {
	text := strings.ToLower(strings.TrimSpace(name))
	text = strings.TrimPrefix(text, ModelIDPrefix)
	switch text {
	case "", "radixnet", "default", "active":
		return ""
	}
	return text
}

func assistantToken() string {
	raw := make([]byte, 12)
	if _, err := rand.Read(raw); err != nil {
		return fmt.Sprintf("%024x", time.Now().UnixNano())
	}
	return hex.EncodeToString(raw)
}

// AssistantToolCall is a call the model wrote to a tool the request offered.
type AssistantToolCall struct {
	Token string
	Name  string
	Input map[string]any
}

// AssistantChoice is one reply: the thinking, the text, the calls, how it ended, and the turn behind it.
type AssistantChoice struct {
	Index         int
	Thinking      string
	Text          string
	ToolCalls     []AssistantToolCall
	StopReason    string
	StopSequence  *string
	Turn          *Turn
	Guard         map[string]any
	OutputUnits   int
	ThinkingUnits int
}

// AssistantReply is what Respond returns: the choices, the model, and the usage in its units.
type AssistantReply struct {
	Token      string
	ModelName  string
	Created    int64
	Kind       string
	Units      string
	InputUnits int
	Choices    []AssistantChoice
	Thinking   bool
}

// InputUnits is what the request holds, in the model's units (count_tokens).
func InputUnits(enc Encoding, ask *Ask) int {
	n := enc.Len(ask.System)
	for _, m := range ask.Messages {
		n += enc.Len(m.Text)
	}
	return n
}

// Usage is {input, output, thinking, total} in the model's units.
func (r *AssistantReply) Usage() map[string]any {
	output, thinking := 0, 0
	for _, c := range r.Choices {
		output += c.OutputUnits
		thinking += c.ThinkingUnits
	}
	return map[string]any{"input": r.InputUnits, "output": output, "thinking": thinking, "total": r.InputUnits + output}
}

// q is text in double quotes, JSON style - the quoting every port's thinking uses.
func q(text string) string {
	quoted, err := jsonString(text)
	if err != nil {
		return `"` + text + `"`
	}
	return quoted
}

func plural(n int, what string) string { return fmt.Sprintf("%d %s(s)", n, what) }

// narrator turns the search's trace into the lines of the thinking, one event at a time.
type narrator struct {
	index int
	lines []string
	emit  func(map[string]any)
}

func (n *narrator) line(text string) {
	delta := text
	if len(n.lines) > 0 {
		delta = "\n" + text
	}
	n.lines = append(n.lines, text)
	n.emit(map[string]any{"type": "thinking", "index": n.index, "text": delta})
}

func asInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	}
	return 0
}

func (n *narrator) trace(event map[string]any) {
	kind, _ := event["kind"].(string)
	switch kind {
	case "context":
		if usable, _ := event["usable"].(bool); !usable {
			context, _ := event["context"].(string)
			n.line(fmt.Sprintf("looking for %s in the graph: not there whole; dropping a word", q(context)))
		}
	case "candidates":
		offered := asInt(event["offered"])
		what := plural(offered, "path") + " weighed"
		if mode, _ := event["mode"].(string); mode == "sample" {
			what = plural(offered, "walk") + " drawn"
		}
		if context, _ := event["context"].(string); context != "" {
			n.line(fmt.Sprintf("looking for %s in the graph: found, %s", q(context), what))
		} else {
			n.line("starting a fresh text from the beginning: " + what)
		}
	case "pick":
		skipped := asInt(event["skipped"]) - asInt(event["vetoed"])
		if repeat, _ := event["repeat"].(bool); repeat {
			n.line("every path repeats something already said; the best of them is kept as a last resort")
		} else if skipped > 0 {
			n.line(fmt.Sprintf("skipped %d (empty, or already said)", skipped))
		}
	case "rethink":
		noticed, _ := event["noticed"].(string)
		cut, _ := event["cut"].(string)
		caught := fmt.Sprintf("caught itself repeating %s", q(noticed))
		if kindOf, _ := event["kind_of"].(string); kindOf == "stutter" {
			caught = fmt.Sprintf("caught itself saying %s twice", q(noticed))
		}
		explored := asInt(event["explored"])
		var text string
		if asInt(event["steps"]) == 0 {
			text = caught + "; the words it picked up, not its own"
		} else if found, _ := event["found"].(bool); found {
			text = fmt.Sprintf("%s; kept %s and found another way on in %s", caught, q(cut), plural(explored, "path"))
		} else {
			text = fmt.Sprintf("%s; kept %s, weighed %s, found nothing new", caught, q(cut), plural(explored, "path"))
		}
		if asInt(event["taught"]) >= 0 {
			text += " (and learned to hand over there)"
		}
		n.line(text)
	case "fresh":
		n.line("nothing new follows the line; changing the subject with a fresh text")
	}
}

// Deltas cuts text into what each node of the walk added, so the pieces join
// back into text: the first piece is the context the reply picked up plus the
// located node's remainder, every later piece what its node adds beyond the
// overlap.  A text the walk does not line up with (one cut at the cap) comes
// back whole, as one piece.
func Deltas(enc Encoding, labels []string, nodeIDs []int, text string) []string {
	if text == "" {
		return nil
	}
	real := []string{}
	for i, id := range nodeIDs {
		if id >= First && i < len(labels) {
			real = append(real, labels[i])
		}
	}
	if len(real) < 2 {
		return []string{text}
	}
	pieces := make([]string, 0, len(real)-1)
	for _, label := range real[1:] {
		units := enc.Units(label)
		pieces = append(pieces, units.Slice(enc.Overlap(), units.Len()))
	}
	tail := enc.Join(pieces...)
	if tail == "" {
		return []string{text}
	}
	words := enc.Unit == Words
	var head string
	if words {
		switch {
		case text == tail:
			head = ""
		case strings.HasSuffix(text, " "+tail):
			head = text[:len(text)-len(tail)-1]
		default:
			return []string{text}
		}
	} else if strings.HasSuffix(text, tail) {
		head = text[:len(text)-len(tail)]
	} else {
		return []string{text}
	}
	out := []string{}
	if head != "" {
		out = append(out, head)
	}
	for _, piece := range pieces {
		if piece == "" {
			continue
		}
		if words && len(out) > 0 {
			out = append(out, " "+piece)
		} else {
			out = append(out, piece)
		}
	}
	return out
}

// windowed cuts the pieces to the bytes [start, end) of their concatenation.
func windowed(pieces []string, start, end int) []string {
	out := []string{}
	at := 0
	for _, piece := range pieces {
		lo, hi := max(start, at), min(end, at+len(piece))
		if hi > lo {
			out = append(out, piece[lo-at:hi-at])
		}
		at += len(piece)
	}
	return out
}

// firstStop is where the earliest stop sequence begins (and which), or len(text) and "".
func firstStop(text string, stops []string) (int, string) {
	best, which := len(text), ""
	for _, seq := range stops {
		if at := strings.Index(text, seq); at >= 0 && at < best {
			best, which = at, seq
		}
	}
	return best, which
}

// OfferedToolBox is the tools a request offered, as a registry the call parser
// reads arguments against - JSON, key=value pairs, or a bare value for a tool
// with one argument - from the little schema each tool came with.
func OfferedToolBox(tools []OfferedTool) *ToolBox {
	box := NewToolBox()
	for _, tool := range tools {
		properties, _ := tool.Parameters["properties"].(map[string]any)
		required := map[string]bool{}
		if list, ok := tool.Parameters["required"].([]any); ok {
			for _, name := range list {
				if s, ok := name.(string); ok {
					required[s] = true
				}
			}
		}
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		sortStrings(names)
		params := []Param{}
		for _, name := range names {
			spec, _ := properties[name].(map[string]any)
			kind, _ := spec["type"].(string)
			known := false
			for _, t := range []string{"string", "number", "integer", "boolean"} {
				if t == kind {
					known = true
				}
			}
			if !known {
				kind = "string"
			}
			description, _ := spec["description"].(string)
			params = append(params, Param{Name: name, Type: kind, Description: description, Required: required[name]})
		}
		_ = box.Register(&Tool{Name: tool.Name, Description: tool.Description, Params: params})
	}
	return box
}

func sortStrings(items []string) {
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j] < items[j-1]; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}

// assistantGuardReport is what the guard did, for the caller to show: the vetoes, with the reason behind each.
func assistantGuardReport(pair *Filter, verdicts []*FilterVerdict) map[string]any {
	rejected := []*FilterVerdict{}
	for _, v := range verdicts {
		if v.Decision == "reject" {
			rejected = append(rejected, v)
		}
	}
	return map[string]any{"on": true, "vetoed": len(rejected), "rejected": rejected, "verdicts": verdicts,
		"negative": pair.Negative.Stats(), "config": pair.Describe()["config"]}
}

// Respond answers ask with m, streaming every event to emit as it happens, and
// returns the whole reply (Python's respond).  pair is the guard - a Filter
// whose Positive is m - or nil for none; name is the id the reply reports
// ("" = ModelID).  The events, in order, per choice: start; thinking deltas
// (a line each, the second onwards led by a newline); text deltas (one per
// node of the walk); tool_use; done with the stop reason and the turn record.
// After the last choice one end carries the usage of the whole reply.
func Respond(m *Model, pair *Filter, ask *Ask, name string, emit func(map[string]any)) (*AssistantReply, error) {
	if err := ask.Validate(); err != nil {
		return nil, err
	}
	if emit == nil {
		emit = func(map[string]any) {}
	}
	enc := m.Encoding()
	if name == "" {
		name = ModelID(m)
	}
	reply := &AssistantReply{Token: assistantToken(), ModelName: name, Created: time.Now().Unix(), Kind: m.Kind(),
		Units: enc.UnitsName(), InputUnits: InputUnits(enc, ask), Thinking: ask.Thinking}
	said := make([]string, 0, len(ask.Messages))
	for _, msg := range ask.Messages {
		said = append(said, msg.Text)
	}
	heard := NewHeard(said)
	previous := ask.Previous()
	var rng *MT19937
	if ask.Seed != nil {
		rng = NewMT19937(*ask.Seed)
	}
	offered := map[string]bool{}
	names := []string{}
	for _, tool := range ask.Tools {
		offered[tool.Name] = true
		names = append(names, tool.Name)
	}
	var box *ToolBox
	if len(ask.Tools) > 0 {
		box = OfferedToolBox(ask.Tools)
	}
	earlier := len(ask.Messages) - 1
	for index := 0; index < ask.N; index++ {
		emit(map[string]any{"type": "start", "index": index, "id": reply.Token, "model": reply.ModelName,
			"created": reply.Created, "input_units": reply.InputUnits, "thinking": ask.Thinking})
		n := &narrator{index: index, emit: emit}
		opening := fmt.Sprintf("answering %s", q(previous))
		if ask.Prefill() {
			opening = fmt.Sprintf("continuing its own last line %s", q(previous))
		}
		if earlier > 0 {
			opening += fmt.Sprintf(" (%s heard)", plural(earlier, "earlier line"))
		}
		n.line(opening)
		if strings.TrimSpace(ask.System) != "" {
			n.line("a system prompt was given; the network continues text and cannot follow instructions, so it is not read")
		}
		if len(ask.Tools) > 0 {
			n.line(fmt.Sprintf("%s offered (%s); a reply that writes one comes back as a tool call",
				plural(len(ask.Tools), "tool"), strings.Join(names, ", ")))
		}
		verdicts := []*FilterVerdict{}
		seen := map[string]bool{}
		var veto func(string) bool
		if pair != nil {
			veto = func(text string) bool {
				// a candidate offered again after a shorter context is judged once
				if refused, done := seen[text]; done {
					return refused
				}
				verdict := pair.Judge(text)
				refused := verdict.Decision == "reject"
				seen[text] = refused
				verdicts = append(verdicts, verdict)
				if refused {
					n.line(fmt.Sprintf("the negative network vetoed %s: %s", q(text), verdict.Why))
				}
				return refused
			}
		}
		turn, err := m.Reply(previous, ReplyOptions{
			Heard: heard, Index: earlier + 1, Speaker: "assistant", Mode: ask.Mode, MaxLength: ask.MaxTokens,
			Context: ask.Context, Temperature: ask.Temperature, K: ask.K, Beam: ask.Beam, StepPenalty: ask.StepPenalty,
			RNG: rng, AvoidRepeats: ask.AvoidRepeats, AvoidWordRepeats: ask.AvoidWordRepeats, Explore: ask.Explore,
			Learn: ask.Learn, Think: true, ThinkDepth: ThinkDepth, Veto: veto, Trace: n.trace,
		})
		if err != nil {
			return nil, err
		}
		choice := AssistantChoice{Index: index, StopReason: StopEndTurn}
		if pair != nil {
			rejected := []string{}
			for _, v := range verdicts {
				if v.Decision == "reject" {
					rejected = append(rejected, v.Text)
				}
			}
			if len(rejected) > 0 && pair.Config.Learn {
				blamed := []string{}
				for _, text := range rejected {
					if negEnc := pair.Negative.Encoding(); negEnc.Len(text) >= negEnc.N {
						blamed = append(blamed, text)
					}
				}
				reason := pair.Config.Reason
				if reason == "" {
					reason = "filtered"
				}
				if _, err := pair.Negative.Blame(blamed, BlameOptions{Reason: reason, Source: "filter", Note: "vetoed in conversation"}); err != nil {
					return nil, err
				}
			}
			choice.Guard = assistantGuardReport(pair, verdicts)
		}
		vetoedAny := false
		for _, refused := range seen {
			vetoedAny = vetoedAny || refused
		}
		if turn == nil {
			if vetoedAny {
				n.line("nothing to say: the guard vetoed everything it could say")
				choice.StopReason = StopRefusal
			} else {
				n.line("nothing to say: the graph has no way on from here")
			}
		} else {
			ending := fmt.Sprintf("cut at %d %s", ask.MaxTokens, enc.UnitsName())
			if turn.ReachedEnd {
				ending = "reached the end of a text"
			}
			line := fmt.Sprintf("saying %s: cost %.4f, probability %.4f, %s", q(turn.Text), turn.Cost, turn.Probability, ending)
			if turn.Repeat {
				line += "; every path repeated something, so this is a repeat"
			}
			n.line(line)
			whole := turn.Text
			pieces := Deltas(enc, turn.Labels, turn.NodeIDs, whole)
			start := 0
			if ask.Prefill() && turn.Context != "" {
				start = len(turn.Context)
			}
			end := len(whole)
			if !turn.ReachedEnd {
				choice.StopReason = StopMaxTokens
			}
			cut, seq := firstStop(whole[start:], ask.Stop)
			if seq != "" {
				end = start + cut
				choice.StopReason = StopStopSequence
				s := seq
				choice.StopSequence = &s
				n.line(fmt.Sprintf("stopped at the stop sequence %s", q(seq)))
			}
			call := ParseCall(whole[start:end], nil)
			if call != nil && offered[call.Name] {
				call = ParseCall(whole[start:end], box) // the offered tool's schema reads the arguments
			}
			if call != nil {
				switch {
				case call.Name == "":
					n.line(fmt.Sprintf("wrote a tool call it could not finish (%s); it stays text", call.Error))
				case !offered[call.Name] && len(offered) == 0:
					n.line(fmt.Sprintf("wrote a call to %s, but no tools were offered; it stays text", call.Name))
				case !offered[call.Name]:
					n.line(fmt.Sprintf("wrote a call to %s, which was not offered; it stays text", call.Name))
				case call.Error != "":
					n.line(fmt.Sprintf("wrote a call to %s it could not finish (%s); it stays text", call.Name, call.Error))
				default:
					n.line("wrote a tool call: " + strings.TrimSpace(CallText(call.Name, call.Arguments)))
					end = start + call.Span[0]
					choice.ToolCalls = append(choice.ToolCalls, AssistantToolCall{Token: assistantToken(), Name: call.Name, Input: call.Arguments})
					choice.StopReason = StopToolUse
					choice.StopSequence = nil
				}
			}
			choice.Text = whole[start:end]
			choice.Turn = turn
			for _, piece := range windowed(pieces, start, end) {
				emit(map[string]any{"type": "text", "index": index, "text": piece})
			}
			for _, tc := range choice.ToolCalls {
				emit(map[string]any{"type": "tool_use", "index": index, "id": tc.Token, "name": tc.Name, "input": tc.Input})
			}
			added := ""
			if turn.Context != "" {
				added = turn.Reply
			}
			heard.Remember(turn.Text, added)
		}
		choice.Thinking = strings.Join(n.lines, "\n")
		choice.ThinkingUnits = enc.Len(choice.Thinking)
		choice.OutputUnits = enc.Len(choice.Text) + choice.ThinkingUnits
		reply.Choices = append(reply.Choices, choice)
		var stopSequence any
		if choice.StopSequence != nil {
			stopSequence = *choice.StopSequence
		}
		var turnDoc any
		if choice.Turn != nil {
			turnDoc = choice.Turn
		}
		var guard any
		if choice.Guard != nil {
			guard = choice.Guard
		}
		emit(map[string]any{"type": "done", "index": index, "stop_reason": choice.StopReason, "stop_sequence": stopSequence,
			"output_units": choice.OutputUnits, "thinking_units": choice.ThinkingUnits, "turn": turnDoc, "guard": guard})
	}
	emit(map[string]any{"type": "end", "usage": reply.Usage()})
	return reply, nil
}

func openAIToolCall(call AssistantToolCall) map[string]any {
	arguments, err := marshalSorted(call.Input)
	if err != nil {
		arguments = "{}"
	}
	return map[string]any{"id": "call_" + call.Token, "type": "function",
		"function": map[string]any{"name": call.Name, "arguments": arguments}}
}

// record is the model's own account of the answer, beside the standard fields.
func (r *AssistantReply) record() map[string]any {
	choices := make([]map[string]any, 0, len(r.Choices))
	for _, c := range r.Choices {
		var turn any
		if c.Turn != nil {
			turn = c.Turn
		}
		var guard any
		if c.Guard != nil {
			guard = c.Guard
		}
		choices = append(choices, map[string]any{"index": c.Index, "stop_reason": c.StopReason, "turn": turn, "guard": guard})
	}
	return map[string]any{"kind": r.Kind, "units": r.Units, "choices": choices}
}

func openAIUsage(usage map[string]any) map[string]any {
	return map[string]any{"prompt_tokens": usage["input"], "completion_tokens": usage["output"], "total_tokens": usage["total"],
		"completion_tokens_details": map[string]any{"reasoning_tokens": usage["thinking"]}}
}

// ToOpenAI is the reply as a Chat Completions chat.completion object.
func ToOpenAI(r *AssistantReply) map[string]any {
	choices := make([]map[string]any, 0, len(r.Choices))
	for _, c := range r.Choices {
		var content any = c.Text
		if c.Text == "" && len(c.ToolCalls) > 0 {
			content = nil
		}
		message := map[string]any{"role": "assistant", "content": content}
		if r.Thinking {
			message["reasoning_content"] = c.Thinking
		}
		if len(c.ToolCalls) > 0 {
			calls := make([]map[string]any, 0, len(c.ToolCalls))
			for _, call := range c.ToolCalls {
				calls = append(calls, openAIToolCall(call))
			}
			message["tool_calls"] = calls
		}
		choices = append(choices, map[string]any{"index": c.Index, "message": message, "logprobs": nil,
			"finish_reason": FinishReason(c.StopReason)})
	}
	return map[string]any{"id": "chatcmpl-" + r.Token, "object": "chat.completion", "created": r.Created,
		"model": r.ModelName, "choices": choices, "usage": openAIUsage(r.Usage()), "radixnet": r.record()}
}

func anthropicContent(c *AssistantChoice, thinking bool) []map[string]any {
	blocks := []map[string]any{}
	if thinking {
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": c.Thinking, "signature": ""})
	}
	if c.Text != "" || len(c.ToolCalls) == 0 {
		blocks = append(blocks, map[string]any{"type": "text", "text": c.Text})
	}
	for _, call := range c.ToolCalls {
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": "toolu_" + call.Token, "name": call.Name, "input": call.Input})
	}
	return blocks
}

// ToAnthropic is the reply as a Messages message object (the first choice is the answer).
func ToAnthropic(r *AssistantReply) map[string]any {
	c := &AssistantChoice{StopReason: StopEndTurn}
	if len(r.Choices) > 0 {
		c = &r.Choices[0]
	}
	var stopSequence any
	if c.StopSequence != nil {
		stopSequence = *c.StopSequence
	}
	return map[string]any{"id": "msg_" + r.Token, "type": "message", "role": "assistant", "model": r.ModelName,
		"content": anthropicContent(c, r.Thinking), "stop_reason": c.StopReason, "stop_sequence": stopSequence,
		"usage": map[string]any{"input_tokens": r.InputUnits, "output_tokens": c.OutputUnits}, "radixnet": r.record()}
}

// ToFormat is ToOpenAI or ToAnthropic, by dialect.
func ToFormat(r *AssistantReply, dialect string) map[string]any {
	if dialect == FormatAnthropic {
		return ToAnthropic(r)
	}
	return ToOpenAI(r)
}

// SSE is one server-sent event: `event: name` (when given), then `data: <json>` and a blank line.
func SSE(data any, event string) string {
	body, _ := marshalCompact(data)
	text := strings.TrimRight(string(body), "\n")
	if event != "" {
		return "event: " + event + "\ndata: " + text + "\n\n"
	}
	return "data: " + text + "\n\n"
}

// OpenAIErrorFrame is the frame a failure mid-stream is reported with, in Chat Completions' shape.
func OpenAIErrorFrame(message string) string {
	return SSE(map[string]any{"error": map[string]any{"message": message, "type": "server_error", "param": nil, "code": nil}}, "")
}

// AnthropicErrorFrame is the frame a failure mid-stream is reported with, in Messages' shape.
func AnthropicErrorFrame(message string) string {
	return SSE(map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": message}}, "error")
}

// OpenAIStream renders Respond's events as Chat Completions chunks.
type OpenAIStream struct {
	Token        string
	Model        string
	Created      int64
	IncludeUsage bool
	Thinking     bool
}

func (s *OpenAIStream) chunk(choices []map[string]any, extra map[string]any) string {
	doc := map[string]any{"id": "chatcmpl-" + s.Token, "object": "chat.completion.chunk", "created": s.Created,
		"model": s.Model, "choices": choices}
	for k, v := range extra {
		doc[k] = v
	}
	return SSE(doc, "")
}

func streamChoice(index int, delta map[string]any, finish any) map[string]any {
	return map[string]any{"index": index, "delta": delta, "finish_reason": finish}
}

// Frames are the frames an event becomes.
func (s *OpenAIStream) Frames(event map[string]any) []string {
	index := asInt(event["index"])
	kind, _ := event["type"].(string)
	switch kind {
	case "start":
		return []string{s.chunk([]map[string]any{streamChoice(index, map[string]any{"role": "assistant", "content": ""}, nil)}, nil)}
	case "thinking":
		if !s.Thinking {
			return nil
		}
		return []string{s.chunk([]map[string]any{streamChoice(index, map[string]any{"reasoning_content": event["text"]}, nil)}, nil)}
	case "text":
		return []string{s.chunk([]map[string]any{streamChoice(index, map[string]any{"content": event["text"]}, nil)}, nil)}
	case "tool_use":
		input, _ := event["input"].(map[string]any)
		arguments, err := marshalSorted(input)
		if err != nil {
			arguments = "{}"
		}
		id, _ := event["id"].(string)
		call := map[string]any{"index": 0, "id": "call_" + id, "type": "function",
			"function": map[string]any{"name": event["name"], "arguments": arguments}}
		return []string{s.chunk([]map[string]any{streamChoice(index, map[string]any{"tool_calls": []map[string]any{call}}, nil)}, nil)}
	case "done":
		stop, _ := event["stop_reason"].(string)
		return []string{s.chunk([]map[string]any{streamChoice(index, map[string]any{}, FinishReason(stop))},
			map[string]any{"radixnet": map[string]any{"stop_reason": stop, "turn": event["turn"], "guard": event["guard"]}})}
	case "end":
		out := []string{}
		if s.IncludeUsage {
			usage, _ := event["usage"].(map[string]any)
			out = append(out, s.chunk([]map[string]any{}, map[string]any{"usage": openAIUsage(usage)}))
		}
		return append(out, "data: [DONE]\n\n")
	}
	return nil
}

// AnthropicStream renders Respond's events as Messages events, one content block at a time.
type AnthropicStream struct {
	Token      string
	Model      string
	Created    int64
	InputUnits int
	Thinking   bool
	block      int
	open       string
	answered   bool
}

// NewAnthropicStream is a stream for one message.
func NewAnthropicStream(token, model string, created int64, inputUnits int, thinking bool) *AnthropicStream {
	return &AnthropicStream{Token: token, Model: model, Created: created, InputUnits: inputUnits, Thinking: thinking, block: -1}
}

func (s *AnthropicStream) delta(delta map[string]any) string {
	return SSE(map[string]any{"type": "content_block_delta", "index": s.block, "delta": delta}, "content_block_delta")
}

func (s *AnthropicStream) openBlock(kind string, block map[string]any) []string {
	out := s.close()
	s.block++
	s.open = kind
	if kind != "thinking" {
		s.answered = true
	}
	return append(out, SSE(map[string]any{"type": "content_block_start", "index": s.block, "content_block": block}, "content_block_start"))
}

func (s *AnthropicStream) close() []string {
	if s.open == "" {
		return nil
	}
	out := []string{}
	if s.open == "thinking" {
		out = append(out, s.delta(map[string]any{"type": "signature_delta", "signature": ""}))
	}
	out = append(out, SSE(map[string]any{"type": "content_block_stop", "index": s.block}, "content_block_stop"))
	s.open = ""
	return out
}

// Frames are the frames an event becomes.
func (s *AnthropicStream) Frames(event map[string]any) []string {
	kind, _ := event["type"].(string)
	if kind == "end" {
		return nil
	}
	if asInt(event["index"]) != 0 {
		return nil // a Messages reply is one message: the first choice
	}
	switch kind {
	case "start":
		message := map[string]any{"id": "msg_" + s.Token, "type": "message", "role": "assistant", "model": s.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": s.InputUnits, "output_tokens": 0}}
		return []string{SSE(map[string]any{"type": "message_start", "message": message}, "message_start")}
	case "thinking":
		if !s.Thinking {
			return nil
		}
		out := []string{}
		if s.open != "thinking" {
			out = s.openBlock("thinking", map[string]any{"type": "thinking", "thinking": "", "signature": ""})
		}
		return append(out, s.delta(map[string]any{"type": "thinking_delta", "thinking": event["text"]}))
	case "text":
		out := []string{}
		if s.open != "text" {
			out = s.openBlock("text", map[string]any{"type": "text", "text": ""})
		}
		return append(out, s.delta(map[string]any{"type": "text_delta", "text": event["text"]}))
	case "tool_use":
		id, _ := event["id"].(string)
		out := s.openBlock("tool_use", map[string]any{"type": "tool_use", "id": "toolu_" + id, "name": event["name"], "input": map[string]any{}})
		input, _ := event["input"].(map[string]any)
		partial, err := marshalSorted(input)
		if err != nil {
			partial = "{}"
		}
		out = append(out, s.delta(map[string]any{"type": "input_json_delta", "partial_json": partial}))
		return append(out, s.close()...)
	case "done":
		out := []string{}
		if !s.answered {
			// a reply with nothing to say is still one (empty) text block
			out = append(out, s.openBlock("text", map[string]any{"type": "text", "text": ""})...)
		}
		out = append(out, s.close()...)
		out = append(out, SSE(map[string]any{"type": "message_delta",
			"delta":    map[string]any{"stop_reason": event["stop_reason"], "stop_sequence": event["stop_sequence"]},
			"usage":    map[string]any{"output_tokens": event["output_units"]},
			"radixnet": map[string]any{"turn": event["turn"], "guard": event["guard"]}}, "message_delta"))
		return append(out, SSE(map[string]any{"type": "message_stop"}, "message_stop"))
	}
	return nil
}

// ShapeV1Error is an error as a /v1 client expects it: the dialect's own envelope.
func ShapeV1Error(dialect string, status int, message, param string) map[string]any {
	if dialect == FormatOpenAI {
		kind := "invalid_request_error"
		if status >= 500 {
			kind = "server_error"
		}
		var code any
		if status == 404 && param == "model" {
			code = "model_not_found"
		}
		var p any
		if param != "" {
			p = param
		}
		return map[string]any{"error": map[string]any{"message": message, "type": kind, "param": p, "code": code}}
	}
	kind := "invalid_request_error"
	switch {
	case status >= 500:
		kind = "api_error"
	case status == 404:
		kind = "not_found_error"
	case status == 413:
		kind = "request_too_large"
	}
	return map[string]any{"type": "error", "error": map[string]any{"type": kind, "message": message}}
}

// DialectOf is which dialect a /v1 path speaks.
func DialectOf(path string) string {
	if strings.HasPrefix(path, "/v1/messages") {
		return FormatAnthropic
	}
	return FormatOpenAI
}
