package radixnet

// External tools the network can call, and the text format it learns them in.
//
// The network is a character-level graph model: it cannot "decide" to call a
// function, it can only emit characters.  So a tool call is a piece of text
// like any other training data:
//
//	<tool>web_fetch {"url": "https://example.com"}</tool>
//
// and what the tool answers comes back as another piece of text:
//
//	<result>Example Domain. This domain is for use in ...</result>
//
// A whole attempt at a task is therefore one training text (TranscriptText)
// that the network can be rewarded or punished for as a unit.  Because the
// emission is only text it is routinely malformed; ParseCall is deliberately
// lenient (JSON arguments, key=value pairs, or a bare value for a
// single-argument tool) and whatever it still cannot read is handed to an LLM
// to repair (agent.go).
//
// The Go twin of the Python radixnet.tools: the same text format, the same
// tool names and schemas, the same lenient parsing and the same refusals.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// The markers of the text format.
const (
	CallOpen    = "<tool>"
	CallClose   = "</tool>"
	ResultOpen  = "<result>"
	ResultClose = "</result>"
	AnswerOpen  = "<answer>"
	AnswerClose = "</answer>"
)

// DefaultObservationChars is how much of a tool's output goes back into the
// transcript by default.
const DefaultObservationChars = 600

// ParamTypes are the argument types a tool may declare.
var ParamTypes = []string{"string", "number", "integer", "boolean"}

var (
	callRe   = regexp.MustCompile(`(?s)` + regexp.QuoteMeta(CallOpen) + `(.*?)(?:` + regexp.QuoteMeta(CallClose) + `|$)`)
	answerRe = regexp.MustCompile(`(?s)` + regexp.QuoteMeta(AnswerOpen) + `(.*?)(?:` + regexp.QuoteMeta(AnswerClose) + `|$)`)
	nameRe   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*`)
	wholeRe  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*$`)
	pairRe   = regexp.MustCompile(`([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*("[^"]*"|'[^']*'|[^,;]+)`)
)

// collapse is the whitespace-collapsed form of a text (Python's " ".join(t.split())).
func collapse(text string) string { return strings.Join(strings.Fields(text), " ") }

// TaskHeader is the first line of every transcript - what the network
// continues from.
func TaskHeader(prompt string) string { return "TASK: " + collapse(prompt) + "\n" }

// CallText is `<tool>name {"arg": "value"}</tool>` - one line, JSON arguments
// with their keys sorted, as Python writes them.
func CallText(name string, arguments map[string]any) string {
	if arguments == nil {
		arguments = map[string]any{}
	}
	payload, err := marshalSorted(arguments)
	if err != nil {
		payload = "{}"
	}
	return CallOpen + name + " " + payload + CallClose + "\n"
}

// marshalSorted renders a value exactly as Python's
// json.dumps(value, ensure_ascii=False, sort_keys=True) does - sorted keys, a
// space after every colon and comma, floats as repr() writes them - so a call
// written on one side is character for character the call the other writes.
func marshalSorted(value any) (string, error) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			name, err := jsonString(key)
			if err != nil {
				return "", err
			}
			item, err := marshalSorted(typed[key])
			if err != nil {
				return "", err
			}
			parts = append(parts, name+": "+item)
		}
		return "{" + strings.Join(parts, ", ") + "}", nil
	case []any:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			rendered, err := marshalSorted(item)
			if err != nil {
				return "", err
			}
			parts = append(parts, rendered)
		}
		return "[" + strings.Join(parts, ", ") + "]", nil
	case []string:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			rendered, err := jsonString(item)
			if err != nil {
				return "", err
			}
			parts = append(parts, rendered)
		}
		return "[" + strings.Join(parts, ", ") + "]", nil
	case string:
		return jsonString(typed)
	case float64:
		return pythonFloat(typed), nil
	case float32:
		return pythonFloat(float64(typed)), nil
	case int:
		return strconv.Itoa(typed), nil
	case int64:
		return strconv.FormatInt(typed, 10), nil
	case bool:
		if typed {
			return "true", nil
		}
		return "false", nil
	case nil:
		return "null", nil
	}
	return jsonOf(value)
}

// jsonString is one string as JSON, non-ASCII left as it is (ensure_ascii=False).
func jsonString(text string) (string, error) { return jsonOf(text) }

// jsonOf is Go's encoder with HTML escaping off.
func jsonOf(value any) (string, error) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return "", err
	}
	return strings.TrimRight(buf.String(), "\n"), nil
}

// ResultText is `<result>...</result>` - the observation, whitespace collapsed
// and clipped to limit characters (limit < 0 = no clipping).
func ResultText(output string, limit int) string {
	text := collapse(output)
	if limit >= 0 && runeLen(text) > limit {
		text = strings.TrimRight(string([]rune(text)[:limit]), " \t") + " ..."
	}
	return ResultOpen + text + ResultClose + "\n"
}

// AnswerText is `<answer>...</answer>` - the last line of a finished transcript.
func AnswerText(answer string) string { return AnswerOpen + collapse(answer) + AnswerClose + "\n" }

// FormatObservation is what a tool call adds to the transcript: the output, or
// the error prefixed with ERROR:.
func FormatObservation(result *ToolResult, limit int) string {
	if result.OK {
		return ResultText(result.Output, limit)
	}
	return ResultText("ERROR: "+result.Error, limit)
}

// TranscriptStep is one call of a transcript with what it answered.
type TranscriptStep struct {
	Name      string
	Arguments map[string]any
	Result    *ToolResult
}

// TranscriptText is one attempt as a single training text: the task, every
// call with its result, then the answer (answer nil: the attempt never
// answered).
func TranscriptText(prompt string, steps []TranscriptStep, answer *string, limit int) string {
	var b strings.Builder
	b.WriteString(TaskHeader(prompt))
	for _, step := range steps {
		b.WriteString(CallText(step.Name, step.Arguments))
		b.WriteString(FormatObservation(step.Result, limit))
	}
	if answer != nil {
		b.WriteString(AnswerText(*answer))
	}
	return b.String()
}

// loadsTruncated reads JSON out of text a generator may have cut off mid-string
// or mid-object.  A call is one line by construction, so the line is tried
// first, then the text up to its last "}", then the same completed with the
// closing quote and braces it is missing.
func loadsTruncated(text string) any {
	head := strings.TrimSpace(strings.SplitN(text, "\n", 2)[0])
	bases := []string{text}
	if head != text {
		bases = []string{head, text} // a call is one line: prefer it over the run-on text
	}
	candidates := []string{}
	for _, base := range bases {
		candidates = append(candidates, base)
		if i := strings.LastIndex(base, "}"); i >= 0 {
			candidates = append(candidates, base[:i+1])
		}
		for _, tail := range []string{`"}`, "}", `"}}`, "}}", `"]}`} {
			candidates = append(candidates, base+tail)
		}
	}
	for _, candidate := range candidates {
		var out any
		if err := json.Unmarshal([]byte(candidate), &out); err == nil {
			return out
		}
	}
	return nil
}

// ParseArguments reads the arguments of a call: JSON, key=value pairs, or a
// bare value for a single-argument tool.
func ParseArguments(raw string, tool *Tool) (map[string]any, error) {
	text := strings.Trim(strings.TrimSpace(raw), ",;")
	if text == "" {
		return map[string]any{}, nil
	}
	if text[0] == '{' || text[0] == '[' { // JSON, possibly truncated by the generator
		data := loadsTruncated(text)
		if object, ok := data.(map[string]any); ok {
			return object, nil
		}
		if list, ok := data.([]any); ok && tool != nil {
			out := map[string]any{}
			for i, param := range tool.Params {
				if i >= len(list) {
					break
				}
				out[param.Name] = list[i]
			}
			return out, nil
		}
	}
	if pairs := pairRe.FindAllStringSubmatch(text, -1); len(pairs) > 0 {
		out := map[string]any{}
		for _, pair := range pairs {
			out[pair[1]] = strings.Trim(strings.TrimSpace(pair[2]), "\"'")
		}
		return out, nil
	}
	if tool != nil {
		wanted := []Param{}
		for _, param := range tool.Params {
			if param.Required {
				wanted = append(wanted, param)
			}
		}
		if len(wanted) == 0 {
			wanted = tool.Params
		}
		if len(wanted) == 1 {
			return map[string]any{wanted[0].Name: strings.Trim(text, "\"'")}, nil
		}
	}
	return nil, &ToolError{fmt.Sprintf("cannot read the arguments of the call: %s", pythonRepr(clipString(strings.TrimSpace(raw), 120)))}
}

// ParseCall is the first <tool> call in text (nil when there is none).
//
// Lenient on purpose: an unterminated call is read to the end of the text, the
// tool name is whatever leading identifier is there, and the arguments go
// through ParseArguments.  A call that cannot be understood comes back with
// Error set rather than dropped, so the caller can hand it to the LLM mediator
// instead of guessing.
func ParseCall(text string, box *ToolBox) *ToolCall {
	match := callRe.FindStringSubmatchIndex(text)
	if match == nil {
		return nil
	}
	body := strings.TrimSpace(text[match[2]:match[3]])
	span := [2]int{match[0], match[1]}
	found := nameRe.FindString(body)
	if found == "" {
		return &ToolCall{Arguments: map[string]any{}, Raw: body, Span: span,
			Error: "no tool name in " + pythonRepr(clipString(body, 80))}
	}
	name := strings.TrimRight(found, ".-")
	rest := strings.TrimSpace(body[len(found):])
	var tool *Tool
	if box != nil {
		if box.Has(name) {
			tool, _ = box.Get(name)
		} else {
			return &ToolCall{Name: name, Arguments: map[string]any{}, Raw: rest, Span: span,
				Error: fmt.Sprintf("unknown tool %s (have: %s)", pythonRepr(name), strings.Join(box.Names(), ", "))}
		}
	}
	arguments, err := ParseArguments(rest, tool)
	if err != nil {
		return &ToolCall{Name: name, Arguments: map[string]any{}, Raw: rest, Span: span, Error: err.Error()}
	}
	return &ToolCall{Name: name, Arguments: arguments, Raw: rest, Span: span}
}

// FindAnswer is the text of the first <answer> block ("" when the attempt did
// not answer).
func FindAnswer(text string) string {
	match := answerRe.FindStringSubmatch(text)
	if match == nil {
		return ""
	}
	return collapse(match[1])
}

// -- tools -------------------------------------------------------------------

// ToolError is a call that could not be made: unknown tool, bad arguments, or
// a refused request.
type ToolError struct{ Message string }

func (e *ToolError) Error() string { return e.Message }

// toolErrorf builds a ToolError.
func toolErrorf(format string, args ...any) *ToolError {
	return &ToolError{fmt.Sprintf(format, args...)}
}

// Param is one argument of a tool, with the little bit of schema the LLM needs
// to fill it in.
type Param struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Required    bool     `json:"required"`
	Default     any      `json:"default"`
	Enum        []string `json:"enum"`
}

// Schema is the parameter as JSON schema.
func (p Param) Schema() map[string]any {
	data := map[string]any{"type": p.Type, "description": p.Description}
	if len(p.Enum) > 0 {
		data["enum"] = p.Enum
	}
	return data
}

// Coerce is value as this parameter's type.
func (p Param) Coerce(value any) (any, error) {
	var out any
	switch p.Type {
	case "string":
		switch typed := value.(type) {
		case string:
			out = typed
		case map[string]any, []any:
			raw, err := marshalSorted(typed)
			if err != nil {
				return nil, toolErrorf("argument %s must be a string (got %v)", pythonRepr(p.Name), value)
			}
			out = raw
		default:
			out = pythonString(value)
		}
	case "boolean":
		if typed, ok := value.(bool); ok {
			out = typed
		} else {
			text := strings.ToLower(strings.TrimSpace(pythonString(value)))
			out = text == "1" || text == "true" || text == "yes" || text == "on"
		}
	case "integer":
		if typed, ok := value.(bool); ok {
			if typed {
				out = 1
			} else {
				out = 0
			}
			break
		}
		number, err := toNumber(value)
		if err != nil {
			return nil, toolErrorf("argument %s must be a %s (got %s)", pythonRepr(p.Name), p.Type, pythonValue(value))
		}
		out = int(number)
	default:
		number, err := toNumber(value)
		if err != nil {
			return nil, toolErrorf("argument %s must be a %s (got %s)", pythonRepr(p.Name), p.Type, pythonValue(value))
		}
		out = number
	}
	if len(p.Enum) > 0 && !contains(p.Enum, pythonString(out)) {
		return nil, toolErrorf("argument %s must be one of %s (got %s)", pythonRepr(p.Name),
			strings.Join(p.Enum, ", "), pythonValue(out))
	}
	return out, nil
}

// toNumber reads a number out of a decoded JSON value or a string.
func toNumber(value any) (float64, error) {
	switch typed := value.(type) {
	case float64:
		return typed, nil
	case float32:
		return float64(typed), nil
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	case json.Number:
		return typed.Float64()
	case string:
		return strconv.ParseFloat(strings.TrimSpace(typed), 64)
	case bool:
		if typed {
			return 1, nil
		}
		return 0, nil
	}
	return 0, fmt.Errorf("not a number")
}

// pythonString renders a value the way Python's str() would, so the two sides
// coerce arguments identically.
func pythonString(value any) string {
	switch typed := value.(type) {
	case nil:
		return "None"
	case string:
		return typed
	case bool:
		if typed {
			return "True"
		}
		return "False"
	case float64:
		return formatFloat(typed)
	case int:
		return strconv.Itoa(typed)
	}
	raw, err := marshalSorted(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return raw
}

// pythonValue is a value as a Python repr() would show it in a message.
func pythonValue(value any) string {
	if text, ok := value.(string); ok {
		return pythonRepr(text)
	}
	return pythonString(value)
}

// formatFloat renders a float as Python's str() does: an integral value keeps
// its ".0", everything else is the shortest round-tripping form.
func formatFloat(value float64) string {
	if value == float64(int64(value)) && value < 1e16 && value > -1e16 {
		return strconv.FormatInt(int64(value), 10) + ".0"
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

// ToolCall is a call read out of generated text: the name, the arguments and
// why it is unusable (Error).
type ToolCall struct {
	Name      string         `json:"tool"`
	Arguments map[string]any `json:"arguments"`
	Raw       string         `json:"raw"`
	Span      [2]int         `json:"-"`
	Error     string         `json:"error"`
}

// OK reports whether the call can be run.
func (c *ToolCall) OK() bool { return c.Error == "" && c.Name != "" }

// Text is the call as the network would have written it.
func (c *ToolCall) Text() string { return CallText(c.Name, c.Arguments) }

// ToolResult is what a tool answered: Output when OK, otherwise Error (both
// readable by the model and the LLM).
type ToolResult struct {
	Tool      string         `json:"tool"`
	Arguments map[string]any `json:"arguments"`
	OK        bool           `json:"ok"`
	Output    string         `json:"output"`
	Error     string         `json:"error"`
	Seconds   float64        `json:"seconds"`
	Meta      map[string]any `json:"meta"`
}

// ToolHandler runs a tool: the validated arguments in, the output and whatever
// metadata the caller wants out.  Returning a *ToolError is the way to report
// a clean failure.
type ToolHandler func(arguments map[string]any) (string, map[string]any, error)

// Tool is a named function the network can call by emitting text.
type Tool struct {
	Name        string
	Description string
	Params      []Param
	Handler     ToolHandler
	Network     bool
}

// Signature is `name(arg: type, [optional]: type)` - the one-liner the prompts
// and the CLI show.
func (t *Tool) Signature() string {
	parts := make([]string, 0, len(t.Params))
	for _, param := range t.Params {
		if param.Required {
			parts = append(parts, param.Name+": "+param.Type)
		} else {
			parts = append(parts, "["+param.Name+": "+param.Type+"]")
		}
	}
	return t.Name + "(" + strings.Join(parts, ", ") + ")"
}

// Schema is the tool in Ollama's / OpenAI's tools format.
func (t *Tool) Schema() map[string]any {
	properties := map[string]any{}
	required := []string{}
	for _, param := range t.Params {
		properties[param.Name] = param.Schema()
		if param.Required {
			required = append(required, param.Name)
		}
	}
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"parameters": map[string]any{
				"type": "object", "properties": properties, "required": required,
			},
		},
	}
}

// Coerce validates arguments: types coerced, defaults filled in, unknown names
// dropped, missing ones refused.
func (t *Tool) Coerce(arguments map[string]any) (map[string]any, error) {
	known := map[string]Param{}
	aliases := map[string]string{}
	for _, param := range t.Params {
		known[param.Name] = param
		aliases[strings.ToLower(param.Name)] = param.Name
	}
	out := map[string]any{}
	for key, value := range arguments {
		name := key
		if _, ok := known[name]; !ok {
			name = aliases[strings.ToLower(strings.TrimSpace(key))]
		}
		if name == "" || value == nil {
			continue
		}
		coerced, err := known[name].Coerce(value)
		if err != nil {
			return nil, err
		}
		out[name] = coerced
	}
	missing := []string{}
	for _, param := range t.Params {
		if param.Required {
			if _, ok := out[param.Name]; !ok {
				missing = append(missing, param.Name)
			}
		}
	}
	if len(missing) > 0 {
		return nil, toolErrorf("%s: missing argument(s) %s — %s", t.Name, strings.Join(missing, ", "), t.Signature())
	}
	for _, param := range t.Params {
		if _, ok := out[param.Name]; !ok && param.Default != nil {
			coerced, err := param.Coerce(param.Default)
			if err != nil {
				return nil, err
			}
			out[param.Name] = coerced
		}
	}
	return out, nil
}

// ToDict is the tool as the API and the CLI report it.
func (t *Tool) ToDict() map[string]any {
	params := make([]map[string]any, 0, len(t.Params))
	for _, param := range t.Params {
		params = append(params, map[string]any{
			"name": param.Name, "type": param.Type, "description": param.Description,
			"required": param.Required, "default": param.Default, "enum": param.Enum,
		})
	}
	return map[string]any{
		"name": t.Name, "description": t.Description, "signature": t.Signature(),
		"network": t.Network, "params": params,
	}
}

// ToolBox is the registry: what exists, what it looks like to the LLM, and how
// a call is run.  Call never fails outright - a failure becomes a ToolResult
// with OK false, because a failed call is training data too.
type ToolBox struct {
	order []string
	tools map[string]*Tool
	Calls int
}

// NewToolBox builds a registry over the given tools.
func NewToolBox(tools ...*Tool) *ToolBox {
	box := &ToolBox{tools: map[string]*Tool{}}
	for _, tool := range tools {
		box.Register(tool)
	}
	return box
}

// Register adds a tool (replacing one of the same name).
func (b *ToolBox) Register(tool *Tool) error {
	if tool == nil {
		return fmt.Errorf("register takes a tool")
	}
	if !wholeRe.MatchString(tool.Name) {
		return fmt.Errorf("invalid tool name %s", pythonRepr(tool.Name))
	}
	if _, seen := b.tools[tool.Name]; !seen {
		b.order = append(b.order, tool.Name)
	}
	b.tools[tool.Name] = tool
	return nil
}

// Remove drops a tool; it reports whether there was one.
func (b *ToolBox) Remove(name string) bool {
	if _, ok := b.tools[name]; !ok {
		return false
	}
	delete(b.tools, name)
	for i, existing := range b.order {
		if existing == name {
			b.order = append(b.order[:i], b.order[i+1:]...)
			break
		}
	}
	return true
}

// Has reports whether a tool of that name is registered.
func (b *ToolBox) Has(name string) bool { _, ok := b.tools[name]; return ok }

// Get is the tool of that name.
func (b *ToolBox) Get(name string) (*Tool, error) {
	tool, ok := b.tools[name]
	if !ok {
		have := strings.Join(b.Names(), ", ")
		if have == "" {
			have = "none"
		}
		return nil, toolErrorf("unknown tool %s (have: %s)", pythonRepr(name), have)
	}
	return tool, nil
}

// Names are the registered names, in registration order.
func (b *ToolBox) Names() []string { return append([]string{}, b.order...) }

// Tools are the registered tools, in registration order.
func (b *ToolBox) Tools() []*Tool {
	out := make([]*Tool, 0, len(b.order))
	for _, name := range b.order {
		out = append(out, b.tools[name])
	}
	return out
}

// Len is how many tools are registered.
func (b *ToolBox) Len() int { return len(b.order) }

// Describe is every tool as the API reports it.
func (b *ToolBox) Describe() []map[string]any {
	out := make([]map[string]any, 0, len(b.order))
	for _, tool := range b.Tools() {
		out = append(out, tool.ToDict())
	}
	return out
}

// Schemas is every tool in Ollama's tools format.
func (b *ToolBox) Schemas() []map[string]any {
	out := make([]map[string]any, 0, len(b.order))
	for _, tool := range b.Tools() {
		out = append(out, tool.Schema())
	}
	return out
}

// Catalogue is the tool list as prompt text: one `name(args) — description`
// line each.
func (b *ToolBox) Catalogue() string {
	lines := make([]string, 0, len(b.order))
	for _, tool := range b.Tools() {
		lines = append(lines, "- "+tool.Signature()+" — "+tool.Description)
	}
	return strings.Join(lines, "\n")
}

// Call runs one tool; failures are reported, not raised.
func (b *ToolBox) Call(name string, arguments map[string]any) *ToolResult {
	started := time.Now()
	b.Calls++
	if arguments == nil {
		arguments = map[string]any{}
	}
	fail := func(message string) *ToolResult {
		return &ToolResult{Tool: name, Arguments: arguments, OK: false, Error: message,
			Seconds: time.Since(started).Seconds(), Meta: map[string]any{}}
	}
	tool, err := b.Get(name)
	if err != nil {
		return fail(err.Error())
	}
	values, err := tool.Coerce(arguments)
	if err != nil {
		return fail(err.Error())
	}
	if tool.Handler == nil {
		return fail(name + ": no handler is installed")
	}
	output, meta, err := tool.Handler(values)
	if err != nil {
		return fail(err.Error())
	}
	if meta == nil {
		meta = map[string]any{}
	}
	return &ToolResult{Tool: name, Arguments: values, OK: true, Output: output,
		Seconds: time.Since(started).Seconds(), Meta: meta}
}

// Run is Call for a parsed ToolCall (a broken one becomes a failed result).
func (b *ToolBox) Run(call *ToolCall) *ToolResult {
	if !call.OK() {
		message := call.Error
		if message == "" {
			message = "unusable tool call"
		}
		return &ToolResult{Tool: call.Name, Arguments: call.Arguments, OK: false, Error: message,
			Meta: map[string]any{}}
	}
	return b.Call(call.Name, call.Arguments)
}

// sortedNames is the registered names in alphabetical order (the API reports
// them that way).
func (b *ToolBox) sortedNames() []string {
	out := b.Names()
	sort.Strings(out)
	return out
}
