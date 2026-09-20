package radixnet

// An MCP server: the network's tools *and* the network itself, over the Model
// Context Protocol.
//
// Two things are exposed to any MCP client (Claude Desktop, an editor, another
// agent), and the second is the interesting one:
//
//   - the tools of tools.go - web_search, web_fetch, web_links, calculator, and
//     python / read_file when they are switched on.  The same registry the
//     network calls by writing text, offered to whoever else wants it;
//   - the model - radixnet_predict, radixnet_generate, radixnet_score,
//     radixnet_stats, radixnet_solve (one task through the whole agent loop:
//     criteria, tool calls, judging) and radixnet_judge (the negative network's
//     verdict on a text).  A client can therefore ask *this* network what it
//     thinks, not just borrow its browser.
//
// MCP is JSON-RPC 2.0 over a stream, so this is the standard library and
// nothing else: newline-delimited JSON on stdin and stdout
// (`radixnet-count mcp`), which is how MCP's stdio transport works.
// initialize, tools/list and tools/call are implemented, plus ping and the
// notifications/* a client sends and expects no answer to.
//
// Anything written to stdout that is not a response would corrupt the stream,
// so nothing here prints: the log goes to stderr.
//
// The Go twin of the Python radixnet.mcp: the same protocol version, the same
// tool names and schemas, the same error codes, and the same answers - so a
// client cannot tell which implementation it is speaking to.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// MCPProtocolVersion is the MCP revision this server speaks; a client may ask
// for another, and theirs is echoed when it is newer.
const MCPProtocolVersion = "2024-11-05"

// MCPServerName is what the server calls itself in serverInfo.
const MCPServerName = "radixnet"

// The JSON-RPC error codes (the spec's, plus MCP's use of them).
const (
	MCPParseError     = -32700
	MCPInvalidRequest = -32600
	MCPMethodNotFound = -32601
	MCPInvalidParams  = -32602
	MCPInternalError  = -32603
)

// MCPMessage is one JSON-RPC request as it arrives.
//
// ID is json.RawMessage rather than any because JSON-RPC lets it be a string,
// a number or null, and a response must echo it back exactly as it came - a
// number that went through float64 would come back as 1 or 1e+06 rather than
// the digits the client sent.
type MCPMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

// MCPError is the error member of a failed response.
type MCPError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// MCPResponse is one JSON-RPC response: Result or Error, never both.
type MCPResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *MCPError       `json:"error,omitempty"`
}

// MCPServer is JSON-RPC 2.0 over a byte stream - MCP's stdio transport - with
// one toolbox behind it.
//
// Handle turns one request into one response (or nil for a notification, which
// must not be answered), so the protocol can be exercised without any streams
// at all, which is how it is tested.
type MCPServer struct {
	Box          *ToolBox
	Name         string
	Version      string
	Instructions string
	Initialized  bool
	Calls        int
}

// NewMCPServer is a server over box.  An empty toolbox is refused: an MCP
// server with no tools is of no use.
func NewMCPServer(box *ToolBox, name, version, instructions string) (*MCPServer, error) {
	if box == nil || box.Len() == 0 {
		return nil, fmt.Errorf("the toolbox is empty: an MCP server with no tools is of no use")
	}
	if name == "" {
		name = MCPServerName
	}
	if version == "" {
		version = Version
	}
	if instructions == "" {
		instructions = "The tools of a RadixCyclicNN instance: browsing and a calculator, plus the " +
			"network's own predictions, generations, scores and judgements."
	}
	return &MCPServer{Box: box, Name: name, Version: version, Instructions: instructions}, nil
}

// Handle answers one message; nil means "a notification, say nothing".
func (s *MCPServer) Handle(raw []byte) *MCPResponse {
	var message MCPMessage
	if err := json.Unmarshal(raw, &message); err != nil || message.JSONRPC != "2.0" {
		return mcpFail(nil, MCPInvalidRequest, "not a JSON-RPC 2.0 message")
	}
	if message.Method == "" {
		return mcpFail(message.ID, MCPInvalidRequest, "no method")
	}
	params, err := mcpParams(message.Params)
	if err != nil {
		return mcpFail(message.ID, MCPInvalidParams, err.Error())
	}
	// no id is a notification: acted on, never answered
	if len(message.ID) == 0 || string(message.ID) == "null" {
		if message.Method == "notifications/initialized" {
			s.Initialized = true
		}
		return nil
	}
	var result any
	switch message.Method {
	case "initialize":
		result = s.initialize(params)
	case "ping":
		result = map[string]any{}
	case "tools/list":
		result = map[string]any{"tools": s.toolSchemas()}
	case "tools/call":
		out, callErr := s.call(params)
		if callErr != nil {
			return mcpFail(message.ID, MCPInvalidParams, callErr.Error())
		}
		result = out
	default:
		// %q would quote it Go's way; Python renders the name with !r, and the
		// two servers' answers are identical down to the error text
		return mcpFail(message.ID, MCPMethodNotFound, fmt.Sprintf("unknown method '%s'", message.Method))
	}
	return &MCPResponse{JSONRPC: "2.0", ID: message.ID, Result: result}
}

func (s *MCPServer) initialize(params map[string]any) map[string]any {
	s.Initialized = true
	version := MCPProtocolVersion
	if asked, ok := params["protocolVersion"].(string); ok && asked != "" {
		version = asked
	}
	return map[string]any{
		"protocolVersion": version,
		"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
		"serverInfo":      map[string]any{"name": s.Name, "version": s.Version},
		"instructions":    s.Instructions,
	}
}

// toolSchemas is every tool in MCP's shape: inputSchema rather than
// function.parameters.
func (s *MCPServer) toolSchemas() []map[string]any {
	tools := s.Box.Tools()
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		function, _ := tool.Schema()["function"].(map[string]any)
		out = append(out, map[string]any{
			"name": tool.Name, "description": tool.Description, "inputSchema": function["parameters"],
		})
	}
	return out
}

// call runs one tool. A tool that *failed* is a result with isError, not a
// protocol error: the client shows it to its model. Only a call that could not
// be made at all (no name, unknown tool) is an error of the protocol.
func (s *MCPServer) call(params map[string]any) (map[string]any, error) {
	name, _ := params["name"].(string)
	if name == "" {
		return nil, fmt.Errorf("tools/call needs a tool 'name'")
	}
	arguments := map[string]any{}
	if given, ok := params["arguments"]; ok && given != nil {
		typed, ok := given.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("'arguments' must be an object")
		}
		arguments = typed
	}
	if !s.Box.Has(name) {
		return nil, fmt.Errorf("unknown tool '%s' (have: %s)", name, strings.Join(s.Box.Names(), ", "))
	}
	result := s.Box.Call(name, arguments)
	s.Calls++
	text := result.Output
	if !result.OK {
		text = "ERROR: " + result.Error
	}
	return map[string]any{
		"content": []any{map[string]any{"type": "text", "text": text}}, "isError": !result.OK,
	}, nil
}

// Run reads newline-delimited JSON requests and writes the responses until the
// input ends. A message that cannot be handled is answered and the session
// carries on; only the end of the stream ends it.
func (s *MCPServer) Run(in io.Reader, out io.Writer, log io.Writer) error {
	reader := bufio.NewReader(in)
	writer := bufio.NewWriter(out)
	for {
		line, err := reader.ReadBytes('\n')
		trimmed := strings.TrimSpace(string(line))
		if trimmed != "" {
			var response *MCPResponse
			if !json.Valid([]byte(trimmed)) {
				response = mcpFail(nil, MCPParseError, "invalid JSON")
			} else {
				response = s.Handle([]byte(trimmed))
			}
			if response != nil {
				if writeErr := mcpWrite(writer, response); writeErr != nil {
					if log != nil {
						fmt.Fprintf(log, "radixnet mcp: %v\n", writeErr)
					}
					return writeErr
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// mcpWrite sends one response as a single line and flushes it: a client is
// waiting on that newline.
func mcpWrite(writer *bufio.Writer, response *MCPResponse) error {
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(response); err != nil {
		return err
	}
	return writer.Flush()
}

func mcpFail(id json.RawMessage, code int, message string) *MCPResponse {
	if len(id) == 0 {
		id = json.RawMessage("null")
	}
	return &MCPResponse{JSONRPC: "2.0", ID: id, Error: &MCPError{Code: code, Message: message}}
}

// mcpParams reads the params member. JSON-RPC allows positional params; MCP
// never uses them, and an empty list means none.
func mcpParams(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("params is not JSON: %v", err)
	}
	switch typed := value.(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return typed, nil
	case []any:
		if len(typed) == 0 {
			return map[string]any{}, nil
		}
	}
	return nil, fmt.Errorf("params must be an object")
}

// MCPModelOptions is what the model tools are built over. Each is optional and
// what it enables is left out when it is missing.
type MCPModelOptions struct {
	Model    *Model      // the network itself; nil offers the external tools only
	Negative *Model      // the negative network, which enables radixnet_judge
	Box      *ToolBox    // the tools radixnet_solve may call
	Client   LLMClient   // the LLM radixnet_solve needs for criteria and judging
	Config   AgentConfig // how radixnet_solve runs one task
	Prefix   string      // the tool-name prefix; "" means radixnet_
}

// MCPModelTools is the network's own operations as tools.
//
// Model is the network; Negative enables judge; Box and Client together enable
// solve. Each is left out when what it needs is missing, so an offline
// instance offers exactly the four tools it can honour rather than failing on
// the fifth.
func MCPModelTools(o MCPModelOptions) []*Tool {
	if o.Model == nil {
		return nil
	}
	prefix := o.Prefix
	if prefix == "" {
		prefix = "radixnet_"
	}
	model := o.Model

	predict := func(arguments map[string]any) (string, map[string]any, error) {
		length := mcpInt(arguments["length"], 40)
		found, err := model.Predict(mcpString(arguments["prefix_text"]), PredictOptions{
			Length: length, Mode: mcpString(arguments["mode"]), Temperature: mcpFloat(arguments["temperature"], 1),
			MaxLength: max(length, 1),
		})
		if err != nil {
			return "", nil, err
		}
		return found.FullText, nil, nil
	}

	generate := func(arguments map[string]any) (string, map[string]any, error) {
		results, err := model.Generate(GenerateOptions{
			Count: mcpInt(arguments["count"], 3), MaxLength: mcpInt(arguments["max_length"], 80),
			Mode: "sample", Temperature: mcpFloat(arguments["temperature"], 1),
		})
		if err != nil {
			return "", nil, err
		}
		lines := make([]string, 0, len(results))
		for i, result := range results {
			lines = append(lines, fmt.Sprintf("%d. %s", i+1, result.Text))
		}
		if len(lines) == 0 {
			return "(the network generated nothing)", nil, nil
		}
		return strings.Join(lines, "\n"), nil, nil
	}

	score := func(arguments map[string]any) (string, map[string]any, error) {
		data := model.Score(mcpString(arguments["text"]))
		return fmt.Sprintf("log probability %.3f, %.4f per character over %d characters (%d unknown transitions)",
			data.LogProb, data.PerChar, data.Chars, data.UnknownTransitions), nil, nil
	}

	stats := func(map[string]any) (string, map[string]any, error) {
		raw, err := json.MarshalIndent(model.Stats(), "", "  ")
		if err != nil {
			return "", nil, err
		}
		return string(raw), nil, nil
	}

	tools := []*Tool{
		{Name: prefix + "predict", Description: "Continue a prefix with the RadixCyclicNN network.",
			Params: []Param{
				{Name: "prefix_text", Type: "string", Description: "the text to continue", Required: true},
				{Name: "length", Type: "integer", Description: "characters to add", Default: 40},
				{Name: "mode", Type: "string", Description: "dijkstra (cheapest path), beam or sample",
					Default: "dijkstra", Enum: []string{"dijkstra", "beam", "sample"}},
				{Name: "temperature", Type: "number", Description: "sampling temperature", Default: 1.0},
			}, Handler: predict},
		{Name: prefix + "generate", Description: "Generate whole texts from the network.",
			Params: []Param{
				{Name: "count", Type: "integer", Description: "how many", Default: 3},
				{Name: "max_length", Type: "integer", Description: "characters per text", Default: 80},
				{Name: "temperature", Type: "number", Description: "sampling temperature", Default: 1.0},
			}, Handler: generate},
		{Name: prefix + "score",
			Description: "How likely the network thinks a text is (log probability per character).",
			Params: []Param{
				{Name: "text", Type: "string", Description: "the text to score", Required: true},
			}, Handler: score},
		{Name: prefix + "stats",
			Description: "The network's size, compression, training history and backend.", Handler: stats},
	}

	if o.Negative != nil {
		negative := o.Negative
		judge := func(arguments map[string]any) (string, map[string]any, error) {
			verdict := negative.Judge(mcpString(arguments["text"]), JudgeOptions{})
			reasons := make([]string, 0, len(verdict.Reasons))
			for _, reason := range verdict.Reasons {
				reasons = append(reasons, fmt.Sprintf("%s (%.1f)", reason.Reason, reason.Blame))
			}
			listed := strings.Join(reasons, ", ")
			if listed == "" {
				listed = "none"
			}
			fragments := make([]string, 0, 3)
			for _, span := range verdict.Spans {
				if len(fragments) == 3 {
					break
				}
				fragments = append(fragments, span.Fragment)
			}
			out := fmt.Sprintf("%s: %s\nrisk %.2f, coverage %.2f\nreasons: %s",
				verdict.Verdict, verdict.Why, verdict.Risk, verdict.Coverage, listed)
			if len(fragments) > 0 {
				out += "\nworst fragments: " + strings.Join(fragments, "; ")
			}
			return out, nil, nil
		}
		tools = append(tools, &Tool{Name: prefix + "judge",
			Description: "Ask the negative network whether a text looks like something that has gone wrong " +
				"before, and why.",
			Params: []Param{
				{Name: "text", Type: "string", Description: "the text to judge", Required: true},
			}, Handler: judge})
	}

	if o.Box != nil && o.Client != nil {
		// Negative is deliberately not passed on: the Go AgentTrainer has no
		// negative network to hand it to (agent.go), so solve judges with the
		// LLM alone, exactly as `radixnet-count agent` does.
		box, client, config := o.Box, o.Client, o.Config
		solve := func(arguments map[string]any) (string, map[string]any, error) {
			config.MaxSteps = mcpInt(arguments["max_steps"], 4)
			config.ModelAttempts = 1
			config.TeachOnFailure = false
			trainer, err := NewAgentTrainer(model, client, box, config)
			if err != nil {
				return "", nil, err
			}
			task := Task{ID: "mcp", Prompt: mcpString(arguments["task"])}
			criteria, err := trainer.CriteriaFor(task)
			if err != nil {
				return "", nil, err
			}
			var attempt *AgentAttempt
			if mcpString(arguments["source"]) == "teacher" {
				attempt, err = trainer.SolveWithTeacher(task, criteria, 0, "")
			} else {
				attempt, err = trainer.SolveWithModel(task, 0, "", 0)
			}
			if err != nil {
				return "", nil, err
			}
			verdict, err := trainer.Judge(task, criteria, attempt)
			if err != nil {
				return "", nil, err
			}
			attempt.Verdict = verdict
			marks := make([]string, 0, len(criteria))
			for _, one := range criteria {
				marks = append(marks, "  - "+one)
			}
			answer := attempt.Answer
			if answer == "" {
				answer = "(none)"
			}
			decided := "failed"
			if verdict.Correct {
				decided = "correct"
			}
			if verdict.Score != nil {
				decided += fmt.Sprintf(" (score %g)", *verdict.Score)
			}
			return fmt.Sprintf("answer: %s\nverdict: %s\ncriteria:\n%s\n\ntranscript:\n%s",
				answer, decided, strings.Join(marks, "\n"), attempt.Text), nil, nil
		}
		tools = append(tools, &Tool{Name: prefix + "solve",
			Description: "Put one task through the whole agent loop: acceptance criteria, tool calls and a " +
				"judged answer.",
			Params: []Param{
				{Name: "task", Type: "string", Description: "the question to settle", Required: true},
				{Name: "max_steps", Type: "integer", Description: "tool calls allowed", Default: 4},
				{Name: "source", Type: "string",
					Description: "model (the network attempts it) or teacher (the LLM demonstrates)",
					Default:     "model", Enum: []string{"model", "teacher"}},
			}, Handler: solve, Network: true})
	}
	return tools
}

// ServeMCPStdio registers extra on box and serves MCP on the two streams until
// the client closes the input.
func ServeMCPStdio(box *ToolBox, extra []*Tool, in io.Reader, out, log io.Writer) error {
	for _, tool := range extra {
		if err := box.Register(tool); err != nil {
			return err
		}
	}
	server, err := NewMCPServer(box, "", "", "")
	if err != nil {
		return err
	}
	return server.Run(in, out, log)
}

// mcpString is value as a string ("" when it is not one).
func mcpString(value any) string {
	text, _ := value.(string)
	return text
}

// mcpInt is value as an int, or fallback when it is missing or not a number.
func mcpInt(value any, fallback int) int {
	switch typed := value.(type) {
	case float64:
		return int(typed)
	case int:
		return typed
	case json.Number:
		number, err := typed.Float64()
		if err == nil {
			return int(number)
		}
	}
	return fallback
}

// mcpFloat is value as a float64, or fallback when it is missing or not a number.
func mcpFloat(value any, fallback float64) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case int:
		return float64(typed)
	case json.Number:
		number, err := typed.Float64()
		if err == nil {
			return number
		}
	}
	return fallback
}
