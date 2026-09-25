package radixnet

// The MCP server without any streams: Handle turns one request into one
// response, which is the whole protocol.  The answers here are the answers
// radixnet/mcp.py gives to the same messages - the point of the port is that a
// client cannot tell which implementation it is speaking to.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// mcpModel is a small trained network for the model tools.
func mcpModel(t *testing.T) *Model {
	t.Helper()
	model, err := NewModel(0, DefaultGraphOptions())
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	model.Exact = true // atomic counters: the racy default is deliberate, but the race detector runs here
	if _, err := model.Train([]string{
		"the cat sat on the mat", "the cat sat on the log", "the dog sat on the mat",
	}, TrainOptions{Epochs: 2, AutoCompress: true}); err != nil {
		t.Fatalf("train: %v", err)
	}
	return model
}

// ask sends one message and returns the decoded response (nil for a notification).
func ask(t *testing.T, server *MCPServer, message string) map[string]any {
	t.Helper()
	response := server.Handle([]byte(message))
	if response == nil {
		return nil
	}
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return decoded
}

// result is the result member, or a failure naming the error that came instead.
func result(t *testing.T, response map[string]any) map[string]any {
	t.Helper()
	if response == nil {
		t.Fatal("no response where one was due")
	}
	if problem, ok := response["error"]; ok {
		t.Fatalf("an error where a result was due: %v", problem)
	}
	out, _ := response["result"].(map[string]any)
	return out
}

// errorOf is the error member's code and message.
func errorOf(t *testing.T, response map[string]any) (int, string) {
	t.Helper()
	if response == nil {
		t.Fatal("no response where an error was due")
	}
	problem, ok := response["error"].(map[string]any)
	if !ok {
		t.Fatalf("a result where an error was due: %v", response["result"])
	}
	code, _ := problem["code"].(float64)
	message, _ := problem["message"].(string)
	return int(code), message
}

func newMCP(t *testing.T, tools ...*Tool) *MCPServer {
	t.Helper()
	box := NewToolBox(CalculatorTool())
	for _, tool := range tools {
		if err := box.Register(tool); err != nil {
			t.Fatalf("register %s: %v", tool.Name, err)
		}
	}
	server, err := NewMCPServer(box, "", "", "")
	if err != nil {
		t.Fatalf("new server: %v", err)
	}
	return server
}

func TestMCPHandshake(t *testing.T) {
	server := newMCP(t)
	if server.Initialized {
		t.Fatal("initialized before anything was said")
	}
	got := result(t, ask(t, server, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`))
	if got["protocolVersion"] != MCPProtocolVersion {
		t.Fatalf("protocolVersion = %v, want %s", got["protocolVersion"], MCPProtocolVersion)
	}
	info, _ := got["serverInfo"].(map[string]any)
	if info["name"] != MCPServerName || info["version"] != Version {
		t.Fatalf("serverInfo = %v", info)
	}
	if !server.Initialized {
		t.Fatal("initialize did not initialize the server")
	}

	// a client on a newer revision gets its own echoed back
	got = result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`))
	if got["protocolVersion"] != "2025-06-18" {
		t.Fatalf("the client's revision was not echoed: %v", got["protocolVersion"])
	}

	// ping answers an empty object, not an error
	if len(result(t, ask(t, server, `{"jsonrpc":"2.0","id":3,"method":"ping"}`))) != 0 {
		t.Fatal("ping answered something")
	}
}

func TestMCPNotificationIsNeverAnswered(t *testing.T) {
	server := newMCP(t)
	for _, message := range []string{
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":null,"method":"notifications/cancelled","params":{"requestId":1}}`,
	} {
		if response := server.Handle([]byte(message)); response != nil {
			t.Fatalf("a notification was answered: %v", response)
		}
	}
	if !server.Initialized {
		t.Fatal("notifications/initialized did not initialize the server")
	}
}

func TestMCPToolsListIsTheToolboxInMCPShape(t *testing.T) {
	server := newMCP(t)
	listed := result(t, ask(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`))
	tools, _ := listed["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("listed %d tools, want 1", len(tools))
	}
	one, _ := tools[0].(map[string]any)
	if one["name"] != "calculator" {
		t.Fatalf("name = %v", one["name"])
	}
	if one["description"] == "" {
		t.Fatal("no description")
	}
	// MCP's inputSchema is the function's parameters, not the whole function
	schema, ok := one["inputSchema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("inputSchema = %v", one["inputSchema"])
	}
	if _, ok := schema["properties"].(map[string]any)["expression"]; !ok {
		t.Fatalf("the calculator's argument is missing: %v", schema["properties"])
	}
	if _, ok := one["function"]; ok {
		t.Fatal("the Ollama shape leaked into the MCP listing")
	}
}

func TestMCPToolsCall(t *testing.T) {
	server := newMCP(t)
	got := result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"2+2*10"}}}`))
	if got["isError"] != false {
		t.Fatalf("isError = %v", got["isError"])
	}
	content, _ := got["content"].([]any)
	first, _ := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "22" {
		t.Fatalf("content = %v", content)
	}
	if server.Calls != 1 {
		t.Fatalf("Calls = %d, want 1", server.Calls)
	}

	// a tool that FAILED is a result with isError, not a protocol error: the
	// client is meant to show it to its model
	got = result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"1/0"}}}`))
	if got["isError"] != true {
		t.Fatalf("a failed tool call was not marked isError: %v", got)
	}
	content, _ = got["content"].([]any)
	first, _ = content[0].(map[string]any)
	if text, _ := first["text"].(string); !strings.HasPrefix(text, "ERROR: ") {
		t.Fatalf("a failed call's text = %q", text)
	}
}

func TestMCPRefusals(t *testing.T) {
	server := newMCP(t)
	for _, one := range []struct {
		message string
		code    int
		says    string
	}{
		{`{"jsonrpc":"2.0","id":1,"method":"sideways"}`, MCPMethodNotFound, "unknown method 'sideways'"},
		{`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"arguments":{}}}`,
			MCPInvalidParams, "needs a tool 'name'"},
		{`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"nope"}}`,
			MCPInvalidParams, "unknown tool 'nope'"},
		{`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"calculator","arguments":7}}`,
			MCPInvalidParams, "'arguments' must be an object"},
		{`{"jsonrpc":"2.0","id":5,"method":"ping","params":7}`, MCPInvalidParams, "params must be an object"},
		{`{"jsonrpc":"1.0","id":6,"method":"ping"}`, MCPInvalidRequest, "not a JSON-RPC 2.0 message"},
		{`{"jsonrpc":"2.0","id":7}`, MCPInvalidRequest, "no method"},
	} {
		code, message := errorOf(t, ask(t, server, one.message))
		if code != one.code || !strings.Contains(message, one.says) {
			t.Fatalf("%s -> (%d, %q), want (%d, ...%s...)", one.message, code, message, one.code, one.says)
		}
	}
	// an empty toolbox is refused outright
	if _, err := NewMCPServer(NewToolBox(), "", "", ""); err == nil {
		t.Fatal("an MCP server was built over an empty toolbox")
	}
}

func TestMCPEchoesTheIDExactly(t *testing.T) {
	server := newMCP(t)
	// JSON-RPC lets an id be a string or a number, and a large number must come
	// back as the digits the client sent rather than through a float
	for _, id := range []string{`"abc"`, `12345678901234567`, `0`} {
		response := server.Handle([]byte(`{"jsonrpc":"2.0","id":` + id + `,"method":"ping"}`))
		if response == nil {
			t.Fatalf("id %s went unanswered", id)
		}
		if string(response.ID) != id {
			t.Fatalf("id came back as %s, want %s", response.ID, id)
		}
	}
}

func TestMCPModelTools(t *testing.T) {
	model := mcpModel(t)
	tools := MCPModelTools(MCPModelOptions{Model: model})
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	want := []string{"radixnet_predict", "radixnet_generate", "radixnet_score", "radixnet_stats"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("tools = %v, want %v", names, want)
	}
	// no model at all: no model tools, and the external ones still stand
	if got := MCPModelTools(MCPModelOptions{}); got != nil {
		t.Fatalf("tools without a model: %v", got)
	}

	server := newMCP(t, tools...)
	got := result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"radixnet_predict","arguments":{"prefix_text":"the cat","length":8}}}`))
	content, _ := got["content"].([]any)
	first, _ := content[0].(map[string]any)
	text, _ := first["text"].(string)
	if !strings.HasPrefix(text, "the cat") {
		t.Fatalf("predict answered %q", text)
	}

	got = result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"radixnet_score","arguments":{"text":"the cat sat on the mat"}}}`))
	content, _ = got["content"].([]any)
	first, _ = content[0].(map[string]any)
	if text, _ = first["text"].(string); !strings.Contains(text, "log probability") {
		t.Fatalf("score answered %q", text)
	}

	got = result(t, ask(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"radixnet_stats"}}`))
	content, _ = got["content"].([]any)
	first, _ = content[0].(map[string]any)
	text, _ = first["text"].(string)
	var stats map[string]any
	if err := json.Unmarshal([]byte(text), &stats); err != nil {
		t.Fatalf("stats is not JSON: %v", err)
	}
	if stats["kind"] != "count" {
		t.Fatalf("stats kind = %v", stats["kind"])
	}

	// generate asks for whole texts and numbers them
	got = result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"radixnet_generate","arguments":{"count":2,"max_length":30}}}`))
	content, _ = got["content"].([]any)
	first, _ = content[0].(map[string]any)
	if text, _ = first["text"].(string); !strings.HasPrefix(text, "1. ") {
		t.Fatalf("generate answered %q", text)
	}
}

func TestMCPJudgeArrivesWithTheNegativeNetwork(t *testing.T) {
	model := mcpModel(t)
	negative, err := NewNegativeModel(0, DefaultNegativeOptions())
	if err != nil {
		t.Fatalf("negative model: %v", err)
	}
	if _, err := negative.Blame([]string{"qqqq zzzz xxxx"}, BlameOptions{Reason: "gibberish", Severity: 3}); err != nil {
		t.Fatalf("blame: %v", err)
	}
	tools := MCPModelTools(MCPModelOptions{Model: model, Negative: negative})
	if len(tools) != 5 || tools[4].Name != "radixnet_judge" {
		t.Fatalf("judge is missing: %d tools", len(tools))
	}
	server := newMCP(t, tools...)
	got := result(t, ask(t, server,
		`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"radixnet_judge","arguments":{"text":"qqqq zzzz xxxx"}}}`))
	content, _ := got["content"].([]any)
	first, _ := content[0].(map[string]any)
	text, _ := first["text"].(string)
	for _, want := range []string{"risk", "coverage", "reasons:"} {
		if !strings.Contains(text, want) {
			t.Fatalf("the verdict does not mention %q: %q", want, text)
		}
	}
}

func TestMCPRunReadsAStreamAndSurvivesRubbish(t *testing.T) {
	server := newMCP(t)
	in := strings.NewReader(strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"ping"}`,
		``, // a blank line is not a message
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`, // no answer
		`not json at all`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"6*7"}}}`,
	}, "\n") + "\n")
	var out bytes.Buffer
	if err := server.Run(in, &out, nil); err != nil {
		t.Fatalf("run: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("wrote %d lines, want 3 (ping, the parse error, the call):\n%s", len(lines), out.String())
	}
	if !strings.Contains(lines[1], "-32700") {
		t.Fatalf("rubbish was not a parse error: %s", lines[1])
	}
	if !strings.Contains(lines[2], `"42"`) {
		t.Fatalf("the call after the rubbish was not answered: %s", lines[2])
	}
	// a stream that ends without a final newline is still a stream
	out.Reset()
	if err := server.Run(strings.NewReader(`{"jsonrpc":"2.0","id":9,"method":"ping"}`), &out, nil); err != nil {
		t.Fatalf("run without a trailing newline: %v", err)
	}
	if !strings.Contains(out.String(), `"id":9`) {
		t.Fatalf("the last line went unanswered: %q", out.String())
	}
}
