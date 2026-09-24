package radixnet

import (
	"encoding/json"
	"strings"
	"testing"
)

// The model in today's format: the requests of both dialects, the reply as a
// turn with the search's trace as its thinking, the text streamed one node at
// a time, the two documents and the two streams (assistant.go).

func askBody(t *testing.T, text string) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal([]byte(text), &body); err != nil {
		t.Fatal(err)
	}
	return body
}

func TestAssistantParsesBothDialects(t *testing.T) {
	ask, err := ParseOpenAI(askBody(t, `{"model": "radixnet-count", "messages": [
		{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"},
		{"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": "{\"expression\": \"2+2\"}"}}]},
		{"role": "tool", "content": "4"}],
		"max_tokens": 30, "stop": "x", "n": 2, "stream": true, "stream_options": {"include_usage": true}, "mode": "sample", "seed": 3}`))
	if err != nil {
		t.Fatal(err)
	}
	if ask.System != "be brief" || ask.MaxTokens != 30 || ask.N != 2 || !ask.Stream || !ask.IncludeUsage || ask.Mode != "sample" || ask.Seed == nil || *ask.Seed != 3 {
		t.Fatalf("misread: %+v", ask)
	}
	if got := ask.Messages[1].Text; got != `<tool>calc {"expression": "2+2"}</tool>` {
		t.Fatalf("tool call rendered as %q", got)
	}
	if got := ask.Messages[2].Text; got != "<result>4</result>" || ask.Messages[2].Role != "tool" {
		t.Fatalf("tool result rendered as %q (%s)", got, ask.Messages[2].Role)
	}
	if ask.Stop[0] != "x" || !ask.Thinking {
		t.Fatalf("stop / thinking misread: %+v", ask)
	}
	ask, err = ParseAnthropic(askBody(t, `{"system": [{"type": "text", "text": "be brief"}], "messages": [
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "yo"}]}],
		"thinking": {"type": "disabled"}, "stop_sequences": ["!"]}`))
	if err != nil {
		t.Fatal(err)
	}
	if ask.System != "be brief" || !ask.Prefill() || ask.Previous() != "yo" || ask.Thinking || ask.Stop[0] != "!" {
		t.Fatalf("misread: %+v", ask)
	}
	for _, bad := range []struct{ body, param, message string }{
		{`{"messages": "no"}`, "messages", "messages must be a list of {role, content} objects"},
		{`{"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}`, "max_tokens", "max_tokens must be >= 1 (got 0)"},
		{`{"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}`, "messages", "messages[0].content[0]: content of type 'image_url' is not supported; this model reads text"},
		{`{"messages": [{"role": "user", "content": "x"}], "stop": [""]}`, "stop", "stop must not hold an empty sequence"},
	} {
		_, err := ParseOpenAI(askBody(t, bad.body))
		ae, ok := err.(*AskError)
		if !ok || ae.Param != bad.param || ae.Message != bad.message {
			t.Fatalf("%s: got %v", bad.body, err)
		}
	}
	if _, err := ParseAnthropic(askBody(t, `{"messages": [{"role": "system", "content": "x"}]}`)); err == nil ||
		err.Error() != "messages[0].role must be user or assistant (got 'system')" {
		t.Fatalf("a system role in a Messages list: %v", err)
	}
	if KindOfID("RadixNet-Count") != "count" || KindOfID("radixnet") != "" || KindOfID("") != "" {
		t.Fatal("model ids")
	}
}

func TestAssistantReplyIsATurnAndItsThinkingTheTrace(t *testing.T) {
	m := trained(t, 2, 0)
	ask, err := ParseOpenAI(askBody(t, `{"messages": [{"role": "user", "content": "tell me about the cat"}], "learn": false}`))
	if err != nil {
		t.Fatal(err)
	}
	seen := []map[string]any{}
	reply, err := Respond(m, nil, ask, "", func(e map[string]any) { seen = append(seen, e) })
	if err != nil {
		t.Fatal(err)
	}
	choice := reply.Choices[0]
	if choice.Turn == nil || choice.Text != choice.Turn.Text {
		t.Fatalf("the text is the turn's: %+v", choice)
	}
	lines := strings.Split(choice.Thinking, "\n")
	if lines[0] != `answering "tell me about the cat"` {
		t.Fatalf("the thinking opens with the line answered: %q", lines[0])
	}
	if !strings.HasPrefix(lines[len(lines)-1], "saying "+q(choice.Text)+": cost ") {
		t.Fatalf("the thinking ends with what it said: %q", lines[len(lines)-1])
	}
	text, thinking := "", ""
	for _, e := range seen {
		switch e["type"] {
		case "text":
			text += e["text"].(string)
		case "thinking":
			thinking += e["text"].(string)
		}
	}
	if text != choice.Text || thinking != choice.Thinking {
		t.Fatalf("the events join into the reply: %q / %q", text, thinking)
	}
	if seen[0]["type"] != "start" || seen[len(seen)-1]["type"] != "end" || seen[len(seen)-2]["type"] != "done" {
		t.Fatalf("the events start, end and finish: %v", seen)
	}
	if reply.InputUnits != len("tell me about the cat") || choice.OutputUnits != len(choice.Text)+len(choice.Thinking) {
		t.Fatalf("usage counts characters: %d / %d", reply.InputUnits, choice.OutputUnits)
	}
	// one piece per node of the walk, the later ones what each node adds
	pieces := Deltas(m.Encoding(), choice.Turn.Labels, choice.Turn.NodeIDs, choice.Text)
	if strings.Join(pieces, "") != choice.Text {
		t.Fatalf("the pieces join into the text: %v", pieces)
	}
	doc := ToOpenAI(reply)
	message := doc["choices"].([]map[string]any)[0]["message"].(map[string]any)
	if doc["object"] != "chat.completion" || message["content"] != choice.Text || message["reasoning_content"] != choice.Thinking {
		t.Fatalf("chat.completion: %v", doc)
	}
	blocks := ToAnthropic(reply)["content"].([]map[string]any)
	if len(blocks) != 2 || blocks[0]["type"] != "thinking" || blocks[1]["text"] != choice.Text {
		t.Fatalf("message blocks: %v", blocks)
	}
	// what was said is heard, and not said again; a system prompt is acknowledged and not read
	again, err := Respond(m, nil, mustAsk(t, `{"messages": [{"role": "system", "content": "in French"}, {"role": "user", "content": "tell me about the cat"},
		{"role": "assistant", "content": `+q(choice.Text)+`}, {"role": "user", "content": "tell me about the cat"}], "learn": false}`), "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if again.Choices[0].Text == choice.Text {
		t.Fatal("what was said came back")
	}
	if !strings.Contains(again.Choices[0].Thinking, "(2 earlier line(s) heard)") ||
		!strings.Contains(again.Choices[0].Thinking, "a system prompt was given") {
		t.Fatalf("the thinking: %s", again.Choices[0].Thinking)
	}
	// a stop sequence cuts the reply
	seq := choice.Text[3:5]
	cut, err := Respond(m, nil, mustAsk(t, `{"messages": [{"role": "user", "content": "tell me about the cat"}], "learn": false, "stop": [`+q(seq)+`]}`), "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if cut.Choices[0].Text != choice.Text[:3] || cut.Choices[0].StopReason != StopStopSequence || *cut.Choices[0].StopSequence != seq {
		t.Fatalf("the stop sequence: %+v", cut.Choices[0])
	}
	// nothing to say
	empty, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	none, err := Respond(empty, nil, mustAsk(t, `{"messages": [{"role": "user", "content": "anything"}]}`), "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if none.Choices[0].Text != "" || none.Choices[0].Turn != nil || !strings.HasSuffix(none.Choices[0].Thinking, "nothing to say: the graph has no way on from here") {
		t.Fatalf("nothing to say: %+v", none.Choices[0])
	}
}

func mustAsk(t *testing.T, text string) *Ask {
	t.Helper()
	ask, err := ParseOpenAI(askBody(t, text))
	if err != nil {
		t.Fatal(err)
	}
	return ask
}

func TestAssistantDeltasAndTools(t *testing.T) {
	enc := DefaultEncoding()
	got := Deltas(enc, []string{"<s>", "the", "he ", "e c", " ca", "cat", "</s>"}, []int{0, 3, 4, 5, 6, 7, 1}, "the cat")
	if strings.Join(got, "|") != "the| |c|a|t" {
		t.Fatalf("deltas: %v", got)
	}
	got = Deltas(enc, []string{"the", "he ", "e cat"}, []int{3, 4, 5}, "xthe cat")
	if strings.Join(got, "|") != "xthe| |cat" {
		t.Fatalf("a merged node adds its whole tail: %v", got)
	}
	if got = Deltas(enc, []string{"the", "he ", "e cat"}, []int{3, 4, 5}, "the ca"); len(got) != 1 {
		t.Fatalf("a cut text is one piece: %v", got)
	}
	words := Encoding{Unit: Words, N: 2, Stride: 1}
	got = Deltas(words, []string{"<s>", "the cat", "cat sat", "sat on"}, []int{0, 3, 4, 5}, "the cat sat on")
	if strings.Join(got, "|") != "the cat| sat| on" {
		t.Fatalf("word deltas: %v", got)
	}
	// the offered tools read a call's arguments the way the agent does
	box := OfferedToolBox([]OfferedTool{{Name: "calc", Parameters: map[string]any{
		"properties": map[string]any{"expression": map[string]any{"type": "string"}}, "required": []any{"expression"}}}})
	call := ParseCall("<tool>calc 2+2</tool>", box)
	if call == nil || call.Error != "" || call.Arguments["expression"] != "2+2" {
		t.Fatalf("a bare value is the one argument: %+v", call)
	}
}

func TestAssistantStreams(t *testing.T) {
	m := trained(t, 2, 0)
	ask := mustAsk(t, `{"messages": [{"role": "user", "content": "the cat sat"}], "learn": false}`)
	seen := []map[string]any{}
	reply, err := Respond(m, nil, ask, "", func(e map[string]any) { seen = append(seen, e) })
	if err != nil {
		t.Fatal(err)
	}
	chat := &OpenAIStream{Token: reply.Token, Model: reply.ModelName, Created: reply.Created, IncludeUsage: true, Thinking: true}
	frames := []string{}
	for _, e := range seen {
		frames = append(frames, chat.Frames(e)...)
	}
	if frames[len(frames)-1] != "data: [DONE]\n\n" || !strings.Contains(frames[len(frames)-2], `"usage"`) {
		t.Fatalf("the chunks end in the usage and [DONE]: %v", frames[len(frames)-2:])
	}
	if !strings.Contains(frames[0], `"delta":{"content":"","role":"assistant"}`) {
		t.Fatalf("the first chunk names the role: %s", frames[0])
	}
	messages := NewAnthropicStream(reply.Token, reply.ModelName, reply.Created, reply.InputUnits, true)
	frames = frames[:0]
	for _, e := range seen {
		frames = append(frames, messages.Frames(e)...)
	}
	if !strings.HasPrefix(frames[0], "event: message_start\n") || !strings.HasPrefix(frames[len(frames)-1], "event: message_stop\n") {
		t.Fatalf("the events open and close the message: %v", frames)
	}
	starts := 0
	for _, f := range frames {
		if strings.HasPrefix(f, "event: content_block_start") {
			starts++
		}
	}
	if starts != 2 {
		t.Fatalf("a thinking block and a text block: %d", starts)
	}
	if doc := ShapeV1Error(FormatOpenAI, 404, "no", "model"); doc["error"].(map[string]any)["code"] != "model_not_found" {
		t.Fatalf("the error shape: %v", doc)
	}
	if doc := ShapeV1Error(FormatAnthropic, 400, "no", ""); doc["error"].(map[string]any)["type"] != "invalid_request_error" {
		t.Fatalf("the error shape: %v", doc)
	}
}

func TestReplyTraceHearsEveryStep(t *testing.T) {
	m := trained(t, 2, 0)
	kinds := []string{}
	o := DefaultReplyOptions()
	o.Learn = false
	o.Trace = func(e map[string]any) { kinds = append(kinds, e["kind"].(string)) }
	turn, err := m.Reply("the cat sat on the mat", o)
	if err != nil || turn == nil {
		t.Fatalf("no reply: %v", err)
	}
	if kinds[0] != "context" {
		t.Fatalf("the first event is the context tried: %v", kinds)
	}
	saw := map[string]bool{}
	for _, k := range kinds {
		saw[k] = true
	}
	if !saw["candidates"] || !saw["pick"] {
		t.Fatalf("candidates and a pick: %v", kinds)
	}
	o.Trace = nil
	again, err := m.Reply("the cat sat on the mat", o)
	if err != nil || again.Text != turn.Text {
		t.Fatalf("without a trace nothing changes: %v %q", err, again.Text)
	}
}
