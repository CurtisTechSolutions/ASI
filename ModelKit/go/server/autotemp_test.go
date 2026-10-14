package server

import (
	"strings"
	"testing"
)

// Auto-temperature over HTTP (../../../RadixCyclicNN/SPEC-SearchAndTraining.md
// §10): /api/predict, /api/generate, /api/converse (and its stream) and both
// /v1 dialects take auto_temperature, a ceiling up to which a search with no
// answer walks again, hotter; the record of the climb goes beside the answer
// only when it is on, and a ceiling that is not a finite number >= 0 is a 400.

// climbEnv is an env whose model knows three lines and whose negative network
// has been taught that the first of them is a failure.
func climbEnv(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, false)
	texts := []string{"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill"}
	if status, doc := e.post("/api/train", map[string]any{"texts": texts, "epochs": 3}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitJob()
	e.blameFailures(map[string]any{"texts": []string{texts[0]}, "reason": "gibberish"})
	return e
}

func TestPredictClimbsPastTheVeto(t *testing.T) {
	e := climbEnv(t)
	body := map[string]any{"prefix": "the cat", "mode": "beam", "k": 1, "length": 30, "to_end": true}
	status, plain := e.post("/api/predict", body)
	if status != 200 {
		t.Fatalf("predict: %d %+v", status, plain)
	}
	if _, present := plain["auto_temperature"]; present || plain["continuation"] != "" {
		t.Fatalf("off: the guard vetoed the only continuation, and there is no record: %+v", plain)
	}
	body["auto_temperature"] = 2
	status, doc := e.post("/api/predict", body)
	if status != 200 {
		t.Fatalf("predict: %d %+v", status, doc)
	}
	record, _ := doc["auto_temperature"].(map[string]any)
	if strings.TrimSpace(doc["continuation"].(string)) == "" || record["climbed"] != true || record["found"] != true {
		t.Fatalf("on: a warmer walk the guard lets through, and the record: %+v", doc)
	}
	guard, before := doc["guard"].(map[string]any), plain["guard"].(map[string]any)
	if toFloat(guard["candidates"]) <= toFloat(before["candidates"]) || toFloat(guard["candidates"]) != toFloat(record["tries"]) {
		t.Fatalf("every search was judged: %+v / %+v", guard["candidates"], record)
	}
}

func TestGenerateAndConverseCarryTheRecord(t *testing.T) {
	e := climbEnv(t)
	status, doc := e.post("/api/generate", map[string]any{"count": 2, "mode": "sample", "max_length": 40, "seed": 2,
		"auto_temperature": 2})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	if record, _ := doc["auto_temperature"].(map[string]any); record == nil || toFloat(record["ceiling"]) != 2 {
		t.Fatalf("the generation's record: %+v", doc)
	}
	status, doc = e.post("/api/generate", map[string]any{"count": 2, "guard": false, "auto_temperature": 2})
	if record, _ := doc["auto_temperature"].(map[string]any); status != 200 || record == nil {
		t.Fatalf("unguarded, the model's own climb: %d %+v", status, doc)
	}
	if status, doc = e.post("/api/generate", map[string]any{"count": 2}); status != 200 || doc["auto_temperature"] != nil {
		t.Fatalf("off, no record: %d %+v", status, doc)
	}
	status, doc = e.post("/api/converse", map[string]any{"opening": "the cat", "turns": 2, "auto_temperature": 2})
	if status != 200 {
		t.Fatalf("converse: %d %+v", status, doc)
	}
	turns := doc["turns"].([]any)
	if len(turns) < 2 {
		t.Fatalf("the voices speak: %+v", turns)
	}
	if _, present := turns[0].(map[string]any)["auto_temperature"]; present {
		t.Fatalf("the opening is given, not searched: %+v", turns[0])
	}
	for _, row := range turns[1:] {
		record, _ := row.(map[string]any)["auto_temperature"].(map[string]any)
		if record == nil || toFloat(record["ceiling"]) != 2 {
			t.Fatalf("every reply carries its climb: %+v", row)
		}
	}
	status, doc = e.post("/api/converse", map[string]any{"opening": "the cat", "turns": 1})
	if turn := doc["turns"].([]any)[1].(map[string]any); status != 200 || turn["auto_temperature"] != nil {
		t.Fatalf("off, a turn is what it always was: %d %+v", status, turn)
	}
}

func TestABadCeilingIsA400(t *testing.T) {
	e := climbEnv(t)
	messages := []map[string]any{{"role": "user", "content": "the cat"}}
	for _, route := range []struct {
		path string
		body map[string]any
	}{
		{"/api/predict", map[string]any{"prefix": "the"}}, {"/api/generate", map[string]any{}},
		{"/api/converse", map[string]any{}}, {"/api/converse/stream", map[string]any{}},
		{"/v1/messages", map[string]any{"messages": messages}}, {"/v1/chat/completions", map[string]any{"messages": messages}},
	} {
		for _, bad := range []any{-1, "hot", true} {
			body := map[string]any{"auto_temperature": bad}
			for k, v := range route.body {
				body[k] = v
			}
			if status, doc := e.post(route.path, body); status != 400 {
				t.Fatalf("%s with %v: %d %+v", route.path, bad, status, doc)
			}
		}
	}
	// Chat Completions names the field to blame
	status, doc := e.post("/v1/chat/completions", map[string]any{"messages": messages, "auto_temperature": -1})
	if inner, _ := doc["error"].(map[string]any); status != 400 || inner["param"] != "auto_temperature" {
		t.Fatalf("the field is named: %d %+v", status, doc)
	}
	// and a number that JSON cannot hold is no ceiling either
	if status, _, _ := e.do("POST", "/api/predict", `{"prefix": "the", "auto_temperature": 1e999}`, nil); status != 400 {
		t.Fatalf("an infinite ceiling: %d", status)
	}
}

func TestTodaysFormatTakesTheCeiling(t *testing.T) {
	e := climbEnv(t)
	status, doc := e.post("/v1/messages", map[string]any{"messages": []map[string]any{{"role": "user", "content": "the cat"}},
		"max_tokens": 20, "auto_temperature": 2, "learn": false})
	if status != 200 {
		t.Fatalf("messages: %d %+v", status, doc)
	}
	choice := doc["radixnet"].(map[string]any)["choices"].([]any)[0].(map[string]any)
	turn, _ := choice["turn"].(map[string]any)
	if turn == nil || turn["auto_temperature"] == nil || len(doc["content"].([]any)) == 0 {
		t.Fatalf("the reply's turn carries its climb: %+v", doc)
	}
}
