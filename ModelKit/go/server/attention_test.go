package server

import (
	"reflect"
	"strings"
	"testing"
)

// The attention band through the HTTP API: the same routes and documents as the Python server's.
func TestAttentionBandRoutes(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.get("/api/model/attention")
	if status != 200 || doc["kind"] != "count" {
		t.Fatalf("attention: %d %v", status, doc)
	}
	want := map[string]any{
		"on": false, "blur": nil, "weights": nil, "ngram": 3.0, "stride": 1.0, "unit": "char", "units": "chars",
		"applies": true, "default_blur": 0.5,
	}
	if !reflect.DeepEqual(doc["attention"], want) {
		t.Fatalf("off: %v", doc["attention"])
	}
	status, doc = e.post("/api/model/attention", map[string]any{"on": true})
	if status != 200 || doc["attention"].(map[string]any)["blur"] != 0.5 || doc["stats"].(map[string]any)["attention_blur"] != 0.5 {
		t.Fatalf("on: %d %v", status, doc)
	}
	status, doc = e.post("/api/model/attention", map[string]any{"blur": 0.2})
	if status != 200 || !reflect.DeepEqual(doc["attention"].(map[string]any)["weights"], []any{0.8, 1.0, 0.8}) {
		t.Fatalf("blur 0.2: %d %v", status, doc)
	}
	if _, st := e.get("/api/status"); st["attention_blur"] != 0.2 {
		t.Fatalf("status: %v", st["attention_blur"])
	}
	if _, model := e.get("/api/model"); model["attention"].(map[string]any)["blur"] != 0.2 {
		t.Fatalf("model: %v", model["attention"])
	}
	status, doc = e.post("/api/model/attention", map[string]any{"blur": 3})
	if status != 400 || !strings.Contains(doc["error"].(string), "[0, 1]") {
		t.Fatalf("blur 3: %d %v", status, doc)
	}
	if status, _ = e.post("/api/model/attention", map[string]any{"on": "yes"}); status != 400 {
		t.Fatalf("on must be a boolean: %d", status)
	}
	status, doc = e.post("/api/model/attention/preview", map[string]any{"wrong": "the cat sat", "right": "the bat sat", "blur": 1})
	if status != 200 || doc["blur"] != 1.0 || !reflect.DeepEqual(doc["weights"], []any{0.0, 1.0, 0.0}) {
		t.Fatalf("preview: %d %v", status, doc)
	}
	charges := doc["wrong"].(map[string]any)["charges"].([]any)
	if !reflect.DeepEqual(charges[2:5], []any{0.0, 1.0, 0.0}) {
		t.Fatalf("preview charges: %v", charges)
	}
	if _, doc = e.get("/api/model/attention"); doc["attention"].(map[string]any)["blur"] != 0.2 {
		t.Fatal("a preview changes nothing")
	}
	status, doc = e.post("/api/model/attention", map[string]any{"on": false, "blur": 0.9})
	if status != 200 || doc["attention"].(map[string]any)["on"] != false || doc["attention"].(map[string]any)["blur"] != nil {
		t.Fatalf("off: %d %v", status, doc)
	}
}
