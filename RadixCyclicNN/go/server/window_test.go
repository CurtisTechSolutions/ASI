package server

import (
	"reflect"
	"strings"
	"testing"
)

// The dynamic window through the HTTP API: the same routes and documents as the Python server's.
func TestDynamicWindowRoutes(t *testing.T) {
	e := newEnv(t, false)
	texts := []string{
		"the quick brown fox jumps over the lazy dog while the cat sat on the mat and the bird sang in the tree all afternoon",
		"the quick brown fox jumps over the lazy dog while the cat ran to the door",
		"a bird sang in the tree all afternoon",
	}
	if status, doc := e.post("/api/train", map[string]any{"texts": texts, "epochs": 1}); status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	e.waitJob()
	status, doc := e.get("/api/model/window")
	if status != 200 || doc["kind"] != "count" {
		t.Fatalf("window: %d %v", status, doc)
	}
	window := doc["window"].(map[string]any)
	if window["on"] != false || window["size"] != nil || window["longer"] != nil || window["longest"].(float64) <= 16 {
		t.Fatalf("off: %v", window)
	}
	if status, doc = e.post("/api/model/window/step", map[string]any{}); status != 400 || !strings.Contains(doc["error"].(string), "off") {
		t.Fatalf("a step while off: %d %v", status, doc)
	}
	status, doc = e.post("/api/model/window", map[string]any{"on": true})
	window = doc["window"].(map[string]any)
	if status != 200 || !reflect.DeepEqual(window["sizes"], []any{32.0, 16.0, 8.0, 4.0}) || window["size"] != 32.0 || doc["stats"].(map[string]any)["dynamic_window"] != 32.0 {
		t.Fatalf("on: %d %v", status, doc)
	}
	if _, st := e.get("/api/status"); st["dynamic_window"] != 32.0 {
		t.Fatalf("status: %v", st["dynamic_window"])
	}
	if _, model := e.get("/api/model"); model["dynamic_window"].(map[string]any)["size"] != 32.0 {
		t.Fatalf("model: %v", model["dynamic_window"])
	}
	status, doc = e.post("/api/model/window", map[string]any{"size": 16})
	window = doc["window"].(map[string]any)
	if status != 200 || window["size"] != 16.0 || window["next"] != 8.0 || window["longer"].(float64) <= 0 {
		t.Fatalf("size 16: %d %v", status, doc)
	}
	status, doc = e.post("/api/model/window/step", map[string]any{"steps": 2})
	if status != 200 {
		t.Fatalf("step: %d %v", status, doc)
	}
	step := doc["step"].(map[string]any)
	if !reflect.DeepEqual(step["sizes"], []any{16.0, 8.0}) || doc["window"].(map[string]any)["size"] != 4.0 || step["splits"].(float64) <= 0 {
		t.Fatalf("two steps: %v", doc)
	}
	if step["nodes_after"] != doc["stats"].(map[string]any)["nodes"] {
		t.Fatalf("nodes after the step are the stats' nodes: %v", doc)
	}
	status, doc = e.post("/api/model/window", map[string]any{"top": 16, "floor": 8, "auto": false})
	window = doc["window"].(map[string]any)
	if status != 200 || !reflect.DeepEqual(window["sizes"], []any{16.0, 8.0}) || window["size"] != 8.0 || window["auto"] != false {
		t.Fatalf("a new ladder: %d %v", status, doc)
	}
	for _, body := range []map[string]any{{"top": 12}, {"size": 3}, {"floor": 64}, {"on": "yes"}} {
		if status, _ = e.post("/api/model/window", body); status != 400 {
			t.Fatalf("%v should be refused: %d", body, status)
		}
	}
	if status, _ = e.post("/api/model/window/step", map[string]any{"steps": 0}); status != 400 {
		t.Fatal("steps must be >= 1")
	}
	status, doc = e.post("/api/model/window", map[string]any{"on": false})
	if status != 200 || doc["window"].(map[string]any)["on"] != false || doc["stats"].(map[string]any)["dynamic_window"] != nil {
		t.Fatalf("off: %d %v", status, doc)
	}
}
