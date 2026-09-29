package server

import "testing"

// The guard over HTTP: /api/generate, /api/predict and /api/converse run the
// pair by default - the count model writes, the negative network vetoes - and
// every answer says what it stopped.

// guardEnv is an env with a trained model, ready to write something.
func guardEnv(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, false)
	if status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitJob()
	return e
}

func TestGenerateIsUnguardedUntilAFailureIsBlamed(t *testing.T) {
	e := guardEnv(t)
	status, doc := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	if doc["guard"] != nil {
		t.Fatalf("an empty negative network vetoes nothing: %+v", doc["guard"])
	}
	if len(doc["samples"].([]any)) != 3 {
		t.Fatalf("three texts: %+v", doc["samples"])
	}
}

func TestGenerateGoesThroughTheGuard(t *testing.T) {
	e := guardEnv(t)
	e.blameFailures(nil)
	status, doc := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	guard, ok := doc["guard"].(map[string]any)
	if !ok || guard["on"] != true {
		t.Fatalf("the guard reports itself: %+v", doc["guard"])
	}
	if int(toFloat(guard["asked"])) != 9 { // three wanted, over-sampled three times
		t.Fatalf("the model over-samples: %+v", guard["asked"])
	}
	if len(guard["verdicts"].([]any)) != int(toFloat(guard["candidates"])) {
		t.Fatalf("one verdict per candidate: %+v", guard)
	}
	if len(guard["rejected"].([]any)) != int(toFloat(guard["vetoed"])) {
		t.Fatalf("the vetoes are the rejected ones: %+v", guard)
	}
	refused := map[string]bool{}
	for _, row := range guard["rejected"].([]any) {
		refused[row.(map[string]any)["text"].(string)] = true
	}
	for _, row := range doc["samples"].([]any) {
		if refused[row.(map[string]any)["text"].(string)] {
			t.Fatalf("a vetoed text was handed out anyway: %+v", row)
		}
	}
	// and the caller can still ask for what the positive model wrote
	status, doc = e.post("/api/generate", map[string]any{"count": 3, "guard": false})
	if status != 200 || doc["guard"] != nil {
		t.Fatalf("guard=false turns it off: %d %+v", status, doc["guard"])
	}
}

func TestGenerateVetoesEverythingItHasBeenTaughtToHate(t *testing.T) {
	e := guardEnv(t)
	e.blameFailures(map[string]any{"texts": corpus, "reason": "gibberish"})
	if status, doc := e.post("/api/negative/settings", map[string]any{"threshold": 0.0, "min_coverage": 0.0}); status != 200 {
		t.Fatalf("settings: %d %+v", status, doc)
	}
	status, doc := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	if len(doc["samples"].([]any)) != 0 {
		t.Fatalf("everything it can write is known-bad, so nothing comes back: %+v", doc["samples"])
	}
	guard := doc["guard"].(map[string]any)
	if toFloat(guard["vetoed"]) == 0 || toFloat(guard["rate"]) != 0 {
		t.Fatalf("and the report says so: %+v", guard)
	}
}

func TestPredictAndConverseGoThroughTheGuard(t *testing.T) {
	e := guardEnv(t)
	e.blameFailures(nil)
	status, doc := e.post("/api/predict", map[string]any{"prefix": "the ", "mode": "beam", "k": 4})
	if status != 200 {
		t.Fatalf("predict: %d %+v", status, doc)
	}
	guard, ok := doc["guard"].(map[string]any)
	if !ok || guard["on"] != true {
		t.Fatalf("the guard reports itself: %+v", doc["guard"])
	}
	if len(doc["top"].([]any)) != int(toFloat(guard["kept"])) {
		t.Fatalf("only the survivors are offered: %+v", doc)
	}
	if len(doc["top"].([]any)) > 0 {
		best := doc["top"].([]any)[0].(map[string]any)
		if doc["continuation"] != best["continuation"] {
			t.Fatalf("the best survivor is the prediction: %+v vs %+v", doc["continuation"], best["continuation"])
		}
	}
	status, doc = e.post("/api/converse", map[string]any{"opening": corpus[0], "turns": 3})
	if status != 200 {
		t.Fatalf("converse: %d %+v", status, doc)
	}
	if guard, ok = doc["guard"].(map[string]any); !ok || guard["on"] != true {
		t.Fatalf("the guard reports itself: %+v", doc["guard"])
	}
	refusals := 0
	for _, row := range doc["turns"].([]any) {
		turn := row.(map[string]any)
		if _, ok := turn["vetoed"]; !ok {
			t.Fatalf("every turn counts its own vetoes: %+v", turn)
		}
		refusals += int(toFloat(turn["vetoed"]))
	}
	if refusals != int(toFloat(guard["refusals"])) {
		t.Fatalf("the report adds them up: %d vs %+v", refusals, guard["refusals"])
	}
}

func TestTheGuardCanKeepItsProvenanceToItself(t *testing.T) {
	e := guardEnv(t)
	e.blameFailures(nil)
	status, doc := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40, "provenance": false})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	guard, ok := doc["guard"].(map[string]any)
	if !ok || guard["on"] != true || guard["provenance"] != false {
		t.Fatalf("the guard still reports itself, without provenance: %+v", doc["guard"])
	}
	hasKeys(t, guard, "judged", "vetoed", "negative", "config", "candidates", "kept", "asked", "rate")
	for _, key := range []string{"verdicts", "rejected"} {
		if _, ok := guard[key]; ok {
			t.Fatalf("%q is not listed without provenance: %+v", key, guard)
		}
	}
	if toFloat(guard["judged"]) != toFloat(guard["candidates"]) || toFloat(guard["vetoed"])+toFloat(guard["kept"]) != toFloat(guard["judged"]) {
		t.Fatalf("the counts add up: %+v", guard)
	}
	if guard["config"].(map[string]any)["provenance"] != false {
		t.Fatalf("the config says so: %+v", guard["config"])
	}
	// the vetoes still apply: the same texts as with provenance
	status, full := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, full)
	}
	if len(full["samples"].([]any)) != len(doc["samples"].([]any)) || full["guard"].(map[string]any)["provenance"] != nil {
		t.Fatalf("the same answer, whole by default: %+v vs %+v", full, doc)
	}
	if _, ok := full["guard"].(map[string]any)["verdicts"]; !ok {
		t.Fatalf("the whole report lists the verdicts: %+v", full["guard"])
	}
	// predict and converse read the flag too
	status, doc = e.post("/api/predict", map[string]any{"prefix": "the ", "mode": "beam", "k": 4, "provenance": false})
	if status != 200 || doc["guard"].(map[string]any)["provenance"] != false {
		t.Fatalf("predict: %d %+v", status, doc["guard"])
	}
	status, doc = e.post("/api/converse", map[string]any{"opening": corpus[0], "turns": 3, "provenance": false})
	if status != 200 || doc["guard"].(map[string]any)["provenance"] != false {
		t.Fatalf("converse: %d %+v", status, doc["guard"])
	}
	hasKeys(t, doc["guard"].(map[string]any), "judged", "refusals")
	// the server-wide setting, and one answer overriding it
	status, settings := e.post("/api/negative/settings", map[string]any{"provenance": false})
	if status != 200 || settings["settings"].(map[string]any)["provenance"] != false {
		t.Fatalf("settings: %d %+v", status, settings)
	}
	_, tab := e.get("/api/negative")
	if tab["settings"].(map[string]any)["provenance"] != false {
		t.Fatalf("GET /api/negative reports it: %+v", tab["settings"])
	}
	_, doc = e.post("/api/generate", map[string]any{"count": 2, "mode": "beam", "max_length": 40})
	if doc["guard"].(map[string]any)["provenance"] != false {
		t.Fatalf("the server's setting applies: %+v", doc["guard"])
	}
	_, doc = e.post("/api/generate", map[string]any{"count": 2, "mode": "beam", "max_length": 40, "provenance": true})
	if _, ok := doc["guard"].(map[string]any)["verdicts"]; !ok {
		t.Fatalf("one answer may ask for the provenance back: %+v", doc["guard"])
	}
	e.post("/api/negative/settings", map[string]any{"provenance": true})
	_, tab = e.get("/api/negative")
	if tab["settings"].(map[string]any)["provenance"] != true {
		t.Fatalf("and back on: %+v", tab["settings"])
	}
	if status, _ := e.post("/api/generate", map[string]any{"count": 1, "provenance": "yes"}); status != 400 {
		t.Fatalf("a non-boolean provenance is a 400: %d", status)
	}
}
