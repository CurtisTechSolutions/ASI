package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// The same fake teacher the radixnet tests use, over HTTP: exercises as JSON,
// marks by a rule (only the corpus sentences are correct English), drills as lines.
type fakeTeacher struct {
	server  *httptest.Server
	prompts map[string][]string
}

var tutorCorpus = []string{"the cat sat on the mat", "the dogs run in the park", "the cat likes the mat"}

var gradeLine = regexp.MustCompile(`^\[(\d+)\] <<(.*?)>>(.*)$`)

func newFakeTeacher(t *testing.T) *fakeTeacher {
	t.Helper()
	fake := &fakeTeacher{prompts: map[string][]string{}}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "fake:latest"}}})
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		prompt, _ := body["prompt"].(string)
		system, _ := body["system"].(string)
		kind := "other"
		switch {
		case strings.Contains(system, "writing exercises"):
			kind = "exercises"
		case strings.Contains(system, "marking sentence completions"):
			kind = "grades"
		case strings.Contains(system, "model sentences"):
			kind = "drills"
		}
		fake.prompts[kind] = append(fake.prompts[kind], prompt)
		json.NewEncoder(w).Encode(map[string]any{"response": fake.answer(kind, prompt)})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeTeacher) answer(kind, prompt string) string {
	switch kind {
	case "exercises":
		raw, _ := json.Marshal(map[string]any{"exercises": []map[string]any{
			{"prefix": "the cat sat on", "focus": "prepositions of place", "answer": "the cat sat on the mat"},
			{"prefix": "the dogs run", "focus": "subject-verb agreement", "answer": "the dogs run in the park"},
		}})
		return string(raw)
	case "grades":
		grades := []map[string]any{}
		for _, line := range strings.Split(prompt, "\n") {
			match := gradeLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			index := 0
			fmt.Sscanf(match[1], "%d", &index)
			sentence := strings.Join(strings.Fields(match[2]+strings.Split(match[3], "   (drilling:")[0]), " ")
			correct := false
			for _, known := range tutorCorpus {
				if sentence == known {
					correct = true
				}
			}
			if correct {
				grades = append(grades, map[string]any{"index": index, "grammar": 9, "spelling": 8, "fluency": 7,
					"error": "none", "correction": sentence, "comment": "Well written."})
			} else {
				grades = append(grades, map[string]any{"index": index, "grammar": 2, "spelling": 3, "fluency": 1,
					"error": "subject-verb agreement", "correction": "the dogs run in the park",
					"comment": "A plural subject takes a plural verb."})
			}
		}
		raw, _ := json.Marshal(map[string]any{"grades": grades})
		return string(raw)
	case "drills":
		return "1. the cat sat on the mat\n2. the dogs run in the park"
	}
	return "unexpected request"
}

func trainOpts(epochs int) radixnet.TrainOptions {
	opts := radixnet.DefaultTrainOptions()
	opts.Epochs = epochs
	return opts
}

// tutorEnv is an env whose Ollama defaults point at the fake teacher, with a trained model.
func tutorEnv(t *testing.T) (*env, *fakeTeacher) {
	t.Helper()
	fake := newFakeTeacher(t)
	dir := t.TempDir()
	svc, err := NewService(Options{
		ModelPath: filepath.Join(dir, "model.count.json"), Seed: 1, Exact: true, Quiet: true,
		OllamaURL: fake.server.URL, OllamaModel: "fake:latest",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Model().Train(tutorCorpus, trainOpts(6)); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(NewHandler(svc, "", true, nil))
	t.Cleanup(ts.Close)
	return &env{t: t, ts: ts, svc: svc, dir: dir}, fake
}

func waitForJob(e *env, timeout time.Duration) map[string]any {
	e.t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		status, job := e.get("/api/job")
		if status != 200 {
			e.t.Fatalf("GET /api/job = %d", status)
		}
		if job != nil && job["state"] != "running" {
			return job
		}
		time.Sleep(10 * time.Millisecond)
	}
	e.t.Fatal("the job did not finish in time")
	return nil
}

func TestTutorDescribe(t *testing.T) {
	e, fake := tutorEnv(t)
	status, body := e.get("/api/tutor")
	if status != 200 {
		t.Fatalf("GET /api/tutor = %d: %v", status, body)
	}
	if body["url"] != fake.server.URL || body["model"] != "fake:latest" {
		t.Fatalf("the server defaults should be reported: %v", body)
	}
	if types, _ := body["error_types"].([]any); len(types) == 0 || types[0] != "none" {
		t.Fatalf("error types missing: %v", body["error_types"])
	}
	defaults, _ := body["defaults"].(map[string]any)
	if defaults["threshold"] != 6.0 || defaults["grammar_weight"] != 0.6 {
		t.Fatalf("defaults wrong: %v", defaults)
	}
}

func TestTutorLessonDryRun(t *testing.T) {
	e, fake := tutorEnv(t)
	status, body := e.post("/api/tutor/lesson", map[string]any{"topic": "animals", "exercises": 2})
	if status != 200 {
		t.Fatalf("POST /api/tutor/lesson = %d: %v", status, body)
	}
	if body["source"] != "ollama" || body["model"] != "fake:latest" {
		t.Fatalf("source / model wrong: %v", body)
	}
	lessons, _ := body["lessons"].([]any)
	if len(lessons) != 2 {
		t.Fatalf("expected 2 lessons, got %d", len(lessons))
	}
	first, _ := lessons[0].(map[string]any)
	exercise, _ := first["exercise"].(map[string]any)
	if exercise["prefix"] != "the cat sat on" {
		t.Fatalf("exercise wrong: %v", exercise)
	}
	if !strings.HasPrefix(first["sentence"].(string), "the cat sat on ") {
		t.Fatalf("sentence wrong: %v", first["sentence"])
	}
	grade, _ := first["grade"].(map[string]any)
	if grade["graded_by"] != "ollama" && grade["graded_by"] != "empty" {
		t.Fatalf("grade wrong: %v", grade)
	}
	report, _ := body["report"].(map[string]any)
	if report["lessons"] != 2.0 {
		t.Fatalf("report card wrong: %v", report)
	}
	if e.svc.JobStatus() != nil {
		t.Fatal("a dry run must not start a job")
	}
	if len(fake.prompts["grades"]) != 1 {
		t.Fatalf("expected one marking call, got %d", len(fake.prompts["grades"]))
	}
}

func TestTutorLessonWithGivenPrefixes(t *testing.T) {
	e, fake := tutorEnv(t)
	status, body := e.post("/api/tutor/lesson", map[string]any{
		"prefixes": []string{"the cat sat on", "the dogs run"}, "exercises": 5,
	})
	if status != 200 {
		t.Fatalf("POST /api/tutor/lesson = %d: %v", status, body)
	}
	if body["source"] != "given" {
		t.Fatalf("source = %v", body["source"])
	}
	if len(fake.prompts["exercises"]) != 0 {
		t.Fatal("given prefixes must skip the exercise writer")
	}
	exercises, _ := body["exercises"].([]any)
	if len(exercises) != 2 {
		t.Fatalf("expected the two given prefixes, got %v", exercises)
	}
}

func TestTutorStartAndHistory(t *testing.T) {
	e, _ := tutorEnv(t)
	before := e.svc.Model().MetaInt("feedback_passes")
	status, body := e.post("/api/tutor/start", map[string]any{
		"topic": "animals", "rounds": 2, "exercises": 2, "threshold": 9.5, "drills": 2,
		"neg_epochs": 1, "pos_epochs": 1,
	})
	if status != 202 {
		t.Fatalf("POST /api/tutor/start = %d: %v", status, body)
	}
	job, _ := body["job"].(map[string]any)
	if job["type"] != "tutor" {
		t.Fatalf("job = %v", job)
	}
	finished := waitForJob(e, 30*time.Second)
	if finished["state"] != "done" || finished["error"] != nil {
		t.Fatalf("job did not finish cleanly: %v", finished)
	}
	status, history := e.get("/api/tutor/history")
	if status != 200 {
		t.Fatalf("GET /api/tutor/history = %d", status)
	}
	records, _ := history["history"].([]any)
	kinds := map[string]int{}
	for _, entry := range records {
		record, _ := entry.(map[string]any)
		kinds[record["kind"].(string)]++
	}
	if kinds["lesson"] != 4 || kinds["round"] != 2 || kinds["report"] != 1 {
		t.Fatalf("records = %v", kinds)
	}
	last, _ := records[len(records)-1].(map[string]any)
	if last["rounds"] != 2.0 || last["lessons"] != 4.0 {
		t.Fatalf("report card wrong: %v", last)
	}
	if errors, _ := last["errors"].(map[string]any); errors["agreement"] == nil {
		t.Fatalf("the mistakes should be counted: %v", last["errors"])
	}
	if e.svc.Model().MetaInt("feedback_passes") <= before {
		t.Fatal("the lessons should have taught the model")
	}
	// every failed lesson carries what the teacher changed, for the panel to show
	changed := 0
	for _, entry := range records {
		record, _ := entry.(map[string]any)
		if record["kind"] != "lesson" {
			continue
		}
		if changes, _ := record["changes"].([]any); len(changes) > 0 {
			changed++
			first, _ := changes[0].(map[string]any)
			if first["op"] == nil {
				t.Fatalf("a change names what it did: %v", first)
			}
		}
	}
	if changed == 0 {
		t.Fatal("the corrected lessons should say what changed")
	}
}

func TestTutorDryRunJobLeavesTheModelAlone(t *testing.T) {
	e, _ := tutorEnv(t)
	before := e.svc.Model().MetaInt("twonrl_runs")
	status, body := e.post("/api/tutor/start", map[string]any{
		"topic": "animals", "rounds": 1, "exercises": 2, "threshold": 9.5, "learn": false,
	})
	if status != 202 {
		t.Fatalf("POST /api/tutor/start = %d: %v", status, body)
	}
	waitForJob(e, 30*time.Second)
	if e.svc.Model().MetaInt("twonrl_runs") != before {
		t.Fatal("learn=false must not train")
	}
}

func TestTutorBadRequests(t *testing.T) {
	e, _ := tutorEnv(t)
	cases := []struct {
		body map[string]any
		want string
	}{
		{map[string]any{"topic": " "}, "topic"},
		{map[string]any{"mode": "nope"}, "mode"},
		{map[string]any{"twonrl_per": "hourly"}, "twonrl_per"},
		{map[string]any{"threshold": 99}, "threshold"},
		{map[string]any{"exercises": 0}, "exercises"},
		{map[string]any{"rounds": "many"}, "rounds"},
	}
	for _, c := range cases {
		status, body := e.post("/api/tutor/start", c.body)
		if status != 400 {
			t.Errorf("%v: status = %d (%v)", c.body, status, body)
			continue
		}
		if message, _ := body["error"].(string); !strings.Contains(message, c.want) {
			t.Errorf("%v: error = %q, want it to mention %q", c.body, message, c.want)
		}
	}
}

func TestTutorUnreachableOllamaIs502(t *testing.T) {
	e, _ := tutorEnv(t)
	status, body := e.post("/api/tutor/lesson", map[string]any{
		"topic": "animals", "url": "http://127.0.0.1:1", "timeout": 2,
	})
	if status != 502 {
		t.Fatalf("status = %d: %v", status, body)
	}
	if message, _ := body["error"].(string); !strings.Contains(message, "cannot reach Ollama") {
		t.Fatalf("error = %q", message)
	}
}

func TestFeedbackRatingsWeighWhatIsLearned(t *testing.T) {
	e := newEnv(t, false)
	if _, err := e.svc.Model().Train(tutorCorpus, trainOpts(4)); err != nil {
		t.Fatal(err)
	}
	status, body := e.post("/api/feedback", map[string]any{
		"good": tutorCorpus[:2], "good_ratings": []float64{10, 5},
		"bad": []string{"zzz qqq"}, "bad_ratings": []float64{8},
		"neg_epochs": 1, "pos_epochs": 1,
	})
	if status != 202 {
		t.Fatalf("POST /api/feedback = %d: %v", status, body)
	}
	weights, _ := body["good_weights"].([]any)
	if len(weights) != 2 || weights[0] != 1.0 || weights[1] != 0.5 {
		t.Fatalf("good_weights = %v", weights)
	}
	job := waitForJob(e, 30*time.Second)
	if job["state"] != "done" {
		t.Fatalf("job = %v", job)
	}
	history, _ := job["history"].([]any)
	seen := []float64{}
	for _, entry := range history {
		record, _ := entry.(map[string]any)
		if record["phase"] == "positive" {
			if weight, ok := record["weight"].(float64); ok {
				seen = append(seen, weight)
			}
		}
	}
	if len(seen) != 2 || seen[0] != 1.0 || seen[1] != 0.5 {
		t.Fatalf("the positive phase should follow the marks: %v", seen)
	}
}

func TestFeedbackRatingErrors(t *testing.T) {
	e := newEnv(t, false)
	cases := []struct {
		body map[string]any
		want string
	}{
		{map[string]any{"good": tutorCorpus[:1], "good_ratings": []float64{1, 2}}, "1 text"},
		{map[string]any{"good": tutorCorpus[:1], "good_ratings": []any{"nine"}}, "list of numbers"},
		{map[string]any{"good": tutorCorpus[:1], "good_ratings": []float64{-1}}, ">= 0"},
		{map[string]any{"good": tutorCorpus[:1], "good_ratings": []float64{10}, "good_weights": []float64{1}}, "not both"},
	}
	for _, c := range cases {
		status, body := e.post("/api/feedback", c.body)
		if status != 400 {
			t.Errorf("%v: status = %d (%v)", c.body, status, body)
			continue
		}
		if message, _ := body["error"].(string); !strings.Contains(message, c.want) {
			t.Errorf("%v: error = %q, want it to mention %q", c.body, message, c.want)
		}
	}
}
