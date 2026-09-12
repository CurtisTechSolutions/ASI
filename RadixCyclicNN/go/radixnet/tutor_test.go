package radixnet

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

// A fake Ollama plays the English teacher: it writes exercises as JSON, marks
// completions by a rule (only the corpus sentences are correct English) and
// answers drill requests with plain lines.  Every prompt is recorded.
type fakeTeacher struct {
	server   *httptest.Server
	prompts  map[string][]string
	failWith int
	// answers replace the generated ones when set
	exerciseAnswer string
	gradeAnswer    string
}

var tutorCorpus = []string{"the cat sat on the mat", "the dogs run in the park", "the cat likes the mat"}

var gradeLine = regexp.MustCompile(`^\[(\d+)\] <<(.*?)>>(.*)$`)

func markSentence(prefix, continuation string) map[string]any {
	sentence := strings.Join(strings.Fields(prefix+continuation), " ")
	for _, correct := range tutorCorpus {
		if sentence == correct {
			return map[string]any{"grammar": 9, "spelling": 8, "fluency": 7, "error": "none",
				"correction": sentence, "comment": "Well written."}
		}
	}
	correction := strings.TrimSpace(prefix) + " the mat"
	for _, candidate := range tutorCorpus {
		if strings.HasPrefix(candidate, strings.TrimSpace(prefix)) {
			correction = candidate
			break
		}
	}
	return map[string]any{"grammar": 2, "spelling": 3, "fluency": 1, "error": "subject-verb agreement",
		"correction": correction, "comment": "A plural subject takes a plural verb."}
}

func newFakeTeacher(t *testing.T) *fakeTeacher {
	t.Helper()
	fake := &fakeTeacher{prompts: map[string][]string{}}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fake.failWith != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(fake.failWith)
			fmt.Fprint(w, `{"error": "boom"}`)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/api/tags" {
			writeJSONBody(w, map[string]any{"models": []map[string]any{{"name": "fake:latest", "size": 1}}})
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
		writeJSONBody(w, map[string]any{"model": body["model"], "response": fake.answer(kind, prompt, system), "done": true})
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (f *fakeTeacher) answer(kind, prompt, system string) string {
	switch kind {
	case "exercises":
		if f.exerciseAnswer != "" {
			return f.exerciseAnswer
		}
		count := numberIn(system, `exactly (\d+) entries`, 2)
		pool := []map[string]any{
			{"prefix": "the cat sat on", "focus": "prepositions of place", "answer": "the cat sat on the mat"},
			{"prefix": "the dogs run", "focus": "subject-verb agreement", "answer": "the dogs run in the park"},
			{"prefix": "the cat likes", "focus": "verb agreement", "answer": "the cat likes the mat"},
		}
		exercises := []map[string]any{}
		for i := 0; i < count; i++ {
			exercises = append(exercises, pool[i%len(pool)])
		}
		raw, _ := json.Marshal(map[string]any{"exercises": exercises})
		return string(raw)
	case "grades":
		if f.gradeAnswer != "" {
			return f.gradeAnswer
		}
		grades := []map[string]any{}
		for _, line := range strings.Split(prompt, "\n") {
			match := gradeLine.FindStringSubmatch(line)
			if match == nil {
				continue
			}
			index := numberIn(match[1], `(\d+)`, 0)
			continuation := strings.Split(match[3], "   (drilling:")[0]
			grade := markSentence(match[2], continuation)
			grade["index"] = index
			grades = append(grades, grade)
		}
		raw, _ := json.Marshal(map[string]any{"grades": grades})
		return string(raw)
	case "drills":
		count := numberIn(system, `exactly (\d+) lines`, 3)
		lines := make([]string, count)
		for i := range lines {
			lines[i] = fmt.Sprintf("%d. the cat sat on the mat number %d", i+1, i)
		}
		return strings.Join(lines, "\n")
	}
	return "unexpected request"
}

func numberIn(text, pattern string, fallback int) int {
	match := regexp.MustCompile(pattern).FindStringSubmatch(text)
	if match == nil {
		return fallback
	}
	value := 0
	fmt.Sscanf(match[1], "%d", &value)
	return value
}

func writeJSONBody(w http.ResponseWriter, payload any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(payload)
}

func (f *fakeTeacher) client(t *testing.T) *OllamaClient {
	t.Helper()
	client, err := NewOllamaClient(f.server.URL, "fake:latest", 0)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func tutorModel(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel(7, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	opts := DefaultTrainOptions()
	opts.Epochs = 6
	if _, err := m.Train(tutorCorpus, opts); err != nil {
		t.Fatal(err)
	}
	return m
}

func tutorConfig() TutorConfig {
	cfg := DefaultTutorConfig()
	cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.TutorModel = "animals", 1, 2, "fake:latest"
	cfg.NegEpochs, cfg.PosEpochs = 1, 1
	return cfg
}

// -- the client -------------------------------------------------------------

func TestOllamaClientBasics(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	models, err := client.Models()
	if err != nil || len(models) != 1 || models[0]["name"] != "fake:latest" {
		t.Fatalf("models = %v, %v", models, err)
	}
	if !client.Available() {
		t.Fatal("the fake server should be available")
	}
	if _, err := NewOllamaClient("", "", 0); err != nil {
		t.Fatalf("the defaults should build a client: %v", err)
	}
	url, err := NormaliseOllamaURL("127.0.0.1:11434/")
	if err != nil || url != "http://127.0.0.1:11434" {
		t.Fatalf("NormaliseOllamaURL = %q, %v", url, err)
	}
	if _, err := NormaliseOllamaURL("  "); err == nil {
		t.Fatal("an empty URL must be refused")
	}
	fake.failWith = 500
	if _, err := client.Models(); !IsOllamaError(err) {
		t.Fatalf("a failing server must give an OllamaError, got %v", err)
	}
	if client.Available() {
		t.Fatal("a failing server is not available")
	}
}

func TestParseLines(t *testing.T) {
	lines := ParseLines("```\n1. first line\n\n- second line\n\"first line\"\n```", 0)
	if want := []string{"first line", "second line"}; !equalStrings(lines, want) {
		t.Fatalf("ParseLines = %v, want %v", lines, want)
	}
	if got := ParseLines("a\nb\nc", 2); len(got) != 2 {
		t.Fatalf("the limit was ignored: %v", got)
	}
}

// -- marks and mistakes -----------------------------------------------------

func TestCueAndOverallScore(t *testing.T) {
	if got := Cue("the cat  "); got != "the cat " {
		t.Fatalf("Cue = %q", got)
	}
	if got := Cue("   "); got != "" {
		t.Fatalf("Cue of blank = %q", got)
	}
	ten, zero, six := 10.0, 0.0, 6.0
	if got := *OverallScore(&ten, &zero, &zero, 0.6); got != 6.0 {
		t.Fatalf("grammar weight ignored: %v", got)
	}
	if got := *OverallScore(&zero, &ten, &ten, 0.6); got != 4.0 {
		t.Fatalf("spelling / fluency share wrong: %v", got)
	}
	if got := *OverallScore(&six, nil, nil, 0.6); got != 6.0 {
		t.Fatalf("a lone grammar mark should be the score: %v", got)
	}
	if got := *OverallScore(nil, &ten, &zero, 0.6); got != 5.0 {
		t.Fatalf("missing grammar should mean the rest: %v", got)
	}
	if OverallScore(nil, nil, nil, 0.6) != nil {
		t.Fatal("no marks at all should give no score")
	}
}

func TestErrorTypeOf(t *testing.T) {
	cases := map[string]string{
		"subject-verb agreement": "agreement", "Verb Tense": "tense", "word_order": "word-order",
		"": "none", "no error": "none", "style": "other", "plural": "plural",
	}
	for input, want := range cases {
		if got := ErrorTypeOf(input); got != want {
			t.Errorf("ErrorTypeOf(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestParseExercises(t *testing.T) {
	raw := `{"exercises": [
		{"prefix": "the cat sat on.", "focus": "prepositions", "answer": "the cat sat on the mat"},
		{"prefix": "the dogs run", "focus": "agreement", "answer": "in the park"},
		{"opening": "The Cat Sat On"},
		{"nothing": "usable"}
	]}`
	exercises := ParseExercises(raw, 5, 2)
	if len(exercises) != 2 {
		t.Fatalf("got %d exercises: %+v", len(exercises), exercises)
	}
	if exercises[0].ID != "r2e1" || exercises[0].Prefix != "the cat sat on" || exercises[0].Cue() != "the cat sat on " {
		t.Fatalf("first exercise wrong: %+v", exercises[0])
	}
	if exercises[1].Answer != "the dogs run in the park" { // a continuation-only answer is joined onto the prefix
		t.Fatalf("answer not completed: %q", exercises[1].Answer)
	}
	plain := ParseExercises("1. the cat sat on\n2. the dogs run", 5, 1)
	if len(plain) != 2 || plain[1].Prefix != "the dogs run" {
		t.Fatalf("plain lines not read: %+v", plain)
	}
	if got := ParseExercises("", 3, 1); len(got) != 0 {
		t.Fatalf("nothing should come of nothing: %+v", got)
	}
}

func TestParseGrades(t *testing.T) {
	raw := `{"grades": [
		{"index": 0, "grammar": 9, "spelling": 8, "fluency": 7, "error": "none", "correction": "a", "comment": "good"},
		{"index": 1, "score": 3, "mistake": "plural", "corrected": "b", "critique": "bad"},
		{"index": 9, "grammar": 5},
		"nonsense"
	]}`
	grades := ParseGrades(raw, 2, 0.6, 6.0, "")
	if len(grades) != 2 {
		t.Fatalf("got %d grades: %+v", len(grades), grades)
	}
	first := grades[0]
	if !first.Passed || !closeTo(*first.Score, 0.6*9+0.4*7.5) || first.Error != "none" || first.Correction != "a" {
		t.Fatalf("first grade wrong: %+v (score %v)", first, *first.Score)
	}
	second := grades[1]
	if second.Passed || *second.Score != 3 || second.Error != "plural" || second.Comment != "bad" {
		t.Fatalf("second grade wrong: %+v", second)
	}
	if got := ParseGrades("not json", 2, 0.6, 6, ""); len(got) != 0 {
		t.Fatalf("junk should grade nothing: %+v", got)
	}
	if got := ParseGrades(`{"grades": [{"index": 0}]}`, 1, 0.6, 6, ""); len(got) != 0 {
		t.Fatalf("an entry without marks should be skipped: %+v", got)
	}
}

func TestReportCard(t *testing.T) {
	eight, two := 8.0, 2.0
	lessons := []*Lesson{
		{Grade: Grade{Score: &eight, Grammar: &eight, Passed: true, Error: "none"}},
		{Grade: Grade{Score: &two, Grammar: &two, Error: "agreement"}},
		{Grade: Grade{Error: "agreement", GradedBy: "unrated"}},
	}
	card := ReportCard(lessons)
	if card["lessons"] != 3 || card["graded"] != 2 || card["passed"] != 1 || card["failed"] != 2 {
		t.Fatalf("counts wrong: %v", card)
	}
	if card["mean_score"].(float64) != 5.0 || card["mean_grammar"].(float64) != 5.0 {
		t.Fatalf("means wrong: %v", card)
	}
	if errors := card["errors"].(map[string]int); errors["agreement"] != 2 {
		t.Fatalf("mistakes wrong: %v", errors)
	}
	if weak := card["weakest"].([]string); len(weak) != 1 || weak[0] != "agreement" {
		t.Fatalf("weakest wrong: %v", weak)
	}
}

// -- talking to the teacher -------------------------------------------------

func TestWriteExercisesAndDrills(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	exercises, err := WriteExercises(client, ExerciseRequest{
		Topic: "animals", Count: 2, Focus: "plurals", Weak: []string{"agreement", "none"},
	})
	if err != nil || len(exercises) != 2 {
		t.Fatalf("WriteExercises = %+v, %v", exercises, err)
	}
	prompt := fake.prompts["exercises"][0]
	for _, want := range []string{"Topic: animals", "must drill: plurals", "agreement"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt %q is missing %q", prompt, want)
		}
	}
	if _, err := WriteExercises(client, ExerciseRequest{Topic: "  ", Count: 2}); err == nil {
		t.Fatal("an empty topic must be refused")
	}
	if _, err := WriteExercises(client, ExerciseRequest{Topic: "animals", Count: 0}); err == nil {
		t.Fatal("a count of 0 must be refused")
	}
	fake.exerciseAnswer = `{"exercises": []}`
	// an unusable answer is not the provider's transport failing
	if _, err := WriteExercises(client, ExerciseRequest{Topic: "animals", Count: 2}); !IsLLMError(err) {
		t.Fatalf("no usable exercises must be an LLMError, got %v", err)
	}
	fake.exerciseAnswer = ""
	drills, err := DrillSentences(client, "animals", 3, []string{"plural"}, "")
	if err != nil || len(drills) != 3 {
		t.Fatalf("DrillSentences = %v, %v", drills, err)
	}
	if !strings.Contains(fake.prompts["drills"][0], "plural") {
		t.Fatalf("the drill prompt should name the weak point: %q", fake.prompts["drills"][0])
	}
}

func lessonFor(prefix, continuation string) *Lesson {
	exercise := Exercise{ID: "e1", Prefix: prefix, Focus: "focus", Answer: prefix + " the mat"}
	return &Lesson{Exercise: exercise, Mode: "beam", Continuation: continuation, Sentence: Cue(prefix) + continuation}
}

func TestGradeCompletions(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	lessons := []*Lesson{lessonFor("the cat sat on", "the mat"), lessonFor("the dogs run", "the mat")}
	if err := GradeCompletions(client, lessons, GradeOptions{Topic: "animals", Threshold: 6, GrammarWeight: 0.6, Batch: 10}); err != nil {
		t.Fatal(err)
	}
	if !lessons[0].Grade.Passed || lessons[0].Grade.Error != "none" {
		t.Fatalf("the correct sentence should pass: %+v", lessons[0].Grade)
	}
	if lessons[1].Grade.Passed || lessons[1].Grade.Error != "agreement" {
		t.Fatalf("the wrong sentence should fail on agreement: %+v", lessons[1].Grade)
	}
	if lessons[1].Grade.Correction != "the dogs run in the park" {
		t.Fatalf("correction = %q", lessons[1].Grade.Correction)
	}
	prompt := fake.prompts["grades"][0]
	for _, want := range []string{"Topic of the lesson: animals", "[0] <<the cat sat on >>the mat", "(drilling: focus)"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt %q is missing %q", prompt, want)
		}
	}
}

func TestGradeCompletionsBatchesAndEdgeCases(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	lessons := []*Lesson{}
	for i := 0; i < 5; i++ {
		lessons = append(lessons, lessonFor("the cat sat on", "the mat"))
	}
	if err := GradeCompletions(client, lessons, GradeOptions{Threshold: 6, GrammarWeight: 0.6, Batch: 2}); err != nil {
		t.Fatal(err)
	}
	if len(fake.prompts["grades"]) != 3 {
		t.Fatalf("expected 3 batched calls, got %d", len(fake.prompts["grades"]))
	}
	if err := GradeCompletions(client, lessons, GradeOptions{Batch: 0}); err == nil {
		t.Fatal("a batch of 0 must be refused")
	}

	fake.prompts["grades"] = nil
	empty := []*Lesson{lessonFor("the cat sat on", "")}
	if err := GradeCompletions(client, empty, GradeOptions{Threshold: 6, GrammarWeight: 0.6, Batch: 10}); err != nil {
		t.Fatal(err)
	}
	grade := empty[0].Grade
	if grade.GradedBy != "empty" || grade.Passed || *grade.Score != 0 || grade.Error != "nonsense" {
		t.Fatalf("an empty completion should fail without asking: %+v", grade)
	}
	if grade.Correction != "the cat sat on the mat" {
		t.Fatalf("the teacher's answer should teach instead: %q", grade.Correction)
	}
	if len(fake.prompts["grades"]) != 0 {
		t.Fatal("nothing to mark should not call Ollama")
	}

	fake.gradeAnswer = "the teacher wandered off"
	unrated := []*Lesson{lessonFor("the cat sat on", "the sofa")}
	if err := GradeCompletions(client, unrated, GradeOptions{Threshold: 6, GrammarWeight: 0.6, Batch: 10}); err != nil {
		t.Fatal(err)
	}
	if unrated[0].Grade.GradedBy != "unrated" || unrated[0].Grade.Score != nil || unrated[0].Grade.Passed {
		t.Fatalf("an unreadable answer should leave the lesson unrated: %+v", unrated[0].Grade)
	}
}

// -- the loop ---------------------------------------------------------------

func TestTutorRunTeachesTheCorrections(t *testing.T) {
	fake := newFakeTeacher(t)
	model := tutorModel(t)
	before := model.MetaInt("twonrl_runs")
	cfg := tutorConfig()
	cfg.Rounds = 2
	trainer, err := NewTutorTrainer(model, fake.client(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	records, err := trainer.Run()
	if err != nil {
		t.Fatal(err)
	}
	kinds := []string{}
	for _, record := range records {
		kinds = append(kinds, record["kind"].(string))
	}
	if want := []string{"round", "round", "report"}; !equalStrings(kinds, want) {
		t.Fatalf("records = %v, want %v", kinds, want)
	}
	if len(trainer.Lessons) != 4 {
		t.Fatalf("expected 4 lessons, got %d", len(trainer.Lessons))
	}
	report := records[len(records)-1]
	if report["rounds"] != 2 || report["lessons"] != 4 {
		t.Fatalf("report card wrong: %v", report)
	}
	if model.MetaInt("twonrl_runs") <= before && records[0]["action"] == nil {
		t.Fatal("the round should have taught the network something")
	}
}

func TestTutorLessonRecordsCarryTheMarking(t *testing.T) {
	fake := newFakeTeacher(t)
	trainer, err := NewTutorTrainer(tutorModel(t), fake.client(t), tutorConfig())
	if err != nil {
		t.Fatal(err)
	}
	seen := []map[string]any{}
	trainer.Progress = func(record map[string]any) { seen = append(seen, record) }
	if _, err := trainer.Run(); err != nil {
		t.Fatal(err)
	}
	lessons := 0
	for _, record := range seen {
		if record["kind"] != "lesson" {
			continue
		}
		lessons++
		for _, key := range []string{"score", "grammar", "spelling", "fluency", "passed", "error", "correction", "comment", "sentence", "prefix", "focus"} {
			if _, ok := record[key]; !ok {
				t.Fatalf("lesson record is missing %q: %v", key, record)
			}
		}
		if !strings.HasPrefix(record["sentence"].(string), record["prefix"].(string)) {
			t.Fatalf("the sentence should start with the prefix: %v", record)
		}
	}
	if lessons != 2 {
		t.Fatalf("expected 2 lesson records, got %d", lessons)
	}
}

// scriptedModel is a stand-in network: the same completion every time.
func scriptedTrainer(t *testing.T, fake *fakeTeacher, cfg TutorConfig) *TutorTrainer {
	t.Helper()
	trainer, err := NewTutorTrainer(tutorModel(t), fake.client(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return trainer
}

func TestTutorWeighsGarbageAndRewardsByTheMark(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := tutorConfig()
	cfg.DiffCorrections = false // the whole-sentence path: the diff has a test of its own
	trainer := scriptedTrainer(t, fake, cfg)
	nine, one := 9.9, 1.0
	exercise := Exercise{ID: "e1", Prefix: "the dogs run", Focus: "agreement", Answer: "the dogs run in the park"}
	passed := &Lesson{Exercise: exercise, Continuation: "the park", Sentence: "the dogs run the park",
		Grade: Grade{Score: &nine, Passed: true, Error: "none", Correction: "the dogs run the park"}}
	failed := &Lesson{Exercise: exercise, Continuation: "run run", Sentence: "the dogs run run run",
		Grade: Grade{Score: &one, Error: "agreement", Correction: "the dogs run in the park"}}
	empty := &Lesson{Exercise: exercise, Continuation: "", Sentence: "the dogs run ",
		Grade: Grade{Score: floatPtr(0), Error: "nonsense", Correction: "the dogs run in the park", GradedBy: "empty"}}
	graded := trainer.TextsOf([]*Lesson{passed, failed, empty})
	if want := []string{"the dogs run run run"}; !equalStrings(graded.Bad, want) {
		t.Fatalf("bad = %v, want %v (an empty completion is never punished)", graded.Bad, want)
	}
	rewards := map[string]float64{}
	for i, text := range graded.Good {
		rewards[text] = graded.GoodWeights[i]
	}
	if got := rewards["the dogs run the park"]; got != 0.99 { // 9.9 out of 10
		t.Fatalf("the reward should follow the mark, got %v", got)
	}
	if got := rewards["the dogs run in the park"]; got != TeacherWeight {
		t.Fatalf("the teacher's own English should weigh %v, got %v", TeacherWeight, got)
	}
	if len(graded.BadWeights) != 1 || graded.BadWeights[0] <= trainer.Config.MinWeight {
		t.Fatalf("a hopeless sentence should weigh more than the minimum: %v", graded.BadWeights)
	}
	if len(graded.Corrections) != 0 {
		t.Fatalf("with the diff off, no correction is carried as a pair: %v", graded.Corrections)
	}
}

// A corrected failure is taught as a correction: the sentence and the teacher's
// version travel together, out of the whole-sentence lists.
func TestTutorCarriesCorrectionsAsPairs(t *testing.T) {
	fake := newFakeTeacher(t)
	trainer := scriptedTrainer(t, fake, tutorConfig())
	one := 1.0
	exercise := Exercise{ID: "e1", Prefix: "the dogs run", Focus: "agreement", Answer: "the dogs run in the park"}
	failed := &Lesson{Exercise: exercise, Continuation: "run run", Sentence: "the dogs run run run",
		Grade: Grade{Score: &one, Error: "agreement", Correction: "the dogs run in the park"}}
	uncorrected := &Lesson{Exercise: exercise, Continuation: "xx", Sentence: "the dogs run xx",
		Grade: Grade{Score: &one, Error: "nonsense"}}
	graded := trainer.TextsOf([]*Lesson{failed, uncorrected})
	if len(graded.Corrections) != 1 || graded.Corrections[0].Wrong != "the dogs run run run" ||
		graded.Corrections[0].Right != "the dogs run in the park" {
		t.Fatalf("the corrected failure should travel as a pair: %v", graded.Corrections)
	}
	if graded.Corrections[0].Weight != trainer.WeightOf(failed.Grade) {
		t.Fatalf("a correction carries the mark's weight: %v", graded.Corrections)
	}
	if want := []string{"the dogs run xx"}; !equalStrings(graded.Bad, want) {
		t.Fatalf("only the uncorrected failure is whole-sentence garbage: %v", graded.Bad)
	}
	for _, text := range graded.Good {
		if text == "the dogs run in the park" && len(graded.Corrections) == 1 {
			continue // the teacher's model answer, which happens to be the same sentence
		}
		if text == "the dogs run run run" {
			t.Fatalf("a diffed sentence must not be rewarded whole: %v", graded.Good)
		}
	}
	if changes := trainer.ChangesOf(failed); len(changes) == 0 {
		t.Fatal("the lesson record should carry what the teacher changed")
	}
	if changes := trainer.ChangesOf(uncorrected); len(changes) != 0 {
		t.Fatalf("nothing was corrected, so nothing changed: %v", changes)
	}
}

func TestTutorReplayKeepsTeachingEarlierCorrections(t *testing.T) {
	fake := newFakeTeacher(t)
	texts := []string{"the cat sat on the mat", "the dogs run in the park", "the cat likes the mat"}
	// 0 = no limit (as for the buffer itself), 1 = only the most recent correction comes back
	for _, tc := range []struct{ limit, replayed int }{{0, 2}, {1, 1}} {
		cfg := tutorConfig()
		cfg.ReplayLimit = tc.limit
		cfg.NegEpochs, cfg.PosEpochs = 0, 0 // the bookkeeping is what is being tested, not the passes
		trainer, err := NewTutorTrainer(tutorModel(t), fake.client(t), cfg)
		if err != nil {
			t.Fatal(err)
		}
		var last map[string]any
		for _, text := range texts {
			last, err = trainer.Learn(Graded{
				Bad: []string{"zzz garbage"}, BadWeights: []float64{1},
				Good: []string{text}, GoodWeights: []float64{1},
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		want := len(texts)
		if tc.limit > 0 {
			want = tc.limit
		}
		if len(trainer.replay) != want {
			t.Errorf("limit %d: the buffer holds %d corrections, want %d", tc.limit, len(trainer.replay), want)
		}
		if got := last["good"].(int); got != tc.replayed+1 {
			t.Errorf("limit %d: the last round taught %d texts, want %d (this round's plus the replayed)",
				tc.limit, got, tc.replayed+1)
		}
	}
}

func TestTutorWeightAndRewardCurves(t *testing.T) {
	fake := newFakeTeacher(t)
	trainer := scriptedTrainer(t, fake, tutorConfig())
	nearMiss := trainer.WeightOf(Grade{Score: floatPtr(5)})
	hopeless := trainer.WeightOf(Grade{Score: floatPtr(0)})
	if !(nearMiss < hopeless) || hopeless != 1.0 {
		t.Fatalf("near miss %v should weigh less than hopeless %v (= 1)", nearMiss, hopeless)
	}
	if got := trainer.WeightOf(Grade{}); got != 1.0 {
		t.Fatalf("an unrated failure counts in full, got %v", got)
	}
	if got := trainer.RewardOf(Grade{Score: floatPtr(10)}); got != 1.0 {
		t.Fatalf("10 out of 10 = full reward, got %v", got)
	}
	if got := trainer.RewardOf(Grade{Score: floatPtr(6)}); got != 0.6 {
		t.Fatalf("6 out of 10 = 0.6, got %v", got)
	}
	if got := trainer.RewardOf(Grade{}); got != 0 {
		t.Fatalf("an unrated pass rewards nothing, got %v", got)
	}
}

func TestTutorAdaptsToTheWeakestPoints(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := tutorConfig()
	cfg.Rounds, cfg.Threshold = 2, 9.5 // the fake never marks that high: every round finds mistakes
	trainer := scriptedTrainer(t, fake, cfg)
	if _, err := trainer.Run(); err != nil {
		t.Fatal(err)
	}
	prompts := fake.prompts["exercises"]
	if len(prompts) != 2 {
		t.Fatalf("expected one exercise call per round, got %d", len(prompts))
	}
	if strings.Contains(prompts[0], "keeps making these mistakes") {
		t.Fatal("the first round has nothing to adapt to")
	}
	if !strings.Contains(prompts[1], "keeps making these mistakes") {
		t.Fatalf("the second round should drill the mistakes: %q", prompts[1])
	}

	fake.prompts["exercises"] = nil
	cfg.Adapt = false
	plain := scriptedTrainer(t, fake, cfg)
	if _, err := plain.Run(); err != nil {
		t.Fatal(err)
	}
	for _, prompt := range fake.prompts["exercises"] {
		if strings.Contains(prompt, "keeps making these mistakes") {
			t.Fatal("adapt off should keep the syllabus")
		}
	}
	if len(plain.Weak) != 0 {
		t.Fatalf("adapt off should record no weak points: %v", plain.Weak)
	}
}

func TestTutorDryRunLeavesTheModelAlone(t *testing.T) {
	fake := newFakeTeacher(t)
	model := tutorModel(t)
	before := model.Stats()
	cfg := tutorConfig()
	cfg.Learn, cfg.Threshold = false, 9.5
	trainer, err := NewTutorTrainer(model, fake.client(t), cfg)
	if err != nil {
		t.Fatal(err)
	}
	records, err := trainer.Run()
	if err != nil {
		t.Fatal(err)
	}
	if records[0]["action"] != nil {
		t.Fatalf("a dry run must not train: %v", records[0])
	}
	if records[0]["corrections"].(int) == 0 && records[0]["bad"].(int) == 0 {
		t.Fatalf("a dry run still reports what it would have taught: %v", records[0])
	}
	after := model.Stats()
	if after["epochs_total"] != before["epochs_total"] || after["twonrl_runs"] != before["twonrl_runs"] {
		t.Fatalf("the model changed: %v -> %v", before, after)
	}
}

func TestTutorDrillsAndPerLessonLearning(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := tutorConfig()
	cfg.Drills, cfg.Threshold, cfg.TwoNRLPer = 3, 9.5, "lesson"
	trainer := scriptedTrainer(t, fake, cfg)
	record, lessons, err := trainer.RunRound(1)
	if err != nil {
		t.Fatal(err)
	}
	if record["drills"] != 3 || len(fake.prompts["drills"]) != 1 {
		t.Fatalf("the drills were not written: %v", record)
	}
	if len(lessons) != 2 || record["corrections"].(int) != 2 {
		t.Fatalf("per-lesson learning should sum the lessons: %v", record)
	}
	if record["action"] == nil {
		t.Fatalf("something should have been learned: %v", record)
	}
}

func TestTutorStopEndsTheRun(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := tutorConfig()
	cfg.Rounds = 3
	trainer := scriptedTrainer(t, fake, cfg)
	trainer.Stop = func() bool { return true }
	records, err := trainer.Run()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0]["kind"] != "report" {
		t.Fatalf("a stopped run should only report: %v", records)
	}
	if len(fake.prompts["exercises"]) != 0 {
		t.Fatal("a stopped run should not write exercises")
	}
}

func TestTutorConfigValidation(t *testing.T) {
	cases := []func(c *TutorConfig){
		func(c *TutorConfig) { c.Topic = " " },
		func(c *TutorConfig) { c.Rounds = 0 },
		func(c *TutorConfig) { c.Exercises = 0 },
		func(c *TutorConfig) { c.Attempts = 0 },
		func(c *TutorConfig) { c.Mode = "nope" },
		func(c *TutorConfig) { c.TwoNRLPer = "hourly" },
		func(c *TutorConfig) { c.Threshold = 11 },
		func(c *TutorConfig) { c.GrammarWeight = 2 },
		func(c *TutorConfig) { c.MinWeight = -1 },
		func(c *TutorConfig) { c.Batch = 0 },
		func(c *TutorConfig) { c.Strength = -1 },
	}
	for i, break_ := range cases {
		cfg := DefaultTutorConfig()
		break_(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Errorf("case %d: the configuration should be refused", i)
		}
	}
	valid := DefaultTutorConfig()
	if err := valid.Validate(); err != nil {
		t.Fatalf("the defaults must be valid: %v", err)
	}
}

func TestTutorPropagatesOllamaFailures(t *testing.T) {
	fake := newFakeTeacher(t)
	fake.failWith = 503
	trainer := scriptedTrainer(t, fake, tutorConfig())
	if _, err := trainer.Run(); !IsOllamaError(err) {
		t.Fatalf("expected an OllamaError, got %v", err)
	}
}

// -- weighted passes on the model ------------------------------------------

func TestWeightGroups(t *testing.T) {
	groups, err := WeightGroups([]string{"a", "b", "c", "d"}, []float64{0.5, 1.0, 0.5, 0.0}, "good_weights")
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0].Weight != 1.0 || groups[1].Weight != 0.5 {
		t.Fatalf("heaviest first, zero dropped: %+v", groups)
	}
	if !equalStrings(groups[1].Texts, []string{"a", "c"}) {
		t.Fatalf("equal weights should share a group: %+v", groups[1])
	}
	if _, err := WeightGroups([]string{"a"}, []float64{1, 2}, "good_weights"); err == nil {
		t.Fatal("a length mismatch must be refused")
	}
	if _, err := WeightGroups([]string{"a"}, []float64{-1}, "bad_weights"); err == nil {
		t.Fatal("a negative weight must be refused")
	}
}

func TestRewardAndPunishByTheMark(t *testing.T) {
	model := tutorModel(t)
	opts := DefaultTrainOptions()
	opts.Epochs = 1
	records, err := model.RewardWeighted(tutorCorpus[:2], []float64{1.0, 0.5}, opts, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[0]["weight"] != 1.0 || records[1]["weight"] != 0.5 {
		t.Fatalf("records should carry their weight: %+v", records)
	}
	for _, record := range records {
		if record["phase"] != "positive" {
			t.Fatalf("phase = %v", record["phase"])
		}
	}
	before := toFloat(model.Stats()["edge_reward_positive"])
	if _, err := model.RewardWeighted(tutorCorpus[:1], []float64{0.25}, opts, 1.0); err != nil {
		t.Fatal(err)
	}
	quarter := toFloat(model.Stats()["edge_reward_positive"]) - before
	before = toFloat(model.Stats()["edge_reward_positive"])
	if _, err := model.RewardWeighted(tutorCorpus[:1], []float64{1.0}, opts, 1.0); err != nil {
		t.Fatal(err)
	}
	full := toFloat(model.Stats()["edge_reward_positive"]) - before
	if !(quarter > 0 && full > quarter*2) {
		t.Fatalf("a quarter mark should add much less reward than a full one: %v vs %v", quarter, full)
	}
	penalties, err := model.PunishWeighted([]string{"zzz qqq garbage"}, []float64{0.5}, opts, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(penalties) != 1 || penalties[0]["weight"] != 0.5 || penalties[0]["phase"] != "negative" {
		t.Fatalf("penalties should carry their weight: %+v", penalties)
	}
	result, err := model.TwoNRLWeighted([]string{"zzz qqq"}, []float64{1.0}, tutorCorpus[:2], []float64{1.0, 0.5},
		TwoNRLOptions{NegEpochs: 1, PosEpochs: 1, Strength: 1.0})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Positive) != 2 || result.Positive[1]["weight"] != 0.5 {
		t.Fatalf("2NRL should weigh the positive phase: %+v", result.Positive)
	}
}

func closeTo(got, want float64) bool { return math.Abs(got-want) < 1e-9 }

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
