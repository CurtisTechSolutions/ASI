package radixnet

import (
	"encoding/json"
	"strings"
	"testing"
)

func planCard() map[string]any {
	return map[string]any{
		"lessons": 10, "graded": 10, "passed": 2, "failed": 8, "pass_rate": 0.2, "mean_score": 3.4,
		"mean_grammar": 3.0, "mean_spelling": 6.0, "mean_fluency": 4.0,
		"errors": map[string]int{"agreement": 5, "tense": 3, "article": 1},
	}
}

func targetsOf(plan LessonPlan) []string {
	out := make([]string, len(plan.Lessons))
	for i, lesson := range plan.Lessons {
		out[i] = lesson.Targets
	}
	return out
}

func TestWeakPointsRankTheMistakes(t *testing.T) {
	points := WeakPoints(planCard(), 3)
	if got := len(points); got != 3 {
		t.Fatalf("WeakPoints gave %d points: %+v", got, points)
	}
	if points[0].Error != "agreement" || points[0].Count != 5 || points[0].Focus != "subject-verb agreement" {
		t.Fatalf("worst point wrong: %+v", points[0])
	}
	if points[0].Share == nil || *points[0].Share != 0.5 {
		t.Fatalf("share wrong: %+v", points[0])
	}
	if got := targetsOfPoints(WeakPoints(planCard(), 2)); strings.Join(got, ",") != "agreement,tense" {
		t.Fatalf("limit ignored: %v", got)
	}
	if points := WeakPoints(map[string]any{"lessons": 3, "errors": map[string]int{}}, 3); len(points) != 0 {
		t.Fatalf("a clean card has no weak points: %+v", points)
	}
}

func targetsOfPoints(points []WeakPoint) []string {
	out := make([]string, len(points))
	for i, point := range points {
		out[i] = point.Error
	}
	return out
}

func TestWeakPointsReadACardFromJSON(t *testing.T) {
	// a card that came back over HTTP: float counts, the teacher's own wording, "none" and zeroes dropped
	card := map[string]any{"lessons": 4.0, "errors": map[string]any{
		"subject-verb agreement": 2.0, "none": 9.0, "tense": "1", "plural": 0.0,
	}}
	points := WeakPoints(card, 3)
	if got := targetsOfPoints(points); strings.Join(got, ",") != "agreement,tense" {
		t.Fatalf("json card read wrong: %v", got)
	}
	if points[0].Share == nil || *points[0].Share != 0.5 {
		t.Fatalf("share wrong: %+v", points[0])
	}
	if point := WeakPoints(map[string]any{"errors": map[string]int{"tense": 1}}, 3)[0]; point.Share != nil {
		t.Fatalf("no lessons: no share, got %v", *point.Share)
	}
}

func TestWeakPointTiesBreakAlphabetically(t *testing.T) {
	card := map[string]any{"lessons": 4, "errors": map[string]int{"tense": 2, "article": 2, "plural": 2}}
	if got := targetsOfPoints(WeakPoints(card, 3)); strings.Join(got, ",") != "article,plural,tense" {
		t.Fatalf("ties wrong: %v", got)
	}
}

func TestFocusForAndNextLevel(t *testing.T) {
	if FocusFor("subject-verb agreement") != "subject-verb agreement" || FocusFor("plural") != "plural nouns" {
		t.Fatalf("FocusFor names the wrong grammar point")
	}
	if FocusFor("none") != "" || FocusFor("") != "" {
		t.Fatalf("a clean sentence drills nothing")
	}
	for _, name := range ErrorTypes[1:] {
		if FocusFor(name) == "" {
			t.Fatalf("no focus for %q", name)
		}
	}
	strong := map[string]any{"pass_rate": 0.9, "mean_score": 8.5}
	for level, want := range map[string]string{"beginner": "intermediate", "intermediate": "advanced", "advanced": "advanced"} {
		if got := NextLevel(level, strong); got != want {
			t.Fatalf("NextLevel(%q) = %q, want %q", level, got, want)
		}
	}
	if got := NextLevel("beginner", planCard()); got != "beginner" {
		t.Fatalf("a weak card does not promote: %q", got)
	}
	if got := NextLevel("beginner", map[string]any{"pass_rate": 0.9}); got != "beginner" {
		t.Fatalf("no mean mark, no promotion: %q", got)
	}
}

func TestCardLinesReadLikeAReportCard(t *testing.T) {
	lines := CardLines(planCard())
	want := []string{
		"Lessons marked: 10",
		"Passed: 2 (20%)",
		"Mean marks out of 10: overall 3.4, grammar 3.0, spelling 6.0, fluency 4.0",
		"Mistakes, worst first: agreement x5 (50% of the lessons), tense x3 (30% of the lessons), " +
			"article x1 (10% of the lessons)",
	}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Fatalf("card lines wrong:\n%s", strings.Join(lines, "\n"))
	}
	clean := CardLines(map[string]any{"lessons": 1})
	if clean[len(clean)-1] != "Mistakes: none were named." || !strings.Contains(clean[2], "none recorded") {
		t.Fatalf("a card with nothing on it reads wrong: %v", clean)
	}
}

func TestPlanFromCardGivesEveryWeakPointALesson(t *testing.T) {
	plan := PlanFromCard(planCard(), PlanRequest{Topic: "animals", Level: "beginner", Count: 3, Exercises: 4, Drills: 2})
	if plan.Source != "report card" || plan.Level != "beginner" {
		t.Fatalf("plan header wrong: %+v", plan)
	}
	if got := strings.Join(targetsOf(plan), ","); got != "agreement,tense,article" {
		t.Fatalf("lessons wrong: %v", got)
	}
	first, last := plan.Lessons[0], plan.Lessons[2]
	if first.Focus != "subject-verb agreement" || first.Topic != "animals" || first.Exercises != 4 || first.Drills != 2 {
		t.Fatalf("first lesson wrong: %+v", first)
	}
	if first.Why != "5 of 10 lessons (50%) were marked down for agreement." {
		t.Fatalf("why wrong: %q", first.Why)
	}
	if last.Why != "1 of 10 lessons (10%) was marked down for article." {
		t.Fatalf("a single mistake reads wrong: %q", last.Why)
	}
	if !strings.HasPrefix(plan.Summary, "2 of 10 lessons passed at a mean of 3.4 out of 10") {
		t.Fatalf("summary wrong: %q", plan.Summary)
	}
	if len(PlanFromCard(planCard(), PlanRequest{Count: 2}).Lessons) != 2 {
		t.Fatalf("count does not cap the lessons")
	}
}

func TestPlanFromACleanCardMovesUpALevel(t *testing.T) {
	card := map[string]any{"lessons": 4, "passed": 4, "pass_rate": 1.0, "mean_score": 9.0,
		"errors": map[string]int{}}
	plan := PlanFromCard(card, PlanRequest{Topic: "the sea", Level: "beginner", Count: 3})
	if plan.Level != "intermediate" || len(plan.Lessons) != 1 || plan.Lessons[0].Targets != "none" {
		t.Fatalf("clean card planned wrong: %+v", plan)
	}
	if !strings.Contains(plan.Summary, "nothing was marked down") {
		t.Fatalf("summary wrong: %q", plan.Summary)
	}
}

func TestPlanMapIsTheSameJSONAsPython(t *testing.T) {
	raw, err := json.Marshal(PlanFromCard(planCard(), PlanRequest{Topic: "animals", Count: 2}).Map())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{"summary", "level", "topic", "source", "weak", "targets", "lessons"} {
		if _, ok := doc[key]; !ok {
			t.Fatalf("missing %q in %s", key, raw)
		}
	}
	lesson := doc["lessons"].([]any)[0].(map[string]any)
	for _, key := range []string{"focus", "topic", "why", "targets", "exercises", "drills", "prefixes"} {
		if _, ok := lesson[key]; !ok {
			t.Fatalf("missing %q in %v", key, lesson)
		}
	}
	if doc["weak"].([]any)[0].(map[string]any)["error"] != "agreement" {
		t.Fatalf("weak points wrong: %v", doc["weak"])
	}
}

func TestParsePlanReadsTheShapesAnLLMDriftsInto(t *testing.T) {
	raw, _ := json.Marshal(map[string]any{"summary": "Verbs are the trouble.", "level": "Intermediate", "lessons": []any{
		map[string]any{"focus": "subject-verb agreement", "targets": "agreement", "topic": "the market", "why": "Most marks."},
		map[string]any{"focus": "past tense", "targets": "verb tense", "topic": "yesterday", "prefixes": []any{"last week we"}},
	}})
	plan := ParsePlan(string(raw), 3, "animals", 4, 1)
	if plan.Summary != "Verbs are the trouble." || plan.Level != "intermediate" || plan.Source != "llm" {
		t.Fatalf("plan header wrong: %+v", plan)
	}
	if got := strings.Join(targetsOf(plan), ","); got != "agreement,tense" {
		t.Fatalf("targets wrong: %v", got)
	}
	if plan.Lessons[1].Topic != "yesterday" || len(plan.Lessons[1].Prefixes) != 1 {
		t.Fatalf("second lesson wrong: %+v", plan.Lessons[1])
	}
	if plan.Lessons[0].Exercises != 4 || plan.Lessons[0].Drills != 1 {
		t.Fatalf("settings do not ride along: %+v", plan.Lessons[0])
	}
	for _, drifted := range []string{
		`{"plan": [{"point": "plural nouns", "reason": "shaky"}]}`,
		`[{"skill": "plural nouns", "because": "shaky"}]`,
		`{"focus": "plural nouns", "note": "shaky"}`,
		"```json\n{\"lessons\": [{\"title\": \"plural nouns\", \"rationale\": \"shaky\"}]}\n```",
	} {
		plan := ParsePlan(drifted, 3, "animals", 5, 0)
		if len(plan.Lessons) != 1 {
			t.Fatalf("%s -> %d lessons", drifted, len(plan.Lessons))
		}
		lesson := plan.Lessons[0]
		if lesson.Focus != "plural nouns" || lesson.Targets != "plural" || lesson.Why != "shaky" || lesson.Topic != "animals" {
			t.Fatalf("%s -> %+v", drifted, lesson)
		}
	}
	lines := ParsePlan("1. plural nouns\n2. prepositions of place", 5, "animals", 5, 0)
	if got := strings.Join(targetsOf(lines), ","); got != "plural,preposition" {
		t.Fatalf("plain lines are still a syllabus: %v", got)
	}
	dupes := ParsePlan(`{"lessons": [{"focus": "plural nouns"}, {"focus": "Plural Nouns"}, {"why": "no focus"},
		{"targets": "tense"}, {"focus": "articles"}]}`, 3, "animals", 5, 0)
	if got := strings.Join(targetsOf(dupes), ","); got != "plural,tense,article" {
		t.Fatalf("duplicates, empties and the overflow wrong: %v", got)
	}
	if len(ParsePlan("", 3, "", 5, 0).Lessons) != 0 || len(ParsePlan(`{"lessons": []}`, 3, "", 5, 0).Lessons) != 0 {
		t.Fatalf("an unreadable answer plans nothing")
	}
}

func TestPlanLessonsAsksTheTeacher(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	plan, err := PlanLessons(client, planCard(), PlanRequest{
		Topic: "animals", Level: "beginner", Count: 2, Exercises: 4, Drills: 1,
	})
	if err != nil {
		t.Fatalf("PlanLessons: %v", err)
	}
	if plan.Source != "ollama" || plan.Summary != "The student writes verbs badly." || plan.Level != "beginner" {
		t.Fatalf("plan header wrong: %+v", plan)
	}
	if plan.Lessons[0].Targets != "agreement" || plan.Lessons[0].Exercises != 4 || plan.Lessons[0].Drills != 1 {
		t.Fatalf("first lesson wrong: %+v", plan.Lessons[0])
	}
	if got := targetsOfPoints(plan.Weak); strings.Join(got, ",") != "agreement,tense" {
		t.Fatalf("the card's weak points are missing: %v", got)
	}
	prompt := fake.prompts["plan"][0]
	for _, want := range []string{
		"The lessons so far were about: animals", "Mistakes, worst first: agreement x5 (50% of the lessons)",
		"Level so far: beginner", "Plan the next 2 lesson(s) now.",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt is missing %q:\n%s", want, prompt)
		}
	}
}

func TestPlanLessonsKeepsEveryWeaknessOfTheCard(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	// the teacher plans plurals (not on the card) and agreement; tense must still get a lesson
	fake.planAnswer = `{"summary": "Verbs.", "lessons": [{"focus": "plural nouns", "targets": "plural"},
		{"focus": "subject-verb agreement", "targets": "agreement"}]}`
	plan, err := PlanLessons(client, planCard(), PlanRequest{Topic: "animals", Count: 2})
	if err != nil {
		t.Fatalf("PlanLessons: %v", err)
	}
	if got := strings.Join(targetsOf(plan), ","); got != "tense,agreement" {
		t.Fatalf("the skipped weakness did not take the off-card slot: %v", got)
	}

	fake.planAnswer = `{"summary": "Verbs.", "lessons": [{"focus": "subject-verb agreement", "targets": "agreement"}]}`
	plan, err = PlanLessons(client, planCard(), PlanRequest{Topic: "animals", Count: 3})
	if err != nil {
		t.Fatalf("PlanLessons: %v", err)
	}
	if got := strings.Join(targetsOf(plan), ","); got != "agreement,tense,article" {
		t.Fatalf("the skipped weaknesses were not appended: %v", got)
	}
	if plan.Lessons[1].Why == "" || plan.Lessons[1].Topic != "animals" {
		t.Fatalf("an appended lesson is missing its settings: %+v", plan.Lessons[1])
	}
}

func TestPlanLessonsFallsBackToTheCard(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	for _, answer := range []string{"the teacher wandered off", `{"lessons": []}`, " "} {
		fake.planAnswer = answer
		plan, err := PlanLessons(client, planCard(), PlanRequest{Topic: "animals", Count: 2})
		if err != nil {
			t.Fatalf("PlanLessons(%q): %v", answer, err)
		}
		if plan.Source != "report card" {
			t.Fatalf("%q was used as a plan: %+v", answer, plan)
		}
		if got := strings.Join(targetsOf(plan), ","); got != "agreement,tense" {
			t.Fatalf("%q -> %v", answer, got)
		}
	}
}

func TestPlanLessonsRejectsBadInput(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	for _, req := range []PlanRequest{{Count: 0}, {Count: 2}} {
		if _, err := PlanLessons(client, map[string]any{}, req); err == nil {
			t.Fatalf("%+v was accepted", req)
		}
	}
	if _, err := PlanLessons(client, map[string]any{"lessons": 0}, PlanRequest{Count: 2}); err == nil {
		t.Fatalf("a card of no lessons was accepted")
	}
	fake.failWith = 500
	if _, err := PlanLessons(client, planCard(), PlanRequest{Count: 2}); err == nil || !IsOllamaError(err) {
		t.Fatalf("a failing teacher must be an Ollama error, got %v", err)
	}
}

func TestTutorRunCanEndWithAPlan(t *testing.T) {
	fake := newFakeTeacher(t)
	model := tutorModel(t)
	cfg := DefaultTutorConfig()
	cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.Threshold, cfg.Plan = "animals", 1, 2, 9.5, 2
	cfg.NegEpochs, cfg.PosEpochs = 1, 1
	trainer, err := NewTutorTrainer(model, fake.client(t), cfg)
	if err != nil {
		t.Fatalf("NewTutorTrainer: %v", err)
	}
	seen := []string{}
	trainer.Progress = func(record map[string]any) { seen = append(seen, record["kind"].(string)) }
	records, err := trainer.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	kinds := []string{}
	for _, record := range records {
		kinds = append(kinds, record["kind"].(string))
	}
	if strings.Join(kinds, ",") != "round,report,plan" {
		t.Fatalf("records wrong: %v", kinds)
	}
	plan := records[len(records)-1]
	if plan["rounds"] != 1 || plan["source"] != "ollama" {
		t.Fatalf("plan record wrong: %v", plan)
	}
	if lessons := plan["lessons"].([]map[string]any); len(lessons) != 2 || lessons[0]["exercises"] != 2 {
		t.Fatalf("planned lessons wrong: %v", plan["lessons"])
	}
	if seen[len(seen)-1] != "plan" {
		t.Fatalf("the plan never reached the progress callback: %v", seen)
	}
	if !strings.Contains(fake.prompts["plan"][0], "Lessons marked: 2") {
		t.Fatalf("the teacher did not see the report card: %s", fake.prompts["plan"][0])
	}
}

func TestTutorPlanFailureIsOnlyANote(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := DefaultTutorConfig()
	cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.Plan, cfg.Learn = "animals", 1, 2, 2, false
	trainer, err := NewTutorTrainer(tutorModel(t), fake.client(t), cfg)
	if err != nil {
		t.Fatalf("NewTutorTrainer: %v", err)
	}
	// the rounds are done before the teacher walks out, so only the plan is lost
	trainer.External = func(fn func() error) error {
		if len(trainer.Lessons) > 0 {
			fake.failWith = 500
		}
		return fn()
	}
	records, err := trainer.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	last := records[len(records)-1]
	if last["kind"] != "report" {
		t.Fatalf("the run did not finish cleanly: %v", last["kind"])
	}
	note := trainer.History[len(trainer.History)-1]
	if note["kind"] != "note" || !strings.Contains(note["message"].(string), "no lesson plan") {
		t.Fatalf("no note about the missing plan: %v", note)
	}
}
