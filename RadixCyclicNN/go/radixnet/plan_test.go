package radixnet

import (
	"encoding/json"
	"fmt"
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

func strongCard() map[string]any {
	return map[string]any{"lessons": 10, "passed": 9, "pass_rate": 0.9, "mean_score": 8.6,
		"errors": map[string]int{"article": 1}}
}

func TestNextWordsClimbsOneRung(t *testing.T) {
	for _, tc := range [][3]string{
		{"3 to 6", "1", "5 to 8"}, {"5 to 8", "1", "7 to 12"}, {WordsLadder[len(WordsLadder)-1], "1",
			WordsLadder[len(WordsLadder)-1]}, {"3 to 6", "2", "7 to 12"}, {"two or three", "1", "two or three"},
		{"", "1", WordsLadder[0]},
	} {
		steps := 1
		if tc[1] == "2" {
			steps = 2
		}
		if got := NextWords(tc[0], steps); got != tc[2] {
			t.Fatalf("NextWords(%q, %d) = %q, want %q", tc[0], steps, got, tc[2])
		}
	}
}

func TestUpgradeFromCard(t *testing.T) {
	advance := UpgradeFromCard(strongCard(), "beginner", "3 to 6", 6.0, 0)
	if advance["step"] != "advance" || advance["level"] != "intermediate" || advance["words"] != "5 to 8" ||
		advance["threshold"] != 7.0 {
		t.Fatalf("a strong card should advance: %v", advance)
	}
	if !strings.Contains(advance["note"].(string), "steps up to intermediate") ||
		!strings.Contains(advance["note"].(string), "passed 90% at 8.6 out of 10") {
		t.Fatalf("note wrong: %v", advance["note"])
	}
	if capped := UpgradeFromCard(strongCard(), "beginner", "3 to 6", 9.0, 0); capped["threshold"] != 9.0 {
		t.Fatalf("the pass mark should cap: %v", capped["threshold"])
	}

	middling := map[string]any{"lessons": 10, "passed": 6, "pass_rate": 0.6, "mean_score": 6.5,
		"errors": map[string]int{"tense": 4}}
	stretch := UpgradeFromCard(middling, "beginner", "3 to 6", 6.0, 0)
	if stretch["step"] != "stretch" || stretch["level"] != "beginner" || stretch["words"] != "5 to 8" ||
		stretch["threshold"] != 6.0 {
		t.Fatalf("a middling card should only stretch: %v", stretch)
	}

	hold := UpgradeFromCard(planCard(), "beginner", "3 to 6", 6.0, 0)
	if hold["step"] != "hold" || hold["words"] != "3 to 6" || hold["drills"] != HoldDrills {
		t.Fatalf("a weak card should be held back with examples: %v", hold)
	}
	if kept := UpgradeFromCard(planCard(), "beginner", "3 to 6", 6.0, 8); kept["drills"] != 8 {
		t.Fatalf("more drills than the floor should be kept: %v", kept["drills"])
	}
	if none := UpgradeFromCard(map[string]any{}, "", "", 6.0, 0); none["step"] != "hold" {
		t.Fatalf("no marks, nothing earned: %v", none)
	}
}

func TestPlanBriefNamesTheDrillsTheStepAndTheTopic(t *testing.T) {
	step := UpgradeFromCard(planCard(), "beginner", "3 to 6", 6.0, 0)
	brief := PlanBrief(WeakPoints(planCard(), 3), step, "the sea")
	want := "Drill subject-verb agreement, verb tenses and articles (a, an, the), worst first:"
	if !strings.HasPrefix(brief, want) {
		t.Fatalf("brief wrong: %s", brief)
	}
	if !strings.Contains(brief, step["note"].(string)) || !strings.HasSuffix(brief, "Keep the sentences about the sea.") {
		t.Fatalf("brief wrong: %s", brief)
	}
	one := PlanBrief(WeakPoints(map[string]any{"lessons": 4, "errors": map[string]int{"tense": 2}}, 3), step, "")
	if !strings.HasPrefix(one, "Drill verb tenses: that is what the last batch got wrong.") ||
		strings.Contains(one, "Keep the sentences") {
		t.Fatalf("one weak point reads wrong: %s", one)
	}
	if !strings.HasPrefix(PlanBrief(nil, step, "the sea"), "Nothing was marked down") {
		t.Fatalf("a clean card reads wrong")
	}
}

func TestPlanCarriesTheBriefAndTheStepUp(t *testing.T) {
	req := PlanRequest{Topic: "animals", Level: "beginner", Words: "3 to 6", Threshold: 6.0, Count: 3, Exercises: 5}
	plan := PlanFromCard(planCard(), req)
	want := UpgradeFromCard(planCard(), "beginner", "3 to 6", 6.0, 0)
	if fmt.Sprint(plan.Upgrade) != fmt.Sprint(want) {
		t.Fatalf("upgrade wrong: %v", plan.Upgrade)
	}
	if plan.Prompt != PlanBrief(plan.Weak, plan.Upgrade, "animals") {
		t.Fatalf("brief wrong: %s", plan.Prompt)
	}
	if plan.Level != plan.Upgrade["level"] {
		t.Fatalf("the level should be the step up's: %s", plan.Level)
	}
	harder := PlanFromCard(strongCard(), req)
	if harder.Level != "intermediate" || harder.Upgrade["words"] != "5 to 8" || harder.Upgrade["threshold"] != 7.0 {
		t.Fatalf("a strong card should plan a harder batch: %v", harder.Upgrade)
	}
	if !strings.Contains(harder.Prompt, "steps up to intermediate") {
		t.Fatalf("the brief should carry the step up: %s", harder.Prompt)
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
	// a card this weak is held back, so every lesson comes with sentences to imitate
	if first.Focus != "subject-verb agreement" || first.Topic != "animals" || first.Exercises != 4 ||
		first.Drills != HoldDrills {
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
	for _, key := range []string{"summary", "prompt", "upgrade", "level", "topic", "source", "weak", "targets",
		"lessons"} {
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
	upgrade, _ := doc["upgrade"].(map[string]any)
	for _, key := range []string{"step", "level", "words", "threshold", "drills", "note"} {
		if _, ok := upgrade[key]; !ok {
			t.Fatalf("missing %q in %v", key, upgrade)
		}
	}
}

func TestParsePlanReadsTheShapesAnLLMDriftsInto(t *testing.T) {
	raw, _ := json.Marshal(map[string]any{"summary": "Verbs are the trouble.", "prompt": "Drill the verbs.", "lessons": []any{
		map[string]any{"focus": "subject-verb agreement", "targets": "agreement", "topic": "the market", "why": "Most marks."},
		map[string]any{"focus": "past tense", "targets": "verb tense", "topic": "yesterday", "prefixes": []any{"last week we"}},
	}})
	plan := ParsePlan(string(raw), 3, "animals", 4, 1)
	if plan.Summary != "Verbs are the trouble." || plan.Prompt != "Drill the verbs." || plan.Source != "llm" {
		t.Fatalf("plan header wrong: %+v", plan)
	}
	// how much harder the next batch gets is never read from the answer
	harder := ParsePlan(`{"level": "advanced", "words": "20 to 30", "lessons": [{"focus": "plural nouns"}]}`,
		3, "", 5, 0)
	if harder.Level != "" || len(harder.Upgrade) != 0 {
		t.Fatalf("the answer set the difficulty: %+v", harder)
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
	if plan.Lessons[0].Targets != "agreement" || plan.Lessons[0].Exercises != 4 ||
		plan.Lessons[0].Drills != HoldDrills {
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

func TestPlanLessonsTakesTheTeachersBriefButNotItsDifficulty(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	fake.planAnswer = `{"summary": "Verbs.", "prompt": "Drill agreement, then tenses. Hold at beginner.",
		"level": "advanced", "lessons": [{"focus": "subject-verb agreement", "targets": "agreement"}]}`
	plan, err := PlanLessons(client, planCard(), PlanRequest{
		Topic: "animals", Level: "beginner", Words: "3 to 6", Threshold: 6.0, Count: 1,
	})
	if err != nil {
		t.Fatalf("PlanLessons: %v", err)
	}
	if plan.Prompt != "Drill agreement, then tenses. Hold at beginner." {
		t.Fatalf("the teacher's brief was lost: %q", plan.Prompt)
	}
	want := UpgradeFromCard(planCard(), "beginner", "3 to 6", 6.0, 0)
	if fmt.Sprint(plan.Upgrade) != fmt.Sprint(want) || plan.Level != "beginner" {
		t.Fatalf("the answer set the difficulty: %v", plan.Upgrade)
	}
	if plan.Lessons[0].Drills != HoldDrills {
		t.Fatalf("the step up should set the drills: %+v", plan.Lessons[0])
	}
	prompt := fake.prompts["plan"][0]
	for _, want := range []string{
		"Level so far: beginner, openings of 3 to 6 words, pass mark 6 out of 10",
		"The step up this batch has earned, to repeat in your brief: ", plan.Upgrade["note"].(string),
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt is missing %q:\n%s", want, prompt)
		}
	}

	fake.planAnswer = `{"summary": "Verbs.", "lessons": [{"focus": "subject-verb agreement", "targets": "agreement"}]}`
	plan, err = PlanLessons(client, planCard(), PlanRequest{Topic: "animals", Count: 1})
	if err != nil {
		t.Fatalf("PlanLessons: %v", err)
	}
	if plan.Prompt != PlanBrief(plan.Weak, plan.Upgrade, "animals") || plan.Source != "ollama" {
		t.Fatalf("a brief the teacher did not write should come from the marks: %q", plan.Prompt)
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

func TestWriteExercisesHandsOnTheBatchBrief(t *testing.T) {
	fake := newFakeTeacher(t)
	client := fake.client(t)
	req := ExerciseRequest{Topic: "animals", Count: 2, Brief: "  Drill plurals.\n  Stay at beginner.  "}
	if _, err := WriteExercises(client, req); err != nil {
		t.Fatalf("WriteExercises: %v", err)
	}
	if want := "The plan for this batch of lessons: Drill plurals. Stay at beginner."; !strings.Contains(
		fake.prompts["exercises"][0], want) {
		t.Fatalf("the brief did not reach the teacher:\n%s", fake.prompts["exercises"][0])
	}
	if _, err := WriteExercises(client, ExerciseRequest{Topic: "animals", Count: 2}); err != nil {
		t.Fatalf("WriteExercises: %v", err)
	}
	if strings.Contains(fake.prompts["exercises"][1], "The plan for this batch") {
		t.Fatalf("no brief, no line:\n%s", fake.prompts["exercises"][1])
	}
}

func TestTutorRunIsTaughtToItsBrief(t *testing.T) {
	fake := newFakeTeacher(t)
	cfg := DefaultTutorConfig()
	cfg.Topic, cfg.Rounds, cfg.Exercises, cfg.Learn = "animals", 2, 2, false
	cfg.Brief = "Drill plural nouns first. Stay at beginner. Keep the sentences about animals."
	trainer, err := NewTutorTrainer(tutorModel(t), fake.client(t), cfg)
	if err != nil {
		t.Fatalf("NewTutorTrainer: %v", err)
	}
	if _, err := trainer.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fake.prompts["exercises"]) != 2 {
		t.Fatalf("expected one exercise call per round, got %d", len(fake.prompts["exercises"]))
	}
	for _, prompt := range fake.prompts["exercises"] { // every round of the batch is written to the same plan
		if !strings.Contains(prompt, "The plan for this batch of lessons: "+cfg.Brief) {
			t.Fatalf("a round was written without the brief:\n%s", prompt)
		}
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
	if prompt, _ := plan["prompt"].(string); prompt == "" {
		t.Fatalf("the plan record should carry the brief for the next batch: %v", plan)
	}
	if upgrade, _ := plan["upgrade"].(map[string]any); upgrade["step"] != "hold" {
		t.Fatalf("nothing passed at 9.5, so nothing should get harder: %v", plan["upgrade"])
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
