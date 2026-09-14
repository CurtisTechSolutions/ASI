package radixnet

// The lesson plan: the report card at the end of a run handed back to the
// teacher - a local Ollama model or ChatGPT, whichever taught - which answers
// with the syllabus of the lessons that follow: one point of grammar each,
// aimed at the mistakes the marking found.  The Go twin of the Python planner
// (radixnet/tutor.py), asking the same questions and falling back the same
// way, so both sides plan the same lessons.

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// ErrorFocus is the point of grammar a lesson drills to fix each mistake of ErrorTypes.
var ErrorFocus = map[string]string{
	"agreement":   "subject-verb agreement",
	"tense":       "verb tenses",
	"article":     "articles (a, an, the)",
	"preposition": "prepositions",
	"plural":      "plural nouns",
	"pronoun":     "pronouns",
	"word-order":  "word order",
	"spelling":    "spelling",
	"punctuation": "punctuation",
	"vocabulary":  "word choice",
	"fragment":    "finishing the sentence",
	"nonsense":    "writing a sentence that means something",
	"other":       "sentence structure",
}

// Levels are how hard the exercises are; a plan moves the student one step up a
// card it has nothing left to fix in.
var Levels = []string{"beginner", "intermediate", "advanced"}

// DefaultPlanLessons is how many lessons a plan holds unless more are asked for.
const DefaultPlanLessons = 3

// A report card at or above both is a student ready for the next level;
// StretchPassRate is not a level up, but the openings get longer.
const (
	StrongPassRate  = 0.8
	StrongScore     = 8.0
	StretchPassRate = 0.5
)

// WordsLadder is how long an opening the teacher writes; an upgrade moves one rung up.
var WordsLadder = []string{"3 to 6", "5 to 8", "7 to 12", "10 to 16"}

// UpgradeSteps are what the next batch does with the difficulty: hold it, stretch it, or move up a level.
var UpgradeSteps = []string{"hold", "stretch", "advance"}

// HoldDrills is how many correct example sentences a held-back batch gets to imitate.
const HoldDrills = 3

const maxBriefChars = 600

// FocusFor is the grammar point that drills one mistake ("" for "none" and anything unrecognised).
func FocusFor(error string) string { return ErrorFocus[ErrorTypeOf(error)] }

// countOf reads a count out of a report card, however it survived the JSON round trip (0 when unusable).
func countOf(value any) int {
	switch number := value.(type) {
	case int:
		return max(0, number)
	case float64:
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return 0
		}
		return max(0, int(number))
	case string:
		if parsed, err := strconv.ParseFloat(strings.TrimSpace(number), 64); err == nil {
			return max(0, int(parsed))
		}
	}
	return 0
}

// markOf reads a mean mark of a report card, or nil when it is missing.
func markOf(value any) *float64 {
	switch number := value.(type) {
	case float64:
		if math.IsNaN(number) {
			return nil
		}
		return floatPtr(number)
	case int:
		return floatPtr(float64(number))
	case *float64:
		return number
	}
	return nil
}

// errorCounts reads a card's error histogram, whether it came from ReportCard or from JSON.
func errorCounts(value any) map[string]int {
	counts := map[string]int{}
	switch errors := value.(type) {
	case map[string]int:
		for name, count := range errors {
			counts[ErrorTypeOf(name)] += max(0, count)
		}
	case map[string]any:
		for name, count := range errors {
			counts[ErrorTypeOf(name)] += countOf(count)
		}
	}
	delete(counts, "none")
	for name, count := range counts {
		if count <= 0 {
			delete(counts, name)
		}
	}
	return counts
}

// WeakPoint is one mistake of a report card: how often it was the worst thing
// in a sentence, in how many of the lessons, and the grammar point that fixes it.
type WeakPoint struct {
	Error string   `json:"error"`
	Count int      `json:"count"`
	Share *float64 `json:"share"` // of the lessons, nil when the card has none
	Focus string   `json:"focus"`
}

// WeakPoints ranks the mistakes of a report card, worst first (ties alphabetically,
// like the card's own "weakest", so the same marks always plan the same lessons).
func WeakPoints(card map[string]any, limit int) []WeakPoint {
	counts := errorCounts(card["errors"])
	names := make([]string, 0, len(counts))
	for name := range counts {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if counts[names[i]] != counts[names[j]] {
			return counts[names[i]] > counts[names[j]]
		}
		return names[i] < names[j]
	})
	if limit >= 0 && len(names) > limit {
		names = names[:limit]
	}
	lessons := countOf(card["lessons"])
	points := []WeakPoint{}
	for _, name := range names {
		point := WeakPoint{Error: name, Count: counts[name], Focus: FocusFor(name)}
		if lessons > 0 {
			point.Share = floatPtr(float64(counts[name]) / float64(lessons))
		}
		points = append(points, point)
	}
	return points
}

// NextLevel is one step up Levels when the card is strong (StrongPassRate passed
// at StrongScore), else the level it was given.
func NextLevel(level string, card map[string]any) string {
	level = strings.ToLower(strings.TrimSpace(level))
	if level == "" {
		level = Levels[0]
	}
	passRate, score := markOf(card["pass_rate"]), markOf(card["mean_score"])
	strong := passRate != nil && *passRate >= StrongPassRate && score != nil && *score >= StrongScore
	if !strong {
		return level
	}
	for i, name := range Levels {
		if name == level {
			return Levels[min(i+1, len(Levels)-1)]
		}
	}
	return level
}

// NextWords is one rung up WordsLadder; a setting that is not on the ladder is left alone.
func NextWords(words string, steps int) string {
	text := strings.Join(strings.Fields(words), " ")
	if steps < 0 {
		steps = 0
	}
	for i, rung := range WordsLadder {
		if rung == text {
			return WordsLadder[min(i+steps, len(WordsLadder)-1)]
		}
	}
	if text == "" {
		return WordsLadder[0]
	}
	return text
}

func percentOf(value *float64) string {
	if value == nil {
		return "too few"
	}
	return fmt.Sprintf("%.0f%%", *value*100)
}

// UpgradeFromCard is the incremental step the next batch takes, read off the marks.
//
// "advance" when the card is strong (StrongPassRate of the lessons passed at
// StrongScore): a level up, longer openings and a higher pass mark.  "stretch"
// when StretchPassRate passed: the same level, but longer openings.  "hold"
// otherwise: nothing gets harder, the weak points are drilled and the
// teacher's own example sentences (HoldDrills) come with them, because a
// student who is failing does not need a harder exercise.
func UpgradeFromCard(card map[string]any, level, words string, threshold float64, drills int) map[string]any {
	rate, score := markOf(card["pass_rate"]), markOf(card["mean_score"])
	level = strings.ToLower(strings.TrimSpace(level))
	if level == "" {
		level = Levels[0]
	}
	words = strings.Join(strings.Fields(words), " ")
	if words == "" {
		words = WordsLadder[0]
	}
	passed, mark := percentOf(rate), "-"
	if score != nil {
		mark = fmt.Sprintf("%.1f", *score)
	}
	var step, note string
	switch {
	case rate != nil && *rate >= StrongPassRate && score != nil && *score >= StrongScore:
		step, level, words = "advance", NextLevel(level, card), NextWords(words, 1)
		threshold = math.Min(9.0, threshold+1.0)
		note = fmt.Sprintf(
			"The last batch passed %s at %s out of 10, so this one steps up to %s: openings of %s words and a "+
				"pass mark of %.0f out of 10.", passed, mark, level, words, threshold)
	case rate != nil && *rate >= StretchPassRate:
		step, words = "stretch", NextWords(words, 1)
		note = fmt.Sprintf(
			"The last batch passed %s, so this one stays at %s but stretches the openings to %s words.",
			passed, level, words)
	default:
		step = "hold"
		drills = max(drills, HoldDrills)
		note = fmt.Sprintf(
			"The last batch passed %s, so this one holds at %s with openings of %s words, drills the mistakes "+
				"and comes with %d correct sentences to imitate.", passed, level, words, drills)
	}
	return map[string]any{
		"step": step, "level": level, "words": words, "threshold": threshold, "drills": drills, "note": note,
	}
}

// andList is "a, b and c" - the teacher's own English, so its brief reads like one.
func andList(names []string) string {
	switch len(names) {
	case 0:
		return ""
	case 1:
		return names[0]
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}

// PlanBrief is what the marks alone write for the next batch: what to drill,
// how it steps up, what it is about.
func PlanBrief(weak []WeakPoint, upgrade map[string]any, topic string) string {
	points := []string{}
	for _, point := range weak {
		if name := point.Focus; name != "" {
			points = append(points, name)
		} else if point.Error != "" {
			points = append(points, point.Error)
		}
	}
	var opening string
	switch {
	case len(points) > 1:
		opening = "Drill " + andList(points) + ", worst first: that is what the last batch got wrong."
	case len(points) == 1:
		opening = "Drill " + points[0] + ": that is what the last batch got wrong."
	default:
		opening = "Nothing was marked down, so widen the vocabulary instead of drilling one point of grammar."
	}
	lines := []string{opening}
	if note, _ := upgrade["note"].(string); strings.TrimSpace(note) != "" {
		lines = append(lines, strings.TrimSpace(note))
	}
	if strings.TrimSpace(topic) != "" {
		lines = append(lines, "Keep the sentences about "+strings.TrimSpace(topic)+".")
	}
	return strings.Join(lines, " ")
}

// PlannedLesson is one lesson of a plan: the point of grammar it drills, what
// its sentences are about and the mistake it is aimed at.
type PlannedLesson struct {
	Focus     string   `json:"focus"`
	Topic     string   `json:"topic"`
	Why       string   `json:"why"`     // one sentence for the student's file: why it is being taught
	Targets   string   `json:"targets"` // the mistake of ErrorTypes it fixes ("none": no particular one)
	Exercises int      `json:"exercises"`
	Drills    int      `json:"drills"`
	Prefixes  []string `json:"prefixes"` // openings the teacher already wrote, if any
}

// LessonPlan is the lessons to run next, written from a report card.
type LessonPlan struct {
	Lessons []PlannedLesson `json:"lessons"`
	Summary string          `json:"summary"` // where the student stands, in a sentence or two
	// Prompt is the brief for the next batch: what to drill, how it steps up, what it is about.
	Prompt string `json:"prompt"`
	// Upgrade is UpgradeFromCard: the step the next batch takes, and the settings it takes it with.
	Upgrade map[string]any `json:"upgrade"`
	Level   string         `json:"level"`  // the level the next lessons should be at
	Topic   string         `json:"topic"`  // the topic they were planned around
	Weak    []WeakPoint    `json:"weak"`   // the weak points of the card behind the plan
	Source  string         `json:"source"` // who wrote it: the provider, or "report card" (the marks alone)
}

// Targets are the mistakes the plan drills, in its own order.
func (p LessonPlan) Targets() []string {
	out, seen := []string{}, map[string]bool{}
	for _, lesson := range p.Lessons {
		if lesson.Targets == "" || lesson.Targets == "none" || seen[lesson.Targets] {
			continue
		}
		seen[lesson.Targets] = true
		out = append(out, lesson.Targets)
	}
	return out
}

// Map is the plan as the API answers it (the Python server's shape).
func (p LessonPlan) Map() map[string]any {
	lessons := make([]map[string]any, 0, len(p.Lessons))
	for _, lesson := range p.Lessons {
		prefixes := lesson.Prefixes
		if prefixes == nil {
			prefixes = []string{}
		}
		lessons = append(lessons, map[string]any{
			"focus": lesson.Focus, "topic": lesson.Topic, "why": lesson.Why, "targets": lesson.Targets,
			"exercises": lesson.Exercises, "drills": lesson.Drills, "prefixes": prefixes,
		})
	}
	weak := p.Weak
	if weak == nil {
		weak = []WeakPoint{}
	}
	upgrade := p.Upgrade
	if upgrade == nil {
		upgrade = map[string]any{}
	}
	return map[string]any{
		"summary": p.Summary, "prompt": p.Prompt, "upgrade": upgrade, "level": p.Level, "topic": p.Topic,
		"source": p.Source, "weak": weak, "targets": p.Targets(), "lessons": lessons,
	}
}

const planSchema = `{"summary": "<one or two sentences>", "prompt": "<the brief for the next batch>", "lessons": ` +
	`[{"focus": "<the point of grammar>", "targets": "<the mistake it fixes>", "topic": "<what its sentences are ` +
	`about>", "why": "<one sentence>"}, ...]}`

const planSystem = "You are an English teacher planning the next lessons for a beginner student from its report " +
	"card. The student is given the opening words of a sentence and writes the rest; the report card is the " +
	"marking of the lessons it has just done - how many it passed, its mean marks out of 10 for grammar, " +
	"spelling and fluency, and how often each kind of mistake was the worst thing in a sentence. Plan the %[1]d " +
	"lessons that repair the most: each one drills a single point of grammar ('focus', two or three words, e.g. " +
	"'subject-verb agreement', 'past tense', 'plural nouns'), names the mistake from the report card it is aimed " +
	"at ('targets', one of: %[2]s), gives the everyday subject its sentences should be about ('topic', two or " +
	"three words) and says in one short sentence why it is being taught ('why'). Put the worst mistake first, " +
	"never plan two lessons for the same mistake. How much harder the next batch gets is decided by the marks " +
	"and given to you below: it is not yours to change. 'summary' is one or two sentences on where the student " +
	"stands. 'prompt' is the brief for the teacher who writes the next batch of exercises: two or three " +
	"sentences addressed to them, naming the points of grammar to drill in order, repeating the step up in " +
	"difficulty word for word, and saying what the sentences should be about - instructions to a colleague, not " +
	"a report on the student. Reply with JSON only, no prose, exactly of the form %[3]s with exactly %[1]d " +
	"lessons."

func planSystemPrompt(count int) string {
	return fmt.Sprintf(planSystem, count, strings.Join(ErrorTypes[1:], ", "), planSchema)
}

// CardLines is the report card as the teacher reads it: the marks, then the mistakes worst first.
func CardLines(card map[string]any) []string {
	passed := fmt.Sprintf("Passed: %d", countOf(card["passed"]))
	if rate := markOf(card["pass_rate"]); rate != nil {
		passed += fmt.Sprintf(" (%.0f%%)", *rate*100)
	}
	lines := []string{fmt.Sprintf("Lessons marked: %d", countOf(card["lessons"])), passed}
	known := []string{}
	for _, mark := range [][2]string{
		{"overall", "mean_score"}, {"grammar", "mean_grammar"}, {"spelling", "mean_spelling"}, {"fluency", "mean_fluency"},
	} {
		if value := markOf(card[mark[1]]); value != nil {
			known = append(known, fmt.Sprintf("%s %.1f", mark[0], *value))
		}
	}
	if len(known) == 0 {
		known = []string{"none recorded"}
	}
	lines = append(lines, "Mean marks out of 10: "+strings.Join(known, ", "))
	points := WeakPoints(card, len(ErrorTypes))
	if len(points) == 0 {
		return append(lines, "Mistakes: none were named.")
	}
	named := make([]string, 0, len(points))
	for _, point := range points {
		text := fmt.Sprintf("%s x%d", point.Error, point.Count)
		if point.Share != nil {
			text += fmt.Sprintf(" (%.0f%% of the lessons)", *point.Share*100)
		}
		named = append(named, text)
	}
	return append(lines, "Mistakes, worst first: "+strings.Join(named, ", "))
}

// ParsePlan turns an LLM answer into a plan of at most count lessons, dropping
// unusable and duplicate entries.  It tolerates the shapes an LLM drifts into:
// {"lessons": [...]}, {"plan": [...]}, a bare list, plain strings, "point" /
// "skill" instead of "focus", "reason" instead of "why", and a "targets" that
// names the mistake in its own words.  How much harder the next batch gets is
// not read from the answer at all: UpgradeFromCard decides it from the marks.
func ParsePlan(raw string, count int, topic string, exercises, drills int) LessonPlan {
	var items []any
	plan := LessonPlan{Lessons: []PlannedLesson{}, Topic: topic, Source: "llm"}
	switch data := loadsLenient(raw).(type) {
	case map[string]any:
		for _, key := range []string{"lessons", "plan", "syllabus", "items", "results"} {
			if list, ok := data[key].([]any); ok {
				items = list
				break
			}
		}
		if items == nil {
			for _, key := range []string{"focus", "point", "skill"} {
				if _, ok := data[key]; ok {
					items = []any{data}
					break
				}
			}
		}
		plan.Summary = clipText(stringField(data, "summary", "assessment", "comment"), maxCommentChars)
		plan.Prompt = clipText(stringField(data, "prompt", "brief", "instructions", "next_prompt"), maxBriefChars)
	case []any:
		items = data
	}
	if items == nil { // plain lines are still a syllabus
		for _, line := range ParseLines(raw, count) {
			items = append(items, map[string]any{"focus": line})
		}
	}
	seen := map[string]bool{}
	for _, entry := range items {
		item, ok := entry.(map[string]any)
		if !ok {
			text, isText := entry.(string)
			if !isText {
				continue
			}
			item = map[string]any{"focus": text}
		}
		focus := clipText(stringField(item, "focus", "point", "skill", "grammar", "lesson", "title"), 60)
		named := stringField(item, "targets", "target", "error", "mistake")
		targets := "none"
		if named != "" {
			targets = ErrorTypeOf(named)
		} else if inferred := ErrorTypeOf(focus); inferred != "none" && inferred != "other" {
			targets = inferred // "plural nouns" is a lesson about plurals whatever it calls itself
		}
		if focus == "" {
			focus = FocusFor(targets)
		}
		if focus == "" || seen[strings.ToLower(focus)] {
			continue
		}
		seen[strings.ToLower(focus)] = true
		subject := clipText(stringField(item, "topic", "subject", "theme"), 60)
		if subject == "" {
			subject = topic
		}
		plan.Lessons = append(plan.Lessons, PlannedLesson{
			Focus:     focus,
			Topic:     subject,
			Why:       clipText(stringField(item, "why", "reason", "because", "rationale", "note"), maxCommentChars),
			Targets:   targets,
			Exercises: exercises,
			Drills:    drills,
			Prefixes:  planPrefixes(item, count),
		})
		if len(plan.Lessons) >= count {
			break
		}
	}
	return plan
}

// planPrefixes reads the openings a planned lesson came with (a list, or lines of text).
func planPrefixes(item map[string]any, count int) []string {
	out := []string{}
	for _, key := range []string{"prefixes", "openings", "examples"} {
		switch value := item[key].(type) {
		case []any:
			for _, entry := range value {
				if text, ok := entry.(string); ok && strings.TrimSpace(text) != "" {
					out = append(out, clipText(text, 120))
				}
			}
		case string:
			for _, line := range ParseLines(value, count) {
				out = append(out, clipText(line, 120))
			}
		default:
			continue
		}
		break
	}
	if len(out) > count {
		out = out[:count]
	}
	return out
}

// PlanRequest is what a plan is asked for: the topic and level of the lessons
// so far, how many lessons to plan and the settings each one carries.
type PlanRequest struct {
	Topic       string
	Level       string
	Words       string  // how long the last batch's openings were (the ladder steps up from here)
	Threshold   float64 // the pass mark it was marked against
	Count       int
	Exercises   int
	Drills      int
	Model       string
	Temperature float64
}

// PlanFromCard is the lesson plan the marks alone imply: one lesson per weak
// point of the card, worst first.  No LLM is involved, so a report card always
// leads to a plan - this is both what PlanLessons asks the teacher to improve
// on and what it falls back to when the answer cannot be read.  The step up
// (UpgradeFromCard) and the brief for the next batch (PlanBrief) come with it.
func PlanFromCard(card map[string]any, req PlanRequest) LessonPlan {
	count := req.Count
	if count < 1 {
		count = DefaultPlanLessons
	}
	weak := WeakPoints(card, count)
	upgrade := UpgradeFromCard(card, req.Level, req.Words, req.Threshold, req.Drills)
	upgradeDrills, _ := upgrade["drills"].(int)
	lessons := []PlannedLesson{}
	total := countOf(card["lessons"])
	for _, point := range weak {
		focus := point.Focus
		if focus == "" {
			focus = point.Error
		}
		why := fmt.Sprintf("%d of %d lessons", point.Count, total)
		if point.Share != nil {
			why += fmt.Sprintf(" (%.0f%%)", *point.Share*100)
		}
		if point.Count == 1 {
			why += " was"
		} else {
			why += " were"
		}
		lessons = append(lessons, PlannedLesson{
			Focus: focus, Topic: req.Topic, Why: why + fmt.Sprintf(" marked down for %s.", point.Error),
			Targets: point.Error, Exercises: req.Exercises, Drills: upgradeDrills, Prefixes: []string{},
		})
	}
	if len(lessons) == 0 {
		lessons = append(lessons, PlannedLesson{
			Focus: "", Topic: req.Topic, Targets: "none", Exercises: req.Exercises, Drills: upgradeDrills,
			Why:      "No mistake was named, so the next lessons stay on the topic and take the step up instead.",
			Prefixes: []string{},
		})
	}
	summary := fmt.Sprintf("%d of %d lessons passed", countOf(card["passed"]), total)
	if score := markOf(card["mean_score"]); score != nil {
		summary += fmt.Sprintf(" at a mean of %.1f out of 10", *score)
	}
	if len(weak) > 0 {
		names := make([]string, len(weak))
		for i, point := range weak {
			names[i] = point.Error
		}
		summary += "; the weakest points are " + strings.Join(names, ", ")
	} else {
		summary += "; nothing was marked down"
	}
	level, _ := upgrade["level"].(string)
	return LessonPlan{
		Lessons: lessons, Summary: summary + ".", Prompt: PlanBrief(weak, upgrade, req.Topic), Upgrade: upgrade,
		Level: level, Topic: req.Topic, Weak: weak, Source: "report card",
	}
}

// PlanLessons asks the teacher for the next lessons, given the report card of
// the ones just marked and the step up in difficulty the marks have earned:
// one point of grammar per lesson, the mistake it repairs, a topic, a line on
// why, and Prompt - the brief for the next batch, which TutorConfig.Brief
// hands to the exercise writer of the run that follows.
//
// PlanFromCard is the floor.  A weakness the teacher's plan does not target
// takes the place of a lesson that drills nothing the card marked down (and is
// appended when there is no such lesson), so every mistake on the card is
// somebody's lesson; a brief it does not write is the one the marks wrote; an
// answer that cannot be read - or that names no mistake and says nothing about
// the student - leaves the whole plan as it stands.  Source says which of the
// two wrote it.  The upgrade is never the answer's to invent: UpgradeFromCard
// computes it from the marks and the teacher is asked to repeat it.
func PlanLessons(client LLMClient, card map[string]any, req PlanRequest) (LessonPlan, error) {
	if req.Count < 1 {
		return LessonPlan{}, fmt.Errorf("count must be >= 1")
	}
	if countOf(card["lessons"]) < 1 {
		return LessonPlan{}, fmt.Errorf("a report card of at least one lesson is needed to plan the next lessons")
	}
	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}
	fallback := PlanFromCard(card, req)
	lines := []string{}
	if strings.TrimSpace(req.Topic) != "" {
		lines = append(lines, "The lessons so far were about: "+strings.TrimSpace(req.Topic))
	}
	lines = append(lines, CardLines(card)...)
	level := strings.TrimSpace(req.Level)
	if level == "" {
		level = Levels[0]
	}
	words := strings.Join(strings.Fields(req.Words), " ")
	if words == "" {
		words = WordsLadder[0]
	}
	note, _ := fallback.Upgrade["note"].(string)
	lines = append(lines,
		fmt.Sprintf("Level so far: %s, openings of %s words, pass mark %.0f out of 10", level, words, req.Threshold),
		"The step up this batch has earned, to repeat in your brief: "+note,
		fmt.Sprintf("Plan the next %d lesson(s) now.", req.Count))
	raw, err := client.Generate(strings.Join(lines, "\n"), LLMOptions{
		System: planSystemPrompt(req.Count), Model: req.Model, JSON: true, Temperature: temperature,
	})
	if err != nil {
		return LessonPlan{}, err
	}
	plan := ParsePlan(raw, req.Count, req.Topic, req.Exercises, req.Drills)
	if len(plan.Lessons) == 0 || (plan.Summary == "" && len(plan.Targets()) == 0) {
		return fallback, nil // nothing usable came back: the marks alone still plan the lessons
	}
	plan.Source = ProviderOf(client)
	plan.Weak = fallback.Weak
	plan.Topic = req.Topic
	if plan.Summary == "" {
		plan.Summary = fallback.Summary
	}
	if plan.Prompt == "" {
		plan.Prompt = fallback.Prompt
	}
	plan.Upgrade = fallback.Upgrade // the marks decide the step up, not the answer
	plan.Level = fallback.Level
	upgradeDrills, _ := plan.Upgrade["drills"].(int)
	covered, marked := map[string]bool{}, map[string]bool{}
	for _, point := range fallback.Weak {
		marked[point.Error] = true
	}
	for i := range plan.Lessons {
		if plan.Lessons[i].Topic == "" {
			plan.Lessons[i].Topic = req.Topic
		}
		plan.Lessons[i].Drills = upgradeDrills
		if plan.Lessons[i].Why == "" && plan.Lessons[i].Targets != "none" {
			for _, lesson := range fallback.Lessons {
				if lesson.Targets == plan.Lessons[i].Targets {
					plan.Lessons[i].Why = lesson.Why
					break
				}
			}
		}
		if plan.Lessons[i].Targets != "none" {
			covered[plan.Lessons[i].Targets] = true
		}
	}
	// a weakness the teacher's plan skipped takes the place of a lesson that drills nothing the card marked down
	spare := []int{}
	for i, lesson := range plan.Lessons {
		if !marked[lesson.Targets] {
			spare = append(spare, i)
		}
	}
	for _, lesson := range fallback.Lessons {
		if lesson.Targets == "none" || covered[lesson.Targets] {
			continue
		}
		covered[lesson.Targets] = true
		if len(spare) > 0 {
			plan.Lessons[spare[0]] = lesson
			spare = spare[1:]
		} else {
			plan.Lessons = append(plan.Lessons, lesson)
		}
	}
	return plan, nil
}
