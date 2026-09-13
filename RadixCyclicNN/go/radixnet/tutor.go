package radixnet

// Automated English lessons: the teacher sets the exercise, the network
// completes it, the teacher marks it.  The teacher is any provider client - a
// local Ollama model or ChatGPT (TutorConfig.TutorProvider) - and the marking
// follows it unless GraderProvider names the other one.
//
//	topic -> prefix (LLM) -> completion (the prediction search) -> grade (LLM) -> 2NRL
//
// The Go port of radixnet/tutor.py, working the same way and speaking the same
// JSON.  One lesson is the whole prediction process run without a human:
//
//  1. the LLM writes sentence openings about a topic, each drilling one point
//     of grammar and each with its own model answer (WriteExercises);
//  2. the network continues the prefix with the ordinary prediction search;
//  3. the LLM marks the finished sentence as an English teacher
//     (GradeCompletions): grammar, spelling and fluency out of 10, the worst
//     mistake named from ErrorTypes, one line of teaching and the correction -
//     the same sentence written out in correct English;
//  4. failed sentences become 2NRL garbage weighted by how bad the mark was
//     and the corrections the reward pass, weighted by how good it was, so a
//     rating decides how much the network learns rather than a single like.
//
// Grammar is what is being taught, so grammar is most of the mark:
// score = GrammarWeight * grammar + (1 - GrammarWeight) * mean(spelling,
// fluency).  The mistakes of a round add up to a report card; with Adapt on,
// the next round's exercises drill the weakest points, and Drills asks the
// teacher for extra correct sentences about them.

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// DefaultTutorModelName is the teacher model when neither the caller nor the environment names one.
func DefaultTutorModel() string {
	if name := strings.TrimSpace(os.Getenv("RADIXNET_TUTOR_MODEL")); name != "" {
		return name
	}
	return DefaultOllamaModel()
}

// ErrorTypes are the mistakes a completion is marked with; "none" is a clean sentence.
var ErrorTypes = []string{
	"none", "agreement", "tense", "article", "preposition", "plural", "pronoun",
	"word-order", "spelling", "punctuation", "vocabulary", "fragment", "nonsense",
}

// TutorModes are the ways the network may complete a prefix (the count model beams; "dijkstra" is its alias).
var TutorModes = []string{"beam", "dijkstra", "sample"}

// TwoNRLPer says when the grades are applied: once per round, or after every lesson.
var TwoNRLPer = []string{"round", "lesson"}

// TeacherWeight is the reward weight of text the teacher wrote (a correction,
// a model answer, a drill): correct by construction, so always the full rate.
const TeacherWeight = 1.0

const maxCommentChars = 300

const gradeSchema = `{"grades": [{"index": <int>, "grammar": <0-10>, "spelling": <0-10>, "fluency": <0-10>, ` +
	`"error": "<one of the error types>", "correction": "<the whole sentence in correct English>", ` +
	`"comment": "<one sentence of teaching>"}, ...]}`

// Cue is what the search is actually given: the prefix with exactly one trailing space, so a new word follows.
func Cue(prefix string) string {
	text := strings.TrimRight(prefix, " \t\r\n")
	if text == "" {
		return ""
	}
	return text + " "
}

func clipText(text string, limit int) string {
	flat := strings.Join(strings.Fields(text), " ")
	if len(flat) <= limit {
		return flat
	}
	return strings.TrimRight(flat[:limit-1], " ") + "…"
}

// ErrorTypeOf maps a reported mistake onto ErrorTypes ("other" for anything unrecognised).
func ErrorTypeOf(value string) string {
	text := strings.ToLower(strings.TrimSpace(value))
	text = strings.ReplaceAll(strings.ReplaceAll(text, "_", "-"), " ", "-")
	switch text {
	case "", "no-error", "no-errors", "correct", "ok", "n/a":
		return "none"
	}
	for _, known := range ErrorTypes {
		if text == known {
			return text
		}
	}
	for _, known := range ErrorTypes[1:] { // "subject-verb-agreement" -> "agreement"
		if strings.Contains(text, known) {
			return known
		}
	}
	return "other"
}

// OverallScore is one mark out of 10 from the three sub-marks, grammar
// carrying grammarWeight of it; missing sub-marks (nil) drop out.
func OverallScore(grammar, spelling, fluency *float64, grammarWeight float64) *float64 {
	rest := []float64{}
	for _, value := range []*float64{spelling, fluency} {
		if value != nil {
			rest = append(rest, *value)
		}
	}
	var meanRest *float64
	if len(rest) > 0 {
		sum := 0.0
		for _, v := range rest {
			sum += v
		}
		mean := sum / float64(len(rest))
		meanRest = &mean
	}
	switch {
	case grammar == nil:
		return meanRest
	case meanRest == nil:
		return grammar
	}
	score := grammarWeight*(*grammar) + (1.0-grammarWeight)*(*meanRest)
	return &score
}

func clampScore(value float64) float64 { return math.Max(0, math.Min(10, value)) }

// -- the exercise ----------------------------------------------------------

// Exercise is one sentence opening for the network to finish, with the point it drills and a model answer.
type Exercise struct {
	ID     string `json:"id"`
	Prefix string `json:"prefix"`
	Focus  string `json:"focus"`  // the grammar point: "past tense", "plural nouns", ...
	Answer string `json:"answer"` // the teacher's own complete sentence, taught when the network fails
}

// Cue is Cue(e.Prefix): what the prediction search is given.
func (e Exercise) Cue() string { return Cue(e.Prefix) }

const exerciseSystem = "You are an English teacher writing exercises for a beginner student who completes sentences: you give the " +
	"opening words, the student writes the rest. Every exercise is one unfinished sentence in plain, simple, " +
	"modern English. The prefix must be %s words, must NOT end with punctuation and must be genuinely " +
	"unfinished, so that finishing it correctly needs the point of grammar you are drilling. Each exercise also " +
	"carries 'focus' (the point of grammar in two or three words, e.g. 'subject-verb agreement', 'past tense', " +
	"'plural nouns', 'articles', 'prepositions of place') and 'answer' (the whole sentence, prefix included, " +
	"finished correctly by you - short, natural and factual). Vary the focus and the vocabulary across the " +
	"exercises. Reply with JSON only, no prose, exactly of the form " +
	`{"exercises": [{"prefix": "...", "focus": "...", "answer": "..."}, ...]} with exactly %d entries.`

func stringField(item map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := item[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.Join(strings.Fields(value), " ")
		}
	}
	return ""
}

// ParseExercises turns an LLM answer into at most count exercises, dropping
// unusable and duplicate entries.  It tolerates the shapes an LLM drifts into:
// {"exercises": [...]}, a bare list, plain strings, "opening" / "stem"
// instead of "prefix", and an answer that omits the prefix.
func ParseExercises(raw string, count, round int) []Exercise {
	if round < 1 {
		round = 1
	}
	var items []any
	switch data := loadsLenient(raw).(type) {
	case map[string]any:
		for _, key := range []string{"exercises", "prefixes", "items", "results"} {
			if list, ok := data[key].([]any); ok {
				items = list
				break
			}
		}
		if items == nil {
			for _, key := range []string{"prefix", "opening", "stem"} {
				if _, ok := data[key]; ok {
					items = []any{data}
					break
				}
			}
		}
	case []any:
		items = data
	}
	if items == nil { // plain lines are still usable
		for _, line := range ParseLines(raw, count) {
			items = append(items, map[string]any{"prefix": line})
		}
	}
	exercises := []Exercise{}
	seen := map[string]bool{}
	for _, entry := range items {
		item, ok := entry.(map[string]any)
		if !ok {
			text, isText := entry.(string)
			if !isText {
				continue
			}
			item = map[string]any{"prefix": text}
		}
		prefix := strings.TrimRight(stringField(item, "prefix", "opening", "stem", "start", "sentence"), " .,:;!?-–—_")
		if prefix == "" || seen[strings.ToLower(prefix)] {
			continue
		}
		seen[strings.ToLower(prefix)] = true
		answer := stringField(item, "answer", "solution", "completion", "full_sentence", "example")
		if answer != "" && !strings.HasPrefix(strings.ToLower(answer), strings.ToLower(prefix)) {
			answer = Cue(prefix) + strings.TrimLeft(answer, " ") // the teacher answered with the continuation only
		}
		exercises = append(exercises, Exercise{
			ID:     fmt.Sprintf("r%de%d", round, len(exercises)+1),
			Prefix: prefix,
			Focus:  clipText(stringField(item, "focus", "point", "skill"), 60),
			Answer: answer,
		})
		if len(exercises) >= count {
			break
		}
	}
	return exercises
}

// ExerciseRequest is what the teacher is asked for.
type ExerciseRequest struct {
	Topic       string
	Count       int
	Focus       string   // pin every exercise to one point of grammar
	Level       string   // "beginner", "intermediate", ...
	Weak        []string // the mistakes the student keeps making
	Words       string   // how long a prefix is, e.g. "3 to 6"
	Brief       string   // the plan this batch of lessons is being taught to (LessonPlan.Prompt)
	Model       string
	Temperature float64
}

// WriteExercises asks the teacher for sentence openings about a topic.
func WriteExercises(client LLMClient, req ExerciseRequest) ([]Exercise, error) {
	if req.Count < 1 {
		return nil, fmt.Errorf("count must be >= 1")
	}
	topic := strings.TrimSpace(req.Topic)
	if topic == "" {
		return nil, fmt.Errorf("topic must not be empty")
	}
	words := req.Words
	if strings.TrimSpace(words) == "" {
		words = "3 to 6"
	}
	level := strings.TrimSpace(req.Level)
	if level == "" {
		level = "beginner"
	}
	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.9
	}
	lines := []string{"Topic: " + topic, "Level: " + level}
	if brief := strings.Join(strings.Fields(req.Brief), " "); brief != "" {
		lines = append(lines, "The plan for this batch of lessons: "+brief)
	}
	if strings.TrimSpace(req.Focus) != "" {
		lines = append(lines, "Every exercise must drill: "+strings.TrimSpace(req.Focus))
	}
	if weak := usefulWeak(req.Weak); len(weak) > 0 {
		lines = append(lines, "The student keeps making these mistakes, so drill them: "+strings.Join(weak, ", ")+".")
	}
	lines = append(lines, fmt.Sprintf("Write the %d exercises now.", req.Count))
	raw, err := client.Generate(strings.Join(lines, "\n"), LLMOptions{
		System: fmt.Sprintf(exerciseSystem, words, req.Count), Model: req.Model, JSON: true, Temperature: temperature,
	})
	if err != nil {
		return nil, err
	}
	exercises := ParseExercises(raw, req.Count, 1)
	if len(exercises) == 0 {
		name := req.Model
		if name == "" {
			name = client.ModelName()
		}
		return nil, llmErrorf("the teacher model %q returned no usable exercises", name)
	}
	return exercises, nil
}

func usefulWeak(weak []string) []string {
	out := []string{}
	for _, name := range weak {
		if name != "" && name != "none" {
			out = append(out, name)
		}
		if len(out) >= 5 {
			break
		}
	}
	return out
}

const drillSystem = "You are an English teacher writing model sentences for a beginner student to imitate. Answer with exactly " +
	"%d lines and nothing else: one short, correct, natural sentence per line, plain text, no numbering, no " +
	"quotes, no commentary. Every sentence must be simple, factual and grammatically perfect, because the " +
	"student learns English by copying them."

// DrillSentences asks for correct example sentences about the topic, demonstrating the weak points.
func DrillSentences(client LLMClient, topic string, count int, weak []string, model string) ([]string, error) {
	if count < 1 {
		return nil, nil
	}
	user := "Topic: " + strings.TrimSpace(topic) + "\n"
	if useful := usefulWeak(weak); len(useful) > 0 {
		user += "Each sentence must clearly demonstrate the correct use of: " + strings.Join(useful, ", ") + ".\n"
	}
	user += fmt.Sprintf("Write the %d sentences now.", count)
	raw, err := client.Generate(user, LLMOptions{
		System: fmt.Sprintf(drillSystem, count), Model: model, Temperature: 0.8,
	})
	if err != nil {
		return nil, err
	}
	return ParseLines(raw, count), nil
}

// -- the grade -------------------------------------------------------------

// Grade is the teacher's marking of one completed sentence.
type Grade struct {
	Score      *float64 `json:"score"` // 0-10 overall (grammar-weighted), nil when the answer could not be read
	Grammar    *float64 `json:"grammar"`
	Spelling   *float64 `json:"spelling"`
	Fluency    *float64 `json:"fluency"`
	Passed     bool     `json:"passed"`
	Error      string   `json:"error"`      // one of ErrorTypes, or "other"
	Correction string   `json:"correction"` // the whole sentence in correct English: what the network is taught
	Comment    string   `json:"comment"`    // one sentence of teaching
	GradedBy   string   `json:"graded_by"`  // the marking provider ("ollama" | "chatgpt"), "empty" or "unrated"
}

// Lesson is one exercise, one completion by the network and one grade.
type Lesson struct {
	Exercise     Exercise `json:"exercise"`
	Attempt      int      `json:"attempt"` // 0-based: attempt 0 uses the configured mode, later ones are sampled
	Mode         string   `json:"mode"`
	Continuation string   `json:"continuation"`
	Sentence     string   `json:"sentence"` // Cue(prefix) + continuation: what is graded and, when it passes, learned
	Cost         float64  `json:"cost"`
	Probability  float64  `json:"probability"`
	ReachedEnd   bool     `json:"reached_end"`
	Seconds      float64  `json:"seconds"`
	Grade        Grade    `json:"grade"`
	// Changes is what the teacher changed, span by span; filled in when the lesson is graded.
	Changes []Edit `json:"changes"`
}

// Empty reports whether the network wrote nothing at all.
func (l *Lesson) Empty() bool { return strings.TrimSpace(l.Continuation) == "" }

// ChangesOfLesson is what the teacher changed in one lesson (empty when it passed or was left uncorrected).
func ChangesOfLesson(lesson *Lesson) []Edit {
	grade := lesson.Grade
	if grade.Passed || strings.TrimSpace(grade.Correction) == "" || strings.TrimSpace(lesson.Sentence) == "" {
		return []Edit{}
	}
	return DiffSummary(strings.TrimSpace(lesson.Sentence), strings.TrimSpace(grade.Correction), 8)
}

const gradeSystem = "You are a strict but constructive English teacher marking sentence completions. The student is a beginner " +
	"learning English: it is given the opening words of a sentence (the prefix, shown in <<>>) and writes the " +
	"rest. Mark what the student wrote, judged as part of the whole sentence. For each completion give three " +
	"marks out of 10 - grammar (agreement, tense, articles, prepositions, word order, sentence structure), " +
	"spelling and fluency (does it read like natural English) - and name the single most important mistake as " +
	"one of: %s. Use \"none\" only for a sentence a teacher would accept as it stands. A completion that " +
	"is empty, cut off, gibberish or not English scores 0 with the error \"nonsense\" or \"fragment\". " +
	"'correction' is the whole sentence written out in correct English: keep the prefix word for word, change " +
	"only what follows it, stay as close to what the student wrote as the mistake allows, and finish the " +
	"sentence properly - the student learns English by being shown this sentence, so it must be correct and " +
	"natural on its own. 'comment' is one short sentence of teaching addressed to the student, naming the rule " +
	"that was broken. Reply with JSON only, no prose, exactly of the form %s with one entry per " +
	"completion, in the given order and with the given index."

func numberField(item map[string]any, keys ...string) *float64 {
	for _, key := range keys {
		value, ok := item[key]
		if !ok {
			continue
		}
		switch v := value.(type) {
		case float64:
			if math.IsNaN(v) {
				continue
			}
			score := clampScore(v)
			return &score
		case string:
			text := strings.TrimSpace(strings.Split(v, "/")[0])
			var parsed float64
			if _, err := fmt.Sscanf(text, "%g", &parsed); err == nil && !math.IsNaN(parsed) {
				score := clampScore(parsed)
				return &score
			}
		}
	}
	return nil
}

// ParseGrades reads the marks of an LLM answer, keyed by the index of the
// completion they belong to; gradedBy (empty: the default provider) is the
// provider that gave them and ends up in every grade.
func ParseGrades(raw string, count int, grammarWeight, threshold float64, gradedBy string) map[int]Grade {
	if gradedBy == "" {
		gradedBy = DefaultProvider
	}
	grades := map[int]Grade{}
	var items []any
	switch data := loadsLenient(raw).(type) {
	case map[string]any:
		for _, key := range []string{"grades", "reviews", "results", "items", "marks"} {
			if list, ok := data[key].([]any); ok {
				items = list
				break
			}
		}
		if items == nil {
			for _, key := range []string{"grammar", "score", "correction"} {
				if _, ok := data[key]; ok {
					items = []any{data}
					break
				}
			}
		}
	case []any:
		items = data
	}
	for position, entry := range items {
		item, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		index := position
		if value := numberFieldRaw(item, "index"); value != nil {
			index = int(*value)
		}
		if index < 0 || index >= count {
			continue
		}
		if _, seen := grades[index]; seen {
			continue
		}
		grammar := numberField(item, "grammar", "grammar_score")
		spelling := numberField(item, "spelling", "spelling_score")
		fluency := numberField(item, "fluency", "fluency_score", "naturalness")
		score := OverallScore(grammar, spelling, fluency, grammarWeight)
		if score == nil {
			score = numberField(item, "score", "rating")
		}
		if score == nil {
			continue // nothing to mark with: leave it unrated
		}
		grades[index] = Grade{
			Score: score, Grammar: grammar, Spelling: spelling, Fluency: fluency, Passed: *score >= threshold,
			Error:      ErrorTypeOf(stringField(item, "error", "error_type", "mistake")),
			Correction: stringField(item, "correction", "corrected", "fixed"),
			Comment:    clipText(stringField(item, "comment", "critique", "feedback", "reason"), maxCommentChars),
			GradedBy:   gradedBy,
		}
	}
	return grades
}

// numberFieldRaw reads a number without clamping it to 0..10 (indices).
func numberFieldRaw(item map[string]any, key string) *float64 {
	switch v := item[key].(type) {
	case float64:
		if math.IsNaN(v) {
			return nil
		}
		return &v
	case string:
		var parsed float64
		if _, err := fmt.Sscanf(strings.TrimSpace(v), "%g", &parsed); err == nil {
			return &parsed
		}
	}
	return nil
}

// GradeOptions are the knobs of the marking.
type GradeOptions struct {
	Topic         string
	Threshold     float64
	GrammarWeight float64
	Model         string
	Batch         int
	Temperature   float64
	GradedBy      string // the provider behind the client (empty: the client's own)
}

// GradeCompletions marks every lesson in place, in batches of Batch.
//
// An empty completion is failed without asking (GradedBy "empty"), with the
// teacher's model answer as the correction; a lesson the LLM said nothing
// usable about keeps a nil score and GradedBy "unrated" and counts as a
// failure.
func GradeCompletions(client LLMClient, lessons []*Lesson, o GradeOptions) error {
	batch := o.Batch
	if batch < 1 {
		return fmt.Errorf("batch must be >= 1")
	}
	gradedBy := o.GradedBy
	if gradedBy == "" {
		gradedBy = ProviderOf(client)
	}
	temperature := o.Temperature
	if temperature <= 0 {
		temperature = 0.2
	}
	system := fmt.Sprintf(gradeSystem, quotedErrorTypes(), gradeSchema)
	for start := 0; start < len(lessons); start += batch {
		end := start + batch
		if end > len(lessons) {
			end = len(lessons)
		}
		chunk := lessons[start:end]
		asked := []int{}
		body := make([]string, 0, len(chunk))
		for i, lesson := range chunk {
			if lesson.Empty() {
				lesson.Grade = Grade{
					Score: floatPtr(0), Grammar: floatPtr(0), Spelling: floatPtr(0), Fluency: floatPtr(0),
					Passed: false, Error: "nonsense", Correction: lesson.Exercise.Answer,
					Comment: "Nothing was written: the sentence has to be finished.", GradedBy: "empty",
				}
				continue
			}
			asked = append(asked, i)
			line := fmt.Sprintf("[%d] <<%s>>%s", i, lesson.Exercise.Cue(), lesson.Continuation)
			if lesson.Exercise.Focus != "" {
				line += "   (drilling: " + lesson.Exercise.Focus + ")"
			}
			body = append(body, line)
		}
		if len(asked) == 0 {
			continue
		}
		user := ""
		if strings.TrimSpace(o.Topic) != "" {
			user = "Topic of the lesson: " + strings.TrimSpace(o.Topic) + "\n\n"
		}
		user += fmt.Sprintf("Mark these %d completions:\n%s\n\nReturn the JSON now.", len(asked), strings.Join(body, "\n"))
		raw, err := client.Generate(user, LLMOptions{
			System: system, Model: o.Model, JSON: true, Temperature: temperature,
		})
		if err != nil {
			return err
		}
		parsed := ParseGrades(raw, len(chunk), o.GrammarWeight, o.Threshold, gradedBy)
		for _, i := range asked {
			lesson := chunk[i]
			grade, ok := parsed[i]
			if !ok {
				lesson.Grade = Grade{
					Passed: false, Error: "other", Correction: lesson.Exercise.Answer,
					Comment: "no grade returned", GradedBy: "unrated",
				}
				continue
			}
			if strings.TrimSpace(grade.Correction) == "" {
				if grade.Passed {
					grade.Correction = lesson.Sentence
				} else {
					grade.Correction = lesson.Exercise.Answer
				}
			}
			lesson.Grade = grade
		}
	}
	for _, lesson := range lessons { // what the teacher changed rides along with the lesson
		lesson.Changes = ChangesOfLesson(lesson)
	}
	return nil
}

func quotedErrorTypes() string {
	quoted := make([]string, len(ErrorTypes))
	for i, name := range ErrorTypes {
		quoted[i] = `"` + name + `"`
	}
	return strings.Join(quoted, ", ")
}

func floatPtr(v float64) *float64 { return &v }

// ReportCard sums up a set of lessons: the marks, the pass rate, the mistakes and the weakest points.
func ReportCard(lessons []*Lesson) map[string]any {
	scores, grammar, spelling, fluency := []float64{}, []float64{}, []float64{}, []float64{}
	errors := map[string]int{}
	passed := 0
	for _, lesson := range lessons {
		grade := lesson.Grade
		if grade.Score != nil {
			scores = append(scores, *grade.Score)
			appendIf(&grammar, grade.Grammar)
			appendIf(&spelling, grade.Spelling)
			appendIf(&fluency, grade.Fluency)
		}
		if grade.Passed {
			passed++
		}
		if grade.Error != "" && grade.Error != "none" {
			errors[grade.Error]++
		}
	}
	card := map[string]any{
		"lessons": len(lessons), "graded": len(scores), "passed": passed, "failed": len(lessons) - passed,
		"pass_rate": nil, "mean_score": mean(scores), "mean_grammar": mean(grammar),
		"mean_spelling": mean(spelling), "mean_fluency": mean(fluency),
		"errors": errors, "weakest": weakest(errors),
	}
	if len(lessons) > 0 {
		card["pass_rate"] = float64(passed) / float64(len(lessons))
	}
	return card
}

func appendIf(values *[]float64, value *float64) {
	if value != nil {
		*values = append(*values, *value)
	}
}

func mean(values []float64) any {
	if len(values) == 0 {
		return nil
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// weakest names the three most frequent mistakes, most frequent first (ties alphabetically).
func weakest(errors map[string]int) []string {
	names := make([]string, 0, len(errors))
	for name := range errors {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if errors[names[i]] != errors[names[j]] {
			return errors[names[i]] > errors[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > 3 {
		names = names[:3]
	}
	return names
}

// -- the loop --------------------------------------------------------------

// TutorConfig holds the settings of a tutoring run.
type TutorConfig struct {
	Topic     string `json:"topic"`
	Rounds    int    `json:"rounds"`
	Exercises int    `json:"exercises"` // sentence openings per round
	Attempts  int    `json:"attempts"`  // completions the network writes per exercise
	Focus     string `json:"focus"`     // pin every exercise to one point of grammar
	Level     string `json:"level"`
	Words     string `json:"words"`
	// Brief is what this batch of lessons is being taught to: the previous batch's plan (LessonPlan.Prompt).
	Brief string `json:"brief"`
	// TutorProvider names who teaches: "ollama" (a local model) or "chatgpt" (OpenAI).
	TutorProvider string `json:"tutor_provider"`
	TutorModel    string `json:"tutor_model"`
	// GraderProvider marks with the other provider; empty means the teacher's own.
	GraderProvider string `json:"grader_provider"`
	GraderModel    string `json:"grader_model"` // a different model for the marking
	// the completion
	Mode        string  `json:"mode"`
	Length      int     `json:"length"`
	MaxLength   int     `json:"max_length"`
	Temperature float64 `json:"temperature"`
	ToEnd       bool    `json:"to_end"`
	Beam        int     `json:"beam"`
	K           int     `json:"k"`
	// the marking
	Threshold     float64 `json:"threshold"`
	GrammarWeight float64 `json:"grammar_weight"`
	Batch         int     `json:"batch"`
	Adapt         bool    `json:"adapt"`        // drill the previous round's weakest points
	Drills        int     `json:"drills"`       // extra correct example sentences per round
	Plan          int     `json:"plan"`         // lessons to plan from the final report card (0 = no plan)
	TeachAnswer   bool    `json:"teach_answer"` // a failed lesson also learns the teacher's model answer
	Learn         bool    `json:"learn"`        // false: a dry run - the grades are reported, nothing is trained
	// 2NRL
	TwoNRLPer string `json:"twonrl_per"`
	// DiffCorrections teaches a correction from its diff with the sentence the network wrote
	// instead of as two whole sentences: only the trigram nodes they disagree on move.
	DiffCorrections bool `json:"diff_corrections"`
	// KeepWeight is what the unchanged part of a correction still earns (1 = the whole sentence, as before).
	KeepWeight float64 `json:"keep_weight"`
	MinWeight  float64 `json:"min_weight"` // negative-phase weight of a near miss (a hopeless answer weighs 1)
	NegEpochs  int     `json:"neg_epochs"`
	PosEpochs  int     `json:"pos_epochs"`
	Strength   float64 `json:"strength"`
	Replay     bool    `json:"replay"`
	// ReplayLimit caps the corrections kept for the replay (0 = no limit).
	ReplayLimit int `json:"replay_limit"`
}

// DefaultTutorConfig mirrors the Python defaults.
func DefaultTutorConfig() TutorConfig {
	return TutorConfig{
		Topic: "everyday life", Rounds: 3, Exercises: 5, Attempts: 1, Level: "beginner", Words: "3 to 6",
		TutorProvider: DefaultProvider, TutorModel: DefaultTutorModel(),
		Mode: "beam", Length: 20, MaxLength: 80, Temperature: 1.0, ToEnd: true,
		K: 5, Threshold: 6.0, GrammarWeight: 0.6, Batch: 10, Adapt: true, TeachAnswer: true, Learn: true,
		TwoNRLPer: "round", DiffCorrections: true, KeepWeight: 0.25, MinWeight: 0.25, NegEpochs: 2, PosEpochs: 3,
		Strength: 1.0, Replay: true, ReplayLimit: 64,
	}
}

// Resolve normalises the providers and fills in the models they imply, the way
// the Python TutorConfig.__post_init__ does; Validate calls it first.
func (c *TutorConfig) Resolve() error {
	tutor, err := NormaliseProvider(c.TutorProvider)
	if err != nil {
		return fmt.Errorf("tutor_provider: %v", err)
	}
	c.TutorProvider = tutor
	if strings.TrimSpace(c.GraderProvider) == "" {
		c.GraderProvider = tutor
	} else {
		grader, err := NormaliseProvider(c.GraderProvider)
		if err != nil {
			return fmt.Errorf("grader_provider: %v", err)
		}
		c.GraderProvider = grader
	}
	if strings.TrimSpace(c.TutorModel) == "" {
		c.TutorModel = DefaultProviderModel(c.TutorProvider)
	}
	return nil
}

// ResolvedGraderModel is the model the marking runs on: GraderModel, else the
// teacher's model on the teacher's provider, else that provider's default.
func (c *TutorConfig) ResolvedGraderModel() string {
	if strings.TrimSpace(c.GraderModel) != "" {
		return c.GraderModel
	}
	if c.GraderProvider == "" || c.GraderProvider == c.TutorProvider {
		return c.TutorModel
	}
	return DefaultProviderModel(c.GraderProvider)
}

// Validate checks the settings the way the Python TutorConfig.validate does.
func (c *TutorConfig) Validate() error {
	if err := c.Resolve(); err != nil {
		return err
	}
	if strings.TrimSpace(c.Topic) == "" {
		return fmt.Errorf("topic must not be empty")
	}
	if c.Rounds < 1 {
		return fmt.Errorf("rounds must be >= 1")
	}
	if c.Exercises < 1 {
		return fmt.Errorf("exercises must be >= 1")
	}
	if c.Attempts < 1 {
		return fmt.Errorf("attempts must be >= 1")
	}
	if !contains(TutorModes, c.Mode) {
		return fmt.Errorf("mode must be one of %s", strings.Join(TutorModes, ", "))
	}
	if !contains(TwoNRLPer, c.TwoNRLPer) {
		return fmt.Errorf("twonrl_per must be one of %s", strings.Join(TwoNRLPer, ", "))
	}
	if c.Length < 0 || c.MaxLength < 1 {
		return fmt.Errorf("length must be >= 0 and max_length >= 1")
	}
	if c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if c.Beam < 0 || c.K < 0 {
		return fmt.Errorf("beam and k must be >= 0")
	}
	if c.Threshold < 0 || c.Threshold > 10 {
		return fmt.Errorf("threshold must lie in [0, 10]")
	}
	if c.GrammarWeight < 0 || c.GrammarWeight > 1 {
		return fmt.Errorf("grammar_weight must lie in [0, 1]")
	}
	if c.MinWeight < 0 || c.MinWeight > 1 {
		return fmt.Errorf("min_weight must lie in [0, 1]")
	}
	if c.KeepWeight < 0 || c.KeepWeight > 1 {
		return fmt.Errorf("keep_weight must lie in [0, 1]")
	}
	if c.Batch < 1 {
		return fmt.Errorf("batch must be >= 1")
	}
	if c.Drills < 0 || c.ReplayLimit < 0 {
		return fmt.Errorf("drills and replay_limit must be >= 0")
	}
	if c.Plan < 0 {
		return fmt.Errorf("plan must be >= 0")
	}
	if c.NegEpochs < 0 || c.PosEpochs < 0 {
		return fmt.Errorf("epochs must be >= 0")
	}
	if c.Strength < 0 {
		return fmt.Errorf("strength must be >= 0")
	}
	return nil
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

// TutorTrainer runs the lessons: the LLM sets and marks the exercises, the
// network completes them and learns from the grades.
type TutorTrainer struct {
	Model  *Model
	Client LLMClient
	// GraderClient marks the completions; nil means the teacher marks its own.
	GraderClient LLMClient
	Config       TutorConfig
	History      []map[string]any
	Lessons      []*Lesson
	Weak         []string
	// Progress receives every lesson, round and report record as it happens.
	Progress func(map[string]any)
	// Stop is polled between steps; a true answer ends the run cleanly.
	Stop func() bool
	// External wraps the slow LLM calls (the server releases its lock around them).
	External func(func() error) error

	replay []string
}

// NewTutorTrainer validates the configuration and returns a trainer whose
// teacher is client; the marking shares it unless GraderClient is set
// afterwards or the configuration names the other provider, in which case a
// client for it is built from the environment.
func NewTutorTrainer(model *Model, client LLMClient, config TutorConfig) (*TutorTrainer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	trainer := &TutorTrainer{Model: model, Client: client, Config: config}
	if config.GraderProvider != "" && config.GraderProvider != ProviderOf(client) {
		grader, err := NewLLMClient(config.GraderProvider, "", config.ResolvedGraderModel(), 0)
		if err != nil {
			return nil, err
		}
		trainer.GraderClient = grader
	}
	return trainer, nil
}

// grader is the client that marks the completions (the teacher by default).
func (t *TutorTrainer) grader() LLMClient {
	if t.GraderClient != nil {
		return t.GraderClient
	}
	return t.Client
}

func (t *TutorTrainer) stopped() bool { return t.Stop != nil && t.Stop() }

func (t *TutorTrainer) outside(fn func() error) error {
	if t.External == nil {
		return fn()
	}
	return t.External(fn)
}

func (t *TutorTrainer) emit(record map[string]any) {
	t.History = append(t.History, record)
	if t.Progress != nil {
		t.Progress(record)
	}
}

// SetExercises is step 1: the teacher writes this round's sentence openings.
func (t *TutorTrainer) SetExercises(round int) ([]Exercise, error) {
	cfg := t.Config
	weak := []string{}
	if cfg.Adapt {
		weak = t.Weak
	}
	var exercises []Exercise
	err := t.outside(func() error {
		written, err := WriteExercises(t.Client, ExerciseRequest{
			Topic: cfg.Topic, Count: cfg.Exercises, Focus: cfg.Focus, Level: cfg.Level, Weak: weak,
			Words: cfg.Words, Brief: cfg.Brief, Model: cfg.TutorModel,
		})
		exercises = written
		return err
	})
	if err != nil {
		return nil, err
	}
	for i := range exercises {
		exercises[i].ID = fmt.Sprintf("r%de%d", round, i+1)
	}
	return exercises, nil
}

// Complete is step 2: the network continues the prefix (attempt 0 in the configured mode, later ones sampled).
func (t *TutorTrainer) Complete(exercise Exercise, attempt int) (*Lesson, error) {
	cfg := t.Config
	mode := cfg.Mode
	if attempt > 0 {
		mode = "sample"
	}
	started := time.Now()
	o := DefaultPredictOptions()
	o.Length, o.Mode, o.Temperature, o.MaxLength = cfg.Length, mode, cfg.Temperature, cfg.MaxLength
	o.ToEnd = cfg.ToEnd && mode != "sample"
	if cfg.Beam > 0 {
		o.Beam = cfg.Beam
	}
	if cfg.K > 0 {
		o.K = cfg.K
	}
	prediction, err := t.Model.Predict(exercise.Cue(), o)
	if err != nil {
		return nil, err
	}
	return &Lesson{
		Exercise: exercise, Attempt: attempt, Mode: mode, Continuation: prediction.Text,
		Sentence: exercise.Cue() + prediction.Text, Cost: prediction.Cost,
		Probability: prediction.Probability(), ReachedEnd: prediction.ReachedEnd,
		Seconds: time.Since(started).Seconds(),
	}, nil
}

// GradeLessons is step 3: the marker marks the completions (one call per Batch).
func (t *TutorTrainer) GradeLessons(lessons []*Lesson) error {
	cfg := t.Config
	grader := t.grader()
	return t.outside(func() error {
		return GradeCompletions(grader, lessons, GradeOptions{
			Topic: cfg.Topic, Threshold: cfg.Threshold, GrammarWeight: cfg.GrammarWeight,
			Model: cfg.ResolvedGraderModel(), Batch: cfg.Batch, GradedBy: ProviderOf(grader),
		})
	})
}

// WeightOf is the negative-phase weight of a failed sentence: MinWeight for a near miss, 1 for a hopeless one.
func (t *TutorTrainer) WeightOf(grade Grade) float64 {
	cfg := t.Config
	if grade.Score == nil || cfg.Threshold <= 0 {
		return 1.0
	}
	badness := math.Max(0, math.Min(1, (cfg.Threshold-*grade.Score)/cfg.Threshold))
	return cfg.MinWeight + (1.0-cfg.MinWeight)*badness
}

// RewardOf is the positive-phase weight of a sentence the network wrote: its mark, score / 10.
//
// The mark decides how much of the sentence the network keeps - a 10 out of 10
// is rewarded in full, a bare pass at a fraction of it - so passing is a
// rating, not a like.  What the teacher wrote always weighs TeacherWeight.
func (t *TutorTrainer) RewardOf(grade Grade) float64 {
	if grade.Score == nil {
		return 0
	}
	return math.Max(0, math.Min(1, *grade.Score/10.0))
}

// Graded is what the grades of a set of lessons mean for the network.
type Graded struct {
	Bad         []string
	BadWeights  []float64
	Good        []string
	GoodWeights []float64
	// Corrections are taught from their diff instead of as whole sentences (DiffCorrections).
	Corrections []TutorCorrection
}

// TutorCorrection is a sentence the network wrote, the sentence the teacher
// wrote instead, and how bad the mark was.
type TutorCorrection struct {
	Wrong  string  `json:"wrong"`
	Right  string  `json:"right"`
	Weight float64 `json:"weight"`
}

// Diffs: is a correction taught from its diff with the sentence it corrects?
func (t *TutorTrainer) Diffs() bool { return t.Config.DiffCorrections && t.Model != nil }

// CorrectionsOf lists the failed lessons a diff can teach.  A lesson that
// wrote nothing has no mistake to align, and one the teacher left uncorrected
// has nothing to align it against; both go the old way, through TextsOf.
func (t *TutorTrainer) CorrectionsOf(lessons []*Lesson) []TutorCorrection {
	if !t.Diffs() {
		return nil
	}
	out := []TutorCorrection{}
	seen := map[string]int{}
	for _, lesson := range lessons {
		grade := lesson.Grade
		if grade.Passed || strings.TrimSpace(lesson.Continuation) == "" || strings.TrimSpace(grade.Correction) == "" {
			continue
		}
		correction := TutorCorrection{
			Wrong: strings.TrimSpace(lesson.Sentence), Right: strings.TrimSpace(grade.Correction),
			Weight: t.WeightOf(grade),
		}
		key := correction.Wrong + "\x00" + correction.Right
		if at, ok := seen[key]; ok { // the same mistake twice (several attempts) keeps its worst mark
			if correction.Weight > out[at].Weight {
				out[at] = correction
			}
			continue
		}
		seen[key] = len(out)
		out = append(out, correction)
	}
	return out
}

// ChangesOf is what the teacher changed in one lesson, span by span (empty when nothing was corrected).
func (t *TutorTrainer) ChangesOf(lesson *Lesson) []Edit { return ChangesOfLesson(lesson) }

// TextsOf splits graded lessons into garbage (weighted by how bad the mark
// was) and good English (the sentences that passed, weighted by their mark,
// plus the teacher's corrections at full weight).  With DiffCorrections on, a
// failure the teacher corrected is left out of both lists and carried in
// Corrections instead, so the diff can punish the words that were actually
// wrong rather than the whole sentence.
func (t *TutorTrainer) TextsOf(lessons []*Lesson) Graded {
	out := Graded{Corrections: t.CorrectionsOf(lessons)}
	diffed := map[string]bool{}
	for _, c := range out.Corrections {
		diffed[c.Wrong+"\x00"+c.Right] = true
	}
	pairs := []weightedText{}
	for _, lesson := range lessons {
		grade := lesson.Grade
		if grade.Passed {
			if strings.TrimSpace(lesson.Sentence) != "" {
				pairs = append(pairs, weightedText{strings.TrimSpace(lesson.Sentence), t.RewardOf(grade)})
			}
			continue
		}
		if diffed[strings.TrimSpace(lesson.Sentence)+"\x00"+strings.TrimSpace(grade.Correction)] {
			if t.Config.TeachAnswer && strings.TrimSpace(lesson.Exercise.Answer) != "" {
				pairs = append(pairs, weightedText{strings.TrimSpace(lesson.Exercise.Answer), TeacherWeight})
			}
			continue // the diff teaches this one, sentence against correction
		}
		if strings.TrimSpace(lesson.Continuation) != "" { // nothing written is nothing to punish
			out.Bad = append(out.Bad, strings.TrimSpace(lesson.Sentence))
			out.BadWeights = append(out.BadWeights, t.WeightOf(grade))
		}
		if strings.TrimSpace(grade.Correction) != "" {
			pairs = append(pairs, weightedText{strings.TrimSpace(grade.Correction), TeacherWeight})
		}
		if t.Config.TeachAnswer && strings.TrimSpace(lesson.Exercise.Answer) != "" {
			pairs = append(pairs, weightedText{strings.TrimSpace(lesson.Exercise.Answer), TeacherWeight})
		}
	}
	out.Good, out.GoodWeights = mergeWeighted(pairs)
	return out
}

type weightedText struct {
	text   string
	weight float64
}

// mergeWeighted keeps first-seen order; a text offered twice keeps its largest weight.
func mergeWeighted(pairs []weightedText) ([]string, []float64) {
	texts := []string{}
	weights := []float64{}
	index := map[string]int{}
	for _, pair := range pairs {
		text := strings.TrimSpace(pair.text)
		if text == "" {
			continue
		}
		if at, ok := index[text]; ok {
			if pair.weight > weights[at] {
				weights[at] = pair.weight
			}
			continue
		}
		index[text] = len(texts)
		texts = append(texts, text)
		weights = append(weights, pair.weight)
	}
	return texts, weights
}

// Learn applies one set of grades: penalise the garbage by how bad it was, count and reward the good English by how good.
func (t *TutorTrainer) Learn(graded Graded) (map[string]any, error) {
	cfg := t.Config
	good, goodWeights := graded.Good, graded.GoodWeights
	if cfg.Replay {
		extra := []string{}
		known := map[string]bool{}
		for _, text := range good {
			known[text] = true
		}
		for _, text := range t.replay {
			if !known[text] {
				extra = append(extra, text)
			}
		}
		if cfg.ReplayLimit > 0 && len(extra) > cfg.ReplayLimit {
			extra = extra[len(extra)-cfg.ReplayLimit:]
		}
		for _, text := range extra {
			good = append(good, text)
			goodWeights = append(goodWeights, TeacherWeight)
		}
	}
	result := map[string]any{
		"bad": len(graded.Bad), "good": len(good), "action": nil, "neg_loss": nil, "pos_loss": nil,
		"mean_weight": mean(graded.BadWeights), "mean_reward": mean(goodWeights),
		"corrections": 0, "edits": 0, "penalised": 0, "rewarded": 0,
	}
	opts := TrainOptions{Epochs: cfg.NegEpochs, AutoCompress: true, Stop: t.Stop}
	strength := cfg.Strength
	if strength <= 0 {
		strength = 1.0
	}
	actions := []string{}
	for _, correction := range graded.Corrections {
		if t.stopped() {
			break
		}
		moved, err := t.Model.Correct(correction.Wrong, correction.Right, CorrectOptions{
			Strength: strength, Weight: correction.Weight, Reward: TeacherWeight, Keep: cfg.KeepWeight,
		})
		if err != nil {
			return result, err
		}
		result["corrections"] = result["corrections"].(int) + 1
		result["edits"] = result["edits"].(int) + moved.Edits
		result["penalised"] = result["penalised"].(int) + moved.Penalised
		result["rewarded"] = result["rewarded"].(int) + moved.Rewarded
		result["pos_loss"] = moved.Loss
		if !containsString(t.replay, correction.Right) {
			t.replay = append(t.replay, correction.Right)
		}
	}
	if result["corrections"].(int) > 0 {
		actions = append(actions, "correct")
	}
	if len(graded.Bad) == 0 && len(good) == 0 {
		result["action"] = joinActions(actions)
		t.trimReplay()
		return result, nil
	}
	switch {
	case len(graded.Bad) > 0 && len(good) > 0:
		res, err := t.Model.TwoNRLWeighted(graded.Bad, graded.BadWeights, good, goodWeights, TwoNRLOptions{
			NegEpochs: cfg.NegEpochs, PosEpochs: cfg.PosEpochs, Strength: strength, Stop: t.Stop,
		})
		if err != nil {
			return result, err
		}
		actions = append(actions, "2nrl")
		result["neg_loss"] = lastLoss(res.Negative)
		result["pos_loss"] = lastLoss(res.Positive)
	case len(good) > 0:
		opts.Epochs, opts.Phase = cfg.PosEpochs, "positive"
		records, err := t.Model.RewardWeighted(good, goodWeights, opts, strength)
		if err != nil {
			return result, err
		}
		actions = append(actions, "reward")
		result["pos_loss"] = lastLoss(records)
	default:
		opts.Epochs, opts.Phase = cfg.NegEpochs, "negative"
		records, err := t.Model.PunishWeighted(graded.Bad, graded.BadWeights, opts, strength)
		if err != nil {
			return result, err
		}
		actions = append(actions, "punish")
		result["neg_loss"] = lastLoss(records)
	}
	result["action"] = joinActions(actions)
	for _, text := range graded.Good {
		if !containsString(t.replay, text) {
			t.replay = append(t.replay, text)
		}
	}
	t.trimReplay()
	return result, nil
}

// trimReplay keeps the replay buffer to ReplayLimit texts (0 = no limit).
func (t *TutorTrainer) trimReplay() {
	if limit := t.Config.ReplayLimit; limit > 0 && len(t.replay) > limit {
		t.replay = t.replay[len(t.replay)-limit:]
	}
}

// joinActions names what a set of grades did, in order and without repeats ("correct+2nrl").
func joinActions(actions []string) any {
	out := []string{}
	for _, action := range actions {
		if action != "" && !containsString(out, action) {
			out = append(out, action)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return strings.Join(out, "+")
}

func containsString(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func lastLoss(records []map[string]any) any {
	if len(records) == 0 {
		return nil
	}
	if loss, ok := records[len(records)-1]["loss"]; ok {
		return loss
	}
	return nil
}

func (t *TutorTrainer) emitLesson(round int, lesson *Lesson) {
	grade := lesson.Grade
	t.emit(map[string]any{
		"kind": "lesson", "round": round, "exercise": lesson.Exercise.ID, "prefix": lesson.Exercise.Prefix,
		"focus": lesson.Exercise.Focus, "attempt": lesson.Attempt + 1, "mode": lesson.Mode,
		"continuation": clipText(lesson.Continuation, 400), "sentence": clipText(lesson.Sentence, 400),
		"score": grade.Score, "grammar": grade.Grammar, "spelling": grade.Spelling, "fluency": grade.Fluency,
		"passed": grade.Passed, "error": grade.Error, "correction": clipText(grade.Correction, 400),
		"comment": grade.Comment, "graded_by": grade.GradedBy, "probability": lesson.Probability,
		"seconds": lesson.Seconds, "changes": t.ChangesOf(lesson),
	})
}

// RunRound runs one round: exercises, completions, grades and the 2NRL they lead to.
func (t *TutorTrainer) RunRound(round int) (map[string]any, []*Lesson, error) {
	cfg := t.Config
	started := time.Now()
	exercises, err := t.SetExercises(round)
	if err != nil {
		return nil, nil, err
	}
	lessons := []*Lesson{}
	for _, exercise := range exercises {
		if t.stopped() {
			break
		}
		for attempt := 0; attempt < cfg.Attempts; attempt++ {
			lesson, err := t.Complete(exercise, attempt)
			if err != nil {
				return nil, nil, err
			}
			lessons = append(lessons, lesson)
		}
	}
	if len(lessons) > 0 && !t.stopped() {
		if err := t.GradeLessons(lessons); err != nil {
			return nil, nil, err
		}
	}
	for _, lesson := range lessons {
		t.emitLesson(round, lesson)
	}
	t.Lessons = append(t.Lessons, lessons...)
	card := ReportCard(lessons)
	drills := []string{}
	if cfg.Drills > 0 && len(lessons) > 0 && !t.stopped() {
		err := t.outside(func() error {
			written, err := DrillSentences(t.Client, cfg.Topic, cfg.Drills, card["weakest"].([]string), cfg.TutorModel)
			drills = written
			return err
		})
		if err != nil { // the lesson stands without its drill sentences
			t.emit(map[string]any{"kind": "note", "round": round, "message": "no drill sentences: " + err.Error()})
		}
	}
	learned := map[string]any{}
	if !t.stopped() {
		learned, err = t.learnLessons(lessons, drills)
		if err != nil {
			return nil, nil, err
		}
	}
	if cfg.Adapt {
		t.Weak = card["weakest"].([]string)
	}
	record := map[string]any{
		"kind": "round", "round": round, "topic": cfg.Topic, "focus": cfg.Focus,
		"exercises": len(exercises), "drills": len(drills), "seconds": time.Since(started).Seconds(),
	}
	for key, value := range card {
		record[key] = value
	}
	for key, value := range learned {
		record[key] = value
	}
	return record, lessons, nil
}

// learnLessons applies the round's grades: once for the whole round, or lesson
// by lesson (the drill sentences are taught once either way).
func (t *TutorTrainer) learnLessons(lessons []*Lesson, drills []string) (map[string]any, error) {
	cfg := t.Config
	withDrills := func(graded Graded) Graded {
		pairs := make([]weightedText, 0, len(graded.Good)+len(drills))
		for i, text := range graded.Good {
			pairs = append(pairs, weightedText{text, graded.GoodWeights[i]})
		}
		for _, text := range drills {
			pairs = append(pairs, weightedText{text, TeacherWeight})
		}
		graded.Good, graded.GoodWeights = mergeWeighted(pairs)
		return graded
	}
	if !cfg.Learn { // a dry run: what would have been learned, without touching the network
		graded := withDrills(t.TextsOf(lessons))
		edits := 0
		for _, c := range graded.Corrections {
			edits += len(DiffSummary(c.Wrong, c.Right, 0))
		}
		return map[string]any{
			"bad": len(graded.Bad), "good": len(graded.Good), "action": nil, "neg_loss": nil, "pos_loss": nil,
			"mean_weight": mean(graded.BadWeights), "mean_reward": mean(graded.GoodWeights),
			"corrections": len(graded.Corrections), "edits": edits, "penalised": 0, "rewarded": 0,
		}, nil
	}
	if cfg.TwoNRLPer != "lesson" {
		return t.Learn(withDrills(t.TextsOf(lessons)))
	}
	merged := map[string]any{
		"bad": 0, "good": 0, "action": nil, "neg_loss": nil, "pos_loss": nil, "mean_weight": nil, "mean_reward": nil,
		"corrections": 0, "edits": 0, "penalised": 0, "rewarded": 0,
	}
	actions := []string{}
	badSeen, goodSeen := []float64{}, []float64{}
	for _, lesson := range lessons {
		if t.stopped() {
			break
		}
		graded := withDrills(t.TextsOf([]*Lesson{lesson}))
		drills = nil // taught once
		outcome, err := t.Learn(graded)
		if err != nil {
			return merged, err
		}
		badSeen = append(badSeen, graded.BadWeights...)
		goodSeen = append(goodSeen, graded.GoodWeights...)
		if action, ok := outcome["action"].(string); ok {
			actions = append(actions, strings.Split(action, "+")...)
		}
		merged["bad"] = merged["bad"].(int) + outcome["bad"].(int)
		merged["good"] = merged["good"].(int) + outcome["good"].(int)
		for _, key := range []string{"corrections", "edits", "penalised", "rewarded"} {
			if value, ok := outcome[key].(int); ok {
				merged[key] = merged[key].(int) + value
			}
		}
		for _, key := range []string{"neg_loss", "pos_loss"} {
			if outcome[key] != nil {
				merged[key] = outcome[key]
			}
		}
	}
	merged["action"] = joinActions(actions)
	merged["mean_weight"] = mean(badSeen)
	merged["mean_reward"] = mean(goodSeen)
	return merged, nil
}

// Run works through every round and finishes with a report card over all of them.
func (t *TutorTrainer) Run() ([]map[string]any, error) {
	records := []map[string]any{}
	for round := 1; round <= t.Config.Rounds; round++ {
		if t.stopped() {
			break
		}
		record, _, err := t.RunRound(round)
		if err != nil {
			return records, err
		}
		t.emit(record)
		records = append(records, record)
	}
	summary := map[string]any{"kind": "report", "rounds": len(records), "topic": t.Config.Topic}
	for key, value := range ReportCard(t.Lessons) {
		summary[key] = value
	}
	t.emit(summary)
	rounds := len(records)
	records = append(records, summary)
	if t.Config.Plan > 0 && len(t.Lessons) > 0 && !t.stopped() {
		plan, err := t.PlanNext(summary, t.Config.Plan)
		if err != nil { // the lessons stand without a plan for the next ones
			t.emit(map[string]any{"kind": "note", "message": "no lesson plan: " + err.Error()})
			return records, nil
		}
		record := map[string]any{"kind": "plan", "rounds": rounds}
		for key, value := range plan.Map() {
			record[key] = value
		}
		t.emit(record)
		records = append(records, record)
	}
	return records, nil
}

// PlanNext is the next lessons, planned by the teacher from a report card - by
// default the run's own.  The weaknesses of the card become the syllabus, each
// lesson carrying the settings to run it with.
func (t *TutorTrainer) PlanNext(card map[string]any, count int) (LessonPlan, error) {
	cfg := t.Config
	if card == nil {
		card = ReportCard(t.Lessons)
	}
	if count < 1 {
		count = DefaultPlanLessons
	}
	var plan LessonPlan
	err := t.outside(func() error {
		var err error
		plan, err = PlanLessons(t.Client, card, PlanRequest{
			Topic: cfg.Topic, Level: cfg.Level, Words: cfg.Words, Threshold: cfg.Threshold, Count: count,
			Exercises: cfg.Exercises, Drills: cfg.Drills, Model: cfg.TutorModel,
		})
		return err
	})
	return plan, err
}
