package kit

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// The teaching loops of a model of sounds: every loop that shows an LLM what
// the model wrote shows it the words its sounds spell, while the model is
// rewarded, punished and blamed for the sounds it said.  Python's
// tests/test_teaching_in_words.py.

// soundOf finds what English never holds: a stressed vowel (AE1), a word
// boundary (#) or a syllable (K.AE1).
var soundOf = regexp.MustCompile(`\b[A-Z]{1,2}[012]\b|(?:^|\s)#(?:\s|$)|[A-Z]+\.[A-Z]`)

var teachingCorpus = []string{
	"the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "the dog ran to the park",
	"a bird sang in the tree",
}

// wordsTeacher answers every request from a script and records every prompt.
type wordsTeacher struct {
	prompts []string
	fix     func(text string) string
}

func newWordsTeacher() *wordsTeacher {
	return &wordsTeacher{fix: func(text string) string {
		if at := strings.LastIndex(text, " "); at > 0 {
			return text[:at] + " mat"
		}
		return text
	}}
}

func (s *wordsTeacher) Provider() string                  { return ProviderOllama }
func (s *wordsTeacher) BaseURL() string                   { return "http://scripted" }
func (s *wordsTeacher) ModelName() string                 { return "scripted" }
func (s *wordsTeacher) Available() bool                   { return true }
func (s *wordsTeacher) Models() ([]map[string]any, error) { return nil, nil }

// indexedLines are a prompt's "[i] text" lines.
func indexedLines(prompt string) map[int]string {
	out := map[int]string{}
	for _, line := range strings.Split(prompt, "\n") {
		if strings.HasPrefix(line, "[") && strings.Contains(line, "]") {
			end := strings.Index(line, "]")
			if i, err := strconv.Atoi(line[1:end]); err == nil {
				out[i] = line[end+2:]
			}
		}
	}
	return out
}

func (s *wordsTeacher) Generate(prompt string, o LLMOptions) (string, error) {
	s.prompts = append(s.prompts, prompt)
	answer := func(v any) (string, error) {
		data, err := json.Marshal(v)
		return string(data), err
	}
	items := func(each func(i int, text string) map[string]any) []map[string]any {
		out := []map[string]any{}
		for i, text := range indexedLines(prompt) {
			out = append(out, each(i, text))
		}
		return out
	}
	switch {
	case strings.Contains(prompt, "exercises now"):
		return answer(map[string]any{"exercises": []map[string]any{
			{"prefix": "The cat", "answer": "The cat sat on the mat.", "focus": "past tense"},
			{"prefix": "The dog", "answer": "The dog sat on the mat.", "focus": "past tense"},
		}})
	case strings.Contains(prompt, "Mark these"):
		return answer(map[string]any{"grades": items(func(i int, _ string) map[string]any {
			return map[string]any{"index": i, "grammar": 3, "spelling": 5, "fluency": 3, "error": "tense",
				"correction": "The cat sits on the mat.", "comment": "Use the present tense."}
		})})
	case strings.Contains(prompt, "Explain these"):
		return answer(map[string]any{"mistakes": items(func(i int, _ string) map[string]any {
			return map[string]any{"index": i, "why": "the tense is wrong",
				"again": []map[string]any{{"wrong": "the dog sit", "right": "the dog sits"}}}
		})})
	case strings.Contains(prompt, "Review these"):
		return answer(map[string]any{"reviews": items(func(i int, _ string) map[string]any {
			return map[string]any{"index": i, "rating": 2, "critique": "it repeats itself"}
		})})
	case strings.Contains(prompt, "Correct these"):
		return answer(map[string]any{"corrections": items(func(i int, text string) map[string]any {
			return map[string]any{"index": i, "correction": s.fix(text), "reason": "vocabulary", "note": "the last word"}
		})})
	case strings.Contains(prompt, "The conversation:"):
		return answer(map[string]any{"reviews": items(func(i int, _ string) map[string]any {
			return map[string]any{"index": i, "rating": 2, "critique": "it does not follow on"}
		}), "overall": map[string]any{"rating": 3, "critique": "short"}})
	}
	return "the dog sat on the mat", nil // a line of a conversation: the partner's, or Ollama's
}

// wordsOnly fails the test when any prompt the LLM was sent holds a sound.
func (s *wordsTeacher) wordsOnly(t *testing.T) {
	t.Helper()
	if len(s.prompts) == 0 {
		t.Fatal("the LLM was never asked")
	}
	for _, prompt := range s.prompts {
		if found := soundOf.FindString(prompt); found != "" {
			t.Fatalf("the LLM was shown the sound %q:\n%s", found, prompt)
		}
	}
}

func soundsModel(t *testing.T) *radixnet.Model {
	t.Helper()
	opts := radixnet.DefaultGraphOptions()
	opts.Encoding = mustEncoding(t, "syllable:2:1")
	m, err := radixnet.NewModel(1, opts)
	if err != nil {
		t.Fatal(err)
	}
	m.Workers, m.G.Workers, m.Exact = 1, 1, true
	train := radixnet.DefaultTrainOptions()
	train.Epochs = 2
	if _, err := m.Train(teachingCorpus, train); err != nil {
		t.Fatal(err)
	}
	return m
}

func soundsNegative(t *testing.T, m *radixnet.Model) *radixnet.Model {
	t.Helper()
	o := radixnet.DefaultNegativeOptions()
	o.Encoding = m.Encoding()
	negative, err := radixnet.NewNegativeModel(1, o)
	if err != nil {
		t.Fatal(err)
	}
	return negative
}

func TestTutorMarksWordsAndTeachesSounds(t *testing.T) {
	m := soundsModel(t)
	teacher := newWordsTeacher()
	cfg := DefaultTutorConfig()
	cfg.Topic, cfg.Exercises, cfg.Attempts, cfg.Variants = "animals", 2, 1, 1
	trainer, err := NewTutorTrainer(m, teacher, cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainer.Negative = soundsNegative(t, m)
	record, lessons, err := trainer.RunRound(1)
	if err != nil {
		t.Fatal(err)
	}
	teacher.wordsOnly(t)
	if len(lessons) != 2 {
		t.Fatalf("%d lessons", len(lessons))
	}
	for _, lesson := range lessons {
		if !strings.HasPrefix(lesson.Sentence, lesson.Exercise.Cue()) || soundOf.MatchString(lesson.Sentence) {
			t.Errorf("the teacher read %q", lesson.Sentence)
		}
		if !strings.HasPrefix(lesson.Said, "DH.AH0 # ") || lesson.Own() != lesson.Said {
			t.Errorf("the model said %q", lesson.Said)
		}
		if lesson.Prefix() != lesson.Exercise.Cue() {
			t.Errorf("prefix %q", lesson.Prefix())
		}
	}
	corrections := trainer.CorrectionsOf(lessons)
	if len(corrections) != 2 || corrections[0].Wrong != lessons[0].Said || corrections[0].Right != "The cat sits on the mat." {
		t.Fatalf("corrections %+v", corrections)
	}
	faults, cleared := FaultsFromLessons(lessons, 6, "tutor")
	for _, fault := range faults {
		if fault.Source == "tutor" && (fault.Text != lessons[0].Said && fault.Text != lessons[1].Said) {
			t.Errorf("blamed %q, not what the model said", fault.Text)
		}
	}
	if !contains(cleared, "The cat sits on the mat.") {
		t.Errorf("cleared %v", cleared)
	}
	if record["corrections"].(int) == 0 || record["edits"].(int) == 0 || record["negative_blamed"].(int) == 0 {
		t.Errorf("record %v", record)
	}
	// a lesson of letters is marked as it writes, and carries nothing more
	data, _ := json.Marshal(&Lesson{Sentence: "the cat", Continuation: "cat"})
	if strings.Contains(string(data), `"said"`) {
		t.Errorf("a lesson of letters wrote %s", data)
	}
}

func TestReviewerAndEditorReadWords(t *testing.T) {
	m := soundsModel(t)
	enc := m.Encoding()
	results, err := m.Generate(radixnet.GenerateOptions{Count: 3, Mode: "beam", Prefix: "the", MaxLength: 20,
		Traversal: "reward", PenaltyScale: 1, MeritScale: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	sounds := []string{}
	for _, r := range results {
		sounds = append(sounds, r.Text)
	}
	teacher := newWordsTeacher()
	reviews, err := ReviewTexts(teacher, sounds, "", "", 6, DefaultReviewBatch, enc)
	if err != nil {
		t.Fatal(err)
	}
	for i, review := range reviews {
		if review.Text != sounds[i] || review.Spelled != enc.Spell(sounds[i]) {
			t.Errorf("review %+v", review)
		}
	}
	corrections, err := CorrectTexts(teacher, sounds, "", "", DefaultCorrectionBatch, enc)
	if err != nil {
		t.Fatal(err)
	}
	teacher.wordsOnly(t)
	for i, entry := range corrections {
		words := enc.Spell(sounds[i])
		if entry.Text != sounds[i] || entry.Spelled != words || *entry.Correction != teacher.fix(words) {
			t.Errorf("correction %+v", entry)
		}
		for _, change := range entry.Changes {
			if soundOf.MatchString(change.Wrong + " " + change.Right) {
				t.Errorf("a change in sounds: %+v", change)
			}
		}
	}
	faults, _ := FaultsFromCorrections(corrections, 1, "correction")
	if len(faults) == 0 || !contains(sounds, faults[0].Text) {
		t.Fatalf("faults %+v", faults)
	}
	report, err := TeachCorrections(soundsNegative(t, m), corrections, 1, true, "correction", TeachOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Edges == 0 {
		t.Errorf("the correction, read as sounds, did not line up with the sounds said: %+v", report)
	}
	// the zero encoding reads every text as it is
	if plain, _ := ReviewTexts(newWordsTeacher(), []string{"the cat sat"}, "", "", 6, 20, radixnet.Encoding{}); plain[0].Spelled != "" {
		t.Errorf("a text of letters spelled %q", plain[0].Spelled)
	}
}

func TestChatPartnerAndJudgeHearWords(t *testing.T) {
	m := soundsModel(t)
	teacher := newWordsTeacher()
	cfg := DefaultChatConfig()
	cfg.Turns, cfg.Opening, cfg.Guard = 2, "the cat", false
	chat, err := NewChat(m, soundsNegative(t, m), teacher, cfg)
	if err != nil {
		t.Fatal(err)
	}
	record, err := chat.RunConversation(nil)
	if err != nil {
		t.Fatal(err)
	}
	teacher.wordsOnly(t)
	said := 0
	for _, line := range record["transcript"].([]map[string]any) {
		if line["speaker"] == ChatSpeakers[1] {
			said++
			if line["spelled"] != m.Encoding().Spell(line["text"].(string)) {
				t.Errorf("line %v", line)
			}
		} else if _, ok := line["spelled"]; ok {
			t.Errorf("the partner's line %v", line)
		}
	}
	if said == 0 {
		t.Fatal("the model said nothing")
	}
}

func TestVoiceOllamaHearsWords(t *testing.T) {
	teacher := newWordsTeacher()
	o := voiceOptions("how are you", "ollama")
	o.Train, o.Speak = false, false
	out, err := VoiceTurn(nil, mustEncoding(t, "syllable:2:1"), o,
		[]Line{{"You", "hello"}, {"Model", "DH.AH0 # K.AE1.T # S.AE1.T"}}, VoiceParts{}, teacher, nil)
	if err != nil {
		t.Fatal(err)
	}
	teacher.wordsOnly(t)
	if !strings.Contains(teacher.prompts[len(teacher.prompts)-1], "Model: the cat sat") || out.Document["by"] != "ollama" {
		t.Fatalf("prompt %q, by %v", teacher.prompts[len(teacher.prompts)-1], out.Document["by"])
	}
}
