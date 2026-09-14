package radixnet

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Ollama integration beyond the chat client: prompt-driven training corpora and
// the adversarial review.  Both are the Python `radixnet/ollama.py` functions,
// and both work with any LLMClient (they only ever call Generate), so ChatGPT
// reviews as happily as a local model does.

// CorpusStyles are the two kinds of corpus the LLM is asked for.
var CorpusStyles = []string{"good", "garbage"}

const corpusSystem = "You produce training data for a small character-level language model. Answer with exactly %d lines " +
	"and nothing else: one short, self-contained sentence per line, plain text, no numbering, no bullets, " +
	"no quotes, no blank lines, no headings and no commentary. "

const styleGood = "Every line must be correct, natural and factual, in the style and about the topic requested."

const styleGarbage = "Every line must be deliberately WRONG in a way a careful reader would reject: false facts, scrambled " +
	"word order, broken grammar, nonsense words, contradictions. Keep the requested topic and vocabulary so " +
	"the errors are the only difference from good text."

// CorpusFromPrompt asks the LLM for `lines` lines of text about prompt; style is
// "good" or "garbage" - which is exactly the pair of inputs 2NRL wants.
func CorpusFromPrompt(client LLMClient, prompt string, lines int, style, model string) ([]string, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, fmt.Errorf("prompt must not be empty")
	}
	if lines < 1 {
		return nil, fmt.Errorf("lines must be >= 1")
	}
	style = strings.ToLower(strings.TrimSpace(style))
	if style == "" {
		style = "good"
	}
	if style != "good" && style != "garbage" {
		return nil, fmt.Errorf("style must be one of %s (got %q)", strings.Join(CorpusStyles, ", "), style)
	}
	system := fmt.Sprintf(corpusSystem, lines)
	temperature := 0.9
	if style == "garbage" {
		system += styleGarbage
		temperature = 1.1
	} else {
		system += styleGood
	}
	user := fmt.Sprintf("Topic / instructions: %s\n\nWrite the %d lines now.", strings.TrimSpace(prompt), lines)
	raw, err := client.Generate(user, LLMOptions{System: system, Model: model, Temperature: temperature})
	if err != nil {
		return nil, err
	}
	return ParseLines(raw, lines), nil
}

// -- the adversarial review ---------------------------------------------------

const reviewSystem = "You are an adversarial reviewer of text produced by a small experimental language model. Assume every " +
	"text is flawed and hunt for the flaws: gibberish, broken grammar, wrong word order, truncated or repeated " +
	"fragments, contradictions, false statements, incoherence. Rate each text from 0 to 10: 10 = " +
	"indistinguishable from correct, fluent, factual human writing; 6 = acceptable with minor flaws; 0 = pure " +
	"gibberish. The verdict is \"pass\" for a rating of %s or more, otherwise \"fail\". Reply with " +
	"JSON only, no prose, exactly of the form {\"reviews\": [{\"index\": <int>, \"rating\": <number>, " +
	"\"verdict\": \"pass\" or \"fail\", \"critique\": \"<one sentence naming the worst flaw, or 'no flaw " +
	"found'>\"}, ...]} with one entry per text, in the given order and with the given index."

// Review is one text as the reviewer marked it.  Rating is nil with verdict
// "unrated" when the answer could not be understood for that text.
type Review struct {
	Index    int      `json:"index"`
	Text     string   `json:"text"`
	Rating   *float64 `json:"rating"`
	Verdict  string   `json:"verdict"`
	Critique string   `json:"critique"`
	// Said is the line this text was a reply to, when it was one (chat.go).
	Said string `json:"said,omitempty"`
}

// DefaultReviewBatch is how many texts go into one review call.
const DefaultReviewBatch = 20

// parseReviews reads whatever shape the LLM drifted into: {"reviews": [...]},
// {"results"/"items": [...]}, a bare list, or one object when a single text was
// asked about; score stands in for rating and reason / comment for critique.
func parseReviews(raw string, count int) map[int]Review {
	parsed := map[int]Review{}
	data := loadsLenient(raw)
	var items []any
	switch value := data.(type) {
	case map[string]any:
		for _, key := range []string{"reviews", "results", "items"} {
			if list, ok := value[key].([]any); ok {
				items = list
				break
			}
		}
		if items == nil {
			if _, ok := value["rating"]; ok {
				items = []any{value}
			} else if _, ok := value["score"]; ok {
				items = []any{value}
			}
		}
	case []any:
		items = value
	}
	for position, item := range items {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		index := position
		if given, ok := numberOf(entry["index"]); ok {
			index = int(given)
		}
		rating, ok := numberOf(entry["rating"])
		if !ok {
			if rating, ok = numberOf(entry["score"]); !ok {
				continue
			}
		}
		if math.IsNaN(rating) {
			continue
		}
		rating = math.Max(0, math.Min(10, rating))
		critique := ""
		for _, key := range []string{"critique", "reason", "comment"} {
			if text, ok := entry[key].(string); ok && strings.TrimSpace(text) != "" {
				critique = strings.TrimSpace(text)
				break
			}
		}
		if critique == "" {
			critique = "no critique given"
		}
		if _, seen := parsed[index]; index >= 0 && index < count && !seen {
			parsed[index] = Review{Rating: &rating, Critique: critique}
		}
	}
	return parsed
}

// numberOf reads a JSON number (or a numeric string) out of an any.
func numberOf(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case string:
		var out float64
		if _, err := fmt.Sscanf(strings.TrimSpace(v), "%g", &out); err == nil {
			return out, true
		}
	}
	return 0, false
}

// ReviewTexts asks the LLM to mark every text, in input order.  A blank text is
// failed without asking; a text the answer said nothing usable about comes back
// unrated, which still counts as a failure but records that nobody said why.
func ReviewTexts(client LLMClient, texts []string, context, model string, threshold float64, batch int) ([]Review, error) {
	if batch < 1 {
		return nil, fmt.Errorf("batch must be >= 1")
	}
	system := fmt.Sprintf(reviewSystem, strconv.FormatFloat(threshold, 'g', -1, 64))
	out := make([]Review, 0, len(texts))
	for start := 0; start < len(texts); start += batch {
		end := start + batch
		if end > len(texts) {
			end = len(texts)
		}
		chunk := texts[start:end]
		var numbered []string
		for i, text := range chunk {
			if strings.TrimSpace(text) != "" {
				numbered = append(numbered, fmt.Sprintf("[%d] %s", i, text))
			}
		}
		parsed := map[int]Review{}
		if len(numbered) > 0 {
			user := ""
			if strings.TrimSpace(context) != "" {
				user = "Context: " + strings.TrimSpace(context) + "\n\n"
			}
			user += fmt.Sprintf("Review these %d texts:\n%s\n\nReturn the JSON now.", len(numbered), strings.Join(numbered, "\n"))
			raw, err := client.Generate(user, LLMOptions{System: system, Model: model, JSON: true, Temperature: 0.2})
			if err != nil {
				return nil, err
			}
			parsed = parseReviews(raw, len(chunk))
		}
		for i, text := range chunk {
			entry := Review{Index: start + i, Text: text}
			switch found, ok := parsed[i]; {
			case strings.TrimSpace(text) == "":
				zero := 0.0
				entry.Rating, entry.Verdict, entry.Critique = &zero, "fail", "empty output"
			case ok:
				entry.Rating, entry.Critique = found.Rating, found.Critique
				entry.Verdict = "fail"
				if *found.Rating >= threshold {
					entry.Verdict = "pass"
				}
			default:
				entry.Verdict, entry.Critique = "unrated", "no review returned"
			}
			out = append(out, entry)
		}
	}
	return out, nil
}

// SampleTexts draws count stochastic texts from the model (continuations of
// prefix when one is given).
func SampleTexts(model *Model, count int, prefix string, maxLength int, temperature float64, seed *int64) ([]string, error) {
	if count < 1 {
		return nil, fmt.Errorf("count must be >= 1")
	}
	if prefix != "" {
		out := make([]string, 0, count)
		for i := 0; i < count; i++ {
			result, err := model.Predict(prefix, PredictOptions{
				Length: maxLength, Mode: "sample", K: 1, Temperature: temperature, MaxLength: maxLength,
			})
			if err != nil {
				return nil, err
			}
			out = append(out, result.FullText)
		}
		return out, nil
	}
	results, err := model.Generate(GenerateOptions{
		MaxLength: maxLength, Mode: "sample", Temperature: temperature, Count: count, Seed: seed,
	})
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(results))
	for _, result := range results {
		out = append(out, result.Text)
	}
	return out, nil
}

// ReviewResult is one adversarial review of a set of texts, split into the ones
// that passed and the ones that did not (unrated texts count as failures).
type ReviewResult struct {
	Source     string   `json:"source"`
	Model      string   `json:"model"`
	Threshold  float64  `json:"threshold"`
	Texts      []string `json:"texts"`
	Reviews    []Review `json:"reviews"`
	MeanRating *float64 `json:"mean_rating"`
	PassRate   *float64 `json:"pass_rate"`
	Good       []string `json:"good"`
	Bad        []string `json:"bad"`
	// Overall is the judge's verdict on a conversation as a whole (chat.go); nil elsewhere.
	Overall *Overall `json:"overall,omitempty"`
}

// Overall is a judge's verdict on a whole conversation.
type Overall struct {
	Rating   *float64 `json:"rating"`
	Critique string   `json:"critique"`
}

// AdversarialReviewOptions are the knobs of one review round.
type AdversarialReviewOptions struct {
	Count       int
	Prefix      string
	MaxLength   int
	Temperature float64
	Texts       []string // review these instead of sampling from the model
	Threshold   float64
	Context     string
	Model       string // the reviewer's model, "" for the client's own
	Seed        *int64
	Batch       int
}

// AdversarialReview lets the LLM judge the network's own output (or the given
// texts) and splits it into good / bad sets, ready for a 2NRL pass or for the
// negative network to be blamed with.
func AdversarialReview(model *Model, client LLMClient, o AdversarialReviewOptions) (*ReviewResult, error) {
	samples, source := o.Texts, "given"
	if o.Texts == nil {
		if model == nil {
			return nil, fmt.Errorf("either a model to sample from or texts to review is required")
		}
		count := o.Count
		if count < 1 {
			count = 8
		}
		drawn, err := SampleTexts(model, count, o.Prefix, o.MaxLength, o.Temperature, o.Seed)
		if err != nil {
			return nil, err
		}
		samples, source = drawn, "model"
	}
	batch := o.Batch
	if batch < 1 {
		batch = DefaultReviewBatch
	}
	reviews, err := ReviewTexts(client, samples, o.Context, o.Model, o.Threshold, batch)
	if err != nil {
		return nil, err
	}
	reviewer := o.Model
	if reviewer == "" {
		reviewer = client.ModelName()
	}
	return SummariseReviews(source, reviewer, o.Threshold, samples, reviews), nil
}

// SummariseReviews splits a marked set into the texts that passed and the ones
// that did not (unrated texts count as failures) and averages the marks.
//
// It is separate from AdversarialReview because a caller holding a model lock
// has to sample and review in two steps: the sampling walks the graph and must
// hold the lock, the reviewing is a network call and must not.
func SummariseReviews(source, model string, threshold float64, texts []string, reviews []Review) *ReviewResult {
	result := &ReviewResult{
		Source: source, Model: model, Threshold: threshold, Texts: texts, Reviews: reviews,
		Good: []string{}, Bad: []string{},
	}
	sum, rated := 0.0, 0
	for _, review := range reviews {
		if review.Rating != nil {
			sum += *review.Rating
			rated++
		}
		if review.Verdict == "pass" {
			result.Good = append(result.Good, review.Text)
		} else {
			result.Bad = append(result.Bad, review.Text)
		}
	}
	if rated > 0 {
		mean := sum / float64(rated)
		result.MeanRating = &mean
	}
	if len(reviews) > 0 {
		rate := float64(len(result.Good)) / float64(len(reviews))
		result.PassRate = &rate
	}
	return result
}

// ReasonOrder sorts a reason histogram the way the tables print it: heaviest
// first, ties alphabetical.
func ReasonOrder(counts map[string]int) []string {
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
	return names
}

// -- conversing with the network, and marking the conversation ----------------

const chatSystem = "You are having a short, ordinary conversation with a very small character-level neural " +
	"network that is learning to talk. It answers by continuing the last few words you wrote, so every line you " +
	"write must be short, plain and concrete, and must end on words that are easy to carry on from. Never mention " +
	"that it is a model, never explain yourself, never ask more than one thing at a time, and never write more " +
	"than one sentence. Reply with the next thing you would say and nothing else."

const conversationSystem = "You are marking a conversation between a person and a very small character-level " +
	"neural network that is learning to talk. Mark each of the network's lines out of 10 for one thing only: is it " +
	"a real reply to the line before it - does it follow on, is it about the same thing, is it a sentence at all. " +
	"Ignore style, length and ambition: a short plain line that follows on is a 10. A line that merely repeats " +
	"what was just said, that is gibberish, or that answers something nobody asked is 0 to 3. %s out of 10 is a " +
	"pass. Reply with JSON only, of the form {\"reviews\": [{\"index\": <n>, \"rating\": <0-10>, " +
	"\"critique\": \"<one sentence>\"}, ...], \"overall\": {\"rating\": <0-10>, \"critique\": " +
	"\"<one sentence about the conversation as a whole>\"}}."

// ChatSpeakers are who is who in a transcript: the LLM, then the network.
var ChatSpeakers = [2]string{"Partner", "Model"}

// Line is one line of a conversation.
type Line struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}

// Exchange is a line said to the network and what it replied.
type Exchange struct {
	Said  string
	Reply string
}

// ChatLineOptions are what the partner is told about the conversation it is holding.
type ChatLineOptions struct {
	Topic       string
	Persona     string
	Model       string
	Temperature float64
}

// ChatLine is the next line of the LLM's side of a conversation with the
// network.  An empty transcript opens the conversation.
//
// The system prompt asks for short, plain lines ending on words that are easy
// to carry on from, because that is what a character-level model can actually
// reply to.
func ChatLine(client LLMClient, transcript []Line, o ChatLineOptions) (string, error) {
	system := chatSystem
	if strings.TrimSpace(o.Topic) != "" {
		system += " The conversation is about " + strings.TrimSpace(o.Topic) + "."
	}
	if strings.TrimSpace(o.Persona) != "" {
		system += " You are " + strings.TrimSpace(o.Persona) + "."
	}
	said := []string{}
	for _, line := range transcript {
		if strings.TrimSpace(line.Text) != "" {
			said = append(said, line.Speaker+": "+line.Text)
		}
	}
	user := "Open the conversation with one short, plain line."
	if strings.TrimSpace(o.Topic) != "" {
		user = "Open the conversation about " + strings.TrimSpace(o.Topic) + " with one short, plain line."
	}
	if len(said) > 0 {
		user = "The conversation so far:\n" + strings.Join(said, "\n") + "\n\nWrite your next line."
	}
	temperature := o.Temperature
	if temperature <= 0 {
		temperature = 0.8
	}
	raw, err := client.Generate(user, LLMOptions{System: system, Model: o.Model, Temperature: temperature})
	if err != nil {
		return "", err
	}
	return firstLine(raw), nil
}

// firstLine is one line out of an LLM answer that may have written several (or
// quoted itself).
func firstLine(raw string) string {
	for _, line := range strings.Split(raw, "\n") {
		text := collapse(line)
		if text == "" {
			continue
		}
		for _, speaker := range []string{"Partner:", "Model:", "You:", "Me:"} {
			if len(text) >= len(speaker) && strings.EqualFold(text[:len(speaker)], speaker) {
				text = strings.TrimSpace(text[len(speaker):])
			}
		}
		text = strings.Trim(text, "\"\u201c\u201d")
		if text != "" {
			return text
		}
	}
	return ""
}

// ReviewConversation marks every reply the network gave in a conversation, and
// the conversation as a whole.
//
// The result is the SummariseReviews shape - so TeachReviews takes it as it is
// - plus Overall, the judge's verdict on the conversation itself (nil when it
// did not give one).
func ReviewConversation(
	client LLMClient, exchanges []Exchange, topic, model string, threshold float64,
) (*ReviewResult, error) {
	replies := make([]string, 0, len(exchanges))
	blocks := []string{}
	for i, exchange := range exchanges {
		replies = append(replies, exchange.Reply)
		if strings.TrimSpace(exchange.Reply) != "" {
			blocks = append(blocks, fmt.Sprintf("[%d] %s: %s\n    %s: %s", i, ChatSpeakers[0], exchange.Said,
				ChatSpeakers[1], exchange.Reply))
		}
	}
	parsed := map[int]Review{}
	var overall *Overall
	if len(blocks) > 0 {
		user := ""
		if strings.TrimSpace(topic) != "" {
			user = "Topic: " + strings.TrimSpace(topic) + "\n\n"
		}
		user += "The conversation:\n" + strings.Join(blocks, "\n\n") + "\n\nReturn the JSON now."
		system := fmt.Sprintf(conversationSystem, strconv.FormatFloat(threshold, 'g', -1, 64))
		raw, err := client.Generate(user, LLMOptions{System: system, Model: model, JSON: true, Temperature: 0.2})
		if err != nil {
			return nil, err
		}
		parsed = parseReviews(raw, len(exchanges))
		overall = parseOverall(raw)
	}
	reviews := make([]Review, 0, len(exchanges))
	for i, exchange := range exchanges {
		entry := Review{Index: i, Text: exchange.Reply, Said: exchange.Said}
		switch found, ok := parsed[i]; {
		case strings.TrimSpace(exchange.Reply) == "":
			zero := 0.0
			entry.Rating, entry.Verdict, entry.Critique = &zero, "fail", "it said nothing"
		case ok:
			entry.Rating, entry.Critique = found.Rating, found.Critique
			entry.Verdict = "fail"
			if found.Rating != nil && *found.Rating >= threshold {
				entry.Verdict = "pass"
			}
		default:
			entry.Verdict, entry.Critique = "unrated", "no review returned"
		}
		reviews = append(reviews, entry)
	}
	judge := model
	if judge == "" {
		judge = client.ModelName()
	}
	result := SummariseReviews("chat", judge, threshold, replies, reviews)
	result.Overall = overall
	return result, nil
}

// parseOverall is the judge's verdict on the conversation as a whole, when it
// gave one.
func parseOverall(raw string) *Overall {
	data, ok := loadsLenient(raw).(map[string]any)
	if !ok {
		return nil
	}
	var item map[string]any
	for _, key := range []string{"overall", "conversation", "summary"} {
		if found, ok := data[key].(map[string]any); ok {
			item = found
			break
		}
	}
	if item == nil {
		return nil
	}
	out := &Overall{}
	if number, err := toNumber(firstPresent(item, "rating", "score")); err == nil {
		value := math.Max(0, math.Min(10, number))
		out.Rating = &value
	}
	out.Critique = collapse(pythonString(firstPresent(item, "critique", "reason", "comment")))
	if out.Critique == "None" {
		out.Critique = ""
	}
	if out.Rating == nil && out.Critique == "" {
		return nil
	}
	if out.Critique == "" {
		out.Critique = "no critique given"
	}
	return out
}
