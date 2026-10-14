package kit

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Auto-temperature through the kit (../../../RadixCyclicNN/SPEC-SearchAndTraining.md
// §10): the guard's veto counts as no answer, and every output path can climb.
// A prediction the guard leaves without an answer walks again up the ladder
// (GuardedPredict), a guarded generation draws again for the samples it
// vetoed (Filter.AutoGenerate), a reply with nothing to say tries again hotter
// (Reply) - and today's format carries the option and narrates the climb.
// Off, every one of them is the path it always was.  The cases of
// tests/test_autotemp.py.

var climbCorpus = []string{"the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill"}

const climbFailure = "the cat sat on the mat"

// climbVoice is the count model the cases speak with: the three lines above, three epochs.
func climbVoice(t *testing.T) *radixnet.Model {
	t.Helper()
	m, err := radixnet.NewModel(1, radixnet.DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true
	opts := radixnet.DefaultTrainOptions()
	opts.Epochs = 3
	if _, err := m.Train(climbCorpus, opts); err != nil {
		t.Fatal(err)
	}
	return m
}

// climbPair is that voice guarded by a negative network that has been taught blamed as failures.
func climbPair(t *testing.T, config FilterConfig, blamed ...string) *Filter {
	t.Helper()
	negative, err := radixnet.NewNegativeModel(2, radixnet.DefaultNegativeOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := negative.Blame(blamed, radixnet.BlameOptions{Reason: "gibberish"}); err != nil {
		t.Fatal(err)
	}
	pair, err := NewFilter(climbVoice(t), negative, config)
	if err != nil {
		t.Fatal(err)
	}
	return pair
}

func vetoedBeam() radixnet.PredictOptions {
	o := radixnet.DefaultPredictOptions()
	o.Mode, o.K, o.Length, o.ToEnd = "beam", 1, 30, true
	return o
}

func TestAVetoedPredictionClimbsToOneTheGuardLetsThrough(t *testing.T) {
	pair := climbPair(t, DefaultFilterConfig(), climbFailure)
	found, verdicts, record, err := GuardedPredict(pair.Positive, "the cat", pair, vetoedBeam(), 0)
	if err != nil || record != nil {
		t.Fatalf("off: %v %+v", err, record)
	}
	if found.Text != "" || len(verdicts) != 1 || verdicts[0].Decision != "reject" {
		t.Fatalf("off, the guard vetoed the only continuation and nothing is said: %q %d", found.Text, len(verdicts))
	}
	found, verdicts, record, err = GuardedPredict(pair.Positive, "the cat", pair, vetoedBeam(), 2)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(found.Text) == "" || !found.ReachedEnd || found.FullText == climbFailure {
		t.Fatalf("a warmer walk the guard lets through: %+v", found.PathResult)
	}
	if !record.Climbed || !record.Found || record.Tries <= 1 {
		t.Fatalf("the record: %+v", record)
	}
	if len(verdicts) != record.Tries || verdicts[0].Decision != "reject" {
		t.Fatalf("every search was judged: %d verdicts, %d tries", len(verdicts), record.Tries)
	}
	if _, _, _, err := GuardedPredict(pair.Positive, "the cat", pair, vetoedBeam(), -1); err == nil {
		t.Fatal("a negative ceiling is refused")
	}
}

func TestAnAnswerTheGuardLetsThroughIsNotWalkedPast(t *testing.T) {
	pair := climbPair(t, DefaultFilterConfig(), "nothing like the corpus at all")
	plain, _, _, err := GuardedPredict(pair.Positive, "the cat", pair, vetoedBeam(), 0)
	if err != nil {
		t.Fatal(err)
	}
	found, _, record, err := GuardedPredict(pair.Positive, "the cat", pair, vetoedBeam(), 2)
	if err != nil {
		t.Fatal(err)
	}
	if found.FullText != plain.FullText {
		t.Fatalf("the answer as asked: %q / %q", found.FullText, plain.FullText)
	}
	if want := (radixnet.AutoTemperature{Ceiling: 2, Found: true, Tries: 1}); !reflect.DeepEqual(*record, want) {
		t.Fatalf("found as asked by the beam, which reads no temperature: %+v", record)
	}
	// and with no guard at all, a search as asked is the model's own
	bare, verdicts, _, err := GuardedPredict(pair.Positive, "the cat", nil, vetoedBeam(), 0)
	if err != nil || len(verdicts) != 0 || bare.FullText != plain.FullText {
		t.Fatalf("unguarded: %v %d %q", err, len(verdicts), bare.FullText)
	}
}

func TestAGuardedGenerationDrawsAgainForWhatItVetoed(t *testing.T) {
	// no over-sampling, so the beam offers just the three it is asked for
	config := DefaultFilterConfig()
	config.OverSample = 1
	beamOpts := radixnet.DefaultGenerateOptions()
	beamOpts.Mode, beamOpts.MaxLength = "beam", 40
	beam, err := climbPair(t, config, "nothing like the corpus at all").Generate(3, beamOpts)
	if err != nil || len(beam.Texts) < 2 {
		t.Fatalf("the beam's texts: %v %v", err, beam)
	}
	pair := climbPair(t, config, beam.Texts[:2]...) // the beam's own two best texts are failures now
	seed := int64(3)
	o := beamOpts
	o.Seed = &seed
	plain, err := pair.AutoGenerate(3, o, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(plain.Texts) >= 3 || plain.AutoTemperature != nil {
		t.Fatalf("off: the beam's texts were vetoed, and there is no record: %v %+v", plain.Texts, plain.AutoTemperature)
	}
	if raw, _ := json.Marshal(plain); strings.Contains(string(raw), "auto_temperature") {
		t.Fatalf("off, no record on the wire: %s", raw)
	}
	out, err := pair.AutoGenerate(3, o, 2)
	if err != nil {
		t.Fatal(err)
	}
	record := out.AutoTemperature
	if record == nil || !record.Climbed {
		t.Fatalf("it climbed: %+v", record)
	}
	if len(out.Texts) != len(out.Results) || len(out.Texts) < len(plain.Texts) {
		t.Fatalf("a walk per text, and no fewer: %v", out.Texts)
	}
	if !reflect.DeepEqual(out.Texts[:len(plain.Texts)], plain.Texts) {
		t.Fatalf("the first generation's come first: %v / %v", out.Texts, plain.Texts)
	}
	seen := map[string]bool{}
	for i, text := range out.Texts {
		if seen[text] || out.Results[i].Text != text {
			t.Fatalf("distinct texts, each with its walk: %v", out.Texts)
		}
		seen[text] = true
	}
	if out.Candidates <= plain.Candidates || out.Asked <= plain.Asked || len(out.Verdicts) != out.Candidates {
		t.Fatalf("every generation adds up: %d / %d candidates, %d asked, %d verdicts", out.Candidates,
			plain.Candidates, out.Asked, len(out.Verdicts))
	}
	if out.Rate == nil || *out.Rate != float64(len(out.Kept))/float64(out.Candidates) {
		t.Fatalf("the rate is kept over judged: %v", out.Rate)
	}
	if record.Found != (len(out.Texts) == 3) {
		t.Fatalf("found means all three: %+v %v", record, out.Texts)
	}
	// a count of 0 asks for nothing: nothing is missing, so nothing climbs - and the ceiling is on, so the
	// record says so, as it does unguarded (Python's NegativeFilter.generate and autotemp.generate alike)
	empty, err := pair.AutoGenerate(0, o, 2)
	if err != nil || len(empty.Texts) != 0 || empty.AutoTemperature == nil {
		t.Fatalf("nothing wanted: %v %+v", err, empty)
	}
	if r := empty.AutoTemperature; !r.Found || r.Climbed || r.Tries != 1 || r.Ceiling != 2 {
		t.Fatalf("found as asked: %+v", r)
	}
	if _, err := pair.AutoGenerate(3, o, -1); err == nil {
		t.Fatal("a negative ceiling is refused")
	}
}

func TestEverythingVetoedIsNoAnswerAtAnyTemperature(t *testing.T) {
	pair := climbPair(t, DefaultFilterConfig(), climbCorpus...)
	seed := int64(3)
	o := radixnet.DefaultGenerateOptions()
	o.Mode, o.MaxLength, o.Seed = "sample", 40, &seed
	out, err := pair.AutoGenerate(2, o, 1)
	if err != nil {
		t.Fatal(err)
	}
	record := out.AutoTemperature
	if !record.Climbed {
		t.Fatalf("it climbed: %+v", record)
	}
	if !record.Found && (len(out.Texts) >= 2 || record.Tries != 1+radixnet.AutoDraws || record.Temperature != nil) {
		t.Fatalf("the ladder (1, 1) is one rung: %+v %v", record, out.Texts)
	}
}

// vetoTheBeam is a veto of every text the beam offers from the start: a reply as asked has nothing it may say.
func vetoTheBeam(t *testing.T) func(string) bool {
	t.Helper()
	o := radixnet.DefaultPredictOptions()
	o.Mode, o.Length, o.K, o.ToEnd, o.MaxLength = "beam", 0, 3, true, 60
	found, err := climbVoice(t).Predict("", o)
	if err != nil {
		t.Fatal(err)
	}
	offered := map[string]bool{}
	for _, r := range found.Top {
		offered[r.FullText] = true
	}
	return func(text string) bool { return offered[text] }
}

func beamReply(veto func(string) bool, seed int64) ReplyOptions {
	o := DefaultReplyOptions()
	o.Mode, o.K, o.Veto, o.RNG = "beam", 3, veto, radixnet.NewMT19937(seed)
	return o
}

func TestAReplyWithNothingToSayTriesAgainHotter(t *testing.T) {
	veto := vetoTheBeam(t)
	if turn, err := Reply(climbVoice(t), "", beamReply(veto, 1)); err != nil || turn != nil {
		t.Fatalf("as asked, nothing may be said: %v %+v", err, turn)
	}
	events := []map[string]any{}
	o := beamReply(veto, 1)
	o.AutoTemperature = 2
	o.Trace = func(event map[string]any) { events = append(events, event) }
	turn, err := Reply(climbVoice(t), "", o)
	if err != nil {
		t.Fatal(err)
	}
	if turn == nil || veto(turn.Text) {
		t.Fatalf("a warmer walk says something else: %+v", turn)
	}
	record := turn.AutoTemperature
	if record == nil || !record.Climbed || !record.Found || record.Temperature == nil {
		t.Fatalf("the turn carries the climb: %+v", record)
	}
	raw, err := json.Marshal(turn)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if got, _ := doc["auto_temperature"].(map[string]any); got == nil || got["climbed"] != true || got["tries"] != float64(record.Tries) {
		t.Fatalf("and its record says so: %s", raw)
	}
	climbs := []map[string]any{}
	for _, event := range events {
		if event["kind"] == "climb" {
			climbs = append(climbs, event)
		}
	}
	if len(climbs) == 0 || !reflect.DeepEqual(climbs[0], map[string]any{"kind": "climb", "temperature": 1.0, "draws": radixnet.AutoDraws}) {
		t.Fatalf("the trace hears the first rung: %v", climbs)
	}
	if climbs[len(climbs)-1]["temperature"] != *record.Temperature {
		t.Fatalf("the last rung climbed is the one that answered: %v / %v", climbs, *record.Temperature)
	}
	rungs := map[float64]bool{}
	for _, event := range climbs {
		temperature := event["temperature"].(float64)
		if rungs[temperature] {
			t.Fatalf("one event a rung: %v", climbs)
		}
		rungs[temperature] = true
	}
}

func TestAReplyThatAnswersIsLeftAlone(t *testing.T) {
	o := beamReply(nil, 1)
	o.AutoTemperature = 2
	turn, err := Reply(climbVoice(t), "the cat", o)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := Reply(climbVoice(t), "the cat", beamReply(nil, 1))
	if err != nil {
		t.Fatal(err)
	}
	if turn.Text != plain.Text {
		t.Fatalf("the reply as asked: %q / %q", turn.Text, plain.Text)
	}
	if want := (radixnet.AutoTemperature{Ceiling: 2, Found: true, Tries: 1}); !reflect.DeepEqual(*turn.AutoTemperature, want) {
		t.Fatalf("found as asked: %+v", turn.AutoTemperature)
	}
	if raw, _ := json.Marshal(plain); strings.Contains(string(raw), "auto_temperature") {
		t.Fatalf("off, a turn's record is what it always was: %s", raw)
	}
	// a reply that was itself a walk answers at its own temperature
	o.Mode, o.Temperature = "sample", 0.7
	walked, err := Reply(climbVoice(t), "the cat", o)
	if err != nil || walked == nil || walked.AutoTemperature.Temperature == nil || *walked.AutoTemperature.Temperature != 0.7 {
		t.Fatalf("a sampled reply's temperature: %v %+v", err, walked)
	}
}

func TestAConversationGoesOnWhereItWouldHaveStopped(t *testing.T) {
	veto := vetoTheBeam(t)
	seed := int64(4)
	o := DefaultConverseOptions()
	o.Turns, o.K, o.Veto, o.Seed = 2, 3, veto, &seed
	stopped, err := Converse(climbVoice(t), "", o)
	if err != nil || len(stopped) != 0 {
		t.Fatalf("nothing may be said: %v %d", err, len(stopped))
	}
	o.AutoTemperature = 2
	going, err := Converse(climbVoice(t), "", o)
	if err != nil || len(going) == 0 {
		t.Fatalf("a warmer voice speaks: %v %d", err, len(going))
	}
	for _, turn := range going {
		if turn.AutoTemperature == nil || veto(turn.Text) {
			t.Fatalf("every turn carries its climb, and none was vetoed: %+v", turn)
		}
	}
}

func TestTheCeilingIsChecked(t *testing.T) {
	o := beamReply(nil, 1)
	o.AutoTemperature = -1
	if _, err := Reply(climbVoice(t), "the cat", o); err == nil {
		t.Fatal("a negative ceiling is refused by a reply")
	}
	c := DefaultConverseOptions()
	c.AutoTemperature = -1
	if _, err := Converse(climbVoice(t), "the cat", c); err == nil {
		t.Fatal("and by a conversation")
	}
}

func TestEitherDialectCarriesTheCeiling(t *testing.T) {
	for _, parse := range []func(map[string]any) (*Ask, error){ParseOpenAI, ParseAnthropic} {
		ask, err := parse(askBody(t, `{"messages": [{"role": "user", "content": "the cat"}], "max_tokens": 20, "auto_temperature": 2}`))
		if err != nil {
			t.Fatal(err)
		}
		if ask.AutoTemperature != 2 || ask.ToDict()["auto_temperature"] != 2.0 {
			t.Fatalf("read: %v %v", ask.AutoTemperature, ask.ToDict())
		}
		plain, err := parse(askBody(t, `{"messages": [{"role": "user", "content": "the cat"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if _, present := plain.ToDict()["auto_temperature"]; present || plain.AutoTemperature != 0 {
			t.Fatalf("off is not in the record: %v", plain.ToDict())
		}
		if raw, _ := json.Marshal(plain); strings.Contains(string(raw), "auto_temperature") || !strings.Contains(string(raw), `"seed":null`) {
			t.Fatalf("the JSON is the record: %s", raw)
		}
		for _, bad := range []string{`-1`, `"hot"`, `true`} {
			_, err := parse(askBody(t, `{"messages": [{"role": "user", "content": "the cat"}], "auto_temperature": `+bad+`}`))
			if ae, ok := err.(*AskError); !ok || ae.Param != "auto_temperature" {
				t.Fatalf("%s is refused, naming the field: %v", bad, err)
			}
		}
	}
	if DefaultAsk().AutoTemperature != 0 {
		t.Fatal("an ask is off by default")
	}
}

func TestTheThinkingSaysWhatTheClimbDoes(t *testing.T) {
	n := &narrator{emit: func(map[string]any) {}}
	n.trace(map[string]any{"kind": "climb", "temperature": 1.0, "draws": 3})
	n.trace(map[string]any{"kind": "climb", "temperature": 0.0, "draws": 1})
	want := []string{
		"nothing to say yet; trying again at temperature 1.00 (at most 3 replies)",
		"nothing to say yet; trying again at temperature 0.00 (at most 1 reply)",
	}
	if !reflect.DeepEqual(n.lines, want) {
		t.Fatalf("the lines: %q", n.lines)
	}
}

func TestTodaysFormatNarratesAClimbingReply(t *testing.T) {
	// "the cat on the mat" vetoed: as asked the reply has nothing to say - every context and the fresh start
	// offer one path, and each is a failure - so it is asked again, a walk at a time
	pair := climbPair(t, DefaultFilterConfig(), "the cat on the mat")
	body := `{"messages": [{"role": "user", "content": "the cat"}], "max_tokens": 30, "k": 1, "learn": false, "seed": 3`
	ask, err := ParseAnthropic(askBody(t, body+`}`))
	if err != nil {
		t.Fatal(err)
	}
	reply, err := Respond(pair.Positive, pair, ask, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if choice := reply.Choices[0]; choice.Turn != nil || choice.StopReason != StopRefusal {
		t.Fatalf("off, the guard vetoed everything it could say: %+v", choice)
	}
	if ask, err = ParseAnthropic(askBody(t, body+`, "auto_temperature": 2}`)); err != nil {
		t.Fatal(err)
	}
	reply, err = Respond(pair.Positive, pair, ask, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	choice := reply.Choices[0]
	if choice.Turn == nil || choice.Turn.AutoTemperature == nil || !choice.Turn.AutoTemperature.Climbed {
		t.Fatalf("on, it climbs to something to say: %+v", choice)
	}
	record := choice.Turn.AutoTemperature
	said := fmt.Sprintf("found something to say at temperature %.2f, on try %d", *record.Temperature, record.Tries)
	lines := strings.Split(choice.Thinking, "\n")
	climbed, found := -1, -1
	for i, line := range lines {
		if strings.HasPrefix(line, "nothing to say yet; trying again at temperature ") && climbed < 0 {
			climbed = i
		}
		if line == said {
			found = i
		}
	}
	if climbed < 0 || found <= climbed || found+1 >= len(lines) || !strings.HasPrefix(lines[found+1], "saying "+q(choice.Turn.Text)) {
		t.Fatalf("the thinking tells the climb, then what it says: %q", lines)
	}
}
