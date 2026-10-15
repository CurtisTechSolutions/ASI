package radixnet

// Auto-temperature: a search that finds no answer walks again, hotter
// (../../SPEC-SearchAndTraining.md §10).
//
// The Go twin of radixnet/autotemp.py, kept draw for draw.  The temperature
// is the one dial that trades how likely a walk is for how far it strays: a
// sampled step weighs every option p ** (1 / temperature), so a cold walk
// keeps to the likeliest paths - and, in a cyclic graph, goes round them -
// while a hot one reaches paths the likely ones crowd out.  When a search
// comes back with no answer, this asks again up a ladder of temperatures, a
// few walks at every rung, and stops at the first answer:
//
//   - the search as asked runs first, exactly as it would have - any mode,
//     any temperature - and when it answers, nothing else happens;
//   - then the ladder (TemperatureLadder): the asked temperature, then
//     doubling - from AutoFloor when the asked one was greedy - up to a
//     ceiling, the last rung clamped to it.  Each rung is AutoDraws sampled
//     walks (one at temperature 0, where every walk is the same): an empty or
//     vetoed walk is mended by another draw as often as by a hotter one, and
//     a hotter walk is a less likely one, so the temperature rises only when
//     the draws at this one ran out;
//   - an answer is a walk that wrote something and, when the caller asked
//     for the end of a text, reached End (Answered); a caller with a guard
//     adds its veto (Climb takes any pick).  A search that offers several
//     continuations - the beam's top K - has found one when any of them
//     answers: the first that does becomes the prediction (FirstAnswer), as
//     the guard promotes the first one it lets through, and nothing is walked.
//
// A deterministic search - the beam, Dijkstra - reads no temperature at all,
// so for it the ladder is where sampling starts: the search found nothing at
// the cheapest paths, and walks at rising temperature look among the others.
// Where the graph has no way on at all, no temperature finds one, and the
// record says so.  The walks draw from the generator the request was already
// drawing from - the model's own, or the private one a seed gives - one search
// after the other, so a climb is reproducible and the ports draw alike.
//
// Off by default (a ceiling of 0): no extra search, no draw, the answer
// exactly the one the search gave, and no record (D-078).

import (
	"fmt"
	"math"
	"strings"
)

// AutoDraws is how many sampled walks every rung of the ladder draws before
// the temperature rises.
//
// Measured on a count model of 1,500 lines of prose: an empty continuation
// comes back from 4% of the walks at temperature 0.5, 6% at 1 and 13% at 4 -
// a hotter walk is no likelier to say something, so a second and a third draw
// mend more than the next rung does.  Three draws at a rung that answers 94%
// of the time leave one search in five thousand with nothing to say.
const AutoDraws = 3

// AutoFloor is the first temperature above greedy: a ladder that starts at 0
// climbs to this, then doubles.
const AutoFloor = 0.5

// CheckAutoTemperature is the error for a ceiling out of range: it must be a
// finite number >= 0 (0 is off).
func CheckAutoTemperature(ceiling float64) error {
	if math.IsNaN(ceiling) || math.IsInf(ceiling, 0) || ceiling < 0 {
		return fmt.Errorf("auto_temperature must be a finite number >= 0 (0 is off), got %v", ceiling)
	}
	return nil
}

// TemperatureLadder is the temperatures a search climbs through: start, then
// doubling up to ceiling.
//
// The rung after t is min(max(2t, AutoFloor), ceiling), so a greedy start
// climbs to AutoFloor first and the last rung is the ceiling itself: (1, 2) is
// [1 2], (1, 3) is [1 2 3], (0, 2) is [0 0.5 1 2].  A start at or above the
// ceiling is the one rung - the ladder never cools a walk - and a ceiling of 0
// is no ladder at all.  Temperature divides a log-probability, so the rungs
// are spaced by a factor rather than a step: each doubling is the same change
// in how flat a step is.
func TemperatureLadder(start, ceiling float64) []float64 {
	if !(ceiling > 0) {
		return nil
	}
	rungs := []float64{start}
	for t := start; t < ceiling; {
		t = math.Min(math.Max(2*t, AutoFloor), ceiling)
		rungs = append(rungs, t)
	}
	return rungs
}

// DrawsAt is how many walks the rung at temperature t draws: AutoDraws, or
// one at temperature 0, where every walk is the same.
func DrawsAt(t float64) int {
	if t == 0 {
		return 1
	}
	return AutoDraws
}

// Answered is whether a search's result is an answer: did it write something
// - more than whitespace - and, when toEnd asked for the end of a text, did it
// reach End?
func Answered(r *PathResult, toEnd bool) bool {
	return r != nil && strings.TrimSpace(r.Text) != "" && (r.ReachedEnd || !toEnd)
}

// FirstAnswer is the search's answer: p itself when it is one, else the first
// of the continuations it offered (Top, in the order the search ranked them)
// that is - made the prediction - else nil.
//
// The promoted continuation's Text, Labels, NodeIDs, Cost, StepCosts,
// ReachedEnd, FullText and Punish replace the prediction's own; Expanded, Top,
// Bottom, K, Beam, Mode and Traversal stay as the search left them.  The beam
// can rank a path cut at the length cap above the ones that reached End - a
// walk round a cycle is cheap - so a request for the end of a text may find
// its answer second or third: that is still the search's own answer, and
// better than any walk at a higher temperature.
func FirstAnswer(p *Prediction, toEnd bool) *Prediction {
	if p == nil {
		return nil
	}
	if Answered(&p.PathResult, toEnd) {
		return p
	}
	for _, candidate := range p.Top {
		if Answered(candidate, toEnd) {
			promoted := *p
			promoted.PathResult = *candidate
			promoted.Expanded = p.Expanded
			return &promoted
		}
	}
	return nil
}

// AutoTemperature is what a climb did: the ceiling it was given, whether the
// search as asked had to be retried (Climbed), whether an answer came back
// (Found), the temperature of the walk that gave it (nil - null - when the
// answer is a deterministic search's own, or there is none) and how many
// searches ran, the one as asked included (Tries).  It goes beside the answer
// as "auto_temperature" whenever the ceiling is on, and never when it is off.
//
// With nothing found, what comes back is the search as asked, and Temperature
// is nil: no walk answered.
type AutoTemperature struct {
	Ceiling     float64  `json:"ceiling"`
	Climbed     bool     `json:"climbed"`
	Found       bool     `json:"found"`
	Temperature *float64 `json:"temperature"`
	Tries       int      `json:"tries"`
}

// Climb runs the search as asked and, while it gives no answer, sampled walks
// up the ladder.
//
// search(nil) runs the caller's search exactly as it was asked for, and
// search(&t) the same request as a sampled walk at temperature t; pick makes
// an answer of what came back, or reports false when there is none in it: the
// model's own test (FirstAnswer), and whatever the caller adds - a guard's
// veto.  temperature is the one the request carries and sampled whether the
// search as asked was itself a sampled walk (then it is the temperature of an
// answer it gives).  The ladder is TemperatureLadder from temperature to
// ceiling, AutoDraws sampled walks at every rung (one at temperature 0), and
// the first answer ends it.  When none comes, the result is the search as
// asked - the most faithful thing there is to return - and the record says
// nothing was found.  An error from any search ends the climb with it.
func Climb[R any](search func(rung *float64) (R, error), pick func(R) (R, bool), temperature, ceiling float64, sampled bool) (R, *AutoTemperature, error) {
	var none R
	first, err := search(nil)
	if err != nil {
		return none, nil, err
	}
	record := &AutoTemperature{Ceiling: ceiling, Tries: 1}
	if chosen, ok := pick(first); ok {
		record.Found = true
		if sampled {
			asked := temperature
			record.Temperature = &asked
		}
		return chosen, record, nil
	}
	record.Climbed = true
	for _, t := range TemperatureLadder(temperature, ceiling) {
		for draw := 0; draw < DrawsAt(t); draw++ {
			record.Tries++
			rung := t
			found, err := search(&rung)
			if err != nil {
				return none, nil, err
			}
			if chosen, ok := pick(found); ok {
				record.Found = true
				record.Temperature = &rung
				return chosen, record, nil
			}
		}
	}
	return first, record, nil
}

// -- the two searches the model has -----------------------------------------------------------------

// AutoPredict is m.Predict(prefix, o), climbing the ladder to ceiling when it
// finds no answer.
//
// The record is nil when ceiling is 0 (off), and the search is then exactly
// the one Predict runs.  Any mode: the walks up the ladder are the same
// request in "sample" mode at each rung's temperature, keeping its cap (Length,
// or MaxLength when given), its traversal and its sampling filter.
func AutoPredict(m *Model, prefix string, o PredictOptions, ceiling float64) (*Prediction, *AutoTemperature, error) {
	if err := CheckAutoTemperature(ceiling); err != nil {
		return nil, nil, err
	}
	if ceiling == 0 {
		p, err := m.Predict(prefix, o)
		return p, nil, err
	}
	search := func(rung *float64) (*Prediction, error) {
		asked := o
		if rung != nil {
			asked.Mode, asked.Temperature = "sample", *rung
		}
		return m.Predict(prefix, asked)
	}
	pick := func(p *Prediction) (*Prediction, bool) {
		answer := FirstAnswer(p, o.ToEnd)
		return answer, answer != nil
	}
	return Climb(search, pick, o.Temperature, ceiling, strings.ToLower(o.Mode) == "sample")
}

// Wrote is whether a generated text said anything beyond its prefix.  A
// generated Text is the whole text, prefix included, so it is measured against
// the prefix in the model's units.
func Wrote(m *Model, r *PathResult, prefix string) bool {
	enc := m.Encoding()
	return r != nil && strings.TrimSpace(r.Text) != "" && enc.Len(r.Text) > enc.Len(prefix)
}

// Wanted is how many texts a generation of count gives when every one of them
// says something: count - and one for "dijkstra", whose generation is the
// single cheapest text however many are asked for, so that one text is the
// whole of its answer.
func Wanted(count int, mode string) int {
	if strings.ToLower(mode) == "dijkstra" && count > 1 {
		return 1
	}
	return count
}

// AutoGenerate is m.Generate(o), with every sample that said nothing walked
// again up the ladder to ceiling.
//
// The record is nil when ceiling is 0 (off), and the generation is then
// exactly the one Generate runs.  On, the generation as asked runs first; when
// fewer of its texts say anything beyond the prefix (Wrote) than it gives at
// most (Wanted: o.Count, and one for Dijkstra, whose generation is a single
// text), the missing ones are drawn as sampled walks up the ladder, AutoDraws
// generations a rung, each asking for as many as are still missing.  A walk
// keeps to the generation it fills: the beam's texts are distinct (and
// Dijkstra's is one), so a walk that repeats one already kept is not kept
// again, while sampled walks may repeat each other, as they always may.  What
// comes back is then the texts that said something - those of the generation
// as asked, in its order, then the ones the ladder found - and fewer only when
// even the ceiling found too few.  A Seed makes every generation draw from a
// generator of its own: the k-th retry draws from Seed + k, so a retry at the
// same temperature is a new draw and the whole climb is reproducible.
func AutoGenerate(m *Model, o GenerateOptions, ceiling float64) ([]*PathResult, *AutoTemperature, error) {
	if err := CheckAutoTemperature(ceiling); err != nil {
		return nil, nil, err
	}
	if ceiling == 0 {
		results, err := m.Generate(o)
		return results, nil, err
	}
	results, err := m.Generate(o)
	if err != nil {
		return nil, nil, err
	}
	mode := strings.ToLower(o.Mode)
	sampled := mode == "" || mode == "sample"
	want := Wanted(o.Count, mode)
	record := &AutoTemperature{Ceiling: ceiling, Tries: 1}
	kept := []*PathResult{}
	for _, r := range results {
		if Wrote(m, r, o.Prefix) {
			kept = append(kept, r)
		}
	}
	if len(kept) >= want {
		record.Found = true
		if sampled {
			asked := o.Temperature
			record.Temperature = &asked
		}
		return results, record, nil
	}
	record.Climbed = true
	var said map[string]bool // what a walk filling distinct texts may not repeat
	if !sampled {
		said = map[string]bool{}
		for _, r := range kept {
			said[r.Text] = true
		}
	}
	for _, t := range TemperatureLadder(o.Temperature, ceiling) {
		for draw := 0; draw < DrawsAt(t); draw++ {
			need := want - len(kept)
			retry := o
			retry.Mode, retry.Temperature, retry.Count = "sample", t, need
			if o.Seed != nil {
				seed := *o.Seed + int64(record.Tries) // the k-th retry draws from seed + k
				retry.Seed = &seed
			}
			record.Tries++
			more, err := m.Generate(retry)
			if err != nil {
				return nil, nil, err
			}
			for _, r := range more {
				if len(kept) < want && Wrote(m, r, o.Prefix) && !said[r.Text] {
					kept = append(kept, r)
					if said != nil {
						said[r.Text] = true
					}
				}
			}
			if len(kept) >= want {
				record.Found = true
				rung := t
				record.Temperature = &rung
				return kept, record, nil
			}
		}
	}
	return kept, record, nil
}
