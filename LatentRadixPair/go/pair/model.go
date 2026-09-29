package pair

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/tokenizer"
)

// Format names the model file format.
const Format = "latentpair"

// Record is one line of a model's history.
type Record map[string]any

// Model is a trained tokenizer with a primed radix pair over its codes.
type Model struct {
	Tok     *tokenizer.Tokenizer
	Pair    *Pair
	Seed    int64
	Created string
	History []Record
	rng     *rand.Rand
	draws   int
}

// Prime allocates both trees for the tokenizer's codes.
func Prime(tok *tokenizer.Tokenizer, s Settings, seed int64) (*Model, error) {
	p, err := NewPair(tok.Radices(), s)
	if err != nil {
		return nil, err
	}
	return &Model{Tok: tok, Pair: p, Seed: seed, Created: time.Now().UTC().Format(time.RFC3339),
		rng: rand.New(rand.NewSource(seed))}, nil
}

func (m *Model) record(r Record) Record {
	m.History = append(m.History, r)
	return r
}

// TrainBytes reads texts into the count tree.
func (m *Model) TrainBytes(texts [][]byte) Record {
	units := 0
	for _, text := range texts {
		m.Pair.Count.Observe(m.Tok.EncodeAll(text), Outcomes(text))
		units += len(text) + 1
	}
	return m.record(Record{"call": "train", "texts": len(texts), "units": units})
}

// Train reads texts into the count tree.
func (m *Model) Train(texts []string) Record { return m.TrainBytes(asBytes(texts)) }

func asBytes(texts []string) [][]byte {
	out := make([][]byte, len(texts))
	for i, t := range texts {
		out[i] = []byte(t)
	}
	return out
}

// credit judges texts: amount = sign * strength * weight / outcomes each, from the position after the
// prefix on, where outcomes is the number of potential outcomes of the space the verdict comes from (0: the
// model's setting). It records the call in the history unless call is empty.
func (m *Model) credit(texts [][]byte, sign, strength float64, weights []float64, read bool, call string, prefix []byte, outcomes int) (Record, error) {
	if strength <= 0 {
		strength = m.Pair.Settings.Strength
	}
	if outcomes <= 0 {
		outcomes = m.Pair.Settings.Outcomes
	}
	unit := 1 / float64(outcomes)
	if weights != nil && len(weights) != len(texts) {
		return nil, fmt.Errorf("%d weights for %d texts", len(weights), len(texts))
	}
	cells := 0
	for i, text := range texts {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if w < 0 {
			return nil, fmt.Errorf("weights must be >= 0, got %v", w)
		}
		full := append(append([]byte(nil), prefix...), text...)
		codes := m.Tok.EncodeAll(full)
		steps := Outcomes(full)
		if read {
			m.Pair.Count.Observe(codes, steps)
		}
		n, err := m.Pair.Reward.Credit(codes, steps, sign*strength*w*unit, m.Pair.Settings.Rungs, len(prefix))
		if err != nil {
			return nil, err
		}
		cells += n
	}
	r := Record{"call": call, "texts": len(texts), "strength": strength, "outcomes": outcomes, "amount": strength * unit, "cells": cells, "read": read}
	if len(prefix) > 0 {
		r["prefix"] = string(prefix)
	}
	if call == "" {
		return r, nil
	}
	return m.record(r), nil
}

// Reward credits texts as correct outcomes (read: also count them), the prefix being context, not outcome;
// outcomes is the size of the space the verdict comes from (0: the model's setting).
func (m *Model) Reward(texts []string, strength float64, weights []float64, read bool, prefix string, outcomes int) (Record, error) {
	return m.credit(asBytes(texts), 1, strength, weights, read, "reward", []byte(prefix), outcomes)
}

// Punish charges texts as wrong outcomes; they are never counted.
func (m *Model) Punish(texts []string, strength float64, weights []float64, prefix string, outcomes int) (Record, error) {
	return m.credit(asBytes(texts), -1, strength, weights, false, "punish", []byte(prefix), outcomes)
}

// TwoNRL punishes the bad texts and rewards the good ones.
func (m *Model) TwoNRL(bad, good []string, strength float64, prefix string, outcomes int) (Record, error) {
	a, err := m.credit(asBytes(bad), -1, strength, nil, false, "", []byte(prefix), outcomes)
	if err != nil {
		return nil, err
	}
	b, err := m.credit(asBytes(good), 1, strength, nil, true, "", []byte(prefix), outcomes)
	if err != nil {
		return nil, err
	}
	r := Record{"call": "two_nrl", "punished": a["cells"], "rewarded": b["cells"], "bad": len(bad), "good": len(good),
		"strength": strength, "outcomes": a["outcomes"], "amount": a["amount"]}
	if prefix != "" {
		r["prefix"] = prefix
	}
	return m.record(r), nil
}

// Feedback judges each text by its mark: a positive mark rewards with that weight, a negative one punishes
// with its size, zero is ignored.
func (m *Model) Feedback(texts []string, marks []float64, prefix string, outcomes int) (Record, error) {
	if len(marks) != len(texts) {
		return nil, fmt.Errorf("%d marks for %d texts", len(marks), len(texts))
	}
	var good, bad []string
	var gw, bw []float64
	for i, mk := range marks {
		switch {
		case mk > 0:
			good = append(good, texts[i])
			gw = append(gw, mk)
		case mk < 0:
			bad = append(bad, texts[i])
			bw = append(bw, -mk)
		}
	}
	if outcomes <= 0 {
		outcomes = m.Pair.Settings.Outcomes
	}
	r := Record{"call": "feedback", "rewarded": len(good), "punished": len(bad), "ignored": len(texts) - len(good) - len(bad), "outcomes": outcomes}
	if len(good) > 0 {
		if _, err := m.credit(asBytes(good), 1, 0, gw, true, "", []byte(prefix), outcomes); err != nil {
			return nil, err
		}
	}
	if len(bad) > 0 {
		if _, err := m.credit(asBytes(bad), -1, 0, bw, false, "", []byte(prefix), outcomes); err != nil {
			return nil, err
		}
	}
	if prefix != "" {
		r["prefix"] = prefix
	}
	return m.record(r), nil
}

// Code is the tokenizer's code of a prefix's context.
func (m *Model) Code(prefix string) []int32 { return m.Tok.Encode([]byte(prefix)) }

// Fold is the next-outcome distribution after a prefix.
func (m *Model) Fold(prefix, traversal, backoff string) ([]float64, error) {
	if traversal == "" {
		traversal = "reward"
	}
	return m.Pair.Fold(m.Code(prefix), traversal, backoff)
}

// Predict continues a prefix by length outcomes: the exact fold's most probable outcome each step (mode
// "greedy") or a sample at a temperature ("sample"); toEnd walks until the end mark (maxUnits at most).
func (m *Model) Predict(prefix string, length int, mode, traversal string, temperature float64, toEnd bool, backoff string, maxUnits int) (PathResult, error) {
	if traversal == "" {
		traversal = "reward"
	}
	encode := func(ctx []byte) []int32 { return m.Tok.Encode(ctx) }
	switch mode {
	case "", "greedy":
		return m.Pair.Walk([]byte(prefix), encode, length, traversal, nil, 0, toEnd, backoff, maxUnits)
	case "sample":
		rng := &countingRand{m.rng, &m.draws}
		return m.Pair.Walk([]byte(prefix), encode, length, traversal, rand.New(rng), temperature, toEnd, backoff, maxUnits)
	}
	return PathResult{}, fmt.Errorf("mode must be greedy or sample, got %q", mode)
}

// countingRand counts the draws so a saved model can replay its random state.
type countingRand struct {
	src   *rand.Rand
	draws *int
}

func (c *countingRand) Int63() int64 {
	*c.draws++
	return c.src.Int63()
}
func (c *countingRand) Seed(int64) {}

// Score prices a text.
func (m *Model) Score(text, traversal, backoff string) (Score, error) {
	if traversal == "" {
		traversal = "reward"
	}
	b := []byte(text)
	return m.Pair.ScoreText(m.Tok.EncodeAll(b), Outcomes(b), traversal, backoff)
}

// Decode reconstructs a prefix's context from its code's first known symbols (0: all of them).
func (m *Model) Decode(prefix string, known int) (string, error) {
	return m.Tok.Decode(m.Code(prefix), known)
}

// Coverage is how much of a text's contexts the count tree has read.
func (m *Model) Coverage(text string) float64 {
	return m.Pair.Count.Coverage(m.Tok.EncodeAll([]byte(text)))
}

// Info describes the model.
func (m *Model) Info() map[string]any {
	count, reward := m.Pair.Count, m.Pair.Reward
	nz, nr := 0, 0
	count.Nonzero(func(int, int64) { nz++ })
	reward.Nonzero(func(int, float64, float64) { nr++ })
	used := 0
	for _, c := range count.Ctx {
		if c > 0 {
			used++
		}
	}
	return map[string]any{
		"format": Format, "seed": m.Seed, "created": m.Created,
		"tokenizer": m.Tok.Info(), "radices": m.Pair.Addr.Radices, "depth": m.Pair.Addr.D(),
		"nodes": m.Pair.Addr.N, "cells": m.Pair.Addr.Cells(), "memory_bytes": m.Pair.MemoryBytes(),
		"read":           map[string]any{"texts": count.Texts, "units": count.Units, "contexts_used": used},
		"judged":         map[string]any{"texts": reward.Judged, "rewards_total": reward.RewardsTotal, "penalties_total": reward.PenaltiesTotal},
		"nonzero_counts": nz, "nonzero_rewards": nr, "settings": m.Pair.Settings,
	}
}

type fileJSON struct {
	Format    string          `json:"format"`
	Version   int             `json:"version"`
	Tokenizer json.RawMessage `json:"tokenizer"`
	Seed      int64           `json:"seed"`
	Draws     int             `json:"draws"`
	Settings  Settings        `json:"settings"`
	Created   string          `json:"created"`
	Read      struct {
		Texts int `json:"texts"`
		Units int `json:"units"`
	} `json:"read"`
	Judged struct {
		Texts          int     `json:"texts"`
		RewardsTotal   float64 `json:"rewards_total"`
		PenaltiesTotal float64 `json:"penalties_total"`
	} `json:"judged"`
	History []Record `json:"history"`
	Counts  struct {
		Cells  []int   `json:"cells"`
		Values []int64 `json:"values"`
	} `json:"counts"`
	Rewards struct {
		Cells []int     `json:"cells"`
		Plus  []float64 `json:"plus"`
		Minus []float64 `json:"minus"`
	} `json:"rewards"`
}

// ToJSON serialises the model, tokenizer included, with only the non-zero cells.
func (m *Model) ToJSON() ([]byte, error) {
	tok, err := m.Tok.ToJSON()
	if err != nil {
		return nil, err
	}
	f := fileJSON{Format: Format, Version: 1, Tokenizer: tok, Seed: m.Seed, Draws: m.draws, Settings: m.Pair.Settings,
		Created: m.Created, History: m.History}
	if f.History == nil {
		f.History = []Record{}
	}
	f.Read.Texts, f.Read.Units = m.Pair.Count.Texts, m.Pair.Count.Units
	f.Judged.Texts, f.Judged.RewardsTotal, f.Judged.PenaltiesTotal = m.Pair.Reward.Judged, m.Pair.Reward.RewardsTotal, m.Pair.Reward.PenaltiesTotal
	f.Counts.Cells, f.Counts.Values = []int{}, []int64{}
	m.Pair.Count.Nonzero(func(cell int, c int64) {
		f.Counts.Cells = append(f.Counts.Cells, cell)
		f.Counts.Values = append(f.Counts.Values, c)
	})
	f.Rewards.Cells, f.Rewards.Plus, f.Rewards.Minus = []int{}, []float64{}, []float64{}
	m.Pair.Reward.Nonzero(func(cell int, plus, minus float64) {
		f.Rewards.Cells = append(f.Rewards.Cells, cell)
		f.Rewards.Plus = append(f.Rewards.Plus, plus)
		f.Rewards.Minus = append(f.Rewards.Minus, minus)
	})
	return json.Marshal(f)
}

// FromJSON reads a model written by ToJSON.
func FromJSON(data []byte) (*Model, error) {
	var f fileJSON
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	if f.Format != Format {
		return nil, fmt.Errorf("not a %s file (format %q)", Format, f.Format)
	}
	tok, err := tokenizer.FromJSON(f.Tokenizer)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %v", err)
	}
	if f.Settings.Outcomes == 0 {
		f.Settings.Outcomes = EnglishPhones // files written before verdicts had units
	}
	m, err := Prime(tok, f.Settings, f.Seed)
	if err != nil {
		return nil, err
	}
	m.Created, m.History = f.Created, f.History
	for i := 0; i < f.Draws; i++ {
		m.rng.Int63()
	}
	m.draws = f.Draws
	count, reward := m.Pair.Count, m.Pair.Reward
	if len(f.Counts.Cells) != len(f.Counts.Values) || len(f.Rewards.Cells) != len(f.Rewards.Plus) || len(f.Rewards.Cells) != len(f.Rewards.Minus) {
		return nil, fmt.Errorf("cell lists of unequal length")
	}
	for i, cell := range f.Counts.Cells {
		if cell < 0 || cell >= len(count.Cnt) {
			return nil, fmt.Errorf("count cell %d outside the address space", cell)
		}
		count.Cnt[cell] = f.Counts.Values[i]
		count.Ctx[cell/m.Pair.Addr.Out] += f.Counts.Values[i]
	}
	for i, cell := range f.Rewards.Cells {
		if cell < 0 || cell >= len(reward.Plus) {
			return nil, fmt.Errorf("reward cell %d outside the address space", cell)
		}
		reward.Plus[cell], reward.Minus[cell] = f.Rewards.Plus[i], f.Rewards.Minus[i]
	}
	count.Texts, count.Units = f.Read.Texts, f.Read.Units
	reward.Judged, reward.RewardsTotal, reward.PenaltiesTotal = f.Judged.Texts, f.Judged.RewardsTotal, f.Judged.PenaltiesTotal
	return m, nil
}

// Save writes the model (gzip when the path ends in .gz).
func (m *Model) Save(path string) error {
	data, err := m.ToJSON()
	if err != nil {
		return err
	}
	return tokenizer.WriteFile(path, data)
}

// Load reads a model written by Save.
func Load(path string) (*Model, error) {
	data, err := tokenizer.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return FromJSON(data)
}
