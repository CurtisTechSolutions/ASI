// Command fixtures writes the parity fixtures for the Rust port (run from LatentRadixPair/go; the output
// directory defaults to ../rust/testdata): a small tokenizer and model written by the Go implementation, with
// the outputs the Rust port must reproduce.
package main

import (
	"encoding/json"
	"os"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/pair"
	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/go/tokenizer"
)

var texts = []string{"the cat sat on the mat", "the cat sat on the log", "the dog ate the bone", "a cat and a dog",
	"the bird sang on the wire", "a dog and a cat sat", "the mat was on the floor", "the log was on the fire"}

func main() {
	out := "../rust/testdata"
	if len(os.Args) > 1 {
		out = os.Args[1]
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		panic(err)
	}
	binary := make([]byte, 0, 300)
	for i := 0; i < 300; i++ {
		binary = append(binary, byte((i*37+11)%256))
	}
	corpus := [][]byte{}
	for _, t := range texts {
		corpus = append(corpus, []byte(t))
	}
	corpus = append(corpus, binary)
	cfg := tokenizer.Config{Window: 8, Embed: 8, EncHidden: 32, DecHidden: 32, OutEmbed: 8,
		Levels: [][]int{{4, 4}, {4}, {2, 2}}, Noise: 0.05, Recency: 0.6, Predict: 2, StartShare: 0.1}
	tok, err := tokenizer.New(cfg, 7)
	if err != nil {
		panic(err)
	}
	o := tokenizer.DefaultOptions()
	o.Steps, o.Batch, o.EvalEvery, o.EvalWindows, o.LR = 300, 64, 100, 64, 5e-3
	if _, err := tok.Train(corpus, o); err != nil {
		panic(err)
	}
	if err := tok.Save(out + "/tokenizer.json"); err != nil {
		panic(err)
	}
	m, err := pair.Prime(tok, pair.DefaultSettings(), 3)
	if err != nil {
		panic(err)
	}
	m.Train(texts)
	must := func(_ pair.Record, err error) {
		if err != nil {
			panic(err)
		}
	}
	must(m.Reward([]string{"mat"}, 5, nil, true, "a cat sat on the ", 2))
	must(m.Punish([]string{"mat"}, 1, nil, "a cat sat on the ", 0))
	must(m.Reward([]string{"the dog ate the bone"}, 1, []float64{0.9}, true, "", 0))
	must(m.Punish([]string{"the bird sang on the wire"}, 1, nil, "", 0))
	must(m.TwoNRL([]string{"the cat sat on the fire"}, []string{"the cat sat on the mat"}, 1, "", 4))
	if err := m.Save(out + "/model.json.gz"); err != nil {
		panic(err)
	}
	type codeCase struct {
		Text  string  `json:"text"`
		Bytes []int   `json:"bytes"`
		Codes []int32 `json:"codes"`
	}
	type decodeCase struct {
		Context string  `json:"context"`
		Known   int     `json:"known"`
		Code    []int32 `json:"code"`
		Symbols []int   `json:"symbols"`
	}
	type foldCase struct {
		Prefix    string    `json:"prefix"`
		Traversal string    `json:"traversal"`
		Backoff   string    `json:"backoff"`
		Probs     []float64 `json:"probs"`
	}
	type predCase struct {
		Prefix     string  `json:"prefix"`
		Length     int     `json:"length"`
		Traversal  string  `json:"traversal"`
		Backoff    string  `json:"backoff"`
		ToEnd      bool    `json:"to_end"`
		Units      []int   `json:"units"`
		Text       string  `json:"text"`
		Cost       float64 `json:"cost"`
		ReachedEnd bool    `json:"reached_end"`
		Peak       float64 `json:"peak"`
	}
	type scoreCase struct {
		Text         string      `json:"text"`
		Traversal    string      `json:"traversal"`
		Bits         float64     `json:"bits"`
		MeanReward   float64     `json:"mean_reward"`
		WorstPenalty float64     `json:"worst_penalty"`
		Units        int         `json:"units"`
		PerUnit      [][]float64 `json:"per_unit"`
	}
	expected := map[string]any{}
	var codes []codeCase
	for _, t := range append(append([][]byte{}, corpus...), []byte(""), []byte("zq")) {
		bs := make([]int, len(t))
		for i, b := range t {
			bs[i] = int(b)
		}
		codes = append(codes, codeCase{Text: string(t), Bytes: bs, Codes: tok.EncodeAll(t).Data})
	}
	expected["codes"] = codes
	var decodes []decodeCase
	for _, ctx := range []string{"the cat sat on the ", "a dog", "", "x", "the log was on the fire"} {
		code := tok.Encode([]byte(ctx))
		for known := 1; known <= 3; known++ {
			syms, _ := tok.DecodeSymbols(code, known)
			decodes = append(decodes, decodeCase{Context: ctx, Known: known, Code: code, Symbols: syms})
		}
	}
	expected["decodes"] = decodes
	var folds []foldCase
	var preds []predCase
	for _, prefix := range []string{"the cat", "a cat sat on the ", "the dog", "", "zzz", "the cat sat on the mat"} {
		for _, traversal := range []string{"reward", "punishment"} {
			for _, backoff := range []string{"all", "deepest", "none"} {
				P, _ := m.Fold(prefix, traversal, backoff)
				folds = append(folds, foldCase{prefix, traversal, backoff, P})
				r, _ := m.Predict(prefix, 5, "greedy", traversal, 0, false, backoff, 0)
				preds = append(preds, predCase{prefix, 5, traversal, backoff, false, r.Units, r.Text, r.Cost, r.ReachedEnd, r.Peak})
			}
		}
		r, _ := m.Predict(prefix, 0, "greedy", "reward", 0, true, "", 30)
		preds = append(preds, predCase{prefix, 30, "reward", "", true, r.Units, r.Text, r.Cost, r.ReachedEnd, r.Peak})
	}
	expected["folds"] = folds
	expected["predictions"] = preds
	var scores []scoreCase
	for _, t := range append(append([]string{}, texts...), "a cat sat on the mat", "the cat sat on the fire", "zq xk", "") {
		for _, traversal := range []string{"reward", "punishment"} {
			s, _ := m.Score(t, traversal, "")
			var pu [][]float64
			for _, u := range s.PerUnit {
				pu = append(pu, []float64{u.Bits, u.Reward, u.Penalty})
			}
			scores = append(scores, scoreCase{t, traversal, s.Bits, s.MeanReward, s.WorstPenalty, s.Units, pu})
		}
	}
	expected["scores"] = scores
	info := m.Info()
	expected["info"] = map[string]any{"nodes": info["nodes"], "cells": info["cells"], "nonzero_counts": info["nonzero_counts"],
		"nonzero_rewards": info["nonzero_rewards"], "read": info["read"], "judged": info["judged"], "history": len(m.History)}
	cov := map[string]float64{}
	for _, t := range []string{"the cat sat on the mat", "zq xk vj qq", "the cat zq"} {
		cov[t] = m.Coverage(t)
	}
	expected["coverage"] = cov
	data, _ := json.Marshal(expected)
	if err := os.WriteFile(out+"/expected.json", data, 0o644); err != nil {
		panic(err)
	}
	println("fixtures written:", len(codes), "code cases,", len(folds), "folds,", len(preds), "predictions,", len(scores), "scores")
}
