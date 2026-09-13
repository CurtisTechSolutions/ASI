package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

var negFailures = []string{"the the the the cat", "the cat cat cat sat"}

// blameFailures teaches the server's negative network the repeated-word failures.
func (e *env) blameFailures(body map[string]any) map[string]any {
	e.t.Helper()
	if body == nil {
		body = map[string]any{}
	}
	if _, ok := body["texts"]; !ok {
		body["texts"] = negFailures
	}
	if _, ok := body["reason"]; !ok {
		body["reason"] = "repetition"
	}
	status, doc := e.post("/api/negative/blame", body)
	if status != 200 {
		e.t.Fatalf("blame: %d %+v", status, doc)
	}
	return doc
}

func TestNegativeStatusAndBlame(t *testing.T) {
	e := newEnv(t, false)
	e.blameFailures(map[string]any{"note": "it repeats", "source": "review"})
	status, doc := e.get("/api/negative")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	hasKeys(t, doc, "path", "stats", "reasons", "journal", "weights", "settings")
	if doc["path"] != filepath.Join(e.dir, "model.count.negative.json") {
		t.Fatalf("the negative network lives beside the model: %v", doc["path"])
	}
	reasons := doc["reasons"].([]any)
	if len(reasons) != 1 || reasons[0].(map[string]any)["reason"] != "repetition" {
		t.Fatalf("the reason table: %+v", reasons)
	}
	journal := doc["journal"].([]any)
	if len(journal) == 0 || journal[0].(map[string]any)["note"] != "it repeats" {
		t.Fatalf("the journal keeps the tutor's words: %+v", journal)
	}
	if doc["stats"].(map[string]any)["kind"] != "negative" {
		t.Fatalf("stats: %+v", doc["stats"])
	}
	if status, doc := e.post("/api/negative/blame", map[string]any{"texts": []string{}}); status != 400 {
		t.Fatalf("nothing to blame is a 400: %d %+v", status, doc)
	}
}

func TestNegativeJudgeAndClear(t *testing.T) {
	e := newEnv(t, false)
	e.blameFailures(nil)
	status, doc := e.post("/api/negative/judge", map[string]any{"text": negFailures[0]})
	if status != 200 {
		t.Fatalf("judge: %d %+v", status, doc)
	}
	verdict := doc["verdicts"].([]any)[0].(map[string]any)
	if verdict["verdict"] != "reject" {
		t.Fatalf("a known failure is rejected: %+v", verdict)
	}
	if verdict["reasons"].([]any)[0].(map[string]any)["reason"] != "repetition" {
		t.Fatalf("with its reason: %+v", verdict["reasons"])
	}
	status, doc = e.post("/api/negative/clear", map[string]any{"texts": []string{"the cat sat on the mat"}})
	if status != 200 {
		t.Fatalf("clear: %d %+v", status, doc)
	}
	if toFloat(doc["matched"])+toFloat(doc["unmatched"]) != 1 {
		t.Fatalf("one text, matched or not: %+v", doc)
	}
	if status, _ := e.post("/api/negative/judge", map[string]any{"texts": []string{}}); status != 400 {
		t.Fatalf("nothing to judge is a 400: %d", status)
	}
}

func TestNegativeFilter(t *testing.T) {
	e := newEnv(t, false)
	if status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitJob()
	e.blameFailures(nil)
	status, doc := e.post("/api/negative/filter", map[string]any{"texts": []string{negFailures[0], "an unseen sentence"}})
	if status != 200 {
		t.Fatalf("filter: %d %+v", status, doc)
	}
	verdicts := doc["verdicts"].([]any)
	if verdicts[0].(map[string]any)["decision"] != "reject" || verdicts[1].(map[string]any)["decision"] != "pass" {
		t.Fatalf("the known failure is vetoed and the unseen text kept: %+v", verdicts)
	}
	if verdicts[0].(map[string]any)["rule"] != "blame" {
		t.Fatalf("by blame: %+v", verdicts[0])
	}
	kept := doc["kept"].([]any)
	if len(kept) != 1 || kept[0] != "an unseen sentence" {
		t.Fatalf("what survives: %+v", kept)
	}
	pair := doc["pair"].(map[string]any)
	if pair["negative"].(map[string]any)["kind"] != "negative" || pair["positive"].(map[string]any)["kind"] != "count" {
		t.Fatalf("the pair names both halves: %+v", pair)
	}
	// generating through the pair
	status, doc = e.post("/api/negative/filter", map[string]any{"count": 2, "max_length": 30, "over_sample": 2})
	if status != 200 {
		t.Fatalf("generate: %d %+v", status, doc)
	}
	if len(doc["texts"].([]any)) > 2 {
		t.Fatalf("at most the wanted count: %+v", doc["texts"])
	}
	if len(doc["verdicts"].([]any)) != int(toFloat(doc["candidates"])) {
		t.Fatalf("one verdict per candidate: %+v", doc)
	}
	// the rules can be tuned
	body := map[string]any{"texts": []string{negFailures[0]}, "threshold": 1e9, "min_coverage": 0.0}
	_, doc = e.post("/api/negative/filter", body)
	if doc["verdicts"].([]any)[0].(map[string]any)["rule"] != "ratio" {
		t.Fatalf("blame cannot reject, the ratio still can: %+v", doc["verdicts"])
	}
	body["no_ratio"] = true
	_, doc = e.post("/api/negative/filter", body)
	if doc["verdicts"].([]any)[0].(map[string]any)["decision"] != "suspect" {
		t.Fatalf("both rules off: %+v", doc["verdicts"])
	}
	body["peak"] = 1.0
	_, doc = e.post("/api/negative/filter", body)
	if doc["verdicts"].([]any)[0].(map[string]any)["rule"] != "peak" {
		t.Fatalf("the peak rule: %+v", doc["verdicts"])
	}
	if status, _ := e.post("/api/negative/filter", map[string]any{"over_sample": 0}); status != 400 {
		t.Fatalf("an over-sample of zero is a 400: %d", status)
	}
}

func TestNegativeSettingsForgetResetAndSave(t *testing.T) {
	e := newEnv(t, false)
	e.blameFailures(nil)
	status, doc := e.post("/api/negative/settings", map[string]any{"threshold": 2.5, "min_coverage": 0.1})
	if status != 200 {
		t.Fatalf("settings: %d %+v", status, doc)
	}
	settings := doc["settings"].(map[string]any)
	if toFloat(settings["threshold"]) != 2.5 || toFloat(settings["min_coverage"]) != 0.1 {
		t.Fatalf("the thresholds move: %+v", settings)
	}
	status, doc = e.post("/api/negative/settings", map[string]any{"clear_scale": 2.0})
	if toFloat(doc["weights"].(map[string]any)["clear_scale"]) != 2 {
		t.Fatalf("the weight scales move: %+v", doc["weights"])
	}
	status, doc = e.post("/api/negative/forget", map[string]any{"reason": "repetition"})
	if status != 200 || toFloat(doc["blame_removed"]) <= 0 {
		t.Fatalf("forget: %d %+v", status, doc)
	}
	status, doc = e.post("/api/negative/save", nil)
	if status != 200 {
		t.Fatalf("save: %d %+v", status, doc)
	}
	path := doc["path"].(string)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the file is written: %v", err)
	}
	loaded, err := radixnet.Load(path)
	if err != nil || !loaded.IsNegative() {
		t.Fatalf("and holds a negative network: %v", err)
	}
	// /api/save writes the negative network beside the model
	e.blameFailures(nil)
	status, doc = e.post("/api/save", nil)
	if status != 200 || doc["negative"] == nil {
		t.Fatalf("/api/save writes both: %d %+v", status, doc)
	}
	status, doc = e.post("/api/negative/reset", nil)
	if status != 200 || toFloat(doc["stats"].(map[string]any)["failures_total"]) != 0 {
		t.Fatalf("reset forgets everything: %d %+v", status, doc)
	}
	if status, _ := e.post("/api/negative/forget", map[string]any{"factor": 2.0}); status != 400 {
		t.Fatalf("a factor of 2 is a 400: %d", status)
	}
}

func TestNegativeModelIsLoadedFromItsFile(t *testing.T) {
	e := newEnv(t, false)
	e.blameFailures(nil)
	if _, doc := e.post("/api/negative/save", nil); doc["path"] == nil {
		t.Fatal("save")
	}
	// a second service on the same model path picks the negative network up
	svc, err := NewService(Options{ModelPath: filepath.Join(e.dir, "model.count.json"), Seed: 1, Quiet: true})
	if err != nil {
		t.Fatal(err)
	}
	status, err := svc.NegativeStatus()
	if err != nil {
		t.Fatal(err)
	}
	reasons := status["reasons"].([]radixnet.ReasonRow)
	if len(reasons) != 1 || reasons[0].Reason != "repetition" {
		t.Fatalf("the negative network is loaded from its file: %+v", reasons)
	}
}
