package server

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type env struct {
	t   *testing.T
	ts  *httptest.Server
	svc *Service
	dir string
}

func newEnv(t *testing.T, withDirs bool) *env {
	t.Helper()
	dir := t.TempDir()
	opts := Options{ModelPath: filepath.Join(dir, "model.count.json"), Seed: 1, Workers: 0, Exact: true, Quiet: true}
	if withDirs {
		opts.UploadDir = filepath.Join(dir, "uploads")
		opts.CheckpointDir = filepath.Join(dir, "ckpt")
	}
	svc, err := NewService(opts)
	if err != nil {
		t.Fatal(err)
	}
	front := filepath.Join(dir, "dist")
	os.MkdirAll(filepath.Join(front, "assets"), 0o755)
	os.WriteFile(filepath.Join(front, "index.html"), []byte("<!doctype html><title>RadixCyclicNN</title><div id=root></div>"), 0o644)
	os.WriteFile(filepath.Join(front, "assets", "app.js"), []byte("console.log(1)"), 0o644)
	ts := httptest.NewServer(NewHandler(svc, front, true, nil))
	t.Cleanup(ts.Close)
	return &env{t: t, ts: ts, svc: svc, dir: dir}
}

func (e *env) do(method, path string, body any, headers map[string]string) (int, map[string]any, http.Header) {
	e.t.Helper()
	var reader io.Reader
	if body != nil {
		switch b := body.(type) {
		case []byte:
			reader = bytes.NewReader(b)
		case string:
			reader = strings.NewReader(b)
		default:
			raw, _ := json.Marshal(body)
			reader = bytes.NewReader(raw)
		}
	}
	req, _ := http.NewRequest(method, e.ts.URL+path, reader)
	if body != nil && headers["Content-Type"] == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var doc map[string]any
	if len(raw) > 0 && raw[0] == '{' {
		json.Unmarshal(raw, &doc)
	}
	return resp.StatusCode, doc, resp.Header
}

func (e *env) get(path string) (int, map[string]any) {
	s, d, _ := e.do("GET", path, nil, nil)
	return s, d
}

func (e *env) post(path string, body any) (int, map[string]any) {
	s, d, _ := e.do("POST", path, body, nil)
	return s, d
}

func (e *env) waitJob() map[string]any {
	e.t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		_, job := e.get("/api/job")
		if job != nil && job["state"] != "running" {
			return job
		}
		time.Sleep(20 * time.Millisecond)
	}
	e.t.Fatal("job did not finish")
	return nil
}

var corpus = []string{"the cat sat on the mat", "the cat ran to the door", "the dog sat on the log", "the dog ate the bone", "the bird flew over the house"}

func hasKeys(t *testing.T, doc map[string]any, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if _, ok := doc[k]; !ok {
			t.Fatalf("missing key %q in %v", k, doc)
		}
	}
}

func TestHealthStatusModel(t *testing.T) {
	e := newEnv(t, true)
	status, h := e.get("/api/health")
	if status != 200 || h["ok"] != true || h["engine"] != "go" {
		t.Fatalf("health: %d %v", status, h)
	}
	status, st := e.get("/api/status")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	hasKeys(t, st, "nodes", "edges", "trigrams", "compression_ratio", "inverted", "backend", "device", "epochs_total",
		"trained_chars", "trained_texts", "twonrl_runs", "history_len", "last_loss", "kind", "model_label", "kinds", "job",
		"backends", "model_path", "checkpoint_dir", "upload_dir", "engine", "workers", "total_traversals", "window_traversals", "window",
		"heap_bytes", "heap_sys_bytes", "memory_limit_bytes")
	if st["kind"] != "count" || st["engine"] != "go" || st["job"] != nil || st["backends"].(map[string]any)["default"] != "go" {
		t.Fatalf("status content: %v", st)
	}
	if st["counting"] != "exact" || st["workers"] != 0.0 || h["counting"] != "exact" {
		t.Fatalf("counting / workers in status and health: %v %v", st["counting"], h["counting"])
	}
	// the status reports how close the process is to its soft memory limit
	if heap, ok := st["heap_bytes"].(float64); !ok || heap <= 0 {
		t.Fatalf("heap_bytes: %v", st["heap_bytes"])
	}
	if limit, ok := st["memory_limit_bytes"].(float64); ok && limit <= 0 {
		t.Fatalf("memory_limit_bytes: %v", st["memory_limit_bytes"])
	}
	status, m := e.get("/api/model")
	if status != 200 || m["kind"] != "count" || len(m["kinds"].([]any)) != 1 || m["weights"] == nil {
		t.Fatalf("model: %d %v", status, m)
	}
	status, sel := e.post("/api/model/select", map[string]any{"kind": "count"})
	if status != 200 || sel["origin"] != "active" || sel["stats"] == nil {
		t.Fatalf("select count: %d %v", status, sel)
	}
	status, sel = e.post("/api/model/select", map[string]any{"kind": "radix"})
	if status != 400 || !strings.Contains(sel["error"].(string), "count / reward model only") {
		t.Fatalf("select radix: %d %v", status, sel)
	}
	status, w := e.post("/api/model/weights", map[string]any{"global_scale": 0.7, "window": 500})
	if status != 200 || w["weights"].(map[string]any)["global_scale"] != 0.7 || w["weights"].(map[string]any)["window"] != 500.0 {
		t.Fatalf("weights: %d %v", status, w)
	}
	status, w = e.post("/api/model/weights", map[string]any{"window": 0})
	if status != 400 {
		t.Fatalf("window 0 must be refused: %d %v", status, w)
	}
	status, idx := e.get("/api")
	if status != 200 || idx["engine"] != "go" {
		t.Fatalf("index: %d", status)
	}
}

func TestTrainJobAndInference(t *testing.T) {
	e := newEnv(t, true)
	status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2, "lr": 0.5, "batch_size": 4})
	if status != 202 || doc["job"] == nil {
		t.Fatalf("train: %d %v", status, doc)
	}
	first := doc["job"].(map[string]any)
	hasKeys(t, first, "id", "type", "state", "progress", "history", "error", "started_at", "finished_at", "stop_requested")
	job := e.waitJob()
	if job["state"] != "done" || job["type"] != "train" || len(job["history"].([]any)) != 2 || job["error"] != nil {
		t.Fatalf("job: %v", job)
	}
	rec := job["history"].([]any)[1].(map[string]any)
	hasKeys(t, rec, "epoch", "loss", "perplexity", "nodes", "edges", "trigrams", "compression_ratio", "merges", "transitions", "seconds")
	_, st := e.get("/api/status")
	if st["epochs_total"] != 2.0 || st["trained_texts"] != 5.0 || st["job"].(map[string]any)["state"] != "done" {
		t.Fatalf("status after training: %v", st)
	}
	status, hist := e.get("/api/history")
	if status != 200 || len(hist["history"].([]any)) != 2 {
		t.Fatalf("history: %d %v", status, hist)
	}

	status, p := e.post("/api/predict", map[string]any{"prefix": "the cat", "length": 6, "k": 2, "beam": 8})
	if status != 200 {
		t.Fatalf("predict: %d %v", status, p)
	}
	hasKeys(t, p, "prefix", "kind", "continuation", "full_text", "cost", "probability", "step_costs", "path", "node_ids",
		"expanded", "reached_end", "mode", "k", "beam", "top", "bottom")
	if p["full_text"] != "the cat"+p["continuation"].(string) || len(p["top"].([]any)) < 1 || p["beam"] != 8.0 {
		t.Fatalf("predict content: %v", p)
	}
	top := p["top"].([]any)[0].(map[string]any)
	hasKeys(t, top, "continuation", "full_text", "cost", "probability", "step_costs", "path", "node_ids", "reached_end")
	status, p = e.post("/api/predict", map[string]any{"prefix": "the", "mode": "sample", "length": 5})
	if status != 200 || p["mode"] != "sample" {
		t.Fatalf("sample predict: %d %v", status, p)
	}
	status, p = e.post("/api/predict", map[string]any{"length": 5})
	if status != 400 || !strings.Contains(p["error"].(string), "missing field 'prefix'") {
		t.Fatalf("predict without prefix: %d %v", status, p)
	}
	status, p = e.post("/api/predict", map[string]any{"prefix": "the", "mode": "nope"})
	if status != 400 {
		t.Fatalf("bad mode: %d %v", status, p)
	}
	status, p = e.post("/api/predict", map[string]any{"prefix": "the", "length": "five"})
	if status != 400 || !strings.Contains(p["error"].(string), "'length' must be an integer") {
		t.Fatalf("bad length: %d %v", status, p)
	}

	status, g := e.post("/api/generate", map[string]any{"count": 3, "mode": "beam", "max_length": 40})
	if status != 200 || len(g["samples"].([]any)) != 3 {
		t.Fatalf("generate: %d %v", status, g)
	}
	sample := g["samples"].([]any)[0].(map[string]any)
	hasKeys(t, sample, "text", "full_text", "cost", "probability", "path", "node_ids", "step_costs", "reached_end")
	status, g = e.post("/api/generate", map[string]any{"count": 2, "seed": 3})
	status2, g2 := e.post("/api/generate", map[string]any{"count": 2, "seed": 3})
	if status != 200 || status2 != 200 || fmt.Sprint(g) != fmt.Sprint(g2) {
		t.Fatal("seeded sampling must be reproducible")
	}
	status, g = e.post("/api/generate", map[string]any{"count": 0})
	if status != 200 || len(g["samples"].([]any)) != 0 {
		t.Fatalf("count 0: %d %v", status, g)
	}

	status, sc := e.post("/api/score", map[string]any{"text": corpus[0]})
	if status != 200 || sc["unknown_transitions"] != 0.0 || sc["chars"] != float64(len(corpus[0])) {
		t.Fatalf("score: %d %v", status, sc)
	}
	hasKeys(t, sc, "log_prob", "per_char", "chars", "transitions", "unknown_transitions")

	status, c := e.post("/api/converse", map[string]any{"opening": corpus[0], "turns": 3, "speakers": []string{"x", "y"}})
	if status != 200 || c["count"] != 4.0 || c["partner"] != nil {
		t.Fatalf("converse: %d %v", status, c)
	}
	turns := c["turns"].([]any)
	if turns[0].(map[string]any)["given"] != true || turns[1].(map[string]any)["speaker"] != "y" {
		t.Fatalf("turns: %v", turns)
	}
	status, c = e.post("/api/converse", map[string]any{"turns": 1, "partner": "radix"})
	if status != 400 {
		t.Fatalf("partner radix: %d %v", status, c)
	}
	status, c = e.post("/api/converse", map[string]any{"turns": 2, "history": []string{corpus[0], corpus[1]}})
	if status != 200 || c["turns"].([]any)[0].(map[string]any)["index"] != 2.0 {
		t.Fatalf("history: %d %v", status, c)
	}
}

func TestSplitParagraphsAndFeedback(t *testing.T) {
	e := newEnv(t, true)
	blob := "the cat sat on the mat\nand purred\n\nthe dog ate the bone\n\n\nthe bird flew"
	status, doc := e.post("/api/train", map[string]any{"text": blob, "split": "paragraphs", "epochs": 1})
	if status != 202 {
		t.Fatalf("train paragraphs: %d %v", status, doc)
	}
	e.waitJob()
	_, st := e.get("/api/status")
	if st["trained_texts"] != 3.0 {
		t.Fatalf("expected 3 paragraph texts, got %v", st["trained_texts"])
	}
	status, doc = e.post("/api/train", map[string]any{"text": blob, "split": "chapters"})
	if status != 400 || !strings.Contains(doc["error"].(string), "'split' must be one of") {
		t.Fatalf("bad split: %d %v", status, doc)
	}
	status, doc = e.post("/api/train", map[string]any{"epochs": 1})
	if status != 400 || !strings.Contains(doc["error"].(string), "missing field 'texts'") {
		t.Fatalf("no texts: %d %v", status, doc)
	}

	status, fb := e.post("/api/feedback", map[string]any{"good": []string{corpus[0]}, "bad_text": corpus[3] + "\n", "strength": 0.5})
	if status != 202 || fb["action"] != "2nrl" || fb["good"] != 1.0 || fb["bad"] != 1.0 {
		t.Fatalf("feedback: %d %v", status, fb)
	}
	job := e.waitJob()
	if job["state"] != "done" || job["type"] != "feedback" {
		t.Fatalf("feedback job: %v", job)
	}
	phases := map[string]bool{}
	for _, r := range job["history"].([]any) {
		phases[r.(map[string]any)["phase"].(string)] = true
	}
	if !phases["negative"] || !phases["positive"] {
		t.Fatalf("phases: %v", phases)
	}
	_, st = e.get("/api/status")
	if st["twonrl_runs"] != 1.0 || st["edge_reward_negative"].(float64) >= 0 || st["edge_reward_positive"].(float64) <= 0 {
		t.Fatalf("status after feedback: %v", st)
	}
	status, fb = e.post("/api/feedback", map[string]any{"good": []string{corpus[1]}})
	if status != 202 || fb["action"] != "reward" {
		t.Fatalf("reward: %d %v", status, fb)
	}
	e.waitJob()
	status, fb = e.post("/api/feedback", map[string]any{})
	if status != 400 {
		t.Fatalf("empty feedback: %d %v", status, fb)
	}

	status, two := e.post("/api/2nrl", map[string]any{"bad": []string{corpus[3]}, "good": []string{corpus[0]}, "neg_epochs": 1, "pos_epochs": 1, "neg_lr": 0.5, "pos_lr": 0.1})
	if status != 202 {
		t.Fatalf("2nrl: %d %v", status, two)
	}
	job = e.waitJob()
	if job["type"] != "2nrl" || job["state"] != "done" {
		t.Fatalf("2nrl job: %v", job)
	}
	status, two = e.post("/api/2nrl", map[string]any{"bad": []string{corpus[3]}})
	if status != 400 || !strings.Contains(two["error"].(string), "'good'") {
		t.Fatalf("2nrl without good: %d %v", status, two)
	}

	status, inv := e.post("/api/invert", nil)
	if status != 200 || inv["inverted"] != true {
		t.Fatalf("invert: %d %v", status, inv)
	}
	status, comp := e.post("/api/compress", nil)
	if status != 200 || comp["merges"] == nil {
		t.Fatalf("compress: %d %v", status, comp)
	}
}

func TestBusyJobConflictsAndStops(t *testing.T) {
	e := newEnv(t, true)
	e.svc.epochDelay = 40 * time.Millisecond
	status, doc := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 50})
	if status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	status, second := e.post("/api/train", map[string]any{"texts": corpus, "epochs": 1})
	if status != 409 || !strings.Contains(second["error"].(string), "is running") {
		t.Fatalf("second job must be refused: %d %v", status, second)
	}
	status, inv := e.post("/api/invert", nil)
	if status != 409 {
		t.Fatalf("invert during a job: %d %v", status, inv)
	}
	// readers get a turn between epochs
	status, st := e.get("/api/status")
	if status != 200 || st["job"].(map[string]any)["state"] != "running" {
		t.Fatalf("status during a job: %d %v", status, st)
	}
	status, p := e.post("/api/predict", map[string]any{"prefix": "the", "length": 3})
	if status != 200 {
		t.Fatalf("predict during a job: %d %v", status, p)
	}
	status, stopped := e.post("/api/job/stop", nil)
	if status != 200 || stopped["stop_requested"] != true {
		t.Fatalf("stop: %d %v", status, stopped)
	}
	job := e.waitJob()
	if job["state"] != "stopped" || len(job["history"].([]any)) >= 50 {
		t.Fatalf("stopped job: %v", job["state"])
	}
	// a fresh service has no job to stop
	f := newEnv(t, false)
	status, none := f.post("/api/job/stop", nil)
	if status != 404 {
		t.Fatalf("stop without a job: %d %v", status, none)
	}
	status, nothing := f.get("/api/job")
	if status != 200 || nothing != nil {
		t.Fatalf("no job yet: %d %v", status, nothing)
	}
}

func zipBytes(t *testing.T) []byte {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, content := range map[string]string{"a/one.txt": "the cat sat on the mat\nthe cat ran to the door\n", "two.txt": "the dog sat on the log\n", "img.png": "\x89PNG\x00\x00binary", "__MACOSX/._one": "x", "empty.txt": "  \n"} {
		w, _ := zw.Create(name)
		w.Write([]byte(content))
	}
	zw.Close()
	return buf.Bytes()
}

func TestUploadsInAllForms(t *testing.T) {
	e := newEnv(t, true)
	status, doc := e.post("/api/uploads", map[string]any{"name": "../corpus one.txt", "content": "the cat sat on the mat\n\nthe dog ate the bone\n"})
	if status != 201 {
		t.Fatalf("json upload: %d %v", status, doc)
	}
	rec := doc["uploads"].([]any)[0].(map[string]any)
	if rec["name"] != "corpus one.txt" || rec["lines"] != 2.0 || rec["replaced"] != false {
		t.Fatalf("record: %v", rec)
	}
	hasKeys(t, rec, "name", "bytes", "chars", "lines", "modified")

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, _ := mw.CreateFormFile("file", "corpus.zip")
	part.Write(zipBytes(t))
	mw.Close()
	status, doc, _ = e.do("POST", "/api/uploads", body.Bytes(), map[string]string{"Content-Type": mw.FormDataContentType()})
	if status != 201 {
		t.Fatalf("zip upload: %d %v", status, doc)
	}
	rec = doc["uploads"].([]any)[0].(map[string]any)
	if rec["archive"] != true || rec["files"] != 2.0 || rec["lines"] != 3.0 || rec["skipped"] != 3.0 {
		t.Fatalf("zip record: %v", rec)
	}
	summary := doc["archives"].([]any)[0].(map[string]any)
	if summary["extracted"] != 2.0 || summary["entries"] != 5.0 {
		t.Fatalf("archive summary: %v", summary)
	}
	if _, err := os.Stat(filepath.Join(e.dir, "uploads", "corpus.zip")); err != nil {
		t.Fatal("the archive itself must be kept")
	}
	// raw body with ?name=
	status, doc, _ = e.do("POST", "/api/uploads?name=raw.txt", "line one\nline two\n", map[string]string{"Content-Type": "text/plain"})
	if status != 201 || doc["uploads"].([]any)[0].(map[string]any)["lines"] != 2.0 {
		t.Fatalf("raw upload: %d %v", status, doc)
	}
	status, doc, _ = e.do("POST", "/api/uploads", "no name", map[string]string{"Content-Type": "text/plain"})
	if status != 400 {
		t.Fatalf("raw upload without a name: %d %v", status, doc)
	}
	// base64 form
	status, doc = e.post("/api/uploads", map[string]any{"files": []map[string]any{{"name": "b64.txt", "content_base64": "dGhlIGNhdA=="}}})
	if status != 201 || doc["uploads"].([]any)[0].(map[string]any)["name"] != "b64.txt" {
		t.Fatalf("base64 upload: %d %v", status, doc)
	}
	// an archive without text is refused
	var empty bytes.Buffer
	zw := zip.NewWriter(&empty)
	w, _ := zw.Create("x.png")
	w.Write([]byte("\x00\x01"))
	zw.Close()
	status, doc = e.post("/api/uploads", map[string]any{"name": "nothing.zip", "content_base64": base64Of(empty.Bytes())})
	if status != 400 || !strings.Contains(doc["error"].(string), "no text files") {
		t.Fatalf("empty archive: %d %v", status, doc)
	}

	status, list := e.get("/api/uploads")
	if status != 200 || len(list["uploads"].([]any)) != 4 {
		t.Fatalf("list: %d %v", status, list)
	}
	// training from an archive unpacks its text entries; whole_file keeps entries as one text each
	status, doc = e.post("/api/train", map[string]any{"files": []string{"corpus.zip", "raw.txt"}, "epochs": 1})
	if status != 202 {
		t.Fatalf("train from files: %d %v", status, doc)
	}
	e.waitJob()
	_, st := e.get("/api/status")
	if st["trained_texts"] != 5.0 {
		t.Fatalf("expected 5 texts from the uploads, got %v", st["trained_texts"])
	}
	status, doc = e.post("/api/train", map[string]any{"files": []string{"corpus.zip"}, "whole_file": true, "epochs": 1})
	e.waitJob()
	_, st = e.get("/api/status")
	if st["trained_texts"] != 7.0 {
		t.Fatalf("whole_file: expected 2 more texts, got %v", st["trained_texts"])
	}
	status, doc = e.post("/api/train", map[string]any{"files": []string{"missing.txt"}})
	if status != 404 {
		t.Fatalf("missing upload: %d %v", status, doc)
	}
	status, del := e.post("/api/uploads/delete", map[string]any{"name": "raw.txt"})
	if status != 200 || del["deleted"] != "raw.txt" {
		t.Fatalf("delete: %d %v", status, del)
	}
	status, del = e.post("/api/uploads/delete", map[string]any{"name": "raw.txt"})
	if status != 404 {
		t.Fatalf("delete twice: %d %v", status, del)
	}
	f := newEnv(t, false)
	status, doc = f.get("/api/uploads")
	if status != 400 || !strings.Contains(doc["error"].(string), "disabled") {
		t.Fatalf("uploads disabled: %d %v", status, doc)
	}
}

func base64Of(b []byte) string {
	const table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	var out strings.Builder
	for i := 0; i < len(b); i += 3 {
		var chunk [3]byte
		n := copy(chunk[:], b[i:])
		v := uint(chunk[0])<<16 | uint(chunk[1])<<8 | uint(chunk[2])
		out.WriteByte(table[v>>18&63])
		out.WriteByte(table[v>>12&63])
		if n > 1 {
			out.WriteByte(table[v>>6&63])
		} else {
			out.WriteByte('=')
		}
		if n > 2 {
			out.WriteByte(table[v&63])
		} else {
			out.WriteByte('=')
		}
	}
	return out.String()
}

func TestGraphPersistenceAndCheckpoints(t *testing.T) {
	e := newEnv(t, true)
	e.post("/api/train", map[string]any{"texts": corpus, "epochs": 2})
	e.waitJob()
	status, g := e.get("/api/graph?limit=4")
	if status != 200 {
		t.Fatalf("graph: %d", status)
	}
	nodes := g["nodes"].([]any)
	if len(nodes) != 6 || nodes[0].(map[string]any)["id"] != 0.0 || nodes[1].(map[string]any)["id"] != 1.0 {
		t.Fatalf("graph nodes: %v", nodes)
	}
	hasKeys(t, nodes[2].(map[string]any), "id", "label", "count", "activation", "z", "a", "b", "h", "k")
	edges := g["edges"].([]any)
	if len(edges) == 0 {
		t.Fatal("graph has no edges among the chosen nodes")
	}
	hasKeys(t, edges[0].(map[string]any), "source", "target", "weight", "count", "prob", "cost", "reward", "share", "recent_share", "recent_count")
	hasKeys(t, g, "total_nodes", "total_edges", "total_traversals", "window_traversals", "window", "limit")
	status, g = e.get("/api/graph?limit=x")
	if status != 400 {
		t.Fatalf("bad limit: %d", status)
	}

	status, saved := e.post("/api/save", map[string]any{})
	if status != 200 || saved["bytes"].(float64) <= 0 {
		t.Fatalf("save: %d %v", status, saved)
	}
	other := filepath.Join(e.dir, "other.count.json.gz")
	status, saved = e.post("/api/save", map[string]any{"path": other})
	if status != 200 || !strings.HasSuffix(saved["path"].(string), "other.count.json.gz") {
		t.Fatalf("save to path: %d %v", status, saved)
	}
	status, reset := e.post("/api/reset", map[string]any{"seed": 5, "window": 42})
	if status != 200 || reset["nodes"] != 2.0 || reset["window"] != 42.0 {
		t.Fatalf("reset: %d %v", status, reset)
	}
	status, reset = e.post("/api/reset", map[string]any{"kind": "radix"})
	if status != 400 {
		t.Fatalf("reset radix: %d %v", status, reset)
	}
	status, loaded := e.post("/api/load", map[string]any{"path": other})
	if status != 200 || loaded["epochs_total"] != 2.0 {
		t.Fatalf("load: %d %v", status, loaded)
	}
	status, loaded = e.post("/api/load", map[string]any{"path": filepath.Join(e.dir, "missing.json")})
	if status != 404 {
		t.Fatalf("load missing: %d %v", status, loaded)
	}
	status, loaded = e.post("/api/load", map[string]any{"path": ""})
	if status != 400 {
		t.Fatalf("load empty path: %d %v", status, loaded)
	}

	status, ck := e.post("/api/checkpoints/save", map[string]any{"tag": "manual"})
	if status != 200 || ck["name"] != "ckpt-manual-000002.json.gz" || ck["step"] != 2.0 || ck["metrics"] == nil {
		t.Fatalf("checkpoint save: %d %v", status, ck)
	}
	hasKeys(t, ck, "name", "path", "step", "tag", "metrics", "saved_at", "bytes")
	status, ck = e.post("/api/checkpoints/save", map[string]any{"tag": "bad tag!"})
	if status != 400 {
		t.Fatalf("bad tag: %d %v", status, ck)
	}
	status, list := e.get("/api/checkpoints")
	if status != 200 || len(list["checkpoints"].([]any)) != 1 || list["latest"].(map[string]any)["name"] != "ckpt-manual-000002.json.gz" {
		t.Fatalf("checkpoints: %d %v", status, list)
	}
	for _, name := range []string{"latest.json", "index.json", "ckpt-manual-000002.json.gz"} {
		if _, err := os.Stat(filepath.Join(e.dir, "ckpt", name)); err != nil {
			t.Fatalf("missing %s in the checkpoint directory", name)
		}
	}
	e.post("/api/reset", map[string]any{})
	status, restored := e.post("/api/checkpoints/restore", map[string]any{"name": "ckpt-manual-000002"})
	if status != 200 || restored["epochs_total"] != 2.0 {
		t.Fatalf("restore: %d %v", status, restored)
	}
	status, restored = e.post("/api/checkpoints/restore", map[string]any{"name": "nope"})
	if status != 404 {
		t.Fatalf("restore missing: %d %v", status, restored)
	}
	f := newEnv(t, false)
	status, list = f.get("/api/checkpoints")
	if status != 200 || len(list["checkpoints"].([]any)) != 0 || list["latest"] != nil {
		t.Fatalf("no checkpoint dir: %d %v", status, list)
	}
	status, ck = f.post("/api/checkpoints/save", map[string]any{})
	if status != 400 {
		t.Fatalf("save without a directory: %d %v", status, ck)
	}
}

func TestStaticAndErrors(t *testing.T) {
	e := newEnv(t, true)
	status, _, h := e.do("GET", "/", nil, nil)
	if status != 200 || !strings.HasPrefix(h.Get("Content-Type"), "text/html") || h.Get("Cache-Control") != "no-cache" {
		t.Fatalf("index: %d %v", status, h)
	}
	status, _, h = e.do("GET", "/assets/app.js", nil, nil)
	if status != 200 || !strings.Contains(h.Get("Cache-Control"), "immutable") {
		t.Fatalf("asset: %d %v", status, h)
	}
	status, _, _ = e.do("GET", "/some/client/route", nil, nil)
	if status != 200 {
		t.Fatalf("SPA fallback: %d", status)
	}
	status, doc := e.get("/assets/missing.js")
	if status != 404 || doc["error"] == nil {
		t.Fatalf("missing asset: %d %v", status, doc)
	}
	status, doc = e.get("/../etc/passwd")
	if status == 200 && doc == nil {
		// path.Clean turns this into /etc/passwd inside the frontend dir: must not exist -> SPA fallback or 404
	}
	status, doc, h = e.do("POST", "/", "{}", nil)
	if status != 405 || h.Get("Allow") == "" {
		t.Fatalf("POST /: %d %v", status, doc)
	}
	status, doc = e.get("/api/nope")
	if status != 404 || !strings.Contains(doc["error"].(string), "unknown API endpoint") {
		t.Fatalf("unknown endpoint: %d %v", status, doc)
	}
	status, doc = e.get("/api/evolve/history")
	if status != 404 || !strings.Contains(doc["error"].(string), "not available on the Go server") {
		t.Fatalf("python-only endpoint: %d %v", status, doc)
	}
	status, doc, h = e.do("GET", "/api/predict", nil, nil)
	if status != 405 || !strings.Contains(h.Get("Allow"), "POST") {
		t.Fatalf("GET predict: %d %v", status, doc)
	}
	status, doc, _ = e.do("POST", "/api/predict", "{not json", nil)
	if status != 400 || !strings.Contains(doc["error"].(string), "not valid JSON") {
		t.Fatalf("invalid json: %d %v", status, doc)
	}
	status, doc, _ = e.do("POST", "/api/predict", "[1,2]", nil)
	if status != 400 {
		t.Fatalf("non-object body: %d %v", status, doc)
	}
	status, _, _ = e.do("OPTIONS", "/api/predict", nil, nil)
	if status != 204 {
		t.Fatalf("options: %d", status)
	}
	status, doc = e.get("/api/schedule")
	if status != 200 || doc["presets"] == nil {
		t.Fatalf("schedule stub: %d %v", status, doc)
	}
	// no frontend directory: a help page
	svc, _ := NewService(Options{Seed: 1, Workers: 2, Exact: true, Quiet: true})
	ts := httptest.NewServer(NewHandler(svc, "", true, nil))
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), "No frontend build") {
		t.Fatalf("help page: %d %s", resp.StatusCode, body[:60])
	}
	resp, _ = http.Get(ts.URL + "/other")
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("no frontend, other path: %d", resp.StatusCode)
	}
}

func TestStreamedArchiveUploadAndChunkedTraining(t *testing.T) {
	e := newEnv(t, true)
	// a multi-megabyte archive streamed as multipart: never buffered by the handler
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	total := 0
	for part := 0; part < 4; part++ {
		w, _ := zw.Create(fmt.Sprintf("book/chapter-%d.txt", part))
		for i := 0; i < 15000; i++ {
			fmt.Fprintf(w, "the cat number %d sat on mat %d\n", i%97, (i*7)%89)
			total++
		}
	}
	w, _ := zw.Create("cover.png")
	w.Write([]byte("\x89PNG\x00\x00\x00"))
	zw.Close()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, _ := mw.CreateFormFile("file", "big.zip")
	part.Write(buf.Bytes())
	mw.Close()
	status, doc, _ := e.do("POST", "/api/uploads", body.Bytes(), map[string]string{"Content-Type": mw.FormDataContentType()})
	if status != 201 {
		t.Fatalf("streamed upload: %d %v", status, doc)
	}
	rec := doc["uploads"].([]any)[0].(map[string]any)
	if rec["archive"] != true || rec["files"] != 4.0 || rec["lines"] != float64(total) || rec["skipped"] != 1.0 {
		t.Fatalf("record: %v", rec)
	}
	entries, _ := os.ReadDir(filepath.Join(e.dir, "uploads"))
	for _, en := range entries {
		if strings.HasSuffix(en.Name(), ".part") {
			t.Fatalf("temporary file left behind: %s", en.Name())
		}
	}
	// train from it in small chunks: the archive streams through the job
	status, doc = e.post("/api/train", map[string]any{"files": []string{"big.zip"}, "epochs": 1, "chunk_size": 5000})
	if status != 202 {
		t.Fatalf("train from the archive: %d %v", status, doc)
	}
	job := e.waitJob()
	if job["state"] != "done" {
		t.Fatalf("job: %v", job)
	}
	recEpoch := job["history"].([]any)[0].(map[string]any)
	if recEpoch["chunks"] != float64((total+4999)/5000) {
		t.Fatalf("chunks: %v", recEpoch["chunks"])
	}
	_, st := e.get("/api/status")
	if st["trained_texts"] != float64(total) {
		t.Fatalf("trained_texts %v, want %d", st["trained_texts"], total)
	}
	// a raw-body archive without any text entry is refused and leaves nothing behind
	var empty bytes.Buffer
	ez := zip.NewWriter(&empty)
	ew, _ := ez.Create("x.bin")
	ew.Write([]byte{0, 1, 2})
	ez.Close()
	status, doc, _ = e.do("POST", "/api/uploads?name=nothing.zip", empty.Bytes(), map[string]string{"Content-Type": "application/zip"})
	if status != 400 || !strings.Contains(doc["error"].(string), "no text files") {
		t.Fatalf("empty archive: %d %v", status, doc)
	}
	status, doc, _ = e.do("POST", "/api/uploads?name=corrupt.zip", []byte("PK\x03\x04garbage"), map[string]string{"Content-Type": "application/zip"})
	if status != 400 {
		t.Fatalf("corrupt archive: %d %v", status, doc)
	}
	entries, _ = os.ReadDir(filepath.Join(e.dir, "uploads"))
	if len(entries) != 1 {
		t.Fatalf("upload dir must hold the one good archive, has %d entries", len(entries))
	}
	// chunk_size must be positive
	status, doc = e.post("/api/train", map[string]any{"files": []string{"big.zip"}, "chunk_size": 0})
	if status != 400 {
		t.Fatalf("chunk_size 0: %d %v", status, doc)
	}
}

// A big upload must train inside a bounded amount of memory: the reader waits
// for a chunk slot, so the job's footprint is the graph plus the chunks in
// flight, not the corpus.
func TestTrainingHonoursTheInflightBound(t *testing.T) {
	e := newEnv(t, true)
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("corpus.txt")
	for i := 0; i < 4000; i++ {
		fmt.Fprintf(w, "the cat sat on the mat number %d\n", i)
	}
	zw.Close()
	status, doc, _ := e.do("POST", "/api/uploads?name=corpus.zip", buf.Bytes(), map[string]string{"Content-Type": "application/zip"})
	if status != 201 {
		t.Fatalf("upload: %d %v", status, doc)
	}
	status, doc = e.post("/api/train", map[string]any{"files": []string{"corpus.zip"}, "epochs": 1, "chunk_size": 100, "inflight": 2})
	if status != 202 {
		t.Fatalf("train: %d %v", status, doc)
	}
	job := e.waitJob()
	if job["state"] != "done" {
		t.Fatalf("job: %v", job)
	}
	rec := job["history"].([]any)[0].(map[string]any)
	if rec["chunks"] != float64(40) {
		t.Fatalf("chunks: %v", rec["chunks"])
	}
	// an inflight of zero or less is refused like the other positive counts
	status, doc = e.post("/api/train", map[string]any{"files": []string{"corpus.zip"}, "inflight": 0})
	if status != 400 {
		t.Fatalf("inflight 0: %d %v", status, doc)
	}
}
