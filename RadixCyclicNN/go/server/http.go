package server

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// MaxBodyBytes caps JSON request bodies (uploads are not limited).
const MaxBodyBytes = 64 << 20

// fields reads typed values out of a JSON body with the Python server's messages.
type fields struct {
	body map[string]any
}

var errMissing = errors.New("missing")

func (f fields) lookup(name string) (any, bool) {
	if f.body == nil {
		return nil, false
	}
	v, ok := f.body[name]
	return v, ok
}

func (f fields) present(name string) bool { _, ok := f.lookup(name); return ok }

func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return "value"
}

func (f fields) text(name string, def *string) (string, error) {
	v, ok := f.lookup(name)
	if !ok || v == nil && def != nil {
		if def == nil {
			return "", badRequest("missing field '%s' (string)", name)
		}
		return *def, nil
	}
	s, ok := v.(string)
	if !ok {
		return "", badRequest("'%s' must be a string (got %s %v)", name, jsonType(v), v)
	}
	return s, nil
}

func (f fields) optText(name, def string) (string, error) { return f.text(name, &def) }

func (f fields) flag(name string, def bool) (bool, error) {
	v, ok := f.lookup(name)
	if !ok {
		return def, nil
	}
	b, ok := v.(bool)
	if !ok {
		return false, badRequest("'%s' must be a boolean (got %s %v)", name, jsonType(v), v)
	}
	return b, nil
}

// integer returns (value, present, error); def is used when the field is absent or null.
func (f fields) integer(name string, def int, minimum *int) (int, bool, error) {
	v, ok := f.lookup(name)
	if !ok || v == nil {
		return def, false, nil
	}
	n, isNum := v.(float64)
	if !isNum || n != math.Trunc(n) || math.IsInf(n, 0) {
		return 0, true, badRequest("'%s' must be an integer (got %s %v)", name, jsonType(v), v)
	}
	if minimum != nil && int(n) < *minimum {
		return 0, true, badRequest("'%s' must be >= %d (got %v)", name, *minimum, int(n))
	}
	return int(n), true, nil
}

func (f fields) number(name string, def float64, minimum *float64) (float64, bool, error) {
	v, ok := f.lookup(name)
	if !ok || v == nil {
		return def, false, nil
	}
	n, isNum := v.(float64)
	if !isNum || math.IsNaN(n) || math.IsInf(n, 0) {
		return 0, true, badRequest("'%s' must be a finite number (got %s %v)", name, jsonType(v), v)
	}
	if minimum != nil && n < *minimum {
		return 0, true, badRequest("'%s' must be >= %v (got %v)", name, *minimum, n)
	}
	return n, true, nil
}

// textsOptional: listName (list of strings) or textName (one text per line); absent -> nil.
func (f fields) textsOptional(listName, textName string) ([]string, error) {
	if v, ok := f.lookup(listName); ok {
		items, isList := v.([]any)
		if !isList {
			return nil, badRequest("'%s' must be a list of strings (got %s %v)", listName, jsonType(v), v)
		}
		out := make([]string, 0, len(items))
		for _, item := range items {
			s, isStr := item.(string)
			if !isStr {
				return nil, badRequest("'%s' must be a list of strings (got %s %v)", listName, jsonType(v), v)
			}
			out = append(out, s)
		}
		return out, nil
	}
	if v, ok := f.lookup(textName); ok {
		blob, isStr := v.(string)
		if !isStr {
			return nil, badRequest("'%s' must be a string (got %s %v)", textName, jsonType(v), v)
		}
		return radixnet.SplitTexts(blob, "lines", 0), nil
	}
	return nil, nil
}

// names: an optional list of non-empty strings.
func (f fields) names(name string) ([]string, error) {
	v, ok := f.lookup(name)
	if !ok || v == nil {
		return nil, nil
	}
	items, isList := v.([]any)
	if !isList {
		return nil, badRequest("'%s' must be a list of non-empty strings (got %s %v)", name, jsonType(v), v)
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		s, isStr := item.(string)
		if !isStr || strings.TrimSpace(s) == "" {
			return nil, badRequest("'%s' must be a list of non-empty strings (got %s %v)", name, jsonType(v), v)
		}
		out = append(out, s)
	}
	return out, nil
}

func intp(v int) *int           { return &v }
func floatp(v float64) *float64 { return &v }

// request bundles what a route handler gets.
type request struct {
	svc    *Service
	f      fields
	query  map[string][]string
	body   []byte
	header http.Header
}

func (rq *request) queryValue(name string) (string, bool) {
	vals := rq.query[name]
	if len(vals) == 0 {
		return "", false
	}
	return vals[len(vals)-1], true
}

type routeFn func(rq *request) (int, any, error)

var routes = map[string]map[string]routeFn{}

func route(method, p string, fn routeFn) {
	if routes[p] == nil {
		routes[p] = map[string]routeFn{}
	}
	routes[p][method] = fn
}

var endpointDocs = []map[string]string{}

func doc(method, p, text string) {
	endpointDocs = append(endpointDocs, map[string]string{"method": method, "path": p, "doc": text})
}

func init() {
	route("GET", "/api", rIndex)
	doc("GET", "/api", "this list")
	route("GET", "/api/health", rHealth)
	doc("GET", "/api/health", "liveness: {ok, version, engine: go, workers}")
	route("GET", "/api/status", rStatus)
	doc("GET", "/api/status", "model statistics, the current job, the engine and its goroutine pool")
	route("GET", "/api/model", rModel)
	doc("GET", "/api/model", "the model kind this server runs (count) and its weight function")
	route("POST", "/api/model/select", rModelSelect)
	doc("POST", "/api/model/select", "{kind: count}: the Go server runs the count / reward model only")
	route("POST", "/api/model/weights", rModelWeights)
	doc("POST", "/api/model/weights", "change the dual frequency weight function: {count_scale, global_scale, window_scale, reward_scale, window}")
	route("POST", "/api/train", rTrain)
	doc("POST", "/api/train", "start a training job: {texts | text | files, whole_file, split: lines | paragraphs | pages | file, page_lines, epochs, auto_compress, chunk_size, inflight, parallel_parts}; uploads stream through in chunks, whatever their size")
	route("GET", "/api/job", rJob)
	doc("GET", "/api/job", "status of the current / last job")
	route("POST", "/api/job/stop", rJobStop)
	doc("POST", "/api/job/stop", "ask the running job to stop")
	route("POST", "/api/predict", rPredict)
	doc("POST", "/api/predict", "continue a prefix: {prefix, length, mode: beam | sample, to_end, step_penalty, temperature, max_length, k, beam}")
	route("POST", "/api/generate", rGenerate)
	doc("POST", "/api/generate", "whole texts: {count, max_length, mode: beam | sample | dijkstra, temperature, seed, prefix, step_penalty, beam}")
	route("POST", "/api/converse", rConverse)
	doc("POST", "/api/converse", "the model converses with itself: {opening, turns, mode, max_length, context, temperature, k, beam, step_penalty, seed, speakers, history, avoid_repeats}")
	route("POST", "/api/score", rScore)
	doc("POST", "/api/score", "log-probability of a text: {text}")
	route("POST", "/api/2nrl", rTwoNRL)
	doc("POST", "/api/2nrl", "start a 2NRL job: {bad | bad_text | bad_files, good | good_text | good_files, neg_epochs, pos_epochs, strength, bad_weights | bad_ratings, good_weights | good_ratings (per text: how bad / how good)}")
	route("POST", "/api/feedback", rFeedback)
	doc("POST", "/api/feedback", "rated texts: {good | good_text | good_files, bad | bad_text | bad_files, good_ratings / bad_ratings (marks out of 10, or good_weights / bad_weights as 0..1 shares), neg_epochs, pos_epochs, strength} -> a 2nrl / reward / punish job; every text is learned in proportion to its rating")
	route("POST", "/api/invert", rInvert)
	doc("POST", "/api/invert", "flip the sign of every reward")
	route("POST", "/api/compress", rCompress)
	doc("POST", "/api/compress", "merge unary chains")
	route("POST", "/api/save", rSave)
	doc("POST", "/api/save", "{path}: write the model (default: the server's model file)")
	route("POST", "/api/load", rLoad)
	doc("POST", "/api/load", "{path}: replace the model with a file")
	route("POST", "/api/reset", rReset)
	doc("POST", "/api/reset", "{seed, count_scale, global_scale, window_scale, reward_scale, window}: a fresh model")
	route("GET", "/api/checkpoints", rCheckpoints)
	doc("GET", "/api/checkpoints", "checkpoint records and the latest one")
	route("POST", "/api/checkpoints/save", rCheckpointSave)
	doc("POST", "/api/checkpoints/save", "{tag}: checkpoint the model")
	route("POST", "/api/checkpoints/restore", rCheckpointRestore)
	doc("POST", "/api/checkpoints/restore", "{name}: load a checkpoint")
	route("GET", "/api/graph", rGraph)
	doc("GET", "/api/graph", "?limit=150: the most visited nodes and the edges among them, with counts, shares and rewards")
	route("GET", "/api/history", rHistory)
	doc("GET", "/api/history", "training history")
	route("GET", "/api/uploads", rUploads)
	doc("GET", "/api/uploads", "uploaded training files (a ZIP counts the lines of its text entries)")
	route("POST", "/api/uploads", rUpload)
	doc("POST", "/api/uploads", "upload text files or ZIP archives: JSON {name, content | content_base64} / {files: [...]}, multipart/form-data, or a raw body with ?name=")
	route("POST", "/api/uploads/delete", rUploadDelete)
	doc("POST", "/api/uploads/delete", "{name}")
	route("GET", "/api/schedule", rSchedule)
	doc("GET", "/api/schedule", "learning-rate schedules do not apply to the count model: an empty description")
}

// pythonOnly lists endpoint prefixes the Python server implements and this one does not.
var pythonOnly = []string{"/api/evolve", "/api/ollama", "/api/images", "/api/codegen", "/api/schedule/preview"}

// The tutor (/api/tutor, see tutor.go) is served here too: Ollama sets and
// marks the exercises, this server's count / reward model answers them.

func rIndex(rq *request) (int, any, error) {
	return 200, map[string]any{"engine": "go", "version": Version, "endpoints": endpointDocs}, nil
}

func rHealth(rq *request) (int, any, error) {
	return 200, map[string]any{"ok": true, "version": Version, "engine": "go", "workers": rq.svc.workers, "counting": map[bool]string{true: "exact", false: "racy"}[rq.svc.exact]}, nil
}

func rStatus(rq *request) (int, any, error) {
	out, err := rq.svc.Status()
	return 200, out, err
}

func rModel(rq *request) (int, any, error) {
	out, err := rq.svc.DescribeModel()
	return 200, out, err
}

func rModelSelect(rq *request) (int, any, error) {
	kind, err := rq.f.text("kind", nil)
	if err != nil {
		return 0, nil, err
	}
	if kind == "" {
		return 0, nil, badRequest("'kind' must not be empty")
	}
	out, err := rq.svc.SelectKind(kind)
	return 200, out, err
}

func weightOptions(f fields) (map[string]float64, error) {
	opts := map[string]float64{}
	for _, name := range []string{"count_scale", "global_scale", "window_scale", "reward_scale"} {
		v, present, err := f.number(name, 0, nil)
		if err != nil {
			return nil, err
		}
		if present {
			opts[name] = v
		}
	}
	w, present, err := f.integer("window", 0, intp(1))
	if err != nil {
		return nil, err
	}
	if present {
		opts["window"] = float64(w)
	}
	return opts, nil
}

func rModelWeights(rq *request) (int, any, error) {
	opts, err := weightOptions(rq.f)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.ConfigureWeights(opts)
	return 200, out, err
}

// textsAndFiles: inline texts and / or uploaded files (split per the unit); at least one text is required.
func textsAndFiles(rq *request, listName, textName, filesName, unit string, pageLines int, wholeFile bool) ([]string, error) {
	names, err := rq.f.names(filesName)
	if err != nil {
		return nil, err
	}
	var texts []string
	if v, ok := rq.f.lookup(textName); ok && unit != "" && unit != "lines" {
		blob, isStr := v.(string)
		if !isStr {
			return nil, badRequest("'%s' must be a string (got %s %v)", textName, jsonType(v), v)
		}
		texts = radixnet.SplitTexts(blob, unit, pageLines)
		if more, err := rq.f.textsOptional(listName, "\x00"); err != nil {
			return nil, err
		} else {
			texts = append(more, texts...)
		}
	} else {
		texts, err = rq.f.textsOptional(listName, textName)
		if err != nil {
			return nil, err
		}
	}
	if len(names) > 0 {
		more, err := rq.svc.UploadTexts(names, unit, pageLines, wholeFile)
		if err != nil {
			return nil, err
		}
		texts = append(texts, more...)
	}
	if len(texts) == 0 {
		if len(names) > 0 {
			return nil, badRequest("'%s' contains no texts", filesName)
		}
		if rq.f.present(listName) || rq.f.present(textName) {
			return nil, badRequest("'%s' contains no texts", listName)
		}
		return nil, badRequest("missing field '%s' (list of strings), '%s' (string, one text per line) or '%s' (list of upload names)", listName, textName, filesName)
	}
	return texts, nil
}

func splitOptions(f fields) (string, int, error) {
	unit, err := f.optText("split", "lines")
	if err != nil {
		return "", 0, err
	}
	unit = strings.ToLower(strings.TrimSpace(unit))
	switch unit {
	case "", "line", "lines":
		unit = "lines"
	case "paragraph", "paragraphs":
		unit = "paragraphs"
	case "page", "pages":
		unit = "pages"
	case "file", "whole", "whole-file":
		unit = "file"
	default:
		return "", 0, badRequest("'split' must be one of lines, paragraphs, pages, file (got %q)", unit)
	}
	pageLines, _, err := f.integer("page_lines", 50, intp(1))
	if err != nil {
		return "", 0, err
	}
	return unit, pageLines, nil
}

// trainSource is the streaming source of a training request: inline texts
// (already in memory) followed by the selected uploads, which stream from
// disk chunk by chunk however large they are.
func trainSource(rq *request, unit string, pageLines int, wholeFile bool) (radixnet.TextSource, error) {
	names, err := rq.f.names("files")
	if err != nil {
		return nil, err
	}
	var inline []string
	if v, ok := rq.f.lookup("text"); ok && unit != "lines" {
		blob, isStr := v.(string)
		if !isStr {
			return nil, badRequest("'text' must be a string (got %s %v)", jsonType(v), v)
		}
		inline = radixnet.SplitTexts(blob, unit, pageLines)
		more, err := rq.f.textsOptional("texts", "\x00")
		if err != nil {
			return nil, err
		}
		inline = append(more, inline...)
	} else if inline, err = rq.f.textsOptional("texts", "text"); err != nil {
		return nil, err
	}
	if len(inline) == 0 && len(names) == 0 {
		if rq.f.present("texts") || rq.f.present("text") {
			return nil, badRequest("'texts' contains no texts")
		}
		return nil, badRequest("missing field 'texts' (list of strings), 'text' (string, one text per line) or 'files' (list of upload names)")
	}
	src := radixnet.MultiSource{radixnet.SliceSource(inline)}
	if len(names) > 0 {
		files, err := rq.svc.UploadSource(names, unit, pageLines, wholeFile)
		if err != nil {
			return nil, err
		}
		src = append(src, files)
		// an archive of any size may still hold no usable text: probe the first text before starting a job
		n := 0
		if err := files.Each(func(string) error { n++; return errStop }); err != nil && !errors.Is(err, errStop) {
			return nil, wrapSourceError(err)
		}
		if n == 0 && len(inline) == 0 {
			return nil, badRequest("'files' contains no texts")
		}
	}
	return src, nil
}

var errStop = errors.New("stop")

func rTrain(rq *request) (int, any, error) {
	wholeFile, err := rq.f.flag("whole_file", false)
	if err != nil {
		return 0, nil, err
	}
	unit, pageLines, err := splitOptions(rq.f)
	if err != nil {
		return 0, nil, err
	}
	src, err := trainSource(rq, unit, pageLines, wholeFile)
	if err != nil {
		return 0, nil, err
	}
	epochs, _, err := rq.f.integer("epochs", 5, intp(0))
	if err != nil {
		return 0, nil, err
	}
	chunkSize, _, err := rq.f.integer("chunk_size", 0, intp(1))
	if err != nil {
		return 0, nil, err
	}
	parallelParts, err := rq.f.flag("parallel_parts", false)
	if err != nil {
		return 0, nil, err
	}
	inflight, _, err := rq.f.integer("inflight", 0, intp(1))
	if err != nil {
		return 0, nil, err
	}
	autoCompress, err := rq.f.flag("auto_compress", true)
	if err != nil {
		return 0, nil, err
	}
	// learning rates, batch sizes and schedules are accepted for interface parity and ignored
	for _, name := range []string{"lr", "act_lr", "clip"} {
		if _, _, err := rq.f.number(name, 0, floatp(0)); err != nil {
			return 0, nil, err
		}
	}
	if _, _, err := rq.f.integer("batch_size", 256, intp(1)); err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTrainSource(src, epochs, autoCompress, chunkSize, parallelParts, inflight)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{"job": job}, nil
}

func rJob(rq *request) (int, any, error) {
	if job := rq.svc.JobStatus(); job != nil {
		return 200, job, nil
	}
	return 200, nil, nil
}

func rJobStop(rq *request) (int, any, error) {
	out, err := rq.svc.StopJob()
	return 200, out, err
}

func pathDict(r *radixnet.PathResult) map[string]any {
	return map[string]any{
		"continuation": r.Text, "full_text": r.FullText, "cost": r.Cost, "probability": r.Probability(),
		"step_costs": r.StepCosts, "path": r.Labels, "node_ids": r.NodeIDs, "reached_end": r.ReachedEnd,
	}
}

func rPredict(rq *request) (int, any, error) {
	f := rq.f
	prefix, err := f.text("prefix", nil)
	if err != nil {
		return 0, nil, err
	}
	o := radixnet.DefaultPredictOptions()
	if o.Length, _, err = f.integer("length", 20, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.Mode, err = f.optText("mode", "beam"); err != nil {
		return 0, nil, err
	}
	if o.ToEnd, err = f.flag("to_end", false); err != nil {
		return 0, nil, err
	}
	if o.StepPenalty, _, err = f.number("step_penalty", 0, floatp(0)); err != nil {
		return 0, nil, err
	}
	if o.Temperature, _, err = f.number("temperature", 1.0, floatp(0)); err != nil {
		return 0, nil, err
	}
	if o.MaxLength, _, err = f.integer("max_length", -1, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.K, _, err = f.integer("k", 5, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.Beam, _, err = f.integer("beam", 0, intp(1)); err != nil {
		return 0, nil, err
	}
	p, err := rq.svc.Predict(prefix, o)
	if err != nil {
		return 0, nil, err
	}
	top := make([]map[string]any, 0, len(p.Top))
	for _, r := range p.Top {
		top = append(top, pathDict(r))
	}
	bottom := make([]map[string]any, 0, len(p.Bottom))
	for _, r := range p.Bottom {
		bottom = append(bottom, pathDict(r))
	}
	return 200, map[string]any{
		"prefix": prefix, "kind": "count", "continuation": p.Text, "full_text": p.FullText, "cost": p.Cost,
		"probability": p.Probability(), "step_costs": p.StepCosts, "path": p.Labels, "node_ids": p.NodeIDs,
		"expanded": p.Expanded, "reached_end": p.ReachedEnd, "mode": p.Mode, "k": p.K, "beam": p.Beam,
		"top": top, "bottom": bottom,
	}, nil
}

func sampleDict(r *radixnet.PathResult) map[string]any {
	return map[string]any{
		"text": r.Text, "full_text": r.FullText, "cost": r.Cost, "probability": r.Probability(), "path": r.Labels,
		"node_ids": r.NodeIDs, "step_costs": r.StepCosts, "reached_end": r.ReachedEnd,
	}
}

func rGenerate(rq *request) (int, any, error) {
	f := rq.f
	o := radixnet.DefaultGenerateOptions()
	var err error
	if o.Count, _, err = f.integer("count", 1, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.MaxLength, _, err = f.integer("max_length", 60, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.Mode, err = f.optText("mode", "sample"); err != nil {
		return 0, nil, err
	}
	if o.Temperature, _, err = f.number("temperature", 1.0, floatp(0)); err != nil {
		return 0, nil, err
	}
	seed, present, err := f.integer("seed", 0, nil)
	if err != nil {
		return 0, nil, err
	}
	if present {
		s := int64(seed)
		o.Seed = &s
	}
	if o.Prefix, err = f.optText("prefix", ""); err != nil {
		return 0, nil, err
	}
	if o.StepPenalty, _, err = f.number("step_penalty", 0, floatp(0)); err != nil {
		return 0, nil, err
	}
	if o.Beam, _, err = f.integer("beam", 0, intp(1)); err != nil {
		return 0, nil, err
	}
	results, err := rq.svc.Generate(o)
	if err != nil {
		return 0, nil, err
	}
	samples := make([]map[string]any, 0, len(results))
	for _, r := range results {
		samples = append(samples, sampleDict(r))
	}
	return 200, map[string]any{"samples": samples}, nil
}

func rConverse(rq *request) (int, any, error) {
	f := rq.f
	opening, err := f.optText("opening", "")
	if err != nil {
		return 0, nil, err
	}
	o := radixnet.DefaultConverseOptions()
	if o.Turns, _, err = f.integer("turns", 6, intp(0)); err != nil {
		return 0, nil, err
	}
	partner, err := f.optText("partner", "")
	if err != nil {
		return 0, nil, err
	}
	if partner != "" && strings.ToLower(partner) != "count" {
		return 0, nil, badRequest("no %s model in memory to converse with; the Go server runs the count / reward model only", partner)
	}
	if o.Mode, err = f.optText("mode", "beam"); err != nil {
		return 0, nil, err
	}
	if o.MaxLength, _, err = f.integer("max_length", 60, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.Context, _, err = f.integer("context", 12, intp(0)); err != nil {
		return 0, nil, err
	}
	if o.Temperature, _, err = f.number("temperature", 1.0, floatp(0)); err != nil {
		return 0, nil, err
	}
	if o.K, _, err = f.integer("k", 5, intp(1)); err != nil {
		return 0, nil, err
	}
	if o.Beam, _, err = f.integer("beam", 0, intp(1)); err != nil {
		return 0, nil, err
	}
	if o.StepPenalty, _, err = f.number("step_penalty", 0, floatp(0)); err != nil {
		return 0, nil, err
	}
	seed, present, err := f.integer("seed", 0, nil)
	if err != nil {
		return 0, nil, err
	}
	if present {
		s := int64(seed)
		o.Seed = &s
	}
	speakers, err := f.names("speakers")
	if err != nil {
		return 0, nil, err
	}
	if len(speakers) > 0 {
		o.Speakers = speakers
	}
	if o.History, err = f.textsOptional("history", "history_text"); err != nil {
		return 0, nil, err
	}
	if o.AvoidRepeats, err = f.flag("avoid_repeats", true); err != nil {
		return 0, nil, err
	}
	turns, err := rq.svc.Converse(opening, o)
	if err != nil {
		return 0, nil, err
	}
	if turns == nil {
		turns = []*radixnet.Turn{}
	}
	return 200, map[string]any{"kind": "count", "partner": nil, "speakers": o.Speakers, "turns": turns, "count": len(turns)}, nil
}

func rScore(rq *request) (int, any, error) {
	text, err := rq.f.text("text", nil)
	if err != nil {
		return 0, nil, err
	}
	return 200, rq.svc.Score(text), nil
}

func rTwoNRL(rq *request) (int, any, error) {
	f := rq.f
	wholeFile, err := f.flag("whole_file", false)
	if err != nil {
		return 0, nil, err
	}
	unit, pageLines, err := splitOptions(f)
	if err != nil {
		return 0, nil, err
	}
	bad, err := textsAndFiles(rq, "bad", "bad_text", "bad_files", unit, pageLines, wholeFile)
	if err != nil {
		return 0, nil, err
	}
	good, err := textsAndFiles(rq, "good", "good_text", "good_files", unit, pageLines, wholeFile)
	if err != nil {
		return 0, nil, err
	}
	negEpochs, _, err := f.integer("neg_epochs", 3, intp(0))
	if err != nil {
		return 0, nil, err
	}
	posEpochs, _, err := f.integer("pos_epochs", 3, intp(0))
	if err != nil {
		return 0, nil, err
	}
	strength, _, err := f.number("strength", 1.0, floatp(0))
	if err != nil {
		return 0, nil, err
	}
	for _, name := range []string{"neg_lr", "pos_lr"} {
		if _, _, err := f.number(name, 0, floatp(0)); err != nil {
			return 0, nil, err
		}
	}
	badWeights, err := ratingsOf(f, "bad", bad)
	if err != nil {
		return 0, nil, err
	}
	goodWeights, err := ratingsOf(f, "good", good)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTwoNRLRated(bad, badWeights, good, goodWeights, negEpochs, posEpochs, strength)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{"job": job}, nil
}

func rFeedback(rq *request) (int, any, error) {
	f := rq.f
	wholeFile, err := f.flag("whole_file", false)
	if err != nil {
		return 0, nil, err
	}
	good, err := f.textsOptional("good", "good_text")
	if err != nil {
		return 0, nil, err
	}
	bad, err := f.textsOptional("bad", "bad_text")
	if err != nil {
		return 0, nil, err
	}
	for _, pair := range []struct {
		field  string
		target *[]string
	}{{"good_files", &good}, {"bad_files", &bad}} {
		names, err := f.names(pair.field)
		if err != nil {
			return 0, nil, err
		}
		if len(names) > 0 {
			more, err := rq.svc.UploadTexts(names, "lines", 50, wholeFile)
			if err != nil {
				return 0, nil, err
			}
			*pair.target = append(*pair.target, more...)
		}
	}
	negEpochs, _, err := f.integer("neg_epochs", 2, intp(0))
	if err != nil {
		return 0, nil, err
	}
	posEpochs, _, err := f.integer("pos_epochs", 3, intp(0))
	if err != nil {
		return 0, nil, err
	}
	strength, _, err := f.number("strength", 1.0, floatp(0))
	if err != nil {
		return 0, nil, err
	}
	for _, name := range []string{"neg_lr", "pos_lr"} {
		if _, _, err := f.number(name, 0, floatp(0)); err != nil {
			return 0, nil, err
		}
	}
	goodWeights, err := ratingsOf(f, "good", good)
	if err != nil {
		return 0, nil, err
	}
	badWeights, err := ratingsOf(f, "bad", bad)
	if err != nil {
		return 0, nil, err
	}
	job, action, err := rq.svc.StartFeedbackRated(good, goodWeights, bad, badWeights, negEpochs, posEpochs, strength)
	if err != nil {
		return 0, nil, err
	}
	return 202, map[string]any{
		"job": job, "action": action, "good": len(good), "bad": len(bad),
		"good_weights": goodWeights, "bad_weights": badWeights,
	}, nil
}

func rInvert(rq *request) (int, any, error) {
	out, err := rq.svc.Invert()
	return 200, out, err
}

func rCompress(rq *request) (int, any, error) {
	out, err := rq.svc.Compress()
	return 200, out, err
}

func rSave(rq *request) (int, any, error) {
	p, err := rq.f.optText("path", "")
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.Save(p)
	return 200, out, err
}

func rLoad(rq *request) (int, any, error) {
	p, err := rq.f.text("path", nil)
	if err != nil {
		return 0, nil, err
	}
	if p == "" {
		return 0, nil, badRequest("'path' must not be empty")
	}
	out, err := rq.svc.Load(p)
	return 200, out, err
}

func rReset(rq *request) (int, any, error) {
	seed, present, err := rq.f.integer("seed", 0, nil)
	if err != nil {
		return 0, nil, err
	}
	var seedp *int64
	if present {
		s := int64(seed)
		seedp = &s
	}
	kind, err := rq.f.optText("kind", "")
	if err != nil {
		return 0, nil, err
	}
	opts, err := weightOptions(rq.f)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.Reset(seedp, kind, opts)
	return 200, out, err
}

func rCheckpoints(rq *request) (int, any, error) { return 200, rq.svc.Checkpoints(), nil }

func rCheckpointSave(rq *request) (int, any, error) {
	tag, err := rq.f.optText("tag", "manual")
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.SaveCheckpoint(tag)
	return 200, out, err
}

func rCheckpointRestore(rq *request) (int, any, error) {
	name, err := rq.f.text("name", nil)
	if err != nil {
		return 0, nil, err
	}
	if name == "" {
		return 0, nil, badRequest("'name' must not be empty")
	}
	out, err := rq.svc.RestoreCheckpoint(name)
	return 200, out, err
}

func rGraph(rq *request) (int, any, error) {
	limit := 150
	if raw, ok := rq.queryValue("limit"); ok {
		n, err := strconv.Atoi(raw)
		if err != nil {
			return 0, nil, badRequest("query parameter 'limit' must be an integer (got %q)", raw)
		}
		limit = n
	}
	out, err := rq.svc.Graph(limit)
	return 200, out, err
}

func rHistory(rq *request) (int, any, error) {
	out, err := rq.svc.History()
	return 200, out, err
}

func rUploads(rq *request) (int, any, error) {
	if rq.svc.uploads == nil {
		return 0, nil, badRequest("uploads are disabled: start the server with --upload-dir")
	}
	out, err := rq.svc.uploads.List()
	return 200, out, err
}

// uploadPayload is one file of an upload request: text (content) or bytes.
type uploadPayload struct {
	name string
	text *string
	data []byte
}

func uploadFiles(rq *request) ([]uploadPayload, error) {
	f := fields{}
	if len(strings.TrimSpace(string(rq.body))) > 0 {
		var body map[string]any
		if err := json.Unmarshal(rq.body, &body); err != nil {
			return nil, badRequest("request body is not a JSON object")
		}
		f.body = body
	}
	one := func(f fields) (uploadPayload, error) {
		name, err := f.text("name", nil)
		if err != nil {
			return uploadPayload{}, err
		}
		if v, ok := f.lookup("content_base64"); ok {
			s, isStr := v.(string)
			if !isStr {
				return uploadPayload{}, badRequest("'content_base64' must be a string")
			}
			data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
			if err != nil {
				return uploadPayload{}, badRequest("'content_base64' is not valid base64: %v", err)
			}
			return uploadPayload{name: name, data: data}, nil
		}
		content, err := f.text("content", nil)
		if err != nil {
			return uploadPayload{}, badRequest("missing field 'content' (string) or 'content_base64'")
		}
		return uploadPayload{name: name, text: &content}, nil
	}
	if v, ok := f.lookup("files"); ok {
		items, isList := v.([]any)
		if !isList || len(items) == 0 {
			return nil, badRequest("'files' must be a non-empty list of {name, content | content_base64} objects")
		}
		out := make([]uploadPayload, 0, len(items))
		for i, item := range items {
			obj, isObj := item.(map[string]any)
			if !isObj {
				return nil, badRequest("'files[%d]' must be an object with 'name' and 'content' (or 'content_base64')", i)
			}
			p, err := one(fields{obj})
			if err != nil {
				return nil, err
			}
			out = append(out, p)
		}
		return out, nil
	}
	p, err := one(f)
	if err != nil {
		return nil, err
	}
	return []uploadPayload{p}, nil
}

// rUpload is registered for the route table; POST /api/uploads is served by
// Handler.streamUpload before the body would be buffered.
func rUpload(rq *request) (int, any, error) {
	return 0, nil, badRequest("uploads are handled by the streaming path")
}

func rUploadDelete(rq *request) (int, any, error) {
	if rq.svc.uploads == nil {
		return 0, nil, badRequest("uploads are disabled: start the server with --upload-dir")
	}
	name, err := rq.f.text("name", nil)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.uploads.Delete(name)
	return 200, out, err
}

func rSchedule(rq *request) (int, any, error) {
	return 200, map[string]any{"variables": []any{}, "constants": []any{}, "functions": []any{}, "helpers": []any{}, "presets": []any{},
		"note": "learning-rate schedules apply to the sine network (Python server); the count model has no learning rate"}, nil
}

// -- the handler -------------------------------------------------------------------------------------------

// Handler serves the API and, when frontendDir is set, the built frontend.
type Handler struct {
	svc         *Service
	frontendDir string
	logf        func(string)
	quiet       bool
}

// NewHandler creates the HTTP handler.
func NewHandler(svc *Service, frontendDir string, quiet bool, logf func(string)) *Handler {
	if logf == nil {
		logf = func(string) {}
	}
	return &Handler{svc: svc, frontendDir: frontendDir, logf: logf, quiet: quiet}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	status := h.serve(w, r)
	if !h.quiet {
		h.logf(fmt.Sprintf("%s %s %d %.1fms", r.Method, r.URL.RequestURI(), status, float64(time.Since(t0).Microseconds())/1000))
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any, method string) int {
	raw, err := json.Marshal(payload)
	if err != nil {
		raw = []byte(`{"error":"response is not serialisable"}`)
		status = 500
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Length", strconv.Itoa(len(raw)))
	w.WriteHeader(status)
	if method != "HEAD" {
		_, _ = w.Write(raw)
	}
	return status
}

func (h *Handler) serve(w http.ResponseWriter, r *http.Request) (status int) {
	defer func() {
		if rec := recover(); rec != nil {
			h.logf(fmt.Sprintf("panic handling %s %s: %v", r.Method, r.URL.Path, rec))
			status = writeJSON(w, 500, map[string]any{"error": fmt.Sprintf("internal error: %v", rec)}, r.Method)
		}
	}()
	p := r.URL.Path
	if p == "" {
		p = "/"
	}
	if r.Method == http.MethodOptions {
		w.WriteHeader(204)
		return 204
	}
	if p == "/api" || strings.HasPrefix(p, "/api/") {
		return h.serveAPI(w, r, strings.TrimRight(p, "/"))
	}
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		return h.serveStatic(w, r, p)
	}
	w.Header().Set("Allow", "GET, HEAD, OPTIONS")
	return writeJSON(w, 405, map[string]any{"error": fmt.Sprintf("method %s is not allowed for %s", r.Method, p)}, r.Method)
}

func (h *Handler) serveAPI(w http.ResponseWriter, r *http.Request, p string) int {
	if p == "" {
		p = "/api"
	}
	table, ok := routes[p]
	if !ok {
		for _, prefix := range pythonOnly {
			if p == prefix || strings.HasPrefix(p, prefix+"/") {
				return writeJSON(w, 404, map[string]any{"error": fmt.Sprintf("%s is not available on the Go server (it serves the count / reward model only); use the Python server for it", p)}, r.Method)
			}
		}
		return writeJSON(w, 404, map[string]any{"error": fmt.Sprintf("unknown API endpoint %s", p)}, r.Method)
	}
	lookup := r.Method
	if lookup == http.MethodHead {
		lookup = http.MethodGet
	}
	fn, ok := table[lookup]
	if !ok {
		methods := make([]string, 0, len(table)+1)
		for m := range table {
			methods = append(methods, m)
		}
		sort.Strings(methods)
		allow := strings.Join(append(methods, "OPTIONS"), ", ")
		w.Header().Set("Allow", allow)
		return writeJSON(w, 405, map[string]any{"error": fmt.Sprintf("method %s is not allowed for %s; use %s", r.Method, p, allow)}, r.Method)
	}
	if lookup == http.MethodPost && p == "/api/uploads" {
		status, payload, err := h.streamUpload(r)
		if err != nil {
			return h.writeError(w, r, err)
		}
		return writeJSON(w, status, payload, r.Method)
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBodyBytes+1))
	if err != nil {
		return writeJSON(w, 400, map[string]any{"error": "could not read the request body"}, r.Method)
	}
	if len(body) > MaxBodyBytes {
		return writeJSON(w, 413, map[string]any{"error": fmt.Sprintf("request body larger than %d bytes", MaxBodyBytes)}, r.Method)
	}
	rq := &request{svc: h.svc, query: r.URL.Query(), body: body, header: r.Header}
	if lookup == http.MethodPost {
		if len(strings.TrimSpace(string(body))) > 0 {
			var parsed any
			if err := json.Unmarshal(body, &parsed); err != nil {
				return writeJSON(w, 400, map[string]any{"error": "request body is not valid JSON"}, r.Method)
			}
			obj, isObj := parsed.(map[string]any)
			if !isObj {
				return writeJSON(w, 400, map[string]any{"error": "request body must be a JSON object"}, r.Method)
			}
			rq.f = fields{obj}
		} else {
			rq.f = fields{map[string]any{}}
		}
	} else {
		rq.f = fields{map[string]any{}}
	}
	status, payload, err := fn(rq)
	if err != nil {
		return h.writeError(w, r, err)
	}
	if payload == nil {
		return writeJSON(w, status, nil, r.Method)
	}
	return writeJSON(w, status, payload, r.Method)
}

func (h *Handler) writeError(w http.ResponseWriter, r *http.Request, err error) int {
	var ae *apiError
	if errors.As(err, &ae) {
		return writeJSON(w, ae.status, map[string]any{"error": ae.message}, r.Method)
	}
	if os.IsNotExist(err) {
		return writeJSON(w, 404, map[string]any{"error": err.Error()}, r.Method)
	}
	return writeJSON(w, 400, map[string]any{"error": err.Error()}, r.Method)
}

// MaxJSONUploadBytes caps the JSON upload forms (inline content); multipart and
// raw uploads stream to disk and have no limit.
const MaxJSONUploadBytes = 512 << 20

// streamUpload handles POST /api/uploads without buffering: multipart file
// parts and raw bodies stream straight into the upload directory (an archive
// of any size), the JSON forms are parsed as before.
func (h *Handler) streamUpload(r *http.Request) (int, any, error) {
	if h.svc.uploads == nil {
		return 0, nil, badRequest("uploads are disabled: start the server with --upload-dir")
	}
	ct, params, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	uploads := []map[string]any{}
	archives := []map[string]any{}
	add := func(result map[string]any) {
		uploads = append(uploads, result["uploads"].([]map[string]any)...)
		if more, ok := result["archives"].([]map[string]any); ok {
			archives = append(archives, more...)
		}
	}
	query := r.URL.Query()
	switch {
	case ct == "multipart/form-data":
		if params["boundary"] == "" {
			return 0, nil, badRequest("malformed multipart/form-data body (missing boundary?)")
		}
		mr, err := r.MultipartReader()
		if err != nil {
			return 0, nil, badRequest("malformed multipart/form-data body: %v", err)
		}
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				return 0, nil, badRequest("malformed multipart/form-data body: %v", err)
			}
			name := part.FileName()
			if name == "" {
				_, _ = io.Copy(io.Discard, part) // plain form fields are ignored
				continue
			}
			result, err := h.svc.uploads.StoreStream(name, part)
			if err != nil {
				return 0, nil, err
			}
			add(result)
		}
		if len(uploads) == 0 {
			return 0, nil, badRequest("multipart body contains no file parts (use -F file=@corpus.txt)")
		}
	case ct == "application/json" || (ct == "" && len(query["name"]) == 0):
		body, err := io.ReadAll(io.LimitReader(r.Body, MaxJSONUploadBytes+1))
		if err != nil {
			return 0, nil, badRequest("could not read the request body")
		}
		if len(body) > MaxJSONUploadBytes {
			return 0, nil, &apiError{413, fmt.Sprintf("JSON upload larger than %d bytes: send the file as multipart/form-data or a raw body, which stream", MaxJSONUploadBytes)}
		}
		rq := &request{svc: h.svc, query: query, body: body, header: r.Header}
		files, err := uploadFiles(rq)
		if err != nil {
			return 0, nil, err
		}
		for _, file := range files {
			if file.text != nil {
				rec, err := h.svc.uploads.StoreText(file.name, *file.text)
				if err != nil {
					return 0, nil, err
				}
				uploads = append(uploads, rec)
				continue
			}
			result, err := h.svc.uploads.StoreBytes(file.name, file.data)
			if err != nil {
				return 0, nil, err
			}
			add(result)
		}
	default:
		names := query["name"]
		if len(names) == 0 || strings.TrimSpace(names[0]) == "" {
			return 0, nil, badRequest("raw uploads need a ?name=<file name> query parameter (or send JSON {name, content})")
		}
		result, err := h.svc.uploads.StoreStream(names[0], r.Body)
		if err != nil {
			return 0, nil, err
		}
		add(result)
	}
	if len(uploads) == 0 {
		return 0, nil, badRequest("nothing was uploaded")
	}
	body := map[string]any{"uploads": uploads}
	if len(archives) > 0 {
		body["archives"] = archives
	}
	return 201, body, nil
}

// -- static files -------------------------------------------------------------------------------------------

var spaSkip = []string{"/assets/", "/favicon", "/robots.txt", "/manifest", "/sw.js"}

func isSPARoute(p string) bool {
	if path.Ext(p) != "" {
		return false
	}
	for _, s := range spaSkip {
		if strings.HasPrefix(p, s) {
			return false
		}
	}
	return true
}

func (h *Handler) serveStatic(w http.ResponseWriter, r *http.Request, p string) int {
	root := h.frontendDir
	if root == "" {
		if p == "/" || p == "/index.html" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(200)
			if r.Method != http.MethodHead {
				_, _ = io.WriteString(w, helpPage)
			}
			return 200
		}
		return writeJSON(w, 404, map[string]any{"error": "not found: " + p}, r.Method)
	}
	clean := path.Clean("/" + p)
	target := filepath.Join(root, filepath.FromSlash(clean))
	rel, err := filepath.Rel(root, target)
	if err != nil || strings.HasPrefix(rel, "..") {
		return writeJSON(w, 404, map[string]any{"error": "not found: " + p}, r.Method)
	}
	st, err := os.Stat(target)
	if err != nil || st.IsDir() {
		if clean == "/" {
			target = filepath.Join(root, "index.html")
		} else if isSPARoute(clean) {
			target = filepath.Join(root, "index.html")
		} else {
			return writeJSON(w, 404, map[string]any{"error": "not found: " + p}, r.Method)
		}
	}
	data, err := os.ReadFile(target)
	if err != nil {
		return writeJSON(w, 404, map[string]any{"error": "not found: " + p}, r.Method)
	}
	ctype := mime.TypeByExtension(filepath.Ext(target))
	if ctype == "" {
		ctype = "application/octet-stream"
	}
	if strings.HasSuffix(target, ".html") {
		ctype = "text/html; charset=utf-8"
	}
	w.Header().Set("Content-Type", ctype)
	relPath := filepath.ToSlash(strings.TrimPrefix(target, root))
	if strings.Contains(relPath, "/assets/") {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		w.Header().Set("Cache-Control", "no-cache")
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	w.WriteHeader(200)
	if r.Method != http.MethodHead {
		_, _ = w.Write(data)
	}
	return 200
}

const helpPage = `<!doctype html><meta charset="utf-8"><title>RadixCyclicNN (Go)</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#1c2130}code{background:#eef0f5;padding:1px 5px;border-radius:4px}</style>
<h1>RadixCyclicNN &middot; count / reward model (Go)</h1>
<p>The API is up at <code>/api</code> (<a href="/api">endpoint list</a>, <a href="/api/status">status</a>). No frontend build was found:
build it with <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code> and start the server with
<code>--frontend-dir frontend/dist</code>.</p>`
