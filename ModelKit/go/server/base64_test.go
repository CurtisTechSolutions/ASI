package server

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// A model of base64 symbols behind the HTTP API: it learns the bytes of what
// it is given - an inline text as its UTF-8 bytes, an upload as every byte it
// holds - is asked prefixes as the symbols their bytes fix, says what its
// answers hold, and takes its own words back as they are when a request says
// symbols.  Python's tests/test_base64.py TestTheApi, case for case.

// theBlob is the head of a PNG and some bytes that are no text: what an image,
// a video or a recording is to the unit.
var theBlob = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1}

var b64Corpus = []string{
	"the cat sat on the mat",
	"the cat sat on the floor",
	"the dog sat on the log",
	"a quick brown fox jumps over the lazy dog",
}

// base64Env is a server whose model is a fresh one of base64 symbols.
func base64Env(t *testing.T) *env {
	t.Helper()
	e := newEnv(t, true)
	if status, doc := e.post("/api/reset", map[string]any{"kind": "count", "encoding": "base64:3:1", "seed": 1}); status != 200 {
		t.Fatalf("reset: %d %v", status, doc)
	}
	return e
}

// trainBase64 runs one train job to the end.
func (e *env) trainBase64(body map[string]any) {
	e.t.Helper()
	if _, ok := body["epochs"]; !ok {
		body["epochs"] = 2
	}
	if status, doc := e.post("/api/train", body); status != 202 {
		e.t.Fatalf("train: %d %v", status, doc)
	}
	if job := e.waitJob(); job["state"] != "done" {
		e.t.Fatalf("train job: %v", job)
	}
}

func lossyOf(symbols string) string { return radixnet.LossyUTF8(radixnet.Base64Bytes(symbols)) }

func TestBase64TheEncodingIsBase64(t *testing.T) {
	e := base64Env(t)
	_, enc := e.get("/api/encoding")
	if enc["encoding"] != "base64:3:1" || enc["unit"] != "base64" {
		t.Fatalf("encoding: %v", enc)
	}
	if _, m := e.get("/api/model"); m["units"] != "symbols" {
		t.Fatalf("model units: %v", m["units"])
	}
	if status, doc := e.post("/api/reset", map[string]any{"unit": "b64", "ngram": 4, "stride": 4}); status != 200 {
		t.Fatalf("reset by unit: %d %v", status, doc)
	}
	if _, enc = e.get("/api/encoding"); enc["encoding"] != "base64:4:4" {
		t.Fatalf("encoding after a reset by unit: %v", enc)
	}
	if status, doc := e.get("/api/words"); status != 400 || !strings.Contains(doc["error"].(string), "no words to list") {
		t.Fatalf("words of a model of symbols: %d %v", status, doc)
	}
}

func TestBase64TextsAreLearnedAsTheirBytesAndAnswersDecoded(t *testing.T) {
	e := base64Env(t)
	e.trainBase64(map[string]any{"texts": b64Corpus})
	for _, label := range e.svc.Model().G.Labels[radixnet.First:] {
		for _, r := range label {
			if !strings.ContainsRune(radixnet.Base64Alphabet, r) {
				t.Fatalf("label %q is not base64 symbols", label)
			}
		}
	}
	status, p := e.post("/api/predict", map[string]any{"prefix": "the cat sat on the", "length": 8, "to_end": true, "k": 2, "guard": false})
	if status != 200 {
		t.Fatalf("predict: %d %v", status, p)
	}
	if p["prefix"] != "the cat sat on the" || p["prefix_symbols"] != radixnet.Base64Prefix([]byte("the cat sat on the")) {
		t.Fatalf("prefix %v, symbols %v", p["prefix"], p["prefix_symbols"])
	}
	if d := p["decoded"]; d != "the cat sat on the mat" && d != "the cat sat on the floor" {
		t.Fatalf("decoded %v", d)
	}
	if !strings.HasSuffix(p["decoded"].(string), p["decoded_continuation"].(string)) || p["decoded_bytes"] != float64(len(p["decoded"].(string))) {
		t.Fatalf("decoded %v, continuation %v, bytes %v", p["decoded"], p["decoded_continuation"], p["decoded_bytes"])
	}
	rows := append(p["top"].([]any), p["bottom"].([]any)...)
	if len(rows) == 0 {
		t.Fatal("no rows")
	}
	for _, row := range rows {
		r := row.(map[string]any)
		if r["decoded"] != lossyOf(r["full_text"].(string)) || !strings.HasSuffix(r["decoded"].(string), r["decoded_continuation"].(string)) {
			t.Fatalf("row %v", r)
		}
	}
	status, g := e.post("/api/generate", map[string]any{"count": 2, "mode": "beam", "prefix": "the", "guard": false})
	if status != 200 || g["prefix_symbols"] != "dGhl" {
		t.Fatalf("generate: %d %v", status, g)
	}
	for _, s := range g["samples"].([]any) {
		sample := s.(map[string]any)
		if !strings.HasPrefix(sample["decoded"].(string), "the") ||
			sample["decoded_bytes"] != float64(len(radixnet.Base64Bytes(sample["full_text"].(string)))) {
			t.Fatalf("sample %v", sample)
		}
	}
}

// A like or a rating gives back what the model said - symbols already - and
// symbols says so; without it, a text is the bytes it is written in.
func TestBase64TheModelsOwnWordsComeBackAsTheyAre(t *testing.T) {
	e := base64Env(t)
	e.trainBase64(map[string]any{"texts": b64Corpus})
	said := radixnet.Base64Text([]byte("the dog sat on the log"))
	if _, sc := e.post("/api/score", map[string]any{"text": said, "symbols": true}); sc["unknown_transitions"] != 0.0 {
		t.Fatalf("the model's own words: %v", sc)
	}
	if _, sc := e.post("/api/score", map[string]any{"text": said}); sc["unknown_transitions"].(float64) == 0 {
		t.Fatalf("the bytes of its symbols are not what it learned: %v", sc)
	}
	if _, sc := e.post("/api/score", map[string]any{"text": "the dog sat on the log"}); sc["unknown_transitions"] != 0.0 {
		t.Fatalf("a text it learned, given in words: %v", sc)
	}
	status, fb := e.post("/api/feedback", map[string]any{"good": []string{radixnet.Base64Text([]byte("the cat ate the rat"))}, "symbols": true})
	if status != 202 || fb["action"] != "reward" {
		t.Fatalf("feedback: %d %v", status, fb)
	}
	if job := e.waitJob(); job["state"] != "done" {
		t.Fatalf("feedback job: %v", job)
	}
	status, p := e.post("/api/predict", map[string]any{"prefix": radixnet.Base64Prefix([]byte("the cat ate")), "symbols": true,
		"length": 6, "to_end": true, "guard": false})
	if status != 200 || p["prefix_symbols"] != radixnet.Base64Prefix([]byte("the cat ate")) {
		t.Fatalf("predict: %d %v", status, p)
	}
	// fourteen symbols fix ten bytes and half the eleventh: the walk finished the "e"
	if p["decoded"] != "the cat ate the rat" || p["decoded_continuation"] != "e the rat" {
		t.Fatalf("decoded %v, continuation %v", p["decoded"], p["decoded_continuation"])
	}
}

func TestBase64ThePreviewShowsTheSymbols(t *testing.T) {
	e := base64Env(t)
	_, doc := e.post("/api/encoding/preview", map[string]any{"text": "the cat"})
	if doc["text"] != "the cat" || doc["symbols"] != "dGhlIGNhdA" || doc["round_trip"] != true || doc["chars"] != 10.0 {
		t.Fatalf("preview: %v", doc)
	}
	if w := doc["windows"].([]any); w[0] != "dGh" || w[1] != "Ghl" {
		t.Fatalf("windows: %v", w)
	}
	_, doc = e.post("/api/encoding/preview", map[string]any{"text": "dGhl", "symbols": true})
	if doc["symbols"] != "dGhl" || !reflect.DeepEqual(doc["windows"], []any{"dGh", "Ghl"}) {
		t.Fatalf("preview of symbols: %v", doc)
	}
}

func TestBase64ATextModelAddsNothing(t *testing.T) {
	e := newEnv(t, true)
	e.trainBase64(map[string]any{"texts": b64Corpus})
	_, p := e.post("/api/predict", map[string]any{"prefix": "the cat", "length": 4, "guard": false, "symbols": true})
	for _, key := range []string{"prefix_symbols", "decoded", "decoded_continuation", "decoded_bytes"} {
		if _, ok := p[key]; ok {
			t.Fatalf("a character model's answer carries %q: %v", key, p)
		}
	}
	if !strings.HasPrefix(p["full_text"].(string), "the cat") {
		t.Fatalf("symbols changed a character model's prefix: %v", p["full_text"])
	}
	_, g := e.post("/api/generate", map[string]any{"count": 1, "mode": "beam", "guard": false})
	if _, ok := g["prefix_symbols"]; ok {
		t.Fatalf("a character model's samples carry prefix_symbols: %v", g)
	}
	if _, doc := e.post("/api/encoding/preview", map[string]any{"text": "the cat"}); doc["symbols"] != nil {
		t.Fatalf("a character model's preview carries symbols: %v", doc)
	}
}

// An upload that is not UTF-8 text is kept byte for byte and learned whole; a
// text upload is still text, its byte order mark dropped - in every form an
// upload can take.
func TestBase64AnUploadKeepsItsBytes(t *testing.T) {
	e := base64Env(t)
	uploads := filepath.Join(e.dir, "uploads")
	status, doc := e.post("/api/uploads", map[string]any{"name": "image.png", "content_base64": base64Of(theBlob)})
	if status != 201 || doc["uploads"].([]any)[0].(map[string]any)["bytes"] != float64(len(theBlob)) {
		t.Fatalf("binary upload: %d %v", status, doc)
	}
	stored := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(uploads, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	if !bytes.Equal(stored("image.png"), theBlob) {
		t.Fatalf("the image was stored as %x", stored("image.png"))
	}
	// the same through a raw body, and a file that only starts like a text keeps its first three bytes too
	odd := append(append([]byte{}, utf8BOM...), theBlob...)
	if status, doc, _ := e.do("POST", "/api/uploads?name=odd.bin", odd, map[string]string{"Content-Type": "application/octet-stream"}); status != 201 {
		t.Fatalf("raw binary upload: %d %v", status, doc)
	}
	if !bytes.Equal(stored("odd.bin"), odd) {
		t.Fatalf("a binary file that starts with a byte order mark was stored as %x", stored("odd.bin"))
	}
	text := "\ufeff" + strings.Join(b64Corpus, "\n") + "\n"
	if status, doc := e.post("/api/uploads", map[string]any{"name": "corpus.txt", "content_base64": base64Of([]byte(text))}); status != 201 {
		t.Fatalf("text upload: %d %v", status, doc)
	}
	if string(stored("corpus.txt")) != text[len("\ufeff"):] {
		t.Fatalf("the text was stored as %q", stored("corpus.txt"))
	}
	if status, doc, _ := e.do("POST", "/api/uploads?name=raw.txt", "\ufeffthe cat\n", map[string]string{"Content-Type": "text/plain"}); status != 201 {
		t.Fatalf("raw text upload: %d %v", status, doc)
	}
	if string(stored("raw.txt")) != "the cat\n" {
		t.Fatalf("a raw text was stored as %q", stored("raw.txt"))
	}
	// from START a model of one file walks that file: the decoded walk is the file again
	e.trainBase64(map[string]any{"files": []string{"image.png"}})
	_, p := e.post("/api/predict", map[string]any{"prefix": "", "length": 1, "to_end": true, "guard": false})
	if got := radixnet.Base64Bytes(p["full_text"].(string)); !bytes.Equal(got, theBlob) || p["decoded_bytes"] != float64(len(theBlob)) {
		t.Fatalf("the walk decoded to %x (%v bytes)", got, p["decoded_bytes"])
	}
	e.trainBase64(map[string]any{"files": []string{"corpus.txt"}})
	if _, sc := e.post("/api/score", map[string]any{"text": b64Corpus[2]}); sc["unknown_transitions"] != 0.0 {
		t.Fatalf("a line of the text upload is not known: %v", sc)
	}
}

// The Go server's split cuts an inline text for a model of base64 symbols as
// for any model, and every piece is then converted; uploads follow their
// bytes, which no split cuts.
func TestBase64SplitCutsTheInlineTextNotTheUploads(t *testing.T) {
	e := base64Env(t)
	if status, doc := e.post("/api/uploads", map[string]any{"name": "notes.txt", "content": "para one line a\npara one line b\n\npara two\n"}); status != 201 {
		t.Fatalf("upload: %d %v", status, doc)
	}
	e.trainBase64(map[string]any{"text": "first para\ncontinues\n\nsecond para\n", "files": []string{"notes.txt"}, "split": "paragraphs", "epochs": 1})
	_, st := e.get("/api/status")
	if st["trained_texts"] != 5.0 { // two paragraphs inline, three lines of the upload
		t.Fatalf("trained %v texts, want 5", st["trained_texts"])
	}
	score := func(body map[string]any) float64 {
		t.Helper()
		_, sc := e.post("/api/score", body)
		return sc["unknown_transitions"].(float64)
	}
	if score(map[string]any{"text": "first para continues"}) != 0 || score(map[string]any{"text": "second para"}) != 0 {
		t.Fatalf("the inline paragraphs were not learned as their bytes")
	}
	if score(map[string]any{"text": "para one line a"}) != 0 || score(map[string]any{"text": "para two"}) != 0 {
		t.Fatalf("the upload's lines were not learned as their bytes")
	}
	if score(map[string]any{"text": "para one line a para one line b"}) == 0 {
		t.Fatalf("split cut the upload into paragraphs")
	}
	// with symbols the pieces are symbols already, and are taken as they are
	pieces := radixnet.Base64Text([]byte("the cat")) + "\n" + radixnet.Base64Text([]byte("on the mat")) + "\n\n" + radixnet.Base64Text([]byte("the dog"))
	e.trainBase64(map[string]any{"text": pieces, "split": "paragraphs", "symbols": true, "epochs": 1})
	joined := radixnet.Base64Text([]byte("the cat")) + " " + radixnet.Base64Text([]byte("on the mat"))
	if score(map[string]any{"text": joined, "symbols": true}) != 0 || score(map[string]any{"text": "the dog"}) != 0 {
		t.Fatalf("the pieces given in symbols were not taken as they are")
	}
}

// A text model reads an upload that is not text as Python does: one U+FFFD
// per maximal invalid subpart (radixnet.LossyUTF8), the byte order mark
// dropped.
func TestBase64ATextModelReadsABinaryUploadAsPythonDoes(t *testing.T) {
	if got := decodeText([]byte("\xef\xbb\xbfok\xed\xa0\x80")); got != "ok\ufffd\ufffd\ufffd" {
		t.Fatalf("decodeText = %q", got)
	}
	e := newEnv(t, true)
	data := []byte("ok\xe2\x82A\xff caf\xc3\xa9 \xf0\x90\x80\n")
	if status, doc := e.post("/api/uploads", map[string]any{"name": "odd.txt", "content_base64": base64Of(data)}); status != 201 {
		t.Fatalf("upload: %d %v", status, doc)
	}
	stored, err := os.ReadFile(filepath.Join(e.dir, "uploads", "odd.txt"))
	if err != nil || !bytes.Equal(stored, data) {
		t.Fatalf("stored %q (%v)", stored, err)
	}
	e.trainBase64(map[string]any{"files": []string{"odd.txt"}, "epochs": 1})
	if _, sc := e.post("/api/score", map[string]any{"text": "ok\ufffdA\ufffd caf\u00e9 \ufffd"}); sc["unknown_transitions"] != 0.0 {
		t.Fatalf("the upload was not read with one U+FFFD per maximal invalid subpart: %v", sc)
	}
}
