package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"strings"
	"testing"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

func testPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 4), uint8(y * 4), 128, 255})
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func testWAV() []byte {
	rate := 8000
	samples := make([]float32, rate/20) // 0.05s
	for i := range samples {
		samples[i] = float32(0.6 * math.Sin(2*math.Pi*220*float64(i)/float64(rate)))
	}
	return radixnet.WAVBytes(samples, rate, 1)
}

func b64(data []byte) string { return base64.StdEncoding.EncodeToString(data) }

func TestImagesDescribe(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.get("/api/images")
	if status != 200 || doc["auto"] != "tiny" || doc["sd"] != false {
		t.Fatalf("images: %d %+v", status, doc)
	}
}

func TestSpeechDescribe(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.get("/api/speech")
	if status != 200 || doc["default_codec"] != "mu" {
		t.Fatalf("speech: %d %+v", status, doc)
	}
}

func TestImageEncodeAndDecode(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.post("/api/images/encode", map[string]any{
		"name": "photo.png", "content_base64": b64(testPNG(t)), "size": 64,
	})
	if status != 200 {
		t.Fatalf("encode: %d %+v", status, doc)
	}
	text, _ := doc["text"].(string)
	if !strings.HasPrefix(text, "img:tiny:64x64:") || doc["encoder"] != "tiny" {
		t.Fatalf("encoded: %+v", doc)
	}
	status, back := e.post("/api/images/decode", map[string]any{"text": text})
	if status != 200 {
		t.Fatalf("decode: %d %+v", status, back)
	}
	raw, err := base64.StdEncoding.DecodeString(back["png_base64"].(string))
	if err != nil {
		t.Fatalf("png_base64: %v", err)
	}
	if _, err := png.Decode(bytes.NewReader(raw)); err != nil {
		t.Errorf("the decoded PNG must be readable: %v", err)
	}
}

func TestImageEncodeCanTrain(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.post("/api/images/encode", map[string]any{
		"name": "photo.png", "content_base64": b64(testPNG(t)), "size": 64, "train": true, "epochs": 2,
	})
	if status != 202 || doc["job"] == nil {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitForJob()
}

func TestImageEncodeRefusesRubbish(t *testing.T) {
	e := newEnv(t, false)
	status, _ := e.post("/api/images/encode", map[string]any{"name": "x.png", "content_base64": b64([]byte("nope"))})
	if status != 400 {
		t.Errorf("unreadable bytes: %d", status)
	}
	status, _ = e.post("/api/images/encode", map[string]any{
		"name": "x.png", "content_base64": b64(testPNG(t)), "encoder": "sd",
	})
	if status != 400 {
		t.Errorf("the sd encoder must be refused here: %d", status)
	}
	status, _ = e.post("/api/images/encode", map[string]any{"text": "no bytes"})
	if status != 400 {
		t.Errorf("a request with no image: %d", status)
	}
}

func TestSpeechTeachAndDecode(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.post("/api/speech/teach", map[string]any{
		"name": "clip.wav", "content_base64": b64(testWAV()), "transcript": "the cat sat on the mat", "rate": 8000,
	})
	if status != 200 {
		t.Fatalf("teach: %d %+v", status, doc)
	}
	token, _ := doc["token"].(string)
	texts, _ := doc["texts"].([]any)
	if !strings.HasPrefix(token, "<speech:") || len(texts) != 2 {
		t.Fatalf("taught: %+v", doc)
	}
	waveform := ""
	for _, text := range texts {
		if s, _ := text.(string); strings.Contains(s, "aud:") {
			waveform = s
		}
	}
	status, back := e.post("/api/speech/decode", map[string]any{"text": waveform})
	if status != 200 {
		t.Fatalf("decode: %d %+v", status, back)
	}
	if back["codec"] != "mu" || back["repaired"] != false {
		t.Errorf("decoded: %+v", back)
	}
	raw, _ := base64.StdEncoding.DecodeString(back["wav_base64"].(string))
	if _, err := radixnet.ParseWAV(raw); err != nil {
		t.Errorf("the decoded WAV must be readable: %v", err)
	}
}

func TestSpeechTeachRefusesNothing(t *testing.T) {
	e := newEnv(t, false)
	status, _ := e.post("/api/speech/teach", map[string]any{"transcript": "only words"})
	if status != 400 {
		t.Errorf("no audio and no name: %d", status)
	}
}

func TestImageTutorMarksGivenTexts(t *testing.T) {
	e := newEnv(t, false)
	status, encoded := e.post("/api/images/encode", map[string]any{
		"name": "photo.png", "content_base64": b64(testPNG(t)), "size": 64,
	})
	if status != 200 {
		t.Fatalf("encode: %d", status)
	}
	text := encoded["text"].(string)
	if status, doc := e.post("/api/train", map[string]any{"texts": []string{text}, "epochs": 3}); status != 202 {
		t.Fatalf("train: %d %+v", status, doc)
	}
	e.waitForJob()
	status, doc := e.post("/api/images/tutor", map[string]any{"texts": []string{text}})
	if status != 200 {
		t.Fatalf("tutor: %d %+v", status, doc)
	}
	if doc["modality"] != "image" {
		t.Errorf("modality: %v", doc["modality"])
	}
	lessons, _ := doc["lessons"].([]any)
	if len(lessons) != 1 {
		t.Fatalf("lessons: %v", doc["lessons"])
	}
	if doc["negative"] != nil {
		t.Error("without blame nothing is taught")
	}
	report, _ := doc["report"].(map[string]any)
	if report["modality"] != "image" {
		t.Errorf("report: %v", report)
	}
}

func TestSpeechTutorEncodesAndQuizzesInOneCall(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.post("/api/speech/tutor", map[string]any{
		"name": "clip.wav", "content_base64": b64(testWAV()), "transcript": "hello there",
		"rate": 8000, "length": 80, "blame": true,
	})
	if status != 200 {
		t.Fatalf("tutor: %d %+v", status, doc)
	}
	lessons, _ := doc["lessons"].([]any)
	if len(lessons) != 1 {
		t.Fatalf("lessons: %v", doc["lessons"])
	}
	lesson, _ := lessons[0].(map[string]any)
	exercise, _ := lesson["exercise"].(map[string]any)
	if label, _ := exercise["label"].(string); !strings.Contains(label, "<speech:") {
		t.Errorf("label: %v", exercise["label"])
	}
	if doc["negative"] == nil {
		t.Error("blame must teach the negative network")
	}
}

func TestRecallTutorRefusesNothingToAskAbout(t *testing.T) {
	e := newEnv(t, false)
	status, _ := e.post("/api/images/tutor", map[string]any{"texts": []string{}})
	if status != 400 {
		t.Errorf("no texts and no image: %d", status)
	}
	status, _ = e.post("/api/speech/tutor", map[string]any{"texts": []string{}})
	if status != 400 {
		t.Errorf("no texts and no recording: %d", status)
	}
}

func TestMediaRoutesAreDocumented(t *testing.T) {
	e := newEnv(t, false)
	status, doc := e.get("/api")
	if status != 200 {
		t.Fatalf("status %d", status)
	}
	raw, _ := json.Marshal(doc)
	for _, path := range []string{
		"/api/images", "/api/images/encode", "/api/images/decode", "/api/images/tutor",
		"/api/speech", "/api/speech/teach", "/api/speech/decode", "/api/speech/tutor",
	} {
		if !strings.Contains(string(raw), path) {
			t.Errorf("%s is not in the endpoint index", path)
		}
	}
}
