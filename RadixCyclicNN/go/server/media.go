package server

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Images and speech as text, and the recall tutor over both.

func init() {
	route("GET", "/api/images", rImages)
	doc("GET", "/api/images", "the image encoders this build has (the Stable Diffusion one needs the Python server)")
	route("POST", "/api/images/encode", rImageEncode)
	doc("POST", "/api/images/encode", "an image as JSON {name, content_base64} + {size, encoder, train, epochs, "+
		"save_as} -> the encoded text (202 with a training job when train is set)")
	route("POST", "/api/images/decode", rImageDecode)
	doc("POST", "/api/images/decode", "{text, encoder} -> {png_base64, ...}; a cut-off or rambling prediction "+
		"is padded / truncated first")
	route("POST", "/api/images/tutor", rImageTutor)
	doc("POST", "/api/images/tutor", "the recall tutor: ask the network to draw back an image it was shown and "+
		"mark what comes back {image bytes | texts, size, encoder, lead, length, attempts, mode, temperature, "+
		"threshold, blame}")
	route("GET", "/api/speech", rSpeech)
	doc("GET", "/api/speech", "the waveform codecs and the text format (transcription is Python-only)")
	route("POST", "/api/speech/teach", rSpeechTeach)
	doc("POST", "/api/speech/teach", "one utterance -> the words and the waveform behind one unique token: "+
		"audio as JSON {name, content_base64} + {transcript, rate, codec, normalise, waveform, pair, token, "+
		"unique, train, epochs, save_as}")
	route("POST", "/api/speech/decode", rSpeechDecode)
	doc("POST", "/api/speech/decode", "{text, codec} -> {wav_base64, ...} - an encoded or predicted waveform "+
		"as playable audio")
	route("POST", "/api/speech/tutor", rSpeechTutor)
	doc("POST", "/api/speech/tutor", "the recall tutor: ask the network to say back an utterance it was taught "+
		"and mark what comes back {recording bytes | texts, transcript, rate, codec, lead, length, attempts, "+
		"mode, temperature, threshold, blame}")
}

func rImages(rq *request) (int, any, error) { return 200, radixnet.DescribeVision(), nil }

func rSpeech(rq *request) (int, any, error) { return 200, radixnet.DescribeSpeech(), nil }

// mediaBytes reads the one binary payload of a media request.
func mediaBytes(rq *request, what string) (string, []byte, error) {
	files, err := uploadFiles(rq)
	if err != nil || len(files) == 0 {
		return "", nil, badRequest("send the %s as bytes: JSON {name, content_base64}", what)
	}
	if files[0].data == nil {
		return "", nil, badRequest("send the %s as bytes: JSON {name, content_base64}", what)
	}
	return files[0].name, files[0].data, nil
}

func rImageEncode(rq *request) (int, any, error) {
	name, data, err := mediaBytes(rq, "image")
	if err != nil {
		return 0, nil, err
	}
	zeroI := 0
	size, _, err := rq.f.integer("size", 0, &zeroI)
	if err != nil {
		return 0, nil, err
	}
	encoder, err := rq.f.optText("encoder", "auto")
	if err != nil {
		return 0, nil, err
	}
	encoded, err := radixnet.EncodeImage(data, size, encoder)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	out := map[string]any{
		"text": encoded.Text, "encoder": encoded.Encoder, "width": encoded.Width, "height": encoded.Height,
		"latent_shape": encoded.LatentShape, "bytes": encoded.Bytes, "chars": encoded.Chars,
		"source_size": encoded.SourceSize, "name": name, "upload": nil, "job": nil,
	}
	return mediaTrain(rq, out, []string{encoded.Text})
}

func rImageDecode(rq *request) (int, any, error) {
	text, err := rq.f.text("text", nil)
	if err != nil {
		return 0, nil, err
	}
	encoder, err := rq.f.optText("encoder", "")
	if err != nil {
		return 0, nil, err
	}
	decoded, err := radixnet.DecodeImageText(text, encoder)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	return 200, map[string]any{
		"png_base64": base64.StdEncoding.EncodeToString(decoded.PNG), "encoder": decoded.Encoder,
		"width": decoded.Width, "height": decoded.Height, "bytes": decoded.Bytes, "repaired": decoded.Repaired,
	}, nil
}

// speechOptionsFrom reads how an utterance becomes text.
func speechOptionsFrom(rq *request) (radixnet.TeachSpeechOptions, error) {
	o := radixnet.DefaultTeachSpeechOptions()
	var err error
	if o.Transcript, err = rq.f.optText("transcript", ""); err != nil {
		return o, err
	}
	zeroI := 0
	if o.Rate, _, err = rq.f.integer("rate", 0, &zeroI); err != nil {
		return o, err
	}
	if o.Rate == 0 {
		o.Rate = radixnet.DefaultSpeechRate
	}
	if o.Codec, err = rq.f.optText("codec", "auto"); err != nil {
		return o, err
	}
	if o.Normalise, err = rq.f.flag("normalise", false); err != nil {
		return o, err
	}
	if o.Waveform, err = rq.f.flag("waveform", true); err != nil {
		return o, err
	}
	if o.Pair, err = rq.f.flag("pair", false); err != nil {
		return o, err
	}
	if o.Token, err = rq.f.optText("token", ""); err != nil {
		return o, err
	}
	if o.Unique, err = rq.f.flag("unique", true); err != nil {
		return o, err
	}
	return o, nil
}

func rSpeechTeach(rq *request) (int, any, error) {
	name, data, err := mediaBytes(rq, "recording")
	if err != nil {
		return 0, nil, err
	}
	o, err := speechOptionsFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	taught, err := radixnet.TeachSpeech(data, o)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	out := map[string]any{
		"token": taught.Token, "transcript": taught.Transcript, "audio": taught.Audio,
		"texts": taught.Texts, "chars": taught.Chars, "pair": taught.Pair, "name": name,
		"upload": nil, "job": nil,
		"asr": map[string]any{"backend": "given", "model": nil, "language": nil, "seconds": 0.0,
			"error": nil},
	}
	return mediaTrain(rq, out, taught.Texts)
}

func rSpeechDecode(rq *request) (int, any, error) {
	text, err := rq.f.text("text", nil)
	if err != nil {
		return 0, nil, err
	}
	codec, err := rq.f.optText("codec", "")
	if err != nil {
		return 0, nil, err
	}
	decoded, err := radixnet.DecodeSpeechText(text, codec)
	if err != nil {
		return 0, nil, badRequest("%v", err)
	}
	return 200, map[string]any{
		"wav_base64": base64.StdEncoding.EncodeToString(decoded.WAV), "codec": decoded.Codec,
		"rate": decoded.Rate, "channels": decoded.Channels, "samples": decoded.Samples,
		"seconds": decoded.Seconds, "bytes": decoded.Bytes, "repaired": decoded.Repaired,
	}, nil
}

// mediaTrain finishes an encode / teach request: save_as keeps the texts as an
// upload, train starts a job on them.
func mediaTrain(rq *request, out map[string]any, texts []string) (int, any, error) {
	saveAs, err := rq.f.optText("save_as", "")
	if err != nil {
		return 0, nil, err
	}
	if saveAs != "" {
		upload, err := rq.svc.uploads.StoreText(saveAs, strings.Join(texts, "\n")+"\n")
		if err != nil {
			return 0, nil, err
		}
		out["upload"] = upload
	}
	train, err := rq.f.flag("train", false)
	if err != nil {
		return 0, nil, err
	}
	if !train {
		return 200, out, nil
	}
	if len(texts) == 0 {
		return 0, nil, badRequest("nothing to train on")
	}
	zeroI := 0
	epochs, _, err := rq.f.integer("epochs", 3, &zeroI)
	if err != nil {
		return 0, nil, err
	}
	job, err := rq.svc.StartTrain(texts, epochs, true)
	if err != nil {
		return 0, nil, err
	}
	out["job"] = job
	return 202, out, nil
}

// -- the recall tutor ---------------------------------------------------------

// recallOptionsFrom reads the recall-tutor options of a request.
func recallOptionsFrom(rq *request) (radixnet.RecallOptions, error) {
	o := radixnet.DefaultRecallOptions()
	zeroI, oneI := 0, 1
	zero := 0.0
	var err error
	if lead, ok, err := rq.f.integer("lead", -1, &zeroI); err != nil {
		return o, err
	} else if ok {
		o.Lead = lead
	}
	if o.Length, _, err = rq.f.integer("length", 0, &zeroI); err != nil {
		return o, err
	}
	if o.Attempts, _, err = rq.f.integer("attempts", 1, &oneI); err != nil {
		return o, err
	}
	if o.Mode, err = rq.f.optText("mode", "beam"); err != nil {
		return o, err
	}
	if o.Temperature, _, err = rq.f.number("temperature", 1, &zero); err != nil {
		return o, err
	}
	if o.Threshold, _, err = rq.f.number("threshold", 6, &zero); err != nil {
		return o, err
	}
	return o, nil
}

// RecallQuiz asks the model to write out texts it was taught, marks what comes
// back and optionally blames the failures.
func (s *Service) RecallQuiz(texts, labels, said []string, modality string, o radixnet.RecallOptions,
	blame bool) (map[string]any, error) {
	if len(texts) == 0 {
		return nil, badRequest("nothing to ask about: send a %s or 'texts' (already encoded)", modality)
	}
	var negative *radixnet.Model
	if blame {
		found, err := s.negativeModel()
		if err != nil {
			return nil, err
		}
		negative = found
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	o.Labels, o.Said = labels, said
	lessons, err := radixnet.Quiz(s.model, texts, o)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	out := map[string]any{
		"modality": modality, "lessons": lessons, "report": radixnet.RecallReportCard(lessons), "negative": nil,
	}
	if negative != nil {
		report, err := radixnet.TeachRecall(negative, lessons, o.Threshold, true, modality, radixnet.TeachOptions{})
		if err != nil {
			return nil, err
		}
		out["negative"] = map[string]any{
			"taught": report, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	return out, nil
}

func rImageTutor(rq *request) (int, any, error) {
	texts, err := rq.f.textsOptional("texts", "text")
	if err != nil {
		return 0, nil, err
	}
	labels := make([]string, len(texts))
	for i := range labels {
		labels[i] = fmt.Sprintf("text %d", i+1)
	}
	if len(texts) == 0 {
		name, data, err := mediaBytes(rq, "image")
		if err != nil {
			return 0, nil, err
		}
		zeroI := 0
		size, _, err := rq.f.integer("size", 0, &zeroI)
		if err != nil {
			return 0, nil, err
		}
		encoder, err := rq.f.optText("encoder", "auto")
		if err != nil {
			return 0, nil, err
		}
		encoded, err := radixnet.EncodeImage(data, size, encoder)
		if err != nil {
			return 0, nil, badRequest("%v", err)
		}
		texts, labels = []string{encoded.Text}, []string{name}
	}
	o, err := recallOptionsFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	blame, err := rq.f.flag("blame", false)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.RecallQuiz(texts, labels, nil, "image", o, blame)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}

func rSpeechTutor(rq *request) (int, any, error) {
	texts, err := rq.f.textsOptional("texts", "text")
	if err != nil {
		return 0, nil, err
	}
	labels, said := make([]string, len(texts)), make([]string, len(texts))
	for i := range labels {
		labels[i] = fmt.Sprintf("text %d", i+1)
	}
	if len(texts) == 0 {
		name, data, err := mediaBytes(rq, "recording")
		if err != nil {
			return 0, nil, err
		}
		speechOpts, err := speechOptionsFrom(rq)
		if err != nil {
			return 0, nil, err
		}
		speechOpts.Waveform = true
		taught, err := radixnet.TeachSpeech(data, speechOpts)
		if err != nil {
			return 0, nil, badRequest("%v", err)
		}
		waveform := ""
		for _, text := range taught.Texts {
			if strings.Contains(text, "aud:") {
				waveform = text
			}
		}
		if waveform == "" {
			return 0, nil, badRequest("nothing to remember: the recording produced no waveform")
		}
		texts = []string{waveform}
		labels = []string{name + " " + taught.Token}
		said = []string{taught.Transcript}
	}
	o, err := recallOptionsFrom(rq)
	if err != nil {
		return 0, nil, err
	}
	blame, err := rq.f.flag("blame", false)
	if err != nil {
		return 0, nil, err
	}
	out, err := rq.svc.RecallQuiz(texts, labels, said, "speech", o, blame)
	if err != nil {
		return 0, nil, err
	}
	return 200, out, nil
}
