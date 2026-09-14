package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Images and speech as text, and the recall tutor that asks the network for
// them back.

func imageUsage() {
	fmt.Fprint(os.Stderr, `usage: radixnet-count [global options] image <action> [options]

An image becomes a text the network trains on and predicts: reduced to a thumbnail at 1/8
of the size, one byte per channel value, base64 - img:tiny:128x128:AAECAwQF...

actions:
  info     which encoders this build has
  encode   encode FILE as text (--size, --encoder, --out, --train)
  tutor    ask the network to draw back what it was shown, and mark what comes back
  decode   turn an encoded or predicted text back into a PNG

The Stable Diffusion encoder needs torch and diffusers: use the Python side for it.  The
text format is the same, so a model trained on sd texts by either side is read by both.
`)
}

func cmdImage(args []string) {
	if len(args) == 0 {
		imageUsage()
		os.Exit(2)
	}
	action, rest := args[0], args[1:]
	switch action {
	case "info":
		cmdImageInfo(rest)
	case "encode":
		cmdImageEncode(rest)
	case "tutor":
		cmdMediaTutor(rest, "image")
	case "decode":
		cmdImageDecode(rest)
	case "help", "-h", "--help":
		imageUsage()
	default:
		fail("unknown image action %q (info, encode, tutor, decode)", action)
	}
}

func cmdImageInfo(args []string) {
	fs := flag.NewFlagSet("image info", flag.ExitOnError)
	_ = fs.Parse(permute(fs, args))
	info := radixnet.DescribeVision()
	sayInfo(info, "encoders", "auto", "default_size", "sd", "sd_note", "text_format", "formats")
	if jsonMode {
		emit(info)
	}
}

func cmdImageEncode(args []string) {
	fs := flag.NewFlagSet("image encode", flag.ExitOnError)
	file := fs.String("file", "", "image file (PNG, JPEG or GIF)")
	size := fs.Int("size", radixnet.DefaultImageSize, "resize to SIZE x SIZE (a multiple of 8) before encoding")
	encoder := fs.String("encoder", "auto", "the encoder (auto = tiny here)")
	out := fs.String("out", "", "write the encoded text to this file")
	train := fs.Bool("train", false, "train the model on the encoded text, then save it")
	epochs := fs.Int("epochs", 3, "training epochs with -train")
	modelOut := fs.String("model-out", "", "where to save the model with -train (default: --model)")
	_ = fs.Parse(permute(fs, args))

	path := *file
	if path == "" && fs.NArg() > 0 {
		path = fs.Arg(0)
	}
	if path == "" {
		fail("an image file is required: radixnet-count image encode FILE")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fail("cannot read %s: %v", path, err)
	}
	encoded, err := radixnet.EncodeImage(data, *size, *encoder)
	if err != nil {
		fail("%v", err)
	}
	say("file     %s", path)
	say("encoder  %s", encoded.Encoder)
	say("size     %dx%d (source %dx%d)", encoded.Width, encoded.Height, encoded.SourceSize[0], encoded.SourceSize[1])
	say("latent   %s = %d bytes", joinInts(encoded.LatentShape, " x "), encoded.Bytes)
	say("text     %d chars", encoded.Chars)
	say("")
	say("%s", clip(encoded.Text, 200))
	doc := map[string]any{"file": path, "encoder": encoded.Encoder, "width": encoded.Width,
		"height": encoded.Height, "latent_shape": encoded.LatentShape, "bytes": encoded.Bytes,
		"chars": encoded.Chars, "source_size": encoded.SourceSize, "text": encoded.Text,
		"out": nil, "trained": nil}
	if *out != "" {
		if err := os.WriteFile(*out, []byte(encoded.Text+"\n"), 0o644); err != nil {
			fail("cannot write %s: %v", *out, err)
		}
		doc["out"] = *out
		say("")
		say("text written to %s", *out)
	}
	if *train {
		doc["trained"] = trainOnTexts([]string{encoded.Text}, *epochs, *modelOut)
	}
	if jsonMode {
		emit(doc)
	}
}

func cmdImageDecode(args []string) {
	fs := flag.NewFlagSet("image decode", flag.ExitOnError)
	text := fs.String("text", "", "the encoded text")
	data := fs.String("data", "", "a file holding the encoded text")
	encoder := fs.String("encoder", "", "override the encoder named in the text")
	out := fs.String("out", "", "where to write the PNG (required)")
	_ = fs.Parse(permute(fs, args))

	body := mediaText(*text, *data)
	if strings.TrimSpace(*out) == "" {
		fail("-out is required: where to write the PNG")
	}
	decoded, err := radixnet.DecodeImageText(body, *encoder)
	if err != nil {
		fail("%v", err)
	}
	if err := os.WriteFile(*out, decoded.PNG, 0o644); err != nil {
		fail("cannot write %s: %v", *out, err)
	}
	say("encoder  %s", decoded.Encoder)
	say("size     %dx%d", decoded.Width, decoded.Height)
	say("repaired %v", decoded.Repaired)
	say("written  %s", *out)
	if jsonMode {
		emit(map[string]any{"encoder": decoded.Encoder, "width": decoded.Width, "height": decoded.Height,
			"bytes": decoded.Bytes, "repaired": decoded.Repaired, "out": *out})
	}
}

// -- speech -------------------------------------------------------------------

func speechUsage() {
	fmt.Fprint(os.Stderr, `usage: radixnet-count [global options] speech <action> [options]

One utterance becomes two texts that start with the same unique token - the words and the
waveform - so in the graph the two leave the same node.

actions:
  info     which codecs this build has
  teach    encode FILE (WAV) and -text into the texts the network learns (--train)
  tutor    ask the network to say back what it was taught, and mark what comes back
  decode   turn an encoded or predicted waveform text back into a WAV file

Transcription is Python-only (faster-whisper / openai-whisper are Python packages): pass
the words with -text, which is what the browser's dictation does.
`)
}

func cmdSpeech(args []string) {
	if len(args) == 0 {
		speechUsage()
		os.Exit(2)
	}
	action, rest := args[0], args[1:]
	switch action {
	case "info":
		cmdSpeechInfo(rest)
	case "teach":
		cmdSpeechTeach(rest)
	case "tutor":
		cmdMediaTutor(rest, "speech")
	case "decode":
		cmdSpeechDecode(rest)
	case "help", "-h", "--help":
		speechUsage()
	default:
		fail("unknown speech action %q (info, teach, tutor, decode)", action)
	}
}

func cmdSpeechInfo(args []string) {
	fs := flag.NewFlagSet("speech info", flag.ExitOnError)
	_ = fs.Parse(permute(fs, args))
	info := radixnet.DescribeSpeech()
	sayInfo(info, "codecs", "default_codec", "default_rate", "token", "token_example",
		"backends", "backends_note", "text_format", "formats")
	if jsonMode {
		emit(info)
	}
}

// addSpeechEncodeFlags registers how an utterance becomes text.  The
// shared-token flag is not here: every caller sets Unique itself, and
// registering the same flag twice on one set panics.
func addSpeechEncodeFlags(fs *flag.FlagSet) *radixnet.TeachSpeechOptions {
	o := radixnet.DefaultTeachSpeechOptions()
	fs.StringVar(&o.Transcript, "text", "", "the transcript (what was said)")
	fs.IntVar(&o.Rate, "rate", o.Rate, "resample the waveform to this many samples per second")
	fs.StringVar(&o.Codec, "codec", o.Codec, "waveform quantisation: auto | mu | pcm8")
	fs.BoolVar(&o.Normalise, "normalise", false, "scale a quiet recording up to full range first")
	fs.StringVar(&o.Token, "token", "", "use this token instead of <speech:digest>")
	return &o
}

func cmdSpeechTeach(args []string) {
	fs := flag.NewFlagSet("speech teach", flag.ExitOnError)
	file := fs.String("file", "", "audio file (WAV)")
	o := addSpeechEncodeFlags(fs)
	noWaveform := fs.Bool("no-waveform", false, "learn the transcript only, not the sound")
	pair := fs.Bool("pair", false, "also learn one text of the waveform followed by its transcript")
	shared := fs.Bool("shared-token", false, "use the plain <speech> for every utterance")
	out := fs.String("out", "", "write the texts (one per line) to this file")
	train := fs.Bool("train", false, "train the model on the texts, then save it")
	epochs := fs.Int("epochs", 3, "training epochs with -train")
	modelOut := fs.String("model-out", "", "where to save the model with -train (default: --model)")
	_ = fs.Parse(permute(fs, args))

	path := *file
	if path == "" && fs.NArg() > 0 {
		path = fs.Arg(0)
	}
	var data []byte
	if path != "" {
		read, err := os.ReadFile(path)
		if err != nil {
			fail("cannot read %s: %v", path, err)
		}
		data = read
	}
	o.Waveform, o.Pair, o.Unique = !*noWaveform, *pair, !*shared
	taught, err := radixnet.TeachSpeech(data, *o)
	if err != nil {
		fail("%v", err)
	}
	say("source     %s", orDash(path))
	say("token      %s", taught.Token)
	say("transcript %s", orDash(taught.Transcript))
	if taught.Audio != nil {
		say("waveform   %s %d Hz, %.2fs, %d bytes", taught.Audio.Codec, taught.Audio.Rate,
			taught.Audio.Seconds, taught.Audio.Bytes)
	}
	say("texts      %d (%d chars)", len(taught.Texts), taught.Chars)
	for _, text := range taught.Texts {
		say("")
		say("%s", clip(text, 200))
	}
	doc := map[string]any{"source": path, "token": taught.Token, "transcript": taught.Transcript,
		"audio": taught.Audio, "texts": taught.Texts, "chars": taught.Chars, "pair": taught.Pair,
		"out": nil, "trained": nil}
	if *out != "" {
		if err := os.WriteFile(*out, []byte(strings.Join(taught.Texts, "\n")+"\n"), 0o644); err != nil {
			fail("cannot write %s: %v", *out, err)
		}
		doc["out"] = *out
		say("")
		say("%d text(s) written to %s", len(taught.Texts), *out)
	}
	if *train {
		if len(taught.Texts) == 0 {
			fail("nothing to train on: no transcript and no waveform")
		}
		doc["trained"] = trainOnTexts(taught.Texts, *epochs, *modelOut)
	}
	if jsonMode {
		emit(doc)
	}
}

func cmdSpeechDecode(args []string) {
	fs := flag.NewFlagSet("speech decode", flag.ExitOnError)
	text := fs.String("text", "", "the encoded text")
	data := fs.String("data", "", "a file holding the encoded text")
	codec := fs.String("codec", "", "override the codec named in the text")
	out := fs.String("out", "", "where to write the WAV (required)")
	_ = fs.Parse(permute(fs, args))

	body := mediaText(*text, *data)
	if strings.TrimSpace(*out) == "" {
		fail("-out is required: where to write the WAV")
	}
	decoded, err := radixnet.DecodeSpeechText(body, *codec)
	if err != nil {
		fail("%v", err)
	}
	if err := os.WriteFile(*out, decoded.WAV, 0o644); err != nil {
		fail("cannot write %s: %v", *out, err)
	}
	say("codec    %s", decoded.Codec)
	say("rate     %d Hz, %d channel(s)", decoded.Rate, decoded.Channels)
	say("samples  %d (%.2fs)", decoded.Samples, decoded.Seconds)
	say("repaired %v", decoded.Repaired)
	say("written  %s", *out)
	if jsonMode {
		emit(map[string]any{"codec": decoded.Codec, "rate": decoded.Rate, "channels": decoded.Channels,
			"samples": decoded.Samples, "seconds": decoded.Seconds, "bytes": decoded.Bytes,
			"repaired": decoded.Repaired, "out": *out})
	}
}

// -- the recall tutor ---------------------------------------------------------

// cmdMediaTutor is `image tutor` and `speech tutor`: encode the files, ask the
// network for them back, mark what comes and blame what it forgot.
func cmdMediaTutor(args []string, modality string) {
	fs := flag.NewFlagSet(modality+" tutor", flag.ExitOnError)
	var files multiFlag
	fs.Var(&files, "file", "file to ask it about (repeatable)")
	o := radixnet.DefaultRecallOptions()
	lead := fs.Int("lead", -1, "payload characters the exercise gives away (-1: the modality's default)")
	length := fs.Int("length", 0, "ask for only the first N payload characters (0: all of it)")
	attempts := fs.Int("attempts", 1, "tries per exercise; it stops at the first pass")
	mode := fs.String("mode", "beam", "how the first attempt is written: beam | sample")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature of the later attempts")
	threshold := fs.Float64("threshold", 6.0, "pass mark out of 10")
	train := fs.Bool("train", false, "teach it the texts first, then ask for them back")
	epochs := fs.Int("epochs", 3, "training epochs with -train")
	modelOut := fs.String("model-out", "", "where to save the model with -train (default: --model)")
	blame := fs.Bool("blame", false, "teach the negative network why each failure failed")
	size := fs.Int("size", radixnet.DefaultImageSize, "image only: resize to SIZE x SIZE before encoding")
	encoder := fs.String("encoder", "auto", "image only: the encoder")
	speechOpts := addSpeechEncodeFlags(fs)
	sharedToken := fs.Bool("shared-token", false, "speech only: use the plain <speech> for every utterance")
	addNegativeFlag(fs)
	_ = fs.Parse(permute(fs, args))

	paths := append([]string{}, files...)
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fail("at least one file is required: radixnet-count %s tutor FILE...", modality)
	}
	texts, labels, said := []string{}, []string{}, []string{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			fail("cannot read %s: %v", path, err)
		}
		if modality == "image" {
			encoded, err := radixnet.EncodeImage(data, *size, *encoder)
			if err != nil {
				fail("%s: %v", path, err)
			}
			texts, labels, said = append(texts, encoded.Text), append(labels, baseName(path)), append(said, "")
			continue
		}
		speechOpts.Waveform, speechOpts.Unique = true, !*sharedToken
		taught, err := radixnet.TeachSpeech(data, *speechOpts)
		if err != nil {
			fail("%s: %v", path, err)
		}
		waveform := ""
		for _, text := range taught.Texts {
			if strings.Contains(text, "aud:") {
				waveform = text
			}
		}
		if waveform == "" {
			fail("%s produced no waveform to remember", path)
		}
		texts = append(texts, waveform)
		labels = append(labels, baseName(path)+" "+taught.Token)
		said = append(said, taught.Transcript)
	}

	model := openModel(!*train)
	o.Lead, o.Labels, o.Attempts, o.Mode = *lead, labels, *attempts, *mode
	o.Temperature, o.Length, o.Threshold, o.Said = *temperature, *length, *threshold, said

	leadShown := *lead
	if leadShown < 0 {
		leadShown = radixnet.DefaultRecallLead[modality]
	}
	exercise := fmt.Sprintf("the header plus %d payload character(s)", leadShown)
	if leadShown == 0 {
		exercise = "the header alone"
		if modality == "speech" {
			exercise = "the token and the header alone"
		}
	}
	unit := "images"
	if modality == "speech" {
		unit = "utterances"
	}
	say("%-11s %d", unit, len(texts))
	say("%-11s %s", "exercise", exercise)
	if *length > 0 {
		say("%-11s %d payload character(s)", "asked for", *length)
	} else {
		say("%-11s the whole payload", "asked for")
	}
	say("%-11s %d (%s, then sampled)", "attempts", *attempts, *mode)
	say("%-11s %g/10", "pass mark", *threshold)
	if *train {
		say("")
		say("teaching it first: epochs=%d", *epochs)
		if _, err := model.Train(texts, radixnet.TrainOptions{Epochs: *epochs, AutoCompress: true}); err != nil {
			fail("%v", err)
		}
	}
	say("")
	lessons, err := radixnet.Quiz(model, texts, o)
	if err != nil {
		fail("%v", err)
	}
	say("%-28s %7s %5s %10s %-11s %s", "what", "attempt", "mark", "agreement", "verdict", "why")
	for _, lesson := range lessons {
		what := lesson.Exercise.Label
		if what == "" {
			what = lesson.Exercise.ID
		}
		verdict := lesson.Grade.Error
		if lesson.Grade.Passed {
			verdict = "pass"
		}
		say("%-28s %7d %5.1f %9.0f%% %-11s %s", clip(what, 28), lesson.Attempt, lesson.Grade.Score,
			lesson.Grade.Facts.Agreement*100, verdict, clip(lesson.Grade.Comment, 52))
	}
	card := radixnet.RecallReportCard(lessons)
	say("")
	summary := fmt.Sprintf("report card: %v/%v remembered, mean %s/10, %.0f%% agreement",
		card["passed"], card["lessons"], fmtMark(card["mean_score"]),
		card["mean_agreement"].(float64)*100)
	if reasons, _ := card["reasons"].(map[string]int); len(reasons) > 0 {
		summary += reasonSummary(reasons)
	}
	say("%s", summary)
	doc := map[string]any{"modality": modality, "lessons": lessons, "report": card,
		"trained": *train, "saved": nil, "negative": nil}
	if *train {
		target := *modelOut
		if target == "" {
			target = modelPath
		}
		if err := model.Save(target); err != nil {
			fail("cannot save %s: %v", target, err)
		}
		doc["saved"] = target
		say("saved %s", target)
	}
	if *blame {
		negative := openNegative(false)
		report, err := radixnet.TeachRecall(negative, lessons, *threshold, true, modality, radixnet.TeachOptions{})
		if err != nil {
			fail("%v", err)
		}
		saved := saveNegative(negative)
		say("")
		say("blamed %d failure(s) over %d edge(s); cleared %d fragment(s) from what it did remember",
			report.Blamed, report.Edges, report.Cleared)
		say("negative model: %s", saved)
		reasonTable(negative, 10)
		doc["negative"] = map[string]any{"path": saved, "taught": report, "reasons": negative.Reasons(),
			"stats": negative.Stats()}
	}
	if jsonMode {
		emit(doc)
	}
}

// -- shared helpers -----------------------------------------------------------

// permute moves the positional arguments after the flags before parsing, so
// `speech tutor FILE --text "..."` works.  Go's flag package stops at the first
// non-flag argument, while the Python CLI takes them in either order, and a
// file name silently swallowing the rest of the command line is the worst kind
// of difference between the two.
func permute(fs *flag.FlagSet, args []string) []string {
	flags, positional := []string{}, []string{}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			positional = append(positional, args[i+1:]...)
			break
		}
		if len(arg) > 1 && arg[0] == '-' {
			flags = append(flags, arg)
			if strings.Contains(arg, "=") {
				continue // --flag=value carries its own value
			}
			name := strings.TrimLeft(arg, "-")
			if found := fs.Lookup(name); found != nil && !isBoolFlag(found) && i+1 < len(args) {
				i++
				flags = append(flags, args[i])
			}
			continue
		}
		positional = append(positional, arg)
	}
	return append(flags, positional...)
}

func isBoolFlag(f *flag.Flag) bool {
	value, ok := f.Value.(interface{ IsBoolFlag() bool })
	return ok && value.IsBoolFlag()
}

// mediaText is --text, else the contents of --data.
func mediaText(text, data string) string {
	if strings.TrimSpace(text) != "" {
		return text
	}
	if strings.TrimSpace(data) == "" {
		fail("give the text with -text, or a file holding it with -data")
	}
	body, err := os.ReadFile(data)
	if err != nil {
		fail("cannot read %s: %v", data, err)
	}
	return strings.TrimSpace(string(body))
}

// trainOnTexts trains the model on texts and saves it.
func trainOnTexts(texts []string, epochs int, out string) map[string]any {
	m := openModel(false)
	records, err := m.Train(texts, radixnet.TrainOptions{Epochs: epochs, AutoCompress: true})
	if err != nil {
		fail("%v", err)
	}
	target := out
	if target == "" {
		target = modelPath
	}
	if err := m.Save(target); err != nil {
		fail("cannot save %s: %v", target, err)
	}
	say("")
	say("trained %d epoch(s) on %d text(s); saved to %s", len(records), len(texts), target)
	return map[string]any{"epochs": len(records), "texts": len(texts), "saved": target}
}

// sayInfo prints the named keys of a describe() map in order.
func sayInfo(info map[string]any, keys ...string) {
	for _, key := range keys {
		value := info[key]
		if list, ok := value.([]string); ok {
			value = strings.Join(list, ", ")
		}
		say("%-14s %v", key, value)
	}
}

func joinInts(values []int, sep string) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprint(value)
	}
	return strings.Join(parts, sep)
}

func baseName(path string) string {
	if at := strings.LastIndexAny(path, "/\\"); at >= 0 {
		return path[at+1:]
	}
	return path
}
