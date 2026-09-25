// Command phonetok is the Go port's command line: the same commands as the
// Python package's (`phonetok tokenize "The cat sat."`, explain, ipa, decode,
// pronounce, rhymes, affinity, coin, blend, lexicon, eval), the same output.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go/phonetok"
)

type options struct {
	level                            string
	noStress, noBoundaries, noPauses bool
	core                             bool
	lexicon                          string
	jsonOut                          bool
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: phonetok <command> [options] [args]

commands:
  tokenize TEXT...        the tokens, one line (--ids for ids)
  explain TEXT...         every word: phones, source, syllables, IPA
  ipa TEXT...             the IPA
  decode TOKENS...        sounds (or ids) back to words
  pronounce WORD...       one word, every pronunciation the lexicon has
  rhymes WORD [WORD]      rhymes in the lexicon, or whether two words rhyme
  affinity A B            how well two sounds go together
  coin                    new pronounceable words (--count, --seed, --syllables)
  blend A B               a portmanteau
  lexicon [WORD...]       what the lexicon holds and where it came from
  eval                    the rules against the full dictionary (PHONETOK_LEXICON)
  say TEXT...             speak: the formant synthesizer to a WAV (--out), a player (--play) or stdout (--raw);
                          tokens on stdin when no text is given (--rate, --pitch, --tempo, --gain)
  hear WAV...             the acoustic units of recordings: sounds learned, nothing written down (--frames)
  learn WAV...            learn a codebook of acoustic units from recordings (--units, --seed, --iterations,
                          --note, --out FILE)
  replay UNIT...          units spoken back through the vocoder (--out, --play, --raw, --polish, --pitch, --gain);
                          units on stdin when none are given
  codebook [FILE]         what a codebook holds (the bundled one when no file is named)

options (after the command):
  --level L  --no-stress  --no-boundaries  --no-pauses  --core  --lexicon FILE  --json
  --codebook FILE         the codebook of acoustic units (hear, replay; the bundled one otherwise)`)
}

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		usage()
		return 2
	}
	command := args[0]
	fs := flag.NewFlagSet(command, flag.ContinueOnError)
	var o options
	fs.StringVar(&o.level, "level", phonetok.Phoneme, "what one token is")
	fs.BoolVar(&o.noStress, "no-stress", false, "drop the vowels' stress digits")
	fs.BoolVar(&o.noBoundaries, "no-boundaries", false, "no # between words")
	fs.BoolVar(&o.noPauses, "no-pauses", false, "drop punctuation instead of making pauses of it")
	fs.BoolVar(&o.core, "core", false, "the bundled core lexicon only")
	fs.StringVar(&o.lexicon, "lexicon", "", "a CMU-format dictionary to read on top of the core")
	fs.BoolVar(&o.jsonOut, "json", false, "one JSON document on stdout")
	ids := fs.Bool("ids", false, "print ids instead of tokens (tokenize)")
	limit := fs.Int("limit", 50, "at most this many (rhymes: 0 = all; eval: words to sample)")
	count := fs.Int("count", 5, "how many (coin)")
	seed := fs.Int("seed", -1, "the seed (coin, eval)")
	syllables := fs.Int("syllables", 0, "syllables per word (coin)")
	out := fs.String("out", "say.wav", "the WAV file to write (say)")
	play := fs.Bool("play", false, "play through aplay / paplay / ffplay / play / afplay as it is made (say)")
	raw := fs.Bool("raw", false, "16-bit mono PCM on stdout, streamed (say)")
	rate := fs.Int("rate", phonetok.Rate, "the sample rate (say)")
	pitch := fs.Float64("pitch", 120, "the base pitch in Hz (say)")
	tempo := fs.Float64("tempo", 1, "the pace, 1 = the table's (say)")
	gain := fs.Float64("gain", 0.5, "the peak level as a share of full scale (say); a multiplier (replay, default 1)")
	codebookPath := fs.String("codebook", "", "the codebook of acoustic units (hear, replay; the bundled one otherwise)")
	frames := fs.Bool("frames", false, "a unit per frame, runs not collapsed (hear)")
	units := fs.Int("units", 64, "units to learn (learn)")
	iterations := fs.Int("iterations", 50, "Lloyd iterations at most (learn)")
	note := fs.String("note", "", "a line to keep in the codebook about where the audio came from (learn)")
	polish := fs.Int("polish", 0, "Griffin-Lim iterations over the whole utterance (replay; 0 = streamed)")
	// options may follow the positional arguments, as in the Python CLI
	var positional []string
	var flags []string
	for _, a := range args[1:] {
		if strings.HasPrefix(a, "-") && len(a) > 1 && !isNumber(a) {
			flags = append(flags, a)
			continue
		}
		if len(flags) > 0 && !strings.Contains(flags[len(flags)-1], "=") && takesValue(flags[len(flags)-1]) && len(flags) > len(positional)*0 {
			// the value of the previous flag
			last := flags[len(flags)-1]
			if !valueGiven[last] {
				flags[len(flags)-1] = last + "=" + a
				valueGiven[last] = true
				continue
			}
		}
		positional = append(positional, a)
	}
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	if *limit == 50 && command == "eval" {
		*limit = 20000
	}
	if *seed == -1 && command == "eval" {
		*seed = 7
	}
	if *seed == -1 && command == "learn" {
		*seed = 1
	}
	visited := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { visited[f.Name] = true })
	if !visited["out"] {
		switch command {
		case "learn":
			*out = "codebook.tsv"
		case "replay":
			*out = "replay.wav"
		}
	}
	if command == "replay" && !visited["gain"] {
		*gain = 1.0
	}
	tok, err := newTokenizer(o)
	if err != nil {
		fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
		return 1
	}
	emit := func(doc any, lines []string) {
		if o.jsonOut {
			enc := json.NewEncoder(os.Stdout)
			enc.SetEscapeHTML(false)
			enc.SetIndent("", " ")
			enc.Encode(doc)
			return
		}
		for _, l := range lines {
			fmt.Println(l)
		}
	}
	text := strings.Join(positional, " ")
	switch command {
	case "tokenize":
		tokens := tok.Tokenize(text)
		idList := tok.Encode(text, true)
		texts, kinds := make([]string, len(tokens)), make([]string, len(tokens))
		for i, t := range tokens {
			texts[i], kinds[i] = t.Text, t.Kind
		}
		line := strings.Join(texts, " ")
		if *ids {
			parts := make([]string, len(idList))
			for i, id := range idList {
				parts[i] = strconv.Itoa(id)
			}
			line = strings.Join(parts, " ")
		}
		emit(map[string]any{"text": text, "level": tok.Level, "tokens": texts, "ids": idList, "kinds": kinds,
			"vocab_size": tok.Vocab.Len()}, []string{line})
	case "explain":
		rows := tok.Explain(text)
		lines := []string{fmt.Sprintf("%-16s %-11s %-32s %-32s ipa", "word", "how", "phones", "syllables")}
		for _, r := range rows {
			lines = append(lines, fmt.Sprintf("%-16s %-11s %-32s %-32s %s", r.Word, r.How, strings.Join(r.Phones, " "),
				strings.Join(r.Syllables, " "), r.IPA))
		}
		if rows == nil {
			rows = []phonetok.Explanation{}
		}
		emit(map[string]any{"words": rows, "tokenizer": tok.Describe()}, lines)
	case "ipa":
		ipa := tok.IPA(text)
		emit(map[string]any{"text": text, "ipa": ipa}, []string{ipa})
	case "decode":
		tokens := strings.Fields(text)
		allDigits := len(tokens) > 0
		for _, t := range tokens {
			if _, err := strconv.Atoi(t); err != nil {
				allDigits = false
			}
		}
		var words string
		if allDigits {
			idList := make([]int, len(tokens))
			for i, t := range tokens {
				idList[i], _ = strconv.Atoi(t)
			}
			words = tok.DecodeIDs(idList)
		} else {
			words = tok.Decode(tokens)
		}
		emit(map[string]any{"tokens": tokens, "text": words}, []string{words})
	case "pronounce":
		var docs []map[string]any
		var lines []string
		for _, word := range positional {
			phones, how := tok.Transcriber.Explain(word)
			variants := tok.Lexicon().Pronunciations(word)
			if variants == nil {
				variants = [][]string{phones}
			}
			var syl []string
			for _, s := range phonetok.Syllabify(phones, true) {
				syl = append(syl, s.Text())
			}
			docs = append(docs, map[string]any{"word": word, "how": how, "phones": phones, "variants": variants,
				"syllables": syl, "ipa": tok.IPA(word), "rules": phonetok.LetterToSound(word)})
			lines = append(lines, fmt.Sprintf("%s: %s  [%s]  /%s/  %s", word, strings.Join(phones, " "), how, tok.IPA(word),
				strings.Join(syl, " | ")))
			pad := strings.Repeat(" ", len(word))
			for _, v := range variants[1:] {
				lines = append(lines, fmt.Sprintf("%s  %s  [lexicon, alternative]", pad, strings.Join(v, " ")))
			}
			if how != "rules" {
				lines = append(lines, fmt.Sprintf("%s  %s  [the rules alone would say]", pad, strings.Join(phonetok.LetterToSound(word), " ")))
			}
		}
		emit(map[string]any{"words": docs}, lines)
	case "rhymes":
		if len(positional) >= 2 {
			a, b := positional[0], positional[1]
			yes := tok.RhymesWith(a, b)
			verdict := "do not rhyme"
			if yes {
				verdict = "rhyme"
			}
			emit(map[string]any{"a": a, "b": b, "rhymes": yes}, []string{fmt.Sprintf("%s / %s: %s", a, b, verdict)})
			return 0
		}
		if len(positional) == 0 {
			usage()
			return 2
		}
		word := positional[0]
		phones := tok.Pronounce(word)
		set := map[string]bool{}
		for _, e := range tok.Lexicon().Items() {
			if e.Word != strings.ToLower(word) && phonetok.Rhymes(phones, e.Phones) {
				set[e.Word] = true
			}
		}
		found := make([]string, 0, len(set))
		for w := range set {
			found = append(found, w)
		}
		sort.Strings(found)
		if *limit > 0 && len(found) > *limit {
			found = found[:*limit]
		}
		line := strings.Join(found, " ")
		if len(found) == 0 {
			line = "(none in the lexicon)"
		}
		emit(map[string]any{"word": word, "phones": phones, "rhymes": found}, []string{line})
	case "affinity":
		if len(positional) < 2 {
			usage()
			return 2
		}
		a, b := strings.ToUpper(positional[0]), strings.ToUpper(positional[1])
		value := tok.Affinity(a, b)
		p := tok.Phonotactics().Prob(a, b)
		verdict := "meet by chance"
		if value > 1 {
			verdict = "go together"
		} else if value < -1 {
			verdict = "avoid each other"
		}
		emit(map[string]any{"a": a, "b": b, "affinity_bits": value, "probability": p, "verdict": verdict},
			[]string{fmt.Sprintf("%s -> %s: %+.2f bits (P = %.3f): they %s", a, b, value, p, verdict)})
	case "coin":
		var rng phonetok.Rng
		if *seed >= 0 {
			rng = phonetok.NewMT(uint32(*seed))
		} else {
			rng = goRng{rand.New(rand.NewSource(int64(os.Getpid())))}
		}
		var docs []map[string]any
		var lines []string
		for i := 0; i < *count; i++ {
			phones := tok.Phonotactics().Build(rng, *syllables, 1, 3, tok.Lexicon().Items(), 50)
			spelling := tok.Transcriber.Spell(phones)
			docs = append(docs, map[string]any{"phones": phones, "spelling": spelling, "ipa": phonetok.ToIPA(phones, true)})
			lines = append(lines, fmt.Sprintf("%-14s %-24s /%s/", spelling, strings.Join(phones, " "), phonetok.ToIPA(phones, true)))
		}
		emit(map[string]any{"words": docs, "seed": *seed}, lines)
	case "blend":
		if len(positional) < 2 {
			usage()
			return 2
		}
		a, b := positional[0], positional[1]
		phones, _, _, err := tok.Phonotactics().Blend(tok.Pronounce(a), tok.Pronounce(b))
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		spelling := tok.Transcriber.Spell(phones)
		emit(map[string]any{"a": a, "b": b, "phones": phones, "spelling": spelling, "ipa": phonetok.ToIPA(phones, true)},
			[]string{fmt.Sprintf("%s + %s = %s  (%s, /%s/)", a, b, spelling, strings.Join(phones, " "), phonetok.ToIPA(phones, true))})
	case "lexicon":
		lex := tok.Lexicon()
		full := "none found (set PHONETOK_LEXICON)"
		if path := os.Getenv(phonetok.EnvLexicon); path != "" {
			full = "file " + path
		}
		lines := []string{fmt.Sprintf("%d words from %s", lex.Len(), strings.Join(lex.Sources, ", ")),
			"full dictionary: " + full, "phonotactics: " + tok.Phonotactics().String()}
		lookups := map[string][][]string{}
		for _, w := range positional {
			prons := lex.Pronunciations(w)
			lookups[w] = prons
			if prons == nil {
				lines = append(lines, w+": (not in the lexicon)")
				continue
			}
			var parts []string
			for _, p := range prons {
				parts = append(parts, strings.Join(p, " "))
			}
			lines = append(lines, w+": "+strings.Join(parts, " | "))
		}
		doc := map[string]any{"words": lex.Len(), "sources": lex.Sources, "full_dictionary": full, "phonotactics": tok.Phonotactics().String()}
		if len(positional) > 0 {
			doc["lookups"] = lookups
		}
		emit(doc, lines)
	case "say":
		voice := phonetok.DefaultVoice()
		voice.Pitch, voice.Tempo, voice.Gain = *pitch, *tempo, *gain
		synth := phonetok.NewSynthesizer(*rate, voice)
		var tokens []string
		if len(positional) > 0 {
			tokens = tok.Tokens(text)
		} else {
			sc := bufio.NewScanner(os.Stdin)
			for sc.Scan() {
				tokens = append(tokens, strings.Fields(sc.Text())...)
			}
		}
		switch {
		case *play:
			player := phonetok.FindPlayer()
			if player == nil {
				fmt.Fprintln(os.Stderr, "phonetok: no player found (aplay, paplay, ffplay, play or afplay)")
				return 2
			}
			cmd := exec.Command(player[0], player[1:]...)
			stdin, err := cmd.StdinPipe()
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			if err := cmd.Start(); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			stdin.Write(phonetok.WavHeader(*rate, -1))
			total := 0
			for _, t := range tokens {
				chunk := synth.Feed(t)
				stdin.Write(chunk)
				total += len(chunk)
			}
			chunk := synth.End()
			stdin.Write(chunk)
			total += len(chunk)
			stdin.Close()
			cmd.Wait()
			emit(map[string]any{"seconds": float64(total) / 2 / float64(*rate), "player": player[0]},
				[]string{fmt.Sprintf("%.2f s spoken through %s", float64(total)/2/float64(*rate), player[0])})
		case *raw:
			w := bufio.NewWriter(os.Stdout)
			for _, t := range tokens {
				w.Write(synth.Feed(t))
				w.Flush()
			}
			w.Write(synth.End())
			w.Flush()
		default:
			pcm := synth.Speak(tokens)
			if err := os.WriteFile(*out, phonetok.WavBytes(pcm, *rate), 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			seconds := float64(len(pcm)) / 2 / float64(*rate)
			emit(map[string]any{"path": *out, "seconds": seconds, "rate": *rate, "bytes": len(pcm) + 44},
				[]string{fmt.Sprintf("%.2f s of speech written to %s (%d Hz)", seconds, *out, *rate)})
		}
	case "hear":
		book, err := loadBook(*codebookPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		tok, err := phonetok.NewAcousticTokenizer(book, !*frames)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		type heardDoc struct {
			Path    string   `json:"path"`
			Units   []string `json:"units"`
			Frames  int      `json:"frames"`
			Seconds float64  `json:"seconds"`
			Codes   []int    `json:"codes"`
		}
		docs := []heardDoc{}
		var lines []string
		for _, path := range positional {
			data, err := os.ReadFile(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			h, err := tok.ListenWAV(data)
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %s: %v\n", path, err)
				return 1
			}
			docs = append(docs, heardDoc{path, h.Units, h.Frames, h.Seconds, h.Codes})
			lines = append(lines, h.Text())
		}
		emit(docs, lines)
	case "learn":
		analysis := phonetok.DefaultAnalysis()
		var recordings [][]float64
		seconds := 0.0
		for _, path := range positional {
			samples, rate, err := phonetok.LoadWAV(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %s: %v\n", path, err)
				return 1
			}
			if samples, err = phonetok.Resample(samples, rate, analysis.Rate); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %s: %v\n", path, err)
				return 1
			}
			seconds += float64(len(samples)) / float64(analysis.Rate)
			recordings = append(recordings, samples)
		}
		if *note == "" {
			*note = fmt.Sprintf("learned from %d recording(s), %.0f s", len(positional), seconds)
		}
		book, err := phonetok.Learn(recordings, *units, int64(*seed), *iterations, analysis, *note)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		if err := book.Dump(*out); err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		total := 0
		for _, c := range book.Counts {
			total += c
		}
		emit(map[string]any{"path": *out, "units": book.K(), "frames": total, "seconds": seconds, "inertia": book.Inertia,
			"seed": *seed, "note": *note},
			[]string{fmt.Sprintf("%d units learned from %d frames (%.1f s of audio), inertia %.1f: written to %s",
				book.K(), total, seconds, book.Inertia, *out)})
	case "replay":
		book, err := loadBook(*codebookPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		tok, err := phonetok.NewAcousticTokenizer(book, true)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		var units []string
		if len(positional) > 0 {
			if units, err = tok.UnitsOf(text); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
		} else {
			sc := bufio.NewScanner(os.Stdin)
			for sc.Scan() {
				units = append(units, strings.Fields(sc.Text())...)
			}
		}
		rate := book.Analysis.Rate
		stream := func(w func([]byte)) error {
			voc, err := tok.Vocoder(*gain, *pitch)
			if err != nil {
				return err
			}
			for _, u := range units {
				chunk, err := voc.Feed(u)
				if err != nil {
					return err
				}
				w(chunk)
			}
			w(voc.End())
			return nil
		}
		switch {
		case *play:
			player := phonetok.FindPlayer()
			if player == nil {
				fmt.Fprintln(os.Stderr, "phonetok: no player found (aplay, paplay, ffplay, play or afplay)")
				return 2
			}
			cmd := exec.Command(player[0], player[1:]...)
			stdin, err := cmd.StdinPipe()
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			if err := cmd.Start(); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			stdin.Write(phonetok.WavHeader(rate, -1))
			total := 0
			err = stream(func(chunk []byte) { stdin.Write(chunk); total += len(chunk) })
			stdin.Close()
			cmd.Wait()
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			seconds := float64(total) / 2 / float64(rate)
			emit(map[string]any{"seconds": seconds, "player": player[0]},
				[]string{fmt.Sprintf("%.2f s replayed through %s", seconds, player[0])})
		case *raw:
			w := bufio.NewWriter(os.Stdout)
			err := stream(func(chunk []byte) { w.Write(chunk); w.Flush() })
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
		default:
			pcm, err := tok.Synthesize(units, *polish, *gain, *pitch)
			if err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			if err := os.WriteFile(*out, phonetok.WavBytes(pcm, rate), 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
				return 1
			}
			seconds := float64(len(pcm)) / 2 / float64(rate)
			emit(map[string]any{"path": *out, "seconds": seconds, "rate": rate, "bytes": len(pcm) + 44, "polish": *polish},
				[]string{fmt.Sprintf("%.2f s of speech written to %s (%d Hz)", seconds, *out, rate)})
		}
	case "codebook":
		path := ""
		if len(positional) > 0 {
			path = positional[0]
		}
		book, err := loadBook(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		if path == "" {
			path = "(bundled)"
		}
		a := book.Analysis
		total := 0
		for _, c := range book.Counts {
			total += c
		}
		lines := []string{
			fmt.Sprintf("%d units over %d log-mel bands (%g-%g Hz), %d ms frames every %d ms at %d Hz: %s", book.K(), a.Bands,
				a.Fmin, a.Fmax, a.Frame*1000/a.Rate, a.Hop*1000/a.Rate, a.Rate, path),
			fmt.Sprintf("learned from %d frames (%.1f s), seed %d, inertia %.1f", total, float64(total*a.Hop)/float64(a.Rate),
				book.Seed, book.Inertia),
		}
		if book.Note != "" {
			lines = append(lines, book.Note)
		}
		lines = append(lines, "", fmt.Sprintf("%-6s %7s %6s %5s", "unit", "frames", "share", "run"))
		for i := 0; i < book.K(); i++ {
			share := 0.0
			if total > 0 {
				share = float64(book.Counts[i]) / float64(total) * 100
			}
			lines = append(lines, fmt.Sprintf("%-6s %7d %5.1f%% %5.1f", book.Name(i), book.Counts[i], share, book.Runs[i]))
		}
		emit(map[string]any{"path": path, "units": book.K(), "frames": total, "seed": book.Seed, "inertia": book.Inertia,
			"note": book.Note, "analysis": map[string]any{"rate": a.Rate, "frame": a.Frame, "hop": a.Hop, "fft": a.FFT,
				"bands": a.Bands, "fmin": a.Fmin, "fmax": a.Fmax, "preemphasis": a.Preemphasis, "normalize": a.Normalize},
			"counts": book.Counts, "runs": book.Runs}, lines)
	case "eval":
		path := os.Getenv(phonetok.EnvLexicon)
		if path == "" {
			fmt.Fprintln(os.Stderr, "no full dictionary found: set PHONETOK_LEXICON")
			return 2
		}
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "phonetok: %v\n", err)
			return 1
		}
		defer f.Close()
		entries, _ := phonetok.ReadEntries(bufio.NewReader(f))
		words, right, rightStress, errors, total := 0, 0, 0, 0, 0
		seen := map[string]bool{}
		for _, e := range entries {
			if !isAlpha(e.Word) || seen[e.Word] {
				continue
			}
			seen[e.Word] = true
			if *limit > 0 && words >= *limit {
				break
			}
			got := phonetok.LetterToSound(e.Word)
			g0, s0 := phonetok.StripStress(e.Phones), phonetok.StripStress(got)
			words++
			if strings.Join(g0, " ") == strings.Join(s0, " ") {
				right++
			}
			if strings.Join(e.Phones, " ") == strings.Join(got, " ") {
				rightStress++
			}
			errors += editDistance(g0, s0)
			total += len(g0)
		}
		doc := map[string]any{"dictionary": "file " + path, "words": words, "word_accuracy": float64(right) / float64(max1(words)),
			"word_accuracy_with_stress": float64(rightStress) / float64(max1(words)), "phone_error_rate": float64(errors) / float64(max1(total))}
		emit(doc, []string{fmt.Sprintf("%d words of file %s: %.1f%% right, %.1f%% with stress, phone error rate %.1f%%", words, path,
			100*float64(right)/float64(max1(words)), 100*float64(rightStress)/float64(max1(words)), 100*float64(errors)/float64(max1(total)))})
	default:
		usage()
		return 2
	}
	return 0
}

// loadBook is the codebook at path, or the bundled one when path is empty.
func loadBook(path string) (*phonetok.Codebook, error) {
	if path == "" {
		return phonetok.DefaultCodebook()
	}
	return phonetok.LoadCodebook(path)
}

var valueGiven = map[string]bool{}

func takesValue(flag string) bool {
	switch strings.TrimLeft(flag, "-") {
	case "level", "lexicon", "limit", "count", "seed", "syllables", "out", "rate", "pitch", "tempo", "gain",
		"codebook", "units", "iterations", "note", "polish":
		return true
	}
	return false
}

func isNumber(s string) bool {
	_, err := strconv.Atoi(s)
	return err == nil
}

func isAlpha(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z') {
			return false
		}
	}
	return true
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

func editDistance(a, b []string) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			cur[j] = min3(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(b)]
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

type goRng struct{ r *rand.Rand }

func (g goRng) Float64() float64    { return g.r.Float64() }
func (g goRng) RandBelow(n int) int { return g.r.Intn(n) }

func newTokenizer(o options) (*phonetok.Tokenizer, error) {
	var lex *phonetok.Lexicon
	var err error
	switch {
	case o.lexicon != "":
		lex = phonetok.CoreLexicon()
		more, e := phonetok.LoadLexicon(o.lexicon)
		if e != nil {
			return nil, e
		}
		lex.Extend(more.Items())
		lex.Sources = append(lex.Sources, "file "+o.lexicon)
	case o.core:
		lex = phonetok.CoreLexicon()
	default:
		lex, err = phonetok.PortableLexicon()
		if err != nil {
			return nil, err
		}
	}
	tok, err := phonetok.NewTokenizer(o.level, lex)
	if err != nil {
		return nil, err
	}
	tok.Stress, tok.Boundaries, tok.PausesOn = !o.noStress, !o.noBoundaries, !o.noPauses
	return tok, nil
}
