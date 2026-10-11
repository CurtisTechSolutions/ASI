package radixnet

import (
	"bytes"
	"encoding/base64"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// The base64 unit: anything - a text, an image, a video - as the n-grams of
// its base64 symbols.  These hold it to what it claims: the dial, the codec,
// how a file becomes texts, what a prediction decodes into, and a model
// trained on the symbols of real bytes, saved and read back.  Python's
// tests/test_base64_unit.py, case for case where the two overlap.

// b64Blob is a tiny binary blob that is not UTF-8: what an image or a
// recording looks like to the unit (the head of a PNG and some bytes that are
// no text).
var b64Blob = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1}

var b64Texts = []string{
	"the cat sat on the mat",
	"the cat sat on the floor",
	"the dog sat on the log",
	"a quick brown fox jumps over the lazy dog",
}

func TestBase64TheDial(t *testing.T) {
	for _, spec := range []string{"base64", "b64", "BASE64:3:1", " base64 "} {
		enc, err := ParseEncoding(spec)
		if err != nil || enc != (Encoding{Base64, 3, 1}) {
			t.Errorf("%q parsed as %v (%v), want base64:3:1", spec, enc, err)
		}
	}
	enc, err := ParseEncoding("base64:4:4")
	if err != nil || enc != (Encoding{Base64, 4, 4}) {
		t.Fatalf("base64:4:4 parsed as %v (%v)", enc, err)
	}
	if enc.String() != "base64:4:4" || enc.Describe() != "4-symbol grams, stride 4 (groups)" {
		t.Errorf("named %q, described %q", enc.String(), enc.Describe())
	}
	if got := (Encoding{Base64, 3, 1}).Describe(); got != "3-symbol grams, stride 1 (sliding)" {
		t.Errorf("described %q", got)
	}
	if enc.UnitsName() != "symbols" || Base64.Word() != "symbol" {
		t.Errorf("counted in %q, one unit a %q", enc.UnitsName(), Base64.Word())
	}
	if !enc.IsBase64() || enc.IsDefault() || DefaultEncoding().IsBase64() {
		t.Errorf("IsBase64 / IsDefault wrong for %v", enc)
	}
	if !Base64.Valid() || !Base64.Characters() || !Chars.Characters() || Words.Characters() || Phones.Characters() {
		t.Errorf("a base64 symbol is a valid unit handled as a character, and only characters are")
	}
	if err := enc.Validate(); err != nil {
		t.Errorf("base64:4:4 does not validate: %v", err)
	}
	_, err = ParseEncoding("base65")
	if err == nil || !strings.Contains(err.Error(), "base64") {
		t.Errorf("the error for an unknown unit must list base64, got %v", err)
	}
	if err := (Encoding{"rune", 3, 1}).Validate(); err == nil || !strings.Contains(err.Error(), "base64") {
		t.Errorf("Validate's error must list base64, got %v", err)
	}
}

// The symbols slice like characters: one code point, one unit - and nowhere
// does a base64 text pick up the space a word encoding would put in it.
func TestBase64ASymbolIsOneUnit(t *testing.T) {
	enc := Encoding{Base64, 3, 1}
	text := "dGhlIGNhdA"
	if u := enc.Units(text); u.Len() != 10 || u.Text() != text {
		t.Errorf("units of %q: %d, %q", text, u.Len(), u.Text())
	}
	if enc.Len(text) != 10 || enc.Slice(text, 2, 5) != "hlI" || enc.Truncate(text, 4) != "dGhl" {
		t.Errorf("Len / Slice / Truncate do not count symbols")
	}
	if got := enc.Join("dGh", "", "lIG"); got != "dGhlIG" {
		t.Errorf("Join = %q", got)
	}
	if !enc.HasUnitPrefix("dGhlIG", "dGh") || enc.HasUnitPrefix("dGhlIG", "dGx") {
		t.Errorf("HasUnitPrefix is not a plain prefix")
	}
	if got := enc.Reverse("abc+/"); got != "/+cba" {
		t.Errorf("Reverse = %q", got)
	}
	if got := enc.Encode("dGhlIA"); !reflect.DeepEqual(got, []string{"dGh", "Ghl", "hlI", "lIA"}) {
		t.Errorf("Encode = %q", got)
	}
	if got := enc.DecodePath([]string{"dGhl", "hlI", "lIA"}, 0, true); got != "dGhlIA" {
		t.Errorf("DecodePath = %q", got)
	}
	if got := (Encoding{Base64, 4, 4}).Encode("dGhlIGNhdA"); !reflect.DeepEqual(got, []string{"dGhl", "IGNh"}) {
		t.Errorf("groups of four = %q", got)
	}
	// the diff aligns symbols, as it aligns characters
	spans, _ := enc.ChangedSpans("dGhlIGNhdA", "dGhlIGRvZw")
	charSpans, _ := DefaultEncoding().ChangedSpans("dGhlIGNhdA", "dGhlIGRvZw")
	if !reflect.DeepEqual(spans, charSpans) || len(spans) == 0 {
		t.Errorf("the diff over symbols %v is not the diff over characters %v", spans, charSpans)
	}
}

func TestBase64BytesBecomeSymbolsWithoutPadding(t *testing.T) {
	for data, want := range map[string]string{"": "", "a": "YQ", "ab": "YWI", "abc": "YWJj", "the cat": "dGhlIGNhdA"} {
		if got := Base64Text([]byte(data)); got != want {
			t.Errorf("Base64Text(%q) = %q, want %q", data, got, want)
		}
	}
	every := make([]byte, 256)
	for i := range every {
		every[i] = byte(i)
	}
	if got, want := Base64Text(every), strings.TrimRight(base64.StdEncoding.EncodeToString(every), "="); got != want {
		t.Errorf("all 256 bytes: %q != %q", got, want)
	}
	if len(Base64Alphabet) != 64 {
		t.Fatalf("the alphabet holds %d symbols", len(Base64Alphabet))
	}
	for _, r := range Base64Text(b64Blob) {
		if !strings.ContainsRune(Base64Alphabet, r) {
			t.Errorf("%q is not a symbol of the alphabet", r)
		}
	}
}

func TestBase64RoundTripOfAnyBytes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	sizes := []int{255, 1000, 4097}
	for size := 0; size < 40; size++ {
		sizes = append(sizes, size)
	}
	for _, size := range sizes {
		data := make([]byte, size)
		rng.Read(data)
		symbols := Base64Text(data)
		if len(symbols) != (size*8+5)/6 {
			t.Errorf("%d bytes became %d symbols", size, len(symbols))
		}
		if back := Base64Bytes(symbols); !bytes.Equal(back, data) {
			t.Errorf("%d bytes did not come back: %x != %x", size, back, data)
		}
	}
}

// A predicted text always decodes: padding, whitespace and stray characters
// are skipped, a dangling symbol is dropped, and the bits a last partial group
// has over are ignored.
func TestBase64TheDecoderTakesWhateverWasWritten(t *testing.T) {
	cases := []struct{ text, want string }{
		{"", ""},
		{"YQ==", "a"},
		{"YW\nJj \t", "abc"},
		{"Y-W_J.j", "abc"}, // the URL-safe symbols are not this alphabet's
		{"YWJjZ", "abc"},   // one symbol cannot make a byte
		{"Y", ""},
		{"YR", "a"}, // R's low four bits are left over
		{"YWJ", "ab"},
		{"YéW\u2028Jj", "abc"}, // a character outside ASCII is a stray character too
	}
	for _, c := range cases {
		if got := Base64Bytes(c.text); string(got) != c.want {
			t.Errorf("Base64Bytes(%q) = %q, want %q", c.text, got, c.want)
		}
	}
}

func TestBase64APrefixIsTheSymbolsItsBytesFix(t *testing.T) {
	for data, want := range map[string]string{"": "", "t": "d", "th": "dG", "the": "dGhl", "the d": "dGhlIG", "the cat": "dGhlIGNhd"} {
		if got := Base64Prefix([]byte(data)); got != want {
			t.Errorf("Base64Prefix(%q) = %q, want %q", data, got, want)
		}
	}
	rng := rand.New(rand.NewSource(3))
	for size := 0; size < 30; size++ {
		data := make([]byte, size)
		rng.Read(data)
		prefix := Base64Prefix(data)
		if len(prefix) != size*8/6 {
			t.Errorf("%d bytes fix %d symbols", size, len(prefix))
		}
		for _, tail := range [][]byte{nil, {0x00}, {0xFF}, []byte("xyz")} { // every continuation of the bytes shares it
			if whole := Base64Text(append(append([]byte{}, data...), tail...)); !strings.HasPrefix(whole, prefix) {
				t.Errorf("%x + %x = %q does not start with %q", data, tail, whole, prefix)
			}
		}
	}
}

func TestBase64FilesBecomeTexts(t *testing.T) {
	enc := func(s string) string { return Base64Text([]byte(s)) }
	cases := []struct {
		data  string
		whole bool
		want  []string
	}{
		// a text file is one text per line: \r and a BOM stay, blank lines go
		{"one\r\n  \ntwo\n\n\tthree \n", false, []string{enc("one\r"), enc("two"), enc("\tthree ")}},
		{"\ufeffone\ntwo", false, []string{enc("\ufeffone"), enc("two")}},
		{"\x0b\x0c\n\u00a0\n", false, []string{enc("\u00a0")}}, // only ASCII whitespace is blank
		// whole, every byte is kept
		{"one\ntwo\n\n", true, []string{enc("one\ntwo\n\n")}},
		{" \n\t\n", true, []string{enc(" \n\t\n")}},
		// nothing comes from nothing
		{"", false, nil},
		{"", true, nil},
		{" \n\t\n", false, nil},
		// anything that is not UTF-8 is one text of every byte, lines or not
		{string(b64Blob), false, []string{Base64Text(b64Blob)}},
		{"ok\n\xed\xa0\x80\n", false, []string{enc("ok\n\xed\xa0\x80\n")}}, // a surrogate is not UTF-8
	}
	for _, c := range cases {
		if got := Base64FileTexts([]byte(c.data), c.whole); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Base64FileTexts(%q, %v) = %q, want %q", c.data, c.whole, got, c.want)
		}
	}
	lines := Base64FileTexts([]byte("one\r\n  \ntwo\n\n\tthree \n"), false)
	var back []string
	for _, line := range lines {
		back = append(back, string(Base64Bytes(line)))
	}
	if !reflect.DeepEqual(back, []string{"one\r", "two", "\tthree "}) {
		t.Errorf("the lines decode to %q", back)
	}
	if got := Base64Bytes(Base64FileTexts(b64Blob, false)[0]); !bytes.Equal(got, b64Blob) {
		t.Errorf("the blob came back as %x", got)
	}
}

// Base64Source streams from disk exactly the texts Base64FileTexts makes of
// the file's bytes: a line longer than any read buffer, a character cut by
// one, and binary data included.
func TestBase64SourceStreamsWhatTheFileGives(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	binary := make([]byte, 300<<10)
	rng.Read(binary)
	long := strings.Repeat("é€😀 the cat ", 40<<10) // past the 64 KiB check and the 256 KiB reader
	contents := [][]byte{
		nil,
		[]byte("one\r\n  \ntwo\n\n\tthree \n"),
		[]byte("\ufeffno newline at the end"),
		[]byte(long + "\n" + long + "\nend"),
		b64Blob,
		binary,
		append([]byte(long), 0xFF),       // text until its very last byte
		append([]byte(long), 0xE2, 0x82), // a character cut off at the end
		[]byte(" \n\t\n"),
	}
	dir := t.TempDir()
	for i, data := range contents {
		path := filepath.Join(dir, "file")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		for _, whole := range []bool{false, true} {
			got, err := CollectTexts(Base64Source{Path: path, WholeFile: whole})
			if err != nil {
				t.Fatalf("case %d: %v", i, err)
			}
			if want := Base64FileTexts(data, whole); !reflect.DeepEqual(got, want) {
				t.Errorf("case %d (whole %v): the source gave %d texts, the bytes %d", i, whole, len(got), len(want))
			}
		}
	}
	if _, err := CollectTexts(Base64Source{Path: filepath.Join(dir, "missing")}); err == nil {
		t.Errorf("a missing file must be an error")
	}
}

// The check reads any split of the bytes the way utf8.Valid reads them whole.
func TestUTF8CheckAgreesWithUTF8Valid(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	pieces := []string{"a", "é", "€", "😀", "\xed\xa0\x80", "\xf0\x90\x80", "\xc0\xaf", "\xf4\x90\x80\x80", "\xff", "\xe2\x82", "\x80", "\ufffd", " "}
	for round := 0; round < 3000; round++ {
		var data []byte
		for n := rng.Intn(12); n > 0; n-- {
			data = append(data, pieces[rng.Intn(len(pieces))]...)
		}
		var check UTF8Check
		for rest := data; len(rest) > 0; {
			k := 1 + rng.Intn(min(len(rest), 5))
			check.Write(rest[:k])
			rest = rest[k:]
		}
		if check.Valid() != utf8.Valid(data) {
			t.Fatalf("%q: the check says %v, utf8.Valid %v", data, check.Valid(), utf8.Valid(data))
		}
		if check.Failed() && utf8.Valid(data) {
			t.Fatalf("%q: failed on text", data)
		}
	}
}

// One U+FFFD per maximal invalid subpart: the Unicode Standard's practice,
// Python's bytes.decode("utf-8", "replace") and Rust's from_utf8_lossy.
func TestLossyUTF8IsTheStandardsPractice(t *testing.T) {
	cases := []struct{ data, want string }{
		{"ok\xe2\x82A\xff", "ok\ufffdA\ufffd"},           // a truncated sequence is one subpart
		{"\xed\xa0\x80", "\ufffd\ufffd\ufffd"},           // a surrogate: three
		{"\xf0\x90\x80", "\ufffd"},                       // a truncated four-byte sequence at the end
		{"\xc0\xaf", "\ufffd\ufffd"},                     // an overlong form: two
		{"\xf4\x90\x80\x80", "\ufffd\ufffd\ufffd\ufffd"}, // past U+10FFFF: four
		{"\xe2\x82\xac", "€"},
		{"\xe0\x80", "\ufffd\ufffd"},
		{"\xe0\xa0", "\ufffd"},
		{"\xf5\x80", "\ufffd\ufffd"},
		{"a\xf0\x9f\x98b", "a\ufffdb"},
		{"\xef\xbf\xbd", "\ufffd"}, // U+FFFD itself is a character
		{"\xed\x9f\xbf", "\ud7ff"},
		{"\xf4\x8f\xbf\xbf", "\U0010ffff"},
		{"\xc2", "\ufffd"},
		{"\xc2A", "\ufffdA"},
		{"\x80\x80", "\ufffd\ufffd"},
		{"\xe1\x80\xe1\x80\x80", "\ufffd\u1000"},
		{"\xf1\x80\x80\xc1", "\ufffd\ufffd"},
		{"", ""},
	}
	for _, c := range cases {
		if got := LossyUTF8([]byte(c.data)); got != c.want {
			t.Errorf("LossyUTF8(%q) = %q, want %q", c.data, got, c.want)
		}
	}
}

func TestBase64WhatComesInAndWhatGoesOut(t *testing.T) {
	b64, plain := Encoding{Base64, 3, 1}, DefaultEncoding()
	if b64.TextInput("the cat") != "dGhlIGNhdA" || b64.PrefixInput("the cat") != "dGhlIGNhd" {
		t.Errorf("a text comes in as %q, a prefix as %q", b64.TextInput("the cat"), b64.PrefixInput("the cat"))
	}
	if got, err := b64.ContentText(b64Blob); err != nil || got != Base64Text(b64Blob) {
		t.Errorf("the blob came in as %q (%v)", got, err)
	}
	if !bytes.Equal(b64.ContentBytes(Base64Text(b64Blob)), b64Blob) {
		t.Errorf("the blob did not go out as itself")
	}
	// every other unit takes text as it is
	if plain.TextInput("the cat") != "the cat" || plain.PrefixInput("the cat") != "the cat" {
		t.Errorf("a character model converted a text")
	}
	if got, err := plain.ContentText([]byte("café")); err != nil || got != "café" {
		t.Errorf("café came in as %q (%v)", got, err)
	}
	if _, err := plain.ContentText(b64Blob); err == nil || !strings.Contains(err.Error(), "offset 0") {
		t.Errorf("bytes that are not text must be refused where they stop being text, got %v", err)
	}
	if string(plain.ContentBytes("café")) != "café" {
		t.Errorf("a character model's text did not go out as its UTF-8")
	}
}

func TestBase64APredictionDecodes(t *testing.T) {
	b64 := Encoding{Base64, 3, 1}
	got := DecodedPrediction(b64, "dGhlIGNhdCBzYXQ", []byte("the cat"))
	want := map[string]any{"decoded": "the cat sat", "decoded_continuation": " sat", "decoded_bytes": 11}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	// no prefix: all of it was written
	if got := DecodedPrediction(b64, "dGhlIGNhdA", nil)["decoded_continuation"]; got != "the cat" {
		t.Errorf("the whole text is the continuation, got %q", got)
	}
	// every other encoding adds nothing
	if got := DecodedPrediction(DefaultEncoding(), "the cat sat", []byte("the cat")); len(got) != 0 {
		t.Errorf("a character model decoded %v", got)
	}
	// asked "the d", the symbols fix four bytes and half the fifth; a walk that finishes it as an l said
	// "the log", and the continuation is everything after the bytes the two share
	prefix := Base64Prefix([]byte("the d"))
	if prefix != "dGhlIG" {
		t.Fatalf("the prefix is %q", prefix)
	}
	got = DecodedPrediction(b64, prefix+"xvZw", []byte("the d"))
	if got["decoded"] != "the log" || got["decoded_continuation"] != "log" {
		t.Errorf("the d -> %v", got)
	}
	got = DecodedPrediction(b64, Base64Text([]byte("the dog")), []byte("the d"))
	if got["decoded"] != "the dog" || got["decoded_continuation"] != "og" {
		t.Errorf("the dog -> %v", got)
	}
	// bytes that are not text read as replacement characters, one per maximal invalid subpart
	for data, text := range map[string]string{"ok\xe2\x82A\xff": "ok\ufffdA\ufffd", "\xed\xa0\x80": "\ufffd\ufffd\ufffd", "\xe2\x82\xac": "€"} {
		if got := DecodedPrediction(b64, Base64Text([]byte(data)), nil)["decoded"]; got != text {
			t.Errorf("%q decoded as %q, want %q", data, got, text)
		}
	}
}

// Every symbol is a label like any other: a model of base64 symbols learns
// the bytes of texts and of a binary file, says them back, scores them, and
// survives a save.
func TestBase64AModelLearnsTheSymbols(t *testing.T) {
	enc := Encoding{Base64, 3, 1}
	var train []string
	for _, text := range b64Texts {
		train = append(train, enc.TextInput(text))
	}
	train = append(train, Base64Text(b64Blob))
	m := modelWith(t, enc, train, 2)
	stats := m.Stats()
	if stats["encoding"] != "base64:3:1" || stats["units"] != "symbols" {
		t.Errorf("stats: encoding %v, units %v", stats["encoding"], stats["units"])
	}
	if _, ok := stats["vocabulary"]; ok {
		t.Errorf("a model of symbols has no vocabulary of words")
	}
	if m.TopWords(0) != nil {
		t.Errorf("a model of symbols has no words to list")
	}
	for _, label := range m.G.Labels[First:] {
		for _, r := range label {
			if !strings.ContainsRune(Base64Alphabet, r) {
				t.Fatalf("label %q is not base64 symbols", label)
			}
		}
	}
	if err := m.G.CheckInvariants(train, false); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "b64.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"encoding":{"unit":"base64","n":3,"stride":1}`) {
		t.Errorf("the file does not carry the encoding block")
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Encoding() != enc || loaded.G.NumGrams() != m.G.NumGrams() {
		t.Fatalf("loaded as %v with %d grams", loaded.Encoding(), loaded.G.NumGrams())
	}
	// asked the symbols "the cat sat on the" fixes, it says one of the sentences back
	asked := enc.PrefixInput("the cat sat on the")
	out, err := loaded.Predict(asked, PredictOptions{Length: 8, Mode: "beam", K: 2, ToEnd: true, MaxLength: -1})
	if err != nil {
		t.Fatal(err)
	}
	said := DecodedPrediction(enc, out.FullText, []byte("the cat sat on the"))
	if said["decoded"] != "the cat sat on the mat" && said["decoded"] != "the cat sat on the floor" {
		t.Errorf("said %v", said)
	}
	// from START a model of one file walks that file: the decoded walk is the file again
	one := modelWith(t, enc, []string{Base64Text(b64Blob)}, 2)
	out, err = one.Predict("", PredictOptions{Length: 1, Mode: "beam", K: 1, ToEnd: true, MaxLength: -1})
	if err != nil {
		t.Fatal(err)
	}
	if got := Base64Bytes(out.FullText); !bytes.Equal(got, b64Blob) {
		t.Errorf("the file came back as %x", got)
	}
	// a score reads the bytes
	known := m.Score(enc.TextInput(b64Texts[0]))
	noise := m.Score(enc.TextInput("qzx wqzj vbn qzx wqzj vbn"))
	if known.UnknownTransitions != 0 || noise.UnknownTransitions == 0 {
		t.Errorf("known %+v, noise %+v", known, noise)
	}
}
