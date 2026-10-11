package radixnet

// The base64 unit: anything - a text, an image, a video, a recording - as the
// n-grams of its base64 symbols.
//
// A model of base64 symbols (`--encoding base64:3:1`) reads bytes, not text: a
// text is its UTF-8 bytes, a file the bytes it holds, and either is written in
// the 64 symbols of RFC 4648's standard alphabet before the graph sees it
// (Base64Text).  From there on nothing is new.  A symbol is one code point, so
// the graph slices, joins, diffs and reverses symbols exactly as it does
// characters (UnitKind.Characters), and every dial of the encoding means what
// it always meant: base64:4:4 is one gram per three bytes.  What the model says
// is symbols too, decoded back into bytes on the way out (Base64Bytes,
// DecodedPrediction).
//
// The conversion happens once, at the boundary - where a data file, an inline
// text or a prefix comes in, and where an answer goes out - and never inside
// the graph or Units: a text of the graph is already base64.  The Go twin of
// the base64 half of Python's radixnet/encoding.py, byte for byte.

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Base64Alphabet is the 64 symbols of the base64 unit: RFC 4648's standard
// alphabet, in its order.
const Base64Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

// -- the codec ---------------------------------------------------------------------

// Base64Text is bytes as base64 symbols: the standard alphabet, no padding (3
// bytes -> 4 symbols, 1 -> 2, 2 -> 3).  It is the encoder of the base64 unit:
// a text is its UTF-8 bytes, a file the bytes it holds, and either becomes the
// text a model over base64 symbols is built from.
func Base64Text(data []byte) string { return base64.RawStdEncoding.EncodeToString(data) }

// Base64Prefix is the symbols every continuation of data shares: the first
// floor(8 * len(data) / 6) of its base64.  The last symbol of a text's base64
// also holds bits of the bytes that would follow it, so a prefix given as
// content is asked as the symbols it fixes - "the c" as dGhlIG - and the walk
// writes on from there, the bits of the c that the next symbol carries
// included.
func Base64Prefix(data []byte) string { return Base64Text(data)[:len(data)*8/6] }

// Base64Bytes is base64 symbols back to bytes, whatever was written: the
// decoder of the base64 unit, and it never fails.  Anything outside the 64
// symbols is skipped (padding, whitespace, a stray character, the URL-safe -
// and _), a last symbol that cannot complete a byte is dropped, and the bits a
// final partial group has over are ignored - so a predicted or generated text
// always decodes, into floor(6 * symbols / 8) bytes.
func Base64Bytes(text string) []byte {
	clean := make([]byte, 0, len(text))
	for i := 0; i < len(text); i++ {
		if isBase64Symbol(text[i]) {
			clean = append(clean, text[i])
		}
	}
	if len(clean)%4 == 1 { // six bits cannot make a byte
		clean = clean[:len(clean)-1]
	}
	out := make([]byte, base64.RawStdEncoding.DecodedLen(len(clean)))
	// the decoder is not strict, so the bits a last group has over are dropped:
	// symbols of the alphabet in any count but 4k + 1 always decode
	n, _ := base64.RawStdEncoding.Decode(out, clean)
	return out[:n]
}

// isBase64Symbol reports whether c is one of the 64 symbols.
func isBase64Symbol(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/'
}

// Base64FileTexts is the texts a file gives a model of base64 symbols, each in
// its symbols (Base64Text).
//
// A file that reads as UTF-8 text gives one text per line - split on \n
// alone, each line's bytes as they are (a \r stays, so does a byte order
// mark), a line of nothing but ASCII whitespace skipped - unless wholeFile.
// Anything else (an image, a video, a recording, an archive), and any file
// taken whole, is one text: every byte it holds, so the decoded text is the
// file again.  An empty file gives nothing.  Python's base64_file_texts;
// Base64Source streams the same texts from a file on disk.
func Base64FileTexts(data []byte, wholeFile bool) []string {
	if len(data) == 0 {
		return nil
	}
	if wholeFile || !utf8.Valid(data) {
		return []string{Base64Text(data)}
	}
	var texts []string
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		if !asciiBlank(line) {
			texts = append(texts, Base64Text(line))
		}
	}
	return texts
}

// asciiBlank reports whether a line holds nothing but ASCII whitespace (space,
// \t, \n, \r, \v, \f): the lines a text file's base64 leaves out.  Only these
// six, as Python strips them, so a line of a no-break space is a line.
func asciiBlank(line []byte) bool {
	for _, c := range line {
		switch c {
		case ' ', '\t', '\n', '\r', '\v', '\f':
		default:
			return false
		}
	}
	return true
}

// -- the encoding at the boundary --------------------------------------------------

// IsBase64 reports whether the units are base64 symbols - six bits of the
// bytes of any content (Base64) - so that what comes in is converted and what
// goes out is decoded.
func (e Encoding) IsBase64() bool { return e.Unit == Base64 }

// TextInput is a text given in plain words - a sentence to train on, to score
// or to blame - as this encoding reads it: base64 writes its UTF-8 bytes in
// symbols, and every other unit takes it as it is.
func (e Encoding) TextInput(text string) string {
	if e.IsBase64() {
		return Base64Text([]byte(text))
	}
	return text
}

// PrefixInput is a prefix given in plain words, as this encoding asks it:
// base64 asks the symbols its UTF-8 bytes fix (Base64Prefix), so the walk
// writes on from the last whole symbol, and every other unit takes it as it
// is.
func (e Encoding) PrefixInput(text string) string {
	if e.IsBase64() {
		return Base64Prefix([]byte(text))
	}
	return text
}

// ContentText is content - the bytes of a file - as the text this encoding
// reads: base64 writes the bytes in its symbols, whatever they hold, and every
// other unit reads them as UTF-8 text, so bytes that are not text are an error
// there.
func (e Encoding) ContentText(data []byte) (string, error) {
	if e.IsBase64() {
		return Base64Text(data), nil
	}
	if err := checkUTF8(data); err != nil {
		return "", err
	}
	return string(data), nil
}

// ContentBytes is what a text of this encoding holds, as bytes: base64 symbols
// decoded (Base64Bytes), any other text as its UTF-8.  It is the output
// decoder `--decode-to` writes with.
func (e Encoding) ContentBytes(text string) []byte {
	if e.IsBase64() {
		return Base64Bytes(text)
	}
	return []byte(text)
}

// checkUTF8 says where bytes stop being UTF-8 text, or nil when they never do.
func checkUTF8(data []byte) error {
	for i := 0; i < len(data); {
		r, size := utf8.DecodeRune(data[i:])
		if r == utf8.RuneError && size == 1 {
			return fmt.Errorf("byte %#02x at offset %d does not read as UTF-8", data[i], i)
		}
		i += size
	}
	return nil
}

// -- what the model says, as the bytes it holds ------------------------------------

// DecodedPrediction is what a text of a model of base64 symbols holds:
// {"decoded", "decoded_continuation", "decoded_bytes"}.
//
// decoded is the whole text - the prefix and what the walk wrote after it -
// decoded (Base64Bytes) and read as UTF-8, a byte sequence that is not text
// showing as U+FFFD (LossyUTF8); decoded_bytes is how many bytes that is.
// decoded_continuation is what the model wrote that prefix - the content the
// walk was asked to continue, as bytes - did not already hold: everything
// after the bytes the two have in common.  A prefix that ends inside a symbol
// leaves its last byte to the walk, which may write it differently (asked
// "the d" it may say "the log"), so the continuation starts where the model's
// bytes and the prefix's part.  Every other encoding gets an empty map, as
// anything but sounds does from SpelledPrediction.  Python's
// decoded_prediction.
func DecodedPrediction(enc Encoding, fullText string, prefix []byte) map[string]any {
	if !enc.IsBase64() {
		return map[string]any{}
	}
	whole := Base64Bytes(fullText)
	same := 0
	for same < len(whole) && same < len(prefix) && whole[same] == prefix[same] {
		same++
	}
	return map[string]any{
		"decoded": LossyUTF8(whole), "decoded_continuation": LossyUTF8(whole[same:]), "decoded_bytes": len(whole),
	}
}

// -- bytes that are text, and bytes that are not -----------------------------------

// LossyUTF8 reads bytes as UTF-8 text the way Python's
// bytes.decode("utf-8", "replace") does: a byte sequence that is not text
// becomes U+FFFD, one per *maximal invalid subpart* - the longest start of a
// well-formed sequence (the Unicode Standard's Table 3-7) before it goes
// wrong, or the one byte that starts none.  That is the Standard's own
// practice, and Rust's String::from_utf8_lossy.  Go's own two ways count
// differently: strings.ToValidUTF8 writes one U+FFFD per run of bad bytes (a
// surrogate, \xed\xa0\x80, is one where the Standard says three), and decoding
// rune by rune one per byte (a cut-off \xf0\x90\x80 is three where it says
// one).  So a decoded answer, and an upload a text model reads, says the same
// in every port.
func LossyUTF8(data []byte) string {
	if utf8.Valid(data) {
		return string(data)
	}
	var b strings.Builder
	b.Grow(len(data) + len(data)/2)
	for i := 0; i < len(data); {
		r, size := utf8.DecodeRune(data[i:])
		if r != utf8.RuneError || size > 1 { // a whole character, U+FFFD itself included
			b.Write(data[i : i+size])
			i += size
			continue
		}
		b.WriteRune(utf8.RuneError)
		i += maximalSubpart(data[i:])
	}
	return b.String()
}

// maximalSubpart is how many bytes of data - which does not start with a whole
// character - one U+FFFD stands for: the lead byte and the continuation bytes
// after it that still fit a well-formed sequence (Table 3-7: E0 is followed by
// A0..BF, ED by 80..9F, F0 by 90..BF, F4 by 80..8F, and every other
// continuation byte is 80..BF), or the one byte when it can start no sequence
// at all (80..C1, F5..FF).
func maximalSubpart(data []byte) int {
	lo, hi, size := byte(0x80), byte(0xBF), 0
	switch c := data[0]; {
	case 0xC2 <= c && c <= 0xDF:
		size = 2
	case c == 0xE0:
		size, lo = 3, 0xA0
	case c == 0xED:
		size, hi = 3, 0x9F
	case 0xE1 <= c && c <= 0xEF:
		size = 3
	case c == 0xF0:
		size, lo = 4, 0x90
	case c == 0xF4:
		size, hi = 4, 0x8F
	case 0xF1 <= c && c <= 0xF3:
		size = 4
	default:
		return 1
	}
	n := 1
	for n < size && n < len(data) && lo <= data[n] && data[n] <= hi {
		n++
		lo, hi = 0x80, 0xBF // only the byte after the lead has a range of its own
	}
	return n
}

// UTF8Check reads the bytes written through it as UTF-8, one piece at a time:
// what decides whether a file or an upload is text without holding all of it.
// A character the pieces cut in two is put back together, so after any
// sequence of writes Valid is what utf8.Valid would say of all of them at
// once - strict, as Python's decoder is: a surrogate or an overlong form is
// not text.  Write never fails, so a check can sit in an io.MultiWriter beside
// the file it is checking.
type UTF8Check struct {
	tail [utf8.UTFMax - 1]byte // the start of a character the last piece ended inside
	held int
	bad  bool
}

// Write reads the next piece.
func (c *UTF8Check) Write(p []byte) (int, error) {
	n := len(p)
	if c.bad {
		return n, nil
	}
	if c.held > 0 { // finish the character the last piece cut off
		var buf [2 * utf8.UTFMax]byte
		k := copy(buf[:], c.tail[:c.held])
		k += copy(buf[k:], p[:min(len(p), utf8.UTFMax-1)])
		if !utf8.FullRune(buf[:k]) { // still cut off: this piece was too short to finish it
			c.held = copy(c.tail[:], buf[:k])
			return n, nil
		}
		r, size := utf8.DecodeRune(buf[:k])
		if r == utf8.RuneError && size == 1 {
			c.bad = true
			return n, nil
		}
		p = p[size-c.held:]
		c.held = 0
	}
	cut := len(p)
	for i := len(p) - 1; i >= 0 && i >= len(p)-(utf8.UTFMax-1); i-- {
		if utf8.RuneStart(p[i]) {
			if !utf8.FullRune(p[i:]) { // a character the next piece may finish
				cut = i
			}
			break
		}
	}
	if !utf8.Valid(p[:cut]) {
		c.bad = true
		return n, nil
	}
	c.held = copy(c.tail[:], p[cut:])
	return n, nil
}

// Valid reports whether everything written is UTF-8 text; a character still
// cut off at the end is not.
func (c *UTF8Check) Valid() bool { return !c.bad && c.held == 0 }

// Failed reports whether a byte that cannot be UTF-8 has been seen: whatever
// follows, the bytes are not text, so a reader may stop.
func (c *UTF8Check) Failed() bool { return c.bad }
