package radixnet

import (
	"archive/zip"
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// A TextSource streams the texts of a corpus in order.  Training re-reads it
// for every pass, so a source must be re-openable; nothing needs to hold the
// whole corpus in memory - a massive ZIP archive streams entry by entry, line
// by line, and the model consumes it in chunks (TrainOptions.ChunkSize).
type TextSource interface {
	Each(fn func(text string) error) error
}

// errStop ends a streaming walk early without reporting an error.
var errStop = errors.New("stop")

// bom is the UTF-8 byte order mark some editors put in front of a text file.
var bom = string([]byte{0xEF, 0xBB, 0xBF})

// SliceSource is a corpus already in memory.
type SliceSource []string

// Each yields the texts.
func (s SliceSource) Each(fn func(string) error) error {
	for _, t := range s {
		if err := fn(t); err != nil {
			return err
		}
	}
	return nil
}

// MultiSource concatenates sources in order.
type MultiSource []TextSource

// Each yields every source's texts in turn.
func (m MultiSource) Each(fn func(string) error) error {
	for _, src := range m {
		if err := src.Each(fn); err != nil {
			return err
		}
	}
	return nil
}

// FuncSource adapts a function to TextSource.
type FuncSource func(fn func(string) error) error

// Each calls the function.
func (f FuncSource) Each(fn func(string) error) error { return f(fn) }

// CollectTexts drains a source into a slice (for small corpora and the callers that need a list).
func CollectTexts(src TextSource) ([]string, error) {
	var out []string
	err := src.Each(func(t string) error {
		out = append(out, t)
		return nil
	})
	return out, err
}

// CountTexts counts the texts of a source without keeping them.
func CountTexts(src TextSource) (int, error) {
	n := 0
	err := src.Each(func(string) error { n++; return nil })
	return n, err
}

// -- splitting a stream of lines into texts -----------------------------------------------------------------

// splitter turns lines into texts the way SplitTexts does, but streaming:
// lines (one text per non-blank line), paragraphs (blank-line separated,
// joined with spaces), pages (form feeds, or every pageLines lines) and file
// (everything as one text).
type splitter struct {
	unit      string
	pageLines int
	emit      func(string) error
	block     []string
	lines     int
	sawFF     bool
	all       strings.Builder
	any       bool
}

func normalizeUnit(unit string) string {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "", "line", "lines":
		return "lines"
	case "paragraph", "paragraphs":
		return "paragraphs"
	case "page", "pages":
		return "pages"
	case "file", "whole", "whole-file":
		return "file"
	}
	return "lines"
}

func newSplitter(unit string, pageLines int, emit func(string) error) *splitter {
	if pageLines <= 0 {
		pageLines = 50
	}
	return &splitter{unit: normalizeUnit(unit), pageLines: pageLines, emit: emit}
}

func (s *splitter) flushBlock() error {
	if len(s.block) == 0 {
		return nil
	}
	text := strings.Join(s.block, " ")
	s.block = s.block[:0]
	return s.emit(text)
}

// line feeds one line (without its terminator).
func (s *splitter) line(line string) error {
	switch s.unit {
	case "lines":
		if strings.TrimSpace(line) != "" {
			return s.emit(line)
		}
		return nil
	case "paragraphs":
		if strings.TrimSpace(line) == "" {
			return s.flushBlock()
		}
		s.block = append(s.block, strings.TrimSpace(line))
		return nil
	case "pages":
		if strings.Contains(line, "\f") {
			s.sawFF = true
			parts := strings.Split(line, "\f")
			for i, part := range parts {
				if i > 0 {
					if err := s.flushBlock(); err != nil {
						return err
					}
					s.lines = 0
				}
				if strings.TrimSpace(part) != "" {
					s.block = append(s.block, strings.TrimSpace(part))
				}
			}
			return nil
		}
		if strings.TrimSpace(line) != "" {
			s.block = append(s.block, strings.TrimSpace(line))
		}
		s.lines++
		if !s.sawFF && s.lines >= s.pageLines {
			s.lines = 0
			return s.flushBlock()
		}
		return nil
	default: // file
		if s.any {
			s.all.WriteByte('\n')
		}
		s.all.WriteString(line)
		s.any = true
		return nil
	}
}

// finish emits what is still buffered.
func (s *splitter) finish() error {
	switch s.unit {
	case "lines":
		return nil
	case "paragraphs", "pages":
		return s.flushBlock()
	default:
		if !s.any {
			return nil
		}
		text := strings.TrimRight(s.all.String(), "\r\n")
		s.all.Reset()
		s.any = false
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return s.emit(text)
	}
}

// streamLines reads r line by line (any line length, CRLF tolerated, BOM and
// invalid UTF-8 repaired) and feeds the splitter; memory stays O(one line).
func streamLines(r io.Reader, unit string, pageLines int, emit func(string) error) error {
	sp := newSplitter(unit, pageLines, emit)
	br := bufio.NewReaderSize(r, 256<<10)
	first := true
	for {
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			if first {
				line = strings.TrimPrefix(line, bom)
				first = false
			}
			line = strings.TrimRight(line, "\r\n")
			if !isValidUTF8(line) {
				line = strings.ToValidUTF8(line, "�")
			}
			if lerr := sp.line(line); lerr != nil {
				return lerr
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return sp.finish()
}

func isValidUTF8(s string) bool {
	for i := 0; i < len(s); {
		if s[i] < 0x80 {
			i++
			continue
		}
		return strings.ToValidUTF8(s, "�") == s
	}
	return true
}

// StreamSplit splits the content of r into texts (streaming version of SplitTexts).
func StreamSplit(r io.Reader, unit string, pageLines int, emit func(string) error) error {
	return streamLines(r, unit, pageLines, emit)
}

// FileSource streams a text file.
type FileSource struct {
	Path      string
	Unit      string
	PageLines int
}

// Each streams the file's texts.
func (f FileSource) Each(fn func(string) error) error {
	fh, err := os.Open(f.Path)
	if err != nil {
		return err
	}
	defer fh.Close()
	return streamLines(fh, f.Unit, f.PageLines, fn)
}

// -- ZIP archives ----------------------------------------------------------------------------------------------

// ZipSkip is an archive entry that was not used, with the reason.
type ZipSkip struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

var zipMagic = [][]byte{[]byte("PK\x03\x04"), []byte("PK\x05\x06"), []byte("PK\x07\x08")}

// IsZipMagic reports whether data starts with a ZIP signature.
func IsZipMagic(data []byte) bool {
	for _, m := range zipMagic {
		if bytes.HasPrefix(data, m) {
			return true
		}
	}
	return false
}

// IsZipFile reports whether the file starts with a ZIP signature (reads 4 bytes).
func IsZipFile(path string) bool {
	fh, err := os.Open(path)
	if err != nil {
		return false
	}
	defer fh.Close()
	head := make([]byte, 4)
	n, _ := io.ReadFull(fh, head)
	return IsZipMagic(head[:n])
}

var archiveSuffixes = []string{".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar"}

// ZipSkipReason classifies an entry that must not be read as text: directories,
// macOS metadata, system files, nested archives and encrypted entries.
func ZipSkipReason(f *zip.File) string {
	path := strings.ReplaceAll(f.Name, "\\", "/")
	trimmed := strings.TrimRight(path, "/")
	base := trimmed
	if i := strings.LastIndex(trimmed, "/"); i >= 0 {
		base = trimmed[i+1:]
	}
	if f.FileInfo().IsDir() || strings.HasSuffix(path, "/") {
		return "directory"
	}
	if strings.HasPrefix(path, "__MACOSX/") || strings.Contains(path, "/__MACOSX/") {
		return "macOS metadata"
	}
	if strings.HasPrefix(base, "._") || base == ".DS_Store" || base == "Thumbs.db" {
		return "system file"
	}
	lower := strings.ToLower(base)
	for _, suffix := range archiveSuffixes {
		if strings.HasSuffix(lower, suffix) {
			return "nested archive"
		}
	}
	if f.Flags&0x1 != 0 {
		return "encrypted"
	}
	return ""
}

const sniffBytes = 8192

// openZipText opens an entry as a text stream: the first 8 KB are sniffed for
// NUL bytes (a binary entry) and replayed in front of the rest.
func openZipText(f *zip.File) (io.ReadCloser, string, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, fmt.Sprintf("unreadable (%v)", err), nil
	}
	head := make([]byte, sniffBytes)
	n, err := io.ReadFull(rc, head)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		rc.Close()
		return nil, fmt.Sprintf("unreadable (%v)", err), nil
	}
	head = head[:n]
	if bytes.IndexByte(head, 0) >= 0 {
		rc.Close()
		return nil, "binary", nil
	}
	return &replayReader{Reader: io.MultiReader(bytes.NewReader(head), rc), closer: rc}, "", nil
}

type replayReader struct {
	io.Reader
	closer io.Closer
}

func (r *replayReader) Close() error { return r.closer.Close() }

// WalkZip streams every text entry of an archive to fn (in archive order) and
// returns the skipped entries.  fn gets the entry path and a reader over its
// bytes; returning errStop ends the walk early.
func WalkZip(zr *zip.Reader, fn func(path string, r io.Reader) error) ([]ZipSkip, error) {
	var skipped []ZipSkip
	for _, f := range zr.File {
		path := strings.ReplaceAll(f.Name, "\\", "/")
		if reason := ZipSkipReason(f); reason != "" {
			skipped = append(skipped, ZipSkip{path, reason})
			continue
		}
		rc, reason, err := openZipText(f)
		if err != nil {
			return skipped, err
		}
		if reason != "" {
			skipped = append(skipped, ZipSkip{path, reason})
			continue
		}
		ferr := fn(path, rc)
		rc.Close()
		if ferr != nil {
			if errors.Is(ferr, errStop) {
				return skipped, nil
			}
			return skipped, ferr
		}
	}
	return skipped, nil
}

// OpenZip opens an archive on disk (the central directory only; entries stream on demand).
func OpenZip(path string) (*zip.ReadCloser, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("not a valid ZIP archive (%v)", err)
	}
	return zr, nil
}

// ZipEntryInfo describes one text entry of an archive.
type ZipEntryInfo struct {
	Path  string `json:"path"`
	Bytes int64  `json:"bytes"`
	Lines int    `json:"lines"`
	Chars int    `json:"chars"`
}

// ZipInfo is what an archive holds, computed by streaming (no entry is held in memory).
type ZipInfo struct {
	Entries []ZipEntryInfo
	Skipped []ZipSkip
	Lines   int
	Chars   int
}

// InspectZip streams every text entry, counting non-blank lines and
// characters; entries without any non-blank line are reported as empty.
func InspectZip(zr *zip.Reader) (*ZipInfo, error) {
	info := &ZipInfo{Entries: []ZipEntryInfo{}}
	var emptyPaths []string
	skipped, err := WalkZip(zr, func(path string, r io.Reader) error {
		entry := ZipEntryInfo{Path: path}
		br := bufio.NewReaderSize(r, 256<<10)
		for {
			line, rerr := br.ReadString('\n')
			if len(line) > 0 {
				entry.Bytes += int64(len(line))
				entry.Chars += len([]rune(strings.TrimRight(line, "\r\n")))
				if strings.TrimSpace(line) != "" {
					entry.Lines++
				}
				if strings.HasSuffix(line, "\n") {
					entry.Chars++ // Python counts the newline character too
				}
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				return rerr
			}
		}
		if entry.Lines == 0 {
			emptyPaths = append(emptyPaths, path)
			return nil
		}
		info.Entries = append(info.Entries, entry)
		info.Lines += entry.Lines
		info.Chars += entry.Chars
		return nil
	})
	if err != nil {
		return nil, err
	}
	info.Skipped = skipped
	for _, p := range emptyPaths {
		info.Skipped = append(info.Skipped, ZipSkip{p, "empty"})
	}
	if info.Skipped == nil {
		info.Skipped = []ZipSkip{}
	}
	return info, nil
}

// ZipSource streams the texts of a ZIP archive on disk: entry by entry, line
// by line, cut into texts per Unit ("file" makes every entry one text).
type ZipSource struct {
	Path      string
	Unit      string
	PageLines int
}

// Each streams the archive's texts.
func (z ZipSource) Each(fn func(string) error) error {
	zr, err := OpenZip(z.Path)
	if err != nil {
		return err
	}
	defer zr.Close()
	_, err = WalkZip(&zr.Reader, func(_ string, r io.Reader) error {
		return streamLines(r, z.Unit, z.PageLines, fn)
	})
	return err
}

// SourceForFile picks a streaming source for a path: ZipSource for an archive (by
// its magic bytes), FileSource otherwise.
func SourceForFile(path, unit string, pageLines int) TextSource {
	if IsZipFile(path) {
		return ZipSource{Path: path, Unit: unit, PageLines: pageLines}
	}
	return FileSource{Path: path, Unit: unit, PageLines: pageLines}
}
