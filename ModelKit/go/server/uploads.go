// Package server is the HTTP API of the Go count / reward model: the same
// JSON contract as the Python server (DESIGN.md section 12) for everything
// the count model supports, so the prebuilt React frontend runs unchanged
// against it.
package server

import (
	"archive/zip"
	"bufio"
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

const maxUploadName = 128

var uploadNameChars = "._- +@()"

// SanitizeUploadName reduces a name to a safe base name inside the upload directory.
func SanitizeUploadName(name string) (string, error) {
	base := strings.TrimSpace(filepath.Base(strings.ReplaceAll(name, "\\", "/")))
	var b strings.Builder
	for _, ch := range base {
		if unicode.IsLetter(ch) || unicode.IsDigit(ch) || strings.ContainsRune(uploadNameChars, ch) {
			b.WriteRune(ch)
		} else {
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), " .")
	if out == "" || out == "." || out == ".." {
		return "", &apiError{400, fmt.Sprintf("invalid upload name %q", name)}
	}
	if len(out) > maxUploadName {
		out = out[:maxUploadName]
	}
	return out, nil
}

// IsZip reports whether data starts with a ZIP signature (the bytes decide, not the name).
func IsZip(data []byte) bool { return radixnet.IsZipMagic(data) }

// Extracted is one text entry of an archive held in memory (small JSON uploads).
type Extracted struct {
	Name  string
	Path  string
	Text  string
	Bytes int
}

// Skipped is an archive entry that was not used, with the reason.
type Skipped = radixnet.ZipSkip

// FlatName is <archive stem>__<dir>__<file>: a flat, safe name for an entry.
func FlatName(archiveName, entryPath string) string {
	stem := strings.TrimSpace(filepath.Base(strings.ReplaceAll(archiveName, "\\", "/")))
	if strings.HasSuffix(strings.ToLower(stem), ".zip") {
		stem = stem[:len(stem)-4]
	}
	pieces := []string{safePiece(stem)}
	for _, p := range strings.Split(strings.ReplaceAll(entryPath, "\\", "/"), "/") {
		if p != "" && p != "." && p != ".." {
			pieces = append(pieces, safePiece(p))
		}
	}
	kept := pieces[:0]
	for _, p := range pieces {
		if p != "" {
			kept = append(kept, p)
		}
	}
	name := strings.Join(kept, "__")
	if name == "" {
		name = "entry"
	}
	if len(name) > maxUploadName {
		ext := filepath.Ext(name)
		if len(ext) > 16 {
			ext = ext[:16]
		}
		root := strings.TrimSuffix(name, ext)
		sum := sha1.Sum([]byte(archiveName + "/" + entryPath))
		digest := hex.EncodeToString(sum[:])[:8]
		cut := maxUploadName - len(ext) - 9
		if cut > len(root) {
			cut = len(root)
		}
		name = root[:cut] + "-" + digest + ext
	}
	return name
}

func safePiece(text string) string {
	var b strings.Builder
	for _, ch := range text {
		if unicode.IsLetter(ch) || unicode.IsDigit(ch) || strings.ContainsRune(uploadNameChars, ch) {
			b.WriteRune(ch)
		} else {
			b.WriteRune('_')
		}
	}
	return strings.Trim(b.String(), " .")
}

// ExtractTexts returns the text entries of a ZIP archive held in memory (the
// JSON upload forms); large archives take the streaming path instead.
func ExtractTexts(data []byte, archiveName string) ([]Extracted, []Skipped, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, fmt.Errorf("not a valid ZIP archive (%v)", err)
	}
	var extracted []Extracted
	var empty []string
	skipped, err := radixnet.WalkZip(zr, func(path string, r io.Reader) error {
		payload, rerr := io.ReadAll(r)
		if rerr != nil {
			return rerr
		}
		text := decodeText(payload)
		if strings.TrimSpace(text) == "" {
			empty = append(empty, path)
			return nil
		}
		extracted = append(extracted, Extracted{Name: FlatName(archiveName, path), Path: path, Text: text, Bytes: len(payload)})
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	for _, p := range empty {
		skipped = append(skipped, Skipped{Path: p, Reason: "empty"})
	}
	return extracted, skipped, nil
}

// decodeText turns uploaded bytes into text: UTF-8 with the BOM dropped, invalid bytes replaced.
func decodeText(data []byte) string {
	data = bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
	return strings.ToValidUTF8(string(data), "�")
}

func countLines(text string) int {
	n := 0
	for _, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}

// Uploads manages the directory of training files (text files and ZIP archives kept whole).
type Uploads struct {
	Dir   string
	mu    sync.Mutex
	cache map[string]archiveCache
}

type archiveCache struct {
	size    int64
	modTime time.Time
	info    archiveInfo
}

type archiveInfo struct {
	Files   int
	Entries []map[string]any
	Skipped []Skipped
	Lines   int
	Chars   int
	Error   string
}

// NewUploads creates the manager (the directory is created on first use).
func NewUploads(dir string) *Uploads { return &Uploads{Dir: dir, cache: map[string]archiveCache{}} }

func (u *Uploads) ensureDir() (string, error) {
	if u == nil || u.Dir == "" {
		return "", &apiError{400, "uploads are disabled: start the server with --upload-dir"}
	}
	if err := os.MkdirAll(u.Dir, 0o755); err != nil {
		return "", err
	}
	return u.Dir, nil
}

func (u *Uploads) path(name string) (string, error) {
	root, err := u.ensureDir()
	if err != nil {
		return "", err
	}
	safe, err := SanitizeUploadName(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(root, safe), nil
}

func isoMillis(t time.Time) string { return t.UTC().Format("2006-01-02T15:04:05.000-07:00") }

// textRecord describes a text upload; lines and characters are counted by streaming the file.
func textRecord(path string) (map[string]any, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	lines, chars := 0, 0
	br := bufio.NewReaderSize(fh, 256<<10)
	for {
		line, rerr := br.ReadString('\n')
		if len(line) > 0 {
			chars += len([]rune(line))
			if strings.TrimSpace(line) != "" {
				lines++
			}
		}
		if rerr != nil {
			break
		}
	}
	return map[string]any{
		"name": filepath.Base(path), "bytes": info.Size(), "chars": chars, "lines": lines, "modified": isoMillis(info.ModTime()),
	}, nil
}

func (u *Uploads) archiveInfo(path string, data []byte) archiveInfo {
	st, err := os.Stat(path)
	if err == nil {
		if cached, ok := u.cache[path]; ok && cached.size == st.Size() && cached.modTime.Equal(st.ModTime()) {
			return cached.info
		}
	}
	info := inspectArchiveFile(path)
	if st != nil {
		u.cache[path] = archiveCache{st.Size(), st.ModTime(), info}
	}
	return info
}

// inspectArchiveFile streams an archive on disk: entry by entry, line by line,
// so a multi-gigabyte upload is described without being held in memory.
func inspectArchiveFile(path string) archiveInfo {
	zr, err := radixnet.OpenZip(path)
	if err != nil {
		return archiveInfo{Entries: []map[string]any{}, Skipped: []Skipped{}, Error: err.Error()}
	}
	defer zr.Close()
	zi, err := radixnet.InspectZip(&zr.Reader)
	if err != nil {
		return archiveInfo{Entries: []map[string]any{}, Skipped: []Skipped{}, Error: err.Error()}
	}
	info := archiveInfo{Entries: []map[string]any{}, Skipped: zi.Skipped, Lines: zi.Lines, Chars: zi.Chars, Files: len(zi.Entries)}
	for _, e := range zi.Entries {
		info.Entries = append(info.Entries, map[string]any{"path": e.Path, "bytes": e.Bytes, "lines": e.Lines})
	}
	return info
}

func summarize(extracted []Extracted, skipped []Skipped) archiveInfo {
	info := archiveInfo{Skipped: skipped, Entries: []map[string]any{}}
	if info.Skipped == nil {
		info.Skipped = []Skipped{}
	}
	for _, e := range extracted {
		lines := countLines(e.Text)
		info.Entries = append(info.Entries, map[string]any{"path": e.Path, "bytes": e.Bytes, "lines": lines})
		info.Lines += lines
		info.Chars += len([]rune(e.Text))
	}
	info.Files = len(extracted)
	return info
}

func isArchiveFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 4)
	n, _ := io.ReadFull(f, head)
	return IsZip(head[:n])
}

func (u *Uploads) record(path string) (map[string]any, error) {
	if !isArchiveFile(path) {
		return textRecord(path)
	}
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	info := u.archiveInfo(path, nil)
	rec := map[string]any{
		"name": filepath.Base(path), "bytes": st.Size(), "chars": info.Chars, "lines": info.Lines,
		"modified": isoMillis(st.ModTime()), "archive": true, "files": info.Files, "skipped": len(info.Skipped),
	}
	if info.Error != "" {
		rec["error"] = info.Error
	}
	return rec, nil
}

// List describes every upload.
func (u *Uploads) List() (map[string]any, error) {
	root, err := u.ensureDir()
	if err != nil {
		return nil, err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	names := []string{}
	for _, e := range entries {
		if !e.IsDir() && !strings.HasSuffix(e.Name(), ".part") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	records := []map[string]any{}
	for _, name := range names {
		rec, err := u.record(filepath.Join(root, name))
		if err == nil {
			records = append(records, rec)
		}
	}
	return map[string]any{"uploads": records, "upload_dir": root}, nil
}

func writeAtomic(path string, data []byte) error {
	part := path + ".part"
	if err := os.WriteFile(part, data, 0o644); err != nil {
		return err
	}
	return os.Rename(part, path)
}

// StoreText stores content as the upload name (replacing an existing one).
func (u *Uploads) StoreText(name, content string) (map[string]any, error) {
	path, err := u.path(name)
	if err != nil {
		return nil, err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	_, statErr := os.Stat(path)
	replaced := statErr == nil
	if err := writeAtomic(path, []byte(content)); err != nil {
		return nil, err
	}
	rec, err := textRecord(path)
	if err != nil {
		return nil, err
	}
	rec["replaced"] = replaced
	return rec, nil
}

// StoreBytes stores a binary upload held in memory (the JSON forms): a ZIP
// archive is kept whole, anything else is stored as text.
func (u *Uploads) StoreBytes(name string, data []byte) (map[string]any, error) {
	return u.StoreStream(name, bytes.NewReader(data))
}

// StoreStream stores an upload from a stream without holding it in memory:
// the bytes go straight to a .part file; the first four decide whether it is
// a ZIP archive, which is then validated by streaming its entries (at least
// one text entry, or 400) and kept whole under a .zip name.  Anything else is
// a text file.  Returns {"uploads": [record], "archives": [summary]?}.
func (u *Uploads) StoreStream(name string, r io.Reader) (map[string]any, error) {
	root, err := u.ensureDir()
	if err != nil {
		return nil, err
	}
	safe, err := SanitizeUploadName(name)
	if err != nil {
		return nil, err
	}
	part, err := os.CreateTemp(root, ".upload-*.part")
	if err != nil {
		return nil, err
	}
	partName := part.Name()
	cleanup := func() { _ = os.Remove(partName) }
	head := make([]byte, 4)
	n, herr := io.ReadFull(r, head)
	if herr != nil && herr != io.EOF && herr != io.ErrUnexpectedEOF {
		part.Close()
		cleanup()
		return nil, herr
	}
	head = head[:n]
	size := int64(n)
	if _, err := part.Write(head); err != nil {
		part.Close()
		cleanup()
		return nil, err
	}
	if herr == nil {
		copied, err := io.Copy(part, r)
		if err != nil {
			part.Close()
			cleanup()
			return nil, err
		}
		size += copied
	}
	if err := part.Close(); err != nil {
		cleanup()
		return nil, err
	}
	if !IsZip(head) {
		// a text file: stored as it is (lines are repaired when read)
		path := filepath.Join(root, safe)
		u.mu.Lock()
		defer u.mu.Unlock()
		_, statErr := os.Stat(path)
		if err := os.Rename(partName, path); err != nil {
			cleanup()
			return nil, err
		}
		rec, err := textRecord(path)
		if err != nil {
			return nil, err
		}
		rec["replaced"] = statErr == nil
		return map[string]any{"uploads": []map[string]any{rec}}, nil
	}
	archiveName := safe
	if !strings.HasSuffix(strings.ToLower(archiveName), ".zip") {
		archiveName += ".zip"
	}
	info := inspectArchiveFile(partName)
	if info.Error != "" {
		cleanup()
		return nil, &apiError{400, fmt.Sprintf("%s: %s", archiveName, info.Error)}
	}
	if info.Files == 0 {
		reasons := []string{}
		for i, sk := range info.Skipped {
			if i >= 8 {
				break
			}
			reasons = append(reasons, sk.Path+": "+sk.Reason)
		}
		msg := archiveName + " holds no text files to train on"
		if len(reasons) > 0 {
			msg += " (" + strings.Join(reasons, "; ") + ")"
		} else {
			msg += " (it is empty)"
		}
		cleanup()
		return nil, &apiError{400, msg}
	}
	path := filepath.Join(root, archiveName)
	u.mu.Lock()
	defer u.mu.Unlock()
	_, statErr := os.Stat(path)
	if err := os.Rename(partName, path); err != nil {
		cleanup()
		return nil, err
	}
	if st, err := os.Stat(path); err == nil {
		u.cache[path] = archiveCache{st.Size(), st.ModTime(), info}
	}
	rec, err := u.record(path)
	if err != nil {
		return nil, err
	}
	rec["replaced"] = statErr == nil
	skipped := make([]Skipped, 0, len(info.Skipped))
	skipped = append(skipped, info.Skipped...)
	summary := map[string]any{"name": archiveName, "bytes": size, "entries": info.Files + len(info.Skipped), "extracted": info.Files, "skipped": skipped}
	return map[string]any{"uploads": []map[string]any{rec}, "archives": []map[string]any{summary}}, nil
}

// Delete removes an upload.
func (u *Uploads) Delete(name string) (map[string]any, error) {
	path, err := u.path(name)
	if err != nil {
		return nil, err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	if _, err := os.Stat(path); err != nil {
		return nil, &apiError{404, fmt.Sprintf("no upload named %q", filepath.Base(path))}
	}
	if err := os.Remove(path); err != nil {
		return nil, err
	}
	delete(u.cache, path)
	return map[string]any{"deleted": filepath.Base(path)}, nil
}

// Source is a streaming text source over uploads: a ZIP archive streams its
// text entries, a text file its lines; unit cuts them into texts (wholeFile:
// every file / entry is one text).  Nothing is held in memory.
func (u *Uploads) Source(names []string, unit string, pageLines int, wholeFile bool) (radixnet.TextSource, error) {
	if wholeFile {
		unit = "file"
	}
	var sources radixnet.MultiSource
	for _, name := range names {
		path, err := u.path(name)
		if err != nil {
			return nil, err
		}
		if _, err := os.Stat(path); err != nil {
			return nil, &apiError{404, fmt.Sprintf("no upload named %q", filepath.Base(path))}
		}
		sources = append(sources, radixnet.SourceForFile(path, unit, pageLines))
	}
	return sources, nil
}

// Texts collects the texts of uploads (for the small rated sets of feedback / 2NRL).
func (u *Uploads) Texts(names []string, unit string, pageLines int, wholeFile bool) ([]string, error) {
	src, err := u.Source(names, unit, pageLines, wholeFile)
	if err != nil {
		return nil, err
	}
	texts, err := radixnet.CollectTexts(src)
	if err != nil {
		return nil, wrapSourceError(err)
	}
	return texts, nil
}

// wrapSourceError turns a streaming failure into a client error (a corrupt archive is a 400).
func wrapSourceError(err error) error {
	var ae *apiError
	if errors.As(err, &ae) {
		return err
	}
	return &apiError{400, err.Error()}
}
