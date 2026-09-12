// Package server is the HTTP API of the Go count / reward model: the same
// JSON contract as the Python server (DESIGN.md section 12) for everything
// the count model supports, so the prebuilt React frontend runs unchanged
// against it.
package server

import (
	"archive/zip"
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
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
func IsZip(data []byte) bool {
	return bytes.HasPrefix(data, []byte("PK\x03\x04")) || bytes.HasPrefix(data, []byte("PK\x05\x06")) || bytes.HasPrefix(data, []byte("PK\x07\x08"))
}

// Extracted is one text entry of an archive.
type Extracted struct {
	Name  string
	Path  string
	Text  string
	Bytes int
}

// Skipped is an archive entry that was not used, with the reason.
type Skipped struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

var archiveSuffixes = []string{".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar"}

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

func skipReason(f *zip.File) string {
	path := strings.ReplaceAll(f.Name, "\\", "/")
	base := path
	if i := strings.LastIndex(strings.TrimRight(path, "/"), "/"); i >= 0 {
		base = strings.TrimRight(path, "/")[i+1:]
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

// ExtractTexts returns the text entries of a ZIP archive and the skipped ones
// (same rules as the Python archive module: directories, macOS metadata,
// system files, nested archives, encrypted, binary and empty entries are skipped).
func ExtractTexts(data []byte, archiveName string) ([]Extracted, []Skipped, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, fmt.Errorf("not a valid ZIP archive (%v)", err)
	}
	var extracted []Extracted
	var skipped []Skipped
	for _, f := range zr.File {
		path := strings.ReplaceAll(f.Name, "\\", "/")
		if reason := skipReason(f); reason != "" {
			skipped = append(skipped, Skipped{path, reason})
			continue
		}
		rc, err := f.Open()
		if err != nil {
			skipped = append(skipped, Skipped{path, fmt.Sprintf("unreadable (%v)", err)})
			continue
		}
		payload, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			skipped = append(skipped, Skipped{path, fmt.Sprintf("unreadable (%v)", err)})
			continue
		}
		sniff := payload
		if len(sniff) > 8192 {
			sniff = sniff[:8192]
		}
		if bytes.IndexByte(sniff, 0) >= 0 {
			skipped = append(skipped, Skipped{path, "binary"})
			continue
		}
		text := decodeText(payload)
		if strings.TrimSpace(text) == "" {
			skipped = append(skipped, Skipped{path, "empty"})
			continue
		}
		extracted = append(extracted, Extracted{Name: FlatName(archiveName, path), Path: path, Text: text, Bytes: len(payload)})
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

func textRecord(path string) (map[string]any, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := strings.ToValidUTF8(string(raw), "�")
	return map[string]any{
		"name": filepath.Base(path), "bytes": info.Size(), "chars": len([]rune(text)), "lines": countLines(text),
		"modified": isoMillis(info.ModTime()),
	}, nil
}

func (u *Uploads) archiveInfo(path string, data []byte) archiveInfo {
	st, err := os.Stat(path)
	if err == nil {
		if cached, ok := u.cache[path]; ok && cached.size == st.Size() && cached.modTime.Equal(st.ModTime()) {
			return cached.info
		}
	}
	if data == nil {
		data, _ = os.ReadFile(path)
	}
	extracted, skipped, xerr := ExtractTexts(data, filepath.Base(path))
	info := summarize(extracted, skipped)
	if xerr != nil {
		info.Error = xerr.Error()
	}
	if st != nil {
		u.cache[path] = archiveCache{st.Size(), st.ModTime(), info}
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

// StoreBytes stores a binary upload: a ZIP archive is kept whole (and
// validated), anything else is stored as text.  Returns {"uploads", "archives"?}.
func (u *Uploads) StoreBytes(name string, data []byte) (map[string]any, error) {
	if !IsZip(data) {
		rec, err := u.StoreText(name, decodeText(data))
		if err != nil {
			return nil, err
		}
		return map[string]any{"uploads": []map[string]any{rec}}, nil
	}
	root, err := u.ensureDir()
	if err != nil {
		return nil, err
	}
	archiveName, err := SanitizeUploadName(name)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(strings.ToLower(archiveName), ".zip") {
		archiveName += ".zip"
	}
	extracted, skipped, xerr := ExtractTexts(data, archiveName)
	if xerr != nil {
		return nil, &apiError{400, fmt.Sprintf("%s: %v", archiveName, xerr)}
	}
	if len(extracted) == 0 {
		reasons := []string{}
		for i, s := range skipped {
			if i >= 8 {
				break
			}
			reasons = append(reasons, s.Path+": "+s.Reason)
		}
		msg := archiveName + " holds no text files to train on"
		if len(reasons) > 0 {
			msg += " (" + strings.Join(reasons, "; ") + ")"
		} else {
			msg += " (it is empty)"
		}
		return nil, &apiError{400, msg}
	}
	path := filepath.Join(root, archiveName)
	u.mu.Lock()
	defer u.mu.Unlock()
	_, statErr := os.Stat(path)
	replaced := statErr == nil
	if err := writeAtomic(path, data); err != nil {
		return nil, err
	}
	if st, err := os.Stat(path); err == nil {
		u.cache[path] = archiveCache{st.Size(), st.ModTime(), summarize(extracted, skipped)}
	}
	rec, err := u.record(path)
	if err != nil {
		return nil, err
	}
	rec["replaced"] = replaced
	skippedDocs := make([]Skipped, 0, len(skipped))
	skippedDocs = append(skippedDocs, skipped...)
	summary := map[string]any{"name": archiveName, "bytes": len(data), "entries": len(extracted) + len(skipped), "extracted": len(extracted), "skipped": skippedDocs}
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

// Entries returns (file name, text) pairs of an upload: one for a text file, one per text entry of a ZIP.
func (u *Uploads) Entries(name string) ([][2]string, error) {
	path, err := u.path(name)
	if err != nil {
		return nil, err
	}
	u.mu.Lock()
	data, readErr := os.ReadFile(path)
	u.mu.Unlock()
	if readErr != nil {
		return nil, &apiError{404, fmt.Sprintf("no upload named %q", filepath.Base(path))}
	}
	base := filepath.Base(path)
	if IsZip(data) {
		extracted, _, xerr := ExtractTexts(data, base)
		if xerr != nil {
			return nil, &apiError{400, fmt.Sprintf("%s: %v", base, xerr)}
		}
		out := make([][2]string, 0, len(extracted))
		for _, e := range extracted {
			out = append(out, [2]string{e.Path, e.Text})
		}
		return out, nil
	}
	return [][2]string{{base, decodeText(data)}}, nil
}

// Texts reads training texts from uploads: split per the unit (lines by
// default), or every file / archive entry as one text with wholeFile.
func (u *Uploads) Texts(names []string, unit string, pageLines int, wholeFile bool) ([]string, error) {
	var texts []string
	for _, name := range names {
		entries, err := u.Entries(name)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if wholeFile {
				if strings.TrimSpace(entry[1]) != "" {
					texts = append(texts, entry[1])
				}
				continue
			}
			texts = append(texts, splitTexts(entry[1], unit, pageLines)...)
		}
	}
	return texts, nil
}
