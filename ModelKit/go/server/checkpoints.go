package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

const (
	ckptPrefix = "ckpt-"
	ckptLatest = "latest.json"
	ckptIndex  = "index.json"
)

var (
	ckptSuffixes = []string{".json.gz", ".json"}
	tagRe        = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_\-]*$`)
	ckptNameRe   = regexp.MustCompile(`^ckpt-([A-Za-z0-9][A-Za-z0-9_\-]*)-(\d{6,})\.json(?:\.gz)?$`)
)

// Checkpoints is a checkpoint directory in the Python CheckpointManager's
// layout: ckpt-<tag>-<step:06d>.json.gz files, latest.json and index.json.
type Checkpoints struct {
	Dir      string
	Keep     int
	Compress bool
	mu       sync.Mutex
}

// NewCheckpoints creates the manager (keep < 1 means 5).
func NewCheckpoints(dir string, keep int) *Checkpoints {
	if keep < 1 {
		keep = 5
	}
	return &Checkpoints{Dir: dir, Keep: keep, Compress: true}
}

func (c *Checkpoints) path(name string) string { return filepath.Join(c.Dir, name) }

func (c *Checkpoints) readJSON(name string) map[string]any {
	raw, err := os.ReadFile(c.path(name))
	if err != nil {
		return nil
	}
	var doc map[string]any
	if json.Unmarshal(raw, &doc) != nil {
		return nil
	}
	return doc
}

func (c *Checkpoints) writeJSON(name string, doc any) error {
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(c.path(name), raw)
}

func (c *Checkpoints) recordFromFile(name string) map[string]any {
	m := ckptNameRe.FindStringSubmatch(name)
	if m == nil {
		return nil
	}
	st, err := os.Stat(c.path(name))
	if err != nil {
		return nil
	}
	var step int
	fmt.Sscanf(m[2], "%d", &step)
	return map[string]any{
		"name": name, "path": c.path(name), "step": step, "tag": m[1], "metrics": nil,
		"saved_at": st.ModTime().UTC().Format("2006-01-02T15:04:05.000000-07:00"), "bytes": st.Size(),
	}
}

func (c *Checkpoints) index() map[string]map[string]any {
	stored := c.readJSON(ckptIndex)
	index := map[string]map[string]any{}
	changed := false
	for name, raw := range stored {
		rec, ok := raw.(map[string]any)
		if ok {
			if _, err := os.Stat(c.path(name)); err == nil {
				rec["path"] = c.path(name)
				index[name] = rec
				continue
			}
		}
		changed = true
	}
	entries, _ := os.ReadDir(c.Dir)
	for _, e := range entries {
		name := e.Name()
		if _, ok := index[name]; ok || !strings.HasPrefix(name, ckptPrefix) {
			continue
		}
		if !strings.HasSuffix(name, ".json") && !strings.HasSuffix(name, ".json.gz") {
			continue
		}
		if rec := c.recordFromFile(name); rec != nil {
			index[name] = rec
			changed = true
		}
	}
	if changed {
		_ = c.writeJSON(ckptIndex, index)
	}
	return index
}

func sortKey(rec map[string]any) (int, string, string) {
	step := 0
	switch v := rec["step"].(type) {
	case float64:
		step = int(v)
	case int:
		step = v
	}
	savedAt, _ := rec["saved_at"].(string)
	name, _ := rec["name"].(string)
	return step, savedAt, name
}

func sortedRecords(index map[string]map[string]any) []map[string]any {
	records := make([]map[string]any, 0, len(index))
	for _, r := range index {
		records = append(records, r)
	}
	sort.Slice(records, func(i, j int) bool {
		si, ai, ni := sortKey(records[i])
		sj, aj, nj := sortKey(records[j])
		if si != sj {
			return si < sj
		}
		if ai != aj {
			return ai < aj
		}
		return ni < nj
	})
	return records
}

func (c *Checkpoints) prune(index map[string]map[string]any, protect string) {
	records := sortedRecords(index)
	for len(records) > c.Keep {
		victim := -1
		for i, r := range records {
			if r["name"] != protect {
				victim = i
				break
			}
		}
		if victim < 0 {
			break
		}
		name, _ := records[victim]["name"].(string)
		records = append(records[:victim], records[victim+1:]...)
		delete(index, name)
		_ = os.Remove(c.path(name))
	}
}

// Save writes ckpt-<tag>-<step:06d>.json.gz, updates latest.json and prunes.
func (c *Checkpoints) Save(m *radixnet.Model, step int, tag string, metrics map[string]any) (map[string]any, error) {
	if !tagRe.MatchString(tag) {
		return nil, &apiError{400, fmt.Sprintf("invalid checkpoint tag %q (letters, digits, '_' and '-' only)", tag)}
	}
	if step < 0 {
		return nil, &apiError{400, fmt.Sprintf("step must be >= 0, got %d", step)}
	}
	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return nil, err
	}
	suffix := ".json"
	if c.Compress {
		suffix = ".json.gz"
	}
	name := fmt.Sprintf("%s%s-%06d%s", ckptPrefix, tag, step, suffix)
	path := c.path(name)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := m.Save(path); err != nil {
		return nil, err
	}
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	var metricsAny any
	if metrics != nil {
		metricsAny = metrics
	}
	record := map[string]any{
		"name": name, "path": path, "step": step, "tag": tag, "metrics": metricsAny,
		"saved_at": time.Now().UTC().Format("2006-01-02T15:04:05.000000-07:00"), "bytes": st.Size(),
	}
	index := c.index()
	index[name] = record
	if err := c.writeJSON(ckptLatest, record); err != nil {
		return nil, err
	}
	c.prune(index, name)
	if err := c.writeJSON(ckptIndex, index); err != nil {
		return nil, err
	}
	return record, nil
}

// List returns every checkpoint record sorted by (step, saved_at).
func (c *Checkpoints) List() []map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	return sortedRecords(c.index())
}

// Latest is the record latest.json points at (nil when absent or its file is gone).
func (c *Checkpoints) Latest() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	rec := c.readJSON(ckptLatest)
	if rec == nil {
		return nil
	}
	name, ok := rec["name"].(string)
	if !ok {
		return nil
	}
	if _, err := os.Stat(c.path(name)); err != nil {
		return nil
	}
	rec["path"] = c.path(name)
	return rec
}

// Resolve finds a checkpoint by record name, bare stem or path.
func (c *Checkpoints) Resolve(nameOrPath string) (string, error) {
	stem := nameOrPath
	for _, suffix := range ckptSuffixes {
		if strings.HasSuffix(stem, suffix) {
			stem = strings.TrimSuffix(stem, suffix)
			break
		}
	}
	candidates := []string{nameOrPath, c.path(nameOrPath)}
	for _, suffix := range ckptSuffixes {
		candidates = append(candidates, c.path(stem+suffix))
	}
	for _, cand := range candidates {
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			abs, err := filepath.Abs(cand)
			if err != nil {
				return cand, nil
			}
			return abs, nil
		}
	}
	return "", &apiError{404, fmt.Sprintf("no checkpoint %q in %s", nameOrPath, c.Dir)}
}
