package radixnet

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
)

// The Go collector runs when the heap has grown to about twice what is live
// (GOGC=100), so a training run whose live graph is a gigabyte reaches two
// before it collects - on a container with a hard limit that is an OOM kill
// rather than a garbage collection.  A soft memory limit makes the collector
// work harder as the heap approaches the limit instead, which is what keeps a
// long streaming pass over a huge archive inside the memory it was given.

// DefaultMemoryFraction is how much of the machine's (or the container's)
// memory a process gives its heap before the collector runs continuously.
const DefaultMemoryFraction = 0.8

// MemoryLimit is the soft limit in force, and where it came from.
type MemoryLimit struct {
	Bytes  int64
	Source string // "cgroup", "meminfo", "GOMEMLIMIT", "flag" or "" when none
}

// String describes the limit for a log line.
func (m MemoryLimit) String() string {
	if m.Bytes <= 0 {
		return "no memory limit"
	}
	return fmt.Sprintf("%s soft memory limit (%s)", FormatBytes(m.Bytes), m.Source)
}

// ApplyMemoryLimit sets the soft heap limit: want bytes when positive, else a
// fraction of the memory the process may use (its cgroup limit, or the
// machine's RAM).  GOMEMLIMIT from the environment always wins, and a want of
// -1 turns the limit off.
func ApplyMemoryLimit(want int64, fraction float64) MemoryLimit {
	if env := strings.TrimSpace(os.Getenv("GOMEMLIMIT")); env != "" {
		return MemoryLimit{Bytes: debug.SetMemoryLimit(-1), Source: "GOMEMLIMIT"}
	}
	if want < 0 {
		return MemoryLimit{}
	}
	if want > 0 {
		debug.SetMemoryLimit(want)
		return MemoryLimit{Bytes: want, Source: "flag"}
	}
	total, source := AvailableMemory()
	if total <= 0 {
		return MemoryLimit{}
	}
	if fraction <= 0 || fraction > 1 {
		fraction = DefaultMemoryFraction
	}
	limit := int64(float64(total) * fraction)
	if limit < 64<<20 {
		limit = 64 << 20
	}
	debug.SetMemoryLimit(limit)
	return MemoryLimit{Bytes: limit, Source: source}
}

// AvailableMemory is the memory the process may use: its cgroup limit when it
// has one, else the machine's RAM.
func AvailableMemory() (int64, string) {
	if n := cgroupLimit(); n > 0 {
		if total := memTotal(); total <= 0 || n < total {
			return n, "cgroup"
		}
	}
	if n := memTotal(); n > 0 {
		return n, "meminfo"
	}
	return 0, ""
}

// cgroupLimit is the smallest memory limit the process's cgroup carries
// (v2 and v1, the process's own path and the root of the hierarchy - a
// container may see either, and nested groups take the tighter one).
func cgroupLimit() int64 {
	best := int64(0)
	for _, path := range cgroupLimitFiles() {
		n := readCgroupLimit(path)
		if n > 0 && (best == 0 || n < best) {
			best = n
		}
	}
	return best
}

// cgroupLimitFiles are the files that may hold this process's memory limit.
func cgroupLimitFiles() []string {
	files := []string{"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}
	raw, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return files
	}
	for _, line := range strings.Split(string(raw), "\n") {
		// "0::/path" (v2) or "<id>:<controllers>:/path" (v1)
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) != 3 || parts[2] == "" || parts[2] == "/" {
			continue
		}
		rel := strings.TrimPrefix(parts[2], "/")
		if parts[1] == "" {
			files = append(files, filepath.Join("/sys/fs/cgroup", rel, "memory.max"))
			continue
		}
		for _, controller := range strings.Split(parts[1], ",") {
			if controller == "memory" {
				files = append(files, filepath.Join("/sys/fs/cgroup/memory", rel, "memory.limit_in_bytes"))
			}
		}
	}
	return files
}

// readCgroupLimit reads one limit file ("max" and the unlimited sentinel are no limit).
func readCgroupLimit(path string) int64 {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	text := strings.TrimSpace(string(raw))
	if text == "max" {
		return 0
	}
	n, err := strconv.ParseInt(text, 10, 64)
	if err != nil || n <= 0 || n >= 1<<62 {
		return 0
	}
	return n
}

// memTotal is what /proc/meminfo says the process may use: MemAvailable when
// the kernel reports it (the machine's RAM minus what other processes hold and
// what the page cache cannot give back), else MemTotal.
func memTotal() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	var total, available int64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		name, ok := strings.CutSuffix(strings.Fields(line)[0], ":")
		if !ok || (name != "MemTotal" && name != "MemAvailable") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		if name == "MemTotal" {
			total = kb * 1024
		} else {
			available = kb * 1024
		}
	}
	if available > 0 && (total == 0 || available < total) {
		return available
	}
	return total
}

// ParseSize reads a byte size: a plain number, or one with a K / M / G / T
// suffix (powers of 1024; "kb", "mib" and so on are accepted too).
func ParseSize(text string) (int64, error) {
	s := strings.TrimSpace(strings.ToLower(text))
	if s == "" {
		return 0, nil
	}
	if s == "off" || s == "none" {
		return -1, nil
	}
	s = strings.TrimSuffix(strings.TrimSuffix(s, "b"), "i")
	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "k"):
		mult = 1 << 10
	case strings.HasSuffix(s, "m"):
		mult = 1 << 20
	case strings.HasSuffix(s, "g"):
		mult = 1 << 30
	case strings.HasSuffix(s, "t"):
		mult = 1 << 40
	}
	if mult > 1 {
		s = s[:len(s)-1]
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, fmt.Errorf("cannot read %q as a byte size", text)
	}
	if value < 0 {
		return -1, nil
	}
	return int64(value * float64(mult)), nil
}

// FormatBytes renders a byte count like 1.5 GiB.
func FormatBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KiB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
