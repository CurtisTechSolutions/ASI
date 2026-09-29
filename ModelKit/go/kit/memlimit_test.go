package kit

import (
	"runtime/debug"
	"strings"
	"testing"
)

// A soft memory limit is the difference between a collection and an OOM kill.
func TestParseSizeAndMemoryLimit(t *testing.T) {
	cases := map[string]int64{"": 0, "1024": 1024, "2k": 2 << 10, "3MiB": 3 << 20, "1.5g": 1536 << 20, "off": -1, "none": -1}
	for text, want := range cases {
		got, err := ParseSize(text)
		if err != nil || got != want {
			t.Fatalf("ParseSize(%q) = %d, %v; want %d", text, got, err, want)
		}
	}
	if _, err := ParseSize("later"); err == nil {
		t.Fatal("ParseSize must reject a non-size")
	}
	if n, source := AvailableMemory(); n <= 0 || source == "" {
		t.Fatalf("AvailableMemory() = %d, %q", n, source)
	}
	before := debug.SetMemoryLimit(-1)
	defer debug.SetMemoryLimit(before)
	if got := ApplyMemoryLimit(512<<20, DefaultMemoryFraction); got.Bytes != 512<<20 || got.Source != "flag" {
		t.Fatalf("explicit limit: %+v", got)
	}
	if got := debug.SetMemoryLimit(-1); got != 512<<20 {
		t.Fatalf("limit not applied: %d", got)
	}
	if got := ApplyMemoryLimit(-1, DefaultMemoryFraction); got.Bytes != 0 || got.String() != "no memory limit" {
		t.Fatalf("off: %+v", got)
	}
	auto := ApplyMemoryLimit(0, DefaultMemoryFraction)
	if auto.Bytes <= 0 || auto.Source == "" || !strings.Contains(auto.String(), "soft memory limit") {
		t.Fatalf("auto limit: %+v", auto)
	}
	if total, _ := AvailableMemory(); auto.Bytes > total {
		t.Fatalf("auto limit %d above the available %d", auto.Bytes, total)
	}
}
