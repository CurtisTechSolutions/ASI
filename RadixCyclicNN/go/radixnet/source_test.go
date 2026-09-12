package radixnet

import (
	"archive/zip"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func streamed(t *testing.T, content, unit string, pageLines int) []string {
	t.Helper()
	var out []string
	if err := StreamSplit(strings.NewReader(content), unit, pageLines, func(s string) error { out = append(out, s); return nil }); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestStreamSplitMatchesSplitTexts(t *testing.T) {
	contents := []string{
		"first line\nsecond line\n\nthird para line one\nline two\n\n\nlast\n",
		"\ufeffbom line\r\nwindows line\r\n\r\nnext\r\n",
		"a\nb\nc\nd\ne",
		"page one\nstill one\fpage two\nmore two\f\fpage four",
		"   \n\n  \n",
		"single",
		strings.Repeat("x", 300000) + "\nshort\n", // a line longer than any buffer
	}
	for _, content := range contents {
		for _, unit := range []string{"lines", "paragraphs", "pages", "file"} {
			for _, pageLines := range []int{2, 50} {
				want := SplitTexts(content, unit, pageLines)
				got := streamed(t, content, unit, pageLines)
				if len(want) == 0 && len(got) == 0 {
					continue
				}
				if !reflect.DeepEqual(want, got) {
					t.Fatalf("unit %s pageLines %d content %q:\n in-memory %q\n streaming %q", unit, pageLines, truncateRunes(content, 60), want, got)
				}
			}
		}
	}
}

func writeZip(t *testing.T, path string, entries map[string]string) {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	// deterministic order
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	for _, name := range names {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(entries[name]))
	}
	zw.Close()
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestZipSourceStreamsTextEntriesOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.zip")
	writeZip(t, path, map[string]string{
		"a/one.txt":      "the cat sat on the mat\nthe cat ran to the door\n",
		"b/two.txt":      "the dog sat on the log\n\nthe dog ate the bone\n",
		"img.png":        "\x89PNG\x00\x00binary",
		"__MACOSX/._one": "meta",
		"empty.txt":      "  \n\n",
		"nested.zip":     "PK\x03\x04not really",
	})
	if !IsZipFile(path) {
		t.Fatal("IsZipFile")
	}
	texts, err := CollectTexts(ZipSource{Path: path, Unit: "lines"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"the cat sat on the mat", "the cat ran to the door", "the dog sat on the log", "the dog ate the bone"}
	if !reflect.DeepEqual(texts, want) {
		t.Fatalf("lines: %q", texts)
	}
	paras, _ := CollectTexts(ZipSource{Path: path, Unit: "paragraphs"})
	if !reflect.DeepEqual(paras, []string{"the cat sat on the mat the cat ran to the door", "the dog sat on the log", "the dog ate the bone"}) {
		t.Fatalf("paragraphs: %q", paras)
	}
	whole, _ := CollectTexts(ZipSource{Path: path, Unit: "file"})
	if len(whole) != 2 || whole[0] != "the cat sat on the mat\nthe cat ran to the door" {
		t.Fatalf("whole entries: %q", whole)
	}
	zr, err := OpenZip(path)
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	info, err := InspectZip(&zr.Reader)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Entries) != 2 || info.Lines != 4 {
		t.Fatalf("inspect: %+v", info)
	}
	reasons := map[string]string{}
	for _, sk := range info.Skipped {
		reasons[sk.Path] = sk.Reason
	}
	if reasons["img.png"] != "binary" || reasons["__MACOSX/._one"] != "macOS metadata" || reasons["empty.txt"] != "empty" || reasons["nested.zip"] != "nested archive" {
		t.Fatalf("skip reasons: %v", reasons)
	}
	// a text file source and the generic picker
	txt := filepath.Join(dir, "plain.txt")
	os.WriteFile(txt, []byte("\ufeffone\ntwo\n\nthree\n"), 0o644)
	lines, _ := CollectTexts(SourceForFile(txt, "lines", 0))
	if !reflect.DeepEqual(lines, []string{"one", "two", "three"}) {
		t.Fatalf("file source: %q", lines)
	}
	if _, ok := SourceForFile(path, "lines", 0).(ZipSource); !ok {
		t.Fatal("SourceForFile must pick ZipSource for an archive")
	}
	n, _ := CountTexts(MultiSource{ZipSource{Path: path}, FileSource{Path: txt}})
	if n != 7 {
		t.Fatalf("multi source count %d", n)
	}
	if _, err := CollectTexts(ZipSource{Path: txt}); err == nil {
		t.Fatal("a text file is not a ZIP")
	}
}

func TestChunkedStreamingEqualsInMemoryTraining(t *testing.T) {
	texts := corpus(t)
	dir := t.TempDir()
	// the corpus split over three archive entries plus noise entries
	third := len(texts) / 3
	writeZip(t, filepath.Join(dir, "corpus.zip"), map[string]string{
		"1.txt":     strings.Join(texts[:third], "\n") + "\n",
		"2/2.txt":   strings.Join(texts[third:2*third], "\n") + "\n",
		"3.txt":     strings.Join(texts[2*third:], "\n") + "\n",
		"skip.bin":  "\x00\x01",
		".DS_Store": "x",
	})
	reference := trained(t, 3, 0)
	for _, chunk := range []int{1, 7, 64, DefaultChunkSize} {
		m, err := NewModel(1, DefaultGraphOptions())
		if err != nil {
			t.Fatal(err)
		}
		m.Exact = true
		opts := DefaultTrainOptions()
		opts.Epochs = 3
		opts.ChunkSize = chunk
		records, err := m.TrainSource(ZipSource{Path: filepath.Join(dir, "corpus.zip"), Unit: "lines"}, opts)
		if err != nil {
			t.Fatal(err)
		}
		a, b := reference.ToDoc().Graph, m.ToDoc().Graph
		a.Version, b.Version = 0, 0
		if !reflect.DeepEqual(a.Nodes, b.Nodes) || !reflect.DeepEqual(a.Edges, b.Edges) || !reflect.DeepEqual(a.RngState, b.RngState) || !reflect.DeepEqual(a.Weights, b.Weights) {
			t.Fatalf("chunk %d: streaming the ZIP in chunks must give the in-memory model", chunk)
		}
		if m.MetaInt("trained_texts") != reference.MetaInt("trained_texts") || m.MetaInt("trained_chars") != reference.MetaInt("trained_chars") {
			t.Fatalf("chunk %d: meta differs", chunk)
		}
		for i, r := range records {
			ref := reference.History[i]
			if r["transitions"] != ref["transitions"] || r["nodes"] != ref["nodes"] {
				t.Fatalf("chunk %d epoch %d: record differs: %v vs %v", chunk, i, r, ref)
			}
			if d := r["loss"].(float64) - ref["loss"].(float64); d > 1e-9 || d < -1e-9 {
				t.Fatalf("chunk %d epoch %d: loss %v vs %v", chunk, i, r["loss"], ref["loss"])
			}
			if r["chunks"].(int) != (len(texts)+chunk-1)/chunk {
				t.Fatalf("chunk %d: %v chunks recorded", chunk, r["chunks"])
			}
		}
	}
}

func TestLargeArchiveStreamsThroughInChunks(t *testing.T) {
	// 30,000 generated lines (~1 MB) over 3 entries: the corpus is never held whole
	dir := t.TempDir()
	words := []string{"the", "cat", "dog", "sat", "ran", "on", "mat", "log", "door", "bone", "bird", "flew", "over", "house", "quick", "brown", "fox"}
	entries := map[string]string{}
	total := 0
	for e := 0; e < 3; e++ {
		var b strings.Builder
		for i := 0; i < 10000; i++ {
			n := 4 + (i % 7)
			for w := 0; w < n; w++ {
				if w > 0 {
					b.WriteByte(' ')
				}
				b.WriteString(words[(i*7+w*3+e)%len(words)])
			}
			b.WriteByte('\n')
			total++
		}
		entries[fmt.Sprintf("part-%d.txt", e)] = b.String()
	}
	path := filepath.Join(dir, "big.zip")
	writeZip(t, path, entries)
	m, err := NewModel(1, DefaultGraphOptions())
	if err != nil {
		t.Fatal(err)
	}
	m.Exact = true // this test is about streaming; racy counting has its own test
	opts := DefaultTrainOptions()
	opts.Epochs = 2
	opts.ChunkSize = 2048
	records, err := m.TrainSource(ZipSource{Path: path, Unit: "lines"}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if m.MetaInt("trained_texts") != int64(total) {
		t.Fatalf("trained %d of %d texts", m.MetaInt("trained_texts"), total)
	}
	if records[1]["chunks"].(int) != (total+2047)/2048 {
		t.Fatalf("chunks: %v", records[1]["chunks"])
	}
	if err := m.G.CheckInvariants(nil, true); err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict("the cat", DefaultPredictOptions())
	if err != nil || p.Text == "" {
		t.Fatalf("predict after streaming: %v %+v", err, p)
	}
}
