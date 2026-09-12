package radixnet

import (
	"runtime"
	"strings"
	"sync"
)

// Workers caps the goroutines fanned out over texts and nodes; 0 (the
// default) means no cap: one goroutine per text, and a goroutine per small
// block of nodes.  Model.Workers overrides it per model.
var Workers = 0

// NumCPU is the machine's logical CPU count (for the few places that need a number).
var NumCPU = runtime.NumCPU()

// parallelFor runs fn(i) for i in [0, n) and waits.  workers <= 0 spawns one
// goroutine per index - as many as there are texts - otherwise a pool of that
// size.  Every index is handled exactly once.  fn may write racily to shared
// counters when the caller accepts lost updates (see Model.Exact); it must
// never write to maps or append to shared slices, which the Go runtime does
// not survive.
func parallelFor(n, workers int, fn func(i int)) {
	if n <= 0 {
		return
	}
	if workers == 1 || n == 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	if workers <= 0 || workers >= n {
		wg.Add(n)
		for i := 0; i < n; i++ {
			go func(i int) {
				defer wg.Done()
				fn(i)
			}(i)
		}
		wg.Wait()
		return
	}
	next := make(chan int, workers*4)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

// parallelRanges splits [0, n) into contiguous chunks, one goroutine each, and
// calls fn(lo, hi) for every chunk (node ranges of the recompute).  Without a
// worker cap the chunks are small (64 nodes), so the goroutine count scales
// with the graph.
func parallelRanges(n, workers int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	chunk := 64
	if workers > 0 {
		chunk = (n + workers - 1) / workers
		if chunk < 256 {
			chunk = 256
		}
	}
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// SplitTexts cuts a file's content into training texts - the units the
// goroutines fan out over:
//
//   - "lines": one text per non-blank line (the Python CLI's default);
//   - "paragraphs": blocks separated by blank lines, newlines inside a block
//     joined with spaces;
//   - "pages": form-feed separated pages, or every pageLines lines when the
//     content has no form feed (0 = 50); newlines inside a page joined with
//     spaces;
//   - "file": the whole content as one text (trailing newlines stripped).
func SplitTexts(content, unit string, pageLines int) []string {
	content = strings.TrimPrefix(strings.ReplaceAll(content, "\r\n", "\n"), bom)
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "", "line", "lines":
		var texts []string
		for _, line := range strings.Split(content, "\n") {
			if strings.TrimSpace(line) != "" {
				texts = append(texts, line)
			}
		}
		return texts
	case "paragraph", "paragraphs":
		var texts []string
		var block []string
		flush := func() {
			if len(block) > 0 {
				texts = append(texts, strings.Join(block, " "))
				block = block[:0]
			}
		}
		for _, line := range strings.Split(content, "\n") {
			if strings.TrimSpace(line) == "" {
				flush()
			} else {
				block = append(block, strings.TrimSpace(line))
			}
		}
		flush()
		return texts
	case "page", "pages":
		if pageLines <= 0 {
			pageLines = 50
		}
		var pages []string
		if strings.Contains(content, "\f") {
			pages = strings.Split(content, "\f")
		} else {
			lines := strings.Split(content, "\n")
			for lo := 0; lo < len(lines); lo += pageLines {
				hi := lo + pageLines
				if hi > len(lines) {
					hi = len(lines)
				}
				pages = append(pages, strings.Join(lines[lo:hi], "\n"))
			}
		}
		var texts []string
		for _, page := range pages {
			var parts []string
			for _, line := range strings.Split(page, "\n") {
				if strings.TrimSpace(line) != "" {
					parts = append(parts, strings.TrimSpace(line))
				}
			}
			if len(parts) > 0 {
				texts = append(texts, strings.Join(parts, " "))
			}
		}
		return texts
	case "file", "whole", "whole-file":
		text := strings.TrimRight(content, "\r\n")
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return []string{text}
	}
	return SplitTexts(content, "lines", pageLines)
}
