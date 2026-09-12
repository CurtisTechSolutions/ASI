package radixnet

import (
	"archive/zip"
	"os"
	"sync"
	"sync/atomic"
)

// A PartSource splits its corpus into independently streamable parts that
// keep their order - the entries of a ZIP archive, the files of a set - so
// that every part decompresses and streams on its own goroutine while the
// corpus order is restored by position.
type PartSource interface {
	TextSource
	OpenParts() (Parts, error)
}

// Parts is an open PartSource: Len parts, each streamed by Each(part, fn).
type Parts interface {
	Len() int
	Each(part int, fn func(text string) error) error
	Close() error
}

// singlePart wraps any TextSource as one part.
type singlePart struct{ src TextSource }

func (s singlePart) Len() int { return 1 }
func (s singlePart) Each(_ int, fn func(string) error) error {
	return s.src.Each(fn)
}
func (s singlePart) Close() error { return nil }

// openParts opens a source's parts (one part for sources that have none).
func openParts(src TextSource) (Parts, error) {
	if ps, ok := src.(PartSource); ok {
		return ps.OpenParts()
	}
	return singlePart{src}, nil
}

// zipParts is an open archive: every entry is a part (non-text entries yield nothing).
type zipParts struct {
	zr        *zip.ReadCloser
	unit      string
	pageLines int
}

// OpenParts opens the archive once; entries stream on demand.
func (z ZipSource) OpenParts() (Parts, error) {
	zr, err := OpenZip(z.Path)
	if err != nil {
		return nil, err
	}
	return &zipParts{zr: zr, unit: z.Unit, pageLines: z.pageLinesOrDefault()}, nil
}

func (z ZipSource) pageLinesOrDefault() int {
	if z.PageLines <= 0 {
		return 50
	}
	return z.PageLines
}

func (p *zipParts) Len() int { return len(p.zr.File) }

func (p *zipParts) Each(part int, fn func(string) error) error {
	f := p.zr.File[part]
	if ZipSkipReason(f) != "" {
		return nil
	}
	rc, reason, err := openZipText(f)
	if err != nil {
		return err
	}
	if reason != "" {
		return nil
	}
	defer rc.Close()
	return streamLines(rc, p.unit, p.pageLines, fn)
}

func (p *zipParts) Close() error { return p.zr.Close() }

// OpenParts: a text file is one part.
func (f FileSource) OpenParts() (Parts, error) {
	if _, err := os.Stat(f.Path); err != nil {
		return nil, err
	}
	return singlePart{f}, nil
}

// OpenParts: a slice is one part.
func (s SliceSource) OpenParts() (Parts, error) { return singlePart{s}, nil }

// multiParts concatenates the parts of several sources in order.
type multiParts struct {
	subs   []Parts
	offset []int // first part index of every sub
	total  int
}

// OpenParts opens every source's parts.
func (m MultiSource) OpenParts() (Parts, error) {
	mp := &multiParts{}
	for _, src := range m {
		p, err := openParts(src)
		if err != nil {
			mp.Close()
			return nil, err
		}
		mp.subs = append(mp.subs, p)
		mp.offset = append(mp.offset, mp.total)
		mp.total += p.Len()
	}
	return mp, nil
}

func (m *multiParts) Len() int { return m.total }

func (m *multiParts) Each(part int, fn func(string) error) error {
	for i := len(m.subs) - 1; i >= 0; i-- {
		if part >= m.offset[i] {
			return m.subs[i].Each(part-m.offset[i], fn)
		}
	}
	return nil
}

func (m *multiParts) Close() error {
	var first error
	for _, s := range m.subs {
		if err := s.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// -- the sequencer: corpus order restored by position -------------------------------------------------------

type seqKey struct{ part, idx int }

// sequencer applies chunk results in corpus order - (part, chunk index) -
// while the chunks themselves finish in any order on any number of
// goroutines.  A submitted apply runs on the submitting goroutine as soon as
// everything before it has run; out-of-order results wait in a map.
type sequencer struct {
	mu         sync.Mutex
	pending    map[seqKey]func()
	counts     map[int]int // part -> number of chunks, known once the part's reader finished
	parts      int
	nextPart   int
	nextIdx    int
	finished   chan struct{}
	onApplied  func(part int) // one chunk of a part was applied: its slot is free at last
	onPartDone func(part int) // every result of a part is in: its reader may be replaced
}

func newSequencer(parts int) *sequencer {
	s := &sequencer{pending: map[seqKey]func(){}, counts: map[int]int{}, parts: parts, finished: make(chan struct{})}
	if parts == 0 {
		close(s.finished)
	}
	return s
}

// drain runs every pending apply that is next in line (caller holds mu).
func (s *sequencer) drain() {
	for s.nextPart < s.parts {
		if n, known := s.counts[s.nextPart]; known && s.nextIdx >= n {
			delete(s.counts, s.nextPart)
			if s.onPartDone != nil {
				s.onPartDone(s.nextPart) // everything this part produced has been applied
			}
			s.nextPart++
			s.nextIdx = 0
			continue
		}
		key := seqKey{s.nextPart, s.nextIdx}
		apply, ok := s.pending[key]
		if !ok {
			return
		}
		delete(s.pending, key)
		apply()
		if s.onApplied != nil {
			s.onApplied(s.nextPart) // only now is the chunk's memory free
		}
		s.nextIdx++
	}
	select {
	case <-s.finished:
	default:
		close(s.finished)
	}
}

// submit hands in the result of chunk idx of part.
func (s *sequencer) submit(part, idx int, apply func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pending[seqKey{part, idx}] = apply
	s.drain()
}

// finishPart records how many chunks a part produced.
func (s *sequencer) finishPart(part, chunks int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counts[part] = chunks
	s.drain()
}

// wait blocks until every part's chunks were applied.
func (s *sequencer) wait() { <-s.finished }

// -- fan-out over parts, chunks and texts -------------------------------------------------------------------

// streamStats counts what a pass streamed (atomics: the parts run concurrently).
type streamStats struct {
	texts, chars, skippedShort int64
}

// runParts streams the parts, cuts each into chunks of chunkSize texts and
// hands every chunk to fn on its own goroutine without waiting for it - the
// goroutine count is uncapped.  The one thing a reader waits for is a free
// chunk slot: at most inflight chunks (default two per CPU, and never more
// than workers when that caps the fan-out) are being read, processed or
// waiting for their turn at the sequencer, which is what keeps a corpus of any
// size from piling up in memory ahead of the counting.  The parts themselves
// stream in corpus order on one reader, or all at once on a goroutine each
// with parallelParts.  Texts shorter than a trigram are dropped and counted.
// Returns once every reader and every chunk goroutine finished; the first
// error wins.
func runParts(parts Parts, chunkSize, workers, inflight int, parallelParts bool, stats *streamStats, seq *sequencer, fn func(part, idx int, chunk []string) error) error {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	if inflight <= 0 {
		inflight = DefaultInflight()
	}
	if workers > 0 && workers < inflight {
		inflight = workers
	}
	var wg sync.WaitGroup
	var firstErr atomic.Value
	setErr := func(err error) {
		if err != nil {
			firstErr.CompareAndSwap(nil, err)
		}
	}
	// Backpressure, and the only thing the readers wait for: a chunk holds a slot
	// from the moment it is read until the sequencer has applied it - a finished
	// chunk waiting for its turn occupies memory like any other - so a pass costs
	// the graph plus inflight chunks whatever the corpus size.  The goroutines per
	// text inside a chunk stay uncapped.
	n := parts.Len()
	sem := make(chan struct{}, inflight)
	acquireFor := func(int) { sem <- struct{}{} }
	releaseFor := func(int) { <-sem }
	window := make(chan struct{}, 1)
	if parallelParts {
		// Read every part at once and the slots would all be taken by parts the
		// sequencer cannot reach yet, with the part it waits for unable to start:
		// each open part gets its own budget of chunks instead, and only a window
		// of parts is open at a time.  Permits are taken in part order, so the
		// oldest unfinished part always holds one and can always make progress.
		open := inflight
		if open > NumCPU {
			open = NumCPU
		}
		if open < 1 {
			open = 1
		}
		perPart := inflight / open
		if perPart < 1 {
			perPart = 1
		}
		budgets := make([]chan struct{}, n)
		for part := range budgets {
			budgets[part] = make(chan struct{}, perPart)
		}
		window = make(chan struct{}, open)
		acquireFor = func(part int) { budgets[part] <- struct{}{} }
		releaseFor = func(part int) {
			select {
			case <-budgets[part]:
			default:
			}
		}
	}
	seq.onApplied = releaseFor
	seq.onPartDone = func(int) {
		select {
		case <-window:
		default:
		}
	}
	var readers sync.WaitGroup
	for part := 0; part < n; part++ {
		if parallelParts {
			window <- struct{}{}
			readers.Add(1)
			go func(part int) {
				defer readers.Done()
				streamPart(parts, part, chunkSize, stats, seq, &wg, &firstErr, setErr, acquireFor, fn)
			}(part)
			if firstErr.Load() != nil {
				break
			}
			continue
		}
		streamPart(parts, part, chunkSize, stats, seq, &wg, &firstErr, setErr, acquireFor, fn)
		if firstErr.Load() != nil {
			break
		}
	}
	readers.Wait()
	wg.Wait()
	if err, ok := firstErr.Load().(error); ok && err != nil {
		return err
	}
	return nil
}

// DefaultInflight is the default bound on chunks in flight: two per CPU.
func DefaultInflight() int { return 2 * NumCPU }

// streamPart reads one part into chunks, spawning a goroutine per chunk.
func streamPart(parts Parts, part, chunkSize int, stats *streamStats, seq *sequencer, wg *sync.WaitGroup, firstErr *atomic.Value, setErr func(error), acquire func(part int), fn func(part, idx int, chunk []string) error) {
	idx := 0
	chunk := make([]string, 0, chunkSize)
	flush := func() {
		if len(chunk) == 0 {
			return
		}
		batch := chunk
		chunk = make([]string, 0, chunkSize)
		myIdx := idx
		idx++
		acquire(part) // the reader waits here until a chunk slot frees; the sequencer frees it
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := fn(part, myIdx, batch); err != nil {
				setErr(err)
				seq.submit(part, myIdx, func() {}) // keep the order moving
			}
		}()
	}
	err := parts.Each(part, func(t string) error {
		if firstErr.Load() != nil {
			return errStop
		}
		if runeLen(t) < Window {
			atomic.AddInt64(&stats.skippedShort, 1)
			return nil
		}
		atomic.AddInt64(&stats.texts, 1)
		atomic.AddInt64(&stats.chars, int64(runeLen(t)))
		chunk = append(chunk, t)
		if len(chunk) >= chunkSize {
			flush()
		}
		return nil
	})
	if err != nil && err != errStop {
		setErr(err)
	}
	flush()
	seq.finishPart(part, idx)
}
