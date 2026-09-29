package main

import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/pair"
	"github.com/CurtisTechSolutions/ASI/LatentRadixPair/tokenizer"
)

// ngram is a byte-context baseline that folds exactly like the pair (own = ctx / (ctx + alpha), no
// smoothing, the same floor) over the last k bytes, k = 0..order, so the only difference is what the
// context is: raw bytes, or the tokenizer's code.
type ngram struct {
	order  int
	levels []map[string]*ngramNode
	alpha  float64
	floor  float64
}

type ngramNode struct {
	total int
	next  map[int]int
}

func newNgram(order int, alpha, floor float64) *ngram {
	g := &ngram{order: order, alpha: alpha, floor: floor}
	for k := 0; k <= order; k++ {
		g.levels = append(g.levels, map[string]*ngramNode{})
	}
	return g
}

func (g *ngram) observe(text []byte) {
	outcomes := pair.Outcomes(text)
	for i, x := range outcomes {
		for k := 0; k <= g.order; k++ {
			if k > i {
				break
			}
			key := string(text[i-k : i])
			n := g.levels[k][key]
			if n == nil {
				n = &ngramNode{next: map[int]int{}}
				g.levels[k][key] = n
			}
			n.total++
			n.next[x]++
		}
	}
}

func (g *ngram) prob(text []byte, i, x int) float64 {
	p := 1.0 / pair.Out
	for k := 0; k <= g.order && k <= i; k++ {
		n := g.levels[k][string(text[i-k:i])]
		if n == nil {
			break
		}
		own := float64(n.total) / (float64(n.total) + g.alpha)
		p = own*float64(n.next[x])/float64(n.total) + (1-own)*p
	}
	return (1-g.floor)*p + g.floor/pair.Out
}

func (g *ngram) contexts() int {
	n := 0
	for k := 1; k <= g.order; k++ {
		n += len(g.levels[k])
	}
	return n
}

// benchCmd holds out every k-th file, reads the rest, and prices the held-out files under the latent
// model and under raw byte contexts of orders 0..maxOrder; then measures throughput.
func benchCmd(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	tokPath := fs.String("tokenizer", "tokenizer.json.gz", "tokenizer file")
	holdout := fs.Int("holdout", 5, "hold out every k-th file")
	maxOrder := fs.Int("max-order", 3, "highest raw byte context length")
	s := pair.DefaultSettings()
	settingsFlags(fs, &s)
	fs.Parse(args)
	tok, err := tokenizer.Load(*tokPath)
	if err != nil {
		return err
	}
	texts, err := readFiles(fs.Args())
	if err != nil {
		return err
	}
	var train, test [][]byte
	for i, t := range texts {
		if (i+1)%*holdout == 0 {
			test = append(test, t)
		} else {
			train = append(train, t)
		}
	}
	if len(test) == 0 || len(train) == 0 {
		return fmt.Errorf("need at least one training and one held-out file")
	}
	trainBytes, testBytes := 0, 0
	for _, t := range train {
		trainBytes += len(t)
	}
	for _, t := range test {
		testBytes += len(t)
	}
	fmt.Printf("train %d files (%d bytes), held out %d files (%d bytes); alpha %.1f floor %.3f smoothing %.2f\n",
		len(train), trainBytes, len(test), testBytes, s.Alpha, s.Floor, s.Smoothing)
	m, err := pair.Prime(tok, s, 1)
	if err != nil {
		return err
	}
	started := time.Now()
	m.TrainBytes(train)
	readSecs := time.Since(started).Seconds()
	started = time.Now()
	var bits float64
	units := 0
	for _, t := range test {
		sc, err := m.Score(string(t), "reward", "")
		if err != nil {
			return err
		}
		bits += sc.Bits * float64(len(t)+1)
		units += len(t) + 1
	}
	scoreSecs := time.Since(started).Seconds()
	used := 0
	for _, c := range m.Pair.Count.Ctx {
		if c > 0 {
			used++
		}
	}
	fmt.Printf("\n%-34s %10s %12s %10s\n", "context", "contexts", "bits/byte", "bits")
	fmt.Printf("%-34s %10d %12.3f %10.1f\n", fmt.Sprintf("latent code %v (%.0f bits)", tok.Radices(), tok.Bits()), used, bits/float64(units), tok.Bits())
	for order := 0; order <= *maxOrder; order++ {
		g := newNgram(order, s.Alpha, s.Floor)
		for _, t := range train {
			g.observe(t)
		}
		var b float64
		for _, t := range test {
			outcomes := pair.Outcomes(t)
			for i, x := range outcomes {
				b -= math.Log2(g.prob(t, i, x))
			}
		}
		fmt.Printf("%-34s %10d %12.3f %10d\n", fmt.Sprintf("raw bytes, last %d", order), g.contexts(), b/float64(units), 8*order)
	}
	fmt.Printf("\nthroughput: read %.0f bytes/s, score %.0f bytes/s", float64(trainBytes)/readSecs, float64(testBytes)/scoreSecs)
	sample := train[0]
	if len(sample) > 200000 {
		sample = sample[:200000]
	}
	started = time.Now()
	tok.EncodeAll(sample)
	fmt.Printf(", encode %.0f bytes/s", float64(len(sample)+1)/time.Since(started).Seconds())
	started = time.Now()
	n := 0
	for time.Since(started) < time.Second {
		m.Predict("the ", 20, "greedy", "reward", 0, false, "", 0)
		n += 20
	}
	fmt.Printf(", predict %.0f bytes/s\n", float64(n)/time.Since(started).Seconds())
	return nil
}
