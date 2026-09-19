package radixnet

// Cyclic counters - the odometer every growing integer in the model runs on,
// identical to the Python implementation's radixnet/counter.py.
//
// A number that only ever counts up - traversals, node and edge visit counts,
// epochs, trained characters, version stamps - eventually leaves the range of
// whatever holds it: int64 here, and long before that the 53-bit mantissa of
// the JSON number that carries it through a model file, the HTTP API and the
// frontend.  So no counter is an unbounded integer.  Each one is a two-digit
// odometer in base CounterLimit:
//
//	total = resets * CounterLimit + value        with 0 <= value < CounterLimit
//
// The value counts up as before; the moment it reaches the limit it is set
// back to 0 and resets - the number of times that happened - goes up by one.
// Nothing is lost: the total is the exact number of events, the value is what
// the odometer reads now and resets is how often it went round.
//
// Counting stays cheap because wrapping is not done per increment: the hot
// loops (plain or atomic increments, one goroutine per text) add to a raw
// int64 and a carry sweep at a safe point - the end of an epoch, before a save
// - moves whatever crossed the limit into the resets.  CounterLimit leaves
// four orders of magnitude of head room under int64, so no epoch can push a
// counter past the end of the world between two sweeps.

// CounterLimit is where every counter wraps back to 0, counting the wrap as a
// reset: 10^15, exactly representable as a float64 (< 2^53) so a counter
// survives a model file, a JSON response and a JavaScript Number unchanged,
// and four orders of magnitude below the int64 maximum.
const CounterLimit int64 = 1_000_000_000_000_000

// Carry normalises a raw (value, resets) pair into 0 <= value < CounterLimit:
// the whole turns of the odometer move from the value into the resets, which
// wrap at the same limit (so the pair itself cycles after 10^30 events).
func Carry(value, resets int64) (int64, int64) {
	turns := value / CounterLimit
	value -= turns * CounterLimit
	if value < 0 { // Go truncates towards zero; the odometer reads forwards
		value += CounterLimit
		turns--
	}
	resets = (resets + turns) % CounterLimit
	if resets < 0 {
		resets += CounterLimit
	}
	return value, resets
}

// Counter is one scalar odometer: the reading plus how often it wrapped.  It
// is comparable, so a cached version stamp can be checked with == and stays
// exact across a reset - the point of keeping the resets at all.
type Counter struct {
	Value  int64
	Resets int64
}

// NewCounter normalises value and resets into a reading.
func NewCounter(value, resets int64) Counter {
	v, r := Carry(value, resets)
	return Counter{Value: v, Resets: r}
}

// Float is the exact number of events counted, as a float (the Python
// implementation computes it in the same order, to the same bits).
func (c Counter) Float() float64 {
	return float64(c.Resets)*float64(CounterLimit) + float64(c.Value)
}

// Bumped is the reading n events later, wrapping as often as it takes.
func (c Counter) Bumped(n int64) Counter { return NewCounter(c.Value+n, c.Resets) }

// Add advances the counter in place.
func (c *Counter) Add(n int64) { *c = c.Bumped(n) }

// Less orders two readings by the number of events they stand for.
func (c Counter) Less(o Counter) bool {
	if c.Resets != o.Resets {
		return c.Resets < o.Resets
	}
	return c.Value < o.Value
}

// CarrySeries wraps every entry of a parallel counter slice in place and
// returns how many wrapped.  resets holds the reset counts of the ids that
// ever wrapped (absent means 0), so the common case - nothing has come near
// the limit - costs one comparison per entry and stores nothing.
func CarrySeries(values []int64, resets map[int]int64) int {
	wrapped := 0
	for i, v := range values {
		if v >= CounterLimit || v < 0 {
			nv, nr := Carry(v, resets[i])
			values[i] = nv
			if nr != 0 {
				resets[i] = nr
			} else {
				delete(resets, i)
			}
			wrapped++
		}
	}
	return wrapped
}

// anyNonZero reports whether a reset array is worth writing to a model file.
func anyNonZero(values []int64) bool {
	for _, v := range values {
		if v != 0 {
			return true
		}
	}
	return false
}

// resetsMap turns a model file's reset array into the sparse map the graph
// keeps; a file that carries none (the usual case) gives an empty map.
func resetsMap(values []int64) map[int]int64 {
	out := make(map[int]int64)
	for i, v := range values {
		if v != 0 {
			out[i] = v
		}
	}
	return out
}

// counterTotal is the exact count of an id in a parallel counter slice, as a
// float: what the weight function and the shares are computed from.
func counterTotal(value int64, resets map[int]int64, id int) float64 {
	if r := resets[id]; r != 0 {
		return float64(r)*float64(CounterLimit) + float64(value)
	}
	return float64(value)
}
