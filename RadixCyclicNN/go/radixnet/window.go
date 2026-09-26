package radixnet

import (
	"fmt"
	"strings"
)

// The dynamic window: a ladder of node sizes, halving from 32 to 4 and back up
// (radixnet/window.py, ../../SPEC-DynamicWindow.md).
//
// A merged node can hold a whole sentence, and a walk through it has nowhere
// to branch.  The dynamic window is a ceiling on that length, sized in the
// binary number system: it starts at 32 units, then moves to 16, then 8, then
// 4 (the floor), and then goes back up to 32 and runs again.  One step of the
// ladder merges what fits the window (compression, which merges nothing
// longer than the window), halves every node that is longer at its middle
// gram - both halves carry the node's data, joined by a heavy connection that
// carries every traversal the node ever had - and moves the window down the
// ladder.  A step happens by hand (the window command, the routes, the
// frontend's Step button) or automatically at the end of every training epoch
// while Auto is set.  Off (the zero value, and every model before it
// existed), compression is unbounded and no node is ever halved.

// The ladder the window starts with: the largest window, and the smallest
// before it goes back up.
const (
	DefaultWindowTop   = 32
	DefaultWindowFloor = 4
)

// DynamicWindow is a graph's window: off, or a ladder Top .. Floor standing at
// Size.  Auto says whether it steps by itself at the end of every training
// epoch.  It travels with the model file while it is on.
type DynamicWindow struct {
	On    bool
	Top   int
	Floor int
	Size  int
	Auto  bool
}

// IsPowerOfTwo reports whether v is 1, 2, 4, 8, ... - a size in the binary number system.
func IsPowerOfTwo(v int) bool { return v >= 1 && v&(v-1) == 0 }

// CheckLadder is the one rule for a ladder: top and floor powers of two with
// floor <= top, and size (when given, > 0) a power of two between them.  The
// messages are Python's, word for word.
func CheckLadder(top, floor, size int) error {
	if !IsPowerOfTwo(top) {
		return fmt.Errorf("the window is sized in the binary number system: top must be a power of two, got %d", top)
	}
	if !IsPowerOfTwo(floor) {
		return fmt.Errorf("the window is sized in the binary number system: floor must be a power of two, got %d", floor)
	}
	if floor > top {
		return fmt.Errorf("floor must not exceed top, got floor %d over top %d", floor, top)
	}
	if size != 0 && (!IsPowerOfTwo(size) || size < floor || size > top) {
		return fmt.Errorf("size must be a power of two on the ladder %d..%d, got %d", floor, top, size)
	}
	return nil
}

// Ladder is the sizes the window takes, in order: top, top / 2, ..., floor.
func Ladder(top, floor int) []int {
	out := []int{}
	for size := top; size >= floor; size /= 2 {
		out = append(out, size)
	}
	return out
}

// NewDynamicWindow is a window on the ladder top .. floor at size (0 = the top), checked.
func NewDynamicWindow(top, floor, size int, auto bool) (DynamicWindow, error) {
	if size == 0 {
		size = top
	}
	if err := CheckLadder(top, floor, size); err != nil {
		return DynamicWindow{}, err
	}
	return DynamicWindow{On: true, Top: top, Floor: floor, Size: size, Auto: auto}, nil
}

// Sizes is the ladder, top to floor; empty while off.
func (w DynamicWindow) Sizes() []int {
	if !w.On {
		return []int{}
	}
	return Ladder(w.Top, w.Floor)
}

// Next is the size after this one: half of it, or the top again from the floor (0 while off).
func (w DynamicWindow) Next() int {
	if !w.On {
		return 0
	}
	if half := w.Size / 2; half >= w.Floor {
		return half
	}
	return w.Top
}

// Advanced is the window one step down the ladder (back at the top from the floor).
func (w DynamicWindow) Advanced() DynamicWindow {
	if !w.On {
		return w
	}
	w.Size = w.Next()
	return w
}

// SizeOrNil is the size a report carries: the number, or nil (JSON null) while off.
func (w DynamicWindow) SizeOrNil() any {
	if !w.On {
		return nil
	}
	return w.Size
}

// String is "off" or the ladder and where it stands.
func (w DynamicWindow) String() string {
	if !w.On {
		return "off"
	}
	return fmt.Sprintf("%s, at %d", joinInts(w.Sizes(), " -> "), w.Size)
}

// Describe is the human form, with the ladder in units.
func (w DynamicWindow) Describe(units string) string {
	if !w.On {
		return "off: compression is unbounded, and no node is halved"
	}
	when := "stepping at the end of every training epoch"
	if !w.Auto {
		when = "stepping by hand (window --step)"
	}
	return fmt.Sprintf("on: %s %s, at %d (next %d), %s", joinInts(w.Sizes(), " -> "), units, w.Size, w.Next(), when)
}

func joinInts(values []int, sep string) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprint(v)
	}
	return strings.Join(parts, sep)
}

// -- the graph -----------------------------------------------------------------

// GramsHeld is how many grams a node's label holds: (length - n) / stride + 1.
func (g *Graph) GramsHeld(node int) int {
	return (g.labelLen[node]-g.Enc.N)/g.Enc.Stride + 1
}

// LongerThan counts the real nodes longer than size units - what a window step
// would halve - and finds the longest label.
func (g *Graph) LongerThan(size int) (longer, longest int) {
	for node := First; node < len(g.Labels); node++ {
		if !g.Alive[node] {
			continue
		}
		length := g.labelLen[node]
		if length > size {
			longer++
		}
		if length > longest {
			longest = length
		}
	}
	return longer, longest
}

// SplitWindow halves every real node longer than size units until none is;
// returns how many splits were made.  The step of the dynamic window.
//
// Nodes are visited in id order; a node longer than the window is split at
// its middle gram - the first half keeps (grams + 1) / 2 of its grams, the id
// and the in-edges, the second half (a new id at the end, halved in its turn
// when the scan reaches it) takes the rest and the out-edges, and both carry
// the node's count (Split).  The edge between them is the heavy connection:
// it carries every traversal the node ever had, which is what makes it heavy
// in a model whose weights are computed from counts (the sine model, which
// the Go port does not run, sets its weight).  A node of one gram cannot be
// halved and is left as it is.  Nothing is merged here: Compress does that,
// within the window.  Takes the structure write lock.
func (g *Graph) SplitWindow(size int) (int, error) {
	if size < 1 {
		return 0, fmt.Errorf("size must be >= 1, got %d", size)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	splits := 0
	for node := First; node < len(g.Labels); node++ { // the second halves are appended, and are halved in turn
		if !g.Alive[node] {
			continue
		}
		for g.labelLen[node] > size {
			grams := g.GramsHeld(node)
			if grams < 2 {
				break
			}
			if _, _, err := g.Split(node, ((grams+1)/2)*g.Enc.Stride); err != nil {
				return splits, err
			}
			splits++
		}
	}
	return splits, nil
}

// heldApart reports whether the dynamic window keeps a unary chain p -> c from
// merging: merged, the node would be longer than the window.
func (g *Graph) heldApart(p, c int) bool {
	w := g.DynamicWindow
	return w.On && g.labelLen[p]+g.labelLen[c]-g.Enc.Overlap() > w.Size
}

// -- the model -----------------------------------------------------------------

// WindowConfig is the dynamic window as the API, the CLI and the frontend show it.
type WindowConfig struct {
	On           bool     `json:"on"`
	Top          *int     `json:"top"`
	Floor        int      `json:"floor"`
	Size         *int     `json:"size"`
	Auto         bool     `json:"auto"`
	Sizes        []int    `json:"sizes"`
	Next         *int     `json:"next"`
	Unit         string   `json:"unit"`
	Units        string   `json:"units"`
	Ngram        int      `json:"ngram"`
	Longer       *int     `json:"longer"`
	Longest      int      `json:"longest"`
	Nodes        int      `json:"nodes"`
	Heavy        *float64 `json:"heavy"`
	DefaultTop   int      `json:"default_top"`
	DefaultFloor int      `json:"default_floor"`
}

// WindowConfig describes the model's window: the ladder, where it stands,
// how many real nodes a step would halve (nil while off) and the longest
// label, in the encoding's units.  Heavy is nil: both kinds the Go port runs
// compute their weights from counts, so their bridge is heavy by its count.
func (m *Model) WindowConfig() WindowConfig {
	g := m.G
	w := g.DynamicWindow
	enc := m.Encoding()
	size := 0
	if w.On {
		size = w.Size
	}
	longer, longest := g.LongerThan(size)
	out := WindowConfig{
		On: w.On, Floor: DefaultWindowFloor, Auto: true, Sizes: w.Sizes(), Unit: string(enc.Unit),
		Units: enc.UnitsName(), Ngram: enc.N, Longest: longest, Nodes: g.NumNodes() - First,
		DefaultTop: DefaultWindowTop, DefaultFloor: DefaultWindowFloor,
	}
	if w.On {
		top, current, next, count := w.Top, w.Size, w.Next(), longer
		out.Top, out.Floor, out.Size, out.Auto, out.Next, out.Longer = &top, w.Floor, &current, w.Auto, &next, &count
	}
	return out
}

// ConfigureWindow switches the window on or off, or moves its ladder.
//
// on=false switches it off whatever else is given: the graph stays as the
// last step left it, and compression is unbounded again.  on=true or any
// setting switches it on - at the values given over the ones it had, else
// the defaults (32 down to 4, standing at the top, stepping every epoch).  A
// new top or floor keeps the size on the ladder: above the top it becomes
// the top, below the floor the floor.  Nothing here touches the graph: only a
// step does (WindowStep).
func (m *Model) ConfigureWindow(on *bool, top, floor, size *int, auto *bool) (WindowConfig, error) {
	current := m.G.DynamicWindow
	if on != nil && !*on {
		m.G.DynamicWindow = DynamicWindow{}
		return m.WindowConfig(), nil
	}
	if on == nil && top == nil && floor == nil && size == nil && auto == nil {
		return m.WindowConfig(), nil
	}
	newTop, newFloor, newAuto := DefaultWindowTop, DefaultWindowFloor, true
	if current.On {
		newTop, newFloor, newAuto = current.Top, current.Floor, current.Auto
	}
	if top != nil {
		newTop = *top
	}
	if floor != nil {
		newFloor = *floor
	}
	if auto != nil {
		newAuto = *auto
	}
	if err := CheckLadder(newTop, newFloor, 0); err != nil {
		return m.WindowConfig(), err
	}
	newSize := newTop
	if size != nil {
		newSize = *size
	} else {
		if current.On {
			newSize = current.Size
		}
		if newSize > newTop { // a new top or floor keeps the size on the ladder
			newSize = newTop
		}
		if newSize < newFloor {
			newSize = newFloor
		}
	}
	w, err := NewDynamicWindow(newTop, newFloor, newSize, newAuto)
	if err != nil {
		return m.WindowConfig(), err
	}
	m.G.DynamicWindow = w
	return m.WindowConfig(), nil
}

// WindowStep is what one or more steps of the ladder did.
type WindowStep struct {
	Steps       int          `json:"steps"`
	Sizes       []int        `json:"sizes"`
	From        int          `json:"from"`
	To          int          `json:"to"`
	Merges      int          `json:"merges"`
	Splits      int          `json:"splits"`
	NodesBefore int          `json:"nodes_before"`
	NodesAfter  int          `json:"nodes_after"`
	EdgesBefore int          `json:"edges_before"`
	EdgesAfter  int          `json:"edges_after"`
	Window      WindowConfig `json:"window"`
}

// WindowStep steps the ladder steps times: each step compresses the graph
// (within the current size; compress=false leaves that to a caller that has
// just done it), halves every node longer than the size and moves the window
// down the ladder - back to the top from the floor.  An error while the
// window is off.
func (m *Model) WindowStep(steps int, compress bool) (*WindowStep, error) {
	g := m.G
	if !g.DynamicWindow.On {
		return nil, fmt.Errorf("the dynamic window is off: switch it on first (window --on)")
	}
	if steps < 1 {
		return nil, fmt.Errorf("steps must be >= 1, got %d", steps)
	}
	out := &WindowStep{Steps: steps, Sizes: []int{}, NodesBefore: g.NumNodes(), EdgesBefore: g.NumEdges()}
	for i := 0; i < steps; i++ {
		w := g.DynamicWindow
		if compress {
			out.Merges += g.Compress()
		}
		splits, err := g.SplitWindow(w.Size)
		out.Splits += splits
		if err != nil {
			return nil, err
		}
		out.Sizes = append(out.Sizes, w.Size)
		g.DynamicWindow = w.Advanced()
	}
	out.From, out.To = out.Sizes[0], g.DynamicWindow.Size
	out.NodesAfter, out.EdgesAfter = g.NumNodes(), g.NumEdges()
	out.Window = m.WindowConfig()
	return out, nil
}

// windowEpoch is the window's automatic step at the end of a training epoch:
// nil when it is off or stepped by hand.  Every loop has compressed the graph
// just before, so the step only halves and moves.
func (m *Model) windowEpoch() *WindowStep {
	w := m.G.DynamicWindow
	if !w.On || !w.Auto {
		return nil
	}
	done, err := m.WindowStep(1, false)
	if err != nil {
		return nil
	}
	return done
}
