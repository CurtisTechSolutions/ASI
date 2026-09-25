//! The dynamic window: a ladder of node sizes, halving from 32 to 4 and back
//! up (`radixnet/window.py`, `go/radixnet/window.go`, `../SPEC-DynamicWindow.md`).
//!
//! A merged node can hold a whole sentence, and a walk through it has nowhere
//! to branch.  The dynamic window is a ceiling on that length, sized in the
//! binary number system: it starts at 32 units, then moves to 16, then 8,
//! then 4 (the floor), and then goes back up to 32 and runs again.  One step
//! of the ladder merges what fits the window (compression, which merges
//! nothing longer than the window), halves every node that is longer at its
//! middle gram - both halves carry the node's data, joined by a **heavy
//! connection** - and moves the window down the ladder.  A step happens by
//! hand (`radixnet window --step`, the routes, the frontend's Step button) or
//! automatically at the end of every training epoch while `auto` is set.  Off
//! (the default, and every model before it existed), compression is unbounded
//! and no node is ever halved.
//!
//! The heavy connection is heavy in each kind's own currency: the sine model
//! sets its weight to [`W_HEAVY`], the count and negative kinds give it every
//! traversal the node ever had (what a split hands its bridge), and the phase
//! model - which counts its edges, not its nodes - gives it what passed
//! through the node, the traversals of the out-edges it stands before.

use std::fmt;
use std::sync::atomic::Ordering;

use crate::cli::{Args, Ctx};
use crate::counter::COUNTER_LIMIT;
use crate::graph::{Graph, FIRST};
use crate::json::Json;
use crate::model::Model;

/// Where the ladder starts: the largest window, in the encoding's units.
pub const DEFAULT_TOP: usize = 32;
/// Where the ladder ends before it goes back up: the smallest window.
pub const DEFAULT_FLOOR: usize = 4;
/// The weight of the heavy connection between two halves in the sine model
/// (Python's `W_HEAVY`): negated while inverted, as every fresh weight is.
pub const W_HEAVY: f64 = 8.0;

/// Whether `value` is `1, 2, 4, 8, ...` - a size in the binary number system.
pub fn is_power_of_two(value: i64) -> bool {
    value >= 1 && value & (value - 1) == 0
}

/// The one rule for a ladder: `top` and `floor` powers of two with `floor <=
/// top`, and `size` (when given) a power of two between them.  The messages
/// are Python's, word for word.
pub fn check_ladder(top: i64, floor: i64, size: Option<i64>) -> Result<(), String> {
    if !is_power_of_two(top) {
        return Err(format!(
            "the window is sized in the binary number system: top must be a power of two, got {top}"
        ));
    }
    if !is_power_of_two(floor) {
        return Err(format!(
            "the window is sized in the binary number system: floor must be a power of two, got {floor}"
        ));
    }
    if floor > top {
        return Err(format!("floor must not exceed top, got floor {floor} over top {top}"));
    }
    if let Some(size) = size {
        if !is_power_of_two(size) || size < floor || size > top {
            return Err(format!(
                "size must be a power of two on the ladder {floor}..{top}, got {size}"
            ));
        }
    }
    Ok(())
}

/// The sizes the window takes, in order: `top, top / 2, ..., floor`.
pub fn ladder(top: usize, floor: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut size = top;
    while size >= floor && size > 0 {
        out.push(size);
        size /= 2;
    }
    out
}

/// The window a graph is held to: off (`top` is `None`), or a ladder `top ..
/// floor` standing at `size`.  `auto` says whether it steps by itself at the
/// end of every training epoch.  It travels with the model file while it is
/// on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicWindow {
    pub top: Option<usize>,
    pub floor: usize,
    pub size: usize,
    pub auto: bool,
}

impl Default for DynamicWindow {
    /// Off - with the floor and the automatic step Python's off window reports.
    fn default() -> DynamicWindow {
        DynamicWindow {
            top: None,
            floor: DEFAULT_FLOOR,
            size: 0,
            auto: true,
        }
    }
}

impl DynamicWindow {
    /// A window on the ladder `top .. floor` at `size` (the top when `None`), checked.
    pub fn new(top: i64, floor: i64, size: Option<i64>, auto: bool) -> Result<DynamicWindow, String> {
        check_ladder(top, floor, size)?;
        Ok(DynamicWindow {
            top: Some(top as usize),
            floor: floor as usize,
            size: size.unwrap_or(top) as usize,
            auto,
        })
    }

    /// Is the window on?  Off, compression is unbounded and no node is ever halved.
    pub fn is_on(&self) -> bool {
        self.top.is_some()
    }

    /// The ladder, top to floor; empty while off.
    pub fn sizes(&self) -> Vec<usize> {
        self.top.map(|top| ladder(top, self.floor)).unwrap_or_default()
    }

    /// The size after this one: half of it, or the top again from the floor (`None` while off).
    pub fn next(&self) -> Option<usize> {
        let top = self.top?;
        let half = self.size / 2;
        Some(if half >= self.floor { half } else { top })
    }

    /// The window one step down the ladder (back at the top from the floor).
    pub fn advanced(&self) -> DynamicWindow {
        match self.next() {
            Some(size) => DynamicWindow { size, ..*self },
            None => *self,
        }
    }

    /// The `dynamic_window` block of a graph document, or `None` - nothing is written - while off.
    pub fn to_json(&self) -> Option<Json> {
        self.top.map(|top| {
            Json::obj([
                ("top", Json::Int(top as i64)),
                ("floor", Json::Int(self.floor as i64)),
                ("size", Json::Int(self.size as i64)),
                ("auto", Json::Bool(self.auto)),
            ])
        })
    }

    /// The size a report carries: the number, or null while off.
    pub fn size_json(&self) -> Json {
        match self.top {
            Some(_) => Json::Int(self.size as i64),
            None => Json::Null,
        }
    }

    /// Reads a `dynamic_window` block; a document without one was written with the window off.
    pub fn from_json(block: Option<&Json>) -> Result<DynamicWindow, String> {
        let block = match block {
            None | Some(Json::Null) => return Ok(DynamicWindow::default()),
            Some(block @ Json::Obj(_)) => block,
            Some(other) => return Err(format!("a dynamic_window block is an object, got {}", other.render(0))),
        };
        fn whole(block: &Json, key: &str) -> Result<Option<i64>, String> {
            match block.get(key) {
                None | Some(Json::Null) => Ok(None),
                Some(Json::Int(value)) => Ok(Some(*value)),
                Some(other) => Err(format!(
                    "dynamic_window {key} must be a whole number, got {}",
                    other.render(0)
                )),
            }
        }
        let Some(top) = whole(block, "top")? else {
            return Ok(DynamicWindow::default());
        };
        let floor = whole(block, "floor")?.unwrap_or(DEFAULT_FLOOR as i64);
        let size = whole(block, "size")?;
        let auto = match block.get("auto") {
            None | Some(Json::Null) => true,
            Some(Json::Bool(value)) => *value,
            Some(other) => {
                return Err(format!(
                    "dynamic_window auto must be true or false, got {}",
                    other.render(0)
                ))
            }
        };
        DynamicWindow::new(top, floor, size, auto)
    }

    /// The human form: the ladder, where it stands and whether it steps by itself.
    pub fn describe(&self, units: &str) -> String {
        if !self.is_on() {
            return "off: compression is unbounded, and no node is halved".to_string();
        }
        let when = if self.auto {
            "stepping at the end of every training epoch"
        } else {
            "stepping by hand (window --step)"
        };
        format!(
            "on: {} {units}, at {} (next {}), {when}",
            join(&self.sizes()),
            self.size,
            self.next().unwrap_or(self.size)
        )
    }
}

impl fmt::Display for DynamicWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.top {
            None => write!(f, "off"),
            Some(_) => write!(f, "{}, at {}", join(&self.sizes()), self.size),
        }
    }
}

fn join(sizes: &[usize]) -> String {
    sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" -> ")
}

// -- the graph ------------------------------------------------------------------------------------

impl Graph {
    /// How many grams a node's label holds: `(length - n) / stride + 1`.
    pub fn grams_held(&self, node: usize) -> usize {
        (self.label_len[node] - self.enc.n) / self.enc.stride + 1
    }

    /// `(how many real nodes are longer than size, the longest label)` - what
    /// a window step would halve, in the encoding's units.
    pub fn longer_than(&self, size: usize) -> (usize, usize) {
        let (mut longer, mut longest) = (0, 0);
        for node in FIRST..self.labels.len() {
            if !self.alive[node] {
                continue;
            }
            let length = self.label_len[node];
            if length > size {
                longer += 1;
            }
            longest = longest.max(length);
        }
        (longer, longest)
    }

    /// Whether the dynamic window keeps a unary chain `p -> c` from merging:
    /// merged, the node would be longer than the window.
    pub(crate) fn held_apart(&self, p: usize, c: usize) -> bool {
        let w = self.dynamic_window;
        w.is_on() && self.label_len[p] + self.label_len[c] - self.enc.overlap() > w.size
    }

    /// Makes the edge `a -> b` between two halves the heavy connection, in
    /// this kind's currency: the sine model's weight becomes [`W_HEAVY`]
    /// (negated while inverted); the phase model's bridge takes `through`
    /// traversals - what passed through the node before it was halved - and
    /// no phase, since a step inside a merged node was never fired at one;
    /// the count and negative kinds' bridge already carries the node's count.
    fn heavy_bridge(&mut self, a: usize, b: usize, through: i64) {
        let Some(e) = self.children[a].get(b) else { return };
        if self.radix.is_some() {
            self.edge_w[e] = if self.inverted { -W_HEAVY } else { W_HEAVY };
            self.version.add(1);
        } else if self.res.is_some() {
            self.edge_count[e].store(through % COUNTER_LIMIT, Ordering::Relaxed);
            match through / COUNTER_LIMIT {
                0 => {
                    self.edge_count_resets.remove(&e);
                }
                resets => {
                    self.edge_count_resets.insert(e, resets);
                }
            }
        }
    }

    /// Halves every real node longer than `size` units until none is; returns
    /// how many splits were made.  The step of the dynamic window.
    ///
    /// Nodes are visited in id order; a node longer than the window is split
    /// at its middle gram - the first half keeps `(grams + 1) / 2` of its
    /// grams, the id and the in-edges, the second half (a new id at the end,
    /// halved in its turn when the scan reaches it) takes the rest and the
    /// out-edges, and both carry the node's data ([`Graph::split`]).  The edge
    /// between them is the heavy connection.  A node of one gram cannot be
    /// halved and is left as it is.  Nothing is merged here: [`Graph::compress`]
    /// does that, within the window.
    pub fn split_window(&mut self, size: usize) -> Result<usize, String> {
        if size < 1 {
            return Err(format!("size must be >= 1, got {size}"));
        }
        let stride = self.enc.stride;
        let mut splits = 0;
        let mut node = FIRST;
        // the second halves are appended, and are halved in turn when the scan reaches them
        while node < self.labels.len() {
            if self.alive[node] {
                let mut through: Option<i64> = None;
                while self.label_len[node] > size {
                    let grams = self.grams_held(node);
                    if grams < 2 {
                        break;
                    }
                    if through.is_none() {
                        // what passed through the node, read before it is halved: every walk
                        // that entered it left it over one of these edges
                        let mut total = 0i64;
                        for &e in &self.children[node].edges {
                            let c = self.edge_traversals(e);
                            total += c.resets * COUNTER_LIMIT + c.value;
                        }
                        through = Some(total);
                    }
                    // the first half keeps (grams + 1) / 2 of the grams: the odd one, when there is one
                    let (a, b) = self.split(node, grams.div_ceil(2) * stride)?;
                    self.heavy_bridge(a, b, through.unwrap_or(0));
                    splits += 1;
                }
            }
            node += 1;
        }
        if splits > 0 && self.res.is_some() {
            self.resonant_recompute();
        }
        Ok(splits)
    }
}

// -- the model ------------------------------------------------------------------------------------

/// What one or more steps of the ladder did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowStep {
    pub steps: usize,
    pub sizes: Vec<usize>,
    pub from: usize,
    pub to: usize,
    pub merges: usize,
    pub splits: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub edges_before: usize,
    pub edges_after: usize,
}

impl WindowStep {
    /// The step as the API and the CLI report it, with the window as it stands afterwards.
    pub fn to_json(&self, window: Json) -> Json {
        Json::obj([
            ("steps", Json::Int(self.steps as i64)),
            (
                "sizes",
                Json::Arr(self.sizes.iter().map(|&s| Json::Int(s as i64)).collect()),
            ),
            ("from", Json::Int(self.from as i64)),
            ("to", Json::Int(self.to as i64)),
            ("merges", Json::Int(self.merges as i64)),
            ("splits", Json::Int(self.splits as i64)),
            ("nodes_before", Json::Int(self.nodes_before as i64)),
            ("nodes_after", Json::Int(self.nodes_after as i64)),
            ("edges_before", Json::Int(self.edges_before as i64)),
            ("edges_after", Json::Int(self.edges_after as i64)),
            ("window", window),
        ])
    }
}

impl Model {
    /// The dynamic window as the API, the CLI and the frontend show it
    /// (`GraphModel.window_config`): the ladder, where it stands, how many
    /// real nodes a step would halve (null while off), the longest label and
    /// the sine model's bridge weight (null for a kind whose bridge is heavy
    /// by its count).
    pub fn window_config(&self) -> Json {
        let g = &self.g;
        let w = g.dynamic_window;
        let enc = g.enc;
        let (longer, longest) = g.longer_than(if w.is_on() { w.size } else { 0 });
        let int = |value: usize| Json::Int(value as i64);
        Json::obj([
            ("on", Json::Bool(w.is_on())),
            ("top", w.top.map(int).unwrap_or(Json::Null)),
            ("floor", int(w.floor)),
            ("size", w.size_json()),
            ("auto", Json::Bool(w.auto)),
            ("sizes", Json::Arr(w.sizes().into_iter().map(int).collect())),
            ("next", w.next().map(int).unwrap_or(Json::Null)),
            ("unit", Json::str(enc.unit.name())),
            ("units", Json::str(enc.units_name())),
            ("ngram", int(enc.n)),
            ("longer", if w.is_on() { int(longer) } else { Json::Null }),
            ("longest", int(longest)),
            ("nodes", int(g.num_nodes() - FIRST)),
            ("heavy", if g.is_radix() { Json::Num(W_HEAVY) } else { Json::Null }),
            ("default_top", int(DEFAULT_TOP)),
            ("default_floor", int(DEFAULT_FLOOR)),
        ])
    }

    /// Switches the window on or off, or moves its ladder
    /// (`GraphModel.configure_window`).  `on = false` switches it off whatever
    /// else is given; `on = true` or any setting switches it on at the values
    /// given over the ones it had, else the defaults; a new top or floor keeps
    /// the size on the ladder.  Nothing here touches the graph: only a step does.
    pub fn configure_window(
        &mut self,
        on: Option<bool>,
        top: Option<i64>,
        floor: Option<i64>,
        size: Option<i64>,
        auto: Option<bool>,
    ) -> Result<Json, String> {
        let current = self.g.dynamic_window;
        if on == Some(false) {
            self.g.dynamic_window = DynamicWindow::default();
            return Ok(self.window_config());
        }
        if on.is_none() && top.is_none() && floor.is_none() && size.is_none() && auto.is_none() {
            return Ok(self.window_config());
        }
        let top = top.unwrap_or(current.top.map(|t| t as i64).unwrap_or(DEFAULT_TOP as i64));
        let floor = floor.unwrap_or(if current.is_on() {
            current.floor as i64
        } else {
            DEFAULT_FLOOR as i64
        });
        let auto = auto.unwrap_or(if current.is_on() { current.auto } else { true });
        check_ladder(top, floor, None)?;
        let size = match size {
            Some(size) => size,
            None => {
                // a new top or floor keeps the size on the ladder
                let standing = if current.is_on() { current.size as i64 } else { top };
                standing.min(top).max(floor)
            }
        };
        self.g.dynamic_window = DynamicWindow::new(top, floor, Some(size), auto)?;
        Ok(self.window_config())
    }

    /// Steps the ladder `steps` times (`GraphModel.window_step`): each step
    /// compresses the graph within the current size (`compress = false` leaves
    /// that to a caller that has just done it), halves every node longer than
    /// the size and moves the window down the ladder - back to the top from
    /// the floor.  An error while the window is off.
    pub fn window_step(&mut self, steps: usize, compress: bool) -> Result<WindowStep, String> {
        if !self.g.dynamic_window.is_on() {
            return Err("the dynamic window is off: switch it on first (window --on)".to_string());
        }
        if steps < 1 {
            return Err(format!("steps must be >= 1, got {steps}"));
        }
        let mut done = WindowStep {
            steps,
            sizes: Vec::with_capacity(steps),
            from: 0,
            to: 0,
            merges: 0,
            splits: 0,
            nodes_before: self.g.num_nodes(),
            nodes_after: 0,
            edges_before: self.g.num_edges(),
            edges_after: 0,
        };
        for _ in 0..steps {
            let w = self.g.dynamic_window;
            if compress {
                done.merges += self.g.compress();
            }
            done.splits += self.g.split_window(w.size)?;
            done.sizes.push(w.size);
            self.g.dynamic_window = w.advanced();
        }
        done.from = done.sizes[0];
        done.to = self.g.dynamic_window.size;
        done.nodes_after = self.g.num_nodes();
        done.edges_after = self.g.num_edges();
        Ok(done)
    }

    /// [`Model::window_step`] as the API and the CLI report it, the window as it stands afterwards inside.
    pub fn window_step_json(&mut self, steps: usize) -> Result<Json, String> {
        let done = self.window_step(steps, true)?;
        Ok(done.to_json(self.window_config()))
    }

    /// The window's automatic step at the end of a training epoch: `None`
    /// when it is off or stepped by hand.  Every kind's loop has compressed
    /// the graph just before, so the step only halves and moves; the phase
    /// model, which compresses once before its passes, asks for the merging
    /// here with `compress = true`.
    pub(crate) fn window_epoch(&mut self, compress: bool) -> Option<WindowStep> {
        let w = self.g.dynamic_window;
        if !(w.is_on() && w.auto) {
            return None;
        }
        self.window_step(1, compress).ok()
    }
}

// -- the command ----------------------------------------------------------------------------------

fn maybe_int(args: &Args, name: &str) -> Result<Option<i64>, String> {
    match args.get(name) {
        Some(_) => Ok(Some(args.int(name, 0)?)),
        None => Ok(None),
    }
}

/// `radixnet window`: shows the dynamic window, sets its ladder (`--top`,
/// `--floor`, `--size`, `--auto` / `--manual`), switches it `--on` or `--off`,
/// or steps it by hand (`--step [N]`); saved unless `--dry-run`.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let (on, off) = (args.on("on"), args.on("off"));
    let (auto, manual) = (args.on("auto"), args.on("manual"));
    let (top, floor, size) = (
        maybe_int(args, "top")?,
        maybe_int(args, "floor")?,
        maybe_int(args, "size")?,
    );
    let step = maybe_int(args, "step")?;
    let settings = on || auto || manual || top.is_some() || floor.is_some() || size.is_some();
    if on && off {
        return Err("argument --off: not allowed with argument --on".to_string());
    }
    if off && settings {
        return Err("--off takes no other setting: switching the window off is all it does".to_string());
    }
    if off && step.is_some() {
        return Err("--off and --step contradict each other: a step needs the window on".to_string());
    }
    if auto && manual {
        return Err("--auto and --manual contradict each other".to_string());
    }
    let mut model = ctx.open(true)?;
    let mut changed = Json::Obj(Vec::new());
    let mut stepped = Json::Null;
    let mut saved = Json::Null;
    if settings || off {
        let auto = if auto {
            Some(true)
        } else if manual {
            Some(false)
        } else {
            None
        };
        model.configure_window(Some(!off), top, floor, size, auto)?;
        changed = model
            .g
            .dynamic_window
            .to_json()
            .unwrap_or_else(|| Json::obj([("on", Json::Bool(false))]));
    }
    if let Some(steps) = step {
        if steps < 1 {
            return Err(format!("steps must be >= 1, got {steps}"));
        }
        stepped = model.window_step_json(steps as usize)?;
    }
    let did_something = !matches!(&changed, Json::Obj(pairs) if pairs.is_empty()) || !matches!(stepped, Json::Null);
    if did_something && !args.on("dry-run") {
        saved = Json::str(ctx.save(&mut model)?);
    }
    ctx.emit(Json::obj([
        ("window", model.window_config()),
        ("changed", changed),
        ("step", stepped),
        ("saved", saved),
    ]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::{GraphOptions, TrainOptions};

    const LONG: &str = "the quick brown fox jumps over the lazy dog while the cat sat on the mat and the bird sang in the tree all afternoon";
    const TEXTS: [&str; 3] = [
        LONG,
        "the quick brown fox jumps over the lazy dog while the cat ran to the door",
        "a bird sang in the tree all afternoon",
    ];

    fn texts() -> Vec<String> {
        TEXTS.iter().map(|s| s.to_string()).collect()
    }

    fn trained() -> Model {
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        model.workers = 1;
        model.g.workers = 1;
        model
            .train(
                &texts(),
                &TrainOptions {
                    epochs: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    fn labels(g: &Graph) -> Vec<String> {
        (FIRST..g.labels.len())
            .filter(|&n| g.alive[n])
            .map(|n| g.labels[n].clone())
            .collect()
    }

    #[test]
    fn the_ladder_halves_from_the_top_to_the_floor() {
        assert_eq!(ladder(32, 4), vec![32, 16, 8, 4]);
        assert_eq!(ladder(8, 8), vec![8]);
        assert_eq!(ladder(64, 1), vec![64, 32, 16, 8, 4, 2, 1]);
        for (top, floor, size) in [
            (20, 4, None),
            (32, 3, None),
            (4, 32, None),
            (32, 4, Some(12)),
            (32, 4, Some(64)),
            (32, 8, Some(4)),
        ] {
            assert!(check_ladder(top, floor, size).is_err(), "{top} {floor} {size:?}");
        }
        assert!(check_ladder(32, 4, Some(8)).is_ok());
        let mut window = DynamicWindow::new(32, 4, None, true).unwrap();
        let mut seen = Vec::new();
        for _ in 0..6 {
            seen.push(window.size);
            window = window.advanced();
        }
        assert_eq!(seen, vec![32, 16, 8, 4, 32, 16]);
        let off = DynamicWindow::default();
        assert!(!off.is_on() && off.next().is_none() && off.sizes().is_empty() && off.to_json().is_none());
        assert_eq!(off.to_string(), "off");
        assert_eq!(off.advanced(), off);
        assert_eq!(DynamicWindow::new(8, 8, None, false).unwrap().next(), Some(8));
    }

    #[test]
    fn the_file_block() {
        let window = DynamicWindow::new(32, 4, Some(16), false).unwrap();
        assert_eq!(
            window.to_json().unwrap().render(0),
            r#"{"top":32,"floor":4,"size":16,"auto":false}"#
        );
        assert_eq!(DynamicWindow::from_json(window.to_json().as_ref()).unwrap(), window);
        assert_eq!(DynamicWindow::from_json(None).unwrap(), DynamicWindow::default());
        assert_eq!(
            DynamicWindow::from_json(Some(&Json::obj([("top", Json::Int(16))]))).unwrap(),
            DynamicWindow::new(16, 4, None, true).unwrap()
        );
        for bad in [
            Json::obj([("top", Json::Int(12))]),
            Json::obj([("top", Json::Int(32)), ("floor", Json::Int(64))]),
            Json::obj([("top", Json::Int(32)), ("size", Json::Int(3))]),
            Json::obj([("top", Json::str("32"))]),
            Json::obj([("top", Json::Int(32)), ("auto", Json::str("yes"))]),
        ] {
            assert!(DynamicWindow::from_json(Some(&bad)).is_err(), "{}", bad.render(0));
        }
    }

    #[test]
    fn a_node_longer_than_the_window_is_halved_at_its_middle_gram() {
        let mut g = Graph::new(1, GraphOptions::default()).unwrap();
        g.observe(&g.enc.encode("abcdefghij"), true).unwrap();
        g.compress();
        assert_eq!(labels(&g), vec!["abcdefghij"]);
        assert_eq!(g.split_window(8).unwrap(), 1);
        assert_eq!(labels(&g), vec!["abcdef", "efghij"]);
        assert_eq!(g.split_window(4).unwrap(), 2);
        let mut got = labels(&g);
        got.sort();
        assert_eq!(got, vec!["abcd", "cdef", "efgh", "ghij"]);
        assert_eq!(g.split_window(4).unwrap(), 0);
        assert_eq!(g.split_window(3).unwrap(), 4);
        assert_eq!(g.split_window(1).unwrap(), 0, "a node of one gram cannot be halved");
        assert!(g.split_window(0).is_err());
        g.check_invariants(&["abcdefghij".to_string()], false).unwrap();
    }

    #[test]
    fn a_node_is_cut_into_its_two_halves() {
        // ABCD under a grouping encoding is the grams AB and CD, and halves into exactly those two nodes;
        // under the sliding trigram it is ABC -> BCD merged, and halves into those, sharing the pivot BC
        let mut grouped = Graph::new(
            1,
            GraphOptions {
                encoding: crate::encoding::parse_encoding("char:2:2").unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        grouped.observe(&grouped.enc.encode("ABCD"), true).unwrap();
        grouped.compress();
        assert_eq!(labels(&grouped), vec!["ABCD"]);
        assert_eq!(grouped.split_window(2).unwrap(), 1);
        assert_eq!(labels(&grouped), vec!["AB", "CD"]);
        let mut sliding = Graph::new(1, GraphOptions::default()).unwrap();
        sliding.observe(&sliding.enc.encode("ABCD"), true).unwrap();
        sliding.compress();
        assert_eq!(labels(&sliding), vec!["ABCD"]);
        assert_eq!(sliding.split_window(2).unwrap(), 1);
        assert_eq!(labels(&sliding), vec!["ABC", "BCD"]);
        assert_eq!(
            sliding.split_window(2).unwrap(),
            0,
            "a half is never shorter than one gram"
        );
    }

    #[test]
    fn the_halves_carry_the_count_and_the_bridge_is_heavy_by_it() {
        let mut g = Graph::new(1, GraphOptions::default()).unwrap();
        for _ in 0..3 {
            g.observe(&g.enc.encode("abcdefghij"), true).unwrap();
        }
        g.compress();
        let node = (FIRST..g.labels.len()).find(|&n| g.alive[n]).unwrap();
        assert_eq!(g.node_count(node).value, 3);
        assert_eq!(g.split_window(8).unwrap(), 1);
        let (a, b) = (node, g.labels.len() - 1);
        assert_eq!((g.node_count(a).value, g.node_count(b).value), (3, 3));
        let e = g.edge(a, b).unwrap();
        assert_eq!(
            g.edge_traversals(e).value,
            3,
            "the bridge carries every traversal of the node"
        );
        g.prepare();
        let costs = g.child_costs(a);
        assert_eq!((costs.len(), costs[0].child, costs[0].cost), (1, b, 0.0));
        g.observe(&g.enc.encode("abcdefxyz"), true).unwrap();
        g.prepare();
        let best = g
            .child_costs(a)
            .iter()
            .min_by(|x, y| x.cost.partial_cmp(&y.cost).unwrap())
            .unwrap()
            .child;
        assert_eq!(best, b, "the bridge stays the likeliest way on");
    }

    #[test]
    fn the_sine_model_sets_the_heavy_weight() {
        let mut g = Graph::new_radix(1, Encoding::default()).unwrap();
        g.observe(&g.enc.encode("abcdefghij"), true).unwrap();
        g.compress();
        assert_eq!(g.split_window(8).unwrap(), 1);
        let heavy: Vec<f64> = (0..g.edge_w.len())
            .filter(|&e| g.edge_alive[e] && g.edge_w[e].abs() == W_HEAVY)
            .map(|e| g.edge_w[e])
            .collect();
        assert_eq!(heavy, vec![W_HEAVY]);
        let node = (FIRST..g.labels.len()).find(|&n| g.alive[n]).unwrap();
        let half = g.labels.len() - 1;
        let r = g.radix.as_ref().unwrap();
        assert_eq!(
            (r.z[node], r.a[node], r.b[node], r.h[node], r.k[node]),
            (r.z[half], r.a[half], r.b[half], r.h[half], r.k[half])
        );
        let mut inverted = Graph::new_radix(1, Encoding::default()).unwrap();
        inverted.observe(&inverted.enc.encode("abcdefghij"), true).unwrap();
        inverted.compress();
        inverted.inverted = true;
        inverted.split_window(8).unwrap();
        assert!(inverted.edge_w.iter().any(|&w| w == -W_HEAVY), "negated while inverted");
    }

    #[test]
    fn compression_stops_at_the_window_and_the_top_regrows() {
        let mut model = trained();
        assert!(model.g.longer_than(0).1 > 16);
        assert!(model.window_step(1, true).is_err(), "off: nothing to step");
        model.configure_window(Some(true), None, None, None, None).unwrap();
        let nodes = model.g.num_nodes();
        let done = model.window_step(4, true).unwrap();
        assert_eq!((done.sizes.clone(), done.from, done.to), (vec![32, 16, 8, 4], 32, 32));
        assert!(done.splits > 0 && model.g.longer_than(0).1 <= 4);
        assert_eq!(done.nodes_after - done.nodes_before, done.splits);
        let again = model.window_step(1, true).unwrap();
        assert!(
            again.merges > 0 && again.splits == 0 && model.g.num_nodes() == nodes,
            "{again:?}"
        );
        for text in texts() {
            assert_eq!(model.score(&text).unknown_transitions, 0);
        }
        model.configure_window(None, Some(16), Some(4), Some(16), None).unwrap();
        model.g.split_window(16).unwrap();
        assert_eq!(model.g.compress(), 0, "the window holds the halves apart");
        model.g.check_invariants(&texts(), true).unwrap();
        model.configure_window(Some(false), None, None, None, None).unwrap();
        assert!(model.g.compress() > 0, "off: the unary halves merge back");
        model.g.check_invariants(&texts(), true).unwrap();
    }

    #[test]
    fn the_settings() {
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        let cfg = model.window_config();
        assert_eq!(cfg.at("on").as_bool(), Some(false));
        assert!(
            matches!(cfg.at("size"), Json::Null)
                && matches!(cfg.at("longer"), Json::Null)
                && matches!(cfg.at("heavy"), Json::Null)
        );
        let cfg = model.configure_window(Some(true), None, None, None, None).unwrap();
        assert_eq!(
            (cfg.at("top").as_i64(), cfg.at("size").as_i64(), cfg.at("next").as_i64()),
            (Some(32), Some(32), Some(16))
        );
        let cfg = model.configure_window(None, None, None, Some(8), Some(false)).unwrap();
        assert_eq!(
            (cfg.at("size").as_i64(), cfg.at("auto").as_bool()),
            (Some(8), Some(false))
        );
        let cfg = model.configure_window(None, Some(16), None, None, None).unwrap();
        assert_eq!((cfg.at("top").as_i64(), cfg.at("size").as_i64()), (Some(16), Some(8)));
        let cfg = model.configure_window(None, None, Some(16), None, None).unwrap();
        assert_eq!(
            (cfg.at("floor").as_i64(), cfg.at("size").as_i64()),
            (Some(16), Some(16))
        );
        for (top, floor, size) in [
            (Some(20), None, None),
            (None, Some(3), None),
            (None, Some(128), None),
            (None, None, Some(12)),
            (None, None, Some(1)),
        ] {
            assert!(model.configure_window(None, top, floor, size, None).is_err());
        }
        let cfg = model.configure_window(Some(false), None, None, Some(8), None).unwrap();
        assert_eq!(cfg.at("on").as_bool(), Some(false));
        assert!(matches!(model.g.dynamic_window.size_json(), Json::Null));
    }

    #[test]
    fn it_steps_by_itself_at_the_end_of_every_epoch() {
        let mut model = trained();
        model.configure_window(Some(true), None, None, None, None).unwrap();
        let records = model
            .train(
                &texts(),
                &TrainOptions {
                    epochs: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        let windows: Vec<Option<usize>> = records.iter().map(|r| r.window).collect();
        assert_eq!(windows, vec![Some(32), Some(16), Some(8), Some(4), Some(32)]);
        let splits: Vec<bool> = records.iter().map(|r| r.splits.unwrap_or(0) > 0).collect();
        assert_eq!(splits, vec![false, true, true, true, false]);
        assert!(
            records[4].merges > 0,
            "back at the top, the epoch's compression regrew the chains"
        );
        assert_eq!(model.g.dynamic_window.size, 16);
        let entry = records[1].to_json().render(0);
        assert!(
            entry.contains(r#""merges":0,"splits":"#) && entry.contains(r#","window":16,"transitions":"#),
            "{entry}"
        );
        for text in texts() {
            assert_eq!(model.score(&text).unknown_transitions, 0);
        }
        model.configure_window(None, None, None, None, Some(false)).unwrap();
        let more = model
            .train(
                &texts(),
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            more[0].window.is_none() && model.g.dynamic_window.size == 16,
            "by hand: the ladder does not move"
        );
    }

    #[test]
    fn it_travels_with_the_file_and_off_is_the_old_file() {
        let mut model = trained();
        let before = model.g.to_doc().render(0);
        model
            .configure_window(None, Some(16), Some(4), Some(8), Some(false))
            .unwrap();
        let doc = model.g.to_doc().render(0);
        assert!(
            doc.contains(r#""format_version":4,"dynamic_window":{"top":16,"floor":4,"size":8,"auto":false},"seed":1,"#),
            "{}",
            &doc[..200]
        );
        let path = std::env::temp_dir().join(format!("radixnet-window-{}.json", std::process::id()));
        model.save(path.to_str().unwrap()).unwrap();
        let again = Model::load(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            again.g.dynamic_window,
            DynamicWindow::new(16, 4, Some(8), false).unwrap()
        );
        model.configure_window(Some(false), None, None, None, None).unwrap();
        assert_eq!(
            model.g.to_doc().render(0),
            before,
            "off is the old document to the byte"
        );
    }
}
