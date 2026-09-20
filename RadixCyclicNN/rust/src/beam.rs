//! The two beams: the k best complete paths and the k worst ones.
//!
//! "Best" is the traversal's word.  Under [`Traversal::Reward`] it is the
//! cheapest path, the way the model has always searched; under
//! [`Traversal::LeastPunished`] it is the path whose *worst* step carries the
//! least blame, with the cost deciding only between paths that carry the same
//! worst step.  The bottom beam is the same order upside down, so the k worst
//! come back as the dearest paths - or, following the blame, the most punished
//! ones.

use crate::encoding::OVERLAP;
use crate::graph::{Graph, END, START};
use crate::penalty::PenaltyCosts;
use crate::search::{build_result, least_punished, onward, start_emission, PathResult, Traversal};
use crate::weights::ChildCost;

/// The best path plus the top-K and bottom-K paths of one search.
#[derive(Clone, Debug, Default)]
pub struct Prediction {
    pub best: PathResult,
    pub top: Vec<PathResult>,
    pub bottom: Vec<PathResult>,
    pub k: usize,
    pub beam: usize,
    pub mode: String,
    /// The traversal's *name*, as the CLI and the API spell it: "reward",
    /// "punishment" ([`crate::penalty`]) or "least-punished".
    pub traversal: String,
    pub expanded: usize,
}

/// The beam width used when none is given.
pub fn default_beam(k: usize) -> usize {
    (4 * k.max(1)).max(16)
}

#[derive(Clone, Copy)]
struct BeamState {
    cost: f64,
    chars: usize,
    node: usize,
    entry: usize,
    /// the worst step on the way here (LeastPunished)
    punish: f64,
}

#[derive(Clone, Copy)]
struct BeamEntry {
    node: usize,
    parent: Option<usize>,
    step: f64,
}

/// A finished path: its cost, its nodes, its steps and its worst step.
struct Finished {
    /// The beam's own running sum, kept for the record; the result's cost is
    /// the exact sum of the steps (`fsum`), which is what the other two
    /// implementations report.
    #[allow(dead_code)]
    cost: f64,
    ids: Vec<usize>,
    step: Vec<f64>,
    punish: f64,
}

/// A kept path as two numbers, higher being better in this beam's own
/// direction: what the walk was punished for and what it cost.  Under
/// [`Traversal::Reward`] the first is 0 on every path, so the whole comparison
/// is the cost order the beam has always used.
#[derive(Clone, Copy)]
struct DoneKey {
    first: f64,
    second: f64,
    entry: usize,
}

impl DoneKey {
    /// The strict order a kept path is replaced by.
    fn better_than(self, other: DoneKey) -> bool {
        if self.first != other.first {
            return self.first > other.first;
        }
        self.second > other.second
    }

    /// The order that picks which kept path to replace.
    fn ranks_last(self, other: DoneKey) -> bool {
        if self.first != other.first {
            return self.first < other.first;
        }
        if self.second != other.second {
            return self.second < other.second;
        }
        self.entry < other.entry
    }
}

fn less_state(a: &BeamState, b: &BeamState, traversal: Traversal) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if traversal == Traversal::LeastPunished && a.punish != b.punish {
        // the walk that has the least against it leads
        return if a.punish < b.punish {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if a.cost != b.cost {
        return if a.cost < b.cost {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    a.chars
        .cmp(&b.chars)
        .then(a.node.cmp(&b.node))
        .then(a.entry.cmp(&b.entry))
}

/// The knobs of one beam.
#[derive(Clone, Copy)]
struct BeamRun<'a> {
    /// The punishment traversal's cost function, or `None` for the graph's own.
    costs: Option<&'a PenaltyCosts>,
    start_node: usize,
    start_chars: usize,
    min_chars: usize,
    cap: Option<usize>,
    k: usize,
    width: usize,
    step_penalty: f64,
    to_end: bool,
    max_steps: usize,
    max_expansions: usize,
    worst: bool,
    traversal: Traversal,
}

/// One beam: the k best (or, with `worst`, the k worst) complete paths.
fn run_beam(g: &Graph, o: BeamRun<'_>) -> (Vec<Finished>, usize) {
    let mut entries: Vec<BeamEntry> = vec![BeamEntry {
        node: o.start_node,
        parent: None,
        step: 0.0,
    }];
    let sign = if o.worst { -1.0 } else { 1.0 };
    let complete = |node: usize, chars: usize| -> bool {
        node == END || o.cap.is_some_and(|cap| chars >= cap) || (!o.to_end && chars >= o.min_chars)
    };
    let key_of = |cost: f64, punish: f64, entry: usize| -> DoneKey {
        let first = if o.traversal == Traversal::LeastPunished {
            -punish * sign
        } else {
            0.0
        };
        DoneKey {
            first,
            second: -cost * sign,
            entry,
        }
    };
    let mut done: Vec<DoneKey> = Vec::with_capacity(o.k);
    // the kept path that ranks last: the first to be replaced
    let worst_index = |done: &[DoneKey]| -> usize {
        let mut best = 0;
        for i in 1..done.len() {
            if done[i].ranks_last(done[best]) {
                best = i;
            }
        }
        best
    };
    let offer = |done: &mut Vec<DoneKey>, cost: f64, punish: f64, entry: usize| {
        if o.k == 0 {
            return;
        }
        let key = key_of(cost, punish, entry);
        if done.len() < o.k {
            done.push(key);
            return;
        }
        let w = worst_index(done);
        if key.better_than(done[w]) {
            done[w] = key;
        }
    };

    let mut frontier: Vec<BeamState> = Vec::new();
    let start = BeamState {
        cost: 0.0,
        chars: o.start_chars,
        node: o.start_node,
        entry: 0,
        punish: 0.0,
    };
    if complete(o.start_node, o.start_chars) {
        offer(&mut done, 0.0, 0.0, 0);
    } else {
        frontier.push(start);
    }
    let mut fallback = frontier.clone();
    let (mut expanded, mut steps) = (0usize, 0usize);
    let mut children: Vec<ChildCost> = Vec::new();
    let mut candidates: Vec<BeamState> = Vec::new();
    while !frontier.is_empty() && steps < o.max_steps && expanded < o.max_expansions {
        steps += 1;
        candidates.clear();
        for st in &frontier {
            expanded += 1;
            // where the walk came from: a judged path prices the next step by it
            let prev = match entries[st.entry].parent {
                Some(parent) => Some(entries[parent].node),
                None if st.node == START => Some(START),
                None => None,
            };
            g.step_costs_into(st.node, prev, o.costs, &mut children);
            onward(&mut children);
            if o.traversal == Traversal::LeastPunished {
                least_punished(&mut children);
            }
            for &cc in children.iter() {
                let mut nchars = st.chars;
                if cc.child != END {
                    nchars += g.label_len(cc.child) - OVERLAP;
                }
                let step = cc.cost + o.step_penalty;
                let ncost = st.cost + step;
                // a path is as punished as its worst step
                let npunish = if cc.punish > st.punish { cc.punish } else { st.punish };
                let child_entry = entries.len();
                entries.push(BeamEntry {
                    node: cc.child,
                    parent: Some(st.entry),
                    step,
                });
                if complete(cc.child, nchars) {
                    offer(&mut done, ncost, npunish, child_entry);
                } else {
                    candidates.push(BeamState {
                        cost: ncost,
                        chars: nchars,
                        node: cc.child,
                        entry: child_entry,
                        punish: npunish,
                    });
                }
            }
            if expanded >= o.max_expansions {
                break;
            }
        }
        if !candidates.is_empty() {
            fallback.clone_from(&candidates);
        }
        if o.worst {
            candidates.sort_by(|a, b| less_state(b, a, o.traversal));
        } else {
            candidates.sort_by(|a, b| less_state(a, b, o.traversal));
        }
        candidates.truncate(o.width);
        std::mem::swap(&mut frontier, &mut candidates);
        // neither number shrinks along a path - a cost is a sum of costs and a
        // punishment the worst step so far - so once k paths finished and the
        // best partial one already ranks below the k-th of them, the best side
        // is settled
        if !o.worst && o.k > 0 && done.len() == o.k && !frontier.is_empty() {
            let w = worst_index(&done);
            let head = key_of(frontier[0].cost, frontier[0].punish, frontier[0].entry);
            if !head.better_than(done[w]) {
                break;
            }
        }
    }

    let path_of = |entry: usize| -> (Vec<usize>, Vec<f64>) {
        let mut ids = Vec::new();
        let mut steps_out = Vec::new();
        let mut at = Some(entry);
        while let Some(i) = at {
            let en = entries[i];
            ids.push(en.node);
            if en.parent.is_some() {
                steps_out.push(en.step);
            }
            at = en.parent;
        }
        ids.reverse();
        steps_out.reverse();
        (ids, steps_out)
    };

    struct Pick {
        cost: f64,
        punish: f64,
        entry: usize,
    }
    let mut picks: Vec<Pick> = Vec::new();
    if !done.is_empty() {
        for d in &done {
            picks.push(Pick {
                cost: -d.second * sign,
                // `+ 0.0` so a negative zero does not read as a punishment that is not there
                punish: -d.first * sign + 0.0,
                entry: d.entry,
            });
        }
        picks.sort_by(|a, b| {
            if o.traversal == Traversal::LeastPunished && a.punish != b.punish {
                return (a.punish * sign).total_cmp(&(b.punish * sign));
            }
            (a.cost * sign).total_cmp(&(b.cost * sign)).then(a.entry.cmp(&b.entry))
        });
    } else if !fallback.is_empty() {
        // nothing completed within the limits: the surviving partial paths, most characters first
        let mut fb = fallback;
        fb.sort_by(|a, b| {
            if a.chars != b.chars {
                return b.chars.cmp(&a.chars);
            }
            if o.traversal == Traversal::LeastPunished && a.punish != b.punish {
                return (a.punish * sign).total_cmp(&(b.punish * sign));
            }
            (a.cost * sign).total_cmp(&(b.cost * sign)).then(a.entry.cmp(&b.entry))
        });
        fb.truncate(o.k);
        for s in fb {
            picks.push(Pick {
                cost: s.cost,
                punish: s.punish,
                entry: s.entry,
            });
        }
    }
    let out = picks
        .into_iter()
        .map(|p| {
            let (ids, step) = path_of(p.entry);
            Finished {
                cost: p.cost,
                ids,
                step,
                punish: p.punish,
            }
        })
        .collect();
    (out, expanded)
}

/// The settings of a beam prediction.
#[derive(Clone, Copy)]
pub struct BeamOptions {
    pub k: usize,
    /// 0 = [`default_beam`]
    pub beam: usize,
    /// `None` = no cap
    pub max_chars: Option<usize>,
    pub step_penalty: f64,
    pub to_end: bool,
    /// 0 = the default (500 with `to_end`, else `min_chars + 50`)
    pub max_steps: usize,
    /// 0 = 200000
    pub max_expansions: usize,
    /// How the two beams rank a path.
    pub traversal: Traversal,
}

impl Default for BeamOptions {
    fn default() -> BeamOptions {
        BeamOptions {
            k: 5,
            beam: 0,
            max_chars: None,
            step_penalty: 0.0,
            to_end: false,
            max_steps: 0,
            max_expansions: 0,
            traversal: Traversal::Reward,
        }
    }
}

impl Graph {
    /// The k best complete paths (best first) and the k worst ones not among
    /// them, plus the expansions both beams made.
    ///
    /// The Go port runs the two beams on two goroutines; here they run in turn
    /// on the calling thread, which is why the benchmark compares the two ports
    /// with one worker each as well as at their own default parallelism.
    pub fn beam_predict(
        &mut self,
        start_node: usize,
        start_offset: usize,
        min_chars: usize,
        opts: BeamOptions,
    ) -> Result<(Vec<PathResult>, Vec<PathResult>, usize), String> {
        self.beam_predict_by(start_node, start_offset, min_chars, opts, None)
    }

    /// [`Graph::beam_predict`] reading the graph through a cost function of its
    /// own - the punishment traversal's ([`crate::penalty`]).
    pub fn beam_predict_by(
        &mut self,
        start_node: usize,
        start_offset: usize,
        min_chars: usize,
        opts: BeamOptions,
        costs: Option<&PenaltyCosts>,
    ) -> Result<(Vec<PathResult>, Vec<PathResult>, usize), String> {
        if opts.step_penalty < 0.0 {
            return Err("step_penalty must be >= 0".to_string());
        }
        let width = if opts.beam == 0 {
            default_beam(opts.k)
        } else {
            opts.beam
        };
        let max_chars = opts.max_chars.map(|c| c.max(min_chars));
        let max_steps = match opts.max_steps {
            0 if opts.to_end => 500,
            0 => min_chars + 50,
            n => n,
        };
        let max_exp = if opts.max_expansions == 0 {
            200_000
        } else {
            opts.max_expansions
        };
        self.prepare();
        let start_chars = start_emission(self, start_node, start_offset)?;
        if opts.k == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let run = BeamRun {
            costs,
            start_node,
            start_chars,
            min_chars,
            cap: max_chars,
            k: opts.k,
            width,
            step_penalty: opts.step_penalty,
            to_end: opts.to_end,
            max_steps,
            max_expansions: max_exp,
            worst: false,
            traversal: opts.traversal,
        };
        let (best, expanded) = run_beam(self, run);
        let mut bottom_cap = max_chars;
        if bottom_cap.is_none() && opts.to_end {
            // the bottom cap depends on the best side
            let (mut longest, mut emitted) = (0usize, 0usize);
            for f in &best {
                let mut sum = 0usize;
                for &n in &f.ids[1..] {
                    longest = longest.max(self.label_len(n));
                    if n != END {
                        sum += self.label_len(n) - OVERLAP;
                    }
                }
                emitted = emitted.max(sum);
            }
            bottom_cap = Some((2 * emitted + longest + 8).max(min_chars).max(16));
        }
        let (worst_paths, expanded_worst) = run_beam(
            self,
            BeamRun {
                cap: bottom_cap,
                worst: true,
                ..run
            },
        );
        let expanded = expanded + expanded_worst;

        let mut top = Vec::with_capacity(best.len());
        let mut seen: Vec<Vec<usize>> = Vec::with_capacity(best.len());
        for f in best {
            seen.push(f.ids.clone());
            let mut result = build_result(self, f.ids, f.step, start_offset, max_chars, expanded, None);
            result.punish = f.punish;
            top.push(result);
        }
        let mut bottom = Vec::with_capacity(worst_paths.len());
        for f in worst_paths {
            if seen.contains(&f.ids) {
                continue;
            }
            let mut result = build_result(self, f.ids, f.step, start_offset, bottom_cap, expanded, None);
            result.punish = f.punish;
            bottom.push(result);
        }
        Ok((top, bottom, expanded))
    }
}
