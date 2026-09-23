//! The exact cheapest path over `(node, units emitted)` - the radix model's
//! default prediction.
//!
//! The count model treats `dijkstra` as another name for the beam, because its
//! costs depend on the path a walk came in by and a one-label-per-state search
//! cannot see that.  The sine model's costs depend on nothing but the node, so
//! it gets the real thing: a shortest path over the depth-unrolled graph, where
//! moving over `p -> c` costs `-log softmax(children of p)[c] + step_penalty`
//! (never negative, which is what makes Dijkstra correct) and emits the units of
//! `c` that do not overlap `p`.
//!
//! A port of `dijkstra_predict` in `radixnet/search.py`: the same goal, cap and
//! fallback rules, the same tie-break (the order states were pushed in), so the
//! same path comes back - and its cost is the exact sum of its steps, as
//! Python's `math.fsum` adds them.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::graph::{Graph, END, START};
use crate::hash::{map, Map};
use crate::penalty::PenaltyCosts;
use crate::search::{build_result, onward, start_emission, PathResult};
use crate::weights::ChildCost;

/// How many states the search settles before it gives up and returns its
/// best partial path.
pub const MAX_EXPANSIONS: usize = 200_000;

/// A heap entry: `(cost, tie)` decides the order, as Python's tuple does.
struct Entry {
    cost: f64,
    tie: u64,
    came_from: Option<usize>,
    node: usize,
    chars: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Entry) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Entry) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    /// Reversed, so the max-heap pops the cheapest; equal costs fall to the
    /// push order (`-0.0` and `0.0` are equal, as they are to Python's `<`).
    fn cmp(&self, other: &Entry) -> Ordering {
        if self.cost < other.cost {
            Ordering::Greater
        } else if self.cost > other.cost {
            Ordering::Less
        } else {
            other.tie.cmp(&self.tie)
        }
    }
}

/// The cheapest path from `(start_node, start_offset)` emitting at least
/// `min_chars` units.
///
/// `max_chars` is an optional hard cap: states at or past it are not expanded
/// and the text is truncated to it.  The goal is `END` with `to_end`, else the
/// first state popped with enough units (reaching `END` earlier counts too);
/// when no goal is reachable the popped state with the most units (ties: the
/// cheapest) is returned instead, so this never fails for a valid start.
/// `costs` replaces the graph's own cost function - the punishment traversal.
#[allow(clippy::too_many_arguments)] // the search's knobs, one per knob, as in Python
pub fn dijkstra_predict(
    g: &mut Graph,
    start_node: usize,
    start_offset: usize,
    min_chars: usize,
    max_chars: Option<usize>,
    step_penalty: f64,
    to_end: bool,
    include_context: Option<bool>,
    costs: Option<&PenaltyCosts>,
) -> Result<PathResult, String> {
    if step_penalty < 0.0 {
        return Err("step_penalty must be >= 0 (Dijkstra needs non-negative costs)".to_string());
    }
    let max_chars = max_chars.map(|cap| cap.max(min_chars));
    g.prepare();
    let g: &Graph = g;
    let start_chars = start_emission(g, start_node, start_offset)?;
    let overlap = g.enc.overlap();
    let start_key = (start_node, start_chars);
    let mut best: Map<(usize, usize), f64> = map();
    best.insert(start_key, 0.0);
    let mut prev: Map<(usize, usize), ((usize, usize), f64)> = map();
    let mut heap = BinaryHeap::new();
    heap.push(Entry {
        cost: 0.0,
        tie: 0,
        came_from: (start_node == START).then_some(START),
        node: start_node,
        chars: start_chars,
    });
    let mut tie = 0u64;
    let mut expanded = 0usize;
    let mut goal: Option<(usize, usize)> = None;
    let (mut fallback, mut fb_chars, mut fb_cost) = (start_key, start_chars, 0.0f64);
    let mut children: Vec<ChildCost> = Vec::new();
    while let Some(Entry {
        cost,
        came_from,
        node,
        chars,
        ..
    }) = heap.pop()
    {
        let key = (node, chars);
        if cost > best[&key] {
            continue; // a stale entry
        }
        expanded += 1;
        if node == END || (!to_end && chars >= min_chars) {
            goal = Some(key);
            break;
        }
        if chars > fb_chars || (chars == fb_chars && cost < fb_cost) {
            (fallback, fb_chars, fb_cost) = (key, chars, cost);
        }
        if expanded >= MAX_EXPANSIONS {
            break;
        }
        if max_chars.is_some_and(|cap| chars >= cap) {
            continue;
        }
        g.step_costs_into(node, came_from, costs, &mut children);
        onward(&mut children);
        for cc in &children {
            let nchars = if cc.child == END {
                chars
            } else {
                chars + g.label_len(cc.child) - overlap
            };
            let step = cc.cost + step_penalty;
            let ncost = cost + step;
            let nkey = (cc.child, nchars);
            if best.get(&nkey).is_none_or(|&b| ncost < b) {
                best.insert(nkey, ncost);
                prev.insert(nkey, (key, step));
                tie += 1;
                heap.push(Entry {
                    cost: ncost,
                    tie,
                    came_from: Some(node),
                    node: cc.child,
                    chars: nchars,
                });
            }
        }
    }
    let mut key = goal.unwrap_or(fallback);
    let mut node_ids = Vec::new();
    let mut step_costs = Vec::new();
    loop {
        node_ids.push(key.0);
        match prev.get(&key) {
            Some(&(from, step)) => {
                step_costs.push(step);
                key = from;
            }
            None => break,
        }
    }
    node_ids.reverse();
    step_costs.reverse();
    Ok(build_result(
        g,
        node_ids,
        step_costs,
        start_offset,
        max_chars,
        expanded,
        include_context,
    ))
}
