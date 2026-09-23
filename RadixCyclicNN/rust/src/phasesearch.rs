//! Search over the phase-unrolled graph, with the metacognitive handoff on
//! cycles.
//!
//! A state here is `(node, units emitted, phase bucket)` rather than
//! [`crate::dijkstra`]'s `(node, units emitted)`.  The extra coordinate is what
//! the phase model ([`crate::resonance`]) scores edges against, and it is a
//! plain function of the state: entering node `c` moves the phase on by
//! `advance[c]` buckets (0 for a sentinel, which emits nothing).  Costs are
//! `-log softmax` over the children *at that phase* (plus `step_penalty`), so
//! `exp(-cost)` is still a path's probability.
//!
//! Because the phase is part of the state, the cost of an edge depends on the
//! whole path that led to it, and yet the search stays exact: the product graph
//! is finite (`buckets` times bigger), so [`phase_dijkstra`] is a true shortest
//! path over it.
//!
//! Revisiting a node at a *different* phase is progress; revisiting it at the
//! *same* phase is a loop that would repeat for ever.  The searches that carry
//! a path ([`phase_kbest`], [`phase_beam`], [`phase_walk`]) watch for the
//! second kind and hand the decision to a [`MetaLayer`]: closing the loop takes
//! the layer's `ride` cost, every other child its `escape` cost and `END` its
//! `abort` cost.  [`phase_dijkstra`] keeps its exactness instead and runs
//! without the layer.
//!
//! A port of `radixnet/phasesearch.py`, operation for operation: the same
//! tie-breaks, the same heaps (the beam's finished paths are read back in
//! `heapq`'s array order, so they live in [`crate::pyheap`]), the same
//! insertion orders - so the same paths come back in the same order.

use crate::fsum::fsum;
use crate::graph::{Graph, BACK, END, FIRST};
use crate::hash::{map, Map};
use crate::metacog::{action_index, cycle_signature, MetaLayer, ABORT, ESCAPE, RIDE};
use crate::mt19937::Mt19937;
use crate::pyheap::{heappop, heappush};
use crate::resonance::PhasePenalty;
use crate::search::{build_result, onward, start_emission, PathResult};
use crate::weights::ChildCost;

/// How many states [`phase_dijkstra`] and [`phase_kbest`] settle before they
/// give up and return their best partial path.
pub const MAX_EXPANSIONS: usize = 200_000;

/// How many steps [`phase_beam`] takes at most.
pub const MAX_STEPS: usize = 4_000;

/// Where each `(node, phase)` already on a path first appeared.
type Depth = Map<(usize, usize), usize>;

/// One way on out of a partial path.
struct Step {
    child: usize,
    cost: f64,
    phase: usize,
}

/// `p`'s children at `bucket`: the graph's own phase-aware costs, or the
/// punishment traversal's in their place.
fn costs_at(g: &Graph, costs: Option<&PhasePenalty>, p: usize, bucket: usize) -> Vec<ChildCost> {
    match costs {
        Some(pp) => pp.costs(g, p, bucket),
        None => g.child_costs_at(p, bucket),
    }
}

fn buckets_of(g: &Graph) -> usize {
    g.res.as_ref().map(|r| r.buckets).unwrap_or(1)
}

fn advance_of(g: &Graph, c: usize) -> usize {
    g.res.as_ref().and_then(|r| r.advance.get(c).copied()).unwrap_or(0)
}

/// Units a step into `c` emits past `chars`.
fn chars_after(g: &Graph, chars: usize, c: usize) -> usize {
    if c < FIRST {
        chars
    } else {
        chars + g.label_len(c) - g.enc.overlap()
    }
}

/// `P(hand over | node, phase)`: the share the node's `BACK` edge takes, 0
/// when it has none.
fn back_probability(costs: &[ChildCost]) -> f64 {
    costs
        .iter()
        .find(|cc| cc.child == BACK)
        .map(|cc| (-cc.cost).exp())
        .unwrap_or(0.0)
}

/// The children of one partial path, priced.
///
/// `depth` maps every `(node, phase)` already on the path to where it first
/// appeared: a child landing on one of them closes a phase-locked cycle, the
/// tightest such loop names it, and the layer's price for riding, escaping or
/// aborting it is added to every child's cost.  `BACK` is the reflex and the
/// layer the memory: the node's hand-over probability enters the layer's
/// policy as evidence against riding, and a hand-over `onward` made is
/// overruled only when the layer remembers riding cycles here.
fn expand(
    g: &Graph,
    meta: Option<&MetaLayer>,
    node: usize,
    phase: usize,
    depth: &Depth,
    step_penalty: f64,
    costs: Option<&PhasePenalty>,
) -> Vec<Step> {
    let buckets = buckets_of(g);
    let every = costs_at(g, costs, node, phase);
    let mut raw = every.clone();
    onward(&mut raw); // a node the model expects to go round offers nothing
    let mut extra: Option<[f64; 3]> = None;
    // (child, loop length) in the order the children came
    let mut loops: Vec<(usize, i64)> = Vec::new();
    if let Some(meta) = meta {
        if !depth.is_empty() {
            let here = depth.get(&(node, phase)).copied().unwrap_or(0) as i64;
            // the cycle is read off *every* child, so it is still seen when BACK has vetoed the branch
            for cc in &every {
                if cc.child < FIRST {
                    continue; // a sentinel is not a node to loop through
                }
                let nphase = (phase + advance_of(g, cc.child)) % buckets;
                if let Some(&first) = depth.get(&(cc.child, nphase)) {
                    let len = here + 1 - first as i64;
                    match loops.iter_mut().find(|(c, _)| *c == cc.child) {
                        Some(slot) => slot.1 = len,
                        None => loops.push((cc.child, len)),
                    }
                }
            }
            if !loops.is_empty() {
                // the tightest loop names the cycle (the first of equals, as `min` picks)
                let mut target = loops[0];
                for &item in &loops[1..] {
                    if item.1 < target.1 {
                        target = item;
                    }
                }
                let signature = cycle_signature(g.label(target.0), target.1);
                extra = Some(meta.costs(&signature, back_probability(&every)));
            }
        }
        if raw.is_empty() && meta.rides(g.label(node)) {
            // `onward` handed the branch over, but the layer remembers riding cycles here: it overrules
            raw = every.iter().filter(|cc| cc.child != BACK).cloned().collect();
        }
    }
    raw.into_iter()
        .map(|cc| {
            let mut cost = cc.cost;
            if let Some(extra) = extra {
                let action = if loops.iter().any(|(c, _)| *c == cc.child) {
                    RIDE
                } else if cc.child == END {
                    ABORT
                } else {
                    ESCAPE
                };
                cost += extra[action_index(action).expect("a known action")];
            }
            Step {
                child: cc.child,
                cost: cost + step_penalty,
                phase: (phase + advance_of(g, cc.child)) % buckets,
            }
        })
        .collect()
}

/// A heap item ordered as Python orders `(cost, tie, ...)` tuples.
fn tuple_less(a: &(f64, u64, usize), b: &(f64, u64, usize)) -> bool {
    if a.0 == b.0 {
        (a.1, a.2) < (b.1, b.2)
    } else {
        a.0 < b.0
    }
}

/// The cheapest path over `(node, units, phase)` emitting at least
/// `min_chars` units - [`crate::dijkstra::dijkstra_predict`]'s contract (goal,
/// cap, fallback) with the phase added to the state and to the costs.  No
/// metacognition: one label per state cannot say which walk reached it.
#[allow(clippy::too_many_arguments)] // the search's knobs, as in Python
pub fn phase_dijkstra(
    g: &Graph,
    start_node: usize,
    start_offset: usize,
    start_phase: usize,
    min_chars: usize,
    max_chars: Option<usize>,
    step_penalty: f64,
    to_end: bool,
    costs: Option<&PhasePenalty>,
) -> Result<PathResult, String> {
    if step_penalty < 0.0 {
        return Err("step_penalty must be >= 0 (Dijkstra needs non-negative costs)".to_string());
    }
    let max_chars = max_chars.map(|cap| cap.max(min_chars));
    let buckets = buckets_of(g);
    let start_chars = start_emission(g, start_node, start_offset)?;
    let start_key = (start_node, start_chars, start_phase % buckets);
    // states are numbered in the order they are first reached
    let mut keys: Vec<(usize, usize, usize)> = vec![start_key];
    let mut ids: Map<(usize, usize, usize), usize> = map();
    ids.insert(start_key, 0);
    let mut best: Vec<f64> = vec![0.0];
    let mut prev: Vec<Option<(usize, f64)>> = vec![None];
    let mut heap: Vec<(f64, u64, usize)> = vec![(0.0, 0, 0)];
    let mut tie = 0u64;
    let mut expanded = 0usize;
    let mut goal: Option<usize> = None;
    let (mut fallback, mut fb_chars, mut fb_cost) = (0usize, start_chars, 0.0f64);
    while let Some((cost, _, id)) = heappop(&mut heap, &tuple_less) {
        if cost > best[id] {
            continue; // a stale entry
        }
        let (node, chars, phase) = keys[id];
        expanded += 1;
        if node == END || (!to_end && chars >= min_chars) {
            goal = Some(id);
            break;
        }
        if chars > fb_chars || (chars == fb_chars && cost < fb_cost) {
            (fallback, fb_chars, fb_cost) = (id, chars, cost);
        }
        if expanded >= MAX_EXPANSIONS {
            break;
        }
        if max_chars.is_some_and(|cap| chars >= cap) {
            continue;
        }
        let mut children = costs_at(g, costs, node, phase);
        onward(&mut children);
        for cc in &children {
            let nchars = chars_after(g, chars, cc.child);
            let step = cc.cost + step_penalty;
            let ncost = cost + step;
            let nkey = (cc.child, nchars, (phase + advance_of(g, cc.child)) % buckets);
            let nid = match ids.get(&nkey) {
                Some(&nid) => nid,
                None => {
                    keys.push(nkey);
                    best.push(f64::INFINITY);
                    prev.push(None);
                    ids.insert(nkey, keys.len() - 1);
                    keys.len() - 1
                }
            };
            if ncost < best[nid] {
                best[nid] = ncost;
                prev[nid] = Some((id, step));
                tie += 1;
                heappush(&mut heap, (ncost, tie, nid), &tuple_less);
            }
        }
    }
    let mut id = goal.unwrap_or(fallback);
    let mut node_ids = Vec::new();
    let mut step_costs = Vec::new();
    loop {
        node_ids.push(keys[id].0);
        match prev[id] {
            Some((from, step)) => {
                step_costs.push(step);
                id = from;
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
        None,
    ))
}

/// One label of the k-best search: a state, the step that reached it and the
/// label it came from - the chain is the walk.
struct Label {
    node: usize,
    chars: usize,
    phase: usize,
    step: f64,
    parent: Option<usize>,
}

/// A label's nodes, its step costs and where each `(node, phase)` on it first
/// appeared.
fn walk_back(chain: &[Label], idx: usize) -> (Vec<usize>, Vec<f64>, Depth) {
    let mut nodes = Vec::new();
    let mut steps = Vec::new();
    let mut seen = Vec::new();
    let mut i = Some(idx);
    while let Some(at) = i {
        let label = &chain[at];
        nodes.push(label.node);
        seen.push((label.node, label.phase));
        if label.parent.is_some() {
            steps.push(label.step);
        }
        i = label.parent;
    }
    nodes.reverse();
    steps.reverse();
    seen.reverse();
    let mut depth: Depth = map();
    for (position, key) in seen.into_iter().enumerate() {
        depth.entry(key).or_insert(position); // the first visit is what names the loop
    }
    (nodes, steps, depth)
}

/// The `k` cheapest walks, exactly - Dijkstra with `k` labels per state
/// instead of one.  Every label is a distinct walk that can be read back to
/// its own path, so the cycle it stands in is visible and the layer prices it.
/// `k = 1` is [`phase_dijkstra`].  Returns `(paths, expansions)`.
#[allow(clippy::too_many_arguments)] // the search's knobs, as in Python
pub fn phase_kbest(
    g: &Graph,
    meta: Option<&MetaLayer>,
    start_node: usize,
    start_offset: usize,
    start_phase: usize,
    min_chars: usize,
    k: usize,
    max_chars: Option<usize>,
    step_penalty: f64,
    to_end: bool,
    costs: Option<&PhasePenalty>,
) -> Result<(Vec<PathResult>, usize), String> {
    if step_penalty < 0.0 {
        return Err("step_penalty must be >= 0 (a k-best search needs non-negative costs)".to_string());
    }
    if k < 1 {
        return Err(format!("k must be >= 1, got {k}"));
    }
    let max_chars = max_chars.map(|cap| cap.max(min_chars));
    let start_chars = start_emission(g, start_node, start_offset)?;
    let phase0 = start_phase % buckets_of(g);
    let mut chain = vec![Label {
        node: start_node,
        chars: start_chars,
        phase: phase0,
        step: 0.0,
        parent: None,
    }];
    let mut heap: Vec<(f64, u64, usize)> = vec![(0.0, 0, 0)];
    let mut settled: Map<(usize, usize, usize), usize> = map();
    let mut goals: Vec<usize> = Vec::new();
    let mut expanded = 0usize;
    let (mut fallback, mut fb_chars, mut fb_cost) = (0usize, start_chars, 0.0f64);
    let empty: Depth = map();
    while goals.len() < k {
        let Some((cost, _, idx)) = heappop(&mut heap, &tuple_less) else {
            break;
        };
        let (node, chars, phase) = (chain[idx].node, chain[idx].chars, chain[idx].phase);
        let count = settled.get(&(node, chars, phase)).copied().unwrap_or(0);
        if count >= k {
            continue; // this state already has its k cheapest ways of being reached
        }
        settled.insert((node, chars, phase), count + 1);
        expanded += 1;
        if node == END || (!to_end && chars >= min_chars) {
            goals.push(idx);
            continue;
        }
        if chars > fb_chars || (chars == fb_chars && cost < fb_cost) {
            (fallback, fb_chars, fb_cost) = (idx, chars, cost);
        }
        if expanded >= MAX_EXPANSIONS {
            break;
        }
        if max_chars.is_some_and(|cap| chars >= cap) {
            continue;
        }
        let depth = if meta.is_some() {
            walk_back(&chain, idx).2
        } else {
            map()
        };
        let depth = if meta.is_some() { &depth } else { &empty };
        for step in expand(g, meta, node, phase, depth, step_penalty, costs) {
            let nchars = chars_after(g, chars, step.child);
            if max_chars.is_some_and(|cap| nchars > cap) && step.child >= FIRST {
                continue;
            }
            if settled.get(&(step.child, nchars, step.phase)).copied().unwrap_or(0) >= k {
                continue;
            }
            chain.push(Label {
                node: step.child,
                chars: nchars,
                phase: step.phase,
                step: step.cost,
                parent: Some(idx),
            });
            let at = chain.len() - 1;
            heappush(&mut heap, (cost + step.cost, at as u64, at), &tuple_less);
        }
    }
    if goals.is_empty() {
        goals.push(fallback);
    }
    let results = goals
        .into_iter()
        .map(|idx| {
            let (nodes, steps, _) = walk_back(&chain, idx);
            build_result(g, nodes, steps, start_offset, max_chars, expanded, None)
        })
        .collect();
    Ok((results, expanded))
}

/// One partial path in a beam: its tail state, its cost, the entry it grew
/// from and where it has already been.
struct Entry {
    node: usize,
    chars: usize,
    phase: usize,
    cost: f64,
    parent: Option<usize>,
    step: f64,
    depth: std::rc::Rc<Depth>,
}

/// An entry's nodes and step costs, root first.
fn entry_path(arena: &[Entry], idx: usize) -> (Vec<usize>, Vec<f64>) {
    let mut nodes = Vec::new();
    let mut steps = Vec::new();
    let mut at = Some(idx);
    while let Some(i) = at {
        nodes.push(arena[i].node);
        if arena[i].parent.is_some() {
            steps.push(arena[i].step);
        }
        at = arena[i].parent;
    }
    nodes.reverse();
    steps.reverse();
    (nodes, steps)
}

/// A finished path in a beam's `heapq`: `(-score, seq, entry)`.
fn done_less(a: &(f64, u64, usize), b: &(f64, u64, usize)) -> bool {
    if a.0 == b.0 {
        a.1 < b.1
    } else {
        a.0 < b.0
    }
}

/// Stable sort by a float key, as Python's `sorted(key=...)` orders them.
fn sort_by_key(items: &mut [usize], key: impl Fn(usize) -> f64) {
    items.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
}

/// The `k` cheapest and the `k` dearest complete paths, with metacognition
/// on cycles: a top beam of the cheapest partial paths and a bottom beam of
/// the dearest, both `beam` wide (0: the default width), over the
/// phase-unrolled graph.  Returns `(top, bottom, expansions)`.
#[allow(clippy::too_many_arguments)] // the search's knobs, as in Python
pub fn phase_beam(
    g: &Graph,
    meta: Option<&MetaLayer>,
    start_node: usize,
    start_offset: usize,
    start_phase: usize,
    min_chars: usize,
    k: usize,
    beam: usize,
    max_chars: Option<usize>,
    step_penalty: f64,
    to_end: bool,
    costs: Option<&PhasePenalty>,
) -> Result<(Vec<PathResult>, Vec<PathResult>, usize), String> {
    let width = if beam == 0 { crate::beam::default_beam(k) } else { beam };
    let max_chars = max_chars.map(|cap| cap.max(min_chars));
    let start_chars = start_emission(g, start_node, start_offset)?;
    let phase0 = start_phase % buckets_of(g);
    let mut root_depth: Depth = map();
    root_depth.insert((start_node, phase0), 0);
    let mut arena = vec![Entry {
        node: start_node,
        chars: start_chars,
        phase: phase0,
        cost: 0.0,
        parent: None,
        step: 0.0,
        depth: std::rc::Rc::new(root_depth),
    }];
    let mut top_beam: Vec<usize> = vec![0];
    let mut bottom_beam: Vec<usize> = vec![0];
    let mut top_done: Vec<(f64, u64, usize)> = Vec::new();
    let mut bottom_done: Vec<(f64, u64, usize)> = Vec::new();
    let keep = k.max(1);
    let mut seq = 0u64;
    let mut expanded = 0usize;
    let mut depth = 0usize;
    let limit_chars = max_chars;
    let mut bottom_limit = max_chars;
    while (!top_beam.is_empty() || !bottom_beam.is_empty()) && depth < MAX_STEPS {
        depth += 1;
        let mut nxt_top: Vec<usize> = Vec::new();
        let mut nxt_bottom: Vec<usize> = Vec::new();
        for side in 0..2 {
            let source = if side == 0 {
                std::mem::take(&mut top_beam)
            } else {
                std::mem::take(&mut bottom_beam)
            };
            let cap_chars = if side == 0 { limit_chars } else { bottom_limit };
            for &at in &source {
                let (node, chars, phase, cost) = (arena[at].node, arena[at].chars, arena[at].phase, arena[at].cost);
                if node == END || cap_chars.is_some_and(|cap| chars >= cap) {
                    continue;
                }
                expanded += 1;
                let here = std::rc::Rc::clone(&arena[at].depth);
                for step in expand(g, meta, node, phase, &here, step_penalty, costs) {
                    let c = step.child;
                    let nchars = chars_after(g, chars, c);
                    if cap_chars.is_some_and(|cap| nchars > cap) && c >= FIRST {
                        continue;
                    }
                    let key = (c, step.phase);
                    let child_depth = if here.contains_key(&key) {
                        std::rc::Rc::clone(&here)
                    } else {
                        let mut grown = (*here).clone();
                        grown.insert(key, depth);
                        std::rc::Rc::new(grown)
                    };
                    arena.push(Entry {
                        node: c,
                        chars: nchars,
                        phase: step.phase,
                        cost: cost + step.cost,
                        parent: Some(at),
                        step: step.cost,
                        depth: child_depth,
                    });
                    let child = arena.len() - 1;
                    let complete = c == END || (!to_end && nchars >= min_chars);
                    if complete {
                        seq += 1;
                        let done = if side == 0 { &mut top_done } else { &mut bottom_done };
                        let score = if side == 0 {
                            arena[child].cost
                        } else {
                            -arena[child].cost
                        };
                        if done.len() < keep {
                            heappush(done, (-score, seq, child), &done_less);
                        } else if -score > done[0].0 {
                            heappop(done, &done_less);
                            heappush(done, (-score, seq, child), &done_less);
                        }
                        if c == END {
                            continue;
                        }
                    }
                    if side == 0 {
                        nxt_top.push(child);
                    } else {
                        nxt_bottom.push(child);
                    }
                }
            }
        }
        sort_by_key(&mut nxt_top, |i| arena[i].cost);
        sort_by_key(&mut nxt_bottom, |i| -arena[i].cost);
        nxt_top.truncate(width);
        nxt_bottom.truncate(width);
        top_beam = nxt_top;
        bottom_beam = nxt_bottom;
        if bottom_limit.is_none() && top_done.len() >= keep {
            // no cap and running to the end: keep the dearest paths roughly as long as the cheapest
            let longest = top_done.iter().map(|&(_, _, e)| arena[e].chars).max().unwrap_or(0);
            bottom_limit = Some(2 * longest);
        }
        if top_done.len() >= keep && !top_beam.is_empty() {
            let mut worst_kept = -top_done[0].0;
            for &(s, _, _) in &top_done[1..] {
                if -s > worst_kept {
                    worst_kept = -s;
                }
            }
            if arena[top_beam[0]].cost >= worst_kept {
                top_beam.clear(); // nothing left can beat the k-th finished path
            }
        }
        if top_beam.is_empty() && bottom_beam.is_empty() {
            break;
        }
    }
    let results = |done: &[(f64, u64, usize)]| -> Vec<PathResult> {
        done.iter()
            .map(|&(_, _, e)| {
                let (nodes, steps) = entry_path(&arena, e);
                build_result(g, nodes, steps, start_offset, max_chars, expanded, None)
            })
            .collect()
    };
    let mut top = results(&top_done);
    top.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(k);
    let taken: Vec<String> = top.iter().map(|r| r.text.clone()).collect();
    let mut bottom = results(&bottom_done);
    bottom.sort_by(|a, b| (-a.cost).partial_cmp(&-b.cost).unwrap_or(std::cmp::Ordering::Equal));
    let bottom: Vec<PathResult> = bottom
        .into_iter()
        .filter(|r| !taken.contains(&r.text))
        .take(k)
        .collect();
    Ok((top, bottom, expanded))
}

/// One stochastic walk over the phase-unrolled graph, the layer included:
/// each child is drawn from `softmax(-cost / temperature)`, `temperature == 0`
/// is greedy.  The walk stops at `END` or once it has emitted `max_chars`.
#[allow(clippy::too_many_arguments)] // the search's knobs, as in Python
pub fn phase_walk(
    g: &Graph,
    meta: Option<&MetaLayer>,
    start_node: usize,
    start_offset: usize,
    start_phase: usize,
    max_chars: Option<usize>,
    temperature: f64,
    rng: &mut Mt19937,
    costs: Option<&PhasePenalty>,
) -> Result<PathResult, String> {
    if temperature < 0.0 {
        return Err("temperature must be >= 0".to_string());
    }
    let mut node = start_node;
    let mut phase = start_phase % buckets_of(g);
    let mut chars = start_emission(g, start_node, start_offset)?;
    let mut node_ids = vec![node];
    let mut step_costs: Vec<f64> = Vec::new();
    let mut depth: Depth = map();
    depth.insert((node, phase), 0);
    let mut steps = 0usize;
    loop {
        if node == END || max_chars.is_some_and(|cap| chars >= cap) {
            break;
        }
        let options = expand(g, meta, node, phase, &depth, 0.0, costs);
        if options.is_empty() {
            break;
        }
        let pick = if temperature == 0.0 || options.len() == 1 {
            let mut pick = 0;
            for (i, o) in options.iter().enumerate().skip(1) {
                if o.cost < options[pick].cost {
                    pick = i;
                }
            }
            pick
        } else {
            let inv_t = 1.0 / temperature;
            let mut lowest = options[0].cost;
            for o in &options[1..] {
                if o.cost < lowest {
                    lowest = o.cost;
                }
            }
            let weights: Vec<f64> = options.iter().map(|o| (-(o.cost - lowest) * inv_t).exp()).collect();
            let r = rng.next_f64() * fsum(&weights);
            let mut pick = options.len() - 1;
            let mut acc = 0.0;
            for (i, w) in weights.iter().enumerate() {
                acc += w;
                if r < acc {
                    pick = i;
                    break;
                }
            }
            pick
        };
        let Step {
            child: c,
            cost,
            phase: nphase,
        } = options.into_iter().nth(pick).expect("picked");
        step_costs.push(cost);
        node_ids.push(c);
        chars = chars_after(g, chars, c);
        steps += 1;
        depth.entry((c, nphase)).or_insert(steps);
        node = c;
        phase = nphase;
    }
    Ok(build_result(
        g,
        node_ids,
        step_costs,
        start_offset,
        max_chars,
        steps,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::model::{Model, TrainOptions};
    use crate::resonance::ResonantOptions;

    fn trained() -> Model {
        let mut m = Model::new_resonant(7, &ResonantOptions::default(), Encoding::default()).unwrap();
        let texts: Vec<String> = ["the cat sat on the mat", "the cat ate the rat", "a cat sat on a hat"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        m.train(&texts, &TrainOptions::default()).unwrap();
        m.g.prepare();
        m
    }

    #[test]
    fn kbest_of_one_is_dijkstra() {
        let m = trained();
        let d = phase_dijkstra(&m.g, crate::graph::START, 0, 0, 5, None, 0.0, true, None).unwrap();
        let (k, _) = phase_kbest(&m.g, None, crate::graph::START, 0, 0, 5, 1, None, 0.0, true, None).unwrap();
        assert_eq!(d.node_ids, k[0].node_ids);
        assert_eq!(d.cost, k[0].cost);
        assert!(d.reached_end);
    }

    #[test]
    fn kbest_returns_paths_in_cost_order() {
        let m = trained();
        let meta = m.res.as_ref().map(|r| r.metacog.clone());
        let (found, _) = phase_kbest(
            &m.g,
            meta.as_ref(),
            crate::graph::START,
            0,
            0,
            0,
            4,
            Some(40),
            0.0,
            true,
            None,
        )
        .unwrap();
        assert!(!found.is_empty());
        for pair in found.windows(2) {
            assert!(pair[0].cost <= pair[1].cost);
        }
    }

    #[test]
    fn the_beam_returns_cheapest_and_dearest() {
        let m = trained();
        let meta = m.res.as_ref().map(|r| r.metacog.clone());
        let (top, bottom, n) = phase_beam(
            &m.g,
            meta.as_ref(),
            crate::graph::START,
            0,
            0,
            0,
            3,
            0,
            Some(30),
            0.0,
            true,
            None,
        )
        .unwrap();
        assert!(n > 0);
        assert!(!top.is_empty());
        for r in &bottom {
            assert!(!top.iter().any(|t| t.text == r.text));
        }
    }

    #[test]
    fn a_greedy_walk_never_draws() {
        let m = trained();
        let mut rng = Mt19937::new(3);
        let before = rng.state();
        let w = phase_walk(&m.g, None, crate::graph::START, 0, 0, Some(40), 0.0, &mut rng, None).unwrap();
        assert_eq!(rng.state(), before);
        assert!(!w.text.is_empty());
        assert!(phase_walk(&m.g, None, crate::graph::START, 0, 0, Some(40), -1.0, &mut rng, None).is_err());
    }
}
