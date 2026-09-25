//! Paths: what a *walk* did, not what an edge did.
//!
//! An edge is the right move in one sentence and the wrong one in another, so
//! counting rewards per edge blurs the two together.  A path context is the step
//! in the company it kept - the node that called the edge's parent, and the edge
//! it then took - and it carries three numbers: how often the step was seen at
//! all, how often it was part of an output judged correct, and how often it was
//! part of one judged wrong.  Those counters then price the step: the same edge
//! is cheap for the walk that was right here and dear for the one that was wrong.
//!
//! A context is born when a path is judged and is kept up to date by every later
//! traversal; an unjudged pass never creates one, so training a corpus cannot
//! fill the table with the second-order counts of a whole language.

use crate::graph::{Graph, Transition, START};
use crate::weights::SMOOTHING;

/// One step in its context: the node before the edge's parent, and the edge.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PathKey {
    pub prev: usize,
    pub edge: usize,
}

/// What a context did.
#[derive(Clone, Copy, Default, Debug)]
pub struct PathRow {
    pub seen: i64,
    pub correct: i64,
    pub incorrect: i64,
}

/// What a walk was judged to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathOutcome {
    Unjudged,
    Correct,
    Incorrect,
}

/// How much of the graph has been judged as paths rather than as edges.
#[derive(Clone, Copy, Default, Debug)]
pub struct PathTotals {
    pub contexts: usize,
    pub judged: usize,
    pub seen: i64,
    pub correct: i64,
    pub incorrect: i64,
}

impl Graph {
    /// Counts one walk of a whole text, step by step, in its own context.  The
    /// context of step `k` is the parent of step `k - 1` (`START` for the first
    /// step, which has none); `create` decides whether contexts never seen
    /// before are added.
    pub fn record_path(&mut self, transitions: &[Transition], outcome: PathOutcome, create: bool) -> usize {
        if transitions.is_empty() || (!create && self.paths.is_empty()) {
            return 0;
        }
        let mut touched = 0;
        // the origin is the context of its own first step (START for a text, THINK for a thought)
        let first = transitions[0].p;
        let mut prev = if crate::graph::is_origin(first) { first } else { START };
        for (i, t) in transitions.iter().enumerate() {
            if i > 0 {
                prev = transitions[i - 1].p;
            }
            if t.e >= self.edge_alive.len() || !self.edge_alive[t.e] {
                continue;
            }
            let Some(row) = self.path_row(prev, t.e, create) else {
                continue;
            };
            row.seen += 1;
            match outcome {
                PathOutcome::Correct => row.correct += 1,
                PathOutcome::Incorrect => row.incorrect += 1,
                PathOutcome::Unjudged => {}
            }
            touched += 1;
        }
        if touched > 0 && outcome != PathOutcome::Unjudged {
            self.version.add(1); // the contexts that moved make their node's costs stale
        }
        touched
    }

    /// Judges single steps - what a diff blames or teaches - rather than a whole walk.
    pub fn mark_steps(&mut self, steps: &[PathKey], correct: bool) -> usize {
        let mut marked = 0;
        for step in steps {
            if step.edge >= self.edge_alive.len() || !self.edge_alive[step.edge] {
                continue;
            }
            let Some(row) = self.path_row(step.prev, step.edge, true) else {
                continue;
            };
            row.seen += 1;
            if correct {
                row.correct += 1;
            } else {
                row.incorrect += 1;
            }
            marked += 1;
        }
        if marked > 0 {
            self.version.add(1);
        }
        marked
    }

    /// One context's counters, created on demand (and indexed both ways so a
    /// split can find them).
    // not the entry API: a missing context either bows out (`create` is false)
    // or has to be indexed in two more maps, neither of which an `or_insert_with`
    // closure may borrow while this one is held
    #[allow(clippy::map_entry)]
    fn path_row(&mut self, prev: usize, edge: usize, create: bool) -> Option<&mut PathRow> {
        let key = PathKey { prev, edge };
        if !self.paths.contains_key(&key) {
            if !create {
                return None;
            }
            self.paths.insert(key, PathRow::default());
            self.paths_by_edge.entry(edge).or_default().insert(prev);
            self.paths_by_prev.entry(prev).or_default().insert(edge);
        }
        self.paths.get_mut(&key)
    }

    fn drop_path(&mut self, prev: usize, edge: usize) -> Option<PathRow> {
        let row = self.paths.remove(&PathKey { prev, edge })?;
        if let Some(holder) = self.paths_by_edge.get_mut(&edge) {
            holder.remove(&prev);
            if holder.is_empty() {
                self.paths_by_edge.remove(&edge);
            }
        }
        if let Some(holder) = self.paths_by_prev.get_mut(&prev) {
            holder.remove(&edge);
            if holder.is_empty() {
                self.paths_by_prev.remove(&prev);
            }
        }
        Some(row)
    }

    /// Adds one context's counters into another (a split or a merge moved the step).
    fn add_path(&mut self, prev: usize, edge: usize, row: PathRow) {
        if let Some(into) = self.path_row(prev, edge, true) {
            into.seen += row.seen;
            into.correct += row.correct;
            into.incorrect += row.incorrect;
        }
    }

    /// `log((correct + s) / (incorrect + s))` of a context: zero until a path is
    /// judged, and symmetric.
    /// Whether any context has been judged at all: the searches skip the
    /// context lookup entirely where nothing has.
    pub fn paths_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn path_term(&self, prev: usize, edge: usize) -> f64 {
        match self.paths.get(&PathKey { prev, edge }) {
            Some(row) if row.correct != 0 || row.incorrect != 0 => {
                ((row.correct as f64 + SMOOTHING) / (row.incorrect as f64 + SMOOTHING)).ln()
            }
            _ => 0.0,
        }
    }

    /// Sets one context's counters as a file recorded them.
    pub(crate) fn load_path(&mut self, prev: usize, edge: usize, seen: i64, correct: i64, incorrect: i64) {
        if edge >= self.edge_alive.len() || prev >= self.labels.len() {
            return;
        }
        if let Some(row) = self.path_row(prev, edge, true) {
            row.seen = seen;
            row.correct = correct;
            row.incorrect = incorrect;
        }
    }

    /// How often a walk that came from `prev` was judged wrong here: the one
    /// number the least-punished traversal steers by, counted against nothing.
    pub fn path_incorrect(&self, prev: usize, edge: usize) -> i64 {
        self.paths
            .get(&PathKey { prev, edge })
            .map(|row| row.incorrect)
            .unwrap_or(0)
    }

    /// One context's counters, or `None` when it has never been walked.
    pub fn path_stats_of(&self, prev: usize, edge: usize) -> Option<PathRow> {
        self.paths.get(&PathKey { prev, edge }).copied()
    }

    /// Counts the whole table.
    pub fn path_totals(&self) -> PathTotals {
        let mut out = PathTotals {
            contexts: self.paths.len(),
            ..Default::default()
        };
        for row in self.paths.values() {
            out.seen += row.seen;
            out.correct += row.correct;
            out.incorrect += row.incorrect;
            if row.correct != 0 || row.incorrect != 0 {
                out.judged += 1;
            }
        }
        out
    }

    /// Every context, most judged first (`limit` 0 returns all of them).
    pub fn path_contexts(&self, limit: usize) -> Vec<(PathKey, PathRow)> {
        let mut out: Vec<(PathKey, PathRow)> = self.paths.iter().map(|(&k, &v)| (k, v)).collect();
        out.sort_by(|(ka, a), (kb, b)| {
            let (ja, jb) = (a.correct + a.incorrect, b.correct + b.incorrect);
            jb.cmp(&ja)
                .then(b.seen.cmp(&a.seen))
                .then(ka.prev.cmp(&kb.prev))
                .then(ka.edge.cmp(&kb.edge))
        });
        if limit > 0 && out.len() > limit {
            out.truncate(limit);
        }
        out
    }

    /// Follows the contexts through a split: `q -> P -> c` becomes
    /// `q -> A -> B -> c`, so a context `(q, e)` of a moved edge becomes
    /// `(A, e)` and the new edge `A -> B` inherits `(q, A->B)`, the step `q`
    /// now calls.
    pub(crate) fn split_paths(&mut self, a: usize, b: usize, moved: &[usize], bridge: Option<usize>) {
        let _ = b;
        if self.paths.is_empty() || moved.is_empty() {
            return;
        }
        for &edge in moved {
            let callers: Vec<usize> = match self.paths_by_edge.get(&edge) {
                Some(set) => set.iter().copied().collect(),
                None => continue,
            };
            for prev in callers {
                let Some(row) = self.drop_path(prev, edge) else {
                    continue;
                };
                self.add_path(a, edge, row);
                if let Some(bridge) = bridge {
                    self.add_path(prev, bridge, row);
                }
            }
        }
        self.ctx_version = crate::counter::INVALID_STAMP;
    }

    /// Follows the contexts through a merge: the chain was unary, so what it
    /// knew about was never a choice.  The edge `p -> c` dies with its
    /// contexts, and so do the contexts of `c`'s out-edges; contexts that
    /// arrive through `c` are re-keyed to `p`.
    pub(crate) fn merge_paths(&mut self, p: usize, child: usize, dying: usize, moved: &[usize]) {
        if self.paths.is_empty() {
            return;
        }
        let callers: Vec<usize> = self
            .paths_by_edge
            .get(&dying)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for prev in callers {
            self.drop_path(prev, dying);
        }
        for &edge in moved {
            self.drop_path(p, edge);
        }
        let called: Vec<usize> = self
            .paths_by_prev
            .get(&child)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for edge in called {
            if let Some(row) = self.drop_path(child, edge) {
                self.add_path(p, edge, row);
            }
        }
        self.ctx_version = crate::counter::INVALID_STAMP;
    }
}
