//! The negative network: the failures, and *why* they were failures.
//!
//! A copy of the network that keeps only its **negative portions**.  The same
//! self-compressing cyclic graph - same encoding, same splits and merges, same
//! searches - but nothing in it comes from correct data: **every node and every
//! edge exists because something went wrong there**, and every edge remembers
//! *why*, with how much blame each reason carries.
//!
//! # The tutor supplies the negatives
//!
//! The negative network never invents failures.  They arrive from outside - a
//! tutor's critique, a sandbox error, a discriminator, a thumbs-down in the
//! frontend - and each carries a reason tag and a severity.  Training on
//! anything else is a category error: a pass over this model *is* a blame pass.
//!
//! # What an edge remembers
//!
//! * `blame` - the summed severity of the failures that ran through it,
//! * `fails` - how many failing texts ran through it,
//! * `clear` - how much *cleared* (tutor-passed) text ran through it,
//! * `reasons` - the tutor's reasons split by the blame each contributed, at
//!   most [`MAX_EDGE_REASONS`] of them.
//!
//! The **net evidence** against an edge is `max(0, blame - clear_scale * clear)`:
//! blame and clearing cancel, so an edge the tutor's passes cross as often as
//! its failures do carries no verdict at all - which is how a common fragment
//! stays out of the judgement.
//!
//! # The weight function
//!
//! No learning rate and no gradient, as in the count model.  An edge's weight
//! is its share of the failure mass leaving its parent:
//!
//! ```text
//! net    = max(0, blame - clear_scale * clear)
//! R_bad  = (net + s) / (net leaving the parent + s * children)
//! weight = share_scale * log(R_bad) + blame_scale * log(1 + net)
//! ```
//!
//! with `s = 0.5`.  A softmax over the children therefore reads
//! `P(child | parent) ∝ R_bad * (1 + net) ** blame_scale`: the network models
//! **how text goes wrong**.
//!
//! This mirrors `radixnet/negative.py` and `go/radixnet/negative.go`, down to
//! the model file they share.

use crate::counter::Counter;
use crate::encoding::Encoding;
use crate::graph::{Graph, GraphOptions};
use crate::hash::{map, Map};
use crate::weights::SMOOTHING;

/// The model file format of the negative network.
pub const NEGATIVE_FORMAT: &str = "radixnet-negative";

/// Recorded when a failure arrives without a reason (a tutor always gives one).
pub const UNSPECIFIED_REASON: &str = "unspecified";

/// How many reasons an edge keeps; the weakest is dropped when a ninth appears,
/// and the graph-level totals keep every one of them.
pub const MAX_EDGE_REASONS: usize = 8;

/// How many failures the journal keeps (the newest win).
pub const MAX_LOG_ENTRIES: usize = 200;

/// How much of a failing text the journal keeps.
const MAX_LOG_TEXT_CHARS: usize = 160;

/// The scales of the blame weight function.
#[derive(Clone, Copy, Debug)]
pub struct NegativeOptions {
    pub share_scale: f64,
    pub blame_scale: f64,
    pub clear_scale: f64,
    /// How the failure structure reads text.
    ///
    /// It has to be the encoding of the count model it accompanies: the two
    /// walk the same texts, and a gram of one is meaningless to the other.
    pub encoding: Encoding,
}

impl Default for NegativeOptions {
    /// The Python defaults: the edge's share of the failure mass leaving its
    /// parent, with one cleared traversal cancelling one ordinary failure.
    fn default() -> NegativeOptions {
        NegativeOptions {
            share_scale: 1.0,
            blame_scale: 0.0,
            clear_scale: 1.0,
            encoding: Encoding::default(),
        }
    }
}

/// One reason's contribution to an edge's blame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReasonBlame {
    pub id: usize,
    pub blame: f64,
}

/// Everything a negative graph keeps that a count graph does not.
///
/// `None` on a count / reward graph, so the count model pays nothing for it.
#[derive(Clone, Debug)]
pub struct NegativeData {
    // per edge, parallel to Graph::edge_w
    pub blame: Vec<f64>,
    pub fails: Vec<i64>,
    pub clear: Vec<f64>,
    pub reasons: Vec<Vec<ReasonBlame>>,
    /// How often each edge's fail counter wrapped (absent = never).
    pub fails_resets: Map<usize, i64>,

    // the reason registry
    pub reason_names: Vec<String>,
    pub reason_blame: Vec<f64>,
    pub reason_fails: Vec<i64>,
    pub reason_fails_resets: Map<usize, i64>,
    reason_ids: Map<String, usize>,

    pub total_blame: f64,
    pub total_fails: Counter,
    pub total_clear: f64,

    pub share_scale: f64,
    pub blame_scale: f64,
    pub clear_scale: f64,
}

impl NegativeData {
    fn new(o: &NegativeOptions) -> NegativeData {
        NegativeData {
            blame: Vec::new(),
            fails: Vec::new(),
            clear: Vec::new(),
            reasons: Vec::new(),
            fails_resets: map(),
            reason_names: Vec::new(),
            reason_blame: Vec::new(),
            reason_fails: Vec::new(),
            reason_fails_resets: map(),
            reason_ids: map(),
            total_blame: 0.0,
            total_fails: Counter::default(),
            total_clear: 0.0,
            share_scale: o.share_scale,
            blame_scale: o.blame_scale,
            clear_scale: o.clear_scale,
        }
    }

    /// Grows the per-edge arrays for one new edge.
    pub(crate) fn append_edge(&mut self) {
        self.blame.push(0.0);
        self.fails.push(0);
        self.clear.push(0.0);
        self.reasons.push(Vec::new());
    }

    /// The id of a reason tag, registering it on first sight.
    pub fn reason_id(&mut self, reason: &str) -> usize {
        let label = clean_reason(reason);
        if let Some(&id) = self.reason_ids.get(&label) {
            return id;
        }
        let id = self.reason_names.len();
        self.reason_ids.insert(label.clone(), id);
        self.reason_names.push(label);
        self.reason_blame.push(0.0);
        self.reason_fails.push(0);
        id
    }

    /// The tag of a reason id, or [`UNSPECIFIED_REASON`] for an unknown one.
    pub fn reason_label(&self, id: usize) -> &str {
        self.reason_names
            .get(id)
            .map(|s| s.as_str())
            .unwrap_or(UNSPECIFIED_REASON)
    }

    /// The id of an already registered tag.
    pub fn lookup_reason(&self, reason: &str) -> Option<usize> {
        self.reason_ids.get(&clean_reason(reason)).copied()
    }

    /// Rebuilds the name index after a load.
    pub(crate) fn reindex_reasons(&mut self) {
        self.reason_ids = map();
        for (id, name) in self.reason_names.iter().enumerate() {
            self.reason_ids.insert(name.clone(), id);
        }
    }
}

/// Normalises a reason tag: one trimmed lower-case line, never empty.
pub fn clean_reason(reason: &str) -> String {
    let text: String = reason.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    if text.is_empty() {
        return UNSPECIFIED_REASON.to_string();
    }
    text.chars().take(60).collect()
}

impl Graph {
    /// An empty negative graph: the same cyclic graph with the blame arrays and
    /// the reason registry attached.
    pub fn new_negative(seed: i64, o: &NegativeOptions) -> Result<Graph, String> {
        let mut g = Graph::new(
            seed,
            GraphOptions {
                encoding: o.encoding,
                ..Default::default()
            },
        )?;
        g.neg = Some(Box::new(NegativeData::new(o)));
        Ok(g)
    }

    /// Whether this graph keeps blame instead of counts and rewards.
    pub fn is_negative(&self) -> bool {
        self.neg.is_some()
    }

    /// The net evidence against an edge: `max(0, blame - clear_scale * clear)`.
    ///
    /// Blame and clearing cancel: an edge the tutor's passes cross as often as
    /// its failures do is not evidence of anything, however busy it is.
    pub fn evidence(&self, e: usize) -> f64 {
        let Some(neg) = &self.neg else { return 0.0 };
        if e >= neg.blame.len() || !self.edge_alive[e] {
            return 0.0;
        }
        (neg.blame[e] - neg.clear_scale * neg.clear[e]).max(0.0)
    }

    /// The blame weight of one edge from its net evidence.
    pub fn negative_edge_weight(&self, evidence: f64, parent_evidence: f64, degree: usize) -> f64 {
        let Some(neg) = &self.neg else { return 0.0 };
        let evidence = evidence.max(0.0);
        let d = degree.max(1) as f64;
        let r_bad = (evidence + SMOOTHING) / (parent_evidence.max(0.0) + SMOOTHING * d);
        neg.share_scale * r_bad.ln() + neg.blame_scale * evidence.ln_1p()
    }

    /// Writes the blame weight to every edge leaving `p`.
    pub(crate) fn recompute_negative_row(&mut self, p: usize) {
        if !self.alive[p] || self.children[p].size() == 0 {
            return;
        }
        let edges: Vec<usize> = self.children[p].edges.clone();
        let degree = edges.len();
        let net: Vec<f64> = edges.iter().map(|&e| self.evidence(e)).collect();
        let total: f64 = net.iter().sum();
        for (i, &e) in edges.iter().enumerate() {
            self.edge_w[e] = self.negative_edge_weight(net[i], total, degree);
        }
    }

    /// Blames every listed alive edge by `severity` for `reason`; returns how
    /// many were blamed.
    pub fn record_failure(&mut self, edges: &[usize], severity: f64, reason: &str) -> usize {
        if self.neg.is_none() {
            return 0;
        }
        let amount = severity.abs();
        if !amount.is_finite() {
            return 0;
        }
        let id = self.neg.as_mut().expect("checked").reason_id(reason);
        let mut touched = 0;
        let mut parents: Vec<usize> = Vec::new();
        {
            let alive = &self.edge_alive;
            let parent_of = &self.edge_parent;
            let neg = self.neg.as_mut().expect("checked");
            for &e in edges {
                if e >= neg.blame.len() || !alive[e] {
                    continue;
                }
                neg.blame[e] += amount;
                neg.fails[e] += 1;
                if amount > 0.0 {
                    add_reason_blame(&mut neg.reasons[e], id, amount);
                }
                parents.push(parent_of[e]);
                touched += 1;
            }
            if touched > 0 {
                neg.total_blame += amount * touched as f64;
                neg.total_fails.add(touched as i64);
                neg.reason_blame[id] += amount * touched as f64;
                neg.reason_fails[id] += 1;
            }
        }
        for p in parents {
            self.dirty.insert(p);
        }
        if touched > 0 {
            self.version.add(1);
        }
        touched
    }

    /// [`Graph::record_failure`] with every edge blamed `severity` times its
    /// share - the attention band's charges ([`crate::attention`]), so a step
    /// that only glimpsed a change at the edge of its gram carries less of it.
    /// Every listed edge still counts one failure: it was part of one.
    pub fn record_failure_shared(&mut self, edges: &[usize], severity: f64, shares: &[f64], reason: &str) -> usize {
        if self.neg.is_none() || shares.len() != edges.len() {
            return 0;
        }
        let amount = severity.abs();
        if !amount.is_finite() {
            return 0;
        }
        let id = self.neg.as_mut().expect("checked").reason_id(reason);
        let mut touched = 0;
        let mut added = 0.0; // the blame laid, edge by edge
        let mut parents: Vec<usize> = Vec::new();
        {
            let alive = &self.edge_alive;
            let parent_of = &self.edge_parent;
            let neg = self.neg.as_mut().expect("checked");
            for (&e, &share) in edges.iter().zip(shares) {
                if e >= neg.blame.len() || !alive[e] {
                    continue;
                }
                let owed = amount * share;
                neg.blame[e] += owed;
                neg.fails[e] += 1;
                if owed > 0.0 {
                    add_reason_blame(&mut neg.reasons[e], id, owed);
                }
                parents.push(parent_of[e]);
                added += owed;
                touched += 1;
            }
            if touched > 0 {
                neg.total_blame += added;
                neg.total_fails.add(touched as i64);
                neg.reason_blame[id] += added;
                neg.reason_fails[id] += 1;
            }
        }
        for p in parents {
            self.dirty.insert(p);
        }
        if touched > 0 {
            self.version.add(1);
        }
        touched
    }

    /// Credits every listed alive edge with `weight` of cleared text; returns
    /// how many were credited.
    pub fn record_clear(&mut self, edges: &[usize], weight: f64) -> usize {
        if self.neg.is_none() {
            return 0;
        }
        let amount = weight.abs();
        if !amount.is_finite() {
            return 0;
        }
        let mut touched = 0;
        let mut parents: Vec<usize> = Vec::new();
        {
            let alive = &self.edge_alive;
            let parent_of = &self.edge_parent;
            let neg = self.neg.as_mut().expect("checked");
            for &e in edges {
                if e >= neg.clear.len() || !alive[e] {
                    continue;
                }
                neg.clear[e] += amount;
                parents.push(parent_of[e]);
                touched += 1;
            }
            if touched > 0 {
                neg.total_clear += amount * touched as f64;
            }
        }
        for p in parents {
            self.dirty.insert(p);
        }
        if touched > 0 {
            self.version.add(1);
        }
        touched
    }

    /// Whether a unary chain's edge carries evidence and must survive
    /// compression.
    ///
    /// A step *inside* a node has no edge to carry blame, so a correction that
    /// blamed a single transition would be folded away and forgotten.
    /// Everything the tutor never ruled on still merges.
    pub(crate) fn blocks_merge(&self, e: usize) -> bool {
        let Some(neg) = &self.neg else { return false };
        e < neg.blame.len() && (neg.blame[e] > 0.0 || neg.clear[e] > 0.0)
    }

    /// Forgets a reason, or scales every blame down by `factor`.
    ///
    /// `factor` 0 removes the blame entirely; 0.5 halves it. An edge left with
    /// no evidence at all is not deleted - the structure is what the failures
    /// built, and a node that stops being evidence is still where one was.
    pub fn forget(&mut self, reason: Option<&str>, factor: f64) -> Result<ForgetResult, String> {
        if self.neg.is_none() {
            return Err("not a negative graph".to_string());
        }
        if !(0.0..=1.0).contains(&factor) {
            return Err(format!("factor must be between 0 and 1, got {factor}"));
        }
        let keep = factor;
        let target = reason.map(clean_reason);
        let id = match &target {
            Some(label) => match self.neg.as_ref().expect("checked").lookup_reason(label) {
                Some(id) => Some(id),
                // a reason nothing was ever blamed for: nothing to forget, which
                // is an answer rather than an error
                None => {
                    return Ok(ForgetResult {
                        reason: label.clone(),
                        edges: 0,
                        blame_removed: 0.0,
                    })
                }
            },
            None => None,
        };
        let mut edges = 0;
        let mut removed = 0.0;
        let mut parents: Vec<usize> = Vec::new();
        {
            let alive = &self.edge_alive;
            let parent_of = &self.edge_parent;
            let neg = self.neg.as_mut().expect("checked");
            for e in 0..neg.blame.len() {
                if !alive[e] {
                    continue;
                }
                let before = neg.blame[e];
                let gone = match id {
                    // one reason: only what it contributed leaves
                    Some(id) => {
                        let mut share = 0.0;
                        neg.reasons[e].retain(|r| {
                            if r.id != id {
                                return true;
                            }
                            share = r.blame * (1.0 - keep);
                            if keep > 0.0 {
                                return true;
                            }
                            false
                        });
                        if keep > 0.0 {
                            for r in neg.reasons[e].iter_mut().filter(|r| r.id == id) {
                                r.blame *= keep;
                            }
                        }
                        share.min(before)
                    }
                    // everything: the whole edge scales
                    None => {
                        for r in neg.reasons[e].iter_mut() {
                            r.blame *= keep;
                        }
                        if keep == 0.0 {
                            neg.reasons[e].clear();
                        }
                        before * (1.0 - keep)
                    }
                };
                if gone <= 0.0 {
                    continue;
                }
                neg.blame[e] = (before - gone).max(0.0);
                removed += gone;
                edges += 1;
                parents.push(parent_of[e]);
            }
            neg.total_blame = (neg.total_blame - removed).max(0.0);
            match id {
                Some(id) => {
                    neg.reason_blame[id] *= keep;
                    if keep == 0.0 {
                        neg.reason_fails[id] = 0;
                    }
                }
                None => {
                    for blame in neg.reason_blame.iter_mut() {
                        *blame *= keep;
                    }
                    if keep == 0.0 {
                        for fails in neg.reason_fails.iter_mut() {
                            *fails = 0;
                        }
                        neg.total_clear = 0.0;
                        for clear in neg.clear.iter_mut() {
                            *clear = 0.0;
                        }
                    }
                }
            }
        }
        for p in parents {
            self.dirty.insert(p);
        }
        if edges > 0 {
            self.version.add(1);
        }
        crate::log_info!(
            "negative",
            "forgot {}: {edges} edge(s), {removed:.2} blame",
            target.as_deref().unwrap_or("everything")
        );
        Ok(ForgetResult {
            reason: target.unwrap_or_else(|| "*".to_string()),
            edges,
            blame_removed: removed,
        })
    }

    /// Changes the blame weight function's scales.
    pub fn configure_negative(&mut self, options: &[(String, f64)]) -> Result<(), String> {
        if self.neg.is_none() {
            return Err("not a negative graph".to_string());
        }
        for (name, value) in options {
            if !value.is_finite() || *value < 0.0 {
                return Err(format!("{name} must be a number >= 0, got {value}"));
            }
            let neg = self.neg.as_mut().expect("checked");
            match name.as_str() {
                "share_scale" => neg.share_scale = *value,
                "blame_scale" => neg.blame_scale = *value,
                "clear_scale" => neg.clear_scale = *value,
                other => return Err(format!("unknown weight option {other:?}")),
            }
        }
        self.dirty_all = true;
        self.version.add(1);
        Ok(())
    }

    /// The blame weight function's scales.
    pub fn negative_weight_config(&self) -> Vec<(&'static str, f64)> {
        match &self.neg {
            Some(neg) => vec![
                ("share_scale", neg.share_scale),
                ("blame_scale", neg.blame_scale),
                ("clear_scale", neg.clear_scale),
            ],
            None => Vec::new(),
        }
    }

    /// Every reason, most blamed first (ties by name, as Python sorts them),
    /// with how many live edges carry it.
    pub fn reason_table(&self) -> Vec<ReasonRow> {
        let Some(neg) = &self.neg else { return Vec::new() };
        let mut edges = vec![0usize; neg.reason_names.len()];
        for (e, reasons) in neg.reasons.iter().enumerate() {
            if !self.edge_alive[e] {
                continue;
            }
            for r in reasons {
                if r.id < edges.len() {
                    edges[r.id] += 1;
                }
            }
        }
        let total: f64 = neg.reason_blame.iter().sum();
        let total = if total == 0.0 { 1.0 } else { total };
        let mut rows: Vec<ReasonRow> = neg
            .reason_names
            .iter()
            .enumerate()
            .map(|(id, name)| ReasonRow {
                id,
                reason: name.clone(),
                blame: neg.reason_blame[id],
                fails: neg.reason_fails[id],
                fails_resets: neg.reason_fails_resets.get(&id).copied().unwrap_or(0),
                edges: edges[id],
                share: neg.reason_blame[id] / total,
            })
            .collect();
        rows.sort_by(|a, b| {
            b.blame
                .partial_cmp(&a.blame)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        rows
    }

    /// One edge's reasons, most blamed first.
    pub fn edge_reasons(&self, e: usize) -> Vec<ReasonRow> {
        let Some(neg) = &self.neg else { return Vec::new() };
        if e >= neg.reasons.len() {
            return Vec::new();
        }
        let total: f64 = neg.reasons[e].iter().map(|r| r.blame).sum();
        let mut rows: Vec<ReasonRow> = neg.reasons[e]
            .iter()
            .map(|r| ReasonRow {
                id: r.id,
                reason: neg.reason_label(r.id).to_string(),
                blame: r.blame,
                fails: 0,
                fails_resets: 0,
                edges: 0,
                share: if total > 0.0 { r.blame / total } else { 0.0 },
            })
            .collect();
        rows.sort_by(|a, b| {
            b.blame
                .partial_cmp(&a.blame)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        rows
    }

    /// How many failures ran through an edge, as an odometer reading.
    pub fn edge_failures(&self, e: usize) -> Counter {
        match &self.neg {
            Some(neg) if e < neg.fails.len() => Counter {
                value: neg.fails[e],
                resets: neg.fails_resets.get(&e).copied().unwrap_or(0),
            },
            _ => Counter::default(),
        }
    }
}

/// Adds `amount` to one reason of an edge, dropping the weakest entry when the
/// list is full (the graph totals keep it).
fn add_reason_blame(list: &mut Vec<ReasonBlame>, id: usize, amount: f64) {
    if let Some(found) = list.iter_mut().find(|r| r.id == id) {
        found.blame += amount;
        return;
    }
    list.push(ReasonBlame { id, blame: amount });
    if list.len() > MAX_EDGE_REASONS {
        // the weakest of the *others* goes: a reason just added carries the
        // least blame of all by construction and would evict itself
        let mut weakest = 0;
        for i in 1..list.len() - 1 {
            if list[i].blame < list[weakest].blame {
                weakest = i;
            }
        }
        list.remove(weakest);
    }
}

/// What a [`Graph::forget`] call removed.
#[derive(Clone, Debug)]
pub struct ForgetResult {
    pub reason: String,
    pub edges: usize,
    pub blame_removed: f64,
}

/// One reason, in a table.
#[derive(Clone, Debug)]
pub struct ReasonRow {
    pub id: usize,
    pub reason: String,
    pub blame: f64,
    pub fails: i64,
    pub fails_resets: i64,
    /// How many live edges carry this reason (the reason table only).
    pub edges: usize,
    pub share: f64,
}

impl ReasonRow {
    /// `{"reason", "blame", "fails", "fails_resets", "edges", "share"}` - a
    /// row of the reason table as Python's `reason_table` writes it.
    pub fn to_json(&self) -> crate::json::Json {
        use crate::json::Json;
        Json::obj([
            ("reason", Json::str(self.reason.clone())),
            ("blame", Json::Num(self.blame)),
            ("fails", Json::Int(self.fails)),
            ("fails_resets", Json::Int(self.fails_resets)),
            ("edges", Json::Int(self.edges as i64)),
            ("share", Json::Num(self.share)),
        ])
    }
}

/// One failure the journal remembers.
#[derive(Clone, Debug, Default)]
pub struct LogEntry {
    pub at: String,
    pub text: String,
    pub reason: String,
    pub severity: f64,
    pub source: String,
    pub note: String,
    pub edges: usize,
}

/// Clips a failing text to what the journal keeps: its first 160 characters
/// as written (Python's `text[:160]`, Go's `truncateRunes`) - an evolve fake
/// ends where the sampler stopped, trailing space and all.
pub(crate) fn log_text(text: &str) -> String {
    text.chars().take(MAX_LOG_TEXT_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphOptions;

    #[test]
    fn a_reason_is_one_trimmed_lower_case_line() {
        assert_eq!(clean_reason("  Gibberish  "), "gibberish");
        assert_eq!(clean_reason("two\n  words"), "two words");
        assert_eq!(clean_reason(""), UNSPECIFIED_REASON);
        assert_eq!(clean_reason("   "), UNSPECIFIED_REASON);
        assert_eq!(clean_reason(&"x".repeat(100)).chars().count(), 60);
    }

    #[test]
    fn an_edge_keeps_at_most_eight_reasons_and_drops_the_weakest() {
        let mut list: Vec<ReasonBlame> = Vec::new();
        for id in 0..MAX_EDGE_REASONS {
            add_reason_blame(&mut list, id, (id + 2) as f64);
        }
        assert_eq!(list.len(), MAX_EDGE_REASONS);
        // a ninth evicts the weakest of the others - reason 0, with the least blame
        add_reason_blame(&mut list, 99, 1.0);
        assert_eq!(list.len(), MAX_EDGE_REASONS);
        assert!(list.iter().any(|r| r.id == 99), "the new reason was dropped instead");
        assert!(!list.iter().any(|r| r.id == 0), "the weakest reason survived");
        // and blaming a reason already there adds to it rather than appending
        let before = list.len();
        add_reason_blame(&mut list, 99, 5.0);
        assert_eq!(list.len(), before);
        assert_eq!(list.iter().find(|r| r.id == 99).map(|r| r.blame), Some(6.0));
    }

    #[test]
    fn blame_and_clearing_cancel() {
        let mut g = Graph::new_negative(0, &NegativeOptions::default()).unwrap();
        let grams = g.enc.encode("the cat sat on the mat");
        let transitions = g.observe(&grams, false).unwrap();
        let edges: Vec<usize> = transitions.iter().map(|t| t.e).collect();

        assert_eq!(g.record_failure(&edges, 2.0, "gibberish"), edges.len());
        let blamed = g.evidence(edges[0]);
        assert!(blamed > 0.0, "a blamed edge carries evidence");

        // one cleared traversal cancels one ordinary failure
        g.record_clear(&edges, 2.0);
        assert_eq!(g.evidence(edges[0]), 0.0, "blame and clearing did not cancel");

        // and it never goes below zero, however much is cleared
        g.record_clear(&edges, 100.0);
        assert_eq!(g.evidence(edges[0]), 0.0);
    }

    #[test]
    fn the_reason_table_is_most_blamed_first() {
        let mut g = Graph::new_negative(0, &NegativeOptions::default()).unwrap();
        let grams = g.enc.encode("the cat sat on the mat");
        let edges: Vec<usize> = g.observe(&grams, false).unwrap().iter().map(|t| t.e).collect();
        g.record_failure(&edges, 1.0, "tense");
        g.record_failure(&edges, 5.0, "gibberish");
        let rows = g.reason_table();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].reason, "gibberish", "not most blamed first: {rows:?}");
        assert!(rows[0].share > rows[1].share);
        assert!((rows.iter().map(|r| r.share).sum::<f64>() - 1.0).abs() < 1e-12);
        // one edge's reasons are the same two, scaled to that edge
        let per_edge = g.edge_reasons(edges[0]);
        assert_eq!(per_edge.len(), 2);
        assert_eq!(per_edge[0].reason, "gibberish");
    }

    #[test]
    fn forgetting_a_reason_removes_only_its_blame() {
        let mut g = Graph::new_negative(0, &NegativeOptions::default()).unwrap();
        let grams = g.enc.encode("the cat sat on the mat");
        let edges: Vec<usize> = g.observe(&grams, false).unwrap().iter().map(|t| t.e).collect();
        g.record_failure(&edges, 3.0, "gibberish");
        g.record_failure(&edges, 1.0, "tense");
        let before = g.evidence(edges[0]);

        let result = g.forget(Some("gibberish"), 0.0).unwrap();
        assert_eq!(result.reason, "gibberish");
        assert!(result.edges > 0);
        let after = g.evidence(edges[0]);
        assert!(after < before, "forgetting removed nothing");
        assert!(after > 0.0, "forgetting one reason removed the other's blame too");
        assert_eq!(
            g.reason_table()
                .iter()
                .find(|r| r.reason == "gibberish")
                .map(|r| r.blame),
            Some(0.0)
        );

        // a reason nothing was blamed for is an answer, not an error
        let none = g.forget(Some("never happened"), 0.0).unwrap();
        assert_eq!(none.edges, 0);
        assert_eq!(none.blame_removed, 0.0);
        // and a factor outside 0..1 is refused
        assert!(g.forget(None, 2.0).is_err());
    }

    #[test]
    fn a_blamed_edge_must_survive_compression() {
        let mut g = Graph::new_negative(0, &NegativeOptions::default()).unwrap();
        let grams = g.enc.encode("the cat sat on the mat");
        let edges: Vec<usize> = g.observe(&grams, false).unwrap().iter().map(|t| t.e).collect();
        assert!(!g.blocks_merge(edges[0]), "nothing was blamed yet");
        g.record_failure(&[edges[0]], 1.0, "gibberish");
        assert!(
            g.blocks_merge(edges[0]),
            "a blamed edge would be merged away and forgotten"
        );
        // clearing counts too: it is evidence the tutor looked
        g.record_clear(&[edges[1]], 1.0);
        assert!(g.blocks_merge(edges[1]));
    }

    /// A negative network taught three failures, for the model-level tests.
    fn taught() -> Model {
        let mut m = Model::new_negative(0, &NegativeOptions::default()).expect("a negative model");
        for (text, reason) in [
            ("the cat sat on the sky", "nonsense"),
            ("the cat sat on the moon", "nonsense"),
            ("water boils at 50 degrees", "wrong fact"),
        ] {
            m.blame(
                &[text.to_string()],
                &BlameOptions {
                    reason: reason.to_string(),
                    severity: 2.0,
                    source: "test".to_string(),
                    ..Default::default()
                },
            )
            .expect("a blamed text");
        }
        m
    }

    #[test]
    fn a_network_that_keeps_learning_in_memory_keeps_the_blame_weights() {
        // an evolve run's or a tutor's network: blamed and cleared again and
        // again without a reload, so only the touched rows are brought up to
        // date - and they must come out as a full recompute writes them
        let mut m = taught();
        m.g.prepare();
        for (text, clear) in [("the cat sat on the sky", false), ("the cat sat on the mat", true)] {
            if clear {
                m.clear_with(&[text.to_string()], 1.0, 1, &mut |_| true)
                    .expect("a clearing pass");
            } else {
                m.blame(&[text.to_string()], &BlameOptions::default())
                    .expect("a blamed text");
            }
            m.g.prepare();
            let lazy = m.g.edge_w.clone();
            m.g.recompute_weights();
            assert_eq!(lazy, m.g.edge_w, "after {text:?}");
        }
        // and the journal keeps a text as it was written
        m.blame(&["the dog sat on the  sky ".to_string()], &BlameOptions::default())
            .expect("a blamed text");
        assert_eq!(m.recent(1)[0].text, "the dog sat on the  sky ");
    }

    #[test]
    fn a_negative_model_is_built_only_out_of_failures() {
        let mut m = taught();
        assert!(m.is_negative());
        assert_eq!(m.kind(), "negative");
        assert!(m.g.num_nodes() > 3, "nothing was registered");
        // the journal remembers what it was told, newest first
        let recent = m.recent(10);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].reason, "wrong fact");
        assert_eq!(recent[0].source, "test");
        assert_eq!(recent[0].severity, 2.0);
        // and the reasons are ranked by blame
        let reasons = m.reasons();
        assert_eq!(reasons[0].reason, "nonsense", "two failures outweigh one: {reasons:?}");
        // a text too short to hold one gram is skipped and counted, not refused
        let records = m.blame(&["hi".to_string()], &BlameOptions::default()).expect("a pass");
        assert_eq!(records[0].skipped_short, 1);
        assert_eq!(records[0].transitions, 0, "a skipped text blamed something");
    }

    #[test]
    fn judging_a_known_failure_rejects_it_and_says_why() {
        let mut m = taught();
        let verdict = m
            .judge("the cat sat on the sky", &JudgeOptions::default())
            .expect("a verdict");
        assert_eq!(verdict.verdict, "reject", "{}", verdict.why);
        assert!(verdict.blamed > 0);
        assert!(verdict.risk >= verdict.threshold);
        assert!(verdict.coverage >= verdict.min_coverage);
        assert_eq!(verdict.reasons[0].reason, "nonsense");
        assert!(!verdict.spans.is_empty(), "a rejected text names no fragment");
        assert!(
            verdict.why.contains("known failures") && verdict.why.contains("rejected"),
            "{}",
            verdict.why
        );

        // text nothing has failed on passes, and says so in one sentence
        let clean = m
            .judge("a completely different sentence entirely", &JudgeOptions::default())
            .unwrap();
        assert_eq!(clean.verdict, "pass");
        assert_eq!(clean.blamed, 0);
        assert_eq!(clean.why, "nothing here has failed before");

        // and the counters moved
        let neg = m.neg.as_ref().unwrap();
        assert_eq!(neg.judgements.value, 2);
        assert_eq!(neg.rejected.value, 1);
    }

    #[test]
    fn clearing_a_text_takes_back_the_verdict() {
        let mut m = taught();
        let before = m.judge("the cat sat on the sky", &JudgeOptions::default()).unwrap();
        assert_eq!(before.verdict, "reject");

        // the tutor looked at it again and passed it: enough clearing cancels
        // the blame, and the verdict has to follow the evidence
        for _ in 0..4 {
            m.clear_text(&["the cat sat on the sky".to_string()], 2.0)
                .expect("cleared");
        }
        let after = m.judge("the cat sat on the sky", &JudgeOptions::default()).unwrap();
        assert!(after.blame < before.blame, "clearing removed no blame");
        assert_eq!(after.verdict, "pass", "{}", after.why);
    }

    #[test]
    fn a_stricter_threshold_changes_the_verdict_and_not_the_evidence() {
        let mut m = taught();
        let lenient = m
            .judge(
                "the cat sat on the sky",
                &JudgeOptions {
                    threshold: Some(1000.0),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(lenient.verdict, "suspect", "nothing should clear that bar");
        assert!(lenient.blamed > 0, "the evidence is the same either way");
        assert!(lenient.why.contains("below the threshold, kept"), "{}", lenient.why);

        // and the settings stick when they are set on the model
        m.set_judgement(Some(1000.0), None).unwrap();
        assert_eq!(
            m.judge("the cat sat on the sky", &JudgeOptions::default())
                .unwrap()
                .verdict,
            "suspect"
        );
        assert!(m.set_judgement(Some(-1.0), None).is_err());
        assert!(m.set_judgement(None, Some(2.0)).is_err());
    }

    #[test]
    fn a_count_model_has_none_of_this() {
        let mut m = Model::new(0, GraphOptions::default()).unwrap();
        assert!(!m.is_negative());
        assert!(m.judge("anything", &JudgeOptions::default()).is_none());
        assert!(m.crossings("anything").is_empty());
        assert!(m.reasons().is_empty());
        assert!(m
            .blame(&["the cat sat on the mat".to_string()], &BlameOptions::default())
            .is_err());
        assert!(m.clear_text(&["the cat sat on the mat".to_string()], 1.0).is_err());
    }

    #[test]
    fn the_file_round_trips() {
        let dir = std::env::temp_dir().join(format!("radixnet-neg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a test directory");
        let path = dir.join("model.negative.json");
        let path = path.to_str().expect("a path");

        let mut m = taught();
        m.clear_text(&["water boils at 100 degrees".to_string()], 1.0).unwrap();
        let before = m.judge("the cat sat on the sky", &JudgeOptions::default()).unwrap();
        let reasons_before = m.reasons();
        m.save(path).expect("the model saves");

        let doc = crate::file::read_document(path).expect("the file reads");
        assert_eq!(doc.at("format").as_str(), Some(NEGATIVE_FORMAT));
        assert_eq!(doc.at("kind").as_str(), Some("negative"));
        assert_eq!(doc.at("graph").at("weights").at("function").as_str(), Some("blame"));
        assert_eq!(doc.at("graph").at("weights").at("kind").as_str(), Some("negative"));
        // the blame rides on the edges, where the count model keeps rewards
        let edges = doc.at("graph").at("edges");
        assert!(!edges.at("blame").to_f64s().is_empty());
        assert!(edges.get("reward").is_none(), "a negative file carries no rewards");
        assert!(!doc.at("log").as_array().is_empty(), "the journal was not written");
        assert_eq!(doc.at("filter").at("threshold").as_f64(), Some(DEFAULT_THRESHOLD));

        let mut again = Model::load(path).expect("the model loads");
        assert!(again.is_negative());
        assert_eq!(again.kind(), "negative");
        let after = again.judge("the cat sat on the sky", &JudgeOptions::default()).unwrap();
        assert_eq!(after.verdict, before.verdict);
        assert_eq!(after.blame, before.blame, "the blame did not survive the round trip");
        assert_eq!(after.why, before.why);
        assert_eq!(
            again.reasons().iter().map(|r| r.reason.clone()).collect::<Vec<_>>(),
            reasons_before.iter().map(|r| r.reason.clone()).collect::<Vec<_>>()
        );
        assert_eq!(again.recent(10).len(), m.recent(10).len());
        let neg = again.neg.as_ref().unwrap();
        assert_eq!(
            neg.sources.iter().find(|(n, _)| n == "test").map(|(_, c)| c.value),
            Some(3)
        );

        // and a count reader refuses it by the format it already checks
        let mut count = Model::new(0, GraphOptions::default()).unwrap();
        let count_path = dir.join("model.count.json");
        count.save(count_path.to_str().unwrap()).unwrap();
        let count_doc = crate::file::read_document(count_path.to_str().unwrap()).unwrap();
        assert_eq!(count_doc.at("format").as_str(), Some(crate::MODEL_FORMAT));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_verdict_quotes_the_way_python_does() {
        assert_eq!(python_repr("plain"), "'plain'");
        assert_eq!(python_repr("it's"), "\"it's\"");
        assert_eq!(python_repr("both ' and \""), "'both \\' and \"'");
        assert_eq!(python_repr("a\nb"), "'a\\nb'");
    }

    #[test]
    fn the_weight_is_the_share_of_the_failure_mass() {
        let g = Graph::new_negative(0, &NegativeOptions::default()).unwrap();
        // one edge carrying all of it outweighs one carrying none
        let all = g.negative_edge_weight(4.0, 4.0, 2);
        let none = g.negative_edge_weight(0.0, 4.0, 2);
        assert!(all > none, "the blamed edge is not the likelier failure");
        // and with nothing blamed anywhere the children are level
        let a = g.negative_edge_weight(0.0, 0.0, 3);
        let b = g.negative_edge_weight(0.0, 0.0, 3);
        assert_eq!(a, b);
    }
}

// -- the model layer ------------------------------------------------------------------------------

use crate::json::Json;
use crate::model::Model;

/// A text that carries, on average, a whole failure's worth of net evidence per
/// transition is rejected.
pub const DEFAULT_THRESHOLD: f64 = 1.0;

/// The share of a text's transitions that must be known failures before any
/// rule may reject it.
pub const DEFAULT_MIN_COVERAGE: f64 = 0.5;

/// The model-level state of a negative network: the journal and how strictly it
/// judges.  `None` on a count model.
#[derive(Clone, Debug)]
pub struct Negative {
    pub log: Vec<LogEntry>,
    pub threshold: f64,
    pub min_coverage: f64,
    /// How many tutors each source has reported, and how often each wrapped.
    pub sources: Vec<(String, Counter)>,
    pub failures_total: Counter,
    pub blame_total: f64,
    pub cleared_total: Counter,
    pub judgements: Counter,
    pub rejected: Counter,
}

impl Default for Negative {
    fn default() -> Negative {
        Negative {
            log: Vec::new(),
            threshold: DEFAULT_THRESHOLD,
            min_coverage: DEFAULT_MIN_COVERAGE,
            sources: Vec::new(),
            failures_total: Counter::default(),
            blame_total: 0.0,
            cleared_total: Counter::default(),
            judgements: Counter::default(),
            rejected: Counter::default(),
        }
    }
}

/// One transition of a judged text, and what the failure structure has against it.
#[derive(Clone, Debug)]
pub struct Crossing {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub fragment: String,
    pub parent: usize,
    /// The edge this transition crossed, or `None` where the structure does not
    /// know the step at all.
    pub edge: Option<usize>,
    pub blame: f64,
    pub fails: i64,
    pub clear: f64,
    pub evidence: f64,
    pub reason: String,
    pub reasons: Vec<ReasonRow>,
}

/// One blamed fragment of a judged text.
#[derive(Clone, Debug)]
pub struct BlamedSpan {
    pub start: usize,
    pub end: usize,
    pub fragment: String,
    pub blame: f64,
    pub fails: i64,
    pub clear: f64,
    pub reason: String,
}

/// Why a text looks like a failure.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub text: String,
    pub chars: usize,
    pub transitions: usize,
    pub known: usize,
    pub blamed: usize,
    pub coverage: f64,
    pub blame: f64,
    pub risk: f64,
    pub peak: f64,
    pub per_char: f64,
    pub threshold: f64,
    pub min_coverage: f64,
    pub verdict: String,
    pub reasons: Vec<ReasonRow>,
    pub spans: Vec<BlamedSpan>,
    pub why: String,
}

/// How strictly one judgement reads the evidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct JudgeOptions {
    pub threshold: Option<f64>,
    pub min_coverage: Option<f64>,
    /// How many of the worst fragments to report; 0 means five.
    pub spans: usize,
}

/// One lesson: why a text failed, how badly, and who says so.
#[derive(Clone, Debug, Default)]
pub struct BlameOptions {
    pub reason: String,
    /// 0 means one ordinary failure.
    pub severity: f64,
    pub source: String,
    pub note: String,
    /// 0 means one pass.
    pub epochs: usize,
    pub no_compress: bool,
}

/// What one taught correction blamed and cleared.
#[derive(Clone, Debug, Default)]
pub struct Correction {
    /// The first eight changes, for the record.
    pub changes: Vec<crate::diff::Edit>,
    /// How many changes there were in all.
    pub edits: usize,
    pub blamed: usize,
    pub cleared: usize,
    pub reason: String,
    pub severity: f64,
    pub wrong_chars: usize,
    pub right_chars: usize,
}

impl Correction {
    /// `{"changes", "edits", "blamed", "cleared", "reason", "severity",
    /// "phase", "wrong_chars", "right_chars"}`, as Python's `correct` returns it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("changes", Json::Arr(self.changes.iter().map(|e| e.to_json()).collect())),
            ("edits", Json::Int(self.edits as i64)),
            ("blamed", Json::Int(self.blamed as i64)),
            ("cleared", Json::Int(self.cleared as i64)),
            ("reason", Json::str(self.reason.clone())),
            ("severity", Json::Num(self.severity)),
            ("phase", Json::str("correction")),
            ("wrong_chars", Json::Int(self.wrong_chars as i64)),
            ("right_chars", Json::Int(self.right_chars as i64)),
        ])
    }
}

impl BlameOptions {
    fn severity(&self) -> f64 {
        if self.severity == 0.0 {
            1.0
        } else {
            self.severity.abs()
        }
    }
    fn epochs(&self) -> usize {
        self.epochs.max(1)
    }
}

impl Model {
    /// An empty negative network: the same machinery as the count model, but
    /// every node and edge it ever grows is there because something went wrong.
    pub fn new_negative(seed: i64, o: &NegativeOptions) -> Result<Model, String> {
        let g = Graph::new_negative(seed, o)?;
        let mut m = Model::from_graph(g);
        m.neg = Some(Box::new(Negative::default()));
        Ok(m)
    }

    /// Whether this model is the negative network.
    pub fn is_negative(&self) -> bool {
        self.neg.is_some() && self.g.is_negative()
    }

    /// Guards the operations only a negative network has.
    fn require_negative(&self, what: &str) -> Result<(), String> {
        if self.is_negative() {
            return Ok(());
        }
        Err(format!(
            "{what} needs the negative network (this is the {} model)",
            self.kind()
        ))
    }

    /// Counts the tutors a failure came from.
    fn add_source(&mut self, source: &str) {
        if source.is_empty() {
            return;
        }
        let neg = self.neg.as_mut().expect("a negative model");
        match neg.sources.iter_mut().find(|(name, _)| name == source) {
            Some((_, counter)) => counter.add(1),
            None => {
                let mut counter = Counter::default();
                counter.add(1);
                neg.sources.push((source.to_string(), counter));
            }
        }
    }

    /// Appends one failure to the journal; the oldest entries drop out.
    fn note(&mut self, text: &str, reason: &str, severity: f64, source: &str, comment: &str, edges: usize) {
        let entry = LogEntry {
            at: crate::clock::utc_now(),
            text: log_text(text),
            reason: reason.to_string(),
            severity,
            source: source.to_string(),
            note: comment
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .chars()
                .take(300)
                .collect(),
            edges,
        };
        let neg = self.neg.as_mut().expect("a negative model");
        neg.log.push(entry);
        if neg.log.len() > MAX_LOG_ENTRIES {
            let drop = neg.log.len() - MAX_LOG_ENTRIES;
            neg.log.drain(..drop);
        }
    }

    /// The newest journal entries, newest first.
    pub fn recent(&self, limit: usize) -> Vec<LogEntry> {
        let Some(neg) = &self.neg else { return Vec::new() };
        neg.log.iter().rev().take(limit).cloned().collect()
    }

    /// Everything the tutor has blamed, heaviest first.
    pub fn reasons(&self) -> Vec<ReasonRow> {
        if !self.is_negative() {
            return Vec::new();
        }
        self.g.reason_table()
    }

    /// Teaches the network one failure: the text is registered structurally and
    /// every transition it walks is blamed.
    ///
    /// This *is* the training pass of a negative network - there is nothing
    /// else to train it on, and training it on correct data would be a category
    /// error.  It is the only operation that adds structure to the network.
    pub fn blame(&mut self, texts: &[String], o: &BlameOptions) -> Result<Vec<crate::model::EpochRecord>, String> {
        self.blame_with(texts, o, &mut |_| true)
    }

    /// [`Model::blame`], calling `on_epoch` after every epoch; returning
    /// `false` from it stops the run (a job's stop button).
    pub fn blame_with(
        &mut self,
        texts: &[String],
        o: &BlameOptions,
        on_epoch: &mut dyn FnMut(&crate::model::EpochRecord) -> bool,
    ) -> Result<Vec<crate::model::EpochRecord>, String> {
        self.require_negative("blame")?;
        let severity = o.severity();
        let reason = clean_reason(&o.reason);
        let records = self.blame_pass(texts, o, true, severity, &reason, on_epoch)?;
        crate::log_info!(
            "negative",
            "blamed {} text(s) for {reason:?} at severity {severity}",
            texts.len()
        );
        Ok(records)
    }

    /// The tutor passed these texts: credits the edges they share with known
    /// failures, creating nothing.  A text the failure structure cannot walk
    /// is counted as `unmatched` and changes nothing.
    pub fn clear(
        &mut self,
        texts: &[String],
        weight: f64,
        epochs: usize,
    ) -> Result<Vec<crate::model::EpochRecord>, String> {
        self.clear_with(texts, weight, epochs, &mut |_| true)
    }

    /// [`Model::clear`] with a callback after every epoch, as
    /// [`Model::blame_with`].
    pub fn clear_with(
        &mut self,
        texts: &[String],
        weight: f64,
        epochs: usize,
        on_epoch: &mut dyn FnMut(&crate::model::EpochRecord) -> bool,
    ) -> Result<Vec<crate::model::EpochRecord>, String> {
        self.require_negative("clear")?;
        let o = BlameOptions {
            epochs,
            ..Default::default()
        };
        let records = self.blame_pass(texts, &o, false, weight.abs(), "", on_epoch)?;
        crate::log_info!("negative", "cleared {} text(s)", texts.len());
        Ok(records)
    }

    /// Clears `texts` and says how many edges it credited - the one-number
    /// form the CLI's `negative clear` reports.
    pub fn clear_text(&mut self, texts: &[String], weight: f64) -> Result<usize, String> {
        let records = self.clear(texts, weight, 1)?;
        Ok(records
            .last()
            .and_then(|r| r.extra_value("edges_touched"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize)
    }

    /// One routine for both halves, as Python's `_pass`: blame (which
    /// registers and creates structure) and clearing (which only credits what
    /// is already there), `o.epochs()` passes each.
    fn blame_pass(
        &mut self,
        texts: &[String],
        o: &BlameOptions,
        blame: bool,
        amount: f64,
        reason: &str,
        on_epoch: &mut dyn FnMut(&crate::model::EpochRecord) -> bool,
    ) -> Result<Vec<crate::model::EpochRecord>, String> {
        let enc = self.g.enc;
        let kept: Vec<&String> = texts.iter().filter(|t| enc.len(t) >= enc.n).collect();
        let skipped_short = texts.len() - kept.len();
        let grams: Vec<Vec<String>> = kept.iter().map(|t| enc.encode(t)).collect();
        if blame {
            self.meta.trained_texts.add(kept.len() as i64);
            self.meta
                .trained_chars
                .add(kept.iter().map(|t| enc.len(t) as i64).sum());
            for _ in &kept {
                self.add_source(&o.source);
            }
            for text in &kept {
                self.note(text, reason, amount, &o.source, &o.note, 0);
            }
            // settle the structure once, so every epoch walks the same
            // transitions - and compress it *before* anything is blamed,
            // because a blamed edge blocks its own merge and would otherwise
            // freeze the structure at whatever the first pass built
            self.register(&grams)?;
        }
        let mut pending_merges = if blame && !o.no_compress { self.g.compress() } else { 0 };
        let mut records = Vec::new();
        for _ in 0..o.epochs() {
            let started = std::time::Instant::now();
            let per_text: Vec<Vec<crate::graph::Transition>> = if blame {
                self.register(&grams)?
            } else {
                kept.iter().map(|t| self.shared_transitions(t)).collect()
            };
            let mut touched = 0;
            for transitions in &per_text {
                if transitions.is_empty() {
                    continue;
                }
                let edges: Vec<usize> = transitions.iter().map(|t| t.e).collect();
                touched += if blame {
                    self.g.record_failure(&edges, amount, reason)
                } else {
                    self.g.record_clear(&edges, amount)
                };
            }
            let flat: Vec<crate::graph::Transition> = per_text.iter().flatten().copied().collect();
            let matched = per_text.iter().filter(|t| !t.is_empty()).count();
            {
                let neg = self.neg.as_mut().expect("a negative model");
                if blame {
                    neg.failures_total.add(matched as i64);
                    neg.blame_total += amount * touched as f64;
                } else {
                    neg.cleared_total.add(matched as i64);
                }
            }
            self.g.prepare();
            let loss = self.mean_cost(&flat);
            let merges = if blame && !o.no_compress { self.g.compress() } else { 0 } + pending_merges;
            pending_merges = 0;
            // the epoch is over: wrap whatever reached the limit
            self.g.carry_counters(false);
            self.meta.epochs_total.add(1);
            self.g.prepare();
            let record = crate::model::EpochRecord {
                epoch: self.meta.epochs_total.value,
                loss,
                perplexity: loss.min(700.0).exp(),
                nodes: self.g.num_nodes(),
                edges: self.g.num_edges(),
                trigrams: self.g.num_trigrams(),
                compression_ratio: self.g.compression_ratio(),
                merges,
                transitions: flat.len() as i64,
                seconds: started.elapsed().as_secs_f64(),
                skipped_short,
                traversed: false,
                reward: 0.0,
                phase: Some(if blame { "negative" } else { "clear" }.to_string()),
                extra: vec![
                    ("texts".to_string(), Json::Int(kept.len() as i64)),
                    ("matched".to_string(), Json::Int(matched as i64)),
                    ("unmatched".to_string(), Json::Int((kept.len() - matched) as i64)),
                    ("edges_touched".to_string(), Json::Int(touched as i64)),
                    ("reason".to_string(), if blame { Json::str(reason) } else { Json::Null }),
                    (
                        "severity".to_string(),
                        if blame { Json::Num(amount) } else { Json::Null },
                    ),
                    (
                        "source".to_string(),
                        if o.source.is_empty() {
                            Json::Null
                        } else {
                            Json::str(o.source.clone())
                        },
                    ),
                ],
            };
            self.history.push(record.clone());
            self.epoch_done(&record)?;
            let go_on = on_epoch(&record);
            records.push(record);
            if !go_on {
                break;
            }
        }
        Ok(records)
    }

    /// The mean `-log P` of transitions under the current weights.
    fn mean_cost(&self, transitions: &[crate::graph::Transition]) -> f64 {
        if transitions.is_empty() {
            return 0.0;
        }
        // a running sum, in order, as Python's `_mean_cost` adds them
        let mut total = 0.0;
        for t in transitions {
            total += self.g.edge_cost(t.e);
        }
        total / transitions.len() as f64
    }

    /// The transitions a text shares with the failure structure (nothing is
    /// created).  Cleared text is ordinary, correct text: it will not walk a
    /// graph of failures end to end, so every crossing it *does* share counts
    /// and the rest is skipped.
    fn shared_transitions(&mut self, text: &str) -> Vec<crate::graph::Transition> {
        self.crossings(text)
            .iter()
            .filter_map(|c| c.edge.map(|e| crate::graph::Transition { p: c.parent, e }))
            .collect()
    }

    /// Registers encoded failures structurally and returns their transitions.
    ///
    /// Two passes: a text registered later can split a node an earlier one
    /// pointed at, so the transitions are re-derived once the structure has
    /// settled (the second pass never splits again).
    fn register(&mut self, grams: &[Vec<String>]) -> Result<Vec<Vec<crate::graph::Transition>>, String> {
        for gram in grams {
            if !gram.is_empty() {
                self.g.observe(gram, false)?;
            }
        }
        let mut out = Vec::with_capacity(grams.len());
        for gram in grams {
            if gram.is_empty() {
                out.push(Vec::new());
                continue;
            }
            out.push(self.g.observe(gram, false)?);
        }
        Ok(out)
    }

    /// Learns one correction: blames only the units the tutor changed, and
    /// clears what it kept.
    ///
    /// `wrong` is the sentence the network wrote, `right` the one the teacher
    /// wrote instead.  Only the steps of `wrong` that wrote a unit the teacher
    /// struck out or replaced are blamed; the correction clears blame wherever
    /// the failure structure already knows it (nothing is created); an edge
    /// the correction walks too is never blamed.  With the attention band on
    /// ([`crate::attention`]) a step is blamed `severity * charge` for the
    /// changed units its grams see, the gram with a change at its centre the
    /// most.
    pub fn blame_correction(
        &mut self,
        wrong: &str,
        right: &str,
        o: &BlameOptions,
        clear: f64,
    ) -> Result<Correction, String> {
        self.require_negative("teaching a correction")?;
        let reason = clean_reason(&o.reason);
        let amount = o.severity();
        let enc = self.g.enc;
        let changes = crate::diff::summary(wrong, right, 0, &enc);
        let (wrong_spans, right_spans) = crate::diff::changed_spans(wrong, right, &enc);
        let mut out = Correction {
            changes: changes.iter().take(8).cloned().collect(),
            edits: changes.len(),
            blamed: 0,
            cleared: 0,
            reason: reason.clone(),
            severity: amount,
            wrong_chars: wrong_spans.iter().map(|(lo, hi)| hi - lo).sum(),
            right_chars: right_spans.iter().map(|(lo, hi)| hi - lo).sum(),
        };
        let grams = enc.encode(wrong);
        if grams.is_empty() {
            return Ok(out);
        }
        // the failure joins the structure; the correction never does
        self.register(std::slice::from_ref(&grams))?;
        let mut cleared: Vec<usize> = Vec::new();
        if !enc.encode(right).is_empty() {
            for t in self.shared_transitions(right) {
                if !cleared.contains(&t.e) {
                    cleared.push(t.e);
                }
            }
        }
        let steps: Vec<(usize, f64)> = self
            .charged_steps(&grams, enc.len(wrong), &wrong_spans)
            .into_iter()
            .map(|s| (s.edge, s.charge))
            .filter(|(e, _)| !cleared.contains(e))
            .collect();
        let blamed: Vec<usize> = steps.iter().map(|&(e, _)| e).collect();
        if !blamed.is_empty() && amount != 0.0 {
            let laid = if self.g.attention.is_on() {
                // each step blamed by its share of what its grams saw
                let shares: Vec<f64> = steps.iter().map(|&(_, charge)| charge).collect();
                out.blamed = self.g.record_failure_shared(&blamed, amount, &shares, &reason);
                // every edge of a traced path is alive: what record_failure_shared laid
                let mut laid = 0.0;
                for share in &shares {
                    laid += amount * share;
                }
                laid
            } else {
                out.blamed = self.g.record_failure(&blamed, amount, &reason);
                amount * out.blamed as f64
            };
            {
                let neg = self.neg.as_mut().expect("a negative model");
                neg.failures_total.add(1);
                neg.blame_total += laid;
            }
            self.meta.trained_texts.add(1);
            self.meta.trained_chars.add(wrong.chars().count() as i64);
            self.add_source(&o.source);
            self.note(wrong, &reason, amount, &o.source, &o.note, 0);
        }
        if !cleared.is_empty() && clear != 0.0 {
            out.cleared = self.g.record_clear(&cleared, clear);
            let neg = self.neg.as_mut().expect("a negative model");
            neg.cleared_total.add(1);
        }
        self.g.prepare();
        Ok(out)
    }

    /// Walks a text through the failure structure, in order.
    pub fn crossings(&mut self, text: &str) -> Vec<Crossing> {
        if !self.is_negative() {
            return Vec::new();
        }
        let enc = self.g.enc;
        let grams = enc.encode(text);
        if grams.is_empty() {
            return Vec::new();
        }
        self.g.prepare();
        let length = enc.len(text);
        let mut node = crate::graph::START;
        let mut offset = 0usize;
        let mut lost = false;
        let mut out: Vec<Crossing> = Vec::with_capacity(grams.len() + 1);

        for (i, gram) in grams.iter().enumerate() {
            let found = self.g.lookup(gram);
            let Some(loc) = found else {
                out.push(self.crossing(text, i, length, node, None));
                lost = true;
                continue;
            };
            if !lost && loc.node == node && loc.off == offset + enc.stride {
                // a deterministic step inside a compressed node: no edge, nothing to blame
                offset = loc.off;
                continue;
            }
            let edge = if loc.off == 0 {
                self.edge_from(node, offset, loc.node, lost)
            } else {
                None
            };
            out.push(self.crossing(text, i, length, node, edge));
            node = loc.node;
            offset = loc.off;
            lost = false;
        }
        let last = self.edge_from(node, offset, crate::graph::END, lost);
        out.push(self.crossing(text, grams.len(), length, node, last));
        out
    }

    /// The edge from `node` to `to`, or `None` where the walk cannot take it.
    fn edge_from(&self, node: usize, offset: usize, to: usize, lost: bool) -> Option<usize> {
        if lost || (node != crate::graph::START && offset + self.g.enc.n != self.g.label_len(node)) {
            return None;
        }
        self.g.edge(node, to)
    }

    /// One crossing, with whatever the structure has against it.
    fn crossing(&self, text: &str, i: usize, length: usize, node: usize, edge: Option<usize>) -> Crossing {
        let enc = self.g.enc;
        let start = i.saturating_sub(1);
        let end = (i + enc.n).min(length);
        let mut entry = Crossing {
            index: i,
            start,
            end,
            fragment: enc.slice(text, start, end),
            parent: node,
            edge,
            blame: 0.0,
            fails: 0,
            clear: 0.0,
            evidence: 0.0,
            reason: String::new(),
            reasons: Vec::new(),
        };
        if let (Some(e), Some(neg)) = (edge, &self.g.neg) {
            entry.blame = neg.blame.get(e).copied().unwrap_or(0.0);
            entry.fails = neg.fails.get(e).copied().unwrap_or(0);
            entry.clear = neg.clear.get(e).copied().unwrap_or(0.0);
            entry.evidence = self.g.evidence(e);
            entry.reasons = self.g.edge_reasons(e);
            if let Some(first) = entry.reasons.first() {
                entry.reason = first.reason.clone();
            }
        }
        entry
    }

    /// The filter: how much of a text is built out of known failure, which
    /// reasons that blame carries and which fragments carry it.
    ///
    /// * **Blame** is the summed net evidence of the transitions the text shares
    ///   with known failures, and **risk** that sum over all of the text's
    ///   transitions, so repeating a failure blamed once scores about 1 and
    ///   sharing a third of one's transitions with it scores about 0.33.
    ///   Compression moves both numbers together, so the ratio does not depend
    ///   on it.
    /// * **Coverage** is the share of transitions that are known failures, and
    ///   **peak** the worst single one.
    pub fn judge(&mut self, text: &str, o: &JudgeOptions) -> Option<Verdict> {
        if !self.is_negative() {
            return None;
        }
        let neg = self.neg.as_ref().expect("a negative model");
        let limit = o.threshold.unwrap_or(neg.threshold);
        let floor = o.min_coverage.unwrap_or(neg.min_coverage);
        let spans = if o.spans == 0 { 5 } else { o.spans };

        let crossings = self.crossings(text);
        let chars = self.g.enc.len(text);
        let known = crossings.iter().filter(|c| c.edge.is_some()).count();
        let blamed: Vec<&Crossing> = crossings.iter().filter(|c| c.evidence > 0.0).collect();
        let blame: f64 = blamed.iter().map(|c| c.evidence).sum();

        let mut out = Verdict {
            text: text.to_string(),
            chars,
            transitions: crossings.len(),
            known,
            blamed: blamed.len(),
            threshold: limit,
            min_coverage: floor,
            blame,
            ..Default::default()
        };
        if !crossings.is_empty() {
            out.coverage = blamed.len() as f64 / crossings.len() as f64;
            out.risk = blame / crossings.len() as f64;
        }
        if chars > 0 {
            out.per_char = blame / chars as f64;
        }

        // the blame split by reason: each crossing's evidence shared out the way
        // its own reasons divide its blame
        let mut by_reason: Vec<(String, f64)> = Vec::new();
        for c in &blamed {
            let share = if c.blame > 0.0 { c.evidence / c.blame } else { 0.0 };
            if c.reasons.is_empty() {
                add_to(&mut by_reason, UNSPECIFIED_REASON, c.evidence);
                continue;
            }
            for row in &c.reasons {
                add_to(&mut by_reason, &row.reason, row.blame * share);
            }
        }
        let total: f64 = by_reason.iter().map(|(_, v)| *v).sum();
        let total = if total == 0.0 { 1.0 } else { total };
        out.reasons = by_reason
            .into_iter()
            .map(|(reason, blame)| ReasonRow {
                id: 0,
                reason,
                blame,
                edges: 0,
                fails: 0,
                fails_resets: 0,
                share: blame / total,
            })
            .collect();
        out.reasons.sort_by(|a, b| {
            b.blame
                .partial_cmp(&a.blame)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.reason.cmp(&b.reason))
        });

        let mut ranked: Vec<&Crossing> = blamed.clone();
        ranked.sort_by(|a, b| b.evidence.partial_cmp(&a.evidence).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(worst) = ranked.first() {
            out.peak = worst.evidence;
        }
        out.spans = ranked
            .iter()
            .take(spans)
            .map(|c| BlamedSpan {
                start: c.start,
                end: c.end,
                fragment: c.fragment.clone(),
                blame: c.evidence,
                fails: c.fails,
                clear: c.clear,
                reason: c.reason.clone(),
            })
            .collect();

        out.verdict = if out.coverage >= floor && !blamed.is_empty() && out.risk >= limit {
            "reject".to_string()
        } else if !blamed.is_empty() {
            "suspect".to_string()
        } else {
            "pass".to_string()
        };
        out.why = why_verdict(
            &out.verdict,
            blamed.len(),
            crossings.len(),
            out.risk,
            &out.reasons,
            &out.spans,
        );

        let rejected = out.verdict == "reject";
        let neg = self.neg.as_mut().expect("a negative model");
        neg.judgements.add(1);
        if rejected {
            neg.rejected.add(1);
        }
        crate::log_debug!(
            "negative",
            "judged {} transition(s): {} (risk {:.2}, coverage {:.2})",
            out.transitions,
            out.verdict,
            out.risk,
            out.coverage
        );
        Some(out)
    }

    /// Changes how strictly the network judges.
    pub fn set_judgement(&mut self, threshold: Option<f64>, min_coverage: Option<f64>) -> Result<(), String> {
        self.require_negative("settings")?;
        if let Some(value) = threshold {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("threshold must be a number >= 0, got {value}"));
            }
            self.neg.as_mut().expect("checked").threshold = value;
        }
        if let Some(value) = min_coverage {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("min_coverage must be between 0 and 1, got {value}"));
            }
            self.neg.as_mut().expect("checked").min_coverage = value;
        }
        Ok(())
    }
}

/// Adds `amount` to a named entry, keeping first-seen order.
fn add_to(list: &mut Vec<(String, f64)>, name: &str, amount: f64) {
    match list.iter_mut().find(|(n, _)| n == name) {
        Some((_, value)) => *value += amount,
        None => list.push((name.to_string(), amount)),
    }
}

/// The one sentence a verdict rests on.
///
/// Character for character the sentence the Python implementation writes: the
/// parity tests compare them.
fn why_verdict(
    verdict: &str,
    blamed: usize,
    total: usize,
    risk: f64,
    reasons: &[ReasonRow],
    spans: &[BlamedSpan],
) -> String {
    if blamed == 0 {
        return "nothing here has failed before".to_string();
    }
    let mut out = format!("{blamed} of {total} transitions are known failures (risk {risk:.2})");
    if let Some(first) = reasons.first() {
        out += &format!(", mostly {}", python_repr(&first.reason));
    }
    if let Some(worst) = spans.first() {
        out += &format!(", worst at {}", python_repr(&worst.fragment));
    }
    match verdict {
        "reject" => out += "; rejected",
        "suspect" => out += "; below the threshold, kept",
        _ => {}
    }
    out
}

/// Quotes a string the way Python's `repr()` does, so the sentence a verdict
/// carries is character for character the one Python writes.
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            c if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A negative network's statistics, with the keys Python's `NegativeNet.stats`
/// reports - what `/api/negative`, the CLI and the frontend's Negative tab read.
pub fn stats_json(model: &Model) -> Json {
    let g = &model.g;
    let enc = g.enc;
    let Some(neg) = model.neg.as_ref() else {
        return Json::Obj(Vec::new());
    };
    let Some(data) = g.neg.as_ref() else {
        return Json::Obj(Vec::new());
    };
    let mut pairs: Vec<(String, Json)> = vec![
        ("kind".to_string(), Json::str(model.kind())),
        ("nodes".to_string(), Json::Int(g.num_nodes() as i64)),
        ("edges".to_string(), Json::Int(g.num_edges() as i64)),
        ("trigrams".to_string(), Json::Int(g.num_trigrams() as i64)),
        ("grams".to_string(), Json::Int(g.num_trigrams() as i64)),
        ("encoding".to_string(), Json::str(enc.to_string())),
        ("unit".to_string(), Json::str(enc.unit.name())),
        ("units".to_string(), Json::str(enc.units_name())),
        ("ngram".to_string(), Json::Int(enc.n as i64)),
        ("stride".to_string(), Json::Int(enc.stride as i64)),
        // where a correction lands: the band's blur, null while it is off
        (
            "attention_blur".to_string(),
            g.attention.blur.map(Json::Num).unwrap_or(Json::Null),
        ),
        ("compression_ratio".to_string(), Json::Num(g.compression_ratio())),
        ("inverted".to_string(), Json::Bool(g.inverted)),
        ("backend".to_string(), Json::str("rust")),
        ("device".to_string(), Json::str(model.device_label())),
    ];
    for (key, counter) in [
        ("epochs_total", model.meta.epochs_total),
        ("trained_chars", model.meta.trained_chars),
        ("trained_texts", model.meta.trained_texts),
        ("twonrl_runs", model.meta.twonrl_runs),
        ("failures_total", neg.failures_total),
        ("cleared_total", neg.cleared_total),
        ("judgements", neg.judgements),
        ("rejected", neg.rejected),
    ] {
        pairs.push((key.to_string(), Json::Int(counter.value)));
        pairs.push((format!("{key}_resets"), Json::Int(counter.resets)));
    }
    let resets: Vec<(String, Json)> = neg
        .sources
        .iter()
        .filter(|(_, c)| c.resets != 0)
        .map(|(name, c)| (name.clone(), Json::Int(c.resets)))
        .collect();
    pairs.extend([
        ("history_len".to_string(), Json::Int(model.history.len() as i64)),
        (
            "last_loss".to_string(),
            model.history.last().map(|r| Json::Num(r.loss)).unwrap_or(Json::Null),
        ),
        ("blame_total".to_string(), Json::Num(neg.blame_total)),
        (
            "sources".to_string(),
            Json::Obj(
                neg.sources
                    .iter()
                    .map(|(name, c)| (name.clone(), Json::Int(c.value)))
                    .collect(),
            ),
        ),
        ("sources_resets".to_string(), Json::Obj(resets)),
        ("edge_fails_total".to_string(), Json::Int(data.total_fails.value)),
        (
            "edge_fails_total_resets".to_string(),
            Json::Int(data.total_fails.resets),
        ),
        ("edge_blame_total".to_string(), Json::Num(data.total_blame)),
        ("edge_clear_total".to_string(), Json::Num(data.total_clear)),
        ("reason_count".to_string(), Json::Int(data.reason_names.len() as i64)),
        (
            "top_reasons".to_string(),
            Json::Arr(g.reason_table().iter().take(5).map(|r| r.to_json()).collect()),
        ),
        ("threshold".to_string(), Json::Num(neg.threshold)),
        ("min_coverage".to_string(), Json::Num(neg.min_coverage)),
        ("share_scale".to_string(), Json::Num(data.share_scale)),
        ("blame_scale".to_string(), Json::Num(data.blame_scale)),
        ("clear_scale".to_string(), Json::Num(data.clear_scale)),
        ("log_entries".to_string(), Json::Int(neg.log.len() as i64)),
    ]);
    Json::Obj(pairs)
}

impl LogEntry {
    /// A journal entry as Python keeps it: `{"at", "text", "reason",
    /// "severity", "source", "note"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("at", Json::str(self.at.clone())),
            ("text", Json::str(self.text.clone())),
            ("reason", Json::str(self.reason.clone())),
            ("severity", Json::Num(self.severity)),
            ("source", Json::str(self.source.clone())),
            ("note", Json::str(self.note.clone())),
        ])
    }
}
