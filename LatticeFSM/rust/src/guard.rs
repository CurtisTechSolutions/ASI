//! The guard: a negative network - only the failures, and why they were failures.
//!
//! The same guard as `latticefsm/guard.py`, number for number: every transition (the last `order - 1` tokens, then
//! the next) of a blamed text is charged the blame under its reason; accepted text clears what it crosses; the
//! evidence against a transition is `max(0, blame - clear)`; and `judge` reports a text's risk (evidence per
//! transition), coverage (the share of its transitions that are known failures), peak, reasons and worst spans,
//! and its verdict - `reject` (coverage at least `min_coverage` and risk at least `threshold`), `suspect` or `pass`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};

use crate::gzip;
use crate::json::{parse, Json};
use crate::ngram::{tokenize, BOS, EOS, UNITS};

const FORMAT: &str = "latticefsm-guard";
pub const MAX_EDGE_REASONS: usize = 8;
pub const MAX_JOURNAL: usize = 200;
pub const UNSPECIFIED: &str = "unspecified";
const JOURNAL_CHARS: usize = 160;

/// A reason tag: whitespace collapsed, ASCII lower-cased, at most 60 characters, never empty.
pub fn clean_reason(reason: &str) -> String {
    let text: String = reason
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let text: String = text.chars().take(60).collect();
    if text.is_empty() {
        UNSPECIFIED.to_string()
    } else {
        text
    }
}

/// A transition's record.
#[derive(Clone, Debug, Default)]
pub struct Edge {
    pub blame: f64,
    pub fails: u64,
    pub clear: f64,
    pub reasons: BTreeMap<String, f64>,
}

impl Edge {
    pub fn evidence(&self) -> f64 {
        (self.blame - self.clear).max(0.0)
    }

    pub fn top_reason(&self) -> String {
        let mut best: Option<(&String, f64)> = None;
        for (r, &b) in &self.reasons {
            if best.is_none_or(|(_, bb)| b > bb) {
                best = Some((r, b));
            }
        }
        best.map(|(r, _)| r.clone()).unwrap_or_else(|| UNSPECIFIED.to_string())
    }
}

pub struct Guard {
    pub unit: String,
    pub order: usize,
    pub lower: bool,
    pub threshold: f64,
    pub min_coverage: f64,
    /// `[*context, next]` -> its record.
    pub edges: BTreeMap<Vec<String>, Edge>,
    pub totals: BTreeMap<String, f64>,
    pub journal: Vec<Json>,
    pub blamed: u64,
    pub cleared: u64,
    pub judgements: u64,
    pub rejected: u64,
}

impl Default for Guard {
    fn default() -> Self {
        Guard::new("word", 3, true, 1.0, 0.5).expect("valid")
    }
}

impl Guard {
    pub fn new(unit: &str, order: usize, lower: bool, threshold: f64, min_coverage: f64) -> Result<Guard, String> {
        if !UNITS.contains(&unit) {
            return Err(format!("unit must be one of {UNITS:?}, got {unit:?}"));
        }
        if !(1..=8).contains(&order) {
            return Err(format!("order is 1 to 8, got {order}"));
        }
        Ok(Guard {
            unit: unit.to_string(),
            order,
            lower,
            threshold,
            min_coverage,
            edges: BTreeMap::new(),
            totals: BTreeMap::new(),
            journal: Vec::new(),
            blamed: 0,
            cleared: 0,
            judgements: 0,
            rejected: 0,
        })
    }

    /// A text's tokens and its transitions, `[*the last order - 1 tokens, next]`, `</s>` last.
    pub fn transitions(&self, text: &str) -> (Vec<String>, Vec<Vec<String>>) {
        let toks = tokenize(text, &self.unit, self.lower);
        let k = self.order - 1;
        let mut padded: Vec<String> = vec![BOS.to_string(); k];
        padded.extend(toks.iter().cloned());
        padded.push(EOS.to_string());
        let trans = (k..padded.len()).map(|i| padded[i - k..=i].to_vec()).collect();
        (toks, trans)
    }

    /// Charge every transition of a failed text `severity` of blame under `reason`.  Returns how many.
    pub fn blame(&mut self, text: &str, reason: &str, severity: f64) -> usize {
        let reason = clean_reason(reason);
        let severity = severity.max(0.0);
        let (_, trans) = self.transitions(text);
        if trans.is_empty() || severity == 0.0 {
            return 0;
        }
        let n = trans.len();
        for key in trans {
            let e = self.edges.entry(key).or_default();
            e.blame += severity;
            e.fails += 1;
            *e.reasons.entry(reason.clone()).or_insert(0.0) += severity;
            if e.reasons.len() > MAX_EDGE_REASONS {
                let mut worst: Option<(String, f64)> = None;
                for (r, &b) in &e.reasons {
                    if worst.as_ref().is_none_or(|(_, wb)| b < *wb) {
                        worst = Some((r.clone(), b));
                    }
                }
                if let Some((r, _)) = worst {
                    e.reasons.remove(&r);
                }
            }
        }
        *self.totals.entry(reason.clone()).or_insert(0.0) += severity;
        self.blamed += 1;
        self.journal.push(
            Json::object()
                .with(
                    "text",
                    text.chars().take(JOURNAL_CHARS).collect::<String>().as_str().into(),
                )
                .with("reason", reason.as_str().into())
                .with("severity", severity.into())
                .with("transitions", n.into()),
        );
        if self.journal.len() > MAX_JOURNAL {
            let drop = self.journal.len() - MAX_JOURNAL;
            self.journal.drain(..drop);
        }
        n
    }

    /// An accepted text: every transition of it the guard knows is cleared by `amount`.  Returns how many.
    pub fn clear(&mut self, text: &str, amount: f64) -> usize {
        let amount = amount.max(0.0);
        let (_, trans) = self.transitions(text);
        let mut n = 0;
        for key in trans {
            if let Some(e) = self.edges.get_mut(&key) {
                e.clear += amount;
                n += 1;
            }
        }
        if n > 0 {
            self.cleared += 1;
        }
        n
    }

    /// Take a reason's blame back out of every transition (dropping those left with none).  Returns how many.
    pub fn forget(&mut self, reason: &str) -> usize {
        let reason = clean_reason(reason);
        let mut n = 0;
        let mut gone = Vec::new();
        for (key, e) in self.edges.iter_mut() {
            if let Some(b) = e.reasons.remove(&reason) {
                e.blame = (e.blame - b).max(0.0);
                n += 1;
                if e.reasons.is_empty() || e.blame <= 0.0 {
                    gone.push(key.clone());
                }
            }
        }
        for key in gone {
            self.edges.remove(&key);
        }
        self.totals.remove(&reason);
        n
    }

    pub fn reset(&mut self) {
        self.edges.clear();
        self.totals.clear();
        self.journal.clear();
        self.blamed = 0;
        self.cleared = 0;
        self.judgements = 0;
        self.rejected = 0;
    }

    /// Why this text looks like a failure, as `Guard.judge` reports it.
    pub fn judge(&mut self, text: &str, threshold: Option<f64>, min_coverage: Option<f64>, spans: usize) -> Json {
        let limit = threshold.unwrap_or(self.threshold);
        let floor = min_coverage.unwrap_or(self.min_coverage);
        let (toks, trans) = self.transitions(text);
        let k = self.order - 1;
        let rows: Vec<(usize, Option<&Edge>, f64)> = trans
            .iter()
            .enumerate()
            .map(|(i, key)| {
                let e = self.edges.get(key);
                (i, e, e.map(Edge::evidence).unwrap_or(0.0))
            })
            .collect();
        let total = rows.len();
        let blamed: Vec<&(usize, Option<&Edge>, f64)> = rows.iter().filter(|r| r.2 > 0.0).collect();
        let known = rows.iter().filter(|r| r.1.is_some()).count();
        let mut evidence = 0.0;
        for r in &blamed {
            evidence += r.2;
        }
        let coverage = if total > 0 {
            blamed.len() as f64 / total as f64
        } else {
            0.0
        };
        let risk = evidence / total.max(1) as f64;
        let mut by_reason: BTreeMap<String, f64> = BTreeMap::new();
        for (_, e, ev) in &blamed {
            let e = e.expect("blamed");
            let share = if e.blame > 0.0 { ev / e.blame } else { 0.0 };
            for (name, &b) in &e.reasons {
                *by_reason.entry(name.clone()).or_insert(0.0) += b * share;
            }
        }
        let mut reasons: Vec<(String, f64)> = by_reason.into_iter().collect();
        reasons.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let mut weight = 0.0;
        for (_, b) in &reasons {
            weight += b;
        }
        if weight == 0.0 {
            weight = 1.0;
        }
        let mut worst: Vec<&&(usize, Option<&Edge>, f64)> = blamed.iter().collect();
        worst.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        worst.truncate(spans);
        let peak = worst.first().map(|r| r.2).unwrap_or(0.0);
        let verdict = if !blamed.is_empty() && coverage >= floor && risk >= limit {
            "reject"
        } else if !blamed.is_empty() {
            "suspect"
        } else {
            "pass"
        };
        let why = match verdict {
            "reject" => format!(
                "rejected: {:.0}% of its transitions are known failures, {risk:.2} blame each on average (the limit \
                 {limit:.2}), mostly '{}'",
                coverage * 100.0,
                reasons.first().map(|r| r.0.as_str()).unwrap_or(UNSPECIFIED)
            ),
            "suspect" => format!(
                "suspect: {} of its {total} transitions are known failures ({:.0}%), {risk:.2} blame each - under the \
                 limit",
                blamed.len(),
                coverage * 100.0
            ),
            _ => "pass: none of it is a known failure".to_string(),
        };
        let out_spans: Vec<Json> = worst
            .iter()
            .map(|(i, e, ev)| {
                let first = i.saturating_sub(k);
                let last = (i + 1).min(toks.len());
                Json::object()
                    .with("start", first.into())
                    .with("end", last.into())
                    .with("tokens", Json::strings(&toks[first.min(last)..last]))
                    .with("evidence", (*ev).into())
                    .with("reason", e.expect("blamed").top_reason().as_str().into())
                    .with("next", trans[*i][k].as_str().into())
            })
            .collect();
        let blamed_n = blamed.len();
        let reasons_json: Vec<Json> = reasons
            .iter()
            .map(|(n, b)| {
                Json::object()
                    .with("reason", n.as_str().into())
                    .with("blame", (*b).into())
                    .with("share", (b / weight).into())
            })
            .collect();
        self.judgements += 1;
        if verdict == "reject" {
            self.rejected += 1;
        }
        Json::object()
            .with("text", text.into())
            .with("tokens", toks.len().into())
            .with("transitions", total.into())
            .with("known", known.into())
            .with("blamed", blamed_n.into())
            .with("evidence", evidence.into())
            .with("risk", risk.into())
            .with("coverage", coverage.into())
            .with("peak", peak.into())
            .with("reasons", Json::Array(reasons_json))
            .with("spans", Json::Array(out_spans))
            .with("verdict", verdict.into())
            .with("why", why.as_str().into())
    }

    pub fn stats(&self) -> Json {
        let mut evidence = 0.0;
        for e in self.edges.values() {
            evidence += e.evidence();
        }
        let mut totals: Vec<(&String, &f64)> = self.totals.iter().collect();
        totals.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(b.0))
        });
        Json::object()
            .with("unit", self.unit.as_str().into())
            .with("order", self.order.into())
            .with("threshold", self.threshold.into())
            .with("min_coverage", self.min_coverage.into())
            .with("transitions", self.edges.len().into())
            .with("evidence", evidence.into())
            .with("blamed", (self.blamed as f64).into())
            .with("cleared", (self.cleared as f64).into())
            .with("judgements", (self.judgements as f64).into())
            .with("rejected", (self.rejected as f64).into())
            .with(
                "reasons",
                Json::Array(
                    totals
                        .iter()
                        .map(|(r, b)| {
                            Json::object()
                                .with("reason", r.as_str().into())
                                .with("blame", (**b).into())
                        })
                        .collect(),
                ),
            )
            .with(
                "journal",
                Json::Array(self.journal.iter().rev().take(20).cloned().collect()),
            )
    }

    pub fn to_json(&self) -> Json {
        let edges: Vec<Json> = self
            .edges
            .iter()
            .map(|(key, e)| {
                Json::Array(vec![
                    Json::strings(key),
                    e.blame.into(),
                    (e.fails as f64).into(),
                    e.clear.into(),
                    Json::Array(
                        e.reasons
                            .iter()
                            .map(|(r, b)| Json::Array(vec![r.as_str().into(), (*b).into()]))
                            .collect(),
                    ),
                ])
            })
            .collect();
        Json::object()
            .with("format", FORMAT.into())
            .with("version", 1.0.into())
            .with("unit", self.unit.as_str().into())
            .with("order", self.order.into())
            .with("lower", self.lower.into())
            .with("threshold", self.threshold.into())
            .with("min_coverage", self.min_coverage.into())
            .with("edges", Json::Array(edges))
            .with(
                "totals",
                Json::Array(
                    self.totals
                        .iter()
                        .map(|(r, b)| Json::Array(vec![r.as_str().into(), (*b).into()]))
                        .collect(),
                ),
            )
            .with("journal", Json::Array(self.journal.clone()))
            .with("blamed", (self.blamed as f64).into())
            .with("cleared", (self.cleared as f64).into())
            .with("judgements", (self.judgements as f64).into())
            .with("rejected", (self.rejected as f64).into())
    }

    pub fn from_json(v: &Json) -> Result<Guard, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let mut g = Guard::new(
            v.str_or("unit", "word"),
            v.num("order", 3.0) as usize,
            v.bool_or("lower", true),
            v.num("threshold", 1.0),
            v.num("min_coverage", 0.5),
        )?;
        for row in v
            .get("edges")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let r = row.as_array().ok_or("an edge row is a list")?;
            if r.len() != 5 {
                return Err("an edge row of the wrong length".to_string());
            }
            let key: Vec<String> = r[0]
                .as_array()
                .ok_or("an edge's key is a list")?
                .iter()
                .map(|t| t.as_str().unwrap_or("").to_string())
                .collect();
            let mut e = Edge {
                blame: r[1].as_f64().unwrap_or(0.0),
                fails: r[2].as_f64().unwrap_or(0.0) as u64,
                clear: r[3].as_f64().unwrap_or(0.0),
                reasons: BTreeMap::new(),
            };
            for pair in r[4].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(p) = pair.as_array() {
                    if p.len() == 2 {
                        e.reasons
                            .insert(p[0].as_str().unwrap_or("").to_string(), p[1].as_f64().unwrap_or(0.0));
                    }
                }
            }
            g.edges.insert(key, e);
        }
        for pair in v
            .get("totals")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if let Some(p) = pair.as_array() {
                if p.len() == 2 {
                    g.totals
                        .insert(p[0].as_str().unwrap_or("").to_string(), p[1].as_f64().unwrap_or(0.0));
                }
            }
        }
        g.journal = v.get("journal").and_then(Json::as_array).cloned().unwrap_or_default();
        g.blamed = v.num("blamed", 0.0) as u64;
        g.cleared = v.num("cleared", 0.0) as u64;
        g.judgements = v.num("judgements", 0.0) as u64;
        g.rejected = v.num("rejected", 0.0) as u64;
        Ok(g)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let mut f = fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

pub fn load_guard(path: &str) -> Result<Guard, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if path.ends_with(".gz") || bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    Guard::from_json(&parse(&text)?)
}
