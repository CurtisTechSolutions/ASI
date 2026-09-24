//! Teach one correction: only what changed moves (`radixnet/countnet.py`
//! `CountRewardNet.correct`, `go/radixnet/correct.go`).
//!
//! The tutor's marking used to be a verdict on a whole sentence: the attempt
//! was punished, the correction rewarded, and every edge of either path moved
//! by the same amount.  But most of a corrected sentence is word for word what
//! the network wrote - "the cat sit on the mat" is one letter from right - and
//! punishing every step of it teaches the network that "the cat" was the
//! mistake.  So the two sentences are aligned ([`crate::diff`], in the
//! model's own units: characters, or words under a word encoding) and every
//! step of either path is charged with the units it adds
//! ([`Model::steps_over`]):
//!
//! * the steps of `wrong` that wrote a unit the teacher struck out or replaced
//!   lose `strength * weight` of reward - and only those;
//! * the steps of `right` that write what the teacher put there instead gain
//!   `strength * reward`, and the rest of the correction earns `keep` times as
//!   much - 0 by default, because a whole path is rewarded when the *output*
//!   was correct, not when it had to be corrected;
//! * the correction is traversed once, as a training pass would, unless
//!   `count` is off: a corrected sentence is correct text whatever changed;
//! * an edge both sentences walk over a changed span (the network wrote the
//!   right units by another route) is rewarded, never penalised;
//! * the judged-path counters follow the rewards: the blamed steps are marked
//!   incorrect in their context, the taught ones correct.
//!
//! The rewards are applied the way Python applies them - the penalty in one
//! call, then one call per distinct reward amount - so the counts it reports,
//! and the totals it keeps, come out the same number for number.
//!
//! Also here: `radixnet correct`, with Python's flags and JSON, and `--blame`
//! to teach the negative network the same diff.

use crate::cli::{negative_stats, Args, Ctx};
use crate::diff::Edit;
use crate::encoding::Encoding;
use crate::json::Json;
use crate::model::Model;
use crate::negative::{python_repr, BlameOptions};
use crate::paths::{PathKey, PathOutcome};
use crate::report::stats;

/// What this module's lines are filed under.
const LOG: &str = "train";

/// How hard the two halves of a correction push.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectOptions {
    /// The magnitude of one unit of feedback (its absolute value is used).
    pub strength: f64,
    /// How bad the attempt was: the penalty is `strength * weight`.
    pub weight: f64,
    /// What the correction is worth: the fix gains `strength * reward`.
    pub reward: f64,
    /// What the unchanged part of the correction still earns, as a fraction
    /// of `reward` (0: the fix alone).
    pub keep: f64,
    /// Traverse the correction once, as a training pass does.
    pub count: bool,
}

impl Default for CorrectOptions {
    /// One unit of feedback either way, nothing for the words that did not
    /// change, and the correction counted.
    fn default() -> CorrectOptions {
        CorrectOptions {
            strength: 1.0,
            weight: 1.0,
            reward: 1.0,
            keep: 0.0,
            count: true,
        }
    }
}

/// What one taught correction moved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Correction {
    /// How many changes the alignment found.
    pub edits: usize,
    /// The first eight of them.
    pub changes: Vec<Edit>,
    /// Steps of the attempt penalised.
    pub penalised: usize,
    /// Steps of the fix rewarded in full.
    pub rewarded: usize,
    /// Steps of the rest of the correction rewarded at `keep`.
    pub kept: usize,
    pub penalty: f64,
    pub reward: f64,
    /// The mean cost of the correction's steps once it was counted; `None`
    /// when there was no correction to walk.
    pub loss: Option<f64>,
    /// Units of the attempt the teacher changed.
    pub wrong_chars: usize,
    /// Units of the correction the teacher wrote.
    pub right_chars: usize,
    pub marked_correct: usize,
    pub marked_incorrect: usize,
}

impl Correction {
    /// The pairs of Python's result, in its order, `changes` given apart
    /// (the CLI reports the whole alignment where the model keeps eight).
    fn pairs(&self, changes: Json) -> Vec<(String, Json)> {
        vec![
            ("edits".to_string(), Json::Int(self.edits as i64)),
            ("changes".to_string(), changes),
            ("penalised".to_string(), Json::Int(self.penalised as i64)),
            ("rewarded".to_string(), Json::Int(self.rewarded as i64)),
            ("kept".to_string(), Json::Int(self.kept as i64)),
            ("penalty".to_string(), Json::Num(self.penalty)),
            ("reward".to_string(), Json::Num(self.reward)),
            ("loss".to_string(), self.loss.map_or(Json::Null, Json::Num)),
            ("wrong_chars".to_string(), Json::Int(self.wrong_chars as i64)),
            ("right_chars".to_string(), Json::Int(self.right_chars as i64)),
            ("marked_correct".to_string(), Json::Int(self.marked_correct as i64)),
            ("marked_incorrect".to_string(), Json::Int(self.marked_incorrect as i64)),
        ]
    }

    /// Python's result: `{"edits", "changes", "penalised", "rewarded", "kept",
    /// "penalty", "reward", "loss", "wrong_chars", "right_chars",
    /// "marked_correct", "marked_incorrect"}`.
    pub fn to_json(&self) -> Json {
        Json::Obj(self.pairs(Json::Arr(self.changes.iter().map(Edit::to_json).collect())))
    }
}

impl Model {
    /// Teaches one correction: moves only the nodes the two texts disagree
    /// on (see the module docs for what moves and by how much).
    ///
    /// `wrong` is what the network wrote, `right` what the teacher wrote
    /// instead.  Both join the structure before either is measured: observing
    /// one can split a node the other's path runs through, and the split moves
    /// the very edge a penalty was meant for.
    ///
    /// Only the count model learns this way; the sine and phase models refuse,
    /// as Python's (which have no `correct`) do, and the tutor teaches them a
    /// corrected failure through 2NRL instead.
    pub fn correct(&mut self, wrong: &str, right: &str, o: &CorrectOptions) -> Result<Correction, String> {
        if self.is_negative() {
            return Err(
                "the negative network learns a correction by blaming it (`radixnet correct --blame`), not from rewards"
                    .to_string(),
            );
        }
        if self.is_radix() || self.is_resonant() {
            // Python's kinds without a `correct`: the sine model learns by its
            // rates and the phase model by its locks, not by moving rewards
            // along a diff (`cmd_correct`'s refusal)
            let kind = self.kind();
            return Err(format!(
                "the {kind} ({}) model cannot learn from a diff; use feedback instead",
                crate::kinds::label(kind)
            ));
        }
        let base = o.strength.abs();
        let enc = self.g.enc;
        let changes = crate::diff::summary(wrong, right, 0, &enc);
        let (wrong_spans, right_spans) = crate::diff::changed_spans(wrong, right, &enc);
        let mut out = Correction {
            edits: changes.len(),
            changes: changes.into_iter().take(8).collect(),
            wrong_chars: wrong_spans.iter().map(|(lo, hi)| hi - lo).sum(),
            right_chars: right_spans.iter().map(|(lo, hi)| hi - lo).sum(),
            ..Default::default()
        };
        let wrong_grams = enc.encode(wrong);
        let right_grams = enc.encode(right);
        if wrong_grams.is_empty() && right_grams.is_empty() {
            return Ok(out);
        }
        for grams in [&wrong_grams, &right_grams] {
            if !grams.is_empty() && self.g.node_path(grams).is_none() {
                self.g.observe(grams, false)?;
            }
        }
        // the edges the attempt is blamed on, first seen first
        let mut penalised: Vec<usize> = Vec::new();
        let mut blamed_steps: Vec<(usize, usize)> = Vec::new();
        if !wrong_grams.is_empty() && !wrong_spans.is_empty() && base * o.weight > 0.0 {
            blamed_steps = self.steps_over(&wrong_grams, enc.len(wrong), &wrong_spans);
            for &(_, edge) in &blamed_steps {
                if !penalised.contains(&edge) {
                    penalised.push(edge);
                }
            }
        }
        // what each edge of the correction is owed, first seen first
        let mut rewards: Vec<(usize, f64)> = Vec::new();
        let mut fixed: Vec<usize> = Vec::new();
        let mut taught_steps: Vec<(usize, usize)> = Vec::new();
        if !right_grams.is_empty() {
            let transitions = self.g.observe(&right_grams, o.count)?;
            if o.count {
                let edges: Vec<usize> = transitions.iter().map(|t| t.e).collect();
                self.g.record_traversals(&edges);
                self.meta.trained_texts.add(1);
                self.meta.trained_chars.add(enc.len(right) as i64);
                // the correction's own traffic, in the contexts that already exist
                self.g.record_path(&transitions, PathOutcome::Unjudged, false);
            }
            if !right_spans.is_empty() {
                taught_steps = self.steps_over(&right_grams, enc.len(right), &right_spans);
                fixed = taught_steps.iter().map(|&(_, edge)| edge).collect();
            }
            for t in &transitions {
                let share = if fixed.contains(&t.e) { 1.0 } else { o.keep };
                let amount = base * o.reward * share;
                if amount > 0.0 {
                    match rewards.iter_mut().find(|(e, _)| *e == t.e) {
                        Some(slot) => slot.1 = amount,
                        None => rewards.push((t.e, amount)),
                    }
                }
            }
            self.g.prepare();
            let total: f64 = transitions.iter().map(|t| self.g.edge_cost(t.e)).sum();
            out.loss = Some(if transitions.is_empty() {
                0.0
            } else {
                total / transitions.len() as f64
            });
        }
        let owed = |edge: usize| rewards.iter().any(|(e, _)| *e == edge);
        // the teacher wrote it too: it is not the mistake
        let blamed: Vec<usize> = penalised.into_iter().filter(|e| !owed(*e)).collect();
        if !blamed.is_empty() {
            let penalty = -base * o.weight;
            out.penalised = self.g.add_reward(&blamed, penalty);
            out.penalty = -penalty * out.penalised as f64;
        }
        // one call per distinct amount: the penalty, the fix, and what the rest keeps
        let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for &(edge, amount) in &rewards {
            match groups.iter_mut().find(|(a, _)| *a == amount) {
                Some(group) => group.1.push(edge),
                None => groups.push((amount, vec![edge])),
            }
        }
        for (amount, edges) in groups {
            let touched = self.g.add_reward(&edges, amount);
            if edges.iter().all(|e| fixed.contains(e)) {
                out.rewarded += touched;
            } else {
                out.kept += touched;
            }
            out.reward += amount * touched as f64;
        }
        // the counters follow the reward: what was blamed is a wrong path here, what was taught a right one
        let still_blamed: Vec<PathKey> = blamed_steps
            .iter()
            .filter(|(_, edge)| !owed(*edge))
            .map(|&(prev, edge)| PathKey { prev, edge })
            .collect();
        out.marked_incorrect = self.g.mark_steps(&still_blamed, false);
        let taught: Vec<PathKey> = taught_steps
            .iter()
            .map(|&(prev, edge)| PathKey { prev, edge })
            .collect();
        out.marked_correct = self.g.mark_steps(&taught, true);
        if out.penalised > 0 || out.rewarded > 0 || out.kept > 0 {
            self.meta.feedback_passes.add(1);
            self.meta.rewards_total += out.reward;
            self.meta.penalties_total += out.penalty;
        }
        self.g.prepare();
        crate::log_debug!(
            LOG,
            "correction: {} penalised, {} taught, {} kept",
            out.penalised,
            out.rewarded,
            out.kept
        );
        Ok(out)
    }
}

/// A flag that must be a finite number no smaller than 0.
fn nonneg(args: &Args, name: &str, fallback: f64) -> Result<f64, String> {
    let Some(text) = args.get(name) else {
        return Ok(fallback);
    };
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("argument --{name}: expected a number, got {}", python_repr(text)))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("argument --{name}: must be a finite number >= 0, got {text}"));
    }
    Ok(value)
}

/// `radixnet correct --wrong TEXT --right TEXT`: teaches one correction, or
/// with `--dry-run` only shows the alignment; `--blame` also teaches the
/// negative network the same diff.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let (Some(wrong), Some(right)) = (args.get("wrong"), args.get("right")) else {
        return Err("the following arguments are required: --wrong, --right".to_string());
    };
    if wrong.trim().is_empty() && right.trim().is_empty() {
        return Err(
            "nothing to correct: give --wrong (what the network wrote) and --right (what it should say)".to_string(),
        );
    }
    let options = CorrectOptions {
        strength: nonneg(args, "strength", 1.0)?,
        weight: nonneg(args, "weight", 1.0)?,
        reward: nonneg(args, "reward", 1.0)?,
        keep: nonneg(args, "keep", 0.0)?,
        count: !args.on("no-count"),
    };
    // the alignment is shown character by character, as Python shows it,
    // before any model is opened
    let changes = Json::Arr(
        crate::diff::summary(wrong, right, 0, &Encoding::default())
            .iter()
            .map(Edit::to_json)
            .collect(),
    );
    if args.on("dry-run") {
        ctx.emit(Json::obj([
            ("wrong", Json::str(wrong)),
            ("right", Json::str(right)),
            ("changes", changes),
            ("dry_run", Json::Bool(true)),
        ]));
        return Ok(());
    }
    let mut model = ctx.open(true)?;
    let moved = model.correct(wrong, right, &options)?;
    let negative = if args.on("blame") {
        let mut negative = ctx.open_negative(false)?;
        let blamed = negative.blame_correction(
            wrong,
            right,
            &BlameOptions {
                reason: args.str("reason", "corrected"),
                severity: options.weight * options.strength,
                source: "cli".to_string(),
                note: args.str("note", ""),
                ..Default::default()
            },
            1.0,
        )?;
        let path = ctx.negative_path();
        negative.save(&path)?;
        let reasons: Vec<Json> = negative
            .reasons()
            .iter()
            .map(|r| {
                Json::obj([
                    ("reason", Json::str(r.reason.clone())),
                    ("blame", Json::Num(r.blame)),
                    ("fails", Json::Int(r.fails)),
                    ("share", Json::Num(r.share)),
                ])
            })
            .collect();
        Json::obj([
            ("blamed", blamed.to_json()),
            ("path", Json::str(path.clone())),
            ("saved", Json::str(path)),
            ("reasons", Json::Arr(reasons)),
            ("stats", negative_stats(&mut negative)),
        ])
    } else {
        Json::Null
    };
    let saved = ctx.save(&mut model)?;
    crate::log_info!(
        LOG,
        "correction taught: {} step(s) penalised, {} taught, {} kept; saved {saved}",
        moved.penalised,
        moved.rewarded,
        moved.kept
    );
    let mut doc = vec![
        ("out".to_string(), Json::str(saved.clone())),
        ("wrong".to_string(), Json::str(wrong)),
        ("right".to_string(), Json::str(right)),
    ];
    doc.extend(moved.pairs(changes));
    doc.push(("saved".to_string(), Json::str(saved)));
    doc.push(("stats".to_string(), stats(&model)));
    doc.push(("negative".to_string(), negative));
    ctx.emit(Json::Obj(doc));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphOptions, TrainOptions};

    fn trained(extra: &[&str]) -> Model {
        let mut texts: Vec<String> = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "a bird flew over the hill",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        texts.extend(extra.iter().map(|s| s.to_string()));
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    #[test]
    fn only_the_difference_moves() {
        let mut model = trained(&["the cat sits on the mat", "the dogs run in the park"]);
        let (wrong, right) = ("the cat sit on the mat", "the cat sits on the mat");
        let before: Vec<f64> = (0..model.g.num_edge_ids()).map(|e| model.g.edge_reward(e)).collect();
        let out = model.correct(wrong, right, &CorrectOptions::default()).unwrap();
        assert_eq!(
            (out.edits, out.penalised, out.rewarded, out.kept),
            (1, 1, 1, 0),
            "{out:?}"
        );
        let moved: Vec<(usize, f64)> = (0..model.g.num_edge_ids())
            .map(|e| (e, model.g.edge_reward(e) - before.get(e).copied().unwrap_or(0.0)))
            .filter(|(_, delta)| *delta != 0.0)
            .collect();
        assert_eq!(
            moved.len(),
            2,
            "keep=0 moves exactly the two steps that differ: {moved:?}"
        );
        let blamed = moved.iter().find(|(_, d)| *d < 0.0).unwrap().0;
        let grams = model.encoding().encode(wrong);
        let want = model.steps_over(&grams, wrong.chars().count(), &[(11, 11)]);
        assert_eq!(want.iter().map(|&(_, e)| e).collect::<Vec<_>>(), [blamed]);
        assert_eq!(
            out.to_json().at("changes").as_array()[0].at("right").as_str(),
            Some("s")
        );
    }

    #[test]
    fn keep_spreads_a_smaller_reward_and_the_correction_is_counted() {
        let mut model = trained(&[]);
        let texts = model.meta.trained_texts.value;
        let out = model
            .correct(
                "the cat sit on the mat",
                "the cat sits on the mat",
                &CorrectOptions {
                    keep: 0.25,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.kept > 0, "{out:?}");
        assert_eq!((out.marked_correct, out.marked_incorrect), (1, 1));
        assert!(out.reward > 0.0 && out.penalty > 0.0 && out.loss.is_some());
        assert_eq!(model.meta.trained_texts.value - texts, 1);
        assert!(model.meta.feedback_passes.value > 0);
        let same = model
            .correct(
                "the cat sits on the mat",
                "the cat sits on the mat",
                &CorrectOptions::default(),
            )
            .unwrap();
        assert_eq!(
            (same.edits, same.penalised, same.rewarded, same.marked_correct),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn an_early_end_is_blamed_on_the_step_into_end() {
        let mut model = trained(&[]);
        let (wrong, right) = ("the cat sat", "the cat sat on the mat");
        let out = model.correct(wrong, right, &CorrectOptions::default()).unwrap();
        assert_eq!(out.penalised, 1, "{out:?}");
        let grams = model.encoding().encode(wrong);
        let path = model.g.node_path(&grams).unwrap();
        let last = model.g.edge(path[path.len() - 2], crate::END).unwrap();
        let blamed = model.steps_over(&grams, 11, &[(11, 11)]);
        assert_eq!(blamed.iter().map(|&(_, e)| e).collect::<Vec<_>>(), [last]);
    }

    #[test]
    fn short_empty_and_repeated_corrections_keep_the_graph_sound() {
        let mut model = trained(&[]);
        for (wrong, right) in [("", ""), ("ab", "ab"), ("", "the cat sits"), ("the cat sits", "")] {
            model.correct(wrong, right, &CorrectOptions::default()).unwrap();
        }
        let empty = model.correct("", "", &CorrectOptions::default()).unwrap();
        assert_eq!(empty.loss, None);
        for i in 0..12 {
            let wrong = format!("the cat sit on the mat number {i}");
            let right = format!("the cat sits on the mat number {i}");
            model.correct(&wrong, &right, &CorrectOptions::default()).unwrap();
            model.g.check_invariants(&[], false).unwrap();
        }
        let negative = Model::new_negative(0, &Default::default()).unwrap();
        let mut negative = negative;
        assert!(negative.correct("a", "b", &CorrectOptions::default()).is_err());
    }
}
