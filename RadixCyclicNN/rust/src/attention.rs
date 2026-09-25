//! The attention band: where inside an n-gram a correction's blame and credit
//! land (`radixnet/attention.py`, `go/radixnet/attention.go`,
//! `../SPEC-AttentionBand.md`).
//!
//! A reader's eye fixes on one point of a line and sees it sharply, and the
//! letters either side of it blur with the distance.  A gram is this model's
//! fixation, and the band is how sharply each of its `n` positions is seen: 1
//! at the centre, falling off in a straight line to `1 - blur` at the first
//! and the last unit.
//!
//! * **Off** (the default, and every model before it existed): each unit a
//!   correction changes is charged in full to the step that wrote it - the
//!   step whose gram ends on it.
//! * **On**: each changed unit hands out exactly one charge, shared among the
//!   grams that see it in proportion to how sharply each sees it, so the gram
//!   with the change at its centre takes the most.  A step is charged what its
//!   grams collected, never more than one full charge, and the judged-path
//!   verdict goes to the step that saw the change most sharply (the focus).
//!
//! A judgement of a whole text - a thumbs up, a thumbs down, a training pass -
//! marks every unit alike, and no band changes it.  [`spread`] is the whole
//! rule over a text's grams; its order of operations is part of the
//! definition, so the three implementations add the same floats in the same
//! order and write the same bits.

use crate::cli::Ctx;
use crate::diff::{Edit, Span};
use crate::encoding::Encoding;
use crate::graph::{END, START};
use crate::json::Json;
use crate::model::Model;

/// What switching the band on means when no blur is given: the ends of a gram
/// seen half as sharply as its centre.
pub const DEFAULT_BLUR: f64 = 0.5;

/// A model's band: `blur` in `[0, 1]` while it is on, `None` while it is off.
/// Unlike the encoding it changes nothing the graph holds, so it can be
/// switched at any time; it travels with the model file while it is on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttentionBand {
    pub blur: Option<f64>,
}

/// The one rule for a blur: a finite number in `[0, 1]`.
pub fn check_blur(blur: f64) -> Result<(), String> {
    if !blur.is_finite() || !(0.0..=1.0).contains(&blur) {
        return Err(format!(
            "blur must lie in [0, 1], got {}",
            crate::json::Json::Num(blur).render(0)
        ));
    }
    Ok(())
}

/// How sharply each of a gram's `n` positions is seen: 1 at the centre,
/// `1 - blur` at both ends, linear between.  A gram of one is all centre and a
/// gram of two all ends, so both are flat.  Each weight is `1 - blur * (d / c)`
/// with `c = (n - 1) / 2` and `d` the distance from it: one correctly rounded
/// step at a time, as Python and Go compute it.
pub fn band_weights(n: usize, blur: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }
    let centre = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|j| {
            let d = (j as f64 - centre).abs();
            1.0 - blur * (d / centre)
        })
        .collect()
}

impl AttentionBand {
    /// The band on at `blur`, checked.
    pub fn on(blur: f64) -> Result<AttentionBand, String> {
        check_blur(blur)?;
        Ok(AttentionBand { blur: Some(blur) })
    }

    /// Is the band on?  Off, each changed unit is charged to the step that wrote it.
    pub fn is_on(&self) -> bool {
        self.blur.is_some()
    }

    /// The band over one gram of `n` units, or `None` while it is off.
    pub fn weights(&self, n: usize) -> Option<Vec<f64>> {
        self.blur.map(|blur| band_weights(n, blur))
    }
}

/// The units a correction marks, ascending, within `[0, length]`: every unit
/// of every span, and for an empty span - an insertion point - the unit it
/// stands in front of.  `length` itself is the position after the last unit,
/// which only the step into END answers for.
pub fn judged_units(spans: &[Span], length: usize) -> Vec<usize> {
    let mut marked: Vec<usize> = Vec::new();
    for &(lo, hi) in spans {
        let top = if hi > lo { hi } else { lo + 1 }.min(length + 1);
        marked.extend(lo..top);
    }
    marked.sort_unstable();
    marked.dedup();
    marked
}

/// What [`spread`] hands each gram of a text.
#[derive(Clone, Debug, PartialEq)]
pub struct Spread {
    /// Per gram, the charge it collected before any cap.
    pub shares: Vec<f64>,
    /// Per gram: does it see some marked unit most sharply of all the grams
    /// that see that unit?
    pub focus: Vec<bool>,
    /// Is the position after the last unit marked?  The step into END answers
    /// for it, in full.
    pub end: bool,
}

/// Shares every marked unit out over the grams that see it, in proportion to
/// the band: gram `g` covers `[g * stride, g * stride + n)` and sees unit `u`
/// at position `u - g * stride`.  Each marked unit hands out exactly one
/// charge: `weight / total` to each gram that sees it, or an even split when
/// every one of them sees it at a weight of 0.  Units ascending, the grams
/// that see one ascending, every sum left to right.
pub fn spread(n: usize, stride: usize, grams: usize, length: usize, spans: &[Span], weights: &[f64]) -> Spread {
    let mut out = Spread {
        shares: vec![0.0; grams],
        focus: vec![false; grams],
        end: false,
    };
    for unit in judged_units(spans, length) {
        if unit >= length {
            out.end = true;
            continue;
        }
        let lo = if unit < n { 0 } else { (unit - n) / stride + 1 };
        let Some(last) = grams.checked_sub(1) else {
            continue;
        };
        let hi = (unit / stride).min(last);
        if lo > hi {
            continue; // no gram covers it: the tail a grouping encoding drops
        }
        let mut total = 0.0;
        let mut best = 0.0;
        for g in lo..=hi {
            let w = weights[unit - g * stride];
            total += w;
            if w > best {
                best = w;
            }
        }
        let viewers = (hi - lo + 1) as f64;
        for g in lo..=hi {
            let w = weights[unit - g * stride];
            out.shares[g] += if total > 0.0 { w / total } else { 1.0 / viewers };
            if w == best {
                out.focus[g] = true;
            }
        }
    }
    out
}

/// The rule with the band off, gram by gram: does gram `g` write a marked unit
/// (the first gram all of its units, every later one the `stride` units past
/// the overlap), and is the end marked.
pub fn writer_marks(n: usize, stride: usize, grams: usize, length: usize, spans: &[Span]) -> (Vec<bool>, bool) {
    let marked = judged_units(spans, length);
    let overlap = n - stride;
    let writes = (0..grams)
        .map(|g| {
            let lo = if g == 0 { 0 } else { g * stride + overlap };
            let hi = g * stride + n;
            marked.iter().any(|&u| lo <= u && u < hi)
        })
        .collect();
    (writes, marked.contains(&length))
}

/// One step of a traced text a correction charges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChargedStep {
    /// Who called the step: START begins every walk.
    pub prev: usize,
    pub edge: usize,
    /// At most 1: a step is one decision, however much of it was wrong.
    pub charge: f64,
    /// Does it see a changed unit most sharply?  The verdict goes there.
    pub focus: bool,
}

/// One side of a correction, gram by gram: what the writer rule charges and
/// what the band would (`radixnet/attention.py` `preview`).
pub fn preview_side(enc: &Encoding, weights: &[f64], text: &str, spans: &[Span]) -> Json {
    let grams = enc.encode(text);
    let length = enc.len(text);
    let (writes, _) = writer_marks(enc.n, enc.stride, grams.len(), length, spans);
    let shared = spread(enc.n, enc.stride, grams.len(), length, spans, weights);
    Json::obj([
        ("text", Json::str(text)),
        ("units", Json::Int(length as i64)),
        ("grams", Json::strs(grams.iter().map(|g| g.as_str()))),
        (
            "spans",
            Json::Arr(
                spans
                    .iter()
                    .map(|&(lo, hi)| Json::ints([lo as i64, hi as i64]))
                    .collect(),
            ),
        ),
        ("writer", Json::Arr(writes.into_iter().map(Json::Bool).collect())),
        ("charges", Json::nums(shared.shares.iter().map(|&s| s.min(1.0)))),
        ("focus", Json::Arr(shared.focus.into_iter().map(Json::Bool).collect())),
        ("end", Json::Bool(shared.end)),
    ])
}

impl Model {
    /// Does this kind learn from a diff?  Only such a kind has anything for
    /// the band to spread (Python's `takes_corrections`): the count model and
    /// the negative network.
    pub fn takes_corrections(&self) -> bool {
        !self.g.is_radix() && !self.g.is_resonant()
    }

    /// The steps of a traced text a correction charges, in path order.  Off,
    /// they are [`Model::steps_over`]'s, each charged 1 and each the focus;
    /// on, every changed unit is shared out over the grams that see it and a
    /// step is charged what the grams of its node collected, capped at 1.
    /// `length` is the text's length in units.
    pub fn charged_steps(&self, grams: &[String], length: usize, spans: &[Span]) -> Vec<ChargedStep> {
        let g = &self.g;
        let Some(weights) = g.attention.weights(g.enc.n) else {
            return self
                .steps_over(grams, length, spans)
                .into_iter()
                .map(|(prev, edge)| ChargedStep {
                    prev,
                    edge,
                    charge: 1.0,
                    focus: true,
                })
                .collect();
        };
        let Some(path) = g.node_path(grams) else {
            return Vec::new();
        };
        if path.len() < 2 {
            return Vec::new();
        }
        let (n, stride) = (g.enc.n, g.enc.stride);
        let shared = spread(n, stride, grams.len(), length, spans, &weights);
        let mut out = Vec::new();
        let mut gram = 0usize; // the text's first gram inside the node being entered
        for index in 1..path.len() {
            let node = path[index];
            let prev = if index >= 2 { path[index - 2] } else { START };
            let edge = g.edge(path[index - 1], node);
            if node == END {
                if let (Some(edge), true) = (edge, shared.end) {
                    out.push(ChargedStep {
                        prev,
                        edge,
                        charge: 1.0,
                        focus: true,
                    });
                }
                break;
            }
            // the grams a node of this length holds
            let held = (g.label_len(node).saturating_sub(n)) / stride + 1;
            let mut charge = 0.0;
            let mut focus = false;
            for w in gram..(gram + held).min(grams.len()) {
                charge += shared.shares[w];
                focus = focus || shared.focus[w];
            }
            gram += held;
            if let Some(edge) = edge {
                if charge > 0.0 {
                    out.push(ChargedStep {
                        prev,
                        edge,
                        charge: charge.min(1.0),
                        focus,
                    });
                }
            }
        }
        out
    }

    /// The band as the API, the CLI and the frontend show it.
    pub fn attention_config(&self) -> Json {
        let band = self.g.attention;
        let enc = self.g.enc;
        Json::obj([
            ("on", Json::Bool(band.is_on())),
            ("blur", band.blur.map(Json::Num).unwrap_or(Json::Null)),
            ("weights", band.weights(enc.n).map(Json::nums).unwrap_or(Json::Null)),
            ("ngram", Json::Int(enc.n as i64)),
            ("stride", Json::Int(enc.stride as i64)),
            ("unit", Json::str(enc.unit.name())),
            ("units", Json::str(enc.units_name())),
            ("applies", Json::Bool(self.takes_corrections())),
            ("default_blur", Json::Num(DEFAULT_BLUR)),
        ])
    }

    /// Switches the band on (at `blur`, else the blur it had, else
    /// [`DEFAULT_BLUR`]) or off: a blur alone switches it on, `on = false`
    /// switches it off whatever the blur says, and neither changes nothing.
    /// A kind that is never corrected refuses.
    pub fn configure_attention(&mut self, on: Option<bool>, blur: Option<f64>) -> Result<Json, String> {
        if !self.takes_corrections() {
            return Err(format!(
                "the {} model is never corrected, so an attention band would have nothing to spread; \
                 it belongs to the kinds that learn from a diff (count, negative)",
                self.kind()
            ));
        }
        let current = self.g.attention;
        match (on, blur) {
            (Some(false), _) => self.g.attention = AttentionBand::default(),
            (Some(true), _) | (None, Some(_)) => {
                let value = blur.or(current.blur).unwrap_or(DEFAULT_BLUR);
                self.g.attention = AttentionBand::on(value)?;
            }
            (None, None) => {}
        }
        Ok(self.attention_config())
    }

    /// Where one correction would land, gram by gram, under the writer rule and
    /// under a band: `blur` when given, else the model's own, else the default.
    /// Needs only the encoding, and changes nothing.
    pub fn attention_preview(&self, wrong: &str, right: &str, blur: Option<f64>) -> Result<Json, String> {
        let value = blur.or(self.g.attention.blur).unwrap_or(DEFAULT_BLUR);
        check_blur(value)?;
        let enc = self.g.enc;
        let weights = band_weights(enc.n, value);
        let (wrong_spans, right_spans) = crate::diff::changed_spans(wrong, right, &enc);
        let changes: Vec<Json> = crate::diff::summary(wrong, right, 0, &enc)
            .iter()
            .map(Edit::to_json)
            .collect();
        Ok(Json::obj([
            ("attention", self.attention_config()),
            ("blur", Json::Num(value)),
            ("weights", Json::nums(weights.iter().copied())),
            ("changes", Json::Arr(changes)),
            ("wrong", preview_side(&enc, &weights, wrong, &wrong_spans)),
            ("right", preview_side(&enc, &weights, right, &right_spans)),
        ]))
    }
}

/// `radixnet attention`: shows the band, switches it (`--on`, `--blur X`,
/// `--off`; saved unless `--dry-run`), and with `--wrong` and `--right` shows
/// where one correction would land through it.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let blur = args.maybe_float("blur")?;
    if args.on("on") && args.on("off") {
        return Err("argument --off: not allowed with argument --on".to_string());
    }
    if args.on("off") && blur.is_some() {
        return Err("--off and --blur contradict each other: a blur switches the band on".to_string());
    }
    let (wrong, right) = (args.get("wrong"), args.get("right"));
    if wrong.is_some() != right.is_some() {
        return Err(
            "a preview needs both --wrong (what the network wrote) and --right (what it should say)".to_string(),
        );
    }
    let mut model = ctx.open(true)?;
    let mut changed = Json::Obj(Vec::new());
    let mut saved = Json::Null;
    if args.on("on") || args.on("off") || blur.is_some() {
        let on = if args.on("on") {
            Some(true)
        } else if args.on("off") {
            Some(false)
        } else {
            None
        };
        model.configure_attention(on, blur)?;
        let band = model.g.attention;
        changed = Json::obj([
            ("on", Json::Bool(band.is_on())),
            ("blur", band.blur.map(Json::Num).unwrap_or(Json::Null)),
        ]);
        if !args.on("dry-run") {
            saved = Json::str(ctx.save(&mut model)?);
        }
    }
    let preview = match (wrong, right) {
        (Some(wrong), Some(right)) => model.attention_preview(wrong, right, None)?,
        _ => Json::Null,
    };
    ctx.emit(Json::obj([
        ("attention", model.attention_config()),
        ("changed", changed),
        ("saved", saved),
        ("preview", preview),
    ]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphOptions, TrainOptions};

    const TEXTS: [&str; 5] = [
        "the cat sat on the mat",
        "the cat ran to the door",
        "the dog sat on the log",
        "the dog ate the bone",
        "the bat sat on the mat",
    ];

    fn flat() -> Model {
        let texts: Vec<String> = TEXTS.iter().map(|s| s.to_string()).collect();
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 1,
                    auto_compress: false,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    /// The label of the node an edge enters.
    fn child(model: &Model, edge: usize) -> String {
        let g = &model.g;
        for p in 0..g.labels.len() {
            for c in 0..g.labels.len() {
                if g.alive[p] && g.alive[c] && g.edge(p, c) == Some(edge) {
                    return g.labels[c].clone();
                }
            }
        }
        String::new()
    }

    #[test]
    fn the_band_peaks_at_the_centre() {
        assert_eq!(band_weights(3, 0.5), vec![0.5, 1.0, 0.5]);
        assert_eq!(band_weights(5, 0.5), vec![0.5, 0.75, 1.0, 0.75, 0.5]);
        assert_eq!(band_weights(2, 0.5), vec![0.5, 0.5]);
        assert_eq!(band_weights(1, 1.0), vec![1.0]);
        for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(check_blur(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_centre_of_the_gram_takes_the_most() {
        let enc = Encoding::default();
        let (spans, _) = crate::diff::changed_spans("the cat sat", "the bat sat", &enc);
        let grams = enc.encode("the cat sat");
        let out = spread(3, 1, grams.len(), 11, &spans, &band_weights(3, 0.5));
        assert_eq!(out.shares, vec![0.0, 0.0, 0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.focus.iter().position(|&f| f), Some(3));
        let (writes, _) = writer_marks(3, 1, grams.len(), 11, &spans);
        assert_eq!(writes.iter().position(|&w| w), Some(2));
    }

    #[test]
    fn one_unit_one_charge_and_a_whole_text_left_alone() {
        let text = "the quick brown fox jumps over the lazy dog again and again";
        for spec in [
            "char:3:1", "char:5:1", "char:4:4", "char:6:3", "char:2:1", "word:1:1", "word:2:1", "word:3:1",
        ] {
            let enc = crate::parse_encoding(spec).unwrap();
            let grams = enc.encode(text).len();
            let length = enc.len(text);
            let covered = enc.covered(length);
            for blur in [0.0, 0.3, 0.5, 1.0] {
                let weights = band_weights(enc.n, blur);
                for unit in 0..covered {
                    let out = spread(enc.n, enc.stride, grams, length, &[(unit, unit + 1)], &weights);
                    let total: f64 = out.shares.iter().sum();
                    assert!((total - 1.0).abs() < 1e-12, "{spec} {blur} {unit}: {total}");
                }
                let out = spread(enc.n, enc.stride, grams, length, &[(0, length)], &weights);
                assert!(
                    out.shares.iter().all(|&s| s >= 1.0 - 1e-12),
                    "{spec} {blur}: {:?}",
                    out.shares
                );
                assert!(out.focus.iter().all(|&f| f));
            }
        }
        // one gram sees each letter of a group: it takes the whole charge
        assert_eq!(
            spread(4, 4, 3, 12, &[(4, 5)], &band_weights(4, 1.0)).shares,
            vec![0.0, 1.0, 0.0]
        );
        assert!(spread(3, 1, 9, 11, &[(11, 11)], &band_weights(3, 0.5)).end);
    }

    #[test]
    fn a_correction_charges_the_centre_most() {
        let mut model = flat();
        model.configure_attention(None, Some(0.5)).unwrap();
        let before = model.g.edge_reward.clone();
        let out = model
            .correct(
                "the cat sat on the mat",
                "the bat sat on the mat",
                &crate::correct::CorrectOptions {
                    strength: 1.0,
                    weight: 2.0,
                    reward: 1.0,
                    keep: 0.0,
                    count: true,
                },
            )
            .unwrap();
        let mut penalties = Vec::new();
        let mut rewards = Vec::new();
        for (e, &value) in model.g.edge_reward.iter().enumerate() {
            let delta = value - before.get(e).copied().unwrap_or(0.0);
            if delta < 0.0 {
                penalties.push((child(&model, e), delta));
            } else if delta > 0.0 {
                rewards.push((child(&model, e), delta));
            }
        }
        penalties.sort_by(|a, b| a.0.cmp(&b.0));
        rewards.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            penalties,
            vec![
                (" ca".to_string(), -1.0),
                ("cat".to_string(), -0.5),
                ("e c".to_string(), -0.5)
            ]
        );
        assert_eq!(
            rewards,
            vec![
                (" ba".to_string(), 0.5),
                ("bat".to_string(), 0.25),
                ("e b".to_string(), 0.25)
            ]
        );
        assert_eq!(
            (out.penalised, out.penalty, out.marked_incorrect, out.marked_correct),
            (3, 2.0, 1, 1)
        );
    }

    #[test]
    fn switched_off_it_is_the_writer_rule() {
        let mut model = flat();
        let enc = model.g.enc;
        let grams = enc.encode("the cat sat on the mat");
        let spans = [(4usize, 5usize)];
        let off = model.charged_steps(&grams, 22, &spans);
        let writer: Vec<ChargedStep> = model
            .steps_over(&grams, 22, &spans)
            .into_iter()
            .map(|(prev, edge)| ChargedStep {
                prev,
                edge,
                charge: 1.0,
                focus: true,
            })
            .collect();
        assert_eq!(off, writer);
        model.configure_attention(Some(true), None).unwrap();
        assert_eq!(model.g.attention.blur, Some(DEFAULT_BLUR));
        assert_eq!(model.charged_steps(&grams, 22, &spans).len(), 3);
        model.configure_attention(Some(false), Some(0.9)).unwrap();
        assert!(!model.g.attention.is_on());
        assert!(model.configure_attention(None, Some(1.5)).is_err());
    }

    #[test]
    fn a_kind_that_is_never_corrected_refuses() {
        let mut model = crate::kinds::new_model("radix", 1, Encoding::default(), &[]).unwrap();
        assert_eq!(model.attention_config().at("applies").as_bool(), Some(false));
        assert!(model.configure_attention(None, Some(0.5)).is_err());
    }
}
