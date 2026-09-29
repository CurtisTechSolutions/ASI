//! `radixnet correct`: one correction taught from the command line
//! ([`crate::correct`]), with Python's flags and JSON, and `--blame` to teach
//! the negative network the same diff.

use crate::cli::{negative_stats, Args, Ctx};
use crate::correct::CorrectOptions;
use crate::diff::Edit;
use crate::encoding::Encoding;
use crate::json::Json;
use crate::negative::{python_repr, BlameOptions};
use crate::report::stats;

/// What this module's lines are filed under.
const LOG: &str = "train";

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
