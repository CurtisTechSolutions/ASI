//! `radixnet window`: the command over the dynamic window
//! ([`radixnet::window`]).

use crate::cli::{Args, Ctx};
use radixnet::json::Json;

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
