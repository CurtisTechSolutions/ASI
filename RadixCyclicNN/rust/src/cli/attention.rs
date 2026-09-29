//! `radixnet attention`: the command over the attention band
//! ([`crate::attention`]).

use crate::cli::Ctx;
use crate::json::Json;

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
