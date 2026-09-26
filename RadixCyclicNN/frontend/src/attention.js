/**
 * The attention band, read for display (../../SPEC-AttentionBand.md, radixnet/attention.py): the band over
 * one gram - the server's own formula, so the Model settings card can draw a blur while it is being chosen,
 * before anything is applied - and how sharply a unit is drawn. Where a correction actually lands is the
 * server's to say (POST /api/model/attention/preview); nothing here decides a charge. Pure functions,
 * checked by ../test/attention.test.mjs.
 */

/** What switching the band on means when no blur is given (radixnet.attention.DEFAULT_BLUR). */
export const DEFAULT_BLUR = 0.5;

/** A blur the band accepts: a finite number in [0, 1]; anything else is `fallback`. */
export function clampBlur(value, fallback = DEFAULT_BLUR) {
  const number = typeof value === "string" && value.trim() === "" ? NaN : Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(1, Math.max(0, number));
}

/**
 * How sharply each of a gram's n positions is seen: 1 at the centre, 1 - blur at both ends, linear
 * between (radixnet.attention.band_weights, the same doubles). A gram of one is all centre and a gram of
 * two all ends, so both are flat.
 */
export function bandWeights(n, blur) {
  const size = Math.max(1, Math.floor(Number(n) || 1));
  if (size === 1) return [1];
  const b = clampBlur(blur);
  const centre = (size - 1) / 2;
  return Array.from({ length: size }, (_, j) => 1 - b * (Math.abs(j - centre) / centre));
}

/** The CSS a unit seen at `weight` is drawn with: blurred and faded the way the eye's periphery is. */
export function unitStyle(weight) {
  const w = Math.min(1, Math.max(0, Number(weight) || 0));
  return {
    filter: w >= 1 ? "none" : `blur(${((1 - w) * 2.4).toFixed(2)}px)`,
    opacity: Number((0.35 + 0.65 * w).toFixed(3)),
  };
}

/** The units of a gram: its characters, or its space-separated tokens under a word, phone, syllable or acoustic encoding. */
export function gramUnits(gram, unit) {
  const text = String(gram ?? "");
  return unit && unit !== "char" ? text.split(" ") : Array.from(text);
}

/** One sentence for the band as GET /api/model/attention reports it. */
export function describeBand(attention) {
  if (!attention || typeof attention !== "object") return "unknown";
  if (!attention.applies) return "not used: this kind is never corrected";
  if (!attention.on) return "off - each changed unit is charged to the step that wrote it";
  const weights = Array.isArray(attention.weights) ? attention.weights.map((w) => fmtShare(w)).join(" ") : "";
  return `on, blur ${fmtShare(attention.blur)} - the centre of each gram is charged most${weights ? ` (${weights})` : ""}`;
}

/** A share or a weight as the card prints it: up to two decimals, no trailing zeros. */
export function fmtShare(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "–";
  return String(Math.round(number * 100) / 100);
}

/**
 * The rows of one side of a preview worth showing: every gram that either rule charges, in order, as
 * {index, gram, writer, charge, focus}, and the step into END when the end is marked.
 */
export function chargedRows(side) {
  if (!side || typeof side !== "object") return [];
  const grams = Array.isArray(side.grams) ? side.grams : [];
  const rows = [];
  grams.forEach((gram, index) => {
    const writer = Boolean(side.writer && side.writer[index]);
    const charge = Number((side.charges && side.charges[index]) || 0);
    if (!writer && !(charge > 0)) return;
    rows.push({ index, gram, writer, charge, focus: Boolean(side.focus && side.focus[index]) && charge > 0 });
  });
  if (side.end) rows.push({ index: "end", gram: null, writer: true, charge: 1, focus: true });
  return rows;
}
