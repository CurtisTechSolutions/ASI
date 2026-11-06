/** Parsing and formatting helpers shared by the panels. */

/** Parse a numeric input value; blank or invalid -> fallback. */
export function parseNumber(value, fallback) {
  if (value === "" || value === null || value === undefined) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Parse an integer input value; blank or invalid -> fallback. */
export function parseInteger(value, fallback) {
  const n = parseNumber(value, fallback);
  return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

/** Fixed-point number; missing values render as an en dash, other types are echoed. */
export function fmtNum(value, digits = 4) {
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : String(value);
  if (value === null || value === undefined) return "–";
  return String(value);
}

export function fmtInt(value) {
  if (typeof value === "number" && Number.isFinite(value)) return Math.round(value).toLocaleString();
  if (value === null || value === undefined) return "–";
  return String(value);
}


export function yesNo(value) {
  return value ? "yes" : "no";
}

/** Coerce anything the API might send into an array. */
export function asArray(value) {
  return Array.isArray(value) ? value : [];
}
