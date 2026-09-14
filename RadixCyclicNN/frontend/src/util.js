/** Parsing and formatting helpers shared by the panels. */

/** Textarea content -> list of texts: one per line, blank lines dropped (mirrors the API's newline form). */
export function splitLines(text) {
  return String(text ?? "")
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "");
}

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

export function fmtBytes(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "–";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

/** ISO timestamp (or epoch seconds / milliseconds) -> local date-time; unknown input is echoed back. */
export function fmtTime(value) {
  if (value === null || value === undefined || value === "") return "–";
  const date = typeof value === "number" ? new Date(value < 1e11 ? value * 1000 : value) : new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

/** Make whitespace inside node labels visible (labels legitimately contain spaces and newlines). */
export function showWhitespace(label) {
  return String(label ?? "")
    .replace(/\n/g, "↵")
    .replace(/\t/g, "⇥")
    .replace(/ /g, "␣");
}

export function yesNo(value) {
  return value ? "yes" : "no";
}

/** Coerce anything the API might send into an array. */
export function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/** True when the status payload reports a running background job. */
export function jobIsRunning(status) {
  return Boolean(status && status.job && status.job.state === "running");
}

/** The active model kind reported by the status payload (`""` when offline). */
export function modelKind(status) {
  return status && typeof status.kind === "string" ? status.kind : "";
}

/**
 * True for a model that learns by counting rather than by a gradient: the
 * count / reward model and the resonant one. Those have no learning rates or
 * batches to set, and take a `strength` instead.
 */
export function countingKind(status) {
  const kind = modelKind(status);
  return kind !== "" && kind !== "radix";
}
