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
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(2)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function yesNo(value) {
  return value ? "yes" : "no";
}

/** Coerce anything the API might send into an array. */
export function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/** "0,2" -> [0, 2]: the accepting states as typed into a field; blanks and non-numbers dropped. */
export function parseStateList(text) {
  return String(text ?? "")
    .split(/[,\s]+/)
    .filter((s) => s !== "")
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n) && n >= 0);
}

/** A time the server sends (unix seconds, or milliseconds, or an ISO string) as the browser writes dates. */
export function fmtTime(value) {
  if (value === null || value === undefined || value === "") return "–";
  const date = typeof value === "number" ? new Date(value < 1e11 ? value * 1000 : value) : new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

/** "a while ago" for a unix time in seconds against `now` (milliseconds): "just now", "5 min ago", "3 h ago", "2 d ago". */
export function fmtAgo(seconds, now = Date.now()) {
  const t = Number(seconds);
  if (!Number.isFinite(t)) return "–";
  const ago = Math.max(0, now / 1000 - (t < 1e11 ? t : t / 1000));
  if (ago < 60) return "just now";
  if (ago < 3600) return `${Math.floor(ago / 60)} min ago`;
  if (ago < 86400) return `${Math.floor(ago / 3600)} h ago`;
  return `${Math.floor(ago / 86400)} d ago`;
}

/** "chain-20261011-120000.json.gz" -> trimmed and with no path: a name the server would keep as it is. */
export function cleanName(name) {
  return String(name ?? "")
    .split(/[\\/]/)
    .pop()
    .trim();
}
