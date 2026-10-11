/**
 * Pure helpers behind the guard (the negative network: it keeps only failures, and why they were failures):
 * how the server cuts a text into tokens, how a verdict's blamed spans - windows of token indices - land on
 * the text's own words, and what the guard notice says. No React, no fetch, so `test/guard.test.mjs` runs
 * them under plain `node --test`.
 */

/** A verdict as a badge class: rejected red, suspect amber, passed green. */
export const VERDICT_CLASS = { reject: "fail", suspect: "unrated", pass: "pass" };

export function verdictClass(verdict) {
  return VERDICT_CLASS[verdict] || "unrated";
}

/** The units the guard reads (a new guard picks one); its words and letters can be marked inside a text. */
export const GUARD_UNITS = [
  ["word", "words"],
  ["char", "characters"],
  ["sound", "sounds"],
  ["base64", "base64 letters"],
];

const WORD = /[\p{Alphabetic}\p{N}_]/u;
const ALPHA = /\p{Alphabetic}/u;
const SPACE = /\p{White_Space}/u;

/**
 * A text as the guard's pieces: `[{text, index}]`, every token with its index and the spaces between them with
 * index -1, so that joined together they are the text again. Words, the way the server cuts them: a run of
 * letters, digits and `_` (an apostrophe inside, between a word character and a letter: "don't"), every other
 * non-space character alone. Characters: the trimmed text letter by letter, a space inside a token too. Any
 * other unit (sounds, base64) cannot be laid on the text: null.
 */
export function guardPieces(text, unit = "word") {
  const chars = Array.from(String(text ?? ""));
  const n = chars.length;
  const out = [];
  const gap = (s) => {
    if (!s) return;
    const last = out[out.length - 1];
    if (last && last.index === -1) last.text += s;
    else out.push({ text: s, index: -1 });
  };
  if (unit === "char") {
    let a = 0;
    let b = n;
    while (a < b && SPACE.test(chars[a])) a += 1;
    while (b > a && SPACE.test(chars[b - 1])) b -= 1;
    gap(chars.slice(0, a).join(""));
    for (let i = a; i < b; i += 1) out.push({ text: chars[i], index: i - a });
    gap(chars.slice(b).join(""));
    return out;
  }
  if (unit !== "word") return null;
  let index = 0;
  let i = 0;
  while (i < n) {
    const c = chars[i];
    if (SPACE.test(c)) {
      gap(c);
      i += 1;
    } else if (WORD.test(c)) {
      let j = i + 1;
      while (
        j < n &&
        (WORD.test(chars[j]) || (chars[j] === "'" && j + 1 < n && ALPHA.test(chars[j + 1]) && WORD.test(chars[j - 1])))
      ) {
        j += 1;
      }
      out.push({ text: chars.slice(i, j).join(""), index: index++ });
      i = j;
    } else {
      out.push({ text: c, index: index++ });
      i += 1;
    }
  }
  return out;
}

/**
 * A text's tokens as the guard reads them (lower-cased, as it matches them): "Don't stop!" -> ["don't", "stop",
 * "!"]; for characters a space inside is "_". Null for a unit that cannot be laid on the text.
 */
export function guardTokens(text, unit = "word", lower = true) {
  const pieces = guardPieces(text, unit);
  if (!pieces) return null;
  return pieces
    .filter((p) => p.index >= 0)
    .map((p) => {
      const t = unit === "char" && SPACE.test(p.text) ? "_" : p.text;
      return lower ? t.toLowerCase() : t;
    });
}

/** A span's [start, end) as whole numbers, or null when it is not a window at all. */
function spanWindow(span) {
  if (!span || typeof span !== "object") return null;
  const start = Number(span.start);
  const end = Number(span.end);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) return null;
  return { start, end };
}

/**
 * Whether every span's tokens are the text's tokens at its window (ignoring case): then the windows can be marked
 * on the text. A span without its tokens is taken on trust if it fits inside the text.
 */
export function spansAlign(tokens, spans) {
  if (!Array.isArray(tokens)) return false;
  const list = Array.isArray(spans) ? spans : [];
  return list.every((span) => {
    const w = spanWindow(span);
    if (!w || w.end > tokens.length) return false;
    if (!Array.isArray(span.tokens)) return true;
    const have = tokens.slice(w.start, w.end);
    return span.tokens.length === have.length && span.tokens.every((t, k) => String(t).toLowerCase() === have[k].toLowerCase());
  });
}

/**
 * For each of `count` tokens, what the spans covering it say: null when none does, else `{evidence, reasons,
 * spans}` - the most evidence against any span over it, the reasons (each once, worst first) and the indices of
 * the spans over it.
 */
export function tokenMarks(count, spans) {
  const marks = Array.from({ length: Math.max(0, count | 0) }, () => null);
  (Array.isArray(spans) ? spans : []).forEach((span, s) => {
    const w = spanWindow(span);
    if (!w) return;
    const evidence = Number(span.evidence) || 0;
    for (let i = w.start; i < Math.min(w.end, marks.length); i += 1) {
      const m = marks[i] || { evidence: 0, reasons: [], spans: [] };
      m.evidence = Math.max(m.evidence, evidence);
      const reason = span.reason ? String(span.reason) : "blamed";
      if (!m.reasons.includes(reason)) m.reasons.push(reason);
      m.spans.push(s);
      marks[i] = m;
    }
  });
  return marks;
}

/** How strongly a mark is tinted: 0.15 for the faintest evidence up to 0.6 at the text's worst (`peak`). */
export function markAlpha(evidence, peak) {
  const e = Number(evidence);
  const p = Number(peak);
  if (!Number.isFinite(e) || e <= 0) return 0;
  const share = Number.isFinite(p) && p > 0 ? Math.min(1, e / p) : 1;
  return Math.round((0.15 + 0.45 * share) * 100) / 100;
}

/**
 * A judged text with its blamed spans laid on it: `{unit, pieces: [{text, mark}]}` - every piece of the text
 * with the mark of the spans over it (null for none; a space between two tokens of one span is marked with
 * them) - or null when the spans cannot be laid on the text in `unit` (or, without one, in words or in
 * characters, whichever fits).
 */
export function highlightPieces(text, spans, unit) {
  const list = (Array.isArray(spans) ? spans : []).filter((s) => spanWindow(s));
  const units = unit ? [unit] : ["word", "char"];
  for (const u of units) {
    const pieces = guardPieces(text, u);
    if (!pieces) continue;
    const tokens = guardTokens(text, u);
    if (!spansAlign(tokens, list)) continue;
    const marks = tokenMarks(tokens.length, list);
    const out = pieces.map((p, k) => {
      if (p.index >= 0) return { text: p.text, mark: marks[p.index] };
      const before = k > 0 ? pieces[k - 1] : null;
      const after = k + 1 < pieces.length ? pieces[k + 1] : null;
      const a = before && before.index >= 0 ? marks[before.index] : null;
      const b = after && after.index >= 0 ? marks[after.index] : null;
      const shared = a && b ? a.spans.filter((s) => b.spans.includes(s)) : [];
      return { text: p.text, mark: shared.length ? { ...a, evidence: Math.min(a.evidence, b.evidence) } : null };
    });
    return { unit: u, pieces: out };
  }
  return null;
}

const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/**
 * The guard notice's line for a guarded draw's summary (`{judged, vetoed, all_vetoed}`): "the guard judged 9
 * samples and vetoed 2", with " · all vetoed: kept the least risky" when nothing passed. "" without a summary.
 */
export function guardNoticeText(guard, what = "samples") {
  if (!guard || typeof guard !== "object") return "";
  const judged = Number(guard.judged) || 0;
  const vetoed = Number(guard.vetoed) || 0;
  const one = what.endsWith("s") ? what.slice(0, -1) : what;
  let line = `the guard judged ${plural(judged, one, what)} and vetoed ${vetoed}`;
  if (guard.all_vetoed) line += " · all vetoed: kept the least risky";
  return line;
}

/** A share as a percentage: 0.25 -> "25%". */
export function pct(share, digits = 0) {
  if (share === null || share === undefined || share === "") return "–";
  const x = Number(share);
  return Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "–";
}
