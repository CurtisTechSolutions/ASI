import { fmtInt } from "./util.js";

/**
 * The site-wide settings of this browser: what every search and every
 * training run starts from (../../SPEC-SearchAndTraining.md).
 *
 * Pure - no React, no storage - so the rules can be tested with plain
 * `node --test` and shared by the Settings tab and the tabs that use them
 * (Predict, Generate, Train). The values are kept as the strings the form
 * fields hold, so a field can be cleared while typing; everything here parses
 * them the way the servers will.
 *
 * Every setting is off at its default, and a request only carries a setting
 * that is not: a server that predates the setting never sees it, and a search
 * or a run with everything off is exactly the one it always was.
 */

/** The sampling filters and the beam's diversity, as the form holds them. */
export const SEARCH_DEFAULTS = Object.freeze({ topK: "0", topP: "1", minP: "0", diversity: "0" });

/** How a training run walks its texts, as the form holds them; a blank replay size leaves the model's buffer alone. */
export const TRAINING_DEFAULTS = Object.freeze({
  order: "corpus",
  curriculum: "1",
  replay: "0",
  replaySize: "",
  patience: "0",
  minDelta: "0",
});

/** The orders an epoch can walk the texts in, with what each means. */
export const ORDERS = Object.freeze([
  ["corpus", "corpus (as given)"],
  ["shortest-first", "shortest first"],
  ["longest-first", "longest first"],
  ["shuffle", "shuffle (fresh each epoch)"],
]);

const ORDER_NAMES = new Set(ORDERS.map(([name]) => name));

/** A form value as a number; `null` when it is blank or not a finite number. */
function numberOf(value) {
  if (value === "" || value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** A form value as a count (a whole number >= 0); `null` when it is not one. */
function countOf(value) {
  const n = numberOf(value);
  return n !== null && Number.isInteger(n) && n >= 0 ? n : null;
}

/**
 * What is wrong with the search settings, one message per setting (an empty
 * object when nothing is), in the ranges every server enforces: `top_k` a
 * whole number >= 0, `top_p` in (0, 1], `min_p` in [0, 1), `diversity` >= 0.
 */
export function searchProblems(values) {
  const v = { ...SEARCH_DEFAULTS, ...(values || {}) };
  const out = {};
  if (countOf(v.topK) === null) out.topK = "Top-K must be a whole number of options, 0 or more (0 = off).";
  const topP = numberOf(v.topP);
  if (topP === null || !(topP > 0 && topP <= 1)) out.topP = "Top-p must lie in (0, 1] (1 = off).";
  const minP = numberOf(v.minP);
  if (minP === null || !(minP >= 0 && minP < 1)) out.minP = "Min-p must lie in [0, 1) (0 = off).";
  const diversity = numberOf(v.diversity);
  if (diversity === null || diversity < 0) out.diversity = "Diversity must be a number, 0 or more (0 = off).";
  return out;
}

/** What is wrong with the training settings, one message per setting (empty when nothing is). */
export function trainingProblems(values) {
  const v = { ...TRAINING_DEFAULTS, ...(values || {}) };
  const out = {};
  if (!ORDER_NAMES.has(v.order)) out.order = `Unknown order ${JSON.stringify(v.order)}.`;
  const curriculum = numberOf(v.curriculum);
  if (curriculum === null || !(curriculum > 0 && curriculum <= 1)) {
    out.curriculum = "The curriculum must lie in (0, 1] (1 = off).";
  }
  const replay = numberOf(v.replay);
  if (replay === null || replay < 0) out.replay = "Replay must be a number, 0 or more (0 = off).";
  if (String(v.replaySize).trim() !== "" && countOf(v.replaySize) === null) {
    out.replaySize = "The replay buffer's size must be a whole number of texts, 0 or more (blank = leave it).";
  }
  if (countOf(v.patience) === null) out.patience = "Patience must be a whole number of epochs, 0 or more (0 = off).";
  const minDelta = numberOf(v.minDelta);
  if (minDelta === null || minDelta < 0) out.minDelta = "The minimum improvement must be a number, 0 or more.";
  return out;
}

/** The problems as one message, or `null` - what a form shows before it refuses to send. */
export function problemText(problems) {
  const messages = Object.values(problems || {});
  return messages.length ? messages.join(" ") : null;
}

/** Which of the search settings a mode reads: the filters shape a sampled walk, diversity a beam. */
export function searchSettingsFor(mode) {
  return { filters: mode === "sample", diversity: mode === "beam" };
}

/** The problems of only the settings `mode` reads - what stops that search from being sent. */
export function searchProblemsFor(values, mode) {
  const reads = searchSettingsFor(mode);
  return Object.fromEntries(
    Object.entries(searchProblems(values)).filter(([name]) => (name === "diversity" ? reads.diversity : reads.filters)),
  );
}

/**
 * The fields of a /api/predict or /api/generate body for these settings and
 * the mode the search actually runs in (the count model's "dijkstra" is its
 * beam, so pass "beam" for it). Only what the mode reads and what is not off
 * is sent; a setting out of range is left out too - the form refuses to send
 * it first (`searchProblems`).
 */
export function searchBody(values, mode) {
  const v = { ...SEARCH_DEFAULTS, ...(values || {}) };
  const reads = searchSettingsFor(mode);
  const problems = searchProblems(v);
  const body = {};
  if (reads.filters) {
    const topK = countOf(v.topK);
    if (!problems.topK && topK > 0) body.top_k = topK;
    const topP = numberOf(v.topP);
    if (!problems.topP && topP < 1) body.top_p = topP;
    const minP = numberOf(v.minP);
    if (!problems.minP && minP > 0) body.min_p = minP;
  }
  if (reads.diversity) {
    const diversity = numberOf(v.diversity);
    if (!problems.diversity && diversity > 0) body.diversity = diversity;
  }
  return body;
}

/** Whether any search setting is on - what a tab mentions beside its own fields. */
export function searchActive(values) {
  return Object.keys(searchBody(values, "sample")).length > 0 || Object.keys(searchBody(values, "beam")).length > 0;
}

/** The fields of a /api/train body for these settings: only the ones that are not off (and are in range). */
export function trainingBody(values) {
  const v = { ...TRAINING_DEFAULTS, ...(values || {}) };
  const problems = trainingProblems(v);
  const body = {};
  if (!problems.order && v.order !== "corpus") body.order = v.order;
  const curriculum = numberOf(v.curriculum);
  if (!problems.curriculum && curriculum < 1) body.curriculum = curriculum;
  const replay = numberOf(v.replay);
  if (!problems.replay && replay > 0) body.replay = replay;
  if (!problems.replaySize && String(v.replaySize).trim() !== "") body.replay_size = countOf(v.replaySize);
  const patience = countOf(v.patience);
  if (!problems.patience && patience > 0) {
    body.patience = patience;
    const minDelta = numberOf(v.minDelta);
    if (!problems.minDelta && minDelta > 0) body.min_delta = minDelta;
  }
  return body;
}

/** Whether any training setting is on. */
export function trainingActive(values) {
  return Object.keys(trainingBody(values)).length > 0;
}

/**
 * How many texts each epoch walks under a curriculum - the rule the servers
 * follow (`ceil(frac_j * n)`, clamped to [1, n], with `frac_j` growing from
 * the curriculum to 1 over the run), for the preview beside the field.
 */
export function pacedCounts(n, curriculum, epochs) {
  const count = Math.max(0, Math.trunc(Number(n) || 0));
  const e = Math.max(0, Math.trunc(Number(epochs) || 0));
  const c = Number(curriculum);
  const out = [];
  for (let j = 0; j < e; j += 1) {
    if (count === 0) {
      out.push(0);
    } else if (!(c < 1) || e <= 1) {
      out.push(count);
    } else {
      const frac = c + ((1 - c) * j) / (e - 1);
      out.push(Math.min(count, Math.max(1, Math.ceil(frac * count))));
    }
  }
  return out;
}

/** How many buffered texts an epoch rehearses: `replay` of the run's `n` texts, rounded half up, at most the pool. */
export function rehearsalCount(replay, n, pool) {
  const r = Number(replay);
  if (!(r > 0) || !(pool > 0)) return 0;
  return Math.min(pool, Math.floor(r * n + 0.5));
}

/**
 * The encodings a new model can be made in, as `unit:n:stride` specs the
 * servers parse, with a word for each. `custom` is the three dials by hand.
 */
export const ENCODING_PRESETS = Object.freeze([
  ["char:3:1", "character trigram (the default)"],
  ["char:2:1", "character bigram"],
  ["char:5:1", "character 5-gram"],
  ["char:4:4", "groups of 4 letters"],
  ["char:5:5", "groups of 5 letters"],
  ["word:2:1", "word bigram"],
  ["word:3:1", "word trigram"],
  ["phone:3:1", "phone trigram (the text as sounds)"],
  ["syllable:2:1", "syllable bigram"],
  ["acoustic:3:1", "acoustic unit trigram (sounds learned from recordings)"],
  ["custom", "custom (unit, n, stride)"],
]);

/**
 * What one unit of a model can be, as the servers name it, with what to call
 * it: characters and words, and the sounds the phonetic tokenizer reads -
 * phones and syllables from the text, acoustic units learned from recordings.
 */
export const UNITS = Object.freeze([
  ["char", "characters"],
  ["word", "words"],
  ["phone", "phones (the text as sounds)"],
  ["syllable", "syllables"],
  ["acoustic", "acoustic units (learned from recordings)"],
]);

const UNIT_NAMES = Object.freeze({
  char: ["character", "characters"],
  word: ["word", "words"],
  phone: ["phone", "phones"],
  syllable: ["syllable", "syllables"],
  acoustic: ["acoustic unit", "acoustic units"],
});

/** True for a unit the servers know. */
export function isUnit(unit) {
  return Object.prototype.hasOwnProperty.call(UNIT_NAMES, unit);
}

/** A unit as an English word: "character" / "characters", "acoustic unit" / "acoustic units" (`char` when unknown). */
export function unitWord(unit, plural = true) {
  const names = UNIT_NAMES[unit] || UNIT_NAMES.char;
  return names[plural ? 1 : 0];
}

/** True for the units the phonetic tokenizer reads: the server needs the PhoneticTokenizer package beside it. */
export function soundUnit(unit) {
  return unit === "phone" || unit === "syllable" || unit === "acoustic";
}

/**
 * An encoding spec from the three dials, or an error message: the unit one
 * of `UNITS`, `n` a whole number >= 1 and the stride one from 1 to `n`
 * (1 slides the window, `n` cuts non-overlapping groups).
 */
export function encodingSpec(unit, n, stride) {
  if (!isUnit(unit)) {
    return { error: `The unit must be char, word, phone, syllable or acoustic, not ${JSON.stringify(unit)}.` };
  }
  const size = countOf(n);
  if (size === null || size < 1) return { error: "n must be a whole number of units, 1 or more." };
  const step = countOf(stride);
  if (step === null || step < 1 || step > size) return { error: `The stride must be a whole number from 1 to n (${size}).` };
  return { spec: `${unit}:${size}:${step}` };
}

/** `unit:n:stride` in words: "word trigram", "groups of 5 characters", "7-grams of phones, stride 2". */
export function describeEncoding(spec) {
  const [unit, rawN, rawStride] = String(spec || "").split(":");
  const n = Number(rawN);
  const stride = Number(rawStride);
  if (!isUnit(unit) || !Number.isInteger(n) || !Number.isInteger(stride)) return String(spec || "");
  const units = unitWord(unit);
  const named = { 2: "bigram", 3: "trigram" }[n];
  if (stride === 1) return named ? `${unitWord(unit, false)} ${named}` : `${n}-grams of ${units}`;
  if (stride === n) return `groups of ${n} ${units}`;
  return `${n}-grams of ${units}, stride ${stride}`;
}

/** The model's replay buffer in one line, from `/api/status` (`replay`: {size, texts, seen} or null). */
export function replayLine(replay) {
  if (!replay || typeof replay !== "object") return "This model keeps no replay buffer.";
  return (
    `This model keeps a replay buffer of ${fmtInt(replay.size)} texts: ${fmtInt(replay.texts)} held, a uniform ` +
    `sample of the ${fmtInt(replay.seen)} it was ever trained on.`
  );
}
