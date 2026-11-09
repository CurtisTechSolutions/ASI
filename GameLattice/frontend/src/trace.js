/**
 * Pure helpers behind the naming trace (the Body and Game tabs): what `POST /api/classify` and `POST /api/run`
 * answer per character - the window's seats, the auction, the equilibrium, the aggregate over the lattice's
 * states, the named state and the lattice's prediction - turned into what the tables and bar strips draw.
 * No React, no fetch, so `test/trace.test.mjs` runs them under plain `node --test`.
 */

/** Where a seat sits in the walk window (`encoder.rs` BUCKETS), in order, with a word for each. */
export const BUCKETS = [
  ["current", "the current gram's node"],
  ["previous", "the node one position back"],
  ["older", "a node further back in the window"],
  ["child", "a child of the current node in the gram graph"],
];

/** The bucket's description, or the name itself for one the server added later. */
export function bucketTitle(bucket) {
  const found = BUCKETS.find(([name]) => name === bucket);
  return found ? found[1] : String(bucket ?? "");
}

/** Index of the largest value, ties to the lowest (the server's `argmax_low`); -1 for nothing. */
export function bestState(values) {
  const v = Array.isArray(values) ? values : [];
  let best = -1;
  for (let i = 0; i < v.length; i += 1) {
    const x = Number(v[i]);
    if (Number.isFinite(x) && (best < 0 || x > Number(v[best]))) best = i;
  }
  return best;
}

/** What a seat played: "s3", "abstain", or "-" for a candidate that was not seated. */
export function actionLabel(action) {
  if (action === null || action === undefined) return "-";
  if (action === "abstain") return "abstain";
  return `s${action}`;
}

/**
 * The bars of a strip over the `S` states: each state's share of the aggregate (scaled to the largest, so the
 * winner fills its slot), whether it is the named state and whether it is the lattice's prediction.
 */
export function stripBars(values, named = -1, predicted = -1) {
  const v = (Array.isArray(values) ? values : []).map((x) => (Number.isFinite(Number(x)) ? Number(x) : 0));
  const top = v.reduce((m, x) => (x > m ? x : m), 0);
  return v.map((p, state) => ({
    state,
    p,
    height: top > 0 ? p / top : 0,
    named: state === named,
    predicted: state === predicted,
  }));
}

/** A belief over `S + 1` slots split into the states and abstain. */
export function splitBelief(belief) {
  const b = Array.isArray(belief) ? belief.map(Number) : [];
  if (b.length === 0) return { states: [], abstain: 0 };
  return { states: b.slice(0, -1), abstain: b[b.length - 1] };
}

/** The top `k` states of a distribution as "5 0.92 · 2 0.05", highest first, ties to the lower state. */
export function fmtAggregate(values, k = 3, digits = 2) {
  const v = Array.isArray(values) ? values : [];
  return v
    .map((p, state) => ({ state, p: Number(p) }))
    .filter((x) => Number.isFinite(x.p))
    .sort((a, b) => b.p - a.p || a.state - b.state)
    .slice(0, k)
    .map((x) => `${x.state} ${x.p.toFixed(digits)}`)
    .join(" · ");
}

/** Why the meta-player would have been asked at a step, in words; "" when nothing would. */
export function triggerText(step) {
  const t = step && Array.isArray(step.triggers) ? step.triggers : [];
  return t.join(", ");
}

/** The seats of a step that were seated (in bid order), and the candidates that were not. */
export function seatsOf(step) {
  const all = step && Array.isArray(step.seats) ? step.seats : [];
  return { seated: all.filter((s) => s.seated), unseated: all.filter((s) => !s.seated) };
}

/** Whether a read carries Shapley values: only a learning run settles its seats. */
export function phiKnown(read) {
  const trace = read && Array.isArray(read.trace) ? read.trace : [];
  return trace.some((s) => s.learned);
}

/** The walk as one line: `0 -a-> 3 -b-> 3`, each arrow to the named state, `(lattice 2)` where the lattice predicted
 * another. The empty string's walk is "(empty)". */
export function walkLine(read) {
  const trace = read && Array.isArray(read.trace) ? read.trace : [];
  if (trace.length === 0) return "(empty)";
  const parts = [String(trace[0].prev)];
  for (const s of trace) {
    parts.push(`-${s.symbol}->`, s.predicted === s.named ? `${s.named}` : `${s.named} (lattice ${s.predicted})`);
  }
  return parts.join(" ");
}

/** "right" / "wrong" for a verdict against the language, or "" when membership is unknown. */
export function verdictWord(verdict, member) {
  if (typeof member !== "boolean") return "";
  return verdict === member ? "right" : "wrong";
}

/** How many steps the encoder and the lattice agreed on, of how many. */
export function agreementOf(read) {
  const trace = read && Array.isArray(read.trace) ? read.trace : [];
  return { agreed: trace.filter((s) => s.named === s.predicted).length, steps: trace.length };
}

/** A step index kept inside the trace: a number in 0..len-1, 0 for anything else. */
export function clampStep(index, length) {
  const n = Number(index);
  if (!Number.isInteger(n) || length <= 0) return 0;
  return Math.min(length - 1, Math.max(0, n));
}

/** A signed number with its sign always written: "+0.120", "-0.034". */
export function fmtSigned(value, digits = 3) {
  const n = Number(value);
  if (value === null || value === undefined || !Number.isFinite(n)) return "–";
  return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
}
