/**
 * Pure helpers behind the language-model tabs (Train, Predict, Generate, Talk, Converse, Chat, Score, Nodes,
 * Feedback): the two models the selector offers, how a chain's walk is drawn, how a token is tinted by its
 * probability or its depth, how marks out of 10 become good and bad texts. No React, no fetch, so
 * `test/lm.test.mjs` runs them under plain `node --test`.
 */

/** The language models the header's selector offers, in its order. */
export const LMS = [
  {
    id: "chain",
    label: "Chain",
    description:
      "input nodes and an output node: a prediction walks back through the input nodes only until it has gathered enough confidence",
  },
  {
    id: "ngram",
    label: "N-gram",
    description: "interpolated Kneser-Ney over the lattice's adaptive edges: the context is the last N-1 units",
  },
];

/** A stored or typed choice of model, made safe: anything but a known model is the chain. */
export function lmId(value) {
  return LMS.some((m) => m.id === value) ? value : "chain";
}

export function lmLabel(id) {
  const found = LMS.find((m) => m.id === id);
  return found ? found.label : String(id ?? "");
}

/** The units a model reads, as its New-model card offers them. */
export const UNITS = [
  ["word", "words"],
  ["char", "characters"],
  ["sound", "sounds (phonemes, spelled back as English)"],
  ["base64", "base64 letters (any file: text, images, audio, video)"],
];

/** "letter" for a model of characters, "base64 letter", "word", "sound": what one token is called. */
export function unitWord(unit) {
  if (unit === "char") return "letter";
  if (unit === "base64") return "base64 letter";
  return unit || "token";
}

/** A share in [0, 1] as "17.2%"; nothing as "-". */
export function pct(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

/**
 * The text the model wrote, in context, as it is credited: what it was given and what it went on with.
 * Base64 continuations already hold the decoded prompt and continuation; a model of letters goes straight
 * on from the last letter (its continuation carries its own spaces); words and sounds take a space between.
 */
export function joinContinuation(text, completion, unit) {
  const head = String(text ?? "");
  const tail = String(completion ?? "");
  if (unit === "base64") return tail;
  if (unit === "char") return head + tail;
  if (!head.trim()) return tail.trim();
  if (!tail.trim()) return head;
  return `${head.replace(/\s+$/, "")} ${tail.replace(/^\s+/, "")}`;
}

const NO_SPACE_BEFORE = new Set([".", ",", "!", "?", ";", ":", ")", "]", "}", "%", "'", '"', "</s>"]);

/**
 * Tokens (and, for the chain, how deep the walk went for each) as the pieces they are shown in: a letter as it
 * is (`_` a space), a word or a sound after a space unless it is punctuation. `[{text, token, depth}]`.
 */
export function tokenPieces(tokens, depths, unit) {
  const list = Array.isArray(tokens) ? tokens : [];
  const deep = Array.isArray(depths) ? depths : [];
  return list.map((token, i) => {
    const t = String(token ?? "");
    let text;
    if (t === "</s>") text = " ⏎";
    else if (unit === "char" || unit === "base64") text = t === "_" && unit === "char" ? " " : t;
    else text = i === 0 || NO_SPACE_BEFORE.has(t) ? t : ` ${t}`;
    const d = deep[i];
    return { text, token: t, depth: Number.isFinite(Number(d)) && d !== null ? Number(d) : null };
  });
}

/** The mean of the finite numbers in a list, or null when there are none. */
export function mean(values) {
  const finite = (Array.isArray(values) ? values : [])
    .filter((v) => v !== null && v !== undefined && v !== "")
    .map(Number)
    .filter((v) => Number.isFinite(v));
  if (!finite.length) return null;
  return finite.reduce((a, b) => a + b, 0) / finite.length;
}

/**
 * How strongly a token is tinted by its probability (0-85, the share of the colour mixed in): on a log scale,
 * 1 full, one in ten thousand or less not at all - so a surprising token stands out pale against likely ones.
 */
export function probShare(p) {
  const v = Number(p);
  if (!Number.isFinite(v) || v <= 0) return 0;
  const scaled = (Math.log10(Math.min(1, v)) + 4) / 4;
  return Math.round(Math.min(1, Math.max(0, scaled)) * 85);
}

/** How strongly a token is tinted by the depth its walk went to, against the deepest one could go (0-85). */
export function depthShare(depth, maxDepth) {
  const d = Number(depth);
  const m = Number(maxDepth);
  if (!Number.isFinite(d) || !(m > 0)) return 0;
  return Math.round(Math.min(1, Math.max(0, d / m)) * 85);
}

/** The inline style of a scored token: the accent mixed in by its probability. */
export function probStyle(p) {
  return { background: `color-mix(in srgb, var(--accent) ${probShare(p)}%, transparent)` };
}

/** The inline style of a token by its walk's depth: the second accent mixed in, the deeper the stronger. */
export function depthStyle(depth, maxDepth) {
  return { background: `color-mix(in srgb, var(--accent-2) ${depthShare(depth, maxDepth)}%, transparent)` };
}

/**
 * A chain's prediction as the row of boxes the Predict tab draws: the input nodes 1..nodes-1 holding their
 * inputs, then the output node. The walk goes from the output node (depth 0) back, nearest first, so the
 * input node at depth d is node `nodes - d`; a box carries its step of the walk, or null when the walk
 * stopped before it (it is drawn greyed).
 */
export function chainBoxes(prediction) {
  const p = prediction && typeof prediction === "object" ? prediction : {};
  const inputs = Array.isArray(p.inputs) ? p.inputs : [];
  const walk = Array.isArray(p.walk) ? p.walk : [];
  const nodes = Number.isInteger(p.nodes) && p.nodes > 0 ? p.nodes : inputs.length + 1;
  const byDepth = new Map();
  walk.forEach((step, i) => {
    const depth = Number.isFinite(Number(step && step.depth)) ? Number(step.depth) : i;
    byDepth.set(depth, step);
  });
  const boxes = [];
  for (let node = 1; node < nodes; node += 1) {
    const depth = nodes - node;
    const input = inputs[node - 1];
    boxes.push({
      node,
      depth,
      output: false,
      input: input === undefined ? null : input,
      step: byDepth.get(depth) || null,
    });
  }
  boxes.push({ node: nodes, depth: 0, output: true, input: null, step: byDepth.get(0) || null });
  return boxes;
}

/** Marks a rating can carry: a whole number from 0 to 10, 5 the default (neither good nor bad). */
export function clampMark(value, fallback = 5) {
  const n = Number(value);
  if (value === "" || value === null || value === undefined || !Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(10, Math.round(n)));
}

/**
 * Marked texts -> the body of `POST /api/{m}/feedback`: a mark of `good` or more (6) is a good text, `bad` or
 * less (4) a bad one, anything between is left out. Texts are kept once each, in their order.
 */
export function splitByMark(ratings, { good = 6, bad = 4 } = {}) {
  const out = { good: [], bad: [], skipped: 0 };
  const seen = new Set();
  for (const r of Array.isArray(ratings) ? ratings : []) {
    const text = String((r && r.text) ?? "");
    if (!text.trim() || seen.has(text)) continue;
    seen.add(text);
    const mark = clampMark(r.mark);
    if (mark >= good) out.good.push(text);
    else if (mark <= bad) out.bad.push(text);
    else out.skipped += 1;
  }
  return out;
}

/** The training history as chart points: `[{x: round, y}]` of one field, rounds without it dropped. */
export function historyPoints(history, field) {
  return (Array.isArray(history) ? history : [])
    .map((r, i) => ({ x: Number.isFinite(Number(r && r.round)) ? Number(r.round) : i, y: r ? r[field] : undefined }))
    .filter((p) => p.y !== null && p.y !== undefined && Number.isFinite(Number(p.y)))
    .map((p) => ({ x: p.x, y: Number(p.y) }));
}

/** A confidence sweep's rows as chart series: accuracy (a share) and mean depth against the threshold. */
export function sweepSeries(rows) {
  const list = (Array.isArray(rows) ? rows : []).filter((r) => r && Number.isFinite(Number(r.confidence)));
  const pick = (field) =>
    list
      .filter((r) => Number.isFinite(Number(r[field])))
      .map((r) => ({ x: Number(r.confidence), y: Number(r[field]) }));
  return { accuracy: pick("accuracy"), depth: pick("depth"), perplexity: pick("perplexity") };
}

/** The sweep's row with the best accuracy (the shallower one on a tie), or null. */
export function bestSweepRow(rows) {
  let best = null;
  for (const r of Array.isArray(rows) ? rows : []) {
    if (!r || !Number.isFinite(Number(r.accuracy))) continue;
    if (
      !best ||
      Number(r.accuracy) > Number(best.accuracy) ||
      (Number(r.accuracy) === Number(best.accuracy) && Number(r.depth) < Number(best.depth))
    ) {
      best = r;
    }
  }
  return best;
}

/**
 * Where each of the chain's contexts sits in the Nodes tab's rings: one ring per depth (depth 1 the
 * innermost), the contexts of a depth spread evenly round it, each circle sized by how often it was seen.
 * `[{...node, x, y, r}]` in a square of `size`.
 */
export function ringLayout(nodes, size = 520) {
  const list = (Array.isArray(nodes) ? nodes : []).filter((n) => n && Number.isFinite(Number(n.depth)));
  if (!list.length) return [];
  const depths = [...new Set(list.map((n) => Number(n.depth)))].sort((a, b) => a - b);
  const centre = size / 2;
  const outer = size / 2 - 36;
  const step = outer / Math.max(1, depths.length);
  const maxSeen = Math.max(1, ...list.map((n) => Number(n.seen) || 0));
  const out = [];
  depths.forEach((depth, ring) => {
    const here = list.filter((n) => Number(n.depth) === depth);
    const radius = step * (ring + 1);
    here.forEach((n, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / here.length + ring * 0.3;
      out.push({
        ...n,
        ring,
        x: centre + radius * Math.cos(angle),
        y: centre + radius * Math.sin(angle),
        r: 4 + 12 * Math.sqrt((Number(n.seen) || 0) / maxSeen),
      });
    });
  });
  return out;
}

/** A context of tokens as one line: letters run together (`_` a space, shown ␣), words and sounds spaced. */
export function contextText(context, unit) {
  const list = Array.isArray(context) ? context.map((t) => String(t ?? "")) : [];
  if (unit === "char") return list.map((t) => (t === "_" || t === " " ? "␣" : t)).join("");
  if (unit === "base64") return list.join("");
  return list.join(" ");
}

/**
 * How teaching until N of N ended (`POST /api/{m}/teach`, `/api/network/teach`): "taught in 7 rounds: 33 of 33
 * right", or "stopped after 200 rounds at 30 of 33"; an answer without rounds is just "12 of 33 right".
 */
export function untilNote(r) {
  if (!r) return "";
  if (r.rounds === undefined || r.rounds === null) return `${r.right} of ${r.of} right`;
  const rounds = `${r.rounds} round${r.rounds === 1 ? "" : "s"}`;
  return r.reached ? `taught in ${rounds}: ${r.right} of ${r.of} right` : `stopped after ${rounds} at ${r.right} of ${r.of}`;
}

/**
 * Training until N of N (`until` on `/api/{m}/train` and `/text`): its rounds `[{round, right, of}]` and
 * `reached` as one line - "33 of 33 reached in 4 rounds" or "stopped after 20 rounds at 30 of 33".
 */
export function roundsNote(rounds, reached) {
  const list = Array.isArray(rounds) ? rounds : [];
  if (!list.length) return "";
  const last = list[list.length - 1];
  const n = `${list.length} round${list.length === 1 ? "" : "s"}`;
  return reached ? `${last.of} of ${last.of} reached in ${n}` : `stopped after ${n} at ${last.right} of ${last.of}`;
}
