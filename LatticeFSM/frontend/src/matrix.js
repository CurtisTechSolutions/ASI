/**
 * Pure helpers behind the matrix and run panels: how a cell is shaded, how a
 * run is written out, how a curve becomes chart points. No React, no fetch,
 * so `test/matrix.test.mjs` runs them under plain `node --test`.
 */

/** A probability in [0, 1] -> the share (0-85) of the accent colour a cell is shaded with. */
export function shade(probability) {
  const p = Number(probability);
  if (!Number.isFinite(p)) return 0;
  return Math.round(Math.min(1, Math.max(0, p)) * 85);
}

/** The inline style of a matrix cell: the accent mixed into the cell colour by its probability. */
export function cellStyle(probability) {
  return { background: `color-mix(in srgb, var(--accent) ${shade(probability)}%, var(--cell))` };
}

/** `0 -a(0.25)-> 2  2 -b(0.50)-> 1`: a run's transitions as one line; the empty run is "(empty)". */
export function describeRun(run) {
  const steps = Array.isArray(run && run.transitions) ? run.transitions : [];
  if (steps.length === 0) return "(empty)";
  return steps
    .map((t) =>
      t.skipped === null || t.skipped === undefined
        ? `${t.source} -${t.symbol}(${Number(t.probability).toFixed(2)})-> ${t.target}`
        : `${t.source} =${t.symbol}(${Number(t.probability).toFixed(2)})=> ${t.target} [skipping ${t.skipped}]`,
    )
    .join("  ");
}

/** `[[episode, accuracy], ...]` from the API -> `[{x, y}]` for the line chart. */
export function curvePoints(curve) {
  return (Array.isArray(curve) ? curve : [])
    .filter((p) => Array.isArray(p) && p.length === 2)
    .map(([x, y]) => ({ x: Number(x), y: Number(y) }));
}

/** The greedy table as rows of text: `0*: a -> 0   b -> 3`, the star marking an accepting state. */
export function tableRows(table) {
  if (!table || !Array.isArray(table.table)) return [];
  const alphabet = Array.isArray(table.alphabet) ? table.alphabet : [];
  const accepting = Array.isArray(table.accepting) ? table.accepting : [];
  return table.table.map((row, s) => ({
    state: s,
    accepting: accepting.includes(s),
    cells: row.map((t, a) => `${alphabet[a] ?? a} -> ${t}`),
  }));
}

/** `1234567` -> "1.2 MB" style: bytes as the compress tab shows them. */
export function fmtSize(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n)) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

/** Whether `(source, symbolIndex, target)` is the matrix's central node, given the stats' `center`. */
export function isCenter(center, source, symbolIndex, target) {
  return Array.isArray(center) && center[0] === source && center[1] === symbolIndex && center[2] === target;
}

/** Which node of the central vertical vector a focus picks: `floor(focus * n)`, the top of the range in the last band. */
export function focusIndex(n, focus) {
  if (focus === null || focus === undefined || focus === "") return 0;
  const f = Math.min(1, Math.max(0, Number(focus)));
  return Math.min(n - 1, Math.floor(f * n));
}

/** Whether `(source, symbolIndex, target)` lies on the central vertical vector `(any, center[1], center[2])`. */
export function onVector(center, symbolIndex, target) {
  return Array.isArray(center) && center[1] === symbolIndex && center[2] === target;
}

/** The stimulation to run at, from the slider's string: blank or invalid means "the machine's own". */
export function stimulationFrom(text) {
  if (text === "" || text === null || text === undefined) return undefined;
  const n = Number(text);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

/** The cell at row-major index `i` of an `S x A x S` cube: `[source, symbol, target]`. */
export function cellAt(shape, i) {
  const [, a, t] = shape;
  return [Math.floor(i / (a * t)), Math.floor(i / t) % a, i % t];
}

/**
 * A cube value mapped onto 0..1 for drawing (net onto -1..1): probability as it is, seen on a log scale against the
 * busiest edge, width on a log scale between its bounds (0.05 .. 20; the rest width 1 is 0.5), net clipped.
 */
export function cubeValue(mode, v, max = 1) {
  if (!Number.isFinite(v)) return 0;
  if (mode === "seen") return max > 0 ? Math.log1p(Math.max(0, v)) / Math.log1p(max) : 0;
  if (mode === "width") return Math.min(1, Math.max(0, (Math.log(Math.max(v, 1e-9)) + Math.log(20)) / (2 * Math.log(20))));
  if (mode === "net") return Math.min(1, Math.max(-1, v));
  return Math.min(1, Math.max(0, v));
}

/** A point turned by `yaw` about the vertical axis and `pitch` about the horizontal, then seen in perspective from
 * `distance` along z: `[screen x, screen y, depth, scale]` with the centre of the view at the origin. */
export function project([x, y, z], yaw, pitch, distance, focal) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  const scale = focal / Math.max(1e-6, distance - z2);
  return [x1 * scale, -y2 * scale, z2, scale];
}

const NO_SPACE_BEFORE = new Set([".", ",", "!", "?", ";", ":", ")", "]", "}", "%", "'", '"']);

/** A predicted token as it is shown: a space character as ␣, the end of a sequence as "end". */
export function showToken(token, unit) {
  if (token === "</s>") return "⏎ end";
  if (unit === "char" && token === "_") return "␣";
  return token;
}

/** `text` with a predicted token taken: a word after a space (punctuation without one), a character as it is (␣ a
 * space), a sound after a space (the tokenizer reads its own sounds back). */
export function appendToken(text, token, unit) {
  if (token === "</s>" || token === "<s>" || token === "<unk>") return text;
  if (unit === "char") return text + (token === "_" ? " " : token);
  if (unit === "word" && NO_SPACE_BEFORE.has(token)) return text.replace(/\s+$/, "") + token;
  return text === "" || /\s$/.test(text) ? text + token : `${text} ${token}`;
}
