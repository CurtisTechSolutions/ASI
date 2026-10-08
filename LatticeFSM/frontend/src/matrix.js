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
  return steps.map((t) => `${t.source} -${t.symbol}(${Number(t.probability).toFixed(2)})-> ${t.target}`).join("  ");
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

/** The stimulation to run at, from the slider's string: blank or invalid means "the machine's own". */
export function stimulationFrom(text) {
  if (text === "" || text === null || text === undefined) return undefined;
  const n = Number(text);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}
