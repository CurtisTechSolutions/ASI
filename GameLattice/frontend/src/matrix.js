/**
 * Pure helpers behind the Lattice tab and the training chart: how a cell is shaded (LatticeFSM's matrix view, as
 * it is), how the greedy table is written, how a held-out curve becomes chart series. No React, no fetch.
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

/** The greedy table as rows: per state whether it accepts, whether it is the start, and its target per symbol. */
export function tableRows(table) {
  if (!table || !Array.isArray(table.table)) return [];
  const alphabet = Array.isArray(table.alphabet) ? table.alphabet : [];
  const accepting = Array.isArray(table.accepting) ? table.accepting : [];
  return table.table.map((row, s) => ({
    state: s,
    accepting: accepting.includes(s),
    start: table.start === s,
    cells: row.map((t, a) => ({ symbol: alphabet[a] ?? String(a), target: t })),
  }));
}

/**
 * The held-out curve `[[episodes, encoder, lattice], ...]` (`POST /api/train`, `GET /api/job`) as the line chart's
 * two series; points that are not triples of numbers are skipped.
 */
export function curveSeries(curve, colors = ["var(--accent)", "var(--accent-2)"]) {
  const points = (Array.isArray(curve) ? curve : []).filter(
    (p) => Array.isArray(p) && p.length >= 3 && p.slice(0, 3).every((x) => Number.isFinite(Number(x))),
  );
  return [
    { name: "encoder (the named state accepts)", color: colors[0], values: points.map((p) => ({ x: Number(p[0]), y: Number(p[1]) })) },
    { name: "lattice (its own prediction accepts)", color: colors[1], dashed: true, values: points.map((p) => ({ x: Number(p[0]), y: Number(p[2]) })) },
  ];
}
