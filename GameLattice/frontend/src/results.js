/**
 * Pure helpers behind the Results tab: `results/e1.json` and `results/e2.json` (what `make e1` and `make e2`
 * wrote, served as they are by `GET /api/results`) shaped into the tables the tab draws. No React, no fetch.
 */

/** The marks E1 is measured against when the file does not carry them (`experiment.rs`). */
export const E1_TO_BEAT = { "own-whole-run": 0.894, "own-oracle": 0.924, "explore-oracle": 0.944 };

/** The mean of the finite numbers in `values`, or null when there are none. */
export function mean(values) {
  const v = (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** `[[name, mark], ...]` of E1's marks, in the file's order. */
export function e1Marks(doc) {
  const marks = doc && doc.to_beat && typeof doc.to_beat === "object" ? doc.to_beat : E1_TO_BEAT;
  return Object.entries(marks).map(([name, v]) => [name, Number(v)]);
}

/**
 * E1's summary: one row per arm (in the file's order) with the mean held-out accuracy of the encoder's verdict and
 * of the lattice's own, and per mark whether each beats it.
 */
export function e1Summary(doc) {
  const summary = doc && doc.summary && typeof doc.summary === "object" ? doc.summary : {};
  const marks = e1Marks(doc);
  return Object.entries(summary).map(([arm, s]) => {
    const encoder = Number(s && s.encoder);
    const lattice = Number(s && s.lattice);
    return {
      arm,
      encoder,
      lattice,
      beats: Object.fromEntries(marks.map(([name, mark]) => [name, { encoder: encoder > mark, lattice: lattice > mark }])),
    };
  });
}

/** E1's rows grouped by arm and language: per pair the mean over seeds, and the seeds' own rows beneath. */
export function e1ByLanguage(doc) {
  const rows = doc && Array.isArray(doc.rows) ? doc.rows : [];
  const groups = new Map();
  for (const r of rows) {
    const key = `${r.arm}\u0000${r.language}`;
    if (!groups.has(key)) groups.set(key, { arm: r.arm, language: r.language, seeds: [] });
    groups.get(key).seeds.push(r);
  }
  return [...groups.values()].map((g) => ({
    ...g,
    encoder: mean(g.seeds.map((r) => r.encoder_accuracy)),
    lattice: mean(g.seeds.map((r) => r.lattice_accuracy)),
    agreement: mean(g.seeds.map((r) => r.agreement)),
    diversity: mean(g.seeds.map((r) => r.diversity_bits)),
  }));
}

/** The arms and the languages E1 ran, each in first-seen order. */
export function e1Axes(doc) {
  const rows = doc && Array.isArray(doc.rows) ? doc.rows : [];
  const arms = [];
  const languages = [];
  for (const r of rows) {
    if (!arms.includes(r.arm)) arms.push(r.arm);
    if (!languages.includes(r.language)) languages.push(r.language);
  }
  return { arms, languages };
}

/** The measures E2 reports per window, in the order the table shows them: `[key, heading]`. */
export const E2_COLUMNS = [
  ["legal_first_try", "legal first try"],
  ["refusals_per_move", "refusals / move"],
  ["win_rate", "wins"],
  ["win_rate_second_half", "wins (2nd half)"],
  ["optimal_rate", "optimal"],
  ["information_bits", "I(state; nim-sum) bits"],
  ["states_used", "states used"],
  ["agreement", "agreement"],
];

/** E2 by window width, ascending: per window the seeds it ran and the mean of every measure over them. */
export function e2ByWindow(doc) {
  const rows = doc && Array.isArray(doc.rows) ? doc.rows : [];
  const windows = [...new Set(rows.map((r) => Number(r.window)).filter(Number.isFinite))].sort((a, b) => a - b);
  return windows.map((w) => {
    const seeds = rows.filter((r) => Number(r.window) === w);
    const out = { window: w, seeds: seeds.length, rows: seeds };
    for (const [key] of E2_COLUMNS) out[key] = mean(seeds.map((r) => r[key]));
    return out;
  });
}
