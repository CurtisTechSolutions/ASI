/** Tests for the Results tab's shaping (`src/results.js`) against the experiments' file shapes - plain `node --test`. */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { E1_TO_BEAT, E2_COLUMNS, e1Axes, e1ByLanguage, e1Marks, e1Summary, e2ByWindow, mean } from "../src/results.js";

const e1 = {
  experiment: "E1",
  to_beat: { "own-whole-run": 0.894, "own-oracle": 0.924, "explore-oracle": 0.944 },
  summary: { "plain-oracle": { encoder: 0.96, lattice: 0.96 }, "game-whole-run": { encoder: 0.98, lattice: 0.66 } },
  rows: [
    { arm: "plain-oracle", language: "even-b", seed: 1, encoder_accuracy: 1, lattice_accuracy: 1, agreement: 0.9, diversity_bits: 1 },
    { arm: "plain-oracle", language: "even-b", seed: 2, encoder_accuracy: 0.9, lattice_accuracy: 0.8, agreement: 0.7, diversity_bits: 2 },
    { arm: "game-whole-run", language: "mod3-a", seed: 1, encoder_accuracy: 0.98, lattice_accuracy: 0.66, agreement: 0.5, diversity_bits: 3 },
  ],
};

const e2 = {
  experiment: "E2",
  rows: [
    { window: 4, seed: 1, legal_first_try: 0.2, win_rate: 0.5, information_bits: 0.02, states_used: 13 },
    { window: 1, seed: 1, legal_first_try: 0.1, win_rate: 0.4, information_bits: 0.01, states_used: 12 },
    { window: 4, seed: 2, legal_first_try: 0.3, win_rate: 0.6, information_bits: 0.04, states_used: 11 },
  ],
};

test("the mean skips what is not a number and is null for nothing", () => {
  assert.equal(mean([1, 2, "x", 3]), 2);
  assert.equal(mean([]), null);
  assert.equal(mean(undefined), null);
});

test("E1's marks come from the file, or the ones the crate measures against", () => {
  assert.deepEqual(e1Marks(e1), Object.entries(E1_TO_BEAT));
  assert.deepEqual(e1Marks({}), Object.entries(E1_TO_BEAT));
});

test("E1's summary is one row an arm, in the file's order, with what each beats", () => {
  const rows = e1Summary(e1);
  assert.deepEqual(
    rows.map((r) => r.arm),
    ["plain-oracle", "game-whole-run"],
  );
  assert.equal(rows[0].beats["explore-oracle"].encoder, true);
  assert.equal(rows[1].beats["own-whole-run"].lattice, false);
  assert.deepEqual(e1Summary(null), []);
});

test("E1's rows group by arm and language with the seeds beneath", () => {
  const groups = e1ByLanguage(e1);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].seeds.length, 2);
  assert.ok(Math.abs(groups[0].encoder - 0.95) < 1e-12);
  assert.ok(Math.abs(groups[0].lattice - 0.9) < 1e-12);
  assert.deepEqual(e1Axes(e1), { arms: ["plain-oracle", "game-whole-run"], languages: ["even-b", "mod3-a"] });
});

test("E2 groups by window, ascending, with the mean of every measure", () => {
  const w = e2ByWindow(e2);
  assert.deepEqual(
    w.map((r) => [r.window, r.seeds]),
    [
      [1, 1],
      [4, 2],
    ],
  );
  assert.ok(Math.abs(w[1].legal_first_try - 0.25) < 1e-12);
  assert.equal(w[1].states_used, 12);
  assert.equal(w[0].optimal_rate, null, "a measure the rows do not carry is null");
  assert.ok(E2_COLUMNS.every(([key]) => key in w[0]));
});

test("the files the experiments wrote shape without loss", () => {
  const dir = new URL("../../results/", import.meta.url);
  const e1Path = new URL("e1.json", dir);
  const e2Path = new URL("e2.json", dir);
  if (existsSync(e1Path)) {
    const doc = JSON.parse(readFileSync(e1Path, "utf8"));
    const summary = e1Summary(doc);
    assert.equal(summary.length, Object.keys(doc.summary).length);
    const groups = e1ByLanguage(doc);
    assert.equal(
      groups.reduce((n, g) => n + g.seeds.length, 0),
      doc.rows.length,
    );
    const { arms, languages } = e1Axes(doc);
    assert.equal(groups.length, arms.length * languages.length);
  }
  if (existsSync(e2Path)) {
    const doc = JSON.parse(readFileSync(e2Path, "utf8"));
    const windows = e2ByWindow(doc);
    assert.equal(
      windows.reduce((n, w) => n + w.seeds, 0),
      doc.rows.length,
    );
  }
});
