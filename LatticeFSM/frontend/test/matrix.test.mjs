/** Tests for the pure helpers in `src/matrix.js` - plain `node --test`. Run with `npm test`. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  cellStyle,
  curvePoints,
  describeRun,
  fmtSize,
  isCenter,
  shade,
  stimulationFrom,
  tableRows,
} from "../src/matrix.js";

test("shade maps a probability onto 0..85 and clips", () => {
  assert.equal(shade(0), 0);
  assert.equal(shade(1), 85);
  assert.equal(shade(0.5), 43);
  assert.equal(shade(2), 85);
  assert.equal(shade(-1), 0);
  assert.equal(shade("x"), 0);
  assert.match(cellStyle(0.5).background, /43%/);
});

test("a run is written out step by step, and the empty run says so", () => {
  const run = {
    transitions: [
      { source: 0, symbol: "a", target: 2, probability: 0.25 },
      { source: 2, symbol: "b", target: 1, probability: 0.5 },
    ],
  };
  assert.equal(describeRun(run), "0 -a(0.25)-> 2  2 -b(0.50)-> 1");
  assert.equal(describeRun({ transitions: [] }), "(empty)");
  assert.equal(describeRun(null), "(empty)");
});

test("a curve becomes chart points and skips what is not a pair", () => {
  assert.deepEqual(curvePoints([[100, 0.5], [200, 1], "x", [3]]), [
    { x: 100, y: 0.5 },
    { x: 200, y: 1 },
  ]);
  assert.deepEqual(curvePoints(undefined), []);
});

test("the greedy table is rows of text with accepting states starred", () => {
  const rows = tableRows({ alphabet: ["a", "b"], accepting: [0], table: [[0, 3], [1, 0]] });
  assert.deepEqual(rows, [
    { state: 0, accepting: true, cells: ["a -> 0", "b -> 3"] },
    { state: 1, accepting: false, cells: ["a -> 1", "b -> 0"] },
  ]);
  assert.deepEqual(tableRows(null), []);
});

test("sizes read as bytes, kilobytes and megabytes", () => {
  assert.equal(fmtSize(275), "275 B");
  assert.equal(fmtSize(48947), "47.8 KB");
  assert.equal(fmtSize(369096), "360.4 KB");
  assert.equal(fmtSize(3 * 1024 * 1024), "3.00 MB");
  assert.equal(fmtSize(undefined), "–");
});

test("the central node is the cell the stats name", () => {
  assert.equal(isCenter([6, 6, 6], 6, 6, 6), true);
  assert.equal(isCenter([6, 6, 6], 6, 5, 6), false);
  assert.equal(isCenter(undefined, 6, 6, 6), false);
});

test("the stimulation field is a number or nothing", () => {
  assert.equal(stimulationFrom("2.5"), 2.5);
  assert.equal(stimulationFrom(""), undefined);
  assert.equal(stimulationFrom("-1"), undefined);
  assert.equal(stimulationFrom("abc"), undefined);
});
