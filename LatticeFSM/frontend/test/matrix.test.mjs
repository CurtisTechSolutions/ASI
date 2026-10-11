/** Tests for the pure helpers in `src/matrix.js` - plain `node --test`. Run with `npm test`. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  cellAt,
  cubeValue,
  project,
  cellStyle,
  curvePoints,
  describeRun,
  fmtSize,
  focusIndex,
  isCenter,
  onVector,
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

test("a skip is written as one move past the node it skips", () => {
  const run = { transitions: [{ source: 0, symbol: "ab", target: 5, probability: 0.31, skipped: 3 }] };
  assert.equal(describeRun(run), "0 =ab(0.31)=> 5 [skipping 3]");
});

test("focus picks a node of the central vertical vector, in equal bands", () => {
  assert.equal(focusIndex(13, null), 0);
  assert.equal(focusIndex(13, 0), 0);
  assert.equal(focusIndex(13, 0.5), 6);
  assert.equal(focusIndex(13, 1), 12);
  assert.equal(focusIndex(13, 1 / 13), 1);
  assert.equal(focusIndex(13, 0.92), 11);
  assert.equal(onVector([6, 6, 6], 6, 6), true);
  assert.equal(onVector([6, 6, 6], 5, 6), false);
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

test("cellAt reads a row-major cube index back as source, symbol, target", () => {
  assert.deepEqual(cellAt([13, 13, 13], 0), [0, 0, 0]);
  assert.deepEqual(cellAt([13, 13, 13], 6 * 169 + 6 * 13 + 6), [6, 6, 6]);
  assert.deepEqual(cellAt([3, 2, 3], 17), [2, 1, 2]);
});

test("cubeValue maps each measure onto 0..1, net onto -1..1", () => {
  assert.equal(cubeValue("probability", 0.25), 0.25);
  assert.equal(cubeValue("probability", 2), 1);
  assert.equal(cubeValue("seen", 0, 10), 0);
  assert.equal(cubeValue("seen", 10, 10), 1);
  assert.ok(Math.abs(cubeValue("width", 1) - 0.5) < 1e-12);
  assert.equal(cubeValue("width", 20), 1);
  assert.equal(cubeValue("net", -3), -1);
  assert.equal(cubeValue("probability", NaN), 0);
});

test("project turns and sees a point in perspective", () => {
  const [x, y, z, s] = project([1, 0, 0], 0, 0, 10, 100);
  assert.ok(Math.abs(x - 10) < 1e-9 && Math.abs(y) < 1e-9 && z === 0 && Math.abs(s - 10) < 1e-9);
  const [x2, , z2] = project([1, 0, 0], Math.PI / 2, 0, 10, 100);
  assert.ok(Math.abs(x2) < 1e-9 && Math.abs(z2 + 1) < 1e-9);
  const near = project([0, 0, 5], 0, 0, 10, 100)[3];
  const far = project([0, 0, -5], 0, 0, 10, 100)[3];
  assert.ok(near > far);
});
