/** Tests for the Lattice tab's and the chart's helpers (`src/matrix.js`) and the Nim tab's (`src/nim.js`). */

import assert from "node:assert/strict";
import { test } from "node:test";

import { cellStyle, curveSeries, shade, tableRows } from "../src/matrix.js";
import { NIM_TILES, nimTiles, recordLine } from "../src/nim.js";

test("shade maps a probability onto 0..85 and clips", () => {
  assert.equal(shade(0), 0);
  assert.equal(shade(1), 85);
  assert.equal(shade(0.5), 43);
  assert.equal(shade(2), 85);
  assert.equal(shade(-1), 0);
  assert.equal(shade("x"), 0);
  assert.match(cellStyle(1).background, /85%/);
});

test("the greedy table is rows with the start and the accepting states marked", () => {
  const rows = tableRows({ alphabet: ["a", "b"], accepting: [0], start: 0, table: [[0, 1], [1, 0]] });
  assert.equal(rows.length, 2);
  assert.equal(rows[0].accepting, true);
  assert.equal(rows[0].start, true);
  assert.equal(rows[1].start, false);
  assert.deepEqual(rows[1].cells, [
    { symbol: "a", target: 1 },
    { symbol: "b", target: 0 },
  ]);
  assert.deepEqual(tableRows(null), []);
});

test("a held-out curve becomes the encoder's and the lattice's series", () => {
  const [enc, lat] = curveSeries([[0, 0.5, 0.6], [100, 0.9, 0.8], ["x"], [200, 1, 1]]);
  assert.deepEqual(
    enc.values.map((p) => [p.x, p.y]),
    [
      [0, 0.5],
      [100, 0.9],
      [200, 1],
    ],
  );
  assert.deepEqual(
    lat.values.map((p) => p.y),
    [0.6, 0.8, 1],
  );
  assert.equal(lat.dashed, true);
  assert.equal(curveSeries(null)[0].values.length, 0);
});

test("Nim's totals become tiles, rates as percentages", () => {
  const tiles = nimTiles({
    totals: { win_rate: 0.5, legal_first_try: 0.125, refusals_per_move: 2, optimal_rate: 0.4, abstained_per_move: 0.1 },
    information_bits: 0.0213,
    states_used: 13,
    agreement: 0.88,
  });
  assert.equal(tiles.length, NIM_TILES.length + 3);
  const by = Object.fromEntries(tiles.map((t) => [t.key, t.value]));
  assert.equal(by.win_rate, "50.0%");
  assert.equal(by.legal_first_try, "12.5%");
  assert.equal(by.refusals_per_move, "2.00");
  assert.equal(by.information_bits, "0.021 bits");
  assert.equal(by.states_used, 13);
  assert.equal(by.agreement, "0.88");
  assert.equal(nimTiles(null)[0].value, "–");
});

test("a game reads in a line", () => {
  assert.equal(
    recordLine({ won: true, first: true, plies: 7, first_try: 1, our_moves: 4, refusals: 4, optimal: 1 }),
    "won, moved first, 7 plies: 1/4 legal at once, 4 refused, 1 optimal",
  );
  assert.equal(recordLine(null), "");
});
