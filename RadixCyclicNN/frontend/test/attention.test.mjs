/**
 * Tests for the attention band's display helpers (`src/attention.js`) - plain `node --test`.
 * The band is checked against the same numbers radixnet.attention.band_weights, the Go BandWeights and
 * the Rust band_weights are pinned to, down to the double. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_BLUR,
  bandWeights,
  chargedRows,
  clampBlur,
  describeBand,
  fmtShare,
  gramUnits,
  unitStyle,
} from "../src/attention.js";

test("the band peaks at the centre and falls to 1 - blur at the ends", () => {
  assert.deepEqual(bandWeights(3, 0.5), [0.5, 1, 0.5]);
  assert.deepEqual(bandWeights(5, 0.5), [0.5, 0.75, 1, 0.75, 0.5]);
  assert.deepEqual(bandWeights(4, 0.5), [0.5, 0.8333333333333334, 0.8333333333333334, 0.5]); // Python's doubles
  assert.deepEqual(bandWeights(3, 0.3), [0.7, 1, 0.7]);
  assert.deepEqual(bandWeights(7, 1), [0, 0.33333333333333337, 0.6666666666666667, 1, 0.6666666666666667, 0.33333333333333337, 0]);
});

test("a gram without a centre is seen evenly", () => {
  assert.deepEqual(bandWeights(1, 1), [1]);
  assert.deepEqual(bandWeights(2, 0.5), [0.5, 0.5]);
  assert.deepEqual(bandWeights(5, 0), [1, 1, 1, 1, 1]);
});

test("a blur is a number in [0, 1]", () => {
  assert.equal(clampBlur("0.25"), 0.25);
  assert.equal(clampBlur(3), 1);
  assert.equal(clampBlur(-1), 0);
  assert.equal(clampBlur(""), DEFAULT_BLUR);
  assert.equal(clampBlur("sharp", 0.1), 0.1);
  assert.deepEqual(bandWeights(3, "nonsense"), [0.5, 1, 0.5]); // the default blur
});

test("a unit is drawn as sharply as the band sees it", () => {
  assert.deepEqual(unitStyle(1), { filter: "none", opacity: 1 });
  assert.deepEqual(unitStyle(0.5), { filter: "blur(1.20px)", opacity: 0.675 });
  assert.deepEqual(unitStyle(0), { filter: "blur(2.40px)", opacity: 0.35 });
  assert.deepEqual(gramUnits("cat", "char"), ["c", "a", "t"]);
  assert.deepEqual(gramUnits("the cat sat", "word"), ["the", "cat", "sat"]);
});

test("the band described the way the card says it", () => {
  assert.equal(describeBand(null), "unknown");
  assert.equal(describeBand({ applies: false }), "not used: this kind is never corrected");
  assert.match(describeBand({ applies: true, on: false }), /^off/);
  assert.equal(
    describeBand({ applies: true, on: true, blur: 0.5, weights: [0.5, 1, 0.5] }),
    "on, blur 0.5 - the centre of each gram is charged most (0.5 1 0.5)",
  );
  assert.equal(fmtShare(0.833333), "0.83");
  assert.equal(fmtShare(undefined), "–");
});

test("a preview's rows are the grams either rule charges, and the end", () => {
  // what POST /api/model/attention/preview answers for "the cat sat" -> "the bat sat" at blur 0.5
  const side = {
    grams: ["the", "he ", "e c", " ca", "cat", "at ", "t s", " sa", "sat"],
    writer: [false, false, true, false, false, false, false, false, false],
    charges: [0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0],
    focus: [false, false, false, true, false, false, false, false, false],
    end: false,
  };
  assert.deepEqual(chargedRows(side), [
    { index: 2, gram: "e c", writer: true, charge: 0.25, focus: false },
    { index: 3, gram: " ca", writer: false, charge: 0.5, focus: true },
    { index: 4, gram: "cat", writer: false, charge: 0.25, focus: false },
  ]);
  const stopped = { grams: ["the"], writer: [false], charges: [0], focus: [false], end: true };
  assert.deepEqual(chargedRows(stopped), [{ index: "end", gram: null, writer: true, charge: 1, focus: true }]);
  assert.deepEqual(chargedRows(null), []);
});
