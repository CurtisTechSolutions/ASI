/**
 * Tests for the dynamic window's display helpers (`src/window.js`) - plain `node --test`. The ladder is
 * checked against the sizes radixnet.window.ladder, the Go Ladder and the Rust ladder produce. Run with
 * `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_FLOOR,
  DEFAULT_TOP,
  SIZES,
  clampSize,
  describeStep,
  describeWindow,
  isPowerOfTwo,
  ladder,
  nextSize,
} from "../src/window.js";

test("a size is a power of two", () => {
  assert.deepEqual(
    Array.from({ length: 70 }, (_, i) => i).filter(isPowerOfTwo),
    [1, 2, 4, 8, 16, 32, 64],
  );
  assert.equal(isPowerOfTwo("16"), true);
  assert.equal(isPowerOfTwo(2.5), false);
  assert.ok(SIZES.every(isPowerOfTwo));
});

test("the ladder halves from the top to the floor", () => {
  assert.deepEqual(ladder(DEFAULT_TOP, DEFAULT_FLOOR), [32, 16, 8, 4]);
  assert.deepEqual(ladder(8, 8), [8]);
  assert.deepEqual(ladder("64", "1"), [64, 32, 16, 8, 4, 2, 1]);
  assert.deepEqual(ladder(20, 4), []);
  assert.deepEqual(ladder(4, 32), []);
});

test("the next size goes down the ladder and back to the top", () => {
  assert.equal(nextSize(32, 32, 4), 16);
  assert.equal(nextSize(4, 32, 4), 32);
  assert.equal(nextSize(8, 8, 8), 8);
  assert.equal(nextSize(12, 32, 4), null);
});

test("a size is kept on the ladder the way the servers keep it", () => {
  assert.equal(clampSize(8, 32, 4), 8);
  assert.equal(clampSize(64, 32, 4), 32);
  assert.equal(clampSize(2, 32, 4), 4);
  assert.equal(clampSize(12, 32, 4), 8);
  assert.equal(clampSize("nonsense", 32, 4), 32);
  assert.equal(clampSize(8, 20, 4), null);
});

test("the window described the way the card says it", () => {
  assert.equal(describeWindow(null), "unknown");
  assert.match(describeWindow({ on: false }), /^off/);
  assert.equal(
    describeWindow({ on: true, sizes: [32, 16, 8, 4], size: 16, next: 8, auto: true, units: "chars" }),
    "on: 32 → 16 → 8 → 4 chars, at 16 (next 8), stepping at the end of every training epoch",
  );
  assert.match(describeWindow({ on: true, sizes: [8], size: 8, next: 8, auto: false, units: "words" }), /by hand$/);
});

test("a step described in one line", () => {
  assert.equal(describeStep(null), "");
  assert.equal(
    describeStep({ steps: 2, sizes: [32, 16], from: 32, to: 8, merges: 0, splits: 3, nodes_before: 29, nodes_after: 32, edges_before: 40, edges_after: 43 }),
    "2 steps at 32, 16: 0 merges, 3 splits, nodes 29 → 32, edges 40 → 43; the window now stands at 8",
  );
  assert.match(describeStep({ steps: 1, sizes: [4], from: 4, to: 32, merges: 1, splits: 1, nodes_before: 1, nodes_after: 2, edges_before: 1, edges_after: 2 }), /^1 step at 4: 1 merge, 1 split/);
});
