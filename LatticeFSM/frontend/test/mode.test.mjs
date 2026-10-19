import assert from "node:assert/strict";
import test from "node:test";
import { EXPLORE, MODES, modeLabel } from "../src/matrix.js";

test("a machine's traversal mode in a few words", () => {
  assert.deepEqual(MODES.map(([v]) => v), ["own", "explore"]);
  assert.equal(EXPLORE, Math.SQRT2);
  assert.equal(modeLabel(null), "own weighting");
  assert.equal(modeLabel({ mode: "own" }), "own weighting");
  assert.equal(modeLabel({ mode: "explore", explore: 0.5 }), "explore (bonus 0.50)");
  assert.equal(modeLabel({ mode: "explore" }), "explore (bonus 1.41)");
});
