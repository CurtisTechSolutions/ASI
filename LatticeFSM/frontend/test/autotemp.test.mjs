import assert from "node:assert/strict";
import test from "node:test";
import { MAX_TEMPERATURE, autoBody, autoLabel } from "../src/lm.js";

test("auto-temperature in a request: on with its top rung, or nothing", () => {
  assert.deepEqual(autoBody(false, 0.5), {});
  assert.deepEqual(autoBody(true, 0.5), { auto_temperature: true, max_temperature: 0.5 });
  assert.deepEqual(autoBody(true, "x"), { auto_temperature: true, max_temperature: MAX_TEMPERATURE });
  assert.deepEqual(autoBody(true, -1), { auto_temperature: true, max_temperature: MAX_TEMPERATURE });
});

test("how a reply was found, in a few words: nothing to say when the first draw was an answer", () => {
  assert.equal(autoLabel(null), "");
  assert.equal(autoLabel({ found: true, temperature: 0, attempts: [{ temperature: 0, why: null }] }), "");
  const climbed = {
    found: true,
    temperature: 0.4,
    attempts: [
      { temperature: 0, why: "nothing" },
      { temperature: 0.2, why: "no words" },
      { temperature: 0.4, why: null },
    ],
  };
  assert.equal(autoLabel(climbed), "at 0.4 (0: nothing)");
  const missed = {
    found: false,
    temperature: 0.2,
    why: "a loop",
    attempts: [
      { temperature: 0, why: "nothing" },
      { temperature: 0.2, why: "a loop" },
      { temperature: 0.8, why: "nothing" },
    ],
  };
  assert.equal(autoLabel(missed), "no answer up to 0.8: a loop");
  // floating point rungs read as they were meant
  assert.equal(autoLabel({ ...climbed, temperature: 0.6000000000000001 }), "at 0.6 (0: nothing)");
});
