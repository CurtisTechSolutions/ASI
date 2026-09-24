/**
 * Tests for the site-wide settings (`src/settings.js`) - plain `node --test`:
 * the module is pure, so the rules the Settings, Predict, Generate and Train
 * tabs share are checked here number for number against
 * ../../SPEC-SearchAndTraining.md. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ENCODING_PRESETS,
  ORDERS,
  SEARCH_DEFAULTS,
  TRAINING_DEFAULTS,
  describeEncoding,
  encodingSpec,
  pacedCounts,
  problemText,
  rehearsalCount,
  replayLine,
  searchActive,
  searchBody,
  searchProblems,
  searchProblemsFor,
  searchSettingsFor,
  trainingActive,
  trainingBody,
  trainingProblems,
} from "../src/settings.js";

test("every setting is off at its default and sends nothing", () => {
  for (const mode of ["sample", "beam", "dijkstra", "kbest"]) assert.deepEqual(searchBody(SEARCH_DEFAULTS, mode), {});
  assert.deepEqual(trainingBody(TRAINING_DEFAULTS), {});
  assert.deepEqual(searchProblems(SEARCH_DEFAULTS), {});
  assert.deepEqual(trainingProblems(TRAINING_DEFAULTS), {});
  assert.equal(searchActive(SEARCH_DEFAULTS), false);
  assert.equal(trainingActive(TRAINING_DEFAULTS), false);
  assert.deepEqual(searchBody(undefined, "sample"), {});
  assert.deepEqual(trainingBody(undefined), {});
});

test("the filters go with a sampled walk and the diversity with a beam", () => {
  const values = { topK: "3", topP: "0.9", minP: "0.05", diversity: "2" };
  assert.deepEqual(searchBody(values, "sample"), { top_k: 3, top_p: 0.9, min_p: 0.05 });
  assert.deepEqual(searchBody(values, "beam"), { diversity: 2 });
  assert.deepEqual(searchBody(values, "dijkstra"), {}); // exact searches read neither
  assert.deepEqual(searchBody(values, "kbest"), {});
  assert.deepEqual(searchSettingsFor("sample"), { filters: true, diversity: false });
  assert.deepEqual(searchSettingsFor("beam"), { filters: false, diversity: true });
  assert.equal(searchActive(values), true);
  assert.equal(searchActive({ ...SEARCH_DEFAULTS, diversity: "0.5" }), true);
});

test("only what is not off is sent", () => {
  assert.deepEqual(searchBody({ topK: "0", topP: "1", minP: "0.2" }, "sample"), { min_p: 0.2 });
  assert.deepEqual(searchBody({ topP: "0.5" }, "sample"), { top_p: 0.5 });
  assert.deepEqual(searchBody({ diversity: "0" }, "beam"), {});
});

test("search settings out of range are named and never sent", () => {
  const bad = { topK: "-1", topP: "0", minP: "1", diversity: "-0.5" };
  assert.deepEqual(Object.keys(searchProblems(bad)).sort(), ["diversity", "minP", "topK", "topP"]);
  assert.deepEqual(searchBody(bad, "sample"), {});
  assert.deepEqual(searchBody(bad, "beam"), {});
  assert.ok(searchProblems({ topK: "2.5" }).topK, "top-K is a whole number");
  assert.ok(searchProblems({ topP: "1.5" }).topP);
  assert.ok(searchProblems({ topP: "" }).topP, "a cleared field is not a value");
  assert.ok(searchProblems({ minP: "abc" }).minP);
  assert.deepEqual(searchProblems({ topP: "1", minP: "0.999", topK: "7", diversity: "10" }), {});
  // a tab refuses only what its mode reads
  assert.deepEqual(Object.keys(searchProblemsFor(bad, "beam")), ["diversity"]);
  assert.deepEqual(Object.keys(searchProblemsFor(bad, "sample")).sort(), ["minP", "topK", "topP"]);
  assert.deepEqual(searchProblemsFor(bad, "dijkstra"), {});
  assert.equal(problemText({}), null);
  assert.match(problemText(searchProblems({ topP: "0" })), /Top-p must lie in \(0, 1\]/);
});

test("the training settings travel under the server's names", () => {
  const values = { order: "shortest-first", curriculum: "0.5", replay: "0.25", replaySize: "256", patience: "3", minDelta: "0.01" };
  assert.deepEqual(trainingBody(values), {
    order: "shortest-first",
    curriculum: 0.5,
    replay: 0.25,
    replay_size: 256,
    patience: 3,
    min_delta: 0.01,
  });
  assert.equal(trainingActive(values), true);
  // a size of 0 is sent - it drops the buffer - and a blank one is not: it leaves the buffer alone
  assert.deepEqual(trainingBody({ replaySize: "0" }), { replay_size: 0 });
  assert.deepEqual(trainingBody({ replaySize: " " }), {});
  // the minimum improvement means nothing without patience
  assert.deepEqual(trainingBody({ minDelta: "0.5" }), {});
  assert.deepEqual(trainingBody({ patience: "2", minDelta: "0" }), { patience: 2 });
  assert.deepEqual(ORDERS.map(([name]) => name), ["corpus", "shortest-first", "longest-first", "shuffle"]);
});

test("training settings out of range are named and never sent", () => {
  const bad = { order: "random", curriculum: "0", replay: "-1", replaySize: "1.5", patience: "-2", minDelta: "-1" };
  assert.deepEqual(Object.keys(trainingProblems(bad)).sort(), [
    "curriculum",
    "minDelta",
    "order",
    "patience",
    "replay",
    "replaySize",
  ]);
  assert.deepEqual(trainingBody(bad), {});
  assert.ok(trainingProblems({ curriculum: "1.5" }).curriculum);
  assert.deepEqual(trainingProblems({ curriculum: "1", replay: "3", replaySize: "", patience: "0" }), {});
});

test("the curriculum preview follows the servers' rule", () => {
  // SPEC-SearchAndTraining.md section 3: ceil(frac_j * n), clamped to [1, n]
  assert.deepEqual(pacedCounts(10, 0.3, 4), [3, 6, 8, 10]);
  assert.deepEqual(pacedCounts(20, 0.9, 6), [18, 19, 19, 20, 20, 20]); // the spec's own example
  assert.deepEqual(pacedCounts(6, 0.5, 4), [3, 4, 5, 6]);
  assert.deepEqual(pacedCounts(10, 0.01, 3), [1, 6, 10]);
  assert.deepEqual(pacedCounts(10, 1, 3), [10, 10, 10]);
  assert.deepEqual(pacedCounts(10, 0.5, 1), [10]); // one epoch walks the whole list
  assert.deepEqual(pacedCounts(0, 0.5, 2), [0, 0]);
  assert.deepEqual(pacedCounts(5, 0.5, 0), []);
});

test("the rehearsal preview rounds half up and stops at the buffer", () => {
  assert.equal(rehearsalCount(0.25, 10, 100), 3);
  assert.equal(rehearsalCount(0.24, 10, 100), 2);
  assert.equal(rehearsalCount(5, 10, 7), 7);
  assert.equal(rehearsalCount(0, 10, 7), 0);
  assert.equal(rehearsalCount(1, 10, 0), 0);
  assert.equal(rehearsalCount("0.5", 6, 5), 3);
});

test("a new model's encoding is checked before it is sent", () => {
  assert.deepEqual(encodingSpec("word", "3", "1"), { spec: "word:3:1" });
  assert.deepEqual(encodingSpec("char", 5, 5), { spec: "char:5:5" });
  assert.ok(encodingSpec("rune", "3", "1").error);
  assert.ok(encodingSpec("char", "0", "1").error);
  assert.ok(encodingSpec("char", "2", "3").error, "a stride longer than the gram skips text");
  assert.ok(encodingSpec("char", "3", "0").error);
  assert.ok(encodingSpec("char", "2.5", "1").error);
  for (const [spec] of ENCODING_PRESETS.filter(([name]) => name !== "custom")) {
    const [unit, n, stride] = spec.split(":");
    assert.deepEqual(encodingSpec(unit, n, stride), { spec }, spec);
  }
});

test("an encoding reads as words", () => {
  assert.equal(describeEncoding("char:3:1"), "character trigram");
  assert.equal(describeEncoding("word:2:1"), "word bigram");
  assert.equal(describeEncoding("word:3:1"), "word trigram");
  assert.equal(describeEncoding("char:5:5"), "groups of 5 characters");
  assert.equal(describeEncoding("char:5:1"), "5-grams of characters");
  assert.equal(describeEncoding("word:4:2"), "4-grams of words, stride 2");
  assert.equal(describeEncoding("nonsense"), "nonsense");
  assert.equal(describeEncoding(undefined), "");
});

test("the replay buffer in one line", () => {
  assert.equal(replayLine(null), "This model keeps no replay buffer.");
  assert.equal(replayLine(undefined), "This model keeps no replay buffer.");
  const line = replayLine({ size: 256, texts: 200, seen: 1234 });
  assert.match(line, /replay buffer of 256 texts/);
  assert.match(line, /200 held/);
  assert.match(line, /1,234 it was ever trained on/);
});
