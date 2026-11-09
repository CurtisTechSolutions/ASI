/** Tests for the naming trace's helpers (`src/trace.js`) - plain `node --test`. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BUCKETS,
  actionLabel,
  agreementOf,
  bestState,
  bucketTitle,
  clampStep,
  fmtAggregate,
  fmtSigned,
  phiKnown,
  seatsOf,
  splitBelief,
  stripBars,
  triggerText,
  verdictWord,
  walkLine,
} from "../src/trace.js";

/** A read as `POST /api/classify` answers it, cut down to what the helpers look at. */
const read = {
  text: "ab",
  member: false,
  trace: [
    {
      symbol: "a",
      prev: 0,
      named: 3,
      predicted: 3,
      aggregate: [0.1, 0.2, 0.05, 0.65],
      triggers: [],
      learned: false,
      seats: [
        { node: 4, bucket: "current", seated: true, action: 3, phi: null },
        { node: 5, bucket: "child", seated: true, action: "abstain", phi: null },
        { node: 6, bucket: "older", seated: false, action: null, phi: null },
      ],
    },
    { symbol: "b", prev: 3, named: 1, predicted: 2, aggregate: [0.3, 0.32, 0.3, 0.08], triggers: ["photo finish"], learned: false, seats: [] },
  ],
};

test("the buckets are the encoder's four, in its order", () => {
  assert.deepEqual(
    BUCKETS.map(([name]) => name),
    ["current", "previous", "older", "child"],
  );
  assert.match(bucketTitle("child"), /child/);
  assert.equal(bucketTitle("elsewhere"), "elsewhere");
});

test("the best state is the largest, ties to the lowest", () => {
  assert.equal(bestState([0.1, 0.4, 0.4, 0.1]), 1);
  assert.equal(bestState([0.5]), 0);
  assert.equal(bestState([]), -1);
  assert.equal(bestState(null), -1);
  assert.equal(bestState(read.trace[0].aggregate), read.trace[0].named);
});

test("an action reads as a state, abstain, or nothing for an unseated candidate", () => {
  assert.equal(actionLabel(3), "s3");
  assert.equal(actionLabel(0), "s0");
  assert.equal(actionLabel("abstain"), "abstain");
  assert.equal(actionLabel(null), "-");
  assert.equal(actionLabel(undefined), "-");
});

test("a strip scales to its largest bar and marks the named and the predicted state", () => {
  const bars = stripBars([0.1, 0.2, 0.05, 0.65], 3, 1);
  assert.equal(bars.length, 4);
  assert.equal(bars[3].height, 1);
  assert.ok(Math.abs(bars[1].height - 0.2 / 0.65) < 1e-12);
  assert.deepEqual(
    bars.map((b) => [b.named, b.predicted]),
    [
      [false, false],
      [false, true],
      [false, false],
      [true, false],
    ],
  );
  assert.deepEqual(
    stripBars([0, 0]).map((b) => b.height),
    [0, 0],
  );
  assert.deepEqual(stripBars(undefined), []);
});

test("a belief splits into the states and abstain", () => {
  assert.deepEqual(splitBelief([0.5, 0.3, 0.2]), { states: [0.5, 0.3], abstain: 0.2 });
  assert.deepEqual(splitBelief([]), { states: [], abstain: 0 });
});

test("an aggregate is written as its top states, highest first", () => {
  assert.equal(fmtAggregate([0.1, 0.2, 0.05, 0.65]), "3 0.65 · 1 0.20 · 0 0.10");
  assert.equal(fmtAggregate([0.5, 0.5], 1), "0 0.50");
  assert.equal(fmtAggregate(null), "");
});

test("triggers, seats, phi, the walk, the verdict and the agreement of a read", () => {
  assert.equal(triggerText(read.trace[1]), "photo finish");
  assert.equal(triggerText(read.trace[0]), "");
  const { seated, unseated } = seatsOf(read.trace[0]);
  assert.equal(seated.length, 2);
  assert.equal(unseated.length, 1);
  assert.equal(phiKnown(read), false);
  assert.equal(phiKnown({ trace: [{ ...read.trace[0], learned: true }] }), true);
  assert.equal(walkLine(read), "0 -a-> 3 -b-> 1 (lattice 2)");
  assert.equal(walkLine({ trace: [] }), "(empty)");
  assert.equal(verdictWord(false, read.member), "right");
  assert.equal(verdictWord(true, read.member), "wrong");
  assert.equal(verdictWord(true, undefined), "");
  assert.deepEqual(agreementOf(read), { agreed: 1, steps: 2 });
});

test("a step index stays inside the trace", () => {
  assert.equal(clampStep(5, 3), 2);
  assert.equal(clampStep(-1, 3), 0);
  assert.equal(clampStep("x", 3), 0);
  assert.equal(clampStep(1, 0), 0);
});

test("a signed number always carries its sign", () => {
  assert.equal(fmtSigned(0.12), "+0.120");
  assert.equal(fmtSigned(-0.034, 2), "−0.03");
  assert.equal(fmtSigned(null), "–");
});
