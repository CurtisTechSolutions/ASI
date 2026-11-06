import assert from "node:assert/strict";
import test from "node:test";
import { fmtElapsed, jobLine, jobName, jobUnits } from "../src/jobs.js";

test("elapsed times read as seconds, minutes or hours", () => {
  assert.equal(fmtElapsed(6.54), "6.5 s");
  assert.equal(fmtElapsed(125), "2 min 05 s");
  assert.equal(fmtElapsed(3725), "1 h 02 min");
  assert.equal(fmtElapsed(undefined), "0.0 s");
});

test("a training job counts episodes and symbols, a Nim job games and plies", () => {
  assert.deepEqual(jobUnits({ model: "body", kind: "train" }), ["episodes", "symbols"]);
  assert.deepEqual(jobUnits({ model: "nim", kind: "play" }), ["games", "plies"]);
  assert.deepEqual(jobUnits(null), ["episodes", "symbols"]);
});

test("a job in a line: what, on what, how far, how fast", () => {
  const job = {
    id: 3,
    model: "body",
    kind: "train",
    state: "running",
    elapsed: 6.5,
    progress: { phase: "training", label: "even-b", sequences: 1200, tokens: 4000, tokens_per_second: 5200, round: 0, stopping: false },
  };
  assert.equal(jobName(job), "job 3: body train");
  assert.equal(
    jobLine(job),
    `job 3: body train - training even-b · ${(1200).toLocaleString()} episodes · ${(4000).toLocaleString()} symbols (${(5200).toLocaleString()}/s) · 6.5 s`,
  );
  const stopping = { ...job, progress: { ...job.progress, round: 2, stopping: true } };
  assert.match(jobLine(stopping), /tenth 2 · 6\.5 s · stopping…$/);
  assert.match(jobLine({ ...job, model: "nim", kind: "play" }), /1,?200 games · 4,?000 plies/);
  assert.equal(jobLine(null), "");
  assert.equal(jobLine({ id: 1, kind: "train", elapsed: 0 }), "job 1: train - 0.0 s");
});
