import assert from "node:assert/strict";
import test from "node:test";
import { fmtElapsed, jobLine, jobName } from "../src/jobs.js";

test("elapsed times read as seconds, minutes or hours", () => {
  assert.equal(fmtElapsed(6.54), "6.5 s");
  assert.equal(fmtElapsed(125), "2 min 05 s");
  assert.equal(fmtElapsed(3725), "1 h 02 min");
  assert.equal(fmtElapsed(undefined), "0.0 s");
});

test("a job in a line: what, on what, how far, how fast", () => {
  const job = {
    id: 3,
    model: "chain",
    kind: "train",
    state: "running",
    elapsed: 6.5,
    progress: { phase: "training", label: "a.txt", sequences: 1200, tokens: 340000, tokens_per_second: 52000, round: 0, stopping: false },
  };
  assert.equal(jobName(job), "job 3: chain train");
  assert.equal(
    jobLine(job),
    `job 3: chain train - training a.txt · ${(1200).toLocaleString()} sequences · ${(340000).toLocaleString()} tokens (${(52000).toLocaleString()}/s) · 6.5 s`,
  );
  const stopping = { ...job, progress: { ...job.progress, round: 2, stopping: true } };
  assert.match(jobLine(stopping), /round 2 · 6\.5 s · stopping…$/);
  assert.equal(jobLine(null), "");
  assert.equal(jobLine({ id: 1, kind: "sweep", elapsed: 0 }), "job 1: sweep - 0.0 s");
});
