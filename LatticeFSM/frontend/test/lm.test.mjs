/** Tests for the language-model helpers (`src/lm.js`), the speech helpers and the new util formatters. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  bestSweepRow,
  chainBoxes,
  clampMark,
  contextText,
  depthShare,
  historyPoints,
  joinContinuation,
  lmId,
  mean,
  probShare,
  probStyle,
  ringLayout,
  roundsNote,
  splitByMark,
  sweepSeries,
  tokenPieces,
  untilNote,
  goalShare,
} from "../src/lm.js";
import { recognitionText, speakable, speechSupport, spokenText } from "../src/speech.js";
import { cleanName, fmtAgo, fmtTime } from "../src/util.js";

test("lmId keeps a known model and makes anything else the chain", () => {
  assert.equal(lmId("ngram"), "ngram");
  assert.equal(lmId("chain"), "chain");
  assert.equal(lmId("gpt"), "chain");
  assert.equal(lmId(undefined), "chain");
});

test("splitByMark: 6 or more good, 4 or less bad, 5 left out, each text once", () => {
  const out = splitByMark([
    { text: "a", mark: 10 },
    { text: "b", mark: 6 },
    { text: "c", mark: 5 },
    { text: "d", mark: 4 },
    { text: "e", mark: 0 },
    { text: "a", mark: 0 },
    { text: "  ", mark: 9 },
    { text: "f", mark: "" },
  ]);
  assert.deepEqual(out.good, ["a", "b"]);
  assert.deepEqual(out.bad, ["d", "e"]);
  assert.equal(out.skipped, 2); // c, and f's blank mark falls back to 5
  assert.deepEqual(splitByMark([{ text: "x", mark: 7 }], { good: 8, bad: 7 }), { good: [], bad: ["x"], skipped: 0 });
  assert.deepEqual(splitByMark(null), { good: [], bad: [], skipped: 0 });
});

test("clampMark rounds into 0..10", () => {
  assert.equal(clampMark(11), 10);
  assert.equal(clampMark(-2), 0);
  assert.equal(clampMark(6.6), 7);
  assert.equal(clampMark("x"), 5);
  assert.equal(clampMark("", 3), 3);
});

test("probShare tints on a log scale, 1 full and 1e-4 or less none", () => {
  assert.equal(probShare(1), 85);
  assert.equal(probShare(0.1), 64);
  assert.equal(probShare(0.01), 43);
  assert.equal(probShare(1e-4), 0);
  assert.equal(probShare(1e-9), 0);
  assert.equal(probShare(0), 0);
  assert.equal(probShare("x"), 0);
  assert.equal(probShare(2), 85);
  assert.match(probStyle(1).background, /85%/);
});

test("depthShare tints by the depth against the deepest", () => {
  assert.equal(depthShare(0, 7), 0);
  assert.equal(depthShare(7, 7), 85);
  assert.equal(depthShare(14, 7), 85);
  assert.equal(depthShare(1, 0), 0);
  assert.equal(depthShare(null, 7), 0);
});

test("joinContinuation per unit", () => {
  assert.equal(joinContinuation("the cat", "sat down", "word"), "the cat sat down");
  assert.equal(joinContinuation("the cat ", " sat", "word"), "the cat sat");
  assert.equal(joinContinuation("", "sat", "word"), "sat");
  assert.equal(joinContinuation("the ca", "t sat", "char"), "the cat sat");
  assert.equal(joinContinuation("aGk", "aGkgdGhlcmU", "base64"), "aGkgdGhlcmU");
});

test("tokenPieces spaces words, keeps letters, and carries depths", () => {
  assert.deepEqual(
    tokenPieces(["the", "cat", "."], [1, 2, 3], "word").map((p) => [p.text, p.depth]),
    [
      ["the", 1],
      [" cat", 2],
      [".", 3],
    ],
  );
  assert.deepEqual(
    tokenPieces(["a", "_", "b"], null, "char").map((p) => p.text),
    ["a", " ", "b"],
  );
  assert.equal(tokenPieces(["x", "</s>"], [0], "word")[1].text, " ⏎");
  assert.equal(tokenPieces(["x", "</s>"], [0], "word")[1].depth, null);
});

test("mean ignores what is not a number", () => {
  assert.equal(mean([1, 2, 3]), 2);
  assert.equal(mean([1, null, "x", 3]), 2);
  assert.equal(mean([]), null);
  assert.equal(mean(undefined), null);
});

test("chainBoxes maps the walk's depths onto the input nodes", () => {
  const boxes = chainBoxes({
    nodes: 4,
    inputs: ["a", "b", null],
    walk: [
      { depth: 0, confidence: 0.2, gathered: 0.2 },
      { depth: 1, input: null, confidence: 0.5, gathered: 0.6 },
    ],
  });
  assert.equal(boxes.length, 4);
  assert.deepEqual(
    boxes.map((b) => [b.node, b.depth, b.input, b.output, Boolean(b.step)]),
    [
      [1, 3, "a", false, false],
      [2, 2, "b", false, false],
      [3, 1, null, false, true],
      [4, 0, null, true, true],
    ],
  );
  // without `nodes`, the inputs say how many there are
  assert.equal(chainBoxes({ inputs: ["x"] }).length, 2);
  assert.deepEqual(chainBoxes(null).map((b) => b.output), [true]);
});

test("historyPoints and sweepSeries make chart points", () => {
  assert.deepEqual(historyPoints([{ round: 0, perplexity: 10 }, { round: 1 }, { round: 2, perplexity: 8 }], "perplexity"), [
    { x: 0, y: 10 },
    { x: 2, y: 8 },
  ]);
  const rows = [
    { confidence: 0, accuracy: 0.2, depth: 0, perplexity: 30 },
    { confidence: 0.9, accuracy: 0.4, depth: 3.5, perplexity: 12 },
    { confidence: 1, accuracy: 0.4, depth: 6, perplexity: 13 },
  ];
  const s = sweepSeries(rows);
  assert.deepEqual(s.accuracy.map((p) => p.x), [0, 0.9, 1]);
  assert.deepEqual(s.depth.map((p) => p.y), [0, 3.5, 6]);
  assert.equal(bestSweepRow(rows), rows[1]); // the shallower of the two best
  assert.equal(bestSweepRow([]), null);
});

test("untilNote and roundsNote say how teaching until N of N ended", () => {
  assert.equal(untilNote({ right: 33, of: 33, rounds: 7, reached: true }), "taught in 7 rounds: 33 of 33 right");
  assert.equal(untilNote({ right: 30, of: 33, rounds: 200, reached: false }), "stopped after 200 rounds at 30 of 33");
  assert.equal(untilNote({ right: 1, of: 1, rounds: 1, reached: true }), "taught in 1 round: 1 of 1 right");
  assert.equal(untilNote({ right: 12, of: 33 }), "12 of 33 right");
  assert.equal(roundsNote([{ round: 1, right: 20, of: 33 }, { round: 2, right: 33, of: 33 }], true), "33 of 33 reached in 2 rounds");
  assert.equal(
    roundsNote([{ round: 1, right: 20, of: 33 }, { round: 2, right: 30, of: 33 }], true, 29, 31),
    "30 of 33 reached in 2 rounds (the goal: 29 of the 31 possible)",
  );
  assert.equal(untilNote({ right: 41, of: 43, need: 41, rounds: 8, reached: true }), "taught in 8 rounds: 41 of 43 right (the goal 41)");
  assert.equal(goalShare("95"), 0.95);
  assert.equal(goalShare("150"), 1);
  assert.equal(goalShare("x"), 0.95);
  assert.equal(roundsNote([{ round: 1, right: 20, of: 33 }], false), "stopped after 1 round at 20 of 33");
  assert.equal(roundsNote([], true), "");
});

test("ringLayout puts each depth on its own ring", () => {
  const placed = ringLayout(
    [
      { depth: 1, seen: 10 },
      { depth: 1, seen: 5 },
      { depth: 3, seen: 1 },
    ],
    200,
  );
  assert.equal(placed.length, 3);
  const radius = (p) => Math.hypot(p.x - 100, p.y - 100);
  assert.ok(Math.abs(radius(placed[0]) - radius(placed[1])) < 1e-9);
  assert.ok(radius(placed[2]) > radius(placed[0]));
  assert.ok(placed[0].r > placed[2].r);
  assert.deepEqual(ringLayout([]), []);
});

test("contextText per unit", () => {
  assert.equal(contextText(["a", "_", "b"], "char"), "a␣b");
  assert.equal(contextText(["the", "cat"], "word"), "the cat");
  assert.equal(contextText(null, "word"), "");
});

test("speech helpers", () => {
  assert.deepEqual(speechSupport({}), { synthesis: false, Recognition: null });
  function Rec() {}
  function Utterance() {}
  const s = speechSupport({ webkitSpeechRecognition: Rec, speechSynthesis: {}, SpeechSynthesisUtterance: Utterance });
  assert.equal(s.Recognition, Rec);
  assert.equal(s.synthesis, true);
  assert.equal(speakable("hello", "word"), true);
  assert.equal(speakable("...", "word"), false);
  assert.equal(speakable("aGVsbG8", "base64"), false);
  assert.equal(spokenText("the_cat  sat</s>"), "the cat sat");
  const event = {
    resultIndex: 1,
    results: [
      Object.assign([{ transcript: "old" }], { isFinal: true }),
      Object.assign([{ transcript: "hello there " }], { isFinal: true }),
      Object.assign([{ transcript: "how" }], { isFinal: false }),
    ],
  };
  assert.deepEqual(recognitionText(event), { final: "hello there", interim: "how" });
});

test("fmtTime, fmtAgo, cleanName", () => {
  assert.equal(fmtTime(null), "–");
  assert.equal(fmtTime(0), new Date(0).toLocaleString());
  assert.equal(fmtTime("nope"), "nope");
  const now = 1_000_000_000_000;
  assert.equal(fmtAgo(now / 1000 - 5, now), "just now");
  assert.equal(fmtAgo(now / 1000 - 300, now), "5 min ago");
  assert.equal(fmtAgo(now / 1000 - 7200, now), "2 h ago");
  assert.equal(fmtAgo(now / 1000 - 3 * 86400, now), "3 d ago");
  assert.equal(fmtAgo("x", now), "–");
  assert.equal(cleanName(" dir/sub/chain.json.gz "), "chain.json.gz");
  assert.equal(cleanName("C:\\x\\y.gz"), "y.gz");
});
