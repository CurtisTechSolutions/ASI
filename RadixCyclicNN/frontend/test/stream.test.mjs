import assert from "node:assert/strict";
import { test } from "node:test";

import { LineParser, STREAM_EVENTS, applyEvent } from "../src/stream.js";

test("lines are handed back whole, however the chunks fall", () => {
  const parser = new LineParser();
  assert.deepEqual(parser.feed('{"event":"look","from":"the cat"}\n{"event":"dra'), [{ event: "look", from: "the cat" }]);
  assert.deepEqual(parser.feed('ft","text":"the cat sat"}\n\n'), [{ event: "draft", text: "the cat sat" }]);
  assert.deepEqual(parser.feed('{"event":"done"}'), []);
  assert.deepEqual(parser.end(), [{ event: "done" }]);
  assert.deepEqual(parser.end(), []);
});

test("the window follows a turn through its events", () => {
  assert.deepEqual(STREAM_EVENTS, ["look", "draft", "caught", "backtrack", "found", "stuck", "turn"]);
  let live = applyEvent(null, { event: "look", index: 1, speaker: "B", from: "the cat" });
  assert.equal(live.note, "continuing “the cat”");
  live = applyEvent(live, { event: "draft", index: 1, speaker: "B", text: "the cat the cat sat", cost: 1 });
  assert.equal(live.kept, "the cat the cat sat");
  live = applyEvent(live, { event: "caught", index: 1, speaker: "B", kind: "stutter", noticed: "the cat", cut: "the cat " });
  assert.equal(live.kept, "the cat ");
  assert.equal(live.retracted, "the cat sat");
  assert.match(live.note, /saying “the cat” twice/);
  const before = { ...live };
  live = applyEvent(live, { event: "backtrack", index: 1, speaker: "B", step: 1, cut: "the cat ", wider: 10 });
  assert.equal(before.step, 0, "the old state is left alone");
  assert.equal(live.step, 1);
  live = applyEvent(live, { event: "found", index: 1, speaker: "B", text: "the cat ran to the door", cost: 2, explored: 3 });
  assert.equal(live.found, "ran to the door");
  assert.equal(applyEvent(live, { event: "turn", index: 1, speaker: "B", turn: {} }), null);
  // a repeat the voice cannot back out of retracts the whole draft
  live = applyEvent(null, { event: "draft", index: 2, speaker: "A", text: "ho hum", cost: 0 });
  live = applyEvent(live, { event: "caught", index: 2, speaker: "A", kind: "repeat", noticed: "ho hum", cut: "" });
  assert.equal(live.kept, "");
  assert.equal(live.retracted, "ho hum");
  assert.match(live.note, /the words it picked up, not its own/);
  live = applyEvent(live, { event: "stuck", index: 2, speaker: "A", explored: 4 });
  assert.equal(live.note, "nothing new in 4 path(s)");
  // a new turn starts a new window
  const other = applyEvent(live, { event: "look", index: 3, speaker: "B", from: "" });
  assert.equal(other.index, 3);
  assert.equal(other.draft, "");
  assert.equal(other.note, "starting a fresh text");
});
