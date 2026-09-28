/**
 * Tests for what a model of sounds said, in words (`src/spelled.js`) - plain
 * `node --test`: the English a server spelled an answer into is split where
 * the model's own part begins, so the panels can highlight it the way they
 * highlight a continuation of letters. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { spelledParts } from "../src/spelled.js";

test("a prediction splits where its continuation begins", () => {
  const prediction = {
    full_text: "DH.AH0 # K.AE1.T # S.AE1.T",
    continuation: "# S.AE1.T",
    spelled: "the cat sat",
    spelled_continuation: " sat",
  };
  assert.deepEqual(spelledParts(prediction), { whole: "the cat sat", head: "the cat", tail: " sat" });
  // the two parts are the whole again
  const { head, tail, whole } = spelledParts(prediction);
  assert.equal(head + tail, whole);
});

test("a turn splits where its reply begins", () => {
  const turn = { text: "DH.AH0 # K.AE1.T", spelled: "the cat", spelled_reply: " cat" };
  assert.deepEqual(spelledParts(turn, "spelled_reply"), { whole: "the cat", head: "the", tail: " cat" });
  // a fresh turn is all reply
  assert.deepEqual(spelledParts({ spelled: "hello", spelled_reply: "hello" }, "spelled_reply"), {
    whole: "hello",
    head: "",
    tail: "hello",
  });
});

test("a record without the part is all tail", () => {
  assert.deepEqual(spelledParts({ text: "DH.AH0", spelled: "the" }), { whole: "the", head: "", tail: "the" });
  // nor does a tail that does not end the whole cut it
  assert.deepEqual(spelledParts({ spelled: "the cat", spelled_continuation: "dog" }), {
    whole: "the cat",
    head: "",
    tail: "the cat",
  });
  // an answer that added nothing: the whole is the head
  assert.deepEqual(spelledParts({ spelled: "the cat", spelled_continuation: "" }), {
    whole: "the cat",
    head: "the cat",
    tail: "",
  });
});

test("a record of letters, words or acoustic units carries no spelling", () => {
  assert.equal(spelledParts({ text: "the cat", full_text: "the cat" }), null);
  assert.equal(spelledParts(null), null);
  assert.equal(spelledParts(undefined), null);
  assert.equal(spelledParts("the cat"), null);
  assert.equal(spelledParts({ spelled: 3 }), null);
});
