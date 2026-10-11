/**
 * Tests for what a model of base64 symbols said, as bytes (`src/decoded.js`) -
 * plain `node --test`: the text a server decoded an answer into is split where
 * the model's own part begins, and the symbols decode to the same bytes the
 * servers' decoder gives. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { base64Bytes, bytesName, decodedParts } from "../src/decoded.js";

test("a prediction splits where its continuation begins", () => {
  const prediction = {
    full_text: "dGhlIGNhdCBzYXQ",
    continuation: "CBzYXQ",
    decoded: "the cat sat",
    decoded_continuation: " sat",
    decoded_bytes: 11,
  };
  assert.deepEqual(decodedParts(prediction), { whole: "the cat sat", head: "the cat", tail: " sat", bytes: 11 });
  const { head, tail, whole } = decodedParts(prediction);
  assert.equal(head + tail, whole);
});

test("a record without the part is all tail, and one without decoding is null", () => {
  assert.deepEqual(decodedParts({ decoded: "the" }), { whole: "the", head: "", tail: "the", bytes: null });
  assert.deepEqual(decodedParts({ decoded: "the log", decoded_continuation: "dog" }), {
    whole: "the log",
    head: "",
    tail: "the log",
    bytes: null,
  });
  assert.equal(decodedParts({ full_text: "the cat", continuation: " cat" }), null);
  assert.equal(decodedParts(null), null);
  assert.equal(decodedParts("dGhl"), null);
});

test("the symbols decode as the servers decode them", () => {
  const text = (bytes) => Array.from(bytes, (b) => String.fromCharCode(b)).join("");
  assert.equal(text(base64Bytes("dGhlIGNhdA")), "the cat");
  assert.equal(text(base64Bytes("YQ==")), "a"); // padding is skipped
  assert.equal(text(base64Bytes("YW\nJj \t")), "abc"); // so is whitespace
  assert.equal(text(base64Bytes("YWJjZ")), "abc"); // a dangling symbol makes no byte
  assert.equal(text(base64Bytes("YR")), "a"); // the bits left over are ignored
  assert.equal(base64Bytes("").length, 0);
  assert.equal(base64Bytes("Y").length, 0);
  assert.equal(base64Bytes(null).length, 0);
  // every byte value, round trip through the platform's own encoder
  const all = Uint8Array.from({ length: 256 }, (_, i) => i);
  const symbols = Buffer.from(all).toString("base64").replace(/=+$/, "");
  assert.deepEqual(Array.from(base64Bytes(symbols)), Array.from(all));
});

test("a file name for the bytes", () => {
  assert.equal(bytesName("prediction"), "prediction.bin");
  assert.equal(bytesName("image.png"), "image.png");
  assert.equal(bytesName("sample 2"), "sample-2.bin");
  assert.equal(bytesName(""), "decoded.bin");
});
