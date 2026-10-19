/** Tests for the chunked upload helpers (`src/upload.js`) - plain `node --test`. Run with `npm test`. */

import assert from "node:assert/strict";
import { test } from "node:test";

import { CHUNK_BYTES, bytesToBase64, chunkRanges, progressText, uploadInChunks, uploadName } from "../src/upload.js";

const reference = (bytes) => Buffer.from(bytes).toString("base64");

test("bytesToBase64 matches Node's encoder, padding and all", () => {
  for (const n of [0, 1, 2, 3, 4, 5, 255, 256, 24575, 24576, 24577, 100000]) {
    const bytes = new Uint8Array(n).map((_, i) => (i * 37 + 11) & 255);
    assert.equal(bytesToBase64(bytes), reference(bytes), `length ${n}`);
  }
  assert.equal(bytesToBase64(new TextEncoder().encode("hello")), "aGVsbG8=");
  assert.equal(bytesToBase64([0, 255]), "AP8=");
  assert.equal(bytesToBase64(new Uint8Array([1, 2, 3]).buffer), "AQID");
});

test("bytesToBase64 encodes by hand without btoa", () => {
  const saved = globalThis.btoa;
  try {
    globalThis.btoa = undefined;
    for (const n of [0, 1, 2, 3, 10, 1000]) {
      const bytes = new Uint8Array(n).map((_, i) => (i * 101) & 255);
      assert.equal(bytesToBase64(bytes), reference(bytes));
    }
  } finally {
    globalThis.btoa = saved;
  }
});

test("chunkRanges cuts at the chunk size, the first piece starting the file", () => {
  assert.equal(CHUNK_BYTES, 8 * 1024 * 1024);
  assert.deepEqual(chunkRanges(0), [{ start: 0, end: 0, append: false }]);
  assert.deepEqual(chunkRanges(5, 10), [{ start: 0, end: 5, append: false }]);
  assert.deepEqual(chunkRanges(10, 10), [{ start: 0, end: 10, append: false }]);
  assert.deepEqual(chunkRanges(25, 10), [
    { start: 0, end: 10, append: false },
    { start: 10, end: 20, append: true },
    { start: 20, end: 25, append: true },
  ]);
  const big = chunkRanges(CHUNK_BYTES * 2 + 1);
  assert.equal(big.length, 3);
  assert.ok(big.every((r) => r.end - r.start <= CHUNK_BYTES));
});

test("uploadName keeps a folder path and drops a leading slash", () => {
  assert.equal(uploadName({ name: "a.txt" }), "a.txt");
  assert.equal(uploadName({ name: "a.txt", webkitRelativePath: "books/a.txt" }), "books/a.txt");
  assert.equal(uploadName({ name: "a.txt", relativePath: "/drop/a.txt" }), "drop/a.txt");
  assert.equal(uploadName(null), "");
});

test("uploadInChunks sends every piece in order and the pieces rebuild the file", async () => {
  const bytes = new Uint8Array(25).map((_, i) => i * 7);
  const sent = [];
  const progress = [];
  const last = await uploadInChunks(
    new Blob([bytes]),
    async (b64, append) => {
      sent.push({ b64, append });
      return { name: "f", n: sent.length };
    },
    { chunk: 10, onProgress: (s, t) => progress.push([s, t]) },
  );
  assert.deepEqual(
    sent.map((s) => s.append),
    [false, true, true],
  );
  const rebuilt = Buffer.concat(sent.map((s) => Buffer.from(s.b64, "base64")));
  assert.deepEqual([...rebuilt], [...bytes]);
  assert.deepEqual(progress, [
    [10, 25],
    [20, 25],
    [25, 25],
  ]);
  assert.deepEqual(last, { name: "f", n: 3 });
});

test("uploadInChunks still creates an empty file", async () => {
  const sent = [];
  await uploadInChunks(new Blob([]), async (b64, append) => sent.push([b64, append]));
  assert.deepEqual(sent, [["", false]]);
});

test("progressText", () => {
  assert.equal(progressText(0, 100), "0%");
  assert.equal(progressText(50, 100), "50%");
  assert.equal(progressText(100, 100), "100%");
  assert.equal(progressText(0, 0), "100%");
});
