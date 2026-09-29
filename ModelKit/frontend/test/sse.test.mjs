import assert from "node:assert/strict";
import { test } from "node:test";
import { EventStreamParser, parseFrame } from "../src/sse.js";

test("a frame is its event name and its data, decoded when it is JSON", () => {
  assert.deepEqual(parseFrame('event: message_start\ndata: {"type":"message_start","message":{}}'), {
    event: "message_start",
    data: { type: "message_start", message: {} },
  });
  assert.deepEqual(parseFrame("data: [DONE]"), { event: null, data: "[DONE]" });
  assert.deepEqual(parseFrame('data: {"type":"ping"}'), { event: "ping", data: { type: "ping" } });
  assert.equal(parseFrame(": a comment"), null);
  assert.equal(parseFrame(""), null);
  assert.deepEqual(parseFrame("data: one\ndata: two"), { event: null, data: "one\ntwo" });
});

test("frames come out whole however the bytes were cut", () => {
  const parser = new EventStreamParser();
  const frames = [];
  const text =
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the"}}\n\n' +
    'event: content_block_delta\r\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" cat"}}\r\n\r\n' +
    "data: [DONE]\n\n";
  for (let i = 0; i < text.length; i += 7) frames.push(...parser.push(text.slice(i, i + 7)));
  frames.push(...parser.end());
  assert.deepEqual(
    frames.map((f) => f.event),
    ["content_block_delta", "content_block_delta", null],
  );
  assert.equal(frames[0].data.delta.text + frames[1].data.delta.text, "the cat");
  assert.equal(frames[2].data, "[DONE]");
});

test("the tail of a stream that ends without its blank line is still a frame", () => {
  const parser = new EventStreamParser();
  assert.deepEqual(parser.push('data: {"a":1}\n\ndata: {"b":2}'), [{ event: null, data: { a: 1 } }]);
  assert.deepEqual(parser.end(), [{ event: null, data: { b: 2 } }]);
  assert.deepEqual(parser.end(), []);
});
