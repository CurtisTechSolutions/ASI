import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { agentReply, agentRequestBody, parseAgentConfig, requestAgentCompletion, validateAgentConfig } from "../src/agent-config.js";

const sampleText = await readFile(new URL("../../../RadixCyclicNN/agent.json", import.meta.url), "utf8");
const sample = JSON.parse(sampleText);
const body = { messages: [{ role: "user", content: "the cat sat" }] };
const reply = {
  model: "radixnet-count", choices: [{ message: { role: "assistant", content: "on the mat", reasoning_content: "a trace" }, finish_reason: "stop" }],
  usage: { completion_tokens: 10 }, radixnet: { units: "chars", choices: [{ turn: { context: "the cat" }, guard: { vetoed: 0 } }] },
};

test("the supplied descriptor and optional defaults match the Python client", () => {
  assert.deepEqual(parseAgentConfig(sampleText), sample);
  assert.deepEqual(parseAgentConfig(`\uFEFF${sampleText}`), sample);
  const { id, type, base_url, model } = sample;
  assert.deepEqual(validateAgentConfig({ id, type, base_url, model }), sample);
  for (const [url, expected] of [
    ["http://localhost:8000/", "http://localhost:8000/v1"],
    ["http://[::1]:8000/v1/", "http://[::1]:8000/v1"],
    ["https://example.test/prefix/v1/", "https://example.test/prefix/v1"],
  ]) assert.equal(validateAgentConfig({ ...sample, base_url: url }).base_url, expected);
});

test("malformed files and invalid fields give actionable errors", () => {
  assert.throws(() => parseAgentConfig("{oops"), /valid JSON/);
  for (const text of ["null", "[]", '"hello"']) assert.throws(() => parseAgentConfig(text), /JSON object/);
  const invalid = {
    id: [null, "", 1], type: [false, "completion"], model: [null, "", []],
    base_url: ["localhost:8000", "file:///tmp", "http:example.test", "http://a:99999", "http://u:p@localhost/v1",
      "http://@localhost/v1", "http://a/v1?key=x", "http://a/#v1", "http://a b", "http://a\\v1"],
    timeout: [0, -1, true, "60", Infinity, NaN], max_tokens: [0, -1, true, 1.5, "128", Infinity],
    temperature: [-1, true, "0", Infinity, NaN], structured: [null, 0, 1, "true"],
  };
  for (const [key, values] of Object.entries(invalid)) {
    for (const value of values) assert.throws(() => validateAgentConfig({ ...sample, [key]: value }), new RegExp(key));
  }
  for (const key of ["id", "type", "base_url", "model"]) {
    const config = { ...sample };
    delete config[key];
    assert.throws(() => validateAgentConfig(config), new RegExp(key));
  }
  assert.throws(() => validateAgentConfig({ ...sample, temperatur: 1 }), /Unknown agent setting: temperatur/);
});

test("defaults, zero temperature, structured output and explicit request overrides reach the request", () => {
  assert.deepEqual(agentRequestBody(sample, body), {
    ...body, model: "radixcyclicnn", max_tokens: 128, temperature: 0, response_format: { type: "json_object" }, stream: false,
  });
  const plain = agentRequestBody({ ...sample, structured: false }, { ...body, max_tokens: 4, temperature: 0.75 });
  assert.equal(plain.response_format, undefined);
  assert.equal(plain.max_tokens, 4);
  assert.equal(plain.temperature, 0.75);
  const override = agentRequestBody(sample, { ...body, model: "count", max_completion_tokens: 8, response_format: { type: "text" } });
  assert.equal(override.model, "count");
  assert.equal(override.max_tokens, undefined);
  assert.equal(override.max_completion_tokens, 8);
  assert.deepEqual(override.response_format, { type: "text" });
  assert.equal(body.model, undefined, "the caller's messages are not changed");
  assert.throws(() => agentRequestBody(sample, { ...body, stream: true }), /streaming/);
});

test("the transport contacts the configured endpoint without credentials and keeps the response", async () => {
  const result = await requestAgentCompletion(sample, body, { fetchImpl: async (url, options) => {
    assert.equal(url, "http://127.0.0.1:8000/v1/chat/completions");
    assert.equal(options.method, "POST");
    assert.equal(options.credentials, "omit");
    assert.deepEqual(options.headers, { "Content-Type": "application/json", Accept: "application/json" });
    assert.deepEqual(JSON.parse(options.body), agentRequestBody(sample, body));
    return Response.json(reply);
  } });
  assert.deepEqual(result, reply);
});

test("bad configuration never contacts the server", async () => {
  let called = false;
  await assert.rejects(requestAgentCompletion({ ...sample, timeout: 0 }, body, { fetchImpl: async () => { called = true; } }), /timeout/);
  assert.equal(called, false);
});

test("network, HTTP and malformed responses become readable failures", async () => {
  await assert.rejects(requestAgentCompletion(sample, body, { fetchImpl: async () => { throw new TypeError("Failed to fetch"); } }), /allows requests from this browser/);
  for (const [response, expected] of [
    [Response.json({ error: { message: "unknown model" } }, { status: 404 }), /HTTP 404.*unknown model/],
    [new Response("unavailable", { status: 503 }), /HTTP 503/],
    [new Response("not json"), /invalid JSON/],
    [Response.json({ choices: [] }), /no chat completion choices/],
    [Response.json({ choices: [{ message: null }] }), /no chat completion choices/],
  ]) await assert.rejects(requestAgentCompletion(sample, body, { fetchImpl: async () => response }), expected);
});

const waitForAbort = async (_url, { signal }) => new Promise((_resolve, reject) => {
  const fail = () => reject(new DOMException("Aborted", "AbortError"));
  if (signal.aborted) fail();
  else signal.addEventListener("abort", fail, { once: true });
});

test("the configured timeout ends a stalled request", async () => {
  await assert.rejects(requestAgentCompletion({ ...sample, timeout: 0.01 }, body, { fetchImpl: waitForAbort }), /within 0.01 seconds/);
});

test("Stop cancels in-flight requests and prevents an already-stopped request", async () => {
  const controller = new AbortController();
  const pending = requestAgentCompletion(sample, body, { signal: controller.signal, fetchImpl: waitForAbort });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  let called = false;
  await assert.rejects(requestAgentCompletion(sample, body, { signal: controller.signal, fetchImpl: async () => { called = true; } }), { name: "AbortError" });
  assert.equal(called, false);
});

test("the deadline also covers reading the response body", async () => {
  await assert.rejects(requestAgentCompletion({ ...sample, timeout: 0.01 }, body, { fetchImpl: async (_url, options) => ({
    ok: true, status: 200, text: () => waitForAbort(_url, options),
  }) }), /within 0.01 seconds/);
});

test("chat replies display text, thinking, usage, the model record and stop reasons", () => {
  assert.deepEqual(agentReply(reply), {
    text: "on the mat", thinking: "a trace", stop: "end_turn", stopSequence: null,
    turn: { context: "the cat" }, guard: { vetoed: 0 }, outputTokens: 10, units: "characters", toolCalls: [], streaming: false,
  });
  const parts = agentReply({ choices: [{ message: { content: [{ type: "text", text: "hello" }, { type: "text", text: " world" }] }, finish_reason: "length" }] });
  assert.equal(parts.text, "hello world");
  assert.equal(parts.stop, "max_tokens");
  assert.equal(parts.units, "tokens");
  const refusal = agentReply({ choices: [{ message: { content: null, refusal: "no answer" }, finish_reason: "content_filter" }] });
  assert.equal(refusal.text, "no answer");
  assert.equal(refusal.stop, "refusal");
  const calls = [{ id: "call-1", type: "function", function: { name: "calculator", arguments: '{"expression":"2+2"}' } }];
  const tool = agentReply({ choices: [{ message: { content: null, tool_calls: calls }, finish_reason: "tool_calls" }] });
  assert.equal(tool.stop, "tool_use");
  assert.deepEqual(tool.toolCalls, calls);
});
