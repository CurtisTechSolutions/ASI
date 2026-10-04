/** The same chat descriptor as modelkit.agent_config, read entirely in the browser. */
const DEFAULTS = { timeout: 60, max_tokens: 128, temperature: 0, structured: true };
const FIELDS = new Set(["id", "type", "base_url", "model", ...Object.keys(DEFAULTS)]);

export function validateAgentConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("agent.json must contain a JSON object.");
  }
  for (const key of Object.keys(value)) {
    if (!FIELDS.has(key)) throw new Error(`Unknown agent setting: ${key}.`);
  }
  const config = { ...DEFAULTS, ...value };
  for (const key of ["id", "type", "base_url", "model"]) {
    if (typeof config[key] !== "string" || !config[key].trim()) {
      throw new Error(`Agent ${key} must be a non-empty string.`);
    }
    config[key] = config[key].trim();
  }
  if (config.type !== "chat") throw new Error('Agent type must be "chat".');
  if (!Number.isFinite(config.timeout) || config.timeout <= 0) throw new Error("Agent timeout must be a positive number of seconds.");
  if (!Number.isSafeInteger(config.max_tokens) || config.max_tokens <= 0) throw new Error("Agent max_tokens must be a positive integer.");
  if (!Number.isFinite(config.temperature) || config.temperature < 0) throw new Error("Agent temperature must be a non-negative number.");
  if (typeof config.structured !== "boolean") throw new Error("Agent structured must be true or false.");
  let url;
  try {
    const authority = config.base_url.match(/^https?:\/\/([^/?#]*)/i);
    if (!authority || authority[1].includes("@") || /[\s\\]/.test(config.base_url)) throw new Error();
    url = new URL(config.base_url);
    if (!url.hostname || url.search || url.hash) throw new Error();
  } catch {
    throw new Error("Agent base_url must be an http(s) URL without credentials, query or fragment.");
  }
  url.pathname = url.pathname.replace(/\/+$/, "") || "/v1";
  config.base_url = url.href.replace(/\/+$/, "");
  return config;
}

export function parseAgentConfig(text) {
  let value;
  try {
    value = JSON.parse(text.replace(/^\uFEFF/, ""));
  } catch {
    throw new Error("Could not read agent.json. Check that the file contains valid JSON.");
  }
  return validateAgentConfig(value);
}

/** Apply defaults without mutating the descriptor or the caller's request. */
export function agentRequestBody(config, body) {
  const agent = validateAgentConfig(config);
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("The agent request must be a JSON object.");
  if (body.stream != null && body.stream !== false) throw new Error("Agent connections return a complete reply; streaming is not supported.");
  const payload = {
    model: agent.model, max_tokens: agent.max_tokens, temperature: agent.temperature,
    ...(agent.structured ? { response_format: { type: "json_object" } } : {}),
    ...body, stream: false,
  };
  if (Object.hasOwn(body, "max_completion_tokens")) delete payload.max_tokens;
  return payload;
}

function choicesOf(doc) {
  if (!Array.isArray(doc?.choices) || !doc.choices.length || doc.choices.some((c) =>
    !c?.message || typeof c.message !== "object" || Array.isArray(c.message))) {
    throw new Error("The agent returned no chat completion choices.");
  }
  return doc.choices;
}

/** Direct browser request: no credentials, a deadline, and the Talk panel's Stop signal. */
export async function requestAgentCompletion(config, body, { signal, fetchImpl = globalThis.fetch } = {}) {
  const agent = validateAgentConfig(config);
  const payload = agentRequestBody(agent, body);
  const controller = new AbortController();
  const stop = () => controller.abort();
  if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
  signal?.addEventListener("abort", stop, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, Math.min(agent.timeout * 1000, 2147483647));
  try {
    let response;
    try {
      response = await fetchImpl(`${agent.base_url}/chat/completions`, {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload), credentials: "omit", signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted) throw err;
      throw new Error(`Could not connect to ${agent.base_url}. Check that the server is running and allows requests from this browser.`);
    }
    const text = await response.text();
    let doc;
    try {
      doc = JSON.parse(text);
    } catch {
      throw new Error(response.ok ? "The agent returned invalid JSON." : `Agent request failed (HTTP ${response.status}).`);
    }
    if (!response.ok || doc?.error) {
      const detail = typeof doc?.error === "string" ? doc.error : doc?.error?.message;
      throw new Error(`Agent request failed (HTTP ${response.status})${detail ? `: ${detail}` : "."}`);
    }
    choicesOf(doc);
    return doc;
  } catch (err) {
    if (timedOut) throw new Error(`The agent did not reply within ${agent.timeout} seconds.`);
    if (controller.signal.aborted) throw new DOMException("Stopped", "AbortError");
    throw err;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", stop);
  }
}

/** Translate a Chat Completions answer into the existing conversation view. */
export function agentReply(doc) {
  const choice = choicesOf(doc)[0];
  const message = choice.message;
  const content = message.content;
  const text = typeof content === "string" ? content : Array.isArray(content)
    ? content.filter((p) => typeof p?.text === "string").map((p) => p.text).join("") : "";
  const record = doc.radixnet?.choices?.[0] || {};
  const stops = { stop: "end_turn", length: "max_tokens", tool_calls: "tool_use", content_filter: "refusal" };
  return {
    text: text || (typeof message.refusal === "string" ? message.refusal : ""),
    thinking: typeof message.reasoning_content === "string" ? message.reasoning_content : "",
    stop: message.refusal ? "refusal" : stops[choice.finish_reason] || null,
    stopSequence: record.stop_sequence || null, turn: record.turn || null, guard: record.guard || null,
    outputTokens: doc.usage?.completion_tokens ?? null,
    units: doc.radixnet?.units === "chars" ? "characters" : doc.radixnet?.units === "words" ? "words" : "tokens",
    toolCalls: Array.isArray(message.tool_calls) ? message.tool_calls : [],
    streaming: false,
  };
}
