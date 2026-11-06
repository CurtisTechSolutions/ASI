/**
 * Thin fetch wrapper for the GameLattice JSON API (`gamelattice serve`; the routes are in ../rust/src/server.rs and
 * ../README.md).
 *
 * Every call resolves with the parsed JSON body or rejects with an ApiError.
 * The server reports failures as {"error": "..."} with a 4xx/5xx status;
 * network failures and non-JSON bodies are normalised into ApiError too, so
 * panels only ever deal with one error shape (`err.message`).
 */

/** Base URL prefix; empty by default so relative /api URLs work when the server serves dist. */
export const API_BASE = String(import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function request(method, path, body) {
  const init = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(API_BASE + path, init);
  } catch (err) {
    throw new ApiError(`Network error: ${err && err.message ? err.message : "request failed"}`);
  }
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      const detail = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
      throw new ApiError(response.ok ? `Invalid JSON in response (${detail})` : detail, response.status);
    }
  }
  const serverError = data && typeof data === "object" && typeof data.error === "string" ? data.error : null;
  if (!response.ok) {
    const detail = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
    throw new ApiError(serverError || detail, response.status, data);
  }
  if (serverError) throw new ApiError(serverError, response.status, data);
  return data;
}

const get = (path) => request("GET", path);
const post = (path, body = {}) => request("POST", path, body);

/** How often a running job's progress is asked for. */
export const JOB_POLL_MS = 500;

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * POST `body` to `path` as a background job (`"job": true`): the server starts it on its own thread and answers at
 * once, and every other request is answered meanwhile. Polls `GET /api/job`, handing each report to `onProgress`
 * ({id, kind, model, state, elapsed, progress: {phase, label, sequences, tokens, round, tokens_per_second,
 * stopping}, curve}: a training job's held-out curve so far), and resolves with the job's result - the same body the plain request would have answered with - or
 * rejects with its error (`err.data.state` "stopped" when it was stopped).
 */
export async function runJob(path, body = {}, onProgress = null) {
  const started = await post(path, { ...body, job: true });
  const job = started && started.job;
  if (!job || typeof job.id !== "number") throw new ApiError("the server did not start a job");
  if (onProgress) onProgress(job);
  let misses = 0;
  for (;;) {
    await pause(JOB_POLL_MS);
    let now;
    try {
      now = await get("/api/job");
      misses = 0;
    } catch (err) {
      misses += 1;
      if (misses >= 10) throw err; // the server is gone; a moment's hiccup is not
      continue;
    }
    if (!now || now.id !== job.id) throw new ApiError("the job is gone - was the server restarted?");
    if (onProgress) onProgress(now);
    if (now.state === "done") return now.result;
    if (now.state !== "running") throw new ApiError(now.error || `the job ${now.state}`, 0, now);
  }
}
const q = (params) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join("&");

export const api = {
  health: () => get("/api/health"),
  /** Everything in brief: the body's settings, episodes, diversity, wealth, agreement, the lattice's stats, Nim, the job. */
  status: () => get("/api/status"),
  languages: () => get("/api/languages"),
  /** A fresh body: `{language, states, window, n, seed, encoder_sets_state, forward_lessons, punish, whole_run, label_set}`. */
  newBody: (body) => post("/api/new", body),
  /** `episodes` random strings, every step credited by the oracle, as a job; resolves with the held-out curve. */
  train: (body, onProgress) => runJob("/api/train", body, onProgress),
  /** Quiet: the verdicts and the naming trace; nothing moves. */
  classify: (text) => post("/api/classify", { text }),
  /** One learning episode on `text`: the verdicts and the trace with every seat's Shapley value. */
  run: (text) => post("/api/run", { text }),
  matrix: (symbol) => get(`/api/matrix?${q({ symbol })}`),
  edge: (source, symbol, target) => get(`/api/edge?${q({ source, symbol, target })}`),
  table: () => get("/api/table"),
  encoder: (top) => get(`/api/encoder?${q({ top })}`),
  nim: () => get("/api/nim"),
  nimNew: (body) => post("/api/nim/new", body),
  nimBabble: (body, onProgress) => runJob("/api/nim/babble", body, onProgress),
  nimPlay: (body, onProgress) => runJob("/api/nim/play", body, onProgress),
  /** `results/e1.json` and `e2.json` as the experiments wrote them (null where missing). */
  results: () => get("/api/results"),

  /** The background job running now, or the last one (null: none yet); stop asks it to stop at the next episode. */
  job: () => get("/api/job"),
  jobStop: () => post("/api/job/stop"),
};
