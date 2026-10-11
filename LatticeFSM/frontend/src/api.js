/**
 * Thin fetch wrapper for the LatticeFSM JSON API (`latticefsm serve`; the routes are in ../rust/README.md).
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
const q = (params) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join("&");

/** The route prefix of a language model: anything but "ngram" is the chain. */
const lmPath = (m) => (m === "ngram" ? "/api/ngram" : "/api/chain");

export const api = {
  health: () => get("/api/health"),
  stats: () => get("/api/stats"),
  matrix: (symbol) => get(`/api/matrix?${q({ symbol })}`),
  cube: () => get("/api/cube"),
  edge: (source, symbol, target) => get(`/api/edge?${q({ source, symbol, target })}`),
  table: () => get("/api/table"),
  languages: () => get("/api/languages"),
  run: (body) => post("/api/run", body),
  credit: (amount) => post("/api/credit", { amount }),
  teach: (body) => post("/api/teach", body),
  train: (body) => post("/api/train", body),
  tick: (ticks) => post("/api/tick", { ticks }),
  stimulate: (body) => post("/api/stimulate", body),
  newMachine: (body) => post("/api/new", body),
  save: (path) => post("/api/save", { path }),
  load: (path) => post("/api/load", { path }),
  compress: (body) => post("/api/compress", body),
  core: () => get("/api/core"),
  expand: (shells) => post("/api/expand", shells === undefined || shells === null ? {} : { shells }),
  coreRun: (text, fromMiddle) => post("/api/core/run", { text, from_middle: fromMiddle }),
  coreSave: (path) => post("/api/core/save", { path }),
  coreLoad: (path) => post("/api/core/load", { path }),
  compression: (body) => post("/api/compression", body),
  focus: (body) => post("/api/focus", body),
  skip: (body) => post("/api/skip", body),
  rearrange: (body) => post("/api/rearrange", body),
  ngram: () => get("/api/ngram"),
  ngramNew: (body) => post("/api/ngram/new", body),
  ngramTrain: (body) => post("/api/ngram/train", body),
  ngramText: (text, opts = {}) => post("/api/ngram/text", { text, ...opts }),
  ngramPredict: (text, k) => post("/api/ngram/predict", { text, k }),
  ngramGenerate: (body) => post("/api/ngram/generate", body),
  ngramCredit: (text, amount) => post("/api/ngram/credit", { text, amount }),
  ngramOllama: (model, url) => get(`/api/ngram/ollama?${q({ model, url })}`),
  ngramAutotrain: (body) => post("/api/ngram/autotrain", body),
  ngramSave: (path) => post("/api/ngram/save", { path }),
  ngramLoad: (path) => post("/api/ngram/load", { path }),
  network: () => get("/api/network"),
  networkNew: (body) => post("/api/network/new", body),
  talk: (text) => post("/api/network/talk", { text }),
  networkTeach: (body) => post("/api/network/teach", body),
  networkFit: (body) => post("/api/network/fit", body),
  networkSave: (path) => post("/api/network/save", { path }),
  networkLoad: (path) => post("/api/network/load", { path }),

  // the language model the header's selector picks: `m` is "chain" or "ngram" (the same shapes for both)
  lm: (m) => get(lmPath(m)),
  lmNew: (m, body) => post(`${lmPath(m)}/new`, body),
  lmTrain: (m, body) => post(`${lmPath(m)}/train`, body),
  lmText: (m, text, opts = {}) => post(`${lmPath(m)}/text`, { text, ...opts }),
  lmPredict: (m, text, k) => post(`${lmPath(m)}/predict`, { text, k }),
  lmGenerate: (m, body) => post(`${lmPath(m)}/generate`, body),
  lmCredit: (m, text, amount) => post(`${lmPath(m)}/credit`, { text, amount }),
  lmFeedback: (m, body) => post(`${lmPath(m)}/feedback`, body),
  lmScore: (m, text) => post(`${lmPath(m)}/score`, { text }),
  lmSave: (m, path) => post(`${lmPath(m)}/save`, { path }),
  lmLoad: (m, path) => post(`${lmPath(m)}/load`, { path }),
  /** Teach that `reply` follows `text`, round after round, until the greedy continuation of `text` is exactly `reply`. */
  lmTeach: (m, body) => post(`${lmPath(m)}/teach`, body),
  // the chain alone
  chainConfidence: (confidence) => post("/api/chain/confidence", { confidence }),
  chainSweep: (body) => post("/api/chain/sweep", body),
  chainGraph: (limit) => get(`/api/chain/graph?${q({ limit })}`),

  status: () => get("/api/status"),
  converse: (body) => post("/api/converse", body),
  chatLlm: (body) => post("/api/chat/llm", body),
  ollamaCorpus: (body) => post("/api/ollama/corpus", body),

  uploads: () => get("/api/uploads"),
  /** One piece of a file's bytes (base64; at most 8 MiB raw, see upload.js): the first starts it, `append` adds on. */
  upload: (name, contentBase64, append = false) => post("/api/uploads", { name, content_base64: contentBase64, append }),
  uploadText: (name, content) => post("/api/uploads", { name, content }),
  uploadDelete: (name) => post("/api/uploads/delete", { name }),

  checkpoints: () => get("/api/checkpoints"),
  checkpointSave: (model, tag) => post("/api/checkpoints/save", { model, ...(tag ? { tag } : {}) }),
  checkpointRestore: (name) => post("/api/checkpoints/restore", { name }),
  checkpointDelete: (name) => post("/api/checkpoints/delete", { name }),
  checkpointUpload: (name, contentBase64, append = false) =>
    post("/api/checkpoints/upload", { name, content_base64: contentBase64, append }),
  /** A plain link (the server answers with the file as an attachment), not a fetch. */
  checkpointDownloadUrl: (name) => `${API_BASE}/api/checkpoints/download?${q({ name })}`,
};
