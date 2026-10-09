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

export const api = {
  health: () => get("/api/health"),
  stats: () => get("/api/stats"),
  matrix: (symbol) => get(`/api/matrix?${q({ symbol })}`),
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
};
