/**
 * Thin fetch wrapper for the RadixCyclicNN JSON API (DESIGN.md section 12).
 *
 * Every call resolves with the parsed JSON body or rejects with an ApiError.
 * The server reports failures as {"error": "..."} with a 4xx/5xx status;
 * network failures and non-JSON bodies are normalised into ApiError too, so
 * panels only ever deal with one error shape (`err.message`).
 */

/** Base URL prefix; empty by default so relative /api URLs work when the Python server serves dist. */
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
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    init.body = body; // multipart/form-data: the browser sets the boundary
  } else if (body !== undefined) {
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

/**
 * Normalise a job payload. Job-starting endpoints answer {"job": {...}}, while
 * GET /api/job and the stop endpoints answer with the job status itself; both
 * may carry null when no job exists yet.
 */
export function unwrapJob(data) {
  if (!data || typeof data !== "object") return null;
  if ("job" in data) return data.job && typeof data.job === "object" ? data.job : null;
  return "state" in data ? data : null;
}

export const api = {
  health: () => get("/api/health"),
  status: () => get("/api/status"),
  /** Model kinds: the active one, every kind and their files; switching keeps the previous model in memory. */
  model: () => get("/api/model"),
  selectModel: (kind) => post("/api/model/select", { kind }),
  /** Count model: change the dual frequency weight function (scales and the sliding window). */
  modelWeights: (body) => post("/api/model/weights", body),
  train: (body) => post("/api/train", body),
  /** Learning-rate schedules: what an expression may use (presets, variables, functions) and a per-epoch preview. */
  schedule: () => get("/api/schedule"),
  schedulePreview: (body) => post("/api/schedule/preview", body),
  job: () => get("/api/job"),
  stopJob: () => post("/api/job/stop"),
  predict: (body) => post("/api/predict", body),
  generate: (body) => post("/api/generate", body),
  score: (body) => post("/api/score", body),
  twoNrl: (body) => post("/api/2nrl", body),
  /** Rated texts -> 2NRL (both kinds), reward (thumbs up only) or punish (thumbs down only). */
  feedback: (body) => post("/api/feedback", body),
  invert: () => post("/api/invert"),
  compress: () => post("/api/compress"),
  evolveStart: (body) => post("/api/evolve/start", body),
  evolveStop: () => post("/api/evolve/stop"),
  evolveHistory: () => get("/api/evolve/history"),
  save: (body) => post("/api/save", body),
  load: (body) => post("/api/load", body),
  reset: (body) => post("/api/reset", body),
  checkpoints: () => get("/api/checkpoints"),
  checkpointSave: (body) => post("/api/checkpoints/save", body),
  checkpointRestore: (body) => post("/api/checkpoints/restore", body),
  graph: (limit) => get(`/api/graph?limit=${encodeURIComponent(limit)}`),
  history: () => get("/api/history"),
  uploads: () => get("/api/uploads"),
  /** Upload one text file (read in the browser); the server keeps it under its upload directory. */
  upload: (name, content) => post("/api/uploads", { name, content }),
  /** Upload a file as bytes (multipart); a ZIP archive is unpacked into text files on the server. */
  uploadFile: (file) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return post("/api/uploads", form);
  },
  deleteUpload: (name) => post("/api/uploads/delete", { name }),
  /** Ollama (see OllamaPanel). `url` optionally overrides the server's configured Ollama URL. */
  ollamaModels: (url) => get(`/api/ollama/models${url ? `?url=${encodeURIComponent(url)}` : ""}`),
  ollamaCorpus: (body) => post("/api/ollama/corpus", body),
  ollamaReview: (body) => post("/api/ollama/review", body),
  /** Images as text (see ImagesPanel): the Stable Diffusion VAE run backwards, quantised and base64-encoded. */
  images: () => get("/api/images"),
  imageEncode: (file, { size, encoder, train, saveAs } = {}) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const params = new URLSearchParams();
    if (size) params.set("size", String(size));
    if (encoder) params.set("encoder", String(encoder));
    if (train) params.set("train", "true");
    if (saveAs) params.set("save_as", String(saveAs));
    const query = params.toString();
    return post(`/api/images/encode${query ? `?${query}` : ""}`, form);
  },
  imageDecode: (text, encoder) => post("/api/images/decode", encoder ? { text, encoder } : { text }),
  /** Code generation (see CodeGenPanel): sandbox runs, an Ollama teacher / judge and 2NRL rewards. */
  codegenStart: (body) => post("/api/codegen/start", body),
  codegenHistory: () => get("/api/codegen/history"),
  codegenSolve: (body) => post("/api/codegen/solve", body),
  codegenRun: (body) => post("/api/codegen/run", body),
};
